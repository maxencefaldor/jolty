# Copyright 2026 Maxence Faldor
"""Canonical circuit intermediate representation read by the simulation engine.

A [`Circuit`][jolty.ir.Circuit] holds named component definitions and the top-level
component name. Build scripts construct circuits through
[`jolty.builder`][jolty.builder], and the engine validates and elaborates the IR.

Parameter values use SI base units, and devices can attach
[`Tolerance`][jolty.ir.Tolerance] entries addressed by [path][jolty.ir.parameter] (such
as ``waveform.dc``) and [`Rating`][jolty.ir.Rating] stress limits. A
[`RandomVariable`][jolty.ir.RandomVariable] represents a Monte Carlo draw and its
parameter derivatives produced by tolerances or foundry statistical functions.
"""

from collections import deque
from collections.abc import Callable, Iterator, Mapping
from enum import StrEnum
from typing import Annotated, Final, Literal, Self, override

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)

SEPARATOR: Final = "/"
"""Path separator between hierarchy levels (such as ``X1/R1``), forbidden in names."""


class FrozenModel(BaseModel):
    """Immutable Pydantic model forbidding unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Role(StrEnum):
    """Electrical or functional role of a component port."""

    SIGNAL = "signal"
    INPUT = "input"
    OUTPUT = "output"
    SUPPLY = "supply"
    GROUND = "ground"


class Port(FrozenModel):
    """External net exposed by a component under an internal net name."""

    name: str
    role: Role


class DeviceOf(FrozenModel):
    """Reference identifying an instance as a primitive device of `model`."""

    model: str


class ComponentOf(FrozenModel):
    """Reference identifying an instance as an instantiation of `component`."""

    component: str


class SourceLocation(FrozenModel):
    """Build-script source location that placed an instance."""

    file: str
    line: int
    function: str | None = None

    @override
    def __str__(self) -> str:
        """Formats the source location, such as ``filter.py:12 in low_pass()``."""
        within = (
            "at module level"
            if self.function in {None, "<module>"}
            else f"in {self.function}()"
        )
        return f"{self.file}:{self.line} {within}"


type Distribution = Literal["uniform", "normal", "log_uniform", "log_normal"]
"""Statistical distribution describing how parameter values spread within a tolerance.

Linear tolerances scale the nominal value by ``1 + d``, where ``d`` is uniform over the
tolerance range (``uniform``) or normal with the tolerance at three standard deviations
(``normal``). Logarithmic tolerances scale by ``(1 + relative) ** u``, with ``u``
uniform on ``[-1, 1]`` (``log_uniform``) or normal with ``1`` at three standard
deviations (``log_normal``), modeling exponential parameters such as saturation
currents.
"""


class _Spread(FrozenModel):
    """Base tolerance specification combining relative spread and distribution."""

    relative: float = Field(ge=0, allow_inf_nan=False)
    distribution: Distribution = "uniform"

    @model_validator(mode="after")
    def _taken(self) -> Self:
        """Validates that linear tolerances stay below 1 so values remain positive."""
        if self.relative >= 1 and not self.distribution.startswith("log"):
            message = (
                f"a {self.distribution} tolerance is below 1, not "
                f"{self.relative}; only a logarithmic tolerance may exceed 1"
            )
            raise ValueError(message)
        return self


class Lot(_Spread):
    """Shared manufacturing spread applied across all parts from the same lot.

    Parts from one reel or die drift together by sharing one per-sample lot factor that
    scales each individual part factor. Matching lot names anywhere in the hierarchy
    share a single draw per parameter.
    """

    name: str


class Tolerance(_Spread):
    """Relative spread of a parameter value around its nominal value.

    Each part draws an independent per-instance factor; parts assigned to a `lot` also
    multiply by the shared lot factor.
    """

    lot: Lot | None = Field(default=None, exclude_if=lambda lot: lot is None)


class RandomVariable(FrozenModel):
    """Monte Carlo random draw and the parameter derivatives driven by that draw.

    Each tolerance draws a per-instance factor and an optional lot factor, while
    foundry statistical functions such as ``agauss`` draw once per placement and call.
    A mismatch analysis ([`Mismatch`][jolty.plan.Mismatch]) builds tolerance variables
    from the circuit and reads foundry variables from the plan
    ([`Library.random_variables`][jolty.spice.Library.random_variables]).
    """

    instance: str | None = None
    """Path of the owning instance or statistical placement, or ``None`` for a lot."""
    lot: str | None = None
    """Name of the shared lot when this variable represents a lot draw."""
    name: str
    """Identifier within the instance or lot, such as ``resistance`` or ``vth0#0``."""
    mean: float = Field(allow_inf_nan=False)
    """Mean value of the random variable at the nominal circuit operating point."""
    standard_deviation: float = Field(ge=0, allow_inf_nan=False)
    derivatives: Mapping[str, Mapping[str, float]]
    """Parameter derivatives keyed by device path and parameter path in SI units."""


def part_number(number: str) -> str:
    """Validates and returns manufacturer part number `number`.

    Raises:
        ValueError: When `number` is empty or has leading or trailing whitespace.
    """
    if not number or number.strip() != number:
        message = f"a part number is neither empty nor spaced at either end: {number!r}"
        raise ValueError(message)
    return number


type PartNumber = Annotated[str, AfterValidator(part_number)]
"""Validated manufacturer part number, such as ``2N3904``."""


type Stress = Literal["voltage", "current", "power", "temperature"]
"""Electrical or thermal stress quantity constrained by a rating.

Supported stresses are terminal-pair voltage, terminal input current, total electrical
power dissipation, and device temperature (ambient plus the ``thermal`` net rise for
heated devices)."""

_TERMINALS: dict[str, int] = {"voltage": 2, "current": 1, "power": 0, "temperature": 0}


class Rating(FrozenModel):
    """Manufacturer stress limit checked across simulation operating points and time.

    Voltage ratings require two terminals (first minus second), current ratings require
    one terminal (current into the terminal), and power or temperature ratings require
    zero terminals. Bounds use SI base units: power bounds only a positive maximum, and
    temperature bounds are positive kelvins.
    """

    stress: Stress
    terminals: tuple[str, ...] = Field(
        default=(), exclude_if=lambda terminals: not terminals
    )
    maximum: float | None = Field(
        default=None, allow_inf_nan=False, exclude_if=lambda maximum: maximum is None
    )
    minimum: float | None = Field(
        default=None, allow_inf_nan=False, exclude_if=lambda minimum: minimum is None
    )

    @model_validator(mode="after")
    def _bounded(self) -> Self:
        """Validates terminal count and bound ordering for the rating stress type."""
        needed = _TERMINALS[self.stress]
        if len(self.terminals) != needed or len(set(self.terminals)) != needed:
            message = (
                f"a {self.stress} rating names {needed} distinct terminals, "
                f"not {list(self.terminals)}"
            )
            raise ValueError(message)
        if self.maximum is None and self.minimum is None:
            message = f"a {self.stress} rating states a maximum, a minimum or both"
            raise ValueError(message)
        if self.stress == "power" and (
            self.minimum is not None or self.maximum is None or self.maximum <= 0
        ):
            message = "a power rating states only a maximum, above 0"
            raise ValueError(message)
        if self.stress == "temperature" and any(
            bound is not None and bound <= 0 for bound in (self.minimum, self.maximum)
        ):
            message = "a temperature rating's bounds are above 0 K"
            raise ValueError(message)
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            message = (
                f"a rating's minimum, {self.minimum!r}, exceeds its maximum, "
                f"{self.maximum!r}"
            )
            raise ValueError(message)
        return self


class Instance(FrozenModel):
    """Placed device or child component together with its net connections."""

    name: str
    of: DeviceOf | ComponentOf
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    terminals: dict[str, str]
    tolerances: dict[str, Tolerance] = Field(
        default_factory=dict, exclude_if=lambda tolerances: not tolerances
    )
    ratings: tuple[Rating, ...] = Field(
        default=(), exclude_if=lambda ratings: not ratings
    )
    part: PartNumber | None = Field(
        default=None,
        exclude_if=lambda part: part is None,
        description=(
            "Manufacturer part number for the instance; a numbered component prices "
            "as a single part regardless of internal placements."
        ),
    )
    source_location: SourceLocation | None = None


class Component(FrozenModel):
    """Component definition holding external ports and placed child instances."""

    ports: tuple[Port, ...] = ()
    instances: tuple[Instance, ...] = ()

    def instance(self, name: str) -> Instance | None:
        """Returns the placed instance named `name`, or ``None`` if absent."""
        return next(
            (instance for instance in self.instances if instance.name == name), None
        )


class Circuit(FrozenModel):
    """Complete circuit IR containing all component definitions and the top name."""

    format: Literal["jolty-circuit"] = "jolty-circuit"
    version: Literal[5] = 5
    top: str
    components: dict[str, Component]

    def find(self, path: str) -> Instance | None:
        """Returns the instance at hierarchical `path` (such as ``X1/R1``), or None."""
        component = self.components.get(self.top)
        found = None
        for name in path.split(SEPARATOR):
            found = component.instance(name) if component is not None else None
            if found is None:
                return None
            component = (
                self.components.get(found.of.component)
                if isinstance(found.of, ComponentOf)
                else None
            )
        return found

    def ground_port(self, component: str | None = None) -> str | None:
        """Returns the ground port name of `component`, defaulting to the top."""
        found = self.components.get(self.top if component is None else component)
        if found is None:
            return None
        return next(
            (port.name for port in found.ports if port.role is Role.GROUND),
            None,
        )

    def walk(
        self,
        component: str | None = None,
        *,
        prefix: str = "",
        breadth_first: bool = False,
        descend: Callable[[Instance], bool] | None = None,
    ) -> Iterator[tuple[str, Instance]]:
        """Yields `(path, instance)` across the component hierarchy iteratively.

        Starts at `component` (defaulting to `self.top`), prepending `prefix` to each
        path. Traverses depth-first in placement order by default, or breadth-first
        (outer instances first) when `breadth_first` is true. Descends into a
        `ComponentOf` instance when `descend` is None or `descend(instance)` is true.
        """
        root = self.top if component is None else component
        if breadth_first:
            yield from self._walk_breadth_first(root, prefix, descend)
        else:
            yield from self._walk_depth_first(root, prefix, descend)

    def _walk_breadth_first(
        self,
        root: str,
        prefix: str,
        descend: Callable[[Instance], bool] | None,
    ) -> Iterator[tuple[str, Instance]]:
        pending: deque[tuple[str, str]] = deque([(prefix, root)])
        while pending:
            current_prefix, name = pending.popleft()
            for instance in self.components[name].instances:
                path = f"{current_prefix}{instance.name}"
                yield path, instance
                if (inner := _descend_into(instance, descend)) is not None:
                    pending.append((f"{path}{SEPARATOR}", inner))

    def _walk_depth_first(
        self,
        root: str,
        prefix: str,
        descend: Callable[[Instance], bool] | None,
    ) -> Iterator[tuple[str, Instance]]:
        stack: deque[tuple[str, Instance]] = deque(
            (f"{prefix}{instance.name}", instance)
            for instance in reversed(self.components[root].instances)
        )
        while stack:
            path, instance = stack.pop()
            yield path, instance
            if (inner := _descend_into(instance, descend)) is not None:
                children = self.components[inner].instances
                stack.extend(
                    (f"{path}{SEPARATOR}{child.name}", child)
                    for child in reversed(children)
                )


def _descend_into(
    instance: Instance, descend: Callable[[Instance], bool] | None
) -> str | None:
    """Returns the child component name if `walk` descends into `instance`."""
    if isinstance(instance.of, ComponentOf) and (descend is None or descend(instance)):
        return instance.of.component
    return None


def parameter(parameters: Mapping[str, JsonValue], path: str) -> JsonValue | None:
    """Returns the value at dotted `path` within `parameters`, or None if missing.

    Path segments traverse mapping keys and zero-based list indices, matching the
    parameter-path format used by sensitivity analyses.

    Examples:
        >>> parameter({"waveform": {"dc": 5.0}}, "waveform.dc")
        5.0
        >>> parameter(
        ...     {"waveform": {"piecewise": {"points": [[0, 1.5]]}}},
        ...     "waveform.piecewise.points.0.1",
        ... )
        1.5
        >>> parameter({"waveform": {"dc": 5.0}}, "waveform.sine") is None
        True
    """
    found: JsonValue = dict(parameters)
    for key in path.split("."):
        match found:
            case dict():
                found = found.get(key)
            case list() if key.isascii() and key.isdigit() and int(key) < len(found):
                found = found[int(key)]
            case _:
                return None
    return found


def numbers(parameters: Mapping[str, JsonValue]) -> dict[str, float]:
    """Collects all numeric entries in `parameters` keyed by dotted path in order.

    Path formatting matches [`parameter`][jolty.ir.parameter]; booleans are excluded.

    Examples:
        >>> numbers({"resistance": 1e3, "waveform": {"dc": 5, "on": True}})
        {'resistance': 1000.0, 'waveform.dc': 5.0}
    """
    found: dict[str, float] = {}

    def walk(path: str, value: JsonValue) -> None:
        match value:
            case bool():
                pass
            case int() | float():
                found[path] = float(value)
            case dict():
                for key, inner in value.items():
                    walk(f"{path}.{key}", inner)
            case list():
                for index, inner in enumerate(value):
                    walk(f"{path}.{index}", inner)

    for key, value in parameters.items():
        walk(key, value)
    return found
