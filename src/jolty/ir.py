# Copyright 2026 Maxence Faldor
"""The canonical circuit format, as the engine reads it.

A [`Circuit`][jolty.ir.Circuit] is a set of named components and the name of the top
one. Build scripts never write this format by hand: [`jolty.builder`][jolty.builder]
produces it, and the engine validates and elaborates it. Parameter values are numbers in
SI base units, and a device may give some of them a [`Tolerance`][jolty.ir.Tolerance],
each naming its parameter by [path][jolty.ir.parameter], so that it may reach into an
object, as ``waveform.dc`` names a source's DC level. A device may also state
[`Rating`][jolty.ir.Rating]s, the limits its maker states on its stresses. A
[`RandomVariable`][jolty.ir.RandomVariable], a number a sample draws at random and the
parameters it moves, is no part of a circuit, but what its tolerances and a process
kit's statistical functions make.
"""

from collections.abc import Mapping
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
"""What separates the levels of a path, such as ``X1/R1``, which no name holds."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Role(StrEnum):
    """What a port carries: a signal, in both directions or one, a supply or ground."""

    SIGNAL = "signal"
    INPUT = "input"
    OUTPUT = "output"
    SUPPLY = "supply"
    GROUND = "ground"


class Port(_Model):
    """A net a component exposes, under the name of the net inside it."""

    name: str
    role: Role


class DeviceOf(_Model):
    """What a device instance is: a device of the named model."""

    model: str


class ComponentOf(_Model):
    """What a component instance is: an instance of the named component."""

    component: str


class SourceLocation(_Model):
    """The line of a build script that placed an instance."""

    file: str
    line: int
    function: str | None = None

    @override
    def __str__(self) -> str:
        """The line, such as ``filter.py:12 in low_pass()``."""
        within = (
            "at module level"
            if self.function in {None, "<module>"}
            else f"in {self.function}()"
        )
        return f"{self.file}:{self.line} {within}"


type Distribution = Literal["uniform", "normal", "log_uniform", "log_normal"]
"""How values spread within a tolerance.

A value lies off its nominal one by a factor: ``1 + d``, with ``d`` evenly within a
``uniform`` tolerance or normally within a ``normal`` one, the tolerance three standard
deviations; or ``(1 + relative) ** u``, with ``u`` evenly within 1 of 0 for
``log_uniform`` or normally for ``log_normal``, 1 three standard deviations, as an
exponential's argument spreads, such as a saturation current's when the junction voltage
it sets does.
"""


class _Spread(_Model):
    """A tolerance and how values spread within it."""

    relative: float = Field(ge=0, allow_inf_nan=False)
    distribution: Distribution = "uniform"

    @model_validator(mode="after")
    def _taken(self) -> Self:
        """Holds a linear spread below 1, so that no value reaches 0."""
        if self.relative >= 1 and not self.distribution.startswith("log"):
            message = (
                f"a {self.distribution} tolerance needs to be below 1, not "
                f"{self.relative}; only a logarithmic one may exceed it"
            )
            raise ValueError(message)
        return self


class Lot(_Spread):
    """A spread that the parts of one lot share.

    As parts from one reel or one die drift together, each parameter of theirs takes
    one factor from the lot per sample, which scales each part's own. A lot is known by
    its name wherever in the hierarchy a part names it, and spreads each parameter one
    way.
    """

    name: str


class Tolerance(_Spread):
    """How far a parameter's value may lie from its nominal one, relative to it.

    A part's own factor is drawn apart from every other part's; one in a `lot` is scaled
    by the lot's too.
    """

    lot: Lot | None = Field(default=None, exclude_if=lambda lot: lot is None)


class RandomVariable(_Model):
    """A number a Monte Carlo sample draws at random, and how it moves the devices.

    A part's tolerance draws one for its own factor and one for its lot's, and a process
    kit's statistical function, such as ``agauss``, one per placement and call. It is
    not part of a circuit: a mismatch analysis
    ([`Mismatch`][jolty.plan.Mismatch]) makes the tolerances' from the circuit and takes
    a kit's from the plan, as its reader finds them
    ([`Library.random_variables`][jolty.spice.Library.random_variables]).
    """

    instance: str | None = None
    """The path of the instance that draws it: the part whose tolerance it is, or the
    placement whose definitions call a kit's statistical function; None for a lot's."""
    lot: str | None = None
    """The name of the lot whose shared spread it is, if it is one."""
    name: str
    """What it is within its instance or lot: the parameter a tolerance spreads, such as
    ``resistance``, or which call of a statistical function it is, such as ``vth0#0``,
    its definition's name and its place among the calls the definition makes."""
    mean: float = Field(allow_inf_nan=False)
    """Its mean, which the circuit's parameters are at."""
    standard_deviation: float = Field(ge=0, allow_inf_nan=False)
    derivatives: Mapping[str, Mapping[str, float]]
    """The derivative of each number it moves with respect to it, by the device's path,
    then by the number's path among its parameters, in SI base units per unit of it."""


def part_number(number: str) -> str:
    """`number`, if it is a maker's part number: not empty, nor spaced at either end.

    Raises:
        ValueError: When it is empty or begins or ends with whitespace.
    """
    if not number or number.strip() != number:
        message = f"a part number is neither empty nor spaced at either end: {number!r}"
        raise ValueError(message)
    return number


type PartNumber = Annotated[str, AfterValidator(part_number)]
"""A maker's part number, such as ``2N3904``."""


type Stress = Literal["voltage", "current", "power", "temperature"]
"""What a rating limits: the voltage of one terminal over another, the current into a
terminal, the power a part dissipates, the sum over its electrical terminals of each
one's voltage times the current into it, or the part's temperature, the simulation's
raised by its thermal net's where it heats, the net its terminal ``thermal`` joins."""

_TERMINALS: dict[str, int] = {"voltage": 2, "current": 1, "power": 0, "temperature": 0}


class Rating(_Model):
    """A limit a part's maker states on one of its stresses, which a simulation checks.

    The terminals are two for a voltage, the first's over the second's, one for a
    current, into it, and none for the power or the temperature. A rating bounds its
    stress from above, from below or both, in SI base units; a power's only from above,
    and above 0, and a temperature's in kelvins, above 0.
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
        """Holds the terminals to the stress, and the bounds in order."""
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


class Instance(_Model):
    """One placed device or component, and the net each of its terminals joins."""

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
            "The maker's number of the part the instance stands for, if any: a "
            "component instance so numbered is that one part, whatever it places."
        ),
    )
    source_location: SourceLocation | None = None


class Component(_Model):
    """A component's ports and the instances it places."""

    ports: tuple[Port, ...] = ()
    instances: tuple[Instance, ...] = ()

    def instance(self, id: str) -> Instance | None:
        """The instance `id`, or None when the component places none by that id."""
        return next(
            (instance for instance in self.instances if instance.name == id), None
        )


class Circuit(_Model):
    """A circuit: every component, and the one at the top of the hierarchy."""

    format: Literal["jolty-circuit"] = "jolty-circuit"
    version: Literal[4] = 4
    top: str
    components: dict[str, Component]

    def find(self, path: str) -> Instance | None:
        """The instance at `path`, such as ``X1/R1``, or None when there is none."""
        component = self.components.get(self.top)
        found = None
        for id in path.split(SEPARATOR):
            found = component.instance(id) if component is not None else None
            if found is None:
                return None
            component = (
                self.components.get(found.of.component)
                if isinstance(found.of, ComponentOf)
                else None
            )
        return found


def parameter(parameters: Mapping[str, JsonValue], path: str) -> JsonValue | None:
    """The value at `path` among `parameters`, or None when none is there.

    A path is a parameter's name, or its keys joined by dots within an object and a
    list's entries keyed by their index from 0, as sensitivities name parameters.

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
    """Every number among `parameters`, in the order they are written, by its path.

    A path is as [`parameter`][jolty.ir.parameter] reads it; a flag is no number.

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
