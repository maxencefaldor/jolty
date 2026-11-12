# Copyright 2026 Maxence Faldor
"""The canonical circuit format, as the engine reads it.

A [`Circuit`][jolty.ir.Circuit] is a set of named components and the name of the top
one. Build scripts never write this format by hand: [`jolty.builder`][jolty.builder]
produces it, and the engine validates and elaborates it. Parameter values are numbers in
SI base units, and a device may give some of them a [`Tolerance`][jolty.ir.Tolerance],
each naming its parameter by [path][jolty.ir.parameter], so that it may reach into an
object, as ``waveform.dc`` names a source's DC level.
"""

from collections.abc import Mapping
from enum import StrEnum
from typing import Literal, Self, override

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


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

    device: str


class ComponentOf(_Model):
    """What a component instance is: an instance of the named component."""

    component: str


class Source(_Model):
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


class Instance(_Model):
    """One placed device or component, and the net each of its terminals joins."""

    id: str
    of: DeviceOf | ComponentOf
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    terminals: dict[str, str]
    tolerances: dict[str, Tolerance] = Field(
        default_factory=dict, exclude_if=lambda tolerances: not tolerances
    )
    source: Source | None = None


class Component(_Model):
    """A component's ports and the instances it places."""

    ports: tuple[Port, ...] = ()
    instances: tuple[Instance, ...] = ()

    def instance(self, id: str) -> Instance | None:
        """The instance `id`, or None when the component places none by that id."""
        return next(
            (instance for instance in self.instances if instance.id == id), None
        )


class Circuit(_Model):
    """A circuit: every component, and the one at the top of the hierarchy."""

    format: Literal["jolty-circuit"] = "jolty-circuit"
    version: Literal[1] = 1
    top: str
    components: dict[str, Component]

    def find(self, path: str) -> Instance | None:
        """The instance at `path`, such as ``X1/R1``, or None when there is none."""
        component = self.components.get(self.top)
        found = None
        for id in path.split("/"):
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
            case list() if key.isdecimal() and int(key) < len(found):
                found = found[int(key)]
            case _:
                return None
    return found
