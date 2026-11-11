# Copyright 2026 Maxence Faldor
"""The canonical circuit format, as the engine reads it.

A [`Circuit`][jolty.ir.Circuit] is a set of named components and the name of the top
one. Build scripts never write this format by hand: [`jolty.builder`][jolty.builder]
produces it, and the engine validates and elaborates it. Parameter values are numbers in
SI base units, and a device may give some of them a [`Tolerance`][jolty.ir.Tolerance].
"""

from enum import StrEnum
from typing import Literal, override

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Role(StrEnum):
    """What a port carries."""

    SIGNAL = "signal"
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


class Tolerance(_Model):
    """How far a parameter's value may lie from its nominal one, relative to it.

    Values spread evenly within a ``uniform`` tolerance, and normally within a
    ``normal`` one, with the tolerance three standard deviations.
    """

    relative: float = Field(ge=0, lt=1)
    distribution: Literal["uniform", "normal"] = "uniform"


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
