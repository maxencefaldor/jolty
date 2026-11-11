# Copyright 2026 Maxence Faldor
"""The canonical circuit format, as the engine reads it.

A :class:`Circuit` is a set of named components and the name of the top one. Build
scripts never write this format by hand: :mod:`jolty.builder` produces it, and the
engine validates and elaborates it. Parameter values are numbers in SI base units.
"""

from enum import StrEnum
from typing import Literal

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


class Instance(_Model):
    """One placed device or component, and the net each of its terminals joins."""

    id: str
    of: DeviceOf | ComponentOf
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    terminals: dict[str, str]
    source: Source | None = None


class Component(_Model):
    """A component's ports and the instances it places."""

    ports: tuple[Port, ...] = ()
    instances: tuple[Instance, ...] = ()


class Circuit(_Model):
    """A circuit: every component, and the one at the top of the hierarchy."""

    format: Literal["jolty-circuit"] = "jolty-circuit"
    version: Literal[1] = 1
    top: str
    components: dict[str, Component]
