# Copyright 2026 Maxence Faldor
"""The results of analyses, read by net and by instance.

Every result holds the circuit's unknowns: node voltages, named by net path such as
``out`` or ``amplifier/mid``, and branch currents, named by instance path and label such
as ``V1:current``. ``voltage`` and ``current`` read them by name.
"""

import difflib
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

type Real = npt.NDArray[np.float64]
type Complex = npt.NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class Unknown:
    """A node voltage or branch current the engine solves for."""

    name: str
    quantity: Literal["voltage", "current"]


def column(unknowns: tuple[Unknown, ...], name: str) -> int:
    """The column of the unknown `name` in the values of a result over `unknowns`.

    Raises:
        KeyError: When no unknown has that name; the message suggests the closest names.
    """
    for index, unknown in enumerate(unknowns):
        if unknown.name == name:
            return index
    names = [unknown.name for unknown in unknowns]
    close = difflib.get_close_matches(name, names, n=3)
    hint = f"; close names: {', '.join(close)}" if close else ""
    message = f"the circuit has no unknown named {name!r}{hint}"
    raise KeyError(message)


def _current(instance: str) -> str:
    return f"{instance}:current"


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """A DC solution, and the strategy that found it."""

    unknowns: tuple[Unknown, ...]
    values: Real
    strategy: str

    def voltage(self, net: str, reference: str | None = None) -> float:
        """The voltage of `net`, relative to `reference` or to ground."""
        offset = 0.0 if reference is None else self.voltage(reference)
        return float(self.values[column(self.unknowns, net)]) - offset

    def current(self, instance: str) -> float:
        """The current through `instance`, a source or an inductor, in its direction."""
        return float(self.values[column(self.unknowns, _current(instance))])


@dataclass(frozen=True, slots=True)
class Sweep:
    """The operating points of a DC sweep, one row per level of its source."""

    unknowns: tuple[Unknown, ...]
    source: str
    """The path of the swept source, such as ``V1``."""
    quantity: Literal["voltage", "current"]
    """Whether the levels are voltages or currents."""
    levels: Real
    values: Real

    def voltage(self, net: str, reference: str | None = None) -> Real:
        """The voltage of `net` at each level, relative to `reference` or to ground."""
        values = self.values[:, column(self.unknowns, net)]
        return values if reference is None else values - self.voltage(reference)

    def current(self, instance: str) -> Real:
        """The current through `instance` at each level."""
        return self.values[:, column(self.unknowns, _current(instance))]


@dataclass(frozen=True, slots=True)
class Response:
    """A small-signal response: phasors per unit excitation, one row per frequency."""

    unknowns: tuple[Unknown, ...]
    source: str
    """The path of the excited source, such as ``V1``."""
    frequencies: Real
    values: Complex
    operating_point: OperatingPoint

    def voltage(self, net: str, reference: str | None = None) -> Complex:
        """The phasor of `net` at each frequency, relative to `reference` or ground."""
        values = self.values[:, column(self.unknowns, net)]
        return values if reference is None else values - self.voltage(reference)

    def current(self, instance: str) -> Complex:
        """The phasor of the current through `instance` at each frequency."""
        return self.values[:, column(self.unknowns, _current(instance))]


@dataclass(frozen=True, slots=True)
class Statistics:
    """How a transient analysis went."""

    initial: str
    accepted: int
    rejected: int
    iterations: int


@dataclass(frozen=True, slots=True)
class Waveforms:
    """The waveforms of a transient analysis, one row per accepted time point."""

    unknowns: tuple[Unknown, ...]
    times: Real
    values: Real
    statistics: Statistics

    def voltage(self, net: str, reference: str | None = None) -> Real:
        """The voltage of `net` over time, relative to `reference` or to ground."""
        values = self.values[:, column(self.unknowns, net)]
        return values if reference is None else values - self.voltage(reference)

    def current(self, instance: str) -> Real:
        """The current through `instance` over time."""
        return self.values[:, column(self.unknowns, _current(instance))]


type Result = OperatingPoint | Sweep | Response | Waveforms
"""The result of any analysis."""
