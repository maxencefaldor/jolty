# Copyright 2026 Maxence Faldor
"""The results of analyses, read by net and by instance.

Every result holds the circuit's unknowns: node voltages, named by net path such as
``out`` or ``amplifier/mid``, and branch currents, named by instance path and label such
as ``V1:current``. ``voltage`` and ``current`` read them by name. It holds the levels of
the circuit's logic nets too, which ``level`` reads by name: 0, 1, or NaN for an unknown
level.
"""

import difflib
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

type Real = npt.NDArray[np.float64]
type Complex = npt.NDArray[np.complex128]


def _none() -> Real:
    return np.empty(0)


def _no_rows() -> Real:
    return np.empty((0, 0))


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
    names = [unknown.name for unknown in unknowns]
    return _find(names, name, "the circuit has no unknown")


def net_column(nets: tuple[str, ...], name: str) -> int:
    """The column of the logic net `name` in the levels of a result over `nets`.

    Raises:
        KeyError: When no logic net has that name; the message suggests the closest.
    """
    return _find(list(nets), name, "the circuit has no logic net")


def _find(names: list[str], name: str, missing: str) -> int:
    """The index of `name` among `names`.

    Raises:
        KeyError: Saying `missing`, such as "the circuit has no unknown", named `name`,
            and the closest names.
    """
    if name in names:
        return names.index(name)
    close = difflib.get_close_matches(name, names, n=3)
    hint = f"; close names: {', '.join(close)}" if close else ""
    message = f"{missing} named {name!r}{hint}"
    raise KeyError(message)


@dataclass(frozen=True, slots=True)
class Trace:
    """A logic net's level over time: its level at time zero, then at each change.

    Examples:
        >>> trace = Trace(np.array([0.0, 2e-9]), np.array([0.0, 1.0]))
        >>> trace.at(np.array([1e-9, 2e-9, 3e-9])).tolist()
        [0.0, 1.0, 1.0]
    """

    times: Real
    levels: Real

    def at(self, times: Real) -> Real:
        """The level at each of `times`: that of the last change at or before it."""
        indices = np.searchsorted(self.times, times, side="right") - 1
        return self.levels[np.maximum(indices, 0)]


def _current(instance: str) -> str:
    return f"{instance}:current"


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """A DC solution, and the strategy that found it."""

    unknowns: tuple[Unknown, ...]
    values: Real
    strategy: str
    nets: tuple[str, ...] = ()
    """The logic nets, in the order of `logic`."""
    logic: Real = field(default_factory=_none)
    """Each logic net's level."""

    def level(self, net: str) -> float:
        """The level of the logic net `net`: 0, 1, or NaN when unknown."""
        return float(self.logic[net_column(self.nets, net)])

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
    nets: tuple[str, ...] = ()
    """The logic nets, in the order of `logic`'s columns."""
    logic: Real = field(default_factory=_no_rows)
    """Each logic net's level at each level of the source, one row per level."""

    def level(self, net: str) -> Real:
        """The level of the logic net `net` at each level: 0, 1, or NaN when unknown."""
        return self.logic[:, net_column(self.nets, net)]

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
    nets: tuple[str, ...] = ()
    """The logic nets, in the order of `logic`."""
    logic: tuple[Trace, ...] = ()
    """Each logic net's level over time."""

    def level(self, net: str) -> Trace:
        """The level of the logic net `net` over time."""
        return self.logic[net_column(self.nets, net)]

    def voltage(self, net: str, reference: str | None = None) -> Real:
        """The voltage of `net` over time, relative to `reference` or to ground."""
        values = self.values[:, column(self.unknowns, net)]
        return values if reference is None else values - self.voltage(reference)

    def current(self, instance: str) -> Real:
        """The current through `instance` over time."""
        return self.values[:, column(self.unknowns, _current(instance))]


@dataclass(frozen=True, slots=True)
class Parameter:
    """One number of the parameters of a device instance."""

    instance: str
    """The device's path, such as ``X1/R1``."""
    name: str
    """The parameter's name, its keys joined by dots within an object, such as
    ``resistance`` or ``waveform.dc``."""
    value: float
    """Its value, in SI base units."""


@dataclass(frozen=True, slots=True)
class Sensitivities:
    """DC sensitivities: each output's derivative with respect to each parameter.

    `derivatives` has one row per output and one column per parameter, in SI base
    units, at `operating_point`.

    Examples:
        >>> point = OperatingPoint((Unknown("out", "voltage"),), np.array([2.0]), "")
        >>> sensitivities = Sensitivities(
        ...     point.unknowns,
        ...     ("out",),
        ...     (Parameter("R1", "resistance", 1e3), Parameter("V1", "waveform.dc", 5)),
        ...     np.array([[-1e-3, 0.4]]),
        ...     point,
        ... )
        >>> sensitivities.derivative("out", "V1", "waveform.dc")
        0.4
        >>> sensitivities.per_percent("out").tolist()
        [-0.01, 0.02]
    """

    unknowns: tuple[Unknown, ...]
    outputs: tuple[str, ...]
    """The unknowns differentiated, by name, in the order of `derivatives`' rows."""
    parameters: tuple[Parameter, ...]
    """Every parameter, in the order of `derivatives`' columns."""
    derivatives: Real
    operating_point: OperatingPoint

    def derivative(self, output: str, instance: str, parameter: str) -> float:
        """The derivative of `output` with respect to `parameter` of `instance`.

        Raises:
            KeyError: When the analysis has no such output or parameter.
        """
        labels = [f"{found.instance} {found.name}" for found in self.parameters]
        missing = "the analysis has no parameter"
        index = _find(labels, f"{instance} {parameter}", missing)
        return float(self.derivatives[self._row(output), index])

    def per_percent(self, output: str) -> Real:
        """How much `output` moves, to first order, as each parameter grows by 1 %."""
        values = np.array([parameter.value for parameter in self.parameters])
        return self.derivatives[self._row(output)] * values / 100

    def _row(self, output: str) -> int:
        return _find(list(self.outputs), output, "the analysis has no output")


type Result = OperatingPoint | Sweep | Response | Waveforms | Sensitivities
"""The result of any analysis."""
