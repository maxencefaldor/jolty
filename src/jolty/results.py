# Copyright 2026 Maxence Faldor
"""The results of analyses, read by net and by instance.

Every result holds the circuit's unknowns: node voltages, named by net path such as
``out`` or ``amplifier/mid``, and branch currents and other internal unknowns, named by
instance path and label such as ``V1:current``. ``voltage`` and ``current`` read them by
name. It holds the current into every terminal of every device, and the charge the
terminal holds, which ``current`` and ``charge`` read by instance and terminal, such as
``X1/R1`` and ``a``; a component instance's or a logic device's terminal, such as
``X1`` and ``vdd``, sums those of the device terminals behind it.

It holds each device's own small-signal model too: the derivative of each of its
equations with respect to each of its unknowns, which ``conductance`` and
``capacitance`` read, and the variables it computes on its way, such as a transistor's
threshold voltage, which ``variable`` reads. And it holds the levels of the circuit's
logic nets, which ``level`` reads by name: 0, 1, or NaN for an unknown level.
"""

import difflib
import math
from dataclasses import dataclass, field
from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt

type Real = npt.NDArray[np.float64]
type Complex = npt.NDArray[np.complex128]


def _none() -> Real:
    return np.empty(0)


def _no_rows() -> Real:
    return np.empty((0, 0))


def _no_phasors() -> Complex:
    return np.empty((0, 0), dtype=np.complex128)


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


type Composites = tuple[tuple[str, tuple[int, ...]], ...]
"""Each terminal of a component instance or a logic device, such as ``X1.vdd``, with the
indices of the device terminals whose currents and charges sum to its own."""


class Observed(TypedDict):
    """What every device did, as fields of an operating point, sweep or transient."""

    terminals: tuple[str, ...]
    currents: Real
    charges: Real
    composites: Composites
    derivatives: tuple[tuple[str, str], ...]
    conductances: Real
    capacitances: Real
    variables: tuple[tuple[str, str], ...]
    readings: Real


def terminal_columns(
    terminals: tuple[str, ...], composites: Composites, instance: str, terminal: str
) -> list[int]:
    """The columns of a result's currents that sum to the current into a terminal.

    The terminal `terminal` of `instance` has its own column for a device, and those of
    the device terminals behind it for a component instance or a logic device.

    Raises:
        KeyError: When no instance has that terminal; the message suggests the closest.

    Examples:
        >>> terminal_columns(("R1.a", "R1.b"), (("X1.p", (0, 1)),), "X1", "p")
        [0, 1]
    """
    name = f"{instance}.{terminal}"
    if name in terminals:
        return [terminals.index(name)]
    for composite, members in composites:
        if composite == name:
            return list(members)
    names = [*terminals, *(composite for composite, _ in composites)]
    return [_find(names, name, "the circuit has no terminal")]


def derivative_column(
    derivatives: tuple[tuple[str, str], ...], equation: str, unknown: str
) -> int:
    """The column of a result's conductances and capacitances for one derivative.

    It holds the derivative of `equation` with respect to `unknown`, two of one device's
    terminals or internal unknowns. Each is named as a terminal, such as ``M1.drain``,
    for the current into it and the voltage at it, or as an internal unknown, such as
    ``M1:drain``, for its equation and
    its value.

    Raises:
        KeyError: When the two are not of one device; the message suggests the closest.
    """
    labels = [f"{row} by {column}" for row, column in derivatives]
    missing = "no device has the derivative"
    return _find(labels, f"{equation} by {unknown}", missing)


def variable_column(
    variables: tuple[tuple[str, str], ...], instance: str, name: str
) -> int:
    """The column of a result's variables that holds `name` of `instance`.

    Raises:
        KeyError: When the device has no such variable; the message suggests the
            closest.
    """
    labels = [f"{device} {variable}" for device, variable in variables]
    return _find(labels, f"{instance} {name}", "the circuit has no device variable")


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


def _summed[T: (np.float64, np.complex128)](
    array: npt.NDArray[T], columns: list[int]
) -> npt.NDArray[T]:
    """The sum of `columns` of `array`, at each row of a two-dimensional one."""
    return array[..., columns].sum(axis=-1)


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
    terminals: tuple[str, ...] = ()
    """Every terminal of every device, such as ``X1/R1.a``, in `currents`' order."""
    currents: Real = field(default_factory=_none)
    """The current into each terminal, in amperes."""
    charges: Real = field(default_factory=_none)
    """The charge each terminal holds, in coulombs."""
    composites: Composites = ()
    """Each terminal of a component instance or a logic device, such as ``X1.vdd``, with
    the indices of the device terminals whose currents and charges sum to its own."""
    derivatives: tuple[tuple[str, str], ...] = ()
    """For each device, every pair of its terminals and internal unknowns, its equation
    then its unknown, in `conductances`' order."""
    conductances: Real = field(default_factory=_none)
    """The derivative of each equation's residual with respect to each unknown."""
    capacitances: Real = field(default_factory=_none)
    """The derivative of each equation's charge with respect to each unknown."""
    variables: tuple[tuple[str, str], ...] = ()
    """Every variable of every device, its path and its name, in `readings`' order."""
    readings: Real = field(default_factory=_none)
    """Each variable's value, in SI base units."""

    def level(self, net: str) -> float:
        """The level of the logic net `net`: 0, 1, or NaN when unknown."""
        return float(self.logic[net_column(self.nets, net)])

    def voltage(self, net: str, reference: str | None = None) -> float:
        """The voltage of `net`, relative to `reference` or to ground."""
        offset = 0.0 if reference is None else self.voltage(reference)
        return float(self.values[column(self.unknowns, net)]) - offset

    def current(self, instance: str, terminal: str | None = None) -> float:
        """The current into `terminal` of `instance`.

        The instance is a device, a component instance or a logic device. Without a
        terminal, the current through `instance`, a source or an inductor, in its
        direction.
        """
        if terminal is not None:
            columns = terminal_columns(
                self.terminals, self.composites, instance, terminal
            )
            return float(_summed(self.currents, columns))
        return float(self.values[column(self.unknowns, _current(instance))])

    def charge(self, instance: str, terminal: str) -> float:
        """The charge `terminal` of `instance` holds."""
        columns = terminal_columns(self.terminals, self.composites, instance, terminal)
        return float(_summed(self.charges, columns))

    def conductance(self, equation: str, unknown: str) -> float:
        """The derivative of a device's `equation`'s residual with respect to `unknown`.

        For two terminals, such as ``M1.drain`` and ``M1.gate``, it is the change of the
        current into the first with the voltage at the second.
        """
        return float(
            self.conductances[derivative_column(self.derivatives, equation, unknown)]
        )

    def capacitance(self, equation: str, unknown: str) -> float:
        """The derivative of a device's `equation`'s charge with respect to `unknown`.

        For two terminals, it is the change of the first's charge with the voltage at
        the second.
        """
        return float(
            self.capacitances[derivative_column(self.derivatives, equation, unknown)]
        )

    def variable(self, instance: str, name: str) -> float:
        """The variable `name` of the device `instance`, such as a threshold voltage."""
        return float(self.readings[variable_column(self.variables, instance, name)])


@dataclass(frozen=True, slots=True)
class OperatingPoints:
    """Every DC solution a search found, the operating point first.

    Examples:
        >>> unknowns = (Unknown("q", "voltage"),)
        >>> found = OperatingPoints(
        ...     unknowns,
        ...     (
        ...         OperatingPoint(unknowns, np.array([1.65]), "direct"),
        ...         OperatingPoint(unknowns, np.array([3.3]), "direct"),
        ...     ),
        ... )
        >>> found.voltages("q").tolist()
        [1.65, 3.3]
    """

    unknowns: tuple[Unknown, ...]
    points: tuple[OperatingPoint, ...]

    def voltages(self, net: str) -> Real:
        """The voltage of `net` at each solution, in the order found."""
        index = column(self.unknowns, net)
        return np.array([float(point.values[index]) for point in self.points])


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
    terminals: tuple[str, ...] = ()
    """Every terminal of every device, in the order of `currents`' columns."""
    currents: Real = field(default_factory=_no_rows)
    """The current into each terminal at each level, one row per level."""
    charges: Real = field(default_factory=_no_rows)
    """The charge each terminal holds at each level, one row per level."""
    composites: Composites = ()
    """Each terminal of a component instance or a logic device, with the indices of the
    device terminals whose currents and charges sum to its own."""
    derivatives: tuple[tuple[str, str], ...] = ()
    """For each device, every pair of its terminals and internal unknowns, in the order
    of `conductances`' columns."""
    conductances: Real = field(default_factory=_no_rows)
    """Each derivative of a residual at each level, one row per level."""
    capacitances: Real = field(default_factory=_no_rows)
    """Each derivative of a charge at each level, one row per level."""
    variables: tuple[tuple[str, str], ...] = ()
    """Every variable of every device, in the order of `readings`' columns."""
    readings: Real = field(default_factory=_no_rows)
    """Each variable's value at each level, one row per level."""

    def level(self, net: str) -> Real:
        """The level of the logic net `net` at each level: 0, 1, or NaN when unknown."""
        return self.logic[:, net_column(self.nets, net)]

    def voltage(self, net: str, reference: str | None = None) -> Real:
        """The voltage of `net` at each level, relative to `reference` or to ground."""
        values = self.values[:, column(self.unknowns, net)]
        return values if reference is None else values - self.voltage(reference)

    def current(self, instance: str, terminal: str | None = None) -> Real:
        """The current into `terminal` of `instance` at each level.

        Without a terminal, the current through `instance`, a source or an inductor.
        """
        if terminal is not None:
            columns = terminal_columns(
                self.terminals, self.composites, instance, terminal
            )
            return _summed(self.currents, columns)
        return self.values[:, column(self.unknowns, _current(instance))]

    def charge(self, instance: str, terminal: str) -> Real:
        """The charge `terminal` of `instance` holds at each level."""
        columns = terminal_columns(self.terminals, self.composites, instance, terminal)
        return _summed(self.charges, columns)

    def conductance(self, equation: str, unknown: str) -> Real:
        """The derivative of `equation`'s residual by `unknown` at each level."""
        index = derivative_column(self.derivatives, equation, unknown)
        return self.conductances[:, index]

    def capacitance(self, equation: str, unknown: str) -> Real:
        """The derivative of `equation`'s charge by `unknown` at each level."""
        index = derivative_column(self.derivatives, equation, unknown)
        return self.capacitances[:, index]

    def variable(self, instance: str, name: str) -> Real:
        """The variable `name` of the device `instance` at each level."""
        return self.readings[:, variable_column(self.variables, instance, name)]


@dataclass(frozen=True, slots=True)
class Response:
    """A small-signal response: phasors per unit excitation, one row per frequency."""

    unknowns: tuple[Unknown, ...]
    source: str
    """The path of the excited source, such as ``V1``."""
    frequencies: Real
    values: Complex
    operating_point: OperatingPoint
    terminals: tuple[str, ...] = ()
    """Every terminal of every device, in the order of `currents`' columns."""
    currents: Complex = field(default_factory=_no_phasors)
    """The phasor of the current into each terminal, one row per frequency."""
    composites: Composites = ()
    """Each terminal of a component instance or a logic device, with the indices of the
    device terminals whose currents sum to its own."""

    def voltage(self, net: str, reference: str | None = None) -> Complex:
        """The phasor of `net` at each frequency, relative to `reference` or ground."""
        values = self.values[:, column(self.unknowns, net)]
        return values if reference is None else values - self.voltage(reference)

    def current(self, instance: str, terminal: str | None = None) -> Complex:
        """The phasor of the current into `terminal` of `instance` at each frequency.

        Without a terminal, the phasor of the current through `instance`.
        """
        if terminal is not None:
            columns = terminal_columns(
                self.terminals, self.composites, instance, terminal
            )
            return _summed(self.currents, columns)
        return self.values[:, column(self.unknowns, _current(instance))]


@dataclass(frozen=True, slots=True)
class Statistics:
    """How a transient analysis went."""

    initial: str
    accepted: int
    rejected: int
    iterations: int
    periods: int | None = None
    """For a periodic steady state, the periods simulated to find it, the last the one
    returned."""


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
    terminals: tuple[str, ...] = ()
    """Every terminal of every device, in the order of `currents`' columns."""
    currents: Real = field(default_factory=_no_rows)
    """The current into each terminal at each time, one row per time; it includes its
    charge's derivative, by the step's integration formula."""
    charges: Real = field(default_factory=_no_rows)
    """The charge each terminal holds at each time, one row per time."""
    composites: Composites = ()
    """Each terminal of a component instance or a logic device, with the indices of the
    device terminals whose currents and charges sum to its own."""
    derivatives: tuple[tuple[str, str], ...] = ()
    """For each device, every pair of its terminals and internal unknowns, in the order
    of `conductances`' columns."""
    conductances: Real = field(default_factory=_no_rows)
    """Each derivative of a residual at each time, one row per time."""
    capacitances: Real = field(default_factory=_no_rows)
    """Each derivative of a charge at each time, one row per time."""
    variables: tuple[tuple[str, str], ...] = ()
    """Every variable of every device, in the order of `readings`' columns."""
    readings: Real = field(default_factory=_no_rows)
    """Each variable's value at each time, one row per time."""

    def level(self, net: str) -> Trace:
        """The level of the logic net `net` over time."""
        return self.logic[net_column(self.nets, net)]

    def voltage(self, net: str, reference: str | None = None) -> Real:
        """The voltage of `net` over time, relative to `reference` or to ground."""
        values = self.values[:, column(self.unknowns, net)]
        return values if reference is None else values - self.voltage(reference)

    def current(self, instance: str, terminal: str | None = None) -> Real:
        """The current into `terminal` of `instance` over time.

        Without a terminal, the current through `instance`, a source or an inductor.
        """
        if terminal is not None:
            columns = terminal_columns(
                self.terminals, self.composites, instance, terminal
            )
            return _summed(self.currents, columns)
        return self.values[:, column(self.unknowns, _current(instance))]

    def charge(self, instance: str, terminal: str) -> Real:
        """The charge `terminal` of `instance` holds over time."""
        columns = terminal_columns(self.terminals, self.composites, instance, terminal)
        return _summed(self.charges, columns)

    def conductance(self, equation: str, unknown: str) -> Real:
        """The derivative of `equation`'s residual by `unknown` over time."""
        index = derivative_column(self.derivatives, equation, unknown)
        return self.conductances[:, index]

    def capacitance(self, equation: str, unknown: str) -> Real:
        """The derivative of `equation`'s charge by `unknown` over time."""
        index = derivative_column(self.derivatives, equation, unknown)
        return self.capacitances[:, index]

    def variable(self, instance: str, name: str) -> Real:
        """The variable `name` of the device `instance` over time."""
        return self.readings[:, variable_column(self.variables, instance, name)]


@dataclass(frozen=True, slots=True)
class Parameter:
    """One number of the parameters of a device instance."""

    instance: str
    """The device's path, such as ``X1/R1``."""
    name: str
    """The parameter's name, its keys joined by dots within an object and a list's
    entries keyed by their index from 0, such as ``resistance`` or ``waveform.dc``."""
    value: float
    """Its value, in SI base units."""


@dataclass(frozen=True, slots=True)
class Sensitivities:
    """Sensitivities: each output's derivative with respect to each parameter.

    `derivatives` has one row per output and one column per parameter, in SI base
    units, at an operating point. Along a DC sweep, over a response or over time, it has
    one such matrix per level, frequency or time of `result`, first, as `result`'s own
    values have one row per point; a response's are the derivatives of its phasors.

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
        >>> sensitivities.gradient("out").tolist()
        [-0.001, 0.4]
    """

    unknowns: tuple[Unknown, ...]
    outputs: tuple[str, ...]
    """The unknowns differentiated, by name, in the order of `derivatives`' rows."""
    parameters: tuple[Parameter, ...]
    """Every parameter, in the order of `derivatives`' columns."""
    derivatives: Real | Complex
    result: OperatingPoint | Sweep | Response | Waveforms
    """The result of the analysis differentiated."""

    def derivative(
        self, output: str, instance: str, parameter: str
    ) -> float | Real | Complex:
        """The derivative of `output` with respect to `parameter` of `instance`.

        It is one number at an operating point, and one per point of `result` otherwise.

        Raises:
            KeyError: When the analysis has no such output or parameter.
        """
        labels = [f"{found.instance} {found.name}" for found in self.parameters]
        missing = "the analysis has no parameter"
        index = _find(labels, f"{instance} {parameter}", missing)
        found = self.derivatives[..., self._row(output), index]
        return float(found) if found.ndim == 0 else found

    def gradient(self, output: str) -> Real | Complex:
        """The derivative of `output` with respect to each parameter, in their order.

        It is one per point of `result`, first, but at an operating point.
        """
        return self.derivatives[..., self._row(output), :]

    def _row(self, output: str) -> int:
        return _find(list(self.outputs), output, "the analysis has no output")


@dataclass(frozen=True, slots=True)
class LoopGain:
    """The loop gain through a probe, at each frequency.

    Negative feedback makes it positive at low frequencies; the specs' `phase_margin`,
    `gain_margin` and `crossover` read how far it stays from -1.
    """

    unknowns: tuple[Unknown, ...]
    probe: str
    """The path of the probe the loop is broken at, such as ``VP``."""
    frequencies: Real
    gains: Complex
    """The loop gain at each frequency."""
    operating_point: OperatingPoint


@dataclass(frozen=True, slots=True)
class NoiseSource:
    """One noise source of a device."""

    instance: str
    """The device's path, such as ``X1/Q1``."""
    name: str
    """What makes the noise, such as ``thermal``, ``collector shot`` or ``flicker``."""


@dataclass(frozen=True, slots=True)
class NoiseInput:
    """The independent source a noise analysis refers its output's noise back to."""

    source: str
    """The source's path, such as ``V1``."""
    quantity: Literal["voltage", "current"]
    """What the source's level is."""
    gains: Complex
    """The gain from the source's level to the output at each frequency."""


@dataclass(frozen=True, slots=True)
class NoiseSpectrum:
    """The noise at an output, at each frequency.

    `densities` are the output's one-sided noise power densities, in V²/Hz for a
    voltage and A²/Hz for a current, and `contributions` what each of `sources` adds to
    them, one row per source. Given an input, ``referred`` divides them by the gain's
    square, as the input's own noise would be. ``rms`` integrates a density over a
    band, as a power law between frequencies, as SPICE does.

    Examples:
        >>> point = OperatingPoint((Unknown("out", "voltage"),), np.array([2.0]), "")
        >>> spectrum = NoiseSpectrum(
        ...     point.unknowns,
        ...     "out",
        ...     None,
        ...     "voltage",
        ...     np.array([1.0, 1e4]),
        ...     np.array([4e-18, 4e-18]),
        ...     (NoiseSource("R1", "thermal"),),
        ...     np.array([[4e-18, 4e-18]]),
        ...     NoiseInput("V1", "voltage", np.array([0.5, 0.5 + 0j])),
        ...     point,
        ... )
        >>> f"{spectrum.rms():.3g}", spectrum.referred().tolist()
        ('2e-07', [1.6e-17, 1.6e-17])
        >>> spectrum.density(100.0)
        4e-18
    """

    unknowns: tuple[Unknown, ...]
    output: str
    """The output's unknown, by name."""
    reference: str | None
    """The node the output voltage is taken over; the ground if None."""
    quantity: Literal["voltage", "current"]
    """What the output is."""
    frequencies: Real
    densities: Real
    sources: tuple[NoiseSource, ...]
    contributions: Real
    input: NoiseInput | None
    operating_point: OperatingPoint

    def referred(self) -> Real:
        """The noise referred to the input at each frequency, in V²/Hz or A²/Hz.

        It is the output's density over the gain's square, in the unit of the source's
        level squared per hertz.

        Raises:
            ValueError: When the analysis named no source to refer its noise to.
        """
        if self.input is None:
            message = "the noise analysis named no source to refer its noise to"
            raise ValueError(message)
        with np.errstate(divide="ignore"):
            return self.densities / np.abs(self.input.gains) ** 2

    def contribution(self, instance: str, name: str | None = None) -> Real:
        """What the noise of `instance`, or its source `name`, adds at each frequency.

        It is in the unit of the output's densities.

        Raises:
            KeyError: When the instance makes no such noise.
        """
        rows = [
            index
            for index, source in enumerate(self.sources)
            if source.instance == instance and name in {None, source.name}
        ]
        if not rows:
            labels = [f"{source.instance} {source.name}" for source in self.sources]
            wanted = instance if name is None else f"{instance} {name}"
            _find(labels, wanted, "the analysis has no noise source")
        return self.contributions[rows].sum(axis=0)

    def density(self, at: float, *, referred: bool = False) -> float:
        """The density at the frequency `at`, in hertz.

        It is a power law between the frequencies around it.

        Raises:
            ValueError: When the frequency does not lie within the frequencies, or the
                analysis named no source to refer its noise to.
        """
        densities = self.referred() if referred else self.densities
        order = np.argsort(self.frequencies)
        points = self.frequencies[order]
        if not points[0] <= at <= points[-1]:
            message = (
                f"{at!r} Hz does not lie within the frequencies, "
                f"{float(points[0])!r} Hz to {float(points[-1])!r} Hz"
            )
            raise ValueError(message)
        return _power_law(points, densities[order], at)

    def rms(
        self,
        start: float | None = None,
        stop: float | None = None,
        *,
        referred: bool = False,
    ) -> float:
        """The RMS noise from `start` to `stop`, in hertz, by default every frequency.

        It is the square root of the density integrated over the band, in volts or
        amperes, the density a power law between frequencies, as SPICE integrates it.

        Raises:
            ValueError: When the band does not lie within the frequencies, or is
                empty, or the analysis named no source to refer its noise to.
        """
        densities = self.referred() if referred else self.densities
        return math.sqrt(integrate(self.frequencies, densities, start, stop))


def integrate(
    frequencies: Real,
    densities: Real,
    start: float | None = None,
    stop: float | None = None,
) -> float:
    """The integral of `densities` from `start` to `stop`, by default every frequency.

    The density is a power law between `frequencies`, as SPICE integrates noise.

    Examples:
        >>> integrate(np.array([1.0, 10.0]), np.array([1.0, 0.1]))
        2.3025850929940463
        >>> round(integrate(np.array([1.0, 100.0]), np.array([2.0, 2.0]), 10.0))
        180

    Raises:
        ValueError: When the band does not lie within the frequencies, or is empty.
    """
    order = np.argsort(frequencies)
    points, values = frequencies[order], densities[order]
    low = float(points[0]) if start is None else start
    high = float(points[-1]) if stop is None else stop
    if not points[0] <= low < high <= points[-1]:
        message = (
            f"a band from {low!r} Hz to {high!r} Hz does not lie within the "
            f"frequencies, {float(points[0])!r} Hz to {float(points[-1])!r} Hz"
        )
        raise ValueError(message)
    inside = (points > low) & (points < high)
    band = np.concatenate(([low], points[inside], [high]))
    levels = np.concatenate(
        (
            [_power_law(points, values, low)],
            values[inside],
            [_power_law(points, values, high)],
        )
    )
    return math.fsum(
        _segment(
            *map(float, band[index : index + 2]), *map(float, levels[index : index + 2])
        )
        for index in range(band.size - 1)
    )


def _power_law(points: Real, values: Real, at: float) -> float:
    """The density at `at`, a power law between the two frequencies around it."""
    index = int(np.clip(np.searchsorted(points, at), 1, points.size - 1))
    first, last = float(points[index - 1]), float(points[index])
    low, high = float(values[index - 1]), float(values[index])
    if at in {first, last}:
        return low if at == first else high
    if low <= 0 or high <= 0:
        return low + (high - low) * (at - first) / (last - first)
    return low * math.exp(
        math.log(high / low) * math.log(at / first) / math.log(last / first)
    )


def _segment(first: float, last: float, low: float, high: float) -> float:
    """The integral from `first` to `last` of the power law from `low` to `high`."""
    if low <= 0 or high <= 0:
        return 0.5 * (low + high) * (last - first)
    span = math.log(last / first)
    # With the density ∝ f^a, the integral is low·first·(e^{(a+1)·span} - 1)/(a + 1).
    growth = math.log(high / low) + span
    factor = 1.0 if growth == 0 else math.expm1(growth) / growth
    return low * first * span * factor


@dataclass(frozen=True, slots=True)
class PolesZeros:
    """The poles of the linearized circuit, and the zeros of a transfer function.

    The circuit is linearized at its operating point, and the transfer function is from
    a source to an output, given both. Each is a complex frequency `s`, in radians per
    second, by increasing magnitude, a conjugate pair's member above the real axis
    first; a pole in the right half-plane, with a positive real part, grows.
    ``natural_frequencies`` and ``dampings`` read them as a frequency and a damping
    ratio, as a second-order section's are written.

    Examples:
        >>> point = OperatingPoint((Unknown("out", "voltage"),), np.array([0.0]), "")
        >>> found = PolesZeros(
        ...     point.unknowns,
        ...     np.array([-3 + 4j, -3 - 4j]),
        ...     np.array([0j]),
        ...     "V1",
        ...     "out",
        ...     None,
        ...     point,
        ... )
        >>> natural_frequencies(found.poles) * 2 * np.pi
        array([5., 5.])
        >>> dampings(found.poles)
        array([0.6, 0.6])
    """

    unknowns: tuple[Unknown, ...]
    poles: Complex
    zeros: Complex | None
    """The zeros, given a source and an output."""
    source: str | None
    """The path of the source the transfer function is from."""
    output: str | None
    """The output the transfer function is to."""
    reference: str | None
    """The node the output voltage is taken over; the ground if None."""
    operating_point: OperatingPoint


def natural_frequencies(roots: Complex) -> Real:
    """The natural frequency of each of `roots`, its magnitude over 2π, in hertz."""
    return np.abs(roots) / (2 * np.pi)


def dampings(roots: Complex) -> Real:
    """The damping ratio of each of `roots`, minus its real part over its magnitude.

    It is 1 for a real root in the left half-plane, between 0 and 1 for a conjugate pair
    that rings, and negative in the right half-plane; NaN at the origin.
    """
    with np.errstate(invalid="ignore"):
        return -roots.real / np.abs(roots)


type Result = (
    OperatingPoint
    | OperatingPoints
    | Sweep
    | Response
    | Waveforms
    | Sensitivities
    | LoopGain
    | NoiseSpectrum
    | PolesZeros
)
"""The result of any analysis."""
