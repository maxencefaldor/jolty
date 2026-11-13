# Copyright 2026 Maxence Faldor
"""The results of analyses, read by net and by instance.

Every result holds the circuit's unknowns: node voltages, named by net path such as
``out`` or ``amplifier/mid``, and branch currents and other internal unknowns, named by
instance path and label such as ``V1:current``. ``voltage`` and ``current`` read them by
name. It holds the levels of the circuit's logic nets too, which ``level`` reads by
name: 0, 1, or NaN for an unknown level.

It holds what its [`Recording`][jolty.results.Recording] says of the instances its
plan's or its analysis's ``record`` names. For a device: the current into each terminal,
and the charge the terminal holds, which ``current`` and ``charge`` read by instance and
terminal, such as ``X1/R1`` and ``a``; its own small-signal model, the derivative of
each of its equations with respect to each of its unknowns, which ``conductance`` and
``capacitance`` read; and the variables it computes on its way, such as a transistor's
threshold voltage, which ``variable`` reads. For a component instance or a logic device:
its terminals' currents and charges, such as ``X1`` and ``vdd``, each the sum of the
device terminals behind it.
"""

import difflib
import functools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Never, TypedDict, override

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


type Quantity = Literal["voltage", "current", "temperature", "state"]
"""What an unknown is: a node's voltage, a branch's current, a thermal net's
temperature above the ambient, in kelvins, or a device's dimensionless state, such as a
memristor's."""


@dataclass(frozen=True, slots=True)
class Unknown:
    """An unknown solved for: a voltage, a current, a temperature or a state."""

    name: str
    quantity: Quantity


def column(
    unknowns: tuple[Unknown, ...], name: str, quantity: Quantity | None = None
) -> int:
    """The column of the unknown `name` in the values of a result over `unknowns`.

    The column is found in a dictionary of the unknowns by name, made the first time
    one of them is read and kept for the last four tuples of unknowns read, each a
    circuit's that its results share.

    Raises:
        KeyError: When no unknown has that name, the message suggesting the closest
            names, or when `quantity` is given and the unknown is not of it.
    """
    index = _columns(_Same(unknowns)).get(name)
    if index is None:
        names = [unknown.name for unknown in unknowns]
        message = f"the circuit has no unknown named {name!r}{close_names(names, name)}"
        raise KeyError(message)
    found = unknowns[index].quantity
    if quantity is not None and found != quantity:
        message = f"{name!r} holds a {found}, not a {quantity}"
        raise KeyError(message)
    return index


@dataclass(frozen=True, slots=True, eq=False)
class _Same:
    """The unknowns of one circuit, the same object only equal to itself.

    It keys `_columns` by identity, which compares in constant time where the tuple
    would compare and hash every unknown.
    """

    unknowns: tuple[Unknown, ...]

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Same) and other.unknowns is self.unknowns

    @override
    def __hash__(self) -> int:
        return id(self.unknowns)


@functools.lru_cache(maxsize=4)
def _columns(same: _Same) -> dict[str, int]:
    """The column of each unknown, by name."""
    return {unknown.name: index for index, unknown in enumerate(same.unknowns)}


@dataclass(frozen=True, slots=True)
class RecordedDevice:
    """A device whose terminals, derivatives and variables a result recorded."""

    path: str
    """Its path, such as ``X1/M1``."""
    terminals: tuple[str, ...]
    """Its terminals' names, such as ``drain``, in their order among its unknowns."""
    internals: tuple[str, ...] = ()
    """Its internal unknowns, by name, such as ``X1/M1:drain``, after its terminals."""
    variables: tuple[str, ...] = ()
    """Its variables' names, such as ``threshold_voltage``."""

    @property
    def unknowns(self) -> tuple[str, ...]:
        """Its unknowns as its derivatives name them, terminals then internal ones.

        A terminal is named by the device's path and its own name, such as
        ``X1/M1.drain``.
        """
        return (
            *(f"{self.path}.{terminal}" for terminal in self.terminals),
            *self.internals,
        )


@dataclass(frozen=True, slots=True)
class _Index:
    """The columns of a recording, by name."""

    terminals: tuple[str, ...]
    """The name of each column of the currents."""
    columns: dict[str, int]
    """The column of the currents of each terminal, by name."""
    unknowns: dict[str, tuple[int, int]]
    """Each device unknown's device and its index among the device's, by name."""
    blocks: tuple[tuple[int, int], ...]
    """Each device's first column of the derivatives, and its number of unknowns."""
    variables: dict[tuple[str, str], int]
    """The column of the readings of each variable, by device and name."""
    paths: frozenset[str]
    """The devices' paths."""

    @classmethod
    def of(cls, recording: Recording) -> _Index:
        """The columns of `recording`."""
        terminals = (
            *(
                f"{device.path}.{terminal}"
                for device in recording.devices
                for terminal in device.terminals
            ),
            *recording.composites,
        )
        unknowns: dict[str, tuple[int, int]] = {}
        blocks: list[tuple[int, int]] = []
        start = 0
        for position, device in enumerate(recording.devices):
            local = device.unknowns
            unknowns.update(
                (name, (position, index)) for index, name in enumerate(local)
            )
            blocks.append((start, len(local)))
            start += len(local) ** 2
        variables = (
            (device.path, variable)
            for device in recording.devices
            for variable in device.variables
        )
        return cls(
            terminals,
            {name: index for index, name in enumerate(terminals)},
            unknowns,
            tuple(blocks),
            {variable: index for index, variable in enumerate(variables)},
            frozenset(device.path for device in recording.devices),
        )


@dataclass(frozen=True, slots=True)
class Recording:
    """What a result recorded of the devices beside the unknowns, and where.

    A result's `currents` and `charges` have a column for each terminal of each of
    `devices`, in order, then one for each of `composites`, each the sum of the device
    terminals behind it; its `conductances` and `capacitances` have one for each pair of
    each device's unknowns, its terminals then its internal unknowns, row by row; and
    its `readings` one for each of each device's variables. Each is found by name in a
    dictionary, made the first time one is read. A simulation records what the plan's or
    the analysis's ``record`` names (see [`Plan`][jolty.plan.Plan]).

    Examples:
        >>> recording = Recording((RecordedDevice("R1", ("a", "b")),), ("X1.p",))
        >>> recording.terminals
        ('R1.a', 'R1.b', 'X1.p')
        >>> recording.terminal("X1", "p"), recording.derivative("R1.b", "R1.a")
        (2, 2)
    """

    devices: tuple[RecordedDevice, ...] = ()
    """The devices recorded, in order."""
    composites: tuple[str, ...] = ()
    """The terminals recorded of component instances and logic devices, such as
    ``X1.vdd``, in order."""
    thermal: frozenset[str] = frozenset()
    """The thermal terminals among the devices' and the composites', such as
    ``Q1.thermal``: each one's current is the heat flowing into it, in watts."""
    _index: list[_Index] = field(
        default_factory=list, init=False, repr=False, compare=False
    )

    @property
    def terminals(self) -> tuple[str, ...]:
        """The name of each column of the currents and charges, such as ``X1/R1.a``."""
        return self._columns().terminals

    def find(self, instance: str, terminal: str) -> int | None:
        """The column of the currents and charges of `terminal` of `instance`.

        None when the result recorded no such terminal.
        """
        return self._columns().columns.get(f"{instance}.{terminal}")

    def terminal(self, instance: str, terminal: str) -> int:
        """The column of the currents and charges of `terminal` of `instance`.

        Raises:
            KeyError: When the result recorded no such terminal; the message suggests
                the closest names and how to record it.
        """
        found = self.find(instance, terminal)
        if found is None:
            name = f"{instance}.{terminal}"
            recorded = self._holds(instance)
            what = f"terminal named {name!r}"
            _unrecorded(what, self.terminals, name, instance, recorded=recorded)
        return found

    def derivative(self, equation: str, unknown: str) -> int:
        """The column of the derivatives of `equation` with respect to `unknown`.

        Both are unknowns of one device: a terminal, such as ``M1.drain``, for the
        current into it and the voltage at it, or an internal unknown, such as
        ``M1:drain``, for its equation and its value.

        Raises:
            KeyError: When the result recorded no device with both unknowns; the
                message suggests the closest names and how to record the device, its
                path what comes before the last ``.`` or ``:`` of the name.
        """
        index = self._columns()
        found = []
        for name in (equation, unknown):
            located = index.unknowns.get(name)
            if located is None:
                cut = max(name.rfind("."), name.rfind(":"))
                device = name[:cut] if cut > 0 else name
                recorded = device in index.paths
                what = f"{'' if recorded else 'device '}unknown named {name!r}"
                _unrecorded(what, index.unknowns, name, device, recorded=recorded)
            found.append(located)
        (device, row), (other, column) = found
        if device != other:
            message = f"{equation!r} and {unknown!r} are unknowns of two devices"
            raise KeyError(message)
        start, size = index.blocks[device]
        return start + row * size + column

    def variable(self, instance: str, name: str) -> int:
        """The column of the readings of the variable `name` of the device `instance`.

        Raises:
            KeyError: When the result recorded no such variable; the message suggests
                the closest names and how to record it.
        """
        index = self._columns()
        found = index.variables.get((instance, name))
        if found is None:
            labels = [f"{device} {variable}" for device, variable in index.variables]
            recorded = instance in index.paths
            what = f"variable named {name!r}"
            _unrecorded(what, labels, f"{instance} {name}", instance, recorded=recorded)
        return found

    def _holds(self, instance: str) -> bool:
        """Whether the result recorded the device or the terminals at `instance`."""
        prefix = f"{instance}."
        return instance in self._columns().paths or any(
            composite.startswith(prefix) for composite in self.composites
        )

    def _columns(self) -> _Index:
        if not self._index:
            self._index.append(_Index.of(self))
        return self._index[0]


def _unrecorded(
    what: str, names: Iterable[str], name: str, instance: str, *, recorded: bool
) -> Never:
    """Raises the error of reading `what` of the instance at `instance`.

    The result did not record `what`, though it `recorded` the instance when the name
    is mistaken.

    Raises:
        KeyError: Suggesting the closest of the recorded `names` to `name` and, when
            the result did not record the instance, how to record it or every instance.
    """
    close = close_names(list(names), name)
    if recorded:
        message = f"{instance!r}, which the result recorded, has no {what}{close}"
    else:
        message = (
            f"the result recorded no {what}{close}; record its instance, as an "
            f"analysis's or a plan's record=[{instance!r}] does, or every instance, "
            "as record=['**'] does"
        )
    raise KeyError(message)


class Observed(TypedDict):
    """What the recorded devices did, as fields of a point, a sweep or a transient."""

    recording: Recording
    currents: Real
    charges: Real
    conductances: Real
    capacitances: Real
    readings: Real


def net_column(nets: tuple[str, ...], name: str) -> int:
    """The column of the logic net `name` in the levels of a result over `nets`.

    Raises:
        KeyError: When no logic net has that name; the message suggests the closest.
    """
    return _find(list(nets), name, "the circuit has no logic net")


def close_names(names: Sequence[str], name: str) -> str:
    """The names among `names` closest to `name`, as the end of a message, if any.

    Examples:
        >>> close_names(["out", "in"], "ou")
        '; close names: out'
    """
    close = difflib.get_close_matches(name, names, n=3)
    return f"; close names: {', '.join(close)}" if close else ""


def _find(names: list[str], name: str, missing: str) -> int:
    """The index of `name` among `names`.

    Raises:
        KeyError: Saying `missing`, such as "the circuit has no unknown", named `name`,
            and the closest names.
    """
    if name in names:
        return names.index(name)
    message = f"{missing} named {name!r}{close_names(names, name)}"
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


class _Unknowns[T]:
    """A result's unknowns and its recorded terminals' currents, read by name.

    A reading is one number of an operating point, or one per row of a result of many,
    such as a sweep's levels, a transient's times or a response's frequencies.
    """

    __slots__ = ()

    unknowns: tuple[Unknown, ...]
    values: npt.NDArray[Any]
    recording: Recording
    currents: npt.NDArray[Any]

    def _read(self, values: npt.NDArray[Any]) -> T:
        """`values`, a number or a column, as a reading."""
        raise NotImplementedError

    def _unknown(self, name: str, quantity: Quantity | None = None) -> T:
        return self._read(self.values[..., column(self.unknowns, name, quantity)])

    def voltage(self, net: str, reference: str | None = None) -> T:
        """The voltage of `net`, relative to `reference` or to ground."""
        values = self.values[..., column(self.unknowns, net, "voltage")]
        if reference is not None:
            values = (
                values - self.values[..., column(self.unknowns, reference, "voltage")]
            )
        return self._read(values)

    def rise(self, net: str) -> T:
        """The temperature of the thermal net `net` above the ambient, in kelvins."""
        return self._unknown(net, "temperature")

    def state(self, name: str) -> T:
        """The device's state `name`, such as a memristor's ``M1:state``."""
        return self._unknown(name, "state")

    def current(self, instance: str, terminal: str | None = None) -> T:
        """The current into `terminal` of `instance`.

        The instance is a device, a component instance or a logic device, which the
        result recorded; into a thermal terminal, the current is the heat flowing in, in
        watts. Without a terminal, the current through `instance`, a source or an
        inductor, in its direction, which every result holds.

        Raises:
            KeyError: When the result holds no such current; the message suggests the
                closest names, and how to record a terminal.
        """
        if terminal is None:
            return self._unknown(_current(instance))
        return self._read(
            self.currents[..., self.recording.terminal(instance, terminal)]
        )


class _Devices[T](_Unknowns[T]):
    """A result's unknowns, and its devices' charges, derivatives and variables, read.

    Each is read by name, as `_Unknowns` reads an unknown, from what the result
    recorded.
    """

    __slots__ = ()

    charges: npt.NDArray[Any]
    conductances: npt.NDArray[Any]
    capacitances: npt.NDArray[Any]
    readings: npt.NDArray[Any]

    def charge(self, instance: str, terminal: str) -> T:
        """The charge `terminal` of `instance` holds.

        Raises:
            KeyError: When the result recorded no such terminal.
        """
        return self._read(
            self.charges[..., self.recording.terminal(instance, terminal)]
        )

    def conductance(self, equation: str, unknown: str) -> T:
        """The derivative of a device's `equation`'s residual with respect to `unknown`.

        For two terminals, such as ``M1.drain`` and ``M1.gate``, it is the change of the
        current into the first with the voltage at the second.

        Raises:
            KeyError: When the result recorded no device with both unknowns.
        """
        index = self.recording.derivative(equation, unknown)
        return self._read(self.conductances[..., index])

    def capacitance(self, equation: str, unknown: str) -> T:
        """The derivative of a device's `equation`'s charge with respect to `unknown`.

        For two terminals, it is the change of the first's charge with the voltage at
        the second.

        Raises:
            KeyError: When the result recorded no device with both unknowns.
        """
        index = self.recording.derivative(equation, unknown)
        return self._read(self.capacitances[..., index])

    def variable(self, instance: str, name: str) -> T:
        """The variable `name` of the device `instance`, such as a threshold voltage.

        Raises:
            KeyError: When the result recorded no such variable.
        """
        return self._read(self.readings[..., self.recording.variable(instance, name)])


@dataclass(frozen=True, slots=True)
class OperatingPoint(_Devices[float]):
    """A DC solution, and the strategy that found it."""

    unknowns: tuple[Unknown, ...]
    values: Real
    strategy: str
    nets: tuple[str, ...] = ()
    """The logic nets, in the order of `logic`."""
    logic: Real = field(default_factory=_none)
    """Each logic net's level."""
    recording: Recording = field(default_factory=Recording)
    """What the result recorded of the devices, and where its arrays hold each."""
    currents: Real = field(default_factory=_none)
    """The current into each recorded terminal, in amperes."""
    charges: Real = field(default_factory=_none)
    """The charge each recorded terminal holds, in coulombs."""
    conductances: Real = field(default_factory=_none)
    """The derivative of each recorded equation's residual with respect to each unknown
    of its device."""
    capacitances: Real = field(default_factory=_none)
    """The derivative of each recorded equation's charge with respect to each unknown
    of its device."""
    readings: Real = field(default_factory=_none)
    """Each recorded variable's value, in SI base units."""

    def level(self, net: str) -> float:
        """The level of the logic net `net`: 0, 1, or NaN when unknown."""
        return float(self.logic[net_column(self.nets, net)])

    @override
    def _read(self, values: Real) -> float:
        return float(values)


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
        index = column(self.unknowns, net, "voltage")
        return np.array([float(point.values[index]) for point in self.points])


@dataclass(frozen=True, slots=True)
class Sweep(_Devices[Real]):
    """The operating points of a DC sweep, one row per level of its source."""

    unknowns: tuple[Unknown, ...]
    source: str
    """The path of the swept source, such as ``V1``."""
    quantity: Quantity
    """Whether the levels are voltages or currents."""
    levels: Real
    values: Real
    nets: tuple[str, ...] = ()
    """The logic nets, in the order of `logic`'s columns."""
    logic: Real = field(default_factory=_no_rows)
    """Each logic net's level at each level of the source, one row per level."""
    recording: Recording = field(default_factory=Recording)
    """What the result recorded of the devices, and where its arrays hold each."""
    currents: Real = field(default_factory=_no_rows)
    """The current into each recorded terminal at each level, one row per level."""
    charges: Real = field(default_factory=_no_rows)
    """The charge each recorded terminal holds at each level, one row per level."""
    conductances: Real = field(default_factory=_no_rows)
    """Each recorded derivative of a residual at each level, one row per level."""
    capacitances: Real = field(default_factory=_no_rows)
    """Each recorded derivative of a charge at each level, one row per level."""
    readings: Real = field(default_factory=_no_rows)
    """Each recorded variable's value at each level, one row per level."""

    def level(self, net: str) -> Real:
        """The level of the logic net `net` at each level: 0, 1, or NaN when unknown."""
        return self.logic[:, net_column(self.nets, net)]

    @override
    def _read(self, values: Real) -> Real:
        return values


@dataclass(frozen=True, slots=True)
class Response(_Unknowns[Complex]):
    """A small-signal response: phasors per unit excitation, one row per frequency."""

    unknowns: tuple[Unknown, ...]
    source: str
    """The path of the excited source, such as ``V1``."""
    frequencies: Real
    values: Complex
    operating_point: OperatingPoint
    recording: Recording = field(default_factory=Recording)
    """What the result recorded of the devices, and where its currents hold each."""
    currents: Complex = field(default_factory=_no_phasors)
    """The phasor of the current into each recorded terminal, one row per frequency."""

    @override
    def _read(self, values: Complex) -> Complex:
        return values


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
class Waveforms(_Devices[Real]):
    """The waveforms of a transient analysis, one row per accepted time point."""

    unknowns: tuple[Unknown, ...]
    times: Real
    values: Real
    statistics: Statistics
    nets: tuple[str, ...] = ()
    """The logic nets, in the order of `logic`."""
    logic: tuple[Trace, ...] = ()
    """Each logic net's level over time."""
    recording: Recording = field(default_factory=Recording)
    """What the result recorded of the devices, and where its arrays hold each."""
    currents: Real = field(default_factory=_no_rows)
    """The current into each recorded terminal at each time, one row per time; it
    includes its charge's derivative, by the step's integration formula."""
    charges: Real = field(default_factory=_no_rows)
    """The charge each recorded terminal holds at each time, one row per time."""
    conductances: Real = field(default_factory=_no_rows)
    """Each recorded derivative of a residual at each time, one row per time."""
    capacitances: Real = field(default_factory=_no_rows)
    """Each recorded derivative of a charge at each time, one row per time."""
    readings: Real = field(default_factory=_no_rows)
    """Each recorded variable's value at each time, one row per time."""

    def level(self, net: str) -> Trace:
        """The level of the logic net `net` over time."""
        return self.logic[net_column(self.nets, net)]

    @override
    def _read(self, values: Real) -> Real:
        return values


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
    quantity: Quantity
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
    quantity: Quantity
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
