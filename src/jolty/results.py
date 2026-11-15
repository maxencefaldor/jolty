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

Each result states where its analysis used logic devices outside the domains their
timing was measured over, its ``excursions``, as the engine watched it.

A [`BatchResult`][jolty.results.BatchResult] holds what many runs of one circuit found
as columns, one row per run: the values each run took, whether each analysis ended,
what each measurement read, and the values of some unknowns at every point of an
analysis.
"""

import difflib
import functools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Never, TypedDict, override

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from jolty import _core
from jolty.units import Dimension
from jolty.unknowns import Quantity, Unknown

type Real = npt.NDArray[np.float64]
type Complex = npt.NDArray[np.complex128]

type Plain = Literal["dB", "°", "%", ""]
"""The unit of a measurement that reads a plain number: none for a ratio."""


def _none() -> Real:
    return np.empty(0)


def _no_rows() -> Real:
    return np.empty((0, 0))


def _no_phasors() -> Complex:
    return np.empty((0, 0), dtype=np.complex128)


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
    regions: tuple[str, ...] = ()
    """The regions of operation its model's equations distinguish, such as
    ``saturated``, in the order its variable ``region`` numbers them from 0: none for a
    device that is no transistor."""
    margins: tuple[str, ...] = ()
    """Its variables that are its margins to the regions' boundaries, in volts, such as
    ``saturation_margin``: each is zero on a boundary, where its model's equations
    switch from one region to the next, and changes sign across it."""

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
    devices: dict[str, RecordedDevice]
    """The devices, by path."""

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
            {device.path: device for device in recording.devices},
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
                recorded = device in index.devices
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
            recorded = instance in index.devices
            what = f"variable named {name!r}"
            _unrecorded(what, labels, f"{instance} {name}", instance, recorded=recorded)
        return found

    def regions(self, instance: str) -> tuple[str, ...]:
        """The regions of operation the model of the device `instance` distinguishes.

        Raises:
            KeyError: When the result recorded no such device, suggesting the closest
                and how to record it, or its model distinguishes no regions.
        """
        index = self._columns()
        device = index.devices.get(instance)
        if device is None:
            what = f"device named {instance!r}"
            _unrecorded(what, index.devices, instance, instance, recorded=False)
        if not device.regions:
            message = f"the model of {instance!r} distinguishes no regions of operation"
            raise KeyError(message)
        return device.regions

    def _holds(self, instance: str) -> bool:
        """Whether the result recorded the device or the terminals at `instance`."""
        prefix = f"{instance}."
        return instance in self._columns().devices or any(
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


class Excursion(BaseModel):
    """A logic device used outside its domain, where its timing was never measured."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(description="The device's path.")
    bound: Literal["input transition", "load", "supply"] = Field(
        description="What left the domain."
    )
    low: float = Field(description="The domain's least value, in SI base units.")
    high: float = Field(description="The domain's most value.")
    value: float = Field(description="The value furthest outside it.")
    time: float = Field(
        description="When it first left, in seconds; zero at an operating point."
    )
    count: int = Field(description="How many times it was read outside.")


@dataclass(frozen=True, slots=True)
class _Watched:
    """A result of an analysis that the engine watches for logic devices' domains."""

    excursions: tuple[Excursion, ...] | None = field(default=None, kw_only=True)
    """Where the analysis used logic devices outside their domains, each device and
    bound once, as [`Simulation.run`][jolty.simulate.Simulation.run] watched it, and as
    a results file keeps it. None where no run watched it: a run of a batch, a run in
    ngspice, and a result within another's, such as a response's operating point, whose
    analysis's result states them."""


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
class OperatingPoint(_Devices[float], _Watched):
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

    def region(self, instance: str) -> str:
        """The region of operation of the transistor `instance`, such as ``saturated``.

        Its model's equations say what it is, numbering it among its recorded device's
        `regions` with its variable ``region``; its margins to the regions' boundaries
        are variables too, which the device's `margins` names, such as
        ``variable("M1", "saturation_margin")``.

        Raises:
            KeyError: When the result recorded no such device, or its model
                distinguishes no regions.
        """
        regions = self.recording.regions(instance)
        return regions[int(self.variable(instance, _core.REGION))]

    @override
    def _read(self, values: Real) -> float:
        return float(values)


@dataclass(frozen=True, slots=True)
class OperatingPoints(_Watched):
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
class Sweep(_Devices[Real], _Watched):
    """The operating points of a DC sweep, one row per level of its source.

    A sweep of a parameter of a device names it, and its levels are the parameter's
    values, in SI base units.
    """

    unknowns: tuple[Unknown, ...]
    source: str
    """The path of the swept instance, such as ``V1``."""
    quantity: Quantity | None
    """Whether the levels are voltages or currents; None for a parameter's."""
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
    parameter: str | None = None
    """The parameter swept, by its path among the instance's, such as ``resistance``;
    None for a source's level."""

    def level(self, net: str) -> Real:
        """The level of the logic net `net` at each level: 0, 1, or NaN when unknown."""
        return self.logic[:, net_column(self.nets, net)]

    @override
    def _read(self, values: Real) -> Real:
        return values


@dataclass(frozen=True, slots=True)
class Response(_Unknowns[Complex], _Watched):
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
class Work:
    """The work an analysis did: its Newton iterations and its sparse solver's."""

    iterations: int
    """Every Newton iteration: those of the operating point it solved, unless the
    simulation took the one it remembered, and those of every step it tried."""
    factorizations: int
    """Factorizations that searched for every pivot: the simulation's first, and any
    after a matrix the solver could not factor."""
    refactorizations: int
    """Factorizations that kept every pivot of the last, with no search."""
    fallbacks: int
    """Factorizations that kept the last's pivots up to one their matrix made too small,
    and searched for the pivots from that one on."""
    reuses: int
    """Factorizations given the very matrix factored last, which kept its factors."""


@dataclass(frozen=True, slots=True)
class Profile:
    """Where an analysis spent its time, in seconds, each span in one activity.

    Examples:
        >>> Profile(0.1, 0.4, 0.1, 0.2, 0.1, 0.05, 0.03, 0.02).total
        1.0
    """

    preparing: float
    """Checking the options, making again the devices the analysis records, and
    gathering the sources' breakpoints."""
    evaluating: float
    """Evaluating the devices."""
    assembling: float
    """Turning what the devices contributed into each Newton iteration's matrix and
    right-hand side, and judging its residuals against their tolerances."""
    factoring: float
    """Factoring matrices."""
    solving: float
    """Solving with the factors."""
    step_control: float
    """Judging each step by its error, choosing the next, and finding the events a step
    lands on."""
    recording: float
    """Reading the recorded devices at each point kept."""
    other: float
    """Everything else, such as Newton iteration's updates and the tests of their
    steps."""

    @property
    def total(self) -> float:
        """The whole time: the sum of every activity's."""
        return (
            self.preparing
            + self.evaluating
            + self.assembling
            + self.factoring
            + self.solving
            + self.step_control
            + self.recording
            + self.other
        )


@dataclass(frozen=True, slots=True)
class Statistics:
    """How a transient analysis went."""

    initial: str
    accepted: int
    rejected: int
    """Steps rejected for their error, or for Newton iteration that did not converge."""
    landed: int
    """Steps tried again shorter to land on an event within them: a detector's crossing,
    a logic driver's change or a device's jump."""
    iterations: int
    """The Newton iterations of the accepted steps."""
    periods: int | None = None
    """For a periodic steady state, the periods simulated to find it, the last the one
    returned."""
    work: Work | None = None
    """The work the engine did, its operating point's included, and for a periodic
    steady state every period's; None for ngspice's, which it does not count. Unlike
    the results, it depends on what the simulation ran before: on an operating point it
    remembers, and on the pivots its sparse solver kept."""
    profile: Profile | None = None
    """Where the engine's time went, when the options' ``profile`` is on."""


@dataclass(frozen=True, slots=True)
class Waveforms(_Devices[Real], _Watched):
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
class Sensitivities(_Watched):
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
class LoopGain(_Watched):
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
class Transfer(_Watched):
    """The small-signal transfer function at DC from a source to an output.

    The `gain` is the output per unit of the source's level, in volts or amperes of
    output per volt or ampere of input; the `input_resistance` is what the source sees,
    and the `output_resistance` what the output sees with the source at its level, each
    in ohms and infinite where no current can flow.
    """

    unknowns: tuple[Unknown, ...]
    source: str
    """The path of the input source, such as ``V1``."""
    input: Quantity
    """What the source's level is: ``voltage`` or ``current``."""
    output: str
    """The output: a net, such as ``out``, or a voltage source's current."""
    reference: str | None
    """The net the output voltage is taken over, or None for the ground."""
    gain: float
    input_resistance: float
    output_resistance: float
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
class NoiseSpectrum(_Watched):
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
        return _core.referred(
            np.asarray(self.densities, dtype=np.float64),
            np.asarray(self.input.gains, dtype=np.complex128),
        )

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
        try:
            return _core.density(
                np.asarray(self.frequencies, dtype=np.float64),
                np.asarray(densities, dtype=np.float64),
                at,
            )
        except _core.MeasureError as error:
            reason, numbers = error.args
            if reason != "outside_at":
                raise ValueError(_UNSPECTRAL[reason]) from None
            first, last = numbers
            message = (
                f"{at!r} Hz does not lie within the frequencies, "
                f"{first!r} Hz to {last!r} Hz"
            )
            raise ValueError(message) from None

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


_UNSPECTRAL: Final[dict[str, str]] = {
    "empty": "a spectrum of no frequencies has no density",
    "mismatch": "a spectrum holds a density at each frequency",
}
"""Why frequencies and densities make no spectrum, by the engine's reason."""


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
    try:
        return _core.integrate(
            np.asarray(frequencies, dtype=np.float64),
            np.asarray(densities, dtype=np.float64),
            start,
            stop,
        )
    except _core.MeasureError as error:
        reason, numbers = error.args
        if reason != "band":
            raise ValueError(_UNSPECTRAL[reason]) from None
        low, high, first, last = numbers
        message = (
            f"a band from {low!r} Hz to {high!r} Hz does not lie within the "
            f"frequencies, {first!r} Hz to {last!r} Hz"
        )
        raise ValueError(message) from None


@dataclass(frozen=True, slots=True)
class PolesZeros(_Watched):
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
    return _core.natural_frequencies(np.asarray(roots, dtype=np.complex128))


def dampings(roots: Complex) -> Real:
    """The damping ratio of each of `roots`, minus its real part over its magnitude.

    It is 1 for a real root in the left half-plane, between 0 and 1 for a conjugate pair
    that rings, and negative in the right half-plane; NaN at the origin.
    """
    return _core.dampings(np.asarray(roots, dtype=np.complex128))


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
    | Transfer
)
"""The result of any analysis."""


@dataclass(frozen=True, slots=True)
class Traced:
    """The values of some unknowns at each point of one analysis, over a batch's runs.

    `values` has one row per run, one column per point, then one entry per unknown:
    not a number where the run did not end the analysis. The points are an operating
    point's one, a DC sweep's levels, a response's frequencies, whose values are
    phasors, or the times a transient names, interpolated linearly between the times
    it accepted, so exact at each it stepped onto.
    """

    analysis: str
    """The analysis, by name."""
    along: Literal["point", "level", "frequency", "time"]
    """What the points are."""
    axis: Real
    """Each point: a level, a frequency or a time; none at an operating point."""
    unknowns: tuple[Unknown, ...]
    """The unknowns traced, in the order of `values`' last axis."""
    values: Real | Complex
    source: str | None = None
    """The swept or excited source, or the swept instance, of a sweep or a response."""
    quantity: Quantity | None = None
    """Whether a sweep's levels are voltages or currents; None for a parameter's."""
    parameter: str | None = None
    """The parameter a sweep steps, if any."""

    def values_of(self, name: str) -> Real | Complex:
        """The values of the unknown `name`, one row per run, one column per point.

        Raises:
            KeyError: When the batch traced no unknown of that name.
        """
        names = [unknown.name for unknown in self.unknowns]
        return self.values[:, :, _find(names, name, "the batch traced no unknown")]


@dataclass(frozen=True, slots=True)
class Run:
    """What one run of a batch found, as a function given to the batch reads it."""

    index: int
    """The run's index among the variants' runs."""
    results: dict[str, Result]
    """The result of each analysis that ended, by name."""
    errors: dict[str, str]
    """Why each analysis that did not end failed, by name."""
    problems: tuple[str, ...] = ()
    """The problems the run's values make, which no analysis runs past."""


@dataclass(frozen=True, slots=True)
class BatchResult:
    """What many runs of one circuit found, as columns: one row per run, in order.

    Each run took the `values` of the `parameters` its variants vary, each named as
    ``X1/R1 resistance``, or ``temperature``, in kelvins, and ran the `analyses` in
    order; `ended` says whether each analysis of each run ended. `measured` holds what
    each of the `measurements` read on each run, in its unit among `units`, and not a
    number where it read none, `causes` then indexing `reasons`, the words that say why,
    each once, in the order the cells first name them, row by row, and -1 where it read
    a number. `traced` holds the unknowns a batch traced along its analyses. Every row
    is a pure function of its run's index, whatever the threads, and the reasons a
    function of the rows, so the rows of a range of runs, and the runs of results joined
    one after another, hold what a batch of those runs alone holds.
    """

    analyses: tuple[str, ...]
    parameters: tuple[str, ...]
    values: Real
    ended: npt.NDArray[np.bool_]
    measurements: tuple[str, ...]
    """Each measurement, as its requirement names it, such as ``out in op``."""
    units: tuple[Dimension | Plain | None, ...]
    """The unit of each measurement's values, ``dB``, ``°``, ``%`` or ``""`` for a
    plain number, or None where it read none."""
    measured: Real
    causes: npt.NDArray[np.int32]
    reasons: tuple[str, ...]
    problems: dict[int, tuple[str, ...]] = field(default_factory=dict)
    """The problems the values of each run that made some made, by the run's index."""
    errors: dict[int, dict[str, str]] = field(default_factory=dict)
    """Why each analysis that failed on a run failed, by the run's index then name."""
    traced: tuple[Traced, ...] = ()

    def __post_init__(self) -> None:
        """Keeps each reason once, in the order the cells of `causes` first name them.

        Raises:
            ValueError: When a cause names no reason.
        """
        causes, reasons = _canonical(self.causes, self.reasons)
        object.__setattr__(self, "causes", causes)
        object.__setattr__(self, "reasons", reasons)

    @property
    def runs(self) -> int:
        """The number of runs."""
        return int(self.values.shape[0])

    @property
    def failed(self) -> npt.NDArray[np.bool_]:
        """Whether each run failed: its values made problems, or an analysis failed."""
        return ~self.ended.all(axis=1)

    def failure(self, run: int) -> str | None:
        """Why the run `run` failed, in words; None when it did not."""
        if run in self.problems:
            return "\n\n".join(self.problems[run])
        if run in self.errors:
            return "; ".join(
                f"{name}: {error}" for name, error in self.errors[run].items()
            )
        return None

    def reason(self, run: int, measurement: int) -> str | None:
        """Why the measurement at `measurement` read nothing on `run`, if it did not."""
        cause = int(self.causes[run, measurement])
        return None if cause < 0 else self.reasons[cause]

    def parameter(self, name: str) -> Real:
        """The values of the parameter `name`, such as ``R1 resistance``, one per run.

        Raises:
            KeyError: When the variants vary no such parameter.
        """
        found = _find(list(self.parameters), name, "the variants vary no parameter")
        return self.values[:, found]

    @staticmethod
    def joined(parts: Sequence[BatchResult]) -> BatchResult:
        """The runs of `parts`, one after another, as one result: the inverse of `rows`.

        Each part's runs follow the part's before it, each measurement in the unit its
        parts read it in, and its reasons are those of every part.

        Raises:
            ValueError: When there is no part; or the parts ran other analyses, varied
                other numbers, read other measurements or traced other unknowns or
                points, naming the first part that differs from the first, by its place;
                or read a measurement in other units, naming them.
        """
        if not parts:
            message = "a batch joins one part or more"
            raise ValueError(message)
        first, *rest = parts
        for place, part in enumerate(rest, start=1):
            if (part.analyses, part.parameters, part.measurements) != (
                first.analyses,
                first.parameters,
                first.measurements,
            ) or not _alike(first.traced, part.traced):
                message = (
                    f"part {place} ran other analyses, varied other numbers, read "
                    "other measurements or traced otherwise than part 0"
                )
                raise ValueError(message)
        units = tuple(
            _unit_of(measurement, [part.units[column] for part in parts])
            for column, measurement in enumerate(first.measurements)
        )
        texts: dict[str, int] = {}
        causes = []
        problems: dict[int, tuple[str, ...]] = {}
        errors: dict[int, dict[str, str]] = {}
        offset = 0
        for part in parts:
            table = np.array(
                [texts.setdefault(reason, len(texts)) for reason in part.reasons]
                + [-1],
                dtype=np.int32,
            )
            causes.append(table[part.causes])
            problems.update(
                (offset + index, found) for index, found in part.problems.items()
            )
            errors.update(
                (offset + index, found) for index, found in part.errors.items()
            )
            offset += part.runs
        traced = tuple(
            Traced(
                each.analysis,
                each.along,
                each.axis,
                each.unknowns,
                np.concatenate([part.traced[index].values for part in parts]),
                each.source,
                each.quantity,
                each.parameter,
            )
            for index, each in enumerate(first.traced)
        )
        return BatchResult(
            first.analyses,
            first.parameters,
            np.concatenate([part.values for part in parts]),
            np.concatenate([part.ended for part in parts]),
            first.measurements,
            units,
            np.concatenate([part.measured for part in parts]),
            np.concatenate(causes),
            tuple(texts),
            problems,
            errors,
            traced,
        )

    def rows(self, start: int, stop: int) -> BatchResult:
        """The runs from `start` up to `stop`, from 0, as a result of their own."""

        def kept[T](found: dict[int, T]) -> dict[int, T]:
            return {
                index - start: value
                for index, value in found.items()
                if start <= index < stop
            }

        return BatchResult(
            self.analyses,
            self.parameters,
            self.values[start:stop],
            self.ended[start:stop],
            self.measurements,
            self.units,
            self.measured[start:stop],
            self.causes[start:stop],
            self.reasons,
            kept(self.problems),
            kept(self.errors),
            tuple(
                Traced(
                    traced.analysis,
                    traced.along,
                    traced.axis,
                    traced.unknowns,
                    traced.values[start:stop],
                    traced.source,
                    traced.quantity,
                    traced.parameter,
                )
                for traced in self.traced
            ),
        )


def _canonical(
    causes: npt.NDArray[np.int32], reasons: tuple[str, ...]
) -> tuple[npt.NDArray[np.int32], tuple[str, ...]]:
    """`reasons`, each text once, in the order the cells of `causes` first name them.

    Returns `causes` renumbered with them, unchanged when they already are.

    Raises:
        ValueError: When a cause names no reason.
    """
    named = causes.ravel()
    named = named[named >= 0]
    if named.size and int(named.max()) >= len(reasons):
        message = f"a cause names reason {int(named.max())} of {len(reasons)}"
        raise ValueError(message)
    found, first = np.unique(named, return_index=True)
    texts: dict[str, int] = {}
    table = np.full(len(reasons) + 1, -1, dtype=np.int32)
    for old in found[np.argsort(first, kind="stable")]:
        table[old] = texts.setdefault(reasons[old], len(texts))
    if tuple(texts) == reasons and bool((table[:-1] == np.arange(len(reasons))).all()):
        return causes, reasons
    # A cause of -1 reads the table's last entry, -1 too.
    return table[causes], tuple(texts)


def _alike(traced: tuple[Traced, ...], other: tuple[Traced, ...]) -> bool:
    """Whether `other` traces the unknowns `traced` does, along the same points."""
    return len(traced) == len(other) and all(
        (one.analysis, one.along, one.unknowns, one.source, one.quantity, one.parameter)
        == (
            two.analysis,
            two.along,
            two.unknowns,
            two.source,
            two.quantity,
            two.parameter,
        )
        and np.array_equal(one.axis, two.axis)
        for one, two in zip(traced, other, strict=True)
    )


def _unit_of(
    measurement: str, units: Sequence[Dimension | Plain | None]
) -> Dimension | Plain | None:
    """The one unit among `units` of the parts that read `measurement`, if any read it.

    Raises:
        ValueError: When parts read it in different units.
    """
    read = list(dict.fromkeys(unit for unit in units if unit is not None))
    if len(read) > 1:
        listed = ", ".join(repr(str(unit)) for unit in read)
        message = f"the parts read {measurement} in different units: {listed}"
        raise ValueError(message)
    return read[0] if read else None
