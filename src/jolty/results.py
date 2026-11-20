# Copyright 2026 Maxence Faldor
"""Analysis result containers providing access by net name and instance path.

Every analysis result stores circuit unknowns: node voltages named by net path (such
as ``out`` or ``amplifier/mid``), and branch currents or internal unknowns named by
instance path and label (such as ``V1:current``). The ``voltage`` and ``current``
methods look up unknowns by name. Results also store logic net levels accessed through
``logic_level`` as 0, 1, or NaN for an unknown level.

Results also include data described by their [`Recording`][jolty.results.Recording]
for instances matched by the plan or analysis ``record`` patterns. For primitive
devices, recorded data covers terminal currents and charges (via ``current`` and
``charge``), small-signal derivatives (via ``conductance`` and ``capacitance``), and
internal model variables such as threshold voltage (via ``variable``). For component
instances and logic devices, recorded terminal currents and charges sum the underlying
device terminals.

Watched analyses record logic-device timing-domain excursions in ``excursions``.

A [`BatchResult`][jolty.results.BatchResult] stores tabular data across multiple runs
of one circuit, with one row per run. Columns hold sampled parameter values, analysis
completion flags, evaluated measures, and optional per-point unknown traces.
"""

__lazy_modules__ = ["numpy", "numpy.typing"]

import functools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, ClassVar, Final, Literal, Never, TypedDict, override

import numpy as np
import numpy.typing as npt
from pydantic import Field

from jolty import _core, ir
from jolty.errors import close_names
from jolty.units import Dimension
from jolty.unknowns import Nature, Unknown

type Real = npt.NDArray[np.float64]
type Complex = npt.NDArray[np.complex128]
type _Flags = npt.NDArray[np.bool_]
type _Causes = npt.NDArray[np.int32]

type Plain = Literal["dB", "°", "%", ""]
"""Unit string for a dimensionless measure, with an empty string for a plain ratio."""


def _none() -> Real:
    return np.empty(0)


def _no_rows() -> Real:
    return np.empty((0, 0))


def _no_phasors() -> Complex:
    return np.empty((0, 0), dtype=np.complex128)


def column(
    unknowns: tuple[Unknown, ...], name: str, nature: Nature | None = None
) -> int:
    """Returns the column index of unknown `name` in `unknowns`.

    Lookup uses an identity-keyed dictionary cached for the four most recently
    accessed `unknowns` tuples shared across results of a circuit.

    Args:
        unknowns: Ordered circuit `Unknown` descriptors from the result.
        name: Target unknown name (net path or ``instance:label``).
        nature: Optional expected `Nature` validated against the unknown.

    Returns:
        Zero-based column index of `name` within `unknowns`.

    Raises:
        KeyError: If no unknown matches `name` (suggesting close matches) or if
            `nature` is given and differs from the unknown's physical quantity.
    """
    index = _columns(_Same(unknowns)).get(name)
    if index is None:
        names = [unknown.name for unknown in unknowns]
        message = f"the circuit has no unknown named {name!r}{close_names(names, name)}"
        raise KeyError(message)
    found = unknowns[index].nature
    if nature is not None and found != nature:
        message = f"{name!r} holds a {found}, not a {nature}"
        raise KeyError(message)
    return index


@dataclass(frozen=True, slots=True, eq=False)
class _Same:
    """Identity-equality wrapper around a circuit's `unknowns` tuple.

    Wrapping the tuple lets `_columns` cache lookups by object identity in constant
    time instead of hashing and comparing every unknown.
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
    """Builds a mapping from unknown name to column index for `same.unknowns`."""
    return {unknown.name: index for index, unknown in enumerate(same.unknowns)}


@dataclass(frozen=True, slots=True)
class RecordedDevice:
    """Metadata for a device whose terminals, derivatives, or variables were recorded.

    A transistor recorded only for operating-region reporting has empty `terminals`
    and `internals`, and records only its ``region`` and `margins` variables.
    """

    path: str
    """Hierarchical instance path of the recorded device, such as ``X1/M1``."""
    terminals: tuple[str, ...]
    """Terminal names such as ``drain``, ordered as they appear in device unknowns."""
    internals: tuple[str, ...] = ()
    """Internal unknown names such as ``X1/M1:drain``, following device terminals."""
    variables: tuple[str, ...] = ()
    """Recorded internal model variable names, such as ``threshold_voltage``."""
    regions: tuple[str, ...] = ()
    """Operating region names such as ``saturated`` indexed from 0 by the ``region``
    variable, or empty for non-transistor devices."""
    margins: tuple[str, ...] = ()
    """Variable names for operating-region boundary margins in volts, such as
    ``saturation_margin``. Each margin is zero on a region boundary, changes sign
    across the boundary, and is positive when on or saturated regardless of device
    polarity."""

    @property
    def unknowns(self) -> tuple[str, ...]:
        """Device unknown names in derivative order, terminals before internals.

        Each terminal is qualified by device path and terminal name, such as
        ``X1/M1.drain``.
        """
        return (
            *(f"{self.path}.{terminal}" for terminal in self.terminals),
            *self.internals,
        )


@dataclass(frozen=True, slots=True)
class _Index:
    """Name-to-column lookup tables derived from a `Recording`."""

    terminals: tuple[str, ...]
    """Column names of the recorded terminal current and charge arrays."""
    columns: dict[str, int]
    """Mapping from qualified terminal name to column index."""
    unknowns: dict[str, tuple[int, int]]
    """Mapping from device unknown name to ``(device_index, local_unknown_index)``."""
    blocks: tuple[tuple[int, int], ...]
    """Starting derivative column offset and unknown count for each recorded device."""
    variables: dict[tuple[str, str], int]
    """Mapping from ``(device_path, variable_name)`` to variable column index."""
    devices: dict[str, RecordedDevice]
    """Recorded devices keyed by hierarchical instance path."""

    @classmethod
    def from_recording(cls, recording: Recording) -> _Index:
        """Builds column lookup tables from `recording`."""
        terminals = (
            *(
                f"{device.path}.{terminal}"
                for device in recording.devices
                for terminal in device.terminals
            ),
            *recording.summed_terminals,
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
    """Schema describing recorded device arrays and their column layout in a result.

    Recorded `currents` and `charges` arrays have one column per terminal of each
    device in `devices`, followed by each entry in `summed_terminals`. Recorded
    `conductances` and `capacitances` arrays store row-major derivative blocks over
    each device's terminals and internal unknowns. Recorded `variables` arrays store
    one column per variable of each device. Lookup tables are built lazily on first
    access, covering instances matched by ``record`` in [`Plan`][jolty.plan.Plan].

    Examples:
        >>> recording = Recording((RecordedDevice("R1", ("a", "b")),), ("X1.p",))
        >>> recording.terminals
        ('R1.a', 'R1.b', 'X1.p')
        >>> recording.terminal("X1", "p"), recording.derivative("R1.b", "R1.a")
        (2, 2)
    """

    devices: tuple[RecordedDevice, ...] = ()
    """Recorded primitive devices in column order."""
    summed_terminals: tuple[str, ...] = ()
    """Recorded component-instance and logic-device terminals such as ``X1.vdd``."""
    thermal: frozenset[str] = frozenset()
    """Qualified thermal terminal names such as ``Q1.thermal``, whose currents
    represent heat flow in watts."""
    _index: list[_Index] = field(
        default_factory=list, init=False, repr=False, compare=False
    )

    @property
    def terminals(self) -> tuple[str, ...]:
        """Column names of recorded currents and charges, such as ``X1/R1.a``."""
        return self._columns().terminals

    def find(self, instance: str, terminal: str) -> int | None:
        """Returns the column index for `terminal` on `instance`, or None.

        Args:
            instance: Hierarchical path of the device or component instance.
            terminal: Terminal name on `instance`.

        Returns:
            Zero-based current/charge column index, or ``None`` when unrecorded.
        """
        return self._columns().columns.get(f"{instance}.{terminal}")

    def terminal(self, instance: str, terminal: str) -> int:
        """Returns the column index for `terminal` currents and charges on `instance`.

        Args:
            instance: Hierarchical path of the recorded device or component.
            terminal: Terminal name on `instance`.

        Returns:
            Zero-based column index in the recorded `currents` and `charges` arrays.

        Raises:
            KeyError: If the terminal was not recorded, suggesting close names and how
                to enable recording.
        """
        found = self.find(instance, terminal)
        if found is None:
            name = f"{instance}.{terminal}"
            recorded = self._holds(instance)
            what = f"terminal named {name!r}"
            _unrecorded(what, self.terminals, name, instance, recorded=recorded)
        return found

    def derivative(self, equation: str, unknown: str) -> int:
        """Returns the column index for derivatives of `equation` by `unknown`.

        Both arguments name unknowns of a single device: either a terminal such as
        ``M1.drain`` or an internal unknown such as ``M1:drain``.

        Args:
            equation: Qualified device terminal or internal equation name.
            unknown: Qualified device terminal or internal unknown name.

        Returns:
            Zero-based column index in `conductances` and `capacitances`.

        Raises:
            KeyError: If no single recorded device owns both unknowns, suggesting close
                names and how to record the device.
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
        """Returns the column index for variable `name` on device `instance`.

        Args:
            instance: Hierarchical path of the recorded primitive device.
            name: Internal model variable name on `instance`.

        Returns:
            Zero-based column index in the recorded `variables` array.

        Raises:
            KeyError: If the variable was not recorded, suggesting close names and how
                to enable recording.
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
        """Returns the operating region names defined by the model of `instance`.

        Args:
            instance: Hierarchical path of the recorded transistor instance.

        Returns:
            Ordered tuple of operating-region names indexed from 0.

        Raises:
            KeyError: If `instance` was not recorded or its device model defines no
                operating regions.
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
        """Returns whether `instance` or any summed terminal on it was recorded."""
        prefix = f"{instance}."
        return instance in self._columns().devices or any(
            summed_terminal.startswith(prefix)
            for summed_terminal in self.summed_terminals
        )

    def _columns(self) -> _Index:
        if not self._index:
            self._index.append(_Index.from_recording(self))
        return self._index[0]


def _unrecorded(
    what: str, names: Iterable[str], name: str, instance: str, *, recorded: bool
) -> Never:
    """Raises a `KeyError` describing an unrecorded lookup `what` on `instance`.

    When `recorded` is true, the error notes that `instance` was recorded but lacks
    `what`. Otherwise the error explains how to add `instance` to ``record``.

    Raises:
        KeyError: Always raised with close matches from `names` and recording details.
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


class RecordedValues(TypedDict):
    """Typed mapping of recorded device arrays shared across result dataclasses."""

    recording: Recording
    currents: Real
    charges: Real
    conductances: Real
    capacitances: Real
    variables: Real


def net_column(nets: tuple[str, ...], name: str) -> int:
    """Returns the column index of logic net `name` in `nets`.

    Args:
        nets: Ordered digital logic net names stored on the result.
        name: Target digital logic net name.

    Returns:
        Zero-based column index of `name` within `nets`.

    Raises:
        KeyError: If `name` is not in `nets`, suggesting close matches.
    """
    return _find(list(nets), name, "the circuit has no logic net")


def _find(names: list[str], name: str, missing: str) -> int:
    """Returns the index of `name` in `names`.

    Raises:
        KeyError: If `name` is absent, formatting `missing`, `name`, and close matches.
    """
    if name in names:
        return names.index(name)
    message = f"{missing} named {name!r}{close_names(names, name)}"
    raise KeyError(message)


@dataclass(frozen=True, slots=True)
class LogicTrace:
    """Piecewise-constant logic net level over time, starting at time zero.

    Examples:
        >>> trace = LogicTrace(np.array([0.0, 2e-9]), np.array([0.0, 1.0]))
        >>> trace.at(np.array([1e-9, 2e-9, 3e-9])).tolist()
        [0.0, 1.0, 1.0]
    """

    times: Real
    levels: Real

    def at(self, times: Real) -> Real:
        """Evaluates the logic level at each point in `times` via step interpolation.

        Args:
            times: One-dimensional array of evaluation times in seconds.

        Returns:
            One-dimensional array of step-interpolated logic levels (`0.0`, `1.0`, or
            NaN).
        """
        indices = np.searchsorted(self.times, times, side="right") - 1
        return self.levels[np.maximum(indices, 0)]


def _current(instance: str) -> str:
    return f"{instance}:current"


class Excursion(ir.FrozenModel):
    """Record of a logic device operated outside its characterized timing domain."""

    instance: str = Field(description="Instance path of the logic device.")
    bound: Literal["input transition", "load", "supply"] = Field(
        description="Characterized timing parameter that exceeded its domain."
    )
    minimum: float = Field(
        description="Lower bound of the characterized domain, in SI base units."
    )
    maximum: float = Field(
        description="Upper bound of the characterized domain, in SI base units."
    )
    value: float = Field(
        description="Observed value farthest outside the characterized domain."
    )
    time: float = Field(
        description="First simulation time of the excursion in seconds, or zero at "
        "an operating point."
    )
    count: int = Field(description="Number of evaluations observed outside the domain.")


@dataclass(frozen=True, slots=True)
class _Watched:
    """Base mixin for analysis results that track logic-device domain excursions."""

    excursions: tuple[Excursion, ...] | None = field(default=None, kw_only=True)
    """Logic-device timing-domain excursions observed during
    [`Simulation.run`][jolty.simulate.Simulation.run] and preserved in results
    archives, deduplicated per device and bound. Value is None for unwatched runs
    such as batch runs, ngspice runs, or nested sub-results."""


class _Unknowns[T]:
    """Mixin providing named access to circuit unknowns and recorded terminal currents.

    Lookups return a scalar for an operating point or a one-dimensional array across
    sweep values, frequencies, or time points.
    """

    __slots__ = ()

    unknowns: tuple[Unknown, ...]
    values: npt.NDArray[Any]
    recording: Recording
    currents: npt.NDArray[Any]

    def _read(self, values: npt.NDArray[Any]) -> T:
        """Converts raw scalar or array `values` into the concrete reading type `T`."""
        raise NotImplementedError

    def _unknown(self, name: str, nature: Nature | None = None) -> T:
        return self._read(self.values[..., column(self.unknowns, name, nature)])

    def voltage(self, net: str, reference: str | None = None) -> T:
        """Returns the voltage of `net` relative to `reference` or ground.

        Args:
            net: Hierarchical net path whose potential is read.
            reference: Optional reference net subtracted from `net`; ``None`` uses
                ground.

        Returns:
            Scalar voltage in volts at DC, or a one-dimensional array across points.
        """
        values = self.values[..., column(self.unknowns, net, "voltage")]
        if reference is not None:
            values = (
                values - self.values[..., column(self.unknowns, reference, "voltage")]
            )
        return self._read(values)

    def rise(self, net: str) -> T:
        """Returns the temperature rise of thermal `net` above ambient, in kelvins.

        Args:
            net: Hierarchical path of the solved thermal net.

        Returns:
            Temperature rise above ambient in kelvins as a scalar or array.
        """
        return self._unknown(net, "temperature")

    def state(self, name: str) -> T:
        """Returns the internal device state `name`, such as ``M1:state``.

        Args:
            name: Qualified internal state unknown name.

        Returns:
            Dimensionless internal state reading as a scalar or one-dimensional array.
        """
        return self._unknown(name, "state")

    def current(self, instance: str, terminal: str | None = None) -> T:
        """Returns the current into `terminal` of `instance`, or its branch current.

        When `terminal` is given, returns the recorded current entering that terminal
        on a device, component instance, or logic device (or heat flow in watts for a
        thermal terminal). When `terminal` is None, returns the branch current through
        a source or inductor.

        Args:
            instance: Hierarchical instance path of the device or component.
            terminal: Optional recorded terminal name; ``None`` reads branch current.

        Returns:
            Current in amperes (or thermal power in watts) as a scalar or array.

        Raises:
            KeyError: If the requested current is unavailable, suggesting close names
                and how to record terminals.
        """
        if terminal is None:
            return self._unknown(_current(instance))
        return self._read(
            self.currents[..., self.recording.terminal(instance, terminal)]
        )


class _Devices[T](_Unknowns[T]):
    """Mixin providing named access to unknowns and recorded device quantities.

    Subclasses extend `_Unknowns` with recorded terminal charges, small-signal
    derivatives, and internal device variables.
    """

    __slots__ = ()

    charges: npt.NDArray[Any]
    conductances: npt.NDArray[Any]
    capacitances: npt.NDArray[Any]
    variables: npt.NDArray[Any]

    def charge(self, instance: str, terminal: str) -> T:
        """Returns the recorded charge stored on `terminal` of `instance`.

        Args:
            instance: Hierarchical instance path of the recorded device or component.
            terminal: Terminal name on `instance`.

        Returns:
            Stored terminal charge in coulombs as a scalar or one-dimensional array.

        Raises:
            KeyError: If `terminal` of `instance` was not recorded.
        """
        return self._read(
            self.charges[..., self.recording.terminal(instance, terminal)]
        )

    def conductance(self, equation: str, unknown: str) -> T:
        """Returns the residual derivative of `equation` with respect to `unknown`.

        For two terminals such as ``M1.drain`` and ``M1.gate``, the conductance is the
        partial derivative of current entering the first terminal with respect to
        voltage at the second.

        Args:
            equation: Qualified terminal or internal equation name on the device.
            unknown: Qualified terminal or internal unknown name on the same device.

        Returns:
            Small-signal residual derivative in SI base units as a scalar or array.

        Raises:
            KeyError: If no recorded device owns both unknowns.
        """
        index = self.recording.derivative(equation, unknown)
        return self._read(self.conductances[..., index])

    def capacitance(self, equation: str, unknown: str) -> T:
        """Returns the charge derivative of `equation` with respect to `unknown`.

        For two terminals, the capacitance is the partial derivative of charge on the
        first terminal with respect to voltage at the second.

        Args:
            equation: Qualified terminal or internal charge equation on the device.
            unknown: Qualified terminal or internal unknown name on the same device.

        Returns:
            Small-signal charge derivative in SI base units as a scalar or array.

        Raises:
            KeyError: If no recorded device owns both unknowns.
        """
        index = self.recording.derivative(equation, unknown)
        return self._read(self.capacitances[..., index])

    def variable(self, instance: str, name: str) -> T:
        """Returns the recorded internal variable `name` of `instance`.

        Args:
            instance: Hierarchical instance path of the recorded primitive device.
            name: Model-defined internal variable identifier.

        Returns:
            Recorded internal variable value in SI base units as a scalar or array.

        Raises:
            KeyError: If variable `name` of `instance` was not recorded.
        """
        return self._read(self.variables[..., self.recording.variable(instance, name)])


@dataclass(frozen=True, slots=True)
class OperatingPointResult(_Devices[float], _Watched):
    """DC operating-point solution, solver strategy, and Newton iteration count."""

    unknowns: tuple[Unknown, ...]
    values: Real
    strategy: str
    """Convergence strategy that found the operating point: ``direct``,
    ``gmin_stepping``, ``junction_gmin_stepping``, ``source_stepping``,
    ``pseudo_transient``, or ``ngspice``."""
    iterations: int | None = None
    """Total Newton iterations across the converging strategy, earlier fallback
    strategies, and logic-settling re-solves, or None for ngspice runs that do not
    report iteration counts."""
    nets: tuple[str, ...] = ()
    """Logic net names ordered to match `logic`."""
    logic: Real = field(default_factory=_none)
    """Logic level of each digital net at the operating point."""
    recording: Recording = field(default_factory=Recording)
    """Schema of recorded device quantities and their column offsets."""
    currents: Real = field(default_factory=_none)
    """Current entering each recorded terminal, in amperes (or watts if thermal)."""
    charges: Real = field(default_factory=_none)
    """Charge stored on each recorded terminal, in coulombs."""
    conductances: Real = field(default_factory=_none)
    """Recorded residual derivatives with respect to device unknowns."""
    capacitances: Real = field(default_factory=_none)
    """Recorded charge derivatives with respect to device unknowns."""
    variables: Real = field(default_factory=_none)
    """Recorded internal device variable values, in SI base units."""

    def logic_level(self, net: str) -> float:
        """Returns the logic level of `net`: 0, 1, or NaN when unknown.

        Args:
            net: Digital logic net name.

        Returns:
            Settled logic level (`0.0`, `1.0`, or NaN when unknown).
        """
        return float(self.logic[net_column(self.nets, net)])

    def region(self, instance: str) -> str:
        """Returns the operating region of transistor `instance`, such as ``saturated``.

        The operating region is decoded from the device's recorded ``region`` variable
        using its `regions` table. Boundary margins are available through ``variable``,
        such as ``variable("M1", "saturation_margin")``.

        Args:
            instance: Hierarchical path of a recorded transistor instance.

        Returns:
            Operating region label (such as ``'saturated'`` or ``'cutoff'``).

        Raises:
            KeyError: If `instance` was not recorded or its model defines no operating
                regions.
        """
        regions = self.recording.regions(instance)
        return regions[int(self.variable(instance, _core.REGION))]

    @override
    def _read(self, values: Real) -> float:
        return float(values)


@dataclass(frozen=True, slots=True)
class OperatingPointsResult(_Watched):
    """Collection of all DC solutions found by a search, operating point first.

    Examples:
        >>> unknowns = (Unknown("q", "voltage"),)
        >>> found = OperatingPointsResult(
        ...     unknowns,
        ...     (
        ...         OperatingPointResult(unknowns, np.array([1.65]), "direct"),
        ...         OperatingPointResult(unknowns, np.array([3.3]), "direct"),
        ...     ),
        ... )
        >>> found.voltages("q").tolist()
        [1.65, 3.3]
    """

    unknowns: tuple[Unknown, ...]
    points: tuple[OperatingPointResult, ...]

    def voltages(self, net: str) -> Real:
        """Returns the voltage of `net` at each DC solution in discovery order.

        Args:
            net: Hierarchical electrical net name.

        Returns:
            One-dimensional array of `net` voltages in volts across DC solutions.
        """
        index = column(self.unknowns, net, "voltage")
        return np.array([float(point.values[index]) for point in self.points])


@dataclass(frozen=True, slots=True)
class DcSweepResult(_Devices[Real], _Watched):
    """Operating-point results across a DC sweep, with one row per swept value.

    When sweeping a device parameter, `parameter` names the swept parameter and `swept`
    holds its values in SI base units.
    """

    unknowns: tuple[Unknown, ...]
    source: str
    """Instance path of the swept source or device, such as ``V1``."""
    nature: Nature | None
    """Physical quantity of swept levels (voltage or current), or None for parameter
    sweeps."""
    swept: Real
    values: Real
    nets: tuple[str, ...] = ()
    """Logic net names ordered to match the columns of `logic`."""
    logic: Real = field(default_factory=_no_rows)
    """Logic net levels across the sweep, with one row per swept value."""
    recording: Recording = field(default_factory=Recording)
    """Schema of recorded device quantities and their column offsets."""
    currents: Real = field(default_factory=_no_rows)
    """Recorded terminal currents across the sweep, one row per swept value."""
    charges: Real = field(default_factory=_no_rows)
    """Recorded terminal charges across the sweep, one row per swept value."""
    conductances: Real = field(default_factory=_no_rows)
    """Recorded residual derivatives across the sweep, one row per swept value."""
    capacitances: Real = field(default_factory=_no_rows)
    """Recorded charge derivatives across the sweep, one row per swept value."""
    variables: Real = field(default_factory=_no_rows)
    """Recorded internal variable values across the sweep, one row per swept value."""
    parameter: str | None = None
    """Dot-separated name of the swept device parameter such as ``resistance``, or None
    when sweeping a source value."""
    strategies: tuple[str, ...] = ()
    """Solver strategy that found each sweep point, or an empty tuple for ngspice runs
    that do not report strategies."""
    iterations: tuple[int, ...] = ()
    """Newton iteration count at each sweep point using continuation, or an empty tuple
    for ngspice runs that do not report counts."""

    def logic_level(self, net: str) -> Real:
        """Returns the logic level of `net` across sweep points: 0, 1, or NaN.

        Args:
            net: Digital logic net name.

        Returns:
            One-dimensional array of logic levels (`0.0`, `1.0`, or NaN) across the
            sweep.
        """
        return self.logic[:, net_column(self.nets, net)]

    @override
    def _read(self, values: Real) -> Real:
        return values


@dataclass(frozen=True, slots=True)
class AcResult(_Unknowns[Complex], _Watched):
    """Small-signal AC response storing phasors per unit excitation per frequency."""

    unknowns: tuple[Unknown, ...]
    source: str
    """Instance path of the excited AC source, such as ``V1``."""
    frequencies: Real
    values: Complex
    operating_point: OperatingPointResult
    recording: Recording = field(default_factory=Recording)
    """Schema of recorded device quantities and their current column offsets."""
    currents: Complex = field(default_factory=_no_phasors)
    """Phasors of currents entering each recorded terminal, one row per frequency."""

    @override
    def _read(self, values: Complex) -> Complex:
        return values


@dataclass(frozen=True, slots=True)
class Work:
    """Solver work counters for Newton iterations and sparse matrix factorizations."""

    iterations: int
    """Total Newton iterations across the initial operating point (unless reused from
    cache) and all attempted time steps."""
    factorizations: int
    """Full sparse factorizations with pivot search, including the initial solve and
    recoveries after singular matrices."""
    refactorizations: int
    """Sparse factorizations reusing all pivot selections from the previous solve."""
    fallbacks: int
    """Sparse factorizations that reused pivots until encountering an undersized pivot
    and then switched to pivot search."""
    reuses: int
    """Cached sparse factorizations reused directly for an unchanged matrix."""


@dataclass(frozen=True, slots=True)
class Profile:
    """Wall-clock time breakdown of an analysis across solver phases, in seconds.

    Examples:
        >>> Profile(0.1, 0.4, 0.1, 0.2, 0.1, 0.05, 0.03, 0.02).total
        1.0
    """

    preparing: float
    """Time spent validating options, preparing recorded devices, and gathering source
    breakpoints."""
    evaluating: float
    """Time spent evaluating device models."""
    assembling: float
    """Time spent assembling Newton Jacobian matrices and right-hand sides and checking
    residual tolerances."""
    factoring: float
    """Time spent factoring sparse matrices."""
    solving: float
    """Time spent performing triangular solves with factored matrices."""
    step_control: float
    """Time spent estimating local truncation error, selecting step sizes, and locating
    step events."""
    recording: float
    """Time spent collecting recorded device data at accepted output points."""
    other: float
    """Remaining analysis time, including Newton step updates and convergence tests."""

    @property
    def total(self) -> float:
        """Total elapsed analysis time across all profiled phases, in seconds."""
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


@dataclass(frozen=True, slots=True, eq=False)
class ChargeLeak:
    """Converged equation residuals integrated over accepted transient steps.

    Entries follow the order of the result `unknowns` (Kundert and Clifford, "Achieving
    Accurate Results with a Circuit Simulator", IEEE CICC 1993). Units are coulombs
    (``A·s``) for electrical nodes and displacement equations, webers (``V·s``) for
    branch currents, joules (``W·s``) for thermal nets, and state units for device
    states.

    At every accepted step, Newton iteration stops with residual
    ``r_n = f(x_n, t_n) + dq/dt``. Integrating ``r_n`` with the step integration
    formula (backward Euler, trapezoidal, or Gear2) gives step increment ``Δℓ_n``. On
    any closed node or cutset with zero static current sum, ``q(T) - q(0)`` matches the
    sum of `signed` to floating-point rounding, and intermediate charge excursions stay
    within the sum of `absolute`.

    Examples:
        >>> leak = ChargeLeak(np.array([-1e-18]), np.array([2e-18]))
        >>> leak == ChargeLeak(np.array([-1e-18]), np.array([2e-18]))
        True
    """

    signed: Real
    """Signed integral ``∑ Δℓ_n`` for each unknown, representing the net charge, flux,
    or energy introduced by nonzero solver residuals."""
    absolute: Real
    """Cumulative absolute integral ``∑ |Δℓ_n|`` for each unknown without cancellation,
    bounding excursions at every intermediate accepted time."""

    __hash__: ClassVar[None] = None

    def __post_init__(self) -> None:
        """Validates that `signed` and `absolute` are 1D arrays of equal length.

        Raises:
            ValueError: If either array is not one-dimensional or their lengths differ.
        """
        if self.signed.ndim != 1 or self.absolute.ndim != 1:
            message = (
                "charge leak arrays are one-dimensional, not shapes "
                f"{self.signed.shape} and {self.absolute.shape}"
            )
            raise ValueError(message)
        if self.signed.shape != self.absolute.shape:
            message = (
                "signed and absolute charge leak arrays have equal length, "
                f"not {self.signed.shape[0]} and {self.absolute.shape[0]}"
            )
            raise ValueError(message)

    @override
    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other) or not isinstance(other, ChargeLeak):
            return NotImplemented
        return np.array_equal(self.signed, other.signed) and np.array_equal(
            self.absolute, other.absolute
        )


@dataclass(frozen=True, slots=True)
class Statistics:
    """Execution statistics and diagnostics for a transient analysis."""

    initial: str | None
    """Operating-point strategy such as ``direct``, or None when starting directly from
    initial conditions."""
    accepted: int
    rejected: int
    """Number of steps rejected due to truncation error or Newton non-convergence."""
    landed: int
    """Number of steps shortened to land on detector crossings, logic transitions, or
    device discontinuities."""
    iterations: int
    """Total Newton iterations across all accepted time steps."""
    work: Work | None = None
    """Solver work counters including the operating point and all periodic steady-state
    periods, or None for ngspice runs. Unlike simulation waveforms, work counts depend
    on prior cached operating points and solver pivots."""
    profile: Profile | None = None
    """Phase timing breakdown when profiling is enabled in options, or None."""
    charge_leak: ChargeLeak | None = None
    """Integrated converged residuals across accepted steps in `unknowns` order, or
    None for ngspice runs."""


@dataclass(frozen=True, slots=True)
class TransientResult(_Devices[Real], _Watched):
    """Time-domain transient waveforms, with one row per accepted time point."""

    unknowns: tuple[Unknown, ...]
    times: Real
    values: Real
    statistics: Statistics
    nets: tuple[str, ...] = ()
    """Logic net names ordered to match `logic`."""
    logic: tuple[LogicTrace, ...] = ()
    """Piecewise-constant logic traces for each digital net."""
    recording: Recording = field(default_factory=Recording)
    """Schema of recorded device quantities and their column offsets."""
    currents: Real = field(default_factory=_no_rows)
    """Recorded terminal currents at each time point, including integrated charge
    derivatives from the step formula."""
    charges: Real = field(default_factory=_no_rows)
    """Recorded terminal charges at each time point, one row per time point."""
    conductances: Real = field(default_factory=_no_rows)
    """Recorded residual derivatives at each time point, one row per time point."""
    capacitances: Real = field(default_factory=_no_rows)
    """Recorded charge derivatives at each time point, one row per time point."""
    variables: Real = field(default_factory=_no_rows)
    """Recorded internal variable values at each time point, one row per time point."""

    def logic_level(self, net: str) -> LogicTrace:
        """Returns the logic trace of `net` over simulation time.

        Args:
            net: Digital logic net name.

        Returns:
            Piecewise-constant `LogicTrace` recorded for `net`.
        """
        return self.logic[net_column(self.nets, net)]

    def charge_leak(self, unknown: str, *, absolute: bool = False) -> float:
        """Returns the integrated residual of `unknown` over the transient.

        Returns the signed net integral by default, or the cumulative absolute integral
        when `absolute` is true. Units are coulombs for electrical nodes, webers
        (``V·s``) for branch currents, joules for thermal nets, and state units for
        device states.

        Args:
            unknown: Circuit unknown name indexing `statistics.charge_leak`.
            absolute: When ``True``, returns cumulative absolute residual integral.

        Returns:
            Integrated equation residual in coulombs, webers, joules, or state units.

        Raises:
            KeyError: If `unknown` is absent from the circuit or the result contains no
                charge leak data (such as an ngspice run).
        """
        leak = self.statistics.charge_leak
        if leak is None:
            message = "the result holds no charge leak"
            raise KeyError(message)
        values = leak.absolute if absolute else leak.signed
        return float(values[column(self.unknowns, unknown)])

    @override
    def _read(self, values: Real) -> Real:
        return values


@dataclass(frozen=True, slots=True)
class PeriodicSteadyStateResult(TransientResult):
    """Waveforms over one steady-state period starting from the periodic orbit.

    Sample times span from zero to `period`. Statistics record step counts for the
    final steady-state period and total work across all shooting periods, the operating
    point, and oscillator settling.
    """

    period: float = field(kw_only=True)
    """Steady-state period in seconds, either specified or solved for an oscillator."""
    periods: int = field(kw_only=True)
    """Total shooting periods simulated to converge, with the final period returned."""


@dataclass(frozen=True, slots=True)
class StabilityResult(_Watched):
    """Loop gain measured through a stability probe across frequencies.

    Negative feedback yields positive loop gain at low frequencies. Stability measures
    `phase_margin`, `gain_margin`, and `crossover` evaluate proximity to -1.
    """

    unknowns: tuple[Unknown, ...]
    probe: str
    """Instance path of the stability probe breaking the loop, such as ``VP``."""
    frequencies: Real
    gains: Complex
    """Complex loop gain at each frequency in `frequencies`."""
    operating_point: OperatingPointResult


@dataclass(frozen=True, slots=True)
class NoiseSource:
    """Identifier for a single noise mechanism within a device instance."""

    instance: str
    """Instance path of the device, such as ``X1/Q1``."""
    name: str
    """Noise mechanism name, such as ``thermal``, ``collector shot``, or
    ``flicker``."""


@dataclass(frozen=True, slots=True)
class NoiseInput:
    """Input source metadata and transfer gains used for input-referred noise."""

    source: str
    """Instance path of the input source, such as ``V1``."""
    nature: Nature
    """Physical quantity of the input source level."""
    gains: Complex
    """Complex transfer gain from the input source to the output at each frequency."""


@dataclass(frozen=True, slots=True)
class NoiseResult(_Watched):
    """Output noise spectral density and per-source contributions across frequencies.

    The `densities` array stores one-sided output noise power spectral densities in
    V²/Hz for voltage outputs or A²/Hz for current outputs, and `contributions` stores
    one row per entry in `sources`. When an input source is configured, ``referred``
    divides output densities by squared transfer gain to yield equivalent input noise.
    The ``rms`` method integrates noise density over a frequency band using SPICE
    power-law interpolation between frequencies.

    Examples:
        >>> point = OperatingPointResult(
        ...     (Unknown("out", "voltage"),), np.array([2.0]), ""
        ... )
        >>> spectrum = NoiseResult(
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
    """Name of the output unknown."""
    reference: str | None
    """Reference net for a differential output voltage, or None for ground."""
    nature: Nature
    """Physical quantity of the output unknown."""
    frequencies: Real
    densities: Real
    sources: tuple[NoiseSource, ...]
    contributions: Real
    input: NoiseInput | None
    operating_point: OperatingPointResult

    def referred(self) -> Real:
        """Returns input-referred noise power spectral density at each frequency.

        Values equal output noise density divided by squared transfer gain, in V²/Hz or
        A²/Hz matching the input source quantity.

        Returns:
            One-dimensional array of input-referred noise densities across frequency.

        Raises:
            ValueError: If the noise analysis specified no input source.
        """
        if self.input is None:
            message = "the noise analysis named no source to refer its noise to"
            raise ValueError(message)
        return _core.referred(
            np.asarray(self.densities, dtype=np.float64),
            np.asarray(self.input.gains, dtype=np.complex128),
        )

    def contribution(self, instance: str, name: str | None = None) -> Real:
        """Returns noise power spectral density contributed by `instance`.

        When `name` is provided, only that noise mechanism is returned; otherwise all
        noise mechanisms of `instance` are summed in the unit of `densities`.

        Args:
            instance: Hierarchical path of the contributing device instance.
            name: Optional noise mechanism name; ``None`` sums all mechanisms.

        Returns:
            One-dimensional array of output noise spectral densities across
            `frequencies`.

        Raises:
            KeyError: If `instance` has no matching noise source.
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
        """Returns the interpolated noise spectral density at frequency `at`, in hertz.

        Interpolation uses a power law between neighboring frequency points.

        Args:
            at: Query frequency in hertz within the simulated span.
            referred: When ``True``, interpolates input-referred noise density.

        Returns:
            Power-law interpolated noise power spectral density in V²/Hz or A²/Hz.

        Raises:
            ValueError: If `at` lies outside `frequencies`, or if `referred` is true
                with no input source configured.
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
        """Returns integrated RMS noise from `start` to `stop` hertz.

        When `start` or `stop` is omitted, integration covers the full frequency span
        using power-law interpolation and returns the square root of integrated noise
        power in volts or amperes.

        Args:
            start: Optional lower integration frequency in hertz.
            stop: Optional upper integration frequency in hertz.
            referred: When ``True``, integrates input-referred noise density.

        Returns:
            Integrated RMS noise amplitude in volts or amperes.

        Raises:
            ValueError: If the frequency band is empty or out of bounds, or if
                `referred` is true with no input source configured.
        """
        densities = self.referred() if referred else self.densities
        return math.sqrt(integrate(self.frequencies, densities, start, stop))


type DifferentiatedResult = (
    OperatingPointResult
    | DcSweepResult
    | AcResult
    | StabilityResult
    | NoiseResult
    | TransientResult
)
"""Analysis result types supported by sensitivity and mismatch analyses."""

DIFFERENTIATED_TYPES: Final = (
    OperatingPointResult,
    DcSweepResult,
    AcResult,
    StabilityResult,
    NoiseResult,
    TransientResult,
)
"""Runtime tuple of types matching `DifferentiatedResult`."""


@dataclass(frozen=True, slots=True)
class Parameter:
    """Scalar numeric parameter of a device instance."""

    instance: str
    """Hierarchical instance path of the device, such as ``X1/R1``."""
    name: str
    """Dot-separated parameter path such as ``resistance`` or ``waveform.dc``, using
    zero-based indices for list entries."""
    value: float
    """Nominal parameter value, in SI base units."""


@dataclass(frozen=True, slots=True)
class SensitivityResult(_Watched):
    """First-order derivatives of selected outputs with respect to circuit parameters.

    At an operating point, `derivatives` has one row per output and one column per
    parameter in SI base units. For DC sweeps, AC responses, and transients, a leading
    axis indexes each point of `result`. AC sensitivities differentiate complex
    phasors, while periodic steady-state sensitivities evaluate at each point's
    fraction of the period. For oscillator periodic steady states, `period_derivatives`
    stores derivatives of the oscillation period with respect to each parameter.

    Examples:
        >>> point = OperatingPointResult(
        ...     (Unknown("out", "voltage"),), np.array([2.0]), ""
        ... )
        >>> sensitivities = SensitivityResult(
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
    """Differentiated unknown names matching the rows of `derivatives`."""
    parameters: tuple[Parameter, ...]
    """Differentiated parameters matching the columns of `derivatives`."""
    derivatives: Real | Complex
    result: DifferentiatedResult
    """Underlying analysis result that was differentiated."""
    period_derivatives: Real | None = None
    """Derivatives of an oscillator period with respect to each parameter in order, or
    None for non-oscillator analyses."""

    def derivative(
        self, output: str, instance: str, parameter: str
    ) -> float | Real | Complex:
        """Returns the derivative of `output` with respect to `parameter` on `instance`.

        Returns a scalar at an operating point, or an array indexed by points of
        `result` for multi-point analyses.

        Args:
            output: Differentiated output unknown name.
            instance: Hierarchical path of the target device instance.
            parameter: Dotted parameter path on `instance`.

        Returns:
            Scalar sensitivity at DC, or a one-dimensional array across `self.result`.

        Raises:
            KeyError: If `output` or `(instance, parameter)` is not in the analysis.
        """
        found = self.derivatives[
            ..., self._row(output), self._column(instance, parameter)
        ]
        return float(found) if found.ndim == 0 else found

    def period_derivative(self, instance: str, parameter: str) -> float:
        """Returns the derivative of an oscillator period with respect to `parameter`.

        The parameter belongs to `instance`, and the derivative is in seconds per
        parameter SI unit.

        Args:
            instance: Hierarchical path of the target device instance.
            parameter: Dotted parameter path on `instance`.

        Returns:
            Oscillator period derivative in seconds per SI parameter unit.

        Raises:
            KeyError: If the parameter is absent or the analysis does not differentiate
                an oscillator period.
        """
        if self.period_derivatives is None:
            message = "the analysis differentiates no oscillator's period"
            raise KeyError(message)
        return float(self.period_derivatives[self._column(instance, parameter)])

    def _column(self, instance: str, parameter: str) -> int:
        labels = [f"{found.instance} {found.name}" for found in self.parameters]
        missing = "the analysis has no parameter"
        return _find(labels, f"{instance} {parameter}", missing)

    def gradient(self, output: str) -> Real | Complex:
        """Returns derivatives of `output` with respect to all parameters in order.

        Returns a 1D array at an operating point, or a 2D array with a leading point
        axis matching `result` for multi-point analyses.

        Args:
            output: Differentiated output unknown name.

        Returns:
            One-dimensional (DC) or two-dimensional (multi-point) derivative array.
        """
        return self.derivatives[..., self._row(output), :]

    def _row(self, output: str) -> int:
        return _find(list(self.outputs), output, "the analysis has no output")


type PhasorPart = Literal["magnitude", "phase"]
"""Component of a complex AC phasor analyzed for mismatch: magnitude or phase in
radians."""


@dataclass(frozen=True, slots=True)
class MismatchResult(_Watched):
    """First-order statistical spread of outputs across independent random variables.

    The `derivatives` array stores partial derivatives of each output with respect to
    `random_variables`, using the same layout as `SensitivityResult`. At an operating
    point, rows index outputs and columns index random variables; multi-point analyses
    add a leading point axis, with complex phasor derivatives for AC responses.

    Each random variable contributes its derivative multiplied by its standard
    deviation. Independent random variables combine in quadrature: first-order output
    standard deviation equals the root sum of squared contributions, and each variance
    share equals its squared contribution divided by total variance. For AC responses,
    `part` selects first-order spread of either phasor magnitude or phase.

    Examples:
        >>> point = OperatingPointResult(
        ...     (Unknown("out", "voltage"),), np.array([0.5]), ""
        ... )
        >>> moves = {"R2": {"resistance": 1e3}}
        >>> mismatch = MismatchResult(
        ...     point.unknowns,
        ...     ("out",),
        ...     (
        ...         ir.RandomVariable(
        ...             instance="R2",
        ...             name="resistance",
        ...             mean=0,
        ...             standard_deviation=0.01,
        ...             derivatives=moves,
        ...         ),
        ...     ),
        ...     np.array([[2.5e-4]]),
        ...     point,
        ... )
        >>> mismatch.standard_deviation("out")
        2.5e-06
    """

    unknowns: tuple[Unknown, ...]
    outputs: tuple[str, ...]
    """Analyzed output unknown names matching the rows of `derivatives`."""
    random_variables: tuple[ir.RandomVariable, ...]
    """Random variables matching the columns of `derivatives`: per-part tolerances,
    shared lot tolerances, and explicit analysis random variables."""
    derivatives: Real | Complex
    result: DifferentiatedResult
    """Underlying analysis result that was differentiated."""
    period_derivatives: Real | None = None
    """Derivatives of an oscillator period with respect to each random variable in
    order, or None for non-oscillator analyses."""

    def contributions(self, output: str, part: PhasorPart | None = None) -> Real:
        """Returns each random variable's first-order contribution to `output` in order.

        Returns a 1D array at an operating point, or a 2D array indexed by points of
        `result` otherwise. For AC responses, `part` selects phasor magnitude or phase.

        Args:
            output: Analyzed output unknown name.
            part: AC phasor component (`'magnitude'` or `'phase'`); ``None`` otherwise.

        Returns:
            Signed one-sigma contribution array per random variable.

        Raises:
            KeyError: If `output` is not in the analysis.
            ValueError: If `part` is supplied for a real-valued analysis or omitted for
                an AC response.
        """
        deviations = np.array(
            [variable.standard_deviation for variable in self.random_variables]
        )
        terms = self.derivatives[..., self._row(output), :] * deviations
        if not isinstance(self.result, AcResult | StabilityResult):
            if part is not None:
                message = f"a mismatch of real values spreads no {part}"
                raise ValueError(message)
            return terms
        if part is None:
            message = "a response's mismatch spreads its magnitude or its phase"
            raise ValueError(message)
        phasors = (
            self.result.gains
            if isinstance(self.result, StabilityResult)
            else self.result.values[:, column(self.unknowns, output)]
        )
        along = np.conj(phasors)[:, None] * terms
        magnitudes = np.abs(phasors)[:, None]
        if part == "magnitude":
            return along.real / magnitudes
        return along.imag / magnitudes**2

    def standard_deviation(
        self, output: str, part: PhasorPart | None = None
    ) -> float | Real:
        """Returns the first-order standard deviation of `output`.

        Returns a scalar at an operating point, or an array indexed by points of
        `result` otherwise. For AC responses, `part` selects phasor magnitude or phase.

        Args:
            output: Analyzed output unknown name.
            part: AC phasor component (`'magnitude'` or `'phase'`); ``None`` otherwise.

        Returns:
            Root-sum-square one-sigma spread as a scalar (DC) or one-dimensional array.

        Raises:
            KeyError: If `output` is not in the analysis.
            ValueError: If `part` is supplied for a real-valued analysis or omitted for
                an AC response.
        """
        found = np.sqrt(np.sum(self.contributions(output, part) ** 2, axis=-1))
        return float(found) if found.ndim == 0 else found

    def period_standard_deviation(self) -> float:
        """Returns the first-order oscillator period standard deviation, in seconds.

        Returns:
            First-order one-sigma standard deviation of oscillator period in seconds.

        Raises:
            KeyError: If the analysis does not differentiate an oscillator period.
        """
        if self.period_derivatives is None:
            message = "the analysis differentiates no oscillator's period"
            raise KeyError(message)
        deviations = np.array(
            [variable.standard_deviation for variable in self.random_variables]
        )
        return float(np.sqrt(np.sum((self.period_derivatives * deviations) ** 2)))

    def shares(self, output: str, part: PhasorPart | None = None) -> Real:
        """Returns each random variable's share of the variance of `output` in order.

        Returns a 1D array at an operating point, or a 2D array indexed by points of
        `result` otherwise. For AC responses, `part` selects phasor magnitude or phase.
        Points with zero variance return zero shares.

        Args:
            output: Analyzed output unknown name.
            part: AC phasor component (`'magnitude'` or `'phase'`); ``None`` otherwise.

        Returns:
            Fractional variance shares summing to 1 where total variance is positive.

        Raises:
            KeyError: If `output` is not in the analysis.
            ValueError: If `part` is supplied for a real-valued analysis or omitted for
                an AC response.
        """
        squares = self.contributions(output, part) ** 2
        variance = np.sum(squares, axis=-1, keepdims=True)
        return np.divide(
            squares, variance, out=np.zeros_like(squares), where=variance > 0
        )

    def _row(self, output: str) -> int:
        return _find(list(self.outputs), output, "the analysis has no output")


@dataclass(frozen=True, slots=True)
class TransferFunctionResult(_Watched):
    """DC small-signal transfer function from an input source to an output unknown.

    The `gain` field gives small-signal output per unit input level in volts or
    amperes. The `input_resistance` field is the resistance seen by the source in ohms.
    The `output_resistance` field is the resistance seen at the output port with the
    source fixed, or infinity when no current flows.
    """

    unknowns: tuple[Unknown, ...]
    source: str
    """Instance path of the input source, such as ``V1``."""
    input: Nature
    """Physical quantity of the input source: ``voltage`` or ``current``."""
    output: str
    """Output unknown name, such as net ``out`` or a voltage-source current."""
    reference: str | None
    """Reference net for a differential output voltage, or None for ground."""
    gain: float
    input_resistance: float
    output_resistance: float
    operating_point: OperatingPointResult


_UNSPECTRAL: Final[dict[str, str]] = {
    "empty": "a spectrum of no frequencies has no density",
    "mismatch": "a spectrum holds a density at each frequency",
}
"""Mapping from engine spectral error reason codes to user-facing error messages."""


def integrate(
    frequencies: Real,
    densities: Real,
    start: float | None = None,
    stop: float | None = None,
) -> float:
    """Integrates noise power `densities` from `start` to `stop` hertz.

    When `start` or `stop` is omitted, integration spans all of `frequencies` using
    SPICE power-law interpolation between adjacent points.

    Args:
        frequencies: Strictly positive one-dimensional frequency array in hertz.
        densities: One-sided power spectral density array matching `frequencies`.
        start: Optional lower band limit in hertz, defaulting to `frequencies[0]`.
        stop: Optional upper band limit in hertz, defaulting to `frequencies[-1]`.

    Returns:
        Integrated total noise power over the band in squared output units.

    Raises:
        ValueError: If the integration band is empty or outside `frequencies`.

    Examples:
        >>> integrate(np.array([1.0, 10.0]), np.array([1.0, 0.1]))
        2.3025850929940463
        >>> round(integrate(np.array([1.0, 100.0]), np.array([2.0, 2.0]), 10.0))
        180
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
class PoleZeroResult(_Watched):
    """Poles of the linearized circuit and optional transfer-function zeros.

    The circuit is linearized at `operating_point`, and zeros are computed when both
    `source` and `output` are specified. Roots are complex frequencies ``s`` in radians
    per second sorted by increasing magnitude, with the upper-half-plane member of each
    conjugate pair listed first. Poles with positive real parts grow in time. The
    ``natural_frequencies`` and ``dampings`` functions convert roots to natural
    frequencies and damping ratios.

    Examples:
        >>> point = OperatingPointResult(
        ...     (Unknown("out", "voltage"),), np.array([0.0]), ""
        ... )
        >>> found = PoleZeroResult(
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
    """Transfer-function zeros when both `source` and `output` are given, or None."""
    source: str | None
    """Instance path of the transfer-function input source, or None."""
    output: str | None
    """Name of the transfer-function output unknown, or None."""
    reference: str | None
    """Reference net for a differential output voltage, or None for ground."""
    operating_point: OperatingPointResult


def natural_frequencies(roots: Complex) -> Real:
    """Computes natural frequencies of `roots` in hertz as magnitude divided by 2π.

    Args:
        roots: One-dimensional complex array of s-plane roots in rad/s.

    Returns:
        One-dimensional real array of natural frequencies `|s| / (2π)` in hertz.
    """
    return _core.natural_frequencies(np.asarray(roots, dtype=np.complex128))


def dampings(roots: Complex) -> Real:
    """Computes damping ratios of `roots` as negated real part divided by magnitude.

    The damping ratio is 1 for left-half-plane real roots, between 0 and 1 for
    underdamped conjugate pairs, negative in the right half-plane, and NaN at the
    origin.

    Args:
        roots: One-dimensional complex array of s-plane roots in rad/s.

    Returns:
        One-dimensional real array of damping ratios `-Re(s) / |s|` (NaN at origin).
    """
    return _core.dampings(np.asarray(roots, dtype=np.complex128))


type AnalysisResult = (
    OperatingPointResult
    | OperatingPointsResult
    | DcSweepResult
    | AcResult
    | TransientResult
    | PeriodicSteadyStateResult
    | SensitivityResult
    | MismatchResult
    | StabilityResult
    | NoiseResult
    | PoleZeroResult
    | TransferFunctionResult
)
"""Union of all analysis result types."""

_RESULT_KIND_MAP: dict[type[object], str] = {
    OperatingPointResult: "an operating point",
    DcSweepResult: "a DC sweep",
    AcResult: "a small-signal response",
    TransientResult: "a transient",
    PeriodicSteadyStateResult: "a periodic steady state",
    SensitivityResult: "sensitivities",
    MismatchResult: "a mismatch",
    StabilityResult: "a loop gain",
    NoiseResult: "a noise spectrum",
    PoleZeroResult: "poles and zeros",
    OperatingPointsResult: "DC solutions",
    TransferFunctionResult: "a transfer function",
}

RESULT_KINDS: Final[Mapping[type[object], str]] = MappingProxyType(_RESULT_KIND_MAP)
"""Human-readable label for each analysis result type."""


def result_kind(result: AnalysisResult) -> str:
    """Returns the human-readable analysis kind label for `result`.

    Args:
        result: Completed `AnalysisResult` instance.

    Returns:
        Human-readable phrase naming the analysis category of `result`.
    """
    return RESULT_KINDS.get(type(result), type(result).__name__)


def axis_of(
    result: DcSweepResult | AcResult | StabilityResult | NoiseResult | TransientResult,
) -> Real:
    """Returns the sweep values, frequencies, or time axis array of `result`.

    Args:
        result: Multi-point `DcSweepResult`, `AcResult`, `StabilityResult`,
            `NoiseResult`, or `TransientResult`.

    Returns:
        One-dimensional coordinate array (`swept`, `frequencies`, or `times`).
    """
    match result:
        case DcSweepResult():
            return result.swept
        case AcResult() | StabilityResult() | NoiseResult():
            return result.frequencies
        case TransientResult():
            return result.times


def nets_of(result: AnalysisResult) -> tuple[str, ...]:
    """Returns the logic net names recorded by `result`, or an empty tuple.

    Args:
        result: Completed `AnalysisResult` instance.

    Returns:
        Tuple of recorded digital logic net names, or ``()`` when unsupported.
    """
    if isinstance(result, OperatingPointResult | DcSweepResult | TransientResult):
        return result.nets
    return ()


@dataclass(frozen=True, slots=True)
class Traces:
    """Selected unknown values recorded across runs at every point of one analysis.

    The `values` array has shape ``(runs, points, unknowns)`` and stores NaN for runs
    where the analysis failed. Axis points correspond to an operating point, DC sweep
    levels, AC frequencies (with complex phasors), or requested transient times
    linearly interpolated between accepted steps.
    """

    analysis: str
    """Name of the traced analysis."""
    along: Literal["point", "swept", "frequency", "time"]
    """Independent axis kind for the traced points."""
    axis: Real
    """Coordinate array along the sweep, frequency, or time axis; empty for DC."""
    unknowns: tuple[Unknown, ...]
    """Traced unknowns ordered to match the last axis of `values`."""
    values: Real | Complex
    source: str | None = None
    """Swept or excited instance path for a DC sweep or AC response, or None."""
    nature: Nature | None = None
    """Physical quantity of sweep levels (voltage or current), or None for parameter
    sweeps."""
    parameter: str | None = None
    """Swept parameter name for a parameter DC sweep, or None."""

    def values_of(self, name: str) -> Real | Complex:
        """Returns traced values for unknown `name` shaped ``(runs, points)``.

        Args:
            name: Traced circuit unknown name.

        Returns:
            Two-dimensional array of traced values shaped ``(runs, points)``.

        Raises:
            KeyError: If unknown `name` was not traced by the batch.
        """
        names = [unknown.name for unknown in self.unknowns]
        return self.values[:, :, _find(names, name, "the batch traced no unknown")]


@dataclass(frozen=True, slots=True)
class Run:
    """Per-run results, errors, and elaboration problems passed to a batch callback."""

    index: int
    """Zero-based index of the run within the batch."""
    results: dict[str, AnalysisResult]
    """Mapping from analysis name to result for each analysis that completed."""
    errors: dict[str, str]
    """Mapping from analysis name to error message for each failed analysis."""
    problems: tuple[str, ...] = ()
    """Elaboration errors that prevented analyses from running on this run."""


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Tabular results across multiple runs of one circuit, with one row per run.

    Each run applies `values` for `parameters` (such as ``X1/R1 resistance`` or
    ``temperature`` in kelvins) and executes `analyses` in order, recording completion
    status in `ended`. The `measured` array stores evaluated `measures` in `units`,
    using NaN when evaluation fails. For failed cells, `causes` indexes deduplicated
    failure messages in `reasons` in row-major first-appearance order, and stores -1
    for valid numbers.

    The `traces` tuple stores selected unknown waveforms across runs. Each row depends
    only on its run index regardless of thread scheduling, so slicing with `rows` and
    concatenating with `joined` match running those subsets directly.
    """

    analyses: tuple[str, ...]
    parameters: tuple[str, ...]
    values: Real
    ended: _Flags
    measures: tuple[str, ...]
    """Evaluated measure names as defined by requirements, such as ``out in op``."""
    units: tuple[Dimension | Plain | None, ...]
    """Unit for each measure column (``dB``, ``°``, ``%``, ``""`` for plain ratios, or
    None if never evaluated)."""
    measured: Real
    causes: _Causes
    reasons: tuple[str, ...]
    problems: dict[int, tuple[str, ...]] = field(default_factory=dict)
    """Elaboration problems per failed run, keyed by zero-based run index."""
    errors: dict[int, dict[str, str]] = field(default_factory=dict)
    """Failed analysis error messages keyed by run index and analysis name."""
    traces: tuple[Traces, ...] = ()

    def __post_init__(self) -> None:
        """Deduplicates `reasons` and renumbers `causes` in first-appearance order.

        Raises:
            ValueError: If any entry in `causes` references an out-of-range reason.
        """
        causes, reasons = _canonical(self.causes, self.reasons)
        object.__setattr__(self, "causes", causes)
        object.__setattr__(self, "reasons", reasons)

    @property
    def runs(self) -> int:
        """Total number of runs stored in the batch result."""
        return int(self.values.shape[0])

    @property
    def failed(self) -> npt.NDArray[np.bool_]:
        """Boolean array marking runs with elaboration problems or failed analyses."""
        return ~self.ended.all(axis=1)

    def failure(self, run: int) -> str | None:
        """Returns the formatted failure summary for `run`, or None if `run` passed.

        Args:
            run: Zero-based batch run index.

        Returns:
            Formatted elaboration or analysis failure text, or ``None`` when passing.
        """
        if run in self.problems:
            return "\n\n".join(self.problems[run])
        if run in self.errors:
            return "; ".join(
                f"{name}: {error}" for name, error in self.errors[run].items()
            )
        return None

    def reason(self, run: int, measure: int) -> str | None:
        """Returns the failure reason for `measure` on `run`, or None if valid.

        Args:
            run: Zero-based batch run index.
            measure: Zero-based measure column index.

        Returns:
            Deduplicated failure message for the cell, or ``None`` when numeric.
        """
        cause = int(self.causes[run, measure])
        return None if cause < 0 else self.reasons[cause]

    def parameter(self, name: str) -> Real:
        """Returns the array of values for parameter `name` across all runs.

        Args:
            name: Parameter column label from `self.parameters`.

        Returns:
            One-dimensional array of parameter values across all runs in SI base units.

        Raises:
            KeyError: If `name` is not among the varied `parameters`.
        """
        found = _find(list(self.parameters), name, "the variants vary no parameter")
        return self.values[:, found]

    @staticmethod
    def joined(parts: Sequence[BatchResult]) -> BatchResult:
        """Concatenates runs from `parts` in order into a single `BatchResult`.

        Concatenation acts as the inverse of `rows`, preserving run order, reconciling
        measure units, and merging deduplicated failure reasons across all parts.

        Args:
            parts: Non-empty sequence of compatible `BatchResult` slices.

        Returns:
            Combined `BatchResult` concatenating all runs and deduplicating reasons.

        Raises:
            ValueError: If `parts` is empty, if parts differ in analyses, parameters,
                measures, or trace layout, or if parts disagree on a measure's unit.
        """
        if not parts:
            message = "a batch joins one part or more"
            raise ValueError(message)
        first, *rest = parts
        for place, part in enumerate(rest, start=1):
            if (part.analyses, part.parameters, part.measures) != (
                first.analyses,
                first.parameters,
                first.measures,
            ) or not _alike(first.traces, part.traces):
                message = (
                    f"part {place} ran other analyses, varied other numbers, read "
                    "other measures or traced otherwise than part 0"
                )
                raise ValueError(message)
        units = tuple(
            _unit_of(measure, [part.units[column] for part in parts])
            for column, measure in enumerate(first.measures)
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
        traces = tuple(
            replace(
                each,
                values=np.concatenate([part.traces[index].values for part in parts]),
            )
            for index, each in enumerate(first.traces)
        )
        return BatchResult(
            first.analyses,
            first.parameters,
            np.concatenate([part.values for part in parts]),
            np.concatenate([part.ended for part in parts]),
            first.measures,
            units,
            np.concatenate([part.measured for part in parts]),
            np.concatenate(causes),
            tuple(texts),
            problems,
            errors,
            traces,
        )

    def rows(self, start: int, stop: int) -> BatchResult:
        """Returns a new `BatchResult` containing runs from `start` up to `stop`.

        Args:
            start: Inclusive zero-based starting run index.
            stop: Exclusive zero-based ending run index.

        Returns:
            Sliced `BatchResult` containing runs in ``range(start, stop)``.
        """

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
            self.measures,
            self.units,
            self.measured[start:stop],
            self.causes[start:stop],
            self.reasons,
            kept(self.problems),
            kept(self.errors),
            tuple(
                replace(traces, values=traces.values[start:stop])
                for traces in self.traces
            ),
        )


def _canonical(
    causes: npt.NDArray[np.int32], reasons: tuple[str, ...]
) -> tuple[npt.NDArray[np.int32], tuple[str, ...]]:
    """Deduplicates `reasons` in first-appearance order and remaps `causes`.

    Returns `(causes, reasons)` unchanged when already canonical.

    Raises:
        ValueError: If any entry in `causes` indexes outside `reasons`.
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
    # A sentinel cause of -1 indexes the final table entry, which is also -1.
    return table[causes], tuple(texts)


def _alike(traces: tuple[Traces, ...], other: tuple[Traces, ...]) -> bool:
    """Returns whether `traces` and `other` share identical metadata and axis arrays."""
    return len(traces) == len(other) and all(
        (one.analysis, one.along, one.unknowns, one.source, one.nature, one.parameter)
        == (
            two.analysis,
            two.along,
            two.unknowns,
            two.source,
            two.nature,
            two.parameter,
        )
        and np.array_equal(one.axis, two.axis)
        for one, two in zip(traces, other, strict=True)
    )


def _unit_of(
    measure: str, units: Sequence[Dimension | Plain | None]
) -> Dimension | Plain | None:
    """Returns the unique non-None unit in `units` for `measure`, or None.

    Raises:
        ValueError: If parts report conflicting units for `measure`.
    """
    read = list(dict.fromkeys(unit for unit in units if unit is not None))
    if len(read) > 1:
        listed = ", ".join(repr(str(unit)) for unit in read)
        message = f"the parts read {measure} in different units: {listed}"
        raise ValueError(message)
    return read[0] if read else None
