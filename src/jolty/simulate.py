# Copyright 2026 Maxence Faldor
"""Run analyses on a circuit with the Rust engine.

A [`Simulation`][jolty.simulate.Simulation] elaborates a circuit once, its devices at
one temperature, 27 °C unless given, and runs any number of analyses on it, or a batch
of runs of them, each at the values of one of its variants, in one call that spreads
the runs across the engine's threads. A circuit
[`Elaborated`][jolty.simulate.Elaborated] once is shared by simulations of it at any
parameters: each makes its devices again from its parameters, and none elaborates the
circuit anew.

Examples:
    >>> from jolty import Builder, Role, component, devices
    >>> from jolty.plan import OperatingPoint
    >>> from jolty.units import V, kohm
    >>> @component
    ... def divider(c: Builder) -> None:
    ...     ground = c.port("ground", Role.GROUND)
    ...     source = c.place(devices.voltage_source(5 * V), name="V1")
    ...     upper = c.place(devices.resistor(1 * kohm), name="R1")
    ...     lower = c.place(devices.resistor(1 * kohm), name="R2")
    ...     c.connect(source.positive, upper.a)
    ...     c.connect(upper.b, lower.a, name="out")
    ...     c.connect(lower.b, source.negative, ground)
    >>> Simulation(divider()).run(OperatingPoint()).voltage("out")
    2.5
"""

import contextlib
import contextvars
import functools
import json
import math
import threading
import time
from collections.abc import (
    Callable,
    Collection,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Final, Literal, overload

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from jolty import _core, checks, foundry, ir, results, specs
from jolty.builder import Component, circuit
from jolty.devices import Device
from jolty.errors import ElaborationError, JoltyError, SimulationError
from jolty.plan import (
    Ac,
    Analysis,
    Conditions,
    DcSweep,
    Explicit,
    Grid,
    Mismatch,
    MonteCarlo,
    Noise,
    OperatingPoint,
    OperatingPoints,
    Options,
    PeriodicSteadyState,
    Plan,
    PoleZero,
    Product,
    Recorded,
    Sensitivity,
    Stability,
    TransferFunction,
    Transient,
    Variants,
    differentiated,
)
from jolty.settings import Settings
from jolty.units import (
    CURRENT,
    DIMENSIONLESS,
    FREQUENCY,
    TEMPERATURE,
    TIME,
    VOLTAGE,
    ZERO_CELSIUS,
    Dimension,
    Quantity,
)


def kelvins(temperature: Quantity | None) -> float:
    """`temperature` in kelvins; 27 °C when it is None.

    Raises:
        DimensionError: When the temperature is not one.
    """
    if temperature is None:
        return _core.REFERENCE_TEMPERATURE
    return temperature.to(TEMPERATURE, "a temperature")


@functools.cache
def pool(count: int | None = None) -> _core.Pool:
    """A pool of `count` threads for the engine, or of one per processor with None.

    Each is made once per process, so analyses run at once from several Python threads
    share its threads rather than each starting its own.

    Raises:
        ValueError: When `count` is not positive.
    """
    if count is not None and count < 1:
        message = f"a pool holds a thread or more, not {count}"
        raise ValueError(message)
    return _core.Pool(count)


_CHOSEN: Final = contextvars.ContextVar[_core.Pool | None]("pool", default=None)
"""The pool a block of code chose with `threads`, if any."""


@contextlib.contextmanager
def threads(count: int | None) -> Generator[None]:
    """Runs the engine on `count` threads within the block, or on one per processor.

    Outside any such block, the engine runs on the settings' ``threads``. No result
    depends on how many threads run it.

    Raises:
        ValueError: When `count` is not positive.
    """
    token = _CHOSEN.set(pool(count))
    try:
        yield
    finally:
        _CHOSEN.reset(token)


def _engine() -> _core.Pool:
    """The pool the engine runs on here: the one `threads` chose, or the settings'.

    Raises:
        JoltyError: When the settings are invalid.
    """
    return _CHOSEN.get() or pool(_settings().threads)


@functools.cache
def _settings() -> Settings:
    """The settings, read once per process.

    Raises:
        JoltyError: When the settings are invalid.
    """
    return Settings.load()


class Logged(BaseModel):
    """An analysis a simulation ran, as a line of the settings' ``analysis_log``.

    Every analysis a simulation runs, alone or in a batch, whatever runs it, the
    command line, the MCP server or a script, appends one, as it starts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis: str = Field(
        description="Its kind, by the name of its class, such as Transient."
    )
    runs: int = Field(
        ge=1,
        description="The runs it runs on: 1 alone, and a batch's runs in a batch.",
    )
    time: float = Field(description="When it started, in seconds since the epoch.")


def _log(analyses: Iterable[Analysis], runs: int) -> None:
    """Appends each of `analyses`, starting on `runs` runs, to the settings' log.

    It writes nothing when the settings name no log or the runs are none. Its lines go
    in one write to the file opened to append, so analyses that start at once, in any
    process, leave whole lines.

    Raises:
        JoltyError: When the settings are invalid, or the log cannot be written.
    """
    log = _settings().analysis_log
    if log is None or runs == 0:
        return
    started = time.time()
    logged = [
        Logged(analysis=type(each).__name__, runs=runs, time=started)
        for each in analyses
    ]
    lines = "".join(f"{each.model_dump_json()}\n" for each in logged)
    try:
        with log.open("a", encoding="utf-8") as file:
            file.write(lines)
    except OSError as error:
        message = f"cannot append to the analysis log {log}: {error.strerror}"
        raise JoltyError(message) from None


_WILDCARDS: Final = frozenset("*?[")
"""The characters that make a recorded path a pattern."""


@dataclass(frozen=True, slots=True)
class Joined:
    """Where a terminal of a device, a component instance or a logic device joins."""

    node: str | None
    """The unknown that is its voltage, by name; None on the ground."""
    thermal: bool
    """Whether it is thermal: its voltage a temperature above the ambient, and its
    current the heat flowing into it."""


def _substitutes(substitutes: Mapping[str, Device] | None) -> str:
    """The models of `substitutes` as the engine reads them, by path, in JSON."""
    stand_ins = {
        path: {"model": device.model, "parameters": device.parameters}
        for path, device in (substitutes or {}).items()
    }
    return json.dumps(stand_ins, allow_nan=False)


@contextlib.contextmanager
def _elaborating() -> Generator[None]:
    """Raises the engine's elaboration error as an `ElaborationError` of its problems.

    Raises:
        ElaborationError: With every problem the engine finds, each naming the instance
            and the line of the build script that placed it.
    """
    try:
        yield
    except _core.ElaborationError as error:
        (found,) = error.args
        raise ElaborationError([checks.problem(each) for each in found]) from None


def _swept_level(built: ir.Circuit, sweep: DcSweep) -> Dimension:
    """The dimension of the level of the source `sweep` sweeps in `built`.

    A parameter's levels are all of the dimension of the first.
    """
    if sweep.parameter is not None:
        return sweep.values[0].dimension if sweep.values else DIMENSIONLESS
    match built.find(sweep.source):
        case ir.Instance(of=ir.DeviceOf(model="current_source")):
            return CURRENT
        case ir.Instance(of=ir.DeviceOf(model="voltage_source")):
            return VOLTAGE
    if not sweep.values:
        return VOLTAGE
    # Not an independent source: the engine reports it, whatever the levels.
    return sweep.values[0].dimension


class Elaborated:
    """A circuit elaborated once, which simulations of it at any parameters share.

    Its devices run at one temperature, 27 °C unless given, with the models of
    `substitutes` standing in for component instances. A simulation made from it by
    [`Simulation.from_elaborated`][jolty.simulate.Simulation.from_elaborated] replaces
    some of its devices' numbers, as a Monte Carlo sample does, and makes its devices
    again from them, as elaborating the circuit with them would, bit for bit, without
    elaborating it anew.
    Numbers that give a device another structure, as a resistance below 1 Ω carries its
    current as an unknown of its own, run on the circuit elaborated once for that
    structure, which every simulation whose numbers give it shares.

    Examples:
        >>> from jolty import Builder, Role, component, devices
        >>> from jolty.plan import OperatingPoint
        >>> from jolty.units import V, kohm
        >>> @component
        ... def divider(c: Builder) -> None:
        ...     ground = c.port("ground", Role.GROUND)
        ...     source = c.place(devices.voltage_source(5 * V), name="V1")
        ...     upper = c.place(devices.resistor(1 * kohm), name="R1")
        ...     lower = c.place(devices.resistor(1 * kohm), name="R2")
        ...     c.connect(source.positive, upper.a)
        ...     c.connect(upper.b, lower.a, name="out")
        ...     c.connect(lower.b, source.negative, ground)
        >>> elaborated = Elaborated(divider())
        >>> sample = Simulation.from_elaborated(
        ...     elaborated, {"R1": {"resistance": 3000.0}}
        ... )
        >>> sample.run(OperatingPoint()).voltage("out")
        1.25
    """

    circuit: ir.Circuit
    """The circuit, in the canonical format."""

    temperature: Quantity | None
    """The temperature its devices run at; 27 °C when it is None."""

    substituted: frozenset[str]
    """The paths of the component instances a model stands in for."""

    def __init__(
        self,
        circuit_or_top: ir.Circuit | Component,
        temperature: Quantity | None = None,
        substitutes: Mapping[str, Device] | None = None,
    ) -> None:
        """Elaborates a circuit, or the circuit with a component at its top.

        Raises:
            ElaborationError: With every problem the engine finds, each naming the
                instance and the line of the build script that placed it.
            DimensionError: When the temperature is not one.
        """
        self.circuit = (
            circuit_or_top
            if isinstance(circuit_or_top, ir.Circuit)
            else circuit(circuit_or_top)
        )
        self.temperature = temperature
        self.substituted = frozenset(substitutes or {})
        self._kelvins = kelvins(temperature)
        with _elaborating():
            self._core = _core.Elaborated(
                self.circuit.model_dump_json(), _substitutes(substitutes)
            )
        self._structures: dict[_core.StructureKey, _core.Elaborated] = {}
        self._structure_lock = threading.Lock()

    def simulator(
        self, overrides: Mapping[str, Mapping[str, float]] | None = None
    ) -> _core.Simulator:
        """The engine's simulator of the circuit with the numbers of `overrides`.

        It runs on this circuit, or on the circuit elaborated for the structure the
        numbers give it, elaborated the first time they give it.

        Raises:
            ElaborationError: With every number `overrides` names that the circuit's
                devices do not write, and every device whose numbers they put out of
                range.
        """
        with _elaborating():
            values = self._core.values(
                json.dumps(overrides or {}, allow_nan=False), self._kelvins
            )
            key = self._core.structure_of(values)
            shared = self._core
            if not key.unchanged:
                with self._structure_lock:
                    cached = self._structures.get(key)
                    if cached is None:
                        cached = self._core.restructured(values)
                        self._structures[key] = cached
                    shared = cached
            return _core.Simulator.from_elaborated(shared, values)

    def check(self, parameters: Mapping[str, Iterable[str]]) -> None:
        """Checks that the circuit has each instance and number `parameters` names.

        `parameters` holds each instance's path with its parameters' paths, the paths
        overrides name. The engine reads each name to the number it
        is and checks no value, so any value stands in for each.

        Raises:
            ElaborationError: With every instance the circuit lacks and every number its
                devices do not write.
        """
        overrides = {
            path: dict.fromkeys(names, 0.0) for path, names in parameters.items()
        }
        with _elaborating():
            self._core.values(json.dumps(overrides, allow_nan=False), self._kelvins)

    def variants(self, runs: Iterable[Parameters]) -> _core.Variants:
        """The engine's variants of a batch of `runs`, each the numbers of one run.

        Each run is at the circuit's temperature, its numbers in place of the circuit's.

        Raises:
            ElaborationError: With every number a run names that the circuit's
                devices do not write.
        """
        listed: list[tuple[str, float | None]] = [
            (json.dumps(run, allow_nan=False), self._kelvins) for run in runs
        ]
        with _elaborating():
            return _core.Variants.explicit(self._core, listed)


class Simulation:
    """A circuit elaborated once, ready to run any number of analyses.

    Each result holds the circuit's unknowns, and what the analysis records of its
    devices and component instances: their names are made only for what a result
    records, and its devices are made again to be read only once a result records them.

    The engine solves no foundry's device, such as a SKY130 transistor, so ngspice runs
    the analyses of a circuit that holds one, with its kit's models, as
    [`foundry`][jolty.foundry] describes.

    Each analysis it runs, alone or in a batch, appends a line to the log the settings'
    ``analysis_log`` names, if any, as it starts ([`Logged`][jolty.simulate.Logged]).
    """

    circuit: ir.Circuit
    """The circuit, in the canonical format."""

    unknowns: tuple[results.Unknown, ...]
    """The node voltages and branch currents the engine solves for, in its order."""

    nets: tuple[str, ...]
    """The logic nets the event scheduler simulates, in its order."""

    devices: int
    """The number of devices, with every component instance unfolded."""

    conditions: Conditions
    """What the circuit is simulated at."""

    foundry: tuple[str, ...]
    """The paths of the foundry's devices, which ngspice simulates in the engine's
    place."""

    def __init__(  # noqa: PLR0913 - the circuit, and what it runs at
        self,
        circuit_or_top: ir.Circuit | Component,
        temperature: Quantity | None = None,
        overrides: Mapping[str, Mapping[str, float]] | None = None,
        substitutes: Mapping[str, Device] | None = None,
        *,
        process: str | None = None,
        mismatch: int | None = None,
    ) -> None:
        """Elaborates a circuit, or the circuit with a component at its top.

        Its devices are simulated at `temperature`, 27 °C unless given. `overrides`
        replaces parameters of devices, by instance path then by parameter path, such
        as ``waveform.dc``, in SI base units, as a Monte Carlo sample does. In
        `substitutes`, a device stands in for each component instance, by path, its
        terminals the component's ports, as a plan's fidelity chooses. A foundry's
        devices run at the `process` corner, such as ``ss``, and draw their kit's
        mismatch from the seed `mismatch`, as a Monte Carlo sample does; other devices
        have none to draw.

        Raises:
            ElaborationError: With every problem the engine finds, each naming the
                instance and the line of the build script that placed it.
            DimensionError: When the temperature is not one.
            ValueError: When a process corner is given for a circuit that holds no
                foundry's device.
        """
        built = (
            circuit_or_top
            if isinstance(circuit_or_top, ir.Circuit)
            else circuit(circuit_or_top)
        )
        with _elaborating():
            core = _core.Simulator(
                built.model_dump_json(),
                kelvins(temperature),
                json.dumps(overrides or {}, allow_nan=False),
                _substitutes(substitutes),
            )
        self._start(
            built,
            core,
            (temperature, overrides, frozenset(substitutes or {})),
            (process, mismatch),
        )

    @classmethod
    def from_elaborated(
        cls,
        elaborated: Elaborated,
        overrides: Mapping[str, Mapping[str, float]] | None = None,
        *,
        process: str | None = None,
        mismatch: int | None = None,
    ) -> Simulation:
        """The circuit `elaborated` holds, its parameters replaced by `overrides`.

        It simulates as the circuit elaborated with them does, bit for bit, as
        [`Elaborated`][jolty.simulate.Elaborated] describes, at the circuit's
        temperature, its foundry's devices at the `process` corner drawing their kit's
        mismatch from the seed `mismatch`.

        Raises:
            ElaborationError: With every number `overrides` names that the circuit's
                devices do not write, and every device whose numbers they put out of
                range.
            ValueError: When a process corner is given for a circuit that holds no
                foundry's device.
        """
        simulation = cls.__new__(cls)
        core = elaborated.simulator(overrides)
        replaced = (elaborated.temperature, overrides, elaborated.substituted)
        start = simulation._start  # noqa: SLF001 - readies what it made, as __init__ does
        start(elaborated.circuit, core, replaced, (process, mismatch))
        return simulation

    def _start(
        self,
        built: ir.Circuit,
        core: _core.Simulator,
        replaced: tuple[
            Quantity | None, Mapping[str, Mapping[str, float]] | None, frozenset[str]
        ],
        foundry_conditions: tuple[str | None, int | None],
    ) -> None:
        """Readies the simulation of `built` that the engine's `core` runs.

        Its devices run at a temperature, their parameters replaced by overrides and
        the component instances a model stands in for substituted, `replaced`; a
        foundry's at a process corner, drawing their kit's mismatch from a seed,
        `foundry_conditions`.

        Raises:
            ValueError: When a process corner is given for a circuit that holds no
                foundry's device.
        """
        temperature, overrides, substituted = replaced
        process, mismatch = foundry_conditions
        self.circuit = built
        self._substituted = substituted
        self._recordings: dict[tuple[str, bool], results.Recording] = {}
        self._records: dict[tuple[str, ...], frozenset[str]] = {}
        self._core = core
        self.unknowns = tuple(
            results.Unknown(name, nature) for name, nature in self._core.unknowns
        )
        self.nets = tuple(self._core.nets)
        self.devices = self._core.devices
        self.foundry = tuple(self._core.foundry_devices)
        if process is not None and not self.foundry:
            message = (
                f"the process corner {process} sets a foundry's devices, and the "
                "circuit holds none"
            )
            raise ValueError(message)
        self.conditions = Conditions(
            kelvins(temperature),
            process,
            {path: dict(replaced) for path, replaced in (overrides or {}).items()},
            mismatch,
        )

    @functools.cached_property
    def instances(self) -> tuple[str, ...]:
        """Every path a record can name, in the engine's order.

        Each device's comes first, such as ``X1/R1``, then each component instance's
        and logic device's whose terminals the engine sums, such as ``X1``.
        """
        return tuple(self._core.instances)

    def joined(self, instance: str, terminal: str) -> Joined | None:
        """Where `terminal` of the device, component instance or logic device joins.

        The instance is named by its path, such as ``X1/R1``. None when it has no such
        terminal on an electrical or a thermal net.
        """
        found = self._core.terminal(instance, terminal)
        return None if found is None else Joined(*found)

    def check(self) -> tuple[checks.Diagnostic, ...]:
        """Every diagnostic the structural checks find, at its rule's default level."""
        return checks.diagnostics(self._core.check())

    def check_operating_point(
        self, options: Options | None = None
    ) -> tuple[checks.Diagnostic, ...]:
        """Every diagnostic of the rules at the circuit's operating point.

        The operating point is solved with `options`, and each diagnostic is at its
        rule's default level, such as a net only gmin holds, or a mode that grows away
        from the operating point, which a circuit of more than 500 unknowns is not
        checked for.

        Raises:
            SimulationError: When the circuit has no operating point.
        """
        chosen = Options() if options is None else options
        settings = json.dumps(chosen.to_json(), allow_nan=False)
        try:
            found = self._core.check_operating_point(settings, _engine())
        except _core.SimulationError as error:
            raise SimulationError(str(error)) from None
        return checks.diagnostics(found)

    def check_results(
        self, solved: Mapping[str, results.AnalysisResult]
    ) -> tuple[checks.Diagnostic, ...]:
        """Every diagnostic of the rules on the analyses' results `solved`.

        Each stability analysis's loop is held to its usual margins, as
        [`specs.diagnose`][jolty.specs.diagnose] holds it, and the ratings of each
        device and component instance to every operating point and transient, at the
        furthest point of each. A rating of a current or a power is held where the
        result recorded the instance's terminals, as it does by default.
        """
        found = list(specs.diagnose(solved))
        joined = functools.cache(self.joined)
        rated = [
            (path, instance, rating)
            for path, instance in _rated(self.circuit, self._substituted)
            if any(
                joined(path, terminal) is not None for terminal in instance.terminals
            )
            for rating in instance.ratings
        ]
        for name, result in solved.items():
            match result:
                case results.OperatingPointResult():
                    values = result.values[np.newaxis, :]
                    currents = result.currents.reshape(1, -1)
                    times = None
                case results.TransientResult():
                    values, currents, times = (
                        result.values,
                        result.currents,
                        result.times,
                    )
                case _:
                    continue
            columns = {
                unknown.name: column for column, unknown in enumerate(result.unknowns)
            }
            at = _Terminals(
                joined,
                self.conditions.temperature,
                self.conditions.overrides,
                columns,
                values,
                currents,
                result.recording,
            )
            for path, instance, rating in rated:
                stress = at.stress(path, instance, rating)
                if stress is not None:
                    judged = Stressed(name, path, instance, rating, stress, times)
                    found.extend(judged.diagnostics())
        return tuple(found)

    @overload
    def run(
        self, analysis: OperatingPoint, options: Options | None = None
    ) -> results.OperatingPointResult: ...
    @overload
    def run(
        self, analysis: OperatingPoints, options: Options | None = None
    ) -> results.OperatingPointsResult: ...
    @overload
    def run(
        self, analysis: DcSweep, options: Options | None = None
    ) -> results.DcSweepResult: ...
    @overload
    def run(self, analysis: Ac, options: Options | None = None) -> results.AcResult: ...
    @overload
    def run(
        self, analysis: Transient, options: Options | None = None
    ) -> results.TransientResult: ...
    @overload
    def run(
        self, analysis: PeriodicSteadyState, options: Options | None = None
    ) -> results.PeriodicSteadyStateResult: ...
    @overload
    def run(
        self, analysis: Sensitivity, options: Options | None = None
    ) -> results.SensitivityResult: ...
    @overload
    def run(
        self, analysis: Mismatch, options: Options | None = None
    ) -> results.MismatchResult: ...
    @overload
    def run(
        self, analysis: Stability, options: Options | None = None
    ) -> results.StabilityResult: ...
    @overload
    def run(
        self, analysis: Noise, options: Options | None = None
    ) -> results.NoiseResult: ...
    @overload
    def run(
        self, analysis: TransferFunction, options: Options | None = None
    ) -> results.TransferFunctionResult: ...
    @overload
    def run(
        self, analysis: PoleZero, options: Options | None = None
    ) -> results.PoleZeroResult: ...
    def run(
        self, analysis: Analysis, options: Options | None = None
    ) -> results.AnalysisResult:
        """Runs `analysis` with `options`, or with the engine's defaults.

        The result records the instances the analysis's ``record`` names, and at an
        operating point and in a transient those its ratings read besides (see
        [`Plan`][jolty.plan.Plan]), and with its ``regions`` each other transistor's
        region and margins alone, and states where the analysis used logic devices
        outside their domains, its ``excursions``, none in ngspice.

        Raises:
            SimulationError: When the analysis fails, saying why, or the record names
                an instance the circuit does not have.
            JoltyError: When the settings are invalid, or their analysis log cannot be
                written.
        """
        chosen = Options() if options is None else options
        if self.foundry:
            _log([analysis], 1)
            return foundry.run(
                self.circuit, analysis, chosen, self.conditions, self.unknowns
            )
        encoded = json.dumps(self._encode(analysis), allow_nan=False)
        record = self._record(analysis)
        regions = isinstance(analysis, Recorded) and analysis.regions
        settings = json.dumps(
            {**chosen.to_json(), "record": record, "regions": regions}, allow_nan=False
        )
        recording = self._recording(json.dumps(record), regions=regions)
        _log([analysis], 1)
        try:
            outcome = self._core.run(encoded, settings, _engine())
        except _core.SimulationError as error:
            raise SimulationError(str(error)) from None
        watched = tuple(
            results.Excursion.model_validate(found) for found in self._core.excursions()
        )
        return replace(self._result(outcome, recording), excursions=watched)

    def batch(  # noqa: PLR0913 - the runs, their analyses, and what each reads
        self,
        analyses: Mapping[str, Analysis],
        variants: Variants,
        options: Options | None = None,
        *,
        measures: Sequence[specs.Measure] = (),
        traces: Sequence[str] = (),
        each: Callable[[results.Run], object] | None = None,
        runs: range | None = None,
    ) -> results.BatchResult:
        """Runs `analyses`, by name, on each run of `variants`, as one batch.

        It runs the variants' runs `runs`, or every one, row 0 the run at the range's
        start, so that a batch of a range holds the rows of the whole batch that the
        range names, bit for bit, but for the unit of a measure Python reads, none
        where the range reads none, and the batches of ranges that tile the runs join
        into it ([`BatchResult.joined`][jolty.results.BatchResult.joined]).
        Run `i` takes the values of the variants' run `i` in place of the simulation's,
        and runs each analysis in order with `options`, finding what
        [`run`][jolty.simulate.Simulation.run] finds at those values, bit for bit,
        however many threads the engine runs the batch on; runs at values that give
        the circuit another structure run on the circuit elaborated once for it. The
        result holds, one row per run, the values each run took, whether each analysis
        ended, what each of `measures` read, the engine reading each it can resolve
        from the analysis's kind and the circuit's unknowns, and the values of the
        unknowns `traces` names at every point of each operating point, DC sweep,
        response and transient that names its times. `each`, if given, is called with
        what each run found, under the interpreter's lock, as each run ends, in no
        particular order. A circuit that holds a foundry's devices runs each run in
        ngspice instead, as many at once as the engine has threads, a Monte Carlo
        sample drawing the kit's mismatch of its own.

        Raises:
            ElaborationError: When the variants name a number the circuit's devices do
                not write.
            KeyError: When `traces` names no unknown.
            ValueError: When `runs` steps by other than one, or does not lie within the
                variants' runs.
            JoltyError: When the settings are invalid, or their analysis log cannot be
                written.
        """
        chosen = Options() if options is None else options
        planned = _Planned.from_simulation(self, analyses, chosen, measures, traces)
        ran = self._variants(variants)
        taken = range(len(ran)) if runs is None else runs
        if taken.step != 1 or not 0 <= taken.start <= taken.stop <= len(ran):
            message = f"{taken} is not a range of the variants' {len(ran)} runs"
            raise ValueError(message)
        _log([analyses[name] for name in planned.engine], len(taken))
        found = (
            planned.in_ngspice(ran, taken, each)
            if self.foundry
            else planned.in_engine(ran, taken, each)
        )
        return planned.result(found)

    def varied(self, variants: Variants) -> tuple[str, ...]:
        """The numbers `variants` vary, as a batch's result names its columns of values.

        Raises:
            ElaborationError: When they name a number the circuit's devices do not
                write.
        """
        return _labels(self._variants(variants).parameters)

    def run_plan(self, plan: Plan) -> dict[str, results.AnalysisResult]:
        """Runs every analysis of `plan`, by name, with its options.

        Raises:
            ValueError: When the plan gives a temperature other than the simulation's.
        """
        if plan.temperature is not None:
            temperature = plan.temperature.to(TEMPERATURE, "a plan's temperature")
            own = self.conditions.temperature
            if temperature != own:
                message = (
                    f"the plan's temperature, {plan.temperature}, is not the "
                    f"simulation's, {Quantity(own, TEMPERATURE)}"
                )
                raise ValueError(message)
        return {
            name: self.run(analysis, plan.options)
            for name, analysis in plan.analyses.items()
        }

    def _record(self, analysis: Analysis) -> JsonValue:
        """What `analysis` records, in the engine's format.

        An operating point, a transient and a periodic steady state record the
        instances rated for a current or a power, whose currents the checks of
        ratings read, and every analysis that records
        ([`Recorded`][jolty.plan.Recorded]) the instances its record names besides.

        Raises:
            SimulationError: When the record names an instance the circuit does not
                have, or holds a pattern that matches none.
        """
        named = tuple(analysis.record or ()) if isinstance(analysis, Recorded) else ()
        if named not in self._records:
            self._records[named] = self._resolve(named)
        found = self._records[named]
        if isinstance(analysis, OperatingPoint | Transient | PeriodicSteadyState):
            found |= frozenset(self._rated)
        if found == self._known:
            return "all"
        instances: list[JsonValue] = [*sorted(found)]
        return {"instances": instances}

    def _resolve(self, record: tuple[str, ...]) -> frozenset[str]:
        """The paths of the instances `record` names.

        Each pattern names the instances it matches, and one after ``!`` takes them
        back, the last pattern that matches an instance deciding.

        Raises:
            SimulationError: When the record names an instance the circuit does not
                have, or holds a pattern that matches none.
        """
        named: set[str] = set()
        for pattern in record:
            body = pattern.removeprefix("!")
            if _WILDCARDS.isdisjoint(body):
                if body not in self._known:
                    hint = results.close_names(self.instances, body)
                    message = (
                        f"the circuit has no instance named {body!r} to record{hint}"
                    )
                    raise SimulationError(message)
                matched = {body}
            else:
                matched = {
                    path
                    for path in self.instances
                    if PurePosixPath(path).full_match(body)
                }
                if not matched:
                    message = f"the record's pattern {body!r} matches no instance"
                    raise SimulationError(message)
            if pattern.startswith("!"):
                named -= matched
            else:
                named |= matched
        return frozenset(named)

    @functools.cached_property
    def _known(self) -> frozenset[str]:
        """Every path a record can name."""
        return frozenset(self.instances)

    @functools.cached_property
    def _rated(self) -> tuple[str, ...]:
        """The paths of the instances rated for a current or a power."""
        return tuple(
            path
            for path, instance in _rated(self.circuit, self._substituted)
            if any(rating.stress in {"current", "power"} for rating in instance.ratings)
        )

    @functools.cached_property
    def transistors(self) -> tuple[str, ...]:
        """The paths of the devices whose models distinguish regions of operation.

        An operating point that records one, or every one's region, reads its region by
        its path.
        """
        recording = self._recording(json.dumps("all"), regions=False)
        return tuple(device.path for device in recording.devices if device.regions)

    def _recording(self, record: str, *, regions: bool) -> results.Recording:
        """What the record whose JSON text is `record` keeps, made once each.

        It keeps every other transistor's region besides when `regions`.
        """
        found = self._recordings.get((record, regions))
        if found is not None:
            return found
        devices, summed_terminals = self._core.recording(record, regions)
        thermal = {name for name, heat in summed_terminals if heat}
        recorded = []
        for (
            path,
            terminals,
            heats,
            first,
            count,
            variables,
            names,
            margins,
        ) in devices:
            thermal.update(
                f"{path}.{terminal}"
                for terminal, heat in zip(terminals, heats, strict=True)
                if heat
            )
            internals = self.unknowns[first : first + count]
            recorded.append(
                results.RecordedDevice(
                    path,
                    tuple(terminals),
                    tuple(unknown.name for unknown in internals),
                    tuple(variables),
                    tuple(names),
                    tuple(margins),
                )
            )
        found = results.Recording(
            tuple(recorded),
            tuple(name for name, _ in summed_terminals),
            frozenset(thermal),
        )
        self._recordings[record, regions] = found
        return found

    def _encode(self, analysis: Analysis) -> JsonValue:
        match analysis:
            case DcSweep():
                return analysis.to_json(self._level(analysis))
            case Sensitivity(of=DcSweep() as sweep):
                return analysis.to_json(self._level(sweep))
            case Mismatch(of=of):
                level = self._level(of) if isinstance(of, DcSweep) else VOLTAGE
                return analysis.to_json(level, self._toleranced())
        return analysis.to_json()

    def _level(self, sweep: DcSweep) -> Dimension:
        """The dimension of the swept source's level, so its levels are checked."""
        return _swept_level(self.circuit, sweep)

    def _result(
        self, outcome: _core.AnalysisResult, recording: results.Recording
    ) -> results.AnalysisResult:
        """The result of `outcome`, which recorded what `recording` says."""
        unknowns, nets = self.unknowns, self.nets
        match outcome["kind"]:
            case "operating_point":
                return self._point(outcome, recording)
            case "operating_points":
                points = tuple(
                    self._point(point, recording) for point in outcome["points"]
                )
                return results.OperatingPointsResult(unknowns, points)
            case "dc_sweep":
                return results.DcSweepResult(
                    unknowns,
                    outcome["source"],
                    outcome["nature"],
                    outcome["swept"],
                    outcome["values"],
                    nets,
                    outcome["logic"],
                    **_observed(outcome, recording),
                    parameter=outcome["parameter"],
                    strategies=tuple(outcome["strategies"]),
                    iterations=tuple(outcome["iterations"]),
                )
            case "ac":
                return results.AcResult(
                    unknowns,
                    outcome["source"],
                    outcome["frequencies"],
                    outcome["values"],
                    self._point(outcome["operating_point"], recording),
                    recording=recording,
                    currents=outcome["currents"],
                )
            case "transient" | "periodic_steady_state":
                return self._waveforms(outcome, recording)
            case _:
                return self._linear(outcome, recording)

    def _waveforms(
        self,
        outcome: _core.TransientResult | _core.PeriodicSteadyStateResult,
        recording: results.Recording,
    ) -> results.TransientResult:
        """The waveforms of a transient or a periodic steady state's period."""
        traces = tuple(
            results.LogicTrace(times, levels) for times, levels in outcome["logic"]
        )
        fields = (
            self.unknowns,
            outcome["times"],
            outcome["values"],
            _statistics(outcome["statistics"]),
            self.nets,
            traces,
        )
        observed = _observed(outcome, recording)
        if outcome["kind"] == "transient":
            return results.TransientResult(*fields, **observed)
        return results.PeriodicSteadyStateResult(
            *fields,
            **observed,
            period=outcome["period"],
            periods=outcome["periods"],
        )

    def _linear(
        self,
        outcome: _core.SensitivityResult
        | _core.MismatchResult
        | _core.StabilityResult
        | _core.NoiseResult
        | _core.PoleZeroResult
        | _core.TransferFunctionResult,
        recording: results.Recording,
    ) -> results.AnalysisResult:
        """The result of an analysis of the linearized circuit."""
        unknowns = self.unknowns
        match outcome["kind"]:
            case "sensitivity":
                parameters = tuple(
                    results.Parameter(instance, name, float(value))
                    for (instance, name), value in zip(
                        outcome["parameters"], outcome["values"], strict=True
                    )
                )
                differentiated, derivatives = self._differentiated(outcome, recording)
                return results.SensitivityResult(
                    unknowns,
                    tuple(outcome["outputs"]),
                    parameters,
                    derivatives,
                    differentiated,
                    outcome["period_derivatives"],
                )
            case "mismatch":
                return self._mismatch(outcome, recording)
            case "transfer_function":
                return results.TransferFunctionResult(
                    unknowns,
                    outcome["source"],
                    outcome["input"],
                    outcome["output"],
                    outcome["reference"],
                    outcome["gain"],
                    outcome["input_resistance"],
                    outcome["output_resistance"],
                    self._point(outcome["operating_point"], recording),
                )
            case "stability":
                return results.StabilityResult(
                    unknowns,
                    outcome["probe"],
                    outcome["frequencies"],
                    outcome["gains"],
                    self._point(outcome["operating_point"], recording),
                )
            case "noise":
                found = outcome["input"]
                noise_input = None
                if found is not None:
                    noise_input = results.NoiseInput(
                        found["source"], found["nature"], found["gains"]
                    )
                return results.NoiseResult(
                    unknowns,
                    outcome["output"],
                    outcome["reference"],
                    outcome["nature"],
                    outcome["frequencies"],
                    outcome["densities"],
                    tuple(
                        results.NoiseSource(instance, name)
                        for instance, name in outcome["sources"]
                    ),
                    outcome["contributions"],
                    noise_input,
                    self._point(outcome["operating_point"], recording),
                )
            case "pole_zero":
                return results.PoleZeroResult(
                    unknowns,
                    outcome["poles"],
                    outcome["zeros"],
                    outcome["source"],
                    outcome["output"],
                    outcome["reference"],
                    self._point(outcome["operating_point"], recording),
                )

    def _differentiated(
        self,
        outcome: _core.SensitivityResult | _core.MismatchResult,
        recording: results.Recording,
    ) -> tuple[
        results.OperatingPointResult
        | results.DcSweepResult
        | results.AcResult
        | results.TransientResult,
        results.Real | results.Complex,
    ]:
        """What a sensitivity or mismatch analysis differentiated, and its derivatives.

        At an operating point, the derivatives are its one matrix of them.

        Raises:
            SimulationError: When the engine differentiated another analysis.
        """
        differentiated = self._result(outcome["result"], recording)
        if not isinstance(
            differentiated,
            results.OperatingPointResult
            | results.DcSweepResult
            | results.AcResult
            | results.TransientResult,
        ):
            message = f"the engine differentiated {outcome['result']['kind']}"
            raise SimulationError(message)
        derivatives = outcome["derivatives"]
        if isinstance(differentiated, results.OperatingPointResult):
            derivatives = derivatives[0]
        return differentiated, derivatives

    def _mismatch(
        self, outcome: _core.MismatchResult, recording: results.Recording
    ) -> results.MismatchResult:
        """The result of a mismatch analysis, which recorded what `recording` says."""
        random_variables = tuple(
            ir.RandomVariable(
                instance=instance,
                lot=lot,
                name=name,
                mean=mean,
                standard_deviation=deviation,
                derivatives=derivatives,
            )
            for instance, lot, name, mean, deviation, derivatives in outcome[
                "random_variables"
            ]
        )
        differentiated, derivatives = self._differentiated(outcome, recording)
        return results.MismatchResult(
            self.unknowns,
            tuple(outcome["outputs"]),
            random_variables,
            derivatives,
            differentiated,
            outcome["period_derivatives"],
        )

    def _variants(self, variants: Variants) -> _core.Variants:
        """`variants` as the engine runs them, of the circuit this simulation runs.

        Raises:
            ElaborationError: When they name a number the circuit's devices do not
                write.
        """
        elaborated = self._core.elaborated
        with _elaborating():
            match variants:
                case MonteCarlo(samples=samples, seed=seed):
                    toleranced = json.dumps(self._toleranced(), allow_nan=False)
                    return _core.Variants.monte_carlo(
                        elaborated, toleranced, seed, samples
                    )
                case Grid(axes=axes, warm=warm):
                    stepped = [
                        (
                            None
                            if axis.instance is None or axis.parameter is None
                            else (axis.instance, axis.parameter),
                            axis.numbers(),
                        )
                        for axis in axes
                    ]
                    return _core.Variants.grid(elaborated, stepped, warm)
                case Explicit(variants=listed):
                    runs = [
                        (
                            json.dumps(variant.overrides, allow_nan=False),
                            None
                            if variant.temperature is None
                            else kelvins(variant.temperature),
                        )
                        for variant in listed
                    ]
                    return _core.Variants.explicit(elaborated, runs)
                case Product(first=first, second=second):
                    return _core.Variants.product(
                        self._variants(first), self._variants(second)
                    )

    def _toleranced(self) -> list[JsonValue]:
        """The circuit's toleranced numbers, as the engine draws samples of them.

        They leave out the numbers within a component instance a model stands in for,
        which the circuit the simulation runs does not hold.
        """
        return [
            {
                "instance": path,
                "parameter": name,
                "nominal": nominal,
                "tolerance": json.loads(tolerance),
            }
            for path, name, nominal, tolerance in _core.toleranced(
                self.circuit.model_dump_json()
            )
            if not any(path.startswith(f"{inside}/") for inside in self._substituted)
        ]

    def _shape(self, analysis: Analysis) -> results.AnalysisResult:
        """A result of `analysis` that holds no point, for measures to resolve.

        It holds the analysis's kind and the circuit's unknowns, and a sweep's source
        and quantity and a noise analysis's, all that a measure resolves against.
        """
        unknowns = self.unknowns
        width = len(unknowns)
        point = results.OperatingPointResult(unknowns, np.zeros(width), "")
        no, rows = np.empty(0), np.empty((0, width))
        match analysis:
            case OperatingPoint():
                return point
            case OperatingPoints():
                return results.OperatingPointsResult(unknowns, ())
            case DcSweep():
                nature = self._nature(analysis)
                return results.DcSweepResult(
                    unknowns,
                    analysis.source,
                    nature,
                    no,
                    rows,
                    parameter=analysis.parameter,
                )
            case Ac():
                phasors = np.empty((0, width), dtype=np.complex128)
                return results.AcResult(unknowns, analysis.source, no, phasors, point)
            case Transient() | PeriodicSteadyState():
                statistics = results.Statistics("", 0, 0, 0, 0)
                return (
                    results.TransientResult(unknowns, no, rows, statistics)
                    if isinstance(analysis, Transient)
                    else results.PeriodicSteadyStateResult(
                        unknowns, no, rows, statistics, period=math.nan, periods=0
                    )
                )
        return self._linear_shape(analysis, point)

    def _linear_shape(
        self,
        analysis: Sensitivity
        | Mismatch
        | Stability
        | Noise
        | PoleZero
        | TransferFunction,
        point: results.OperatingPointResult,
    ) -> results.AnalysisResult:
        """A small-signal result or sensitivities, as `_shape` makes it.

        It is of `analysis`, about the operating point `point`.
        """
        unknowns = self.unknowns
        roots = np.empty(0, dtype=np.complex128)
        match analysis:
            case Sensitivity():
                rows = np.empty((0, len(unknowns)))
                outputs = tuple(analysis.outputs)
                return results.SensitivityResult(unknowns, outputs, (), rows, point)
            case Mismatch():
                rows = np.empty((0, len(unknowns)))
                outputs = tuple(analysis.outputs)
                return results.MismatchResult(unknowns, outputs, (), rows, point)
            case Stability():
                frequencies = np.empty(0)
                return results.StabilityResult(
                    unknowns, analysis.probe, frequencies, roots, point
                )
            case Noise():
                return self._noise_shape(analysis, point)
            case TransferFunction():
                # A voltage source's current is an unknown of its own, named after it.
                branch = f"{analysis.source}:current"
                held = any(unknown.name == branch for unknown in unknowns)
                return results.TransferFunctionResult(
                    unknowns,
                    analysis.source,
                    "voltage" if held else "current",
                    analysis.output,
                    analysis.reference,
                    math.nan,
                    math.nan,
                    math.nan,
                    point,
                )
            case PoleZero():
                transfer = analysis.source is not None and analysis.output is not None
                return results.PoleZeroResult(
                    unknowns,
                    roots,
                    roots if transfer else None,
                    analysis.source,
                    analysis.output,
                    analysis.reference,
                    point,
                )

    def _noise_shape(
        self, noise: Noise, point: results.OperatingPointResult
    ) -> results.NoiseResult:
        """A noise analysis's result that holds no frequency, as `_shape` makes it."""
        names = {unknown.name: unknown.nature for unknown in self.unknowns}
        given = None
        if noise.source is not None:
            nature = "current" if self._source(noise.source) == CURRENT else "voltage"
            gains = np.empty(0, dtype=np.complex128)
            given = results.NoiseInput(noise.source, nature, gains)
        no = np.empty(0)
        return results.NoiseResult(
            self.unknowns,
            noise.output,
            noise.reference,
            names.get(noise.output, "voltage"),
            no,
            no,
            (),
            np.empty((0, 0)),
            given,
            point,
        )

    def _source(self, path: str) -> Dimension | None:
        """The dimension of the level of the independent source at `path`, if any."""
        match self.circuit.find(path):
            case ir.Instance(of=ir.DeviceOf(model="current_source")):
                return CURRENT
            case ir.Instance(of=ir.DeviceOf(model="voltage_source")):
                return VOLTAGE
        return None

    def _nature(self, sweep: DcSweep) -> results.Nature | None:
        """Whether `sweep`'s levels are voltages or currents; None for a parameter's."""
        if sweep.parameter is not None:
            return None
        return "current" if self._level(sweep) == CURRENT else "voltage"

    def _view(self, core: _core.Simulator) -> Simulation:
        """The simulation of the same circuit that the engine's `core` runs.

        Runs on another structure of the circuit run on such a one, which reads their
        results.
        """
        view = Simulation.__new__(Simulation)
        temperature = Quantity(self.conditions.temperature, TEMPERATURE)
        replaced = (temperature, self.conditions.overrides, self._substituted)
        start = view._start  # noqa: SLF001 - readies what it made, as __init__ does
        start(self.circuit, core, replaced, (None, None))
        return view

    def _point(
        self, solved: _core.SolvedPoint, recording: results.Recording
    ) -> results.OperatingPointResult:
        return results.OperatingPointResult(
            self.unknowns,
            solved["values"],
            solved["strategy"],
            solved["iterations"],
            self.nets,
            solved["logic"],
            **_observed(solved, recording),
        )


_RAISED: Final[dict[str, type[Exception]]] = {
    "overflow": OverflowError,
    "indeterminate": ValueError,
    "memory": MemoryError,
}
"""What Python raises where the engine's arithmetic stopped a measure, by its reason,
as `math.fsum` or the allocation of a distortion's samples would."""


def mismatch(seed: int, index: int) -> int:
    """The seed of the kit's mismatch of the sample `index` of a Monte Carlo analysis.

    It is the same at every corner, so a corner runs the same parts.
    """
    return int(np.random.SeedSequence([seed, index]).generate_state(1)[0])


@dataclass(frozen=True, slots=True)
class _Reading:
    """How a batch reads one measure: by the engine, in Python, or not at all."""

    measure: specs.Measure
    resolved: specs.ResolvedMeasure | None = None
    """The measure resolved for the engine, which reads it."""
    slot: int = -1
    """Its place among the measures the engine reads."""
    reason: str | None = None
    """Why it reads nothing where its analysis ended: it cannot be resolved."""
    everywhere: bool = False
    """Whether it reads nothing on any run, its analysis never running."""


@dataclass(slots=True)
class _Planned:
    """A batch as the simulation runs it.

    It holds its analyses, each as the engine reads it or why it fails on every run,
    and how it reads each measure and traces unknowns.
    """

    simulation: Simulation
    analyses: Mapping[str, Analysis]
    options: Options
    names: tuple[str, ...]
    engine: dict[str, int]
    """The place among the engine's analyses of each it runs, by name."""
    settings: list[tuple[str, str]]
    """Each analysis the engine runs and its options, as JSON texts."""
    records: dict[str, str]
    """What each analysis the engine runs records, as JSON."""
    refused: dict[str, str]
    """Why each analysis that fails on every run fails, by name."""
    readings: list[_Reading]
    traces: list[tuple[str, list[int]]]
    """Each analysis traced, by name, and the unknowns traced, by their columns."""
    read: dict[tuple[int, int], specs.Reading | str]
    """What each measure read in Python read of each run, or why it read nothing,
    by the run's index and the measure's."""

    @classmethod
    def from_simulation(
        cls,
        simulation: Simulation,
        analyses: Mapping[str, Analysis],
        options: Options,
        measures: Sequence[specs.Measure],
        traces: Sequence[str],
    ) -> _Planned:
        """The batch of `analyses` on `simulation`.

        Raises:
            KeyError: When `traces` names no unknown.
        """
        planned = cls(
            simulation,
            analyses,
            options,
            tuple(analyses),
            {},
            [],
            {},
            {},
            [],
            [],
            {},
        )
        for name, analysis in analyses.items():
            if simulation.foundry:
                # ngspice runs every analysis, and says why one fails on each run.
                planned.engine[name] = len(planned.engine)
                continue
            try:
                encoded = json.dumps(simulation._encode(analysis), allow_nan=False)  # noqa: SLF001 - one simulation's own
                record = simulation._record(analysis)  # noqa: SLF001 - one simulation's own
            except JoltyError as error:
                planned.refused[name] = str(error)
                continue
            settings = {**options.to_json(), "record": record}
            planned.engine[name] = len(planned.settings)
            planned.records[name] = json.dumps(record)
            planned.settings.append((encoded, json.dumps(settings, allow_nan=False)))
        for measure in measures:
            planned.readings.append(planned._reading(measure))
        columns = [results.column(simulation.unknowns, name) for name in traces]
        planned.traces = [
            (name, columns)
            for name, analysis in analyses.items()
            if columns
            and name in planned.engine
            and (
                isinstance(analysis, OperatingPoint | DcSweep | Ac)
                or (
                    isinstance(analysis, Transient | PeriodicSteadyState)
                    and analysis.breakpoints
                )
            )
        ]
        return planned

    def _reading(self, measure: specs.Measure) -> _Reading:
        """How the batch reads `measure`."""
        name = measure.analysis
        if name not in self.analyses:
            reason = f"the batch runs no analysis named {name}"
            return _Reading(measure, reason=reason, everywhere=True)
        if name in self.refused:
            reason = f"the analysis {name} failed"
            return _Reading(measure, reason=reason, everywhere=True)
        if measure.resolved is None or self.simulation.foundry:
            return _Reading(measure)
        shape = self.simulation._shape(self.analyses[name])  # noqa: SLF001 - one simulation's own
        try:
            resolved = measure.resolved(shape)
        except (JoltyError, KeyError) as error:
            return _Reading(measure, reason=str(error.args[0]))
        slot = sum(reading.resolved is not None for reading in self.readings)
        return _Reading(measure, resolved=resolved, slot=slot)

    def _in_python(self) -> list[int]:
        """The measures read in Python, by their indices."""
        return [
            index
            for index, reading in enumerate(self.readings)
            if reading.resolved is None and reading.reason is None
        ]

    def _readings(self, index: int, run: results.Run) -> None:
        """Reads each measure read in Python of the results of the run `index`."""
        for place in self._in_python():
            measure = self.readings[place].measure
            result = run.results.get(measure.analysis)
            if result is None:
                self.read[index, place] = f"the analysis {measure.analysis} failed"
                continue
            try:
                self.read[index, place] = measure.read(result)
            except (JoltyError, KeyError) as error:
                self.read[index, place] = str(error.args[0])

    def in_engine(
        self,
        ran: _core.Variants,
        runs: range,
        each: Callable[[results.Run], object] | None,
    ) -> _core.BatchResult:
        """What the engine finds on the runs `runs` of `ran`, each handed to `each`."""
        simulation = self.simulation
        measures = [
            (self.engine[reading.measure.analysis], reading.resolved.measure)
            for reading in self.readings
            if reading.resolved is not None
        ]
        traces = [(self.engine[name], columns) for name, columns in self.traces]
        views: dict[int, Simulation] = {}

        def hand(
            index: int,
            structure: _core.Simulator | None,
            problems: list[_core.Problem] | None,
            outcomes: list[_core.AnalysisResult | str],
        ) -> None:
            view = simulation
            if structure is not None:
                view = views.get(id(structure)) or views.setdefault(
                    id(structure),
                    simulation._view(structure),  # noqa: SLF001 - one simulation's own
                )
            run = self._run(view, index, problems, outcomes)
            self._readings(index - runs.start, run)
            if each is not None:
                each(run)

        handing = each is not None or bool(self._in_python())
        found = simulation._core.batch(  # noqa: SLF001 - one simulation's own
            self.settings,
            ran,
            (runs.start, runs.stop),
            measures,
            traces,
            _engine(),
            hand if handing else None,
        )
        found["parameters"] = ran.parameters
        return found

    def _run(
        self,
        view: Simulation,
        index: int,
        problems: list[_core.Problem] | None,
        outcomes: list[_core.AnalysisResult | str],
    ) -> results.Run:
        """What the run `index` found, its outcomes read by `view`."""
        if problems is not None:
            return results.Run(
                index, {}, {}, tuple(checks.problem(each) for each in problems)
            )
        found: dict[str, results.AnalysisResult] = {}
        errors = dict(self.refused)
        for name, place in self.engine.items():
            outcome = outcomes[place]
            if isinstance(outcome, str):
                errors[name] = outcome
                continue
            recording = view._recording(self.records[name], regions=False)  # noqa: SLF001 - one simulation's own
            found[name] = view._result(outcome, recording)  # noqa: SLF001 - one simulation's own
        ordered = {name: errors[name] for name in self.names if name in errors}
        return results.Run(index, found, ordered)

    def in_ngspice(
        self,
        ran: _core.Variants,
        runs: range,
        each: Callable[[results.Run], object] | None,
    ) -> _core.BatchResult:
        """What ngspice finds on the runs `runs` of `ran`, each handed to `each`.

        The runs go as many at once as the engine has threads, each waiting on its own
        ngspice, and are handed to `each` in order.
        """
        simulation = self.simulation
        with ThreadPoolExecutor(_engine().threads) as executor:
            found = list(executor.map(functools.partial(self._ngspice, ran), runs))
        count = len(found)
        ended = np.zeros((count, len(self.names)), dtype=np.bool_)
        problems: dict[int, list[_core.Problem]] = {}
        errors: dict[int, list[tuple[int, str]]] = {}
        traces: list[results.Real | results.Complex] = []
        for name, columns in self.traces:
            unknowns = [simulation.unknowns[column] for column in columns]
            analysis = self.analyses[name]
            shape = (count, _points(analysis), len(unknowns))
            array: results.Real | results.Complex = (
                np.empty(shape, dtype=np.complex128)
                if isinstance(analysis, Ac)
                else np.empty(shape)
            )
            for row, (run, _, _) in enumerate(found):
                array[row] = _at(run.results.get(name), analysis, unknowns)
            traces.append(array)
        for row, (run, problem, _) in enumerate(found):
            self._readings(row, run)
            if problem is not None:
                problems[row] = problem
            for name in run.results:
                ended[row, self.names.index(name)] = True
            if run.errors:
                errors[row] = [
                    (self.names.index(name), error)
                    for name, error in run.errors.items()
                ]
            if each is not None:
                each(run)
        empty = np.empty((count, 0))
        return {
            "values": np.array([numbers for _, _, numbers in found]).reshape(
                count, len(ran.parameters)
            ),
            "parameters": ran.parameters,
            "measured": empty,
            "causes": empty.astype(np.int32),
            "reasons": [],
            "ended": ended,
            "problems": problems,
            "errors": errors,
            "traces": traces,
        }

    def _ngspice(
        self, ran: _core.Variants, index: int
    ) -> tuple[results.Run, list[_core.Problem] | None, list[float]]:
        """What ngspice finds on the run `index` of `ran`, and the values it took.

        It returns the problems the run's values make too, if any.
        """
        simulation = self.simulation
        replaced, temperature = ran.overrides(index)
        own = simulation.conditions.overrides
        overrides: dict[str, dict[str, float]] = {
            path: {**own.get(path, {}), **replaced.get(path, {})}
            for path in {*own, *replaced}
        }
        kelvin = (
            simulation.conditions.temperature if temperature is None else temperature
        )
        numbers: list[float] = [
            kelvin
            if parameter is None
            else overrides.get(parameter[0], {}).get(parameter[1], math.nan)
            for parameter in ran.parameters
        ]
        elaborated = simulation._core.elaborated  # noqa: SLF001 - one simulation's own
        try:
            values = elaborated.values(json.dumps(overrides, allow_nan=False), kelvin)
            elaborated.structure_of(values)
        except _core.ElaborationError as error:
            (found,) = error.args
            problems = tuple(checks.problem(each) for each in found)
            return results.Run(index, {}, {}, problems), found, numbers
        sample = ran.sample(index)
        conditions = Conditions(
            kelvin,
            simulation.conditions.process,
            overrides,
            None if sample is None else mismatch(*sample),
        )
        solved: dict[str, results.AnalysisResult] = {}
        failed: dict[str, str] = {}
        for name, analysis in self.analyses.items():
            try:
                solved[name] = foundry.run(
                    simulation.circuit,
                    analysis,
                    self.options,
                    conditions,
                    simulation.unknowns,
                )
            except JoltyError as error:
                failed[name] = str(error)
        return results.Run(index, solved, failed), None, numbers

    def result(self, found: _core.BatchResult) -> results.BatchResult:
        """The result of the batch, from what the engine or ngspice `found`.

        Raises:
            OverflowError: Where a measure's sum overflowed on a run.
            ValueError: Where a measure's sum was indeterminate on a run.
            MemoryError: Where a distortion's samples would not fit in memory.
        """
        ended_engine = found["ended"]
        runs = int(found["values"].shape[0])
        ended = np.zeros((runs, len(self.names)), dtype=np.bool_)
        for name, place in self.engine.items():
            ended[:, self.names.index(name)] = ended_engine[:, place]
        columns = _Columns(runs, len(self.readings))
        units: list[Dimension | results.Plain | None] = []
        for place, reading in enumerate(self.readings):
            if reading.resolved is not None:
                columns.engine(place, reading, found)
                units.append(reading.resolved.unit)
            elif reading.reason is not None:
                analysis = reading.measure.analysis
                where = (
                    np.zeros(runs, dtype=np.bool_)
                    if reading.everywhere
                    else ended[:, self.names.index(analysis)]
                )
                columns.unresolved(place, reading, where)
                units.append(None)
            else:
                units.append(self._read_in_python(place, columns))
        if columns.raised:
            _, _, error = min(columns.raised, key=lambda raised: raised[:2])
            raise error
        return results.BatchResult(
            self.names,
            _labels(found["parameters"]),
            np.asarray(found["values"], dtype=np.float64),
            ended,
            tuple(str(reading.measure) for reading in self.readings),
            tuple(units),
            columns.measured,
            columns.causes,
            tuple(columns.texts),
            {
                int(index): tuple(checks.problem(each) for each in problems)
                for index, problems in found["problems"].items()
            },
            self._errors(found, runs),
            self._traces(found),
        )

    def _read_in_python(
        self, place: int, columns: _Columns
    ) -> Dimension | results.Plain | None:
        """Writes what the measure at `place` read in Python of each run, or not.

        Returns the unit of the first run's reading, which every reading takes.
        """
        unit: Dimension | results.Plain | None = None
        for (index, at), read in sorted(self.read.items()):
            if at != place:
                continue
            if isinstance(read, str):
                columns.causes[index, place] = columns.told(read)
                continue
            if unit is None:
                unit = read.unit
            if read.unit != unit:
                reason = f"read in {read.unit}, not {unit}"
                columns.causes[index, place] = columns.told(reason)
                continue
            columns.measured[index, place] = read.value
        return unit

    def _errors(self, found: _core.BatchResult, runs: int) -> dict[int, dict[str, str]]:
        """Why each analysis that failed on a run failed, by the run, then by name."""
        problems = found["problems"]
        failed: dict[int, dict[str, str]] = {}
        by_place = {place: name for name, place in self.engine.items()}
        for index, errors in found["errors"].items():
            named = {
                by_place.get(place, self.names[place]): error for place, error in errors
            }
            failed[int(index)] = named
        if self.refused:
            for index in range(runs):
                if index not in problems:
                    failed.setdefault(index, {}).update(self.refused)
        return {
            index: {name: errors[name] for name in self.names if name in errors}
            for index, errors in sorted(failed.items())
        }

    def _traces(self, found: _core.BatchResult) -> tuple[results.Traces, ...]:
        """Each analysis traced, from the arrays `found` holds."""
        simulation = self.simulation
        traces = []
        for (name, columns), values in zip(self.traces, found["traces"], strict=True):
            analysis = self.analyses[name]
            source = getattr(analysis, "source", None)
            traces.append(
                results.Traces(
                    name,
                    _ALONG[type(analysis)],
                    _axis(analysis, simulation),
                    tuple(simulation.unknowns[column] for column in columns),
                    values,
                    source,
                    simulation._nature(analysis)  # noqa: SLF001 - one simulation's own
                    if isinstance(analysis, DcSweep)
                    else None,
                    analysis.parameter if isinstance(analysis, DcSweep) else None,
                )
            )
        return tuple(traces)


def _labels(parameters: Sequence[tuple[str, str] | None]) -> tuple[str, ...]:
    """The name of each of `parameters`: ``R1 resistance``, or ``temperature``."""
    return tuple(
        "temperature" if parameter is None else f"{parameter[0]} {parameter[1]}"
        for parameter in parameters
    )


@dataclass(slots=True)
class _Columns:
    """What a batch's measures read of each run, or why not, as it is written."""

    measured: results.Real
    causes: npt.NDArray[np.int32]
    texts: dict[str, int]
    """Each reason a measure read nothing for, by its text, at its place."""
    raised: list[tuple[int, int, Exception]]
    """What the arithmetic of a measure raised, at the run and the measure."""

    def __init__(self, runs: int, width: int) -> None:
        """The columns of `width` measures of `runs` runs, none read yet."""
        self.measured = np.full((runs, width), np.nan)
        self.causes = np.full((runs, width), -1, dtype=np.int32)
        self.texts = {}
        self.raised = []

    def told(self, reason: str) -> np.int32:
        """The place of `reason` among the reasons, added if it is new."""
        return np.int32(self.texts.setdefault(reason, len(self.texts)))

    def engine(self, place: int, reading: _Reading, found: _core.BatchResult) -> None:
        """Writes what the engine read of the measure at `place`, or why not."""
        if reading.resolved is None:
            return
        failed = f"the analysis {reading.measure.analysis} failed"
        column = found["causes"][:, reading.slot]
        self.measured[:, place] = found["measured"][:, reading.slot]
        table = np.full(len(found["reasons"]) + 1, -1, dtype=np.int32)
        for cause in sorted({int(each) for each in column[column >= 0]}):
            reason, numbers, message = found["reasons"][cause]
            if reason in _RAISED:
                first = int(np.argmax(column == cause))
                self.raised.append((first, place, _RAISED[reason](message)))
            elif reason == "failed":
                table[cause] = self.told(failed)
            else:
                error = _core.MeasureError(reason, numbers)
                table[cause] = self.told(reading.resolved.explain(error))
        self.causes[:, place] = table[np.where(column >= 0, column, -1)]

    def unresolved(
        self, place: int, reading: _Reading, where: npt.NDArray[np.bool_]
    ) -> None:
        """Writes why the measure at `place`, which resolves against none, read nil.

        It is why it does not resolve against a result of its analysis, `where` the
        analysis ended, and else that the analysis failed.
        """
        if reading.reason is None:
            return
        if reading.everywhere:
            if len(where):
                self.causes[:, place] = self.told(reading.reason)
            return
        failed = f"the analysis {reading.measure.analysis} failed"
        for mask, reason in ((where, reading.reason), (~where, failed)):
            if mask.any():
                self.causes[mask, place] = self.told(reason)


_ALONG: Final[dict[type, Literal["point", "swept", "frequency", "time"]]] = {
    OperatingPoint: "point",
    DcSweep: "swept",
    Ac: "frequency",
    Transient: "time",
    PeriodicSteadyState: "time",
}
"""What the points of each kind of analysis a batch traces are."""


def _points(analysis: Analysis) -> int:
    """How many points a batch traces `analysis` along."""
    match analysis:
        case DcSweep(values=levels):
            return len(levels)
        case Ac(frequencies=frequencies):
            return len(frequencies)
        case Transient(breakpoints=times) | PeriodicSteadyState(breakpoints=times):
            return len(times)
    return 1


def _axis(analysis: Analysis, simulation: Simulation) -> results.Real:
    """The points a batch traces `analysis` along: none at an operating point."""
    match analysis:
        case DcSweep():
            level = simulation._level(analysis)  # noqa: SLF001 - one simulation's own
            return np.array(
                [value.to(level, "a sweep level") for value in analysis.values]
            )
        case Ac(frequencies=frequencies):
            return np.array([each.to(FREQUENCY, "a frequency") for each in frequencies])
        case Transient(breakpoints=times) | PeriodicSteadyState(breakpoints=times):
            return np.array([each.to(TIME, "a time to step onto") for each in times])
    return np.empty(0)


def _at(
    result: results.AnalysisResult | None,
    analysis: Analysis,
    unknowns: Sequence[results.Unknown],
) -> results.Real | results.Complex:
    """The values of `unknowns` at each point `analysis` is traced along, in `result`.

    They are traced as the engine traces them, and not a number for one the result does
    not hold, or where the analysis did not end, with no result.
    """
    points = _points(analysis)
    if isinstance(analysis, Ac):
        phasors = np.full((points, len(unknowns)), np.nan, dtype=np.complex128)
        if isinstance(result, results.AcResult):
            _picked(result.values, result.unknowns, unknowns, phasors)
        return phasors
    found = np.full((points, len(unknowns)), np.nan)
    match result:
        case results.OperatingPointResult():
            _picked(result.values[np.newaxis, :], result.unknowns, unknowns, found)
        case results.DcSweepResult():
            _picked(result.values, result.unknowns, unknowns, found)
        case results.TransientResult() if isinstance(
            analysis, Transient | PeriodicSteadyState
        ):
            times = [
                time.to(TIME, "a time to step onto") for time in analysis.breakpoints
            ]
            rows = np.array(
                [
                    [
                        _interpolated(result.times, column, time)
                        for column in result.values.T
                    ]
                    for time in times
                ]
            ).reshape(len(times), -1)
            _picked(rows, result.unknowns, unknowns, found)
    return found


def _picked(
    rows: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
    held: Sequence[results.Unknown],
    unknowns: Sequence[results.Unknown],
    into: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
) -> None:
    """Writes the columns of `rows`, over the unknowns `held`, of `unknowns` in `into`.

    Each is found by name, and the column of one `held` does not name left as it is.
    """
    named = {unknown.name: place for place, unknown in enumerate(held)}
    for place, unknown in enumerate(unknowns):
        column = named.get(unknown.name)
        if column is not None:
            into[:, place] = rows[:, column]


def _interpolated(times: results.Real, values: results.Real, time: float) -> float:
    """The value at `time`, as the engine traces it.

    It is the value at an accepted time there, and else interpolated linearly between
    the accepted times on either side; not a number outside them.
    """
    after = int(np.searchsorted(times, time, side="left"))
    if after < len(times) and times[after] == time:
        return float(values[after])
    if after == 0 or after == len(times):
        return math.nan
    start, stop = float(times[after - 1]), float(times[after])
    low, high = float(values[after - 1]), float(values[after])
    return low + (high - low) * ((time - start) / (stop - start))


def _observed(
    outcome: _core.RecordedValues, recording: results.Recording
) -> results.RecordedValues:
    """What the devices `recording` names did in `outcome`, as a result's fields."""
    return {
        "recording": recording,
        "currents": outcome["currents"],
        "charges": outcome["charges"],
        "conductances": outcome["conductances"],
        "capacitances": outcome["capacitances"],
        "variables": outcome["variables"],
    }


def _statistics(outcome: _core.Statistics) -> results.Statistics:
    """How the transient of `outcome` went, its work and its profile included."""
    profile = outcome["profile"]
    return results.Statistics(
        outcome["initial"],
        outcome["accepted"],
        outcome["rejected"],
        outcome["landed"],
        outcome["iterations"],
        results.Work(**outcome["work"]),
        None if profile is None else results.Profile(**profile),
    )


_UNITS: dict[ir.Stress, str] = {
    "voltage": "V",
    "current": "A",
    "power": "W",
    "temperature": "°C",
}


def _rated(
    circuit: ir.Circuit, substituted: Collection[str]
) -> Iterator[tuple[str, ir.Instance]]:
    """Every rated instance of `circuit`, by path, each before those it places.

    The walk skips the inside of each component instance whose path is among
    `substituted`, a model standing in for it, and of each component that holds no
    rated instance.
    """
    holds: dict[str, bool] = {}

    def holding(component: str) -> bool:
        """Whether `component` holds a rated instance, at any depth."""
        if component not in holds:
            holds[component] = any(
                instance.ratings
                or (
                    isinstance(instance.of, ir.ComponentOf)
                    and holding(instance.of.component)
                )
                for instance in circuit.components[component].instances
            )
        return holds[component]

    def walk(component: str, prefix: str) -> Iterator[tuple[str, ir.Instance]]:
        for instance in circuit.components[component].instances:
            path = f"{prefix}{instance.name}"
            if instance.ratings:
                yield path, instance
            if (
                isinstance(instance.of, ir.ComponentOf)
                and path not in substituted
                and holding(instance.of.component)
            ):
                yield from walk(instance.of.component, f"{path}/")

    yield from walk(circuit.top, "")


def _own(ambient: float, parameters: Mapping[str, JsonValue]) -> float:
    """The temperature of an instance with `parameters`, the simulation's `ambient`.

    It is the instance's own, held or offset from the simulation's, if it has one.
    """
    match parameters:
        case {"temperature": int() | float() as held}:
            return float(held)
        case {"temperature_offset": int() | float() as offset}:
            return ambient + offset
    return ambient


@dataclass(frozen=True, slots=True)
class _Terminals:
    """The voltage at and the current into each reached terminal, over a result."""

    joined: Callable[[str, str], Joined | None]
    """Where each terminal of an instance joins, by the instance's path and its name."""
    ambient: float
    """The simulation's temperature, in kelvins, which instances run at but those with
    temperatures of their own."""
    overrides: Mapping[str, Mapping[str, float]]
    """The parameters that replace the circuit's, by instance path then by path."""
    columns: Mapping[str, int]
    """The column of each unknown's values, by name."""
    values: results.Real
    currents: results.Real
    recording: results.Recording
    """What the result recorded, and where its currents hold each terminal's."""

    def stress(
        self, path: str, instance: ir.Instance, rating: ir.Rating
    ) -> results.Real | None:
        """The stress `rating` limits on the instance at `path`, at each point.

        There is none when a rated terminal is on a logic or a thermal net, or the
        result recorded no current into it and the stress needs one.
        """
        if rating.stress == "temperature":
            replaced = {**instance.parameters, **self.overrides.get(path, {})}
            return _own(self.ambient, replaced) + self._rise(path)
        named = rating.terminals or tuple(
            terminal
            for terminal in instance.terminals
            if self._electrical(path, terminal)
        )
        if not all(self._electrical(path, terminal) for terminal in named):
            return None
        if rating.stress == "voltage":
            first, second = named
            return self._voltage(path, first) - self._voltage(path, second)
        columns = [self.recording.find(path, terminal) for terminal in named]
        if None in columns:
            return None
        currents = [self.currents[:, column] for column in columns]
        if rating.stress == "current":
            (current,) = currents
            return current
        return sum(
            (
                self._voltage(path, terminal) * current
                for terminal, current in zip(named, currents, strict=True)
            ),
            start=np.zeros(self.values.shape[0]),
        )

    def _electrical(self, path: str, terminal: str) -> bool:
        """Whether `terminal` of the instance at `path` is on an electrical net."""
        found = self.joined(path, terminal)
        return found is not None and not found.thermal

    def _rise(self, path: str) -> results.Real:
        """The temperature above the ambient of the instance at `path`.

        It is that of the thermal net its terminal ``thermal`` joins, or none where it
        has no such terminal or the terminal is on the ground.
        """
        found = self.joined(path, "thermal")
        if found is None or not found.thermal or found.node is None:
            return np.zeros(self.values.shape[0])
        return self.values[:, self.columns[found.node]]

    def _voltage(self, path: str, terminal: str) -> results.Real:
        found = self.joined(path, terminal)
        if found is None or found.node is None:
            return np.zeros(self.values.shape[0])
        return self.values[:, self.columns[found.node]]


@dataclass(frozen=True, slots=True)
class Stressed:
    """A rated stress of a device over one analysis's points, judged by its rating."""

    analysis: str
    path: str
    instance: ir.Instance
    rating: ir.Rating
    stress: results.Real
    times: results.Real | None
    """The time of each point, or None at an operating point."""

    def diagnostics(self) -> list[checks.Diagnostic]:
        """The diagnostics of over-rating and near-rating, one per bound at most."""
        found = []
        high = int(np.argmax(self.stress))
        low = int(np.argmin(self.stress))
        # Derating keeps its margin from zero, a temperature's from 0 °C.
        origin = ZERO_CELSIUS if self.rating.stress == "temperature" else 0.0
        for point, bound, sign in (
            (high, self.rating.maximum, 1),
            (low, self.rating.minimum, -1),
        ):
            if bound is None:
                continue
            value = float(self.stress[point])
            reach = sign * (bound - origin)
            if sign * value > sign * bound:
                found.append(self._diagnostic(checks.OVER_RATING, point, bound, value))
            elif reach > 0 and sign * (value - origin) > checks.DERATING_FACTOR * reach:
                found.append(self._diagnostic(checks.NEAR_RATING, point, bound, value))
        return found

    def _diagnostic(
        self, rule: checks.Rule, point: int, bound: float, value: float
    ) -> checks.Diagnostic:
        unit = _UNITS[self.rating.stress]
        terminals = [f"`{terminal}`" for terminal in self.rating.terminals]
        described = {
            "voltage": f"the voltage of `{self.path}` at {' over '.join(terminals)}",
            "current": f"the current into `{self.path}` at {''.join(terminals)}",
            "power": f"the power `{self.path}` dissipates",
            "temperature": f"the temperature of `{self.path}`",
        }[self.rating.stress]
        if self.rating.stress == "temperature":
            # In degrees Celsius, as datasheets state them, rounded past the
            # conversion's own error.
            bound, value = (round(each - ZERO_CELSIUS, 9) for each in (bound, value))
        where = (
            f"at the operating point of `{self.analysis}`"
            if self.times is None
            else f"at {float(self.times[point])!r} s in `{self.analysis}`"
        )
        moves = "reaches" if value >= 0 else "falls to"
        if rule is checks.OVER_RATING:
            beyond = "beyond" if value > bound else "below"
            judged = f"{beyond} its rating of {bound!r} {unit}"
        else:
            judged = (
                f"past {checks.DERATING_FACTOR:.0%} of its rating of {bound!r} {unit}"
            )
        return checks.Diagnostic(
            rule=rule.name,
            level=rule.level,
            message=f"{described} {moves} {value!r} {unit} {where}, {judged}",
            instances=(
                checks.Location(
                    path=self.path, source_location=self.instance.source_location
                ),
            ),
            nets=(),
        )


type Parameters = Mapping[str, Mapping[str, float]]
"""Numbers of devices' parameters, by instance path, such as ``X1/R1``, then by
parameter path, such as ``resistance`` or ``waveform.dc``, in SI base units, as a
simulation's overrides name them."""

type Outputs = dict[str, npt.NDArray[np.float64] | npt.NDArray[np.complex128] | float]
"""What a [`Function`][jolty.simulate.Function] gives: each unknown's values at every
point of its analysis, by the unknown's name, in SI base units, phasors over
frequency; and an oscillator's period under [`PERIOD`][jolty.simulate.PERIOD]."""

type Differentiable = OperatingPoint | DcSweep | Ac | Transient | PeriodicSteadyState
"""An analysis a [`Function`][jolty.simulate.Function] runs."""

type Gradient = dict[str, dict[str, float | complex]]
"""Each parameter's value of a cotangent, by instance then parameter."""

type Pullback = Callable[[Mapping[str, npt.ArrayLike]], Gradient]
"""A function's pullback, as [`vjp`][jolty.simulate.vjp] gives it: each parameter's
value of a cotangent on its outputs, by instance then parameter."""

PERIOD: Final = "the period"
"""The key of an oscillator's period among a function's outputs, in seconds: no
unknown's name, as names hold no whitespace."""


class Function:
    """An analysis of a circuit as a function of its devices' parameters.

    Called with parameters, ``f(params)``, it gives the analysis's outputs at them, each
    unknown's values over the analysis's points, the circuit's other parameters as they
    are; [`jvp`][jolty.simulate.jvp] and [`vjp`][jolty.simulate.vjp] differentiate it,
    as ``jax.jvp`` and ``jax.vjp`` differentiate a function.
    The circuit is elaborated once, and each call makes its devices again from the
    parameters given, as a [`Simulation`][jolty.simulate.Simulation] with them as
    overrides does.
    """

    elaborated: Elaborated
    """The circuit, elaborated once."""

    analysis: Differentiable
    """The analysis it runs."""

    options: Options
    """The options the analysis runs with."""

    def __init__(
        self,
        circuit_or_top: ir.Circuit | Component,
        analysis: Differentiable,
        options: Options | None = None,
    ) -> None:
        """The function that runs `analysis` on a circuit with `options`.

        Raises:
            ElaborationError: With every problem the engine finds in the circuit.
            TypeError: When `analysis` is of a kind a function does not run.
        """
        if not isinstance(
            analysis, OperatingPoint | DcSweep | Ac | Transient | PeriodicSteadyState
        ):
            message = (
                "a function runs an operating point, a DC sweep, a small-signal "
                f"response, a transient or a periodic steady state, not "
                f"{type(analysis).__name__}"
            )
            raise TypeError(message)
        self.elaborated = Elaborated(circuit_or_top)
        self.analysis = analysis
        self.options = Options() if options is None else options

    def __call__(self, params: Parameters) -> Outputs:
        """The analysis's outputs with the parameters `params`.

        Raises:
            ElaborationError: With every number `params` names that the circuit's
                devices do not write, and every device it puts out of range.
            SimulationError: When the analysis fails.
        """
        core, outcome = self._run(
            params,
            lambda core, analysis, settings: core.run(analysis, settings, _engine()),
        )
        return self._outputs(core, outcome)

    def _encoded(self) -> str:
        """The analysis, differentiated, in the engine's format."""
        of = None if isinstance(self.analysis, OperatingPoint) else self.analysis
        level = (
            _swept_level(self.elaborated.circuit, of)
            if isinstance(of, DcSweep)
            else VOLTAGE
        )
        return json.dumps(differentiated(of, level), allow_nan=False)

    def _settings(self) -> str:
        """The options in the engine's format."""
        return json.dumps(self.options.to_json(), allow_nan=False)

    def _outputs(self, core: _core.Simulator, outcome: _core.AnalysisResult) -> Outputs:
        """The outputs of `outcome`, which `core` found.

        Each unknown's values over its points, by name, and an oscillator's period.
        """
        rows, period = _values(outcome)
        found: Outputs = {
            name: rows[:, column] for column, (name, _) in enumerate(core.unknowns)
        }
        if period is not None and self.oscillates:
            found[PERIOD] = float(period)
        return found

    @property
    def oscillates(self) -> bool:
        """Whether its analysis is an oscillator's, whose outputs hold its period."""
        analysis = self.analysis
        return isinstance(analysis, PeriodicSteadyState) and analysis.period is None

    def _named(self, params: Parameters) -> list[tuple[str, str]]:
        """The parameters `params` names, each its instance's path and its name."""
        return [(path, name) for path, named in params.items() for name in named]

    def _run[T](
        self, params: Parameters, run: Callable[[_core.Simulator, str, str], T]
    ) -> tuple[_core.Simulator, T]:
        """The simulator at `params`, and what `run` finds of it.

        `run` takes the simulator, and the analysis and the options in the engine's
        format.

        Raises:
            ElaborationError: As calling the function raises it.
            SimulationError: When the engine fails.
        """
        core = self.elaborated.simulator(params)
        try:
            return core, run(core, self._encoded(), self._settings())
        except _core.SimulationError as error:
            raise SimulationError(str(error)) from None

    def pushed(
        self, params: Parameters, tangents: Parameters
    ) -> tuple[Outputs, Outputs]:
        """As [`jvp`][jolty.simulate.jvp] finds it."""
        names = self._named(params)
        given = {(path, name) for path, named in tangents.items() for name in named}
        if unknown := sorted(given - set(names)):
            message = f"a tangent moves only the parameters given, not {unknown}"
            raise ValueError(message)
        tangent = [float(tangents.get(path, {}).get(name, 0.0)) for path, name in names]
        core, found = self._run(
            params,
            lambda core, analysis, settings: core.jvp(
                analysis, names, [tangent], settings, _engine()
            ),
        )
        outputs = self._outputs(core, found["result"])
        moved = found["tangents"][:, :, 0]
        pushed: Outputs = {
            name: moved[:, column] for column, (name, _) in enumerate(core.unknowns)
        }
        if (period := found["period"]) is not None and self.oscillates:
            pushed[PERIOD] = float(period[0])
        return outputs, pushed

    def pulled(self, params: Parameters) -> tuple[Outputs, Pullback]:
        """As [`vjp`][jolty.simulate.vjp] finds it."""
        names = self._named(params)
        core, (outcome, pullback) = self._run(
            params,
            lambda core, analysis, settings: core.vjp(
                analysis, names, settings, _engine()
            ),
        )
        outputs = self._outputs(core, outcome)
        columns = {name: column for column, (name, _) in enumerate(core.unknowns)}
        points = _values(outcome)[0].shape[0]
        phasors = isinstance(self.analysis, Ac)
        settings = self._settings()

        def pull(
            cotangent: Mapping[str, npt.ArrayLike],
        ) -> Gradient:
            """Each parameter's value of `cotangent`, by instance then parameter.

            Raises:
                ValueError: When the cotangent names an output the function does not
                    give.
                SimulationError: When the pullback fails.
            """
            values, period = _cotangents(
                cotangent,
                columns,
                (points, len(columns)),
                phasors=phasors,
                oscillates=self.oscillates,
            )
            try:
                found = core.pull_back(
                    pullback, values, float(period), settings, _engine()
                )
            except _core.SimulationError as error:
                raise SimulationError(str(error)) from None
            pulled: Gradient = {}
            for (path, name), value in zip(names, found.tolist(), strict=True):
                pulled.setdefault(path, {})[name] = value
            return pulled

        return outputs, pull

    @functools.cached_property
    def unknowns(self) -> tuple[str, ...]:
        """The names of the circuit's unknowns, its outputs, in the engine's order."""
        return tuple(name for name, _ in self.elaborated.simulator().unknowns)

    @property
    def points(self) -> int:
        """How many points each output of a run of a batch has.

        An operating point's one, a DC sweep's levels, a response's frequencies, and a
        transient's or a periodic steady state's breakpoints, none for an oscillator's.
        """
        return _points(self.analysis)

    def batch(self, names: Sequence[tuple[str, str]], values: npt.ArrayLike) -> Outputs:
        """The analysis's outputs on each run of a batch, each a row of runs by points.

        The parameters `names`, each an instance's path and a parameter's path, take
        the numbers of each row of `values`, a row of names by run, in SI base units;
        each output has the analysis's [`points`][jolty.simulate.Function.points], at
        a transient's breakpoints rather than its own times, and an oscillator's
        period is a number per run under [`PERIOD`][jolty.simulate.PERIOD]. The engine
        runs the batch across its threads, each run's bits the same however many
        threads run it and however many runs run beside it; a call of the function
        runs alone and may dissect its matrix, and so find other last digits.

        Raises:
            ValueError: When `values` is not a row of names by run.
            ElaborationError: With the problems of the first run whose numbers make
                some.
            SimulationError: Naming the first run whose analysis fails.
        """
        outputs, _ = self._batched(names, values, [])
        return outputs

    def batch_vjp(
        self, names: Sequence[tuple[str, str]], values: npt.ArrayLike
    ) -> tuple[Outputs, BatchPullback]:
        """The outputs of a [`batch`][jolty.simulate.Function.batch], and its pullback.

        As [`vjp`][jolty.simulate.vjp] for each run: the pullback takes a cotangent on
        each run's outputs and gives its value on each parameter of `names`, by the
        adjoint method, one transposed solve at each point of a run however many the
        parameters, the runs' pullbacks spread across the engine's threads.

        Raises:
            ValueError: When `values` is not a row of names by run.
            ElaborationError: With the problems of the first run whose numbers make
                some.
            SimulationError: Naming the first run whose analysis fails.
        """
        outputs, core = self._batched(names, values, list(names))
        pullback = BatchPullback(
            core,
            self.unknowns,
            self.points,
            phasors=isinstance(self.analysis, Ac),
            oscillates=self.oscillates,
            settings=self._settings(),
        )
        return outputs, pullback

    def _batched(
        self,
        names: Sequence[tuple[str, str]],
        values: npt.ArrayLike,
        parameters: list[tuple[str, str]],
    ) -> tuple[Outputs, _core.BatchPullback]:
        """The outputs of a batch at `values` of `names`, and the engine's pullback.

        The pullback is with respect to `parameters`, none for a batch without one.

        Raises:
            ValueError: When `values` is not a row of names by run.
            ElaborationError: With the problems of the first run whose numbers make
                some.
            SimulationError: Naming the first run whose analysis fails.
        """
        rows = np.asarray(values, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(names):  # noqa: PLR2004 - a matrix
            message = f"values are a row of {len(names)} names by run, not {rows.shape}"
            raise ValueError(message)
        runs: list[dict[str, dict[str, float]]] = []
        for row in rows.tolist():
            run: dict[str, dict[str, float]] = {}
            for (path, name), value in zip(names, row, strict=True):
                run.setdefault(path, {})[name] = value
            runs.append(run)
        variants = self.elaborated.variants(runs)
        base = self.elaborated.simulator()
        columns = list(range(len(base.unknowns)))
        try:
            found, pullback = base.batch_vjp(
                self._encoded(),
                self._settings(),
                parameters,
                variants,
                (0, len(runs)),
                columns,
                _engine(),
            )
        except _core.SimulationError as error:
            raise SimulationError(str(error)) from None
        if found["problems"]:
            problems = found["problems"][min(found["problems"])]
            raise ElaborationError([checks.problem(each) for each in problems])
        if found["errors"]:
            first = min(found["errors"])
            message = f"batch run {first}: {found['errors'][first]}"
            raise SimulationError(message)
        traces = found["traces"]
        outputs: Outputs = {
            name: traces[:, :, column] for column, name in enumerate(self.unknowns)
        }
        if self.oscillates:
            outputs[PERIOD] = found["periods"]
        return outputs, pullback


class BatchPullback:
    """The pullback of a [`batch`][jolty.simulate.Function.batch], a run each.

    [`Function.batch_vjp`][jolty.simulate.Function.batch_vjp] gives it; called with a
    cotangent on the batch's outputs, it gives each row's value on each parameter.
    """

    def __init__(  # noqa: PLR0913 - what a batch's pullback reads of its function
        self,
        core: _core.BatchPullback,
        unknowns: Sequence[str],
        points: int,
        *,
        phasors: bool,
        oscillates: bool,
        settings: str,
    ) -> None:
        """The pullback `core` of a batch of a function of `unknowns` at `points`.

        Its cotangents are complex where `phasors`, and take an oscillator's period
        where `oscillates`; it pulls them back with the options `settings` names.
        """
        self._core = core
        self._columns = {name: column for column, name in enumerate(unknowns)}
        self._points = points
        self._phasors = phasors
        self._oscillates = oscillates
        self._settings = settings

    def __call__(
        self, cotangent: Mapping[str, npt.ArrayLike], runs: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
        """Each row's value of `cotangent` on each parameter, by row then parameter.

        `cotangent` gives each output's, by its name, by row then point, those it
        leaves out zero, and an oscillator's period's, a number per row, under
        [`PERIOD`][jolty.simulate.PERIOD]; row `i` is on the batch's run at `runs[i]`,
        each run in order without `runs`, a run named by any number of rows. Over
        frequency, the cotangents and their values are complex, as for
        [`vjp`][jolty.simulate.vjp].

        Raises:
            ValueError: When the cotangent names an output the function does not give.
            SimulationError: When a run failed, a cotangent moves an unknown its run's
                structure dropped, or a pullback fails.
        """
        rows = np.atleast_1d(
            np.arange(self._core.runs) if runs is None else np.asarray(runs, np.intp)
        )
        values, periods = _cotangents(
            cotangent,
            self._columns,
            (len(rows), self._points, len(self._columns)),
            phasors=self._phasors,
            oscillates=self._oscillates,
        )
        try:
            return self._core.pull_back(
                rows.tolist(), values, periods.tolist(), self._settings, _engine()
            )
        except _core.SimulationError as error:
            raise SimulationError(str(error)) from None


def _cotangents(
    cotangent: Mapping[str, npt.ArrayLike],
    columns: Mapping[str, int],
    shape: tuple[int, ...],
    *,
    phasors: bool,
    oscillates: bool,
) -> tuple[
    npt.NDArray[np.float64] | npt.NDArray[np.complex128], npt.NDArray[np.float64]
]:
    """The values `cotangent` gives the unknowns at `columns`, and the periods.

    The values have `shape`, its last axis the unknowns' and the one before it the
    points', complex where `phasors`; the periods have the axes before them and take
    an oscillator's under [`PERIOD`][jolty.simulate.PERIOD] where `oscillates`. What
    the cotangent leaves out is zero.

    Raises:
        ValueError: When the cotangent names an output the function does not give.
    """
    values: npt.NDArray[np.float64] | npt.NDArray[np.complex128] = (
        np.zeros(shape, np.complex128) if phasors else np.zeros(shape)
    )
    periods = np.zeros(shape[:-2])
    for name, given in cotangent.items():
        if name == PERIOD and oscillates:
            periods[...] = np.asarray(given, dtype=np.float64)
        elif name in columns:
            values[..., columns[name]] = given
        else:
            message = f"the function gives no output `{name}`"
            raise ValueError(message)
    return values, periods


def _values(
    outcome: _core.AnalysisResult,
) -> tuple[npt.NDArray[np.float64] | npt.NDArray[np.complex128], float | None]:
    """The values of every unknown at each point of `outcome`, and its period.

    The values are by point, then unknown; the period is a periodic steady state's.

    Raises:
        SimulationError: For an outcome of an analysis a function does not run.
    """
    match outcome["kind"]:
        case "periodic_steady_state":
            return np.atleast_2d(outcome["values"]), outcome["period"]
        case "operating_point" | "dc_sweep" | "ac" | "transient":
            return np.atleast_2d(outcome["values"]), None
        case other:
            message = f"a function runs no {other} analysis"
            raise SimulationError(message)


def function(
    circuit_or_top: ir.Circuit | Component,
    analysis: Differentiable,
    options: Options | None = None,
) -> Function:
    """The analysis `analysis` of a circuit as a function of its parameters.

    ``function(circuit, analysis)(params)`` runs it with `options` at the parameters
    `params` and gives its outputs; see [`Function`][jolty.simulate.Function].

    Raises:
        ElaborationError: With every problem the engine finds in the circuit.
    """
    return Function(circuit_or_top, analysis, options)


def jvp(
    f: Function, params: Parameters, tangents: Parameters
) -> tuple[Outputs, Outputs]:
    """The outputs of `f` at `params`, and their derivative along `tangents`.

    As ``jax.jvp(f, primals, tangents)``: the parameters `params` names are those
    differentiated, and `tangents` moves some of them, in SI base units, the others
    held. The derivatives, by output as the outputs are, are exact, by the direct
    method: one solve at each point of the analysis.

    Raises:
        ValueError: When a tangent moves a parameter `params` does not give.
        ElaborationError: As calling `f` raises it.
        SimulationError: When the analysis or a Jacobian's solve fails.
    """
    return f.pushed(params, tangents)


def vjp(f: Function, params: Parameters) -> tuple[Outputs, Pullback]:
    """The outputs of `f` at `params`, and its pullback.

    As ``jax.vjp(f, primals)``: the pullback takes a cotangent on the outputs, by
    output as they are, each unknown's at every point, those it leaves out zero, and
    gives its value on each parameter of `params`, by instance then parameter, by the
    adjoint method: one transposed solve at each point of the analysis, however many
    the parameters. A scalar loss's derivatives with respect to the outputs are its
    cotangent, and the pullback gives its gradient. Over frequency, the cotangents and
    their values are complex, `vᵀ dx/dp`: a real loss of the phasors `a + jb` has the
    real part of the pullback of `∂L/∂a - j ∂L/∂b`.

    Raises:
        ElaborationError: As calling `f` raises it.
        SimulationError: When the analysis fails, or over time on a circuit with delay
            lines or logic, whose points the adjoint method cannot evaluate again as
            they were.
    """
    return f.pulled(params)
