# Copyright 2026 Maxence Faldor
"""Run analyses on a circuit with the Rust engine.

A [`Simulation`][jolty.simulate.Simulation] elaborates a circuit once, its devices at
one temperature, 27 °C unless given, and runs any number of analyses on it. A circuit
[`Elaborated`][jolty.simulate.Elaborated] once is shared by simulations of it at any
parameters, as a Monte Carlo analysis's samples are: each makes its devices again from
its parameters, and none elaborates the circuit anew.

Examples:
    >>> from jolty import Builder, Role, component, devices
    >>> from jolty.plan import OperatingPoint
    >>> from jolty.units import V, kohm
    >>> @component
    ... def divider(c: Builder) -> None:
    ...     ground = c.port("ground", Role.GROUND)
    ...     source = c.place(devices.voltage_source(5 * V), id="V1")
    ...     upper = c.place(devices.resistor(1 * kohm), id="R1")
    ...     lower = c.place(devices.resistor(1 * kohm), id="R2")
    ...     c.connect(source.positive, upper.a)
    ...     c.connect(upper.b, lower.a, name="out")
    ...     c.connect(lower.b, source.negative, ground)
    >>> Simulation(divider()).run(OperatingPoint()).voltage("out")
    2.5
"""

import contextlib
import functools
import json
from collections.abc import Callable, Collection, Generator, Iterator, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final, Literal, overload

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from jolty import _core, checks, foundry, ir, results, specs
from jolty.builder import Component, circuit
from jolty.devices import Device
from jolty.errors import ElaborationError, SimulationError
from jolty.plan import (
    Ac,
    Analysis,
    Conditions,
    DcSweep,
    Noise,
    OperatingPoint,
    OperatingPoints,
    Options,
    Plan,
    PoleZero,
    Recorded,
    Sensitivity,
    Stability,
    Transient,
)
from jolty.units import (
    CURRENT,
    TEMPERATURE,
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


def _stand_ins(substitutes: Mapping[str, Device] | None) -> str:
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


class Elaborated:
    """A circuit elaborated once, which simulations of it at any parameters share.

    Its devices run at one temperature, 27 °C unless given, with the models of
    `substitutes` standing in for component instances. A
    [`Simulation.of`][jolty.simulate.Simulation.of] it replaces some of its devices'
    numbers, as a Monte Carlo sample does, and makes its devices again from them, as
    elaborating the circuit with them would, bit for bit, without elaborating it anew.
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
        ...     source = c.place(devices.voltage_source(5 * V), id="V1")
        ...     upper = c.place(devices.resistor(1 * kohm), id="R1")
        ...     lower = c.place(devices.resistor(1 * kohm), id="R2")
        ...     c.connect(source.positive, upper.a)
        ...     c.connect(upper.b, lower.a, name="out")
        ...     c.connect(lower.b, source.negative, ground)
        >>> elaborated = Elaborated(divider())
        >>> sample = Simulation.of(elaborated, {"R1": {"resistance": 3000.0}})
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
                self.circuit.model_dump_json(), _stand_ins(substitutes)
            )
        self._structures: dict[_core.StructureKey, _core.Elaborated] = {}

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
                shared = self._structures.get(key) or self._structures.setdefault(
                    key, self._core.restructured(values)
                )
            return _core.Simulator.from_elaborated(shared, values)


class Simulation:
    """A circuit elaborated once, ready to run any number of analyses.

    Each result holds the circuit's unknowns, and what the analysis records of its
    devices and component instances: their names are made only for what a result
    records, and its devices are made again to be read only once a result records them.

    The engine solves no foundry's device, such as a SKY130 transistor, so ngspice runs
    the analyses of a circuit that holds one, with its kit's models, as
    [`foundry`][jolty.foundry] describes.
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

    excursions: tuple[Excursion, ...]
    """Where the last analysis used logic devices outside their domains."""

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
                _stand_ins(substitutes),
            )
        self._start(
            built,
            core,
            (temperature, overrides, frozenset(substitutes or {})),
            (process, mismatch),
        )

    @classmethod
    def of(
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
        self._recordings: dict[str, results.Recording] = {}
        self._records: dict[tuple[str, ...], frozenset[str]] = {}
        self._core = core
        self.unknowns = tuple(
            results.Unknown(name, quantity) for name, quantity in self._core.unknowns
        )
        self.nets = tuple(self._core.nets)
        self.devices = self._core.devices
        self.excursions = ()
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
            found = self._core.check_operating_point(settings)
        except _core.SimulationError as error:
            raise SimulationError(str(error)) from None
        return checks.diagnostics(found)

    def check_results(
        self, solved: Mapping[str, results.Result]
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
                case results.OperatingPoint():
                    values = result.values[np.newaxis, :]
                    currents = result.currents.reshape(1, -1)
                    times = None
                case results.Waveforms():
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
    ) -> results.OperatingPoint: ...
    @overload
    def run(
        self, analysis: OperatingPoints, options: Options | None = None
    ) -> results.OperatingPoints: ...
    @overload
    def run(
        self, analysis: DcSweep, options: Options | None = None
    ) -> results.Sweep: ...
    @overload
    def run(self, analysis: Ac, options: Options | None = None) -> results.Response: ...
    @overload
    def run(
        self, analysis: Transient, options: Options | None = None
    ) -> results.Waveforms: ...
    @overload
    def run(
        self, analysis: Sensitivity, options: Options | None = None
    ) -> results.Sensitivities: ...
    @overload
    def run(
        self, analysis: Stability, options: Options | None = None
    ) -> results.LoopGain: ...
    @overload
    def run(
        self, analysis: Noise, options: Options | None = None
    ) -> results.NoiseSpectrum: ...
    @overload
    def run(
        self, analysis: PoleZero, options: Options | None = None
    ) -> results.PolesZeros: ...
    def run(self, analysis: Analysis, options: Options | None = None) -> results.Result:
        """Runs `analysis` with `options`, or with the engine's defaults.

        The result records the instances the analysis's ``record`` names, and at an
        operating point and in a transient those its ratings read besides (see
        [`Plan`][jolty.plan.Plan]).

        Raises:
            SimulationError: When the analysis fails, saying why, or the record names
                an instance the circuit does not have.
        """
        chosen = Options() if options is None else options
        if self.foundry:
            self.excursions = ()
            return foundry.run(
                self.circuit, analysis, chosen, self.conditions, self.unknowns
            )
        encoded = json.dumps(self._encode(analysis), allow_nan=False)
        record = self._record(analysis)
        settings = json.dumps({**chosen.to_json(), "record": record}, allow_nan=False)
        recording = self._recording(json.dumps(record))
        try:
            outcome = self._core.run(encoded, settings)
        except _core.SimulationError as error:
            raise SimulationError(str(error)) from None
        finally:
            self.excursions = tuple(
                Excursion.model_validate(found) for found in self._core.excursions()
            )
        return self._result(outcome, recording)

    def run_plan(self, plan: Plan) -> dict[str, results.Result]:
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

        An operating point and a transient record the instances rated for a current or
        a power, whose currents the checks of ratings read, and every analysis that
        records ([`Recorded`][jolty.plan.Recorded]) the instances its record names
        besides.

        Raises:
            SimulationError: When the record names an instance the circuit does not
                have, or holds a pattern that matches none.
        """
        named = tuple(analysis.record or ()) if isinstance(analysis, Recorded) else ()
        if named not in self._records:
            self._records[named] = self._resolve(named)
        found = self._records[named]
        if isinstance(analysis, OperatingPoint | Transient):
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

    def _recording(self, record: str) -> results.Recording:
        """What the record whose JSON text is `record` keeps, made once each."""
        found = self._recordings.get(record)
        if found is not None:
            return found
        devices, composites = self._core.recording(record)
        thermal = {name for name, heat in composites if heat}
        recorded = []
        for path, terminals, heats, first, count, variables in devices:
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
                )
            )
        found = results.Recording(
            tuple(recorded),
            tuple(name for name, _ in composites),
            frozenset(thermal),
        )
        self._recordings[record] = found
        return found

    def _encode(self, analysis: Analysis) -> JsonValue:
        match analysis:
            case DcSweep():
                return analysis.to_json(self._level(analysis))
            case Sensitivity(of=DcSweep() as sweep):
                return analysis.to_json(self._level(sweep))
        return analysis.to_json()

    def _level(self, sweep: DcSweep) -> Dimension:
        """The dimension of the swept source's level, so its levels are checked."""
        match self.circuit.find(sweep.source):
            case ir.Instance(of=ir.DeviceOf(device="current_source")):
                return CURRENT
            case ir.Instance(of=ir.DeviceOf(device="voltage_source")):
                return VOLTAGE
        if not sweep.levels:
            return VOLTAGE
        # Not an independent source: the engine reports it, whatever the levels.
        return sweep.levels[0].dimension

    def _result(
        self, outcome: _core.Outcome, recording: results.Recording
    ) -> results.Result:
        """The result of `outcome`, which recorded what `recording` says."""
        unknowns, nets = self.unknowns, self.nets
        match outcome["kind"]:
            case "operating_point":
                return self._point(outcome, recording)
            case "operating_points":
                points = tuple(
                    self._point(point, recording) for point in outcome["points"]
                )
                return results.OperatingPoints(unknowns, points)
            case "dc_sweep":
                quantity = outcome["quantity"]
                if quantity is None:
                    message = (
                        f"the sweep of `{outcome['source']}`'s {outcome['parameter']} "
                        "is of a parameter, which a result reads only of a source"
                    )
                    raise SimulationError(message)
                return results.Sweep(
                    unknowns,
                    outcome["source"],
                    quantity,
                    outcome["levels"],
                    outcome["values"],
                    nets,
                    outcome["logic"],
                    **_observed(outcome, recording),
                )
            case "ac":
                return results.Response(
                    unknowns,
                    outcome["source"],
                    outcome["frequencies"],
                    outcome["values"],
                    self._point(outcome["operating_point"], recording),
                    recording=recording,
                    currents=outcome["currents"],
                )
            case "transient":
                statistics = _statistics(outcome["statistics"])
                traces = tuple(
                    results.Trace(times, levels) for times, levels in outcome["logic"]
                )
                return results.Waveforms(
                    unknowns,
                    outcome["times"],
                    outcome["values"],
                    statistics,
                    nets,
                    traces,
                    **_observed(outcome, recording),
                )
            case _:
                return self._linear(outcome, recording)

    def _linear(
        self,
        outcome: _core.SensitivityOutcome
        | _core.StabilityOutcome
        | _core.NoiseOutcome
        | _core.PoleZeroOutcome,
        recording: results.Recording,
    ) -> results.Result:
        """The result of sensitivities, a loop gain, a noise spectrum or poles."""
        unknowns = self.unknowns
        match outcome["kind"]:
            case "sensitivity":
                parameters = tuple(
                    results.Parameter(instance, name, float(value))
                    for (instance, name), value in zip(
                        outcome["parameters"], outcome["values"], strict=True
                    )
                )
                differentiated = self._result(outcome["result"], recording)
                if not isinstance(
                    differentiated,
                    results.OperatingPoint
                    | results.Sweep
                    | results.Response
                    | results.Waveforms,
                ):
                    message = f"the engine differentiated {outcome['result']['kind']}"
                    raise SimulationError(message)
                derivatives = outcome["derivatives"]
                if isinstance(differentiated, results.OperatingPoint):
                    derivatives = derivatives[0]
                return results.Sensitivities(
                    unknowns,
                    tuple(outcome["outputs"]),
                    parameters,
                    derivatives,
                    differentiated,
                )
            case "stability":
                return results.LoopGain(
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
                        found["source"], found["quantity"], found["gains"]
                    )
                return results.NoiseSpectrum(
                    unknowns,
                    outcome["output"],
                    outcome["reference"],
                    outcome["quantity"],
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
                return results.PolesZeros(
                    unknowns,
                    outcome["poles"],
                    outcome["zeros"],
                    outcome["source"],
                    outcome["output"],
                    outcome["reference"],
                    self._point(outcome["operating_point"], recording),
                )

    def _point(
        self, solved: _core.SolvedPoint, recording: results.Recording
    ) -> results.OperatingPoint:
        return results.OperatingPoint(
            self.unknowns,
            solved["values"],
            solved["strategy"],
            self.nets,
            solved["logic"],
            **_observed(solved, recording),
        )


def _observed(
    outcome: _core.Observed, recording: results.Recording
) -> results.Observed:
    """What the devices `recording` names did in `outcome`, as a result's fields."""
    return {
        "recording": recording,
        "currents": outcome["currents"],
        "charges": outcome["charges"],
        "conductances": outcome["conductances"],
        "capacitances": outcome["capacitances"],
        "readings": outcome["variables"],
    }


def _statistics(outcome: _core.StatisticsOutcome) -> results.Statistics:
    """How the transient of `outcome` went, its work and its profile included."""
    profile = outcome["profile"]
    return results.Statistics(
        outcome["initial"],
        outcome["accepted"],
        outcome["rejected"],
        outcome["landed"],
        outcome["iterations"],
        outcome["periods"],
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
            path = f"{prefix}{instance.id}"
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
        """The diagnostics of over-rating and derating, one per bound at most."""
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
            elif reach > 0 and sign * (value - origin) > checks.DERATED * reach:
                found.append(self._diagnostic(checks.DERATING, point, bound, value))
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
            judged = f"past {checks.DERATED:.0%} of its rating of {bound!r} {unit}"
        return checks.Diagnostic(
            rule=rule.name,
            level=rule.level,
            message=f"{described} {moves} {value!r} {unit} {where}, {judged}",
            instances=(checks.Instance(path=self.path, source=self.instance.source),),
            nets=(),
        )
