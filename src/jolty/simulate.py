# Copyright 2026 Maxence Faldor
"""Run analyses on a circuit with the Rust engine.

A [`Simulation`][jolty.simulate.Simulation] elaborates a circuit once, its devices at
one temperature, 27 °C unless given, and runs any number of analyses on it.

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

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from jolty import _core, checks, ir, results, specs
from jolty.builder import Component, circuit
from jolty.devices import Device
from jolty.errors import ElaborationError, SimulationError
from jolty.plan import (
    Ac,
    Analysis,
    DcSweep,
    Noise,
    OperatingPoint,
    OperatingPoints,
    Options,
    Plan,
    PoleZero,
    Sensitivity,
    Stability,
    Transient,
)
from jolty.units import CURRENT, TEMPERATURE, VOLTAGE, Dimension, Quantity


def kelvins(temperature: Quantity | None) -> float:
    """`temperature` in kelvins; 27 °C when it is None.

    Raises:
        DimensionError: When the temperature is not one.
    """
    if temperature is None:
        return _core.REFERENCE_TEMPERATURE
    return temperature.to(TEMPERATURE, "a temperature")


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


class Simulation:
    """A circuit elaborated once, ready to run any number of analyses."""

    circuit: ir.Circuit
    """The circuit, in the canonical format."""

    unknowns: tuple[results.Unknown, ...]
    """The node voltages and branch currents the engine solves for, in its order."""

    nets: tuple[str, ...]
    """The logic nets the event scheduler simulates, in its order."""

    terminals: tuple[str, ...]
    """Every terminal of every device, such as ``X1/R1.a``, in the order of every
    result's currents and charges."""

    terminal_nodes: tuple[str | None, ...]
    """The unknown that is each terminal's voltage, by name, in the order of
    `terminals`, or None for a terminal on the ground."""

    composites: results.Composites
    """Each terminal of a component instance or a logic device, such as ``X1.vdd``, with
    the indices of the device terminals whose currents and charges sum to its own."""

    composite_nodes: tuple[str | None, ...]
    """The unknown that is each composite terminal's voltage, by name, in the order of
    `composites`, or None for one on the ground."""

    derivatives: tuple[tuple[str, str], ...]
    """For each device, every pair of its terminals and internal unknowns, its equation
    then its unknown, in the order of every result's conductances and capacitances."""

    variables: tuple[tuple[str, str], ...]
    """Every variable of every device, its path and its name, in the order of every
    result's readings."""

    devices: int
    """The number of devices, with every component instance unfolded."""

    temperature: float
    """The temperature the devices are simulated at, in kelvins."""

    excursions: tuple[Excursion, ...]
    """Where the last analysis used logic devices outside their domains."""

    def __init__(
        self,
        circuit_or_top: ir.Circuit | Component,
        temperature: Quantity | None = None,
        overrides: Mapping[str, Mapping[str, float]] | None = None,
        substitutes: Mapping[str, Device] | None = None,
    ) -> None:
        """Elaborates a circuit, or the circuit with a component at its top.

        Its devices are simulated at `temperature`, 27 °C unless given. `overrides`
        replaces parameters of devices, by instance path then by parameter path, such
        as ``waveform.dc``, in SI base units, as a Monte Carlo sample does. In
        `substitutes`, a device stands in for each component instance, by path, its
        terminals the component's ports, as a plan's fidelity chooses.

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
        self.temperature = kelvins(temperature)
        try:
            stand_ins = {
                path: {"model": device.model, "parameters": device.parameters}
                for path, device in (substitutes or {}).items()
            }
            self._core = _core.Simulator(
                self.circuit.model_dump_json(),
                self.temperature,
                json.dumps(overrides or {}, allow_nan=False),
                json.dumps(stand_ins, allow_nan=False),
            )
        except _core.ElaborationError as error:
            (found,) = error.args
            raise ElaborationError([checks.problem(each) for each in found]) from None
        self.unknowns = tuple(
            results.Unknown(name, quantity) for name, quantity in self._core.unknowns
        )
        self.nets = tuple(self._core.nets)
        self.terminals = tuple(self._core.terminals)
        self.terminal_nodes = tuple(self._core.terminal_nodes)
        self.composites = tuple(
            (name, tuple(members)) for name, members in self._core.composites
        )
        self.composite_nodes = tuple(self._core.composite_nodes)
        self.derivatives = tuple(
            (equation, unknown) for equation, unknown in self._core.derivatives
        )
        self.variables = tuple(
            (instance, name) for instance, name in self._core.variables
        )
        self.devices = self._core.devices
        self.excursions = ()

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
        furthest point of each.
        """
        found = list(specs.diagnose(solved))
        reached = self._reached()
        columns = {unknown.name: column for column, unknown in enumerate(self.unknowns)}
        rated = [
            (path, instance, rating)
            for path, instance in _instances(self.circuit, self.circuit.top, "")
            if any(f"{path}.{terminal}" in reached for terminal in instance.terminals)
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
            at = _Terminals(reached, columns, values, currents, len(self.terminals))
            for path, instance, rating in rated:
                stress = at.stress(path, instance, rating)
                if stress is not None:
                    judged = Stressed(name, path, instance, rating, stress, times)
                    found.extend(judged.diagnostics())
        return tuple(found)

    def _reached(self) -> dict[str, tuple[str | None, tuple[int, ...]]]:
        """Each terminal on an electrical net, such as ``X1/R1.a`` or ``X1.vdd``.

        A terminal is of a device, a component instance or a logic device, and maps to
        the unknown that is its voltage, None on the ground, and the columns of the
        currents that sum to its own.
        """
        reached: dict[str, tuple[str | None, tuple[int, ...]]] = {
            terminal: (node, (column,))
            for column, (terminal, node) in enumerate(
                zip(self.terminals, self.terminal_nodes, strict=True)
            )
        }
        for (composite, members), node in zip(
            self.composites, self.composite_nodes, strict=True
        ):
            reached[composite] = (node, members)
        return reached

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

        Raises:
            SimulationError: When the analysis fails, saying why.
        """
        encoded = json.dumps(self._encode(analysis), allow_nan=False)
        chosen = Options() if options is None else options
        settings = json.dumps(chosen.to_json(), allow_nan=False)
        try:
            outcome = self._core.run(encoded, settings)
        except _core.SimulationError as error:
            raise SimulationError(str(error)) from None
        finally:
            self.excursions = tuple(
                Excursion.model_validate(found) for found in self._core.excursions()
            )
        return self._result(outcome)

    def run_plan(self, plan: Plan) -> dict[str, results.Result]:
        """Runs every analysis of `plan`, by name, with its options.

        Raises:
            ValueError: When the plan gives a temperature other than the simulation's.
        """
        if plan.temperature is not None:
            temperature = plan.temperature.to(TEMPERATURE, "a plan's temperature")
            if temperature != self.temperature:
                message = (
                    f"the plan's temperature, {plan.temperature}, is not the "
                    f"simulation's, {Quantity(self.temperature, TEMPERATURE)}"
                )
                raise ValueError(message)
        return {
            name: self.run(analysis, plan.options)
            for name, analysis in plan.analyses.items()
        }

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

    def _result(self, outcome: _core.Outcome) -> results.Result:
        unknowns, nets = self.unknowns, self.nets
        match outcome["kind"]:
            case "operating_point":
                return self._point(outcome)
            case "operating_points":
                points = tuple(self._point(point) for point in outcome["points"])
                return results.OperatingPoints(unknowns, points)
            case "dc_sweep":
                return results.Sweep(
                    unknowns,
                    outcome["source"],
                    outcome["quantity"],
                    outcome["levels"],
                    outcome["values"],
                    nets,
                    outcome["logic"],
                    **self._observed(outcome),
                )
            case "ac":
                return results.Response(
                    unknowns,
                    outcome["source"],
                    outcome["frequencies"],
                    outcome["values"],
                    self._point(outcome["operating_point"]),
                    terminals=self.terminals,
                    currents=outcome["currents"],
                    composites=self.composites,
                )
            case "transient":
                statistics = results.Statistics(**outcome["statistics"])
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
                    **self._observed(outcome),
                )
            case _:
                return self._linear(outcome)

    def _linear(
        self,
        outcome: _core.SensitivityOutcome
        | _core.StabilityOutcome
        | _core.NoiseOutcome
        | _core.PoleZeroOutcome,
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
                differentiated = self._result(outcome["result"])
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
                    self._point(outcome["operating_point"]),
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
                    self._point(outcome["operating_point"]),
                )
            case "pole_zero":
                return results.PolesZeros(
                    unknowns,
                    outcome["poles"],
                    outcome["zeros"],
                    outcome["source"],
                    outcome["output"],
                    outcome["reference"],
                    self._point(outcome["operating_point"]),
                )

    def _point(self, solved: _core.SolvedPoint) -> results.OperatingPoint:
        return results.OperatingPoint(
            self.unknowns,
            solved["values"],
            solved["strategy"],
            self.nets,
            solved["logic"],
            **self._observed(solved),
        )

    def _observed(self, outcome: _core.Observed) -> results.Observed:
        """What every device did in `outcome`, as a result's fields."""
        return {
            "terminals": self.terminals,
            "currents": outcome["currents"],
            "charges": outcome["charges"],
            "composites": self.composites,
            "derivatives": self.derivatives,
            "conductances": outcome["conductances"],
            "capacitances": outcome["capacitances"],
            "variables": self.variables,
            "readings": outcome["variables"],
        }


_UNITS: dict[ir.Stress, str] = {"voltage": "V", "current": "A", "power": "W"}


def _instances(
    circuit: ir.Circuit, component: str, prefix: str
) -> Iterator[tuple[str, ir.Instance]]:
    """Every instance within `component`, by path, each before those it places."""
    for instance in circuit.components[component].instances:
        path = f"{prefix}{instance.id}"
        yield path, instance
        if isinstance(instance.of, ir.ComponentOf):
            yield from _instances(circuit, instance.of.component, f"{path}/")


@dataclass(frozen=True, slots=True)
class _Terminals:
    """The voltage at and the current into each reached terminal, over a result."""

    reached: Mapping[str, tuple[str | None, tuple[int, ...]]]
    columns: Mapping[str, int]
    """The column of each unknown's values, by name."""
    values: results.Real
    currents: results.Real
    terminals: int
    """The number of device terminals, which the currents have a column each for, if
    the result holds them."""

    def stress(
        self, path: str, instance: ir.Instance, rating: ir.Rating
    ) -> results.Real | None:
        """The stress `rating` limits on the instance at `path`, at each point.

        There is none when a rated terminal is on a logic net, or the result holds no
        currents and the stress needs them.
        """
        named = rating.terminals or tuple(
            terminal
            for terminal in instance.terminals
            if f"{path}.{terminal}" in self.reached
        )
        if any(f"{path}.{terminal}" not in self.reached for terminal in named):
            return None
        if rating.stress == "voltage":
            first, second = named
            return self._voltage(path, first) - self._voltage(path, second)
        if self.currents.shape[1] != self.terminals:
            return None
        if rating.stress == "current":
            (terminal,) = named
            return self._current(path, terminal)
        return sum(
            (
                self._voltage(path, terminal) * self._current(path, terminal)
                for terminal in named
            ),
            start=np.zeros(self.values.shape[0]),
        )

    def _voltage(self, path: str, terminal: str) -> results.Real:
        node, _ = self.reached[f"{path}.{terminal}"]
        if node is None:
            return np.zeros(self.values.shape[0])
        return self.values[:, self.columns[node]]

    def _current(self, path: str, terminal: str) -> results.Real:
        _, columns = self.reached[f"{path}.{terminal}"]
        return self.currents[:, list(columns)].sum(axis=1)


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
        for point, bound, sign in (
            (high, self.rating.maximum, 1),
            (low, self.rating.minimum, -1),
        ):
            if bound is None:
                continue
            value = float(self.stress[point])
            if sign * value > sign * bound:
                found.append(self._diagnostic(checks.OVER_RATING, point, bound, value))
            elif sign * bound > 0 and sign * value > checks.DERATED * sign * bound:
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
        }[self.rating.stress]
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
