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
from collections.abc import Mapping
from typing import Literal, overload

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from jolty import _core, checks, ir, results
from jolty.builder import Component, circuit
from jolty.devices import Device
from jolty.errors import ElaborationError, SimulationError
from jolty.plan import (
    Ac,
    Analysis,
    DcSweep,
    OperatingPoint,
    Options,
    Plan,
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

    composites: results.Composites
    """Each terminal of a component instance or a logic device, such as ``X1.vdd``, with
    the indices of the device terminals whose currents and charges sum to its own."""

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
            raise ElaborationError(error.args[0]) from None
        self.unknowns = tuple(
            results.Unknown(name, quantity) for name, quantity in self._core.unknowns
        )
        self.nets = tuple(self._core.nets)
        self.terminals = tuple(self._core.terminals)
        self.composites = tuple(
            (name, tuple(members)) for name, members in self._core.composites
        )
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

    @overload
    def run(
        self, analysis: OperatingPoint, options: Options | None = None
    ) -> results.OperatingPoint: ...
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
