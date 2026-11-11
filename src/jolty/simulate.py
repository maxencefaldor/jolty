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
from typing import overload

from pydantic import JsonValue

from jolty import _core, checks, ir, results
from jolty.builder import Component, circuit
from jolty.errors import ElaborationError, SimulationError
from jolty.plan import Ac, Analysis, DcSweep, OperatingPoint, Options, Plan, Transient
from jolty.units import CURRENT, TEMPERATURE, VOLTAGE, Dimension, Quantity


class Simulation:
    """A circuit elaborated once, ready to run any number of analyses."""

    circuit: ir.Circuit
    """The circuit, in the canonical format."""

    unknowns: tuple[results.Unknown, ...]
    """The node voltages and branch currents the engine solves for, in its order."""

    devices: int
    """The number of devices, with every component instance unfolded."""

    temperature: float
    """The temperature the devices are simulated at, in kelvins."""

    def __init__(
        self,
        circuit_or_top: ir.Circuit | Component,
        temperature: Quantity | None = None,
    ) -> None:
        """Elaborates a circuit, or the circuit with a component at its top.

        Its devices are simulated at `temperature`, 27 °C unless given.

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
        self.temperature = (
            _core.REFERENCE_TEMPERATURE
            if temperature is None
            else temperature.to(TEMPERATURE, "a temperature")
        )
        try:
            self._core = _core.Simulator(
                self.circuit.model_dump_json(), self.temperature
            )
        except _core.ElaborationError as error:
            raise ElaborationError(error.args[0]) from None
        self.unknowns = tuple(
            results.Unknown(name, quantity) for name, quantity in self._core.unknowns
        )
        self.devices = self._core.devices

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
        if isinstance(analysis, DcSweep):
            return analysis.to_json(self._level(analysis))
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
        unknowns = self.unknowns
        match outcome["kind"]:
            case "operating_point":
                return results.OperatingPoint(
                    unknowns, outcome["values"], outcome["strategy"]
                )
            case "dc_sweep":
                return results.Sweep(
                    unknowns,
                    outcome["source"],
                    outcome["quantity"],
                    outcome["levels"],
                    outcome["values"],
                )
            case "ac":
                solved = outcome["operating_point"]
                point = results.OperatingPoint(
                    unknowns, solved["values"], solved["strategy"]
                )
                return results.Response(
                    unknowns,
                    outcome["source"],
                    outcome["frequencies"],
                    outcome["values"],
                    point,
                )
            case "transient":
                statistics = results.Statistics(**outcome["statistics"])
                return results.Waveforms(
                    unknowns, outcome["times"], outcome["values"], statistics
                )
