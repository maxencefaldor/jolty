# Copyright 2026 Maxence Faldor
"""Circuit elaboration, simulation, batching, and differentiation via the Rust engine.

A [`Simulation`][jolty.simulate.Simulation] elaborates a circuit once at a chosen
temperature (27 °C by default) and runs single analyses or parallel parameter batches.
An [`Elaborated`][jolty.simulate.Elaborated] circuit is elaborated once and reused
across parameter overrides without re-elaborating the topology.

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

import functools
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import overload

from pydantic import JsonValue

from jolty import _core, checks, devices, foundry, ir, results, specs
from jolty.builder import Component, circuit
from jolty.devices import Device, ParameterValue
from jolty.errors import ElaborationError, PlanError, SimulationError
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
)
from jolty.simulate import _batch, _decode, _stress, _validate
from jolty.simulate._batch import mismatch
from jolty.simulate._decode import Logged, _settings, kelvins, pool, threads
from jolty.simulate._function import (
    PERIOD,
    BatchPullback,
    Differentiable,
    Elaborated,
    Function,
    Gradient,
    Outputs,
    Parameters,
    Pullback,
    function,
    jvp,
    vjp,
)
from jolty.simulate._stress import Joined, Stressed
from jolty.simulate._validate import process_problems
from jolty.specs import Specs, validate_specs
from jolty.units import TEMPERATURE, Quantity

__all__ = [
    "PERIOD",
    "BatchPullback",
    "Differentiable",
    "Elaborated",
    "Function",
    "Gradient",
    "Joined",
    "Logged",
    "Outputs",
    "Parameters",
    "Pullback",
    "Simulation",
    "Stressed",
    "function",
    "jvp",
    "kelvins",
    "mismatch",
    "pool",
    "process_problems",
    "threads",
    "vjp",
]

# Re-exported for tests that clear the process-wide settings cache.
_ = _settings


class Simulation:
    """Elaborated circuit prepared to run any number of analyses.

    Each analysis result contains the circuit unknowns plus any recorded device and
    component quantities. Recording metadata is constructed lazily only for recorded
    instances.

    Circuits containing foundry PDK devices (such as SKY130 transistors) delegate
    execution to ngspice with kit models, as documented in
    [`foundry`][jolty.foundry].

    Each executed analysis, standalone or batched, appends a start entry to the
    configured ``analysis_log`` file when enabled ([`Logged`][jolty.simulate.Logged]).
    """

    circuit: ir.Circuit
    """Canonical circuit representation."""

    unknowns: tuple[results.Unknown, ...]
    """Node voltages and branch currents solved by the engine, in engine order."""

    nets: tuple[str, ...]
    """Logic nets simulated by the event scheduler, in engine order."""

    devices: int
    """Total unfolded device count across all component instances."""

    conditions: Conditions
    """Temperature, process corner, overrides, and mismatch seed for the simulation."""

    foundry: tuple[str, ...]
    """Paths of foundry devices delegated to ngspice instead of the native engine."""

    def __init__(  # noqa: PLR0913 - the circuit, and what it runs at
        self,
        circuit_or_top: ir.Circuit | Component,
        temperature: Quantity | None = None,
        overrides: Mapping[str, Mapping[str, ParameterValue]] | None = None,
        substitutes: Mapping[str, Device] | None = None,
        *,
        process: str | None = None,
        mismatch: int | None = None,
    ) -> None:
        """Elaborates a circuit or a circuit rooted at a top component.

        Raises:
            ElaborationError: Lists every problem reported by the engine, naming the
                instance and build-script line.
            DimensionError: When `temperature` does not carry temperature dimensions or
                a parameter override has the wrong dimension or kind.
            ValueError: When a process corner is supplied for a circuit containing no
                foundry devices.
        """
        built = (
            circuit_or_top
            if isinstance(circuit_or_top, ir.Circuit)
            else circuit(circuit_or_top)
        )
        converted = devices.convert_overrides(built, overrides)
        with _decode.elaborating():
            core = _core.Simulator(
                built.model_dump_json(),
                kelvins(temperature),
                json.dumps(converted, allow_nan=False),
                _decode.substitutes_json(substitutes),
            )
        self._start(
            built,
            core,
            (temperature, overrides, frozenset(substitutes or {})),
            (process, mismatch),
        )

    @classmethod
    def _from_core(
        cls,
        built: ir.Circuit,
        core: _core.Simulator,
        replaced: tuple[
            Quantity | None,
            Mapping[str, Mapping[str, ParameterValue]] | None,
            frozenset[str],
        ],
        foundry_conditions: tuple[str | None, int | None],
    ) -> Simulation:
        """Constructs a `Simulation` around an already-created engine `core`."""
        simulation = cls.__new__(cls)
        cls._start(simulation, built, core, replaced, foundry_conditions)
        return simulation

    @classmethod
    def from_elaborated(
        cls,
        elaborated: Elaborated,
        overrides: Mapping[str, Mapping[str, ParameterValue]] | None = None,
        *,
        process: str | None = None,
        mismatch: int | None = None,
    ) -> Simulation:
        """Creates a simulation from `elaborated` with parameter `overrides` applied.

        Args:
            elaborated: Previously elaborated circuit topology and temperature.
            overrides: Nested mapping of instance paths to parameter overrides validated
                against each device's declared parameter dimensions.
            process: Optional foundry process corner name.
            mismatch: Optional foundry statistical mismatch seed.

        Returns:
            Simulation configured with `overrides`, `process`, and `mismatch`.

        Raises:
            DimensionError: When an override for a known parameter has the wrong
                physical dimension or kind.
            ElaborationError: Lists unknown parameters in `overrides` or devices whose
                overridden parameters fall out of range.
            ValueError: When a process corner is supplied for a circuit containing no
                foundry devices.
        """
        core = elaborated.simulator(overrides)
        replaced = (elaborated.temperature, overrides, elaborated.substituted)
        return cls._from_core(elaborated.circuit, core, replaced, (process, mismatch))

    def _start(
        self,
        built: ir.Circuit,
        core: _core.Simulator,
        replaced: tuple[
            Quantity | None,
            Mapping[str, Mapping[str, ParameterValue]] | None,
            frozenset[str],
        ],
        foundry_conditions: tuple[str | None, int | None],
    ) -> None:
        """Readies the simulation of `built` that the engine's `core` runs."""
        temperature, overrides, substituted = replaced
        process, mismatch_seed = foundry_conditions
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
            Quantity(kelvins(temperature), TEMPERATURE),
            process,
            {path: dict(values) for path, values in (overrides or {}).items()},
            mismatch_seed,
        )

    @functools.cached_property
    def instances(self) -> tuple[str, ...]:
        """Every path a record can name, in the engine's order."""
        return tuple(self._core.instances)

    @functools.cached_property
    def _catalog(self) -> _validate.Catalog:
        """Static catalog of devices, sources, probes, nets, and unknowns."""
        return _validate.build_catalog(
            self.circuit,
            self._substituted,
            self.unknowns,
            self.nets,
            overrides=self.conditions.overrides,
            joined=self.joined,
        )

    def validate_analysis(
        self, analysis: Analysis, options: Options | None = None
    ) -> None:
        """Checks `analysis` and `options` against this circuit before running.

        Args:
            analysis: Analysis specification to validate against circuit topology.
            options: Optional solver tolerances, iteration limits, and nodesets.

        Raises:
            DimensionError: When the sole problem is a DC sweep level's dimension.
            PlanError: With every problem found, each naming close valid names.
        """
        chosen = Options() if options is None else options
        problems = _validate.nodeset_problems(chosen.nodesets, "nodeset", self._catalog)
        found, dimension_error = _validate.analysis_problems(
            analysis,
            self._catalog,
            self._record_problems,
            has_logic_nets=bool(self.nets),
        )
        problems.extend(found)
        if dimension_error is not None and len(problems) == 1:
            raise dimension_error
        if problems:
            raise PlanError(problems)

    def validate_plan(
        self,
        plan: Plan,
        *,
        specs: Specs | None = None,
        traces: Sequence[str] = (),
        check_temperature: bool = False,
    ) -> None:
        """Checks every name and setting `plan`, `specs`, and `traces` name upfront.

        Args:
            plan: Simulation plan whose analyses, options, fidelity scopes, and
                variants are validated.
            specs: Optional specification requirements validated against planned
                analyses.
            traces: Unknown or logic net names requested for plotting or batch tracing.
            check_temperature: Flag enforcing equality between `plan.temperature` and
                the simulation ambient temperature.

        Raises:
            PlanError: With every problem found, each naming close valid names.
        """
        catalog = self._catalog
        problems = _validate.plan_condition_problems(
            plan,
            temperature=self.conditions.temperature,
            has_foundry=bool(self.foundry),
            check_temperature=check_temperature,
        )
        problems.extend(
            _validate.nodeset_problems(plan.options.nodesets, "nodeset", catalog)
        )
        problems.extend(
            _validate.fidelity_scope_problems(plan.fidelity, catalog.component_paths)
        )
        if plan.record is not None:
            problems.extend(self._record_problems(plan.record))
        analysis_found, analysis_dimension_error = _validate.analyses_problems(
            plan.analyses,
            catalog,
            self._record_problems,
            has_logic_nets=bool(self.nets),
        )
        problems.extend(analysis_found)
        shapes = {
            name: _decode.analysis_shape(self.circuit, self.unknowns, analysis)
            for name, analysis in plan.analyses.items()
        }
        unknown_units = {
            name
            for name, analysis in plan.analyses.items()
            if isinstance(analysis, Noise)
            and _validate.noise_unit_unknown(analysis, catalog)
        }
        variant_dimension_error = None
        if plan.variants is not None:
            variant_found, variant_dimension_error = _validate.variant_problems(
                plan.variants, catalog
            )
            problems.extend(variant_found)
        if specs is not None:
            problems.extend(
                validate_specs(
                    specs,
                    tuple(plan.analyses),
                    shapes,
                    unknown_units=unknown_units,
                )
            )
        if traces:
            problems.extend(
                _validate.trace_problems(traces, (*catalog.unknown_names, *self.nets))
            )
        sole_dimension_error = analysis_dimension_error or variant_dimension_error
        if sole_dimension_error is not None and len(problems) == 1:
            raise sole_dimension_error
        if problems:
            raise PlanError(problems)

    def joined(self, instance: str, terminal: str) -> Joined | None:
        """Returns the node and thermal status for `terminal` on `instance`, if any.

        Args:
            instance: Hierarchical device, component, or logic instance path.
            terminal: Terminal or port name on `instance`.

        Returns:
            Connected node descriptor, or ``None`` when `terminal` is unconnected or
            logic-only.
        """
        found = self._core.terminal(instance, terminal)
        return None if found is None else Joined(*found)

    def check(self) -> tuple[checks.Diagnostic, ...]:
        """Runs structural checks and returns all diagnostics at default levels.

        Returns:
            Structural check diagnostics produced by the engine.
        """
        return checks.diagnostics(self._core.check())

    def check_operating_point(
        self, options: Options | None = None
    ) -> tuple[checks.Diagnostic, ...]:
        """Runs operating-point rule checks and returns every resulting diagnostic.

        Args:
            options: Optional solver tolerances, iteration limits, and nodesets.

        Returns:
            Operating-point structural diagnostics produced by the engine.

        Raises:
            SimulationError: When the circuit has no operating point.
        """
        chosen = Options() if options is None else options
        problems = _validate.nodeset_problems(chosen.nodesets, "nodeset", self._catalog)
        if problems:
            raise PlanError(problems)
        settings = json.dumps(chosen.to_json(), allow_nan=False)
        try:
            found = self._core.check_operating_point(settings, _decode.engine_pool())
        except _core.SimulationError as error:
            raise SimulationError(str(error)) from None
        return checks.diagnostics(found)

    def check_results(
        self, solved: Mapping[str, results.AnalysisResult]
    ) -> tuple[checks.Diagnostic, ...]:
        """Evaluates result-level rules on `solved` and returns all diagnostics.

        Args:
            solved: Mapping from analysis names to completed analysis results.

        Returns:
            Stability-margin and device-rating diagnostics across `solved`.
        """
        return _stress.check_results(
            self.circuit, self._substituted, self.conditions, self.joined, solved
        )

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
        """Runs `analysis` with `options`, or with engine defaults when omitted.

        Args:
            analysis: Analysis specification to solve on the elaborated circuit.
            options: Optional solver tolerances, iteration limits, and nodesets.

        Returns:
            Typed analysis result matching the kind of `analysis`.

        Raises:
            SimulationError: When the analysis fails or `record` names a nonexistent
                instance.
            JoltyError: When settings are invalid or the analysis log cannot be
                written.
        """
        chosen = Options() if options is None else options
        self.validate_analysis(analysis, chosen)
        return self._execute(analysis, chosen)

    def _execute(self, analysis: Analysis, chosen: Options) -> results.AnalysisResult:
        """Runs an already-validated `analysis` with `chosen` options."""
        if self.foundry:
            _decode.log_analyses([analysis], 1)
            return foundry.run(
                self.circuit, analysis, chosen, self.conditions, self.unknowns
            )
        encoded = json.dumps(
            _decode.encode_analysis(self.circuit, self._substituted, analysis),
            allow_nan=False,
        )
        record = self._record(analysis)
        regions = isinstance(analysis, Recorded) and analysis.regions
        settings = json.dumps(
            {**chosen.to_json(), "record": record, "regions": regions}, allow_nan=False
        )
        recording = self._recording(json.dumps(record), regions=regions)
        _decode.log_analyses([analysis], 1)
        try:
            outcome = self._core.run(encoded, settings, _decode.engine_pool())
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

        Args:
            analyses: Mapping from analysis names to specifications run per variant.
            variants: Parameter sweep, Monte Carlo, explicit, or product variants.
            options: Optional solver tolerances, iteration limits, and nodesets.
            measures: Measures evaluated across every run of the batch.
            traces: Circuit unknown names whose waveforms are retained across runs.
            each: Optional callback invoked with each decoded `Run` upon completion.
            runs: Optional contiguous slice of variant indices to execute.

        Returns:
            Aggregated batch result containing parameter values, measures, and traces.

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
        problems = _validate.nodeset_problems(chosen.nodesets, "nodeset", self._catalog)
        analysis_found, sole_dimension_error = _validate.analyses_problems(
            analyses,
            self._catalog,
            self._record_problems,
            has_logic_nets=bool(self.nets),
        )
        problems.extend(analysis_found)
        if sole_dimension_error is not None and len(problems) == 1:
            raise sole_dimension_error
        if problems:
            raise PlanError(problems)
        planned = _batch.Planned.from_host(
            self._batch_host(), analyses, chosen, measures, traces
        )
        ran = self._variants(variants)
        taken = range(len(ran)) if runs is None else runs
        if taken.step != 1 or not 0 <= taken.start <= taken.stop <= len(ran):
            message = f"{taken} is not a range of the variants' {len(ran)} runs"
            raise ValueError(message)
        _decode.log_analyses([analyses[name] for name in planned.engine], len(taken))
        found = (
            planned.in_ngspice(ran, taken, each)
            if self.foundry
            else planned.in_engine(ran, taken, each)
        )
        return planned.result(found)

    def varied(self, variants: Variants) -> tuple[str, ...]:
        """Returns the column labels for parameters varied by `variants`.

        Args:
            variants: Parameter variation specification to inspect.

        Returns:
            Ordered parameter column labels such as ``"R1 resistance"``.
        """
        return _batch.labels(self._variants(variants).parameters)

    def run_plan(
        self, plan: Plan, *, specs: Specs | None = None
    ) -> dict[str, results.AnalysisResult]:
        """Runs every analysis of `plan`, by name, with its options.

        Checks `plan` (and `specs`, when provided) once against this circuit before
        running the first analysis.

        Args:
            plan: Simulation plan whose analyses and shared options are executed.
            specs: Optional specifications validated upfront before execution.

        Returns:
            Mapping from analysis names in `plan` to solved analysis results.

        Raises:
            PlanError: With every problem found in `plan` or `specs`, including a
                temperature other than the simulation's.
            SimulationError: When an analysis fails to solve.
        """
        self.validate_plan(plan, specs=specs, check_temperature=True)
        return {
            name: self._execute(
                replace(analysis, record=plan.record)
                if isinstance(analysis, Recorded)
                and analysis.record is None
                and plan.record is not None
                else analysis,
                plan.options,
            )
            for name, analysis in plan.analyses.items()
        }

    def _decoder(
        self,
    ) -> tuple[
        Callable[[str], results.Recording],
        Callable[[_core.AnalysisResult, results.Recording], results.AnalysisResult],
    ]:
        """Returns `(recording_for, decode)` bound to this simulation instance."""
        return (
            lambda record: self._recording(record, regions=False),
            self._result,
        )

    @classmethod
    def _decoder_of(
        cls, view: Simulation
    ) -> tuple[
        Callable[[str], results.Recording],
        Callable[[_core.AnalysisResult, results.Recording], results.AnalysisResult],
    ]:
        """Returns `(recording_for, decode)` bound to `view`."""
        return cls._decoder(view)

    def _batch_host(self) -> _batch.BatchHost:
        """Builds the `BatchHost` adapter for `Planned`."""
        return _batch.BatchHost(
            circuit=self.circuit,
            unknowns=self.unknowns,
            conditions=self.conditions,
            foundry=self.foundry,
            core=self._core,
            substituted=self._substituted,
            record_for=self._record,
            decoder=self._decoder(),
            view_for=lambda core: self._decoder_of(self._view(core)),
        )

    def _record(self, analysis: Analysis) -> JsonValue:
        """Encodes the resolved instance recording specification for `analysis`."""
        named = tuple(analysis.record or ()) if isinstance(analysis, Recorded) else ()
        if named not in self._records:
            self._records[named] = _validate.resolve_record(
                named, self.instances, self._known
            )
        found = self._records[named]
        if isinstance(analysis, OperatingPoint | Transient | PeriodicSteadyState):
            found |= frozenset(self._rated)
        if found == self._known:
            return "all"
        instances: list[JsonValue] = [*sorted(found)]
        return {"instances": instances}

    def _record_problems(self, record: Sequence[str]) -> list[str]:
        """Collects unknown instance paths and unmatched patterns in `record`."""
        return _validate.record_problems(record, self.instances, self._known)

    @functools.cached_property
    def _known(self) -> frozenset[str]:
        """Set of all recordable instance paths in the simulation."""
        return frozenset(self.instances)

    @functools.cached_property
    def _rated(self) -> tuple[str, ...]:
        """Paths of instances rated for current or power limits."""
        return tuple(
            path
            for path, instance in _stress.rated_instances(
                self.circuit, self._substituted
            )
            if any(rating.stress in {"current", "power"} for rating in instance.ratings)
        )

    @functools.cached_property
    def transistors(self) -> tuple[str, ...]:
        """Paths of devices whose models report operating regions."""
        recording = self._recording(json.dumps("all"), regions=False)
        return tuple(device.path for device in recording.devices if device.regions)

    def _recording(self, record: str, *, regions: bool) -> results.Recording:
        """Builds and caches recording metadata for JSON `record` and `regions`."""
        found = self._recordings.get((record, regions))
        if found is not None:
            return found
        found = _decode.build_recording(
            self._core, self.unknowns, record, regions=regions
        )
        self._recordings[record, regions] = found
        return found

    def _result(
        self, outcome: _core.AnalysisResult, recording: results.Recording
    ) -> results.AnalysisResult:
        """Decodes engine `outcome` using `recording` metadata."""
        return _decode.decode_result(self.unknowns, self.nets, outcome, recording)

    def _variants(self, variants: Variants) -> _core.Variants:
        """Compiles `variants` into the engine representation for this circuit."""
        problems, sole_dimension_error = _validate.variant_problems(
            variants, self._catalog
        )
        if sole_dimension_error is not None:
            raise sole_dimension_error
        if problems:
            raise ElaborationError(problems)
        elaborated = self._core.elaborated
        with _decode.elaborating():
            match variants:
                case MonteCarlo(samples=samples, seed=seed):
                    toleranced = json.dumps(
                        _decode.toleranced_payload(self.circuit, self._substituted),
                        allow_nan=False,
                    )
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
                            json.dumps(
                                devices.convert_overrides(
                                    self.circuit, variant.overrides
                                ),
                                allow_nan=False,
                            ),
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

    def _view(self, core: _core.Simulator) -> Simulation:
        """Wraps an engine `core` view of the same circuit in a `Simulation`."""
        replaced = (
            self.conditions.temperature,
            self.conditions.overrides,
            self._substituted,
        )
        return Simulation._from_core(self.circuit, core, replaced, (None, None))
