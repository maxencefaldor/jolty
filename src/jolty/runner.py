# Copyright 2026 Maxence Faldor
"""Subprocess runner for executing circuit build scripts and simulation plans.

The service launches ``python -P -m jolty.runner DIRECTORY`` for every command that
executes user scripts. The runner reads the [`Job`][jolty.runner.Job] from
``DIRECTORY/job.json``, invokes ``build()``, and elaborates the returned circuit.
When the job specifies a plan script, the runner calls ``plan()`` and executes every
analysis at nominal plan conditions and then at each corner. Corners with custom
arguments rebuild the circuit before simulation. Every run uses logic models selected
by plan fidelity (characterized once each), while spot checks re-simulate sampled
instances at full circuit fidelity.

When the plan script defines ``specs()``, the runner evaluates every requirement
across nominal analyses, corners, Monte Carlo samples, and variant runs. Output
artifacts are written to ``DIRECTORY/outcome.json`` as a
[`JobResult`][jolty.runner.JobResult], ``DIRECTORY/results.npz`` for nominal analysis
results, and ``DIRECTORY/batches.npz`` for nominal Monte Carlo and variant batches.

A job requesting a Monte Carlo sample range runs only those samples as a shard
([`jolty.shards`][jolty.shards]) across nominal and corner conditions. Shard runs
skip standalone nominal analyses and write their output to ``DIRECTORY/shard.npz``.

All non-nominal runs execute in multithreaded engine batches over shared circuits.
Conditions that share an identical elaborated circuit, such as temperature-only
corners, reuse one circuit instance and pass temperatures as batch run values.
Corners with custom build arguments elaborate and simulate on a separate circuit.

Running scripts in an isolated subprocess protects the CLI and MCP server standard
output channels. Subprocess isolation captures script print output, enforces timeouts
on runaway scripts or analyses, and confines crashes to the worker process.
"""

# Jolty modules load lazily on first use, keeping worker startup lean.
__lazy_modules__ = ["jolty"]

import json
import runpy
import sys
import sysconfig
import traceback
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Annotated, Final, Literal, override

import numpy as np
from pydantic import Field, JsonValue, TypeAdapter

from jolty import checks, fidelity, files, ir, report, shards, unknowns, variation
from jolty.builder import Component, circuit
from jolty.devices import ParameterValue
from jolty.errors import (
    ElaborationError,
    JoltyError,
    PlanError,
    ScriptError,
    SimulationError,
)
from jolty.fidelity import FidelityReport
from jolty.ir import FrozenModel
from jolty.plan import (
    Analysis,
    Explicit,
    MonteCarlo,
    OperatingPoint,
    Plan,
    Product,
    Recorded,
    Variant,
    Variants,
    WorstCase,
    runs,
)
from jolty.results import AnalysisResult, BatchResult, Run
from jolty.simulate import Simulation, kelvins, process_problems
from jolty.specs import (
    Requirement,
    Specs,
    Verdict,
    evaluate,
    judge,
    reading,
    validate_specs,
)
from jolty.units import TEMPERATURE, Quantity

JOB: Final = "job.json"
"""Filename of the serialized job specification inside a job directory."""

OUTCOME: Final = "outcome.json"
"""Filename of the serialized `JobResult` inside a job directory."""

RESULTS: Final = "results.npz"
"""Filename of the nominal analysis results archive inside a job directory."""

BATCHES: Final = "batches.npz"
"""Filename of the nominal batch archive inside a job directory."""

SHARD: Final = "shard.npz"
"""Filename of the Monte Carlo shard archive inside a job directory."""

MONTE_CARLO: Final = "monte-carlo"
"""Batch archive key for a plan's Monte Carlo analysis (not a valid identifier)."""

VARIANT_RUNS: Final = "variant-runs"
"""Batch archive key for a plan's variant runs (not a valid identifier)."""

_LIBRARIES: Final = tuple(
    {
        Path(__file__).resolve().parent,
        *(
            Path(sysconfig.get_path(name)).resolve()
            for name in ("stdlib", "platstdlib", "purelib", "platlib")
        ),
    }
)
"""Directory roots excluded from script tracebacks: jolty, stdlib, and packages."""


class Thresholds(FrozenModel):
    """Parallelism threshold overrides supplied by a job in place of plan defaults."""

    min_solves_per_task: int | None = Field(default=None, ge=1)
    min_instances_per_task: int | None = Field(default=None, ge=1)
    min_unknowns_per_task: int | None = Field(default=None, ge=1)
    max_unknowns_per_interior: int | None = Field(default=None, ge=1)


class Job(FrozenModel):
    """Serializable specification of a build, simulation, or shard runner job."""

    script: Path = Field(description="Path to the Python script defining build().")
    plan: Path | None = Field(
        default=None,
        description="Path to the script defining plan(), or null to elaborate only.",
    )
    max_unknowns: int = Field(
        ge=1, description="Maximum number of unknowns listed in each analysis report."
    )
    check: bool = Field(
        default=True,
        description="Flag enabling rule checks at the plan operating point.",
    )
    record: list[str] | None = Field(
        default=None,
        description="Instance patterns overriding plan recording, or null to keep "
        "plan settings.",
    )
    plot: bool = Field(
        default=False,
        description="Flag enabling batch trace collection for nominal condition plots.",
    )
    traces: list[str] = Field(
        default_factory=list,
        description="Unknown names traced by batches for plotting; defaults to "
        "top-level nets.",
    )
    parallelism: Thresholds = Field(
        default_factory=Thresholds,
        description="Per-threshold parallelism overrides applied on top of plan "
        "options.",
    )
    samples: tuple[int, int] | None = Field(
        default=None,
        description="Half-open Monte Carlo sample range for shard execution, or null "
        "for a full plan.",
    )


class Built(FrozenModel):
    """Successful build output with elaboration metadata, diagnostics, and reports."""

    kind: Literal["built"] = "built"
    circuit: ir.Circuit
    devices: int | None = Field(
        default=None,
        description="Unfolded primitive device count, or null when elaboration fails.",
    )
    unknowns: int | None = Field(
        default=None,
        description="Solved circuit unknown count, or null when elaboration fails.",
    )
    problems: tuple[str, ...] = Field(
        default=(),
        description="Elaboration error messages explaining why elaboration failed.",
    )
    diagnostics: tuple[checks.Diagnostic, ...] = Field(
        default=(),
        description="Default-level structural diagnostics followed by operating-point "
        "diagnostics.",
    )
    analyses: tuple[report.AnalysisReport, ...] = Field(
        default=(), description="Analysis reports for the plan, in definition order."
    )
    corners: tuple[report.CornerReport, ...] = Field(
        default=(), description="Corner reports for the plan, in definition order."
    )
    monte_carlo: report.MonteCarloReport | None = Field(
        default=None,
        description="Summary report for the plan Monte Carlo analysis, or null if "
        "absent.",
    )
    variants: report.VariantsReport | None = Field(
        default=None,
        description="Summary report for the plan variant runs, or null if absent.",
    )
    worst_case: report.WorstCaseReport | None = Field(
        default=None,
        description="Summary report for the plan worst-case analysis, or null if "
        "absent.",
    )
    shard: report.ShardReport | None = Field(
        default=None,
        description="Summary report for a sharded Monte Carlo run, or null for "
        "non-shard jobs.",
    )
    specs: tuple[Verdict, ...] = Field(
        default=(),
        description="Requirement verdicts in specs() order, or empty when specs() is "
        "undefined.",
    )
    fidelity: FidelityReport | None = Field(
        default=None,
        description="Characterized fidelity models and spot-check results, or null "
        "when unused.",
    )


class Failure(FrozenModel):
    """Structured error record describing why a runner script failed."""

    kind: Literal["failure"] = "failure"
    error: str = Field(description="Exception class name, such as 'BuildError'.")
    message: str = Field(description="Exception message string.")
    frames: tuple[ir.SourceLocation, ...] = Field(
        default=(),
        description="User-script traceback locations ordered innermost to outermost.",
    )

    @override
    def __str__(self) -> str:
        """Formats the script traceback, error type, and message for display."""
        head = f"{self.error}: {self.message}" if self.message else self.error
        if not self.frames:
            return head
        innermost, *callers = (str(frame) for frame in self.frames)
        return f"{innermost}: {head}" + "".join(
            f"\n  called from {caller}" for caller in callers
        )


type JobResult = Annotated[Built | Failure, Field(discriminator="kind")]
"""Discriminated union representing either a built result or a script failure."""

JOB_RESULTS: Final = TypeAdapter[JobResult](JobResult)
"""Pydantic `TypeAdapter` for serializing and deserializing `JobResult` JSON."""


@dataclass(slots=True)
class _Condition:
    """Nominal or corner condition paired with resolved fidelity and results."""

    corner: str | None
    """Corner name, or `None` for nominal plan conditions."""
    temperature: Quantity
    chosen: fidelity.Fidelity | None
    """Resolved fidelity state at this condition, or `None` on elaboration failure."""
    solved: Mapping[str, AnalysisResult] = field(default_factory=dict)
    """Completed analysis results at this condition, keyed by analysis name."""
    failures: Mapping[str, str] = field(default_factory=dict)
    """Failure messages for analyses that errored at this condition, by name."""


@dataclass(frozen=True, slots=True)
class _RunContext:
    """Shared context for executing a validated plan across corners and batches."""

    job: Job
    directory: Path
    built: ir.Circuit
    plan: Plan
    specs: Specs | None
    chosen: fidelity.Fidelity
    simulation: Simulation
    conditions: tuple[_Condition, ...] = ()


@dataclass(slots=True)
class _Structure:
    """Group of conditions sharing one elaborated circuit and its `Simulation`.

    Grouped conditions differ only in temperature, so batch runs elaborate the shared
    circuit once and pass each condition's temperature as a run parameter.
    """

    simulation: Simulation
    members: list[int]
    """Indices of grouped conditions within the plan's condition list."""


def run(job: Job, directory: Path) -> JobResult:
    """Executes `job` in the current process and writes outputs into `directory`.

    Script directories are prepended to `sys.path` during execution so scripts can
    import sibling modules just as under ``python script.py``.

    Args:
        job: Subprocess build or simulation job specification.
        directory: Scratch directory where results, batches, or shards are staged.

    Returns:
        `Built` outcome on completion, or `Failure` recording any script exception.
    """
    try:
        with _importable(job.script, job.plan):
            return _run(job, directory)
    except SystemExit as error:
        return _failure(error, f"the script exited with {error.code!r}")
    except BaseException as error:  # noqa: BLE001 - a script may raise anything
        return _failure(error, str(error))


def _precheck_fidelity(simulation: Simulation, plan: Plan, specs: Specs | None) -> None:
    """Validates fidelity scopes, process corners, and spec names before modeling."""
    problems: list[str] = []
    try:
        fidelity.scopes(simulation.circuit, plan.fidelity)
    except PlanError as error:
        problems.extend(error.problems)
    problems.extend(process_problems(plan, has_foundry=bool(simulation.foundry)))
    if specs is not None:
        problems.extend(validate_specs(specs, tuple(plan.analyses), {}))
    if problems:
        raise PlanError(problems)


def _prepare_simulation(
    simulation: Simulation,
    plan: Plan,
    specs: Specs | None,
    *,
    traces: Sequence[str],
    cache: fidelity.Cache,
) -> tuple[fidelity.Fidelity, Simulation]:
    """Resolves `plan`'s fidelity and validates `plan`, `specs`, and `traces`."""
    if plan.fidelity:
        _precheck_fidelity(simulation, plan, specs)
    elif plan.process is not None and not simulation.foundry:
        # Report the invalid process corner alongside every other plan/spec problem.
        simulation.validate_plan(plan, specs=specs, traces=traces)
    chosen = fidelity.resolve(
        simulation.circuit, plan, plan.temperature, cache, plan.process
    )
    active = simulation
    if plan.temperature is not None or plan.process is not None or chosen.matched:
        active = chosen.simulation()
    active.validate_plan(plan, specs=specs, traces=traces)
    return chosen, active


def _run(job: Job, directory: Path) -> Built:
    namespaces: dict[Path, dict[str, object]] = {}
    built = _built(job.script, namespaces, {})
    try:
        simulation = Simulation(built)
    except ElaborationError as error:
        return Built(circuit=built, problems=error.problems)
    elaborated = Built(
        circuit=built,
        devices=simulation.devices,
        unknowns=len(simulation.unknowns),
        diagnostics=simulation.check(),
    )
    if job.plan is None:
        return elaborated
    plan = _call(job.plan, "plan", namespaces)
    if not isinstance(plan, Plan):
        message = f"plan() returned {type(plan).__name__}, not a Plan"
        raise ScriptError(message)
    if job.record is not None:
        plan = replace(plan, record=job.record)
    if given := job.parallelism.model_dump(exclude_none=True):
        spread = replace(plan.options.parallelism, **given)
        plan = replace(plan, options=replace(plan.options, parallelism=spread))
    specs = _specs(job.plan, namespaces)
    cache: fidelity.Cache = {}
    try:
        chosen, simulation = _prepare_simulation(
            simulation, plan, specs, traces=job.traces, cache=cache
        )
    except ElaborationError as error:
        return Built(circuit=built, problems=error.problems)
    context = _RunContext(job, directory, built, plan, specs, chosen, simulation)
    if job.samples is not None:
        # Shard jobs skip standalone nominal analyses.
        conditions, _ = _corners(context, namespaces, cache, {})
        shard = _shard(replace(context, conditions=conditions))
        return elaborated.model_copy(update={"shard": shard})
    outside: list[fidelity.Outside] = []
    solved, reports = _analyses(
        simulation, plan, job.max_unknowns, outside, recorded=True
    )
    at_point = (
        _at_operating_point(simulation, plan) + simulation.check_results(solved)
        if job.check
        else ()
    )
    if solved:
        files.save_results(directory / RESULTS, solved, circuit=built, plan=plan)
    conditions, corners = _corners(context, namespaces, cache, solved)
    context = replace(context, conditions=conditions)
    at_corners = {
        condition.corner: condition.solved
        for condition in conditions
        if condition.corner is not None
    }
    verdicts = (
        () if specs is None else evaluate(specs, plan.analyses, solved, at_corners)
    )
    checked = _fidelity(context, solved)
    monte_carlo, variants = _batched(context)
    worst_case = (
        None if plan.worst_case is None else _worst_case(context, plan.worst_case)
    )
    return elaborated.model_copy(
        update={
            "diagnostics": elaborated.diagnostics + at_point,
            "analyses": reports,
            "corners": corners,
            "monte_carlo": monte_carlo,
            "variants": variants,
            "worst_case": worst_case,
            "specs": verdicts,
            "fidelity": checked
            and checked.model_copy(update={"outside": tuple(outside)}),
        }
    )


def _batched(
    context: _RunContext,
) -> tuple[report.MonteCarloReport | None, report.VariantsReport | None]:
    """Executes plan Monte Carlo and variant batches and returns summary reports.

    Nominal condition batches, including plot traces when enabled, are written to
    ``context.directory``.
    """
    traces = _traces(context)
    batches: dict[str, files.Batch] = {}
    monte_carlo = None
    if context.plan.monte_carlo is not None:
        monte_carlo, batches[MONTE_CARLO] = _monte_carlo(
            context, context.plan.monte_carlo, traces=traces
        )
    variants = None
    if context.plan.variants is not None:
        variants, batches[VARIANT_RUNS] = _variants(
            context, context.plan.variants, traces=traces
        )
    if batches:
        files.save_batches(context.directory / BATCHES, batches)
    return monte_carlo, variants


def _at_operating_point(
    simulation: Simulation, plan: Plan
) -> tuple[checks.Diagnostic, ...]:
    """Returns rule diagnostics evaluated at the plan operating point.

    Circuits without a valid operating point yield no diagnostics here; failing
    analyses report the underlying solver error separately.
    """
    try:
        return simulation.check_operating_point(plan.options)
    except SimulationError:
        return ()


def _fidelity(
    context: _RunContext, solved: Mapping[str, AnalysisResult]
) -> FidelityReport | None:
    """Builds a `FidelityReport` with characterized models and spot checks of `solved`.

    Each spot check validates a sampled instance against its fidelity contract and
    re-evaluates `context.specs` to identify verdicts sensitive to fidelity. Returns
    `None` when no component instances match a characterized model.
    """
    chosen, plan, specs = context.chosen, context.plan, context.specs
    if not chosen.matched:
        return None
    contracts = []
    dependences: list[fidelity.Dependence] = []
    modeled = () if specs is None else evaluate(specs, plan.analyses, solved)
    for path in fidelity.sampled(chosen, plan):
        try:
            opened, _ = _analyses(chosen.simulation(opened=path), plan, 1)
        except ElaborationError:
            opened = {}
        contracts.append(fidelity.contract(chosen, plan, path, solved, opened))
        if specs is not None:
            exact = evaluate(specs, plan.analyses, opened)
            dependences.extend(fidelity.dependences(path, modeled, exact))
    return FidelityReport(
        models=fidelity.models(chosen),
        contracts=tuple(contracts),
        dependences=tuple(dependences),
    )


def _built(
    script: Path,
    namespaces: dict[Path, dict[str, object]],
    arguments: Mapping[str, object],
) -> ir.Circuit:
    """Invokes ``build()`` in `script` with `arguments` and returns an `ir.Circuit`.

    Raises:
        ScriptError: When ``build()`` returns neither a `Component` nor a `Circuit`.
    """
    match _call(script, "build", namespaces, arguments=arguments):
        case Component() as top:
            return circuit(top)
        case ir.Circuit() as built:
            return built
        case other:
            message = (
                f"build() returned {type(other).__name__}, not a Component or a Circuit"
            )
            raise ScriptError(message)


def _kelvins(temperature: Quantity | None) -> Quantity:
    """Converts `temperature` to a kelvin `Quantity`, defaulting `None` to 27 °C."""
    return Quantity(kelvins(temperature), TEMPERATURE)


def _unrecorded(plan: Plan) -> dict[str, Analysis]:
    """Returns copies of plan analyses with recording cleared beyond check needs."""
    return {
        name: _recording(analysis, plan, recorded=False)
        for name, analysis in plan.analyses.items()
    }


def _corners(
    context: _RunContext,
    namespaces: dict[Path, dict[str, object]],
    cache: fidelity.Cache,
    solved: Mapping[str, AnalysisResult],
) -> tuple[tuple[_Condition, ...], tuple[report.CornerReport, ...]]:
    """Executes corner simulations and returns all conditions and corner reports.

    Corners with custom arguments rebuild the circuit before resolving fidelity models
    via `cache`. Conditions sharing an identical circuit execute together in one batch.
    """
    plan = context.plan
    conditions = [
        _Condition(None, _kelvins(plan.temperature_at()), context.chosen, dict(solved))
    ]
    problems: dict[int, tuple[str, ...]] = {}
    for name, corner in plan.corners.items():
        rebuilt = (
            _built(context.job.script, namespaces, corner.arguments)
            if corner.arguments
            else context.built
        )
        temperature, process = plan.temperature_at(corner), plan.process_at(corner)
        found: fidelity.Fidelity | None = None
        try:
            found = fidelity.resolve(rebuilt, plan, temperature, cache, process)
        except ElaborationError as error:
            problems[len(conditions)] = error.problems
        conditions.append(_Condition(name, _kelvins(temperature), found))
    analyses = _unrecorded(plan)
    for structure in _structures(conditions, problems):
        cornered = [member for member in structure.members if member != 0]
        if not cornered:
            continue
        kept = _kept(
            structure.simulation,
            analyses,
            _at_temperatures(conditions, cornered),
            plan,
        )
        for run, member in zip(kept, cornered, strict=True):
            if run.problems:
                problems[member] = run.problems
            else:
                conditions[member].solved = run.results
                conditions[member].failures = run.errors
    reports = []
    for place, condition in enumerate(conditions[1:], start=1):
        name = condition.corner or ""
        process = plan.process_at(plan.corners[name])
        temperature = condition.temperature.value
        if place in problems:
            condition.chosen = None
            reports.append(
                report.CornerReport(
                    name=name,
                    temperature=temperature,
                    process=process,
                    problems=problems[place],
                )
            )
            continue
        failures = tuple(
            report.FailureReport(name=analysis, error=error)
            for analysis, error in condition.failures.items()
        )
        reports.append(
            report.CornerReport(
                name=name, temperature=temperature, process=process, failures=failures
            )
        )
    return tuple(conditions), tuple(reports)


def _structures(
    conditions: Sequence[_Condition], problems: dict[int, tuple[str, ...]]
) -> list[_Structure]:
    """Groups conditions that share an elaborated circuit with their `Simulation`.

    Two conditions share a circuit when they use the same built circuit, matched
    models, and process corner, regardless of temperature. Each group elaborates from
    its first valid condition, recording elaboration errors in `problems` by index.
    """
    structures: list[_Structure] = []
    keys: list[tuple[ir.Circuit, object, object, str | None]] = []
    for place, condition in enumerate(conditions):
        chosen = condition.chosen
        if chosen is None or place in problems:
            continue
        key = (chosen.built, chosen.matched, chosen.models, chosen.process)
        found = next(
            (
                index
                for index, other in enumerate(keys)
                if other[0] is key[0] and other[1:] == key[1:]
            ),
            None,
        )
        if found is not None:
            structures[found].members.append(place)
            continue
        try:
            simulation = chosen.simulation()
        except ElaborationError as error:
            problems[place] = error.problems
            continue
        keys.append(key)
        structures.append(_Structure(simulation, [place]))
    return structures


def _at_temperatures(
    conditions: Sequence[_Condition], members: Sequence[int]
) -> Explicit:
    """Builds an `Explicit` variant list with each indexed member's temperature."""
    return Explicit(
        [Variant(temperature=conditions[member].temperature) for member in members]
    )


def _kept(
    simulation: Simulation,
    analyses: Mapping[str, Analysis],
    variants: Variants,
    plan: Plan,
) -> list[Run]:
    """Executes `analyses` across `variants` on `simulation`, returning ordered runs.

    Runs that fail elaboration or parameter application record their problems without
    producing analysis results.
    """
    kept: dict[int, Run] = {}

    def keep(run: Run) -> None:
        kept[run.index] = run

    found = simulation.batch(analyses, variants, plan.options, each=keep)
    return [kept[index] for index in range(found.runs)]


def _batches(
    context: _RunContext,
    variants: Variants,
    *,
    traces: Sequence[str] = (),
    taken: range | None = None,
) -> list[BatchResult | None]:
    """Executes plan analyses on `variants` across conditions, sliced by `taken`.

    Every variant run executes unless `taken` selects a subrange. Each elaborating
    condition runs a batch measuring `context.specs` on its shared circuit, while
    nominal conditions also collect `traces`. Conditions failing elaboration yield
    `None`.
    """
    plan, specs, conditions = context.plan, context.specs, context.conditions
    analyses = _unrecorded(plan)
    measures = [] if specs is None else [each.measure for each in specs.requirements]
    found: list[BatchResult | None] = [None] * len(conditions)
    for structure in _structures(conditions, {}):
        for member in structure.members:
            found[member] = structure.simulation.batch(
                analyses,
                Product(_at_temperatures(conditions, [member]), variants),
                plan.options,
                measures=measures,
                traces=traces if member == 0 else (),
                runs=taken,
            )
    return found


def _monte_carlo(
    context: _RunContext, monte_carlo: MonteCarlo, *, traces: Sequence[str]
) -> tuple[report.MonteCarloReport, files.Batch]:
    """Executes `monte_carlo` across conditions, returning report and nominal batch.

    Each sample draws circuit tolerances and foundry mismatch, running across every
    elaborating condition with nominal characterized models. Every condition seeds its
    draws independently from `monte_carlo.seed`, aligning identical toleranced circuits
    across corners while taking the worst requirement verdict per sample. Nominal
    samples also collect `traces`.
    """
    return shards.monte_carlo(
        _sampled(context, monte_carlo, taken=range(monte_carlo.samples), traces=traces)
    )


def _sampled(
    context: _RunContext,
    monte_carlo: MonteCarlo,
    *,
    taken: range,
    traces: Sequence[str],
) -> files.Shard:
    """Executes the `taken` slice of `monte_carlo` across conditions as a `Shard`.

    Samples are measured against `context.specs`, and the nominal condition batch
    records `traces`.
    """
    nominal = context.conditions[0].chosen
    described = _described(context, monte_carlo, traces)
    found = _batches(context, monte_carlo, traces=traces, taken=taken)
    return files.Shard(
        start=taken.start,
        stop=taken.stop,
        digest=shards.digest(described),
        samples=monte_carlo.samples,
        seed=monte_carlo.seed,
        toleranced=0 if nominal is None else len(variation.toleranced(nominal.built)),
        mismatched=len(context.simulation.foundry),
        corners=tuple(condition.corner or "" for condition in context.conditions[1:]),
        analyses=tuple(context.plan.analyses),
        specs=context.specs,
        conditions=tuple(found),
    )


def _shard(context: _RunContext) -> report.ShardReport:
    """Executes the Monte Carlo sample slice requested by `context.job` as a shard.

    Requested samples execute across nominal and corner conditions, collecting plot
    traces on nominal conditions and saving the shard archive into `context.directory`.

    Raises:
        ScriptError: When the plan defines no Monte Carlo analysis or the requested
            sample range exceeds the configured sample count.
    """
    monte_carlo = context.plan.monte_carlo
    if monte_carlo is None:
        message = "the plan has no Monte Carlo analysis whose samples a shard runs"
        raise ScriptError(message)
    if context.job.samples is None or context.job.samples[1] > monte_carlo.samples:
        message = (
            f"the plan's Monte Carlo analysis draws {monte_carlo.samples} samples, "
            f"not samples {context.job.samples}"
        )
        raise ScriptError(message)
    taken = range(*context.job.samples)
    shard = _sampled(context, monte_carlo, taken=taken, traces=_traces(context))
    files.save_shard(context.directory / SHARD, shard)
    failed, _, _ = shards.spreads(shard.analyses, shard.conditions, len(taken), None)
    size = sum(_size(batch) for batch in shard.conditions if batch is not None)
    return report.ShardReport(
        start=taken.start,
        stop=taken.stop,
        samples=monte_carlo.samples,
        corners=shard.corners,
        failed=failed,
        size=size,
        total=size * monte_carlo.samples // len(taken),
        digest=shard.digest,
    )


def _size(batch: BatchResult) -> int:
    """Computes the total byte size of all NumPy arrays stored in `batch`."""
    arrays = [batch.values, batch.ended, batch.measured, batch.causes]
    arrays.extend(traces.values for traces in batch.traces)
    return sum(array.nbytes for array in arrays)


def _described(
    context: _RunContext, monte_carlo: MonteCarlo, traces: Sequence[str]
) -> dict[str, JsonValue]:
    """Builds a JSON-serializable digest payload capturing all shard-affecting inputs.

    The payload includes analyses, fidelity models, sample count, seed, per-condition
    metadata, and spec requirements and yield target. Options exclude parallelism
    fields other than `max_unknowns_per_interior`, which can alter factorization order.
    Each elaborating condition records its
    [`circuit_digest`][jolty.files.circuit_digest] (ignoring source line locations), or
    `None` when elaboration fails.
    """
    plan, specs = context.plan, context.specs
    digests: dict[int, str] = {}

    def circuit_of(condition: _Condition) -> str | None:
        chosen = condition.chosen
        if chosen is None:
            return None
        if id(chosen.built) not in digests:
            digests[id(chosen.built)] = files.circuit_digest(chosen.built)
        return digests[id(chosen.built)]

    options = plan.options.to_json()
    spread = options.pop("parallelism", None)
    # Dissected interior size alters factorization order and floating-point rounding.
    if isinstance(spread, dict) and "max_unknowns_per_interior" in spread:
        options["parallelism"] = {
            "max_unknowns_per_interior": spread["max_unknowns_per_interior"]
        }
    return {
        "analyses": {name: repr(analysis) for name, analysis in plan.analyses.items()},
        "options": options,
        "fidelity": {scope: repr(model) for scope, model in plan.fidelity.items()},
        "monte_carlo": {"samples": monte_carlo.samples, "seed": monte_carlo.seed},
        "conditions": [
            {
                "corner": condition.corner,
                "temperature": condition.temperature.value,
                "process": None
                if condition.chosen is None
                else condition.chosen.process,
                "circuit": circuit_of(condition),
            }
            for condition in context.conditions
        ],
        "specs": None
        if specs is None
        else {
            "requirements": [str(each) for each in specs.requirements],
            "yield_target": specs.yield_target,
        },
        "traces": list(traces),
    }


def _variants(
    context: _RunContext, variants: Variants, *, traces: Sequence[str]
) -> tuple[report.VariantsReport, files.Batch]:
    """Executes `variants` across conditions and returns the report and nominal batch.

    Each variant runs across every elaborating condition, taking the worst requirement
    verdict across conditions. The returned nominal batch includes `traces`.
    """
    found = _batches(context, variants, traces=traces)
    count = runs(variants)
    planned = tuple(context.plan.analyses)
    failed, spreads, passing = shards.spreads(planned, found, count, context.specs)
    structures = _structures(context.conditions, {})
    varied = structures[0].simulation.varied(variants) if structures else ()
    analyzed = report.VariantsReport(
        runs=count,
        parameters=varied,
        corners=tuple(condition.corner or "" for condition in context.conditions[1:]),
        failed=failed,
        spreads=spreads,
        yield_=passing,
    )
    return analyzed, shards.saved(found[0], count, planned, context.specs)


def _traces(context: _RunContext) -> tuple[str, ...]:
    """Selects unknown names to trace in batch runs for job plotting.

    Returns an empty tuple when plotting is disabled. Otherwise filters `context.job`
    traces, or default top-level net traces, to unknowns present in the circuit.
    """
    job, simulation = context.job, context.simulation
    if not job.plot:
        return ()
    named = {unknown.name for unknown in simulation.unknowns}
    traces = job.traces or unknowns.default_traces(simulation.unknowns, simulation.nets)
    return tuple(trace for trace in traces if trace in named)


def _worst_case(context: _RunContext, worst_case: WorstCase) -> report.WorstCaseReport:
    """Evaluates `context.specs` requirements at tolerance extremes that worsen them.

    At each condition, one perturbed run per toleranced parameter determines sensitivity
    signs for every requirement. Subsequent runs set each parameter and shared lot
    spread to the extreme moving a requirement toward each bound, deduplicating
    identical combinations. Each requirement receives its worst verdict across bounds
    and elaborating conditions.
    """
    conditions, plan, specs = context.conditions, context.plan, context.specs
    requirements = () if specs is None else specs.requirements
    found: list[list[tuple[int, str | None, Mapping[str, AnalysisResult]]]] = [
        [] for _ in requirements
    ]
    total_runs = 0
    for structure in _structures(conditions, {}) if requirements else ():
        ran, judged = _extremes(structure, context, worst_case, requirements)
        total_runs += ran
        for at, more in zip(found, judged, strict=True):
            at.extend(more)
    # Judge each requirement at nominal plan conditions first, then at each corner.
    verdicts = tuple(
        judge(
            requirement,
            plan.analyses,
            [(corner, solved) for _, corner, solved in sorted(at, key=_condition)],
        )
        for requirement, at in zip(requirements, found, strict=True)
        if at
    )
    first = next(
        (condition.chosen for condition in conditions if condition.chosen), None
    )
    return report.WorstCaseReport(
        step=worst_case.step,
        toleranced=0 if first is None else len(variation.toleranced(first.built)),
        corners=tuple(condition.corner or "" for condition in conditions[1:]),
        runs=total_runs,
        verdicts=verdicts,
    )


def _condition(judged: tuple[int, str | None, Mapping[str, AnalysisResult]]) -> int:
    """Extracts the plan condition index from a judged worst-case result tuple."""
    return judged[0]


def _extremes(
    structure: _Structure,
    context: _RunContext,
    worst_case: WorstCase,
    requirements: Sequence[Requirement],
) -> tuple[int, list[list[tuple[int, str | None, Mapping[str, AnalysisResult]]]]]:
    """Executes worst-case sensitivity and extreme runs for conditions in `structure`.

    Returns the total batch run count alongside per-requirement lists of
    `(condition_index, corner_name, results)` tuples ordered by condition and bound.
    """
    conditions, plan = context.conditions, context.plan
    analyses = _unrecorded(plan)
    members = [
        (member, condition, chosen)
        for member in structure.members
        if (chosen := (condition := conditions[member]).chosen) is not None
    ]
    varied = {
        member: variation.toleranced(chosen.built) for member, _, chosen in members
    }
    moves = [
        Variant(
            chosen.overrides(variation.moved(varied[member], index, worst_case.step)),
            condition.temperature,
        )
        for member, condition, chosen in members
        for index in range(len(varied[member]))
    ]
    moved = (
        _kept(structure.simulation, analyses, Explicit(moves), plan) if moves else []
    )
    extremes: list[Variant] = []
    wanted: list[tuple[int, _Condition, list[list[int]]]] = []
    total_runs = 0
    for member, condition, chosen in members:
        count = len(varied[member])
        ran, moved = moved[:count], moved[count:]
        slopes = _slopes(requirements, condition.solved, ran)
        keyed: dict[str, int] = {}
        keys = [
            [
                _keyed(
                    _extreme(
                        varied[member], [toward * slope[index] for slope in slopes]
                    ),
                    (chosen, condition.temperature),
                    keyed,
                    extremes,
                )
                for toward in _bounds(requirement)
            ]
            for index, requirement in enumerate(requirements)
        ]
        total_runs += count + len(keyed)
        wanted.append((member, condition, keys))
    solved = (
        _kept(structure.simulation, analyses, Explicit(extremes), plan)
        if extremes
        else []
    )
    judged: list[list[tuple[int, str | None, Mapping[str, AnalysisResult]]]] = [
        [] for _ in requirements
    ]
    for member, condition, keys in wanted:
        for at, bounds in zip(judged, keys, strict=True):
            at.extend((member, condition.corner, solved[key].results) for key in bounds)
    return total_runs, judged


def _keyed(
    overrides: dict[str, dict[str, ParameterValue]],
    at: tuple[fidelity.Fidelity, Quantity],
    keyed: dict[str, int],
    extremes: list[Variant],
) -> int:
    """Returns the index of `overrides` in `extremes`, deduplicating via `keyed`.

    Unmodeled parameter overrides are combined with the condition temperature in `at`
    so identical extreme combinations within one condition run once.
    """
    normalized = {
        path: {
            parameter: value.value if isinstance(value, Quantity) else value
            for parameter, value in named.items()
        }
        for path, named in overrides.items()
    }
    key = json.dumps(normalized, sort_keys=True)
    if key not in keyed:
        chosen, temperature = at
        keyed[key] = len(extremes)
        extremes.append(Variant(chosen.overrides(overrides), temperature))
    return keyed[key]


def _slopes(
    requirements: Sequence[Requirement],
    nominal: Mapping[str, AnalysisResult],
    moved: Sequence[Run],
) -> list[list[float]]:
    """Computes finite-difference requirement deltas for each perturbed run in `moved`.

    Deltas are measured relative to `nominal`, falling back to `0.0` whenever either
    nominal or perturbed reading is unavailable.
    """
    base = [reading(requirement, nominal) for requirement in requirements]
    slopes = []
    for run in moved:
        slope = []
        for requirement, before in zip(requirements, base, strict=True):
            after = reading(requirement, run.results)
            slope.append(0.0 if before is None or after is None else after - before)
        slopes.append(slope)
    return slopes


def _bounds(requirement: Requirement) -> list[float]:
    """Returns bound directions for `requirement`: -1.0 (minimum) and 1.0 (maximum)."""
    return [
        direction
        for direction, bound in (
            (-1.0, requirement.minimum),
            (1.0, requirement.maximum),
        )
        if bound is not None
    ]


def _extreme(
    varied: Sequence[variation.Toleranced], directions: Sequence[float]
) -> dict[str, dict[str, ParameterValue]]:
    """Computes extreme overrides for `varied` and lot spreads along `directions`.

    Each lot's overall direction is determined by summing the directions of its member
    parameters.
    """
    lots: dict[variation.LotMember, float] = {}
    for parameter, direction in zip(varied, directions, strict=True):
        if (key := variation.lot(parameter)) is not None:
            lots[key] = lots.get(key, 0.0) + direction
    return variation.extreme(
        varied,
        [float(np.sign(direction)) for direction in directions],
        {key: float(np.sign(direction)) for key, direction in lots.items()},
    )


def _specs(script: Path, namespaces: dict[Path, dict[str, object]]) -> Specs | None:
    """Invokes optional ``specs()`` in `script`, returning `None` when undefined.

    Raises:
        ScriptError: When ``specs()`` returns an object that is not a `Specs` instance.
    """
    match _call(script, "specs", namespaces, required=False):
        case None | Specs() as specs:
            return specs
        case other:
            message = f"specs() returned {type(other).__name__}, not Specs"
            raise ScriptError(message)


def _analyses(
    simulation: Simulation,
    plan: Plan,
    max_unknowns: int,
    outside: list[fidelity.Outside] | None = None,
    *,
    recorded: bool = False,
) -> tuple[dict[str, AnalysisResult], tuple[report.AnalysisReport, ...]]:
    """Executes all analyses in `plan` and returns completed results and reports.

    Logic domain excursions from completed analyses are appended to `outside` when
    provided. When `recorded` is true, results retain plan and per-analysis device
    recordings for saving and checks; otherwise recordings stay minimal for specs and
    fidelity contracts.
    """
    solved: dict[str, AnalysisResult] = {}
    reports: list[report.AnalysisReport] = []
    for name, analysis in plan.analyses.items():
        try:
            chosen = _recording(analysis, plan, recorded=recorded)
            result = simulation.run(chosen, plan.options)
        except JoltyError as error:
            reports.append(report.FailureReport(name=name, error=str(error)))
        else:
            solved[name] = result
            reports.append(report.report(name, result, max_unknowns))
            if outside is not None:
                outside.extend(
                    fidelity.Outside(analysis=name, excursion=excursion)
                    for excursion in result.excursions or ()
                )
    return solved, tuple(reports)


def _recording(analysis: Analysis, plan: Plan, *, recorded: bool) -> Analysis:
    """Configures recording fields on `analysis` according to `plan` and `recorded`.

    When `recorded` is true, applies analysis or plan recording patterns and enables
    transistor operating-region recording on `OperatingPoint`. When `recorded` is false,
    clears explicit recording patterns.
    """
    if not isinstance(analysis, Recorded):
        return analysis
    if not recorded:
        return replace(analysis, record=())
    record = plan.record if analysis.record is None else analysis.record
    regions = isinstance(analysis, OperatingPoint)
    return replace(analysis, record=record, regions=regions)


def _call(
    script: Path,
    name: str,
    namespaces: dict[Path, dict[str, object]],
    *,
    required: bool = True,
    arguments: Mapping[str, object] | None = None,
) -> object:
    """Invokes function `name` in `script` with `arguments`, caching script execution.

    Each script executes at most once in `namespaces` regardless of call count. Missing
    functions return `None` when `required` is false.

    Raises:
        ScriptError: When `script` defines no callable `name` and `required` is true.
    """
    if script not in namespaces:
        namespaces[script] = runpy.run_path(str(script), run_name="__jolty__")
    function = namespaces[script].get(name)
    if callable(function):
        return function(**(arguments or {}))
    if not required:
        return None
    message = f"{_relative(str(script))} defines no {name}() function"
    raise ScriptError(message)


@contextmanager
def _importable(*scripts: Path | None) -> Generator[None]:
    directories = list(
        dict.fromkeys(str(script.parent.resolve()) for script in scripts if script)
    )
    sys.path[:0] = directories
    try:
        yield
    finally:
        for directory in directories:
            sys.path.remove(directory)


def _failure(error: BaseException, message: str) -> Failure:
    """Describes `error` with user-script traceback frames ordered innermost first."""
    frames = [
        ir.SourceLocation(
            file=_relative(frame.f_code.co_filename),
            line=line or 1,
            function=frame.f_code.co_name,
        )
        for frame, line in traceback.walk_tb(error.__traceback__)
        if _is_script(frame.f_code.co_filename)
    ]
    frames.reverse()
    return Failure(error=type(error).__name__, message=message, frames=tuple(frames))


def _is_script(filename: str) -> bool:
    if filename.startswith("<"):
        return False
    path = Path(filename).resolve()
    return not any(path.is_relative_to(library) for library in _LIBRARIES)


def _relative(filename: str) -> str:
    path = Path(filename).resolve()
    return (
        path.relative_to(Path.cwd()).as_posix()
        if path.is_relative_to(Path.cwd())
        else path.as_posix()
    )


def main(argv: list[str] | None = None) -> None:
    """Executes the runner job located in the directory passed as the CLI argument.

    Args:
        argv: Single-element argument list naming the job directory, defaulting to
            ``sys.argv[1:]`` when ``None``.
    """
    (argument,) = sys.argv[1:] if argv is None else argv
    directory = Path(argument)
    job = Job.model_validate_json((directory / JOB).read_bytes())
    (directory / OUTCOME).write_bytes(JOB_RESULTS.dump_json(run(job, directory)))


if __name__ == "__main__":
    main()
