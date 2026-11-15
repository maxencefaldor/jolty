# Copyright 2026 Maxence Faldor
"""Run scripts in a process of their own.

The service starts ``python -P -m jolty.runner DIRECTORY`` for every command that runs
scripts. The runner reads the [`Job`][jolty.runner.Job] in ``DIRECTORY/job.json``. It
calls the script's ``build()`` and elaborates the circuit it returns, then, when the job
names a plan, calls the plan's ``plan()`` and runs each analysis, at the plan's own
conditions then at each of its corners, building the circuit again with a corner's
arguments. Every run takes the models the plan's fidelity chooses, characterized once
each, and spot checks run again with sampled instances at their circuit's fidelity. It
evaluates the requirements its ``specs()`` returns, when it defines one, on every run,
then on each sample of the plan's Monte Carlo analysis and each run of its variants,
which run at the plan's own conditions and at each corner. It writes the
[`Outcome`][jolty.runner.Outcome], what was built or why it failed, to
``DIRECTORY/outcome.json``, the results of the analyses at the plan's
own conditions to ``DIRECTORY/results.npz``, and the plan's batches of runs, its Monte
Carlo analysis's and its variants', at the plan's own conditions, to
``DIRECTORY/batches.npz``.

A job that names samples of the plan's Monte Carlo analysis runs them alone, as a shard
([`jolty.shards`][jolty.shards]): at the plan's own conditions and at every corner, and
writes them to ``DIRECTORY/shard.npz``, running none of the plan's other analyses.

Every run but those of the plan's own conditions runs in a batch, one call to the
engine for many runs of one circuit across its threads: the conditions whose circuits
are one, the plan's own and those of the corners that change only the temperature,
share the circuit and take their temperatures as the values of their runs, the
corners' runs one batch, and a Monte Carlo analysis's samples or the variants' runs a
batch at each condition; a corner whose arguments build the circuit again runs on a
circuit of its own.

Scripts and analyses therefore run outside the command line and the MCP server, whose
standard output may be a protocol channel: what a script prints is captured, a timeout
stops a script or an analysis that runs too long, and a crash ends only the runner.
"""

# jolty's modules load as the runner first uses them, so that a job that draws no batch
# never imports matplotlib.
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
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from jolty import checks, fidelity, files, ir, plots, report, shards, variation
from jolty.builder import Component, circuit
from jolty.errors import ElaborationError, JoltyError, ScriptError, SimulationError
from jolty.fidelity import FidelityReport
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
from jolty.results import BatchResult, Result, Run
from jolty.simulate import Simulation, kelvins
from jolty.specs import Requirement, Specs, Verdict, evaluate, judge, reading
from jolty.units import TEMPERATURE, Quantity

JOB: Final = "job.json"
"""The file in the job's directory that holds the job."""

OUTCOME: Final = "outcome.json"
"""The file in the job's directory where the runner writes what the job produced."""

RESULTS: Final = "results.npz"
"""The file in the job's directory where the runner writes the analyses' results."""

BATCHES: Final = "batches.npz"
"""The file in the job's directory where the runner writes the plan's batches."""

SHARD: Final = "shard.npz"
"""The file in the job's directory where the runner writes the shard a job runs."""

MONTE_CARLO: Final = "monte-carlo"
"""The name of a plan's Monte Carlo analysis's batch, which no analysis's name is."""

VARIANT_RUNS: Final = "variant-runs"
"""The name of the batch of a plan's variants, which no analysis's name is."""

_LIBRARIES: Final = tuple(
    {
        Path(__file__).resolve().parent,
        *(
            Path(sysconfig.get_path(name)).resolve()
            for name in ("stdlib", "platstdlib", "purelib", "platlib")
        ),
    }
)
"""Where code that is not a script lives: jolty, the standard library, packages."""


class Thresholds(BaseModel):
    """The thresholds of the parallelism a job sets in place of its plan's."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_solves_per_task: int | None = Field(default=None, ge=1)
    min_instances_per_task: int | None = Field(default=None, ge=1)
    min_unknowns_inside: int | None = Field(default=None, ge=1)
    max_unknowns_per_interior: int | None = Field(default=None, ge=1)


class Job(BaseModel):
    """What the runner runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    script: Path = Field(description="The script that defines build().")
    plan: Path | None = Field(
        default=None,
        description="The script that defines plan(); null to elaborate only.",
    )
    listed: int = Field(ge=1, description="Unknowns each report lists.")
    check: bool = Field(
        default=True,
        description="Whether to check the rules at the plan's operating point.",
    )
    record: list[str] | None = Field(
        default=None,
        description="The instances the results record, in the plan's place; null "
        "for the plan's own record.",
    )
    plot: bool = Field(
        default=False,
        description="Whether each batch traces the unknowns its plots draw, at the "
        "plan's own conditions.",
    )
    probes: list[str] = Field(
        default_factory=list,
        description="The unknowns a batch traces for its plots; by default the nets "
        "the top component names, as a plot draws them.",
    )
    parallelism: Thresholds = Field(
        default_factory=Thresholds,
        description="The thresholds of the plan's parallelism replaced, where given.",
    )
    samples: tuple[int, int] | None = Field(
        default=None,
        description="The samples of the plan's Monte Carlo analysis to run alone, as "
        "a shard, from the first up to the second; null to run the plan.",
    )


class Built(BaseModel):
    """A circuit, whether the engine elaborates it, and the reports of its analyses."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["built"] = "built"
    circuit: ir.Circuit
    devices: int | None = Field(
        default=None,
        description="Devices with every component instance unfolded; null when the "
        "circuit does not elaborate.",
    )
    unknowns: int | None = Field(
        default=None,
        description="Unknowns the engine solves for; null when the circuit does not "
        "elaborate.",
    )
    problems: tuple[str, ...] = Field(
        default=(), description="Why the circuit does not elaborate."
    )
    diagnostics: tuple[checks.Diagnostic, ...] = Field(
        default=(),
        description="The checks' diagnostics, at their default levels: the structural "
        "ones, then, with a plan, those at its operating point.",
    )
    analyses: tuple[report.AnalysisReport, ...] = Field(
        default=(), description="A report per analysis of the plan, in its order."
    )
    corners: tuple[report.CornerReport, ...] = Field(
        default=(), description="A report per corner of the plan, in its order."
    )
    monte_carlo: report.MonteCarloReport | None = Field(
        default=None,
        description="What the plan's Monte Carlo analysis found; null without one.",
    )
    variants: report.VariantsReport | None = Field(
        default=None,
        description="What the runs of the plan's variants found; null without them.",
    )
    worst_case: report.WorstCaseReport | None = Field(
        default=None,
        description="What the plan's worst-case analysis found; null without one.",
    )
    shard: report.ShardReport | None = Field(
        default=None,
        description="What the shard of the plan's Monte Carlo samples ran; null but "
        "for a job of samples.",
    )
    specs: tuple[Verdict, ...] = Field(
        default=(),
        description="The verdict on each requirement of the plan's specs, in order; "
        "empty when the plan defines no specs().",
    )
    fidelity: FidelityReport | None = Field(
        default=None,
        description="The models the plan's fidelity chose, and its spot checks; null "
        "when it chooses none.",
    )


class Failure(BaseModel):
    """Why a script produced nothing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["failure"] = "failure"
    error: str = Field(description="The exception's type, such as 'BuildError'.")
    message: str = Field(description="The exception's message.")
    frames: tuple[ir.Source, ...] = Field(
        default=(), description="The script lines of the traceback, innermost first."
    )

    @override
    def __str__(self) -> str:
        """The error and its message, after the script line that raised it."""
        head = f"{self.error}: {self.message}" if self.message else self.error
        if not self.frames:
            return head
        innermost, *callers = (str(frame) for frame in self.frames)
        return f"{innermost}: {head}" + "".join(
            f"\n  called from {caller}" for caller in callers
        )


type Outcome = Annotated[Built | Failure, Field(discriminator="kind")]
"""What a job produced: a circuit, or why the script failed."""

OUTCOMES: Final = TypeAdapter[Outcome](Outcome)
"""Reads and writes outcomes as JSON."""


def run(job: Job, directory: Path) -> Outcome:
    """Runs `job` in this process, writing the results of its analyses to `directory`.

    The directories of the scripts come first on the import path while they run, so a
    script imports the modules beside it, as under ``python script.py``.
    """
    try:
        with _importable(job.script, job.plan):
            return _run(job, directory)
    except SystemExit as error:
        return _failure(error, f"the script exited with {error.code!r}")
    except BaseException as error:  # noqa: BLE001 - a script may raise anything
        return _failure(error, str(error))


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
    cache: fidelity.Cache = {}
    chosen = fidelity.resolve(built, plan, plan.temperature, cache, plan.process)
    if plan.temperature is not None or plan.process is not None or chosen.matched:
        try:
            simulation = chosen.simulation()
        except ElaborationError as error:
            return Built(circuit=built, problems=error.problems)
    if job.samples is not None:
        # A shard runs none of the plan's analyses at its own conditions.
        conditions, _ = _corners(
            job, namespaces, built, plan, cache, chosen=chosen, solved={}
        )
        specs = _specs(job.plan, namespaces)
        shard = _shard(
            job, directory, conditions, plan, specs=specs, simulation=simulation
        )
        return elaborated.model_copy(update={"shard": shard})
    outside: list[fidelity.Outside] = []
    solved, reports = _analyses(simulation, plan, job.listed, outside, recorded=True)
    at_point = (
        _at_operating_point(simulation, plan) + simulation.check_results(solved)
        if job.check
        else ()
    )
    if solved:
        files.save_results(directory / RESULTS, solved, circuit=built, plan=plan)
    conditions, corners = _corners(
        job, namespaces, built, plan, cache, chosen=chosen, solved=solved
    )
    specs = _specs(job.plan, namespaces)
    at_corners = {
        condition.corner: condition.solved
        for condition in conditions
        if condition.corner is not None
    }
    verdicts = (
        () if specs is None else evaluate(specs, plan.analyses, solved, at_corners)
    )
    checked = _fidelity(chosen, plan, solved, specs)
    monte_carlo, variants = _batched(
        job, directory, conditions, plan, specs, simulation=simulation
    )
    worst_case = (
        None
        if plan.worst_case is None
        else _worst_case(conditions, plan, plan.worst_case, specs)
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


def _batched(  # noqa: PLR0913 - the job, the plan, and where its batches run
    job: Job,
    directory: Path,
    conditions: Sequence[_Condition],
    plan: Plan,
    specs: Specs | None,
    *,
    simulation: Simulation,
) -> tuple[report.MonteCarloReport | None, report.VariantsReport | None]:
    """What the plan's Monte Carlo analysis and its variants found, if it has them.

    Their batches at the plan's own conditions, tracing the unknowns the job's plots
    draw, which `simulation` runs, are written to `directory`.
    """
    traced = _traced(job, simulation)
    batches: dict[str, files.Batch] = {}
    monte_carlo = None
    if plan.monte_carlo is not None:
        monte_carlo, batches[MONTE_CARLO] = _monte_carlo(
            conditions,
            plan,
            plan.monte_carlo,
            specs,
            mismatched=len(simulation.foundry),
            traced=traced,
        )
    variants = None
    if plan.variants is not None:
        variants, batches[VARIANT_RUNS] = _variants(
            conditions, plan, plan.variants, specs, traced=traced
        )
    if batches:
        files.save_batches(directory / BATCHES, batches)
    return monte_carlo, variants


def _at_operating_point(
    simulation: Simulation, plan: Plan
) -> tuple[checks.Diagnostic, ...]:
    """The diagnostics of the rules at the plan's operating point.

    A circuit with no operating point has none: the analyses that need it report why.
    """
    try:
        return simulation.check_operating_point(plan.options)
    except SimulationError:
        return ()


def _fidelity(
    chosen: fidelity.Fidelity,
    plan: Plan,
    solved: Mapping[str, Result],
    specs: Specs | None,
) -> FidelityReport | None:
    """The models `chosen` holds, and the spot checks of `plan`'s results `solved`.

    Each spot check holds its instance to its contract, and measures `specs` again to
    find the requirements whose verdict depends on the instance's fidelity. It is None
    when no model stands in for an instance.
    """
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
    """The circuit `script`'s ``build()`` returns, called with `arguments`.

    Raises:
        ScriptError: When it returns neither a component nor a circuit.
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


@dataclass(slots=True)
class _Condition:
    """Where a plan's analyses run, its own or a corner's conditions, and results."""

    corner: str | None
    """The corner, None for the plan's own conditions."""
    temperature: Quantity
    chosen: fidelity.Fidelity | None
    """The models chosen there, None where the circuit does not elaborate."""
    solved: Mapping[str, Result] = field(default_factory=dict)
    """The results there of the analyses that end, by name."""
    failures: Mapping[str, str] = field(default_factory=dict)
    """Why each analysis that failed there failed, by name."""


@dataclass(slots=True)
class _Structure:
    """Conditions whose circuits are one, and the simulation of it batches run on.

    Each condition's runs take its temperature as their values, so the conditions run
    on one circuit, elaborated once.
    """

    simulation: Simulation
    members: list[int]
    """The conditions, by their places among the plan's."""


def _kelvins(temperature: Quantity | None) -> Quantity:
    """`temperature`, 27 °C when it is None, as a run's values give it."""
    return Quantity(kelvins(temperature), TEMPERATURE)


def _unrecorded(plan: Plan) -> dict[str, Analysis]:
    """The plan's analyses, recording nothing beyond what the checks read."""
    return {
        name: _recording(analysis, plan, recorded=False)
        for name, analysis in plan.analyses.items()
    }


def _corners(  # noqa: PLR0913 - the script, the plan, and what its own run found
    job: Job,
    namespaces: dict[Path, dict[str, object]],
    built: ir.Circuit,
    plan: Plan,
    cache: fidelity.Cache,
    *,
    chosen: fidelity.Fidelity,
    solved: Mapping[str, Result],
) -> tuple[list[_Condition], tuple[report.CornerReport, ...]]:
    """The plan's own conditions and its corners', and what each corner found.

    A corner builds the circuit again with its arguments, if it gives any, and takes the
    models the plan's fidelity chooses there, characterized unless `cache` holds them;
    the corners whose circuits are one run in one batch.
    """
    conditions = [
        _Condition(None, _kelvins(plan.temperature_at()), chosen, dict(solved))
    ]
    problems: dict[int, tuple[str, ...]] = {}
    for name, corner in plan.corners.items():
        rebuilt = (
            _built(job.script, namespaces, corner.arguments)
            if corner.arguments
            else built
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
    return conditions, tuple(reports)


def _structures(
    conditions: Sequence[_Condition], problems: dict[int, tuple[str, ...]]
) -> list[_Structure]:
    """The conditions whose circuits are one, each with the simulation of it.

    Two conditions' circuits are one when they build the same circuit and take the same
    models at the same process corner, whatever their temperatures. The simulation is
    the first condition's whose circuit elaborates; a condition whose circuit does not
    joins `problems`, by its place, with why.
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
    """A run at the temperature of each of `members`, among `conditions`."""
    return Explicit(
        [Variant(temperature=conditions[member].temperature) for member in members]
    )


def _kept(
    simulation: Simulation,
    analyses: Mapping[str, Analysis],
    variants: Variants,
    plan: Plan,
) -> list[Run]:
    """What `analyses` find on each run of `variants`, batched on `simulation`.

    A run whose values make problems holds them, and no result.
    """
    kept: dict[int, Run] = {}

    def keep(run: Run) -> None:
        kept[run.index] = run

    found = simulation.batch(analyses, variants, plan.options, each=keep)
    return [kept[index] for index in range(found.runs)]


def _batches(  # noqa: PLR0913 - the runs, where they run, and what they measure
    conditions: Sequence[_Condition],
    plan: Plan,
    variants: Variants,
    specs: Specs | None,
    *,
    traced: Sequence[str] = (),
    taken: range | None = None,
) -> list[BatchResult | None]:
    """What the plan's analyses find on runs `taken` of `variants` at each condition.

    It runs every run of the variants unless `taken` names some. Each condition whose
    circuit elaborates holds the batch of its runs, a batch of its own on the circuit
    it shares with others, measuring the requirements of `specs`, the plan's own
    tracing the unknowns `traced` names; one whose circuit does not, none.
    """
    analyses = _unrecorded(plan)
    measurements = (
        [] if specs is None else [each.measurement for each in specs.requirements]
    )
    found: list[BatchResult | None] = [None] * len(conditions)
    for structure in _structures(conditions, {}):
        for member in structure.members:
            found[member] = structure.simulation.batch(
                analyses,
                Product(_at_temperatures(conditions, [member]), variants),
                plan.options,
                measurements=measurements,
                traced=traced if member == 0 else (),
                runs=taken,
            )
    return found


def _monte_carlo(  # noqa: PLR0913 - the analysis, where it runs, and what it judges
    conditions: Sequence[_Condition],
    plan: Plan,
    monte_carlo: MonteCarlo,
    specs: Specs | None,
    *,
    mismatched: int,
    traced: Sequence[str],
) -> tuple[report.MonteCarloReport, files.Batch]:
    """What `plan`'s analyses find on each sample of `monte_carlo`, against `specs`.

    Each sample draws the circuit's tolerances, and the kit's mismatch of its
    `mismatched` foundry's devices, and runs at every one of `conditions` whose
    circuit elaborates, with the models chosen there; the models stand for their
    circuits' nominal parts. Every condition draws its samples from the seed afresh, so
    a corner whose circuit holds the same toleranced parameters runs each sample on the
    same parts, and a requirement's verdict on a sample is its worst over the runs.
    Also returns the samples at the plan's own conditions, tracing `traced`. They are
    the one shard of every sample, which [`jolty.shards`][jolty.shards] reports as it
    reports the merge of many.
    """
    every = range(monte_carlo.samples)
    return shards.monte_carlo(
        _sampled(
            conditions,
            plan,
            monte_carlo,
            specs,
            taken=every,
            traced=traced,
            mismatched=mismatched,
        )
    )


def _sampled(  # noqa: PLR0913 - the analysis, where it runs, and what it judges
    conditions: Sequence[_Condition],
    plan: Plan,
    monte_carlo: MonteCarlo,
    specs: Specs | None,
    *,
    taken: range,
    traced: Sequence[str],
    mismatched: int,
) -> files.Shard:
    """The shard of the samples `taken` of `monte_carlo`, run at each of `conditions`.

    They are measured against `specs`, and the plan's own conditions trace `traced`.
    """
    nominal = conditions[0].chosen
    described = _described(conditions, plan, monte_carlo, specs, traced)
    found = _batches(conditions, plan, monte_carlo, specs, traced=traced, taken=taken)
    return files.Shard(
        start=taken.start,
        stop=taken.stop,
        fingerprint=shards.fingerprint(described),
        samples=monte_carlo.samples,
        seed=monte_carlo.seed,
        toleranced=0 if nominal is None else len(variation.toleranced(nominal.built)),
        mismatched=mismatched,
        corners=tuple(condition.corner or "" for condition in conditions[1:]),
        analyses=tuple(plan.analyses),
        specs=specs,
        conditions=tuple(found),
    )


def _shard(  # noqa: PLR0913 - the job, the plan, and where its samples run
    job: Job,
    directory: Path,
    conditions: Sequence[_Condition],
    plan: Plan,
    *,
    specs: Specs | None,
    simulation: Simulation,
) -> report.ShardReport:
    """Runs the samples `job` names of the plan's Monte Carlo analysis, as a shard.

    They run at every one of `conditions`, as all the samples do, the batch at the
    plan's own conditions, which `simulation` runs, tracing the unknowns the job's
    plots draw, and the shard is written to `directory`.

    Raises:
        ScriptError: When the plan has no Monte Carlo analysis, or its samples hold
            none of those the job names.
    """
    monte_carlo = plan.monte_carlo
    if monte_carlo is None:
        message = "the plan has no Monte Carlo analysis whose samples a shard runs"
        raise ScriptError(message)
    if job.samples is None or job.samples[1] > monte_carlo.samples:
        message = (
            f"the plan's Monte Carlo analysis draws {monte_carlo.samples} samples, "
            f"not samples {job.samples}"
        )
        raise ScriptError(message)
    taken = range(*job.samples)
    shard = _sampled(
        conditions,
        plan,
        monte_carlo,
        specs,
        taken=taken,
        traced=_traced(job, simulation),
        mismatched=len(simulation.foundry),
    )
    files.save_shard(directory / SHARD, shard)
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
        fingerprint=shard.fingerprint,
    )


def _size(batch: BatchResult) -> int:
    """The bytes of the arrays of `batch`, the same for each of its runs."""
    arrays = [batch.values, batch.ended, batch.measured, batch.causes]
    arrays.extend(traced.values for traced in batch.traced)
    return sum(array.nbytes for array in arrays)


def _described(
    conditions: Sequence[_Condition],
    plan: Plan,
    monte_carlo: MonteCarlo,
    specs: Specs | None,
    traced: Sequence[str],
) -> dict[str, JsonValue]:
    """What the numbers of a shard of `monte_carlo`'s samples depend on, as JSON.

    It is the analyses and the options, but for the parallelism, which changes no
    result save for the interiors its factorization is dissected into; the fidelity's
    models, by scope; the samples and their seed; each condition's corner,
    temperature, process corner, and the [`circuit_hash`][jolty.files.circuit_hash] of
    its circuit, which leaves out the script lines that placed its instances, none where
    it does not elaborate; the requirements of `specs`, in words, and their yield
    target; and the unknowns `traced` names.
    """
    hashed: dict[int, str] = {}

    def circuit_of(condition: _Condition) -> str | None:
        chosen = condition.chosen
        if chosen is None:
            return None
        if id(chosen.built) not in hashed:
            hashed[id(chosen.built)] = files.circuit_hash(chosen.built)
        return hashed[id(chosen.built)]

    options = plan.options.to_json()
    spread = options.pop("parallelism", None)
    # The interiors order the factorization's operations, so the numbers' last digits.
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
            for condition in conditions
        ],
        "specs": None
        if specs is None
        else {
            "requirements": [str(each) for each in specs.requirements],
            "yield_target": specs.yield_target,
        },
        "traced": list(traced),
    }


def _variants(
    conditions: Sequence[_Condition],
    plan: Plan,
    variants: Variants,
    specs: Specs | None,
    *,
    traced: Sequence[str],
) -> tuple[report.VariantsReport, files.Batch]:
    """What `plan`'s analyses find on each run of `variants`, against `specs`.

    Each run takes its values at every one of `conditions` whose circuit elaborates,
    and a requirement's verdict on it is its worst over the conditions. Also returns
    the runs at the plan's own conditions, tracing `traced`.
    """
    found = _batches(conditions, plan, variants, specs, traced=traced)
    count = runs(variants)
    planned = tuple(plan.analyses)
    failed, spreads, passing = shards.spreads(planned, found, count, specs)
    structures = _structures(conditions, {})
    varied = structures[0].simulation.varied(variants) if structures else ()
    analysed = report.VariantsReport(
        runs=count,
        parameters=varied,
        corners=tuple(condition.corner or "" for condition in conditions[1:]),
        failed=failed,
        spreads=spreads,
        passing=passing,
    )
    return analysed, shards.saved(found[0], count, planned, specs)


def _traced(job: Job, simulation: Simulation) -> tuple[str, ...]:
    """The unknowns each batch traces for the plots of `job`, by name; none unplotted.

    They are the job's probes, or the nets the top component names, as a plot draws
    them, that are unknowns of `simulation`'s circuit.
    """
    if not job.plot:
        return ()
    named = {unknown.name for unknown in simulation.unknowns}
    probes = job.probes or plots.default_probes(simulation.unknowns, simulation.nets)
    return tuple(probe for probe in probes if probe in named)


def _worst_case(
    conditions: Sequence[_Condition],
    plan: Plan,
    worst_case: WorstCase,
    specs: Specs | None,
) -> report.WorstCaseReport:
    """Each requirement of `specs` at the extremes of the tolerances that worsen it.

    At each of `conditions`, a run per toleranced parameter, moved its `worst_case`
    step, finds which way each requirement's value moves with it, and a run per bound
    of each requirement sets every parameter, and every lot's spread, at the extreme
    that moves the value toward the bound; runs at the same extremes are shared. A
    requirement's verdict is its worst over every bound and conditions where the
    circuit elaborates. The runs of the conditions whose circuits are one make a batch.
    """
    requirements = () if specs is None else specs.requirements
    found: list[list[tuple[int, str | None, Mapping[str, Result]]]] = [
        [] for _ in requirements
    ]
    runs = 0
    for structure in _structures(conditions, {}) if requirements else ():
        ran, judged = _extremes(structure, conditions, plan, worst_case, requirements)
        runs += ran
        for at, more in zip(found, judged, strict=True):
            at.extend(more)
    # Each requirement is judged at the plan's own conditions, then at each corner.
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
        runs=runs,
        verdicts=verdicts,
    )


def _condition(judged: tuple[int, str | None, Mapping[str, Result]]) -> int:
    """The place among the plan's conditions of the condition a result comes from."""
    return judged[0]


def _extremes(
    structure: _Structure,
    conditions: Sequence[_Condition],
    plan: Plan,
    worst_case: WorstCase,
    requirements: Sequence[Requirement],
) -> tuple[int, list[list[tuple[int, str | None, Mapping[str, Result]]]]]:
    """How many runs a worst case makes at the conditions of `structure`, and more.

    It returns what each requirement is judged on there too: at each condition, by its
    place and its corner, the results at the extremes of each of its bounds, in order.
    """
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
    runs = 0
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
        runs += count + len(keyed)
        wanted.append((member, condition, keys))
    solved = (
        _kept(structure.simulation, analyses, Explicit(extremes), plan)
        if extremes
        else []
    )
    judged: list[list[tuple[int, str | None, Mapping[str, Result]]]] = [
        [] for _ in requirements
    ]
    for member, condition, keys in wanted:
        for at, bounds in zip(judged, keys, strict=True):
            at.extend((member, condition.corner, solved[key].results) for key in bounds)
    return runs, judged


def _keyed(
    overrides: dict[str, dict[str, float]],
    at: tuple[fidelity.Fidelity, Quantity],
    keyed: dict[str, int],
    extremes: list[Variant],
) -> int:
    """The place among `extremes` of the run at `overrides`, unless `keyed` holds it.

    The run takes the numbers the models `at` holds do not stand in for, at its
    temperature; the runs at the same extremes of one condition are shared.
    """
    key = json.dumps(overrides, sort_keys=True)
    if key not in keyed:
        chosen, temperature = at
        keyed[key] = len(extremes)
        extremes.append(Variant(chosen.overrides(overrides), temperature))
    return keyed[key]


def _slopes(
    requirements: Sequence[Requirement],
    nominal: Mapping[str, Result],
    moved: Sequence[Run],
) -> list[list[float]]:
    """How far each requirement's value moves, per run of `moved`, one parameter moved.

    A value moves from its value on the `nominal` results, and 0 where either cannot be
    read.
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
    """The way toward each of `requirement`'s bounds: -1 down, 1 up."""
    return [
        direction
        for direction, bound in (
            (-1.0, requirement.minimum),
            (1.0, requirement.maximum),
        )
        if bound is not None
    ]


def _extreme(
    varied: Sequence[variation.Varied], directions: Sequence[float]
) -> dict[str, dict[str, float]]:
    """Each of `varied`, and each lot's spread, at the extreme in its direction.

    A lot's direction is the sum of its parts'.
    """
    lots: dict[variation.Lot, float] = {}
    for parameter, direction in zip(varied, directions, strict=True):
        if (key := variation.lot(parameter)) is not None:
            lots[key] = lots.get(key, 0.0) + direction
    return variation.extreme(
        varied,
        [float(np.sign(direction)) for direction in directions],
        {key: float(np.sign(direction)) for key, direction in lots.items()},
    )


def _specs(script: Path, namespaces: dict[Path, dict[str, object]]) -> Specs | None:
    """The specs `script`'s ``specs()`` returns; None when it defines none.

    Raises:
        ScriptError: When ``specs()`` returns no specs.
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
    listed: int,
    outside: list[fidelity.Outside] | None = None,
    *,
    recorded: bool = False,
) -> tuple[dict[str, Result], tuple[report.AnalysisReport, ...]]:
    """The results of the analyses of `plan` that end, by name, and a report on each.

    The excursions outside its logic devices' domains of each analysis that ends join
    `outside`. The results keep what the plan and each analysis record when they are
    `recorded`, as
    the plan's own run's are, which are saved and checked, and nothing more than the
    checks read otherwise, as a corner's, a sample's and a spot check's are, which only
    specs and contracts read.
    """
    solved: dict[str, Result] = {}
    reports: list[report.AnalysisReport] = []
    for name, analysis in plan.analyses.items():
        try:
            transistors = simulation.transistors if recorded else ()
            chosen = _recording(
                analysis, plan, recorded=recorded, transistors=transistors
            )
            result = simulation.run(chosen, plan.options)
        except JoltyError as error:
            reports.append(report.FailureReport(name=name, error=str(error)))
        else:
            solved[name] = result
            reports.append(report.report(name, result, listed))
            if outside is not None:
                outside.extend(
                    fidelity.Outside(analysis=name, excursion=excursion)
                    for excursion in result.excursions or ()
                )
    return solved, tuple(reports)


def _recording(
    analysis: Analysis,
    plan: Plan,
    *,
    recorded: bool,
    transistors: tuple[str, ...] = (),
) -> Analysis:
    """`analysis` recording what it names, or else the `plan`'s record, when `recorded`.

    An operating point records the `transistors` besides, whose regions its report
    lists. Otherwise it records nothing, beyond what the checks read.
    """
    if not isinstance(analysis, Recorded):
        return analysis
    if not recorded:
        return replace(analysis, record=())
    record = plan.record if analysis.record is None else analysis.record
    if isinstance(analysis, OperatingPoint):
        record = (*(record or ()), *transistors)
    return replace(analysis, record=record)


def _call(
    script: Path,
    name: str,
    namespaces: dict[Path, dict[str, object]],
    *,
    required: bool = True,
    arguments: Mapping[str, object] | None = None,
) -> object:
    """Calls the function `name` of `script` with `arguments`.

    The script runs once however often it is used.
    A script that defines no function `name` returns None when it is not `required`.

    Raises:
        ScriptError: When the script defines no function `name` and it is required.
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
    """Describes `error` by the script lines it passed through, innermost first."""
    frames = [
        ir.Source(
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
    """Runs the job in the directory named by the only argument."""
    (argument,) = sys.argv[1:] if argv is None else argv
    directory = Path(argument)
    job = Job.model_validate_json((directory / JOB).read_bytes())
    (directory / OUTCOME).write_bytes(OUTCOMES.dump_json(run(job, directory)))


if __name__ == "__main__":
    main()
