# Copyright 2026 Maxence Faldor
"""The service layer: every command, defined once.

Each command is a request model and an async function that takes it with the
settings. The command line and the MCP server derive from them, so both front ends share
one set of argument names, defaults and descriptions, and print the same text.

Scripts run in a child process (see [`jolty.runner`][jolty.runner]) within the
configured time limit, and what they print is captured and returned.
"""

# jolty's modules load as each command first uses them, so that a command that draws
# nothing never imports matplotlib.
__lazy_modules__ = ["jolty"]

import asyncio
import secrets
import shutil
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Self, override

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_settings import CliPositionalArg

from jolty import (
    catalog,
    checks,
    files,
    ir,
    kicad,
    plots,
    results,
    runner,
    schematic,
    shards,
    specs,
    viewer,
)
from jolty.errors import ElaborationError, JoltyError, NetlistError
from jolty.fidelity import FidelityReport
from jolty.report import (
    AnalysisReport,
    CornerReport,
    FailureReport,
    MonteCarloReport,
    ShardReport,
    VariantsReport,
    WorstCaseReport,
)
from jolty.settings import Level, Settings
from jolty.simulate import Simulation
from jolty.specs import Verdict

_SCRIPT_DESCRIPTION = (
    "Python file defining build(), which returns the top Component of the circuit."
)
_CHECK_DESCRIPTION = "Check the circuit's structure after building it."
_ERROR_DESCRIPTION = (
    "Rules to report as errors, such as 'shorted-device', or 'all' for every rule; "
    "overrides the configured levels."
)
_WARN_DESCRIPTION = (
    "Rules to report as warnings, such as 'no-dc-path', or 'all' for every rule; "
    "overrides the configured levels."
)
_IGNORE_DESCRIPTION = (
    "Rules to leave out, such as 'shorted-device', or 'all' for every rule; "
    "overrides the configured levels."
)
_MAX_DIAGNOSTICS_DESCRIPTION = (
    "Diagnostics to list at most; defaults to the configured limit."
)
_SCHEMATIC_DESCRIPTION = (
    "Draw the top component's schematic beside the circuit, marking the instances the "
    "listed diagnostics name."
)
_ANNOTATED_DESCRIPTION = (
    "Draw the top component's schematic beside the circuit, marking the instances the "
    "listed diagnostics name and writing the voltages of the plan's first operating "
    "point."
)
_RECORD_DESCRIPTION = (
    "Instances whose currents, charges, derivatives and variables the results keep "
    "beside the unknowns, by path or pattern, such as 'M1', 'X1/**' or '!X1/bias/**', "
    "a pattern after ! taking back what earlier ones named; replaces the plan's record."
)
_PROBES_DESCRIPTION = (
    "Unknowns to plot, such as 'out', 'X1/mid' or 'V1:current'; defaults to the "
    "nets the top component names, up to eight."
)
_SOLVES_DESCRIPTION = (
    "The fewest frequencies or right-hand sides a task takes when a run spreads its "
    "small-signal analyses and sensitivities across threads; replaces the plan's. "
    "No result depends on it."
)
_INSTANCES_DESCRIPTION = (
    "The device instances a task evaluates when a run spreads its devices' evaluation "
    "across threads; replaces the plan's. No result depends on it."
)
_INSIDE_DESCRIPTION = (
    "The fewest unknowns a circuit has for a run to spread the work of its Newton "
    "iteration and its step control on each unknown across threads; replaces the "
    "plan's. No result depends on it."
)
_INTERIORS_DESCRIPTION = (
    "The most unknowns of an interior when a run factors its matrix in parallel, each "
    "large block ordered by nested dissection into interiors and the separators "
    "between them; replaces the plan's. It orders the factorization's operations, so "
    "results depend on it in their last digits, but not on the threads."
)
_SAMPLES_DESCRIPTION = (
    "Run only the samples of the plan's Monte Carlo analysis from START up to STOP, "
    "written START:STOP, such as 0:1000000, as a shard of the analysis: at the plan's "
    "own conditions and at every corner, running none of its other analyses, into "
    "the directory of shards, for jolty merge to report with the others. A shard "
    "draws no schematic and no plot, and traces only the probes named."
)
_SHARDS_DESCRIPTION = (
    "The directory a shard of samples is written to, as shard-START-STOP.npz; "
    "defaults to <script>.shards."
)


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CommandResult(_Model, ABC):
    """What a command produced, as data and as text."""

    @property
    @abstractmethod
    def ok(self) -> bool:
        """Whether the circuit is sound; if not, the command line exits with 1."""

    @abstractmethod
    def describe(self) -> str:
        """The result as text, for people and agents."""

    def images(self) -> tuple[Path, ...]:
        """The images the command wrote, which the MCP server sends with its text."""
        return ()


class ScriptOutput(_Model):
    """What a script printed."""

    stdout: str = Field(default="", description="Standard output, up to the limit.")
    stderr: str = Field(default="", description="Standard error, up to the limit.")
    truncated: bool = Field(
        default=False, description="Whether either stream was cut at the limit."
    )

    def describe(self) -> str:
        """The output as text, empty when the script printed nothing."""
        blocks = [
            f"script {title}:\n{_indent(text.rstrip())}"
            for title, text in (("output", self.stdout), ("errors", self.stderr))
            if text.strip()
        ]
        if self.truncated:
            blocks.append("(output cut at the configured limit)")
        return "\n\n".join(blocks)


class Plot(_Model):
    """A plot of an analysis's results."""

    analysis: str = Field(description="The analysis's name.")
    path: Path = Field(description="The PNG image.")


class CircuitSummary(_Model):
    """A saved circuit, and whether the engine elaborates it."""

    path: Path = Field(description="The saved circuit file.")
    components: int = Field(description="Components the circuit defines.")
    devices: int | None = Field(
        description="Devices with every component instance unfolded; null when the "
        "circuit does not elaborate."
    )
    unknowns: int | None = Field(
        description="Node voltages and branch currents the engine solves for; null "
        "when the circuit does not elaborate."
    )
    problems: tuple[str, ...] = Field(
        description="Why the circuit does not elaborate, each naming an instance and "
        "the script line that placed it."
    )

    def describe(self) -> str:
        """The summary as text."""
        counts = [_plural(self.components, "component")]
        if self.devices is not None:
            counts.append(_plural(self.devices, "device"))
        if self.unknowns is not None:
            counts.append(_plural(self.unknowns, "unknown"))
        text = f"saved {self.path}: {', '.join(counts)}"
        if self.problems:
            text += f"\n\n{_unelaborated('the circuit', self.problems)}"
        return text


class BuildRequest(_Model):
    """Run a script's build(), save the circuit it returns, and elaborate it.

    The script runs as ordinary Python with the user's permissions, in its own process
    with a time limit; that process is not a sandbox.
    """

    script: CliPositionalArg[Path] = Field(description=_SCRIPT_DESCRIPTION)
    output: Path | None = Field(
        default=None,
        description="Where to save the circuit; defaults to <script>.jolty.json.",
    )
    check: bool = Field(default=True, description=_CHECK_DESCRIPTION)
    error: list[str] = Field(default_factory=list, description=_ERROR_DESCRIPTION)
    warn: list[str] = Field(default_factory=list, description=_WARN_DESCRIPTION)
    ignore: list[str] = Field(default_factory=list, description=_IGNORE_DESCRIPTION)
    max_diagnostics: int | None = Field(
        default=None, ge=1, description=_MAX_DIAGNOSTICS_DESCRIPTION
    )
    schematic: bool = Field(default=True, description=_SCHEMATIC_DESCRIPTION)


class BuildResult(CommandResult):
    """What a build produced."""

    circuit: CircuitSummary
    report: checks.Report | None = Field(
        description="What the structural checks found; null when checking is off or "
        "the circuit does not elaborate."
    )
    schematic: Path | None = Field(
        description="The top component's schematic; null when drawing is off."
    )
    output: ScriptOutput = Field(description="What the script printed.")
    seconds: float = Field(description="Seconds the command took.")

    @property
    @override
    def ok(self) -> bool:
        """Whether the circuit elaborates and its checks found no error."""
        return not self.circuit.problems and (self.report is None or self.report.ok)

    @override
    def describe(self) -> str:
        """The result as text."""
        blocks = [
            self.circuit.describe(),
            self.report.describe() if self.report else "",
            self.output.describe(),
            f"drew {self.schematic}" if self.schematic else "",
            f"took {self.seconds:.2f} s",
        ]
        return "\n\n".join(block for block in blocks if block)

    @override
    def images(self) -> tuple[Path, ...]:
        """The schematic."""
        return (self.schematic,) if self.schematic else ()


class SimulateRequest(_Model):
    """Build a script's circuit, run every analysis of a plan on it, check its specs.

    A plan is a Python file defining plan(), which returns a jolty Plan: named analyses,
    the numerical options they share, among them the instances whose currents and
    small-signal models the saved results record beside the unknowns, a temperature,
    corners, other temperatures or arguments to build() at which the analyses run
    again, a Monte Carlo analysis that runs them on samples of the circuit's
    tolerances, and variants, the values of many runs of it. It may define specs(),
    which
    returns jolty Specs: requirements on measures of the analyses' results, each of
    which passes, is marginal, fails or goes unmeasured, at its worst corner, and a
    yield target for the Monte Carlo samples. It can be the build script itself.
    Given samples, it runs only those of the Monte Carlo analysis, as a shard, which
    jolty merge reports with the others as one run of every sample.
    The scripts run as ordinary Python with the user's permissions, in their own process
    with a time limit; that process is not a sandbox.
    """

    script: CliPositionalArg[Path] = Field(description=_SCRIPT_DESCRIPTION)
    plan: Path | None = Field(
        default=None,
        description="Python file defining plan(), which returns a Plan; defaults to "
        "the script.",
    )
    output: Path | None = Field(
        default=None,
        description="Where to save the circuit; defaults to <script>.jolty.json.",
    )
    results: Path | None = Field(
        default=None,
        description="Where to save the results; defaults to <script>.results.npz.",
    )
    check: bool = Field(default=True, description=_CHECK_DESCRIPTION)
    error: list[str] = Field(default_factory=list, description=_ERROR_DESCRIPTION)
    warn: list[str] = Field(default_factory=list, description=_WARN_DESCRIPTION)
    ignore: list[str] = Field(default_factory=list, description=_IGNORE_DESCRIPTION)
    max_diagnostics: int | None = Field(
        default=None, ge=1, description=_MAX_DIAGNOSTICS_DESCRIPTION
    )
    schematic: bool = Field(default=True, description=_ANNOTATED_DESCRIPTION)
    plot: bool = Field(
        default=True,
        description="Plot every analysis but operating points and sensitivities "
        "beside the results.",
    )
    probes: list[str] = Field(default_factory=list, description=_PROBES_DESCRIPTION)
    record: list[str] | None = Field(default=None, description=_RECORD_DESCRIPTION)
    min_solves_per_task: int | None = Field(
        default=None, ge=1, description=_SOLVES_DESCRIPTION
    )
    min_instances_per_task: int | None = Field(
        default=None, ge=1, description=_INSTANCES_DESCRIPTION
    )
    min_unknowns_inside: int | None = Field(
        default=None, ge=1, description=_INSIDE_DESCRIPTION
    )
    max_unknowns_per_interior: int | None = Field(
        default=None, ge=1, description=_INTERIORS_DESCRIPTION
    )
    samples: str | None = Field(
        default=None, pattern=r"^[0-9]+:[0-9]+$", description=_SAMPLES_DESCRIPTION
    )
    shards: Path | None = Field(default=None, description=_SHARDS_DESCRIPTION)

    @model_validator(mode="after")
    def _sharded(self) -> Self:
        """Checks that the samples run from a start up to a later stop, if named.

        Raises:
            ValueError: When they do not, or a directory of shards takes no samples.
        """
        taken = self.taken()
        if taken is not None and taken.start >= taken.stop:
            message = f"samples run from a start up to a later stop, not {self.samples}"
            raise ValueError(message)
        if self.shards is not None and taken is None:
            message = "a directory of shards holds samples, which --samples names"
            raise ValueError(message)
        return self

    def taken(self) -> range | None:
        """The samples the request names, None for a run of the whole plan."""
        if self.samples is None:
            return None
        start, stop = self.samples.split(":")
        return range(int(start), int(stop))


class SimulateResult(CommandResult):
    """What a simulation produced."""

    circuit: CircuitSummary
    report: checks.Report | None = Field(
        description="What the structural checks found; null when checking is off or "
        "the circuit does not elaborate."
    )
    results: Path | None = Field(
        description="The saved results file; null when no analysis ran to its end."
    )
    analyses: tuple[AnalysisReport, ...] = Field(
        description="A report per analysis, in the plan's order."
    )
    corners: tuple[CornerReport, ...] = Field(
        description="A report per corner of the plan, in its order."
    )
    monte_carlo: MonteCarloReport | None = Field(
        description="What the plan's Monte Carlo analysis found; null without one."
    )
    variants: VariantsReport | None = Field(
        description="What the runs of the plan's variants found; null without them."
    )
    worst_case: WorstCaseReport | None = Field(
        description="What the plan's worst-case analysis found; null without one."
    )
    batches: Path | None = Field(
        description="The saved batches of the plan's Monte Carlo analysis and "
        "variants, at its own conditions; null without them."
    )
    shard: ShardReport | None = Field(
        default=None,
        description="What the shard of the plan's Monte Carlo samples ran; null "
        "unless the request names samples.",
    )
    shard_file: Path | None = Field(
        default=None,
        description="The shard's file; null unless the request names samples.",
    )
    specs: tuple[Verdict, ...] = Field(
        description="The verdict on each requirement of the plan's specs, in order; "
        "empty when the plan defines no specs()."
    )
    fidelity: FidelityReport | None = Field(
        description="The models the plan's fidelity chose, and its spot checks; null "
        "when it chooses none."
    )
    schematic: Path | None = Field(
        description="The top component's schematic; null when drawing is off."
    )
    plots: tuple[Plot, ...] = Field(
        description="A plot per analysis but operating points and sensitivities, "
        "then a histogram of each batch's measures and an envelope of each "
        "analysis it traced; empty when plotting is off or nothing ran to its end."
    )
    output: ScriptOutput = Field(description="What the scripts printed.")
    seconds: float = Field(description="Seconds the command took.")

    @override
    def images(self) -> tuple[Path, ...]:
        """The schematic, then the plots."""
        drawn = (self.schematic,) if self.schematic else ()
        return (*drawn, *(plot.path for plot in self.plots))

    @property
    @override
    def ok(self) -> bool:
        """Whether the circuit elaborates, checks clean, and meets its plan.

        It meets its plan when every analysis ends, at the plan's own conditions and
        at every corner, every requirement passes, if only marginally, the Monte Carlo
        analysis's yield and the variants' reach the specs' target, every requirement
        passes at its worst-case extremes, and every model its fidelity chose agrees
        with its circuit where a spot check looked.
        """
        checked = self.report is None or self.report.ok
        return (
            not self.circuit.problems
            and checked
            and not any(
                isinstance(analysis, FailureReport) for analysis in self.analyses
            )
            and all(corner.ok for corner in self.corners)
            and all(
                batch is None or batch.passing is None or batch.passing.met
                for batch in (self.monte_carlo, self.variants)
            )
            and (self.worst_case is None or self.worst_case.ok)
            and all(verdict.status in {"passed", "marginal"} for verdict in self.specs)
            and (self.fidelity is None or self.fidelity.ok)
        )

    @override
    def describe(self) -> str:
        """The result as text."""
        blocks = [
            self.circuit.describe(),
            self.report.describe() if self.report else "",
            self.output.describe(),
        ]
        saved = (self.results, self.batches, self.shard_file)
        blocks.extend(f"saved {path}" for path in saved if path is not None)
        if self.shard is not None:
            blocks.append(self.shard.describe())
        blocks.extend(analysis.describe() for analysis in self.analyses)
        blocks.extend(corner.describe() for corner in self.corners)
        batched = (self.monte_carlo, self.variants, self.worst_case)
        blocks.extend(found.describe() for found in batched if found is not None)
        if self.specs:
            blocks.append(specs.describe(self.specs))
        if self.fidelity is not None:
            blocks.append(self.fidelity.describe())
        if self.schematic:
            blocks.append(f"drew {self.schematic}")
        if self.plots:
            blocks.append(_describe_plots(self.plots))
        blocks.append(f"took {self.seconds:.2f} s")
        return "\n\n".join(block for block in blocks if block)


async def build(request: BuildRequest, settings: Settings) -> BuildResult:
    """Runs a script's ``build()`` in its own process, saves the circuit, elaborates it.

    Raises:
        JoltyError: When the script is missing, fails, runs out of time or returns no
            circuit.
    """
    started = time.perf_counter()
    chosen = _levels(request, settings)
    script = await _existing(request.script)
    job = runner.Job(script=script, max_unknowns=settings.max_unknowns)
    path = request.output or files.circuit_path(request.script)
    built, output = await _run(job, settings, results=None)
    circuit = await asyncio.to_thread(_save, built, path)
    report = _report(built, request, chosen, settings)
    drawn = (
        await asyncio.to_thread(
            _draw, built.circuit, files.schematic_path(path), marked=_marked(report)
        )
        if request.schematic
        else None
    )
    return BuildResult(
        circuit=circuit,
        report=report,
        schematic=drawn,
        output=output,
        seconds=time.perf_counter() - started,
    )


async def simulate(request: SimulateRequest, settings: Settings) -> SimulateResult:
    """Builds a script's circuit and runs every analysis of a plan on it.

    The results file holds the analyses that ran to their end. When none did, a results
    file left at its path by an earlier run is removed, so it never describes another
    circuit.

    Raises:
        JoltyError: When a script is missing, fails, runs out of time, or returns no
            circuit or no plan.
    """
    started = time.perf_counter()
    chosen = _levels(request, settings)
    script = await _existing(request.script)
    plan = script if request.plan is None else await _existing(request.plan)
    taken = request.taken()
    job = runner.Job(
        script=script,
        plan=plan,
        max_unknowns=settings.max_unknowns,
        check=request.check,
        record=request.record,
        # A shard's outcome is its measures, and the probes named alone besides.
        plot=request.plot and (taken is None or bool(request.probes)),
        probes=request.probes,
        parallelism=runner.Thresholds(
            min_solves_per_task=request.min_solves_per_task,
            min_instances_per_task=request.min_instances_per_task,
            min_unknowns_inside=request.min_unknowns_inside,
            max_unknowns_per_interior=request.max_unknowns_per_interior,
        ),
        samples=None if taken is None else (taken.start, taken.stop),
    )
    path = request.output or files.circuit_path(request.script)
    if taken is not None:
        directory = request.shards or files.shards_path(request.script)
        placed = directory / files.shard_name(taken.start, taken.stop)
        built, output = await _run(job, settings, results=None, shard=placed)
        circuit = await asyncio.to_thread(_save, built, path)
        return SimulateResult(
            circuit=circuit,
            report=_report(built, request, chosen, settings),
            results=None,
            analyses=(),
            corners=(),
            monte_carlo=None,
            variants=None,
            worst_case=None,
            batches=None,
            shard=built.shard,
            shard_file=placed if built.shard is not None else None,
            specs=(),
            fidelity=None,
            schematic=None,
            plots=(),
            output=output,
            seconds=time.perf_counter() - started,
        )
    target = request.results or files.results_path(request.script)
    batched = files.batches_path(target)
    built, output = await _run(job, settings, results=target, batches=batched)
    circuit = await asyncio.to_thread(_save, built, path)
    report = _report(built, request, chosen, settings)
    saved = await asyncio.to_thread(target.is_file)
    loaded = await asyncio.to_thread(_load, target) if saved else {}
    saved_batches = await asyncio.to_thread(batched.is_file)
    drawn = (
        await asyncio.to_thread(
            _draw,
            built.circuit,
            files.schematic_path(path),
            point=_first_point(loaded),
            marked=_marked(report),
        )
        if request.schematic
        else None
    )
    plotted = (
        await asyncio.to_thread(_plot, target, loaded, [], request.probes)
        if request.plot
        else ()
    )
    if request.plot and saved_batches:
        plotted += await asyncio.to_thread(_plot_batches, target, batched)
    return SimulateResult(
        circuit=circuit,
        report=report,
        results=target if saved else None,
        analyses=built.analyses,
        corners=built.corners,
        monte_carlo=built.monte_carlo,
        variants=built.variants,
        worst_case=built.worst_case,
        batches=batched if saved_batches else None,
        specs=built.specs,
        fidelity=built.fidelity,
        schematic=drawn,
        plots=plotted,
        output=output,
        seconds=time.perf_counter() - started,
    )


class MergeRequest(_Model):
    """Merge the shards of a plan's Monte Carlo analysis, and report them as one run.

    The shards are the shard-START-STOP.npz files that jolty simulate --samples writes
    into a directory, each the samples from START up to STOP. The merge refuses shards
    whose fingerprints differ, as they ran other circuits, plans, options, builds or
    platforms, and shards whose samples leave a gap or overlap; it joins their batches
    in the order of their samples, and reports every sample against the specs the
    shards measured, as a simulation of every sample in one run does, bit for bit. It
    saves the batch at the plan's own conditions and plots its distribution as a
    simulation does, and runs no script.
    """

    shards: CliPositionalArg[Path] = Field(
        description="The directory of the shards, such as amp.shards."
    )
    output: Path | None = Field(
        default=None,
        description="Where to save the merged batch; defaults to <name>.batches.npz "
        "beside the directory, its name without .shards.",
    )
    plot: bool = Field(
        default=True,
        description="Plot a histogram of each requirement's values, and the envelope "
        "of each analysis the shards traced, beside the merged batch.",
    )


class MergeResult(CommandResult):
    """What the merge of a Monte Carlo analysis's shards found."""

    shards: tuple[Path, ...] = Field(
        description="The shards merged, in the order of their samples."
    )
    monte_carlo: MonteCarloReport = Field(
        description="What the analysis found over every sample."
    )
    batches: Path = Field(
        description="The saved batch of the samples at the plan's own conditions."
    )
    plots: tuple[Plot, ...] = Field(
        description="A histogram of the batch's measures and an envelope of each "
        "analysis it traced; empty when plotting is off."
    )
    seconds: float = Field(description="Seconds the command took.")

    @property
    @override
    def ok(self) -> bool:
        """Whether the samples' yield reaches the specs' target, if they state one."""
        passing = self.monte_carlo.passing
        return passing is None or passing.met

    @override
    def describe(self) -> str:
        """The result as text."""
        blocks = [
            f"merged {_plural(len(self.shards), 'shard')}",
            f"saved {self.batches}",
            self.monte_carlo.describe(),
        ]
        if self.plots:
            blocks.append(_describe_plots(self.plots))
        blocks.append(f"took {self.seconds:.2f} s")
        return "\n\n".join(blocks)

    @override
    def images(self) -> tuple[Path, ...]:
        """The plots."""
        return tuple(plot.path for plot in self.plots)


async def merge(request: MergeRequest, settings: Settings) -> MergeResult:
    """Merges the shards in a directory, and saves and plots their batch; no settings.

    Raises:
        JoltyError: When the directory holds no shard, a file is not one, the shards'
            fingerprints differ, or their samples leave a gap or overlap.
    """
    del settings
    started = time.perf_counter()
    if not await asyncio.to_thread(request.shards.is_dir):
        message = f"no such directory of shards: {request.shards}"
        raise JoltyError(message)
    target = request.output or files.merged_path(request.shards)
    found, merged = await asyncio.to_thread(_merged, request.shards, target)
    plotted = (
        await asyncio.to_thread(_plot_batches, target, target) if request.plot else ()
    )
    return MergeResult(
        shards=found,
        monte_carlo=merged,
        batches=target,
        plots=plotted,
        seconds=time.perf_counter() - started,
    )


def _merged(directory: Path, target: Path) -> tuple[tuple[Path, ...], MonteCarloReport]:
    """Merges the shards in `directory`, saves their batch at `target`, and reports.

    Returns the shards, in the order of their samples, and what they found.

    Raises:
        JoltyError: When the shards do not merge.
    """
    shard, ordered = shards.merge(sorted(directory.glob(files.SHARD_PATTERN)))
    analysed, batch = shards.monte_carlo(shard)
    with TemporaryDirectory(prefix="jolty-") as scratch:
        written = Path(scratch) / target.name
        files.save_batches(written, {runner.MONTE_CARLO: batch})
        _place(written, target)
    return ordered, analysed


class PlotRequest(_Model):
    """Plot the analyses of a saved results file."""

    results: CliPositionalArg[Path] = Field(description="A .results.npz file.")
    analyses: list[str] = Field(
        default_factory=list,
        description="Analyses to plot, by name; every one but operating points and "
        "sensitivities by default.",
    )
    probes: list[str] = Field(default_factory=list, description=_PROBES_DESCRIPTION)


class PlotResult(CommandResult):
    """The plots drawn."""

    plots: tuple[Plot, ...] = Field(description="A plot per analysis.")

    @property
    @override
    def ok(self) -> bool:
        """Always: plotting finds nothing wrong."""
        return True

    @override
    def describe(self) -> str:
        """The plots, one line each."""
        return _describe_plots(self.plots) if self.plots else "nothing to plot"

    @override
    def images(self) -> tuple[Path, ...]:
        """The plots."""
        return tuple(plot.path for plot in self.plots)


async def plot(request: PlotRequest, settings: Settings) -> PlotResult:
    """Plots the analyses of a saved results file beside it; settings play no part.

    Raises:
        JoltyError: When the file is missing or holds no results, an analysis named is
            not in it or is taken at DC solutions, or a probe names no unknown.
    """
    del settings
    loaded = await asyncio.to_thread(_load, request.results)
    drawn = await asyncio.to_thread(
        _plot, request.results, loaded, request.analyses, request.probes
    )
    return PlotResult(plots=drawn)


def _load(path: Path) -> dict[str, results.AnalysisResult]:
    """The results saved at `path`, by analysis.

    Raises:
        JoltyError: When there is no such file.
    """
    try:
        return files.load_results(path)
    except FileNotFoundError:
        message = f"no such results file: {path}"
        raise JoltyError(message) from None


def _plot(
    path: Path,
    loaded: Mapping[str, results.AnalysisResult],
    analyses: list[str],
    probes: list[str],
) -> tuple[Plot, ...]:
    """Plots the `analyses` of the results `loaded` from `path`, or all, beside it.

    Raises:
        JoltyError: When an analysis named is not in the results or is taken at DC
            solutions, or a probe names no unknown.
    """
    missing = [name for name in analyses if name not in loaded]
    if missing:
        message = f"{path} holds no analysis named {', '.join(missing)}"
        raise JoltyError(message)
    drawn = []
    for name, result in loaded.items():
        if analyses and name not in analyses:
            continue
        if isinstance(
            result,
            results.OperatingPointResult
            | results.OperatingPointsResult
            | results.SensitivityResult
            | results.TransferFunctionResult,
        ):
            if analyses:
                message = f"{name} is taken at DC solutions: nothing to plot"
                raise JoltyError(message)
            continue
        nets = (
            result.nets
            if isinstance(result, results.DcSweepResult | results.TransientResult)
            else ()
        )
        chosen = (
            ()
            if isinstance(
                result,
                results.StabilityResult | results.NoiseResult | results.PoleZeroResult,
            )
            else tuple(probes) or plots.default_probes(result.unknowns, nets)
        )
        target = files.plot_path(path, name)
        plots.save(plots.plot(name, result, chosen), target)
        drawn.append(Plot(analysis=name, path=target))
    return tuple(drawn)


def _plot_batches(results_path: Path, path: Path) -> tuple[Plot, ...]:
    """Plots each batch saved at `path` beside the results at `results_path`.

    A batch draws a histogram of its measures, if it has any, and an envelope of
    each analysis it traced along points of an axis.
    """
    drawn = []
    for name, batch in files.load_batches(path).items():
        if batch.requirements:
            target = files.plot_path(results_path, name)
            plots.save(plots.histograms(name, batch), target)
            drawn.append(Plot(analysis=name, path=target))
        for traced in batch.result.traced:
            if traced.along == "point":
                continue
            named = f"{name}.{traced.analysis}"
            target = files.plot_path(results_path, named)
            plots.save(plots.envelope(name, traced), target)
            drawn.append(Plot(analysis=named, path=target))
    return tuple(drawn)


def _describe_plots(drawn: tuple[Plot, ...]) -> str:
    return "\n".join(f"plotted {plot.analysis}: {plot.path}" for plot in drawn)


class DrawRequest(_Model):
    """Draw the schematic of a saved circuit's top component, or of an instance's."""

    circuit: CliPositionalArg[Path] = Field(description="A .jolty.json circuit file.")
    instance: str = Field(
        default="",
        description="The path of the component instance to draw, such as 'X1' or "
        "'X1/X2'; the top component by default.",
    )
    results: Path | None = Field(
        default=None,
        description="A .results.npz file of the circuit, whose first operating point "
        "gives the voltage written on each net.",
    )
    output: Path | None = Field(
        default=None,
        description="Where to save the image; defaults to <stem>.schematic.png beside "
        "the circuit, or <stem>.X1.X2.schematic.png for the instance X1/X2.",
    )


class DrawResult(CommandResult):
    """The schematic drawn."""

    schematic: Path = Field(description="The PNG image.")

    @property
    @override
    def ok(self) -> bool:
        """Always: drawing finds nothing wrong."""
        return True

    @override
    def describe(self) -> str:
        """The schematic, in one line."""
        return f"drew {self.schematic}"

    @override
    def images(self) -> tuple[Path, ...]:
        """The schematic."""
        return (self.schematic,)


async def draw(request: DrawRequest, settings: Settings) -> DrawResult:
    """Draws a component of a saved circuit beside it; settings play no part.

    Raises:
        JoltyError: When a file is missing or holds no circuit, no component instance
            is at the path, or the results hold no operating point.
    """
    del settings
    circuit = await asyncio.to_thread(_load_circuit, request.circuit)
    point = None
    if request.results is not None:
        loaded = await asyncio.to_thread(_load, request.results)
        point = _first_point(loaded)
        if point is None:
            message = f"{request.results} holds no operating point"
            raise JoltyError(message)
    target = request.output or files.schematic_path(request.circuit, request.instance)
    drawn = await asyncio.to_thread(
        _draw, circuit, target, request.instance, point=point
    )
    return DrawResult(schematic=drawn)


class ViewRequest(_Model):
    """Write a 3D viewer of a saved circuit and its results, one HTML page to open.

    The viewer lays each component out on a board, its parts on their pins and its
    wires as traces, and plays every analysis of the results on it: each trace takes
    its net's voltage as a color, light runs along each lead as its current flows, and
    clicking a part shows its parameters and its state at each point. It loads three.js
    from jsDelivr when opened.
    """

    circuit: CliPositionalArg[Path] = Field(description="A .jolty.json circuit file.")
    results: Path | None = Field(
        default=None,
        description="A .results.npz file of the circuit, whose analyses the viewer "
        "plays; none shows the circuit alone.",
    )
    output: Path | None = Field(
        default=None,
        description="Where to save the page; defaults to <stem>.view.html beside the "
        "circuit.",
    )
    title: str = Field(
        default="", description="The page's title; the top component's name if empty."
    )


class ViewResult(CommandResult):
    """The 3D viewer written."""

    viewer: Path = Field(description="The HTML page.")

    @property
    @override
    def ok(self) -> bool:
        """Always: writing the viewer finds nothing wrong."""
        return True

    @override
    def describe(self) -> str:
        """The page, in one line."""
        return f"wrote the 3D viewer {self.viewer}"


async def view(request: ViewRequest, settings: Settings) -> ViewResult:
    """Writes the 3D viewer of a saved circuit and its results; settings play no part.

    Raises:
        JoltyError: When a file is missing or holds no circuit or no results.
    """
    del settings
    circuit = await asyncio.to_thread(_load_circuit, request.circuit)
    analyses = (
        await asyncio.to_thread(_load, request.results)
        if request.results is not None
        else {}
    )
    target = request.output or files.viewer_path(request.circuit)
    await asyncio.to_thread(
        viewer.write_viewer, target, circuit, analyses, request.title
    )
    return ViewResult(viewer=target)


class CostRequest(_Model):
    """Price a saved circuit's top component, or an instance's, with a catalog."""

    circuit: CliPositionalArg[Path] = Field(description="A .jolty.json circuit file.")
    catalog: Path = Field(description="A catalog file of parts and their prices.")
    instance: str = Field(
        default="",
        description="The path of the component instance to price, such as 'X1'; the "
        "top component by default.",
    )


class CostResult(CommandResult):
    """The bill of materials."""

    bill: catalog.Bill

    @property
    @override
    def ok(self) -> bool:
        """Whether a part builds every device instance."""
        return not self.bill.unpriced

    @override
    def describe(self) -> str:
        """The bill, a line per instance."""
        return self.bill.describe()


async def cost(request: CostRequest, settings: Settings) -> CostResult:
    """Prices a component of a saved circuit with a catalog; settings play no part.

    Raises:
        JoltyError: When a file is missing or holds no circuit or no catalog, or no
            component instance is at the path.
    """
    del settings
    circuit = await asyncio.to_thread(_load_circuit, request.circuit)
    parts = await asyncio.to_thread(_load_catalog, request.catalog)
    try:
        return CostResult(bill=catalog.bill(circuit, parts, request.instance))
    except KeyError as error:
        raise JoltyError(error.args[0]) from None


class KicadRequest(_Model):
    """Write a saved circuit's top component, or an instance's, as a KiCad netlist.

    Each part of its bill of materials becomes a footprint, which its catalog part
    states with the pads each terminal joins, and KiCad's PCB editor imports the file.
    """

    circuit: CliPositionalArg[Path] = Field(description="A .jolty.json circuit file.")
    catalog: Path = Field(
        description="A catalog file of parts, with their footprints and pins."
    )
    instance: str = Field(
        default="",
        description="The path of the component instance to write, such as 'X1'; the "
        "top component by default.",
    )
    output: Path | None = Field(
        default=None,
        description="Where to save the netlist; defaults to <stem>.net beside the "
        "circuit, or <stem>.X1.X2.net for the instance X1/X2.",
    )


class KicadResult(CommandResult):
    """The KiCad netlist written, or what keeps the component off a board."""

    netlist: kicad.Netlist | None = Field(
        default=None, description="The netlist, when it was written."
    )
    path: Path | None = Field(default=None, description="The netlist file written.")
    problems: tuple[str, ...] = Field(
        default=(),
        description="What keeps the component, or an instance by its path, off the "
        "board; the netlist is written only when there is none.",
    )

    @property
    @override
    def ok(self) -> bool:
        """Whether the netlist was written."""
        return not self.problems

    @override
    def describe(self) -> str:
        """The netlist, a line per footprint, or each problem."""
        if self.netlist is None or self.path is None:
            listed = "".join(f"\n  {problem}" for problem in self.problems)
            return f"cannot write the netlist:{listed}"
        return f"wrote {self.path}: {self.netlist.describe()}"


async def write_kicad(request: KicadRequest, settings: Settings) -> KicadResult:
    """Writes a component of a saved circuit as a KiCad netlist; settings play no part.

    Raises:
        JoltyError: When a file is missing or holds no circuit or no catalog, or no
            component instance is at the path.
    """
    del settings
    circuit = await asyncio.to_thread(_load_circuit, request.circuit)
    parts = await asyncio.to_thread(_load_catalog, request.catalog)
    try:
        written = await asyncio.to_thread(
            kicad.netlist, circuit, parts, request.instance
        )
    except (ElaborationError, NetlistError) as error:
        return KicadResult(problems=error.problems)
    target = request.output or files.netlist_path(request.circuit, request.instance)
    await asyncio.to_thread(written.save, target)
    return KicadResult(netlist=written, path=target)


def _load_catalog(path: Path) -> catalog.Catalog:
    """The catalog saved at `path`.

    Raises:
        JoltyError: When there is no such file, or it holds no catalog.
    """
    try:
        return catalog.Catalog.load(path)
    except FileNotFoundError:
        message = f"no such catalog file: {path}"
        raise JoltyError(message) from None
    except ValidationError as error:
        message = f"{path} is not a catalog file: {error.error_count()} errors"
        raise JoltyError(message) from None


def _draw(
    circuit: ir.Circuit,
    path: Path,
    instance: str = "",
    *,
    point: results.OperatingPointResult | None = None,
    marked: Collection[str] = (),
) -> Path:
    """Draws the component at `instance` to `path`, and returns `path`.

    Raises:
        JoltyError: When no component instance is at `instance`.
    """
    figure = schematic.draw(circuit, instance, point=point, marked=marked)
    path.parent.mkdir(parents=True, exist_ok=True)
    schematic.save(figure, path)
    return path


def _first_point(
    loaded: Mapping[str, results.AnalysisResult],
) -> results.OperatingPointResult | None:
    """The first operating point among the results, if any."""
    return next(
        (
            result
            for result in loaded.values()
            if isinstance(result, results.OperatingPointResult)
        ),
        None,
    )


def _marked(report: checks.Report | None) -> tuple[str, ...]:
    """The paths of the instances the listed diagnostics name."""
    if report is None:
        return ()
    return tuple(
        instance.path
        for diagnostic in report.diagnostics
        for instance in diagnostic.instances
    )


class CheckRequest(_Model):
    """Check the structure of a saved circuit file."""

    circuit: CliPositionalArg[Path] = Field(description="A .jolty.json circuit file.")
    error: list[str] = Field(default_factory=list, description=_ERROR_DESCRIPTION)
    warn: list[str] = Field(default_factory=list, description=_WARN_DESCRIPTION)
    ignore: list[str] = Field(default_factory=list, description=_IGNORE_DESCRIPTION)
    max_diagnostics: int | None = Field(
        default=None, ge=1, description=_MAX_DIAGNOSTICS_DESCRIPTION
    )


class CheckResult(CommandResult):
    """What a check found."""

    circuit: Path = Field(description="The circuit file.")
    problems: tuple[str, ...] = Field(
        description="Why the circuit does not elaborate, each naming an instance and "
        "the script line that placed it."
    )
    report: checks.Report | None = Field(
        description="What the structural checks found; null when the circuit does "
        "not elaborate."
    )

    @property
    @override
    def ok(self) -> bool:
        """Whether the circuit elaborates and its checks found no error."""
        return not self.problems and self.report is not None and self.report.ok

    @override
    def describe(self) -> str:
        """The result as text."""
        if self.report is None:
            return _unelaborated(str(self.circuit), self.problems)
        return self.report.describe()


async def check(request: CheckRequest, settings: Settings) -> CheckResult:
    """Elaborates a saved circuit and checks its structure.

    Raises:
        JoltyError: When the file is missing or holds no circuit, or a rule named in
            the options or settings does not exist.
    """
    chosen = _levels(request, settings)
    circuit = await asyncio.to_thread(_load_circuit, request.circuit)
    try:
        simulation = await asyncio.to_thread(Simulation, circuit)
    except ElaborationError as error:
        return CheckResult(
            circuit=request.circuit, problems=error.problems, report=None
        )
    limit = request.max_diagnostics or settings.max_diagnostics
    found = await asyncio.to_thread(simulation.check)
    return CheckResult(
        circuit=request.circuit,
        problems=(),
        report=checks.report(found, chosen, limit),
    )


class ExplainRequest(_Model):
    """Explain rules: the exact condition each checks, and its level."""

    rules: CliPositionalArg[list[str]] = Field(
        default_factory=list, description="Rules to explain; every rule by default."
    )


class RuleInfo(_Model):
    """One rule: its name, its levels and its condition."""

    name: str
    level: Level = Field(description="The level the settings give it.")
    default_level: Level = Field(description="The level it has unless configured.")
    condition: str = Field(description="The exact condition it checks.")


class ExplainResult(CommandResult):
    """The rules explained."""

    rules: tuple[RuleInfo, ...]

    @property
    @override
    def ok(self) -> bool:
        """Always: explaining finds nothing wrong."""
        return True

    @override
    def describe(self) -> str:
        """The rules as text, one paragraph each."""
        return "\n\n".join(
            f"{rule.name} ({rule.level}"
            + (
                ""
                if rule.level == rule.default_level
                else f", default {rule.default_level}"
            )
            + f"): {rule.condition}"
            for rule in self.rules
        )


async def explain(request: ExplainRequest, settings: Settings) -> ExplainResult:
    """Explains the rules `request` names, or every rule.

    Raises:
        JoltyError: When a name is no rule, or the settings name a rule that does not
            exist.
    """
    known = checks.rules()
    chosen = checks.levels(settings.rules)
    unknown = [name for name in request.rules if name not in known]
    if unknown:
        message = (
            f"no rule is named {', '.join(unknown)}; the rules are {', '.join(known)}"
        )
        raise JoltyError(message)
    names = request.rules or list(known)
    return ExplainResult(
        rules=tuple(
            RuleInfo(
                name=name,
                level=chosen[name],
                default_level=known[name].level,
                condition=known[name].condition,
            )
            for name in names
        )
    )


def _load_circuit(path: Path) -> ir.Circuit:
    """The circuit saved at `path`.

    Raises:
        JoltyError: When there is no such file, or it holds no circuit.
    """
    try:
        return files.load_circuit(path)
    except FileNotFoundError:
        message = f"no such circuit file: {path}"
        raise JoltyError(message) from None
    except ValidationError as error:
        message = f"{path} is not a circuit file: {error.error_count()} errors"
        raise JoltyError(message) from None


def _levels(
    request: BuildRequest | SimulateRequest | CheckRequest, settings: Settings
) -> dict[str, Level]:
    return checks.levels(
        settings.rules, error=request.error, warn=request.warn, ignore=request.ignore
    )


def _report(
    built: runner.Built,
    request: BuildRequest | SimulateRequest,
    chosen: dict[str, Level],
    settings: Settings,
) -> checks.Report | None:
    """The report of the checks the runner made, when the request asks for it."""
    if not request.check or built.problems:
        return None
    limit = request.max_diagnostics or settings.max_diagnostics
    return checks.report(built.diagnostics, chosen, limit)


async def _existing(script: Path) -> Path:
    """The absolute path of `script`.

    Raises:
        JoltyError: When there is no such file.
    """
    if not await asyncio.to_thread(script.is_file):
        message = f"no such script: {script}"
        raise JoltyError(message)
    return await asyncio.to_thread(script.resolve)


async def _run(
    job: runner.Job,
    settings: Settings,
    *,
    results: Path | None,
    batches: Path | None = None,
    shard: Path | None = None,
) -> tuple[runner.Built, ScriptOutput]:
    """Runs `job` in a child process, and moves the results it writes to `results`.

    The batches it writes go to `batches`, when given, and the shard to `shard`, all at
    once (see `_place`).

    Raises:
        JoltyError: When the job fails, runs out of time, or the child process ends
            before reporting.
    """
    with TemporaryDirectory(prefix="jolty-") as scratch:
        directory = Path(scratch)
        await asyncio.to_thread(
            (directory / runner.JOB).write_text, job.model_dump_json(), "utf-8"
        )
        try:
            async with asyncio.timeout(settings.timeout):
                output, code = await _child(directory, settings.max_output)
        except TimeoutError:
            message = (
                f"{job.script.name} did not finish within {settings.timeout:g} s "
                "and was stopped"
            )
            raise JoltyError(message) from None
        match await asyncio.to_thread(_read, directory / runner.OUTCOME):
            case runner.Built() as built:
                if results is not None:
                    await asyncio.to_thread(_move, directory / runner.RESULTS, results)
                if batches is not None:
                    await asyncio.to_thread(_move, directory / runner.BATCHES, batches)
                if shard is not None:
                    await asyncio.to_thread(_place, directory / runner.SHARD, shard)
                return built, output
            case runner.Failure() as failure:
                message = f"the script failed: {failure}"
            case None:
                message = f"the runner exited with code {code} before reporting"
    raise JoltyError(_with_output(message, output))


async def _child(directory: Path, limit: int) -> tuple[ScriptOutput, int]:
    """Runs the runner on `directory` and returns its output and exit code.

    When the caller stops waiting, through a timeout or a cancelled request, the child
    is killed before the cancellation goes on, so no script outlives its command.
    """
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-P",
        "-m",
        "jolty.runner",
        str(directory),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = _pipes(process)
        (out, out_cut), (err, err_cut) = await asyncio.gather(
            _read_capped(stdout, limit), _read_capped(stderr, limit)
        )
        code = await process.wait()
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    output = ScriptOutput(stdout=out, stderr=err, truncated=out_cut or err_cut)
    return output, code


def _pipes(
    process: asyncio.subprocess.Process,
) -> tuple[asyncio.StreamReader, asyncio.StreamReader]:
    """The standard output and error of `process`, started with both as pipes.

    Raises:
        JoltyError: Never, unless the process was started without pipes.
    """
    if process.stdout is None or process.stderr is None:
        message = "could not capture the runner's output"
        raise JoltyError(message)
    return process.stdout, process.stderr


async def _read_capped(stream: asyncio.StreamReader, limit: int) -> tuple[str, bool]:
    """The first `limit` characters of `stream`, read to its end, and if it had more.

    It keeps no more bytes than `limit` characters can take, however much is written.
    """
    kept = bytearray()
    dropped = False
    while chunk := await stream.read(1 << 16):
        room = 4 * limit - len(kept)  # A character takes four bytes at most in UTF-8.
        kept += chunk[:room]
        dropped = dropped or len(chunk) > room
    text = kept.decode("utf-8", errors="replace")
    return text[:limit], dropped or len(text) > limit


def _read(path: Path) -> runner.Outcome | None:
    """The runner's outcome, or None when it exited before writing all of it."""
    try:
        return runner.OUTCOMES.validate_json(path.read_bytes())
    except FileNotFoundError, ValidationError:
        return None


def _move(source: Path, destination: Path) -> None:
    """Moves `source` to `destination`, or removes `destination` without a source."""
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, destination)
    else:
        destination.unlink(missing_ok=True)


def _place(source: Path, destination: Path) -> None:
    """Moves `source` to `destination` at once, or removes `destination` without one.

    A reader of `destination` finds the whole file there, or the one there before,
    never part of it, even when the move is stopped: the file is copied to a hidden
    name beside `destination`, then renamed, which a file system does at once. A move
    killed outright leaves the hidden copy behind, which no merge reads.
    """
    if not source.is_file():
        destination.unlink(missing_ok=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    copy = destination.with_name(f".{destination.name}.{secrets.token_hex(8)}")
    try:
        shutil.copyfile(source, copy)
        copy.replace(destination)
    except BaseException:
        copy.unlink(missing_ok=True)
        raise


def _save(built: runner.Built, path: Path) -> CircuitSummary:
    """Saves the circuit `built` holds at `path`, at once, as `_place` moves a file.

    Shards of one analysis, run at once, each save the same circuit there.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="jolty-") as scratch:
        written = Path(scratch) / path.name
        files.save_circuit(written, built.circuit)
        _place(written, path)
    return CircuitSummary(
        path=path,
        components=len(built.circuit.components),
        devices=built.devices,
        unknowns=built.unknowns,
        problems=built.problems,
    )


def _with_output(message: str, output: ScriptOutput) -> str:
    described = output.describe()
    return f"{message}\n\n{described}" if described else message


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" if line else line for line in text.splitlines())


def _unelaborated(circuit: str, problems: Collection[str]) -> str:
    """Each of a circuit's `problems`, then how many stop it from elaborating."""
    count = _plural(len(problems), "error")
    return "\n\n".join([*problems, f"{circuit} does not elaborate: {count}"])


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"
