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

import ast
import asyncio
import importlib.resources
import json
import math
import re
import secrets
import shutil
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final, Literal, Self, override

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
from jolty.specs import Edge, ExtendedReal, Verdict
from jolty.units import (
    CURRENT,
    DIMENSIONLESS,
    FREQUENCY,
    TEMPERATURE,
    TIME,
    VOLTAGE,
    Dimension,
    Quantity,
    format_si,
)
from jolty.unknowns import Unknown

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
_TRACES_DESCRIPTION = (
    "Unknowns to plot, such as 'out', 'X1/mid' or 'V1:current'; defaults to the "
    "nets the top component names, up to eight."
)
_SOLVES_DESCRIPTION = (
    "The fewest frequencies or right-hand sides a task takes when a run spreads its "
    "small-signal analyses and sensitivities across threads, once they fill one for "
    "every thread; "
    "replaces the plan's. No result depends on it."
)
_INSTANCES_DESCRIPTION = (
    "The fewest device instances of a group a task evaluates when a run spreads its "
    "devices' evaluation across threads, once they fill one for every thread, a linear "
    "device, "
    "evaluated once a run, counting none; replaces the plan's. No result depends on it."
)
_UNKNOWNS_DESCRIPTION = (
    "The fewest unknowns, or stored entries of the matrix, a task takes when a run "
    "spreads the work of its Newton iteration and its step control on each across "
    "threads, once they fill one for every thread; replaces the plan's. No result "
    "depends on it."
)
_INTERIORS_DESCRIPTION = (
    "The most unknowns of an interior when a run orders a block of its matrix by "
    "nested dissection into interiors and the separators between them, which factor "
    "in parallel, as it does each larger block whose elimination that makes cheaper; "
    "replaces the plan's. It orders the factorization's operations, so results "
    "depend on it in their last digits, but not on the threads."
)
_SAMPLES_DESCRIPTION = (
    "Run only the samples of the plan's Monte Carlo analysis from START up to STOP, "
    "written START:STOP, such as 0:1000000, as a shard of the analysis: at the plan's "
    "own conditions and at every corner, running none of its other analyses, into "
    "the directory of shards, for jolty merge to report with the others. A shard "
    "draws no schematic and no plot, and traces only the unknowns its traces name."
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
    report: checks.CheckReport | None = Field(
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
    raw: Path | None = Field(
        default=None,
        description="Where to save a SPICE raw file of completed operating point, "
        "DC sweep, small-signal response and transient analyses; omitted when unset.",
    )
    raw_format: Literal["binary", "ascii"] = Field(
        default="binary",
        description="Encoding of the SPICE raw file named by --raw: 'binary' for "
        "little-endian doubles, or 'ascii'.",
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
    traces: list[str] = Field(default_factory=list, description=_TRACES_DESCRIPTION)
    record: list[str] | None = Field(default=None, description=_RECORD_DESCRIPTION)
    min_solves_per_task: int | None = Field(
        default=None, ge=1, description=_SOLVES_DESCRIPTION
    )
    min_instances_per_task: int | None = Field(
        default=None, ge=1, description=_INSTANCES_DESCRIPTION
    )
    min_unknowns_per_task: int | None = Field(
        default=None, ge=1, description=_UNKNOWNS_DESCRIPTION
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
        """Checks that shard and raw options agree with one another.

        Raises:
            ValueError: When samples are empty, shards lack samples, raw is set on a
                shard, or raw_format is changed without raw.
        """
        taken = self.taken()
        if taken is not None and taken.start >= taken.stop:
            message = f"samples run from a start up to a later stop, not {self.samples}"
            raise ValueError(message)
        if self.shards is not None and taken is None:
            message = "a directory of shards holds samples, which --samples names"
            raise ValueError(message)
        if self.raw is not None and taken is not None:
            message = "a shard runs Monte Carlo samples alone and writes no raw file"
            raise ValueError(message)
        if self.raw is None and self.raw_format != "binary":
            message = "--raw-format applies only when --raw is set"
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
    report: checks.CheckReport | None = Field(
        description="What the structural checks found; null when checking is off or "
        "the circuit does not elaborate."
    )
    results: Path | None = Field(
        description="The saved results file; null when no analysis ran to its end."
    )
    raw: Path | None = Field(
        default=None,
        description="The saved SPICE raw file; null unless --raw was set and a "
        "raw-writable analysis ran to its end.",
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
                batch is None or batch.yield_ is None or batch.yield_.met
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
        saved = (self.results, self.raw, self.batches, self.shard_file)
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
        # A shard's outcome is its measures, and the traces named alone besides.
        plot=request.plot and (taken is None or bool(request.traces)),
        traces=request.traces,
        parallelism=runner.Thresholds(
            min_solves_per_task=request.min_solves_per_task,
            min_instances_per_task=request.min_instances_per_task,
            min_unknowns_per_task=request.min_unknowns_per_task,
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
    saved_raw: Path | None = None
    if request.raw is not None:
        if saved:
            saved_raw = await asyncio.to_thread(
                _save_raw,
                loaded,
                request.raw,
                binary=request.raw_format == "binary",
            )
        else:
            await asyncio.to_thread(request.raw.unlink, missing_ok=True)
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
        await asyncio.to_thread(_plot, target, loaded, [], request.traces)
        if request.plot
        else ()
    )
    if request.plot and saved_batches:
        plotted += await asyncio.to_thread(_plot_batches, target, batched)
    return SimulateResult(
        circuit=circuit,
        report=report,
        results=target if saved else None,
        raw=saved_raw,
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
    whose digests differ, as they ran other circuits, plans, options, builds or
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
        passing = self.monte_carlo.yield_
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
            digests differ, or their samples leave a gap or overlap.
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
    traces: list[str] = Field(default_factory=list, description=_TRACES_DESCRIPTION)


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
            not in it or is taken at DC solutions, or a trace names no unknown.
    """
    del settings
    loaded = await asyncio.to_thread(_load, request.results)
    drawn = await asyncio.to_thread(
        _plot, request.results, loaded, request.analyses, request.traces
    )
    return PlotResult(plots=drawn)


type MeasureKind = Literal[
    "value",
    "gain",
    "phase",
    "bandwidth",
    "crossover",
    "phase_margin",
    "gain_margin",
    "maximum",
    "minimum",
    "peak_to_peak",
    "mean",
    "rms",
    "rise_time",
    "overshoot",
    "settling_time",
    "crossing",
    "frequency",
    "distortion",
    "period",
    "noise_density",
    "integrated_noise",
    "solutions",
    "damping",
    "dominant_pole",
]
"""Every measurement that ``jolty measure`` reads from a saved results file."""

_MEASURE_OPTION_FIELDS: Final[tuple[str, ...]] = (
    "signal",
    "reference",
    "over",
    "at",
    "start",
    "stop",
    "drop",
    "low",
    "high",
    "band",
    "level",
    "edge",
    "fundamental",
    "harmonics",
    "referred",
)

_WINDOW_FIELDS: Final = frozenset({"signal", "reference", "start", "stop"})
_TRANSFER_FIELDS: Final = frozenset({"signal", "reference", "over"})

_ALLOWED_MEASURE_OPTIONS: Final[dict[MeasureKind, frozenset[str]]] = {
    "value": frozenset({"signal", "reference", "at"}),
    "gain": _TRANSFER_FIELDS | {"at"},
    "phase": _TRANSFER_FIELDS | {"at"},
    "bandwidth": _TRANSFER_FIELDS | {"drop"},
    "crossover": _TRANSFER_FIELDS,
    "phase_margin": _TRANSFER_FIELDS,
    "gain_margin": _TRANSFER_FIELDS,
    "maximum": _WINDOW_FIELDS,
    "minimum": _WINDOW_FIELDS,
    "peak_to_peak": _WINDOW_FIELDS,
    "mean": _WINDOW_FIELDS,
    "rms": _WINDOW_FIELDS,
    "rise_time": _WINDOW_FIELDS | {"low", "high"},
    "overshoot": _WINDOW_FIELDS,
    "settling_time": _WINDOW_FIELDS | {"band"},
    "crossing": _WINDOW_FIELDS | {"level", "edge"},
    "frequency": _WINDOW_FIELDS | {"level"},
    "distortion": _WINDOW_FIELDS | {"fundamental", "harmonics"},
    "period": frozenset(),
    "noise_density": frozenset({"at", "referred"}),
    "integrated_noise": frozenset({"start", "stop", "referred"}),
    "solutions": frozenset(),
    "damping": frozenset(),
    "dominant_pole": frozenset(),
}


def _result_kind(result: results.AnalysisResult) -> str:
    """What kind of analysis result `result` is, in words."""
    kinds: dict[type, str] = {
        results.OperatingPointResult: "an operating point",
        results.DcSweepResult: "a DC sweep",
        results.AcResult: "a small-signal response",
        results.TransientResult: "a transient",
        results.PeriodicSteadyStateResult: "a periodic steady state",
        results.SensitivityResult: "sensitivities",
        results.MismatchResult: "a mismatch",
        results.StabilityResult: "a loop gain",
        results.NoiseResult: "a noise spectrum",
        results.PoleZeroResult: "poles and zeros",
        results.OperatingPointsResult: "DC solutions",
        results.TransferFunctionResult: "a transfer function",
    }
    return kinds.get(type(result), type(result).__name__)


_NATURE_DIMENSIONS: Final[dict[results.Nature, Dimension]] = {
    "voltage": VOLTAGE,
    "current": CURRENT,
    "temperature": TEMPERATURE,
    "state": DIMENSIONLESS,
}


class MeasureRequest(_Model):
    """Read a measurement from a saved results file without running a simulation.

    Evaluates any measure of ``jolty.specs`` (``value``, ``gain``, ``phase``,
    ``bandwidth``, ``crossover``, ``phase_margin``, ``gain_margin``, ``maximum``,
    ``minimum``, ``peak_to_peak``, ``mean``, ``rms``, ``rise_time``, ``overshoot``,
    ``settling_time``, ``crossing``, ``frequency``, ``distortion``, ``noise_density``,
    ``integrated_noise``, ``solutions``, ``damping``, ``dominant_pole``), plus
    ``period`` on a periodic steady state, directly from a saved ``.results.npz``
    file. Numeric options (``at``, ``start``, ``stop``, ``level``, ``fundamental``)
    are in SI base units: seconds for transients and periodic steady states, hertz
    for small-signal and noise analyses, and volts, amperes, kelvins or plain
    numbers for signal levels and DC sweeps.
    """

    results: CliPositionalArg[Path] = Field(description="A .results.npz file.")
    measure: MeasureKind = Field(
        description="Measurement to read from the analysis result."
    )
    analysis: str | None = Field(
        default=None,
        description="Name of the analysis in the results file; optional when the file "
        "holds a single analysis.",
    )
    signal: str | None = Field(
        default=None,
        description="Unknown to read, such as 'out', 'X1/mid' or 'V1:current'; omitted "
        "for a stability loop gain, period, noise, solutions or pole-zero measure.",
    )
    reference: str | None = Field(
        default=None,
        description="Reference net for a differential voltage measurement.",
    )
    over: str | None = Field(
        default=None,
        description="Divisor unknown for a small-signal transfer measurement (gain, "
        "phase, bandwidth, crossover, phase_margin, gain_margin).",
    )
    at: float | None = Field(
        default=None,
        description="Point along the analysis axis in SI base units: time in seconds "
        "for a transient or periodic steady state, swept level for a DC sweep, or "
        "frequency in hertz for gain, phase and noise_density.",
    )
    start: float | None = Field(
        default=None,
        description="Window start in SI base units: time in seconds, swept level, or "
        "frequency in hertz for integrated_noise; defaults to the start of the axis.",
    )
    stop: float | None = Field(
        default=None,
        description="Window stop in SI base units: time in seconds, swept level, or "
        "frequency in hertz for integrated_noise; defaults to the end of the axis.",
    )
    drop: float | None = Field(
        default=None,
        description="Gain drop in decibels for bandwidth; defaults to 3.0.",
    )
    low: float | None = Field(
        default=None,
        description="Lower step fraction (between 0 and 1) for rise_time; defaults "
        "to 0.1.",
    )
    high: float | None = Field(
        default=None,
        description="Upper step fraction (between 0 and 1) for rise_time; defaults "
        "to 0.9.",
    )
    band: float | None = Field(
        default=None,
        description="Settling band fraction (positive) for settling_time; defaults "
        "to 0.02.",
    )
    level: float | None = Field(
        default=None,
        description="Threshold level in SI base units of the signal (volts, amperes, "
        "kelvins or plain number) for crossing (required) or frequency (optional "
        "midpoint default).",
    )
    edge: Edge | None = Field(
        default=None,
        description="Crossing direction for crossing: 'rising', 'falling' or 'either'; "
        "defaults to 'either'.",
    )
    fundamental: float | None = Field(
        default=None,
        description="Fundamental frequency in hertz for distortion (required).",
    )
    harmonics: int | None = Field(
        default=None,
        ge=2,
        description="Highest harmonic counted for distortion; defaults to 9.",
    )
    referred: bool = Field(
        default=False,
        description="Whether noise_density or integrated_noise is referred to the "
        "input source.",
    )


class MeasureResult(CommandResult):
    """A measurement read from a saved results file."""

    results: Path = Field(description="The saved results file.")
    analysis: str = Field(description="The analysis read.")
    measure: MeasureKind = Field(description="The measure read.")
    description: str = Field(
        description="The measure in words, such as 'gain of out at 1 kHz in bode'."
    )
    value: ExtendedReal = Field(
        description="The value read, in SI base units, or in dB, degrees or percent."
    )
    unit: str = Field(
        description="The value's unit, such as 'V', 'Hz', 's', 'dB', '°', '%', or '' "
        "for a plain number."
    )
    formatted: str = Field(
        description="The value formatted with SI prefix or unit, such as '2.5 V' or "
        "'-3.01 dB'."
    )

    @property
    @override
    def ok(self) -> bool:
        """Always: a reading succeeded (invalid or unmeasured queries raise)."""
        return True

    @override
    def describe(self) -> str:
        """The reading as text, in one line."""
        return f"{self.description}: {self.formatted}"


async def measure(request: MeasureRequest, settings: Settings) -> MeasureResult:
    """Reads a measurement from a saved results file without running any simulation.

    Raises:
        JoltyError: When the file is missing or invalid, the requested analysis or
            unknown is missing, an option does not apply to the measure, or the
            measure cannot be read from the analysis result.
    """
    _check_measure_options(request)
    loaded = await asyncio.to_thread(_load, request.results)
    analysis_name, analysis_result = _select_analysis(
        request.results, loaded, request.analysis
    )
    description, reading = await asyncio.to_thread(
        _evaluate_measure,
        request,
        analysis_name,
        analysis_result,
        settings.max_unknowns,
    )
    if math.isnan(reading.value):
        message = f"{description} read not a number"
        raise JoltyError(message)
    unit = "" if reading.unit == DIMENSIONLESS else str(reading.unit)
    formatted = _format_reading(reading.value, unit)
    return MeasureResult(
        results=request.results,
        analysis=analysis_name,
        measure=request.measure,
        description=description,
        value=reading.value,
        unit=unit,
        formatted=formatted,
    )


def _check_measure_options(request: MeasureRequest) -> None:
    """Rejects options that do not apply to `request.measure`."""
    allowed = _ALLOWED_MEASURE_OPTIONS[request.measure]
    extra = [
        name
        for name in _MEASURE_OPTION_FIELDS
        if name not in allowed
        and (
            bool(getattr(request, name))
            if name == "referred"
            else getattr(request, name) is not None
        )
    ]
    if extra:
        listed = ", ".join(extra)
        message = f"measure {request.measure} takes no {listed}"
        raise JoltyError(message)


def _select_analysis(
    path: Path,
    loaded: Mapping[str, results.AnalysisResult],
    analysis: str | None,
) -> tuple[str, results.AnalysisResult]:
    """The named analysis from `loaded`, or the sole analysis when `analysis` is None.

    Raises:
        JoltyError: Naming available analyses when selection is missing or ambiguous.
    """
    if analysis is None:
        if not loaded:
            message = f"{path} holds no analyses"
            raise JoltyError(message)
        if len(loaded) > 1:
            available = ", ".join(loaded)
            message = (
                f"{path} holds {len(loaded)} analyses ({available}); analysis is unset"
            )
            raise JoltyError(message)
        sole = next(iter(loaded))
        return sole, loaded[sole]
    if analysis not in loaded:
        if loaded:
            available = ", ".join(loaded)
            message = (
                f"{path} holds no analysis named {analysis!r}; "
                f"the analyses are {available}"
            )
        else:
            message = (
                f"{path} holds no analysis named {analysis!r}; "
                "the file holds no analyses"
            )
        raise JoltyError(message)
    return analysis, loaded[analysis]


def _evaluate_measure(
    request: MeasureRequest,
    analysis_name: str,
    result: results.AnalysisResult,
    max_unknowns: int,
) -> tuple[str, specs.Reading]:
    """Constructs and evaluates the measure requested against `result`."""
    if request.measure == "period":
        if not isinstance(result, results.PeriodicSteadyStateResult):
            message = (
                f"period reads a periodic steady state, not {_result_kind(result)}"
            )
            raise JoltyError(message)
        return f"period in {analysis_name}", specs.Reading(result.period, TIME)
    try:
        built = _build_measure(request, analysis_name, result, max_unknowns)
        reading = built.read(result)
    except KeyError as error:
        available = _available_unknowns(result.unknowns, max_unknowns)
        message = f"{error.args[0]}; unknowns in {analysis_name}: {available}"
        raise JoltyError(message) from None
    except ValueError as error:
        raise JoltyError(str(error)) from None
    return str(built), reading


def _build_measure(
    request: MeasureRequest,
    analysis_name: str,
    result: results.AnalysisResult,
    max_unknowns: int,
) -> specs.Measure:
    """Builds the `jolty.specs.Measure` described by `request`."""
    kind = request.measure
    if kind in {"solutions", "damping", "dominant_pole"}:
        factories: dict[str, Callable[[str], specs.Measure]] = {
            "solutions": specs.solutions,
            "damping": specs.damping,
            "dominant_pole": specs.dominant_pole,
        }
        return factories[kind](analysis_name)
    if kind in {"noise_density", "integrated_noise"}:
        return _build_noise_measure(request, analysis_name)
    if kind in {
        "gain",
        "phase",
        "bandwidth",
        "crossover",
        "phase_margin",
        "gain_margin",
    }:
        return _build_transfer_measure(request, analysis_name, result, max_unknowns)
    signal = _require_signal(request, analysis_name, result, max_unknowns)
    if kind == "value":
        axis = _axis_dimension(result)
        point = None if request.at is None else Quantity(request.at, axis)
        return specs.value(analysis_name, signal, at=point, reference=request.reference)
    if kind in {"maximum", "minimum", "peak_to_peak", "mean", "rms"}:
        return _build_statistic_measure(request, analysis_name, signal, result)
    return _build_transient_measure(request, analysis_name, signal, result)


def _require_signal(
    request: MeasureRequest,
    analysis_name: str,
    result: results.AnalysisResult,
    max_unknowns: int,
) -> str:
    """Returns `request.signal` or raises a factual error listing available unknowns."""
    if request.signal is None:
        available = _available_unknowns(result.unknowns, max_unknowns)
        message = (
            f"measure {request.measure} takes a signal; "
            f"unknowns in {analysis_name}: {available}"
        )
        raise JoltyError(message)
    return request.signal


def _build_noise_measure(request: MeasureRequest, analysis_name: str) -> specs.Measure:
    """Builds a `noise_density` or `integrated_noise` measure."""
    if request.measure == "noise_density":
        if request.at is None:
            message = "measure noise_density takes a frequency as at, in Hz"
            raise JoltyError(message)
        return specs.noise_density(
            analysis_name,
            at=Quantity(request.at, FREQUENCY),
            referred=request.referred,
        )
    start = None if request.start is None else Quantity(request.start, FREQUENCY)
    stop = None if request.stop is None else Quantity(request.stop, FREQUENCY)
    return specs.integrated_noise(
        analysis_name, start=start, stop=stop, referred=request.referred
    )


def _build_transfer_measure(
    request: MeasureRequest,
    analysis_name: str,
    result: results.AnalysisResult,
    max_unknowns: int,
) -> specs.Measure:
    """Builds a small-signal response or stability loop-gain measure."""
    if isinstance(result, results.AcResult) and request.signal is None:
        available = _available_unknowns(result.unknowns, max_unknowns)
        message = (
            f"measure {request.measure} on a small-signal response takes a signal; "
            f"unknowns in {analysis_name}: {available}"
        )
        raise JoltyError(message)
    kind = request.measure
    if kind in {"gain", "phase"}:
        if request.at is None:
            message = f"measure {kind} takes a frequency as at, in Hz"
            raise JoltyError(message)
        builder = specs.gain if kind == "gain" else specs.phase
        return builder(
            analysis_name,
            request.signal,
            at=Quantity(request.at, FREQUENCY),
            over=request.over,
            reference=request.reference,
        )
    if kind == "bandwidth":
        drop = 3.0 if request.drop is None else request.drop
        return specs.bandwidth(
            analysis_name,
            request.signal,
            drop=drop,
            over=request.over,
            reference=request.reference,
        )
    builders = {
        "crossover": specs.crossover,
        "phase_margin": specs.phase_margin,
        "gain_margin": specs.gain_margin,
    }
    return builders[kind](
        analysis_name,
        request.signal,
        over=request.over,
        reference=request.reference,
    )


def _build_statistic_measure(
    request: MeasureRequest,
    analysis_name: str,
    signal: str,
    result: results.AnalysisResult,
) -> specs.Measure:
    """Builds `maximum`, `minimum`, `peak_to_peak`, `mean` or `rms`."""
    axis = _axis_dimension(result)
    start = None if request.start is None else Quantity(request.start, axis)
    stop = None if request.stop is None else Quantity(request.stop, axis)
    builders = {
        "maximum": specs.maximum,
        "minimum": specs.minimum,
        "peak_to_peak": specs.peak_to_peak,
        "mean": specs.mean,
        "rms": specs.rms,
    }
    return builders[request.measure](
        analysis_name, signal, start=start, stop=stop, reference=request.reference
    )


def _build_transient_measure(
    request: MeasureRequest,
    analysis_name: str,
    signal: str,
    result: results.AnalysisResult,
) -> specs.Measure:
    """Builds a transient timing, crossing, frequency or distortion measure."""
    start = None if request.start is None else Quantity(request.start, TIME)
    stop = None if request.stop is None else Quantity(request.stop, TIME)
    kind = request.measure
    if kind == "rise_time":
        return specs.rise_time(
            analysis_name,
            signal,
            low=0.1 if request.low is None else request.low,
            high=0.9 if request.high is None else request.high,
            start=start,
            stop=stop,
            reference=request.reference,
        )
    if kind == "overshoot":
        return specs.overshoot(
            analysis_name, signal, start=start, stop=stop, reference=request.reference
        )
    if kind == "settling_time":
        return specs.settling_time(
            analysis_name,
            signal,
            band=0.02 if request.band is None else request.band,
            start=start,
            stop=stop,
            reference=request.reference,
        )
    if kind == "crossing":
        if request.level is None:
            message = "measure crossing takes a level"
            raise JoltyError(message)
        dimension = _signal_dimension(result, signal)
        return specs.crossing(
            analysis_name,
            signal,
            Quantity(request.level, dimension),
            edge="either" if request.edge is None else request.edge,
            start=start,
            stop=stop,
            reference=request.reference,
        )
    if kind == "frequency":
        level = (
            None
            if request.level is None
            else Quantity(request.level, _signal_dimension(result, signal))
        )
        return specs.frequency(
            analysis_name,
            signal,
            level=level,
            start=start,
            stop=stop,
            reference=request.reference,
        )
    if request.fundamental is None:
        message = "measure distortion takes a fundamental frequency, in Hz"
        raise JoltyError(message)
    return specs.distortion(
        analysis_name,
        signal,
        fundamental=Quantity(request.fundamental, FREQUENCY),
        harmonics=9 if request.harmonics is None else request.harmonics,
        start=start,
        stop=stop,
        reference=request.reference,
    )


def _axis_dimension(result: results.AnalysisResult) -> Dimension:
    """The dimension of a transient's or DC sweep's axis; seconds otherwise."""
    if isinstance(result, results.DcSweepResult):
        if result.nature == "voltage":
            return VOLTAGE
        if result.nature == "current":
            return CURRENT
        return DIMENSIONLESS
    return TIME


def _signal_dimension(result: results.AnalysisResult, signal: str) -> Dimension:
    """The dimension of `signal` in `result.unknowns` (voltage for non-transients)."""
    if not isinstance(result, results.TransientResult):
        return VOLTAGE
    index = results.column(result.unknowns, signal)
    return _NATURE_DIMENSIONS[result.unknowns[index].nature]


def _available_unknowns(unknowns: tuple[Unknown, ...], limit: int) -> str:
    """The unknown names in `unknowns`, capped at `limit`."""
    if not unknowns:
        return "none"
    names = [unknown.name for unknown in unknowns]
    if len(names) <= limit:
        return ", ".join(names)
    shown = ", ".join(names[:limit])
    return f"{shown}, ... ({len(names) - limit} more)"


def _format_reading(value: float, unit: str) -> str:
    """`value` in `unit`, to four significant digits, with an SI prefix if any."""
    if unit in {"dB", "°", "%"}:
        return f"{value:.4g} {unit}"
    if not unit:
        return f"{value:.4g}"
    return format_si(value, unit)


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
    traces: list[str],
) -> tuple[Plot, ...]:
    """Plots the `analyses` of the results `loaded` from `path`, or all, beside it.

    Raises:
        JoltyError: When an analysis named is not in the results or is taken at DC
            solutions, or a trace names no unknown.
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
            | results.MismatchResult
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
            else tuple(traces) or plots.default_traces(result.unknowns, nets)
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
        for traces in batch.result.traces:
            if traces.along == "point":
                continue
            named = f"{name}.{traces.analysis}"
            target = files.plot_path(results_path, named)
            plots.save(plots.envelope(name, traces), target)
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


class ExportKicadRequest(_Model):
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


class ExportKicadResult(CommandResult):
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


async def export_kicad(
    request: ExportKicadRequest, settings: Settings
) -> ExportKicadResult:
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
        return ExportKicadResult(problems=error.problems)
    target = request.output or files.netlist_path(request.circuit, request.instance)
    await asyncio.to_thread(written.save, target)
    return ExportKicadResult(netlist=written, path=target)


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


def _marked(report: checks.CheckReport | None) -> tuple[str, ...]:
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
    report: checks.CheckReport | None = Field(
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


type DocumentSection = Literal["guide", "examples"]
"""Whether an offline document belongs to the guide or the example scripts."""

DOCS_URI_PREFIX: Final = "jolty://docs"
"""Base URI for offline documentation resources exposed over MCP."""

GUIDE_MIME_TYPE: Final = "text/markdown"
"""MIME type of offline guide pages."""

EXAMPLE_MIME_TYPE: Final = "text/x-python"
"""MIME type of offline example scripts."""

_DOCS_SECTIONS: Final[tuple[DocumentSection, ...]] = ("guide", "examples")
_DOCS_BUNDLE: Final = "docs.json"
_MARKDOWN_LINK: Final = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MIN_PRINTABLE_ASCII: Final = 32
_QUALIFIED_SEGMENT_COUNT: Final = 2
_REPOSITORY_PARENT_INDEX: Final = 2


_DOCUMENT_SECTION_DESCRIPTION: Final = (
    "Whether the document is a 'guide' page or an 'examples' script."
)
_DOCUMENT_SLUG_DESCRIPTION: Final = (
    "The topic slug, such as 'units', 'rc_filter' or 'cluster/amplifier'."
)
_DOCUMENT_URI_DESCRIPTION: Final = (
    "The MCP resource URI, such as 'jolty://docs/guide/units'."
)
_DOCUMENT_TITLE_DESCRIPTION: Final = (
    "The heading of a guide page, or the slug of an example script."
)
_DOCUMENT_SUMMARY_DESCRIPTION: Final = "The first sentence describing the document."
_DOCUMENT_MIME_DESCRIPTION: Final = (
    "MIME type: 'text/markdown' for guide pages, 'text/x-python' for examples."
)


class DocumentSummary(_Model):
    """Metadata for one offline guide page or example script."""

    section: DocumentSection = Field(description=_DOCUMENT_SECTION_DESCRIPTION)
    slug: str = Field(description=_DOCUMENT_SLUG_DESCRIPTION)
    uri: str = Field(description=_DOCUMENT_URI_DESCRIPTION)
    title: str = Field(description=_DOCUMENT_TITLE_DESCRIPTION)
    summary: str = Field(description=_DOCUMENT_SUMMARY_DESCRIPTION)
    mime_type: str = Field(description=_DOCUMENT_MIME_DESCRIPTION)


class Document(_Model):
    """One offline document: its metadata and full UTF-8 text."""

    section: DocumentSection = Field(description=_DOCUMENT_SECTION_DESCRIPTION)
    slug: str = Field(description=_DOCUMENT_SLUG_DESCRIPTION)
    uri: str = Field(description=_DOCUMENT_URI_DESCRIPTION)
    title: str = Field(description=_DOCUMENT_TITLE_DESCRIPTION)
    summary: str = Field(description=_DOCUMENT_SUMMARY_DESCRIPTION)
    mime_type: str = Field(description=_DOCUMENT_MIME_DESCRIPTION)
    content: str = Field(description="The full UTF-8 text of the document.")


class DocsRequest(_Model):
    """List or read jolty's offline guide pages and example scripts."""

    topic: CliPositionalArg[str | None] = Field(
        default=None,
        description="Topic to read: a bare slug such as 'units' or 'rc_filter', "
        "a qualified name such as 'guide/units' or 'examples/cluster/amplifier', "
        "or a resource URI such as 'jolty://docs/guide/units'. Lists all topics "
        "when omitted.",
    )
    section: DocumentSection | None = Field(
        default=None,
        description="Limit listing or lookup to 'guide' or 'examples'.",
    )


class DocsResult(CommandResult):
    """Either the listing of offline documents, or the single document requested."""

    entries: tuple[DocumentSummary, ...] = Field(
        default=(),
        description="Available documents when listing; empty when reading one topic.",
    )
    document: Document | None = Field(
        default=None,
        description="The requested document with its full text; null when listing.",
    )

    @property
    @override
    def ok(self) -> bool:
        """Always: listing or reading documentation finds nothing wrong."""
        return True

    @override
    def describe(self) -> str:
        """The document text when reading one, or the grouped listing of topics."""
        if self.document is not None:
            return self.document.content.rstrip("\n")
        sections: list[str] = []
        for name in _DOCS_SECTIONS:
            grouped = [entry for entry in self.entries if entry.section == name]
            if not grouped:
                continue
            noun = "page" if name == "guide" else "script"
            width = max(len(entry.slug) for entry in grouped)
            header = (
                f"{name} ({_plural(len(grouped), noun)}, "
                f"{DOCS_URI_PREFIX}/{name}/<slug>):"
            )
            lines = [header]
            for entry in grouped:
                detail = (
                    f"{entry.title}: {entry.summary}"
                    if name == "guide" and entry.summary
                    else (entry.summary or entry.title)
                )
                lines.append(f"  {entry.slug:<{width}}  {detail}")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)


def _repository_root() -> Path | None:
    """The repository root inferred from this module's location, if accessible."""
    try:
        parents = Path(__file__).resolve().parents
    except OSError:
        return None
    return (
        parents[_REPOSITORY_PARENT_INDEX]
        if len(parents) > _REPOSITORY_PARENT_INDEX
        else None
    )


def _load_directory_documents(directory: Path, suffix: str) -> dict[str, str]:
    """Reads every file under `directory` ending in `suffix`, keyed by POSIX slug."""
    documents: dict[str, str] = {}
    pending = [directory]
    while pending:
        current = pending.pop()
        for entry in sorted(current.iterdir(), key=lambda path: path.name):
            if entry.is_dir():
                if not entry.name.startswith(".") and entry.name != "__pycache__":
                    pending.append(entry)
                continue
            if entry.suffix == suffix and entry.is_file():
                slug = entry.relative_to(directory).with_suffix("").as_posix()
                documents[slug] = entry.read_text(encoding="utf-8")
    return dict(sorted(documents.items()))


def _load_docs_from_checkout(
    root: Path,
) -> dict[DocumentSection, dict[str, str]] | None:
    """Loads guide pages and examples from a checkout at `root`, if readable."""
    guide_directory = root / "docs" / "guide"
    examples_directory = root / "examples"
    try:
        if not (guide_directory.is_dir() and examples_directory.is_dir()):
            return None
        guide = _load_directory_documents(guide_directory, ".md")
        examples = _load_directory_documents(examples_directory, ".py")
    except OSError:
        return None
    if not guide or not examples:
        return None
    return {"guide": guide, "examples": examples}


def _parse_docs_bundle(raw: str) -> dict[DocumentSection, dict[str, str]] | None:
    """Parses a bundled ``docs.json`` payload into sorted section mappings."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    catalog_map: dict[DocumentSection, dict[str, str]] = {}
    for section in _DOCS_SECTIONS:
        entries = payload.get(section)
        if not isinstance(entries, dict):
            return None
        normalized: dict[str, str] = {}
        for slug, text in sorted(entries.items()):
            if not isinstance(slug, str) or not isinstance(text, str):
                return None
            normalized[slug] = text.replace("\r\n", "\n")
        if not normalized:
            return None
        catalog_map[section] = normalized
    return catalog_map


def _load_docs_from_bundle() -> dict[DocumentSection, dict[str, str]] | None:
    """Loads guide pages and examples from the packaged ``assets/docs.json`` bundle."""
    try:
        resource = importlib.resources.files("jolty").joinpath("assets", _DOCS_BUNDLE)
        raw = resource.read_text(encoding="utf-8")
    except ModuleNotFoundError, OSError:
        return None
    return _parse_docs_bundle(raw)


def _docs_catalog() -> dict[DocumentSection, dict[str, str]]:
    """Loads offline guide pages and examples from the checkout or packaged bundle.

    Raises:
        JoltyError: When neither a readable checkout nor a packaged bundle exists.
    """
    root = _repository_root()
    if root is not None:
        checkout = _load_docs_from_checkout(root)
        if checkout is not None:
            return checkout
    bundled = _load_docs_from_bundle()
    if bundled is not None:
        return bundled
    message = "offline documentation bundle is not available in this installation"
    raise JoltyError(message)


def _guide_title_and_summary(slug: str, content: str) -> tuple[str, str]:
    """Extracts the Markdown heading and first summary sentence of a guide page."""
    title = slug
    summary = ""
    seen_heading = False
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not seen_heading and stripped.startswith("# "):
            title = stripped.removeprefix("# ").strip()
            seen_heading = True
            continue
        if stripped.startswith("#"):
            continue
        summary = _MARKDOWN_LINK.sub(r"\1", stripped)
        break
    return title, summary


def _example_summary(content: str) -> str:
    """Extracts the first line of an example script's module docstring."""
    try:
        module = ast.parse(content)
    except SyntaxError:
        return ""
    docstring = ast.get_docstring(module)
    if docstring is None:
        return ""
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _build_summary(
    section: DocumentSection, slug: str, content: str
) -> DocumentSummary:
    """Builds metadata for one offline document."""
    uri = f"{DOCS_URI_PREFIX}/{section}/{slug}"
    if section == "guide":
        title, summary = _guide_title_and_summary(slug, content)
        mime_type = GUIDE_MIME_TYPE
    else:
        title = slug
        summary = _example_summary(content)
        mime_type = EXAMPLE_MIME_TYPE
    return DocumentSummary(
        section=section,
        slug=slug,
        uri=uri,
        title=title,
        summary=summary,
        mime_type=mime_type,
    )


def _available_topics_message(
    loaded: Mapping[DocumentSection, Mapping[str, str]],
    section: DocumentSection | None = None,
) -> str:
    """Formats the factual listing of available topics for error messages."""
    if section is not None:
        return f"the {section} topics are {', '.join(loaded[section])}"
    guide_names = ", ".join(loaded["guide"])
    example_names = ", ".join(loaded["examples"])
    return (
        f"the guide topics are {guide_names}; the examples topics are {example_names}"
    )


def _validate_segments(original: str, value: str, available: str) -> tuple[str, ...]:
    """Validates a topic path, rejecting traversal, absolute paths and control bytes.

    Raises:
        JoltyError: When `value` is empty, absolute, contains ``..``, backslashes,
            colons or control characters.
    """
    invalid_control = any(ord(character) < _MIN_PRINTABLE_ASCII for character in value)
    segments = tuple(value.split("/"))
    if (
        not value
        or value != value.strip()
        or invalid_control
        or "\\" in value
        or ":" in value
        or value.startswith("/")
        or value.endswith("/")
        or any(segment in {"", ".", ".."} for segment in segments)
    ):
        message = (
            f"invalid documentation topic {original!r}: a topic names a bare slug "
            f"with no '..', absolute path, backslash or control characters; {available}"
        )
        raise JoltyError(message)
    return segments


def _read_document_from(
    loaded: Mapping[DocumentSection, Mapping[str, str]],
    section: DocumentSection,
    slug: str,
) -> Document:
    """Reads one document from an already-loaded `loaded` catalog."""
    available = _available_topics_message(loaded, section)
    segments = _validate_segments(slug, slug, available)
    normalized = "/".join(segments)
    content = loaded[section].get(normalized)
    if content is None:
        message = f"no {section} topic is named {slug!r}; {available}"
        raise JoltyError(message)
    summary = _build_summary(section, normalized, content)
    return Document(
        section=summary.section,
        slug=summary.slug,
        uri=summary.uri,
        title=summary.title,
        summary=summary.summary,
        mime_type=summary.mime_type,
        content=content,
    )


def document_summaries(
    section: DocumentSection | None = None,
) -> tuple[DocumentSummary, ...]:
    """Lists available offline guide pages and examples in deterministic order.

    Raises:
        JoltyError: When neither a readable checkout nor a packaged bundle exists.
    """
    loaded = _docs_catalog()
    sections = _DOCS_SECTIONS if section is None else (section,)
    return tuple(
        _build_summary(current, slug, content)
        for current in sections
        for slug, content in loaded[current].items()
    )


def read_document(section: DocumentSection, slug: str) -> Document:
    """Reads one offline document from `section` by exact validated `slug`.

    Raises:
        JoltyError: When `slug` is invalid, attempts path traversal, or matches no
            document in `section`.
    """
    return _read_document_from(_docs_catalog(), section, slug)


def resolve_document(topic: str, section: DocumentSection | None = None) -> Document:
    """Resolves a bare slug, qualified name or ``jolty://docs/...`` URI to a document.

    Raises:
        JoltyError: When `topic` is invalid, mismatches `section`, is ambiguous, or
            matches no available document.
    """
    loaded = _docs_catalog()
    available = _available_topics_message(loaded, section)
    prefix = f"{DOCS_URI_PREFIX}/"
    trimmed = topic.removeprefix(prefix) if topic.startswith(prefix) else topic
    segments = _validate_segments(topic, trimmed, available)
    if len(segments) >= _QUALIFIED_SEGMENT_COUNT and segments[0] in _DOCS_SECTIONS:
        explicit: DocumentSection = "guide" if segments[0] == "guide" else "examples"
        if section is not None and section != explicit:
            message = (
                f"topic {topic!r} belongs to section {explicit!r}, not {section!r}"
            )
            raise JoltyError(message)
        return _read_document_from(loaded, explicit, "/".join(segments[1:]))
    slug = "/".join(segments)
    if section is not None:
        return _read_document_from(loaded, section, slug)
    matches = [candidate for candidate in _DOCS_SECTIONS if slug in loaded[candidate]]
    if len(matches) == 1:
        return _read_document_from(loaded, matches[0], slug)
    if len(matches) > 1:
        qualified = ", ".join(f"{candidate}/{slug}" for candidate in matches)
        message = f"topic {topic!r} is ambiguous and matches {qualified}"
        raise JoltyError(message)
    message = f"no documentation topic is named {topic!r}; {available}"
    raise JoltyError(message)


async def docs(request: DocsRequest, settings: Settings) -> DocsResult:
    """Lists available offline documents or reads the topic `request` names.

    Raises:
        JoltyError: When the offline bundle is unavailable or the requested topic is
            invalid or unknown.
    """
    del settings
    if request.topic is None:
        entries = await asyncio.to_thread(document_summaries, request.section)
        return DocsResult(entries=entries)
    document = await asyncio.to_thread(resolve_document, request.topic, request.section)
    return DocsResult(document=document)


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
) -> checks.CheckReport | None:
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


def _save_raw(
    loaded: Mapping[str, results.AnalysisResult],
    target: Path,
    *,
    binary: bool,
) -> Path:
    """Writes every raw-writable analysis in `loaded` atomically to `target`.

    Raises:
        JoltyError: When `loaded` contains no operating point, DC sweep, small-signal
            response or transient result; any earlier file at `target` is removed.
    """
    writable: dict[str, results.AnalysisResult] = {
        name: result
        for name, result in loaded.items()
        if isinstance(
            result,
            results.OperatingPointResult
            | results.DcSweepResult
            | results.AcResult
            | results.TransientResult,
        )
    }
    if not writable:
        target.unlink(missing_ok=True)
        kinds = ", ".join(
            f"{name} ({type(result).__name__})" for name, result in loaded.items()
        )
        message = (
            "a SPICE raw file holds operating points, DC sweeps, small-signal "
            f"responses and transients; none of the completed analyses match ({kinds})"
        )
        raise JoltyError(message)
    with TemporaryDirectory(prefix="jolty-") as scratch:
        written = Path(scratch) / target.name
        files.save_raw(written, writable, binary=binary)
        _place(written, target)
    return target


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
