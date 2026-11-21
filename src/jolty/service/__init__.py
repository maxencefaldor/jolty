# Copyright 2026 Maxence Faldor
"""Service layer defining every jolty command once.

Each command pairs a request model with an async handler that accepts the request and
runtime settings. Both the CLI and the MCP server derive from these models so argument
names, defaults, descriptions, and formatted output match across front ends.

User scripts execute in a child process (see [`jolty.runner`][jolty.runner]) under the
configured timeout, with stdout and stderr captured and returned.
"""

# jolty's modules load as each command first uses them, so that a command that draws
# nothing never imports matplotlib and startup skips the simulation engine.
__lazy_modules__ = [
    "importlib.util",
    "jolty",
    "jolty.service._instructions",
    "jolty.simulate",
    "secrets",
    "shutil",
]

import asyncio
import contextlib
import importlib.util
import secrets
import shutil
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable, Collection, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Self, override

from pydantic import Field, ValidationError, model_validator
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
from jolty.errors import (
    ElaborationError,
    JoltyError,
    NetlistError,
    PlanError,
    close_names,
)
from jolty.fidelity import FidelityReport
from jolty.ir import FrozenModel
from jolty.report import (
    AnalysisReport,
    CornerReport,
    FailureReport,
    MonteCarloReport,
    ShardReport,
    VariantsReport,
    WorstCaseReport,
)
from jolty.service import _docs, _measure
from jolty.service._docs import (
    DOCS_URI_PREFIX,
    EXAMPLE_MIME_TYPE,
    GUIDE_MIME_TYPE,
    Document,
    DocumentSection,
    DocumentSummary,
    ReadDocsRequest,
    document_summaries,
    read_document,
    resolve_document,
)
from jolty.service._docs import (
    _cached_checkout as _cached_checkout,
)
from jolty.service._docs import (
    _docs_catalog as _docs_catalog,
)
from jolty.service._docs import (
    _load_docs_from_checkout as _load_docs_from_checkout,
)
from jolty.service._docs import (
    _repository_root as _repository_root,
)
from jolty.service._instructions import build_instructions
from jolty.service._measure import MeasureKind, MeasureRequest
from jolty.settings import Level, Settings
from jolty.simulate import Simulation
from jolty.specs import ExtendedReal, Verdict

__all__ = [
    "DOCS_URI_PREFIX",
    "EXAMPLE_MIME_TYPE",
    "GUIDE_MIME_TYPE",
    "BuildRequest",
    "BuildResult",
    "CheckRequest",
    "CheckResult",
    "CircuitSummary",
    "CommandResult",
    "CostRequest",
    "CostResult",
    "Document",
    "DocumentSection",
    "DocumentSummary",
    "DrawRequest",
    "DrawResult",
    "ExplainRequest",
    "ExplainResult",
    "ExportKicadRequest",
    "ExportKicadResult",
    "MeasureKind",
    "MeasureRequest",
    "MeasureResult",
    "MergeRequest",
    "MergeResult",
    "Plot",
    "PlotRequest",
    "PlotResult",
    "ReadDocsRequest",
    "ReadDocsResult",
    "RuleSummary",
    "ScriptOutput",
    "SimulateRequest",
    "SimulateResult",
    "ViewRequest",
    "ViewResult",
    "build",
    "build_instructions",
    "check",
    "cost",
    "document_summaries",
    "draw",
    "explain",
    "export_kicad",
    "measure",
    "merge",
    "plot",
    "read_docs",
    "read_document",
    "resolve_document",
    "simulate",
    "view",
]

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
    "Rules to omit, such as 'shorted-device', or 'all' for every rule; "
    "overrides the configured levels."
)
_MAX_DIAGNOSTICS_DESCRIPTION = (
    "Maximum diagnostics to list; defaults to the configured limit."
)
_SCHEMATIC_DESCRIPTION = (
    "Draw the top component's schematic next to the circuit, marking instances named "
    "in listed diagnostics."
)
_ANNOTATED_DESCRIPTION = (
    "Draw the top component's schematic next to the circuit, marking instances named "
    "in listed diagnostics and annotating voltages from the plan's first operating "
    "point."
)
_RECORD_DESCRIPTION = (
    "Instance paths or glob patterns (such as 'M1', 'X1/**', or '!X1/bias/**') whose "
    "currents, charges, derivatives, and variables are recorded in addition to "
    "unknowns. Patterns prefixed with ! exclude earlier matches; replaces the plan "
    "record list."
)
_TRACES_DESCRIPTION = (
    "Unknowns to plot, such as 'out', 'X1/mid', or 'V1:current'; defaults to up to "
    "eight named nets on the top component."
)
_SOLVES_DESCRIPTION = (
    "Minimum frequencies or right-hand sides per task when parallelizing small-signal "
    "analyses and sensitivities across threads; replaces the plan setting. Numerical "
    "results are unaffected."
)
_INSTANCES_DESCRIPTION = (
    "Minimum nonlinear device instances per task when parallelizing device evaluation "
    "across threads (linear devices are evaluated once and excluded); replaces the "
    "plan setting. Numerical results are unaffected."
)
_UNKNOWNS_DESCRIPTION = (
    "Minimum unknowns or stored matrix entries per task when parallelizing Newton "
    "iteration and step control across threads; replaces the plan setting. Numerical "
    "results are unaffected."
)
_INTERIORS_DESCRIPTION = (
    "Maximum unknowns per interior block during nested-dissection parallel matrix "
    "factorization; replaces the plan setting. Changing block boundaries alters "
    "operation order in the lowest floating-point bits, independently of thread count."
)
_SAMPLES_DESCRIPTION = (
    "Monte Carlo sample slice formatted as START:STOP (such as 0:1000000) to run as a "
    "shard across nominal conditions and corners without other analyses. Shard runs "
    "write to the shards directory for jolty merge, skip schematics and plots, and "
    "trace only requested unknowns."
)
_SHARDS_DESCRIPTION = (
    "Directory where sample shards are written as shard-START-STOP.npz; "
    "defaults to <script>.shards."
)


class CommandResult(FrozenModel, ABC):
    """Structured data and formatted text produced by a service command."""

    @property
    def ok(self) -> bool:
        """True when the command succeeded; false causes CLI exit status 1."""
        return True

    @abstractmethod
    def describe(self) -> str:
        """Formats the command result as human- and agent-readable text.

        Returns:
            Human- and agent-readable summary of the command outcome.
        """

    def images(self) -> tuple[Path, ...]:
        """Returns image paths written by the command for MCP attachment.

        Returns:
            PNG image paths written during execution, or an empty tuple by default.
        """
        return ()


class ScriptOutput(FrozenModel):
    """Captured standard output and standard error from a script execution."""

    stdout: str = Field(default="", description="Standard output, up to the limit.")
    stderr: str = Field(default="", description="Standard error, up to the limit.")
    truncated: bool = Field(
        default=False,
        description="True when stdout or stderr reached the capture limit.",
    )

    def describe(self) -> str:
        """Formats captured script output as text, or empty if nothing was written.

        Returns:
            Indented stdout and stderr blocks, or an empty string when silent.
        """
        blocks = [
            f"script {title}:\n{_indent(text.rstrip())}"
            for title, text in (("output", self.stdout), ("errors", self.stderr))
            if text.strip()
        ]
        if self.truncated:
            blocks.append("(output cut at the configured limit)")
        return "\n\n".join(blocks)


class Plot(FrozenModel):
    """Saved plot image for a single analysis."""

    analysis: str = Field(description="Name of the plotted analysis.")
    path: Path = Field(description="Path to the generated PNG image.")


class CircuitSummary(FrozenModel):
    """Summary of a saved circuit file and its elaboration status."""

    path: Path = Field(description="Path to the saved circuit file.")
    components: int = Field(description="Number of components defined by the circuit.")
    devices: int | None = Field(
        description="Number of unfolded device instances, or null if elaboration "
        "failed."
    )
    unknowns: int | None = Field(
        description="Number of node voltages and branch currents solved by the "
        "engine, or null if elaboration failed."
    )
    problems: tuple[str, ...] = Field(
        description="Elaboration errors, each naming the affected instance and "
        "build-script line."
    )

    def describe(self) -> str:
        """Formats the circuit summary and any elaboration errors as text.

        Returns:
            Saved circuit summary line plus any elaboration problem block.
        """
        counts = [_plural(self.components, "component")]
        if self.devices is not None:
            counts.append(_plural(self.devices, "device"))
        if self.unknowns is not None:
            counts.append(_plural(self.unknowns, "unknown"))
        text = f"saved {self.path}: {', '.join(counts)}"
        if self.problems:
            text += f"\n\n{_unelaborated('the circuit', self.problems)}"
        return text


class BuildRequest(FrozenModel):
    """Run a script's build(), save the circuit it returns, and elaborate it.

    The script runs as standard Python with the user's permissions in a separate timed
    subprocess; that subprocess is not a sandbox.
    """

    script: CliPositionalArg[Path] = Field(description=_SCRIPT_DESCRIPTION)
    output: Path | None = Field(
        default=None,
        description="Destination path for the saved circuit; defaults to "
        "<script>.jolty.json.",
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
    """Outcome of building, elaborating, and optionally checking a circuit."""

    circuit: CircuitSummary
    report: checks.CheckReport | None = Field(
        description="Structural check report, or null when checking is disabled or "
        "elaboration fails."
    )
    schematic: Path | None = Field(
        description="Path to the top component schematic, or null when drawing is off."
    )
    output: ScriptOutput = Field(description="Captured stdout and stderr from build.")
    seconds: float = Field(description="Elapsed wall-clock seconds for the command.")

    @property
    @override
    def ok(self) -> bool:
        """True when the circuit elaborates and structural checks find no errors."""
        return not self.circuit.problems and (self.report is None or self.report.ok)

    @override
    def describe(self) -> str:
        """Formats the build result summary as text.

        Returns:
            Multi-paragraph summary of circuit size, checks, output, and elapsed time.
        """
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
        """Returns the generated schematic path when drawing was enabled.

        Returns:
            Single-element tuple containing `schematic`, or empty when omitted.
        """
        return (self.schematic,) if self.schematic else ()


class SimulateRequest(FrozenModel):
    """Build a script's circuit, run every analysis in a plan, and check its specs.

    A plan script defines plan() returning a jolty Plan with named analyses, shared
    numerical options, recorded instance patterns, temperature, operating or process
    corners, Monte Carlo tolerances, and batch variants. The plan script may also
    define specs() returning jolty Specs with requirements evaluated at the worst
    corner and an optional Monte Carlo yield target. The plan path defaults to the
    build script itself. When samples are given, simulation runs only that Monte Carlo
    shard for later combination via jolty merge. Scripts run as standard Python with
    the user's permissions in a separate timed subprocess; that subprocess is not a
    sandbox.
    """

    script: CliPositionalArg[Path] = Field(description=_SCRIPT_DESCRIPTION)
    plan: Path | None = Field(
        default=None,
        description="Python file defining plan(), which returns a Plan; defaults to "
        "the script.",
    )
    output: Path | None = Field(
        default=None,
        description="Destination path for the saved circuit; defaults to "
        "<script>.jolty.json.",
    )
    results: Path | None = Field(
        default=None,
        description="Destination path for the saved results; defaults to "
        "<script>.results.npz.",
    )
    raw: Path | None = Field(
        default=None,
        description="Destination path for a SPICE raw file of completed operating "
        "point, DC sweep, small-signal response, and transient analyses; omitted when "
        "unset.",
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
        description="Plot all non-DC analyses (excluding operating points and "
        "sensitivities) next to the results.",
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
        """Validates that shard and raw options agree with one another.

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
        """Returns the requested sample index range, or None for a full plan run.

        Returns:
            Sample index `range(start, stop)` when `samples` is set, or ``None``.
        """
        if self.samples is None:
            return None
        start, stop = self.samples.split(":")
        return range(int(start), int(stop))


class SimulateResult(CommandResult):
    """Outcome of building a circuit, running its plan, and checking its specs."""

    circuit: CircuitSummary
    report: checks.CheckReport | None = Field(
        description="Structural check report, or null when checking is disabled or "
        "elaboration fails."
    )
    results: Path | None = Field(
        description="Path to the saved results file, or null when no analysis "
        "completed."
    )
    raw: Path | None = Field(
        default=None,
        description="Path to the saved SPICE raw file, or null unless --raw was set "
        "and a raw-writable analysis completed.",
    )
    analyses: tuple[AnalysisReport, ...] = Field(
        description="Report for each analysis, in plan order."
    )
    corners: tuple[CornerReport, ...] = Field(
        description="Report for each plan corner, in plan order."
    )
    monte_carlo: MonteCarloReport | None = Field(
        description="Monte Carlo analysis report, or null when no Monte Carlo "
        "analysis is configured."
    )
    variants: VariantsReport | None = Field(
        description="Batch variants report, or null when no variants are configured."
    )
    worst_case: WorstCaseReport | None = Field(
        description="Worst-case analysis report, or null when no worst-case analysis "
        "is configured."
    )
    batches: Path | None = Field(
        description="Path to saved nominal Monte Carlo and variant batches, or null "
        "when neither ran."
    )
    shard: ShardReport | None = Field(
        default=None,
        description="Summary of the executed Monte Carlo shard, or null unless "
        "samples were specified.",
    )
    shard_file: Path | None = Field(
        default=None,
        description="Path to the written shard file, or null unless samples were "
        "specified.",
    )
    specs: tuple[Verdict, ...] = Field(
        description="Verdict for each requirement in specs(), in order; empty when "
        "no specs() function is defined."
    )
    fidelity: FidelityReport | None = Field(
        description="Selected fidelity models and spot-check results, or null when "
        "no fidelity overrides are configured."
    )
    schematic: Path | None = Field(
        description="Path to the top component schematic, or null when drawing is off."
    )
    plots: tuple[Plot, ...] = Field(
        description="Plots for all non-DC analyses plus batch histograms and trace "
        "envelopes; empty when plotting is off or no analysis completed."
    )
    output: ScriptOutput = Field(description="Captured stdout and stderr from scripts.")
    seconds: float = Field(description="Elapsed wall-clock seconds for the command.")

    @override
    def images(self) -> tuple[Path, ...]:
        """Returns the schematic path followed by all generated plot paths.

        Returns:
            Ordered PNG paths for the schematic and every generated analysis plot.
        """
        drawn = (self.schematic,) if self.schematic else ()
        return (*drawn, *(plot.path for plot in self.plots))

    @property
    @override
    def ok(self) -> bool:
        """True when the circuit elaborates, checks clean, and satisfies the plan.

        A simulation satisfies its plan when every nominal and corner analysis
        completes, every spec requirement passes or is marginal, Monte Carlo and
        variant yields meet any target, worst-case extremes pass, and fidelity spot
        checks agree with the underlying circuit.
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
        """Formats the simulation result summary as text.

        Returns:
            Multi-paragraph summary of circuit, checks, analyses, specs, and files.
        """
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
    """Runs ``build()`` in a separate subprocess, saves the circuit, and elaborates it.

    Args:
        request: Build command parameters naming the script, output path, and rules.
        settings: Runtime settings controlling subprocess timeout, diagnostics, and
            output limits.

    Returns:
        Structured `BuildResult` summarizing elaboration, checks, and schematic output.

    Raises:
        JoltyError: When the script is missing, fails, times out, or returns no
            circuit.
    """
    started = time.perf_counter()
    chosen = _levels(request, settings)
    script = await _existing(request.script)
    job = runner.Job(
        script=script,
        max_unknowns=settings.max_unknowns,
        runtime=runner.RuntimeSettings(
            threads=settings.threads, analysis_log=settings.analysis_log
        ),
    )
    path = request.output or files.circuit_path(request.script)
    draw_schematic = request.schematic and _plots_available()
    async with _preloading_imaging(schematic_image=draw_schematic, plot_images=False):
        built, output = await _run(job, settings, results=None)
    circuit = await asyncio.to_thread(_save, built, path)
    report = _report(built, request, chosen, settings)
    drawn = (
        await asyncio.to_thread(
            _draw, built.circuit, files.schematic_path(path), marked=_marked(report)
        )
        if draw_schematic
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
    """Builds a script's circuit and runs every analysis in the plan on it.

    The results file contains all completed analyses. When no analysis completes, any
    stale results file at the target path is removed.

    Args:
        request: Simulation command parameters naming scripts, outputs, and options.
        settings: Runtime settings controlling subprocess timeout, diagnostics, and
            report limits.

    Returns:
        Structured `SimulateResult` summarizing analyses, specs, batches, and plots.

    Raises:
        JoltyError: When a script is missing, fails, times out, or returns no circuit
            or plan.
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
        # Shards evaluate measures plus any explicitly requested traces.
        plot=request.plot and (taken is None or bool(request.traces)),
        traces=request.traces,
        parallelism=runner.Thresholds(
            min_solves_per_task=request.min_solves_per_task,
            min_instances_per_task=request.min_instances_per_task,
            min_unknowns_per_task=request.min_unknowns_per_task,
            max_unknowns_per_interior=request.max_unknowns_per_interior,
        ),
        samples=None if taken is None else (taken.start, taken.stop),
        runtime=runner.RuntimeSettings(
            threads=settings.threads, analysis_log=settings.analysis_log
        ),
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
    can_plot = (request.schematic or request.plot) and _plots_available()
    draw_schematic = request.schematic and can_plot
    draw_plots = request.plot and can_plot
    async with _preloading_imaging(
        schematic_image=draw_schematic, plot_images=draw_plots
    ):
        built, output = await _run(job, settings, results=target, batches=batched)
    circuit = await asyncio.to_thread(_save, built, path)
    report = _report(built, request, chosen, settings)
    saved = await asyncio.to_thread(target.is_file)
    needs_results = request.raw is not None or draw_schematic or draw_plots
    loaded = await asyncio.to_thread(_load, target) if saved and needs_results else {}
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
        if draw_schematic
        else None
    )
    plotted = (
        await asyncio.to_thread(_plot, target, loaded, [], request.traces)
        if draw_plots
        else ()
    )
    if draw_plots and saved_batches:
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


class MergeRequest(FrozenModel):
    """Merge the shards of a plan's Monte Carlo analysis and report them as one run.

    Input shards are shard-START-STOP.npz files produced by jolty simulate --samples
    for sample ranges [START, STOP). Merging rejects shards with mismatched digests
    (different circuits, plans, options, builds, or platforms) or gaps and overlaps in
    sample ranges. Valid shards concatenate in sample order and produce a bit-identical
    report against the measured specs. Merging saves the nominal batch, plots the
    resulting distributions, and executes no user script.
    """

    shards: CliPositionalArg[Path] = Field(
        description="Directory containing shard files, such as amp.shards."
    )
    output: Path | None = Field(
        default=None,
        description="Destination path for the merged batch; defaults to "
        "<name>.batches.npz next to the directory with the .shards suffix removed.",
    )
    plot: bool = Field(
        default=True,
        description="Plot histograms for each requirement and trace envelopes next to "
        "the merged batch.",
    )


class MergeResult(CommandResult):
    """Outcome of merging Monte Carlo analysis shards."""

    shards: tuple[Path, ...] = Field(description="Merged shard paths in sample order.")
    monte_carlo: MonteCarloReport = Field(
        description="Aggregated Monte Carlo report across all merged samples."
    )
    batches: Path = Field(
        description="Path to the saved nominal batch across all merged samples."
    )
    plots: tuple[Plot, ...] = Field(
        description="Histograms for batch measures and envelopes for traced analyses; "
        "empty when plotting is off."
    )
    seconds: float = Field(description="Elapsed wall-clock seconds for the command.")

    @property
    @override
    def ok(self) -> bool:
        """True when sample yield meets the specs target, or when no target is set."""
        passing = self.monte_carlo.yield_
        return passing is None or passing.met

    @override
    def describe(self) -> str:
        """Formats the merge summary as text.

        Returns:
            Multi-paragraph summary of merged shards, batch path, yield, and plots.
        """
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
        """Returns the generated plot paths.

        Returns:
            PNG paths of all generated histogram and trace-envelope plots.
        """
        return tuple(plot.path for plot in self.plots)


async def merge(request: MergeRequest, settings: Settings) -> MergeResult:
    """Merges shards in a directory, saves their batch, and plots distributions.

    Args:
        request: Merge command parameters naming the shards directory and output path.
        settings: Runtime settings, unused by shard merging.

    Returns:
        Structured `MergeResult` reporting merged shards, Monte Carlo yield, and plots.

    Raises:
        JoltyError: When the directory contains no shards, a file is invalid, shard
            digests differ, or sample ranges have gaps or overlaps.
    """
    del settings
    started = time.perf_counter()
    if not await asyncio.to_thread(request.shards.is_dir):
        message = f"no such directory of shards: {request.shards}"
        raise JoltyError(message)
    target = request.output or files.merged_path(request.shards)
    found, merged = await asyncio.to_thread(_merged, request.shards, target)
    plotted = (
        await asyncio.to_thread(_plot_batches, target, target)
        if request.plot and _plots_available()
        else ()
    )
    return MergeResult(
        shards=found,
        monte_carlo=merged,
        batches=target,
        plots=plotted,
        seconds=time.perf_counter() - started,
    )


def _merged(directory: Path, target: Path) -> tuple[tuple[Path, ...], MonteCarloReport]:
    """Merges shards in `directory`, saves the nominal batch to `target`, and reports.

    Returns the ordered shard paths and the aggregated Monte Carlo report.

    Raises:
        JoltyError: When the shards fail validation or cannot be merged.
    """
    shard, ordered = shards.merge(sorted(directory.glob(files.SHARD_PATTERN)))
    analyzed, batch = shards.monte_carlo(shard)
    _write_atomic(
        target, lambda staged: files.save_batches(staged, {runner.MONTE_CARLO: batch})
    )
    return ordered, analyzed


class PlotRequest(FrozenModel):
    """Plot the analyses of a saved results file."""

    results: CliPositionalArg[Path] = Field(description="A .results.npz file.")
    analyses: list[str] = Field(
        default_factory=list,
        description="Analysis names to plot; defaults to all non-DC analyses "
        "(excluding operating points and sensitivities).",
    )
    traces: list[str] = Field(default_factory=list, description=_TRACES_DESCRIPTION)


class PlotResult(CommandResult):
    """Collection of generated analysis plots."""

    plots: tuple[Plot, ...] = Field(description="Generated plot per analysis.")

    @override
    def describe(self) -> str:
        """Formats the generated plot paths, one per line.

        Returns:
            Newline-separated plot output lines, or ``"nothing to plot"`` when empty.
        """
        return _describe_plots(self.plots) if self.plots else "nothing to plot"

    @override
    def images(self) -> tuple[Path, ...]:
        """Returns the generated plot paths.

        Returns:
            PNG paths for all generated analysis plots.
        """
        return tuple(plot.path for plot in self.plots)


async def plot(request: PlotRequest, settings: Settings) -> PlotResult:
    """Plots analyses from a saved results file next to it; settings are unused.

    Args:
        request: Plot command parameters naming the results archive, analyses, and
            traces.
        settings: Runtime settings, unused by result plotting.

    Returns:
        Structured `PlotResult` listing every generated plot image.

    Raises:
        JoltyError: When the optional `matplotlib` extra is missing, the file is
            missing or contains no results, a named analysis is absent or DC-only, or
            a trace names an unknown signal.
    """
    del settings
    _require_plots("plotting results")
    loaded = await asyncio.to_thread(_load, request.results)
    drawn = await asyncio.to_thread(
        _plot, request.results, loaded, request.analyses, request.traces
    )
    return PlotResult(plots=drawn)


class MeasureResult(CommandResult):
    """Single measure evaluated from a saved results file."""

    results: Path = Field(description="Path to the saved results file.")
    analysis: str = Field(description="Name of the evaluated analysis.")
    measure: MeasureKind = Field(description="Evaluated measure kind.")
    description: str = Field(
        description="Human-readable measure description, such as 'gain of out at "
        "1 kHz in bode'."
    )
    value: ExtendedReal = Field(
        description="Numeric value in SI base units, dB, degrees, or percent."
    )
    unit: str = Field(
        description="Unit string such as 'V', 'Hz', 's', 'dB', '°', '%', or '' for "
        "dimensionless values."
    )
    formatted: str = Field(
        description="Formatted value with SI prefix or unit, such as '2.5 V' or "
        "'-3.01 dB'."
    )

    @override
    def describe(self) -> str:
        """Formats the measure reading as a single line.

        Returns:
            Measure description paired with its formatted value.
        """
        return f"{self.description}: {self.formatted}"


async def measure(request: MeasureRequest, settings: Settings) -> MeasureResult:
    """Reads a measure from a saved results file without running any simulation.

    Args:
        request: Measure command parameters naming the results archive and target
            measure.
        settings: Runtime settings controlling maximum unknowns listed in errors.

    Returns:
        Structured `MeasureResult` holding the evaluated value, unit, and formatted
        text.

    Raises:
        JoltyError: When the file is missing or invalid, the requested analysis or
            unknown is missing, an option does not apply to the measure, or the
            measure cannot be read from the analysis result.
    """
    _measure.check_options(request)
    loaded = await asyncio.to_thread(_load, request.results)
    analysis_name, analysis_result = _measure.select_analysis(
        request.results, loaded, request.analysis
    )
    description, value, unit, formatted = await asyncio.to_thread(
        _measure.evaluate_measure,
        request,
        analysis_name,
        analysis_result,
        settings.max_unknowns,
    )
    return MeasureResult(
        results=request.results,
        analysis=analysis_name,
        measure=request.measure,
        description=description,
        value=value,
        unit=unit,
        formatted=formatted,
    )


def _load(path: Path) -> dict[str, results.AnalysisResult]:
    """Loads saved analysis results from `path`, keyed by analysis name.

    Raises:
        JoltyError: When `path` does not exist.
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
    """Plots requested `analyses` (or all plottable ones) from `loaded` next to `path`.

    Raises:
        JoltyError: When a named analysis is absent or DC-only, or a trace names an
            unknown signal.
    """
    missing = [name for name in analyses if name not in loaded]
    if missing:
        known = tuple(loaded)
        problems = [
            f"{path} contains no analysis named {name!r}{close_names(known, name)}"
            for name in missing
        ]
        raise PlanError(problems)
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
        nets = results.nets_of(result)
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
    """Plots each saved batch in `path` next to `results_path`.

    Each batch produces a histogram of its requirement measures (when present) and an
    envelope plot for every traced analysis along a multi-point axis.
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


class DrawRequest(FrozenModel):
    """Draw the schematic of a saved circuit's top component or a selected instance."""

    circuit: CliPositionalArg[Path] = Field(description="A .jolty.json circuit file.")
    instance: str = Field(
        default="",
        description="Path of the component instance to draw, such as 'X1' or 'X1/X2'; "
        "defaults to the top component.",
    )
    results: Path | None = Field(
        default=None,
        description="Saved .results.npz file whose first operating point annotates "
        "net voltages on the schematic.",
    )
    output: Path | None = Field(
        default=None,
        description="Destination path for the PNG image; defaults to "
        "<stem>.schematic.png next to the circuit, or <stem>.X1.X2.schematic.png for "
        "instance X1/X2.",
    )


class DrawResult(CommandResult):
    """Generated schematic image result."""

    schematic: Path = Field(description="Path to the generated PNG schematic.")

    @override
    def describe(self) -> str:
        """Formats the schematic output path as a single line.

        Returns:
            Single-line confirmation naming `schematic`.
        """
        return f"drew {self.schematic}"

    @override
    def images(self) -> tuple[Path, ...]:
        """Returns the generated schematic path.

        Returns:
            Single-element tuple containing `schematic`.
        """
        return (self.schematic,)


async def draw(request: DrawRequest, settings: Settings) -> DrawResult:
    """Draws a component schematic next to a saved circuit file; settings are unused.

    Args:
        request: Draw command parameters naming the circuit, instance, and optional
            results.
        settings: Runtime settings, unused by schematic rendering.

    Returns:
        Structured `DrawResult` naming the written schematic PNG path.

    Raises:
        JoltyError: When the optional `matplotlib` extra is missing, a file is missing
            or contains no circuit, no component instance exists at the path, or the
            results contain no operating point.
    """
    del settings
    _require_plots("drawing schematics")
    circuit = await asyncio.to_thread(_load_circuit, request.circuit)
    point = None
    if request.results is not None:
        loaded = await asyncio.to_thread(_load, request.results)
        point = _first_point(loaded)
        if point is None:
            message = f"{request.results} contains no operating point"
            raise JoltyError(message)
    target = request.output or files.schematic_path(request.circuit, request.instance)
    drawn = await asyncio.to_thread(
        _draw, circuit, target, request.instance, point=point
    )
    return DrawResult(schematic=drawn)


class ViewRequest(FrozenModel):
    """Write a standalone HTML 3D viewer for a saved circuit and its results.

    The viewer lays out components on a board with pins and wire traces and animates
    all loaded analyses. Trace colors encode net voltages, light pulses indicate lead
    currents, and clicking a part displays its parameters and state at each point. The
    generated HTML page loads three.js from jsDelivr when opened in a browser.
    """

    circuit: CliPositionalArg[Path] = Field(description="A .jolty.json circuit file.")
    results: Path | None = Field(
        default=None,
        description="Saved .results.npz file whose analyses the viewer animates; "
        "omitting it displays the circuit geometry alone.",
    )
    output: Path | None = Field(
        default=None,
        description="Destination path for the HTML page; defaults to <stem>.view.html "
        "next to the circuit.",
    )
    title: str = Field(
        default="",
        description="HTML page title; defaults to the top component name when empty.",
    )


class ViewResult(CommandResult):
    """Generated 3D HTML viewer result."""

    viewer: Path = Field(description="Path to the generated HTML page.")

    @override
    def describe(self) -> str:
        """Formats the 3D viewer output path as a single line.

        Returns:
            Single-line confirmation naming `viewer`.
        """
        return f"wrote the 3D viewer {self.viewer}"


async def view(request: ViewRequest, settings: Settings) -> ViewResult:
    """Writes a 3D HTML viewer for a saved circuit and its results; settings unused.

    Args:
        request: View command parameters naming the circuit, optional results, and
            title.
        settings: Runtime settings, unused by 3D viewer generation.

    Returns:
        Structured `ViewResult` naming the written HTML viewer path.

    Raises:
        JoltyError: When a file is missing or contains no valid circuit or results.
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
        viewer.save_viewer, target, circuit, analyses, request.title
    )
    return ViewResult(viewer=target)


class CostRequest(FrozenModel):
    """Price a saved circuit's top component or a selected instance with a catalog."""

    circuit: CliPositionalArg[Path] = Field(description="A .jolty.json circuit file.")
    catalog: Path = Field(description="Catalog file of parts and their prices.")
    instance: str = Field(
        default="",
        description="Path of the component instance to price, such as 'X1'; defaults "
        "to the top component.",
    )


class CostResult(CommandResult):
    """Priced bill of materials for a circuit component."""

    bill: catalog.Bill

    @property
    @override
    def ok(self) -> bool:
        """True when every device instance matched a priced catalog part."""
        return not self.bill.unpriced

    @override
    def describe(self) -> str:
        """Formats the bill of materials, one line per instance.

        Returns:
            Human-readable bill of materials summary and total cost.
        """
        return self.bill.describe()


async def cost(request: CostRequest, settings: Settings) -> CostResult:
    """Prices a component of a saved circuit using a catalog; settings are unused.

    Args:
        request: Cost command parameters naming the circuit, catalog, and instance.
        settings: Runtime settings, unused by catalog pricing.

    Returns:
        Structured `CostResult` containing the priced bill of materials.

    Raises:
        JoltyError: When a file is missing or contains no valid circuit or catalog, or
            no component instance exists at the path.
    """
    del settings
    circuit = await asyncio.to_thread(_load_circuit, request.circuit)
    parts = await asyncio.to_thread(_load_catalog, request.catalog)
    try:
        return CostResult(bill=catalog.bill(circuit, parts, request.instance))
    except KeyError as error:
        raise JoltyError(error.args[0]) from None


class ExportKicadRequest(FrozenModel):
    """Write a saved circuit's top component or a selected instance as a KiCad netlist.

    Each part in the bill of materials becomes a footprint defined by its catalog
    entry and pad-to-terminal mapping for import into the KiCad PCB editor.
    """

    circuit: CliPositionalArg[Path] = Field(description="A .jolty.json circuit file.")
    catalog: Path = Field(
        description="Catalog file of parts with their footprints and pins."
    )
    instance: str = Field(
        default="",
        description="Path of the component instance to export, such as 'X1'; defaults "
        "to the top component.",
    )
    output: Path | None = Field(
        default=None,
        description="Destination path for the netlist; defaults to <stem>.net next to "
        "the circuit, or <stem>.X1.X2.net for instance X1/X2.",
    )


class ExportKicadResult(CommandResult):
    """Generated KiCad netlist or the list of blockers preventing export."""

    netlist: kicad.Netlist | None = Field(
        default=None, description="Generated netlist when export succeeded."
    )
    path: Path | None = Field(default=None, description="Path to the written netlist.")
    problems: tuple[str, ...] = Field(
        default=(),
        description="Blockers preventing PCB export for the component or an instance; "
        "the netlist is written only when this list is empty.",
    )

    @property
    @override
    def ok(self) -> bool:
        """True when the netlist was written with no export blockers."""
        return not self.problems

    @override
    def describe(self) -> str:
        """Formats the written netlist summary or each export blocker.

        Returns:
            Summary of the exported KiCad netlist, or an indented blocker list.
        """
        if self.netlist is None or self.path is None:
            listed = "".join(f"\n  {problem}" for problem in self.problems)
            return f"cannot write the netlist:{listed}"
        return f"wrote {self.path}: {self.netlist.describe()}"


async def export_kicad(
    request: ExportKicadRequest, settings: Settings
) -> ExportKicadResult:
    """Writes a component of a saved circuit as a KiCad netlist; settings are unused.

    Args:
        request: KiCad export parameters naming the circuit, catalog, instance, and
            output.
        settings: Runtime settings, unused by KiCad netlist export.

    Returns:
        Structured `ExportKicadResult` containing the written netlist or export
        blockers.

    Raises:
        JoltyError: When a file is missing or contains no valid circuit or catalog, or
            no component instance exists at the path.
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


def _plots_available() -> bool:
    """Returns whether the optional `matplotlib` dependency is installed."""
    return importlib.util.find_spec("matplotlib") is not None


def _require_plots(what: str) -> None:
    """Raises `JoltyError` when the optional `matplotlib` dependency is missing."""
    if not _plots_available():
        message = (
            f"{what} needs the optional `matplotlib` dependency; "
            "install it with `pip install 'jolty[plots]'`"
        )
        raise JoltyError(message)


def _draw(
    circuit: ir.Circuit,
    path: Path,
    instance: str = "",
    *,
    point: results.OperatingPointResult | None = None,
    marked: Collection[str] = (),
) -> Path:
    """Draws the component at `instance` to `path` and returns `path`.

    Raises:
        JoltyError: When no component instance exists at `instance`.
    """
    figure = schematic.draw(circuit, instance, point=point, marked=marked)
    path.parent.mkdir(parents=True, exist_ok=True)
    schematic.save(figure, path)
    return path


def _first_point(
    loaded: Mapping[str, results.AnalysisResult],
) -> results.OperatingPointResult | None:
    """Returns the first operating point result in `loaded`, if present."""
    return next(
        (
            result
            for result in loaded.values()
            if isinstance(result, results.OperatingPointResult)
        ),
        None,
    )


def _marked(report: checks.CheckReport | None) -> tuple[str, ...]:
    """Returns instance paths named by diagnostics in `report`."""
    if report is None:
        return ()
    return tuple(
        instance.path
        for diagnostic in report.diagnostics
        for instance in diagnostic.instances
    )


class CheckRequest(FrozenModel):
    """Check the structure of a saved circuit file."""

    circuit: CliPositionalArg[Path] = Field(description="A .jolty.json circuit file.")
    error: list[str] = Field(default_factory=list, description=_ERROR_DESCRIPTION)
    warn: list[str] = Field(default_factory=list, description=_WARN_DESCRIPTION)
    ignore: list[str] = Field(default_factory=list, description=_IGNORE_DESCRIPTION)
    max_diagnostics: int | None = Field(
        default=None, ge=1, description=_MAX_DIAGNOSTICS_DESCRIPTION
    )


class CheckResult(CommandResult):
    """Outcome of elaborating and checking a saved circuit."""

    circuit: Path = Field(description="Path to the checked circuit file.")
    problems: tuple[str, ...] = Field(
        description="Elaboration errors, each naming the affected instance and "
        "build-script line."
    )
    report: checks.CheckReport | None = Field(
        description="Structural check report, or null when elaboration fails."
    )

    @property
    @override
    def ok(self) -> bool:
        """True when the circuit elaborates and structural checks find no errors."""
        return not self.problems and self.report is not None and self.report.ok

    @override
    def describe(self) -> str:
        """Formats the check report or elaboration problems as text.

        Returns:
            Formatted structural check report, or elaboration failure summary.
        """
        if self.report is None:
            return _unelaborated(str(self.circuit), self.problems)
        return self.report.describe()


async def check(request: CheckRequest, settings: Settings) -> CheckResult:
    """Elaborates a saved circuit and checks its structure.

    Args:
        request: Check command parameters naming the circuit file and rule overrides.
        settings: Runtime settings supplying default rule levels and diagnostic limits.

    Returns:
        Structured `CheckResult` reporting elaboration errors or structural diagnostics.

    Raises:
        JoltyError: When the file is missing or contains no valid circuit, or a named
            rule in options or settings does not exist.
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


class ExplainRequest(FrozenModel):
    """Explain check rules, including each rule's condition and severity level."""

    rules: CliPositionalArg[list[str]] = Field(
        default_factory=list, description="Rules to explain; every rule by default."
    )


class RuleSummary(FrozenModel):
    """Metadata describing a single structural check rule."""

    name: str
    level: Level = Field(description="Active severity level from current settings.")
    default_level: Level = Field(description="Built-in default severity level.")
    condition: str = Field(description="Exact circuit condition checked by the rule.")


class ExplainResult(CommandResult):
    """Formatted explanation of structural check rules."""

    rules: tuple[RuleSummary, ...]

    @override
    def describe(self) -> str:
        """Formats the explained rules as text, one paragraph each.

        Returns:
            Blank-line-separated paragraphs explaining every selected rule.
        """
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
    """Explains the rules named in `request`, or all rules when none are listed.

    Args:
        request: Explain command parameters listing optional rule names.
        settings: Runtime settings supplying active rule severity overrides.

    Returns:
        Structured `ExplainResult` summarizing each rule's level and condition.

    Raises:
        JoltyError: When a requested or configured rule name does not exist.
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
            RuleSummary(
                name=name,
                level=chosen[name],
                default_level=known[name].level,
                condition=known[name].condition,
            )
            for name in names
        )
    )


class ReadDocsResult(CommandResult):
    """Listing of offline documents or the single requested document."""

    entries: tuple[DocumentSummary, ...] = Field(
        default=(),
        description="Available documents when listing; empty when reading one topic.",
    )
    document: Document | None = Field(
        default=None,
        description="Requested document with full text; null when listing.",
    )

    @override
    def describe(self) -> str:
        """Formats the requested document text or the grouped topic listing.

        Returns:
            Full document content without trailing newlines, or grouped topic listing.
        """
        if self.document is not None:
            return self.document.content.rstrip("\n")
        return _docs.describe_entries(self.entries)


async def read_docs(request: ReadDocsRequest, settings: Settings) -> ReadDocsResult:
    """Lists available offline documents or reads the topic named in `request`.

    Args:
        request: Documentation query parameters naming an optional topic and section.
        settings: Runtime settings, unused by offline documentation queries.

    Returns:
        Structured `ReadDocsResult` holding either summary entries or one `Document`.

    Raises:
        JoltyError: When the offline bundle is unavailable or the requested topic is
            invalid or unknown.
    """
    del settings
    if request.topic is None:
        entries = await asyncio.to_thread(document_summaries, request.section)
        return ReadDocsResult(entries=entries)
    document = await asyncio.to_thread(resolve_document, request.topic, request.section)
    return ReadDocsResult(document=document)


def _load_validated[Loaded](
    path: Path, kind: str, loader: Callable[[Path], Loaded]
) -> Loaded:
    """Loads a validated file of `kind` at `path`, raising `JoltyError` on failure."""
    try:
        return loader(path)
    except FileNotFoundError:
        message = f"no such {kind} file: {path}"
        raise JoltyError(message) from None
    except ValidationError as error:
        message = f"{path} is not a {kind} file: {error.error_count()} errors"
        raise JoltyError(message) from None


def _load_circuit(path: Path) -> ir.Circuit:
    """Loads the circuit saved at `path`."""
    return _load_validated(path, "circuit", files.load_circuit)


def _load_catalog(path: Path) -> catalog.Catalog:
    """Loads the parts catalog saved at `path`."""
    return _load_validated(path, "catalog", catalog.Catalog.load)


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
    """Builds the structural check report from runner diagnostics when requested."""
    if not request.check or built.problems:
        return None
    limit = request.max_diagnostics or settings.max_diagnostics
    return checks.report(built.diagnostics, chosen, limit)


async def _existing(script: Path) -> Path:
    """Resolves the absolute path of `script` after verifying the file exists.

    Raises:
        JoltyError: When `script` does not exist.
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


def _preload_imaging(*, schematic_image: bool, plot_images: bool) -> None:
    """Resolves lazy schematic and plotting imports while the runner executes."""
    if schematic_image:
        _ = schematic.draw
    if plot_images:
        _ = plots.plot


@contextlib.asynccontextmanager
async def _preloading_imaging(
    *, schematic_image: bool, plot_images: bool
) -> AsyncGenerator[None]:
    """Overlaps optional schematic and plotting imports with the wrapped runner call."""
    if not (schematic_image or plot_images):
        yield
        return
    preload = asyncio.create_task(
        asyncio.to_thread(
            _preload_imaging,
            schematic_image=schematic_image,
            plot_images=plot_images,
        )
    )
    try:
        yield
    except BaseException:
        preload.cancel()
        await asyncio.gather(preload, return_exceptions=True)
        raise
    await preload


async def _child(directory: Path, limit: int) -> tuple[ScriptOutput, int]:
    """Runs the runner on `directory` and returns its output and exit code.

    When the caller stops waiting, through a timeout or a canceled request, the child
    is killed, its pipes are read to the end, and it is reaped before the cancellation
    goes on, so no script outlives its command and no pipe stays open.
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
        stdout_pipe, stderr_pipe = _pipes(process)
        (stdout, stdout_truncated), (stderr, stderr_truncated) = await asyncio.gather(
            _read_capped(stdout_pipe, limit), _read_capped(stderr_pipe, limit)
        )
        code = await process.wait()
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.communicate()
        raise
    output = ScriptOutput(
        stdout=stdout, stderr=stderr, truncated=stdout_truncated or stderr_truncated
    )
    return output, code


def _pipes(
    process: asyncio.subprocess.Process,
) -> tuple[asyncio.StreamReader, asyncio.StreamReader]:
    """Returns stdout and stderr pipe readers for `process`.

    Raises:
        JoltyError: When `process` was started without stdout or stderr pipes.
    """
    if process.stdout is None or process.stderr is None:
        message = "could not capture the runner's output"
        raise JoltyError(message)
    return process.stdout, process.stderr


async def _read_capped(stream: asyncio.StreamReader, limit: int) -> tuple[str, bool]:
    """Reads `stream` to EOF, keeping at most `limit` characters and a truncation flag.

    Buffered bytes are bounded by the maximum UTF-8 width of `limit` characters
    regardless of total stream volume.
    """
    kept = bytearray()
    dropped = False
    while chunk := await stream.read(1 << 16):
        # A character takes four bytes at most in UTF-8.
        room = 4 * limit - len(kept)
        kept += chunk[:room]
        dropped = dropped or len(chunk) > room
    text = kept.decode("utf-8", errors="replace")
    return text[:limit], dropped or len(text) > limit


def _read(path: Path) -> runner.JobResult | None:
    """Reads the runner job result from `path`, or returns None if incomplete."""
    try:
        return runner.JOB_RESULTS.validate_json(path.read_bytes())
    except FileNotFoundError, ValidationError:
        return None


def _move(source: Path, destination: Path) -> None:
    """Moves `source` to `destination`, or removes `destination` when absent."""
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, destination)
    else:
        destination.unlink(missing_ok=True)


def _write_atomic(target: Path, writer: Callable[[Path], object]) -> None:
    """Writes `target` atomically via a hidden sibling file in the same directory."""
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f".{target.name}.{secrets.token_hex(8)}")
    try:
        writer(staged)
        staged.replace(target)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def _place(source: Path, destination: Path) -> None:
    """Copies `source` into `destination` atomically, or removes `destination`.

    Concurrent readers observe either the complete new file or the prior file, never a
    partial write, because the copy stages to a hidden sibling file before renaming.
    """
    if not source.is_file():
        destination.unlink(missing_ok=True)
        return
    _write_atomic(destination, lambda staged: shutil.copyfile(source, staged))


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
    _write_atomic(
        target, lambda staged: files.save_raw(staged, writable, binary=binary)
    )
    return target


def _save(built: runner.Built, path: Path) -> CircuitSummary:
    """Saves the circuit from `built` atomically to `path` via a sibling file.

    Concurrent shards of a single analysis safely write identical circuit files here.
    """
    _write_atomic(path, lambda staged: files.save_circuit(staged, built.circuit))
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
    """Formats elaboration `problems` for `circuit` with the total error count."""
    count = _plural(len(problems), "error")
    return "\n\n".join([*problems, f"{circuit} does not elaborate: {count}"])


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"
