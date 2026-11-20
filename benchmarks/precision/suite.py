# Copyright 2026 Maxence Faldor
"""Work-precision benchmark runner, scorer, and plotting suite.

``scripts/bench --precision`` invokes this module via ``benchmarks/speed/bench.py``.
Every fixture from ``fixtures.py`` runs on all installed simulators (jolty, ngspice 47
with Sparse 1.3 or KLU, and Xyce 7.10) through ``benchmarks/simulators.py``. Runs cover
default tolerances plus two quarter-decade sweeps from 1e-2 to 1e-8: varying `reltol`
alone, and scaling every tolerance together.

Each simulator output is scored against independent exact answers at the simulator's
chosen sample points. Scores report worst-point significant correct digits
(``-log₁₀ max |found - exact| / |exact|``) and a mixed form adding an output scale near
zero, capped by reference rounding. Each record also stores median engine time, step
counts, rejections, and Newton iterations.

Usage::

    scripts/bench --precision [FIXTURE ...]          every fixture, or those named
    scripts/bench --precision --simulator jolty      one simulator, or several
    scripts/bench --precision --sweeps reltol        one sweep, or both
    scripts/bench --precision --plot RESULTS         the diagrams and table of a file

Results are saved to ``benchmarks/precision/results/<commit>.json`` (with ``-dirty``
when tracked files differ), merging newly measured fixtures into existing commit files.
Per-fixture plots and Markdown tables (defaults and equal-error) are written to a
sibling directory. See ``benchmarks/precision/README.md`` for fixture descriptions.
"""

import math
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib import metadata
from pathlib import Path
from typing import Final, Literal

import numpy as np
from exact import EPSILON
from fixtures import Compared, Fixture, Found, ScoredOutput, fixtures
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter, FuncFormatter
from measuring import (
    Provenance,
    duration,
    load,
    load_average,
    markdown_table,
    provenance,
    save,
)
from pydantic import BaseModel, Field
from simulators import (
    RUSAGE,
    XYCE_STEPS,
    Jolty,
    Ngspice,
    Simulator,
    Unsupported,
    Xyce,
    command,
    netlist,
    ngspice_options_line,
    runner,
    stopped,
)

from jolty import circuit, foundry, plan, plots, results
from jolty.errors import JoltyError
from jolty.plan import Ac, Noise, Options, PeriodicSteadyState, PoleZero, Transient
from jolty.simulate import Elaborated, Simulation
from jolty.spice import export, ngspice, xyce
from jolty.units import VOLTAGE

RELATIVE_TOLERANCES: Final = tuple(10 ** (-2 - step / 4) for step in range(25))
"""The relative tolerances each sweep runs: 1e-2 to 1e-8 in quarter decades, as the
IVP test set's diagrams sweep theirs (Lioen and de Swart, 1998)."""

TOLERANCES: Final = (
    "relative_tolerance",
    "voltage_tolerance",
    "current_tolerance",
    "charge_tolerance",
)
"""The tolerances the scaled sweep scales together."""

DEFAULTS: Final[dict[str, float]] = {
    "relative_tolerance": plan.DEFAULTS["relative_tolerance"],
    "voltage_tolerance": plan.DEFAULTS["voltage_tolerance"].value,
    "current_tolerance": plan.DEFAULTS["current_tolerance"].value,
    "charge_tolerance": plan.DEFAULTS["charge_tolerance"].value,
}
"""The tolerances' defaults in SI base units, matching the engine and ngspice."""

SPICE_NAMES: Final = {name: foundry.SPICE_OPTIONS[name] for name in TOLERANCES}
"""ngspice's name of each tolerance."""

SIMULATORS: Final = (
    Simulator.JOLTY,
    Simulator.NGSPICE_SPARSE,
    Simulator.NGSPICE_KLU,
    Simulator.XYCE_SERIAL,
)
"""The simulators that run the precision suite, as the research's design names
them, and Xyce."""

XYCE_DEFAULTS: Final = {
    "nonlin": {"reltol": 1e-3, "abstol": 1e-12, "rhstol": 1e-6},
    "nonlin-tran": {"reltol": 1e-2, "abstol": 1e-6, "rhstol": 1e-2},
    "timeint": {"reltol": 1e-3, "abstol": 1e-6},
}
"""Default Xyce 7.10.0 tolerances by option block: ``NONLIN`` and ``NONLIN-TRAN``
(``N_NLS_NLParams.C``, lines 444-448 and 109-111) plus ``TIMEINT``
(``N_TIA_TIAParams.C``, lines 102 and 104)."""


class Sweep(StrEnum):
    """Tolerance sweep mode: varying reltol alone or scaling all tolerances."""

    RELTOL = "reltol"
    SCALED = "scaled"


@dataclass(frozen=True, slots=True)
class Setting:
    """Tolerance configuration: defaults or a sweep point at `relative_tolerance`."""

    sweep: str
    relative_tolerance: float

    def options(self) -> Options:
        """Returns jolty `Options` configured for this tolerance setting."""
        if self.sweep == "defaults":
            return Options()
        if self.sweep == Sweep.RELTOL:
            return Options(relative_tolerance=self.relative_tolerance)
        factor = self.relative_tolerance / plan.DEFAULTS["relative_tolerance"]
        return Options(
            relative_tolerance=self.relative_tolerance,
            voltage_tolerance=plan.DEFAULTS["voltage_tolerance"] * factor,
            current_tolerance=plan.DEFAULTS["current_tolerance"] * factor,
            charge_tolerance=plan.DEFAULTS["charge_tolerance"] * factor,
        )

    def xyce_options(self) -> tuple[str, ...]:
        """Builds Xyce ``.options`` directives for this setting, empty at defaults.

        Sets every group's `reltol` to `relative_tolerance`, scaling absolute tolerances
        by the same factor in `Sweep.SCALED`.
        """
        if self.sweep == "defaults":
            return ()
        factor = (
            self.relative_tolerance / DEFAULTS["relative_tolerance"]
            if self.sweep == Sweep.SCALED
            else 1.0
        )
        return tuple(
            f".options {group} "
            + " ".join(
                f"{name}="
                + repr(self.relative_tolerance if name == "reltol" else value * factor)
                for name, value in tolerances.items()
            )
            for group, tolerances in XYCE_DEFAULTS.items()
        )

    def tolerances(self) -> dict[str, float]:
        """Returns jolty tolerance overrides for this setting, empty at defaults."""
        if self.sweep == "defaults":
            return {}
        if self.sweep == Sweep.RELTOL:
            return {"relative_tolerance": self.relative_tolerance}
        factor = self.relative_tolerance / DEFAULTS["relative_tolerance"]
        return {name: value * factor for name, value in DEFAULTS.items()}


@dataclass(frozen=True, slots=True)
class Measured:
    """Raw simulation output, per-repeat engine times, and solver counters."""

    found: Found
    seconds: list[float]
    counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class Prepared:
    """Pre-elaborated fixture pairing an `Elaborated` circuit with indexed unknowns."""

    fixture: Fixture
    elaborated: Elaborated
    unknowns: Mapping[str, results.Unknown]

    @classmethod
    def from_fixture(cls, fixture: Fixture) -> Prepared:
        """Elaborates `fixture.top` once and indexes its unknowns by name."""
        elaborated = Elaborated(fixture.top)
        unknowns = Simulation.from_elaborated(elaborated).unknowns
        return cls(fixture, elaborated, {unknown.name: unknown for unknown in unknowns})


def settings(sweeps: Iterable[Sweep]) -> list[Setting]:
    """Returns the default setting followed by every tolerance point in `sweeps`."""
    swept = [
        Setting(sweep, tolerance)
        for sweep in sweeps
        for tolerance in RELATIVE_TOLERANCES
    ]
    return [Setting("defaults", DEFAULTS["relative_tolerance"]), *swept]


def run_jolty(prepared: Prepared, setting: Setting, repeats: int) -> Measured:
    """Runs `prepared.fixture` in jolty `repeats` times, each on a fresh `Simulation`.

    Creates a fresh `Simulation` per repeat so operating points solve from scratch.
    Discards one warm-up run and times ``Simulation.run`` (engine work plus Python
    result construction).
    """
    options = setting.options()
    seconds: list[float] = []
    # Discard one warm-up run, matching the speed suite timing protocol.
    Simulation.from_elaborated(prepared.elaborated).run(
        prepared.fixture.analysis, options
    )
    result: results.AnalysisResult | None = None
    for _ in range(repeats):
        simulation = Simulation.from_elaborated(prepared.elaborated)
        started = time.perf_counter()
        result = simulation.run(prepared.fixture.analysis, options)
        seconds.append(time.perf_counter() - started)
    if result is None:
        message = "a run repeats once at least"
        raise ValueError(message)
    return Measured(_found(result, prepared.fixture.reads), seconds, _counted(result))


def _found(result: results.AnalysisResult, reads: Sequence[str]) -> Found:
    """Extracts requested unknowns `reads` and coordinates from a jolty result."""
    match result:
        case (
            results.TransientResult(times=points)
            | results.DcSweepResult(swept=points)
            | results.AcResult(frequencies=points)
        ):
            values = {
                read: result.values[:, results.column(result.unknowns, read)]
                for read in reads
            }
            return Found(np.asarray(points), values)
        case results.NoiseResult():
            return Found(result.frequencies, {"noise": result.densities})
        case results.PoleZeroResult():
            return Found(np.empty(0), {"poles": result.poles})
    message = f"no fixture reads a {type(result).__name__}"
    raise TypeError(message)


def _counted(result: results.AnalysisResult) -> dict[str, int]:
    """Extracts transient step counts, rejections, and Newton iterations from `result`.

    Returns an empty dictionary for non-transient analyses.
    """
    if not isinstance(result, results.TransientResult):
        return {}
    statistics = result.statistics
    counted = {
        "accepted": statistics.accepted,
        "rejected": statistics.rejected,
        "transient iterations": statistics.iterations,
    }
    if statistics.work is not None:
        counted["iterations"] = statistics.work.iterations
    return counted


def run_ngspice(
    prepared: Prepared, setting: Setting, repeats: int, runner: Ngspice
) -> Measured:
    """Runs `prepared.fixture` `repeats` times in one ngspice process via `runner`.

    Exports the netlist via ``jolty.spice.export``, saves requested vectors, and applies
    setting tolerances and solver options at jolty's thermal temperature. Records
    incremental ``rusage`` analysis times per repeat and counts from the final repeat.

    Raises:
        Unsupported: When the simulator cannot run the analysis.
        JoltyError: When ngspice fails, aborts the analysis, runs another solver, or
            prints a time for another number of runs.
    """
    fixture = prepared.fixture
    reason = fixture.unsupported.get(runner.simulator)
    if reason is not None:
        raise Unsupported(reason)
    analysis = fixture.analysis
    if runner.simulator is Simulator.NGSPICE_KLU and isinstance(
        analysis, Noise | PoleZero
    ):
        message = (
            f"ngspice 47 runs no {type(analysis).__name__} with KLU: "
            "'not (yet) supported with option KLU'"
        )
        raise Unsupported(message)
    built = circuit(fixture.top)
    lines = netlist(built, (analysis,))
    vectors = {
        read: export.vector(built, prepared.unknowns[read]) for read in fixture.reads
    }
    if vectors:
        lines.append(f".save {' '.join(vectors.values())}")
    found = command(analysis, export.flatten(built))
    # The noise spectrum's densities squared, and its plot the one before the
    # integrated noise each run ends on.
    analyzed, after = (
        (f"set sqrnoise\n{found}", ["setplot previous"])
        if isinstance(analysis, Noise)
        else (found, [])
    )
    commands = "\n".join([*[f"{analyzed}\nrusage all"] * repeats, *after])
    given = [
        f"{SPICE_NAMES[name]}={value!r}" for name, value in setting.tolerances().items()
    ]
    options = ngspice_options_line(given, fallback=ngspice.NO_OPTIONS)
    (raw,) = runner.runs(lines, (commands,), options=options)
    if raw.printed.count(runner.solver) < repeats:
        message = f"ngspice did not run {runner.simulator}'s solver:\n{raw.printed}"
        raise JoltyError(message)
    printed = {
        name: [float(value) for value in pattern.findall(raw.printed)]
        for name, pattern in RUSAGE.items()
    }
    totals = printed.pop("analysis_seconds")
    if len(totals) != repeats:
        message = f"ngspice printed {len(totals)} analysis times for {repeats} runs"
        raise JoltyError(message)
    seconds = [b - a for a, b in zip([0.0, *totals], totals, strict=False)]
    kept = (
        printed
        if isinstance(analysis, Transient)
        else {"iterations": printed["iterations"]}
    )
    counts = {
        name: round(found[-1] - (found[-2] if len(found) > 1 else 0))
        for name, found in kept.items()
        if found
    }
    return Measured(_spice_found(raw, analysis, vectors), seconds, counts)


def run_xyce(
    prepared: Prepared, setting: Setting, repeats: int, runner: Xyce
) -> Measured:
    """Runs `prepared.fixture` `repeats` times in separate Xyce processes via `runner`.

    Exports the netlist in Xyce dialect, prints requested vectors, and applies setting
    tolerances at jolty's thermal-voltage temperature. Records Xyce's reported analysis
    time per repeat and step counts from the final repeat.

    Raises:
        Unsupported: When Xyce cannot run the analysis, or write the circuit.
        JoltyError: When Xyce fails, or runs out of time.
    """
    fixture = prepared.fixture
    analysis = fixture.analysis
    if isinstance(analysis, PoleZero | PeriodicSteadyState):
        message = (
            "Xyce 7.10 has no pole-zero analysis, and finds a periodic steady state "
            "by harmonic balance alone, which the suite does not run"
        )
        raise Unsupported(message)
    built = circuit(fixture.top)
    try:
        lines = netlist(built, (analysis,), dialect=export.XYCE)
    except JoltyError as error:
        raise Unsupported(str(error)) from error
    vectors = {
        read: export.vector(built, prepared.unknowns[read], dialect=export.XYCE)
        for read in fixture.reads
    }
    statement = f".{command(analysis, export.flatten(built))}"
    printed = None if isinstance(analysis, Noise) else list(vectors.values())
    found = [
        runner.run(lines, statement, printed=printed, options=setting.xyce_options())
        for _ in range(repeats)
    ]
    last = found[-1]
    counts = {
        name: int(stated[-1])
        for name, pattern in XYCE_STEPS.items()
        if isinstance(analysis, Transient)
        for stated in [pattern.findall(last.printed)]
        if stated
    }
    return Measured(
        _xyce_found(last, analysis, vectors),
        [each.analysis_seconds for each in found],
        counts,
    )


def _xyce_found(
    found: xyce.Result, analysis: object, vectors: Mapping[str, str]
) -> Found:
    """Extracts fixture reads and sample coordinates from a Xyce result.

    Reconstructs DC sweep coordinates from the plan and reads Xyce's ``onoise`` column
    directly; delegates other analyses to `_spice_found`.
    """
    match analysis:
        case plan.DcSweep():
            points = np.array(analysis.numbers(VOLTAGE))
            return Found(
                points, {read: found.real(name) for read, name in vectors.items()}
            )
        case Noise():
            return Found(found.real("frequency"), {"noise": found.real("onoise")})
    return _spice_found(found, analysis, vectors)


def _spice_found(
    raw: ngspice.Result | xyce.Result, analysis: object, vectors: Mapping[str, str]
) -> Found:
    """Extracts fixture reads and sample coordinates from an ngspice or Xyce result."""
    match analysis:
        case Transient():
            points = raw.real("time")
        case plan.DcSweep():
            points = next(iter(raw.vectors.values())).real
        case Ac():
            points = raw.real("frequency")
            return Found(
                points, {read: raw.vectors[name] for read, name in vectors.items()}
            )
        case Noise():
            return Found(raw.real("frequency"), {"noise": raw.real("onoise_spectrum")})
        case _:
            poles = [
                value
                for name, values in raw.vectors.items()
                if name.startswith("v(pole(")
                for value in values
            ]
            return Found(np.empty(0), {"poles": np.array(poles)})
    return Found(points, {read: raw.real(name) for read, name in vectors.items()})


class Scored(BaseModel):
    """Accuracy score holding achieved digits, reference ceiling, and worst point."""

    digits: float
    limit: float
    at: float


class FixtureRun(BaseModel):
    """Scored outcome of one fixture in one simulator at one tolerance setting."""

    simulator: Simulator
    sweep: str
    relative_tolerance: float
    load: float
    """Host one-minute load average recorded when the run started."""
    status: Literal["scored", "failed", "unsupported"]
    message: str = ""
    seconds: float = 0.0
    """Median engine runtime across repeats in seconds."""
    repeats: list[float] = Field(default_factory=list)
    points: int = 0
    counts: dict[str, int] = Field(default_factory=dict)
    scores: dict[str, dict[str, Scored]] = Field(default_factory=dict)

    def scored(self, output: ScoredOutput) -> Scored | None:
        """Returns the plotted `Scored` entry for `output`, or None when absent."""
        return self.scores.get(output.name, {}).get(output.score)


class FixtureResults(BaseModel):
    """Aggregated runs, metadata, and exact-reference description for one fixture."""

    describes: str
    reference: str
    outputs: list[ScoredOutput]
    seconds: float
    """Total wall time in seconds spent running, computing references, and scoring."""
    runs: list[FixtureRun]


class SuiteResults(Provenance):
    """Root precision results schema holding provenance, versions, and fixtures."""

    versions: dict[str, str]
    repeats: int
    fixtures: dict[str, FixtureResults] = Field(default_factory=dict)


def score(compared: Compared, output: ScoredOutput) -> dict[str, Scored]:
    """Computes relative and mixed significant digits and ceilings for `compared`.

    Computes ``-log₁₀`` of the worst-point error divided by ``|exact|`` (adding
    `output.scale` for the mixed score, omitted when `output.scale` is zero). Caps
    reported digits at the tighter of the exact reference rounding bound and half
    `EPSILON`, and omits any score whose denominator reaches zero.

    Raises:
        ValueError: When simulated values differ in shape or contain non-finite entries.
    """
    simulated = np.asarray(compared.simulated)
    expected = np.asarray(compared.exact)
    if simulated.shape != expected.shape or not np.all(np.isfinite(simulated)):
        message = (
            f"the run found {simulated.shape} values, not all finite, where the "
            f"comparison has {expected.shape}"
        )
        raise ValueError(message)
    error = np.abs(simulated - expected)
    scored: dict[str, Scored] = {}
    floors = {"relative": 0.0} | ({"mixed": output.scale} if output.scale else {})
    for kind, floor in floors.items():
        scale = np.abs(expected) + floor
        if np.any(scale == 0):
            continue
        ratios = error / scale
        worst = int(np.argmax(ratios))
        rounding = max(float(np.max(compared.bound / scale)), EPSILON / 2)
        limit = -math.log10(rounding)
        found = float(ratios.flat[worst])
        scored[kind] = Scored(
            digits=limit if found == 0 else min(-math.log10(found), limit),
            limit=limit,
            at=float(np.asarray(compared.points).flat[worst]),
        )
    return scored


def measure(  # noqa: PLR0913, PLR0917 - the fixtures, how they run, and their files
    names: Sequence[str],
    simulators: Sequence[Simulator],
    sweeps: Sequence[Sweep],
    repeats: int,
    output: Path,
    plot: Path | None,
    digits: float | None,
) -> SuiteResults:
    """Runs the fixtures `names`, writes their results to `output`, and draws them.

    Every fixture runs without names, in the `SIMULATORS` installed without
    simulators named; given a results file to `plot`, it draws that file's diagrams
    and tables instead, its table at equal error at `digits`. Returns the results
    drawn.

    Raises:
        SystemExit: When a fixture named does not exist, or a simulator named does not
            run the suite.
    """
    if plot is not None:
        read = SuiteResults.model_validate(load(plot))
        draw(read, plot.with_suffix(""), digits)
        return read
    every = fixtures()
    unknown = [name for name in names if name not in every]
    if unknown:
        message = (
            f"no fixture is named {', '.join(unknown)}; there are {', '.join(every)}"
        )
        raise SystemExit(message)
    strange = [simulator for simulator in simulators if simulator not in SIMULATORS]
    if strange:
        message = (
            f"the precision suite runs no {', '.join(strange)}; "
            f"it runs {', '.join(SIMULATORS)}"
        )
        raise SystemExit(message)
    installed = [each for each in SIMULATORS if runner(each).version() is not None]
    document = SuiteResults(
        **provenance().model_dump(), versions=_versions(), repeats=repeats
    )
    if output.is_file():
        document.fixtures = SuiteResults.model_validate(load(output)).fixtures
    started = time.perf_counter()
    for name in names or list(every):
        document.fixtures[name] = run(
            Prepared.from_fixture(every[name]), simulators or installed, sweeps, repeats
        )
        save(output, document)
    print(f"\nmeasured in {duration(time.perf_counter() - started)}")
    draw(document, output.with_suffix(""), digits)
    return document


def _versions() -> dict[str, str]:
    """Collects installed version strings for mpmath, jolty, ngspice, and Xyce.

    Includes external simulators only when installed locally.
    """
    others = {"ngspice": Ngspice().version(), "xyce": Xyce().version()}
    return {"mpmath": metadata.version("mpmath"), "jolty": Jolty().version()} | {
        name: stated for name, stated in others.items() if stated is not None
    }


def run(
    prepared: Prepared,
    simulators: Sequence[Simulator],
    sweeps: Sequence[Sweep],
    repeats: int,
) -> FixtureResults:
    """Runs and scores every setting of `prepared` across `simulators`.

    Interleaves simulators per setting so all configurations share comparable host load.
    """
    fixture = prepared.fixture
    print(f"\n{fixture.name}: {fixture.describes}", flush=True)
    started = time.perf_counter()
    runs: list[FixtureRun] = []
    for setting in settings(sweeps):
        cells = []
        for simulator in simulators:
            ran = _run(prepared, setting, repeats, simulator)
            runs.append(ran)
            cells.append(_cell(fixture, ran))
        tolerance = f"{setting.sweep:>8} {setting.relative_tolerance:<8.3g} "
        print(tolerance + " | ".join(cells), flush=True)
    return FixtureResults(
        describes=fixture.describes,
        reference=fixture.reference,
        outputs=list(fixture.outputs),
        seconds=time.perf_counter() - started,
        runs=runs,
    )


def _run(
    prepared: Prepared, setting: Setting, repeats: int, simulator: Simulator
) -> FixtureRun:
    """Executes and scores one fixture run in `simulator` at `setting`."""
    fixture = prepared.fixture
    begun = FixtureRun(
        simulator=simulator,
        sweep=setting.sweep,
        relative_tolerance=setting.relative_tolerance,
        load=load_average(),
        status="scored",
    )
    try:
        match simulator:
            case Simulator.NGSPICE_SPARSE | Simulator.NGSPICE_KLU:
                measured = run_ngspice(prepared, setting, repeats, Ngspice(simulator))
            case Simulator.XYCE_SERIAL:
                measured = run_xyce(prepared, setting, repeats, Xyce())
            case _:
                measured = run_jolty(prepared, setting, repeats)
    except Unsupported as reason:
        return begun.model_copy(
            update={"status": "unsupported", "message": str(reason)}
        )
    except JoltyError as error:
        message = str(error)
        said = stopped(message)
        return begun.model_copy(update={"status": "failed", "message": said})
    try:
        compared = fixture.compare(measured.found)
        scores = {
            output.name: score(compared[output.name], output)
            for output in fixture.outputs
        }
    except ValueError as error:
        return begun.model_copy(update={"status": "failed", "message": str(error)})
    return begun.model_copy(
        update={
            "seconds": statistics.median(measured.seconds),
            "repeats": measured.seconds,
            "points": len(measured.found.points),
            "counts": measured.counts,
            "scores": scores,
        }
    )


def _cell(fixture: Fixture, ran: FixtureRun) -> str:
    """Formats scored digits and duration of a fixture run for console progress."""
    if ran.status != "scored":
        return f"{ran.simulator} {ran.status}"
    digits = []
    for output in fixture.outputs:
        found = ran.scored(output)
        digits.append("-" if found is None else f"{found.digits:.1f}")
    return f"{ran.simulator} {'/'.join(digits)} in {duration(ran.seconds)}"


COLORS: Final = {
    Simulator.JOLTY: plots.COLORS[0],
    Simulator.NGSPICE_SPARSE: plots.COLORS[1],
    Simulator.NGSPICE_KLU: plots.COLORS[2],
    Simulator.XYCE_SERIAL: plots.COLORS[3],
}
"""Each simulator's color, the plots' first four, in their fixed order."""

STYLES: Final = {Sweep.RELTOL: "-", Sweep.SCALED: "--"}
"""Each sweep's line: solid for the relative tolerance alone, dashed for all scaled."""


def draw(document: SuiteResults, directory: Path, digits: float | None) -> None:
    """Draws each fixture's diagram into `directory`, and writes its tables there.

    The table at defaults, which is printed too, and the table at equal error, at
    `digits`.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for name, measured in document.fixtures.items():
        diagram(name, measured, document).savefig(directory / f"{name}.png")
    table = defaults_table(document)
    (directory / "defaults.md").write_text(table + "\n", encoding="utf-8")
    equal = equal_error_table(document, digits)
    (directory / "equal-error.md").write_text(equal + "\n", encoding="utf-8")
    print(f"\n{table}")
    print(f"drew {len(document.fixtures)} diagrams into {directory}")


def diagram(name: str, measured: FixtureResults, document: SuiteResults) -> Figure:
    """Renders a work-precision plot of correct digits against median engine time.

    Creates one panel per output on a logarithmic time axis, plotting each simulator and
    sweep in tolerance order. Rings default settings and marks the reference rounding
    ceiling with a dotted line.
    """
    figure = Figure(
        figsize=(4.2 * len(measured.outputs) + 1.8, 4.2),
        dpi=150,
        layout="constrained",
        facecolor=plots.SURFACE,
    )
    machine = document.machine
    cpu = machine.get("cpu", "")
    loads = [each.load for each in measured.runs]
    figure.suptitle(
        f"{name}: correct digits against engine time\n{cpu}, each time the median "
        f"of {document.repeats} runs,\nat a load average of {min(loads):.1f} to "
        f"{max(loads):.1f}; commit {document.commit[:7]}",
        color=plots.INK,
        x=0.02,
        ha="left",
        fontsize=9,
    )
    panels = figure.subplots(1, len(measured.outputs), squeeze=False)[0]
    for axes, output in zip(panels, measured.outputs, strict=True):
        _panel(axes, output, measured.runs)
    panels[-1].legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7
    )
    return figure


def _panel(axes: Axes, output: ScoredOutput, runs: Sequence[FixtureRun]) -> None:
    """Draws `output`'s digits against time on `axes`, a line per simulator."""
    limits: list[float] = []
    for simulator in SIMULATORS:
        for sweep in Sweep:
            chosen = [
                (each.seconds, found)
                for each in runs
                if each.simulator == simulator and each.sweep == sweep
                for found in [each.scored(output)]
                if found is not None
            ]
            if not chosen:
                continue
            limits += [found.limit for _, found in chosen]
            failed = sum(
                1
                for each in runs
                if each.simulator == simulator
                and each.sweep == sweep
                and each.status == "failed"
            )
            label = f"{simulator}, {sweep}" + (f" ({failed} failed)" if failed else "")
            axes.plot(
                [seconds for seconds, _ in chosen],
                [found.digits for _, found in chosen],
                STYLES[sweep],
                color=COLORS[simulator],
                linewidth=1.2,
                marker="o",
                markersize=2.5,
                label=label,
            )
        for each in runs:
            found = each.scored(output)
            if each.simulator == simulator and each.sweep == "defaults" and found:
                axes.plot(
                    [each.seconds],
                    [found.digits],
                    marker="o",
                    markersize=9,
                    markerfacecolor="none",
                    markeredgewidth=1.6,
                    color=COLORS[simulator],
                    linestyle="none",
                )
    if limits:
        axes.axhline(
            min(limits),
            color=plots.MUTED,
            linewidth=0.8,
            linestyle=":",
            label="the exact answer's own digits",
        )
    axes.plot(
        [],
        [],
        marker="o",
        markersize=9,
        markerfacecolor="none",
        markeredgewidth=1.6,
        color=plots.MUTED,
        linestyle="none",
        label="defaults",
    )
    axes.set_xscale("log")
    axes.xaxis.set_major_formatter(EngFormatter(unit="s", sep=" "))
    axes.xaxis.set_minor_formatter(FuncFormatter(_minor))
    axes.tick_params(axis="x", which="minor", labelsize=6, labelcolor=plots.MUTED)
    axes.set_xlabel("engine time", color=plots.MUTED)
    axes.set_ylabel(
        f"correct digits of {output.name}, {output.score}", color=plots.MUTED
    )
    axes.set_facecolor(plots.SURFACE)
    axes.grid(visible=True, color=plots.GRID, linewidth=0.6)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)


LABELED: Final = frozenset({2, 5})
"""The leading digits of the minor ticks a log axis labels, so that labels never
crowd: 2 and 5 times each power of ten."""


def _minor(value: float, _: int | None) -> str:
    """Formats minor logarithmic time-axis tick labels at 2 and 5 times each decade."""
    leading = round(value / 10 ** math.floor(math.log10(value)))
    return EngFormatter(unit="s", sep=" ")(value) if leading in LABELED else ""


def simulators_in(document: SuiteResults) -> list[Simulator]:
    """Returns simulators present in `document`, preserving `SIMULATORS` order."""
    found = {
        each.simulator
        for fixture in document.fixtures.values()
        for each in fixture.runs
    }
    return [simulator for simulator in SIMULATORS if simulator in found]


def defaults_table(document: SuiteResults) -> str:
    """Formats a Markdown table of default-setting digits and times per output."""
    simulators = simulators_in(document)
    heads = ["fixture", "output", *(str(simulator) for simulator in simulators)]
    rows = [
        [
            f"`{name}`",
            f"{output.name}, {output.score}",
            *(_default(measured, output, each) for each in simulators),
        ]
        for name, measured in document.fixtures.items()
        for output in measured.outputs
    ]
    return markdown_table(heads, rows)


def _default(
    measured: FixtureResults, output: ScoredOutput, simulator: Simulator
) -> str:
    """Formats one simulator's default-setting digits and runtime as a table cell."""
    found = next(
        (
            each
            for each in measured.runs
            if each.simulator == simulator and each.sweep == "defaults"
        ),
        None,
    )
    if found is None:
        return ""
    if found.status != "scored":
        return found.status
    scored = found.scored(output)
    digits = "" if scored is None else f"{scored.digits:.1f} digits, "
    return f"{digits}{duration(found.seconds)}"


def equal_error_table(document: SuiteResults, digits: float | None) -> str:
    """Formats a Markdown table of minimum engine time at equal accuracy per output.

    Uses `digits` as the accuracy target, or the highest digit count achieved by every
    simulator running the fixture. Reports each simulator's fastest qualifying run
    alongside its setting and achieved digits. Annotates simulators that missed the
    target with their maximum digits and notes any failed runs.
    """
    simulators = simulators_in(document)
    heads = ["fixture", "output", "target", *(str(each) for each in simulators)]
    rows: list[list[str]] = []
    for name, measured in document.fixtures.items():
        for output in measured.outputs:
            scored = {
                simulator: [
                    (each, found)
                    for each in measured.runs
                    if each.simulator == simulator
                    for found in [each.scored(output)]
                    if found is not None
                ]
                for simulator in simulators
            }
            reached = [
                max(f.digits for _, f in runs) for runs in scored.values() if runs
            ]
            target = digits if digits is not None else min(reached, default=math.nan)
            cells = [f"`{name}`", f"{output.name}, {output.score}", f"{target:.1f}"]
            cells.extend(
                _equal_error(measured.runs, simulator, scored[simulator], target)
                for simulator in simulators
            )
            rows.append(cells)
    return markdown_table(heads, rows)


def _equal_error(
    runs: Sequence[FixtureRun],
    simulator: Simulator,
    scored: Sequence[tuple[FixtureRun, Scored]],
    target: float,
) -> str:
    """Formats a simulator's fastest run reaching `target` digits as a table cell."""
    own = [each for each in runs if each.simulator == simulator]
    failed = [each for each in own if each.status == "failed"]
    if not own or all(each.status == "unsupported" for each in own):
        return "unsupported" if own else ""
    if not scored:
        return f"failed: {failed[0].message}" if failed else ""
    noted = f"; {len(failed)} failed" if failed else ""
    enough = [(each, found) for each, found in scored if found.digits >= target]
    if not enough:
        most = max(found.digits for _, found in scored)
        return f"never: {most:.1f} digits at most{noted}"
    each, found = min(enough, key=lambda pair: pair[0].seconds)
    setting = (
        each.sweep
        if each.sweep == "defaults"
        else f"{each.sweep} {each.relative_tolerance:.3g}"
    )
    return f"{duration(each.seconds)}, {setting}, {found.digits:.1f} digits{noted}"
