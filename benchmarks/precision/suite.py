# Copyright 2026 Maxence Faldor
"""The precision suite: each simulator's error against exact answers, against its cost.

``scripts/bench --precision`` runs it through ``benchmarks/speed/bench.py``. Each
fixture of ``fixtures.py`` runs in each simulator, jolty and ngspice 47 with Sparse 1.3
and with KLU, each by its runner of ``benchmarks/simulators.py``, at its defaults and
along two sweeps: the relative tolerance alone,
from 1e-2 to 1e-8 in quarter decades, the other tolerances at their defaults, and every
tolerance scaled together from the defaults by the same factor. At each point a run
produced, the exact answer there scores it: the significant correct digits of its
worst point, ``-log₁₀ max |found - exact| / |exact|``, and their mixed form, with the
quantity's scale added to ``|exact|``, for a quantity that passes through or near
zero; no score claims more digits than the exact answer's own rounding allows. Beside
each score, the run's engine time, the median of its repeats, and its steps,
rejections and Newton iterations.

Usage::

    scripts/bench --precision [FIXTURE ...]          every fixture, or those named
    scripts/bench --precision --simulator jolty      one simulator, or several
    scripts/bench --precision --sweeps reltol        one sweep, or both
    scripts/bench --precision --plot RESULTS         the diagrams and table of a file

The results go to ``benchmarks/precision/results/<commit>.json``, ``-dirty`` for a
changed tree, a run adding its fixtures to what an earlier one on the same commit
wrote; the diagrams, one per fixture, and the table at defaults go to the directory of
the same name. ``benchmarks/precision/README.md`` says what each fixture measures, and
what the first diagrams found.
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
    provenance,
    save,
)
from pydantic import BaseModel, Field
from simulators import (
    RUSAGE,
    Jolty,
    Ngspice,
    Simulator,
    Unsupported,
    command,
    netlist,
    trouble,
)

from jolty import circuit, foundry, plan, plots, results
from jolty.errors import JoltyError
from jolty.plan import Ac, Noise, Options, PoleZero, Transient
from jolty.simulate import Elaborated, Simulation
from jolty.spice import export, ngspice

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

DEFAULTS: Final = {name: plan.DEFAULTS[name] for name in TOLERANCES}
"""The tolerances' defaults, the engine's, SPICE's own, which ngspice's are too."""

SPICE_NAMES: Final = {name: foundry.SPICE_OPTIONS[name] for name in TOLERANCES}
"""ngspice's name of each tolerance."""

SIMULATORS: Final = (
    Simulator.JOLTY,
    Simulator.NGSPICE_SPARSE,
    Simulator.NGSPICE_KLU,
)
"""The simulators that run the precision suite, as the research's design names
them."""


class Sweep(StrEnum):
    """How a sweep sets the tolerances: the relative alone, or all of them scaled."""

    RELTOL = "reltol"
    SCALED = "scaled"


@dataclass(frozen=True, slots=True)
class Setting:
    """The tolerances of one run: the defaults, or a sweep's at a relative tolerance."""

    sweep: str
    relative_tolerance: float

    def options(self) -> Options:
        """The options jolty runs with, the others at their defaults."""
        given = self.tolerances()
        return Options(
            relative_tolerance=given.get("relative_tolerance"),
            voltage_tolerance=given.get("voltage_tolerance"),
            current_tolerance=given.get("current_tolerance"),
            charge_tolerance=given.get("charge_tolerance"),
        )

    def tolerances(self) -> dict[str, float]:
        """The tolerances the run sets, by jolty's option names; none at defaults."""
        if self.sweep == "defaults":
            return {}
        if self.sweep == Sweep.RELTOL:
            return {"relative_tolerance": self.relative_tolerance}
        factor = self.relative_tolerance / DEFAULTS["relative_tolerance"]
        return {name: value * factor for name, value in DEFAULTS.items()}


@dataclass(frozen=True, slots=True)
class Measured:
    """What one run found, how long each of its repeats took, and what it counted."""

    found: Found
    seconds: list[float]
    counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class Prepared:
    """A fixture with its circuit elaborated once, and its unknowns by name."""

    fixture: Fixture
    elaborated: Elaborated
    unknowns: Mapping[str, results.Unknown]

    @classmethod
    def of(cls, fixture: Fixture) -> Prepared:
        """`fixture`, its circuit elaborated."""
        elaborated = Elaborated(fixture.top)
        unknowns = Simulation.of(elaborated).unknowns
        return cls(fixture, elaborated, {unknown.name: unknown for unknown in unknowns})


def settings(sweeps: Iterable[Sweep]) -> list[Setting]:
    """The defaults, then each sweep's relative tolerances in turn."""
    swept = [
        Setting(sweep, tolerance)
        for sweep in sweeps
        for tolerance in RELATIVE_TOLERANCES
    ]
    return [Setting("defaults", DEFAULTS["relative_tolerance"]), *swept]


def run_jolty(prepared: Prepared, setting: Setting, repeats: int) -> Measured:
    """The fixture's analysis in jolty, `repeats` times, each on a fresh simulation.

    Each simulation of the circuit elaborated once starts with nothing remembered, so
    every repeat solves its operating point again; the time is that of
    ``Simulation.run``, the engine's work and its results made in Python, after one
    run to warm up.
    """
    options = setting.options()
    seconds: list[float] = []
    # One run to warm up first, as the speed suite's timing does, its time left out.
    Simulation.of(prepared.elaborated).run(prepared.fixture.analysis, options)
    result: results.AnalysisResult | None = None
    for _ in range(repeats):
        simulation = Simulation.of(prepared.elaborated)
        started = time.perf_counter()
        result = simulation.run(prepared.fixture.analysis, options)
        seconds.append(time.perf_counter() - started)
    if result is None:
        message = "a run repeats once at least"
        raise ValueError(message)
    return Measured(_found(result, prepared.fixture.reads), seconds, _counted(result))


def _found(result: results.AnalysisResult, reads: Sequence[str]) -> Found:
    """What a jolty result holds of the unknowns `reads`, at its points."""
    match result:
        case (
            results.TransientResult(times=points)
            | results.DcSweepResult(levels=points)
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
    """A transient's steps, rejections and Newton iterations.

    The other analyses count none yet.
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
    """The fixture's analysis in ngspice, by `runner`, `repeats` times in one process.

    The deck is the circuit's netlist as ``jolty.spice.export`` writes it, saving the
    vectors the fixture reads, with the setting's tolerances and the runner's solver;
    ngspice runs at the temperature where its thermal voltage is jolty's, as
    ``jolty.spice.ngspice`` runs it. Each repeat's time is the growth of ngspice's own
    analysis time; the counts are the last repeat's.

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
    analysed, after = (
        (f"set sqrnoise\n{found}", ["setplot previous"])
        if isinstance(analysis, Noise)
        else (found, [])
    )
    commands = "\n".join([*[f"{analysed}\nrusage all"] * repeats, *after])
    given = [
        f"{SPICE_NAMES[name]}={value!r}" for name, value in setting.tolerances().items()
    ]
    options = f".options {' '.join(given)}" if given else "*"
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


def _spice_found(
    raw: ngspice.Raw, analysis: object, vectors: Mapping[str, str]
) -> Found:
    """What ngspice's plot holds of the vectors a fixture reads, at its points."""
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
    """A score: the correct digits it claims, and those its exact answer allows.

    Beside them, the point where its error was worst.
    """

    digits: float
    limit: float
    at: float


class FixtureRun(BaseModel):
    """One run of a fixture in one simulator at one setting, scored."""

    simulator: Simulator
    sweep: str
    relative_tolerance: float
    load: float
    """The load average over the last minute as the run began."""
    status: Literal["scored", "failed", "unsupported"]
    message: str = ""
    seconds: float = 0.0
    """The median of its repeats' engine times."""
    repeats: list[float] = Field(default_factory=list)
    points: int = 0
    counts: dict[str, int] = Field(default_factory=dict)
    scores: dict[str, dict[str, Scored]] = Field(default_factory=dict)

    def scored(self, output: ScoredOutput) -> Scored | None:
        """The score its diagram draws of `output`, if the run has one."""
        return self.scores.get(output.name, {}).get(output.score)


class FixtureResults(BaseModel):
    """A fixture's runs, with what it is and how its exact answer is computed."""

    describes: str
    reference: str
    outputs: list[ScoredOutput]
    seconds: float
    """How long its runs, its exact answer and its scoring took."""
    runs: list[FixtureRun]


class SuiteResults(Provenance):
    """A results file: its provenance, the versions, and each fixture's results."""

    versions: dict[str, str]
    repeats: int
    fixtures: dict[str, FixtureResults] = Field(default_factory=dict)


def score(compared: Compared, output: ScoredOutput) -> dict[str, Scored]:
    """The correct digits of the worst point, relative and mixed, and their limits.

    Each score is ``-log₁₀`` of the largest ratio of a point's error to its exact
    value's magnitude, plus the output's scale for the mixed score, which a scale of
    zero leaves out; its limit is the same of the exact value's bound, or of half the
    double's epsilon, its rounding, where that is larger; the digits it claims are
    the fewer of the two. A score is left out where an exact value is zero.

    Raises:
        ValueError: When the simulator found other points, or values not finite.
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
) -> None:
    """Runs the fixtures `names`, writes their results to `output`, and draws them.

    Every fixture runs without names, in the `SIMULATORS` without simulators named;
    given a results file to `plot`, it draws that file's diagrams and table instead.

    Raises:
        SystemExit: When a fixture named does not exist, or a simulator named does not
            run the suite.
    """
    if plot is not None:
        draw(SuiteResults.model_validate(load(plot)), plot.with_suffix(""))
        return
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
    document = SuiteResults(
        **provenance().model_dump(), versions=_versions(), repeats=repeats
    )
    if output.is_file():
        document.fixtures = SuiteResults.model_validate(load(output)).fixtures
    started = time.perf_counter()
    for name in names or list(every):
        document.fixtures[name] = run(
            Prepared.of(every[name]), simulators or SIMULATORS, sweeps, repeats
        )
        save(output, document)
    print(f"\nmeasured in {duration(time.perf_counter() - started)}")
    draw(document, output.with_suffix(""))


def _versions() -> dict[str, str]:
    """The versions of jolty, of ngspice, and of the exact answers' arithmetic."""
    found = {"mpmath": metadata.version("mpmath"), "jolty": Jolty().version()}
    stated = Ngspice().version()
    if stated is not None:
        found["ngspice"] = stated
    return found


def run(
    prepared: Prepared,
    simulators: Sequence[Simulator],
    sweeps: Sequence[Sweep],
    repeats: int,
) -> FixtureResults:
    """Every run of a fixture, each setting in each simulator in turn, scored.

    The simulators take each setting one after another, so that each runs under the
    load the others do.
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
    """One run of a fixture in `simulator` at `setting`, scored."""
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
            case _:
                measured = run_jolty(prepared, setting, repeats)
    except Unsupported as reason:
        return begun.model_copy(
            update={"status": "unsupported", "message": str(reason)}
        )
    except JoltyError as error:
        message = str(error)
        said = trouble(message) or message.splitlines()[0]
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
    """A run's digits and time, as the progress prints them."""
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
}
"""Each simulator's color, the plots' first three, in their fixed order."""

STYLES: Final = {Sweep.RELTOL: "-", Sweep.SCALED: "--"}
"""Each sweep's line: solid for the relative tolerance alone, dashed for all scaled."""


def draw(document: SuiteResults, directory: Path) -> None:
    """Draws each fixture's diagram into `directory`, and writes the table at defaults.

    The table is printed too.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for name, measured in document.fixtures.items():
        diagram(name, measured, document).savefig(directory / f"{name}.png")
    table = defaults_table(document)
    (directory / "defaults.md").write_text(table + "\n", encoding="utf-8")
    print(f"\n{table}")
    print(f"drew {len(document.fixtures)} diagrams into {directory}")


def diagram(name: str, measured: FixtureResults, document: SuiteResults) -> Figure:
    """A fixture's work-precision diagram: correct digits against engine time.

    A panel per output draws its score's digits against the median engine time, on
    a log axis: a line per simulator configuration and sweep, in the order of their
    tolerances, the defaults ringed, and the digits the exact answer's rounding
    allows dotted.
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
    """A minor tick's label: its time with an SI prefix, at 2 and 5 times a decade."""
    leading = round(value / 10 ** math.floor(math.log10(value)))
    return EngFormatter(unit="s", sep=" ")(value) if leading in LABELED else ""


def defaults_table(document: SuiteResults) -> str:
    """Each simulator's digits and time at its defaults, by fixture and output."""
    heads = ["fixture", "output", *(str(simulator) for simulator in SIMULATORS)]
    rows = [
        "| " + " | ".join(heads) + " |",
        "| " + " | ".join("---" for _ in heads) + " |",
    ]
    for name, measured in document.fixtures.items():
        for output in measured.outputs:
            cells = [f"`{name}`", f"{output.name}, {output.score}"]
            cells += [_default(measured, output, each) for each in SIMULATORS]
            rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _default(
    measured: FixtureResults, output: ScoredOutput, simulator: Simulator
) -> str:
    """A simulator's digits and time at its defaults, as the table writes them."""
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
