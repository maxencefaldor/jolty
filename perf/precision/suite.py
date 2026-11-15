# Copyright 2026 Maxence Faldor
"""The precision suite: each simulator's error against exact answers, against its cost.

``scripts/precision`` runs this script from the repository root. Each fixture of
``fixtures.py`` runs in each simulator configuration, jolty and ngspice 47 with Sparse
1.3 and with KLU, at its defaults and along two sweeps: the relative tolerance alone,
from 1e-2 to 1e-8 in quarter decades, the other tolerances at their defaults, and every
tolerance scaled together from the defaults by the same factor. At each point a run
produced, the exact answer there scores it: the significant correct digits of its
worst point, ``-log₁₀ max |found - exact| / |exact|``, and their mixed form, with the
quantity's scale added to ``|exact|``, for a quantity that passes through or near
zero; no score claims more digits than the exact answer's own rounding allows. Beside
each score, the run's engine time, the median of its repeats, and its steps,
rejections and Newton iterations.

Usage::

    scripts/precision [FIXTURE ...]           every fixture, or those named
    scripts/precision --simulators jolty      one simulator configuration, or several
    scripts/precision --sweeps reltol         one sweep, or both
    scripts/precision --plot RESULTS          the diagrams and table of a results file

The results go to ``perf/results/precision/<commit>.json``, ``-dirty`` for a changed
tree, a run adding its fixtures to what an earlier one on the same commit wrote; the
diagrams, one per fixture, and the table at defaults go to the directory of the same
name. ``perf/precision/README.md`` says what each fixture measures, and what the first
diagrams found.
"""

import json
import math
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib import metadata
from pathlib import Path
from typing import Final, Literal

import numpy as np
from exact import EPSILON
from fixtures import Compared, Fixture, Found, Quantity, fixtures
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter, FuncFormatter
from pydantic import BaseModel, Field, JsonValue
from pydantic_settings import CliPositionalArg, SettingsConfigDict

from jolty import circuit, cli, ir, plots, results
from jolty.errors import JoltyError
from jolty.plan import Ac, DcSweep, Noise, Options, PoleZero, Transient
from jolty.simulate import Elaborated, Simulation
from jolty.spice import export, ngspice
from jolty.units import FREQUENCY, TIME, VOLTAGE

PRECISION: Final = Path(__file__).resolve().parent
"""The ``perf/precision`` directory."""

PERF: Final = PRECISION.parent
"""The ``perf`` directory."""

sys.path.insert(0, str(PERF))
import bench  # noqa: E402 - the speed suite, beside this one, whose machine and commit it reads

RESULTS: Final = PERF / "results" / "precision"
"""Where the results go."""

RELATIVE_TOLERANCES: Final = tuple(10 ** (-2 - step / 4) for step in range(25))
"""The relative tolerances each sweep runs: 1e-2 to 1e-8 in quarter decades, as the
IVP test set's diagrams sweep theirs (Lioen and de Swart, 1998)."""

DEFAULTS: Final = {
    "relative_tolerance": 1e-3,
    "voltage_tolerance": 1e-6,
    "current_tolerance": 1e-12,
    "charge_tolerance": 1e-14,
}
"""The tolerances' defaults, SPICE's own, jolty's and ngspice's alike: the guide's
table of options, and ngspice's manual on ``.options``."""

SPICE_NAMES: Final = {
    "relative_tolerance": "reltol",
    "voltage_tolerance": "vntol",
    "current_tolerance": "abstol",
    "charge_tolerance": "chgtol",
}
"""ngspice's name of each tolerance."""

STATISTICS: Final = {
    "seconds": re.compile(r"Total analysis time \(seconds\) = (\S+)"),
    "iterations": re.compile(r"Total iterations = (\d+)"),
    "transient iterations": re.compile(r"Transient iterations = (\d+)"),
    "accepted": re.compile(r"Accepted timepoints = (\d+)"),
    "rejected": re.compile(r"Rejected timepoints = (\d+)"),
}
"""What ngspice's ``rusage all`` prints, summed over the analyses it has run: its
analysis time, which its clock measures as wall time, its Newton iterations, those of
its transients, and the time points it accepted and rejected."""

SOLVERS: Final = {
    "ngspice-sparse": "Using SPARSE 1.3 as Direct Linear Solver",
    "ngspice-klu": "Using KLU as Direct Linear Solver",
}
"""What ngspice prints of the solver each of its configurations runs."""


class Simulator(StrEnum):
    """A simulator configuration, named as the research's design names them."""

    JOLTY = "jolty"
    NGSPICE_SPARSE = "ngspice-sparse"
    NGSPICE_KLU = "ngspice-klu"


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
class Ran:
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


class Unsupported(Exception):  # noqa: N818 - a reason, not a failure
    """A simulator configuration cannot run a fixture's analysis, for a reason."""


def settings(sweeps: Iterable[Sweep]) -> list[Setting]:
    """The defaults, then each sweep's relative tolerances in turn."""
    swept = [
        Setting(sweep, tolerance)
        for sweep in sweeps
        for tolerance in RELATIVE_TOLERANCES
    ]
    return [Setting("defaults", DEFAULTS["relative_tolerance"]), *swept]


def jolty(prepared: Prepared, setting: Setting, repeats: int) -> Ran:
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
    result: results.Result | None = None
    for _ in range(repeats):
        simulation = Simulation.of(prepared.elaborated)
        started = time.perf_counter()
        result = simulation.run(prepared.fixture.analysis, options)
        seconds.append(time.perf_counter() - started)
    if result is None:
        message = "a run repeats once at least"
        raise ValueError(message)
    return Ran(_found(result, prepared.fixture.reads), seconds, _counted(result))


def _found(result: results.Result, reads: Sequence[str]) -> Found:
    """What a jolty result holds of the unknowns `reads`, at its points."""
    match result:
        case (
            results.Waveforms(times=points)
            | results.Sweep(levels=points)
            | results.Response(frequencies=points)
        ):
            values = {
                read: result.values[:, results.column(result.unknowns, read)]
                for read in reads
            }
            return Found(np.asarray(points), values)
        case results.NoiseSpectrum():
            return Found(result.frequencies, {"noise": result.densities})
        case results.PolesZeros():
            return Found(np.empty(0), {"poles": result.poles})
    message = f"no fixture reads a {type(result).__name__}"
    raise TypeError(message)


def _counted(result: results.Result) -> dict[str, int]:
    """A transient's steps, rejections and Newton iterations.

    The other analyses count none yet.
    """
    if not isinstance(result, results.Waveforms):
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


def spice(
    prepared: Prepared, setting: Setting, repeats: int, simulator: Simulator
) -> Ran:
    """The fixture's analysis in ngspice, `repeats` times in one process.

    The deck is the circuit's netlist as ``jolty.spice.export`` writes it, saving the
    vectors the fixture reads, with the setting's tolerances and, for KLU, ``klu``;
    ngspice runs at the temperature where its thermal voltage is jolty's, as
    ``jolty.spice.ngspice`` runs it. Each repeat's time is the growth of ngspice's own
    analysis time; the counts are the last repeat's.

    Raises:
        Unsupported: When the configuration cannot run the analysis.
        JoltyError: When ngspice fails, aborts the analysis, runs another solver, or
            prints a time for another number of runs.
    """
    fixture = prepared.fixture
    reason = fixture.unsupported.get(simulator)
    if reason is not None:
        raise Unsupported(reason)
    analysis = fixture.analysis
    if simulator is Simulator.NGSPICE_KLU and isinstance(analysis, Noise | PoleZero):
        message = (
            f"ngspice 47 runs no {type(analysis).__name__} with KLU: "
            "'not (yet) supported with option KLU'"
        )
        raise Unsupported(message)
    built = circuit(fixture.top)
    flat = export.flatten(built)
    excited = analysis.source if isinstance(analysis, Ac | Noise) else None
    lines = export.netlist(
        built, excited=excited, transient=isinstance(analysis, Transient)
    )
    vectors = {
        read: export.vector(built, prepared.unknowns[read]) for read in fixture.reads
    }
    if vectors:
        lines.append(f".save {' '.join(vectors.values())}")
    command, after = _command(flat, analysis)
    commands = "\n".join([*[f"{command}\nrusage all"] * repeats, *after])
    given = [
        f"{SPICE_NAMES[name]}={value!r}" for name, value in setting.tolerances().items()
    ]
    if simulator is Simulator.NGSPICE_KLU:
        given.append("klu")
    options = f".options {' '.join(given)}" if given else "*"
    (raw,) = ngspice.runs(lines, (commands,), options=options)
    if raw.printed.count(SOLVERS[simulator]) < repeats:
        message = f"ngspice did not run {simulator}'s solver:\n{raw.printed}"
        raise JoltyError(message)
    printed = {
        name: [float(value) for value in pattern.findall(raw.printed)]
        for name, pattern in STATISTICS.items()
    }
    totals = printed.pop("seconds")
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
    return Ran(_spice_found(raw, analysis, vectors), seconds, counts)


def _command(flat: ir.Circuit, analysis: object) -> tuple[str, list[str]]:
    """The ngspice command that runs `analysis`, and the commands to run after it.

    Raises:
        TypeError: When the analysis is one no fixture runs.
    """
    top = flat.components[flat.top]
    match analysis:
        case Transient(stop=stop, max_step=max_step):
            end = stop.to(TIME, "a stop")
            longest = end / 50 if max_step is None else max_step.to(TIME, "a step")
            return f"tran {longest!r} {end!r} 0 {longest!r}", []
        case DcSweep(source=source, levels=levels):
            given = [level.to(VOLTAGE, "a level") for level in levels]
            step = (given[-1] - given[0]) / (len(given) - 1)
            element = export.element(_instance(top, source))
            return f"dc {element} {given[0]!r} {given[-1]!r} {step!r}", []
        case Ac(frequencies=frequencies) | Noise(frequencies=frequencies):
            first = frequencies[0].to(FREQUENCY, "a frequency")
            last = frequencies[-1].to(FREQUENCY, "a frequency")
            per_decade = round((len(frequencies) - 1) / math.log10(last / first))
            points = f"dec {per_decade} {first!r} {last!r}"
            if isinstance(analysis, Ac):
                return f"ac {points}", []
            source = export.element(_instance(top, analysis.source or ""))
            command = f"noise v({analysis.output}) {source} {points}"
            # The spectrum is the plot before the integrated noise each run ends on.
            return f"set sqrnoise\n{command}", ["setplot previous"]
        case PoleZero(source=str(source), output=str(output)):
            positive = _instance(top, source).terminals["positive"]
            ground = next(
                port.name for port in top.ports if port.role is ir.Role.GROUND
            )
            node = "0" if positive == ground else positive
            return f"pz {node} 0 {output} 0 cur pol", []
    message = f"no fixture runs {type(analysis).__name__} in ngspice"
    raise TypeError(message)


def _instance(top: ir.Component, id: str) -> ir.Instance:
    """The instance `id` of the flattened circuit's `top`.

    Raises:
        KeyError: When it has none.
    """
    found = top.instance(id)
    if found is None:
        message = f"the circuit has no instance {id!r}"
        raise KeyError(message)
    return found


def _spice_found(
    raw: ngspice.Raw, analysis: object, vectors: Mapping[str, str]
) -> Found:
    """What ngspice's plot holds of the vectors a fixture reads, at its points."""
    match analysis:
        case Transient():
            points = raw.real("time")
        case DcSweep():
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


class Run(BaseModel):
    """One run of a fixture in one simulator configuration at one setting, scored."""

    simulator: Simulator
    sweep: str
    relative_tolerance: float
    load: float
    """The load average over the last minute as the run began."""
    status: Literal["ok", "failed", "unsupported"]
    message: str = ""
    seconds: float = 0.0
    """The median of its repeats' engine times."""
    repeats: list[float] = Field(default_factory=list)
    points: int = 0
    counts: dict[str, int] = Field(default_factory=dict)
    scores: dict[str, dict[str, Scored]] = Field(default_factory=dict)

    def scored(self, quantity: Quantity) -> Scored | None:
        """The score its diagram draws of `quantity`, if the run has one."""
        return self.scores.get(quantity.name, {}).get(quantity.score)


class Measured(BaseModel):
    """A fixture's runs, with what it is and how its exact answer is computed."""

    describes: str
    reference: str
    quantities: list[Quantity]
    seconds: float
    """How long its runs, its exact answer and its scoring took."""
    runs: list[Run]


class Document(BaseModel):
    """A results file: the commit, the machine, the versions, and each fixture's."""

    commit: str
    machine: JsonValue
    versions: dict[str, str]
    repeats: int
    fixtures: dict[str, Measured] = Field(default_factory=dict)


def score(compared: Compared, quantity: Quantity) -> dict[str, Scored]:
    """The correct digits of the worst point, relative and mixed, and their limits.

    Each score is ``-log₁₀`` of the largest ratio of a point's error to its exact
    value's magnitude, plus the quantity's scale for the mixed score, which a scale of
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
    floors = {"relative": 0.0} | ({"mixed": quantity.scale} if quantity.scale else {})
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


def measure(arguments: Precision) -> None:
    """Runs the fixtures named, writes their results, and draws their diagrams.

    Raises:
        SystemExit: When a fixture named does not exist.
    """
    if arguments.plot is not None:
        text = arguments.plot.read_text(encoding="utf-8")
        draw(Document.model_validate_json(text), arguments.plot.with_suffix(""))
        return
    every = fixtures()
    names = arguments.fixtures or list(every)
    unknown = [name for name in names if name not in every]
    if unknown:
        message = (
            f"no fixture is named {', '.join(unknown)}; there are {', '.join(every)}"
        )
        raise SystemExit(message)
    output = arguments.output or _results_path()
    machine = json.loads(json.dumps(bench.machine()))
    document = Document(
        commit=bench.git("rev-parse", "HEAD"),
        machine=machine,
        versions=_versions(),
        repeats=arguments.repeats,
    )
    if output.is_file():
        earlier = Document.model_validate_json(output.read_text(encoding="utf-8"))
        document.fixtures = earlier.fixtures
    started = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    for name in names:
        document.fixtures[name] = run(
            Prepared.of(every[name]),
            arguments.simulators,
            arguments.sweeps,
            arguments.repeats,
        )
        output.write_text(document.model_dump_json(indent=1) + "\n", encoding="utf-8")
    print(f"\nmeasured in {bench.duration(time.perf_counter() - started)}")
    shown = (
        output.relative_to(bench.ROOT) if output.is_relative_to(bench.ROOT) else output
    )
    print(f"wrote {shown}")
    draw(document, output.with_suffix(""))


def _results_path() -> Path:
    """Where the checked-out commit's results go: ``-dirty`` for a changed tree."""
    commit = bench.git("rev-parse", "HEAD")
    dirty = bench.git("status", "--porcelain", "--untracked-files=no") != ""
    return RESULTS / f"{commit}{'-dirty' if dirty else ''}.json"


def _versions() -> dict[str, str]:
    """The versions of jolty, of ngspice, and of the exact answers' arithmetic."""
    found = {"mpmath": metadata.version("mpmath"), "jolty": metadata.version("jolty")}
    if ngspice.EXECUTABLE is not None:
        done = subprocess.run(  # noqa: S603 - ngspice prints its version
            [ngspice.EXECUTABLE, "--version"],
            capture_output=True,
            check=False,
            text=True,
        )
        lines = [line.strip("* ") for line in done.stdout.splitlines()]
        found["ngspice"] = next((line for line in lines if "ngspice-" in line), "")
    return found


def run(
    prepared: Prepared,
    simulators: Sequence[Simulator],
    sweeps: Sequence[Sweep],
    repeats: int,
) -> Measured:
    """Every run of a fixture, each setting in each simulator in turn, scored.

    The simulators take each setting one after another, so that each runs under the
    load the others do.
    """
    fixture = prepared.fixture
    print(f"\n{fixture.name}: {fixture.describes}", flush=True)
    started = time.perf_counter()
    runs: list[Run] = []
    for setting in settings(sweeps):
        cells = []
        for simulator in simulators:
            ran = _run(prepared, setting, repeats, simulator)
            runs.append(ran)
            cells.append(_cell(fixture, ran))
        tolerance = f"{setting.sweep:>8} {setting.relative_tolerance:<8.3g} "
        print(tolerance + " | ".join(cells), flush=True)
    return Measured(
        describes=fixture.describes,
        reference=fixture.reference,
        quantities=list(fixture.quantities),
        seconds=time.perf_counter() - started,
        runs=runs,
    )


def _run(
    prepared: Prepared, setting: Setting, repeats: int, simulator: Simulator
) -> Run:
    """One run of a fixture in `simulator` at `setting`, scored."""
    fixture = prepared.fixture
    begun = Run(
        simulator=simulator,
        sweep=setting.sweep,
        relative_tolerance=setting.relative_tolerance,
        load=bench.load_average(),
        status="ok",
    )
    try:
        if simulator is Simulator.JOLTY:
            ran = jolty(prepared, setting, repeats)
        else:
            ran = spice(prepared, setting, repeats, simulator)
    except Unsupported as reason:
        return begun.model_copy(
            update={"status": "unsupported", "message": str(reason)}
        )
    except JoltyError as error:
        return begun.model_copy(update={"status": "failed", "message": _reason(error)})
    try:
        compared = fixture.compare(ran.found)
        scores = {
            quantity.name: score(compared[quantity.name], quantity)
            for quantity in fixture.quantities
        }
    except ValueError as error:
        return begun.model_copy(update={"status": "failed", "message": str(error)})
    return begun.model_copy(
        update={
            "seconds": statistics.median(ran.seconds),
            "repeats": ran.seconds,
            "points": len(ran.found.points),
            "counts": ran.counts,
            "scores": scores,
        }
    )


REASONS: Final = re.compile(r"error|abort|too small|singular|converge", re.IGNORECASE)
"""What marks the lines of ngspice's output that say why it failed."""


def _reason(error: JoltyError) -> str:
    """Why a run failed: the lines of its message that say so, or its first line."""
    lines = str(error).splitlines()
    said = [line.strip() for line in lines if REASONS.search(line)]
    return " / ".join(said or lines[:1])


def _cell(fixture: Fixture, ran: Run) -> str:
    """A run's digits and time, as the progress prints them."""
    if ran.status != "ok":
        return f"{ran.simulator} {ran.status}"
    digits = []
    for quantity in fixture.quantities:
        found = ran.scored(quantity)
        digits.append("-" if found is None else f"{found.digits:.1f}")
    return f"{ran.simulator} {'/'.join(digits)} in {bench.duration(ran.seconds)}"


COLORS: Final = {
    Simulator.JOLTY: plots.COLORS[0],
    Simulator.NGSPICE_SPARSE: plots.COLORS[1],
    Simulator.NGSPICE_KLU: plots.COLORS[2],
}
"""Each simulator's color, the plots' first three, in their fixed order."""

STYLES: Final = {Sweep.RELTOL: "-", Sweep.SCALED: "--"}
"""Each sweep's line: solid for the relative tolerance alone, dashed for all scaled."""

SURFACE: Final = "#fcfcfb"
INK: Final = "#0b0b0b"
MUTED: Final = "#52514e"
GRID: Final = "#e4e3df"
"""The surface, ink, muted ink and grid of jolty's plots, as the gallery draws its
own."""


def draw(document: Document, directory: Path) -> None:
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


def diagram(name: str, measured: Measured, document: Document) -> Figure:
    """A fixture's work-precision diagram: correct digits against engine time.

    A panel per quantity draws its score's digits against the median engine time, on
    a log axis: a line per simulator configuration and sweep, in the order of their
    tolerances, the defaults ringed, and the digits the exact answer's rounding
    allows dotted.
    """
    figure = Figure(
        figsize=(4.2 * len(measured.quantities) + 1.8, 4.2),
        dpi=150,
        layout="constrained",
        facecolor=SURFACE,
    )
    machine = document.machine
    cpu = machine.get("cpu", "") if isinstance(machine, dict) else ""
    loads = [each.load for each in measured.runs]
    figure.suptitle(
        f"{name}: correct digits against engine time\n{cpu}, each time the median "
        f"of {document.repeats} runs,\nat a load average of {min(loads):.1f} to "
        f"{max(loads):.1f}; commit {document.commit[:7]}",
        color=INK,
        x=0.02,
        ha="left",
        fontsize=9,
    )
    panels = figure.subplots(1, len(measured.quantities), squeeze=False)[0]
    for axes, quantity in zip(panels, measured.quantities, strict=True):
        _panel(axes, quantity, measured.runs)
    panels[-1].legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7
    )
    return figure


def _panel(axes: Axes, quantity: Quantity, runs: Sequence[Run]) -> None:
    """Draws `quantity`'s digits against time on `axes`, a line per configuration."""
    limits: list[float] = []
    for simulator in Simulator:
        for sweep in Sweep:
            chosen = [
                (each.seconds, found)
                for each in runs
                if each.simulator == simulator and each.sweep == sweep
                for found in [each.scored(quantity)]
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
            found = each.scored(quantity)
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
            color=MUTED,
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
        color=MUTED,
        linestyle="none",
        label="defaults",
    )
    axes.set_xscale("log")
    axes.xaxis.set_major_formatter(EngFormatter(unit="s", sep=" "))
    axes.xaxis.set_minor_formatter(FuncFormatter(_minor))
    axes.tick_params(axis="x", which="minor", labelsize=6, labelcolor=MUTED)
    axes.set_xlabel("engine time", color=MUTED)
    axes.set_ylabel(f"correct digits of {quantity.name}, {quantity.score}", color=MUTED)
    axes.set_facecolor(SURFACE)
    axes.grid(visible=True, color=GRID, linewidth=0.6)
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


def defaults_table(document: Document) -> str:
    """Each simulator's digits and time at its defaults, by fixture and quantity."""
    heads = ["fixture", "quantity", *(str(simulator) for simulator in Simulator)]
    rows = [
        "| " + " | ".join(heads) + " |",
        "| " + " | ".join("---" for _ in heads) + " |",
    ]
    for name, measured in document.fixtures.items():
        for quantity in measured.quantities:
            cells = [f"`{name}`", f"{quantity.name}, {quantity.score}"]
            cells += [_default(measured, quantity, each) for each in Simulator]
            rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _default(measured: Measured, quantity: Quantity, simulator: Simulator) -> str:
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
    if found.status != "ok":
        return found.status
    scored = found.scored(quantity)
    digits = "" if scored is None else f"{scored.digits:.1f} digits, "
    return f"{digits}{bench.duration(found.seconds)}"


class Precision(cli.CommandLine):
    """Score each simulator against exact answers, and draw error against cost."""

    model_config = SettingsConfigDict(cli_prog_name="scripts/precision")

    fixtures: CliPositionalArg[list[str]] = Field(
        default_factory=list, description="The fixtures to run, every one by default."
    )
    simulators: list[Simulator] = Field(
        default_factory=lambda: list(Simulator),
        description="The simulator configurations to run, every one by default.",
    )
    sweeps: list[Sweep] = Field(
        default_factory=lambda: list(Sweep),
        description="The sweeps of the tolerances to run, both by default.",
    )
    repeats: int = Field(
        default=5,
        ge=1,
        description="The runs of each setting, whose median is its time.",
    )
    plot: Path | None = Field(
        default=None,
        description="Draw the diagrams and table of this results file instead.",
    )
    output: Path | None = Field(default=None, description="The results file to write.")

    def cli_cmd(self) -> None:
        """Runs the suite."""
        measure(self)


def main(argv: list[str] | None = None) -> None:
    """Runs the command line, `argv` or the process's own arguments."""
    cli.run(Precision, argv)


if __name__ == "__main__":
    main()
