# Copyright 2026 Maxence Faldor
"""Time jolty on the speed suite's fixtures, and compare the times with a baseline's.

``scripts/bench`` runs this script from the repository root, the command line of every
suite of ``benchmarks/``. It first checks that every fixture's files are what its
script in ``benchmarks/speed/circuits`` writes, then times the suite's fixtures twice:
in Rust, by the criterion benchmarks of ``crates/jolty-engine/benches/fixtures.rs``,
and end to end in Python, each fixture's circuit read from its file and its plan's
analyses run through ``Simulation.run``. A Monte Carlo fixture runs its first samples
as the runner does, as one batch of the circuit elaborated once, on one thread and on a
thread per core. It writes the times to ``benchmarks/speed/results/<commit>.json``,
adding to what an earlier run on the same commit wrote, and prints them against a
baseline's, flagging every change over 5 %.

Usage::

    scripts/bench [FIXTURE ...]       the suite's fixtures, or those named
    scripts/bench --scale             the large fixtures, each run once in Python
    scripts/bench --scale --largest   and the 50,000 inverters' chain on each too
    scripts/bench --view matched      every simulator on the fixtures, options matched
    scripts/bench --view defaults     every simulator on them, each at its defaults
    scripts/bench --view equal-error  every simulator's least time to equal error, on
                                      the precision suite's fixtures
    scripts/bench --baseline COMMIT   against COMMIT's results, or a results file's
    scripts/bench --break-even NAME   where the threshold NAME of Parallelism pays
    scripts/bench --robustness        the hard circuits' corpus, in every simulator
    scripts/bench --coverage          the coverage matrix, each row's fixture in each
                                      simulator
    scripts/bench --precision         each simulator's error against exact answers,
                                      against its cost

Without ``--baseline``, the baseline is the newest ancestor of the checked-out commit
whose results ``benchmarks/speed/results`` holds. A working tree with changes to
tracked files writes ``<commit>-dirty.json``, which is never taken as a baseline.

``--scale`` runs each large fixture serially, on one thread with no block of its matrix
dissected, then on 1, 2, 4, 8 threads and every core with the default ``Parallelism``,
every threshold at its break-even, and prints the speed-ups and its results'
fingerprint, which must be the same on every number of threads; it runs the 50,000
inverters' chain so only with ``--largest``, and else once with the defaults.

``--view`` times every simulator instead, the owner's three views of them: jolty,
ngspice with Sparse 1.3 and with KLU, and Xyce, each installed, or those ``--simulator``
names. ``matched`` runs each at options as tight as jolty's tightest,
``jolty.plan.TIGHT``, ``jolty.spice.ngspice.OPTIONS`` and ``jolty.spice.xyce.OPTIONS``,
and ``defaults`` each at its own; each records its time end to end and in its analyses,
as it reports them, and how far the voltages it found lie from jolty's at the matched
options, or what stopped it, and prints the view's table, a row per fixture.
With ``--scale``, Xyce's parallel build, ``--xyce-mpi``, runs each large fixture on 1,
2, 4, 8 and 16 MPI ranks too. ``equal-error`` runs the precision suite, as
``--precision`` does, and prints each simulator's least time to the same correct
digits of each fixture's exact answer.

``--break-even`` measures instead from which size of circuit setting one threshold of
``Parallelism`` makes one run faster than running it serially, on 2, 4, 8 threads and
every core: on circuits of the fixtures' scripts at several sizes, each run serially and
with the threshold set and the others spreading nothing, interleaved, and that size's
value of the threshold, its break-even.
``benchmarks/speed/README.md`` says how it sets each threshold, and what it found.

``--robustness`` runs instead the corpus of hard circuits of ``benchmarks/robustness``,
each fixture in every simulator installed, or those ``--simulator`` names, and prints
each outcome, classified, and each simulator's failure rates;
``benchmarks/robustness/README.md`` says what it holds and how it judges.

``--coverage`` runs instead the fixtures of ``benchmarks/coverage``, each row's in jolty
and ngspice, or the simulators ``--simulator`` names, judges each cell against the row's
known answer, and renders the matrix; ``benchmarks/coverage/README.md`` says how.

``--precision`` runs instead the fixtures of ``benchmarks/precision``, each in jolty,
ngspice with Sparse 1.3 and with KLU, and Xyce, each installed, or the simulators
``--simulator`` names, at their defaults and along the ``--sweeps`` of their
tolerances, scores each run against its exact answer, and draws a work-precision
diagram per fixture; ``benchmarks/precision/README.md`` says what each fixture
measures.

Each suite runs every simulator by its runner of ``benchmarks/simulators.py``, and
writes its results headed by their provenance, ``benchmarks/measuring.py``'s.
"""

import dataclasses
import functools
import hashlib
import importlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Final

import numpy as np
import numpy.typing as npt

# The suites' measuring helpers and runners lie beside the suites' directories.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from measuring import (
    BENCHMARKS,
    ROOT,
    Suite,
    duration,
    git,
    load,
    load_average,
    provenance,
    results_directory,
    results_path,
    save,
    shown,
)
from pydantic import Field
from pydantic_settings import CliPositionalArg, SettingsConfigDict
from simulators import (
    RUSAGE,
    Jolty,
    Ngspice,
    Simulator,
    Xyce,
    command,
    netlist,
    runner,
    stopped,
)

from jolty import circuit, cli, ir, plan, results, simulate
from jolty.errors import JoltyError
from jolty.plan import Ac, Analysis, MonteCarlo, Options, Parallelism, Plan, Transient
from jolty.simulate import Elaborated, Simulation
from jolty.spice import export, ngspice, xyce
from jolty.unknowns import Unknown

sys.path.insert(0, str(BENCHMARKS / "speed" / "circuits"))
sys.path.insert(0, str(BENCHMARKS / "robustness"))
sys.path.insert(0, str(BENCHMARKS / "coverage"))
sys.path.insert(0, str(BENCHMARKS / "precision"))
import bjt_amp
import corpus
import inverter_chain
import matrix
import rc_mesh
import suite
from fixture import CIRCUITS, PLANS, Fixture, engine_plan

SUITE: Final = (
    "bjt_amp",
    "ac_wide",
    "buck",
    "inverter_chain_500",
    "rc_mesh_100",
    "monte_carlo",
)
"""The fixtures timed by default: each runs in seconds at most, so the suite runs in a
few minutes and is timed before and after every change."""

SCALE: Final = (
    "inverter_chain_5000",
    "inverter_chain_50000",
    "rc_mesh_300",
    "monte_carlo",
)
"""The fixtures ``--scale`` times, each once: the large circuits, which take minutes,
and every sample of the Monte Carlo fixture."""

THREADS: Final = (1, 2, 4, 8)
"""The threads ``--scale`` spreads a run of a large fixture on, as far as there are
cores, and then every core, as the Monte Carlo fixture's batches run."""

LONGEST: Final = "inverter_chain_50000"
"""The fixture ``--scale`` runs serially and on each number of threads only with
``--largest``: each run takes minutes."""

TIMEOUT: Final = ngspice.TIMEOUT
"""The seconds ``--robustness`` lets a run take by default: as long as the differential
tests let ngspice run an analysis."""

SAMPLES: Final = 1000
"""The samples of a Monte Carlo fixture the suite times: a few seconds on one thread."""

FLAGGED: Final = 0.05
"""The relative change in a time that the comparison flags, as AGENTS.md says."""

FEWEST_RUNS: Final = 10
"""The fewest times the Python timing repeats a fixture: criterion's fewest samples."""

FEWEST_SECONDS: Final = 5.0
"""The least time the Python timing repeats a fixture for, as long as criterion
measures each benchmark."""

FEWEST_ROUNDS: Final = 3
"""The fewest rounds a break-even's sweep runs a circuit's settings in, one after
another: the fewest whose median no one slow run moves."""

SWEPT_THREADS: Final = (2, 4, 8)
"""The threads a break-even is measured on, as far as there are cores, and then
every core."""


class Threshold(StrEnum):
    """A threshold of `Parallelism`, whose break-even ``--break-even`` measures."""

    MIN_SOLVES_PER_TASK = "min_solves_per_task"
    MIN_INSTANCES_PER_TASK = "min_instances_per_task"
    MIN_UNKNOWNS_PER_TASK = "min_unknowns_per_task"
    MAX_UNKNOWNS_PER_INTERIOR = "max_unknowns_per_interior"


SWEPT: Final[Mapping[Threshold, Mapping[str, tuple[int, ...]]]] = {
    Threshold.MIN_SOLVES_PER_TASK: {
        "bjt_amp": (1, 2, 5, 10, 20, 30, 40, 50, 100, 200, 500, 1111)
    },
    Threshold.MIN_INSTANCES_PER_TASK: {
        "inverter_chain": (1, 2, 5, 10, 20, 50, 70, 100, 200, 300, 500, 700, 1000),
        "rc_mesh": (10, 20, 30, 40, 50, 70, 100, 300),
    },
    Threshold.MIN_UNKNOWNS_PER_TASK: {
        "inverter_chain": (100, 500, 1000, 2000, 3000, 4000, 5000),
        "rc_mesh": (10, 20, 50, 100, 300),
    },
    Threshold.MAX_UNKNOWNS_PER_INTERIOR: {
        "inverter_chain": (10, 100, 500, 1000, 2000, 5000),
        "rc_mesh": (10, 20, 50, 70, 100, 300),
    },
}
"""The circuits each threshold's break-even is measured on: those of the scripts of
``benchmarks/speed/circuits`` it applies to, at the sizes each script's circuits take, a
chain's inverters, a mesh's nodes along a side, or the amplifier's frequencies a decade;
the fixtures' own among them, and others around the break-even."""


@dataclass(frozen=True, slots=True)
class Timing:
    """A time measured: the median of its runs, in seconds, and their number.

    Beside them, the machine's load average over the last minute as they began, when
    it was measured.
    """

    seconds: float
    runs: int
    load: float | None = None


@dataclass(frozen=True, slots=True)
class Loaded:
    """A fixture as the timing runs it: its circuit read from its file, and its plan."""

    circuit: ir.Circuit
    plan: Plan


def scripts() -> list[ModuleType]:
    """The fixtures' scripts of ``benchmarks/speed/circuits``, by their names."""
    return [
        importlib.import_module(path.stem)
        for path in sorted(CIRCUITS.glob("*.py"))
        if path.stem != "fixture"
    ]


def fixtures() -> dict[str, Fixture]:
    """Every fixture, by name, as its script defines it.

    Raises:
        SystemExit: When a fixture's files are not what its script writes.
    """
    found: dict[str, Fixture] = {}
    stale: list[str] = []
    for script in scripts():
        for name, top in script.circuits().items():
            written = circuit(top).model_dump_json() + "\n"
            path = CIRCUITS / f"{name}.json"
            if not path.is_file() or path.read_text(encoding="utf-8") != written:
                stale.append(f"{path.relative_to(ROOT)} ({script.__name__}.py)")
        for name, fixture in script.fixtures().items():
            written = json.dumps(engine_plan(fixture), indent=2) + "\n"
            path = PLANS / f"{name}.json"
            if not path.is_file() or path.read_text(encoding="utf-8") != written:
                stale.append(f"{path.relative_to(ROOT)} ({script.__name__}.py)")
            found[name] = fixture
    if stale:
        message = "these files are not what their scripts write; run the scripts:\n"
        raise SystemExit(message + "\n".join(f"  {each}" for each in stale))
    return found


def load_fixture(fixture: Fixture) -> Loaded:
    """The fixture's circuit, read from its file, and its plan."""
    text = (CIRCUITS / f"{fixture.circuit}.json").read_text(encoding="utf-8")
    return Loaded(ir.Circuit.model_validate_json(text), fixture.plan)


def run(
    loaded: Loaded, options: Options | None = None
) -> dict[str, results.AnalysisResult]:
    """Runs the fixture in jolty: its circuit elaborated, its analyses with `options`.

    The options are the plan's unless given. Returns each analysis's result, by name.
    """
    chosen = loaded.plan.options if options is None else options
    return Jolty().run(loaded.circuit, loaded.plan.analyses, chosen).found


def sampled(loaded: Loaded, samples: int, threads: int) -> Callable[[], None]:
    """The first `samples` of the fixture's Monte Carlo analysis, run on `threads`.

    The samples run as the runner runs them: one batch of the plan's analyses on the
    circuit elaborated once, each sample drawn by the engine from its index, on a pool
    of `threads` threads, the engine releasing the interpreter while it runs; the one
    elaboration counts in the time.
    """
    monte_carlo = loaded.plan.monte_carlo
    seed = 0 if monte_carlo is None else monte_carlo.seed

    def every() -> None:
        simulation = Simulation.from_elaborated(Elaborated(loaded.circuit))
        with simulate.threads(threads):
            simulation.batch(
                loaded.plan.analyses, MonteCarlo(samples, seed), loaded.plan.options
            )

    return every


def repeated[T](work: Callable[[], T], *, once: bool) -> tuple[list[T], float]:
    """What each run of `work` returned, and the load average as they began.

    `work` runs once if `once`; else once to warm up, and then again until it has run
    `FEWEST_RUNS` times and for `FEWEST_SECONDS`, as criterion measures a benchmark.
    """
    load = load_average()
    if once:
        return [work()], load
    work()
    found: list[T] = []
    begun = time.perf_counter()
    while len(found) < FEWEST_RUNS or time.perf_counter() - begun < FEWEST_SECONDS:
        found.append(work())
    return found, load


def timed(work: Callable[[], float], *, once: bool = False) -> Timing:
    """The median of the seconds each run of `work` returns, `repeated`."""
    found, load = repeated(work, once=once)
    return Timing(statistics.median(found), len(found), load)


def clocked(work: Callable[[], object]) -> Callable[[], float]:
    """`work`, returning the seconds it took by the wall clock."""

    def measured() -> float:
        started = time.perf_counter()
        work()
        return time.perf_counter() - started

    return measured


def python(
    names: Iterable[str], found: Mapping[str, Fixture], *, scale: bool
) -> dict[str, Timing]:
    """The Python timings of the fixtures `names`, each run once if at `scale`.

    A Monte Carlo fixture's time is per sample, named by the number of samples run: of
    its first samples on one thread and on a thread per core, or of every sample on a
    thread per core at scale.
    """
    timings: dict[str, Timing] = {}
    cores = os.cpu_count() or 1
    for name in names:
        loaded = load_fixture(found[name])
        monte_carlo = loaded.plan.monte_carlo
        print(f"timing {name} in Python", flush=True)
        if monte_carlo is None:
            work = clocked(functools.partial(run, loaded))
            timings[f"python/{name}"] = timed(work, once=scale)
            continue
        samples = monte_carlo.samples if scale else min(SAMPLES, monte_carlo.samples)
        threads = (
            ((cores, "every core"),)
            if scale
            else ((1, "one core"), (cores, "every core"))
        )
        for count, label in threads:
            whole = timed(clocked(sampled(loaded, samples, count)), once=scale)
            timings[f"python/{name}/{samples} samples, {label}"] = Timing(
                whole.seconds / samples, whole.runs, whole.load
            )
    return timings


def serially(loaded: Loaded) -> Options:
    """The fixture's options for a serial run.

    Its run takes one thread, and no block of its matrix is dissected, its interiors as
    large as its circuit.
    """
    unknowns = len(Simulation(loaded.circuit).unknowns)
    spread = dataclasses.replace(
        loaded.plan.options.parallelism,
        max_threads=1,
        max_unknowns_per_interior=unknowns,
    )
    return dataclasses.replace(loaded.plan.options, parallelism=spread)


def scaling(
    name: str, loaded: Loaded, counts: Sequence[int]
) -> tuple[dict[str, Timing], dict[str, str]]:
    """One run of the fixture `name` serially, then one spread on each of `counts`.

    The serial run runs on one thread with no block of its matrix dissected, `serially`;
    each spread run takes the fixture's options, which leave every setting of its
    parallelism to its default, on a pool of so many threads. Prints each run's time,
    its speed-up against the serial run and against the spread run on one thread, and
    the fingerprint of its results; returns the times and the serial and spread
    runs' fingerprints.

    Raises:
        SystemExit: When, but for a circuit large enough for a block to be dissected,
            whose default interiors depend on the threads, a spread run finds other
            bits than the serial run.
    """
    print(f"timing {name} in Python, serially and spread", flush=True)
    alone_options = serially(loaded)
    serial, alone = once(on(1, functools.partial(run, loaded, alone_options)))
    timings = {f"python/{name}": serial}
    rows = [("serially", serial, alone)]
    print(f"{name}, serially: {duration(serial.seconds)}, load {serial.load:.2f}")
    for count in counts:
        label = f"{count} thread{'s' if count > 1 else ''}"
        spread, found = once(on(count, functools.partial(run, loaded)))
        timings[f"python/{name}/spread on {label}"] = spread
        rows.append((label, spread, found))
        print(f"{name}, spread on {label}: {duration(spread.seconds)}", flush=True)
    print(f"\n{name}, serially and spread by the default parallelism:")
    widths = (10, 10, 14, 16, 16, 0)
    heads = ("threads", "time", "against serial", "against 1 thread", "results", "load")
    print(table(heads, widths))
    one = next((each.seconds for label, each, _ in rows if label == "1 thread"), None)
    for label, each, found in rows:
        against = (
            "" if one is None or label == "serially" else f"{one / each.seconds:.2f}x"
        )
        cells = (
            label,
            duration(each.seconds),
            f"{serial.seconds / each.seconds:.2f}x",
            against,
            found,
            f"{each.load:.2f}",
        )
        print(table(cells, widths))
    spread_prints = {found for label, _, found in rows if label != "serially"}
    unknowns = alone_options.parallelism.max_unknowns_per_interior or 0
    interiors = plan.BREAK_EVENS["max_unknowns_per_interior"]
    dissectable = unknowns > min(each for each in interiors if each is not None)
    if not dissectable and spread_prints != {alone}:
        message = f"{name} found other bits spread: {', '.join(f for *_, f in rows)}"
        raise SystemExit(message)
    return timings, {"serial": alone, "spread": next(iter(spread_prints), alone)}


def once(work: Callable[[], dict[str, results.AnalysisResult]]) -> tuple[Timing, str]:
    """The time of one run of `work`, and the fingerprint of what it found."""
    load = load_average()
    started = time.perf_counter()
    found = work()
    return Timing(time.perf_counter() - started, 1, load), fingerprint(found)


@dataclass(frozen=True, slots=True)
class Profiled:
    """What a transient did in a run with its options' ``profile`` on."""

    work: results.Work
    profile: results.Profile


def profiles(
    names: Iterable[str], found: Mapping[str, Fixture]
) -> dict[str, dict[str, Profiled]]:
    """The work and the profile of each transient of the fixtures `names`, by analysis.

    Each fixture runs once more for them, with its options' ``profile`` on, which costs
    a little time of its own; a Monte Carlo fixture is not profiled.
    """
    found_profiles: dict[str, dict[str, Profiled]] = {}
    for name in names:
        loaded = load_fixture(found[name])
        if loaded.plan.monte_carlo is not None:
            continue
        print(f"profiling {name}", flush=True)
        simulation = Simulation(loaded.circuit)
        options = dataclasses.replace(loaded.plan.options, profile=True)
        for analysis, planned in loaded.plan.analyses.items():
            result = simulation.run(planned, options)
            if not isinstance(result, results.TransientResult):
                continue
            work, profile = result.statistics.work, result.statistics.profile
            if work is not None and profile is not None:
                found_profiles.setdefault(name, {})[analysis] = Profiled(work, profile)
    return found_profiles


def shares(found_profiles: Mapping[str, Mapping[str, Profiled]]) -> None:
    """Prints where each profiled transient's time went, and its work."""
    if found_profiles:
        print("\nwhere each transient's time went, in one run with its profile on:")
    for name, analyses in found_profiles.items():
        for analysis, profiled in analyses.items():
            total = profiled.profile.total
            spent = ", ".join(
                f"{activity.replace('_', ' ')} {seconds / total:.0%}"
                for activity, seconds in dataclasses.asdict(profiled.profile).items()
            )
            counted = ", ".join(
                f"{count:,} {kind}"
                for kind, count in dataclasses.asdict(profiled.work).items()
            )
            print(f"{name}/{analysis}: {duration(total)}; {spent}; {counted}")


def rust(names: Iterable[str], found: Mapping[str, Fixture]) -> dict[str, Timing]:
    """The criterion timings of the fixtures `names`.

    A Monte Carlo fixture's time is per sample of its first samples, run as one batch,
    named by the number of samples and of threads.

    Raises:
        SystemExit: When the benchmarks fail.
    """
    chosen = list(names)
    if not chosen:
        return {}
    pattern = "^(" + "|".join(chosen) + ")/"
    command = [
        "cargo",
        "bench",
        "--locked",
        "-p",
        "jolty-engine",
        "--bench",
        "fixtures",
    ]
    done = subprocess.run(  # noqa: S603 - cargo runs the benchmarks
        [*command, "--", pattern], check=False, cwd=ROOT
    )
    if done.returncode != 0:
        message = f"the criterion benchmarks failed with {done.returncode}"
        raise SystemExit(message)
    metadata = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],  # noqa: S607 - cargo is on the path
        capture_output=True,
        check=True,
        cwd=ROOT,
        text=True,
    )
    measured = Path(json.loads(metadata.stdout)["target_directory"]) / "criterion"
    timings: dict[str, Timing] = {}
    for name in chosen:
        monte_carlo = found[name].plan.monte_carlo
        if monte_carlo is None:
            for function in ("setup", "shared", "run"):
                timings[f"rust/{name}/{function}"] = _criterion(
                    measured / name / function / "new", 1
                )
            continue
        samples = min(SAMPLES, monte_carlo.samples)
        for directory in sorted(
            (measured / name / "batch").iterdir(), key=lambda path: int(path.name)
        ):
            threads = int(directory.name)
            label = "1 thread" if threads == 1 else f"{threads} threads"
            timings[f"rust/{name}/{samples} samples, {label}"] = _criterion(
                directory / "new", samples
            )
    return timings


def _criterion(new: Path, runs: int) -> Timing:
    """Criterion's median in `new`, per run of the `runs` each iteration makes.

    The directory holds criterion's latest estimates and samples of one benchmark.
    """
    estimates = json.loads((new / "estimates.json").read_text("utf-8"))
    sample = json.loads((new / "sample.json").read_text("utf-8"))
    seconds = estimates["median"]["point_estimate"] * 1e-9 / runs
    return Timing(seconds, round(sum(sample["iters"])))


def commands(loaded: Loaded) -> list[str]:
    """Each analysis of the fixture's plan as ngspice's command, in order.

    Raises:
        TypeError: When the plan runs an analysis no suite runs in another simulator.
    """
    return [
        command(analysis, loaded.circuit) for analysis in loaded.plan.analyses.values()
    ]


class View(StrEnum):
    """A view of every simulator's times, the owner's three, which ``--view`` gives.

    Every simulator at matched options, each as tight as jolty's tightest; each at its
    own defaults; and each at equal measured error, its least time to the same correct
    digits of the precision suite's exact answers.
    """

    MATCHED = "matched"
    DEFAULTS = "defaults"
    EQUAL_ERROR = "equal-error"


VIEWED: Final = (
    Simulator.JOLTY,
    Simulator.NGSPICE_SPARSE,
    Simulator.NGSPICE_KLU,
    Simulator.XYCE_SERIAL,
)
"""The simulators a view of the fixtures' times runs, each that is installed."""

RANKS: Final = (1, 2, 4, 8, 16)
"""The MPI ranks a view with ``--scale`` runs Xyce's parallel build on: as far as the
16 cores of `dogwood`, the machine Xyce is built on, which the lab shares."""

PRINTED: Final = 2
"""The nets at each end of a fixture's top that ngspice saves and Xyce prints, so that
what they write costs little beside what they solve, as other simulators' comparisons
turn output off: the first two and the last two."""


def printed(loaded: Loaded) -> list[str]:
    """The nets of the fixture's top the views read: `PRINTED` at each end, in order."""
    top = loaded.circuit.components[loaded.circuit.top]
    ground = {port.name for port in top.ports if port.role is ir.Role.GROUND}
    nets = list(
        dict.fromkeys(
            net
            for instance in top.instances
            for net in instance.terminals.values()
            if net not in ground
        )
    )
    return list(dict.fromkeys(nets[:PRINTED] + nets[-PRINTED:]))


def tightened(options: Options) -> Options:
    """`options` as tight as `jolty.plan.TIGHT`, as the matched view runs jolty."""
    return dataclasses.replace(
        options, **{name: getattr(plan.TIGHT, name) for name in plan.TIGHT.to_json()}
    )


def stated(view: View, runner: Jolty | Ngspice | Xyce) -> str:
    """The options `runner`'s simulator runs the fixtures at in `view`, as stated.

    At matched options, jolty's are `jolty.plan.TIGHT`'s, ngspice's
    `jolty.spice.ngspice.OPTIONS` and Xyce's `jolty.spice.xyce.OPTIONS`; ngspice with
    KLU adds its solver's line in either view.
    """
    matched = view is View.MATCHED
    match runner:
        case Jolty():
            tight = ", ".join(f"{n}={v!r}" for n, v in plan.TIGHT.to_json().items())
            lines = [f"jolty.plan.TIGHT: {tight}"] if matched else []
        case Ngspice():
            lines = [*([ngspice.OPTIONS] if matched else []), *runner.options]
        case Xyce():
            lines = list(xyce.OPTIONS) if matched else []
    return "; ".join(f"`{line}`" for line in lines) or "its defaults"


@dataclass(frozen=True, slots=True)
class Voltages:
    """The voltages of the nets a view reads, as one analysis found them.

    At the points it chose: a transient's times or a response's frequencies, and none
    at an operating point.
    """

    points: npt.NDArray[np.float64] | None
    nets: dict[str, npt.NDArray[np.complex128]]


def jolty_voltages(found: results.AnalysisResult, nets: Sequence[str]) -> Voltages:
    """The voltages of `nets` jolty `found`.

    Raises:
        TypeError: When the result is of an analysis other than an operating point, a
            small-signal response or a transient.
    """
    match found:
        case results.TransientResult():
            points = found.times
        case results.AcResult():
            points = found.frequencies
        case results.OperatingPointResult():
            points = None
        case _:
            message = f"no fixture's {type(found).__name__} is viewed"
            raise TypeError(message)
    return Voltages(
        points,
        {net: np.atleast_1d(np.asarray(found.voltage(net), complex)) for net in nets},
    )


def spice_voltages(
    found: ngspice.Result | xyce.Result,
    analysis: Analysis,
    vectors: Mapping[str, str],
) -> Voltages:
    """The voltages ngspice or Xyce `found` of `analysis`, each net by its vector."""
    match analysis:
        case Transient():
            points = found.real("time")
        case Ac():
            points = found.real("frequency")
        case _:
            points = None
    return Voltages(
        points,
        {
            net: found.vectors[name][: None if points is not None else 1]
            for net, name in vectors.items()
        },
    )


def deviations(
    analyses: Mapping[str, Analysis],
    reference: Mapping[str, Voltages],
    found: Mapping[str, Voltages],
) -> dict[str, float]:
    """How far each analysis `found` lies from the `reference`, as a fraction of it.

    For each analysis, the largest difference over the nets, each over the largest
    magnitude of the reference's voltage or, in a transient, its swing, the reference
    interpolated at the points found.
    """
    worst: dict[str, float] = {}
    for name, voltages in found.items():
        expected = reference[name]
        largest = 0.0
        for net, values in voltages.nets.items():
            ours = expected.nets[net]
            there = (
                ours
                if voltages.points is None or expected.points is None
                else np.interp(voltages.points, expected.points, ours)
            )
            swing = isinstance(analyses[name], Transient)
            scale = float(np.ptp(ours.real) if swing else np.abs(ours).max())
            largest = max(largest, float(np.abs(values - there).max()) / (scale or 1.0))
        worst[name] = largest
    return worst


@dataclass(frozen=True, slots=True)
class Ran:
    """One run of a fixture's analyses in one simulator, and the voltages it found."""

    seconds: float
    """Its time end to end: jolty's elaboration and analyses, their results made in
    Python, or each process of another simulator from its start to its exit, summed."""
    analysis_seconds: float
    """Its analyses' own time: jolty's ``Simulation.run``, or what the other simulator
    reports of them, summed."""
    voltages: dict[str, Voltages]
    """By analysis."""


def ran(
    view: View,
    runner: Jolty | Ngspice | Xyce,
    loaded: Loaded,
    nets: Sequence[str],
    timeout: float,
) -> Ran:
    """One run of the fixture's analyses by `runner`, at `view`'s options.

    jolty runs in this process; ngspice runs every analysis in one process, saving the
    `nets` alone, and Xyce each in a process of its own, printing them; each for at most
    `timeout` seconds.

    Raises:
        JoltyError: When the run fails, runs out of time, or another simulator cannot
            read the fixture's netlist.
    """
    matched = view is View.MATCHED
    analyses = loaded.plan.analyses
    match runner:
        case Jolty():
            options = loaded.plan.options
            started = time.perf_counter()
            analysed = runner.run(
                loaded.circuit, analyses, tightened(options) if matched else options
            )
            seconds = time.perf_counter() - started
            voltages = {
                name: jolty_voltages(found, nets)
                for name, found in analysed.found.items()
            }
            return Ran(seconds, analysed.analysis_seconds, voltages)
        case Ngspice():
            vectors = {
                net: export.vector(loaded.circuit, Unknown(net, "voltage"))
                for net in nets
            }
            lines = netlist(loaded.circuit, analyses.values())
            lines.append(f".save {' '.join(vectors.values())}")
            *others, last = commands(loaded)
            found = runner.runs(
                lines,
                [*others, f"{last}\nrusage all"],
                options=ngspice.OPTIONS if matched else ngspice.NO_OPTIONS,
                timeout=timeout,
            )
            said, seconds = found[-1].printed, found[-1].seconds
            stated = RUSAGE["analysis_seconds"].search(said)
            if runner.solver not in said or stated is None or seconds is None:
                message = f"ngspice ran another solver, or stated no time:\n{said}"
                raise JoltyError(message)
            voltages = {
                name: spice_voltages(each, analysis, vectors)
                for (name, analysis), each in zip(analyses.items(), found, strict=True)
            }
            return Ran(seconds, float(stated.group(1)), voltages)
        case Xyce():
            vectors = {
                net: export.vector(
                    loaded.circuit, Unknown(net, "voltage"), dialect=export.XYCE
                )
                for net in nets
            }
            lines = netlist(loaded.circuit, analyses.values(), dialect=export.XYCE)
            seconds = analysis_seconds = 0.0
            voltages = {}
            for name, analysis in analyses.items():
                each = runner.run(
                    lines,
                    f".{command(analysis, loaded.circuit)}",
                    printed=list(vectors.values()),
                    options=xyce.OPTIONS if matched else (),
                    timeout=timeout,
                )
                seconds += each.seconds
                analysis_seconds += each.analysis_seconds
                voltages[name] = spice_voltages(each, analysis, vectors)
            return Ran(seconds, analysis_seconds, voltages)


@dataclass(frozen=True, slots=True)
class Cell:
    """A fixture's runs in one simulator in one view, as its table shows them.

    Their median times end to end and in their analyses, and how far each analysis
    lay from the reference, jolty at the matched options; or what stopped them.
    """

    options: str
    """The options the simulator ran at, as `stated`."""
    load: float
    """The load average over the last minute as the runs began."""
    load_after: float
    """The same as they ended."""
    seconds: Timing | None = None
    analysis_seconds: Timing | None = None
    deviations: dict[str, float] = dataclasses.field(default_factory=dict)
    failed: str | None = None
    """What stopped the runs, if anything did."""


def cell(
    options: str,
    work: Callable[[], Ran],
    analyses: Mapping[str, Analysis],
    reference: Mapping[str, Voltages] | None,
    *,
    once: bool,
) -> Cell:
    """The runs of `work` at the `options` stated, `repeated` or `once`, as a `Cell`.

    The first run that fails stops them, and the cell says what stopped it; else each
    of the `analyses` the first run found is compared with the `reference`, if there
    is one.
    """
    load = load_average()
    try:
        found, _ = repeated(work, once=once)
    except JoltyError as error:
        return Cell(options, load, load_average(), failed=stopped(str(error)))
    return Cell(
        options,
        load,
        load_average(),
        Timing(statistics.median(each.seconds for each in found), len(found), load),
        Timing(
            statistics.median(each.analysis_seconds for each in found),
            len(found),
            load,
        ),
        {} if reference is None else deviations(analyses, reference, found[0].voltages),
    )


def view_timings(  # noqa: PLR0913 - the view, its fixtures, and how they run
    view: View,
    names: Iterable[str],
    found: Mapping[str, Fixture],
    simulators: Sequence[Simulator],
    *,
    parallel: Path | None,
    ranks: Sequence[int],
    scale: bool,
    timeout: float,
) -> dict[str, dict[str, Cell]]:
    """Each fixture of `names` in each of `simulators` in `view`, by configuration.

    Each simulator named or, without names, each of `VIEWED` installed, and given
    Xyce's `parallel` build at `scale`, it on each of `ranks` MPI ranks; each runs a
    fixture `repeated`, or once at `scale`, for at most `timeout` seconds a run; a
    Monte Carlo fixture is not viewed. Each fixture runs first in jolty at the matched
    options, the reference every run's voltages are compared with, unless it fails.

    Raises:
        SystemExit: When a simulator named is not installed or not one a view runs.
    """
    chosen = simulators or [
        each for each in VIEWED if runner(each).version() is not None
    ]
    strange = [each for each in chosen if each not in VIEWED]
    missing = [
        each
        for each in chosen
        if each not in strange and runner(each).version() is None
    ]
    if strange or missing:
        message = (
            f"a view runs {', '.join(VIEWED)}, each that is installed, "
            f"not {', '.join([*strange, *missing])}"
        )
        raise SystemExit(message)
    runners: dict[str, Jolty | Ngspice | Xyce] = {
        str(each): runner(each) for each in chosen
    }
    if parallel is not None and scale:
        runners |= {
            f"{Simulator.XYCE_MPI}/{count} ranks": Xyce.from_parallel_build(
                parallel, count
            )
            for count in ranks
        }
    cells: dict[str, dict[str, Cell]] = {}
    for name in names:
        loaded = load_fixture(found[name])
        if loaded.plan.monte_carlo is not None:
            continue
        print(f"running {name} in jolty at the matched options, the reference")
        nets = printed(loaded)
        try:
            found_tight = run(loaded, tightened(loaded.plan.options))
        except JoltyError as error:
            print(f"  no reference: {stopped(str(error))}", flush=True)
            reference = None
        else:
            reference = {
                analysis: jolty_voltages(result, nets)
                for analysis, result in found_tight.items()
            }
        for configuration, each in runners.items():
            print(f"timing {name} in {configuration}", flush=True)
            work = functools.partial(ran, view, each, loaded, nets, timeout)
            measured = cell(
                stated(view, each), work, loaded.plan.analyses, reference, once=scale
            )
            cells.setdefault(name, {})[configuration] = measured
            print(f"  {entry(measured)}", flush=True)
    return cells


def entry(measured: Cell) -> str:
    """A cell as the view's table writes it."""
    if measured.failed is not None:
        return f"failed: {measured.failed}"
    if measured.seconds is None or measured.analysis_seconds is None:
        return ""
    lying = ", ".join(f"{n} {v:.1e}" for n, v in measured.deviations.items()) or (
        "no reference, as jolty failed at the matched options"
    )
    return (
        f"{duration(measured.seconds.seconds)}, "
        f"{duration(measured.analysis_seconds.seconds)} in its analyses; {lying}"
    )


def view_table(cells: Mapping[str, Mapping[str, Cell]]) -> str:
    """A view's table of `cells`: a row per fixture, a column per configuration.

    Its first row states each configuration's options.
    """
    configurations = list(dict.fromkeys(c for row in cells.values() for c in row))
    options = {c: row[c].options for row in cells.values() for c in row}
    rows = [
        ["fixture", *configurations],
        ["---"] * (len(configurations) + 1),
        ["options", *(options[c] for c in configurations)],
    ]
    rows += [
        [f"`{name}`", *(entry(row[c]) if c in row else "" for c in configurations)]
        for name, row in cells.items()
    ]
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def baseline(given: str | None) -> Path | None:
    """The results to compare with: `given`'s, a commit's or a file, if given.

    Otherwise they are those of the newest ancestor of the checked-out commit that has
    results, if any.

    Raises:
        SystemExit: When `given` names no results.
    """
    if given is not None:
        path = Path(given)
        if path.is_file():
            return path
        found = results_directory(Suite.SPEED) / f"{git('rev-parse', given)}.json"
        if found.is_file():
            return found
        message = f"no results of {given}; run scripts/bench on it first"
        raise SystemExit(message)
    for commit in git("rev-list", "--max-count=1000", "HEAD").split()[1:]:
        found = results_directory(Suite.SPEED) / f"{commit}.json"
        if found.is_file():
            return found
    return None


def compare(now: Mapping[str, Timing], before: Mapping[str, Timing], name: str) -> None:
    """Prints the times `now` against those `before`, from the results `name`.

    Each change over 5 % is flagged.
    """
    shared = [key for key in now if key in before]
    if not shared:
        print(f"\nno time measured here is in {name}")
        return
    width = max(len(key) for key in shared)
    print(f"\nagainst {name}:")
    print(f"{'':{width}}  {'before':>10}  {'now':>10}  {'change':>8}")
    flagged = 0
    for key in shared:
        old, new = before[key].seconds, now[key].seconds
        change = new / old - 1
        mark = ""
        if abs(change) > FLAGGED:
            flagged += 1
            mark = "  slower" if change > 0 else "  faster"
        print(
            f"{key:{width}}  {duration(old):>10}  {duration(new):>10}  "
            f"{change:>+8.1%}{mark}"
        )
    print(f"{flagged} of {len(shared)} changed by more than {FLAGGED:.0%}")


def fingerprint(found: Mapping[str, results.AnalysisResult]) -> str:
    """The first 16 hexadecimal digits of the SHA-256 of `found`'s arrays.

    The arrays of each analysis's result, by name, in order: the same digits for the
    same bits.
    """
    digest = hashlib.sha256()
    for name, result in found.items():
        digest.update(name.encode())
        for each in dataclasses.fields(result):
            value = getattr(result, each.name)
            if isinstance(value, np.ndarray):
                digest.update(value.tobytes())
    return digest.hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class Sized:
    """A circuit of a fixture's script at one size, and what the thresholds count."""

    size: int
    loaded: Loaded
    devices: int
    """The devices of its hierarchy, flattened, which tasks of instances split."""
    unknowns: int
    """Its unknowns, which the work inside and a dissection count."""
    solves: int
    """The frequencies of its responses but each one's first, which tasks of solves
    split."""


def devices(built: ir.Circuit) -> int:
    """The devices of `built`, each component's counted as often as it is placed."""

    @functools.cache
    def within(name: str) -> int:
        return sum(
            within(instance.of.component)
            if isinstance(instance.of, ir.ComponentOf)
            else 1
            for instance in built.components[name].instances
        )

    return within(built.top)


def sized(script: str, size: int) -> Sized:
    """The circuit of `script` at `size`, and the plan its fixtures run on it.

    A chain of `size` inverters and a mesh of `size` nodes a side run their fixtures'
    transients, and the amplifier the response of ``ac_wide`` at `size` frequencies a
    decade.

    Raises:
        ValueError: When `script` is none of the three.
    """
    match script:
        case "inverter_chain":
            loaded = Loaded(circuit(inverter_chain.build(size)), inverter_chain.plan())
        case "rc_mesh":
            loaded = Loaded(circuit(rc_mesh.build(size)), rc_mesh.plan())
        case "bjt_amp":
            loaded = Loaded(circuit(bjt_amp.build()), bjt_amp.wide(size))
        case _:
            message = f"no break-even is measured on {script}'s circuits"
            raise ValueError(message)
    solves = sum(
        len(analysis.frequencies) - 1
        for analysis in loaded.plan.analyses.values()
        if isinstance(analysis, Ac)
    )
    unknowns = len(Simulation(loaded.circuit).unknowns)
    return Sized(size, loaded, devices(loaded.circuit), unknowns, solves)


def counted(threshold: Threshold, at: Sized) -> int:
    """What `threshold` counts of `at`: its solves, devices or unknowns."""
    match threshold:
        case Threshold.MIN_SOLVES_PER_TASK:
            return at.solves
        case Threshold.MIN_INSTANCES_PER_TASK:
            return at.devices
        case Threshold.MIN_UNKNOWNS_PER_TASK | Threshold.MAX_UNKNOWNS_PER_INTERIOR:
            return at.unknowns


def setting(threshold: Threshold, at: Sized, count: int) -> int:
    """The value of `threshold` that spreads `at`'s work over `count` threads.

    Tasks of at least a `count`-th of its solves, devices or unknowns, rounded down, or
    interiors of at most a `count`-th of its unknowns, rounded up.
    """
    if threshold is Threshold.MAX_UNKNOWNS_PER_INTERIOR:
        return -(-at.unknowns // count)
    return max(1, counted(threshold, at) // count)


def spreading_none(at: Sized) -> dict[str, int]:
    """Each threshold set to spread none of `at`'s work.

    Tasks of all of its solves, devices or unknowns, and interiors as large as the
    circuit.
    """
    return {each.value: max(1, counted(each, at)) for each in Threshold}


def on(
    count: int, work: Callable[[], dict[str, results.AnalysisResult]]
) -> Callable[[], dict[str, results.AnalysisResult]]:
    """`work`, run on the engine's pool of `count` threads."""

    def within() -> dict[str, results.AnalysisResult]:
        with simulate.threads(count):
            return work()

    return within


def interleaved(
    works: Sequence[Callable[[], dict[str, results.AnalysisResult]]],
) -> tuple[list[Timing], list[str]]:
    """The median time of each of `works`, run in turn, and what each first found.

    Each work runs once a round, one after another, so that each side of a comparison
    runs under the same load as the others, and what it found in the first round is
    fingerprinted. The rounds run the works in their order and in the reverse order by
    turns, so that no work always follows the same one: on `cedar`, the median of a
    serial run of a quarter of a second was 261 ms when it always followed one on 64
    threads, and 225 ms when it followed one on 2. The rounds go on until there are
    `FEWEST_ROUNDS` of them and they have taken `FEWEST_SECONDS` for each work; none is
    warmed up, the median discarding a first slow run.
    """
    load = load_average()
    times: list[list[float]] = [[] for _ in works]
    prints: list[str | None] = [None for _ in works]
    begun = time.perf_counter()
    while len(
        times[0]
    ) < FEWEST_ROUNDS or time.perf_counter() - begun < FEWEST_SECONDS * len(works):
        order = list(range(len(works)))
        if len(times[0]) % 2:
            order.reverse()
        for index in order:
            started = time.perf_counter()
            found = works[index]()
            times[index].append(time.perf_counter() - started)
            if prints[index] is None:
                prints[index] = fingerprint(found)
    timings = [Timing(statistics.median(each), len(each), load) for each in times]
    return timings, [found or "" for found in prints]


SIZE: Final = {"inverter_chain": "inverters", "rc_mesh": "side", "bjt_amp": "a decade"}
"""What a size counts of each script's circuits."""


def table(cells: Sequence[str], widths: Sequence[int]) -> str:
    """A row of `cells`, each right-aligned to its width, two spaces apart."""
    return "  ".join(
        f"{cell:>{width}}" for cell, width in zip(cells, widths, strict=True)
    ).rstrip()


def pays_from(changes: Sequence[float]) -> int | None:
    """The index of the first of `changes` from which each is a gain, if any."""
    first: int | None = None
    for index, change in enumerate(changes):
        if change >= 0:
            first = None
        elif first is None:
            first = index
    return first


def sweep(
    threshold: Threshold, script: str, sizes: Sequence[int], counts: Sequence[int]
) -> dict[str, Timing]:
    """The times of `script`'s circuits at `sizes`, serially and `threshold` set.

    Each circuit runs serially, on one thread with no threshold spreading its work, and
    with `threshold` set to spread it over each of `counts` threads, on so many, the
    others `spreading_none`, interleaved. The times are printed as they are
    measured, and then the smallest size from which setting it pays on each count, and
    its break-even there.

    Raises:
        SystemExit: When a run with `threshold` set finds other bits than the serial
            run, but for a dissection, whose interiors, a size for each number of
            threads, order the factorization otherwise.
    """
    labels = [f"{count} thread{'s' if count > 1 else ''}" for count in counts]
    heads = [SIZE[script], "devices", "unknowns", "solves", "serially"]
    widths = [10, 9, 9, 7, 10, *(len(label) + 18 for label in labels), 0]
    print(f"\n{threshold} on {script}'s circuits, serially and set, to the value in")
    print("brackets, to spread each circuit over each number of threads:")
    print(table([*heads, *labels, "load"], widths), flush=True)
    timings: dict[str, Timing] = {}
    measured: list[tuple[Sized, list[float]]] = []
    for size in sizes:
        at = sized(script, size)
        values = [setting(threshold, at, count) for count in counts]
        none = spreading_none(at)
        serial = dataclasses.replace(
            at.loaded.plan.options, parallelism=Parallelism(**none)
        )
        works = [on(1, functools.partial(run, at.loaded, serial))]
        for count, value in zip(counts, values, strict=True):
            spread = Parallelism(**none | {threshold.value: value})
            options = dataclasses.replace(at.loaded.plan.options, parallelism=spread)
            works.append(on(count, functools.partial(run, at.loaded, options)))
        found, prints = interleaved(works)
        # Interiors of other sizes order the factorization otherwise, so only the
        # other thresholds, which change no result, must find the serial run's bits.
        dissected = threshold is Threshold.MAX_UNKNOWNS_PER_INTERIOR
        if not dissected and len(set(prints)) != 1:
            message = (
                f"{script} at {size} found other bits with {threshold} set: "
                + ", ".join(prints)
            )
            raise SystemExit(message)
        unset = found[0].seconds
        changes = [each.seconds / unset - 1 for each in found[1:]]
        measured.append((at, changes))
        key = f"python/break-even/{threshold}/{script}_{size}"
        timings[f"{key}/serially"] = found[0]
        for label, each in zip(labels, found[1:], strict=True):
            timings[f"{key}/{label}"] = each
        cells = [f"{each:,}" for each in (size, at.devices, at.unknowns, at.solves)]
        cells.append(duration(unset))
        cells += [
            f"{duration(each.seconds)} {change:+.1%} ({value:,})"
            for each, change, value in zip(found[1:], changes, values, strict=True)
        ]
        cells.append(f"{found[0].load:.2f} to {load_average():.2f}")
        print(table(cells, widths), flush=True)
    for place, label in enumerate(labels):
        changes = [each[place] for _, each in measured]
        first = pays_from(changes)
        if first is None:
            print(f"on {label}: no gain from any size up to {sizes[-1]:,}")
            continue
        at = measured[first][0]
        before = (
            ""
            if first == 0
            else f", after {changes[first - 1]:+.1%} at {sizes[first - 1]:,}"
        )
        print(
            f"on {label}: pays from {at.size:,}, {changes[first]:+.1%}{before}; "
            f"break-even {setting(threshold, at, counts[place]):,}"
        )
    return timings


class Bench(cli.CommandLine):
    """Time the fixtures, write the times, and compare them with a baseline's."""

    model_config = SettingsConfigDict(cli_prog_name="scripts/bench")

    fixtures: CliPositionalArg[list[str]] = Field(
        default_factory=list,
        description=(
            "The fixtures to time, the suite's by default; with --break-even, the "
            "scripts whose circuits it sweeps, every one it applies to by default; "
            "with --robustness, --coverage or --precision, the fixtures, or the "
            "coverage matrix's rows, it runs, every one by default."
        ),
    )
    scale: bool = Field(
        default=False, description="Time the large fixtures, once each."
    )
    break_even: Threshold | None = Field(
        default=None,
        description=(
            "Measure this threshold's break-even instead: the circuits of the "
            "fixtures' scripts it applies to, serially and set, from which size on "
            "setting it pays."
        ),
    )
    largest: bool = Field(
        default=False,
        description=(
            "With --scale, run the 50,000 inverters' chain serially and on each "
            "number of threads too: 17 minutes more on 12 cores."
        ),
    )
    sizes: list[int] = Field(
        default_factory=list,
        description="The sizes --break-even sweeps; the README's by default.",
    )
    threads: list[int] = Field(
        default_factory=list,
        description=(
            "The threads --break-even sets the threshold on, 2, 4, 8 and every core "
            "by default, or --scale spreads a run on, 1, 2, 4, 8 and every core."
        ),
    )
    view: View | None = Field(
        default=None,
        description=(
            "Time every simulator instead, in one view: at matched options, at its "
            "own defaults, or at equal error on the precision suite's fixtures."
        ),
    )
    digits: float | None = Field(
        default=None,
        gt=0,
        description=(
            "The correct digits --precision and --view equal-error time each "
            "simulator to; on each output, the most every simulator reached by "
            "default."
        ),
    )
    robustness: bool = Field(
        default=False,
        description=(
            "Run the corpus of hard circuits instead: each fixture in each "
            "simulator, its outcome classified, and each simulator's failure rates."
        ),
    )
    coverage: bool = Field(
        default=False,
        description=(
            "Run the coverage matrix's fixtures instead: each row in each simulator, "
            "each cell judged against the row's known answer, the matrix rendered."
        ),
    )
    precision: bool = Field(
        default=False,
        description=(
            "Run the precision suite instead: each fixture in each simulator, at its "
            "defaults and along the sweeps of its tolerances, scored against its "
            "exact answer, and a work-precision diagram per fixture."
        ),
    )
    simulator: list[Simulator] = Field(
        default_factory=list,
        description=(
            "The simulators --view, --robustness, --coverage or --precision runs, "
            "such as jolty, ngspice-sparse or xyce-serial; by default every one "
            "installed for --view, --robustness and --precision, and jolty and "
            "ngspice with Sparse and with KLU for --coverage."
        ),
    )
    timeout: float = Field(
        default=TIMEOUT,
        gt=0,
        description=(
            "The seconds --view or --robustness lets a run take before it stops it."
        ),
    )
    sweeps: list[suite.Sweep] = Field(
        default_factory=lambda: list(suite.Sweep),
        description="The sweeps of the tolerances --precision runs, both by default.",
    )
    repeats: int = Field(
        default=5,
        ge=1,
        description="The runs of each setting --precision takes the median time of.",
    )
    plot: Path | None = Field(
        default=None,
        description=(
            "With --precision or --view equal-error, draw the diagrams and tables "
            "of this results file of the precision suite instead."
        ),
    )
    xyce_mpi: Path | None = Field(
        default=None,
        description=(
            "Xyce's parallel build, which --view with --scale runs on 1, 2, 4, 8 and "
            "16 MPI ranks too, or the --threads given."
        ),
    )
    baseline: str | None = Field(
        default=None, description="The commit or results file to compare with."
    )
    output: Path | None = Field(default=None, description="The results file to write.")

    def cli_cmd(self) -> None:
        """Runs the benchmarks."""
        measure(self)


def main(argv: list[str] | None = None) -> None:
    """Runs the command line, `argv` or the process's own arguments."""
    cli.run(Bench, argv)


def every(counts: Iterable[int]) -> list[int]:
    """Each of `counts` that is fewer than the cores, and then every core."""
    cores = os.cpu_count() or 1
    return [count for count in counts if count < cores] + [cores]


def merged(
    output: Path,
    timings: Mapping[str, Timing],
    profiles: Mapping[str, object] | None = None,
    fingerprints: Mapping[str, object] | None = None,
    views: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """The results file `output`, with what this run measured added.

    `timings`, `profiles`, `fingerprints` and `views` each add to what the file holds
    of its kind, if anything: `views`, each cell of a view's table, by its view, its
    configuration and its fixture.
    """
    earlier = load(output) if output.is_file() else {}
    written = {**earlier.get("timings", {})}
    written |= {
        key: {"seconds": each.seconds, "runs": each.runs}
        | ({} if each.load is None else {"load": each.load})
        for key, each in timings.items()
    }
    return {
        **provenance().model_dump(mode="json"),
        "timings": written,
        "profiles": earlier.get("profiles", {}) | dict(profiles or {}),
        "fingerprints": earlier.get("fingerprints", {}) | dict(fingerprints or {}),
        "views": earlier.get("views", {}) | dict(views or {}),
    }


def break_evens(arguments: Bench, threshold: Threshold) -> None:
    """Measures `threshold`'s break-even on the scripts named, and writes the times.

    Raises:
        SystemExit: When a script named is not one the threshold applies to.
    """
    swept = SWEPT[threshold]
    scripts = arguments.fixtures or list(swept)
    unknown = [script for script in scripts if script not in swept]
    if unknown:
        message = (
            f"{threshold} is measured on {', '.join(swept)}, not {', '.join(unknown)}"
        )
        raise SystemExit(message)
    counts = arguments.threads or every(SWEPT_THREADS)
    started = time.perf_counter()
    timings: dict[str, Timing] = {}
    for script in scripts:
        sizes = arguments.sizes or swept[script]
        timings |= sweep(threshold, script, sizes, counts)
    print(f"\nmeasured in {duration(time.perf_counter() - started)}")
    output = arguments.output or results_path(Suite.SPEED)
    save(output, merged(output, timings))


def robustness(arguments: Bench) -> None:
    """Runs the corpus of hard circuits, and writes its outcomes."""
    output = arguments.output or results_path(Suite.ROBUSTNESS)
    corpus.measure(arguments.fixtures, arguments.simulator, arguments.timeout, output)


def coverage(arguments: Bench) -> None:
    """Runs the coverage matrix's fixtures, and renders it."""
    matrix.measure(arguments.fixtures, arguments.simulator)


def precision(arguments: Bench) -> suite.SuiteResults:
    """Runs the precision suite's fixtures, or draws a results file's diagrams.

    Returns the results it drew.
    """
    return suite.measure(
        arguments.fixtures,
        arguments.simulator,
        arguments.sweeps,
        arguments.repeats,
        arguments.output or results_path(Suite.PRECISION),
        arguments.plot,
        arguments.digits,
    )


def equal_error(arguments: Bench) -> None:
    """Prints each simulator's least time to equal error, from the precision suite.

    The suite runs as ``--precision`` runs it, or its results file `--plot` names is
    read, and draws the table beside its diagrams.
    """
    print(f"\n{suite.equal_error_table(precision(arguments), arguments.digits)}")


def instead(arguments: Bench) -> bool:
    """Runs the suite `arguments` name instead of the timing, if they name one.

    Returns whether they did: ``--robustness``, ``--coverage``, ``--precision``,
    ``--view equal-error`` or ``--break-even``.
    """
    if arguments.robustness:
        robustness(arguments)
    elif arguments.coverage:
        coverage(arguments)
    elif arguments.precision:
        precision(arguments)
    elif arguments.view is View.EQUAL_ERROR:
        equal_error(arguments)
    elif arguments.break_even is not None:
        break_evens(arguments, arguments.break_even)
    else:
        return False
    return True


def viewed(
    arguments: Bench,
    view: View,
    names: Sequence[str],
    found: Mapping[str, Fixture],
    output: Path,
) -> tuple[dict[str, Timing], dict[str, object]]:
    """Times the fixtures `names` in `view`, and writes its table beside `output`.

    Returns each configuration's times end to end and in its analyses, and each cell,
    by view, configuration and fixture.
    """
    cells = view_timings(
        view,
        names,
        found,
        arguments.simulator,
        parallel=arguments.xyce_mpi,
        ranks=arguments.threads or RANKS,
        scale=arguments.scale,
        timeout=arguments.timeout,
    )
    timings: dict[str, Timing] = {}
    views: dict[str, object] = {}
    for name, row in cells.items():
        for configuration, each in row.items():
            key = f"{view}/{configuration}/{name}"
            views[key] = dataclasses.asdict(each)
            if each.seconds is not None and each.analysis_seconds is not None:
                timings[key] = each.seconds
                timings[f"{key}/analysis"] = each.analysis_seconds
    table = view_table(cells)
    path = output.with_suffix("") / f"{view}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(table + "\n", encoding="utf-8")
    print(f"\n{table}\nwrote {shown(path)}")
    return timings, views


def measure(arguments: Bench) -> None:
    """Times the fixtures, writes the times, and compares them with a baseline's."""
    if instead(arguments):
        return
    if arguments.largest and not arguments.scale:
        message = "--largest runs a fixture of --scale; give --scale too"
        raise SystemExit(message)
    found = fixtures()
    names = arguments.fixtures or (SCALE if arguments.scale else SUITE)
    unknown = [name for name in names if name not in found]
    if unknown:
        message = (
            f"no fixture is named {', '.join(unknown)}; there are {', '.join(found)}"
        )
        raise SystemExit(message)
    output = arguments.output or results_path(Suite.SPEED)
    against = baseline(arguments.baseline)
    started = time.perf_counter()
    found_profiles: dict[str, dict[str, Profiled]] = {}
    fingerprints: dict[str, dict[str, str]] = {}
    views: dict[str, object] = {}
    if arguments.view is not None:
        timings, views = viewed(arguments, arguments.view, names, found, output)
    else:
        timings = {} if arguments.scale else rust(names, found)
        spread = {
            name
            for name in SCALE
            if arguments.scale
            and found[name].plan.monte_carlo is None
            and (name != LONGEST or arguments.largest)
        }
        counts = arguments.threads or every(THREADS)
        for name in names:
            if name in spread:
                measured, fingerprints[name] = scaling(
                    name, load_fixture(found[name]), counts
                )
                timings |= measured
            else:
                timings |= python([name], found, scale=arguments.scale)
        found_profiles = profiles(names, found)
    print(f"\nmeasured in {duration(time.perf_counter() - started)}")
    shares(found_profiles)
    profiled = {
        name: {
            analysis: dataclasses.asdict(each) for analysis, each in analyses.items()
        }
        for name, analyses in found_profiles.items()
    }
    save(output, merged(output, timings, profiled, fingerprints, views))
    if against is not None and against.resolve() != output.resolve():
        before = {
            key: Timing(**value) for key, value in load(against)["timings"].items()
        }
        compare(timings, before, against.name)


if __name__ == "__main__":
    main()
