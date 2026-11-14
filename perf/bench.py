# Copyright 2026 Maxence Faldor
"""Time jolty on the fixtures of ``perf/``, and compare the times with a baseline's.

``scripts/bench`` runs this script from the repository root. It first checks that every
fixture's files are what its script in ``perf/circuits`` writes, then times the suite's
fixtures twice: in Rust, by the criterion benchmarks of
``crates/jolty-engine/benches/fixtures.rs``, and end to end in Python, each fixture's
circuit read from its file and its plan's analyses run through ``Simulation.run``. A
Monte Carlo fixture runs its first samples as the runner does, as one batch of the
circuit elaborated once, on one thread and on a thread per core. It writes the times
to ``perf/results/<commit>.json``, adding to what an earlier run on the same commit
wrote, and prints them against a baseline's, flagging every change over 5 %.

Usage::

    scripts/bench [FIXTURE ...]       the suite's fixtures, or those named
    scripts/bench --scale             the large fixtures, each run once in Python
    scripts/bench --ngspice           ngspice on the fixtures' netlists, for context
    scripts/bench --baseline COMMIT   against COMMIT's results, or a results file's

Without ``--baseline``, the baseline is the newest ancestor of the checked-out commit
whose results ``perf/results`` holds. A working tree with changes to tracked files
writes ``<commit>-dirty.json``, which is never taken as a baseline.
"""

import dataclasses
import functools
import importlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Final

from pydantic import Field
from pydantic_settings import CliPositionalArg, SettingsConfigDict

from jolty import circuit, cli, ir, results, simulate
from jolty.plan import Ac, MonteCarlo, OperatingPoint, Plan, Transient
from jolty.simulate import Elaborated, Simulation
from jolty.spice import export, ngspice
from jolty.units import FREQUENCY, TIME

PERF: Final = Path(__file__).resolve().parent
"""The ``perf`` directory."""

ROOT: Final = PERF.parent
"""The repository's root."""

sys.path.insert(0, str(PERF / "circuits"))
from fixture import CIRCUITS, PLANS, Fixture, engine_plan  # noqa: E402 - beside the scripts

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

SAMPLES: Final = 1000
"""The samples of a Monte Carlo fixture the suite times: a few seconds on one thread."""

FLAGGED: Final = 0.05
"""The relative change in a time that the comparison flags, as AGENTS.md says."""

FEWEST_RUNS: Final = 10
"""The fewest times the Python timing repeats a fixture: criterion's fewest samples."""

FEWEST_SECONDS: Final = 5.0
"""The least time the Python timing repeats a fixture for, as long as criterion
measures each benchmark."""


@dataclass(frozen=True, slots=True)
class Timing:
    """A time measured: the median of its runs, in seconds, and their number."""

    seconds: float
    runs: int


@dataclass(frozen=True, slots=True)
class Loaded:
    """A fixture as the timing runs it: its circuit read from its file, and its plan."""

    circuit: ir.Circuit
    plan: Plan


def scripts() -> list[ModuleType]:
    """The fixtures' scripts in ``perf/circuits``, in the order of their names."""
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


def load(fixture: Fixture) -> Loaded:
    """The fixture's circuit, read from its file, and its plan."""
    text = (CIRCUITS / f"{fixture.circuit}.json").read_text(encoding="utf-8")
    return Loaded(ir.Circuit.model_validate_json(text), fixture.plan)


def run(loaded: Loaded) -> None:
    """Elaborates the circuit, and runs the plan's analyses on it."""
    simulation = Simulation(loaded.circuit)
    for analysis in loaded.plan.analyses.values():
        simulation.run(analysis, loaded.plan.options)


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
        simulation = Simulation.of(Elaborated(loaded.circuit))
        with simulate.threads(threads):
            simulation.batch(
                loaded.plan.analyses, MonteCarlo(samples, seed), loaded.plan.options
            )

    return every


def timed(work: Callable[[], object], *, once: bool = False) -> Timing:
    """The median time of `work`, once warmed up, over enough runs; or of one run."""
    if once:
        started = time.perf_counter()
        work()
        return Timing(time.perf_counter() - started, 1)
    work()
    times: list[float] = []
    begun = time.perf_counter()
    while len(times) < FEWEST_RUNS or time.perf_counter() - begun < FEWEST_SECONDS:
        started = time.perf_counter()
        work()
        times.append(time.perf_counter() - started)
    return Timing(statistics.median(times), len(times))


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
        loaded = load(found[name])
        monte_carlo = loaded.plan.monte_carlo
        print(f"timing {name} in Python", flush=True)
        if monte_carlo is None:
            timings[f"python/{name}"] = timed(
                functools.partial(run, loaded), once=scale
            )
            continue
        samples = monte_carlo.samples if scale else min(SAMPLES, monte_carlo.samples)
        threads = (
            ((cores, "every core"),)
            if scale
            else ((1, "one core"), (cores, "every core"))
        )
        for count, label in threads:
            whole = timed(sampled(loaded, samples, count), once=scale)
            timings[f"python/{name}/{samples} samples, {label}"] = Timing(
                whole.seconds / samples, whole.runs
            )
    return timings


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
        loaded = load(found[name])
        if loaded.plan.monte_carlo is not None:
            continue
        print(f"profiling {name}", flush=True)
        simulation = Simulation(loaded.circuit)
        options = dataclasses.replace(loaded.plan.options, profile=True)
        for analysis, planned in loaded.plan.analyses.items():
            result = simulation.run(planned, options)
            if not isinstance(result, results.Waveforms):
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


def deck(loaded: Loaded) -> list[str]:
    """The fixture as an ngspice deck: its netlist, then its analyses as commands.

    Raises:
        TypeError: When the plan runs an analysis other than an operating point, a
            small-signal response or a transient.
    """
    analyses = list(loaded.plan.analyses.values())
    excited = next((each.source for each in analyses if isinstance(each, Ac)), None)
    transient = any(isinstance(each, Transient) for each in analyses)
    lines = export.netlist(loaded.circuit, excited=excited, transient=transient)
    commands: list[str] = []
    for analysis in analyses:
        match analysis:
            case OperatingPoint():
                commands.append("op")
            case Ac(frequencies=frequencies):
                first = frequencies[0].to(FREQUENCY, "a frequency")
                last = frequencies[-1].to(FREQUENCY, "a frequency")
                per_decade = round((len(frequencies) - 1) / math.log10(last / first))
                commands.append(f"ac dec {per_decade} {first!r} {last!r}")
            case Transient(stop=stop, max_step=max_step):
                end = stop.to(TIME, "a stop")
                longest = end / 50 if max_step is None else max_step.to(TIME, "a step")
                commands.append(f"tran {longest!r} {end!r} 0 {longest!r}")
            case _:
                message = f"no fixture runs {type(analysis).__name__} in ngspice"
                raise TypeError(message)
    return [*lines, ".control", *commands, ".endc", ".end"]


def ngspice_run(executable: str, path: Path, analyses: int) -> None:
    """Runs ngspice on the deck at `path`, checking that it ran its `analyses`.

    Raises:
        SystemExit: When ngspice fails, or finishes fewer analyses.
    """
    done = subprocess.run(  # noqa: S603 - ngspice runs the deck written here
        [executable, "-b", str(path)], capture_output=True, check=False, text=True
    )
    printed = done.stdout + done.stderr
    if done.returncode != 0 or printed.count("No. of Data Rows") != analyses:
        message = f"ngspice did not run every analysis of {path.name}:\n{printed}"
        raise SystemExit(message)


def spice(
    names: Iterable[str], found: Mapping[str, Fixture], *, scale: bool
) -> dict[str, Timing]:
    """The times of ngspice, from its start to its exit, on the fixtures `names`.

    Each runs once if at `scale`; a Monte Carlo fixture is not timed.

    Raises:
        SystemExit: When ngspice is not installed.
    """
    if ngspice.EXECUTABLE is None:
        message = "ngspice is not installed"
        raise SystemExit(message)
    timings: dict[str, Timing] = {}
    for name in names:
        loaded = load(found[name])
        if loaded.plan.monte_carlo is not None:
            continue
        print(f"timing {name} in ngspice", flush=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"{name}.cir"
            path.write_text("\n".join(deck(loaded)) + "\n", encoding="utf-8")
            analyses = len(loaded.plan.analyses)
            work = functools.partial(ngspice_run, ngspice.EXECUTABLE, path, analyses)
            timings[f"ngspice/{name}"] = timed(work, once=scale)
    return timings


def git(*arguments: str) -> str:
    """What git prints for `arguments`, in the repository."""
    done = subprocess.run(  # noqa: S603 - git reads the repository
        ["git", *arguments],  # noqa: S607 - git is on the path
        capture_output=True,
        check=True,
        cwd=ROOT,
        text=True,
    )
    return done.stdout.strip()


def sysctl(name: str) -> str | None:
    """The value of the macOS kernel setting `name`, or None elsewhere."""
    try:
        done = subprocess.run(  # noqa: S603 - sysctl reads the machine's settings
            ["sysctl", "-n", name],  # noqa: S607 - sysctl is on the path on macOS
            capture_output=True,
            check=True,
            text=True,
        )
    except OSError, subprocess.CalledProcessError:
        return None
    return done.stdout.strip()


def machine() -> dict[str, object]:
    """The machine the times are measured on: its processor, memory and system."""
    cpu = sysctl("machdep.cpu.brand_string")
    sockets = sysctl("hw.packages")
    memory = sysctl("hw.memsize")
    # macOS names each kind of core, such as Apple silicon's performance and efficiency
    # cores, and counts them.
    kinds = int(sysctl("hw.nperflevels") or 0)
    levels = (f"hw.perflevel{level}" for level in range(kinds))
    cores = {
        sysctl(f"{level}.name"): sysctl(f"{level}.physicalcpu") for level in levels
    }
    information = Path("/proc/cpuinfo")
    if cpu is None and information.is_file():
        lines = information.read_text(encoding="utf-8").splitlines()
        names = [
            line.split(":", 1)[1].strip()
            for line in lines
            if line.startswith("model name")
        ]
        cpu = names[0] if names else None
        packages = {line for line in lines if line.startswith("physical id")}
        sockets = str(len(packages)) if packages else None
    totals = Path("/proc/meminfo")
    if memory is None and totals.is_file():
        first = totals.read_text(encoding="utf-8").splitlines()[0]
        memory = str(int(first.split()[1]) * 1024)
    rustc = subprocess.run(
        ["rustc", "--version"],  # noqa: S607 - rustc is on the path
        capture_output=True,
        check=True,
        cwd=ROOT,
        text=True,
    )
    return {
        "cpu": cpu or platform.processor(),
        "cores": os.cpu_count(),
        "kinds of core": {str(kind): int(count or 0) for kind, count in cores.items()},
        "sockets": None if sockets is None else int(sockets),
        "memory": None if memory is None else int(memory),
        "system": platform.platform(),
        "python": platform.python_version(),
        "rustc": rustc.stdout.strip(),
    }


def results_path() -> Path:
    """Where the times of the checked-out commit go: ``-dirty`` for a changed tree."""
    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=no") != ""
    return PERF / "results" / f"{commit}{'-dirty' if dirty else ''}.json"


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
        found = PERF / "results" / f"{git('rev-parse', given)}.json"
        if found.is_file():
            return found
        message = f"no results of {given}; run scripts/bench on it first"
        raise SystemExit(message)
    for commit in git("rev-list", "--max-count=1000", "HEAD").split()[1:]:
        found = PERF / "results" / f"{commit}.json"
        if found.is_file():
            return found
    return None


def duration(seconds: float) -> str:
    """`seconds` with an SI prefix, to four figures."""
    for factor, unit in ((1.0, "s"), (1e-3, "ms"), (1e-6, "µs")):
        if seconds >= factor:
            return f"{seconds / factor:.4g} {unit}"
    return f"{seconds / 1e-9:.4g} ns"


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


def read(path: Path) -> dict[str, Timing]:
    """The times a results file holds."""
    document = json.loads(path.read_text(encoding="utf-8"))
    return {key: Timing(**value) for key, value in document["timings"].items()}


class Bench(cli.CommandLine):
    """Time the fixtures, write the times, and compare them with a baseline's."""

    model_config = SettingsConfigDict(cli_prog_name="scripts/bench")

    fixtures: CliPositionalArg[list[str]] = Field(
        default_factory=list,
        description="The fixtures to time; the suite's by default.",
    )
    scale: bool = Field(
        default=False, description="Time the large fixtures, once each."
    )
    ngspice: bool = Field(
        default=False, description="Time ngspice instead, for context."
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


def measure(arguments: Bench) -> None:
    """Times the fixtures, writes the times, and compares them with a baseline's."""
    found = fixtures()
    names = arguments.fixtures or (SCALE if arguments.scale else SUITE)
    unknown = [name for name in names if name not in found]
    if unknown:
        message = (
            f"no fixture is named {', '.join(unknown)}; there are {', '.join(found)}"
        )
        raise SystemExit(message)
    output = arguments.output or results_path()
    against = baseline(arguments.baseline)
    started = time.perf_counter()
    found_profiles: dict[str, dict[str, Profiled]] = {}
    if arguments.ngspice:
        timings = spice(names, found, scale=arguments.scale)
    else:
        timings = {} if arguments.scale else rust(names, found)
        timings |= python(names, found, scale=arguments.scale)
        found_profiles = profiles(names, found)
    print(f"\nmeasured in {duration(time.perf_counter() - started)}")
    shares(found_profiles)
    earlier = json.loads(output.read_text("utf-8")) if output.is_file() else {}
    written = {**earlier.get("timings", {})}
    written |= {
        key: {"seconds": each.seconds, "runs": each.runs}
        for key, each in timings.items()
    }
    document = {
        "commit": git("rev-parse", "HEAD"),
        "machine": machine(),
        "timings": written,
        "profiles": earlier.get("profiles", {})
        | {
            name: {
                analysis: dataclasses.asdict(profiled)
                for analysis, profiled in analyses.items()
            }
            for name, analyses in found_profiles.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {output.relative_to(ROOT) if output.is_relative_to(ROOT) else output}"
    )
    if against is not None and against.resolve() != output.resolve():
        compare(timings, read(against), against.name)


if __name__ == "__main__":
    main()
