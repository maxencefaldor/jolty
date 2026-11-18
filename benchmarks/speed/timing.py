# Copyright 2026 Maxence Faldor
"""One run of a fixture, timed by the wall clock and by the processor.

The speed suite's timings run each fixture as this module does: its circuit read from
its file, elaborated, and its plan's analyses run in turn (`run`), or the first samples
of its Monte Carlo analysis run as one batch (`sampled`), each run timed by the wall
clock and by the processor time of every thread of the process (`clocked`).

``scripts/bench --compare`` times two builds of jolty, each in a worktree of its own,
so each build's runs take place in a process of its own: the build's interpreter runs
this module, which imports the build's jolty and reads requests from its standard
input, one JSON object a line, each a fixture and how to run it (`Request`); it runs the
fixture once and writes its wall and processor seconds and peak resident memory (`Run`)
as a line of JSON. Its first line says which jolty it imported, so the caller can check
that it is the build's. A request with ``"prepare": true`` loads the fixture without
running it and replies ``{"ready": true}``, so concurrent copies can start together at a
barrier. `probe` runs one request in a fresh child process to read an isolated fixture's
peak resident memory without inheriting another fixture's high-water mark. What jolty
or a simulation prints goes to the standard error, so that only the answers reach the
standard output.
"""

import dataclasses
import functools
import importlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

# The suites' runners lie beside the suites' directories, and the fixtures' scripts in
# the speed suite's circuits.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "circuits"))
from fixture import CIRCUITS, Fixture
from measuring import ROOT, peak_memory_bytes
from simulators import Jolty

import jolty
from jolty import ir, results, simulate
from jolty.plan import MonteCarlo, Options, Plan
from jolty.simulate import Elaborated, Simulation


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


def load_fixture(fixture: Fixture) -> Loaded:
    """The fixture's circuit, read from its file, and its plan."""
    text = (CIRCUITS / f"{fixture.circuit}.json").read_text(encoding="utf-8")
    return Loaded(ir.Circuit.model_validate_json(text), fixture.plan)


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


@dataclass(frozen=True, slots=True)
class Run:
    """One run's seconds, by the wall clock and of the processor, and peak memory.

    The processor's are those of every thread of the process, so a run spread over
    threads takes more of them than of the wall clock. `peak_memory_bytes` is the
    resident high-water mark of an isolated child process that ran this fixture alone,
    or None when measured in a shared process.
    """

    wall: float
    cpu: float
    peak_memory_bytes: int | None = None


def clocked(work: Callable[[], object]) -> Callable[[], Run]:
    """`work`, returning its `Run`'s seconds."""

    def measured() -> Run:
        wall, cpu = time.perf_counter(), time.process_time()
        work()
        return Run(time.perf_counter() - wall, time.process_time() - cpu)

    return measured


@dataclass(frozen=True, slots=True)
class Request:
    """A run of a fixture asked of a build: the fixture's, or its first `samples`.

    A Monte Carlo fixture runs its first `samples` on a pool of `threads` threads, as
    `sampled` runs them, and its seconds are per sample; any other runs its analyses
    once, as `run` runs them, optionally restricted to `threads` threads or run
    `serial` (on one thread with no block of its matrix dissected).
    """

    fixture: str
    samples: int | None = None
    threads: int | None = None
    serial: bool = False


def planned() -> dict[str, Fixture]:
    """Every fixture, by name, as its script defines it."""
    return {
        name: fixture
        for script in scripts()
        for name, fixture in script.fixtures().items()
    }


def work(request: Request, fixtures: dict[str, Fixture]) -> Callable[[], Run]:
    """The run `request` asks, clocked; per sample for a Monte Carlo fixture's."""
    loaded = load_fixture(fixtures[request.fixture])
    if request.samples is None:
        if request.serial:
            options = serially(loaded)

            def serial_run() -> None:
                with simulate.threads(1):
                    run(loaded, options)

            return clocked(serial_run)
        if request.threads is not None:
            thread_count = request.threads

            def threaded_run() -> None:
                with simulate.threads(thread_count):
                    run(loaded)

            return clocked(threaded_run)
        return clocked(functools.partial(run, loaded))
    thread_count = 1 if request.serial else (request.threads or 1)
    every = clocked(sampled(loaded, request.samples, thread_count))
    samples = request.samples

    def per_sample() -> Run:
        whole = every()
        return Run(whole.wall / samples, whole.cpu / samples)

    return per_sample


def probe(request: Request, *, python_executable: str = sys.executable) -> Run:
    """Runs `request` once in a fresh `timing.py` child process.

    Running in a fresh child process ensures `peak_memory_bytes` reflects this
    fixture alone rather than a shared process's monotonic high-water mark.

    Raises:
        RuntimeError: When the child process exits with an error or returns no run.
    """
    script = Path(__file__).resolve()
    payload = json.dumps(asdict(request)) + "\n"
    done = subprocess.run(  # noqa: S603 - runs this module with the active interpreter
        [python_executable, str(script)],
        input=payload,
        capture_output=True,
        check=False,
        cwd=ROOT,
        text=True,
    )
    if done.returncode != 0:
        message = (
            f"timing probe for {request.fixture!r} exited with {done.returncode}: "
            f"{done.stderr.strip()}"
        )
        raise RuntimeError(message)
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        message = (
            f"timing probe for {request.fixture!r} returned no result: "
            f"{done.stderr.strip()}"
        )
        raise RuntimeError(message)
    return Run(**json.loads(lines[-1]))


def serve() -> None:
    """Runs each request its standard input holds, and answers each on its output.

    Every run of one request reuses the fixture as first loaded, as a timing in one
    process does. When only a single distinct request has run in this process, each
    answer also carries `peak_memory_bytes`; once a second distinct request runs,
    `peak_memory_bytes` is reported as None so an earlier fixture's high-water mark
    cannot be attributed to another.
    """
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8") as answers:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        answers.write(json.dumps({"jolty": jolty.__file__}) + "\n")
        answers.flush()
        fixtures = planned()
        works: dict[Request, Callable[[], Run]] = {}
        for line in sys.stdin:
            if not line.strip():
                continue
            payload = json.loads(line)
            prepare_only = bool(payload.pop("prepare", False))
            request = Request(**payload)
            if request not in works:
                works[request] = work(request, fixtures)
            if prepare_only:
                answers.write(json.dumps({"ready": True}) + "\n")
                answers.flush()
                continue
            measured = works[request]()
            isolated = len(works) == 1
            answered = Run(
                measured.wall,
                measured.cpu,
                peak_memory_bytes() if isolated else None,
            )
            answers.write(json.dumps(asdict(answered)) + "\n")
            answers.flush()


if __name__ == "__main__":
    serve()
