# Copyright 2026 Maxence Faldor
"""Single-fixture execution and child-process timing worker for the speed suite.

Speed benchmarks load a circuit from JSON, elaborate the netlist, and run either every
plan analysis sequentially (`run`) or a Monte Carlo sample batch (`sampled`). Timing
wrappers record wall-clock time and total process CPU time across threads (`clocked`).

``scripts/bench --compare`` evaluates two jolty revisions in separate worktrees using
isolated worker processes running this module. Each worker prints the imported jolty
path on line 1 for verification, reads one JSON `Request` per stdin line, and emits a
JSON `Run` containing wall seconds, CPU seconds, and peak resident memory.

Requests carrying ``"prepare": true`` preload fixtures and emit ``{"ready": true}`` so
concurrent copies release together at a barrier. `probe` runs one request in a fresh
child process to measure clean peak resident memory. Simulation output is redirected to
stderr so stdout carries only protocol JSON.
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

# Add shared benchmark helpers and speed circuit scripts to the module search path.
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
    """Imports and returns all fixture scripts in ``benchmarks/speed/circuits``."""
    return [
        importlib.import_module(path.stem)
        for path in sorted(CIRCUITS.glob("*.py"))
        if path.stem != "fixture"
    ]


def load_fixture(fixture: Fixture) -> Loaded:
    """Loads the circuit JSON for `fixture` and pairs it with `fixture.plan`."""
    text = (CIRCUITS / f"{fixture.circuit}.json").read_text(encoding="utf-8")
    return Loaded(ir.Circuit.model_validate_json(text), fixture.plan)


def serially(loaded: Loaded) -> Options:
    """Returns serial execution options for `loaded` with matrix dissection disabled.

    Restricts execution to a single thread and sets `max_unknowns_per_interior` to the
    full circuit unknown count so no matrix block is dissected.
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
    """Elaborates `loaded.circuit` and runs all plan analyses in jolty.

    Uses `loaded.plan.options` when `options` is None, returning results keyed by name.
    """
    chosen = loaded.plan.options if options is None else options
    return Jolty().run(loaded.circuit, loaded.plan.analyses, chosen).found


def sampled(loaded: Loaded, samples: int, threads: int) -> Callable[[], None]:
    """Builds a callable executing `samples` Monte Carlo draws of `loaded` on `threads`.

    Elaborates the circuit once per invocation and executes one parallel engine batch
    across `threads` worker threads, matching runner behavior and including elaboration.
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
    """Measured wall time, multi-thread process CPU time, and optional peak memory.

    CPU time sums across all worker threads and can exceed wall time on parallel runs.
    `peak_memory_bytes` holds the high-water mark of an isolated child process, or None
    when shared across distinct fixtures.
    """

    wall: float
    cpu: float
    peak_memory_bytes: int | None = None


def clocked(work: Callable[[], object]) -> Callable[[], Run]:
    """Wraps `work` in a callable measuring wall and process CPU seconds as a `Run`."""

    def measured() -> Run:
        wall, cpu = time.perf_counter(), time.process_time()
        work()
        return Run(time.perf_counter() - wall, time.process_time() - cpu)

    return measured


@dataclass(frozen=True, slots=True)
class Request:
    """Timing request sent to a worker process for a named speed fixture.

    Monte Carlo requests execute the first `samples` draws across `threads` threads via
    `sampled` and report per-sample seconds. Non-Monte-Carlo requests run all analyses
    once via `run`, optionally constraining `threads` or forcing `serial` execution.
    """

    fixture: str
    samples: int | None = None
    threads: int | None = None
    serial: bool = False


def planned() -> dict[str, Fixture]:
    """Collects all defined speed fixtures across circuit scripts keyed by name."""
    return {
        name: fixture
        for script in scripts()
        for name, fixture in script.fixtures().items()
    }


def work(request: Request, fixtures: dict[str, Fixture]) -> Callable[[], Run]:
    """Builds a timed callable for `request`, normalizing Monte Carlo per sample."""
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
