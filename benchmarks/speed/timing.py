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
fixture once and writes its wall and processor seconds (`Run`) as a line of JSON. Its
first line says which jolty it imported, so the caller can check that it is the build's.
What jolty or a simulation prints goes to the standard error, so that only the answers
reach the standard output.
"""

import functools
import importlib
import json
import os
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
    """One run's seconds: by the wall clock, and of the processor.

    The processor's are those of every thread of the process, so a run spread over
    threads takes more of them than of the wall clock.
    """

    wall: float
    cpu: float


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
    once, as `run` runs them.
    """

    fixture: str
    samples: int | None = None
    threads: int | None = None


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
        return clocked(functools.partial(run, loaded))
    every = clocked(sampled(loaded, request.samples, request.threads or 1))
    samples = request.samples

    def per_sample() -> Run:
        whole = every()
        return Run(whole.wall / samples, whole.cpu / samples)

    return per_sample


def serve() -> None:
    """Runs each request its standard input holds, and answers each on its output.

    Every run of one request reuses the fixture as first loaded, as a timing in one
    process does.
    """
    answers = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    answers.write(json.dumps({"jolty": jolty.__file__}) + "\n")
    answers.flush()
    fixtures = planned()
    works: dict[Request, Callable[[], Run]] = {}
    for line in sys.stdin:
        request = Request(**json.loads(line))
        if request not in works:
            works[request] = work(request, fixtures)
        answers.write(json.dumps(asdict(works[request]())) + "\n")
        answers.flush()


if __name__ == "__main__":
    serve()
