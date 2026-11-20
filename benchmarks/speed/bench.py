# Copyright 2026 Maxence Faldor
"""Unified CLI runner for speed, scaling, comparison, robustness, coverage, precision.

``scripts/bench`` invokes this entry point from the repository root. Default mode first
verifies that generated JSON files match ``benchmarks/speed/circuits`` scripts. Timing
then runs both Rust Criterion benchmarks (``crates/jolty-engine/benches/fixtures.rs``)
and end-to-end Python runs through ``Simulation.run``. Monte Carlo fixtures run one
elaborated batch on one core and on all cores.

Results merge into ``benchmarks/speed/results/<commit>.json`` and print alongside a
baseline with 95 % confidence intervals. Changes are flagged only when intervals
separate.

Usage:

```text
scripts/bench [FIXTURE ...]       default speed fixtures, or named fixtures
scripts/bench --compare A,B       alternating build comparison of revisions A and B
scripts/bench --scale             large fixtures serially and across thread counts
scripts/bench --scale --largest   include the 50,000-inverter chain across thread counts
scripts/bench --view matched      all simulators at matched tight tolerances
scripts/bench --view defaults     all simulators at default tolerances
scripts/bench --view equal-error  minimum simulator time at equal precision accuracy
scripts/bench --baseline COMMIT   compare against COMMIT or a results file
scripts/bench --break-even NAME   measure Parallelism break-even for threshold NAME
scripts/bench --robustness        hard-circuit robustness corpus across simulators
scripts/bench --coverage          capability coverage matrix across simulators
scripts/bench --precision         work-precision sweeps against exact references
```

Without ``--baseline``, the suite selects the newest ancestor commit under
``benchmarks/speed/results``. Dirty working trees write ``<commit>-dirty.json``, which
is never selected as a baseline. For controlled A/B decisions, ``--compare`` builds each
revision in a dedicated worktree under ``--builds`` and executes `afresh` worker
processes (`benchmarks/speed/timing.py`) in `alternated` rounds. Comparison tables
report wall and CPU times separately alongside JSON results.

``--scale`` benchmarks large fixtures serially (one thread, no matrix dissection) and
across 1, 2, 4, 8, and all cores under default ``Parallelism`` in `alternated` rounds.
Output reports speed-ups, attainable throughput against N concurrent single-thread
copies, isolated peak resident memory, and result digests. The 50,000-inverter chain
sweeps thread counts only with ``--largest``. Pass ``--no-concurrent-copies`` or
``--no-peak-memory`` to skip either probe.

``--view`` benchmarks installed simulators (jolty, ngspice Sparse 1.3, ngspice KLU, and
Xyce, or ``--simulator`` choices) across three views. ``matched`` aligns tight options
(`jolty.plan.TIGHT`, `jolty.spice.ngspice.OPTIONS`, `jolty.spice.xyce.OPTIONS`), while
``defaults`` uses simulator defaults. Both report end-to-end time, analysis time, and
maximum voltage deviation from matched jolty. Combining ``--scale`` with ``--xyce-mpi``
adds Xyce parallel MPI ranks (1, 2, 4, 8, 16). ``equal-error`` delegates to the
precision suite and prints minimum times at equal digit accuracy.

``--break-even`` sweeps circuit sizes across 2, 4, 8, and all cores to find where
enabling one ``Parallelism`` threshold beats serial execution. Each size runs serially
and with only the selected threshold active in `alternated` rounds until intervals
settle. See ``benchmarks/speed/README.md`` for threshold methodology and findings.

Sub-suite flags dispatch directly to sibling suites: ``--robustness`` runs the hard
circuit corpus (``benchmarks/robustness/README.md``), ``--coverage`` builds the
capability matrix (``benchmarks/coverage/README.md``), and ``--precision`` runs
work-precision sweeps and plots (``benchmarks/precision/README.md``). All suites share
runners from ``benchmarks/simulators.py`` and provenance from
``benchmarks/measuring.py``.
"""

import concurrent.futures
import contextlib
import dataclasses
import functools
import hashlib
import itertools
import json
import math
import mmap
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import IO, Any, Final

import numpy as np
import numpy.typing as npt

# Shared benchmark helper modules live in the parent benchmarks directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from measuring import (
    BENCHMARKS,
    CONFIDENCE,
    ROOT,
    Median,
    Provenance,
    Suite,
    aligned_row,
    aligned_table,
    available_memory_bytes,
    duration,
    format_json,
    git,
    load,
    load_average,
    markdown_table,
    median,
    memory,
    provenance,
    rank,
    results_directory,
    results_path,
    save,
    shown,
    usable_cores,
)
from pydantic import BaseModel, Field, JsonValue
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
from timing import (
    Loaded,
    Request,
    Run,
    clocked,
    load_fixture,
    planned,
    probe,
    run,
    scripts,
    serially,
    work,
)

from jolty import circuit, cli, ir, plan, results, simulate
from jolty.errors import JoltyError
from jolty.plan import Ac, Analysis, Options, Parallelism, Transient
from jolty.simulate import Simulation
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
from fixture import CIRCUITS, PLANS, SPEED, Fixture, engine_plan

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
"""Target relative timing change that alternating comparisons must resolve via
`Median.resolution`, matching AGENTS.md."""

MIN_RUNS: Final = 10
"""The fewest times the Python timing repeats a fixture: criterion's fewest samples."""

MIN_SECONDS: Final = 5.0
"""The least time the Python timing repeats a fixture for, as long as criterion
measures each benchmark."""

MIN_ROUNDS: Final = next(
    count for count in itertools.count(1) if rank(count) is not None
)
"""The fewest rounds of compared works: the fewest runs whose median has an interval
at `CONFIDENCE`, 6 at 95%, as `measuring.rank` derives."""

MAX_ROUNDS: Final = 4 * MIN_ROUNDS
"""Default maximum alternating rounds once each work meets `MIN_SECONDS`, overridable
via ``--most-rounds``. Running four times `MIN_ROUNDS` roughly halves interval width
before reporting any unresolved comparison as undecided."""

COPY_ROUNDS: Final = 3
"""The default synchronized rounds of concurrent single-thread copies.

Taking the median of three rounds filters single-round scheduling noise without
inflating ``--scale`` wall time as full confidence-interval rounds would.
"""

MEMORY_HEADROOM: Final = 0.85
"""The fraction of available machine memory concurrent copies may budget.

Leaves room for the operating system, the parent process, and transient allocator
peaks so large fixtures (such as the 50,000-inverter chain, ~2 to 4 GiB per copy)
never trigger an out-of-memory kill when launching many copies at once.
"""

HYBRID_SPREAD_THRESHOLD: Final = 1.25
"""The max-to-min copy time ratio above which the scaling table notes core
heterogeneity (such as performance and efficiency cores)."""

_BARRIER_TIMEOUT_SECONDS: Final = 60.0
"""Upper bound in seconds waiting for worker threads to meet at the launch barrier."""

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
"""Default circuit scripts and sizes swept per `Threshold` in ``--break-even``. Sizes
count inverters, mesh side length, or AC points per decade near break-evens."""


@dataclass(frozen=True, slots=True)
class Timing:
    """Aggregated benchmark timing storing median wall seconds and sample count.

    Optional fields store starting and ending host load averages, per-run wall and CPU
    timings used for confidence intervals, and isolated child-process peak resident
    memory in bytes. Earlier committed results files may omit optional fields.
    """

    seconds: float
    runs: int
    load: float | None = None
    load_after: float | None = None
    wall: tuple[float, ...] = ()
    cpu: tuple[float, ...] = ()
    peak_memory_bytes: int | None = None

    @classmethod
    def from_samples(
        cls,
        runs: Sequence[Run],
        load: float,
        load_after: float,
        peak_memory_bytes: int | None = None,
    ) -> Timing:
        """Builds a `Timing` from `runs` recorded between `load` and `load_after`.

        Takes the maximum peak resident memory observed across `runs` when
        `peak_memory_bytes` is None.
        """
        wall = tuple(each.wall for each in runs)
        cpu = tuple(each.cpu for each in runs)
        if peak_memory_bytes is None:
            measured_memory = [
                each.peak_memory_bytes
                for each in runs
                if each.peak_memory_bytes is not None
            ]
            peak_memory_bytes = max(measured_memory) if measured_memory else None
        return cls(
            statistics.median(wall),
            len(wall),
            load,
            load_after,
            wall,
            cpu,
            peak_memory_bytes,
        )

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Timing:
        """Deserializes a `Timing` from a results dictionary created by `to_json`."""
        return cls(
            value["seconds"],
            value["runs"],
            value.get("load"),
            value.get("load_after"),
            tuple(value.get("wall", ())),
            tuple(value.get("cpu", ())),
            value.get("peak_memory_bytes"),
        )

    def to_json(self) -> dict[str, object]:
        """Serializes this `Timing` to a dict, omitting unmeasured optional fields."""
        written = dataclasses.asdict(self)
        return {
            key: value
            for key, value in written.items()
            if value is not None and value != ()
        }

    def wall_median(self) -> Median:
        """Returns the median wall time with confidence bounds when samples exist."""
        return median(self.wall) if self.wall else Median(self.seconds)

    def cpu_median(self) -> Median | None:
        """Returns the median CPU time with confidence bounds when CPU samples exist."""
        return median(self.cpu) if self.cpu else None


class TimingRecord(BaseModel):
    """Serialized timing entry in a speed suite results file."""

    seconds: float
    runs: int
    load: float | None = None
    load_after: float | None = None
    wall: list[float] = Field(default_factory=list)
    cpu: list[float] = Field(default_factory=list)
    peak_memory_bytes: int | None = Field(default=None, ge=0)

    def to_timing(self) -> Timing:
        """Converts this record into a `Timing` dataclass."""
        return Timing(
            self.seconds,
            self.runs,
            self.load,
            self.load_after,
            tuple(self.wall),
            tuple(self.cpu),
            self.peak_memory_bytes,
        )


class ConcurrentCopies(BaseModel):
    """Throughput record for `copies` independent single-thread copies run concurrently.

    Each copy executes in an isolated `timing.py` worker process with `max_threads=1`
    and no matrix dissection. `wall_seconds` holds the median makespan across `rounds`
    synchronized rounds, while `copy_seconds` holds sorted per-copy wall times from the
    representative median round. Sets `wall_seconds` to None with `skipped_reason` when
    estimated memory exceeds the host budget.
    """

    copies: int = Field(ge=1)
    rounds: int = Field(default=0, ge=0)
    wall_seconds: float | None = Field(default=None, gt=0.0)
    round_seconds: list[float] = Field(default_factory=list)
    copy_seconds: list[float] = Field(default_factory=list)
    per_copy_peak_memory_bytes: int | None = Field(default=None, ge=0)
    load: float | None = None
    load_after: float | None = None
    skipped_reason: str | None = None

    def wall_median(self) -> Median | None:
        """Returns the median makespan with confidence bounds when measured, or None."""
        if self.round_seconds:
            return median(self.round_seconds)
        if self.wall_seconds is not None:
            return Median(self.wall_seconds)
        return None


class ThroughputEfficiency(BaseModel):
    """Parallel efficiency of one N-thread run against N concurrent 1-thread copies.

    Given serial time ``T_1``, N-thread time ``T_par``, and concurrent makespan
    ``T_copies`` for ``N`` copies:

    - ``speedup = T_1 / T_par``
    - ``attainable_speedup = N * T_1 / T_copies``
    - ``efficiency = speedup / attainable_speedup = T_copies / (N * T_par)``

    Efficiency is never clamped to 1.0: concurrent copies of a memory-bound fixture
    may contend for memory bandwidth more than a single shared run does, yielding
    efficiency above 100%, which is reported as found.

    When individual `copy_seconds` are provided for all ``N`` copies, also records
    `rate_sum_attainable_speedup = T_1 * sum(1 / t_i)`, `rate_sum_efficiency`, and
    `copy_spread_ratio = max(t_i) / min(t_i)` to surface hybrid-core heterogeneity
    where slow efficiency cores stretch batch makespan.
    """

    threads: int = Field(ge=1)
    speedup: float
    attainable_speedup: float
    efficiency: float
    rate_sum_attainable_speedup: float | None = None
    rate_sum_efficiency: float | None = None
    copy_spread_ratio: float | None = None


class SpeedResults(Provenance):
    """A speed suite results file, compatible with earlier committed baselines.

    Every field added after the initial baseline (`digests`, `views`, `copies`,
    and `TimingRecord.peak_memory_bytes`) defaults to empty or None so committed
    results files from earlier commits validate without migration.
    """

    timings: dict[str, TimingRecord] = Field(default_factory=dict)
    profiles: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    digests: dict[str, dict[str, str]] = Field(default_factory=dict)
    views: dict[str, JsonValue] = Field(default_factory=dict)
    copies: dict[str, dict[str, ConcurrentCopies]] = Field(default_factory=dict)


def load_results(path: Path) -> SpeedResults:
    """Loads and validates a speed suite results file from `path`."""
    return SpeedResults.model_validate_json(path.read_text(encoding="utf-8"))


def throughput_efficiency(
    *,
    threads: int,
    serial_seconds: float,
    parallel_seconds: float,
    copies_wall_seconds: float,
    copy_seconds: Sequence[float] = (),
) -> ThroughputEfficiency | None:
    """Computes attainable speed-up and throughput efficiency for `threads` threads.

    Returns None when any timing is non-positive or non-finite.

    Raises:
        ValueError: When `threads` is less than 1, or `copy_seconds` is non-empty
            and its length does not equal `threads`.
    """
    if threads < 1:
        message = f"threads must be at least 1, not {threads}"
        raise ValueError(message)
    if copy_seconds and len(copy_seconds) != threads:
        message = f"expected {threads} copy times, not {len(copy_seconds)}"
        raise ValueError(message)
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (serial_seconds, parallel_seconds, copies_wall_seconds)
    ):
        return None
    speedup = serial_seconds / parallel_seconds
    attainable = threads * serial_seconds / copies_wall_seconds
    efficiency = copies_wall_seconds / (threads * parallel_seconds)
    rate_sum_attainable: float | None = None
    rate_sum_efficiency: float | None = None
    spread_ratio: float | None = None
    if copy_seconds and all(
        math.isfinite(value) and value > 0.0 for value in copy_seconds
    ):
        rate_sum = sum(1.0 / value for value in copy_seconds)
        rate_sum_attainable = serial_seconds * rate_sum
        rate_sum_efficiency = speedup / rate_sum_attainable
        spread_ratio = max(copy_seconds) / min(copy_seconds)
    return ThroughputEfficiency(
        threads=threads,
        speedup=speedup,
        attainable_speedup=attainable,
        efficiency=efficiency,
        rate_sum_attainable_speedup=rate_sum_attainable,
        rate_sum_efficiency=rate_sum_efficiency,
        copy_spread_ratio=spread_ratio,
    )


def fixtures() -> dict[str, Fixture]:
    """Validates generated circuit and plan files and returns speed fixtures by name.

    Raises:
        SystemExit: When any committed JSON file differs from its generator script.
    """
    stale: list[str] = []
    for script in scripts():
        for name, top in script.circuits().items():
            written = circuit(top).model_dump_json() + "\n"
            path = CIRCUITS / f"{name}.json"
            if not path.is_file() or path.read_text(encoding="utf-8") != written:
                stale.append(f"{path.relative_to(ROOT)} ({script.__name__}.py)")
    found = planned()
    for name, fixture in found.items():
        written = format_json(engine_plan(fixture)) + "\n"
        path = PLANS / f"{name}.json"
        if not path.is_file() or path.read_text(encoding="utf-8") != written:
            stale.append(f"{path.relative_to(ROOT)}")
    if stale:
        message = "these files are not what their scripts write; run the scripts:\n"
        raise SystemExit(message + "\n".join(f"  {each}" for each in stale))
    return found


def repeated[T](work: Callable[[], T], *, once: bool) -> tuple[list[T], float, float]:
    """Runs `work` once or repeatedly and returns outputs plus start and end loads.

    When `once` is False, discards one warm-up run and repeats until reaching at least
    `MIN_RUNS` samples and `MIN_SECONDS` wall time, matching Criterion.
    """
    load = load_average()
    if once:
        return [work()], load, load_average()
    work()
    found: list[T] = []
    begun = time.perf_counter()
    while len(found) < MIN_RUNS or time.perf_counter() - begun < MIN_SECONDS:
        found.append(work())
    return found, load, load_average()


def timed(work: Callable[[], Run], *, once: bool = False) -> Timing:
    """Times repeated executions of `work` and returns an aggregated `Timing`."""
    found, load, load_after = repeated(work, once=once)
    return Timing.from_samples(found, load, load_after)


@dataclass(frozen=True, slots=True)
class Rounds:
    """Alternating-round results pairing per-work `Timing` records with round loads.

    Each entry in `loads` stores the one-minute load average before and after one round.
    """

    timings: list[Timing]
    loads: list[tuple[float, float]]

    def summary(self) -> str:
        """Formats the round count and observed host load range as a summary string."""
        seen = [each for pair in self.loads for each in pair]
        return f"{len(self.loads)} rounds, load {min(seen):.2f} to {max(seen):.2f}"


def alternated(works: Sequence[Callable[[], Run]], max_rounds: int) -> Rounds:
    """Times `works` in alternating forward and reverse rounds until comparisons settle.

    Alternating order prevents carry-over bias between adjacent runs (on `cedar`, a
    serial run averaged 261 ms after a 64-thread run versus 225 ms after a 2-thread
    run). Every work is compared against `works[0]` after discarding one warm-up round.

    Rounds continue until reaching at least `MIN_ROUNDS` and `MIN_SECONDS` per work,
    stopping once every comparison is `separated` or resolves `FLAGGED`, or at
    `max_rounds`. Stopping depends only on interval width, avoiding verdict bias.
    """
    for each in works:
        each()
    runs: list[list[Run]] = [[] for _ in works]
    loads: list[tuple[float, float]] = []
    begun = time.perf_counter()

    def pending() -> bool:
        count = len(loads)
        if count < MIN_ROUNDS:
            return True
        if time.perf_counter() - begun < MIN_SECONDS * len(works):
            return True
        if count >= max_rounds:
            return False
        first = median([each.wall for each in runs[0]])
        return any(
            not settled(first, median([each.wall for each in others]))
            for others in runs[1:]
        )

    while pending():
        order = list(range(len(works)))
        if len(loads) % 2:
            order.reverse()
        before = load_average()
        for index in order:
            runs[index].append(works[index]())
        loads.append((before, load_average()))
    timings = [Timing.from_samples(each, loads[0][0], loads[-1][1]) for each in runs]
    return Rounds(timings, loads)


def settled(before: Median, now: Median) -> bool:
    """Returns whether the comparison between `before` and `now` has settled."""
    if before.separated(now):
        return True
    resolution = before.resolution(now)
    return resolution is not None and resolution <= FLAGGED


def decided(before: Median, now: Median) -> str:
    """Returns the verdict ("slower", "faster", "undecided", "no interval", or "").

    Reports a change only when intervals separate; returns "no interval" when bounds
    are absent and "undecided" when intervals neither separate nor resolve `FLAGGED`.
    """
    if before.resolution(now) is None:
        return "no interval"
    if before.separated(now):
        return "slower" if now.value > before.value else "faster"
    return "" if settled(before, now) else "undecided"


def interval(found: Median) -> str:
    """Formats `found` as a median duration with optional confidence brackets."""
    if found.lower is None or found.upper is None:
        return f"{duration(found.value)} (no interval)"
    return f"{duration(found.value)} [{duration(found.lower)}, {duration(found.upper)}]"


def requests(
    names: Iterable[str], found: Mapping[str, Fixture], *, scale: bool = False
) -> dict[str, Request]:
    """Builds timed `Request` entries for `names` keyed by result label.

    Maps standard fixtures directly and expands Monte Carlo fixtures across one core and
    all cores (or all samples on all cores when `scale` is True).
    """
    cores = usable_cores()
    asked: dict[str, Request] = {}
    for name in names:
        monte_carlo = found[name].plan.monte_carlo
        if monte_carlo is None:
            asked[f"python/{name}"] = Request(name)
            continue
        samples = monte_carlo.samples if scale else min(SAMPLES, monte_carlo.samples)
        threads = (
            ((cores, "every core"),)
            if scale
            else ((1, "one core"), (cores, "every core"))
        )
        for count, label in threads:
            asked[f"python/{name}/{samples} samples, {label}"] = Request(
                name, samples, count
            )
    return asked


def python(
    names: Iterable[str],
    found: Mapping[str, Fixture],
    *,
    scale: bool,
    measure_memory: bool = True,
) -> dict[str, Timing]:
    """Times Python execution for `names`, running once each when `scale` is True.

    Monte Carlo timings are normalized per sample. When `measure_memory` is True,
    probes `peak_memory_bytes` in a fresh child process (reusing that single child run
    directly when `scale` is True).
    """
    timings: dict[str, Timing] = {}
    for key, request in requests(names, found, scale=scale).items():
        print(f"timing {key}", flush=True)
        if scale and measure_memory:
            load_before = load_average()
            measured = probe(request)
            timings[key] = Timing.from_samples([measured], load_before, load_average())
            continue
        measured_timing = timed(work(request, dict(found)), once=scale)
        if measure_memory:
            probed = probe(request)
            measured_timing = dataclasses.replace(
                measured_timing, peak_memory_bytes=probed.peak_memory_bytes
            )
        timings[key] = measured_timing
    return timings


@dataclass(frozen=True, slots=True)
class CopyOptions:
    """Options governing `concurrent_copies`."""

    rounds: int = COPY_ROUNDS
    warmup: bool = True
    samples: int | None = None
    single_copy_memory_bytes: int | None = None
    memory_budget_bytes: int | None = None
    python_executable: str = sys.executable


@dataclass(frozen=True, slots=True)
class TimingWorker:
    """One isolated `timing.py` child process for concurrent copies or build timing."""

    process: subprocess.Popen[str]
    requests: IO[str]
    answers: IO[str]
    jolty_path: Path
    label: str = "timing worker"
    built: Build | None = None

    @classmethod
    def start(
        cls,
        python_executable: str | Path = sys.executable,
        *,
        env: Mapping[str, str] | None = None,
        label: str = "concurrent copy worker",
    ) -> TimingWorker:
        """Starts a fresh `timing.py` child process and waits for its handshake.

        Raises:
            RuntimeError: When the child fails to start or exits before handshaking.
        """
        process = subprocess.Popen(  # noqa: S603 - active or build Python interpreter
            [str(python_executable), str(SPEED / "timing.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            cwd=ROOT,
            env=env,
        )
        if process.stdin is None or process.stdout is None:
            process.terminate()
            process.wait()
            message = f"{label} started without stdio pipes"
            raise RuntimeError(message)
        said = process.stdout.readline()
        if not said:
            with contextlib.suppress(OSError):
                process.stdin.close()
            process.wait()
            with contextlib.suppress(OSError):
                process.stdout.close()
            message = f"{label} exited before completing startup"
            raise RuntimeError(message)
        jolty_path = Path(json.loads(said)["jolty"]).resolve()
        return cls(process, process.stdin, process.stdout, jolty_path, label=label)

    @classmethod
    def for_build(cls, built: Build, padding: int) -> TimingWorker:
        """Starts a worker for `built` after verifying it imported `built`'s jolty.

        Adds a `padding`-byte environment variable to randomize process stack alignment.

        Raises:
            SystemExit: When the worker imported another jolty or failed to start.
        """
        interpreter = built.path / ".venv" / "bin" / "python"
        environment = outside() | {"JOLTY_BENCH_PADDING": "x" * padding}
        try:
            worker = cls.start(
                interpreter, env=environment, label=f"{built.revision}'s process"
            )
        except RuntimeError as error:
            raise SystemExit(str(error)) from error
        if not worker.jolty_path.is_relative_to(built.path.resolve()):
            worker.close()
            message = (
                f"{built.revision}'s process imported jolty from {worker.jolty_path}"
            )
            raise SystemExit(message)
        return dataclasses.replace(worker, built=built)

    def prepare(self, line: str) -> None:
        """Pre-loads a fixture in the worker before timing starts.

        Raises:
            RuntimeError: When the worker exits instead of acknowledging readiness.
        """
        self.requests.write(line)
        self.requests.flush()
        answer = self.answers.readline()
        if not answer or not json.loads(answer).get("ready"):
            message = f"{self.label} failed to prepare: {answer!r}"
            raise RuntimeError(message)

    def run_line(self, line: str) -> Run:
        """Sends a pre-serialized JSON request line and returns the worker's `Run`.

        Raises:
            RuntimeError: When the worker exits instead of returning a run.
        """
        self.requests.write(line)
        self.requests.flush()
        answer = self.answers.readline()
        if not answer:
            message = f"{self.label} exited while executing a run"
            raise RuntimeError(message)
        return Run(**json.loads(answer))

    def time(self, request: Request) -> Run:
        """Sends `request` to the worker and returns its timed `Run`.

        Raises:
            SystemExit: When a build worker exits instead of returning a result.
        """
        try:
            return self.run_line(json.dumps(dataclasses.asdict(request)) + "\n")
        except RuntimeError as error:
            if self.built is not None:
                message = f"{self.built.revision}'s process ended running {request}"
                raise SystemExit(message) from error
            raise

    def close(self, *, timeout: float | None = 5.0) -> None:
        """Closes stdio pipes and reaps the worker process cleanly."""
        with contextlib.suppress(OSError):
            self.requests.close()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait()
        finally:
            with contextlib.suppress(OSError):
                self.answers.close()


Worker = TimingWorker


def _synchronized_round(
    executor: concurrent.futures.ThreadPoolExecutor,
    workers: Sequence[TimingWorker],
    request_line: str,
) -> tuple[float, list[Run]]:
    """Releases all `workers` at a barrier and returns `(makespan, runs)`."""
    started = 0.0

    def mark_start() -> None:
        nonlocal started
        started = time.perf_counter()

    barrier = threading.Barrier(len(workers), action=mark_start)

    def run_copy(worker: TimingWorker) -> Run:
        barrier.wait(timeout=_BARRIER_TIMEOUT_SECONDS)
        return worker.run_line(request_line)

    runs = list(executor.map(run_copy, workers))
    return time.perf_counter() - started, runs


def _memory_budget_for_copies(options: CopyOptions) -> int | None:
    """Computes the byte budget available for launching concurrent copies."""
    if options.memory_budget_bytes is not None:
        return options.memory_budget_bytes
    available = available_memory_bytes()
    if available is None:
        return None
    return int(available * MEMORY_HEADROOM)


def concurrent_copies(
    fixture_name: str,
    copies: int,
    options: CopyOptions | None = None,
) -> ConcurrentCopies:
    """Runs `copies` independent single-thread copies of `fixture_name` concurrently.

    Each copy runs in an isolated fresh `timing.py` worker process with `serial=True`
    (`max_threads=1`, no matrix dissection) to measure attainable multi-copy throughput
    without GIL or allocator coupling. Every worker prepares the fixture before a
    synchronized barrier release, keeping startup and JSON parsing out of makespan.

    When `copies > 1` and the estimated memory (`copies * single_copy_memory_bytes`)
    exceeds the memory budget, returns a skipped `ConcurrentCopies` record instead
    of risking an out-of-memory kill.

    Raises:
        ValueError: When `copies < 1` or `options.rounds < 1`.
    """
    chosen = CopyOptions() if options is None else options
    if copies < 1:
        message = f"copies must be at least 1, not {copies}"
        raise ValueError(message)
    if chosen.rounds < 1:
        message = f"rounds must be at least 1, not {chosen.rounds}"
        raise ValueError(message)
    budget = _memory_budget_for_copies(chosen)
    if (
        copies > 1
        and chosen.single_copy_memory_bytes is not None
        and budget is not None
    ):
        estimated = copies * chosen.single_copy_memory_bytes
        if estimated > budget:
            reason = (
                f"estimated {memory(estimated)} for {copies} copies exceeds "
                f"{memory(budget)} memory budget"
            )
            return ConcurrentCopies(copies=copies, skipped_reason=reason)
    request = Request(fixture_name, samples=chosen.samples, threads=1, serial=True)
    request_payload = dataclasses.asdict(request)
    prepare_line = json.dumps(request_payload | {"prepare": True}) + "\n"
    request_line = json.dumps(request_payload) + "\n"
    workers: list[TimingWorker] = []
    workers_lock = threading.Lock()

    def start_worker(_index: int) -> TimingWorker:
        worker = TimingWorker.start(chosen.python_executable)
        with workers_lock:
            workers.append(worker)
        return worker

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=copies) as executor:
            started_workers = list(executor.map(start_worker, range(copies)))
            list(
                executor.map(
                    lambda worker: worker.prepare(prepare_line), started_workers
                )
            )
            if chosen.warmup:
                _synchronized_round(executor, started_workers, request_line)
            load_before = load_average()
            round_seconds: list[float] = []
            round_runs: list[list[Run]] = []
            for _ in range(chosen.rounds):
                elapsed, runs = _synchronized_round(
                    executor, started_workers, request_line
                )
                round_seconds.append(elapsed)
                round_runs.append(runs)
            load_after = load_average()
    finally:
        for worker in workers:
            worker.close()
    median_makespan = statistics.median(round_seconds)
    median_index = min(
        range(len(round_seconds)),
        key=lambda index: abs(round_seconds[index] - median_makespan),
    )
    copy_seconds = sorted(each.wall for each in round_runs[median_index])
    peak_samples = [
        each.peak_memory_bytes
        for batch in round_runs
        for each in batch
        if each.peak_memory_bytes is not None
    ]
    per_copy_peak = max(peak_samples) if peak_samples else None
    return ConcurrentCopies(
        copies=copies,
        rounds=chosen.rounds,
        wall_seconds=median_makespan,
        round_seconds=round_seconds,
        copy_seconds=copy_seconds,
        per_copy_peak_memory_bytes=per_copy_peak,
        load=load_before,
        load_after=load_after,
    )


@dataclass(frozen=True, slots=True)
class ScalingOptions:
    """Optional memory and concurrent-copy probes attached to `scaling`."""

    peak_memory: bool = True
    concurrent_copies: bool = True
    copy_rounds: int = COPY_ROUNDS


@dataclass(frozen=True, slots=True)
class ScalingRow:
    """One row of a `--scale` table: label, thread count, timing and digest."""

    label: str
    threads: int | None
    timing: Timing
    digest: str
    copies: ConcurrentCopies | None = None


@dataclass(frozen=True, slots=True)
class ScalingResult:
    """The times, digests, and concurrent-copies results of `scaling`."""

    timings: dict[str, Timing]
    digests: dict[str, str]
    copies: dict[str, ConcurrentCopies]


def _ratio_text(numerator: float, denominator: float) -> str:
    """Formats `numerator / denominator` as a speed-up multiplier, or empty on zero."""
    if not (
        math.isfinite(numerator)
        and math.isfinite(denominator)
        and numerator > 0.0
        and denominator > 0.0
    ):
        return ""
    return f"{numerator / denominator:.2f}x"


def _copy_cells(
    row: ScalingRow,
    serial_seconds: float,
    notes: list[str],
) -> tuple[str, str, str]:
    """Formats `(N copies, attainable, efficiency)` cells and appends any notes."""
    if row.copies is None or row.threads is None:
        return ("", "", "")
    if row.copies.skipped_reason is not None:
        notes.append(
            f"  {row.copies.copies} copies skipped: {row.copies.skipped_reason}"
        )
        return ("skipped", "", "")
    if row.copies.wall_seconds is None:
        return ("", "", "")
    measured = throughput_efficiency(
        threads=row.threads,
        serial_seconds=serial_seconds,
        parallel_seconds=row.timing.seconds,
        copies_wall_seconds=row.copies.wall_seconds,
        copy_seconds=row.copies.copy_seconds,
    )
    if measured is None:
        return (duration(row.copies.wall_seconds), "", "")
    if (
        measured.copy_spread_ratio is not None
        and measured.copy_spread_ratio >= HYBRID_SPREAD_THRESHOLD
        and measured.rate_sum_attainable_speedup is not None
        and measured.rate_sum_efficiency is not None
    ):
        notes.append(
            f"  note: {row.threads} copies span "
            f"{duration(min(row.copies.copy_seconds))} to "
            f"{duration(max(row.copies.copy_seconds))} "
            f"({measured.copy_spread_ratio:.2f}x spread across cores); "
            f"steady-state rate sum attainable "
            f"{measured.rate_sum_attainable_speedup:.2f}x, "
            f"efficiency {measured.rate_sum_efficiency:.1%}"
        )
    return (
        duration(row.copies.wall_seconds),
        f"{measured.attainable_speedup:.2f}x",
        f"{measured.efficiency:.1%}",
    )


def format_scaling_table(name: str, summary: str, rows: Sequence[ScalingRow]) -> str:
    """Formats the `--scale` table with optional concurrent-copy and memory columns."""
    if not rows:
        return f"\n{name}, serially and spread by the default parallelism, {summary}:"
    has_copies = any(row.copies is not None for row in rows)
    has_memory = any(row.timing.peak_memory_bytes is not None for row in rows)
    serial = rows[0].timing
    one = next((row.timing for row in rows[1:] if row.label == "1 thread"), None)
    heads: list[str] = ["threads", "time", "against serial", "against 1 thread"]
    widths: list[int] = [10, 34, 14, 16]
    if has_copies:
        heads.extend(["N copies", "attainable", "efficiency"])
        widths.extend([12, 10, 10])
    if has_memory:
        heads.append("peak memory")
        widths.append(12)
    heads.extend(["", "results"])
    widths.extend([11, 16])
    data_rows: list[list[str]] = []
    notes: list[str] = []
    for index, row in enumerate(rows):
        cells: list[str] = [
            row.label,
            interval(row.timing.wall_median()),
            _ratio_text(serial.seconds, row.timing.seconds),
            ""
            if one is None or index == 0
            else _ratio_text(one.seconds, row.timing.seconds),
        ]
        if has_copies:
            cells.extend(_copy_cells(row, serial.seconds, notes))
        if has_memory:
            bytes_used = row.timing.peak_memory_bytes
            cells.append("" if bytes_used is None else memory(bytes_used))
        cells.extend(
            [
                ""
                if index == 0
                else decided(serial.wall_median(), row.timing.wall_median()),
                row.digest,
            ]
        )
        data_rows.append(cells)
    lines = [
        f"\n{name}, serially and spread by the default parallelism, {summary}:",
        aligned_table(heads, data_rows, widths),
        *notes,
    ]
    return "\n".join(lines)


def scaling(
    name: str,
    loaded: Loaded,
    counts: Sequence[int],
    max_rounds: int,
    options: ScalingOptions | None = None,
) -> ScalingResult:
    """Benchmarks `name` serially and across thread `counts` in interleaved rounds.

    Compares a serial run (`serially`, one thread, no dissection) against default
    parallel runs on each thread count for at most `max_rounds` rounds. When enabled in
    `options`, probes isolated peak memory and concurrent single-thread copies for
    throughput efficiency, prints the table, and returns a `ScalingResult`.

    Raises:
        SystemExit: When a non-dissected parallel run produces a different result
            digest than the serial run.
    """
    chosen = ScalingOptions() if options is None else options
    print(f"timing {name} in Python, serially and spread, in rounds", flush=True)
    alone_options = serially(loaded)
    labels = [
        "serially",
        *(f"{count} thread{'s' if count > 1 else ''}" for count in counts),
    ]
    works = [
        on(1, functools.partial(run, loaded, alone_options)),
        *(on(count, functools.partial(run, loaded)) for count in counts),
    ]
    measured, digests = interleaved(works, max_rounds)
    found = list(measured.timings)
    if chosen.peak_memory:
        print(f"probing peak memory of {name} in isolated processes", flush=True)
        serial_probe = probe(Request(name, serial=True))
        found[0] = dataclasses.replace(
            found[0], peak_memory_bytes=serial_probe.peak_memory_bytes
        )
        for index, count in enumerate(counts, start=1):
            spread_probe = probe(Request(name, threads=count))
            found[index] = dataclasses.replace(
                found[index], peak_memory_bytes=spread_probe.peak_memory_bytes
            )
    copies_by_count: dict[int, ConcurrentCopies] = {}
    if chosen.concurrent_copies:
        single_memory = found[0].peak_memory_bytes
        for count in counts:
            print(
                f"timing {count} concurrent 1-thread "
                f"{'copy' if count == 1 else 'copies'} of {name}",
                flush=True,
            )
            copies_by_count[count] = concurrent_copies(
                name,
                count,
                CopyOptions(
                    rounds=chosen.copy_rounds,
                    single_copy_memory_bytes=single_memory,
                ),
            )
    timings = {f"python/{name}": found[0]} | {
        f"python/{name}/spread on {label}": each
        for label, each in zip(labels[1:], found[1:], strict=True)
    }
    rows = [
        ScalingRow(labels[0], None, found[0], digests[0]),
        *(
            ScalingRow(
                label,
                count,
                each,
                digested,
                copies_by_count.get(count),
            )
            for label, count, each, digested in zip(
                labels[1:], counts, found[1:], digests[1:], strict=True
            )
        ),
    ]
    print(format_scaling_table(name, measured.summary(), rows))
    alone = digests[0]
    spread_digests = set(digests[1:])
    unknowns = alone_options.parallelism.max_unknowns_per_interior or 0
    interiors = plan.BREAK_EVENS["max_unknowns_per_interior"]
    dissectable = unknowns > min(each for each in interiors if each is not None)
    if not dissectable and spread_digests != {alone}:
        message = f"{name} found other bits spread: {', '.join(digests)}"
        raise SystemExit(message)
    return ScalingResult(
        timings=timings,
        digests={"serial": alone, "spread": next(iter(spread_digests), alone)},
        copies={str(count): result for count, result in copies_by_count.items()},
    )


@dataclass(frozen=True, slots=True)
class Profiled:
    """Solver work counters and phase timing profile recorded with ``profile=True``."""

    work: results.Work
    profile: results.Profile


def profiles(
    names: Iterable[str], found: Mapping[str, Fixture]
) -> dict[str, dict[str, Profiled]]:
    """Collects transient solver counters and phase profiles for `names` by analysis.

    Executes each non-Monte-Carlo fixture once with ``profile=True`` so profiling
    overhead stays separate from main benchmark timings.
    """
    found_profiles: dict[str, dict[str, Profiled]] = {}
    for name in names:
        loaded = load_fixture(found[name])
        if loaded.plan.monte_carlo is not None:
            continue
        print(f"profiling {name}", flush=True)
        simulation = Simulation(loaded.circuit)
        options = dataclasses.replace(loaded.plan.options, profile=True)
        for analysis, asked in loaded.plan.analyses.items():
            result = simulation.run(asked, options)
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
    """Runs Rust Criterion benchmarks for `names` and returns parsed `Timing` records.

    Monte Carlo timings are normalized per sample and keyed by sample count and threads.

    Raises:
        SystemExit: When ``cargo bench`` exits with a non-zero status.
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
    """Parses Criterion `sample.json` in `new` into a per-run `Timing`.

    Divides each sample's nanoseconds by iteration count and `runs`, returning the
    median and full wall sample tuple.
    """
    sample = json.loads((new / "sample.json").read_text("utf-8"))
    wall = tuple(
        seconds * 1e-9 / iterations / runs
        for seconds, iterations in zip(sample["times"], sample["iters"], strict=True)
    )
    return Timing(statistics.median(wall), round(sum(sample["iters"])), wall=wall)


def commands(loaded: Loaded) -> list[str]:
    """Translates every analysis in `loaded.plan` into an ngspice command in order.

    Raises:
        TypeError: When the plan contains an unsupported external-simulator analysis.
    """
    return [
        command(analysis, loaded.circuit) for analysis in loaded.plan.analyses.values()
    ]


class View(StrEnum):
    """Cross-simulator comparison view selected by ``--view``.

    Supports matched tight tolerances, simulator default tolerances, and minimum time
    to reach equal accuracy on the precision suite.
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
"""Installed simulator configurations eligible for cross-simulator views."""

RANKS: Final = (1, 2, 4, 8, 16)
"""Default MPI ranks for parallel Xyce under ``--view --scale`` (up to 16 cores)."""

PRINTED: Final = 2
"""Number of leading and trailing top-level nets saved in ngspice and printed in Xyce
so output I/O stays negligible relative to solver work."""


def printed(loaded: Loaded) -> list[str]:
    """Selects up to `PRINTED` non-ground top-level nets from each circuit end."""
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
    """Returns `options` with all tolerance fields tightened to `jolty.plan.TIGHT`."""
    return dataclasses.replace(
        options, **{name: getattr(plan.TIGHT, name) for name in plan.TIGHT.to_json()}
    )


def stated(view: View, runner: Jolty | Ngspice | Xyce) -> str:
    """Formats the simulator options active for `runner` under `view`.

    Matched mode reports `jolty.plan.TIGHT`, `jolty.spice.ngspice.OPTIONS`, or
    `jolty.spice.xyce.OPTIONS`; ngspice KLU appends its solver directive in both views.
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
    """Extracts voltages of `nets` and sample coordinates from a jolty result.

    Raises:
        TypeError: When `found` is not an operating-point, AC, or transient result.
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
    """Extracts voltages and coordinates for `analysis` from ngspice or Xyce."""
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


_MIN_CUBIC_POINTS: Final = 4
"""Minimum distinct grid points required for fourth-order cubic Hermite slopes."""

_MAX_EXTREMUM_STEP_RATIO: Final = 4.0
"""Maximum adjacent step-size ratio where smooth extrema relax monotonicity."""


def _minmod(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Computes component-wise minmod (smaller magnitude when signs match, else 0)."""
    return np.where(
        first * second > 0.0,
        np.sign(first) * np.minimum(np.abs(first), np.abs(second)),
        0.0,
    )


def _median_of_three(
    first: np.ndarray, second: np.ndarray, third: np.ndarray
) -> np.ndarray:
    """Computes the component-wise median of three arrays."""
    return np.maximum(
        np.minimum(first, second), np.minimum(np.maximum(first, second), third)
    )


def _cubic_slopes(
    steps: np.ndarray,
    secants: np.ndarray,
    curvatures: np.ndarray,
) -> np.ndarray:
    """Computes fourth-order non-uniform divided-difference slopes (exact cubic)."""
    count = len(steps) + 1
    parabolic = np.empty(count, dtype=np.float64)
    parabolic[1:-1] = (steps[1:] * secants[:-1] + steps[:-1] * secants[1:]) / (
        steps[:-1] + steps[1:]
    )
    parabolic[0] = (
        (2.0 * steps[0] + steps[1]) * secants[0] - steps[0] * secants[1]
    ) / (steps[0] + steps[1])
    parabolic[-1] = (
        (2.0 * steps[-1] + steps[-2]) * secants[-1] - steps[-1] * secants[-2]
    ) / (steps[-1] + steps[-2])
    third_differences = (curvatures[1:] - curvatures[:-1]) / (
        steps[:-2] + steps[1:-1] + steps[2:]
    )
    slopes = parabolic.copy()
    if count >= 5:
        weight_left = steps[3:] + steps[2:-1]
        weight_right = steps[:-3] + steps[1:-2]
        averaged_third = (
            weight_left * third_differences[:-1] + weight_right * third_differences[1:]
        ) / (weight_left + weight_right)
        slopes[2:-2] = parabolic[2:-2] - steps[1:-2] * steps[2:-1] * averaged_third
    slopes[1] = parabolic[1] - steps[0] * steps[1] * third_differences[0]
    slopes[-2] = parabolic[-2] - steps[-2] * steps[-1] * third_differences[-1]
    slopes[0] = parabolic[0] + steps[0] * (steps[0] + steps[1]) * third_differences[0]
    slopes[-1] = (
        parabolic[-1] + steps[-1] * (steps[-1] + steps[-2]) * third_differences[-1]
    )
    return slopes


def _limit_slopes(
    steps: np.ndarray,
    secants: np.ndarray,
    curvatures: np.ndarray,
    slopes: np.ndarray,
) -> np.ndarray:
    """Applies the Huynh (1993) monotonicity-preserving limiter with extremum check.

    Monotone segments and flat shelves clamp to `[0, 3 * minmod(s_left, s_right)]`
    so sharp corners and step edges never overshoot. Smooth extrema where three
    adjacent divided-difference curvatures share sign (and adjacent steps stay within
    `_MAX_EXTREMUM_STEP_RATIO`) relax within the local parabolic bounds so smooth
    sine peaks reconstruct to fourth order instead of suffering linear chord sag.
    """
    limited = slopes.copy()
    padded_curvatures = np.empty(len(slopes), dtype=np.float64)
    padded_curvatures[1:-1] = curvatures
    padded_curvatures[0] = curvatures[0]
    padded_curvatures[-1] = curvatures[-1]

    left_secants = secants[:-1]
    right_secants = secants[1:]
    left_extrapolated = left_secants + padded_curvatures[:-2] * steps[:-1]
    right_extrapolated = right_secants - padded_curvatures[2:] * steps[1:]

    bounded = _minmod(left_secants, right_secants)
    monotone_slopes = _median_of_three(
        slopes[1:-1], np.zeros_like(bounded), 3.0 * bounded
    )

    smooth = (padded_curvatures[:-2] * padded_curvatures[1:-1] > 0.0) & (
        padded_curvatures[1:-1] * padded_curvatures[2:] > 0.0
    )
    step_ratio = np.maximum(steps[:-1], steps[1:]) / np.minimum(steps[:-1], steps[1:])
    lower_bound = np.minimum(
        np.minimum(left_secants, right_secants),
        np.minimum(left_extrapolated, right_extrapolated),
    )
    upper_bound = np.maximum(
        np.maximum(left_secants, right_secants),
        np.maximum(left_extrapolated, right_extrapolated),
    )
    smooth_slopes = np.clip(
        slopes[1:-1],
        np.minimum(lower_bound, 1.5 * lower_bound),
        np.maximum(upper_bound, 1.5 * upper_bound),
    )
    limited[1:-1] = np.where(
        smooth & (step_ratio <= _MAX_EXTREMUM_STEP_RATIO),
        smooth_slopes,
        monotone_slopes,
    )
    limited[0] = float(_median_of_three(slopes[0], np.array(0.0), 3.0 * secants[0]))
    limited[-1] = float(_median_of_three(slopes[-1], np.array(0.0), 3.0 * secants[-1]))
    return limited


def interpolate_transient(
    query: np.ndarray, times: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Interpolates a transient waveform with Huynh (1993) cubic Hermite interpolation.

    Reference: H. T. Huynh, "Accurate Monotone Cubic Interpolation",
    SIAM J. Numer. Anal. 30(1):57-100, 1993.

    Deduplicates repeated breakpoint timestamps by keeping the last sample at each
    distinct time. Falls back to piecewise linear `np.interp` when fewer than four
    distinct time points exist. Out-of-range queries clamp to endpoint values.
    """
    sample_times = np.asarray(times, dtype=np.float64)
    sample_values = np.asarray(values, dtype=np.float64)
    query_times = np.asarray(query, dtype=np.float64)
    if len(sample_times) == 0:
        return np.interp(query_times, sample_times, sample_values)
    keep = np.ones(len(sample_times), dtype=bool)
    keep[:-1] = sample_times[:-1] < sample_times[1:]
    sample_times = sample_times[keep]
    sample_values = sample_values[keep]
    if len(sample_times) < _MIN_CUBIC_POINTS:
        return np.interp(query_times, sample_times, sample_values)

    steps = np.diff(sample_times)
    secants = np.diff(sample_values) / steps
    curvatures = (secants[1:] - secants[:-1]) / (steps[:-1] + steps[1:])
    raw_slopes = _cubic_slopes(steps, secants, curvatures)
    slopes = _limit_slopes(steps, secants, curvatures, raw_slopes)

    clamped = np.clip(query_times, sample_times[0], sample_times[-1])
    interval_index = np.clip(
        np.searchsorted(sample_times, clamped, side="right") - 1,
        0,
        len(sample_times) - 2,
    )
    interval_width = steps[interval_index]
    offset = clamped - sample_times[interval_index]
    left_value = sample_values[interval_index]
    left_slope = slopes[interval_index]
    right_slope = slopes[interval_index + 1]
    secant = secants[interval_index]
    quadratic = (3.0 * secant - 2.0 * left_slope - right_slope) / interval_width
    cubic = (left_slope + right_slope - 2.0 * secant) / (interval_width**2)
    return left_value + offset * (left_slope + offset * (quadratic + offset * cubic))


def deviations(
    analyses: Mapping[str, Analysis],
    reference: Mapping[str, Voltages],
    found: Mapping[str, Voltages],
) -> dict[str, float]:
    """Computes maximum normalized voltage deviation between `found` and `reference`.

    Normalizes each net's peak error by reference voltage magnitude (or peak-to-peak
    swing for transients). Evaluates transient references with Huynh (1993) cubic
    Hermite interpolation to avoid chord sag across extrema and overshoot at corners.
    """
    worst: dict[str, float] = {}
    for name, voltages in found.items():
        expected = reference[name]
        swing = isinstance(analyses[name], Transient)
        same_grid = (
            voltages.points is not None
            and expected.points is not None
            and np.array_equal(voltages.points, expected.points)
        )
        largest = 0.0
        for net, values in voltages.nets.items():
            ours = expected.nets[net]
            if voltages.points is None or expected.points is None or same_grid:
                there = ours
            elif swing:
                there = interpolate_transient(
                    voltages.points, expected.points, ours.real
                )
            else:
                there = np.interp(voltages.points, expected.points, ours)
            scale = float(np.ptp(ours.real) if swing else np.abs(ours).max())
            largest = max(largest, float(np.abs(values - there).max()) / (scale or 1.0))
        worst[name] = largest
    return worst


@dataclass(frozen=True, slots=True)
class TrialResult:
    """Single-run timing and voltage results for a fixture in one simulator."""

    seconds: float
    """End-to-end wall seconds including elaboration or child startup and exit."""
    analysis_seconds: float
    """Summed solver analysis seconds from ``Simulation.run`` or simulator reports."""
    voltages: dict[str, Voltages]
    """Extracted node voltages keyed by analysis name."""


def ran(
    view: View,
    runner: Jolty | Ngspice | Xyce,
    loaded: Loaded,
    nets: Sequence[str],
    timeout: float,
) -> TrialResult:
    """Executes all analyses of `loaded` once via `runner` under `view` options.

    Runs jolty in-process, ngspice in a single process saving only `nets`, and Xyce in
    separate processes per analysis printing `nets`, each bounded by `timeout` seconds.

    Raises:
        JoltyError: When execution fails, times out, or an external simulator rejects
            the exported netlist.
    """
    matched = view is View.MATCHED
    analyses = loaded.plan.analyses
    match runner:
        case Jolty():
            options = loaded.plan.options
            started = time.perf_counter()
            analyzed = runner.run(
                loaded.circuit, analyses, tightened(options) if matched else options
            )
            seconds = time.perf_counter() - started
            voltages = {
                name: jolty_voltages(found, nets)
                for name, found in analyzed.found.items()
            }
            return TrialResult(seconds, analyzed.analysis_seconds, voltages)
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
            return TrialResult(seconds, float(stated.group(1)), voltages)
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
            return TrialResult(seconds, analysis_seconds, voltages)


@dataclass(frozen=True, slots=True)
class Cell:
    """Summary cell for one fixture in one simulator under a comparison view.

    Stores median end-to-end and analysis timings plus maximum voltage deviations from
    matched jolty, or records the failure reason.
    """

    options: str
    """Formatted simulator options string from `stated`."""
    load: float
    """One-minute host load average recorded before execution."""
    load_after: float
    """One-minute host load average recorded after execution."""
    seconds: Timing | None = None
    analysis_seconds: Timing | None = None
    deviations: dict[str, float] = dataclasses.field(default_factory=dict)
    failed: str | None = None
    """Failure summary string when execution failed, otherwise None."""


def cell(
    options: str,
    work: Callable[[], TrialResult],
    analyses: Mapping[str, Analysis],
    reference: Mapping[str, Voltages] | None,
    *,
    once: bool,
) -> Cell:
    """Executes `work` once or repeatedly under `options` and builds a summary `Cell`.

    Stops on the first `JoltyError` and records the failure summary; otherwise compares
    the first run's voltages against `reference` when available.
    """
    load = load_average()
    try:
        found, _, load_after = repeated(work, once=once)
    except JoltyError as error:
        return Cell(options, load, load_average(), failed=stopped(str(error)))
    walls = tuple(each.seconds for each in found)
    analyzed = tuple(each.analysis_seconds for each in found)
    return Cell(
        options,
        load,
        load_after,
        Timing(statistics.median(walls), len(found), load, load_after, walls),
        Timing(statistics.median(analyzed), len(found), load, load_after, analyzed),
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
    """Benchmarks `names` across `simulators` under `view`, returning per-config cells.

    Selects named simulators (or all installed `VIEWED` entries) plus parallel Xyce
    across `ranks` when `parallel` and `scale` are set, skipping Monte Carlo fixtures.
    Runs each fixture repeatedly (or once at `scale`) under `timeout` seconds and
    compares voltages against matched jolty.

    Raises:
        SystemExit: When a requested simulator is unsupported or not installed.
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
    """Formats a `Cell` timing and deviation summary for a view table."""
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
    """Formats a Markdown view table with fixture rows and configuration columns.

    Prepends an options row stating each configuration's settings.
    """
    configurations = list(dict.fromkeys(c for row in cells.values() for c in row))
    options = {c: row[c].options for row in cells.values() for c in row}
    rows = [
        ["options", *(options[c] for c in configurations)],
        *(
            [f"`{name}`", *(entry(row[c]) if c in row else "" for c in configurations)]
            for name, row in cells.items()
        ),
    ]
    return markdown_table(["fixture", *configurations], rows)


def baseline(given: str | None) -> Path | None:
    """Resolves a baseline path from `given` or the newest ancestor with results.

    Returns None when `given` is None and no ancestor commit has saved speed results.

    Raises:
        SystemExit: When `given` matches neither a file nor saved commit results.
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
    for ancestor in git("rev-list", "--max-count=1000", "HEAD").split()[1:]:
        found = results_directory(Suite.SPEED) / f"{ancestor}.json"
        if found.is_file():
            return found
    return None


def _memory_delta(old_bytes: int | None, new_bytes: int | None) -> str:
    """Formats a timing's peak memory against an optional earlier measurement."""
    if new_bytes is not None:
        formatted = memory(new_bytes)
        if old_bytes is not None and old_bytes > 0:
            return f"{formatted} ({new_bytes / old_bytes - 1:+.1%})"
        return formatted
    if old_bytes is not None:
        return f"was {memory(old_bytes)}"
    return ""


def format_speed_table(timings: Mapping[str, Timing]) -> str:
    """Formats measured times and peak resident memory as an aligned table."""
    if not timings:
        return ""
    width = max(len(key) for key in timings)
    has_memory = any(each.peak_memory_bytes is not None for each in timings.values())
    heads: list[str] = ["fixture", "time"]
    widths: list[int] = [max(width, len("fixture")), 34]
    if has_memory:
        heads.append("peak memory")
        widths.append(12)
    data_rows = []
    for key, each in timings.items():
        row = [key, interval(each.wall_median())]
        if has_memory:
            peak = each.peak_memory_bytes
            row.append("" if peak is None else memory(peak))
        data_rows.append(row)
    return "\n".join(["\nspeed summary:", aligned_table(heads, data_rows, widths)])


def format_against_table(
    now: Mapping[str, Timing], before: Mapping[str, Timing], name: str
) -> str:
    """Formats the times `now` against those `before`, from the results `name`."""
    shared = [key for key in now if key in before]
    if not shared:
        return f"\nno time measured here is in {name}"
    width = max(len(key) for key in shared)
    has_memory = any(
        before[key].peak_memory_bytes is not None
        or now[key].peak_memory_bytes is not None
        for key in shared
    )
    heads: list[str] = [" ", "before", "now", "change", ""]
    widths: list[int] = [width, 34, 34, 8, 11]
    if has_memory:
        heads.append("peak memory")
        widths.append(22)
    named = 0
    data_rows: list[list[str]] = []
    for key in shared:
        old, new = before[key].wall_median(), now[key].wall_median()
        verdict = decided(old, new)
        named += verdict in {"slower", "faster"}
        change_text = f"{new.value / old.value - 1:+.1%}" if old.value > 0 else ""
        cells: list[str] = [
            key,
            interval(old),
            interval(new),
            change_text,
            verdict,
        ]
        if has_memory:
            cells.append(
                _memory_delta(before[key].peak_memory_bytes, now[key].peak_memory_bytes)
            )
        data_rows.append(cells)
    lines = [
        f"\nagainst {name}, medians with their {CONFIDENCE:.0%} intervals:",
        aligned_table(heads, data_rows, widths),
        f"{named} of {len(shared)} changed, their intervals apart",
    ]
    return "\n".join(lines)


def against(now: Mapping[str, Timing], before: Mapping[str, Timing], name: str) -> None:
    """Prints the times `now` against those `before`, from the results `name`.

    Each median with its interval, peak resident memory when measured, and a change
    named only where the intervals separate; a time of results that kept no runs has
    no interval, so no change of it is named. The two were measured at different
    times, under whatever load each met: ``--compare`` alternates the builds' runs
    instead.
    """
    print(format_against_table(now, before, name))


def digest(found: Mapping[str, results.AnalysisResult]) -> str:
    """Computes a 16-hex-char SHA-256 digest over all NumPy result arrays in `found`.

    Hashes each analysis name and array payload in order so identical bits yield
    identical digests.
    """
    hasher = hashlib.sha256()
    for name, result in found.items():
        hasher.update(name.encode())
        for each in dataclasses.fields(result):
            value = getattr(result, each.name)
            if isinstance(value, np.ndarray):
                hasher.update(value.tobytes())
    return hasher.hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class Sized:
    """Circuit built at one parameter size together with parallelism workload counts."""

    size: int
    loaded: Loaded
    devices: int
    """Flattened device count split by instance-parallel tasks."""
    unknowns: int
    """Total circuit unknown count governed by unknown tasks and matrix dissection."""
    solves: int
    """Parallelizable AC frequency count (all points excluding the first per sweep)."""


def devices(built: ir.Circuit) -> int:
    """Counts total flattened device instances across the hierarchy of `built`."""

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
    """Constructs the circuit, plan, and workload counts for `script` at `size`.

    Uses transient plans for ``inverter_chain`` and ``rc_mesh``, and the ``ac_wide``
    plan at `size` points per decade for ``bjt_amp``.

    Raises:
        ValueError: When `script` is not one of the supported break-even scripts.
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
    """Returns the workload count of `at` controlled by `threshold`."""
    match threshold:
        case Threshold.MIN_SOLVES_PER_TASK:
            return at.solves
        case Threshold.MIN_INSTANCES_PER_TASK:
            return at.devices
        case Threshold.MIN_UNKNOWNS_PER_TASK | Threshold.MAX_UNKNOWNS_PER_INTERIOR:
            return at.unknowns


def setting(threshold: Threshold, at: Sized, count: int) -> int:
    """Computes the `threshold` value splitting `at`'s work across `count` threads.

    Rounds minimum task sizes down and maximum dissection interior sizes up.
    """
    if threshold is Threshold.MAX_UNKNOWNS_PER_INTERIOR:
        return -(-at.unknowns // count)
    return max(1, counted(threshold, at) // count)


def spreading_none(at: Sized) -> dict[str, int]:
    """Returns `Parallelism` overrides keeping all work for `at` in one task."""
    return {each.value: max(1, counted(each, at)) for each in Threshold}


def on(
    count: int, work: Callable[[], dict[str, results.AnalysisResult]]
) -> Callable[[], dict[str, results.AnalysisResult]]:
    """Wraps `work` to execute inside a `simulate.threads` pool of `count` threads."""

    def within() -> dict[str, results.AnalysisResult]:
        with simulate.threads(count):
            return work()

    return within


def interleaved(
    works: Sequence[Callable[[], dict[str, results.AnalysisResult]]], max_rounds: int
) -> tuple[Rounds, list[str]]:
    """Times `works` in `alternated` rounds and returns `Rounds` and warm-up digests."""
    digests: list[str | None] = [None for _ in works]

    def remembered(index: int) -> Callable[[], Run]:
        def kept() -> None:
            found = works[index]()
            if digests[index] is None:
                digests[index] = digest(found)

        return clocked(kept)

    measured = alternated(
        [remembered(index) for index in range(len(works))], max_rounds
    )
    return measured, [found or "" for found in digests]


SIZE: Final = {"inverter_chain": "inverters", "rc_mesh": "side", "bjt_amp": "a decade"}
"""Column label describing the size parameter of each break-even script."""


def table(cells: Sequence[str], widths: Sequence[int]) -> str:
    """Formats `cells` as a right-aligned row with two spaces between columns."""
    return aligned_row(cells, widths)


MARKS: Final = {
    "faster": "<",
    "slower": ">",
    "undecided": "?",
    "no interval": "?",
    "": "",
}
"""Single-character verdict symbols used in break-even sweep tables."""


def pays_from(gains: Sequence[bool]) -> int | None:
    """Returns the first index in `gains` where every later entry is True, or None."""
    first: int | None = None
    for index, gain in enumerate(gains):
        if not gain:
            first = None
        elif first is None:
            first = index
    return first


def sweep(
    threshold: Threshold,
    script: str,
    sizes: Sequence[int],
    counts: Sequence[int],
    max_rounds: int,
) -> dict[str, Timing]:
    """Sweeps `threshold` across `sizes` and `counts` for `script` to find break-evens.

    Runs each size serially and with only `threshold` active across `counts` threads in
    `interleaved` rounds up to `max_rounds`. Prints per-size verdicts and reports the
    smallest size and threshold value from which every larger size stays faster.

    Raises:
        SystemExit: When a non-dissection threshold changes numerical bits relative to
            the serial run.
    """
    labels = [f"{count} thread{'s' if count > 1 else ''}" for count in counts]
    heads = [SIZE[script], "devices", "unknowns", "solves", "serially"]
    widths = [10, 9, 9, 7, 10, *(len(label) + 19 for label in labels), 0]
    print(f"\n{threshold} on {script}'s circuits, serially and set, to the value in")
    print("brackets, to spread each circuit over each number of threads; < marks a")
    print("run faster than the serial run, > slower, their intervals apart, and ? one")
    print(f"whose intervals neither part nor resolve {FLAGGED:.0%}:")
    print(table([*heads, *labels, "rounds and load"], widths), flush=True)
    timings: dict[str, Timing] = {}
    measured: list[tuple[Sized, list[float], list[bool]]] = []
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
        rounds, digests = interleaved(works, max_rounds)
        found = rounds.timings
        # Interiors of other sizes order the factorization otherwise, so only the
        # other thresholds, which change no result, must find the serial run's bits.
        dissected = threshold is Threshold.MAX_UNKNOWNS_PER_INTERIOR
        if not dissected and len(set(digests)) != 1:
            message = (
                f"{script} at {size} found other bits with {threshold} set: "
                + ", ".join(digests)
            )
            raise SystemExit(message)
        unset = found[0].seconds
        changes = [each.seconds / unset - 1 for each in found[1:]]
        verdicts = [
            decided(found[0].wall_median(), each.wall_median()) for each in found[1:]
        ]
        measured.append((at, changes, [each == "faster" for each in verdicts]))
        key = f"python/break-even/{threshold}/{script}_{size}"
        timings[f"{key}/serially"] = found[0]
        for label, each in zip(labels, found[1:], strict=True):
            timings[f"{key}/{label}"] = each
        cells = [f"{each:,}" for each in (size, at.devices, at.unknowns, at.solves)]
        cells.append(duration(unset))
        cells += [
            f"{duration(each.seconds)} {change:+.1%}{MARKS[verdict]} ({value:,})"
            for each, change, verdict, value in zip(
                found[1:], changes, verdicts, values, strict=True
            )
        ]
        cells.append(rounds.summary())
        print(table(cells, widths), flush=True)
    for place, label in enumerate(labels):
        changes = [each[place] for _, each, _ in measured]
        first = pays_from([gains[place] for _, _, gains in measured])
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
    """CLI for speed, scaling, comparison, robustness, coverage, and precision."""

    model_config = SettingsConfigDict(cli_prog_name="scripts/bench")

    fixtures: CliPositionalArg[list[str]] = Field(
        default_factory=list,
        description=(
            "Fixtures, break-even scripts, or coverage rows to run. Defaults to the "
            "active suite's full set."
        ),
    )
    scale: bool = Field(
        default=False, description="Benchmark the large scale fixtures once each."
    )
    break_even: Threshold | None = Field(
        default=None,
        description=(
            "Measure the break-even size for this Parallelism threshold across "
            "applicable circuit scripts."
        ),
    )
    largest: bool = Field(
        default=False,
        description=(
            "With --scale, also sweep the 50,000-inverter chain across thread counts "
            "in rounds (about 17 minutes per sweep on 12 cores)."
        ),
    )
    sizes: list[int] = Field(
        default_factory=list,
        description=(
            "Circuit sizes swept by --break-even (defaults to sizes in README.md)."
        ),
    )
    threads: list[int] = Field(
        default_factory=list,
        description=(
            "Thread counts for --break-even (default 2, 4, 8, all cores) or --scale "
            "(default 1, 2, 4, 8, all cores)."
        ),
    )
    view: View | None = Field(
        default=None,
        description=(
            "Compare simulators under one view: matched options, simulator defaults, "
            "or equal accuracy on the precision suite."
        ),
    )
    digits: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Target significant correct digits for --precision and --view equal-error "
            "(defaults to the highest accuracy reached by every simulator per output)."
        ),
    )
    robustness: bool = Field(
        default=False,
        description=(
            "Run the hard-circuit robustness corpus across simulators and report "
            "classified outcomes and failure rates."
        ),
    )
    coverage: bool = Field(
        default=False,
        description=(
            "Run coverage matrix fixtures across simulators, score against known "
            "answers, and render matrix.md."
        ),
    )
    precision: bool = Field(
        default=False,
        description=(
            "Run precision tolerance sweeps across simulators, score against exact "
            "references, and generate work-precision diagrams."
        ),
    )
    simulator: list[Simulator] = Field(
        default_factory=list,
        description=(
            "Simulators for --view, --robustness, --coverage, or --precision (such as "
            "jolty, ngspice-sparse, or xyce-serial). Defaults to installed simulators "
            "(or jolty and ngspice Sparse/KLU for --coverage)."
        ),
    )
    timeout: float = Field(
        default=TIMEOUT,
        gt=0,
        description="Per-run timeout in seconds for --view and --robustness.",
    )
    sweeps: list[suite.Sweep] = Field(
        default_factory=lambda: list(suite.Sweep),
        description=(
            "Tolerance sweeps executed by --precision (both sweeps by default)."
        ),
    )
    repeats: int = Field(
        default=5,
        ge=1,
        description=(
            "Number of repeats per tolerance setting for median timing in --precision."
        ),
    )
    plot: Path | None = Field(
        default=None,
        description=(
            "Precision results file to re-plot with --precision or --view equal-error "
            "without rerunning simulations."
        ),
    )
    xyce_mpi: Path | None = Field(
        default=None,
        description=(
            "Path to parallel Xyce binary run with --view --scale across 1, 2, 4, 8, "
            "and 16 MPI ranks (or --threads)."
        ),
    )
    baseline: str | None = Field(
        default=None,
        description="Baseline commit hash or results JSON file to compare against.",
    )
    compare: list[str] = Field(
        default_factory=list,
        description=(
            "Two revisions (such as main and HEAD) to build in isolated worktrees and "
            "benchmark in alternating rounds. Reports wall and CPU medians with 95% "
            "intervals, flagging changes only when intervals separate."
        ),
    )
    builds: Path = Field(
        default=ROOT / "target" / "builds",
        description="Directory holding isolated git worktrees built by --compare.",
    )
    most_rounds: int = Field(
        default=MAX_ROUNDS,
        ge=MIN_ROUNDS,
        description=(
            "Maximum alternating rounds for --compare, --scale, and --break-even "
            "before stopping undecided."
        ),
    )
    peak_memory: bool = Field(
        default=True,
        description=(
            "Measure each fixture's peak resident memory (VmHWM on Linux, "
            "getrusage(2) ru_maxrss on macOS) in a fresh child process."
        ),
    )
    concurrent_copies: bool = Field(
        default=True,
        description=(
            "With --scale, also run N independent single-thread copies concurrently "
            "to measure attainable speed-up and throughput efficiency."
        ),
    )
    copy_rounds: int = Field(
        default=COPY_ROUNDS,
        ge=1,
        description=(
            "Number of synchronized concurrent-copy rounds used for the median "
            "makespan in --scale."
        ),
    )
    output: Path | None = Field(
        default=None, description="Destination path for the results JSON file."
    )

    def cli_cmd(self) -> None:
        """Runs the benchmarks."""
        measure(self)


def main(argv: list[str] | None = None) -> None:
    """Runs the command line, `argv` or the process's own arguments."""
    cli.run(Bench, argv)


def every(counts: Iterable[int]) -> list[int]:
    """Returns counts below `usable_cores()` followed by `usable_cores()`."""
    cores = usable_cores()
    return [count for count in counts if count < cores] + [cores]


def merged(  # noqa: PLR0913 - each section of a speed suite results file
    output: Path,
    timings: Mapping[str, Timing],
    profiles: Mapping[str, object] | None = None,
    digests: Mapping[str, object] | None = None,
    views: Mapping[str, object] | None = None,
    *,
    copies: Mapping[str, Mapping[str, ConcurrentCopies]] | None = None,
) -> dict[str, object]:
    """Merges new timings, profiles, digests, views, and copies into `output`.

    Preserves previously recorded entries in each section while updating newly measured
    keys.
    """
    earlier = load(output) if output.is_file() and output.stat().st_size > 0 else {}
    written = {**earlier.get("timings", {})}
    written |= {key: each.to_json() for key, each in timings.items()}
    merged_copies: dict[str, dict[str, object]] = {
        fixture_name: dict(entries)
        for fixture_name, entries in earlier.get("copies", {}).items()
    }
    for fixture_name, entries in (copies or {}).items():
        merged_copies[fixture_name] = merged_copies.get(fixture_name, {}) | {
            count: entry.model_dump(mode="json", exclude_none=True)
            for count, entry in entries.items()
        }
    return {
        **provenance().model_dump(mode="json"),
        "timings": written,
        "profiles": earlier.get("profiles", {}) | dict(profiles or {}),
        "digests": earlier.get("digests", {}) | dict(digests or {}),
        "views": earlier.get("views", {}) | dict(views or {}),
        "copies": merged_copies,
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
        timings |= sweep(threshold, script, sizes, counts, arguments.most_rounds)
    print(f"\nmeasured in {duration(time.perf_counter() - started)}")
    output = arguments.output or results_path(Suite.SPEED)
    save(output, merged(output, timings))


@dataclass(frozen=True, slots=True)
class Build:
    """Jolty build checked out at one commit in a dedicated git worktree."""

    revision: str
    """User-supplied revision specification such as ``main``."""
    commit: str
    path: Path


def outside() -> dict[str, str]:
    """Returns the current process environment excluding `VIRTUAL_ENV`.

    Strips the active ``uv run`` virtual environment so child builds resolve isolated
    interpreters.
    """
    return {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}


def build(revision: str, directory: Path, taken: Iterable[Path]) -> Build:
    """Checks out and builds `revision` in a dedicated worktree under `directory`.

    Names the worktree after the commit hash (appending ``-2`` if `taken` already
    includes that path) and builds via ``uv sync`` into an isolated Cargo target
    directory.
    """
    found = git("rev-parse", "--verify", f"{revision}^{{commit}}")
    path = directory / found
    if path in taken:
        path = directory / f"{found}-2"
    if not path.is_dir():
        directory.mkdir(parents=True, exist_ok=True)
        git("worktree", "add", "--detach", str(path), found)
    elif git("-C", str(path), "rev-parse", "HEAD") != found:
        git("-C", str(path), "checkout", "--detach", found)
    print(f"building {revision} ({found[:12]}) in {shown(path)}", flush=True)
    environment = outside() | {"CARGO_TARGET_DIR": str(path / "target")}
    subprocess.run(
        ["uv", "sync", "--locked"],  # noqa: S607 - uv is on the path
        check=True,
        cwd=path,
        env=environment,
    )
    return Build(revision, found, path)


def afresh(built: Build, request: Request, padding: Callable[[], int]) -> Run:
    """Runs `request` in a fresh padded worker for `built` after one warm-up run.

    Environment size shifts stack alignment and can bias runtime by several percent
    (Mytkowicz et al., ASPLOS 2009). Memory and thread placement also differ across
    processes: on `dogwood`, reusing one worker process flagged four spurious changes on
    identical commits (+3.0% on buck, -6.3% on ``rc_mesh_100``). Launching a fresh
    worker with sub-page `padding` turns layout bias into unbiased interval variance.
    """
    worker = TimingWorker.for_build(built, padding())
    try:
        worker.time(request)
        return worker.time(request)
    finally:
        worker.close(timeout=None)


def builds_compared(arguments: Bench) -> None:
    """Times the builds of the two revisions ``--compare`` names against each other.

    Raises:
        SystemExit: When it names other than two.
    """
    if len(arguments.compare) != 2:
        message = f"--compare names two revisions, not {len(arguments.compare)}"
        raise SystemExit(message)
    found = fixtures()
    names = arguments.fixtures or SUITE
    first, second = arguments.compare
    output = arguments.output or results_directory(Suite.SPEED) / (
        f"{git('rev-parse', '--short=12', first)}.."
        f"{git('rev-parse', '--short=12', second)}.json"
    )
    compared((first, second), names, found, arguments, output)


def compared(
    revisions: tuple[str, str],
    names: Sequence[str],
    found: Mapping[str, Fixture],
    arguments: Bench,
    output: Path,
) -> list[str]:
    """Benchmarks `revisions` against each other on `names` in alternating rounds.

    Builds each revision in a dedicated worktree under ``--builds`` and executes every
    sample `afresh` in an isolated `Worker` with deterministically seeded environment
    padding. Runs each fixture request for up to ``--most-rounds`` rounds, prints and
    writes JSON plus sibling Markdown reports, and returns flagged changes.
    """
    first = build(revisions[0], arguments.builds, ())
    second = build(revisions[1], arguments.builds, (first.path,))
    pair = (first, second)
    generator = random.Random(0)  # noqa: S311 - paddings, drawn alike every time
    page = mmap.PAGESIZE

    def padding() -> int:
        return generator.randrange(page)

    measured: dict[str, Rounds] = {}
    started = time.perf_counter()
    for key, request in requests(names, found).items():
        print(f"timing {key} in both builds, round after round", flush=True)
        measured[key] = alternated(
            [functools.partial(afresh, each, request, padding) for each in pair],
            arguments.most_rounds,
        )
    print(f"\nmeasured in {duration(time.perf_counter() - started)}")
    text, named = comparison(pair, measured)
    print(f"\n{text}")
    save(
        output,
        {
            **provenance().model_dump(mode="json"),
            "builds": [
                {"revision": each.revision, "commit": each.commit} for each in pair
            ],
            "timings": {
                key: {
                    "loads": rounds.loads,
                    "builds": [each.to_json() for each in rounds.timings],
                }
                for key, rounds in measured.items()
            },
            "named": named,
        },
    )
    output.with_suffix(".md").write_text(text + "\n", encoding="utf-8")
    print(f"wrote {shown(output.with_suffix('.md'))}")
    return named


def comparison(
    pair: tuple[Build, Build], measured: Mapping[str, Rounds]
) -> tuple[str, list[str]]:
    """Formats a Markdown comparison table for `pair` and returns flagged changes.

    Emits one row per fixture and clock (wall and CPU), showing medians with intervals,
    relative change, minimum resolvable shift, verdict, and round load summary.
    """
    first, second = pair
    headers = [
        "Time",
        "Clock",
        f"`{first.revision}` ({first.commit[:12]})",
        f"`{second.revision}` ({second.commit[:12]})",
        "Change",
        "Resolution",
        "Verdict",
        "Rounds and load",
    ]
    rows: list[list[str]] = []
    named: list[str] = []
    for key, rounds in measured.items():
        before, now = rounds.timings
        clocks = [("wall", before.wall_median(), now.wall_median())]
        cpu_before, cpu_now = before.cpu_median(), now.cpu_median()
        if cpu_before is not None and cpu_now is not None:
            clocks.append(("processor", cpu_before, cpu_now))
        for clock, old, new in clocks:
            change = new.value / old.value - 1
            resolution = old.resolution(new)
            verdict = decided(old, new)
            if verdict in {"slower", "faster"}:
                named.append(f"{key}, {clock}: {change:+.1%}, {verdict}")
            rows.append(
                [
                    f"`{key}`",
                    clock,
                    interval(old),
                    interval(new),
                    f"{change:+.1%}",
                    "" if resolution is None else f"{resolution:.1%}",
                    verdict,
                    rounds.summary(),
                ]
            )
    lines = [markdown_table(headers, rows)]
    memory_rows: list[list[str]] = []
    for key, rounds in measured.items():
        old_bytes = rounds.timings[0].peak_memory_bytes
        new_bytes = rounds.timings[1].peak_memory_bytes
        if old_bytes is None and new_bytes is None:
            continue
        delta = (
            f"{new_bytes / old_bytes - 1:+.1%}"
            if old_bytes is not None and old_bytes > 0 and new_bytes is not None
            else ""
        )
        memory_rows.append(
            [
                f"`{key}`",
                "" if old_bytes is None else memory(old_bytes),
                "" if new_bytes is None else memory(new_bytes),
                delta,
            ]
        )
    if memory_rows:
        memory_headers = [
            "Peak memory",
            f"`{first.revision}` ({first.commit[:12]})",
            f"`{second.revision}` ({second.commit[:12]})",
            "Change",
        ]
        lines.extend(["", markdown_table(memory_headers, memory_rows)])
    lines.append("")
    lines.append(
        f"Each time is the median of its rounds, with its {CONFIDENCE:.0%} interval; "
        "a change is named only where the two intervals separate."
    )
    lines.append(
        "Changes named: " + "; ".join(named) + "." if named else "No change named."
    )
    return "\n".join(lines), named


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
    """Prints each simulator's minimum time to equal accuracy from the precision suite.

    Runs ``--precision`` (or loads ``--plot``) and writes the equal-error table
    alongside the work-precision diagrams.
    """
    print(f"\n{suite.equal_error_table(precision(arguments), arguments.digits)}")


def instead(arguments: Bench) -> bool:
    """Dispatches non-default benchmark sub-modes selected in `arguments`.

    Returns True when ``--robustness``, ``--coverage``, ``--precision``,
    ``--view equal-error``, ``--break-even``, or ``--compare`` ran.
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
    elif arguments.compare:
        builds_compared(arguments)
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
    """Benchmarks `names` under `view` and writes the Markdown table alongside `output`.

    Returns end-to-end and analysis timings together with serialized cells keyed by
    view, configuration, and fixture.
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
    baseline_path = baseline(arguments.baseline)
    started = time.perf_counter()
    found_profiles: dict[str, dict[str, Profiled]] = {}
    digests: dict[str, dict[str, str]] = {}
    views: dict[str, object] = {}
    found_copies: dict[str, dict[str, ConcurrentCopies]] = {}
    if arguments.view is not None:
        timings, views = viewed(arguments, arguments.view, names, found, output)
    else:
        timings = {} if arguments.scale else rust(names, found)
        spread = {
            name
            for name in names
            if arguments.scale
            and found[name].plan.monte_carlo is None
            and (name != LONGEST or arguments.largest)
        }
        counts = arguments.threads or every(THREADS)
        scaling_options = ScalingOptions(
            peak_memory=arguments.peak_memory,
            concurrent_copies=arguments.concurrent_copies,
            copy_rounds=arguments.copy_rounds,
        )
        for name in names:
            if name in spread:
                scaled = scaling(
                    name,
                    load_fixture(found[name]),
                    counts,
                    arguments.most_rounds,
                    scaling_options,
                )
                timings |= scaled.timings
                digests[name] = scaled.digests
                if scaled.copies:
                    found_copies[name] = scaled.copies
            else:
                timings |= python(
                    [name],
                    found,
                    scale=arguments.scale,
                    measure_memory=arguments.peak_memory,
                )
        found_profiles = profiles(names, found)
    print(f"\nmeasured in {duration(time.perf_counter() - started)}")
    if any(each.peak_memory_bytes is not None for each in timings.values()):
        print(format_speed_table(timings))
    shares(found_profiles)
    profiled = {
        name: {
            analysis: dataclasses.asdict(each) for analysis, each in analyses.items()
        }
        for name, analyses in found_profiles.items()
    }
    save(
        output,
        merged(output, timings, profiled, digests, views, copies=found_copies),
    )
    if baseline_path is not None and baseline_path.resolve() != output.resolve():
        before = {
            key: Timing.from_json(value)
            for key, value in load(baseline_path)["timings"].items()
        }
        against(timings, before, baseline_path.name)


if __name__ == "__main__":
    main()
