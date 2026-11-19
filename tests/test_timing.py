# Copyright 2026 Maxence Faldor
"""The speed suite's timing protocol: medians' intervals, rounds and verdicts."""

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import bench
import numpy as np
import pytest
from measuring import CONFIDENCE, Median, median, rank, usable_cores
from timing import Run

from jolty.simulate import pool


def coverage(count: int, k: int) -> float:
    """Computes the binomial coverage probability between the `k`-th order statistics.

    Sums binomial probabilities in floating point as an independent check alongside
    `rank`'s exact rational computation.
    """
    outside = sum(math.comb(count, below) for below in range(k)) / (1 << count)
    return 1 - 2 * outside


def test_the_rank_is_the_largest_whose_interval_holds_the_median_at_95() -> None:

    for count in range(1, 300):
        k = rank(count)
        if k is None:
            assert coverage(count, 1) < CONFIDENCE
            continue
        assert coverage(count, k) >= CONFIDENCE
        assert k + 1 > count // 2 or coverage(count, k + 1) < CONFIDENCE
    # Conover, "Practical Nonparametric Statistics", 3rd edition, 1999, table A3: the
    # ranks of the 95% interval of the median, 1 and 6 of 6, 2 and 9 of 10, 6 and 15
    # of 20, 40 and 61 of 100.
    assert [rank(count) for count in (5, 6, 10, 20, 100)] == [None, 1, 2, 6, 40]


def test_the_interval_holds_the_median_as_often_as_it_says() -> None:

    # Runs of an exponential distribution, skewed as times are, whose median is ln 2.
    generator = np.random.default_rng(20261116)
    trials, count = 20_000, 12
    held = 0
    for runs in generator.exponential(size=(trials, count)):
        found = median(runs.tolist())
        assert found.lower is not None
        assert found.upper is not None
        held += found.lower <= math.log(2) <= found.upper
    k = rank(count)
    assert k is not None
    # Within four standard deviations of the binomial count of trials that hold it,
    # around the interval's exact coverage, 96.1% for 12 runs, above the 95% stated.
    exact = coverage(count, k)
    assert exact >= CONFIDENCE
    spread = math.sqrt(trials * exact * (1 - exact))
    assert abs(held - trials * exact) <= 4 * spread


def test_too_few_runs_have_no_interval_and_separate_from_nothing() -> None:

    few, many = median([1.0, 2.0, 3.0]), median([10.0] * 6)
    assert (few.lower, few.upper) == (None, None)
    assert not few.separated(many)
    assert not many.separated(few)
    assert few.resolution(many) is None


def test_intervals_apart_are_separated_and_a_resolution_is_their_inner_widths() -> None:

    before = Median(1.0, 0.99, 1.02)
    slower = Median(1.1, 1.08, 1.11)
    touching = Median(1.03, 1.02, 1.04)
    assert before.separated(slower)
    assert slower.separated(before)
    assert not before.separated(touching)
    # Above: 0.02 of the first's and 0.02 of the second's; below: 0.01 and 0.01.
    # Default pytest.approx (rel=1e-6) covers binary float subtraction rounding.
    assert before.resolution(slower) == pytest.approx(0.04)


def noisy(level: float, seed: int, calls: list[str], name: str) -> Callable[[], Run]:
    """Builds a seeded mock workload with runtimes within 1 % of `level` seconds."""
    generator = np.random.default_rng(seed)

    def ran() -> Run:
        calls.append(name)
        seconds = level * (1 + 0.01 * generator.standard_normal())
        return Run(seconds, seconds)

    return ran


def test_rounds_alternate_their_order_and_name_only_a_separated_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    # No minimum time, so the rounds stop on their count and their intervals alone.
    monkeypatch.setattr(bench, "MIN_SECONDS", 0.0)
    calls: list[str] = []
    same = bench.alternated(
        [noisy(1.0, 1, calls, "a"), noisy(1.0, 2, calls, "b")], bench.MAX_ROUNDS
    )
    # A warm-up round, then rounds in the reverse order of the one before.
    assert calls[:6] == ["a", "b", "a", "b", "b", "a"]
    first, second = (each.wall_median() for each in same.timings)
    assert bench.decided(first, second) == ""
    assert len(same.loads) >= bench.MIN_ROUNDS
    # A 10% slowdown, ten times the runs' spread, is named.
    slower = bench.alternated(
        [noisy(1.0, 3, [], "a"), noisy(1.1, 4, [], "b")], bench.MAX_ROUNDS
    )
    before, now = (each.wall_median() for each in slower.timings)
    assert bench.decided(before, now) == "slower"
    assert bench.decided(now, before) == "faster"


def test_a_comparison_stops_undecided_at_its_max_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setattr(bench, "MIN_SECONDS", 0.0)
    generator = np.random.default_rng(5)

    def wild() -> Run:
        seconds = float(generator.uniform(0.5, 1.5))
        return Run(seconds, seconds)

    found = bench.alternated([wild, wild], bench.MIN_ROUNDS + 1)
    assert len(found.loads) == bench.MIN_ROUNDS + 1
    first, second = (each.wall_median() for each in found.timings)
    assert bench.decided(first, second) == "undecided"


def test_a_time_written_reads_back_and_one_of_an_older_file_has_no_interval() -> None:

    runs = [Run(1.0 + index / 100, 2.0) for index in range(6)]
    timing = bench.Timing.from_samples(runs, 1.5, 2.5)
    assert bench.Timing.from_json(timing.to_json()) == timing
    assert timing.wall_median().lower == 1.0
    older = bench.Timing.from_json({"seconds": 1.0, "runs": 10, "load": 3.0})
    assert older.wall_median().lower is None
    assert older.cpu_median() is None


def test_a_night_compares_its_build_with_the_last_and_keeps_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tonight, last, older = "b" * 40, "a" * 40, "c" * 40
    compared: list[tuple[tuple[str, str], Path]] = []

    def compare(
        revisions: tuple[str, str],
        names: object,
        found: object,
        arguments: object,
        output: Path,
    ) -> list[str]:
        del names, found, arguments
        compared.append((revisions, output))
        output.parent.mkdir(parents=True, exist_ok=True)
        builds = [{"revision": each, "commit": each} for each in revisions]
        output.write_text(json.dumps({"builds": builds}), encoding="utf-8")
        return []

    removed: list[str] = []

    def git(*arguments: str) -> str:
        if arguments[:2] == ("worktree", "remove"):
            removed.append(Path(arguments[-1]).name)
        return ""

    monkeypatch.setattr(bench, "compared", compare)
    monkeypatch.setattr(bench, "fixtures", dict)
    monkeypatch.setattr(bench, "commit", lambda: tonight)
    monkeypatch.setattr(bench, "git", git)
    builds = tmp_path / "builds"
    for name in (tonight, f"{tonight}-2", last):
        (builds / name).mkdir(parents=True)
    arguments = bench.Bench.model_construct(builds=builds)
    nights = tmp_path / "nights"
    # The first night builds its commit twice, which must name no change.
    bench.nightly(arguments, nights)
    today = nights / time.strftime("%Y-%m-%d") / "compare.json"
    assert compared == [((tonight, tonight), today)]
    assert sorted(removed) == sorted([f"{tonight}-2", last])
    # A later night compares with the second build of the newest earlier night.
    for night, commit in (("2026-11-14", older), ("2026-11-15", last)):
        (nights / night).mkdir()
        builds_then = [{"commit": "d" * 40}, {"commit": commit}]
        written = json.dumps({"builds": builds_then})
        (nights / night / "compare.json").write_text(written, encoding="utf-8")
    compared.clear()
    bench.nightly(arguments, nights)
    assert compared == [((last, tonight), today)]


def test_usable_cores_are_the_engines_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    assert usable_cores() == pool().threads >= 1

    @dataclass(frozen=True)
    class Pool:
        threads: int

    monkeypatch.setattr("measuring.pool", lambda: Pool(16))
    assert usable_cores() == 16
    assert bench.every((1, 2, 4, 8, 32, 64)) == [1, 2, 4, 8, 16]
