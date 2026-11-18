# Copyright 2026 Maxence Faldor
"""Tests for the per-platform deterministic benchmark baseline checker."""

import json
import re
from pathlib import Path

import deterministic
import numpy as np
import pytest

from jolty import results, unknowns

ADVICE_WORDS = re.compile(
    r"\b(use|pass|try|instead|should|give|did you mean|please|consider)\b",
    re.IGNORECASE,
)


def _sample_record() -> deterministic.PlatformRecord:
    """Minimal self-contained baseline record exercising precision and speed checks."""
    return deterministic.PlatformRecord(
        precision={
            "rc_step": deterministic.PrecisionRecord(
                counts={"accepted": 42, "iterations": 84},
                digits={"out": 3.4567},
            ),
        },
        speed={
            "bjt_amp": {
                "op": deterministic.AnalysisCounts(
                    strategy="direct",
                    iterations=5,
                    fingerprint="1111111111111111",
                ),
                "tran": deterministic.AnalysisCounts(
                    strategy="direct",
                    accepted=106,
                    rejected=0,
                    landed=0,
                    transient_iterations=159,
                    work_iterations=159,
                    factorizations=30,
                    refactorizations=0,
                    fallbacks=0,
                    reuses=129,
                    fingerprint="2222222222222222",
                ),
            },
            "buck": {
                "tran": deterministic.AnalysisCounts(
                    strategy="direct",
                    accepted=1000,
                    rejected=10,
                    landed=5,
                    transient_iterations=2000,
                    work_iterations=2000,
                    factorizations=500,
                    refactorizations=20,
                    fallbacks=0,
                    reuses=1480,
                    fingerprint="3333333333333333",
                ),
            },
        },
    )


def test_exact_match_passes() -> None:
    """Identical records produce zero mismatches in both exact and bounded modes."""
    record = _sample_record()
    assert deterministic.compare_records(record, record, exact=True) == []
    assert deterministic.compare_records(record, record, exact=False) == []


def test_exact_mismatch_reports_all_facts_without_advice() -> None:
    """Every count, digit, fingerprint and missing-entry difference is reported."""
    expected = _sample_record()
    actual = expected.model_copy(deep=True)
    actual.precision["rc_step"].counts["iterations"] += 1
    actual.precision["rc_step"].digits["out"] -= 0.01
    actual.speed["bjt_amp"]["tran"].factorizations = 31
    actual.speed["bjt_amp"]["tran"].fingerprint = "ffffffffffffffff"
    actual.speed["bjt_amp"]["extra_ac"] = deterministic.AnalysisCounts(
        strategy="direct",
        iterations=3,
        fingerprint="4444444444444444",
    )
    del actual.speed["buck"]

    differences = deterministic.compare_records(expected, actual, exact=True)
    joined = "\n".join(differences)
    assert "precision/rc_step.counts['iterations']: recorded 84, measured 85" in joined
    assert "precision/rc_step.digits['out']: recorded 3.4567, measured 3.4467" in joined
    assert "speed/bjt_amp/tran.factorizations: recorded 30, measured 31" in joined
    assert (
        "speed/bjt_amp/tran.fingerprint: recorded '2222222222222222', "
        "measured 'ffffffffffffffff'"
    ) in joined
    assert "speed/bjt_amp/extra_ac:" in joined
    assert "speed/buck: missing from measured run" in joined
    for line in differences:
        assert not ADVICE_WORDS.search(line), f"advice word in message: {line}"


def test_cross_platform_rules_distinguish_linear_and_nonlinear_fixtures() -> None:
    """Unrecorded platforms enforce exact linear counts and bounded nonlinear shifts."""
    expected = _sample_record()
    within = expected.model_copy(deep=True)
    # Different libm fingerprint is ignored on unrecorded platforms, while linear
    # fixture counts stay exact (0 tolerance).
    within.speed["bjt_amp"]["tran"].fingerprint = "9999999999999999"
    # Nonlinear fixture (`buck`) allows +/- 6% shifts (1000 -> 1050 = +5%).
    within.speed["buck"]["tran"].accepted = 1050
    within.speed["buck"]["tran"].transient_iterations = 2100
    # Precision digits allow +/- 0.05 margin across platforms.
    within.precision["rc_step"].digits["out"] = 3.4567 - 0.04
    assert deterministic.compare_records(expected, within, exact=False) == []

    # Linear fixture count shift by +1 fails even cross-platform.
    linear_shifted = expected.model_copy(deep=True)
    linear_shifted.speed["bjt_amp"]["tran"].accepted = 107
    linear_differences = deterministic.compare_records(
        expected, linear_shifted, exact=False
    )
    assert len(linear_differences) == 1
    assert (
        "speed/bjt_amp/tran.accepted: recorded 106, measured 107"
        in linear_differences[0]
    )

    # Nonlinear fixture shift beyond 6% fails cross-platform.
    nonlinear_shifted = expected.model_copy(deep=True)
    nonlinear_shifted.speed["buck"]["tran"].accepted = 1061
    nonlinear_differences = deterministic.compare_records(
        expected, nonlinear_shifted, exact=False
    )
    assert len(nonlinear_differences) == 1
    assert "speed/buck/tran.accepted" in nonlinear_differences[0]

    # Precision digit shift beyond 0.05 fails cross-platform.
    digit_shifted = expected.model_copy(deep=True)
    digit_shifted.precision["rc_step"].digits["out"] = 3.4567 - 0.06
    digit_differences = deterministic.compare_records(
        expected, digit_shifted, exact=False
    )
    assert len(digit_differences) == 1
    assert "precision/rc_step.digits['out']" in digit_differences[0]


def test_select_reference_prefers_exact_then_same_os() -> None:
    """Platform selection prefers exact key, then same OS, then lexicographic first."""
    record = _sample_record()
    empty = deterministic.DeterministicBaseline()
    assert deterministic.select_reference(empty, "linux-x86_64-glibc2.43") is None

    baseline = deterministic.DeterministicBaseline(
        platforms={
            "darwin-arm64": record,
            "linux-x86_64-glibc2.39": record,
            "linux-x86_64-glibc2.27": record,
        }
    )
    exact = deterministic.select_reference(baseline, "linux-x86_64-glibc2.39")
    assert exact is not None
    assert (exact[0], exact[2]) == ("linux-x86_64-glibc2.39", True)

    same_os = deterministic.select_reference(baseline, "linux-x86_64-glibc2.43")
    assert same_os is not None
    assert (same_os[0], same_os[2]) == ("linux-x86_64-glibc2.27", False)

    fallback = deterministic.select_reference(baseline, "windows-amd64")
    assert fallback is not None
    assert (fallback[0], fallback[2]) == ("darwin-arm64", False)


def test_save_and_load_baseline_round_trip(tmp_path: Path) -> None:
    """Baseline JSON saves deterministically with sorted keys and reloads cleanly."""
    path = tmp_path / "deterministic.json"
    record = _sample_record()
    baseline = deterministic.DeterministicBaseline(
        platforms={"linux-x86_64-glibc2.39": record, "darwin-arm64": record}
    )
    deterministic.save_baseline(baseline, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert list(raw["platforms"].keys()) == ["darwin-arm64", "linux-x86_64-glibc2.39"]
    reloaded = deterministic.load_baseline(path)
    assert reloaded == baseline


def test_periodic_steady_state_fingerprint_includes_period() -> None:
    """`PeriodicSteadyStateResult` hashes its period, unlike `TransientResult`."""
    node = (unknowns.Unknown("out", "voltage"),)
    times = np.array([0.0, 1e-6], dtype=np.float64)
    values = np.array([[0.0], [1.0]], dtype=np.float64)
    statistics = results.Statistics(
        initial="direct", accepted=2, rejected=0, landed=0, iterations=4
    )
    first = results.PeriodicSteadyStateResult(
        unknowns=node,
        times=times,
        values=values,
        statistics=statistics,
        period=1e-6,
        periods=3,
    )
    second = results.PeriodicSteadyStateResult(
        unknowns=node,
        times=times,
        values=values,
        statistics=statistics,
        period=2e-6,
        periods=3,
    )
    transient = results.TransientResult(
        unknowns=node,
        times=times,
        values=values,
        statistics=statistics,
    )
    first_counts = deterministic.extract_counts(first)
    second_counts = deterministic.extract_counts(second)
    transient_counts = deterministic.extract_counts(transient)
    assert first_counts.periods == 3
    assert first_counts.fingerprint != second_counts.fingerprint
    assert first_counts.fingerprint != transient_counts.fingerprint


def test_cli_strict_platform_refuses_unrecorded_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--strict-platform` exits with a factual message when the key is absent."""
    path = tmp_path / "deterministic.json"
    record = _sample_record()
    deterministic.save_baseline(
        deterministic.DeterministicBaseline(platforms={"darwin-arm64": record}),
        path,
    )
    monkeypatch.setattr(deterministic, "platform_key", lambda: "linux-x86_64-glibc2.99")
    monkeypatch.setattr(deterministic, "measure_platform", lambda **_kwargs: record)
    command = deterministic.Deterministic(
        record=False, strict_platform=True, path=path, workers=1
    )
    with pytest.raises(SystemExit) as caught:
        command.cli_cmd()
    message = str(caught.value)
    assert "linux-x86_64-glibc2.99" in message
    assert "darwin-arm64" in message
    assert not ADVICE_WORDS.search(message), f"advice word in message: {message}"


def test_committed_baseline_covers_every_declared_fixture() -> None:
    """`benchmarks/deterministic.json` holds every declared precision/speed fixture."""
    baseline = deterministic.load_baseline()
    assert baseline.platforms, "baseline must hold at least one platform entry"
    for key, record in baseline.platforms.items():
        assert set(record.precision) == set(deterministic.PRECISION_NAMES), key
        assert set(record.speed) == set(deterministic.SPEED_NAMES), key
        for name, precision_entry in record.precision.items():
            assert precision_entry.digits, f"{key}/precision/{name} missing digits"
        for name, analyses in record.speed.items():
            assert analyses, f"{key}/speed/{name} missing analyses"
            for analysis, counts in analyses.items():
                assert len(counts.fingerprint) == 16, f"{key}/speed/{name}/{analysis}"
