# Copyright 2026 Maxence Faldor
"""Deterministic counts, precision digits and result digests checked per platform.

Records what is independent of machine load and wall-clock time at default options:

- Every non-scale speed fixture (`bjt_amp`, `ac_wide`, `buck`, `inverter_chain_500`,
  `rc_mesh_100`): per-analysis Newton iterations, accepted/rejected/landed steps,
  factorizations, refactorizations, fallbacks, reuses, DC strategies and a SHA-256
  digest of the solved values.
- The 14 precision fixtures under 1,000 nodes (`rc_step`, `rc_ramp`, `rc_femto`,
  `rlc_step`, `lc_tank`, `diode`, `rlc_ac`, `rc_noise`, `butterworth_poles`,
  `rc_periodic`, `ladder_10`, `ladder_100`, `tree_10`, `tree_100`): solver counts
  and exact significant correct digits against mpmath/closed-form references.

Because `libm` (`exp`, `ln`, `pow`) rounds differently across `glibc` versions and
operating systems (`plans/accuracy.md` item 13), records are keyed by
`platform_key()` (e.g. `linux-x86_64-glibc2.39`, `linux-x86_64-glibc2.27`,
`darwin-arm64`). When the active platform has an exact recorded entry, every count,
digest and rounded digit must match exactly. On an unrecorded platform, linear
and analytic fixtures are checked tightly while chaotic multi-step nonlinear fixtures
(`buck`, `inverter_chain_500`) are checked within a bounded relative margin unless
`--strict-platform` is enabled.

Run from the repository root:

    uv run python benchmarks/deterministic/bench.py
    uv run python benchmarks/deterministic/bench.py --record
"""

import hashlib
import json
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import numpy as np
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "precision"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "speed"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "speed" / "circuits"))

from fixtures import fixtures as precision_fixtures
from suite import DEFAULTS, Prepared, Setting, run_jolty, score
from timing import load_fixture, planned

from jolty import Simulation, cli, results

BENCHMARKS: Final = Path(__file__).resolve().parents[1]
ROOT: Final = Path(__file__).resolve().parents[2]
PRECISION_DIRECTORY: Final = BENCHMARKS / "precision"
SPEED_DIRECTORY: Final = BENCHMARKS / "speed"
CIRCUITS_DIRECTORY: Final = SPEED_DIRECTORY / "circuits"

BASELINE_PATH: Final = Path(__file__).resolve().with_name("baseline.json")
"""Committed per-platform baseline of deterministic counts, digits and digests."""

BASELINE_VERSION: Final = 2
"""Schema version of the committed deterministic baseline JSON file."""

PRECISION_NAMES: Final = (
    "rc_step",
    "rc_ramp",
    "rc_femto",
    "rlc_step",
    "lc_tank",
    "diode",
    "rlc_ac",
    "rc_noise",
    "butterworth_poles",
    "rc_periodic",
    "ladder_10",
    "ladder_100",
    "tree_10",
    "tree_100",
)
"""The 14 precision fixtures under 1,000 nodes checked in CI."""

SPEED_NAMES: Final = (
    "bjt_amp",
    "ac_wide",
    "buck",
    "inverter_chain_500",
    "rc_mesh_100",
)
"""The five non-scale speed fixtures checked at default options."""

NONLINEAR_TRANSIENT_FIXTURES: Final = frozenset({"buck", "inverter_chain_500"})
"""Fixtures where 1-ULP `libm` rounding differences across `glibc` versions shift
adaptive step boundaries on unrecorded platforms (observed: 0.8% on `buck` and
3.3% on `inverter_chain_500` between glibc 2.27 and glibc 2.43)."""

DIGIT_PRECISION: Final = 4
"""Decimal places kept when storing significant correct digits."""

CROSS_PLATFORM_DIGIT_MARGIN: Final = 0.05
"""Digit tolerance on an unrecorded platform/libm to cover 1-ULP rounding."""

CROSS_PLATFORM_COUNT_RATIO: Final = 0.06
"""Relative count band for `NONLINEAR_TRANSIENT_FIXTURES` on unrecorded platforms:
derived from the 3.3% iteration shift observed between glibc 2.27 and glibc 2.43 on
`inverter_chain_500`, with margin for macOS `libSystem`."""


def platform_key() -> str:
    """Stable identifier for the OS, architecture and C library (`libm`).

    Examples: `linux-x86_64-glibc2.39`, `linux-x86_64-glibc2.27`, `darwin-arm64`.
    """
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux":
        libc_name, libc_version = platform.libc_ver()
        if libc_name and libc_version:
            return f"{system}-{machine}-{libc_name.lower()}{libc_version}"
    return f"{system}-{machine}"


def _result_digest(result: results.AnalysisResult) -> str:
    """Short SHA-256 digest of an analysis result's numerical arrays."""
    hasher = hashlib.sha256()
    hasher.update(type(result).__name__.encode("utf-8"))
    match result:
        case results.OperatingPointResult(values=values):
            hasher.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())
        case results.DcSweepResult(swept=swept, values=values):
            hasher.update(np.ascontiguousarray(swept, dtype=np.float64).tobytes())
            hasher.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())
        case results.AcResult(frequencies=frequencies, values=values):
            hasher.update(np.ascontiguousarray(frequencies, dtype=np.float64).tobytes())
            hasher.update(np.ascontiguousarray(values, dtype=np.complex128).tobytes())
        case results.PeriodicSteadyStateResult(
            times=times, values=values, period=period
        ):
            hasher.update(np.array([period], dtype=np.float64).tobytes())
            hasher.update(np.ascontiguousarray(times, dtype=np.float64).tobytes())
            hasher.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())
        case results.TransientResult(times=times, values=values):
            hasher.update(np.ascontiguousarray(times, dtype=np.float64).tobytes())
            hasher.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())
        case results.NoiseResult(frequencies=frequencies, densities=densities):
            hasher.update(np.ascontiguousarray(frequencies, dtype=np.float64).tobytes())
            hasher.update(np.ascontiguousarray(densities, dtype=np.float64).tobytes())
        case results.PoleZeroResult(poles=poles, zeros=zeros):
            hasher.update(np.ascontiguousarray(poles, dtype=np.complex128).tobytes())
            if zeros is not None:
                hasher.update(
                    np.ascontiguousarray(zeros, dtype=np.complex128).tobytes()
                )
    return hasher.hexdigest()[:16]


class AnalysisCounts(BaseModel):
    """Deterministic solver counts extracted from one `AnalysisResult`."""

    strategy: str | None = None
    iterations: int | None = None
    accepted: int | None = None
    rejected: int | None = None
    landed: int | None = None
    transient_iterations: int | None = None
    work_iterations: int | None = None
    factorizations: int | None = None
    refactorizations: int | None = None
    fallbacks: int | None = None
    reuses: int | None = None
    periods: int | None = None
    digest: str


def _optional_int(value: int | None) -> int | None:
    """Converts an optional integer counter to `int | None`."""
    return None if value is None else int(value)


def extract_counts(result: results.AnalysisResult) -> AnalysisCounts:
    """Reads every deterministic solver counter from `result`."""
    digest = _result_digest(result)
    match result:
        case results.OperatingPointResult():
            return AnalysisCounts(
                strategy=str(result.strategy),
                iterations=_optional_int(result.iterations),
                digest=digest,
            )
        case results.DcSweepResult():
            return AnalysisCounts(
                iterations=int(sum(result.iterations)),
                digest=digest,
            )
        case results.TransientResult() | results.PeriodicSteadyStateResult():
            statistics = result.statistics
            work = statistics.work
            periods = (
                int(result.periods)
                if isinstance(result, results.PeriodicSteadyStateResult)
                else None
            )
            return AnalysisCounts(
                strategy=str(statistics.initial),
                accepted=int(statistics.accepted),
                rejected=int(statistics.rejected),
                landed=int(statistics.landed),
                transient_iterations=int(statistics.iterations),
                work_iterations=None if work is None else int(work.iterations),
                factorizations=None if work is None else int(work.factorizations),
                refactorizations=None if work is None else int(work.refactorizations),
                fallbacks=None if work is None else int(work.fallbacks),
                reuses=None if work is None else int(work.reuses),
                periods=periods,
                digest=digest,
            )
        case results.AcResult() | results.NoiseResult() | results.PoleZeroResult():
            operating_point = result.operating_point
            return AnalysisCounts(
                strategy=str(operating_point.strategy),
                iterations=_optional_int(operating_point.iterations),
                digest=digest,
            )
    return AnalysisCounts(digest=digest)


class PrecisionRecord(BaseModel):
    """Deterministic counts and scored digits for one small precision fixture."""

    counts: dict[str, int] = Field(default_factory=dict)
    digits: dict[str, float] = Field(default_factory=dict)


class PlatformRecord(BaseModel):
    """Deterministic counts, digits and digests recorded on one platform."""

    precision: dict[str, PrecisionRecord] = Field(default_factory=dict)
    speed: dict[str, dict[str, AnalysisCounts]] = Field(default_factory=dict)


class DeterministicBaseline(BaseModel):
    """Committed per-platform baseline (`benchmarks/deterministic/baseline.json`)."""

    version: int = BASELINE_VERSION
    platforms: dict[str, PlatformRecord] = Field(default_factory=dict)


def measure_precision_fixture(name: str) -> PrecisionRecord:
    """Runs one small precision fixture at default options and scores its digits."""
    fixture = precision_fixtures()[name]
    setting = Setting("defaults", DEFAULTS["relative_tolerance"])
    prepared = Prepared.from_fixture(fixture)
    measured = run_jolty(prepared, setting, 1)
    compared = fixture.compare(measured.found)
    digits = {
        output.name: round(
            float(score(compared[output.name], output)[output.score].digits),
            DIGIT_PRECISION,
        )
        for output in fixture.outputs
    }
    counts = {key: int(value) for key, value in sorted(measured.counts.items())}
    return PrecisionRecord(counts=counts, digits=digits)


def measure_speed_fixture(name: str) -> dict[str, AnalysisCounts]:
    """Runs one speed fixture at default options and records counts per analysis."""
    loaded = load_fixture(planned()[name])
    simulation = Simulation(loaded.circuit)
    options = loaded.plan.options
    recorded: dict[str, AnalysisCounts] = {}
    for analysis_name, analysis in loaded.plan.analyses.items():
        result = simulation.run(analysis, options)
        recorded[analysis_name] = extract_counts(result)
    return recorded


def measure_platform(
    *,
    precision_names: Sequence[str] = PRECISION_NAMES,
    speed_names: Sequence[str] = SPEED_NAMES,
    workers: int = 4,
) -> PlatformRecord:
    """Measures all deterministic precision and speed records in parallel."""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        precision_futures = {
            name: executor.submit(measure_precision_fixture, name)
            for name in precision_names
        }
        speed_futures = {
            name: executor.submit(measure_speed_fixture, name) for name in speed_names
        }
        precision = {name: precision_futures[name].result() for name in precision_names}
        speed = {name: speed_futures[name].result() for name in speed_names}
    return PlatformRecord(precision=precision, speed=speed)


def _compare_counts_exact(
    prefix: str, expected: AnalysisCounts, actual: AnalysisCounts
) -> list[str]:
    """Compares every counter and digest of one analysis strictly."""
    differences: list[str] = []
    for field_name in AnalysisCounts.model_fields:
        before = getattr(expected, field_name)
        now = getattr(actual, field_name)
        if before != now:
            differences.append(
                f"{prefix}.{field_name}: recorded {before!r}, measured {now!r}"
            )
    return differences


def _compare_counts_bounded(
    prefix: str,
    expected: AnalysisCounts,
    actual: AnalysisCounts,
    ratio: float,
) -> list[str]:
    """Compares numeric counts within `ratio` on unrecorded platforms."""
    differences: list[str] = []
    for field_name in (
        "iterations",
        "accepted",
        "rejected",
        "transient_iterations",
        "work_iterations",
        "factorizations",
        "refactorizations",
    ):
        before = getattr(expected, field_name)
        now = getattr(actual, field_name)
        if before is None or now is None:
            if before != now:
                differences.append(
                    f"{prefix}.{field_name}: recorded {before!r}, measured {now!r}"
                )
            continue
        limit = 0 if ratio == 0.0 else max(2, math.ceil(abs(before) * ratio))
        if abs(now - before) > limit:
            suffix = "" if limit == 0 else f" (outside +/-{limit})"
            differences.append(
                f"{prefix}.{field_name}: recorded {before}, measured {now}{suffix}"
            )
    return differences


def _compare_precision_entry(
    name: str,
    before: PrecisionRecord,
    now: PrecisionRecord,
    *,
    exact: bool,
) -> list[str]:
    """Compares counts and scored digits for one precision fixture."""
    differences = [
        f"precision/{name}.counts[{key!r}]: "
        f"recorded {before.counts.get(key)!r}, measured {now.counts.get(key)!r}"
        for key in sorted(set(before.counts) | set(now.counts))
        if before.counts.get(key) != now.counts.get(key)
    ]
    for output in sorted(set(before.digits) | set(now.digits)):
        recorded_digits = before.digits.get(output)
        measured_digits = now.digits.get(output)
        if recorded_digits is None or measured_digits is None:
            differences.append(
                f"precision/{name}.digits[{output!r}]: "
                f"recorded {recorded_digits!r}, measured {measured_digits!r}"
            )
        elif exact:
            if recorded_digits != measured_digits:
                differences.append(
                    f"precision/{name}.digits[{output!r}]: "
                    f"recorded {recorded_digits:.4f}, "
                    f"measured {measured_digits:.4f}"
                )
        elif abs(measured_digits - recorded_digits) > CROSS_PLATFORM_DIGIT_MARGIN:
            differences.append(
                f"precision/{name}.digits[{output!r}]: "
                f"recorded {recorded_digits:.4f}, measured {measured_digits:.4f} "
                f"(margin {CROSS_PLATFORM_DIGIT_MARGIN:.2f})"
            )
    return differences


def _compare_precision(
    expected: Mapping[str, PrecisionRecord],
    actual: Mapping[str, PrecisionRecord],
    *,
    exact: bool,
) -> list[str]:
    """Compares precision counts and digits."""
    differences: list[str] = []
    for name in sorted(set(expected) | set(actual)):
        if name not in expected:
            differences.append(f"precision/{name}: missing from baseline")
        elif name not in actual:
            differences.append(f"precision/{name}: missing from measured run")
        else:
            differences.extend(
                _compare_precision_entry(
                    name, expected[name], actual[name], exact=exact
                )
            )
    return differences


def _compare_speed(
    expected: Mapping[str, Mapping[str, AnalysisCounts]],
    actual: Mapping[str, Mapping[str, AnalysisCounts]],
    *,
    exact: bool,
) -> list[str]:
    """Compares speed fixture counts and digests."""
    differences: list[str] = []
    for name in sorted(set(expected) | set(actual)):
        if name not in expected:
            differences.append(f"speed/{name}: missing from baseline")
            continue
        if name not in actual:
            differences.append(f"speed/{name}: missing from measured run")
            continue
        before_map, now_map = expected[name], actual[name]
        for analysis in sorted(set(before_map) | set(now_map)):
            if analysis not in before_map or analysis not in now_map:
                differences.append(
                    f"speed/{name}/{analysis}: recorded keys {sorted(before_map)}, "
                    f"measured keys {sorted(now_map)}"
                )
                continue
            prefix = f"speed/{name}/{analysis}"
            if exact:
                differences.extend(
                    _compare_counts_exact(
                        prefix, before_map[analysis], now_map[analysis]
                    )
                )
            elif name in NONLINEAR_TRANSIENT_FIXTURES:
                differences.extend(
                    _compare_counts_bounded(
                        prefix,
                        before_map[analysis],
                        now_map[analysis],
                        CROSS_PLATFORM_COUNT_RATIO,
                    )
                )
            else:
                differences.extend(
                    _compare_counts_bounded(
                        prefix, before_map[analysis], now_map[analysis], 0.0
                    )
                )
    return differences


def compare_records(
    expected: PlatformRecord,
    actual: PlatformRecord,
    *,
    exact: bool = True,
) -> list[str]:
    """Returns every factual mismatch between `expected` and `actual`."""
    return [
        *_compare_precision(expected.precision, actual.precision, exact=exact),
        *_compare_speed(expected.speed, actual.speed, exact=exact),
    ]


def load_baseline(path: Path = BASELINE_PATH) -> DeterministicBaseline:
    """Reads and validates the committed deterministic baseline file."""
    return DeterministicBaseline.model_validate_json(path.read_text(encoding="utf-8"))


def save_baseline(baseline: DeterministicBaseline, path: Path = BASELINE_PATH) -> None:
    """Writes `baseline` deterministically with sorted keys."""
    ordered = DeterministicBaseline(
        version=baseline.version,
        platforms=dict(sorted(baseline.platforms.items())),
    )
    payload = ordered.model_dump(mode="json", exclude_none=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def select_reference(
    baseline: DeterministicBaseline, current: str
) -> tuple[str, PlatformRecord, bool] | None:
    """Returns `(key, record, exact)` matching `current` or a cross-platform entry."""
    if current in baseline.platforms:
        return current, baseline.platforms[current], True
    if not baseline.platforms:
        return None
    same_os = [
        key for key in baseline.platforms if key.startswith(current.split("-", 1)[0])
    ]
    chosen = min(same_os) if same_os else min(baseline.platforms)
    return chosen, baseline.platforms[chosen], False


class Deterministic(cli.CommandLine):
    """Checks or records deterministic solver counts, digits and digests."""

    record: bool = Field(
        default=False,
        description="Record the current platform's counts and digits in the baseline.",
    )
    strict_platform: bool = Field(
        default=False,
        description="Fail when the current platform key has no exact recorded entry.",
    )
    path: Path = Field(
        default=BASELINE_PATH,
        description="Path to the deterministic baseline JSON file.",
    )
    workers: int = Field(
        default=4,
        ge=1,
        description="Concurrent worker threads measuring fixtures.",
    )

    def cli_cmd(self) -> None:
        """Executes `--record` or checks against the recorded baseline."""
        key = platform_key()
        measured = measure_platform(workers=self.workers)
        if self.record:
            baseline = (
                load_baseline(self.path)
                if self.path.is_file()
                else DeterministicBaseline()
            )
            baseline.platforms[key] = measured
            save_baseline(baseline, self.path)
            print(f"recorded deterministic baseline for {key} in {self.path}")
            return
        if not self.path.is_file():
            message = f"deterministic baseline file not found: {self.path}"
            raise SystemExit(message)
        baseline = load_baseline(self.path)
        selected = select_reference(baseline, key)
        if selected is None:
            message = f"deterministic baseline {self.path} holds no platform entries"
            raise SystemExit(message)
        reference_key, expected, exact = selected
        if self.strict_platform and not exact:
            known = ", ".join(sorted(baseline.platforms))
            message = (
                f"no deterministic baseline is recorded for `{key}`; "
                f"recorded platforms: {known}"
            )
            raise SystemExit(message)
        differences = compare_records(expected, measured, exact=exact)
        if differences:
            mode = "exact" if exact else f"cross-platform against {reference_key}"
            header = (
                f"{len(differences)} deterministic mismatch(es) on `{key}` ({mode}):"
            )
            raise SystemExit("\n".join([header, *differences]))
        mode = "exact" if exact else f"bounded against {reference_key}"
        print(
            f"deterministic check passed on {key} ({mode}): "
            f"{len(measured.precision)} precision fixtures, "
            f"{len(measured.speed)} speed fixtures"
        )


def main() -> None:
    """CLI entry point."""
    cli.run(Deterministic)


if __name__ == "__main__":
    main()
