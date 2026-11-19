# Copyright 2026 Maxence Faldor
"""Shared provenance, I/O, formatting, and statistics helpers for benchmark suites.

Every suite under ``benchmarks/`` runs on one host at one commit and writes a results
file headed by its `Provenance` (commit hash with optional ``-dirty`` suffix plus host
metadata). Core helpers cover suite identifiers (`Suite`), host metrics (`load_average`,
`usable_cores`, `commit`, `machine`, `provenance`), and JSON storage (`results_path`,
`results_directory`, `format_json`, `save`, `load`). Additional utilities format times
(`duration`), memory (`memory`, `ru_maxrss_to_bytes`, `peak_memory_bytes`,
`available_memory_bytes`), tables (`markdown_table`, `aligned_row`, `aligned_table`),
and distribution-free median intervals (`median`, `rank`, `CONFIDENCE`).
"""

import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, JsonValue

from jolty.simulate import pool

BENCHMARKS: Final = Path(__file__).resolve().parent
"""Root directory of the ``benchmarks/`` tree, containing one subdirectory per suite."""

ROOT: Final = BENCHMARKS.parent
"""Root directory of the repository."""


class Suite(StrEnum):
    """Benchmark suite identifier, matching its directory name under ``benchmarks/``."""

    SPEED = "speed"
    PRECISION = "precision"
    ROBUSTNESS = "robustness"
    COVERAGE = "coverage"
    AGENTS = "agents"


DIRTY: Final = "-dirty"
"""Suffix appended to a commit hash when tracked files have uncommitted changes."""

KIBIBYTE: Final = 1024
"""Bytes per kibibyte, matching Linux ``getrusage(2)`` and ``/proc`` units."""

_MEMORY_UNITS: Final = (
    (1 << 40, "TiB"),
    (1 << 30, "GiB"),
    (1 << 20, "MiB"),
    (1 << 10, "KiB"),
)
"""IEC binary prefixes for `memory`, ordered from largest to smallest."""

_DARWIN: Final = "darwin"
"""macOS `sys.platform` string, where ``getrusage(2)`` reports raw bytes."""

_PROC_STATUS: Final = Path("/proc/self/status")
"""Linux process status file containing `VmHWM` (`proc_pid_status(5)`)."""

_PROC_MEMINFO: Final = Path("/proc/meminfo")
"""Linux memory summary file (`proc_meminfo(5)`)."""

_PROC_FIELDS: Final = 2
"""Minimum number of whitespace-separated tokens on a Linux ``/proc`` key-value line."""


def load_average() -> float:
    """Returns the machine's one-minute load average."""
    return os.getloadavg()[0]


def usable_cores() -> int:
    """Returns the default thread count used by the engine, one per usable processor.

    Matches Rust's `available_parallelism` count (respecting CPU affinity and
    container CPU quotas) so Python benchmarks and engine batches agree on cores.
    """
    return pool().threads


def git(*arguments: str) -> str:
    """Runs git with `arguments` in the repository root and returns stripped stdout."""
    done = subprocess.run(  # noqa: S603 - git reads the repository
        ["git", *arguments],  # noqa: S607 - git is on the path
        capture_output=True,
        check=True,
        cwd=ROOT,
        text=True,
    )
    return done.stdout.strip()


def commit() -> str:
    """Returns the checked-out commit hash, adding `DIRTY` if tracked files differ."""
    dirty = git("status", "--porcelain", "--untracked-files=no") != ""
    return git("rev-parse", "HEAD") + (DIRTY if dirty else "")


def _sysctl(name: str) -> str | None:
    """Returns the macOS kernel setting `name`, or None when unavailable."""
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


def machine() -> dict[str, JsonValue]:
    """Collects host CPU, memory, OS, Python, and Rust metadata."""
    cpu = _sysctl("machdep.cpu.brand_string")
    sockets = _sysctl("hw.packages")
    total_memory = _sysctl("hw.memsize")
    # macOS names each kind of core, such as Apple silicon's performance and efficiency
    # cores, and counts them.
    kinds = int(_sysctl("hw.nperflevels") or 0)
    levels = (f"hw.perflevel{level}" for level in range(kinds))
    cores = {
        _sysctl(f"{level}.name"): _sysctl(f"{level}.physicalcpu") for level in levels
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
    if total_memory is None and _PROC_MEMINFO.is_file():
        first = _PROC_MEMINFO.read_text(encoding="utf-8").splitlines()[0]
        total_memory = str(int(first.split()[1]) * KIBIBYTE)
    try:
        rustc = subprocess.run(
            ["rustc", "--version"],  # noqa: S607 - rustc is on the path
            capture_output=True,
            check=True,
            cwd=ROOT,
            text=True,
        )
        rustc_version: str | None = rustc.stdout.strip()
    except OSError, subprocess.CalledProcessError:
        rustc_version = None
    return {
        "cpu": cpu or platform.processor(),
        "cores": usable_cores(),
        "kinds of core": {str(kind): int(count or 0) for kind, count in cores.items()},
        "sockets": None if sockets is None else int(sockets),
        "memory": None if total_memory is None else int(total_memory),
        "system": platform.platform(),
        "python": platform.python_version(),
        "rustc": rustc_version,
    }


class Provenance(BaseModel):
    """Measurement provenance: commit hash (with optional `DIRTY`) and host metadata.

    Every benchmark suite records a `Provenance` header in its results file.
    """

    commit: str
    machine: dict[str, JsonValue]


def provenance() -> Provenance:
    """Builds a `Provenance` record for the current commit and host machine."""
    return Provenance(commit=commit(), machine=machine())


def results_directory(suite: Suite) -> Path:
    """Returns the results directory for `suite`."""
    return BENCHMARKS / suite / "results"


def results_path(suite: Suite) -> Path:
    """Returns the ``<commit>.json`` results path for `suite` at HEAD.

    Includes the `DIRTY` suffix when tracked files have changes, placed inside
    `results_directory(suite)`.
    """
    return results_directory(suite) / f"{commit()}.json"


def shown(path: Path) -> Path:
    """Returns `path` relative to `ROOT` when inside the repo, else unchanged."""
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


_DEFAULT_JSON_INDENT: Final = 2
"""Default spaces per indentation level in diffable JSON files."""


@dataclass(frozen=True, slots=True)
class _JsonStyle:
    """Indentation width and key-ordering policy for `format_json`."""

    indent: int
    sort_keys: bool


def _is_json_scalar(value: object) -> bool:
    """Returns whether `value` is a JSON scalar: null, boolean, number, or string."""
    return value is None or isinstance(value, bool | int | float | str)


def _is_flat_json_object(value: object) -> bool:
    """Returns whether `value` is a mapping of string keys to JSON scalars."""
    return isinstance(value, Mapping) and all(
        isinstance(key, str) and _is_json_scalar(item) for key, item in value.items()
    )


def _entry_key[V](entry: tuple[str, V]) -> str:
    """Returns the key of a `(key, value)` entry for sorting."""
    return entry[0]


def _mapping_entries[K, V](
    mapping: Mapping[K, V], *, sort_keys: bool
) -> list[tuple[str, V]]:
    """Returns `(key, value)` pairs from `mapping`, verifying that all keys are strings.

    Raises:
        TypeError: When any key in `mapping` is not a string.
    """
    entries: list[tuple[str, V]] = []
    for key, item in mapping.items():
        if not isinstance(key, str):
            message = f"JSON object keys must be strings, not {type(key).__name__}"
            raise TypeError(message)
        entries.append((key, item))
    if sort_keys:
        entries.sort(key=_entry_key)
    return entries


def _format_flat_object[K, V](mapping: Mapping[K, V], style: _JsonStyle) -> str:
    """Serializes a flat mapping of string keys to JSON scalars onto one line."""
    entries = _mapping_entries(mapping, sort_keys=style.sort_keys)
    return json.dumps(dict(entries), allow_nan=False)


def _format_sequence(items: Sequence[object], style: _JsonStyle, level: int) -> str:
    """Formats a JSON array, keeping scalar lists and flat object items on one line."""
    if not items:
        return "[]"
    if all(_is_json_scalar(item) for item in items):
        return json.dumps(items, allow_nan=False)
    child_indent = " " * (style.indent * (level + 1))
    closing_indent = " " * (style.indent * level)
    if all(_is_flat_json_object(item) for item in items):
        formatted = [
            f"{child_indent}{_format_flat_object(item, style)}"
            for item in items
            if isinstance(item, Mapping)
        ]
    else:
        formatted = [
            f"{child_indent}{_format_json_node(item, style, level + 1)}"
            for item in items
        ]
    joined = ",\n".join(formatted)
    return f"[\n{joined}\n{closing_indent}]"


def _format_mapping[K, V](mapping: Mapping[K, V], style: _JsonStyle, level: int) -> str:
    """Formats a JSON object with one key per line at indentation depth `level`."""
    entries = _mapping_entries(mapping, sort_keys=style.sort_keys)
    if not entries:
        return "{}"
    child_indent = " " * (style.indent * (level + 1))
    closing_indent = " " * (style.indent * level)
    formatted = [
        f"{child_indent}{json.dumps(key, allow_nan=False)}: "
        f"{_format_json_node(item, style, level + 1)}"
        for key, item in entries
    ]
    joined = ",\n".join(formatted)
    return f"{{\n{joined}\n{closing_indent}}}"


def _format_json_node(value: object, style: _JsonStyle, level: int) -> str:
    """Formats a JSON value recursively at indentation depth `level`.

    Raises:
        TypeError: When `value` or any nested element is not JSON-serializable.
    """
    if _is_json_scalar(value):
        return json.dumps(value, allow_nan=False)
    if isinstance(value, list | tuple):
        return _format_sequence(value, style, level)
    if isinstance(value, Mapping):
        return _format_mapping(value, style, level)
    message = f"unsupported JSON value of type {type(value).__name__}"
    raise TypeError(message)


def format_json(
    value: object,
    *,
    indent: int = _DEFAULT_JSON_INDENT,
    sort_keys: bool = False,
) -> str:
    """Formats `value` as diffable compact JSON without a trailing newline.

    Nested objects and mixed arrays are indented by `indent` spaces for readable
    diffs. Arrays of JSON scalars (such as numeric samples or `[real, imaginary]`
    pairs) are kept inline on a single line, and arrays whose elements are all flat
    objects (such as catalog records) place one object per line. Non-finite floats
    (`NaN`, `Infinity`) are rejected.

    Raises:
        TypeError: When `value` contains an unsupported type or a non-string key.
        ValueError: When `indent` is negative or `value` contains non-finite floats.
    """
    if indent < 0:
        message = f"indent must be non-negative, not {indent}"
        raise ValueError(message)
    style = _JsonStyle(indent=indent, sort_keys=sort_keys)
    return _format_json_node(value, style, 0)


def save(path: Path, results: BaseModel | Mapping[str, object]) -> None:
    """Writes `results` to `path` as compact JSON and prints the destination path."""
    document = (
        results.model_dump(mode="json") if isinstance(results, BaseModel) else results
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    text = format_json(document)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {shown(path)}")


def load(path: Path) -> dict[str, Any]:
    """Reads and parses a JSON results file written by `save`."""
    return json.loads(path.read_text(encoding="utf-8"))


CONFIDENCE: Final = 0.95
"""Confidence level used for all median confidence intervals across benchmark suites."""


def rank(count: int, confidence: float = CONFIDENCE) -> int | None:
    """Computes the 1-based order-statistic rank bounding the median of `count` runs.

    For `count` independent samples from any continuous distribution, each sample
    falls below the true median with probability 0.5. The count below the median
    follows a binomial distribution, so the `k`-th smallest and `k`-th largest
    samples bracket the median with probability `1 - 2 P(B < k)`. Selects the
    largest `k` whose exact binomial coverage reaches `confidence`, or returns None
    when even the full range falls short (as with fewer than 6 runs at 95%, where
    `1 - 2 / 2**5` is 93.75%).

    Examples:
        >>> rank(5), rank(6), rank(10), rank(100)
        (None, 1, 2, 40)
    """
    total = 2**count
    below = 0
    found: int | None = None
    for k in range(1, count // 2 + 1):
        below += math.comb(count, k - 1)
        if 2 * below > (1 - Fraction(confidence)) * total:
            break
        found = k
    return found


@dataclass(frozen=True, slots=True)
class Median:
    """Sample median of benchmark runs with a distribution-free confidence interval.

    Interval bounds cover the true distribution median with probability at least
    `CONFIDENCE`, using order-statistic endpoints from `rank` (None when too few
    runs exist).
    """

    value: float
    lower: float | None = None
    upper: float | None = None

    def separated(self, other: Median) -> bool:
        """Returns whether two confidence intervals are disjoint (distinct medians).

        Returns False when either median lacks confidence interval bounds.
        """
        if self.lower is None or self.upper is None:
            return False
        if other.lower is None or other.upper is None:
            return False
        return self.upper < other.lower or other.upper < self.lower

    def resolution(self, other: Median) -> float | None:
        """Computes the minimum relative shift separating `self` and `other` intervals.

        Expressed as a fraction of `self.value`, taking the wider of the upward and
        downward shifts required for `other`'s interval to clear `self`'s interval.
        Returns None when either median lacks interval bounds.
        """
        if self.lower is None or self.upper is None:
            return None
        if other.lower is None or other.upper is None:
            return None
        above = (self.upper - self.value) + (other.value - other.lower)
        below = (self.value - self.lower) + (other.upper - other.value)
        return max(above, below) / self.value


def median(runs: Sequence[float], confidence: float = CONFIDENCE) -> Median:
    """Computes the median of `runs` with a `rank`-based interval when available."""
    value = statistics.median(runs)
    k = rank(len(runs), confidence)
    if k is None:
        return Median(value)
    ordered = sorted(runs)
    return Median(value, ordered[k - 1], ordered[-k])


def duration(seconds: float) -> str:
    """Formats `seconds` with an SI prefix to four significant figures.

    Examples:
        >>> duration(0.0123456)
        '12.35 ms'
    """
    for factor, unit in ((1.0, "s"), (1e-3, "ms"), (1e-6, "µs")):
        if seconds >= factor:
            return f"{seconds / factor:.4g} {unit}"
    return f"{seconds / 1e-9:.4g} ns"


def ru_maxrss_to_bytes(ru_maxrss: int, platform_name: str | None = None) -> int:
    """Converts ``getrusage(2)``'s `ru_maxrss` reading into bytes.

    On macOS (`darwin`), ``getrusage(2)`` reports `ru_maxrss` directly in bytes.
    On Linux and BSD systems, ``getrusage(2)`` reports `ru_maxrss` in kibibytes
    (1,024 bytes).

    Raises:
        ValueError: When `ru_maxrss` is negative.

    Examples:
        >>> ru_maxrss_to_bytes(1024, "linux")
        1048576
        >>> ru_maxrss_to_bytes(1048576, "darwin")
        1048576
    """
    if ru_maxrss < 0:
        message = f"ru_maxrss must be non-negative, not {ru_maxrss}"
        raise ValueError(message)
    target_platform = sys.platform if platform_name is None else platform_name
    return ru_maxrss if target_platform == _DARWIN else ru_maxrss * KIBIBYTE


def _linux_vm_hwm_bytes(status_path: Path = _PROC_STATUS) -> int | None:
    """Reads `VmHWM` (peak resident set size) from Linux ``proc_pid_status(5)``.

    Linux records `VmHWM` on the active `mm_struct` created by ``execve(2)`` in
    kibibytes (`kB` = 1,024 bytes). Unlike ``getrusage(2)``'s `ru_maxrss`, which
    carries the parent process's `hiwater_rss` across ``fork(2)``/``vfork(2)`` and
    ``execve(2)`` via `signal->maxrss`, `VmHWM` resets cleanly for each exec'd child.
    """
    try:
        text = status_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("VmHWM:"):
            parts = line.split()
            if len(parts) >= _PROC_FIELDS and parts[1].isdigit():
                return int(parts[1]) * KIBIBYTE
    return None


def peak_memory_bytes() -> int:
    """Returns the current process's peak resident set size in bytes.

    On Linux, reads `VmHWM` from ``/proc/self/status`` (`proc_pid_status(5)`) so a
    child spawned via ``fork(2)`` or ``vfork(2)`` and ``execve(2)`` is not polluted
    by its parent's higher resident memory. Falls back to ``getrusage(2)``
    (`resource.RUSAGE_SELF`), converting `ru_maxrss` via `ru_maxrss_to_bytes`, on
    macOS and any system without ``/proc/self/status``.
    """
    if sys.platform.startswith("linux"):
        high_water_mark = _linux_vm_hwm_bytes()
        if high_water_mark is not None:
            return high_water_mark
    return ru_maxrss_to_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def available_memory_bytes(meminfo_path: Path = _PROC_MEMINFO) -> int | None:
    """Returns available machine memory in bytes for budgeting concurrent runs.

    On Linux, reads `MemAvailable` (falling back to `MemTotal`) from
    ``/proc/meminfo`` (`proc_meminfo(5)`). On macOS, reads `hw.memsize` via
    ``sysctl``. Returns None when neither source is available.
    """
    if meminfo_path.is_file():
        values: dict[str, int] = {}
        try:
            lines = meminfo_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in lines:
            if line.startswith(("MemAvailable:", "MemTotal:")):
                parts = line.split()
                if len(parts) >= _PROC_FIELDS and parts[1].isdigit():
                    values[parts[0].rstrip(":")] = int(parts[1]) * KIBIBYTE
        if "MemAvailable" in values:
            return values["MemAvailable"]
        if "MemTotal" in values:
            return values["MemTotal"]
    total = _sysctl("hw.memsize")
    return None if total is None or not total.isdigit() else int(total)


def memory(byte_count: int) -> str:
    """Formats `byte_count` with an IEC binary prefix to four significant figures.

    Uses binary prefixes (`TiB`, `GiB`, `MiB`, `KiB`, `B`) matching the convention
    in `benchmarks/speed/README.md` (1 KiB = 1,024 bytes).

    Raises:
        ValueError: When `byte_count` is negative.

    Examples:
        >>> memory(512)
        '512 B'
        >>> memory(1536 * 1024)
        '1.5 MiB'
    """
    if byte_count < 0:
        message = f"byte_count must be non-negative, not {byte_count}"
        raise ValueError(message)
    for factor, unit in _MEMORY_UNITS:
        if byte_count >= factor:
            return f"{byte_count / factor:.4g} {unit}"
    return f"{byte_count} B"


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    """Formats `headers` and `rows` as a GitHub-flavored Markdown table.

    Examples:
        >>> markdown_table(["A", "B"], [["1", "2"]]).splitlines()
        ['| A | B |', '| --- | --- |', '| 1 | 2 |']
    """
    all_rows: list[Sequence[str]] = [headers, ["---"] * len(headers), *rows]
    return "\n".join("| " + " | ".join(row) + " |" for row in all_rows)


def aligned_row(cells: Sequence[str], widths: Sequence[int]) -> str:
    """Formats a row of `cells`, each right-aligned to its width, two spaces apart.

    Examples:
        >>> aligned_row(["a", "1"], [3, 4])
        '  a     1'
    """
    return "  ".join(
        f"{cell:>{width}}" for cell, width in zip(cells, widths, strict=True)
    ).rstrip()


def aligned_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    widths: Sequence[int],
) -> str:
    """Formats `headers` and `rows` as right-aligned fixed-width columns."""
    return "\n".join(aligned_row(row, widths) for row in (headers, *rows))
