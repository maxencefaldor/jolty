# Copyright 2026 Maxence Faldor
"""What every suite records of where it measured, and its results files.

Each suite of ``benchmarks/`` measures on one machine, at one commit, under the load
other programs put on it, and writes what it found to a results file headed by its
provenance: the commit, marked ``-dirty`` when tracked files have changes, and the
machine. This module is the one definition of each, which every suite imports: the
suites themselves (`Suite`), `load_average`, `commit`, `machine`, `provenance`,
`results_path` in each suite's `results_directory`, and `format_json`, `save` and
`load` for the results files; `duration`, how the suites print a time; `memory`, how
they print a byte count; `ru_maxrss_to_bytes`, `peak_memory_bytes` and
`available_memory_bytes` for resident memory; and `median`, the median of a time's runs
with the interval that holds the true median at `CONFIDENCE`.
"""

import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, JsonValue

from jolty.simulate import pool

BENCHMARKS: Final = Path(__file__).resolve().parent
"""The ``benchmarks`` directory, which holds a directory for each suite."""

ROOT: Final = BENCHMARKS.parent
"""The repository's root."""


class Suite(StrEnum):
    """A suite of fixtures run and scored one way, named as its directory is."""

    SPEED = "speed"
    PRECISION = "precision"
    ROBUSTNESS = "robustness"
    COVERAGE = "coverage"
    AGENTS = "agents"


DIRTY: Final = "-dirty"
"""What marks a commit whose working tree has changes to tracked files."""

KIBIBYTE: Final = 1024
"""Bytes per kibibyte, the unit Linux ``getrusage(2)`` and ``/proc`` report."""

_MEMORY_UNITS: Final = (
    (1 << 40, "TiB"),
    (1 << 30, "GiB"),
    (1 << 20, "MiB"),
    (1 << 10, "KiB"),
)
"""IEC binary prefixes for `memory`, largest first."""

_DARWIN: Final = "darwin"
"""`sys.platform` value on macOS, where ``getrusage(2)`` reports bytes."""

_PROC_STATUS: Final = Path("/proc/self/status")
"""Linux process status file holding `VmHWM` (`proc_pid_status(5)`)."""

_PROC_MEMINFO: Final = Path("/proc/meminfo")
"""Linux memory summary file (`proc_meminfo(5)`)."""

_PROC_FIELDS: Final = 2
"""Minimum whitespace-separated tokens on a Linux ``/proc`` key-value line."""


def load_average() -> float:
    """The machine's load average over the last minute."""
    return os.getloadavg()[0]


def usable_cores() -> int:
    """The threads the engine runs on by default, a thread per processor it may use.

    The engine's own count, as Rust's `available_parallelism` reads it: the processors
    the process's affinity allows, within its container's CPU quota, so that "every
    core" is the same number in the Python timings as in the engine's batches.
    """
    return pool().threads


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


def commit() -> str:
    """The checked-out commit, marked `DIRTY` when tracked files have changes."""
    dirty = git("status", "--porcelain", "--untracked-files=no") != ""
    return git("rev-parse", "HEAD") + (DIRTY if dirty else "")


def _sysctl(name: str) -> str | None:
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


def machine() -> dict[str, JsonValue]:
    """The machine the suites measure on: its processor, memory and system."""
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
    """Where a suite measured: the commit, with its `DIRTY` mark, and the machine.

    It heads every suite's results file.
    """

    commit: str
    machine: dict[str, JsonValue]


def provenance() -> Provenance:
    """The provenance of a measurement made now, here."""
    return Provenance(commit=commit(), machine=machine())


def results_directory(suite: Suite) -> Path:
    """The directory of the results files of `suite`."""
    return BENCHMARKS / suite / "results"


def results_path(suite: Suite) -> Path:
    """Where the checked-out commit's results of `suite` go: ``<commit>.json``.

    The commit carries its `DIRTY` mark, and the file lies in the suite's
    `results_directory`.
    """
    return results_directory(suite) / f"{commit()}.json"


def shown(path: Path) -> Path:
    """`path` from the repository's root, if it lies within it, as messages give it."""
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


_DEFAULT_JSON_INDENT: Final = 2
"""Default spaces per indentation level in diffable JSON files."""


@dataclass(frozen=True, slots=True)
class _JsonStyle:
    """Indentation width and key-ordering policy for `format_json`."""

    indent: int
    sort_keys: bool


def _is_json_scalar(value: object) -> bool:
    """Return whether `value` is a JSON scalar: null, boolean, number, or string."""
    return value is None or isinstance(value, bool | int | float | str)


def _is_flat_json_object(value: object) -> bool:
    """Return whether `value` is a mapping of string keys to JSON scalars."""
    return isinstance(value, Mapping) and all(
        isinstance(key, str) and _is_json_scalar(item) for key, item in value.items()
    )


def _entry_key[V](entry: tuple[str, V]) -> str:
    """Return the key of a `(key, value)` entry for sorting."""
    return entry[0]


def _mapping_entries[K, V](
    mapping: Mapping[K, V], *, sort_keys: bool
) -> list[tuple[str, V]]:
    """Return `(key, value)` pairs from `mapping`, verifying string keys.

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
    """Serialize a flat mapping of string keys to JSON scalars onto one line."""
    entries = _mapping_entries(mapping, sort_keys=style.sort_keys)
    return json.dumps(dict(entries), allow_nan=False)


def _format_sequence(items: Sequence[object], style: _JsonStyle, level: int) -> str:
    """Format a JSON array, keeping scalar lists and flat object items on one line."""
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
    """Format a JSON object with one key per line at indentation depth `level`."""
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
    """Recursively format a JSON value at indentation depth `level`.

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
    """Format `value` as diffable compact JSON without a trailing newline.

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
    """Writes the results file `path`, as diffable compact JSON, and says so."""
    document = (
        results.model_dump(mode="json") if isinstance(results, BaseModel) else results
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    text = format_json(document)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {shown(path)}")


def load(path: Path) -> dict[str, Any]:
    """What the results file `path` holds, as `save` wrote it."""
    return json.loads(path.read_text(encoding="utf-8"))


CONFIDENCE: Final = 0.95
"""The confidence of every interval the suites state."""


def rank(count: int, confidence: float = CONFIDENCE) -> int | None:
    """The rank, from 1, of the order statistics that bound the median of `count` runs.

    Of `count` runs of any continuous distribution, each falls below its median with
    probability one half, independently, so the number below is binomial, of `count`
    trials and one half; the `k`-th smallest and the `k`-th largest run then hold the
    median between them unless `k` or more fall on one side of it, with probability
    `1 - 2 P(B < k)`. The rank is the largest `k` whose interval holds the median with
    at least `confidence`, from the binomial's exact sums; None when even the smallest
    and the largest run do not, as with fewer than 6 runs at 95%: `1 - 2 / 2**5` is
    93.75%.

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
    """The median of a time's runs, and the interval that holds its true median.

    The interval holds the median of the distribution the runs are drawn from with
    `CONFIDENCE` at least, whatever the distribution, as `rank` sets its ends: None when
    there are too few runs for one.
    """

    value: float
    lower: float | None = None
    upper: float | None = None

    def separated(self, other: Median) -> bool:
        """Whether the two intervals do not meet, the medians then different.

        Never when either has no interval.
        """
        if self.lower is None or self.upper is None:
            return False
        if other.lower is None or other.upper is None:
            return False
        return self.upper < other.lower or other.upper < self.lower

    def resolution(self, other: Median) -> float | None:
        """The least change from this median to `other` their intervals would separate.

        As a fraction of this median: the change at which `other`'s interval, moved as
        its median moves, would clear this one's, on whichever side is wider; None when
        either has no interval.
        """
        if self.lower is None or self.upper is None:
            return None
        if other.lower is None or other.upper is None:
            return None
        above = (self.upper - self.value) + (other.value - other.lower)
        below = (self.value - self.lower) + (other.upper - other.value)
        return max(above, below) / self.value


def median(runs: Sequence[float], confidence: float = CONFIDENCE) -> Median:
    """The median of `runs`, with the interval of `rank`, if they are enough for one."""
    value = statistics.median(runs)
    k = rank(len(runs), confidence)
    if k is None:
        return Median(value)
    ordered = sorted(runs)
    return Median(value, ordered[k - 1], ordered[-k])


def duration(seconds: float) -> str:
    """`seconds` with an SI prefix, to four figures.

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
    """The current process's peak resident set size, in bytes.

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
    """Available machine memory in bytes, for budgeting concurrent runs.

    On Linux, reads `MemAvailable` (falling back to `MemTotal`) from
    ``/proc/meminfo`` (`proc_meminfo(5)`). On macOS, reads `hw.memsize` via
    ``sysctl``. Returns None if neither is available.
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
    """`byte_count` with an IEC binary prefix, to four figures.

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
