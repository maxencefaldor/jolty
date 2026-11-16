# Copyright 2026 Maxence Faldor
"""What every suite records of where it measured, and its results files.

Each suite of ``benchmarks/`` measures on one machine, at one commit, under the load
other programs put on it, and writes what it found to a results file headed by its
provenance: the commit, marked ``-dirty`` when tracked files have changes, and the
machine. This module is the one definition of each, which every suite imports: the
suites themselves (`Suite`), `load_average`, `commit`, `machine`, `provenance`,
`results_path` in each suite's `results_directory`, and `save` and `load` for the
results files; `duration`, how the suites print a time; and `median`, the median of a
time's runs with the interval that holds the true median at `CONFIDENCE`.
"""

import json
import math
import os
import platform
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, JsonValue

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


def load_average() -> float:
    """The machine's load average over the last minute."""
    return os.getloadavg()[0]


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
    memory = _sysctl("hw.memsize")
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


def save(path: Path, results: BaseModel | Mapping[str, object]) -> None:
    """Writes the results file `path`, as JSON, and says so."""
    document = (
        results.model_dump(mode="json") if isinstance(results, BaseModel) else results
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=2, allow_nan=False)
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
