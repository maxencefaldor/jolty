# Copyright 2026 Maxence Faldor
"""What every suite records of where it measured, and its results files.

Each suite of ``benchmarks/`` measures on one machine, at one commit, under the load
other programs put on it, and writes what it found to a results file headed by its
provenance: the commit, marked ``-dirty`` when tracked files have changes, and the
machine. This module is the one definition of each, which every suite imports: the
suites themselves (`Suite`), `load_average`, `commit`, `machine`, `provenance`,
`results_path` in each suite's `results_directory`, and `save` and `load` for the
results files; and `duration`, how the suites print a time.
"""

import json
import os
import platform
import subprocess
from collections.abc import Mapping
from enum import StrEnum
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
