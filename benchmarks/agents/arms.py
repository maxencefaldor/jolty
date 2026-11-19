# Copyright 2026 Maxence Faldor
"""Benchmark arm definitions controlling agent tools, prompts, and staged files.

The jolty arm provides the jolty CLI, MCP server, and Python package, along with the
unmodified task prompt (``task.md``) and given files. The ngspice arm provides ngspice
and Python with numpy, scipy, and matplotlib, without jolty. For ngspice, the prompt
substitutes task-specific sentences from ``ngspice/task.toml`` and tool instructions
from ``ngspice.toml``. Staged files for ngspice exclude jolty-dependent Python scripts
and include the replacement SPICE files from ``ngspice/given``.

Each arm writes designs in a dedicated format: Python scripts (such as ``design.py``)
for jolty, and SPICE netlists (such as ``design.cir``) for ngspice. The scorer reads
ngspice netlists through jolty's SPICE reader after loading the provided ``.lib``
files.
"""

import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Final


class Arm(StrEnum):
    """Tooling configuration assigned to an agent: jolty or ngspice with Python."""

    JOLTY = "jolty"
    NGSPICE = "ngspice"


TOOLS: Final[Path] = Path(__file__).with_name("ngspice.toml")
"""Tool-description replacement table applied to every ngspice arm prompt."""
DISCRETE_PARTS: Final[Path] = Path(__file__).with_name("discrete_parts.py")
"""Standalone 2N3904 and 2N3906 definitions shared by discrete transistor tasks."""
SUFFIXES: Final = {Arm.JOLTY: ".py", Arm.NGSPICE: ".cir"}
"""Design file suffix for each arm: Python scripts for jolty, netlists for ngspice."""


def _replacement(table: dict[str, object], source: Path) -> tuple[str, str]:
    """Extracts a ``(jolty, ngspice)`` prompt replacement pair from `table`.

    Raises:
        ValueError: When `table`, read from `source`, does not pair two strings.
    """
    match table:
        case {"jolty": str(jolty), "ngspice": str(ngspice), **rest} if not rest:
            return jolty, ngspice
    message = f"{source} pairs no jolty text with an ngspice one: {table}"
    raise ValueError(message)


def replacements(task: Path) -> list[tuple[str, str]]:
    """Returns ordered prompt replacement pairs for `task` on the ngspice arm.

    Task-specific sentence replacements from ``ngspice/task.toml`` appear first,
    followed by the shared tool description replacement from `TOOLS`.

    Raises:
        ValueError: When a replacement table does not pair two strings.
    """
    source = task / "ngspice" / "task.toml"
    with source.open("rb") as file:
        tables = tomllib.load(file).get("replace", [])
    with TOOLS.open("rb") as file:
        tools = tomllib.load(file)
    return [
        *(_replacement(table, source) for table in tables),
        _replacement(tools, TOOLS),
    ]


def prompt(task: Path, arm: Arm) -> str:
    """Returns the rendered task prompt for `arm` on `task`.

    Raises:
        ValueError: When a jolty prompt substring targeted for ngspice replacement does
            not occur in ``task.md`` exactly once.
    """
    text = (task / "task.md").read_text(encoding="utf-8")
    if arm is Arm.JOLTY:
        return text
    for jolty, ngspice in replacements(task):
        count = text.count(jolty)
        if count != 1:
            message = f"{task.name}'s task.md holds {count} times, not once: {jolty!r}"
            raise ValueError(message)
        text = text.replace(jolty, ngspice)
    return text


def given(task: Path, arm: Arm) -> list[Path]:
    """Returns the sorted source files provided to an agent of `arm` on `task`.

    Raises:
        ValueError: When the ngspice arm resolves two files with the same filename.
    """
    files = sorted(path for path in (task / "given").glob("*") if path.is_file())
    if arm is Arm.JOLTY:
        return files
    kept = [path for path in files if path.suffix != SUFFIXES[Arm.JOLTY]]
    found = [*kept, *_spice(task)]
    names = [path.name for path in found]
    twice = sorted({name for name in names if names.count(name) > 1})
    if twice:
        message = f"the ngspice arm of {task.name} gets two files named {twice[0]}"
        raise ValueError(message)
    return sorted(found, key=lambda path: path.name)


def _standalone_source(path: Path) -> Path:
    """Returns `DISCRETE_PARTS` when `path` is a stub delegating to it."""
    if path.name == "parts.py" and path.suffix == SUFFIXES[Arm.JOLTY]:
        text = path.read_text(encoding="utf-8")
        if "from discrete_parts import" in text:
            return DISCRETE_PARTS
    return path


def staged(task: Path, arm: Arm) -> list[tuple[str, Path]]:
    """Returns ``(name, path)`` pairs staged into the working directory for `arm`.

    When ``given/parts.py`` delegates to `DISCRETE_PARTS`, staging copies the
    standalone `DISCRETE_PARTS` module as ``parts.py`` so sandboxed agents can read
    full transistor definitions inside the working directory.
    """
    return [
        (path.name, _standalone_source(path) if arm is Arm.JOLTY else path)
        for path in given(task, arm)
    ]


def _spice(task: Path) -> list[Path]:
    """Returns sorted SPICE files from ``ngspice/given`` for `task`."""
    directory = task / "ngspice" / "given"
    return sorted(path for path in directory.glob("*") if path.is_file())


def libraries(task: Path) -> list[Path]:
    """Returns ngspice ``.lib`` libraries for `task`, loaded first by the scorer."""
    return [path for path in _spice(task) if path.suffix == ".lib"]
