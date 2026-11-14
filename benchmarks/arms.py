# Copyright 2026 Maxence Faldor
"""The comparison's two arms: what an agent has, gets and writes on a task.

An agent of the jolty arm has jolty, its command line, MCP server and Python package,
and gets the task's prompt, ``task.md``, and its given files as they are. An agent of
the ngspice arm has ngspice and a Python with numpy, scipy and matplotlib, and nothing
of jolty. Its prompt is ``task.md`` with each sentence the task's ``ngspice/task.toml``
replaces in place, and the tools of ``ngspice.toml`` in place of jolty's; it gets the
task's given files but its Python scripts, which need jolty, and the SPICE files of the
task's ``ngspice/given``, which stand for them.

Each arm writes its design in its own form: the jolty arm a script, such as
``design.py``, and the ngspice arm a netlist, such as ``design.cir``, which the scorer
reads with jolty's SPICE reader after the ``.lib`` files it was given.
"""

import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Final


class Arm(StrEnum):
    """The tools an agent has: jolty's, or ngspice and a plain Python."""

    JOLTY = "jolty"
    NGSPICE = "ngspice"


TOOLS: Final = Path(__file__).with_name("ngspice.toml")
"""The ngspice arm's tools, in place of the jolty arm's in every prompt."""
SUFFIXES: Final = {Arm.JOLTY: ".py", Arm.NGSPICE: ".cir"}
"""The suffix of each arm's designs: a script's, or a netlist's."""


def _replacement(table: dict[str, object], source: Path) -> tuple[str, str]:
    """The jolty arm's text and the ngspice arm's in its place, as `table` holds them.

    Raises:
        ValueError: When `table`, read from `source`, does not pair two texts.
    """
    match table:
        case {"jolty": str(jolty), "ngspice": str(ngspice), **rest} if not rest:
            return jolty, ngspice
    message = f"{source} pairs no jolty text with an ngspice one: {table}"
    raise ValueError(message)


def replacements(task: Path) -> list[tuple[str, str]]:
    """The texts of `task`'s prompt the ngspice arm's replace, each with its own.

    They are the task's own sentences, then the tools.

    Raises:
        ValueError: When a table pairs no two texts.
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
    """The prompt an agent of `arm` gets on `task`.

    Raises:
        ValueError: When a text the ngspice arm's prompt replaces is not in the task's
            prompt exactly once.
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
    """The files an agent of `arm` gets on `task`, by name.

    Raises:
        ValueError: When the ngspice arm gets two files of one name.
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


def _spice(task: Path) -> list[Path]:
    """The SPICE files of `task`'s ngspice arm, in place of its Python scripts."""
    directory = task / "ngspice" / "given"
    return sorted(path for path in directory.glob("*") if path.is_file())


def libraries(task: Path) -> list[Path]:
    """The SPICE libraries of `task`'s ngspice arm, which the scorer reads first."""
    return [path for path in _spice(task) if path.suffix == ".lib"]
