# Copyright 2026 Maxence Faldor
"""The guide's tables of options and of parallelism state the engine's defaults."""

import re
from pathlib import Path

from jolty import plan
from jolty.units import Quantity

GUIDE = Path(__file__).parents[1] / "docs" / "guide" / "simulation.md"
"""The guide's page on simulation, which tables the options."""

ROW = re.compile(r"^\| `(\w+)` \| ([0-9.e+-]+)(?: ([A-Za-z]+))? \|", re.MULTILINE)
"""A row of the table: an option's name and its default, with its optional unit."""

DISPLAY_UNITS: dict[str, str] = {"max_memory": "GiB"}
"""Non-SI display units shown in the guide for non-`Quantity` option defaults."""


def test_the_guide_states_each_option_s_default_as_the_engine_has_it() -> None:
    stated = {
        name: (float(value), unit or None)
        for name, value, unit in ROW.findall(GUIDE.read_text(encoding="utf-8"))
        if name in plan.DEFAULTS
    }
    expected = {
        name: (
            (default.value, str(default.dimension))
            if isinstance(default, Quantity)
            else (default, DISPLAY_UNITS.get(name))
        )
        for name, default in plan.DEFAULTS.items()
    }
    assert stated == expected


BREAK_EVEN = re.compile(r"^\| `(\w+)` \|((?: (?:[0-9,]+|none) \|){4})$", re.MULTILINE)
"""A row of the table of break-evens: a threshold's name and its break-even on each
number of threads, its thousands apart, or none."""


def test_the_guide_states_each_break_even_as_the_engine_has_it() -> None:
    stated = {
        name: tuple(
            None if cell.strip() == "none" else int(cell.replace(",", ""))
            for cell in cells.split("|")[:-1]
        )
        for name, cells in BREAK_EVEN.findall(GUIDE.read_text(encoding="utf-8"))
    }
    assert stated == {
        name: values for name, values in plan.BREAK_EVENS.items() if name != "threads"
    }
    assert plan.BREAK_EVENS["threads"] == (2, 4, 8, 64)
