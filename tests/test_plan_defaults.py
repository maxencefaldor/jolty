# Copyright 2026 Maxence Faldor
"""The guide's table of options states the engine's defaults, which it restates."""

import re
from pathlib import Path

from jolty import plan

GUIDE = Path(__file__).parents[1] / "docs" / "guide" / "simulation.md"
"""The guide's page on simulation, which tables the options."""

ROW = re.compile(r"^\| `(\w+)` \| ([0-9.e+-]+)(?: [A-Za-z]+)? \|", re.MULTILINE)
"""A row of the table: an option's name and its default, with its unit."""


def test_the_guide_states_each_option_s_default_as_the_engine_has_it() -> None:
    stated = {
        name: float(value)
        for name, value in ROW.findall(GUIDE.read_text(encoding="utf-8"))
        if name in plan.DEFAULTS
    }
    assert stated == dict(plan.DEFAULTS)
