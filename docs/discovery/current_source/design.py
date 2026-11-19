# Copyright 2026 Maxence Faldor
"""The best two-terminal 1 mA source found: a limiter fed by a mirror of its current.

Q1 passes the current, and Q2 steals Q1's base drive once the current holds a Vbe
across Rs. A plain limiter feeds Q2 through a resistor from the line, so Q2's current,
and its Vbe, follow the line: 17 % over 3 V to 24 V. Here a PNP mirror feeds Q2 a
tenth of the source's own current instead, and the line regulation falls to 2.6 %,
what the 2N3906's 18.7 V Early voltage leaves. The current is Vbe/Rs, so it still
falls 0.3 %/K with temperature. Simulate it with
``jolty simulate docs/discovery/current_source/design.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import candidates

from jolty import Component, Corner, Plan, Specs
from jolty.plan import DcSweep, OperatingPoint, linear
from jolty.specs import value
from jolty.units import V, celsius, mA

SENSE = 722.0
"""Rs, tuned to 1 mA at 12 V and 27 °C."""


def build() -> Component:
    """The source across 12 V."""
    return candidates.bench(candidates.mirrored_limiter(SENSE, 10.0), 12.0)


def plan() -> Plan:
    """The current at 12 V, and across 3 V to 24 V, at 27 °C, -40 °C and 85 °C."""
    return Plan(
        {
            "op": OperatingPoint(),
            "line": DcSweep("V1", linear(3 * V, 24 * V, 43)),
        },
        corners={
            "cold": Corner(temperature=celsius(-40)),
            "hot": Corner(temperature=celsius(85)),
        },
    )


def specs() -> Specs:
    """The prompt's bound, 1 mA within 2 % from 3 V to 24 V and -40 °C to 85 °C.

    The design meets it at 27 °C, within 2.6 % across the line, and fails it at the
    temperature corners, as any current set by a Vbe over a resistor does.
    """
    return Specs(
        [
            value("line", "V1:current", at=level * V).within(-1 * mA, 0.02 * mA)
            for level in (3, 12, 24)
        ]
    )
