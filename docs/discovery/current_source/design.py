# Copyright 2026 Maxence Faldor
"""Mirror-fed two-transistor 1 mA current source.

Transistor Q1 passes the load current while Q2 shunts Q1 base drive once sense
resistor Rs drops one Vbe.
Feeding Q2 through a supply resistor causes Q2 current and Vbe to track the line,
yielding 17% regulation error from 3 V to 24 V.
Replacing the resistor with a 1:10 PNP mirror lowers line regulation error to 2.6%,
limited by the 2N3906 18.7 V Early voltage.
Output current equals Vbe/Rs and still falls at 0.3%/K with temperature.
Simulate via ``jolty simulate docs/discovery/current_source/design.py``.
"""

import candidates

from jolty import Component, Corner, Plan, Specs
from jolty.plan import DcSweep, OperatingPoint, linear
from jolty.specs import value
from jolty.units import V, celsius, mA

SENSE = 722.0
"""Sense resistance Rs in ohms, tuned for 1 mA at 12 V and 27 °C."""


def build() -> Component:
    """Builds the mirror-fed current source across a 12 V supply."""
    return candidates.bench(candidates.mirrored_limiter(SENSE, 10.0), 12.0)


def plan() -> Plan:
    """Returns the plan for 12 V DC and 3 V to 24 V sweeps across temperature."""
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
    """Returns specs checking 1 mA within 2% across 3 V to 24 V and temperature.

    The design meets the bound at 27 °C within 2.6% line variation and fails at extreme
    temperature corners due to Vbe thermal drift.
    """
    return Specs(
        [
            value("line", "V1:current", at=level * V).within(-1 * mA, 0.02 * mA)
            for level in (3, 12, 24)
        ]
    )
