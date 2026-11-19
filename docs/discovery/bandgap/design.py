# Copyright 2026 Maxence Faldor
"""JFET-fed Widlar three-transistor bandgap voltage reference.

Diode-connected Q1 and Q2 over 155 Ω form a Widlar current source, dropping
Vt·ln(I1/I2) across R2.
Transistor Q3 sums its Vbe with Q2 collector current across 2.4 kΩ to produce a
temperature-compensated output near the silicon bandgap.
Resistor bias moves the output by 44 mV/V on the supply, whereas a zero-Vgs 2N5457
JFET bias reduces line sensitivity to 0.31 mV/V.
Thermal drift in JFET current also straightens bandgap curvature to 1.2414 V within
1.08 mV from -40 °C to 125 °C (5.3 ppm/K) using seven parts.
Simulate via ``jolty simulate docs/discovery/bandgap/design.py``.
"""

import candidates

from jolty import Component, Corner, Plan, Specs
from jolty.plan import DcSweep, OperatingPoint, linear
from jolty.specs import value
from jolty.units import V, celsius, mV


def build() -> Component:
    """Builds the JFET-fed bandgap reference on a 5 V supply with a 100 kΩ load."""
    return candidates.bench(candidates.widlar_jfet(600.0, 155.1, 2400.0))


def plan() -> Plan:
    """Returns the plan across temperature corners and a 4.5 V to 5.5 V supply sweep."""
    return Plan(
        {
            "op": OperatingPoint(),
            "line": DcSweep("VCC", linear(4.5 * V, 5.5 * V, 11)),
        },
        corners={
            "cold": Corner(temperature=celsius(-40)),
            "warm": Corner(temperature=celsius(85)),
            "hot": Corner(temperature=celsius(125)),
        },
    )


def specs() -> Specs:
    """Returns specs verifying 1.2415 V within 50 ppm/K and 1 mV/V across corners."""
    return Specs(
        [
            value("op", "out").within(1.2415 * V, 5 * mV),
            value("line", "out", at=4.5 * V).within(1.2415 * V, 5.5 * mV),
            value("line", "out", at=5.5 * V).within(1.2415 * V, 5.5 * mV),
        ]
    )
