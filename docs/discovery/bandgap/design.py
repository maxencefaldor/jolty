# Copyright 2026 Maxence Faldor
"""The best reference found: Widlar's three-transistor bandgap, fed by a JFET.

Q1, diode-connected, and Q2 over 155 Ω form a Widlar current source, so R2 holds
Vt·ln(I1/I2), and Q3 holds the output at its own Vbe plus Q2's current across 2.4 kΩ:
a falling voltage and a rising one, summed near the band gap. Fed by a resistor, the
cell's output moves 44 mV for each volt of supply; fed by a 2N5457 whose gate is on its
source, a current source of its Idss, it moves 0.31 mV, and the JFET's own drift turns
out to straighten the bandgap's curvature: 1.2414 V within 1.08 mV from -40 °C to
125 °C, 5.3 ppm/K, with seven parts. Simulate it with
``jolty simulate docs/discovery/bandgap/design.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import candidates

from jolty import Component, Corner, Plan, Specs
from jolty.plan import DcSweep, OperatingPoint, linear
from jolty.specs import value
from jolty.units import V, celsius, mV


def build() -> Component:
    """The reference on 5 V, loaded by 100 kΩ."""
    return candidates.bench(candidates.widlar_jfet(600.0, 155.1, 2400.0))


def plan() -> Plan:
    """The output at 27 °C, -40 °C, 85 °C and 125 °C, and over a supply of 4.5-5.5 V."""
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
    """1.2415 V within the prompt's 50 ppm/K and 1 mV/V, everywhere."""
    return Specs(
        [
            value("op", "out").within(1.2415 * V, 5 * mV),
            value("line", "out", at=4.5 * V).within(1.2415 * V, 5.5 * mV),
            value("line", "out", at=5.5 * V).within(1.2415 * V, 5.5 * mV),
        ]
    )
