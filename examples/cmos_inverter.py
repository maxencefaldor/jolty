# Copyright 2026 Maxence Faldor
"""The CMOS inverter: its transfer curve, its switching threshold, its edges.

A p-channel transistor twice as wide as its n-channel partner, which offsets the holes'
lower mobility only partly here, as both share one transconductance parameter: the
p-channel is the stronger, so the switching threshold, where both saturate,
Vm = (VDD - |Vtp| + Vtn·√(βn/βp))/(1 + √(βn/βp)), sits at 1.81 V of 3.3 V. Simulate it
with ``jolty simulate examples/cmos_inverter.py``.
"""

import math

from common import SUPPLY, inverter

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.plan import DcSweep, Transient, linear
from jolty.specs import value
from jolty.units import V, fF, mV, ns, ps
from jolty.waveforms import Pulse

RATIO = math.sqrt(1 / 2)
THRESHOLD = (SUPPLY - 0.7 + 0.7 * RATIO) / (1 + RATIO)


@component
def bench(c: Builder) -> None:
    """The inverter on 3.3 V, driven by a pulse, loading 20 fF."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(SUPPLY * V), id="VDD")
    edges = Pulse(
        initial=0 * V,
        pulsed=SUPPLY * V,
        delay=1 * ns,
        rise=100 * ps,
        fall=100 * ps,
        width=4 * ns,
        period=10 * ns,
    )
    source = c.place(devices.voltage_source(edges), id="VIN")
    gate = c.place(inverter(), id="X1")
    load = c.place(devices.capacitor(20 * fF), id="CL")
    c.connect(supply.positive, gate.vdd, name="vdd")
    c.connect(source.positive, gate.a, name="in")
    c.connect(gate.y, load.a, name="out")
    c.connect(supply.negative, source.negative, gate.gnd, load.b, ground)


def build() -> Component:
    """The inverter on its bench."""
    return bench()


def plan() -> Plan:
    """The transfer curve, and two periods of edges."""
    return Plan(
        {
            "transfer": DcSweep("VIN", linear(0 * V, SUPPLY * V, 331)),
            "edges": Transient(stop=20 * ns, max_step=10 * ps),
        }
    )


def specs() -> Specs:
    """Rail to rail, and the output at half the supply near the switching threshold."""
    return Specs(
        [
            value("transfer", "out", at=0 * V).within(SUPPLY * V, 1 * mV),
            value("transfer", "out", at=SUPPLY * V).within(0 * V, 1 * mV),
            value("transfer", "out", at=THRESHOLD * V).within(SUPPLY / 2 * V, 50 * mV),
        ]
    )
