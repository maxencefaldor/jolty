# Copyright 2026 Maxence Faldor
"""The CMOS inverter: transfer curve, switching threshold, and transient edges.

The p-channel transistor is twice as wide as the n-channel transistor while sharing the
same transconductance parameter, making the pull-up stronger. Both transistors saturate
at the switching threshold Vm = (VDD - |Vtp| + Vtn·√(βn/βp))/(1 + √(βn/βp)), which sits
at 1.81 V on a 3.3 V supply. Simulate with ``jolty simulate examples/cmos_inverter.py``.
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
    """Builds the inverter on 3.3 V driven by a pulse into a 20 fF load."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(SUPPLY * V), name="VDD")
    edges = Pulse(
        initial=0 * V,
        pulsed=SUPPLY * V,
        delay=1 * ns,
        rise=100 * ps,
        fall=100 * ps,
        width=4 * ns,
        period=10 * ns,
    )
    source = c.place(devices.voltage_source(edges), name="VIN")
    gate = c.place(inverter(), name="X1")
    load = c.place(devices.capacitor(20 * fF), name="CL")
    c.connect(supply.positive, gate.vdd, name="vdd")
    c.connect(source.positive, gate.a, name="in")
    c.connect(gate.y, load.a, name="out")
    c.connect(supply.negative, source.negative, gate.gnd, load.b, ground)


def build() -> Component:
    """Returns the CMOS inverter test bench component."""
    return bench()


def plan() -> Plan:
    """Returns DC transfer-sweep and two-period transient edge analyses."""
    return Plan(
        {
            "transfer": DcSweep("VIN", linear(0 * V, SUPPLY * V, 331)),
            "edges": Transient(stop=20 * ns, max_step=10 * ps),
        }
    )


def specs() -> Specs:
    """Returns specs for rail-to-rail outputs and the mid-supply switching threshold."""
    return Specs(
        [
            value("transfer", "out", at=0 * V).within(SUPPLY * V, 1 * mV),
            value("transfer", "out", at=SUPPLY * V).within(0 * V, 1 * mV),
            value("transfer", "out", at=THRESHOLD * V).within(SUPPLY / 2 * V, 50 * mV),
        ]
    )
