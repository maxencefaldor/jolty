# Copyright 2026 Maxence Faldor
"""The astable multivibrator: two cross-coupled 2N3904s flipping each other in turn.

Each transistor's collector couples to the other's base through a capacitor. When one
turns on, its collector falls by nearly the supply and drives the other's base that far
below zero; that base recovers through its 47 kΩ toward the supply, turning its
transistor on at half way. Each half period is RC·ln 2 = 0.326 ms with 10 nF: a period
of 0.652 ms, 1.53 kHz. Slightly unequal capacitors break the symmetry that would hold
the circuit balanced. Simulate it with
``jolty simulate examples/astable_multivibrator.py``.
"""

import math

from common import npn

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.plan import Transient
from jolty.specs import frequency, maximum, minimum
from jolty.units import Hz, V, kohm, ms, nF, us
from jolty.waveforms import Pulse

IDEAL = 1 / (2 * 47e3 * 10e-9 * math.log(2))


@component
def multivibrator(c: Builder) -> None:
    """Two switches, each one's collector holding the other's base off for a while."""
    left, right = c.port("left", Role.OUTPUT), c.port("right", Role.OUTPUT)
    supply, ground = c.port("vcc", Role.SUPPLY), c.port("ground", Role.GROUND)
    first, second = c.place(npn(), id="Q1"), c.place(npn(), id="Q2")
    load_1 = c.place(devices.resistor(1 * kohm), id="RC1")
    load_2 = c.place(devices.resistor(1 * kohm), id="RC2")
    bias_1 = c.place(devices.resistor(47 * kohm), id="RB1")
    bias_2 = c.place(devices.resistor(47 * kohm), id="RB2")
    cross_1 = c.place(devices.capacitor(10 * nF), id="C1")
    cross_2 = c.place(devices.capacitor(10.1 * nF), id="C2")
    c.connect(supply, load_1.a, load_2.a, bias_1.a, bias_2.a)
    c.connect(load_1.b, first.collector, cross_1.a, left)
    c.connect(load_2.b, second.collector, cross_2.a, right)
    c.connect(cross_1.b, bias_2.b, second.base, name="base2")
    c.connect(cross_2.b, bias_1.b, first.base, name="base1")
    c.connect(first.emitter, second.emitter, ground)


@component
def bench(c: Builder) -> None:
    """The multivibrator on a 9 V supply that rises in 10 µs."""
    ground = c.port("ground", Role.GROUND)
    ramp = Pulse(
        initial=0 * V,
        pulsed=9 * V,
        delay=0 * us,
        rise=10 * us,
        fall=10 * us,
        width=1e3 * ms,
    )
    supply = c.place(devices.voltage_source(ramp), id="VCC")
    stage = c.place(multivibrator(), id="X1")
    c.connect(supply.positive, stage.vcc, name="vcc")
    c.connect(stage.left, name="left")
    c.connect(stage.right, name="right")
    c.connect(supply.negative, stage.ground, ground)


def build() -> Component:
    """The multivibrator on its bench."""
    return bench()


def plan() -> Plan:
    """Eight milliseconds, a dozen periods."""
    return Plan({"run": Transient(stop=8 * ms, max_step=1 * us)})


def specs() -> Specs:
    """A period within 10 % of 2RC·ln 2, each collector swinging rail to rail."""
    return Specs(
        [
            frequency("run", "left", start=2 * ms).within(IDEAL * Hz, 0.1 * IDEAL * Hz),
            maximum("run", "left", start=2 * ms).between(8.5 * V, 9.1 * V),
            minimum("run", "left", start=2 * ms) <= 0.3 * V,
        ]
    )
