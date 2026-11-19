# Copyright 2026 Maxence Faldor
"""A twin-T notch filter: two T networks in parallel that cancel at one frequency.

One T is two resistors R with a capacitor 2C to ground; the other two capacitors C with
a resistor R/2 to ground. With R = 10 kΩ and C = 10 nF, their outputs cancel exactly at
f₀ = 1/(2πRC) = 1.592 kHz, and the filter passes the frequencies far from it. Simulate
it with ``jolty simulate examples/twin_t_notch.py``.
"""

import math

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.plan import Ac, decade
from jolty.specs import gain
from jolty.units import Hz, MHz, Mohm, V, kohm, nF

RESISTANCE, CAPACITANCE = 10e3, 10e-9
NOTCH = 1 / (2 * math.pi * RESISTANCE * CAPACITANCE)


@component
def twin_t(c: Builder) -> None:
    """Builds the parallel resistive T and capacitive T notch networks."""
    source, output = c.port("input"), c.port("output")
    ground = c.port("ground", Role.GROUND)
    r1 = c.place(devices.resistor(10 * kohm), name="R1")
    r2 = c.place(devices.resistor(10 * kohm), name="R2")
    c3 = c.place(devices.capacitor(20 * nF), name="C3")
    c1 = c.place(devices.capacitor(10 * nF), name="C1")
    c2 = c.place(devices.capacitor(10 * nF), name="C2")
    r3 = c.place(devices.resistor(5 * kohm), name="R3")
    c.connect(source, r1.a, c1.a)
    c.connect(r1.b, r2.a, c3.a, name="resistive")
    c.connect(c1.b, c2.a, r3.a, name="capacitive")
    c.connect(r2.b, c2.b, output)
    c.connect(c3.b, r3.b, ground)


@component
def bench(c: Builder) -> None:
    """Builds the twin-T notch filter driven by a voltage source into a 10 MΩ load."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(0 * V), name="V1")
    notch = c.place(twin_t(), name="X1")
    load = c.place(devices.resistor(10 * Mohm), name="RL")
    c.connect(source.positive, notch.input, name="in")
    c.connect(notch.output, load.a, name="out")
    c.connect(source.negative, notch.ground, load.b, ground)


def build() -> Component:
    """Returns the twin-T notch filter test bench component."""
    return bench()


def plan() -> Plan:
    """Returns an AC sweep sampled densely around the notch frequency."""
    frequencies = sorted(
        [*decade(10 * Hz, 1 * MHz, 60), NOTCH * Hz],
        key=lambda frequency: frequency.value,
    )
    return Plan({"bode": Ac("V1", frequencies)})


def specs() -> Specs:
    """Returns specifications for unity passband gain and deep notch attenuation."""
    return Specs(
        [
            gain("bode", "out", at=10 * Hz).between(-0.03, 0),
            gain("bode", "out", at=NOTCH * Hz) <= -100,
        ]
    )
