# Copyright 2026 Maxence Faldor
"""A Colpitts oscillator: a 2N3904 in common base, its tank a coil and two capacitors.

The tank's 100 µH resonates with its capacitors in series, 1 nF and 10 nF making
0.909 nF, at 1/(2π√(LC)) = 528 kHz; their divider feeds the emitter a tenth of the
collector's swing in phase, closing the loop. The supply's rise starts it. Simulate it
with ``jolty simulate examples/colpitts_oscillator.py``.
"""

import math

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.parts import q2n3904
from jolty.plan import Transient
from jolty.specs import frequency, peak_to_peak
from jolty.units import Hz, V, kohm, nF, ns, uF, uH, us
from jolty.waveforms import Pulse

TANK = 1 / (2 * math.pi * math.sqrt(100e-6 * 1e-9 * 10e-9 / 11e-9))


@component
def colpitts(c: Builder) -> None:
    """The transistor, its bias, and the tank between collector, emitter and ground."""
    output = c.port("output", Role.OUTPUT)
    supply, ground = c.port("vcc", Role.SUPPLY), c.port("ground", Role.GROUND)
    transistor = c.place(q2n3904(), id="Q1")
    upper = c.place(devices.resistor(22 * kohm), id="R1")
    lower = c.place(devices.resistor(10 * kohm), id="R2")
    emitter = c.place(devices.resistor(1 * kohm), id="RE")
    decoupling = c.place(devices.capacitor(1 * uF), id="CB")
    coil = c.place(devices.inductor(100 * uH), id="L1")
    top = c.place(devices.capacitor(1 * nF), id="C1")
    bottom = c.place(devices.capacitor(10 * nF), id="C2")
    c.connect(supply, upper.a, coil.a)
    c.connect(upper.b, lower.a, decoupling.a, transistor.base, name="base")
    c.connect(coil.b, transistor.collector, top.a, output)
    c.connect(top.b, bottom.a, transistor.emitter, emitter.a, name="emitter")
    c.connect(lower.b, decoupling.b, bottom.b, emitter.b, ground)


@component
def bench(c: Builder) -> None:
    """The oscillator on 9 V, rising in 1 µs."""
    ground = c.port("ground", Role.GROUND)
    ramp = Pulse(
        initial=0 * V,
        pulsed=9 * V,
        delay=0 * us,
        rise=1 * us,
        fall=1 * us,
        width=1e6 * us,
    )
    supply = c.place(devices.voltage_source(ramp), id="VCC")
    stage = c.place(colpitts(), id="X1")
    c.connect(supply.positive, stage.vcc, name="vcc")
    c.connect(stage.output, name="out")
    c.connect(supply.negative, stage.ground, ground)


def build() -> Component:
    """The oscillator on its bench."""
    return bench()


def plan() -> Plan:
    """A hundred microseconds: the start and some forty cycles."""
    return Plan({"run": Transient(stop=100 * us, max_step=10 * ns)})


def specs() -> Specs:
    """Oscillation within 3 % of the tank's resonance, with a few volts of swing."""
    return Specs(
        [
            frequency("run", "out", start=80 * us).within(TANK * Hz, 0.03 * TANK * Hz),
            peak_to_peak("run", "out", start=80 * us) >= 2 * V,
        ]
    )
