# Copyright 2026 Maxence Faldor
"""A Colpitts oscillator: a common-base 2N3904 with an LC tank and capacitive divider.

The 100 µH inductor resonates with the 1 nF and 10 nF series capacitors (0.909 nF
equivalent) at 1/(2π√(LC)) = 528 kHz. The capacitive divider feeds one tenth of the
collector swing back to the emitter in phase, and the supply ramp initiates oscillation.
Simulate with ``jolty simulate examples/colpitts_oscillator.py``.
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
    """Builds the common-base transistor, bias network, and capacitive LC tank."""
    output = c.port("output", Role.OUTPUT)
    supply, ground = c.port("vcc", Role.SUPPLY), c.port("ground", Role.GROUND)
    transistor = c.place(q2n3904(), name="Q1")
    upper = c.place(devices.resistor(22 * kohm), name="R1")
    lower = c.place(devices.resistor(10 * kohm), name="R2")
    emitter = c.place(devices.resistor(1 * kohm), name="RE")
    decoupling = c.place(devices.capacitor(1 * uF), name="CB")
    coil = c.place(devices.inductor(100 * uH), name="L1")
    top = c.place(devices.capacitor(1 * nF), name="C1")
    bottom = c.place(devices.capacitor(10 * nF), name="C2")
    c.connect(supply, upper.a, coil.a)
    c.connect(upper.b, lower.a, decoupling.a, transistor.base, name="base")
    c.connect(coil.b, transistor.collector, top.a, output)
    c.connect(top.b, bottom.a, transistor.emitter, emitter.a, name="emitter")
    c.connect(lower.b, decoupling.b, bottom.b, emitter.b, ground)


@component
def bench(c: Builder) -> None:
    """Builds the oscillator powered by a 9 V supply ramping over 1 µs."""
    ground = c.port("ground", Role.GROUND)
    ramp = Pulse(
        initial=0 * V,
        pulsed=9 * V,
        delay=0 * us,
        rise=1 * us,
        fall=1 * us,
        width=1e6 * us,
    )
    supply = c.place(devices.voltage_source(ramp), name="VCC")
    stage = c.place(colpitts(), name="X1")
    c.connect(supply.positive, stage.vcc, name="vcc")
    c.connect(stage.output, name="out")
    c.connect(supply.negative, stage.ground, ground)


def build() -> Component:
    """Returns the Colpitts oscillator test bench component."""
    return bench()


def plan() -> Plan:
    """Returns a 100 µs transient plan covering start-up and roughly 40 cycles."""
    return Plan({"run": Transient(stop=100 * us, max_step=10 * ns)})


def specs() -> Specs:
    """Returns specifications for oscillation within 3 % of tank resonance and swing."""
    return Specs(
        [
            frequency("run", "out", start=80 * us).within(TANK * Hz, 0.03 * TANK * Hz),
            peak_to_peak("run", "out", start=80 * us) >= 2 * V,
        ]
    )
