# Copyright 2026 Maxence Faldor
"""A unity-gain Sallen-Key low-pass filter: the second-order Butterworth response.

Two 10 kΩ resistors, 20 nF from their junction back to the output and 10 nF to ground
make ω₀ = 1/√(R₁R₂C₁C₂) and Q = √(R₁R₂C₁C₂)/((R₁ + R₂)C₂) = 1/√2: a cutoff of
1.125 kHz with no peaking, -3.01 dB there and -40 dB a decade beyond. A follower drives
the output: the op-amp of the shared parts, 100 dB of gain and 1 MHz of bandwidth.
Simulate it with ``jolty simulate examples/sallen_key.py``.
"""

import math

from common import opamp

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.plan import Ac, Transient, decade
from jolty.specs import gain, overshoot
from jolty.units import Hz, MHz, V, kohm, ms, nF, us
from jolty.waveforms import Pulse

CUTOFF = 1 / (2 * math.pi * math.sqrt(10e3 * 10e3 * 20e-9 * 10e-9))


@component
def sallen_key(c: Builder) -> None:
    """The two resistors, the two capacitors and the follower."""
    source, output = c.port("input"), c.port("output")
    ground = c.port("ground", Role.GROUND)
    first = c.place(devices.resistor(10 * kohm), name="R1")
    second = c.place(devices.resistor(10 * kohm), name="R2")
    feedback = c.place(devices.capacitor(20 * nF), name="C1")
    shunt = c.place(devices.capacitor(10 * nF), name="C2")
    amplifier = c.place(opamp(), name="U1")
    c.connect(source, first.a)
    c.connect(first.b, second.a, feedback.a, name="junction")
    c.connect(second.b, shunt.a, amplifier.inp, name="plus")
    c.connect(amplifier.out, amplifier.inn, feedback.b, output)
    c.connect(shunt.b, amplifier.ground, ground)


@component
def bench(c: Builder) -> None:
    """The filter, stepped by 1 V at 0.1 ms."""
    ground = c.port("ground", Role.GROUND)
    step = Pulse(
        initial=0 * V,
        pulsed=1 * V,
        delay=0.1 * ms,
        rise=1 * us,
        fall=1 * us,
        width=4 * ms,
    )
    source = c.place(devices.voltage_source(step), name="V1")
    stage = c.place(sallen_key(), name="X1")
    c.connect(source.positive, stage.input, name="in")
    c.connect(stage.output, name="out")
    c.connect(source.negative, stage.ground, ground)


def build() -> Component:
    """The filter on its bench."""
    return bench()


def plan() -> Plan:
    """The frequency response, and the step response."""
    return Plan(
        {
            "bode": Ac("V1", decade(10 * Hz, 1 * MHz, 40)),
            "step": Transient(stop=3 * ms, max_step=5 * us),
        }
    )


def specs() -> Specs:
    """Butterworth's -3.01 dB at the cutoff, -40 dB a decade above, 4.3 % overshoot."""
    return Specs(
        [
            gain("bode", "out", at=10 * Hz).within(0, 0.001),
            gain("bode", "out", at=CUTOFF * Hz).within(-3.0103, 0.01),
            gain("bode", "out", at=10 * CUTOFF * Hz).within(-40.0004, 0.05),
            overshoot("step", "out").within(4.32, 0.05),
        ]
    )
