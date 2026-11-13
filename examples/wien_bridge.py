# Copyright 2026 Maxence Faldor
"""A Wien bridge oscillator, its amplitude held by two diodes.

The bridge's series and parallel RC arms, 10 kΩ and 10 nF each, pass a third of the
output with no phase shift at f₀ = 1/(2πRC) = 1.592 kHz, so the loop oscillates there
when the op-amp's non-inverting gain is three. Its gain is 1 + 22 kΩ/10 kΩ = 3.2, enough
to start; as the swing grows, two 1N4148 diodes across part of the feedback resistor
conduct and bring the gain to three, holding the amplitude. A 1 µA pulse at the start
sets it going. Simulate it with ``jolty simulate examples/wien_bridge.py``.
"""

import math

from common import opamp

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.parts import d1n4148
from jolty.plan import Transient
from jolty.specs import frequency, peak_to_peak
from jolty.units import Hz, V, kohm, ms, nF, uA, us
from jolty.waveforms import Pulse

OSCILLATION = 1 / (2 * math.pi * 10e3 * 10e-9)


@component
def oscillator(c: Builder) -> None:
    """The op-amp, the RC bridge to its non-inverting input, and its feedback.

    A short current pulse into the bridge sets it going.
    """
    output = c.port("output", Role.OUTPUT)
    ground = c.port("ground", Role.GROUND)
    kick = Pulse(
        initial=0 * uA,
        pulsed=1 * uA,
        delay=10 * us,
        rise=1 * us,
        fall=1 * us,
        width=10 * us,
    )
    start = c.place(devices.current_source(kick), id="I1")
    amplifier = c.place(opamp(), id="U1")
    series_r = c.place(devices.resistor(10 * kohm), id="R1")
    series_c = c.place(devices.capacitor(10 * nF), id="C1")
    shunt_r = c.place(devices.resistor(10 * kohm), id="R2")
    shunt_c = c.place(devices.capacitor(10 * nF), id="C2")
    feedback = c.place(devices.resistor(12 * kohm), id="R3")
    limited = c.place(devices.resistor(10 * kohm), id="R4")
    gain = c.place(devices.resistor(10 * kohm), id="R5")
    forward, backward = (c.place(d1n4148(), id=name) for name in ("D1", "D2"))
    c.connect(amplifier.out, series_r.a, feedback.a, output)
    c.connect(series_r.b, series_c.a, name="arm")
    c.connect(
        series_c.b, shunt_r.a, shunt_c.a, amplifier.inp, start.negative, name="plus"
    )
    c.connect(feedback.b, limited.a, forward.anode, backward.cathode, name="tap")
    c.connect(
        limited.b, forward.cathode, backward.anode, gain.a, amplifier.inn, name="minus"
    )
    c.connect(shunt_r.b, shunt_c.b, gain.b, amplifier.ground, start.positive, ground)


@component
def bench(c: Builder) -> None:
    """The oscillator alone."""
    ground = c.port("ground", Role.GROUND)
    stage = c.place(oscillator(), id="X1")
    c.connect(stage.output, name="out")
    c.connect(stage.ground, ground)


def build() -> Component:
    """The oscillator on its bench."""
    return bench()


def plan() -> Plan:
    """Forty milliseconds: the start, the growth, and a steady swing."""
    return Plan({"start": Transient(stop=40 * ms, max_step=5 * us)})


def specs() -> Specs:
    """Oscillation at f₀ within 2 %, at a steady swing."""
    return Specs(
        [
            frequency("start", "out", start=30 * ms).within(
                OSCILLATION * Hz, 0.02 * OSCILLATION * Hz
            ),
            peak_to_peak("start", "out", start=30 * ms).between(2 * V, 20 * V),
        ]
    )
