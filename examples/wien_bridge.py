# Copyright 2026 Maxence Faldor
"""A Wien bridge oscillator, its amplitude held by two diodes.

The series and parallel 10 kΩ, 10 nF RC arms pass one third of the output with zero
phase shift at f₀ = 1/(2πRC) = 1.592 kHz, sustaining oscillation at a non-inverting
gain of 3. Start-up gain is 1 + 22 kΩ/10 kΩ = 3.2; as amplitude grows, two 1N4148
diodes across part of the feedback resistor reduce gain toward 3 to stabilize the
swing. A 1 µA start-up pulse initiates oscillation. Simulate with
``jolty simulate examples/wien_bridge.py``.
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
    """Builds the op-amp, Wien RC bridge, and diode-limited feedback network.

    A short current pulse into the bridge initiates oscillation.
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
    start = c.place(devices.current_source(kick), name="I1")
    amplifier = c.place(opamp(), name="U1")
    series_r = c.place(devices.resistor(10 * kohm), name="R1")
    series_c = c.place(devices.capacitor(10 * nF), name="C1")
    shunt_r = c.place(devices.resistor(10 * kohm), name="R2")
    shunt_c = c.place(devices.capacitor(10 * nF), name="C2")
    feedback = c.place(devices.resistor(12 * kohm), name="R3")
    limited = c.place(devices.resistor(10 * kohm), name="R4")
    gain = c.place(devices.resistor(10 * kohm), name="R5")
    forward, backward = (c.place(d1n4148(), name=name) for name in ("D1", "D2"))
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
    """Builds the standalone Wien bridge oscillator test bench."""
    ground = c.port("ground", Role.GROUND)
    stage = c.place(oscillator(), name="X1")
    c.connect(stage.output, name="out")
    c.connect(stage.ground, ground)


def build() -> Component:
    """Returns the Wien bridge oscillator test bench component."""
    return bench()


def plan() -> Plan:
    """Returns a 40 ms transient plan covering start-up, growth, and steady swing."""
    return Plan({"start": Transient(stop=40 * ms, max_step=5 * us)})


def specs() -> Specs:
    """Returns specifications for steady oscillation within 2 % of f₀."""
    return Specs(
        [
            frequency("start", "out", start=30 * ms).within(
                OSCILLATION * Hz, 0.02 * OSCILLATION * Hz
            ),
            peak_to_peak("start", "out", start=30 * ms).between(2 * V, 20 * V),
        ]
    )
