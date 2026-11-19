# Copyright 2026 Maxence Faldor
"""An op-amp relaxation oscillator: a Schmitt trigger timing itself through an RC.

Two equal resistors feed half the output back to the non-inverting input, so the
op-amp's output flips whenever the capacitor, charging through R toward the rail,
crosses half of it. Each half period lasts RC·ln((1 + β)/(1 - β)) with β = 1/2: with
10 kΩ and 10 nF, a period of 2RC·ln 3 = 219.7 µs, 4.551 kHz, a square wave between the
op-amp's ±13 V rails. The op-amp's 13 V/µs slew rate stretches each edge by 2 µs, and
the period by some 1.5 %. Simulate it with
``jolty simulate examples/relaxation_oscillator.py``.
"""

import math

from common import SWING, opamp

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.plan import Transient
from jolty.specs import frequency, maximum, minimum
from jolty.units import Hz, V, kohm, ms, nF, uA, us
from jolty.waveforms import Pulse

PERIOD = 2 * 10e3 * 10e-9 * math.log(3)


@component
def astable(c: Builder) -> None:
    """Builds an op-amp with resistive positive feedback and RC negative feedback.

    At rest all nodes sit balanced at zero; a short current pulse into the divider
    initiates oscillation.
    """
    output = c.port("output", Role.OUTPUT)
    ground = c.port("ground", Role.GROUND)
    kick = Pulse(
        initial=0 * uA,
        pulsed=1 * uA,
        delay=1 * us,
        rise=0.1 * us,
        fall=0.1 * us,
        width=1 * us,
    )
    start = c.place(devices.current_source(kick), name="I1")
    amplifier = c.place(opamp(), name="U1")
    timing_r = c.place(devices.resistor(10 * kohm), name="R1")
    timing_c = c.place(devices.capacitor(10 * nF), name="C1")
    upper = c.place(devices.resistor(10 * kohm), name="R2")
    lower = c.place(devices.resistor(10 * kohm), name="R3")
    c.connect(amplifier.out, timing_r.a, upper.a, output)
    c.connect(timing_r.b, timing_c.a, amplifier.inn, name="timing")
    c.connect(upper.b, lower.a, amplifier.inp, start.negative, name="threshold")
    c.connect(timing_c.b, lower.b, amplifier.ground, start.positive, ground)


@component
def bench(c: Builder) -> None:
    """Builds the standalone relaxation oscillator test bench."""
    ground = c.port("ground", Role.GROUND)
    stage = c.place(astable(), name="X1")
    c.connect(stage.output, name="out")
    c.connect(stage.ground, ground)


def build() -> Component:
    """Returns the relaxation oscillator test bench component."""
    return bench()


def plan() -> Plan:
    """Returns a 2 ms Gear-2 transient plan covering roughly nine cycles."""
    return Plan({"run": Transient(stop=2 * ms, max_step=0.5 * us, method="gear2")})


def specs() -> Specs:
    """Returns specifications for period near 2RC·ln 3 and rail-to-rail swing."""
    return Specs(
        [
            frequency("run", "out", start=0.5 * ms).within(
                Hz / PERIOD, 0.02 * Hz / PERIOD
            ),
            maximum("run", "out", start=0.5 * ms).within(SWING * V, 0.3 * V),
            minimum("run", "out", start=0.5 * ms).within(-SWING * V, 0.3 * V),
        ]
    )
