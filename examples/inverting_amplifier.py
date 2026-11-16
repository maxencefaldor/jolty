# Copyright 2026 Maxence Faldor
"""An inverting amplifier: a gain of -10 set by two resistors around an op-amp.

The op-amp's finite gain A = 10⁵ takes the closed-loop gain to -10/(1 + 11/A), and its
1 MHz gain-bandwidth product sets the bandwidth at 1 MHz/11 = 90.9 kHz, where the gain
is 3 dB down. Simulate it with ``jolty simulate examples/inverting_amplifier.py``.
"""

import math

from common import opamp

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.plan import Ac, OperatingPoint, Transient, decade
from jolty.specs import bandwidth, gain, peak_to_peak
from jolty.units import Hz, MHz, V, kHz, kohm, ms, mV, us
from jolty.waveforms import Sine

OPEN_LOOP = 1e5
CLOSED_LOOP = 10 / (1 + 11 / OPEN_LOOP)


@component
def inverting(c: Builder) -> None:
    """Rin from the input to the inverting input, Rf from there to the output."""
    source, output = c.port("input"), c.port("output")
    ground = c.port("ground", Role.GROUND)
    series = c.place(devices.resistor(1 * kohm), name="Rin")
    feedback = c.place(devices.resistor(10 * kohm), name="Rf")
    amplifier = c.place(opamp(), name="U1")
    c.connect(source, series.a)
    c.connect(series.b, feedback.a, amplifier.inn, name="summing")
    c.connect(feedback.b, amplifier.out, output)
    c.connect(amplifier.inp, amplifier.ground, ground)


@component
def bench(c: Builder) -> None:
    """The amplifier, fed 100 mV at 1 kHz."""
    ground = c.port("ground", Role.GROUND)
    tone = Sine(offset=0 * V, amplitude=100 * mV, frequency=1 * kHz)
    source = c.place(devices.voltage_source(tone), name="V1")
    stage = c.place(inverting(), name="X1")
    c.connect(source.positive, stage.input, name="in")
    c.connect(stage.output, name="out")
    c.connect(source.negative, stage.ground, ground)


def build() -> Component:
    """The amplifier on its bench."""
    return bench()


def plan() -> Plan:
    """The operating point, the frequency response and three periods of the tone."""
    return Plan(
        {
            "op": OperatingPoint(),
            "bode": Ac("V1", decade(10 * Hz, 10 * MHz, 40)),
            "tone": Transient(stop=3 * ms, max_step=5 * us),
        }
    )


def specs() -> Specs:
    """A gain of 20 dB, 90.9 kHz of bandwidth, 2 V peak to peak at the output."""
    return Specs(
        [
            gain("bode", "out", at=100 * Hz).within(20 * math.log10(CLOSED_LOOP), 1e-4),
            bandwidth("bode", "out").within(90.9 * kHz, 0.5 * kHz),
            peak_to_peak("tone", "out", start=1 * ms).within(
                2 * CLOSED_LOOP * 0.1 * V, 2 * mV
            ),
        ]
    )
