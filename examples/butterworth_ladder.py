# Copyright 2026 Maxence Faldor
"""A fifth-order Butterworth low-pass filter: an LC ladder between two 50 Ω ends.

Its elements are the Butterworth prototype's, 0.618, 1.618, 2, 1.618 and 0.618, scaled
to a 1 MHz cutoff and 50 Ω. Between its terminations its gain is half, times
1/√(1 + (f/f_c)¹⁰): -6.02 dB in its passband, 3.01 dB less at the cutoff, and falling
100 dB a decade above it. Simulate it with
``jolty simulate examples/butterworth_ladder.py``.
"""

import math

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.plan import Ac, Transient, decade
from jolty.specs import gain
from jolty.units import F, H, Hz, MHz, V, kHz, ns, ohm, us
from jolty.waveforms import Pulse

CUTOFF = 1e6
"""The cutoff frequency, in hertz."""
IMPEDANCE = 50.0
"""The terminations, in ohms."""
PROTOTYPE = [2 * math.sin((2 * k - 1) * math.pi / 10) for k in range(1, 6)]
"""The Butterworth prototype's elements, g₁ to g₅."""


def butterworth(frequency: float) -> float:
    """The ladder's gain between its terminations at `frequency`, in decibels."""
    return 20 * math.log10(0.5 / math.sqrt(1 + (frequency / CUTOFF) ** 10))


@component
def ladder(c: Builder) -> None:
    """Shunt capacitors and series inductors, alternating, from input to output."""
    source, output = c.port("input"), c.port("output")
    ground = c.port("ground", Role.GROUND)
    omega = 2 * math.pi * CUTOFF
    node = source
    for index, element in enumerate(PROTOTYPE, start=1):
        if index % 2:
            shunt = c.place(
                devices.capacitor(element / (omega * IMPEDANCE) * F), name=f"C{index}"
            )
            c.connect(shunt.a, node)
            c.connect(shunt.b, ground)
        else:
            series = c.place(
                devices.inductor(element * IMPEDANCE / omega * H), name=f"L{index}"
            )
            c.connect(series.a, node)
            node = c.net(f"n{index}") if index < len(PROTOTYPE) - 1 else output
            c.connect(series.b, node)
    c.connect(node, output)


@component
def bench(c: Builder) -> None:
    """The ladder between a 50 Ω source and a 50 Ω load."""
    ground = c.port("ground", Role.GROUND)
    step = Pulse(
        initial=0 * V,
        pulsed=1 * V,
        delay=100 * ns,
        rise=1 * ns,
        fall=1 * ns,
        width=10 * us,
    )
    source = c.place(devices.voltage_source(step), name="V1")
    inside = c.place(devices.resistor(IMPEDANCE * ohm), name="RS")
    filter_ = c.place(ladder(), name="X1")
    load = c.place(devices.resistor(IMPEDANCE * ohm), name="RL")
    c.connect(source.positive, inside.a, name="in")
    c.connect(inside.b, filter_.input, name="driven")
    c.connect(filter_.output, load.a, name="out")
    c.connect(source.negative, filter_.ground, load.b, ground)


def build() -> Component:
    """The ladder between its terminations."""
    return bench()


def plan() -> Plan:
    """The frequency response, and the step response."""
    return Plan(
        {
            "bode": Ac("V1", decade(10 * kHz, 100 * MHz, 40)),
            "step": Transient(stop=5 * us, max_step=5 * ns),
        }
    )


def specs() -> Specs:
    """The Butterworth magnitude to a thousandth of a decibel, from 10 kHz to 10 MHz."""
    return Specs(
        [
            gain("bode", "out", at=frequency * Hz).within(butterworth(frequency), 1e-3)
            for frequency in (1e4, 5e5, 1e6, 2e6, 1e7)
        ]
    )
