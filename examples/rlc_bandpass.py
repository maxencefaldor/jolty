# Copyright 2026 Maxence Faldor
"""A series RLC band-pass filter, its output across the resistor.

With 10 mH, 100 nF and 100 Ω, it resonates at f₀ = 1/(2π√(LC)) = 5.033 kHz with a
quality factor Q = √(L/C)/R = 3.162, so its half-power band spans f₀/Q = 1.592 kHz,
between f₀(√(1 + 1/4Q²) ∓ 1/2Q). Simulate it with
``jolty simulate examples/rlc_bandpass.py``.
"""

import math

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.plan import Ac, OperatingPoint, Transient, decade
from jolty.specs import gain
from jolty.units import Hz, MHz, V, mH, ms, nF, ohm, us
from jolty.waveforms import Sine

INDUCTANCE, CAPACITANCE, RESISTANCE = 10e-3, 100e-9, 100.0
RESONANCE = 1 / (2 * math.pi * math.sqrt(INDUCTANCE * CAPACITANCE))
QUALITY = math.sqrt(INDUCTANCE / CAPACITANCE) / RESISTANCE
EDGES = tuple(
    RESONANCE * (math.sqrt(1 + 1 / (4 * QUALITY**2)) + sign / (2 * QUALITY))
    for sign in (-1, 1)
)


@component
def bench(c: Builder) -> None:
    """The inductor, the capacitor and the resistor in series across a source."""
    ground = c.port("ground", Role.GROUND)
    tone = Sine(offset=0 * V, amplitude=1 * V, frequency=RESONANCE * Hz)
    source = c.place(devices.voltage_source(tone), id="V1")
    inductor = c.place(devices.inductor(INDUCTANCE * 1e3 * mH), id="L1")
    capacitor = c.place(devices.capacitor(CAPACITANCE * 1e9 * nF), id="C1")
    resistor = c.place(devices.resistor(RESISTANCE * ohm), id="R1")
    c.connect(source.positive, inductor.a, name="in")
    c.connect(inductor.b, capacitor.a, name="mid")
    c.connect(capacitor.b, resistor.a, name="out")
    c.connect(resistor.b, source.negative, ground)


def build() -> Component:
    """The filter on its bench."""
    return bench()


def plan() -> Plan:
    """The frequency response, and ten periods at resonance."""
    return Plan(
        {
            "op": OperatingPoint(),
            "bode": Ac(
                "V1",
                sorted(
                    [
                        *decade(100 * Hz, 1 * MHz, 50),
                        *(f * Hz for f in (RESONANCE, *EDGES)),
                    ],
                    key=lambda frequency: frequency.value,
                ),
            ),
            "ring": Transient(stop=2 * ms, max_step=2 * us),
        }
    )


def specs() -> Specs:
    """Unity gain at resonance, half the power at each edge of the band."""
    return Specs(
        [
            gain("bode", "out", at=RESONANCE * Hz).within(0, 1e-9),
            *(
                gain("bode", "out", at=edge * Hz).within(-3.0103, 1e-4)
                for edge in EDGES
            ),
        ]
    )
