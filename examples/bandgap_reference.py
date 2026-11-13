# Copyright 2026 Maxence Faldor
"""A bandgap voltage reference: two transistors' base-emitter voltages cancelling drift.

An op-amp holds its two inputs equal, so equal resistors carry equal currents into a
2N3904 and into eight in parallel behind 1 kΩ. The resistor then holds the difference
of their base-emitter voltages, Vt·ln 8, proportional to absolute temperature, and the
output adds R₁/R₃ times it to a base-emitter voltage, which falls with temperature.
With R₁/R₃ = 10.7 the two drifts cancel: the output stays within 2.2 mV of 1.168 V from
-40 °C to 125 °C, some 13 ppm/K, near silicon's band gap less its curvature. Ten
megohms from a 5 V supply start it. Simulate it with
``jolty simulate examples/bandgap_reference.py``.
"""

from common import opamp

from jolty import Builder, Component, Corner, Plan, Role, Specs, component, devices
from jolty.parts import q2n3904
from jolty.plan import OperatingPoint
from jolty.specs import value
from jolty.units import Mohm, V, celsius, kohm, mV

RATIO = 10.7


@component
def bandgap(c: Builder) -> None:
    """The op-amp, its two branches of resistors and transistors, and the start-up."""
    output = c.port("output", Role.OUTPUT)
    supply, ground = c.port("vcc", Role.SUPPLY), c.port("ground", Role.GROUND)
    amplifier = c.place(opamp(), id="U1")
    single = c.place(q2n3904(), id="Q1")
    wide = [c.place(q2n3904(), id=f"Q{index}") for index in range(2, 10)]
    left = c.place(devices.resistor(RATIO * kohm), id="R1")
    right = c.place(devices.resistor(RATIO * kohm), id="R2")
    ptat = c.place(devices.resistor(1 * kohm), id="R3")
    start = c.place(devices.resistor(10 * Mohm), id="RS")
    c.connect(amplifier.out, left.a, right.a, output)
    c.connect(left.b, single.collector, single.base, amplifier.inp, start.b, name="a")
    c.connect(right.b, ptat.a, amplifier.inn, name="b")
    c.connect(ptat.b, *(q.collector for q in wide), *(q.base for q in wide), name="c")
    c.connect(supply, start.a)
    c.connect(single.emitter, *(q.emitter for q in wide), amplifier.ground, ground)


@component
def bench(c: Builder) -> None:
    """The reference on a 5 V supply."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(5 * V), id="VCC")
    reference = c.place(bandgap(), id="X1")
    c.connect(supply.positive, reference.vcc, name="vcc")
    c.connect(reference.output, name="vref")
    c.connect(supply.negative, reference.ground, ground)


def build() -> Component:
    """The reference on its bench."""
    return bench()


def plan() -> Plan:
    """The operating point at 27 °C, and at -40 °C, 85 °C and 125 °C."""
    return Plan(
        {"op": OperatingPoint()},
        corners={
            "cold": Corner(temperature=celsius(-40)),
            "warm": Corner(temperature=celsius(85)),
            "hot": Corner(temperature=celsius(125)),
        },
    )


def specs() -> Specs:
    """Within 1.168 V ± 3 mV at every corner, from -40 °C to 125 °C."""
    return Specs([value("op", "vref").within(1.168 * V, 3 * mV)])
