# Copyright 2026 Maxence Faldor
"""A stage between two capacitors, with no DC path to the ground: `no-dc-path`."""

from jolty import Builder, Component, Role, component, devices
from jolty.units import V, kohm, nF


@component
def coupled(c: Builder) -> None:
    """A source, then a capacitor, a resistor and a capacitor to the ground."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1 * V), name="V1")
    first = c.place(devices.capacitor(10 * nF), name="C1")
    resistor = c.place(devices.resistor(10 * kohm), name="R1")
    second = c.place(devices.capacitor(10 * nF), name="C2")
    c.connect(source.positive, first.a, name="in")
    c.connect(first.b, resistor.a, name="mid")
    c.connect(resistor.b, second.a, name="out")
    c.connect(second.b, source.negative, ground)


def build() -> Component:
    """The coupled stage."""
    return coupled()
