# Copyright 2026 Maxence Faldor
"""A low-pass filter placed with a negative capacitance, which no circuit can hold."""

from jolty import Builder, Component, Role, component, devices
from jolty.units import Quantity, V, kohm, nF


@component
def low_pass(c: Builder, capacitance: Quantity) -> None:
    """A series resistor of 1 kΩ, then the capacitance to the ground."""
    source, output = c.port("input"), c.port("output")
    ground = c.port("ground", Role.GROUND)
    resistor = c.place(devices.resistor(1 * kohm), name="R1")
    capacitor = c.place(devices.capacitor(capacitance), name="C1")
    c.connect(source, resistor.a)
    c.connect(resistor.b, capacitor.a, output)
    c.connect(capacitor.b, ground)


@component
def bench(c: Builder) -> None:
    """The filter of -1 nF, fed 1 V."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1 * V), name="V1")
    stage = c.place(low_pass(-1 * nF), name="X1")
    c.connect(source.positive, stage.input, name="in")
    c.connect(stage.output, name="out")
    c.connect(source.negative, stage.ground, ground)


def build() -> Component:
    """The filter on its bench."""
    return bench()
