# Copyright 2026 Maxence Faldor
"""A divider whose upper resistance, 4.8 kΩ, no E series holds: `non-standard-value`."""

from jolty import Builder, Component, Role, component, devices
from jolty.units import V, ohm


@component
def divider(c: Builder) -> None:
    """4.8 kΩ over 4.7 kΩ, fed 5 V."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(5 * V), name="V1")
    upper = c.place(devices.resistor(4800 * ohm), name="R1")
    lower = c.place(devices.resistor(4700 * ohm), name="R2")
    c.connect(source.positive, upper.a, name="in")
    c.connect(upper.b, lower.a, name="out")
    c.connect(lower.b, source.negative, ground)


def build() -> Component:
    """The divider."""
    return divider()
