# Copyright 2026 Maxence Faldor
"""A node whose current is about `v² - 2v + 0.1`, zero at two voltages.

The current leaves `x` through a behavioral source, and a resistor of 1 GΩ gives `x` a
DC path to the ground, so its DC solutions are near the roots `1 ± √0.9` V:
`multiple-operating-points`.
"""

from jolty import Builder, Component, Plan, Role, component, devices
from jolty.plan import OperatingPoint
from jolty.units import Mohm


@component
def bistable(c: Builder) -> None:
    """The behavioral source and the resistor from `x` to the ground."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(
        devices.behavioral_current_source("v(x) * v(x) - 2 * v(x) + 0.1"), name="B1"
    )
    leak = c.place(devices.resistor(1000 * Mohm), name="R1")
    c.connect(source.positive, source.x, leak.a, name="x")
    c.connect(source.negative, leak.b, ground)


def build() -> Component:
    """The node and its source."""
    return bistable()


def plan() -> Plan:
    """The operating point."""
    return Plan({"op": OperatingPoint()})
