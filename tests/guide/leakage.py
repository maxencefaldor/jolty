# Copyright 2026 Maxence Faldor
"""A nanoampere into a diode held off, whose voltage gmin sets: `gmin-dependence`."""

from jolty import Builder, Component, Plan, Role, component, devices
from jolty.plan import OperatingPoint
from jolty.units import nA


@component
def leakage(c: Builder) -> None:
    """1 nA into the cathode of a diode whose anode is the ground."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.current_source(1 * nA), name="I1")
    diode = c.place(devices.diode(), name="D1")
    c.connect(source.negative, diode.cathode, name="n")
    c.connect(source.positive, diode.anode, ground)


def build() -> Component:
    """The diode and its current."""
    return leakage()


def plan() -> Plan:
    """The operating point."""
    return Plan({"op": OperatingPoint()})
