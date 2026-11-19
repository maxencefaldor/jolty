# Copyright 2026 Maxence Faldor
"""A nanoampere into a diode held off, whose voltage gmin sets: `gmin-dependence`."""

from jolty import Builder, Component, Plan, Role, component, devices
from jolty.plan import OperatingPoint
from jolty.units import nA


@component
def leakage(c: Builder) -> None:
    """Builds a 1 nA source driving the cathode of a grounded-anode diode."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.current_source(1 * nA), name="I1")
    diode = c.place(devices.diode(), name="D1")
    c.connect(source.negative, diode.cathode, name="n")
    c.connect(source.positive, diode.anode, ground)


def build() -> Component:
    """Returns the reverse-biased diode test circuit."""
    return leakage()


def plan() -> Plan:
    """Returns the operating-point simulation plan."""
    return Plan({"op": OperatingPoint()})
