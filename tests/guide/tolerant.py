# Copyright 2026 Maxence Faldor
"""A divider of a 1 % uniform resistor over a 1 % normal one, over 1000 samples."""

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.plan import MonteCarlo, OperatingPoint
from jolty.specs import value
from jolty.units import V, kohm


@component
def divider(c: Builder) -> None:
    """Builds a 10 kΩ over 10 kΩ toleranced divider driven by 5 V."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(5 * V), name="V1")
    upper = c.place(devices.resistor(10 * kohm, tolerance=0.01), name="R1")
    lower = devices.resistor(10 * kohm).toleranced("resistance", 0.01, "normal")
    placed = c.place(lower, name="R2")
    c.connect(source.positive, upper.a, name="in")
    c.connect(upper.b, placed.a, name="out")
    c.connect(placed.b, source.negative, ground)


def build() -> Component:
    """Returns the toleranced divider component."""
    return divider()


def plan() -> Plan:
    """Returns the 1000-sample Monte Carlo operating-point plan."""
    return Plan({"op": OperatingPoint()}, monte_carlo=MonteCarlo(1000, seed=1))


def specs() -> Specs:
    """Returns the 2.5 V ± 15 mV specification with a 90 % yield target."""
    return Specs([value("op", "out").within(2.5 * V, 15e-3 * V)], yield_target=0.9)
