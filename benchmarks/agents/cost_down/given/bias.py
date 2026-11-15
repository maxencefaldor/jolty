# Copyright 2026 Maxence Faldor
"""A common-emitter stage's bias, its resistors 0.1 % parts.

A divider sets the base, the emitter resistor the current, and the collector resistor
the collector's voltage. The parts hold the emitter at 1 V within 100 mV and the
collector at 2.5 V within 300 mV on every sample of their tolerances: the baseline of
the cost-down task. Simulate it with ``jolty simulate bias.py``.
"""

from jolty import Builder, Component, MonteCarlo, Plan, Role, Specs, component, devices
from jolty.plan import OperatingPoint
from jolty.specs import value
from jolty.units import A, Quantity, V, kohm, mV

TOLERANCE = 0.001
"""The tolerance of every resistor."""


@component
def stage(
    c: Builder,
    upper: Quantity,
    lower: Quantity,
    emitter: Quantity,
    collector: Quantity,
) -> None:
    """An NPN transistor biased by a divider, with emitter and collector resistors."""
    supply = c.port("supply")
    output = c.port("output")
    ground = c.port("ground", Role.GROUND)
    transistor = c.place(
        devices.bipolar_transistor(
            polarity="npn",
            saturation_current=6.734e-15 * A,
            forward_beta=416.4,
            forward_early_voltage=74.03 * V,
        ),
        id="Q1",
    )
    top = c.place(devices.resistor(upper, tolerance=TOLERANCE), id="R1")
    bottom = c.place(devices.resistor(lower, tolerance=TOLERANCE), id="R2")
    degeneration = c.place(devices.resistor(emitter, tolerance=TOLERANCE), id="RE")
    load = c.place(devices.resistor(collector, tolerance=TOLERANCE), id="RC")
    c.connect(supply, top.a, load.a)
    c.connect(top.b, bottom.a, transistor.base, name="base")
    c.connect(load.b, transistor.collector, output)
    c.connect(transistor.emitter, degeneration.a, name="emitter")
    c.connect(bottom.b, degeneration.b, ground)


@component
def bench(c: Builder) -> None:
    """The stage, from a 5 V supply."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(5 * V), id="V1")
    biased = c.place(stage(33.2 * kohm, 16.5 * kohm, 1 * kohm, 2.49 * kohm), id="X1")
    c.connect(source.positive, biased.supply)
    c.connect(biased.output, name="out")
    c.connect(source.negative, biased.ground, ground)


def build() -> Component:
    """The stage on its bench."""
    return bench()


def plan() -> Plan:
    """The operating point, over 200 samples of the resistors' tolerances."""
    return Plan({"op": OperatingPoint()}, monte_carlo=MonteCarlo(200, seed=1))


def specs() -> Specs:
    """The emitter at 1 V within 100 mV, the collector at 2.5 V within 300 mV."""
    return Specs(
        [
            value("op", "X1/emitter").within(1 * V, 100 * mV),
            value("op", "out").within(2.5 * V, 300 * mV),
        ],
        yield_target=0.99,
    )
