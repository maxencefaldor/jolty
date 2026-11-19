# Copyright 2026 Maxence Faldor
"""Baseline common-emitter amplifier bias circuit using 0.1 % resistors.

A resistive divider sets the base bias, an emitter resistor sets quiescent current, and
a collector resistor sets collector voltage. The baseline holds the emitter at 1 V
within 100 mV and the collector at 2.5 V within 300 mV across Monte Carlo samples. Run
with ``jolty simulate bias.py``.
"""

from jolty import Builder, Component, MonteCarlo, Plan, Role, Specs, component, devices
from jolty.plan import OperatingPoint
from jolty.specs import value
from jolty.units import A, Quantity, V, kohm, mV

TOLERANCE = 0.001
"""Relative resistance tolerance applied to every resistor in the baseline stage."""


@component
def stage(
    c: Builder,
    upper: Quantity,
    lower: Quantity,
    emitter: Quantity,
    collector: Quantity,
) -> None:
    """Builds an NPN common-emitter stage with bias divider and load resistors."""
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
        name="Q1",
    )
    top = c.place(devices.resistor(upper, tolerance=TOLERANCE), name="R1")
    bottom = c.place(devices.resistor(lower, tolerance=TOLERANCE), name="R2")
    degeneration = c.place(devices.resistor(emitter, tolerance=TOLERANCE), name="RE")
    load = c.place(devices.resistor(collector, tolerance=TOLERANCE), name="RC")
    c.connect(supply, top.a, load.a)
    c.connect(top.b, bottom.a, transistor.base, name="base")
    c.connect(load.b, transistor.collector, output)
    c.connect(transistor.emitter, degeneration.a, name="emitter")
    c.connect(bottom.b, degeneration.b, ground)


@component
def bench(c: Builder) -> None:
    """Places the baseline bias stage onto a 5 V supply test bench."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(5 * V), name="V1")
    biased = c.place(stage(33.2 * kohm, 16.5 * kohm, 1 * kohm, 2.49 * kohm), name="X1")
    c.connect(source.positive, biased.supply)
    c.connect(biased.output, name="out")
    c.connect(source.negative, biased.ground, ground)


def build() -> Component:
    """Returns the complete common-emitter bias stage on its test bench."""
    return bench()


def plan() -> Plan:
    """Returns the operating-point simulation plan with 200 Monte Carlo samples."""
    return Plan({"op": OperatingPoint()}, monte_carlo=MonteCarlo(200, seed=1))


def specs() -> Specs:
    """Returns emitter and collector voltage window specifications at 99 % yield."""
    return Specs(
        [
            value("op", "X1/emitter").within(1 * V, 100 * mV),
            value("op", "out").within(2.5 * V, 300 * mV),
        ],
        yield_target=0.99,
    )
