# Copyright 2026 Maxence Faldor
"""A three-op-amp instrumentation amplifier: a gain of 10.09 that rejects common mode.

Two followers with 10 kΩ feedback resistors share a 2.2 kΩ gain resistor, a gain of
1 + 2·10/2.2 = 10.09 to the difference of their inputs and of one to what the inputs
share; a difference amplifier of four matched 10 kΩ resistors then keeps the difference
alone. A 10 mV difference riding on 2 V comes out as 100.9 mV. Simulate it with
``jolty simulate examples/instrumentation_amplifier.py``.
"""

from common import opamp

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.plan import DcSweep, OperatingPoint, linear
from jolty.specs import value
from jolty.units import V, kohm, mV

GAIN = 1 + 2 * 10 / 2.2


@component
def instrumentation(c: Builder) -> None:
    """Two input followers sharing a gain resistor, then a difference amplifier."""
    plus, minus = c.port("plus", Role.INPUT), c.port("minus", Role.INPUT)
    output = c.port("output", Role.OUTPUT)
    ground = c.port("ground", Role.GROUND)
    first, second, third = (c.place(opamp(), name=f"U{index}") for index in (1, 2, 3))
    feedback_a = c.place(devices.resistor(10 * kohm), name="R1")
    feedback_b = c.place(devices.resistor(10 * kohm), name="R2")
    gain = c.place(devices.resistor(2.2 * kohm), name="RG")
    ins = [c.place(devices.resistor(10 * kohm), name=f"R{index}") for index in (3, 4)]
    feedback = c.place(devices.resistor(10 * kohm), name="R5")
    shunt = c.place(devices.resistor(10 * kohm), name="R6")
    c.connect(plus, first.inp)
    c.connect(minus, second.inp)
    c.connect(first.inn, feedback_a.a, gain.a, name="top")
    c.connect(second.inn, feedback_b.a, gain.b, name="bottom")
    c.connect(first.out, feedback_a.b, ins[0].a, name="a")
    c.connect(second.out, feedback_b.b, ins[1].a, name="b")
    c.connect(ins[1].b, third.inn, feedback.a, name="inverting")
    c.connect(ins[0].b, third.inp, shunt.a, name="noninverting")
    c.connect(third.out, feedback.b, output)
    c.connect(shunt.b, first.ground, second.ground, third.ground, ground)


@component
def bench(c: Builder) -> None:
    """The amplifier reading a 10 mV difference on 2 V of common mode."""
    ground = c.port("ground", Role.GROUND)
    common = c.place(devices.voltage_source(2 * V), name="VCM")
    difference = c.place(devices.voltage_source(10 * mV), name="VD")
    amplifier = c.place(instrumentation(), name="X1")
    c.connect(common.positive, difference.negative, amplifier.minus, name="minus")
    c.connect(difference.positive, amplifier.plus, name="plus")
    c.connect(amplifier.output, name="out")
    c.connect(common.negative, amplifier.ground, ground)


def build() -> Component:
    """The amplifier on its bench."""
    return bench()


def plan() -> Plan:
    """The operating point, the difference swept, and the common mode swept."""
    return Plan(
        {
            "op": OperatingPoint(),
            "difference": DcSweep("VD", linear(-0.5 * V, 0.5 * V, 41)),
            "common": DcSweep("VCM", linear(-5 * V, 5 * V, 41)),
        }
    )


def specs() -> Specs:
    """The difference times 10.09, and the common mode rejected."""
    return Specs(
        [
            value("op", "out").within(GAIN * 10 * mV, 0.05 * mV),
            value("difference", "out", at=0.5 * V).within(GAIN * 0.5 * V, 5 * mV),
            value("common", "out", at=-5 * V).within(GAIN * 10 * mV, 0.05 * mV),
            value("common", "out", at=5 * V).within(GAIN * 10 * mV, 0.05 * mV),
        ]
    )
