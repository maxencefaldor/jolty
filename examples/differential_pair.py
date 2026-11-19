# Copyright 2026 Maxence Faldor
"""A long-tailed pair: two 2N3904s sharing a 1 mA tail, steering it between them.

The base voltage difference splits the tail current as I·tanh(Vd/2Vt). The differential
output swings ±I·Rc = ±10 V with a small-signal gain of gm·Rc = (I/2)/Vt·Rc (about 193).
Simulate with ``jolty simulate examples/differential_pair.py``.
"""

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.parts import q2n3904
from jolty.plan import DcSweep, OperatingPoint, linear
from jolty.specs import value
from jolty.units import V, kohm, mA, mV


@component
def pair(c: Builder) -> None:
    """Builds two transistors with collector loads sharing a common emitter tail."""
    plus, minus = c.port("plus", Role.INPUT), c.port("minus", Role.INPUT)
    left, right = c.port("left", Role.OUTPUT), c.port("right", Role.OUTPUT)
    supply, tail = c.port("vcc", Role.SUPPLY), c.port("tail")
    first, second = c.place(q2n3904(), name="Q1"), c.place(q2n3904(), name="Q2")
    load_left = c.place(devices.resistor(10 * kohm), name="RC1")
    load_right = c.place(devices.resistor(10 * kohm), name="RC2")
    c.connect(plus, first.base)
    c.connect(minus, second.base)
    c.connect(supply, load_left.a, load_right.a)
    c.connect(load_left.b, first.collector, left)
    c.connect(load_right.b, second.collector, right)
    c.connect(first.emitter, second.emitter, tail)


@component
def bench(c: Builder) -> None:
    """Builds the pair on ±12 V with a 1 mA tail and a swept differential input."""
    ground = c.port("ground", Role.GROUND)
    positive = c.place(devices.voltage_source(12 * V), name="VCC")
    negative = c.place(devices.voltage_source(12 * V), name="VEE")
    tail = c.place(devices.current_source(1 * mA), name="I1")
    difference = c.place(devices.voltage_source(0 * V), name="VD")
    stage = c.place(pair(), name="X1")
    c.connect(positive.positive, stage.vcc, name="vcc")
    c.connect(negative.negative, tail.negative, name="vee")
    c.connect(tail.positive, stage.tail, name="tail")
    c.connect(difference.positive, stage.plus, name="plus")
    c.connect(stage.left, name="left")
    c.connect(stage.right, name="right")
    c.connect(
        positive.negative, negative.positive, difference.negative, stage.minus, ground
    )


def build() -> Component:
    """Returns the differential pair test bench component."""
    return bench()


def plan() -> Plan:
    """Returns balanced operating-point and ±200 mV differential DC sweep analyses."""
    return Plan(
        {
            "op": OperatingPoint(),
            "transfer": DcSweep("VD", linear(-200 * mV, 200 * mV, 81)),
        }
    )


def specs() -> Specs:
    """Returns specifications for zero offset at balance and full tail steering."""
    return Specs(
        [
            value("op", "left", reference="right").within(0 * V, 1 * mV),
            value("transfer", "right", reference="left", at=200 * mV).within(
                9.95 * V, 0.1 * V
            ),
            value("transfer", "right", reference="left", at=-200 * mV).within(
                -9.95 * V, 0.1 * V
            ),
        ]
    )
