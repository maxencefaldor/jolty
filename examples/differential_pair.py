# Copyright 2026 Maxence Faldor
"""A long-tailed pair: two 2N3904s sharing a 1 mA tail, steering it between them.

The difference of the bases splits the tail as a hyperbolic tangent: the collector
currents' difference is I·tanh(Vd/2Vt), so the outputs' difference swings ±I·Rc =
±10 V and its small-signal gain is gm·Rc = (I/2)/Vt·Rc, some 193. Simulate it with
``jolty simulate examples/differential_pair.py``.
"""

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.parts import q2n3904
from jolty.plan import DcSweep, OperatingPoint, linear
from jolty.specs import value
from jolty.units import V, kohm, mA, mV


@component
def pair(c: Builder) -> None:
    """Two transistors with collector loads, their emitters on one tail."""
    plus, minus = c.port("plus", Role.INPUT), c.port("minus", Role.INPUT)
    left, right = c.port("left", Role.OUTPUT), c.port("right", Role.OUTPUT)
    supply, tail = c.port("vcc", Role.SUPPLY), c.port("tail")
    first, second = c.place(q2n3904(), id="Q1"), c.place(q2n3904(), id="Q2")
    load_left = c.place(devices.resistor(10 * kohm), id="RC1")
    load_right = c.place(devices.resistor(10 * kohm), id="RC2")
    c.connect(plus, first.base)
    c.connect(minus, second.base)
    c.connect(supply, load_left.a, load_right.a)
    c.connect(load_left.b, first.collector, left)
    c.connect(load_right.b, second.collector, right)
    c.connect(first.emitter, second.emitter, tail)


@component
def bench(c: Builder) -> None:
    """The pair on ±12 V, its tail a 1 mA source, one base swept against the other."""
    ground = c.port("ground", Role.GROUND)
    positive = c.place(devices.voltage_source(12 * V), id="VCC")
    negative = c.place(devices.voltage_source(12 * V), id="VEE")
    tail = c.place(devices.current_source(1 * mA), id="I1")
    difference = c.place(devices.voltage_source(0 * V), id="VD")
    stage = c.place(pair(), id="X1")
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
    """The pair on its bench."""
    return bench()


def plan() -> Plan:
    """The balanced operating point, and the difference swept across ±200 mV."""
    return Plan(
        {
            "op": OperatingPoint(),
            "transfer": DcSweep("VD", linear(-200 * mV, 200 * mV, 81)),
        }
    )


def specs() -> Specs:
    """Balanced at zero, the tail all on one side 200 mV away."""
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
