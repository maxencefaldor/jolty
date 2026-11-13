# Copyright 2026 Maxence Faldor
"""A Widlar current source: a small output current from a large reference.

The reference transistor carries about 1 mA, set by 11.3 kΩ from 12 V; an emitter
resistor of 1.55 kΩ under the output transistor takes the difference of their
base-emitter voltages, Vt·ln(Iref/Iout) = Iout·Re, so the output settles near 50 µA
and stays there as its collector's voltage moves. Simulate it with
``jolty simulate examples/widlar_source.py``.
"""

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.parts import q2n3904
from jolty.plan import DcSweep, OperatingPoint, linear
from jolty.specs import value
from jolty.units import V, kohm, uA


@component
def widlar(c: Builder) -> None:
    """The diode-connected reference, and the output transistor over its resistor."""
    output = c.port("output", Role.OUTPUT)
    supply, ground = c.port("vcc", Role.SUPPLY), c.port("ground", Role.GROUND)
    reference = c.place(q2n3904(), id="Q1")
    mirror = c.place(q2n3904(), id="Q2")
    feed = c.place(devices.resistor(11.3 * kohm), id="R1")
    emitter = c.place(devices.resistor(1.55 * kohm), id="RE")
    c.connect(supply, feed.a)
    c.connect(feed.b, reference.collector, reference.base, mirror.base, name="bias")
    c.connect(mirror.emitter, emitter.a, name="emitter")
    c.connect(mirror.collector, output)
    c.connect(reference.emitter, emitter.b, ground)


@component
def bench(c: Builder) -> None:
    """The source on 12 V, its output held by a swept voltage."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(12 * V), id="VCC")
    compliance = c.place(devices.voltage_source(5 * V), id="VOUT")
    source = c.place(widlar(), id="X1")
    c.connect(supply.positive, source.vcc, name="vcc")
    c.connect(compliance.positive, source.output, name="out")
    c.connect(supply.negative, compliance.negative, source.ground, ground)


def build() -> Component:
    """The source on its bench."""
    return bench()


def plan() -> Plan:
    """The operating point, and the output's voltage swept from 0 to 12 V."""
    return Plan(
        {
            "op": OperatingPoint(),
            "compliance": DcSweep("VOUT", linear(0 * V, 12 * V, 61)),
        }
    )


def specs() -> Specs:
    """About 50 µA into the output, nearly flat from 1 V up."""
    return Specs(
        [
            value("op", "VOUT:current").within(-50 * uA, 5 * uA),
            value("compliance", "VOUT:current", at=1 * V).within(-50 * uA, 5 * uA),
            value("compliance", "VOUT:current", at=12 * V).within(-50 * uA, 5 * uA),
        ]
    )
