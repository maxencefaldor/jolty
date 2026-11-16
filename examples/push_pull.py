# Copyright 2026 Maxence Faldor
"""A class AB push-pull output stage: complementary followers, biased past crossover.

A 2N3904 sources the load's current and a 2N3906 sinks it, each an emitter follower.
Two 1N4148 diodes between their bases hold both just on, so neither lets go as the
signal crosses zero, and the output follows the input without the dead band of a
class B stage. Simulate it with ``jolty simulate examples/push_pull.py``.
"""

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.parts import d1n4148, q2n3904, q2n3906
from jolty.plan import DcSweep, Transient, linear
from jolty.specs import value
from jolty.units import V, kHz, kohm, ms, mV, ohm, us
from jolty.waveforms import Sine


@component
def stage(c: Builder) -> None:
    """The two followers, their bias diodes and resistors, and emitter resistors."""
    source, output = c.port("input"), c.port("output")
    positive, negative = c.port("vcc", Role.SUPPLY), c.port("vee", Role.SUPPLY)
    upper, lower = c.place(q2n3904(), name="Q1"), c.place(q2n3906(), name="Q2")
    feed_up = c.place(devices.resistor(4.7 * kohm), name="RB1")
    feed_down = c.place(devices.resistor(4.7 * kohm), name="RB2")
    top, bottom = c.place(d1n4148(), name="D1"), c.place(d1n4148(), name="D2")
    emitter_up = c.place(devices.resistor(2.2 * ohm), name="RE1")
    emitter_down = c.place(devices.resistor(2.2 * ohm), name="RE2")
    c.connect(positive, feed_up.a, upper.collector)
    c.connect(negative, feed_down.b, lower.collector)
    c.connect(feed_up.b, top.anode, upper.base, name="high")
    c.connect(top.cathode, bottom.anode, source)
    c.connect(bottom.cathode, feed_down.a, lower.base, name="low")
    c.connect(upper.emitter, emitter_up.a, name="pushed")
    c.connect(lower.emitter, emitter_down.a, name="pulled")
    c.connect(emitter_up.b, emitter_down.b, output)


@component
def bench(c: Builder) -> None:
    """The stage on ±12 V, driving 100 Ω with a 4 V sine at 1 kHz."""
    ground = c.port("ground", Role.GROUND)
    positive = c.place(devices.voltage_source(12 * V), name="VCC")
    negative = c.place(devices.voltage_source(12 * V), name="VEE")
    tone = Sine(offset=0 * V, amplitude=4 * V, frequency=1 * kHz)
    source = c.place(devices.voltage_source(tone), name="V1")
    output = c.place(stage(), name="X1")
    load = c.place(devices.resistor(100 * ohm), name="RL")
    c.connect(positive.positive, output.vcc, name="vcc")
    c.connect(negative.negative, output.vee, name="vee")
    c.connect(source.positive, output.input, name="in")
    c.connect(output.output, load.a, name="out")
    c.connect(positive.negative, negative.positive, source.negative, load.b, ground)


def build() -> Component:
    """The stage on its bench."""
    return bench()


def plan() -> Plan:
    """The transfer from -6 V to 6 V, and two periods of the sine."""
    return Plan(
        {
            "transfer": DcSweep("V1", linear(-6 * V, 6 * V, 121)),
            "tone": Transient(stop=2 * ms, max_step=2 * us),
        }
    )


def specs() -> Specs:
    """The output within 100 mV of the input around zero, and some 0.95 of it at 4 V."""
    return Specs(
        [
            value("transfer", "out", at=0 * V).within(0 * V, 100 * mV),
            value("transfer", "out", at=0.1 * V).within(0.1 * V, 100 * mV),
            value("transfer", "out", at=4 * V).between(3.6 * V, 4 * V),
            value("transfer", "out", at=-4 * V).between(-4 * V, -3.6 * V),
        ]
    )
