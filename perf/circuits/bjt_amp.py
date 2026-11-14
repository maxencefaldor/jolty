# Copyright 2026 Maxence Faldor
"""A two-stage bipolar amplifier, the cost of one small run and of many.

A differential pair of 2N3904s on a 1 mA tail, loaded by a mirror of 2N3906s, drives a
2N3904 follower, and 9.1 kΩ over 1 kΩ feed the output back to the pair, for a gain of
about 10 from ±12 V. Its resistors are within 5 %, which only a Monte Carlo analysis
reads. Three fixtures run it:

- ``bjt_amp``: the operating point, the response at 71 frequencies from 10 Hz to
  100 MHz, and 200 µs of a 10 kHz sine, the cost of one small run;
- ``ac_wide``: the response at 10,000 frequencies from 1 Hz to 1 GHz, for running
  frequencies in parallel;
- ``monte_carlo``: ``bjt_amp``'s analyses on 100,000 samples of the resistors, the
  throughput of many runs.
"""

from fixture import Fixture, write

from jolty import Builder, Component, Plan, Role, component, devices
from jolty.parts import q2n3904, q2n3906
from jolty.plan import Ac, MonteCarlo, OperatingPoint, Transient, decade
from jolty.units import GHz, Hz, MHz, V, kHz, kohm, mV, pF, us
from jolty.waveforms import Sine

TOLERANCE = 0.05
"""Every resistor's tolerance."""


def resistor(resistance: float) -> devices.Device:
    """A resistor of `resistance` kilohms, within the tolerance."""
    return devices.resistor(resistance * kohm, tolerance=TOLERANCE)


@component
def amplifier(c: Builder) -> None:
    """The pair, its mirror and the follower, fed back for a gain of about 10."""
    plus, out = c.port("plus", Role.INPUT), c.port("out", Role.OUTPUT)
    vcc, vee = c.port("vcc", Role.SUPPLY), c.port("vee", Role.SUPPLY)
    ground = c.port("ground", Role.GROUND)
    left, right = c.place(q2n3904(), id="Q1"), c.place(q2n3904(), id="Q2")
    diode, mirror = c.place(q2n3906(), id="Q3"), c.place(q2n3906(), id="Q4")
    follower = c.place(q2n3904(), id="Q5")
    tail = c.place(resistor(11), id="RT")
    load = c.place(resistor(4.7), id="RE")
    feedback, shunt = c.place(resistor(9.1), id="RF"), c.place(resistor(1), id="RG")
    compensation = c.place(devices.capacitor(100 * pF), id="CC")
    c.connect(plus, left.base)
    c.connect(left.collector, diode.collector, diode.base, mirror.base, name="mirror")
    c.connect(
        right.collector, mirror.collector, follower.base, compensation.a, name="gain"
    )
    c.connect(left.emitter, right.emitter, tail.a, name="tail")
    c.connect(right.base, feedback.b, shunt.a, name="minus")
    c.connect(vcc, diode.emitter, mirror.emitter, follower.collector)
    c.connect(follower.emitter, load.a, feedback.a, out)
    c.connect(vee, tail.b, load.b)
    c.connect(shunt.b, compensation.b, ground)


@component
def bench(c: Builder) -> None:
    """The amplifier on ±12 V, driven through 1 kΩ by a 100 mV sine at 10 kHz."""
    ground = c.port("ground", Role.GROUND)
    positive = c.place(devices.voltage_source(12 * V), id="VCC")
    negative = c.place(devices.voltage_source(12 * V), id="VEE")
    signal = Sine(offset=0 * V, amplitude=100 * mV, frequency=10 * kHz)
    source = c.place(devices.voltage_source(signal), id="V1")
    series = c.place(resistor(1), id="RS")
    stage = c.place(amplifier(), id="X1")
    c.connect(source.positive, series.a, name="in")
    c.connect(series.b, stage.plus, name="plus")
    c.connect(positive.positive, stage.vcc, name="vcc")
    c.connect(negative.negative, stage.vee, name="vee")
    c.connect(stage.out, name="out")
    c.connect(
        positive.negative, negative.positive, source.negative, stage.ground, ground
    )


def build() -> Component:
    """The amplifier on its bench."""
    return bench()


def plan() -> Plan:
    """The operating point, 71 frequencies and two periods of the sine."""
    return Plan(
        {
            "op": OperatingPoint(),
            "ac": Ac("V1", decade(10 * Hz, 100 * MHz, 10)),
            "tran": Transient(stop=200 * us),
        }
    )


def circuits() -> dict[str, Component]:
    """The amplifier's one circuit, by the name of its file."""
    return {"bjt_amp": build()}


def fixtures() -> dict[str, Fixture]:
    """The three fixtures of the amplifier."""
    one = plan()
    return {
        "bjt_amp": Fixture("bjt_amp", one),
        "ac_wide": Fixture(
            "bjt_amp", Plan({"ac": Ac("V1", decade(1 * Hz, 1 * GHz, 1111))})
        ),
        "monte_carlo": Fixture(
            "bjt_amp", Plan(one.analyses, monte_carlo=MonteCarlo(100_000))
        ),
    }


if __name__ == "__main__":
    write(circuits(), fixtures())
