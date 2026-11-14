# Copyright 2026 Maxence Faldor
"""Ten million Monte Carlo samples of a two-stage bipolar amplifier, to run in shards.

This is the circuit and the analyses of the benchmarks' ``monte_carlo`` fixture,
``perf/circuits/bjt_amp.py``: a differential pair of 2N3904s on a 1 mA tail, loaded by
a mirror of 2N3906s, drives a 2N3904 follower, and 9.1 kΩ over 1 kΩ feed the output
back for a gain of about 10 from ±12 V. Each sample draws its resistors within 5 %,
and runs the operating point, the response at 71 frequencies and 200 µs of a 10 kHz
sine. Its specs hold the gain and the output's offset. ``launch.py`` beside it runs
the samples in shards over a pool of processes and merges them.
"""

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.parts import q2n3904, q2n3906
from jolty.plan import Ac, MonteCarlo, OperatingPoint, Transient, decade
from jolty.specs import gain, value
from jolty.units import Hz, MHz, V, kHz, kohm, mV, pF, us
from jolty.waveforms import Sine

SAMPLES = 10_000_000
"""The samples of the Monte Carlo analysis."""

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
    """The operating point, 71 frequencies and two periods of the sine, on samples."""
    return Plan(
        {
            "op": OperatingPoint(),
            "ac": Ac("V1", decade(10 * Hz, 100 * MHz, 10)),
            "tran": Transient(stop=200 * us),
        },
        monte_carlo=MonteCarlo(SAMPLES),
    )


def specs() -> Specs:
    """The gain at 1 kHz, and the output's offset."""
    return Specs(
        [
            gain("ac", "out", at=1 * kHz).between(19.25, 20.75),
            value("op", "out").between(100 * mV, 200 * mV),
        ],
        yield_target=0.95,
    )
