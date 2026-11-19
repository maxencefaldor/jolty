# Copyright 2026 Maxence Faldor
"""Ten million Monte Carlo samples of a two-stage bipolar amplifier, to run in shards.

The circuit and analyses match the ``monte_carlo`` speed benchmark fixture
(``benchmarks/speed/circuits/bjt_amp.py``). A differential pair of 2N3904s on a 1 mA
tail, loaded by a 2N3906 mirror, drives a 2N3904 follower with 9.1 kΩ over 1 kΩ
feedback for a gain near 10 from ±12 V. Each sample varies resistor values within 5 %
and runs the operating point, 71 AC frequencies, and 200 µs of a 10 kHz sine. The
specifications verify the 1 kHz gain and output DC offset. The companion ``launch.py``
script runs shards across a process pool and merges the results.
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
    """Returns a resistor of `resistance` kilohms with the module tolerance."""
    return devices.resistor(resistance * kohm, tolerance=TOLERANCE)


@component
def amplifier(c: Builder) -> None:
    """Builds the differential pair, active mirror, and follower with gain near 10."""
    plus, out = c.port("plus", Role.INPUT), c.port("out", Role.OUTPUT)
    vcc, vee = c.port("vcc", Role.SUPPLY), c.port("vee", Role.SUPPLY)
    ground = c.port("ground", Role.GROUND)
    left, right = c.place(q2n3904(), name="Q1"), c.place(q2n3904(), name="Q2")
    diode, mirror = c.place(q2n3906(), name="Q3"), c.place(q2n3906(), name="Q4")
    follower = c.place(q2n3904(), name="Q5")
    tail = c.place(resistor(11), name="RT")
    load = c.place(resistor(4.7), name="RE")
    feedback, shunt = c.place(resistor(9.1), name="RF"), c.place(resistor(1), name="RG")
    compensation = c.place(devices.capacitor(100 * pF), name="CC")
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
    """Builds the amplifier on ±12 V, driven through 1 kΩ by a 100 mV, 10 kHz sine."""
    ground = c.port("ground", Role.GROUND)
    positive = c.place(devices.voltage_source(12 * V), name="VCC")
    negative = c.place(devices.voltage_source(12 * V), name="VEE")
    signal = Sine(offset=0 * V, amplitude=100 * mV, frequency=10 * kHz)
    source = c.place(devices.voltage_source(signal), name="V1")
    series = c.place(resistor(1), name="RS")
    stage = c.place(amplifier(), name="X1")
    c.connect(source.positive, series.a, name="in")
    c.connect(series.b, stage.plus, name="plus")
    c.connect(positive.positive, stage.vcc, name="vcc")
    c.connect(negative.negative, stage.vee, name="vee")
    c.connect(stage.out, name="out")
    c.connect(
        positive.negative, negative.positive, source.negative, stage.ground, ground
    )


def build() -> Component:
    """Returns the amplifier test bench component."""
    return bench()


def plan() -> Plan:
    """Returns the Monte Carlo plan covering DC, 71 AC points, and 200 µs transient."""
    return Plan(
        {
            "op": OperatingPoint(),
            "ac": Ac("V1", decade(10 * Hz, 100 * MHz, 10)),
            "tran": Transient(stop=200 * us),
        },
        monte_carlo=MonteCarlo(SAMPLES),
    )


def specs() -> Specs:
    """Returns specifications for the 1 kHz gain and output DC offset."""
    return Specs(
        [
            gain("ac", "out", at=1 * kHz).between(19.25, 20.75),
            value("op", "out").between(100 * mV, 200 * mV),
        ],
        yield_target=0.95,
    )
