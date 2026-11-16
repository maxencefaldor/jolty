# Copyright 2026 Maxence Faldor
"""A common-emitter amplifier: a 2N3904 biased by a divider, its emitter degenerated.

The divider sets the base near 1.8 V, so about 1.1 mA flows through the emitter's
100 Ω and 1 kΩ; the 1 kΩ is bypassed above a few hertz. The midband gain is the
collector's 4.7 kΩ over the emitter's 100 Ω and the transistor's own 1/gm, some 32 dB,
inverted. Coupling capacitors pass the band from tens of hertz up. Simulate it with
``jolty simulate examples/common_emitter.py``.
"""

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.parts import q2n3904
from jolty.plan import Ac, OperatingPoint, Transient, decade
from jolty.specs import gain, value
from jolty.units import Hz, MHz, V, kHz, kohm, ms, mV, ohm, uF, us
from jolty.waveforms import Sine


@component
def amplifier(c: Builder) -> None:
    """The transistor, its bias divider, its loads, and its coupling capacitors."""
    source, output = c.port("input"), c.port("output")
    supply, ground = c.port("vcc", Role.SUPPLY), c.port("ground", Role.GROUND)
    transistor = c.place(q2n3904(), name="Q1")
    upper = c.place(devices.resistor(56 * kohm), name="R1")
    lower = c.place(devices.resistor(10 * kohm), name="R2")
    collector = c.place(devices.resistor(4.7 * kohm), name="RC")
    degeneration = c.place(devices.resistor(100 * ohm), name="RE1")
    emitter = c.place(devices.resistor(1 * kohm), name="RE2")
    bypass = c.place(devices.capacitor(100 * uF), name="CE")
    coupling_in = c.place(devices.capacitor(10 * uF), name="C1")
    coupling_out = c.place(devices.capacitor(10 * uF), name="C2")
    load = c.place(devices.resistor(100 * kohm), name="RL")
    c.connect(source, coupling_in.a)
    c.connect(coupling_in.b, upper.b, lower.a, transistor.base, name="base")
    c.connect(supply, upper.a, collector.a)
    c.connect(collector.b, transistor.collector, coupling_out.a, name="collector")
    c.connect(coupling_out.b, load.a, output)
    c.connect(transistor.emitter, degeneration.a, name="emitter")
    c.connect(degeneration.b, emitter.a, bypass.a, name="bypassed")
    c.connect(lower.b, emitter.b, bypass.b, load.b, ground)


@component
def bench(c: Builder) -> None:
    """The amplifier on 12 V, fed 10 mV at 1 kHz."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(12 * V), name="VCC")
    tone = Sine(offset=0 * V, amplitude=10 * mV, frequency=1 * kHz)
    source = c.place(devices.voltage_source(tone), name="V1")
    stage = c.place(amplifier(), name="X1")
    c.connect(supply.positive, stage.vcc, name="vcc")
    c.connect(source.positive, stage.input, name="in")
    c.connect(stage.output, name="out")
    c.connect(supply.negative, source.negative, stage.ground, ground)


def build() -> Component:
    """The amplifier on its bench."""
    return bench()


def plan() -> Plan:
    """The bias point, the frequency response, and the amplified tone."""
    return Plan(
        {
            "op": OperatingPoint(),
            "bode": Ac("V1", decade(1 * Hz, 100 * MHz, 30)),
            "tone": Transient(stop=5 * ms, max_step=5 * us),
        }
    )


def specs() -> Specs:
    """The collector biased mid-supply, and some 31.5 dB in the midband."""
    return Specs(
        [
            value("op", "X1/collector").between(5 * V, 8 * V),
            gain("bode", "out", at=10 * kHz).between(30.5, 32.5),
        ]
    )
