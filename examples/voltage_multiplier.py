# Copyright 2026 Maxence Faldor
"""A three-stage Cockcroft-Walton multiplier: six diodes and six capacitors.

Each stage adds twice the input's peak: from a 10 V sine, an unloaded ladder of three
stages charges toward 60 V, less two diode drops a stage. The capacitors charge a
little each cycle, so the output climbs over tens of cycles. Simulate it with
``jolty simulate examples/voltage_multiplier.py``.
"""

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.parts import d1n4148
from jolty.plan import Transient
from jolty.specs import value
from jolty.units import Mohm, V, kHz, ms, uF, us
from jolty.waveforms import Sine

STAGES = 3


@component
def multiplier(c: Builder) -> None:
    """Builds two capacitor columns coupled by an alternating diode ladder."""
    source, output = c.port("input"), c.port("output")
    ground = c.port("ground", Role.GROUND)
    pushed = source
    held = ground
    for stage in range(1, STAGES + 1):
        push = c.place(devices.capacitor(1 * uF), name=f"C{2 * stage - 1}")
        hold = c.place(devices.capacitor(1 * uF), name=f"C{2 * stage}")
        up = c.place(d1n4148(), name=f"D{2 * stage - 1}")
        over = c.place(d1n4148(), name=f"D{2 * stage}")
        swing = c.net(f"swing{stage}")
        level = output if stage == STAGES else c.net(f"level{stage}")
        c.connect(push.a, pushed)
        c.connect(push.b, up.cathode, over.anode, swing)
        c.connect(up.anode, held)
        c.connect(over.cathode, hold.a, level)
        c.connect(hold.b, held)
        pushed, held = swing, level


@component
def bench(c: Builder) -> None:
    """Builds the multiplier driven by a 10 V, 1 kHz sine into a 10 MΩ load."""
    ground = c.port("ground", Role.GROUND)
    tone = Sine(offset=0 * V, amplitude=10 * V, frequency=1 * kHz)
    source = c.place(devices.voltage_source(tone), name="V1")
    ladder = c.place(multiplier(), name="X1")
    load = c.place(devices.resistor(10 * Mohm), name="RL")
    c.connect(source.positive, ladder.input, name="in")
    c.connect(ladder.output, load.a, name="out")
    c.connect(source.negative, ladder.ground, load.b, ground)


def build() -> Component:
    """Returns the Cockcroft-Walton multiplier test bench component."""
    return bench()


def plan() -> Plan:
    """Returns a 200 ms transient charging plan spanning 200 input cycles."""
    return Plan({"charge": Transient(stop=200 * ms, max_step=20 * us)})


def specs() -> Specs:
    """Returns output voltage specifications near six times peak minus diode drops."""
    return Specs([value("charge", "out", at=200 * ms).between(50 * V, 60 * V)])
