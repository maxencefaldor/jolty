# Copyright 2026 Maxence Faldor
"""The 555 timer, built from its block diagram, running astable.

Inside: three 5 kΩ resistors divide the supply into thirds, two comparators watch the
threshold against two thirds and the trigger against one, an SR latch of NOR gates
remembers which crossed last, and a 2N3904 discharges the timing capacitor while the
output is low. Outside: 1 kΩ and 10 kΩ charge 10 nF from 9 V to two thirds of it, and
the 10 kΩ alone discharges it to one third. The output is high for
0.693(Ra + Rb)C = 76.2 µs and low for 0.693·Rb·C = 69.3 µs: a period of 145.5 µs,
6.87 kHz. Simulate it with ``jolty simulate examples/timer_555.py``.
"""

import math

from common import npn

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.devices import Gate
from jolty.plan import Transient
from jolty.specs import frequency, maximum
from jolty.units import Hz, V, fF, kohm, ms, nF, ns, us

SUPPLY = 9.0
PERIOD = math.log(2) * (1e3 + 2 * 10e3) * 10e-9


def comparator() -> devices.Device:
    """A comparator, its output the supply or ground as `in` is above `ref` or not."""
    half = SUPPLY / 2
    return devices.behavioral_voltage_source(
        f"{half} + {half} * tanh(v(in, ref) / 1e-3)"
    )


def gate(kind: Gate) -> devices.Device:
    """A gate of `kind`, 100 ns from input to output, on the supply."""
    inputs = 1 if kind == "buffer" else 2
    return devices.logic_gate(
        kind, inputs, rise=100 * ns, transition=20 * ns, electrical=True
    )


@component
def timer(c: Builder) -> None:
    """The divider, the comparators, the latch, the output and the discharge."""
    trigger, threshold = c.port("trigger", Role.INPUT), c.port("threshold", Role.INPUT)
    output, discharge = c.port("output", Role.OUTPUT), c.port("discharge")
    vcc, gnd = c.port("vcc", Role.SUPPLY), c.port("gnd", Role.GROUND)
    divider = [
        c.place(devices.resistor(5 * kohm), id=f"R{index}") for index in (1, 2, 3)
    ]
    upper, lower = c.net("two_thirds"), c.net("one_third")
    for resistor, (a, b) in zip(
        divider, [(vcc, upper), (upper, lower), (lower, gnd)], strict=True
    ):
        c.connect(resistor.a, a)
        c.connect(resistor.b, b)
    high = c.place(comparator(), id="B1")
    low = c.place(comparator(), id="B2")
    c.connect(high["in"], threshold)
    c.connect(high.ref, upper)
    c.connect(low["in"], lower)
    c.connect(low.ref, trigger)
    reset, set_ = c.place(gate("nor"), id="U1"), c.place(gate("nor"), id="U2")
    drive, sink = c.place(gate("buffer"), id="U3"), c.place(gate("buffer"), id="U4")
    for placed in (reset, set_, drive, sink):
        c.connect(placed.supply, vcc)
        c.connect(placed.ground, gnd)
    c.connect(high.positive, reset.a, name="reset")
    c.connect(low.positive, set_.a, name="set")
    c.connect(reset.y, set_.b, drive.a, name="q")
    c.connect(set_.y, reset.b, sink.a, name="q_bar")
    c.connect(drive.y, output)
    base = c.place(devices.resistor(10 * kohm), id="RB")
    switch = c.place(npn(), id="Q1")
    c.connect(sink.y, base.a, name="discharging")
    c.connect(base.b, switch.base, name="base")
    c.connect(switch.collector, discharge)
    c.connect(high.negative, low.negative, switch.emitter, gnd)


@component
def bench(c: Builder) -> None:
    """The timer astable on 9 V: Ra, Rb and the timing capacitor, the output loaded."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(SUPPLY * V), id="VCC")
    chip = c.place(timer(), id="X1")
    charge = c.place(devices.resistor(1 * kohm), id="RA")
    timing = c.place(devices.resistor(10 * kohm), id="RB")
    capacitor = c.place(devices.capacitor(10 * nF), id="C1")
    load = c.place(devices.capacitor(100 * fF), id="CL")
    c.connect(supply.positive, chip.vcc, charge.a, name="vcc")
    c.connect(charge.b, timing.a, chip.discharge, name="discharge")
    c.connect(timing.b, capacitor.a, chip.threshold, chip.trigger, name="timing")
    c.connect(chip.output, load.a, name="out")
    c.connect(supply.negative, chip.gnd, capacitor.b, load.b, ground)


def build() -> Component:
    """The timer on its bench."""
    return bench()


def plan() -> Plan:
    """Two milliseconds, some fourteen periods."""
    return Plan({"run": Transient(stop=2 * ms, max_step=0.2 * us)})


def specs() -> Specs:
    """The period 0.693(Ra + 2Rb)C, within 3 %, and the capacitor to two thirds.

    The discharge transistor's saturation and the latch's delays add some 2 %.
    """
    return Specs(
        [
            frequency("run", "out", start=0.5 * ms).within(
                Hz / PERIOD, 0.03 * Hz / PERIOD
            ),
            maximum("run", "timing", start=0.5 * ms).within(
                2 * SUPPLY / 3 * V, 0.1 * V
            ),
        ]
    )
