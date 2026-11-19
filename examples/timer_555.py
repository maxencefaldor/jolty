# Copyright 2026 Maxence Faldor
"""The 555 timer, built from its block diagram, running astable.

Inside the timer, three 5 kΩ resistors divide the supply into thirds, and two
comparators monitor threshold against two thirds and trigger against one third. A NOR
SR latch tracks the last crossing, and a 2N3904 discharges the timing capacitor while
the output is low. Outside, 1 kΩ and 10 kΩ charge 10 nF from 9 V to two thirds, and
the 10 kΩ resistor discharges the capacitor to one third. The output stays high for
0.693(Ra + Rb)C = 76.2 µs and low for 0.693·Rb·C = 69.3 µs, yielding a 145.5 µs period
(6.87 kHz). Simulate with ``jolty simulate examples/timer_555.py``.
"""

import math

from common import comparator

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.devices import Gate
from jolty.parts import q2n3904
from jolty.plan import Transient
from jolty.specs import frequency, maximum
from jolty.units import Hz, V, fF, kohm, ms, nF, ns, us

SUPPLY = 9.0
PERIOD = math.log(2) * (1e3 + 2 * 10e3) * 10e-9


def gate(kind: Gate) -> devices.Device:
    """Returns a logic gate of `kind` with a 100 ns propagation delay."""
    inputs = 1 if kind == "buffer" else 2
    return devices.logic_gate(
        kind, inputs, rise=100 * ns, transition=20 * ns, electrical=True
    )


@component
def timer(c: Builder) -> None:
    """Builds the divider, comparators, SR latch, output buffers, and discharge."""
    trigger, threshold = c.port("trigger", Role.INPUT), c.port("threshold", Role.INPUT)
    output, discharge = c.port("output", Role.OUTPUT), c.port("discharge")
    vcc, gnd = c.port("vcc", Role.SUPPLY), c.port("gnd", Role.GROUND)
    divider = [
        c.place(devices.resistor(5 * kohm), name=f"R{index}") for index in (1, 2, 3)
    ]
    upper, lower = c.net("two_thirds"), c.net("one_third")
    for resistor, (a, b) in zip(
        divider, [(vcc, upper), (upper, lower), (lower, gnd)], strict=True
    ):
        c.connect(resistor.a, a)
        c.connect(resistor.b, b)
    high = c.place(comparator(SUPPLY), name="B1")
    low = c.place(comparator(SUPPLY), name="B2")
    c.connect(high["in"], threshold)
    c.connect(high.ref, upper)
    c.connect(low["in"], lower)
    c.connect(low.ref, trigger)
    reset, set_ = c.place(gate("nor"), name="U1"), c.place(gate("nor"), name="U2")
    drive, sink = c.place(gate("buffer"), name="U3"), c.place(gate("buffer"), name="U4")
    for placed in (reset, set_, drive, sink):
        c.connect(placed.supply, vcc)
        c.connect(placed.ground, gnd)
    c.connect(high.positive, reset.a, name="reset")
    c.connect(low.positive, set_.a, name="set")
    c.connect(reset.y, set_.b, drive.a, name="q")
    c.connect(set_.y, reset.b, sink.a, name="q_bar")
    c.connect(drive.y, output)
    base = c.place(devices.resistor(10 * kohm), name="RB")
    switch = c.place(q2n3904(), name="Q1")
    c.connect(sink.y, base.a, name="discharging")
    c.connect(base.b, switch.base, name="base")
    c.connect(switch.collector, discharge)
    c.connect(high.negative, low.negative, switch.emitter, gnd)


@component
def bench(c: Builder) -> None:
    """Builds the astable 555 timer on 9 V with Ra, Rb, timing capacitor, and load."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(SUPPLY * V), name="VCC")
    chip = c.place(timer(), name="X1")
    charge = c.place(devices.resistor(1 * kohm), name="RA")
    timing = c.place(devices.resistor(10 * kohm), name="RB")
    capacitor = c.place(devices.capacitor(10 * nF), name="C1")
    load = c.place(devices.capacitor(100 * fF), name="CL")
    c.connect(supply.positive, chip.vcc, charge.a, name="vcc")
    c.connect(charge.b, timing.a, chip.discharge, name="discharge")
    c.connect(timing.b, capacitor.a, chip.threshold, chip.trigger, name="timing")
    c.connect(chip.output, load.a, name="out")
    c.connect(supply.negative, chip.gnd, capacitor.b, load.b, ground)


def build() -> Component:
    """Returns the 555 timer test bench component."""
    return bench()


def plan() -> Plan:
    """Returns a 2 ms transient plan spanning roughly fourteen oscillation periods."""
    return Plan({"run": Transient(stop=2 * ms, max_step=0.2 * us)})


def specs() -> Specs:
    """Returns specifications for period near 0.693(Ra + 2Rb)C and peak timing level.

    Discharge transistor saturation and latch propagation delays add roughly 2 %.
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
