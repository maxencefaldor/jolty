# Copyright 2026 Maxence Faldor
"""A six-transistor SRAM cell: two cross-coupled inverters and two access transistors.

The inverters hold a bit, each the other's complement. Raising the word line opens the
access transistors onto the bit lines: driven apart, the bit lines overpower the cell's
weaker pull-ups and write it, and once the word line falls the cell keeps what was
written. The bench starts writing a 0, so the operating point holds one state of the
two the cell could, then writes a 1, holds it, writes a 0 and holds it. Simulate it with
``jolty simulate examples/sram_cell.py``.
"""

from common import SUPPLY, inverter, mos

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.plan import Transient
from jolty.specs import value
from jolty.units import V, mV, ns, ps
from jolty.waveforms import Piecewise


@component
def cell(c: Builder) -> None:
    """Builds two cross-coupled inverters and two bit-line access transistors."""
    word = c.port("word", Role.INPUT)
    bit, bit_bar = c.port("bit"), c.port("bit_bar")
    vdd, gnd = c.port("vdd", Role.SUPPLY), c.port("gnd", Role.GROUND)
    left, right = c.place(inverter(), name="X1"), c.place(inverter(), name="X2")
    access = c.place(mos("n", width=2), name="M5")
    access_bar = c.place(mos("n", width=2), name="M6")
    c.connect(left.vdd, right.vdd, vdd)
    c.connect(left.gnd, right.gnd, access.bulk, access_bar.bulk, gnd)
    c.connect(left.y, right.a, access.source, name="q")
    c.connect(right.y, left.a, access_bar.source, name="q_bar")
    c.connect(access.drain, bit)
    c.connect(access_bar.drain, bit_bar)
    c.connect(access.gate, access_bar.gate, word)


def steps(*levels: tuple[float, float]) -> Piecewise:
    """Returns a piecewise waveform from `(time_ns, volts)` steps with 100 ps edges."""
    points = []
    for time, level in levels:
        if points:
            points.append((time * ns, points[-1][1]))
        points.append(((time + 0.1) * ns, level * V))
    return Piecewise(points)


@component
def bench(c: Builder) -> None:
    """Builds the SRAM cell initialized to 0, written 1 at 2 ns, and 0 at 12 ns."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(SUPPLY * V), name="VDD")
    word = c.place(
        devices.voltage_source(
            steps((0, SUPPLY), (0.5, 0), (2, SUPPLY), (5, 0), (12, SUPPLY), (15, 0))
        ),
        name="VWL",
    )
    bit = c.place(
        devices.voltage_source(steps((0, 0), (1, SUPPLY), (10, 0))), name="VBL"
    )
    bit_bar = c.place(
        devices.voltage_source(steps((0, SUPPLY), (1, 0), (10, SUPPLY))), name="VBLB"
    )
    memory = c.place(cell(), name="X1")
    c.connect(supply.positive, memory.vdd, name="vdd")
    c.connect(word.positive, memory.word, name="wl")
    c.connect(bit.positive, memory.bit, name="bl")
    c.connect(bit_bar.positive, memory.bit_bar, name="blb")
    c.connect(
        supply.negative,
        word.negative,
        bit.negative,
        bit_bar.negative,
        memory.gnd,
        ground,
    )


def build() -> Component:
    """Returns the SRAM cell test bench component."""
    return bench()


def plan() -> Plan:
    """Returns a 20 ns transient plan covering write-1, hold, write-0, and hold."""
    return Plan({"write": Transient(stop=20 * ns, max_step=10 * ps)})


def specs() -> Specs:
    """Returns specifications verifying retention of 1 and 0 after each write."""
    return Specs(
        [
            value("write", "X1/q", at=9 * ns).within(SUPPLY * V, 10 * mV),
            value("write", "X1/q_bar", at=9 * ns).within(0 * V, 10 * mV),
            value("write", "X1/q", at=19 * ns).within(0 * V, 10 * mV),
            value("write", "X1/q_bar", at=19 * ns).within(SUPPLY * V, 10 * mV),
        ]
    )
