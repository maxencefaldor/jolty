# Copyright 2026 Maxence Faldor
"""A two-bit ripple counter of edge-triggered D flip-flops, each six NAND gates.

Each flip-flop is the 7474's: two latches of three-input NAND gates that pass the data
only on the clock's rising edge, with a preset and a clear. Fed its own complement, a
flip-flop toggles on every rising edge and halves its clock; the second, clocked by the
first's complement, halves it again, so the two count 0, 1, 2, 3 in binary. The clear
holds both at zero for the first nanoseconds. Simulate it with
``jolty simulate examples/ripple_counter.py``.
"""

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.devices import LogicLevel
from jolty.plan import Transient
from jolty.specs import frequency, value
from jolty.units import MHz, Quantity, V, fF, mV, ns, ps

SUPPLY = 3.3
PERIOD = 10e-9
"""The clock's period, in seconds."""


def nand() -> devices.Device:
    """A three-input NAND gate, 200 ps from input to output, on a supply."""
    return devices.logic_gate(
        "nand", 3, rise=200 * ps, transition=50 * ps, electrical=True
    )


@component
def flip_flop(c: Builder) -> None:
    """The 7474's positive-edge-triggered D flip-flop, with preset and clear."""
    data, clock = c.port("d", Role.INPUT), c.port("clock", Role.INPUT)
    preset, clear = c.port("preset", Role.INPUT), c.port("clear", Role.INPUT)
    q, q_bar = c.port("q", Role.OUTPUT), c.port("q_bar", Role.OUTPUT)
    vdd, gnd = c.port("vdd", Role.SUPPLY), c.port("gnd", Role.GROUND)
    gates = [c.place(nand(), name=f"U{index}") for index in range(1, 7)]
    for gate in gates:
        c.connect(gate.supply, vdd)
        c.connect(gate.ground, gnd)
    nets = {name: c.net(name) for name in ("n1", "n2", "n3", "n4")}
    wiring = [
        (preset, nets["n4"], nets["n2"], nets["n1"]),
        (nets["n1"], clear, clock, nets["n2"]),
        (nets["n2"], clock, nets["n4"], nets["n3"]),
        (nets["n3"], clear, data, nets["n4"]),
        (preset, nets["n2"], q_bar, q),
        (q, nets["n3"], clear, q_bar),
    ]
    for gate, (a, b, c_, y) in zip(gates, wiring, strict=True):
        c.connect(gate.a, a)
        c.connect(gate.b, b)
        c.connect(gate.c, c_)
        c.connect(gate.y, y)


@component
def counter(c: Builder) -> None:
    """Two flip-flops, each toggling, the second clocked by the first's complement."""
    clock, clear = c.port("clock", Role.INPUT), c.port("clear", Role.INPUT)
    low, high = c.port("q0", Role.OUTPUT), c.port("q1", Role.OUTPUT)
    vdd, gnd = c.port("vdd", Role.SUPPLY), c.port("gnd", Role.GROUND)
    first, second = c.place(flip_flop(), name="FF0"), c.place(flip_flop(), name="FF1")
    for stage in (first, second):
        c.connect(stage.vdd, stage.preset, vdd)
        c.connect(stage.gnd, gnd)
        c.connect(stage.clear, clear)
    c.connect(first.clock, clock)
    c.connect(first.q, low)
    c.connect(first.q_bar, first.d, second.clock, name="q0_bar")
    c.connect(second.q, high)
    c.connect(second.q_bar, second.d, name="q1_bar")


@component
def bench(c: Builder) -> None:
    """The counter on 3.3 V, its clock at 100 MHz, cleared for the first 3 ns."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(SUPPLY * V), name="VDD")
    ticks: list[tuple[Quantity, LogicLevel]] = [
        ((index + 1) / 2 * PERIOD * 1e9 * ns, 1 if index % 2 == 0 else 0)
        for index in range(15)
    ]
    clock = c.place(devices.logic_source(0, ticks, transition=100 * ps), name="CLK")
    clear = c.place(
        devices.logic_source(0, [(3 * ns, 1)], transition=100 * ps), name="CLR"
    )
    count = c.place(counter(), name="X1")
    c.connect(supply.positive, count.vdd, name="vdd")
    c.connect(clock.output, count.clock, name="clock")
    c.connect(clear.output, count.clear, name="clear")
    for bit in ("q0", "q1"):
        load = c.place(devices.capacitor(10 * fF), name=f"C{bit}")
        c.connect(count[bit], load.a, name=bit)
        c.connect(load.b, ground)
    c.connect(supply.negative, count.gnd, ground)


def build() -> Component:
    """The counter on its bench."""
    return bench()


def plan() -> Plan:
    """Eighty nanoseconds, eight clock edges."""
    return Plan({"count": Transient(stop=80 * ns, max_step=0.1 * ns)})


def specs() -> Specs:
    """The count after each rising edge, and each bit at its fraction of the clock."""
    requirements = []
    for edge in range(1, 8):
        at = (edge + 0.5) * PERIOD * 1e9 * ns
        count = edge % 4
        for bit in range(2):
            level = SUPPLY if (count >> bit) & 1 else 0.0
            requirements.append(
                value("count", f"q{bit}", at=at).within(level * V, 10 * mV)
            )
    requirements.append(frequency("count", "q0").within(50 * MHz, 0.5 * MHz))
    return Specs(requirements)
