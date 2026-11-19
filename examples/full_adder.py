# Copyright 2026 Maxence Faldor
"""A full adder of nine CMOS NAND gates, each gate four transistors.

The plan characterizes the transistor-level NAND gate into an event-driven logic cell.
Nine characterized cells simulate the full adder at event speed, while spot checks
re-simulate individual gates at transistor level against the contract. Simulate with
``jolty simulate examples/full_adder.py``.
"""

from jolty import Builder, Component, Plan, Role, component, devices
from jolty.plan import Characterized, Transient
from jolty.units import V, fF, nm, ns, ps, um
from jolty.waveforms import Pulse

SUPPLY = 3.3 * V


def transistor(channel: str) -> devices.Device:
    """Returns a 1 µm level 1 MOSFET with double width for the p-channel."""
    p = channel == "p"
    return devices.mosfet(
        channel="p" if p else "n",
        width=(2 if p else 1) * um,
        length=1 * um,
        threshold_voltage=(-0.7 if p else 0.7) * V,
        oxide_thickness=20 * nm,
        bulk_drain_capacitance=2 * fF,
        bulk_source_capacitance=2 * fF,
    )


@component
def nand(c: Builder) -> None:
    """Builds a two-input CMOS NAND with parallel p-channel and series n-channel."""
    a, b = c.port("a", Role.INPUT), c.port("b", Role.INPUT)
    y = c.port("y", Role.OUTPUT)
    vdd, gnd = c.port("vdd", Role.SUPPLY), c.port("gnd", Role.GROUND)
    up_a, up_b = (
        c.place(transistor("p"), name="P1"),
        c.place(transistor("p"), name="P2"),
    )
    down_a, down_b = (
        c.place(transistor("n"), name="N1"),
        c.place(transistor("n"), name="N2"),
    )
    c.connect(up_a.source, up_b.source, up_a.bulk, up_b.bulk, vdd)
    c.connect(up_a.drain, up_b.drain, down_a.drain, y)
    c.connect(up_a.gate, down_a.gate, a)
    c.connect(up_b.gate, down_b.gate, b)
    c.connect(down_a.source, down_b.drain, name="stack")
    c.connect(down_b.source, down_a.bulk, down_b.bulk, gnd)


@component
def full_adder(c: Builder) -> None:
    """Builds a three-bit full adder from nine two-input NAND gates."""
    a, b, carry_in = (c.port(name, Role.INPUT) for name in ("a", "b", "carry_in"))
    total, carry_out = c.port("sum", Role.OUTPUT), c.port("carry_out", Role.OUTPUT)
    vdd, gnd = c.port("vdd", Role.SUPPLY), c.port("gnd", Role.GROUND)
    gates = [c.place(nand(), name=f"U{index}") for index in range(1, 10)]
    for gate in gates:
        c.connect(gate.vdd, vdd)
        c.connect(gate.gnd, gnd)
    wiring = [
        (a, b, "n1"),
        (a, "n1", "n2"),
        (b, "n1", "n3"),
        ("n2", "n3", "half"),
        ("half", carry_in, "n4"),
        ("half", "n4", "n5"),
        (carry_in, "n4", "n6"),
        ("n5", "n6", total),
        ("n4", "n1", carry_out),
    ]
    named = dict.fromkeys(
        point for points in wiring for point in points if isinstance(point, str)
    )
    nets = {name: c.net(name) for name in named}
    for gate, points in zip(gates, wiring, strict=True):
        for pin, point in zip((gate.a, gate.b, gate.y), points, strict=True):
            c.connect(pin, nets[point] if isinstance(point, str) else point)


STEP = 2 * ns
"""Duration for which the inputs hold each Gray-code combination."""


@component
def bench(c: Builder) -> None:
    """Builds the adder with inputs stepping through all eight Gray-code combinations.

    One input changes every 2 ns.
    """
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(SUPPLY), name="VDD")
    adder = c.place(full_adder(), name="adder")
    c.connect(supply.positive, adder.vdd, name="vdd")
    c.connect(supply.negative, adder.gnd, ground)
    # Each input's first rise, how long it stays high, and how often it repeats, in
    # steps: carry_in, b and a count 000, 001, 011, 010, 110, 111, 101, 100.
    for port, (delay, width, period) in {
        "carry_in": (1, 2, 4),
        "b": (2, 4, 8),
        "a": (4, 4, 8),
    }.items():
        gray = Pulse(
            initial=0 * V,
            pulsed=SUPPLY,
            delay=delay * STEP,
            rise=100 * ps,
            fall=100 * ps,
            width=width * STEP - 100 * ps,
            period=period * STEP,
        )
        source = c.place(devices.voltage_source(gray), name=f"V{port}")
        c.connect(source.positive, adder[port], name=port)
        c.connect(source.negative, ground)
    for port in ("sum", "carry_out"):
        load = c.place(devices.capacitor(10 * fF), name=f"C{port}")
        c.connect(load.a, adder[port], name=port)
        c.connect(load.b, ground)


def build() -> Component:
    """Returns the full adder test bench component."""
    return bench()


def plan() -> Plan:
    """Returns a 16 ns transient plan using characterized logic cells for the gates.

    Each characterized arc measures a single switching input. Inside the adder XOR, two
    inputs can switch within one gate delay so the output reverses before reaching a
    rail. Spot checks allow half a mean delay tolerance for those partial glitches.
    """
    return Plan(
        {"count": Transient(stop=16 * ns)},
        fidelity={"adder/*": Characterized(SUPPLY, spot_checks=2, tolerance=0.5)},
    )
