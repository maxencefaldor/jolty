# Copyright 2026 Maxence Faldor
"""A four-bit ripple-carry adder of logic gates, adding pairs of numbers in turn.

Each bit is a full adder of two XOR gates, two AND gates and an OR gate, its carry
rippling into the next bit's. Every gate takes 100 ps, so a carry that ripples through
all four bits settles the sum within a nanosecond. The gates between the bits trade
logic levels at event speed; the outputs drive 10 fF on 3.3 V, read as voltages. The
bench adds 3 + 5, 7 + 6, 15 + 1 and 10 + 11, one pair every 2 ns. Simulate it with
``jolty simulate examples/ripple_adder.py``.
"""

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.devices import Gate, LogicLevel
from jolty.plan import Transient
from jolty.specs import value
from jolty.units import V, fF, mV, ns, ps

SUPPLY = 3.3

BITS = 4
PAIRS = [(3, 5), (7, 6), (15, 1), (10, 11)]
"""The numbers added, a pair every 2 ns."""
STEP = 2e-9


def gate(kind: Gate) -> devices.Device:
    """Returns a two-input logic gate of `kind` with a 100 ps propagation delay."""
    return devices.logic_gate(kind, rise=100 * ps, transition=20 * ps, electrical=True)


@component
def full_adder(c: Builder) -> None:
    """Builds a full adder computing sum a ⊕ b ⊕ carry_in and carry ab + c(a ⊕ b)."""
    a, b = c.port("a", Role.INPUT), c.port("b", Role.INPUT)
    carry_in = c.port("carry_in", Role.INPUT)
    total, carry_out = c.port("sum", Role.OUTPUT), c.port("carry_out", Role.OUTPUT)
    vdd, gnd = c.port("vdd", Role.SUPPLY), c.port("gnd", Role.GROUND)
    half, whole = c.place(gate("xor"), name="U1"), c.place(gate("xor"), name="U2")
    both, through = c.place(gate("and"), name="U3"), c.place(gate("and"), name="U4")
    either = c.place(gate("or"), name="U5")
    for placed in (half, whole, both, through, either):
        c.connect(placed.supply, vdd)
        c.connect(placed.ground, gnd)
    c.connect(a, half.a, both.a)
    c.connect(b, half.b, both.b)
    c.connect(half.y, whole.a, through.a, name="propagate")
    c.connect(carry_in, whole.b, through.b)
    c.connect(whole.y, total)
    c.connect(both.y, either.a, name="generate")
    c.connect(through.y, either.b, name="carried")
    c.connect(either.y, carry_out)


@component
def adder(c: Builder) -> None:
    """Builds four ripple-chained full adders passing each carry to the next bit."""
    a = [c.port(f"a{bit}", Role.INPUT) for bit in range(BITS)]
    b = [c.port(f"b{bit}", Role.INPUT) for bit in range(BITS)]
    carry_in = c.port("carry_in", Role.INPUT)
    total = [c.port(f"s{bit}", Role.OUTPUT) for bit in range(BITS)]
    carry_out = c.port("carry_out", Role.OUTPUT)
    vdd, gnd = c.port("vdd", Role.SUPPLY), c.port("gnd", Role.GROUND)
    carry = carry_in
    for bit in range(BITS):
        stage = c.place(full_adder(), name=f"FA{bit}")
        c.connect(stage.vdd, vdd)
        c.connect(stage.gnd, gnd)
        c.connect(stage.a, a[bit])
        c.connect(stage.b, b[bit])
        c.connect(stage.carry_in, carry)
        c.connect(stage.sum, total[bit])
        carry = carry_out if bit == BITS - 1 else c.net(f"c{bit + 1}")
        c.connect(stage.carry_out, carry)


def bits(numbers: list[int], bit: int) -> list[tuple[float, LogicLevel]]:
    """Returns timed `(time, level)` pairs for `bit` across the operand `numbers`."""
    return [
        (index * STEP, 1 if (number >> bit) & 1 else 0)
        for index, number in enumerate(numbers)
    ]


@component
def bench(c: Builder) -> None:
    """Builds the adder on 3.3 V with stepped operand pairs and capacitive loads."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(SUPPLY * V), name="VDD")
    total = c.place(adder(), name="X1")
    c.connect(supply.positive, total.vdd, name="vdd")
    c.connect(supply.negative, total.gnd, ground)
    zero = c.place(devices.logic_source(0, transition=20 * ps), name="CIN")
    c.connect(zero.output, total.carry_in, name="carry_in")
    for operand, numbers in (
        ("a", [a for a, _ in PAIRS]),
        ("b", [b for _, b in PAIRS]),
    ):
        for bit in range(BITS):
            levels = bits(numbers, bit)
            source = c.place(
                devices.logic_source(
                    levels[0][1],
                    [(time * 1e9 * ns, level) for time, level in levels[1:]],
                    transition=20 * ps,
                ),
                name=f"V{operand}{bit}",
            )
            c.connect(source.output, total[f"{operand}{bit}"], name=f"{operand}{bit}")
    for output in [*(f"s{bit}" for bit in range(BITS)), "carry_out"]:
        load = c.place(devices.capacitor(10 * fF), name=f"C{output}")
        c.connect(total[output], load.a, name=output)
        c.connect(load.b, ground)


def build() -> Component:
    """Returns the ripple-carry adder test bench component."""
    return bench()


def plan() -> Plan:
    """Returns an 8 ns transient plan covering all four operand additions."""
    return Plan({"add": Transient(stop=8 * ns, max_step=10 * ps)})


def specs() -> Specs:
    """Returns sum and carry bit specifications sampled 1.5 ns after each change."""
    requirements = []
    for index, (a, b) in enumerate(PAIRS):
        total = a + b
        at = (index * STEP + 1.5e-9) * 1e9 * ns
        for bit, output in enumerate(
            [*(f"s{bit}" for bit in range(BITS)), "carry_out"]
        ):
            level = SUPPLY if (total >> bit) & 1 else 0.0
            requirements.append(value("add", output, at=at).within(level * V, 10 * mV))
    return Specs(requirements)
