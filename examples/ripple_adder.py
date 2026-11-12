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
    """A two-input gate of `kind`, 100 ps from input to output, on a supply."""
    return devices.logic_gate(kind, rise=100 * ps, transition=20 * ps, electrical=True)


@component
def full_adder(c: Builder) -> None:
    """The sum a ⊕ b ⊕ carry_in, and the carry ab + carry_in(a ⊕ b)."""
    a, b = c.port("a", Role.INPUT), c.port("b", Role.INPUT)
    carry_in = c.port("carry_in", Role.INPUT)
    total, carry_out = c.port("sum", Role.OUTPUT), c.port("carry_out", Role.OUTPUT)
    vdd, gnd = c.port("vdd", Role.SUPPLY), c.port("gnd", Role.GROUND)
    half, whole = c.place(gate("xor"), id="U1"), c.place(gate("xor"), id="U2")
    both, through = c.place(gate("and"), id="U3"), c.place(gate("and"), id="U4")
    either = c.place(gate("or"), id="U5")
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
    """Four full adders, each one's carry the next one's."""
    a = [c.port(f"a{bit}", Role.INPUT) for bit in range(BITS)]
    b = [c.port(f"b{bit}", Role.INPUT) for bit in range(BITS)]
    carry_in = c.port("carry_in", Role.INPUT)
    total = [c.port(f"s{bit}", Role.OUTPUT) for bit in range(BITS)]
    carry_out = c.port("carry_out", Role.OUTPUT)
    vdd, gnd = c.port("vdd", Role.SUPPLY), c.port("gnd", Role.GROUND)
    carry = carry_in
    for bit in range(BITS):
        stage = c.place(full_adder(), id=f"FA{bit}")
        c.connect(stage.vdd, vdd)
        c.connect(stage.gnd, gnd)
        c.connect(stage.a, a[bit])
        c.connect(stage.b, b[bit])
        c.connect(stage.carry_in, carry)
        c.connect(stage.sum, total[bit])
        carry = carry_out if bit == BITS - 1 else c.net(f"c{bit + 1}")
        c.connect(stage.carry_out, carry)


def bits(numbers: list[int], bit: int) -> list[tuple[float, LogicLevel]]:
    """The level of `bit` of each number, from each one's time on."""
    return [
        (index * STEP, 1 if (number >> bit) & 1 else 0)
        for index, number in enumerate(numbers)
    ]


@component
def bench(c: Builder) -> None:
    """The adder on 3.3 V, its inputs stepping through the pairs, its outputs loaded."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(SUPPLY * V), id="VDD")
    total = c.place(adder(), id="X1")
    c.connect(supply.positive, total.vdd, name="vdd")
    c.connect(supply.negative, total.gnd, ground)
    zero = c.place(devices.logic_source(0, transition=20 * ps), id="CIN")
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
                id=f"V{operand}{bit}",
            )
            c.connect(source.output, total[f"{operand}{bit}"], name=f"{operand}{bit}")
    for output in [*(f"s{bit}" for bit in range(BITS)), "carry_out"]:
        load = c.place(devices.capacitor(10 * fF), id=f"C{output}")
        c.connect(total[output], load.a, name=output)
        c.connect(load.b, ground)


def build() -> Component:
    """The adder on its bench."""
    return bench()


def plan() -> Plan:
    """Eight nanoseconds, the four additions."""
    return Plan({"add": Transient(stop=8 * ns, max_step=10 * ps)})


def specs() -> Specs:
    """Each sum's bits, read 1.5 ns after its operands change."""
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
