# Copyright 2026 Maxence Faldor
"""A four-bit R-2R ladder DAC, counting from 0 to 15.

Each bit's logic level drives a 20 kΩ leg of a ladder of 10 kΩ rungs, terminated by
20 kΩ to ground. Every node of the ladder looks into 20 kΩ either way, so each bit adds
its weight, half the previous bit's, and the output is the supply times the code over
16, exactly. A counter of logic sources steps the code every 10 ns, through buffers on
3.3 V. Simulate it with ``jolty simulate examples/r2r_dac.py``.
"""

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.devices import LogicLevel
from jolty.plan import Transient
from jolty.specs import value
from jolty.units import Quantity, V, kohm, mV, ns, ps

BITS = 4
SUPPLY = 3.3
STEP = 10e-9


@component
def ladder(c: Builder) -> None:
    """The 2R legs of each bit and the R rungs between them, the LSB's end grounded."""
    legs = [c.port(f"d{bit}", Role.INPUT) for bit in range(BITS)]
    output = c.port("output", Role.OUTPUT)
    ground = c.port("ground", Role.GROUND)
    end = c.place(devices.resistor(20 * kohm), id="RT")
    node = c.net("n0")
    c.connect(end.a, node)
    c.connect(end.b, ground)
    for bit in range(BITS):
        leg = c.place(devices.resistor(20 * kohm), id=f"R2_{bit}")
        c.connect(leg.a, legs[bit])
        c.connect(leg.b, node)
        if bit < BITS - 1:
            rung = c.place(devices.resistor(10 * kohm), id=f"R1_{bit}")
            following = output if bit == BITS - 2 else c.net(f"n{bit + 1}")
            c.connect(rung.a, node)
            c.connect(rung.b, following)
            node = following


@component
def bench(c: Builder) -> None:
    """The ladder, driven by buffers from a counter of logic sources."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(SUPPLY * V), id="VDD")
    converter = c.place(ladder(), id="X1")
    vdd = c.net("vdd")
    c.connect(supply.positive, vdd)
    for bit in range(BITS):
        changes: list[tuple[Quantity, LogicLevel]] = [
            (code * STEP * 1e9 * ns, 1 if (code >> bit) & 1 else 0)
            for code in range(1, 2**BITS)
        ]
        count = c.place(
            devices.logic_source(0, changes, transition=100 * ps), id=f"V{bit}"
        )
        buffer = c.place(
            devices.logic_gate(
                "buffer", rise=200 * ps, transition=100 * ps, electrical=True
            ),
            id=f"U{bit}",
        )
        c.connect(count.output, buffer.a, name=f"code{bit}")
        c.connect(buffer.y, converter[f"d{bit}"], name=f"d{bit}")
        c.connect(buffer.supply, vdd)
        c.connect(buffer.ground, ground)
    c.connect(converter.output, name="out")
    c.connect(supply.negative, converter.ground, ground)


def build() -> Component:
    """The converter on its bench."""
    return bench()


def plan() -> Plan:
    """The sixteen codes, 10 ns each."""
    return Plan({"count": Transient(stop=2**BITS * STEP * 1e9 * ns, max_step=0.5 * ns)})


def specs() -> Specs:
    """Each code's output the supply times the code over 16, to a microvolt."""
    return Specs(
        [
            value("count", "out", at=(code + 0.5) * STEP * 1e9 * ns).within(
                SUPPLY * code / 2**BITS * V, 0.001 * mV
            )
            for code in range(2**BITS)
        ]
    )
