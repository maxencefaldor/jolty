# Copyright 2026 Maxence Faldor
"""Chains of CMOS inverters, the cost of evaluating devices and of Newton iteration.

Each inverter is a pair of level 1 MOSFETs of a 1 µm process on 3.3 V, and drives the
next one's gates. A pulse of 3.3 V, from 1 ns to 21 ns, starts an edge down the chain at
each of its corners. The fixtures ``inverter_chain_500``, ``inverter_chain_5000`` and
``inverter_chain_50000`` run 40 ns of chains of 500, 5,000 and 50,000 inverters, from
the operating point.

A chain of ten to the power ``n`` inverters is ten chains of ten to the ``n - 1`` in
series, so an inverter's path spells its place in the chain in decimal digits, and the
circuit's file holds each size of chain once.
"""

from fixture import Fixture, write

from jolty import Builder, Component, Plan, Role, component, devices
from jolty.plan import Options, Transient
from jolty.units import V, fF, nm, ns, um
from jolty.waveforms import Pulse

SUPPLY = 3.3
"""The supply, in volts."""

LENGTHS = (500, 5000, 50_000)
"""The fixtures' numbers of inverters."""

MEMORY = 4.0
"""The gibibytes a chain's results may hold: the 350,004 unknowns of 50,000 inverters,
each of whose transistors has three Meyer displacements of its own, at some 800 points,
hold 2.1, more than the 2 the engine allows by default."""


def mos(channel: str) -> devices.Device:
    """A level 1 MOSFET 1 µm long, 1 µm wide for an n-channel and 2 µm for a p."""
    p = channel == "p"
    return devices.mosfet(
        channel="p" if p else "n",
        width=(2 if p else 1) * um,
        length=1 * um,
        threshold_voltage=(-0.7 if p else 0.7) * V,
        channel_length_modulation=0.05 / V,
        oxide_thickness=20 * nm,
        bulk_drain_capacitance=2 * fF,
        bulk_source_capacitance=2 * fF,
    )


@component
def inverter(c: Builder) -> None:
    """A p-channel transistor up to the supply and an n-channel one down to ground."""
    a, y = c.port("a", Role.INPUT), c.port("y", Role.OUTPUT)
    vdd, gnd = c.port("vdd", Role.SUPPLY), c.port("gnd", Role.GROUND)
    up, down = c.place(mos("p"), id="MP"), c.place(mos("n"), id="MN")
    c.connect(a, up.gate, down.gate)
    c.connect(y, up.drain, down.drain)
    c.connect(vdd, up.source, up.bulk)
    c.connect(gnd, down.source, down.bulk)


def stages(decades: int) -> Component:
    """A chain of ten to the power `decades` inverters."""
    return inverter() if decades == 0 else chain(decades)


@component
def chain(c: Builder, decades: int) -> None:
    """Ten chains of ten to the power ``decades - 1`` inverters, in series."""
    a, y = c.port("a", Role.INPUT), c.port("y", Role.OUTPUT)
    vdd, gnd = c.port("vdd", Role.SUPPLY), c.port("gnd", Role.GROUND)
    previous = a
    for index in range(10):
        part = c.place(stages(decades - 1), id=f"X{index}")
        c.connect(previous, part.a)
        c.connect(vdd, part.vdd)
        c.connect(gnd, part.gnd)
        previous = c.net(f"n{index}") if index < 9 else y
        c.connect(part.y, previous)


@component
def bench(c: Builder, length: int) -> None:
    """A chain of `length` inverters, a digit times a power of ten, under the pulse.

    Raises:
        ValueError: When the length is not a digit times a power of ten.
    """
    decades = len(str(length)) - 1
    count, rest = divmod(length, 10**decades)
    if rest:
        message = f"a chain's length is a digit times a power of ten, not {length}"
        raise ValueError(message)
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(SUPPLY * V), id="VDD")
    pulse = Pulse(
        initial=0 * V,
        pulsed=SUPPLY * V,
        delay=1 * ns,
        rise=0.1 * ns,
        fall=0.1 * ns,
        width=19.9 * ns,
    )
    source = c.place(devices.voltage_source(pulse), id="V1")
    vdd, previous = c.net("vdd"), c.net("in")
    c.connect(supply.positive, vdd)
    c.connect(source.positive, previous)
    c.connect(supply.negative, source.negative, ground)
    for index in range(count):
        part = c.place(stages(decades), id=f"X{index}")
        c.connect(previous, part.a)
        c.connect(vdd, part.vdd)
        c.connect(ground, part.gnd)
        previous = c.net(f"n{index}") if index < count - 1 else c.net("out")
        c.connect(part.y, previous)


def build(length: int = LENGTHS[0]) -> Component:
    """A chain of `length` inverters on its bench."""
    return bench(length)


def plan() -> Plan:
    """Forty nanoseconds from the operating point."""
    return Plan({"tran": Transient(stop=40 * ns)}, options=Options(memory=MEMORY))


def circuits() -> dict[str, Component]:
    """A chain of each length, by the name of its file."""
    return {f"inverter_chain_{length}": build(length) for length in LENGTHS}


def fixtures() -> dict[str, Fixture]:
    """A fixture for each length of chain."""
    return {
        f"inverter_chain_{length}": Fixture(f"inverter_chain_{length}", plan())
        for length in LENGTHS
    }


if __name__ == "__main__":
    write(circuits(), fixtures())
