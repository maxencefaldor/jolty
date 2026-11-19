# Copyright 2026 Maxence Faldor
"""CMOS inverter chain fixtures measuring device evaluation and Newton cost.

Each inverter pairs level-1 1 µm MOSFETs on a 3.3 V supply driving the next stage.
A 3.3 V input pulse from 1 ns to 21 ns launches rising and falling edges down the
chain. The fixtures ``inverter_chain_500``, ``inverter_chain_5000``, and
``inverter_chain_50000`` simulate 500, 5,000, and 50,000 inverters for 40 ns from
the operating point.

Each 10^n-stage chain composes ten 10^(n - 1)-stage subchains in series. Decimal
instance paths encode stage positions directly, and serialized circuit JSON stores
each decade block once.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples"))

from common import SUPPLY, mos
from fixture import Fixture, write

from jolty import Builder, Component, Plan, Role, component, devices
from jolty.plan import Options, Transient
from jolty.units import V, ns
from jolty.waveforms import Pulse

LENGTHS = (500, 5000, 50_000)
"""Inverter counts generated as speed benchmark fixtures."""

MEMORY = 16.0
"""Memory limit in gibibytes for inverter chain transient outputs. The 50,000-stage
chain has 350,004 unknowns (three Meyer displacements per transistor), using 2.1 GiB
at 800 points and over 4 GiB under SPICE charge tolerance; 16 GiB covers runs on
`cedar`."""


@component
def inverter(c: Builder) -> None:
    """Builds a CMOS inverter with PMOS to supply and NMOS to ground."""
    a, y = c.port("a", Role.INPUT), c.port("y", Role.OUTPUT)
    vdd, gnd = c.port("vdd", Role.SUPPLY), c.port("gnd", Role.GROUND)
    up, down = c.place(mos("p"), name="MP"), c.place(mos("n"), name="MN")
    c.connect(a, up.gate, down.gate)
    c.connect(y, up.drain, down.drain)
    c.connect(vdd, up.source, up.bulk)
    c.connect(gnd, down.source, down.bulk)


def stages(decades: int) -> Component:
    """Returns a subchain component containing ten to the power `decades` inverters."""
    return inverter() if decades == 0 else chain(decades)


@component
def chain(c: Builder, decades: int) -> None:
    """Builds ten series subchains of ten to the power ``decades - 1`` inverters."""
    a, y = c.port("a", Role.INPUT), c.port("y", Role.OUTPUT)
    vdd, gnd = c.port("vdd", Role.SUPPLY), c.port("gnd", Role.GROUND)
    previous = a
    for index in range(10):
        part = c.place(stages(decades - 1), name=f"X{index}")
        c.connect(previous, part.a)
        c.connect(vdd, part.vdd)
        c.connect(gnd, part.gnd)
        previous = c.net(f"n{index}") if index < 9 else y
        c.connect(part.y, previous)


@component
def bench(c: Builder, length: int) -> None:
    """Builds a pulsed bench for `length` inverters (a digit times a power of ten).

    Raises:
        ValueError: When `length` is not a single digit times a power of ten.
    """
    decades = len(str(length)) - 1
    count, rest = divmod(length, 10**decades)
    if rest:
        message = f"a chain's length is a digit times a power of ten, not {length}"
        raise ValueError(message)
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(SUPPLY * V), name="VDD")
    pulse = Pulse(
        initial=0 * V,
        pulsed=SUPPLY * V,
        delay=1 * ns,
        rise=0.1 * ns,
        fall=0.1 * ns,
        width=19.9 * ns,
    )
    source = c.place(devices.voltage_source(pulse), name="V1")
    vdd, previous = c.net("vdd"), c.net("in")
    c.connect(supply.positive, vdd)
    c.connect(source.positive, previous)
    c.connect(supply.negative, source.negative, ground)
    for index in range(count):
        part = c.place(stages(decades), name=f"X{index}")
        c.connect(previous, part.a)
        c.connect(vdd, part.vdd)
        c.connect(ground, part.gnd)
        previous = c.net(f"n{index}") if index < count - 1 else c.net("out")
        c.connect(part.y, previous)


def build(length: int = LENGTHS[0]) -> Component:
    """Returns the pulsed test bench for a chain of `length` inverters."""
    return bench(length)


def plan() -> Plan:
    """Returns the 40 ns transient plan with the `MEMORY` ceiling."""
    return Plan({"tran": Transient(stop=40 * ns)}, options=Options(max_memory=MEMORY))


def circuits() -> dict[str, Component]:
    """Returns inverter chain circuits for all `LENGTHS` keyed by file stem."""
    return {f"inverter_chain_{length}": build(length) for length in LENGTHS}


def fixtures() -> dict[str, Fixture]:
    """Returns speed benchmark fixtures for all `LENGTHS`."""
    return {
        f"inverter_chain_{length}": Fixture(f"inverter_chain_{length}", plan())
        for length in LENGTHS
    }


if __name__ == "__main__":
    write(circuits(), fixtures())
