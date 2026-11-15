# Copyright 2026 Maxence Faldor
"""Square meshes of resistors and capacitors, the cost of the sparse solver's fill.

Each node of an ``n`` by ``n`` grid has 1 pF to ground and 1 kΩ to each neighbour, and
the nodes of the bottom row 1 kΩ to ground besides. A pulse of 1 V, from 1 ns for 1 µs,
into the top left corner spreads through the mesh, its time constant 1 ns. The fixtures
``rc_mesh_100`` and ``rc_mesh_300`` run 10 µs of meshes of 10,000 and 90,000 nodes,
from the operating point.

A row places its nodes' capacitors and resistors, those down to the row below
included, and a block of ten to the power ``n`` rows is ten blocks of ten to the
``n - 1`` stacked, so the circuit's file holds each size of block once.
"""

from fixture import Fixture, write

from jolty import Builder, Component, Plan, Role, component, devices
from jolty.plan import Transient
from jolty.units import V, kohm, ns, pF, us
from jolty.waveforms import Pulse

SIDES = (100, 300)
"""The fixtures' numbers of nodes along each side."""


@component
def row(c: Builder, columns: int) -> None:
    """A row of `columns` nodes, joined to each other and down to the row below."""
    ground = c.port("gnd", Role.GROUND)
    own = [c.port(f"n{column}") for column in range(columns)]
    below = [c.port(f"b{column}") for column in range(columns)]
    for column in range(columns):
        load = c.place(devices.capacitor(1 * pF), id=f"C{column}")
        down = c.place(devices.resistor(1 * kohm), id=f"RV{column}")
        c.connect(own[column], load.a, down.a)
        c.connect(below[column], down.b)
        c.connect(ground, load.b)
        if column + 1 < columns:
            across = c.place(devices.resistor(1 * kohm), id=f"RH{column}")
            c.connect(own[column], across.a)
            c.connect(own[column + 1], across.b)


def rows(decades: int, columns: int) -> Component:
    """A block of ten to the power `decades` rows of `columns` nodes."""
    return row(columns) if decades == 0 else block(decades, columns)


@component
def block(c: Builder, decades: int, columns: int) -> None:
    """Ten blocks of ten to the power ``decades - 1`` rows, each above the next."""
    ground = c.port("gnd", Role.GROUND)
    top = [c.port(f"n{column}") for column in range(columns)]
    bottom = [c.port(f"b{column}") for column in range(columns)]
    above = top
    for index in range(10):
        part = c.place(rows(decades - 1, columns), id=f"X{index}")
        c.connect(ground, part.gnd)
        below = (
            [c.net(f"r{index}c{column}") for column in range(columns)]
            if index < 9
            else bottom
        )
        for column in range(columns):
            c.connect(above[column], part[f"n{column}"])
            c.connect(below[column], part[f"b{column}"])
        above = below


@component
def bench(c: Builder, side: int) -> None:
    """A mesh of `side` by `side` nodes, a digit times a power of ten, under the pulse.

    Raises:
        ValueError: When the side is not a digit times a power of ten.
    """
    decades = len(str(side)) - 1
    count, rest = divmod(side, 10**decades)
    if rest:
        message = f"a mesh's side is a digit times a power of ten, not {side}"
        raise ValueError(message)
    ground = c.port("ground", Role.GROUND)
    pulse = Pulse(
        initial=0 * V,
        pulsed=1 * V,
        delay=1 * ns,
        rise=1 * ns,
        fall=1 * ns,
        width=1 * us,
    )
    source = c.place(devices.voltage_source(pulse), id="V1")
    above = [c.net(f"r0c{column}") for column in range(side)]
    c.connect(source.positive, above[0])
    c.connect(source.negative, ground)
    for index in range(count):
        part = c.place(rows(decades, side), id=f"X{index}")
        c.connect(ground, part.gnd)
        below = (
            [c.net(f"r{index + 1}c{column}") for column in range(side)]
            if index < count - 1
            else [ground] * side
        )
        for column in range(side):
            c.connect(above[column], part[f"n{column}"])
            c.connect(below[column], part[f"b{column}"])
        above = below


def build(side: int = SIDES[0]) -> Component:
    """A mesh of `side` by `side` nodes on its bench."""
    return bench(side)


def plan() -> Plan:
    """Ten microseconds from the operating point."""
    return Plan({"tran": Transient(stop=10 * us)})


def circuits() -> dict[str, Component]:
    """A mesh of each size, by the name of its file."""
    return {f"rc_mesh_{side}": build(side) for side in SIDES}


def fixtures() -> dict[str, Fixture]:
    """A fixture for each size of mesh."""
    return {f"rc_mesh_{side}": Fixture(f"rc_mesh_{side}", plan()) for side in SIDES}


if __name__ == "__main__":
    write(circuits(), fixtures())
