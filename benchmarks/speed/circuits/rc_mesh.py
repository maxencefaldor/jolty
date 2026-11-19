# Copyright 2026 Maxence Faldor
"""Square RC mesh fixtures measuring sparse linear solver fill-in cost.

Each node of an ``n`` by ``n`` grid connects via 1 pF to ground and 1 kΩ to each
neighbor, with 1 kΩ ground resistors along the bottom row. A 1 V, 1 µs pulse at
1 ns drives the top-left corner through the 1 ns time-constant mesh. The fixtures
``rc_mesh_100`` and ``rc_mesh_300`` simulate 10,000- and 90,000-node grids for
10 µs from the operating point.

Each row places capacitors, horizontal resistors, and vertical links to the row
below. Stacking ten 10^(n - 1)-row blocks builds 10^n rows while storing each
decade block once in serialized JSON.
"""

from fixture import Fixture, write

from jolty import Builder, Component, Plan, Role, component, devices
from jolty.plan import Transient
from jolty.units import V, kohm, ns, pF, us
from jolty.waveforms import Pulse

SIDES = (100, 300)
"""Grid side lengths generated as speed benchmark fixtures."""


@component
def row(c: Builder, columns: int) -> None:
    """Builds a row of `columns` RC nodes linked horizontally and to the row below."""
    ground = c.port("gnd", Role.GROUND)
    own = [c.port(f"n{column}") for column in range(columns)]
    below = [c.port(f"b{column}") for column in range(columns)]
    for column in range(columns):
        load = c.place(devices.capacitor(1 * pF), name=f"C{column}")
        down = c.place(devices.resistor(1 * kohm), name=f"RV{column}")
        c.connect(own[column], load.a, down.a)
        c.connect(below[column], down.b)
        c.connect(ground, load.b)
        if column + 1 < columns:
            across = c.place(devices.resistor(1 * kohm), name=f"RH{column}")
            c.connect(own[column], across.a)
            c.connect(own[column + 1], across.b)


def rows(decades: int, columns: int) -> Component:
    """Returns a block of ten to the power `decades` rows with `columns` nodes each."""
    return row(columns) if decades == 0 else block(decades, columns)


@component
def block(c: Builder, decades: int, columns: int) -> None:
    """Stacks ten sub-blocks of ten to the power ``decades - 1`` rows vertically."""
    ground = c.port("gnd", Role.GROUND)
    top = [c.port(f"n{column}") for column in range(columns)]
    bottom = [c.port(f"b{column}") for column in range(columns)]
    above = top
    for index in range(10):
        part = c.place(rows(decades - 1, columns), name=f"X{index}")
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
    """Builds a pulsed `side`-by-`side` RC mesh bench (a digit times a power of ten).

    Raises:
        ValueError: When `side` is not a single digit times a power of ten.
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
    source = c.place(devices.voltage_source(pulse), name="V1")
    above = [c.net(f"r0c{column}") for column in range(side)]
    c.connect(source.positive, above[0])
    c.connect(source.negative, ground)
    for index in range(count):
        part = c.place(rows(decades, side), name=f"X{index}")
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
    """Returns the pulsed test bench for a `side`-by-`side` RC mesh."""
    return bench(side)


def plan() -> Plan:
    """Returns the 10 µs transient plan from the operating point."""
    return Plan({"tran": Transient(stop=10 * us)})


def circuits() -> dict[str, Component]:
    """Returns RC mesh circuits for all `SIDES` keyed by file stem."""
    return {f"rc_mesh_{side}": build(side) for side in SIDES}


def fixtures() -> dict[str, Fixture]:
    """Returns speed benchmark fixtures for all `SIDES`."""
    return {f"rc_mesh_{side}": Fixture(f"rc_mesh_{side}", plan()) for side in SIDES}


if __name__ == "__main__":
    write(circuits(), fixtures())
