# Copyright 2026 Maxence Faldor
"""Reusable test circuits shared across unit and integration tests."""

from typing import Final

from jolty import Builder, Net, Role, component, devices
from jolty.units import Quantity, V, kohm, nF, us
from jolty.waveforms import Piecewise

CROSSBAR_WEIGHTS: Final[dict[str, dict[str, float]]] = {
    f"W{row}{column}": {"resistance": 1e3 * (1 + row + 2 * column)}
    for row in range(2)
    for column in range(2)
}
"""Nominal crossbar weights mapping each resistor path to its resistance in ohms."""


@component
def divider(
    c: Builder,
    level: Quantity = 5 * V,
    upper: Quantity = 1 * kohm,
    lower: Quantity = 1 * kohm,
    *,
    source_name: str = "V1",
) -> None:
    """Build a resistive voltage divider driving net `out` from `source_name`."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(level), name=source_name)
    top = c.place(devices.resistor(upper), name="R1")
    bottom = c.place(devices.resistor(lower), name="R2")
    c.connect(source.positive, top.a, name="in")
    c.connect(top.b, bottom.a, name="out")
    c.connect(bottom.b, source.negative, ground)


@component
def low_pass(c: Builder) -> None:
    """Build a unit-step 1 kohm / 1 nF RC low-pass filter."""
    ground = c.port("ground", Role.GROUND)
    step = Piecewise([(0 * us, 0 * V), (1e-3 * us, 1 * V)])
    source = c.place(devices.voltage_source(step), name="V1")
    resistor = c.place(devices.resistor(1 * kohm), name="R1")
    capacitor = c.place(devices.capacitor(1 * nF), name="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)


@component
def crossbar(c: Builder, first_weight: Quantity = 1 * kohm) -> None:
    """Build a two-input, two-output loaded resistive crossbar."""
    ground = c.port("ground", Role.GROUND)
    for row, level in enumerate((0.3 * V, -0.2 * V)):
        source = c.place(devices.voltage_source(level), name=f"V{row}")
        c.connect(source.positive, name=f"in{row}")
        c.connect(source.negative, ground)
    for column in range(2):
        load = c.place(devices.resistor(1 * kohm), name=f"RL{column}")
        hold = c.place(devices.capacitor(1 * nF), name=f"CL{column}")
        c.connect(load.a, hold.a, name=f"out{column}")
        c.connect(load.b, hold.b, ground)
        for row in range(2):
            nominal = first_weight if (row == 0 and column == 0) else 1 * kohm
            weight = c.place(devices.resistor(nominal), name=f"W{row}{column}")
            c.connect(weight.a, Net(f"in{row}"))
            c.connect(weight.b, Net(f"out{column}"))
