# Copyright 2026 Maxence Faldor
"""MAGIC three-memristor NOR gates and four-case write test bench.

Each gate connects two parallel input memristors in series with an output memristor
from `drive` to ground per MAGIC (Kvatinsky et al., 2014).
Current drives inputs toward the on state and the output toward the off state.
The output begins at logic 1 (on) and stays on only when both inputs are 0 (off),
keeping voltage across the output below its switching threshold.
"""

from typing import Final

from jolty import Builder, Role, component, devices
from jolty.units import V, kohm, ns, ohm, s
from jolty.waveforms import Pulse

CASES: Final = ((0, 0), (0, 1), (1, 0), (1, 1))
"""Input bit pairs assigned to gates X1 through X4 on the test bench."""

WIDTH: Final = 10.0
"""Nominal write pulse duration in nanoseconds, matching the prompt limit."""


def memristor(bit: int) -> devices.Device:
    """Builds the prompt's VTEAM memristor initialized to `bit` (1 kΩ on for 1)."""
    return devices.voltage_controlled_memristor(
        on_resistance=1 * kohm,
        off_resistance=100 * kohm,
        on_voltage=-1 * V,
        off_voltage=0.5 * V,
        on_rate=2e8 / s,
        off_rate=2e8 / s,
        on_exponent=3,
        off_exponent=3,
        law="exponential",
        state=0.0 if bit else 1.0,
    )


@component
def nor(c: Builder, a: int, b: int, shunt: float | None = None) -> None:
    """Builds a MAGIC NOR gate for bits `a` and `b` with optional `shunt` ohms."""
    drive = c.port("drive")
    ground = c.port("ground", Role.GROUND)
    first = c.place(memristor(a), name="MA")
    second = c.place(memristor(b), name="MB")
    output = c.place(memristor(1), name="MO")
    c.connect(drive, first.negative, second.negative)
    c.connect(first.positive, second.positive, output.positive, name="joint")
    c.connect(output.negative, ground)
    if shunt is not None:
        resistor = c.place(devices.resistor(shunt * ohm), name="RS")
        c.connect(drive, resistor.a)
        c.connect(resistor.b, output.positive)


@component
def bench(
    c: Builder, amplitude: float, shunt: float | None = None, width: float = WIDTH
) -> None:
    """Builds four MAGIC NOR gates (X1 to X4) driven by a pulse of `amplitude`.

    The pulse rises in 0.1 ns after a 1 ns delay, holds for `width` nanoseconds, and
    falls in 0.1 ns.
    """
    ground = c.port("ground", Role.GROUND)
    pulse = Pulse(
        initial=0 * V,
        pulsed=amplitude * V,
        delay=1 * ns,
        rise=0.1 * ns,
        fall=0.1 * ns,
        width=width * ns,
    )
    source = c.place(devices.voltage_source(pulse), name="V1")
    gates = [
        c.place(nor(a, b, shunt), name=f"X{index}")
        for index, (a, b) in enumerate(CASES, start=1)
    ]
    c.connect(source.positive, *(gate.drive for gate in gates), name="drive")
    c.connect(source.negative, *(gate.ground for gate in gates), ground)
