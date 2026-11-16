# Copyright 2026 Maxence Faldor
"""NOR gates of three memristors, MAGIC's, and the bench that writes all four inputs.

Each gate drives its two inputs in parallel from `drive`, in series with its output to
the ground, oriented as MAGIC's (Kvatinsky et al., 2014): the current turns the inputs
toward on and the output toward off. Its output starts at 1, on, and stays there only
if both inputs are 0, off, so their resistance holds the voltage across it below its
threshold.
"""

from typing import Final

from jolty import Builder, Role, component, devices
from jolty.units import V, kohm, ns, ohm, s
from jolty.waveforms import Pulse

CASES: Final = ((0, 0), (0, 1), (1, 0), (1, 1))
"""The inputs each gate of the bench holds, from X1 to X4."""

WIDTH: Final = 10.0
"""The pulse's width, in nanoseconds, the longest the prompt allows."""


def memristor(bit: int) -> devices.Device:
    """The prompt's VTEAM memristor, holding `bit`: on, at 1 kΩ, for 1."""
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
    """MAGIC's NOR of the bits `a` and `b`, its inputs shunted by `shunt` ohms."""
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
    """Four gates, X1 to X4, one per pair of `CASES`, written by a pulse of `amplitude`.

    The pulse rises in 0.1 ns after 1 ns, holds for `width` nanoseconds, and falls in
    0.1 ns.
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
