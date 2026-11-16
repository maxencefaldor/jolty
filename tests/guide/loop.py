# Copyright 2026 Maxence Faldor
"""A loop of a gain of 1000 and poles at 1 kHz and 100 kHz: `low-phase-margin`.

An inverting gain drives two RC sections, each buffered, whose output feeds back
through the probe `VP` to the gain's input, so the loop gain falls through 0 dB near
316 kHz with about 18° of phase margin.
"""

from jolty import Builder, Component, Plan, Role, component, devices
from jolty.plan import Stability, decade
from jolty.units import GHz, Hz, V, kohm, nF


@component
def loop(c: Builder) -> None:
    """The gain, the two sections and the probe, in a ring."""
    ground = c.port("ground", Role.GROUND)
    gain = c.place(devices.voltage_controlled_voltage_source(1000), name="E1")
    first = c.place(devices.resistor(1 * kohm), name="R1")
    slow = c.place(devices.capacitor(150 * nF), name="C1")
    buffer = c.place(devices.voltage_controlled_voltage_source(1), name="E2")
    second = c.place(devices.resistor(1 * kohm), name="R2")
    fast = c.place(devices.capacitor(1.5 * nF), name="C2")
    probe = c.place(devices.voltage_source(0 * V), name="VP")
    c.connect(gain.positive, first.a, name="amplified")
    c.connect(first.b, slow.a, buffer.control_positive, name="slow")
    c.connect(buffer.positive, second.a, name="buffered")
    c.connect(second.b, fast.a, probe.negative, name="out")
    c.connect(probe.positive, gain.control_negative, name="back")
    c.connect(
        gain.negative,
        gain.control_positive,
        slow.b,
        buffer.negative,
        buffer.control_negative,
        fast.b,
        ground,
    )


def build() -> Component:
    """The loop."""
    return loop()


def plan() -> Plan:
    """The loop gain through the probe."""
    return Plan({"loop": Stability("VP", decade(1 * Hz, 1 * GHz, 20))})
