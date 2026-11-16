# Copyright 2026 Maxence Faldor
"""A tank of 1 mH and 1 nF that a negative conductance of 100 µS undamps.

Its poles are `5e4 ± j·√(10¹² - 2.5·10⁹)` rad/s, a mode near 159 kHz that grows from
the operating point: `unstable-pole`.
"""

from jolty import Builder, Component, Plan, Role, component, devices
from jolty.plan import OperatingPoint
from jolty.units import S, mH, nF


@component
def oscillator(c: Builder) -> None:
    """The tank across `n` and the ground, and a source feeding its voltage back."""
    ground = c.port("ground", Role.GROUND)
    inductor = c.place(devices.inductor(1 * mH), name="L1")
    capacitor = c.place(devices.capacitor(1 * nF), name="C1")
    feedback = c.place(devices.voltage_controlled_current_source(1e-4 * S), name="G1")
    c.connect(inductor.a, capacitor.a, feedback.negative, feedback.control_positive)
    c.connect(inductor.a, name="n")
    c.connect(
        inductor.b, capacitor.b, feedback.positive, feedback.control_negative, ground
    )


def build() -> Component:
    """The tank and its feedback."""
    return oscillator()


def plan() -> Plan:
    """The operating point."""
    return Plan({"op": OperatingPoint()})
