# Copyright 2026 Maxence Faldor
"""A 1 mH / 1 nF tank undamped by a 100 µS negative conductance.

The linearized poles sit at `5e4 ± j·√(10¹² - 2.5·10⁹)` rad/s, creating a
growing mode near 159 kHz (`unstable-pole`).
"""

from jolty import Builder, Component, Plan, Role, component, devices
from jolty.plan import OperatingPoint
from jolty.units import S, mH, nF


@component
def oscillator(c: Builder) -> None:
    """Builds the tank across `n` and ground with positive-feedback transconductance."""
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
    """Returns the unstable-tank test circuit."""
    return oscillator()


def plan() -> Plan:
    """Returns the operating-point simulation plan."""
    return Plan({"op": OperatingPoint()})
