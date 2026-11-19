# Copyright 2026 Maxence Faldor
"""Toleranced 50 Hz notch filter candidates and AC evaluation bench."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples"))

from common import opamp

from jolty import Builder, Component, Role, component, devices
from jolty.builder import Connectable
from jolty.units import F, V, ohm

FREQUENCY = 50.0
CAPACITANCE = 100e-9
RESISTANCE = 1 / (2 * math.pi * FREQUENCY * CAPACITANCE)
RESISTOR, CAPACITOR = 0.01, 0.05
"""Relative tolerances applied to resistors (1%) and capacitors (5%)."""


def r(ohms: float) -> devices.Device:
    """Builds a resistor with 1% tolerance."""
    return devices.resistor(ohms * ohm, tolerance=RESISTOR)


def cap(farads: float) -> devices.Device:
    """Builds a capacitor with 5% tolerance."""
    return devices.capacitor(farads * F, tolerance=CAPACITOR)


def _twin_t_bridge(
    c: Builder, source: Connectable
) -> tuple[tuple[Connectable, Connectable], tuple[Connectable, Connectable]]:
    """Places the twin-T bridge from `source` and returns its output and leg pairs."""
    r1, r2, c3 = (
        c.place(r(RESISTANCE), name="R1"),
        c.place(r(RESISTANCE), name="R2"),
        c.place(cap(2 * CAPACITANCE), name="C3"),
    )
    c1, c2, r3 = (
        c.place(cap(CAPACITANCE), name="C1"),
        c.place(cap(CAPACITANCE), name="C2"),
        c.place(r(RESISTANCE / 2), name="R3"),
    )
    c.connect(source, r1.a, c1.a)
    c.connect(r1.b, r2.a, c3.a, name="resistive")
    c.connect(c1.b, c2.a, r3.a, name="capacitive")
    return (r2.b, c2.b), (c3.b, r3.b)


@component
def twin_t(c: Builder) -> None:
    """Builds a passive twin-T notch with R over 2C and C over R/2 branches."""
    source, output = c.port("input"), c.port("output")
    ground = c.port("ground", Role.GROUND)
    bridge_out, leg = _twin_t_bridge(c, source)
    c.connect(*bridge_out, output)
    c.connect(*leg, ground)


@component
def summed(c: Builder, quality: float = 1.0) -> None:
    """Builds a Tow-Thomas biquad summing band-pass and inverted low-pass outputs.

    The band-pass output through C and the inverted low-pass output through R meet at
    the summing amplifier virtual ground.
    Both outputs remain exactly 90° apart at every frequency, so adding another 90°
    through C and R guarantees exact cancellation wherever magnitudes match.
    Component tolerances shift notch frequency without reducing notch depth.
    """
    source, output = c.port("input"), c.port("output")
    ground = c.port("ground", Role.GROUND)
    lossy, integrator, inverter, summer = (
        c.place(opamp(), name=f"U{n}") for n in (1, 2, 3, 4)
    )
    r_in, r_q = (
        c.place(r(RESISTANCE), name="RIN"),
        c.place(r(quality * RESISTANCE), name="RQ"),
    )
    r2, r4 = c.place(r(RESISTANCE), name="R2"), c.place(r(RESISTANCE), name="R4")
    r3a, r3b = c.place(r(10e3), name="R3A"), c.place(r(10e3), name="R3B")
    r_n, r_f = c.place(r(RESISTANCE), name="RN"), c.place(r(RESISTANCE), name="RF")
    c1, c2, c_n = (
        c.place(cap(CAPACITANCE), name="C1"),
        c.place(cap(CAPACITANCE), name="C2"),
        c.place(cap(CAPACITANCE), name="CN"),
    )
    c.connect(source, r_in.a)
    c.connect(r_in.b, r_q.a, c1.a, r4.b, lossy.inn, name="sum1")
    c.connect(lossy.out, r_q.b, c1.b, r2.a, c_n.a, name="band")
    c.connect(r2.b, c2.a, integrator.inn, name="sum2")
    c.connect(integrator.out, c2.b, r3a.a, name="low")
    c.connect(r3a.b, r3b.a, inverter.inn, name="sum3")
    c.connect(inverter.out, r3b.b, r4.a, r_n.a, name="low_inverted")
    c.connect(c_n.b, r_n.b, r_f.a, summer.inn, name="sum4")
    c.connect(summer.out, r_f.b, output)
    c.connect(
        lossy.inp,
        integrator.inp,
        inverter.inp,
        summer.inp,
        lossy.ground,
        integrator.ground,
        inverter.ground,
        summer.ground,
        ground,
    )


@component
def bench(c: Builder, notch: Component) -> None:
    """Builds a test bench driving `notch` from an AC source with a 1 MΩ load."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(0 * V), name="V1")
    placed = c.place(notch, name="X1")
    load = c.place(devices.resistor(1e6 * ohm), name="RL")
    c.connect(source.positive, placed.input, name="in")
    c.connect(placed.output, load.a, name="out")
    c.connect(source.negative, placed.ground, load.b, ground)


@component
def bootstrapped(c: Builder, feedback: float = 0.75) -> None:
    """Builds a twin-T notch whose ground leg is driven by a buffered output fraction.

    A unity follower buffers the notch output and a second follower drives a fraction
    into the R/2 and 2C junction, raising Q to 1/(4(1 - k)).
    Notch depth still depends on the passive twin-T bridge balance.
    """
    source, output = c.port("input"), c.port("output")
    ground = c.port("ground", Role.GROUND)
    bridge_out, leg = _twin_t_bridge(c, source)
    buffer, bootstrap = c.place(opamp(), name="U1"), c.place(opamp(), name="U2")
    upper, lower = (
        c.place(r(10e3 * (1 - feedback)), name="RK1"),
        c.place(r(10e3 * feedback), name="RK2"),
    )
    c.connect(*bridge_out, buffer.inp, name="bridge")
    c.connect(buffer.out, buffer.inn, upper.a, output)
    c.connect(upper.b, lower.a, bootstrap.inp, name="fraction")
    c.connect(bootstrap.out, bootstrap.inn, *leg, name="leg")
    c.connect(lower.b, buffer.ground, bootstrap.ground, ground)
