# Copyright 2026 Maxence Faldor
"""Notch filters for 50 Hz, with toleranced parts, and their bench."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

from common import opamp

from jolty import Builder, Component, Role, component, devices
from jolty.units import F, V, ohm

FREQUENCY = 50.0
CAPACITANCE = 100e-9
RESISTANCE = 1 / (2 * math.pi * FREQUENCY * CAPACITANCE)
RESISTOR, CAPACITOR = 0.01, 0.05
"""The tolerances of the resistors and of the capacitors."""


def r(ohms: float) -> devices.Device:
    """A 1 % resistor."""
    return devices.resistor(ohms * ohm, tolerance=RESISTOR)


def cap(farads: float) -> devices.Device:
    """A 5 % capacitor."""
    return devices.capacitor(farads * F, tolerance=CAPACITOR)


@component
def twin_t(c: Builder) -> None:
    """The passive twin-T: two resistors R over 2C, two capacitors C over R/2."""
    source, output = c.port("input"), c.port("output")
    ground = c.port("ground", Role.GROUND)
    r1, r2, c3 = (
        c.place(r(RESISTANCE), id="R1"),
        c.place(r(RESISTANCE), id="R2"),
        c.place(cap(2 * CAPACITANCE), id="C3"),
    )
    c1, c2, r3 = (
        c.place(cap(CAPACITANCE), id="C1"),
        c.place(cap(CAPACITANCE), id="C2"),
        c.place(r(RESISTANCE / 2), id="R3"),
    )
    c.connect(source, r1.a, c1.a)
    c.connect(r1.b, r2.a, c3.a, name="resistive")
    c.connect(c1.b, c2.a, r3.a, name="capacitive")
    c.connect(r2.b, c2.b, output)
    c.connect(c3.b, r3.b, ground)


@component
def summed(c: Builder, quality: float = 1.0) -> None:
    """A Tow-Thomas biquad's band-pass and inverted low-pass outputs, summed.

    The band-pass output, through C, and the inverted low-pass output, through R, meet
    at a summing amplifier's virtual ground.

    The two outputs are exactly 90° apart at every frequency, so their currents, 90°
    apart again through C and R, cancel wherever their magnitudes meet: the parts'
    values move the notch, never its depth.
    """
    source, output = c.port("input"), c.port("output")
    ground = c.port("ground", Role.GROUND)
    lossy, integrator, inverter, summer = (
        c.place(opamp(), id=f"U{n}") for n in (1, 2, 3, 4)
    )
    r_in, r_q = (
        c.place(r(RESISTANCE), id="RIN"),
        c.place(r(quality * RESISTANCE), id="RQ"),
    )
    r2, r4 = c.place(r(RESISTANCE), id="R2"), c.place(r(RESISTANCE), id="R4")
    r3a, r3b = c.place(r(10e3), id="R3A"), c.place(r(10e3), id="R3B")
    r_n, r_f = c.place(r(RESISTANCE), id="RN"), c.place(r(RESISTANCE), id="RF")
    c1, c2, c_n = (
        c.place(cap(CAPACITANCE), id="C1"),
        c.place(cap(CAPACITANCE), id="C2"),
        c.place(cap(CAPACITANCE), id="CN"),
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
    """The notch fed by a source, its output loaded by 1 MΩ."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(0 * V), id="V1")
    placed = c.place(notch, id="X1")
    load = c.place(devices.resistor(1e6 * ohm), id="RL")
    c.connect(source.positive, placed.input, name="in")
    c.connect(placed.output, load.a, name="out")
    c.connect(source.negative, placed.ground, load.b, ground)


@component
def bootstrapped(c: Builder, feedback: float = 0.75) -> None:
    """The twin-T with its ground leg driven by a fraction of its buffered output.

    A follower buffers the notch, and a second follows a divider of its output into the
    junction of R/2 and 2C, which raises the notch's Q to 1/(4(1 - k)): 1 at k = 3/4.
    The bridge's balance, and so the depth, is the passive twin-T's.
    """
    source, output = c.port("input"), c.port("output")
    ground = c.port("ground", Role.GROUND)
    r1, r2, c3 = (
        c.place(r(RESISTANCE), id="R1"),
        c.place(r(RESISTANCE), id="R2"),
        c.place(cap(2 * CAPACITANCE), id="C3"),
    )
    c1, c2, r3 = (
        c.place(cap(CAPACITANCE), id="C1"),
        c.place(cap(CAPACITANCE), id="C2"),
        c.place(r(RESISTANCE / 2), id="R3"),
    )
    buffer, bootstrap = c.place(opamp(), id="U1"), c.place(opamp(), id="U2")
    upper, lower = (
        c.place(r(10e3 * (1 - feedback)), id="RK1"),
        c.place(r(10e3 * feedback), id="RK2"),
    )
    c.connect(source, r1.a, c1.a)
    c.connect(r1.b, r2.a, c3.a, name="resistive")
    c.connect(c1.b, c2.a, r3.a, name="capacitive")
    c.connect(r2.b, c2.b, buffer.inp, name="bridge")
    c.connect(buffer.out, buffer.inn, upper.a, output)
    c.connect(upper.b, lower.a, bootstrap.inp, name="fraction")
    c.connect(bootstrap.out, bootstrap.inn, c3.b, r3.b, name="leg")
    c.connect(lower.b, buffer.ground, bootstrap.ground, ground)
