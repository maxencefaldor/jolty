# Copyright 2026 Maxence Faldor
"""VCO candidates exposing ``control``, ``output``, and ``ground`` ports."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples"))

from common import mos, opamp

from jolty import Builder, Component, Role, component, devices
from jolty.builder import Net, Placed
from jolty.parts import d1n4148, q2n3904
from jolty.units import F, V, ohm, uA, us
from jolty.waveforms import Pulse


def _integrator_schmitt_core(
    c: Builder,
    timing: float,
    capacitance: float,
    threshold: float,
) -> tuple[Net, Net, Placed, Placed, Placed, Placed, Placed, Placed]:
    """Places the shared integrator, Schmitt trigger, timing network, and start kick."""
    control, output = c.port("control", Role.INPUT), c.port("output", Role.OUTPUT)
    ground = c.port("ground", Role.GROUND)
    integrator, schmitt = c.place(opamp(), name="U1"), c.place(opamp(), name="U2")
    r1 = c.place(devices.resistor(timing * ohm), name="R1")
    r2 = c.place(devices.resistor(50e3 * ohm), name="R2")
    r3 = c.place(devices.resistor(50e3 * ohm), name="R3")
    r4 = c.place(devices.resistor(timing / 2 * ohm), name="R4")
    cap = c.place(devices.capacitor(capacitance * F), name="C1")
    # R5 connects the Schmitt non-inverting input to ground; R6 feeds back from output.
    rin = c.place(devices.resistor(10e3 * threshold / (1 - threshold) * ohm), name="R5")
    rf = c.place(devices.resistor(10e3 * ohm), name="R6")
    kick = Pulse(
        initial=0 * uA,
        pulsed=10 * uA,
        delay=1 * us,
        rise=0.1 * us,
        fall=0.1 * us,
        width=2 * us,
    )
    start = c.place(devices.current_source(kick), name="I1")
    c.connect(control, r1.a, r2.a)
    c.connect(r2.b, r3.a, integrator.inp, name="half")
    c.connect(r1.b, r4.a, cap.a, integrator.inn, start.negative, name="summing")
    c.connect(r3.b, rin.a, integrator.ground, schmitt.ground, start.positive, ground)
    return output, ground, integrator, schmitt, r4, cap, rin, rf


@component
def integrator_schmitt(
    c: Builder,
    timing: float = 96.2e3,
    capacitance: float = 1e-9,
    threshold: float = 0.1,
) -> None:
    """Builds a BJT-switched integrator and Schmitt trigger VCO.

    The integrator non-inverting input sits at half the control voltage so C1 charges
    at Vc/(2R) through R1.
    When the Schmitt output turns Q1 on, R4 (R/2) to ground sinks Vc/R from the
    summing node and reverses integration current.
    An inverting Schmitt trigger toggles Q1 at ±`threshold` of its ±13 V output,
    giving nominal frequency Vc/(8·R·C·Vth).
    """
    output, ground, integrator, schmitt, r4, cap, rin, rf = _integrator_schmitt_core(
        c, timing, capacitance, threshold
    )
    switch = c.place(q2n3904(), name="Q1")
    base = c.place(devices.resistor(10e3 * ohm), name="RB")
    clamp = c.place(d1n4148(), name="D1")
    c.connect(r4.b, switch.collector, name="switched")
    c.connect(integrator.out, cap.b, schmitt.inn, output)
    c.connect(rin.b, rf.a, schmitt.inp, name="positive")
    c.connect(schmitt.out, rf.b, base.a, name="square")
    c.connect(base.b, switch.base, clamp.cathode, name="base")
    c.connect(switch.emitter, clamp.anode, ground)


@component
def bench(c: Builder, oscillator: Component, volts: float = 1.0) -> None:
    """Builds a test bench driving `oscillator` with a `volts` DC control voltage."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(volts * V), name="VC")
    placed = c.place(oscillator, name="X1")
    c.connect(source.positive, placed.control, name="control")
    c.connect(placed.output, name="out")
    c.connect(source.negative, placed.ground, ground)


@component
def mosfet_switched(  # noqa: PLR0913 - timing, threshold, and two compensation terms
    c: Builder,
    *,
    timing: float = 96.2e3,
    capacitance: float = 1e-9,
    threshold: float = 0.1,
    compensation: float = 0.0,
    speedup: float = 0.0,
) -> None:
    """Builds an integrator and Schmitt trigger VCO switched by an n-channel MOSFET.

    Unlike BJT saturation offset, the MOSFET channel behaves as a pure resistance that
    only scales reversed current.
    Optional `compensation` ohms in series with C1 step the triangle toward the next
    threshold on each reversal, and `speedup` farads across feedback accelerate
    trigger regeneration.
    """
    output, ground, integrator, schmitt, r4, cap, rin, rf = _integrator_schmitt_core(
        c, timing, capacitance, threshold
    )
    switch = c.place(mos("n", width=50), name="M1")
    c.connect(r4.b, switch.drain, name="switched")
    if compensation:
        series = c.place(devices.resistor(compensation * ohm), name="R7")
        c.connect(cap.b, series.a, name="capacitor")
        c.connect(integrator.out, series.b, schmitt.inn, output)
    else:
        c.connect(integrator.out, cap.b, schmitt.inn, output)
    positive, square = [rin.b, rf.a, schmitt.inp], [schmitt.out, rf.b, switch.gate]
    if speedup:
        hurry = c.place(devices.capacitor(speedup * F), name="C2")
        positive.append(hurry.a)
        square.append(hurry.b)
    c.connect(*positive, name="positive")
    c.connect(*square, name="square")
    c.connect(switch.source, switch.bulk, ground)
