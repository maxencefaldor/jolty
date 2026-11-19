# Copyright 2026 Maxence Faldor
"""Voltage-controlled oscillators, each of ports control, output and ground."""

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
    # R5 from the Schmitt trigger's non-inverting input to ground, R6 from its output.
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
    """An integrator whose input current a transistor reverses, and a Schmitt trigger.

    The integrator's non-inverting input sits at half the control voltage, so its
    capacitor charges at Vc/2R through R from the control; when the Schmitt trigger's
    output turns Q1 on, R/2 to ground draws Vc/R from the inverting input, and the
    current reverses. An inverting Schmitt trigger turns Q1 on at the triangle's low
    threshold and off at its high one, `threshold` of its ±13 V output, so the
    frequency is Vc/(8·R·C·Vth).
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
    """The oscillator with a control of `volts`."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(volts * V), name="VC")
    placed = c.place(oscillator, name="X1")
    c.connect(source.positive, placed.control, name="control")
    c.connect(placed.output, name="out")
    c.connect(source.negative, placed.ground, ground)


@component
def mosfet_switched(  # noqa: PLR0913 - the timing, the threshold and two remedies
    c: Builder,
    *,
    timing: float = 96.2e3,
    capacitance: float = 1e-9,
    threshold: float = 0.1,
    compensation: float = 0.0,
    speedup: float = 0.0,
) -> None:
    """The integrator and Schmitt trigger, its current reversed by an n-channel MOSFET.

    A MOSFET's channel is a resistance, which only scales the reversed current, where a
    transistor's saturation voltage offsets it. `compensation` ohms in series with the
    capacitor step the triangle towards the next threshold at each reversal, and
    `speedup` farads across the trigger's feedback resistor hasten its switching; there
    are neither by default.
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
