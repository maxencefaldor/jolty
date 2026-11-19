# Copyright 2026 Maxence Faldor
"""Wien bridge oscillators, each with its own amplitude control, and their bench."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples"))

from common import j2n5457, opamp

from jolty import Builder, Component, Role, component, devices
from jolty.builder import Connectable, Net, Placed
from jolty.parts import d1n4148
from jolty.units import F, ohm, uA, us
from jolty.waveforms import Pulse

FREQUENCY = 1e3
CAPACITANCE = 10e-9
RESISTANCE = 1 / (2 * math.pi * FREQUENCY * CAPACITANCE)

jfet = j2n5457


def _jfet_wien_stage(
    c: Builder, fixed: float
) -> tuple[Net, Net, Placed, Placed, Placed, Net]:
    """Places the op-amp, Wien bridge, feedback leg, and gain-control JFET."""
    output, ground = c.port("output", Role.OUTPUT), c.port("ground", Role.GROUND)
    amplifier = c.place(opamp(), name="U1")
    plus, minus = c.net("plus"), c.net("minus")
    bridge(c, output, plus, ground)
    feedback = c.place(devices.resistor(20e3 * ohm), name="RF")
    series = c.place(devices.resistor(fixed * ohm), name="RG")
    channel = c.place(jfet(), name="J1")
    drain = c.net("drain")
    c.connect(feedback.b, series.a, amplifier.inn, minus)
    c.connect(series.b, channel.drain, drain)
    c.connect(amplifier.inp, plus)
    return output, ground, amplifier, feedback, channel, drain


def bridge(
    c: Builder, output: Connectable, plus: Connectable, ground: Connectable
) -> None:
    """The Wien network: R and C in series from the output, R and C in parallel."""
    series_r = c.place(devices.resistor(RESISTANCE * ohm), name="R1")
    series_c = c.place(devices.capacitor(CAPACITANCE * F), name="C1")
    shunt_r = c.place(devices.resistor(RESISTANCE * ohm), name="R2")
    shunt_c = c.place(devices.capacitor(CAPACITANCE * F), name="C2")
    kick = Pulse(
        initial=0 * uA,
        pulsed=1 * uA,
        delay=10 * us,
        rise=1 * us,
        fall=1 * us,
        width=10 * us,
    )
    start = c.place(devices.current_source(kick), name="I1")
    c.connect(output, series_r.a)
    c.connect(series_r.b, series_c.a, name="arm")
    c.connect(series_c.b, shunt_r.a, shunt_c.a, start.negative, plus)
    c.connect(shunt_r.b, shunt_c.b, start.positive, ground)


@component
def diodes(c: Builder, fixed: float = 12e3, limited: float = 10e3) -> None:
    """Gain 1 + (fixed + limited)/10 kΩ, two diodes across `limited` holding it."""
    output, ground = c.port("output", Role.OUTPUT), c.port("ground", Role.GROUND)
    amplifier = c.place(opamp(), name="U1")
    plus, minus = c.net("plus"), c.net("minus")
    bridge(c, output, plus, ground)
    feedback = c.place(devices.resistor(fixed * ohm), name="R3")
    across = c.place(devices.resistor(limited * ohm), name="R4")
    gain = c.place(devices.resistor(10e3 * ohm), name="R5")
    forward, backward = (
        c.place(d1n4148(), name="D1"),
        c.place(d1n4148(), name="D2"),
    )
    c.connect(amplifier.out, feedback.a, output)
    c.connect(feedback.b, across.a, forward.anode, backward.cathode, name="tap")
    c.connect(across.b, forward.cathode, backward.anode, gain.a, amplifier.inn, minus)
    c.connect(amplifier.inp, plus)
    c.connect(gain.b, amplifier.ground, ground)


@component
def agc(
    c: Builder,
    fixed: float = 9.5e3,
    divider: float = 0.5,
    *,
    linearized: bool = True,
) -> None:
    """Gain 1 + 20 kΩ/(fixed + the JFET's channel), its gate on the output's peak.

    A negative peak detector of the output holds the JFET's gate, so a larger swing
    pinches the channel and lowers the gain. When `linearized`, two 100 kΩ resistors
    from the drain feed half the channel's voltage to its gate, which cancels the square
    law's even distortion.
    """
    output, ground, amplifier, feedback, channel, drain = _jfet_wien_stage(c, fixed)
    detector = c.place(d1n4148(), name="D1")
    hold = c.place(devices.capacitor(1e-6 * F), name="CH")
    upper = c.place(devices.resistor(1e6 * (1 - divider) / divider * ohm), name="RD1")
    lower = c.place(devices.resistor(1e6 * ohm), name="RD2")
    peak, control = c.net("peak"), c.net("control")
    c.connect(amplifier.out, feedback.a, detector.cathode, output)
    c.connect(detector.anode, hold.a, upper.a, peak)
    c.connect(upper.b, lower.a, control)
    if linearized:
        top = c.place(devices.resistor(100e3 * ohm), name="RL1")
        bottom = c.place(devices.resistor(100e3 * ohm), name="RL2")
        c.connect(top.a, drain)
        c.connect(top.b, bottom.a, channel.gate, name="gate")
        c.connect(bottom.b, control)
    else:
        c.connect(channel.gate, control)
    c.connect(channel.source, hold.b, lower.b, amplifier.ground, ground)


@component
def bench(c: Builder, oscillator: Component) -> None:
    """The oscillator alone."""
    ground = c.port("ground", Role.GROUND)
    placed = c.place(oscillator, name="X1")
    c.connect(placed.output, name="out")
    c.connect(placed.ground, ground)


@component
def sensed_agc(
    c: Builder, fixed: float = 9.5e3, sense: float = 0.45, hold: float = 0.22e-6
) -> None:
    """The JFET's gate held at the negative peak of a fraction `sense` of the output.

    Dividing the output before the detector, rather than after it, scales the JFET's
    threshold and the diode's drop alike, so their drifts, -2.5 mV/K and -2 mV/K,
    nearly cancel in the amplitude. The hold capacitor's 1 MΩ discharge sets how fast
    the loop settles, and how much ripple reaches the gate.
    """
    output, ground, amplifier, feedback, channel, _ = _jfet_wien_stage(c, fixed)
    upper = c.place(devices.resistor(10e3 * (1 - sense) / sense * ohm), name="RS1")
    lower = c.place(devices.resistor(10e3 * ohm), name="RS2")
    detector = c.place(d1n4148(), name="D1")
    capacitor = c.place(devices.capacitor(hold * F), name="CH")
    leak = c.place(devices.resistor(1e6 * ohm), name="RH")
    c.connect(amplifier.out, feedback.a, upper.a, output)
    c.connect(upper.b, lower.a, detector.cathode, name="sensed")
    c.connect(detector.anode, capacitor.a, leak.a, channel.gate, name="gate")
    c.connect(channel.source, lower.b, capacitor.b, leak.b, amplifier.ground, ground)
