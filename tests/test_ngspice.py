# Copyright 2026 Maxence Faldor
"""Differential tests: jolty and ngspice simulate the same circuits and must agree.

Each circuit of the corpus is written once with the builder, simulated by jolty, and
written as a SPICE netlist for ngspice, whose results come back as an ASCII raw file.
Both engines run with tight tolerances. Operating points, DC sweeps and AC responses
must agree to the Newton tolerance; transients, interpolated onto ngspice's time points,
to a small fraction of their swing.

The corpus is flat: hierarchy is jolty's own, tested on its own, while these tests check
the devices, the waveforms and the numerics against the reference simulator.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pytest

from differential import (
    SKIP,
    closest,
    compare,
    compared,
    netlist,
    ngspice,
    size,
    vector,
)
from jolty import (
    Ac,
    Builder,
    Component,
    DcSweep,
    Noise,
    OperatingPoint,
    PoleZero,
    Role,
    Simulation,
    Stability,
    TransferFunction,
    Transient,
    _core,
    component,
    devices,
    files,
    results,
    spice,
)
from jolty.plan import TIGHT, decade, linear
from jolty.spice.export import node_voltages
from jolty.units import (
    A,
    F,
    GHz,
    Hz,
    K,
    MHz,
    Mohm,
    Quantity,
    V,
    celsius,
    fA,
    fF,
    kHz,
    kohm,
    m,
    mA,
    mF,
    mH,
    ms,
    mV,
    nA,
    nF,
    nm,
    ns,
    ohm,
    pA,
    pF,
    s,
    uA,
    uC,
    uF,
    uH,
    um,
    uS,
    us,
)
from jolty.waveforms import Exponential, Piecewise, Pulse, Sine

pytestmark = SKIP

# The corpus.


@component
def divider(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(5 * V), name="V1")
    upper = c.place(devices.resistor(1 * kohm), name="R1")
    lower = c.place(devices.resistor(3 * kohm), name="R2")
    c.connect(source.positive, upper.a, name="in")
    c.connect(upper.b, lower.a, name="out")
    c.connect(lower.b, source.negative, ground)


@component
def low_pass(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    step = Piecewise([(1 * us, 0 * V), (1 * us + 10 * ns, 1 * V)])
    source = c.place(devices.voltage_source(step), name="V1")
    resistor = c.place(devices.resistor(1 * kohm), name="R1")
    capacitor = c.place(devices.capacitor(1 * nF), name="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)


@component
def exponential(c: Builder) -> None:
    """An RC driven by SPICE's ``EXP``: a rise at 1 µs and a fall back from 5 µs."""
    ground = c.port("ground", Role.GROUND)
    rise_and_fall = Exponential(
        initial=0.5 * V,
        pulsed=2 * V,
        rise_delay=1 * us,
        rise_time_constant=0.5 * us,
        fall_delay=5 * us,
        fall_time_constant=1 * us,
    )
    source = c.place(devices.voltage_source(rise_and_fall), name="V1")
    resistor = c.place(devices.resistor(1 * kohm), name="R1")
    capacitor = c.place(devices.capacitor(1 * nF), name="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)


@component
def resonator(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    pulse = Pulse(
        initial=0 * V,
        pulsed=1 * V,
        delay=1 * us,
        rise=10 * ns,
        fall=10 * ns,
        width=20 * us,
    )
    source = c.place(devices.voltage_source(pulse), name="V1")
    resistor = c.place(devices.resistor(20 * ohm), name="R1")
    inductor = c.place(devices.inductor(1 * mH), name="L1")
    capacitor = c.place(devices.capacitor(10 * nF), name="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, inductor.a, name="mid")
    c.connect(inductor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)


@component
def triangle(c: Builder) -> None:
    """A pulse with no width, a triangle wave, which ngspice reads its own way."""
    ground = c.port("ground", Role.GROUND)
    pulse = Pulse(
        initial=0 * V,
        pulsed=1 * V,
        rise=10 * us,
        fall=10 * us,
        width=0 * us,
        period=20 * us,
    )
    source = c.place(devices.voltage_source(pulse), name="V1")
    resistor = c.place(devices.resistor(1 * kohm), name="R1")
    capacitor = c.place(devices.capacitor(1 * nF), name="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)


@component
def clipper(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(0 * V), name="V1")
    resistor = c.place(devices.resistor(1 * kohm), name="R1")
    diode = c.place(
        devices.diode(saturation_current=2.52 * nA, emission_coefficient=1.752),
        name="D1",
    )
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, diode.anode, name="out")
    c.connect(diode.cathode, source.negative, ground)


@component
def zener_clamp(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(0 * V), name="V1")
    resistor = c.place(devices.resistor(1 * kohm), name="R1")
    zener = c.place(
        devices.diode(
            saturation_current=2 * nA,
            emission_coefficient=1.5,
            series_resistance=2 * ohm,
            junction_capacitance=20 * pF,
            breakdown_voltage=5.1 * V,
            breakdown_current=5 * mA,
            breakdown_emission_coefficient=1.2,
        ),
        name="D1",
    )
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, zener.anode, name="out")
    c.connect(zener.cathode, source.negative, ground)


@component
def rectifier(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    mains = Sine(offset=0 * V, amplitude=5 * V, frequency=1 * kHz, phase=0.3)
    source = c.place(devices.voltage_source(mains), name="V1")
    diode = c.place(
        devices.diode(
            saturation_current=2.52 * nA,
            emission_coefficient=1.752,
            series_resistance=0.568 * ohm,
            junction_capacitance=4 * pF,
            junction_potential=0.7 * V,
            grading_coefficient=0.4,
            depletion_coefficient=0.5,
            transit_time=20 * ns,
        ),
        name="D1",
    )
    smoothing = c.place(devices.capacitor(1 * uF), name="C1")
    load = c.place(devices.resistor(1 * kohm), name="R1")
    c.connect(source.positive, diode.anode, name="in")
    c.connect(diode.cathode, smoothing.a, load.a, name="out")
    c.connect(smoothing.b, load.b, source.negative, ground)


@component
def amplifier(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(0.1 * V), name="V1")
    gain = c.place(devices.voltage_controlled_voltage_source(10.0), name="E1")
    coupling = c.place(devices.resistor(1 * kohm), name="R1")
    stage = c.place(devices.voltage_controlled_current_source(500 * uS), name="G1")
    load = c.place(devices.resistor(2 * kohm), name="R2")
    bypass = c.place(devices.capacitor(1 * nF), name="C1")
    c.connect(source.positive, gain.control_positive, name="in")
    c.connect(gain.positive, coupling.a, name="gained")
    c.connect(coupling.b, stage.control_positive, name="coupled")
    c.connect(stage.negative, load.a, bypass.a, name="out")
    c.connect(
        source.negative,
        gain.negative,
        gain.control_negative,
        stage.positive,
        stage.control_negative,
        load.b,
        bypass.b,
        ground,
    )


@component
def driven_tank(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    drive = Pulse(
        initial=0 * mA,
        pulsed=1 * mA,
        delay=0 * us,
        rise=1 * us,
        fall=1 * us,
        width=3 * us,
        period=10 * us,
    )
    source = c.place(devices.current_source(drive), name="I1")
    inductor = c.place(devices.inductor(10 * mH), name="L1")
    damping = c.place(devices.resistor(5 * kohm), name="R1")
    capacitor = c.place(devices.capacitor(1 * nF), name="C1")
    c.connect(source.negative, inductor.a, damping.a, capacitor.a, name="out")
    c.connect(source.positive, inductor.b, damping.b, capacitor.b, ground)


@component
def transformer(c: Builder) -> None:
    """A pulse into three coupled windings, the third wound backward, each loaded."""
    ground = c.port("ground", Role.GROUND)
    drive = Pulse(
        initial=0 * V,
        pulsed=1 * V,
        delay=1 * us,
        rise=1 * us,
        fall=1 * us,
        width=20 * us,
        period=50 * us,
    )
    source = c.place(devices.voltage_source(drive), name="V1")
    feed = c.place(devices.resistor(50 * ohm), name="R1")
    windings = c.place(
        devices.coupled_inductors(
            [
                devices.inductor(1 * mH),
                devices.inductor(4 * mH),
                devices.inductor(0.25 * mH),
            ],
            {(1, 2): 0.95, (1, 3): 0.9, (2, 3): 0.85},
        ),
        name="K1",
    )
    load = c.place(devices.resistor(1 * kohm), name="RL")
    hold = c.place(devices.capacitor(10 * nF), name="C1")
    tertiary = c.place(devices.resistor(100 * ohm), name="RT")
    c.connect(source.positive, feed.a, name="in")
    c.connect(feed.b, windings.a1, name="primary")
    c.connect(windings.a2, load.a, hold.a, name="out")
    c.connect(windings.b3, tertiary.a, name="tertiary")
    c.connect(
        source.negative,
        windings.b1,
        windings.b2,
        windings.a3,
        load.b,
        hold.b,
        tertiary.b,
        ground,
    )


_JFET = devices.jfet(
    threshold_voltage=-1.8 * V,
    transconductance_parameter=1.3 * mA / V**2,
    channel_length_modulation=0.02 / V,
    drain_resistance=20 * ohm,
    source_resistance=10 * ohm,
    gate_source_capacitance=4 * pF,
    gate_drain_capacitance=2 * pF,
    junction_potential=0.8 * V,
    saturation_current=1e-14 * A,
    doping_tail=0.85,
)


@component
def jfet_stage(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(12 * V), name="V2")
    # Starting at its peak, the sine's slope is zero, so the operating point at zero
    # time is also the transient's start, with no capacitive current to jump.
    signal = Sine(
        offset=-0.8 * V, amplitude=0.6 * V, frequency=200 * kHz, phase=math.pi / 2
    )
    source = c.place(devices.voltage_source(signal), name="V1")
    gate = c.place(devices.resistor(1 * kohm), name="RG")
    transistor = c.place(_JFET, name="J1")
    drain = c.place(devices.resistor(2.2 * kohm), name="RD")
    degeneration = c.place(devices.resistor(470 * ohm), name="RS")
    c.connect(source.positive, gate.a, name="in")
    c.connect(gate.b, transistor.gate, name="gate")
    c.connect(supply.positive, drain.a, name="supply")
    c.connect(drain.b, transistor.drain, name="out")
    c.connect(transistor.source, degeneration.a, name="tail")
    c.connect(source.negative, supply.negative, degeneration.b, ground)


@component
def jfet_curves(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    bias = c.place(devices.voltage_source(-0.5 * V), name="V1")
    drain = c.place(devices.voltage_source(0 * V), name="V2")
    transistor = c.place(_JFET, name="J1")
    c.connect(bias.positive, transistor.gate, name="gate")
    c.connect(drain.positive, transistor.drain, name="drain")
    c.connect(bias.negative, drain.negative, transistor.source, ground)


@component
def p_jfet_follower(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(-9 * V), name="V2")
    source = c.place(devices.voltage_source(-1 * V), name="V1")
    transistor = c.place(
        devices.jfet(
            channel="p",
            threshold_voltage=-2.2 * V,
            transconductance_parameter=0.9 * mA / V**2,
            channel_length_modulation=0.01 / V,
            gate_source_capacitance=3 * pF,
            gate_drain_capacitance=1 * pF,
        ),
        name="J1",
    )
    load = c.place(devices.resistor(3.3 * kohm), name="RL")
    c.connect(source.positive, transistor.gate, name="in")
    c.connect(supply.positive, transistor.drain, name="supply")
    c.connect(transistor.source, load.a, name="out")
    c.connect(source.negative, supply.negative, load.b, ground)


def _bipolar(
    polarity: Literal["npn", "pnp"],
    base_resistance: Literal["current", "charge", "fixed"],
    substrate: Literal["vertical", "lateral"] | None = None,
    excess_phase: float | None = None,
) -> devices.Device:
    """A small-signal transistor with every effect of the model switched on.

    With a `substrate`, so are its substrate junction's, and with an `excess_phase`,
    in degrees, its delay. Its base resistance falls with
    the base current, with the base charge, or stays fixed. ngspice leaves a falling
    base resistance out of its small-signal model, where jolty keeps it, so a frequency
    response compares transistors with a fixed one.
    """
    junction = substrate is not None
    return devices.bipolar_transistor(
        polarity=polarity,
        saturation_current=14 * fA,
        forward_beta=150.0,
        reverse_beta=2.0,
        forward_emission_coefficient=1.01,
        forward_early_voltage=80 * V,
        reverse_early_voltage=15 * V,
        forward_knee_current=40 * mA,
        reverse_knee_current=5 * mA,
        base_emitter_leakage_current=50 * fA,
        base_emitter_leakage_emission_coefficient=1.6,
        base_collector_leakage_current=100 * fA,
        base_resistance=50 * ohm,
        min_base_resistance=None if base_resistance == "fixed" else 10 * ohm,
        base_resistance_current=1 * mA if base_resistance == "current" else None,
        emitter_resistance=1 * ohm,
        collector_resistance=5 * ohm,
        base_emitter_capacitance=8 * pF,
        base_emitter_potential=0.75 * V,
        base_emitter_grading=0.33,
        forward_transit_time=0.4 * ns,
        transit_time_bias_coefficient=3.0,
        transit_time_voltage=4 * V,
        transit_time_current=0.6 * A,
        base_collector_capacitance=3.5 * pF,
        base_collector_potential=0.75 * V,
        base_collector_grading=0.4,
        internal_base_fraction=0.6,
        reverse_transit_time=50 * ns,
        excess_phase=excess_phase,
        substrate=substrate,
        substrate_saturation_current=0.2 * fA if junction else None,
        substrate_emission_coefficient=1.05 if junction else None,
        substrate_capacitance=2 * pF if junction else None,
        substrate_potential=0.65 * V if junction else None,
        substrate_grading=0.45 if junction else None,
    )


@component
def bjt_stage(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(5 * V), name="V2")
    source = c.place(devices.voltage_source(0.7 * V), name="V1")
    base = c.place(devices.resistor(10 * kohm), name="RB")
    transistor = c.place(_bipolar("npn", "current"), name="Q1")
    load = c.place(devices.resistor(1 * kohm), name="RC")
    c.connect(source.positive, base.a, name="in")
    c.connect(base.b, transistor.base, name="base")
    c.connect(supply.positive, load.a, name="supply")
    c.connect(load.b, transistor.collector, name="out")
    c.connect(source.negative, supply.negative, transistor.emitter, ground)


@component
def bjt_amplifier(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(12 * V), name="V2")
    signal = Sine(offset=0 * V, amplitude=20 * mV, frequency=1 * MHz, phase=math.pi / 2)
    source = c.place(devices.voltage_source(signal), name="V1")
    coupling = c.place(devices.capacitor(100 * nF), name="CIN")
    upper = c.place(devices.resistor(47 * kohm), name="R1")
    lower = c.place(devices.resistor(10 * kohm), name="R2")
    transistor = c.place(_bipolar("npn", "fixed"), name="Q1")
    load = c.place(devices.resistor(3.3 * kohm), name="RC")
    degeneration = c.place(devices.resistor(1 * kohm), name="RE")
    bypass = c.place(devices.capacitor(1 * nF), name="CE")
    c.connect(source.positive, coupling.a, name="in")
    c.connect(coupling.b, upper.b, lower.a, transistor.base, name="base")
    c.connect(supply.positive, upper.a, load.a, name="supply")
    c.connect(load.b, transistor.collector, name="out")
    c.connect(transistor.emitter, degeneration.a, bypass.a, name="tail")
    c.connect(
        source.negative, supply.negative, lower.b, degeneration.b, bypass.b, ground
    )


@component
def delayed_stage(c: Builder) -> None:
    """A common-emitter stage whose transistor delays its transport current.

    An excess phase of 60° on a transit time of 0.4 ns delays it by 0.42 ns, through
    the Bessel filter ngspice integrates in a transient, and edges of 1 ns show it.
    """
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(5 * V), name="V2")
    edges = Pulse(
        initial=0.68 * V,
        pulsed=0.74 * V,
        delay=2 * ns,
        rise=1 * ns,
        fall=1 * ns,
        width=5 * ns,
        period=12 * ns,
    )
    source = c.place(devices.voltage_source(edges), name="V1")
    transistor = c.place(_bipolar("npn", "fixed", excess_phase=60.0), name="Q1")
    load = c.place(devices.resistor(1 * kohm), name="RC")
    c.connect(source.positive, transistor.base, name="base")
    c.connect(supply.positive, load.a, name="supply")
    c.connect(load.b, transistor.collector, name="out")
    c.connect(source.negative, supply.negative, transistor.emitter, ground)


@component
def feedback_pair(c: Builder) -> None:
    """A common-emitter stage and a follower in a loop broken at the probe ``VP``.

    The follower feeds back through a resistor with a capacitor across it. The source
    ``IP`` drives nothing until an AC analysis excites it, injecting a current from the
    ground where the loop enters the feedback network.
    """
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(12 * V), name="V1")
    stage = c.place(_bipolar("npn", "fixed"), name="Q1")
    follower = c.place(_bipolar("npn", "fixed"), name="Q2")
    collector = c.place(devices.resistor(10 * kohm), name="RC")
    emitter = c.place(devices.resistor(470 * ohm), name="RE")
    load = c.place(devices.resistor(2.2 * kohm), name="RL")
    lower = c.place(devices.resistor(10 * kohm), name="R2")
    feedback = c.place(devices.resistor(47 * kohm), name="RF")
    across = c.place(devices.capacitor(2 * pF), name="CF")
    probe = c.place(devices.voltage_source(0 * V), name="VP")
    injection = c.place(devices.current_source(0 * A), name="IP")
    c.connect(supply.positive, collector.a, follower.collector, name="supply")
    c.connect(collector.b, stage.collector, follower.base, name="drive")
    c.connect(follower.emitter, load.a, probe.negative, name="out")
    c.connect(probe.positive, feedback.a, across.a, injection.negative, name="fb")
    c.connect(feedback.b, across.b, lower.a, stage.base, name="base")
    c.connect(stage.emitter, emitter.a, name="tail")
    c.connect(supply.negative, emitter.b, load.b, lower.b, injection.positive, ground)


@component
def integrated_stage(c: Builder) -> None:
    """A vertical NPN loaded by a lateral PNP, on a substrate fed through a resistor.

    The source feeding the substrate drives the transistors' junctions to it forward.
    """
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(5 * V), name="V2")
    signal = Sine(
        offset=0.7 * V, amplitude=10 * mV, frequency=1 * MHz, phase=math.pi / 2
    )
    source = c.place(devices.voltage_source(signal), name="V1")
    bias = c.place(devices.voltage_source(-1 * V), name="V3")
    base = c.place(devices.resistor(10 * kohm), name="RB")
    amplifier = c.place(_bipolar("npn", "fixed", "vertical"), name="Q1")
    load = c.place(_bipolar("pnp", "fixed", "lateral"), name="Q2")
    feed = c.place(devices.resistor(10 * kohm), name="RS")
    c.connect(source.positive, base.a, name="in")
    c.connect(base.b, amplifier.base, name="base")
    c.connect(supply.positive, load.emitter, name="supply")
    c.connect(amplifier.collector, load.collector, load.base, name="out")
    c.connect(bias.positive, feed.a, name="bias")
    c.connect(feed.b, amplifier.substrate, load.substrate, name="substrate")
    c.connect(
        source.negative, supply.negative, bias.negative, amplifier.emitter, ground
    )


@component
def pnp_mirror(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(10 * V), name="V1")
    reference = c.place(devices.resistor(10 * kohm), name="RR")
    left = c.place(_bipolar("pnp", "charge"), name="Q1")
    right = c.place(_bipolar("pnp", "charge"), name="Q2")
    load = c.place(devices.resistor(4.7 * kohm), name="RL")
    c.connect(supply.positive, left.emitter, right.emitter, name="supply")
    c.connect(left.base, left.collector, right.base, reference.a, name="mirror")
    c.connect(right.collector, load.a, name="out")
    c.connect(supply.negative, reference.b, load.b, ground)


@component
def current_controlled(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(2 * V), name="V1")
    feed = c.place(devices.resistor(1 * kohm), name="R1")
    amplifier = c.place(devices.current_controlled_current_source(3.0), name="F1")
    load = c.place(devices.resistor(2 * kohm), name="R2")
    transducer = c.place(
        devices.current_controlled_voltage_source(500 * ohm), name="H1"
    )
    output = c.place(devices.resistor(4.7 * kohm), name="R3")
    c.connect(source.positive, feed.a, name="in")
    c.connect(feed.b, amplifier.control_positive, name="sensed")
    c.connect(amplifier.negative, transducer.control_positive, name="mid")
    c.connect(transducer.control_negative, load.a, name="loaded")
    c.connect(transducer.positive, output.a, name="out")
    c.connect(
        source.negative,
        amplifier.control_negative,
        amplifier.positive,
        load.b,
        transducer.negative,
        output.b,
        ground,
    )


@component
def negative_load(c: Builder) -> None:
    """A transistor saturated into a negative resistance at its emitter.

    A transconductance from its emitter to itself pushes 97 µA per volt into it, and
    another feeds back its collector's voltage; a diode keeps the emitter above the
    ground. Newton iteration and gmin stepping from every node fail, in ngspice as in
    jolty, and gmin stepping across the junctions finds it.
    """
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(4.8 * V), name="V1")
    bias = c.place(devices.resistor(230 * kohm), name="R1")
    sink = c.place(devices.current_source(0.2 * mA), name="I1")
    feedback = c.place(devices.voltage_controlled_current_source(1.75 * uS), name="G1")
    clamp = c.place(devices.diode(), name="D1")
    transistor = c.place(devices.bipolar_transistor(polarity="pnp"), name="Q1")
    negative = c.place(devices.voltage_controlled_current_source(97 * uS), name="G2")
    c.connect(supply.positive, bias.b, transistor.base, name="base")
    c.connect(bias.a, sink.positive, feedback.control_negative, name="low")
    c.connect(transistor.collector, feedback.control_positive, name="collector")
    c.connect(
        transistor.emitter,
        feedback.negative,
        clamp.cathode,
        negative.negative,
        negative.control_positive,
        name="emitter",
    )
    c.connect(
        supply.negative,
        sink.negative,
        feedback.positive,
        clamp.anode,
        negative.positive,
        negative.control_negative,
        ground,
    )


_NMOS = devices.mosfet(
    width=100 * um,
    length=2 * um,
    lateral_diffusion=0.2 * um,
    threshold_voltage=0.7 * V,
    transconductance_parameter=110 * uA / V**2,
    body_effect=0.45,
    surface_potential=0.7 * V,
    channel_length_modulation=0.03 / V,
    oxide_thickness=20 * nm,
    drain_resistance=15 * ohm,
    source_resistance=10 * ohm,
    saturation_current=10 * fA,
    drain_area=200 * um**2,
    source_area=150 * um**2,
    drain_perimeter=104 * um,
    source_perimeter=103 * um,
    bottom_capacitance=3e-4 * F / m**2,
    sidewall_capacitance=5e-10 * F / m,
    bottom_grading=0.45,
    sidewall_grading=0.3,
    gate_source_overlap=3e-10 * F / m,
    gate_drain_overlap=3e-10 * F / m,
    gate_bulk_overlap=2e-10 * F / m,
)


def _level_three(
    max_drift_velocity: Quantity | None,
    saturation_field_factor: float = 0.4,
    lateral_diffusion: Quantity = 0.2 * um,
) -> devices.Device:
    """A level 3 n-channel with every effect of its law, and velocity saturation."""
    return devices.mosfet(
        level=3,
        width=50 * um,
        length=2 * um,
        lateral_diffusion=lateral_diffusion,
        width_narrowing=0.1 * um,
        length_adjustment=0.05 * um,
        width_adjustment=-0.05 * um,
        threshold_voltage=0.7 * V,
        threshold_adjustment=0.02 * V,
        transconductance_parameter=100 * uA / V**2,
        body_effect=0.5,
        surface_potential=0.7 * V,
        oxide_thickness=20 * nm,
        surface_mobility=0.05 * m**2 / (V * s),
        substrate_doping=1e22 / m**3,
        junction_depth=0.3 * um,
        static_feedback=0.3,
        narrow_width_factor=0.5,
        mobility_modulation=0.08 / V,
        saturation_field_factor=saturation_field_factor,
        max_drift_velocity=max_drift_velocity,
        fast_surface_state_density=5e15 / m**2,
        drain_resistance=15 * ohm,
        source_resistance=10 * ohm,
        drain_area=100 * um**2,
        source_area=80 * um**2,
        drain_perimeter=54 * um,
        source_perimeter=53 * um,
        bottom_capacitance=3e-4 * F / m**2,
        sidewall_capacitance=5e-10 * F / m,
        gate_source_overlap=3e-10 * F / m,
        gate_drain_overlap=3e-10 * F / m,
        gate_bulk_overlap=2e-10 * F / m,
    )


@component
def level_three_stage(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(5 * V), name="V2")
    signal = Sine(
        offset=1.4 * V, amplitude=0.4 * V, frequency=2 * MHz, phase=math.pi / 2
    )
    source = c.place(devices.voltage_source(signal), name="V1")
    gate = c.place(devices.resistor(100 * kohm), name="RG")
    transistor = c.place(_level_three(1.5e5 * m / s), name="M1")
    drain = c.place(devices.resistor(4.7 * kohm), name="RD")
    c.connect(source.positive, gate.a, name="in")
    c.connect(gate.b, transistor.gate, name="gate")
    c.connect(supply.positive, drain.a, name="supply")
    c.connect(drain.b, transistor.drain, name="out")
    c.connect(
        source.negative, supply.negative, transistor.source, transistor.bulk, ground
    )


@component
def level_three_curves(
    c: Builder,
    max_drift_velocity: Quantity | None = None,
    saturation_field_factor: float = 0.4,
    lateral_diffusion: Quantity = 0.2 * um,
) -> None:
    ground = c.port("ground", Role.GROUND)
    bias = c.place(devices.voltage_source(1.5 * V), name="V1")
    drain = c.place(devices.voltage_source(0 * V), name="V2")
    body = c.place(devices.voltage_source(-1 * V), name="V3")
    transistor = c.place(
        _level_three(max_drift_velocity, saturation_field_factor, lateral_diffusion),
        name="M1",
    )
    c.connect(bias.positive, transistor.gate, name="gate")
    c.connect(drain.positive, transistor.drain, name="drain")
    c.connect(body.positive, transistor.bulk, name="body")
    c.connect(bias.negative, drain.negative, body.negative, transistor.source, ground)


@component
def drifting(c: Builder, *, apart: bool = False) -> None:
    """Every temperature coefficient jolty models, on every kind of device.

    A zener reference biases a bipolar stage and a MOSFET, beside a JFET current source.
    `apart`, the drifting devices run at their own temperatures, held or offset.
    """

    def held(device: devices.Device, temperature: Quantity) -> devices.Device:
        return device.held_at(temperature) if apart else device

    def offset(device: devices.Device, kelvins: float) -> devices.Device:
        return device.warmer_by(kelvins * K) if apart else device

    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(12 * V), name="V1")
    feed = c.place(
        held(
            devices.resistor(
                2 * kohm,
                temperature_coefficient=3e-3 / K,
                quadratic_temperature_coefficient=1e-5 / K**2,
            ),
            celsius(85),
        ),
        name="R1",
    )
    zener = c.place(
        offset(
            devices.diode(
                saturation_current=1 * nA,
                emission_coefficient=1.2,
                junction_capacitance=10 * pF,
                breakdown_voltage=5.1 * V,
                breakdown_current=5 * mA,
                breakdown_temperature_coefficient=2 * mV / K,
                energy_gap=1.2 * V,
                saturation_current_temperature_exponent=2.5,
                nominal_temperature=celsius(50),
            ),
            30,
        ),
        name="D1",
    )
    hold = c.place(
        offset(devices.capacitor(100 * pF, temperature_coefficient=-1e-3 / K), -20),
        name="C1",
    )
    base = c.place(devices.resistor(47 * kohm), name="RB")
    bipolar = c.place(
        held(
            devices.bipolar_transistor(
                saturation_current=10 * fA,
                forward_beta=150,
                reverse_beta=2,
                base_emitter_leakage_current=10 * fA,
                base_collector_leakage_current=5 * fA,
                base_emitter_capacitance=2 * pF,
                base_collector_capacitance=1 * pF,
                substrate="vertical",
                substrate_saturation_current=1 * fA,
                substrate_capacitance=1 * pF,
                energy_gap=1.15 * V,
                saturation_current_temperature_exponent=3.5,
                beta_temperature_exponent=1.5,
                nominal_temperature=celsius(35),
            ),
            celsius(-10),
        ),
        name="Q1",
    )
    collector = c.place(
        offset(
            devices.resistor(4.7 * kohm, exponential_temperature_coefficient=0.3), 15
        ),
        name="RC",
    )
    emitter = c.place(devices.resistor(1 * kohm), name="RE")
    unipolar = c.place(
        held(
            devices.jfet(
                threshold_voltage=-2 * V,
                transconductance_parameter=1 * mA / V**2,
                saturation_current=1 * fA,
                gate_source_capacitance=2 * pF,
                gate_drain_capacitance=1 * pF,
                threshold_voltage_temperature_coefficient=-2 * mV / K,
                transconductance_parameter_temperature_exponent=-1.5,
                energy_gap=1.1 * V,
                saturation_current_temperature_exponent=2,
            ),
            celsius(60),
        ),
        name="J1",
    )
    sense = c.place(devices.resistor(1 * kohm), name="RS")
    channel = c.place(
        offset(
            devices.mosfet(
                width=20 * um,
                length=2 * um,
                oxide_thickness=40 * nm,
                substrate_doping=1e22 / m**3,
                bottom_capacitance=0.3 * mF / m**2,
                drain_area=40 * um**2,
                source_area=40 * um**2,
                nominal_temperature=celsius(40),
            ),
            25,
        ),
        name="M1",
    )
    drain = c.place(devices.resistor(10 * kohm), name="RD")
    c.connect(
        supply.positive, feed.a, collector.a, unipolar.drain, drain.a, name="supply"
    )
    c.connect(feed.b, zener.cathode, hold.a, base.a, channel.gate, name="reference")
    c.connect(base.b, bipolar.base, name="base")
    c.connect(collector.b, bipolar.collector, name="out")
    c.connect(bipolar.emitter, emitter.a, name="emitter")
    c.connect(unipolar.source, sense.a, name="sink")
    c.connect(drain.b, channel.drain, name="channel")
    c.connect(
        supply.negative,
        zener.anode,
        hold.b,
        emitter.b,
        bipolar.substrate,
        unipolar.gate,
        sense.b,
        channel.source,
        channel.bulk,
        ground,
    )


def drifting_apart() -> Component:
    """`drifting`, each of its drifting devices at a temperature of its own."""
    return drifting(apart=True)


@component
def nmos_stage(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(9 * V), name="V2")
    signal = Sine(
        offset=1.6 * V, amplitude=0.3 * V, frequency=2 * MHz, phase=math.pi / 2
    )
    source = c.place(devices.voltage_source(signal), name="V1")
    gate = c.place(devices.resistor(100 * kohm), name="RG")
    transistor = c.place(_NMOS, name="M1")
    drain = c.place(devices.resistor(4.7 * kohm), name="RD")
    degeneration = c.place(devices.resistor(1 * kohm), name="RS")
    c.connect(source.positive, gate.a, name="in")
    c.connect(gate.b, transistor.gate, name="gate")
    c.connect(supply.positive, drain.a, name="supply")
    c.connect(drain.b, transistor.drain, name="out")
    c.connect(transistor.source, degeneration.a, name="tail")
    c.connect(source.negative, supply.negative, degeneration.b, transistor.bulk, ground)


@component
def mosfet_curves(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    bias = c.place(devices.voltage_source(2 * V), name="V1")
    drain = c.place(devices.voltage_source(0 * V), name="V2")
    body = c.place(devices.voltage_source(-3 * V), name="V3")
    transistor = c.place(_NMOS, name="M1")
    c.connect(bias.positive, transistor.gate, name="gate")
    c.connect(drain.positive, transistor.drain, name="drain")
    c.connect(body.positive, transistor.bulk, name="body")
    c.connect(bias.negative, drain.negative, body.negative, transistor.source, ground)


@component
def cmos_inverter(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(3.3 * V), name="V2")
    pulse = Pulse(
        initial=0 * V,
        pulsed=3.3 * V,
        delay=10 * ns,
        rise=10 * ns,
        fall=10 * ns,
        width=60 * ns,
        period=160 * ns,
    )
    source = c.place(devices.voltage_source(pulse), name="V1")
    # A driver's resistance, so the gates' charging current flows without jumps.
    driver = c.place(devices.resistor(10 * kohm), name="RIN")
    lower = c.place(
        devices.mosfet(
            width=10 * um,
            length=1 * um,
            oxide_thickness=15 * nm,
            substrate_doping=5e22 / m**3,
            surface_mobility=0.045 * m**2 / (V * s),
            surface_state_density=1e14 / m**2,
            channel_length_modulation=0.05 / V,
            saturation_current_density=1e-4 * A / m**2,
            drain_area=30 * um**2,
            source_area=30 * um**2,
            drain_perimeter=26 * um,
            source_perimeter=26 * um,
            bottom_capacitance=4e-4 * F / m**2,
            sidewall_capacitance=3e-10 * F / m,
            gate_source_overlap=2e-10 * F / m,
            gate_drain_overlap=2e-10 * F / m,
        ),
        name="M1",
    )
    upper = c.place(
        devices.mosfet(
            channel="p",
            width=12 * um,
            length=1 * um,
            multiplier=2.0,
            oxide_thickness=15 * nm,
            substrate_doping=5e22 / m**3,
            surface_mobility=0.015 * m**2 / (V * s),
            gate_material="same_doping",
            channel_length_modulation=0.08 / V,
            sheet_resistance=40 * ohm,
            drain_squares=0.5,
            source_squares=0.25,
            bulk_drain_capacitance=20 * fF,
            bulk_source_capacitance=15 * fF,
            gate_source_overlap=2e-10 * F / m,
            gate_drain_overlap=2e-10 * F / m,
            gate_bulk_overlap=1e-10 * F / m,
        ),
        name="M2",
    )
    load = c.place(devices.capacitor(200 * fF), name="CL")
    c.connect(source.positive, driver.a, name="in")
    c.connect(driver.b, lower.gate, upper.gate, name="gate")
    c.connect(supply.positive, upper.source, upper.bulk, name="supply")
    c.connect(lower.drain, upper.drain, load.a, name="out")
    c.connect(
        source.negative, supply.negative, lower.source, lower.bulk, load.b, ground
    )


@component
def behavioral(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    signal = Sine(
        offset=0.2 * V, amplitude=1 * V, frequency=10 * kHz, phase=math.pi / 2
    )
    source = c.place(devices.voltage_source(signal), name="V1")
    upper = c.place(devices.resistor(1 * kohm), name="R1")
    lower = c.place(devices.resistor(1 * kohm), name="R2")
    shaper = c.place(
        devices.behavioral_voltage_source(
            "2 * tanh(v(x) / 0.5) + 0.1 * v(x) ^ 2 + 0.05 * sin(2 * pi * 1e4 * time)"
        ),
        name="B1",
    )
    series = c.place(devices.resistor(1 * kohm), name="R3")
    load = c.place(
        devices.behavioral_current_source("v(y) > 0 ? 2e-3 * v(y) : 1e-4 * v(y)"),
        name="B2",
    )
    smoothing = c.place(devices.capacitor(100 * nF), name="C1")
    sink = c.place(devices.resistor(2 * kohm), name="R4")
    mirror = c.place(
        devices.behavioral_current_source("0.5 * i(s) + v(y) * i(s)"), name="B3"
    )
    output = c.place(devices.resistor(1 * kohm), name="R5")
    c.connect(source.positive, upper.a, name="in")
    c.connect(upper.b, lower.a, shaper.x, name="x")
    c.connect(shaper.positive, series.a, name="shaped")
    c.connect(
        series.b,
        load.positive,
        load.y,
        smoothing.a,
        mirror.s_positive,
        mirror.y,
        name="y",
    )
    c.connect(mirror.s_negative, sink.a, name="w")
    c.connect(mirror.negative, output.a, name="z")
    c.connect(
        source.negative,
        lower.b,
        shaper.negative,
        load.negative,
        smoothing.b,
        sink.b,
        mirror.positive,
        output.b,
        ground,
    )


@component
def curved(c: Builder) -> None:
    """Resistors and a capacitor whose values change with their voltages, either way."""
    ground = c.port("ground", Role.GROUND)
    sine = Sine(offset=0 * V, amplitude=2 * V, frequency=1 * kHz)
    source = c.place(devices.voltage_source(sine), name="V1")
    feed = c.place(
        devices.resistor(
            1 * kohm,
            temperature_coefficient=1e-3 / K,
            voltage_coefficient=0.05 / V,
            quadratic_voltage_coefficient=0.02 / V**2,
        ),
        name="R1",
    )
    load = c.place(devices.resistor(2 * kohm, voltage_coefficient=-0.03 / V), name="R2")
    hold = c.place(
        devices.capacitor(
            1 * uF,
            voltage_coefficient=0.1 / V,
            quadratic_voltage_coefficient=0.05 / V**2,
        ),
        name="C1",
    )
    c.connect(source.positive, feed.a, name="in")
    c.connect(feed.b, load.a, hold.a, name="out")
    c.connect(source.negative, load.b, hold.b, ground)


@component
def lit(c: Builder) -> None:
    """A photocell under a divider, its light a pulse, its response lagging it."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(5 * V), name="V1")
    pulse = Pulse(
        initial=1 * V,
        pulsed=100 * V,
        delay=2 * ms,
        rise=1 * ms,
        fall=1 * ms,
        width=10 * ms,
    )
    light = c.place(devices.voltage_source(pulse), name="V2")
    upper = c.place(devices.resistor(10 * kohm), name="R1")
    cell = c.place(
        devices.photocell(10 * kohm, gamma=0.8, response_time=3 * ms), name="P1"
    )
    c.connect(supply.positive, upper.a, name="in")
    c.connect(upper.b, cell.a, name="out")
    c.connect(light.positive, cell.light_positive, name="light")
    c.connect(supply.negative, light.negative, cell.b, cell.light_negative, ground)


EXPRESSIONS: Final = (
    "-v(x) ^ 2 ^ 3 + 2 ^ 3 ^ 2 * 1e-3 * v(x)",
    "pow(v(x), 3) - v(x) ^ 2 + pow(3 + v(x), v(two)) + (1 + v(x) ^ 2) ^ -1",
    "(2 + v(x)) ^ 1.5 + pow(v(x) - 3, -3) + (v(x) - 3) ^ -2",
    "pow(abs(v(x)) + 1, 0.5) * u(v(x)) + uramp(v(x)) - sgn(v(x))",
    "limit(v(x), 1, -0.5) + limit(2 * v(x), -1, 0.25)",
    "table(v(x), -1, 0, 0, 1, 1, 0.5) + pwl(v(x), -1, 0, 1, 2)",
    "rounded_table(v(x), 0.3, -1, 0, 0, 1, 1, 0.5, 1.2, 2)",
    "v(x) > 0 && v(x) < 1 ? 1 : (!(v(x) > -1) || v(x) == 2 ? 2 : 3)",
    "if(v(x) >= 0.5, floor(v(x)), ceil(v(x))) + log(2 + v(x)) + log10(3 + v(x))",
    "atan(v(x)) + asinh(v(x)) + tanh(v(x)) + exp(-v(x) ^ 2) / sqrt(2)",
)
"""Expressions written for ngspice from their trees: each operator and function whose
reading differs between the two, from the grouping of `^` and a negative base's power
to `u(0)`, `limit` and the tables."""


@component
def expressions(c: Builder) -> None:
    """Each of `EXPRESSIONS` of the voltage `x` and of a constant 2 V, through 1 kΩ."""
    ground = c.port("ground", Role.GROUND)
    swept = c.place(devices.voltage_source(0.3 * V), name="V1")
    two = c.place(devices.voltage_source(2 * V), name="V2")
    c.connect(swept.positive, name="x")
    c.connect(two.positive, name="two")
    for index, expression in enumerate(EXPRESSIONS, start=1):
        source = c.place(
            devices.behavioral_voltage_source(expression), name=f"B{index}"
        )
        load = c.place(devices.resistor(1 * kohm), name=f"R{index}")
        c.connect(source.positive, load.a, name=f"e{index}")
        c.connect(source.x, swept.positive)
        if "two" in expression:
            c.connect(source.two, two.positive)
        c.connect(source.negative, load.b, ground)
    c.connect(swept.negative, two.negative, ground)


@component
def memristive(c: Builder) -> None:
    """VTEAM's memristor under a sine, fed through a resistor.

    Its state moves the voltage across it, and its thresholds, rates and exponents
    differ in each direction.
    """
    ground = c.port("ground", Role.GROUND)
    sine = Sine(0 * V, 2 * V, 1 * kHz)
    source = c.place(devices.voltage_source(sine), name="V1")
    feed = c.place(devices.resistor(1 * kohm), name="R1")
    cell = c.place(
        devices.voltage_controlled_memristor(
            on_resistance=1 * kohm,
            off_resistance=20 * kohm,
            on_voltage=-0.8 * V,
            off_voltage=0.4 * V,
            on_rate=4e3 / s,
            off_rate=1.5e3 / s,
            on_exponent=2,
            off_exponent=1.5,
            law="exponential",
            state=0.6,
        ),
        name="M1",
    )
    c.connect(source.positive, feed.a, name="in")
    c.connect(feed.b, cell.positive, name="cell")
    c.connect(source.negative, cell.negative, ground)


@component
def ion_drift(c: Builder) -> None:
    """HP's memristor under a sine, fed through a resistor, driven into its window."""
    ground = c.port("ground", Role.GROUND)
    sine = Sine(0 * V, 1.5 * V, 1 * Hz)
    source = c.place(devices.voltage_source(sine), name="V1")
    feed = c.place(devices.resistor(1 * kohm), name="R1")
    cell = c.place(
        devices.current_controlled_memristor(
            on_resistance=100 * ohm,
            off_resistance=16 * kohm,
            charge=60 * uC,
            state=0.5,
            window=0.05,
        ),
        name="M1",
    )
    c.connect(source.positive, feed.a, name="in")
    c.connect(feed.b, cell.positive, name="cell")
    c.connect(source.negative, cell.negative, ground)


@component
def switched(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    pulse = Pulse(
        initial=0 * V,
        pulsed=3 * V,
        delay=10 * us,
        rise=5 * us,
        fall=5 * us,
        width=30 * us,
        period=100 * us,
    )
    control = c.place(devices.voltage_source(pulse), name="V1")
    supply = c.place(devices.voltage_source(5 * V), name="V2")
    feed = c.place(devices.resistor(1 * kohm), name="R1")
    switch = c.place(
        devices.voltage_controlled_switch(
            on_resistance=10 * ohm,
            off_resistance=1 * Mohm,
            on_voltage=2 * V,
            off_voltage=1 * V,
        ),
        name="S1",
    )
    hold = c.place(devices.capacitor(10 * nF), name="C1")
    c.connect(control.positive, switch.control_positive, name="control")
    c.connect(supply.positive, feed.a, name="supply")
    c.connect(feed.b, switch.positive, hold.a, name="out")
    c.connect(
        control.negative,
        supply.negative,
        switch.negative,
        switch.control_negative,
        hold.b,
        ground,
    )


@component
def current_switched(c: Builder) -> None:
    """A switch the current through its control port moves, its parameters defaulted."""
    ground = c.port("ground", Role.GROUND)
    pulse = Pulse(
        initial=0 * V,
        pulsed=3 * V,
        delay=10 * us,
        rise=5 * us,
        fall=5 * us,
        width=30 * us,
        period=100 * us,
    )
    control = c.place(devices.voltage_source(pulse), name="V1")
    limit = c.place(devices.resistor(1 * kohm), name="R2")
    supply = c.place(devices.voltage_source(5 * V), name="V2")
    feed = c.place(devices.resistor(1 * kohm), name="R1")
    switch = c.place(
        devices.current_controlled_switch(
            on_resistance=10 * ohm, on_current=2 * mA, off_current=1 * mA
        ),
        name="W1",
    )
    hold = c.place(devices.capacitor(10 * nF), name="C1")
    c.connect(control.positive, limit.a, name="control")
    c.connect(limit.b, switch.control_positive, name="sensed")
    c.connect(supply.positive, feed.a, name="supply")
    c.connect(feed.b, switch.positive, hold.a, name="out")
    c.connect(
        control.negative,
        supply.negative,
        switch.negative,
        switch.control_negative,
        hold.b,
        ground,
    )


def _power(
    *,
    quasi_saturating: bool = False,
    flicker_coefficient: float | None = None,
    flicker_exponent: float | None = None,
) -> devices.Device:
    """An n-channel VDMOS after LTspice's IRF540, with every part of its model in play.

    It quasi-saturates only if asked, as ngspice's small-signal model leaves out the
    slope of a quasi-saturating drain resistance.
    """
    return devices.vdmos(
        threshold_voltage=4 * V,
        transconductance_parameter=20 * A / V**2,
        channel_length_modulation=0.003 / V,
        mobility_modulation=0.05 / V,
        triode_conductance_multiplier=0.8,
        subthreshold_slope=0.12 * V,
        subthreshold_shift=0.05 * V,
        drain_resistance=0.044 * ohm,
        source_resistance=0.015 * ohm,
        gate_resistance=3 * ohm,
        quasi_saturation_resistance=0.5 * ohm if quasi_saturating else None,
        quasi_saturation_voltage=20 * V if quasi_saturating else None,
        drain_source_resistance=10 * Mohm,
        gate_source_capacitance=1.5 * nF,
        max_gate_drain_capacitance=1.7 * nF,
        min_gate_drain_capacitance=0.25 * nF,
        gate_drain_capacitance_coefficient=0.5 / V,
        saturation_current=88 * pA,
        emission_coefficient=1.05,
        body_resistance=0.024 * ohm,
        junction_capacitance=1.1 * nF,
        junction_potential=0.7 * V,
        grading_coefficient=0.45,
        transit_time=30 * ns,
        breakdown_voltage=100 * V,
        breakdown_current=250 * uA,
        breakdown_emission_coefficient=4,
        threshold_voltage_temperature_coefficient=-4 * mV / K,
        transconductance_parameter_temperature_exponent=-1.3,
        subthreshold_slope_temperature_coefficient=1e-3 / K,
        drain_resistance_temperature_coefficient=4e-3 / K,
        source_resistance_temperature_coefficient=3e-3 / K,
        body_resistance_temperature_coefficient=2e-3 / K,
        flicker_coefficient=flicker_coefficient,
        flicker_exponent=flicker_exponent,
    )


_QUASI_SATURATING = _power(quasi_saturating=True)
"""The n-channel, its drain resistance quasi-saturating."""

_P_POWER = devices.vdmos(
    channel="p",
    threshold_voltage=-4 * V,
    transconductance_parameter=10 * A / V**2,
    channel_length_modulation=0.007 / V,
    triode_conductance_multiplier=0.3,
    subthreshold_slope=0.12 * V,
    drain_resistance=0.02 * ohm,
    source_resistance=0.01 * ohm,
    gate_resistance=2 * ohm,
    quasi_saturation_resistance=0.5 * ohm,
    quasi_saturation_voltage=100 * V,
    drain_source_resistance=200 * Mohm,
    gate_source_capacitance=5 * nF,
    max_gate_drain_capacitance=6 * nF,
    min_gate_drain_capacitance=100 * pF,
    gate_drain_capacitance_coefficient=0.25 / V,
    saturation_current=2 * uA,
    body_resistance=0.02 * ohm,
    junction_capacitance=9 * nF,
    transit_time=260 * ns,
    breakdown_voltage=200 * V,
    breakdown_current=250 * uA,
    breakdown_emission_coefficient=4,
    drain_resistance_temperature_exponent=1.5,
    quasi_saturation_temperature_exponent=0.4,
)
"""A p-channel VDMOS after David Zan's IXTH48P20P, of ngspice's examples, whose
quasi-saturation exaggerates its datasheet's."""


type Switch = Component | devices.Device
"""A power MOSFET placed as a device, or a part read from a SPICE library."""


@component
def power_curves(
    c: Builder,
    transistor: Switch,
    sign: Literal[1, -1] = 1,
    pins: tuple[str, str, str] = ("drain", "gate", "source"),
) -> None:
    """A power MOSFET's output and transfer curves: its gate at `V1`, its drain at `V2`.

    Its drain, gate and source are its `pins`, and `sign` its channel's, 1 for an
    n-channel and -1 for a p-channel. The drain sweeps from where its body diode
    conducts to where it breaks down, and the gate from below the threshold to the full
    channel.
    """
    drain, gate, source = pins
    ground = c.port("ground", Role.GROUND)
    bias = c.place(devices.voltage_source(sign * 6 * V), name="V1")
    supply = c.place(devices.voltage_source(sign * 10 * V), name="V2")
    placed = c.place(transistor, name="M1")
    c.connect(bias.positive, placed[gate], name="gate")
    c.connect(supply.positive, placed[drain], name="drain")
    c.connect(bias.negative, supply.negative, placed[source], ground)


def n_power_curves() -> Component:
    """The quasi-saturating n-channel's curves."""
    return power_curves(_QUASI_SATURATING)


def p_power_curves() -> Component:
    """The p-channel's curves."""
    return power_curves(_P_POWER, -1)


@component
def power_stage(c: Builder) -> None:
    """A VDMOS amplifier: about an ampere through 10 Ω, its gate fed through 1 kΩ."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(24 * V), name="V2")
    signal = Sine(
        offset=5.3 * V, amplitude=0.05 * V, frequency=200 * kHz, phase=math.pi / 2
    )
    source = c.place(devices.voltage_source(signal), name="V1")
    feed = c.place(devices.resistor(1 * kohm), name="RG")
    transistor = c.place(
        _power(flicker_coefficient=1e-25, flicker_exponent=1.2),
        name="M1",
    )
    load = c.place(devices.resistor(10 * ohm), name="RD")
    degeneration = c.place(devices.resistor(1 * ohm), name="RS")
    c.connect(source.positive, feed.a, name="in")
    c.connect(feed.b, transistor.gate, name="gate")
    c.connect(supply.positive, load.a, name="supply")
    c.connect(load.b, transistor.drain, name="out")
    c.connect(transistor.source, degeneration.a, name="tail")
    c.connect(source.negative, supply.negative, degeneration.b, ground)


@component
def synchronous_buck(
    c: Builder,
    high: Switch,
    low: Switch,
    pins: tuple[str, str, str] = ("drain", "gate", "source"),
) -> None:
    """A synchronous buck from 12 V: `high`, a p-channel VDMOS, and `low`, an n-channel.

    Each switch's drain, gate and source are its `pins`, and its drain the switching
    node, through 10 µH into 10 µF and 5 Ω. Sines at 250 kHz drive the gates, the high
    side's source-gate voltage `3 + 7 sin` and the low side's gate-source voltage
    `3 - 7 sin`, so each conducts in its half of the period and neither while the sine
    crosses zero, when the low side's body diode carries the inductor's current.
    Smooth drives keep ngspice converging at its tight tolerances, which it does not
    where a pulse's corners switch a gate behind a resistance.
    """
    drain, gate, source = pins
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(12 * V), name="V1")
    high_drive = Sine(offset=9 * V, amplitude=-7 * V, frequency=250 * kHz)
    low_drive = Sine(offset=3 * V, amplitude=-7 * V, frequency=250 * kHz)
    high_gate = c.place(devices.voltage_source(high_drive), name="VH")
    low_gate = c.place(devices.voltage_source(low_drive), name="VL")
    upper = c.place(high, name="MH")
    lower = c.place(low, name="ML")
    inductor = c.place(devices.inductor(10 * uH), name="L1")
    capacitor = c.place(devices.capacitor(10 * uF), name="C1")
    load = c.place(devices.resistor(5 * ohm), name="RL")
    c.connect(supply.positive, upper[source], name="supply")
    c.connect(high_gate.positive, upper[gate], name="high_gate")
    c.connect(low_gate.positive, lower[gate], name="low_gate")
    c.connect(upper[drain], lower[drain], inductor.a, name="switch")
    c.connect(inductor.b, capacitor.a, load.a, name="out")
    c.connect(
        supply.negative,
        high_gate.negative,
        low_gate.negative,
        lower[source],
        capacitor.b,
        load.b,
        ground,
    )


def vdmos_buck() -> Component:
    """The buck of the corpus's two VDMOS."""
    return synchronous_buck(_P_POWER, _QUASI_SATURATING)


@component
def flickering(c: Builder) -> None:
    """Five devices with flicker noise, their currents meeting at the output.

    They are a diode, a bipolar transistor, a JFET, a MOSFET of two copies and a
    resistor.
    """
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(15 * V), name="V2")
    source = c.place(devices.voltage_source(1.5 * V), name="V1")
    load = c.place(devices.resistor(1 * kohm), name="RD")
    unipolar = c.place(
        devices.mosfet(
            width=10 * um,
            length=2 * um,
            lateral_diffusion=0.1 * um,
            multiplier=2,
            threshold_voltage=1 * V,
            transconductance_parameter=50 * uA / V**2,
            oxide_thickness=20 * nm,
            flicker_coefficient=1e-24,
            flicker_exponent=1.3,
        ),
        name="M1",
    )
    degeneration = c.place(devices.resistor(1 * kohm), name="RS")
    bipolar = c.place(
        devices.bipolar_transistor(
            saturation_current=1e-15 * A,
            base_resistance=200 * ohm,
            flicker_coefficient=1e-14,
            flicker_exponent=1.2,
        ),
        name="Q1",
    )
    base = c.place(devices.resistor(470 * kohm), name="RB")
    channel = c.place(
        devices.jfet(
            threshold_voltage=-2 * V,
            transconductance_parameter=1 * mA / V**2,
            flicker_coefficient=1e-15,
        ),
        name="J1",
    )
    tail = c.place(devices.resistor(2 * kohm), name="RJ")
    diode = c.place(
        devices.diode(
            saturation_current=1e-14 * A,
            series_resistance=5 * ohm,
            flicker_coefficient=1e-16,
            flicker_exponent=1.1,
        ),
        name="D1",
    )
    bleed = c.place(
        devices.resistor(47 * kohm, flicker_coefficient=1e-10, flicker_exponent=1.4),
        name="RL",
    )
    c.connect(source.positive, unipolar.gate, name="in")
    c.connect(supply.positive, load.a, base.a, name="supply")
    c.connect(
        load.b,
        unipolar.drain,
        bipolar.collector,
        channel.drain,
        bleed.a,
        name="out",
    )
    c.connect(unipolar.source, degeneration.a, name="sink")
    c.connect(base.b, bipolar.base, name="base")
    c.connect(channel.source, tail.a, name="tail")
    c.connect(bleed.b, diode.anode, name="bleed")
    c.connect(
        source.negative,
        supply.negative,
        degeneration.b,
        unipolar.bulk,
        bipolar.emitter,
        channel.gate,
        tail.b,
        diode.cathode,
        ground,
    )


@pytest.mark.parametrize(
    "top",
    [
        divider,
        clipper,
        zener_clamp,
        amplifier,
        rectifier,
        jfet_stage,
        p_jfet_follower,
        bjt_stage,
        bjt_amplifier,
        integrated_stage,
        pnp_mirror,
        current_controlled,
        negative_load,
        nmos_stage,
        mosfet_curves,
        level_three_stage,
        level_three_curves,
        cmos_inverter,
        n_power_curves,
        p_power_curves,
        power_stage,
        behavioral,
        switched,
        curved,
        lit,
        memristive,
        ion_drift,
        expressions,
    ],
)
def test_operating_points_agree(top: Callable[[], Component]) -> None:
    simulation = Simulation(top())
    point = simulation.run(OperatingPoint(), TIGHT)
    raw = ngspice.run(netlist(simulation.circuit), "op")
    for column, unknown in compared(simulation):
        spice = raw.real(vector(simulation.circuit, unknown))[0]
        compare(point.values[column], spice, unknown, scale=abs(spice), tolerance=1e-6)


@component
def bare_transistor(c: Builder, polarity: Literal["npn", "pnp"]) -> None:
    """A transistor without a substrate, its base fed through 100 kΩ from the ground.

    Its emitter is held 1 V away, and its collector current is its beta times the
    base's, so a picoampere into its base moves its collector a hundred times as much.
    """
    ground = c.port("ground", Role.GROUND)
    level = 1 * V if polarity == "pnp" else -1 * V
    supply = c.place(devices.voltage_source(level), name="V1")
    base = c.place(devices.resistor(100 * kohm), name="RB")
    load = c.place(devices.resistor(1 * kohm), name="RC")
    transistor = c.place(
        devices.bipolar_transistor(polarity=polarity, saturation_current=8.7 * fA),
        name="Q1",
    )
    c.connect(supply.positive, transistor.emitter, name="emitter")
    c.connect(base.a, transistor.base, name="base")
    c.connect(load.a, transistor.collector, name="collector")
    c.connect(supply.negative, base.b, load.b, ground)


@pytest.mark.parametrize("polarity", ["npn", "pnp"])
def test_a_transistor_without_a_substrate_agrees_to_rounding(
    polarity: Literal["npn", "pnp"],
) -> None:
    """A transistor without a substrate agrees with ngspice's to rounding.

    ngspice grounds a missing substrate, whose junction conducts gmin; written on the
    junction's own side, it carries nothing, as jolty's transistor has no junction.
    """
    simulation = Simulation(bare_transistor(polarity))
    point = simulation.run(OperatingPoint(), TIGHT)
    raw = ngspice.run(netlist(simulation.circuit), "op")
    for column, unknown in compared(simulation):
        spice = raw.real(vector(simulation.circuit, unknown))[0]
        compare(point.values[column], spice, unknown, scale=abs(spice), tolerance=1e-10)


TEMPERATURES = {"cold": celsius(-40), "hot": celsius(125)}
"""The temperatures, away from the nominal 27 °C, at which both engines also run."""


@pytest.mark.parametrize("temperature", TEMPERATURES.values(), ids=TEMPERATURES.keys())
@pytest.mark.parametrize(
    "top",
    [
        clipper,
        zener_clamp,
        jfet_stage,
        p_jfet_follower,
        bjt_stage,
        integrated_stage,
        pnp_mirror,
        nmos_stage,
        level_three_stage,
        cmos_inverter,
        n_power_curves,
        p_power_curves,
        power_stage,
        drifting,
        drifting_apart,
        curved,
    ],
)
def test_operating_points_agree_across_temperatures(
    top: Callable[[], Component], temperature: Quantity
) -> None:
    simulation = Simulation(top(), temperature)
    point = simulation.run(OperatingPoint(), TIGHT)
    raw = ngspice.run(netlist(simulation.circuit), "op", temperature=temperature.value)
    for column, unknown in compared(simulation):
        spice = raw.real(vector(simulation.circuit, unknown))[0]
        compare(point.values[column], spice, unknown, scale=abs(spice), tolerance=1e-6)


@pytest.mark.parametrize("temperature", TEMPERATURES.values(), ids=TEMPERATURES.keys())
@pytest.mark.parametrize(
    ("top", "source", "levels"),
    [
        (zener_clamp, "V1", linear(-12 * V, 2 * V, 141)),
        (jfet_curves, "V2", linear(-2 * V, 6 * V, 81)),
        (pnp_mirror, "V1", linear(0 * V, 12 * V, 61)),
        (mosfet_curves, "V1", linear(-1 * V, 5 * V, 61)),
        (level_three_curves, "V1", linear(-0.5 * V, 3 * V, 71)),
        (n_power_curves, "V2", linear(-1 * V, 101 * V, 103)),
        (n_power_curves, "V1", linear(0 * V, 8 * V, 81)),
        (p_power_curves, "V2", linear(-201 * V, 1 * V, 102)),
        (p_power_curves, "V1", linear(-8 * V, 0 * V, 81)),
        (drifting, "V1", linear(0 * V, 15 * V, 76)),
        (drifting_apart, "V1", linear(0 * V, 15 * V, 76)),
    ],
)
def test_dc_sweeps_agree_across_temperatures(
    top: Callable[[], Component],
    source: str,
    levels: list[Quantity],
    temperature: Quantity,
) -> None:
    simulation = Simulation(top(), temperature)
    sweep = simulation.run(DcSweep(source, levels), TIGHT)
    first, last = float(sweep.swept[0]), float(sweep.swept[-1])
    step = (last - first) / (len(levels) - 1)
    command = f"dc V{source} {first!r} {last!r} {step!r}"
    raw = ngspice.run(
        netlist(simulation.circuit), command, temperature=temperature.value
    )
    for column, unknown in compared(simulation):
        spice = raw.real(vector(simulation.circuit, unknown))
        compare(
            sweep.values[:, column], spice, unknown, scale=size(spice), tolerance=1e-6
        )


@pytest.mark.parametrize("temperature", TEMPERATURES.values(), ids=TEMPERATURES.keys())
@pytest.mark.parametrize(
    "top", [bjt_amplifier, nmos_stage, power_stage, drifting, drifting_apart]
)
def test_ac_responses_agree_across_temperatures(
    top: Callable[[], Component], temperature: Quantity
) -> None:
    simulation = Simulation(top(), temperature)
    response = simulation.run(Ac("V1", decade(10 * Hz, 10 * MHz, 20)), TIGHT)
    lines = netlist(simulation.circuit, excited="V1")
    raw = ngspice.run(lines, "ac dec 20 10 10meg", temperature=temperature.value)
    for column, unknown in compared(simulation):
        spice = raw.vectors[vector(simulation.circuit, unknown)]
        compare(
            response.values[:, column],
            spice,
            unknown,
            scale=size(spice),
            tolerance=1e-6,
        )


def test_loop_gains_agree() -> None:
    """Jolty's loop gain against Middlebrook's, from ngspice's two injections."""
    simulation = Simulation(feedback_pair())
    loop = simulation.run(Stability("VP", decade(10 * Hz, 1 * GHz, 10)), TIGHT)
    circuit = simulation.circuit
    command = "ac dec 10 10 1g"
    series = ngspice.run(netlist(circuit, excited="VP"), command)
    shunt = ngspice.run(netlist(circuit, excited="IP"), command)
    entering = series.vectors[vector(circuit, results.Unknown("fb", "voltage"))]
    current = results.Unknown("VP:current", "current")
    u = entering - shunt.vectors[vector(circuit, current)]
    expected = -u / (1 + u)
    assert abs(expected[0]) > 1
    compare(loop.gains, expected, current, scale=size(expected), tolerance=1e-6)


@pytest.mark.parametrize(
    ("top", "source", "levels"),
    [
        (divider, "V1", linear(-5 * V, 5 * V, 21)),
        (clipper, "V1", linear(-2 * V, 2 * V, 81)),
        (zener_clamp, "V1", linear(-12 * V, 2 * V, 141)),
        (jfet_stage, "V1", linear(-3 * V, 1 * V, 81)),
        (jfet_curves, "V2", linear(-2 * V, 6 * V, 81)),
        (p_jfet_follower, "V1", linear(-4 * V, 2 * V, 61)),
        (bjt_stage, "V1", linear(-1 * V, 3 * V, 81)),
        (integrated_stage, "V3", linear(-2 * V, 6 * V, 81)),
        (pnp_mirror, "V1", linear(0 * V, 12 * V, 61)),
        (current_controlled, "V1", linear(-3 * V, 3 * V, 13)),
        (nmos_stage, "V1", linear(-1 * V, 5 * V, 61)),
        (mosfet_curves, "V2", linear(-2 * V, 6 * V, 81)),
        (mosfet_curves, "V1", linear(-1 * V, 5 * V, 61)),
        (level_three_stage, "V1", linear(0 * V, 3 * V, 61)),
        (level_three_curves, "V2", linear(-3 * V, 5 * V, 81)),
        (level_three_curves, "V1", linear(-0.5 * V, 3 * V, 71)),
        # `KAPPA=0`, read by ngspice as no channel shortening, with and without
        # velocity saturation.
        (lambda: level_three_curves(None, 0.0), "V2", linear(-3 * V, 5 * V, 81)),
        # A negative lateral diffusion lengthens the channel, as CircuitSim90's jge's
        # cards have it.
        (
            lambda: level_three_curves(lateral_diffusion=-0.025 * um),
            "V2",
            linear(-3 * V, 5 * V, 81),
        ),
        (
            lambda: level_three_curves(1.5e5 * m / s, 0.0),
            "V2",
            linear(-3 * V, 5 * V, 81),
        ),
        (cmos_inverter, "V1", linear(0 * V, 3.3 * V, 67)),
        (n_power_curves, "V2", linear(-1 * V, 101 * V, 103)),
        (n_power_curves, "V1", linear(0 * V, 8 * V, 81)),
        (p_power_curves, "V2", linear(-201 * V, 1 * V, 102)),
        (p_power_curves, "V1", linear(-8 * V, 0 * V, 81)),
        (power_stage, "V1", linear(3 * V, 6 * V, 61)),
        (behavioral, "V1", linear(-3 * V, 3 * V, 61)),
        (switched, "V1", linear(0 * V, 3 * V, 61)),
        (curved, "V1", linear(-3 * V, 3 * V, 61)),
        (lit, "V2", linear(-1 * V, 100 * V, 102)),
        (expressions, "V1", linear(-1.95 * V, 1.95 * V, 40)),
    ],
)
def test_dc_sweeps_agree(
    top: Callable[[], Component], source: str, levels: list[Quantity]
) -> None:
    simulation = Simulation(top())
    sweep = simulation.run(DcSweep(source, levels), TIGHT)
    first, last = float(sweep.swept[0]), float(sweep.swept[-1])
    step = (last - first) / (len(levels) - 1)
    command = f"dc V{source} {first!r} {last!r} {step!r}"
    raw = ngspice.run(netlist(simulation.circuit), command)
    for column, unknown in compared(simulation):
        spice = raw.real(vector(simulation.circuit, unknown))
        compare(
            sweep.values[:, column], spice, unknown, scale=size(spice), tolerance=1e-6
        )


@pytest.mark.parametrize(
    "top",
    [
        low_pass,
        resonator,
        amplifier,
        jfet_stage,
        bjt_amplifier,
        integrated_stage,
        current_controlled,
        nmos_stage,
        level_three_stage,
        cmos_inverter,
        power_stage,
        behavioral,
        switched,
        current_switched,
        transformer,
        expressions,
    ],
)
def test_ac_responses_agree(top: Callable[[], Component]) -> None:
    simulation = Simulation(top())
    response = simulation.run(Ac("V1", decade(10 * Hz, 10 * MHz, 20)), TIGHT)
    lines = netlist(simulation.circuit, excited="V1")
    raw = ngspice.run(lines, "ac dec 20 10 10meg")
    np.testing.assert_allclose(response.frequencies, raw.real("frequency"), rtol=1e-9)
    for column, unknown in compared(simulation):
        spice = raw.vectors[vector(simulation.circuit, unknown)]
        compare(
            response.values[:, column],
            spice,
            unknown,
            scale=size(spice),
            tolerance=1e-6,
        )


@pytest.mark.parametrize(
    ("top", "source"),
    [
        (low_pass, "V1"),
        (resonator, "V1"),
        (jfet_stage, "V1"),
        (bjt_amplifier, "V1"),
        (integrated_stage, "V1"),
        (nmos_stage, "V1"),
        (level_three_stage, "V1"),
        (power_stage, "V1"),
        (flickering, "V1"),
    ],
)
def test_noise_agrees(top: Callable[[], Component], source: str) -> None:
    """The noise at the output, and referred to the input, as ngspice finds them."""
    simulation = Simulation(top())
    spectrum = simulation.run(
        Noise("out", decade(1 * Hz, 100 * MHz, 5), source=source), TIGHT
    )
    lines = netlist(simulation.circuit, excited=source)
    raw = ngspice.run(lines, f"noise v(out) V{source} dec 5 1 100meg\nsetplot noise1")
    np.testing.assert_allclose(spectrum.frequencies, raw.real("frequency"), rtol=1e-9)
    np.testing.assert_allclose(
        np.sqrt(spectrum.densities), raw.real("onoise_spectrum"), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.sqrt(spectrum.referred()), raw.real("inoise_spectrum"), rtol=1e-5
    )


NGSPICE_INFINITE_RESISTANCE: Final = 1e20
"""What ngspice's `.tf` writes for a resistance through which no current flows."""


@pytest.mark.parametrize(
    "top", [divider, jfet_stage, bjt_amplifier, integrated_stage, nmos_stage]
)
def test_transfer_functions_agree(top: Callable[[], Component]) -> None:
    """The gain at DC and both resistances, as ngspice's `.tf` finds them.

    Both linearize the circuit at operating points that agree within a millionth, so
    the gain and the resistances agree within the noise comparison's 1e-5, and a
    resistance ngspice writes as its 1e20 is infinite in jolty.
    """
    simulation = Simulation(top())
    found = simulation.run(TransferFunction("V1", "out"), TIGHT)
    circuit = simulation.circuit
    output = vector(circuit, results.Unknown("out", "voltage"))
    raw = ngspice.run(netlist(circuit), f"tf {output} VV1")

    def read(part: str) -> float:
        (name,) = [name for name in raw.vectors if part in name.lower()]
        value = float(raw.vectors[name].real[0])
        # ngspice's TFanal writes 1e20 for a resistance through which less than 1e-20 A
        # flows, as into a coupling capacitor at DC: an infinite one.
        return math.inf if value == NGSPICE_INFINITE_RESISTANCE else value

    expected = (
        read("transfer_function"),
        read("input_impedance"),
        read("output_impedance"),
    )
    assert (
        found.gain,
        found.input_resistance,
        found.output_resistance,
    ) == pytest.approx(expected, rel=1e-5)


def _ordered(roots: results.Complex) -> results.Complex:
    """`roots` by increasing magnitude, a conjugate pair's upper member first."""
    return roots[np.lexsort((-roots.imag, np.round(np.abs(roots), 6)))]


@pytest.mark.parametrize(
    "top", [low_pass, resonator, amplifier, jfet_stage, bjt_amplifier, nmos_stage]
)
def test_poles_and_zeros_agree(top: Callable[[], Component]) -> None:
    """The poles, and the zeros from the input to the output, as ngspice finds them."""
    simulation = Simulation(top())
    found = simulation.run(PoleZero(source="V1", output="out"), TIGHT)
    lines = netlist(simulation.circuit, excited="V1")
    raw = ngspice.run(lines, "pz in 0 out 0 vol pz")
    roots = {
        kind: _ordered(
            np.array(
                [
                    values[0]
                    for name, values in raw.vectors.items()
                    if name.startswith(f"v({kind}(")
                ],
                dtype=np.complex128,
            )
        )
        for kind in ("pole", "zero")
    }
    assert found.zeros is not None
    for kind, jolty in (("pole", found.poles), ("zero", found.zeros)):
        spice = roots[kind]
        assert jolty.size == spice.size, (kind, jolty, spice)
        scale = max(np.abs(spice).max(initial=0.0), 1.0)
        np.testing.assert_allclose(
            _ordered(jolty), spice, rtol=1e-6, atol=1e-9 * scale, err_msg=kind
        )


@pytest.mark.parametrize(
    ("top", "stop", "step"),
    [
        (low_pass, 8 * us, 10 * ns),
        (exponential, 12 * us, 10 * ns),
        (resonator, 60 * us, 50 * ns),
        (triangle, 60 * us, 50 * ns),
        (rectifier, 3 * ms, 1 * us),
        (driven_tank, 40 * us, 20 * ns),
        (transformer, 100 * us, 20 * ns),
        (jfet_stage, 15 * us, 10 * ns),
        (bjt_amplifier, 4 * us, 2 * ns),
        (integrated_stage, 4 * us, 2 * ns),
        (delayed_stage, 40 * ns, 0.002 * ns),
        (nmos_stage, 2 * us, 1 * ns),
        (level_three_stage, 2 * us, 1 * ns),
        (cmos_inverter, 160 * ns, 0.01 * ns),
        (power_stage, 20 * us, 20 * ns),
        (vdmos_buck, 16 * us, 10 * ns),
        (behavioral, 300 * us, 0.5 * us),
        (switched, 200 * us, 0.2 * us),
        (current_switched, 200 * us, 0.2 * us),
        (curved, 3 * ms, 2 * us),
        (lit, 30 * ms, 20 * us),
        (memristive, 3 * ms, 1 * us),
        (ion_drift, 2 * s, 1 * ms),
    ],
)
def test_transients_agree(
    top: Callable[[], Component], stop: Quantity, step: Quantity
) -> None:
    simulation = Simulation(top())
    waveforms = simulation.run(Transient(stop=stop, max_step=step), TIGHT)
    command = f"tran {step.value!r} {stop.value!r} 0 {step.value!r}"
    raw = ngspice.run(netlist(simulation.circuit, transient=True), command)
    times = raw.real("time")
    for column, unknown in compared(simulation):
        spice = raw.real(vector(simulation.circuit, unknown))
        jolty = closest(
            waveforms.times, waveforms.values[:, column], times, spice, step.value
        )
        compare(jolty, spice, unknown, scale=size(spice), tolerance=1e-3)


@pytest.mark.parametrize("operating_point", [True, False], ids=["op", "uic"])
@pytest.mark.parametrize(
    ("top", "stop", "step"),
    [(low_pass, 8 * us, 10 * ns), (resonator, 60 * us, 50 * ns)],
)
def test_transients_from_initial_conditions_agree(
    top: Callable[[], Component],
    stop: Quantity,
    step: Quantity,
    *,
    operating_point: bool,
) -> None:
    """Each circuit's capacitor starts at 0.5 V, as ngspice's ``.ic`` sets it.

    It is held there while the operating point is solved, the resonator's inductor
    carrying the current its resistor then passes, or, as ``uic``, set there alone,
    every other unknown at zero.
    """
    simulation = Simulation(top())
    initial = {"out": 0.5 * V}
    analysis = Transient(
        stop=stop,
        max_step=step,
        initial_conditions=initial,
        operating_point=operating_point,
    )
    waveforms = simulation.run(analysis, TIGHT)
    start = "" if operating_point else " uic"
    command = f"tran {step.value!r} {stop.value!r} 0 {step.value!r}{start}"
    deck = [
        *netlist(simulation.circuit, transient=True),
        *node_voltages(".ic", simulation.circuit, {"out": 0.5}),
    ]
    raw = ngspice.run(deck, command)
    times = raw.real("time")
    assert waveforms.voltage("out")[0] == pytest.approx(0.5, abs=1e-12)
    for column, unknown in compared(simulation):
        spice_values = raw.real(vector(simulation.circuit, unknown))
        jolty = closest(
            waveforms.times,
            waveforms.values[:, column],
            times,
            spice_values,
            step.value,
        )
        compare(jolty, spice_values, unknown, scale=size(spice_values), tolerance=1e-3)


@component
def tanh_latch(c: Builder) -> None:
    """A node fed ``2 mA·tanh(v)`` and loaded by 1 kΩ: three DC solutions.

    They are the roots of ``v = 2 tanh(v)``, 0 and ±1.915 V.
    """
    ground = c.port("ground", Role.GROUND)
    feedback = c.place(
        devices.behavioral_current_source("2e-3 * tanh(v(x))"), name="B1"
    )
    load = c.place(devices.resistor(1 * kohm), name="R1")
    c.connect(feedback.negative, feedback.x, load.a, name="x")
    c.connect(feedback.positive, load.b, ground)


@pytest.mark.parametrize("nodeset", [None, 1.5, -1.5])
def test_nodesets_lead_both_to_one_operating_point(nodeset: float | None) -> None:
    """Both find the latch's solution at 0 V, or with a nodeset that of its sign."""
    simulation = Simulation(tanh_latch())
    nodesets = {} if nodeset is None else {"x": nodeset}
    options = replace(TIGHT, nodesets={net: v * V for net, v in nodesets.items()})
    point = simulation.run(OperatingPoint(), options)
    deck = [
        *netlist(simulation.circuit),
        *node_voltages(".nodeset", simulation.circuit, nodesets),
    ]
    raw = ngspice.run(deck, "op")
    for column, unknown in compared(simulation):
        found = raw.real(vector(simulation.circuit, unknown))[0]
        compare(point.values[column], found, unknown, scale=abs(found), tolerance=1e-6)
    found = point.voltage("x")
    if nodeset is None:
        assert abs(found) < 1e-9
    else:
        # The root of v = 2 tanh(v) of the nodeset's sign, 1.915 V away from zero.
        assert found * nodeset > 0
        assert abs(found) > 1.9


@pytest.mark.parametrize("binary", [False, True], ids=["ascii", "binary"])
def test_raw_files_read_back_as_ngspices_own(tmp_path: Path, *, binary: bool) -> None:
    """Results written as SPICE's raw file read back as computed, and as ngspice's.

    A divider's operating point and an RC's response and step read back exactly; the
    response's file has the vectors ngspice's raw file of the same run has, written in
    the same format, and agrees with them within 1e-6, the Newton tolerance.
    """
    rc = Simulation(low_pass())
    analyses = {
        "op": Simulation(divider()).run(OperatingPoint(), TIGHT),
        "bode": rc.run(Ac("V1", decade(1 * kHz, 1 * MHz, 2)), TIGHT),
        "step": rc.run(Transient(stop=4 * us, max_step=10 * ns), TIGHT),
    }
    path = tmp_path / "jolty.raw"
    files.save_raw(path, analyses, binary=binary)
    ours = ngspice.read(path.read_bytes())
    assert [plot.plot for plot in ours] == [
        "Operating Point",
        "AC Analysis",
        "Transient Analysis",
    ]
    for plot, result in zip(ours, analyses.values(), strict=True):
        assert isinstance(
            result,
            results.OperatingPointResult | results.AcResult | results.TransientResult,
        )
        points = len(next(iter(plot.vectors.values())))
        table = np.asarray(result.values).reshape(points, -1)
        for column, unknown in enumerate(result.unknowns):
            written = plot.vectors[ngspice_name(unknown)]
            np.testing.assert_array_equal(written, table[:, column])
    filetype = "binary" if binary else "ascii"
    temperature = ngspice.option_temperature(_core.REFERENCE_TEMPERATURE)
    executed = ngspice.execute(
        [
            *netlist(rc.circuit, excited="V1"),
            ngspice.OPTIONS,
            f".options temp={temperature!r} tnom={temperature!r}",
            ".control",
            f"set filetype={filetype}",
            "set appendwrite",
            "ac dec 2 1k 1meg",
            f"write {ngspice.VALUES} all",
            ".endc",
            ".end",
        ]
    )
    assert executed.values is not None
    (theirs,) = ngspice.read(executed.values)
    (_, bode, _) = ours
    response = analyses["bode"]
    # ngspice steps each frequency from the last by the decade's ratio, gathering an
    # ulp a step, where jolty's are each the nearest float to the exact frequency.
    np.testing.assert_allclose(
        theirs.real("frequency"), bode.real("frequency"), rtol=1e-14, atol=0
    )
    for unknown in response.unknowns:
        theirs_vector = theirs.vectors[vector(rc.circuit, unknown)]
        scale = float(np.max(np.abs(theirs_vector)))
        ours_vector = bode.vectors[ngspice_name(unknown)]
        compare(ours_vector, theirs_vector, unknown, scale=scale, tolerance=1e-6)


def ngspice_name(unknown: results.Unknown) -> str:
    """The name ngspice's raw reader gives the vector of `unknown`."""
    if unknown.nature == "current":
        return f"i({unknown.name.removesuffix(':current').lower()})"
    return f"v({unknown.name.lower()})"


VENDOR_CARDS: Final = """\
* Two power MOSFETs' VDMOS cards, as ngspice 47's examples/vdmos give them:
* International Rectifier's IRFZ48Z, as LTspice's library holds it, and David Zan's
* IXTH48P20P, with the quasi-saturation ngspice's example adds.
.model IRFZ48Z VDMOS (Rg = 1.77 Vto=4 Rd=1.85m Rs=0.0m Rb=3.75m Kp=25 Cgdmax=2.1n
+ Cgdmin=0.05n Cgs=1.8n Cjo=0.55n Is=2.5p tt=20n ksubthres=0.1
+ mfg=International_Rectifier Vds=55 Ron=8.6m Qg=43n)
.MODEL IXTH48P20P VDMOS Pchan Vds=200
+ VTO=-4 KP=10
+ Lambda=7m ; the example's, the original's 5m
+ Mtriode=0.3
+ Ksubthres=120m
+ Rs=10m Rd=20m Rds=200e6
+ Cgdmax=6000p Cgdmin=100p A=0.25
+ Cgs=5000p Cjo=9000p
+ Is=2e-6 Rb=20m
+ BV=200 IBV=250e-6
+ NBV=4
+ TT=260e-9
+ vq=100
+ rq=0.5
.subckt irfz48z d g s
M1 d g s IRFZ48Z
.ends
.subckt ixth48p20p d g s
M1 d g s s IXTH48P20P
.ends
"""
"""A library of the two parts, each a subcircuit of its transistor as a vendor's
library places it, the p-channel's with its fourth node on its source."""

_PINS: Final = ("d", "g", "s")
"""The vendor parts' drain, gate and source."""


@pytest.fixture(scope="module")
def vendor_cards(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The library of `VENDOR_CARDS`, as a file both engines read."""
    path = tmp_path_factory.mktemp("vendor") / "vdmos.lib"
    path.write_text(VENDOR_CARDS, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("part", "sign", "source", "levels"),
    [
        ("irfz48z", 1, "V2", linear(-1 * V, 60 * V, 62)),
        ("irfz48z", 1, "V1", linear(0 * V, 8 * V, 81)),
        ("ixth48p20p", -1, "V2", linear(-201 * V, 1 * V, 102)),
        ("ixth48p20p", -1, "V1", linear(-8 * V, 0 * V, 81)),
    ],
)
def test_vendors_cards_sweep_as_in_ngspice(
    vendor_cards: Path,
    part: str,
    sign: Literal[1, -1],
    source: str,
    levels: list[Quantity],
) -> None:
    """A vendor's VDMOS card, read by the SPICE reader, against ngspice reading it."""
    read = spice.load(vendor_cards).component(part)
    simulation = Simulation(power_curves(read, sign, _PINS))
    sweep = simulation.run(DcSweep(source, levels), TIGHT)
    first, last = float(sweep.swept[0]), float(sweep.swept[-1])
    step = (last - first) / (len(levels) - 1)
    command = f"dc V{source} {first!r} {last!r} {step!r}"
    raw = ngspice.run(netlist(simulation.circuit), command, libraries=(vendor_cards,))
    for column, unknown in compared(simulation):
        spice_values = raw.real(vector(simulation.circuit, unknown))
        compare(
            sweep.values[:, column],
            spice_values,
            unknown,
            scale=size(spice_values),
            tolerance=1e-6,
        )


def test_vendors_cards_switch_as_in_ngspice(vendor_cards: Path) -> None:
    """The synchronous buck of the two parts, read by the SPICE reader, as ngspice."""
    library = spice.load(vendor_cards)
    top = synchronous_buck(
        library.component("ixth48p20p"), library.component("irfz48z"), _PINS
    )
    simulation = Simulation(top)
    stop, step = 16 * us, 10 * ns
    waveforms = simulation.run(Transient(stop=stop, max_step=step), TIGHT)
    command = f"tran {step.value!r} {stop.value!r} 0 {step.value!r}"
    raw = ngspice.run(
        netlist(simulation.circuit, transient=True),
        command,
        libraries=(vendor_cards,),
    )
    times = raw.real("time")
    for column, unknown in compared(simulation):
        spice_values = raw.real(vector(simulation.circuit, unknown))
        jolty = closest(
            waveforms.times,
            waveforms.values[:, column],
            times,
            spice_values,
            step.value,
        )
        compare(jolty, spice_values, unknown, scale=size(spice_values), tolerance=1e-3)


# BSIM3v3.3: the transistors of the processes from 0.8 µm to 0.13 µm.

_ROOT_VOLT: Final = V ** Fraction(1, 2)
"""A square-root volt, the unit of BSIM3's body effects."""

_MOBILITY: Final = m**2 / (V * s)
"""A square meter per volt-second, the unit of a mobility."""


def _bsim3(  # noqa: PLR0913 - the transistor's channel, length and choices of model
    channel: Literal["n", "p"],
    length: Quantity,
    *,
    mobility_model: Literal[1, 2, 3] | None = None,
    capacitance_model: Literal[0, 1, 2, 3] | None = None,
    noise_model: Literal[1, 2, 3, 4, 5, 6] | None = None,
    charge_partition: float | None = None,
    more: Mapping[str, Quantity | float] | None = None,
) -> devices.Device:
    """A BSIM3 transistor 10 µm wide and `length` long of a 0.35 µm process.

    Its threshold, body effect, short-channel effects, poly depletion, parasitic
    resistance, non-saturation, output resistance, substrate current and bias-dependent
    overlaps are set off their defaults, and its diffusions and sheet resistance given;
    a p-channel's threshold is negative and its mobility lower. Its models are
    `mobility_model`, `capacitance_model`, `noise_model` and `charge_partition`, and its
    other parameters `more`.
    """
    sign = 1 if channel == "n" else -1
    card: dict[str, Quantity | float] = {
        "tox": 7.6 * nm,
        "nch": 1.7e23 / m**3,
        "vth0": sign * 0.55 * V,
        "u0": (0.04 if channel == "n" else 0.012) * _MOBILITY,
        "k1": 0.6 * _ROOT_VOLT,
        "k2": -0.02,
        "k3": 20.0,
        "dvt0": 2.5,
        "rdsw": 600 * ohm,
        "prwg": 0.1 / V,
        "pclm": 1.2,
        "pdiblc2": 0.005,
        "alpha0": 1e-6 * m / V,
        "beta0": 20 * V,
        "ngate": 3e26 / m**3,
        "a1": 0.1 / V,
        "a2": 0.9,
        "pvag": 0.5,
        "keta": -0.03 / V,
        "cit": 1e-4 * F / m**2,
        "lint": 20 * nm,
        "wint": 30 * nm,
        "cgsl": 1e-10 * F / m,
        "cgdl": 1e-10 * F / m,
        "rsh": 5 * ohm,
        "cj": 1e-3 * F / m**2,
        "cjsw": 3e-10 * F / m,
        "cjswg": 2e-10 * F / m,
        "js": 1e-5 * A / m**2,
    }
    card.update(more or {})
    return devices.bsim3(
        channel=channel,
        width=10 * um,
        length=length,
        mobility_model=mobility_model,
        capacitance_model=capacitance_model,
        noise_model=noise_model,
        charge_partition=charge_partition,
        drain_area=10 * um**2,
        source_area=10 * um**2,
        drain_perimeter=22 * um,
        source_perimeter=22 * um,
        card=card,
    )


BSIM3_TRANSISTORS: Final = {
    "short-n": _bsim3("n", 0.35 * um),
    "long-n": _bsim3("n", 5 * um, mobility_model=2),
    "short-p": _bsim3(
        "p", 0.35 * um, mobility_model=3, more={"uc": -0.05 / V, "uc1": -0.06 / V}
    ),
    "long-p": _bsim3("p", 5 * um, capacitance_model=1),
}
"""BSIM3 transistors of each channel, a short one and a long one, of each mobmod."""


def bsim_kelvins(temperature: Quantity) -> float:
    """The temperature to run ngspice at so its BSIM runs at jolty's `temperature`.

    BSIM3 and BSIM4 compute with constants of their own, Boltzmann's over the electron's
    charge among them, so ngspice's match jolty's at the same kelvins.
    """
    return ngspice.same_kelvins(temperature.value)


@component
def bsim_curves(c: Builder, transistor: devices.Device) -> None:
    """A BSIM3 or BSIM4 transistor's curves: its gate at `V1`, its drain at `V2`.

    Its bulk, at `V3`, lies 0.5 V behind its source in reverse.
    """
    ground = c.port("ground", Role.GROUND)
    sign = 1 if transistor.parameters.get("channel", "n") == "n" else -1
    bias = c.place(devices.voltage_source(sign * 2 * V), name="V1")
    drain = c.place(devices.voltage_source(sign * 2 * V), name="V2")
    body = c.place(devices.voltage_source(sign * -0.5 * V), name="V3")
    placed = c.place(transistor, name="M1")
    c.connect(bias.positive, placed.gate, name="gate")
    c.connect(drain.positive, placed.drain, name="drain")
    c.connect(body.positive, placed.bulk, name="body")
    c.connect(bias.negative, drain.negative, body.negative, placed.source, ground)


@component
def bsim_stage(c: Builder, transistor: devices.Device) -> None:
    """A common-source stage of a BSIM3 or BSIM4 n-channel, which steps up at 1 µs.

    Its gate is fed through 100 kΩ from `V1` at 1.2 V, and its drain loaded by 2 kΩ
    from 3.3 V and by 0.5 pF.
    """
    ground = c.port("ground", Role.GROUND)
    step = Piecewise([(1 * us, 1.2 * V), (1 * us + 5 * ns, 1.4 * V)])
    source = c.place(devices.voltage_source(step), name="V1")
    supply = c.place(devices.voltage_source(3.3 * V), name="V2")
    gate = c.place(devices.resistor(100 * kohm), name="RG")
    load = c.place(devices.resistor(2 * kohm), name="RD")
    capacitance = c.place(devices.capacitor(0.5 * pF), name="CL")
    placed = c.place(transistor, name="M1")
    c.connect(source.positive, gate.a, name="in")
    c.connect(gate.b, placed.gate, name="gate")
    c.connect(supply.positive, load.a, name="supply")
    c.connect(load.b, capacitance.a, placed.drain, name="out")
    c.connect(
        source.negative,
        supply.negative,
        capacitance.b,
        placed.source,
        placed.bulk,
        ground,
    )


@component
def bsim3_inverter(c: Builder, lower: devices.Device, upper: devices.Device) -> None:
    """A CMOS inverter of BSIM3 transistors at 3.3 V, loaded by 20 fF.

    A pulse drives its gates through 10 kΩ.
    """
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(3.3 * V), name="V2")
    pulse = Pulse(
        initial=0 * V,
        pulsed=3.3 * V,
        delay=2 * ns,
        rise=1 * ns,
        fall=1 * ns,
        width=8 * ns,
        period=20 * ns,
    )
    source = c.place(devices.voltage_source(pulse), name="V1")
    driver = c.place(devices.resistor(10 * kohm), name="RIN")
    load = c.place(devices.capacitor(20 * fF), name="CL")
    n = c.place(lower, name="MN")
    p = c.place(upper, name="MP")
    c.connect(source.positive, driver.a, name="in")
    c.connect(driver.b, n.gate, p.gate, name="gate")
    c.connect(supply.positive, p.source, p.bulk, name="supply")
    c.connect(n.drain, p.drain, load.a, name="out")
    c.connect(source.negative, supply.negative, n.source, n.bulk, load.b, ground)


def _bsim_sweep(
    simulation: Simulation, source: str, levels: list[Quantity], kelvins: float
) -> None:
    """Asserts a DC sweep of `source` over `levels` agrees with ngspice's."""
    sweep = simulation.run(DcSweep(source, levels), TIGHT)
    first, last = float(sweep.swept[0]), float(sweep.swept[-1])
    step = (last - first) / (len(levels) - 1)
    command = f"dc V{source} {first!r} {last!r} {step!r}"
    raw = ngspice.run(netlist(simulation.circuit), command, temperature=kelvins)
    for column, unknown in compared(simulation):
        spice_values = raw.real(vector(simulation.circuit, unknown))
        compare(
            sweep.values[:, column],
            spice_values,
            unknown,
            scale=size(spice_values),
            tolerance=1e-6,
        )


@pytest.mark.parametrize(
    "temperature",
    [celsius(27), *TEMPERATURES.values()],
    ids=["nominal", *TEMPERATURES.keys()],
)
@pytest.mark.parametrize("name", BSIM3_TRANSISTORS)
@pytest.mark.parametrize("source", ["V1", "V2"])
def test_bsim3_curves_agree(name: str, source: str, temperature: Quantity) -> None:
    """A BSIM3 transistor's output and transfer curves, in both modes, as ngspice's.

    The drain sweeps from 1 V behind the source, the bulk junction forward and the
    channel reversed, to 3.3 V; the gate from below the threshold to 3.3 V.
    """
    transistor = BSIM3_TRANSISTORS[name]
    sign = 1 if name.endswith("-n") else -1
    simulation = Simulation(bsim_curves(transistor), temperature)
    bounds = (-1.0, 3.3) if source == "V2" else (-0.5, 3.3)
    levels = linear(sign * bounds[0] * V, sign * bounds[1] * V, 87)
    _bsim_sweep(simulation, source, levels, bsim_kelvins(temperature))


BSIM4_TRANSISTORS: Final = {
    "short-n": devices.bsim4(
        width=1 * um,
        length=0.15 * um,
        drain_squares=1,
        body_resistance_model=1,
        card={
            "toxe": 4.1 * nm,
            "vth0": 0.45 * V,
            "k2": -0.02,
            "ngate": 1e26 / m**3,
            "rdsw": 200 * ohm,
            "rsh": 10 * ohm,
            "alpha0": 1e-6 * m / V,
            "beta0": 20 * V,
            "agidl": 1e-9 / ohm,
            "lambda": 1e-10,
            "vtl": 2e5 * m / s,
            "jtss": 1e-7 * A / m**2,
        },
    ),
    "external-n": devices.bsim4(
        width=2 * um,
        length=0.5 * um,
        drain_squares=2,
        source_squares=1.5,
        drain_source_resistance_model=1,
        thermal_noise_model=1,
        card={
            "vth0": 0.5 * V,
            "rsh": 5 * ohm,
            "rdw": 150 * ohm,
            "rsw": 120 * ohm,
            "rdwmin": 10 * ohm,
            "rswmin": 8 * ohm,
            "prwg": 0.2 / V,
            "prwb": -0.1 / V ** Fraction(1, 2),
            "ngate": 1e26 / m**3,
        },
    ),
    "long-p": devices.bsim4(
        channel="p",
        width=5 * um,
        length=2 * um,
        fingers=2,
        card={"vth0": -0.5 * V, "ute": -1.3, "kt1": -0.2 * V, "tvoff": 1e-3 / K},
    ),
}
"""BSIM4 transistors written for ngspice: a short n-channel with its drain's sheet
resistance, its body's network, its substrate current and gate-induced leakage, its
velocity overshoot, its source end's limit and its junctions' tunneling, an n-channel
whose bias-dependent drain and source resistances lie outside its channel, `rdsmod` 1,
and a long p-channel of two fingers whose temperature scales its mobility and
threshold."""


@pytest.mark.parametrize(
    "temperature",
    [celsius(27), *TEMPERATURES.values()],
    ids=["nominal", *TEMPERATURES.keys()],
)
@pytest.mark.parametrize("name", BSIM4_TRANSISTORS)
@pytest.mark.parametrize("source", ["V1", "V2"])
def test_bsim4_curves_agree(name: str, source: str, temperature: Quantity) -> None:
    """A BSIM4 transistor's output and transfer curves, in both modes, as ngspice's.

    The drain sweeps from 1 V behind the source, the body's junction forward and the
    channel reversed, to 1.8 V; the gate from below the threshold to 1.8 V.
    """
    transistor = BSIM4_TRANSISTORS[name]
    sign = 1 if name.endswith("-n") else -1
    simulation = Simulation(bsim_curves(transistor), temperature)
    bounds = (-1.0, 1.8) if source == "V2" else (-0.5, 1.8)
    levels = linear(sign * bounds[0] * V, sign * bounds[1] * V, 57)
    _bsim_sweep(simulation, source, levels, bsim_kelvins(temperature))


@pytest.mark.parametrize(("name", "decades"), [("short-n", 5), ("external-n", 8)])
def test_bsim4_noise_agrees(name: str, decades: int) -> None:
    """A BSIM4 stage's noise at its output, as ngspice's, from 1 Hz up `decades`.

    The short n-channel's thermal noise is the charge-based model's, `tnoimod` 0, the
    other's the holistic model's, `tnoimod` 1, which moves a share of it to the
    resistance on its source side, there bias-dependent, `rdsmod` 1; both make BSIM4's
    flicker noise, `fnoimod` 1. The short one's body network carries its resistances'
    noise through the body's capacitances, which ngspice's are not the derivatives of
    its charges (the guide's departures): above 100 kHz, where they carry it, its noise
    departs from ngspice's, and is compared below.
    """
    simulation = Simulation(bsim_stage(BSIM4_TRANSISTORS[name]))
    stop = 10.0**decades
    spectrum = simulation.run(
        Noise("out", decade(1 * Hz, stop * Hz, 5), source="V1"), TIGHT
    )
    lines = netlist(simulation.circuit, excited="V1")
    command = f"noise v(out) VV1 dec 5 1 {stop!r}\nsetplot noise1"
    raw = ngspice.run(lines, command, temperature=bsim_kelvins(celsius(27)))
    np.testing.assert_allclose(
        np.sqrt(spectrum.densities), raw.real("onoise_spectrum"), rtol=1e-5
    )


_PARTITIONED: Final[list[tuple[Literal[0, 1, 2, 3], float]]] = [
    (0, 0.0),
    (1, 0.0),
    (2, 0.0),
    (3, 0.0),
    (3, 0.5),
    (0, 1.0),
    (2, 1.0),
    (1, -1.0),
]
"""Each charge model, at the partition 40/60, and with others: 50/50, 0/100 and none."""

_STAGES: Final = {
    f"capmod-{capacitance_model}-xpart-{charge_partition}": _bsim3(
        "n",
        0.5 * um,
        capacitance_model=capacitance_model,
        charge_partition=charge_partition,
        more={"ngate": 0 / m**3} if capacitance_model == 0 else None,
    )
    for capacitance_model, charge_partition in _PARTITIONED
}
"""The stage's transistor in each charge model and partition.

`capacitance_model` 0's has no poly depletion: ngspice's capacitances of
`capacitance_model` 0 multiply by the depletion's slope of the gate voltage twice,
which leaves its AC response 2.5e-4 of the signal from the derivative of its own
charge, which jolty's is."""


@pytest.mark.parametrize("name", _STAGES)
def test_bsim3_ac_responses_agree(name: str) -> None:
    """A BSIM3 stage's operating point and AC response, as ngspice's.

    Its transistor's charges follow each charge model and partition.
    """
    simulation = Simulation(bsim_stage(_STAGES[name]))
    kelvins = bsim_kelvins(celsius(27))
    point = simulation.run(OperatingPoint(), TIGHT)
    raw = ngspice.run(netlist(simulation.circuit), "op", temperature=kelvins)
    for column, unknown in compared(simulation):
        spice_value = raw.real(vector(simulation.circuit, unknown))[0]
        compare(
            point.values[column],
            spice_value,
            unknown,
            scale=abs(spice_value),
            tolerance=1e-6,
        )
    response = simulation.run(Ac("V1", decade(10 * Hz, 10 * GHz, 10)), TIGHT)
    lines = netlist(simulation.circuit, excited="V1")
    raw = ngspice.run(lines, "ac dec 10 10 10g", temperature=kelvins)
    for column, unknown in compared(simulation):
        spice_values = raw.vectors[vector(simulation.circuit, unknown)]
        compare(
            response.values[:, column],
            spice_values,
            unknown,
            scale=size(spice_values),
            tolerance=1e-6,
        )


@pytest.mark.parametrize("noise_model", [1, 2, 3, 4, 5, 6])
def test_bsim3_noise_agrees(noise_model: Literal[1, 2, 3, 4, 5, 6]) -> None:
    """A BSIM3 stage's noise at its output and at its input, as ngspice's.

    Its transistor's noise follows each noise model: SPICE2's or BSIM3's thermal and
    flicker noise.
    """
    transistor = _bsim3(
        "n", 0.5 * um, noise_model=noise_model, more={"kf": 1e-26, "af": 1.2, "ef": 0.9}
    )
    simulation = Simulation(bsim_stage(transistor))
    spectrum = simulation.run(
        Noise("out", decade(1 * Hz, 100 * MHz, 5), source="V1"), TIGHT
    )
    lines = netlist(simulation.circuit, excited="V1")
    command = "noise v(out) VV1 dec 5 1 100meg\nsetplot noise1"
    raw = ngspice.run(lines, command, temperature=bsim_kelvins(celsius(27)))
    np.testing.assert_allclose(spectrum.frequencies, raw.real("frequency"), rtol=1e-9)
    np.testing.assert_allclose(
        np.sqrt(spectrum.densities), raw.real("onoise_spectrum"), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.sqrt(spectrum.referred()), raw.real("inoise_spectrum"), rtol=1e-5
    )


@pytest.mark.parametrize(
    ("top", "stop", "step"),
    [
        (lambda: bsim_stage(_STAGES["capmod-3-xpart-0.0"]), 2 * us, 1 * ns),
        (
            lambda: bsim3_inverter(
                BSIM3_TRANSISTORS["short-n"], BSIM3_TRANSISTORS["short-p"]
            ),
            40 * ns,
            0.01 * ns,
        ),
    ],
    ids=["stage", "inverter"],
)
def test_bsim3_transients_agree(
    top: Callable[[], Component], stop: Quantity, step: Quantity
) -> None:
    """A BSIM3 stage's step response and a CMOS inverter's switching, as ngspice's."""
    simulation = Simulation(top())
    waveforms = simulation.run(Transient(stop=stop, max_step=step), TIGHT)
    command = f"tran {step.value!r} {stop.value!r} 0 {step.value!r}"
    raw = ngspice.run(
        netlist(simulation.circuit, transient=True),
        command,
        temperature=bsim_kelvins(celsius(27)),
    )
    times = raw.real("time")
    for column, unknown in compared(simulation):
        spice_values = raw.real(vector(simulation.circuit, unknown))
        jolty = closest(
            waveforms.times,
            waveforms.values[:, column],
            times,
            spice_values,
            step.value,
        )
        compare(jolty, spice_values, unknown, scale=size(spice_values), tolerance=1e-3)


BERKELEY_CARDS: Final = """\
* UC Berkeley's BSIM3v3.3.0 model cards of a 0.35 um process, modelcard.nmos and
* modelcard.pmos of its release (bsim.berkeley.edu, ftpv330.zip), as ngspice 47's
* examples/mos give them, and transistors as its nmos_pmos_BSIM330.sp places them.
.model N1 NMOS
+Level=        8 version=3.3.0
+Tnom=27.0
+Nch= 2.498E+17  Tox=9E-09 Xj=1.00000E-07
+Lint=9.36e-8 Wint=1.47e-7
+Vth0= .6322    K1= .756  K2= -3.83e-2  K3= -2.612
+Dvt0= 2.812  Dvt1= 0.462  Dvt2=-9.17e-2
+Nlx= 3.52291E-08  W0= 1.163e-6
+K3b= 2.233
+Vsat= 86301.58  Ua= 6.47e-9  Ub= 4.23e-18  Uc=-4.706281E-11
+Rdsw= 650  U0= 388.3203 wr=1
+A0= .3496967 Ags=.1    B0=0.546    B1= 1
+Dwg = -6.0E-09 Dwb = -3.56E-09 Prwb = -.213
+Keta=-3.605872E-02  A1= 2.778747E-02  A2= .9
+Voff=-6.735529E-02  NFactor= 1.139926  Cit= 1.622527E-04
+Cdsc=-2.147181E-05
+Cdscb= 0  Dvt0w =  0 Dvt1w =  0 Dvt2w =  0
+Cdscd =  0 Prwg =  0
+Eta0= 1.0281729E-02  Etab=-5.042203E-03
+Dsub= .31871233
+Pclm= 1.114846  Pdiblc1= 2.45357E-03  Pdiblc2= 6.406289E-03
+Drout= .31871233  Pscbe1= 5000000  Pscbe2= 5E-09 Pdiblcb = -.234
+Pvag= 0 delta=0.01
+Wl =  0 Ww = -1.420242E-09 Wwl =  0
+Wln =  0 Wwn =  .2613948 Ll =  1.300902E-10
+Lw =  0 Lwl =  0 Lln =  .316394
+Lwn =  0
+kt1=-.3  kt2=-.051
+At= 22400
+Ute=-1.48
+Ua1= 3.31E-10  Ub1= 2.61E-19 Uc1= -3.42e-10
+Kt1l=0 Prt=764.3
+vgs_max=4 vds_max=4 vbs_max=4
.model P1 PMOS
+Level=        8 version=3.3.0
+Tnom=27.0
+Nch= 3.533024E+17  Tox=9E-09 Xj=1.00000E-07
+Lint=6.23e-8 Wint=1.22e-7
+Vth0=-.6732829 K1= .8362093  K2=-8.606622E-02  K3= 1.82
+Dvt0= 1.903801  Dvt1= .5333922  Dvt2=-.1862677
+Nlx= 1.28e-8  W0= 2.1e-6
+K3b= -0.24 Prwg=-0.001 Prwb=-0.323
+Vsat= 103503.2  Ua= 1.39995E-09  Ub= 1.e-19  Uc=-2.73e-11
+Rdsw= 460  U0= 138.7609
+A0= .4716551 Ags=0.12
+Keta=-1.871516E-03  A1= .3417965  A2= 0.83
+Voff=-.074182  NFactor= 1.54389  Cit=-1.015667E-03
+Cdsc= 8.937517E-04
+Cdscb= 1.45e-4  Cdscd=1.04e-4
+Dvt0w=0.232 Dvt1w=4.5e6 Dvt2w=-0.0023
+Eta0= 6.024776E-02  Etab=-4.64593E-03
+Dsub= .23222404
+Pclm= .989  Pdiblc1= 2.07418E-02  Pdiblc2= 1.33813E-3
+Drout= .3222404  Pscbe1= 118000  Pscbe2= 1E-09
+Pvag= 0
+kt1= -0.25  kt2= -0.032 prt=64.5
+At= 33000
+Ute= -1.5
+Ua1= 4.312e-9 Ub1= 6.65e-19  Uc1= 0
+Kt1l=0
+vgs_max=4 vds_max=4 vbs_max=4
.subckt nfet d g s b
M1 d g s b n1 W=1u L=0.35u Pd=1.5u Ps=1.5u ad=1.5p as=1.5p
.ends
.subckt pfet d g s b
M1 d g s b p1 W=2.5u L=0.35u Pd=3u Ps=3u ad=2.5p as=2.5p
.ends
"""
"""A library of Berkeley's published cards, each transistor a subcircuit of its own."""


@pytest.fixture(scope="module")
def berkeley_cards(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The library of `BERKELEY_CARDS`, as a file both engines read."""
    path = tmp_path_factory.mktemp("bsim3") / "berkeley.lib"
    path.write_text(BERKELEY_CARDS, encoding="utf-8")
    return path


@component
def part_curves(c: Builder, part: Component, sign: Literal[1, -1]) -> None:
    """A library's transistor's curves: its gate at `V1`, its drain at `V2`.

    Its bulk, at `V3`, lies 0.5 V behind its source in reverse.
    """
    ground = c.port("ground", Role.GROUND)
    bias = c.place(devices.voltage_source(sign * 2 * V), name="V1")
    drain = c.place(devices.voltage_source(sign * 2 * V), name="V2")
    body = c.place(devices.voltage_source(sign * -0.5 * V), name="V3")
    placed = c.place(part, name="X1")
    c.connect(bias.positive, placed["g"], name="gate")
    c.connect(drain.positive, placed["d"], name="drain")
    c.connect(body.positive, placed["b"], name="body")
    c.connect(bias.negative, drain.negative, body.negative, placed["s"], ground)


@pytest.mark.parametrize(
    "temperature",
    [celsius(27), *TEMPERATURES.values()],
    ids=["nominal", *TEMPERATURES.keys()],
)
@pytest.mark.parametrize(("part", "sign"), [("nfet", 1), ("pfet", -1)])
@pytest.mark.parametrize("source", ["V1", "V2"])
def test_published_bsim3_cards_sweep_as_in_ngspice(
    berkeley_cards: Path,
    part: str,
    sign: Literal[1, -1],
    source: str,
    temperature: Quantity,
) -> None:
    """Berkeley's published BSIM3v3.3 cards, read by the reader, as ngspice reads them.

    The drain sweeps from 1 V behind the source to 3.3 V, and the gate from below the
    threshold.
    """
    read = spice.load(berkeley_cards).component(part)
    simulation = Simulation(part_curves(read, sign), temperature)
    bounds = (-1.0, 3.3) if source == "V2" else (-0.5, 3.3)
    levels = linear(sign * bounds[0] * V, sign * bounds[1] * V, 87)
    sweep = simulation.run(DcSweep(source, levels), TIGHT)
    first, last = float(sweep.swept[0]), float(sweep.swept[-1])
    step = (last - first) / (len(levels) - 1)
    raw = ngspice.run(
        netlist(simulation.circuit),
        f"dc V{source} {first!r} {last!r} {step!r}",
        libraries=(berkeley_cards,),
        temperature=bsim_kelvins(temperature),
    )
    for column, unknown in compared(simulation):
        spice_values = raw.real(vector(simulation.circuit, unknown))
        compare(
            sweep.values[:, column],
            spice_values,
            unknown,
            scale=size(spice_values),
            tolerance=1e-6,
        )


BINNED_CARDS: Final = """\
* A binned n-channel in two bins of length, either side of 1 um, of ngspice's binunit 1:
* each dependence per micrometer of the effective length or width, the doping per cubic
* centimeter and the mobility in cm^2/(V s), as a foundry writes them.
.model nch.1 nmos (level=49 version=3.3.0 lmin=0.3u lmax=1u wmin=1u wmax=100u
+ tox=7.6n nch=1.7e17 lnch=2e16 vth0=0.5 lvth0=0.02 wvth0=-0.01 pvth0=0.002
+ u0=420 lu0=-30 wu0=5 k1=0.6 k2=-0.02 lk2=0.005 rdsw=500 wrdsw=100
+ vsat=8e4 lvsat=1e4 pclm=1.2 lpclm=0.1 cgsl=1e-10 lint=20n)
.model nch.2 nmos (level=49 version=3.3.0 lmin=1u lmax=10u wmin=1u wmax=100u
+ tox=7.6n nch=1.7e17 vth0=0.55 lvth0=-0.05 u0=430 lu0=20 k1=0.6 k2=-0.02
+ rdsw=500 vsat=8e4 pclm=1.3 cgsl=1e-10 lint=20n)
.subckt short d g s b
M1 d g s b nch W=10u L=0.5u
.ends
.subckt long d g s b
M1 d g s b nch W=10u L=2u
.ends
.subckt edge d g s b
M1 d g s b nch W=10u L=1u
.ends
"""
"""A binned card in two bins, and transistors each bin takes: on its bound, the bin
ngspice defines last."""


@pytest.fixture(scope="module")
def binned_cards(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The library of `BINNED_CARDS`, as a file both engines read."""
    path = tmp_path_factory.mktemp("binned") / "binned.lib"
    path.write_text(BINNED_CARDS, encoding="utf-8")
    return path


@pytest.mark.parametrize("part", ["short", "long", "edge"])
@pytest.mark.parametrize("source", ["V1", "V2"])
def test_binned_bsim3_cards_sweep_as_in_ngspice(
    binned_cards: Path, part: str, source: str
) -> None:
    """A binned BSIM3 card's transistors, each of the bin it takes, as ngspice's."""
    read = spice.load(binned_cards).component(part)
    simulation = Simulation(part_curves(read, 1))
    bounds = (-1.0, 3.3) if source == "V2" else (-0.5, 3.3)
    levels = linear(bounds[0] * V, bounds[1] * V, 87)
    sweep = simulation.run(DcSweep(source, levels), TIGHT)
    first, last = float(sweep.swept[0]), float(sweep.swept[-1])
    step = (last - first) / (len(levels) - 1)
    raw = ngspice.run(
        netlist(simulation.circuit),
        f"dc V{source} {first!r} {last!r} {step!r}",
        libraries=(binned_cards,),
        temperature=bsim_kelvins(celsius(27)),
    )
    for column, unknown in compared(simulation):
        spice_values = raw.real(vector(simulation.circuit, unknown))
        compare(
            sweep.values[:, column],
            spice_values,
            unknown,
            scale=size(spice_values),
            tolerance=1e-6,
        )


def _vbic(
    polarity: Literal["npn", "pnp"], *, excess_phase: bool = True
) -> devices.Device:
    """A VBIC transistor of a small-signal process with every effect of the model on.

    Its quasi-saturation, its base, emitter, substrate and parasitic resistances, its
    three junctions' depletion, the base-emitter one smoothed and the others extended
    linearly past their forward bound, the weak avalanche, the parasitic substrate
    transistor, the Early and knee effects, the transit times with their bias
    dependence, the excess phase if `excess_phase`, and flicker noise are set off their
    defaults.
    """
    return devices.vbic(
        polarity=polarity,
        extrinsic_collector_resistance=10 * ohm,
        intrinsic_collector_resistance=20 * ohm,
        epi_saturation_voltage=2 * V,
        epi_doping_parameter=2e-11,
        high_current_collector_factor=2.0,
        extrinsic_base_resistance=10 * ohm,
        intrinsic_base_resistance=40 * ohm,
        emitter_resistance=2 * ohm,
        substrate_resistance=20 * ohm,
        parasitic_base_resistance=30 * ohm,
        saturation_current=1e-16 * A,
        reverse_emission_coefficient=1.01,
        base_emitter_overlap_capacitance=50 * fF,
        base_emitter_capacitance=0.5 * pF,
        base_emitter_potential=0.75 * V,
        base_emitter_grading=0.35,
        base_emitter_smoothing=0.2 * V,
        base_collector_overlap_capacitance=20 * fF,
        base_collector_capacitance=0.3 * pF,
        epi_charge=1e-8 * uC,
        extrinsic_base_collector_capacitance=0.1 * pF,
        base_collector_potential=0.7 * V,
        base_collector_grading=0.33,
        collector_substrate_capacitance=0.4 * pF,
        collector_substrate_potential=0.6 * V,
        collector_substrate_grading=0.4,
        ideal_base_emitter_current=1e-18 * A,
        intrinsic_base_emitter_fraction=0.8,
        nonideal_base_emitter_current=1e-15 * A,
        nonideal_base_emitter_emission_coefficient=2.0,
        ideal_base_collector_current=1e-17 * A,
        nonideal_base_collector_current=1e-14 * A,
        nonideal_base_collector_emission_coefficient=2.0,
        avalanche_coefficient=2 / V,
        avalanche_exponent_coefficient=15.0,
        parasitic_saturation_current=1e-18 * A,
        parasitic_ideal_base_emitter_current=1e-19 * A,
        parasitic_ideal_base_collector_current=1e-19 * A,
        forward_early_voltage=50 * V,
        reverse_early_voltage=5 * V,
        forward_knee_current=20 * mA,
        reverse_knee_current=2 * mA,
        parasitic_knee_current=1 * mA,
        forward_transit_time=0.01 * ns,
        transit_time_base_width_coefficient=0.1,
        transit_time_bias_coefficient=3.0,
        transit_time_voltage=2 * V,
        transit_time_current=50 * mA,
        reverse_transit_time=1 * ns,
        excess_phase_delay=0.002 * ns if excess_phase else None,
        flicker_coefficient=1e-14,
        flicker_exponent=1.0,
        flicker_frequency_exponent=1.0,
    )


VBIC_TRANSISTORS: Final = {"npn": _vbic("npn"), "pnp": _vbic("pnp", excess_phase=False)}
"""The VBIC transistor of each polarity, the pnp without excess phase.

ngspice 47 reads its excess-phase network's nodes without the polarity (`vbicload.c`,
lines 449 and 485 to 486, where every other branch voltage takes it, lines 419 to 443),
and multiplies the delayed transport current by it (line 1139), so its pnp with a delay
is not its npn's mirror: with `TD` alone set, its collector current at DC is 3.01 mA
where the npn's mirror is 56.7 µA. jolty's pnp is its npn's mirror, bit for bit.
"""


@component
def vbic_curves(c: Builder, transistor: devices.Device) -> None:
    """A VBIC transistor's curves: its base at `V1`, its collector at `V2`.

    Its substrate, at `V3`, lies 1 V behind its emitter in reverse.
    """
    ground = c.port("ground", Role.GROUND)
    sign = 1 if transistor.parameters.get("polarity", "npn") == "npn" else -1
    base = c.place(devices.voltage_source(sign * 0.8 * V), name="V1")
    collector = c.place(devices.voltage_source(sign * 2 * V), name="V2")
    substrate = c.place(devices.voltage_source(sign * -1 * V), name="V3")
    placed = c.place(transistor, name="Q1")
    c.connect(base.positive, placed.base, name="base")
    c.connect(collector.positive, placed.collector, name="collector")
    c.connect(substrate.positive, placed.substrate, name="substrate")
    c.connect(
        base.negative, collector.negative, substrate.negative, placed.emitter, ground
    )


@component
def vbic_stage(c: Builder, transistor: devices.Device) -> None:
    """A common-emitter stage of a VBIC npn, which steps up at 1 µs.

    Its base is fed through 10 kΩ from `V1` at 0.8 V, its collector loaded by 1 kΩ from
    5 V and by 1 pF, and its substrate grounded.
    """
    ground = c.port("ground", Role.GROUND)
    step = Piecewise([(1 * us, 0.8 * V), (1 * us + 5 * ns, 0.85 * V)])
    source = c.place(devices.voltage_source(step), name="V1")
    supply = c.place(devices.voltage_source(5 * V), name="V2")
    base = c.place(devices.resistor(10 * kohm), name="RB")
    load = c.place(devices.resistor(1 * kohm), name="RC")
    capacitance = c.place(devices.capacitor(1 * pF), name="CL")
    placed = c.place(transistor, name="Q1")
    c.connect(source.positive, base.a, name="in")
    c.connect(base.b, placed.base, name="base")
    c.connect(supply.positive, load.a, name="supply")
    c.connect(load.b, placed.collector, capacitance.a, name="out")
    c.connect(
        source.negative,
        supply.negative,
        capacitance.b,
        placed.emitter,
        placed.substrate,
        ground,
    )


def vbic_kelvins(temperature: Quantity) -> float:
    """The temperature to run ngspice at so its VBIC runs at jolty's `temperature`.

    VBIC computes its thermal voltage with constants of its own in both engines, so
    they agree at the same kelvins.
    """
    return ngspice.same_kelvins(temperature.value)


@pytest.mark.parametrize(
    "temperature",
    [celsius(27), *TEMPERATURES.values()],
    ids=["nominal", *TEMPERATURES.keys()],
)
@pytest.mark.parametrize("polarity", ["npn", "pnp"])
@pytest.mark.parametrize("source", ["V1", "V2"])
def test_vbic_curves_agree(polarity: str, source: str, temperature: Quantity) -> None:
    """A VBIC transistor's Gummel and output curves, in both modes, as ngspice's.

    The base sweeps from 0.2 V to 1 V; the collector from 0.5 V behind the emitter,
    the transistor reversed, to 3 V.
    """
    transistor = VBIC_TRANSISTORS[polarity]
    sign = 1 if polarity == "npn" else -1
    simulation = Simulation(vbic_curves(transistor), temperature)
    bounds = (0.2, 1.0) if source == "V1" else (-0.5, 3.0)
    levels = linear(sign * bounds[0] * V, sign * bounds[1] * V, 81)
    sweep = simulation.run(DcSweep(source, levels), TIGHT)
    first, last = float(sweep.swept[0]), float(sweep.swept[-1])
    step = (last - first) / (len(levels) - 1)
    command = f"dc V{source} {first!r} {last!r} {step!r}"
    raw = ngspice.run(
        netlist(simulation.circuit), command, temperature=vbic_kelvins(temperature)
    )
    for column, unknown in compared(simulation):
        spice_values = raw.real(vector(simulation.circuit, unknown))
        compare(
            sweep.values[:, column],
            spice_values,
            unknown,
            scale=size(spice_values),
            tolerance=1e-6,
        )


def test_vbic_ac_response_agrees() -> None:
    """A VBIC stage's operating point and AC response, as ngspice's."""
    simulation = Simulation(vbic_stage(VBIC_TRANSISTORS["npn"]))
    kelvins = vbic_kelvins(celsius(27))
    point = simulation.run(OperatingPoint(), TIGHT)
    raw = ngspice.run(netlist(simulation.circuit), "op", temperature=kelvins)
    for column, unknown in compared(simulation):
        spice_value = raw.real(vector(simulation.circuit, unknown))[0]
        compare(
            point.values[column],
            spice_value,
            unknown,
            scale=abs(spice_value),
            tolerance=1e-6,
        )
    response = simulation.run(Ac("V1", decade(10 * Hz, 10 * GHz, 10)), TIGHT)
    lines = netlist(simulation.circuit, excited="V1")
    raw = ngspice.run(lines, "ac dec 10 10 10g", temperature=kelvins)
    for column, unknown in compared(simulation):
        spice_values = raw.vectors[vector(simulation.circuit, unknown)]
        compare(
            response.values[:, column],
            spice_values,
            unknown,
            scale=size(spice_values),
            tolerance=1e-6,
        )


def test_vbic_noise_agrees() -> None:
    """A VBIC stage's noise at its output and its input, but one source's, as ngspice's.

    Its transistor's resistances make thermal noise, its currents shot noise, and its
    base currents flicker noise. ngspice 47 places the parasitic base resistance's
    thermal noise across the emitter resistance (`vbicnoise.c`, between `emitEINode` and
    `emitNode`), where jolty places it across the parasitic base resistance itself, so
    each engine's spectra are compared without that source's contribution, which
    ngspice gives apart when asked for every source's.
    """
    simulation = Simulation(vbic_stage(VBIC_TRANSISTORS["npn"]))
    spectrum = simulation.run(
        Noise("out", decade(1 * Hz, 100 * MHz, 5), source="V1"), TIGHT
    )
    lines = netlist(simulation.circuit, excited="V1")
    command = "noise v(out) VV1 dec 5 1 100meg 1\nsetplot noise1"
    raw = ngspice.run(lines, command, temperature=vbic_kelvins(celsius(27)))
    misplaced = next(
        index
        for index, source in enumerate(spectrum.sources)
        if (source.instance, source.name) == ("Q1", "parasitic base thermal")
    )
    densities = spectrum.densities - spectrum.contributions[misplaced]
    spice = raw.real("onoise_spectrum") ** 2
    left = 1 - raw.real("onoise_qq1_rbp") ** 2 / spice
    np.testing.assert_allclose(spectrum.frequencies, raw.real("frequency"), rtol=1e-9)
    np.testing.assert_allclose(np.sqrt(densities), np.sqrt(spice * left), rtol=1e-5)
    gains = np.abs(spectrum.input.gains) ** 2 if spectrum.input else 1.0
    np.testing.assert_allclose(
        np.sqrt(densities / gains),
        np.sqrt(raw.real("inoise_spectrum") ** 2 * left),
        rtol=1e-5,
    )


def test_vbic_transient_agrees() -> None:
    """A VBIC stage's step response, as ngspice's."""
    simulation = Simulation(vbic_stage(VBIC_TRANSISTORS["npn"]))
    stop, step = 2 * us, 1 * ns
    waveforms = simulation.run(Transient(stop=stop, max_step=step), TIGHT)
    command = f"tran {step.value!r} {stop.value!r} 0 {step.value!r}"
    raw = ngspice.run(
        netlist(simulation.circuit, transient=True),
        command,
        temperature=vbic_kelvins(celsius(27)),
    )
    times = raw.real("time")
    for column, unknown in compared(simulation):
        spice_values = raw.real(vector(simulation.circuit, unknown))
        jolty = closest(
            waveforms.times,
            waveforms.values[:, column],
            times,
            spice_values,
            step.value,
        )
        compare(jolty, spice_values, unknown, scale=size(spice_values), tolerance=1e-3)


HEATED_VBIC: Final = """\
* A VBIC npn heated through its fifth node, 500 K/W from the ambient, with its
* currents', resistances' and forward emission's temperature dependences set.
.model QHEAT NPN(level=4 is=1e-16 ibei=1e-18 iben=1e-15 nen=2 ikf=20m vef=50 ver=5
+ rcx=10 rci=20 rbx=10 rbi=40 re=2 cje=0.5p cjc=0.3p tf=10p rth=500 cth=10n
+ xis=3 ea=1.12 xii=3 xin=3 eaie=1.12 xre=1 xrb=1 xrbi=1 xrc=1 xrci=1 tnf=1e-4)
.subckt heated c b e
Q1 c b e 0 dt QHEAT
.ends
"""


@pytest.fixture(scope="module")
def heated_cards(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The library of `HEATED_VBIC`, as a file both engines read."""
    path = tmp_path_factory.mktemp("vbic") / "heated.lib"
    path.write_text(HEATED_VBIC, encoding="utf-8")
    return path


@component
def heated_stage(c: Builder, part: Component) -> None:
    """A library's transistor in a common-emitter stage, its substrate grounded.

    Its base is fed through 10 kΩ from `V1`, its collector loaded by 100 Ω from 5 V.
    """
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(0 * V), name="V1")
    supply = c.place(devices.voltage_source(5 * V), name="V2")
    base = c.place(devices.resistor(10 * kohm), name="RB")
    load = c.place(devices.resistor(100 * ohm), name="RC")
    placed = c.place(part, name="X1")
    c.connect(source.positive, base.a, name="in")
    c.connect(base.b, placed["b"], name="base")
    c.connect(supply.positive, load.a, name="supply")
    c.connect(load.b, placed["c"], name="out")
    c.connect(source.negative, supply.negative, placed["e"], placed["ground"], ground)


def test_a_heated_vbic_agrees(heated_cards: Path) -> None:
    """A VBIC heated by its card's `RTH`, read by the reader, as ngspice heats it.

    Its base drive sweeps from off to saturation, so it warms from the ambient by up to
    about 29 K. Its rise, ngspice's thermal node's voltage, is compared beside its nets;
    unheated, its output at 1.5 V lies 4.7 mV, a thousand tolerances, from its heated
    one.
    """
    read = spice.load(heated_cards).component("heated")
    simulation = Simulation(heated_stage(read))
    sweep = simulation.run(DcSweep("V1", linear(0 * V, 5 * V, 51)), TIGHT)
    raw = ngspice.run(
        netlist(simulation.circuit),
        "dc VV1 0 5 0.1",
        libraries=(heated_cards,),
        temperature=vbic_kelvins(celsius(27)),
    )
    for column, unknown in compared(simulation):
        # The netlist places the library's subcircuit as `XX1`, its thermal node in it.
        name = "v(xx1.dt)" if unknown.nature == "temperature" else None
        spice_values = raw.real(name or vector(simulation.circuit, unknown))
        compare(
            sweep.values[:, column],
            spice_values,
            unknown,
            scale=size(spice_values),
            tolerance=1e-6,
        )
    # The bench heats: both engines warming it alike is no vacuous agreement.
    rise = sweep.values[
        :, simulation.unknowns.index(results.Unknown("X1/dt", "temperature"))
    ]
    assert rise.max() > 25.0  # Most of its 29 K.


def test_a_run_states_its_time_and_stops_at_its_timeout() -> None:
    """The results of a run of ngspice carry its time, and a timeout stops it.

    A transient whose longest step is a picosecond takes ten million steps over
    10 µs, far longer than a hundredth of a second.
    """
    lines = ["rc", "R1 in out 1k", "C1 out 0 1n", "V1 in 0 PULSE(0 1 0 1n 1n 1u 2u)"]
    (point,) = ngspice.runs(lines, ["op"])
    assert point.seconds is not None
    assert point.seconds > 0
    with pytest.raises(ngspice.NgspiceTimeoutError, match=r"more than 0\.01 s"):
        ngspice.runs(lines, ["tran 1p 10u"], timeout=0.01)
