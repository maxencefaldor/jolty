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
from collections.abc import Callable
from typing import Literal

import numpy as np
import pytest

from differential import (
    SKIP,
    TIGHT,
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
    OperatingPoint,
    Role,
    Simulation,
    Transient,
    component,
    devices,
)
from jolty.plan import decade, linear
from jolty.units import (
    A,
    F,
    Hz,
    MHz,
    Mohm,
    Quantity,
    V,
    fA,
    fF,
    kHz,
    kohm,
    m,
    mA,
    mH,
    ms,
    mV,
    nA,
    nF,
    nm,
    ns,
    ohm,
    pF,
    s,
    uA,
    uF,
    um,
    uS,
    us,
)
from jolty.waveforms import Piecewise, Pulse, Sine

pytestmark = SKIP

# The corpus.


@component
def divider(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(5 * V), id="V1")
    upper = c.place(devices.resistor(1 * kohm), id="R1")
    lower = c.place(devices.resistor(3 * kohm), id="R2")
    c.connect(source.positive, upper.a, name="in")
    c.connect(upper.b, lower.a, name="out")
    c.connect(lower.b, source.negative, ground)


@component
def low_pass(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    step = Piecewise([(1 * us, 0 * V), (1 * us + 10 * ns, 1 * V)])
    source = c.place(devices.voltage_source(step), id="V1")
    resistor = c.place(devices.resistor(1 * kohm), id="R1")
    capacitor = c.place(devices.capacitor(1 * nF), id="C1")
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
    source = c.place(devices.voltage_source(pulse), id="V1")
    resistor = c.place(devices.resistor(20 * ohm), id="R1")
    inductor = c.place(devices.inductor(1 * mH), id="L1")
    capacitor = c.place(devices.capacitor(10 * nF), id="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, inductor.a, name="mid")
    c.connect(inductor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)


@component
def clipper(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(0 * V), id="V1")
    resistor = c.place(devices.resistor(1 * kohm), id="R1")
    diode = c.place(
        devices.diode(saturation_current=2.52 * nA, emission_coefficient=1.752),
        id="D1",
    )
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, diode.anode, name="out")
    c.connect(diode.cathode, source.negative, ground)


@component
def zener_clamp(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(0 * V), id="V1")
    resistor = c.place(devices.resistor(1 * kohm), id="R1")
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
        id="D1",
    )
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, zener.anode, name="out")
    c.connect(zener.cathode, source.negative, ground)


@component
def rectifier(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    mains = Sine(offset=0 * V, amplitude=5 * V, frequency=1 * kHz, phase=0.3)
    source = c.place(devices.voltage_source(mains), id="V1")
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
        id="D1",
    )
    smoothing = c.place(devices.capacitor(1 * uF), id="C1")
    load = c.place(devices.resistor(1 * kohm), id="R1")
    c.connect(source.positive, diode.anode, name="in")
    c.connect(diode.cathode, smoothing.a, load.a, name="out")
    c.connect(smoothing.b, load.b, source.negative, ground)


@component
def amplifier(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(0.1 * V), id="V1")
    gain = c.place(devices.voltage_controlled_voltage_source(10.0), id="E1")
    coupling = c.place(devices.resistor(1 * kohm), id="R1")
    stage = c.place(devices.voltage_controlled_current_source(500 * uS), id="G1")
    load = c.place(devices.resistor(2 * kohm), id="R2")
    bypass = c.place(devices.capacitor(1 * nF), id="C1")
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
    source = c.place(devices.current_source(drive), id="I1")
    inductor = c.place(devices.inductor(10 * mH), id="L1")
    damping = c.place(devices.resistor(5 * kohm), id="R1")
    capacitor = c.place(devices.capacitor(1 * nF), id="C1")
    c.connect(source.negative, inductor.a, damping.a, capacitor.a, name="out")
    c.connect(source.positive, inductor.b, damping.b, capacitor.b, ground)


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
    supply = c.place(devices.voltage_source(12 * V), id="V2")
    # Starting at its peak, the sine's slope is zero, so the operating point at zero
    # time is also the transient's start, with no capacitive current to jump.
    signal = Sine(
        offset=-0.8 * V, amplitude=0.6 * V, frequency=200 * kHz, phase=math.pi / 2
    )
    source = c.place(devices.voltage_source(signal), id="V1")
    gate = c.place(devices.resistor(1 * kohm), id="RG")
    transistor = c.place(_JFET, id="J1")
    drain = c.place(devices.resistor(2.2 * kohm), id="RD")
    degeneration = c.place(devices.resistor(470 * ohm), id="RS")
    c.connect(source.positive, gate.a, name="in")
    c.connect(gate.b, transistor.gate, name="gate")
    c.connect(supply.positive, drain.a, name="supply")
    c.connect(drain.b, transistor.drain, name="out")
    c.connect(transistor.source, degeneration.a, name="tail")
    c.connect(source.negative, supply.negative, degeneration.b, ground)


@component
def jfet_curves(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    bias = c.place(devices.voltage_source(-0.5 * V), id="V1")
    drain = c.place(devices.voltage_source(0 * V), id="V2")
    transistor = c.place(_JFET, id="J1")
    c.connect(bias.positive, transistor.gate, name="gate")
    c.connect(drain.positive, transistor.drain, name="drain")
    c.connect(bias.negative, drain.negative, transistor.source, ground)


@component
def p_jfet_follower(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(-9 * V), id="V2")
    source = c.place(devices.voltage_source(-1 * V), id="V1")
    transistor = c.place(
        devices.jfet(
            channel="p",
            threshold_voltage=-2.2 * V,
            transconductance_parameter=0.9 * mA / V**2,
            channel_length_modulation=0.01 / V,
            gate_source_capacitance=3 * pF,
            gate_drain_capacitance=1 * pF,
        ),
        id="J1",
    )
    load = c.place(devices.resistor(3.3 * kohm), id="RL")
    c.connect(source.positive, transistor.gate, name="in")
    c.connect(supply.positive, transistor.drain, name="supply")
    c.connect(transistor.source, load.a, name="out")
    c.connect(source.negative, supply.negative, load.b, ground)


def _bipolar(
    polarity: Literal["npn", "pnp"],
    base_resistance: Literal["current", "charge", "fixed"],
    substrate: Literal["vertical", "lateral"] | None = None,
) -> devices.Device:
    """A small-signal transistor with every effect of the model switched on.

    With a `substrate`, so are its substrate junction's. Its base resistance falls with
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
        minimum_base_resistance=None if base_resistance == "fixed" else 10 * ohm,
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
    supply = c.place(devices.voltage_source(5 * V), id="V2")
    source = c.place(devices.voltage_source(0.7 * V), id="V1")
    base = c.place(devices.resistor(10 * kohm), id="RB")
    transistor = c.place(_bipolar("npn", "current"), id="Q1")
    load = c.place(devices.resistor(1 * kohm), id="RC")
    c.connect(source.positive, base.a, name="in")
    c.connect(base.b, transistor.base, name="base")
    c.connect(supply.positive, load.a, name="supply")
    c.connect(load.b, transistor.collector, name="out")
    c.connect(source.negative, supply.negative, transistor.emitter, ground)


@component
def bjt_amplifier(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(12 * V), id="V2")
    signal = Sine(offset=0 * V, amplitude=20 * mV, frequency=1 * MHz, phase=math.pi / 2)
    source = c.place(devices.voltage_source(signal), id="V1")
    coupling = c.place(devices.capacitor(100 * nF), id="CIN")
    upper = c.place(devices.resistor(47 * kohm), id="R1")
    lower = c.place(devices.resistor(10 * kohm), id="R2")
    transistor = c.place(_bipolar("npn", "fixed"), id="Q1")
    load = c.place(devices.resistor(3.3 * kohm), id="RC")
    degeneration = c.place(devices.resistor(1 * kohm), id="RE")
    bypass = c.place(devices.capacitor(1 * nF), id="CE")
    c.connect(source.positive, coupling.a, name="in")
    c.connect(coupling.b, upper.b, lower.a, transistor.base, name="base")
    c.connect(supply.positive, upper.a, load.a, name="supply")
    c.connect(load.b, transistor.collector, name="out")
    c.connect(transistor.emitter, degeneration.a, bypass.a, name="tail")
    c.connect(
        source.negative, supply.negative, lower.b, degeneration.b, bypass.b, ground
    )


@component
def integrated_stage(c: Builder) -> None:
    """A vertical NPN loaded by a lateral PNP, on a substrate fed through a resistor.

    The source feeding the substrate drives the transistors' junctions to it forward.
    """
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(5 * V), id="V2")
    signal = Sine(
        offset=0.7 * V, amplitude=10 * mV, frequency=1 * MHz, phase=math.pi / 2
    )
    source = c.place(devices.voltage_source(signal), id="V1")
    bias = c.place(devices.voltage_source(-1 * V), id="V3")
    base = c.place(devices.resistor(10 * kohm), id="RB")
    amplifier = c.place(_bipolar("npn", "fixed", "vertical"), id="Q1")
    load = c.place(_bipolar("pnp", "fixed", "lateral"), id="Q2")
    feed = c.place(devices.resistor(10 * kohm), id="RS")
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
    supply = c.place(devices.voltage_source(10 * V), id="V1")
    reference = c.place(devices.resistor(10 * kohm), id="RR")
    left = c.place(_bipolar("pnp", "charge"), id="Q1")
    right = c.place(_bipolar("pnp", "charge"), id="Q2")
    load = c.place(devices.resistor(4.7 * kohm), id="RL")
    c.connect(supply.positive, left.emitter, right.emitter, name="supply")
    c.connect(left.base, left.collector, right.base, reference.a, name="mirror")
    c.connect(right.collector, load.a, name="out")
    c.connect(supply.negative, reference.b, load.b, ground)


@component
def current_controlled(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(2 * V), id="V1")
    feed = c.place(devices.resistor(1 * kohm), id="R1")
    amplifier = c.place(devices.current_controlled_current_source(3.0), id="F1")
    load = c.place(devices.resistor(2 * kohm), id="R2")
    transducer = c.place(devices.current_controlled_voltage_source(500 * ohm), id="H1")
    output = c.place(devices.resistor(4.7 * kohm), id="R3")
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


def _level_three(max_drift_velocity: Quantity | None) -> devices.Device:
    """A level 3 n-channel with every effect of its law, and velocity saturation."""
    return devices.mosfet(
        level=3,
        width=50 * um,
        length=2 * um,
        lateral_diffusion=0.2 * um,
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
        saturation_field_factor=0.4,
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
    supply = c.place(devices.voltage_source(5 * V), id="V2")
    signal = Sine(
        offset=1.4 * V, amplitude=0.4 * V, frequency=2 * MHz, phase=math.pi / 2
    )
    source = c.place(devices.voltage_source(signal), id="V1")
    gate = c.place(devices.resistor(100 * kohm), id="RG")
    transistor = c.place(_level_three(1.5e5 * m / s), id="M1")
    drain = c.place(devices.resistor(4.7 * kohm), id="RD")
    c.connect(source.positive, gate.a, name="in")
    c.connect(gate.b, transistor.gate, name="gate")
    c.connect(supply.positive, drain.a, name="supply")
    c.connect(drain.b, transistor.drain, name="out")
    c.connect(
        source.negative, supply.negative, transistor.source, transistor.bulk, ground
    )


@component
def level_three_curves(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    bias = c.place(devices.voltage_source(1.5 * V), id="V1")
    drain = c.place(devices.voltage_source(0 * V), id="V2")
    body = c.place(devices.voltage_source(-1 * V), id="V3")
    transistor = c.place(_level_three(None), id="M1")
    c.connect(bias.positive, transistor.gate, name="gate")
    c.connect(drain.positive, transistor.drain, name="drain")
    c.connect(body.positive, transistor.bulk, name="body")
    c.connect(bias.negative, drain.negative, body.negative, transistor.source, ground)


@component
def nmos_stage(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(9 * V), id="V2")
    signal = Sine(
        offset=1.6 * V, amplitude=0.3 * V, frequency=2 * MHz, phase=math.pi / 2
    )
    source = c.place(devices.voltage_source(signal), id="V1")
    gate = c.place(devices.resistor(100 * kohm), id="RG")
    transistor = c.place(_NMOS, id="M1")
    drain = c.place(devices.resistor(4.7 * kohm), id="RD")
    degeneration = c.place(devices.resistor(1 * kohm), id="RS")
    c.connect(source.positive, gate.a, name="in")
    c.connect(gate.b, transistor.gate, name="gate")
    c.connect(supply.positive, drain.a, name="supply")
    c.connect(drain.b, transistor.drain, name="out")
    c.connect(transistor.source, degeneration.a, name="tail")
    c.connect(source.negative, supply.negative, degeneration.b, transistor.bulk, ground)


@component
def mosfet_curves(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    bias = c.place(devices.voltage_source(2 * V), id="V1")
    drain = c.place(devices.voltage_source(0 * V), id="V2")
    body = c.place(devices.voltage_source(-3 * V), id="V3")
    transistor = c.place(_NMOS, id="M1")
    c.connect(bias.positive, transistor.gate, name="gate")
    c.connect(drain.positive, transistor.drain, name="drain")
    c.connect(body.positive, transistor.bulk, name="body")
    c.connect(bias.negative, drain.negative, body.negative, transistor.source, ground)


@component
def cmos_inverter(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(3.3 * V), id="V2")
    pulse = Pulse(
        initial=0 * V,
        pulsed=3.3 * V,
        delay=10 * ns,
        rise=10 * ns,
        fall=10 * ns,
        width=60 * ns,
        period=160 * ns,
    )
    source = c.place(devices.voltage_source(pulse), id="V1")
    # A driver's resistance, so the gates' charging current flows without jumps.
    driver = c.place(devices.resistor(10 * kohm), id="RIN")
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
        id="M1",
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
        id="M2",
    )
    load = c.place(devices.capacitor(200 * fF), id="CL")
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
    source = c.place(devices.voltage_source(signal), id="V1")
    upper = c.place(devices.resistor(1 * kohm), id="R1")
    lower = c.place(devices.resistor(1 * kohm), id="R2")
    shaper = c.place(
        devices.behavioral_voltage_source(
            "2 * tanh(v(x) / 0.5) + 0.1 * v(x) ^ 2 + 0.05 * sin(2 * pi * 1e4 * time)"
        ),
        id="B1",
    )
    series = c.place(devices.resistor(1 * kohm), id="R3")
    load = c.place(
        devices.behavioral_current_source("v(y) > 0 ? 2e-3 * v(y) : 1e-4 * v(y)"),
        id="B2",
    )
    smoothing = c.place(devices.capacitor(100 * nF), id="C1")
    sink = c.place(devices.resistor(2 * kohm), id="R4")
    mirror = c.place(
        devices.behavioral_current_source("0.5 * i(s) + v(y) * i(s)"), id="B3"
    )
    output = c.place(devices.resistor(1 * kohm), id="R5")
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
    control = c.place(devices.voltage_source(pulse), id="V1")
    supply = c.place(devices.voltage_source(5 * V), id="V2")
    feed = c.place(devices.resistor(1 * kohm), id="R1")
    switch = c.place(
        devices.voltage_controlled_switch(
            on_resistance=10 * ohm,
            off_resistance=1 * Mohm,
            on_voltage=2 * V,
            off_voltage=1 * V,
        ),
        id="S1",
    )
    hold = c.place(devices.capacitor(10 * nF), id="C1")
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
        nmos_stage,
        mosfet_curves,
        level_three_stage,
        level_three_curves,
        cmos_inverter,
        behavioral,
        switched,
    ],
)
def test_operating_points_agree(top: Callable[[], Component]) -> None:
    simulation = Simulation(top())
    point = simulation.run(OperatingPoint(), TIGHT)
    raw = ngspice(netlist(simulation.circuit), "op")
    for column, unknown in compared(simulation):
        spice = raw.real(vector(simulation.circuit, unknown))[0]
        compare(point.values[column], spice, unknown, scale=abs(spice), tolerance=1e-6)


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
        (cmos_inverter, "V1", linear(0 * V, 3.3 * V, 67)),
        (behavioral, "V1", linear(-3 * V, 3 * V, 61)),
        (switched, "V1", linear(0 * V, 3 * V, 61)),
    ],
)
def test_dc_sweeps_agree(
    top: Callable[[], Component], source: str, levels: list[Quantity]
) -> None:
    simulation = Simulation(top())
    sweep = simulation.run(DcSweep(source, levels), TIGHT)
    first, last = float(sweep.levels[0]), float(sweep.levels[-1])
    step = (last - first) / (len(levels) - 1)
    command = f"dc V{source} {first!r} {last!r} {step!r}"
    raw = ngspice(netlist(simulation.circuit), command)
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
        behavioral,
        switched,
    ],
)
def test_ac_responses_agree(top: Callable[[], Component]) -> None:
    simulation = Simulation(top())
    response = simulation.run(Ac("V1", decade(10 * Hz, 10 * MHz, 20)), TIGHT)
    lines = netlist(simulation.circuit, excited="V1")
    raw = ngspice(lines, "ac dec 20 10 10meg")
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
    ("top", "stop", "step"),
    [
        (low_pass, 8 * us, 10 * ns),
        (resonator, 60 * us, 50 * ns),
        (rectifier, 3 * ms, 1 * us),
        (driven_tank, 40 * us, 20 * ns),
        (jfet_stage, 15 * us, 10 * ns),
        (bjt_amplifier, 4 * us, 2 * ns),
        (integrated_stage, 4 * us, 2 * ns),
        (nmos_stage, 2 * us, 1 * ns),
        (level_three_stage, 2 * us, 1 * ns),
        (cmos_inverter, 160 * ns, 0.01 * ns),
        (behavioral, 300 * us, 0.5 * us),
        (switched, 200 * us, 0.2 * us),
    ],
)
def test_transients_agree(
    top: Callable[[], Component], stop: Quantity, step: Quantity
) -> None:
    simulation = Simulation(top())
    waveforms = simulation.run(Transient(stop=stop, max_step=step), TIGHT)
    command = f"tran {step.value!r} {stop.value!r} 0 {step.value!r}"
    raw = ngspice(netlist(simulation.circuit), command)
    times = raw.real("time")
    for column, unknown in compared(simulation):
        spice = raw.real(vector(simulation.circuit, unknown))
        jolty = closest(
            waveforms.times, waveforms.values[:, column], times, spice, step.value
        )
        compare(jolty, spice, unknown, scale=size(spice), tolerance=1e-3)
