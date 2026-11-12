# Copyright 2026 Maxence Faldor
"""The parts the examples share: an op-amp, transistors, diodes and a CMOS process.

The bipolar transistors and the diodes take the SPICE parameters their makers publish
for the 2N3904, the 2N3906 and the 1N4148. The op-amp is a macromodel of three
behavioral elements, written so ngspice reads it as jolty does: its input stage's
current saturates, which sets its slew rate, and its output clamps at its rails. The
CMOS transistors are level 1 MOSFETs of a 1 µm process.
"""

from jolty import Builder, Role, component, devices
from jolty.units import A, V, fF, nA, nm, ns, ohm, pF, ps, um

# Bipolar transistors and diodes.


def npn() -> devices.Device:
    """A 2N3904, the small-signal NPN transistor, with its maker's SPICE parameters."""
    return devices.bipolar_transistor(
        saturation_current=6.734e-15 * A,
        forward_beta=416.4,
        reverse_beta=0.7371,
        forward_emission_coefficient=1.0,
        forward_early_voltage=74.03 * V,
        forward_knee_current=66.78e-3 * A,
        base_emitter_leakage_current=6.734e-15 * A,
        base_emitter_leakage_emission_coefficient=1.259,
        base_collector_leakage_emission_coefficient=2.0,
        base_resistance=10 * ohm,
        collector_resistance=1 * ohm,
        base_emitter_capacitance=4.493 * pF,
        base_emitter_grading=0.2593,
        base_collector_capacitance=3.638 * pF,
        base_collector_grading=0.3085,
        forward_transit_time=301.2 * ps,
        transit_time_bias_coefficient=2.0,
        transit_time_voltage=4 * V,
        transit_time_current=0.4 * A,
        reverse_transit_time=239.5 * ns,
        beta_temperature_exponent=1.5,
    )


def pnp() -> devices.Device:
    """A 2N3906, the small-signal PNP transistor, with its maker's SPICE parameters."""
    return devices.bipolar_transistor(
        polarity="pnp",
        saturation_current=1.41e-15 * A,
        forward_beta=180.7,
        reverse_beta=4.977,
        forward_early_voltage=18.7 * V,
        forward_knee_current=80e-3 * A,
        base_emitter_leakage_emission_coefficient=1.5,
        base_collector_leakage_emission_coefficient=2.0,
        base_resistance=10 * ohm,
        collector_resistance=2.5 * ohm,
        base_emitter_capacitance=8.063 * pF,
        base_emitter_grading=0.3677,
        base_collector_capacitance=9.728 * pF,
        base_collector_grading=0.5776,
        forward_transit_time=179.3 * ps,
        transit_time_bias_coefficient=6.0,
        transit_time_voltage=4 * V,
        transit_time_current=0.4 * A,
        reverse_transit_time=33.42 * ns,
        beta_temperature_exponent=1.5,
    )


def signal_diode() -> devices.Device:
    """A 1N4148, the small-signal silicon diode, with its maker's SPICE parameters."""
    return devices.diode(
        saturation_current=2.52 * nA,
        emission_coefficient=1.752,
        series_resistance=0.568 * ohm,
        junction_capacitance=4 * pF,
        grading_coefficient=0.4,
        transit_time=20 * ns,
    )


def zener(voltage: float) -> devices.Device:
    """A zener diode that breaks down at `voltage` volts, 5 mA at its knee."""
    return devices.diode(
        saturation_current=1e-14 * A,
        series_resistance=2 * ohm,
        junction_capacitance=100 * pF,
        breakdown_voltage=voltage * V,
        breakdown_current=5e-3 * A,
    )


# The op-amp.

SWING = 13.0
"""The op-amp's rails, in volts either side of ground."""


@component
def opamp(c: Builder) -> None:
    """A macromodel of an op-amp: 100 dB of gain, 1 MHz of bandwidth, 13 V/µs, ±13 V.

    The input stage is a current of 130 µA times the hyperbolic tangent of the input
    over 2.07 V, so its transconductance is 62.8 µS, into a pole of 1.59 GΩ and 10 pF:
    a gain of 10⁵ that falls from 10 Hz, crossing one at 1 MHz, and a slew rate of the
    current over the capacitance. A conductance of 1 mS clamps the pole within the
    rails, and a buffer drives the output from it.
    """
    noninverting = c.port("inp", Role.INPUT)
    inverting = c.port("inn", Role.INPUT)
    output = c.port("out", Role.OUTPUT)
    ground = c.port("ground", Role.GROUND)
    stage = c.place(
        devices.behavioral_current_source("130e-6 * tanh(v(p, n) / 2.07)"), id="G1"
    )
    load = c.place(devices.resistor(1.59e9 * ohm), id="R1")
    pole = c.place(devices.capacitor(10 * pF), id="C1")
    clamp = c.place(
        devices.behavioral_current_source(
            f"1e-3 * (uramp(v(x) - {SWING}) - uramp(-v(x) - {SWING}))"
        ),
        id="G2",
    )
    buffer = c.place(devices.voltage_controlled_voltage_source(1.0), id="E1")
    c.connect(stage.p, noninverting)
    c.connect(stage.n, inverting)
    c.connect(
        stage.negative,
        load.a,
        pole.a,
        clamp.positive,
        clamp.x,
        buffer.control_positive,
        name="pole",
    )
    c.connect(buffer.positive, output)
    c.connect(
        stage.positive,
        load.b,
        pole.b,
        clamp.negative,
        buffer.negative,
        buffer.control_negative,
        ground,
    )


# A CMOS process.

SUPPLY = 3.3
"""The CMOS process's supply, in volts."""


def mos(channel: str, width: float = 1.0) -> devices.Device:
    """A level 1 MOSFET of a 1 µm process, `width` µm wide, its p-channel's twice."""
    p = channel == "p"
    return devices.mosfet(
        channel="p" if p else "n",
        width=(2 if p else 1) * width * um,
        length=1 * um,
        threshold_voltage=(-0.7 if p else 0.7) * V,
        channel_length_modulation=0.05 / V,
        oxide_thickness=20 * nm,
        bulk_drain_capacitance=2 * fF,
        bulk_source_capacitance=2 * fF,
    )


@component
def inverter(c: Builder) -> None:
    """A CMOS inverter: a p-channel transistor up to the supply, an n-channel down."""
    a, y = c.port("a", Role.INPUT), c.port("y", Role.OUTPUT)
    vdd, gnd = c.port("vdd", Role.SUPPLY), c.port("gnd", Role.GROUND)
    up = c.place(mos("p"), id="P1")
    down = c.place(mos("n"), id="N1")
    c.connect(up.source, up.bulk, vdd)
    c.connect(up.gate, down.gate, a)
    c.connect(up.drain, down.drain, y)
    c.connect(down.source, down.bulk, gnd)
