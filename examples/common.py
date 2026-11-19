# Copyright 2026 Maxence Faldor
"""What the examples share beyond `jolty.parts`: zeners, an op-amp, a JFET and CMOS.

The op-amp is a macromodel of three behavioral elements, written so ngspice reads it as
jolty does: its input stage's current saturates, which sets its slew rate, and its
output clamps at its rails. The CMOS transistors are level 1 MOSFETs of a 1 µm process.
"""

from jolty import Builder, Role, component, devices
from jolty.units import A, K, V, fF, nm, ohm, pF, um

# Zener diodes.


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
        devices.behavioral_current_source("130e-6 * tanh(v(p, n) / 2.07)"), name="G1"
    )
    load = c.place(devices.resistor(1.59e9 * ohm), name="R1")
    pole = c.place(devices.capacitor(10 * pF), name="C1")
    clamp = c.place(
        devices.behavioral_current_source(
            f"1e-3 * (uramp(v(x) - {SWING}) - uramp(-v(x) - {SWING}))"
        ),
        name="G2",
    )
    buffer = c.place(devices.voltage_controlled_voltage_source(1.0), name="E1")
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
    up = c.place(mos("p"), name="P1")
    down = c.place(mos("n"), name="N1")
    c.connect(up.source, up.bulk, vdd)
    c.connect(up.gate, down.gate, a)
    c.connect(up.drain, down.drain, y)
    c.connect(down.source, down.bulk, gnd)


def comparator(supply: float = SUPPLY) -> devices.Device:
    """A comparator, its output `supply` or ground as `in` is above `ref` or not."""
    half = supply / 2
    return devices.behavioral_voltage_source(
        f"{half} + {half} * tanh(v(in, ref) / 1e-3)"
    )


def j2n5457() -> devices.Device:
    """A 2N5457 JFET, with the SPICE parameters its maker publishes."""
    return devices.jfet(
        threshold_voltage=-1.372 * V,
        transconductance_parameter=1.125e-3 * A / V**2,
        channel_length_modulation=2.3e-3 / V,
        drain_resistance=1 * ohm,
        source_resistance=1 * ohm,
        gate_source_capacitance=2.06 * pF,
        gate_drain_capacitance=2.16 * pF,
        saturation_current=181.3e-15 * A,
        threshold_voltage_temperature_coefficient=-2.5e-3 * V / K,
        transconductance_parameter_temperature_exponent=-0.5,
    )
