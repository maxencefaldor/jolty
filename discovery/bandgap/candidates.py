# Copyright 2026 Maxence Faldor
"""References without an op-amp, each a component of ports vcc, out and ground."""

from jolty import Builder, Component, Role, component, devices
from jolty.parts import d1n4148, q2n3904
from jolty.units import A, K, V, ohm, pF


@component
def diode(c: Builder, bias: float = 4.3e3) -> None:
    """A forward-biased diode: one Vbe, falling 2 mV/K."""
    vcc, out = c.port("vcc", Role.SUPPLY), c.port("out", Role.OUTPUT)
    ground = c.port("ground", Role.GROUND)
    feed = c.place(devices.resistor(bias * ohm), name="R1")
    junction = c.place(d1n4148(), name="D1")
    c.connect(vcc, feed.a)
    c.connect(feed.b, junction.anode, out)
    c.connect(junction.cathode, ground)


@component
def widlar(
    c: Builder, top: float = 600.0, widlar_r: float = 60.0, ptat: float = 600.0
) -> None:
    """Widlar's bandgap: a CTAT voltage and a PTAT one, summed.

    Q1 is diode-connected, Q2 a Widlar source of it whose emitter resistor takes
    Vt·ln(I1/I2), and Q3 holds the output at its own Vbe plus Q2's current across
    `ptat`.
    """
    vcc, out = c.port("vcc", Role.SUPPLY), c.port("out", Role.OUTPUT)
    ground = c.port("ground", Role.GROUND)
    first, second, third = (c.place(q2n3904(), name=f"Q{n}") for n in (1, 2, 3))
    feed = c.place(devices.resistor(1e3 * ohm), name="RB")
    r1 = c.place(devices.resistor(top * ohm), name="R1")
    r2 = c.place(devices.resistor(widlar_r * ohm), name="R2")
    r3 = c.place(devices.resistor(ptat * ohm), name="R3")
    c.connect(vcc, feed.a)
    c.connect(feed.b, r1.a, r3.a, third.collector, out)
    c.connect(r1.b, first.collector, first.base, second.base, name="diode")
    c.connect(r3.b, second.collector, third.base, name="sense")
    c.connect(second.emitter, r2.a, name="widlar")
    c.connect(first.emitter, r2.b, third.emitter, ground)


@component
def bench(c: Builder, reference: Component, supply: float = 5.0) -> None:
    """The reference on its supply, loaded by 100 kΩ."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(supply * V), name="VCC")
    placed = c.place(reference, name="X1")
    load = c.place(devices.resistor(100e3 * ohm), name="RL")
    c.connect(source.positive, placed.vcc, name="vcc")
    c.connect(placed.out, load.a, name="out")
    c.connect(source.negative, placed.ground, load.b, ground)


def jfet_source() -> devices.Device:
    """A 2N5457 JFET, its gate on its source: a current source of its Idss."""
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


@component
def widlar_jfet(
    c: Builder, top: float = 600.0, widlar_r: float = 500.0, ptat: float = 6000.0
) -> None:
    """Widlar's bandgap fed by a JFET's Idss instead of a resistor.

    The current it takes, and so its output, barely follow the supply.
    """
    vcc, out = c.port("vcc", Role.SUPPLY), c.port("out", Role.OUTPUT)
    ground = c.port("ground", Role.GROUND)
    first, second, third = (c.place(q2n3904(), name=f"Q{n}") for n in (1, 2, 3))
    feed = c.place(jfet_source(), name="J1")
    r1 = c.place(devices.resistor(top * ohm), name="R1")
    r2 = c.place(devices.resistor(widlar_r * ohm), name="R2")
    r3 = c.place(devices.resistor(ptat * ohm), name="R3")
    c.connect(vcc, feed.drain)
    c.connect(feed.gate, feed.source, r1.a, r3.a, third.collector, out)
    c.connect(r1.b, first.collector, first.base, second.base, name="diode")
    c.connect(r3.b, second.collector, third.base, name="sense")
    c.connect(second.emitter, r2.a, name="widlar")
    c.connect(first.emitter, r2.b, third.emitter, ground)
