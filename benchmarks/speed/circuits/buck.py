# Copyright 2026 Maxence Faldor
"""A synchronous buck converter, the cost of many tiny time steps.

Two level 1 MOSFETs switch the input's 12 V onto an LC filter at 500 kHz, the upper
one 40 % of each period and the lower one the rest, each turning on 20 ns after the
other turns off, while the lower one's body diode carries the inductor's current. The
filter's 10 µH and 47 µF settle into 5 Ω near 4.8 V. The fixture ``buck`` runs 2 ms of
it from the operating point: a thousand periods, each with four edges.
"""

from fixture import Fixture, write

from jolty import Builder, Component, Plan, Role, component, devices
from jolty.plan import Transient
from jolty.units import V, mm, ms, nF, nm, ohm, uF, uH, um, us
from jolty.waveforms import Pulse

PERIOD = 2.0
"""The switching period, in microseconds."""

DUTY = 0.4
"""The fraction of each period the upper switch is on, its edges included."""

DEAD = 0.02
"""The time between one switch turning off and the other turning on, in
microseconds."""

EDGE = 0.01
"""The gate drives' rise and fall, in microseconds."""


def switch() -> devices.Device:
    """An n-channel power MOSFET of level 1: 0.1 m wide and 1 µm long."""
    return devices.mosfet(
        channel="n",
        width=100 * mm,
        length=1 * um,
        threshold_voltage=2 * V,
        channel_length_modulation=0.01 / V,
        oxide_thickness=50 * nm,
        drain_resistance=0.01 * ohm,
        source_resistance=0.01 * ohm,
        bulk_drain_capacitance=1 * nF,
        bulk_source_capacitance=1 * nF,
    )


def drive(delay: float, width: float) -> Pulse:
    """A 10 V gate drive on for `width` microseconds of each period from `delay`."""
    return Pulse(
        initial=0 * V,
        pulsed=10 * V,
        delay=delay * us,
        rise=EDGE * us,
        fall=EDGE * us,
        width=(width - EDGE) * us,
        period=PERIOD * us,
    )


@component
def converter(c: Builder) -> None:
    """The two switches, each driven from its own source, and the LC filter."""
    source, output = c.port("input", Role.SUPPLY), c.port("output", Role.OUTPUT)
    ground = c.port("ground", Role.GROUND)
    upper, lower = c.place(switch(), name="M1"), c.place(switch(), name="M2")
    on = DUTY * PERIOD
    high = c.place(devices.voltage_source(drive(0, on)), name="VH")
    low_width = PERIOD - on - 2 * DEAD
    low = c.place(devices.voltage_source(drive(on + DEAD, low_width)), name="VL")
    inductor = c.place(devices.inductor(10 * uH), name="L1")
    smoothing = c.place(devices.capacitor(47 * uF), name="C1")
    c.connect(source, upper.drain)
    c.connect(high.positive, upper.gate, name="high")
    c.connect(low.positive, lower.gate, name="low")
    c.connect(
        upper.source,
        upper.bulk,
        high.negative,
        lower.drain,
        inductor.a,
        name="switched",
    )
    c.connect(inductor.b, smoothing.a, output)
    c.connect(lower.source, lower.bulk, low.negative, smoothing.b, ground)


@component
def bench(c: Builder) -> None:
    """The converter from 12 V into 5 Ω."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(12 * V), name="VIN")
    stage = c.place(converter(), name="X1")
    load = c.place(devices.resistor(5 * ohm), name="RL")
    c.connect(supply.positive, stage.input, name="in")
    c.connect(stage.output, load.a, name="out")
    c.connect(supply.negative, stage.ground, load.b, ground)


def build() -> Component:
    """The converter on its bench."""
    return bench()


def plan() -> Plan:
    """Two milliseconds from the operating point."""
    return Plan({"tran": Transient(stop=2 * ms)})


def circuits() -> dict[str, Component]:
    """The converter's one circuit, by the name of its file."""
    return {"buck": build()}


def fixtures() -> dict[str, Fixture]:
    """The converter's one fixture."""
    return {"buck": Fixture("buck", plan())}


if __name__ == "__main__":
    write(circuits(), fixtures())
