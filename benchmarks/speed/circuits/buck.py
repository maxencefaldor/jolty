# Copyright 2026 Maxence Faldor
"""Synchronous buck converter fixture measuring many-step switching transient cost.

Two level-1 MOSFETs switch 12 V onto an LC filter (10 µH, 47 µF into 5 Ω, settling
near 4.8 V) at 500 kHz. The upper switch conducts for 40 % of each period and the
lower switch conducts the remainder, with 20 ns dead time while the lower body diode
carries inductor current. The ``buck`` fixture runs a 2 ms transient (1,000 periods,
four edges per period) from the operating point.
"""

from fixture import Fixture, write

from jolty import Builder, Component, Plan, Role, component, devices
from jolty.plan import Transient
from jolty.units import V, mm, ms, nF, nm, ohm, uF, uH, um, us
from jolty.waveforms import Pulse

PERIOD = 2.0
"""Switching period in microseconds."""

DUTY = 0.4
"""Duty cycle fraction during which the upper switch is on, including edges."""

DEAD = 0.02
"""Dead time in microseconds between one switch turning off and the other turning
on."""

EDGE = 0.01
"""Gate-drive rise and fall duration in microseconds."""


def switch() -> devices.Device:
    """Creates a level-1 power NMOS switch (0.1 m wide, 1 µm long)."""
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
    """Creates a 10 V periodic gate-drive pulse active for `width` us from `delay`."""
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
    """Builds the half-bridge power switches, gate drivers, and output LC filter."""
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
    """Builds the 12 V to 5 Ω synchronous buck converter test bench."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(12 * V), name="VIN")
    stage = c.place(converter(), name="X1")
    load = c.place(devices.resistor(5 * ohm), name="RL")
    c.connect(supply.positive, stage.input, name="in")
    c.connect(stage.output, load.a, name="out")
    c.connect(supply.negative, stage.ground, load.b, ground)


def build() -> Component:
    """Returns the synchronous buck converter bench component."""
    return bench()


def plan() -> Plan:
    """Returns the 2 ms transient simulation plan for the buck converter."""
    return Plan({"tran": Transient(stop=2 * ms)})


def circuits() -> dict[str, Component]:
    """Returns the buck converter circuit keyed by output file stem."""
    return {"buck": build()}


def fixtures() -> dict[str, Fixture]:
    """Returns the ``buck`` speed benchmark fixture."""
    return {"buck": Fixture("buck", plan())}


if __name__ == "__main__":
    write(circuits(), fixtures())
