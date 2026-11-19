# Copyright 2026 Maxence Faldor
"""A buck converter: a switch, a diode and an LC filter stepping 12 V down to 6 V.

A switch closes for half of each 10 µs period, and the inductor carries the current on
through the diode while it is open, so the filter's output settles near the duty cycle
times the input, less the diode's share. With 100 µH and 100 µF, the LC filter rings at
1.6 kHz and its ripple is small. Simulate it with
``jolty simulate examples/buck_converter.py``.
"""

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.plan import Transient
from jolty.specs import mean, peak_to_peak
from jolty.units import A, V, ms, mV, nA, ns, ohm, uF, uH, us
from jolty.waveforms import Pulse

PERIOD = 10e-6
DUTY = 0.5


@component
def buck(c: Builder) -> None:
    """Builds the buck stage with input switch, catch diode, and LC output filter."""
    source, output, gate = c.port("input"), c.port("output"), c.port("gate", Role.INPUT)
    ground = c.port("ground", Role.GROUND)
    switch = c.place(
        devices.voltage_controlled_switch(
            on_resistance=0.05 * ohm,
            off_resistance=1e6 * ohm,
            on_voltage=4 * V,
            off_voltage=1 * V,
        ),
        name="S1",
    )
    catch = c.place(
        devices.diode(
            saturation_current=30 * nA,
            series_resistance=0.05 * ohm,
            emission_coefficient=1.0,
        ),
        name="D1",
    )
    inductor = c.place(devices.inductor(100 * uH), name="L1")
    smoothing = c.place(devices.capacitor(100 * uF), name="C1")
    c.connect(source, switch.positive)
    c.connect(gate, switch.control_positive)
    c.connect(switch.negative, catch.cathode, inductor.a, name="switched")
    c.connect(inductor.b, smoothing.a, output)
    c.connect(catch.anode, smoothing.b, switch.control_negative, ground)


@component
def bench(c: Builder) -> None:
    """Builds the buck converter switched at 100 kHz and 50 % duty into 10 Ω."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(12 * V), name="VIN")
    width = DUTY * PERIOD - 20e-9
    drive = Pulse(
        initial=0 * V,
        pulsed=5 * V,
        delay=0 * us,
        rise=10 * ns,
        fall=10 * ns,
        width=width * 1e6 * us,
        period=PERIOD * 1e6 * us,
    )
    clock = c.place(devices.voltage_source(drive), name="VG")
    converter = c.place(buck(), name="X1")
    load = c.place(devices.resistor(10 * ohm), name="RL")
    c.connect(supply.positive, converter.input, name="in")
    c.connect(clock.positive, converter.gate, name="gate")
    c.connect(converter.output, load.a, name="out")
    c.connect(supply.negative, clock.negative, converter.ground, load.b, ground)


def build() -> Component:
    """Returns the buck converter test bench component."""
    return bench()


def plan() -> Plan:
    """Returns a 6 ms transient plan covering LC settling and steady-state ripple."""
    return Plan({"start": Transient(stop=6 * ms, max_step=0.2 * us)})


def specs() -> Specs:
    """Returns specifications for steady output near 5.8 V, low ripple, and current."""
    return Specs(
        [
            mean("start", "out", start=5 * ms).between(5.5 * V, 6 * V),
            peak_to_peak("start", "out", start=5.9 * ms) <= 20 * mV,
            mean("start", "X1/L1:current", start=5 * ms).between(0.55 * A, 0.6 * A),
        ]
    )
