# Copyright 2026 Maxence Faldor
"""A boost converter: an inductor charged by a switch and dumped through a diode.

The switch shorts the inductor to ground for 60 % of each 10 µs period, storing energy
it then delivers through the diode on top of the input: the output settles near
Vin/(1 - D) = 12.5 V from 5 V, less the losses. Simulate it with
``jolty simulate examples/boost_converter.py``.
"""

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.plan import Transient
from jolty.specs import mean
from jolty.units import V, ms, nA, ns, ohm, uF, uH, us
from jolty.waveforms import Pulse

PERIOD = 10e-6
DUTY = 0.6


@component
def boost(c: Builder) -> None:
    """The inductor from the input, the switch to ground, the diode to the output."""
    source, output, gate = c.port("input"), c.port("output"), c.port("gate", Role.INPUT)
    ground = c.port("ground", Role.GROUND)
    inductor = c.place(devices.inductor(100 * uH), name="L1")
    switch = c.place(
        devices.voltage_controlled_switch(
            on_resistance=0.05 * ohm,
            off_resistance=1e6 * ohm,
            on_voltage=4 * V,
            off_voltage=1 * V,
        ),
        name="S1",
    )
    rectifier = c.place(
        devices.diode(saturation_current=30 * nA, series_resistance=0.05 * ohm),
        name="D1",
    )
    smoothing = c.place(devices.capacitor(100 * uF), name="C1")
    c.connect(source, inductor.a)
    c.connect(inductor.b, switch.positive, rectifier.anode, name="switched")
    c.connect(gate, switch.control_positive)
    c.connect(rectifier.cathode, smoothing.a, output)
    c.connect(switch.negative, switch.control_negative, smoothing.b, ground)


@component
def bench(c: Builder) -> None:
    """The converter, switched at 100 kHz and 60 %, into 100 Ω."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(5 * V), name="VIN")
    drive = Pulse(
        initial=0 * V,
        pulsed=5 * V,
        delay=0 * us,
        rise=10 * ns,
        fall=10 * ns,
        width=(DUTY * PERIOD - 20e-9) * 1e6 * us,
        period=PERIOD * 1e6 * us,
    )
    clock = c.place(devices.voltage_source(drive), name="VG")
    converter = c.place(boost(), name="X1")
    load = c.place(devices.resistor(100 * ohm), name="RL")
    c.connect(supply.positive, converter.input, name="in")
    c.connect(clock.positive, converter.gate, name="gate")
    c.connect(converter.output, load.a, name="out")
    c.connect(supply.negative, clock.negative, converter.ground, load.b, ground)


def build() -> Component:
    """The converter on its bench."""
    return bench()


def plan() -> Plan:
    """Ten milliseconds from rest."""
    return Plan({"start": Transient(stop=10 * ms, max_step=0.2 * us)})


def specs() -> Specs:
    """Near Vin/(1 - D), less the diode's drop and the switch's loss."""
    return Specs([mean("start", "out", start=9 * ms).between(11 * V, 12.5 * V)])
