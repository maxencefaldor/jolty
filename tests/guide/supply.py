# Copyright 2026 Maxence Faldor
"""A capacitor rated 16 V ramped to 20 V, and a transistor heated past 150 °C.

The ramp reaches 20 V across `C1` at 2 µs, and `Q1`, which dissipates about 0.55 W
through a thermal resistance of 240 K/W, rests near 160 °C: `over-rating`, twice.
"""

from jolty import Builder, Component, Plan, Role, component, devices
from jolty.plan import OperatingPoint, Transient
from jolty.units import K, V, W, celsius, kohm, nF, ohm, us
from jolty.waveforms import Piecewise


@component
def supply(c: Builder) -> None:
    """Builds the ramped capacitor and heated 10 V bipolar transistor bench."""
    ground = c.port("ground", Role.GROUND)
    ramp = Piecewise([(0 * us, 0 * V), (2 * us, 20 * V)])
    source = c.place(devices.voltage_source(ramp), name="V1")
    capacitor = c.place(devices.capacitor(1 * nF).rated("voltage", 16 * V), name="C1")
    c.connect(source.positive, capacitor.a, name="ramp")
    c.connect(source.negative, capacitor.b, ground)
    rail = c.place(devices.voltage_source(10 * V), name="V2")
    transistor = devices.bipolar_transistor().heated()
    switch = c.place(transistor.rated("temperature", celsius(150)), name="Q1")
    bias = c.place(devices.resistor(15 * kohm), name="RB")
    emitter = c.place(devices.resistor(10 * ohm), name="RE")
    mount = c.place(devices.thermal_resistance(240 * K / W), name="RTH")
    c.connect(rail.positive, switch.collector, bias.a, name="vcc")
    c.connect(bias.b, switch.base, name="base")
    c.connect(switch.emitter, emitter.a, name="emitter")
    c.connect(switch.thermal, mount.a, name="junction")
    c.connect(rail.negative, emitter.b, mount.b, ground)


def build() -> Component:
    """Returns the over-rating test circuit."""
    return supply()


def plan() -> Plan:
    """Returns the operating-point and transient ramp plan."""
    return Plan({"op": OperatingPoint(), "tran": Transient(stop=4 * us)})
