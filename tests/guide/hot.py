# Copyright 2026 Maxence Faldor
"""An RC low-pass whose resistor drifts 1000 ppm/K, so it rises slowest when hot."""

from jolty import Builder, Component, Corner, Plan, Role, Specs, component, devices
from jolty.plan import Transient
from jolty.specs import rise_time
from jolty.units import K, V, celsius, kohm, nF, ns, us
from jolty.waveforms import Piecewise


@component
def bench(c: Builder) -> None:
    """1 kΩ and 1 nF, driven by a 1 V step at 1 µs."""
    ground = c.port("ground", Role.GROUND)
    step = Piecewise([(1 * us, 0 * V), (1 * us + 1 * ns, 1 * V)])
    source = c.place(devices.voltage_source(step), name="V1")
    resistor = c.place(
        devices.resistor(1 * kohm, temperature_coefficient=1e-3 / K), name="R1"
    )
    capacitor = c.place(devices.capacitor(1 * nF), name="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, capacitor.a, name="out")
    c.connect(source.negative, capacitor.b, ground)


def build() -> Component:
    """The filter on its bench."""
    return bench()


def plan() -> Plan:
    """The step response at 27 °C, -40 °C and 125 °C."""
    return Plan(
        {"step": Transient(stop=10 * us)},
        corners={
            "cold": Corner(temperature=celsius(-40)),
            "hot": Corner(temperature=celsius(125)),
        },
    )


def specs() -> Specs:
    """A rise within 2.3 µs."""
    return Specs([rise_time("step", "out") <= 2.3 * us])
