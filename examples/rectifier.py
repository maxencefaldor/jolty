# Copyright 2026 Maxence Faldor
"""A half-wave rectifier: a diode charging a smoothing capacitor from a 50 Hz sine.

The diode's parameters are those of a small-signal silicon diode such as the 1N4148.
Simulate it with ``jolty simulate examples/rectifier.py``.
"""

from jolty import Builder, Component, Plan, Role, component, devices
from jolty.plan import DcSweep, Transient, linear
from jolty.units import Hz, Quantity, V, kohm, ms, nA, ns, ohm, pF, uF, us
from jolty.waveforms import Sine


@component
def rectifier(c: Builder, capacitance: Quantity, load: Quantity) -> None:
    """Builds a series diode feeding a parallel smoothing capacitor and load."""
    source = c.port("input")
    output = c.port("output")
    ground = c.port("ground", Role.GROUND)
    diode = c.place(
        devices.diode(
            saturation_current=2.52 * nA,
            emission_coefficient=1.752,
            series_resistance=0.568 * ohm,
            junction_capacitance=4 * pF,
            grading_coefficient=0.4,
            transit_time=20 * ns,
        ),
        name="D",
    )
    smoothing = c.place(devices.capacitor(capacitance), name="C")
    resistor = c.place(devices.resistor(load), name="R")
    c.connect(source, diode.anode)
    c.connect(diode.cathode, smoothing.a, resistor.a, output)
    c.connect(smoothing.b, resistor.b, ground)


@component
def bench(c: Builder) -> None:
    """Builds the rectifier driven by a 10 V, 50 Hz sine wave."""
    ground = c.port("ground", Role.GROUND)
    mains = Sine(offset=0 * V, amplitude=10 * V, frequency=50 * Hz)
    source = c.place(devices.voltage_source(mains), name="V1")
    stage = c.place(rectifier(100 * uF, 1 * kohm), name="X1")
    c.connect(source.positive, stage.input, name="in")
    c.connect(stage.output, name="out")
    c.connect(source.negative, stage.ground, ground)


def build() -> Component:
    """Returns the half-wave rectifier test bench component."""
    return bench()


def plan() -> Plan:
    """Returns a three-cycle transient ripple plan and a DC diode transfer sweep."""
    return Plan(
        {
            "ripple": Transient(stop=60 * ms, max_step=50 * us),
            "transfer": DcSweep("V1", linear(-2 * V, 2 * V, 41)),
        }
    )
