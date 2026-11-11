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
    """A diode into a smoothing capacitor and a load in parallel."""
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
        id="D",
    )
    smoothing = c.place(devices.capacitor(capacitance), id="C")
    resistor = c.place(devices.resistor(load), id="R")
    c.connect(source, diode.anode)
    c.connect(diode.cathode, smoothing.a, resistor.a, output)
    c.connect(smoothing.b, resistor.b, ground)


@component
def bench(c: Builder) -> None:
    """The rectifier, fed a 10 V sine at 50 Hz."""
    ground = c.port("ground", Role.GROUND)
    mains = Sine(offset=0 * V, amplitude=10 * V, frequency=50 * Hz)
    source = c.place(devices.voltage_source(mains), id="V1")
    stage = c.place(rectifier(100 * uF, 1 * kohm), id="X1")
    c.connect(source.positive, stage.input, name="in")
    c.connect(stage.output, name="out")
    c.connect(source.negative, stage.ground, ground)


def build() -> Component:
    """The rectifier on its bench."""
    return bench()


def plan() -> Plan:
    """Three periods of the mains, and the DC transfer curve through the diode."""
    return Plan(
        {
            "ripple": Transient(stop=60 * ms, max_step=50 * us),
            "transfer": DcSweep("V1", linear(-2 * V, 2 * V, 41)),
        }
    )
