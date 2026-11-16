# Copyright 2026 Maxence Faldor
"""A first-order RC low-pass filter: its step response and its frequency response.

The filter's time constant is 1 µs and its cutoff frequency 159 kHz. Simulate it with
``jolty simulate examples/rc_filter.py``, which checks its specs too.
"""

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.plan import Ac, OperatingPoint, Transient, decade
from jolty.specs import bandwidth, gain, overshoot, rise_time
from jolty.units import MHz, Quantity, V, kHz, kohm, nF, ns, us
from jolty.waveforms import Piecewise


@component
def low_pass(c: Builder, resistance: Quantity, capacitance: Quantity) -> None:
    """A series resistor, then a capacitor to ground."""
    source = c.port("input")
    output = c.port("output")
    ground = c.port("ground", Role.GROUND)
    resistor = c.place(devices.resistor(resistance), name="R")
    capacitor = c.place(devices.capacitor(capacitance), name="C")
    c.connect(source, resistor.a)
    c.connect(resistor.b, capacitor.a, output)
    c.connect(capacitor.b, ground)


@component
def bench(c: Builder) -> None:
    """The filter, driven by a 1 V step at 1 µs."""
    ground = c.port("ground", Role.GROUND)
    step = Piecewise([(1 * us, 0 * V), (1 * us + 1 * ns, 1 * V)])
    source = c.place(devices.voltage_source(step), name="V1")
    stage = c.place(low_pass(1 * kohm, 1 * nF), name="X1")
    c.connect(source.positive, stage.input, name="in")
    c.connect(stage.output, name="out")
    c.connect(source.negative, stage.ground, ground)


def build() -> Component:
    """The filter on its bench."""
    return bench()


def plan() -> Plan:
    """The operating point, the step response and the frequency response."""
    return Plan(
        {
            "op": OperatingPoint(),
            "step": Transient(stop=10 * us),
            "bode": Ac("V1", decade(1 * kHz, 100 * MHz, 10)),
        }
    )


def specs() -> Specs:
    """A flat passband to 150 kHz, and a step that rises within 2.5 µs, cleanly."""
    return Specs(
        [
            gain("bode", "out", at=1 * kHz).within(0, 0.1),
            bandwidth("bode", "out").between(150 * kHz, 170 * kHz),
            (rise_time("step", "out") <= 2.5 * us).with_margin(100 * ns),
            overshoot("step", "out") <= 1,
        ]
    )
