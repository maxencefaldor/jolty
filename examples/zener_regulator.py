# Copyright 2026 Maxence Faldor
"""A zener shunt regulator: a 5.1 V zener holding a load's supply against its input.

A 220 Ω resistor feeds the zener and a 1 kΩ load from a 12 V input. Above the zener's
knee, the input's changes fall across the resistor, and the output moves by the
zener's small resistance over the resistor's: a line regulation of some 1 %. Simulate
it with ``jolty simulate examples/zener_regulator.py``.
"""

from common import zener

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.plan import DcSweep, OperatingPoint, Transient, linear
from jolty.specs import value
from jolty.units import Hz, V, kohm, ms, mV, ohm, us
from jolty.waveforms import Sine


@component
def regulator(c: Builder) -> None:
    """Builds the series dropping resistor and shunt 5.1 V zener diode."""
    source, output = c.port("input"), c.port("output")
    ground = c.port("ground", Role.GROUND)
    series = c.place(devices.resistor(220 * ohm), name="R1")
    diode = c.place(zener(5.1), name="D1")
    c.connect(source, series.a)
    c.connect(series.b, diode.cathode, output)
    c.connect(diode.anode, ground)


@component
def bench(c: Builder) -> None:
    """Builds the zener regulator loaded by 1 kΩ on a 12 V supply with 1 V ripple."""
    ground = c.port("ground", Role.GROUND)
    rippled = Sine(offset=12 * V, amplitude=1 * V, frequency=100 * Hz)
    source = c.place(devices.voltage_source(rippled), name="V1")
    stage = c.place(regulator(), name="X1")
    load = c.place(devices.resistor(1 * kohm), name="RL")
    c.connect(source.positive, stage.input, name="in")
    c.connect(stage.output, load.a, name="out")
    c.connect(source.negative, stage.ground, load.b, ground)


def build() -> Component:
    """Returns the zener shunt regulator test bench component."""
    return bench()


def plan() -> Plan:
    """Returns operating-point, 0 V to 20 V line-sweep, and ripple analyses."""
    return Plan(
        {
            "op": OperatingPoint(),
            "line": DcSweep("V1", linear(0 * V, 20 * V, 81)),
            "ripple": Transient(stop=30 * ms, max_step=20 * us),
        }
    )


def specs() -> Specs:
    """Returns output specs within 0.2 V of 5.1 V from a 9 V to 20 V input."""
    return Specs(
        [
            value("line", "out", at=level * V).within(5.1 * V, 200 * mV)
            for level in (9, 12, 15, 20)
        ]
    )
