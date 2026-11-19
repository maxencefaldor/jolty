# Copyright 2026 Maxence Faldor
"""A three-bit flash ADC: seven comparators on a resistor ladder, and an encoder.

Eight equal resistors divide the 3.3 V reference into seven taps, an eighth of it
apart. Each comparator compares the input to one tap; together they read the input as
a thermometer code, as many ones as taps below it, and a logic encoder counts the ones
into three bits. A ramp from 0 to 3.3 V walks the code from 0 to 7. Simulate it with
``jolty simulate examples/flash_adc.py``.
"""

from common import SUPPLY, comparator

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.plan import Transient
from jolty.specs import value
from jolty.units import V, fF, kohm, mV, ns, ps, us
from jolty.waveforms import Piecewise

BITS = 3
TAPS = 2**BITS - 1
RAMP = 8e-6
"""Duration for the input ramp to rise from 0 V to the reference, in seconds."""


@component
def flash(c: Builder) -> None:
    """Builds the resistor ladder, comparators, and thermometer-code encoder."""
    source = c.port("input", Role.INPUT)
    outputs = [c.port(f"b{bit}", Role.OUTPUT) for bit in range(BITS)]
    reference, ground = c.port("vref", Role.SUPPLY), c.port("ground", Role.GROUND)
    taps = [c.net(f"tap{index}") for index in range(1, TAPS + 1)]
    ends = [ground, *taps, reference]
    for index in range(TAPS + 1):
        rung = c.place(devices.resistor(1 * kohm), name=f"R{index + 1}")
        c.connect(rung.a, ends[index])
        c.connect(rung.b, ends[index + 1])
    names = [f"t{index}" for index in range(1, TAPS + 1)]
    encoder = c.place(
        devices.logic_cell(
            names,
            {
                f"b{bit}": (lambda *levels, bit=bit: bool((sum(levels) >> bit) & 1))
                for bit in range(BITS)
            },
            rise=200 * ps,
            transition=50 * ps,
            electrical=True,
        ),
        name="U1",
    )
    for index, tap in enumerate(taps, start=1):
        compare = c.place(comparator(), name=f"B{index}")
        c.connect(compare["in"], source)
        c.connect(compare.ref, tap)
        c.connect(compare.positive, encoder[f"t{index}"], name=f"thermometer{index}")
        c.connect(compare.negative, ground)
    for bit in range(BITS):
        c.connect(encoder[f"b{bit}"], outputs[bit])
    c.connect(encoder.supply, reference)
    c.connect(encoder.ground, ground)


@component
def bench(c: Builder) -> None:
    """Builds the converter with a 3.3 V reference, input ramp, and capacitive loads."""
    ground = c.port("ground", Role.GROUND)
    reference = c.place(devices.voltage_source(SUPPLY * V), name="VREF")
    ramp = Piecewise([(0 * us, 0 * V), (RAMP * 1e6 * us, SUPPLY * V)])
    source = c.place(devices.voltage_source(ramp), name="VIN")
    converter = c.place(flash(), name="X1")
    c.connect(reference.positive, converter.vref, name="vref")
    c.connect(source.positive, converter.input, name="in")
    for bit in range(BITS):
        load = c.place(devices.capacitor(10 * fF), name=f"C{bit}")
        c.connect(converter[f"b{bit}"], load.a, name=f"b{bit}")
        c.connect(load.b, ground)
    c.connect(reference.negative, source.negative, converter.ground, ground)


def build() -> Component:
    """Returns the flash ADC test bench component."""
    return bench()


def plan() -> Plan:
    """Returns a transient plan sweeping the input ramp from 0 V to the reference."""
    return Plan({"ramp": Transient(stop=RAMP * 1e6 * us, max_step=10 * ns)})


def specs() -> Specs:
    """Returns bit-level specifications sampled at the midpoint of each code step."""
    requirements = []
    for code in range(2**BITS):
        at = (code + 0.5) / 2**BITS * RAMP * 1e6 * us
        for bit in range(BITS):
            level = SUPPLY if (code >> bit) & 1 else 0.0
            requirements.append(
                value("ramp", f"b{bit}", at=at).within(level * V, 10 * mV)
            )
    return Specs(requirements)
