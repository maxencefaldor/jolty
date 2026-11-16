# Copyright 2026 Maxence Faldor
"""A three-bit flash ADC: seven comparators on a resistor ladder, and an encoder.

Eight equal resistors divide the 3.3 V reference into seven taps, an eighth of it
apart. Each comparator compares the input to one tap; together they read the input as
a thermometer code, as many ones as taps below it, and a logic encoder counts the ones
into three bits. A ramp from 0 to 3.3 V walks the code from 0 to 7. Simulate it with
``jolty simulate examples/flash_adc.py``.
"""

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.plan import Transient
from jolty.specs import value
from jolty.units import V, fF, kohm, mV, ns, ps, us
from jolty.waveforms import Piecewise

SUPPLY = 3.3
BITS = 3
TAPS = 2**BITS - 1
RAMP = 8e-6
"""How long the input takes to rise from 0 V to the reference, in seconds."""


def comparator() -> devices.Device:
    """A comparator, its output the supply or ground as `in` is above `ref` or not."""
    half = SUPPLY / 2
    return devices.behavioral_voltage_source(
        f"{half} + {half} * tanh(v(in, ref) / 1e-3)"
    )


@component
def flash(c: Builder) -> None:
    """The ladder, the comparators, and the encoder of the thermometer code."""
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
    """The converter, its reference 3.3 V, its input a ramp, its outputs loaded."""
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
    """The converter on its bench."""
    return bench()


def plan() -> Plan:
    """The ramp, from 0 V to the reference."""
    return Plan({"ramp": Transient(stop=RAMP * 1e6 * us, max_step=10 * ns)})


def specs() -> Specs:
    """Each code's bits, read where the ramp is mid-way through its step."""
    requirements = []
    for code in range(2**BITS):
        at = (code + 0.5) / 2**BITS * RAMP * 1e6 * us
        for bit in range(BITS):
            level = SUPPLY if (code >> bit) & 1 else 0.0
            requirements.append(
                value("ramp", f"b{bit}", at=at).within(level * V, 10 * mV)
            )
    return Specs(requirements)
