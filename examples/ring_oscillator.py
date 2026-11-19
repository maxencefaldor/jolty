# Copyright 2026 Maxence Faldor
"""A ring oscillator of five CMOS inverters, transistor by transistor.

An odd ring of inverters has no stable state: each edge runs round the ring and comes
back inverted, so the ring oscillates with a period of twice the five stages' delays.
Each inverter stage drives the next stage's gate plus 10 fF, oscillating near 930 MHz.
A pulse on the first stage's input sets oscillation going. Simulate with
``jolty simulate examples/ring_oscillator.py``.
"""

from common import SUPPLY, inverter

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.plan import Transient
from jolty.specs import frequency, peak_to_peak
from jolty.units import GHz, MHz, V, fF, ns, ps, uA
from jolty.waveforms import Pulse

STAGES = 5


@component
def ring(c: Builder) -> None:
    """Builds five CMOS inverters chained in a closed ring with 10 fF stage loads."""
    output = c.port("output", Role.OUTPUT)
    vdd, gnd = c.port("vdd", Role.SUPPLY), c.port("gnd", Role.GROUND)
    gates = [c.place(inverter(), name=f"X{index}") for index in range(1, STAGES + 1)]
    nets = [output, *(c.net(f"n{index}") for index in range(1, STAGES))]
    for index, gate in enumerate(gates):
        load = c.place(devices.capacitor(10 * fF), name=f"C{index + 1}")
        c.connect(gate.vdd, vdd)
        c.connect(gate.gnd, load.b, gnd)
        c.connect(gate.a, nets[index - 1])
        c.connect(gate.y, load.a, nets[index])


@component
def bench(c: Builder) -> None:
    """Builds the ring oscillator on 3.3 V kicked by a short current pulse."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(SUPPLY * V), name="VDD")
    kick = Pulse(
        initial=0 * uA,
        pulsed=200 * uA,
        delay=0.1 * ns,
        rise=10 * ps,
        fall=10 * ps,
        width=50 * ps,
    )
    start = c.place(devices.current_source(kick), name="I1")
    oscillator = c.place(ring(), name="X1")
    c.connect(supply.positive, oscillator.vdd, name="vdd")
    c.connect(oscillator.output, start.negative, name="out")
    c.connect(supply.negative, start.positive, oscillator.gnd, ground)


def build() -> Component:
    """Returns the ring oscillator test bench component."""
    return bench()


def plan() -> Plan:
    """Returns a 20 ns transient plan covering roughly a dozen oscillation periods."""
    return Plan({"run": Transient(stop=20 * ns, max_step=5 * ps)})


def specs() -> Specs:
    """Returns specifications for rail-to-rail oscillation between 400 MHz and 1 GHz."""
    return Specs(
        [
            frequency("run", "out", start=5 * ns).between(400 * MHz, 1 * GHz),
            peak_to_peak("run", "out", start=5 * ns) >= 3 * V,
        ]
    )
