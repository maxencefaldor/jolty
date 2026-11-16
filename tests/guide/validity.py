# Copyright 2026 Maxence Faldor
"""An inverter characterized as a logic cell, driving a load far past its domain.

The cell is characterized over loads up to four of its own fanouts, some 20 fF, and
drives 470 fF here, so it reads its tables outside their domain, `outside-validity`,
and the time its output falls through half the supply depends on whether the inverter
is its cell or its circuit, `fidelity-dependence`.
"""

from common import SUPPLY, inverter

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.plan import Characterized, Transient
from jolty.specs import crossing
from jolty.units import V, fF, ns, ps
from jolty.waveforms import Piecewise


@component
def bench(c: Builder) -> None:
    """The inverter, its input stepping up at 1 ns, its output on 470 fF."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(SUPPLY * V), name="VDD")
    step = Piecewise([(1 * ns, 0 * V), (1 * ns + 100 * ps, SUPPLY * V)])
    source = c.place(devices.voltage_source(step), name="VIN")
    gate = c.place(inverter(), name="X1")
    load = c.place(devices.capacitor(470 * fF), name="CL")
    c.connect(supply.positive, gate.vdd, name="vdd")
    c.connect(source.positive, gate.a, name="in")
    c.connect(gate.y, load.a, name="mid")
    c.connect(supply.negative, source.negative, gate.gnd, load.b, ground)


def build() -> Component:
    """The inverter on its bench."""
    return bench()


def plan() -> Plan:
    """The step, the inverter a characterized cell, and its circuit in a spot check."""
    return Plan(
        {"step": Transient(stop=8 * ns)},
        fidelity={"X1": Characterized(SUPPLY * V, spot_checks=1)},
    )


def specs() -> Specs:
    """The output's fall through half the supply, by 2.37 ns."""
    return Specs([crossing("step", "mid", SUPPLY / 2 * V, edge="falling") <= 2.37 * ns])
