# Copyright 2026 Maxence Faldor
"""An inverter characterized as a logic cell driving a load beyond its domain.

The cell is characterized up to four standard fanouts (~20 fF) and drives
470 fF here, triggering `outside-validity` table extrapolation. The falling
delay through half-supply then diverges between the characterized cell and
transistor-level circuit (`fidelity-dependence`).
"""

from common import SUPPLY, inverter

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.plan import Characterized, Transient
from jolty.specs import crossing
from jolty.units import V, fF, ns, ps
from jolty.waveforms import Piecewise


@component
def bench(c: Builder) -> None:
    """Builds the inverter driven by a 1 ns step into a 470 fF load."""
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
    """Returns the overloaded-inverter bench component."""
    return bench()


def plan() -> Plan:
    """Returns the characterized transient plan with one circuit spot check."""
    return Plan(
        {"step": Transient(stop=8 * ns)},
        fidelity={"X1": Characterized(SUPPLY * V, spot_checks=1)},
    )


def specs() -> Specs:
    """Returns the 2.37 ns falling half-supply crossing specification."""
    return Specs([crossing("step", "mid", SUPPLY / 2 * V, edge="falling") <= 2.37 * ns])
