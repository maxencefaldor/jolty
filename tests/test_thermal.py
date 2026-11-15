# Copyright 2026 Maxence Faldor
"""Thermal networks: temperatures above the ambient, and the heat that flows."""

import pytest

from jolty import (
    Builder,
    ElaborationError,
    OperatingPoint,
    Role,
    Simulation,
    component,
    devices,
)
from jolty.plan import Transient
from jolty.plots import default_probes, plot
from jolty.units import J, K, V, W, ms, ohm


@component
def heated(c: Builder) -> None:
    """1 W into 50 K/W to the ambient, beside 2 mJ/K."""
    ground = c.port("ground", Role.GROUND)
    heat = c.place(devices.heat_source(1 * W), id="P1")
    sink = c.place(devices.thermal_resistance(50 * K / W), id="RTH")
    store = c.place(devices.thermal_capacitance(2e-3 * J / K), id="CTH")
    c.connect(heat.b, sink.a, store.a, name="hot")
    c.connect(heat.a, sink.b, store.b, ground)


def test_heat_raises_a_net_by_its_flow_through_the_resistance_to_the_ambient() -> None:
    simulation = Simulation(heated())
    assert [(u.name, u.nature) for u in simulation.unknowns] == [("hot", "temperature")]
    point = simulation.run(OperatingPoint())
    assert point.rise("hot") == pytest.approx(50.0, rel=1e-12)
    with pytest.raises(KeyError, match="'hot' holds a temperature, not a voltage"):
        point.voltage("hot")
    waveforms = simulation.run(Transient(stop=100 * ms))
    assert waveforms.rise("hot")[-1] == pytest.approx(50.0, rel=1e-9)
    assert default_probes(simulation.unknowns) == ("hot",)
    figure = plot("warm", waveforms, ["hot"])
    assert figure.axes[0].get_ylabel() == "temperature rise"


def test_a_thermal_terminal_joins_no_electrical_net() -> None:
    @component
    def mixed(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(1 * V), id="V1")
        load = c.place(devices.resistor(1 * ohm), id="R1")
        sink = c.place(devices.thermal_resistance(50 * K / W), id="RTH")
        c.connect(source.positive, load.a, sink.a, name="out")
        c.connect(source.negative, load.b, sink.b, ground)

    with pytest.raises(ElaborationError) as raised:
        Simulation(mixed())
    (problem,) = raised.value.problems
    assert problem.startswith(
        "error: joins its thermal terminal `a` to the net `out`, which electrical "
        "terminals or logic pins join too, and a temperature is no voltage\n  = RTH, "
    )


def test_thermal_values_take_their_units() -> None:
    assert devices.thermal_resistance(50 * K / W).parameters == {"resistance": 50.0}
    assert devices.thermal_capacitance(2 * J / K).parameters == {"capacitance": 2.0}
    assert devices.heat_source(0.5 * W).parameters == {"power": 0.5}
    with pytest.raises(Exception, match="K/W"):
        devices.thermal_resistance(50 * ohm)
