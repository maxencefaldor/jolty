# Copyright 2026 Maxence Faldor
"""Self-heating: devices at their thermal net's temperature, heated by their power."""

import math

import numpy as np
import pytest

from jolty import (
    Builder,
    Component,
    JoltyError,
    OperatingPoint,
    Role,
    Simulation,
    SimulationError,
    circuit,
    component,
    devices,
    parts,
    results,
)
from jolty.plan import Ac, Sensitivity, Transient
from jolty.simulate import Joined
from jolty.spice.export import netlist
from jolty.units import A, F, Hz, J, K, V, W, celsius, kohm, ohm, s, us
from jolty.waveforms import Pulse

RESISTANCE = 100.0
"""The heated resistor's resistance at the ambient, in ohms."""
COEFFICIENT = 4e-3
"""Its linear temperature coefficient, per kelvin."""
SINK = 50.0
"""The thermal resistance from it to the ambient, in kelvins per watt."""
STORE = 2e-3
"""The heat capacity beside it, in joules per kelvin."""


def _warmed(supply: devices.Device, coefficient: float = COEFFICIENT) -> Component:
    """`supply` across a heated resistor, held to the ambient by `SINK` and `STORE`."""

    @component
    def warmed(c: Builder, supply: devices.Device, coefficient: float) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(supply, id="V1")
        load = c.place(
            devices.resistor(
                RESISTANCE * ohm, temperature_coefficient=coefficient / K
            ).heated(),
            id="R1",
        )
        sink = c.place(devices.thermal_resistance(SINK * K / W), id="RTH")
        store = c.place(devices.thermal_capacitance(STORE * J / K), id="CTH")
        c.connect(source.positive, load.a, name="top")
        c.connect(load.thermal, sink.a, store.a, name="junction")
        c.connect(source.negative, load.b, sink.b, store.b, ground)

    return warmed(supply, coefficient)


def _rise(voltage: float, coefficient: float = COEFFICIENT) -> float:
    """The rise that heats `R(ΔT) = r · (1 + tc · ΔT)` by `v² / R(ΔT)` through `SINK`.

    `ΔT = SINK · v² / R(ΔT)` is `tc · ΔT² + ΔT - SINK · v² / r = 0`.
    """
    heat = SINK * voltage**2 / RESISTANCE
    if coefficient == 0:
        return heat
    return (math.sqrt(1 + 4 * coefficient * heat) - 1) / (2 * coefficient)


def test_a_resistor_heats_itself_to_where_its_power_balances_the_sink() -> None:
    simulation = Simulation(_warmed(devices.voltage_source(10 * V)))
    point = simulation.run(OperatingPoint(record=["**"]))
    rise = _rise(10.0)
    assert point.rise("junction") == pytest.approx(rise, rel=1e-9)
    hot = RESISTANCE * (1 + COEFFICIENT * rise)
    assert point.current("R1", "a") == pytest.approx(10.0 / hot, rel=1e-9)
    # The heat into its thermal terminal is the power it dissipates, out of it.
    assert point.current("R1", "thermal") == pytest.approx(-100.0 / hot, rel=1e-9)
    assert point.recording.thermal == {
        "R1.thermal",
        "RTH.a",
        "RTH.b",
        "CTH.a",
        "CTH.b",
    }


def test_a_heated_device_warms_through_its_thermal_time_constant() -> None:
    # The supply steps on at once from the operating point, where the resistor is at
    # the ambient.
    step = Pulse(0 * V, 10 * V, rise=1 * us, fall=1 * us, width=1 * s)
    simulation = Simulation(_warmed(devices.voltage_source(step), coefficient=0.0))
    tau = SINK * STORE
    waveforms = simulation.run(Transient(stop=5 * tau * s, max_step=tau / 200 * s))
    expected = _rise(10.0, 0.0) * (1 - np.exp(-waveforms.times / tau))
    assert waveforms.rise("junction") == pytest.approx(expected, abs=2e-3)


def test_heating_slows_a_resistors_small_signal_response_below_its_thermal_pole() -> (
    None
):
    bias = 0.1
    tau = SINK * STORE
    simulation = Simulation(_warmed(devices.current_source(bias * A)))
    frequencies = np.array([1e-3, 1e3]) / tau
    response = simulation.run(Ac("V1", [each * Hz for each in frequencies]))
    hot = RESISTANCE * (1 + COEFFICIENT * response.operating_point.rise("junction"))
    slope = RESISTANCE * COEFFICIENT
    # dV = R · dI + I · R' · dT, and dT = Z · (2 · I · R · dI + I² · R' · dT), with Z
    # the sink beside the store.
    mount = SINK / (1 + 2j * np.pi * frequencies * tau)
    warming = 2 * bias * hot * mount / (1 - bias**2 * slope * mount)
    expected = hot + bias * slope * warming
    # The source drives its current out of its negative terminal.
    assert -response.voltage("top") == pytest.approx(expected, rel=1e-9)
    slow, fast = np.abs(expected)
    assert slow > 1.4 * fast


def test_sensitivities_carry_through_the_heat() -> None:
    def rise(resistance: float) -> float:
        warmed = _warmed(devices.voltage_source(10 * V))
        return (
            Simulation(warmed, overrides={"R1": {"resistance": resistance}})
            .run(OperatingPoint())
            .rise("junction")
        )

    simulation = Simulation(_warmed(devices.voltage_source(10 * V)))
    found = simulation.run(Sensitivity(["junction"]))
    step = 1e-4
    expected = (rise(RESISTANCE + step) - rise(RESISTANCE - step)) / (2 * step)
    assert found.derivative("junction", "R1", "resistance") == pytest.approx(
        expected, rel=1e-6
    )


@component
def follower(c: Builder, *, heated: bool, sink: float, emitter: float) -> None:
    """A 2N3904 from 12 V into `emitter` ohms, joined by `sink` K/W to the ambient."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(12 * V), id="V1")
    base = c.place(devices.voltage_source(1.5 * V), id="V2")
    transistor = c.place(parts.q2n3904(heated=heated), id="Q1")
    load = c.place(devices.resistor(emitter * ohm), id="RE")
    c.connect(supply.positive, transistor.collector, name="vcc")
    c.connect(base.positive, transistor.base, name="b")
    c.connect(transistor.emitter, load.a, name="e")
    c.connect(supply.negative, base.negative, load.b, ground)
    if heated:
        mount = c.place(devices.thermal_resistance(sink * K / W), id="RTH")
        c.connect(transistor.thermal, mount.a, name="junction")
        c.connect(mount.b, ground)


def _power(point: results.OperatingPoint) -> float:
    """The power the follower's transistor dissipates at `point`."""
    return sum(
        point.voltage(net) * point.current("Q1", terminal)
        for net, terminal in (("vcc", "collector"), ("b", "base"), ("e", "emitter"))
    )


def test_a_heated_transistor_is_the_transistor_at_the_temperature_it_reaches() -> None:
    hot = Simulation(follower(heated=True, sink=200.0, emitter=20.0)).run(
        OperatingPoint()
    )
    rise = hot.rise("junction")
    # Within the tolerances Newton's iteration converges to.
    assert rise == pytest.approx(200.0 * _power(hot), rel=1e-6)
    plain = Simulation(
        follower(heated=False, sink=200.0, emitter=20.0),
        temperature=celsius(27) + rise * K,
    ).run(OperatingPoint())
    assert hot.voltage("e") == pytest.approx(plain.voltage("e"), rel=1e-6)
    assert hot.current("Q1", "collector") == pytest.approx(
        plain.current("Q1", "collector"), rel=1e-6
    )


def _diagnosed(simulation: Simulation, stress: str) -> list[str]:
    """The diagnostics of the follower's transistor's rating of `stress`."""
    point = simulation.run(OperatingPoint())
    return [
        each.message
        for each in simulation.check_results({"op": point})
        if f" {stress}" in each.message
    ]


def test_a_junction_temperature_is_held_to_its_rating() -> None:
    # About 0.66 W through 200 K/W lifts the junction past 150 °C; 0.48 W, past 80 %
    # of it, 120 °C, measured from 0 °C.
    (over,) = _diagnosed(
        Simulation(follower(heated=True, sink=200.0, emitter=15.0)), "temperature"
    )
    assert over.startswith("the temperature of `Q1` reaches 159.79")
    assert over.endswith(
        "°C at the operating point of `op`, beyond its rating of 150.0 °C"
    )
    (derated,) = _diagnosed(
        Simulation(follower(heated=True, sink=200.0, emitter=20.0)), "temperature"
    )
    assert derated.endswith("past 80% of its rating of 150.0 °C")
    # Not heated, it is at the simulation's temperature.
    plain = follower(heated=False, sink=0.0, emitter=20.0)
    assert _diagnosed(Simulation(plain), "temperature") == []
    (cold,) = _diagnosed(Simulation(plain, temperature=celsius(-60)), "temperature")
    assert cold == (
        "the temperature of `Q1` falls to -60.0 °C at the operating point of `op`, "
        "below its rating of -55.0 °C"
    )


def test_a_part_at_a_temperature_of_its_own_is_held_to_its_rating_there() -> None:
    @component
    def clamp(c: Builder, diode: devices.Device) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(5 * V), id="V1")
        feed = c.place(devices.resistor(1 * kohm), id="R1")
        placed = c.place(diode, id="D1")
        c.connect(source.positive, feed.a, name="in")
        c.connect(feed.b, placed.anode, name="out")
        c.connect(source.negative, placed.cathode, ground)
        if "thermal" in diode.terminals:
            mount = c.place(devices.thermal_resistance(300 * K / W), id="RTH")
            c.connect(placed.thermal, mount.a, name="junction")
            c.connect(mount.b, ground)

    def diagnosed(diode: devices.Device) -> list[str]:
        simulation = Simulation(clamp(diode))
        point = simulation.run(OperatingPoint())
        return [
            each.message
            for each in simulation.check_results({"op": point})
            if "temperature" in each.message
        ]

    (over,) = diagnosed(parts.d1n4148().held_at(celsius(180)))
    assert over.startswith("the temperature of `D1` reaches 180.0 °C")
    # An override of its own temperature, as a Monte Carlo sample's, holds too.
    simulation = Simulation(
        clamp(parts.d1n4148().held_at(celsius(180))),
        overrides={"D1": {"temperature": celsius(20).value}},
    )
    point = simulation.run(OperatingPoint())
    assert [
        each.message
        for each in simulation.check_results({"op": point})
        if "temperature" in each.message
    ] == []
    # Heated, it warms from its own temperature: about 2 mW through 300 K/W, 0.5 K.
    (warmed,) = diagnosed(parts.d1n4148(heated=True).held_at(celsius(174.9)))
    reached = float(warmed.split(" reaches ")[1].split(" °C")[0])
    assert 175.0 < reached < 175.9
    assert warmed.endswith("beyond its rating of 175.0 °C")


def test_a_heated_parts_power_is_what_its_electrical_terminals_take() -> None:
    simulation = Simulation(follower(heated=True, sink=50.0, emitter=15.0))
    power = _power(simulation.run(OperatingPoint()))
    # Rated 0.625 W: derated past 0.5 W, with nothing of the heat that flows.
    (derated,) = _diagnosed(simulation, "dissipates")
    stated = float(derated.split(" reaches ")[1].split(" W ")[0])
    assert stated == pytest.approx(power, rel=1e-12)
    assert derated.endswith("past 80% of its rating of 0.625 W")


def test_a_heated_power_mosfet_heats_its_channel_and_body_diode_together() -> None:
    @component
    def switch(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        supply = c.place(devices.voltage_source(5 * V), id="V1")
        gate = c.place(devices.voltage_source(10 * V), id="V2")
        transistor = c.place(parts.irf511(heated=True), id="X1")
        load = c.place(devices.resistor(1 * ohm), id="RL")
        mount = c.place(devices.thermal_resistance(2 * K / W), id="RTH")
        c.connect(supply.positive, load.a, name="vdd")
        c.connect(load.b, transistor.drain, name="d")
        c.connect(gate.positive, transistor.gate, name="g")
        c.connect(transistor.thermal, mount.a, name="junction")
        c.connect(supply.negative, gate.negative, transistor.source, mount.b, ground)

    simulation = Simulation(switch())
    assert simulation.joined("X1", "thermal") == Joined("junction", thermal=True)
    point = simulation.run(OperatingPoint())
    # The part is rated for a power, so the point records its ports by default.
    assert "X1.thermal" in point.recording.thermal
    power = point.voltage("d") * point.current("X1", "drain")
    assert point.rise("junction") == pytest.approx(2.0 * power, rel=1e-9)
    assert point.current("X1", "thermal") == pytest.approx(-power, rel=1e-9)


def test_a_port_on_the_ground_sums_its_electrical_terminals_only() -> None:
    @component
    def cell(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        top = c.port("top")
        load = c.place(devices.resistor(10 * ohm).heated(), id="R1")
        mount = c.place(devices.thermal_resistance(20 * K / W), id="RTH")
        c.connect(top, load.a)
        c.connect(load.thermal, mount.a, name="junction")
        c.connect(load.b, mount.b, ground)

    @component
    def board(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(1 * V), id="V1")
        inner = c.place(cell(), id="X1")
        c.connect(source.positive, inner.top, name="top")
        c.connect(source.negative, inner.ground, ground)

    point = Simulation(board()).run(OperatingPoint(record=["X1"]))
    # 0.1 A in at the top and out at the ground, whatever heat flows to the ambient.
    assert point.current("X1", "ground") == pytest.approx(-0.1, rel=1e-9)
    assert point.rise("X1/junction") == pytest.approx(2.0, rel=1e-9)


def test_only_devices_that_heat_take_a_thermal_terminal() -> None:
    assert devices.mosfet().heated().terminals[-1] == "thermal"
    assert parts.d1n4148(heated=True).parameters["thermal"] is True
    with pytest.raises(ValueError, match="a capacitor does not heat"):
        devices.capacitor(1 * F).heated()
    with pytest.raises(ValueError, match="a thermal_resistance does not heat"):
        devices.thermal_resistance(1 * K / W).heated()


def test_spice_has_no_heated_device() -> None:
    with pytest.raises(
        JoltyError,
        match="R1 is a heated resistor, and ngspice heats no device on a thermal",
    ):
        netlist(circuit(_warmed(devices.voltage_source(1 * V))))

    @component
    def cool(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(1 * V), id="V1")
        load = c.place(
            devices.device("resistor", {"resistance": 10.0, "thermal": False}),
            id="R1",
        )
        c.connect(source.positive, load.a, name="top")
        c.connect(source.negative, load.b, ground)

    assert netlist(circuit(cool()))[-1] == "RR1 top 0 10.0"


def test_a_temperature_rating_is_in_kelvins_above_zero() -> None:
    with pytest.raises(ValueError, match="a temperature rating's bounds are above 0 K"):
        devices.diode().rated("temperature", 400 * K, minimum=0 * K)
    with pytest.raises(Exception, match="temperature"):
        devices.diode().rated("temperature", 1 * V)


def test_a_device_heating_a_net_the_ambient_never_cools_has_no_operating_point() -> (
    None
):
    @component
    def insulated(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(1 * V), id="V1")
        load = c.place(devices.resistor(10 * ohm).heated(), id="R1")
        store = c.place(devices.thermal_capacitance(STORE * J / K), id="CTH")
        c.connect(source.positive, load.a, name="top")
        c.connect(load.thermal, store.a, name="junction")
        c.connect(source.negative, load.b, store.b, ground)

    simulation = Simulation(insulated())
    assert [each.rule for each in simulation.check()] == ["no-dc-path"]
    with pytest.raises(SimulationError, match="singular"):
        simulation.run(OperatingPoint())
