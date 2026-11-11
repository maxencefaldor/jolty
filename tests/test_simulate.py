# Copyright 2026 Maxence Faldor

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jolty import (
    Ac,
    Builder,
    DcSweep,
    DimensionError,
    ElaborationError,
    OperatingPoint,
    Options,
    Plan,
    Role,
    Simulation,
    SimulationError,
    Transient,
    component,
    devices,
)
from jolty.plan import decade, linear
from jolty.units import (
    A,
    Hz,
    K,
    Quantity,
    V,
    celsius,
    kHz,
    kohm,
    mA,
    nF,
    ohm,
    uA,
    us,
)
from jolty.waveforms import Piecewise


@component
def divider(c: Builder, level: Quantity, upper: Quantity, lower: Quantity) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(level), id="V1")
    top = c.place(devices.resistor(upper), id="R1")
    bottom = c.place(devices.resistor(lower), id="R2")
    c.connect(source.positive, top.a, name="in")
    c.connect(top.b, bottom.a, name="out")
    c.connect(bottom.b, source.negative, ground)


@component
def low_pass(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    step = Piecewise([(0 * us, 0 * V), (1e-3 * us, 1 * V)])
    source = c.place(devices.voltage_source(step), id="V1")
    resistor = c.place(devices.resistor(1 * kohm), id="R1")
    capacitor = c.place(devices.capacitor(1 * nF), id="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)


@component
def diode_load(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(0 * V), id="V1")
    resistor = c.place(devices.resistor(1 * kohm), id="R1")
    diode = c.place(devices.diode(), id="D1")
    c.connect(source.positive, resistor.a)
    c.connect(resistor.b, diode.anode, name="anode")
    c.connect(diode.cathode, source.negative, ground)


def test_an_operating_point_solves_kirchhoffs_laws() -> None:
    simulation = Simulation(divider(5 * V, 1 * kohm, 4 * kohm))
    assert simulation.devices == 3
    assert [unknown.name for unknown in simulation.unknowns] == [
        "in",
        "out",
        "V1:current",
    ]
    point = simulation.run(OperatingPoint())
    assert point.strategy == "direct"
    assert point.voltage("out") == pytest.approx(4.0)
    assert point.voltage("in", reference="out") == pytest.approx(1.0)
    assert point.current("V1") == pytest.approx(-1e-3)


@settings(max_examples=50, deadline=None)
@given(
    level=st.floats(min_value=-100, max_value=100),
    upper=st.floats(min_value=1e-3, max_value=1e9),
    lower=st.floats(min_value=1e-3, max_value=1e9),
)
def test_a_divider_divides_at_any_scale(
    level: float, upper: float, lower: float
) -> None:
    circuit = divider(level * V, upper * ohm, lower * ohm)
    point = Simulation(circuit).run(OperatingPoint())
    expected = level * lower / (upper + lower)
    assert point.voltage("out") == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_a_dc_sweep_solves_each_level() -> None:
    simulation = Simulation(divider(0 * V, 1 * kohm, 1 * kohm))
    sweep = simulation.run(DcSweep("V1", linear(-1 * V, 1 * V, 5)))
    np.testing.assert_allclose(sweep.levels, [-1, -0.5, 0, 0.5, 1])
    np.testing.assert_allclose(sweep.voltage("out"), sweep.levels / 2, atol=1e-12)


def test_a_diode_conducts_forward_and_blocks_in_reverse() -> None:
    sweep = Simulation(diode_load()).run(DcSweep("V1", linear(-5 * V, 5 * V, 101)))
    anode = sweep.voltage("anode")
    assert np.all(np.diff(anode) > 0)
    assert anode[0] == pytest.approx(-5.0, abs=1e-6)
    assert 0.6 < anode[-1] < 0.8
    current = -sweep.current("V1")
    assert current[-1] == pytest.approx((5 - anode[-1]) / 1e3)


def test_a_diode_s_forward_voltage_falls_as_it_warms() -> None:
    forward = DcSweep("V1", [5 * V])
    anodes = [
        Simulation(diode_load(), temperature).run(forward).voltage("anode")[0]
        for temperature in (celsius(0), celsius(100))
    ]
    slope = (anodes[1] - anodes[0]) / 100
    assert -2.5e-3 < slope < -1.5e-3


def test_a_resistance_drifts_with_its_temperature_coefficients() -> None:
    @component
    def drifting(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(1 * V), id="V1")
        load = c.place(
            devices.resistor(
                1 * kohm,
                temperature_coefficient=1e-3 / K,
                quadratic_temperature_coefficient=1e-6 / K**2,
                nominal_temperature=celsius(25),
            ),
            id="R1",
        )
        c.connect(source.positive, load.a)
        c.connect(source.negative, load.b, ground)

    point = Simulation(drifting(), celsius(75)).run(OperatingPoint())
    assert -point.current("V1") == pytest.approx(1e-3 / (1 + 50e-3 + 2.5e-3))


def test_the_ac_response_of_a_low_pass_filter() -> None:
    cutoff = 1 / (2 * math.pi * 1e3 * 1e-9)
    frequencies = [*decade(1 * kHz, 1e5 * kHz, 10), cutoff * kHz / 1e3]
    response = Simulation(low_pass()).run(Ac("V1", frequencies))
    expected = 1 / (1 + 1j * response.frequencies / cutoff)
    np.testing.assert_allclose(response.voltage("out"), expected, rtol=1e-9)
    assert abs(response.voltage("out")[-1]) == pytest.approx(1 / math.sqrt(2))
    assert response.operating_point.voltage("out") == pytest.approx(0.0)


def test_the_step_response_of_a_low_pass_filter() -> None:
    waveforms = Simulation(low_pass()).run(
        Transient(stop=5 * us), Options(relative_tolerance=1e-6, charge_tolerance=1e-18)
    )
    late = waveforms.times > 0.1e-6
    exact = 1 - np.exp(-(waveforms.times[late] - 0.5e-9) / 1e-6)
    np.testing.assert_allclose(waveforms.voltage("out")[late], exact, atol=2e-4)
    assert waveforms.times[-1] == pytest.approx(5e-6)
    assert waveforms.statistics.accepted == len(waveforms.times) - 1


@component
def common_emitter(c: Builder, transistor: devices.Device) -> None:
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(5 * V), id="V2")
    source = c.place(devices.voltage_source(0.75 * V), id="V1")
    base = c.place(devices.resistor(2 * kohm), id="RB")
    amplifier = c.place(transistor, id="Q1")
    load = c.place(devices.resistor(1 * kohm), id="RC")
    c.connect(source.positive, base.a, name="in")
    c.connect(base.b, amplifier.base, name="base")
    c.connect(supply.positive, load.a, name="supply")
    c.connect(load.b, amplifier.collector, name="out")
    c.connect(source.negative, supply.negative, amplifier.emitter, ground)


@pytest.mark.parametrize(
    ("minimum", "halfway"),
    [(None, None), (10 * ohm, None), (10 * ohm, 20 * uA)],
    ids=["fixed", "charge", "current"],
)
def test_a_small_signal_response_is_the_slope_of_the_dc_transfer(
    minimum: Quantity | None, halfway: Quantity | None
) -> None:
    # The base resistance, fixed or falling with the base charge or current, enters
    # the small-signal model with every derivative.
    transistor = devices.bipolar_transistor(
        saturation_current=1e-15 * A,
        forward_early_voltage=50 * V,
        forward_knee_current=10 * mA,
        base_resistance=300 * ohm,
        minimum_base_resistance=minimum,
        base_resistance_current=halfway,
    )
    simulation = Simulation(common_emitter(transistor))
    tight = Options(relative_tolerance=1e-11, voltage_tolerance=1e-15)
    # A central difference over 0.1 mV, small beside the thermal voltage, errs by about
    # 1e-5; leaving out how the base resistance falls would err by a percent.
    step = 1e-4
    levels = [(0.75 - step) * V, (0.75 + step) * V]
    sweep = simulation.run(DcSweep("V1", levels), tight)
    slopes = (sweep.values[1] - sweep.values[0]) / (2 * step)
    response = simulation.run(Ac("V1", [1 * Hz]), tight)
    np.testing.assert_allclose(response.values[0].real, slopes, rtol=1e-4, atol=1e-12)
    assert abs(response.voltage("out")[0]) > 10  # The stage amplifies.


def test_a_plan_runs_its_analyses_by_name() -> None:
    simulation = Simulation(divider(5 * V, 1 * kohm, 1 * kohm))
    plan = Plan({"op": OperatingPoint(), "sweep": DcSweep("V1", [1 * V, 2 * V])})
    results = simulation.run_plan(plan)
    assert list(results) == ["op", "sweep"]


def test_a_plan_runs_at_the_simulation_s_temperature() -> None:
    plan = Plan({"op": OperatingPoint()}, temperature=celsius(85))
    simulation = Simulation(divider(5 * V, 1 * kohm, 1 * kohm), celsius(85))
    assert list(simulation.run_plan(plan)) == ["op"]
    with pytest.raises(ValueError, match=r"is not the simulation's, 300\.1 K"):
        Simulation(divider(5 * V, 1 * kohm, 1 * kohm)).run_plan(plan)
    with pytest.raises(DimensionError, match="a plan's temperature"):
        Plan({"op": OperatingPoint()}, temperature=85 * V)


def test_plans_name_their_analyses_by_identifiers() -> None:
    with pytest.raises(ValueError, match="at least one analysis"):
        Plan({})
    with pytest.raises(ValueError, match="identifier, not 'step 1'"):
        Plan({"step 1": OperatingPoint()})


def test_problems_name_the_instance_and_the_script_line() -> None:
    @component
    def floating(c: Builder) -> None:
        c.port("ground", Role.GROUND)
        resistor = c.place(devices.resistor(1 * kohm), id="R1")
        c.connect(resistor.a, resistor.b)

    @component
    def ungrounded(c: Builder) -> None:
        c.place(devices.resistor(-1 * kohm), id="R1")

    Simulation(floating())
    with pytest.raises(ElaborationError) as raised:
        Simulation(ungrounded())
    problems = raised.value.problems
    assert any("ground" in problem for problem in problems)
    assert any(
        "`R1` (tests/test_simulate.py:" in problem and "positive" in problem
        for problem in problems
    )


def test_analyses_check_their_sources() -> None:
    simulation = Simulation(divider(5 * V, 1 * kohm, 1 * kohm))
    with pytest.raises(SimulationError, match="there is no instance at `V9`"):
        simulation.run(DcSweep("V9", [1 * V]))
    with pytest.raises(SimulationError, match="`R1` is not an independent source"):
        simulation.run(DcSweep("R1", [1 * V]))
    with pytest.raises(DimensionError, match="a sweep level takes a quantity in V"):
        simulation.run(DcSweep("V1", [1 * mA]))


def test_a_current_source_sweeps_in_amperes() -> None:
    @component
    def biased(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.current_source(1 * mA), id="I1")
        resistor = c.place(devices.resistor(1 * kohm), id="R1")
        c.connect(source.negative, resistor.a, name="top")
        c.connect(source.positive, resistor.b, ground)

    simulation = Simulation(biased())
    assert simulation.run(OperatingPoint()).voltage("top") == pytest.approx(1.0)
    sweep = simulation.run(DcSweep("I1", [2 * mA, 1 * A]))
    np.testing.assert_allclose(sweep.voltage("top"), [2.0, 1000.0])
