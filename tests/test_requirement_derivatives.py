# Copyright 2026 Maxence Faldor
"""End-to-end verification of measure cotangents, slack derivatives, and worst-case.

Checks every differentiable measure family (OP, DC sweep, AC, transient, and periodic
steady state) against central finite differences, verifies signed requirement slack
derivatives for lower, upper, two-sided, and multi-corner specifications, verifies
linear worst-case slack estimates with shared lots against explicit corner evaluation,
and checks optional slack-derivative reporting on failing requirements.
"""

import itertools
import math
import sys
from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from jolty import (
    Ac,
    Builder,
    Corner,
    DcSweep,
    Noise,
    OperatingPoint,
    OperatingPoints,
    Options,
    PeriodicSteadyState,
    Plan,
    Role,
    Simulation,
    Specs,
    Stability,
    Transient,
    component,
    devices,
    function,
    ir,
    report,
    results,
    specs,
    variation,
    vjp,
    waveforms,
)
from jolty.plan import decade
from jolty.simulate import (
    Function,
    Parameters,
    estimate_worst_case,
    linear_worst_case,
    measure_derivatives,
    requirement_derivatives,
    slack_derivatives,
)
from jolty.spice import Library
from jolty.units import (
    VOLTAGE,
    A,
    F,
    H,
    Hz,
    K,
    MHz,
    V,
    celsius,
    kHz,
    kohm,
    mA,
    mH,
    nF,
    ns,
    ohm,
    pF,
    uA,
    uF,
    um,
    us,
)

RELATIVE_STEP = 1e-4
"""Central-difference relative parameter step balancing truncation and rounding."""

SMOOTH_FINITE_DIFFERENCE = 1e-5
"""Relative tolerance for central differences on smooth OP, DC sweep, and AC measures.

Central differences at relative step 1e-4 carry O(step^2) = 1e-8 truncation error times
local curvature plus O(eps / step) ~ 1e-12 rounding; 1e-5 catches any missing term or
wrong sign while accommodating nonlinear diode and transfer curvature.
"""

TIME_DOMAIN_FINITE_DIFFERENCE = 1e-4
"""Relative tolerance for central differences on transient and PSS measures.

Fixed-step trapezoidal integration and autonomous shooting at reltol=1e-10 combined
with central differences at relative step 1e-4 agree within 1e-4 across crossings,
settling bands, harmonic distortion, and oscillator periods.
"""

LINEAR_CORNER_ROUNDING = 1e-11
"""Relative tolerance comparing linear worst-case slack to explicit linear corners.

On linear resistor networks, first-order Taylor expansion over tolerance corners is
exact up to sparse LU solve rounding (~1e-15) and subtraction cancellation across
multi-term sums.
"""


def _tight_options(*, fixed_steps: bool = False) -> Options:
    """Returns tight solver options, optionally holding time steps at `max_step`."""
    return Options(
        relative_tolerance=1e-10,
        voltage_tolerance=1e-13 * V,
        current_tolerance=1e-16 * A,
        truncation_tolerance=1e12 if fixed_steps else None,
    )


def _perturbed(
    base: Parameters, instance: str, parameter: str, value: float
) -> dict[str, dict[str, float]]:
    """Returns a copy of `base` with `instance.parameter` replaced by `value`."""
    copied = {path: dict(named) for path, named in base.items()}
    copied[instance][parameter] = value
    return copied


def _assert_measure_derivatives_match_central_differences(
    evaluated: Function,
    parameters: Parameters,
    measures: Sequence[specs.Measure],
    tolerance: float,
) -> None:
    """Checks `measure_derivatives` against central differences within `tolerance`.

    Each central difference uses `RELATIVE_STEP` and compares against the exact reverse
    derivative within `tolerance` covering O(step^2) truncation and solver rounding.
    """
    derived = measure_derivatives(evaluated, parameters, measures)
    assert len(derived) == len(measures)

    def read_all(point: Parameters) -> list[float]:
        wrapped = devices.wrap_overrides(evaluated.elaborated.circuit, point)
        simulation = Simulation.from_elaborated(evaluated.elaborated, wrapped)
        solved = simulation.run(evaluated.analysis, evaluated.options)
        return [measure.read(solved).value for measure in measures]

    nominal = read_all(parameters)
    for item, primal in zip(derived, nominal, strict=True):
        # Primal values come from the identical forward solve; float rounding only.
        assert item.reading.value == pytest.approx(primal, rel=1e-12, abs=1e-15)

    for instance, named in parameters.items():
        for parameter, value in named.items():
            step = RELATIVE_STEP * abs(value)
            above = read_all(_perturbed(parameters, instance, parameter, value + step))
            below = read_all(_perturbed(parameters, instance, parameter, value - step))
            for item, high, low in zip(derived, above, below, strict=True):
                exact = item.derivatives[instance][parameter]
                finite = (high - low) / (2.0 * step)
                peak_value = max(abs(high), abs(low), 1.0)
                # Central-difference cancellation floor when an exact derivative is
                # identically zero (e.g. AC 3 dB bandwidth wrt real scalar VCVS gain):
                # solver/subtraction rounding scales as ~1e3 * eps * peak / (2 * step).
                roundoff_floor = (
                    1e3 * sys.float_info.epsilon * peak_value / (2.0 * step)
                )
                bound = max(
                    tolerance * max(abs(exact), abs(finite)),
                    roundoff_floor,
                )
                assert abs(exact - finite) <= bound, (
                    f"{item.measure} d/d({instance}.{parameter}): "
                    f"exact {exact:.9e} vs finite {finite:.9e}"
                )


@component
def _diode_network(c: Builder) -> None:
    """Nonlinear diode-resistor network for OP and DC sweep measure differentiation."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(3 * V), name="V1")
    series = c.place(devices.resistor(1 * kohm), name="R1")
    diode = c.place(
        devices.diode(
            saturation_current=1e-14 * A,
            emission_coefficient=1.2,
            series_resistance=10 * ohm,
        ),
        name="D1",
    )
    upper = c.place(devices.resistor(2 * kohm), name="R2")
    lower = c.place(devices.resistor(3 * kohm), name="R3")
    c.connect(source.positive, series.a, name="in")
    c.connect(series.b, diode.anode, upper.a, name="a")
    c.connect(upper.b, lower.a, name="out")
    c.connect(source.negative, diode.cathode, lower.b, ground)


@component
def _three_pole_filter(c: Builder) -> None:
    """Three-pole buffered active RC filter with DC gain 5 V/V."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1 * V), name="V1")
    stage_one = c.place(devices.voltage_controlled_voltage_source(5.0), name="E1")
    resistor_one = c.place(devices.resistor(1 * kohm), name="R1")
    capacitor_one = c.place(devices.capacitor(1 * uF), name="C1")
    stage_two = c.place(devices.voltage_controlled_voltage_source(1.0), name="E2")
    resistor_two = c.place(devices.resistor(1 * kohm), name="R2")
    capacitor_two = c.place(devices.capacitor(1 * uF), name="C2")
    stage_three = c.place(devices.voltage_controlled_voltage_source(1.0), name="E3")
    resistor_three = c.place(devices.resistor(1 * kohm), name="R3")
    capacitor_three = c.place(devices.capacitor(1 * uF), name="C3")
    c.connect(source.positive, stage_one.control_positive, name="in")
    c.connect(stage_one.positive, resistor_one.a, name="g")
    c.connect(resistor_one.b, capacitor_one.a, stage_two.control_positive, name="p1")
    c.connect(stage_two.positive, resistor_two.a, name="b1")
    c.connect(resistor_two.b, capacitor_two.a, stage_three.control_positive, name="p2")
    c.connect(stage_three.positive, resistor_three.a, name="b2")
    c.connect(resistor_three.b, capacitor_three.a, name="out")
    c.connect(
        source.negative,
        stage_one.negative,
        stage_one.control_negative,
        capacitor_one.b,
        stage_two.negative,
        stage_two.control_negative,
        capacitor_two.b,
        stage_three.negative,
        stage_three.control_negative,
        capacitor_three.b,
        ground,
    )


@component
def _underdamped_rlc(c: Builder) -> None:
    """Underdamped series RLC step circuit with ~35% overshoot."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(5 * V), name="V1")
    resistor = c.place(devices.resistor(20 * ohm), name="R1")
    inductor = c.place(devices.inductor(1 * mH), name="L1")
    capacitor = c.place(devices.capacitor(1 * uF), name="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, inductor.a, name="mid")
    c.connect(inductor.b, capacitor.a, name="out")
    c.connect(source.negative, capacitor.b, ground)


@component
def _rectifier(c: Builder) -> None:
    """Driven 1 MHz diode rectifier for frequency and distortion differentiation."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(
        devices.voltage_source(
            waveforms.Sine(offset=0.2 * V, amplitude=1 * V, frequency=1 * MHz)
        ),
        name="V1",
    )
    series = c.place(devices.resistor(1 * kohm), name="R1")
    diode = c.place(
        devices.diode(
            saturation_current=1e-14 * A,
            series_resistance=10 * ohm,
            transit_time=1 * ns,
            junction_capacitance=2 * pF,
        ),
        name="D1",
    )
    capacitor = c.place(devices.capacitor(100 * pF), name="C1")
    load = c.place(devices.resistor(10 * kohm), name="R2")
    c.connect(source.positive, series.a, name="in")
    c.connect(series.b, diode.anode, name="a")
    c.connect(diode.cathode, capacitor.a, load.a, name="out")
    c.connect(source.negative, capacitor.b, load.b, ground)


@component
def _van_der_pol_tank(c: Builder) -> None:
    """Autonomous LC tank oscillator with Van der Pol amplitude regulation."""
    inductance = 1e-6
    capacitance = 1e-9
    impedance = math.sqrt(inductance / capacitance)
    expression = (
        f"-{0.2 / impedance!r} * v(x) * (1 - (v(x) ^ 2 + ({impedance!r} * i(p)) ^ 2))"
    )
    ground = c.port("ground", Role.GROUND)
    capacitor = c.place(devices.capacitor(capacitance * F), name="C1")
    inductor = c.place(devices.inductor(inductance * H), name="L1")
    regulation = c.place(devices.behavioral_current_source(expression), name="B1")
    c.connect(capacitor.a, inductor.a, regulation.positive, regulation.x, name="t")
    c.connect(inductor.b, regulation.p_positive, name="l")
    c.connect(capacitor.b, regulation.negative, regulation.p_negative, ground)


def test_operating_point_and_dc_sweep_measure_derivatives() -> None:
    parameters: Parameters = {
        "R1": {"resistance": 1e3},
        "R2": {"resistance": 2e3},
        "R3": {"resistance": 3e3},
    }
    tight = _tight_options()
    operating = function(_diode_network(), OperatingPoint(), tight)
    op_measures = (
        specs.value("op", "out"),
        specs.value("op", "out", reference="a"),
    )
    _assert_measure_derivatives_match_central_differences(
        operating, parameters, op_measures, SMOOTH_FINITE_DIFFERENCE
    )

    # Sparse RealCotangent passed directly to jolty.vjp pullback matches bit-for-bit.
    point_result = Simulation(_diode_network()).run(OperatingPoint(), tight)
    _, cotangent = op_measures[1].cotangent(point_result)
    assert isinstance(cotangent, specs.RealCotangent)
    _, pull = vjp(operating, parameters)
    via_vjp = pull(cotangent)
    via_helper = measure_derivatives(operating, parameters, [op_measures[1]])[0]
    assert via_vjp == via_helper.derivatives

    sweep = function(
        _diode_network(),
        DcSweep("V1", [1 * V, 2 * V, 3 * V, 4 * V, 5 * V]),
        tight,
    )
    sweep_measures = (
        specs.value("dc", "out", at=2.4 * V),
        specs.maximum("dc", "out", start=1.5 * V, stop=4.5 * V),
        specs.minimum("dc", "out", start=1.5 * V, stop=4.5 * V),
        specs.peak_to_peak("dc", "out", start=1.5 * V, stop=4.5 * V),
        specs.mean("dc", "out", start=1.5 * V, stop=4.5 * V),
        specs.rms("dc", "out", start=1.5 * V, stop=4.5 * V),
    )
    _assert_measure_derivatives_match_central_differences(
        sweep, parameters, sweep_measures, SMOOTH_FINITE_DIFFERENCE
    )


def test_ac_measure_derivatives_match_central_differences() -> None:
    analysis = Ac("V1", decade(10 * Hz, 10 * kHz, 10))
    tight = _tight_options()
    evaluated = function(_three_pole_filter(), analysis, tight)
    parameters: Parameters = {
        "E1": {"gain": 5.0},
        "R1": {"resistance": 1e3},
        "C2": {"capacitance": 1e-6},
    }
    measures = (
        specs.gain("ac", "out", at=180 * Hz),
        specs.phase("ac", "out", at=180 * Hz, over="in"),
        specs.bandwidth("ac", "out", drop=3.0),
        specs.crossover("ac", "out", over="in"),
        specs.phase_margin("ac", "out"),
        specs.gain_margin("ac", "out", over="in"),
    )
    _assert_measure_derivatives_match_central_differences(
        evaluated, parameters, measures, SMOOTH_FINITE_DIFFERENCE
    )

    # Sparse ComplexCotangent passed directly to jolty.vjp pullback matches bit-for-bit.
    response = Simulation(_three_pole_filter()).run(analysis, tight)
    _, cotangent = measures[2].cotangent(response)
    assert isinstance(cotangent, specs.ComplexCotangent)
    _, pull = vjp(evaluated, parameters)
    via_vjp = pull(cotangent)
    via_helper = measure_derivatives(evaluated, parameters, [measures[2]])[0]
    assert via_vjp == via_helper.derivatives


def test_transient_step_measure_derivatives_match_central_differences() -> None:
    analysis = Transient(
        600 * us, max_step=2 * us, method="trapezoidal", operating_point=False
    )
    evaluated = function(_underdamped_rlc(), analysis, _tight_options(fixed_steps=True))
    parameters: Parameters = {
        "R1": {"resistance": 20.0},
        "L1": {"inductance": 1e-3},
        "C1": {"capacitance": 1e-6},
    }
    measures = (
        specs.value("step", "out", at=130 * us),
        specs.maximum("step", "out"),
        specs.minimum("step", "out"),
        specs.peak_to_peak("step", "out"),
        specs.mean("step", "out"),
        specs.rms("step", "out"),
        specs.rise_time("step", "out", lower=0.1, upper=0.9),
        specs.overshoot("step", "out"),
        specs.settling_time("step", "out", band=0.05),
        specs.crossing("step", "out", 2.5 * V, edge="rising"),
    )
    _assert_measure_derivatives_match_central_differences(
        evaluated, parameters, measures, TIME_DOMAIN_FINITE_DIFFERENCE
    )


def test_transient_frequency_and_distortion_derivatives_match_central_differences() -> (
    None
):
    analysis = Transient(3 * us, max_step=10 * ns, method="trapezoidal")
    evaluated = function(_rectifier(), analysis, _tight_options(fixed_steps=True))
    parameters: Parameters = {
        "R1": {"resistance": 1e3},
        "C1": {"capacitance": 1e-10},
    }
    measures = (
        specs.frequency("rect", "a"),
        specs.distortion("rect", "a", fundamental=1 * MHz, harmonics=5),
    )
    _assert_measure_derivatives_match_central_differences(
        evaluated, parameters, measures, TIME_DOMAIN_FINITE_DIFFERENCE
    )


def test_periodic_steady_state_period_and_waveform_derivatives_match_finite_diff() -> (
    None
):
    analysis = PeriodicSteadyState()
    options = _tight_options(fixed_steps=True)
    evaluated = function(_van_der_pol_tank(), analysis, options)
    parameters: Parameters = {
        "C1": {"capacitance": 1e-9},
        "L1": {"inductance": 1e-6},
    }
    measures = (
        specs.period("osc"),
        specs.value("osc", "t", at=70 * ns),
        specs.crossing("osc", "t", 0.25 * V, edge="rising"),
    )
    _assert_measure_derivatives_match_central_differences(
        evaluated, parameters, measures, TIME_DOMAIN_FINITE_DIFFERENCE
    )

    # Verify `specs.period` reads a PSS result and refuses a non-PSS result.
    steady = Simulation(_van_der_pol_tank()).run(analysis, options)
    period_reading = specs.period("osc").read(steady)
    assert period_reading.value == steady.period
    with pytest.raises(specs.MeasureError, match="periodic steady state"):
        specs.period("op").read(Simulation(_diode_network()).run(OperatingPoint()))


@component
def _divider(c: Builder) -> None:
    """Resistive divider (5 V, 1 kOhm / 1.5 kOhm -> 3.0 V nominal output)."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(5 * V), name="V1")
    upper = c.place(devices.resistor(1 * kohm), name="R1")
    lower = c.place(devices.resistor(1.5 * kohm), name="R2")
    c.connect(source.positive, upper.a, name="in")
    c.connect(upper.b, lower.a, name="out")
    c.connect(lower.b, source.negative, ground)


def test_requirement_slack_derivatives_match_central_differences_for_all_bounds() -> (
    None
):
    evaluated = function(_divider(), OperatingPoint(), _tight_options())
    parameters: Parameters = {
        "R1": {"resistance": 1e3},
        "R2": {"resistance": 1.5e3},
    }
    out = specs.value("op", "out")
    base_measure = measure_derivatives(evaluated, parameters, [out])[0]
    requirements = (
        out >= 2.0 * V,  # Lower bound only (+1).
        out <= 2.5 * V,  # Upper bound only (-1, failing).
        out.between(2.8 * V, 4.0 * V),  # Two-sided, lower bound active (+1).
        out.between(1.0 * V, 3.2 * V),  # Two-sided, upper bound active (-1).
        # Center on the solved value (half-width 0.5 V is binary-exact in [2, 4)) so
        # lower and upper slacks tie bit-for-bit across platforms.
        out.within(base_measure.reading.value * V, 0.5 * V),
    )
    derived = slack_derivatives(evaluated, parameters, requirements)

    # Exact midpoint tie deterministically takes the lower-bound (+1) sign.
    assert derived[4].derivatives == base_measure.derivatives

    # Compare all non-tie requirements against central finite differences on slack.
    def eval_slacks(point: Parameters) -> list[float]:
        wrapped = devices.wrap_overrides(evaluated.elaborated.circuit, point)
        solved = Simulation.from_elaborated(evaluated.elaborated, wrapped).run(
            OperatingPoint(), evaluated.options
        )
        return [
            requirement.judge(requirement.measure.read(solved))[1]
            for requirement in requirements[:4]
        ]

    for instance, named in parameters.items():
        for parameter, value in named.items():
            step = RELATIVE_STEP * abs(value)
            above = eval_slacks(
                _perturbed(parameters, instance, parameter, value + step)
            )
            below = eval_slacks(
                _perturbed(parameters, instance, parameter, value - step)
            )
            for item, high, low in zip(derived[:4], above, below, strict=True):
                exact = item.derivatives[instance][parameter]
                finite = (high - low) / (2.0 * step)
                scale = max(abs(exact), abs(finite), 1e-12)
                # Central difference O(step^2) bounded by SMOOTH_FINITE_DIFFERENCE.
                assert abs(exact - finite) <= SMOOTH_FINITE_DIFFERENCE * scale

    # Verify exact sign flips between lower-active and upper-active bounds.
    assert derived[0].derivatives["R2"]["resistance"] > 0.0
    assert derived[1].derivatives["R2"]["resistance"] < 0.0
    assert derived[2].derivatives["R2"]["resistance"] > 0.0
    assert derived[3].derivatives["R2"]["resistance"] < 0.0

    # Verify Requirement.slack_cotangent matches sign directly on OperatingPointResult.
    point = Simulation(_divider()).run(OperatingPoint())
    _, _, lower_cotangent = requirements[2].slack_cotangent(point)
    _, _, upper_cotangent = requirements[3].slack_cotangent(point)
    assert isinstance(lower_cotangent, specs.RealCotangent)
    assert isinstance(upper_cotangent, specs.RealCotangent)
    assert lower_cotangent.scale(-1.0) == upper_cotangent


@component
def _temperature_rc_filter(c: Builder) -> None:
    """RC low-pass filter with positive resistance temperature coefficient."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1 * V), name="V1")
    resistor = c.place(
        devices.resistor(1 * kohm, temperature_coefficient=4e-3 / K), name="R1"
    )
    capacitor = c.place(devices.capacitor(1 * nF), name="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)


def test_requirement_derivatives_evaluates_at_each_requirements_worst_corner() -> None:
    bode = Ac("V1", decade(1 * kHz, 10 * MHz, 25))
    plan = Plan(
        {"bode": bode, "states": OperatingPoints()},
        options=_tight_options(),
        corners={
            "cold": Corner(temperature=celsius(-40)),
            "hot": Corner(temperature=celsius(125)),
        },
    )
    specifications = Specs(
        [
            specs.bandwidth("bode", "out") >= 150 * kHz,
            specs.gain("bode", "out", at=1 * MHz) <= -18,
            specs.solutions("states") >= 1,
        ]
    )
    parameters = [("R1", "resistance"), ("C1", "capacitance")]
    derived = requirement_derivatives(
        _temperature_rc_filter(), plan, specifications, parameters
    )

    bandwidth_item, gain_item, solutions_item = derived
    assert bandwidth_item.verdict.corner == "hot"
    assert bandwidth_item.verdict.status == "failed"
    assert gain_item.verdict.corner == "cold"
    assert gain_item.verdict.status == "failed"
    assert solutions_item.verdict.status == "passed"
    assert solutions_item.derivatives == {}
    assert solutions_item.reason is not None

    # Verify each failing requirement's derivative matches central differences AT its
    # worst temperature corner (`hot` for bandwidth, `cold` for 1 MHz gain).
    corner_temperatures = {"cold": celsius(-40), "hot": celsius(125)}
    nominals = {"R1": ("resistance", 1e3), "C1": ("capacitance", 1e-9)}
    for item in (bandwidth_item, gain_item):
        assert item.verdict.corner is not None
        temperature = corner_temperatures[item.verdict.corner]
        for instance, (parameter, nominal) in nominals.items():
            step = RELATIVE_STEP * nominal
            slacks: list[float] = []
            for sign in (1.0, -1.0):
                unit = ohm if parameter == "resistance" else F
                simulation = Simulation(
                    _temperature_rc_filter(),
                    temperature=temperature,
                    overrides={instance: {parameter: (nominal + sign * step) * unit}},
                )
                solved = simulation.run(bode, plan.options)
                _, slack = item.requirement.judge(item.requirement.measure.read(solved))
                slacks.append(slack)
            exact = item.derivatives[instance][parameter]
            finite = (slacks[0] - slacks[1]) / (2.0 * step)
            scale = max(abs(exact), abs(finite), 1e-12)
            # Central difference O(step^2) = 1e-8 bounded by SMOOTH_FINITE_DIFFERENCE.
            assert abs(exact - finite) <= SMOOTH_FINITE_DIFFERENCE * scale


def _exhaustive_corner_slacks(
    built: ir.Circuit,
    analysis: OperatingPoint,
    requirement: specs.Requirement,
    varied: Sequence[variation.Toleranced],
) -> list[float]:
    """Evaluates `requirement` slack at every explicit `(instance, lot)` extreme."""
    lot_keys = list(
        dict.fromkeys(
            key for parameter in varied if (key := variation.lot(parameter)) is not None
        )
    )
    slacks: list[float] = []
    for instance_signs in itertools.product((-1.0, 1.0), repeat=len(varied)):
        for lot_signs in itertools.product((-1.0, 1.0), repeat=len(lot_keys)):
            lots = dict(zip(lot_keys, lot_signs, strict=True))
            overrides = variation.extreme(varied, instance_signs, lots)
            point = Simulation(built, overrides=overrides).run(analysis)
            _, slack = requirement.judge(requirement.measure.read(point))
            slacks.append(slack)
    return slacks


def test_linear_worst_case_matches_explicit_corners_with_independent_and_lot_tols() -> (
    None
):
    pack = ir.Lot(name="pack", relative=0.05)

    @component
    def series_network(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.current_source(1 * mA), name="I1")
        r1 = c.place(
            devices.resistor(1 * kohm).toleranced("resistance", 0.02, lot=pack),
            name="R1",
        )
        r2 = c.place(
            devices.resistor(2 * kohm).toleranced("resistance", 0.01, lot=pack),
            name="R2",
        )
        r3 = c.place(
            devices.resistor(500 * ohm).toleranced(
                "resistance", 0.04, distribution="log_uniform"
            ),
            name="R3",
        )
        c.connect(source.negative, r1.a, name="out")
        c.connect(r1.b, r2.a, name="n1")
        c.connect(r2.b, r3.a, name="n2")
        c.connect(r3.b, source.positive, ground)

    built = Simulation(series_network()).circuit
    varied = variation.toleranced(built)
    plan = Plan({"op": OperatingPoint()})
    out = specs.value("op", "out")
    specifications = Specs(
        [
            out >= 3.2 * V,
            out <= 3.75 * V,
            out.within(3.5 * V, 0.25 * V),  # Exact nominal midpoint 3.5 V!
        ]
    )
    estimates = linear_worst_case(built, plan, specifications)
    for estimate in estimates:
        # Compare against simulating at the returned worst-case overrides.
        corner_point = Simulation(built, overrides=estimate.overrides).run(
            OperatingPoint()
        )
        corner_reading = estimate.requirement.measure.read(corner_point)
        _, corner_slack = estimate.requirement.judge(corner_reading)
        # Linear network: first-order estimate equals explicit corner up to LU rounding.
        assert estimate.estimated_value == pytest.approx(
            corner_reading.value, rel=LINEAR_CORNER_ROUNDING, abs=1e-12
        )
        assert estimate.estimated_slack == pytest.approx(
            corner_slack, rel=LINEAR_CORNER_ROUNDING, abs=1e-12
        )
        # Compare against exhaustive 2^4 = 16 explicit tolerance+lot corners.
        all_slacks = _exhaustive_corner_slacks(
            built, OperatingPoint(), estimate.requirement, varied
        )
        assert estimate.estimated_slack == pytest.approx(
            min(all_slacks), rel=LINEAR_CORNER_ROUNDING, abs=1e-12
        )


def test_linear_worst_case_handles_opposite_sign_shared_lots_and_cancellation() -> None:
    pack = ir.Lot(name="pack", relative=0.05)

    @component
    def matched_pair(
        c: Builder,
        first_resistance: float,
        second_resistance: float,
        individual_tolerance: float,
    ) -> None:
        ground = c.port("ground", Role.GROUND)
        source_a = c.place(devices.current_source(1 * mA), name="I1")
        source_b = c.place(devices.current_source(1 * mA), name="I2")
        r1 = c.place(
            devices.resistor(first_resistance * ohm).toleranced(
                "resistance", individual_tolerance, lot=pack
            ),
            name="R1",
        )
        r2 = c.place(
            devices.resistor(second_resistance * ohm).toleranced(
                "resistance", individual_tolerance, lot=pack
            ),
            name="R2",
        )
        c.connect(source_a.negative, r1.a, name="a")
        c.connect(source_b.negative, r2.a, name="b")
        c.connect(r1.b, r2.b, source_a.positive, source_b.positive, ground)

    # Equal nominals with pure lot tolerance cancel completely (worst-case shift = 0).
    equal_built = Simulation(matched_pair(2e3, 2e3, 0.0)).circuit
    diff = specs.value("op", "a", reference="b")
    (cancelled,) = linear_worst_case(
        equal_built,
        Plan({"op": OperatingPoint()}),
        Specs([diff.within(0 * V, 0.1 * V)]),
    )
    # Exact differential cancellation on identical shared-lot resistors (rounding only).
    assert cancelled.estimated_value == pytest.approx(0.0, abs=1e-12)
    assert cancelled.estimated_slack == pytest.approx(
        0.1, rel=LINEAR_CORNER_ROUNDING, abs=1e-12
    )

    # Unequal nominals (3 kOhm vs 1 kOhm) with 1% part + 5% lot match all 8 corners.
    unequal_built = Simulation(matched_pair(3e3, 1e3, 0.01)).circuit
    unequal_varied = variation.toleranced(unequal_built)
    requirement = diff >= 1.9 * V
    (estimate,) = linear_worst_case(
        unequal_built, Plan({"op": OperatingPoint()}), Specs([requirement])
    )
    all_slacks = _exhaustive_corner_slacks(
        unequal_built, OperatingPoint(), requirement, unequal_varied
    )
    # Linear network: first-order estimate matches minimum of all 8 explicit corners.
    assert estimate.estimated_slack == pytest.approx(
        min(all_slacks), rel=LINEAR_CORNER_ROUNDING, abs=1e-12
    )
    assert estimate.status == "failed"

    # Pure `estimate_worst_case` with empty toleranced list preserves nominal.
    unchanged = estimate_worst_case(
        requirement,
        estimate.nominal,
        estimate.derivatives,
        (),
    )
    assert unchanged.estimated_slack == estimate.nominal_slack
    assert unchanged.overrides == {}


def test_specs_report_surfaces_largest_slack_derivatives_on_failing_requirements() -> (
    None
):
    verdicts = (
        specs.Verdict(
            requirement="out in op ≥ 3 V",
            status="failed",
            value=2.5,
            unit="V",
            slack=-0.5,
        ),
        specs.Verdict(
            requirement="gain in ac ≥ -3 dB",
            status="passed",
            value=-1.0,
            unit="dB",
            slack=2.0,
        ),
    )
    # Omitting derivatives leaves formatted output identical to `specs.describe`.
    plain_reports = report.specs_report(verdicts)
    assert report.describe_specs(plain_reports) == specs.describe(verdicts)

    derivatives: Sequence[Mapping[str, Mapping[str, float]] | None] = (
        {
            "R1": {"resistance": -1.25e-3},
            "R2": {"resistance": 2.5e-3},
            "R3": {"resistance": 0.0},
            "R4": {"resistance": -5e-4},
        },
        {"R1": {"resistance": 10.0}},
    )
    detailed = report.specs_report(verdicts, derivatives, max_derivatives=2)
    failing_report, passing_report = detailed
    assert failing_report.derivatives == (
        report.SlackDerivative(
            instance="R2", parameter="resistance", derivative=2.5e-3
        ),
        report.SlackDerivative(
            instance="R1", parameter="resistance", derivative=-1.25e-3
        ),
    )
    assert failing_report.unlisted == 1
    assert passing_report.derivatives == ()
    assert passing_report.describe() == verdicts[1].describe()
    rendered = report.describe_specs(detailed)
    assert "largest slack derivatives (first order):" in rendered
    assert "R2  resistance  0.0025" in rendered
    assert "R1  resistance  -0.00125" in rendered
    assert "... and 1 more" in rendered
    assert "R3" not in rendered

    with pytest.raises(ValueError, match="max_derivatives is at least 1"):
        report.specs_report(verdicts, max_derivatives=0)
    with pytest.raises(ValueError, match="entries for 2 verdicts"):
        report.specs_report(verdicts, [{}])


def test_non_differentiable_and_custom_measures_raise_clear_errors() -> None:
    point = Simulation(_divider()).run(OperatingPoint())
    states = Simulation(_divider()).run(OperatingPoints())
    with pytest.raises(specs.MeasureError, match="not differentiable"):
        specs.solutions("states").cotangent(states)

    custom = specs.Measure(
        analysis="op",
        description="custom out",
        plain=None,
        read=lambda result: specs.Reading(
            float(np.asarray(result.values)[0])
            if isinstance(result, results.OperatingPointResult)
            else 0.0,
            VOLTAGE,
        ),
    )
    with pytest.raises(specs.MeasureError, match="custom Python measure"):
        custom.cotangent(point)

    # Mismatched cotangent domains on `vjp` pullback raise `ValueError`.
    op_function = function(_divider(), OperatingPoint())
    _, op_pullback = vjp(op_function, {"R1": {"resistance": 1e3}})
    with pytest.raises(ValueError, match="ComplexCotangent"):
        op_pullback(specs.ComplexCotangent(((0, 0, 1.0, 0.0),)))

    ac_function = function(
        _temperature_rc_filter(), Ac("V1", decade(1 * kHz, 100 * kHz, 5))
    )
    _, ac_pullback = vjp(ac_function, {"R1": {"resistance": 1e3}})
    with pytest.raises(ValueError, match="RealCotangent"):
        ac_pullback(specs.RealCotangent(((0, 0, 1.0),), 0.0))
    with pytest.raises(ValueError, match="LoopCotangent"):
        ac_pullback(specs.LoopCotangent(((0, 1.0, 0.0),)))
    with pytest.raises(ValueError, match="NoiseCotangent"):
        op_pullback(specs.NoiseCotangent(((0, 1.0),), ()))


@component
def _three_pole_feedback_loop(c: Builder) -> None:
    """Three-stage unity-feedback amplifier broken by floating probe `VP`."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1 * V), name="VIN")
    c.connect(source.negative, ground)
    stages = [
        c.place(devices.voltage_controlled_current_source(1 * mA / V), name=f"G{i}")
        for i in range(1, 4)
    ]
    poles = ((90 * kohm, 1e3), (10 * kohm, 120e3), (1 * kohm, 7e6))
    driving = source.positive
    for index, (stage, (load, pole)) in enumerate(zip(stages, poles, strict=True), 1):
        resistor = c.place(devices.resistor(load), name=f"R{index}")
        capacitance = 1 / (2 * math.pi * pole * load.value) * F
        capacitor = c.place(devices.capacitor(capacitance), name=f"C{index}")
        c.connect(stage.control_positive, driving)
        c.connect(stage.positive, resistor.b, capacitor.b, ground)
        c.connect(stage.negative, resistor.a, capacitor.a, name=f"n{index}")
        driving = stage.negative
    for stage in stages[1:]:
        c.connect(stage.control_negative, ground)
    buffer = c.place(devices.voltage_controlled_voltage_source(1.0), name="E1")
    c.connect(buffer.control_positive, driving)
    c.connect(buffer.control_negative, buffer.negative, ground)
    probe = c.place(devices.voltage_source(0 * V), name="VP")
    c.connect(buffer.positive, probe.negative, name="out")
    c.connect(probe.positive, stages[0].control_negative, name="fb")


@component
def _noisy_common_source(c: Builder) -> None:
    """Noisy MOSFET common-source amplifier with thermal and flicker noise."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(3.3 * V), name="VDD")
    gate = c.place(devices.voltage_source(1.2 * V), name="VG")
    source_resistor = c.place(devices.resistor(2 * kohm), name="RG")
    load = c.place(devices.resistor(10 * kohm), name="RD")
    load_cap = c.place(devices.capacitor(5 * pF), name="CL")
    transistor = c.place(
        devices.mosfet(
            width=2 * um,
            length=1 * um,
            threshold_voltage=0.7 * V,
            transconductance_parameter=200 * uA / V**2,
            channel_length_modulation=0.02 / V,
            flicker_coefficient=1e-24,
            flicker_exponent=1.2,
        ),
        name="M1",
    )
    c.connect(supply.positive, load.a, name="vdd")
    c.connect(load.b, load_cap.a, transistor.drain, name="out")
    c.connect(gate.positive, source_resistor.a, name="vg")
    c.connect(source_resistor.b, transistor.gate, name="gate")
    c.connect(
        supply.negative,
        gate.negative,
        load_cap.b,
        transistor.source,
        transistor.bulk,
        ground,
    )


def test_stability_measure_derivatives_match_central_differences() -> None:
    analysis = Stability("VP", decade(100 * Hz, 10 * MHz, 12))
    tight = _tight_options()
    evaluated = function(_three_pole_feedback_loop(), analysis, tight)
    parameters: Parameters = {
        "R1": {"resistance": 90e3},
        "C1": {"capacitance": 1 / (2 * math.pi * 1e3 * 90e3)},
        "C2": {"capacitance": 1 / (2 * math.pi * 120e3 * 10e3)},
    }
    measures = (
        specs.gain("stb", at=10 * kHz),
        specs.phase("stb", at=10 * kHz),
        specs.bandwidth("stb"),
        specs.crossover("stb"),
        specs.phase_margin("stb"),
        specs.gain_margin("stb"),
    )
    _assert_measure_derivatives_match_central_differences(
        evaluated, parameters, measures, SMOOTH_FINITE_DIFFERENCE
    )

    # Sparse LoopCotangent passed directly to `vjp` pullback matches bit-for-bit.
    solved = Simulation(_three_pole_feedback_loop()).run(analysis, tight)
    _, cotangent = specs.phase_margin("stb").cotangent(solved)
    assert isinstance(cotangent, specs.LoopCotangent)
    _, pullback = vjp(evaluated, parameters)
    direct = pullback(cotangent)
    (batched,) = measure_derivatives(
        evaluated, parameters, (specs.phase_margin("stb"),)
    )
    for instance, named in parameters.items():
        for name in named:
            assert direct[instance][name] == pytest.approx(
                batched.derivatives[instance][name], rel=1e-13, abs=1e-15
            )


def test_noise_measure_derivatives_match_central_differences() -> None:
    analysis = Noise("out", decade(10 * Hz, 10 * MHz, 6), source="VG")
    tight = _tight_options()
    evaluated = function(_noisy_common_source(), analysis, tight)
    parameters: Parameters = {
        "RG": {"resistance": 2e3},
        "RD": {"resistance": 10e3},
        "CL": {"capacitance": 5e-12},
        "M1": {"width": 2e-6, "flicker_coefficient": 1e-24},
    }
    measures = (
        specs.noise_density("noise", at=1 * kHz),
        specs.noise_density("noise", at=1 * kHz, referred=True),
        specs.integrated_noise("noise", start=100 * Hz, stop=1 * MHz),
        specs.integrated_noise("noise", start=100 * Hz, stop=1 * MHz, referred=True),
    )
    _assert_measure_derivatives_match_central_differences(
        evaluated, parameters, measures, SMOOTH_FINITE_DIFFERENCE
    )

    # Sparse NoiseCotangent passed directly to `vjp` pullback matches bit-for-bit.
    solved = Simulation(_noisy_common_source()).run(analysis, tight)
    _, cotangent = specs.integrated_noise(
        "noise", start=100 * Hz, stop=1 * MHz, referred=True
    ).cotangent(solved)
    assert isinstance(cotangent, specs.NoiseCotangent)
    _, pullback = vjp(evaluated, parameters)
    direct = pullback(cotangent)
    (batched,) = measure_derivatives(
        evaluated,
        parameters,
        (specs.integrated_noise("noise", start=100 * Hz, stop=1 * MHz, referred=True),),
    )
    for instance, named in parameters.items():
        for name in named:
            assert direct[instance][name] == pytest.approx(
                batched.derivatives[instance][name], rel=1e-13, abs=1e-15
            )


def test_stability_and_noise_requirement_derivatives_and_linear_worst_case() -> None:
    @component
    def toleranced_loop(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(1 * V), name="VIN")
        c.connect(source.negative, ground)
        stages = [
            c.place(devices.voltage_controlled_current_source(1 * mA / V), name=f"G{i}")
            for i in range(1, 4)
        ]
        poles = ((90 * kohm, 1e3), (10 * kohm, 120e3), (1 * kohm, 7e6))
        driving = source.positive
        for index, (stage, (load, pole)) in enumerate(
            zip(stages, poles, strict=True), 1
        ):
            resistor = c.place(
                devices.resistor(load, tolerance=ir.Tolerance(relative=0.02)),
                name=f"R{index}",
            )
            capacitance = 1 / (2 * math.pi * pole * load.value) * F
            capacitor = c.place(
                devices.capacitor(capacitance, tolerance=ir.Tolerance(relative=0.02)),
                name=f"C{index}",
            )
            c.connect(stage.control_positive, driving)
            c.connect(stage.positive, resistor.b, capacitor.b, ground)
            c.connect(stage.negative, resistor.a, capacitor.a, name=f"n{index}")
            driving = stage.negative
        for stage in stages[1:]:
            c.connect(stage.control_negative, ground)
        buffer = c.place(devices.voltage_controlled_voltage_source(1.0), name="E1")
        c.connect(buffer.control_positive, driving)
        c.connect(buffer.control_negative, buffer.negative, ground)
        probe = c.place(devices.voltage_source(0 * V), name="VP")
        c.connect(buffer.positive, probe.negative, name="out")
        c.connect(probe.positive, stages[0].control_negative, name="fb")

    plan = Plan(
        analyses={
            "stb": Stability("VP", decade(100 * Hz, 10 * MHz, 12)),
            "noise": Noise("out", decade(100 * Hz, 10 * MHz, 8), source="VIN"),
        },
        options=_tight_options(),
    )
    requirements = Specs(
        [
            specs.phase_margin("stb") >= 30,
            specs.gain_margin("stb") >= 5,
            specs.integrated_noise("noise", start=1 * kHz, stop=1 * MHz) <= 1e-3 * V,
        ]
    )
    derived = requirement_derivatives(
        toleranced_loop(),
        plan,
        requirements,
        [("R1", "resistance"), ("C1", "capacitance"), ("C2", "capacitance")],
    )
    assert len(derived) == 3
    for item in derived:
        assert item.reason is None
        assert item.derivatives["R1"]["resistance"] != 0.0

    worst = linear_worst_case(toleranced_loop(), plan, requirements)
    assert len(worst) == 3
    for estimate in worst:
        assert estimate.estimated_slack < estimate.nominal_slack
        assert estimate.overrides


def test_subcircuit_measure_and_slack_derivatives_match_central_differences() -> None:
    library = Library()
    library.read(
        ".model rsheet r rsh=1000\n"
        ".subckt sized_res a b w=1u l=2u\n"
        "R1 a b rsheet w={w} l={l}\n"
        ".ends\n",
        "sized_measure.lib",
    )
    upper = library.placement("sized_res", w=1e-6, l=2e-6)

    @component
    def sized_filter(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(1 * V), name="V1")
        resistor = c.place(upper.component(), name="X1")
        capacitor = c.place(devices.capacitor(10 * nF), name="C1")
        c.connect(source.positive, resistor.a, name="in")
        c.connect(resistor.b, capacitor.a, name="out")
        c.connect(capacitor.b, source.negative, ground)

    analysis = Ac("V1", decade(100 * Hz, 100 * kHz, 12))
    tight = _tight_options()
    evaluated = function(sized_filter(), analysis, tight, subcircuits={"X1": upper})
    parameters: Parameters = {
        "X1": {"w": 1.2e-6, "l": 2.5e-6},
        "C1": {"capacitance": 12e-9},
    }
    measures = (specs.gain("ac", "out", at=5 * kHz), specs.bandwidth("ac", "out"))
    derived = measure_derivatives(evaluated, parameters, measures)
    assert len(derived) == 2

    def read_at(point: Parameters) -> list[float]:
        built_x1 = library.component("sized_res", **point["X1"])

        @component
        def rebuilt(c: Builder) -> None:
            ground = c.port("ground", Role.GROUND)
            source = c.place(devices.voltage_source(1 * V), name="V1")
            resistor = c.place(built_x1, name="X1")
            capacitor = c.place(
                devices.capacitor(point["C1"]["capacitance"] * F), name="C1"
            )
            c.connect(source.positive, resistor.a, name="in")
            c.connect(resistor.b, capacitor.a, name="out")
            c.connect(capacitor.b, source.negative, ground)

        solved = Simulation(rebuilt()).run(analysis, tight)
        return [measure.read(solved).value for measure in measures]

    for instance, named in parameters.items():
        for name, value in named.items():
            step = RELATIVE_STEP * abs(value)
            high = read_at(_perturbed(parameters, instance, name, value + step))
            low = read_at(_perturbed(parameters, instance, name, value - step))
            for item, above, below in zip(derived, high, low, strict=True):
                exact = item.derivatives[instance][name]
                finite = (above - below) / (2.0 * step)
                assert exact == pytest.approx(
                    finite, rel=SMOOTH_FINITE_DIFFERENCE, abs=1e-12
                )

    (slack_item,) = slack_derivatives(
        evaluated, parameters, (specs.bandwidth("ac", "out") >= 5 * kHz,)
    )
    assert slack_item.reason is None
    assert slack_item.derivatives["X1"]["w"] == pytest.approx(
        derived[1].derivatives["X1"]["w"], rel=1e-13, abs=1e-15
    )
    assert slack_item.derivatives["X1"]["l"] == pytest.approx(
        derived[1].derivatives["X1"]["l"], rel=1e-13, abs=1e-15
    )
