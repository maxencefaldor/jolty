# Copyright 2026 Maxence Faldor

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from jolty import results
from jolty.runner import Built, Failure, Job, run
from jolty.specs import (
    MeasureError,
    Requirement,
    Specs,
    Verdict,
    bandwidth,
    crossing,
    crossover,
    damping,
    describe,
    diagnose,
    distortion,
    dominant_pole,
    evaluate,
    frequency,
    gain,
    gain_margin,
    integrated_noise,
    judge,
    maximum,
    mean,
    minimum,
    noise_density,
    overshoot,
    peak_to_peak,
    phase,
    phase_margin,
    rise_time,
    rms,
    settling_time,
    solutions,
    spread,
    spread_of,
    value,
    wilson,
    worst,
)
from jolty.specs import judged as judged_requirement
from jolty.units import (
    CURRENT,
    FREQUENCY,
    TEMPERATURE,
    TIME,
    VOLTAGE,
    A,
    Hz,
    K,
    V,
    kHz,
    mA,
    ms,
    nV,
    rtHz,
    s,
    us,
    uV,
)

UNKNOWNS = (
    results.Unknown("out", "voltage"),
    results.Unknown("ref", "voltage"),
    results.Unknown("V1:current", "current"),
)

TAU = 1e-6
"""The time constant of the first-order step response, in seconds."""


def point(out: float) -> results.OperatingPointResult:
    return results.OperatingPointResult(
        UNKNOWNS, np.array([out, 0.5, -out / 1e3]), "newton"
    )


def waveforms(times: results.Real, out: results.Real) -> results.TransientResult:
    values = np.column_stack([out, np.full_like(out, 0.5), -out / 1e3])
    statistics = results.Statistics("operating point", times.size, 0, 0, 0)
    return results.TransientResult(UNKNOWNS, times, values, statistics)


def response(frequencies: results.Real, out: results.Complex) -> results.AcResult:
    values = np.column_stack([out, np.full_like(out, 2), out / 1e3])
    return results.AcResult(UNKNOWNS, "V1", frequencies, values, point(0.0))


def first_order() -> results.TransientResult:
    """A unit step through a time constant of 1 µs, held for 20 time constants."""
    times = np.linspace(0, 20 * TAU, 200_001)
    return waveforms(times, 1 - np.exp(-times / TAU))


def low_pass() -> results.AcResult:
    """A first-order low pass with its corner at 1 kHz, densely sampled."""
    frequencies = np.logspace(0, 6, 6001)
    transfer = np.asarray(1 / (1 + 1j * frequencies / 1e3), dtype=np.complex128)
    return response(frequencies, transfer)


def verdict_of(requirement: Requirement, result: results.AnalysisResult) -> Verdict:
    analysis = requirement.measure.analysis
    (verdict,) = evaluate(Specs([requirement]), [analysis], {analysis: result})
    return verdict


def unread(requirement: Requirement, result: results.AnalysisResult) -> str:
    verdict = verdict_of(requirement, result)
    assert verdict.status == "unmeasured"
    assert verdict.reason is not None
    return verdict.reason


def test_a_value_reads_an_operating_point_or_interpolates_an_axis() -> None:
    assert value("op", "out").read(point(2.0)).value == 2.0
    assert value("op", "out", reference="ref").read(point(2.0)).value == 1.5
    current = value("op", "V1:current").read(point(2.0))
    assert (current.value, current.dimension) == (-2e-3, CURRENT)
    step = value("step", "out", at=TAU * s).read(first_order())
    # Exact grid point at t = TAU on first_order() (dt = 1e-4*TAU); float rounding.
    assert step.value == pytest.approx(1 - math.exp(-1))
    assert step.dimension == VOLTAGE


def test_a_value_reads_a_sweep_whose_levels_fall() -> None:
    levels = np.linspace(5, 0, 11)
    values = np.column_stack([levels / 2, levels, levels])
    sweep = results.DcSweepResult(UNKNOWNS, "V1", "voltage", levels, values)
    # Exact linear function on reversed axis; default pytest.approx covers rounding.
    assert value("dc", "out", at=1.25 * V).read(sweep).value == pytest.approx(0.625)
    assert maximum("dc", "out", stop=4 * V).read(sweep).value == pytest.approx(2)


def test_windowed_statistics_match_a_sine_over_whole_periods() -> None:
    """Windowed stats on 3 periods of 1 kHz sine at dt = 1e-4*T (10,000 pts/period).

    Integer quarter-periods hit exact extrema within float rounding (`rel=1e-6`),
    full-cycle trapezoidal mean cancels to rounding (`abs=1e-9`), and half-cycle
    trapezoidal quadrature
    has O((omega*dt)^2 / 12) ~= 3.3e-8 error (bounded by rel=1e-6).
    """
    times = np.linspace(0, 3 * ms.value, 30_001)
    sine = waveforms(times, 2 * np.sin(2 * np.pi * 1e3 * times))
    assert maximum("t", "out").read(sine).value == pytest.approx(2, rel=1e-6)
    assert minimum("t", "out").read(sine).value == pytest.approx(-2, rel=1e-6)
    assert peak_to_peak("t", "out").read(sine).value == pytest.approx(4, rel=1e-6)
    assert mean("t", "out").read(sine).value == pytest.approx(0, abs=1e-9)
    assert rms("t", "out").read(sine).value == pytest.approx(math.sqrt(2), rel=1e-6)
    measured = frequency("t", "out").read(sine)
    assert (measured.value, measured.dimension) == (pytest.approx(1e3), FREQUENCY)
    half = mean("t", "out", start=0 * s, stop=0.5 * ms).read(sine).value
    assert half == pytest.approx(4 / math.pi, rel=1e-6)


def test_edges_match_a_first_order_step() -> None:
    """Edge timings on first_order() sampled at dt = 1e-4 * TAU.

    Linearly interpolated crossings (`rise_time`, `crossing`) have `O(dt^2) ~= 1e-8`
    chord error (within `pytest.approx` `rel=1e-6`); uninterpolated `settling_time`
    quantizes to `dt`,
    relative error <= dt / (TAU * ln(50)) ~= 2.55e-5 (bounded by rel=1e-4).
    """
    step = first_order()
    rise = rise_time("t", "out").read(step)
    assert (rise.value, rise.dimension) == (pytest.approx(TAU * math.log(9)), TIME)
    settling = settling_time("t", "out").read(step).value
    assert settling == pytest.approx(TAU * math.log(50), rel=1e-4)
    assert overshoot("t", "out").read(step).value == 0
    half = crossing("t", "out", 0.5 * V, edge="rising").read(step).value
    assert half == pytest.approx(TAU * math.log(2))


def test_a_fall_time_reads_like_a_rise_time() -> None:
    step = first_order()
    falling = waveforms(step.times, 1 - step.values[:, 0])
    # Linear crossing interpolation on dt = 1e-4*TAU gives O(dt^2) ~= 1e-8 chord error.
    assert rise_time("t", "out").read(falling).value == pytest.approx(TAU * math.log(9))
    assert not crossing("t", "out", 0.5 * V, edge="rising").read(step).value < 0
    with pytest.raises(MeasureError, match="does not cross 500 mV"):
        crossing("t", "out", 0.5 * V, edge="rising").read(falling)


def test_overshoot_matches_an_underdamped_second_order_step() -> None:
    damping, natural = 0.3, 2 * math.pi * 1e6
    damped = natural * math.sqrt(1 - damping**2)
    times = np.linspace(0, 20e-6, 400_001)
    envelope = np.exp(-damping * natural * times)
    ringing = np.cos(damped * times) + damping * natural / damped * np.sin(
        damped * times
    )
    step = waveforms(times, 1 - envelope * ringing)
    expected = 100 * math.exp(-math.pi * damping / math.sqrt(1 - damping**2))
    measured = overshoot("t", "out").read(step)
    # Discrete peak miss on dt = 5e-11 s: (damped * dt)^2 / 8 ~= 1.1e-8 < rel=1e-6.
    assert (measured.value, measured.unit) == (pytest.approx(expected, rel=1e-6), "%")


def test_a_first_order_response_falls_3_db_at_its_corner() -> None:
    """On low_pass() (1000 pts/decade log grid), 1 kHz and 1 Hz are exact grid points.

    Default pytest.approx (rel=1e-6) and abs=1e-6 cover dB/atan2 float rounding and
    log-linear bandwidth interpolation.
    """
    corner = low_pass()
    assert gain("ac", "out", at=1 * kHz).read(corner).value == pytest.approx(
        -10 * math.log10(2), abs=1e-6
    )
    assert phase("ac", "out", at=1 * kHz).read(corner).value == pytest.approx(-45)
    measured = bandwidth("ac", "out", drop=10 * math.log10(2)).read(corner)
    assert (measured.value, measured.dimension) == (pytest.approx(1e3), FREQUENCY)
    over = gain("ac", "out", over="ref", at=1 * Hz).read(corner).value
    assert over == pytest.approx(20 * math.log10(abs(1 / (1 + 1e-3j)) / 2))


def test_a_response_of_exactly_nothing_reads_minus_infinite_decibels() -> None:
    """A twin T at its notch, or a balanced bridge, passes nothing: -inf dB, a number.

    Its verdict passes an upper bound with infinite slack and fails a lower one, and
    JSON, which has no infinity, writes it as a string its schema names.
    """
    frequencies = np.array([1e2, 1e3, 1e4])
    silent = response(frequencies, np.array([1e-3, 0, 1e-3], dtype=np.complex128))
    notch = verdict_of(gain("ac", "out", at=1 * kHz) <= -100, silent)
    assert (notch.status, notch.value, notch.slack) == ("passed", -math.inf, math.inf)
    assert notch.describe().startswith(
        "passed: gain of out at 1 kHz in ac ≤ -100 dB: read -inf dB, slack inf dB\n"
    )
    written = notch.model_dump_json()
    assert '"value":"-Infinity"' in written
    assert '"slack":"Infinity"' in written
    assert Verdict.model_validate_json(written) == notch
    extended = Verdict.model_json_schema(mode="serialization")["$defs"]["ExtendedReal"]
    assert {"enum": ["Infinity", "-Infinity"]} in extended["anyOf"]
    deep = verdict_of(gain("ac", "out", at=1 * kHz) >= -100, silent)
    assert (deep.status, deep.slack) == ("failed", -math.inf)
    passing = verdict_of(gain("ac", "out", at=100 * Hz) <= -100, silent)
    specs = Specs([gain("ac", "out", at=1 * kHz) <= -100])
    spreads, _ = spread(specs, [(notch,), (passing,)])
    assert spreads[0].values is not None
    assert (spreads[0].values.minimum, spreads[0].values.mean) == (-math.inf, -math.inf)
    assert spreads[0].values.standard_deviation is None
    assert spreads[0].describe(2).endswith("mean -inf dB, standard deviation undefined")


def test_the_phase_margin_of_an_integrator_with_a_pole_at_unity() -> None:
    frequencies = np.logspace(-3, 3, 60_001)
    loop = 1 / (1j * frequencies) / (1 + 1j * frequencies)
    unity = math.sqrt((math.sqrt(5) - 1) / 2)
    expected = 90 - math.degrees(math.atan(unity))
    measured = phase_margin("ac", "out").read(response(frequencies, loop))
    # Log-linear interpolation at 10,000 pts/decade: O(h^2) chord error < 1e-6 deg.
    assert (measured.value, measured.unit) == (pytest.approx(expected, abs=1e-6), "°")


def loop_gain(
    frequencies: results.Real, gains: results.Complex
) -> results.StabilityResult:
    return results.StabilityResult(UNKNOWNS, "VP", frequencies, gains, point(0.0))


def test_a_loop_gains_margins_are_read_without_an_output() -> None:
    frequencies = np.logspace(-3, 3, 60_001)
    gains = np.asarray(10 / (1 + 1j * frequencies) ** 3, dtype=np.complex128)
    loop = loop_gain(frequencies, gains)
    unity = math.sqrt(10 ** (2 / 3) - 1)
    inverted = math.tan(math.radians(60))
    expected = {
        crossover("loop"): (unity, FREQUENCY),
        phase_margin("loop"): (180 - 3 * math.degrees(math.atan(unity)), "°"),
        gain_margin("loop"): (-20 * math.log10(10 / (1 + inverted**2) ** 1.5), "dB"),
        gain("loop", at=1e-3 * Hz): (20.0, "dB"),
        phase("loop", at=1 * Hz): (-135.0, "°"),
    }
    for measure, (value_read, unit) in expected.items():
        measured = measure.read(loop)
        assert measured.unit == unit, measure
        # Log-linear crossing interpolation at 10,000 pts/decade: O(h^2) error < 1e-6.
        assert measured.value == pytest.approx(value_read, rel=1e-6), measure
    assert str(phase_margin("loop") >= 45) == "phase margin of the loop in loop ≥ 45 °"
    assert str(gain_margin("loop") >= 6) == "gain margin of the loop in loop ≥ 6 dB"


def test_a_loop_short_of_its_margins_raises_their_rules() -> None:
    frequencies = np.logspace(-3, 3, 60_001)
    # Three poles under a gain of 10: about -7° of phase margin and -1.9 dB of gain.
    poor = np.asarray(10 / (1 + 1j * frequencies) ** 3, dtype=np.complex128)
    raised = diagnose({"loop": loop_gain(frequencies, poor)})
    assert [diagnostic.rule for diagnostic in raised] == [
        "low-phase-margin",
        "low-gain-margin",
    ]
    unity = math.sqrt(10 ** (2 / 3) - 1)
    margin = 180 - 3 * math.degrees(math.atan(unity))
    assert raised[0].message.startswith("the loop broken at `VP` has a phase margin of")
    # Log-linear interpolation at 10,000 pts/decade plus formatted text rounding (1e-6).
    assert float(raised[0].message.split(" of ")[1].split("°")[0]) == pytest.approx(
        margin, rel=1e-6
    )
    assert raised[1].message.startswith(
        "the loop broken at `VP` has a gain margin of -"
    )
    # An integrator crosses unity with 90° to spare and never inverts.
    sound = np.asarray(1 / (1j * frequencies), dtype=np.complex128)
    assert diagnose({"loop": loop_gain(frequencies, sound), "ac": low_pass()}) == ()


def test_a_loop_gain_is_read_without_an_output_and_a_response_at_one() -> None:
    frequencies = np.logspace(0, 2, 3)
    loop = loop_gain(frequencies, np.full(3, 0.5 + 0j))
    assert unread(phase_margin("loop", "out") >= 45, loop) == (
        "a loop gain is read without an output, not at out"
    )
    assert unread(phase_margin("ac") >= 45, low_pass()) == (
        "a small-signal response is read at an output"
    )
    assert unread(gain_margin("loop") >= 6, loop) == (
        "the phase of the loop never falls through -180°"
    )
    assert unread(crossover("loop") >= 1 * Hz, loop) == (
        "the gain of the loop never falls through 0 dB"
    )
    with pytest.raises(ValueError, match="without over or a reference"):
        phase_margin("loop", over="in")


def test_distortion_reads_the_harmonics_of_the_last_period() -> None:
    """Linear resampling onto 1024 pts/period from dt=1e-4*T bounds THD error.

    Chord sag `O((k*w*dt)^2)` attenuates harmonics by `< 1e-6` linear, well below
    `abs=1e-3` dB for all three `distortion()` assertions below.
    """
    times = np.linspace(0, 3e-3, 30_001)
    angle = 2 * math.pi * 1e3 * times
    settled = 0.2 + np.sin(angle) + 0.01 * np.sin(2 * angle) + 0.005 * np.cos(3 * angle)
    start_up = np.where(times < 1e-3, np.sin(7 * angle), 0.0)
    signal = waveforms(times, settled + start_up)
    expected = 20 * math.log10(math.hypot(0.01, 0.005))
    measured = distortion("t", "out", fundamental=1 * kHz).read(signal)
    assert (measured.value, measured.unit) == (pytest.approx(expected, abs=1e-3), "dB")
    two = distortion("t", "out", fundamental=1 * kHz, start=1 * ms).read(signal)
    assert two.value == pytest.approx(expected, abs=1e-3)
    second = distortion("t", "out", fundamental=1 * kHz, harmonics=2).read(signal)
    assert second.value == pytest.approx(-40.0, abs=1e-3)
    clean = waveforms(times, np.sin(angle))
    assert distortion("t", "out", fundamental=1 * kHz).read(clean).value < -150
    assert str(distortion("t", "out", fundamental=1 * kHz) <= -40) == (
        "distortion of out at 1 kHz in t ≤ -40 dB"
    )


def test_distortion_needs_a_period_and_a_fundamental() -> None:
    times = np.linspace(0, 1e-3, 1001)
    short = waveforms(times, np.sin(2 * math.pi * 1e3 * times))
    assert unread(distortion("t", "out", fundamental=500 * Hz) <= 0, short) == (
        "out spans less than a period of 500 Hz"
    )
    flat = waveforms(times, np.zeros_like(times))
    assert unread(distortion("t", "out", fundamental=1 * kHz) <= 0, flat) == (
        "out has no component at 1 kHz"
    )
    with pytest.raises(ValueError, match="between 2 and 511, not 1"):
        distortion("t", "out", fundamental=1 * kHz, harmonics=1)
    with pytest.raises(ValueError, match="positive and finite"):
        distortion("t", "out", fundamental=0 * Hz)


def test_requirements_state_their_bounds_in_the_unit_read() -> None:
    assert str(value("op", "out") >= 2 * V) == "out in op ≥ 2 V"
    assert str(gain("ac", "out", at=kHz) <= 3) == "gain of out at 1 kHz in ac ≤ 3 dB"
    reflected = -3 <= gain("ac", "out", at=kHz)  # noqa: SIM300 - reflection is tested
    assert str(reflected) == "gain of out at 1 kHz in ac ≥ -3 dB"
    assert (
        str(maximum("t", "V1:current", start=1 * us).within(1 * mA, 0.1 * mA))
        == "maximum of V1:current from 1 µs in t ≥ 900 µA and ≤ 1.1 mA"
    )
    assert (
        str((rise_time("t", "out") <= 2 * us).with_margin(0.1 * us))
        == "rise time of out in t ≤ 2 µs with a margin of 100 ns"
    )
    assert (
        str(value("t", "out", at=1 * us, reference="ref").between(0 * V, 1 * V))
        == "out relative to ref at 1 µs in t ≥ 0 V and ≤ 1 V"
    )


def test_requirements_record_the_line_that_states_them() -> None:
    requirement = value("op", "out") >= 1 * V
    assert requirement.source_location is not None
    assert requirement.source_location.file.endswith("test_specs.py")
    assert requirement.source_location.function == (
        "test_requirements_record_the_line_that_states_them"
    )


def test_a_verdict_states_the_value_and_the_slack() -> None:
    passed = verdict_of(value("op", "out").between(1 * V, 3 * V), point(2.5))
    assert (passed.status, passed.value, passed.unit) == ("passed", 2.5, "V")
    # Exact subtraction 3.0 - 2.5 = 0.5; default pytest.approx covers float rounding.
    assert passed.slack == pytest.approx(0.5)
    marginal = verdict_of((value("op", "out") >= 2 * V).with_margin(1 * V), point(2.5))
    assert marginal.status == "marginal"
    failed = verdict_of(value("op", "out") <= 2 * V, point(2.5))
    # Exact subtraction 2.0 - 2.5 = -0.5; default pytest.approx covers float rounding.
    assert (failed.status, failed.slack) == ("failed", pytest.approx(-0.5))
    assert failed.describe().startswith(
        "failed: out in op ≤ 2 V: read 2.5 V, slack -500 mV\n  = "
    )


def test_a_requirement_is_unmeasured_when_it_cannot_be_read() -> None:
    assert (
        unread(value("op", "out") >= 1 * A, point(2.0))
        == "the minimum, 1 A, is not in V, the unit of the value read"
    )
    assert unread(gain("op", "out", at=kHz) >= 0, point(2.0)) == (
        "a gain or a phase reads a small-signal response or a loop gain, not an "
        "operating point"
    )
    assert unread(value("op", "outt") >= 0 * V, point(2.0)) == (
        "the circuit has no unknown named 'outt'; close names: out"
    )
    assert unread(value("op", "out", at=us) >= 0 * V, point(2.0)) == (
        "an operating point holds one value, not one at 1 µs"
    )
    assert unread(value("t", "out") >= 0 * V, first_order()) == (
        "a transient holds a value per time, and at is unset"
    )
    assert unread(value("t", "out", at=1 * s) >= 0 * V, first_order()) == (
        "at, 1 s, lies outside the times, 0 s to 20 µs"
    )
    assert unread(value("op", "V1:current", reference="out") >= 0 * A, point(2.0)) == (
        "V1:current relative to out names the current V1:current, not a voltage"
    )
    assert unread(rise_time("t", "out") <= 1 * s, point(1.0)) == (
        "a rise time reads a transient, not an operating point"
    )
    parameter = results.Parameter("R1", "resistance", 1e3)
    sensitivities = results.SensitivityResult(
        UNKNOWNS, ("out",), (parameter,), np.ones((1, 1)), point(1.0)
    )
    assert unread(value("sens", "out") >= 0 * V, sensitivities) == (
        "a value reads an operating point, a transient or a DC sweep, not sensitivities"
    )
    flat = waveforms(np.linspace(0, 1, 11), np.ones(11))
    assert (
        unread(overshoot("t", "out") <= 1, flat) == "out ends where it starts, at 1 V"
    )
    assert unread(frequency("t", "out") >= 1 * Hz, flat) == (
        "out rises through 1 V fewer than twice"
    )
    assert unread(phase_margin("ac", "out") >= 45, low_pass()) == (
        "the gain of out never falls through 0 dB"
    )


def test_unplanned_and_failed_analyses_leave_requirements_unmeasured() -> None:
    specs = Specs([value("op", "out") >= 0 * V, value("dc", "out") >= 0 * V])
    planned, failed = evaluate(specs, ["op"], {})
    assert (planned.reason, failed.reason) == (
        "the analysis op failed",
        "the plan runs no analysis named dc",
    )
    assert describe((planned, failed)).endswith("specs: 2 unmeasured")


def test_impossible_requirements_and_specs_are_errors() -> None:
    with pytest.raises(ValueError, match="exceeds its maximum"):
        value("op", "out").between(2 * V, 1 * V)
    with pytest.raises(ValueError, match="margin of out in op is negative"):
        (value("op", "out") >= 1 * V).with_margin(-1 * V)
    with pytest.raises(ValueError, match="states a bound"):
        Requirement(value("op", "out"))
    with pytest.raises(ValueError, match="must be finite"):
        _ = value("op", "out") >= math.inf
    with pytest.raises(ValueError, match="at least one requirement"):
        Specs([])
    with pytest.raises(ValidationError, match="hold requirements"):
        Specs([value("op", "out")])  # ty: ignore[invalid-argument-type] - the wrong type is tested
    with pytest.raises(ValueError, match="between 0 and 1"):
        rise_time("t", "out", low=0.9, high=0.1)


SCRIPT = """
from jolty import Builder, OperatingPoint, Plan, Role, Specs, component, devices
from jolty.specs import value
from jolty.units import V, kohm


@component
def divider(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(5 * V), name="V1")
    upper = c.place(devices.resistor(1 * kohm), name="R1")
    lower = c.place(devices.resistor(1 * kohm), name="R2")
    c.connect(source.positive, upper.a)
    c.connect(upper.b, lower.a, name="out")
    c.connect(lower.b, source.negative, ground)


def build():
    return divider()


def plan():
    return Plan({"op": OperatingPoint()})


def specs():
    return SPECS
"""


def run_specs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, specs: str
) -> Built | Failure:
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "divider.py"
    script.write_text(SCRIPT.replace("SPECS", specs), encoding="utf-8")
    return run(Job(script=script, plan=script, max_unknowns=10), tmp_path)


def test_the_runner_evaluates_the_plans_specs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome = run_specs(
        tmp_path, monkeypatch, 'Specs([value("op", "out").within(2.5 * V, 1e-6 * V)])'
    )
    assert isinstance(outcome, Built)
    (verdict,) = outcome.specs
    # Exact 1:1 divider output 2.5 V via runner JSON; default pytest.approx rounding.
    assert (verdict.status, verdict.value) == ("passed", pytest.approx(2.5))
    assert verdict.source_location is not None
    assert (verdict.source_location.file, verdict.source_location.function) == (
        "divider.py",
        "specs",
    )


def test_specs_must_be_specs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outcome = run_specs(tmp_path, monkeypatch, '[value("op", "out") >= 1 * V]')
    assert isinstance(outcome, Failure)
    assert outcome.message == "specs() returned list, not Specs"


CORNERED = """
from jolty import Ac, Builder, Corner, OperatingPoint, Plan, Role, Specs
from jolty import component, devices
from jolty.plan import decade
from jolty.specs import bandwidth
from jolty.units import K, MHz, V, celsius, kHz, kohm, nF


@component
def low_pass(c: Builder, resistance) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(0 * V), name="V1")
    resistor = c.place(
        devices.resistor(resistance, temperature_coefficient=4e-3 / K), name="R1"
    )
    capacitor = c.place(devices.capacitor(1 * nF), name="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)


def build(resistance=1 * kohm):
    return low_pass(resistance)


def plan():
    return Plan(
        {"bode": Ac("V1", decade(1 * kHz, 10 * MHz, 50))},
        corners={
            "hot": Corner(temperature=celsius(125)),
            "slow": Corner(resistance=1.2 * kohm),
            "broken": Corner(resistance=-1 * kohm),
        },
    )


def specs():
    return Specs([bandwidth("bode", "out") >= 120 * kHz])
"""


def test_requirements_hold_at_their_worst_corner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "filter.py"
    script.write_text(CORNERED, encoding="utf-8")
    outcome = run(Job(script=script, plan=script, max_unknowns=10), tmp_path)
    assert isinstance(outcome, Built), str(outcome)
    hot, slow, broken = outcome.corners
    # 125 C in kelvin (125 + 273.15 = 398.15 K); default pytest.approx rounding.
    assert (hot.name, hot.temperature) == ("hot", pytest.approx(398.15))
    assert hot.ok
    assert slow.ok
    assert not broken.ok
    assert "resistance must be positive" in broken.problems[0]
    # At 125 °C, the resistance has grown by 39 %, and the corner falls to 114 kHz.
    (verdict,) = outcome.specs
    assert (verdict.status, verdict.corner) == ("failed", "hot")
    assert verdict.describe().startswith(
        "failed: 3 dB bandwidth of out in bode ≥ 120 kHz: read 114.1 kHz at the "
        "corner hot, slack"
    )


def test_a_verdict_is_the_worst_of_every_run() -> None:
    requirement = value("op", "out") >= 1 * V
    runs = {"low": {"op": point(0.5)}, "high": {"op": point(3.0)}, "lost": {}}
    verdict, unmeasured = evaluate(
        Specs([requirement, value("op", "out") <= 4 * V]),
        ["op"],
        {"op": point(2.0)},
        runs,
    )
    assert (verdict.status, verdict.corner, verdict.value) == ("failed", "low", 0.5)
    assert (unmeasured.status, unmeasured.corner) == ("unmeasured", "lost")


def hiss(*, referred: bool = True) -> results.NoiseResult:
    """White noise of 16e-18 V²/Hz from 10 Hz to 100 kHz, plus flicker noise.

    The flicker noise is 1e-15 V²/Hz at 1 Hz, falling as 1/f, and the input's gain 2.
    """
    frequencies = np.logspace(1, 5, 41)
    white = np.full(frequencies.size, 16e-18)
    flicker = 1e-15 / frequencies
    noise_input = results.NoiseInput("V1", "voltage", np.full(frequencies.size, 2 + 0j))
    return results.NoiseResult(
        UNKNOWNS,
        "out",
        None,
        "voltage",
        frequencies,
        white + flicker,
        (results.NoiseSource("R1", "thermal"), results.NoiseSource("D1", "flicker")),
        np.stack([white, flicker]),
        noise_input if referred else None,
        point(0.0),
    )


def test_noise_reads_densities_and_rms_noise() -> None:
    spectrum = hiss()
    density = noise_density("hiss", at=1 * kHz).read(spectrum)
    assert density.unit == VOLTAGE / rtHz.dimension
    # Exact 1 kHz grid point (~4.12e-9 V/rtHz); abs=0 prevents default abs=1e-12
    # from masking nanovolt-scale errors; rel=1e-12 covers sqrt rounding.
    assert density.value == pytest.approx(math.sqrt(16e-18 + 1e-18), rel=1e-12, abs=0)
    between = noise_density("hiss", at=1.5 * kHz).read(spectrum)
    # Piecewise power-law interpolation of white + 1/f at 1.5 kHz has 9.62e-5 relative
    # error (~3.93e-13 V/rtHz on 4.08e-9 V/rtHz); abs=0 prevents default abs=1e-12 mask.
    assert between.value == pytest.approx(
        math.sqrt(16e-18 + 1e-15 / 1.5e3), rel=1e-4, abs=0
    )
    referred = noise_density("hiss", at=1 * kHz, referred=True).read(spectrum)
    # Input gain is constant 2; division by 2 matches within rounding (rel=1e-12).
    assert referred.value == pytest.approx(density.value / 2, rel=1e-12, abs=0)
    # The sum of two power laws is not a power law: piecewise power-law quadrature on
    # 10 pts/decade has 9.94e-6 relative error across the band (bounded by rel=1e-4).
    power = 16e-18 * (1e5 - 10) + 1e-15 * math.log(1e4)
    total = integrated_noise("hiss").read(spectrum)
    assert total.unit == VOLTAGE
    assert total.value == pytest.approx(math.sqrt(power), rel=1e-4, abs=0)
    band = integrated_noise("hiss", start=100 * Hz, stop=10 * kHz).read(spectrum)
    expected = 16e-18 * (1e4 - 100) + 1e-15 * math.log(100)
    # Same piecewise power-law quadrature error (5.60e-5 relative on 10 pts/decade).
    assert band.value == pytest.approx(math.sqrt(expected), rel=1e-4, abs=0)
    assert band.value == spectrum.rms(100.0, 1e4)
    assert str(noise_density("hiss", at=1 * kHz) <= 10 * nV / rtHz) == (
        "noise density at 1 kHz in hiss ≤ 10 nV/√Hz"
    )
    assert str(integrated_noise("hiss", referred=True) <= 1 * uV) == (
        "RMS noise referred to the input from the lowest frequency to the highest "
        "in hiss ≤ 1 µV"
    )
    assert verdict_of(
        noise_density("hiss", at=1 * kHz) <= 5 * nV / rtHz, spectrum
    ).status == ("passed")


def test_noise_reads_a_noise_analysis_within_its_band() -> None:
    spectrum = hiss(referred=False)
    assert unread(noise_density("hiss", at=1 * kHz) <= 1 * V / rtHz, low_pass()) == (
        "a noise density reads a noise analysis, not a small-signal response"
    )
    assert unread(integrated_noise("hiss", referred=True) <= 1 * V, spectrum) == (
        "the noise analysis names no source to refer its noise to"
    )
    assert unread(noise_density("hiss", at=1 * Hz) <= 1 * V / rtHz, spectrum) == (
        "at, 1 Hz, lies outside the frequencys, 10 Hz to 100 kHz"
    )
    assert unread(
        integrated_noise("hiss", start=1 * kHz, stop=100 * Hz) <= 1 * V, spectrum
    ) == ("the band from 1 kHz to 100 Hz is empty")
    assert unread(value("hiss", "out") <= 1 * V, spectrum) == (
        "a value reads an operating point, a transient or a DC sweep, not a noise "
        "spectrum"
    )


def test_a_yield_of_nothing_or_everything_has_an_exact_end() -> None:
    """No sample passing puts the Wilson interval's lower end at 0, not at rounding.

    Its ends otherwise agree with the textbook formula to within rounding.
    """
    requirement = value("op", "out") <= 2 * V
    failed, passed = (
        verdict_of(requirement, point(2.5)),
        verdict_of(requirement, point(1)),
    )
    specs = Specs([requirement])
    _, none = spread(specs, [(failed,)] * 2000)
    _, every = spread(specs, [(passed,)] * 2000)
    assert none.lower == 0.0
    assert every.upper == 1.0
    _, some = spread(specs, [(passed,)] * 37 + [(failed,)] * 163)
    z, n, p = 1.959963984540054, 200, 37 / 200
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    assert (some.lower, some.upper) == (
        pytest.approx(centre - half, rel=1e-14),
        pytest.approx(centre + half, rel=1e-14),
    )


def test_the_wilson_interval_is_the_yield_s_and_refuses_what_is_no_proportion() -> None:
    """`wilson` is the interval a yield states, of any proportion of samples.

    Its ends agree with the textbook formula to within rounding, as the yield's do.
    """
    requirement = value("op", "out") <= 2 * V
    failed, passed = (
        verdict_of(requirement, point(2.5)),
        verdict_of(requirement, point(1)),
    )
    _, some = spread(Specs([requirement]), [(passed,)] * 37 + [(failed,)] * 163)
    assert wilson(37, 200) == (some.lower, some.upper)
    z, n, p = 1.959963984540054, 7, 3 / 7
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    assert wilson(3, 7) == (
        pytest.approx(centre - half, rel=1e-14),
        pytest.approx(centre + half, rel=1e-14),
    )
    assert wilson(0, 1)[0] == 0.0
    assert wilson(1, 1)[1] == 1.0
    with pytest.raises(ValueError, match="of one sample or more, not 0"):
        wilson(0, 0)
    for passing in (-1, 8):
        with pytest.raises(
            ValueError, match=f"{passing} of 7 samples is no proportion"
        ):
            wilson(passing, 7)


def test_poles_read_their_least_damping_and_the_dominant_one() -> None:
    poles = np.array([-3e3 + 4e3j, -3e3 - 4e3j, -1e6 + 0j])
    found = results.PoleZeroResult(UNKNOWNS, poles, None, None, None, None, point(0.0))
    least = damping("pz").read(found)
    # Damping -Re(-3e3 + 4e3j)/|-3e3 + 4e3j| = 0.6 and |s|/(2*pi); float rounding.
    assert (least.value, least.unit) == (pytest.approx(0.6), "")
    slowest = dominant_pole("pz").read(found)
    assert slowest.value == pytest.approx(5e3 / (2 * math.pi))
    assert slowest.unit == FREQUENCY
    assert str(damping("pz") >= 0.5) == "least damping ratio of the poles in pz ≥ 0.5"
    assert (
        verdict_of(damping("pz") >= 0.7, found)
        .describe()
        .startswith("failed: least damping ratio of the poles in pz ≥ 0.7: read 0.6")
    )
    growing = results.PoleZeroResult(
        UNKNOWNS, np.array([1e3 + 0j]), None, None, None, None, point(0.0)
    )
    assert damping("pz").read(growing).value == -1.0
    empty = results.PoleZeroResult(
        UNKNOWNS, np.empty(0, dtype=complex), None, None, None, None, point(0.0)
    )
    assert unread(damping("pz") >= 0, empty) == "the circuit has no poles"
    assert unread(dominant_pole("pz") >= 1 * Hz, low_pass()) == (
        "a dominant pole reads a pole-zero analysis, not a small-signal response"
    )


def test_solutions_count_a_searchs_dc_solutions() -> None:
    found = results.OperatingPointsResult(
        UNKNOWNS, (point(0.0), point(3.3), point(1.65))
    )
    reading = solutions("states").read(found)
    assert (reading.value, reading.unit) == (3.0, "")
    assert str(solutions("states") <= 1) == "DC solutions in states ≤ 1"
    assert unread(solutions("states") <= 1, point(0.0)) == (
        "solutions reads a search for operating points, not an operating point"
    )
    assert unread(value("states", "out") >= 0 * V, found) == (
        "a value reads an operating point, a transient or a DC sweep, not DC solutions"
    )


type Readings = list[list[list[float | None]]]
"""What each requirement read on each run, by condition then by requirement."""


@st.composite
def judged_runs(draw: st.DrawFn) -> tuple[Specs, list[bool], Readings]:
    """Specs, and what each of their requirements read on each run at each condition.

    Each reading is a number, possibly not finite, or None where the analysis failed;
    a condition whose circuit does not elaborate, as `elaborates` says, reads none. A
    bound in amperes is not in the reading's unit, which leaves it unmeasured.
    """
    numbers = st.floats(-3.0, 3.0, allow_nan=False)
    requirements = []
    for _ in range(draw(st.integers(1, 3))):
        low, high = sorted((draw(numbers), draw(numbers)))
        bounds = draw(st.sampled_from(["minimum", "maximum", "both"]))
        unit = draw(st.sampled_from([V, V, A]))
        measured = value("op", "out")
        requirement = {
            "minimum": measured >= low * unit,
            "maximum": measured <= high * unit,
            "both": measured.between(low * unit, high * unit),
        }[bounds]
        if draw(st.booleans()):
            requirement = requirement.with_margin(draw(st.floats(0, 1)) * unit)
        requirements.append(requirement)
    runs = draw(st.integers(1, 12))
    conditions = draw(st.integers(1, 3))
    elaborates = [
        draw(st.booleans()) or condition == 0 for condition in range(conditions)
    ]
    reading = st.one_of(
        st.none(),
        st.sampled_from([math.nan, math.inf, -math.inf]),
        numbers,
        st.sampled_from([-1.0, 0.0, 1.0, -0.0]),
    )
    readings = [
        [[draw(reading) for _ in range(runs)] for _ in requirements]
        for _ in range(conditions)
    ]
    return Specs(requirements), elaborates, readings


def _batch(readings: Sequence[Sequence[float | None]]) -> results.BatchResult:
    """A batch of a measure per requirement that read `readings`, by requirement."""
    measured = np.array(
        [[math.nan if read is None else read for read in each] for each in readings]
    ).T
    causes = np.array(
        [[0 if read is None else -1 for read in each] for each in readings],
        dtype=np.int32,
    ).T
    width = len(readings)
    ended = np.array([[True] for _ in range(measured.shape[0])])
    return results.BatchResult(
        ("op",),
        (),
        np.empty((measured.shape[0], 0)),
        ended,
        ("out in op",) * width,
        (VOLTAGE,) * width,
        measured,
        causes,
        ("the analysis op failed",),
    )


@given(judged_runs())
@settings(max_examples=300, deadline=None)
def test_verdicts_on_a_batch_are_those_on_each_run(
    drawn: tuple[Specs, list[bool], Readings],
) -> None:
    """The spreads and the yield of a batch's columns are those of its runs' verdicts.

    Each run's verdict on a requirement is its worst over the conditions, as `evaluate`
    judges a Monte Carlo sample, and the columns' are the same numbers, bit for bit.
    """
    specs, elaborates, readings = drawn
    runs = len(readings[0][0])
    batches = [
        _batch(readings[condition]) if elaborates[condition] else None
        for condition in range(len(readings))
    ]
    judged = [
        worst(
            [
                judged_requirement(requirement, ["op"], batch, place, runs)
                for batch in batches
            ]
        )
        for place, requirement in enumerate(specs.requirements)
    ]
    sampled = []
    for index in range(runs):
        verdicts = []
        for place, requirement in enumerate(specs.requirements):
            runs_of = []
            for condition, elaborated in enumerate(elaborates):
                read = readings[condition][place][index] if elaborated else None
                found = {} if read is None else {"op": point(read)}
                runs_of.append((None if condition == 0 else f"c{condition}", found))
            verdicts.append(judge(requirement, ["op"], runs_of))
        sampled.append(tuple(verdicts))
    assert spread_of(specs, judged) == spread(specs, sampled)


def test_temperature_dc_sweep_uses_temperature_axis_dimension() -> None:
    sweep = results.DcSweepResult(
        UNKNOWNS,
        "T1",
        "temperature",
        np.array([273.15, 300.0, 350.0]),
        np.array([[1.0, 0.5, -1e-3], [2.0, 0.5, -2e-3], [3.0, 0.5, -3e-3]]),
    )
    reading = value("sweep", "out", at=300 * K).read(sweep)
    assert reading.value == 2.0
    assert reading.dimension == VOLTAGE
    assert unread(value("sweep", "out", at=1 * A) >= 0 * V, sweep) == (
        "at takes a quantity in K, not 1 A"
    )
    assert K.dimension == TEMPERATURE
