# Copyright 2026 Maxence Faldor

import math
from pathlib import Path

import numpy as np
import pytest

from jolty import results
from jolty.runner import Built, Failure, Job, run
from jolty.specs import (
    MeasurementError,
    Requirement,
    Specs,
    Verdict,
    average,
    bandwidth,
    crossing,
    describe,
    evaluate,
    frequency,
    gain,
    maximum,
    minimum,
    overshoot,
    peak_to_peak,
    phase,
    phase_margin,
    rise_time,
    rms,
    settling_time,
    value,
)
from jolty.units import CURRENT, FREQUENCY, TIME, VOLTAGE, A, Hz, V, kHz, mA, ms, s, us

UNKNOWNS = (
    results.Unknown("out", "voltage"),
    results.Unknown("ref", "voltage"),
    results.Unknown("V1:current", "current"),
)

TAU = 1e-6
"""The time constant of the first-order step response, in seconds."""


def point(out: float) -> results.OperatingPoint:
    return results.OperatingPoint(UNKNOWNS, np.array([out, 0.5, -out / 1e3]), "newton")


def waveforms(times: results.Real, out: results.Real) -> results.Waveforms:
    values = np.column_stack([out, np.full_like(out, 0.5), -out / 1e3])
    statistics = results.Statistics("operating point", times.size, 0, 0)
    return results.Waveforms(UNKNOWNS, times, values, statistics)


def response(frequencies: results.Real, out: results.Complex) -> results.Response:
    values = np.column_stack([out, np.full_like(out, 2), out / 1e3])
    return results.Response(UNKNOWNS, "V1", frequencies, values, point(0.0))


def first_order() -> results.Waveforms:
    """A unit step through a time constant of 1 µs, held for 20 time constants."""
    times = np.linspace(0, 20 * TAU, 200_001)
    return waveforms(times, 1 - np.exp(-times / TAU))


def low_pass() -> results.Response:
    """A first-order low pass with its corner at 1 kHz, densely sampled."""
    frequencies = np.logspace(0, 6, 6001)
    transfer = np.asarray(1 / (1 + 1j * frequencies / 1e3), dtype=np.complex128)
    return response(frequencies, transfer)


def verdict_of(requirement: Requirement, result: results.Result) -> Verdict:
    analysis = requirement.measurement.analysis
    (verdict,) = evaluate(Specs([requirement]), [analysis], {analysis: result})
    return verdict


def unread(requirement: Requirement, result: results.Result) -> str:
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
    assert step.value == pytest.approx(1 - math.exp(-1))
    assert step.dimension == VOLTAGE


def test_a_value_reads_a_sweep_whose_levels_fall() -> None:
    levels = np.linspace(5, 0, 11)
    values = np.column_stack([levels / 2, levels, levels])
    sweep = results.Sweep(UNKNOWNS, "V1", "voltage", levels, values)
    assert value("dc", "out", at=1.25 * V).read(sweep).value == pytest.approx(0.625)
    assert maximum("dc", "out", stop=4 * V).read(sweep).value == pytest.approx(2)


def test_windowed_statistics_match_a_sine_over_whole_periods() -> None:
    times = np.linspace(0, 3 * ms.value, 30_001)
    sine = waveforms(times, 2 * np.sin(2 * np.pi * 1e3 * times))
    assert maximum("t", "out").read(sine).value == pytest.approx(2, rel=1e-6)
    assert minimum("t", "out").read(sine).value == pytest.approx(-2, rel=1e-6)
    assert peak_to_peak("t", "out").read(sine).value == pytest.approx(4, rel=1e-6)
    assert average("t", "out").read(sine).value == pytest.approx(0, abs=1e-9)
    assert rms("t", "out").read(sine).value == pytest.approx(math.sqrt(2), rel=1e-6)
    measured = frequency("t", "out").read(sine)
    assert (measured.value, measured.dimension) == (pytest.approx(1e3), FREQUENCY)
    half = average("t", "out", start=0 * s, stop=0.5 * ms).read(sine).value
    assert half == pytest.approx(4 / math.pi, rel=1e-6)


def test_edges_match_a_first_order_step() -> None:
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
    assert rise_time("t", "out").read(falling).value == pytest.approx(TAU * math.log(9))
    assert not crossing("t", "out", 0.5 * V, edge="rising").read(step).value < 0
    with pytest.raises(MeasurementError, match="does not cross 500 mV"):
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
    assert (measured.value, measured.unit) == (pytest.approx(expected, rel=1e-6), "%")


def test_a_first_order_response_falls_3_db_at_its_corner() -> None:
    corner = low_pass()
    assert gain("ac", "out", at=1 * kHz).read(corner).value == pytest.approx(
        -10 * math.log10(2), abs=1e-6
    )
    assert phase("ac", "out", at=1 * kHz).read(corner).value == pytest.approx(-45)
    measured = bandwidth("ac", "out", drop=10 * math.log10(2)).read(corner)
    assert (measured.value, measured.dimension) == (pytest.approx(1e3), FREQUENCY)
    over = gain("ac", "out", over="ref", at=1 * Hz).read(corner).value
    assert over == pytest.approx(20 * math.log10(abs(1 / (1 + 1e-3j)) / 2))


def test_the_phase_margin_of_an_integrator_with_a_pole_at_unity() -> None:
    frequencies = np.logspace(-3, 3, 60_001)
    loop = 1 / (1j * frequencies) / (1 + 1j * frequencies)
    unity = math.sqrt((math.sqrt(5) - 1) / 2)
    expected = 90 - math.degrees(math.atan(unity))
    measured = phase_margin("ac", "out").read(response(frequencies, loop))
    assert (measured.value, measured.unit) == (pytest.approx(expected, abs=1e-6), "°")


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
    assert requirement.source is not None
    assert requirement.source.file.endswith("test_specs.py")
    assert requirement.source.function == (
        "test_requirements_record_the_line_that_states_them"
    )


def test_a_verdict_states_the_value_and_the_slack() -> None:
    passed = verdict_of(value("op", "out").between(1 * V, 3 * V), point(2.5))
    assert (passed.status, passed.value, passed.unit) == ("passed", 2.5, "V")
    assert passed.slack == pytest.approx(0.5)
    marginal = verdict_of((value("op", "out") >= 2 * V).with_margin(1 * V), point(2.5))
    assert marginal.status == "marginal"
    failed = verdict_of(value("op", "out") <= 2 * V, point(2.5))
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
        "a gain or a phase reads a small-signal response, not an operating point"
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
    with pytest.raises(TypeError, match="hold requirements"):
        Specs([value("op", "out")])  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="between 0 and 1"):
        rise_time("t", "out", low=0.9, high=0.1)


SCRIPT = """
from jolty import Builder, OperatingPoint, Plan, Role, Specs, component, devices
from jolty.specs import value
from jolty.units import V, kohm


@component
def divider(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(5 * V), id="V1")
    upper = c.place(devices.resistor(1 * kohm), id="R1")
    lower = c.place(devices.resistor(1 * kohm), id="R2")
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
    return run(Job(script=script, plan=script, listed=10), tmp_path)


def test_the_runner_evaluates_the_plans_specs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome = run_specs(
        tmp_path, monkeypatch, 'Specs([value("op", "out").within(2.5 * V, 1e-6 * V)])'
    )
    assert isinstance(outcome, Built)
    (verdict,) = outcome.specs
    assert (verdict.status, verdict.value) == ("passed", pytest.approx(2.5))
    assert verdict.source is not None
    assert (verdict.source.file, verdict.source.function) == ("divider.py", "specs")


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
    source = c.place(devices.voltage_source(0 * V), id="V1")
    resistor = c.place(
        devices.resistor(resistance, temperature_coefficient=4e-3 / K), id="R1"
    )
    capacitor = c.place(devices.capacitor(1 * nF), id="C1")
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
    outcome = run(Job(script=script, plan=script, listed=10), tmp_path)
    assert isinstance(outcome, Built), str(outcome)
    hot, slow, broken = outcome.corners
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
