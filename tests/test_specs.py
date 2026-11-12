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
    crossover,
    describe,
    distortion,
    evaluate,
    frequency,
    gain,
    gain_margin,
    integrated_noise,
    maximum,
    minimum,
    noise_density,
    overshoot,
    peak_to_peak,
    phase,
    phase_margin,
    rise_time,
    rms,
    settling_time,
    value,
)
from jolty.units import (
    CURRENT,
    FREQUENCY,
    TIME,
    VOLTAGE,
    A,
    Hz,
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


def loop_gain(frequencies: results.Real, gains: results.Complex) -> results.LoopGain:
    return results.LoopGain(UNKNOWNS, "VP", frequencies, gains, point(0.0))


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
    for measurement, (value_read, unit) in expected.items():
        measured = measurement.read(loop)
        assert measured.unit == unit, measurement
        assert measured.value == pytest.approx(value_read, rel=1e-6), measurement
    assert str(phase_margin("loop") >= 45) == "phase margin of the loop in loop ≥ 45 °"
    assert str(gain_margin("loop") >= 6) == "gain margin of the loop in loop ≥ 6 dB"


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
    sensitivities = results.Sensitivities(
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


def hiss(*, referred: bool = True) -> results.NoiseSpectrum:
    """White noise of 16e-18 V²/Hz from 10 Hz to 100 kHz, plus flicker noise.

    The flicker noise is 1e-15 V²/Hz at 1 Hz, falling as 1/f, and the input's gain 2.
    """
    frequencies = np.logspace(1, 5, 41)
    white = np.full(frequencies.size, 16e-18)
    flicker = 1e-15 / frequencies
    noise_input = results.NoiseInput("V1", "voltage", np.full(frequencies.size, 2 + 0j))
    return results.NoiseSpectrum(
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
    assert density.value == pytest.approx(math.sqrt(16e-18 + 1e-18), rel=1e-12)
    between = noise_density("hiss", at=1.5 * kHz).read(spectrum)
    assert between.value == pytest.approx(math.sqrt(16e-18 + 1e-15 / 1.5e3), rel=1e-12)
    referred = noise_density("hiss", at=1 * kHz, referred=True).read(spectrum)
    assert referred.value == pytest.approx(density.value / 2, rel=1e-12)
    # The sum of two power laws is none, so the integral is close, not exact.
    power = 16e-18 * (1e5 - 10) + 1e-15 * math.log(1e4)
    total = integrated_noise("hiss").read(spectrum)
    assert total.unit == VOLTAGE
    assert total.value == pytest.approx(math.sqrt(power), rel=1e-4)
    band = integrated_noise("hiss", start=100 * Hz, stop=10 * kHz).read(spectrum)
    expected = 16e-18 * (1e4 - 100) + 1e-15 * math.log(100)
    assert band.value == pytest.approx(math.sqrt(expected), rel=1e-4)
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
