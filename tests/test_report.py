# Copyright 2026 Maxence Faldor

import math

import numpy as np
import pytest

from jolty import report, results

UNKNOWNS = (
    results.Unknown("in", "voltage"),
    results.Unknown("out", "voltage"),
    results.Unknown("V1:current", "current"),
)


def test_an_operating_point_lists_each_value() -> None:
    point = results.OperatingPoint(UNKNOWNS, np.array([5.0, 2.5, -2.5e-3]), "direct")
    summary = report.report("op", point, listed=10)
    assert summary == report.PointReport(
        name="op",
        strategy="direct",
        values=(
            report.Value(unknown="in", quantity="voltage", value=5.0),
            report.Value(unknown="out", quantity="voltage", value=2.5),
            report.Value(unknown="V1:current", quantity="current", value=-2.5e-3),
        ),
        unlisted=0,
    )
    assert summary.describe() == (
        "op: operating point, found by Newton iteration\n"
        "  in          5 V\n"
        "  out         2.5 V\n"
        "  V1:current  -2.5 mA"
    )


def test_a_transient_states_the_range_of_each_unknown() -> None:
    values = np.array([[0.0, 0.0, 0.0], [1.0, 0.4, -1e-3], [1.0, 0.9, -1e-4]])
    statistics = results.Statistics(
        initial="direct", accepted=2, rejected=1, iterations=9
    )
    waveforms = results.Waveforms(
        UNKNOWNS, np.array([0.0, 1e-6, 2e-6]), values, statistics
    )
    summary = report.report("step", waveforms, listed=2)
    assert isinstance(summary, report.TransientReport)
    assert summary.extents[1] == report.Extent(
        unknown="out", quantity="voltage", first=0, minimum=0, maximum=0.9, last=0.9
    )
    assert summary.describe() == (
        "step: transient to 2 µs in 2 steps, 1 rejected\n"
        "       first  min  max     last\n"
        "  in   0 V    0 V  1 V     1 V\n"
        "  out  0 V    0 V  900 mV  900 mV\n"
        "  ... and 1 more in the results file"
    )


def test_a_response_states_magnitudes() -> None:
    point = results.OperatingPoint(UNKNOWNS, np.zeros(3), "direct")
    values = np.array([[1.0, 1.0, 0.0], [1.0, -1j, 1e-3j]])
    response = results.Response(UNKNOWNS, "V1", np.array([1e3, 1e6]), values, point)
    summary = report.report("bode", response, listed=10)
    assert isinstance(summary, report.ResponseReport)
    assert (summary.lowest, summary.highest, summary.frequencies) == (1e3, 1e6, 2)
    assert [extent.maximum for extent in summary.magnitudes] == [1.0, 1.0, 1e-3]
    assert summary.describe().startswith(
        "bode: response to V1 at 2 frequencies from 1 kHz to 1 MHz"
    )


def test_a_loop_gain_states_its_margins() -> None:
    point = results.OperatingPoint(UNKNOWNS, np.array([5.0, 4.0, -1e-3]), "direct")
    frequencies = np.logspace(-2, 2, 4001)
    gains = np.asarray(5 / (1 + 1j * frequencies) ** 3, dtype=np.complex128)
    loop = results.LoopGain(UNKNOWNS, "VP", frequencies, gains, point)
    summary = report.report("loop", loop, listed=10)
    assert isinstance(summary, report.StabilityReport)
    assert summary.crossover == pytest.approx(math.sqrt(5 ** (2 / 3) - 1), rel=1e-4)
    assert summary.describe() == (
        "loop: loop gain through VP at 4001 frequencies from 10 mHz to 100 Hz\n"
        "  13.98 dB and -1.719° at 10 mHz\n"
        "  crossover at 1.387 Hz, phase margin 17.37°\n"
        "  gain margin 4.082 dB"
    )
    stable = results.LoopGain(UNKNOWNS, "VP", frequencies[:2], gains[:2] / 100, point)
    assert (
        report.report("loop", stable, listed=10)
        .describe()
        .endswith(
            "  it never falls through 0 dB\n  its phase never falls through -180°"
        )
    )


def test_sensitivities_list_each_derivative_in_the_circuit_s_order() -> None:
    point = results.OperatingPoint(UNKNOWNS, np.array([5.0, 4.0, -1e-3]), "direct")
    parameters = tuple(
        results.Parameter(f"R{index}", "resistance", 1e3) for index in range(3)
    )
    derivatives = np.array([[1e-6, -2e-6, 0.0], [0.0, 0.0, 1e-9]])
    sensitivities = results.Sensitivities(
        UNKNOWNS, ("out", "V1:current"), parameters, derivatives, point
    )
    summary = report.report("sens", sensitivities, listed=2)
    assert isinstance(summary, report.SensitivityReport)
    out, current = summary.outputs
    assert out.derivatives[1] == report.Derivative(
        instance="R1", parameter="resistance", value=1e3, derivative=-2e-6
    )
    assert [each.derivative for each in current.derivatives] == [0.0, 0.0]
    assert summary.unlisted == 1
    assert summary.describe() == (
        "sens: derivatives with respect to 3 parameters at the operating point, found "
        "by Newton iteration\n"
        "  out = 4 V\n"
        "    R0  resistance  at 1000.0  1e-06\n"
        "    R1  resistance  at 1000.0  -2e-06\n"
        "      ... and 1 more in the results file\n"
        "  V1:current = -1 mA\n"
        "    R0  resistance  at 1000.0  0.0\n"
        "    R1  resistance  at 1000.0  0.0\n"
        "      ... and 1 more in the results file"
    )


def test_sensitivities_along_an_analysis_state_where_each_is_largest() -> None:
    point = results.OperatingPoint(UNKNOWNS, np.array([5.0, 4.0, -1e-3]), "direct")
    frequencies = np.array([1e3, 1e6])
    values = np.ones((2, 3), dtype=np.complex128)
    response = results.Response(UNKNOWNS, "V1", frequencies, values, point)
    parameter = results.Parameter("C1", "capacitance", 1e-9)
    derivatives = np.array([[[1j]], [[-3 + 4j]]])
    sensitivities = results.Sensitivities(
        UNKNOWNS, ("out",), (parameter,), derivatives, response
    )
    summary = report.report("sens", sensitivities, listed=5)
    assert isinstance(summary, report.SensitivityReport)
    assert (summary.of, summary.points, summary.unit) == ("ac", 2, "Hz")
    assert summary.describe() == (
        "sens: derivatives with respect to 1 parameters over a response at 2 "
        "frequencies\n"
        "  out\n"
        "    C1  capacitance  at 1e-09  largest 5.0 at 1 MHz"
    )
    times = np.array([0.0, 1e-6, 2e-6])
    statistics = results.Statistics("direct", 2, 0, 4)
    waveforms = results.Waveforms(UNKNOWNS, times, np.zeros((3, 3)), statistics)
    falling = results.Sensitivities(
        UNKNOWNS,
        ("out",),
        (parameter,),
        np.array([[[0.5]], [[-2.0]], [[1.0]]]),
        waveforms,
    )
    summary = report.report("t", falling, listed=5)
    assert isinstance(summary, report.SensitivityReport)
    (derivative,) = summary.outputs[0].derivatives
    assert (derivative.derivative, derivative.at) == (-2.0, 1e-6)


def test_a_failure_says_why() -> None:
    failure = report.FailureReport(
        name="step", error="the time step fell below 1e-18 s"
    )
    assert failure.describe() == "step: failed: the time step fell below 1e-18 s"
