# Copyright 2026 Maxence Faldor

import numpy as np

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


def test_sensitivities_list_the_parameters_each_output_moves_most_with() -> None:
    point = results.OperatingPoint(UNKNOWNS, np.array([5.0, 4.0, -1e-3]), "direct")
    parameters = tuple(
        results.Parameter(f"R{index}", "resistance", 1e3) for index in range(7)
    )
    derivatives = np.array(
        [[1e-6, -2e-6, 3e-6, -4e-6, 5e-6, -6e-6, 0], [0, 0, 0, 0, 0, 0, 1e-9]]
    )
    sensitivities = results.Sensitivities(
        UNKNOWNS, ("out", "V1:current"), parameters, derivatives, point
    )
    summary = report.report("sens", sensitivities, listed=10)
    assert isinstance(summary, report.SensitivityReport)
    out, current = summary.outputs
    assert [influence.instance for influence in out.influences] == [
        "R5",
        "R4",
        "R3",
        "R2",
        "R1",
    ]
    assert out.influences[0] == report.Influence(
        instance="R5", parameter="resistance", value=1e3, derivative=-6e-6, change=-6e-5
    )
    assert [influence.instance for influence in current.influences] == ["R6"]
    assert summary.describe() == (
        "sens: sensitivities to 7 parameters at the operating point, found by Newton "
        "iteration\n"
        "  out = 4 V moves, as each parameter grows by 1 %, by\n"
        "    R5  resistance  1000  -60 µV\n"
        "    R4  resistance  1000  +50 µV\n"
        "    R3  resistance  1000  -40 µV\n"
        "    R2  resistance  1000  +30 µV\n"
        "    R1  resistance  1000  -20 µV\n"
        "  V1:current = -1 mA moves, as each parameter grows by 1 %, by\n"
        "    R6  resistance  1000  +10 nA"
    )


def test_a_failure_says_why() -> None:
    failure = report.FailureReport(
        name="step", error="the time step fell below 1e-18 s"
    )
    assert failure.describe() == "step: failed: the time step fell below 1e-18 s"
