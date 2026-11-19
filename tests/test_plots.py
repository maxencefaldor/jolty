# Copyright 2026 Maxence Faldor

import json
from pathlib import Path

import numpy as np
import pytest

from jolty import (
    Ac,
    Builder,
    JoltyError,
    OperatingPoint,
    Role,
    Simulation,
    component,
    devices,
    results,
)
from jolty.cli import main
from jolty.files import Batch, save_results
from jolty.plan import MonteCarlo, decade
from jolty.plots import MAX_SERIES, default_traces, envelope, histograms, plot
from jolty.specs import bandwidth, value
from jolty.units import MHz, V, kHz, kohm, nF
from jolty.unknowns import MAX_SERIES as UNKNOWNS_MAX_SERIES
from jolty.unknowns import default_traces as unknowns_default_traces

UNKNOWNS = (
    results.Unknown("in", "voltage"),
    results.Unknown("out", "voltage"),
    results.Unknown("R1.b", "voltage"),
    results.Unknown("X1/mid", "voltage"),
    results.Unknown("V1:current", "current"),
)
TIMES = np.linspace(0, 1e-6, 5)
STATISTICS = results.Statistics("direct", 4, 0, 0, 8)


def waveforms() -> results.TransientResult:
    values = np.outer(TIMES / TIMES[-1], [1.0, 0.5, 0.25, 0.1, -1e-3])
    return results.TransientResult(UNKNOWNS, TIMES, values, STATISTICS)


def test_plots_draw_the_nets_the_top_names_by_default() -> None:
    assert default_traces is unknowns_default_traces
    assert MAX_SERIES == UNKNOWNS_MAX_SERIES
    assert default_traces(UNKNOWNS) == ("in", "out")
    unnamed = (results.Unknown("R1.b", "voltage"), results.Unknown("X1/a", "voltage"))
    assert default_traces(unnamed) == ("R1.b", "X1/a")


def test_voltages_and_currents_never_share_an_axis() -> None:
    figure = plot("step", waveforms(), ["out", "V1:current"])
    assert [axes.get_ylabel() for axes in figure.axes] == ["voltage", "current"]
    assert figure.axes[-1].get_xlabel() == "time"


def test_a_response_plots_magnitude_and_phase() -> None:
    point = results.OperatingPointResult(UNKNOWNS, np.zeros(5), "direct")
    frequencies = np.array([1e3, 1e4, 1e5])
    values = np.ones((3, 5)) / (1 + 1j * frequencies[:, None] / 1e4)
    response = results.AcResult(UNKNOWNS, "V1", frequencies, values, point)
    figure = plot("bode", response, ["out"])
    assert [axes.get_ylabel() for axes in figure.axes] == [
        "voltage magnitude",
        "voltage phase",
    ]
    assert figure.axes[0].get_xscale() == "log"
    corner = np.asarray(figure.axes[0].get_lines()[0].get_ydata(), dtype=float)[1]
    # Exact -10 * log10(2) ~= -3.0103 dB; rounding to -3.01 dB leaves 3e-4 dB error.
    assert corner == pytest.approx(-3.01, abs=0.01)


def test_a_loop_gain_plots_alone_against_its_margins_levels() -> None:
    point = results.OperatingPointResult(UNKNOWNS, np.zeros(5), "direct")
    frequencies = np.array([0.0, 1e3, 1e4, 1e5])
    gains = 100 / (1 + 1j * frequencies / 1e3) ** 2
    loop = results.StabilityResult(UNKNOWNS, "VP", frequencies, gains, point)
    figure = plot("loop", loop, ["out"])
    magnitude, phase = figure.axes
    assert (magnitude.get_ylabel(), phase.get_ylabel()) == ("magnitude", "phase")
    drawn, *_, level = magnitude.get_lines()
    assert np.asarray(drawn.get_xdata(), dtype=float).tolist() == [1e3, 1e4, 1e5]
    assert np.asarray(level.get_ydata(), dtype=float).tolist() == [0.0, 0.0]
    inverted = phase.get_lines()[-1]
    assert np.asarray(inverted.get_ydata(), dtype=float).tolist() == [-180.0, -180.0]
    assert phase.get_xlabel() == "frequency"


def test_a_noise_spectrum_plots_its_density_and_its_noisiest_sources() -> None:
    point = results.OperatingPointResult(UNKNOWNS, np.zeros(5), "direct")
    frequencies = np.array([10.0, 1e3, 1e5])
    contributions = np.stack([np.full(3, 4e-18), 1e-15 / frequencies])
    spectrum = results.NoiseResult(
        UNKNOWNS,
        "out",
        None,
        "voltage",
        frequencies,
        contributions.sum(axis=0),
        (results.NoiseSource("R1", "thermal"), results.NoiseSource("D1", "flicker")),
        contributions,
        results.NoiseInput("I1", "current", np.full(3, 1e3 + 0j)),
        point,
    )
    figure = plot("hiss", spectrum, ["in"])
    density, referred = figure.axes
    assert (density.get_ylabel(), referred.get_ylabel()) == (
        "noise density",
        "input noise density",
    )
    assert (density.get_xscale(), density.get_yscale()) == ("log", "log")
    total, noisiest, quieter = density.get_lines()[:3]
    assert [total.get_label(), noisiest.get_label(), quieter.get_label()] == [
        "out",
        "R1 thermal",
        "D1 flicker",
    ]
    drawn = np.asarray(referred.get_lines()[0].get_ydata(), dtype=float)
    # Default assert_allclose (rtol=1e-7) covers sqrt/division float rounding.
    np.testing.assert_allclose(drawn, np.sqrt(contributions.sum(axis=0)) / 1e3)
    assert referred.get_xlabel() == "frequency"


def test_poles_and_zeros_plot_at_their_frequency_and_damping() -> None:
    point = results.OperatingPointResult(UNKNOWNS, np.zeros(5), "direct")
    found = results.PoleZeroResult(
        UNKNOWNS,
        np.array([-3e3 + 4e3j, -3e3 - 4e3j, 1e6 + 0j]),
        np.array([0j, -2e3 + 0j]),
        "V1",
        "out",
        None,
        point,
    )
    figure = plot("pz", found, ["in"])
    (axes,) = figure.axes
    assert (axes.get_ylabel(), axes.get_xlabel(), axes.get_xscale()) == (
        "damping ratio",
        "natural frequency",
        "log",
    )
    poles, zeros, boundary = axes.get_lines()
    # Natural frequency |s| / (2*pi) and damping -Re(s)/|s| = 3/5 = 0.6; default
    # tolerances (rtol=1e-7, pytest.approx rel=1e-6) cover float rounding.
    np.testing.assert_allclose(
        np.asarray(poles.get_xdata(), dtype=float),
        [5e3 / (2 * np.pi), 1e6 / (2 * np.pi)],
    )
    assert np.asarray(poles.get_ydata(), dtype=float).tolist() == [
        pytest.approx(0.6),
        -1.0,
    ]
    assert np.asarray(zeros.get_ydata(), dtype=float).tolist() == [1.0]
    assert np.asarray(boundary.get_ydata(), dtype=float).tolist() == [0.0, 0.0]


def test_a_sweep_plots_along_its_source() -> None:
    levels = np.array([0.0, 1.0])
    sweep = results.DcSweepResult(UNKNOWNS, "I1", "current", levels, np.zeros((2, 5)))
    figure = plot("transfer", sweep, ["in"])
    assert figure.axes[0].get_xlabel() == "I1 current"


def test_untyped_dc_sweep_axis_label_omits_none() -> None:
    levels = np.array([0.0, 1.0])
    sweep = results.DcSweepResult(UNKNOWNS, "V1", None, levels, np.zeros((2, 5)))
    figure = plot("transfer", sweep, ["in"])
    assert figure.axes[0].get_xlabel() == "V1"
    traces = results.Traces(
        analysis="transfer",
        along="swept",
        axis=levels,
        unknowns=(UNKNOWNS[0],),
        values=np.zeros((2, 2, 1)),
        source="V1",
        nature=None,
        parameter=None,
    )
    band_figure = envelope("sweep", traces)
    assert band_figure.axes[-1].get_xlabel() == "V1"
    anonymous = results.Traces(
        analysis="transfer",
        along="swept",
        axis=levels,
        unknowns=(UNKNOWNS[0],),
        values=np.zeros((2, 2, 1)),
        source=None,
        nature=None,
        parameter=None,
    )
    anonymous_figure = envelope("sweep", anonymous)
    assert anonymous_figure.axes[-1].get_xlabel() == "swept"


def test_probes_name_unknowns_and_fit_the_palette() -> None:
    with pytest.raises(JoltyError, match="no unknown named 'outt'; close names: out"):
        plot("step", waveforms(), ["outt"])
    with pytest.raises(JoltyError, match=f"at most {MAX_SERIES} unknowns"):
        plot("step", waveforms(), ["in"] * (MAX_SERIES + 1))


def run(*arguments: str) -> int:
    """Runs the command line and returns its exit status."""
    try:
        main(arguments)
    except SystemExit as exit_:
        return exit_.code if isinstance(exit_.code, int) else 0
    return 0


def test_the_plot_command_draws_saved_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "rc.results.npz"
    point = results.OperatingPointResult(UNKNOWNS, np.zeros(5), "direct")
    parameter = results.Parameter("R1", "resistance", 1e3)
    sensitivity = results.SensitivityResult(
        UNKNOWNS, ("out",), (parameter,), np.ones((1, 1)), point
    )
    save_results(path, {"op": point, "step": waveforms(), "sens": sensitivity})
    assert run("plot", str(path), "--output-format", "json") == 0
    (drawn,) = json.loads(capsys.readouterr().out)["plots"]
    target = tmp_path / "rc.step.plot.png"
    assert drawn == {"analysis": "step", "path": str(target)}
    assert target.read_bytes().startswith(b"\x89PNG")
    for unplotted in ("op", "sens"):
        assert run("plot", str(path), "--analyses", unplotted) == 2
        message = f"{unplotted} is taken at DC solutions: nothing to plot"
        assert message in capsys.readouterr().err
    assert run("plot", str(path), "--analyses", "ac") == 2
    assert run("plot", str(tmp_path / "missing.results.npz")) == 2


def test_a_batch_draws_a_histogram_per_measurement_and_an_envelope_per_analysis() -> (
    None
):
    @component
    def filtered(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(1 * V), name="V1")
        resistor = c.place(devices.resistor(1 * kohm, tolerance=0.05), name="R1")
        capacitor = c.place(devices.capacitor(1 * nF, tolerance=0.1), name="C1")
        c.connect(source.positive, resistor.a, name="in")
        c.connect(resistor.b, capacitor.a, name="out")
        c.connect(capacitor.b, source.negative, ground)

    requirements = [
        value("op", "out") >= 0.5 * V,
        bandwidth("bode", "out").between(100 * kHz, 200 * kHz),
    ]
    found = Simulation(filtered()).batch(
        {"op": OperatingPoint(), "bode": Ac("V1", decade(1 * kHz, 10 * MHz, 5))},
        MonteCarlo(30),
        measures=[requirement.measure for requirement in requirements],
        traces=["out", "V1:current"],
    )
    batch = Batch(
        found,
        tuple(str(requirement) for requirement in requirements),
        ((0.5, None), (1e5, 2e5)),
    )
    figure = histograms("monte-carlo", batch)
    assert [axes.get_title(loc="left") for axes in figure.axes] == [
        "out in op ≥ 500 mV",
        "3 dB bandwidth of out in bode ≥ 100 kHz and ≤ 200 kHz",
    ]
    # Each panel counts every run, and draws its requirement's bounds.
    assert [len(axes.lines) for axes in figure.axes] == [1, 2]
    _, bode = found.traces
    figure = envelope("monte-carlo", bode)
    assert [axes.get_ylabel() for axes in figure.axes] == [
        "voltage magnitude",
        "voltage phase",
        "current magnitude",
        "current phase",
    ]
    assert figure.axes[0].get_xscale() == "log"
