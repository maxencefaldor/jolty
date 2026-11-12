# Copyright 2026 Maxence Faldor

import json
from pathlib import Path

import numpy as np
import pytest

from jolty import JoltyError, results
from jolty.cli import main
from jolty.files import save_results
from jolty.plots import MAX_SERIES, default_probes, plot

UNKNOWNS = (
    results.Unknown("in", "voltage"),
    results.Unknown("out", "voltage"),
    results.Unknown("R1.b", "voltage"),
    results.Unknown("X1/mid", "voltage"),
    results.Unknown("V1:current", "current"),
)
TIMES = np.linspace(0, 1e-6, 5)
STATISTICS = results.Statistics("direct", 4, 0, 8)


def waveforms() -> results.Waveforms:
    values = np.outer(TIMES / TIMES[-1], [1.0, 0.5, 0.25, 0.1, -1e-3])
    return results.Waveforms(UNKNOWNS, TIMES, values, STATISTICS)


def test_plots_draw_the_nets_the_top_names_by_default() -> None:
    assert default_probes(UNKNOWNS) == ("in", "out")
    unnamed = (results.Unknown("R1.b", "voltage"), results.Unknown("X1/a", "voltage"))
    assert default_probes(unnamed) == ("R1.b", "X1/a")


def test_voltages_and_currents_never_share_an_axis() -> None:
    figure = plot("step", waveforms(), ["out", "V1:current"])
    assert [axes.get_ylabel() for axes in figure.axes] == ["voltage", "current"]
    assert figure.axes[-1].get_xlabel() == "time"


def test_a_response_plots_magnitude_and_phase() -> None:
    point = results.OperatingPoint(UNKNOWNS, np.zeros(5), "direct")
    frequencies = np.array([1e3, 1e4, 1e5])
    values = np.ones((3, 5)) / (1 + 1j * frequencies[:, None] / 1e4)
    response = results.Response(UNKNOWNS, "V1", frequencies, values, point)
    figure = plot("bode", response, ["out"])
    assert [axes.get_ylabel() for axes in figure.axes] == [
        "voltage magnitude",
        "voltage phase",
    ]
    assert figure.axes[0].get_xscale() == "log"
    corner = np.asarray(figure.axes[0].get_lines()[0].get_ydata(), dtype=float)[1]
    assert corner == pytest.approx(-3.01, abs=0.01)


def test_a_loop_gain_plots_alone_against_its_margins_levels() -> None:
    point = results.OperatingPoint(UNKNOWNS, np.zeros(5), "direct")
    frequencies = np.array([0.0, 1e3, 1e4, 1e5])
    gains = 100 / (1 + 1j * frequencies / 1e3) ** 2
    loop = results.LoopGain(UNKNOWNS, "VP", frequencies, gains, point)
    figure = plot("loop", loop, ["out"])
    magnitude, phase = figure.axes
    assert (magnitude.get_ylabel(), phase.get_ylabel()) == ("magnitude", "phase")
    drawn, *_, level = magnitude.get_lines()
    assert np.asarray(drawn.get_xdata(), dtype=float).tolist() == [1e3, 1e4, 1e5]
    assert np.asarray(level.get_ydata(), dtype=float).tolist() == [0.0, 0.0]
    inverted = phase.get_lines()[-1]
    assert np.asarray(inverted.get_ydata(), dtype=float).tolist() == [-180.0, -180.0]
    assert phase.get_xlabel() == "frequency"


def test_a_sweep_plots_along_its_source() -> None:
    levels = np.array([0.0, 1.0])
    sweep = results.Sweep(UNKNOWNS, "I1", "current", levels, np.zeros((2, 5)))
    figure = plot("transfer", sweep, ["in"])
    assert figure.axes[0].get_xlabel() == "I1 current"


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
    point = results.OperatingPoint(UNKNOWNS, np.zeros(5), "direct")
    parameter = results.Parameter("R1", "resistance", 1e3)
    sensitivity = results.Sensitivities(
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
        message = f"{unplotted} is taken at one operating point: nothing to plot"
        assert message in capsys.readouterr().err
    assert run("plot", str(path), "--analyses", "ac") == 2
    assert run("plot", str(tmp_path / "missing.results.npz")) == 2
