# Copyright 2026 Maxence Faldor

import json
from pathlib import Path

import pytest

from jolty import __version__
from jolty.cli import main
from jolty.files import load_results
from jolty.results import TransientResult

FILTER = """
from jolty import Builder, Plan, Role, Transient, component, devices
from jolty.units import V, kohm, nF, us


@component
def low_pass(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1 * V), name="V1")
    resistor = c.place(devices.resistor(1 * kohm), name="R1")
    capacitor = c.place(devices.capacitor(1 * nF), name="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)


def build():
    print("building the filter")
    return low_pass()


def plan():
    return Plan({"step": Transient(stop=5 * us)})
"""


@pytest.fixture(autouse=True)
def in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs each test in its own directory, where scripts and files are written."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def script(tmp_path: Path) -> Path:
    path = tmp_path / "low_pass.py"
    path.write_text(FILTER, encoding="utf-8")
    return path


def run(*arguments: str) -> int:
    """Runs the command line and returns its exit status."""
    try:
        main(arguments)
    except SystemExit as exit_:
        return exit_.code if isinstance(exit_.code, int) else 0
    return 0


def test_version_prints_the_installed_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run("--version") == 0
    assert capsys.readouterr().out == f"jolty {__version__}\n"


def test_build_saves_the_circuit_and_reports_its_size(
    script: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("build", "low_pass.py") == 0
    out = capsys.readouterr().out
    assert out.startswith(
        "saved low_pass.jolty.json: 1 component, 3 devices, 3 unknowns\n\n"
        "0 errors, 0 warnings\n\n"
        "script output:\n  building the filter\n\n"
        "drew low_pass.schematic.png\n\ntook "
    )
    assert json.loads(script.with_name("low_pass.jolty.json").read_text())["top"] == (
        "low_pass"
    )


def test_simulate_saves_the_results_and_reports_each_analysis(
    script: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("simulate", str(script), "--output-format", "json") == 0
    result = json.loads(capsys.readouterr().out)
    assert result["results"] == str(script.with_name("low_pass.results.npz"))
    (step,) = result["analyses"]
    assert step["kind"] == "transient"
    out = next(extent for extent in step["extents"] if extent["unknown"] == "out")
    assert out["last"] == pytest.approx(1.0, abs=1e-3)
    waveforms = load_results(Path(result["results"]))["step"]
    assert isinstance(waveforms, TransientResult)
    assert waveforms.voltage("out")[-1] == pytest.approx(out["last"])


def test_a_circuit_that_does_not_elaborate_exits_with_1(
    script: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("simulate", "low_pass.py") == 0
    capsys.readouterr()
    script.write_text(FILTER.replace("1 * nF", "-1 * nF"), encoding="utf-8")
    assert run("simulate", "low_pass.py") == 1
    out = capsys.readouterr().out
    assert (
        "error: capacitance must be at least 0, not -1e-9\n"
        "  = C1, low_pass.py:11 in low_pass()\n"
        "\n"
        "the circuit does not elaborate: 1 error"
    ) in out
    assert not script.with_name("low_pass.results.npz").exists()


def test_a_failed_analysis_exits_with_1(
    script: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script.write_text(
        FILTER.replace("Transient(stop=5 * us)", "Transient(stop=-5 * us)"),
        encoding="utf-8",
    )
    assert run("simulate", "low_pass.py") == 1
    assert "step: failed: invalid analysis" in capsys.readouterr().out


def test_a_failed_requirement_exits_with_1(
    script: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    specs = """

def specs():
    from jolty import Specs
    from jolty.specs import value

    return Specs([value("step", "out", at=5 * us) >= LEVEL])
"""
    script.write_text(FILTER + specs.replace("LEVEL", "0.9 * V"), encoding="utf-8")
    assert run("simulate", "low_pass.py", "--no-plot", "--no-schematic") == 0
    assert "specs: 1 passed" in capsys.readouterr().out
    script.write_text(FILTER + specs.replace("LEVEL", "1.1 * V"), encoding="utf-8")
    assert run("simulate", "low_pass.py", "--no-plot", "--no-schematic") == 1
    out = capsys.readouterr().out
    assert "failed: out at 5 µs in step ≥ 1.1 V: read 1 V, slack -100 mV" in out
    assert "  = low_pass.py:30 in specs()" in out


def test_corners_report_what_they_found(
    script: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    patched = FILTER.replace("def low_pass(c: Builder)", "def low_pass(c: Builder, f)")
    patched = patched.replace("devices.capacitor(1 * nF)", "devices.capacitor(f * nF)")
    patched = patched.replace("def build():", "def build(f=1):")
    patched = patched.replace("return low_pass()", "return low_pass(f)")
    corners = """

def plan():
    from jolty import Corner
    from jolty.units import celsius

    corners = {"cold": Corner(temperature=celsius(-40)), "broken": Corner(f=-1)}
    return Plan({"step": Transient(stop=5 * us)}, corners=corners)
"""
    script.write_text(patched + corners, encoding="utf-8")
    assert run("simulate", "low_pass.py", "--no-plot", "--no-schematic") == 1
    out = capsys.readouterr().out
    assert "corner cold, at -40 °C: every analysis ended" in out
    assert "corner broken, at 27 °C: the circuit does not elaborate:" in out


def test_failing_scripts_exit_with_2_and_say_why(
    script: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script.write_text(FILTER.replace('name="R1"', 'name="R 1"'), encoding="utf-8")
    assert run("build", "low_pass.py") == 2
    err = capsys.readouterr().err
    assert err.startswith(
        "error: the script failed: low_pass.py:10 in low_pass(): BuildError: "
        "an instance name uses letters"
    )
    assert "called from low_pass.py:19 in build()" in err
    assert not script.with_name("low_pass.jolty.json").exists()
    assert run("build", "missing.py", "--output-format=json") == 2
    assert json.loads(capsys.readouterr().out) == {
        "error": "no such script: missing.py"
    }


def test_scripts_that_run_too_long_are_stopped(
    script: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script.write_text("import time\n\n\ndef build():\n    time.sleep(60)\n")
    monkeypatch.setenv("JOLTY_TIMEOUT", "0.5")
    assert run("build", "low_pass.py") == 2
    assert capsys.readouterr().err == (
        "error: low_pass.py did not finish within 0.5 s and was stopped\n"
    )


def test_invalid_arguments_and_settings_exit_with_2(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run() == 2
    assert "CLI subcommand is required" in capsys.readouterr().err
    assert run("build", "low_pass.py", "--output-format", "yaml") == 2
    capsys.readouterr()
    monkeypatch.setenv("JOLTY_MAX_UNKNOWNS", "0")
    assert run("build", "low_pass.py") == 2
    assert capsys.readouterr().err == (
        "error: invalid settings: max_unknowns: "
        "Input should be greater than or equal to 1\n"
    )


def test_settings_come_from_pyproject(
    tmp_path: Path, script: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.jolty]\nmax_unknowns = 1\n")
    assert run("simulate", script.name) == 0
    assert "  ... and 2 more in the results file" in capsys.readouterr().out
