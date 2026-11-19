# Copyright 2026 Maxence Faldor

import json
import re
from pathlib import Path

import pytest

from jolty import __version__, specs
from jolty.cli import main
from jolty.files import load_raw, load_results
from jolty.results import AcResult, OperatingPointResult, TransientResult

pytestmark = pytest.mark.usefixtures("in_tmp_path")

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
ADVICE_WORDS = re.compile(
    r"\b(?:use|pass|try|instead|should|give|did you mean)\b", re.IGNORECASE
)

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


class _Captured:
    """Captured standard output and error streams from a CLI invocation."""

    __slots__ = ("err", "out")

    def __init__(self, out: str, err: str) -> None:
        self.out = out
        self.err = err


def _invoke(
    capsys: pytest.CaptureFixture[str],
    *arguments: str,
    code: int = 0,
) -> _Captured:
    assert run(*arguments) == code
    captured = capsys.readouterr()
    return _Captured(captured.out, captured.err)


def test_version_prints_the_installed_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _invoke(capsys, "--version").out == f"jolty {__version__}\n"


def test_build_saves_the_circuit_and_reports_its_size(
    script: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _invoke(capsys, "build", "low_pass.py").out
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
    result = json.loads(
        _invoke(capsys, "simulate", str(script), "--output-format", "json").out
    )
    assert result["results"] == str(script.with_name("low_pass.results.npz"))
    (step,) = result["analyses"]
    assert step["kind"] == "transient"
    out = next(extent for extent in step["extents"] if extent["unknown"] == "out")
    # Steady DC 1 V source; abs=1e-3 matches default engine reltol=1e-3 at 1 V.
    assert out["last"] == pytest.approx(1.0, abs=1e-3)
    waveforms = load_results(Path(result["results"]))["step"]
    assert isinstance(waveforms, TransientResult)
    # Default pytest.approx (rel=1e-6) covers JSON text round-trip of the same value.
    assert waveforms.voltage("out")[-1] == pytest.approx(out["last"])


def test_a_circuit_that_does_not_elaborate_exits_with_1(
    script: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _invoke(capsys, "simulate", "low_pass.py")
    script.write_text(FILTER.replace("1 * nF", "-1 * nF"), encoding="utf-8")
    out = _invoke(capsys, "simulate", "low_pass.py", code=1).out
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
        FILTER.replace(
            "from jolty import Builder, Plan",
            "from jolty import Builder, Options, Plan",
        ).replace(
            'Plan({"step": Transient(stop=5 * us)})',
            'Plan({"step": Transient(stop=5 * us)}, options=Options(max_memory=1e-9))',
        ),
        encoding="utf-8",
    )
    out = _invoke(capsys, "simulate", "low_pass.py", code=1).out
    assert "step: failed: the transient's results would pass" in out


def test_a_failed_requirement_exits_with_1(
    script: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec_code = """

def specs():
    from jolty import Specs
    from jolty.specs import value

    return Specs([value("step", "out", at=5 * us) >= LEVEL])
"""
    script.write_text(FILTER + spec_code.replace("LEVEL", "0.9 * V"), encoding="utf-8")
    assert (
        "specs: 1 passed"
        in _invoke(capsys, "simulate", "low_pass.py", "--no-plot", "--no-schematic").out
    )
    script.write_text(FILTER + spec_code.replace("LEVEL", "1.1 * V"), encoding="utf-8")
    out = _invoke(
        capsys, "simulate", "low_pass.py", "--no-plot", "--no-schematic", code=1
    ).out
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
    out = _invoke(
        capsys, "simulate", "low_pass.py", "--no-plot", "--no-schematic", code=1
    ).out
    assert "corner cold, at -40 °C: every analysis ended" in out
    assert "corner broken, at 27 °C: the circuit does not elaborate:" in out


def test_failing_scripts_exit_with_2_and_say_why(
    script: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script.write_text(FILTER.replace('name="R1"', 'name="R 1"'), encoding="utf-8")
    err = _invoke(capsys, "build", "low_pass.py", code=2).err
    assert err.startswith(
        "error: the script failed: low_pass.py:10 in low_pass(): BuildError: "
        "an instance name uses letters"
    )
    assert "called from low_pass.py:19 in build()" in err
    assert not script.with_name("low_pass.jolty.json").exists()
    out = _invoke(capsys, "build", "missing.py", "--output-format=json", code=2).out
    assert json.loads(out) == {"error": "no such script: missing.py"}


def test_scripts_that_run_too_long_are_stopped(
    script: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script.write_text("import time\n\n\ndef build():\n    time.sleep(60)\n")
    monkeypatch.setenv("JOLTY_TIMEOUT", "0.5")
    assert _invoke(capsys, "build", "low_pass.py", code=2).err == (
        "error: low_pass.py did not finish within 0.5 s and was stopped\n"
    )


def test_invalid_arguments_and_settings_exit_with_2(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert "CLI subcommand is required" in _invoke(capsys, code=2).err
    _invoke(capsys, "build", "low_pass.py", "--output-format", "yaml", code=2)
    monkeypatch.setenv("JOLTY_MAX_UNKNOWNS", "0")
    assert _invoke(capsys, "build", "low_pass.py", code=2).err == (
        "error: invalid settings: max_unknowns: "
        "Input should be greater than or equal to 1\n"
    )


def test_settings_come_from_pyproject(
    tmp_path: Path, script: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.jolty]\nmax_unknowns = 1\n")
    assert (
        "  ... and 2 more in the results file"
        in _invoke(capsys, "simulate", script.name).out
    )


def test_simulate_raw_binary_and_ascii_round_trip(
    script: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_binary = tmp_path / "out.raw"
    payload = json.loads(
        _invoke(
            capsys,
            "simulate",
            str(script),
            "--raw",
            str(raw_binary),
            "--no-plot",
            "--no-schematic",
            "--output-format",
            "json",
        ).out
    )
    assert payload["raw"] == str(raw_binary)
    assert b"Binary:\n" in raw_binary.read_bytes()
    expected = load_results(Path(payload["results"]))["step"]
    loaded = load_raw(raw_binary)["step"]
    assert isinstance(expected, TransientResult)
    assert isinstance(loaded, TransientResult)
    assert loaded.unknowns == expected.unknowns
    assert (loaded.times.view("u8") == expected.times.view("u8")).all()
    assert (loaded.values.view("u8") == expected.values.view("u8")).all()

    raw_ascii = tmp_path / "nested" / "out_ascii.raw"
    out = _invoke(
        capsys,
        "simulate",
        str(script),
        "--raw",
        str(raw_ascii),
        "--raw-format",
        "ascii",
        "--no-plot",
        "--no-schematic",
    ).out
    assert f"saved {raw_ascii}" in out
    assert b"Values:\n" in raw_ascii.read_bytes()
    loaded_ascii = load_raw(raw_ascii)["step"]
    assert isinstance(loaded_ascii, TransientResult)
    assert (loaded_ascii.times.view("u8") == expected.times.view("u8")).all()
    assert (loaded_ascii.values.view("u8") == expected.values.view("u8")).all()


def test_simulate_raw_skips_non_raw_analyses_and_refuses_empty_set(
    script: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mixed = FILTER.replace(
        'return Plan({"step": Transient(stop=5 * us)})',
        "from jolty import Ac, Noise, OperatingPoint\n"
        "    from jolty.plan import decade\n"
        "    from jolty.units import kHz, MHz\n"
        "    return Plan({\n"
        '        "op": OperatingPoint(),\n'
        '        "bode": Ac("V1", decade(1 * kHz, 1 * MHz, 2)),\n'
        '        "step": Transient(stop=5 * us),\n'
        '        "hiss": Noise("out", decade(1 * kHz, 1 * MHz, 2)),\n'
        "    })",
    )
    script.write_text(mixed, encoding="utf-8")
    raw_path = tmp_path / "mixed.raw"
    _invoke(
        capsys,
        "simulate",
        str(script),
        "--raw",
        str(raw_path),
        "--no-plot",
        "--no-schematic",
    )
    loaded = load_raw(raw_path)
    assert list(loaded) == ["op", "bode", "step"]
    assert isinstance(loaded["op"], OperatingPointResult)
    assert isinstance(loaded["bode"], AcResult)
    assert isinstance(loaded["step"], TransientResult)

    noise_only = FILTER.replace(
        'return Plan({"step": Transient(stop=5 * us)})',
        "from jolty import Noise\n"
        "    from jolty.plan import decade\n"
        "    from jolty.units import kHz, MHz\n"
        '    return Plan({"hiss": Noise("out", decade(1 * kHz, 1 * MHz, 2))})',
    )
    script.write_text(noise_only, encoding="utf-8")
    assert raw_path.is_file()
    err = _invoke(
        capsys,
        "simulate",
        str(script),
        "--raw",
        str(raw_path),
        "--no-plot",
        "--no-schematic",
        code=2,
    ).err
    assert "NoiseResult" in err
    assert not raw_path.exists()


def test_simulate_raw_removes_stale_file_on_elaboration_failure(
    script: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_path = tmp_path / "stale.raw"
    raw_path.write_text("stale", encoding="utf-8")
    script.write_text(FILTER.replace("1 * nF", "-1 * nF"), encoding="utf-8")
    _invoke(
        capsys,
        "simulate",
        str(script),
        "--raw",
        str(raw_path),
        "--no-plot",
        "--no-schematic",
        code=1,
    )
    assert not raw_path.exists()


def test_simulate_raw_option_validation_errors(
    script: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    flags = ("--no-plot", "--no-schematic")
    err_format = _invoke(
        capsys, "simulate", str(script), "--raw-format", "ascii", *flags, code=2
    ).err
    assert "--raw-format applies only when --raw is set" in err_format
    err_shard = _invoke(
        capsys,
        "simulate",
        str(script),
        "--samples",
        "0:2",
        "--raw",
        str(tmp_path / "shard.raw"),
        *flags,
        code=2,
    ).err
    assert "writes no raw file" in err_shard


def test_read_docs_lists_and_reads_guide_and_examples(
    capsys: pytest.CaptureFixture[str],
) -> None:
    listed = _invoke(capsys, "read-docs").out
    assert "guide (17 pages, jolty://docs/guide/<slug>):" in listed
    assert "examples (33 scripts, jolty://docs/examples/<slug>):" in listed
    assert "units" in listed
    assert "cluster/amplifier" in listed

    guide_list = _invoke(capsys, "read-docs", "--section", "guide").out
    assert "units" in guide_list
    assert "rc_filter" not in guide_list

    examples_list = _invoke(capsys, "read-docs", "--section", "examples").out
    assert "rc_filter" in examples_list
    assert "cluster/amplifier" in examples_list
    assert "guide (" not in examples_list

    expected_units = (REPOSITORY_ROOT / "docs" / "guide" / "units.md").read_text(
        encoding="utf-8"
    )
    assert _invoke(capsys, "read-docs", "units").out == expected_units
    assert _invoke(capsys, "read-docs", "guide/units").out == expected_units

    expected_example = (
        REPOSITORY_ROOT / "examples" / "cluster" / "amplifier.py"
    ).read_text(encoding="utf-8")
    assert _invoke(capsys, "read-docs", "cluster/amplifier").out == expected_example

    payload = json.loads(
        _invoke(capsys, "read-docs", "units", "--output-format", "json").out
    )
    assert payload["entries"] == []
    assert payload["document"]["uri"] == "jolty://docs/guide/units"
    assert payload["document"]["mime_type"] == "text/markdown"
    assert payload["document"]["content"] == expected_units

    listing_payload = json.loads(
        _invoke(capsys, "read-docs", "--output-format", "json").out
    )
    assert listing_payload["document"] is None
    assert len(listing_payload["entries"]) == 50


@pytest.mark.parametrize(
    "arguments",
    [
        ("read-docs", "nonexistent"),
        ("read-docs", "../README.md"),
        ("read-docs", "/etc/passwd"),
        ("read-docs", "cluster/../rc_filter"),
        ("read-docs", ""),
        ("read-docs", "guide/units", "--section", "examples"),
    ],
)
def test_read_docs_rejects_invalid_and_unknown_topics_with_facts(
    arguments: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    error_text = _invoke(capsys, *arguments, code=2).err
    assert error_text.startswith("error: ")
    assert not ADVICE_WORDS.search(error_text), error_text

    json_error = json.loads(
        _invoke(capsys, *arguments, "--output-format=json", code=2).out
    )
    assert isinstance(json_error["error"], str)
    assert not ADVICE_WORDS.search(json_error["error"]), json_error["error"]


def test_measure_reads_saved_results_without_running_a_simulation(
    script: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _invoke(capsys, "simulate", str(script), "--no-plot", "--no-schematic")
    saved = script.with_name("low_pass.results.npz")

    log_path = tmp_path / "analyses.jsonl"
    monkeypatch.setenv("JOLTY_ANALYSIS_LOG", str(log_path))

    # Single-analysis file: --analysis can be omitted.
    payload = json.loads(
        _invoke(
            capsys,
            "measure",
            str(saved),
            "--measure",
            "rms",
            "--signal",
            "out",
            "--output-format",
            "json",
        ).out
    )
    loaded = load_results(saved)["step"]

    expected = specs.rms("step", "out").read(loaded)
    assert payload["analysis"] == "step"
    assert payload["measure"] == "rms"
    assert payload["value"] == expected.value
    assert payload["unit"] == "V"

    # Text output format.
    text = _invoke(
        capsys,
        "measure",
        str(saved),
        "--analysis",
        "step",
        "--measure",
        "value",
        "--signal",
        "out",
        "--at",
        "5e-6",
    ).out.strip()
    assert text.startswith("out at 5 µs in step: ")

    # Missing analysis fails with exit code 2 and names available analyses.
    err = _invoke(
        capsys,
        "measure",
        str(saved),
        "--analysis",
        "missing",
        "--measure",
        "value",
        "--signal",
        "out",
        code=2,
    ).err
    assert "holds no analysis named 'missing'" in err
    assert "step" in err

    # Zero simulations executed during any measure command.
    assert not log_path.exists()
