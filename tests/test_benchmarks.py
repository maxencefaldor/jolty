# Copyright 2026 Maxence Faldor

import sys
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).parent.parent / "benchmarks"
COST_DOWN = BENCHMARKS / "cost_down"
CUBIC = BENCHMARKS / "cubic"


@pytest.fixture(autouse=True)
def importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Imports the benchmarks' scripts, which import each other, as their runs do."""
    monkeypatch.syspath_prepend(str(BENCHMARKS))


def design(work: Path, tolerance: str) -> None:
    """Writes the original bias as `work`'s design, its resistors at `tolerance`."""
    original = (COST_DOWN / "given" / "bias.py").read_text(encoding="utf-8")
    changed = original.replace("TOLERANCE = 0.001", f"TOLERANCE = {tolerance}")
    (work / "design.py").write_text(changed, encoding="utf-8")


def test_a_cheaper_design_that_meets_the_specs_meets_the_task(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    design(tmp_path, "0.01")
    verdict = score(COST_DOWN, tmp_path)
    assert verdict.met, verdict.reasons
    assert verdict.measurements == {
        "baseline": pytest.approx(0.2),
        "cost": pytest.approx(0.016),
    }


def test_the_score_says_why_work_falls_short(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    assert score(COST_DOWN, tmp_path).reasons == (
        f"there is no {tmp_path / 'design.py'}",
    )
    design(tmp_path, "0.001")
    assert score(COST_DOWN, tmp_path).reasons == (
        "the design costs 0.2, not less than 0.2",
    )
    design(tmp_path, "0.05")
    reasons = score(COST_DOWN, tmp_path).reasons
    assert "the design does not meet the plan of bias.py" in reasons
    original = (COST_DOWN / "given" / "bias.py").read_text(encoding="utf-8")
    (tmp_path / "design.py").write_text(
        original.replace("TOLERANCE = 0.001", "TOLERANCE = 0.01").replace(
            "voltage_source(5 * V)", "voltage_source(5.1 * V)"
        ),
        encoding="utf-8",
    )
    assert "the bench's V1 differs" in score(COST_DOWN, tmp_path).reasons


def test_the_runner_runs_the_agent_in_a_copy_of_the_task(tmp_path: Path) -> None:
    from run import run  # noqa: PLC0415 - importable only once the path holds it

    agent = [
        sys.executable,
        "-c",
        (
            "import pathlib, sys; source = pathlib.Path('bias.py').read_text(); "
            "pathlib.Path('design.py').write_text("
            "source.replace('TOLERANCE = 0.001', 'TOLERANCE = 0.01')); "
            "print(sys.argv[1].splitlines()[0])"
        ),
    ]
    assert run(COST_DOWN, agent, tmp_path) == 0
    assert (tmp_path / "catalog.json").is_file()
    assert not (tmp_path / "task.md").exists()
    assert not (tmp_path / "scorer.py").exists()
    transcript = (tmp_path / "transcript.txt").read_text(encoding="utf-8")
    assert transcript.startswith("# Cost-down: a transistor's bias")


def test_the_patented_cubic_generator_scores_one(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    patent = CUBIC / "patent.py"
    (tmp_path / "design.py").write_text(
        f"import runpy\nbuild = runpy.run_path({str(patent)!r})['cubic']\n",
        encoding="utf-8",
    )
    verdict = score(CUBIC, tmp_path)
    assert verdict.met, verdict.reasons
    assert verdict.measurements["score"] == pytest.approx(1.0)
    assert verdict.measurements["devices"] == 9


def test_the_cubic_scorer_names_a_forbidden_part(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    (tmp_path / "design.py").write_text(
        "from jolty import Builder, component, devices\n"
        "from jolty.units import A, kohm\n"
        "@component\n"
        "def cubic(c: Builder) -> None:\n"
        "    ground = c.port('ground')\n"
        "    for name in ('supply', 'reference', 'input'):\n"
        "        load = c.place(devices.resistor(1 * kohm), id=f'R_{name}')\n"
        "        c.connect(c.port(name), load.a)\n"
        "        c.connect(load.b, ground)\n"
        "    clamp = c.place(devices.diode(saturation_current=1e-14 * A), id='D9')\n"
        "    c.connect(c.port('output'), clamp.anode)\n"
        "    c.connect(clamp.cathode, ground)\n"
        "build = cubic\n",
        encoding="utf-8",
    )
    verdict = score(CUBIC, tmp_path)
    assert not verdict.met
    assert "D9 is a diode, which the task forbids" in verdict.reasons


ANALOGCODER = BENCHMARKS / "analogcoder"


def reference_designs(work: Path) -> None:
    """Writes each of AnalogCoder's problems' reference design into `work`."""
    reference = ANALOGCODER / "reference.py"
    for number in range(1, 25):
        (work / f"p{number}.py").write_text(
            "import runpy\n"
            f"_reference = runpy.run_path({str(reference)!r})\n"
            f"LEVELS = _reference['LEVELS'].get({number}, {{}})\n"
            "def build():\n"
            f"    return _reference['build']({number})\n",
            encoding="utf-8",
        )


def test_the_reference_designs_pass_every_analogcoder_check(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    reference_designs(tmp_path)
    verdict = score(ANALOGCODER, tmp_path)
    assert verdict.met, verdict.reasons
    assert verdict.measurements["solved"] == 24


def test_the_analogcoder_scorer_says_why_designs_fail(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    reference_designs(tmp_path)
    (tmp_path / "p3.py").unlink()
    (tmp_path / "p7.py").write_text(
        "from jolty import Builder, Role, component, devices\n"
        "@component\n"
        "def inverter(c: Builder) -> None:\n"
        "    ground = c.port('ground', Role.GROUND)\n"
        "    c.port('vdd', Role.SUPPLY)\n"
        "    amplifier = c.place(devices.voltage_controlled_voltage_source(-1.0), "
        "id='E1')\n"
        "    c.connect(amplifier.positive, c.port('vout'))\n"
        "    c.connect(amplifier.control_positive, c.port('vin'))\n"
        "    c.connect(amplifier.negative, amplifier.control_negative, ground)\n"
        "build = inverter\n",
        encoding="utf-8",
    )
    (tmp_path / "p18.py").write_text(
        "from jolty import Builder, Role, component, devices\n"
        "from jolty.units import V, kohm, uF\n"
        "LEVELS = {'vin': 2.5}\n"
        "@component\n"
        "def passive(c: Builder) -> None:\n"
        "    ground = c.port('ground', Role.GROUND)\n"
        "    c.port('vdd', Role.SUPPLY)\n"
        "    resistor = c.place(devices.resistor(10 * kohm), id='R1')\n"
        "    capacitor = c.place(devices.capacitor(3 * uF), id='CF')\n"
        "    rest = c.place(devices.voltage_source(2.5 * V), id='VR')\n"
        "    c.connect(resistor.a, c.port('vin'))\n"
        "    c.connect(resistor.b, capacitor.a, c.port('vout'))\n"
        "    c.connect(capacitor.b, rest.positive)\n"
        "    c.connect(rest.negative, ground)\n"
        "build = passive\n",
        encoding="utf-8",
    )
    verdict = score(ANALOGCODER, tmp_path)
    assert not verdict.met
    assert verdict.measurements["solved"] == 21
    assert "p3: there is no p3.py" in verdict.reasons
    assert "p7: E1 is a voltage_controlled_voltage_source, which the task forbids" in (
        verdict.reasons
    )
    assert any(
        reason.startswith("p18: vout ramps at") and "while vin is above" in reason
        for reason in verdict.reasons
    ), verdict.reasons


VOLTAGE_CURRENT = BENCHMARKS / "voltage_current"


def converter(work: Path, body: str) -> None:
    """Writes a voltage-current converter whose placements are `body` into `work`."""
    (work / "design.py").write_text(
        "from jolty import Builder, Role, component, devices\n"
        "from jolty.units import A, V, kohm\n"
        "@component\n"
        "def converter(c: Builder) -> None:\n"
        "    ground = c.port('ground', Role.GROUND)\n"
        "    c.port('vee', Role.SUPPLY)\n"
        "    plus, minus, output = c.port('plus'), c.port('minus'), c.port('output')\n"
        f"{body}"
        "build = converter\n",
        encoding="utf-8",
    )


def test_an_ideal_converter_has_no_error_but_is_forbidden(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    converter(
        tmp_path,
        "    source = c.place(devices.voltage_controlled_current_source(1e-3 * A / V), "
        "id='G1')\n"
        "    c.connect(source.control_positive, plus)\n"
        "    c.connect(source.control_negative, minus)\n"
        "    c.connect(source.positive, ground)\n"
        "    c.connect(source.negative, output)\n",
    )
    verdict = score(VOLTAGE_CURRENT, tmp_path)
    assert verdict.measurements["fitness"] == pytest.approx(0.0, abs=1e-9)
    assert verdict.reasons == (
        "G1 is a voltage_controlled_current_source, which the task forbids",
    )


def test_a_converter_blind_to_one_input_falls_short(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    converter(
        tmp_path,
        "    resistor = c.place(devices.resistor(1 * kohm), id='R1')\n"
        "    c.connect(resistor.a, plus)\n"
        "    c.connect(resistor.b, output)\n",
    )
    verdict = score(VOLTAGE_CURRENT, tmp_path)
    assert not verdict.met
    (reason,) = verdict.reasons
    assert reason.startswith("the fitness is ")
    assert verdict.measurements["devices"] == 1
