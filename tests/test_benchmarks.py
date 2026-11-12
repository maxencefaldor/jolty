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
