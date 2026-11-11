# Copyright 2026 Maxence Faldor

import sys
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).parent.parent / "benchmarks"
COST_DOWN = BENCHMARKS / "cost_down"


@pytest.fixture(autouse=True)
def importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Imports the benchmarks' scripts, which import each other, as their runs do."""
    monkeypatch.syspath_prepend(str(BENCHMARKS))


def design(work: Path, tolerance: str) -> None:
    """Writes the original bias as `work`'s design, its resistors at `tolerance`."""
    original = (COST_DOWN / "bias.py").read_text(encoding="utf-8")
    changed = original.replace("TOLERANCE = 0.001", f"TOLERANCE = {tolerance}")
    (work / "design.py").write_text(changed, encoding="utf-8")


def test_a_cheaper_design_that_meets_the_specs_meets_the_task(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    design(tmp_path, "0.01")
    verdict = score(COST_DOWN, tmp_path)
    assert verdict.met, verdict.reasons
    assert (verdict.baseline, verdict.cost) == (
        pytest.approx(0.2),
        pytest.approx(0.016),
    )


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
    original = (COST_DOWN / "bias.py").read_text(encoding="utf-8")
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
    transcript = (tmp_path / "transcript.txt").read_text(encoding="utf-8")
    assert transcript.startswith("# Cost-down: a transistor's bias")
