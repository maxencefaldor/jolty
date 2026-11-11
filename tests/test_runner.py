# Copyright 2026 Maxence Faldor

from pathlib import Path

import pytest

from jolty import report
from jolty.files import load_results
from jolty.runner import RESULTS, Built, Failure, Job, Outcome, run

DIVIDER = """
from jolty import Builder, OperatingPoint, Plan, Role, Transient, component, devices
from jolty.units import V, kohm, us


@component
def divider(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(LEVEL), id="V1")
    upper = c.place(devices.resistor(1 * kohm), id="R1")
    lower = c.place(devices.resistor(1 * kohm), id="R2")
    c.connect(source.positive, upper.a)
    c.connect(upper.b, lower.a, name="out")
    c.connect(lower.b, source.negative, ground)


LEVEL = 5 * V


def build():
    return divider()


def plan():
    return Plan({"op": OperatingPoint(), "step": Transient(stop=1 * us)})
"""


NO_FREQUENCIES = """
from jolty import Ac, Plan


def plan():
    return Plan({"ac": Ac("V1", [])})
"""


@pytest.fixture(autouse=True)
def in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs each test in its own directory, where script paths are relative."""
    monkeypatch.chdir(tmp_path)


def write(directory: Path, text: str, name: str = "script.py") -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def built(outcome: Outcome) -> Built:
    assert isinstance(outcome, Built), str(outcome)
    return outcome


def failed(outcome: Outcome) -> Failure:
    assert isinstance(outcome, Failure)
    return outcome


def test_a_build_elaborates_the_circuit(tmp_path: Path) -> None:
    job = Job(script=write(tmp_path, DIVIDER), listed=10)
    outcome = built(run(job, tmp_path))
    assert outcome.circuit.top == "divider"
    assert (outcome.devices, outcome.unknowns) == (3, 3)
    assert outcome.problems == ()
    assert outcome.analyses == ()
    assert not (tmp_path / RESULTS).exists()


def test_a_plan_runs_its_analyses_and_saves_their_results(tmp_path: Path) -> None:
    script = write(tmp_path, DIVIDER)
    outcome = built(run(Job(script=script, plan=script, listed=10), tmp_path))
    point, step = outcome.analyses
    assert isinstance(point, report.PointReport)
    assert [value.value for value in point.values][1] == 2.5
    assert isinstance(step, report.TransientReport)
    assert step.stop == 1e-6
    assert set(load_results(tmp_path / RESULTS)) == {"op", "step"}


def test_a_plan_runs_at_its_temperature(tmp_path: Path) -> None:
    drifting = DIVIDER.replace(
        'devices.resistor(1 * kohm), id="R2"',
        'devices.resistor(1 * kohm, temperature_coefficient=1e-2 / K), id="R2"',
    ).replace("import V, kohm, us", "import K, V, kohm, us")
    hot = "def plan():\n    return Plan({'op': OperatingPoint()}, temperature=400 * K)"
    script = write(tmp_path, f"{drifting}\n\n{hot}\n")
    outcome = built(run(Job(script=script, plan=script, listed=10), tmp_path))
    (point,) = outcome.analyses
    assert isinstance(point, report.PointReport)
    factor = 1 + 1e-2 * (400 - 300.15)
    assert point.values[1].value == pytest.approx(5 * factor / (1 + factor))


def test_a_plan_may_live_in_its_own_script(tmp_path: Path) -> None:
    script = write(tmp_path, DIVIDER)
    plan = write(tmp_path, NO_FREQUENCIES, "plan.py")
    outcome = built(run(Job(script=script, plan=plan, listed=10), tmp_path))
    (failure,) = outcome.analyses
    assert isinstance(failure, report.FailureReport)
    assert failure.name == "ac"
    assert "frequenc" in failure.error
    assert not (tmp_path / RESULTS).exists()


def test_problems_come_back_with_the_circuit(tmp_path: Path) -> None:
    script = write(
        tmp_path, DIVIDER.replace('1 * kohm), id="R2"', '-1 * kohm), id="R2"')
    )
    outcome = built(run(Job(script=script, plan=script, listed=10), tmp_path))
    assert outcome.devices is None
    assert outcome.unknowns is None
    (problem,) = outcome.problems
    assert problem.startswith("instance `R2` (script.py:11): resistance must be")
    assert outcome.analyses == ()


def test_failures_name_the_script_lines_innermost_first(tmp_path: Path) -> None:
    text = "def helper():\n    return 1 / 0\n\n\ndef build():\n    return helper()\n"
    failure = failed(run(Job(script=write(tmp_path, text), listed=10), tmp_path))
    assert failure.error == "ZeroDivisionError"
    assert [frame.line for frame in failure.frames] == [2, 6]
    assert str(failure) == (
        "script.py:2 in helper(): ZeroDivisionError: division by zero\n"
        "  called from script.py:6 in build()"
    )


def test_errors_inside_jolty_point_at_the_calling_line(tmp_path: Path) -> None:
    script = write(tmp_path, DIVIDER.replace("LEVEL = 5 * V", "LEVEL = 5 * kohm"))
    failure = failed(run(Job(script=script, listed=10), tmp_path))
    assert failure.error == "DimensionError"
    assert [frame.line for frame in failure.frames] == [9, 21]
    assert "a DC level takes a quantity in V, not 5 kΩ" in failure.message


def test_exits_and_wrong_entry_points_are_failures(tmp_path: Path) -> None:
    def failure_of(text: str) -> str:
        return str(failed(run(Job(script=write(tmp_path, text), listed=10), tmp_path)))

    assert failure_of("import sys\nsys.exit(3)\n") == (
        "script.py:2 at module level: SystemExit: the script exited with 3"
    )
    assert failure_of("x = 1\n") == "ScriptError: script.py defines no build() function"
    assert failure_of("def build():\n    return 1\n") == (
        "ScriptError: build() returned int, not a Component or a Circuit"
    )
    no_plan = DIVIDER.replace("def plan():", "def other():")
    script = write(tmp_path, no_plan)
    failure = failed(run(Job(script=script, plan=script, listed=10), tmp_path))
    assert failure.message == "script.py defines no plan() function"


def test_scripts_import_the_modules_beside_them(tmp_path: Path) -> None:
    write(tmp_path, "from jolty.units import V\n\nLEVEL = 3 * V\n", "levels.py")
    text = DIVIDER.replace("LEVEL = 5 * V", "from levels import LEVEL")
    built(run(Job(script=write(tmp_path, text), listed=10), tmp_path))
