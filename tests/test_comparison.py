# Copyright 2026 Maxence Faldor

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from differential import SKIP

BENCHMARKS = Path(__file__).parent.parent / "benchmarks"
COST_DOWN = BENCHMARKS / "cost_down"


@pytest.fixture(autouse=True)
def importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Imports the benchmarks' scripts, which import each other, as their runs do."""
    monkeypatch.syspath_prepend(str(BENCHMARKS))


def test_a_wrapper_logs_each_run_then_runs_the_program(tmp_path: Path) -> None:
    from run import wrap  # noqa: PLC0415 - importable only once the path holds it

    program = tmp_path / "simulator"
    program.write_text(
        '#!/bin/sh\nprintf "%s|" "$@"\necho "$PWD" >&2\nexit 3\n', encoding="utf-8"
    )
    program.chmod(0o755)
    wrappers = tmp_path / "bin"
    wrappers.mkdir()
    log = tmp_path / "simulations.jsonl"
    wrap(wrappers, program, log)
    arguments = ["-b", "a deck.cir", "it's", ""]
    for _ in range(2):
        ran = subprocess.run(  # noqa: S603 - the wrapper just written
            [str(wrappers / "simulator"), *arguments],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert ran.returncode == 3
        assert ran.stdout == "-b|a deck.cir|it's||"
        assert ran.stderr == f"{tmp_path.resolve()}\n"
    runs = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [(each["program"], each["arguments"]) for each in runs] == [
        ("simulator", arguments)
    ] * 2
    assert runs[0]["time"] <= runs[1]["time"]


@SKIP
def test_the_ngspice_arm_has_ngspice_and_python_but_no_jolty(tmp_path: Path) -> None:
    from arms import Arm  # noqa: PLC0415 - importable only once the path holds it
    from run import PACKAGES, SIMULATIONS, environment  # noqa: PLC0415 - importable only once the path holds it

    prepared = environment(Arm.NGSPICE, tmp_path)
    assert set(PACKAGES) <= set(prepared.packages)
    assert "jolty" not in prepared.packages
    path = prepared.variables["PATH"].split(os.pathsep)
    assert path[:2] == [str(tmp_path / "bin"), str(tmp_path / "python" / "bin")]
    assert not [each for each in path if (Path(each) / "jolty").exists()]

    def ran(*command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - the arm's own programs
            command,
            env=prepared.variables,
            capture_output=True,
            text=True,
            check=False,
        )

    probe = "import numpy, scipy, matplotlib, sys; print(sys.prefix)"
    assert ran("python", "-c", probe).stdout.strip() == str(tmp_path / "python")
    assert "No module named 'jolty'" in ran("python", "-c", "import jolty").stderr
    assert "ngspice" in ran("ngspice", "--version").stdout
    (logged,) = (tmp_path / SIMULATIONS).read_text(encoding="utf-8").splitlines()
    assert json.loads(logged)["arguments"] == ["--version"]


def test_the_jolty_arm_counts_its_command_lines_runs(tmp_path: Path) -> None:
    from arms import Arm  # noqa: PLC0415 - importable only once the path holds it
    from run import SIMULATIONS, environment  # noqa: PLC0415 - importable only once the path holds it

    prepared = environment(Arm.JOLTY, tmp_path)
    assert not prepared.packages
    path = prepared.variables["PATH"].split(os.pathsep)
    assert path[:2] == [str(tmp_path / "bin"), str(Path(sys.executable).parent)]
    ran = subprocess.run(
        ["jolty", "--version"],  # noqa: S607 - the wrapper, first on the arm's path
        env=prepared.variables,
        capture_output=True,
        text=True,
        check=True,
    )
    assert ran.stdout.startswith("jolty ")
    (logged,) = (tmp_path / SIMULATIONS).read_text(encoding="utf-8").splitlines()
    assert json.loads(logged)["program"] == "jolty"


def _assistant(id: str, *blocks: dict[str, object], parent: str | None = None) -> str:
    """A line of an assistant event: blocks of the model's message `id`."""
    message = {"id": id, "role": "assistant", "content": list(blocks)}
    event = {"type": "assistant", "message": message, "parent_tool_use_id": parent}
    return json.dumps(event)


def _call(id: str, name: str) -> dict[str, object]:
    return {"type": "tool_use", "id": id, "name": name, "input": {}}


def test_the_transcript_reader_counts_turns_and_tool_calls(tmp_path: Path) -> None:
    from transcript import read  # noqa: PLC0415 - importable only once the path holds it

    lines = [
        "a warning the agent printed",
        json.dumps(
            {
                "type": "system",
                "subtype": "init",
                "model": "claude-opus-5-5",
                "claude_code_version": "2.1.325",
                "tools": ["Bash"],
            }
        ),
        _assistant("msg_1", {"type": "text", "text": "I read the task."}),
        # One message, streamed as one event per block.
        _assistant("msg_2", {"type": "thinking", "thinking": ""}),
        _assistant("msg_2", _call("toolu_1", "Bash")),
        _assistant("msg_2", _call("toolu_2", "Write")),
        json.dumps(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "ok",
                        }
                    ],
                },
            }
        ),
        json.dumps([1, 2]),
        # A subagent's message, its calls counted with the agent's own.
        _assistant("msg_3", _call("toolu_3", "Bash"), parent="toolu_2"),
        _assistant("msg_4", _call("toolu_4", "mcp__jolty__simulate")),
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "num_turns": 5,
                "duration_ms": 61500,
                "total_cost_usd": 1.25,
                "usage": {
                    "input_tokens": 1200,
                    "output_tokens": 340,
                    "cache_read_input_tokens": 9000,
                    "service_tier": "x",
                },
            }
        ),
    ]
    path = tmp_path / "transcript.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    session = read(path)
    assert session is not None
    assert (session.model, session.agent) == ("claude-opus-5-5", "2.1.325")
    assert session.turns == 4
    assert session.tool_calls == {"Bash": 2, "Write": 1, "mcp__jolty__simulate": 1}
    assert session.calls == 4
    assert (session.reported_turns, session.seconds, session.cost) == (5, 61.5, 1.25)
    assert session.tokens == {
        "input_tokens": 1200,
        "output_tokens": 340,
        "cache_read_input_tokens": 9000,
    }
    assert session.outcome == "success"


def test_a_transcript_of_no_events_says_nothing_of_its_session(tmp_path: Path) -> None:
    from transcript import read  # noqa: PLC0415 - importable only once the path holds it

    path = tmp_path / "transcript.txt"
    path.write_text("I wrote design.py.\n", encoding="utf-8")
    assert read(path) is None
    path.write_text(_assistant("msg_1", _call("toolu_1", "Bash")), encoding="utf-8")
    session = read(path)
    assert session is not None
    assert (session.turns, session.calls, session.seconds) == (1, 1, None)


@SKIP
def test_the_comparison_runs_and_reports_both_arms_with_the_stub(
    tmp_path: Path,
) -> None:
    from arms import Arm  # noqa: PLC0415 - importable only once the path holds it
    from compare import Report, compare, wilson  # noqa: PLC0415 - importable only once the path holds it

    work = tmp_path / "work"
    work.mkdir()
    path = compare(
        "stub",
        tasks=["cost_down"],
        arms=list(Arm),
        repetitions=2,
        results=tmp_path,
        work=work,
    )
    report = Report.model_validate_json(path.read_text(encoding="utf-8"))
    assert path == tmp_path / f"{report.date}-stub.json"
    assert [(run.arm, run.repetition) for run in report.runs] == [
        (Arm.JOLTY, 1),
        (Arm.NGSPICE, 1),
        (Arm.JOLTY, 2),
        (Arm.NGSPICE, 2),
    ]
    assert all(run.met and run.exit_code == 0 for run in report.runs)
    assert all(Path(run.work).parent == work for run in report.runs)
    # The stub writes design.py, or design.cir and tolerances.json, a turn each, then
    # runs the arm's simulator once in a turn of its own.
    written = {Arm.JOLTY: 1, Arm.NGSPICE: 2}
    for run in report.runs:
        assert run.session is not None
        assert run.session.turns == written[run.arm] + 1
        assert run.session.tool_calls == {"Bash": 1, "Write": written[run.arm]}
        assert run.simulations.commands == {f"{run.arm} --version": 1}
        assert bool(run.packages) == (run.arm is Arm.NGSPICE)
        kept = tmp_path / f"{report.date}-stub" / "cost_down" / run.arm
        for name in ("transcript.txt", "simulations.jsonl"):
            assert (kept / str(run.repetition) / name).is_file()
    task = report.tasks["cost_down"]
    low, high = wilson(2, 2)
    for arm in Arm:
        summary = task.arms[arm]
        assert summary.success.model_dump() == {
            "met": 2,
            "runs": 2,
            "rate": 1.0,
            "low": low,
            "high": high,
        }
        assert summary.turns is not None
        assert summary.turns.median == written[arm] + 1
        assert report.arms[arm] == summary.success
    difference = task.difference
    assert difference is not None
    # Newcombe's interval about a difference of 0 spans each rate's Wilson interval.
    assert (difference.success, difference.low, difference.high) == (
        0,
        low - 1,
        1 - low,
    )
    assert (difference.turns, difference.tool_calls, difference.simulations) == (
        -1,
        -1,
        0,
    )
    with pytest.raises(FileExistsError):
        compare(
            "stub",
            tasks=["cost_down"],
            arms=[Arm.JOLTY],
            repetitions=1,
            results=tmp_path,
        )


def test_the_comparison_runs_no_session_without_its_programs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from arms import Arm  # noqa: PLC0415 - importable only once the path holds it
    from compare import compare  # noqa: PLC0415 - importable only once the path holds it

    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="claude, ngspice, uv not on the path"):
        compare(
            "claude",
            tasks=["cost_down"],
            arms=list(Arm),
            repetitions=1,
            results=tmp_path,
        )
    assert not list(tmp_path.iterdir())
    monkeypatch.undo()
    with pytest.raises(ValueError, match="lies in the repository"):
        compare(
            "stub",
            tasks=["cost_down"],
            arms=[Arm.JOLTY],
            repetitions=1,
            results=tmp_path,
            work=BENCHMARKS,
        )


def test_each_session_is_scored_in_a_process_of_its_own(tmp_path: Path) -> None:
    from arms import Arm  # noqa: PLC0415 - importable only once the path holds it
    from run import VERDICT, scored  # noqa: PLC0415 - importable only once the path holds it

    bias = (COST_DOWN / "given" / "bias.py").read_text(encoding="utf-8")
    verdicts = []
    # Each design imports its resistors' tolerance from a module of the same name
    # beside it: scored in one process, the second would find the first's.
    for name, tolerance in (("cheap", 0.01), ("loose", 0.05)):
        work = tmp_path / name
        work.mkdir()
        (work / "helper.py").write_text(f"TOLERANCE = {tolerance}\n", encoding="utf-8")
        design = bias.replace("TOLERANCE = 0.001", "from helper import TOLERANCE")
        (work / "design.py").write_text(design, encoding="utf-8")
        verdicts.append(scored(COST_DOWN, work, Arm.JOLTY))
        assert (work / VERDICT).is_file()
    assert [verdict.met for verdict in verdicts] == [True, False]
    assert "the design does not meet the plan of bias.py" in verdicts[1].reasons


def test_a_scorer_that_crashes_fails_the_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from arms import Arm  # noqa: PLC0415 - importable only once the path holds it
    from run import scored  # noqa: PLC0415 - importable only once the path holds it

    # As on CI, which forces colors: the scorer's traceback is read, not shown.
    monkeypatch.setenv("FORCE_COLOR", "1")
    task = tmp_path / "broken"
    task.mkdir()
    (task / "scorer.py").write_text(
        "def score(task, work, arm):\n    raise RuntimeError('a broken scorer')\n",
        encoding="utf-8",
    )
    verdict = scored(task, tmp_path, Arm.JOLTY)
    assert not verdict.met
    assert verdict.reasons == (
        "the scorer exited with 1, having written RuntimeError: a broken scorer",
    )
