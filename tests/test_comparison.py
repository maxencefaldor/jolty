# Copyright 2026 Maxence Faldor

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import compare
import pytest
import run as agent_run
import score as agent_score
import transcript
from arms import Arm
from sandbox import Sandbox, bubblewrap, mechanism
from verdict import Verdict

from differential import SKIP
from jolty.settings import Settings
from jolty.specs import wilson

BENCHMARKS = Path(__file__).parent.parent / "benchmarks" / "agents"
COST_DOWN = BENCHMARKS / "cost_down"


def test_a_wrapper_logs_each_run_then_runs_the_program(tmp_path: Path) -> None:
    program = tmp_path / "simulator"
    program.write_text(
        '#!/bin/sh\nprintf "%s|" "$@"\necho "$PWD" >&2\nexit 3\n', encoding="utf-8"
    )
    program.chmod(0o755)
    wrappers = tmp_path / "bin"
    wrappers.mkdir()
    log = tmp_path / "simulations.jsonl"
    agent_run.wrap(wrappers, program, log, Path(sys.executable))
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
    prepared = agent_run.environment(Arm.NGSPICE, tmp_path)
    assert set(agent_run.PACKAGES) <= set(prepared.packages)
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
    assert prepared.log == tmp_path / "log" / agent_run.SIMULATIONS
    (logged,) = prepared.log.read_text(encoding="utf-8").splitlines()
    assert json.loads(logged)["arguments"] == ["--version"]
    # Neither jolty's Python nor its package is among what the arm reads.
    repository = BENCHMARKS.parent.resolve()
    assert not [each for each in prepared.reads if each.is_relative_to(repository)]


def test_the_jolty_arm_counts_its_command_lines_runs_and_its_analyses(
    tmp_path: Path,
) -> None:
    prepared = agent_run.environment(Arm.JOLTY, tmp_path)
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
    (logged,) = (
        (tmp_path / "log" / agent_run.SIMULATIONS)
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert json.loads(logged)["program"] == "jolty"
    # Every process of a session names the log of analyses to jolty's settings.
    assert prepared.analyses == tmp_path / "log" / agent_run.ANALYSES
    assert prepared.variables[agent_run.ANALYSIS_LOG] == str(prepared.analyses)
    with pytest.MonkeyPatch.context() as patched:
        patched.setenv(
            agent_run.ANALYSIS_LOG, prepared.variables[agent_run.ANALYSIS_LOG]
        )
        assert Settings.load().analysis_log == prepared.analyses


def test_a_session_holds_nothing_of_the_runner_s_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("ANTHROPIC_MODEL", "a model of the user's")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a key")
    monkeypatch.setenv("LANG", "en_GB.UTF-8")
    prepared = agent_run.environment(Arm.JOLTY, tmp_path)
    variables = prepared.variables
    # Its home, Claude Code's configuration in it, and its temporary directory are its
    # own, and empty; its path is the arm's then the system's.
    home = tmp_path / "home"
    assert (variables["HOME"], variables["CLAUDE_CONFIG_DIR"], variables["TMPDIR"]) == (
        str(home),
        str(home / ".claude"),
        str(tmp_path / "tmp"),
    )
    assert not list((home / ".claude").iterdir())
    assert variables["PATH"].endswith(f"{os.pathsep}{os.confstr('CS_PATH')}")
    # Of the runner's own, only the locale and the credentials remain.
    assert set(variables) == {
        "HOME",
        "CLAUDE_CONFIG_DIR",
        "TMPDIR",
        "PATH",
        "JOLTY_ANALYSIS_LOG",
        "LANG",
        "ANTHROPIC_API_KEY",
        *(name for name in ("LC_ALL", "LC_CTYPE") if name in os.environ),
    }
    assert variables["ANTHROPIC_API_KEY"] == "a key"
    assert agent_run.SET == "(set)"


def _assistant(id: str, *blocks: dict[str, object], parent: str | None = None) -> str:
    """A line of an assistant event: blocks of the model's message `id`."""
    message = {"id": id, "role": "assistant", "content": list(blocks)}
    event = {"type": "assistant", "message": message, "parent_tool_use_id": parent}
    return json.dumps(event)


def _call(id: str, name: str) -> dict[str, object]:
    return {"type": "tool_use", "id": id, "name": name, "input": {}}


def test_the_transcript_reader_counts_turns_and_tool_calls(tmp_path: Path) -> None:
    lines = [
        "a warning the agent printed",
        json.dumps(
            {
                "type": "system",
                "subtype": "init",
                "model": "claude-opus-5-5",
                "claude_code_version": "2.1.325",
                "cwd": "/work",
                "tools": ["Bash", "Write", "mcp__jolty__simulate"],
                "mcp_servers": [{"name": "jolty", "status": "connected"}],
                "permissionMode": "bypassPermissions",
                "apiKeySource": "none",
                "output_style": "default",
                "skills": [],
                "agents": ["general-purpose"],
                "slash_commands": ["compact"],
                "plugins": [],
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
                "result": "design.py meets the specs.",
                "terminal_reason": "completed",
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
    session = transcript.read(path)
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
    assert session.report == "design.py meets the specs."
    assert session.terminal_reason == "completed"
    setup = session.setup
    assert setup is not None
    assert setup.mcp_servers == {"jolty": "connected"}
    assert (setup.permission_mode, setup.api_key_source) == (
        "bypassPermissions",
        "none",
    )
    assert (setup.skills, setup.agents, setup.plugins) == ((), ("general-purpose",), ())


def test_the_transcript_reader_reads_the_setup_and_how_the_session_ended(
    tmp_path: Path,
) -> None:
    path = tmp_path / "transcript.txt"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "system",
                        "subtype": "init",
                        "tools": ["Bash"],
                        "mcp_servers": [{"name": "jolty", "status": "failed"}, {}],
                        "plugins": [{"name": "a", "path": "/p"}],
                        "permissionMode": "bypassPermissions",
                    }
                ),
                _assistant("msg_1", {"type": "text", "text": "I tried."}),
                _assistant("msg_2", _call("toolu_1", "Bash")),
                json.dumps(
                    {
                        "type": "result",
                        "subtype": "error_max_turns",
                        "is_error": True,
                        "terminal_reason": "max_turns",
                        "errors": ["Reached the maximum number of turns (3)"],
                        "num_turns": 3,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    session = transcript.read(path)
    assert session is not None
    assert session.setup == transcript.Setup(
        tools=("Bash",),
        mcp_servers={"jolty": "failed"},
        permission_mode="bypassPermissions",
        plugins=("a",),
    )
    assert (session.outcome, session.terminal_reason, session.is_error) == (
        "error_max_turns",
        "max_turns",
        True,
    )
    assert session.errors == ("Reached the maximum number of turns (3)",)
    # Without a result, the last report is the last text the model wrote.
    assert session.report == "I tried."


def test_a_session_ends_as_the_runner_saw_and_the_agent_reports() -> None:
    def reported(
        outcome: str | None, *, is_error: bool | None = False
    ) -> transcript.Session:
        return transcript.Session(
            turns=1, tool_calls={}, outcome=outcome, is_error=is_error
        )

    assert (
        compare.ending(0, reported("success"), timed_out=False)
        is compare.Ending.FINISHED
    )
    assert (
        compare.ending(-9, reported(None, is_error=None), timed_out=True)
        is compare.Ending.OUT_OF_TIME
    )
    assert (
        compare.ending(1, reported("error_max_turns", is_error=True), timed_out=False)
        is compare.Ending.OUT_OF_TURNS
    )
    assert (
        compare.ending(1, reported("error_max_budget_usd"), timed_out=False)
        is compare.Ending.OUT_OF_BUDGET
    )
    # A success that is an error, an error, a failed exit and no report all errored.
    for code, session in (
        (0, reported("success", is_error=True)),
        (1, reported("error_during_execution")),
        (1, reported("success")),
        (0, reported(None, is_error=None)),
    ):
        assert compare.ending(code, session, timed_out=False) is compare.Ending.ERRORED
    # An agent that prints no events ended as its exit code says.
    assert compare.ending(0, None, timed_out=False) is compare.Ending.FINISHED
    assert compare.ending(2, None, timed_out=False) is compare.Ending.ERRORED


def test_a_transcript_of_no_events_says_nothing_of_its_session(tmp_path: Path) -> None:
    path = tmp_path / "transcript.txt"
    path.write_text("I wrote design.py.\n", encoding="utf-8")
    assert transcript.read(path) is None
    path.write_text(_assistant("msg_1", _call("toolu_1", "Bash")), encoding="utf-8")
    session = transcript.read(path)
    assert session is not None
    assert (session.turns, session.calls, session.seconds) == (1, 1, None)


STUB_CALLS = {
    "jolty": {"Bash": 2, "Write": 2, "mcp__jolty__simulate_circuit": 1},
    "ngspice": {"Bash": 1, "Write": 2},
}
"""The stub's tool calls in each arm, by name, a turn each.

It writes design.py and the plan it simulates it with, then simulates it through each of
jolty's front ends: the command line's and the MCP server's operating points, and a
script's transient. Or it writes design.cir and tolerances.json, then runs ngspice once.
"""

STUB_COUNTED = {
    "jolty": ({"jolty simulate": 1}, 3, 3, {"OperatingPoint": 2, "Transient": 1}, 3),
    "ngspice": ({"ngspice --version": 1}, None, None, {}, 1),
}
"""What the stub's session counts in each arm: the runs of its simulator's command line
by command, its analyses, their runs and kinds, and its simulations.

jolty logs every analysis, whatever runs it; ngspice none."""


@SKIP
def test_the_comparison_runs_and_reports_both_arms_with_the_stub(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    path = compare.compare(
        compare.Agent.STUB,
        tasks=["cost_down"],
        arms=list(Arm),
        repetitions=2,
        results=tmp_path,
        work=work,
    )
    report = compare.Report.model_validate_json(path.read_text(encoding="utf-8"))
    assert path == tmp_path / f"{report.date}-stub.json"
    assert [(run.arm, run.repetition) for run in report.runs] == [
        (Arm.JOLTY, 1),
        (Arm.NGSPICE, 1),
        (Arm.JOLTY, 2),
        (Arm.NGSPICE, 2),
    ]
    assert all(run.met and run.exit_code == 0 for run in report.runs)
    assert all(Path(run.work).parent == work for run in report.runs)
    assert all(run.ending == "finished" and not run.shortfalls for run in report.runs)
    assert report.runs[0].report == "I wrote the known good design, design.py."
    for run in report.runs:
        configuration = run.configuration
        # Each session ran in a home of its own, in the sandbox wherever there is one,
        # writing its working directory and its own places alone.
        assert configuration.variables["HOME"].endswith("/home")
        assert configuration.command[1] == str(BENCHMARKS / "stub.py")
        if mechanism() is not None:
            assert configuration.sandbox is not None
            assert configuration.sandbox.writes[0] == str(Path(run.work).resolve())
    for run in report.runs:
        assert run.session is not None
        assert run.session.turns == sum(STUB_CALLS[run.arm].values())
        assert run.session.tool_calls == STUB_CALLS[run.arm]
        counted = run.simulations
        assert (
            counted.commands,
            counted.analyses,
            counted.analysis_runs,
            counted.kinds,
            counted.count,
        ) == STUB_COUNTED[run.arm]
        assert bool(run.packages) == (run.arm is Arm.NGSPICE)
        kept = tmp_path / f"{report.date}-stub" / "cost_down" / run.arm
        logs = ("transcript.txt", "simulations.jsonl", "analyses.jsonl")
        assert [(kept / str(run.repetition) / name).is_file() for name in logs] == [
            True,
            True,
            run.arm is Arm.JOLTY,
        ]
    task = report.tasks["cost_down"]
    lower, upper = wilson(2, 2)
    for arm in Arm:
        summary = task.arms[arm]
        assert (summary.endings, summary.shortfalls, summary.specs) == (
            {"finished": 2},
            {},
            {},
        )
        assert summary.success.model_dump() == {
            "met": 2,
            "runs": 2,
            "rate": 1.0,
            "lower": lower,
            "upper": upper,
        }
        assert summary.turns is not None
        assert summary.turns.median == sum(STUB_CALLS[arm].values())
        assert summary.simulations is not None
        assert summary.simulations.median == {Arm.JOLTY: 3, Arm.NGSPICE: 1}[arm]
        assert report.arms[arm] == summary.success
    difference = task.difference
    assert difference is not None
    # Newcombe's interval about a difference of 0 spans each rate's Wilson interval.
    assert (difference.success, difference.lower, difference.upper) == (
        0,
        lower - 1,
        1 - lower,
    )
    assert (difference.turns, difference.tool_calls, difference.simulations) == (
        2,
        2,
        2,
    )
    with pytest.raises(FileExistsError):
        compare.compare(
            compare.Agent.STUB,
            tasks=["cost_down"],
            arms=[Arm.JOLTY],
            repetitions=1,
            results=tmp_path,
        )


def test_the_comparison_runs_no_session_without_its_programs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="claude, ngspice, uv not on the path"):
        compare.compare(
            compare.Agent.CLAUDE,
            tasks=["cost_down"],
            arms=list(Arm),
            repetitions=1,
            results=tmp_path,
        )
    assert not list(tmp_path.iterdir())
    monkeypatch.undo()
    with pytest.raises(ValueError, match="lies in the repository"):
        compare.compare(
            compare.Agent.STUB,
            tasks=["cost_down"],
            arms=[Arm.JOLTY],
            repetitions=1,
            results=tmp_path,
            work=BENCHMARKS,
        )


def test_each_session_is_scored_in_a_process_of_its_own(tmp_path: Path) -> None:
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
        verdicts.append(agent_run.scored(COST_DOWN, work, Arm.JOLTY))
        assert (work / agent_run.VERDICT).is_file()
    assert [verdict.met for verdict in verdicts] == [True, False]
    assert [shortfall.spec for shortfall in verdicts[1].shortfalls] == ["yield"]


def test_a_scorer_that_crashes_fails_the_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As on CI, which forces colors: the scorer's traceback is read, not shown.
    monkeypatch.setenv("FORCE_COLOR", "1")
    task = tmp_path / "broken"
    task.mkdir()
    (task / "scorer.py").write_text(
        "def score(task, work, arm):\n    raise RuntimeError('a broken scorer')\n",
        encoding="utf-8",
    )
    verdict = agent_run.scored(task, tmp_path, Arm.JOLTY)
    assert not verdict.met
    assert verdict.reasons == (
        "the scorer exited with 1, having written RuntimeError: a broken scorer",
    )


def test_the_runner_hands_the_agent_its_words_after_the_dashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with pytest.raises(SystemExit) as prepared:
        agent_run.main(["--arm", "ngspice", str(COST_DOWN), "-"])
    assert prepared.value.code == 0
    work = Path(capsys.readouterr().out.strip())
    assert work.parent == tmp_path
    assert sorted(path.name for path in work.iterdir()) == [
        "bias.cir",
        "catalog.json",
        "stage.cir",
    ]
    # After the dashes, every word is the agent's as it is: one the runner reads
    # itself, one with commas, and one that holds JSON.
    words = ["--arm", "ngspice", "a, b", '{"x": [1, 2]}']
    agent = [
        sys.executable,
        "-c",
        (
            "import json, pathlib, sys; "
            "pathlib.Path('words.json').write_text(json.dumps(sys.argv[1:-1]))"
        ),
        *words,
    ]
    with pytest.raises(SystemExit) as ran:
        agent_run.main([str(COST_DOWN), "--", *agent])
    assert ran.value.code == 1
    (work,) = (path for path in tmp_path.iterdir() if (path / "words.json").is_file())
    assert json.loads((work / "words.json").read_text(encoding="utf-8")) == words
    assert (work / "bias.py").is_file()


def test_the_scorer_s_command_line_prints_and_writes_its_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "verdict.json"
    with pytest.raises(SystemExit) as scored:
        agent_score.main(
            [str(COST_DOWN), str(tmp_path), "--arm", "ngspice", "--verdict", str(path)]
        )
    assert scored.value.code == 1
    verdict = Verdict.model_validate_json(path.read_text(encoding="utf-8"))
    assert verdict.reasons == (f"there is no {tmp_path / 'design.cir'}",)
    assert Verdict.model_validate_json(capsys.readouterr().out) == verdict


def test_the_comparison_s_command_line_reads_lists_and_refuses_bad_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    asked: list[dict[str, object]] = []

    def recorded(agent: compare.Agent, **options: object) -> Path:
        asked.append({"agent": agent, **options})
        return tmp_path / "report.json"

    monkeypatch.setattr(compare, "compare", recorded)
    results = tmp_path / "results"
    compare.main(
        [
            *("--agent", "stub", "--tasks", "cost_down,balun", "--arms", "ngspice"),
            *("--repetitions", "2", "--results", str(results)),
            *("--timeout", "600", "--turns", "40", "--budget", "2.5"),
        ]
    )
    assert asked == [
        {
            "agent": compare.Agent.STUB,
            "tasks": ["cost_down", "balun"],
            "arms": [Arm.NGSPICE],
            "repetitions": 2,
            "results": results,
            "model": None,
            "work": None,
            "limits": compare.Limits(timeout=600, turns=40, budget=2.5),
        }
    ]
    assert results.is_dir()
    assert capsys.readouterr().out == f"{tmp_path / 'report.json'}\n"
    with pytest.raises(SystemExit) as refused:
        compare.main(["--repetitions", "0", "--tasks", "cost_down,nonesuch"])
    assert refused.value.code == 2
    assert capsys.readouterr().err == (
        "error: invalid arguments: --repetitions: Input should be greater than or "
        "equal to 1; --tasks: Value error, no task is named nonesuch; there are "
        f"{', '.join(compare.TASKS)}\n"
    )
    assert len(asked) == 1


PROBE = """
import json, pathlib, sys

def tried(action):
    try:
        action()
    except PermissionError:
        return "refused"
    return "done"

secret, scorer, home = (pathlib.Path(each) for each in sys.argv[1:4])
found = {
    "its own file": tried(lambda: pathlib.Path("bias.py").read_text()),
    "a file outside": tried(secret.read_text),
    "the scorer": tried(scorer.read_text),
    "the user's home": tried(lambda: list(home.iterdir())),
    "writing outside": tried(lambda: (secret.parent / "new.txt").write_text("x")),
    "writing its own": tried(lambda: pathlib.Path("new.txt").write_text("x")),
    "writing its home": tried(lambda: (pathlib.Path.home() / "x").write_text("x")),
}
pathlib.Path("probe.json").write_text(json.dumps(found))
"""
"""An agent that tries to read and write in and out of its directory, and says how
it went, in ``probe.json``."""


def sandboxed() -> pytest.MarkDecorator:
    """Skips a test where this platform has no sandbox, sandbox-exec or bwrap."""
    found = shutil.which("sandbox-exec" if sys.platform == "darwin" else "bwrap")
    return pytest.mark.skipif(found is None, reason="no sandbox on this platform")


@sandboxed()
def test_a_sandboxed_agent_reads_and_writes_its_own_directory_alone(
    tmp_path: Path,
) -> None:
    secret = tmp_path / "outside" / "secret.txt"
    secret.parent.mkdir()
    secret.write_text("the user's", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    scorer = COST_DOWN / "scorer.py"
    program = agent_run.Program(
        (sys.executable, "-c", PROBE, str(secret), str(scorer), str(Path.home()))
    )
    ran = agent_run.session(COST_DOWN, program, work, Arm.JOLTY)
    assert ran.configuration.sandbox is not None
    assert json.loads((work / "probe.json").read_text(encoding="utf-8")) == {
        "its own file": "done",
        "a file outside": "refused",
        "the scorer": "refused",
        "the user's home": "refused",
        "writing outside": "refused",
        "writing its own": "done",
        "writing its home": "done",
    }
    assert not (secret.parent / "new.txt").exists()


@sandboxed()
def test_the_scorer_runs_the_agent_s_code_confined_as_its_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a key")
    secret = tmp_path / "secret.txt"
    secret.write_text("the user's", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    # The design runs as the scorer builds it, after the agent is gone.
    (work / "design.py").write_text(
        "import json, os, pathlib\n"
        "here = pathlib.Path(__file__).with_name('environ.json')\n"
        "here.write_text(json.dumps(sorted(os.environ)))\n"
        f"pathlib.Path({str(secret)!r}).read_text()\n",
        encoding="utf-8",
    )
    verdict = agent_run.scored(COST_DOWN, work, Arm.JOLTY)
    (shortfall,) = verdict.shortfalls
    assert shortfall.kind == "unbuilt"
    assert "PermissionError" in shortfall.reason
    found = json.loads((work / "environ.json").read_text(encoding="utf-8"))
    assert "ANTHROPIC_API_KEY" not in found


def test_the_runner_stops_a_session_and_what_it_left_running(tmp_path: Path) -> None:
    def left(work: Path) -> bool:
        pid = int((work / "child.pid").read_text(encoding="utf-8"))
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    for name, script, timeout in (
        ("exits", "sleep 300 & echo $! > child.pid", None),
        ("runs on", "sleep 300 & echo $! > child.pid; sleep 300", 2.0),
    ):
        work = tmp_path / name
        work.mkdir()
        program = agent_run.Program(("/bin/sh", "-c", script), confined=False)
        ran = agent_run.session(COST_DOWN, program, work, Arm.JOLTY, timeout)
        assert ran.timed_out == (timeout is not None)
        assert ran.configuration.timeout == timeout
        assert not left(work)


def _claude(directory: Path, *, logged_in: bool) -> None:
    """Writes a stand-in for Claude Code into `directory`, that answers for free.

    It states its version, and its authentication as logged in or not, exiting with 1
    when not, as Claude Code does.
    """
    status = json.dumps(
        {"loggedIn": logged_in, "authMethod": "api_key", "apiProvider": "firstParty"}
    )
    script = (
        "#!/bin/sh\n"
        'case "$1" in\n'
        '  --version) echo "2.1.325 (Claude Code)" ;;\n'
        f"  auth) echo '{status}'; exit {0 if logged_in else 1} ;;\n"
        "esac\n"
    )
    directory.mkdir(exist_ok=True)
    (directory / "claude").write_text(script, encoding="utf-8")
    (directory / "claude").chmod(0o755)


def test_claude_runs_only_in_a_sandbox_and_holding_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    programs = tmp_path / "bin"
    _claude(programs, logged_in=True)
    monkeypatch.setenv("PATH", f"{programs}{os.pathsep}{os.environ['PATH']}")
    results = tmp_path / "results"
    results.mkdir()

    def compared() -> Path:
        return compare.compare(
            compare.Agent.CLAUDE,
            tasks=["cost_down"],
            arms=[Arm.JOLTY],
            repetitions=1,
            results=results,
        )

    # Claude Code started in the sandbox, which costs nothing, says who it is.
    agent = agent_run.default_agent(Arm.JOLTY, model="opus", turns=40, budget=2.5)
    assert agent.command[0] == str((programs / "claude").resolve())
    assert agent.command[-6:] == (
        "--model",
        "opus",
        "--max-turns",
        "40",
        "--max-budget-usd",
        "2.5",
    )
    assert "--setting-sources=" in agent.command
    if compare.mechanism() is not None:
        assert compare.started(agent) == (
            "2.1.325 (Claude Code)",
            {"authMethod": "api_key", "apiProvider": "firstParty"},
        )
        _claude(programs, logged_in=False)
        with pytest.raises(RuntimeError, match="holds no credentials"):
            compared()
    monkeypatch.setattr(compare, "mechanism", lambda: None)
    with pytest.raises(RuntimeError, match="Claude Code's sessions run only in one"):
        compared()
    assert not list(results.iterdir())


def test_bubblewrap_hides_the_private_places_but_what_a_session_is_given(
    tmp_path: Path,
) -> None:
    work, program = tmp_path / "work", tmp_path / "program"
    within = Sandbox(
        mechanism="bwrap",
        private=(str(tmp_path), str(tmp_path / "absent")),
        reads=(str(program),),
        writes=(str(work),),
    )
    # The system read-only, the private places that exist emptied, then what the
    # session reads there, read-only, and what it writes; processes of its own.
    assert bubblewrap(within, work) == [
        *("bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc"),
        *("--tmpfs", str(tmp_path)),
        *("--ro-bind", str(program), str(program)),
        *("--bind", str(work), str(work)),
        *("--unshare-pid", "--die-with-parent", "--new-session"),
        *("--chdir", str(work.resolve())),
    ]
