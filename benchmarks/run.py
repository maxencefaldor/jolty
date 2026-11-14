# Copyright 2026 Maxence Faldor
"""Run one agent on a benchmark task, on its own, then score what it did.

Usage: ``uv run python benchmarks/run.py [--arm ARM] TASK [AGENT ...]``, where ``TASK``
is a task's directory, such as ``benchmarks/cost_down``, ``ARM`` the tools the agent
has, ``jolty`` by default or ``ngspice``, and ``AGENT`` the command that runs the agent,
Claude Code by default. The runner copies the files the arm gets of the task, and only
those, into a fresh working directory, runs the agent there with the arm's prompt of the
task as its last argument, keeps what it printed in ``transcript.txt``, and scores the
directory with the task's own scorer, which the agent never sees. It prints the working
directory and the verdict, and exits with 0 when the work meets the task.

An agent of the jolty arm has jolty's command line and MCP server on its path. One of
the ngspice arm has ngspice, and a Python of its own with numpy, scipy and matplotlib
alone, made afresh for the run, and no directory of its path holds jolty's command line.
Each simulator the arm has, ``jolty`` or ``ngspice``, is a wrapper that logs every run
of it before it runs it, and the runner keeps the log beside the transcript, as
``simulations.jsonl``, one JSON object a run.

With ``-`` as the agent, it only prepares the working directory and prints it, for an
agent started some other way; ``score.py`` scores the directory afterwards.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from arms import Arm, given, prompt
from pydantic import BaseModel
from verdict import Verdict

MCP: dict[str, object] = {"mcpServers": {"jolty": {"command": "jolty-mcp"}}}
"""The MCP configuration that gives the agent jolty's server."""
COUNTED: Final = Path(__file__).with_name("counted.py")
"""The program each simulator's wrapper runs, which logs the run and then runs it."""
SIMULATIONS: Final = "simulations.jsonl"
"""The log of the runs of the arm's simulators, one JSON object a run."""
TRANSCRIPT: Final = "transcript.txt"
"""What the agent printed, its error stream with its output."""
VERDICT: Final = "verdict.json"
"""The scorer's verdict on the session's work."""
SCORE: Final = Path(__file__).with_name("score.py")
"""The scorer, which judges the work in a process of its own."""
PACKAGES: Final = ("numpy", "scipy", "matplotlib")
"""The packages of the ngspice arm's Python."""
JOLTY: Final = ("jolty", "jolty-mcp")
"""jolty's command line and MCP server, which no directory of the ngspice arm's path
holds."""


def default_agent(arm: Arm) -> list[str]:
    """Claude Code, on its own, printing its session as JSON events.

    It has the MCP servers of `arm` alone, jolty's or none, whatever the user's
    configuration gives it. ``--mcp-config`` takes every word after it up to the next
    option, so it comes first.
    """
    servers = ["--mcp-config", json.dumps(MCP)] if arm is Arm.JOLTY else []
    return [
        "claude",
        *servers,
        "--strict-mcp-config",
        "--print",
        "--dangerously-skip-permissions",
        "--output-format",
        "stream-json",
        "--verbose",
    ]


def prepare(task: Path, work: Path, arm: Arm = Arm.JOLTY) -> str:
    """Copies the files `arm` gets of `task` into `work`, and returns its prompt."""
    for path in given(task, arm):
        shutil.copy(path, work / path.name)
    return prompt(task, arm)


def wrap(directory: Path, program: Path, log: Path) -> None:
    """Writes into `directory` a wrapper of `program`, named as it, logging to `log`."""
    command = shlex.join([sys.executable, "-I", str(COUNTED), str(log), str(program)])
    wrapper = directory / program.name
    wrapper.write_text(f'#!/bin/sh\nexec {command} "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)


def _python(directory: Path) -> dict[str, str]:
    """Makes the ngspice arm's Python in `directory`, and returns its packages.

    Its packages are `PACKAGES` and what they need, by name, each with its version.

    Raises:
        FileNotFoundError: When uv, which makes it, is not on the path.
    """
    uv = shutil.which("uv")
    if uv is None:
        message = "uv, which makes the ngspice arm's Python, is not on the path"
        raise FileNotFoundError(message)
    python = directory / "bin" / "python"
    commands = (
        [uv, "venv", "--quiet", "--python", sys.executable, str(directory)],
        [uv, "pip", "install", "--quiet", "--python", str(python), *PACKAGES],
    )
    for command in commands:
        subprocess.run(command, check=True)  # noqa: S603 - uv, found on the path
    listed = subprocess.run(  # noqa: S603 - uv, found on the path
        [uv, "pip", "list", "--format", "json", "--python", str(python)],
        check=True,
        capture_output=True,
        text=True,
    )
    return {each["name"]: each["version"] for each in json.loads(listed.stdout)}


@dataclass(frozen=True, slots=True)
class Environment:
    """What an agent of an arm runs with: its environment and its Python's packages."""

    variables: dict[str, str]
    packages: dict[str, str] = field(default_factory=dict)


def environment(arm: Arm, directory: Path) -> Environment:
    """The environment of an agent of `arm`, its programs made in `directory`.

    The wrappers of the arm's simulators lead its path, from ``directory/bin``, and log
    to ``directory/simulations.jsonl``; the ngspice arm's Python is made in
    ``directory/python``.

    Raises:
        FileNotFoundError: When ngspice, for the ngspice arm, is not on the path.
    """
    wrappers = directory / "bin"
    wrappers.mkdir()
    log = directory / SIMULATIONS
    scripts = Path(sys.executable).parent
    path = os.environ.get("PATH", "").split(os.pathsep)
    variables = dict(os.environ)
    if arm is Arm.JOLTY:
        wrap(wrappers, scripts / "jolty", log)
        variables["PATH"] = os.pathsep.join([str(wrappers), str(scripts), *path])
        return Environment(variables)
    found = shutil.which("ngspice")
    if found is None:
        message = "ngspice, the ngspice arm's simulator, is not on the path"
        raise FileNotFoundError(message)
    wrap(wrappers, Path(found), log)
    python = directory / "python"
    packages = _python(python)
    kept = [
        each
        for each in path
        if each and not any((Path(each) / name).exists() for name in JOLTY)
    ]
    for name in ("PYTHONPATH", "PYTHONHOME"):
        variables.pop(name, None)
    variables["VIRTUAL_ENV"] = str(python)
    variables["PATH"] = os.pathsep.join([str(wrappers), str(python / "bin"), *kept])
    return Environment(variables, packages)


@dataclass(frozen=True, slots=True)
class Outcome:
    """How one agent's session on a task went, as the runner saw it, and its verdict."""

    exit_code: int
    seconds: float
    """The session's wall time, from the agent's start to its exit."""
    verdict: Verdict
    packages: dict[str, str]
    """The packages of the ngspice arm's Python, by name; none for the jolty arm."""


def session(task: Path, agent: list[str], work: Path, arm: Arm) -> Outcome:
    """Runs `agent` of `arm` on `task` in `work`, keeps its transcript, and scores it.

    The transcript, the log of the simulators' runs and the verdict end in `work`.
    """
    text = prepare(task, work, arm)
    with tempfile.TemporaryDirectory(prefix=f"jolty-{arm}-tools-") as directory:
        tools = Path(directory)
        prepared = environment(arm, tools)
        started = time.perf_counter()
        with (work / TRANSCRIPT).open("w", encoding="utf-8") as transcript:
            finished = subprocess.run(  # noqa: S603 - the agent is the command given
                [*agent, text],
                cwd=work,
                env=prepared.variables,
                stdout=transcript,
                stderr=subprocess.STDOUT,
                check=False,
            )
        seconds = time.perf_counter() - started
        log = tools / SIMULATIONS
        if log.is_file():
            shutil.copy(log, work / SIMULATIONS)
        else:
            (work / SIMULATIONS).touch()
    return Outcome(
        finished.returncode, seconds, scored(task, work, arm), prepared.packages
    )


def scored(task: Path, work: Path, arm: Arm) -> Verdict:
    """The verdict of `task`'s scorer on `work`, as `arm` wrote it, written beside it.

    It scores in a process of its own, as score.py does, since the scorer runs the
    agent's scripts: a module one imports from beside it, or a change it makes to
    the path, stays in that process, and a crash ends that process alone, which fails
    the work, with the last line the scorer wrote to its error stream.
    """
    path = work / VERDICT
    path.unlink(missing_ok=True)
    command = [sys.executable, str(SCORE), str(task), str(work), "--arm", str(arm)]
    finished = subprocess.run(  # noqa: S603 - the scorer, beside the runner
        [*command, "--verdict", str(path)], capture_output=True, text=True, check=False
    )
    if path.is_file():
        return Verdict.model_validate_json(path.read_text(encoding="utf-8"))
    last = finished.stderr.strip().splitlines()[-1:] or ["nothing"]
    reason = f"the scorer exited with {finished.returncode}, having written {last[0]}"
    return Verdict(met=False, reasons=(reason,))


def run(task: Path, agent: list[str], work: Path, arm: Arm = Arm.JOLTY) -> int:
    """Runs `agent` of `arm` on `task` in `work` and prints its verdict: 0 if met."""
    ran = session(task, agent, work, arm)
    print(f"worked in {work}, agent exited with {ran.exit_code}")
    print(ran.verdict.model_dump_json(indent=2))
    return 0 if ran.verdict.met else 1


class Arguments(BaseModel):
    """The runner's arguments."""

    arm: Arm
    task: Path
    agent: list[str]


def main() -> None:
    """Runs the agent given, or Claude Code, on the task given."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arm", type=Arm, choices=list(Arm), default=Arm.JOLTY, help="its tools"
    )
    parser.add_argument("task", type=Path, help="the task's directory")
    parser.add_argument("agent", nargs=argparse.REMAINDER, help="the agent's command")
    arguments = Arguments.model_validate(vars(parser.parse_args()))
    task = arguments.task.resolve()
    agent = arguments.agent or default_agent(arguments.arm)
    work = Path(tempfile.mkdtemp(prefix=f"jolty-{task.name}-"))
    if agent == ["-"]:
        prepare(task, work, arguments.arm)
        print(work)
        raise SystemExit(0)
    raise SystemExit(run(task, agent, work, arguments.arm))


if __name__ == "__main__":
    main()
