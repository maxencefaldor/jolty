# Copyright 2026 Maxence Faldor
"""Run one agent on a benchmark task, on its own, then score what it did.

Usage: ``uv run python benchmarks/run.py [--arm ARM] TASK [AGENT ...]``, where ``TASK``
is a task's directory, such as ``benchmarks/cost_down``, ``ARM`` the tools the agent
has, ``jolty`` by default or ``ngspice``, and ``AGENT`` the command that runs the agent,
Claude Code by default: every word the runner does not read, verbatim and in order, and
every word after ``--``, even one the runner reads, such as ``--arm``. The runner copies
the files the arm gets of the task, and only those, into a fresh working directory,
runs the agent there with the arm's prompt of the task as its last argument, keeps what
it printed in ``transcript.txt``, and scores the directory with the task's own scorer,
which the agent never sees. It prints the working directory and the verdict, and exits
with 0 when the work meets the task.

An agent of the jolty arm has jolty's command line and MCP server on its path. One of
the ngspice arm has ngspice, and a Python of its own with numpy, scipy and matplotlib
alone, made afresh for the run, and no directory of its path holds jolty's command line.
Each simulator the arm has, ``jolty`` or ``ngspice``, is a wrapper that logs every run
of it before it runs it, and the runner keeps the log beside the transcript, as
``simulations.jsonl``, one JSON object a run. In the jolty arm, every analysis jolty
runs, whatever runs it, its command line, its MCP server or a script through its Python
package, appends a line to the log jolty's setting ``analysis_log`` names, which the
runner keeps beside them, as ``analyses.jsonl``.

With ``-`` as the agent, it only prepares the working directory and prints it, for an
agent started some other way; ``score.py`` scores the directory afterwards.
"""

import contextlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from arms import Arm, given, prompt
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import CliPositionalArg, CliUnknownArgs, SettingsConfigDict
from sandbox import Sandbox, confined, mechanism, sandbox
from verdict import Kind, Shortfall, Verdict

from jolty import cli

MCP: dict[str, object] = {"mcpServers": {"jolty": {"command": "jolty-mcp"}}}
"""The MCP configuration that gives the agent jolty's server."""
COUNTED: Final = Path(__file__).with_name("counted.py")
"""The program each simulator's wrapper runs, which logs the run and then runs it."""
SIMULATIONS: Final = "simulations.jsonl"
"""The log of the runs of the arm's simulators, one JSON object a run."""
ANALYSES: Final = "analyses.jsonl"
"""The log of the analyses jolty runs in a session of the jolty arm, whatever runs them,
a `jolty.simulate.Logged` a line."""
ANALYSIS_LOG: Final = "JOLTY_ANALYSIS_LOG"
"""The variable of jolty's setting ``analysis_log``, which names the log of analyses to
every process of a session of the jolty arm."""
TRANSCRIPT: Final = "transcript.txt"
"""What the agent printed, its error stream with its output."""
VERDICT: Final = "verdict.json"
"""The scorer's verdict on the session's work."""
SCORE: Final = Path(__file__).with_name("score.py")
"""The scorer, which judges the work in a process of its own."""
PACKAGES: Final = ("numpy", "scipy", "matplotlib")
"""The packages of the ngspice arm's Python."""
CREDENTIALS: Final = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
)
"""The variables that may hold Claude Code's credentials: an API key, a token of a
gateway, or the long-lived token ``claude setup-token`` makes for a subscription."""
LOCALE: Final = ("LANG", "LC_ALL", "LC_CTYPE")
"""The variables of the runner's locale, which a session keeps."""
SET: Final = "(set)"
"""What a session's record says in place of a credential's value."""
MANAGED: Final = {
    "darwin": Path("/Library/Application Support/ClaudeCode"),
    "linux": Path("/etc/claude-code"),
}
"""Where Claude Code reads the managed settings an administrator sets, by platform,
which apply whatever else it is told."""


def claude() -> Path:
    """Claude Code's program, found on the path.

    Raises:
        FileNotFoundError: When there is none.
    """
    found = shutil.which("claude")
    if found is None:
        message = "Claude Code, claude, is not on the path"
        raise FileNotFoundError(message)
    return Path(found).resolve()


@dataclass(frozen=True, slots=True)
class Program:
    """An agent's program: its command, but its prompt, and the files it reads."""

    command: tuple[str, ...]
    reads: tuple[Path, ...] = ()
    """What the program itself reads in the private places, such as its installation,
    which its sandbox lets it read."""
    confined: bool = True
    """Whether it must run in a sandbox, and so not run where there is none."""


def default_agent(
    arm: Arm,
    *,
    model: str | None = None,
    turns: int | None = None,
    budget: float | None = None,
) -> Program:
    """Claude Code, on its own, printing its session as JSON events.

    It loads no settings file and no CLAUDE.md, ``--setting-sources=`` naming no source
    of them, and has the MCP servers of `arm` alone, jolty's or none, whatever any
    configuration gives it. ``--mcp-config`` takes every word after it up to the next
    option, so it comes first. It runs `model`, or its own default, and stops after
    `turns` turns of its own or once it has spent `budget` US dollars, when given. It
    reads its own installation, and runs only in a sandbox.
    """
    program = claude()
    servers = ["--mcp-config", json.dumps(MCP)] if arm is Arm.JOLTY else []
    options = [
        *([] if model is None else ["--model", model]),
        *([] if turns is None else ["--max-turns", str(turns)]),
        *([] if budget is None else ["--max-budget-usd", f"{budget:g}"]),
    ]
    return Program(
        (
            str(program),
            *servers,
            "--strict-mcp-config",
            "--setting-sources=",
            "--print",
            "--dangerously-skip-permissions",
            "--output-format",
            "stream-json",
            "--verbose",
            *options,
        ),
        reads=(program.parent,),
    )


def prepare(task: Path, work: Path, arm: Arm = Arm.JOLTY) -> str:
    """Copies the files `arm` gets of `task` into `work`, and returns its prompt."""
    for path in given(task, arm):
        shutil.copy(path, work / path.name)
    return prompt(task, arm)


def wrap(directory: Path, program: Path, log: Path, python: Path) -> None:
    """Writes into `directory` a wrapper of `program`, named as it, logging to `log`.

    It runs a copy of `COUNTED` in `directory` with `python`, isolated.
    """
    counted = directory / COUNTED.name
    shutil.copy(COUNTED, counted)
    command = shlex.join([str(python), "-I", str(counted), str(log), str(program)])
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


def imported(python: Path) -> list[Path]:
    """Where `python` imports from, isolated: its prefixes and every path it imports.

    Its output is read, not shown, so it writes it without colors.
    """
    probe = (
        "import json, sys; print(json.dumps([sys.prefix, sys.base_prefix, *sys.path]))"
    )
    listed = subprocess.run(  # noqa: S603 - the arm's own Python
        [str(python), "-I", "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHON_COLORS": "0"},
    )
    return [Path(each) for each in json.loads(listed.stdout) if Path(each).exists()]


@dataclass(frozen=True, slots=True)
class Environment:
    """What an agent of an arm runs with: its variables, its places and its Python.

    It reads `reads`, its programs and its directory of them, and writes `writes`, its
    own directories, which the session's sandbox lets it.
    """

    variables: dict[str, str]
    log: Path
    """The log its simulators' wrappers write."""
    reads: tuple[Path, ...]
    writes: tuple[Path, ...]
    packages: dict[str, str] = field(default_factory=dict)
    analyses: Path | None = None
    """The log of the analyses jolty runs, in the jolty arm; none in the ngspice arm."""


def environment(arm: Arm, directory: Path) -> Environment:
    """The environment of an agent of `arm`, its programs and places in `directory`.

    It holds nothing of the runner's but its locale and Claude Code's credentials: its
    home is ``directory/home``, Claude Code's configuration ``directory/home/.claude``,
    which holds nothing, and its temporary directory ``directory/tmp``. Its path holds
    the wrappers of the arm's simulators, in ``directory/bin``, which log to
    ``directory/log/simulations.jsonl``, then the arm's Python's programs, then the
    system's own, ``confstr("CS_PATH")``. In the jolty arm, ``JOLTY_ANALYSIS_LOG``
    names ``directory/log/analyses.jsonl``, which every analysis jolty runs appends to,
    whatever runs it. The ngspice arm's Python is made in ``directory/python``.

    Raises:
        FileNotFoundError: When ngspice, for the ngspice arm, is not on the path.
    """
    wrappers, logs, home, temporary = (
        directory / name for name in ("bin", "log", "home", "tmp")
    )
    for each in (wrappers, logs, home / ".claude", temporary):
        each.mkdir(parents=True)
    log = logs / SIMULATIONS
    variables = {name: os.environ[name] for name in LOCALE if name in os.environ}
    variables |= {name: os.environ[name] for name in CREDENTIALS if name in os.environ}
    variables |= {
        "HOME": str(home),
        "CLAUDE_CONFIG_DIR": str(home / ".claude"),
        "TMPDIR": str(temporary),
    }
    system = os.confstr("CS_PATH") or ""
    if arm is Arm.JOLTY:
        python = Path(sys.executable)
        wrap(wrappers, python.parent / "jolty", log, python)
        analyses = logs / ANALYSES
        variables["PATH"] = os.pathsep.join([str(wrappers), str(python.parent), system])
        variables[ANALYSIS_LOG] = str(analyses)
        return Environment(
            variables,
            log,
            (directory, *imported(python)),
            (logs, home, temporary),
            analyses=analyses,
        )
    found = shutil.which("ngspice")
    if found is None:
        message = "ngspice, the ngspice arm's simulator, is not on the path"
        raise FileNotFoundError(message)
    venv = directory / "python"
    packages = _python(venv)
    python = venv / "bin" / "python"
    wrap(wrappers, Path(found), log, python)
    simulator = Path(found).resolve().parent.parent
    variables["VIRTUAL_ENV"] = str(venv)
    variables["PATH"] = os.pathsep.join([str(wrappers), str(python.parent), system])
    return Environment(
        variables,
        log,
        (directory, *imported(python), simulator),
        (logs, home, temporary),
        packages,
    )


class Configuration(BaseModel):
    """What a session ran with, as the runner set it up, so a report can say."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command: tuple[str, ...] = Field(description="The agent's command, but its prompt.")
    variables: dict[str, str] = Field(
        description="Its whole environment, a credential's value written as (set)."
    )
    managed: dict[str, str] = Field(
        description="The managed settings Claude Code reads on this machine, which "
        "apply whatever it is told: each file and its text; none on most machines."
    )
    sandbox: Sandbox | None = Field(description="Its sandbox; null without one.")
    timeout: float | None = Field(
        description="The seconds it was given before the runner stopped it; null for "
        "as long as it needs."
    )


def managed() -> dict[str, str]:
    """The managed settings files on this machine, each with its text."""
    directory = MANAGED.get(sys.platform)
    if directory is None or not directory.is_dir():
        return {}
    return {
        str(path): path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def confinement(program: Program, prepared: Environment, work: Path) -> Sandbox | None:
    """The sandbox of `program`'s session, in `prepared`, working in `work`.

    It reads its programs, the agent's and the arm's, and writes its own directories:
    `work` and those of `prepared`.

    Raises:
        RuntimeError: When `program` must run in a sandbox and this platform has none.
    """
    if not program.confined and mechanism() is None:
        return None
    found = sandbox([*program.reads, *prepared.reads], [work, *prepared.writes])
    if found is None:
        message = (
            f"no sandbox, sandbox-exec or bwrap, is on the path of this {sys.platform} "
            f"machine, and {Path(program.command[0]).name} runs only in one"
        )
        raise RuntimeError(message)
    return found


@dataclass(frozen=True, slots=True)
class Outcome:
    """How one agent's session on a task went, as the runner saw it, and its verdict."""

    exit_code: int
    """The agent's exit code, or less the signal that stopped it."""
    seconds: float
    """The session's wall time, from the agent's start to its exit."""
    timed_out: bool
    """Whether the runner stopped it, having run out of its time."""
    verdict: Verdict
    configuration: Configuration
    packages: dict[str, str]
    """The packages of the ngspice arm's Python, by name; none for the jolty arm."""


def session(
    task: Path,
    program: Program,
    work: Path,
    arm: Arm,
    timeout: float | None = None,
) -> Outcome:
    """Runs `program` of `arm` on `task` in `work`, keeps its transcript, and scores it.

    It runs in a sandbox wherever this platform has one, and `program` runs only in
    one if it must. The runner stops it after `timeout` seconds, if given, and once it
    exits, whatever it left running. The transcript, the log of the simulators' runs,
    in the jolty arm the log of its analyses, and the verdict end in `work`.

    Raises:
        RuntimeError: When `program` must run in a sandbox and this platform has none.
    """
    text = prepare(task, work, arm)
    with tempfile.TemporaryDirectory(prefix=f"jolty-{arm}-tools-") as directory:
        prepared = environment(arm, Path(directory))
        within = confinement(program, prepared, work)
        command = [*program.command, text]
        if within is not None:
            command = confined(command, within, work)
        started = time.perf_counter()
        with (work / TRANSCRIPT).open("w", encoding="utf-8") as transcript:
            process = subprocess.Popen(  # noqa: S603 - the agent is the command given
                command,
                cwd=work,
                env=prepared.variables,
                stdout=transcript,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            timed_out = _waited(process, timeout)
        seconds = time.perf_counter() - started
        _keep(prepared.log, work / SIMULATIONS)
        if prepared.analyses is not None:
            _keep(prepared.analyses, work / ANALYSES)
    configuration = Configuration(
        command=program.command,
        variables={
            name: SET if name in CREDENTIALS else value
            for name, value in prepared.variables.items()
        },
        managed=managed(),
        sandbox=within,
        timeout=timeout,
    )
    return Outcome(
        process.returncode,
        seconds,
        timed_out,
        scored(task, work, arm),
        configuration,
        prepared.packages,
    )


def _keep(log: Path, path: Path) -> None:
    """Copies the session's `log` to `path`, or writes an empty one if it wrote none."""
    if log.is_file():
        shutil.copy(log, path)
    else:
        path.touch()


def probe(program: Program, arguments: Sequence[str]) -> str:
    """What `program`'s own program prints when given `arguments`, as in a session.

    It runs as an agent of the jolty arm runs, with its environment and in its
    sandbox, from an empty directory, so a check that costs nothing, such as of its
    version, tells what a session will find. Its output is read, not shown.

    Raises:
        RuntimeError: When `program` must run in a sandbox and this platform has none.
        subprocess.CalledProcessError: When the program exits with other than 0.
    """
    with tempfile.TemporaryDirectory(prefix="jolty-probe-") as directory:
        place = Path(directory)
        work = place / "work"
        work.mkdir()
        prepared = environment(Arm.JOLTY, place / "tools")
        within = confinement(program, prepared, work)
        command = [program.command[0], *arguments]
        finished = subprocess.run(  # noqa: S603 - the agent's own program
            command if within is None else confined(command, within, work),
            cwd=work,
            env=prepared.variables | {"PYTHON_COLORS": "0", "NO_COLOR": "1"},
            capture_output=True,
            text=True,
            check=True,
        )
    return finished.stdout


def _waited(process: subprocess.Popen[bytes], timeout: float | None) -> bool:
    """Waits for `process` to exit, for `timeout` seconds at most, then stops the rest.

    The process leads a session and a process group of its own, which a signal to the
    group ends whole: the agent when it runs out of time, and whatever it left running
    when it exits. The group is signalled while its leader's id is still held, before
    the leader is reaped, so no other process can have taken it.

    Returns whether it ran out of time.
    """
    expired = threading.Event()

    def expire() -> None:
        expired.set()
        _signal(process.pid)

    timer = None if timeout is None else threading.Timer(timeout, expire)
    if timer is not None:
        timer.start()
    os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
    if timer is not None:
        timer.cancel()
        timer.join()
    _signal(process.pid)
    process.wait()
    return expired.is_set()


def _signal(group: int) -> None:
    """Kills every process of the process group `group`, if it holds any.

    A group that holds none is no error: Linux says there is no such group, and macOS
    that it may not be signalled, when its leader alone remains, a zombie.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(group, signal.SIGKILL)


def scored(task: Path, work: Path, arm: Arm) -> Verdict:
    """The verdict of `task`'s scorer on `work`, as `arm` wrote it, written beside it.

    It scores in a process of its own, as score.py does, since the scorer runs the
    agent's scripts: a module one imports from beside it, or a change it makes to
    the path, stays in that process, and a crash ends that process alone, which fails
    the work, with the last line the scorer wrote to its error stream. That line is
    read, not shown, so Python writes it without colors, whatever the environment
    forces.

    Since it runs the agent's code, it runs as a session does: in the sandbox wherever
    there is one, reading the benchmarks, the task and jolty, and writing `work` and a
    home and temporary directory of its own, in an environment that holds nothing of
    the runner's but its locale and jolty's settings, ``JOLTY_*``. It runs from that
    empty directory, so no ``pyproject.toml`` the agent left changes jolty's settings.
    """
    task, work = task.resolve(), work.resolve()
    path = work / VERDICT
    path.unlink(missing_ok=True)
    command = [sys.executable, str(SCORE), str(task), str(work), "--arm", str(arm)]
    command += ["--verdict", str(path)]
    with tempfile.TemporaryDirectory(prefix="jolty-score-") as directory:
        place = Path(directory)
        reads = [SCORE.parent, task, *imported(Path(sys.executable))]
        within = sandbox(reads, [work, place])
        if within is not None:
            command = confined(command, within, place)
        variables = {
            name: value
            for name, value in os.environ.items()
            if name in LOCALE or name.startswith("JOLTY_")
        }
        variables |= {
            "HOME": str(place),
            "TMPDIR": str(place),
            "PATH": os.confstr("CS_PATH") or "",
            "PYTHON_COLORS": "0",
        }
        finished = subprocess.run(  # noqa: S603 - the scorer, beside the runner
            command,
            cwd=place,
            capture_output=True,
            text=True,
            check=False,
            env=variables,
        )
    if path.is_file():
        return Verdict.model_validate_json(path.read_text(encoding="utf-8"))
    last = finished.stderr.strip().splitlines()[-1:] or ["nothing"]
    reason = f"the scorer exited with {finished.returncode}, having written {last[0]}"
    return Verdict(met=False, shortfalls=(Shortfall(kind=Kind.SCORER, reason=reason),))


def run(task: Path, agent: Program, work: Path, arm: Arm = Arm.JOLTY) -> int:
    """Runs `agent` of `arm` on `task` in `work` and prints its verdict: 0 if met."""
    ran = session(task, agent, work, arm)
    print(f"worked in {work}, agent exited with {ran.exit_code}")
    print(ran.verdict.model_dump_json(indent=2))
    return 0 if ran.verdict.met else 1


class Runner(cli.CommandLine):
    """Run one agent on a benchmark task, on its own, then score what it did.

    The agent's command follows the task, or ``--``, after which every word is the
    agent's: Claude Code by default, or ``-`` to only prepare the working directory.
    """

    # The agent's command is the words the runner does not read, verbatim and in order.
    model_config = SettingsConfigDict(
        cli_prog_name="benchmarks/run.py", cli_ignore_unknown_args=True
    )

    arm: Arm = Field(default=Arm.JOLTY, description="The tools the agent has.")
    task: CliPositionalArg[Path] = Field(description="The task's directory.")
    agent: CliUnknownArgs = Field(description="The command that runs the agent.")

    def cli_cmd(self) -> None:
        """Runs the agent given, or Claude Code, on the task given.

        The agent given runs in a sandbox wherever this platform has one, reading
        there what the arm's programs read; Claude Code runs only in one.

        Raises:
            SystemExit: With 0 when the work meets the task, or once the directory is
                prepared, and 1 when it falls short.
        """
        task = self.task.resolve()
        work = Path(tempfile.mkdtemp(prefix=f"jolty-{task.name}-"))
        if self.agent == ["-"]:
            prepare(task, work, self.arm)
            print(work)
            raise SystemExit(0)
        agent = (
            Program(tuple(self.agent), confined=False)
            if self.agent
            else default_agent(self.arm)
        )
        raise SystemExit(run(task, agent, work, self.arm))


def main(arguments: list[str] | None = None) -> None:
    """Runs the command line, `arguments` or the process's own."""
    cli.run(Runner, arguments)


if __name__ == "__main__":
    main()
