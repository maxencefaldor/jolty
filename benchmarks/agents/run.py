# Copyright 2026 Maxence Faldor
"""Isolated single-session benchmark runner that executes and scores one agent.

Usage: ``uv run python benchmarks/agents/run.py [--arm ARM] TASK [AGENT ...]``. The
``TASK`` path names a task directory such as ``benchmarks/agents/cost_down``, and
``ARM`` selects ``jolty`` (default) or ``ngspice``. Remaining unrecognized arguments
(plus all tokens after ``--``) specify the agent command, defaulting to Claude Code.
The runner copies only the arm's given files into a fresh working directory and invokes
the agent with the rendered prompt as the final argument. Combined stdout and stderr
are saved to ``transcript.txt``, and the hidden task scorer evaluates the resulting
directory. Output includes the working directory and JSON verdict, with exit status 0
on success.

On the jolty arm, the agent receives the jolty CLI and MCP server on ``PATH``. On the
ngspice arm, the agent receives ngspice plus an isolated Python environment with numpy,
scipy, and matplotlib and no jolty binary on ``PATH``. Shell wrappers for ``jolty`` and
``ngspice`` log every CLI invocation to ``simulations.jsonl`` as one JSON object per
line alongside the transcript. On the jolty arm, jolty also appends every executed
analysis (from CLI, MCP server, or Python API) to ``analyses.jsonl`` via the
``analysis_log`` setting.

Passing ``-`` as the agent command prepares the working directory and prints the path
without launching an agent, allowing later scoring with ``score.py``.
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

from arms import Arm, prompt, staged
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import CliPositionalArg, CliUnknownArgs, SettingsConfigDict
from sandbox import Sandbox, confined, mechanism, sandbox
from verdict import Kind, Shortfall, Verdict

from jolty import cli

MCP: dict[str, object] = {"mcpServers": {"jolty": {"command": "jolty-mcp"}}}
"""MCP server configuration exposing the jolty server to the agent."""
COUNTED: Final = Path(__file__).with_name("counted.py")
"""Path to the logging wrapper script invoked by simulator shims."""
SIMULATIONS: Final = "simulations.jsonl"
"""Filename of the JSONL log recording simulator CLI invocations."""
ANALYSES: Final = "analyses.jsonl"
"""Filename of the JSONL log recording jolty analyses (`Logged` object per line)."""
ANALYSIS_LOG: Final = "JOLTY_ANALYSIS_LOG"
"""Environment variable setting jolty's ``analysis_log`` path for session processes."""
TRANSCRIPT: Final = "transcript.txt"
"""Filename storing combined agent stdout and stderr output."""
VERDICT: Final = "verdict.json"
"""Filename storing the JSON scoring verdict."""
SCORE: Final = Path(__file__).with_name("score.py")
"""Path to the standalone scorer script executed in a separate process."""
PACKAGES: Final = ("numpy", "scipy", "matplotlib")
"""Python packages installed into the isolated ngspice arm virtual environment."""
CREDENTIALS: Final = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
)
"""Environment variables checked for Claude Code API, gateway, or OAuth tokens."""
LOCALE: Final = ("LANG", "LC_ALL", "LC_CTYPE")
"""Locale environment variables forwarded from the host into agent sessions."""
SET: Final = "(set)"
"""Redacted placeholder recorded in configuration dumps for set credential variables."""
MANAGED: Final = {
    "darwin": Path("/Library/Application Support/ClaudeCode"),
    "linux": Path("/etc/claude-code"),
}
"""Per-platform directories where Claude Code reads administrator-managed settings."""


def claude() -> Path:
    """Locates and returns the resolved ``claude`` executable path from ``PATH``.

    Raises:
        FileNotFoundError: When ``claude`` is not found on ``PATH``.
    """
    found = shutil.which("claude")
    if found is None:
        message = "Claude Code, claude, is not on the path"
        raise FileNotFoundError(message)
    return Path(found).resolve()


@dataclass(frozen=True, slots=True)
class Program:
    """Agent executable spec, extra sandbox read paths, and confinement policy."""

    command: tuple[str, ...]
    reads: tuple[Path, ...] = ()
    """Additional paths inside private roots readable by the agent, such as
    installation directories."""
    confined: bool = True
    """Boolean requiring sandbox confinement and failing without a sandbox backend."""


def default_agent(
    arm: Arm,
    *,
    model: str | None = None,
    turns: int | None = None,
    budget: float | None = None,
) -> Program:
    """Builds the default isolated Claude Code `Program` for JSON event output.

    Disables user settings and ``CLAUDE.md`` via ``--setting-sources=`` and restricts
    MCP servers strictly to `arm` (jolty or none). Places ``--mcp-config`` first
    because that flag consumes tokens until the next option. Applies optional `model`,
    `turns`, and `budget` limits, grants read access to the installation directory,
    and requires sandboxing.
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
    """Stages the files for `arm` on `task` into `work` and returns the prompt."""
    for name, source in staged(task, arm):
        shutil.copy(source, work / name)
    return prompt(task, arm)


def wrap(directory: Path, program: Path, log: Path, python: Path) -> None:
    """Creates an executable logging shim for `program` in `directory` writing to `log`.

    The shim executes a local copy of `COUNTED` under `python -I`.
    """
    counted = directory / COUNTED.name
    shutil.copy(COUNTED, counted)
    command = shlex.join([str(python), "-I", str(counted), str(log), str(program)])
    wrapper = directory / program.name
    wrapper.write_text(f'#!/bin/sh\nexec {command} "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)


def _python(directory: Path) -> dict[str, str]:
    """Creates the ngspice arm virtual environment in `directory` and returns packages.

    Returns a mapping of package names to versions covering `PACKAGES` and
    dependencies.

    Raises:
        FileNotFoundError: When ``uv`` is not found on ``PATH``.
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
    """Returns existing import prefixes and ``sys.path`` entries for isolated `python`.

    Disables ANSI color output because stdout is parsed programmatically.
    """
    probe = (
        "import json, sys; print(json.dumps([sys.prefix, sys.base_prefix, *sys.path]))"
    )
    listed = subprocess.run(  # noqa: S603 - arm Python interpreter
        [str(python), "-I", "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHON_COLORS": "0"},
    )
    return [Path(each) for each in json.loads(listed.stdout) if Path(each).exists()]


@dataclass(frozen=True, slots=True)
class Environment:
    """Prepared runtime environment for an agent arm, including variables and paths.

    Grants sandbox read access to `reads` (executables and libraries) and write access
    to `writes` (temporary session directories).
    """

    variables: dict[str, str]
    log: Path
    """Path to the JSONL log written by simulator CLI wrappers."""
    reads: tuple[Path, ...]
    writes: tuple[Path, ...]
    packages: dict[str, str] = field(default_factory=dict)
    analyses: Path | None = None
    """Path to the jolty analysis JSONL log on the jolty arm, or null for ngspice."""


def environment(arm: Arm, directory: Path) -> Environment:
    """Constructs an isolated `Environment` for `arm` rooted in `directory`.

    Forwards only host locale variables and Claude Code credentials while assigning
    fresh ``HOME`` (``directory/home``), empty ``CLAUDE_CONFIG_DIR``, and ``TMPDIR``
    paths. Prepends ``directory/bin`` simulator wrappers (logging to
    ``directory/log/simulations.jsonl``) and arm Python binaries ahead of
    ``confstr("CS_PATH")``. On the jolty arm, sets ``JOLTY_ANALYSIS_LOG`` to
    ``directory/log/analyses.jsonl``. On the ngspice arm, builds a fresh virtual
    environment in ``directory/python``.

    Raises:
        FileNotFoundError: When ``ngspice`` is not found on ``PATH`` for ngspice.
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
    """Recorded runner configuration captured alongside a session report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command: tuple[str, ...] = Field(
        description="Agent command tokens excluding the final task prompt."
    )
    variables: dict[str, str] = Field(
        description="Complete session environment with credential values redacted as "
        "(set)."
    )
    managed: dict[str, str] = Field(
        description="Administrator-managed Claude Code settings files found on this "
        "host, mapped from path to contents."
    )
    sandbox: Sandbox | None = Field(
        description="Applied sandbox specification, or null if unsandboxed."
    )
    timeout: float | None = Field(
        description="Maximum session wall-clock time in seconds, or null if unlimited."
    )


def managed() -> dict[str, str]:
    """Reads administrator-managed Claude Code settings files on the current host."""
    directory = MANAGED.get(sys.platform)
    if directory is None or not directory.is_dir():
        return {}
    return {
        str(path): path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def confinement(program: Program, prepared: Environment, work: Path) -> Sandbox | None:
    """Builds the `Sandbox` for running `program` with `prepared` inside `work`.

    Combines read paths from `program` and `prepared`, and allows writes only to
    `work` and the writable directories in `prepared`.

    Raises:
        RuntimeError: When `program` requires sandboxing and no backend exists.
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
class TaskResult:
    """Observed execution outcome, timing, config, and verdict for one session."""

    exit_code: int
    """Agent exit code, or negated signal number if terminated by a signal."""
    seconds: float
    """Measured session wall-clock duration in seconds from start to exit."""
    timed_out: bool
    """Boolean indicating whether the runner killed the session on timeout."""
    verdict: Verdict
    configuration: Configuration
    packages: dict[str, str]
    """Installed Python package versions for the ngspice arm, or empty for jolty."""


def session(
    task: Path,
    program: Program,
    work: Path,
    arm: Arm,
    timeout: float | None = None,
) -> TaskResult:
    """Executes `program` for `arm` on `task` inside `work`, logs output, and scores it.

    Applies platform sandboxing whenever available and fails if `program` requires a
    sandbox on an unsupported host. Enforces the optional `timeout` and terminates any
    remaining child processes after exit. Saves the transcript, simulator log,
    optional jolty analysis log, and verdict inside `work`.

    Raises:
        RuntimeError: When `program` requires sandboxing and no backend exists.
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
    return TaskResult(
        process.returncode,
        seconds,
        timed_out,
        scored(task, work, arm),
        configuration,
        prepared.packages,
    )


def _keep(log: Path, path: Path) -> None:
    """Copies session `log` to `path`, creating an empty file if `log` is absent."""
    if log.is_file():
        shutil.copy(log, path)
    else:
        path.touch()


def probe(program: Program, arguments: Sequence[str]) -> str:
    """Runs the executable of `program` with `arguments` in a clean sandbox.

    Uses the jolty arm environment and sandbox inside an empty directory to run
    zero-cost preflight checks such as version queries with ANSI colors disabled.

    Raises:
        RuntimeError: When `program` requires sandboxing and no backend exists.
        subprocess.CalledProcessError: When the probed executable exits non-zero.
    """
    with tempfile.TemporaryDirectory(prefix="jolty-probe-") as directory:
        place = Path(directory)
        work = place / "work"
        work.mkdir()
        prepared = environment(Arm.JOLTY, place / "tools")
        within = confinement(program, prepared, work)
        command = [program.command[0], *arguments]
        finished = subprocess.run(  # noqa: S603 - probed agent executable
            command if within is None else confined(command, within, work),
            cwd=work,
            env=prepared.variables | {"PYTHON_COLORS": "0", "NO_COLOR": "1"},
            capture_output=True,
            text=True,
            check=True,
        )
    return finished.stdout


def _waited(process: subprocess.Popen[bytes], timeout: float | None) -> bool:
    """Monitors `process` until exit or `timeout`, then kills remaining group members.

    Signaling the dedicated process group led by `process` terminates both a timed-out
    agent and any background children left after exit. Waiting with ``WNOWAIT`` holds
    the leader PID until the signal completes so no other process can reuse that PID.
    Returns whether the timeout expired.
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
    """Sends ``SIGKILL`` to every process in process group `group` if any remain.

    Ignores ``ProcessLookupError`` (Linux) and ``PermissionError`` (macOS) raised when
    only a zombie group leader remains.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(group, signal.SIGKILL)


def scored(task: Path, work: Path, arm: Arm) -> Verdict:
    """Scores `work` for `task` and `arm` in an isolated subprocess and writes verdict.

    Subprocess execution isolates module imports, ``sys.path`` edits, and crashes
    inside agent scripts. Crashes produce a `Kind.SCORER` shortfall quoting the last
    stderr line, with Python colors disabled for clean parsing.

    Scoring runs inside the platform sandbox when available, granting read access to
    the benchmarks, task, and Python runtime and write access only to `work` and a
    temporary directory. Forwarded host variables include only locale settings and
    ``JOLTY_*`` options, and running from the empty temporary directory prevents any
    agent ``pyproject.toml`` from altering jolty settings.
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
        finished = subprocess.run(  # noqa: S603 - scorer script alongside runner
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
    """Executes `agent` for `arm` on `task`, prints the verdict, and returns status."""
    ran = session(task, agent, work, arm)
    print(f"worked in {work}, agent exited with {ran.exit_code}")
    print(ran.verdict.model_dump_json(indent=2))
    return 0 if ran.verdict.met else 1


class Runner(cli.CommandLine):
    """CLI command for executing one agent session on a benchmark task and scoring it.

    Trailing arguments after the task path (or after ``--``) define the agent command,
    defaulting to Claude Code, or ``-`` to stage the working directory only.
    """

    # Unrecognized trailing arguments are forwarded verbatim as the agent command.
    model_config = SettingsConfigDict(
        cli_prog_name="benchmarks/agents/run.py", cli_ignore_unknown_args=True
    )

    arm: Arm = Field(
        default=Arm.JOLTY,
        description="Benchmark arm specifying the tools available to the agent.",
    )
    task: CliPositionalArg[Path] = Field(
        description="Path to the benchmark task directory."
    )
    agent: CliUnknownArgs = Field(
        description="Agent executable and arguments, or '-' to stage files only."
    )

    def cli_cmd(self) -> None:
        """Executes the configured agent or default Claude Code on the configured task.

        Custom agent commands run sandboxed whenever the host provides a sandbox
        backend, whereas Claude Code strictly requires sandboxing.

        Raises:
            SystemExit: Exits with 0 on passing verdict or staging-only mode, and 1
                otherwise.
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
    """Executes the runner CLI using `arguments` or ``sys.argv``."""
    cli.run(Runner, arguments)


if __name__ == "__main__":
    main()
