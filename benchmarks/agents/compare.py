# Copyright 2026 Maxence Faldor
"""Benchmark comparison harness evaluating jolty agents against ngspice-only agents.

Usage: ``uv run python benchmarks/agents/compare.py [--agent AGENT] [--model MODEL]
[--repetitions N] [--tasks TASK,...] [--arms ARM,...] [--results DIRECTORY]
[--work DIRECTORY] [--timeout SECONDS] [--turns N] [--budget USD]``.

The harness runs ``N`` sessions per arm per task (5 by default) via ``run.py``. Each
session executes in a fresh working directory under ``--work`` or the system temporary
directory, never inside the repository, using the arm's prompt, files, tools, and
hidden task scorer. Every session uses an isolated configuration without host user
settings and runs under the platform sandbox in ``sandbox.py``. Claude Code requires an
active sandbox and passes zero-cost startup and credential checks before any session
launches. Sessions enforce optional ``--timeout``, ``--turns``, and ``--budget`` limits.
Execution proceeds sequentially, interleaving arms and tasks across repetition rounds
so time-varying API behavior affects both arms equally.

Results are written to ``DIRECTORY/DATE-AGENT.json`` (defaulting to
``benchmarks/agents/results``). Per-session transcripts, simulator logs, jolty analysis
logs, and verdicts are copied into ``DIRECTORY/DATE-AGENT/TASK/ARM/REPETITION``. The
JSON report stores every session plus per-task, per-arm summaries:

- Success rate with a 95 % Wilson confidence interval matching jolty's yield method.
- Median and min-max range for model turns, tool calls, simulations, wall-clock time,
  and cost. Simulation counts track every logged jolty analysis on the jolty arm
  (counting each batch analysis once regardless of batch size) and every ngspice CLI
  invocation on the ngspice arm.
- Session ending counts (finished, timed out, out of turns, out of budget, or errored)
  and shortfall counts grouped by `verdict.Kind` and spec name, with full shortfalls
  and final agent reports preserved per session.
- Arm differences (jolty minus ngspice) for success rates with a 95 % Newcombe hybrid
  score confidence interval and for median resource metrics.

AnalogCoder's twenty-four circuits form a single task passed only when all twenty-four
pass, while retaining per-problem measures on every session.

Supported agents are ``claude`` (Claude Code via ``run.py`` with optional ``--model``)
and ``stub`` (zero-cost ``stub.py`` emitting known passing designs).
"""

import datetime as dt
import importlib.metadata
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final

from arms import Arm
from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from pydantic_settings import SettingsConfigDict
from run import (
    ANALYSES,
    CREDENTIALS,
    SIMULATIONS,
    TRANSCRIPT,
    VERDICT,
    Configuration,
    Program,
    default_agent,
    imported,
    probe,
    session,
)
from sandbox import mechanism
from transcript import Session, read
from verdict import Kind, Shortfall

from jolty import cli
from jolty.simulate import Logged
from jolty.specs import wilson
from jolty.spice import ngspice

BENCHMARKS: Final = Path(__file__).parent
"""Root directory containing one subdirectory per agent benchmark task."""
TASKS: Final = tuple(sorted(path.parent.name for path in BENCHMARKS.glob("*/task.md")))
"""Sorted tuple of available benchmark task names discovered via ``task.md``."""
STUB: Final = BENCHMARKS / "stub.py"
"""Path to the zero-cost stand-in agent script."""


class Agent(StrEnum):
    """Agent implementation executed by the comparison harness: Claude Code or stub."""

    CLAUDE = "claude"
    STUB = "stub"


class Limits(BaseModel):
    """Per-session resource caps for wall-clock seconds, model turns, and USD budget."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timeout: float | None = Field(
        default=None,
        gt=0,
        description="Maximum wall-clock duration in seconds before runner termination, "
        "or null if unlimited.",
    )
    turns: int | None = Field(
        default=None,
        ge=1,
        description="Maximum Claude Code turn count before termination, or null if "
        "unlimited.",
    )
    budget: float | None = Field(
        default=None,
        gt=0,
        description="Maximum Claude Code spend in US dollars before termination, or "
        "null if unlimited.",
    )


def command(
    agent: Agent, task: Path, arm: Arm, model: str | None, limits: Limits
) -> Program:
    """Builds the `Program` command for running `agent` of `arm` on `task`.

    The stub grants read access to the benchmark directory and Python runtime and runs
    unsandboxed if no sandbox backend exists. Claude Code strictly requires a sandbox.
    """
    if agent is Agent.STUB:
        python = Path(sys.executable)
        return Program(
            (str(python), str(STUB), str(task), str(arm)),
            reads=(BENCHMARKS, *imported(python)),
            confined=False,
        )
    return default_agent(arm, model=model, turns=limits.turns, budget=limits.budget)


class Simulations(BaseModel):
    """Simulation activity recorded for one session: CLI runs and logged analyses.

    CLI runs count wrapper-logged ``jolty`` or ``ngspice`` invocations. Logged
    analyses count every analysis executed by jolty through the CLI, MCP server, or
    Python API.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    runs: int = Field(
        description="Total wrapper-logged invocations of the arm simulator CLI."
    )
    commands: dict[str, int] = Field(
        description="Simulator CLI invocation counts keyed by program and first "
        "argument, such as 'jolty simulate' or 'ngspice -b'."
    )
    analyses: int | None = Field(
        description="Total analyses executed by jolty (counting each batch analysis "
        "once regardless of run count), or null on the ngspice arm."
    )
    analysis_runs: int | None = Field(
        description="Total individual simulation runs across logged jolty analyses "
        "(counting every run in a batch), or null on the ngspice arm."
    )
    kinds: dict[str, int] = Field(
        description="Logged jolty analysis counts keyed by analysis kind such as "
        "'Transient', or empty on the ngspice arm."
    )

    @property
    def count(self) -> int:
        """Primary simulation count for the arm: jolty analyses or ngspice CLI runs.

        Uses logged analysis counts on the jolty arm and wrapper-logged CLI
        invocations on the ngspice arm, where individual analyses inside a netlist are
        invisible to the wrapper.
        """
        return self.runs if self.analyses is None else self.analyses


def simulations(work: Path) -> Simulations:
    """Reads simulator and analysis logs from `work` into a `Simulations` summary.

    Parses one JSON object per line from the simulator log and, when present on the
    jolty arm, one `Logged` record per line from the analysis log.
    """
    log, analyses = work / SIMULATIONS, work / ANALYSES
    lines = log.read_text(encoding="utf-8").splitlines() if log.is_file() else []
    runs = [json.loads(line) for line in lines if line]
    counted = Counter(
        " ".join([str(run["program"]), *[str(each) for each in run["arguments"][:1]]])
        for run in runs
    )
    commands = dict(sorted(counted.items()))
    if not analyses.is_file():
        return Simulations(
            runs=len(runs),
            commands=commands,
            analyses=None,
            analysis_runs=None,
            kinds={},
        )
    logged = [
        Logged.model_validate_json(line)
        for line in analyses.read_text(encoding="utf-8").splitlines()
        if line
    ]
    kinds = Counter(each.analysis for each in logged)
    return Simulations(
        runs=len(runs),
        commands=commands,
        analyses=len(logged),
        analysis_runs=sum(each.runs for each in logged),
        kinds=dict(sorted(kinds.items())),
    )


class Ending(StrEnum):
    """Session termination category derived from runner status and transcript data."""

    FINISHED = "finished"
    """Agent reported a successful outcome and exited with status 0."""
    OUT_OF_TIME = "out of time"
    """Runner terminated the session after the wall-clock timeout expired."""
    OUT_OF_TURNS = "out of turns"
    """Agent reported termination upon reaching the turn limit."""
    OUT_OF_BUDGET = "out of budget"
    """Agent reported termination upon reaching the spending budget."""
    ERRORED = "errored"
    """Agent reported an error, exited non-zero, or produced no final result."""


def ending(exit_code: int, session: Session | None, *, timed_out: bool) -> Ending:
    """Determines the `Ending` classification from exit status, transcript, and timeout.

    Falls back to the process exit code when `session` is null, as with agents that do
    not emit structured JSON events.
    """
    if timed_out:
        return Ending.OUT_OF_TIME
    if session is None:
        return Ending.FINISHED if exit_code == 0 else Ending.ERRORED
    match session.outcome:
        case "error_max_turns":
            return Ending.OUT_OF_TURNS
        case "error_max_budget_usd":
            return Ending.OUT_OF_BUDGET
        case "success" if session.is_error is False and exit_code == 0:
            return Ending.FINISHED
    return Ending.ERRORED


class Run(BaseModel):
    """Complete record of one agent session on a task with verdict and telemetry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: str
    arm: Arm
    repetition: int = Field(description="One-based repetition round index.")
    work: str = Field(
        description="Path to the session working directory until external cleanup."
    )
    exit_code: int = Field(description="Process exit code returned by the agent.")
    seconds: float = Field(
        description="Wall-clock session duration in seconds measured by the runner."
    )
    ending: Ending
    met: bool = Field(
        description="Boolean indicating whether the submission passed the task scorer."
    )
    measures: dict[str, float] = Field(
        description="Named scalar measurements returned by the task scorer."
    )
    shortfalls: tuple[Shortfall, ...] = Field(
        description="Categorized shortfalls explaining every failure or unmet "
        "requirement."
    )
    report: str | None = Field(
        description="Final report text written by the agent, or null if absent."
    )
    session: Session | None = Field(
        description="Parsed session transcript summary, or null if no JSON events "
        "were emitted."
    )
    simulations: Simulations
    configuration: Configuration
    packages: dict[str, str] = Field(
        description="Python packages installed in the ngspice arm virtual "
        "environment, keyed by name, or empty for jolty."
    )

    @property
    def turns(self) -> int | None:
        """Distinct model turn count from the transcript, or null if unavailable."""
        return None if self.session is None else self.session.turns

    @property
    def calls(self) -> int | None:
        """Total tool call count from the transcript, or null if unavailable."""
        return None if self.session is None else self.session.calls

    @property
    def cost(self) -> float | None:
        """Reported session cost in US dollars, or null if unavailable."""
        return None if self.session is None else self.session.cost


class Rate(BaseModel):
    """Pass rate across sessions paired with a 95 % Wilson confidence interval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    met: int
    runs: int
    rate: float
    lower: float
    upper: float

    @classmethod
    def from_counts(cls, runs: Sequence[Run]) -> Rate:
        """Computes the pass `Rate` and 95 % Wilson interval for non-empty `runs`.

        Raises:
            ValueError: When `runs` is empty.
        """
        if not runs:
            message = "a rate is of one run or more"
            raise ValueError(message)
        met = sum(run.met for run in runs)
        lower, upper = wilson(met, len(runs))
        return cls(
            met=met, runs=len(runs), rate=met / len(runs), lower=lower, upper=upper
        )


class Spread(BaseModel):
    """Median and min-max range of a metric across sessions reporting a value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    median: float
    minimum: float
    maximum: float
    runs: int = Field(description="Number of sessions that reported a non-null value.")

    @classmethod
    def from_values(cls, values: Sequence[float | None]) -> Spread | None:
        """Computes a `Spread` over non-null entries in `values`, or returns null."""
        stated = [value for value in values if value is not None]
        if not stated:
            return None
        return cls(
            median=statistics.median(stated),
            minimum=min(stated),
            maximum=max(stated),
            runs=len(stated),
        )


class Summary(BaseModel):
    """Aggregated metrics, endings, and shortfall counts for one arm on one task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: Rate
    turns: Spread | None
    tool_calls: Spread | None
    simulations: Spread | None
    seconds: Spread | None
    cost: Spread | None
    endings: dict[Ending, int] = Field(
        description="Session counts grouped by termination classification."
    )
    shortfalls: dict[Kind, int] = Field(
        description="Session counts per shortfall kind, counting each session at most "
        "once per kind."
    )
    specs: dict[str, int] = Field(
        description="Session counts per unmet specification name."
    )

    @classmethod
    def from_runs(cls, runs: Sequence[Run]) -> Summary:
        """Aggregates a non-empty sequence of `runs` into a `Summary`."""
        return cls(
            success=Rate.from_counts(runs),
            turns=Spread.from_values([run.turns for run in runs]),
            tool_calls=Spread.from_values([run.calls for run in runs]),
            simulations=Spread.from_values([run.simulations.count for run in runs]),
            seconds=Spread.from_values([run.seconds for run in runs]),
            cost=Spread.from_values([run.cost for run in runs]),
            endings=dict(sorted(Counter(run.ending for run in runs).items())),
            shortfalls=_tally(
                {shortfall.kind for shortfall in run.shortfalls} for run in runs
            ),
            specs=_tally(
                {
                    shortfall.spec
                    for shortfall in run.shortfalls
                    if shortfall.spec is not None
                }
                for run in runs
            ),
        )


def _tally[T: str](found: Iterable[set[T]]) -> dict[T, int]:
    """Counts how many sets in `found` contain each distinct member."""
    return dict(sorted(Counter(member for each in found for member in each).items()))


class Difference(BaseModel):
    """Metric differences on a task defined as jolty minus ngspice.

    The success-rate difference carries a 95 % Newcombe hybrid score interval (method
    10) derived from the per-arm Wilson intervals (R. G. Newcombe, Statistics in
    Medicine 17, 1998). Other fields hold median differences and are null if either arm
    lacks data.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: float
    lower: float
    upper: float
    turns: float | None
    tool_calls: float | None
    simulations: float | None
    seconds: float | None
    cost: float | None

    @classmethod
    def from_rates(cls, jolty: Summary, ngspice: Summary) -> Difference:
        """Computes the `Difference` between `jolty` and `ngspice` task summaries."""
        first, second = jolty.success, ngspice.success
        difference = first.rate - second.rate
        below = math.hypot(first.rate - first.lower, second.upper - second.rate)
        above = math.hypot(first.upper - first.rate, second.rate - second.lower)

        return cls(
            success=difference,
            lower=max(-1.0, difference - below),
            upper=min(1.0, difference + above),
            turns=_less(jolty.turns, ngspice.turns),
            tool_calls=_less(jolty.tool_calls, ngspice.tool_calls),
            simulations=_less(jolty.simulations, ngspice.simulations),
            seconds=_less(jolty.seconds, ngspice.seconds),
            cost=_less(jolty.cost, ngspice.cost),
        )


def _less(mine: Spread | None, theirs: Spread | None) -> float | None:
    """Returns the median difference between `mine` and `theirs`, or null if absent."""
    if mine is None or theirs is None:
        return None
    return mine.median - theirs.median


class Task(BaseModel):
    """Per-arm summaries and optional arm comparison difference for a single task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    arms: dict[Arm, Summary]
    difference: Difference | None = Field(
        description="Metric difference (jolty minus ngspice), or null unless both "
        "arms ran."
    )


class Report(BaseModel):
    """Top-level comparison report with environment, config, summaries, and runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    date: str
    agent: Agent
    model: str | None = Field(
        description="Requested model override, or null to use the agent default."
    )
    repetitions: int
    limits: Limits
    versions: dict[str, str] = Field(
        description="Version strings for jolty, ngspice, Python, the host platform, "
        "and the sandboxed agent."
    )
    authentication: dict[str, str] = Field(
        description="Authentication metadata reported by Claude Code in the sandboxed "
        "configuration, or empty for the stub."
    )
    commands: dict[Arm, list[str]] = Field(
        description="Representative agent command tokens per arm on the first task, "
        "excluding the prompt."
    )
    tasks: dict[str, Task]
    arms: dict[Arm, Rate] = Field(
        description="Overall pass rate for each arm aggregated across all sessions."
    )
    runs: list[Run]


def versions() -> dict[str, str]:
    """Collects version strings for jolty, ngspice, Python, and the host platform."""
    found = {
        "jolty": importlib.metadata.version("jolty"),
        "python": sys.version,
        "platform": platform.platform(),
    }
    stated = ngspice.version()
    if stated is not None:
        found["ngspice"] = stated
    return found


def kept(run: Path, work: Path) -> None:
    """Copies transcript, simulator logs, analysis logs, and verdict into `run`."""
    run.mkdir(parents=True)
    for name in (TRANSCRIPT, SIMULATIONS, ANALYSES, VERDICT):
        if (work / name).is_file():
            shutil.copy(work / name, run / name)


def preflight(agent: Agent, arms: Sequence[Arm]) -> None:
    """Verifies required executables and sandbox availability before launching sessions.

    Raises:
        FileNotFoundError: When ``claude``, ``ngspice``, or ``uv`` is missing from
            ``PATH``.
        RuntimeError: When `agent` is Claude Code and no platform sandbox is available.
    """
    needed = ["claude"] if agent is Agent.CLAUDE else []
    if Arm.NGSPICE in arms:
        needed += ["ngspice", "uv"]
    missing = [name for name in needed if shutil.which(name) is None]
    if missing:
        message = f"{', '.join(missing)} not on the path, which the sessions need"
        raise FileNotFoundError(message)
    if agent is Agent.CLAUDE and mechanism() is None:
        message = (
            f"no sandbox, sandbox-exec or bwrap, is on the path of this {sys.platform} "
            "machine, and Claude Code's sessions run only in one"
        )
        raise RuntimeError(message)


def started(program: Program) -> tuple[str, dict[str, str]]:
    """Queries Claude Code version and authentication status inside the sandbox.

    Runs ``claude --version`` and ``claude auth status --json`` locally without
    invoking a model.

    Raises:
        RuntimeError: When Claude Code reports no active credentials in the sandbox.
    """
    version = probe(program, ["--version"]).strip()
    try:
        status = json.loads(probe(program, ["auth", "status", "--json"]))
    except subprocess.CalledProcessError as error:
        status = json.loads(error.stdout)
    if not status.get("loggedIn"):
        message = (
            "Claude Code holds no credentials in its sessions' configuration: set "
            f"one of {', '.join(CREDENTIALS)}"
        )
        raise RuntimeError(message)
    kept = ("authMethod", "apiProvider")
    return version, {key: str(status[key]) for key in kept if key in status}


def compare(  # noqa: PLR0913 - agent, task/arm selection, and output configuration
    agent: Agent,
    *,
    tasks: Sequence[str],
    arms: Sequence[Arm],
    repetitions: int,
    results: Path,
    model: str | None = None,
    work: Path | None = None,
    limits: Limits | None = None,
) -> Path:
    """Runs the comparison, writes the report in `results`, and returns its path.

    Creates fresh per-session working directories under `work` (defaulting to the
    system temporary directory) and applies optional resource `limits`.

    Raises:
        FileExistsError: When the dated report file or artifact directory exists.
        FileNotFoundError: When a required executable is missing from ``PATH``.
        RuntimeError: When Claude Code lacks a platform sandbox or valid credentials.
        ValueError: When `work` resides inside the repository, exposing scorers and
            prompts.
    """
    limits = limits or Limits()
    preflight(agent, arms)
    repository = BENCHMARKS.parent.resolve()
    if work is not None and work.resolve().is_relative_to(repository):
        message = f"{work} lies in the repository, {repository}"
        raise ValueError(message)
    date = dt.datetime.now(tz=dt.UTC).date().isoformat()
    path = results / f"{date}-{agent}.json"
    directory = results / f"{date}-{agent}"
    if path.exists() or directory.exists():
        message = f"{path} or {directory} exists: give another --results"
        raise FileExistsError(message)
    found = versions()
    authentication: dict[str, str] = {}
    if agent is Agent.CLAUDE:
        programs = command(agent, BENCHMARKS / tasks[0], arms[0], model, limits)
        found["agent"], authentication = started(programs)
    runs: list[Run] = []
    for repetition in range(1, repetitions + 1):
        for name in tasks:
            task = BENCHMARKS / name
            for arm in arms:
                made = tempfile.mkdtemp(prefix=f"jolty-{name}-{arm}-", dir=work)
                program = command(agent, task, arm, model, limits)
                ran = session(task, program, Path(made), arm, limits.timeout)
                kept(directory / name / str(arm) / str(repetition), Path(made))
                told = read(Path(made) / TRANSCRIPT)
                runs.append(
                    Run(
                        task=name,
                        arm=arm,
                        repetition=repetition,
                        work=made,
                        exit_code=ran.exit_code,
                        seconds=ran.seconds,
                        ending=ending(ran.exit_code, told, timed_out=ran.timed_out),
                        met=ran.verdict.met,
                        measures=ran.verdict.measures,
                        shortfalls=ran.verdict.shortfalls,
                        report=None if told is None else told.report,
                        session=told,
                        simulations=simulations(Path(made)),
                        configuration=ran.configuration,
                        packages=ran.packages,
                    )
                )
                print(f"{name}, {arm}, {repetition}: met {ran.verdict.met}")
    report = Report(
        date=date,
        agent=agent,
        model=model,
        repetitions=repetitions,
        limits=limits,
        versions=found,
        authentication=authentication,
        commands={
            arm: list(command(agent, BENCHMARKS / tasks[0], arm, model, limits).command)
            for arm in arms
        },
        tasks={
            name: _task([run for run in runs if run.task == name]) for name in tasks
        },
        arms={
            arm: Rate.from_counts([run for run in runs if run.arm == arm])
            for arm in arms
        },
        runs=runs,
    )
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def _task(runs: Sequence[Run]) -> Task:
    """Aggregates session `runs` for one task into per-arm summaries and difference."""
    summaries = {
        arm: Summary.from_runs([run for run in runs if run.arm == arm])
        for arm in Arm
        if any(run.arm == arm for run in runs)
    }
    both = Arm.JOLTY in summaries and Arm.NGSPICE in summaries
    return Task(
        arms=summaries,
        difference=Difference.from_rates(summaries[Arm.JOLTY], summaries[Arm.NGSPICE])
        if both
        else None,
    )


def _task_named(name: str) -> str:
    """Validates and returns `name` as a known benchmark task identifier.

    Raises:
        ValueError: When `name` is not listed in `TASKS`.
    """
    if name not in TASKS:
        message = f"no task is named {name}; there are {', '.join(TASKS)}"
        raise ValueError(message)
    return name


class Comparison(cli.CommandLine):
    """CLI command for running a head-to-head jolty versus ngspice agent comparison."""

    model_config = SettingsConfigDict(cli_prog_name="benchmarks/agents/compare.py")

    agent: Agent = Field(
        default=Agent.CLAUDE, description="Agent implementation executed across runs."
    )
    model: str | None = Field(
        default=None, description="Optional model identifier passed to Claude Code."
    )
    repetitions: int = Field(
        default=5, ge=1, description="Number of repetition rounds per arm per task."
    )
    tasks: list[Annotated[str, AfterValidator(_task_named)]] = Field(
        default_factory=lambda: list(TASKS),
        description="Benchmark tasks to run, defaulting to all available tasks.",
    )
    arms: list[Arm] = Field(
        default_factory=lambda: list(Arm),
        description="Benchmark arms to run, defaulting to both jolty and ngspice.",
    )
    results: Path = Field(
        default=BENCHMARKS / "results",
        description="Output directory for the JSON report and copied session files.",
    )
    work: Path | None = Field(
        default=None,
        description="Parent directory for fresh session working directories.",
    )
    timeout: float | None = Field(
        default=None,
        gt=0,
        description="Maximum wall-clock duration in seconds per session.",
    )
    turns: int | None = Field(
        default=None, ge=1, description="Maximum turn limit passed to Claude Code."
    )
    budget: float | None = Field(
        default=None,
        gt=0,
        description="Maximum spending budget in US dollars passed to Claude Code.",
    )

    def cli_cmd(self) -> None:
        """Executes the configured comparison and prints the written report path."""
        self.results.mkdir(parents=True, exist_ok=True)
        print(
            compare(
                self.agent,
                tasks=self.tasks,
                arms=self.arms,
                repetitions=self.repetitions,
                results=self.results,
                model=self.model,
                work=self.work,
                limits=Limits(
                    timeout=self.timeout, turns=self.turns, budget=self.budget
                ),
            )
        )


def main(arguments: list[str] | None = None) -> None:
    """Executes the comparison CLI using `arguments` or ``sys.argv``."""
    cli.run(Comparison, arguments)


if __name__ == "__main__":
    main()
