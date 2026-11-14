# Copyright 2026 Maxence Faldor
"""Compare agents that have jolty with agents that have ngspice alone, task by task.

Usage: ``uv run python benchmarks/compare.py [--agent AGENT] [--model MODEL]
[--repetitions N] [--tasks TASK,...] [--arms ARM,...] [--results DIRECTORY]
[--work DIRECTORY] [--timeout SECONDS] [--turns N] [--budget USD]``.

It runs ``N`` sessions of the agent per arm per task, 5 by default, each as run.py runs
one: in a fresh working directory, made in ``--work`` or the system's temporary
directory and never in the repository, with the arm's prompt, files and tools, and
scored by the task's own scorer. Each session has a configuration of its own, which
holds nothing of the user's, and runs in the platform's sandbox, ``sandbox.py``: Claude
Code runs only in one, and only once it is found to start there and to hold
credentials, which costs nothing. The runner stops a session after ``--timeout``
seconds, and Claude Code stops itself after ``--turns`` turns or once it has spent
``--budget`` US dollars, when given. The sessions run one at a time, so none slows
another, the arms and tasks taking turns in each round of repetitions, so a change in
the agent's service over the hours affects both arms alike. It writes
``DIRECTORY/DATE-AGENT.json``, ``benchmarks/results`` by default, and keeps each
session's transcript, log of simulator runs and verdict beside it, in
``DIRECTORY/DATE-AGENT/TASK/ARM/REPETITION``. The report holds every session, and for
each task and arm:

- the success rate, with its 95 % Wilson interval, as jolty states a yield's;
- the median and the range of the iterations, the model's turns and tool calls, and
  the runs of the arm's simulator, and of the session's wall time and cost;
- the failure kinds: how the sessions ended, finished, out of time, turns or budget,
  or in an error, as the runner saw and the agent reports, and how many fell short in
  each kind the scorer found, and of each spec, each session with its verdict's
  shortfalls and the agent's last report beside them;
- the difference between the arms on the task, jolty's less ngspice's: of the success
  rates, with the 95 % interval of Newcombe's hybrid score method built from the arms'
  Wilson intervals, and of the medians.

AnalogCoder's twenty-four problems are one task, met when all are; each of its sessions
keeps the scorer's measurement of every problem.

The agents are ``claude``, Claude Code as run.py runs it, on the model ``--model``
names, if any, and ``stub``, stub.py, which writes a known good design for nothing.
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
from jolty.specs import wilson

BENCHMARKS: Final = Path(__file__).parent
"""The benchmarks, a task's directory each."""
TASKS: Final = tuple(sorted(path.parent.name for path in BENCHMARKS.glob("*/task.md")))
"""Every task, by its directory's name."""
STUB: Final = BENCHMARKS / "stub.py"
"""The stand-in agent."""


class Agent(StrEnum):
    """The agents a comparison runs: Claude Code, or the stub in its place."""

    CLAUDE = "claude"
    STUB = "stub"


class Limits(BaseModel):
    """How long a session may run: in seconds, in the agent's turns, in US dollars."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timeout: float | None = Field(
        default=None,
        gt=0,
        description="The seconds after which the runner stops a session; null for "
        "none.",
    )
    turns: int | None = Field(
        default=None,
        ge=1,
        description="The turns after which Claude Code stops; null for none.",
    )
    budget: float | None = Field(
        default=None,
        gt=0,
        description="The US dollars after which Claude Code stops; null for none.",
    )


def command(
    agent: Agent, task: Path, arm: Arm, model: str | None, limits: Limits
) -> Program:
    """The program that runs `agent` of `arm` on `task`, but its prompt.

    The stub reads itself and its known good designs, in the benchmarks' directory,
    and the jolty it writes them with; it runs outside a sandbox only where there is
    none. Claude Code runs only in one.
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
    """The runs of the arm's simulator a session's log holds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runs: int = Field(description="Every run of the simulator's command line.")
    commands: dict[str, int] = Field(
        description="The runs by the program and its first argument, such as "
        "'jolty simulate' or 'ngspice -b'."
    )


def simulations(log: Path) -> Simulations:
    """The runs of the log at `log`, one JSON object a run."""
    lines = log.read_text(encoding="utf-8").splitlines() if log.is_file() else []
    runs = [json.loads(line) for line in lines if line]
    counted = Counter(
        " ".join([str(run["program"]), *[str(each) for each in run["arguments"][:1]]])
        for run in runs
    )
    return Simulations(runs=len(runs), commands=dict(sorted(counted.items())))


class Ending(StrEnum):
    """How an agent's session ended, as the runner saw it and the agent reports it."""

    FINISHED = "finished"
    """It reported its session a success, and exited with 0."""
    OUT_OF_TIME = "out of time"
    """The runner stopped it, at its timeout."""
    OUT_OF_TURNS = "out of turns"
    """It stopped at its limit of turns, as it reports."""
    OUT_OF_BUDGET = "out of budget"
    """It stopped at its budget, as it reports."""
    ERRORED = "errored"
    """It reported an error, exited with other than 0, or left no report."""


def ending(exit_code: int, session: Session | None, *, timed_out: bool) -> Ending:
    """How a session ended, by the runner's facts and the agent's own report.

    An agent that prints no events, as one other than Claude Code may not, ended as its
    exit code says.
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
    """One session of the agent on a task, and its verdict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: str
    arm: Arm
    repetition: int = Field(description="The round of repetitions, from 1.")
    work: str = Field(description="The working directory, kept until it is cleared.")
    exit_code: int = Field(description="The agent's exit code.")
    seconds: float = Field(description="The session's wall time, as the runner saw it.")
    ending: Ending
    met: bool = Field(description="Whether the work meets the task.")
    measurements: dict[str, float] = Field(description="What the scorer measured.")
    shortfalls: tuple[Shortfall, ...] = Field(
        description="Why the work falls short, if it does, each of its kind."
    )
    report: str | None = Field(
        description="The agent's last report, kept beside its verdict; null when it "
        "wrote none."
    )
    session: Session | None = Field(
        description="What the transcript says of the session; null without events."
    )
    simulations: Simulations
    configuration: Configuration
    packages: dict[str, str] = Field(
        description="The ngspice arm's Python's packages, by name; none for jolty's."
    )

    @property
    def turns(self) -> int | None:
        """The model's turns, if the transcript says."""
        return None if self.session is None else self.session.turns

    @property
    def calls(self) -> int | None:
        """The model's tool calls, if the transcript says."""
        return None if self.session is None else self.session.calls

    @property
    def cost(self) -> float | None:
        """The session's cost, if the agent reports it."""
        return None if self.session is None else self.session.cost


class Rate(BaseModel):
    """How many sessions met the task, with the 95 % Wilson interval of the rate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    met: int
    runs: int
    rate: float
    low: float
    high: float

    @classmethod
    def of(cls, runs: Sequence[Run]) -> Rate:
        """The rate of `runs` that met their task.

        Raises:
            ValueError: When there is no run.
        """
        if not runs:
            message = "a rate is of one run or more"
            raise ValueError(message)
        met = sum(run.met for run in runs)
        low, high = wilson(met, len(runs))
        return cls(met=met, runs=len(runs), rate=met / len(runs), low=low, high=high)


class Spread(BaseModel):
    """The median and the range of a quantity over the sessions that state it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    median: float
    least: float
    greatest: float
    runs: int = Field(description="The sessions that state the quantity.")

    @classmethod
    def of(cls, values: Sequence[float | None]) -> Spread | None:
        """The spread of the `values` stated, or null when none is."""
        stated = [value for value in values if value is not None]
        if not stated:
            return None
        return cls(
            median=statistics.median(stated),
            least=min(stated),
            greatest=max(stated),
            runs=len(stated),
        )


class Summary(BaseModel):
    """What an arm's sessions on a task found."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: Rate
    turns: Spread | None
    tool_calls: Spread | None
    simulations: Spread | None
    seconds: Spread | None
    cost: Spread | None
    endings: dict[Ending, int] = Field(description="The sessions by how they ended.")
    shortfalls: dict[Kind, int] = Field(
        description="The sessions that fell short in each kind, each counted once in "
        "every kind its verdict shows."
    )
    specs: dict[str, int] = Field(
        description="The sessions that fell short of each spec, by the spec's name."
    )

    @classmethod
    def of(cls, runs: Sequence[Run]) -> Summary:
        """The summary of `runs`, one or more."""
        return cls(
            success=Rate.of(runs),
            turns=Spread.of([run.turns for run in runs]),
            tool_calls=Spread.of([run.calls for run in runs]),
            simulations=Spread.of([run.simulations.runs for run in runs]),
            seconds=Spread.of([run.seconds for run in runs]),
            cost=Spread.of([run.cost for run in runs]),
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
    """How many of the sets `found` hold each of their members, by member."""
    return dict(sorted(Counter(member for each in found for member in each).items()))


class Difference(BaseModel):
    """The jolty arm's results on a task less the ngspice arm's.

    The success rates' difference has the 95 % interval of Newcombe's hybrid score
    method, his tenth, from the arms' Wilson intervals (R. G. Newcombe, "Interval
    estimation for the difference between independent proportions: comparison of eleven
    methods", Statistics in Medicine 17, 1998). The others are differences of medians,
    null where either arm states none.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: float
    low: float
    high: float
    turns: float | None
    tool_calls: float | None
    simulations: float | None
    seconds: float | None
    cost: float | None

    @classmethod
    def of(cls, jolty: Summary, ngspice: Summary) -> Difference:
        """`jolty`'s summary less `ngspice`'s."""
        first, second = jolty.success, ngspice.success
        difference = first.rate - second.rate
        below = math.hypot(first.rate - first.low, second.high - second.rate)
        above = math.hypot(first.high - first.rate, second.rate - second.low)

        return cls(
            success=difference,
            low=max(-1.0, difference - below),
            high=min(1.0, difference + above),
            turns=_less(jolty.turns, ngspice.turns),
            tool_calls=_less(jolty.tool_calls, ngspice.tool_calls),
            simulations=_less(jolty.simulations, ngspice.simulations),
            seconds=_less(jolty.seconds, ngspice.seconds),
            cost=_less(jolty.cost, ngspice.cost),
        )


def _less(mine: Spread | None, theirs: Spread | None) -> float | None:
    """`mine`'s median less `theirs`', or null when either is none."""
    if mine is None or theirs is None:
        return None
    return mine.median - theirs.median


class Task(BaseModel):
    """What the sessions on a task found, by arm, and the arms' difference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    arms: dict[Arm, Summary]
    difference: Difference | None = Field(
        description="jolty's results less ngspice's; null unless both arms ran."
    )


class Report(BaseModel):
    """A comparison: its conditions, its sessions and what they found."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    date: str
    agent: Agent
    model: str | None = Field(description="The model asked for; null for the agent's.")
    repetitions: int
    limits: Limits
    versions: dict[str, str] = Field(
        description="jolty's, ngspice's, Python's, the platform's, and the agent's, "
        "as it states it in its sandbox."
    )
    authentication: dict[str, str] = Field(
        description="How Claude Code authenticates in its sessions' configuration, as "
        "it states it there, such as authMethod api_key; none for the stub."
    )
    commands: dict[Arm, list[str]] = Field(
        description="The command that ran each arm's agent on the first task, but "
        "its prompt."
    )
    tasks: dict[str, Task]
    arms: dict[Arm, Rate] = Field(description="Each arm's rate over every session.")
    runs: list[Run]


def versions() -> dict[str, str]:
    """The versions of jolty, ngspice, Python and the platform the comparison ran on."""
    found = {
        "jolty": importlib.metadata.version("jolty"),
        "python": sys.version,
        "platform": platform.platform(),
    }
    ngspice = shutil.which("ngspice")
    if ngspice is not None:
        banner = subprocess.run(  # noqa: S603 - ngspice, found on the path
            [ngspice, "--version"], capture_output=True, text=True, check=False
        )
        found["ngspice"] = banner.stdout.strip()
    return found


def kept(run: Path, work: Path) -> None:
    """Keeps in `run` the transcript, simulator log and verdict of `work`'s session."""
    run.mkdir(parents=True)
    for name in (TRANSCRIPT, SIMULATIONS, VERDICT):
        if (work / name).is_file():
            shutil.copy(work / name, run / name)


def preflight(agent: Agent, arms: Sequence[Arm]) -> None:
    """Checks, before any session, that every program the sessions need is there.

    Raises:
        FileNotFoundError: When Claude Code, for it, or ngspice or uv, for the ngspice
            arm, is not on the path.
        RuntimeError: When Claude Code, for it, would run without a sandbox, on a
            platform that has none.
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
    """Claude Code's version and authentication, as it states them in its sandbox.

    Both cost nothing: ``claude --version``, and ``claude auth status --json``, which
    reads its configuration and credentials and calls no model.

    Raises:
        RuntimeError: When Claude Code holds no credentials there, so that every
            session would fail.
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


def compare(  # noqa: PLR0913 - the agent, what it runs on, and where results go
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
    """Runs the comparison, writes its report in `results`, and returns its path.

    Each session works in a fresh directory in `work`, the system's temporary directory
    by default, within `limits`, none by default.

    Raises:
        FileExistsError: When the report of the day's comparison of `agent` exists.
        FileNotFoundError: When a program the sessions need is not on the path.
        RuntimeError: When Claude Code would run without a sandbox, or holds no
            credentials in its sessions' configuration.
        ValueError: When `work` lies in the repository, where an agent would find its
            instructions and the scorers.
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
                        measurements=ran.verdict.measurements,
                        shortfalls=ran.verdict.shortfalls,
                        report=None if told is None else told.report,
                        session=told,
                        simulations=simulations(Path(made) / SIMULATIONS),
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
        arms={arm: Rate.of([run for run in runs if run.arm == arm]) for arm in arms},
        runs=runs,
    )
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def _task(runs: Sequence[Run]) -> Task:
    """What `runs`, the sessions on one task, found."""
    summaries = {
        arm: Summary.of([run for run in runs if run.arm == arm])
        for arm in Arm
        if any(run.arm == arm for run in runs)
    }
    both = Arm.JOLTY in summaries and Arm.NGSPICE in summaries
    return Task(
        arms=summaries,
        difference=Difference.of(summaries[Arm.JOLTY], summaries[Arm.NGSPICE])
        if both
        else None,
    )


def _task_named(name: str) -> str:
    """`name`, a task's.

    Raises:
        ValueError: When no task is named `name`.
    """
    if name not in TASKS:
        message = f"no task is named {name}; there are {', '.join(TASKS)}"
        raise ValueError(message)
    return name


class Comparison(cli.CommandLine):
    """Compare agents that have jolty with agents that have ngspice, task by task."""

    model_config = SettingsConfigDict(cli_prog_name="benchmarks/compare.py")

    agent: Agent = Field(
        default=Agent.CLAUDE, description="The agent the sessions run."
    )
    model: str | None = Field(default=None, description="The model Claude Code runs.")
    repetitions: int = Field(
        default=5, ge=1, description="The sessions per arm per task."
    )
    tasks: list[Annotated[str, AfterValidator(_task_named)]] = Field(
        default_factory=lambda: list(TASKS),
        description="The tasks, every one by default.",
    )
    arms: list[Arm] = Field(
        default_factory=lambda: list(Arm), description="The arms, both by default."
    )
    results: Path = Field(
        default=BENCHMARKS / "results", description="Where the report is written."
    )
    work: Path | None = Field(
        default=None, description="Where each session's directory is made."
    )
    timeout: float | None = Field(
        default=None, gt=0, description="The seconds after which a session stops."
    )
    turns: int | None = Field(
        default=None, ge=1, description="The turns after which Claude Code stops."
    )
    budget: float | None = Field(
        default=None,
        gt=0,
        description="The US dollars a session of Claude Code may spend.",
    )

    def cli_cmd(self) -> None:
        """Runs the comparison, and prints where its report is."""
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
    """Runs the command line, `arguments` or the process's own."""
    cli.run(Comparison, arguments)


if __name__ == "__main__":
    main()
