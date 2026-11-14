# Copyright 2026 Maxence Faldor
"""Compare agents that have jolty with agents that have ngspice alone, task by task.

Usage: ``uv run python benchmarks/compare.py [--agent AGENT] [--model MODEL]
[--repetitions N] [--tasks TASK ...] [--arms ARM ...] [--results DIRECTORY]
[--work DIRECTORY]``.

It runs ``N`` sessions of the agent per arm per task, 5 by default, each as run.py runs
one: in a fresh working directory, made in ``--work`` or the system's temporary
directory and never in the repository, with the arm's prompt, files and tools, and
scored by the task's own scorer. The sessions run one at a time, so none
slows another, the arms and tasks taking turns in each round of repetitions, so a change
in the agent's service over the hours affects both arms alike. It writes
``DIRECTORY/DATE-AGENT.json``, ``benchmarks/results`` by default, and keeps each
session's transcript and log of simulator runs beside it, in
``DIRECTORY/DATE-AGENT/TASK/ARM/REPETITION``. The report holds every session, and for
each task and arm:

- the success rate, with its 95 % Wilson interval, as jolty states a yield's;
- the median and the range of the iterations, the model's turns and tool calls, and
  the runs of the arm's simulator, and of the session's wall time and cost;
- the difference between the arms on the task, jolty's less ngspice's: of the success
  rates, with the 95 % interval of Newcombe's hybrid score method built from the arms'
  Wilson intervals, and of the medians.

AnalogCoder's twenty-four problems are one task, met when all are; each of its sessions
keeps the scorer's measurement of every problem.

The agents are ``claude``, Claude Code as run.py runs it, on the model ``--model``
names, if any, and ``stub``, stub.py, which writes a known good design for nothing.
"""

import argparse
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
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal

from arms import Arm
from pydantic import BaseModel, ConfigDict, Field
from run import SIMULATIONS, TRANSCRIPT, default_agent, session
from transcript import Session, read

# jolty's own 95 % interval of a yield's rate, which jolty.specs keeps to itself.
from jolty.specs._requirements import _wilson as wilson

BENCHMARKS: Final = Path(__file__).parent
"""The benchmarks, a task's directory each."""
TASKS: Final = tuple(sorted(path.parent.name for path in BENCHMARKS.glob("*/task.md")))
"""Every task, by its directory's name."""
STUB: Final = BENCHMARKS / "stub.py"
"""The stand-in agent."""
type Agent = Literal["claude", "stub"]


def command(agent: Agent, task: Path, arm: Arm, model: str | None) -> list[str]:
    """The command that runs `agent` of `arm` on `task`, but its prompt."""
    if agent == "stub":
        return [sys.executable, str(STUB), str(task), str(arm)]
    return [*default_agent(arm), *([] if model is None else ["--model", model])]


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


class Run(BaseModel):
    """One session of the agent on a task, and its verdict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: str
    arm: Arm
    repetition: int = Field(description="The round of repetitions, from 1.")
    work: str = Field(description="The working directory, kept until it is cleared.")
    exit_code: int = Field(description="The agent's exit code.")
    seconds: float = Field(description="The session's wall time, as the runner saw it.")
    met: bool = Field(description="Whether the work meets the task.")
    measurements: dict[str, float] = Field(description="What the scorer measured.")
    reasons: tuple[str, ...] = Field(
        description="Why the work falls short, if it does."
    )
    session: Session | None = Field(
        description="What the transcript says of the session; null without events."
    )
    simulations: Simulations
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
        )


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
    versions: dict[str, str] = Field(
        description="jolty's, ngspice's, Python's and the platform's."
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
    """Keeps the transcript and the simulator log of the session in `work` in `run`."""
    run.mkdir(parents=True)
    for name in (TRANSCRIPT, SIMULATIONS):
        if (work / name).is_file():
            shutil.copy(work / name, run / name)


def preflight(agent: Agent, arms: Sequence[Arm]) -> None:
    """Checks, before any session, that every program the sessions need is there.

    Raises:
        FileNotFoundError: When Claude Code, for it, or ngspice or uv, for the ngspice
            arm, is not on the path.
    """
    needed = ["claude"] if agent == "claude" else []
    if Arm.NGSPICE in arms:
        needed += ["ngspice", "uv"]
    missing = [name for name in needed if shutil.which(name) is None]
    if missing:
        message = f"{', '.join(missing)} not on the path, which the sessions need"
        raise FileNotFoundError(message)


def compare(  # noqa: PLR0913 - the agent, what it runs on, and where results go
    agent: Agent,
    *,
    tasks: Sequence[str],
    arms: Sequence[Arm],
    repetitions: int,
    results: Path,
    model: str | None = None,
    work: Path | None = None,
) -> Path:
    """Runs the comparison, writes its report in `results`, and returns its path.

    Each session works in a fresh directory in `work`, the system's temporary directory
    by default.

    Raises:
        FileExistsError: When the report of the day's comparison of `agent` exists.
        FileNotFoundError: When a program the sessions need is not on the path.
        ValueError: When `work` lies in the repository, where an agent would find its
            instructions and the scorers.
    """
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
    runs: list[Run] = []
    for repetition in range(1, repetitions + 1):
        for name in tasks:
            task = BENCHMARKS / name
            for arm in arms:
                made = tempfile.mkdtemp(prefix=f"jolty-{name}-{arm}-", dir=work)
                ran = session(task, command(agent, task, arm, model), Path(made), arm)
                kept(directory / name / str(arm) / str(repetition), Path(made))
                runs.append(
                    Run(
                        task=name,
                        arm=arm,
                        repetition=repetition,
                        work=made,
                        exit_code=ran.exit_code,
                        seconds=ran.seconds,
                        met=ran.verdict.met,
                        measurements=ran.verdict.measurements,
                        reasons=ran.verdict.reasons,
                        session=read(Path(made) / TRANSCRIPT),
                        simulations=simulations(Path(made) / SIMULATIONS),
                        packages=ran.packages,
                    )
                )
                print(f"{name}, {arm}, {repetition}: met {ran.verdict.met}")
    report = Report(
        date=date,
        agent=agent,
        model=model,
        repetitions=repetitions,
        versions=versions(),
        commands={
            arm: command(agent, BENCHMARKS / tasks[0], arm, model) for arm in arms
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


class Arguments(BaseModel):
    """The comparison's arguments."""

    agent: Agent
    model: str | None
    repetitions: int = Field(ge=1)
    tasks: list[str]
    arms: list[Arm]
    results: Path
    work: Path | None


def main() -> None:
    """Runs the comparison the arguments describe, and prints where its report is."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agent", choices=["claude", "stub"], default="claude")
    parser.add_argument("--model", default=None, help="the model Claude Code runs")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument(
        "--arms", nargs="+", type=Arm, choices=list(Arm), default=list(Arm)
    )
    parser.add_argument("--results", type=Path, default=BENCHMARKS / "results")
    parser.add_argument(
        "--work", type=Path, default=None, help="where each session's directory is made"
    )
    arguments = Arguments.model_validate(vars(parser.parse_args()))
    arguments.results.mkdir(parents=True, exist_ok=True)
    print(
        compare(
            arguments.agent,
            tasks=arguments.tasks,
            arms=arguments.arms,
            repetitions=arguments.repetitions,
            results=arguments.results,
            model=arguments.model,
            work=arguments.work,
        )
    )


if __name__ == "__main__":
    main()
