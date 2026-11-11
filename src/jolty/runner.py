# Copyright 2026 Maxence Faldor
"""Run scripts in a process of their own.

The service starts ``python -P -m jolty.runner DIRECTORY`` for every command that runs
scripts. The runner reads the [`Job`][jolty.runner.Job] in ``DIRECTORY/job.json``. It
calls the script's ``build()`` and elaborates the circuit it returns, then, when the job
names a plan, calls the plan's ``plan()`` and runs each analysis, then evaluates the
requirements its ``specs()`` returns, when it defines one. It writes the
[`Outcome`][jolty.runner.Outcome], what was built or why it failed, to
``DIRECTORY/outcome.json``, and the results of the analyses to
``DIRECTORY/results.npz``.

Scripts and analyses therefore run outside the command line and the MCP server, whose
standard output may be a protocol channel: what a script prints is captured, a timeout
stops a script or an analysis that runs too long, and a crash ends only the runner.
"""

import runpy
import sys
import sysconfig
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Final, Literal, override

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from jolty import checks, files, ir, report
from jolty.builder import Component, circuit
from jolty.errors import ElaborationError, JoltyError, ScriptError
from jolty.plan import Plan
from jolty.results import Result
from jolty.simulate import Simulation
from jolty.specs import Specs, Verdict, evaluate

JOB: Final = "job.json"
"""The file in the job's directory that holds the job."""

OUTCOME: Final = "outcome.json"
"""The file in the job's directory where the runner writes what the job produced."""

RESULTS: Final = "results.npz"
"""The file in the job's directory where the runner writes the analyses' results."""

_LIBRARIES: Final = tuple(
    {
        Path(__file__).resolve().parent,
        *(
            Path(sysconfig.get_path(name)).resolve()
            for name in ("stdlib", "platstdlib", "purelib", "platlib")
        ),
    }
)
"""Where code that is not a script lives: jolty, the standard library, packages."""


class Job(BaseModel):
    """What the runner runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    script: Path = Field(description="The script that defines build().")
    plan: Path | None = Field(
        default=None,
        description="The script that defines plan(); null to elaborate only.",
    )
    listed: int = Field(ge=1, description="Unknowns each report lists.")


class Built(BaseModel):
    """A circuit, whether the engine elaborates it, and the reports of its analyses."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["built"] = "built"
    circuit: ir.Circuit
    devices: int | None = Field(
        default=None,
        description="Devices with every component instance unfolded; null when the "
        "circuit does not elaborate.",
    )
    unknowns: int | None = Field(
        default=None,
        description="Unknowns the engine solves for; null when the circuit does not "
        "elaborate.",
    )
    problems: tuple[str, ...] = Field(
        default=(), description="Why the circuit does not elaborate."
    )
    diagnostics: tuple[checks.Diagnostic, ...] = Field(
        default=(),
        description="The structural checks' diagnostics, at their default levels.",
    )
    analyses: tuple[report.AnalysisReport, ...] = Field(
        default=(), description="A report per analysis of the plan, in its order."
    )
    specs: tuple[Verdict, ...] = Field(
        default=(),
        description="The verdict on each requirement of the plan's specs, in order; "
        "empty when the plan defines no specs().",
    )


class Failure(BaseModel):
    """Why a script produced nothing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["failure"] = "failure"
    error: str = Field(description="The exception's type, such as 'BuildError'.")
    message: str = Field(description="The exception's message.")
    frames: tuple[ir.Source, ...] = Field(
        default=(), description="The script lines of the traceback, innermost first."
    )

    @override
    def __str__(self) -> str:
        """The error and its message, after the script line that raised it."""
        head = f"{self.error}: {self.message}" if self.message else self.error
        if not self.frames:
            return head
        innermost, *callers = (str(frame) for frame in self.frames)
        return f"{innermost}: {head}" + "".join(
            f"\n  called from {caller}" for caller in callers
        )


type Outcome = Annotated[Built | Failure, Field(discriminator="kind")]
"""What a job produced: a circuit, or why the script failed."""

OUTCOMES: Final = TypeAdapter[Outcome](Outcome)
"""Reads and writes outcomes as JSON."""


def run(job: Job, directory: Path) -> Outcome:
    """Runs `job` in this process, writing the results of its analyses to `directory`.

    The directories of the scripts come first on the import path while they run, so a
    script imports the modules beside it, as under ``python script.py``.
    """
    try:
        with _importable(job.script, job.plan):
            return _run(job, directory)
    except SystemExit as error:
        return _failure(error, f"the script exited with {error.code!r}")
    except BaseException as error:  # noqa: BLE001 - a script may raise anything
        return _failure(error, str(error))


def _run(job: Job, directory: Path) -> Built:
    namespaces: dict[Path, dict[str, object]] = {}
    top = _call(job.script, "build", namespaces)
    match top:
        case Component():
            built = circuit(top)
        case ir.Circuit():
            built = top
        case _:
            message = (
                f"build() returned {type(top).__name__}, not a Component or a Circuit"
            )
            raise ScriptError(message)
    try:
        simulation = Simulation(built)
    except ElaborationError as error:
        return Built(circuit=built, problems=error.problems)
    elaborated = Built(
        circuit=built,
        devices=simulation.devices,
        unknowns=len(simulation.unknowns),
        diagnostics=simulation.check(),
    )
    if job.plan is None:
        return elaborated
    plan = _call(job.plan, "plan", namespaces)
    if not isinstance(plan, Plan):
        message = f"plan() returned {type(plan).__name__}, not a Plan"
        raise ScriptError(message)
    solved, reports = _analyses(simulation, plan, job.listed)
    if solved:
        files.save_results(directory / RESULTS, solved)
    match _call(job.plan, "specs", namespaces, required=False):
        case None:
            verdicts: tuple[Verdict, ...] = ()
        case Specs() as specs:
            verdicts = evaluate(specs, solved, plan.analyses)
        case other:
            message = f"specs() returned {type(other).__name__}, not Specs"
            raise ScriptError(message)
    return elaborated.model_copy(update={"analyses": reports, "specs": verdicts})


def _analyses(
    simulation: Simulation, plan: Plan, listed: int
) -> tuple[dict[str, Result], tuple[report.AnalysisReport, ...]]:
    """The results of the analyses of `plan` that end, by name, and a report on each."""
    solved: dict[str, Result] = {}
    reports: list[report.AnalysisReport] = []
    for name, analysis in plan.analyses.items():
        try:
            result = simulation.run(analysis, plan.options)
        except JoltyError as error:
            reports.append(report.FailureReport(name=name, error=str(error)))
        else:
            solved[name] = result
            reports.append(report.report(name, result, listed))
    return solved, tuple(reports)


def _call(
    script: Path,
    name: str,
    namespaces: dict[Path, dict[str, object]],
    *,
    required: bool = True,
) -> object:
    """Calls the function `name` of `script`, which runs once however often it is used.

    A script that defines no function `name` returns None when it is not `required`.

    Raises:
        ScriptError: When the script defines no function `name` and it is required.
    """
    if script not in namespaces:
        namespaces[script] = runpy.run_path(str(script), run_name="__jolty__")
    function = namespaces[script].get(name)
    if callable(function):
        return function()
    if not required:
        return None
    message = f"{_relative(str(script))} defines no {name}() function"
    raise ScriptError(message)


@contextmanager
def _importable(*scripts: Path | None) -> Generator[None]:
    directories = list(
        dict.fromkeys(str(script.parent.resolve()) for script in scripts if script)
    )
    sys.path[:0] = directories
    try:
        yield
    finally:
        for directory in directories:
            sys.path.remove(directory)


def _failure(error: BaseException, message: str) -> Failure:
    """Describes `error` by the script lines it passed through, innermost first."""
    frames = [
        ir.Source(
            file=_relative(frame.f_code.co_filename),
            line=line or 1,
            function=frame.f_code.co_name,
        )
        for frame, line in traceback.walk_tb(error.__traceback__)
        if _is_script(frame.f_code.co_filename)
    ]
    frames.reverse()
    return Failure(error=type(error).__name__, message=message, frames=tuple(frames))


def _is_script(filename: str) -> bool:
    if filename.startswith("<"):
        return False
    path = Path(filename).resolve()
    return not any(path.is_relative_to(library) for library in _LIBRARIES)


def _relative(filename: str) -> str:
    path = Path(filename).resolve()
    return (
        path.relative_to(Path.cwd()).as_posix()
        if path.is_relative_to(Path.cwd())
        else path.as_posix()
    )


def main(argv: list[str] | None = None) -> None:
    """Runs the job in the directory named by the only argument."""
    (argument,) = sys.argv[1:] if argv is None else argv
    directory = Path(argument)
    job = Job.model_validate_json((directory / JOB).read_bytes())
    (directory / OUTCOME).write_bytes(OUTCOMES.dump_json(run(job, directory)))


if __name__ == "__main__":
    main()
