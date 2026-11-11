# Copyright 2026 Maxence Faldor
"""Run scripts in a process of their own.

The service starts ``python -P -m jolty.runner DIRECTORY`` for every command that runs
scripts. The runner reads the [`Job`][jolty.runner.Job] in ``DIRECTORY/job.json``. It
calls the script's ``build()`` and elaborates the circuit it returns, then, when the job
names a plan, calls the plan's ``plan()`` and runs each analysis, at the plan's own
conditions then at each of its corners, building the circuit again with a corner's
arguments. Every run takes the models the plan's fidelity chooses, characterized once
each, and spot checks run again with sampled instances at their circuit's fidelity. It
evaluates the requirements its ``specs()`` returns, when it defines one, on every run,
then on each sample of the plan's Monte Carlo analysis. It writes the
[`Outcome`][jolty.runner.Outcome], what was built or why it failed, to
``DIRECTORY/outcome.json``, and the results of the analyses at the plan's
own conditions to ``DIRECTORY/results.npz``.

Scripts and analyses therefore run outside the command line and the MCP server, whose
standard output may be a protocol channel: what a script prints is captured, a timeout
stops a script or an analysis that runs too long, and a crash ends only the runner.
"""

import runpy
import sys
import sysconfig
import traceback
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Final, Literal, override

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from jolty import checks, fidelity, files, ir, report, variation
from jolty.builder import Component, circuit
from jolty.errors import ElaborationError, JoltyError, ScriptError
from jolty.fidelity import FidelityReport
from jolty.plan import Corner, MonteCarlo, Plan
from jolty.results import Result
from jolty.simulate import Simulation, kelvins
from jolty.specs import Specs, Verdict, evaluate, spread

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
    corners: tuple[report.CornerReport, ...] = Field(
        default=(), description="A report per corner of the plan, in its order."
    )
    monte_carlo: report.MonteCarloReport | None = Field(
        default=None,
        description="What the plan's Monte Carlo analysis found; null without one.",
    )
    specs: tuple[Verdict, ...] = Field(
        default=(),
        description="The verdict on each requirement of the plan's specs, in order; "
        "empty when the plan defines no specs().",
    )
    fidelity: FidelityReport | None = Field(
        default=None,
        description="The models the plan's fidelity chose, and its spot checks; null "
        "when it chooses none.",
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
    built = _built(job.script, namespaces, {})
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
    cache: fidelity.Cache = {}
    chosen = fidelity.resolve(built, plan, plan.temperature, cache)
    if plan.temperature is not None or chosen.matched:
        try:
            simulation = chosen.simulation()
        except ElaborationError as error:
            return Built(circuit=built, problems=error.problems)
    outside: list[fidelity.Outside] = []
    solved, reports = _analyses(simulation, plan, job.listed, outside)
    if solved:
        files.save_results(directory / RESULTS, solved)
    corners: list[report.CornerReport] = []
    at_corners: dict[str, dict[str, Result]] = {}
    for name, corner in plan.corners.items():
        rebuilt = (
            _built(job.script, namespaces, corner.arguments)
            if corner.arguments
            else built
        )
        found, at_corners[name] = _corner(name, corner, rebuilt, plan, cache)
        corners.append(found)
    specs = _specs(job.plan, namespaces)
    verdicts = (
        () if specs is None else evaluate(specs, plan.analyses, solved, at_corners)
    )
    checked = _fidelity(chosen, plan, solved, specs)
    monte_carlo = (
        None
        if plan.monte_carlo is None
        else _monte_carlo(chosen, plan, plan.monte_carlo, specs)
    )
    return elaborated.model_copy(
        update={
            "analyses": reports,
            "corners": tuple(corners),
            "monte_carlo": monte_carlo,
            "specs": verdicts,
            "fidelity": checked
            and checked.model_copy(update={"outside": tuple(outside)}),
        }
    )


def _fidelity(
    chosen: fidelity.Fidelity,
    plan: Plan,
    solved: Mapping[str, Result],
    specs: Specs | None,
) -> FidelityReport | None:
    """The models `chosen` holds, and the spot checks of `plan`'s results `solved`.

    Each spot check holds its instance to its contract, and measures `specs` again to
    find the requirements whose verdict depends on the instance's fidelity. It is None
    when no model stands in for an instance.
    """
    if not chosen.matched:
        return None
    contracts = []
    dependences: list[fidelity.Dependence] = []
    modeled = () if specs is None else evaluate(specs, plan.analyses, solved)
    for path in fidelity.sampled(chosen, plan):
        try:
            opened, _ = _analyses(chosen.simulation(opened=path), plan, 1)
        except ElaborationError:
            opened = {}
        contracts.append(fidelity.contract(chosen, plan, path, solved, opened))
        if specs is not None:
            exact = evaluate(specs, plan.analyses, opened)
            dependences.extend(fidelity.dependences(path, modeled, exact))
    return FidelityReport(
        models=fidelity.models(chosen),
        contracts=tuple(contracts),
        dependences=tuple(dependences),
    )


def _corner(
    name: str,
    corner: Corner,
    built: ir.Circuit,
    plan: Plan,
    cache: fidelity.Cache,
) -> tuple[report.CornerReport, dict[str, Result]]:
    """What `plan`'s analyses find on the circuit `built` at `corner`.

    The circuit takes the models the plan's fidelity chooses there, characterized
    unless `cache` holds them. Returns the corner's report, and the results of the
    analyses that end, by name.
    """
    temperature = plan.temperature_at(corner)
    try:
        simulation = fidelity.resolve(built, plan, temperature, cache).simulation()
    except ElaborationError as error:
        found = report.CornerReport(
            name=name, temperature=kelvins(temperature), problems=error.problems
        )
        return found, {}
    solved, reports = _analyses(simulation, plan, 1)
    failures = tuple(r for r in reports if isinstance(r, report.FailureReport))
    found = report.CornerReport(
        name=name, temperature=simulation.temperature, failures=failures
    )
    return found, solved


def _built(
    script: Path,
    namespaces: dict[Path, dict[str, object]],
    arguments: Mapping[str, object],
) -> ir.Circuit:
    """The circuit `script`'s ``build()`` returns, called with `arguments`.

    Raises:
        ScriptError: When it returns neither a component nor a circuit.
    """
    match _call(script, "build", namespaces, arguments=arguments):
        case Component() as top:
            return circuit(top)
        case ir.Circuit() as built:
            return built
        case other:
            message = (
                f"build() returned {type(other).__name__}, not a Component or a Circuit"
            )
            raise ScriptError(message)


def _monte_carlo(
    chosen: fidelity.Fidelity,
    plan: Plan,
    monte_carlo: MonteCarlo,
    specs: Specs | None,
) -> report.MonteCarloReport:
    """What `plan`'s analyses find on each sample of `monte_carlo`, against `specs`.

    Each sample draws the circuit's tolerances, at the plan's own temperature, with the
    models `chosen` holds, which stand for their circuits' nominal parts.
    """
    toleranced = variation.toleranced(chosen.built)
    generator = np.random.default_rng(monte_carlo.seed)
    sampled: list[tuple[Verdict, ...]] = []
    failed = 0
    for _ in range(monte_carlo.samples):
        overrides = variation.sample(toleranced, generator)
        try:
            simulation = chosen.simulation(overrides)
        except ElaborationError:
            solved, ended = {}, False
        else:
            solved, reports = _analyses(simulation, plan, 1)
            ended = not any(isinstance(r, report.FailureReport) for r in reports)
        failed += not ended
        if specs is not None:
            sampled.append(evaluate(specs, plan.analyses, solved))
    spreads, passing = ((), None) if specs is None else spread(specs, sampled)
    return report.MonteCarloReport(
        samples=monte_carlo.samples,
        seed=monte_carlo.seed,
        toleranced=len(toleranced),
        failed=failed,
        spreads=spreads,
        passing=passing,
    )


def _specs(script: Path, namespaces: dict[Path, dict[str, object]]) -> Specs | None:
    """The specs `script`'s ``specs()`` returns; None when it defines none.

    Raises:
        ScriptError: When ``specs()`` returns no specs.
    """
    match _call(script, "specs", namespaces, required=False):
        case None | Specs() as specs:
            return specs
        case other:
            message = f"specs() returned {type(other).__name__}, not Specs"
            raise ScriptError(message)


def _analyses(
    simulation: Simulation,
    plan: Plan,
    listed: int,
    outside: list[fidelity.Outside] | None = None,
) -> tuple[dict[str, Result], tuple[report.AnalysisReport, ...]]:
    """The results of the analyses of `plan` that end, by name, and a report on each.

    Each analysis's excursions outside its logic devices' domains join `outside`.
    """
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
        if outside is not None:
            outside.extend(
                fidelity.Outside(analysis=name, excursion=excursion)
                for excursion in simulation.excursions
            )
    return solved, tuple(reports)


def _call(
    script: Path,
    name: str,
    namespaces: dict[Path, dict[str, object]],
    *,
    required: bool = True,
    arguments: Mapping[str, object] | None = None,
) -> object:
    """Calls the function `name` of `script` with `arguments`.

    The script runs once however often it is used.
    A script that defines no function `name` returns None when it is not `required`.

    Raises:
        ScriptError: When the script defines no function `name` and it is required.
    """
    if script not in namespaces:
        namespaces[script] = runpy.run_path(str(script), run_name="__jolty__")
    function = namespaces[script].get(name)
    if callable(function):
        return function(**(arguments or {}))
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
