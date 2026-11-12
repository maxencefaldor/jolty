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
then on each sample of the plan's Monte Carlo analysis, which runs at the plan's own
conditions and at each corner. It writes the
[`Outcome`][jolty.runner.Outcome], what was built or why it failed, to
``DIRECTORY/outcome.json``, and the results of the analyses at the plan's
own conditions to ``DIRECTORY/results.npz``.

Scripts and analyses therefore run outside the command line and the MCP server, whose
standard output may be a protocol channel: what a script prints is captured, a timeout
stops a script or an analysis that runs too long, and a crash ends only the runner.
"""

import functools
import itertools
import json
import runpy
import sys
import sysconfig
import traceback
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Final, Literal, override

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from jolty import checks, fidelity, files, ir, report, variation
from jolty.builder import Component, circuit
from jolty.errors import ElaborationError, JoltyError, ScriptError
from jolty.fidelity import FidelityReport
from jolty.plan import Corner, MonteCarlo, Plan, WorstCase
from jolty.results import Result
from jolty.simulate import Simulation, kelvins
from jolty.specs import Requirement, Specs, Verdict, evaluate, judge, reading, spread

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
    worst_case: report.WorstCaseReport | None = Field(
        default=None,
        description="What the plan's worst-case analysis found; null without one.",
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
    chosen_at: dict[str, fidelity.Fidelity | None] = {}
    for name, corner in plan.corners.items():
        rebuilt = (
            _built(job.script, namespaces, corner.arguments)
            if corner.arguments
            else built
        )
        found, at_corners[name], chosen_at[name] = _corner(
            name, corner, rebuilt, plan, cache
        )
        corners.append(found)
    specs = _specs(job.plan, namespaces)
    verdicts = (
        () if specs is None else evaluate(specs, plan.analyses, solved, at_corners)
    )
    checked = _fidelity(chosen, plan, solved, specs)
    monte_carlo = (
        None
        if plan.monte_carlo is None
        else _monte_carlo(chosen, chosen_at, plan, plan.monte_carlo, specs)
    )
    worst_case = (
        None
        if plan.worst_case is None
        else _worst_case(
            [(None, chosen, solved)]
            + [(name, chosen_at[name], at_corners[name]) for name in chosen_at],
            plan,
            plan.worst_case,
            specs,
        )
    )
    return elaborated.model_copy(
        update={
            "analyses": reports,
            "corners": tuple(corners),
            "monte_carlo": monte_carlo,
            "worst_case": worst_case,
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
) -> tuple[report.CornerReport, dict[str, Result], fidelity.Fidelity | None]:
    """What `plan`'s analyses find on the circuit `built` at `corner`.

    The circuit takes the models the plan's fidelity chooses there, characterized
    unless `cache` holds them. Returns the corner's report, the results of the analyses
    that end, by name, and the models chosen, none when the circuit does not elaborate.
    """
    temperature = plan.temperature_at(corner)
    try:
        chosen = fidelity.resolve(built, plan, temperature, cache)
        simulation = chosen.simulation()
    except ElaborationError as error:
        found = report.CornerReport(
            name=name, temperature=kelvins(temperature), problems=error.problems
        )
        return found, {}, None
    solved, reports = _analyses(simulation, plan, 1)
    failures = tuple(r for r in reports if isinstance(r, report.FailureReport))
    found = report.CornerReport(
        name=name, temperature=simulation.temperature, failures=failures
    )
    return found, solved, chosen


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
    chosen_at: Mapping[str, fidelity.Fidelity | None],
    plan: Plan,
    monte_carlo: MonteCarlo,
    specs: Specs | None,
) -> report.MonteCarloReport:
    """What `plan`'s analyses find on each sample of `monte_carlo`, against `specs`.

    Each sample draws the circuit's tolerances and runs at the plan's own conditions,
    with the models `chosen` holds, and at each corner, with the models `chosen_at`
    holds there, none where the circuit does not elaborate; the models stand for their
    circuits' nominal parts. Every run draws its samples from the seed afresh, so a
    corner whose circuit holds the same toleranced parameters runs each sample on the
    same parts, and a requirement's verdict on a sample is its worst over the runs.
    """
    runs = zip(
        _samples(chosen, plan, monte_carlo),
        *(_samples(at, plan, monte_carlo) for at in chosen_at.values()),
        strict=True,
    )
    sampled: list[tuple[Verdict, ...]] = []
    failed = 0
    for (solved, ended), *there in runs:
        corners = {
            name: found for name, (found, _) in zip(chosen_at, there, strict=True)
        }
        failed += not (ended and all(ended_there for _, ended_there in there))
        if specs is not None:
            sampled.append(evaluate(specs, plan.analyses, solved, corners))
    spreads, passing = ((), None) if specs is None else spread(specs, sampled)
    return report.MonteCarloReport(
        samples=monte_carlo.samples,
        seed=monte_carlo.seed,
        toleranced=len(variation.toleranced(chosen.built)),
        corners=tuple(chosen_at),
        failed=failed,
        spreads=spreads,
        passing=passing,
    )


def _samples(
    chosen: fidelity.Fidelity | None, plan: Plan, monte_carlo: MonteCarlo
) -> Iterator[tuple[dict[str, Result], bool]]:
    """What `plan`'s analyses find on each sample of `monte_carlo`, in turn.

    On each, the results of the analyses that end, by name, and whether all of them do.
    The samples are drawn from the seed, with the models `chosen` holds; with none, no
    analysis ends on any.
    """
    if chosen is None:
        yield from itertools.repeat(({}, False), monte_carlo.samples)
        return
    toleranced = variation.toleranced(chosen.built)
    generator = np.random.default_rng(monte_carlo.seed)
    for _ in range(monte_carlo.samples):
        overrides = variation.sample(toleranced, generator)
        try:
            simulation = chosen.simulation(overrides)
        except ElaborationError:
            yield {}, False
            continue
        solved, reports = _analyses(simulation, plan, 1)
        yield solved, not any(isinstance(r, report.FailureReport) for r in reports)


type _Conditions = tuple[str | None, fidelity.Fidelity | None, Mapping[str, Result]]
"""A run's conditions: its corner, None for the plan's own; the models chosen there,
None where the circuit does not elaborate; and the results of its nominal run."""


def _worst_case(
    conditions: Sequence[_Conditions],
    plan: Plan,
    worst_case: WorstCase,
    specs: Specs | None,
) -> report.WorstCaseReport:
    """Each requirement of `specs` at the extremes of the tolerances that worsen it.

    At each of `conditions`, a run per toleranced parameter, moved its `worst_case`
    step, finds which way each requirement's value moves with it, and a run per bound
    of each requirement sets every parameter, and every lot's spread, at the extreme
    that moves the value toward the bound; runs at the same extremes are shared. A
    requirement's verdict is its worst over every bound and conditions where the
    circuit elaborates.
    """
    requirements = () if specs is None else specs.requirements
    found: list[list[tuple[str | None, Mapping[str, Result]]]] = [
        [] for _ in requirements
    ]
    runs = 0
    for corner, chosen, nominal in conditions:
        if chosen is None or not requirements:
            continue
        varied = variation.toleranced(chosen.built)
        run = functools.partial(_overridden, chosen, plan)
        slopes = _slopes(varied, requirements, nominal, worst_case.step, run)
        runs += len(varied)
        extremes: dict[str, Mapping[str, Result]] = {}
        for index, requirement in enumerate(requirements):
            bounds = [
                direction
                for direction, bound in (
                    (-1.0, requirement.minimum),
                    (1.0, requirement.maximum),
                )
                if bound is not None
            ]
            for toward in bounds:
                directions = [toward * slope[index] for slope in slopes]
                lots: dict[variation.Lot, float] = {}
                for parameter, direction in zip(varied, directions, strict=True):
                    if (key := variation.lot(parameter)) is not None:
                        lots[key] = lots.get(key, 0.0) + direction
                overrides = variation.extreme(
                    varied,
                    [float(np.sign(direction)) for direction in directions],
                    {key: float(np.sign(direction)) for key, direction in lots.items()},
                )
                key = json.dumps(overrides, sort_keys=True)
                if key not in extremes:
                    extremes[key] = run(overrides)
                    runs += 1
                found[index].append((corner, extremes[key]))
    verdicts = tuple(
        judge(requirement, plan.analyses, at)
        for requirement, at in zip(requirements, found, strict=True)
        if at
    )
    first = next((chosen for _, chosen, _ in conditions if chosen is not None), None)
    return report.WorstCaseReport(
        step=worst_case.step,
        toleranced=0 if first is None else len(variation.toleranced(first.built)),
        corners=tuple(corner for corner, _, _ in conditions if corner is not None),
        runs=runs,
        verdicts=verdicts,
    )


def _overridden(
    chosen: fidelity.Fidelity, plan: Plan, overrides: dict[str, dict[str, float]]
) -> dict[str, Result]:
    """The results of `plan`'s analyses that end on the circuit `chosen` holds.

    Its parameters are replaced by `overrides`; no analysis ends where it does not
    elaborate.
    """
    solved: dict[str, Result] = {}
    try:
        simulation = chosen.simulation(overrides)
    except ElaborationError:
        return solved
    solved, _ = _analyses(simulation, plan, 1)
    return solved


def _slopes(
    varied: Sequence[variation.Varied],
    requirements: Sequence[Requirement],
    nominal: Mapping[str, Result],
    step: float,
    run: Callable[[dict[str, dict[str, float]]], Mapping[str, Result]],
) -> list[list[float]]:
    """How far each requirement's value moves, per parameter of `varied` moved.

    Each parameter moves `step` of its tolerance, and `run` runs the analyses there;
    a value moves from its value on the `nominal` results, and 0 where either cannot be
    read.
    """
    base = [reading(requirement, nominal) for requirement in requirements]
    slopes = []
    for index in range(len(varied)):
        moved = run(variation.moved(varied, index, step))
        slope = []
        for requirement, before in zip(requirements, base, strict=True):
            after = reading(requirement, moved)
            slope.append(0.0 if before is None or after is None else after - before)
        slopes.append(slope)
    return slopes


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
