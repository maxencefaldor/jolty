# Copyright 2026 Maxence Faldor
"""The corpus of hard circuits, each simulator's outcome on each, and its failure rates.

``scripts/bench --robustness`` runs this module from the repository root. Each
fixture is one analysis of one circuit: an operating point, a DC sweep or a transient,
of a circuit jolty builds (``own.py``) or of a SPICE deck fetched at a pinned commit
and never copied (``decks.py``, whose manifest is ``external.json``). Every simulator
installed runs every fixture, by its runner of ``benchmarks/simulators.py``, each run in
a process of its own that a timeout stops (``runs.py``), and each run's outcome is
classified alike:

- ``agreed``: solved, and it agrees with the reference, every DC solution jolty finds
  for an operating point or a sweep, the DC solutions a transient must settle into, or
  the closed form a transient must stop at;
- ``disagreed``: solved, and it agrees with no reference solution;
- ``solved``: solved, where the fixture has no reference to judge it against;
- ``failed``: the simulator gave up, with its message;
- ``timed out``: it ran past the timeout;
- ``unsupported``: it cannot read or simulate what the fixture holds, with the reason,
  which never counts as a failure.

Two results agree, as the differential tests judge them, when each voltage is within
the fixture's relative tolerance of the largest voltage of the reference, plus twice its
absolute tolerance on voltages: the tolerances Newton iteration is asked to meet.

Each simulator's failure rate is its failed and timed out runs over the runs it
supports, DC and transient apart, with the 95 % Wilson interval a yield states.
``benchmarks/robustness/README.md`` says what each fixture holds and what makes it hard.
"""

import dataclasses
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Protocol

from measuring import duration, load_average, provenance, save
from simulators import Runner, Simulator, runner

from jolty import ir
from jolty.plan import Method, Options
from jolty.specs import wilson

ROBUSTNESS: Final = Path(__file__).resolve().parent
"""The corpus's directory, ``benchmarks/robustness``."""


class Kind(StrEnum):
    """Which failure rate a fixture counts toward."""

    DC = "dc"
    TRANSIENT = "transient"


@dataclass(frozen=True, slots=True)
class Point:
    """The operating point, every source at its level at time zero."""


@dataclass(frozen=True, slots=True)
class Sweep:
    """The operating point at each level of a source, from `start` to `stop` by `step`.

    The source is named as jolty's circuit names its instance, and as the deck names
    its element; the levels are in volts or amperes.
    """

    source: str
    start: float
    stop: float
    step: float

    @property
    def levels(self) -> list[float]:
        """Each level, as SPICE counts them: from the start, a step apart, to the stop.

        Examples:
            >>> Sweep("V1", 0.0, 15.0, 1.0).levels[-3:]
            [13.0, 14.0, 15.0]
        """
        count = round((self.stop - self.start) / self.step)
        return [self.start + index * self.step for index in range(count + 1)]


@dataclass(frozen=True, slots=True)
class Transient:
    """A transient from the operating point to `stop`, each step at most `longest`.

    `step` is SPICE's ``TSTEP``, the interval its output is printed at. Every simulator
    takes the same longest step: the one given, or SPICE's default, the smaller of
    ``TSTEP`` and a fiftieth of the run.
    """

    step: float
    stop: float
    max_step: float | None = None
    method: Method = "trapezoidal"

    @property
    def longest(self) -> float:
        """The longest step every simulator takes."""
        return (
            min(self.step, self.stop / 50) if self.max_step is None else self.max_step
        )


type Analysis = Point | Sweep | Transient
"""The analysis a fixture runs."""


type Solution = Mapping[str, float]
"""A DC solution's voltage at each net compared, in volts, by jolty's names."""


@dataclass(frozen=True, slots=True)
class Solutions:
    """An operating point or a sweep: at each point, any DC solution jolty finds."""


@dataclass(frozen=True, slots=True)
class Settles:
    """A transient whose sources end constant, and whose end is a DC solution there.

    The reference is every DC solution of the circuit with each source at its last
    level, that `into` accepts, as `expected` says; any, without `into`.
    """

    expected: str
    into: Callable[[Solution, Sequence[Solution]], bool] | None = None
    """Whether one of the solutions, beside them all, is the one expected."""


@dataclass(frozen=True, slots=True)
class Oscillates:
    """A transient whose end is at none of the DC solutions of its sources' last levels.

    Each of them is unstable, so the solution from rest leaves them and oscillates.
    """


@dataclass(frozen=True, slots=True)
class Impasse:
    """A transient whose solution ends at `time`, where no solution goes on.

    A run is right when it stops within its relative tolerance of `time`, and agrees
    with no solution when it goes on past it.
    """

    time: float


@dataclass(frozen=True, slots=True)
class Completes:
    """A transient with no reference to judge it against: it is solved if it ends."""


type Expectation = Solutions | Settles | Oscillates | Impasse | Completes
"""How a fixture's outcome is judged."""


@dataclass(frozen=True, slots=True)
class Built:
    """A fixture's circuit as jolty simulates it: at a temperature, with options."""

    circuit: ir.Circuit
    temperature: float
    """In kelvins."""
    options: Options
    nets: tuple[str, ...]
    """The nets whose voltages are compared, by jolty's names."""


@dataclass(frozen=True, slots=True)
class Spiced:
    """A fixture's circuit as a SPICE deck, without analyses: its lines, title first."""

    lines: tuple[str, ...]
    vectors: Mapping[str, str]
    """The vector of each net compared, by jolty's name of the net."""


class FixtureCircuit(Protocol):
    """A circuit, as each simulator reads it."""

    def jolty(self) -> Built:
        """The circuit as jolty simulates it.

        Raises:
            Unsupported: When jolty cannot read it.
        """
        ...

    def spice(self) -> Spiced:
        """The circuit as a SPICE deck.

        Raises:
            Unsupported: When it cannot be written as one.
        """
        ...

    def settled(self) -> Built | None:
        """The circuit with every source at its last level, if expected of it."""
        ...

    def temperature(self) -> float:
        """The temperature every simulator runs the circuit at, in kelvins."""
        ...


@dataclass(frozen=True, slots=True)
class Fixture:
    """One analysis of one circuit, and how its outcome is judged."""

    name: str
    kind: Kind
    origin: str
    """Where the circuit comes from: ``jolty``, or the manifest's suite."""
    hard: str
    """What makes it hard."""
    circuit: FixtureCircuit = field(repr=False)
    analysis: Analysis
    expectation: Expectation
    options: Options = field(default_factory=Options)
    """The options every simulator runs it with, their defaults unless the deck sets
    them."""
    rules: tuple[str, ...] = ()
    """The recorded rules that fixed the deck's defects, alike for every simulator."""


class Status(StrEnum):
    """What a run of a fixture came to."""

    AGREED = "agreed"
    DISAGREED = "disagreed"
    SOLVED = "solved"
    FAILED = "failed"
    TIMED_OUT = "timed out"
    UNSUPPORTED = "unsupported"


FAILURES: Final = frozenset({Status.FAILED, Status.TIMED_OUT})
"""The outcomes a failure rate counts."""


@dataclass(frozen=True, slots=True, kw_only=True)
class FixtureRun:
    """What a simulator returned for a fixture, before it is judged."""

    completed: bool
    message: str = ""
    """Why it failed or does not support the fixture, in its own words."""
    unsupported: bool = False
    timed_out: bool = False
    seconds: float | None = None
    """The wall time of its process, from its start to its exit."""
    analysis_seconds: float | None = None
    """The time of the analysis alone, as the simulator measured it."""
    iterations: int | None = None
    """Its Newton iterations, where it counts them."""
    strategy: str | None = None
    """How it found the operating point, where it says: for a sweep, each strategy it
    found a level's by, in the order it first did."""
    values: tuple[Solution, ...] = ()
    """The voltages compared: at the operating point, at each level of a sweep, or at
    a transient's end."""
    reached: float | None = None
    """The last time a transient reached, where it says."""
    diagnostics: tuple[str, ...] = ()
    """What jolty's checks found in the circuit before it ran, each a fact."""

    def to_json(self) -> dict[str, object]:
        """The run as JSON."""
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in dataclasses.asdict(self).items()
        }

    @classmethod
    def of(cls, document: Mapping[str, Any]) -> FixtureRun:
        """The run a document of `to_json` holds."""
        return cls(
            completed=bool(document["completed"]),
            message=str(document.get("message", "")),
            unsupported=bool(document.get("unsupported", False)),
            timed_out=bool(document.get("timed_out", False)),
            seconds=document.get("seconds"),
            analysis_seconds=document.get("analysis_seconds"),
            iterations=document.get("iterations"),
            strategy=document.get("strategy"),
            values=tuple(document.get("values", ())),
            reached=document.get("reached"),
            diagnostics=tuple(document.get("diagnostics", ())),
        )


@dataclass(frozen=True, slots=True)
class Reference:
    """What a fixture's runs are judged against: the DC solutions at each point.

    For an operating point, one point; for a sweep, a point per level; for a transient
    that settles or oscillates, its end. None where jolty found no solution there, with
    its message.
    """

    points: tuple[tuple[Solution, ...], ...] = ()
    message: str = ""


@dataclass(frozen=True, slots=True)
class Outcome:
    """A run of a fixture, judged."""

    status: Status
    detail: str
    run: FixtureRun


def agrees(found: Solution, reference: Solution, options: Options) -> bool:
    """Whether `found` agrees with `reference`, as the differential tests judge it.

    Each voltage is within the relative tolerance of the largest voltage of the
    reference, plus twice the absolute tolerance on voltages, both the options' or
    SPICE's defaults: the tolerances Newton iteration is asked to meet, at the scale of
    the whole solution, as ``tests/differential.py`` judges two engines at its tight
    options.

    Examples:
        >>> agrees({"a": 1.0004}, {"a": 1.0}, Options())
        True
        >>> agrees({"a": 1.01}, {"a": 1.0}, Options())
        False
    """
    relative = options.relative_tolerance or DEFAULT_RELATIVE_TOLERANCE
    absolute = options.voltage_tolerance or DEFAULT_VOLTAGE_TOLERANCE
    scale = max((abs(value) for value in reference.values()), default=0.0) or 1.0
    allowed = relative * scale + 2 * absolute
    return all(
        name in found and abs(found[name] - value) <= allowed
        for name, value in reference.items()
    )


DEFAULT_RELATIVE_TOLERANCE: Final = 1e-3
"""SPICE's ``RELTOL``, the default of jolty's `relative_tolerance` and ngspice's."""

DEFAULT_VOLTAGE_TOLERANCE: Final = 1e-6
"""SPICE's ``VNTOL``, in volts, the default of jolty's `voltage_tolerance` and
ngspice's."""


def which(
    found: Solution, solutions: Sequence[Solution], options: Options
) -> int | None:
    """The index of the first of `solutions` that `found` agrees with, if any."""
    return next(
        (
            index
            for index, solution in enumerate(solutions)
            if agrees(found, solution, options)
        ),
        None,
    )


def judge(
    fixture: Fixture, run: FixtureRun, reference: Reference, options: Options
) -> Outcome:
    """The outcome of `run` of `fixture`, against its `reference`, at its `options`."""
    expectation = fixture.expectation
    unfinished = _unfinished(expectation, run, options)
    if unfinished is not None:
        return unfinished
    if isinstance(expectation, Completes) or not reference.points:
        detail = "no reference" + (
            f": {reference.message}" if reference.message else ""
        )
        return Outcome(Status.SOLVED, detail, run)
    if isinstance(expectation, Solutions):
        return _solutions(run, reference, options)
    (solutions,) = reference.points
    (end,) = run.values
    index = which(end, solutions, options)
    if isinstance(expectation, Settles):
        status, detail = _settles(expectation, index, solutions)
    else:
        status, detail = _oscillates(index)
    return Outcome(status, detail, run)


def _unfinished(
    expectation: Expectation, run: FixtureRun, options: Options
) -> Outcome | None:
    """The outcome of a run that did not finish, or of an impasse's; None otherwise."""
    if run.unsupported:
        return Outcome(Status.UNSUPPORTED, run.message, run)
    if run.timed_out:
        return Outcome(Status.TIMED_OUT, run.message, run)
    if isinstance(expectation, Impasse):
        return _impasse(expectation, run, options)
    if not run.completed:
        return Outcome(Status.FAILED, run.message, run)
    return None


def _oscillates(index: int | None) -> tuple[Status, str]:
    """A ring's outcome, whose end agrees with the DC solution `index`, if any."""
    if index is None:
        return Status.AGREED, "oscillates"
    return Status.DISAGREED, f"rests at DC solution {index + 1}"


def _settles(
    expectation: Settles, index: int | None, solutions: Sequence[Solution]
) -> tuple[Status, str]:
    """A transient's outcome, whose end agrees with the DC solution `index`, if any."""
    if index is None:
        return Status.DISAGREED, "ends at no DC solution"
    if expectation.into is not None and not expectation.into(
        solutions[index], solutions
    ):
        return (
            Status.DISAGREED,
            f"ends at DC solution {index + 1}, not {expectation.expected}",
        )
    return Status.AGREED, expectation.expected


def _solutions(run: FixtureRun, reference: Reference, options: Options) -> Outcome:
    """An operating point or a sweep, against every DC solution jolty finds."""
    found: list[int] = []
    for level, (values, solutions) in enumerate(
        zip(run.values, reference.points, strict=True)
    ):
        index = which(values, solutions, options)
        if index is None:
            where = "" if len(reference.points) == 1 else f" at level {level + 1}"
            return Outcome(Status.DISAGREED, f"agrees with no DC solution{where}", run)
        found.append(index)
    if len(reference.points) == 1:
        count = len(reference.points[0])
        return Outcome(Status.AGREED, f"solution {found[0] + 1} of {count}", run)
    several = sum(len(solutions) > 1 for solutions in reference.points)
    return Outcome(
        Status.AGREED, f"{len(found)} levels, {several} with several solutions", run
    )


def _impasse(expectation: Impasse, run: FixtureRun, options: Options) -> Outcome:
    """A transient that must stop at its impasse point."""
    if run.completed:
        return Outcome(
            Status.DISAGREED,
            f"goes on past the impasse point at {expectation.time:.4g} s",
            run,
        )
    relative = options.relative_tolerance or DEFAULT_RELATIVE_TOLERANCE
    if run.reached is not None and abs(run.reached - expectation.time) <= (
        relative * expectation.time
    ):
        return Outcome(Status.AGREED, f"stops at {run.reached:.4g} s", run)
    return Outcome(Status.FAILED, run.message, run)


type Runs = Callable[[Fixture, float], FixtureRun]
"""How the corpus runs a fixture in one simulator, stopped after a timeout in seconds.

A simulator joins the corpus by one of these, listed in `runs.RUNS`.
"""


def fixtures() -> dict[str, Fixture]:
    """Every fixture of the corpus, by name: jolty's own circuits, then the decks'."""
    import decks  # noqa: PLC0415 - the decks import the corpus's types
    import own  # noqa: PLC0415 - jolty's own circuits import the corpus's types

    found = [*own.fixtures(), *decks.fixtures()]
    return {fixture.name: fixture for fixture in found}


def wilsons(counts: Mapping[Status, int]) -> tuple[int, int, float, float] | None:
    """The failures among the supported runs, and the 95 % Wilson interval of the rate.

    None when no run is supported.
    """
    supported = sum(
        count for status, count in counts.items() if status is not Status.UNSUPPORTED
    )
    if supported == 0:
        return None
    failed = sum(count for status, count in counts.items() if status in FAILURES)
    low, high = wilson(failed, supported)
    return failed, supported, low, high


def cell(outcome: Outcome) -> str:
    """A table's cell for `outcome`: the status, its analysis's time, its iterations.

    The time is the analyses' alone, as each simulator measured it: jolty's around
    its analysis, ngspice's own ``Total analysis time``; the results keep each
    process's time from its start to its exit beside it.
    """
    run = outcome.run
    parts = [outcome.status.value]
    if run.analysis_seconds is not None and outcome.status is not Status.UNSUPPORTED:
        parts.append(duration(run.analysis_seconds))
    if run.iterations is not None:
        parts.append(f"{run.iterations:,} it.")
    return ", ".join(parts)


def table(
    chosen: Sequence[Fixture],
    names: Sequence[str],
    outcomes: Mapping[tuple[str, str], Outcome],
) -> list[str]:
    """The Markdown table of `chosen` fixtures by simulator, and each one's rates."""
    lines = [
        "| Fixture | Kind | " + " | ".join(names) + " |",
        "| --- | --- | " + " | ".join("---" for _ in names) + " |",
    ]
    lines.extend(
        f"| `{fixture.name}` | {fixture.kind} | "
        + " | ".join(cell(outcomes[fixture.name, name]) for name in names)
        + " |"
        for fixture in chosen
    )
    lines += ["", "| Simulator | Kind | Failed | Runs | Rate | 95 % interval |"]
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for name in names:
        for kind in Kind:
            counts: dict[Status, int] = {}
            for fixture in chosen:
                if fixture.kind is kind:
                    status = outcomes[fixture.name, name].status
                    counts[status] = counts.get(status, 0) + 1
            rate = wilsons(counts)
            if rate is None:
                lines.append(f"| {name} | {kind} | | 0 | | |")
                continue
            failed, supported, low, high = rate
            lines.append(
                f"| {name} | {kind} | {failed} | {supported} | "
                f"{failed / supported:.0%} | {low:.0%} to {high:.0%} |"
            )
    return lines


def details(
    chosen: Sequence[Fixture],
    names: Sequence[str],
    outcomes: Mapping[tuple[str, str], Outcome],
) -> list[str]:
    """Each outcome's detail: which solution, or the message, and jolty's checks."""
    lines: list[str] = []
    for fixture in chosen:
        lines.append(f"{fixture.name}:")
        for name in names:
            outcome = outcomes[fixture.name, name]
            strategy = outcome.run.strategy
            how = "" if strategy is None else f" ({strategy})"
            detail = " ".join(outcome.detail.split())
            lines.append(f"  {name}: {outcome.status}{how}: {detail[:400]}")
        found = next(
            (
                outcomes[fixture.name, name].run.diagnostics
                for name in names
                if outcomes[fixture.name, name].run.diagnostics
            ),
            (),
        )
        lines.extend(f"  jolty's checks: {each}" for each in _rules(found))
    return lines


def _rules(diagnostics: Sequence[str]) -> list[str]:
    """Each rule that `diagnostics` raise, how often, and the first one's fact."""
    by_rule: dict[str, list[str]] = {}
    for diagnostic in diagnostics:
        rule, _, fact = diagnostic.partition(": ")
        by_rule.setdefault(rule, []).append(fact)
    return [
        f"{rule}, {len(facts)} times, such as: {facts[0][:200]}"
        for rule, facts in by_rule.items()
    ]


def measure(
    names: Iterable[str],
    chosen: Sequence[Simulator],
    timeout: float,
    output: Path,
) -> None:
    """Runs the fixtures `names` in the simulators `chosen`, prints and saves them.

    Every fixture runs when none is named, in every simulator installed when none is
    chosen. The results go to `output`, headed by their provenance.

    Raises:
        SystemExit: When a fixture named does not exist, or a simulator chosen does not
            run the corpus or is not installed.
    """
    import runs  # noqa: PLC0415 - the runs import the corpus's types

    every = fixtures()
    wanted = list(names) or list(every)
    unknown = [name for name in wanted if name not in every]
    if unknown:
        message = (
            f"no fixture is named {', '.join(unknown)}; there are {', '.join(every)}"
        )
        raise SystemExit(message)
    missing = [simulator for simulator in chosen if simulator not in runs.RUNS]
    if missing:
        message = (
            f"the corpus runs no {', '.join(missing)}; it runs {', '.join(runs.RUNS)}"
        )
        raise SystemExit(message)
    runners: list[Runner] = [runner(simulator) for simulator in chosen or runs.RUNS]
    versions = {each.simulator: each.version() for each in runners}
    absent = [simulator for simulator, version in versions.items() if version is None]
    if absent and chosen:
        message = f"{', '.join(absent)} not installed"
        raise SystemExit(message)
    names_run = [
        simulator for simulator, version in versions.items() if version is not None
    ]
    selected = [every[name] for name in wanted]
    outcomes: dict[tuple[str, str], Outcome] = {}
    references: dict[str, Reference] = {}
    started = time.perf_counter()
    for fixture in selected:
        load = load_average()
        reference = runs.reference(fixture, timeout)
        references[fixture.name] = reference
        for simulator in names_run:
            run = runs.RUNS[simulator](fixture, timeout)
            outcome = judge(fixture, run, reference, fixture.options)
            outcomes[fixture.name, simulator] = outcome
            print(
                f"{fixture.name} in {simulator}: {cell(outcome)}, load {load:.2f}",
                flush=True,
            )
    print()
    print("\n".join(table(selected, names_run, outcomes)))
    print()
    print("\n".join(details(selected, names_run, outcomes)))
    print(f"\nmeasured in {duration(time.perf_counter() - started)}")
    document = {
        **provenance().model_dump(mode="json"),
        "simulators": versions,
        "timeout": timeout,
        "fixtures": {
            fixture.name: {
                "kind": fixture.kind.value,
                "origin": fixture.origin,
                "hard": fixture.hard,
                "rules": list(fixture.rules),
                "reference": {
                    "points": [
                        list(point) for point in references[fixture.name].points
                    ],
                    "message": references[fixture.name].message,
                },
                "outcomes": {
                    name: {
                        "status": outcomes[fixture.name, name].status.value,
                        "detail": outcomes[fixture.name, name].detail,
                        "run": outcomes[fixture.name, name].run.to_json(),
                    }
                    for name in names_run
                },
            }
            for fixture in selected
        },
    }
    save(output, document)
