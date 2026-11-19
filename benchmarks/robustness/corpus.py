# Copyright 2026 Maxence Faldor
"""Robustness corpus definitions, outcome classification, and failure-rate reporting.

``scripts/bench --robustness`` runs this module from the repository root. Each fixture
pairs a single analysis (operating point, DC sweep, or transient) with a circuit built
in ``own.py`` or fetched at a pinned commit via ``decks.py`` (``external.json``).
Every installed simulator executes each fixture in a separate timed child process
(``runs.py``) through ``benchmarks/simulators.py`` and receives a uniform outcome:

- ``agreed``: solved and matches a reference DC solution, expected settling state, or
  closed-form impasse time.
- ``disagreed``: solved, but matches no reference solution.
- ``solved``: completed on a fixture without a reference solution.
- ``failed``: solver terminated with an error message.
- ``timed out``: exceeded the per-fixture timeout.
- ``unsupported``: simulator cannot read or execute the fixture (excluded from failure
  rates).

Two solutions agree when every node voltage differs by at most the relative tolerance
times the largest reference voltage plus twice the voltage tolerance. Failure rates
aggregate failed and timed-out runs over supported runs separately for DC and transient
fixtures with 95 % Wilson intervals. See ``benchmarks/robustness/README.md`` for fixture
descriptions.
"""

import dataclasses
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Protocol

from measuring import duration, load_average, markdown_table, provenance, save
from simulators import Runner, Simulator, runner

from jolty import ir
from jolty.plan import Method, Options
from jolty.specs import wilson

ROBUSTNESS: Final = Path(__file__).resolve().parent
"""The corpus's directory, ``benchmarks/robustness``."""


class Kind(StrEnum):
    """Failure-rate bucket assigned to a robustness fixture (DC or transient)."""

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
    """Expectation for a transient ending at a DC solution of the final sources.

    Matches every DC solution of the final-level circuit accepted by `into` (or any DC
    solution when `into` is None), described by `expected`.
    """

    expected: str
    into: Callable[[Solution, Sequence[Solution]], bool] | None = None
    """Optional predicate selecting expected DC solutions among all found solutions."""


@dataclass(frozen=True, slots=True)
class Oscillates:
    """Expectation for a transient that avoids all unstable final-level DC solutions.

    All DC equilibria are unstable, so a start from rest departs them and oscillates.
    """


@dataclass(frozen=True, slots=True)
class Impasse:
    """Expectation for a transient terminating at a fold impasse point at `time`.

    Agrees when the simulator halts within relative tolerance of `time`, and disagrees
    when integration continues past `time`.
    """

    time: float


@dataclass(frozen=True, slots=True)
class Completes:
    """Expectation for a transient without a reference, passing on completion."""


type Expectation = Solutions | Settles | Oscillates | Impasse | Completes
"""Verification criterion applied to a robustness fixture run."""


@dataclass(frozen=True, slots=True)
class Built:
    """Jolty circuit representation with temperature, options, and probed nets."""

    circuit: ir.Circuit
    temperature: float
    """Simulation temperature in kelvins."""
    options: Options
    nets: tuple[str, ...]
    """Jolty net names whose voltages are compared across simulators."""
    initial_conditions: Mapping[str, float] = field(default_factory=dict)
    """Initial transient node voltages in volts corresponding to SPICE ``.ic`` cards."""


@dataclass(frozen=True, slots=True)
class Spiced:
    """SPICE netlist representation of a fixture circuit without analysis cards."""

    lines: tuple[str, ...]
    vectors: Mapping[str, str]
    """Mapping from jolty net name to SPICE vector expression."""


class FixtureCircuit(Protocol):
    """Protocol providing jolty and SPICE representations of a robustness circuit."""

    def jolty(self) -> Built:
        """Builds the jolty representation of the circuit.

        Raises:
            Unsupported: When jolty cannot read or elaborate the circuit.
        """
        ...

    def spice(self) -> Spiced:
        """Builds the SPICE netlist representation of the circuit.

        Raises:
            Unsupported: When the circuit cannot be exported as SPICE.
        """
        ...

    def settled(self) -> Built | None:
        """Returns the circuit at final source levels for a settling reference."""
        ...

    def temperature(self) -> float:
        """Returns the simulation temperature in kelvins shared by all simulators."""
        ...


@dataclass(frozen=True, slots=True)
class Fixture:
    """Robustness benchmark fixture pairing one circuit, analysis, and expectation."""

    name: str
    kind: Kind
    origin: str
    """Source suite identifier: ``jolty`` or the manifest suite name."""
    hard: str
    """Summary of the numerical or modeling difficulty tested by the fixture."""
    circuit: FixtureCircuit = field(repr=False)
    analysis: Analysis
    expectation: Expectation
    options: Options = field(default_factory=Options)
    """Simulation options applied uniformly across simulators."""
    rules: tuple[str, ...] = ()
    """Recorded deck-normalization rules applied identically for every simulator."""


class Status(StrEnum):
    """Outcome classification for a robustness fixture run."""

    AGREED = "agreed"
    DISAGREED = "disagreed"
    SOLVED = "solved"
    FAILED = "failed"
    TIMED_OUT = "timed out"
    UNSUPPORTED = "unsupported"


FAILURES: Final = frozenset({Status.FAILED, Status.TIMED_OUT})
"""Run statuses counted in simulator failure rates."""


@dataclass(frozen=True, slots=True, kw_only=True)
class FixtureRun:
    """Unjudged execution record returned by a simulator runner for one fixture."""

    completed: bool
    message: str = ""
    """Verbatim simulator diagnostic explaining a failure or unsupported result."""
    unsupported: bool = False
    timed_out: bool = False
    seconds: float | None = None
    """Total child-process wall time in seconds from launch to exit."""
    analysis_seconds: float | None = None
    """Simulator-reported analysis time in seconds."""
    iterations: int | None = None
    """Total Newton iteration count when reported by the simulator."""
    strategy: str | None = None
    """Operating-point strategy reported by the simulator, ordered for sweeps."""
    values: tuple[Solution, ...] = ()
    """Compared voltages at the operating point, sweep points, or transient end."""
    reached: float | None = None
    """Final simulated timestamp in seconds reached by a transient run."""
    diagnostics: tuple[str, ...] = ()
    """Pre-simulation circuit check facts reported by jolty."""

    def to_json(self) -> dict[str, object]:
        """Serializes this `FixtureRun` into a JSON-compatible dictionary."""
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in dataclasses.asdict(self).items()
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> FixtureRun:
        """Deserializes a `FixtureRun` from a dictionary created by `to_json`."""
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
    """Reference DC solutions used to score a fixture run at each evaluation point.

    Contains one solution set for an operating point or settled transient, or one per
    level for a sweep. Leaves `points` empty with an error `message` when unavailable.
    """

    points: tuple[tuple[Solution, ...], ...] = ()
    message: str = ""


@dataclass(frozen=True, slots=True)
class FixtureStatus:
    """Judged robustness outcome pairing status, detail text, and raw `FixtureRun`."""

    status: Status
    detail: str
    run: FixtureRun


def agrees(found: Solution, reference: Solution, options: Options) -> bool:
    """Returns whether `found` agrees with `reference` under Newton voltage tolerances.

    Checks that every voltage in `reference` matches within the relative tolerance times
    the largest reference magnitude plus twice the voltage tolerance (using `options` or
    SPICE defaults), matching ``tests/differential.py``.

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
    """Returns the first index in `solutions` agreeing with `found`, or None."""
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
) -> FixtureStatus:
    """Classifies `run` of `fixture` against `reference` under `options`."""
    expectation = fixture.expectation
    unfinished = _unfinished(expectation, run, options)
    if unfinished is not None:
        return unfinished
    if isinstance(expectation, Completes) or not reference.points:
        detail = "no reference" + (
            f": {reference.message}" if reference.message else ""
        )
        return FixtureStatus(Status.SOLVED, detail, run)
    if isinstance(expectation, Solutions):
        return _solutions(run, reference, options)
    (solutions,) = reference.points
    (end,) = run.values
    index = which(end, solutions, options)
    if isinstance(expectation, Settles):
        status, detail = _settles(expectation, index, solutions)
    else:
        status, detail = _oscillates(index)
    return FixtureStatus(status, detail, run)


def _unfinished(
    expectation: Expectation, run: FixtureRun, options: Options
) -> FixtureStatus | None:
    """Classifies incomplete, unsupported, timed-out, or impasse runs, else None."""
    if run.unsupported:
        return FixtureStatus(Status.UNSUPPORTED, run.message, run)
    if run.timed_out:
        return FixtureStatus(Status.TIMED_OUT, run.message, run)
    if isinstance(expectation, Impasse):
        return _impasse(expectation, run, options)
    if not run.completed:
        return FixtureStatus(Status.FAILED, run.message, run)
    return None


def _oscillates(index: int | None) -> tuple[Status, str]:
    """Classifies an oscillator transient by whether its end matches a DC state."""
    if index is None:
        return Status.AGREED, "oscillates"
    return Status.DISAGREED, f"rests at DC solution {index + 1}"


def _settles(
    expectation: Settles, index: int | None, solutions: Sequence[Solution]
) -> tuple[Status, str]:
    """Classifies a settling transient using matched DC `index` and `expectation`."""
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


def _solutions(
    run: FixtureRun, reference: Reference, options: Options
) -> FixtureStatus:
    """Scores an operating-point or sweep run against `reference` DC solutions."""
    found: list[int] = []
    for level, (values, solutions) in enumerate(
        zip(run.values, reference.points, strict=True)
    ):
        index = which(values, solutions, options)
        if index is None:
            where = "" if len(reference.points) == 1 else f" at level {level + 1}"
            return FixtureStatus(
                Status.DISAGREED, f"agrees with no DC solution{where}", run
            )
        found.append(index)
    if len(reference.points) == 1:
        count = len(reference.points[0])
        return FixtureStatus(Status.AGREED, f"solution {found[0] + 1} of {count}", run)
    several = sum(len(solutions) > 1 for solutions in reference.points)
    return FixtureStatus(
        Status.AGREED, f"{len(found)} levels, {several} with several solutions", run
    )


def _impasse(expectation: Impasse, run: FixtureRun, options: Options) -> FixtureStatus:
    """Scores a transient expected to halt at fold impasse `expectation.time`."""
    if run.completed:
        return FixtureStatus(
            Status.DISAGREED,
            f"goes on past the impasse point at {expectation.time:.4g} s",
            run,
        )
    relative = options.relative_tolerance or DEFAULT_RELATIVE_TOLERANCE
    if run.reached is not None and abs(run.reached - expectation.time) <= (
        relative * expectation.time
    ):
        return FixtureStatus(Status.AGREED, f"stops at {run.reached:.4g} s", run)
    return FixtureStatus(Status.FAILED, run.message, run)


type Runs = Callable[[Fixture, float], FixtureRun]
"""Callable signature for executing one robustness fixture under a timeout in seconds.

Each simulator registered in `runs.RUNS` implements this callable interface.
"""


def fixtures() -> dict[str, Fixture]:
    """Returns all robustness fixtures keyed by name (built-in circuits, then decks)."""
    import decks  # noqa: PLC0415 - the decks import the corpus's types
    import own  # noqa: PLC0415 - jolty's own circuits import the corpus's types

    found = [*own.fixtures(), *decks.fixtures()]
    return {fixture.name: fixture for fixture in found}


def wilsons(counts: Mapping[Status, int]) -> tuple[int, int, float, float] | None:
    """Computes supported failure counts and the 95 % Wilson confidence interval.

    Returns None when zero runs are supported.
    """
    supported = sum(
        count for status, count in counts.items() if status is not Status.UNSUPPORTED
    )
    if supported == 0:
        return None
    failed = sum(count for status, count in counts.items() if status in FAILURES)
    low, high = wilson(failed, supported)
    return failed, supported, low, high


def cell(outcome: FixtureStatus) -> str:
    """Formats a table cell for `outcome` showing status, analysis time, and iterations.

    Displays simulator-reported analysis time (excluding process startup); total
    process wall time is recorded separately in the JSON results file.
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
    outcomes: Mapping[tuple[str, str], FixtureStatus],
) -> list[str]:
    """Formats Markdown tables of fixture outcomes and per-simulator failure rates."""
    fixture_rows = [
        [
            f"`{fixture.name}`",
            fixture.kind,
            *(cell(outcomes[fixture.name, name]) for name in names),
        ]
        for fixture in chosen
    ]
    summary_rows: list[list[str]] = []
    for name in names:
        for kind in Kind:
            counts: dict[Status, int] = {}
            for fixture in chosen:
                if fixture.kind is kind:
                    status = outcomes[fixture.name, name].status
                    counts[status] = counts.get(status, 0) + 1
            rate = wilsons(counts)
            if rate is None:
                summary_rows.append([name, kind, "", "0", "", ""])
                continue
            failed, supported, low, high = rate
            summary_rows.append(
                [
                    name,
                    kind,
                    str(failed),
                    str(supported),
                    f"{failed / supported:.0%}",
                    f"{low:.0%} to {high:.0%}",
                ]
            )
    return [
        *markdown_table(["Fixture", "Kind", *names], fixture_rows).splitlines(),
        "",
        *markdown_table(
            ["Simulator", "Kind", "Failed", "Runs", "Rate", "95 % interval"],
            summary_rows,
        ).splitlines(),
    ]


def details(
    chosen: Sequence[Fixture],
    names: Sequence[str],
    outcomes: Mapping[tuple[str, str], FixtureStatus],
) -> list[str]:
    """Formats detailed per-fixture outcome lines and jolty pre-check diagnostics."""
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
    """Summarizes `diagnostics` by rule name, count, and first example fact."""
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
    outcomes: dict[tuple[str, str], FixtureStatus] = {}
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
