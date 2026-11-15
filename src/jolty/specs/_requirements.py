# Copyright 2026 Maxence Faldor
"""Requirements, their verdicts, and their spread over Monte Carlo samples.

What [`jolty.specs`][jolty.specs] describes: a measure compared with a bound, the
verdict on it at each of a plan's conditions, and the yield. Over a batch's runs, the
verdicts are [`Judged`][jolty.specs.Judged] as columns, one entry per run, each what the
verdict on that run alone states.
"""

import math
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Annotated, Final, Literal, override

import numpy as np
import numpy.typing as npt
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    WithJsonSchema,
)

from jolty import _core, ir, results
from jolty.builder import caller
from jolty.errors import DimensionError, JoltyError
from jolty.units import (
    DIMENSIONLESS,
    Dimension,
    Quantity,
    finite,
    format_si,
)

type Plain = results.Plain
"""The unit of a measure that reads a plain number: none for a ratio."""

type Bound = Quantity | float
"""A bound: a quantity, or a number for a measure in dB, degrees or percent."""

type Status = Literal["passed", "marginal", "failed", "unmeasured"]
"""What became of a requirement."""

_STATUSES: Final[tuple[Status, ...]] = ("passed", "marginal", "failed", "unmeasured")

_SEVERITY: Final[dict[Status, int]] = {
    "failed": 0,
    "unmeasured": 1,
    "marginal": 2,
    "passed": 3,
}
"""How bad each status is, the worst first: the order a verdict picks its run by."""


def _a_number(value: float) -> float:
    if math.isnan(value):
        message = "a value read is a number, finite or infinite"
        raise ValueError(message)
    return value


def _written(value: float) -> float | str:
    if math.isfinite(value):
        return value
    return "Infinity" if value > 0 else "-Infinity"


type ExtendedReal = Annotated[
    float,
    AfterValidator(_a_number),
    PlainSerializer(_written, when_used="json"),
    WithJsonSchema(
        {"anyOf": [{"type": "number"}, {"enum": ["Infinity", "-Infinity"]}]}
    ),
]
"""A real number or an infinity, which JSON has no number for: there it is written
``"Infinity"`` or ``"-Infinity"``, as a gain of exactly nothing is ``-inf`` dB."""


class MeasureError(JoltyError):
    """A measure cannot be read from the result of its analysis."""


@dataclass(frozen=True, slots=True)
class Reading:
    """What a measure read: a value in SI base units, and its unit."""

    value: float
    unit: Dimension | Plain

    @property
    def dimension(self) -> Dimension:
        """The value's dimension; a plain number has none."""
        return self.unit if isinstance(self.unit, Dimension) else DIMENSIONLESS


@dataclass(frozen=True, slots=True)
class Native:
    """A measure resolved for the engine to read, against one kind of result.

    Its `measure` holds its signals as the indices of the result's unknowns, and its
    times, levels and frequencies in SI base units, so the engine reads it without
    Python.
    """

    measure: _core.Measure
    unit: Dimension | Plain
    """The unit of what the engine reads."""
    explain: Callable[[_core.MeasureError], str] = field(repr=False, compare=False)
    """Says in words why the engine could not read a result, from its error."""

    def read(self, result: results.AnalysisResult) -> Reading:
        """What the engine reads of `result`, a result of the kind it was resolved for.

        Raises:
            MeasureError: When the engine cannot read it.
            ValueError: When a noise integral's band does not lie within the
                frequencies, as `results.integrate` raises.
        """
        try:
            value = _engine(self.measure, result)
        except _core.MeasureError as error:
            raise MeasureError(self.explain(error)) from None
        return Reading(value, self.unit)


def _engine(measure: _core.Measure, result: results.AnalysisResult) -> float:
    """What the engine's `measure` reads of the arrays of `result`.

    Raises:
        MeasureError: When it cannot read them.
    """
    match result:
        case results.OperatingPointResult():
            value = measure.read_operating_point(_real(result.values))
        case results.OperatingPointsResult():
            value = measure.read_operating_points(len(result.points))
        case results.DcSweepResult():
            value = measure.read_dc_sweep(_real(result.levels), _real(result.values))
        case results.TransientResult():
            value = measure.read_transient(_real(result.times), _real(result.values))
        case results.AcResult():
            value = measure.read_ac(_real(result.frequencies), _complex(result.values))
        case results.StabilityResult():
            value = measure.read_stability(
                _real(result.frequencies), _complex(result.gains)
            )
        case results.NoiseResult():
            gains = None if result.input is None else _complex(result.input.gains)
            value = measure.read_noise(
                _real(result.frequencies), _real(result.densities), gains
            )
        case results.PoleZeroResult():
            value = measure.read_pole_zero(_complex(result.poles))
        case results.SensitivityResult() | results.TransferFunctionResult():
            # No measure reads sensitivities or a transfer function, so none is resolved
            # against them.
            message = f"the engine measures no {type(result).__name__}"
            raise MeasureError(message)
    return value


def _real(values: np.typing.ArrayLike) -> results.Real:
    return np.asarray(values, dtype=np.float64)


def _complex(values: np.typing.ArrayLike) -> results.Complex:
    return np.asarray(values, dtype=np.complex128)


@dataclass(frozen=True, slots=True)
class Measure:
    """A number read from the result of one named analysis."""

    analysis: str
    description: str
    """What the measure reads, such as ``gain of out at 1 kHz``."""
    plain: Plain | None
    """The unit of a measure that reads a plain number; None for a quantity."""
    read: Callable[[results.AnalysisResult], Reading] = field(repr=False, compare=False)
    """Reads the measure from the analysis's result; raises a `JoltyError`, or a
    `KeyError` for an unknown the result lacks, when it cannot."""
    native: Callable[[results.AnalysisResult], Native] | None = field(
        default=None, repr=False, compare=False
    )
    """Resolves the measure for the engine to read against a result, from no more
    than the result's kind and unknowns, a sweep's source and quantity, and a noise
    analysis's output's and source's quantities; raises as `read` does when it cannot.
    None for a measure only Python reads, such as one a plan script writes."""

    def __ge__(self, minimum: Bound) -> Requirement:
        """The requirement that the measure be at least `minimum`."""
        return Requirement(self, minimum=_quantity(minimum), source=caller())

    def __le__(self, maximum: Bound) -> Requirement:
        """The requirement that the measure be at most `maximum`."""
        return Requirement(self, maximum=_quantity(maximum), source=caller())

    def between(self, minimum: Bound, maximum: Bound) -> Requirement:
        """The requirement that the measure lie between `minimum` and `maximum`."""
        return Requirement(
            self, _quantity(minimum), _quantity(maximum), source=caller()
        )

    def within(self, target: Bound, tolerance: Bound) -> Requirement:
        """The requirement that the measure lie within `tolerance` of `target`.

        Raises:
            DimensionError: When the target and the tolerance differ in dimension.
        """
        centre, spread = _quantity(target), _quantity(tolerance)
        return Requirement(self, centre - spread, centre + spread, source=caller())

    @override
    def __str__(self) -> str:
        """What the measure reads, and in which analysis."""
        return f"{self.description} in {self.analysis}"


@dataclass(frozen=True, slots=True)
class Requirement:
    """A measure and the bounds it must lie within.

    A requirement's slack is how far the value read lies inside its bounds, negative
    outside them. It passes when its slack is at least its margin, is marginal when
    its slack is less, and fails when its slack is negative.
    """

    measure: Measure
    minimum: Quantity | None = None
    maximum: Quantity | None = None
    margin: Quantity | None = None
    source: ir.Source | None = field(default=None, compare=False)
    """The script line that stated the requirement."""

    def __post_init__(self) -> None:
        """Checks the bounds and the margin.

        Raises:
            ValueError: When there is no bound, the minimum exceeds the maximum, or the
                margin is negative.
        """
        if self.minimum is None and self.maximum is None:
            message = f"a requirement on {self.measure} states a bound"
            raise ValueError(message)
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum.value > self.maximum.value
        ):
            message = (
                f"the minimum of {self.measure}, {self._text(self.minimum)}, "
                f"exceeds its maximum, {self._text(self.maximum)}"
            )
            raise ValueError(message)
        if self.margin is not None and self.margin.value < 0:
            message = f"the margin of {self.measure} is negative: {self.margin}"
            raise ValueError(message)

    def with_margin(self, margin: Bound) -> Requirement:
        """The requirement, marginal when it passes by less than `margin`."""
        return replace(self, margin=_quantity(margin))

    def judge(self, reading: Reading) -> tuple[Status, float]:
        """Whether `reading` passes, is marginal or fails, and its slack.

        Raises:
            DimensionError: When a bound or the margin is not in the reading's unit.
        """
        slack = math.inf
        if self.minimum is not None:
            slack = reading.value - _against(self.minimum, reading, "the minimum")
        if self.maximum is not None:
            slack = min(
                slack, _against(self.maximum, reading, "the maximum") - reading.value
            )
        margin = (
            0.0 if self.margin is None else _against(self.margin, reading, "the margin")
        )
        if slack < 0:
            return "failed", slack
        return ("marginal" if slack < margin else "passed"), slack

    def _text(self, bound: Quantity) -> str:
        plain = self.measure.plain
        if plain is not None and bound.dimension == DIMENSIONLESS:
            return _format(bound.value, plain)
        return str(bound)

    @override
    def __str__(self) -> str:
        """The requirement, such as ``gain of out at 1 kHz in bode ≥ -3 dB``."""
        bounds = [
            f"{relation} {self._text(bound)}"
            for relation, bound in (("≥", self.minimum), ("≤", self.maximum))
            if bound is not None
        ]
        margin = (
            ""
            if self.margin is None
            else f" with a margin of {self._text(self.margin)}"
        )
        return f"{self.measure} {' and '.join(bounds)}{margin}"


@dataclass(frozen=True, slots=True)
class Specs:
    """The requirements a circuit must meet, in the order a report lists them.

    The `yield_target` is the fraction of a Monte Carlo analysis's samples that must
    meet every requirement, such as 0.99.
    """

    requirements: Sequence[Requirement]
    yield_target: float | None = None

    def __post_init__(self) -> None:
        """Checks that there are requirements, only requirements, and a fair target.

        Raises:
            ValueError: When there is no requirement, or the yield target is not above
                0 and at most 1.
            TypeError: When one is not a requirement.
        """
        if self.yield_target is not None and not 0 < self.yield_target <= 1:
            message = (
                f"a yield target lies above 0 and at most 1, not {self.yield_target}"
            )
            raise ValueError(message)
        if not self.requirements:
            message = "specs state at least one requirement"
            raise ValueError(message)
        for requirement in self.requirements:
            if not isinstance(requirement, Requirement):
                message = f"specs hold requirements, not {requirement!r}"
                raise TypeError(message)


class Verdict(BaseModel):
    """What became of one requirement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    requirement: str = Field(description="The requirement, in words.")
    status: Status
    value: ExtendedReal | None = Field(
        default=None, description="The value read, in SI base units; null when unread."
    )
    unit: str | None = Field(
        default=None, description="The value's unit, such as 'V' or 'dB'."
    )
    slack: ExtendedReal | None = Field(
        default=None,
        description="How far the value lies inside the bounds, negative outside them.",
    )
    corner: str | None = Field(
        default=None,
        description="The corner of the plan the verdict comes from; null for the "
        "plan's own conditions.",
    )
    reason: str | None = Field(
        default=None, description="Why the requirement is unmeasured."
    )
    source: ir.Source | None = Field(
        default=None, description="The script line that stated the requirement."
    )

    def describe(self) -> str:
        """The verdict, then the script line that stated the requirement."""
        at = "" if self.corner is None else f" at the corner {self.corner}"
        if self.value is None or self.unit is None or self.slack is None:
            head = f"unmeasured: {self.requirement}: {self.reason}{at}"
        else:
            read = _format(self.value, self.unit)
            slack = _format(self.slack, self.unit)
            head = f"{self.status}: {self.requirement}: read {read}{at}, slack {slack}"
        return head if self.source is None else f"{head}\n  = {self.source}"


class Values(BaseModel):
    """The values a requirement read over samples, in SI base units."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    unit: str = Field(description="The values' unit, such as 'V' or 'dB'.")
    minimum: ExtendedReal
    maximum: ExtendedReal
    mean: ExtendedReal | None = Field(
        description="The values' mean; null when it is not a number, as when some "
        "values are infinities of either sign."
    )
    deviation: float | None = Field(
        description="The values' standard deviation; null when some value is infinite."
    )

    @classmethod
    def of(cls, numbers: npt.NDArray[np.float64], unit: str) -> Values:
        """The least, the greatest, the mean and the deviation of `numbers`.

        An infinity among them is their mean, unless one of each sign is, and leaves
        their deviation undefined.
        """
        finite = np.isfinite(numbers)
        infinities = {float(number) for number in numbers[~finite]}
        if finite.all():
            mean, deviation = float(numbers.mean()), float(numbers.std())
        else:
            mean = infinities.pop() if len(infinities) == 1 else None
            deviation = None
        return cls(
            unit=unit,
            minimum=float(numbers.min()),
            maximum=float(numbers.max()),
            mean=mean,
            deviation=deviation,
        )


class Spread(BaseModel):
    """What one requirement read over a Monte Carlo analysis's samples."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    requirement: str = Field(description="The requirement, in words.")
    measured: int = Field(description="Samples on which it was measured.")
    passing: int = Field(description="Samples on which it passed, if marginally.")
    values: Values | None = Field(
        default=None, description="The values read; null when none was."
    )

    def describe(self, samples: int) -> str:
        """The spread as one line, over `samples` samples."""
        head = f"{self.requirement}: {self.passing} of {samples} pass"
        if self.values is None:
            return f"{head}, none measured"
        unmeasured = samples - self.measured
        missing = f", {unmeasured} unmeasured" if unmeasured else ""
        unit = self.values.unit
        low, high = (
            _format(self.values.minimum, unit),
            _format(self.values.maximum, unit),
        )
        mean, deviation = (
            "undefined" if value is None else _format(value, unit)
            for value in (self.values.mean, self.values.deviation)
        )
        return (
            f"{head}{missing}; read {low} to {high}, mean {mean}, deviation {deviation}"
        )


class Yield(BaseModel):
    """How many of a Monte Carlo analysis's samples meet every requirement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    samples: int
    passing: int = Field(description="Samples on which every requirement passed.")
    estimate: float = Field(description="The fraction of samples that pass.")
    low: float = Field(description="The 95 % Wilson interval's lower end.")
    high: float = Field(description="The 95 % Wilson interval's upper end.")
    target: float | None = Field(
        default=None, description="The specs' yield target; null for none."
    )

    @property
    def met(self) -> bool:
        """Whether the estimate reaches the target, or there is no target."""
        return self.target is None or self.estimate >= self.target

    def describe(self) -> str:
        """The yield as one line."""
        head = (
            f"yield: {self.passing} of {self.samples}, {100 * self.estimate:.4g} % "
            f"(95 % interval {100 * self.low:.4g} to {100 * self.high:.4g} %)"
        )
        if self.target is None:
            return head
        reached = "reaches" if self.met else "is below"
        return f"{head}, which {reached} the target {100 * self.target:.4g} %"


def spread(
    specs: Specs, sampled: Sequence[Sequence[Verdict]]
) -> tuple[tuple[Spread, ...], Yield]:
    """Each requirement's spread, and the yield, over the verdicts on each sample.

    `sampled` holds, for each sample, the verdict on each requirement in order. A sample
    passes when every requirement passes, if marginally.
    """
    spreads = tuple(
        _spread(requirement, [verdicts[index] for verdicts in sampled])
        for index, requirement in enumerate(specs.requirements)
    )
    passing = sum(all(_passes(verdict) for verdict in verdicts) for verdicts in sampled)
    low, high = wilson(passing, len(sampled))
    return spreads, Yield(
        samples=len(sampled),
        passing=passing,
        estimate=passing / len(sampled),
        low=low,
        high=high,
        target=specs.yield_target,
    )


def _spread(requirement: Requirement, verdicts: Sequence[Verdict]) -> Spread:
    read = [
        (verdict.value, verdict.unit)
        for verdict in verdicts
        if verdict.value is not None and verdict.unit is not None
    ]
    passing = sum(_passes(verdict) for verdict in verdicts)
    stated = Spread(requirement=str(requirement), measured=len(read), passing=passing)
    if not read:
        return stated
    values = Values.of(np.array([value for value, _ in read]), read[0][1])
    return stated.model_copy(update={"values": values})


def _passes(verdict: Verdict) -> bool:
    return verdict.status in {"passed", "marginal"}


@dataclass(frozen=True, slots=True)
class Judged:
    """A requirement's verdicts on many runs, as columns, one entry per run.

    Each run's `severity` is its status's, 0 failed, 1 unmeasured, 2 marginal and 3
    passed, with its `slack`, infinite where unmeasured, its `value`, not a number
    where unmeasured, and, among `units`, the unit of the values of the conditions
    each came from, by `origin`.
    """

    severity: npt.NDArray[np.int8]
    slack: results.Real
    value: results.Real
    origin: npt.NDArray[np.intp]
    units: tuple[str, ...]


def judged(
    requirement: Requirement,
    planned: Collection[str],
    batch: results.BatchResult | None,
    place: int,
    runs: int,
) -> Judged:
    """`requirement`'s verdict on each of `runs` runs of `batch`, as `evaluate` says.

    `planned` names every analysis of the plan, and the measure at `place` among
    the batch's is the requirement's; with no batch, the circuit does not elaborate,
    and no analysis ends on any run.
    """
    severity = np.full(runs, _SEVERITY["unmeasured"], dtype=np.int8)
    slack = np.full(runs, math.inf)
    value = np.full(runs, math.nan)
    unit = ""
    found = (
        None if batch is None or requirement.measure.analysis not in planned else place
    )
    if batch is not None and found is not None:
        read = batch.measured[:, found]
        measured = (batch.causes[:, found] < 0) & ~np.isnan(read)
        given = batch.units[found]
        if given is not None and measured.any():
            reading = Reading(0.0, given)
            try:
                requirement.judge(reading)
            except DimensionError:
                measured[:] = False
            unit = "" if given == DIMENSIONLESS else str(given)
            values = read[measured]
            room = np.full(values.shape, math.inf)
            if requirement.minimum is not None:
                room = values - requirement.minimum.value
            if requirement.maximum is not None:
                other = requirement.maximum.value - values
                room = np.where(other < room, other, room)
            margin = 0.0 if requirement.margin is None else requirement.margin.value
            severity[measured] = np.where(
                room < 0,
                _SEVERITY["failed"],
                np.where(room < margin, _SEVERITY["marginal"], _SEVERITY["passed"]),
            )
            slack[measured] = room
            value[measured] = values
    return Judged(severity, slack, value, np.zeros(runs, dtype=np.intp), (unit,))


def worst(conditions: Sequence[Judged]) -> Judged:
    """The worst verdict on each run over `conditions`, as `judge` picks it.

    The worst is failed, then unmeasured, then marginal, then passed, and the least
    slack among equals, the first condition among those alike.

    Raises:
        ValueError: When there is no condition.
    """
    if not conditions:
        message = "the worst verdict is of a condition or more"
        raise ValueError(message)
    first, *rest = conditions
    severity, slack, value = (
        first.severity.copy(),
        first.slack.copy(),
        first.value.copy(),
    )
    origin = np.zeros(severity.shape, dtype=np.intp)
    for place, each in enumerate(rest, start=1):
        worse = (each.severity < severity) | (
            (each.severity == severity) & (each.slack < slack)
        )
        severity[worse] = each.severity[worse]
        slack[worse] = each.slack[worse]
        value[worse] = each.value[worse]
        origin[worse] = place
    units = tuple(
        each.units[int(each.origin[0])] if each.units else "" for each in conditions
    )
    return Judged(severity, slack, value, origin, units)


def spread_of(
    specs: Specs, judged: Sequence[Judged]
) -> tuple[tuple[Spread, ...], Yield]:
    """Each requirement's spread, and the yield, as `spread` states them, of `judged`.

    `judged` holds each requirement's verdicts on every sample, in the specs' order.
    """
    spreads = tuple(
        _spread_of(requirement, each)
        for requirement, each in zip(specs.requirements, judged, strict=True)
    )
    samples = int(judged[0].severity.shape[0]) if judged else 0
    passing = (
        int(
            np.all(
                [each.severity >= _SEVERITY["marginal"] for each in judged], axis=0
            ).sum()
        )
        if judged
        else samples
    )
    low, high = wilson(passing, samples)
    return spreads, Yield(
        samples=samples,
        passing=passing,
        estimate=passing / samples,
        low=low,
        high=high,
        target=specs.yield_target,
    )


def _spread_of(requirement: Requirement, judged: Judged) -> Spread:
    measured = ~np.isnan(judged.value)
    passing = int((judged.severity >= _SEVERITY["marginal"]).sum())
    stated = Spread(
        requirement=str(requirement), measured=int(measured.sum()), passing=passing
    )
    if not measured.any():
        return stated
    first = int(np.argmax(measured))
    unit = judged.units[int(judged.origin[first])]
    values = Values.of(judged.value[measured], unit)
    return stated.model_copy(update={"values": values})


_Z: Final = 1.959963984540054
"""The normal distribution's 97.5th percentile: a 95 % interval spans ±_Z deviations."""


def wilson(passing: int, samples: int) -> tuple[float, float]:
    """The 95 % Wilson score interval of a proportion of `passing` out of `samples`.

    It is the interval a [`Yield`][jolty.specs.Yield] states (E. B. Wilson,
    "Probable inference, the law of succession, and statistical inference", Journal
    of the American Statistical Association 22, 1927), whose coverage stays near 95 %
    for few samples and for proportions near 0 or 1, where the normal approximation's
    falls short (L. D. Brown, T. T. Cai and A. DasGupta, "Interval estimation for a
    binomial proportion", Statistical Science 16, 2001). The lower end,
    `(p + w/2 - z·s) / (1 + w)` with `w = z²/n`, is computed in its conjugate form
    `p² / (p + w/2 + z·s)`, which subtracts nothing: it is exactly 0 when nothing
    passes, where the textbook form leaves rounding. The interval is symmetric, so its
    upper end is one less the lower end of the proportion that fails, exactly 1 when
    everything passes.

    Examples:
        >>> low, high = wilson(37, 200)
        >>> f"{low:.4f} to {high:.4f}"
        '0.1373 to 0.2446'
        >>> wilson(0, 10)[0], wilson(10, 10)[1]
        (0.0, 1.0)

    Raises:
        ValueError: When there is no sample, or `passing` is not from 0 to `samples`.
    """
    if samples < 1:
        message = f"a proportion is of one sample or more, not {samples}"
        raise ValueError(message)
    if not 0 <= passing <= samples:
        message = f"{passing} of {samples} samples is no proportion"
        raise ValueError(message)

    def lower(estimate: float) -> float:
        weight = _Z * _Z / samples
        spread = _Z * math.sqrt(
            estimate * (1 - estimate) / samples + weight / (4 * samples)
        )
        return estimate * estimate / (estimate + weight / 2 + spread)

    return lower(passing / samples), 1 - lower((samples - passing) / samples)


def describe(verdicts: Sequence[Verdict]) -> str:
    """The verdicts as text: each verdict, then how many have each status."""
    counts = Counter(verdict.status for verdict in verdicts)
    tally = ", ".join(
        f"{counts[status]} {status}" for status in _STATUSES if counts[status]
    )
    return "\n\n".join(
        [*(verdict.describe() for verdict in verdicts), f"specs: {tally}"]
    )


def evaluate(
    specs: Specs,
    planned: Collection[str],
    solved: Mapping[str, results.AnalysisResult],
    corners: Mapping[str, Mapping[str, results.AnalysisResult]] | None = None,
) -> tuple[Verdict, ...]:
    """The verdict on each requirement, the worst of the plan's own run and its corners.

    `planned` names every analysis of the plan; one planned but not solved failed.
    `solved` holds the results of the plan's own run by analysis, and `corners` the
    results of each corner's run, by corner then by analysis.
    """
    runs = [(None, solved), *(corners or {}).items()]
    return tuple(
        judge(requirement, planned, runs) for requirement in specs.requirements
    )


def judge(
    requirement: Requirement,
    planned: Collection[str],
    runs: Iterable[tuple[str | None, Mapping[str, results.AnalysisResult]]],
) -> Verdict:
    """The worst verdict on `requirement` over `runs`.

    Each run holds the results of the analyses that ended, by name, and is named by its
    corner, None for the plan's own conditions; `planned` names every analysis.

    Raises:
        ValueError: When there is no run.
    """
    verdicts = [_verdict(requirement, planned, found, corner) for corner, found in runs]
    if not verdicts:
        message = f"{requirement} is judged on a run or more"
        raise ValueError(message)
    return min(verdicts, key=_badness)


def reading(
    requirement: Requirement, found: Mapping[str, results.AnalysisResult]
) -> float | None:
    """The value `requirement` reads of the results `found`; None when it cannot."""
    verdict = _verdict(requirement, found.keys(), found, None)
    return verdict.value


def _badness(verdict: Verdict) -> tuple[int, float]:
    """How bad a verdict is, the worst least: by status, then by slack."""
    slack = math.inf if verdict.slack is None else verdict.slack
    return _SEVERITY[verdict.status], slack


def _verdict(
    requirement: Requirement,
    planned: Collection[str],
    solved: Mapping[str, results.AnalysisResult],
    corner: str | None,
) -> Verdict:
    stated = Verdict(
        requirement=str(requirement),
        status="unmeasured",
        source=requirement.source,
        corner=corner,
    )
    analysis = requirement.measure.analysis
    if analysis not in planned:
        reason = f"the plan runs no analysis named {analysis}"
        return stated.model_copy(update={"reason": reason})
    if analysis not in solved:
        reason = f"the analysis {analysis} failed"
        return stated.model_copy(update={"reason": reason})
    try:
        reading = requirement.measure.read(solved[analysis])
        if math.isnan(reading.value):
            message = "the value read is not a number"
            raise MeasureError(message)
        status, slack = requirement.judge(reading)
    except (JoltyError, KeyError) as error:
        return stated.model_copy(update={"reason": str(error.args[0])})
    update = {"status": status, "value": reading.value, "slack": slack}
    # A dimensionless reading, such as a device's state, is a plain number.
    unit = "" if reading.unit == DIMENSIONLESS else str(reading.unit)
    return stated.model_copy(update={**update, "unit": unit})


def _quantity(bound: Bound) -> Quantity:
    """`bound` as a quantity: a number is a plain one.

    Raises:
        ValueError: When the bound is infinite or not a number.
    """
    quantity = bound if isinstance(bound, Quantity) else Quantity(bound, DIMENSIONLESS)
    finite(quantity.value, "a bound")
    return quantity


def _against(bound: Quantity, reading: Reading, name: str) -> float:
    """`bound` in SI base units, after checking it is in the unit of `reading`.

    Raises:
        DimensionError: When it is not.
    """
    if bound.dimension != reading.dimension:
        message = (
            f"{name}, {bound}, is not in {reading.unit}, the unit of the value read"
        )
        raise DimensionError(message)
    return bound.value


def _format(value: float, unit: str) -> str:
    """`value` in `unit`, to four significant digits, with an SI prefix if any."""
    if unit in {"dB", "°", "%"}:
        return f"{value:.4g} {unit}"
    if not unit:
        return f"{value:.4g}"
    return format_si(value, unit)
