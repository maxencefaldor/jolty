# Copyright 2026 Maxence Faldor
"""Specification requirements, verdicts, and Monte Carlo yield spreads.

Implements the core data models described in [`jolty.specs`][jolty.specs]: measures
compared with bounds, per-condition verdicts, and yield statistics. Across batched
runs, [`Judged`][jolty.specs.Judged] stores verdicts in columnar arrays with one entry
per run matching single-run evaluation.
"""

__lazy_modules__ = ["jolty.builder", "numpy", "numpy.typing"]

import math
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Annotated, Final, Literal, override

import numpy as np
import numpy.typing as npt
from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    ValidationInfo,
    WithJsonSchema,
    field_validator,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic_core import InitErrorDetails

from jolty import _core, ir, results
from jolty.builder import caller
from jolty.errors import (
    DimensionError,
    JoltyError,
    close_names,
    detail,
    fail,
    raise_if_errors,
)
from jolty.units import (
    DIMENSIONLESS,
    Dimension,
    Quantity,
    finite,
    format_si,
)

type Bound = Quantity | float
"""A bound: a quantity, or a number for a measure in dB, degrees or percent."""

type Edge = Literal["rising", "falling", "either"]
"""Direction in which a signal crosses a threshold."""

type Status = Literal["passed", "marginal", "failed", "unmeasured"]
"""Evaluation outcome for a specification requirement."""

_STATUSES: Final[tuple[Status, ...]] = ("passed", "marginal", "failed", "unmeasured")

_SEVERITY: Final[dict[Status, int]] = {
    "failed": 0,
    "unmeasured": 1,
    "marginal": 2,
    "passed": 3,
}
"""Severity rank for each status, worst first, used to pick the worst-corner verdict."""


_CONFIG: Final = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
"""Shared Pydantic configuration for spec dataclasses."""

_REQUIREMENT_CONFIG: Final = ConfigDict(
    extra="forbid", arbitrary_types_allowed=True, validate_default=True
)
"""Pydantic configuration for `Requirement`, validating omitted default bounds."""


def _format_bound(measure: Measure | None, bound: Quantity) -> str:
    """Formats `bound` in `measure`'s plain unit when applicable."""
    plain = None if measure is None else measure.plain
    if plain is not None and bound.dimension == DIMENSIONLESS:
        return format_reading(bound.value, plain)
    return str(bound)


def _check_bound_quantity(value: object, name: str) -> Quantity | None:
    """Validates `value` is `None` or a finite `Quantity`."""
    if value is None:
        return None
    if not isinstance(value, Quantity):
        fail("quantity_type", f"{name} is a Quantity, not {value!r}")
    if not math.isfinite(value.value):
        fail("finite", f"{name} is finite, not {value}")
    return value


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


type _Plain = results.Plain
type _CoreMeasure = _core.Measure
type _MeasureExplainer = Callable[[_core.MeasureError], str]
type _MeasureReader = Callable[[results.AnalysisResult], Reading]
type _MeasureResolver = Callable[[results.AnalysisResult], ResolvedMeasure]
type _MeasureDiagnoser = Callable[[results.AnalysisResult], Sequence[str]]
type _StatusArray = npt.NDArray[np.int8]
type _RealArray = results.Real
type _OriginArray = npt.NDArray[np.intp]


@dataclass(frozen=True, slots=True)
class Reading:
    """Single measured scalar value in SI base units paired with its unit."""

    value: float
    unit: Dimension | _Plain

    @property
    def dimension(self) -> Dimension:
        """Physical dimension of the reading, or `DIMENSIONLESS` for plain numbers."""
        return self.unit if isinstance(self.unit, Dimension) else DIMENSIONLESS


@dataclass(frozen=True, slots=True)
class RealCotangent:
    """Sparse reverse-mode derivative on a real unknown table and optional period.

    Each entry in `entries` is ``(point, unknown, weight)`` in ascending order,
    matching `OperatingPointResult`, `DcSweepResult`, `TransientResult`, or
    `PeriodicSteadyStateResult`. `period` is the derivative with respect to an
    autonomous oscillator's period in seconds.
    """

    entries: tuple[tuple[int, int, float], ...]
    period: float = 0.0

    def scale(self, factor: float) -> RealCotangent:
        """Returns a copy with every entry and `period` scaled by `factor`."""
        if factor == 0.0:
            return RealCotangent((), 0.0)
        return RealCotangent(
            tuple(
                (point, unknown, weight * factor)
                for point, unknown, weight in self.entries
            ),
            self.period * factor,
        )


@dataclass(frozen=True, slots=True)
class ComplexCotangent:
    """Sparse bilinear complex derivative on an AC response's unknown phasors.

    Each entry in `entries` is ``(point, unknown, real, imaginary)`` following the
    engine's bilinear convention ``v = dL/da - j dL/db``, so ``Re(v * dz) = dL``.
    """

    entries: tuple[tuple[int, int, float, float], ...]

    def scale(self, factor: float) -> ComplexCotangent:
        """Returns a copy with every complex entry scaled by `factor`."""
        if factor == 0.0:
            return ComplexCotangent(())
        return ComplexCotangent(
            tuple(
                (point, unknown, real * factor, imaginary * factor)
                for point, unknown, real, imaginary in self.entries
            )
        )


@dataclass(frozen=True, slots=True)
class LoopCotangent:
    """Sparse bilinear complex derivative on a stability analysis's loop gains.

    Each entry in `entries` is ``(point, real, imaginary)`` in ascending `point`
    order using ``v = dL/da - j dL/db``.
    """

    entries: tuple[tuple[int, float, float], ...]

    def scale(self, factor: float) -> LoopCotangent:
        """Returns a copy with every complex entry scaled by `factor`."""
        if factor == 0.0:
            return LoopCotangent(())
        return LoopCotangent(
            tuple(
                (point, real * factor, imaginary * factor)
                for point, real, imaginary in self.entries
            )
        )


@dataclass(frozen=True, slots=True)
class NoiseCotangent:
    """Sparse derivative on a noise analysis's output and input-referred densities.

    `densities` stores ``(point, weight)`` entries with respect to output power
    spectral densities (`NOISE_DENSITY`), and `referred` stores ``(point, weight)``
    entries with respect to input-referred power spectral densities (`REFERRED_NOISE`).
    """

    densities: tuple[tuple[int, float], ...]
    referred: tuple[tuple[int, float], ...]

    def scale(self, factor: float) -> NoiseCotangent:
        """Returns a copy with every density and referred entry scaled by `factor`."""
        if factor == 0.0:
            return NoiseCotangent((), ())
        return NoiseCotangent(
            tuple((point, weight * factor) for point, weight in self.densities),
            tuple((point, weight * factor) for point, weight in self.referred),
        )


@dataclass(frozen=True, slots=True)
class PolesCotangent:
    """Sparse bilinear complex derivative on a pole-zero analysis's poles vector.

    Each entry in `entries` is ``(pole_index, real, imaginary)`` in ascending order
    using ``v = dL/da - j dL/db``.
    """

    entries: tuple[tuple[int, float, float], ...]

    def scale(self, factor: float) -> PolesCotangent:
        """Returns a copy with every pole entry scaled by `factor`."""
        if factor == 0.0:
            return PolesCotangent(())
        return PolesCotangent(
            tuple(
                (point, real * factor, imaginary * factor)
                for point, real, imaginary in self.entries
            )
        )


type MeasureCotangent = (
    RealCotangent | ComplexCotangent | LoopCotangent | NoiseCotangent | PolesCotangent
)
"""Sparse reverse-mode derivative of a scalar measure or requirement slack."""


@dataclass(frozen=True, slots=True)
class ResolvedMeasure:
    """Engine-ready measure resolved against one result kind.

    Stores unknown indices and SI-unit parameters in `measure` so the Rust engine
    evaluates the reading directly without Python callbacks.
    """

    measure: _CoreMeasure
    unit: Dimension | _Plain
    """Unit of the value computed by the engine."""
    explain: _MeasureExplainer = field(repr=False, compare=False)
    """Formatter converting an engine `MeasureError` into a human-readable message."""

    def read(self, result: results.AnalysisResult) -> Reading:
        """Evaluates this resolved measure on `result` using the Rust engine.

        Args:
            result: Solved analysis result whose arrays are evaluated.

        Returns:
            Measured scalar value paired with `unit`.

        Raises:
            MeasureError: When the engine cannot compute the measure on `result`.
            ValueError: When a noise integration band lies outside the frequency range.
        """
        try:
            value = _engine(self.measure, result)
        except _core.MeasureError as error:
            raise MeasureError(self.explain(error)) from None
        return Reading(value, self.unit)

    def cotangent(
        self, result: results.AnalysisResult
    ) -> tuple[Reading, MeasureCotangent]:
        """Evaluates this resolved measure on `result` with its exact sparse cotangent.

        Args:
            result: Solved analysis result whose arrays are differentiated.

        Returns:
            Pair of measured `Reading` and sparse `MeasureCotangent`.

        Raises:
            MeasureError: When the measure cannot be evaluated or is not
                differentiable at `result`.
            ValueError: When a noise integration band lies outside the frequency range.
        """
        try:
            value, cotangent = _engine_cotangent(self.measure, result)
        except _core.MeasureError as error:
            raise MeasureError(self.explain(error)) from None
        return Reading(value, self.unit), cotangent


def _engine_real(
    measure: _core.Measure,
    result: results.OperatingPointResult
    | results.OperatingPointsResult
    | results.DcSweepResult
    | results.TransientResult,
) -> float:
    """Evaluates `measure` on a real-valued analysis result."""
    match result:
        case results.OperatingPointResult():
            return measure.read_operating_point(_real(result.values))
        case results.OperatingPointsResult():
            return measure.read_operating_points(len(result.points))
        case results.DcSweepResult():
            return measure.read_dc_sweep(_real(result.swept), _real(result.values))
        case results.PeriodicSteadyStateResult():
            return measure.read_periodic_steady_state(
                _real(result.times), _real(result.values), result.period
            )
        case results.TransientResult():
            return measure.read_transient(_real(result.times), _real(result.values))


def _engine(measure: _core.Measure, result: results.AnalysisResult) -> float:
    """Invokes the Rust engine evaluator for `measure` on `result` arrays.

    Raises:
        MeasureError: When `result` cannot be evaluated by the engine.
    """
    match result:
        case (
            results.OperatingPointResult()
            | results.OperatingPointsResult()
            | results.DcSweepResult()
            | results.TransientResult()
        ):
            return _engine_real(measure, result)
        case results.AcResult():
            return measure.read_ac(_real(result.frequencies), _complex(result.values))
        case results.StabilityResult():
            return measure.read_stability(
                _real(result.frequencies), _complex(result.gains)
            )
        case results.NoiseResult():
            gains = None if result.input is None else _complex(result.input.gains)
            return measure.read_noise(
                _real(result.frequencies), _real(result.densities), gains
            )
        case results.PoleZeroResult():
            return measure.read_pole_zero(_complex(result.poles))
        case (
            results.SensitivityResult()
            | results.MismatchResult()
            | results.TransferFunctionResult()
        ):
            # No measure reads sensitivities, a mismatch or a transfer function, so none
            # is resolved against them.
            message = f"the engine measures no {type(result).__name__}"
            raise MeasureError(message)


def _engine_cotangent_real(
    measure: _core.Measure,
    result: results.OperatingPointResult
    | results.OperatingPointsResult
    | results.DcSweepResult
    | results.TransientResult,
) -> tuple[float, RealCotangent]:
    """Evaluates `measure` and its real cotangent on a real-valued analysis result."""
    match result:
        case results.OperatingPointResult():
            value, entries, period = measure.cotangent_operating_point(
                _real(result.values)
            )
        case results.OperatingPointsResult():
            value, entries, period = measure.cotangent_operating_points(
                len(result.points)
            )
        case results.DcSweepResult():
            value, entries, period = measure.cotangent_dc_sweep(
                _real(result.swept), _real(result.values)
            )
        case results.PeriodicSteadyStateResult():
            value, entries, period = measure.cotangent_periodic_steady_state(
                _real(result.times), _real(result.values), result.period
            )
        case results.TransientResult():
            value, entries, period = measure.cotangent_transient(
                _real(result.times), _real(result.values)
            )
    return value, RealCotangent(tuple(entries), period)


def _engine_cotangent(
    measure: _core.Measure, result: results.AnalysisResult
) -> tuple[float, MeasureCotangent]:
    """Invokes the Rust engine cotangent evaluator for `measure` on `result` arrays.

    Raises:
        MeasureError: When `result` cannot be evaluated by the engine.
    """
    match result:
        case (
            results.OperatingPointResult()
            | results.OperatingPointsResult()
            | results.DcSweepResult()
            | results.TransientResult()
        ):
            return _engine_cotangent_real(measure, result)
        case results.AcResult():
            value, complex_entries = measure.cotangent_ac(
                _real(result.frequencies), _complex(result.values)
            )
            return value, ComplexCotangent(tuple(complex_entries))
        case results.StabilityResult():
            value, loop_entries = measure.cotangent_stability(
                _real(result.frequencies), _complex(result.gains)
            )
            return value, LoopCotangent(tuple(loop_entries))
        case results.NoiseResult():
            gains = None if result.input is None else _complex(result.input.gains)
            value, densities, referred = measure.cotangent_noise(
                _real(result.frequencies), _real(result.densities), gains
            )
            return value, NoiseCotangent(tuple(densities), tuple(referred))
        case results.PoleZeroResult():
            value, pole_entries = measure.cotangent_pole_zero(_complex(result.poles))
            return value, PolesCotangent(tuple(pole_entries))
        case (
            results.SensitivityResult()
            | results.MismatchResult()
            | results.TransferFunctionResult()
        ):
            message = f"the engine measures no {type(result).__name__}"
            raise MeasureError(message)


def _real(values: np.typing.ArrayLike) -> results.Real:
    return np.asarray(values, dtype=np.float64)


def _complex(values: np.typing.ArrayLike) -> results.Complex:
    return np.asarray(values, dtype=np.complex128)


@dataclass(frozen=True, slots=True)
class Measure:
    """Scalar quantity extracted from the result of one named analysis."""

    analysis: str
    description: str
    """Human-readable label of the measured value, such as ``gain of out at 1 kHz``."""
    plain: _Plain | None
    """Display unit for plain measures (such as ``dB``), or None for quantities."""
    read: _MeasureReader = field(repr=False, compare=False)
    """Evaluator extracting a `Reading` from an analysis result (raises on failure)."""
    resolved: _MeasureResolver | None = field(default=None, repr=False, compare=False)
    """Optional resolver building an engine `ResolvedMeasure` from result metadata.

    None for Python-only custom measures defined by user scripts.
    """
    diagnose: _MeasureDiagnoser | None = field(default=None, repr=False, compare=False)
    """Optional static validator collecting signal and dimension errors against a shape.

    None when validation relies solely on `resolved` or Python `read`.
    """

    def __ge__(self, minimum: Bound) -> Requirement:
        """Constructs a lower-bound requirement asserting measure >= `minimum`."""
        return Requirement(self, minimum=_quantity(minimum), source_location=caller())

    def __le__(self, maximum: Bound) -> Requirement:
        """Constructs an upper-bound requirement asserting measure <= `maximum`."""
        return Requirement(self, maximum=_quantity(maximum), source_location=caller())

    def between(self, minimum: Bound, maximum: Bound) -> Requirement:
        """Constructs a two-sided requirement on `[minimum, maximum]`.

        Args:
            minimum: Inclusive lower bound for the measured value.
            maximum: Inclusive upper bound for the measured value.

        Returns:
            Two-sided requirement constraining this measure to `[minimum, maximum]`.
        """
        return Requirement(
            self, _quantity(minimum), _quantity(maximum), source_location=caller()
        )

    def within(self, target: Bound, tolerance: Bound) -> Requirement:
        """Constructs a requirement bounding the measure within `tolerance` of `target`.

        Args:
            target: Nominal center value for the measured quantity.
            tolerance: Half-width allowed on either side of `target`.

        Returns:
            Two-sided requirement spanning `target - tolerance` to `target + tolerance`.

        Raises:
            DimensionError: When `target` and `tolerance` have incompatible dimensions.
        """
        center, spread = _quantity(target), _quantity(tolerance)
        return Requirement(
            self, center - spread, center + spread, source_location=caller()
        )

    def cotangent(
        self, result: results.AnalysisResult
    ) -> tuple[Reading, MeasureCotangent]:
        """Evaluates this measure on `result` alongside its exact sparse cotangent.

        Args:
            result: Solved analysis result whose arrays are differentiated.

        Returns:
            Pair of measured `Reading` and sparse `MeasureCotangent`.

        Raises:
            MeasureError: When the measure is a custom Python measure, incompatible
                with `result`, or not differentiable at `result`.
        """
        if self.resolved is None:
            message = (
                f"{self} has no engine cotangent because it is a custom Python measure"
            )
            raise MeasureError(message)
        return self.resolved(result).cotangent(result)

    @override
    def __str__(self) -> str:
        """Formats the measure description together with its analysis name."""
        return f"{self.description} in {self.analysis}"


@pydantic_dataclass(frozen=True, slots=True, config=_REQUIREMENT_CONFIG)
class Requirement:
    """Measure paired with lower, upper, and optional margin bounds.

    Slack is the signed distance inside the bounds (negative when out of bounds). A
    requirement passes when slack is at least `margin`, is marginal when slack is
    non-negative but below `margin`, and fails when slack is negative.
    """

    measure: Measure
    minimum: Quantity | None = None
    maximum: Quantity | None = None
    margin: Quantity | None = None
    source_location: ir.SourceLocation | None = field(default=None, compare=False)
    """The script line that stated the requirement."""

    @field_validator("measure", mode="before")
    @classmethod
    def _validate_measure(cls, value: object) -> Measure:
        if not isinstance(value, Measure):
            fail("measure_type", f"measure is a Measure, not {value!r}")
        return value

    @field_validator("minimum", mode="before")
    @classmethod
    def _validate_minimum(cls, value: object) -> Quantity | None:
        return _check_bound_quantity(value, "minimum")

    @field_validator("maximum", mode="before")
    @classmethod
    def _validate_maximum(cls, value: object, info: ValidationInfo) -> Quantity | None:
        checked = _check_bound_quantity(value, "maximum")
        measure_raw = info.data.get("measure")
        measure = measure_raw if isinstance(measure_raw, Measure) else None
        subject = "the measure" if measure is None else str(measure)
        if "minimum" in info.data:
            minimum = info.data["minimum"]
            if minimum is None and checked is None:
                fail("bound_missing", f"a requirement on {subject} states a bound")
            if (
                isinstance(minimum, Quantity)
                and checked is not None
                and minimum.dimension == checked.dimension
                and minimum.value > checked.value
            ):
                message = (
                    f"the minimum of {subject}, {_format_bound(measure, minimum)}, "
                    f"exceeds its maximum, {_format_bound(measure, checked)}"
                )
                fail("bound_order", message)
        return checked

    @field_validator("margin", mode="before")
    @classmethod
    def _validate_margin(cls, value: object, info: ValidationInfo) -> Quantity | None:
        checked = _check_bound_quantity(value, "margin")
        if checked is not None and checked.value < 0:
            measure_raw = info.data.get("measure")
            subject = (
                str(measure_raw) if isinstance(measure_raw, Measure) else "the measure"
            )
            fail("non_negative", f"the margin of {subject} is negative: {checked}")
        return checked

    @field_validator("source_location", mode="before")
    @classmethod
    def _validate_source_location(cls, value: object) -> ir.SourceLocation | None:
        if value is None:
            return None
        if not isinstance(value, ir.SourceLocation):
            fail(
                "source_location_type",
                f"source_location is a SourceLocation, not {value!r}",
            )
        return value

    def with_margin(self, margin: Bound) -> Requirement:
        """Returns a copy of this requirement flagged marginal when slack < `margin`.

        Args:
            margin: Non-negative slack threshold below which passing runs are marginal.

        Returns:
            Updated requirement carrying `margin`.
        """
        return replace(self, margin=_quantity(margin))

    def _judge_with_sign(self, reading: Reading) -> tuple[Status, float, float]:
        """Evaluates `reading` and returns `(status, slack, sign)`.

        `sign` is `+1.0` when the lower bound governs slack and `-1.0` when the upper
        bound governs slack (breaking exact midpoint ties toward the lower bound).
        """
        slack = math.inf
        sign = 1.0
        if self.minimum is not None:
            slack = reading.value - _against(self.minimum, reading, "the minimum")
        if self.maximum is not None:
            upper_slack = _against(self.maximum, reading, "the maximum") - reading.value
            if upper_slack < slack:
                slack = upper_slack
                sign = -1.0
        margin = (
            0.0 if self.margin is None else _against(self.margin, reading, "the margin")
        )
        if slack < 0:
            return "failed", slack, sign
        return ("marginal" if slack < margin else "passed"), slack, sign

    def judge(self, reading: Reading) -> tuple[Status, float]:
        """Evaluates `reading` against the bounds and returns `(status, slack)`.

        Args:
            reading: Measured scalar value and unit to compare against bounds.

        Returns:
            Pair of requirement status and signed slack in SI base units.

        Raises:
            DimensionError: When a bound or margin does not match the reading unit.
        """
        status, slack, _ = self._judge_with_sign(reading)
        return status, slack

    def slack_cotangent(
        self, result: results.AnalysisResult
    ) -> tuple[Status, float, MeasureCotangent]:
        """Evaluates `result` and returns `(status, slack, cotangent)` for slack.

        The returned cotangent is the sparse reverse-mode derivative of signed slack
        with respect to `result`: `+1` times the measure cotangent when the lower
        bound is active and `-1` times it when the upper bound is active (breaking
        exact midpoint ties on two-sided requirements toward the lower bound).

        Args:
            result: Solved analysis result whose arrays are differentiated.

        Returns:
            Triple of requirement `Status`, signed slack in SI base units, and sparse
            `MeasureCotangent` for signed slack.

        Raises:
            DimensionError: When a bound or margin does not match the reading unit.
            MeasureError: When the measure cannot be evaluated or differentiated at
                `result`.
        """
        reading, measure_cotangent = self.measure.cotangent(result)
        status, slack, sign = self._judge_with_sign(reading)
        cotangent = measure_cotangent if sign > 0 else measure_cotangent.scale(sign)
        return status, slack, cotangent

    def _text(self, bound: Quantity) -> str:
        return _format_bound(self.measure, bound)

    @override
    def __str__(self) -> str:
        """Formats the requirement, such as ``gain of out at 1 kHz in bode ≥ -3 dB``."""
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


@pydantic_dataclass(frozen=True, slots=True, config=_CONFIG)
class Specs:
    """The requirements a circuit must meet, in the order a report lists them.

    The `yield_target` is the fraction of a Monte Carlo analysis's samples that must
    meet every requirement, such as 0.99.
    """

    requirements: Sequence[Requirement]
    yield_target: float | None = None

    @field_validator("requirements", mode="before")
    @classmethod
    def _validate_requirements(cls, value: object) -> tuple[Requirement, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            fail(
                "sequence_type",
                f"specs is a sequence of requirements, not {value!r}",
            )
        listed = tuple(value)
        if not listed:
            fail("empty", "specs state at least one requirement")
        errors: list[InitErrorDetails] = []
        checked: list[Requirement] = []
        for index, requirement in enumerate(listed):
            if not isinstance(requirement, Requirement):
                errors.append(
                    detail(
                        "requirement_type",
                        f"specs hold requirements, not {requirement!r}",
                        requirement,
                        (index,),
                    )
                )
            else:
                checked.append(requirement)
        raise_if_errors("requirements", errors)
        return tuple(checked)

    @field_validator("yield_target", mode="before")
    @classmethod
    def _validate_yield_target(cls, value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            fail("float_type", f"yield_target is a real number, not {value!r}")
        converted = float(value)
        if not math.isfinite(converted):
            fail("finite", f"yield_target is finite, not {converted!r}")
        if not 0 < converted <= 1:
            message = f"a yield target lies above 0 and at most 1, not {converted}"
            fail("yield_range", message)
        return converted


class Verdict(ir.FrozenModel):
    """Evaluation outcome for a single requirement across nominal and corner runs."""

    requirement: str = Field(description="Human-readable requirement specification.")
    status: Status
    value: ExtendedReal | None = Field(
        default=None, description="Measured value in SI base units, or null if unread."
    )
    unit: str | None = Field(
        default=None, description="Unit string for the value, such as 'V' or 'dB'."
    )
    slack: ExtendedReal | None = Field(
        default=None,
        description="Signed margin inside the bounds (negative when out of bounds).",
    )
    corner: str | None = Field(
        default=None,
        description="Plan corner producing the worst verdict, or null for nominal "
        "conditions.",
    )
    reason: str | None = Field(
        default=None, description="Explanation when the requirement is unmeasured."
    )
    source_location: ir.SourceLocation | None = Field(
        default=None, description="Script line that declared the requirement."
    )

    def describe(self) -> str:
        """Formats the verdict line followed by its source location when known.

        Returns:
            Human-readable verdict summary with optional indented source line.
        """
        at = "" if self.corner is None else f" at the corner {self.corner}"
        if self.value is None or self.unit is None or self.slack is None:
            head = f"unmeasured: {self.requirement}: {self.reason}{at}"
        else:
            read = format_reading(self.value, self.unit)
            slack = format_reading(self.slack, self.unit)
            head = f"{self.status}: {self.requirement}: read {read}{at}, slack {slack}"
        return (
            head
            if self.source_location is None
            else f"{head}\n  = {self.source_location}"
        )


class SummaryStatistics(ir.FrozenModel):
    """Summary statistics of measured values across samples, in SI base units."""

    unit: str = Field(
        description="Unit string for the statistics, such as 'V' or 'dB'."
    )
    minimum: ExtendedReal
    maximum: ExtendedReal
    mean: ExtendedReal | None = Field(
        description="Arithmetic mean, or null when opposite-sign infinities occur."
    )
    standard_deviation: float | None = Field(
        description="Population standard deviation, or null when any value is infinite."
    )

    @classmethod
    def from_numbers(
        cls, numbers: npt.NDArray[np.float64], unit: str
    ) -> SummaryStatistics:
        """Computes min, max, mean, and standard deviation of `numbers`.

        A single-sign infinity sets `mean` to that infinity and leaves
        `standard_deviation` undefined; opposite-sign infinities leave both undefined.

        Args:
            numbers: Non-empty array of measured floating-point values.
            unit: Display unit string associated with `numbers`.

        Returns:
            Summary statistics over `numbers` tagged with `unit`.
        """
        finite = np.isfinite(numbers)
        infinities = {float(number) for number in numbers[~finite]}
        if finite.all():
            mean, standard_deviation = float(numbers.mean()), float(numbers.std())
        else:
            mean = infinities.pop() if len(infinities) == 1 else None
            standard_deviation = None
        return cls(
            unit=unit,
            minimum=float(numbers.min()),
            maximum=float(numbers.max()),
            mean=mean,
            standard_deviation=standard_deviation,
        )


class Spread(ir.FrozenModel):
    """Distribution summary for one requirement across Monte Carlo samples."""

    requirement: str = Field(description="Human-readable requirement specification.")
    measured: int = Field(description="Number of samples with a measured value.")
    passing: int = Field(description="Number of samples passing or marginally passing.")
    values: SummaryStatistics | None = Field(
        default=None, description="Measured summary statistics, or null if none."
    )

    def describe(self, samples: int) -> str:
        """Formats the requirement spread as one line across `samples` total runs.

        Args:
            samples: Total number of Monte Carlo or variant runs evaluated.

        Returns:
            Single-line human-readable summary of passing count and value statistics.
        """
        head = f"{self.requirement}: {self.passing} of {samples} pass"
        if self.values is None:
            return f"{head}, none measured"
        unmeasured = samples - self.measured
        missing = f", {unmeasured} unmeasured" if unmeasured else ""
        unit = self.values.unit
        low, high = (
            format_reading(self.values.minimum, unit),
            format_reading(self.values.maximum, unit),
        )
        mean, standard_deviation = (
            "undefined" if value is None else format_reading(value, unit)
            for value in (self.values.mean, self.values.standard_deviation)
        )
        return (
            f"{head}{missing}; read {low} to {high}, mean {mean}, "
            f"standard deviation {standard_deviation}"
        )


class Yield(ir.FrozenModel):
    """Monte Carlo passing count, yield estimate, and 95 % Wilson interval."""

    samples: int
    passing: int = Field(description="Samples satisfying every requirement.")
    estimate: float = Field(description="Observed fraction of passing samples.")
    lower: float = Field(description="Lower bound of the 95 % Wilson interval.")
    upper: float = Field(description="Upper bound of the 95 % Wilson interval.")
    target: float | None = Field(
        default=None, description="Target yield fraction from specs, or null if unset."
    )

    @property
    def met(self) -> bool:
        """True when the yield estimate meets `target`, or when `target` is unset."""
        return self.target is None or self.estimate >= self.target

    def describe(self) -> str:
        """Formats the yield summary and Wilson interval as one line.

        Returns:
            Single-line human-readable yield estimate, confidence interval, and target
            comparison.
        """
        head = (
            f"yield: {self.passing} of {self.samples}, {100 * self.estimate:.4g} % "
            f"(95 % interval {100 * self.lower:.4g} to {100 * self.upper:.4g} %)"
        )
        if self.target is None:
            return head
        reached = "reaches" if self.met else "is below"
        return f"{head}, which {reached} the target {100 * self.target:.4g} %"


def _yield_of(passing: int, samples: int, target: float | None) -> Yield:
    """Computes `Yield` with its 95 % Wilson interval for `passing` of `samples`."""
    low, high = wilson(passing, samples)
    return Yield(
        samples=samples,
        passing=passing,
        estimate=passing / samples,
        lower=low,
        upper=high,
        target=target,
    )


def spread(
    specs: Specs, sampled: Sequence[Sequence[Verdict]]
) -> tuple[tuple[Spread, ...], Yield]:
    """Computes per-requirement spreads and overall yield from per-sample verdicts.

    Each entry in `sampled` contains verdicts for `specs.requirements` in order. A
    sample passes when every requirement passes or is marginal.

    Args:
        specs: Circuit specifications defining ordered requirements and optional yield
            target.
        sampled: Per-sample sequences of requirement verdicts aligned with `specs`.

    Returns:
        Pair of per-requirement `Spread` summaries and aggregate `Yield`.
    """
    spreads = tuple(
        _spread(requirement, [verdicts[index] for verdicts in sampled])
        for index, requirement in enumerate(specs.requirements)
    )
    passing = sum(all(_passes(verdict) for verdict in verdicts) for verdicts in sampled)
    return spreads, _yield_of(passing, len(sampled), specs.yield_target)


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
    values = SummaryStatistics.from_numbers(
        np.array([value for value, _ in read]), read[0][1]
    )
    return stated.model_copy(update={"values": values})


def _passes(verdict: Verdict) -> bool:
    return verdict.status in {"passed", "marginal"}


@dataclass(frozen=True, slots=True)
class Judged:
    """A requirement's verdicts on many runs, as columns, one entry per run.

    Each run's `status` is its status, 0 failed, 1 unmeasured, 2 marginal and 3
    passed, with its `slack`, infinite where unmeasured, its `value`, not a number
    where unmeasured, and, among `units`, the unit of the values of the conditions
    each came from, by `origin`.
    """

    status: _StatusArray
    slack: _RealArray
    value: _RealArray
    origin: _OriginArray
    units: tuple[str, ...]


def judged(
    requirement: Requirement,
    planned: Collection[str],
    batch: results.BatchResult | None,
    place: int,
    runs: int,
) -> Judged:
    """Evaluates `requirement` across `runs` batch entries at measure slot `place`.

    Matches single-run `evaluate` semantics using `planned` analysis names. When
    `batch` is None, all runs are marked unmeasured.

    Args:
        requirement: Requirement to judge across every batch run.
        planned: Planned analysis names defined by the simulation plan.
        batch: Completed batch result holding measured columns, or ``None`` when
            elaboration failed.
        place: Column index of `requirement` inside `batch.measured`.
        runs: Number of runs represented by the returned columnar arrays.

    Returns:
        Columnar `Judged` verdicts covering all `runs` entries.
    """
    status = np.full(runs, _SEVERITY["unmeasured"], dtype=np.int8)
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
            status[measured] = np.where(
                room < 0,
                _SEVERITY["failed"],
                np.where(room < margin, _SEVERITY["marginal"], _SEVERITY["passed"]),
            )
            slack[measured] = room
            value[measured] = values
    return Judged(status, slack, value, np.zeros(runs, dtype=np.intp), (unit,))


def worst(conditions: Sequence[Judged]) -> Judged:
    """Selects the worst per-run verdict across `conditions`, matching `judge`.

    Orders severity as failed, unmeasured, marginal, then passed; breaks ties by least
    slack, then earliest condition.

    Args:
        conditions: Non-empty sequence of per-condition `Judged` columns.

    Returns:
        Combined `Judged` columns recording the worst condition for each run.

    Raises:
        ValueError: When `conditions` is empty.
    """
    if not conditions:
        message = "the worst verdict is of a condition or more"
        raise ValueError(message)
    first, *rest = conditions
    status, slack, value = (
        first.status.copy(),
        first.slack.copy(),
        first.value.copy(),
    )
    origin = np.zeros(status.shape, dtype=np.intp)
    for place, each in enumerate(rest, start=1):
        worse = (each.status < status) | (
            (each.status == status) & (each.slack < slack)
        )
        status[worse] = each.status[worse]
        slack[worse] = each.slack[worse]
        value[worse] = each.value[worse]
        origin[worse] = place
    units = tuple(
        each.units[int(each.origin[0])] if each.units else "" for each in conditions
    )
    return Judged(status, slack, value, origin, units)


def spread_of(
    specs: Specs, judged: Sequence[Judged]
) -> tuple[tuple[Spread, ...], Yield]:
    """Computes per-requirement spreads and overall yield from columnar `judged` arrays.

    Entries in `judged` correspond to `specs.requirements` in order across all samples.

    Args:
        specs: Circuit specifications defining ordered requirements and optional yield
            target.
        judged: Columnar per-requirement verdicts aligned with `specs.requirements`.

    Returns:
        Pair of per-requirement `Spread` summaries and aggregate `Yield`.
    """
    spreads = tuple(
        _spread_of(requirement, each)
        for requirement, each in zip(specs.requirements, judged, strict=True)
    )
    samples = int(judged[0].status.shape[0]) if judged else 0
    passing = (
        int(
            np.all(
                [each.status >= _SEVERITY["marginal"] for each in judged], axis=0
            ).sum()
        )
        if judged
        else samples
    )
    return spreads, _yield_of(passing, samples, specs.yield_target)


def _spread_of(requirement: Requirement, judged: Judged) -> Spread:
    measured = ~np.isnan(judged.value)
    passing = int((judged.status >= _SEVERITY["marginal"]).sum())
    stated = Spread(
        requirement=str(requirement), measured=int(measured.sum()), passing=passing
    )
    if not measured.any():
        return stated
    first = int(np.argmax(measured))
    unit = judged.units[int(judged.origin[first])]
    values = SummaryStatistics.from_numbers(judged.value[measured], unit)
    return stated.model_copy(update={"values": values})


_Z: Final = 1.959963984540054
"""The normal distribution's 97.5th percentile: a 95 % interval spans ±_Z deviations."""


def wilson(passing: int, samples: int) -> tuple[float, float]:
    """Computes the 95 % Wilson score interval for `passing` out of `samples`.

    Produces the confidence interval reported by [`Yield`][jolty.specs.Yield] (E. B.
    Wilson, "Probable inference, the law of succession, and statistical inference",
    JASA 22, 1927). Coverage stays near 95 % even for small sample counts and
    proportions near 0 or 1, where normal approximations under-cover (Brown, Cai, and
    DasGupta, Statistical Science 16, 2001). The lower bound
    `(p + w/2 - z·s) / (1 + w)` with `w = z²/n` is evaluated in the subtraction-free
    conjugate form `p² / (p + w/2 + z·s)`, returning exact 0.0 when `passing == 0`.
    By symmetry, the upper bound equals one minus the lower bound for the failure
    proportion, returning exact 1.0 when every sample passes.

    Args:
        passing: Number of passing samples, between 0 and `samples` inclusive.
        samples: Positive total sample count.

    Returns:
        Lower and upper bounds of the 95 % Wilson confidence interval in `[0.0, 1.0]`.

    Raises:
        ValueError: When `samples < 1` or `passing` lies outside `0..samples`.

    Examples:
        >>> low, high = wilson(37, 200)
        >>> f"{low:.4f} to {high:.4f}"
        '0.1373 to 0.2446'
        >>> wilson(0, 10)[0], wilson(10, 10)[1]
        (0.0, 1.0)
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
    """Formats `verdicts` as text blocks followed by a status tally line.

    Args:
        verdicts: Requirement verdicts to render in specification order.

    Returns:
        Multi-paragraph summary of every verdict and total status counts.
    """
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
    """Evaluates each requirement across nominal results and optional corners.

    `planned` lists all planned analysis names (omissions in `solved` indicate failed
    analyses). `solved` maps analysis names to nominal results, and `corners` maps
    corner names to per-analysis results.

    Args:
        specs: Circuit specifications whose requirements are evaluated in order.
        planned: Planned analysis names defined by the simulation plan.
        solved: Mapping from analysis names to completed nominal results.
        corners: Optional mapping from corner names to per-analysis results.

    Returns:
        Worst-corner `Verdict` for each requirement in `specs`.
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
    """Returns the worst verdict for `requirement` across `runs`.

    Each entry in `runs` pairs a corner name (or None for nominal conditions) with its
    completed analysis results; `planned` lists all planned analysis names.

    Args:
        requirement: Single specification requirement to evaluate.
        planned: Planned analysis names defined by the simulation plan.
        runs: Non-empty iterable of `(corner, results)` pairs to evaluate.

    Returns:
        Worst `Verdict` observed across `runs`.

    Raises:
        ValueError: When `runs` is empty.
    """
    verdicts = [_verdict(requirement, planned, found, corner) for corner, found in runs]
    if not verdicts:
        message = f"{requirement} is judged on a run or more"
        raise ValueError(message)
    return min(verdicts, key=_badness)


def reading(
    requirement: Requirement, found: Mapping[str, results.AnalysisResult]
) -> float | None:
    """Evaluates `requirement` on `found` and returns its numeric value, or None.

    Args:
        requirement: Specification requirement whose measure is evaluated.
        found: Mapping from analysis names to solved analysis results.

    Returns:
        Measured value in SI base units or plain units, or ``None`` when unmeasured.
    """
    verdict = _verdict(requirement, found.keys(), found, None)
    return verdict.value


def _badness(verdict: Verdict) -> tuple[int, float]:
    """Returns the `(severity, slack)` sort key placing worst verdicts first."""
    slack = math.inf if verdict.slack is None else verdict.slack
    return _SEVERITY[verdict.status], slack


def _tag(requirement: Requirement, message: str) -> str:
    """Formats `message` for `requirement`, appending its source line when known."""
    head = f"{requirement}: {message}"
    if requirement.source_location is None:
        return head
    return f"{head}\n  = {requirement.source_location}"


def _bound_problems(
    requirement: Requirement, unit: Dimension | results.Plain
) -> list[str]:
    """Collects dimension mismatches between `requirement` bounds and `unit`."""
    expected = Reading(0.0, unit)
    problems: list[str] = []
    for bound, label in (
        (requirement.minimum, "the minimum"),
        (requirement.maximum, "the maximum"),
        (requirement.margin, "the margin"),
    ):
        if bound is None:
            continue
        try:
            _against(bound, expected, label)
        except DimensionError as error:
            problems.append(_tag(requirement, str(error.args[0])))
    return problems


def _validate_against_shape(
    requirement: Requirement,
    shape: results.AnalysisResult,
    plain_problems: list[str],
    *,
    skip_unit: bool,
) -> list[str]:
    """Checks `requirement` against a resolved analysis `shape`."""
    measure = requirement.measure
    if measure.diagnose is not None:
        measure_problems = list(measure.diagnose(shape))
        if measure_problems:
            return [
                *(_tag(requirement, issue) for issue in measure_problems),
                *plain_problems,
            ]
    if measure.resolved is None:
        return plain_problems
    try:
        resolved = measure.resolved(shape)
    except (JoltyError, KeyError) as error:
        return [_tag(requirement, str(error.args[0])), *plain_problems]
    return [] if skip_unit else _bound_problems(requirement, resolved.unit)


def _validate_requirement(
    requirement: Requirement,
    planned: Sequence[str],
    shapes: Mapping[str, results.AnalysisResult],
    unknown_units: Collection[str],
    defined: str,
) -> list[str]:
    """Checks one `requirement` against `planned` analyses and `shapes`."""
    measure = requirement.measure
    analysis = measure.analysis
    if analysis not in planned:
        close = close_names(planned, analysis)
        listed = f"; defined analyses: {defined}" if defined else ""
        return [
            _tag(
                requirement,
                f"the plan runs no analysis named {analysis!r}{close}{listed}",
            )
        ]
    plain_problems = (
        [] if measure.plain is None else _bound_problems(requirement, measure.plain)
    )
    if analysis not in shapes:
        return plain_problems
    return _validate_against_shape(
        requirement,
        shapes[analysis],
        plain_problems,
        skip_unit=analysis in unknown_units,
    )


def validate_specs(
    specs: Specs,
    planned: Sequence[str],
    shapes: Mapping[str, results.AnalysisResult],
    *,
    unknown_units: Collection[str] = (),
) -> list[str]:
    """Checks every requirement of `specs` against `planned` analyses and `shapes`.

    Reports every unresolved analysis name (with close names), every unresolved
    signal, reference, or divisor name (with close names), incompatible analysis
    kinds, missing or wrongly dimensioned measure parameters, unreferred noise
    measures, and mismatched bound or margin dimensions.

    Analyses named in `unknown_units` still check kind, referral, and parameter
    dimensions, skipping only the bound and margin dimension comparison because the
    analysis output unit could not be determined.

    Args:
        specs: Circuit specifications to validate statically before simulation.
        planned: Ordered analysis names defined by the simulation plan.
        shapes: Mapping from valid analysis names to empty result prototypes.
        unknown_units: Analysis names whose output dimension could not be resolved
            statically.

    Returns:
        Diagnostic messages describing every invalid requirement in `specs`.
    """
    defined = ", ".join(planned)
    problems: list[str] = []
    for requirement in specs.requirements:
        problems.extend(
            _validate_requirement(requirement, planned, shapes, unknown_units, defined)
        )
    return problems


def _verdict(
    requirement: Requirement,
    planned: Collection[str],
    solved: Mapping[str, results.AnalysisResult],
    corner: str | None,
) -> Verdict:
    stated = Verdict(
        requirement=str(requirement),
        status="unmeasured",
        source_location=requirement.source_location,
        corner=corner,
    )
    analysis = requirement.measure.analysis
    if analysis not in planned:
        close = close_names(tuple(planned), analysis)
        reason = f"the plan runs no analysis named {analysis}{close}"
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
    """Converts `bound` into a finite `Quantity`, treating floats as dimensionless.

    Raises:
        ValueError: When `bound` is infinite or NaN.
    """
    quantity = bound if isinstance(bound, Quantity) else Quantity(bound, DIMENSIONLESS)
    finite(quantity.value, "a bound")
    return quantity


def _against(bound: Quantity, reading: Reading, name: str) -> float:
    """Converts `bound` to SI base units after verifying compatibility with `reading`.

    Raises:
        DimensionError: When `bound` does not match the dimension of `reading`.
    """
    if bound.dimension != reading.dimension:
        message = (
            f"{name}, {bound}, is not in {reading.unit}, the unit of the value read"
        )
        raise DimensionError(message)
    return bound.value


def format_reading(value: float, unit: str) -> str:
    """Formats `value` in `unit` to four significant digits with an SI prefix if any.

    Args:
        value: Measured scalar value in SI base units or plain units.
        unit: Unit symbol such as ``"V"``, ``"dB"``, ``"°"``, ``"%"``, or empty string.

    Returns:
        Formatted value string with SI prefix or plain unit suffix.
    """
    if unit in {"dB", "°", "%"}:
        return f"{value:.4g} {unit}"
    if not unit:
        return f"{value:.4g}"
    return format_si(value, unit)
