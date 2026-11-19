# Copyright 2026 Maxence Faldor
"""Measure queries on saved simulation results, driven by `jolty.specs.MEASURES`.

Validates request options against each measure's declarative specification, maps
scalar SI values into dimensioned quantities for the target analysis axis or signal,
and evaluates the resulting measure without running a new simulation.
"""

__lazy_modules__ = ["jolty"]

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal

from pydantic import Field
from pydantic_settings import CliPositionalArg

from jolty import results, specs, unknowns
from jolty.errors import JoltyError
from jolty.ir import FrozenModel
from jolty.specs import Edge
from jolty.units import DIMENSIONLESS, FREQUENCY, TIME, VOLTAGE, Dimension, Quantity
from jolty.unknowns import Unknown

type MeasureKind = Literal[
    "value",
    "gain",
    "phase",
    "bandwidth",
    "crossover",
    "phase_margin",
    "gain_margin",
    "maximum",
    "minimum",
    "peak_to_peak",
    "mean",
    "rms",
    "rise_time",
    "overshoot",
    "settling_time",
    "crossing",
    "frequency",
    "distortion",
    "period",
    "noise_density",
    "integrated_noise",
    "solutions",
    "damping",
    "dominant_pole",
]
"""Every measure that ``jolty measure`` reads from a saved results file."""

_MEASURE_OPTION_FIELDS: Final[tuple[str, ...]] = (
    "signal",
    "reference",
    "over",
    "at",
    "start",
    "stop",
    "drop",
    "lower",
    "upper",
    "band",
    "threshold",
    "edge",
    "fundamental",
    "harmonics",
    "referred",
)


class MeasureRequest(FrozenModel):
    """Read a measure from a saved results file without running a simulation.

    Evaluates any measure of ``jolty.specs`` (``value``, ``gain``, ``phase``,
    ``bandwidth``, ``crossover``, ``phase_margin``, ``gain_margin``, ``maximum``,
    ``minimum``, ``peak_to_peak``, ``mean``, ``rms``, ``rise_time``, ``overshoot``,
    ``settling_time``, ``crossing``, ``frequency``, ``distortion``, ``noise_density``,
    ``integrated_noise``, ``solutions``, ``damping``, ``dominant_pole``), plus
    ``period`` on a periodic steady state, directly from a saved ``.results.npz``
    file. Numeric options (``at``, ``start``, ``stop``, ``threshold``, ``fundamental``)
    are in SI base units: seconds for transients and periodic steady states, hertz
    for small-signal and noise analyses, and volts, amperes, kelvins or plain
    numbers for signal values and DC sweeps.
    """

    results: CliPositionalArg[Path] = Field(description="A .results.npz file.")
    measure: MeasureKind = Field(
        description="Measure to read from the analysis result."
    )
    analysis: str | None = Field(
        default=None,
        description="Name of the analysis in the results file; optional when the file "
        "contains a single analysis.",
    )
    signal: str | None = Field(
        default=None,
        description="Unknown to read, such as 'out', 'X1/mid' or 'V1:current'; omitted "
        "for a stability loop gain, period, noise, solutions or pole-zero measure.",
    )
    reference: str | None = Field(
        default=None,
        description="Reference net for a differential voltage measure.",
    )
    over: str | None = Field(
        default=None,
        description="Divisor unknown for a small-signal transfer measure (gain, "
        "phase, bandwidth, crossover, phase_margin, gain_margin).",
    )
    at: float | None = Field(
        default=None,
        description="Point along the analysis axis in SI base units: time in seconds "
        "for a transient or periodic steady state, swept value for a DC sweep, or "
        "frequency in hertz for gain, phase and noise_density.",
    )
    start: float | None = Field(
        default=None,
        description="Window start in SI base units: time in seconds, swept value, or "
        "frequency in hertz for integrated_noise; defaults to the start of the axis.",
    )
    stop: float | None = Field(
        default=None,
        description="Window stop in SI base units: time in seconds, swept value, or "
        "frequency in hertz for integrated_noise; defaults to the end of the axis.",
    )
    drop: float | None = Field(
        default=None,
        description="Gain drop in decibels for bandwidth; defaults to 3.0.",
    )
    lower: float | None = Field(
        default=None,
        description="Lower step fraction (between 0 and 1) for rise_time; defaults "
        "to 0.1.",
    )
    upper: float | None = Field(
        default=None,
        description="Upper step fraction (between 0 and 1) for rise_time; defaults "
        "to 0.9.",
    )
    band: float | None = Field(
        default=None,
        description="Settling band fraction (positive) for settling_time; defaults "
        "to 0.02.",
    )
    threshold: float | None = Field(
        default=None,
        description="Threshold in SI base units of the signal (volts, amperes, "
        "kelvins or plain number) for crossing (required) or frequency (optional "
        "midpoint default).",
    )
    edge: Edge | None = Field(
        default=None,
        description="Crossing direction for crossing: 'rising', 'falling' or 'either'; "
        "defaults to 'either'.",
    )
    fundamental: float | None = Field(
        default=None,
        description="Fundamental frequency in hertz for distortion (required).",
    )
    harmonics: int | None = Field(
        default=None,
        ge=2,
        description="Highest harmonic counted for distortion; defaults to 9.",
    )
    referred: bool = Field(
        default=False,
        description="Input-referred flag for noise_density or integrated_noise when "
        "true.",
    )


def check_options(request: MeasureRequest) -> None:
    """Rejects options that do not apply to `request.measure`.

    Raises:
        JoltyError: When an option inapplicable to the chosen measure is supplied.
    """
    allowed = (
        frozenset()
        if request.measure == "period"
        else specs.MEASURES[request.measure].options
    )
    extra = [
        name
        for name in _MEASURE_OPTION_FIELDS
        if name not in allowed
        and (
            bool(getattr(request, name))
            if name == "referred"
            else getattr(request, name) is not None
        )
    ]
    if extra:
        listed = ", ".join(extra)
        message = f"measure {request.measure} takes no {listed}"
        raise JoltyError(message)


def select_analysis(
    path: Path,
    loaded: Mapping[str, results.AnalysisResult],
    analysis: str | None,
) -> tuple[str, results.AnalysisResult]:
    """Returns the named analysis from `loaded`, or the sole entry when omitted.

    Raises:
        JoltyError: Naming available analyses when selection is missing or ambiguous.
    """
    if analysis is None:
        if not loaded:
            message = f"{path} contains no analyses"
            raise JoltyError(message)
        if len(loaded) > 1:
            available = ", ".join(loaded)
            message = (
                f"{path} contains {len(loaded)} analyses ({available}); "
                "analysis is unset"
            )
            raise JoltyError(message)
        sole = next(iter(loaded))
        return sole, loaded[sole]
    if analysis not in loaded:
        if loaded:
            available = ", ".join(loaded)
            message = (
                f"{path} contains no analysis named {analysis!r}; "
                f"the analyses are {available}"
            )
        else:
            message = (
                f"{path} contains no analysis named {analysis!r}; "
                "the file contains no analyses"
            )
        raise JoltyError(message)
    return analysis, loaded[analysis]


def evaluate_measure(
    request: MeasureRequest,
    analysis_name: str,
    result: results.AnalysisResult,
    max_unknowns: int,
) -> tuple[str, float, str, str]:
    """Evaluates `request` on `result`, returning description, value, unit, formatted.

    Raises:
        JoltyError: When required inputs are missing, unknown names fail to resolve, or
            the measure cannot be read from `result`.
    """
    description, reading = _read_measure(request, analysis_name, result, max_unknowns)
    if math.isnan(reading.value):
        message = f"{description} read not a number"
        raise JoltyError(message)
    unit = "" if reading.unit == DIMENSIONLESS else str(reading.unit)
    return description, reading.value, unit, specs.format_reading(reading.value, unit)


def _read_measure(
    request: MeasureRequest,
    analysis_name: str,
    result: results.AnalysisResult,
    max_unknowns: int,
) -> tuple[str, specs.Reading]:
    """Constructs and reads the requested measure against `result`."""
    if request.measure == "period":
        if not isinstance(result, results.PeriodicSteadyStateResult):
            kind = results.result_kind(result)
            message = f"period reads a periodic steady state, not {kind}"
            raise JoltyError(message)
        return f"period in {analysis_name}", specs.Reading(result.period, TIME)
    try:
        built = _build_measure(request, analysis_name, result, max_unknowns)
        reading = built.read(result)
    except KeyError as error:
        available = _available_unknowns(result.unknowns, max_unknowns)
        message = f"{error.args[0]}; unknowns in {analysis_name}: {available}"
        raise JoltyError(message) from None
    except ValueError as error:
        raise JoltyError(str(error)) from None
    return str(built), reading


def _build_measure(
    request: MeasureRequest,
    analysis_name: str,
    result: results.AnalysisResult,
    max_unknowns: int,
) -> specs.Measure:
    """Builds the `jolty.specs.Measure` described by `request` via `specs.MEASURES`."""
    spec = specs.MEASURES[request.measure]
    if spec.category == "transfer":
        if isinstance(result, results.AcResult) and request.signal is None:
            available = _available_unknowns(result.unknowns, max_unknowns)
            message = (
                f"measure {request.measure} on a small-signal response takes a signal; "
                f"unknowns in {analysis_name}: {available}"
            )
            raise JoltyError(message)
    elif "signal" in spec.positional and request.signal is None:
        available = _available_unknowns(result.unknowns, max_unknowns)
        message = (
            f"measure {request.measure} takes a signal; "
            f"unknowns in {analysis_name}: {available}"
        )
        raise JoltyError(message)
    for field_name, needed in spec.required.items():
        if getattr(request, field_name) is None:
            message = f"measure {request.measure} takes {needed}"
            raise JoltyError(message)
    values: dict[str, object] = {}
    for field_name in spec.options:
        raw = getattr(request, field_name)
        if raw is None:
            continue
        if (
            field_name in spec.quantities
            and not isinstance(raw, bool)
            and isinstance(raw, int | float)
        ):
            dimension = _dimension(
                spec.quantities[field_name], result, request.signal or ""
            )
            values[field_name] = Quantity(float(raw), dimension)
        else:
            values[field_name] = raw
    positional = [values.pop(field_name, None) for field_name in spec.positional]
    return spec.factory(analysis_name, *positional, **values)


def _dimension(
    kind: specs.MeasureDimension, result: results.AnalysisResult, signal: str
) -> Dimension:
    """Resolves a declarative dimension tag against `result` and `signal`."""
    match kind:
        case "frequency":
            return FREQUENCY
        case "time":
            return TIME
        case "axis":
            return _axis_dimension(result)
        case "signal":
            return _signal_dimension(result, signal)


def _axis_dimension(result: results.AnalysisResult) -> Dimension:
    """Returns the axis dimension of a transient or DC sweep, defaulting to time."""
    if isinstance(result, results.DcSweepResult):
        if result.nature is None:
            return DIMENSIONLESS
        return unknowns.DIMENSIONS[result.nature]
    return TIME


def _signal_dimension(result: results.AnalysisResult, signal: str) -> Dimension:
    """Returns the dimension of `signal` in `result.unknowns`, defaulting to voltage."""
    if not isinstance(result, results.TransientResult):
        return VOLTAGE
    index = results.column(result.unknowns, signal)
    return unknowns.DIMENSIONS[result.unknowns[index].nature]


def _available_unknowns(unknowns_list: tuple[Unknown, ...], limit: int) -> str:
    """Formats up to `limit` unknown names from `unknowns_list`."""
    if not unknowns_list:
        return "none"
    names = [unknown.name for unknown in unknowns_list]
    if len(names) <= limit:
        return ", ".join(names)
    shown = ", ".join(names[:limit])
    return f"{shown}, ... ({len(names) - limit} more)"
