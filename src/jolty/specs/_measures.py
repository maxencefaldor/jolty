# Copyright 2026 Maxence Faldor
"""Built-in circuit measures and loop-stability margin diagnostics.

Each measure constructor binds an analysis name and target signal as described in
[`jolty.specs`][jolty.specs]. Measures resolve signal indices, result kinds, and SI
units against a result so the Rust engine computes the value directly from result
arrays without Python overhead during batched runs.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal, override

from jolty import _core, checks, results, unknowns
from jolty.errors import DimensionError
from jolty.specs._requirements import (
    Edge,
    Measure,
    MeasureError,
    Reading,
    ResolvedMeasure,
)
from jolty.units import (
    FREQUENCY,
    TIME,
    Dimension,
    Quantity,
    finite,
    format_si,
    rtHz,
)
from jolty.unknowns import Unknown

type MeasureCategory = Literal[
    "plain", "noise", "transfer", "value", "statistic", "transient"
]
"""Structural family of a public measure, used by request adapters."""

type MeasureDimension = Literal["axis", "frequency", "time", "signal"]
"""Source domain determining the physical dimension of a scalar request option."""

type _Explainer = Mapping[str, Callable[[Sequence[float]], str]]
"""Mapping from engine failure tags to functions formatting numeric error details."""

type _Resolve = Callable[[results.AnalysisResult], ResolvedMeasure]
"""Resolver callback translating a measure into engine-ready indices and SI values."""

type _Diagnose = Callable[[results.AnalysisResult], Sequence[str]]
"""Validator callback collecting static resolution problems against a result shape."""


def diagnose(
    solved: Mapping[str, results.AnalysisResult],
) -> tuple[checks.Diagnostic, ...]:
    """Collects stability-margin diagnostics across all analyses in `solved`.

    Each stability analysis triggers ``low-phase-margin`` when phase margin at the
    first 0 dB gain crossover is below 45°, and ``low-gain-margin`` when gain margin
    at the first -180° phase crossover is below 6 dB, matching `phase_margin` and
    `gain_margin`.

    Args:
        solved: Mapping from analysis names to completed analysis results.

    Returns:
        Stability-margin diagnostics emitted across all stability analyses in `solved`.
    """
    found: list[checks.Diagnostic] = []
    for name, result in solved.items():
        if not isinstance(result, results.StabilityResult):
            continue
        loop = f"the loop broken at `{result.probe}`"
        try:
            margin = phase_margin(name).read(result).value
            unity = crossover(name).read(result).value
        except MeasureError:
            pass
        else:
            if margin < checks.MIN_PHASE_MARGIN:
                message = (
                    f"{loop} has a phase margin of {margin!r}° at its crossover, "
                    f"{unity!r} Hz"
                )
                found.append(_diagnostic(checks.LOW_PHASE_MARGIN, message))
        try:
            margin = gain_margin(name).read(result).value
        except MeasureError:
            continue
        if margin < checks.MIN_GAIN_MARGIN:
            message = f"{loop} has a gain margin of {margin!r} dB"
            found.append(_diagnostic(checks.LOW_GAIN_MARGIN, message))
    return tuple(found)


def _diagnostic(rule: checks.Rule, message: str) -> checks.Diagnostic:
    return checks.Diagnostic(
        rule=rule.name, level=rule.level, message=message, instances=(), nets=()
    )


def _check_value(
    traces: _Signal, at: Quantity | None, result: results.AnalysisResult
) -> list[str]:
    """Collects static problems resolving a `value` measure against `result`."""
    if isinstance(result, results.OperatingPointResult):
        problems = traces.problems(result.unknowns)
        if at is not None:
            problems.append(f"an operating point holds one value, not one at {at}")
        return problems
    if not isinstance(result, results.DcSweepResult | results.TransientResult):
        message = (
            "a value reads an operating point, a transient or a DC sweep, "
            f"not {results.result_kind(result)}"
        )
        return [message]
    axis = _axis(result, "a value")
    problems = traces.problems(result.unknowns)
    if at is None:
        problems.append(
            f"{results.result_kind(result)} holds a value per {axis.noun}, "
            "and at is unset"
        )
    elif (axis_error := axis.check(at, "at")) is not None:
        problems.append(axis_error)
    return problems


def value(
    analysis: str,
    signal: str,
    *,
    at: Quantity | None = None,
    reference: str | None = None,
) -> Measure:
    """Measures `signal` at an operating point, transient time, or DC sweep level.

    In a transient or DC sweep, the value is linearly interpolated `at` the requested
    time or swept level.

    Args:
        analysis: Plan key naming the target operating-point, transient, or DC sweep
            analysis.
        signal: Circuit unknown name to read.
        at: Evaluation time or swept level for transient and DC sweep analyses, or
            ``None`` at an operating point.
        reference: Optional negative reference net for a differential voltage reading.

    Returns:
        Measure reading `signal` from `analysis`.
    """
    traces = _Signal(signal, reference)
    _check(at, "at")

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        if isinstance(result, results.OperatingPointResult):
            if at is not None:
                message = f"an operating point holds one value, not one at {at}"
                raise MeasureError(message)
            resolved, dimension = traces.resolve(result.unknowns)
            return ResolvedMeasure(
                _core.Measure.value(resolved, None), dimension, _explain()
            )
        if not isinstance(result, results.DcSweepResult | results.TransientResult):
            message = (
                "a value reads an operating point, a transient or a DC sweep, "
                f"not {results.result_kind(result)}"
            )
            raise MeasureError(message)
        axis = _axis(result, "a value")
        resolved, dimension = traces.resolve(result.unknowns)
        if at is None:
            message = (
                f"{results.result_kind(result)} holds a value per {axis.noun}, "
                "and at is unset"
            )
            raise MeasureError(message)
        measure = _core.Measure.value(resolved, axis.convert(at, "at"))
        return ResolvedMeasure(
            measure, dimension, _explain(axis.outside({"at": ("at", at)}))
        )

    described = str(traces) if at is None else f"{traces} at {at}"
    return _native(
        analysis,
        described,
        None,
        resolve,
        lambda result: _check_value(traces, at, result),
    )


def gain(
    analysis: str,
    signal: str | None = None,
    *,
    at: Quantity,
    over: str | None = None,
    reference: str | None = None,
) -> Measure:
    """Measures small-signal gain of `signal` at frequency `at`, in decibels.

    Computes the transfer magnitude of `signal` relative to `over` (or unit
    excitation), interpolated over log frequency. Omitting `signal` measures the loop
    gain of a stability analysis.

    Args:
        analysis: Plan key naming the target AC or stability analysis.
        signal: Output unknown name for AC analyses, or ``None`` for stability loop
            gain.
        at: Frequency at which small-signal gain is evaluated.
        over: Optional denominator unknown name for a ratio transfer gain.
        reference: Optional negative reference net for a differential output voltage.

    Returns:
        Measure evaluating small-signal gain in decibels at `at`.
    """
    return _transfer_at(
        analysis,
        _Transfer(*_signals(signal, over, reference)),
        at,
        ("gain", "dB", _core.Measure.gain),
    )


def phase(
    analysis: str,
    signal: str | None = None,
    *,
    at: Quantity,
    over: str | None = None,
    reference: str | None = None,
) -> Measure:
    """Measures small-signal phase of `signal` at frequency `at`, in degrees.

    Computes the phase of `signal` relative to `over` (or excitation), unwrapped from
    the lowest frequency and interpolated over log frequency. Omitting `signal`
    measures the loop-gain phase of a stability analysis.

    Args:
        analysis: Plan key naming the target AC or stability analysis.
        signal: Output unknown name for AC analyses, or ``None`` for stability loop
            phase.
        at: Frequency at which unwrapped phase is evaluated.
        over: Optional denominator unknown name for a ratio transfer phase.
        reference: Optional negative reference net for a differential output voltage.

    Returns:
        Measure evaluating unwrapped small-signal phase in degrees at `at`.
    """
    return _transfer_at(
        analysis,
        _Transfer(*_signals(signal, over, reference)),
        at,
        ("phase", "°", _core.Measure.phase),
    )


def bandwidth(
    analysis: str,
    signal: str | None = None,
    *,
    drop: float = 3.0,
    over: str | None = None,
    reference: str | None = None,
) -> Measure:
    """Measures the frequency where gain of `signal` first falls `drop` decibels.

    Compares gain of `signal` over `over` (or unit excitation, or stability loop gain
    when `signal` is omitted) against its lowest-frequency gain, interpolating the
    crossing over log frequency.

    Args:
        analysis: Plan key naming the target AC or stability analysis.
        signal: Output unknown name for AC analyses, or ``None`` for stability loop
            gain.
        drop: Positive attenuation in decibels below lowest-frequency gain.
        over: Optional denominator unknown name for a ratio transfer gain.
        reference: Optional negative reference net for a differential output voltage.

    Returns:
        Measure evaluating bandwidth in hertz at the requested `drop`.

    Raises:
        ValueError: When `drop` is not positive.
    """
    if not drop > 0:
        message = f"a bandwidth's drop is positive, not {drop}"
        raise ValueError(message)
    transfer = _Transfer(*_signals(signal, over, reference))

    def stays(numbers: Sequence[float]) -> str:
        lowest = format_si(numbers[0], "Hz")
        return (
            f"the gain of {transfer} stays within {drop:g} dB of its gain at {lowest}"
        )

    return _transfer_measure(
        analysis,
        transfer,
        f"{drop:g} dB bandwidth of {transfer}",
        FREQUENCY,
        lambda resolved: (_core.Measure.bandwidth(resolved, drop), {"stays": stays}),
    )


def crossover(
    analysis: str,
    signal: str | None = None,
    *,
    over: str | None = None,
    reference: str | None = None,
) -> Measure:
    """Measures the frequency where gain of `signal` first falls through 0 dB.

    Evaluates gain of `signal` over `over` (or unit excitation) in an AC response, or
    loop gain of a stability analysis when `signal` is omitted. The crossing frequency
    is interpolated over log frequency.

    Args:
        analysis: Plan key naming the target AC or stability analysis.
        signal: Output unknown name for AC analyses, or ``None`` for stability loop
            gain.
        over: Optional denominator unknown name for a ratio transfer gain.
        reference: Optional negative reference net for a differential output voltage.

    Returns:
        Measure evaluating unity-gain crossover frequency in hertz.
    """
    transfer = _Transfer(*_signals(signal, over, reference))
    return _transfer_measure(
        analysis,
        transfer,
        f"crossover frequency of {transfer}",
        FREQUENCY,
        lambda resolved: (_core.Measure.crossover(resolved), transfer.unity),
    )


def phase_margin(
    analysis: str,
    signal: str | None = None,
    *,
    over: str | None = None,
    reference: str | None = None,
) -> Measure:
    """Measures phase margin (180° plus phase at 0 dB gain crossover) of `signal`.

    Evaluates gain and phase of `signal` relative to `over` (or excitation) in an AC
    response. Omitting `signal` measures the loop phase margin of a stability
    analysis.

    Args:
        analysis: Plan key naming the target AC or stability analysis.
        signal: Output unknown name for AC analyses, or ``None`` for stability loop
            gain.
        over: Optional denominator unknown name for a ratio transfer response.
        reference: Optional negative reference net for a differential output voltage.

    Returns:
        Measure evaluating phase margin in degrees at unity-gain crossover.
    """
    transfer = _Transfer(*_signals(signal, over, reference))
    return _transfer_measure(
        analysis,
        transfer,
        f"phase margin of {transfer}",
        "°",
        lambda resolved: (_core.Measure.phase_margin(resolved), transfer.unity),
    )


def gain_margin(
    analysis: str,
    signal: str | None = None,
    *,
    over: str | None = None,
    reference: str | None = None,
) -> Measure:
    """Measures gain margin (negative dB gain at -180° phase crossover) of `signal`.

    Evaluates gain and unwrapped phase of `signal` relative to `over` (or excitation)
    in an AC response. Omitting `signal` measures the loop gain margin of a stability
    analysis.

    Args:
        analysis: Plan key naming the target AC or stability analysis.
        signal: Output unknown name for AC analyses, or ``None`` for stability loop
            gain.
        over: Optional denominator unknown name for a ratio transfer response.
        reference: Optional negative reference net for a differential output voltage.

    Returns:
        Measure evaluating gain margin in decibels at the -180° phase crossing.
    """
    transfer = _Transfer(*_signals(signal, over, reference))
    inverted = f"the phase of {transfer} never falls through -180°"
    return _transfer_measure(
        analysis,
        transfer,
        f"gain margin of {transfer}",
        "dB",
        lambda resolved: (
            _core.Measure.gain_margin(resolved),
            {"uncrossed": lambda _: inverted},
        ),
    )


def maximum(
    analysis: str,
    signal: str,
    *,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measure:
    """Measures the maximum value of `signal` in a transient or DC sweep.

    The evaluation window spans from `start` to `stop` (times or sweep levels),
    defaulting to the full axis.

    Args:
        analysis: Plan key naming the target transient or DC sweep analysis.
        signal: Circuit unknown name to evaluate.
        start: Inclusive window start time or sweep level, or ``None`` for axis start.
        stop: Inclusive window stop time or sweep level, or ``None`` for axis end.
        reference: Optional negative reference net for a differential voltage reading.

    Returns:
        Measure evaluating the maximum of `signal` across the window.
    """
    window = _Window(_Signal(signal, reference), start, stop)
    return _statistic("maximum", analysis, window, _core.Measure.maximum)


def minimum(
    analysis: str,
    signal: str,
    *,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measure:
    """Measures the minimum value of `signal` in a transient or DC sweep.

    The evaluation window spans from `start` to `stop` (times or sweep levels),
    defaulting to the full axis.

    Args:
        analysis: Plan key naming the target transient or DC sweep analysis.
        signal: Circuit unknown name to evaluate.
        start: Inclusive window start time or sweep level, or ``None`` for axis start.
        stop: Inclusive window stop time or sweep level, or ``None`` for axis end.
        reference: Optional negative reference net for a differential voltage reading.

    Returns:
        Measure evaluating the minimum of `signal` across the window.
    """
    window = _Window(_Signal(signal, reference), start, stop)
    return _statistic("minimum", analysis, window, _core.Measure.minimum)


def peak_to_peak(
    analysis: str,
    signal: str,
    *,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measure:
    """Measures the peak-to-peak span of `signal` in a transient or DC sweep.

    The evaluation window spans from `start` to `stop` (times or sweep levels),
    defaulting to the full axis.

    Args:
        analysis: Plan key naming the target transient or DC sweep analysis.
        signal: Circuit unknown name to evaluate.
        start: Inclusive window start time or sweep level, or ``None`` for axis start.
        stop: Inclusive window stop time or sweep level, or ``None`` for axis end.
        reference: Optional negative reference net for a differential voltage reading.

    Returns:
        Measure evaluating peak-to-peak amplitude of `signal` across the window.
    """
    window = _Window(_Signal(signal, reference), start, stop)
    return _statistic("peak-to-peak", analysis, window, _core.Measure.peak_to_peak)


def mean(
    analysis: str,
    signal: str,
    *,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measure:
    """Measures the trapezoidal mean of `signal` in a transient or DC sweep.

    The evaluation window spans from `start` to `stop` (times or sweep levels),
    defaulting to the full axis.

    Args:
        analysis: Plan key naming the target transient or DC sweep analysis.
        signal: Circuit unknown name to evaluate.
        start: Inclusive window start time or sweep level, or ``None`` for axis start.
        stop: Inclusive window stop time or sweep level, or ``None`` for axis end.
        reference: Optional negative reference net for a differential voltage reading.

    Returns:
        Measure evaluating the trapezoidal average of `signal` across the window.
    """
    window = _Window(_Signal(signal, reference), start, stop)
    return _statistic("mean", analysis, window, _core.Measure.mean)


def rms(
    analysis: str,
    signal: str,
    *,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measure:
    """Measures the root-mean-square value of `signal` in a transient or DC sweep.

    The evaluation window spans from `start` to `stop` (times or sweep levels),
    defaulting to the full axis, integrating the squared waveform by the trapezoidal
    rule.

    Args:
        analysis: Plan key naming the target transient or DC sweep analysis.
        signal: Circuit unknown name to evaluate.
        start: Inclusive window start time or sweep level, or ``None`` for axis start.
        stop: Inclusive window stop time or sweep level, or ``None`` for axis end.
        reference: Optional negative reference net for a differential voltage reading.

    Returns:
        Measure evaluating the root-mean-square level of `signal` across the window.
    """
    window = _Window(_Signal(signal, reference), start, stop)
    return _statistic("rms", analysis, window, _core.Measure.rms)


def rise_time(  # noqa: PLR0913 - a window, its signal and the two bounds
    analysis: str,
    signal: str,
    *,
    lower: float = 0.1,
    upper: float = 0.9,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measure:
    """Measures the transition time of `signal` between `lower` and `upper` fractions.

    The step spans from the signal value at window start to the value at window stop,
    measuring elapsed time from fraction `lower` to fraction `upper`.

    Args:
        analysis: Plan key naming the target transient analysis.
        signal: Circuit unknown name to evaluate.
        lower: Fractional step threshold marking transition start, strictly above 0.
        upper: Fractional step threshold marking transition completion, strictly below
            1.
        start: Inclusive window start time, or ``None`` for transient start.
        stop: Inclusive window stop time, or ``None`` for transient stop.
        reference: Optional negative reference net for a differential voltage reading.

    Returns:
        Measure evaluating transition time in seconds from `lower` to `upper`.

    Raises:
        ValueError: Unless 0 < `lower` < `upper` < 1.
    """
    if not 0 < lower < upper < 1:
        message = (
            f"a rise time's bounds lie between 0 and 1, lower first: {lower}, {upper}"
        )
        raise ValueError(message)
    window = _Window(_Signal(signal, reference), start, stop)
    unresolved = f"{window} steps too little to resolve its levels"
    return _transient_measure(
        analysis,
        window,
        ("a rise time", f"rise time of {window}", TIME),
        lambda resolved: (
            _core.Measure.rise_time(resolved.signal, resolved.window, lower, upper),
            {"uncrossed": lambda _: unresolved},
        ),
    )


def overshoot(
    analysis: str,
    signal: str,
    *,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measure:
    """Measures step overshoot of `signal` in a transient, in percent of the step.

    The step spans from the signal value at window start to the final value at window
    stop.

    Args:
        analysis: Plan key naming the target transient analysis.
        signal: Circuit unknown name to evaluate.
        start: Inclusive window start time, or ``None`` for transient start.
        stop: Inclusive window stop time, or ``None`` for transient stop.
        reference: Optional negative reference net for a differential voltage reading.

    Returns:
        Measure evaluating percentage step overshoot across the window.
    """
    window = _Window(_Signal(signal, reference), start, stop)
    return _transient_measure(
        analysis,
        window,
        ("an overshoot", f"overshoot of {window}", "%"),
        lambda resolved: (
            _core.Measure.overshoot(resolved.signal, resolved.window),
            {},
        ),
    )


def settling_time(  # noqa: PLR0913 - a window, its signal and the band
    analysis: str,
    signal: str,
    *,
    band: float = 0.02,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measure:
    """Measures settling time of `signal` within fractional `band` of its final value.

    The step spans from the signal value at window start to the final value at window
    stop; settling time is the elapsed time until `signal` remains inside `band` times
    the step around its final value.

    Args:
        analysis: Plan key naming the target transient analysis.
        signal: Circuit unknown name to evaluate.
        band: Positive fractional tolerance band around the final step value.
        start: Inclusive window start time, or ``None`` for transient start.
        stop: Inclusive window stop time, or ``None`` for transient stop.
        reference: Optional negative reference net for a differential voltage reading.

    Returns:
        Measure evaluating settling time in seconds within `band`.

    Raises:
        ValueError: When `band` is not positive.
    """
    if not band > 0:
        message = f"a settling band is positive, not {band}"
        raise ValueError(message)
    window = _Window(_Signal(signal, reference), start, stop)
    return _transient_measure(
        analysis,
        window,
        ("a settling time", f"{100 * band:g}% settling time of {window}", TIME),
        lambda resolved: (
            _core.Measure.settling_time(resolved.signal, resolved.window, band),
            {},
        ),
    )


def crossing(  # noqa: PLR0913 - a window, its signal, the threshold and the edge
    analysis: str,
    signal: str,
    threshold: Quantity,
    *,
    edge: Edge = "either",
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measure:
    """Measures the first time `signal` crosses `threshold` in direction `edge`.

    Args:
        analysis: Plan key naming the target transient analysis.
        signal: Circuit unknown name to evaluate.
        threshold: Signal level whose first crossing time is measured.
        edge: Crossing polarity: ``"rising"``, ``"falling"``, or ``"either"``.
        start: Inclusive window start time, or ``None`` for transient start.
        stop: Inclusive window stop time, or ``None`` for transient stop.
        reference: Optional negative reference net for a differential voltage reading.

    Returns:
        Measure evaluating the first crossing time in seconds.
    """
    _check(threshold, "the threshold")
    window = _Window(_Signal(signal, reference), start, stop)
    uncrossed = f"{window} does not cross {threshold}"
    way = "" if edge == "either" else f" {edge}"
    return _transient_measure(
        analysis,
        window,
        ("a crossing", f"first{way} crossing of {threshold} by {window}", TIME),
        lambda resolved: (
            _core.Measure.crossing(
                resolved.signal,
                resolved.window,
                threshold.to(resolved.dimension, "the threshold"),
                edge,
            ),
            {"uncrossed": lambda _: uncrossed},
        ),
        threshold=threshold,
    )


def frequency(  # noqa: PLR0913 - a window, its signal and the threshold
    analysis: str,
    signal: str,
    *,
    threshold: Quantity | None = None,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measure:
    """Measures oscillation frequency of `signal` from rising crossings of `threshold`.

    When `threshold` is omitted, the midpoint between the signal minimum and maximum is
    used. Frequency equals one fewer than the rising-crossing count divided by the span
    between the first and last crossings.

    Args:
        analysis: Plan key naming the target transient analysis.
        signal: Circuit unknown name to evaluate.
        threshold: Optional level defining rising crossings, defaulting to signal
            midpoint.
        start: Inclusive window start time, or ``None`` for transient start.
        stop: Inclusive window stop time, or ``None`` for transient stop.
        reference: Optional negative reference net for a differential voltage reading.

    Returns:
        Measure evaluating oscillation frequency in hertz across the window.
    """
    _check(threshold, "the threshold")
    window = _Window(_Signal(signal, reference), start, stop)

    def make(resolved: _Resolved) -> tuple[_core.Measure, _Explainer]:
        through = (
            None
            if threshold is None
            else threshold.to(resolved.dimension, "the threshold")
        )
        measure = _core.Measure.frequency(resolved.signal, resolved.window, through)

        def rises(numbers: Sequence[float]) -> str:
            middle = format_si(numbers[0], str(resolved.dimension))
            return f"{window} rises through {middle} fewer than twice"

        return measure, {"rises": rises}

    return _transient_measure(
        analysis,
        window,
        ("a frequency", f"frequency of {window}", FREQUENCY),
        make,
        threshold=threshold,
    )


def distortion(  # noqa: PLR0913 - a window, its signal, the fundamental, the harmonics
    analysis: str,
    signal: str,
    *,
    fundamental: Quantity,
    harmonics: int = 9,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measure:
    """Measures total harmonic distortion of `signal` in a transient, in decibels.

    Computes the root-sum-square amplitude of harmonics 2 through `harmonics` of
    `fundamental` divided by the fundamental amplitude in decibels, matching SPICE
    ``.four`` (-40 dB equals 1 %). Amplitudes integrate over the final fundamental
    period before window stop, or over all complete periods between `start` and `stop`
    when `start` is given. A pure tone with zero harmonic content reads -6000 dB.

    Args:
        analysis: Plan key naming the target transient analysis.
        signal: Circuit unknown name to evaluate.
        fundamental: Fundamental tone frequency.
        harmonics: Highest harmonic order included in the distortion sum.
        start: Optional window start time for multi-period Fourier integration.
        stop: Optional window stop time, defaulting to transient stop.
        reference: Optional negative reference net for a differential voltage reading.

    Returns:
        Measure evaluating total harmonic distortion in decibels.

    Raises:
        DimensionError: When `fundamental` does not carry frequency dimensions.
        ValueError: When `fundamental` is not positive, or `harmonics` is less than 2
            or exceeds the Nyquist limit for samples per period.
    """
    hertz = fundamental.to(FREQUENCY, "the fundamental")
    if not 0 < hertz < math.inf:
        message = (
            f"a distortion's fundamental is positive and finite, not {fundamental}"
        )
        raise ValueError(message)
    if not 2 <= harmonics < _core.SAMPLES_PER_PERIOD // 2:  # noqa: PLR2004 - first
        message = (
            f"a distortion counts harmonics up to between 2 and "
            f"{_core.SAMPLES_PER_PERIOD // 2 - 1}, not {harmonics}"
        )
        raise ValueError(message)
    window = _Window(_Signal(signal, reference), start, stop)
    explainer: _Explainer = {
        "short": lambda _: f"{window} spans less than a period of {fundamental}",
        "silent": lambda _: f"{window} has no component at {fundamental}",
    }
    return _transient_measure(
        analysis,
        window,
        ("a distortion", f"distortion of {window} at {fundamental}", "dB"),
        lambda resolved: (
            _core.Measure.distortion(
                resolved.signal, resolved.window, hertz, harmonics
            ),
            explainer,
        ),
    )


def noise_density(analysis: str, *, at: Quantity, referred: bool = False) -> Measure:
    """Measures spot noise spectral density of a noise analysis at frequency `at`.

    Computes the square root of power spectral density in V/√Hz or A/√Hz using
    power-law interpolation between frequency points. When `referred` is true, returns
    input-referred noise density in the units of the analysis input source.

    Args:
        analysis: Plan key naming the target noise analysis.
        at: Frequency at which spot noise spectral density is evaluated.
        referred: Flag selecting input-referred noise density instead of output noise.

    Returns:
        Measure evaluating spot noise spectral density at `at`.
    """
    _check(at, "at")

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        spectrum = _spectrum(result, "a noise density", referred=referred)
        point = _FREQUENCIES.convert(at, "at")
        unit = _noise_unit(spectrum, referred=referred) / rtHz.dimension
        explain = _explain(_FREQUENCIES.outside({"at": ("at", at)}))
        return ResolvedMeasure(
            _core.Measure.noise_density(point, referred), unit, explain
        )

    def check(result: results.AnalysisResult) -> list[str]:
        problems = _spectrum_problems(result, "a noise density", referred=referred)
        if (at_error := _FREQUENCIES.check(at, "at")) is not None:
            problems.append(at_error)
        return problems

    referral = " referred to the input" if referred else ""
    return _native(analysis, f"noise density{referral} at {at}", None, resolve, check)


def integrated_noise(
    analysis: str,
    *,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    referred: bool = False,
) -> Measure:
    """Measures integrated RMS noise of a noise analysis from `start` to `stop`.

    Computes the square root of power spectral density integrated over the frequency
    band (defaulting to the full analysis range) using SPICE power-law interpolation,
    in volts or amperes. When `referred` is true, returns input-referred RMS noise in
    the units of the analysis input source.

    Args:
        analysis: Plan key naming the target noise analysis.
        start: Lower integration frequency bound, or ``None`` for the lowest solved
            frequency.
        stop: Upper integration frequency bound, or ``None`` for the highest solved
            frequency.
        referred: Flag selecting input-referred RMS noise instead of output RMS noise.

    Returns:
        Measure evaluating integrated RMS noise across the frequency band.
    """
    _check(start, "start")
    _check(stop, "stop")

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        spectrum = _spectrum(result, "an integrated noise", referred=referred)
        low = None if start is None else _FREQUENCIES.convert(start, "start")
        high = None if stop is None else _FREQUENCIES.convert(stop, "stop")

        def band(numbers: Sequence[float]) -> str:
            # Delegate out-of-band errors to `results.integrate` for consistent wording.
            spectrum.rms(*numbers[:2], referred=referred)
            return f"the band from {start} to {stop} is empty"

        explainer: _Explainer = {
            "empty_band": lambda _: f"the band from {start} to {stop} is empty",
            "band": band,
        }
        points = {"start": ("start", start), "stop": ("stop", stop)}
        explain = _explain(_FREQUENCIES.outside(points), explainer)
        measure = _core.Measure.integrated_noise(low, high, referred)
        return ResolvedMeasure(
            measure, _noise_unit(spectrum, referred=referred), explain
        )

    def check(result: results.AnalysisResult) -> list[str]:
        problems = _spectrum_problems(result, "an integrated noise", referred=referred)
        for point, name in ((start, "start"), (stop, "stop")):
            if (
                point is not None
                and (point_error := _FREQUENCIES.check(point, name)) is not None
            ):
                problems.append(point_error)
        return problems

    referral = " referred to the input" if referred else ""
    lowest = "the lowest frequency" if start is None else str(start)
    highest = "the highest" if stop is None else str(stop)
    described = f"RMS noise{referral} from {lowest} to {highest}"
    return _native(analysis, described, None, resolve, check)


def period(analysis: str) -> Measure:
    """Measures the steady-state period of a periodic steady-state analysis.

    Args:
        analysis: Plan key naming the target periodic steady-state analysis.

    Returns:
        Measure evaluating steady-state period in seconds.
    """

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        if not isinstance(result, results.PeriodicSteadyStateResult):
            message = (
                "period reads a periodic steady state, "
                f"not {results.result_kind(result)}"
            )
            raise MeasureError(message)
        return ResolvedMeasure(_core.Measure.period(), TIME, _explain())

    return _native(analysis, "period", None, resolve)


def solutions(analysis: str) -> Measure:
    """Counts DC solutions found by an operating-points search, as a plain number.

    ``solutions("states") <= 1`` checks for a unique operating point, while
    ``solutions("states") >= 3`` checks that a latch has two stable states plus its
    metastable equilibrium.

    Args:
        analysis: Plan key naming the target operating-points search analysis.

    Returns:
        Measure counting the solved DC operating points in `analysis`.
    """

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        if not isinstance(result, results.OperatingPointsResult):
            message = (
                "solutions reads a search for operating points, "
                f"not {results.result_kind(result)}"
            )
            raise MeasureError(message)
        return ResolvedMeasure(_core.Measure.solutions(), "", _explain())

    return _native(analysis, "DC solutions", "", resolve)


def damping(analysis: str) -> Measure:
    """Measures the minimum damping ratio across all poles of a pole-zero analysis.

    Each pole's damping ratio equals ``-Re(p) / |p|``: 1 for a real left-half-plane
    pole, between 0 and 1 for an underdamped conjugate pair, and negative for an
    unstable right-half-plane pole. Requiring ``damping("pz") >= 0`` enforces stability
    and ``damping("pz") >= 0.5`` limits ringing.

    Args:
        analysis: Plan key naming the target pole-zero analysis.

    Returns:
        Measure evaluating the minimum damping ratio across all poles in `analysis`.
    """

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        _poles(result, "a damping")
        return ResolvedMeasure(_core.Measure.damping(), "", _explain(_POLES))

    return _native(analysis, "least damping ratio of the poles", "", resolve)


def dominant_pole(analysis: str) -> Measure:
    """Measures the natural frequency of the pole nearest the origin in a pole-zero run.

    Equals ``|p| / (2π)`` in hertz for the pole closest to the origin, which governs
    the slowest settling mode.

    Args:
        analysis: Plan key naming the target pole-zero analysis.

    Returns:
        Measure evaluating dominant-pole natural frequency in hertz.
    """

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        _poles(result, "a dominant pole")
        return ResolvedMeasure(
            _core.Measure.dominant_pole(), FREQUENCY, _explain(_POLES)
        )

    return _native(analysis, "dominant pole", None, resolve)


def _native(
    analysis: str,
    description: str,
    plain: results.Plain | None,
    resolve: _Resolve,
    check: _Diagnose | None = None,
) -> Measure:
    """Constructs a native engine-backed `Measure` from `resolve` and `check`."""

    def read(result: results.AnalysisResult) -> Reading:
        return resolve(result).read(result)

    return Measure(analysis, description, plain, read, resolve, check)


def _transfer_at(
    analysis: str,
    transfer: _Transfer,
    at: Quantity,
    kind: tuple[str, results.Plain, Callable[[_core.Transfer, float], _core.Measure]],
) -> Measure:
    """Builds a small-signal measure evaluated at one frequency `at`."""
    label, unit, build = kind
    _check(at, "at")

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        measure = build(transfer.resolve(result), _FREQUENCIES.convert(at, "at"))
        return ResolvedMeasure(measure, unit, transfer.explain(at=at))

    def check(result: results.AnalysisResult) -> list[str]:
        problems = transfer.problems(result)
        if (at_error := _FREQUENCIES.check(at, "at")) is not None:
            problems.append(at_error)
        return problems

    return _native(analysis, f"{label} of {transfer} at {at}", unit, resolve, check)


def _transfer_measure(
    analysis: str,
    transfer: _Transfer,
    description: str,
    unit: Dimension | results.Plain,
    make: Callable[[_core.Transfer], tuple[_core.Measure, _Explainer]],
) -> Measure:
    """Builds a small-signal transfer or stability measure without a frequency point."""
    plain = None if isinstance(unit, Dimension) else unit

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        measure, extra = make(transfer.resolve(result))
        return ResolvedMeasure(measure, unit, transfer.explain(extra))

    return _native(analysis, description, plain, resolve, transfer.problems)


def _transient_measure(
    analysis: str,
    window: _Window,
    what: tuple[str, str, Dimension | results.Plain],
    make: Callable[[_Resolved], tuple[_core.Measure, _Explainer]],
    *,
    threshold: Quantity | None = None,
) -> Measure:
    """Builds a transient-only windowed measure with optional threshold check."""
    noun, description, unit = what
    plain = None if isinstance(unit, Dimension) else unit

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        resolved = window.resolve(result, noun, transient=True)
        measure, extra = make(resolved)
        return ResolvedMeasure(measure, unit, resolved.explain(extra))

    def check(result: results.AnalysisResult) -> list[str]:
        problems = window.problems(result, noun, transient=True)
        if (
            threshold is not None
            and isinstance(result, results.TransientResult)
            and (
                threshold_error := _check_threshold(
                    window.signal, result.unknowns, threshold
                )
            )
            is not None
        ):
            problems.append(threshold_error)
        return problems

    return _native(analysis, description, plain, resolve, check)


def _check_threshold(
    signal: _Signal, unknowns_list: tuple[Unknown, ...], threshold: Quantity
) -> str | None:
    """Checks `threshold` against `signal`'s dimension when `signal` is known."""
    try:
        index = results.column(unknowns_list, signal.name)
    except KeyError:
        return None
    dimension = unknowns.DIMENSIONS[unknowns_list[index].nature]
    try:
        threshold.to(dimension, "the threshold")
    except DimensionError as error:
        return str(error.args[0])
    return None


@dataclass(frozen=True, slots=True)
class _Signal:
    """Circuit unknown or differential voltage relative to a negative reference net."""

    name: str
    reference: str | None = None

    def resolve(
        self, unknowns_list: tuple[Unknown, ...]
    ) -> tuple[_core.Signal, Dimension]:
        """Resolves unknown indices and physical dimension against `unknowns_list`.

        Raises:
            KeyError: When `name` or `reference` is absent from `unknowns_list`.
            MeasureError: When `reference` is set and either unknown is not a voltage.
        """
        index = results.column(unknowns_list, self.name)
        dimension = unknowns.DIMENSIONS[unknowns_list[index].nature]
        if self.reference is None:
            return (index, None), dimension
        base = results.column(unknowns_list, self.reference)
        for position in (index, base):
            unknown = unknowns_list[position]
            if unknown.nature != "voltage":
                message = (
                    f"{self} names the {unknown.nature} {unknown.name}, not a voltage"
                )
                raise MeasureError(message)
        return (index, base), dimension

    def problems(self, unknowns_list: tuple[Unknown, ...]) -> list[str]:
        """Collects all resolution errors for `self` against `unknowns_list`."""
        found: list[str] = []
        index: int | None = None
        try:
            index = results.column(unknowns_list, self.name)
        except KeyError as error:
            found.append(str(error.args[0]))
        if self.reference is None:
            return found
        base: int | None = None
        try:
            base = results.column(unknowns_list, self.reference)
        except KeyError as error:
            found.append(str(error.args[0]))
        for position in (index, base):
            if position is None:
                continue
            unknown = unknowns_list[position]
            if unknown.nature != "voltage":
                found.append(
                    f"{self} names the {unknown.nature} {unknown.name}, not a voltage"
                )
        return found

    @override
    def __str__(self) -> str:
        """Formats the signal label, such as ``out`` or ``out relative to ref``."""
        if self.reference is None:
            return self.name
        return f"{self.name} relative to {self.reference}"


@dataclass(frozen=True, slots=True)
class _Axis:
    """Coordinate axis of an analysis result: times, sweep levels, or frequencies.

    Parameter DC sweeps carry no fixed dimension on the result, so axis points keep
    the supplied dimension directly.
    """

    dimension: Dimension | None
    noun: str
    """Human-readable axis quantity name, such as ``time``."""

    def convert(self, point: Quantity, name: str) -> float:
        """Converts `point` to SI base units matching the axis dimension.

        Raises:
            DimensionError: When `point` does not match the axis dimension.
        """
        return point.to(
            point.dimension if self.dimension is None else self.dimension, name
        )

    def check(self, point: Quantity, name: str) -> str | None:
        """Returns the dimension error message for `point`, or `None` when valid."""
        try:
            self.convert(point, name)
        except DimensionError as error:
            return str(error.args[0])
        return None

    def outside(self, points: Mapping[str, tuple[str, Quantity | None]]) -> _Explainer:
        """Builds out-of-range error formatters for the axis keys in `points`.

        Each entry maps an engine key (`at`, `start`, or `stop`) to a display label and
        requested quantity.
        """

        def explain(
            name: str, point: Quantity | None
        ) -> Callable[[Sequence[float]], str]:
            def outside(numbers: Sequence[float]) -> str:
                if self.dimension is None:
                    first, last = f"{numbers[0]:.4g}", f"{numbers[1]:.4g}"
                else:
                    symbol = str(self.dimension)
                    first, last = (
                        format_si(numbers[0], symbol),
                        format_si(numbers[1], symbol),
                    )
                return (
                    f"{name}, {point}, lies outside the {self.noun}s, {first} to {last}"
                )

            return outside

        return {
            f"outside_{which}": explain(name, point)
            for which, (name, point) in points.items()
        }


_FREQUENCIES: Final = _Axis(FREQUENCY, "frequency")
"""Frequency axis shared by AC, stability, and noise analyses."""


@dataclass(frozen=True, slots=True)
class _Resolved:
    """Windowed signal resolved to engine indices, SI bounds, and error formatters."""

    signal: _core.Signal
    window: _core.Window
    dimension: Dimension
    explainer: _Explainer

    def explain(self, *explainers: _Explainer) -> Callable[[_core.MeasureError], str]:
        """Builds an error formatter checking `explainers` before window defaults."""
        return _explain(*explainers, self.explainer)


@dataclass(frozen=True, slots=True)
class _Window:
    """Signal evaluated from `start` to `stop` across a transient or DC sweep."""

    signal: _Signal
    start: Quantity | None
    stop: Quantity | None

    def __post_init__(self) -> None:
        _check(self.start, "the start")
        _check(self.stop, "the stop")

    def resolve(
        self, result: results.AnalysisResult, what: str, *, transient: bool
    ) -> _Resolved:
        """Resolves this window against `result` as engine indices and SI bounds.

        Requires a transient result when `transient` is true, or either a transient or
        a DC sweep otherwise.

        Raises:
            MeasureError: When `result` has an incompatible analysis kind.
            KeyError: When `result` lacks the signal or reference unknown.
            DimensionError: When `start` or `stop` fails the axis dimension check.
        """
        if transient and not isinstance(result, results.TransientResult):
            message = f"{what} reads a transient, not {results.result_kind(result)}"
            raise MeasureError(message)
        axis = _axis(result, what)
        signal, dimension = self.signal.resolve(result.unknowns)
        window = (
            None if self.start is None else axis.convert(self.start, "the start"),
            None if self.stop is None else axis.convert(self.stop, "the stop"),
        )
        level = str(dimension)
        explainer: _Explainer = {
            **axis.outside(
                {"start": ("the start", self.start), "stop": ("the stop", self.stop)}
            ),
            "empty_window": lambda _: f"the window of {self} is empty",
            "flat": lambda numbers: (
                f"{self} ends where it starts, at {format_si(numbers[0], level)}"
            ),
        }
        return _Resolved(signal, window, dimension, explainer)

    def problems(
        self, result: results.AnalysisResult, what: str, *, transient: bool
    ) -> list[str]:
        """Collects static problems resolving this window against `result`."""
        if transient and not isinstance(result, results.TransientResult):
            return [f"{what} reads a transient, not {results.result_kind(result)}"]
        try:
            axis = _axis(result, what)
        except MeasureError as error:
            return [str(error.args[0])]
        found = self.signal.problems(result.unknowns)
        for point, name in ((self.start, "the start"), (self.stop, "the stop")):
            if (
                point is not None
                and (point_error := axis.check(point, name)) is not None
            ):
                found.append(point_error)
        return found

    @override
    def __str__(self) -> str:
        """Formats the signal and window span, such as ``out from 1 µs to 2 µs``."""
        start = "" if self.start is None else f" from {self.start}"
        stop = "" if self.stop is None else f" to {self.stop}"
        return f"{self.signal}{start}{stop}"


@dataclass(frozen=True, slots=True)
class _Transfer:
    """Small-signal transfer target: `signal` over `divisor`, or stability loop gain."""

    signal: _Signal | None
    divisor: _Signal | None

    @property
    def unity(self) -> _Explainer:
        """Error formatter for an AC or stability response that never crosses 0 dB."""
        message = f"the gain of {self} never falls through 0 dB"
        return {"uncrossed": lambda _: message}

    def _mismatch(self, result: results.AnalysisResult) -> str:
        """Formats the kind-mismatch error when `result` does not fit this transfer."""
        if isinstance(result, results.StabilityResult):
            return f"a loop gain is read without an output, not at {self.signal}"
        if isinstance(result, results.AcResult):
            return "a small-signal response is read at an output"
        return (
            "a gain or a phase reads a small-signal response or a loop gain, "
            f"not {results.result_kind(result)}"
        )

    def resolve(self, result: results.AnalysisResult) -> _core.Transfer:
        """Resolves this transfer against `result`, returning None for loop gain.

        Raises:
            MeasureError: When `result` is neither an AC response with a signal nor a
                stability analysis without a signal.
            KeyError: When `result` lacks the signal, reference, or divisor unknown.
        """
        if isinstance(result, results.StabilityResult) and self.signal is None:
            return None
        if isinstance(result, results.AcResult) and self.signal is not None:
            signal, _ = self.signal.resolve(result.unknowns)
            if self.divisor is None:
                return signal, None
            divisor, _ = self.divisor.resolve(result.unknowns)
            return signal, divisor
        raise MeasureError(self._mismatch(result))

    def problems(self, result: results.AnalysisResult) -> list[str]:
        """Collects static problems resolving this transfer against `result`."""
        if isinstance(result, results.StabilityResult) and self.signal is None:
            return []
        if isinstance(result, results.AcResult) and self.signal is not None:
            found = self.signal.problems(result.unknowns)
            if self.divisor is not None:
                found.extend(self.divisor.problems(result.unknowns))
            return found
        return [self._mismatch(result)]

    def explain(
        self, *explainers: _Explainer, at: Quantity | None = None
    ) -> Callable[[_core.MeasureError], str]:
        """Builds an error formatter for transfer failures, checking `explainers` first.

        Includes frequency `at` in out-of-range diagnostics when supplied.
        """
        transfer: _Explainer = {
            "vanishes": lambda _: f"{self.divisor} vanishes in the response",
            "unpositive": lambda numbers: (
                "a response is read over log frequency, so not at "
                f"{format_si(numbers[0], 'Hz')}"
            ),
            **_FREQUENCIES.outside({"at": ("at", at)}),
        }
        return _explain(*explainers, transfer)

    @override
    def __str__(self) -> str:
        """Formats the transfer label, such as ``out``, ``out over in``, or loop."""
        if self.signal is None:
            return "the loop"
        if self.divisor is None:
            return str(self.signal)
        return f"{self.signal} over {self.divisor.name}"


def _statistic(
    name: str,
    analysis: str,
    window: _Window,
    statistic: Callable[[_core.Signal, _core.Window], _core.Measure],
) -> Measure:
    """Builds a windowed statistic `Measure` backed by engine factory `statistic`."""

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        resolved = window.resolve(result, f"a {name}", transient=False)
        measure = statistic(resolved.signal, resolved.window)
        return ResolvedMeasure(measure, resolved.dimension, resolved.explain())

    return _native(
        analysis,
        f"{name} of {window}",
        None,
        resolve,
        lambda result: window.problems(result, f"a {name}", transient=False),
    )


def _axis(result: results.AnalysisResult, what: str) -> _Axis:
    """Returns the coordinate axis descriptor for a transient or DC sweep result.

    Raises:
        MeasureError: When `result` is neither a transient nor a DC sweep.
    """
    match result:
        case results.TransientResult():
            return _Axis(TIME, "time")
        case results.DcSweepResult():
            if result.nature is None:
                label = (
                    f"value of {result.source}'s {result.parameter}"
                    if result.parameter is not None
                    else f"level of {result.source}"
                )
                return _Axis(None, label)
            return _Axis(
                unknowns.DIMENSIONS[result.nature], f"level of {result.source}"
            )
        case _:
            message = (
                f"{what} reads a transient or a DC sweep, "
                f"not {results.result_kind(result)}"
            )
            raise MeasureError(message)


def _signals(
    signal: str | None, over: str | None, reference: str | None
) -> tuple[_Signal | None, _Signal | None]:
    """Parses `(signal, divisor)` targets for a small-signal or loop-gain measure.

    Raises:
        ValueError: When `over` or `reference` is provided without `signal`.
    """
    if signal is None:
        if over is not None or reference is not None:
            message = (
                "a loop gain is read without an output, so without over or a reference"
            )
            raise ValueError(message)
        return None, None
    return _Signal(signal, reference), None if over is None else _Signal(over)


def _spectrum(
    result: results.AnalysisResult, what: str, *, referred: bool
) -> results.NoiseResult:
    """Validates that `result` is a `NoiseResult`, with an input source if `referred`.

    Raises:
        MeasureError: When `result` is not a noise result or lacks an input source.
    """
    if not isinstance(result, results.NoiseResult):
        message = f"{what} reads a noise analysis, not {results.result_kind(result)}"
        raise MeasureError(message)
    if referred and result.input is None:
        message = "the noise analysis names no source to refer its noise to"
        raise MeasureError(message)
    return result


def _spectrum_problems(
    result: results.AnalysisResult, what: str, *, referred: bool
) -> list[str]:
    """Collects static problems validating `result` as a noise analysis for `what`."""
    try:
        _spectrum(result, what, referred=referred)
    except MeasureError as error:
        return [str(error.args[0])]
    return []


def _poles(result: results.AnalysisResult, what: str) -> None:
    """Checks that `result` is a `PoleZeroResult`.

    Raises:
        MeasureError: When `result` is not a pole-zero result.
    """
    if not isinstance(result, results.PoleZeroResult):
        message = (
            f"{what} reads a pole-zero analysis, not {results.result_kind(result)}"
        )
        raise MeasureError(message)


_POLES: Final[_Explainer] = {"no_poles": lambda _: "the circuit has no poles"}
"""Error formatter for a pole-zero result containing no poles."""


def _noise_unit(spectrum: results.NoiseResult, *, referred: bool) -> Dimension:
    """Returns the output or input-referred amplitude dimension of `spectrum`."""
    nature = spectrum.nature
    if referred and spectrum.input is not None:
        nature = spectrum.input.nature
    return unknowns.DIMENSIONS[nature]


_MALFORMED: Final[_Explainer] = {
    "empty": lambda _: "the result holds no points",
    "mismatch": lambda _: "the result holds more or fewer values than points",
    "unknown": lambda _: "the result holds fewer unknowns than its circuit",
    "kind": lambda _: "the engine reads no result of this kind",
    "unreferred": lambda _: "the noise analysis names no source to refer its noise to",
    "not_differentiable": lambda _: "the measure is not differentiable at this result",
}
"""Fallback error formatters for malformed engine result payloads."""


def _explain(*explainers: _Explainer) -> Callable[[_core.MeasureError], str]:
    """Builds an error formatter consulting `explainers` before `_MALFORMED`."""

    def explain(error: _core.MeasureError) -> str:
        reason, numbers = error.args
        for explainer in (*explainers, _MALFORMED):
            if reason in explainer:
                return explainer[reason](numbers)
        return f"the engine cannot read the result: {reason}"

    return explain


def _check(point: Quantity | None, name: str) -> None:
    """Checks that `point` is finite.

    Raises:
        ValueError: When it is infinite or not a number.
    """
    if point is not None:
        finite(point.value, name)


@dataclass(frozen=True, slots=True)
class MeasureSpec:
    """Declarative metadata for one public measure in `jolty.specs`."""

    category: MeasureCategory
    options: frozenset[str]
    factory: Callable[..., Measure]
    required: Mapping[str, str] = field(default_factory=dict)
    quantities: Mapping[str, MeasureDimension] = field(default_factory=dict)
    positional: tuple[str, ...] = ()


_WINDOW_OPTIONS: Final = frozenset({"signal", "reference", "start", "stop"})
_TRANSFER_OPTIONS: Final = frozenset({"signal", "reference", "over"})
_AXIS_WINDOW: Final[Mapping[str, MeasureDimension]] = {
    "start": "axis",
    "stop": "axis",
}
_TIME_WINDOW: Final[Mapping[str, MeasureDimension]] = {
    "start": "time",
    "stop": "time",
}
_AT_FREQUENCY: Final[Mapping[str, str]] = {"at": "a frequency as at, in Hz"}


def _transient_spec(
    factory: Callable[..., Measure],
    *extra_options: str,
    required: Mapping[str, str] | None = None,
    extra_quantities: Mapping[str, MeasureDimension] | None = None,
    positional: tuple[str, ...] = ("signal",),
) -> MeasureSpec:
    """Builds a `MeasureSpec` for a windowed transient measure."""
    return MeasureSpec(
        "transient",
        _WINDOW_OPTIONS | frozenset(extra_options),
        factory,
        required={} if required is None else required,
        quantities={
            **_TIME_WINDOW,
            **({} if extra_quantities is None else extra_quantities),
        },
        positional=positional,
    )


_MEASURE_SPECS: dict[str, MeasureSpec] = {
    "value": MeasureSpec(
        "value",
        frozenset({"signal", "reference", "at"}),
        value,
        quantities={"at": "axis"},
        positional=("signal",),
    ),
    **{
        factory.__name__: MeasureSpec(
            "transfer",
            _TRANSFER_OPTIONS | {"at"},
            factory,
            required=_AT_FREQUENCY,
            quantities={"at": "frequency"},
            positional=("signal",),
        )
        for factory in (gain, phase)
    },
    "bandwidth": MeasureSpec(
        "transfer",
        _TRANSFER_OPTIONS | {"drop"},
        bandwidth,
        positional=("signal",),
    ),
    **{
        factory.__name__: MeasureSpec(
            "transfer", _TRANSFER_OPTIONS, factory, positional=("signal",)
        )
        for factory in (crossover, phase_margin, gain_margin)
    },
    **{
        factory.__name__: MeasureSpec(
            "statistic",
            _WINDOW_OPTIONS,
            factory,
            quantities=_AXIS_WINDOW,
            positional=("signal",),
        )
        for factory in (maximum, minimum, peak_to_peak, mean, rms)
    },
    "rise_time": _transient_spec(rise_time, "lower", "upper"),
    "overshoot": _transient_spec(overshoot),
    "settling_time": _transient_spec(settling_time, "band"),
    "crossing": _transient_spec(
        crossing,
        "threshold",
        "edge",
        required={"threshold": "a threshold"},
        extra_quantities={"threshold": "signal"},
        positional=("signal", "threshold"),
    ),
    "frequency": _transient_spec(
        frequency, "threshold", extra_quantities={"threshold": "signal"}
    ),
    "distortion": _transient_spec(
        distortion,
        "fundamental",
        "harmonics",
        required={"fundamental": "a fundamental frequency, in Hz"},
        extra_quantities={"fundamental": "frequency"},
    ),
    "noise_density": MeasureSpec(
        "noise",
        frozenset({"at", "referred"}),
        noise_density,
        required=_AT_FREQUENCY,
        quantities={"at": "frequency"},
    ),
    "integrated_noise": MeasureSpec(
        "noise",
        frozenset({"start", "stop", "referred"}),
        integrated_noise,
        quantities={"start": "frequency", "stop": "frequency"},
    ),
    **{
        factory.__name__: MeasureSpec("plain", frozenset(), factory)
        for factory in (period, solutions, damping, dominant_pole)
    },
}

MEASURES: Final[Mapping[str, MeasureSpec]] = MappingProxyType(_MEASURE_SPECS)
"""Declarative specification of every public measure in `jolty.specs`."""
