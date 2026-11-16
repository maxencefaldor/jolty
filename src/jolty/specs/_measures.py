# Copyright 2026 Maxence Faldor
"""The measures a requirement reads from a result, and the loops' margins.

Each function names an analysis and what to read from its result, as
[`jolty.specs`][jolty.specs] describes. A measure resolves its signals, its result's
kind and its quantities' units against a result, and the engine reads the number, in
Rust, from the result's arrays: one definition of each measure's arithmetic, which
a batch of runs reads without Python.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, override

from jolty import _core, checks, results
from jolty.specs._requirements import (
    Measure,
    MeasureError,
    Plain,
    Reading,
    ResolvedMeasure,
)
from jolty.units import (
    CURRENT,
    DIMENSIONLESS,
    FREQUENCY,
    TEMPERATURE,
    TIME,
    VOLTAGE,
    Dimension,
    Quantity,
    finite,
    format_si,
    rtHz,
)
from jolty.unknowns import Unknown

type Edge = Literal["rising", "falling", "either"]
"""Which way a signal crosses a level."""

_DIMENSIONS: Final[dict[results.Nature, Dimension]] = {
    "voltage": VOLTAGE,
    "current": CURRENT,
    "temperature": TEMPERATURE,
    "state": DIMENSIONLESS,
}
"""The dimension of each quantity an unknown is."""

type _Explainer = Mapping[str, Callable[[Sequence[float]], str]]
"""Messages by the reasons the engine gives, each made from the numbers it states."""

type _Resolve = Callable[[results.AnalysisResult], ResolvedMeasure]
"""Resolves a measure against a result for the engine to read."""


def diagnose(
    solved: Mapping[str, results.AnalysisResult],
) -> tuple[checks.Diagnostic, ...]:
    """The diagnostics of the rules on the analyses' results `solved`, by analysis.

    Each stability analysis's loop gain raises low-phase-margin when it falls through 0
    dB with less than 45° of phase margin, and low-gain-margin when its phase falls
    through -180° with less than 6 dB of gain margin, each measured as `phase_margin`
    and `gain_margin` measure them.
    """
    found: list[checks.Diagnostic] = []
    for name, result in solved.items():
        if not isinstance(result, results.StabilityResult):
            continue
        loop = f"the loop broken at `{result.probe}`"
        try:
            phase = phase_margin(name).read(result).value
            unity = crossover(name).read(result).value
        except MeasureError:
            pass
        else:
            if phase < checks.MIN_PHASE_MARGIN:
                message = (
                    f"{loop} has a phase margin of {phase!r}° at its crossover, "
                    f"{unity!r} Hz"
                )
                found.append(_diagnostic(checks.LOW_PHASE_MARGIN, message))
        try:
            gain = gain_margin(name).read(result).value
        except MeasureError:
            continue
        if gain < checks.MIN_GAIN_MARGIN:
            message = f"{loop} has a gain margin of {gain!r} dB"
            found.append(_diagnostic(checks.LOW_GAIN_MARGIN, message))
    return tuple(found)


def _diagnostic(rule: checks.Rule, message: str) -> checks.Diagnostic:
    return checks.Diagnostic(
        rule=rule.name, level=rule.level, message=message, instances=(), nets=()
    )


def value(
    analysis: str,
    signal: str,
    *,
    at: Quantity | None = None,
    reference: str | None = None,
) -> Measure:
    """The value of `signal` at an operating point, a time or a level.

    In a transient or a DC sweep, the value is interpolated `at` a time or a level of
    the swept source.
    """
    traces = _Signal(signal, reference)
    _check(at, "at")

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        match result:
            case results.OperatingPointResult() if at is None:
                resolved, dimension = traces.resolve(result.unknowns)
                return ResolvedMeasure(
                    _core.Measure.value(resolved, None), dimension, _explain()
                )
            case results.OperatingPointResult():
                message = f"an operating point holds one value, not one at {at}"
                raise MeasureError(message)
            case (
                results.AcResult()
                | results.SensitivityResult()
                | results.StabilityResult()
                | results.NoiseResult()
                | results.PoleZeroResult()
                | results.OperatingPointsResult()
            ):
                message = (
                    "a value reads an operating point, a transient or a DC sweep, "
                    f"not {_kind(result)}"
                )
                raise MeasureError(message)
        axis = _axis(result, "a value")
        resolved, dimension = traces.resolve(result.unknowns)
        if at is None:
            message = f"{_kind(result)} holds a value per {axis.noun}, and at is unset"
            raise MeasureError(message)
        point = axis.convert(at, "at")
        measure = _core.Measure.value(resolved, point)
        return ResolvedMeasure(
            measure, dimension, _explain(axis.outside({"at": ("at", at)}))
        )

    described = str(traces) if at is None else f"{traces} at {at}"
    return _native(analysis, described, None, resolve)


def gain(
    analysis: str,
    output: str | None = None,
    *,
    at: Quantity,
    over: str | None = None,
    reference: str | None = None,
) -> Measure:
    """The gain of `output` at a frequency of a small-signal response, in decibels.

    The gain is of `output` over `over`, or per unit excitation, interpolated over log
    frequency `at` the frequency. Without an output, it is the loop gain of a
    stability analysis.
    """
    transfer = _Transfer(*_signals(output, over, reference))
    _check(at, "at")

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        resolved = transfer.resolve(result)
        measure = _core.Measure.gain(resolved, _FREQUENCIES.convert(at, "at"))
        return ResolvedMeasure(measure, "dB", transfer.explain(at=at))

    return _native(analysis, f"gain of {transfer} at {at}", "dB", resolve)


def phase(
    analysis: str,
    output: str | None = None,
    *,
    at: Quantity,
    over: str | None = None,
    reference: str | None = None,
) -> Measure:
    """The phase of `output` at a frequency of a small-signal response, in degrees.

    The phase is relative to `over`, or to the excitation, unwrapped from the lowest
    frequency and interpolated over log frequency `at` the frequency. Without an output,
    it is the loop gain's, of a stability analysis.
    """
    transfer = _Transfer(*_signals(output, over, reference))
    _check(at, "at")

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        resolved = transfer.resolve(result)
        measure = _core.Measure.phase(resolved, _FREQUENCIES.convert(at, "at"))
        return ResolvedMeasure(measure, "°", transfer.explain(at=at))

    return _native(analysis, f"phase of {transfer} at {at}", "°", resolve)


def bandwidth(
    analysis: str,
    output: str | None = None,
    *,
    drop: float = 3.0,
    over: str | None = None,
    reference: str | None = None,
) -> Measure:
    """The frequency where the gain of `output` first falls `drop` decibels.

    The gain, of `output` over `over` or per unit excitation, or the loop gain of a
    stability analysis without an output, falls below its gain at the lowest frequency;
    the frequency is interpolated over log frequency.

    Raises:
        ValueError: When `drop` is not positive.
    """
    if not drop > 0:
        message = f"a bandwidth's drop is positive, not {drop}"
        raise ValueError(message)
    transfer = _Transfer(*_signals(output, over, reference))

    def stays(numbers: Sequence[float]) -> str:
        lowest = format_si(numbers[0], "Hz")
        return (
            f"the gain of {transfer} stays within {drop:g} dB of its gain at {lowest}"
        )

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        measure = _core.Measure.bandwidth(transfer.resolve(result), drop)
        return ResolvedMeasure(measure, FREQUENCY, transfer.explain({"stays": stays}))

    return _native(analysis, f"{drop:g} dB bandwidth of {transfer}", None, resolve)


def crossover(
    analysis: str,
    output: str | None = None,
    *,
    over: str | None = None,
    reference: str | None = None,
) -> Measure:
    """The frequency where the gain of `output` first falls through 0 dB.

    The gain is of `output` over `over`, or per unit excitation, in a small-signal
    response; without an output, it is the loop gain of a stability analysis, and this
    its crossover frequency. It is interpolated over log frequency.
    """
    transfer = _Transfer(*_signals(output, over, reference))

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        measure = _core.Measure.crossover(transfer.resolve(result))
        return ResolvedMeasure(measure, FREQUENCY, transfer.explain(transfer.unity))

    return _native(analysis, f"crossover frequency of {transfer}", None, resolve)


def phase_margin(
    analysis: str,
    output: str | None = None,
    *,
    over: str | None = None,
    reference: str | None = None,
) -> Measure:
    """180° plus the phase of `output` where its gain first falls through 0 dB.

    The gain and the phase are of `output` relative to `over`, or to the excitation, in
    a small-signal response. Without an output, they are the loop gain's, of a
    stability analysis, and this is the loop's phase margin.
    """
    transfer = _Transfer(*_signals(output, over, reference))

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        measure = _core.Measure.phase_margin(transfer.resolve(result))
        return ResolvedMeasure(measure, "°", transfer.explain(transfer.unity))

    return _native(analysis, f"phase margin of {transfer}", "°", resolve)


def gain_margin(
    analysis: str,
    output: str | None = None,
    *,
    over: str | None = None,
    reference: str | None = None,
) -> Measure:
    """Minus the gain of `output` where its phase first falls through -180°, in dB.

    The gain and the phase, unwrapped from the lowest frequency, are of `output`
    relative to `over`, or to the excitation, in a small-signal response. Without an
    output, they are the loop gain's, of a stability analysis, and this is the loop's
    gain margin.
    """
    transfer = _Transfer(*_signals(output, over, reference))
    inverted = f"the phase of {transfer} never falls through -180°"

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        measure = _core.Measure.gain_margin(transfer.resolve(result))
        return ResolvedMeasure(
            measure, "dB", transfer.explain({"uncrossed": lambda _: inverted})
        )

    return _native(analysis, f"gain margin of {transfer}", "dB", resolve)


def maximum(
    analysis: str,
    signal: str,
    *,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measure:
    """The highest value of `signal` in a transient or a DC sweep.

    The window runs from `start` to `stop`, times or levels, by default over all.
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
    """The lowest value of `signal` in a transient or a DC sweep.

    The window runs from `start` to `stop`, times or levels, by default over all.
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
    """The highest less the lowest value of `signal` in a transient or a DC sweep.

    The window runs from `start` to `stop`, times or levels, by default over all.
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
    """The mean of `signal` in a transient or a DC sweep, by the trapezoidal rule.

    The window runs from `start` to `stop`, times or levels, by default over all.
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
    """The root mean square of `signal` in a transient or a DC sweep.

    The window runs from `start` to `stop`, times or levels, by default over all, and
    the square is integrated by the trapezoidal rule.
    """
    window = _Window(_Signal(signal, reference), start, stop)
    return _statistic("rms", analysis, window, _core.Measure.rms)


def rise_time(  # noqa: PLR0913 - a window, its signal and the two levels
    analysis: str,
    signal: str,
    *,
    low: float = 0.1,
    high: float = 0.9,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measure:
    """The time a transient's `signal` takes to rise, or fall, through its step.

    The step runs from the signal's value at the window's start to its value at the
    window's end, and the time from `low` to `high` of the way.

    Raises:
        ValueError: Unless 0 < `low` < `high` < 1.
    """
    if not 0 < low < high < 1:
        message = f"a rise time's levels lie between 0 and 1, low first: {low}, {high}"
        raise ValueError(message)
    window = _Window(_Signal(signal, reference), start, stop)
    unresolved = f"{window} steps too little to resolve its levels"

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        resolved = window.resolve(result, "a rise time", transient=True)
        measure = _core.Measure.rise_time(resolved.signal, resolved.window, low, high)
        explain = resolved.explain({"uncrossed": lambda _: unresolved})
        return ResolvedMeasure(measure, TIME, explain)

    return _native(analysis, f"rise time of {window}", None, resolve)


def overshoot(
    analysis: str,
    signal: str,
    *,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measure:
    """How far a transient's `signal` passes its final value, in percent of its step.

    The step runs from the signal's value at the window's start to its final value, at
    the window's end.
    """
    window = _Window(_Signal(signal, reference), start, stop)

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        resolved = window.resolve(result, "an overshoot", transient=True)
        measure = _core.Measure.overshoot(resolved.signal, resolved.window)
        return ResolvedMeasure(measure, "%", resolved.explain())

    return _native(analysis, f"overshoot of {window}", "%", resolve)


def settling_time(  # noqa: PLR0913 - a window, its signal and the band
    analysis: str,
    signal: str,
    *,
    band: float = 0.02,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measure:
    """The time a transient's `signal` takes to settle within `band` of its step.

    The step runs from the signal's value at the window's start to its final value, at
    the window's end; the time runs from the window's start until the signal stays
    within `band` of the step from its final value.

    Raises:
        ValueError: When `band` is not positive.
    """
    if not band > 0:
        message = f"a settling band is positive, not {band}"
        raise ValueError(message)
    window = _Window(_Signal(signal, reference), start, stop)

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        resolved = window.resolve(result, "a settling time", transient=True)
        measure = _core.Measure.settling_time(resolved.signal, resolved.window, band)
        return ResolvedMeasure(measure, TIME, resolved.explain())

    described = f"{100 * band:g}% settling time of {window}"
    return _native(analysis, described, None, resolve)


def crossing(  # noqa: PLR0913 - a window, its signal, the level and the edge
    analysis: str,
    signal: str,
    level: Quantity,
    *,
    edge: Edge = "either",
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measure:
    """The first time a transient's `signal` crosses `level` the way `edge` says."""
    _check(level, "the level")
    window = _Window(_Signal(signal, reference), start, stop)
    uncrossed = f"{window} does not cross {level}"

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        resolved = window.resolve(result, "a crossing", transient=True)
        through = level.to(resolved.dimension, "the level")
        measure = _core.Measure.crossing(
            resolved.signal, resolved.window, through, edge
        )
        return ResolvedMeasure(
            measure, TIME, resolved.explain({"uncrossed": lambda _: uncrossed})
        )

    way = "" if edge == "either" else f" {edge}"
    described = f"first{way} crossing of {level} by {window}"
    return _native(analysis, described, None, resolve)


def frequency(  # noqa: PLR0913 - a window, its signal and the level
    analysis: str,
    signal: str,
    *,
    level: Quantity | None = None,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measure:
    """How often a transient's `signal` rises through `level`.

    The level is by default the midpoint of the signal's lowest and highest values, and
    the frequency is the rises less one over the time from the first rise to the last.
    """
    _check(level, "the level")
    window = _Window(_Signal(signal, reference), start, stop)

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        resolved = window.resolve(result, "a frequency", transient=True)
        through = None if level is None else level.to(resolved.dimension, "the level")
        measure = _core.Measure.frequency(resolved.signal, resolved.window, through)

        def rises(numbers: Sequence[float]) -> str:
            middle = format_si(numbers[0], str(resolved.dimension))
            return f"{window} rises through {middle} fewer than twice"

        return ResolvedMeasure(measure, FREQUENCY, resolved.explain({"rises": rises}))

    return _native(analysis, f"frequency of {window}", None, resolve)


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
    """The total harmonic distortion of a transient's `signal`, in decibels.

    It is the root sum of the squares of the amplitudes of harmonics 2 to `harmonics`
    of the `fundamental`, over the fundamental's amplitude, as SPICE's ``.four`` finds
    it, in decibels: -40 dB is 1 %. The amplitudes are read over the last period of the
    fundamental before the window's stop, as SPICE reads them, or over every whole
    period from the window's start to its stop when it has a start. A signal without
    harmonics reads -6000 dB.

    Raises:
        DimensionError: When the fundamental is not a frequency.
        ValueError: When the fundamental is not positive, or `harmonics` is less than 2
            or more than the samples of a period resolve.
    """
    hertz = fundamental.to(FREQUENCY, "the fundamental")
    if not 0 < hertz < math.inf:
        message = (
            f"a distortion's fundamental is positive and finite, not {fundamental}"
        )
        raise ValueError(message)
    if not 2 <= harmonics < _core.SAMPLES_PER_PERIOD // 2:  # noqa: PLR2004 - the first one
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

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        resolved = window.resolve(result, "a distortion", transient=True)
        measure = _core.Measure.distortion(
            resolved.signal, resolved.window, hertz, harmonics
        )
        return ResolvedMeasure(measure, "dB", resolved.explain(explainer))

    described = f"distortion of {window} at {fundamental}"
    return _native(analysis, described, "dB", resolve)


def noise_density(analysis: str, *, at: Quantity, referred: bool = False) -> Measure:
    """The noise density of a noise analysis's output at a frequency.

    It is the square root of the power density, in V/√Hz for a voltage and A/√Hz for a
    current, as datasheets give it, interpolated as a power law between frequencies
    `at` the frequency. Referred, it is the noise referred to the analysis's source,
    as that input's own noise, in the unit of the source's level.
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

    referral = " referred to the input" if referred else ""
    return _native(analysis, f"noise density{referral} at {at}", None, resolve)


def integrated_noise(
    analysis: str,
    *,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    referred: bool = False,
) -> Measure:
    """The RMS noise of a noise analysis's output from `start` to `stop`.

    It is the square root of the power density integrated over the band, by default
    every frequency of the analysis, each density a power law between frequencies as
    SPICE integrates it, in volts or amperes. Referred, it is the noise referred to the
    analysis's source, as that input's own noise, in the unit of the source's level.
    """
    _check(start, "start")
    _check(stop, "stop")

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        spectrum = _spectrum(result, "an integrated noise", referred=referred)
        low = None if start is None else _FREQUENCIES.convert(start, "start")
        high = None if stop is None else _FREQUENCIES.convert(stop, "stop")

        def band(numbers: Sequence[float]) -> str:
            # A band the frequencies do not hold, as one from the highest of them on:
            # the noise's integral raises what `results.integrate` raises for it.
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

    referral = " referred to the input" if referred else ""
    lowest = "the lowest frequency" if start is None else str(start)
    highest = "the highest" if stop is None else str(stop)
    described = f"RMS noise{referral} from {lowest} to {highest}"
    return _native(analysis, described, None, resolve)


def solutions(analysis: str) -> Measure:
    """How many DC solutions a search for operating points found, a plain number.

    ``solutions("states") <= 1`` asks for a unique operating point, and a latch's
    ``solutions("states") >= 3`` for its two states and the balance between them.
    """

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        if not isinstance(result, results.OperatingPointsResult):
            message = (
                f"solutions reads a search for operating points, not {_kind(result)}"
            )
            raise MeasureError(message)
        return ResolvedMeasure(_core.Measure.solutions(), "", _explain())

    return _native(analysis, "DC solutions", "", resolve)


def damping(analysis: str) -> Measure:
    """The least damping ratio of a pole-zero analysis's poles, a plain number.

    A pole's damping ratio is minus its real part over its magnitude: 1 for a real pole
    in the left half-plane, between 0 and 1 for a conjugate pair that rings, the less
    the more it rings, and negative for a pole in the right half-plane, which grows. So
    ``damping("pz") >= 0`` asks for stability, and ``damping("pz") >= 0.5`` for little
    ringing.
    """

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        _poles(result, "a damping")
        return ResolvedMeasure(_core.Measure.damping(), "", _explain(_POLES))

    return _native(analysis, "least damping ratio of the poles", "", resolve)


def dominant_pole(analysis: str) -> Measure:
    """The natural frequency of a pole-zero analysis's pole nearest the origin.

    It is the pole's magnitude over 2π, in hertz: the slowest of the circuit's natural
    frequencies, which dominates how it settles.
    """

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        _poles(result, "a dominant pole")
        return ResolvedMeasure(
            _core.Measure.dominant_pole(), FREQUENCY, _explain(_POLES)
        )

    return _native(analysis, "dominant pole", None, resolve)


def _native(
    analysis: str, description: str, plain: Plain | None, resolve: _Resolve
) -> Measure:
    """The measure the engine reads once `resolve` resolves it against a result."""

    def read(result: results.AnalysisResult) -> Reading:
        return resolve(result).read(result)

    return Measure(analysis, description, plain, read, resolve)


@dataclass(frozen=True, slots=True)
class _Signal:
    """An unknown, or a voltage relative to the voltage of a reference."""

    name: str
    reference: str | None = None

    def resolve(self, unknowns: tuple[Unknown, ...]) -> tuple[_core.Signal, Dimension]:
        """The signal's unknown's index among `unknowns`, its reference's, and its unit.

        Raises:
            KeyError: When there is no unknown of either name.
            MeasureError: When the signal or its reference is not a voltage, yet
                there is a reference.
        """
        index = results.column(unknowns, self.name)
        dimension = _DIMENSIONS[unknowns[index].nature]
        if self.reference is None:
            return (index, None), dimension
        base = results.column(unknowns, self.reference)
        for position in (index, base):
            unknown = unknowns[position]
            if unknown.nature != "voltage":
                message = (
                    f"{self} names the {unknown.nature} {unknown.name}, not a voltage"
                )
                raise MeasureError(message)
        return (index, base), dimension

    @override
    def __str__(self) -> str:
        """The signal, such as ``out`` or ``out relative to ref``."""
        if self.reference is None:
            return self.name
        return f"{self.name} relative to {self.reference}"


@dataclass(frozen=True, slots=True)
class _Axis:
    """What a result's points are: times, levels or frequencies.

    The values of a parameter a DC sweep steps are of no dimension the result holds, so
    a point on their axis is in its own.
    """

    dimension: Dimension | None
    noun: str
    """What a point is, such as ``time``."""

    def convert(self, point: Quantity, name: str) -> float:
        """`point` in SI base units.

        Raises:
            DimensionError: When the point is not of the axis's dimension.
        """
        return point.to(
            point.dimension if self.dimension is None else self.dimension, name
        )

    def outside(self, points: Mapping[str, tuple[str, Quantity | None]]) -> _Explainer:
        """What a point past the axis says, of each of `points`.

        Each point is keyed as the engine names it, `at`, `start` or `stop`, and holds
        the name the message gives it and the point itself.
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
"""The axis of a small-signal response, a loop gain or a noise analysis."""


@dataclass(frozen=True, slots=True)
class _Resolved:
    """A window resolved against a result, and what its messages say.

    It holds its signal, its window in SI base units and the signal's dimension.
    """

    signal: _core.Signal
    window: _core.Window
    dimension: Dimension
    explainer: _Explainer

    def explain(self, *explainers: _Explainer) -> Callable[[_core.MeasureError], str]:
        """Says why the engine could not read the window, by `explainers` first."""
        return _explain(*explainers, self.explainer)


@dataclass(frozen=True, slots=True)
class _Window:
    """A signal from `start` to `stop` of a transient's times or a sweep's levels."""

    signal: _Signal
    start: Quantity | None
    stop: Quantity | None

    def __post_init__(self) -> None:
        _check(self.start, "the start")
        _check(self.stop, "the stop")

    def resolve(
        self, result: results.AnalysisResult, what: str, *, transient: bool
    ) -> _Resolved:
        """The window resolved against `result`, a transient, or a sweep too.

        The result must be a transient if `transient`, else a transient or a sweep.

        Raises:
            MeasureError: When the result is not.
            KeyError: When it has no unknown of the signal's names.
            DimensionError: When the start or the stop is not of its axis's dimension.
        """
        if transient and not isinstance(result, results.TransientResult):
            message = f"{what} reads a transient, not {_kind(result)}"
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

    @override
    def __str__(self) -> str:
        """The signal and the window, such as ``out from 1 µs to 2 µs``."""
        start = "" if self.start is None else f" from {self.start}"
        stop = "" if self.stop is None else f" to {self.stop}"
        return f"{self.signal}{start}{stop}"


@dataclass(frozen=True, slots=True)
class _Transfer:
    """What a small-signal measure reads.

    It reads `output` over `divisor`, or per unit excitation, in a response, and a loop
    gain without an output.
    """

    output: _Signal | None
    divisor: _Signal | None

    @property
    def unity(self) -> _Explainer:
        """What the engine's finding no fall through 0 dB says."""
        message = f"the gain of {self} never falls through 0 dB"
        return {"uncrossed": lambda _: message}

    def resolve(self, result: results.AnalysisResult) -> _core.Transfer:
        """The transfer resolved against `result`: none for a loop gain.

        Raises:
            MeasureError: When the result is neither a small-signal response read at
                an output nor a loop gain read at none.
            KeyError: When the response has no unknown of the output's or the divisor's
                names.
        """
        match result:
            case results.StabilityResult() if self.output is None:
                return None
            case results.AcResult() if self.output is not None:
                output, _ = self.output.resolve(result.unknowns)
                if self.divisor is None:
                    return output, None
                divisor, _ = self.divisor.resolve(result.unknowns)
                return output, divisor
            case results.StabilityResult():
                message = f"a loop gain is read without an output, not at {self.output}"
            case results.AcResult():
                message = "a small-signal response is read at an output"
            case _:
                message = (
                    "a gain or a phase reads a small-signal response or a loop gain, "
                    f"not {_kind(result)}"
                )
        raise MeasureError(message)

    def explain(
        self, *explainers: _Explainer, at: Quantity | None = None
    ) -> Callable[[_core.MeasureError], str]:
        """Says why the engine could not read the transfer, by `explainers` first.

        The point a gain or a phase is read `at` is named in its message.
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
        """The transfer, such as ``out``, ``out over in`` or ``the loop``."""
        if self.output is None:
            return "the loop"
        if self.divisor is None:
            return str(self.output)
        return f"{self.output} over {self.divisor.name}"


def _statistic(
    name: str,
    analysis: str,
    window: _Window,
    statistic: Callable[[_core.Signal, _core.Window], _core.Measure],
) -> Measure:
    """The measure of `statistic` over `window`, from its points and values."""

    def resolve(result: results.AnalysisResult) -> ResolvedMeasure:
        resolved = window.resolve(result, f"a {name}", transient=False)
        measure = statistic(resolved.signal, resolved.window)
        return ResolvedMeasure(measure, resolved.dimension, resolved.explain())

    return _native(analysis, f"{name} of {window}", None, resolve)


def _axis(result: results.AnalysisResult, what: str) -> _Axis:
    """The axis of a transient or a sweep.

    Raises:
        MeasureError: When the result is neither a transient nor a sweep.
    """
    match result:
        case results.TransientResult():
            return _Axis(TIME, "time")
        case results.DcSweepResult() if result.nature is None:
            return _Axis(None, f"value of {result.source}'s {result.parameter}")
        case results.DcSweepResult():
            along = VOLTAGE if result.nature == "voltage" else CURRENT
            return _Axis(along, f"level of {result.source}")
        case _:
            message = f"{what} reads a transient or a DC sweep, not {_kind(result)}"
            raise MeasureError(message)


def _signals(
    output: str | None, over: str | None, reference: str | None
) -> tuple[_Signal | None, _Signal | None]:
    """What a small-signal measure reads, and what over; neither for a loop gain.

    Raises:
        ValueError: When there is a divisor or a reference, but no output.
    """
    if output is None:
        if over is not None or reference is not None:
            message = (
                "a loop gain is read without an output, so without over or a reference"
            )
            raise ValueError(message)
        return None, None
    return _Signal(output, reference), None if over is None else _Signal(over)


def _spectrum(
    result: results.AnalysisResult, what: str, *, referred: bool
) -> results.NoiseResult:
    """`result`, a noise analysis's, with a source to refer its noise to if `referred`.

    Raises:
        MeasureError: When it is not, or has none.
    """
    if not isinstance(result, results.NoiseResult):
        message = f"{what} reads a noise analysis, not {_kind(result)}"
        raise MeasureError(message)
    if referred and result.input is None:
        message = "the noise analysis names no source to refer its noise to"
        raise MeasureError(message)
    return result


def _poles(result: results.AnalysisResult, what: str) -> None:
    """Checks that `result` is a pole-zero analysis's.

    Raises:
        MeasureError: When it is not.
    """
    if not isinstance(result, results.PoleZeroResult):
        message = f"{what} reads a pole-zero analysis, not {_kind(result)}"
        raise MeasureError(message)


_POLES: Final[_Explainer] = {"no_poles": lambda _: "the circuit has no poles"}
"""What the engine's finding no pole says."""


def _noise_unit(spectrum: results.NoiseResult, *, referred: bool) -> Dimension:
    """The unit of the noise's amplitude: the output's, or the input's if `referred`."""
    nature = spectrum.nature
    if referred and spectrum.input is not None:
        nature = spectrum.input.nature
    return VOLTAGE if nature == "voltage" else CURRENT


_MALFORMED: Final[_Explainer] = {
    "empty": lambda _: "the result holds no points",
    "mismatch": lambda _: "the result holds more or fewer values than points",
    "unknown": lambda _: "the result holds fewer unknowns than its circuit",
    "kind": lambda _: "the engine reads no result of this kind",
    "unreferred": lambda _: "the noise analysis names no source to refer its noise to",
}
"""What the engine says of a result whose arrays no analysis returns."""


def _explain(*explainers: _Explainer) -> Callable[[_core.MeasureError], str]:
    """Says why the engine could not read a result.

    The first of `explainers`, then `_MALFORMED`, that knows the reason says it.
    """

    def explain(error: _core.MeasureError) -> str:
        reason, numbers = error.args
        for explainer in (*explainers, _MALFORMED):
            if reason in explainer:
                return explainer[reason](numbers)
        return f"the engine cannot read the result: {reason}"

    return explain


_KINDS: Final[dict[type, str]] = {
    results.OperatingPointResult: "an operating point",
    results.DcSweepResult: "a DC sweep",
    results.AcResult: "a small-signal response",
    results.TransientResult: "a transient",
    results.PeriodicSteadyStateResult: "a periodic steady state",
    results.SensitivityResult: "sensitivities",
    results.StabilityResult: "a loop gain",
    results.NoiseResult: "a noise spectrum",
    results.PoleZeroResult: "poles and zeros",
    results.OperatingPointsResult: "DC solutions",
}
"""What each kind of result is, as messages name it."""


def _kind(result: results.AnalysisResult) -> str:
    return _KINDS[type(result)]


def _check(point: Quantity | None, name: str) -> None:
    """Checks that `point` is finite.

    Raises:
        ValueError: When it is infinite or not a number.
    """
    if point is not None:
        finite(point.value, name)
