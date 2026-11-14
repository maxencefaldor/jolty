# Copyright 2026 Maxence Faldor
"""jolty's measurements as numpy computed them, the reference the engine's must equal.

This is `jolty.specs._measurements` as it stood before the engine read measurements,
at commit cae321d, verbatim but for its noise and its poles: it reads them with the
numpy arithmetic `jolty.results` held then, copied at its end, rather than the engine's
`results` reads with now. `tests/test_native_measures.py` checks that every measurement
of `jolty.specs` reads what the same one of this module reads.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Final, Literal, override

import numpy as np
import numpy.typing as npt

from jolty import checks, results
from jolty.specs._requirements import Measurement, MeasurementError, Reading
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

type Edge = Literal["rising", "falling", "either"]
"""Which way a signal crosses a level."""

type _Solved = (
    results.OperatingPoint | results.Sweep | results.Response | results.Waveforms
)
"""A result that holds values of the circuit's unknowns."""

_DIMENSIONS: Final[dict[results.Quantity, Dimension]] = {
    "voltage": VOLTAGE,
    "current": CURRENT,
    "temperature": TEMPERATURE,
    "state": DIMENSIONLESS,
}
"""The dimension of each quantity an unknown is."""

_ON_AXIS: Final = 1e-9
"""How far past an end of an axis, relative to its span, a point still lies on it."""

_SAMPLES_PER_PERIOD: Final = 1024
"""The points a distortion samples each period of its fundamental at, interpolated."""

LEAST_PHASE_MARGIN: Final = 45.0
"""The phase margin, in degrees, below which the rule phase-margin states a loop's."""

LEAST_GAIN_MARGIN: Final = 6.0
"""The gain margin, in decibels, below which the rule gain-margin states a loop's."""


def diagnose(solved: Mapping[str, results.Result]) -> tuple[checks.Diagnostic, ...]:
    """The diagnostics of the rules on the analyses' results `solved`, by analysis.

    Each stability analysis's loop gain raises phase-margin when it falls through 0 dB
    with less than 45° of phase margin, and gain-margin when its phase falls through
    -180° with less than 6 dB of gain margin, each measured as `phase_margin` and
    `gain_margin` measure them.
    """
    found: list[checks.Diagnostic] = []
    for name, result in solved.items():
        if not isinstance(result, results.LoopGain):
            continue
        loop = f"the loop broken at `{result.probe}`"
        try:
            phase = phase_margin(name).read(result).value
            unity = crossover(name).read(result).value
        except MeasurementError:
            pass
        else:
            if phase < LEAST_PHASE_MARGIN:
                message = (
                    f"{loop} has a phase margin of {phase!r}° at its crossover, "
                    f"{unity!r} Hz"
                )
                found.append(_diagnostic(checks.PHASE_MARGIN, message))
        try:
            gain = gain_margin(name).read(result).value
        except MeasurementError:
            continue
        if gain < LEAST_GAIN_MARGIN:
            message = f"{loop} has a gain margin of {gain!r} dB"
            found.append(_diagnostic(checks.GAIN_MARGIN, message))
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
) -> Measurement:
    """The value of `signal` at an operating point, a time or a level.

    In a transient or a DC sweep, the value is interpolated `at` a time or a level of
    the swept source.
    """
    traced = _Signal(signal, reference)
    _check(at, "at")

    def read(result: results.Result) -> Reading:
        match result:
            case results.OperatingPoint() if at is None:
                values, dimension = traced.trace(result)
                return Reading(float(values), dimension)
            case results.OperatingPoint():
                message = f"an operating point holds one value, not one at {at}"
                raise MeasurementError(message)
            case (
                results.Response()
                | results.Sensitivities()
                | results.LoopGain()
                | results.NoiseSpectrum()
                | results.PolesZeros()
                | results.OperatingPoints()
            ):
                message = (
                    "a value reads an operating point, a transient or a DC sweep, "
                    f"not {_kind(result)}"
                )
                raise MeasurementError(message)
        axis, values, dimension = _traced(result, traced, "a value")
        if at is None:
            message = f"{_kind(result)} holds a value per {axis.noun}, and at is unset"
            raise MeasurementError(message)
        point = axis.locate(at, "at")
        return Reading(float(np.interp(point, axis.points, values)), dimension)

    described = str(traced) if at is None else f"{traced} at {at}"
    return Measurement(analysis, described, None, read)


def gain(
    analysis: str,
    output: str | None = None,
    *,
    at: Quantity,
    over: str | None = None,
    reference: str | None = None,
) -> Measurement:
    """The gain of `output` at a frequency of a small-signal response, in decibels.

    The gain is of `output` over `over`, or per unit excitation, interpolated over log
    frequency `at` the frequency. Without an output, it is the loop gain of a
    stability analysis.
    """
    traced, divisor = _signals(output, over, reference)
    _check(at, "at")

    def read(result: results.Result) -> Reading:
        frequencies, transfer = _transfer(result, traced, divisor)
        point = math.log(_frequency_axis(frequencies).locate(at, "at"))
        gains = _decibels(transfer)
        return Reading(float(np.interp(point, np.log(frequencies), gains)), "dB")

    return Measurement(analysis, f"gain of {_over(traced, over)} at {at}", "dB", read)


def phase(
    analysis: str,
    output: str | None = None,
    *,
    at: Quantity,
    over: str | None = None,
    reference: str | None = None,
) -> Measurement:
    """The phase of `output` at a frequency of a small-signal response, in degrees.

    The phase is relative to `over`, or to the excitation, unwrapped from the lowest
    frequency and interpolated over log frequency `at` the frequency. Without an output,
    it is the loop gain's, of a stability analysis.
    """
    traced, divisor = _signals(output, over, reference)
    _check(at, "at")

    def read(result: results.Result) -> Reading:
        frequencies, transfer = _transfer(result, traced, divisor)
        point = math.log(_frequency_axis(frequencies).locate(at, "at"))
        phases = np.degrees(np.unwrap(np.angle(transfer)))
        return Reading(float(np.interp(point, np.log(frequencies), phases)), "°")

    return Measurement(analysis, f"phase of {_over(traced, over)} at {at}", "°", read)


def bandwidth(
    analysis: str,
    output: str | None = None,
    *,
    drop: float = 3.0,
    over: str | None = None,
    reference: str | None = None,
) -> Measurement:
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
    traced, divisor = _signals(output, over, reference)

    def read(result: results.Result) -> Reading:
        frequencies, transfer = _transfer(result, traced, divisor)
        gains = _decibels(transfer)
        below = _crossings(np.log(frequencies), gains, gains[0] - drop, "falling")
        if not below.size:
            lowest = format_si(frequencies[0], "Hz")
            message = (
                f"the gain of {_over(traced, over)} stays within {drop:g} dB of its "
                f"gain at {lowest}"
            )
            raise MeasurementError(message)
        return Reading(math.exp(below[0]), FREQUENCY)

    described = f"{drop:g} dB bandwidth of {_over(traced, over)}"
    return Measurement(analysis, described, None, read)


def crossover(
    analysis: str,
    output: str | None = None,
    *,
    over: str | None = None,
    reference: str | None = None,
) -> Measurement:
    """The frequency where the gain of `output` first falls through 0 dB.

    The gain is of `output` over `over`, or per unit excitation, in a small-signal
    response; without an output, it is the loop gain of a stability analysis, and this
    its crossover frequency. It is interpolated over log frequency.
    """
    traced, divisor = _signals(output, over, reference)

    def read(result: results.Result) -> Reading:
        frequencies, transfer = _transfer(result, traced, divisor)
        logarithms = np.log(frequencies)
        return Reading(math.exp(_unity(logarithms, transfer, traced, over)), FREQUENCY)

    described = f"crossover frequency of {_over(traced, over)}"
    return Measurement(analysis, described, None, read)


def phase_margin(
    analysis: str,
    output: str | None = None,
    *,
    over: str | None = None,
    reference: str | None = None,
) -> Measurement:
    """180° plus the phase of `output` where its gain first falls through 0 dB.

    The gain and the phase are of `output` relative to `over`, or to the excitation, in
    a small-signal response. Without an output, they are the loop gain's, of a
    stability analysis, and this is the loop's phase margin.
    """
    traced, divisor = _signals(output, over, reference)

    def read(result: results.Result) -> Reading:
        frequencies, transfer = _transfer(result, traced, divisor)
        logarithms = np.log(frequencies)
        unity = _unity(logarithms, transfer, traced, over)
        phases = np.degrees(np.unwrap(np.angle(transfer)))
        return Reading(180.0 + float(np.interp(unity, logarithms, phases)), "°")

    return Measurement(analysis, f"phase margin of {_over(traced, over)}", "°", read)


def gain_margin(
    analysis: str,
    output: str | None = None,
    *,
    over: str | None = None,
    reference: str | None = None,
) -> Measurement:
    """Minus the gain of `output` where its phase first falls through -180°, in dB.

    The gain and the phase, unwrapped from the lowest frequency, are of `output`
    relative to `over`, or to the excitation, in a small-signal response. Without an
    output, they are the loop gain's, of a stability analysis, and this is the loop's
    gain margin.
    """
    traced, divisor = _signals(output, over, reference)

    def read(result: results.Result) -> Reading:
        frequencies, transfer = _transfer(result, traced, divisor)
        logarithms = np.log(frequencies)
        phases = np.degrees(np.unwrap(np.angle(transfer)))
        inverted = _crossings(logarithms, phases, -180.0, "falling")
        if not inverted.size:
            message = f"the phase of {_over(traced, over)} never falls through -180°"
            raise MeasurementError(message)
        gains = _decibels(transfer)
        return Reading(-float(np.interp(inverted[0], logarithms, gains)), "dB")

    return Measurement(analysis, f"gain margin of {_over(traced, over)}", "dB", read)


def maximum(
    analysis: str,
    signal: str,
    *,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measurement:
    """The highest value of `signal` in a transient or a DC sweep.

    The window runs from `start` to `stop`, times or levels, by default over all.
    """
    return _statistic(
        "maximum", analysis, _Window(_Signal(signal, reference), start, stop), _highest
    )


def minimum(
    analysis: str,
    signal: str,
    *,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measurement:
    """The lowest value of `signal` in a transient or a DC sweep.

    The window runs from `start` to `stop`, times or levels, by default over all.
    """
    return _statistic(
        "minimum", analysis, _Window(_Signal(signal, reference), start, stop), _lowest
    )


def peak_to_peak(
    analysis: str,
    signal: str,
    *,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measurement:
    """The highest less the lowest value of `signal` in a transient or a DC sweep.

    The window runs from `start` to `stop`, times or levels, by default over all.
    """
    window = _Window(_Signal(signal, reference), start, stop)
    return _statistic("peak-to-peak", analysis, window, _span)


def average(
    analysis: str,
    signal: str,
    *,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measurement:
    """The mean of `signal` in a transient or a DC sweep, by the trapezoidal rule.

    The window runs from `start` to `stop`, times or levels, by default over all.
    """
    window = _Window(_Signal(signal, reference), start, stop)
    return _statistic("average", analysis, window, _mean)


def rms(
    analysis: str,
    signal: str,
    *,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measurement:
    """The root mean square of `signal` in a transient or a DC sweep.

    The window runs from `start` to `stop`, times or levels, by default over all, and
    the square is integrated by the trapezoidal rule.
    """
    window = _Window(_Signal(signal, reference), start, stop)
    return _statistic("rms", analysis, window, _root_mean_square)


def rise_time(  # noqa: PLR0913 - a window, its signal and the two levels
    analysis: str,
    signal: str,
    *,
    low: float = 0.1,
    high: float = 0.9,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measurement:
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

    def read(result: results.Result) -> Reading:
        times, values, step = _step(result, window, "a rise time")
        edge: Edge = "rising" if step > 0 else "falling"
        lower = _crossings(times, values, values[0] + low * step, edge)
        upper = _crossings(times, values, values[0] + high * step, edge)
        if not lower.size or not upper[upper >= lower[0]].size:
            message = f"{window} steps too little to resolve its levels"
            raise MeasurementError(message)
        return Reading(float(upper[upper >= lower[0]][0] - lower[0]), TIME)

    return Measurement(analysis, f"rise time of {window}", None, read)


def overshoot(
    analysis: str,
    signal: str,
    *,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measurement:
    """How far a transient's `signal` passes its final value, in percent of its step.

    The step runs from the signal's value at the window's start to its final value, at
    the window's end.
    """
    window = _Window(_Signal(signal, reference), start, stop)

    def read(result: results.Result) -> Reading:
        _, values, step = _step(result, window, "an overshoot")
        peak = values.max() if step > 0 else values.min()
        return Reading(100.0 * max(0.0, float(peak - values[-1]) / step), "%")

    return Measurement(analysis, f"overshoot of {window}", "%", read)


def settling_time(  # noqa: PLR0913 - a window, its signal and the band
    analysis: str,
    signal: str,
    *,
    band: float = 0.02,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measurement:
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

    def read(result: results.Result) -> Reading:
        times, values, step = _step(result, window, "a settling time")
        excess = np.abs(values - values[-1]) - band * abs(step)
        outside = np.flatnonzero(excess > 0)
        if not outside.size:
            return Reading(0.0, TIME)
        last = int(outside[-1])
        fraction = excess[last] / (excess[last] - excess[last + 1])
        settled = times[last] + fraction * (times[last + 1] - times[last])
        return Reading(float(settled - times[0]), TIME)

    return Measurement(
        analysis, f"{100 * band:g}% settling time of {window}", None, read
    )


def crossing(  # noqa: PLR0913 - a window, its signal, the level and the edge
    analysis: str,
    signal: str,
    level: Quantity,
    *,
    edge: Edge = "either",
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measurement:
    """The first time a transient's `signal` crosses `level` the way `edge` says."""
    _check(level, "the level")
    window = _Window(_Signal(signal, reference), start, stop)

    def read(result: results.Result) -> Reading:
        times, values, dimension = window.transient(result, "a crossing")
        found = _crossings(times, values, level.to(dimension, "the level"), edge)
        if not found.size:
            message = f"{window} does not cross {level}"
            raise MeasurementError(message)
        return Reading(float(found[0]), TIME)

    way = "" if edge == "either" else f" {edge}"
    return Measurement(
        analysis, f"first{way} crossing of {level} by {window}", None, read
    )


def frequency(  # noqa: PLR0913 - a window, its signal and the level
    analysis: str,
    signal: str,
    *,
    level: Quantity | None = None,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measurement:
    """How often a transient's `signal` rises through `level`.

    The level is by default the midpoint of the signal's lowest and highest values, and
    the frequency is the rises less one over the time from the first rise to the last.
    """
    _check(level, "the level")
    window = _Window(_Signal(signal, reference), start, stop)

    def read(result: results.Result) -> Reading:
        times, values, dimension = window.transient(result, "a frequency")
        middle = (
            (values.max() + values.min()) / 2
            if level is None
            else level.to(dimension, "the level")
        )
        rises = _crossings(times, values, middle, "rising")
        if rises.size < 2:
            through = format_si(float(middle), str(dimension))
            message = f"{window} rises through {through} fewer than twice"
            raise MeasurementError(message)
        return Reading(float((rises.size - 1) / (rises[-1] - rises[0])), FREQUENCY)

    return Measurement(analysis, f"frequency of {window}", None, read)


def distortion(  # noqa: PLR0913 - a window, its signal, the fundamental, the harmonics
    analysis: str,
    signal: str,
    *,
    fundamental: Quantity,
    harmonics: int = 9,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    reference: str | None = None,
) -> Measurement:
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
    frequency = fundamental.to(FREQUENCY, "the fundamental")
    if not 0 < frequency < math.inf:
        message = (
            f"a distortion's fundamental is positive and finite, not {fundamental}"
        )
        raise ValueError(message)
    if not 2 <= harmonics < _SAMPLES_PER_PERIOD // 2:
        message = (
            f"a distortion counts harmonics up to between 2 and "
            f"{_SAMPLES_PER_PERIOD // 2 - 1}, not {harmonics}"
        )
        raise ValueError(message)
    window = _Window(_Signal(signal, reference), start, stop)

    def read(result: results.Result) -> Reading:
        times, values, _ = window.transient(result, "a distortion")
        span = (times[-1] - times[0]) * frequency
        periods = 1 if start is None else math.floor(span * (1 + _ON_AXIS))
        if span * (1 + _ON_AXIS) < periods or periods < 1:
            message = f"{window} spans less than a period of {fundamental}"
            raise MeasurementError(message)
        samples = periods * _SAMPLES_PER_PERIOD
        at = times[-1] - periods / frequency * (1 - np.arange(samples) / samples)
        spectrum = np.abs(np.fft.rfft(np.interp(at, times, values)))
        amplitudes = spectrum[periods : periods * harmonics + 1 : periods]
        if amplitudes[0] == 0:
            message = f"{window} has no component at {fundamental}"
            raise MeasurementError(message)
        total = math.sqrt(float(np.sum(amplitudes[1:] ** 2)))
        ratio = max(total / float(amplitudes[0]), 1e-300)
        return Reading(20 * math.log10(ratio), "dB")

    described = f"distortion of {window} at {fundamental}"
    return Measurement(analysis, described, "dB", read)


def noise_density(
    analysis: str, *, at: Quantity, referred: bool = False
) -> Measurement:
    """The noise density of a noise analysis's output at a frequency.

    It is the square root of the power density, in V/√Hz for a voltage and A/√Hz for a
    current, as datasheets give it, interpolated as a power law between frequencies
    `at` the frequency. Referred, it is the noise referred to the analysis's source,
    as that input's own noise, in the unit of the source's level.
    """
    _check(at, "at")

    def read(result: results.Result) -> Reading:
        spectrum = _spectrum(result, "a noise density", referred=referred)
        point = _frequency_axis(spectrum.frequencies).locate(at, "at")
        density = _density(spectrum, point, referred=referred)
        return Reading(
            math.sqrt(density),
            _noise_unit(spectrum, referred=referred) / rtHz.dimension,
        )

    referral = " referred to the input" if referred else ""
    return Measurement(analysis, f"noise density{referral} at {at}", None, read)


def integrated_noise(
    analysis: str,
    *,
    start: Quantity | None = None,
    stop: Quantity | None = None,
    referred: bool = False,
) -> Measurement:
    """The RMS noise of a noise analysis's output from `start` to `stop`.

    It is the square root of the power density integrated over the band, by default
    every frequency of the analysis, each density a power law between frequencies as
    SPICE integrates it, in volts or amperes. Referred, it is the noise referred to the
    analysis's source, as that input's own noise, in the unit of the source's level.
    """
    _check(start, "start")
    _check(stop, "stop")

    def read(result: results.Result) -> Reading:
        spectrum = _spectrum(result, "an integrated noise", referred=referred)
        axis = _frequency_axis(np.sort(spectrum.frequencies))
        low = None if start is None else axis.locate(start, "start")
        high = None if stop is None else axis.locate(stop, "stop")
        if low is not None and high is not None and low >= high:
            message = f"the band from {start} to {stop} is empty"
            raise MeasurementError(message)
        rms = _rms(spectrum, low, high, referred=referred)
        return Reading(rms, _noise_unit(spectrum, referred=referred))

    referral = " referred to the input" if referred else ""
    lowest = "the lowest frequency" if start is None else str(start)
    highest = "the highest" if stop is None else str(stop)
    described = f"RMS noise{referral} from {lowest} to {highest}"
    return Measurement(analysis, described, None, read)


def solutions(analysis: str) -> Measurement:
    """How many DC solutions a search for operating points found, a plain number.

    ``solutions("states") <= 1`` asks for a unique operating point, and a latch's
    ``solutions("states") >= 3`` for its two states and the balance between them.
    """

    def read(result: results.Result) -> Reading:
        if not isinstance(result, results.OperatingPoints):
            message = (
                f"solutions reads a search for operating points, not {_kind(result)}"
            )
            raise MeasurementError(message)
        return Reading(float(len(result.points)), "")

    return Measurement(analysis, "DC solutions", "", read)


def damping(analysis: str) -> Measurement:
    """The least damping ratio of a pole-zero analysis's poles, a plain number.

    A pole's damping ratio is minus its real part over its magnitude: 1 for a real pole
    in the left half-plane, between 0 and 1 for a conjugate pair that rings, the less
    the more it rings, and negative for a pole in the right half-plane, which grows. So
    ``damping("pz") >= 0`` asks for stability, and ``damping("pz") >= 0.5`` for little
    ringing.
    """

    def read(result: results.Result) -> Reading:
        poles = _poles(result, "a damping")
        return Reading(float(np.min(_dampings(poles))), "")

    return Measurement(analysis, "least damping ratio of the poles", "", read)


def dominant_pole(analysis: str) -> Measurement:
    """The natural frequency of a pole-zero analysis's pole nearest the origin.

    It is the pole's magnitude over 2π, in hertz: the slowest of the circuit's natural
    frequencies, which dominates how it settles.
    """

    def read(result: results.Result) -> Reading:
        poles = _poles(result, "a dominant pole")
        return Reading(float(_natural_frequencies(poles).min()), FREQUENCY)

    return Measurement(analysis, "dominant pole", None, read)


@dataclass(frozen=True, slots=True)
class _Signal:
    """An unknown, or a voltage relative to the voltage of a reference."""

    name: str
    reference: str | None = None

    def trace(self, result: _Solved) -> tuple[npt.NDArray[Any], Dimension]:
        """The signal's values in `result`, and their dimension.

        Raises:
            KeyError: When the result has no unknown of that name.
            MeasurementError: When the signal or its reference is not a voltage, yet
                there is a reference.
        """
        index = results.column(result.unknowns, self.name)
        quantity = result.unknowns[index].quantity
        values = result.values[..., index]
        dimension = _DIMENSIONS[quantity]
        if self.reference is None:
            return values, dimension
        base = results.column(result.unknowns, self.reference)
        for position in (index, base):
            unknown = result.unknowns[position]
            if unknown.quantity != "voltage":
                message = (
                    f"{self} names the {unknown.quantity} {unknown.name}, not a voltage"
                )
                raise MeasurementError(message)
        return values - result.values[..., base], dimension

    @override
    def __str__(self) -> str:
        """The signal, such as ``out`` or ``out relative to ref``."""
        if self.reference is None:
            return self.name
        return f"{self.name} relative to {self.reference}"


@dataclass(frozen=True, slots=True)
class _Axis:
    """The increasing points a result holds values at: times, levels or frequencies."""

    points: results.Real
    dimension: Dimension
    noun: str
    """What a point is, such as ``time``."""

    def locate(self, point: Quantity, name: str) -> float:
        """`point` in SI base units, after checking it lies on the axis.

        Raises:
            DimensionError: When the point is not of the axis's dimension.
            MeasurementError: When it lies past either end.
        """
        located = point.to(self.dimension, name)
        first, last = float(self.points[0]), float(self.points[-1])
        within = _ON_AXIS * (last - first)
        if not first - within <= located <= last + within:
            symbol = str(self.dimension)
            message = (
                f"{name}, {point}, lies outside the {self.noun}s, "
                f"{format_si(first, symbol)} to {format_si(last, symbol)}"
            )
            raise MeasurementError(message)
        return min(max(located, first), last)


@dataclass(frozen=True, slots=True)
class _Window:
    """A signal from `start` to `stop` of a transient's times or a sweep's levels."""

    signal: _Signal
    start: Quantity | None
    stop: Quantity | None

    def __post_init__(self) -> None:
        _check(self.start, "the start")
        _check(self.stop, "the stop")

    def values(
        self, result: results.Result, what: str
    ) -> tuple[results.Real, results.Real, Dimension]:
        """The points of the window, the signal's values at each, and their dimension.

        The window's ends are interpolated.

        Raises:
            MeasurementError: When the result is neither a transient nor a sweep, or
                the window is empty or past the axis.
        """
        axis, values, dimension = _traced(result, self.signal, what)
        first, last = float(axis.points[0]), float(axis.points[-1])
        if self.start is not None:
            first = axis.locate(self.start, "the start")
        if self.stop is not None:
            last = axis.locate(self.stop, "the stop")
        if not first < last:
            message = f"the window of {self} is empty"
            raise MeasurementError(message)
        inside = axis.points[(axis.points > first) & (axis.points < last)]
        points = np.concatenate(([first], inside, [last]))
        return points, np.interp(points, axis.points, values), dimension

    def transient(
        self, result: results.Result, what: str
    ) -> tuple[results.Real, results.Real, Dimension]:
        """The times of the window, the signal's values at each, and their dimension.

        Raises:
            MeasurementError: When the result is not a transient, or the window is
                empty or past its times.
        """
        if not isinstance(result, results.Waveforms):
            message = f"{what} reads a transient, not {_kind(result)}"
            raise MeasurementError(message)
        return self.values(result, what)

    @override
    def __str__(self) -> str:
        """The signal and the window, such as ``out from 1 µs to 2 µs``."""
        start = "" if self.start is None else f" from {self.start}"
        stop = "" if self.stop is None else f" to {self.stop}"
        return f"{self.signal}{start}{stop}"


def _statistic(
    name: str,
    analysis: str,
    window: _Window,
    statistic: Callable[[results.Real, results.Real], float],
) -> Measurement:
    """The measurement of `statistic` over `window`, from its points and values."""

    def read(result: results.Result) -> Reading:
        points, values, dimension = window.values(result, f"a {name}")
        return Reading(statistic(points, values), dimension)

    return Measurement(analysis, f"{name} of {window}", None, read)


def _highest(_: results.Real, values: results.Real) -> float:
    return float(values.max())


def _lowest(_: results.Real, values: results.Real) -> float:
    return float(values.min())


def _span(_: results.Real, values: results.Real) -> float:
    return float(values.max() - values.min())


def _mean(points: results.Real, values: results.Real) -> float:
    return float(np.trapezoid(values, points) / (points[-1] - points[0]))


def _root_mean_square(points: results.Real, values: results.Real) -> float:
    return math.sqrt(_mean(points, values**2))


def _step(
    result: results.Result, window: _Window, what: str
) -> tuple[results.Real, results.Real, float]:
    """The times and values of a transient's window, and the signal's step across it.

    Raises:
        MeasurementError: When the signal ends the window where it starts it.
    """
    times, values, dimension = window.transient(result, what)
    step = float(values[-1] - values[0])
    if step == 0:
        level = format_si(float(values[0]), str(dimension))
        message = f"{window} ends where it starts, at {level}"
        raise MeasurementError(message)
    return times, values, step


def _traced(
    result: results.Result, signal: _Signal, what: str
) -> tuple[_Axis, results.Real, Dimension]:
    """The increasing axis of a transient or a sweep, and the signal's values along it.

    Raises:
        MeasurementError: When the result is neither a transient nor a sweep.
    """
    match result:
        case results.Waveforms():
            axis = _Axis(result.times, TIME, "time")
        case results.Sweep():
            along = VOLTAGE if result.quantity == "voltage" else CURRENT
            axis = _Axis(result.levels, along, f"level of {result.source}")
        case _:
            message = f"{what} reads a transient or a DC sweep, not {_kind(result)}"
            raise MeasurementError(message)
    values, dimension = signal.trace(result)
    if axis.points[0] > axis.points[-1]:
        return replace(axis, points=axis.points[::-1]), values[::-1], dimension
    return axis, values, dimension


def _transfer(
    result: results.Result, output: _Signal | None, divisor: _Signal | None
) -> tuple[results.Real, results.Complex]:
    """The frequencies, increasing, and the output over the divisor, or the loop gain.

    Raises:
        MeasurementError: When the result is neither a small-signal response read at an
            output nor a loop gain read at none, a frequency is not positive, or the
            divisor vanishes.
    """
    match result:
        case results.LoopGain() if output is None:
            transfer = result.gains
        case results.Response() if output is not None:
            phasors, _ = output.trace(result)
            if divisor is None:
                transfer = phasors
            else:
                divisors, _ = divisor.trace(result)
                if np.any(divisors == 0):
                    message = f"{divisor} vanishes in the response"
                    raise MeasurementError(message)
                transfer = phasors / divisors
        case results.LoopGain():
            message = f"a loop gain is read without an output, not at {output}"
            raise MeasurementError(message)
        case results.Response():
            message = "a small-signal response is read at an output"
            raise MeasurementError(message)
        case _:
            message = (
                "a gain or a phase reads a small-signal response or a loop gain, not "
                f"{_kind(result)}"
            )
            raise MeasurementError(message)
    order = np.argsort(result.frequencies)
    frequencies = result.frequencies[order]
    if frequencies[0] <= 0:
        lowest = format_si(frequencies[0], "Hz")
        message = f"a response is read over log frequency, so not at {lowest}"
        raise MeasurementError(message)
    return frequencies, transfer[order]


def _unity(
    logarithms: results.Real,
    transfer: results.Complex,
    output: _Signal | None,
    over: str | None,
) -> float:
    """The log frequency where the gain of `transfer` first falls through 0 dB.

    Raises:
        MeasurementError: When it never does.
    """
    unity = _crossings(logarithms, _decibels(transfer), 0.0, "falling")
    if not unity.size:
        message = f"the gain of {_over(output, over)} never falls through 0 dB"
        raise MeasurementError(message)
    return float(unity[0])


def _frequency_axis(frequencies: results.Real) -> _Axis:
    return _Axis(frequencies, FREQUENCY, "frequency")


def _decibels(phasors: results.Complex) -> results.Real:
    """The magnitudes of `phasors` in decibels; minus infinity where they vanish."""
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.abs(phasors))


def _crossings(
    points: results.Real, values: results.Real, level: float, edge: Edge
) -> results.Real:
    """Where `values` cross `level` the way `edge` says, interpolated between points.

    A value that reaches the level crosses it only once it passes it.
    """
    below, above = values[:-1] - level, values[1:] - level
    rising = (below < 0) & (above >= 0)
    falling = (below > 0) & (above <= 0)
    match edge:
        case "rising":
            crossed = rising
        case "falling":
            crossed = falling
        case "either":
            crossed = rising | falling
    index = np.flatnonzero(crossed)
    fraction = below[index] / (below[index] - above[index])
    return points[index] + fraction * (points[index + 1] - points[index])


def _signals(
    output: str | None, over: str | None, reference: str | None
) -> tuple[_Signal | None, _Signal | None]:
    """What a small-signal measurement reads, and what over; neither for a loop gain.

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
    result: results.Result, what: str, *, referred: bool
) -> results.NoiseSpectrum:
    """`result`, a noise analysis's, with a source to refer its noise to if `referred`.

    Raises:
        MeasurementError: When it is not, or has none.
    """
    if not isinstance(result, results.NoiseSpectrum):
        message = f"{what} reads a noise analysis, not {_kind(result)}"
        raise MeasurementError(message)
    if referred and result.input is None:
        message = "the noise analysis names no source to refer its noise to"
        raise MeasurementError(message)
    return result


def _poles(result: results.Result, what: str) -> results.Complex:
    """The poles of `result`, a pole-zero analysis's.

    Raises:
        MeasurementError: When it is not, or the circuit has no poles.
    """
    if not isinstance(result, results.PolesZeros):
        message = f"{what} reads a pole-zero analysis, not {_kind(result)}"
        raise MeasurementError(message)
    if not result.poles.size:
        message = "the circuit has no poles"
        raise MeasurementError(message)
    return result.poles


def _noise_unit(spectrum: results.NoiseSpectrum, *, referred: bool) -> Dimension:
    """The unit of the noise's amplitude: the output's, or the input's if `referred`."""
    quantity = spectrum.quantity
    if referred and spectrum.input is not None:
        quantity = spectrum.input.quantity
    return VOLTAGE if quantity == "voltage" else CURRENT


def _over(output: _Signal | None, over: str | None) -> str:
    if output is None:
        return "the loop"
    return str(output) if over is None else f"{output} over {over}"


_KINDS: Final[dict[type, str]] = {
    results.OperatingPoint: "an operating point",
    results.Sweep: "a DC sweep",
    results.Response: "a small-signal response",
    results.Waveforms: "a transient",
    results.Sensitivities: "sensitivities",
    results.LoopGain: "a loop gain",
    results.NoiseSpectrum: "a noise spectrum",
    results.PolesZeros: "poles and zeros",
    results.OperatingPoints: "DC solutions",
}
"""What each kind of result is, as messages name it."""


def _kind(result: results.Result) -> str:
    return _KINDS[type(result)]


def _check(point: Quantity | None, name: str) -> None:
    """Checks that `point` is finite.

    Raises:
        ValueError: When it is infinite or not a number.
    """
    if point is not None:
        finite(point.value, name)


def _referred(spectrum: results.NoiseSpectrum) -> results.Real:
    """The noise referred to the input at each frequency, in V²/Hz or A²/Hz.

    It is the output's density over the gain's square, in the unit of the source's
    level squared per hertz.

    Raises:
        ValueError: When the analysis named no source to refer its noise to.
    """
    if spectrum.input is None:
        message = "the noise analysis named no source to refer its noise to"
        raise ValueError(message)
    with np.errstate(divide="ignore"):
        return spectrum.densities / np.abs(spectrum.input.gains) ** 2


def _density(
    spectrum: results.NoiseSpectrum, at: float, *, referred: bool = False
) -> float:
    """The density at the frequency `at`, in hertz.

    It is a power law between the frequencies around it.

    Raises:
        ValueError: When the frequency does not lie within the frequencies, or the
            analysis named no source to refer its noise to.
    """
    densities = _referred(spectrum) if referred else spectrum.densities
    order = np.argsort(spectrum.frequencies)
    points = spectrum.frequencies[order]
    if not points[0] <= at <= points[-1]:
        message = (
            f"{at!r} Hz does not lie within the frequencies, "
            f"{float(points[0])!r} Hz to {float(points[-1])!r} Hz"
        )
        raise ValueError(message)
    return _power_law(points, densities[order], at)


def _rms(
    spectrum: results.NoiseSpectrum,
    start: float | None = None,
    stop: float | None = None,
    *,
    referred: bool = False,
) -> float:
    """The RMS noise from `start` to `stop`, in hertz, by default every frequency.

    It is the square root of the density integrated over the band, in volts or
    amperes, the density a power law between frequencies, as SPICE integrates it.

    Raises:
        ValueError: When the band does not lie within the frequencies, or is
            empty, or the analysis named no source to refer its noise to.
    """
    densities = _referred(spectrum) if referred else spectrum.densities
    return math.sqrt(_integrate(spectrum.frequencies, densities, start, stop))


def _integrate(
    frequencies: results.Real,
    densities: results.Real,
    start: float | None = None,
    stop: float | None = None,
) -> float:
    """The integral of `densities` from `start` to `stop`, by default every frequency.

    The density is a power law between `frequencies`, as SPICE integrates noise.

    Raises:
        ValueError: When the band does not lie within the frequencies, or is empty.
    """
    order = np.argsort(frequencies)
    points, values = frequencies[order], densities[order]
    low = float(points[0]) if start is None else start
    high = float(points[-1]) if stop is None else stop
    if not points[0] <= low < high <= points[-1]:
        message = (
            f"a band from {low!r} Hz to {high!r} Hz does not lie within the "
            f"frequencies, {float(points[0])!r} Hz to {float(points[-1])!r} Hz"
        )
        raise ValueError(message)
    inside = (points > low) & (points < high)
    band = np.concatenate(([low], points[inside], [high]))
    levels = np.concatenate(
        (
            [_power_law(points, values, low)],
            values[inside],
            [_power_law(points, values, high)],
        )
    )
    return math.fsum(
        _segment(
            *map(float, band[index : index + 2]), *map(float, levels[index : index + 2])
        )
        for index in range(band.size - 1)
    )


def _power_law(points: results.Real, values: results.Real, at: float) -> float:
    """The density at `at`, a power law between the two frequencies around it."""
    index = int(np.clip(np.searchsorted(points, at), 1, points.size - 1))
    first, last = float(points[index - 1]), float(points[index])
    low, high = float(values[index - 1]), float(values[index])
    if at in {first, last}:
        return low if at == first else high
    if low <= 0 or high <= 0:
        return low + (high - low) * (at - first) / (last - first)
    return low * math.exp(
        math.log(high / low) * math.log(at / first) / math.log(last / first)
    )


def _segment(first: float, last: float, low: float, high: float) -> float:
    """The integral from `first` to `last` of the power law from `low` to `high`."""
    if low <= 0 or high <= 0:
        return 0.5 * (low + high) * (last - first)
    span = math.log(last / first)
    # With the density ∝ f^a, the integral is low·first·(e^{(a+1)·span} - 1)/(a + 1).
    growth = math.log(high / low) + span
    factor = 1.0 if growth == 0 else math.expm1(growth) / growth
    return low * first * span * factor


def _natural_frequencies(roots: results.Complex) -> results.Real:
    """The natural frequency of each of `roots`, its magnitude over 2π, in hertz."""
    return np.abs(roots) / (2 * np.pi)


def _dampings(roots: results.Complex) -> results.Real:
    """The damping ratio of each of `roots`, minus its real part over its magnitude.

    It is 1 for a real root in the left half-plane, between 0 and 1 for a conjugate pair
    that rings, and negative in the right half-plane; NaN at the origin.
    """
    with np.errstate(invalid="ignore"):
        return -roots.real / np.abs(roots)
