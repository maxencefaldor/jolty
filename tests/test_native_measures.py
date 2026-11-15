# Copyright 2026 Maxence Faldor
"""The engine's measures read what numpy read, on specs' results and random signals.

Every measure of `jolty.specs` resolves itself against a result, and the engine
reads it. `numpy_measures` holds the same measures as numpy computed them
before the engine read them. Where this platform's numpy rounds as the engine does,
every measure but a distortion reads the same float, bit for bit, in the same
unit, or raises the same error with the same message: numpy then fuses the product and
the sum of an interpolation, of Smith's complex division and of a complex magnitude
into one rounding each, as the engine does and as arm64's numpy does, and takes its
logarithms and angles from the platform's math library, as Python does. A distortion's
amplitudes come from numpy's fast Fourier transform in one and from the discrete
transform's sums in the other, so its readings agree within the two's rounding bounds,
derived where the test checks them. On a platform whose numpy rounds otherwise, as an
x86 processor's rounds an interpolation twice, the tests skip.
"""

import importlib.util
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import numpy_measures as reference
from jolty import Simulation, results, specs
from jolty.errors import DimensionError, JoltyError
from jolty.plan import DcSweep, Noise, OperatingPoints, PoleZero
from jolty.specs import Measure, MeasureError
from jolty.units import A, Hz, Quantity, V, kHz, s, us
from numpy_measures import _Signal, _Window

EXAMPLES: Final = Path(__file__).parent.parent / "examples"


def _rounds_as_the_engine() -> bool:
    """Whether numpy rounds here as the engine does.

    It does when it fuses an interpolation's product and sum, a complex division's and a
    magnitude's, and takes its logarithms and angles from the platform's math library.
    Each case is one whose fused and unfused results differ.
    """
    interpolated = float(np.interp(0.3, [0.0, 0.7], [0.1, 0.9]))
    a, b = complex(-0.084, -1.163), complex(-0.629, -0.488)
    ratio = b.imag / b.real
    scale = 1.0 / math.fma(b.imag, ratio, b.real)
    quotient = complex(
        math.fma(a.imag, ratio, a.real) * scale,
        math.fma(-a.real, ratio, a.imag) * scale,
    )
    z = complex(-1.091, -1.355)
    small = abs(z.real) / abs(z.imag)
    magnitude = math.sqrt(math.fma(small, small, 1.0)) * abs(z.imag)
    points = np.linspace(0.1, 1e6, 1001)
    return all(
        (
            interpolated == math.fma(0.8 / 0.7, 0.3, 0.1),
            complex(np.complex128(a) / np.complex128(b)) == quotient,
            float(np.abs(np.array([z]))[0]) == magnitude,
            np.array_equal(np.log(points), [math.log(point) for point in points]),
            np.array_equal(np.log10(points), [math.log10(point) for point in points]),
            np.array_equal(
                np.arctan2(points, -points),
                [math.atan2(point, -point) for point in points],
            ),
        )
    )


pytestmark = pytest.mark.skipif(
    not _rounds_as_the_engine(),
    reason="numpy rounds otherwise here than the engine, as on x86, whose numpy rounds "
    "an interpolation's product and sum twice",
)

SETTINGS: Final = settings(max_examples=300, derandomize=True, deadline=None)
"""Each property on the same 300 cases every run."""

UNKNOWNS: Final = (
    results.Unknown("out", "voltage"),
    results.Unknown("ref", "voltage"),
    results.Unknown("V1:current", "current"),
)

MATH_DOMAIN: Final = "expected a "
"""How Python's `math` starts its message when an argument leaves its domain, as
``expected a nonnegative input, got -1.0``."""

MATH_RANGE: Final = {("OverflowError", "math range error")}
"""What Python's `math` raises when a result of finite arguments overflows."""

type Outcome = tuple[str, str]
"""What a measure read: its value's bits and its unit, or its error and message."""


def outcome(measure: Measure, result: results.AnalysisResult) -> Outcome:
    """What `measure` reads of `result`.

    numpy's floating-point warnings, of an overflow or of minus infinity less itself,
    which the tests would make errors, are off, as they change no number numpy reads.
    """
    try:
        with np.errstate(all="ignore"):
            reading = measure.read(result)
    except (JoltyError, KeyError, ValueError, OverflowError) as error:
        return type(error).__name__, str(error.args[0])
    return float(reading.value).hex(), str(reading.unit)


def alike(
    name: str,
    result: results.AnalysisResult,
    *args: Any,  # noqa: ANN401 - the measure's own arguments
    **kwargs: Any,  # noqa: ANN401 - the measure's own arguments
) -> Outcome:
    """What the measure `name` reads of `result`, the engine's and numpy's alike."""
    native = getattr(specs, name)(*args, **kwargs)
    numpy = getattr(reference, name)(*args, **kwargs)
    assert native.native is not None
    read, expected = outcome(native, result), outcome(numpy, result)
    if expected in MATH_RANGE or (
        expected[0] == "ValueError" and expected[1].startswith(MATH_DOMAIN)
    ):
        # Python's `math` raised where its argument left its domain or its result its
        # range, as the square root of a negative mean over a sweep that turns back, or
        # the power law of an infinite density; the engine reads not a number or an
        # infinity there, which leaves the requirement unmeasured.
        assert read[0] in {"nan", "inf", "-inf"}, (str(native), args, kwargs)
        return read
    assert read == expected, (str(native), args, kwargs)
    return read


def waveforms(
    times: Iterable[float], *columns: Iterable[float]
) -> results.TransientResult:
    """A transient of `out`, `ref` and `V1:current` at `times`, 0 where not given."""
    points = np.asarray(list(times), dtype=np.float64)
    values = np.zeros((points.size, len(UNKNOWNS)))
    for index, column in enumerate(columns):
        values[:, index] = list(column)
    statistics = results.Statistics("operating point", points.size, 0, 0, 0)
    return results.TransientResult(UNKNOWNS, points, values, statistics)


def response(
    frequencies: Iterable[float], out: Iterable[complex], ref: Iterable[complex]
) -> results.AcResult:
    points = np.asarray(list(frequencies), dtype=np.float64)
    values = np.zeros((points.size, len(UNKNOWNS)), dtype=np.complex128)
    values[:, 0], values[:, 1] = list(out), list(ref)
    point = results.OperatingPointResult(UNKNOWNS, np.zeros(len(UNKNOWNS)), "newton")
    return results.AcResult(UNKNOWNS, "V1", points, values, point)


def loop_gain(
    frequencies: Iterable[float], gains: Iterable[complex]
) -> results.StabilityResult:
    point = results.OperatingPointResult(UNKNOWNS, np.zeros(len(UNKNOWNS)), "newton")
    return results.StabilityResult(
        UNKNOWNS,
        "VP",
        np.asarray(list(frequencies), dtype=np.float64),
        np.asarray(list(gains), dtype=np.complex128),
        point,
    )


def spectrum(
    frequencies: Iterable[float],
    densities: Iterable[float],
    gains: Iterable[complex] | None,
) -> results.NoiseResult:
    points = np.asarray(list(frequencies), dtype=np.float64)
    noise_input = (
        None
        if gains is None
        else results.NoiseInput("V1", "current", np.asarray(list(gains), dtype=complex))
    )
    point = results.OperatingPointResult(UNKNOWNS, np.zeros(len(UNKNOWNS)), "newton")
    return results.NoiseResult(
        UNKNOWNS,
        "out",
        None,
        "voltage",
        points,
        np.asarray(list(densities), dtype=np.float64),
        (),
        np.empty((0, points.size)),
        noise_input,
        point,
    )


def windowed(
    result: results.AnalysisResult, start: Quantity | None, stop: Quantity | None
) -> None:
    """Checks every measure of a window of `result` alike."""
    for name in ("maximum", "minimum", "peak_to_peak", "average", "rms"):
        for signal, signal_reference in (
            ("out", None),
            ("out", "ref"),
            ("V1:current", None),
        ):
            alike(
                name,
                result,
                "t",
                signal,
                start=start,
                stop=stop,
                reference=signal_reference,
            )
    if isinstance(result, results.TransientResult):
        for low, high in ((0.1, 0.9), (0.2, 0.8)):
            alike(
                "rise_time",
                result,
                "t",
                "out",
                low=low,
                high=high,
                start=start,
                stop=stop,
            )
        alike("overshoot", result, "t", "out", start=start, stop=stop)
        alike("settling_time", result, "t", "out", band=0.05, start=start, stop=stop)
        for edge in ("rising", "falling", "either"):
            alike(
                "crossing",
                result,
                "t",
                "out",
                0.25 * V,
                edge=edge,
                start=start,
                stop=stop,
            )
        alike("frequency", result, "t", "out", start=start, stop=stop)
        alike("frequency", result, "t", "out", level=0.0 * V, start=start, stop=stop)


def small_signal(
    result: results.AcResult | results.StabilityResult,
    output: str | None,
    over: str | None,
) -> None:
    """Checks every small-signal measure of `result` alike."""
    lowest = float(np.min(result.frequencies)) if result.frequencies.size else 1.0
    for at in (lowest * Hz, 1 * kHz, 1.5 * lowest * Hz):
        alike("gain", result, "ac", output, at=at, over=over)
        alike("phase", result, "ac", output, at=at, over=over)
    for name in ("crossover", "phase_margin", "gain_margin"):
        alike(name, result, "ac", output, over=over)
    alike("bandwidth", result, "ac", output, over=over)
    alike("bandwidth", result, "ac", output, drop=20.0, over=over)


def noise(result: results.NoiseResult, start: float, stop: float) -> None:
    """Checks every noise measure of `result` alike."""
    for referred in (False, True):
        for at in (start, stop, (start + stop) / 2, math.sqrt(abs(start * stop))):
            alike("noise_density", result, "n", at=at * Hz, referred=referred)
        alike("integrated_noise", result, "n", referred=referred)
        alike("integrated_noise", result, "n", start=start * Hz, referred=referred)
        alike("integrated_noise", result, "n", stop=stop * Hz, referred=referred)
        alike(
            "integrated_noise",
            result,
            "n",
            start=start * Hz,
            stop=stop * Hz,
            referred=referred,
        )


def test_closed_forms_read_alike() -> None:
    times = np.linspace(0, 20e-6, 2001)
    step = waveforms(times, 1 - np.exp(-times / 1e-6), np.full_like(times, 0.5))
    damped = 1 - np.exp(-0.3 * 2e6 * np.pi * times) * np.cos(2e6 * np.pi * times)
    ringing = waveforms(times, damped, damped / 3)
    sine = waveforms(times, np.sin(2 * np.pi * 2e5 * times), np.zeros_like(times))
    flat = waveforms(times, np.full_like(times, 1.0))
    for result in (step, ringing, sine, flat):
        for start, stop in (
            (None, None),
            (2 * us, None),
            (None, 15 * us),
            (3 * us, 3 * us),
        ):
            windowed(result, start, stop)
        alike("value", result, "t", "out", at=7 * us)
        alike("value", result, "t", "out", at=1 * s)
        alike("value", result, "t", "out")
    frequencies = np.logspace(0, 6, 601)
    low_pass = 1 / (1 + 1j * frequencies / 1e3)
    corner = response(frequencies, low_pass, np.full(frequencies.size, 2 + 0j))
    small_signal(corner, "out", None)
    small_signal(corner, "out", "ref")
    small_signal(corner, None, None)
    three = loop_gain(frequencies, 10 / (1 + 1j * frequencies / 1e3) ** 3)
    small_signal(three, None, None)
    small_signal(three, "out", None)
    hiss = spectrum(
        np.logspace(1, 5, 41),
        16e-18 + 1e-15 / np.logspace(1, 5, 41),
        np.full(41, 2 + 0j),
    )
    noise(hiss, 10.0, 1e5)
    poles = results.PoleZeroResult(
        UNKNOWNS,
        np.array([-3e3 + 4e3j, -3e3 - 4e3j, -1e6 + 0j]),
        None,
        None,
        None,
        None,
        results.OperatingPointResult(UNKNOWNS, np.zeros(3), "newton"),
    )
    for name in ("damping", "dominant_pole", "solutions"):
        alike(name, poles, "pz")


def _example(name: str) -> Any:  # noqa: ANN401 - an example script, loaded as a module
    path = EXAMPLES / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"examples.{name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_an_examples_results_read_alike() -> None:
    example = _example("rc_filter")
    simulation = Simulation(example.build())
    solved = {
        name: simulation.run(analysis)
        for name, analysis in example.plan().analyses.items()
    }
    solved["dc"] = simulation.run(DcSweep("V1", [1 * V, 0 * V, 0.5 * V, 2 * V]))
    solved["n"] = simulation.run(
        Noise("out", [1 * kHz * 10 ** (k / 10) for k in range(31)], source="V1")
    )
    solved["pz"] = simulation.run(PoleZero())
    solved["states"] = simulation.run(OperatingPoints())
    for signal in ("out", "in", "V1:current"):
        alike("value", solved["op"], "op", signal)
    alike("value", solved["op"], "op", "out", reference="in")
    step = solved["step"]
    for start, stop in ((None, None), (1 * us, None), (1.5 * us, 8 * us)):
        windowed(step, start, stop)
    for at in (0 * us, 1.3 * us, 10 * us, 11 * us):
        alike("value", step, "step", "out", at=at)
    for start, stop in ((None, None), (0.25 * V, None), (None, 1.5 * V)):
        windowed(solved["dc"], start, stop)
    alike("value", solved["dc"], "dc", "out", at=0.75 * V)
    alike("value", solved["dc"], "dc", "out", at=1 * A)
    small_signal(solved["bode"], "out", None)
    small_signal(solved["bode"], "out", "in")
    noise(solved["n"], 1e3, 1e6)
    for name in ("damping", "dominant_pole"):
        alike(name, solved["pz"], "pz")
    alike("solutions", solved["states"], "states")
    alike("solutions", solved["op"], "op")


def test_a_resolution_error_precedes_a_reading_error() -> None:
    """A window whose start lies past the axis and whose stop is no time.

    numpy read the start against the times before it converted the stop, and said the
    start lay outside; the engine's measure resolves the stop first, as a batch
    resolves its measures before it runs.
    """
    step = waveforms(np.linspace(0, 1, 11), np.linspace(0, 1, 11))
    native = specs.maximum("t", "out", start=2 * s, stop=1 * V)
    numpy = reference.maximum("t", "out", start=2 * s, stop=1 * V)
    with pytest.raises(DimensionError, match="the stop"):
        native.read(step)
    with pytest.raises(MeasureError, match="lies outside the times"):
        numpy.read(step)


FINITE: Final = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)
"""A value of a signal, of either sign of zero and none beyond what numpy reads without
an overflow, which the tests' warnings would make an error."""


@st.composite
def transients(draw: st.DrawFn) -> results.TransientResult:
    """A transient at increasing times, its signals random, flat or swinging."""
    times = sorted(
        draw(st.lists(st.floats(0, 100), min_size=1, max_size=24, unique=True))
    )
    count = len(times)
    shape = draw(st.sampled_from(["random", "flat", "swing"]))
    match shape:
        case "random":
            out = draw(st.lists(FINITE, min_size=count, max_size=count))
        case "flat":
            out = [draw(FINITE)] * count
        case _:
            frequency = draw(st.floats(0.01, 1))
            out = [math.sin(2 * math.pi * frequency * time) for time in times]
    ref = draw(st.lists(FINITE, min_size=count, max_size=count))
    return waveforms(
        times, out, ref, draw(st.lists(FINITE, min_size=count, max_size=count))
    )


def ends(points: Iterable[float], unit: Quantity) -> st.SearchStrategy[Quantity | None]:
    """A window's end: none, a point of the axis, or anywhere about it."""
    values = list(points)
    low, high = min(values), max(values)
    span = high - low
    return st.one_of(
        st.none(),
        st.sampled_from(values).map(lambda value: value * unit),
        st.floats(low - 0.1 * span - 1, high + 0.1 * span + 1).map(
            lambda value: value * unit
        ),
    )


@SETTINGS
@given(st.data())
def test_transients_read_alike(data: st.DataObject) -> None:
    result = data.draw(transients())
    start = data.draw(ends(result.times, s))
    stop = data.draw(ends(result.times, s))
    windowed(result, start, stop)
    alike("value", result, "t", "out", at=data.draw(ends(result.times, s)))


@SETTINGS
@given(st.data())
def test_sweeps_read_alike(data: st.DataObject) -> None:
    levels = data.draw(st.lists(st.floats(-10, 10), min_size=1, max_size=24))
    quantity = data.draw(st.sampled_from(QUANTITIES))
    count = len(levels)
    values = np.column_stack(
        [data.draw(st.lists(FINITE, min_size=count, max_size=count)) for _ in UNKNOWNS]
    )
    result = results.DcSweepResult(UNKNOWNS, "V1", quantity, np.array(levels), values)
    unit = V if quantity == "voltage" else A
    windowed(result, data.draw(ends(levels, unit)), data.draw(ends(levels, unit)))
    alike("value", result, "dc", "out", at=data.draw(ends(levels, unit)))


DENSITIES: Final = st.just(0.0) | st.floats(1e-30, 1e-9)
"""A noise density: none, or within 21 decades, as a circuit's are.

Densities 308 decades apart make a power law between them overflow, and Python's
`math`, which numpy's noise took its logarithms with, raises there, where the engine's
reads infinity or not a number, which leaves the requirement unmeasured."""

QUANTITIES: Final[tuple[results.Quantity, ...]] = ("voltage", "current")
"""What a sweep's source levels can be."""

PHASORS: Final = st.builds(complex, st.floats(-1e3, 1e3), st.floats(-1e3, 1e3))
"""A phasor, zero now and then."""


@SETTINGS
@given(st.data())
def test_responses_and_loops_read_alike(data: st.DataObject) -> None:
    frequencies = data.draw(
        st.lists(
            st.one_of(st.floats(1e-3, 1e9), st.floats(-1, 0)),
            min_size=1,
            max_size=24,
            unique=True,
        )
    )
    count = len(frequencies)
    phasors = st.lists(PHASORS, min_size=count, max_size=count)
    result = response(frequencies, data.draw(phasors), data.draw(phasors))
    small_signal(result, "out", data.draw(st.sampled_from([None, "ref"])))
    small_signal(loop_gain(frequencies, data.draw(phasors)), None, None)


@SETTINGS
@given(st.data())
def test_noise_reads_alike(data: st.DataObject) -> None:
    frequencies = data.draw(
        st.lists(st.floats(1e-3, 1e9), min_size=1, max_size=24, unique=True)
    )
    count = len(frequencies)
    densities = data.draw(st.lists(DENSITIES, min_size=count, max_size=count))
    gains = data.draw(
        st.none()
        | st.lists(
            PHASORS,
            min_size=count,
            max_size=count,
        )
    )
    result = spectrum(frequencies, densities, gains)
    low, high = min(frequencies), max(frequencies)
    noise(
        result,
        data.draw(st.floats(low / 2, high * 2)),
        data.draw(st.floats(low / 2, high * 2)),
    )


@SETTINGS
@given(st.lists(PHASORS, max_size=8), st.integers(0, 4))
def test_poles_and_solutions_read_alike(poles: list[complex], count: int) -> None:
    point = results.OperatingPointResult(UNKNOWNS, np.zeros(3), "newton")
    found = results.PoleZeroResult(
        UNKNOWNS, np.array(poles, dtype=complex), None, None, None, None, point
    )
    for name in ("damping", "dominant_pole"):
        alike(name, found, "pz")
    alike(
        "solutions", results.OperatingPointsResult(UNKNOWNS, (point,) * count), "states"
    )


def amplitudes(
    result: results.TransientResult,
    fundamental: float,
    harmonics: int,
    start: Quantity | None,
) -> tuple[results.Real, results.Real]:
    """The samples a distortion reads, and numpy's amplitudes of its harmonics.

    They are as `numpy_measures.distortion` takes them, of harmonics 1 to
    `harmonics`.
    """
    window = _Window(_Signal("out"), start, None)
    times, values, _ = window.transient(result, "a distortion")
    span = (times[-1] - times[0]) * fundamental
    periods = 1 if start is None else math.floor(span * (1 + 1e-9))
    samples = periods * 1024
    at = times[-1] - periods / fundamental * (1 - np.arange(samples) / samples)
    sampled = np.interp(at, times, values)
    spectrum = np.abs(np.fft.rfft(sampled))
    return sampled, spectrum[periods : periods * harmonics + 1 : periods]


def distortion_bound(sampled: results.Real) -> float:
    """How far either's amplitude of a harmonic may lie from the exact transform's.

    The engine sums each part of the transform pairwise, over sums of 16 terms in each
    of 8 accumulators, so each part is the exact sum of its rounded terms within
    (19 + log2 N) roundings of their absolute sum, and each term, a sample times a
    twiddle accurate within 5 units of the last place, within 6 more: at most
    (25 + log2 N) ε ‖x‖₁ for each part, √2 times that for the magnitude. numpy's
    transform lies within log2 N · η · ‖X‖₂ of the exact one, η about 8 ε for twiddles
    accurate within a unit of the last place (Higham, "Accuracy and Stability of
    Numerical Algorithms", 2nd ed., Theorem 24.2), and ‖X‖₂ = √N ‖x‖₂.
    """
    count = sampled.size
    depth = math.log2(count)
    epsilon = float(np.finfo(np.float64).eps)
    engine = math.sqrt(2) * (25 + depth) * epsilon * float(np.sum(np.abs(sampled)))
    numpy = 8 * depth * epsilon * math.sqrt(count) * float(np.linalg.norm(sampled))
    return engine + numpy


@SETTINGS
@given(st.data())
def test_distortions_read_within_rounding(data: st.DataObject) -> None:
    periods = data.draw(st.integers(1, 4))
    times = np.linspace(0, periods * 1e-3, data.draw(st.integers(2, 400)))
    angle = 2 * math.pi * 1e3 * times
    # A fundamental of some weight, which a distortion reads its harmonics against.
    weights = [
        data.draw(st.floats(0.1, 1)),
        *data.draw(st.lists(st.floats(0, 1), min_size=3, max_size=3)),
    ]
    out = np.zeros_like(times)
    for order, weight in enumerate(weights):
        out += weight * np.sin((order + 1) * angle + order)
    result = waveforms(times, out)
    harmonics = data.draw(st.integers(2, 12))
    start = data.draw(st.sampled_from([None, 0 * s, 0.5e-3 * s]))
    fundamental = data.draw(st.sampled_from([1e3, 1.5e3, 0.4e3]))
    native = specs.distortion(
        "t", "out", fundamental=fundamental * Hz, harmonics=harmonics, start=start
    )
    numpy = reference.distortion(
        "t", "out", fundamental=fundamental * Hz, harmonics=harmonics, start=start
    )
    found, expected = outcome(native, result), outcome(numpy, result)
    if expected[1].endswith(f"spans less than a period of {fundamental * Hz}"):
        assert found == expected
        return
    sampled, read_amplitudes = amplitudes(result, fundamental, harmonics, start)
    bound = 2 * distortion_bound(sampled)
    first, total = float(read_amplitudes[0]), float(np.linalg.norm(read_amplitudes[1:]))
    if first <= bound:
        # The fundamental lies within the transforms' rounding of none, so whether
        # either finds it at all is rounding's choice.
        return
    value, read = float.fromhex(found[0]), float.fromhex(expected[0])
    spread = math.sqrt(harmonics - 1) * bound
    if total > spread:
        decibels = 20 / math.log(10)
        allowed = decibels * (spread / (total - spread) + bound / (first - bound))
        assert abs(value - read) <= allowed, (value, read, allowed)
    else:
        # Harmonics within rounding of none: the reading is at most that of the bound.
        highest = 20 * math.log10(max((total + spread) / (first - bound), 1e-300))
        assert value <= highest, (value, highest)
