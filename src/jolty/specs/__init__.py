# Copyright 2026 Maxence Faldor
"""Circuit specifications measured against simulation plan results.

A [`Measure`][jolty.specs.Measure] reads one value from a named analysis result, such
as the gain of ``out`` at 1 kHz in ``bode``. Comparing a measure against a bound creates
a [`Requirement`][jolty.specs.Requirement], and a plan script's ``specs()`` returns a
[`Specs`][jolty.specs.Specs] collection. Each requirement evaluates to passed, marginal
(passing by less than its margin), failed, or unmeasured when the analysis fails or the
measure cannot be read. Requirements evaluate under nominal conditions and across all
plan corners; the resulting [`Verdict`][jolty.specs.Verdict] reports the worst corner,
measured value, and slack (positive inside bounds, negative outside). Over Monte Carlo
samples, [`Spread`][jolty.specs.Spread] summarizes measured values per requirement, and
[`Yield`][jolty.specs.Yield] reports the passing fraction with a 95 %
[`wilson`][jolty.specs.wilson] interval against the yield target.

Measures target a circuit unknown such as node voltage ``out`` or branch current
``V1:current``, or a differential voltage relative to ``reference``. Physical bounds
use dimensioned quantities such as ``2 * V``; decibel gains, phase degrees, percent
overshoot, and damping ratios use plain numbers.

Built-in measures resolve signals to unknown indices and parameters to SI base units as
[`ResolvedMeasure`][jolty.specs.ResolvedMeasure] objects so the Rust engine computes
them directly during batched runs without Python overhead. Custom `Measure` objects
supplying a Python `read` callable evaluate in Python.

Examples:
    >>> from jolty.units import V, kHz
    >>> specs = Specs([value("op", "out") >= 2 * V, gain("bode", "out", at=kHz) >= -3])
    >>> [str(requirement) for requirement in specs.requirements]
    ['out in op ≥ 2 V', 'gain of out at 1 kHz in bode ≥ -3 dB']
"""

__lazy_modules__ = ["jolty.specs._measures", "jolty.specs._requirements"]

from jolty.specs._measures import (
    MEASURES,
    MeasureCategory,
    MeasureDimension,
    MeasureSpec,
    bandwidth,
    crossing,
    crossover,
    damping,
    diagnose,
    distortion,
    dominant_pole,
    frequency,
    gain,
    gain_margin,
    integrated_noise,
    maximum,
    mean,
    minimum,
    noise_density,
    overshoot,
    peak_to_peak,
    period,
    phase,
    phase_margin,
    rise_time,
    rms,
    settling_time,
    solutions,
    value,
)
from jolty.specs._requirements import (
    Bound,
    ComplexCotangent,
    Edge,
    ExtendedReal,
    Judged,
    LoopCotangent,
    Measure,
    MeasureCotangent,
    MeasureError,
    NoiseCotangent,
    PolesCotangent,
    Reading,
    RealCotangent,
    Requirement,
    ResolvedMeasure,
    Specs,
    Spread,
    Status,
    SummaryStatistics,
    Verdict,
    Yield,
    describe,
    evaluate,
    format_reading,
    judge,
    judged,
    reading,
    spread,
    spread_of,
    validate_specs,
    wilson,
    worst,
)

__all__ = [
    "MEASURES",
    "Bound",
    "ComplexCotangent",
    "Edge",
    "ExtendedReal",
    "Judged",
    "LoopCotangent",
    "Measure",
    "MeasureCategory",
    "MeasureCotangent",
    "MeasureDimension",
    "MeasureError",
    "MeasureSpec",
    "NoiseCotangent",
    "PolesCotangent",
    "Reading",
    "RealCotangent",
    "Requirement",
    "ResolvedMeasure",
    "Specs",
    "Spread",
    "Status",
    "SummaryStatistics",
    "Verdict",
    "Yield",
    "bandwidth",
    "crossing",
    "crossover",
    "damping",
    "describe",
    "diagnose",
    "distortion",
    "dominant_pole",
    "evaluate",
    "format_reading",
    "frequency",
    "gain",
    "gain_margin",
    "integrated_noise",
    "judge",
    "judged",
    "maximum",
    "mean",
    "minimum",
    "noise_density",
    "overshoot",
    "peak_to_peak",
    "period",
    "phase",
    "phase_margin",
    "reading",
    "rise_time",
    "rms",
    "settling_time",
    "solutions",
    "spread",
    "spread_of",
    "validate_specs",
    "value",
    "wilson",
    "worst",
]
