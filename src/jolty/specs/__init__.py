# Copyright 2026 Maxence Faldor
"""Specs: what a circuit must do, measured on the results of its plan's analyses.

A [`Measure`][jolty.specs.Measure] reads one number from the result of a named
analysis, such as the gain of ``out`` at 1 kHz in the analysis ``bode``. Comparing it
with a bound makes a [`Requirement`][jolty.specs.Requirement], and a plan script's
``specs()`` returns its requirements as [`Specs`][jolty.specs.Specs]. Each requirement
passes, fails, or goes unmeasured when its analysis did not end or its measure
cannot be read; one that passes by less than its margin is marginal. A requirement
holds at the plan's own conditions and at each of its corners, and its
[`Verdict`][jolty.specs.Verdict] is the worst of them: it states the value read there
and the slack, how far the value lies inside its bounds, negative outside them. Over a
Monte Carlo analysis's samples, each requirement's [`Spread`][jolty.specs.Spread] states
the values read, and the [`Yield`][jolty.specs.Yield] how many samples meet every
requirement, with the [`wilson`][jolty.specs.wilson] interval of their share, against
the specs' yield target.

A measure reads a signal, an unknown of the circuit such as the voltage ``out`` or
the current ``V1:current``, or a voltage relative to the voltage of a ``reference``.
Bounds are quantities, such as ``2 * V``; a gain is in decibels, a phase in degrees, an
overshoot in percent and a damping ratio a plain number, so their bounds are numbers.

The engine reads every measure this module makes, in Rust, from a result's arrays: the
measure resolves its signals to the result's unknowns and its quantities to SI base
units, its [`ResolvedMeasure`][jolty.specs.ResolvedMeasure] form, and the engine
computes the number, as a batch of runs does without Python. A `Measure` a script makes
with a `read` of its own is read in Python.

Examples:
    >>> from jolty.units import V, kHz
    >>> specs = Specs([value("op", "out") >= 2 * V, gain("bode", "out", at=kHz) >= -3])
    >>> [str(requirement) for requirement in specs.requirements]
    ['out in op ≥ 2 V', 'gain of out at 1 kHz in bode ≥ -3 dB']
"""

from jolty.specs._measures import (
    MEASURES,
    Edge,
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
    ExtendedReal,
    Judged,
    Measure,
    MeasureError,
    Plain,
    Reading,
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
    "Edge",
    "ExtendedReal",
    "Judged",
    "Measure",
    "MeasureCategory",
    "MeasureDimension",
    "MeasureError",
    "MeasureSpec",
    "Plain",
    "Reading",
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
