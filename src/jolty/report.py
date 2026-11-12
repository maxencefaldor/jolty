# Copyright 2026 Maxence Faldor
"""Reports: what each analysis found, as data and as text for people and agents.

A report states facts: the value of every unknown at an operating point, and the first,
lowest, highest and last value of every unknown over a sweep, a frequency response or a
transient; and the level of every logic net at an operating point, and its first and
last level and how often it changed over a sweep or a transient. It lists the first
unknowns and logic nets, up to a limit; the results file holds them all.
A corner's report states whether the circuit elaborates there and which analyses fail,
and a Monte Carlo analysis's what each requirement read over its samples and the yield.
"""

import math
from dataclasses import replace
from typing import Annotated, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from jolty import results, specs
from jolty.specs import Spread, Yield
from jolty.units import format_si

type Quantity = Literal["voltage", "current"]

_SYMBOLS = {"voltage": "V", "current": "A"}

_STRATEGIES = {
    "direct": "Newton iteration",
    "gmin_stepping": "gmin stepping",
    "source_stepping": "source stepping",
}


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Value(_Model):
    """The value of one unknown, in volts or amperes."""

    unknown: str
    quantity: Quantity
    value: float


class Extent(_Model):
    """The values one unknown takes over an analysis, in volts or amperes."""

    unknown: str
    quantity: Quantity
    first: float
    minimum: float
    maximum: float
    last: float


type LogicLevel = Literal[0, 1] | None
"""A logic level: 0, 1, or null when unknown."""


class Level(_Model):
    """The level of one logic net."""

    net: str
    level: LogicLevel


class Activity(_Model):
    """The levels one logic net takes over an analysis."""

    net: str
    first: LogicLevel
    changes: int = Field(description="How many times its level changed.")
    last: LogicLevel


class PointReport(_Model):
    """An operating point."""

    kind: Literal["operating_point"] = "operating_point"
    name: str
    strategy: str = Field(
        description="How the engine found it: direct, gmin_stepping or source_stepping."
    )
    values: tuple[Value, ...]
    levels: tuple[Level, ...] = ()
    unlisted: int = Field(description="Unknowns and logic nets left out of the list.")

    def describe(self) -> str:
        """The report as text."""
        strategy = _STRATEGIES.get(self.strategy, self.strategy)
        rows = [
            [value.unknown, format_si(value.value, _SYMBOLS[value.quantity])]
            for value in self.values
        ]
        rows.extend([level.net, _written(level.level)] for level in self.levels)
        head = f"{self.name}: operating point, found by {strategy}"
        return _join(head, _table(rows), self.unlisted)


class SweepReport(_Model):
    """A DC sweep."""

    kind: Literal["dc_sweep"] = "dc_sweep"
    name: str
    source: str = Field(description="The swept source's path.")
    quantity: Quantity = Field(
        description="Whether its levels are voltages or currents."
    )
    first: float = Field(description="The first level, in volts or amperes.")
    last: float = Field(description="The last level, in volts or amperes.")
    levels: int = Field(description="Levels the sweep solved.")
    extents: tuple[Extent, ...]
    activities: tuple[Activity, ...] = ()
    unlisted: int = Field(description="Unknowns and logic nets left out of the list.")

    def describe(self) -> str:
        """The report as text."""
        unit = _SYMBOLS[self.quantity]
        head = (
            f"{self.name}: DC sweep of {self.source} from "
            f"{format_si(self.first, unit)} to {format_si(self.last, unit)} "
            f"in {self.levels} levels"
        )
        lines = _extents(self.extents) + _activities(self.activities)
        return _join(head, lines, self.unlisted)


class ResponseReport(_Model):
    """A small-signal frequency response, as magnitudes per unit excitation."""

    kind: Literal["ac"] = "ac"
    name: str
    source: str = Field(description="The excited source's path.")
    lowest: float = Field(description="Lowest frequency, in hertz.")
    highest: float = Field(description="Highest frequency, in hertz.")
    frequencies: int = Field(description="Frequencies the analysis solved.")
    magnitudes: tuple[Extent, ...] = Field(
        description="Magnitude of each unknown's phasor, from the lowest frequency "
        "to the highest."
    )
    unlisted: int = Field(description="Unknowns left out of the list.")

    def describe(self) -> str:
        """The report as text."""
        head = (
            f"{self.name}: response to {self.source} at {self.frequencies} "
            "frequencies from "
            f"{format_si(self.lowest, 'Hz')} to {format_si(self.highest, 'Hz')}, "
            "magnitudes per unit excitation"
        )
        return _join(head, _extents(self.magnitudes), self.unlisted)


class TransientReport(_Model):
    """A transient analysis."""

    kind: Literal["transient"] = "transient"
    name: str
    stop: float = Field(description="Time the analysis ended, in seconds.")
    accepted: int = Field(description="Time steps accepted.")
    rejected: int = Field(
        description="Time steps rejected for their error or Newton, or to land on a "
        "logic event."
    )
    extents: tuple[Extent, ...]
    activities: tuple[Activity, ...] = ()
    unlisted: int = Field(description="Unknowns and logic nets left out of the list.")

    def describe(self) -> str:
        """The report as text."""
        head = (
            f"{self.name}: transient to {format_si(self.stop, 's')} "
            f"in {self.accepted} steps, {self.rejected} rejected"
        )
        lines = _extents(self.extents) + _activities(self.activities)
        return _join(head, lines, self.unlisted)


class StabilityReport(_Model):
    """The loop gain through a probe, and the margins it leaves."""

    kind: Literal["stability"] = "stability"
    name: str
    probe: str = Field(description="The probe's path.")
    lowest: float = Field(description="Lowest frequency, in hertz.")
    highest: float = Field(description="Highest frequency, in hertz.")
    frequencies: int = Field(description="Frequencies the analysis solved.")
    gain: float = Field(description="The loop gain at the lowest frequency, in dB.")
    phase: float = Field(
        description="The loop gain's phase at the lowest frequency, in degrees."
    )
    crossover: float | None = Field(
        description="Where the loop gain first falls through 0 dB, in hertz; null "
        "when it does not over the positive frequencies."
    )
    phase_margin: float | None = Field(
        description="180° plus the loop gain's phase at the crossover, in degrees."
    )
    gain_margin: float | None = Field(
        description="Minus the loop gain where its phase first falls through -180°, "
        "in dB; null when it does not over the positive frequencies."
    )

    def describe(self) -> str:
        """The report as text."""
        lowest, highest = format_si(self.lowest, "Hz"), format_si(self.highest, "Hz")
        head = (
            f"{self.name}: loop gain through {self.probe} at {self.frequencies} "
            f"frequencies from {lowest} to {highest}"
        )
        lines = [head, f"  {self.gain:.4g} dB and {self.phase:.4g}° at {lowest}"]
        if self.crossover is None or self.phase_margin is None:
            lines.append("  it never falls through 0 dB")
        else:
            lines.append(
                f"  crossover at {format_si(self.crossover, 'Hz')}, phase margin "
                f"{self.phase_margin:.4g}°"
            )
        if self.gain_margin is None:
            lines.append("  its phase never falls through -180°")
        else:
            lines.append(f"  gain margin {self.gain_margin:.4g} dB")
        return "\n".join(lines)


class NoiseContributor(_Model):
    """What one noise source of a device adds to a noise analysis's output."""

    instance: str = Field(description="The device's path.")
    noise: str = Field(
        description="What makes the noise, such as thermal or collector shot."
    )
    rms: float = Field(
        description="Its RMS noise at the output over the frequencies, in volts or "
        "amperes as the output is."
    )
    share: float = Field(
        description="Its fraction of the output's noise power over the frequencies."
    )


class NoiseReport(_Model):
    """The noise at an output, and the devices that make the most of it."""

    kind: Literal["noise"] = "noise"
    name: str
    output: str = Field(description="The output, an unknown by name.")
    reference: str | None = Field(
        description="The node the output voltage is over; null for the ground."
    )
    quantity: Quantity = Field(description="What the output is.")
    lowest: float = Field(description="Lowest frequency, in hertz.")
    highest: float = Field(description="Highest frequency, in hertz.")
    frequencies: int = Field(description="Frequencies the analysis solved.")
    first: float = Field(
        description="The output's noise density at the lowest frequency, in V/√Hz or "
        "A/√Hz."
    )
    last: float = Field(
        description="The output's noise density at the highest frequency, in V/√Hz or "
        "A/√Hz."
    )
    rms: float | None = Field(
        description="The output's RMS noise over the frequencies, in volts or "
        "amperes; null for one frequency."
    )
    source: str | None = Field(
        description="The source the noise is referred to; null for none."
    )
    referred: float | None = Field(
        description="The RMS noise referred to the source over the frequencies, in "
        "the unit of its level; null without a source or for one frequency."
    )
    referred_quantity: Quantity | None = Field(
        description="What the source's level is; null without a source."
    )
    contributors: tuple[NoiseContributor, ...] = Field(
        description="The noise sources that make the most noise, the most first."
    )
    unlisted: int = Field(description="Noise sources left out of the list.")

    def describe(self) -> str:
        """The report as text."""
        lowest, highest = format_si(self.lowest, "Hz"), format_si(self.highest, "Hz")
        over = "" if self.reference is None else f" over {self.reference}"
        unit = _SYMBOLS[self.quantity]
        head = (
            f"{self.name}: noise at {self.output}{over} at {self.frequencies} "
            f"frequencies from {lowest} to {highest}"
        )
        density = f"{unit}/\N{SQUARE ROOT}Hz"
        first, last = format_si(self.first, density), format_si(self.last, density)
        lines = [head, f"  {first} at {lowest}, {last} at {highest}"]
        if self.rms is not None:
            total = f"  {format_si(self.rms, unit)} RMS over the frequencies"
            if self.referred is not None and self.referred_quantity is not None:
                referred = format_si(self.referred, _SYMBOLS[self.referred_quantity])
                total += f", {referred} RMS referred to {self.source}"
            lines.append(total)
            lines.extend(
                _table(
                    [
                        [
                            each.instance,
                            each.noise,
                            f"{format_si(each.rms, unit)} RMS",
                            f"{each.share:.1%}",
                        ]
                        for each in self.contributors
                    ]
                )
            )
        return _join(head, lines[1:], self.unlisted)


class Root(_Model):
    """A pole or a zero, or a conjugate pair of them."""

    real: float = Field(description="The real part, in radians per second.")
    imaginary: float = Field(
        description="The imaginary part, in radians per second: zero for a real root, "
        "and positive for a pair, whose other member is its conjugate."
    )
    frequency: float = Field(
        description="The natural frequency, the magnitude over 2π, in hertz."
    )
    damping: float | None = Field(
        description="The damping ratio, minus the real part over the magnitude: 1 for "
        "a real root in the left half-plane, negative in the right half-plane; null at "
        "the origin."
    )

    def written(self, kind: str) -> list[str]:
        """The root as a table's row: its kind, value, frequency and damping."""
        value = format_si(self.real, "rad/s")
        if self.imaginary:
            value += f" ± j{format_si(self.imaginary, 'rad/s')}"
        damping = "" if self.damping is None else f"damping {self.damping:.4g}"
        return [kind, value, format_si(self.frequency, "Hz"), damping]


class PoleZeroReport(_Model):
    """The poles of the linearized circuit, and the zeros of a transfer function."""

    kind: Literal["pole_zero"] = "pole_zero"
    name: str
    source: str | None = Field(description="The source the transfer function is from.")
    output: str | None = Field(description="The output the transfer function is to.")
    reference: str | None = Field(
        description="The node the output voltage is over; null for the ground."
    )
    poles: tuple[Root, ...] = Field(
        description="The poles, by increasing magnitude, a conjugate pair once."
    )
    zeros: tuple[Root, ...] | None = Field(
        description="The zeros, likewise, given a source and an output."
    )
    unlisted: int = Field(description="Roots left out of the lists.")

    def describe(self) -> str:
        """The report as text."""
        count = sum(2 if root.imaginary else 1 for root in self.poles)
        head = f"{self.name}: {count} poles listed"
        if self.zeros is not None:
            zeros = sum(2 if root.imaginary else 1 for root in self.zeros)
            over = "" if self.reference is None else f" over {self.reference}"
            head += (
                f", and {zeros} zeros of {self.output}{over} per {self.source} listed"
            )
        rows = [root.written("pole") for root in self.poles]
        rows.extend(root.written("zero") for root in self.zeros or ())
        return _join(head, _table(rows), self.unlisted)


class Derivative(_Model):
    """An output's derivative with respect to one parameter."""

    instance: str = Field(description="The device's path.")
    parameter: str = Field(description="The parameter's name, such as resistance.")
    value: float = Field(description="The parameter's value, in SI base units.")
    derivative: float = Field(
        description="The output's derivative with respect to the parameter, in SI "
        "base units: at the operating point, or the largest in magnitude over the "
        "points, signed, a response's magnitude."
    )
    at: float | None = Field(
        default=None,
        description="The level, frequency or time it is largest at, in SI base units; "
        "null at an operating point.",
    )


class OutputSensitivity(_Model):
    """An output's derivatives with respect to the parameters."""

    output: str
    quantity: Quantity
    value: float | None = Field(
        default=None,
        description="Its value at the operating point, in volts or amperes; null "
        "along another analysis.",
    )
    derivatives: tuple[Derivative, ...] = Field(
        description="Its derivative with respect to the first parameters, in the "
        "circuit's order."
    )


type Differentiated = Literal["operating_point", "dc_sweep", "ac", "transient"]
"""The analysis a sensitivity analysis differentiates."""

_ALONG: Final[dict[Differentiated, str]] = {
    "dc_sweep": "along a DC sweep at {points} levels",
    "ac": "over a response at {points} frequencies",
    "transient": "over a transient at {points} times",
}


class SensitivityReport(_Model):
    """Sensitivities at the operating point, or along a sweep, a response or time."""

    kind: Literal["sensitivity"] = "sensitivity"
    name: str
    of: Differentiated = Field(description="The analysis differentiated.")
    points: int = Field(description="The points the derivatives are taken at.")
    strategy: str | None = Field(
        default=None,
        description="How the engine found the operating point: direct, gmin_stepping "
        "or source_stepping; null along a sweep or a transient.",
    )
    unit: str = Field(
        default="",
        description="The unit of the points: V or A along a sweep, Hz, or s.",
    )
    parameters: int = Field(
        description="Parameters the derivatives are with respect to."
    )
    outputs: tuple[OutputSensitivity, ...]
    unlisted: int = Field(description="Parameters left out of each output's list.")

    def describe(self) -> str:
        """The report as text."""
        if self.of == "operating_point":
            strategy = _STRATEGIES.get(self.strategy or "", self.strategy)
            where = f"at the operating point, found by {strategy}"
        else:
            where = _ALONG[self.of].format(points=self.points)
        head = (
            f"{self.name}: derivatives with respect to {self.parameters} parameters "
            f"{where}"
        )
        lines = [head]
        for output in self.outputs:
            if output.value is None:
                lines.append(f"  {output.output}")
            else:
                value = format_si(output.value, _SYMBOLS[output.quantity])
                lines.append(f"  {output.output} = {value}")
            rows = [
                [
                    each.instance,
                    each.parameter,
                    f"at {each.value!r}",
                    self._derivative(each),
                ]
                for each in output.derivatives
            ]
            lines.extend(f"  {row}" for row in _table(rows))
            if self.unlisted:
                lines.append(f"      ... and {self.unlisted} more in the results file")
        return "\n".join(lines)

    def _derivative(self, derivative: Derivative) -> str:
        if derivative.at is None:
            return repr(derivative.derivative)
        at = format_si(derivative.at, self.unit)
        return f"largest {derivative.derivative!r} at {at}"


class FailureReport(_Model):
    """An analysis that failed."""

    kind: Literal["failure"] = "failure"
    name: str
    error: str = Field(description="Why the analysis failed.")

    def describe(self) -> str:
        """The report as text."""
        return f"{self.name}: failed: {self.error}"


class CornerReport(_Model):
    """What a corner of the plan found.

    It lists why the circuit does not elaborate there, or which analyses failed there.
    """

    name: str
    temperature: float = Field(description="The corner's temperature, in kelvins.")
    problems: tuple[str, ...] = Field(
        default=(), description="Why the circuit does not elaborate at the corner."
    )
    failures: tuple[FailureReport, ...] = Field(
        default=(), description="The analyses that failed at the corner."
    )

    @property
    def ok(self) -> bool:
        """Whether the circuit elaborates at the corner and every analysis ends."""
        return not self.problems and not self.failures

    def describe(self) -> str:
        """The report as text."""
        head = f"corner {self.name}, at {self.temperature - 273.15:.4g} °C"
        if self.problems:
            listed = "".join(f"\n  {problem}" for problem in self.problems)
            return f"{head}: the circuit does not elaborate:{listed}"
        if self.failures:
            listed = "".join(f"\n  {failure.describe()}" for failure in self.failures)
            return f"{head}:{listed}"
        return f"{head}: every analysis ended"


class MonteCarloReport(_Model):
    """What a Monte Carlo analysis found over its samples."""

    samples: int
    seed: int
    toleranced: int = Field(description="The toleranced parameters each sample draws.")
    corners: tuple[str, ...] = Field(
        default=(),
        description="The corners each sample also ran at, after the plan's own "
        "conditions.",
    )
    failed: int = Field(
        description="Samples on which the circuit did not elaborate or an analysis "
        "failed, at the plan's own conditions or a corner."
    )
    spreads: tuple[Spread, ...] = Field(
        default=(), description="What each requirement of the specs read."
    )
    passing: Yield | None = Field(
        default=None,
        description="How many samples meet every requirement; null without specs.",
    )

    def describe(self) -> str:
        """The report as text."""
        at = "".join(f", at {name}" for name in self.corners)
        head = (
            f"monte carlo: {self.samples} samples of {self.toleranced} toleranced "
            f"parameters{at}, seed {self.seed}; {self.failed} failed"
        )
        lines = [head]
        lines.extend(f"  {spread.describe(self.samples)}" for spread in self.spreads)
        if self.passing is not None:
            lines.append(f"  {self.passing.describe()}")
        return "\n".join(lines)


type AnalysisReport = Annotated[
    PointReport
    | SweepReport
    | ResponseReport
    | TransientReport
    | SensitivityReport
    | StabilityReport
    | NoiseReport
    | PoleZeroReport
    | FailureReport,
    Field(discriminator="kind"),
]
"""The report of any analysis."""


def report(name: str, result: results.Result, listed: int) -> AnalysisReport:
    """The report of the analysis `name`.

    It lists its first `listed` unknowns, then its first logic nets, up to `listed` in
    all.
    """
    unknowns = result.unknowns[:listed]
    nets = _nets(result)[: listed - len(unknowns)]
    unlisted = len(result.unknowns) + len(_nets(result)) - len(unknowns) - len(nets)
    match result:
        case results.OperatingPoint():
            values = tuple(
                Value(
                    unknown=unknown.name, quantity=unknown.quantity, value=float(value)
                )
                for unknown, value in zip(unknowns, result.values[:listed], strict=True)
            )
            levels = tuple(
                Level(net=net, level=_level(float(level)))
                for net, level in zip(nets, result.logic, strict=False)
            )
            return PointReport(
                name=name,
                strategy=result.strategy,
                values=values,
                levels=levels,
                unlisted=unlisted,
            )
        case results.Sweep():
            return SweepReport(
                name=name,
                source=result.source,
                quantity=result.quantity,
                first=float(result.levels[0]),
                last=float(result.levels[-1]),
                levels=len(result.levels),
                extents=_extents_of(unknowns, result.values[:, :listed]),
                activities=tuple(_activity(net, result.level(net)) for net in nets),
                unlisted=unlisted,
            )
        case results.Response():
            return ResponseReport(
                name=name,
                source=result.source,
                lowest=float(result.frequencies[0]),
                highest=float(result.frequencies[-1]),
                frequencies=len(result.frequencies),
                magnitudes=_extents_of(unknowns, np.abs(result.values[:, :listed])),
                unlisted=unlisted,
            )
        case results.Waveforms():
            return TransientReport(
                name=name,
                stop=float(result.times[-1]),
                accepted=result.statistics.accepted,
                rejected=result.statistics.rejected,
                extents=_extents_of(unknowns, result.values[:, :listed]),
                activities=tuple(
                    _activity(net, result.level(net).levels) for net in nets
                ),
                unlisted=unlisted,
            )
        case _:
            return _linear(name, result, listed)


def _linear(
    name: str,
    result: results.Sensitivities
    | results.LoopGain
    | results.NoiseSpectrum
    | results.PolesZeros,
    listed: int,
) -> AnalysisReport:
    """The report of sensitivities, a loop gain, a noise spectrum or poles and zeros."""
    match result:
        case results.Sensitivities():
            return _sensitivities(name, result, listed)
        case results.LoopGain():
            return _stability(name, result)
        case results.NoiseSpectrum():
            return _noise(name, result, listed)
        case results.PolesZeros():
            return _poles_zeros(name, result, listed)


def _poles_zeros(name: str, result: results.PolesZeros, listed: int) -> PoleZeroReport:
    """The report of the pole-zero analysis `name`, its first `listed` roots."""
    poles = _roots(result.poles)
    zeros = None if result.zeros is None else _roots(result.zeros)
    every = len(poles) + len(zeros or ())
    shown_poles = poles[:listed]
    shown_zeros = None if zeros is None else zeros[: listed - len(shown_poles)]
    return PoleZeroReport(
        name=name,
        source=result.source,
        output=result.output,
        reference=result.reference,
        poles=shown_poles,
        zeros=shown_zeros,
        unlisted=every - len(shown_poles) - len(shown_zeros or ()),
    )


def _roots(roots: results.Complex) -> tuple[Root, ...]:
    """`roots` as the report lists them: real ones, and each conjugate pair once."""
    frequencies = results.natural_frequencies(roots)
    dampings = results.dampings(roots)
    return tuple(
        Root(
            real=float(root.real),
            imaginary=float(root.imag),
            frequency=float(frequency),
            damping=None if np.isnan(damping) else float(damping),
        )
        for root, frequency, damping in zip(roots, frequencies, dampings, strict=True)
        if root.imag >= 0
    )


def _noise(name: str, result: results.NoiseSpectrum, listed: int) -> NoiseReport:
    """The report of the noise analysis `name`, its `listed` noisiest sources."""
    order = np.argsort(result.frequencies)
    frequencies = result.frequencies[order]
    banded = len(frequencies) > 1
    powers = [
        results.integrate(frequencies, row[order]) if banded else float(row[order[0]])
        for row in result.contributions
    ]
    total = math.fsum(powers)
    noisiest = sorted(range(len(powers)), key=lambda index: -powers[index])[:listed]
    contributors = tuple(
        NoiseContributor(
            instance=result.sources[index].instance,
            noise=result.sources[index].name,
            rms=math.sqrt(powers[index]),
            share=powers[index] / total if total > 0 else 0.0,
        )
        for index in noisiest
        if banded
    )
    noise_input = result.input
    return NoiseReport(
        name=name,
        output=result.output,
        reference=result.reference,
        quantity=result.quantity,
        lowest=float(frequencies[0]),
        highest=float(frequencies[-1]),
        frequencies=len(frequencies),
        first=math.sqrt(float(result.densities[order[0]])),
        last=math.sqrt(float(result.densities[order[-1]])),
        rms=result.rms() if banded else None,
        source=None if noise_input is None else noise_input.source,
        referred=result.rms(referred=True)
        if banded and noise_input is not None
        else None,
        referred_quantity=None if noise_input is None else noise_input.quantity,
        contributors=contributors,
        unlisted=len(result.sources) - len(contributors),
    )


def _stability(name: str, result: results.LoopGain) -> StabilityReport:
    """The report of the stability analysis `name`.

    Its margins are read over the positive frequencies, as log frequency needs.
    """
    order = np.argsort(result.frequencies)
    lowest = result.gains[order[0]]
    positive = result.frequencies > 0
    read = replace(
        result,
        frequencies=result.frequencies[positive],
        gains=result.gains[positive],
    )
    return StabilityReport(
        name=name,
        probe=result.probe,
        lowest=float(result.frequencies[order[0]]),
        highest=float(result.frequencies[order[-1]]),
        frequencies=len(result.frequencies),
        gain=float(20 * np.log10(np.abs(lowest))),
        phase=float(np.degrees(np.angle(lowest))),
        crossover=_reading(specs.crossover(name), read),
        phase_margin=_reading(specs.phase_margin(name), read),
        gain_margin=_reading(specs.gain_margin(name), read),
    )


def _reading(measurement: specs.Measurement, result: results.Result) -> float | None:
    """What `measurement` reads of `result`, or None when it cannot read it."""
    try:
        return measurement.read(result).value
    except specs.MeasurementError:
        return None


def _sensitivities(
    name: str, result: results.Sensitivities, listed: int
) -> SensitivityReport:
    """The report of the sensitivity analysis `name`, its first `listed` parameters."""
    listed_parameters = min(listed, len(result.parameters))
    match result.result:
        case results.OperatingPoint() as point:
            of, points, strategy, unit = "operating_point", 1, point.strategy, ""
        case results.Sweep() as sweep:
            of, points, strategy = "dc_sweep", len(sweep.levels), None
            unit = _SYMBOLS[sweep.quantity]
        case results.Response() as response:
            of, points, unit = "ac", len(response.frequencies), "Hz"
            strategy = response.operating_point.strategy
        case results.Waveforms() as waveforms:
            of, points, strategy, unit = "transient", len(waveforms.times), None, "s"
    return SensitivityReport(
        name=name,
        of=of,
        points=points,
        strategy=strategy,
        unit=unit,
        parameters=len(result.parameters),
        outputs=tuple(
            _output(result, output, listed_parameters) for output in result.outputs
        ),
        unlisted=len(result.parameters) - listed_parameters,
    )


def _output(
    result: results.Sensitivities, output: str, listed: int
) -> OutputSensitivity:
    """The value of `output`, and its derivatives by the first `listed` parameters.

    Along an analysis other than the operating point, each is where it is largest.
    """
    gradient = result.gradient(output)
    column = results.column(result.unknowns, output)
    quantity = result.unknowns[column].quantity
    match result.result:
        case results.OperatingPoint() as point:
            value = float(point.values[column])
            found = [(float(derivative.real), None) for derivative in gradient[:listed]]
        case results.Sweep() | results.Response() | results.Waveforms() as along:
            value = None
            axis = _axis(along)
            found = []
            for derivatives in gradient[:, :listed].T:
                largest = int(np.argmax(np.abs(derivatives)))
                peak = derivatives[largest]
                signed = float(abs(peak)) if np.iscomplexobj(peak) else float(peak)
                found.append((signed, float(axis[largest])))
    derivatives = tuple(
        Derivative(
            instance=parameter.instance,
            parameter=parameter.name,
            value=parameter.value,
            derivative=derivative,
            at=at,
        )
        for parameter, (derivative, at) in zip(
            result.parameters[:listed], found, strict=True
        )
    )
    return OutputSensitivity(
        output=output, quantity=quantity, value=value, derivatives=derivatives
    )


def _axis(result: results.Sweep | results.Response | results.Waveforms) -> results.Real:
    """The levels, frequencies or times of `result`."""
    match result:
        case results.Sweep():
            return result.levels
        case results.Response():
            return result.frequencies
        case results.Waveforms():
            return result.times


def _nets(result: results.Result) -> tuple[str, ...]:
    """The logic nets of `result`; small-signal results and sensitivities have none."""
    if isinstance(
        result,
        results.Response
        | results.Sensitivities
        | results.LoopGain
        | results.NoiseSpectrum
        | results.PolesZeros,
    ):
        return ()
    return result.nets


def _level(level: float) -> LogicLevel:
    """A logic level from its number: 0, 1, or None for NaN."""
    if np.isnan(level):
        return None
    return 1 if level else 0


def _written(level: LogicLevel) -> str:
    return "x" if level is None else str(level)


def _activity(net: str, levels: results.Real) -> Activity:
    """What the logic net `net` did over `levels`, its successive levels."""
    known = np.nan_to_num(levels, nan=-1.0)
    return Activity(
        net=net,
        first=_level(float(levels[0])),
        changes=int(np.count_nonzero(known[1:] != known[:-1])),
        last=_level(float(levels[-1])),
    )


def _activities(activities: tuple[Activity, ...]) -> list[str]:
    if not activities:
        return []
    header = ["", "first", "changes", "last"]
    rows = [
        [
            activity.net,
            _written(activity.first),
            str(activity.changes),
            _written(activity.last),
        ]
        for activity in activities
    ]
    return _table([header, *rows])


def _extents_of(
    unknowns: tuple[results.Unknown, ...], values: results.Real
) -> tuple[Extent, ...]:
    """The extent of each unknown, whose values are the matching column of `values`."""
    return tuple(
        Extent(
            unknown=unknown.name,
            quantity=unknown.quantity,
            first=float(values[0, column]),
            minimum=float(values[:, column].min()),
            maximum=float(values[:, column].max()),
            last=float(values[-1, column]),
        )
        for column, unknown in enumerate(unknowns)
    )


def _extents(extents: tuple[Extent, ...]) -> list[str]:
    if not extents:
        return []

    def si(value: float, extent: Extent) -> str:
        return format_si(value, _SYMBOLS[extent.quantity])

    header = ["", "first", "min", "max", "last"]
    rows = [
        [
            extent.unknown,
            si(extent.first, extent),
            si(extent.minimum, extent),
            si(extent.maximum, extent),
            si(extent.last, extent),
        ]
        for extent in extents
    ]
    return _table([header, *rows])


def _table(rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    return [
        "  "
        + "  ".join(
            cell.ljust(width) for cell, width in zip(row, widths, strict=True)
        ).rstrip()
        for row in rows
    ]


def _join(head: str, lines: list[str], unlisted: int) -> str:
    more = [f"  ... and {unlisted} more in the results file"] if unlisted else []
    return "\n".join([head, *lines, *more])
