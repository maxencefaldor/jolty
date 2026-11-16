# Copyright 2026 Maxence Faldor
"""Reports: what each analysis found, as data and as text for people and agents.

A report states facts: the value of every unknown at an operating point, and the first,
lowest, highest and last value of every unknown over a sweep, a frequency response or a
transient; the level of every logic net at an operating point, and its first and last
level and how often it changed over a sweep or a transient; and each transistor's region
of operation at an operating point, with its margins to the regions' boundaries, those
nearest one first. It lists the first unknowns and logic nets, and transistors, up to a
limit; the results file holds them all.
A corner's report states whether the circuit elaborates there and which analyses fail,
and a Monte Carlo analysis's, or a plan's variants', how many runs failed, what each
requirement read over the runs, as its spread, and the yield, with its Wilson interval.
"""

import math
from dataclasses import replace
from typing import Annotated, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from jolty import results, specs
from jolty.devices import LogicLevel, format_logic_level
from jolty.specs import Spread, Verdict, Yield
from jolty.units import format_si

type Nature = results.Nature

_SYMBOLS: dict[Nature, str] = {
    "voltage": "V",
    "current": "A",
    "temperature": "K",
    "state": "",
}


def _shown(value: float, nature: Nature) -> str:
    """`value`, an unknown's of `nature`, with an SI prefix to its unit.

    A device's state has no unit, and is written as a plain number.
    """
    if nature == "state":
        return f"{value:.4g}"
    return format_si(value, _SYMBOLS[nature])


_STRATEGIES = {
    "direct": "Newton iteration",
    "gmin_stepping": "gmin stepping",
    "junction_gmin_stepping": "junction gmin stepping",
    "source_stepping": "source stepping",
    "pseudo_transient": "pseudo-transient continuation",
}


def _found(strategy: str, iterations: int | None) -> str:
    """How an operating point was found, as text: by `strategy`, in `iterations`.

    Examples:
        >>> _found("gmin_stepping", 412)
        'found by gmin stepping in 412 iterations'
        >>> _found("ngspice", None)
        'found by ngspice'
    """
    by = f"found by {_STRATEGIES.get(strategy, strategy)}"
    if iterations is None:
        return by
    return f"{by} in {iterations} {'iteration' if iterations == 1 else 'iterations'}"


_ITERATIONS: Final = (
    "The Newton iterations that found it: the strategy's that found it, and those of "
    "every strategy tried before it; null for ngspice's, which it does not count."
)
"""What a report's iterations of an operating point are."""


class _Model(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", validate_by_name=True, serialize_by_alias=True
    )


class Value(_Model):
    """The value of one unknown, in volts or amperes."""

    unknown: str
    nature: Nature
    value: float


class Extent(_Model):
    """The values one unknown takes over an analysis, in volts or amperes."""

    unknown: str
    nature: Nature
    first: float
    minimum: float
    maximum: float
    last: float


class NetLevel(_Model):
    """The level of one logic net."""

    net: str
    level: LogicLevel


class Activity(_Model):
    """The levels one logic net takes over an analysis."""

    net: str
    first: LogicLevel
    changes: int = Field(description="How many times its level changed.")
    last: LogicLevel


class Margin(_Model):
    """A transistor's margin to a boundary of its regions of operation, in volts."""

    variable: str = Field(description="Its variable, such as saturation_margin.")
    value: float


class Transistor(_Model):
    """A transistor's region of operation and its margins to the regions' bounds."""

    path: str
    region: str = Field(description="Its region, as its model's equations say.")
    margins: tuple[Margin, ...]


class OperatingPointReport(_Model):
    """An operating point."""

    kind: Literal["operating_point"] = "operating_point"
    name: str
    strategy: str = Field(
        description="How the engine found it: direct, gmin_stepping, "
        "junction_gmin_stepping, source_stepping or pseudo_transient."
    )
    iterations: int | None = Field(default=None, description=_ITERATIONS)
    values: tuple[Value, ...]
    levels: tuple[NetLevel, ...] = ()
    unlisted: int = Field(description="Unknowns and logic nets left out of the list.")
    transistors: tuple[Transistor, ...] = Field(
        default=(),
        description="Each transistor's region, nearest a boundary first: by the least "
        "magnitude of its margins.",
    )
    unlisted_transistors: int = Field(
        default=0, description="Transistors left out of the list."
    )

    def describe(self) -> str:
        """The report as text."""
        rows = [
            [value.unknown, _shown(value.value, value.nature)] for value in self.values
        ]
        rows.extend(
            [level.net, format_logic_level(level.level)] for level in self.levels
        )
        head = f"{self.name}: operating point, {_found(self.strategy, self.iterations)}"
        text = _join(head, _table(rows), self.unlisted)
        if not self.transistors:
            return text
        widest = max(len(transistor.margins) for transistor in self.transistors)
        regions = [
            [
                transistor.path,
                transistor.region.replace("_", " "),
                *(
                    f"{margin.variable} {_shown(margin.value, 'voltage')}"
                    for margin in transistor.margins
                ),
                *[""] * (widest - len(transistor.margins)),
            ]
            for transistor in self.transistors
        ]
        head = "regions, nearest a boundary first:"
        listed = _join(head, _table(regions), self.unlisted_transistors)
        return f"{text}\n{_indented(listed)}"


class Solutions(_Model):
    """The value of one unknown at each DC solution, in volts or amperes."""

    unknown: str
    nature: Nature
    values: tuple[float, ...] = Field(description="Its value at each solution.")


class OperatingPointsReport(_Model):
    """Every DC solution a search found, the operating point first."""

    kind: Literal["operating_points"] = "operating_points"
    name: str
    strategies: tuple[str, ...] = Field(
        description="How the engine found each solution: direct, gmin_stepping, "
        "junction_gmin_stepping, source_stepping or pseudo_transient, on the equations "
        "deflated by those before it."
    )
    iterations: tuple[int | None, ...] = Field(
        default=(),
        description="The Newton iterations that found each solution, the search's "
        "on the equations deflated by those before it.",
    )
    values: tuple[Solutions, ...]
    unlisted: int = Field(description="Unknowns left out of the list.")

    def describe(self) -> str:
        """The report as text."""
        count = len(self.strategies)
        found = "solution" if count == 1 else "solutions"
        head = f"{self.name}: {count} DC {found}, the operating point first"
        if self.iterations:
            *each, last = (str(iterations) for iterations in self.iterations)
            spent = f"{', '.join(each)} and {last}" if each else last
            head += f", found in {spent} iterations"
        rows = [
            [
                solutions.unknown,
                *(_shown(value, solutions.nature) for value in solutions.values),
            ]
            for solutions in self.values
        ]
        return _join(head, _table(rows), self.unlisted)


class DcSweepReport(_Model):
    """A DC sweep."""

    kind: Literal["dc_sweep"] = "dc_sweep"
    name: str
    source: str = Field(description="The swept instance's path.")
    nature: Nature | None = Field(
        description="Whether its levels are voltages or currents; null for a "
        "parameter's."
    )
    parameter: str | None = Field(
        default=None,
        description="The parameter swept, such as 'resistance'; null for a source's "
        "level.",
    )
    first: float = Field(description="The first level, in SI base units.")
    last: float = Field(description="The last level, in SI base units.")
    swept: int = Field(description="Values the sweep solved.")
    iterations: int | None = Field(
        default=None,
        description="The Newton iterations that found the operating points at every "
        "level, each from the level before it; null for ngspice's, which it does not "
        "count.",
    )
    strategies: dict[str, int] = Field(
        default_factory=dict,
        description="The levels each strategy found the operating point at, by its "
        "name, in the order the sweep first used it; empty for ngspice's, which does "
        "not say.",
    )
    extents: tuple[Extent, ...]
    activities: tuple[Activity, ...] = ()
    unlisted: int = Field(description="Unknowns and logic nets left out of the list.")

    def describe(self) -> str:
        """The report as text."""
        if self.nature is None:
            swept = f"{self.source}'s {self.parameter}"
            first, last = f"{self.first:.4g}", f"{self.last:.4g}"
        else:
            unit = _SYMBOLS[self.nature]
            swept = self.source
            first, last = format_si(self.first, unit), format_si(self.last, unit)
        head = (
            f"{self.name}: DC sweep of {swept} from {first} to {last} "
            f"in {self.swept} levels"
        )
        if self.iterations is not None:
            head += f", {self.iterations} iterations"
        aided = [
            f"{levels} by {_STRATEGIES.get(strategy, strategy)}"
            for strategy, levels in self.strategies.items()
            if strategy != "direct"
        ]
        if aided:
            head += f", {', '.join(aided)}"
        lines = _extents(self.extents) + _activities(self.activities)
        return _join(head, lines, self.unlisted)


class AcReport(_Model):
    """A small-signal frequency response, as magnitudes per unit excitation."""

    kind: Literal["ac"] = "ac"
    name: str
    source: str = Field(description="The excited source's path.")
    start: float = Field(description="Lowest frequency, in hertz.")
    stop: float = Field(description="Highest frequency, in hertz.")
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
            f"{format_si(self.start, 'Hz')} to {format_si(self.stop, 'Hz')}, "
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
        description="Time steps rejected for their error or for Newton iteration that "
        "did not converge."
    )
    landed: int = Field(
        description="Time steps tried again shorter to land on an event within them: "
        "a detector's crossing, a logic driver's change or a device's jump."
    )
    extents: tuple[Extent, ...]
    activities: tuple[Activity, ...] = ()
    unlisted: int = Field(description="Unknowns and logic nets left out of the list.")

    def describe(self) -> str:
        """The report as text."""
        steps = _steps(self.accepted, self.rejected, self.landed)
        head = f"{self.name}: transient to {format_si(self.stop, 's')} {steps}"
        lines = _extents(self.extents) + _activities(self.activities)
        return _join(head, lines, self.unlisted)


class PeriodicSteadyStateReport(_Model):
    """One period of a periodic steady state."""

    kind: Literal["periodic_steady_state"] = "periodic_steady_state"
    name: str
    period: float = Field(
        description="The period in seconds: the analysis's, or an oscillator's, which "
        "it found."
    )
    periods: int = Field(
        description="The periods shooting simulated to find the steady state; the "
        "steps counted are the last's."
    )
    accepted: int = Field(description="Time steps the last period accepted.")
    rejected: int = Field(
        description="Time steps the last period rejected for their error or for "
        "Newton iteration that did not converge."
    )
    landed: int = Field(
        description="Time steps the last period tried again shorter to land on an "
        "event within them, a device's jump."
    )
    extents: tuple[Extent, ...]
    unlisted: int = Field(description="Unknowns left out of the list.")

    def describe(self) -> str:
        """The report as text."""
        steps = _steps(self.accepted, self.rejected, self.landed)
        head = (
            f"{self.name}: periodic steady state of period "
            f"{format_si(self.period, 's')}, {format_si(1 / self.period, 'Hz')}, "
            f"found in {self.periods} periods, the last {steps}"
        )
        return _join(head, _extents(self.extents), self.unlisted)


def _steps(accepted: int, rejected: int, landed: int) -> str:
    """How many time steps a transient or a period took: its steps' clause."""
    steps = f"in {accepted} steps, {rejected} rejected"
    if landed:
        steps += f", {landed} shortened to land on events"
    return steps


class StabilityReport(_Model):
    """The loop gain through a probe, and the margins it leaves."""

    kind: Literal["stability"] = "stability"
    name: str
    probe: str = Field(description="The probe's path.")
    start: float = Field(description="Lowest frequency, in hertz.")
    stop: float = Field(description="Highest frequency, in hertz.")
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
        lowest, highest = format_si(self.start, "Hz"), format_si(self.stop, "Hz")
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


class TransferFunctionReport(_Model):
    """The small-signal transfer function at DC: a gain and two resistances."""

    kind: Literal["transfer_function"] = "transfer_function"
    name: str
    source: str = Field(description="The input source's path.")
    output: str = Field(description="The output: a net or a voltage source's current.")
    reference: str | None = Field(
        default=None,
        description="The net the output voltage is taken over; null for the ground.",
    )
    gain: specs.ExtendedReal = Field(
        description="The output per volt or ampere of the source's level."
    )
    unit: str = Field(description="The gain's unit, such as 'V/V' or 'A/V'.")
    input_resistance: specs.ExtendedReal = Field(
        description="The resistance the source sees, in ohms."
    )
    output_resistance: specs.ExtendedReal = Field(
        description="The resistance the output sees with the source at its level, in "
        "ohms."
    )

    def describe(self) -> str:
        """The report as text."""
        over = "" if self.reference is None else f" over {self.reference}"
        return "\n".join(
            [
                (
                    f"{self.name}: transfer function from {self.source} to "
                    f"{self.output}{over}"
                ),
                f"  gain {self.gain:.6g} {self.unit}",
                f"  input resistance {format_si(self.input_resistance, 'Ω')}",
                f"  output resistance {format_si(self.output_resistance, 'Ω')}",
            ]
        )


class NoiseContribution(_Model):
    """What one noise source of a device adds to a noise analysis's output."""

    instance: str = Field(description="The device's path.")
    name: str = Field(
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
    nature: Nature = Field(description="What the output is.")
    start: float = Field(description="Lowest frequency, in hertz.")
    stop: float = Field(description="Highest frequency, in hertz.")
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
    referred_nature: Nature | None = Field(
        description="What the source's level is; null without a source."
    )
    contributions: tuple[NoiseContribution, ...] = Field(
        description="The noise sources that make the most noise, the most first."
    )
    unlisted: int = Field(description="Noise sources left out of the list.")

    def describe(self) -> str:
        """The report as text."""
        lowest, highest = format_si(self.start, "Hz"), format_si(self.stop, "Hz")
        over = "" if self.reference is None else f" over {self.reference}"
        unit = _SYMBOLS[self.nature]
        head = (
            f"{self.name}: noise at {self.output}{over} at {self.frequencies} "
            f"frequencies from {lowest} to {highest}"
        )
        density = f"{unit}/\N{SQUARE ROOT}Hz"
        first, last = format_si(self.first, density), format_si(self.last, density)
        lines = [head, f"  {first} at {lowest}, {last} at {highest}"]
        if self.rms is not None:
            total = f"  {format_si(self.rms, unit)} RMS over the frequencies"
            if self.referred is not None and self.referred_nature is not None:
                referred = format_si(self.referred, _SYMBOLS[self.referred_nature])
                total += f", {referred} RMS referred to {self.source}"
            lines.append(total)
            lines.extend(
                _table(
                    [
                        [
                            each.instance,
                            each.name,
                            f"{format_si(each.rms, unit)} RMS",
                            f"{each.share:.1%}",
                        ]
                        for each in self.contributions
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
    nature: Nature
    value: float | None = Field(
        default=None,
        description="Its value at the operating point, in volts or amperes; null "
        "along another analysis.",
    )
    derivatives: tuple[Derivative, ...] = Field(
        description="Its derivative with respect to the first parameters, in the "
        "circuit's order."
    )


type Differentiated = Literal[
    "operating_point", "dc_sweep", "ac", "transient", "periodic_steady_state"
]
"""The analysis a sensitivity or a mismatch analysis differentiates."""

_ALONG: Final[dict[Differentiated, str]] = {
    "dc_sweep": "along a DC sweep at {points} levels",
    "ac": "over a response at {points} frequencies",
    "transient": "over a transient at {points} times",
    "periodic_steady_state": "over a periodic steady state's period at {points} times",
}


class SensitivityReport(_Model):
    """Sensitivities at the operating point, or along a sweep, a response or time.

    Along a periodic steady state's period, too, and an oscillator's period's.
    """

    kind: Literal["sensitivity"] = "sensitivity"
    name: str
    of: Differentiated = Field(description="The analysis differentiated.")
    points: int = Field(description="The points the derivatives are taken at.")
    strategy: str | None = Field(
        default=None,
        description="How the engine found the operating point: direct, gmin_stepping, "
        "junction_gmin_stepping, source_stepping or pseudo_transient; null along a "
        "sweep or a transient.",
    )
    iterations: int | None = Field(default=None, description=_ITERATIONS)
    unit: str = Field(
        default="",
        description="The unit of the points: V or A along a sweep, Hz, or s.",
    )
    parameters: int = Field(
        description="Parameters the derivatives are with respect to."
    )
    outputs: tuple[OutputSensitivity, ...]
    period: float | None = Field(
        default=None,
        description="An oscillator's period, in seconds, whose derivatives "
        "period_derivatives lists; null for any other analysis.",
    )
    period_derivatives: tuple[Derivative, ...] = Field(
        default=(),
        description="An oscillator's period's derivative with respect to the first "
        "parameters, in the circuit's order.",
    )
    unlisted: int = Field(description="Parameters left out of each output's list.")

    def describe(self) -> str:
        """The report as text."""
        if self.of == "operating_point":
            found = _found(self.strategy or "", self.iterations)
            where = f"at the operating point, {found}"
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
                value = _shown(output.value, output.nature)
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
        if self.period is not None:
            lines.append(f"  period = {format_si(self.period, 's')}")
            rows = [
                [
                    each.instance,
                    each.parameter,
                    f"at {each.value!r}",
                    repr(each.derivative),
                ]
                for each in self.period_derivatives
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


class MismatchContribution(_Model):
    """What one random variable adds to an output's spread."""

    instance: str | None = Field(
        description="The path of the instance that draws it: the part whose tolerance "
        "it is, or the placement whose kit's statistical function it is; null for a "
        "lot's."
    )
    lot: str | None = Field(
        description="The lot whose shared spread it is; null for none."
    )
    name: str = Field(
        description="What it is within its instance or lot: the parameter a tolerance "
        "spreads, or which call of a statistical function it is, such as vth0#0."
    )
    contribution: float = Field(
        description="Its derivative times its standard deviation: how far the output "
        "moves, to first order, as it moves one standard deviation up, in the "
        "output's unit."
    )
    share: float = Field(description="Its fraction of the output's variance.")


class OutputMismatch(_Model):
    """How far an output, or a response's magnitude or phase, spreads, and why."""

    output: str
    nature: Nature
    part: results.Part | None = Field(
        default=None,
        description="What of a response's phasor spreads, its magnitude or its phase; "
        "null for real values.",
    )
    value: float | None = Field(
        default=None,
        description="Its value at the operating point, in volts or amperes; null "
        "along another analysis.",
    )
    standard_deviation: float = Field(
        description="Its standard deviation, to first order: at the operating point, "
        "or the largest over the points, a phase's in degrees."
    )
    at: float | None = Field(
        default=None,
        description="The level, frequency or time it is largest at, in SI base units; "
        "null at an operating point.",
    )
    contributions: tuple[MismatchContribution, ...] = Field(
        description="The random variables that contribute most there, the largest "
        "share first, a phase's contributions in degrees."
    )


class MismatchReport(_Model):
    """How far outputs spread with the random variables, to first order, and why."""

    kind: Literal["mismatch"] = "mismatch"
    name: str
    of: Differentiated = Field(description="The analysis differentiated.")
    points: int = Field(description="The points the spreads are taken at.")
    strategy: str | None = Field(
        default=None,
        description="How the engine found the operating point: direct, gmin_stepping, "
        "junction_gmin_stepping, source_stepping or pseudo_transient; null along a "
        "sweep or a transient.",
    )
    iterations: int | None = Field(default=None, description=_ITERATIONS)
    unit: str = Field(
        default="",
        description="The unit of the points: V or A along a sweep, Hz, or s.",
    )
    random_variables: int = Field(
        description="Random variables the outputs spread with."
    )
    outputs: tuple[OutputMismatch, ...]
    period: float | None = Field(
        default=None,
        description="An oscillator's period, in seconds; null for any other analysis.",
    )
    period_standard_deviation: float | None = Field(
        default=None,
        description="The standard deviation of an oscillator's period, to first "
        "order, in seconds; null for any other analysis.",
    )
    unlisted: int = Field(
        description="Random variables left out of each output's list."
    )

    def describe(self) -> str:
        """The report as text."""
        if self.of == "operating_point":
            found = _found(self.strategy or "", self.iterations)
            where = f"at the operating point, {found}"
        else:
            where = _ALONG[self.of].format(points=self.points)
        head = (
            f"{self.name}: standard deviations, to first order, with "
            f"{self.random_variables} random variables {where}"
        )
        lines = [head]
        for output in self.outputs:
            lines.append(f"  {self._spread(output)}")
            rows = [
                [
                    f"lot {each.lot}" if each.instance is None else each.instance,
                    each.name,
                    self._written(each.contribution, output),
                    f"{each.share:.1%}",
                ]
                for each in output.contributions
            ]
            lines.extend(f"  {row}" for row in _table(rows))
            if self.unlisted:
                lines.append(f"      ... and {self.unlisted} more in the results file")
        if self.period is not None and self.period_standard_deviation is not None:
            period = format_si(self.period, "s")
            deviation = format_si(self.period_standard_deviation, "s")
            lines.append(f"  period = {period}, standard deviation {deviation}")
        return "\n".join(lines)

    def _spread(self, output: OutputMismatch) -> str:
        """The line of `output`: what spreads, its value, and its standard deviation."""
        named = (
            output.output if output.part is None else f"{output.output}'s {output.part}"
        )
        deviation = self._written(output.standard_deviation, output)
        if output.value is not None:
            value = _shown(output.value, output.nature)
            return f"{named} = {value}, standard deviation {deviation}"
        at = "" if output.at is None else f" at {format_si(output.at, self.unit)}"
        return f"{named}: standard deviation largest, {deviation},{at}"

    @staticmethod
    def _written(value: float, output: OutputMismatch) -> str:
        """`value`, a spread of `output`, with its unit."""
        if output.part == "phase":
            return f"{value:.4g}°"
        return _shown(value, output.nature)


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
    process: str | None = Field(
        default=None,
        description="The corner's process corner of a foundry's devices, if it names "
        "one.",
    )
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
        if self.process is not None:
            head += f", process {self.process}"
        if self.problems:
            listed = "".join(f"\n{_indented(problem)}" for problem in self.problems)
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
    mismatched: int = Field(
        default=0,
        description="The foundry's devices each sample draws its kit's mismatch for.",
    )
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
    yield_: Yield | None = Field(
        default=None,
        alias="yield",
        description="How many samples meet every requirement; null without specs.",
    )

    def describe(self) -> str:
        """The report as text."""
        at = "".join(f", at {name}" for name in self.corners)
        mismatch = (
            f" and the mismatch of {self.mismatched} foundry's devices"
            if self.mismatched
            else ""
        )
        head = (
            f"monte carlo: {self.samples} samples of {self.toleranced} toleranced "
            f"parameters{mismatch}{at}, seed {self.seed}; {self.failed} failed"
        )
        lines = [head]
        lines.extend(f"  {spread.describe(self.samples)}" for spread in self.spreads)
        if self.yield_ is not None:
            lines.append(f"  {self.yield_.describe()}")
        return "\n".join(lines)


LARGE: Final = 10e9
"""The bytes past which the shards of a Monte Carlo analysis's samples are large: 10 GB,
past which a shard's report warns."""


class ShardReport(_Model):
    """What a shard of a plan's Monte Carlo analysis ran, and the file it wrote."""

    start: int = Field(description="The first sample the shard ran.")
    stop: int = Field(description="The sample the shard ran up to, and not.")
    samples: int = Field(description="The samples of the analysis, in every shard.")
    corners: tuple[str, ...] = Field(
        default=(),
        description="The corners each sample also ran at, after the plan's own "
        "conditions.",
    )
    failed: int = Field(
        description="Samples of the shard on which the circuit did not elaborate or "
        "an analysis failed, at the plan's own conditions or a corner."
    )
    size: int = Field(description="The bytes of the arrays the shard's file holds.")
    total: int = Field(
        description="The bytes of the arrays the shards of every sample hold: as many "
        "a sample as this shard's."
    )
    digest: str = Field(
        description="The SHA-256 hash of what the shard's numbers depend on, which "
        "every shard of one analysis shares."
    )

    def describe(self) -> str:
        """The report as text."""
        at = "".join(f", at {name}" for name in self.corners)
        head = (
            f"shard: samples {self.start} up to {self.stop} of {self.samples}{at}; "
            f"{self.failed} failed; {format_si(self.size, 'B')} of arrays, "
            f"digest {self.digest[:12]}"
        )
        if self.total <= LARGE:
            return head
        return (
            f"{head}\nwarning: the shards of every sample hold "
            f"{format_si(self.total, 'B')} of arrays, more than "
            f"{format_si(LARGE, 'B')}"
        )


class VariantsReport(_Model):
    """What the runs of a plan's variants found."""

    runs: int
    parameters: tuple[str, ...] = Field(
        default=(),
        description="The numbers the variants vary, such as 'R1 resistance' or "
        "'temperature'.",
    )
    corners: tuple[str, ...] = Field(
        default=(),
        description="The corners each run also ran at, after the plan's own "
        "conditions.",
    )
    failed: int = Field(
        description="Runs on which the circuit did not elaborate or an analysis "
        "failed, at the plan's own conditions or a corner."
    )
    spreads: tuple[Spread, ...] = Field(
        default=(), description="What each requirement of the specs read."
    )
    yield_: Yield | None = Field(
        default=None,
        alias="yield",
        description="How many runs meet every requirement; null without specs.",
    )

    def describe(self) -> str:
        """The report as text."""
        at = "".join(f", at {name}" for name in self.corners)
        varied = " and ".join(self.parameters) or "nothing"
        head = f"variants: {self.runs} runs of {varied}{at}; {self.failed} failed"
        lines = [head]
        lines.extend(f"  {spread.describe(self.runs)}" for spread in self.spreads)
        if self.yield_ is not None:
            lines.append(f"  {self.yield_.describe()}")
        return "\n".join(lines)


class WorstCaseReport(_Model):
    """What each requirement read at the extremes of the tolerances that worsen it."""

    step: float = Field(
        description="The fraction of its tolerance each parameter moved to find which "
        "way each requirement's value moves with it."
    )
    toleranced: int = Field(description="The toleranced parameters moved.")
    corners: tuple[str, ...] = Field(
        default=(),
        description="The corners the worst case also ran at, after the plan's own "
        "conditions.",
    )
    runs: int = Field(
        description="The simulations run: one per parameter and conditions to find "
        "which way each value moves, and one per distinct set of extremes."
    )
    verdicts: tuple[Verdict, ...] = Field(
        default=(),
        description="Each requirement's verdict, the worst over its bounds' extremes "
        "at the plan's own conditions and at each corner.",
    )

    @property
    def ok(self) -> bool:
        """Whether every requirement passes, if marginally, at its extremes."""
        return all(
            verdict.status in {"passed", "marginal"} for verdict in self.verdicts
        )

    def describe(self) -> str:
        """The report as text."""
        at = "".join(f", at {name}" for name in self.corners)
        head = (
            f"worst case: {self.toleranced} toleranced parameters, each moved "
            f"{self.step:.4g} of its tolerance{at}; {self.runs} runs"
        )
        lines = [head]
        for verdict in self.verdicts:
            lines.extend(f"  {line}" for line in verdict.describe().splitlines())
        return "\n".join(lines)


type AnalysisReport = Annotated[
    OperatingPointReport
    | OperatingPointsReport
    | DcSweepReport
    | AcReport
    | TransientReport
    | PeriodicSteadyStateReport
    | SensitivityReport
    | MismatchReport
    | StabilityReport
    | NoiseReport
    | PoleZeroReport
    | TransferFunctionReport
    | FailureReport,
    Field(discriminator="kind"),
]
"""The report of any analysis."""


def _waveforms_report(
    name: str,
    result: results.TransientResult,
    listed: tuple[tuple[Extent, ...], tuple[str, ...], int],
) -> TransientReport | PeriodicSteadyStateReport:
    """The report of `result`, a transient's or a period's waveforms.

    It lists `listed`: its unknowns' extents, its logic nets and how many it leaves out.
    """
    extents, nets, unlisted = listed
    statistics = result.statistics
    if isinstance(result, results.PeriodicSteadyStateResult):
        return PeriodicSteadyStateReport(
            name=name,
            period=result.period,
            periods=result.periods,
            accepted=statistics.accepted,
            rejected=statistics.rejected,
            landed=statistics.landed,
            extents=extents,
            unlisted=unlisted,
        )
    return TransientReport(
        name=name,
        stop=float(result.times[-1]),
        accepted=statistics.accepted,
        rejected=statistics.rejected,
        landed=statistics.landed,
        extents=extents,
        activities=tuple(
            _activity(net, result.logic_level(net).levels) for net in nets
        ),
        unlisted=unlisted,
    )


def report(
    name: str, result: results.AnalysisResult, max_unknowns: int
) -> AnalysisReport:
    """The report of the analysis `name`.

    It lists its first `max_unknowns` unknowns, then its first logic nets, up to
    `max_unknowns` in all.
    """
    unknowns = result.unknowns[:max_unknowns]
    nets = _nets(result)[: max_unknowns - len(unknowns)]
    unlisted = len(result.unknowns) + len(_nets(result)) - len(unknowns) - len(nets)
    match result:
        case results.OperatingPointResult():
            values = tuple(
                Value(unknown=unknown.name, nature=unknown.nature, value=float(value))
                for unknown, value in zip(
                    unknowns, result.values[:max_unknowns], strict=True
                )
            )
            levels = tuple(
                NetLevel(net=net, level=_level(float(level)))
                for net, level in zip(nets, result.logic, strict=False)
            )
            transistors = _transistors(result)
            return OperatingPointReport(
                name=name,
                strategy=result.strategy,
                iterations=result.iterations,
                values=values,
                levels=levels,
                unlisted=unlisted,
                transistors=transistors[:max_unknowns],
                unlisted_transistors=max(len(transistors) - max_unknowns, 0),
            )
        case results.DcSweepResult():
            return DcSweepReport(
                name=name,
                source=result.source,
                nature=result.nature,
                parameter=result.parameter,
                first=float(result.swept[0]),
                last=float(result.swept[-1]),
                swept=len(result.swept),
                iterations=sum(result.iterations) if result.iterations else None,
                strategies={
                    strategy: result.strategies.count(strategy)
                    for strategy in dict.fromkeys(result.strategies)
                },
                extents=_extents_of(unknowns, result.values[:, :max_unknowns]),
                activities=tuple(
                    _activity(net, result.logic_level(net)) for net in nets
                ),
                unlisted=unlisted,
            )
        case results.AcResult():
            return AcReport(
                name=name,
                source=result.source,
                start=float(result.frequencies[0]),
                stop=float(result.frequencies[-1]),
                frequencies=len(result.frequencies),
                magnitudes=_extents_of(
                    unknowns, np.abs(result.values[:, :max_unknowns])
                ),
                unlisted=unlisted,
            )
        case results.TransientResult():
            extents = _extents_of(unknowns, result.values[:, :max_unknowns])
            return _waveforms_report(name, result, (extents, nets, unlisted))
        case results.OperatingPointsResult():
            return OperatingPointsReport(
                name=name,
                strategies=tuple(point.strategy for point in result.points),
                iterations=tuple(point.iterations for point in result.points),
                values=tuple(
                    Solutions(
                        unknown=unknown.name,
                        nature=unknown.nature,
                        values=tuple(
                            float(point.values[index]) for point in result.points
                        ),
                    )
                    for index, unknown in enumerate(unknowns)
                ),
                unlisted=len(result.unknowns) - len(unknowns),
            )
        case _:
            return _linear(name, result, max_unknowns)


def _transistors(point: results.OperatingPointResult) -> tuple[Transistor, ...]:
    """The region and margins of each transistor `point` recorded.

    Those nearest a boundary come first, by the least magnitude of their margins.
    """
    found = [
        Transistor(
            path=device.path,
            region=point.region(device.path),
            margins=tuple(
                Margin(variable=margin, value=point.variable(device.path, margin))
                for margin in device.margins
            ),
        )
        for device in point.recording.devices
        if device.regions
    ]
    return tuple(
        sorted(
            found,
            key=lambda transistor: min(
                (abs(margin.value) for margin in transistor.margins), default=math.inf
            ),
        )
    )


def _linear(
    name: str,
    result: results.SensitivityResult
    | results.MismatchResult
    | results.StabilityResult
    | results.NoiseResult
    | results.PoleZeroResult
    | results.TransferFunctionResult,
    max_unknowns: int,
) -> AnalysisReport:
    """The report of a small-signal analysis or of sensitivities."""
    match result:
        case results.TransferFunctionResult():
            return _transfer(name, result)
        case results.SensitivityResult():
            return _sensitivities(name, result, max_unknowns)
        case results.MismatchResult():
            return _mismatch(name, result, max_unknowns)
        case results.StabilityResult():
            return _stability(name, result)
        case results.NoiseResult():
            return _noise(name, result, max_unknowns)
        case results.PoleZeroResult():
            return _poles_zeros(name, result, max_unknowns)


def _poles_zeros(
    name: str, result: results.PoleZeroResult, max_unknowns: int
) -> PoleZeroReport:
    """The report of the pole-zero analysis `name`, its first `max_unknowns` roots."""
    poles = _roots(result.poles)
    zeros = None if result.zeros is None else _roots(result.zeros)
    every = len(poles) + len(zeros or ())
    shown_poles = poles[:max_unknowns]
    shown_zeros = None if zeros is None else zeros[: max_unknowns - len(shown_poles)]
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


def _noise(name: str, result: results.NoiseResult, max_unknowns: int) -> NoiseReport:
    """The report of the noise analysis `name`, its `max_unknowns` noisiest sources."""
    order = np.argsort(result.frequencies)
    frequencies = result.frequencies[order]
    banded = len(frequencies) > 1
    powers = [
        results.integrate(frequencies, row[order]) if banded else float(row[order[0]])
        for row in result.contributions
    ]
    total = math.fsum(powers)
    noisiest = sorted(range(len(powers)), key=lambda index: -powers[index])[
        :max_unknowns
    ]
    contributions = tuple(
        NoiseContribution(
            instance=result.sources[index].instance,
            name=result.sources[index].name,
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
        nature=result.nature,
        start=float(frequencies[0]),
        stop=float(frequencies[-1]),
        frequencies=len(frequencies),
        first=math.sqrt(float(result.densities[order[0]])),
        last=math.sqrt(float(result.densities[order[-1]])),
        rms=result.rms() if banded else None,
        source=None if noise_input is None else noise_input.source,
        referred=result.rms(referred=True)
        if banded and noise_input is not None
        else None,
        referred_nature=None if noise_input is None else noise_input.nature,
        contributions=contributions,
        unlisted=len(result.sources) - len(contributions),
    )


def _transfer(
    name: str, result: results.TransferFunctionResult
) -> TransferFunctionReport:
    """The report of the transfer function analysis `name`."""
    output = next(
        unknown.nature for unknown in result.unknowns if unknown.name == result.output
    )
    return TransferFunctionReport(
        name=name,
        source=result.source,
        output=result.output,
        reference=result.reference,
        gain=result.gain,
        unit=f"{_SYMBOLS[output]}/{_SYMBOLS[result.input]}",
        input_resistance=result.input_resistance,
        output_resistance=result.output_resistance,
    )


def _stability(name: str, result: results.StabilityResult) -> StabilityReport:
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
        start=float(result.frequencies[order[0]]),
        stop=float(result.frequencies[order[-1]]),
        frequencies=len(result.frequencies),
        gain=float(20 * np.log10(np.abs(lowest))),
        phase=float(np.degrees(np.angle(lowest))),
        crossover=_reading(specs.crossover(name), read),
        phase_margin=_reading(specs.phase_margin(name), read),
        gain_margin=_reading(specs.gain_margin(name), read),
    )


def _reading(measure: specs.Measure, result: results.AnalysisResult) -> float | None:
    """What `measure` reads of `result`, or None when it cannot read it."""
    try:
        return measure.read(result).value
    except specs.MeasureError:
        return None


def _sensitivities(
    name: str, result: results.SensitivityResult, max_unknowns: int
) -> SensitivityReport:
    """The report of the sensitivity `name`, its first `max_unknowns` parameters."""
    listed_parameters = min(max_unknowns, len(result.parameters))
    of, points, unit, point = _differentiated(result.result)
    period = None
    period_derivatives: tuple[Derivative, ...] = ()
    if result.period_derivatives is not None and isinstance(
        result.result, results.PeriodicSteadyStateResult
    ):
        period = result.result.period
        period_derivatives = tuple(
            Derivative(
                instance=parameter.instance,
                parameter=parameter.name,
                value=parameter.value,
                derivative=float(derivative),
            )
            for parameter, derivative in zip(
                result.parameters[:listed_parameters],
                result.period_derivatives,
                strict=False,
            )
        )
    return SensitivityReport(
        name=name,
        of=of,
        points=points,
        strategy=None if point is None else point.strategy,
        iterations=None if point is None else point.iterations,
        unit=unit,
        parameters=len(result.parameters),
        outputs=tuple(
            _output(result, output, listed_parameters) for output in result.outputs
        ),
        period=period,
        period_derivatives=period_derivatives,
        unlisted=len(result.parameters) - listed_parameters,
    )


def _output(
    result: results.SensitivityResult, output: str, max_unknowns: int
) -> OutputSensitivity:
    """The value of `output` and its derivatives by the first `max_unknowns` parameters.

    Along an analysis other than the operating point, each is where it is largest.
    """
    gradient = result.gradient(output)
    column = results.column(result.unknowns, output)
    nature = result.unknowns[column].nature
    match result.result:
        case results.OperatingPointResult() as point:
            value = float(point.values[column])
            found = [
                (float(derivative.real), None) for derivative in gradient[:max_unknowns]
            ]
        case (
            results.DcSweepResult()
            | results.AcResult()
            | results.TransientResult() as along
        ):
            value = None
            axis = _axis(along)
            found = []
            for derivatives in gradient[:, :max_unknowns].T:
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
            result.parameters[:max_unknowns], found, strict=True
        )
    )
    return OutputSensitivity(
        output=output, nature=nature, value=value, derivatives=derivatives
    )


def _differentiated(
    result: results.OperatingPointResult
    | results.DcSweepResult
    | results.AcResult
    | results.TransientResult,
) -> tuple[Differentiated, int, str, results.OperatingPointResult | None]:
    """What a sensitivity or a mismatch analysis differentiated, as `result` says.

    That is the analysis, its points, their unit, and its operating point, if it solved
    one.
    """
    match result:
        case results.OperatingPointResult():
            return "operating_point", 1, "", result
        case results.DcSweepResult():
            unit = "" if result.nature is None else _SYMBOLS[result.nature]
            return "dc_sweep", len(result.swept), unit, None
        case results.AcResult():
            return "ac", len(result.frequencies), "Hz", result.operating_point
        case results.PeriodicSteadyStateResult():
            return "periodic_steady_state", len(result.times), "s", None
        case results.TransientResult():
            return "transient", len(result.times), "s", None


def _mismatch(
    name: str, result: results.MismatchResult, max_unknowns: int
) -> MismatchReport:
    """The report of the mismatch analysis `name`.

    It lists each output's `max_unknowns` largest contributions.
    """
    of, points, unit, point = _differentiated(result.result)
    parts: tuple[results.Part | None, ...] = (
        ("magnitude", "phase") if of == "ac" else (None,)
    )
    listed = min(max_unknowns, len(result.random_variables))
    period, period_deviation = None, None
    if result.period_derivatives is not None and isinstance(
        result.result, results.PeriodicSteadyStateResult
    ):
        period = result.result.period
        period_deviation = result.period_standard_deviation()
    return MismatchReport(
        name=name,
        of=of,
        points=points,
        strategy=None if point is None else point.strategy,
        iterations=None if point is None else point.iterations,
        unit=unit,
        random_variables=len(result.random_variables),
        outputs=tuple(
            _output_mismatch(result, output, part, listed)
            for output in result.outputs
            for part in parts
        ),
        period=period,
        period_standard_deviation=period_deviation,
        unlisted=len(result.random_variables) - listed,
    )


def _output_mismatch(
    result: results.MismatchResult,
    output: str,
    part: results.Part | None,
    listed: int,
) -> OutputMismatch:
    """How far `output`, or `part` of its phasor, spreads, and why.

    It lists its `listed` largest contributions, where it spreads most along an analysis
    other than the operating point.
    """
    column = results.column(result.unknowns, output)
    nature = result.unknowns[column].nature
    contributions = np.atleast_2d(result.contributions(output, part))
    deviations = np.sqrt(np.sum(contributions**2, axis=-1))
    largest = int(np.argmax(deviations))
    if part == "phase":
        contributions = np.degrees(contributions)
        deviations = np.degrees(deviations)
    shares = np.atleast_2d(result.shares(output, part))[largest]
    order = np.argsort(-shares, kind="stable")[:listed]
    value, at = None, None
    match result.result:
        case results.OperatingPointResult() as found:
            value = float(found.values[column])
        case results.DcSweepResult() | results.AcResult() | results.TransientResult():
            at = float(_axis(result.result)[largest])
    return OutputMismatch(
        output=output,
        nature=nature,
        part=part,
        value=value,
        standard_deviation=float(deviations[largest]),
        at=at,
        contributions=tuple(
            MismatchContribution(
                instance=result.random_variables[index].instance,
                lot=result.random_variables[index].lot,
                name=result.random_variables[index].name,
                contribution=float(contributions[largest, index]),
                share=float(shares[index]),
            )
            for index in order
        ),
    )


def _axis(
    result: results.DcSweepResult | results.AcResult | results.TransientResult,
) -> results.Real:
    """The levels, frequencies or times of `result`."""
    match result:
        case results.DcSweepResult():
            return result.swept
        case results.AcResult():
            return result.frequencies
        case results.TransientResult():
            return result.times


def _nets(result: results.AnalysisResult) -> tuple[str, ...]:
    """The logic nets of `result`; small-signal results and sensitivities have none."""
    if isinstance(
        result,
        results.AcResult
        | results.SensitivityResult
        | results.MismatchResult
        | results.StabilityResult
        | results.NoiseResult
        | results.PoleZeroResult
        | results.TransferFunctionResult
        | results.OperatingPointsResult,
    ):
        return ()
    return result.nets


def _level(level: float) -> LogicLevel:
    """A logic level from its number: 0, 1, or None for NaN."""
    if np.isnan(level):
        return None
    return 1 if level else 0


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
            format_logic_level(activity.first),
            str(activity.changes),
            format_logic_level(activity.last),
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
            nature=unknown.nature,
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
        return _shown(value, extent.nature)

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


def _indented(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines())


def _join(head: str, lines: list[str], unlisted: int) -> str:
    more = [f"  ... and {unlisted} more in the results file"] if unlisted else []
    return "\n".join([head, *lines, *more])
