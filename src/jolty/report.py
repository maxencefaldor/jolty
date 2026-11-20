# Copyright 2026 Maxence Faldor
"""Structured and text reports summarizing simulation analysis results.

An analysis report presents key facts for people and agents. Operating-point reports
record unknown values, logic net levels, and transistor operating regions sorted by
nearest boundary margin. Sweep, AC, and transient reports summarize the first,
minimum, maximum, and final values of unknowns along with logic net transition counts.
Each report truncates listed unknowns, logic nets, and transistors to a configured
limit, while the results archive retains the complete data.

Corner reports record elaboration problems and failed analyses at each corner. Monte
Carlo and variant batch reports summarize failed run counts, requirement spreads, and
overall yield with its Wilson confidence interval.
"""

__lazy_modules__ = ["jolty.devices", "numpy"]

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Annotated, Final, Literal

import numpy as np
from pydantic import ConfigDict, Field

from jolty import ir, results, specs, unknowns
from jolty.devices import format_logic_level
from jolty.specs import Spread, Verdict, Yield
from jolty.units import format_si
from jolty.unknowns import LogicLevel


def _shown(value: float, nature: unknowns.Nature | None) -> str:
    """Formats an unknown `value` of `nature` with an SI unit prefix.

    Device state unknowns and non-unknown outputs (such as stability loop gain and
    noise spectral densities) are formatted as plain numbers.
    """
    if nature is None or nature == "state":
        return f"{value:.4g}"
    return format_si(value, unknowns.SYMBOLS[nature])


_STRATEGIES: Final = {
    "direct": "Newton iteration",
    "gmin_stepping": "gmin stepping",
    "junction_gmin_stepping": "junction gmin stepping",
    "source_stepping": "source stepping",
    "pseudo_transient": "pseudo-transient continuation",
}


def _found(strategy: str, iterations: int | None) -> str:
    """Formats operating-point `strategy` and optional `iterations` count as text.

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
    "Total Newton iterations across the converging strategy and all earlier "
    "fallback strategies, or null for ngspice runs that do not report counts."
)
"""Field description for operating-point Newton iteration counts."""


class Value(ir.FrozenModel):
    """Scalar value of a single unknown, in SI base units."""

    unknown: str
    nature: unknowns.Nature
    value: float


class Extent(ir.FrozenModel):
    """Summary of values taken by a single unknown across an analysis."""

    unknown: str
    nature: unknowns.Nature
    first: float
    minimum: float
    maximum: float
    last: float


class NetLevel(ir.FrozenModel):
    """Logic level of a single digital net."""

    net: str
    level: LogicLevel


class Activity(ir.FrozenModel):
    """Summary of logic levels taken by a digital net across an analysis."""

    net: str
    first: LogicLevel
    changes: int = Field(description="Number of times the logic net level changed.")
    last: LogicLevel


class Margin(ir.FrozenModel):
    """Transistor margin to an operating-region boundary, in volts."""

    variable: str = Field(
        description="Margin variable name, such as saturation_margin."
    )
    value: float


class Transistor(ir.FrozenModel):
    """Transistor operating region and margins to region boundaries."""

    path: str
    region: str = Field(
        description="Operating region determined by the device model equations."
    )
    margins: tuple[Margin, ...]


class OperatingPointReport(ir.FrozenModel):
    """Report summarizing a DC operating-point analysis."""

    kind: Literal["operating_point"] = "operating_point"
    name: str
    strategy: str = Field(
        description="Solver strategy that found the operating point: direct, "
        "gmin_stepping, junction_gmin_stepping, source_stepping, or pseudo_transient."
    )
    iterations: int | None = Field(default=None, description=_ITERATIONS)
    values: tuple[Value, ...]
    levels: tuple[NetLevel, ...] = ()
    unlisted: int = Field(
        description="Number of unknowns and logic nets omitted from the report."
    )
    transistors: tuple[Transistor, ...] = Field(
        default=(),
        description="Recorded transistors ordered by smallest absolute boundary "
        "margin first.",
    )
    unlisted_transistors: int = Field(
        default=0, description="Number of recorded transistors omitted from the report."
    )

    def describe(self) -> str:
        """Formats the operating-point report as human-readable text.

        Returns:
            Multi-line human-readable operating-point summary.
        """
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
        listed = _join(
            "regions, nearest a boundary first:",
            _table(regions),
            self.unlisted_transistors,
        )
        return f"{text}\n{_indented(listed)}"


class Solutions(ir.FrozenModel):
    """Values of a single unknown across multiple DC solutions, in SI base units."""

    unknown: str
    nature: unknowns.Nature
    values: tuple[float, ...] = Field(
        description="Value of the unknown at each DC solution."
    )


class OperatingPointsReport(ir.FrozenModel):
    """Report listing every DC solution found by a search, operating point first."""

    kind: Literal["operating_points"] = "operating_points"
    name: str
    strategies: tuple[str, ...] = Field(
        description="Solver strategy that found each solution on the deflated "
        "equations: direct, gmin_stepping, junction_gmin_stepping, source_stepping, "
        "or pseudo_transient."
    )
    iterations: tuple[int | None, ...] = Field(
        default=(),
        description="Newton iteration counts for each solution found on the deflated "
        "equations.",
    )
    values: tuple[Solutions, ...]
    unlisted: int = Field(description="Number of unknowns omitted from the report.")

    def describe(self) -> str:
        """Formats the multi-solution DC report as human-readable text.

        Returns:
            Multi-line human-readable summary across all solved DC points.
        """
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


class DcSweepReport(ir.FrozenModel):
    """Report summarizing a DC sweep analysis."""

    kind: Literal["dc_sweep"] = "dc_sweep"
    name: str
    source: str = Field(description="Instance path of the swept source or device.")
    nature: unknowns.Nature | None = Field(
        description="Physical quantity of swept values (voltage or current), or null "
        "for parameter sweeps."
    )
    parameter: str | None = Field(
        default=None,
        description="Swept parameter name, such as 'resistance', or null when sweeping "
        "a source value.",
    )
    first: float = Field(description="Initial swept value, in SI base units.")
    last: float = Field(description="Final swept value, in SI base units.")
    swept: int = Field(description="Number of sweep points solved.")
    iterations: int | None = Field(
        default=None,
        description="Total Newton iterations across all sweep points using "
        "continuation, or null for ngspice runs that do not report counts.",
    )
    strategies: dict[str, int] = Field(
        default_factory=dict,
        description="Count of sweep points solved by each strategy in order of first "
        "use, or empty for ngspice runs.",
    )
    extents: tuple[Extent, ...]
    activities: tuple[Activity, ...] = ()
    unlisted: int = Field(
        description="Number of unknowns and logic nets omitted from the report."
    )

    def describe(self) -> str:
        """Formats the DC sweep report as human-readable text.

        Returns:
            Multi-line human-readable DC sweep summary.
        """
        if self.nature is None:
            swept = f"{self.source}'s {self.parameter}"
            first, last = f"{self.first:.4g}", f"{self.last:.4g}"
        else:
            unit = unknowns.SYMBOLS[self.nature]
            swept = self.source
            first, last = format_si(self.first, unit), format_si(self.last, unit)
        head = (
            f"{self.name}: DC sweep of {swept} from {first} to {last} "
            f"in {self.swept} values"
        )
        if self.iterations is not None:
            head += f", {self.iterations} iterations"
        aided = [
            f"{values} by {_STRATEGIES.get(strategy, strategy)}"
            for strategy, values in self.strategies.items()
            if strategy != "direct"
        ]
        if aided:
            head += f", {', '.join(aided)}"
        return _join(
            head, _extents(self.extents) + _activities(self.activities), self.unlisted
        )


class _FrequencyReport(ir.FrozenModel):
    """Shared frequency-span fields for frequency-domain analysis reports."""

    name: str
    start: float = Field(description="Lowest frequency, in hertz.")
    stop: float = Field(description="Highest frequency, in hertz.")
    frequencies: int = Field(
        description="Number of frequencies solved by the analysis."
    )


class AcReport(_FrequencyReport):
    """Report summarizing a small-signal AC frequency response."""

    kind: Literal["ac"] = "ac"
    source: str = Field(description="Instance path of the excited AC source.")
    magnitudes: tuple[Extent, ...] = Field(
        description="Summary of phasor magnitudes for each unknown from the lowest "
        "frequency to the highest."
    )
    unlisted: int = Field(description="Number of unknowns omitted from the report.")

    def describe(self) -> str:
        """Formats the AC frequency-response report as human-readable text.

        Returns:
            Multi-line human-readable AC frequency-response summary.
        """
        head = (
            f"{self.name}: response to {self.source} at {self.frequencies} "
            "frequencies from "
            f"{format_si(self.start, 'Hz')} to {format_si(self.stop, 'Hz')}, "
            "magnitudes per unit excitation"
        )
        return _join(head, _extents(self.magnitudes), self.unlisted)


class NetChargeLeak(ir.FrozenModel):
    """Electrical net with the largest integrated KCL residual over a run."""

    net: str = Field(
        description="Electrical net with the largest cumulative absolute charge leak."
    )
    signed: float = Field(
        description="Signed net charge leaked into the net, in coulombs."
    )
    absolute: float = Field(
        description="Cumulative absolute charge leaked across all steps at the net, "
        "in coulombs, bounding intermediate charge drift."
    )


class _WaveformReport(ir.FrozenModel):
    """Shared fields and rendering helpers for time-domain waveform reports."""

    name: str
    accepted: int = Field(description="Number of accepted time steps.")
    rejected: int = Field(
        description="Number of time steps rejected due to truncation error or "
        "non-convergent Newton iterations."
    )
    landed: int = Field(
        description="Number of time steps shortened to land on detector crossings, "
        "logic transitions, or device discontinuities."
    )
    extents: tuple[Extent, ...]
    unlisted: int = Field(
        description="Number of unknowns and logic nets omitted from the report."
    )
    charge_leak: NetChargeLeak | None = Field(
        default=None,
        description="Electrical net with the largest nonzero cumulative absolute "
        "charge leak, or null if zero, absent, or unreported.",
    )

    def _steps_text(self) -> str:
        steps = f"in {self.accepted} steps, {self.rejected} rejected"
        if self.landed:
            steps += f", {self.landed} shortened to land on events"
        return steps

    def _format_waveforms(self, head: str, lines: list[str]) -> str:
        text = _join(head, lines, self.unlisted)
        if self.charge_leak is None:
            return text
        signed = format_si(self.charge_leak.signed, "C")
        absolute = format_si(self.charge_leak.absolute, "C")
        return (
            f"{text}\n  largest charge leak: {self.charge_leak.net}  "
            f"{signed} signed, {absolute} absolute"
        )


class TransientReport(_WaveformReport):
    """Report summarizing a transient time-domain analysis."""

    kind: Literal["transient"] = "transient"
    stop: float = Field(
        description="Final simulation time of the transient analysis, in seconds."
    )
    activities: tuple[Activity, ...] = ()

    def describe(self) -> str:
        """Formats the transient analysis report as human-readable text.

        Returns:
            Multi-line human-readable transient waveform summary.
        """
        head = (
            f"{self.name}: transient to {format_si(self.stop, 's')} "
            f"{self._steps_text()}"
        )
        return self._format_waveforms(
            head, _extents(self.extents) + _activities(self.activities)
        )


class PeriodicSteadyStateReport(_WaveformReport):
    """Report summarizing one period of a periodic steady-state analysis."""

    kind: Literal["periodic_steady_state"] = "periodic_steady_state"
    period: float = Field(
        description="Steady-state period in seconds, either fixed by the analysis or "
        "solved for an oscillator."
    )
    periods: int = Field(
        description="Number of shooting periods simulated to reach steady state; "
        "reported step counts cover the final period."
    )

    def describe(self) -> str:
        """Formats the periodic steady-state report as human-readable text.

        Returns:
            Multi-line human-readable periodic steady-state summary.
        """
        head = (
            f"{self.name}: periodic steady state of period "
            f"{format_si(self.period, 's')}, {format_si(1 / self.period, 'Hz')}, "
            f"found in {self.periods} periods, the last {self._steps_text()}"
        )
        return self._format_waveforms(head, _extents(self.extents))


class StabilityReport(_FrequencyReport):
    """Report summarizing loop gain and stability margins through a probe."""

    kind: Literal["stability"] = "stability"
    probe: str = Field(description="Instance path of the stability probe.")
    gain: float = Field(
        description="Loop gain magnitude at the lowest frequency, in dB."
    )
    phase: float = Field(
        description="Loop gain phase at the lowest frequency, in degrees."
    )
    crossover: float | None = Field(
        description="Frequency in hertz where loop gain first crosses 0 dB downward, "
        "or null if no crossover occurs over positive frequencies."
    )
    phase_margin: float | None = Field(
        description="Phase margin in degrees, equal to 180° plus the loop gain phase "
        "at the unity-gain crossover."
    )
    gain_margin: float | None = Field(
        description="Gain margin in dB, equal to the negated loop gain where phase "
        "first crosses -180°, or null if no crossing occurs over positive frequencies."
    )

    def describe(self) -> str:
        """Formats the stability analysis report as human-readable text.

        Returns:
            Multi-line human-readable loop-gain and stability-margin summary.
        """
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


class TransferFunctionReport(ir.FrozenModel):
    """Report summarizing a DC small-signal transfer function analysis."""

    kind: Literal["transfer_function"] = "transfer_function"
    name: str
    source: str = Field(description="Instance path of the input source.")
    output: str = Field(
        description="Output unknown name, either a net voltage or a voltage-source "
        "current."
    )
    reference: str | None = Field(
        default=None,
        description="Reference net for a differential output voltage, or null for "
        "ground.",
    )
    gain: specs.ExtendedReal = Field(
        description="Small-signal output change per volt or ampere of input source "
        "level."
    )
    unit: str = Field(
        description="Unit string for the small-signal gain, such as 'V/V' or 'A/V'."
    )
    input_resistance: specs.ExtendedReal = Field(
        description="Small-signal input resistance seen by the input source, in ohms."
    )
    output_resistance: specs.ExtendedReal = Field(
        description="Small-signal output resistance seen at the output port, in ohms."
    )

    def describe(self) -> str:
        """Formats the transfer-function report as human-readable text.

        Returns:
            Multi-line human-readable small-signal transfer-function summary.
        """
        over = "" if self.reference is None else f" over {self.reference}"
        header = (
            f"{self.name}: transfer function from {self.source} to {self.output}{over}"
        )
        return "\n".join(
            [
                header,
                f"  gain {self.gain:.6g} {self.unit}",
                f"  input resistance {format_si(self.input_resistance, 'Ω')}",
                f"  output resistance {format_si(self.output_resistance, 'Ω')}",
            ]
        )


class NoiseContribution(ir.FrozenModel):
    """Contribution of a single device noise source to output noise."""

    instance: str = Field(description="Instance path of the contributing device.")
    name: str = Field(
        description="Noise mechanism name, such as thermal or collector shot."
    )
    rms: float = Field(
        description="Integrated RMS output noise from the source across the band, in "
        "volts or amperes."
    )
    share: float = Field(
        description="Fraction of total output noise power contributed by the source "
        "across the band."
    )


class NoiseReport(_FrequencyReport):
    """Report summarizing output noise and the dominant device noise sources."""

    kind: Literal["noise"] = "noise"
    output: str = Field(description="Name of the output unknown.")
    reference: str | None = Field(
        description="Reference net for a differential output voltage, or null for "
        "ground."
    )
    nature: unknowns.Nature = Field(
        description="Physical quantity of the output unknown."
    )
    first: float = Field(
        description="Output noise spectral density at the lowest frequency, in V/√Hz "
        "or A/√Hz."
    )
    last: float = Field(
        description="Output noise spectral density at the highest frequency, in V/√Hz "
        "or A/√Hz."
    )
    rms: float | None = Field(
        description="Integrated RMS output noise across the frequency band, in volts "
        "or amperes, or null for a single frequency."
    )
    source: str | None = Field(
        description="Instance path of the input-referred source, or null if omitted."
    )
    referred: float | None = Field(
        description="Integrated input-referred RMS noise across the band, or null "
        "without an input source or for a single frequency."
    )
    referred_nature: unknowns.Nature | None = Field(
        description="Physical quantity of the input-referred source, or null if "
        "omitted."
    )
    contributions: tuple[NoiseContribution, ...] = Field(
        description="Top contributing noise sources ordered from largest to smallest "
        "noise power."
    )
    unlisted: int = Field(
        description="Number of noise sources omitted from the report."
    )

    def describe(self) -> str:
        """Formats the noise analysis report as human-readable text.

        Returns:
            Multi-line human-readable output-noise and contributor summary.
        """
        lowest, highest = format_si(self.start, "Hz"), format_si(self.stop, "Hz")
        over = "" if self.reference is None else f" over {self.reference}"
        unit = unknowns.SYMBOLS[self.nature]
        head = (
            f"{self.name}: noise at {self.output}{over} at {self.frequencies} "
            f"frequencies from {lowest} to {highest}"
        )
        density = f"{unit}/\N{SQUARE ROOT}Hz"
        first, last = format_si(self.first, density), format_si(self.last, density)
        lines = [f"  {first} at {lowest}, {last} at {highest}"]
        if self.rms is not None:
            total = f"  {format_si(self.rms, unit)} RMS over the frequencies"
            if self.referred is not None and self.referred_nature is not None:
                referred = format_si(
                    self.referred, unknowns.SYMBOLS[self.referred_nature]
                )
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
        return _join(head, lines, self.unlisted)


class Root(ir.FrozenModel):
    """Single pole or zero, or a complex-conjugate pair represented once."""

    real: float = Field(description="Real part of the root, in radians per second.")
    imaginary: float = Field(
        description="Imaginary part in radians per second: zero for a real root, or "
        "positive for a complex-conjugate pair."
    )
    frequency: float = Field(
        description="Natural frequency in hertz, equal to the root magnitude over 2π."
    )
    damping: float | None = Field(
        description="Damping ratio equal to the negated real part over magnitude: 1 "
        "in the left half-plane, negative in the right half-plane, or null at origin."
    )

    def written(self, kind: str) -> list[str]:
        """Formats the root as a table row of kind, value, frequency, and damping.

        Args:
            kind: Root label such as ``"pole"`` or ``"zero"``.

        Returns:
            Four-cell row listing root kind, s-plane value, frequency, and damping.
        """
        value = format_si(self.real, "rad/s")
        if self.imaginary:
            value += f" ± j{format_si(self.imaginary, 'rad/s')}"
        damping = "" if self.damping is None else f"damping {self.damping:.4g}"
        return [kind, value, format_si(self.frequency, "Hz"), damping]


class PoleZeroReport(ir.FrozenModel):
    """Report listing poles of a linearized circuit and transfer-function zeros."""

    kind: Literal["pole_zero"] = "pole_zero"
    name: str
    source: str | None = Field(
        description="Instance path of the transfer-function input source, or null."
    )
    output: str | None = Field(
        description="Name of the transfer-function output unknown, or null."
    )
    reference: str | None = Field(
        description="Reference net for a differential output voltage, or null for "
        "ground."
    )
    poles: tuple[Root, ...] = Field(
        description="Listed poles ordered by increasing magnitude, with each "
        "complex-conjugate pair shown once."
    )
    zeros: tuple[Root, ...] | None = Field(
        description="Listed zeros ordered by increasing magnitude when both source "
        "and output are given, or null."
    )
    unlisted: int = Field(
        description="Number of poles and zeros omitted from the report."
    )

    def describe(self) -> str:
        """Formats the pole-zero analysis report as human-readable text.

        Returns:
            Multi-line human-readable pole and zero summary.
        """
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


class Derivative(ir.FrozenModel):
    """First-order derivative of an output with respect to a single parameter."""

    instance: str = Field(
        description="Instance path of the device owning the parameter."
    )
    parameter: str = Field(description="Parameter name, such as resistance.")
    value: float = Field(description="Nominal parameter value, in SI base units.")
    derivative: float = Field(
        description="Derivative of the output with respect to the parameter in SI "
        "base units, evaluated at the operating point or at peak magnitude."
    )
    at: float | None = Field(
        default=None,
        description="Sweep value, frequency, or time of peak derivative magnitude in "
        "SI base units, or null at an operating point.",
    )


class OutputSensitivity(ir.FrozenModel):
    """Summary of parameter derivatives for a single differentiated output."""

    output: str
    nature: unknowns.Nature | None = None
    value: float | None = Field(
        default=None,
        description="Nominal output value at the operating point in SI base units, "
        "or null for multi-point analyses.",
    )
    derivatives: tuple[Derivative, ...] = Field(
        description="Output derivatives with respect to the listed parameters in "
        "circuit order."
    )


type Differentiated = Literal[
    "operating_point",
    "dc_sweep",
    "ac",
    "stability",
    "noise",
    "transient",
    "periodic_steady_state",
]
"""Analysis kind differentiated by a sensitivity or mismatch analysis."""

_ALONG: Final[dict[Differentiated, str]] = {
    "dc_sweep": "along a DC sweep at {points} values",
    "ac": "over a response at {points} frequencies",
    "stability": "over a loop gain at {points} frequencies",
    "noise": "over a noise spectrum at {points} frequencies",
    "transient": "over a transient at {points} times",
    "periodic_steady_state": "over a periodic steady state's period at {points} times",
}


class _DifferentiatedReport(ir.FrozenModel):
    """Shared fields and helpers for sensitivity and mismatch reports."""

    name: str
    of: Differentiated = Field(
        description="Kind of underlying analysis differentiated."
    )
    points: int = Field(
        description="Number of points at which derivatives are evaluated."
    )
    strategy: str | None = Field(
        default=None,
        description="Solver strategy that found the operating point, or null for "
        "sweeps and transients.",
    )
    iterations: int | None = Field(default=None, description=_ITERATIONS)
    unit: str = Field(
        default="",
        description="Axis unit of the evaluated points: V or A for sweeps, Hz, or s.",
    )
    period: float | None = Field(
        default=None,
        description="Solved oscillator period in seconds, or null for other analyses.",
    )
    unlisted: int = Field(
        description="Number of entries omitted from each output list."
    )

    def _where(self) -> str:
        if self.of == "operating_point":
            found = _found(self.strategy or "", self.iterations)
            return f"at the operating point, {found}"
        return _ALONG[self.of].format(points=self.points)

    def _append_rows(self, lines: list[str], rows: list[list[str]]) -> None:
        lines.extend(f"  {row}" for row in _table(rows))
        if self.unlisted:
            lines.append(f"      ... and {self.unlisted} more in the results file")


class SensitivityReport(_DifferentiatedReport):
    """Report summarizing first-order parameter sensitivities.

    Sensitivity reports cover operating points, DC sweeps, AC responses, transients,
    periodic steady states, and oscillator periods.
    """

    kind: Literal["sensitivity"] = "sensitivity"
    parameters: int = Field(description="Total number of differentiated parameters.")
    outputs: tuple[OutputSensitivity, ...]
    period_derivatives: tuple[Derivative, ...] = Field(
        default=(),
        description="Derivatives of an oscillator period with respect to the listed "
        "parameters in circuit order.",
    )

    def describe(self) -> str:
        """Formats the sensitivity report as human-readable text.

        Returns:
            Multi-line human-readable first-order sensitivity summary.
        """
        header = (
            f"{self.name}: derivatives with respect to {self.parameters} parameters "
            f"{self._where()}"
        )
        lines = [header]
        for output in self.outputs:
            lines.append(
                f"  {output.output}"
                if output.value is None
                else f"  {output.output} = {_shown(output.value, output.nature)}"
            )
            self._append_rows(
                lines,
                [
                    [
                        each.instance,
                        each.parameter,
                        f"at {each.value!r}",
                        self._derivative(each),
                    ]
                    for each in output.derivatives
                ],
            )
        if self.period is not None:
            lines.append(f"  period = {format_si(self.period, 's')}")
            self._append_rows(
                lines,
                [
                    [
                        each.instance,
                        each.parameter,
                        f"at {each.value!r}",
                        repr(each.derivative),
                    ]
                    for each in self.period_derivatives
                ],
            )
        return "\n".join(lines)

    def _derivative(self, derivative: Derivative) -> str:
        if derivative.at is None:
            return repr(derivative.derivative)
        where = format_si(derivative.at, self.unit)
        return f"largest {derivative.derivative!r} at {where}"


class MismatchContribution(ir.FrozenModel):
    """Contribution of a single random variable to first-order output spread."""

    instance: str | None = Field(
        description="Instance path owning the random variable for part tolerances or "
        "foundry statistical calls, or null for shared lots."
    )
    lot: str | None = Field(
        description="Shared lot name for a lot-level random variable, or null."
    )
    name: str = Field(
        description="Random variable name within the instance or lot, such as a "
        "parameter name or statistical call label like vth0#0."
    )
    contribution: float = Field(
        description="First-order output shift per one-sigma change in the random "
        "variable, equal to derivative times standard deviation."
    )
    share: float = Field(
        description="Fraction of total first-order output variance from this variable."
    )


class OutputMismatch(ir.FrozenModel):
    """First-order statistical spread summary for one output or phasor component."""

    output: str
    nature: unknowns.Nature | None = None
    part: results.PhasorPart | None = Field(
        default=None,
        description="Phasor component analyzed for AC and stability analyses "
        "(magnitude or phase), or null for real-valued analyses.",
    )
    value: float | None = Field(
        default=None,
        description="Nominal output value at the operating point in SI base units, "
        "or null for multi-point analyses.",
    )
    standard_deviation: float = Field(
        description="First-order output standard deviation at the operating point or "
        "peak across points, in degrees for phasor phase."
    )
    at: float | None = Field(
        default=None,
        description="Sweep value, frequency, or time of peak standard deviation in "
        "SI base units, or null at an operating point.",
    )
    contributions: tuple[MismatchContribution, ...] = Field(
        description="Top random-variable contributors ordered by descending variance "
        "share, with phase contributions in degrees."
    )


class MismatchReport(_DifferentiatedReport):
    """Report summarizing first-order output spread across random variables."""

    kind: Literal["mismatch"] = "mismatch"
    random_variables: int = Field(
        description="Total number of random variables included in the analysis."
    )
    outputs: tuple[OutputMismatch, ...]
    period_standard_deviation: float | None = Field(
        default=None,
        description="First-order standard deviation of an oscillator period in "
        "seconds, or null for other analyses.",
    )

    def describe(self) -> str:
        """Formats the mismatch analysis report as human-readable text.

        Returns:
            Multi-line human-readable statistical mismatch summary.
        """
        header = (
            f"{self.name}: standard deviations, to first order, with "
            f"{self.random_variables} random variables {self._where()}"
        )
        lines = [header]
        for output in self.outputs:
            lines.append(f"  {self._spread(output)}")
            self._append_rows(
                lines,
                [
                    [
                        f"lot {each.lot}" if each.instance is None else each.instance,
                        each.name,
                        self._written(each.contribution, output),
                        f"{each.share:.1%}",
                    ]
                    for each in output.contributions
                ],
            )
        if self.period is not None and self.period_standard_deviation is not None:
            period = format_si(self.period, "s")
            deviation = format_si(self.period_standard_deviation, "s")
            lines.append(f"  period = {period}, standard deviation {deviation}")
        return "\n".join(lines)

    def _spread(self, output: OutputMismatch) -> str:
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
        if output.part == "phase":
            return f"{value:.4g}°"
        return _shown(value, output.nature)


class FailureReport(ir.FrozenModel):
    """Report recording a failed analysis."""

    kind: Literal["failure"] = "failure"
    name: str
    error: str = Field(description="Error message explaining the analysis failure.")

    def describe(self) -> str:
        """Formats the failure report as human-readable text.

        Returns:
            Single-line failure summary pairing analysis name and error message.
        """
        return f"{self.name}: failed: {self.error}"


class CornerReport(ir.FrozenModel):
    """Report summarizing simulation results at a single plan corner.

    Corner reports record circuit elaboration errors or failed analyses at the
    specified temperature and process corner.
    """

    name: str
    temperature: float = Field(description="Corner ambient temperature, in kelvins.")
    process: str | None = Field(
        default=None,
        description="Foundry process corner name, or null if not specified.",
    )
    problems: tuple[str, ...] = Field(
        default=(),
        description="Elaboration errors preventing simulation at the corner.",
    )
    failures: tuple[FailureReport, ...] = Field(
        default=(), description="Analyses that failed at the corner."
    )

    @property
    def ok(self) -> bool:
        """Flag indicating whether elaboration and all analyses succeeded."""
        return not self.problems and not self.failures

    def describe(self) -> str:
        """Formats the corner report as human-readable text.

        Returns:
            Human-readable corner elaboration and analysis status summary.
        """
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


class _BatchReport(ir.FrozenModel):
    """Shared fields and rendering for Monte Carlo and variant batch reports."""

    model_config = ConfigDict(
        **ir.FrozenModel.model_config, validate_by_name=True, serialize_by_alias=True
    )

    corners: tuple[str, ...] = Field(
        default=(),
        description="Additional plan corners evaluated for every run after nominal "
        "conditions.",
    )
    failed: int = Field(
        description="Number of runs with elaboration problems or failed analyses "
        "under nominal conditions or any corner."
    )
    spreads: tuple[Spread, ...] = Field(
        default=(),
        description="Measured value spreads for each specification requirement.",
    )
    yield_: Yield | None = Field(
        default=None,
        alias="yield",
        description="Summary of runs meeting all requirements, or null without specs.",
    )

    def _describe_batch(self, head: str, count: int) -> str:
        lines = [head]
        lines.extend(f"  {spread.describe(count)}" for spread in self.spreads)
        if self.yield_ is not None:
            lines.append(f"  {self.yield_.describe()}")
        return "\n".join(lines)


class MonteCarloReport(_BatchReport):
    """Report summarizing results of a Monte Carlo analysis over all samples."""

    samples: int
    seed: int
    toleranced: int = Field(
        description="Number of toleranced parameters sampled on each run."
    )
    mismatched: int = Field(
        default=0,
        description="Number of foundry devices with sampled statistical mismatch.",
    )

    def describe(self) -> str:
        """Formats the Monte Carlo report as human-readable text.

        Returns:
            Multi-line human-readable Monte Carlo spread and yield summary.
        """
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
        return self._describe_batch(head, self.samples)


LARGE: Final = 10e9
"""Byte threshold (10 GB) above which combined Monte Carlo shard arrays trigger a
warning in the shard report."""


class ShardReport(ir.FrozenModel):
    """Report summarizing a single Monte Carlo shard run and its output file."""

    start: int = Field(
        description="Inclusive starting sample index processed by the shard."
    )
    stop: int = Field(
        description="Exclusive ending sample index processed by the shard."
    )
    samples: int = Field(
        description="Total number of Monte Carlo samples across all shards."
    )
    corners: tuple[str, ...] = Field(
        default=(),
        description="Additional plan corners evaluated per sample after nominal "
        "conditions.",
    )
    failed: int = Field(
        description="Number of shard samples with elaboration problems or failed "
        "analyses under nominal conditions or any corner."
    )
    size: int = Field(
        description="Total byte size of the NumPy arrays stored in the shard file."
    )
    total: int = Field(
        description="Projected total byte size of arrays across all shards at the "
        "same bytes-per-sample rate."
    )
    digest: str = Field(
        description="Shared SHA-256 digest of inputs governing the Monte Carlo "
        "analysis across all shards."
    )

    def describe(self) -> str:
        """Formats the Monte Carlo shard report as human-readable text.

        Returns:
            Human-readable Monte Carlo shard summary, including any size warning.
        """
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


class VariantsReport(_BatchReport):
    """Report summarizing results across parameter variants of a plan."""

    runs: int
    parameters: tuple[str, ...] = Field(
        default=(),
        description="Names of parameters varied across runs, such as 'R1 resistance' "
        "or 'temperature'.",
    )

    def describe(self) -> str:
        """Formats the variants batch report as human-readable text.

        Returns:
            Multi-line human-readable parameter-variants spread and yield summary.
        """
        at = "".join(f", at {name}" for name in self.corners)
        varied = " and ".join(self.parameters) or "nothing"
        head = f"variants: {self.runs} runs of {varied}{at}; {self.failed} failed"
        return self._describe_batch(head, self.runs)


class SlackDerivative(ir.FrozenModel):
    """First-order derivative of a requirement's slack with respect to one parameter."""

    instance: str = Field(
        description="Instance path of the device owning the parameter."
    )
    parameter: str = Field(description="Parameter name, such as resistance.")
    derivative: float = Field(
        description="First-order derivative of signed slack with respect to the "
        "parameter in SI base units (positive increases slack)."
    )


class RequirementReport(ir.FrozenModel):
    """Requirement verdict paired with largest slack derivatives when failing."""

    verdict: Verdict
    derivatives: tuple[SlackDerivative, ...] = Field(
        default=(),
        description="Largest nonzero slack derivatives by magnitude for a failing "
        "requirement, ordered descending; empty when passing or omitted.",
    )
    unlisted: int = Field(
        default=0,
        description="Number of additional nonzero slack derivatives omitted from the "
        "report.",
    )

    def describe(self) -> str:
        """Formats the verdict and, when failing with derivatives, largest levers.

        Returns:
            Text identical to `Verdict.describe()` when passing or without
            derivatives, or followed by an indented table of largest slack
            derivatives.
        """
        head = self.verdict.describe()
        if self.verdict.status != "failed" or not self.derivatives:
            return head
        rows = [
            [each.instance, each.parameter, repr(each.derivative)]
            for each in self.derivatives
        ]
        lines = [head, "  largest slack derivatives (first order):"]
        lines.extend(f"  {row}" for row in _table(rows))
        if self.unlisted:
            lines.append(f"      ... and {self.unlisted} more")
        return "\n".join(lines)


def specs_report(
    verdicts: Sequence[Verdict],
    derivatives: Sequence[Mapping[str, Mapping[str, float]] | None] | None = None,
    *,
    max_derivatives: int = 5,
) -> tuple[RequirementReport, ...]:
    """Builds requirement reports, attaching largest slack derivatives to failures.

    When `derivatives` is omitted, every entry has no derivatives and
    [`describe_specs`][jolty.report.describe_specs] matches
    [`specs.describe`][jolty.specs.describe] verbatim. Non-failing requirements omit
    derivatives.

    Args:
        verdicts: Requirement verdicts in specification order.
        derivatives: Optional per-verdict slack derivative mappings aligned with
            `verdicts`, each mapping instance path and parameter name to
            ``d(slack)/d(p)``.
        max_derivatives: Maximum number of largest nonzero slack derivatives listed
            per failing requirement.

    Returns:
        Ordered `RequirementReport` entries matching `verdicts`.

    Raises:
        ValueError: When `max_derivatives` is less than 1 or `derivatives` length
            does not match `verdicts`.
    """
    if max_derivatives < 1:
        message = f"max_derivatives is at least 1, not {max_derivatives}"
        raise ValueError(message)
    if derivatives is None:
        return tuple(RequirementReport(verdict=verdict) for verdict in verdicts)
    if len(derivatives) != len(verdicts):
        message = (
            f"derivatives has {len(derivatives)} entries for {len(verdicts)} verdicts"
        )
        raise ValueError(message)
    built: list[RequirementReport] = []
    for verdict, mapping in zip(verdicts, derivatives, strict=True):
        if verdict.status != "failed" or mapping is None or not mapping:
            built.append(RequirementReport(verdict=verdict))
            continue
        entries = [
            SlackDerivative(
                instance=instance, parameter=parameter, derivative=float(value)
            )
            for instance, named in mapping.items()
            for parameter, value in named.items()
            if value != 0.0 and math.isfinite(value)
        ]
        entries.sort(
            key=lambda entry: (-abs(entry.derivative), entry.instance, entry.parameter)
        )
        listed = tuple(entries[:max_derivatives])
        built.append(
            RequirementReport(
                verdict=verdict,
                derivatives=listed,
                unlisted=max(len(entries) - max_derivatives, 0),
            )
        )
    return tuple(built)


def describe_specs(reports: Sequence[RequirementReport]) -> str:
    """Formats `reports` as text blocks followed by the status tally line.

    Args:
        reports: Requirement reports produced by `specs_report`.

    Returns:
        Multi-paragraph summary matching `specs.describe` with optional slack
        derivatives on failing requirements.
    """
    counts = Counter(item.verdict.status for item in reports)
    tally = ", ".join(
        f"{counts[status]} {status}"
        for status in ("passed", "marginal", "failed", "unmeasured")
        if counts[status]
    )
    return "\n\n".join([*(item.describe() for item in reports), f"specs: {tally}"])


class WorstCaseReport(ir.FrozenModel):
    """Report summarizing worst-case tolerance evaluation for each requirement."""

    step: float = Field(
        description="Fractional tolerance step used to determine sensitivity "
        "direction for each parameter."
    )
    toleranced: int = Field(description="Number of toleranced parameters perturbed.")
    corners: tuple[str, ...] = Field(
        default=(),
        description="Additional plan corners evaluated during worst-case analysis "
        "after nominal conditions.",
    )
    runs: int = Field(
        description="Total simulations executed for sensitivity probing and distinct "
        "extreme tolerance combinations."
    )
    verdicts: tuple[Verdict, ...] = Field(
        default=(),
        description="Worst-case verdict for each requirement across nominal "
        "conditions and all corners.",
    )

    @property
    def ok(self) -> bool:
        """Flag indicating whether all requirements pass or are marginal at extremes."""
        return all(
            verdict.status in {"passed", "marginal"} for verdict in self.verdicts
        )

    def describe(self) -> str:
        """Formats the worst-case report as human-readable text.

        Returns:
            Multi-line human-readable worst-case requirement verdict summary.
        """
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
"""Discriminated union of all analysis report types."""


def _largest_charge_leak(result: results.TransientResult) -> NetChargeLeak | None:
    """Returns the electrical net with the largest nonzero absolute charge leak."""
    leak = result.statistics.charge_leak
    if leak is None:
        return None
    worst_index: int | None = None
    worst_absolute = 0.0
    for index, unknown in enumerate(result.unknowns):
        if unknown.nature != "voltage":
            continue
        absolute = float(leak.absolute[index])
        if absolute > worst_absolute:
            worst_absolute = absolute
            worst_index = index
    if worst_index is None:
        return None
    return NetChargeLeak(
        net=result.unknowns[worst_index].name,
        signed=float(leak.signed[worst_index]),
        absolute=worst_absolute,
    )


def _waveforms_report(
    name: str,
    result: results.TransientResult,
    listed: tuple[tuple[Extent, ...], tuple[str, ...], int],
) -> TransientReport | PeriodicSteadyStateReport:
    """Builds a transient or periodic steady-state report from `result`."""
    extents, nets, unlisted = listed
    statistics = result.statistics
    charge_leak = _largest_charge_leak(result)
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
            charge_leak=charge_leak,
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
        charge_leak=charge_leak,
    )


def report(
    name: str, result: results.AnalysisResult, max_unknowns: int
) -> AnalysisReport:
    """Builds the structured report for analysis `name` from `result`.

    The report lists up to `max_unknowns` circuit unknowns followed by logic nets
    within the same combined limit.

    Args:
        name: Plan key naming the solved analysis.
        result: Solved analysis result to summarize.
        max_unknowns: Maximum number of unknowns, nets, or contributors to list.

    Returns:
        Structured report variant matching `result`.
    """
    listed_unknowns = result.unknowns[:max_unknowns]
    all_nets = results.nets_of(result)
    nets = all_nets[: max_unknowns - len(listed_unknowns)]
    unlisted = len(result.unknowns) + len(all_nets) - len(listed_unknowns) - len(nets)
    match result:
        case results.OperatingPointResult():
            values = tuple(
                Value(unknown=unknown.name, nature=unknown.nature, value=float(value))
                for unknown, value in zip(
                    listed_unknowns, result.values[:max_unknowns], strict=True
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
                extents=_extents_of(listed_unknowns, result.values[:, :max_unknowns]),
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
                    listed_unknowns, np.abs(result.values[:, :max_unknowns])
                ),
                unlisted=unlisted,
            )
        case results.TransientResult():
            extents = _extents_of(listed_unknowns, result.values[:, :max_unknowns])
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
                    for index, unknown in enumerate(listed_unknowns)
                ),
                unlisted=len(result.unknowns) - len(listed_unknowns),
            )
        case _:
            return _linear(name, result, max_unknowns)


def _transistors(point: results.OperatingPointResult) -> tuple[Transistor, ...]:
    """Returns transistor regions and margins from `point`, nearest boundary first."""
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
    """Builds a report for small-signal, sensitivity, or mismatch results."""
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
    """Builds a pole-zero report for `name` listing up to `max_unknowns` roots."""
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
    """Converts complex `roots` to report rows, keeping each conjugate pair once."""
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
    """Builds a noise report for `name` listing up to `max_unknowns` top sources."""
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
    """Builds a transfer-function report for analysis `name`."""
    output = next(
        unknown.nature for unknown in result.unknowns if unknown.name == result.output
    )
    return TransferFunctionReport(
        name=name,
        source=result.source,
        output=result.output,
        reference=result.reference,
        gain=result.gain,
        unit=f"{unknowns.SYMBOLS[output]}/{unknowns.SYMBOLS[result.input]}",
        input_resistance=result.input_resistance,
        output_resistance=result.output_resistance,
    )


def _stability(name: str, result: results.StabilityResult) -> StabilityReport:
    """Builds a stability report for analysis `name`."""
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
    """Evaluates `measure` on `result`, returning None if evaluation fails."""
    try:
        return measure.read(result).value
    except specs.MeasureError:
        return None


def _differentiated(
    result: results.DifferentiatedResult,
) -> tuple[Differentiated, int, str, results.OperatingPointResult | None]:
    """Extracts analysis kind, point count, unit, and operating point from `result`."""
    match result:
        case results.OperatingPointResult():
            return "operating_point", 1, "", result
        case results.DcSweepResult():
            unit = "" if result.nature is None else unknowns.SYMBOLS[result.nature]
            return "dc_sweep", len(result.swept), unit, None
        case results.AcResult():
            return "ac", len(result.frequencies), "Hz", result.operating_point
        case results.StabilityResult():
            return "stability", len(result.frequencies), "Hz", result.operating_point
        case results.NoiseResult():
            return "noise", len(result.frequencies), "Hz", result.operating_point
        case results.PeriodicSteadyStateResult():
            return "periodic_steady_state", len(result.times), "s", None
        case results.TransientResult():
            return "transient", len(result.times), "s", None


def _sensitivities(
    name: str, result: results.SensitivityResult, max_unknowns: int
) -> SensitivityReport:
    """Builds a sensitivity report for `name` with up to `max_unknowns` parameters."""
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
    """Builds sensitivity data for `output` over the first `max_unknowns` parameters."""
    gradient = result.gradient(output)
    match result.result:
        case results.OperatingPointResult() as point:
            column = results.column(result.unknowns, output)
            nature: results.Nature | None = result.unknowns[column].nature
            value: float | None = float(point.values[column])
            found = [
                (float(derivative.real), None) for derivative in gradient[:max_unknowns]
            ]
        case (
            results.DcSweepResult()
            | results.AcResult()
            | results.StabilityResult()
            | results.NoiseResult()
            | results.TransientResult() as along
        ):
            nature = (
                None
                if isinstance(along, results.StabilityResult | results.NoiseResult)
                else result.unknowns[results.column(result.unknowns, output)].nature
            )
            value = None
            axis = results.axis_of(along)
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


def _mismatch(
    name: str, result: results.MismatchResult, max_unknowns: int
) -> MismatchReport:
    """Builds a mismatch report for analysis `name`."""
    of, points, unit, point = _differentiated(result.result)
    parts: tuple[results.PhasorPart | None, ...] = (
        ("magnitude", "phase") if of in {"ac", "stability"} else (None,)
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
    part: results.PhasorPart | None,
    listed: int,
) -> OutputMismatch:
    """Builds first-order mismatch summary for `output` and optional phasor `part`."""
    contributions = np.atleast_2d(result.contributions(output, part))
    deviations = np.sqrt(np.sum(contributions**2, axis=-1))
    largest = int(np.argmax(deviations))
    if part == "phase":
        contributions = np.degrees(contributions)
        deviations = np.degrees(deviations)
    shares = np.atleast_2d(result.shares(output, part))[largest]
    order = np.argsort(-shares, kind="stable")[:listed]
    value: float | None = None
    at: float | None = None
    match result.result:
        case results.OperatingPointResult() as found:
            column = results.column(result.unknowns, output)
            nature: results.Nature | None = result.unknowns[column].nature
            value = float(found.values[column])
        case (
            results.DcSweepResult()
            | results.AcResult()
            | results.StabilityResult()
            | results.NoiseResult()
            | results.TransientResult() as along
        ):
            nature = (
                None
                if isinstance(along, results.StabilityResult | results.NoiseResult)
                else result.unknowns[results.column(result.unknowns, output)].nature
            )
            at = float(results.axis_of(along)[largest])
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


def _level(level: float) -> LogicLevel:
    """Converts a numeric logic `level` to 0, 1, or None for NaN."""
    if np.isnan(level):
        return None
    return 1 if level else 0


def _activity(net: str, levels: results.Real) -> Activity:
    """Summarizes initial level, final level, and transition count for logic `net`."""
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
    listed_unknowns: tuple[results.Unknown, ...], values: results.Real
) -> tuple[Extent, ...]:
    """Computes first, minimum, maximum, and final values for `listed_unknowns`."""
    return tuple(
        Extent(
            unknown=unknown.name,
            nature=unknown.nature,
            first=float(values[0, column]),
            minimum=float(values[:, column].min()),
            maximum=float(values[:, column].max()),
            last=float(values[-1, column]),
        )
        for column, unknown in enumerate(listed_unknowns)
    )


def _extents(extents: tuple[Extent, ...]) -> list[str]:
    if not extents:
        return []
    header = ["", "first", "min", "max", "last"]
    rows = [
        [
            extent.unknown,
            _shown(extent.first, extent.nature),
            _shown(extent.minimum, extent.nature),
            _shown(extent.maximum, extent.nature),
            _shown(extent.last, extent.nature),
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
