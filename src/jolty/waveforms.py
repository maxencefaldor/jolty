# Copyright 2026 Maxence Faldor
"""Time-varying waveforms for independent voltage and current sources.

Levels represent voltages for voltage sources and currents for current sources; each
source validates physical dimensions when accepting a waveform. Every non-transient
analysis evaluates the waveform at time zero.

Examples:
    >>> from jolty.units import V, ns, us
    >>> pulse = Pulse(
    ...     initial=0 * V, pulsed=1 * V, rise=1 * ns, fall=1 * ns, width=5 * us
    ... )
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import JsonValue

from jolty.units import FREQUENCY, TIME, Dimension, Quantity, finite


@dataclass(frozen=True, slots=True)
class Dc:
    """Constant DC source level."""

    value: Quantity

    def to_json(self, dimension: Dimension) -> JsonValue:
        """Serializes the DC waveform to IR JSON after checking `dimension`."""
        return {"dc": self.value.to(dimension, "a DC level")}


@dataclass(frozen=True, slots=True)
class Pulse:
    """Trapezoidal pulse switching between `initial` and `pulsed`."""

    initial: Quantity
    pulsed: Quantity
    rise: Quantity
    fall: Quantity
    width: Quantity
    delay: Quantity = field(default_factory=lambda: Quantity(0.0, TIME))
    period: Quantity | None = None

    def to_json(self, dimension: Dimension) -> JsonValue:
        """Serializes the pulse waveform to IR JSON after checking `dimension`."""
        pulse: dict[str, JsonValue] = {
            "initial": self.initial.to(dimension, "a pulse's initial level"),
            "pulsed": self.pulsed.to(dimension, "a pulse's pulsed level"),
            "delay": self.delay.to(TIME, "a pulse's delay"),
            "rise": self.rise.to(TIME, "a pulse's rise"),
            "fall": self.fall.to(TIME, "a pulse's fall"),
            "width": self.width.to(TIME, "a pulse's width"),
        }
        if self.period is not None:
            pulse["period"] = self.period.to(TIME, "a pulse's period")
        return {"pulse": pulse}


@dataclass(frozen=True, slots=True)
class Sine:
    """Damped sinusoidal waveform oscillating around `offset` after `delay`.

    The waveform evaluates
    ``offset + amplitude·exp(-damping·t')·sin(2π·frequency·t' + phase)`` where ``t'``
    is elapsed time since `delay` and `phase` is in radians. Prior to `delay`, output
    holds its value at ``t' = 0``.
    """

    offset: Quantity
    amplitude: Quantity
    frequency: Quantity
    delay: Quantity = field(default_factory=lambda: Quantity(0.0, TIME))
    damping: Quantity = field(default_factory=lambda: Quantity(0.0, FREQUENCY))
    phase: float = 0.0

    def to_json(self, dimension: Dimension) -> JsonValue:
        """Serializes the sine waveform to IR JSON after checking `dimension`."""
        return {
            "sine": {
                "offset": self.offset.to(dimension, "a sine's offset"),
                "amplitude": self.amplitude.to(dimension, "a sine's amplitude"),
                "frequency": self.frequency.to(FREQUENCY, "a sine's frequency"),
                "delay": self.delay.to(TIME, "a sine's delay"),
                "damping": self.damping.to(FREQUENCY, "a sine's damping"),
                "phase": finite(self.phase, "a sine's phase"),
            }
        }


@dataclass(frozen=True, slots=True)
class Piecewise:
    """Piecewise-linear waveform through ``(time, level)`` pairs with clamped ends."""

    points: Sequence[tuple[Quantity, Quantity]]

    def to_json(self, dimension: Dimension) -> JsonValue:
        """Serializes the piecewise waveform to IR JSON after checking `dimension`."""
        points: list[JsonValue] = [
            [time.to(TIME, "a point's time"), value.to(dimension, "a point's level")]
            for time, value in self.points
        ]
        return {"piecewise": {"points": points}}


@dataclass(frozen=True, slots=True)
class Exponential:
    """SPICE ``EXP`` waveform rising toward `pulsed` then decaying back to `initial`.

    Output holds `initial` until `rise_delay`, then adds
    ``(pulsed - initial)·(1 - exp(-(t - rise_delay)/rise_time_constant))``. Starting
    at `fall_delay` (at or after `rise_delay`), output adds
    ``(initial - pulsed)·(1 - exp(-(t - fall_delay)/fall_time_constant))``.
    """

    initial: Quantity
    pulsed: Quantity
    rise_time_constant: Quantity
    fall_delay: Quantity
    fall_time_constant: Quantity
    rise_delay: Quantity = field(default_factory=lambda: Quantity(0.0, TIME))

    def to_json(self, dimension: Dimension) -> JsonValue:
        """Serializes the exponential waveform to IR JSON checking `dimension`."""
        return {
            "exponential": {
                "initial": self.initial.to(dimension, "an exponential's initial level"),
                "pulsed": self.pulsed.to(dimension, "an exponential's pulsed level"),
                "rise_delay": self.rise_delay.to(TIME, "an exponential's rise delay"),
                "rise_time_constant": self.rise_time_constant.to(
                    TIME, "an exponential's rise time constant"
                ),
                "fall_delay": self.fall_delay.to(TIME, "an exponential's fall delay"),
                "fall_time_constant": self.fall_time_constant.to(
                    TIME, "an exponential's fall time constant"
                ),
            }
        }


type Waveform = Dc | Pulse | Sine | Piecewise | Exponential
"""Union of all supported independent-source waveform types."""
