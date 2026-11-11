# Copyright 2026 Maxence Faldor
"""Waveforms: the level of an independent source over time.

Levels are voltages for a voltage source and currents for a current source; the source
checks them as it reads the waveform. Every analysis but a transient one reads the level
at time zero.

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
    """A constant level."""

    level: Quantity

    def to_json(self, level: Dimension) -> JsonValue:
        """The waveform in the circuit format, its levels checked against `level`."""
        return {"dc": self.level.to(level, "a DC level")}


@dataclass(frozen=True, slots=True)
class Pulse:
    """A trapezoidal pulse from `initial` to `pulsed` and back, once or every period."""

    initial: Quantity
    pulsed: Quantity
    rise: Quantity
    fall: Quantity
    width: Quantity
    delay: Quantity = field(default_factory=lambda: Quantity(0.0, TIME))
    period: Quantity | None = None

    def to_json(self, level: Dimension) -> JsonValue:
        """The waveform in the circuit format, its levels checked against `level`."""
        pulse: dict[str, JsonValue] = {
            "initial": self.initial.to(level, "a pulse's initial level"),
            "pulsed": self.pulsed.to(level, "a pulse's pulsed level"),
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
    """A sine wave around `offset`, starting after `delay` and decaying at `damping`.

    Its level is ``offset + amplitude·exp(-damping·t')·sin(2π·frequency·t' + phase)``,
    where ``t'`` is the time since the delay; before the delay it holds its level at
    ``t' = 0``. The phase is in radians.
    """

    offset: Quantity
    amplitude: Quantity
    frequency: Quantity
    delay: Quantity = field(default_factory=lambda: Quantity(0.0, TIME))
    damping: Quantity = field(default_factory=lambda: Quantity(0.0, FREQUENCY))
    phase: float = 0.0

    def to_json(self, level: Dimension) -> JsonValue:
        """The waveform in the circuit format, its levels checked against `level`."""
        return {
            "sine": {
                "offset": self.offset.to(level, "a sine's offset"),
                "amplitude": self.amplitude.to(level, "a sine's amplitude"),
                "frequency": self.frequency.to(FREQUENCY, "a sine's frequency"),
                "delay": self.delay.to(TIME, "a sine's delay"),
                "damping": self.damping.to(FREQUENCY, "a sine's damping"),
                "phase": finite(self.phase, "a sine's phase"),
            }
        }


@dataclass(frozen=True, slots=True)
class Piecewise:
    """Straight lines through ``(time, level)`` points, held before and after them."""

    points: Sequence[tuple[Quantity, Quantity]]

    def to_json(self, level: Dimension) -> JsonValue:
        """The waveform in the circuit format, its levels checked against `level`."""
        points: list[JsonValue] = [
            [time.to(TIME, "a point's time"), value.to(level, "a point's level")]
            for time, value in self.points
        ]
        return {"piecewise": {"points": points}}


type Waveform = Dc | Pulse | Sine | Piecewise
"""Any waveform."""
