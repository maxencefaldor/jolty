# Copyright 2026 Maxence Faldor
"""Simulation plans: the analyses to run on a circuit, and the numerical options.

A plan is kept apart from the circuit: the circuit says what exists, the plan how to
simulate it.

Examples:
    >>> from jolty.units import us
    >>> plan = Plan({"step": Transient(stop=10 * us)})
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Literal

from pydantic import JsonValue

from jolty.units import FREQUENCY, TIME, Dimension, Quantity

type Method = Literal["trapezoidal", "gear2", "backward_euler"]
"""An integration method: trapezoidal by default, Gear2 for stiff circuits that ring,
backward Euler for the most damping."""


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """The DC operating point, with every source at its level at time zero."""

    def to_json(self) -> JsonValue:
        """The analysis in the engine's format."""
        return "operating_point"


@dataclass(frozen=True, slots=True)
class DcSweep:
    """The operating point at each level of one independent source.

    The source is named by its path, such as ``V1`` or ``bias/I1``.
    """

    source: str
    levels: Sequence[Quantity]

    def to_json(self, level: Dimension) -> JsonValue:
        """The analysis in the engine's format, its levels checked against `level`."""
        levels: list[JsonValue] = [
            value.to(level, "a sweep level") for value in self.levels
        ]
        return {"dc_sweep": {"source": self.source, "levels": levels}}


@dataclass(frozen=True, slots=True)
class Ac:
    """The small-signal response to a unit excitation of a source, at each frequency."""

    source: str
    frequencies: Sequence[Quantity]

    def to_json(self) -> JsonValue:
        """The analysis in the engine's format."""
        frequencies: list[JsonValue] = [
            frequency.to(FREQUENCY, "a frequency") for frequency in self.frequencies
        ]
        return {"ac": {"source": self.source, "frequencies": frequencies}}


@dataclass(frozen=True, slots=True)
class Transient:
    """The response over time, from the operating point at zero until `stop`."""

    stop: Quantity
    max_step: Quantity | None = None
    method: Method = "trapezoidal"

    def to_json(self) -> JsonValue:
        """The analysis in the engine's format."""
        transient: dict[str, JsonValue] = {
            "stop": self.stop.to(TIME, "a transient's stop time"),
            "method": self.method,
        }
        if self.max_step is not None:
            transient["max_step"] = self.max_step.to(TIME, "a transient's longest step")
        return {"transient": transient}


type Analysis = OperatingPoint | DcSweep | Ac | Transient
"""Any analysis."""


@dataclass(frozen=True, slots=True)
class Options:
    """Numerical settings; each one left out takes the engine's default, SPICE's own.

    Tolerances are in SI base units: volts, amperes and coulombs.
    """

    relative_tolerance: float | None = None
    voltage_tolerance: float | None = None
    current_tolerance: float | None = None
    charge_tolerance: float | None = None
    gmin: float | None = None
    dc_iterations: int | None = None
    transient_iterations: int | None = None
    truncation_tolerance: float | None = None

    def to_json(self) -> JsonValue:
        """The options in the engine's format, without those left to its defaults."""
        values = {setting.name: getattr(self, setting.name) for setting in fields(self)}
        return {name: value for name, value in values.items() if value is not None}


@dataclass(frozen=True, slots=True)
class Plan:
    """Named analyses to run on a circuit, and the options they share.

    Each analysis is named by a Python identifier, such as ``step`` or ``bode``, which
    names its results too.
    """

    analyses: Mapping[str, Analysis]
    options: Options = field(default_factory=Options)

    def __post_init__(self) -> None:
        """Checks that the plan runs an analysis or more, each named by an identifier.

        Raises:
            ValueError: When the plan runs no analysis, or a name is not an identifier.
        """
        if not self.analyses:
            message = "a plan runs at least one analysis"
            raise ValueError(message)
        for name in self.analyses:
            if not name.isidentifier():
                message = f"an analysis is named by a Python identifier, not {name!r}"
                raise ValueError(message)


def decade(start: Quantity, stop: Quantity, per_decade: int) -> list[Quantity]:
    """Frequencies from `start` to `stop`, `per_decade` of them spaced evenly a decade.

    Examples:
        >>> from jolty.units import Hz
        >>> [str(f) for f in decade(1 * Hz, 100 * Hz, 2)]
        ['1 Hz', '3.162 Hz', '10 Hz', '31.62 Hz', '100 Hz']
    """
    low, high = (
        start.to(FREQUENCY, "a sweep's start"),
        stop.to(FREQUENCY, "a sweep's stop"),
    )
    if not 0 < low < high or per_decade < 1:
        message = "a decade sweep rises from a positive start, a point or more a decade"
        raise ValueError(message)
    first, last = math.log10(low), math.log10(high)
    count = max(1, round((last - first) * per_decade))
    return [
        Quantity(10 ** (first + (last - first) * step / count), FREQUENCY)
        for step in range(count + 1)
    ]


def linear(start: Quantity, stop: Quantity, points: int) -> list[Quantity]:
    """`points` quantities evenly spaced from `start` to `stop`, both included.

    Examples:
        >>> from jolty.units import V
        >>> [str(level) for level in linear(0 * V, 1 * V, 3)]
        ['0 V', '500 mV', '1 V']
    """
    if points < 2:  # noqa: PLR2004 - a start and a stop
        message = f"a linear sweep has at least two points, not {points}"
        raise ValueError(message)
    step = (stop - start) / (points - 1)
    return [start + step * index for index in range(points)]
