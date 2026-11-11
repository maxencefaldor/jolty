# Copyright 2026 Maxence Faldor
"""Reports: what each analysis found, as data and as text for people and agents.

A report states facts: the value of every unknown at an operating point, and the first,
lowest, highest and last value of every unknown over a sweep, a frequency response or a
transient. It lists the first unknowns, up to a limit; the results file holds them all.
"""

from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from jolty import results
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


class PointReport(_Model):
    """An operating point."""

    kind: Literal["operating_point"] = "operating_point"
    name: str
    strategy: str = Field(
        description="How the engine found it: direct, gmin_stepping or source_stepping."
    )
    values: tuple[Value, ...]
    unlisted: int = Field(description="Unknowns left out of the list.")

    def describe(self) -> str:
        """The report as text."""
        strategy = _STRATEGIES.get(self.strategy, self.strategy)
        rows = [
            [value.unknown, format_si(value.value, _SYMBOLS[value.quantity])]
            for value in self.values
        ]
        head = f"{self.name}: operating point, found by {strategy}"
        return _join(head, _table(rows), self.unlisted)


class SweepReport(_Model):
    """A DC sweep."""

    kind: Literal["dc_sweep"] = "dc_sweep"
    name: str
    levels: int = Field(description="Levels the sweep solved.")
    extents: tuple[Extent, ...]
    unlisted: int = Field(description="Unknowns left out of the list.")

    def describe(self) -> str:
        """The report as text."""
        head = f"{self.name}: DC sweep over {self.levels} levels"
        return _join(head, _extents(self.extents), self.unlisted)


class ResponseReport(_Model):
    """A small-signal frequency response, as magnitudes per unit excitation."""

    kind: Literal["ac"] = "ac"
    name: str
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
            f"{self.name}: response at {self.frequencies} frequencies from "
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
    rejected: int = Field(description="Time steps rejected for their error or Newton.")
    extents: tuple[Extent, ...]
    unlisted: int = Field(description="Unknowns left out of the list.")

    def describe(self) -> str:
        """The report as text."""
        head = (
            f"{self.name}: transient to {format_si(self.stop, 's')} "
            f"in {self.accepted} steps, {self.rejected} rejected"
        )
        return _join(head, _extents(self.extents), self.unlisted)


class FailureReport(_Model):
    """An analysis that failed."""

    kind: Literal["failure"] = "failure"
    name: str
    error: str = Field(description="Why the analysis failed.")

    def describe(self) -> str:
        """The report as text."""
        return f"{self.name}: failed: {self.error}"


type AnalysisReport = Annotated[
    PointReport | SweepReport | ResponseReport | TransientReport | FailureReport,
    Field(discriminator="kind"),
]
"""The report of any analysis."""


def report(name: str, result: results.Result, listed: int) -> AnalysisReport:
    """The report of the analysis `name`, listing its first `listed` unknowns."""
    unknowns = result.unknowns[:listed]
    unlisted = len(result.unknowns) - len(unknowns)
    match result:
        case results.OperatingPoint():
            values = tuple(
                Value(
                    unknown=unknown.name, quantity=unknown.quantity, value=float(value)
                )
                for unknown, value in zip(unknowns, result.values[:listed], strict=True)
            )
            return PointReport(
                name=name, strategy=result.strategy, values=values, unlisted=unlisted
            )
        case results.Sweep():
            return SweepReport(
                name=name,
                levels=len(result.levels),
                extents=_extents_of(unknowns, result.values[:, :listed]),
                unlisted=unlisted,
            )
        case results.Response():
            return ResponseReport(
                name=name,
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
                unlisted=unlisted,
            )


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
