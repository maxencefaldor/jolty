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

from typing import Annotated, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from jolty import results
from jolty.specs import Spread, Yield
from jolty.units import format_si

type Quantity = Literal["voltage", "current"]

_SYMBOLS = {"voltage": "V", "current": "A"}

INFLUENCES: Final = 5
"""The parameters a sensitivity report lists per output, those it moves most with."""

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


class Influence(_Model):
    """How one output moves with one parameter."""

    instance: str = Field(description="The device's path.")
    parameter: str = Field(description="The parameter's name, such as resistance.")
    value: float = Field(description="The parameter's value, in SI base units.")
    derivative: float = Field(
        description="The output's derivative with respect to the parameter, in SI "
        "base units."
    )
    change: float = Field(
        description="How much the output moves, to first order, as the parameter "
        "grows by 1 %, in volts or amperes."
    )


class OutputSensitivity(_Model):
    """The parameters one output moves most with."""

    output: str
    quantity: Quantity
    value: float = Field(description="Its value, in volts or amperes.")
    influences: tuple[Influence, ...] = Field(
        description="The parameters it moves with, most first; none it does not "
        "move with."
    )


class SensitivityReport(_Model):
    """DC sensitivities."""

    kind: Literal["sensitivity"] = "sensitivity"
    name: str
    strategy: str = Field(
        description="How the engine found the operating point: direct, "
        "gmin_stepping or source_stepping."
    )
    parameters: int = Field(
        description="Parameters the derivatives are with respect to."
    )
    outputs: tuple[OutputSensitivity, ...]

    def describe(self) -> str:
        """The report as text."""
        strategy = _STRATEGIES.get(self.strategy, self.strategy)
        head = (
            f"{self.name}: sensitivities to {self.parameters} parameters at the "
            f"operating point, found by {strategy}"
        )
        lines = [head]
        for output in self.outputs:
            unit = _SYMBOLS[output.quantity]
            value = format_si(output.value, unit)
            if not output.influences:
                lines.append(f"  {output.output} = {value} moves with no parameter")
                continue
            lines.append(
                f"  {output.output} = {value} moves, as each parameter grows by 1 %, by"
            )
            rows = [
                [
                    influence.instance,
                    influence.parameter,
                    f"{influence.value:.4g}",
                    ("+" if influence.change > 0 else "")
                    + format_si(influence.change, unit),
                ]
                for influence in output.influences
            ]
            lines.extend(f"  {row}" for row in _table(rows))
        return "\n".join(lines)


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
    failed: int = Field(
        description="Samples on which the circuit did not elaborate or an analysis "
        "failed."
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
        head = (
            f"monte carlo: {self.samples} samples of {self.toleranced} toleranced "
            f"parameters, seed {self.seed}; {self.failed} failed"
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
        case results.Sensitivities():
            return SensitivityReport(
                name=name,
                strategy=result.operating_point.strategy,
                parameters=len(result.parameters),
                outputs=tuple(_output(result, output) for output in result.outputs),
            )


def _output(result: results.Sensitivities, output: str) -> OutputSensitivity:
    """The parameters `output` moves most with, up to `INFLUENCES` of them."""
    changes = result.per_percent(output)
    largest = np.argsort(-np.abs(changes), kind="stable")[:INFLUENCES]
    row = result.outputs.index(output)
    influences = tuple(
        Influence(
            instance=result.parameters[index].instance,
            parameter=result.parameters[index].name,
            value=result.parameters[index].value,
            derivative=float(result.derivatives[row, index]),
            change=float(changes[index]),
        )
        for index in largest
        if changes[index] != 0
    )
    column = results.column(result.unknowns, output)
    return OutputSensitivity(
        output=output,
        quantity=result.unknowns[column].quantity,
        value=float(result.operating_point.values[column]),
        influences=influences,
    )


def _nets(result: results.Result) -> tuple[str, ...]:
    """The logic nets of `result`; responses and sensitivities have none of theirs."""
    if isinstance(result, results.Response | results.Sensitivities):
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
