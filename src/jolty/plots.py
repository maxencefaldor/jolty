# Copyright 2026 Maxence Faldor
"""Plots of results: waveforms over time, DC sweeps over their levels, and Bode plots.

A plot draws chosen unknowns and logic nets of one result. Voltages, currents and logic
levels never share an axis: each quantity has a panel of its own, and a response has
two, its magnitude and its phase. Logic nets share a panel in lanes of their own, each
named on its axis, with unknown levels hatched. Series take their colors in a fixed
order, checked to stay apart under the common color-vision deficiencies; a legend names
them, and up to four are also labeled at their end.

Examples:
    >>> import numpy as np
    >>> from jolty import results
    >>> unknowns = (results.Unknown("out", "voltage"),)
    >>> statistics = results.Statistics("direct", 1, 0, 2)
    >>> step = results.Waveforms(
    ...     unknowns, np.array([0.0, 1e-6]), np.array([[0.0], [1.0]]), statistics
    ... )
    >>> figure = plot("step", step, default_probes(unknowns))
    >>> [axes.get_ylabel() for axes in figure.axes]
    ['voltage']
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter

from jolty import results
from jolty.errors import JoltyError

COLORS: Final = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
"""The series colors, in the order series take them; adjacent ones stay apart under
every common color-vision deficiency."""

MAX_SERIES: Final = len(COLORS)
"""The most unknowns one plot draws."""

_LABELED: Final = 4
"""The most series a panel labels at their end, beside its legend."""

_SURFACE: Final = "#fcfcfb"
_INK: Final = "#0b0b0b"
_MUTED: Final = "#52514e"
_GRID: Final = "#e4e3df"
_UNITS: Final = {"voltage": "V", "current": "A"}

type Plotted = results.Sweep | results.Response | results.Waveforms
"""A result a plot draws; an operating point has no axis to draw along."""

type _Quantity = Literal["voltage", "current", "logic"]

_LANE: Final = 1.5
"""The height of a logic net's lane, in levels: its level spans 1, and a gap follows."""


def default_probes(
    unknowns: Sequence[results.Unknown], nets: Sequence[str] = ()
) -> tuple[str, ...]:
    """What a plot draws unless told otherwise: the top component's named nets.

    A net the script named at the top has no slash in its path, and no dot or colon,
    which only the names the builder and the engine give other nets carry. Its voltage
    is drawn, or its level if it is a logic net. Without such nets, a plot draws the
    first voltages, or else the first logic nets. Either way, it draws at most
    [`MAX_SERIES`][jolty.plots.MAX_SERIES].
    """
    voltages = [unknown.name for unknown in unknowns if unknown.quantity == "voltage"]
    named = [
        name for name in [*voltages, *nets] if not any(mark in name for mark in "/.:")
    ]
    return tuple((named or voltages or list(nets))[:MAX_SERIES])


def plot(name: str, result: Plotted, probes: Sequence[str]) -> Figure:
    """The plot of the unknowns `probes` in the result of the analysis `name`.

    Raises:
        JoltyError: When a probe names no unknown, or there are more than
            [`MAX_SERIES`][jolty.plots.MAX_SERIES] probes.
    """
    if len(probes) > MAX_SERIES:
        message = f"a plot draws at most {MAX_SERIES} unknowns, not {len(probes)}"
        raise JoltyError(message)
    nets = () if isinstance(result, results.Response) else result.nets
    columns = {}
    groups: dict[_Quantity, list[str]] = {"voltage": [], "current": [], "logic": []}
    for probe in probes:
        if probe in nets:
            groups["logic"].append(probe)
            continue
        try:
            columns[probe] = results.column(result.unknowns, probe)
        except KeyError as error:
            raise JoltyError(error.args[0]) from None
        groups[result.unknowns[columns[probe]].quantity].append(probe)
    quantities = [quantity for quantity, members in groups.items() if members]
    panels = (
        2 * len(quantities) if isinstance(result, results.Response) else len(quantities)
    )
    figure = Figure(
        figsize=(8, 0.8 + 2.4 * max(panels, 1)),
        dpi=150,
        layout="constrained",
        facecolor=_SURFACE,
    )
    figure.suptitle(f"{name}: {_describe(result)}", color=_INK, x=0.02, ha="left")
    axes = list(figure.subplots(max(panels, 1), 1, sharex=True, squeeze=False)[:, 0])
    colors = {probe: COLORS[index] for index, probe in enumerate(probes)}
    x, x_unit = _abscissa(result)
    if isinstance(result, results.Response):
        for index, quantity in enumerate(quantities):
            phasors = [
                (probe, colors[probe], result.values[:, columns[probe]])
                for probe in groups[quantity]
            ]
            magnitudes = [
                (probe, color, _decibels(values)) for probe, color, values in phasors
            ]
            phases = [
                (probe, color, np.degrees(np.unwrap(np.angle(values))))
                for probe, color, values in phasors
            ]
            _draw(axes[2 * index], x, magnitudes, f"{quantity} magnitude", "dB")
            _draw(axes[2 * index + 1], x, phases, f"{quantity} phase", "°")
        for each in axes:
            each.set_xscale("log")
    else:
        for index, quantity in enumerate(quantities):
            if quantity == "logic":
                lanes = [
                    (probe, colors[probe], *_steps(result, probe))
                    for probe in groups[quantity]
                ]
                _draw_logic(axes[index], lanes)
                continue
            series = [
                (probe, colors[probe], result.values[:, columns[probe]])
                for probe in groups[quantity]
            ]
            _draw(axes[index], x, series, quantity, _UNITS[quantity])
    bottom = axes[-1]
    bottom.xaxis.set_major_formatter(EngFormatter(unit=x_unit, sep=" "))
    bottom.set_xlabel(_abscissa_label(result), color=_MUTED)
    return figure


def _steps(
    result: results.Sweep | results.Waveforms, net: str
) -> tuple[results.Real, results.Real]:
    """Where the level of the logic net `net` changes, and the level from each on.

    A transient's last level holds until it stops; a sweep's levels change halfway
    between the source's levels.
    """
    if isinstance(result, results.Waveforms):
        trace = result.level(net)
        return (
            np.append(trace.times, result.times[-1]),
            np.append(trace.levels, trace.levels[-1]),
        )
    levels = result.level(net)
    edges = np.concatenate(
        [
            result.levels[:1],
            (result.levels[1:] + result.levels[:-1]) / 2,
            result.levels[-1:],
        ]
    )
    return edges, np.append(levels, levels[-1])


def _draw_logic(
    axes: Axes, lanes: list[tuple[str, str, results.Real, results.Real]]
) -> None:
    """Draws the logic nets of `lanes` top to bottom, unknown levels hatched.

    Each lane is a net's name, its color, the times its level changes and the level
    from each.
    """
    _style(axes, "logic")
    for index, (_, color, x, levels) in enumerate(lanes):
        base = (len(lanes) - 1 - index) * _LANE
        known = np.where(np.isnan(levels), np.nan, levels + base)
        axes.step(x, known, where="post", color=color, linewidth=1.0)
        axes.fill_between(
            x,
            base,
            base + 1,
            where=np.isnan(levels),
            step="post",
            facecolor="none",
            edgecolor=color,
            hatch="///",
            linewidth=0,
        )
    axes.set_yticks(
        [(len(lanes) - 1 - index) * _LANE + 0.5 for index in range(len(lanes))],
        [name for name, *_ in lanes],
    )
    axes.set_ylim(-0.25, (len(lanes) - 1) * _LANE + 1.25)
    axes.grid(visible=False, axis="y")


def _decibels(phasors: results.Complex) -> results.Real:
    """The magnitude of `phasors` in decibels, a zero magnitude at -6000 dB."""
    return 20 * np.log10(np.maximum(np.abs(phasors), 1e-300))


def save(figure: Figure, path: Path) -> None:
    """Writes `figure` to `path` as a PNG image."""
    figure.savefig(path, format="png", facecolor=figure.get_facecolor())


def _describe(result: Plotted) -> str:
    match result:
        case results.Waveforms():
            return "transient"
        case results.Sweep():
            return f"DC sweep of {result.source}"
        case results.Response():
            return f"response to a unit excitation of {result.source}"


def _abscissa(result: Plotted) -> tuple[results.Real, str]:
    match result:
        case results.Waveforms():
            return result.times, "s"
        case results.Sweep():
            return result.levels, _UNITS[result.quantity]
        case results.Response():
            return result.frequencies, "Hz"


def _abscissa_label(result: Plotted) -> str:
    match result:
        case results.Waveforms():
            return "time"
        case results.Sweep():
            return f"{result.source} {result.quantity}"
        case results.Response():
            return "frequency"


def _draw(
    axes: Axes,
    x: results.Real,
    series: list[tuple[str, str, results.Real]],
    label: str,
    unit: str,
) -> None:
    """Draws `series`, each a name, a color and values along `x`, on `axes`."""
    for name, color, values in series:
        axes.plot(x, values, color=color, linewidth=1.0, label=name)
    _style(axes, label)
    axes.yaxis.set_major_formatter(EngFormatter(unit=unit, sep=" "))
    if len(series) > 1:
        legend = axes.legend(loc="best", frameon=False, fontsize=8)
        for text in legend.get_texts():
            text.set_color(_INK)
    if len(series) <= _LABELED:
        _label_ends(
            axes, [(name, color, float(values[-1])) for name, color, values in series]
        )


def _style(axes: Axes, label: str) -> None:
    """Gives `axes` its surface, its grid and its spines, and labels it `label`."""
    axes.set_facecolor(_SURFACE)
    axes.set_ylabel(label, color=_MUTED)
    axes.grid(visible=True, color=_GRID, linewidth=0.6)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(_MUTED)
        axes.spines[side].set_linewidth(0.6)
    axes.tick_params(colors=_MUTED, labelsize=8)


def _label_ends(axes: Axes, ends: list[tuple[str, str, float]]) -> None:
    """Labels each series beside the right edge of `axes`, at the height it ends at.

    Labels that would overlap are pushed apart; a short line in the series' color
    precedes each name, which is in ink.
    """
    axes.autoscale_view()
    to_axes = axes.transAxes.inverted()
    x_end = axes.get_xlim()[1]
    heights = [
        float(to_axes.transform(axes.transData.transform((x_end, value)))[1])
        for _, _, value in ends
    ]
    order = sorted(range(len(ends)), key=lambda index: heights[index])
    gap = 0.09
    placed = [0.0] * len(ends)
    floor = -np.inf
    for index in order:
        placed[index] = max(heights[index], floor + gap)
        floor = placed[index]
    overflow = max(0.0, floor - 1.0)
    for index, (name, color, _) in enumerate(ends):
        height = placed[index] - overflow
        axes.plot(
            [1.01, 1.04],
            [height, height],
            transform=axes.transAxes,
            color=color,
            linewidth=1.5,
            clip_on=False,
        )
        axes.text(
            1.05,
            height,
            name,
            transform=axes.transAxes,
            color=_INK,
            fontsize=8,
            va="center",
        )
