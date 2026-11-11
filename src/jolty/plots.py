# Copyright 2026 Maxence Faldor
"""Plots of results: waveforms over time, DC sweeps over their levels, and Bode plots.

A plot draws chosen unknowns of one result. Voltages and currents never share an axis:
each quantity has a panel of its own, and a response has two, its magnitude and its
phase. Series take their colors in a fixed order, checked to stay apart under the common
color-vision deficiencies; a legend names them, and up to four are also labeled at
their end.

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

type _Quantity = Literal["voltage", "current"]


def default_probes(unknowns: Sequence[results.Unknown]) -> tuple[str, ...]:
    """The unknowns a plot draws unless told otherwise: the top component's named nets.

    A net the script named at the top has no slash in its path, and no dot, which only
    the names the builder gives unnamed nets carry. Without such nets, a plot draws the
    first voltages. Either way, it draws at most [`MAX_SERIES`][jolty.plots.MAX_SERIES].
    """
    voltages = [unknown.name for unknown in unknowns if unknown.quantity == "voltage"]
    named = [name for name in voltages if "/" not in name and "." not in name]
    return tuple((named or voltages)[:MAX_SERIES])


def plot(name: str, result: Plotted, probes: Sequence[str]) -> Figure:
    """The plot of the unknowns `probes` in the result of the analysis `name`.

    Raises:
        JoltyError: When a probe names no unknown, or there are more than
            [`MAX_SERIES`][jolty.plots.MAX_SERIES] probes.
    """
    if len(probes) > MAX_SERIES:
        message = f"a plot draws at most {MAX_SERIES} unknowns, not {len(probes)}"
        raise JoltyError(message)
    columns = {}
    for probe in probes:
        try:
            columns[probe] = results.column(result.unknowns, probe)
        except KeyError as error:
            raise JoltyError(error.args[0]) from None
    groups: dict[_Quantity, list[str]] = {"voltage": [], "current": []}
    for probe, column in columns.items():
        groups[result.unknowns[column].quantity].append(probe)
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
            series = [
                (probe, colors[probe], result.values[:, columns[probe]])
                for probe in groups[quantity]
            ]
            _draw(axes[index], x, series, quantity, _UNITS[quantity])
    bottom = axes[-1]
    bottom.xaxis.set_major_formatter(EngFormatter(unit=x_unit, sep=" "))
    bottom.set_xlabel(_abscissa_label(result), color=_MUTED)
    return figure


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
    axes.set_facecolor(_SURFACE)
    axes.set_ylabel(label, color=_MUTED)
    axes.yaxis.set_major_formatter(EngFormatter(unit=unit, sep=" "))
    axes.grid(visible=True, color=_GRID, linewidth=0.6)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(_MUTED)
        axes.spines[side].set_linewidth(0.6)
    axes.tick_params(colors=_MUTED, labelsize=8)
    if len(series) > 1:
        legend = axes.legend(loc="best", frameon=False, fontsize=8)
        for text in legend.get_texts():
            text.set_color(_INK)
    if len(series) <= _LABELED:
        _label_ends(
            axes, [(name, color, float(values[-1])) for name, color, values in series]
        )


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
