# Copyright 2026 Maxence Faldor
"""Plots of results: waveforms, DC sweeps, Bode plots, noise spectra and pole maps.

A plot draws chosen unknowns and logic nets of one result, or a loop gain alone, against
0 dB and -180°, where its margins are read, or a noise spectrum alone: the output's
noise density with its noisiest sources', and the noise referred to the input, on log
axes, or poles and zeros alone, each at its natural frequency and damping ratio.
Voltages, currents, temperatures, devices' states and logic levels never share an
axis: each quantity has a panel of its own, and a response has two, its magnitude and
its phase. Logic nets share a panel in lanes of their own, each named on its axis, with
unknown levels hatched. Series take their colors in a fixed order, checked to stay
apart under the common color-vision deficiencies; a legend names them, and up to four
are also labeled at their end.

A batch of runs plots their distribution, not each run: a histogram of each
measure's values over the runs, against its requirement's bounds, and an envelope
of each unknown it traced along an analysis, its least, median and greatest value over
the runs at each point.

Examples:
    >>> import numpy as np
    >>> from jolty import results
    >>> unknowns = (results.Unknown("out", "voltage"),)
    >>> statistics = results.Statistics("direct", 1, 0, 0, 2)
    >>> step = results.TransientResult(
    ...     unknowns, np.array([0.0, 1e-6]), np.array([[0.0], [1.0]]), statistics
    ... )
    >>> figure = plot("step", step, default_traces(unknowns))
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

from jolty import files, results
from jolty.errors import JoltyError
from jolty.units import format_si
from jolty.unknowns import default_traces as default_traces

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

SURFACE: Final = "#fcfcfb"
"""The surface jolty's plots are drawn on."""
INK: Final = "#0b0b0b"
"""The ink of their titles and text."""
MUTED: Final = "#52514e"
"""The muted ink of their axes' labels and reference lines."""
GRID: Final = "#e4e3df"
"""Their grid's color."""
_UNITS: Final = {"voltage": "V", "current": "A", "temperature": "K", "state": ""}

type Plotted = (
    results.DcSweepResult
    | results.AcResult
    | results.TransientResult
    | results.StabilityResult
    | results.NoiseResult
    | results.PoleZeroResult
)
"""A result a plot draws; an operating point has no axis to draw along."""

type _Quantity = Literal["voltage", "current", "temperature", "state", "logic"]

_LANE: Final = 1.5
"""The height of a logic net's lane, in levels: its level spans 1, and a gap follows."""


def plot(name: str, result: Plotted, traces: Sequence[str]) -> Figure:
    """The plot of the unknowns `traces` in the result of the analysis `name`.

    A loop gain's plot draws the loop gain alone, a noise spectrum's the noise alone,
    and a pole-zero analysis's its poles and zeros alone, whatever the traces.

    Raises:
        JoltyError: When a trace names no unknown, or there are more than
            [`MAX_SERIES`][jolty.plots.MAX_SERIES] traces.
    """
    if isinstance(result, results.StabilityResult):
        return _loop(name, result)
    if isinstance(result, results.NoiseResult):
        return _noise(name, result)
    if isinstance(result, results.PoleZeroResult):
        return _poles_zeros(name, result)
    if len(traces) > MAX_SERIES:
        message = f"a plot draws at most {MAX_SERIES} unknowns, not {len(traces)}"
        raise JoltyError(message)
    groups, columns = _grouped(result, traces)
    natures = list(groups)
    panels = 2 * len(natures) if isinstance(result, results.AcResult) else len(natures)
    figure, axes = _figure(name, result, panels)
    colors = {trace: COLORS[index] for index, trace in enumerate(traces)}
    x, x_unit = _abscissa(result)
    if isinstance(result, results.AcResult):
        for index, nature in enumerate(natures):
            phasors = [
                (trace, colors[trace], result.values[:, columns[trace]])
                for trace in groups[nature]
            ]
            magnitudes = [
                (trace, color, _decibels(values)) for trace, color, values in phasors
            ]
            phases = [
                (trace, color, np.degrees(np.unwrap(np.angle(values))))
                for trace, color, values in phasors
            ]
            _draw(axes[2 * index], x, magnitudes, f"{nature} magnitude", "dB")
            _draw(axes[2 * index + 1], x, phases, f"{nature} phase", "°")
        for each in axes:
            each.set_xscale("log")
    else:
        for index, nature in enumerate(natures):
            if nature == "logic":
                lanes = [
                    (trace, colors[trace], *_steps(result, trace))
                    for trace in groups[nature]
                ]
                _draw_logic(axes[index], lanes)
                continue
            series = [
                (trace, colors[trace], result.values[:, columns[trace]])
                for trace in groups[nature]
            ]
            label = "temperature rise" if nature == "temperature" else nature
            _draw(axes[index], x, series, label, _UNITS[nature])
    _label_abscissa(axes[-1], result, x_unit)
    return figure


def _grouped(
    result: results.DcSweepResult | results.AcResult | results.TransientResult,
    traces: Sequence[str],
) -> tuple[dict[_Quantity, list[str]], dict[str, int]]:
    """The `traces` by quantity, each quantity that has one, and each unknown's column.

    Raises:
        JoltyError: When a trace names neither an unknown nor a logic net.
    """
    nets = () if isinstance(result, results.AcResult) else result.nets
    columns = {}
    groups: dict[_Quantity, list[str]] = {
        "voltage": [],
        "current": [],
        "temperature": [],
        "state": [],
        "logic": [],
    }
    for trace in traces:
        if trace in nets:
            groups["logic"].append(trace)
            continue
        try:
            columns[trace] = results.column(result.unknowns, trace)
        except KeyError as error:
            raise JoltyError(error.args[0]) from None
        groups[result.unknowns[columns[trace]].nature].append(trace)
    return {nature: found for nature, found in groups.items() if found}, columns


def _loop(name: str, result: results.StabilityResult) -> Figure:
    """The Bode plot of a loop gain, against 0 dB and -180°, where its margins are read.

    Its frequencies are the positive ones, increasing, as a log axis needs.
    """
    order = np.argsort(result.frequencies)
    order = order[result.frequencies[order] > 0]
    frequencies, gains = result.frequencies[order], result.gains[order]
    figure, (magnitude, phase) = _figure(name, result, 2)
    label = f"T through {result.probe}"
    color = COLORS[0]
    _draw(magnitude, frequencies, [(label, color, _decibels(gains))], "magnitude", "dB")
    phases = np.degrees(np.unwrap(np.angle(gains)))
    _draw(phase, frequencies, [(label, color, phases)], "phase", "°")
    for axes, level in ((magnitude, 0.0), (phase, -180.0)):
        axes.axhline(level, color=MUTED, linewidth=0.8, linestyle="--")
        axes.set_xscale("log")
    _label_abscissa(phase, result, "Hz")
    return figure


def _noise(name: str, result: results.NoiseResult) -> Figure:
    """The noise density of the output, with its noisiest sources', on log axes.

    The sources are the noisiest over the frequencies, as many as colors remain; the
    noise referred to the input, given one, has a panel of its own, in its unit.
    """
    order = np.argsort(result.frequencies)
    frequencies = result.frequencies[order]
    powers = result.contributions[:, order]
    if len(frequencies) > 1:
        totals = [results.integrate(frequencies, row) for row in powers]
    else:
        totals = [float(row[0]) for row in powers]
    noisiest = sorted(range(len(totals)), key=lambda index: -totals[index])
    over = "" if result.reference is None else f" over {result.reference}"
    series = [(f"{result.output}{over}", COLORS[0], np.sqrt(result.densities[order]))]
    series.extend(
        (
            f"{result.sources[index].instance} {result.sources[index].name}",
            color,
            np.sqrt(powers[index]),
        )
        for index, color in zip(noisiest, COLORS[1:], strict=False)
    )
    figure, axes = _figure(name, result, 1 if result.input is None else 2)
    unit = f"{_UNITS[result.nature]}/\N{SQUARE ROOT}Hz"
    _draw(axes[0], frequencies, series, "noise density", unit)
    if result.input is not None:
        referred = np.sqrt(result.referred()[order])
        label = f"referred to {result.input.source}"
        _draw(
            axes[1],
            frequencies,
            [(label, COLORS[0], referred)],
            "input noise density",
            f"{_UNITS[result.input.nature]}/\N{SQUARE ROOT}Hz",
        )
    for each in axes:
        each.set_xscale("log")
        each.set_yscale("log")
    _label_abscissa(axes[-1], result, "Hz")
    return figure


def _poles_zeros(name: str, result: results.PoleZeroResult) -> Figure:
    """Each pole and zero at its natural frequency, on a log axis, and damping ratio.

    A damping ratio below zero is a root in the right half-plane, across the line at
    zero; a conjugate pair shares its point, and a zero at the origin has none.
    """
    figure, (axes,) = _figure(name, result, 1)
    _style(axes, "damping ratio")
    roots = [("poles", "x", COLORS[0], result.poles)]
    if result.zeros is not None:
        roots.append(("zeros", "o", COLORS[1], result.zeros))
    for label, marker, color, found in roots:
        shown = found[(found.imag >= 0) & (np.abs(found) > 0)]
        axes.plot(
            results.natural_frequencies(shown),
            results.dampings(shown),
            linestyle="none",
            marker=marker,
            markersize=8,
            markerfacecolor="none",
            color=color,
            label=label,
        )
    axes.axhline(0.0, color=MUTED, linewidth=0.8, linestyle="--")
    axes.set_xscale("log")
    legend = axes.legend(loc="best", frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color(INK)
    _label_abscissa(axes, result, "Hz")
    return figure


def _figure(name: str, result: Plotted, panels: int) -> tuple[Figure, list[Axes]]:
    """A figure titled for the result of the analysis `name`.

    It stacks `panels` axes, at least one, over a shared abscissa.
    """
    figure = Figure(
        figsize=(8, 0.8 + 2.4 * max(panels, 1)),
        dpi=150,
        layout="constrained",
        facecolor=SURFACE,
    )
    figure.suptitle(f"{name}: {_describe(result)}", color=INK, x=0.02, ha="left")
    figure.subplots(max(panels, 1), 1, sharex=True, squeeze=False)
    return figure, figure.axes


def _label_abscissa(axes: Axes, result: Plotted, unit: str) -> None:
    """Labels the abscissa of `axes`, the bottom panel, for `result`, in `unit`."""
    axes.xaxis.set_major_formatter(EngFormatter(unit=unit, sep=" "))
    axes.set_xlabel(_abscissa_label(result), color=MUTED)


def _steps(
    result: results.DcSweepResult | results.TransientResult, net: str
) -> tuple[results.Real, results.Real]:
    """Where the level of the logic net `net` changes, and the level from each on.

    A transient's last level holds until it stops; a sweep's levels change halfway
    between the source's levels.
    """
    if isinstance(result, results.TransientResult):
        trace = result.logic_level(net)
        return (
            np.append(trace.times, result.times[-1]),
            np.append(trace.levels, trace.levels[-1]),
        )
    levels = result.logic_level(net)
    edges = np.concatenate(
        [
            result.swept[:1],
            (result.swept[1:] + result.swept[:-1]) / 2,
            result.swept[-1:],
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


def histograms(name: str, batch: files.Batch) -> Figure:
    """A histogram of each measure of `batch`, the batch `name`, over its runs.

    Each panel counts the runs whose value lies in each bin, numpy's ``auto`` bins of
    the values read, and draws the requirement's bounds across it; its title says how
    many runs read no value.
    """
    found = batch.result
    figure = Figure(
        figsize=(8, 0.8 + 2.4 * max(len(found.measures), 1)),
        dpi=150,
        layout="constrained",
        facecolor=SURFACE,
    )
    figure.suptitle(
        f"{name}: {found.runs} runs, each measure's values",
        color=INK,
        x=0.02,
        ha="left",
    )
    for place, requirement in enumerate(batch.requirements):
        axes = figure.add_subplot(len(batch.requirements), 1, place + 1)
        measured = found.measured[:, place]
        measured = measured[~np.isnan(measured) & (found.causes[:, place] < 0)]
        read = measured[np.isfinite(measured)]
        unit = found.units[place]
        symbol = "" if unit is None or unit == "" else str(unit)
        if read.size:
            edges = np.histogram_bin_edges(read, bins="auto")
            # An edge in the bars' color draws a bin narrower than a pixel, as one is
            # beside a bound far from every value.
            axes.hist(
                read,
                bins=edges.tolist(),
                color=COLORS[0],
                edgecolor=COLORS[0],
                linewidth=0.8,
            )
        for bound in batch.bounds[place]:
            if bound is not None:
                axes.axvline(bound, color=MUTED, linewidth=0.8, linestyle="--")
        _style(axes, "runs")
        unread = found.runs - measured.size
        missing = f", {unread} read none" if unread else ""
        # An infinite reading, as a gain of exactly nothing, has no bin to fall in.
        infinite = "".join(
            f", {count} read {sign}inf"
            for sign, count in (
                ("-", int((measured == -np.inf).sum())),
                ("", int((measured == np.inf).sum())),
            )
            if count
        )
        axes.set_title(
            f"{requirement}{missing}{infinite}", color=INK, fontsize=9, loc="left"
        )
        if symbol in {"dB", "°", "%"}:
            axes.set_xlabel(symbol, color=MUTED)
        elif symbol:
            axes.xaxis.set_major_formatter(EngFormatter(unit=symbol, sep=" "))
    return figure


def envelope(name: str, traces: results.Traces) -> Figure:
    """The envelope of each unknown `traces` along an analysis of the batch `name`.

    Each draws its least and its greatest value over the runs at each point as a band,
    and its median as a line, the runs that did not end the analysis left out; a
    response draws its magnitude and its phase, each run's phase unwrapped first. The
    unknowns of each quantity share a panel, as a plot's do.
    """
    natures = list(dict.fromkeys(unknown.nature for unknown in traces.unknowns))
    response = traces.along == "frequency"
    panels = 2 * len(natures) if response else len(natures)
    figure = Figure(
        figsize=(8, 0.8 + 2.4 * max(panels, 1)),
        dpi=150,
        layout="constrained",
        facecolor=SURFACE,
    )
    runs = int(traces.values.shape[0])
    figure.suptitle(
        f"{name}: {traces.analysis}, the least, median and greatest of {runs} runs",
        color=INK,
        x=0.02,
        ha="left",
    )
    figure.subplots(max(panels, 1), 1, sharex=True, squeeze=False)
    axes = figure.axes
    colors = {
        unknown.name: COLORS[index % len(COLORS)]
        for index, unknown in enumerate(traces.unknowns)
    }
    for index, nature in enumerate(natures):
        columns = [
            (place, unknown)
            for place, unknown in enumerate(traces.unknowns)
            if unknown.nature == nature
        ]
        if response:
            phasors = [
                (
                    unknown.name,
                    colors[unknown.name],
                    np.asarray(traces.values[:, :, place], dtype=np.complex128),
                )
                for place, unknown in columns
            ]
            magnitudes = [
                (name, color, _decibels(values)) for name, color, values in phasors
            ]
            phases = [
                (name, color, np.degrees(np.unwrap(np.angle(values), axis=1)))
                for name, color, values in phasors
            ]
            _band(axes[2 * index], traces.axis, magnitudes, f"{nature} magnitude", "dB")
            _band(axes[2 * index + 1], traces.axis, phases, f"{nature} phase", "°")
            continue
        series = [
            (unknown.name, colors[unknown.name], np.real(traces.values[:, :, place]))
            for place, unknown in columns
        ]
        label = "temperature rise" if nature == "temperature" else nature
        _band(axes[index], traces.axis, series, label, _UNITS[nature])
    if response:
        for each in axes:
            each.set_xscale("log")
    unit = {"time": "s", "frequency": "Hz"}.get(traces.along, "")
    if traces.along == "swept" and traces.nature is not None:
        unit = _UNITS[traces.nature]
    axes[-1].xaxis.set_major_formatter(EngFormatter(unit=unit, sep=" "))
    along = {
        "time": "time",
        "frequency": "frequency",
        "swept": _sweep_label(traces.source, traces.parameter, traces.nature),
        "point": "",
    }[traces.along]
    axes[-1].set_xlabel(along, color=MUTED)
    return figure


def _band(
    axes: Axes,
    x: results.Real,
    series: list[tuple[str, str, results.Real]],
    label: str,
    unit: str,
) -> None:
    """Draws `series`, each a name, a color and its values, one row per run, along `x`.

    Each is a band from its least to its greatest value at each point, and a line at
    its median, over the runs whose values are numbers.
    """
    for name, color, values in series:
        ended = values[~np.isnan(values).any(axis=1)]
        if not ended.size:
            continue
        low, middle, high = (
            ended.min(axis=0),
            np.median(ended, axis=0),
            ended.max(axis=0),
        )
        axes.fill_between(x, low, high, color=color, alpha=0.25, linewidth=0)
        axes.plot(x, middle, color=color, linewidth=1.0, label=name)
    _style(axes, label)
    if unit:
        axes.yaxis.set_major_formatter(EngFormatter(unit=unit, sep=" "))
    if len(series) > 1:
        legend = axes.legend(loc="best", frameon=False, fontsize=8)
        for text in legend.get_texts():
            text.set_color(INK)


def save(figure: Figure, path: Path) -> None:
    """Writes `figure` to `path` as a PNG image."""
    figure.savefig(path, format="png", facecolor=figure.get_facecolor())


def _describe(result: Plotted) -> str:
    match result:
        case results.TransientResult():
            return (
                f"periodic steady state of period {format_si(result.period, 's')}"
                if isinstance(result, results.PeriodicSteadyStateResult)
                else "transient"
            )
        case results.DcSweepResult():
            swept = "" if result.parameter is None else f"'s {result.parameter}"
            return f"DC sweep of {result.source}{swept}"
        case results.AcResult():
            return f"response to a unit excitation of {result.source}"
        case results.StabilityResult():
            return f"loop gain through {result.probe}"
        case results.NoiseResult():
            over = "" if result.reference is None else f" over {result.reference}"
            return f"noise at {result.output}{over}"
        case results.PoleZeroResult():
            over = "" if result.reference is None else f" over {result.reference}"
            zeros = f", and zeros of {result.output}{over} per {result.source}"
            return "poles" + ("" if result.zeros is None else zeros)


def _abscissa(result: Plotted) -> tuple[results.Real, str]:
    match result:
        case results.TransientResult():
            return result.times, "s"
        case results.DcSweepResult():
            unit = "" if result.nature is None else _UNITS[result.nature]
            return result.swept, unit
        case results.AcResult() | results.StabilityResult() | results.NoiseResult():
            return result.frequencies, "Hz"
        case results.PoleZeroResult():
            return results.natural_frequencies(result.poles), "Hz"


def _sweep_label(
    source: str | None, parameter: str | None, nature: results.Nature | None
) -> str:
    """Formats the horizontal axis label for a DC sweep."""
    kind = parameter or nature
    if source and kind:
        return f"{source} {kind}"
    if source:
        return source
    return kind or "swept"


def _abscissa_label(result: Plotted) -> str:
    match result:
        case results.TransientResult():
            return "time"
        case results.DcSweepResult():
            return _sweep_label(result.source, result.parameter, result.nature)
        case results.AcResult() | results.StabilityResult() | results.NoiseResult():
            return "frequency"
        case results.PoleZeroResult():
            return "natural frequency"


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
    if unit:
        axes.yaxis.set_major_formatter(EngFormatter(unit=unit, sep=" "))
    if len(series) > 1:
        legend = axes.legend(loc="best", frameon=False, fontsize=8)
        for text in legend.get_texts():
            text.set_color(INK)
    if len(series) <= _LABELED:
        _label_ends(
            axes, [(name, color, float(values[-1])) for name, color, values in series]
        )


def _style(axes: Axes, label: str) -> None:
    """Gives `axes` its surface, its grid and its spines, and labels it `label`."""
    axes.set_facecolor(SURFACE)
    axes.set_ylabel(label, color=MUTED)
    axes.grid(visible=True, color=GRID, linewidth=0.6)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(MUTED)
        axes.spines[side].set_linewidth(0.6)
    axes.tick_params(colors=MUTED, labelsize=8)


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
            color=INK,
            fontsize=8,
            va="center",
        )
