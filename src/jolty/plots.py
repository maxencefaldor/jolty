# Copyright 2026 Maxence Faldor
"""Result plotting for waveforms, DC sweeps, Bode plots, noise spectra, and pole maps.

Single-result plots render selected unknowns and logic nets for transient, DC sweep,
and AC analyses. Stability plots render loop gain against 0 dB and -180° reference
lines for phase and gain margins. Noise plots display output noise density alongside
top contributors and optional input-referred noise on log-log axes. Pole-zero plots
place each pole and zero by natural frequency and damping ratio.

Each physical quantity (voltage, current, temperature, device state, and logic level)
occupies a dedicated panel, with AC responses split into magnitude and phase panels.
Logic nets share a panel in separate horizontal lanes, labeled on the vertical axis
with hatched bands for unknown levels. Series use a fixed color palette designed for
distinguishability under common color-vision deficiencies, with legends and end labels
for up to four traces.

Batch plots summarize run distributions instead of individual traces. Histograms show
measured requirement values against specification bounds, while envelopes plot the
minimum, median, and maximum trace values across completed runs.

Install the optional extra with ``pip install 'jolty[plots]'`` to use this module.

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

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, Literal

import numpy as np

try:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.ticker import EngFormatter
except ModuleNotFoundError as error:
    root_module = (error.name or "").split(".", 1)[0]
    if root_module != "matplotlib":
        raise
    message = (
        "jolty.plots needs the optional `matplotlib` dependency; "
        "install it with `pip install 'jolty[plots]'`"
    )
    raise ImportError(message) from error

from jolty import files, results, unknowns
from jolty.errors import JoltyError
from jolty.units import format_si
from jolty.unknowns import default_traces

__all__ = [
    "COLORS",
    "GRID",
    "INK",
    "MAX_SERIES",
    "MUTED",
    "SURFACE",
    "Plotted",
    "default_traces",
    "envelope",
    "histograms",
    "plot",
    "save",
]

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
"""Ordered palette of series colors chosen to remain distinguishable under common
color-vision deficiencies."""

MAX_SERIES: Final = len(COLORS)
"""Maximum number of unknown traces rendered in a single plot."""

_LABELED: Final = 4
"""Maximum number of series labeled directly at the right edge of a panel."""

SURFACE: Final = "#fcfcfb"
"""Background surface color for jolty plots."""
INK: Final = "#0b0b0b"
"""Primary foreground ink color for plot titles and labels."""
MUTED: Final = "#52514e"
"""Secondary muted color for axis labels, spines, and reference lines."""
GRID: Final = "#e4e3df"
"""Grid line color for plot panels."""

type Plotted = (
    results.DcSweepResult
    | results.AcResult
    | results.TransientResult
    | results.StabilityResult
    | results.NoiseResult
    | results.PoleZeroResult
)
"""Analysis result types that have an axis to plot; excludes operating points."""

type _Quantity = Literal["voltage", "current", "temperature", "state", "logic"]
type _Drawer = Callable[
    [Axes, results.Real, list[tuple[str, str, results.Real]], str, str], None
]

_LANE: Final = 1.5
"""Vertical lane height for each logic net, combining a 1.0 level span and gap."""


def plot(name: str, result: Plotted, traces: Sequence[str]) -> Figure:
    """Builds a figure plotting `traces` from the `name` analysis `result`.

    Stability, noise, and pole-zero results always render their dedicated layouts
    regardless of `traces`.

    Args:
        name: Analysis name shown in the figure title.
        result: Simulation analysis result to render.
        traces: Unknown or logic net names to include in the plot.

    Returns:
        Rendered Matplotlib `Figure` plotting `result`.

    Raises:
        JoltyError: If a trace does not match an unknown or logic net, or if `traces`
            exceeds [`MAX_SERIES`][jolty.plots.MAX_SERIES].
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
    figure, axes = _new_figure(f"{name}: {_describe(result)}", max(panels, 1))
    colors = {trace: COLORS[index] for index, trace in enumerate(traces)}
    x, x_unit = _abscissa(result)
    if isinstance(result, results.AcResult):
        for index, nature in enumerate(natures):
            phasors = [
                (trace, colors[trace], result.values[:, columns[trace]])
                for trace in groups[nature]
            ]
            _bode_panels(
                axes[2 * index : 2 * index + 2], x, phasors, f"{nature} ", _draw
            )
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
            _draw(axes[index], x, series, label, unknowns.SYMBOLS[nature])
    _label_abscissa(axes[-1], result, x_unit)
    return figure


def _grouped(
    result: results.DcSweepResult | results.AcResult | results.TransientResult,
    traces: Sequence[str],
) -> tuple[dict[_Quantity, list[str]], dict[str, int]]:
    """Groups `traces` by physical quantity and maps each unknown to its column index.

    Raises:
        JoltyError: If a trace matches neither an unknown nor a logic net in `result`.
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
    """Builds a Bode plot of loop gain against 0 dB and -180° reference lines.

    Positive frequencies are sorted in ascending order for the logarithmic frequency
    axis.
    """
    order = np.argsort(result.frequencies)
    order = order[result.frequencies[order] > 0]
    frequencies, gains = result.frequencies[order], result.gains[order]
    figure, axes = _new_figure(f"{name}: {_describe(result)}", 2)
    _bode_panels(
        axes, frequencies, [(f"T through {result.probe}", COLORS[0], gains)], "", _draw
    )
    for panel, level in zip(axes, (0.0, -180.0), strict=True):
        panel.axhline(level, color=MUTED, linewidth=0.8, linestyle="--")
        panel.set_xscale("log")
    _label_abscissa(axes[1], result, "Hz")
    return figure


def _noise(name: str, result: results.NoiseResult) -> Figure:
    """Builds a log-log plot of output noise density and top noise contributors.

    Contributor traces select the highest-integrated noise sources up to the remaining
    palette slots. When `result` includes an input reference, a second panel plots the
    input-referred noise density in the source's unit.
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
    panels = 1 if result.input is None else 2
    figure, axes = _new_figure(f"{name}: {_describe(result)}", panels)
    unit = f"{unknowns.SYMBOLS[result.nature]}/\N{SQUARE ROOT}Hz"
    _draw(axes[0], frequencies, series, "noise density", unit)
    if result.input is not None:
        referred = np.sqrt(result.referred()[order])
        label = f"referred to {result.input.source}"
        input_unit = f"{unknowns.SYMBOLS[result.input.nature]}/\N{SQUARE ROOT}Hz"
        _draw(
            axes[1],
            frequencies,
            [(label, COLORS[0], referred)],
            "input noise density",
            input_unit,
        )
    for each in axes:
        each.set_xscale("log")
        each.set_yscale("log")
    _label_abscissa(axes[-1], result, "Hz")
    return figure


def _poles_zeros(name: str, result: results.PoleZeroResult) -> Figure:
    """Builds a scatter plot of poles and optional zeros by frequency and damping ratio.

    Negative damping ratios indicate right-half-plane roots below the zero reference
    line. Conjugate pairs share a single point, and roots at the origin are omitted.
    """
    figure, (axes,) = _new_figure(f"{name}: {_describe(result)}", 1)
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
    _add_legend(axes)
    _label_abscissa(axes, result, "Hz")
    return figure


def _new_figure(
    title: str, panels: int, *, sharex: bool = True
) -> tuple[Figure, list[Axes]]:
    """Creates a figure titled `title` with `panels` vertically stacked axes."""
    figure = Figure(
        figsize=(8, 0.8 + 2.4 * max(panels, 1)),
        dpi=150,
        layout="constrained",
        facecolor=SURFACE,
    )
    figure.suptitle(title, color=INK, x=0.02, ha="left")
    if panels > 0:
        figure.subplots(panels, 1, sharex=sharex, squeeze=False)
    return figure, figure.axes


def _add_legend(axes: Axes) -> None:
    """Adds a frameless legend colored in ink to `axes`."""
    legend = axes.legend(loc="best", frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color(INK)


def _label_abscissa(axes: Axes, result: Plotted, unit: str) -> None:
    """Formats the horizontal axis of `axes` for `result` using engineering `unit`."""
    axes.xaxis.set_major_formatter(EngFormatter(unit=unit, sep=" "))
    axes.set_xlabel(_abscissa_label(result), color=MUTED)


def _steps(
    result: results.DcSweepResult | results.TransientResult, net: str
) -> tuple[results.Real, results.Real]:
    """Computes step-plot coordinates and logic levels for `net` in `result`.

    Transient logic levels hold their final value through the end time, while DC sweep
    transitions occur midway between swept source values.
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
    """Draws logic net waveforms from `lanes` top to bottom with hatched unknown bands.

    Each lane entry specifies a net name, line color, transition coordinates, and logic
    levels.
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


def _bode(phasors: results.Complex) -> tuple[results.Real, results.Real]:
    """Computes decibel magnitude (floored at -6000 dB) and unwrapped degree phase."""
    magnitude = np.asarray(
        20 * np.log10(np.maximum(np.abs(phasors), 1e-300)), dtype=np.float64
    )
    phase = np.asarray(
        np.degrees(np.unwrap(np.angle(phasors), axis=-1)), dtype=np.float64
    )
    return magnitude, phase


def _bode_panels(
    axes: Sequence[Axes],
    x: results.Real,
    phasors: list[tuple[str, str, results.Complex]],
    prefix: str,
    draw: _Drawer,
) -> None:
    """Draws magnitude (dB) and unwrapped phase (°) panels for `phasors`."""
    magnitudes: list[tuple[str, str, results.Real]] = []
    phases: list[tuple[str, str, results.Real]] = []
    for label, color, values in phasors:
        magnitude, phase = _bode(values)
        magnitudes.append((label, color, magnitude))
        phases.append((label, color, phase))
    draw(axes[0], x, magnitudes, f"{prefix}magnitude", "dB")
    draw(axes[1], x, phases, f"{prefix}phase", "°")


def histograms(name: str, batch: files.Batch) -> Figure:
    """Builds histograms for each requirement measure across runs in `batch`.

    Each panel bins finite readings with numpy's ``auto`` rule and marks specification
    bounds with dashed lines. Panel titles report counts of missing or infinite
    readings.

    Args:
        name: Batch name shown in the figure title.
        batch: Saved batch result paired with requirement labels and bounds.

    Returns:
        Rendered Matplotlib `Figure` with one histogram panel per requirement.
    """
    found = batch.result
    title = f"{name}: {found.runs} runs, each measure's values"
    figure, axes_list = _new_figure(title, len(batch.requirements), sharex=False)
    for place, (axes, requirement) in enumerate(
        zip(axes_list, batch.requirements, strict=True)
    ):
        measured = found.measured[:, place]
        measured = measured[~np.isnan(measured) & (found.causes[:, place] < 0)]
        read = measured[np.isfinite(measured)]
        unit = found.units[place]
        symbol = "" if unit is None or unit == "" else str(unit)
        if read.size:
            edges = np.histogram_bin_edges(read, bins="auto")
            # Matching bar edge color keeps sub-pixel bins visible when a distant bound
            # widens the axis range.
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
        # Infinite readings, such as zero gain in decibels, cannot be placed in bins.
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
    """Builds min-median-max envelope panels for each traced unknown in `traces`.

    Each series shades the minimum-to-maximum band and plots the median across runs
    that completed the analysis, omitting incomplete runs. Frequency-domain responses
    unwrap each run's phase before rendering magnitude and phase panels grouped by
    physical quantity.

    Args:
        name: Batch name shown in the figure title.
        traces: Batched unknown waveforms recorded along a shared analysis axis.

    Returns:
        Rendered Matplotlib `Figure` shading min-to-max bands around median curves.
    """
    natures = list(dict.fromkeys(unknown.nature for unknown in traces.unknowns))
    response = traces.along == "frequency"
    panels = 2 * len(natures) if response else len(natures)
    runs = int(traces.values.shape[0])
    title = f"{name}: {traces.analysis}, the least, median and greatest of {runs} runs"
    figure, axes = _new_figure(title, max(panels, 1))
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
            _bode_panels(
                axes[2 * index : 2 * index + 2],
                traces.axis,
                phasors,
                f"{nature} ",
                _band,
            )
            continue
        series = [
            (unknown.name, colors[unknown.name], np.real(traces.values[:, :, place]))
            for place, unknown in columns
        ]
        label = "temperature rise" if nature == "temperature" else nature
        _band(axes[index], traces.axis, series, label, unknowns.SYMBOLS[nature])
    if response:
        for each in axes:
            each.set_xscale("log")
    unit = {"time": "s", "frequency": "Hz"}.get(traces.along, "")
    if traces.along == "swept" and traces.nature is not None:
        unit = unknowns.SYMBOLS[traces.nature]
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
    """Draws min-max bands and median curves for `series` along `x` on `axes`.

    Each series entry contains a label, color, and 2D array with one row per run,
    filtering out runs containing NaN values.
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
        _add_legend(axes)


def save(figure: Figure, path: Path) -> None:
    """Writes `figure` to `path` as a PNG image.

    Args:
        figure: Matplotlib figure to export.
        path: Destination PNG file path.
    """
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
            unit = "" if result.nature is None else unknowns.SYMBOLS[result.nature]
            return result.swept, unit
        case results.AcResult() | results.StabilityResult() | results.NoiseResult():
            return result.frequencies, "Hz"
        case results.PoleZeroResult():
            return results.natural_frequencies(result.poles), "Hz"


def _sweep_label(
    source: str | None, parameter: str | None, nature: results.Nature | None
) -> str:
    """Formats the horizontal axis label for a DC sweep."""
    kind = parameter if parameter is not None and parameter != "" else nature
    parts = [part for part in (source, kind) if part is not None and part != ""]
    return " ".join(parts) if parts else "swept"


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
    """Draws named `(label, color, values)` curves from `series` along `x` on `axes`."""
    for name, color, values in series:
        axes.plot(x, values, color=color, linewidth=1.0, label=name)
    _style(axes, label)
    if unit:
        axes.yaxis.set_major_formatter(EngFormatter(unit=unit, sep=" "))
    if len(series) > 1:
        _add_legend(axes)
    if len(series) <= _LABELED:
        _label_ends(
            axes, [(name, color, float(values[-1])) for name, color, values in series]
        )


def _style(axes: Axes, label: str) -> None:
    """Applies surface color, grid, spine styling, and y-axis `label` to `axes`."""
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
    """Annotates each series at the right edge of `axes` near its final y-coordinate.

    Overlapping labels are shifted vertically apart, each preceded by a short swatch
    line in the series color.
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
