# Copyright 2026 Maxence Faldor
"""Standalone 3D HTML viewer for orbiting a circuit and scrubbing simulation results.

The viewer arranges each component on a 3D board matching its schematic layout: devices
stand on pins, wires appear as traces, and ground and supply pins appear as posts.
Simulation playback colors traces by net voltage, animates lead current flow, and
illuminates devices by dissipated power. A timeline scrubs through every analysis point;
clicking a part inspects parameters, terminal states, variables, and conductances, while
double-clicking opens a child component. Per-device currents, charges, variables, and
conductances appear for instances recorded with ``record=["**"]``.

The generated HTML embeds the circuit, layouts, results, script, and styles, and loads
three.js from jsDelivr (requiring network access unless a local mirror is configured).

Examples:
    >>> from jolty import Builder, OperatingPoint, Role, Simulation, circuit, component
    >>> from jolty import devices
    >>> from jolty.units import V, kohm
    >>> @component
    ... def divider(c: Builder) -> None:
    ...     ground = c.port("ground", Role.GROUND)
    ...     source = c.place(devices.voltage_source(5 * V), name="V1")
    ...     top = c.place(devices.resistor(1 * kohm), name="R1")
    ...     bottom = c.place(devices.resistor(1 * kohm), name="R2")
    ...     c.connect(source.positive, top.a, name="in")
    ...     c.connect(top.b, bottom.a, name="out")
    ...     c.connect(bottom.b, source.negative, ground)
    >>> drawn = circuit(divider())
    >>> point = Simulation(drawn).run(OperatingPoint())
    >>> data = scene(drawn, {"op": point})
    >>> data["analyses"][0]["nets"]["out"]
    [2.5]
"""

import html
import json
import math
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import Final

import numpy as np
from pydantic import JsonValue

from jolty import ir, layout, results, unknowns

THREE_VERSION: Final = "0.170.0"
"""Version of three.js loaded by generated viewer pages."""

_THREE: Final = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}"
"""Base jsDelivr CDN URL for the pinned three.js package."""

FRAMES: Final = 240
"""Maximum sampled animation frames per analysis in the 3D viewer."""

_DIGITS: Final = 5
"""Significant digits retained for serialized viewer numbers."""


def _number(value: float) -> JsonValue:
    """Rounds finite `value` to `_DIGITS` significant digits, or returns None."""
    if not math.isfinite(value):
        return None
    if value == 0:
        return 0.0
    return float(f"{value:.{_DIGITS}g}")


def _series(values: np.ndarray) -> JsonValue:
    """Converts numeric array `values` into rounded JSON-safe numbers."""
    return [_number(float(value)) for value in values]


def _frames(count: int) -> np.ndarray:
    """Selects up to `FRAMES` evenly spaced point indices, preserving endpoints."""
    if count <= FRAMES:
        return np.arange(count)
    return np.unique(np.round(np.linspace(0, count - 1, FRAMES)).astype(int))


def _layout(circuit: ir.Circuit, name: str) -> dict[str, JsonValue]:
    """Builds the serialized 3D layout geometry for component `name` in `circuit`."""
    drawn = layout.layout(circuit, name)
    instances: list[JsonValue] = []
    for placed in drawn.instances:
        instance = placed.instance
        kind = (
            {"model": instance.of.model}
            if isinstance(instance.of, ir.DeviceOf)
            else {"component": instance.of.component}
        )
        (left, bottom), (right, top) = placed.body()
        instances.append(
            {
                "name": placed.name,
                **kind,
                "pins": {
                    terminal: list(placed.pin(terminal))
                    for terminal in placed.shape.pins
                },
                "body": [[left, bottom], [right, top]],
                "turns": placed.orientation.turns,
                "flipped": placed.orientation.flipped,
                "terminals": dict(instance.terminals),
                "parameters": instance.parameters,
            }
        )
    nets: list[JsonValue] = [
        {
            "name": net.name,
            "role": net.role.value,
            "pins": [list(point) for point in net.pins],
            "segments": [[list(start), list(end)] for start, end in net.segments],
            "junctions": [list(point) for point in net.junctions],
            "labeled": net.labeled,
        }
        for net in drawn.nets
    ]
    flags: list[JsonValue] = [
        {"name": flag.name, "point": list(flag.point), "side": flag.side}
        for flag in drawn.flags
    ]
    component = circuit.components[name]
    return {
        "instances": instances,
        "nets": nets,
        "flags": flags,
        "ports": [
            {"name": port.name, "role": port.role.value} for port in component.ports
        ],
    }


def _views(circuit: ir.Circuit) -> dict[str, JsonValue]:
    """Maps every component instance path to its component definition and circuit nets.

    The top-level component uses the empty path ``""``. Each view maps local net names
    to their resolved full-circuit net names.
    """
    views: dict[str, JsonValue] = {}

    def walk(name: str, prefix: str, outside: Mapping[str, str]) -> None:
        component = circuit.components[name]
        local: dict[str, JsonValue] = {}
        named: dict[str, str] = {}
        for net in [
            *(
                net
                for instance in component.instances
                for net in instance.terminals.values()
            ),
            *(port.name for port in component.ports),
        ]:
            named.setdefault(net, outside.get(net, f"{prefix}{net}"))
        local.update(named)
        views[prefix.removesuffix("/")] = {"component": name, "nets": local}
        for instance in component.instances:
            if isinstance(instance.of, ir.ComponentOf):
                inner = {port: named[net] for port, net in instance.terminals.items()}
                walk(instance.of.component, f"{prefix}{instance.name}/", inner)

    top = circuit.components[circuit.top]
    walk(circuit.top, "", {port.name: port.name for port in top.ports})
    return views


type _RecordedResult = (
    results.OperatingPointResult | results.DcSweepResult | results.TransientResult
)

_RECORDED: Final = (
    results.OperatingPointResult,
    results.DcSweepResult,
    results.TransientResult,
)
"""Analysis result types that record per-device terminal and state arrays."""


def _axis(result: results.AnalysisResult) -> tuple[str, np.ndarray, str, str] | None:
    """Extracts the analysis kind, sweep axis values, axis label, and unit."""
    match result:
        case results.OperatingPointResult():
            return "operating_point", np.zeros(1), "", ""
        case results.DcSweepResult() if result.parameter is not None:
            return "dc_sweep", result.swept, f"{result.source} {result.parameter}", ""
        case results.DcSweepResult():
            unit = "" if result.nature is None else unknowns.SYMBOLS[result.nature]
            return "dc_sweep", result.swept, f"{result.source} level", unit
        case results.TransientResult():
            periodic = isinstance(result, results.PeriodicSteadyStateResult)
            kind = "periodic_steady_state" if periodic else "transient"
            return kind, result.times, "time", "s"
        case results.AcResult():
            return "ac", result.frequencies, "frequency", "Hz"
    return None


def _rows(result: _RecordedResult, field: str) -> np.ndarray:
    """Extracts a 2-D per-point array for `field` from `result`, even for OP."""
    values = np.asarray(getattr(result, field))
    return (
        values[None, :] if isinstance(result, results.OperatingPointResult) else values
    )


def _levels(result: results.AnalysisResult, axis: np.ndarray) -> dict[str, np.ndarray]:
    """Extracts per-point logic levels for every logic net in `result`."""
    match result:
        case results.OperatingPointResult():
            return {net: result.logic[[index]] for index, net in enumerate(result.nets)}
        case results.DcSweepResult():
            return {
                net: result.logic[:, index] for index, net in enumerate(result.nets)
            }
        case results.TransientResult():
            return {
                net: trace.at(axis)
                for net, trace in zip(result.nets, result.logic, strict=True)
            }
    return {}


def _currents(
    recording: results.Recording, frames: np.ndarray, currents: np.ndarray
) -> dict[str, JsonValue]:
    """Extracts electrical terminal current series sampled at `frames`.

    Component and logic terminals use recorded summed currents; thermal terminals are
    excluded because heat flow is not rendered as electrical current.
    """
    terminals: dict[str, JsonValue] = {}
    if not currents.size:
        return terminals
    sampled = currents[frames]
    for index, terminal in enumerate(recording.terminals):
        if terminal not in recording.thermal:
            terminals[terminal] = _series(sampled[:, index])
    return terminals


def _state(result: _RecordedResult, frames: np.ndarray) -> dict[str, JsonValue]:
    """Extracts per-device charges, internal variables, and conductances at `frames`.

    Charges are keyed by terminal path, and small-signal conductances are keyed by
    terminal pairs.
    """
    recording = result.recording
    stored = _rows(result, "charges")
    computed = _rows(result, "variables")
    conductances = _rows(result, "conductances")
    charges: dict[str, JsonValue] = {}
    variables: dict[str, dict[str, JsonValue]] = {}
    small_signal: dict[str, JsonValue] = {}
    terminal, variable, derivative = 0, 0, 0
    for device in recording.devices:
        names = [f"{device.path}.{own}" for own in device.terminals]
        if stored.size:
            for name in names:
                charges[name] = _series(stored[frames, terminal])
                terminal += 1
        if computed.size:
            for own in device.variables:
                values = _series(computed[frames, variable])
                variables.setdefault(device.path, {})[own] = values
                variable += 1
        size = len(device.unknowns)
        if conductances.size:
            for row, first in enumerate(names):
                for column, second in enumerate(names):
                    index = derivative + row * size + column
                    small_signal[f"{first}|{second}"] = _series(
                        conductances[frames, index]
                    )
        derivative += size * size
    found: dict[str, JsonValue] = dict(variables)
    return {"charges": charges, "variables": found, "conductances": small_signal}


def _analysis(name: str, result: results.AnalysisResult) -> dict[str, JsonValue] | None:
    """Serializes one analysis result sampled at viewer frames, or returns None.

    AC analyses render phasor magnitudes across frequency.
    """
    found = _axis(result)
    if found is None:
        return None
    kind, axis, label, unit = found
    frames = _frames(len(axis))
    devices: dict[str, JsonValue]
    if isinstance(result, results.AcResult):
        values = np.abs(result.values)
        currents = np.abs(result.currents)
        devices = {"currents": _currents(result.recording, frames, currents)}
    elif isinstance(result, _RECORDED):
        values = _rows(result, "values")
        currents = _rows(result, "currents")
        devices = {
            "currents": _currents(result.recording, frames, currents),
            **_state(result, frames),
        }
    else:
        return None
    nets: dict[str, JsonValue] = {
        unknown.name: _series(values[frames, index])
        for index, unknown in enumerate(result.unknowns)
        if unknown.nature == "voltage"
    }
    logic: dict[str, JsonValue] = {
        net: [None if math.isnan(level) else float(level) for level in levels[frames]]
        for net, levels in _levels(result, axis).items()
    }
    return {
        "name": name,
        "kind": kind,
        "axis": _series(axis[frames]),
        "label": label,
        "unit": unit,
        "nets": nets,
        "logic": logic,
        **devices,
    }


def scene(
    circuit: ir.Circuit, analyses: Mapping[str, results.AnalysisResult], title: str = ""
) -> dict[str, JsonValue]:
    """Builds the complete JSON scene payload for `circuit` and `analyses`."""
    layouts: dict[str, JsonValue] = {
        name: _layout(circuit, name)
        for name in sorted(
            {
                circuit.top,
                *(
                    instance.of.component
                    for component in circuit.components.values()
                    for instance in component.instances
                    if isinstance(instance.of, ir.ComponentOf)
                ),
            }
        )
    }
    played: list[JsonValue] = [
        found
        for name, result in analyses.items()
        if (found := _analysis(name, result)) is not None
    ]
    return {
        "title": title or circuit.top,
        "top": circuit.top,
        "ground": circuit.ground_port() or "ground",
        "layouts": layouts,
        "views": _views(circuit),
        "analyses": played,
    }


_SERIES_FIELDS: Final = ("nets", "logic", "currents", "charges", "conductances")
"""The flat per-point series mappings in an analysis entry."""


def _compact_number(value: JsonValue) -> JsonValue:
    """Shortens an integer-valued float when its integer form writes shorter."""
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        whole = int(value)
        if len(str(whole)) < len(repr(value)):
            return whole
    return value


def _is_scalar(value: JsonValue) -> bool:
    """Returns whether `value` is a numeric scalar or null eligible for collapsing."""
    if value is None:
        return True
    return isinstance(value, int | float) and not isinstance(value, bool)


def _compact_series(values: JsonValue, length: int) -> JsonValue:
    """Collapses a constant per-point series to its scalar value."""
    if not isinstance(values, list) or not values or len(values) != length:
        return values
    compacted: list[JsonValue] = [_compact_number(item) for item in values]
    first = compacted[0]
    if _is_scalar(first) and all(
        item == first and type(item) is type(first) for item in compacted
    ):
        return first
    return compacted


def _compact_series_map(mapping: JsonValue, length: int) -> JsonValue:
    """Compacts every series in a flat mapping of names to series."""
    if not isinstance(mapping, dict):
        return mapping
    return {name: _compact_series(values, length) for name, values in mapping.items()}


def _compact_analysis(entry: JsonValue) -> JsonValue:
    """Compacts one analysis entry losslessly for HTML embedding."""
    if not isinstance(entry, dict):
        return entry
    axis = entry.get("axis")
    if not isinstance(axis, list):
        return entry
    compacted: dict[str, JsonValue] = dict(entry)
    compacted["axis"] = [_compact_number(point) for point in axis]
    length = len(axis)
    for field in _SERIES_FIELDS:
        if field in compacted:
            compacted[field] = _compact_series_map(compacted[field], length)
    variables = compacted.get("variables")
    if isinstance(variables, dict):
        compacted["variables"] = {
            device: _compact_series_map(values, length)
            for device, values in variables.items()
        }
    return compacted


def _compact_scene(data: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Returns a losslessly compacted copy of `data` for HTML serialization.

    Integer-valued floats in analysis axes and series take their shorter integer form,
    and any constant series collapses to its scalar value; the viewer script expands
    scalars back to full-length arrays on load.
    """
    compacted: dict[str, JsonValue] = dict(data)
    analyses = compacted.get("analyses")
    if isinstance(analyses, list):
        compacted["analyses"] = [_compact_analysis(entry) for entry in analyses]
    return compacted


_PAGE: Final = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<title>{title}</title>
{style}
<script type="importmap">{imports}</script>
</head>
<body>
<div id="jolty" class="jolty" data-state="loading">
  <div class="splash" role="status">
    <div class="spinner"></div><p>Loading the 3D view</p>
  </div>
</div>
<script type="application/json" id="scene">{scene}</script>
{script}
</body>
</html>
"""


def _asset_tags(assets: str | None) -> tuple[str, str]:
    """Builds inline or external `<style>` and `<script>` tags from `assets`."""
    if assets is None:
        bundled = files("jolty").joinpath("assets")
        style = bundled.joinpath("viewer.css").read_text(encoding="utf-8")
        script = bundled.joinpath("viewer.js").read_text(encoding="utf-8")
        return (
            f"<style>\n{style}\n</style>",
            f'<script type="module">\n{script}\n</script>',
        )
    stripped = assets if assets == "/" else assets.rstrip("/")
    prefix = f"{stripped}/" if stripped and stripped != "/" else stripped
    escaped = html.escape(prefix, quote=True)
    return (
        f'<link rel="stylesheet" href="{escaped}viewer.css">',
        f'<script type="module" src="{escaped}viewer.js"></script>',
    )


def page(
    data: Mapping[str, JsonValue],
    three: str = _THREE,
    *,
    assets: str | None = None,
) -> str:
    """Renders the viewer HTML page for scene `data` loading three.js from `three`.

    Omitting `assets` inlines the bundled stylesheet and module script for a standalone
    file. Passing a base URL or relative path in `assets` links external ``viewer.css``
    and ``viewer.js`` files beneath that base instead.
    """
    imports = {
        "imports": {
            "three": f"{three}/build/three.module.js",
            "three/addons/": f"{three}/examples/jsm/",
        }
    }
    text = json.dumps(_compact_scene(data), separators=(",", ":"), allow_nan=False)
    title = str(data.get("title", "jolty"))
    style, script = _asset_tags(assets)
    return _PAGE.format(
        title=html.escape(title, quote=True),
        style=style,
        imports=json.dumps(imports).replace("</", "<\\/"),
        scene=text.replace("</", "<\\/"),
        script=script,
    )


def save_viewer(  # noqa: PLR0913 - path, circuit, analyses, title and asset bases
    path: Path,
    circuit: ir.Circuit,
    analyses: Mapping[str, results.AnalysisResult],
    title: str = "",
    three: str = _THREE,
    *,
    assets: str | None = None,
) -> None:
    """Writes the interactive 3D viewer HTML for `circuit` and `analyses` to `path`.

    The generated page loads three.js from `three` (jsDelivr by default). Passing
    `assets` links external ``viewer.css`` and ``viewer.js`` beneath that base URL
    instead of inlining them.
    """
    rendered = page(scene(circuit, analyses, title), three, assets=assets)
    path.write_text(rendered, encoding="utf-8")
