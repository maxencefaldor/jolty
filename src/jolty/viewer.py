# Copyright 2026 Maxence Faldor
"""A standalone 3D viewer: one HTML page to orbit a circuit and scrub its simulation.

The page lays each component out as its schematic does, on a board: every device a part
standing on its pins, every wire a trace, the ground and supply pins posts. Its
simulation plays on it: each trace takes its net's voltage as a color, current flows
along each lead as moving light, faster as it grows, and each part glows with the power
it takes. A timeline scrubs through every point of each analysis; clicking a part shows
its parameters and its state at that point, the voltage, current and charge of each
terminal, its variables and its small-signal conductances; double-clicking a component
instance opens it. Currents, charges, variables and conductances are those the results
recorded: a plan records every instance with ``record=["**"]``.

The page holds the circuit, the layouts and the results as JSON, and the viewer's
script and styles, and loads three.js from jsDelivr, so it needs a network connection
when opened.

Examples:
    >>> from jolty import Builder, OperatingPoint, Role, Simulation, circuit, component
    >>> from jolty import devices
    >>> from jolty.units import V, kohm
    >>> @component
    ... def divider(c: Builder) -> None:
    ...     ground = c.port("ground", Role.GROUND)
    ...     source = c.place(devices.voltage_source(5 * V), id="V1")
    ...     top = c.place(devices.resistor(1 * kohm), id="R1")
    ...     bottom = c.place(devices.resistor(1 * kohm), id="R2")
    ...     c.connect(source.positive, top.a, name="in")
    ...     c.connect(top.b, bottom.a, name="out")
    ...     c.connect(bottom.b, source.negative, ground)
    >>> drawn = circuit(divider())
    >>> point = Simulation(drawn).run(OperatingPoint())
    >>> scene_ = scene(drawn, {"op": point})
    >>> scene_["analyses"][0]["nets"]["out"]
    [2.5]
"""

import json
import math
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import Final

import numpy as np
from pydantic import JsonValue

from jolty import ir, layout, results

THREE_VERSION: Final = "0.170.0"
"""The three.js version the viewer loads."""

_THREE: Final = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}"
"""The three.js package on jsDelivr."""

FRAMES: Final = 240
"""The most points of an analysis the viewer plays: a longer one is sampled evenly."""

_DIGITS: Final = 5
"""The significant digits each number keeps."""


def _number(value: float) -> JsonValue:
    """A finite number rounded to its significant digits, or None."""
    if not math.isfinite(value):
        return None
    if value == 0:
        return 0.0
    return float(f"{value:.{_DIGITS}g}")


def _series(values: np.ndarray) -> JsonValue:
    """Numbers, each rounded to its significant digits."""
    return [_number(float(value)) for value in values]


def _frames(count: int) -> np.ndarray:
    """The indices of at most `FRAMES` points, evenly spread, the ends kept."""
    if count <= FRAMES:
        return np.arange(count)
    return np.unique(np.round(np.linspace(0, count - 1, FRAMES)).astype(int))


def _layout(circuit: ir.Circuit, name: str) -> dict[str, JsonValue]:
    """A component's layout as the viewer draws it."""
    drawn = layout.layout(circuit, name)
    instances: list[JsonValue] = []
    for placed in drawn.instances:
        instance = placed.instance
        kind = (
            {"device": instance.of.device}
            if isinstance(instance.of, ir.DeviceOf)
            else {"component": instance.of.component}
        )
        (left, bottom), (right, top) = placed.body()
        instances.append(
            {
                "id": placed.id,
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
    """Each component instance's view, by path, the top's being ``""``.

    A view names the instance's component, and the net of the whole circuit that each
    of the component's own nets is.
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
                walk(instance.of.component, f"{prefix}{instance.id}/", inner)

    top = circuit.components[circuit.top]
    walk(circuit.top, "", {port.name: port.name for port in top.ports})
    return views


def _ground(circuit: ir.Circuit) -> str:
    """The net of the top component's ground port."""
    top = circuit.components[circuit.top]
    return next(
        (port.name for port in top.ports if port.role is ir.Role.GROUND), "ground"
    )


type _Observed = (
    results.OperatingPointResult | results.DcSweepResult | results.TransientResult
)

_OBSERVED: Final = (
    results.OperatingPointResult,
    results.DcSweepResult,
    results.TransientResult,
)
"""The results that hold what every device did at each point."""


def _axis(result: results.AnalysisResult) -> tuple[str, np.ndarray, str, str] | None:
    """What an analysis plays along: its kind, its points, their label and unit."""
    match result:
        case results.OperatingPointResult():
            return "operating_point", np.zeros(1), "", ""
        case results.DcSweepResult() if result.parameter is not None:
            return "dc_sweep", result.levels, f"{result.source} {result.parameter}", ""
        case results.DcSweepResult():
            unit = "A" if result.nature == "current" else "V"
            return "dc_sweep", result.levels, f"{result.source} level", unit
        case results.TransientResult():
            return "transient", result.times, "time", "s"
        case results.AcResult():
            return "ac", result.frequencies, "frequency", "Hz"
    return None


def _rows(result: _Observed, field: str) -> np.ndarray:
    """One of what every device did, a row per point even at an operating point."""
    values = np.asarray(getattr(result, field))
    return (
        values[None, :] if isinstance(result, results.OperatingPointResult) else values
    )


def _levels(result: results.AnalysisResult, axis: np.ndarray) -> dict[str, np.ndarray]:
    """Each logic net's level at each point."""
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
    """The current into every recorded electrical terminal, at `frames`.

    A component instance's or a logic device's terminal carries the sum the result
    recorded. The heat into a thermal terminal is no current: no light moves along its
    lead.
    """
    terminals: dict[str, JsonValue] = {}
    if not currents.size:
        return terminals
    sampled = currents[frames]
    for index, terminal in enumerate(recording.terminals):
        if terminal not in recording.thermal:
            terminals[terminal] = _series(sampled[:, index])
    return terminals


def _state(result: _Observed, frames: np.ndarray) -> dict[str, JsonValue]:
    """Each recorded device's charges, variables and conductances, at `frames`.

    A device's charges are its terminals', and its conductances those between two of
    its terminals.
    """
    recording = result.recording
    stored = _rows(result, "charges")
    readings = _rows(result, "readings")
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
        if readings.size:
            for own in device.variables:
                values = _series(readings[frames, variable])
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
    """One analysis as the viewer plays it, its points sampled.

    A frequency response plays the magnitudes of its phasors along its frequencies.
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
    elif isinstance(result, _OBSERVED):
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
    """Everything the viewer shows of `circuit` and the results of its `analyses`."""
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
        "ground": _ground(circuit),
        "layouts": layouts,
        "views": _views(circuit),
        "analyses": played,
    }


_PAGE: Final = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<title>{title}</title>
<style>
{style}
</style>
<script type="importmap">{imports}</script>
</head>
<body>
<div id="jolty" class="jolty" data-state="loading">
  <div class="splash" role="status">
    <div class="spinner"></div><p>Loading the 3D view</p>
  </div>
</div>
<script type="application/json" id="scene">{scene}</script>
<script type="module">
{script}
</script>
</body>
</html>
"""


def page(scene_: Mapping[str, JsonValue], three: str = _THREE) -> str:
    """The standalone HTML page of a scene, loading three.js from `three`."""
    assets = files("jolty").joinpath("assets")
    imports = {
        "imports": {
            "three": f"{three}/build/three.module.js",
            "three/addons/": f"{three}/examples/jsm/",
        }
    }
    text = json.dumps(scene_, separators=(",", ":"), allow_nan=False)
    title = str(scene_.get("title", "jolty"))
    return _PAGE.format(
        title=title.replace("<", "&lt;"),
        style=assets.joinpath("viewer.css").read_text(encoding="utf-8"),
        imports=json.dumps(imports),
        scene=text.replace("</", "<\\/"),
        script=assets.joinpath("viewer.js").read_text(encoding="utf-8"),
    )


def write_viewer(
    path: Path,
    circuit: ir.Circuit,
    analyses: Mapping[str, results.AnalysisResult],
    title: str = "",
    three: str = _THREE,
) -> None:
    """Writes the viewer of `circuit` and its `analyses` to `path`.

    The page loads three.js from `three`, the package's base URL, jsDelivr's unless
    given.
    """
    path.write_text(page(scene(circuit, analyses, title), three), encoding="utf-8")
