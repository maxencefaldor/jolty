# Copyright 2026 Maxence Faldor

import asyncio
import json
import re
from pathlib import Path
from typing import Final

import numpy as np
from pydantic import JsonValue, TypeAdapter

from jolty import (
    Builder,
    Role,
    Simulation,
    Transient,
    circuit,
    component,
    devices,
)
from jolty.files import save_circuit, save_results
from jolty.results import DcSweepResult
from jolty.service import ViewRequest, view
from jolty.settings import Settings
from jolty.units import V, kohm, nF, us
from jolty.unknowns import Unknown
from jolty.viewer import FRAMES, page, scene, write_viewer
from jolty.waveforms import Piecewise
from workspace import ROOT


@component
def stage(c: Builder) -> None:
    source, output = c.port("input"), c.port("output")
    ground = c.port("ground", Role.GROUND)
    resistor = c.place(devices.resistor(1 * kohm), name="R1")
    capacitor = c.place(devices.capacitor(1 * nF), name="C1")
    c.connect(source, resistor.a)
    c.connect(resistor.b, capacitor.a, output)
    c.connect(capacitor.b, ground)


@component
def bench(c: Builder) -> None:
    ground = c.port("gnd", Role.GROUND)
    step = Piecewise([(0 * us, 0 * V), (1 * us, 1 * V)])
    source = c.place(devices.voltage_source(step), name="V1")
    filter_ = c.place(stage(), name="X1")
    c.connect(source.positive, filter_.input, name="in")
    c.connect(filter_.output, name="out")
    c.connect(source.negative, filter_.ground, ground)


def test_a_scene_names_each_view_s_nets_and_samples_the_analysis() -> None:
    drawn = circuit(bench())
    waveforms = Simulation(drawn).run(
        Transient(stop=20 * us, max_step=0.01 * us, record=["**"])
    )
    found = scene(drawn, {"step": waveforms}, title="RC")
    assert found["views"] == {
        "": {
            "component": drawn.top,
            "nets": {"in": "in", "out": "out", "gnd": "gnd"},
        },
        "X1": {
            "component": "stage",
            "nets": {"input": "in", "output": "out", "ground": "gnd"},
        },
    }
    analyses = found["analyses"]
    assert isinstance(analyses, list)
    (played,) = analyses
    assert isinstance(played, dict)
    assert played["kind"] == "transient"
    axis = played["axis"]
    assert isinstance(axis, list)
    assert len(axis) == FRAMES
    currents = played["currents"]
    assert isinstance(currents, dict)
    assert {"X1/R1.a", "X1.input", "V1.positive"} <= set(currents)
    charges, conductances = played["charges"], played["conductances"]
    assert isinstance(charges, dict)
    assert isinstance(conductances, dict)
    assert set(charges) == {"V1.positive", "V1.negative", *_STAGE}
    assert "X1/R1.a|X1/R1.b" in conductances


_STAGE = ("X1/R1.a", "X1/R1.b", "X1/C1.a", "X1/C1.b")
"""The terminals of the devices of the bench's stage."""


def test_a_scene_plays_only_what_the_results_recorded() -> None:
    drawn = circuit(bench())
    simulation = Simulation(drawn)
    for record, terminals in (
        (None, set()),
        (["X1"], {"X1.input", "X1.output", "X1.ground"}),
        (["X1/R1"], {"X1/R1.a", "X1/R1.b"}),
    ):
        step = simulation.run(Transient(stop=5 * us, record=record))
        analyses = scene(drawn, {"step": step})["analyses"]
        assert isinstance(analyses, list)
        (played,) = analyses
        assert isinstance(played, dict)
        currents = played["currents"]
        assert isinstance(currents, dict)
        assert set(currents) == terminals


def test_the_page_embeds_the_scene_and_the_script(tmp_path: Path) -> None:
    drawn = circuit(bench())
    text = page(scene(drawn, {}, title="</script> RC"))
    embedded = re.search(
        r'<script type="application/json" id="scene">(.*?)</script>', text, re.DOTALL
    )
    assert embedded is not None
    assert json.loads(embedded[1].replace("<\\/", "</"))["title"] == "</script> RC"
    assert "three/addons/" in text
    assert "OrbitControls" in text
    path = tmp_path / "bench.jolty.json"
    save_circuit(path, drawn)
    results = tmp_path / "bench.results.npz"
    waveforms = Simulation(drawn).run(Transient(stop=5 * us))
    save_results(results, {"step": waveforms})
    written = asyncio.run(view(ViewRequest(circuit=path, results=results), Settings()))
    assert written.viewer == tmp_path / "bench.view.html"
    assert written.viewer.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_page_escapes_title_and_importmap_urls() -> None:
    drawn = circuit(bench())
    raw_title = 'A & B <script>"quote"</script>'
    malicious_three = "https://cdn.example/</script><script>alert(1)</script>"
    rendered = page(scene(drawn, {}, title=raw_title), three=malicious_three)
    assert (
        "<title>A &amp; B &lt;script&gt;&quot;quote&quot;&lt;/script&gt;</title>"
        in rendered
    )
    importmap = re.search(
        r'<script type="importmap">(.*?)</script>', rendered, re.DOTALL
    )
    assert importmap is not None
    assert "</script>" not in importmap[1]
    assert "<\\/script>" in importmap[1]
    parsed = json.loads(importmap[1])
    assert parsed["imports"]["three"] == f"{malicious_three}/build/three.module.js"


def test_dc_sweep_temperature_axis_uses_kelvin_unit() -> None:
    drawn = circuit(bench())
    sweep = DcSweepResult(
        unknowns=(Unknown("out", "voltage"),),
        source="T1",
        nature="temperature",
        swept=np.array([273.15, 300.0, 350.0]),
        values=np.array([[1.0], [1.1], [1.2]]),
    )
    current_sweep = DcSweepResult(
        unknowns=(Unknown("out", "voltage"),),
        source="I1",
        nature="current",
        swept=np.array([0.0, 1e-3]),
        values=np.array([[0.0], [1.0]]),
    )
    voltage_sweep = DcSweepResult(
        unknowns=(Unknown("out", "voltage"),),
        source="V1",
        nature="voltage",
        swept=np.array([0.0, 1.0]),
        values=np.array([[0.0], [1.0]]),
    )
    untyped_sweep = DcSweepResult(
        unknowns=(Unknown("out", "voltage"),),
        source="X1",
        nature=None,
        swept=np.array([0.0, 1.0]),
        values=np.array([[0.0], [1.0]]),
    )
    found = scene(
        drawn,
        {
            "temp": sweep,
            "curr": current_sweep,
            "volt": voltage_sweep,
            "none": untyped_sweep,
        },
    )
    analyses = found["analyses"]
    assert isinstance(analyses, list)
    units_by_name: dict[str, str] = {}
    for entry in analyses:
        assert isinstance(entry, dict)
        name = entry["name"]
        unit = entry["unit"]
        assert isinstance(name, str)
        assert isinstance(unit, str)
        units_by_name[name] = unit
    assert units_by_name == {"temp": "K", "curr": "A", "volt": "V", "none": ""}


_SCENE: Final = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue])


def _embedded_scene(rendered: str) -> dict[str, JsonValue]:
    embedded = re.search(
        r'<script type="application/json" id="scene">(.*?)</script>',
        rendered,
        re.DOTALL,
    )
    assert embedded is not None
    return _SCENE.validate_json(embedded[1].replace("<\\/", "</"))


def _expand_series(value: JsonValue, count: int) -> JsonValue:
    return value if isinstance(value, list) else [value] * count


def _expand_map(mapping: JsonValue, count: int) -> JsonValue:
    if not isinstance(mapping, dict):
        return mapping
    return {name: _expand_series(value, count) for name, value in mapping.items()}


def _expand_analysis(entry: JsonValue) -> JsonValue:
    if not isinstance(entry, dict):
        return entry
    axis = entry.get("axis")
    if not isinstance(axis, list):
        return entry
    count = len(axis)
    restored: dict[str, JsonValue] = dict(entry)
    for field in ("nets", "logic", "currents", "charges", "conductances"):
        if field in restored:
            restored[field] = _expand_map(restored[field], count)
    variables = restored.get("variables")
    if isinstance(variables, dict):
        restored["variables"] = {
            device: _expand_map(values, count) for device, values in variables.items()
        }
    return restored


def _expand_scene(data: dict[str, JsonValue]) -> dict[str, JsonValue]:
    expanded: dict[str, JsonValue] = dict(data)
    analyses = expanded.get("analyses")
    if isinstance(analyses, list):
        expanded["analyses"] = [_expand_analysis(entry) for entry in analyses]
    return expanded


def test_page_defaults_to_inlining_viewer_assets() -> None:
    rendered = page(scene(circuit(bench()), {}, title="RC"))
    assert "<style>\n" in rendered
    assert '<script type="module">\n' in rendered
    assert "viewer.css" not in rendered
    assert "viewer.js" not in rendered


def test_page_links_and_escapes_external_viewer_assets(tmp_path: Path) -> None:
    drawn = circuit(bench())
    found = scene(drawn, {}, title="RC")
    linked = page(found, assets="../assets")
    assert linked == page(found, assets="../assets/")
    assert '<link rel="stylesheet" href="../assets/viewer.css">' in linked
    assert '<script type="module" src="../assets/viewer.js"></script>' in linked
    assert "<style>" not in linked
    assert "OrbitControls" not in linked

    malicious = 'https://cdn.example/a"b&c</script>'
    escaped_prefix = "https://cdn.example/a&quot;b&amp;c&lt;/script&gt;/"
    escaped = page(found, assets=malicious)
    assert f'<link rel="stylesheet" href="{escaped_prefix}viewer.css">' in escaped
    assert f'<script type="module" src="{escaped_prefix}viewer.js"></script>' in escaped
    assert escaped.count("</script>") == 3

    target = tmp_path / "linked.view.html"
    write_viewer(target, drawn, {}, title="RC", assets="../assets")
    written = target.read_text(encoding="utf-8")
    assert '<link rel="stylesheet" href="../assets/viewer.css">' in written
    assert '<script type="module" src="../assets/viewer.js"></script>' in written
    assert "<style>" not in written


def test_page_compacts_scene_losslessly_without_mutating_scene_dict() -> None:
    drawn = circuit(bench())
    waveforms = Simulation(drawn).run(Transient(stop=5 * us, record=["**"]))
    original = scene(drawn, {"step": waveforms}, title="RC")
    original_analyses = original["analyses"]
    assert isinstance(original_analyses, list)
    original_step = original_analyses[0]
    assert isinstance(original_step, dict)
    original_charges = original_step["charges"]
    assert isinstance(original_charges, dict)
    assert isinstance(original_charges["V1.positive"], list)

    rendered = page(original, assets="../assets")
    assert isinstance(original_charges["V1.positive"], list)

    compacted = _embedded_scene(rendered)
    compacted_analyses = compacted["analyses"]
    assert isinstance(compacted_analyses, list)
    compacted_step = compacted_analyses[0]
    assert isinstance(compacted_step, dict)
    compacted_charges = compacted_step["charges"]
    compacted_conductances = compacted_step["conductances"]
    compacted_nets = compacted_step["nets"]
    assert isinstance(compacted_charges, dict)
    assert isinstance(compacted_conductances, dict)
    assert isinstance(compacted_nets, dict)
    zero_charge = compacted_charges["V1.positive"]
    assert zero_charge == 0
    assert isinstance(zero_charge, int)
    assert not isinstance(zero_charge, bool)
    assert compacted_conductances["X1/R1.a|X1/R1.a"] == 0.001
    assert isinstance(compacted_nets["out"], list)
    assert _expand_scene(compacted) == original


def test_page_compacts_numeric_and_null_series_edge_cases() -> None:
    synthetic: dict[str, JsonValue] = {
        "title": "Synthetic",
        "top": "top",
        "ground": "ground",
        "layouts": {"top": {"points": [0.0, 0.0, 0.0, 0.0]}},
        "views": {},
        "analyses": [
            {
                "name": "step",
                "kind": "transient",
                "axis": [-0.0, 0.0, 1.0, 1e20],
                "label": "time",
                "unit": "s",
                "nets": {
                    "zero": [-0.0, 0.0, 0.0, 0.0],
                    "constant": [2.5, 2.5, 2.5, 2.5],
                    "varying": [0.0, 1.0, 2.5, -0.0],
                },
                "logic": {
                    "unknown": [None, None, None, None],
                    "toggle": [0.0, 1.0, 0.0, 1.0],
                },
                "currents": {},
                "charges": {},
                "conductances": {},
                "variables": {"M1": {"region": [2.0, 2.0, 2.0, 2.0]}},
            }
        ],
    }
    rendered = page(synthetic, assets="../assets")
    assert '"axis":[0,0,1,1e+20]' in rendered
    compacted = _embedded_scene(rendered)
    analyses = compacted["analyses"]
    assert isinstance(analyses, list)
    entry = analyses[0]
    assert isinstance(entry, dict)
    nets = entry["nets"]
    logic = entry["logic"]
    variables = entry["variables"]
    assert isinstance(nets, dict)
    assert isinstance(logic, dict)
    assert isinstance(variables, dict)
    zero_net = nets["zero"]
    assert zero_net == 0
    assert isinstance(zero_net, int)
    assert not isinstance(zero_net, bool)
    assert nets["constant"] == 2.5
    assert nets["varying"] == [0, 1, 2.5, 0]
    assert logic["unknown"] is None
    assert logic["toggle"] == [0, 1, 0, 1]
    assert variables == {"M1": {"region": 2}}
    layouts = compacted["layouts"]
    assert isinstance(layouts, dict)
    assert layouts["top"] == {"points": [0.0, 0.0, 0.0, 0.0]}
    assert _expand_scene(compacted) == synthetic


def test_docs_viewer_assets_match_package_assets() -> None:
    package_assets = ROOT / "src" / "jolty" / "assets"
    docs_assets = ROOT / "docs" / "assets"
    for name in ("viewer.css", "viewer.js"):
        assert (docs_assets / name).read_bytes() == (package_assets / name).read_bytes()


def test_docs_viewers_link_shared_assets() -> None:
    docs = ROOT / "docs"
    docs_assets = docs / "assets"
    viewers = [
        *sorted((docs / "gallery").rglob("*.view.html")),
        *sorted((docs / "discovery").rglob("*.view.html")),
    ]
    assert viewers
    for viewer_path in viewers:
        text = viewer_path.read_text(encoding="utf-8")
        prefix = docs_assets.relative_to(viewer_path.parent, walk_up=True).as_posix()
        assert "<style>" not in text
        assert f'<link rel="stylesheet" href="{prefix}/viewer.css">' in text
        assert f'<script type="module" src="{prefix}/viewer.js"></script>' in text
        assert (viewer_path.parent / prefix / "viewer.css").is_file()
        assert (viewer_path.parent / prefix / "viewer.js").is_file()
