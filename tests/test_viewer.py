# Copyright 2026 Maxence Faldor

import asyncio
import json
import re
from pathlib import Path

import numpy as np

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
from jolty.viewer import FRAMES, page, scene
from jolty.waveforms import Piecewise


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
