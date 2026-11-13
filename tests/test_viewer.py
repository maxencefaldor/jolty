# Copyright 2026 Maxence Faldor

import asyncio
import json
import re
from pathlib import Path

from jolty import (
    Builder,
    Options,
    Role,
    Simulation,
    Transient,
    circuit,
    component,
    devices,
)
from jolty.files import save_circuit, save_results
from jolty.service import ViewRequest, view
from jolty.settings import Settings
from jolty.units import V, kohm, nF, us
from jolty.viewer import FRAMES, page, scene
from jolty.waveforms import Piecewise


@component
def stage(c: Builder) -> None:
    source, output = c.port("input"), c.port("output")
    ground = c.port("ground", Role.GROUND)
    resistor = c.place(devices.resistor(1 * kohm), id="R1")
    capacitor = c.place(devices.capacitor(1 * nF), id="C1")
    c.connect(source, resistor.a)
    c.connect(resistor.b, capacitor.a, output)
    c.connect(capacitor.b, ground)


@component
def bench(c: Builder) -> None:
    ground = c.port("gnd", Role.GROUND)
    step = Piecewise([(0 * us, 0 * V), (1 * us, 1 * V)])
    source = c.place(devices.voltage_source(step), id="V1")
    filter_ = c.place(stage(), id="X1")
    c.connect(source.positive, filter_.input, name="in")
    c.connect(filter_.output, name="out")
    c.connect(source.negative, filter_.ground, ground)


def test_a_scene_names_each_view_s_nets_and_samples_the_analysis() -> None:
    drawn = circuit(bench())
    waveforms = Simulation(drawn).run(
        Transient(stop=20 * us, max_step=0.01 * us), Options(record=["**"])
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
    step = Transient(stop=5 * us)
    simulation = Simulation(drawn)
    for options, terminals in (
        (None, set()),
        (Options(record=["X1"]), {"X1.input", "X1.output", "X1.ground"}),
        (Options(record=["X1/R1"]), {"X1/R1.a", "X1/R1.b"}),
    ):
        analyses = scene(drawn, {"step": simulation.run(step, options)})["analyses"]
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
