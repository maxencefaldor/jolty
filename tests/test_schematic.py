# Copyright 2026 Maxence Faldor

import json
from collections.abc import Collection
from pathlib import Path

import pytest
from matplotlib.figure import Figure

from jolty import (
    Builder,
    JoltyError,
    Role,
    Simulation,
    circuit,
    component,
    devices,
    ir,
    results,
    schematic,
)
from jolty.cli import main
from jolty.files import save_circuit, save_results
from jolty.plan import OperatingPoint, Transient
from jolty.schematic import draw
from jolty.units import V, kohm, nF, us


@component
def divider(c: Builder) -> None:
    """A source across two resistors, its output named."""
    ground = c.port("ground", Role.GROUND)
    top = c.port("top")
    middle = c.port("middle")
    upper = c.place(devices.resistor(1 * kohm), id="RU")
    lower = c.place(devices.resistor(1 * kohm), id="RL")
    c.connect(top, upper.a)
    c.connect(upper.b, lower.a, middle)
    c.connect(lower.b, ground)


@component
def bench(c: Builder) -> None:
    """A divider fed from a source, loaded by a capacitor."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(4 * V), id="V1")
    halves = c.place(divider(), id="X1")
    load = c.place(devices.capacitor(1 * nF), id="C1")
    c.connect(source.positive, halves.top, name="in")
    c.connect(halves.middle, load.a, name="out")
    c.connect(source.negative, halves.ground, load.b, ground)


def texts(figure: Figure) -> dict[str, str]:
    """Each text of the figure, with its color."""
    return {text.get_text(): str(text.get_color()) for text in figure.axes[0].texts}


def test_the_top_carries_the_voltages_of_an_operating_point() -> None:
    drawn = circuit(bench())
    point = Simulation(drawn).run(OperatingPoint())
    written = texts(draw(drawn, point=point))
    assert {"4 V", "2 V"} <= set(written)
    assert {"V1", "X1", "C1", "divider"} <= set(written)


def test_an_instance_draws_its_component_with_its_nets_voltages() -> None:
    drawn = circuit(bench())
    point = Simulation(drawn).run(OperatingPoint())
    written = texts(draw(drawn, "X1", point=point))
    # The port nets take the voltages of the nets they join outside.
    assert {"RU", "RL", "top", "middle", "4 V", "2 V"} <= set(written)


def test_marked_instances_are_drawn_in_red() -> None:
    drawn = circuit(bench())
    written = texts(draw(drawn, marked=["X1/RL", "C1"]))
    assert written["! X1"] == written["! C1"] == "#c62828"
    assert written["V1"] != "#c62828"
    inside = texts(draw(drawn, "X1", marked=["X1/RL", "C1"]))
    assert "! RL" in inside
    assert "RU" in inside


def test_a_path_must_lead_to_a_component() -> None:
    drawn = circuit(bench())
    with pytest.raises(JoltyError, match="no instance is at X2"):
        draw(drawn, "X2")
    with pytest.raises(JoltyError, match="no instance is at X1/R9"):
        draw(drawn, "X1/R9")
    with pytest.raises(JoltyError, match="C1 is a device, which has no schematic"):
        draw(drawn, "C1")


SCRIPT = """\
from jolty import Builder, Plan, Role, component, devices
from jolty.plan import OperatingPoint
from jolty.units import V, kohm, nF


@component
def shorted(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1 * V), id="V1")
    resistor = c.place(devices.resistor(1 * kohm), id="R1")
    capacitor = c.place(devices.capacitor(1 * nF), id="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, source.negative, ground)
    c.connect(capacitor.a, capacitor.b, ground)


def build():
    return shorted()


def plan():
    return Plan({"op": OperatingPoint()})
"""


def run(*arguments: str) -> int:
    """Runs the command line and returns its exit status."""
    try:
        main(arguments)
    except SystemExit as exit_:
        return exit_.code if isinstance(exit_.code, int) else 0
    return 0


@pytest.fixture
def drawn(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """The arguments of each schematic the commands draw."""
    calls: list[dict[str, object]] = []
    original = schematic.draw

    def spy(
        drawn: ir.Circuit,
        path: str = "",
        *,
        point: results.OperatingPoint | None = None,
        marked: Collection[str] = (),
    ) -> Figure:
        calls.append({"path": path, "point": point, "marked": marked})
        return original(drawn, path, point=point, marked=marked)

    monkeypatch.setattr(schematic, "draw", spy)
    return calls


def test_a_build_draws_the_top_and_marks_what_diagnostics_name(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    drawn: list[dict[str, object]],
) -> None:
    script = tmp_path / "shorted.py"
    script.write_text(SCRIPT, encoding="utf-8")
    assert run("build", str(script), "--output-format", "json") == 0
    result = json.loads(capsys.readouterr().out)
    (diagnostic,) = result["report"]["diagnostics"]
    assert diagnostic["rule"] == "shorted-device"
    image = tmp_path / "shorted.schematic.png"
    assert result["schematic"] == str(image)
    assert image.read_bytes().startswith(b"\x89PNG")
    assert drawn == [{"path": "", "point": None, "marked": ("C1",)}]
    assert run("build", str(script), "--no-schematic", "--output-format", "json") == 0
    assert json.loads(capsys.readouterr().out)["schematic"] is None
    assert len(drawn) == 1


def test_a_simulation_writes_its_operating_point_on_the_schematic(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    drawn: list[dict[str, object]],
) -> None:
    script = tmp_path / "shorted.py"
    script.write_text(SCRIPT, encoding="utf-8")
    assert run("simulate", str(script)) == 0
    assert f"drew {tmp_path / 'shorted.schematic.png'}" in capsys.readouterr().out
    (options,) = drawn
    point = options["point"]
    assert isinstance(point, results.OperatingPoint)
    assert point.voltage("in") == pytest.approx(1.0)


def test_the_draw_command_draws_any_component_of_a_saved_circuit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    drawn = circuit(bench())
    path = tmp_path / "bench.jolty.json"
    save_circuit(path, drawn)
    assert run("draw", str(path)) == 0
    assert capsys.readouterr().out == f"drew {tmp_path / 'bench.schematic.png'}\n"
    results = tmp_path / "bench.results.npz"
    save_results(results, {"op": Simulation(drawn).run(OperatingPoint())})
    assert run("draw", str(path), "--instance", "X1", "--results", str(results)) == 0
    inside = tmp_path / "bench.X1.schematic.png"
    assert capsys.readouterr().out == f"drew {inside}\n"
    assert inside.read_bytes().startswith(b"\x89PNG")


def test_the_draw_command_names_what_it_cannot_draw(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    drawn = circuit(bench())
    path = tmp_path / "bench.jolty.json"
    save_circuit(path, drawn)
    assert run("draw", str(tmp_path / "missing.jolty.json")) == 2
    assert "no such circuit file" in capsys.readouterr().err
    assert run("draw", str(path), "--instance", "V1") == 2
    assert "V1 is a device" in capsys.readouterr().err
    results = tmp_path / "bench.results.npz"
    step = Transient(stop=1 * us, max_step=0.1 * us)
    save_results(results, {"step": Simulation(drawn).run(step)})
    assert run("draw", str(path), "--results", str(results)) == 2
    assert "holds no operating point" in capsys.readouterr().err
