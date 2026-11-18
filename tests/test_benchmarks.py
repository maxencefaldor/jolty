# Copyright 2026 Maxence Faldor

import json
import math
import os
import resource
import runpy
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

import bench
import measuring
import numpy as np
import pytest
import timing

from jolty import (
    Builder,
    DcSweep,
    OperatingPoint,
    Role,
    Simulation,
    Transient,
    circuit,
    component,
    devices,
    ir,
    spice,
)
from jolty.plan import TIGHT, linear
from jolty.units import V, kohm, mA, nF, um, us

BENCHMARKS = Path(__file__).parent.parent / "benchmarks" / "agents"
COST_DOWN = BENCHMARKS / "cost_down"
CUBIC = BENCHMARKS / "cubic"


@pytest.fixture(autouse=True)
def importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Imports the benchmarks' scripts, which import each other, as their runs do."""
    monkeypatch.syspath_prepend(str(BENCHMARKS))


def design(work: Path, tolerance: str) -> None:
    """Writes the original bias as `work`'s design, its resistors at `tolerance`."""
    original = (COST_DOWN / "given" / "bias.py").read_text(encoding="utf-8")
    changed = original.replace("TOLERANCE = 0.001", f"TOLERANCE = {tolerance}")
    (work / "design.py").write_text(changed, encoding="utf-8")


def test_a_cheaper_design_that_meets_the_specs_meets_the_task(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    design(tmp_path, "0.01")
    verdict = score(COST_DOWN, tmp_path)
    assert verdict.met, verdict.reasons
    assert verdict.measurements == {
        "baseline": pytest.approx(0.2),
        "cost": pytest.approx(0.016),
    }


def test_the_score_says_why_work_falls_short(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it
    from verdict import Verdict  # noqa: PLC0415 - importable only once the path holds it

    def kinds(verdict: Verdict) -> list[tuple[str, str | None]]:
        return [(shortfall.kind, shortfall.spec) for shortfall in verdict.shortfalls]

    verdict = score(COST_DOWN, tmp_path)
    assert verdict.reasons == (f"there is no {tmp_path / 'design.py'}",)
    assert kinds(verdict) == [("missing", None)]
    design(tmp_path, "0.001")
    verdict = score(COST_DOWN, tmp_path)
    assert verdict.reasons == ("the design costs 0.2, not less than 0.2",)
    assert kinds(verdict) == [("spec", "cost")]
    # jolty's own reason: every spec passes, but too few samples do.
    design(tmp_path, "0.05")
    verdict = score(COST_DOWN, tmp_path)
    assert kinds(verdict) == [("spec", "yield")]
    assert verdict.reasons[0].startswith("yield: ")
    assert verdict.reasons[0].endswith("which is below the target 99 %")
    original = (COST_DOWN / "given" / "bias.py").read_text(encoding="utf-8")
    cheaper = original.replace("TOLERANCE = 0.001", "TOLERANCE = 0.01")
    (tmp_path / "design.py").write_text(
        cheaper.replace("voltage_source(5 * V)", "voltage_source(5.1 * V)"),
        encoding="utf-8",
    )
    verdict = score(COST_DOWN, tmp_path)
    assert "the bench's V1 differs" in verdict.reasons
    assert ("spec", "bench") in kinds(verdict)
    (tmp_path / "design.py").write_text(
        cheaper.replace("16.5 * kohm, 1 * kohm", "-16.5 * kohm, 1 * kohm"),
        encoding="utf-8",
    )
    verdict = score(COST_DOWN, tmp_path)
    assert kinds(verdict) == [("unelaborated", None), ("spec", "catalog")]
    assert "resistance must be positive" in verdict.reasons[0]
    (tmp_path / "design.py").write_text(
        "def build():\n    raise RuntimeError('unfinished')\n", encoding="utf-8"
    )
    verdict = score(COST_DOWN, tmp_path)
    assert kinds(verdict) == [("unbuilt", None)]
    assert verdict.reasons[0].startswith("the design does not build: ")


def test_the_given_transistor_is_excused_by_what_it_is_not_its_name(
    tmp_path: Path,
) -> None:
    from arms import Arm  # noqa: PLC0415 - importable only once the path holds it
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    unpriced = (
        "is unpriced: a bipolar_transistor is priced by its part number alone, and it "
        "states none"
    )
    # Renamed, in either arm, the given transistor is still excused, as it is in the
    # original, which no catalog part prices either.
    design(tmp_path, "0.01")
    script = (tmp_path / "design.py").read_text(encoding="utf-8")
    (tmp_path / "design.py").write_text(
        script.replace('name="Q1"', 'name="QB"'), encoding="utf-8"
    )
    verdict = score(COST_DOWN, tmp_path)
    assert verdict.met, verdict.reasons
    netlist = (COST_DOWN / "ngspice" / "given" / "stage.cir").read_text(
        encoding="utf-8"
    )
    stage(tmp_path, dict.fromkeys(RESISTORS, 0.01), netlist.replace("Q1 ", "QB "))
    verdict = score(COST_DOWN, tmp_path, Arm.NGSPICE)
    assert verdict.met, verdict.reasons
    # Another transistor in its place is unpriced, whatever its name.
    stage(
        tmp_path, dict.fromkeys(RESISTORS, 0.01), netlist.replace("BF=416.4", "BF=500")
    )
    assert f"X1/Q1 {unpriced}" in score(COST_DOWN, tmp_path, Arm.NGSPICE).reasons
    # The original leaves one transistor out of its total: a second is unpriced.
    second = "QS 0 0 0 q1\n.model q1"
    stage(
        tmp_path, dict.fromkeys(RESISTORS, 0.01), netlist.replace(".model q1", second)
    )
    assert score(COST_DOWN, tmp_path, Arm.NGSPICE).reasons == (f"X1/QS {unpriced}",)


def test_the_runner_runs_the_agent_in_a_copy_of_the_task(tmp_path: Path) -> None:
    from run import Program, run  # noqa: PLC0415 - importable only once the path holds it

    agent = Program(
        (
            sys.executable,
            "-c",
            (
                "import pathlib, sys; source = pathlib.Path('bias.py').read_text(); "
                "pathlib.Path('design.py').write_text("
                "source.replace('TOLERANCE = 0.001', 'TOLERANCE = 0.01')); "
                "print(sys.argv[1].splitlines()[0])"
            ),
        ),
        confined=False,
    )
    assert run(COST_DOWN, agent, tmp_path) == 0
    assert (tmp_path / "catalog.json").is_file()
    assert not (tmp_path / "task.md").exists()
    assert not (tmp_path / "scorer.py").exists()
    transcript = (tmp_path / "transcript.txt").read_text(encoding="utf-8")
    assert transcript.startswith("# Cost-down: a transistor's bias")


def test_the_patented_cubic_generator_scores_one(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    patent = CUBIC / "patent.py"
    (tmp_path / "design.py").write_text(
        f"import runpy\nbuild = runpy.run_path({str(patent)!r})['cubic']\n",
        encoding="utf-8",
    )
    verdict = score(CUBIC, tmp_path)
    assert verdict.met, verdict.reasons
    assert verdict.measurements["score"] == pytest.approx(1.0)
    assert verdict.measurements["devices"] == 9


def test_the_cubic_scorer_names_a_forbidden_part(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    (tmp_path / "design.py").write_text(
        "from jolty import Builder, component, devices\n"
        "from jolty.units import A, kohm\n"
        "@component\n"
        "def cubic(c: Builder) -> None:\n"
        "    ground = c.port('ground')\n"
        "    for name in ('supply', 'reference', 'input'):\n"
        "        load = c.place(devices.resistor(1 * kohm), name=f'R_{name}')\n"
        "        c.connect(c.port(name), load.a)\n"
        "        c.connect(load.b, ground)\n"
        "    clamp = c.place(devices.diode(saturation_current=1e-14 * A), name='D9')\n"
        "    c.connect(c.port('output'), clamp.anode)\n"
        "    c.connect(clamp.cathode, ground)\n"
        "build = cubic\n",
        encoding="utf-8",
    )
    verdict = score(CUBIC, tmp_path)
    assert not verdict.met
    assert "D9 is a diode, which the task forbids" in verdict.reasons
    (forbidden,) = (each for each in verdict.shortfalls if each.spec == "parts")
    assert forbidden.kind == "spec"


ANALOGCODER = BENCHMARKS / "analogcoder"


def reference_designs(work: Path) -> None:
    """Writes each of AnalogCoder's problems' reference design into `work`."""
    reference = ANALOGCODER / "reference.py"
    for number in range(1, 25):
        (work / f"p{number}.py").write_text(
            "import runpy\n"
            f"_reference = runpy.run_path({str(reference)!r})\n"
            f"LEVELS = _reference['LEVELS'].get({number}, {{}})\n"
            "def build():\n"
            f"    return _reference['build']({number})\n",
            encoding="utf-8",
        )


def test_the_reference_designs_pass_every_analogcoder_check(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    reference_designs(tmp_path)
    verdict = score(ANALOGCODER, tmp_path)
    assert verdict.met, verdict.reasons
    assert verdict.measurements["solved"] == 24


def test_the_analogcoder_scorer_says_why_designs_fail(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    reference_designs(tmp_path)
    (tmp_path / "p3.py").unlink()
    (tmp_path / "p7.py").write_text(
        "from jolty import Builder, Role, component, devices\n"
        "@component\n"
        "def inverter(c: Builder) -> None:\n"
        "    ground = c.port('ground', Role.GROUND)\n"
        "    c.port('vdd', Role.SUPPLY)\n"
        "    amplifier = c.place(devices.voltage_controlled_voltage_source(-1.0), "
        "name='E1')\n"
        "    c.connect(amplifier.positive, c.port('vout'))\n"
        "    c.connect(amplifier.control_positive, c.port('vin'))\n"
        "    c.connect(amplifier.negative, amplifier.control_negative, ground)\n"
        "build = inverter\n",
        encoding="utf-8",
    )
    (tmp_path / "p18.py").write_text(
        "from jolty import Builder, Role, component, devices\n"
        "from jolty.units import V, kohm, uF\n"
        "LEVELS = {'vin': 2.5}\n"
        "@component\n"
        "def passive(c: Builder) -> None:\n"
        "    ground = c.port('ground', Role.GROUND)\n"
        "    c.port('vdd', Role.SUPPLY)\n"
        "    resistor = c.place(devices.resistor(10 * kohm), name='R1')\n"
        "    capacitor = c.place(devices.capacitor(3 * uF), name='CF')\n"
        "    rest = c.place(devices.voltage_source(2.5 * V), name='VR')\n"
        "    c.connect(resistor.a, c.port('vin'))\n"
        "    c.connect(resistor.b, capacitor.a, c.port('vout'))\n"
        "    c.connect(capacitor.b, rest.positive)\n"
        "    c.connect(rest.negative, ground)\n"
        "build = passive\n",
        encoding="utf-8",
    )
    verdict = score(ANALOGCODER, tmp_path)
    assert not verdict.met
    assert verdict.measurements["solved"] == 21
    assert "p3: there is no p3.py" in verdict.reasons
    # Each shortfall names its problem, and the spec of its check.
    assert sorted(
        {(each.design, each.kind, each.spec) for each in verdict.shortfalls}
    ) == [
        ("p18", "spec", "integrator"),
        ("p3", "missing", None),
        ("p7", "spec", "parts"),
    ]
    assert "p7: E1 is a voltage_controlled_voltage_source, which the task forbids" in (
        verdict.reasons
    )
    assert any(
        reason.startswith("p18: vout ramps at") and "while vin is above" in reason
        for reason in verdict.reasons
    ), verdict.reasons


VOLTAGE_CURRENT = BENCHMARKS / "voltage_current"


def converter(work: Path, body: str) -> None:
    """Writes a voltage-current converter whose placements are `body` into `work`."""
    (work / "design.py").write_text(
        "from jolty import Builder, Role, component, devices\n"
        "from jolty.units import A, V, kohm\n"
        "@component\n"
        "def converter(c: Builder) -> None:\n"
        "    ground = c.port('ground', Role.GROUND)\n"
        "    vee = c.port('vee', Role.SUPPLY)\n"
        "    plus, minus, output = c.port('plus'), c.port('minus'), c.port('output')\n"
        f"{body}"
        "build = converter\n",
        encoding="utf-8",
    )


def test_an_ideal_converter_has_no_error_but_is_forbidden(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    converter(
        tmp_path,
        "    source = c.place(devices.voltage_controlled_current_source(1e-3 * A / V), "
        "name='G1')\n"
        "    c.connect(source.control_positive, plus)\n"
        "    c.connect(source.control_negative, minus)\n"
        "    c.connect(source.positive, ground)\n"
        "    c.connect(source.negative, output)\n",
    )
    verdict = score(VOLTAGE_CURRENT, tmp_path)
    assert verdict.measurements["fitness"] == pytest.approx(0.0, abs=1e-9)
    assert verdict.reasons == (
        "G1 is a voltage_controlled_current_source, which the task forbids",
    )


def test_a_converter_blind_to_one_input_falls_short(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    converter(
        tmp_path,
        "    resistor = c.place(devices.resistor(1 * kohm), name='R1')\n"
        "    c.connect(resistor.a, plus)\n"
        "    c.connect(resistor.b, output)\n",
    )
    verdict = score(VOLTAGE_CURRENT, tmp_path)
    assert not verdict.met
    fitness = [reason for reason in verdict.reasons if "the fitness is " in reason]
    nominal, *cornered = fitness
    assert nominal.startswith("the fitness is ")
    assert "settles 0.00398 A from its 0.002 A" in verdict.reasons[1]
    assert [reason.split(", the fitness is ")[0] for reason in cornered] == [
        "at 0 °C with the supplies 5 % low",
        "at 0 °C with the supplies 5 % high",
        "at 70 °C with the supplies 5 % low",
        "at 70 °C with the supplies 5 % high",
    ]
    assert verdict.measurements["devices"] == 1


BALUN = BENCHMARKS / "balun"


def balun(work: Path, body: str) -> None:
    """Writes a balun whose placements are `body` into `work`."""
    (work / "design.py").write_text(
        "from jolty import Builder, Role, component, devices\n"
        "from jolty.units import Mohm, ohm\n"
        "@component\n"
        "def balun(c: Builder) -> None:\n"
        "    ground = c.port('ground', Role.GROUND)\n"
        "    c.port('vcc', Role.SUPPLY)\n"
        "    source = c.port('input')\n"
        "    in_phase, out_of_phase = c.port('in_phase'), c.port('out_of_phase')\n"
        f"{body}"
        "build = balun\n",
        encoding="utf-8",
    )


def test_an_ideal_balun_has_no_error_but_is_forbidden(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    balun(
        tmp_path,
        "    bias = c.place(devices.resistor(1000 * Mohm), name='R1')\n"
        "    c.connect(bias.a, source)\n"
        "    c.connect(bias.b, ground)\n"
        "    gains = (('E0', in_phase, 0.5), ('E1', out_of_phase, -0.5))\n"
        "    for name, port, gain in gains:\n"
        "        amplifier = c.place(devices.voltage_controlled_voltage_source(gain), "
        "name=name)\n"
        "        c.connect(amplifier.positive, port)\n"
        "        c.connect(amplifier.control_positive, source)\n"
        "        c.connect(amplifier.negative, amplifier.control_negative, ground)\n",
    )
    verdict = score(BALUN, tmp_path)
    for error in ("difference gain", "worst in-phase gain", "worst difference phase"):
        assert verdict.measurements[f"{error} error"] == pytest.approx(0.0, abs=1e-6)
    assert verdict.measurements["distortion"] < -150
    assert verdict.reasons == (
        "E0 is a voltage_controlled_voltage_source, which the task forbids",
        "E1 is a voltage_controlled_voltage_source, which the task forbids",
    )


def test_a_balun_without_an_inverted_output_falls_short(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    balun(
        tmp_path,
        "    series = c.place(devices.resistor(75 * ohm), name='R1')\n"
        "    c.connect(series.a, source)\n"
        "    c.connect(series.b, in_phase)\n",
    )
    verdict = score(BALUN, tmp_path)
    assert not verdict.met
    assert any("out-of-phase phase" in reason for reason in verdict.reasons)
    assert any("difference's gain" in reason for reason in verdict.reasons)


def test_a_balun_whose_input_floats_fails_to_simulate(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    balun(
        tmp_path,
        "    series = c.place(devices.resistor(75 * ohm), name='R1')\n"
        "    c.connect(series.a, source)\n",
    )
    verdict = score(BALUN, tmp_path)
    assert len(verdict.reasons) == 5
    assert verdict.reasons[0].startswith("the simulation fails: ")
    assert all("the simulation fails: " in reason for reason in verdict.reasons)
    assert {shortfall.kind for shortfall in verdict.shortfalls} == {"analysis"}


def test_a_shortfall_s_kind_is_where_the_scorer_found_it(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    # A design whose ports are not the bench's does not build on it, where the
    # scorer used to crash on the port the bench could not join.
    converter(tmp_path, "")
    (tmp_path / "design.py").write_text(
        (tmp_path / "design.py")
        .read_text(encoding="utf-8")
        .replace("c.port('minus')", "c.port('minus_')"),
        encoding="utf-8",
    )
    (shortfall,) = score(VOLTAGE_CURRENT, tmp_path).shortfalls
    assert shortfall.kind == "unbuilt"
    assert shortfall.reason == (
        "its ports are ground, minus_, output, plus, vee, not ground, minus, output, "
        "plus, vee"
    )
    # A design the engine cannot elaborate, in every corner.
    converter(
        tmp_path,
        "    load = c.place(devices.resistor(-1 * kohm), name='R1')\n"
        "    c.connect(load.a, plus)\n"
        "    c.connect(load.b, output)\n"
        "    bias = c.place(devices.resistor(1 * kohm), name='R2')\n"
        "    c.connect(bias.a, minus)\n"
        "    c.connect(bias.b, vee)\n"
        "    c.connect(ground)\n",
    )
    verdict = score(VOLTAGE_CURRENT, tmp_path)
    assert {shortfall.kind for shortfall in verdict.shortfalls} == {"unelaborated"}
    assert "resistance must be positive" in verdict.reasons[0]


CUBIC_DISCRETE = BENCHMARKS / "cubic_discrete"


def test_an_ideal_discrete_cubic_has_no_error_but_is_forbidden(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    (tmp_path / "design.py").write_text(
        "from jolty import Builder, Role, component, devices\n"
        "from jolty.units import kohm\n"
        "@component\n"
        "def cubic(c: Builder) -> None:\n"
        "    ground = c.port('ground', Role.GROUND)\n"
        "    c.port('vcc', Role.SUPPLY)\n"
        "    signal, output = c.port('input'), c.port('output')\n"
        "    sense = c.place(devices.resistor(1 * kohm), name='R1')\n"
        "    c.connect(sense.a, signal)\n"
        "    c.connect(sense.b, ground)\n"
        "    expression = '(v(s) / 0.1) ^ 3 / 10500'\n"
        "    cube = c.place(devices.behavioral_current_source(expression), name='B1')\n"
        "    c.connect(cube.positive, output)\n"
        "    c.connect(cube.negative, ground)\n"
        "    c.connect(cube.s, signal)\n"
        "build = cubic\n",
        encoding="utf-8",
    )
    verdict = score(CUBIC_DISCRETE, tmp_path)
    # Newton iteration leaves the ideal cubic's output within nanovolts of the cube,
    # where a hit is within 7 mV.
    assert verdict.measurements["mean absolute error"] < 1e-8
    assert verdict.measurements["hits"] == 1
    assert verdict.reasons == (
        "B1 is a behavioral_current_source, which the task forbids",
    )


def test_a_converter_that_oscillates_does_not_settle(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    parts = VOLTAGE_CURRENT / "given" / "parts.py"
    converter(
        tmp_path,
        "    from runpy import run_path\n"
        "    from jolty.units import pF\n"
        f"    npn = run_path({str(parts)!r})['npn']\n"
        "    stages = [c.place(npn(), name=f'Q{index}') for index in range(3)]\n"
        "    for index, stage in enumerate(stages):\n"
        "        load = c.place(devices.resistor(10 * kohm), name=f'RC{index}')\n"
        "        drive = c.place(devices.resistor(100 * kohm), name=f'RB{index}')\n"
        "        slow = c.place(devices.capacitor(100 * pF), name=f'C{index}')\n"
        "        c.connect(load.a, minus)\n"
        "        c.connect(load.b, stage.collector, drive.a, slow.a)\n"
        "        c.connect(drive.b, stages[(index + 1) % 3].base)\n"
        "        c.connect(slow.b, stage.emitter, vee)\n"
        "    out = c.place(devices.resistor(1 * kohm), name='RO')\n"
        "    c.connect(out.a, stages[0].collector)\n"
        "    c.connect(out.b, output)\n",
    )
    verdict = score(VOLTAGE_CURRENT, tmp_path)
    assert any("still swings" in reason for reason in verdict.reasons)


@component
def parallel(c: Builder) -> None:
    """A design of a resistor beside a capacitor, which gives the resistor 20 %."""
    a, b = c.port("a"), c.port("b")
    resistor = c.place(devices.resistor(1 * kohm, tolerance=0.2), name="R1")
    capacitor = c.place(devices.capacitor(1 * nF), name="C1")
    c.connect(resistor.a, capacitor.a, a)
    c.connect(resistor.b, capacitor.b, b)


@component
def loaded(c: Builder) -> None:
    """`parallel` on a bench that loads it with an exact resistor."""
    ground = c.port("ground", Role.GROUND)
    placed = c.place(parallel(), name="X1")
    load = c.place(devices.resistor(1 * kohm), name="RLOAD")
    c.connect(placed.a, load.a, name="a")
    c.connect(placed.b, load.b, ground)


def test_boards_draw_the_designs_resistors_and_capacitors_alone() -> None:
    from scoring import BOARDS, NOMINAL, boards, parts  # noqa: PLC0415 - importable only once the path holds it

    drawn = boards(circuit(loaded()), parts)
    assert len(drawn) == BOARDS
    assert all(set(board.parts) == {"X1/R1", "X1/C1"} for board in drawn)
    assert all(board.corner == NOMINAL for board in drawn)
    for path, parameter, nominal, tolerance in (
        ("X1/R1", "resistance", 1e3, 0.01),
        ("X1/C1", "capacitance", 1e-9, 0.05),
    ):
        strays = [abs(board.parts[path][parameter] / nominal - 1) for board in drawn]
        assert 0.9 * tolerance < max(strays) <= tolerance
    assert boards(circuit(loaded()), parts) == drawn


def test_an_integrated_designs_resistors_stray_together() -> None:
    from scoring import boards, integrated  # noqa: PLC0415 - importable only once the path holds it

    @component
    def pair(c: Builder) -> None:
        a, b = c.port("a"), c.port("b")
        for id, value in (("R1", 1 * kohm), ("R2", 3 * kohm)):
            placed = c.place(devices.resistor(value), name=id)
            c.connect(placed.a, a)
            c.connect(placed.b, b)

    @component
    def die(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        placed = c.place(pair(), name="X1")
        c.connect(placed.a, placed.b, ground)

    drawn = boards(circuit(die()), integrated)
    first = np.array([board.parts["X1/R1"]["resistance"] / 1e3 for board in drawn])
    second = np.array([board.parts["X1/R2"]["resistance"] / 3e3 for board in drawn])
    assert np.abs(first - 1).max() > 0.15
    assert np.abs(first / second - 1).max() <= 0.0025


def test_the_given_transistors_stray_within_their_datasheets() -> None:
    from scoring import REELS  # noqa: PLC0415 - importable only once the path holds it

    given = runpy.run_path(str(VOLTAGE_CURRENT / "given" / "parts.py"))
    measures = {
        "forward_beta": (_gain_at_10_milliamperes, 100.0, 300.0),
        "saturation_current": (_saturation_voltage, 0.65, 0.85),
    }
    for make, polarity, sign in (
        (given["npn"], "npn", 1.0),
        (given["pnp"], "pnp", -1.0),
    ):
        made = make()
        for parameter, tolerance in REELS[polarity].items():
            measure, least, most = measures[parameter]
            for wider in (0.0, 0.05):
                readings = [
                    measure(_scaled(made, parameter, factor), sign)
                    for factor in _extremes(tolerance, wider)
                ]
                within = all(least <= reading <= most for reading in readings)
                assert within == (wider == 0), (polarity, parameter, readings)


def _extremes(tolerance: ir.Tolerance, wider: float) -> tuple[float, float]:
    """The least and greatest factors `tolerance` and its lot draw.

    The lot's spread is widened by `wider`, relative to it.
    """
    assert tolerance.lot is not None
    lot = tolerance.lot.relative * (1 + wider)
    if tolerance.distribution.startswith("log"):
        greatest = (1 + lot) * (1 + tolerance.relative)
        return 1 / greatest, greatest
    return (1 - lot) * (1 - tolerance.relative), (1 + lot) * (1 + tolerance.relative)


def _scaled(
    transistor: devices.Device, parameter: str, factor: float
) -> devices.Device:
    """`transistor` with `parameter` scaled by `factor`."""
    value = transistor.parameters[parameter]
    assert isinstance(value, float)
    return replace(
        transistor, parameters={**transistor.parameters, parameter: value * factor}
    )


def _saturation_voltage(transistor: devices.Device, sign: float) -> float:
    """The base-emitter voltage of `transistor` in saturation, at 10 mA and 1 mA."""

    @component
    def fixture(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        placed = c.place(transistor, name="Q1")
        collector = c.place(devices.current_source(sign * 10 * mA), name="IC")
        base = c.place(devices.current_source(sign * 1 * mA), name="IB")
        c.connect(collector.negative, placed.collector)
        c.connect(base.negative, placed.base, name="base")
        c.connect(collector.positive, base.positive, placed.emitter, ground)

    return abs(Simulation(fixture()).run(OperatingPoint()).voltage("base"))


def _gain_at_10_milliamperes(transistor: devices.Device, sign: float) -> float:
    """The current gain of `transistor` at 10 mA of collector current and 1 V."""

    @component
    def fixture(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        placed = c.place(transistor, name="Q1")
        collector = c.place(devices.voltage_source(sign * 1 * V), name="VC")
        base = c.place(devices.voltage_source(0 * V), name="VB")
        c.connect(collector.positive, placed.collector)
        c.connect(base.positive, placed.base)
        c.connect(collector.negative, base.negative, placed.emitter, ground)

    swept = Simulation(fixture()).run(
        DcSweep("VB", linear(sign * 0.6 * V, sign * 0.9 * V, 301))
    )
    collector = np.abs(swept.current("VC"))
    base = np.abs(swept.current("VB"))
    return float(np.interp(10e-3, collector, collector / base))


def test_a_board_sets_its_corner_on_the_bench() -> None:
    from scoring import Board, Corner  # noqa: PLC0415 - importable only once the path holds it

    @component
    def divider(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        supply = c.place(devices.voltage_source(2 * V), name="VCC")
        upper = c.place(devices.resistor(1 * kohm), name="R1")
        lower = c.place(devices.resistor(1 * kohm), name="R2")
        c.connect(supply.positive, upper.a)
        c.connect(upper.b, lower.a, name="out")
        c.connect(lower.b, supply.negative, ground)

    corner = Corner(temperature=70.0, supply=0.95)
    board = Board({"R2": {"resistance": 3e3}}, corner)
    simulation = board.simulation(circuit(divider()), ["VCC"])
    assert simulation.conditions.temperature == pytest.approx(343.15)
    point = simulation.run(OperatingPoint())
    assert point.voltage("out") == pytest.approx(2 * 0.95 * 3 / 4)
    assert corner.describe() == "at 70 °C with the supplies 5 % low"
    with pytest.raises(KeyError, match="R1 is no DC source"):
        board.simulation(circuit(divider()), ["R1"])


def test_too_few_boards_that_meet_the_task_fall_short() -> None:
    from scoring import Board, parts, tolerate, unmet  # noqa: PLC0415 - importable only once the path holds it
    from verdict import Kind, Shortfall  # noqa: PLC0415 - importable only once the path holds it

    def measure(board: Board) -> dict[str, float]:
        return {"resistance": board.parts["X1/R1"]["resistance"]}

    def below(most: float) -> Callable[[dict[str, float]], list[Shortfall]]:
        high = [unmet("resistance", "it is too high")]
        return lambda found: [] if found["resistance"] <= most else high

    spread, reasons = tolerate(circuit(loaded()), measure, below(1005), parts)
    assert 0.6 < spread["yield"] < 0.9
    assert spread["median resistance"] == pytest.approx(1000, abs=2)
    (reason,) = reasons
    assert (reason.kind, reason.spec) == (Kind.SPEC, "yield")
    assert reason.reason.endswith("on the first that falls short, it is too high")
    spread, reasons = tolerate(circuit(loaded()), measure, below(1010), parts)
    assert spread["yield"] == 1
    assert not reasons


def test_a_design_must_meet_its_task_in_every_corner() -> None:
    from scoring import Board, cornered, parts, tolerate, unmet  # noqa: PLC0415 - importable only once the path holds it
    from verdict import Shortfall  # noqa: PLC0415 - importable only once the path holds it

    def measure(board: Board) -> dict[str, float]:
        return {"temperature": board.corner.temperature}

    def cool(found: dict[str, float]) -> list[Shortfall]:
        hot = [unmet("temperature", "it runs too hot")]
        return [] if found["temperature"] < 50 else hot

    measurements, reasons = cornered(measure, cool)
    assert measurements["temperature"] == 27
    assert measurements["at 70 °C with the supplies 5 % high: temperature"] == 70
    # In a corner, a shortfall keeps its spec and names the corner.
    assert reasons == [
        unmet("temperature", "at 70 °C with the supplies 5 % low, it runs too hot"),
        unmet("temperature", "at 70 °C with the supplies 5 % high, it runs too hot"),
    ]
    spread, (reason,) = tolerate(circuit(loaded()), measure, cool, parts)
    assert spread["yield"] == 0
    assert reason.reason.endswith("at 70 °C with the supplies 5 % low, it runs too hot")


TASKS = sorted(path.parent for path in BENCHMARKS.glob("*/task.md"))


def test_the_ngspice_arm_gets_spice_in_place_of_scripts() -> None:
    from arms import Arm, given, prompt  # noqa: PLC0415 - importable only once the path holds it

    for task in TASKS:
        assert prompt(task, Arm.JOLTY) == (task / "task.md").read_text(encoding="utf-8")
        written = prompt(task, Arm.NGSPICE)
        assert "jolty" not in written.casefold(), task.name
        assert "`ngspice`" in written
        names = [path.name for path in given(task, Arm.NGSPICE)]
        assert not [name for name in names if name.endswith(".py")], task.name
        assert names == sorted(names)
    assert [path.name for path in given(COST_DOWN, Arm.NGSPICE)] == [
        "bias.cir",
        "catalog.json",
        "stage.cir",
    ]
    assert "Write `design.cir`" in prompt(CUBIC, Arm.NGSPICE)


def _spice_defaults(parameters: Mapping[str, object]) -> dict[str, object]:
    """`parameters`, a jolty device's, with what SPICE states of the same device.

    SPICE gives a bipolar transistor a polarity and a substrate, vertical for an NPN
    and lateral for a PNP, joined to node 0; a MOSFET's size is its element's, and its
    level 1 the default the reader leaves out.
    """
    if "forward_beta" in parameters:
        polarity = parameters.get("polarity", "npn")
        substrate = "vertical" if polarity == "npn" else "lateral"
        return {**parameters, "polarity": polarity, "substrate": substrate}
    sized = {"width", "length"} | ({"level"} if parameters.get("level") == 1 else set())
    return {key: value for key, value in parameters.items() if key not in sized}


def test_the_given_spice_models_are_the_given_scripts_parts() -> None:
    from arms import Arm  # noqa: PLC0415 - importable only once the path holds it
    from scoring import given, modeled  # noqa: PLC0415 - importable only once the path holds it

    for task, names in (
        (BENCHMARKS / "balun", ("npn", "pnp")),
        (BENCHMARKS / "cubic_discrete", ("npn", "pnp")),
        (BENCHMARKS / "voltage_current", ("npn", "pnp")),
        (CUBIC, ("npn",)),
    ):
        for script, read in zip(
            given(task, Arm.JOLTY, *names),
            given(task, Arm.NGSPICE, *names),
            strict=True,
        ):
            assert read.parameters == _spice_defaults(script.parameters), task.name
    parts = runpy.run_path(str(ANALOGCODER / "given" / "parts.py"))
    for name in ("nmos", "pmos"):
        script = parts[name](1 * um, 1 * um)
        assert modeled(ANALOGCODER, name).parameters == _spice_defaults(
            script.parameters
        )


def test_the_given_stage_netlist_is_the_given_scripts_stage() -> None:
    bench = runpy.run_path(str(COST_DOWN / "given" / "bias.py"))["build"]()
    (script,) = bench.children
    read = spice.load(COST_DOWN / "ngspice" / "given" / "stage.cir").component("stage")
    assert [port.name for port in read.definition.ports] == [
        port.name for port in script.definition.ports
    ]
    found = {each.name: each for each in read.definition.instances}
    assert set(found) == {each.name for each in script.definition.instances}
    for instance in script.definition.instances:
        twin = found[instance.name]
        assert twin.of == instance.of
        assert twin.parameters == _spice_defaults(instance.parameters)
        assert {
            terminal: net
            for terminal, net in twin.terminals.items()
            if terminal != "substrate"
        } == instance.terminals


def stage(work: Path, tolerances: dict[str, float] | None, text: str = "") -> None:
    """Writes the given stage netlist as `work`'s design, and its `tolerances`.

    `text` replaces nothing when empty, else the netlist itself.
    """
    netlist = (COST_DOWN / "ngspice" / "given" / "stage.cir").read_text(
        encoding="utf-8"
    )
    (work / "design.cir").write_text(text or netlist, encoding="utf-8")
    if tolerances is not None:
        (work / "tolerances.json").write_text(json.dumps(tolerances), encoding="utf-8")


RESISTORS = ("R1", "R2", "RE", "RC")
"""The given stage's resistors."""


def test_a_cheaper_netlist_that_meets_the_specs_meets_the_task(tmp_path: Path) -> None:
    from arms import Arm  # noqa: PLC0415 - importable only once the path holds it
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    # Names are matched without case, as SPICE matches them.
    stage(tmp_path, {name.casefold(): 0.01 for name in RESISTORS})
    verdict = score(COST_DOWN, tmp_path, Arm.NGSPICE)
    assert verdict.met, verdict.reasons
    # The same parts as the cheaper script's, so the same prices.
    assert verdict.measurements == {
        "baseline": pytest.approx(0.2),
        "cost": pytest.approx(0.016),
    }


def test_the_score_says_why_a_netlist_falls_short(tmp_path: Path) -> None:
    from arms import Arm  # noqa: PLC0415 - importable only once the path holds it
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    assert score(COST_DOWN, tmp_path, Arm.NGSPICE).reasons == (
        f"there is no {tmp_path / 'design.cir'}",
    )
    stage(tmp_path, None)
    reasons = score(COST_DOWN, tmp_path, Arm.NGSPICE).reasons
    assert (
        "X1/R1 is unpriced: its resistance has no tolerance, which a part must meet"
        in (reasons)
    )
    stage(tmp_path, dict.fromkeys(RESISTORS, 0.05))
    (shortfall,) = score(COST_DOWN, tmp_path, Arm.NGSPICE).shortfalls
    assert (shortfall.kind, shortfall.spec) == ("spec", "yield")
    stage(tmp_path, {**dict.fromkeys(RESISTORS, 0.01), "R9": 0.01})
    (reason,) = score(COST_DOWN, tmp_path, Arm.NGSPICE).reasons
    assert "tolerances.json names r9, which is no resistor of the stage" in reason
    netlist = (COST_DOWN / "ngspice" / "given" / "stage.cir").read_text(
        encoding="utf-8"
    )
    stage(tmp_path, None, f"V1 supply 0 5\n{netlist}")
    (reason,) = score(COST_DOWN, tmp_path, Arm.NGSPICE).reasons
    assert reason.startswith("the design does not build: ")
    assert "a library defines" in reason
    assert reason.endswith("only: `V1 supply 0 5`")


def reference_netlists(work: Path) -> None:
    """Writes each of AnalogCoder's reference designs into `work` as a netlist."""
    from arms import Arm  # noqa: PLC0415 - importable only once the path holds it
    from stub import DESIGNS  # noqa: PLC0415 - importable only once the path holds it

    for name, text in DESIGNS["analogcoder"](ANALOGCODER, Arm.NGSPICE).items():
        (work / name).write_text(text, encoding="utf-8")


def test_the_reference_netlists_pass_every_analogcoder_check(tmp_path: Path) -> None:
    from arms import Arm  # noqa: PLC0415 - importable only once the path holds it
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    reference_netlists(tmp_path)
    verdict = score(ANALOGCODER, tmp_path, Arm.NGSPICE)
    assert verdict.met, verdict.reasons
    assert verdict.measurements["solved"] == 24


def test_the_analogcoder_scorer_says_why_netlists_fail(tmp_path: Path) -> None:
    from arms import Arm  # noqa: PLC0415 - importable only once the path holds it
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    reference_netlists(tmp_path)
    (tmp_path / "p3.cir").unlink()
    (tmp_path / "p7.cir").write_text(
        ".subckt p7 vdd vin vout\nE1 vout 0 vin 0 -1\n.ends p7\n", encoding="utf-8"
    )
    (tmp_path / "p8.cir").write_text(
        ".subckt p8 vdd vbias vout\nZ1 vout 0 vbias 1\n.ends p8\n", encoding="utf-8"
    )
    levels = json.loads((tmp_path / "levels.json").read_text(encoding="utf-8"))
    (tmp_path / "levels.json").write_text(
        json.dumps({**levels, "p1": {}}), encoding="utf-8"
    )
    verdict = score(ANALOGCODER, tmp_path, Arm.NGSPICE)
    assert verdict.measurements["solved"] == 20
    assert "p1: levels.json gives no level for vin" in verdict.reasons
    assert "p3: there is no p3.cir" in verdict.reasons
    assert "p7: E1 is a voltage_controlled_voltage_source, which the task forbids" in (
        verdict.reasons
    )
    (reason,) = [each for each in verdict.reasons if each.startswith("p8: ")]
    assert reason.startswith("p8: jolty cannot read p8.cir: p8.cir:2: ")


def test_a_netlist_s_transistors_must_be_the_given_models(tmp_path: Path) -> None:
    from arms import Arm  # noqa: PLC0415 - importable only once the path holds it
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    follower = ".subckt design input output vcc\nQ1 vcc input output {}\n.ends\n"
    (tmp_path / "design.cir").write_text(follower.format("npn"), encoding="utf-8")
    reasons = score(CUBIC_DISCRETE, tmp_path, Arm.NGSPICE).reasons
    assert reasons
    assert not [reason for reason in reasons if "given part" in reason]
    (tmp_path / "design.cir").write_text(
        f"{follower.format('fast')}.model fast NPN(IS=6.734e-15 BF=1000)\n",
        encoding="utf-8",
    )
    reasons = score(CUBIC_DISCRETE, tmp_path, Arm.NGSPICE).reasons
    assert reasons[0] == "Q1 is not a given part"


def test_a_script_that_fails_to_build_fails_its_task(tmp_path: Path) -> None:
    from score import score  # noqa: PLC0415 - importable only once the path holds it

    (tmp_path / "design.py").write_text(
        "def build():\n    raise RuntimeError('unfinished')\n", encoding="utf-8"
    )
    assert score(BALUN, tmp_path).reasons == (
        "design.py's build() fails: RuntimeError('unfinished')",
    )


SPEED_RESULTS = Path(__file__).parent.parent / "benchmarks" / "speed" / "results"


def test_ru_maxrss_to_bytes_converts_linux_kibibytes_and_macos_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert measuring.ru_maxrss_to_bytes(1024, "linux") == 1024 * 1024
    assert measuring.ru_maxrss_to_bytes(1024, "freebsd14") == 1024 * 1024
    assert measuring.ru_maxrss_to_bytes(1_048_576, "darwin") == 1_048_576
    assert measuring.ru_maxrss_to_bytes(0, "linux") == 0
    assert measuring.ru_maxrss_to_bytes(0, "darwin") == 0
    with pytest.raises(ValueError, match="non-negative"):
        measuring.ru_maxrss_to_bytes(-1, "linux")
    monkeypatch.setattr(sys, "platform", "darwin")
    assert measuring.ru_maxrss_to_bytes(4096) == 4096
    monkeypatch.setattr(sys, "platform", "linux")
    assert measuring.ru_maxrss_to_bytes(4096) == 4096 * 1024


def test_memory_formats_iec_binary_prefixes() -> None:
    assert measuring.memory(0) == "0 B"
    assert measuring.memory(512) == "512 B"
    assert measuring.memory(1024) == "1 KiB"
    assert measuring.memory(1536 * 1024) == "1.5 MiB"
    assert measuring.memory(2 * (1 << 30)) == "2 GiB"
    assert measuring.memory(3 * (1 << 40)) == "3 TiB"
    with pytest.raises(ValueError, match="non-negative"):
        measuring.memory(-1)


def test_linux_vm_hwm_and_available_memory_parsers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmHWM:\t  12345 kB\nVmRSS:\t 10000 kB\n", "utf-8")
    assert measuring._linux_vm_hwm_bytes(status) == 12345 * 1024  # noqa: SLF001 - unit testing parser
    status.write_text("Name:\tpython\nVmRSS:\t 10000 kB\n", "utf-8")
    assert measuring._linux_vm_hwm_bytes(status) is None  # noqa: SLF001 - unit testing parser
    assert measuring._linux_vm_hwm_bytes(tmp_path / "missing") is None  # noqa: SLF001 - unit testing parser

    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       65536000 kB\nMemFree:        10000000 kB\n"
        "MemAvailable:   32768000 kB\n",
        encoding="utf-8",
    )
    assert measuring.available_memory_bytes(meminfo) == 32_768_000 * 1024
    meminfo.write_text("MemTotal:       65536000 kB\n", encoding="utf-8")
    assert measuring.available_memory_bytes(meminfo) == 65_536_000 * 1024

    assert measuring.peak_memory_bytes() > 0

    class FakeUsage:
        ru_maxrss = 2048

    monkeypatch.setattr(resource, "getrusage", lambda _who: FakeUsage())
    monkeypatch.setattr(sys, "platform", "darwin")
    assert measuring.peak_memory_bytes() == 2048
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(measuring, "_linux_vm_hwm_bytes", lambda: None)
    assert measuring.peak_memory_bytes() == 2048 * 1024


def test_probe_captures_isolated_child_peak_memory_independent_of_parent() -> None:
    measured = timing.probe(timing.Request("bjt_amp", serial=True))
    assert measured.wall > 0
    assert measured.cpu > 0
    assert measured.peak_memory_bytes is not None
    assert measured.peak_memory_bytes > 0

    # Spawn an intermediate parent that allocates and touches 256 MiB before probing
    # `bjt_amp` in a fresh child, proving the child's peak RSS does not inherit the
    # parent's high-water mark (even on Linux where fork+exec copies signal->maxrss).
    parent_script = (
        "import json, measuring, timing\n"
        "buffer = bytearray(256 * 1024 * 1024)\n"
        "buffer[0::4096] = b'\\x01' * (len(buffer) // 4096)\n"
        "parent_peak = measuring.peak_memory_bytes()\n"
        "child = timing.probe(timing.Request('bjt_amp', serial=True))\n"
        "print(json.dumps({'parent': parent_peak, 'child': child.peak_memory_bytes}))\n"
    )
    environment = dict(os.environ)
    paths = [
        str(Path(__file__).parent.parent / "benchmarks"),
        str(Path(__file__).parent.parent / "benchmarks" / "speed"),
        str(Path(__file__).parent.parent / "benchmarks" / "speed" / "circuits"),
    ]
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        [*paths, *([existing] if existing is not None and existing != "" else [])]
    )
    done = subprocess.run(  # noqa: S603 - active Python interpreter
        [sys.executable, "-c", parent_script],
        capture_output=True,
        check=True,
        cwd=measuring.ROOT,
        env=environment,
        text=True,
    )
    reported = json.loads(done.stdout.strip().splitlines()[-1])
    parent_peak = reported["parent"]
    child_peak = reported["child"]
    assert isinstance(parent_peak, int)
    assert isinstance(child_peak, int)
    assert parent_peak >= 256 * 1024 * 1024
    assert child_peak > 0
    assert child_peak < parent_peak - 100 * 1024 * 1024


def test_timing_server_drops_peak_memory_when_reused_for_second_fixture() -> None:
    script = Path(__file__).parent.parent / "benchmarks" / "speed" / "timing.py"
    requests_text = (
        json.dumps({"fixture": "bjt_amp", "serial": True})
        + "\n"
        + json.dumps({"fixture": "ac_wide", "serial": True})
        + "\n"
    )
    done = subprocess.run(  # noqa: S603 - runs timing.py with active interpreter
        [sys.executable, str(script)],
        input=requests_text,
        capture_output=True,
        check=True,
        cwd=measuring.ROOT,
        text=True,
    )
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    assert len(lines) == 3
    first = json.loads(lines[1])
    second = json.loads(lines[2])
    assert first["peak_memory_bytes"] is not None
    assert first["peak_memory_bytes"] > 0
    assert second["peak_memory_bytes"] is None


def test_throughput_efficiency_math_and_edge_cases() -> None:
    # N == 1 baseline case.
    single = bench.throughput_efficiency(
        threads=1,
        serial_seconds=2.0,
        parallel_seconds=2.0,
        copies_wall_seconds=2.0,
        copy_seconds=[2.0],
    )
    assert single is not None
    assert single.speedup == pytest.approx(1.0)
    assert single.attainable_speedup == pytest.approx(1.0)
    assert single.efficiency == pytest.approx(1.0)
    assert single.rate_sum_attainable_speedup == pytest.approx(1.0)
    assert single.rate_sum_efficiency == pytest.approx(1.0)
    assert single.copy_spread_ratio == pytest.approx(1.0)

    # Homogeneous machine with all-core throttling: 8 copies take 12.5s vs 10s serial.
    throttled = bench.throughput_efficiency(
        threads=8,
        serial_seconds=10.0,
        parallel_seconds=2.5,
        copies_wall_seconds=12.5,
    )
    assert throttled is not None
    assert throttled.speedup == pytest.approx(4.0)
    assert throttled.attainable_speedup == pytest.approx(6.4)
    assert throttled.efficiency == pytest.approx(0.625)
    assert throttled.efficiency == pytest.approx(
        throttled.speedup / throttled.attainable_speedup
    )
    assert throttled.rate_sum_attainable_speedup is None
    assert throttled.copy_spread_ratio is None

    # Memory-bound copies contend more than one shared run: efficiency > 100%, never
    # clamped.
    superlinear = bench.throughput_efficiency(
        threads=8,
        serial_seconds=10.0,
        parallel_seconds=2.5,
        copies_wall_seconds=25.0,
    )
    assert superlinear is not None
    assert superlinear.attainable_speedup == pytest.approx(3.2)
    assert superlinear.efficiency == pytest.approx(1.25)

    # Hybrid cores (e.g. 8 performance cores @ 1.0s, 4 efficiency cores @ 2.5s).
    hybrid = bench.throughput_efficiency(
        threads=12,
        serial_seconds=1.0,
        parallel_seconds=0.2,
        copies_wall_seconds=2.5,
        copy_seconds=[1.0] * 8 + [2.5] * 4,
    )
    assert hybrid is not None
    assert hybrid.speedup == pytest.approx(5.0)
    assert hybrid.attainable_speedup == pytest.approx(12 * 1.0 / 2.5)
    assert hybrid.efficiency == pytest.approx(2.5 / (12 * 0.2))
    assert hybrid.rate_sum_attainable_speedup == pytest.approx(9.6)
    assert hybrid.rate_sum_efficiency == pytest.approx(5.0 / 9.6)
    assert hybrid.copy_spread_ratio == pytest.approx(2.5)

    # Zero, negative, and non-finite inputs yield None without ZeroDivisionError.
    for bad in (0.0, -1.0, math.nan, math.inf):
        assert (
            bench.throughput_efficiency(
                threads=4,
                serial_seconds=bad,
                parallel_seconds=1.0,
                copies_wall_seconds=1.0,
            )
            is None
        )
        assert (
            bench.throughput_efficiency(
                threads=4,
                serial_seconds=1.0,
                parallel_seconds=bad,
                copies_wall_seconds=1.0,
            )
            is None
        )
        assert (
            bench.throughput_efficiency(
                threads=4,
                serial_seconds=1.0,
                parallel_seconds=1.0,
                copies_wall_seconds=bad,
            )
            is None
        )

    with pytest.raises(ValueError, match="at least 1"):
        bench.throughput_efficiency(
            threads=0,
            serial_seconds=1.0,
            parallel_seconds=1.0,
            copies_wall_seconds=1.0,
        )
    with pytest.raises(ValueError, match="expected 4 copy times, not 2"):
        bench.throughput_efficiency(
            threads=4,
            serial_seconds=1.0,
            parallel_seconds=1.0,
            copies_wall_seconds=1.0,
            copy_seconds=[1.0, 1.0],
        )


def test_concurrent_copies_runs_real_workers_and_respects_memory_budget() -> None:
    measured = bench.concurrent_copies(
        "bjt_amp",
        2,
        bench.CopyOptions(rounds=2, warmup=False),
    )
    assert measured.copies == 2
    assert measured.rounds == 2
    assert measured.skipped_reason is None
    assert measured.wall_seconds is not None
    assert measured.wall_seconds > 0
    assert len(measured.round_seconds) == 2
    assert len(measured.copy_seconds) == 2
    assert measured.copy_seconds == sorted(measured.copy_seconds)
    assert measured.per_copy_peak_memory_bytes is not None
    assert measured.per_copy_peak_memory_bytes > 0
    wall_median = measured.wall_median()
    assert wall_median is not None
    assert wall_median.value == pytest.approx(measured.wall_seconds)

    # Budget check skips without launching subprocesses when copies * single > budget.
    skipped = bench.concurrent_copies(
        "bjt_amp",
        4,
        bench.CopyOptions(
            single_copy_memory_bytes=2 * (1 << 30),
            memory_budget_bytes=4 * (1 << 30),
        ),
    )
    assert skipped.copies == 4
    assert skipped.wall_seconds is None
    assert skipped.wall_median() is None
    assert skipped.skipped_reason is not None
    assert "8 GiB" in skipped.skipped_reason
    assert "4 GiB" in skipped.skipped_reason

    with pytest.raises(ValueError, match="copies must be at least 1"):
        bench.concurrent_copies("bjt_amp", 0)
    with pytest.raises(ValueError, match="rounds must be at least 1"):
        bench.concurrent_copies("bjt_amp", 1, bench.CopyOptions(rounds=0))


def test_speed_results_schema_round_trip_and_baseline_backwards_compatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        bench,
        "provenance",
        lambda: measuring.Provenance(
            commit="a" * 40,
            machine={"cpu": "test", "cores": 4, "rustc": None},
        ),
    )
    baseline_files = sorted(SPEED_RESULTS.glob("*.json"))
    assert baseline_files
    for baseline_file in baseline_files:
        loaded = bench.load_results(baseline_file)
        assert loaded.timings
        assert loaded.copies == {}
        for record in loaded.timings.values():
            assert record.peak_memory_bytes is None
            assert record.to_timing().peak_memory_bytes is None
        raw = measuring.load(baseline_file)
        for value in raw["timings"].values():
            assert bench.Timing.from_json(value).peak_memory_bytes is None

    # Timing round-trip with peak_memory_bytes (including zero and None).
    runs_with_memory = [
        timing.Run(1.0 + index / 100, 1.5, peak_memory_bytes=100_000 + index * 4096)
        for index in range(6)
    ]
    recorded = bench.Timing.of(runs_with_memory, 1.2, 1.4)
    assert recorded.peak_memory_bytes == 100_000 + 5 * 4096
    assert bench.Timing.from_json(recorded.to_json()) == recorded

    zero_memory = bench.Timing(seconds=1.0, runs=1, peak_memory_bytes=0)
    assert zero_memory.to_json()["peak_memory_bytes"] == 0
    assert bench.Timing.from_json(zero_memory.to_json()) == zero_memory

    no_memory = bench.Timing(seconds=1.0, runs=1)
    assert "peak_memory_bytes" not in no_memory.to_json()

    # Merging into a legacy baseline preserves old entries and round-trips new ones.
    copied_baseline = tmp_path / "merged.json"
    copied_baseline.write_text(baseline_files[0].read_text(encoding="utf-8"), "utf-8")
    copies_data = {
        "rc_mesh_300": {
            "4": bench.ConcurrentCopies(
                copies=4,
                rounds=3,
                wall_seconds=15.5,
                round_seconds=[15.2, 15.5, 15.9],
                copy_seconds=[15.0, 15.1, 15.4, 15.5],
                per_copy_peak_memory_bytes=320 * (1 << 20),
                load=1.0,
                load_after=1.5,
            ),
            "64": bench.ConcurrentCopies(
                copies=64,
                skipped_reason="estimated 20 GiB for 64 copies exceeds 16 GiB budget",
            ),
        }
    }
    document = bench.merged(
        copied_baseline,
        {"python/bjt_amp": recorded},
        copies=copies_data,
    )
    measuring.save(copied_baseline, document)
    round_tripped = bench.load_results(copied_baseline)
    assert "rust/bjt_amp/setup" in round_tripped.timings
    assert round_tripped.timings["python/bjt_amp"].peak_memory_bytes == (
        recorded.peak_memory_bytes
    )
    assert round_tripped.copies["rc_mesh_300"]["4"].wall_seconds == 15.5
    assert round_tripped.copies["rc_mesh_300"][
        "4"
    ].per_copy_peak_memory_bytes == 320 * (1 << 20)
    assert round_tripped.copies["rc_mesh_300"]["64"].wall_seconds is None
    assert "exceeds 16 GiB budget" in (
        round_tripped.copies["rc_mesh_300"]["64"].skipped_reason or ""
    )

    # Merging into an empty pre-created file (e.g. from mktemp) does not crash.
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("", encoding="utf-8")
    measuring.save(empty_file, bench.merged(empty_file, {"python/bjt_amp": recorded}))
    assert bench.load_results(empty_file).timings["python/bjt_amp"].to_timing() == (
        recorded
    )


def test_formatted_speed_scaling_and_comparison_tables(tmp_path: Path) -> None:
    timings_with_memory = {
        "python/bjt_amp": bench.Timing(
            seconds=0.002,
            runs=6,
            wall=(0.0019, 0.0020, 0.0020, 0.0020, 0.0020, 0.0021),
            peak_memory_bytes=100 * (1 << 20),
        ),
        "rust/bjt_amp/run": bench.Timing(seconds=0.001, runs=10),
    }
    summary = bench.format_speed_table(timings_with_memory)
    assert "peak memory" in summary
    assert "100 MiB" in summary
    assert "peak memory" not in bench.format_speed_table(
        {"rust/bjt_amp/run": bench.Timing(seconds=0.001, runs=10)}
    )

    # Against older baseline (no memory) and newer baseline (with memory delta).
    old_no_memory = {"python/bjt_amp": bench.Timing(seconds=0.002, runs=10)}
    text_vs_old = bench.format_against_table(
        {"python/bjt_amp": timings_with_memory["python/bjt_amp"]},
        old_no_memory,
        "legacy.json",
    )
    assert "peak memory" in text_vs_old
    assert "100 MiB" in text_vs_old
    old_with_memory = {
        "python/bjt_amp": bench.Timing(
            seconds=0.002, runs=10, peak_memory_bytes=100 * (1 << 20)
        )
    }
    now_higher_memory = {
        "python/bjt_amp": bench.Timing(
            seconds=0.002, runs=10, peak_memory_bytes=105 * (1 << 20)
        )
    }
    text_vs_new = bench.format_against_table(
        now_higher_memory, old_with_memory, "recent.json"
    )
    assert "105 MiB (+5.0%)" in text_vs_new

    # Scaling table with concurrent copies, hybrid-core spread note, and skipped entry.
    serial_timing = bench.Timing(
        seconds=10.0,
        runs=6,
        wall=(9.9, 10.0, 10.0, 10.0, 10.0, 10.1),
        peak_memory_bytes=200 * (1 << 20),
    )
    rows = [
        bench.ScalingRow("serially", None, serial_timing, "aaaaaaaaaaaaaaaa"),
        bench.ScalingRow(
            "8 threads",
            8,
            bench.Timing(
                seconds=2.5,
                runs=6,
                wall=(2.4, 2.5, 2.5, 2.5, 2.5, 2.6),
                peak_memory_bytes=250 * (1 << 20),
            ),
            "bbbbbbbbbbbbbbbb",
            bench.ConcurrentCopies(
                copies=8,
                rounds=3,
                wall_seconds=12.5,
                round_seconds=[12.4, 12.5, 12.6],
                copy_seconds=[12.5] * 8,
            ),
        ),
        bench.ScalingRow(
            "12 threads",
            12,
            bench.Timing(
                seconds=2.0,
                runs=6,
                wall=(1.9, 2.0, 2.0, 2.0, 2.0, 2.1),
                peak_memory_bytes=260 * (1 << 20),
            ),
            "cccccccccccccccc",
            bench.ConcurrentCopies(
                copies=12,
                rounds=3,
                wall_seconds=25.0,
                round_seconds=[24.8, 25.0, 25.2],
                copy_seconds=[10.0] * 8 + [25.0] * 4,
            ),
        ),
        bench.ScalingRow(
            "64 threads",
            64,
            bench.Timing(seconds=0.0, runs=1),
            "dddddddddddddddd",
            bench.ConcurrentCopies(
                copies=64,
                skipped_reason="estimated 128 GiB for 64 copies exceeds 64 GiB budget",
            ),
        ),
    ]
    scaling_text = bench.format_scaling_table("rc_mesh_300", "6 rounds", rows)
    for expected in (
        "N copies",
        "attainable",
        "efficiency",
        "peak memory",
        "6.40x",
        "62.5%",
        "250 MiB",
        "skipped",
        "64 copies skipped:",
        "2.50x spread across cores",
        "steady-state rate sum attainable",
    ):
        assert expected in scaling_text

    minimal_text = bench.format_scaling_table(
        "bjt_amp",
        "6 rounds",
        [bench.ScalingRow("serially", None, bench.Timing(seconds=1.0, runs=1), "1111")],
    )
    assert "N copies" not in minimal_text
    assert "peak memory" not in minimal_text

    # comparison() includes Peak memory section when present and omits it when absent.
    pair = (
        bench.Build("main", "a" * 40, tmp_path / "a"),
        bench.Build("HEAD", "b" * 40, tmp_path / "b"),
    )
    rounds_without = {
        "python/bjt_amp": bench.Rounds(
            [
                bench.Timing(seconds=1.0, runs=6, wall=(1.0,) * 6),
                bench.Timing(seconds=1.0, runs=6, wall=(1.0,) * 6),
            ],
            [(1.0, 1.0)] * 6,
        )
    }
    plain_md, _ = bench.comparison(pair, rounds_without)
    assert "| Peak memory |" not in plain_md

    rounds_with = {
        "python/bjt_amp": bench.Rounds(
            [
                bench.Timing(
                    seconds=1.0,
                    runs=6,
                    wall=(1.0,) * 6,
                    peak_memory_bytes=100 * (1 << 20),
                ),
                bench.Timing(
                    seconds=1.0,
                    runs=6,
                    wall=(1.0,) * 6,
                    peak_memory_bytes=110 * (1 << 20),
                ),
            ],
            [(1.0, 1.0)] * 6,
        )
    }
    memory_md, _ = bench.comparison(pair, rounds_with)
    assert "| Peak memory |" in memory_md
    assert "100 MiB" in memory_md
    assert "110 MiB" in memory_md
    assert "+10.0%" in memory_md


def test_bench_cli_flags_and_python_single_shot_memory_probe() -> None:
    defaults = bench.Bench(_cli_parse_args=[])
    assert defaults.peak_memory is True
    assert defaults.concurrent_copies is True
    assert defaults.copy_rounds == bench.COPY_ROUNDS

    custom = bench.Bench(
        _cli_parse_args=[
            "--no-peak-memory",
            "--no-concurrent-copies",
            "--copy-rounds",
            "5",
        ]
    )
    assert custom.peak_memory is False
    assert custom.concurrent_copies is False
    assert custom.copy_rounds == 5

    found = timing.planned()
    measured_with_memory = bench.python(
        ["bjt_amp"], found, scale=True, measure_memory=True
    )
    entry = measured_with_memory["python/bjt_amp"]
    assert entry.runs == 1
    assert entry.seconds > 0
    assert entry.peak_memory_bytes is not None
    assert entry.peak_memory_bytes > 0

    measured_without_memory = bench.python(
        ["bjt_amp"], found, scale=True, measure_memory=False
    )
    assert measured_without_memory["python/bjt_amp"].peak_memory_bytes is None


def test_interpolate_transient_removes_chord_sag_without_corner_overshoot() -> None:
    """Huynh (1993) cubic Hermite eliminates sine-peak sag and avoids step overshoot."""
    # 1. Synthetic 10 kHz sine sampled at 4 us straddling the midpoint peak at 125 us:
    # linear interpolation sags by A * omega^2 * h^2 / 8 = 3.94e-3 of the 2 V swing;
    # cubic Hermite cuts this error by >100x.
    coarse_times = np.arange(0.0, 200e-6 + 1e-12, 4e-6) + 1e-6
    coarse_values = np.sin(2.0 * np.pi * 1e4 * coarse_times)
    fine_times = np.linspace(coarse_times[0], coarse_times[-1], 2001)
    exact_values = np.sin(2.0 * np.pi * 1e4 * fine_times)
    linear_error = (
        float(
            np.max(
                np.abs(
                    np.interp(fine_times, coarse_times, coarse_values) - exact_values
                )
            )
        )
        / 2.0
    )
    cubic_error = (
        float(
            np.max(
                np.abs(
                    bench.interpolate_transient(fine_times, coarse_times, coarse_values)
                    - exact_values
                )
            )
        )
        / 2.0
    )
    assert linear_error > 3.5e-3
    assert cubic_error < 5e-5

    # 2. Sharp switching edge with 100x step-size jump: strictly zero overshoot.
    step_times = np.array([0.0, 1.0, 2.0, 2.01, 2.02, 3.0, 4.0, 5.0])
    step_values = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    step_query = np.linspace(0.0, 5.0, 5001)
    step_interp = bench.interpolate_transient(step_query, step_times, step_values)
    assert float(step_interp.min()) == 0.0
    assert float(step_interp.max()) == 1.0

    # 3. Sharp triangle corner (switching inductor current): strictly zero overshoot.
    tri_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    tri_values = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
    tri_query = np.linspace(0.0, 6.0, 6001)
    tri_interp = bench.interpolate_transient(tri_query, tri_times, tri_values)
    assert float(tri_interp.min()) == 0.0
    assert float(tri_interp.max()) == 3.0

    # 4. Duplicate breakpoint timestamps and tiny arrays (< 4 points) stay finite.
    dup_times = np.array([0.0, 1.0, 1.0, 2.0, 3.0, 4.0])
    dup_values = np.array([0.0, 1.0, 1.0, 4.0, 9.0, 16.0])
    assert bench.interpolate_transient(np.array([1.5]), dup_times, dup_values)[
        0
    ] == pytest.approx(2.25, abs=1e-12)
    short_times = np.array([0.0, 1.0, 2.0])
    short_values = np.array([0.0, 1.0, 2.0])
    assert bench.interpolate_transient(
        np.array([0.5, 1.5]), short_times, short_values
    ) == pytest.approx([0.5, 1.5], abs=1e-15)

    # 5. Exact interior reconstruction of a non-uniform cubic polynomial.
    poly_times = np.array([0.0, 0.3, 0.8, 1.5, 2.1, 2.9, 3.5, 4.2, 5.0])
    poly_values = 0.2 * poly_times**3 - 1.1 * poly_times**2 + 2.5 * poly_times - 0.7
    interior = np.linspace(1.5, 2.9, 50)
    expected_interior = 0.2 * interior**3 - 1.1 * interior**2 + 2.5 * interior - 0.7
    reconstructed = bench.interpolate_transient(interior, poly_times, poly_values)
    assert reconstructed == pytest.approx(expected_interior, abs=1e-12)


def test_deviations_bjt_amp_transient_has_no_chord_sag() -> None:
    """`deviations()` agrees on `bjt_amp` within 5e-5 vs 0.1 us fine reference."""
    loaded = timing.load_fixture(timing.planned()["bjt_amp"])
    plan_tran = loaded.plan.analyses["tran"]
    assert isinstance(plan_tran, Transient)
    simulation = Simulation(loaded.circuit)
    tight = simulation.run(plan_tran, TIGHT)
    fine = simulation.run(
        Transient(stop=plan_tran.stop, max_step=0.1 * us),
        TIGHT,
    )
    ref = {"tran": bench.jolty_voltages(tight, ("out",))}
    cand = {"tran": bench.jolty_voltages(fine, ("out",))}
    # Self-comparison on identical grid is exactly 0.
    assert bench.deviations({"tran": plan_tran}, ref, ref)["tran"] == 0.0
    # Against 0.1 us fine grid, linear np.interp gave 3.9e-3; Huynh cubic gives ~1.8e-5.
    found_deviation = bench.deviations({"tran": plan_tran}, ref, cand)["tran"]
    assert found_deviation < 5e-5
