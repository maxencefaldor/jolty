# Copyright 2026 Maxence Faldor

import math
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from jolty import (
    Ac,
    Builder,
    OperatingPoint,
    Role,
    Simulation,
    circuit,
    component,
    devices,
)
from jolty.units import CURRENT, A, MHz, Quantity, V, kohm, nF

BENCHMARKS = Path(__file__).parent.parent / "benchmarks"
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

    assert score(COST_DOWN, tmp_path).reasons == (
        f"there is no {tmp_path / 'design.py'}",
    )
    design(tmp_path, "0.001")
    assert score(COST_DOWN, tmp_path).reasons == (
        "the design costs 0.2, not less than 0.2",
    )
    design(tmp_path, "0.05")
    reasons = score(COST_DOWN, tmp_path).reasons
    assert "the design does not meet the plan of bias.py" in reasons
    original = (COST_DOWN / "given" / "bias.py").read_text(encoding="utf-8")
    (tmp_path / "design.py").write_text(
        original.replace("TOLERANCE = 0.001", "TOLERANCE = 0.01").replace(
            "voltage_source(5 * V)", "voltage_source(5.1 * V)"
        ),
        encoding="utf-8",
    )
    assert "the bench's V1 differs" in score(COST_DOWN, tmp_path).reasons


def test_the_runner_runs_the_agent_in_a_copy_of_the_task(tmp_path: Path) -> None:
    from run import run  # noqa: PLC0415 - importable only once the path holds it

    agent = [
        sys.executable,
        "-c",
        (
            "import pathlib, sys; source = pathlib.Path('bias.py').read_text(); "
            "pathlib.Path('design.py').write_text("
            "source.replace('TOLERANCE = 0.001', 'TOLERANCE = 0.01')); "
            "print(sys.argv[1].splitlines()[0])"
        ),
    ]
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
        "        load = c.place(devices.resistor(1 * kohm), id=f'R_{name}')\n"
        "        c.connect(c.port(name), load.a)\n"
        "        c.connect(load.b, ground)\n"
        "    clamp = c.place(devices.diode(saturation_current=1e-14 * A), id='D9')\n"
        "    c.connect(c.port('output'), clamp.anode)\n"
        "    c.connect(clamp.cathode, ground)\n"
        "build = cubic\n",
        encoding="utf-8",
    )
    verdict = score(CUBIC, tmp_path)
    assert not verdict.met
    assert "D9 is a diode, which the task forbids" in verdict.reasons


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
        "id='E1')\n"
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
        "    resistor = c.place(devices.resistor(10 * kohm), id='R1')\n"
        "    capacitor = c.place(devices.capacitor(3 * uF), id='CF')\n"
        "    rest = c.place(devices.voltage_source(2.5 * V), id='VR')\n"
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
        "id='G1')\n"
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
        "    resistor = c.place(devices.resistor(1 * kohm), id='R1')\n"
        "    c.connect(resistor.a, plus)\n"
        "    c.connect(resistor.b, output)\n",
    )
    verdict = score(VOLTAGE_CURRENT, tmp_path)
    assert not verdict.met
    (reason,) = verdict.reasons
    assert reason.startswith("the fitness is ")
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
        "    bias = c.place(devices.resistor(1000 * Mohm), id='R1')\n"
        "    c.connect(bias.a, source)\n"
        "    c.connect(bias.b, ground)\n"
        "    gains = (('E0', in_phase, 0.5), ('E1', out_of_phase, -0.5))\n"
        "    for name, port, gain in gains:\n"
        "        amplifier = c.place(devices.voltage_controlled_voltage_source(gain), "
        "id=name)\n"
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
        "    series = c.place(devices.resistor(75 * ohm), id='R1')\n"
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
        "    series = c.place(devices.resistor(75 * ohm), id='R1')\n"
        "    c.connect(series.a, source)\n",
    )
    verdict = score(BALUN, tmp_path)
    (reason,) = verdict.reasons
    assert reason.startswith("the simulation fails: ")


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
        "    sense = c.place(devices.resistor(1 * kohm), id='R1')\n"
        "    c.connect(sense.a, signal)\n"
        "    c.connect(sense.b, ground)\n"
        "    expression = '(v(s) / 0.1) ^ 3 / 10500'\n"
        "    cube = c.place(devices.behavioral_current_source(expression), id='B1')\n"
        "    c.connect(cube.positive, output)\n"
        "    c.connect(cube.negative, ground)\n"
        "    c.connect(cube.s, signal)\n"
        "build = cubic\n",
        encoding="utf-8",
    )
    verdict = score(CUBIC_DISCRETE, tmp_path)
    assert verdict.measurements["mean absolute error"] < 1e-9
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
        "    stages = [c.place(npn(), id=f'Q{index}') for index in range(3)]\n"
        "    for index, stage in enumerate(stages):\n"
        "        load = c.place(devices.resistor(10 * kohm), id=f'RC{index}')\n"
        "        drive = c.place(devices.resistor(100 * kohm), id=f'RB{index}')\n"
        "        slow = c.place(devices.capacitor(100 * pF), id=f'C{index}')\n"
        "        c.connect(load.a, minus)\n"
        "        c.connect(load.b, stage.collector, drive.a, slow.a)\n"
        "        c.connect(drive.b, stages[(index + 1) % 3].base)\n"
        "        c.connect(slow.b, stage.emitter, vee)\n"
        "    out = c.place(devices.resistor(1 * kohm), id='RO')\n"
        "    c.connect(out.a, stages[0].collector)\n"
        "    c.connect(out.b, output)\n",
    )
    verdict = score(VOLTAGE_CURRENT, tmp_path)
    assert any("still swings" in reason for reason in verdict.reasons)


@component
def parallel(c: Builder) -> None:
    """A design of a resistor beside a capacitor, which gives the resistor 20 %."""
    a, b = c.port("a"), c.port("b")
    resistor = c.place(devices.resistor(1 * kohm, tolerance=0.2), id="R1")
    capacitor = c.place(devices.capacitor(1 * nF), id="C1")
    c.connect(resistor.a, capacitor.a, a)
    c.connect(resistor.b, capacitor.b, b)


@component
def loaded(c: Builder) -> None:
    """`parallel` on a bench that loads it with an exact resistor."""
    ground = c.port("ground", Role.GROUND)
    placed = c.place(parallel(), id="X1")
    load = c.place(devices.resistor(1 * kohm), id="RLOAD")
    c.connect(placed.a, load.a, name="a")
    c.connect(placed.b, load.b, ground)


def test_boards_draw_the_designs_resistors_and_capacitors_alone() -> None:
    from scoring import BOARDS, boards  # noqa: PLC0415 - importable only once the path holds it

    drawn = boards(circuit(loaded()))
    assert len(drawn) == BOARDS
    assert all(set(board) == {"X1/R1", "X1/C1"} for board in drawn)
    for path, parameter, nominal, tolerance in (
        ("X1/R1", "resistance", 1e3, 0.01),
        ("X1/C1", "capacitance", 1e-9, 0.05),
    ):
        strays = [abs(board[path][parameter] / nominal - 1) for board in drawn]
        assert 0.9 * tolerance < max(strays) <= tolerance
    assert boards(circuit(loaded())) == drawn


def test_too_few_boards_that_meet_the_task_fall_short() -> None:
    from scoring import tolerate  # noqa: PLC0415 - importable only once the path holds it

    def measure(board: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
        return {"resistance": board["X1/R1"]["resistance"]}

    def below(most: float) -> Callable[[dict[str, float]], list[str]]:
        return lambda found: [] if found["resistance"] <= most else ["it is too high"]

    spread, reasons = tolerate(circuit(loaded()), measure, below(1005))
    assert 0.6 < spread["yield"] < 0.9
    assert spread["median resistance"] == pytest.approx(1000, abs=2)
    (reason,) = reasons
    assert reason.endswith("on the first that falls short, it is too high")
    spread, reasons = tolerate(circuit(loaded()), measure, below(1010))
    assert spread["yield"] == 1
    assert not reasons


@pytest.mark.parametrize("name", ["IRF511", "IRF9230", "IRFZ44", "MTP50P03HDL"])
def test_power_mosfets_meet_their_datasheets(name: str) -> None:
    import power_mosfets  # noqa: PLC0415 - importable only once the path holds it

    sheet = getattr(power_mosfets, name)
    sign = 1.0 if sheet.channel == "n" else -1.0

    def bench(gate: float | None, drain: Quantity) -> Simulation:
        """The transistor, its gate at `gate` volts, its drain held or fed.

        The drain takes a voltage or a current, both signed as its channel conducts.
        Without a gate voltage, the gate is tied to the drain, which leaves the
        symmetric channel off whichever way the current flows.
        """

        @component
        def fixture(c: Builder) -> None:
            ground = c.port("ground", Role.GROUND)
            placed = c.place(power_mosfets.power_mosfet(sheet), id="X1")
            c.connect(placed.source, ground)
            if gate is None:
                c.connect(placed.gate, placed.drain)
            else:
                gate_source = c.place(devices.voltage_source(sign * gate * V), id="VG")
                c.connect(gate_source.positive, placed.gate, name="gate")
                c.connect(gate_source.negative, ground)
            if drain.dimension == CURRENT:
                fed = c.place(devices.current_source(sign * drain), id="ID")
                c.connect(fed.negative, placed.drain, name="drain")
                c.connect(fed.positive, ground)
            else:
                held = c.place(devices.voltage_source(sign * drain), id="VD")
                c.connect(held.positive, placed.drain, name="drain")
                c.connect(held.negative, ground)

        return Simulation(fixture())

    on = bench(sheet.on_gate, sheet.on_current * A).run(OperatingPoint())
    resistance = abs(on.voltage("drain")) / sheet.on_current
    assert resistance == pytest.approx(sheet.on_resistance, rel=1e-6)
    drop = bench(None, -sheet.diode_current * A).run(OperatingPoint())
    assert abs(drop.voltage("drain")) == pytest.approx(sheet.diode_drop, rel=1e-3)
    beta = power_mosfets.transconductance_parameter(sheet)
    overdrive = (2 * sheet.transconductance_current / beta) ** 0.5
    currents = [
        abs(
            bench(sheet.threshold + overdrive + step, 20 * V)
            .run(OperatingPoint())
            .current("VD")
        )
        for step in (-1e-3, 1e-3)
    ]
    assert (currents[1] - currents[0]) / 2e-3 >= sheet.transconductance * (1 - 1e-3)
    still = bench(0.0, sheet.capacitance_voltage * V)
    omega = 2 * math.pi * 1e6
    through_gate = still.run(Ac("VG", [1 * MHz]))
    through_drain = still.run(Ac("VD", [1 * MHz]))
    found = {
        "input": abs(through_gate.current("VG")[0]) / omega,
        "reverse": abs(through_drain.current("VG")[0]) / omega,
        "output": abs(through_drain.current("VD")[0]) / omega,
    }
    for kind, capacitance in found.items():
        stated = getattr(sheet, f"{kind}_capacitance")
        assert capacitance == pytest.approx(stated, rel=1e-3), kind
