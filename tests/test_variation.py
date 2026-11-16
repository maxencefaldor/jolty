# Copyright 2026 Maxence Faldor

import math
from pathlib import Path

import numpy as np
import pytest

from jolty import (
    Builder,
    ElaborationError,
    Role,
    Simulation,
    circuit,
    component,
    devices,
    files,
    ir,
)
from jolty.plan import MonteCarlo, OperatingPoint, WorstCase
from jolty.runner import BATCHES, MONTE_CARLO, Built, Job, run
from jolty.specs import Specs, value
from jolty.units import A, V, kohm, nF
from jolty.variation import Varied, extreme, moved, sample, toleranced


@component
def cell(c: Builder) -> None:
    a = c.port("a")
    ground = c.port("ground", Role.GROUND)
    resistor = c.place(devices.resistor(1 * kohm, tolerance=0.05), name="R1")
    c.connect(a, resistor.a)
    c.connect(resistor.b, ground)


@component
def pair(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1 * V), name="V1")
    for id in ("X1", "X2"):
        placed = c.place(cell(), name=id)
        c.connect(source.positive, placed.a)
        c.connect(placed.ground, ground)
    c.connect(source.negative, ground)


def test_each_placement_of_a_component_varies_apart() -> None:
    varied = toleranced(circuit(pair()))
    assert [(parameter.path, parameter.parameter) for parameter in varied] == [
        ("X1/R1", "resistance"),
        ("X2/R1", "resistance"),
    ]
    drawn = sample(varied, 3, 0)
    assert drawn["X1/R1"]["resistance"] != drawn["X2/R1"]["resistance"]
    assert drawn == sample(varied, 3, 0)
    assert drawn != sample(varied, 3, 1)
    point = Simulation(pair(), overrides=drawn).run(OperatingPoint())
    expected = 1 / drawn["X1/R1"]["resistance"] + 1 / drawn["X2/R1"]["resistance"]
    assert -point.current("V1") == pytest.approx(expected)


def test_values_spread_within_their_tolerance() -> None:
    device = devices.resistor(1 * kohm).toleranced("resistance", 0.03, "normal")
    varied = (
        *toleranced(circuit(pair())),
        *toleranced(_alone(device)),
    )
    drawn = [sample(varied, 0, index) for index in range(4000)]
    uniform = np.array([d["X1/R1"]["resistance"] for d in drawn]) / 1e3 - 1
    normal = np.array([d["R1"]["resistance"] for d in drawn]) / 1e3 - 1
    assert np.abs(uniform).max() <= 0.05
    assert uniform.std() == pytest.approx(0.05 / math.sqrt(3), rel=0.05)
    assert normal.std() == pytest.approx(0.01, rel=0.05)


def test_a_tolerance_needs_a_number_below_one() -> None:
    with pytest.raises(ValueError, match="gives inductance no number"):
        devices.resistor(1 * kohm).toleranced("inductance", 0.1)
    with pytest.raises(ValueError, match="needs to be below 1"):
        devices.resistor(1 * kohm, tolerance=1.5)
    built = circuit(pair())
    top = built.components["cell"]
    (resistor,) = top.instances
    raw = resistor.model_copy(
        update={"tolerances": {"missing": ir.Tolerance(relative=0.1)}}
    )
    broken = built.model_copy(
        update={
            "components": {
                **built.components,
                "cell": top.model_copy(update={"instances": (raw,)}),
            }
        }
    )
    with pytest.raises(ElaborationError, match=r"gives `missing` a tolerance of 0\.1"):
        Simulation(broken)
    with pytest.raises(ElaborationError, match="no device at `X9/R1`"):
        Simulation(pair(), overrides={"X9/R1": {"resistance": 1.0}})


DIVIDER = """
from jolty import Builder, MonteCarlo, OperatingPoint, Plan, Role, Specs, component
from jolty import devices
from jolty.specs import value
from jolty.units import V, kohm


@component
def divider(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(5 * V), name="V1")
    upper = c.place(devices.resistor(1 * kohm, tolerance=0.01), name="R1")
    lower = c.place(
        devices.resistor(1 * kohm).toleranced("resistance", 0.01, "normal"), name="R2"
    )
    c.connect(source.positive, upper.a)
    c.connect(upper.b, lower.a, name="out")
    c.connect(lower.b, source.negative, ground)


def build():
    return divider()


def plan():
    return Plan({"op": OperatingPoint()}, monte_carlo=MonteCarlo(SAMPLES, seed=1))


def specs():
    return Specs([value("op", "out").within(2.5 * V, 7.5e-3 * V)], yield_target=0.9)
"""

SAMPLES = 2000


def test_the_yield_matches_an_independent_monte_carlo_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "divider.py"
    script.write_text(DIVIDER.replace("SAMPLES", str(SAMPLES)), encoding="utf-8")
    outcome = run(Job(script=script, plan=script, max_unknowns=10), tmp_path)
    assert isinstance(outcome, Built), str(outcome)
    report = outcome.monte_carlo
    assert report is not None
    assert (report.samples, report.toleranced, report.failed) == (SAMPLES, 2, 0)
    assert report.passing is not None
    (spread,) = report.spreads
    assert spread.passing == report.passing.passing
    assert spread.values is not None
    assert spread.values.mean == pytest.approx(2.5, abs=1e-3)

    # The same divider, sampled by numpy alone with another seed.
    generator = np.random.default_rng(12345)
    upper = 1e3 * (1 + generator.uniform(-0.01, 0.01, SAMPLES))
    lower = 1e3 * (1 + generator.normal(0, 0.01 / 3, SAMPLES))
    out = 5 * lower / (upper + lower)
    independent = float(np.mean(np.abs(out - 2.5) <= 7.5e-3))
    estimate = report.passing.estimate
    error = math.sqrt(2 * independent * (1 - independent) / SAMPLES)
    assert abs(estimate - independent) <= 4 * error
    assert report.passing.lower <= independent <= report.passing.upper
    assert report.passing.met is (estimate >= 0.9)


def test_a_yield_target_lies_above_0_and_at_most_1() -> None:
    assert Specs([value("op", "out") >= 0 * V], yield_target=1).yield_target == 1
    with pytest.raises(ValueError, match="a yield target lies above 0"):
        Specs([value("op", "out") >= 0 * V], yield_target=0)


def _alone(device: devices.Device) -> ir.Circuit:
    @component
    def alone(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        placed = c.place(device, name="R1")
        c.connect(placed.a, placed.b, ground)

    return circuit(alone())


def test_a_monte_carlo_analysis_draws_a_sample_or_more() -> None:
    with pytest.raises(ValueError, match="a sample or more, not 0"):
        MonteCarlo(0)


def test_a_supply_takes_a_tolerance_by_its_path() -> None:
    supply = devices.voltage_source(5 * V).toleranced("waveform.dc", 0.05)

    @component
    def divider(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(supply, name="V1")
        upper = c.place(devices.resistor(1 * kohm), name="R1")
        lower = c.place(devices.resistor(1 * kohm), name="R2")
        c.connect(source.positive, upper.a)
        c.connect(upper.b, lower.a, name="out")
        c.connect(lower.b, source.negative, ground)

    built = circuit(divider())
    (varied,) = toleranced(built)
    assert (varied.path, varied.parameter, varied.nominal) == ("V1", "waveform.dc", 5)
    drawn = sample((varied,), 0, 0)
    level = drawn["V1"]["waveform.dc"]
    assert 4.75 <= level <= 5.25
    point = Simulation(built, overrides=drawn).run(OperatingPoint())
    assert point.voltage("out") == pytest.approx(level / 2)
    with pytest.raises(ValueError, match=r"gives waveform\.sine no number"):
        devices.voltage_source(5 * V).toleranced("waveform.sine", 0.05)
    with pytest.raises(ElaborationError, match=r"`waveform\.sine\.offset` has no"):
        Simulation(built, overrides={"V1": {"waveform.sine.offset": 1.0}})


def test_the_parts_of_a_lot_share_its_factor() -> None:
    die = ir.Lot(name="die", relative=0.2)

    @component
    def integrated(c: Builder, lot: ir.Lot = die) -> None:
        ground = c.port("ground", Role.GROUND)
        for id in ("R1", "R2"):
            matched = devices.resistor(1 * kohm).toleranced(
                "resistance", 0.001, lot=lot
            )
            placed = c.place(matched, name=id)
            c.connect(placed.a, placed.b, ground)
        held = devices.capacitor(1 * nF).toleranced("capacitance", 0.001, lot=lot)
        capacitor = c.place(held, name="C1")
        c.connect(capacitor.a, capacitor.b, ground)

    varied = toleranced(circuit(integrated()))
    drawn = [sample(varied, 0, index) for index in range(2000)]
    first, second = (
        np.array([d[id]["resistance"] for d in drawn]) / 1e3 - 1 for id in ("R1", "R2")
    )
    capacitance = np.array([d["C1"]["capacitance"] for d in drawn]) / 1e-9 - 1
    assert np.abs((1 + first) / (1 + second) - 1).max() <= 0.0021
    assert first.std() == pytest.approx(0.2 / math.sqrt(3), rel=0.05)
    assert abs(np.corrcoef(first, capacitance)[0, 1]) < 0.1
    wider = ir.Lot(name="die", relative=0.3)

    @component
    def mixed(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        for id, lot in (("X1", die), ("X2", wider)):
            placed = c.place(integrated(lot), name=id)
            c.connect(placed.ground, ground)

    with pytest.raises(ElaborationError, match="spread `resistance` of lot `die`"):
        Simulation(mixed())


def test_an_exponentials_argument_spreads_in_its_logarithm() -> None:
    reel = ir.Lot(name="reel", relative=35.0, distribution="log_uniform")
    transistor = devices.bipolar_transistor(saturation_current=1e-14 * A).toleranced(
        "saturation_current", math.expm1(1e-3 / 0.02585), "log_uniform", lot=reel
    )

    @component
    def pair(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        for id in ("Q1", "Q2"):
            placed = c.place(transistor, name=id)
            c.connect(placed.collector, placed.base, placed.emitter, ground)

    varied = toleranced(circuit(pair()))
    nominal = varied[0].nominal
    drawn = [sample(varied, 0, index) for index in range(2000)]
    first, second = (
        np.log(np.array([d[id]["saturation_current"] for d in drawn]) / nominal)
        for id in ("Q1", "Q2")
    )
    assert np.abs(first).max() <= math.log(36) + 1e-3 / 0.02585 + 1e-12
    assert first.std() == pytest.approx(math.log(36) / math.sqrt(3), rel=0.05)
    # Base-emitter voltages within 2 mV of each other, at 0.02585 V per neper.
    assert np.abs(first - second).max() * 0.02585 <= 2e-3 + 1e-12
    with pytest.raises(ValueError, match="only a logarithmic one may exceed it"):
        ir.Tolerance(relative=1.5)
    assert ir.Tolerance(relative=1.5, distribution="log_normal").relative == 1.5


CORNERED = """
from jolty import Builder, Corner, MonteCarlo, OperatingPoint, Plan, Role, Specs
from jolty import component, devices
from jolty.specs import value
from jolty.units import V, kohm


@component
def divider(c: Builder, supply) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(supply), name="V1")
    upper = c.place(devices.resistor(1 * kohm, tolerance=0.01), name="R1")
    lower = c.place(devices.resistor(1 * kohm, tolerance=0.01), name="R2")
    c.connect(source.positive, upper.a)
    c.connect(upper.b, lower.a, name="out")
    c.connect(lower.b, source.negative, ground)


def build(supply=5 * V):
    return divider(supply)


def plan():
    return Plan(
        {"op": OperatingPoint()},
        corners={"low": Corner(supply=4.9 * V)},
        monte_carlo=MonteCarlo(50, seed=2),
    )


def specs():
    return Specs([value("op", "out") >= 2.48 * V])
"""


def test_each_sample_runs_on_the_same_parts_at_every_corner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "divider.py"
    script.write_text(CORNERED, encoding="utf-8")
    outcome = run(Job(script=script, plan=script, max_unknowns=10), tmp_path)
    assert isinstance(outcome, Built), str(outcome)
    report = outcome.monte_carlo
    assert report is not None
    assert (report.corners, report.failed) == (("low",), 0)
    assert "at low, seed 2" in report.describe()
    assert report.passing is not None
    assert report.passing.passing == 0
    # Every sample reads worst at the low supply, on the parts the seed draws.
    (spread,) = report.spreads
    assert spread.values is not None
    varied = [
        Varied(path, "resistance", 1e3, ir.Tolerance(relative=0.01))
        for path in ("R1", "R2")
    ]
    drawn = [sample(varied, 2, index) for index in range(50)]
    low = [
        4.9 * d["R2"]["resistance"] / (d["R1"]["resistance"] + d["R2"]["resistance"])
        for d in drawn
    ]
    assert spread.values.minimum == pytest.approx(min(low), rel=1e-6)
    assert spread.values.maximum == pytest.approx(max(low), rel=1e-6)


WORST = """
from jolty import Builder, Corner, OperatingPoint, Plan, Role, Specs, WorstCase
from jolty import component, devices, ir
from jolty.specs import value
from jolty.units import V, kohm

DIE = ir.Lot(name="die", relative=0.2)


@component
def divider(c: Builder, supply) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(supply), name="V1")
    upper = c.place(devices.resistor(1 * kohm, tolerance=0.01), name="R1")
    lower = c.place(
        devices.resistor(1 * kohm).toleranced("resistance", 0.01, "normal"), name="R2"
    )
    c.connect(source.positive, upper.a)
    c.connect(upper.b, lower.a, name="out")
    c.connect(lower.b, source.negative, ground)
    # A matched pair on one die, whose shared spread cancels in their ratio.
    top = c.place(
        devices.resistor(1 * kohm).toleranced("resistance", 0.001, lot=DIE), name="R3"
    )
    bottom = c.place(
        devices.resistor(1 * kohm).toleranced("resistance", 0.001, lot=DIE), name="R4"
    )
    c.connect(source.positive, top.a)
    c.connect(top.b, bottom.a, name="matched")
    c.connect(bottom.b, ground)


def build(supply=5 * V):
    return divider(supply)


def plan():
    return Plan(
        {"op": OperatingPoint()},
        corners={"low": Corner(supply=4.9 * V)},
        worst_case=WorstCase(),
    )


def specs():
    return Specs(
        [
            value("op", "out").within(2.5 * V, 0.02 * V),
            value("op", "matched").within(2.5 * V, 3e-3 * V),
        ]
    )
"""


def test_a_worst_case_sets_each_tolerance_at_the_extreme_that_worsens_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "divider.py"
    script.write_text(WORST, encoding="utf-8")
    outcome = run(Job(script=script, plan=script, max_unknowns=10), tmp_path)
    assert isinstance(outcome, Built), str(outcome)
    report = outcome.worst_case
    assert report is not None
    # Four parameters moved at each of two conditions, then each bound's extremes.
    assert (report.toleranced, report.corners, report.runs) == (4, ("low",), 16)
    divided, matched = report.verdicts
    # R1 up and R2 down by their whole tolerance, at the low supply.
    assert divided.value == pytest.approx(4.9 * 0.99 / 2, rel=1e-9)
    assert (divided.status, divided.corner) == ("failed", "low")
    # The die's spread cancels in the ratio, so only the parts' own tolerances count.
    assert matched.value == pytest.approx(4.9 * 0.999 / 2, rel=1e-9)
    assert not report.ok
    assert report.describe().startswith(
        "worst case: 4 toleranced parameters, each moved 0.1 of its tolerance, at low; "
        "16 runs\n  failed: out in op ≥ 2.48 V and ≤ 2.52 V: read 2.426 V at the "
        "corner low"
    )


def test_a_worst_case_moves_a_step_of_its_tolerance() -> None:
    with pytest.raises(ValueError, match="above 0 and at most 1, not 0"):
        WorstCase(step=0)
    varied = [
        Varied("R1", "resistance", 1e3, ir.Tolerance(relative=0.1)),
        Varied(
            "R2",
            "resistance",
            1e3,
            ir.Tolerance(
                relative=0.01,
                lot=ir.Lot(name="die", relative=1.0, distribution="log_uniform"),
            ),
        ),
    ]
    assert moved(varied, 0, 0.5) == {"R1": {"resistance": pytest.approx(1050.0)}}
    assert extreme(varied, [-1.0, 1.0], {("die", "resistance"): -1.0}) == {
        "R1": {"resistance": pytest.approx(900.0)},
        "R2": {"resistance": pytest.approx(1e3 * 1.01 / 2)},
    }


GRIDDED = """
from jolty import Builder, Corner, OperatingPoint, Plan, Role, Specs, component, devices
from jolty.plan import Axis, Grid
from jolty.specs import value
from jolty.units import K, V, celsius, kohm


@component
def divider(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(5 * V), name="V1")
    heated = devices.resistor(1 * kohm, temperature_coefficient=1e-3 / K)
    upper = c.place(heated, name="R1")
    lower = c.place(devices.resistor(1 * kohm), name="R2")
    c.connect(source.positive, upper.a)
    c.connect(upper.b, lower.a, name="out")
    c.connect(lower.b, source.negative, ground)


def build():
    return divider()


def plan():
    resistances = [0.8 * kohm, 0.9 * kohm, 1 * kohm, 1.1 * kohm, 1.2 * kohm]
    return Plan(
        {"op": OperatingPoint()},
        corners={"hot": Corner(temperature=celsius(127))},
        variants=Grid([Axis("R2", "resistance", resistances)]),
    )


def specs():
    return Specs([value("op", "out") >= 2.3 * V], yield_target=0.5)
"""


def test_a_plans_variants_run_at_its_conditions_and_every_corner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "divider.py"
    script.write_text(GRIDDED, encoding="utf-8")
    outcome = run(Job(script=script, plan=script, max_unknowns=10), tmp_path)
    assert isinstance(outcome, Built), str(outcome)
    report = outcome.variants
    assert report is not None
    assert (report.runs, report.parameters, report.corners) == (
        5,
        ("R2 resistance",),
        ("hot",),
    )
    assert report.failed == 0
    (spread,) = report.spreads
    # At 127 °C, 100 K above 27 °C, R1 is 1.1 kΩ, which lowers every run's output: the
    # worst of each run's two, within the rounding of a linear divider's solution.
    lower = [800.0, 900.0, 1000.0, 1100.0, 1200.0]
    hot = [5 * each / (1100 + each) for each in lower]
    assert spread.values is not None
    assert spread.values.minimum == pytest.approx(min(hot), rel=1e-14)
    assert spread.values.maximum == pytest.approx(max(hot), rel=1e-14)
    assert spread.passing == sum(each >= 2.3 for each in hot)
    assert report.passing is not None
    assert report.passing.passing == spread.passing
    assert report.describe().startswith(
        "variants: 5 runs of R2 resistance, at hot; 0 failed\n"
    )


def test_the_runner_saves_its_batches_tracing_what_their_plots_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "divider.py"
    script.write_text(DIVIDER.replace("SAMPLES", "20"), encoding="utf-8")
    outcome = run(Job(script=script, plan=script, max_unknowns=10, plot=True), tmp_path)
    assert isinstance(outcome, Built), str(outcome)
    ((name, batch),) = files.load_batches(tmp_path / BATCHES).items()
    assert name == MONTE_CARLO
    assert batch.requirements == ("out in op ≥ 2.493 V and ≤ 2.507 V",)
    assert batch.bounds == ((2.5 - 7.5e-3, 2.5 + 7.5e-3),)
    found = batch.result
    assert found.runs == 20
    (traces,) = found.traces
    # The top component names one net, which a plot draws by default.
    assert [unknown.name for unknown in traces.unknowns] == ["out"]
    assert (traces.values[:, 0, 0] == found.measured[:, 0]).all()
    # Without plots, a batch traces nothing.
    unplotted = tmp_path / "unplotted"
    unplotted.mkdir()
    run(Job(script=script, plan=script, max_unknowns=10), unplotted)
    assert files.load_batches(unplotted / BATCHES)[MONTE_CARLO].result.traces == ()
