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
    ir,
)
from jolty.plan import MonteCarlo, OperatingPoint
from jolty.runner import Built, Job, run
from jolty.specs import Specs, value
from jolty.units import V, kohm, nF
from jolty.variation import Varied, sample, toleranced


@component
def cell(c: Builder) -> None:
    a = c.port("a")
    ground = c.port("ground", Role.GROUND)
    resistor = c.place(devices.resistor(1 * kohm, tolerance=0.05), id="R1")
    c.connect(a, resistor.a)
    c.connect(resistor.b, ground)


@component
def pair(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1 * V), id="V1")
    for id in ("X1", "X2"):
        placed = c.place(cell(), id=id)
        c.connect(source.positive, placed.a)
        c.connect(placed.ground, ground)
    c.connect(source.negative, ground)


def test_each_placement_of_a_component_varies_apart() -> None:
    varied = toleranced(circuit(pair()))
    assert [(parameter.path, parameter.parameter) for parameter in varied] == [
        ("X1/R1", "resistance"),
        ("X2/R1", "resistance"),
    ]
    drawn = sample(varied, np.random.default_rng(3))
    assert drawn["X1/R1"]["resistance"] != drawn["X2/R1"]["resistance"]
    assert drawn == sample(varied, np.random.default_rng(3))
    point = Simulation(pair(), overrides=drawn).run(OperatingPoint())
    expected = 1 / drawn["X1/R1"]["resistance"] + 1 / drawn["X2/R1"]["resistance"]
    assert -point.current("V1") == pytest.approx(expected)


def test_values_spread_within_their_tolerance() -> None:
    device = devices.resistor(1 * kohm).toleranced("resistance", 0.03, "normal")
    varied = (
        *toleranced(circuit(pair())),
        *toleranced(_alone(device)),
    )
    generator = np.random.default_rng(0)
    drawn = [sample(varied, generator) for _ in range(4000)]
    uniform = np.array([d["X1/R1"]["resistance"] for d in drawn]) / 1e3 - 1
    normal = np.array([d["R1"]["resistance"] for d in drawn]) / 1e3 - 1
    assert np.abs(uniform).max() <= 0.05
    assert uniform.std() == pytest.approx(0.05 / math.sqrt(3), rel=0.05)
    assert normal.std() == pytest.approx(0.01, rel=0.05)


def test_a_tolerance_needs_a_number_below_one() -> None:
    with pytest.raises(ValueError, match="gives inductance no number"):
        devices.resistor(1 * kohm).toleranced("inductance", 0.1)
    with pytest.raises(ValueError, match="less than 1"):
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
    source = c.place(devices.voltage_source(5 * V), id="V1")
    upper = c.place(devices.resistor(1 * kohm, tolerance=0.01), id="R1")
    lower = c.place(
        devices.resistor(1 * kohm).toleranced("resistance", 0.01, "normal"), id="R2"
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
    outcome = run(Job(script=script, plan=script, listed=10), tmp_path)
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
    assert report.passing.low <= independent <= report.passing.high
    assert report.passing.met is (estimate >= 0.9)


def test_a_yield_target_lies_above_0_and_at_most_1() -> None:
    assert Specs([value("op", "out") >= 0 * V], yield_target=1).yield_target == 1
    with pytest.raises(ValueError, match="a yield target lies above 0"):
        Specs([value("op", "out") >= 0 * V], yield_target=0)


def _alone(device: devices.Device) -> ir.Circuit:
    @component
    def alone(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        placed = c.place(device, id="R1")
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
        source = c.place(supply, id="V1")
        upper = c.place(devices.resistor(1 * kohm), id="R1")
        lower = c.place(devices.resistor(1 * kohm), id="R2")
        c.connect(source.positive, upper.a)
        c.connect(upper.b, lower.a, name="out")
        c.connect(lower.b, source.negative, ground)

    built = circuit(divider())
    (varied,) = toleranced(built)
    assert (varied.path, varied.parameter, varied.nominal) == ("V1", "waveform.dc", 5)
    drawn = sample((varied,), np.random.default_rng(0))
    level = drawn["V1"]["waveform.dc"]
    assert 4.75 <= level <= 5.25
    point = Simulation(built, overrides=drawn).run(OperatingPoint())
    assert point.voltage("out") == pytest.approx(level / 2)
    with pytest.raises(ValueError, match=r"gives waveform\.sine no number"):
        devices.voltage_source(5 * V).toleranced("waveform.sine", 0.05)
    with pytest.raises(ElaborationError, match=r"`waveform\.sine\.offset` has no"):
        Simulation(built, overrides={"V1": {"waveform.sine.offset": 1.0}})


def test_the_parts_of_a_lot_share_its_deviation() -> None:
    die = ir.Lot(name="die", relative=0.2)

    @component
    def integrated(c: Builder, lot: ir.Lot = die) -> None:
        ground = c.port("ground", Role.GROUND)
        for id in ("R1", "R2"):
            matched = devices.resistor(1 * kohm).toleranced(
                "resistance", 0.001, lot=lot
            )
            placed = c.place(matched, id=id)
            c.connect(placed.a, placed.b, ground)
        held = devices.capacitor(1 * nF).toleranced("capacitance", 0.001, lot=lot)
        capacitor = c.place(held, id="C1")
        c.connect(capacitor.a, capacitor.b, ground)

    varied = toleranced(circuit(integrated()))
    generator = np.random.default_rng(0)
    drawn = [sample(varied, generator) for _ in range(2000)]
    first, second = (
        np.array([d[id]["resistance"] for d in drawn]) / 1e3 - 1 for id in ("R1", "R2")
    )
    capacitance = np.array([d["C1"]["capacitance"] for d in drawn]) / 1e-9 - 1
    assert np.abs(first - second).max() <= 0.002 + 1e-12
    assert first.std() == pytest.approx(0.2 / math.sqrt(3), rel=0.05)
    assert abs(np.corrcoef(first, capacitance)[0, 1]) < 0.1
    wider = ir.Lot(name="die", relative=0.3)

    @component
    def mixed(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        for id, lot in (("X1", die), ("X2", wider)):
            placed = c.place(integrated(lot), id=id)
            c.connect(placed.ground, ground)

    with pytest.raises(ElaborationError, match="spread `resistance` of lot `die`"):
        Simulation(mixed())


CORNERED = """
from jolty import Builder, Corner, MonteCarlo, OperatingPoint, Plan, Role, Specs
from jolty import component, devices
from jolty.specs import value
from jolty.units import V, kohm


@component
def divider(c: Builder, supply) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(supply), id="V1")
    upper = c.place(devices.resistor(1 * kohm, tolerance=0.01), id="R1")
    lower = c.place(devices.resistor(1 * kohm, tolerance=0.01), id="R2")
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
    outcome = run(Job(script=script, plan=script, listed=10), tmp_path)
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
    generator = np.random.default_rng(2)
    varied = [
        Varied(path, "resistance", 1e3, ir.Tolerance(relative=0.01))
        for path in ("R1", "R2")
    ]
    drawn = [sample(varied, generator) for _ in range(50)]
    low = [
        4.9 * d["R2"]["resistance"] / (d["R1"]["resistance"] + d["R2"]["resistance"])
        for d in drawn
    ]
    assert spread.values.minimum == pytest.approx(min(low), rel=1e-6)
    assert spread.values.maximum == pytest.approx(max(low), rel=1e-6)
