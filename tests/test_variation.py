# Copyright 2026 Maxence Faldor

import math
from pathlib import Path

import numpy as np
import pytest

from jolty import (
    Builder,
    Component,
    ElaborationError,
    Role,
    Simulation,
    circuit,
    component,
    devices,
    files,
    ir,
    shards,
)
from jolty.devices import ParameterValue
from jolty.plan import MonteCarlo, OperatingPoint, WorstCase
from jolty.runner import BATCHES, MONTE_CARLO, SHARD, Built, Job, run
from jolty.specs import Specs, value
from jolty.spice import Library, Sample
from jolty.units import A, Quantity, V, kohm, nF, ohm
from jolty.variation import (
    Toleranced,
    extreme,
    moved,
    sample,
    statistical,
    toleranced,
)


def _si(value: ParameterValue) -> float:
    return value.value if isinstance(value, Quantity) else value


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
    for name in ("X1", "X2"):
        placed = c.place(cell(), name=name)
        c.connect(source.positive, placed.a)
        c.connect(placed.ground, ground)
    c.connect(source.negative, ground)


def test_each_placement_of_a_component_varies_apart() -> None:
    varied = toleranced(circuit(pair()))
    assert [(parameter.instance, parameter.parameter) for parameter in varied] == [
        ("X1/R1", "resistance"),
        ("X2/R1", "resistance"),
    ]
    drawn = sample(varied, 3, 0)
    assert drawn["X1/R1"]["resistance"] != drawn["X2/R1"]["resistance"]
    assert drawn == sample(varied, 3, 0)
    assert drawn != sample(varied, 3, 1)
    point = Simulation(pair(), overrides=drawn).run(OperatingPoint())
    first = drawn["X1/R1"]["resistance"]
    second = drawn["X2/R1"]["resistance"]
    assert isinstance(first, Quantity)
    assert isinstance(second, Quantity)
    expected = (1 * V / first + 1 * V / second).value
    # Parallel resistor current 1/R1 + 1/R2; default pytest.approx covers LU rounding.
    assert -point.current("V1") == pytest.approx(expected)


def test_values_spread_within_their_tolerance() -> None:
    device = devices.resistor(1 * kohm).toleranced("resistance", 0.03, "normal")
    varied = (
        *toleranced(circuit(pair())),
        *toleranced(_alone(device)),
    )
    drawn = [sample(varied, 0, index) for index in range(4000)]
    uniform = np.array([_si(d["X1/R1"]["resistance"]) for d in drawn]) / 1e3 - 1
    normal = np.array([_si(d["R1"]["resistance"]) for d in drawn]) / 1e3 - 1
    # Uniform support is exactly [-0.05, 0.05]; at N=4000 draws, relative std SE is
    # sqrt(0.8/(4*4000)) ~= 0.71% (~7 SE) uniform and 1/sqrt(2*4000) ~= 1.12% (~4.5 SE)
    # for normal, both bounded by rel=0.05.
    assert np.abs(uniform).max() <= 0.05
    assert uniform.std() == pytest.approx(0.05 / math.sqrt(3), rel=0.05)
    assert normal.std() == pytest.approx(0.01, rel=0.05)


def test_a_tolerance_needs_a_number_below_one() -> None:
    with pytest.raises(ValueError, match="gives inductance no number"):
        devices.resistor(1 * kohm).toleranced("inductance", 0.1)
    with pytest.raises(ValueError, match="tolerance is below 1"):
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
        Simulation(pair(), overrides={"X9/R1": {"resistance": 1 * ohm}})


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
    assert report.yield_ is not None
    (spread,) = report.spreads
    assert spread.passing == report.yield_.passing
    assert spread.values is not None
    # Monte Carlo mean over 2000 samples: SE ~= 1.86e-4 V, so abs=1e-3 is ~5.4 SE.
    assert spread.values.mean == pytest.approx(2.5, abs=1e-3)

    # The same divider, sampled by numpy alone with another seed.
    generator = np.random.default_rng(12345)
    upper = 1e3 * (1 + generator.uniform(-0.01, 0.01, SAMPLES))
    lower = 1e3 * (1 + generator.normal(0, 0.01 / 3, SAMPLES))
    out = 5 * lower / (upper + lower)
    independent = float(np.mean(np.abs(out - 2.5) <= 7.5e-3))
    estimate = report.yield_.estimate
    error = math.sqrt(2 * independent * (1 - independent) / SAMPLES)
    assert abs(estimate - independent) <= 4 * error
    assert report.yield_.lower <= independent <= report.yield_.upper
    assert report.yield_.met is (estimate >= 0.9)


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
    assert (varied.instance, varied.parameter, varied.nominal) == (
        "V1",
        "waveform.dc",
        5 * V,
    )
    drawn = sample((varied,), 0, 0)
    level = drawn["V1"]["waveform.dc"]
    assert isinstance(level, Quantity)
    assert 4.75 * V <= level <= 5.25 * V
    point = Simulation(built, overrides=drawn).run(OperatingPoint())
    # Equal-resistor divider ratio level / 2; default pytest.approx covers rounding.
    assert point.voltage("out") == pytest.approx(level.value / 2)
    with pytest.raises(ValueError, match=r"gives waveform\.sine no number"):
        devices.voltage_source(5 * V).toleranced("waveform.sine", 0.05)
    with pytest.raises(ElaborationError, match=r"`waveform\.sine\.offset` has no"):
        Simulation(built, overrides={"V1": {"waveform.sine.offset": 1 * V}})


def test_the_parts_of_a_lot_share_its_factor() -> None:
    die = ir.Lot(name="die", relative=0.2)

    @component
    def integrated(c: Builder, lot: ir.Lot = die) -> None:
        ground = c.port("ground", Role.GROUND)
        for name in ("R1", "R2"):
            matched = devices.resistor(1 * kohm).toleranced(
                "resistance", 0.001, lot=lot
            )
            placed = c.place(matched, name=name)
            c.connect(placed.a, placed.b, ground)
        held = devices.capacitor(1 * nF).toleranced("capacitance", 0.001, lot=lot)
        capacitor = c.place(held, name="C1")
        c.connect(capacitor.a, capacitor.b, ground)

    varied = toleranced(circuit(integrated()))
    drawn = [sample(varied, 0, index) for index in range(2000)]
    first, second = (
        np.array([_si(d[name]["resistance"]) for d in drawn]) / 1e3 - 1
        for name in ("R1", "R2")
    )
    capacitance = np.array([_si(d["C1"]["capacitance"]) for d in drawn]) / 1e-9 - 1
    # Shared lot cancels: max independent +-0.001 ratio error is 1.001/0.999 - 1
    # ~= 0.002002 <= 0.0021; at N=2000, uniform std SE is ~1% (rel=0.05 is 5 SE) and
    # independent correlation SE is 1/sqrt(2000) ~= 0.0224 (< 0.1 is ~4.5 SE).
    assert np.abs((1 + first) / (1 + second) - 1).max() <= 0.0021
    assert first.std() == pytest.approx(0.2 / math.sqrt(3), rel=0.05)
    assert abs(np.corrcoef(first, capacitance)[0, 1]) < 0.1
    wider = ir.Lot(name="die", relative=0.3)

    @component
    def mixed(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        for name, lot in (("X1", die), ("X2", wider)):
            placed = c.place(integrated(lot), name=name)
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
        for name in ("Q1", "Q2"):
            placed = c.place(transistor, name=name)
            c.connect(placed.collector, placed.base, placed.emitter, ground)

    varied = toleranced(circuit(pair()))
    nominal = _si(varied[0].nominal)
    drawn = [sample(varied, 0, index) for index in range(2000)]
    first, second = (
        np.log(np.array([_si(d[name]["saturation_current"]) for d in drawn]) / nominal)
        for name in ("Q1", "Q2")
    )
    # Support is exact log(36) lot + 1 mV/Vt mismatch (+ 1e-12 f64 log rounding);
    # at N=2000 draws, uniform std SE is ~1%, so rel=0.05 is ~5 standard errors.
    assert np.abs(first).max() <= math.log(36) + 1e-3 / 0.02585 + 1e-12
    assert first.std() == pytest.approx(math.log(36) / math.sqrt(3), rel=0.05)
    # Shared lot cancels: base-emitter voltages within 2 mV (+ 1e-12 f64 log rounding).
    assert np.abs(first - second).max() * 0.02585 <= 2e-3 + 1e-12
    with pytest.raises(ValueError, match="only a logarithmic tolerance may exceed 1"):
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
    assert report.yield_ is not None
    assert report.yield_.passing == 0
    # Every sample reads worst at the low supply, on the parts the seed draws.
    (spread,) = report.spreads
    assert spread.values is not None
    varied = [
        Toleranced(path, "resistance", 1 * kohm, ir.Tolerance(relative=0.01))
        for path in ("R1", "R2")
    ]
    drawn = [sample(varied, 2, index) for index in range(50)]
    low = [
        4.9
        * _si(d["R2"]["resistance"])
        / (_si(d["R1"]["resistance"]) + _si(d["R2"]["resistance"]))
        for d in drawn
    ]
    # Linear divider extrema (50 samples); rel=1e-6 covers runner JSON & LU rounding.
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
    # R1 up and R2 down by their whole tolerance at 4.9 V; linear DC solve rounding.
    assert divided.value == pytest.approx(4.9 * 0.99 / 2, rel=1e-9)
    assert (divided.status, divided.corner) == ("failed", "low")
    # Shared die spread cancels in ratio; linear DC solve rounding (rel=1e-9).
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
        Toleranced("R1", "resistance", 1 * kohm, ir.Tolerance(relative=0.1)),
        Toleranced(
            "R2",
            "resistance",
            1 * kohm,
            ir.Tolerance(
                relative=0.01,
                lot=ir.Lot(name="die", relative=1.0, distribution="log_uniform"),
            ),
        ),
    ]
    # Shifted tolerance value; default pytest.approx covers float rounding.
    shifted = moved(varied, 0, 0.5)["R1"]["resistance"]
    assert isinstance(shifted, Quantity)
    assert shifted.dimension == ohm.dimension
    assert shifted.value == pytest.approx(1050.0)
    extremes = extreme(varied, [-1.0, 1.0], {("die", "resistance"): -1.0})
    r1 = extremes["R1"]["resistance"]
    r2 = extremes["R2"]["resistance"]
    # Extreme tolerance values; default pytest.approx covers float rounding.
    assert isinstance(r1, Quantity)
    assert r1.dimension == ohm.dimension
    assert r1.value == pytest.approx(900.0)
    assert isinstance(r2, Quantity)
    assert r2.dimension == ohm.dimension
    assert r2.value == pytest.approx(1e3 * 1.01 / 2)


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
    # Linear divider extrema at 127 C (R1 = 1.1 kΩ); LU rounding (~45 eps, rel=1e-14).
    assert spread.values.minimum == pytest.approx(min(hot), rel=1e-14)
    assert spread.values.maximum == pytest.approx(max(hot), rel=1e-14)
    assert spread.passing == sum(each >= 2.3 for each in hot)
    assert report.yield_ is not None
    assert report.yield_.passing == spread.passing
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


STATISTICAL_SPICE = """
.param sw = 1
.param proc_shift = {sw * agauss(0, 0.06, 3)}

.subckt leaf_det a b rnom=500
.param rval = {rnom * (1 + sw * agauss(0, 0.03, 3))}
Rleaf a b {rval}
.ends leaf_det

.subckt leaf_hoist a b rbase=400
.param rval = {rbase * (1 + sw * aunif(0, 0.02))}
Rhoist a b {rval}
.ends leaf_hoist

.subckt stage a b
.param gval = {gauss(1000, sw * 0.04, 3)}
.param uval = {unif(1, sw * 0.02)}
.param lval = {limit(1, sw * 0.01)}
.param folded = {agauss(1, 0, 3)}
.param rbody = {gval * (1 + proc_shift) * uval * lval * folded}
.param rchild = {400 * (1 + 0.5 * proc_shift + 0.02 * sw * agauss(0, 1, 1))}
R1 a n1 {rbody}
Rbeh n1 n2 r = {rbody * (1 + 0.02 * abs(v(n1, n2)))}
Xdet n2 n3 leaf_det rnom=500
Xhoist n3 b leaf_hoist rbase={rchild}
.ends stage
"""


def _statistical_bench(
    library: Library, drawn_sample: Sample | None = None
) -> Component:
    """Builds a bench mixing SPICE statistical stages and a toleranced load."""
    builder = Builder("statistical_bench")
    ground = builder.port("ground", Role.GROUND)
    supply = builder.place(devices.voltage_source(2 * V), name="V1")
    first_stage = (
        library.component("stage")
        if drawn_sample is None
        else library.drawn("stage", drawn_sample, "X1")
    )
    second_stage = (
        library.component("stage")
        if drawn_sample is None
        else library.drawn("stage", drawn_sample, "X2")
    )
    first = builder.place(first_stage, name="X1")
    second = builder.place(second_stage, name="X2")
    load_one = builder.place(devices.resistor(1 * kohm, tolerance=0.05), name="RL1")
    load_two = builder.place(devices.resistor(1 * kohm), name="RL2")
    builder.connect(supply.positive, first.a, second.a, name="vdd")
    builder.connect(first.b, load_one.a, name="out1")
    builder.connect(second.b, load_two.a, name="out2")
    builder.connect(load_one.b, load_two.b, supply.negative, ground)
    return builder.finish()


def test_native_spice_statistical_monte_carlo_matches_drawn() -> None:
    """Verifies native statistical Monte Carlo matches `Library.drawn` sample by sample.

    Exercises global shared process draws, local per-placement mismatch across all five
    SPICE statistical functions (`agauss`, `gauss`, `aunif`, `unif`, `limit`), zero-
    spread folding, nested cached children, hoisted stochastic-argument children,
    behavioral resistors (`p(...)`), and a discrete toleranced load in one batch.
    Operating-point voltages agree within `rel=1e-10` (covering Newton reltol `1e-6`
    residual convergence and floating-point expression order differences).
    """
    active = Library()
    active.read(STATISTICAL_SPICE, "statistical.sp")
    nominal_bench = _statistical_bench(active)
    built = circuit(nominal_bench)
    placements = statistical(built)
    assert [path for path, _ in placements] == ["X1", "X1/Xdet", "X2", "X2/Xdet"]

    seed, count = 23, 12
    batch = Simulation(built).batch(
        {"op": OperatingPoint()},
        MonteCarlo(count, seed=seed),
        traces=["out1", "out2"],
    )
    assert not batch.failed.any()
    assert "RL1 resistance" in batch.parameters
    assert "X1/R1 resistance" in batch.parameters
    assert "X1/Rbeh parameters.rbody" in batch.parameters
    assert "X1/Xdet/Rleaf resistance" in batch.parameters
    assert "X1/Xhoist/Rhoist resistance" in batch.parameters
    assert "X2/R1 resistance" in batch.parameters

    toleranced_loads = toleranced(built)
    out1 = batch.traces[0].values_of("out1")[:, 0]
    out2 = batch.traces[0].values_of("out2")[:, 0]
    for index in range(count):
        drawn_bench = _statistical_bench(active, Sample(seed, index))
        drawn_load = sample(toleranced_loads, seed, index)
        point = Simulation(drawn_bench, overrides=drawn_load).run(OperatingPoint())
        assert out1[index] == pytest.approx(point.voltage("out1"), rel=1e-10)
        assert out2[index] == pytest.approx(point.voltage("out2"), rel=1e-10)
    assert not np.array_equal(out1, out2)

    disabled = Library()
    disabled.read(
        STATISTICAL_SPICE.replace(".param sw = 1", ".param sw = 0"), "disabled.sp"
    )
    inactive_built = circuit(_statistical_bench(disabled))
    assert statistical(inactive_built) == ()
    assert all(
        definition.statistical is None
        for definition in inactive_built.components.values()
    )


STATISTICAL_SCRIPT = """
from jolty import Builder, MonteCarlo, OperatingPoint, Plan, Role, Specs, component
from jolty import devices
from jolty.specs import value
from jolty.spice import Library
from jolty.units import V, kohm

SPICE = \"\"\"
.param proc = {agauss(0, 0.03, 3)}
.subckt cell a b
.param rv = {1000 * (1 + proc + agauss(0, 0.03, 3))}
R1 a b {rv}
.ends cell
\"\"\"


@component
def divider(c: Builder) -> None:
    library = Library()
    library.read(SPICE, "stat_divider.sp")
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(2 * V), name="V1")
    top = c.place(library.component("cell"), name="X1")
    bottom = c.place(library.component("cell"), name="X2")
    load = c.place(devices.resistor(2 * kohm, tolerance=0.05), name="RL")
    c.connect(source.positive, top.a)
    c.connect(top.b, bottom.a, load.a, name="out")
    c.connect(bottom.b, load.b, source.negative, ground)


def build():
    return divider()


def plan():
    return Plan({"op": OperatingPoint()}, monte_carlo=MonteCarlo(24, seed=7))


def specs():
    return Specs([value("op", "out").within(0.8 * V, 0.2 * V)], yield_target=0.5)
"""


def test_statistical_monte_carlo_round_trips_runner_and_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies runner batch serialization and shard merging with statistics."""
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "stat_divider.py"
    script.write_text(STATISTICAL_SCRIPT, encoding="utf-8")
    whole_dir = tmp_path / "whole"
    whole_dir.mkdir()
    outcome = run(
        Job(script=script, plan=script, max_unknowns=20, plot=True), whole_dir
    )
    assert isinstance(outcome, Built), str(outcome)
    mc_report = outcome.monte_carlo
    assert mc_report is not None
    assert (mc_report.samples, mc_report.toleranced, mc_report.mismatched) == (24, 1, 2)
    assert mc_report.failed == 0

    loaded = files.load_batches(whole_dir / BATCHES)[MONTE_CARLO].result
    assert loaded.runs == 24
    assert set(loaded.parameters) == {
        "temperature",
        "RL resistance",
        "X1/R1 resistance",
        "X2/R1 resistance",
    }

    shard_paths: list[Path] = []
    for start, stop in ((0, 12), (12, 24)):
        work = tmp_path / f"shard-{start}-{stop}"
        work.mkdir()
        shard_outcome = run(
            Job(
                script=script,
                plan=script,
                max_unknowns=20,
                samples=(start, stop),
                plot=True,
            ),
            work,
        )
        assert isinstance(shard_outcome, Built), str(shard_outcome)
        shard_file = work / SHARD
        loaded_shard = files.load_shard(shard_file)
        assert (loaded_shard.toleranced, loaded_shard.mismatched) == (1, 2)
        shard_paths.append(shard_file)

    merged_shard, _ = shards.merge(shard_paths)
    merged_report, merged_batch = shards.monte_carlo(merged_shard)
    assert merged_report == mc_report
    assert merged_batch is not None
    assert np.array_equal(merged_batch.result.values, loaded.values)
    assert np.array_equal(merged_batch.result.measured, loaded.measured)
