# Copyright 2026 Maxence Faldor
"""Tests for upfront plan/spec name resolution and multi-error Pydantic validation."""

import math
import re
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from jolty import (
    Ac,
    Builder,
    Characterized,
    Corner,
    DcSweep,
    DimensionError,
    ElaborationError,
    Mismatch,
    MonteCarlo,
    Noise,
    OperatingPoint,
    OperatingPoints,
    Options,
    PeriodicSteadyState,
    Plan,
    PlanError,
    PoleZero,
    Role,
    Sensitivity,
    Simulation,
    Specs,
    Stability,
    TransferFunction,
    Transient,
    WorstCase,
    circuit,
    component,
    devices,
    fidelity,
    ir,
    results,
    service,
    simulate,
)
from jolty.errors import close_names
from jolty.plan import (
    Analysis,
    Axis,
    Conditions,
    Explicit,
    Grid,
    Parallelism,
    Product,
    Variant,
    recorded,
)
from jolty.runner import BATCHES, RESULTS, SHARD, Failure, Job, run
from jolty.specs import (
    Measure,
    Reading,
    Requirement,
    crossing,
    frequency,
    gain,
    integrated_noise,
    noise_density,
    rise_time,
    value,
)
from jolty.units import (
    VOLTAGE,
    A,
    Hz,
    K,
    V,
    celsius,
    kHz,
    kohm,
    mA,
    mH,
    nF,
    ns,
    ohm,
    pS,
    ps,
    us,
)

ADVICE_PATTERN = re.compile(
    r"\b(use|using|used|pass|passing|passed|try|trying|instead|should|give|given|"
    r"did\s+you\s+mean)\b",
    re.IGNORECASE,
)


def _assert_fact_only(text: str) -> None:
    """Asserts that `text` uses zero advice words, zero em dashes, zero en dashes."""
    match = ADVICE_PATTERN.search(text)
    assert match is None, f"advice word {match.group(0)!r} in error:\n{text}"
    assert "\u2014" not in text, f"em dash in error:\n{text}"
    assert "\u2013" not in text, f"en dash in error:\n{text}"


def _validation_messages(error: ValidationError) -> list[str]:
    """Returns every error message in `error`, audited for fact-only wording."""
    messages = [entry["msg"] for entry in error.errors()]
    for message in messages:
        _assert_fact_only(message)
    return messages


def _expect_plan_error(
    action: Callable[[], object],
    *fragments: str,
    count: int | None = None,
    min_count: int | None = None,
) -> str:
    """Asserts `action` raises a fact-only `PlanError` matching counts and fragments."""
    with pytest.raises(PlanError) as raised:
        action()
    if count is not None:
        assert len(raised.value.problems) == count
    if min_count is not None:
        assert len(raised.value.problems) >= min_count
    text = str(raised.value)
    _assert_fact_only(text)
    for fragment in fragments:
        assert fragment in text
    return text


def _expect_validation_errors(
    action: Callable[[], object],
    *fragments: str,
    count: int | None = None,
    min_count: int | None = None,
) -> None:
    """Asserts `action` raises a fact-only `ValidationError` with expected counts."""
    with pytest.raises(ValidationError) as raised:
        action()
    if count is not None:
        assert raised.value.error_count() == count
    if min_count is not None:
        assert raised.value.error_count() >= min_count
    messages = _validation_messages(raised.value)
    for fragment in fragments:
        assert any(fragment in message for message in messages)


@component
def amplifier_stage(c: Builder) -> None:
    """Builds a two-resistor subcomponent for hierarchy and fidelity scope tests."""
    input_port = c.port("input", Role.SIGNAL)
    output_port = c.port("output", Role.SIGNAL)
    ground = c.port("ground", Role.GROUND)
    upper = c.place(devices.resistor(1 * kohm), name="R1")
    lower = c.place(devices.resistor(2 * kohm), name="R2")
    c.connect(input_port, upper.a)
    c.connect(upper.b, lower.a, output_port)
    c.connect(lower.b, ground)


@component
def bench_circuit(c: Builder) -> None:
    """Builds a divider with a floating probe, passive elements, and subcomponents."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(5 * V), name="V1")
    probe = c.place(devices.voltage_source(0 * V), name="VP")
    resistor_a = c.place(devices.resistor(1 * kohm), name="R1")
    resistor_b = c.place(devices.resistor(1 * kohm), name="R2")
    inductor = c.place(devices.inductor(1 * mH), name="L1")
    capacitor = c.place(devices.capacitor(1 * nF), name="C1")
    stage_one = c.place(amplifier_stage(), name="X1")
    stage_two = c.place(amplifier_stage(), name="X2")
    c.connect(source.positive, resistor_a.a, name="in")
    c.connect(resistor_a.b, probe.positive, name="mid")
    c.connect(probe.negative, inductor.a, name="probed")
    c.connect(inductor.b, resistor_b.a, capacitor.a, stage_one.input, name="out")
    c.connect(stage_one.output, stage_two.input, name="tap")
    c.connect(stage_two.output, name="ref")
    c.connect(
        source.negative,
        resistor_b.b,
        capacitor.b,
        stage_one.ground,
        stage_two.ground,
        ground,
    )


@component
def logic_circuit(c: Builder) -> None:
    """Builds a circuit with a logic gate and a logic source."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(1.8 * V), name="V1")
    resistor = c.place(devices.resistor(1 * kohm), name="R1")
    driver = c.place(
        devices.logic_source(0, [(1 * ns, 1)], transition=50 * ps), name="D1"
    )
    gate = c.place(
        devices.logic_gate("not", rise=20 * ps, transition=10 * ps), name="U1"
    )
    c.connect(supply.positive, resistor.a, name="vdd")
    c.connect(resistor.b, supply.negative, ground)
    c.connect(driver.output, gate.a, name="sig_in")
    c.connect(gate.y, name="sig_out")


def _build_simulation() -> Simulation:
    return Simulation(bench_circuit())


def test_close_names_matches_standard_case_insensitive_and_short_typos() -> None:
    assert close_names(["bode", "op", "tran"], "trna") == "; close names: tran"
    assert close_names(["bode", "op", "tran"], "bdoe") == "; close names: bode"
    assert close_names(["bode", "op", "tran"], "po") == "; close names: op"
    assert close_names(["V1", "I1"], "V2") == "; close names: V1"
    assert close_names(["R1", "R2", "V1"], "R9") == "; close names: R2, R1"
    assert close_names(["V1", "R1"], "v1") == "; close names: V1"
    assert close_names(["out", "in"], "OUT") == "; close names: out"
    assert close_names(["op"], "dc") == ""
    assert close_names([], "anything") == ""


def test_misspelled_analysis_in_spec_fails_before_simulating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = _build_simulation()
    calls: list[str] = []

    def forbid_run(
        self: Simulation, analysis: object, options: Options | None = None
    ) -> results.AnalysisResult:
        del self, options
        calls.append(type(analysis).__name__)
        pytest.fail("engine must not run")

    monkeypatch.setattr(Simulation, "run", forbid_run)

    plan = Plan(
        {
            "op": OperatingPoint(),
            "bode": Ac("V1", [1 * kHz, 10 * kHz]),
            "tran": Transient(stop=1 * us),
        }
    )
    custom = Measure("po", "custom check", None, lambda _: Reading(0.0, VOLTAGE))
    requirements = Specs(
        [
            value("trna", "out", at=0.5 * us) >= 0 * V,
            gain("bdoe", "out", at=1 * kHz) >= -20,
            custom >= 0 * V,
        ]
    )

    _expect_plan_error(
        lambda: simulation.run_plan(plan, specs=requirements),
        "'trna'; close names: tran",
        "'bdoe'; close names: bode",
        "'po'; close names: op",
        "defined analyses: op, bode, tran",
        count=3,
    )
    assert calls == []


def test_core_run_is_never_called_when_analysis_or_plan_fails_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "analyses.jsonl"
    monkeypatch.setenv("JOLTY_ANALYSIS_LOG", str(log_path))
    simulate._settings.cache_clear()  # noqa: SLF001 - reload analysis log path
    try:
        simulation = _build_simulation()
        core_calls: list[str] = []

        original_core = simulation._core  # noqa: SLF001 - spy on core engine boundary

        class SpyCore:
            def __getattr__(self, name: str) -> object:
                return getattr(original_core, name)

            def run(self, *_args: object, **_kwargs: object) -> object:
                core_calls.append("run")
                pytest.fail("core engine must not run")

        monkeypatch.setattr(simulation, "_core", SpyCore())

        with pytest.raises(PlanError) as raised:
            simulation.run(
                Noise("outt", [1 * kHz], reference="reff", source="R1"),
                Options(nodesets={"bad_net": 1 * V}),
            )
        assert core_calls == []
        assert not log_path.exists()
        assert len(raised.value.problems) == 4
        _assert_fact_only(str(raised.value))
    finally:
        monkeypatch.undo()
        simulate._settings.cache_clear()  # noqa: SLF001 - restore settings cache


def test_runner_fails_before_simulating_on_bad_spec_or_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "design.py"
    script.write_text(
        """
from jolty import (
    Ac,
    Builder,
    MonteCarlo,
    OperatingPoint,
    Plan,
    Role,
    Specs,
    component,
    devices,
)
from jolty.specs import gain, value
from jolty.units import V, kHz, kohm


@component
def divider(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(5 * V), name="V1")
    upper = c.place(devices.resistor(1 * kohm, tolerance=0.05), name="R1")
    lower = c.place(devices.resistor(1 * kohm, tolerance=0.05), name="R2")
    c.connect(source.positive, upper.a, name="in")
    c.connect(upper.b, lower.a, name="out")
    c.connect(lower.b, source.negative, ground)


def build():
    return divider()


def plan():
    return Plan(
        {"op": OperatingPoint(), "bode": Ac("V1", [1 * kHz])},
        monte_carlo=MonteCarlo(4),
    )


def specs():
    return Specs(
        [
            value("po", "outt") >= 1 * V,
            gain("bdoe", "out", at=1 * kHz) >= -10,
        ]
    )
""",
        encoding="utf-8",
    )

    calls: list[str] = []

    def forbid_run(
        self: Simulation, analysis: object, options: Options | None = None
    ) -> results.AnalysisResult:
        del self, options
        calls.append(type(analysis).__name__)
        pytest.fail("runner must not invoke Simulation.run")

    def forbid_batch(self: Simulation, *args: object, **kwargs: object) -> object:
        del self, args, kwargs
        calls.append("batch")
        pytest.fail("runner must not invoke Simulation.batch")

    monkeypatch.setattr(Simulation, "run", forbid_run)
    monkeypatch.setattr(Simulation, "batch", forbid_batch)

    full = run(Job(script=script, plan=script, max_unknowns=10), tmp_path)
    assert isinstance(full, Failure)
    assert full.error == "PlanError"
    assert calls == []
    assert not (tmp_path / RESULTS).exists()
    assert not (tmp_path / BATCHES).exists()
    _assert_fact_only(full.message)
    assert "'po'; close names: op" in full.message
    assert "'bdoe'; close names: bode" in full.message
    assert "design.py:" in full.message

    shard_dir = tmp_path / "shard"
    shard_dir.mkdir()
    shard = run(
        Job(script=script, plan=script, max_unknowns=10, samples=(0, 2)),
        shard_dir,
    )
    assert isinstance(shard, Failure)
    assert shard.error == "PlanError"
    assert calls == []
    assert not (shard_dir / SHARD).exists()
    _assert_fact_only(shard.message)


def test_fidelity_precheck_fails_before_characterization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "logic_plan.py"
    script.write_text(
        """
from jolty import (
    Builder,
    Characterized,
    OperatingPoint,
    Plan,
    Role,
    Specs,
    component,
    devices,
)
from jolty.specs import value
from jolty.units import V, kohm


@component
def cell(c: Builder) -> None:
    inp = c.port("a", Role.SIGNAL)
    out = c.port("y", Role.SIGNAL)
    ground = c.port("ground", Role.GROUND)
    resistor = c.place(devices.resistor(1 * kohm), name="R1")
    c.connect(inp, resistor.a)
    c.connect(resistor.b, out)
    c.connect(ground)


@component
def top(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1.8 * V), name="V1")
    stage = c.place(cell(), name="X1")
    c.connect(source.positive, stage.a, name="in")
    c.connect(stage.y, name="out")
    c.connect(source.negative, stage.ground, ground)


def build():
    return top()


def plan():
    return Plan(
        {"op": OperatingPoint()},
        fidelity={"X9": Characterized(1.8 * V), "Z*": Characterized(1.8 * V)},
        process="ss",
    )


def specs():
    return Specs([value("po", "out") >= 0 * V])
""",
        encoding="utf-8",
    )

    def forbid_characterize(*args: object, **kwargs: object) -> object:
        del args, kwargs
        pytest.fail("characterize must not run on invalid fidelity/specs")

    monkeypatch.setattr(fidelity, "characterize", forbid_characterize)
    outcome = run(Job(script=script, plan=script, max_unknowns=10), tmp_path)
    assert isinstance(outcome, Failure)
    assert outcome.error == "PlanError"
    _assert_fact_only(outcome.message)
    assert "'X9' matches no component instance; close names: X1" in outcome.message
    assert "'Z*' matches no component instance" in outcome.message
    assert "process corner ss" in outcome.message
    assert "'po'; close names: op" in outcome.message


def test_record_reports_all_missing_paths_and_patterns_at_once() -> None:
    simulation = _build_simulation()
    _expect_plan_error(
        lambda: simulation.run(OperatingPoint(record=["R9", "X3/**", "X1/R9"])),
        "'R9' to record; close names: R2, R1",
        "pattern 'X3/**' matches no instance",
        "'X1/R9' to record; close names: X1/R2, X1/R1",
        count=3,
    )


_ANALYSIS_ERROR_CASES: dict[str, tuple[Analysis, int | None, tuple[str, ...]]] = {
    "dc-missing-source": (
        DcSweep("V9", [1 * V]),
        None,
        (
            "there is no instance at `V9`; close names: VP, V1",
            "independent sources: V1, VP",
        ),
    ),
    "dc-non-source": (
        DcSweep("R1", [1 * V]),
        None,
        ("`R1` is not an independent source",),
    ),
    "dc-param-missing": (
        DcSweep("R9", [1 * kohm], parameter="resistance"),
        None,
        ("there is no instance at `R9`; close names: R2, R1",),
    ),
    "dc-param-misspelled": (
        DcSweep("R1", [1 * kohm], parameter="resistence"),
        None,
        ("`R1` writes no number `resistence` to sweep; close names: resistance",),
    ),
    "ac-bad-source": (Ac("V9", [1 * kHz]), None, ("close names: VP, V1",)),
    "stability-missing-probe": (
        Stability("VQ", [1 * kHz]),
        None,
        ("close names: VP", "probes: VP"),
    ),
    "stability-grounded-source": (
        Stability("V1", [1 * kHz]),
        None,
        ("`V1` is not a floating voltage source", "probes: VP"),
    ),
    "stability-resistor-probe": (
        Stability("R1", [1 * kHz]),
        None,
        ("`R1` is not a floating voltage source", "probes: VP"),
    ),
    "noise-current-ref": (
        Noise("V1:current", [1 * kHz], reference="ground"),
        None,
        ("an output over a node is a voltage",),
    ),
    "noise-all-bad": (
        Noise("outt", [1 * kHz], reference="reff", source="V9"),
        3,
        (
            "`outt`; close names: out",
            "`reff`; close names: ref",
            "`V9`; close names: VP, V1",
        ),
    ),
    "tf-ground-non-source": (
        TransferFunction("R1", "ground"),
        2,
        ("is ground",),
    ),
    "tf-inductor-current-ref": (
        TransferFunction("V1", "L1:current", reference="out"),
        2,
        ("has no reference", "voltage source's, not `L1:current`"),
    ),
    "tf-all-bad": (TransferFunction("V9", "outt", reference="reff"), 3, ()),
    "pz-all-bad": (PoleZero("V9", "outt", reference="reff"), 3, ()),
}


@pytest.mark.parametrize("name", list(_ANALYSIS_ERROR_CASES))
def test_analysis_upfront_plan_error_checks(name: str) -> None:
    analysis, count, fragments = _ANALYSIS_ERROR_CASES[name]
    simulation = _build_simulation()
    _expect_plan_error(lambda: simulation.run(analysis), *fragments, count=count)


def test_dc_sweep_dimension_error_and_valid_ground_references() -> None:
    simulation = _build_simulation()
    # Wrong level dimension alone still raises DimensionError.
    with pytest.raises(DimensionError, match="a sweep level takes a quantity in V"):
        simulation.run(DcSweep("V1", [1 * mA]))
    # Wrong level dimension alongside bad nodeset reports BOTH in PlanError.
    _expect_plan_error(
        lambda: simulation.run(
            DcSweep("V1", [1 * mA]), Options(nodesets={"outt": 1 * V})
        ),
        "close names: out",
        "quantity in V",
        count=2,
    )
    # Ground reference is valid for Noise and PoleZero.
    simulation.validate_analysis(Noise("out", [1 * kHz], reference="ground"))
    simulation.validate_analysis(Noise("ground", [1 * kHz], reference="out"))
    simulation.validate_analysis(PoleZero("V1", "out", reference="ground"))


def test_transient_initial_conditions_and_logic_checks() -> None:
    simulation = _build_simulation()
    _expect_plan_error(
        lambda: simulation.run(
            Transient(
                stop=1 * us,
                initial_conditions={
                    "outt": 1 * V,
                    "V1:current": 0 * V,
                    "ground": 0 * V,
                },
            )
        ),
        "`outt`, which is no net of the circuit; close names: out",
        "`V1:current`, which is no net of the circuit (current unknown)",
        "`ground`, which is ground, not an electrical net",
        count=3,
    )

    logic_simulation = Simulation(logic_circuit())
    _expect_plan_error(
        lambda: logic_simulation.validate_plan(
            Plan(
                {
                    "tran": Transient(stop=5 * ns, operating_point=False),
                    "pss": PeriodicSteadyState(period=5 * ns),
                }
            )
        ),
        "only an operating point settles",
        "without logic",
        count=2,
    )


def test_nodesets_checked_across_all_entry_points() -> None:
    simulation = _build_simulation()
    bad_options = Options(nodesets={"outt": 1 * V, "V1:current": 0 * V})
    _expect_plan_error(lambda: simulation.check_operating_point(bad_options), count=2)
    _expect_plan_error(
        lambda: simulation.batch(
            {"op": OperatingPoint()}, Explicit([Variant()]), bad_options
        ),
        count=2,
    )


def test_sensitivity_and_mismatch_checks_including_random_variables() -> None:
    simulation = _build_simulation()
    _expect_plan_error(
        lambda: simulation.run(Sensitivity(["outt", "reff"], of=Ac("V9", [1 * kHz]))),
        "`outt`; close names: out",
        "`reff`; close names: ref",
        "`V9`; close names: VP, V1",
        count=3,
    )

    first_random_variable = ir.RandomVariable(
        name="v1",
        mean=0.0,
        standard_deviation=1.0,
        derivatives={
            "R9": {"resistance": 1.0},
            "R1": {"resistence": 1.0},
        },
    )
    second_random_variable = ir.RandomVariable(
        name="v2",
        mean=0.0,
        standard_deviation=1.0,
        derivatives={"R9": {"resistance": 2.0}},  # duplicated -> deduplicated!
    )
    _expect_plan_error(
        lambda: simulation.run(
            Mismatch(
                ["outt"],
                of=DcSweep("V9", [1 * V]),
                random_variables=(first_random_variable, second_random_variable),
            )
        ),
        "`outt`; close names: out",
        "no device at `R9`; close names: R2, R1",
        "`R1` writes no number `resistence`; close names: resistance",
        "there is no instance at `V9`; close names: VP, V1",
        count=4,
    )

    # Stability defaults omitted outputs to ("loop_gain",) and rejects other names.
    default_sens = Sensitivity(of=Stability("VP", [1 * kHz]))
    assert default_sens.outputs == ("loop_gain",)
    default_mismatch = Mismatch(of=Stability("VP", [1 * kHz]))
    assert default_mismatch.outputs == ("loop_gain",)
    simulation.validate_plan(Plan({"sens": default_sens, "mis": default_mismatch}))
    _expect_plan_error(
        lambda: simulation.run(
            Sensitivity(["loop_gan"], of=Stability("V1", [1 * kHz]))
        ),
        "only the `loop_gain` output, not `loop_gan`; close names: loop_gain",
        "`V1` is not a floating voltage source; probes: VP",
        count=2,
    )
    _expect_plan_error(
        lambda: simulation.run(Mismatch(["out"], of=Stability("VQ", [1 * kHz]))),
        "only the `loop_gain` output, not `out`",
        "there is no instance at `VQ`; close names: VP; probes: VP",
        count=2,
    )

    # Noise defaults omitted outputs based on `source` and rejects invalid names.
    noise_without_source = Noise("out", [1 * kHz])
    noise_with_source = Noise("out", [1 * kHz], source="V1")
    default_sens_only = Sensitivity(of=noise_without_source)
    default_sens_both = Sensitivity(of=noise_with_source)
    default_mismatch_only = Mismatch(of=noise_without_source)
    default_mismatch_both = Mismatch(of=noise_with_source)
    assert default_sens_only.outputs == ("density",)
    assert default_sens_both.outputs == ("density", "referred")
    assert default_mismatch_only.outputs == ("density",)
    assert default_mismatch_both.outputs == ("density", "referred")
    simulation.validate_plan(
        Plan(
            {
                "s_only": default_sens_only,
                "s_both": default_sens_both,
                "s_ref": Sensitivity(["referred"], of=noise_with_source),
                "m_only": default_mismatch_only,
                "m_both": default_mismatch_both,
            }
        )
    )
    _expect_plan_error(
        lambda: simulation.run(Sensitivity(["referred"], of=Noise("outt", [1 * kHz]))),
        "output `referred` requires the noise analysis to name a `source`",
        "`outt`; close names: out",
        count=2,
    )
    _expect_plan_error(
        lambda: simulation.run(Sensitivity(["densiti"], of=Noise("out", [1 * kHz]))),
        "only the `density` output, not `densiti`; close names: density",
        count=1,
    )
    _expect_plan_error(
        lambda: simulation.run(
            Mismatch(["refered"], of=Noise("out", [1 * kHz], source="V9"))
        ),
        "exposes outputs `density` and `referred`, not `refered`; "
        "close names: referred",
        "there is no instance at `V9`; close names: VP, V1",
        count=2,
    )


def test_variants_and_traces_checked_upfront_with_close_names() -> None:
    simulation = _build_simulation()
    variants = Product(
        Grid(
            [
                Axis("R9", "resistance", [1 * kohm, 2 * kohm]),
                Axis("R1", "resistence", [1 * kohm]),
            ]
        ),
        Explicit(
            [
                Variant({"X9/R1": {"resistance": 1 * kohm}}),
                Variant({"C9": {}}),
            ]
        ),
    )

    _expect_plan_error(
        lambda: simulation.validate_plan(
            Plan({"op": OperatingPoint()}, variants=variants),
            traces=["outt", "tapp"],
        ),
        "no device at `R9`; close names: R2, R1",
        "`R1` writes no number `resistence`; close names: resistance",
        "no device at `X9/R1`; close names: X2/R1, X1/R1",
        "no device at `C9`; close names: C1",
        "trace `outt` names no unknown or logic net",
        "close names: out",
        "trace `tapp` names no unknown or logic net",
        "close names: tap",
        count=6,
    )

    # batch / varied still raise ElaborationError for variants, with close names!
    with pytest.raises(ElaborationError) as elaboration_error:
        simulation.varied(variants)
    assert len(elaboration_error.value.problems) == 4
    assert "close names: R2, R1" in str(elaboration_error.value)
    assert "close names: resistance" in str(elaboration_error.value)
    _assert_fact_only(str(elaboration_error.value))

    # batch collects ALL unknown trace names into ONE KeyError with close names!
    with pytest.raises(KeyError) as key_error:
        simulation.batch(
            {"op": OperatingPoint()},
            Explicit([Variant()]),
            traces=["outt", "reff"],
        )
    message = str(key_error.value.args[0])
    assert "outt" in message
    assert "close names: out" in message
    assert "reff" in message
    assert "close names: ref" in message
    _assert_fact_only(message)

    # Logic nets are valid traces in validate_plan
    logic_simulation = Simulation(logic_circuit())
    logic_simulation.validate_plan(
        Plan({"op": OperatingPoint()}), traces=["vdd", "sig_in", "sig_out"]
    )


def test_fidelity_scopes_report_all_unmatched_scopes_with_close_names() -> None:
    built = circuit(bench_circuit())
    _expect_plan_error(
        lambda: fidelity.scopes(
            built,
            {"X9": Characterized(5 * V), "Z*": Characterized(5 * V)},
        ),
        "'X9' matches no component instance; close names: X2, X1",
        "'Z*' matches no component instance",
        count=2,
    )


def test_specs_upfront_checks_all_measure_signals_kinds_and_bounds() -> None:
    simulation = _build_simulation()
    plan = Plan(
        {
            "op": OperatingPoint(),
            "bode": Ac("V1", [1 * kHz, 10 * kHz]),
            "step": Transient(stop=5 * us),
            "tf": TransferFunction("V1", "out"),
            "noise": Noise("out", [1 * kHz, 10 * kHz]),
        }
    )
    requirements = Specs(
        [
            # Multiple bad signals + wrong at in ONE gain measure!
            gain("bode", "outt", over="inn", reference="reff", at=1 * V) >= -3,
            # OperatingPoint with BOTH bad signal AND extraneous `at`
            value("op", "outt", at=1 * us) >= 0 * V,
            # TransferFunction rejected cleanly (no KeyError in _KINDS!)
            value("tf", "out") >= 0 * V,
            # Missing `at` on transient + wrong signal
            value("step", "outt") >= 0 * V,
            # Wrong analysis kind
            rise_time("op", "out") <= 1 * us,
            # Multiple wrong bound dimensions + margin on ONE requirement!
            Requirement(
                value("op", "out"), minimum=1 * mA, maximum=2 * kHz, margin=1 * us
            ),
            # Unreferred noise measure + wrong `at` dimension
            noise_density("noise", at=1 * V, referred=True) <= 1 * V,
            # Integrated noise: wrong start AND wrong stop + unreferred
            integrated_noise("noise", start=1 * V, stop=2 * mA, referred=True) <= 1 * V,
            # Crossing / frequency: wrong threshold dimension + wrong signal
            crossing("step", "out", 1 * mA) <= 1 * us,
            frequency("step", "out", threshold=1 * mA) >= 1 * Hz,
        ]
    )

    _expect_plan_error(
        lambda: simulation.validate_plan(plan, specs=requirements),
        "'outt'; close names: out",
        "'inn'; close names: in",
        "'reff'; close names: ref",
        "a transfer function",
        "an operating point holds one value, not one at 1 µs",
        "at is unset",
        "the noise analysis names no source to refer its noise to",
        "the minimum, 1 mA",
        "the maximum, 2 kHz",
        "the margin, 1 µs",
        min_count=16,
    )


def test_kitchen_sink_plan_reports_every_error_at_once() -> None:
    simulation = _build_simulation()
    plan = Plan(
        {
            "op": OperatingPoint(record=["R9", "X9/**"]),
            "sweep": DcSweep("V9", [1 * V]),
            "psweep": DcSweep("R1", [1 * kohm], parameter="resistence"),
            "bode": Ac("R1", [1 * kHz]),
            "step": Transient(stop=1 * us, initial_conditions={"outt": 1 * V}),
            "stab": Stability("V1", [1 * kHz]),
            "noise": Noise("outt", [1 * kHz], source="V9"),
            "tf": TransferFunction("V9", "ground"),
            "pz": PoleZero("V9", "outt"),
            "sens": Sensitivity(["outt"]),
            "mis": Mismatch(["reff"]),
        },
        options=Options(nodesets={"bad_node": 0 * V}),
        temperature=celsius(85),
        corners={
            "bad_corner": Corner(process="ff"),
            "rebuilt": Corner(process="ss", x=1),
        },
        fidelity={"X9": Characterized(5 * V)},
        process="ss",
        record=["C9"],
        variants=Grid([Axis("R9", "resistance", [1 * kohm])]),
    )
    requirements = Specs([value("bdoe", "out") >= 0 * V])

    text = _expect_plan_error(
        lambda: simulation.validate_plan(
            plan,
            specs=requirements,
            traces=["outt"],
            check_temperature=True,
        ),
        "corner `bad_corner`",
        "is not the simulation's",
        "process corner ss",
        "bad_node",
        "'X9' matches no component instance; close names: X2, X1",
        "'C9' to record; close names: C1",
        *(
            f"analysis `{name}`:"
            for name in (
                "op",
                "sweep",
                "psweep",
                "bode",
                "step",
                "stab",
                "noise",
                "tf",
                "pz",
                "sens",
                "mis",
            )
        ),
        "no device at `R9`; close names: R2, R1",
        "'bdoe'; close names: bode",
        "trace `outt`",
        min_count=20,
    )
    # Corner with arguments (`rebuilt`) is NOT rejected upfront!
    assert "corner `rebuilt`" not in text


def test_plot_missing_analyses_includes_close_names(tmp_path: Path) -> None:
    simulation = _build_simulation()
    loaded = {
        "op": simulation.run(OperatingPoint()),
        "bode": simulation.run(Ac("V1", [1 * kHz, 10 * kHz])),
    }
    _expect_plan_error(
        lambda: service._plot(tmp_path / "out.npz", loaded, ["bdoe", "po"], []),  # noqa: SLF001 - direct unit test
        "'bdoe'; close names: bode",
        "'po'; close names: op",
        count=2,
    )


_BAD_RANDOM_VARIABLE = ir.RandomVariable(
    name="bad",
    mean=0.0,
    standard_deviation=1.0,
    derivatives={"R1": {"resistance": math.nan}},
)
_OUT_MEASURE = value("op", "out")


_VALIDATION_CASES: dict[str, tuple[Callable[[], object], int]] = {
    "options": (
        lambda: Options(
            relative_tolerance=-1e-3,
            voltage_tolerance=0 * V,
            current_tolerance=math.nan * A,
            charge_tolerance=True,  # ty: ignore[invalid-argument-type] - wrong type
            gmin=-1 * pS,
            max_dc_iterations=0,
            max_transient_iterations=False,
            truncation_tolerance=math.inf,
            error_control="flux",  # ty: ignore[invalid-argument-type] - wrong literal
            max_memory=-1.0,
            profile="yes",  # ty: ignore[invalid-argument-type] - wrong type
            nodesets={"a": "bad", 1: 2 * V, "b": math.nan * V},  # ty: ignore[invalid-argument-type] - wrong types
        ),
        14,
    ),
    "parallelism": (
        lambda: Parallelism(
            min_solves_per_task=0,
            min_instances_per_task=-1,
            min_unknowns_per_task=True,
            max_unknowns_per_interior=0,
        ),
        4,
    ),
    # JSON / dict validation via TypeAdapter reports nested Parallelism + extra fields!
    "options-json": (
        lambda: TypeAdapter(Options).validate_json(
            '{"relative_tolerance": -1, "gmin": -2, '
            '"parallelism": {"min_solves_per_task": 0, "unknown_inner": 5}, '
            '"unknown_outer": 9}'
        ),
        5,
    ),
    "transient": (
        lambda: Transient(
            stop=-1 * us,
            max_step=-2 * us,
            method="rk4",  # ty: ignore[invalid-argument-type] - wrong literal
            operating_point="no",  # ty: ignore[invalid-argument-type] - wrong type
            breakpoints=[0 * us, math.nan * us],
            record="R1",
            regions=1,  # ty: ignore[invalid-argument-type] - wrong type
        ),
        8,
    ),
    "pss": (
        lambda: PeriodicSteadyState(
            period=-1 * us,
            max_step=-1 * us,
            method="rk4",  # ty: ignore[invalid-argument-type] - wrong literal
        ),
        3,
    ),
    "pss-breakpoints": (lambda: PeriodicSteadyState(breakpoints=[1 * us]), 1),
    "dc-sweep": (
        lambda: DcSweep(source="", values=[math.nan * V, math.inf * V], parameter=""),
        4,
    ),
    "dc-sweep-source-float": (lambda: DcSweep("V1", [1.0, 2.0]), 1),
    "ac": (lambda: Ac(source="", frequencies=[-1 * Hz, math.nan * Hz]), 3),
    "noise": (lambda: Noise(output="", frequencies=[0 * Hz, -1 * Hz], source=""), 4),
    "stability": (lambda: Stability(probe="", frequencies=[-1 * Hz]), 2),
    "operating-points": (
        lambda: OperatingPoints(
            max_solutions=0,
            record="X1",
            regions=1,  # ty: ignore[invalid-argument-type] - wrong type
        ),
        3,
    ),
    "transfer-function": (
        lambda: TransferFunction(source="", output="out", reference="out"),
        2,
    ),
    "pole-zero": (lambda: PoleZero(source="", output="out", reference="out"), 2),
    "sensitivity": (lambda: Sensitivity(outputs="out", of=OperatingPoint()), 2),  # ty: ignore[invalid-argument-type] - wrong types
    "mismatch": (
        lambda: Mismatch(
            outputs=[1],  # ty: ignore[invalid-argument-type] - wrong types
            of=OperatingPoint(),  # ty: ignore[invalid-argument-type] - wrong types
            random_variables=[_BAD_RANDOM_VARIABLE, "nope"],  # ty: ignore[invalid-argument-type] - wrong types
        ),
        4,
    ),
    "monte-carlo": (lambda: MonteCarlo(samples=0, seed=-1), 2),
    "variant": (
        lambda: Variant(
            overrides={"R1": {"resistance": math.nan * ohm, "b": "x"}},  # ty: ignore[invalid-argument-type] - wrong types
            temperature=math.nan * celsius(0),
        ),
        3,
    ),
    "axis": (lambda: Axis("R1", None, [math.nan * V, "bad"]), 3),  # ty: ignore[invalid-argument-type] - wrong types
    "grid": (lambda: Grid(axes=["bad", 1], warm="no"), 3),  # ty: ignore[invalid-argument-type] - wrong types
    "product": (lambda: Product(first="a", second="b"), 2),  # ty: ignore[invalid-argument-type] - wrong types
    "worst-case": (lambda: WorstCase(step=2.0), 1),
    "characterized": (
        lambda: Characterized(
            supply=-1 * V,
            slews=(2.0, 1.0),
            fanouts=(-1.0, 1.0),
            spot_checks=-1,
            tolerance=-0.1,
            seed=-1,
        ),
        6,
    ),
    "conditions": (
        lambda: Conditions(temperature=math.nan * K, process="", mismatch=-1),
        3,
    ),
    "corner": (lambda: Corner(temperature=math.nan * celsius(27), process=""), 2),
    "recorded": (lambda: recorded("X1"), 1),
    "specs-bad-items": (lambda: Specs([_OUT_MEASURE, "bad"], yield_target=0.0), 3),  # ty: ignore[invalid-argument-type] - wrong types
}


@pytest.mark.parametrize("name", list(_VALIDATION_CASES))
def test_models_report_all_validation_errors_at_once(name: str) -> None:
    build, count = _VALIDATION_CASES[name]
    _expect_validation_errors(build, count=count)


def test_options_error_control_accepts_charge_and_unknown_and_rejects_others() -> None:
    assert Options().to_json() == {}
    assert Options(error_control="charge").to_json() == {"error_control": "charge"}
    assert Options(error_control="unknown").to_json() == {"error_control": "unknown"}
    parsed = TypeAdapter(Options).validate_json('{"error_control": "unknown"}')
    assert parsed == Options(error_control="unknown")
    for invalid in ("flux", 1, True):
        _expect_validation_errors(
            lambda value=invalid: Options(error_control=value),  # ty: ignore[invalid-argument-type] - wrong value
            "error_control is 'charge' or 'unknown'",
            count=1,
        )


def test_plan_reports_all_validation_errors_at_once() -> None:
    _expect_validation_errors(
        lambda: Plan(
            analyses={"bad name": "not_analysis", "also bad": 123},  # ty: ignore[invalid-argument-type] - wrong types
            options="bad",  # ty: ignore[invalid-argument-type] - wrong type
            corners={"bad corner": "not_corner"},  # ty: ignore[invalid-argument-type] - wrong type
            monte_carlo="bad",  # ty: ignore[invalid-argument-type] - wrong type
            worst_case="bad",  # ty: ignore[invalid-argument-type] - wrong type
            fidelity={"": "not_char"},  # ty: ignore[invalid-argument-type] - wrong type
            process="",
            record="R1",
            variants="bad",  # ty: ignore[invalid-argument-type] - wrong type
        ),
        min_count=11,
    )


_SPECS_MESSAGE_CASES: dict[str, tuple[Callable[[], object], tuple[str, ...]]] = {
    "requirement-no-bound": (
        lambda: Requirement(_OUT_MEASURE, margin=-1 * V),
        ("states a bound", "is negative"),
    ),
    "requirement-inverted": (
        lambda: Requirement(
            _OUT_MEASURE, minimum=3 * V, maximum=1 * V, margin=-0.5 * V
        ),
        ("exceeds its maximum", "is negative"),
    ),
    "specs-empty": (
        lambda: Specs([], yield_target=2.0),
        ("at least one requirement", "above 0 and at most 1"),
    ),
}


@pytest.mark.parametrize("name", list(_SPECS_MESSAGE_CASES))
def test_requirement_and_specs_report_all_errors_with_messages(name: str) -> None:
    build, fragments = _SPECS_MESSAGE_CASES[name]
    _expect_validation_errors(build, *fragments, count=2)


_DIMENSION_ERROR_CASES: dict[str, Callable[[], object]] = {
    "transient-stop": lambda: Transient(stop=1 * V),
    "noise-frequency": lambda: Noise("out", [1 * V]),
    "corner-temperature": lambda: Corner(temperature=5 * V),
    "plan-temperature": lambda: Plan({"op": OperatingPoint()}, temperature=5 * V),
    "options-nodeset": lambda: Options(nodesets={"out": 1 * mA}),
    "options-voltage-tolerance": lambda: Options(voltage_tolerance=1 * mA),
    "options-current-tolerance": lambda: Options(current_tolerance=1 * V),
    "options-charge-tolerance": lambda: Options(charge_tolerance=1 * V),
    "options-gmin": lambda: Options(gmin=1 * kohm),
    "conditions-temperature": lambda: Conditions(temperature=300 * V),
    "dc-sweep-mixed": lambda: DcSweep("R1", [1 * kohm, 2 * nF], parameter="resistance"),
    "dc-sweep-mixed-scalar": lambda: DcSweep(
        "R1", [1 * kohm, 2.0], parameter="resistance"
    ),
    "axis-mixed": lambda: Axis("R1", "resistance", [1 * kohm, 2 * nF]),
    "axis-mixed-scalar": lambda: Axis("R1", "resistance", [1 * kohm, 2.0]),
}


@pytest.mark.parametrize("name", list(_DIMENSION_ERROR_CASES))
def test_dimension_errors_still_propagate_directly(name: str) -> None:
    with pytest.raises(DimensionError):
        _DIMENSION_ERROR_CASES[name]()


def test_parameter_dimensions_checked_on_overrides_variants_and_sweeps() -> None:
    """Verifies circuit-bound overrides, variants, axes, and sweeps check units."""
    built = circuit(bench_circuit())
    simulation = Simulation(built)
    elaborated = simulate.Elaborated(built)

    # Dimensioned parameters reject wrong units and bare scalars everywhere.
    for bad in (1 * nF, 1000.0):
        with pytest.raises(
            DimensionError, match=r"R1 resistance takes a quantity in Ω"
        ):
            Simulation(built, overrides={"R1": {"resistance": bad}})
        with pytest.raises(
            DimensionError, match=r"R1 resistance takes a quantity in Ω"
        ):
            Simulation.from_elaborated(elaborated, {"R1": {"resistance": bad}})
        with pytest.raises(
            DimensionError, match=r"R1 resistance takes a quantity in Ω"
        ):
            simulation.run(DcSweep("R1", [bad], parameter="resistance"))
        with pytest.raises(
            DimensionError, match=r"R1 resistance takes a quantity in Ω"
        ):
            simulation.validate_plan(
                Plan({"sweep": DcSweep("R1", [bad], parameter="resistance")})
            )
        with pytest.raises(
            DimensionError, match=r"R1 resistance takes a quantity in Ω"
        ):
            simulation.validate_plan(
                Plan(
                    {"op": OperatingPoint()},
                    variants=Grid([Axis("R1", "resistance", [bad])]),
                )
            )
        with pytest.raises(
            DimensionError, match=r"R1 resistance takes a quantity in Ω"
        ):
            simulation.validate_plan(
                Plan(
                    {"op": OperatingPoint()},
                    variants=Explicit([Variant({"R1": {"resistance": bad}})]),
                )
            )
        with pytest.raises(
            DimensionError, match=r"R1 resistance takes a quantity in Ω"
        ):
            simulation.batch(
                {"op": OperatingPoint()},
                Explicit([Variant({"R1": {"resistance": bad}})]),
            )

    # Combined name and dimension failures collect into one PlanError.
    _expect_plan_error(
        lambda: simulation.validate_plan(
            Plan(
                {"op": OperatingPoint()},
                variants=Grid(
                    [
                        Axis("R9", "resistance", [1 * kohm]),
                        Axis("R1", "resistance", [1 * nF]),
                    ]
                ),
            )
        ),
        "no device at `R9`",
        "R1 resistance takes a quantity in Ω",
        count=2,
    )

    # Dimensionless parameters accept bare floats and reject dimensioned quantities.
    @component
    def transistor_stage(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        supply = c.place(devices.voltage_source(5 * V), name="VCC")
        base = c.place(devices.voltage_source(0.7 * V), name="VB")
        collector = c.place(devices.resistor(1 * kohm), name="RC")
        transistor = c.place(devices.bipolar_transistor(forward_beta=100), name="Q1")
        c.connect(supply.positive, collector.a)
        c.connect(collector.b, transistor.collector, name="out")
        c.connect(base.positive, transistor.base)
        c.connect(supply.negative, base.negative, transistor.emitter, ground)

    bjt = circuit(transistor_stage())
    overridden = Simulation(bjt, overrides={"Q1": {"forward_beta": 150.0}})
    assert overridden.run(OperatingPoint()).voltage("out") < 5.0
    swept = Simulation(bjt).run(DcSweep("Q1", [50.0, 150.0], parameter="forward_beta"))
    assert len(swept.swept) == 2
    batched = Simulation(bjt).batch(
        {"op": OperatingPoint()}, Grid([Axis("Q1", "forward_beta", [50.0, 150.0])])
    )
    assert batched.runs == 2
    with pytest.raises(DimensionError, match=r"takes a plain number, not 150 Ω"):
        Simulation(bjt, overrides={"Q1": {"forward_beta": 150 * ohm}})
    with pytest.raises(DimensionError, match=r"takes a plain number, not 50 Ω"):
        Simulation(bjt).run(DcSweep("Q1", [50 * ohm], parameter="forward_beta"))
