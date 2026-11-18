# Copyright 2026 Maxence Faldor
"""Tests for upfront plan/spec name resolution and multi-error Pydantic validation."""

import math
import re
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
    Hz,
    V,
    celsius,
    kHz,
    kohm,
    mA,
    mH,
    nF,
    ns,
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
    """Every error message in `error`, audited for fact-only wording."""
    messages = [entry["msg"] for entry in error.errors()]
    for message in messages:
        _assert_fact_only(message)
    return messages


@component
def amplifier_stage(c: Builder) -> None:
    """Subcomponent with two resistors, for testing hierarchy and fidelity scopes."""
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
    """Divider with a floating probe, inductor, capacitor, and two subcomponents."""
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
    """Circuit with a logic gate and a logic source."""
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


# --- close_names matching ---


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


# --- Upfront validation runs ZERO analyses ---


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

    with pytest.raises(PlanError) as raised:
        simulation.run_plan(plan, specs=requirements)

    assert calls == []
    assert len(raised.value.problems) == 3
    text = str(raised.value)
    _assert_fact_only(text)
    assert "'trna'; close names: tran" in text
    assert "'bdoe'; close names: bode" in text
    assert "'po'; close names: op" in text
    assert "defined analyses: op, bode, tran" in text


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


# --- Per-analysis upfront name checks ---


def test_record_reports_all_missing_paths_and_patterns_at_once() -> None:
    simulation = _build_simulation()
    with pytest.raises(PlanError) as raised:
        simulation.run(OperatingPoint(record=["R9", "X3/**", "X1/R9"]))
    assert len(raised.value.problems) == 3
    text = str(raised.value)
    _assert_fact_only(text)
    assert "'R9' to record; close names: R2, R1" in text
    assert "pattern 'X3/**' matches no instance" in text
    assert "'X1/R9' to record; close names: X1/R2, X1/R1" in text


def test_dc_sweep_level_and_parameter_checks() -> None:
    simulation = _build_simulation()
    # Missing source
    with pytest.raises(PlanError) as raised:
        simulation.run(DcSweep("V9", [1 * V]))
    text = str(raised.value)
    _assert_fact_only(text)
    assert "there is no instance at `V9`; close names: VP, V1" in text
    assert "independent sources: V1, VP" in text

    # Non-source
    with pytest.raises(PlanError) as raised:
        simulation.run(DcSweep("R1", [1 * V]))
    assert "`R1` is not an independent source" in str(raised.value)
    _assert_fact_only(str(raised.value))

    # Wrong level dimension alone still raises DimensionError
    with pytest.raises(DimensionError, match="a sweep level takes a quantity in V"):
        simulation.run(DcSweep("V1", [1 * mA]))

    # Wrong level dimension alongside bad nodeset reports BOTH in PlanError
    with pytest.raises(PlanError) as raised:
        simulation.run(DcSweep("V1", [1 * mA]), Options(nodesets={"outt": 1 * V}))
    assert len(raised.value.problems) == 2
    assert "close names: out" in str(raised.value)
    assert "quantity in V" in str(raised.value)
    _assert_fact_only(str(raised.value))

    # Parameter sweep: missing device
    with pytest.raises(PlanError) as raised:
        simulation.run(DcSweep("R9", [1 * kohm], parameter="resistance"))
    assert "there is no instance at `R9`; close names: R2, R1" in str(raised.value)

    # Parameter sweep: misspelled parameter
    with pytest.raises(PlanError) as raised:
        simulation.run(DcSweep("R1", [1 * kohm], parameter="resistence"))
    assert (
        "`R1` writes no number `resistence` to sweep; close names: resistance"
        in str(raised.value)
    )
    _assert_fact_only(str(raised.value))


def test_ac_stability_noise_transfer_pole_zero_checks() -> None:
    simulation = _build_simulation()

    # Ac bad source
    with pytest.raises(PlanError) as raised:
        simulation.run(Ac("V9", [1 * kHz]))
    assert "close names: VP, V1" in str(raised.value)
    _assert_fact_only(str(raised.value))

    # Stability: missing probe, grounded source, resistor
    with pytest.raises(PlanError) as raised:
        simulation.run(Stability("VQ", [1 * kHz]))
    assert "close names: VP" in str(raised.value)
    assert "probes: VP" in str(raised.value)
    _assert_fact_only(str(raised.value))

    for bad_probe in ("V1", "R1"):
        with pytest.raises(PlanError) as raised:
            simulation.run(Stability(bad_probe, [1 * kHz]))
        assert f"`{bad_probe}` is not a voltage source off the reference" in str(
            raised.value
        )
        assert "probes: VP" in str(raised.value)
        _assert_fact_only(str(raised.value))

    # Noise: ground reference is valid!
    simulation.validate_analysis(Noise("out", [1 * kHz], reference="ground"))
    simulation.validate_analysis(Noise("ground", [1 * kHz], reference="out"))

    # Noise: non-voltage over reference rejected
    with pytest.raises(PlanError) as raised:
        simulation.run(Noise("V1:current", [1 * kHz], reference="ground"))
    assert "an output over a node is a voltage" in str(raised.value)
    _assert_fact_only(str(raised.value))

    # Noise: bad output + bad reference + bad source all reported together
    with pytest.raises(PlanError) as raised:
        simulation.run(Noise("outt", [1 * kHz], reference="reff", source="V9"))
    assert len(raised.value.problems) == 3
    text = str(raised.value)
    _assert_fact_only(text)
    assert "`outt`; close names: out" in text
    assert "`reff`; close names: ref" in text
    assert "`V9`; close names: VP, V1" in text

    # TransferFunction: ground, current with reference, and inductor current rejected
    with pytest.raises(PlanError) as raised:
        simulation.run(TransferFunction("R1", "ground"))
    assert len(raised.value.problems) == 2
    assert "is the reference node" in str(raised.value)
    _assert_fact_only(str(raised.value))

    with pytest.raises(PlanError) as raised:
        simulation.run(TransferFunction("V1", "L1:current", reference="out"))
    assert len(raised.value.problems) == 2
    text = str(raised.value)
    _assert_fact_only(text)
    assert "has no reference" in text
    assert "voltage source's, not `L1:current`" in text

    with pytest.raises(PlanError) as raised:
        simulation.run(TransferFunction("V9", "outt", reference="reff"))
    assert len(raised.value.problems) == 3
    _assert_fact_only(str(raised.value))

    # PoleZero: ground reference valid; bad source + output + ref all reported
    simulation.validate_analysis(PoleZero("V1", "out", reference="ground"))
    with pytest.raises(PlanError) as raised:
        simulation.run(PoleZero("V9", "outt", reference="reff"))
    assert len(raised.value.problems) == 3
    _assert_fact_only(str(raised.value))


def test_transient_initial_conditions_and_logic_checks() -> None:
    simulation = _build_simulation()
    with pytest.raises(PlanError) as raised:
        simulation.run(
            Transient(
                stop=1 * us,
                initial_conditions={
                    "outt": 1 * V,
                    "V1:current": 0 * V,
                    "ground": 0 * V,
                },
            )
        )
    assert len(raised.value.problems) == 3
    text = str(raised.value)
    _assert_fact_only(text)
    assert "`outt`, which is no net of the circuit; close names: out" in text
    assert "`V1:current`, which is no net of the circuit (current unknown)" in text
    assert "`ground`, which is the reference node, not an electrical net" in text

    logic_simulation = Simulation(logic_circuit())
    with pytest.raises(PlanError) as raised:
        logic_simulation.validate_plan(
            Plan(
                {
                    "tran": Transient(stop=5 * ns, operating_point=False),
                    "pss": PeriodicSteadyState(period=5 * ns),
                }
            )
        )
    assert len(raised.value.problems) == 2
    text = str(raised.value)
    _assert_fact_only(text)
    assert "only an operating point settles" in text
    assert "without logic" in text


def test_nodesets_checked_across_all_entry_points() -> None:
    simulation = _build_simulation()
    bad_options = Options(nodesets={"outt": 1 * V, "V1:current": 0 * V})

    with pytest.raises(PlanError) as raised:
        simulation.check_operating_point(bad_options)
    assert len(raised.value.problems) == 2
    _assert_fact_only(str(raised.value))

    with pytest.raises(PlanError) as raised:
        simulation.batch({"op": OperatingPoint()}, Explicit([Variant()]), bad_options)
    assert len(raised.value.problems) == 2
    _assert_fact_only(str(raised.value))


def test_sensitivity_and_mismatch_checks_including_random_variables() -> None:
    simulation = _build_simulation()
    with pytest.raises(PlanError) as raised:
        simulation.run(Sensitivity(["outt", "reff"], of=Ac("V9", [1 * kHz])))
    assert len(raised.value.problems) == 3
    text = str(raised.value)
    _assert_fact_only(text)
    assert "`outt`; close names: out" in text
    assert "`reff`; close names: ref" in text
    assert "`V9`; close names: VP, V1" in text

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
    with pytest.raises(PlanError) as raised:
        simulation.run(
            Mismatch(
                ["outt"],
                of=DcSweep("V9", [1 * V]),
                random_variables=(first_random_variable, second_random_variable),
            )
        )
    assert len(raised.value.problems) == 4
    text = str(raised.value)
    _assert_fact_only(text)
    assert "`outt`; close names: out" in text
    assert "no device at `R9`; close names: R2, R1" in text
    assert "`R1` writes no number `resistence`; close names: resistance" in text
    assert "there is no instance at `V9`; close names: VP, V1" in text


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
                Variant({"X9/R1": {"resistance": 1e3}}),
                Variant({"C9": {}}),
            ]
        ),
    )

    with pytest.raises(PlanError) as raised:
        simulation.validate_plan(
            Plan({"op": OperatingPoint()}, variants=variants),
            traces=["outt", "tapp"],
        )
    assert len(raised.value.problems) == 6
    text = str(raised.value)
    _assert_fact_only(text)
    assert "no device at `R9`; close names: R2, R1" in text
    assert "`R1` writes no number `resistence`; close names: resistance" in text
    assert "no device at `X9/R1`; close names: X2/R1, X1/R1" in text
    assert "no device at `C9`; close names: C1" in text
    assert "trace `outt` names no unknown or logic net" in text
    assert "close names: out" in text
    assert "trace `tapp` names no unknown or logic net" in text
    assert "close names: tap" in text

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
    with pytest.raises(PlanError) as raised:
        fidelity.scopes(
            built,
            {
                "X9": Characterized(5 * V),
                "Z*": Characterized(5 * V),
            },
        )
    assert len(raised.value.problems) == 2
    text = str(raised.value)
    _assert_fact_only(text)
    assert "'X9' matches no component instance; close names: X2, X1" in text
    assert "'Z*' matches no component instance" in text


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
            # Crossing / frequency: wrong level dimension + wrong signal
            crossing("step", "out", 1 * mA) <= 1 * us,
            frequency("step", "out", level=1 * mA) >= 1 * Hz,
        ]
    )

    with pytest.raises(PlanError) as raised:
        simulation.validate_plan(plan, specs=requirements)

    text = str(raised.value)
    _assert_fact_only(text)
    assert "'outt'; close names: out" in text
    assert "'inn'; close names: in" in text
    assert "'reff'; close names: ref" in text
    assert "a transfer function" in text
    assert "an operating point holds one value, not one at 1 µs" in text
    assert "at is unset" in text
    assert "the noise analysis names no source to refer its noise to" in text
    assert "the minimum, 1 mA" in text
    assert "the maximum, 2 kHz" in text
    assert "the margin, 1 µs" in text
    assert len(raised.value.problems) >= 16


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

    with pytest.raises(PlanError) as raised:
        simulation.validate_plan(
            plan,
            specs=requirements,
            traces=["outt"],
            check_temperature=True,
        )

    text = str(raised.value)
    _assert_fact_only(text)
    assert len(raised.value.problems) >= 20
    # Corner with arguments (`rebuilt`) is NOT rejected upfront!
    assert "corner `rebuilt`" not in text
    assert "corner `bad_corner`" in text
    assert "is not the simulation's" in text
    assert "process corner ss" in text
    assert "bad_node" in text
    assert "'X9' matches no component instance; close names: X2, X1" in text
    assert "'C9' to record; close names: C1" in text
    assert "analysis `op`:" in text
    assert "analysis `sweep`:" in text
    assert "analysis `psweep`:" in text
    assert "analysis `bode`:" in text
    assert "analysis `step`:" in text
    assert "analysis `stab`:" in text
    assert "analysis `noise`:" in text
    assert "analysis `tf`:" in text
    assert "analysis `pz`:" in text
    assert "analysis `sens`:" in text
    assert "analysis `mis`:" in text
    assert "no device at `R9`; close names: R2, R1" in text
    assert "'bdoe'; close names: bode" in text
    assert "trace `outt`" in text


def test_plot_missing_analyses_includes_close_names(tmp_path: Path) -> None:
    simulation = _build_simulation()
    loaded = {
        "op": simulation.run(OperatingPoint()),
        "bode": simulation.run(Ac("V1", [1 * kHz, 10 * kHz])),
    }
    with pytest.raises(PlanError) as raised:
        service._plot(tmp_path / "out.npz", loaded, ["bdoe", "po"], [])  # noqa: SLF001 - direct unit test
    assert len(raised.value.problems) == 2
    text = str(raised.value)
    _assert_fact_only(text)
    assert "'bdoe'; close names: bode" in text
    assert "'po'; close names: op" in text


# --- Item 9: Multi-error Pydantic validation for Plan / Options / Specs ---


def test_options_and_parallelism_report_all_errors_at_once() -> None:
    with pytest.raises(ValidationError) as raised:
        Options(
            relative_tolerance=-1e-3,
            voltage_tolerance=0.0,
            current_tolerance=math.nan,
            charge_tolerance=True,
            gmin=-1e-12,
            max_dc_iterations=0,
            max_transient_iterations=False,
            truncation_tolerance=math.inf,
            max_memory=-1.0,
            profile="yes",  # ty: ignore[invalid-argument-type] - wrong type
            nodesets={"a": "bad", 1: 2 * V, "b": math.nan * V},  # ty: ignore[invalid-argument-type] - wrong types
        )
    assert raised.value.error_count() == 13
    _validation_messages(raised.value)

    with pytest.raises(ValidationError) as raised:
        Parallelism(
            min_solves_per_task=0,
            min_instances_per_task=-1,
            min_unknowns_per_task=True,
            max_unknowns_per_interior=0,
        )
    assert raised.value.error_count() == 4
    _validation_messages(raised.value)

    # JSON / dict validation via TypeAdapter reports nested Parallelism + extra fields!
    with pytest.raises(ValidationError) as raised:
        TypeAdapter(Options).validate_json(
            '{"relative_tolerance": -1, "gmin": -2, '
            '"parallelism": {"min_solves_per_task": 0, "unknown_inner": 5}, '
            '"unknown_outer": 9}'
        )
    assert raised.value.error_count() == 5
    _validation_messages(raised.value)


def test_analyses_report_all_construction_errors_at_once() -> None:
    with pytest.raises(ValidationError) as raised:
        Transient(
            stop=-1 * us,
            max_step=-2 * us,
            method="rk4",  # ty: ignore[invalid-argument-type] - wrong literal
            operating_point="no",  # ty: ignore[invalid-argument-type] - wrong type
            breakpoints=[0 * us, math.nan * us],
            record="R1",
            regions=1,  # ty: ignore[invalid-argument-type] - wrong type
        )
    assert raised.value.error_count() == 8
    _validation_messages(raised.value)

    with pytest.raises(ValidationError) as raised:
        PeriodicSteadyState(
            period=-1 * us,
            max_step=-1 * us,
            method="rk4",  # ty: ignore[invalid-argument-type] - wrong literal
        )
    assert raised.value.error_count() == 3
    _validation_messages(raised.value)

    with pytest.raises(ValidationError) as raised:
        PeriodicSteadyState(breakpoints=[1 * us])
    assert raised.value.error_count() == 1
    _validation_messages(raised.value)

    with pytest.raises(ValidationError) as raised:
        DcSweep(source="", values=[math.nan * V, math.inf * V], parameter="")
    assert raised.value.error_count() == 4
    _validation_messages(raised.value)

    with pytest.raises(ValidationError) as raised:
        Ac(source="", frequencies=[-1 * Hz, math.nan * Hz])
    assert raised.value.error_count() == 3
    _validation_messages(raised.value)

    with pytest.raises(ValidationError) as raised:
        Noise(output="", frequencies=[0 * Hz, -1 * Hz], source="")
    assert raised.value.error_count() == 4
    _validation_messages(raised.value)

    with pytest.raises(ValidationError) as raised:
        Stability(probe="", frequencies=[-1 * Hz])
    assert raised.value.error_count() == 2
    _validation_messages(raised.value)

    with pytest.raises(ValidationError) as raised:
        OperatingPoints(
            max_solutions=0,
            record="X1",
            regions=1,  # ty: ignore[invalid-argument-type] - wrong type
        )
    assert raised.value.error_count() == 3
    _validation_messages(raised.value)

    with pytest.raises(ValidationError) as raised:
        TransferFunction(source="", output="out", reference="out")
    assert raised.value.error_count() == 2
    _validation_messages(raised.value)

    with pytest.raises(ValidationError) as raised:
        PoleZero(source="", output="out", reference="out")
    assert raised.value.error_count() == 2
    _validation_messages(raised.value)

    with pytest.raises(ValidationError) as raised:
        Sensitivity(outputs="out", of=OperatingPoint())  # ty: ignore[invalid-argument-type] - wrong types
    assert raised.value.error_count() == 2
    _validation_messages(raised.value)

    bad_random_variable = ir.RandomVariable(
        name="bad",
        mean=0.0,
        standard_deviation=1.0,
        derivatives={"R1": {"resistance": math.nan}},
    )
    with pytest.raises(ValidationError) as raised:
        Mismatch(
            outputs=[1],  # ty: ignore[invalid-argument-type] - wrong types
            of=OperatingPoint(),  # ty: ignore[invalid-argument-type] - wrong types
            random_variables=[bad_random_variable, "nope"],  # ty: ignore[invalid-argument-type] - wrong types
        )
    assert raised.value.error_count() == 4
    _validation_messages(raised.value)


def test_variants_conditions_corners_and_plan_report_all_errors_at_once() -> None:
    with pytest.raises(ValidationError) as raised:
        MonteCarlo(samples=0, seed=-1)
    assert raised.value.error_count() == 2
    _validation_messages(raised.value)

    with pytest.raises(ValidationError) as raised:
        Variant(
            overrides={"R1": {"resistance": math.nan, "b": "x"}},  # ty: ignore[invalid-argument-type] - wrong types
            temperature=math.nan * celsius(0),
        )
    assert raised.value.error_count() == 3
    _validation_messages(raised.value)

    with pytest.raises(ValidationError) as raised:
        Axis("R1", None, [math.nan * V, "bad"])  # ty: ignore[invalid-argument-type] - wrong types
    assert raised.value.error_count() == 3
    _validation_messages(raised.value)

    with pytest.raises(ValidationError) as raised:
        Grid(axes=["bad", 1], warm="no")  # ty: ignore[invalid-argument-type] - wrong types
    assert raised.value.error_count() == 3
    _validation_messages(raised.value)

    with pytest.raises(ValidationError) as raised:
        Product(first="a", second="b")  # ty: ignore[invalid-argument-type] - wrong types
    assert raised.value.error_count() == 2
    _validation_messages(raised.value)

    with pytest.raises(ValidationError) as raised:
        WorstCase(step=2.0)
    assert raised.value.error_count() == 1
    _validation_messages(raised.value)

    with pytest.raises(ValidationError) as raised:
        Characterized(
            supply=-1 * V,
            slews=(2.0, 1.0),
            fanouts=(-1.0, 1.0),
            spot_checks=-1,
            tolerance=-0.1,
            seed=-1,
        )
    assert raised.value.error_count() == 6
    _validation_messages(raised.value)

    with pytest.raises(ValidationError) as raised:
        Conditions(temperature=math.nan, process="", mismatch=-1)
    assert raised.value.error_count() == 3
    _validation_messages(raised.value)

    with pytest.raises(ValidationError) as raised:
        Corner(temperature=math.nan * celsius(27), process="")
    assert raised.value.error_count() == 2
    _validation_messages(raised.value)

    with pytest.raises(ValidationError) as raised:
        recorded("X1")
    assert raised.value.error_count() == 1
    _validation_messages(raised.value)

    with pytest.raises(ValidationError) as raised:
        Plan(
            analyses={"bad name": "not_analysis", "also bad": 123},  # ty: ignore[invalid-argument-type] - wrong types
            options="bad",  # ty: ignore[invalid-argument-type] - wrong type
            corners={"bad corner": "not_corner"},  # ty: ignore[invalid-argument-type] - wrong type
            monte_carlo="bad",  # ty: ignore[invalid-argument-type] - wrong type
            worst_case="bad",  # ty: ignore[invalid-argument-type] - wrong type
            fidelity={"": "not_char"},  # ty: ignore[invalid-argument-type] - wrong type
            process="",
            record="R1",
            variants="bad",  # ty: ignore[invalid-argument-type] - wrong type
        )
    assert raised.value.error_count() >= 11
    _validation_messages(raised.value)


def test_requirement_and_specs_report_all_errors_at_once() -> None:
    measure = value("op", "out")

    # Missing bound + negative margin BOTH reported
    with pytest.raises(ValidationError) as raised:
        Requirement(measure, margin=-1 * V)
    assert raised.value.error_count() == 2
    messages = _validation_messages(raised.value)
    assert any("states a bound" in message for message in messages)
    assert any("is negative" in message for message in messages)

    # Minimum exceeds maximum + negative margin BOTH reported
    with pytest.raises(ValidationError) as raised:
        Requirement(measure, minimum=3 * V, maximum=1 * V, margin=-0.5 * V)
    assert raised.value.error_count() == 2
    messages = _validation_messages(raised.value)
    assert any("exceeds its maximum" in message for message in messages)
    assert any("is negative" in message for message in messages)

    # Empty specs + bad yield_target BOTH reported
    with pytest.raises(ValidationError) as raised:
        Specs([], yield_target=2.0)
    assert raised.value.error_count() == 2
    messages = _validation_messages(raised.value)
    assert any("at least one requirement" in message for message in messages)
    assert any("above 0 and at most 1" in message for message in messages)

    # Multiple bad requirements + bad yield_target ALL reported
    with pytest.raises(ValidationError) as raised:
        Specs([measure, "bad"], yield_target=0.0)  # ty: ignore[invalid-argument-type] - wrong types
    assert raised.value.error_count() == 3
    _validation_messages(raised.value)


def test_dimension_errors_still_propagate_directly() -> None:
    with pytest.raises(DimensionError):
        Transient(stop=1 * V)
    with pytest.raises(DimensionError):
        Noise("out", [1 * V])
    with pytest.raises(DimensionError):
        Corner(temperature=5 * V)
    with pytest.raises(DimensionError):
        Plan({"op": OperatingPoint()}, temperature=5 * V)
    with pytest.raises(DimensionError):
        Options(nodesets={"out": 1 * mA})
