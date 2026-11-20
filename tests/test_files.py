# Copyright 2026 Maxence Faldor

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

import jolty
from jolty import (
    Ac,
    Builder,
    DcSweep,
    JoltyError,
    OperatingPoint,
    Role,
    Simulation,
    circuit,
    component,
    devices,
    ir,
    results,
)
from jolty.files import (
    Batch,
    Provenance,
    build,
    circuit_digest,
    circuit_path,
    load_batches,
    load_circuit,
    load_provenance,
    load_raw,
    load_results,
    plan_digest,
    raw_path,
    results_path,
    save_batches,
    save_circuit,
    save_raw,
    save_results,
)
from jolty.plan import (
    Characterized,
    Corner,
    Explicit,
    Mismatch,
    MonteCarlo,
    Plan,
    Product,
    Variant,
    decade,
)
from jolty.specs import gain, value
from jolty.units import VOLTAGE, MHz, V, celsius, kHz, kohm, nF, ohm
from sample_circuits import divider

UNKNOWNS = (results.Unknown("out", "voltage"), results.Unknown("V1:current", "current"))
STATISTICS = results.Statistics(
    initial="direct",
    accepted=2,
    rejected=1,
    landed=3,
    iterations=7,
    work=results.Work(
        iterations=9, factorizations=1, refactorizations=6, fallbacks=1, reuses=2
    ),
    profile=results.Profile(0.1, 0.4, 0.1, 0.2, 0.1, 0.05, 0.03, 0.02),
    charge_leak=results.ChargeLeak(
        signed=np.array([1e-18, -2e-18]),
        absolute=np.array([3e-18, 4e-18]),
    ),
)
PARAMETER = results.Parameter("X1/R1", "resistance", 1e3)
RECORDING = results.Recording(
    (
        results.RecordedDevice("C1", ("a", "b")),
        results.RecordedDevice(
            "M1",
            ("drain",),
            ("M1:drain",),
            variables=("threshold_voltage", "region", "saturation_margin"),
            regions=("linear", "saturated"),
            margins=("saturation_margin",),
        ),
    ),
    ("X1.p",),
    frozenset({"M1.drain"}),
)
"""Two devices recorded, their terminals then a port in four columns of currents,
their derivatives in eight, and three variables, a transistor's region and its margin
among them."""


EXCURSION = results.Excursion(
    instance="X2",
    bound="load",
    minimum=0.0,
    maximum=2e-14,
    value=5e-13,
    time=1e-9,
    count=3,
)
"""A logic cell's load outside the domain of its timing."""


SPREAD = ir.RandomVariable(
    lot="die",
    name="waveform.dc",
    mean=0,
    standard_deviation=0.01,
    derivatives={"V1": {"waveform.dc": 5.0}},
)
"""A lot's random variable, which moves a source's level."""


def every_kind() -> dict[str, results.AnalysisResult]:
    point = results.OperatingPointResult(
        UNKNOWNS,
        np.array([1.0, -1e-3]),
        "gmin_stepping",
        412,
        recording=RECORDING,
        currents=np.array([1e-3, -1e-3, 0.0, 0.0]),
        charges=np.array([1e-12, -1e-12, 0.0, 0.0]),
        conductances=np.zeros(8),
        capacitances=np.array([1e-12, -1e-12, -1e-12, 1e-12, 0.0, 0.0, 0.0, 0.0]),
        variables=np.array([0.7, 1.0, 0.2]),
    )
    return {
        "op": point,
        "sweep": results.DcSweepResult(
            UNKNOWNS,
            "V1",
            "voltage",
            np.array([0.0, 1.0]),
            np.eye(2),
            recording=RECORDING,
            currents=np.eye(2, 4),
            charges=np.zeros((2, 4)),
            conductances=np.zeros((2, 8)),
            capacitances=np.zeros((2, 8)),
            variables=np.zeros((2, 3)),
            strategies=("direct", "source_stepping"),
            iterations=(5, 230),
        ),
        "bode": results.AcResult(
            UNKNOWNS,
            "V1",
            np.array([1e3]),
            np.array([[0.5 - 0.5j, 1j]]),
            point,
            RECORDING,
            np.array([[1j, -1j, 0, 0]]),
        ),
        "step": results.TransientResult(
            UNKNOWNS,
            np.array([0.0, 1e-6]),
            np.ones((2, 2)),
            STATISTICS,
            recording=RECORDING,
            currents=np.array([[1.0, 1.0, 1.0, 2.0], [1.0, 1.0, 1.0, 2.0]]),
            charges=np.ones((2, 4)),
            conductances=np.zeros((2, 8)),
            capacitances=np.ones((2, 8)),
            variables=np.ones((2, 3)),
            excursions=(EXCURSION,),
        ),
        "steady": results.PeriodicSteadyStateResult(
            UNKNOWNS,
            np.array([0.0, 5e-7, 1e-6]),
            np.ones((3, 2)),
            STATISTICS,
            recording=RECORDING,
            currents=np.ones((3, 4)),
            charges=np.ones((3, 4)),
            conductances=np.zeros((3, 8)),
            capacitances=np.ones((3, 8)),
            variables=np.ones((3, 3)),
            period=1e-6,
            periods=3,
        ),
        "sensitivity": results.SensitivityResult(
            UNKNOWNS,
            ("out",),
            (PARAMETER, results.Parameter("V1", "waveform.dc", 5.0)),
            np.array([[-2e-4, 0.4]]),
            point,
        ),
        "tuning": results.SensitivityResult(
            UNKNOWNS,
            ("out",),
            (PARAMETER,),
            np.array([[[0.0]], [[1.0]], [[0.5]]]),
            results.PeriodicSteadyStateResult(
                UNKNOWNS,
                np.array([0.0, 5e-7, 1e-6]),
                np.ones((3, 2)),
                STATISTICS,
                period=1e-6,
                periods=3,
            ),
            np.array([2e-10]),
        ),
        "over": results.SensitivityResult(
            UNKNOWNS,
            ("out",),
            (PARAMETER,),
            np.array([[[0.0]], [[1.0]]]),
            results.TransientResult(
                UNKNOWNS, np.array([0.0, 1e-6]), np.ones((2, 2)), STATISTICS
            ),
        ),
        "spread": results.MismatchResult(
            UNKNOWNS,
            ("out",),
            (SPREAD,),
            np.array([[0.4]]),
            point,
        ),
        "loop": results.StabilityResult(
            UNKNOWNS, "VP", np.array([1e3, 1e6]), np.array([100 - 1j, -0.5j]), point
        ),
        "hiss": results.NoiseResult(
            UNKNOWNS,
            "out",
            None,
            "voltage",
            np.array([1e3, 1e6]),
            np.array([2e-17, 1e-17]),
            (results.NoiseSource("R1", "thermal"), results.NoiseSource("D1", "shot")),
            np.array([[1.5e-17, 0.5e-17], [0.5e-17, 0.5e-17]]),
            results.NoiseInput("V1", "voltage", np.array([0.5, 0.25j])),
            point,
        ),
        "states": results.OperatingPointsResult(UNKNOWNS, (point, point)),
        "pz": results.PoleZeroResult(
            UNKNOWNS,
            np.array([-1e3 + 2e3j, -1e3 - 2e3j]),
            np.array([0j]),
            "V1",
            "out",
            None,
            point,
        ),
        "poles": results.PoleZeroResult(
            UNKNOWNS, np.array([-1e6 + 0j]), None, None, None, None, point
        ),
        "quiet": results.NoiseResult(
            UNKNOWNS,
            "out",
            "in",
            "voltage",
            np.array([1e3]),
            np.array([0.0]),
            (),
            np.empty((0, 1)),
            None,
            point,
        ),
    }


def assert_same(loaded: object, written: object, where: str) -> None:
    """Asserts a result loaded is the one written, field by field, arrays and all."""
    assert type(loaded) is type(written), where
    if dataclasses.is_dataclass(written):
        for field in dataclasses.fields(written):
            if not field.compare:
                continue
            assert_same(
                getattr(loaded, field.name),
                getattr(written, field.name),
                f"{where}.{field.name}",
            )
    elif isinstance(written, np.ndarray):
        assert isinstance(loaded, np.ndarray), where
        np.testing.assert_array_equal(loaded, written, err_msg=where)
    elif isinstance(written, tuple | list):
        assert isinstance(loaded, tuple | list), where
        assert len(loaded) == len(written), where
        for index, (first, second) in enumerate(zip(loaded, written, strict=True)):
            assert_same(first, second, f"{where}[{index}]")
    else:
        assert loaded == written, where


def test_results_files_hold_every_kind_of_result(tmp_path: Path) -> None:
    path = tmp_path / "bench.results.npz"
    written = every_kind()
    save_results(path, written)
    read = load_results(path)
    assert list(read) == list(written)
    for name, result in written.items():
        assert_same(read[name], result, name)
    sweep = read["sweep"]
    assert isinstance(sweep, results.DcSweepResult)
    assert (sweep.source, sweep.nature) == ("V1", "voltage")
    assert (sweep.strategies, sweep.iterations) == (
        ("direct", "source_stepping"),
        (5, 230),
    )
    op = read["op"]
    assert isinstance(op, results.OperatingPointResult)
    assert op.current("C1", "a") == 1e-3
    assert op.charge("C1", "b") == -1e-12
    assert op.current("X1", "p") == 0
    assert op.capacitance("C1.a", "C1.b") == -1e-12
    assert op.conductance("M1:drain", "M1.drain") == 0
    assert op.variable("M1", "threshold_voltage") == 0.7
    assert op.region("M1") == "saturated"
    assert op.recording.thermal == {"M1.drain"}
    bode = read["bode"]
    assert isinstance(bode, results.AcResult)
    np.testing.assert_array_equal(bode.current("C1", "b"), [-1j])
    assert bode.operating_point.current("C1", "a") == 1e-3
    assert bode.source == "V1"
    assert bode.operating_point.strategy == "gmin_stepping"
    assert bode.operating_point.iterations == 412
    step = read["step"]
    assert isinstance(step, results.TransientResult)
    assert step.statistics == STATISTICS
    assert step.charge_leak("out") == 1e-18
    assert step.charge_leak("out", absolute=True) == 3e-18
    assert step.charge_leak("V1:current") == -2e-18
    assert step.charge_leak("V1:current", absolute=True) == 4e-18
    # Each result keeps the excursions its run watched, or that no run watched them.
    assert step.excursions == (EXCURSION,)
    assert op.excursions is None
    np.testing.assert_array_equal(step.current("X1", "p"), [2, 2])
    np.testing.assert_array_equal(step.variable("M1", "threshold_voltage"), [1, 1])


def test_results_files_hold_sensitivities_mismatches_and_what_they_differentiate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bench.results.npz"
    save_results(path, every_kind())
    read = load_results(path)
    sensitivity = read["sensitivity"]
    assert isinstance(sensitivity, results.SensitivityResult)
    assert sensitivity.outputs == ("out",)
    assert sensitivity.parameters[0] == PARAMETER
    assert sensitivity.derivative("out", "V1", "waveform.dc") == 0.4
    assert isinstance(sensitivity.result, results.OperatingPointResult)
    assert sensitivity.result.strategy == "gmin_stepping"
    over = read["over"]
    assert isinstance(over, results.SensitivityResult)
    assert isinstance(over.result, results.TransientResult)
    np.testing.assert_array_equal(over.derivative("out", "X1/R1", "resistance"), [0, 1])
    spread = read["spread"]
    assert isinstance(spread, results.MismatchResult)
    assert spread.random_variables == (SPREAD,)
    # Exact float round-trip through .npz archive plus quadrature (~5 eps rounding).
    assert spread.standard_deviation("out") == pytest.approx(0.004, rel=1e-15)
    assert isinstance(spread.result, results.OperatingPointResult)
    loop = read["loop"]
    assert isinstance(loop, results.StabilityResult)
    assert loop.probe == "VP"
    np.testing.assert_array_equal(loop.frequencies, [1e3, 1e6])
    assert loop.operating_point.current("C1", "a") == 1e-3


def test_a_results_file_describes_one_circuit(tmp_path: Path) -> None:
    other = results.OperatingPointResult(UNKNOWNS[:1], np.array([1.0]), "direct")
    with pytest.raises(ValueError, match="analyses of one circuit"):
        save_results(tmp_path / "mixed.npz", {"a": every_kind()["op"], "b": other})
    with pytest.raises(ValueError, match="analyses of one circuit"):
        save_results(tmp_path / "empty.npz", {})


def test_other_archives_are_not_results_files(tmp_path: Path) -> None:
    path = tmp_path / "other.npz"
    np.savez(path, header=np.array("{}"))
    with pytest.raises(JoltyError, match="is not a jolty results file"):
        load_results(path)


def test_circuit_files_round_trip(tmp_path: Path) -> None:
    @component
    def load(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(1 * V), name="V1")
        resistor = c.place(devices.resistor(1 * kohm), name="R1")
        c.connect(source.positive, resistor.a)
        c.connect(source.negative, resistor.b, ground)

    written = circuit(load())
    path = circuit_path(tmp_path / "load.py")
    assert path.name == "load.jolty.json"
    save_circuit(path, written)
    assert load_circuit(path) == written
    assert results_path(tmp_path / "load.py").name == "load.results.npz"


def test_a_results_file_names_each_recording_once(tmp_path: Path) -> None:
    path = tmp_path / "bench.results.npz"
    save_results(path, every_kind())
    with np.load(path) as data:
        header = json.loads(str(data["header"]))
    assert header["version"] == 20
    # Every result recorded RECORDING but the transient a sensitivity differentiates.
    assert len(header["recordings"]) == 2
    assert header["recordings"][0]["devices"][1] == {
        "path": "M1",
        "terminals": ["drain"],
        "internals": ["M1:drain"],
        "variables": ["threshold_voltage", "region", "saturation_margin"],
        "regions": ["linear", "saturated"],
        "margins": ["saturation_margin"],
    }
    read = load_results(path)
    first, second = read["op"], read["step"]
    assert isinstance(first, results.OperatingPointResult)
    assert isinstance(second, results.TransientResult)
    assert first.recording is second.recording


def _loaded(resistance: float) -> ir.Circuit:
    """Builds a voltage source driving a resistor of `resistance` ohms."""

    @component
    def load(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(1 * V), name="V1")
        resistor = c.place(devices.resistor(resistance * ohm), name="R1")
        c.connect(source.positive, resistor.a)
        c.connect(source.negative, resistor.b, ground)

    return circuit(load())


def test_a_results_file_states_its_provenance(tmp_path: Path) -> None:
    written, plan = _loaded(1e3), Plan({"op": OperatingPoint()})
    path = tmp_path / "load.results.npz"
    save_results(path, every_kind(), circuit=written, plan=plan)
    found = load_provenance(path)
    assert found == Provenance.from_circuit_and_plan(written, plan)
    assert found.jolty == jolty.__version__
    assert found.circuit == circuit_digest(written)
    assert found.plan == plan_digest(plan)
    save_results(path, every_kind())
    assert load_provenance(path).circuit is None
    assert load_provenance(path).build == build()


def test_plan_digest_is_insensitive_to_mapping_insertion_order() -> None:
    variable_first = ir.RandomVariable(
        name="vth0#0",
        mean=0.0,
        standard_deviation=0.01,
        derivatives={"M1": {"vth": 1.0, "kp": 2.0}, "M2": {"vth": 3.0}},
    )
    variable_second = ir.RandomVariable(
        name="vth0#0",
        mean=0.0,
        standard_deviation=0.01,
        derivatives={"M2": {"vth": 3.0}, "M1": {"kp": 2.0, "vth": 1.0}},
    )
    first = Plan(
        analyses={
            "op": OperatingPoint(),
            "ac": Ac("V1", decade(1 * kHz, 1 * MHz, 5)),
            "mis": Mismatch(["out"], random_variables=(variable_first,)),
        },
        corners={
            "hot": Corner(temperature=celsius(85), supply=3.0 * V, bias=1.2 * V),
            "cold": Corner(temperature=celsius(-40), bias=0.8 * V, supply=2.5 * V),
        },
        fidelity={
            "a/*": Characterized(1.8 * V),
            "b/*": Characterized(3.3 * V),
        },
        variants=Explicit(
            [Variant({"R1": {"resistance": 1 * kohm}, "R2": {"resistance": 2 * kohm}})]
        ),
    )
    second = Plan(
        analyses={
            "mis": Mismatch(["out"], random_variables=(variable_second,)),
            "ac": Ac("V1", decade(1 * kHz, 1 * MHz, 5)),
            "op": OperatingPoint(),
        },
        corners={
            "cold": Corner(temperature=celsius(-40), supply=2.5 * V, bias=0.8 * V),
            "hot": Corner(temperature=celsius(85), bias=1.2 * V, supply=3.0 * V),
        },
        fidelity={
            "b/*": Characterized(3.3 * V),
            "a/*": Characterized(1.8 * V),
        },
        variants=Explicit(
            [Variant({"R2": {"resistance": 2 * kohm}, "R1": {"resistance": 1 * kohm}})]
        ),
    )
    altered = Plan(
        analyses={
            "op": OperatingPoint(),
            "ac": Ac("V1", decade(1 * kHz, 1 * MHz, 5)),
            "mis": Mismatch(["out"], random_variables=(variable_first,)),
        },
        corners={
            "hot": Corner(temperature=celsius(90), supply=3.0 * V, bias=1.2 * V),
            "cold": Corner(temperature=celsius(-40), bias=0.8 * V, supply=2.5 * V),
        },
        fidelity={
            "a/*": Characterized(1.8 * V),
            "b/*": Characterized(3.3 * V),
        },
        variants=Explicit(
            [Variant({"R1": {"resistance": 1 * kohm}, "R2": {"resistance": 2 * kohm}})]
        ),
    )
    assert plan_digest(first) == plan_digest(second)
    assert len(plan_digest(first)) == 64
    assert plan_digest(first) != plan_digest(altered)


def test_the_results_of_another_circuit_are_refused(tmp_path: Path) -> None:
    written = _loaded(1e3)
    path = tmp_path / "load.results.npz"
    save_results(path, every_kind(), circuit=written)
    assert set(load_results(path, circuit=written)) == set(every_kind())
    with pytest.raises(
        JoltyError, match="results of another circuit than the one given"
    ):
        load_results(path, circuit=_loaded(2e3))
    # Results saved of no circuit, or read for none, are not compared.
    assert set(load_results(path)) == set(every_kind())
    save_results(path, every_kind())
    assert set(load_results(path, circuit=_loaded(2e3))) == set(every_kind())


def test_a_results_file_of_an_earlier_format_says_so(tmp_path: Path) -> None:
    path = tmp_path / "old.results.npz"
    header = {"format": "jolty-results", "version": 2, "unknowns": [], "analyses": {}}
    np.savez(path, header=np.array(json.dumps(header)))
    with pytest.raises(JoltyError, match="version 2 of the format, which jolty reads"):
        load_results(path)


def test_batch_files_hold_every_column_of_a_batch(tmp_path: Path) -> None:
    @component
    def filtered(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(1 * V), name="V1")
        resistor = c.place(devices.resistor(1 * kohm, tolerance=0.05), name="R1")
        capacitor = c.place(devices.capacitor(1 * nF, tolerance=0.1), name="C1")
        load = c.place(devices.resistor(10 * kohm), name="R2")
        c.connect(source.positive, resistor.a, name="in")
        c.connect(resistor.b, capacitor.a, load.a, name="out")
        c.connect(capacitor.b, load.b, source.negative, ground)

    analyses = {
        "op": OperatingPoint(),
        "bode": Ac("V1", decade(1 * kHz, 10 * MHz, 2)),
        "sweep": DcSweep("R1", [1 * kohm, 2 * kohm], parameter="resistance"),
    }
    measures = [value("op", "out"), gain("bode", "out", at=1 * MHz)]
    found = Simulation(filtered()).batch(
        analyses,
        Product(
            Explicit([Variant(), Variant({"R2": {"resistance": -1 * ohm}})]),
            MonteCarlo(3),
        ),
        measures=measures,
        traces=["out"],
    )
    assert found.problems, "the runs at a negative resistance make problems"
    path = tmp_path / "filtered.batches.npz"
    requirements = ("out in op ≥ 0 V", "gain of out at 1 MHz in bode ≤ 0 dB")
    written = Batch(found, requirements, ((0.0, None), (None, 0.0)))
    save_batches(path, {"monte-carlo": written})
    ((name, read),) = load_batches(path).items()
    assert name == "monte-carlo"
    assert_same(read, written, "batch")
    assert read.result.units == (VOLTAGE, "dB")
    other = tmp_path / "other.npz"
    np.savez(other, values=np.zeros(1))
    with pytest.raises(JoltyError, match="not a jolty batch file"):
        load_batches(other)


def test_results_files_hold_a_sweep_of_a_parameter(tmp_path: Path) -> None:
    sweep = Simulation(divider()).run(
        DcSweep("R2", [1 * kohm, 3 * kohm], parameter="resistance")
    )
    path = tmp_path / "divider.results.npz"
    save_results(path, {"sweep": sweep})
    read = load_results(path)["sweep"]
    assert_same(read, sweep, "sweep")
    assert isinstance(read, results.DcSweepResult)
    assert (read.parameter, read.nature) == ("resistance", None)


def test_results_files_round_trip_transient_without_charge_leak(
    tmp_path: Path,
) -> None:
    unleaked = results.Statistics("direct", 1, 0, 0, 2)
    step = results.TransientResult(
        UNKNOWNS, np.array([0.0, 1e-6]), np.ones((2, 2)), unleaked
    )
    path = tmp_path / "ngspice.results.npz"
    save_results(path, {"step": step})
    loaded = load_results(path)["step"]
    assert_same(loaded, step, "step")
    assert isinstance(loaded, results.TransientResult)
    assert loaded.statistics.charge_leak is None
    with pytest.raises(KeyError, match="no charge leak"):
        loaded.charge_leak("out")


def _assert_bits_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    """Asserts two float64/complex128 arrays have identical IEEE-754 bit patterns."""
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    if np.iscomplexobj(expected):
        np.testing.assert_array_equal(
            actual.real.view(np.uint64), expected.real.view(np.uint64)
        )
        np.testing.assert_array_equal(
            actual.imag.view(np.uint64), expected.imag.view(np.uint64)
        )
    else:
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_raw_path_uses_raw_suffix(tmp_path: Path) -> None:
    assert raw_path(tmp_path / "filter.py") == tmp_path / "filter.raw"


@pytest.mark.parametrize("binary", [True, False], ids=["binary", "ascii"])
def test_save_raw_and_load_raw_round_trip_every_analysis_and_nature(
    tmp_path: Path, *, binary: bool
) -> None:
    rich_unknowns = (
        results.Unknown("out", "voltage"),
        results.Unknown("X1/Mid", "voltage"),
        results.Unknown("D1:anode", "voltage"),
        results.Unknown("V1:current", "current"),
        results.Unknown("X1/V1:current", "current"),
        results.Unknown("K1:current1", "current"),
        results.Unknown("Q1:junction", "temperature"),
        results.Unknown("M1:state", "state"),
    )
    op = results.OperatingPointResult(
        rich_unknowns,
        np.array(
            [2.5, -0.0, 0.7123456789012345, -1.25e-3, 3.5e-6, -0.0, 312.5, 0.42],
            dtype=np.float64,
        ),
        strategy="direct",
        iterations=3,
    )
    dc_v = results.DcSweepResult(
        rich_unknowns,
        source="V1",
        nature="voltage",
        swept=np.array([-1.0, -0.0, 2.5], dtype=np.float64),
        values=np.arange(24, dtype=np.float64).reshape(3, 8) * 0.125,
    )
    dc_i = results.DcSweepResult(
        rich_unknowns,
        source="I1",
        nature="current",
        swept=np.array([0.0, 1e-3], dtype=np.float64),
        values=np.ones((2, 8), dtype=np.float64) * -2.5,
    )
    dc_temp = results.DcSweepResult(
        rich_unknowns,
        source="TEMP",
        nature="temperature",
        swept=np.array([250.0, 300.15], dtype=np.float64),
        values=np.zeros((2, 8), dtype=np.float64),
    )
    dc_param = results.DcSweepResult(
        rich_unknowns,
        source="R1",
        nature=None,
        swept=np.array([1e3, 2e3], dtype=np.float64),
        values=np.full((2, 8), 1.5, dtype=np.float64),
        parameter="resistance",
    )
    ac_values = np.empty((2, 8), dtype=np.complex128)
    ac_values.real = np.array(
        [
            [0.5, -0.0, 1.0, -1e-3, 0.0, -0.0, 0.25, 0.1],
            [0.125, -0.75, 0.0, 2e-3, -3e-6, 4.5, -0.0, -1.0],
        ],
        dtype=np.float64,
    )
    ac_values.imag = np.array(
        [
            [-0.5, -0.0, 0.0, 1e-3, -2e-6, -0.0, 0.0, -0.25],
            [-0.25, 0.5, -1.0, -0.0, 1e-6, -0.125, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    ac = results.AcResult(
        rich_unknowns,
        source="V1",
        frequencies=np.array([1e3, 1e6], dtype=np.float64),
        values=ac_values,
        operating_point=op,
    )
    tran = results.TransientResult(
        rich_unknowns,
        times=np.array([0.0, 1e-9, 5e-9], dtype=np.float64),
        values=np.linspace(-2.0, 3.0, 24, dtype=np.float64).reshape(3, 8),
        statistics=STATISTICS,
    )
    pss = results.PeriodicSteadyStateResult(
        rich_unknowns,
        times=np.array([0.0, 5e-7, 1e-6], dtype=np.float64),
        values=np.linspace(0.0, 1.0, 24, dtype=np.float64).reshape(3, 8),
        statistics=STATISTICS,
        period=1e-6,
        periods=2,
    )

    written: dict[str, results.AnalysisResult] = {
        "op": op,
        "dc_v": dc_v,
        "dc_i": dc_i,
        "dc_temp": dc_temp,
        "dc_param": dc_param,
        "bode": ac,
        "step": tran,
        "steady": pss,
    }
    path = tmp_path / "subdir" / ("all_bin.raw" if binary else "all_ascii.raw")
    save_raw(path, written, binary=binary)
    loaded = load_raw(path)
    assert list(loaded) == list(written)

    loaded_op = loaded["op"]
    assert isinstance(loaded_op, results.OperatingPointResult)
    assert loaded_op.unknowns == rich_unknowns
    assert loaded_op.strategy == "raw"
    assert loaded_op.iterations is None
    _assert_bits_equal(loaded_op.values, op.values)

    for key, expected_dc, expected_scale in (
        ("dc_v", dc_v, "v-sweep"),
        ("dc_i", dc_i, "i-sweep"),
        ("dc_temp", dc_temp, "temp-sweep"),
        ("dc_param", dc_param, "v-sweep"),
    ):
        got = loaded[key]
        assert isinstance(got, results.DcSweepResult)
        assert got.unknowns == rich_unknowns
        assert got.nature == expected_dc.nature
        assert got.source == expected_scale
        _assert_bits_equal(got.swept, expected_dc.swept)
        _assert_bits_equal(got.values, expected_dc.values)

    loaded_ac = loaded["bode"]
    assert isinstance(loaded_ac, results.AcResult)
    assert loaded_ac.unknowns == rich_unknowns
    assert loaded_ac.source == ""
    assert loaded_ac.operating_point.strategy == "raw"
    assert np.all(np.isnan(loaded_ac.operating_point.values))
    _assert_bits_equal(loaded_ac.frequencies, ac.frequencies)
    _assert_bits_equal(loaded_ac.values, ac.values)

    loaded_tran = loaded["step"]
    assert isinstance(loaded_tran, results.TransientResult)
    assert loaded_tran.unknowns == rich_unknowns
    assert loaded_tran.statistics.accepted == 2
    assert loaded_tran.statistics.rejected == 0
    _assert_bits_equal(loaded_tran.times, tran.times)
    _assert_bits_equal(loaded_tran.values, tran.values)

    loaded_pss = loaded["steady"]
    assert isinstance(loaded_pss, results.TransientResult)
    _assert_bits_equal(loaded_pss.times, pss.times)
    _assert_bits_equal(loaded_pss.values, pss.values)


@pytest.mark.parametrize("binary", [True, False], ids=["binary", "ascii"])
def test_save_raw_and_load_raw_handle_zero_points_and_zero_unknowns(
    tmp_path: Path, *, binary: bool
) -> None:
    empty_op = results.OperatingPointResult((), np.empty(0, dtype=np.float64), "direct")
    empty_ac = results.AcResult(
        UNKNOWNS,
        source="V1",
        frequencies=np.empty(0, dtype=np.float64),
        values=np.empty((0, 2), dtype=np.complex128),
        operating_point=results.OperatingPointResult(
            UNKNOWNS, np.zeros(2, dtype=np.float64), "direct"
        ),
    )
    empty_tran = results.TransientResult(
        (),
        times=np.empty(0, dtype=np.float64),
        values=np.empty((0, 0), dtype=np.float64),
        statistics=STATISTICS,
    )
    ac_no_unknowns = results.AcResult(
        (),
        source="V1",
        frequencies=np.array([1e3, 1e4], dtype=np.float64),
        values=np.empty((2, 0), dtype=np.complex128),
        operating_point=empty_op,
    )
    path = tmp_path / "edge.raw"
    save_raw(
        path,
        {
            "op": empty_op,
            "ac0": empty_ac,
            "tran0": empty_tran,
            "ac_empty_cols": ac_no_unknowns,
        },
        binary=binary,
    )
    loaded = load_raw(path)
    got_op = loaded["op"]
    assert isinstance(got_op, results.OperatingPointResult)
    assert got_op.unknowns == ()
    assert got_op.values.shape == (0,)

    got_ac = loaded["ac0"]
    assert isinstance(got_ac, results.AcResult)
    assert got_ac.frequencies.shape == (0,)
    assert got_ac.values.shape == (0, 2)

    got_tran = loaded["tran0"]
    assert isinstance(got_tran, results.TransientResult)
    assert got_tran.times.shape == (0,)
    assert got_tran.values.shape == (0, 0)
    assert got_tran.statistics.accepted == 0

    got_ac_cols = loaded["ac_empty_cols"]
    assert isinstance(got_ac_cols, results.AcResult)
    assert got_ac_cols.unknowns == ()
    np.testing.assert_array_equal(got_ac_cols.frequencies, [1e3, 1e4])
    assert got_ac_cols.values.shape == (2, 0)


def test_load_raw_parses_external_ngspice_conventions_and_duplicate_titles(
    tmp_path: Path,
) -> None:
    raw_text = (
        "Title: * RC circuit\r\n"
        "Date: Mon Nov 18 12:00:00 2026\r\n"
        "Plotname: Operating Point\r\n"
        "Flags: real padded\r\n"
        "No. Variables: 4\r\n"
        "No. Points: 1\r\n"
        "Command: op\r\n"
        "Variables:\r\n"
        "\t0\tv(out)\tvoltage\tgrid=1\r\n"
        "\t1\tV(X1.Mid)\tvoltage\r\n"
        "\t2\ti(V1)\tcurrent\r\n"
        "\t3\tl1#branch\tcurrent\r\n"
        "Values:\r\n"
        " 0\t2.5\r\n"
        "\t1.25\r\n"
        "\t-1e-3\r\n"
        "\t0.002\r\n"
        "Title: * RC circuit\r\n"
        "Plotname: Operating Point\r\n"
        "Flags: real\r\n"
        "No. Variables: 1\r\n"
        "No. Points: 1\r\n"
        "Variables:\r\n"
        "\t0\tv(out)\tvoltage\r\n"
        "Values:\r\n"
        " 0\t5.0\r\n"
        "Title:\r\n"
        "Plotname: Transient Analysis\r\n"
        "Flags: real\r\n"
        "No. Variables: 2\r\n"
        "No. Points: 2\r\n"
        "Variables:\r\n"
        "\t0\ttime\ttime\r\n"
        "\t1\tv(out)\tvoltage\r\n"
        "Values:\r\n"
        " 0\t0.0\r\n"
        "\t0.0\r\n"
        " 1\t1e-6\r\n"
        "\t3.0\r\n"
    )
    path = tmp_path / "external.raw"
    path.write_bytes(raw_text.encode("utf-8"))
    loaded = load_raw(path)
    assert list(loaded) == ["* RC circuit", "* RC circuit#2", "transient"]
    first = loaded["* RC circuit"]
    assert isinstance(first, results.OperatingPointResult)
    assert first.unknowns == (
        results.Unknown("out", "voltage"),
        results.Unknown("X1.Mid", "voltage"),
        results.Unknown("V1:current", "current"),
        results.Unknown("l1:current", "current"),
    )
    assert first.voltage("out") == 2.5
    assert first.voltage("X1.Mid") == 1.25
    assert first.current("V1") == -1e-3
    assert first.current("l1") == 0.002

    second = loaded["* RC circuit#2"]
    assert isinstance(second, results.OperatingPointResult)
    assert second.voltage("out") == 5.0

    tran = loaded["transient"]
    assert isinstance(tran, results.TransientResult)
    np.testing.assert_array_equal(tran.times, [0.0, 1e-6])
    np.testing.assert_array_equal(tran.voltage("out"), [0.0, 3.0])


def test_save_raw_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one or more analyses"):
        save_raw(tmp_path / "empty.raw", {})
    with pytest.raises(ValueError, match="line break"):
        save_raw(tmp_path / "bad.raw", {"op\n2": every_kind()["op"]})
    with pytest.raises(ValueError, match="NoiseResult"):
        save_raw(tmp_path / "bad.raw", {"hiss": every_kind()["hiss"]})


def test_load_raw_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "missing.raw")


@pytest.mark.parametrize(
    ("content", "pattern"),
    [
        (b"", "no SPICE raw plots"),
        (b"   \n\t ", "no SPICE raw plots"),
        (b"Not a raw file\n", "begins with `Title:`"),
        (b"Title: x\nPlotname: Operating Point\n", "no `Values:` or `Binary:`"),
        (
            b"Title: \xff\xfe\nPlotname: Operating Point\nValues:\n",
            "not valid UTF-8",
        ),
        (
            (
                b"Title: x\nFlags: real\nNo. Variables: 0\nNo. Points: 1\n"
                b"Variables:\nValues:\n 0\n"
            ),
            "no `Plotname:`",
        ),
        (
            (
                b"Title: x\nPlotname: Operating Point\n"
                b"No. Variables: 0\nNo. Points: 1\n"
                b"Variables:\nValues:\n 0\n"
            ),
            "exclusively in `Flags:`",
        ),
        (
            (
                b"Title: x\nPlotname: Operating Point\nFlags: real complex\n"
                b"No. Variables: 0\nNo. Points: 1\nVariables:\nValues:\n 0\n"
            ),
            "exclusively in `Flags:`",
        ),
        (
            (
                b"Title: x\nPlotname: Operating Point\nFlags: real\nNo. Points: 1\n"
                b"Variables:\nValues:\n 0\n"
            ),
            "no `No. Variables:`",
        ),
        (
            (
                b"Title: x\nPlotname: Operating Point\nFlags: real\n"
                b"No. Variables: -1\nNo. Points: 1\nVariables:\nValues:\n 0\n"
            ),
            "negative `No. Variables:`",
        ),
        (
            (
                b"Title: x\nPlotname: Operating Point\nFlags: real\n"
                b"No. Variables: nope\nNo. Points: 1\nVariables:\nValues:\n 0\n"
            ),
            "invalid `No. Variables:`",
        ),
        (
            (
                b"Title: x\nPlotname: Operating Point\nFlags: real\n"
                b"No. Variables: 0\nVariables:\nValues:\n"
            ),
            "no `No. Points:`",
        ),
        (
            (
                b"Title: x\nPlotname: Operating Point\nFlags: real\n"
                b"No. Variables: 1\nNo. Points: 1\nVariables:\nValues:\n 0\t1.0\n"
            ),
            "declares 1 variables, but lists 0",
        ),
        (
            (
                b"Title: x\nPlotname: Operating Point\nFlags: real\n"
                b"No. Variables: 1\nNo. Points: 1\nVariables:\n\t0\tonlytwo\nValues:\n"
            ),
            "malformed variable entry",
        ),
        (
            (
                b"Title: x\nPlotname: Operating Point\nFlags: real\n"
                b"No. Variables: 1\nNo. Points: 1\nVariables:\n"
                b"\tx\tout\tvoltage\nValues:\n"
            ),
            "non-integer variable index",
        ),
        (
            (
                b"Title: x\nPlotname: Operating Point\nFlags: real\n"
                b"No. Variables: 1\nNo. Points: 1\nVariables:\n"
                b"\t5\tout\tvoltage\nValues:\n"
            ),
            "lists variable index 5 at position 0",
        ),
        (
            (
                b"Title: x\nPlotname: Operating Point\nFlags: real\n"
                b"No. Variables: 1\nNo. Points: 1\nVariables:\n\t0\tout\tvoltage\n"
                b"Binary:\n\x00\x00\x00"
            ),
            "binary values end short",
        ),
        (
            (
                b"Title: x\nPlotname: Operating Point\nFlags: real\n"
                b"No. Variables: 1\nNo. Points: 1\nVariables:\n\t0\tout\tvoltage\n"
                b"Values:\n 0\n"
            ),
            "values end short",
        ),
        (
            (
                b"Title: x\nPlotname: Operating Point\nFlags: real\n"
                b"No. Variables: 1\nNo. Points: 1\nVariables:\n\t0\tout\tvoltage\n"
                b"Values:\n bad\t1.0\n"
            ),
            "not an integer index",
        ),
        (
            (
                b"Title: x\nPlotname: AC Analysis\nFlags: complex\n"
                b"No. Variables: 1\nNo. Points: 1\nVariables:\n"
                b"\t0\tfrequency\tfrequency\n"
                b"Values:\n 0\t1000.0\n"
            ),
            "complex raw value is `real,imag`",
        ),
        (
            (
                b"Title: x\nPlotname: Operating Point\nFlags: real\n"
                b"No. Variables: 1\nNo. Points: 1\nVariables:\n\t0\tout\tvoltage\n"
                b"Values:\n 0\t1.0,2.0\n"
            ),
            "real raw value is a single number",
        ),
        (
            (
                b"Title: x\nPlotname: Operating Point\nFlags: real\n"
                b"No. Variables: 1\nNo. Points: 1\nVariables:\n\t0\tout\tvoltage\n"
                b"Values:\n 0\tnotanumber\n"
            ),
            "is not a number",
        ),
        (
            (
                b"Title: x\nPlotname: Operating Point\nFlags: real\n"
                b"No. Variables: 1\nNo. Points: 1\nVariables:\n\t0\tout\tvoltage\n"
                b"Values:\n 0\t1.0\ngarbage_after_plot\n"
            ),
            "begins with `Title:`",
        ),
        (
            (
                b"Title: x\nPlotname: Noise Spectral Density\nFlags: real\n"
                b"No. Variables: 1\nNo. Points: 1\nVariables:\n"
                b"\t0\tfrequency\tfrequency\n"
                b"Values:\n 0\t1e3\n"
            ),
            "unsupported SPICE raw plot",
        ),
        (
            (
                b"Title: x\nPlotname: Operating Point\nFlags: complex\n"
                b"No. Variables: 1\nNo. Points: 1\nVariables:\n\t0\tout\tvoltage\n"
                b"Values:\n 0\t1.0,0.0\n"
            ),
            "holds complex values",
        ),
        (
            (
                b"Title: x\nPlotname: Operating Point\nFlags: real\n"
                b"No. Variables: 1\nNo. Points: 2\nVariables:\n\t0\tout\tvoltage\n"
                b"Values:\n 0\t1.0\n 1\t2.0\n"
            ),
            "holds 2 points, not 1",
        ),
        (
            (
                b"Title: x\nPlotname: DC transfer characteristic\nFlags: complex\n"
                b"No. Variables: 1\nNo. Points: 1\nVariables:\n\t0\tv-sweep\tvoltage\n"
                b"Values:\n 0\t1.0,0.0\n"
            ),
            "holds complex values",
        ),
        (
            (
                b"Title: x\nPlotname: DC transfer characteristic\nFlags: real\n"
                b"No. Variables: 0\nNo. Points: 0\nVariables:\nValues:\n"
            ),
            "has no sweep variable",
        ),
        (
            (
                b"Title: x\nPlotname: AC Analysis\nFlags: real\n"
                b"No. Variables: 1\nNo. Points: 1\nVariables:\n"
                b"\t0\tfrequency\tfrequency\n"
                b"Values:\n 0\t1e3\n"
            ),
            "holds real values, not complex",
        ),
        (
            (
                b"Title: x\nPlotname: AC Analysis\nFlags: complex\n"
                b"No. Variables: 0\nNo. Points: 0\nVariables:\nValues:\n"
            ),
            "has no frequency variable",
        ),
        (
            (
                b"Title: x\nPlotname: AC Analysis\nFlags: complex\n"
                b"No. Variables: 1\nNo. Points: 1\nVariables:\n\t0\ttime\ttime\n"
                b"Values:\n 0\t0.0,0.0\n"
            ),
            "not frequency",
        ),
        (
            (
                b"Title: x\nPlotname: Transient Analysis\nFlags: complex\n"
                b"No. Variables: 1\nNo. Points: 1\nVariables:\n\t0\ttime\ttime\n"
                b"Values:\n 0\t0.0,0.0\n"
            ),
            "holds complex values",
        ),
        (
            (
                b"Title: x\nPlotname: Transient Analysis\nFlags: real\n"
                b"No. Variables: 0\nNo. Points: 0\nVariables:\nValues:\n"
            ),
            "has no time variable",
        ),
        (
            (
                b"Title: x\nPlotname: Transient Analysis\nFlags: real\n"
                b"No. Variables: 1\nNo. Points: 1\nVariables:\n"
                b"\t0\tfrequency\tfrequency\n"
                b"Values:\n 0\t1e3\n"
            ),
            "not time",
        ),
    ],
)
def test_load_raw_rejects_corrupt_files(
    tmp_path: Path, content: bytes, pattern: str
) -> None:
    path = tmp_path / "corrupt.raw"
    path.write_bytes(content)
    with pytest.raises(JoltyError, match=pattern):
        load_raw(path)
