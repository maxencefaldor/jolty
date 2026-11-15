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
    circuit_hash,
    circuit_path,
    load_batches,
    load_circuit,
    load_provenance,
    load_results,
    plan_hash,
    results_path,
    save_batches,
    save_circuit,
    save_results,
)
from jolty.plan import Explicit, MonteCarlo, Plan, Product, Variant, decade
from jolty.specs import gain, value
from jolty.units import VOLTAGE, MHz, V, kHz, kohm, nF, ohm

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
)
PARAMETER = results.Parameter("X1/R1", "resistance", 1e3)
RECORDING = results.Recording(
    (
        results.RecordedDevice("C1", ("a", "b")),
        results.RecordedDevice(
            "M1", ("drain",), ("M1:drain",), variables=("threshold_voltage",)
        ),
    ),
    ("X1.p",),
    frozenset({"M1.drain"}),
)
"""Two devices recorded, their terminals then a port in four columns of currents,
their derivatives in eight, and a variable."""


EXCURSION = results.Excursion(
    path="X2", bound="load", low=0.0, high=2e-14, value=5e-13, time=1e-9, count=3
)
"""A logic cell's load outside the domain of its timing."""


def every_kind() -> dict[str, results.Result]:
    point = results.OperatingPoint(
        UNKNOWNS,
        np.array([1.0, -1e-3]),
        "gmin_stepping",
        recording=RECORDING,
        currents=np.array([1e-3, -1e-3, 0.0, 0.0]),
        charges=np.array([1e-12, -1e-12, 0.0, 0.0]),
        conductances=np.zeros(8),
        capacitances=np.array([1e-12, -1e-12, -1e-12, 1e-12, 0.0, 0.0, 0.0, 0.0]),
        readings=np.array([0.7]),
    )
    return {
        "op": point,
        "sweep": results.Sweep(
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
            readings=np.zeros((2, 1)),
        ),
        "bode": results.Response(
            UNKNOWNS,
            "V1",
            np.array([1e3]),
            np.array([[0.5 - 0.5j, 1j]]),
            point,
            RECORDING,
            np.array([[1j, -1j, 0, 0]]),
        ),
        "step": results.Waveforms(
            UNKNOWNS,
            np.array([0.0, 1e-6]),
            np.ones((2, 2)),
            STATISTICS,
            recording=RECORDING,
            currents=np.array([[1.0, 1.0, 1.0, 2.0], [1.0, 1.0, 1.0, 2.0]]),
            charges=np.ones((2, 4)),
            conductances=np.zeros((2, 8)),
            capacitances=np.ones((2, 8)),
            readings=np.ones((2, 1)),
            excursions=(EXCURSION,),
        ),
        "sensitivity": results.Sensitivities(
            UNKNOWNS,
            ("out",),
            (PARAMETER, results.Parameter("V1", "waveform.dc", 5.0)),
            np.array([[-2e-4, 0.4]]),
            point,
        ),
        "over": results.Sensitivities(
            UNKNOWNS,
            ("out",),
            (PARAMETER,),
            np.array([[[0.0]], [[1.0]]]),
            results.Waveforms(
                UNKNOWNS, np.array([0.0, 1e-6]), np.ones((2, 2)), STATISTICS
            ),
        ),
        "loop": results.LoopGain(
            UNKNOWNS, "VP", np.array([1e3, 1e6]), np.array([100 - 1j, -0.5j]), point
        ),
        "hiss": results.NoiseSpectrum(
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
        "states": results.OperatingPoints(UNKNOWNS, (point, point)),
        "pz": results.PolesZeros(
            UNKNOWNS,
            np.array([-1e3 + 2e3j, -1e3 - 2e3j]),
            np.array([0j]),
            "V1",
            "out",
            None,
            point,
        ),
        "poles": results.PolesZeros(
            UNKNOWNS, np.array([-1e6 + 0j]), None, None, None, None, point
        ),
        "quiet": results.NoiseSpectrum(
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
    assert isinstance(sweep, results.Sweep)
    assert (sweep.source, sweep.quantity) == ("V1", "voltage")
    op = read["op"]
    assert isinstance(op, results.OperatingPoint)
    assert op.current("C1", "a") == 1e-3
    assert op.charge("C1", "b") == -1e-12
    assert op.current("X1", "p") == 0
    assert op.capacitance("C1.a", "C1.b") == -1e-12
    assert op.conductance("M1:drain", "M1.drain") == 0
    assert op.variable("M1", "threshold_voltage") == 0.7
    assert op.recording.thermal == {"M1.drain"}
    bode = read["bode"]
    assert isinstance(bode, results.Response)
    np.testing.assert_array_equal(bode.current("C1", "b"), [-1j])
    assert bode.operating_point.current("C1", "a") == 1e-3
    assert bode.source == "V1"
    assert bode.operating_point.strategy == "gmin_stepping"
    step = read["step"]
    assert isinstance(step, results.Waveforms)
    assert step.statistics == STATISTICS
    # Each result keeps the excursions its run watched, or that no run watched them.
    assert step.excursions == (EXCURSION,)
    assert op.excursions is None
    np.testing.assert_array_equal(step.current("X1", "p"), [2, 2])
    np.testing.assert_array_equal(step.variable("M1", "threshold_voltage"), [1, 1])


def test_results_files_hold_sensitivities_and_what_they_differentiate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bench.results.npz"
    save_results(path, every_kind())
    read = load_results(path)
    sensitivity = read["sensitivity"]
    assert isinstance(sensitivity, results.Sensitivities)
    assert sensitivity.outputs == ("out",)
    assert sensitivity.parameters[0] == PARAMETER
    assert sensitivity.derivative("out", "V1", "waveform.dc") == 0.4
    assert isinstance(sensitivity.result, results.OperatingPoint)
    assert sensitivity.result.strategy == "gmin_stepping"
    over = read["over"]
    assert isinstance(over, results.Sensitivities)
    assert isinstance(over.result, results.Waveforms)
    np.testing.assert_array_equal(over.derivative("out", "X1/R1", "resistance"), [0, 1])
    loop = read["loop"]
    assert isinstance(loop, results.LoopGain)
    assert loop.probe == "VP"
    np.testing.assert_array_equal(loop.frequencies, [1e3, 1e6])
    assert loop.operating_point.current("C1", "a") == 1e-3


def test_a_results_file_describes_one_circuit(tmp_path: Path) -> None:
    other = results.OperatingPoint(UNKNOWNS[:1], np.array([1.0]), "direct")
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
        source = c.place(devices.voltage_source(1 * V), id="V1")
        resistor = c.place(devices.resistor(1 * kohm), id="R1")
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
    assert header["version"] == 9
    # Every result recorded RECORDING but the transient a sensitivity differentiates.
    assert len(header["recordings"]) == 2
    assert header["recordings"][0]["devices"][1] == {
        "path": "M1",
        "terminals": ["drain"],
        "internals": ["M1:drain"],
        "variables": ["threshold_voltage"],
    }
    read = load_results(path)
    first, second = read["op"], read["step"]
    assert isinstance(first, results.OperatingPoint)
    assert isinstance(second, results.Waveforms)
    assert first.recording is second.recording


def _loaded(resistance: float) -> ir.Circuit:
    """A source driving a resistor of `resistance` ohms."""

    @component
    def load(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(1 * V), id="V1")
        resistor = c.place(devices.resistor(resistance * ohm), id="R1")
        c.connect(source.positive, resistor.a)
        c.connect(source.negative, resistor.b, ground)

    return circuit(load())


def test_a_results_file_states_its_provenance(tmp_path: Path) -> None:
    written, plan = _loaded(1e3), Plan({"op": OperatingPoint()})
    path = tmp_path / "load.results.npz"
    save_results(path, every_kind(), circuit=written, plan=plan)
    found = load_provenance(path)
    assert found == Provenance.of(written, plan)
    assert found.jolty == jolty.__version__
    assert found.circuit == circuit_hash(written)
    assert found.plan == plan_hash(plan)
    save_results(path, every_kind())
    assert load_provenance(path).circuit is None
    assert load_provenance(path).build == build()


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
        source = c.place(devices.voltage_source(1 * V), id="V1")
        resistor = c.place(devices.resistor(1 * kohm, tolerance=0.05), id="R1")
        capacitor = c.place(devices.capacitor(1 * nF, tolerance=0.1), id="C1")
        load = c.place(devices.resistor(10 * kohm), id="R2")
        c.connect(source.positive, resistor.a, name="in")
        c.connect(resistor.b, capacitor.a, load.a, name="out")
        c.connect(capacitor.b, load.b, source.negative, ground)

    analyses = {
        "op": OperatingPoint(),
        "bode": Ac("V1", decade(1 * kHz, 10 * MHz, 2)),
        "sweep": DcSweep("R1", [1 * kohm, 2 * kohm], parameter="resistance"),
    }
    measurements = [value("op", "out"), gain("bode", "out", at=1 * MHz)]
    found = Simulation(filtered()).batch(
        analyses,
        Product(
            Explicit([Variant(), Variant({"R2": {"resistance": -1.0}})]), MonteCarlo(3)
        ),
        measurements=measurements,
        traced=["out"],
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
    @component
    def divider(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(5 * V), id="V1")
        upper = c.place(devices.resistor(1 * kohm), id="R1")
        lower = c.place(devices.resistor(1 * kohm), id="R2")
        c.connect(source.positive, upper.a)
        c.connect(upper.b, lower.a, name="out")
        c.connect(lower.b, source.negative, ground)

    sweep = Simulation(divider()).run(
        DcSweep("R2", [1 * kohm, 3 * kohm], parameter="resistance")
    )
    path = tmp_path / "divider.results.npz"
    save_results(path, {"sweep": sweep})
    read = load_results(path)["sweep"]
    assert_same(read, sweep, "sweep")
    assert isinstance(read, results.Sweep)
    assert (read.parameter, read.quantity) == ("resistance", None)
