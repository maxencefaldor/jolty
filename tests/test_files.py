# Copyright 2026 Maxence Faldor

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from jolty import Builder, JoltyError, Role, circuit, component, devices, results
from jolty.files import (
    circuit_path,
    load_circuit,
    load_results,
    results_path,
    save_circuit,
    save_results,
)
from jolty.units import V, kohm

UNKNOWNS = (results.Unknown("out", "voltage"), results.Unknown("V1:current", "current"))
STATISTICS = results.Statistics(
    initial="direct", accepted=2, rejected=1, landed=3, iterations=7
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
    assert header["version"] == 4
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


def test_a_results_file_of_an_earlier_format_says_so(tmp_path: Path) -> None:
    path = tmp_path / "old.results.npz"
    header = {"format": "jolty-results", "version": 2, "unknowns": [], "analyses": {}}
    np.savez(path, header=np.array(json.dumps(header)))
    with pytest.raises(JoltyError, match="version 2 of the format, which jolty reads"):
        load_results(path)
