# Copyright 2026 Maxence Faldor

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
STATISTICS = results.Statistics(initial="direct", accepted=2, rejected=1, iterations=7)


def every_kind() -> dict[str, results.Result]:
    point = results.OperatingPoint(UNKNOWNS, np.array([1.0, -1e-3]), "gmin_stepping")
    return {
        "op": point,
        "sweep": results.Sweep(UNKNOWNS, np.array([0.0, 1.0]), np.eye(2)),
        "bode": results.Response(
            UNKNOWNS, np.array([1e3]), np.array([[0.5 - 0.5j, 1j]]), point
        ),
        "step": results.Waveforms(
            UNKNOWNS, np.array([0.0, 1e-6]), np.ones((2, 2)), STATISTICS
        ),
    }


def test_results_files_hold_every_kind_of_result(tmp_path: Path) -> None:
    path = tmp_path / "bench.results.npz"
    written = every_kind()
    save_results(path, written)
    read = load_results(path)
    assert list(read) == list(written)
    for name, result in written.items():
        loaded = read[name]
        assert type(loaded) is type(result)
        assert loaded.unknowns == UNKNOWNS
        np.testing.assert_array_equal(loaded.values, result.values)
    bode = read["bode"]
    assert isinstance(bode, results.Response)
    assert bode.operating_point.strategy == "gmin_stepping"
    step = read["step"]
    assert isinstance(step, results.Waveforms)
    assert step.statistics == STATISTICS


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
