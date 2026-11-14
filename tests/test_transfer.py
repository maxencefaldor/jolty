# Copyright 2026 Maxence Faldor

import math
from pathlib import Path

import pytest

from jolty import (
    Builder,
    Role,
    Simulation,
    TransferFunction,
    component,
    devices,
    files,
    results,
)
from jolty.report import TransferReport, report
from jolty.units import V, kohm, mA


@component
def divider(c: Builder) -> None:
    """1 V across 1 kΩ over 3 kΩ: 0.75 V/V, 4 kΩ in, 750 Ω out."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1 * V), id="V1")
    upper = c.place(devices.resistor(1 * kohm), id="R1")
    lower = c.place(devices.resistor(3 * kohm), id="R2")
    c.connect(source.positive, upper.a, name="in")
    c.connect(upper.b, lower.a, name="out")
    c.connect(lower.b, source.negative, ground)


@component
def driven(c: Builder) -> None:
    """A current source into 2 kΩ, beside a voltage source that drives nothing."""
    ground = c.port("ground", Role.GROUND)
    current = c.place(devices.current_source(1 * mA), id="I1")
    load = c.place(devices.resistor(2 * kohm), id="R1")
    idle = c.place(devices.voltage_source(1 * V), id="V1")
    c.connect(current.negative, load.a, name="load")
    c.connect(current.positive, load.b, ground)
    c.connect(idle.positive, name="alone")
    c.connect(idle.negative, ground)


def test_a_dividers_transfer_function_is_its_ratio_and_its_resistances() -> None:
    found = Simulation(divider()).run(TransferFunction("V1", "out"))
    assert found.input == "voltage"
    # A handful of roundings of numbers of order 1 to 10⁴: within 1e-12 of each.
    assert found.gain == pytest.approx(0.75, rel=1e-12)
    assert found.input_resistance == pytest.approx(4e3, rel=1e-12)
    assert found.output_resistance == pytest.approx(750, rel=1e-12)
    described = report("tf", found, 8).describe()
    assert described == (
        "tf: transfer function from V1 to out\n"
        "  gain 0.75 V/V\n"
        "  input resistance 4 kΩ\n"
        "  output resistance 750 Ω"
    )


def test_a_current_input_gains_in_ohms_and_an_idle_source_sees_no_load() -> None:
    simulation = Simulation(driven())
    through = simulation.run(TransferFunction("I1", "load"))
    assert (through.input, through.gain) == ("current", pytest.approx(2e3, rel=1e-12))
    assert report("tf", through, 8).describe().splitlines()[1] == "  gain 2000 V/A"
    idle = simulation.run(TransferFunction("V1", "alone"))
    assert idle.input_resistance == math.inf
    written = report("idle", idle, 8).model_dump_json()
    assert '"input_resistance":"Infinity"' in written
    assert TransferReport.model_validate_json(written).input_resistance == math.inf


def test_a_transfer_function_survives_its_results_file(tmp_path: Path) -> None:
    found = Simulation(driven()).run(TransferFunction("V1", "alone"))
    path = tmp_path / "driven.results.npz"
    files.save_results(path, {"tf": found})
    (loaded,) = files.load_results(path).values()
    assert isinstance(loaded, results.Transfer)
    numbers = ("gain", "input_resistance", "output_resistance")
    assert [getattr(loaded, name) for name in numbers] == [
        getattr(found, name) for name in numbers
    ]
    assert (loaded.source, loaded.input, loaded.output, loaded.reference) == (
        "V1",
        "voltage",
        "alone",
        None,
    )
    assert (loaded.operating_point.values == found.operating_point.values).all()
