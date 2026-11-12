# Copyright 2026 Maxence Faldor

import json
from pathlib import Path

import pytest

from jolty import Builder, Role, circuit, component, devices, ir
from jolty.catalog import Catalog, Part, bill, mantissas, neighbours, values
from jolty.cli import main
from jolty.files import save_circuit
from jolty.units import V, kohm, nF

CATALOG = Catalog(
    currency="USD",
    parts=(
        Part(device="resistor", value=1e3, tolerance=0.01, price=0.01),
        Part(device="resistor", value=1e3, tolerance=0.001, price=0.08),
        Part(device="resistor", value=1e3, tolerance=0.05, price=0.005),
        Part(device="resistor", value=4.7e3, tolerance=0.01, price=0.01),
        Part(device="capacitor", value=1e-9, tolerance=0.1, price=0.02, number="C1N"),
    ),
)


@component
def stage(c: Builder) -> None:
    source = c.port("input")
    ground = c.port("ground", Role.GROUND)
    upper = c.place(devices.resistor(1 * kohm, tolerance=0.01), id="R1")
    lower = c.place(devices.resistor(4.7 * kohm, tolerance=0.001), id="R2")
    capacitor = c.place(devices.capacitor(1 * nF, tolerance=0.1), id="C1")
    ideal = c.place(devices.resistor(1 * kohm), id="R3")
    c.connect(source, upper.a)
    c.connect(upper.b, lower.a, capacitor.a, ideal.a)
    c.connect(lower.b, capacitor.b, ideal.b, ground)


@component
def bench(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(5 * V), id="V1")
    placed = c.place(stage(), id="X1")
    c.connect(source.positive, placed.input)
    c.connect(source.negative, placed.ground, ground)


def test_the_e_series_are_iec_60063_s() -> None:
    assert [len(mantissas(series)) for series in ("E3", "E6", "E12", "E24")] == [
        3,
        6,
        12,
        24,
    ]
    assert mantissas("E24")[11:14] == (3.0, 3.3, 3.6)
    e96 = mantissas("E96")
    assert (len(e96), e96[-1], e96[48]) == (96, 9.76, 3.16)
    assert mantissas("E48") == e96[::2]
    e192 = mantissas("E192")
    assert e192[::2] == e96
    assert e192[185] == 9.2


def test_values_span_decades_and_neighbours_bracket_a_value() -> None:
    assert values("E6", 0.5, 22.0) == (
        0.68,
        1.0,
        1.5,
        2.2,
        3.3,
        4.7,
        6.8,
        10.0,
        15.0,
        22.0,
    )
    assert neighbours("E12", 1e-9) == (1e-9, 1e-9)
    assert neighbours("E96", 1234.0) == (1210.0, 1240.0)
    assert neighbours("E24", 9.5e3) == (9.1e3, 10e3)
    with pytest.raises(ValueError, match="positive"):
        neighbours("E24", 0.0)
    with pytest.raises(ValueError, match="two positive bounds"):
        values("E24", 2.0, 1.0)


def test_a_part_is_the_cheapest_of_its_value_within_its_tolerance() -> None:
    assert CATALOG.cheapest("resistor", 1e3, 0.01) == CATALOG.parts[0]
    assert CATALOG.cheapest("resistor", 1e3, 0.1) == CATALOG.parts[2]
    assert CATALOG.cheapest("resistor", 1e3, 0.0001) is None
    assert CATALOG.cheapest("inductor", 1e3, 0.1) is None
    assert CATALOG.values("resistor", 0.01) == (1e3, 4.7e3)
    assert CATALOG.values("resistor", 0.001) == (1e3,)


def test_a_bill_prices_each_device_instance_or_says_why_not() -> None:
    built = circuit(bench())
    found = bill(built, CATALOG, "X1")
    assert [(line.instance, line.part.price) for line in found.lines] == [
        ("X1/R1", 0.01),
        ("X1/C1", 0.02),
    ]
    assert found.total == pytest.approx(0.03)
    assert [(each.instance, each.reason) for each in found.unpriced] == [
        ("X1/R2", "no resistor of 4.7 kΩ within 0.001 is in the catalog"),
        ("X1/R3", "its resistance has no tolerance, which a part must meet"),
    ]
    whole = bill(built, CATALOG)
    assert whole.unpriced[0].reason == (
        "the catalog prices resistors, capacitors and inductors, not voltage_sources"
    )
    with pytest.raises(KeyError, match="no component instance is at 'X1/R1'"):
        bill(built, CATALOG, "X1/R1")


def test_a_part_from_a_lot_stays_unpriced() -> None:
    lot = ir.Lot(name="die", relative=0.2)
    matched = devices.resistor(1 * kohm).toleranced("resistance", 0.001, lot=lot)

    @component
    def integrated(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        placed = c.place(matched, id="R1")
        c.connect(placed.a, placed.b, ground)

    (unpriced,) = bill(circuit(integrated()), CATALOG).unpriced
    assert unpriced.reason == (
        "its resistance comes from the lot die, whose shared spread no part of the "
        "catalog states"
    )


def test_the_cost_command_prices_a_saved_circuit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    saved, parts = tmp_path / "bench.jolty.json", tmp_path / "parts.json"
    save_circuit(saved, circuit(bench()))
    CATALOG.save(parts)
    assert Catalog.load(parts) == CATALOG
    arguments = ["cost", str(saved), "--catalog", str(parts), "--instance", "X1"]
    assert _run(*arguments) == 1
    text = capsys.readouterr().out
    assert text.startswith("2 parts for 0.03 USD, 2 instances unpriced\n")
    assert "  X1/R1: resistor of 1 kΩ within 0.01, 0.01 USD" in text
    assert _run(*arguments, "--output-format", "json") == 1
    assert json.loads(capsys.readouterr().out)["bill"]["currency"] == "USD"
    assert _run("cost", str(saved), "--catalog", str(tmp_path / "none.json")) == 2
    assert "no such catalog file" in capsys.readouterr().err


def _run(*arguments: str) -> int:
    try:
        main(arguments)
    except SystemExit as exit_:
        return exit_.code if isinstance(exit_.code, int) else 0
    return 0
