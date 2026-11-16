# Copyright 2026 Maxence Faldor
"""Components written as KiCad netlists, checked against KiCad's footprint libraries.

The library test reads the footprints KiCad installs, or those its environment
variables name, and is skipped where there are none.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from jolty import Builder, Role, circuit, component, devices, kicad, parts
from jolty.catalog import Catalog, Part
from jolty.cli import main
from jolty.errors import JoltyError, NetlistError
from jolty.files import save_circuit
from jolty.units import K, V, W, kohm, m, nF

PARTS = Catalog(
    currency="USD",
    parts=(
        Part(
            kind="resistor",
            value=4.7e3,
            tolerance=0.01,
            price=0.01,
            package="0603",
            footprint="Resistor_SMD:R_0603_1608Metric",
        ),
        Part(
            kind="resistor",
            value=10e3,
            tolerance=0.01,
            price=0.01,
            package="0603",
            footprint="Resistor_SMD:R_0603_1608Metric",
        ),
        Part(
            kind="capacitor",
            value=1e-9,
            tolerance=0.1,
            price=0.02,
            package="0603",
            footprint="Capacitor_SMD:C_0603_1608Metric",
        ),
        Part(
            number="2N3904",
            price=0.12,
            package="TO-92",
            footprint="Package_TO_SOT_THT:TO-92_Inline",
            pins={"emitter": ("1",), "base": ("2",), "collector": ("3",)},
        ),
        Part(
            number="1N4148",
            price=0.02,
            package="DO-35",
            footprint="Diode_THT:D_DO-35_SOD27_P7.62mm_Horizontal",
            pins={"cathode": ("1",), "anode": ("2",)},
        ),
        Part(
            number="IRF511",
            price=1.1,
            package="TO-220AB",
            footprint="Package_TO_SOT_THT:TO-220-3_Vertical",
            pins={"gate": ("1",), "drain": ("2",), "source": ("3",)},
        ),
        Part(
            number="2N7000",
            price=0.1,
            package="TO-92",
            footprint="Package_TO_SOT_THT:TO-92_Inline",
            pins={"source": ("1",), "gate": ("2",), "drain": ("3",), "bulk": ("1",)},
        ),
    ),
)
"""Parts with their footprints in KiCad's libraries, and the pins of their makers'
datasheets: the 2N3904's and the 2N7000's TO-92, the 1N4148's DO-35, its cathode by
the band, and the IRF511's TO-220AB."""


@component
def driver(c: Builder) -> None:
    """A transistor that sinks a load, clamped, and a power MOSFET beside it."""
    supply = c.port("supply", Role.SUPPLY)
    control = c.port("control", Role.INPUT)
    ground = c.port("ground", Role.GROUND)
    load = c.place(devices.resistor(4.7 * kohm, tolerance=0.01), name="R1")
    drive = c.place(devices.resistor(10 * kohm, tolerance=0.01), name="R2")
    snubber = c.place(devices.capacitor(1 * nF, tolerance=0.1), name="C1")
    switch = c.place(parts.q2n3904(heated=True), name="Q1")
    clamp = c.place(parts.d1n4148(), name="D1")
    power = c.place(parts.irf511(), name="Q2")
    sink = c.place(devices.thermal_resistance(50 * K / W), name="RT")
    c.connect(supply, load.a, clamp.cathode)
    c.connect(load.b, switch.collector, clamp.anode, snubber.a, power.drain, name="out")
    c.connect(control, drive.a, power.gate)
    c.connect(drive.b, switch.base, name="base")
    c.connect(switch.emitter, snubber.b, power.source, ground)
    c.connect(switch.thermal, sink.a)
    c.connect(sink.b, ground)


@component
def bench(c: Builder) -> None:
    ground = c.port("0", Role.GROUND)
    supply = c.place(devices.voltage_source(5 * V), name="V1")
    placed = c.place(driver(), name="X1")
    c.connect(supply.positive, placed.supply)
    c.connect(supply.negative, placed.ground, placed.control, ground)


def _pads(netlist: kicad.Netlist) -> dict[str, set[tuple[str, str, str]]]:
    """Each net's pads, as their footprints' references, numbers and terminals."""
    return {
        net.name: {
            (node.reference, node.pad, "/".join(node.terminals)) for node in net.nodes
        }
        for net in netlist.nets
    }


def test_each_part_is_a_footprint_on_the_nets_its_pins_join() -> None:
    written = kicad.netlist(circuit(bench()), PARTS, "X1")
    assert [
        (each.reference, each.instance, each.value, each.part.footprint)
        for each in written.footprints
    ] == [
        ("R1", "R1", "4.7 kΩ", "Resistor_SMD:R_0603_1608Metric"),
        ("R2", "R2", "10 kΩ", "Resistor_SMD:R_0603_1608Metric"),
        ("C1", "C1", "1 nF", "Capacitor_SMD:C_0603_1608Metric"),
        ("Q1", "Q1", "2N3904", "Package_TO_SOT_THT:TO-92_Inline"),
        ("D1", "D1", "1N4148", "Diode_THT:D_DO-35_SOD27_P7.62mm_Horizontal"),
        ("U1", "Q2", "IRF511", "Package_TO_SOT_THT:TO-220-3_Vertical"),
    ]
    # The thermal resistance has no footprint, and the transistor's thermal terminal
    # no pad; the ground net takes the ground port's name, the others theirs.
    assert _pads(written) == {
        "base": {("R2", "2", "b"), ("Q1", "2", "base")},
        "control": {("R2", "1", "a"), ("U1", "1", "gate")},
        "ground": {("C1", "2", "b"), ("Q1", "1", "emitter"), ("U1", "3", "source")},
        "out": {
            ("R1", "2", "b"),
            ("C1", "1", "a"),
            ("Q1", "3", "collector"),
            ("D1", "2", "anode"),
            ("U1", "2", "drain"),
        },
        "supply": {("R1", "1", "a"), ("D1", "1", "cathode")},
    }
    assert [net.name for net in written.nets] == sorted(_pads(written))
    assert written.describe().splitlines()[:2] == [
        "6 footprints on 5 nets",
        "  R1: R1, 4.7 kΩ, Resistor_SMD:R_0603_1608Metric",
    ]


def test_the_netlist_is_kicads_s_expression_format() -> None:
    export = _Expression.parsed(kicad.netlist(circuit(bench()), PARTS, "X1").text())
    assert export.keyword == "export"
    assert [each.keyword for each in export.lists()] == [
        "version",
        "components",
        "nets",
    ]
    assert export.field("version") == "E"
    (components,) = export.lists("components")
    placed = {each.field("ref"): each for each in components.lists("comp")}
    assert set(placed) == {"R1", "R2", "C1", "Q1", "D1", "U1"}
    power = placed["U1"]
    assert power.field("value") == "IRF511"
    assert power.field("footprint") == "Package_TO_SOT_THT:TO-220-3_Vertical"
    fields = {
        each.field("name"): each.field("value") for each in power.lists("property")
    }
    assert fields == {"MPN": "IRF511", "Instance": "Q2"}
    (sheet,) = power.lists("sheetpath")
    assert (sheet.field("names"), sheet.field("tstamps")) == ("/", "/")
    (nets,) = export.lists("nets")
    assert [each.field("code") for each in nets.lists("net")] == [
        "1",
        "2",
        "3",
        "4",
        "5",
    ]
    base = next(each for each in nets.lists("net") if each.field("name") == "base")
    assert {
        (node.field("ref"), node.field("pin"), node.field("pinfunction"))
        for node in base.lists("node")
    } == {("R2", "2", "b"), ("Q1", "2", "base")}


def test_a_footprint_keeps_its_identifier_from_one_netlist_to_the_next() -> None:
    built = circuit(bench())
    first, again = (kicad.netlist(built, PARTS, "X1").text() for _ in range(2))
    assert first == again
    stamps = re.findall(r'\(tstamps "([0-9a-f-]{36})"\)', first)
    assert len(stamps) == len(set(stamps)) == 6


@component
def pair(c: Builder, bulk: str) -> None:
    """A MOSFET whose source and bulk share a pin, its bulk on the net `bulk`."""
    ground = c.port("ground", Role.GROUND)
    gate = c.port("gate", Role.INPUT)
    drain = c.port("drain")
    transistor = c.place(
        devices.mosfet(width=1 * m, length=1 * m).numbered("2N7000"), name="M1"
    )
    c.connect(transistor.drain, drain)
    c.connect(transistor.gate, gate)
    c.connect(transistor.source, ground)
    if bulk == "ground":
        c.connect(transistor.bulk, ground)
    else:
        c.connect(transistor.bulk, gate)


def test_terminals_on_one_pin_share_its_pad_and_its_net() -> None:
    written = kicad.netlist(circuit(pair("ground")), PARTS)
    assert _pads(written)["ground"] == {("Q1", "1", "source/bulk")}
    with pytest.raises(NetlistError) as raised:
        kicad.netlist(circuit(pair("gate")), PARTS)
    assert raised.value.problems == (
        "M1: its part's pins put pad 1 on ground and on gate",
    )


@component
def flawed(c: Builder) -> None:
    """Every way an instance can stay off a board."""
    ground = c.port("ground", Role.GROUND)
    node = c.port("node")
    ideal = c.place(devices.resistor(1 * kohm), name="R1")
    switch = c.place(parts.q2n3904(), name="Q1")
    clamp = c.place(parts.d1n4148(), name="D1")
    power = c.place(parts.irf511(heated=True), name="Q2")
    sink = c.place(devices.thermal_resistance(50 * K / W), name="RT")
    c.connect(node, ideal.a, switch.collector, switch.base, clamp.anode, power.drain)
    c.connect(ideal.b, switch.emitter, clamp.cathode, power.gate, power.source, ground)
    c.connect(power.thermal, sink.a)
    c.connect(sink.b, ground)


def test_every_problem_keeping_an_instance_off_the_board_is_named() -> None:
    flawless = {part.number: part for part in PARTS.parts}
    catalog = Catalog(
        currency="USD",
        parts=(
            flawless["2N3904"].model_copy(update={"footprint": None, "pins": None}),
            flawless["1N4148"].model_copy(update={"pins": {"gate": ("1",)}}),
            flawless["IRF511"].model_copy(
                update={"pins": {**(flawless["IRF511"].pins or {}), "thermal": ("4",)}}
            ),
        ),
    )
    with pytest.raises(NetlistError) as raised:
        kicad.netlist(circuit(flawed()), catalog)
    assert raised.value.problems == (
        "R1: its resistance has no tolerance, which a part must meet",
        "Q1: its part states no footprint: 2N3904, in TO-92",
        "Q1: its part states no pins: 2N3904, in TO-92",
        "D1: its part's pins name gate, which it does not have",
        "D1: its part's pins leave out anode",
        "D1: its part's pins leave out cathode",
        "Q2: its part's pins put thermal, on a thermal net, on pads",
    )
    assert str(raised.value).startswith("cannot write the netlist:\n  R1: its")


@component
def floating(c: Builder) -> None:
    """A resistor between two ports, neither the ground."""
    resistor = c.place(devices.resistor(4.7 * kohm, tolerance=0.01), name="R1")
    c.connect(c.port("a"), resistor.a)
    c.connect(c.port("b"), resistor.b)


def test_a_component_is_written_from_a_ground_port_and_by_its_path() -> None:
    @component
    def holder(c: Builder) -> None:
        ground = c.port("0", Role.GROUND)
        placed = c.place(floating(), name="X1")
        c.connect(placed.a, placed.b, ground)

    built = circuit(holder())
    with pytest.raises(NetlistError, match="has no ground port"):
        kicad.netlist(built, PARTS, "X1")
    with pytest.raises(JoltyError, match="no component instance is at 'X1/R1'"):
        kicad.netlist(built, PARTS, "X1/R1")
    with pytest.raises(JoltyError, match="no component instance is at 'X9'"):
        kicad.netlist(built, PARTS, "X9")


def test_a_part_states_its_footprint_and_its_pins() -> None:
    with pytest.raises(ValueError, match="should match pattern"):
        Part(number="2N3904", price=0.1, footprint="TO-92_Inline")
    with pytest.raises(ValueError, match="give collector a pad twice"):
        Part(number="2N3904", price=0.1, pins={"collector": ("3", "3")})
    with pytest.raises(ValueError, match="at least 1 character"):
        Part(number="2N3904", price=0.1, pins={"collector": ("",)})
    padless = Part(number="LDR1", price=0.1, pins={"a": ("1",), "light": ()})
    assert padless.pins == {"a": ("1",), "light": ()}
    assert json.loads(PARTS.parts[3].model_dump_json())["pins"] == {
        "emitter": ["1"],
        "base": ["2"],
        "collector": ["3"],
    }


def test_the_kicad_command_writes_the_netlist_beside_the_circuit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    saved, catalog = tmp_path / "bench.jolty.json", tmp_path / "parts.json"
    save_circuit(saved, circuit(bench()))
    PARTS.save(catalog)
    assert Catalog.load(catalog) == PARTS
    arguments = ["export-kicad", str(saved), "--catalog", str(catalog)]
    assert _run(*arguments, "--instance", "X1") == 0
    written = tmp_path / "bench.X1.net"
    assert capsys.readouterr().out.startswith(
        f"wrote {written}: 6 footprints on 5 nets\n  R1: R1, 4.7 kΩ,"
    )
    assert written.read_text(encoding="utf-8") == (
        kicad.netlist(circuit(bench()), PARTS, "X1").text()
    )
    assert _run(*arguments) == 1
    assert capsys.readouterr().out == (
        "cannot write the netlist:\n"
        "  V1: a voltage_source is priced by its part number alone, and it states "
        "none\n"
    )
    assert not (tmp_path / "bench.net").exists()
    assert _run(*arguments, "--instance", "V1") == 2
    assert "no component instance is at 'V1'" in capsys.readouterr().err


FOOTPRINTS = next(
    (
        Path(directory)
        for directory in (
            *(
                os.environ.get(f"KICAD{version}_FOOTPRINT_DIR", "")
                for version in (9, 8, 7)
            ),
            "/usr/share/kicad/footprints",
            "/usr/local/share/kicad/footprints",
        )
        if directory and Path(directory).is_dir()
    ),
    None,
)
"""The directory of KiCad's footprint libraries, if KiCad is installed."""


@pytest.mark.skipif(FOOTPRINTS is None, reason="KiCad's footprints are not installed")
def test_every_footprint_and_pad_is_kicads() -> None:
    assert FOOTPRINTS is not None
    for part in PARTS.parts:
        assert part.footprint is not None
        library, name = part.footprint.split(":")
        path = FOOTPRINTS / f"{library}.pretty" / f"{name}.kicad_mod"
        assert path.is_file(), part.footprint
        pads = set(re.findall(r'\(pad\s+"?([^"\s)]*)"?', path.read_text()))
        pins = part.pins if part.pins is not None else kicad.TWO_PADS
        named = {pad for each in pins.values() for pad in each}
        assert named <= pads, (part.footprint, named - pads)


@dataclass(frozen=True, slots=True)
class _Expression:
    """An S-expression read back: its keyword, then its items, strings unquoted."""

    keyword: str
    items: tuple[str | _Expression, ...]

    @staticmethod
    def parsed(text: str) -> _Expression:
        """The one S-expression `text` holds, every list begun by a bare keyword."""
        pieces = re.findall(r'\(|\)|"(?:[^"\\]|\\.)*"|[^\s()"]+', text)
        opened: list[list[str | _Expression]] = []
        found: list[_Expression] = []
        for piece in pieces:
            if piece == "(":
                opened.append([])
            elif piece == ")":
                keyword, *items = opened.pop()
                assert isinstance(keyword, str)
                (opened[-1] if opened else found).append(
                    _Expression(keyword, tuple(items))
                )
            else:
                opened[-1].append(json.loads(piece) if piece[0] == '"' else piece)
        (expression,) = found
        return expression

    def lists(self, keyword: str | None = None) -> list[_Expression]:
        """Its items that are lists, those `keyword` begins if given."""
        return [
            item
            for item in self.items
            if isinstance(item, _Expression) and keyword in (None, item.keyword)
        ]

    def field(self, keyword: str) -> str:
        """The string of its one item that `keyword` begins."""
        (found,) = self.lists(keyword)
        (text,) = found.items
        assert isinstance(text, str)
        return text


def _run(*arguments: str) -> int:
    try:
        main(arguments)
    except SystemExit as exit_:
        return exit_.code if isinstance(exit_.code, int) else 0
    return 0
