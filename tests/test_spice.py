# Copyright 2026 Maxence Faldor

import hashlib
from pathlib import Path

import pytest

from jolty import (
    Builder,
    Component,
    DownloadError,
    JoltyError,
    OperatingPoint,
    Role,
    Simulation,
    SpiceError,
    circuit,
    component,
    devices,
    ir,
    spice,
)
from jolty.spice.export import netlist
from jolty.spice.expressions import evaluate, number, parse, translate
from jolty.spice.netlist import statements, tokens
from jolty.units import K, V, celsius, kohm


def library(text: str) -> spice.Library:
    read = spice.Library()
    read.read(text, "parts.lib")
    return read


def driven(part: Component, level: float, pins: tuple[str, str]) -> Component:
    """`part` driven at its first pin by `level` volts, its second pin `out`."""
    c = Builder("bench")
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(level * V), id="V1")
    placed = c.place(part, id="U1")
    c.connect(source.positive, placed[pins[0]], name="in")
    c.connect(placed[pins[1]], name="out")
    joined = [source.negative, ground]
    if spice.GROUND in (port.name for port in part.definition.ports):
        joined.append(placed[spice.GROUND])
    c.connect(*joined)
    return c.finish()


def output(
    part: Component, level: float, pins: tuple[str, str] = ("in", "out")
) -> float:
    point = Simulation(driven(part, level, pins)).run(OperatingPoint())
    return point.voltage("out")


def test_statements_join_continuations_and_drop_comments() -> None:
    text = "* title\nR1 a b 1k ; load\n+ TC=1m $ ngspice's comment\n\nC1 a 0 1n"
    found = statements(text, "f.cir")
    assert [(card.line, card.text) for card in found] == [
        (2, "R1 a b 1k TC=1m"),
        (5, "C1 a 0 1n"),
    ]


def test_a_continuation_must_continue_a_statement() -> None:
    with pytest.raises(SpiceError, match=r"f\.cir:1: a continuation line"):
        statements("+ R1 a b 1k", "f.cir")


def test_tokens_keep_groups_whole() -> None:
    assert tokens("E1 out 0 VALUE = {LIMIT(V(a), -1, 1)} 'x y'") == [
        *("E1", "out", "0", "VALUE", "="),
        *("{LIMIT(V(a), -1, 1)}", "'x y'"),
    ]
    assert tokens("XU1 a,b POLY(2) (1, 2)") == ["XU1", "a", "b", "POLY(2)", "(1, 2)"]


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("10", 10.0),
        ("4.7k", 4.7e3),
        ("1MEG", 1e6),
        ("2.2uF", 2.2e-6),
        ("1mil", 25.4e-6),
        ("3m", 3e-3),
        ("1e3k", 1e6),
        (".5p", 0.5e-12),
    ],
)
def test_numbers_take_scale_suffixes(text: str, value: float) -> None:
    assert number(text) == pytest.approx(value)


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("1 + 2 * 3", 7.0),
        ("-2 ** 2", -4.0),
        ("2 ^ 3 ^ 2", 512.0),
        ("PWR(-2, 2)", 4.0),
        ("PWRS(-2, 2)", -4.0),
        ("LIMIT(5, 2, 1)", 2.0),
        ("1 < 2 & 3 > 2 ? 10 : 20", 10.0),
        ("IF(0, 1, 2) + INT(-2.5)", 0.0),
        ("TABLE(1.5, 1, 10, 2, 20)", 15.0),
        ("TABLE(5, 1, 10, 2, 20)", 50.0),
        ("{2 * Gain}", 6.0),
    ],
)
def test_expressions_mean_what_pspice_and_ngspice_mean(text: str, value: float) -> None:
    assert evaluate(parse(text), {"gain": 3.0}) == pytest.approx(value)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("1 +", "expected an operand"),
        ("(1", r"expected `\)`"),
        ("1 2", "expected an operator"),
        ("FOO(1)", "no function is named `foo`"),
        ("LIMIT(1, 2)", "`limit` takes 3 arguments, not 2"),
        ("x + 1", "no parameter is named `x`"),
        ("V(a)", "reads no voltage or current, not `a`"),
    ],
)
def test_expressions_say_what_is_wrong(text: str, message: str) -> None:
    with pytest.raises(SpiceError, match=message):
        evaluate(parse(text), {})


def test_translations_name_nodes_and_sources_as_told() -> None:
    expression = parse("V(a, 0) * I(Vsense) - V(0, b)")
    written = translate(expression, {}, lambda node: f"n_{node}", str.lower)
    assert written == "((v(n_a) * i(vsense)) - (-v(n_b)))"


def test_a_subcircuit_becomes_a_component_with_its_parameters() -> None:
    parts = library(
        ".param scale = 2\n"
        ".subckt divider in out params: top = {scale * 1k}\n"
        "R1 in out {top}\n"
        "R2 out 0 1k\n"
        ".ends"
    )
    part = parts.component("Divider")
    assert [port.name for port in part.definition.ports] == ["in", "out", "ground"]
    assert output(part, 3.0) == pytest.approx(1.0)
    assert output(parts.component("divider", top=1e3), 3.0) == pytest.approx(1.5)


def test_subcircuits_place_one_another() -> None:
    parts = library(
        ".subckt half a b\nR1 a b 1k\nR2 b 0 1k\n.ends\n"
        ".subckt quarter in out\nX1 in mid half\nX2 mid out half\n.ends"
    )
    # The second divider loads the first: 4 V across 1k + (1k ∥ 2k).
    assert output(parts.component("quarter"), 5.0) == pytest.approx(1.0)


def test_controlled_sources_read_polynomials_values_and_currents() -> None:
    parts = library(
        ".subckt gains in out\n"
        "Vsense in sensed 0\n"
        "R1 sensed 0 1k\n"
        "E1 a 0 POLY(1) in 0 1 2\n"
        "F1 0 b Vsense 1k\n"
        "Rb b 0 1\n"
        "G1 0 c VALUE = {V(a) * V(b)}\n"
        "Rc c 0 1\n"
        "H1 out 0 POLY(1) Vsense 0 1k\n"
        "Rout out 0 1k\n"
        ".ends"
    )
    part = parts.component("gains")
    point = Simulation(driven(part, 2.0, ("in", "out"))).run(OperatingPoint())
    assert point.voltage("U1/a") == pytest.approx(5.0)
    assert point.voltage("U1/b") == pytest.approx(2.0)
    assert point.voltage("U1/c") == pytest.approx(10.0)
    assert point.voltage("out") == pytest.approx(2.0)


def test_a_table_source_holds_beyond_its_ends() -> None:
    parts = library(".subckt clamp in out\nE1 out 0 TABLE {V(in)} = (0,0) (1,2)\n.ends")
    part = parts.component("clamp")
    assert output(part, 0.5) == pytest.approx(1.0)
    assert output(part, 3.0) == pytest.approx(2.0)


def test_models_inherit_through_ako_and_scale_by_area() -> None:
    parts = library(
        ".model base D(IS=1e-14 RS=10)\n"
        ".model large AKO:base D(N=1.5)\n"
        ".subckt pair in out\nD1 in out large 2\nR1 out 0 1k\n.ends"
    )
    (instance,) = parts.component("pair").definition.instances[:1]
    assert instance.parameters == {
        "saturation_current": 2e-14,
        "series_resistance": 5.0,
        "emission_coefficient": 1.5,
    }


def test_bipolar_transistors_have_spice_s_substrate() -> None:
    parts = library(
        ".model QN NPN(BF=50 CJS=1p TF=0.3n PTF=40)\n.model QL PNP(SUBS=1)\n"
        ".subckt pair c b e s\nQ1 c b e QN\nQ2 c b e s QL 2\n.ends"
    )
    first, second = parts.component("pair").definition.instances
    assert first.terminals["substrate"] == spice.GROUND
    assert first.parameters["substrate"] == "vertical"
    assert first.parameters["excess_phase"] == 40.0
    assert second.terminals["substrate"] == "s"
    assert second.parameters["substrate"] == "vertical"
    lateral = library(".model QP PNP\n.subckt one c b e\nQ1 c b e QP\n.ends")
    (transistor,) = lateral.component("one").definition.instances
    assert transistor.parameters["substrate"] == "lateral"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (".subckt a x\nR1 x 0 1k", r"parts\.lib:1: the subcircuit never ends"),
        (".subckt a x\n.subckt b y\n.ends", "a subcircuit defined within another"),
        (".subckt a x\nZ1 x 0 1\n.ends", "no element of letter `Z`"),
        (".subckt a x\nD1 x 0 DX\n.ends", "no model is named `DX`"),
        (".subckt a 0\nR1 0 0 1\n.ends", "a pin named `0`"),
        (".model D1 D(IKF=5)\n.subckt a x\nD1 x 0 D1\n.ends", "counterpart to `IKF`"),
        (".model M1 NMOS(LEVEL=2)\n.subckt a x\nM1 x x 0 0 M1\n.ends", "level 2"),
        (
            ".model RX RES(T_REL_LOCAL=5)\n.subckt a x\nR1 x 0 RX 1k\n.ends",
            "no `T_REL_LOCAL`",
        ),
        (
            ".model RX RES(T_ABS=50)\n.subckt a x\nR1 x 0 RX 1k DTEMP=5\n.ends",
            "sets its own temperature twice: `T_ABS`, `DTEMP`",
        ),
        (".subckt a x\nR1 x 0 1k TC=1m,2u,3n\n.ends", "two coefficients at most"),
        (".subckt a x\nF1 x 0 Vmissing 2\n.ends", "`Vmissing` is no source in `a`"),
        (".subckt a x\nE1 x 0 POLY(2) x 0 1\n.ends", r"POLY\(2\) reads 2 controls"),
    ],
)
def test_what_jolty_cannot_read_is_an_error_where_it_is(
    text: str, message: str
) -> None:
    with pytest.raises(SpiceError, match=message):
        library(text).component("a")


def test_mutual_inductances_couple_their_inductors_into_one_device() -> None:
    parts = library(
        ".subckt a p s t\nR1 p 0 1\nL1 p 0 1m\nLS s 0 4m TC1=1m\nK1 L1 LS 0.99\n"
        "L3 0 t 250u\nK2 l3 ls -0.5\nL4 t 0 1u\n.ends"
    )
    resistor, coupled, alone = parts.component("a").definition.instances
    assert (resistor.id, coupled.id, alone.id) == ("R1", "K1", "L4")
    assert coupled.of == ir.DeviceOf(device="coupled_inductors")
    assert coupled.parameters == {
        "windings": [
            {"inductance": 1e-3},
            {"inductance": 4e-3, "temperature_coefficient": 1e-3},
            {"inductance": 2.5e-4},
        ],
        "couplings": [
            {"windings": [1, 2], "coefficient": 0.99},
            {"windings": [2, 3], "coefficient": -0.5},
        ],
    }
    assert coupled.terminals == {
        "a1": "p",
        "b1": spice.GROUND,
        "a2": "s",
        "b2": spice.GROUND,
        "a3": spice.GROUND,
        "b3": "t",
    }
    # PSpice couples every pair a mutual inductance names.
    three = library(".subckt a x\nL1 x 0 1m\nL2 x 0 1m\nL3 x 0 1m\nK L1 L2 L3 1\n.ends")
    (coupled,) = three.component("a").definition.instances
    assert ir.parameter(coupled.parameters, "couplings.2.windings") == [2, 3]


def test_coupled_inductors_run_at_their_windings_shared_temperature() -> None:
    shared = library(
        ".subckt a x\nL1 x 0 1m DTEMP=5\nL2 x 0 2m DTEMP=5\nK1 L1 L2 0.5\n.ends"
    )
    (coupled,) = shared.component("a").definition.instances
    assert coupled.parameters["temperature_offset"] == 5.0
    assert coupled.parameters["windings"] == [
        {"inductance": 1e-3},
        {"inductance": 2e-3},
    ]
    apart = library(".subckt a x\nL1 x 0 1m DTEMP=5\nL2 x 0 2m\nK1 L1 L2 0.5\n.ends")
    with pytest.raises(SpiceError, match="run at different temperatures of their own"):
        apart.component("a")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("K1 L1 L9 0.5", "`L9` is no inductor in `a`"),
        ("K1 L1 L2 0.5\nK2 L2 L1 0.4", "`L2` and `L1` are coupled twice"),
        ("K1 L1 L1 0.5", "`L1` is coupled with itself"),
        ("K1 L1 L2 0.5 CORE", "jolty has no magnetic core"),
        ("K1 L1 0.5", "couples two inductors or more"),
    ],
)
def test_a_mutual_inductance_couples_inductors_it_names(
    text: str, message: str
) -> None:
    with pytest.raises(SpiceError, match=message):
        library(f".subckt a x\nL1 x 0 1m\nL2 x 0 1m\n{text}\n.ends").component("a")


def test_a_current_controlled_switch_senses_its_source_in_series() -> None:
    parts = library(
        ".model SW ISWITCH(ION=2m IOFF=1m RON=10)\n"
        ".subckt a x y\nV1 x 0 1\nW1 y 0 V1 SW OFF\n.ends"
    )
    source, switch = parts.component("a").definition.instances
    assert switch.of == ir.DeviceOf(device="current_controlled_switch")
    assert switch.parameters == {
        "on_current": 2e-3,
        "off_current": 1e-3,
        "on_resistance": 10.0,
    }
    assert switch.terminals == {
        "positive": "y",
        "negative": spice.GROUND,
        "control_positive": "x",
        "control_negative": "V1.sense1",
    }
    assert source.terminals["positive"] == "V1.sense1"
    with pytest.raises(SpiceError, match="names its source and its model"):
        library(".subckt a x\nV1 x 0 1\nW1 x 0 SW\n.ends").component("a")


def test_parameters_that_change_nothing_are_left_out() -> None:
    parts = library(
        ".model NX NMOS(VTO=1 NLEV=2 GDSNOI=1)\n"
        ".model QUIET RES(KF=0 AF=1)\n"
        ".subckt a x\nM1 x x 0 0 NX\nR1 x 0 QUIET 1k\n.ends"
    )
    transistor, resistor = parts.component("a").definition.instances
    assert transistor.parameters == {"channel": "n", "threshold_voltage": 1.0}
    assert resistor.parameters == {"resistance": 1e3}


def test_flicker_noise_parameters_carry_over() -> None:
    parts = library(
        ".model DX D(IS=1e-14 KF=1e-16 AF=1.2)\n.subckt a x\nD1 x 0 DX\n.ends"
    )
    (diode,) = parts.component("a").definition.instances
    assert diode.parameters == {
        "saturation_current": 1e-14,
        "flicker_coefficient": 1e-16,
        "flicker_exponent": 1.2,
    }
    loud = library(".model LOUD RES(KF=1e-12)\n.subckt a x\nR1 x 0 LOUD 1k\n.ends")
    (resistor,) = loud.component("a").definition.instances
    assert resistor.parameters == {"resistance": 1e3, "flicker_coefficient": 1e-12}
    with pytest.raises(SpiceError, match="counterpart to `KF`"):
        library(
            ".model LOUD CAP(KF=1e-12)\n.subckt a x\nC1 x 0 LOUD 1n\n.ends"
        ).component("a")


def test_temperature_parameters_come_in_kelvins() -> None:
    parts = library(
        ".model DX D(IS=1e-14 XTI=2 EG=1.2 TNOM=50)\n.model RX RES(TCE=0.5 TNOM=0)\n"
        ".subckt a x\nD1 x 0 DX\nR1 x 0 1k TC=1m,2u\nR2 x 0 RX 1k\n"
        "C1 x 0 1n TC1=3m\n.ends"
    )
    diode, first, second, capacitor = parts.component("a").definition.instances
    assert diode.parameters == {
        "saturation_current": 1e-14,
        "saturation_current_temperature_exponent": 2.0,
        "energy_gap": 1.2,
        "nominal_temperature": pytest.approx(323.15),
    }
    assert first.parameters == {
        "resistance": 1e3,
        "temperature_coefficient": 1e-3,
        "quadratic_temperature_coefficient": 2e-6,
    }
    assert second.parameters == {
        "resistance": 1e3,
        "exponential_temperature_coefficient": 0.5,
        "nominal_temperature": 273.15,
    }
    assert capacitor.parameters == {
        "capacitance": 1e-9,
        "temperature_coefficient": 3e-3,
    }


def test_devices_take_their_own_temperatures() -> None:
    parts = library(
        ".model QUIET RES(T_ABS=-273.15)\n.model WARM D(T_REL_GLOBAL=15)\n"
        ".model DX D\n.model NX NMOS(VTO=1)\n"
        ".subckt a x\nR1 x 0 QUIET 1k\nD1 x 0 WARM\nD2 x 0 DX TEMP=80\n"
        "C1 x 0 1n DTEMP=-10\nM1 x x 0 0 NX W=2u L=1u TEMP=50\n.ends"
    )
    quiet, warm, hot, capacitor, transistor = parts.component("a").definition.instances
    assert quiet.parameters == {"resistance": 1e3, "temperature": 0.0}
    assert warm.parameters == {"temperature_offset": 15.0}
    assert hot.parameters == {"temperature": pytest.approx(353.15)}
    assert capacitor.parameters == {"capacitance": 1e-9, "temperature_offset": -10.0}
    assert transistor.parameters["temperature"] == pytest.approx(323.15)
    assert transistor.parameters["width"] == 2e-6


def test_a_netlist_writes_each_devices_own_temperature() -> None:
    @component
    def apart(c: Builder, switch: devices.Device) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(1 * V).warmer_by(5 * K), id="V1")
        load = c.place(devices.resistor(1 * kohm).held_at(celsius(85)), id="R1")
        diode = c.place(devices.diode().warmer_by(10 * K), id="D1")
        placed = c.place(switch, id="S1")
        c.connect(source.positive, load.a, placed.control_positive, name="in")
        c.connect(load.b, diode.anode, placed.positive, name="out")
        c.connect(
            source.negative,
            diode.cathode,
            placed.negative,
            placed.control_negative,
            ground,
        )

    lines = netlist(circuit(apart(devices.voltage_controlled_switch())))
    assert lines[1] == "VV1 in 0 DC 1.0"
    assert lines[2].startswith("RR1 in out 1000.0 temp=85.0")
    # Each where ngspice's thermal voltage is jolty's.
    assert lines[4].startswith("DD1 out 0 MD1 dtemp=10.00000")
    with pytest.raises(
        JoltyError, match="S1 is a voltage_controlled_switch at its own"
    ):
        netlist(circuit(apart(devices.voltage_controlled_switch().held_at(0 * K))))


def test_resistors_and_capacitors_read_voltage_coefficients() -> None:
    parts = library(
        ".model CX CAP(VC1=-1m)\n"
        ".subckt a x\nR1 x 0 1k VC1=2u VC2=3n\nC1 x 0 CX 1n VC2=4u\n.ends"
    )
    resistor, capacitor = parts.component("a").definition.instances
    assert resistor.parameters == pytest.approx(
        {
            "resistance": 1e3,
            "voltage_coefficient": 2e-6,
            "quadratic_voltage_coefficient": 3e-9,
        }
    )
    assert capacitor.parameters == pytest.approx(
        {
            "capacitance": 1e-9,
            "voltage_coefficient": -1e-3,
            "quadratic_voltage_coefficient": 4e-6,
        }
    )
    with pytest.raises(SpiceError, match="jolty reads no `VC1` here"):
        library(".subckt a x\nL1 x 0 1u VC1=1m\n.ends").component("a")


def test_a_download_is_checked_and_kept(tmp_path: Path) -> None:
    published = tmp_path / "published" / "part.lib"
    published.parent.mkdir()
    published.write_text(".subckt part a\n.ends\n")
    digest = hashlib.sha256(published.read_bytes()).hexdigest()
    cache = tmp_path / "cache"
    kept = spice.fetch(published.as_uri(), digest, cache)
    assert kept.read_text() == published.read_text()
    published.unlink()
    assert spice.fetch(published.as_uri(), digest.upper(), cache) == kept


def test_a_download_must_match_its_digest(tmp_path: Path) -> None:
    published = tmp_path / "part.lib"
    published.write_text("changed")
    with pytest.raises(DownloadError, match="has SHA-256 digest"):
        spice.fetch(published.as_uri(), "0" * 64, tmp_path / "cache")
    with pytest.raises(DownloadError, match="cannot download"):
        spice.fetch((tmp_path / "missing.lib").as_uri(), "0" * 64, tmp_path)


def test_downloads_are_kept_in_the_configured_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published = tmp_path / "part.lib"
    published.write_text("part")
    digest = hashlib.sha256(b"part").hexdigest()
    monkeypatch.setenv("JOLTY_CACHE", str(tmp_path / "configured"))
    kept = spice.fetch(published.as_uri(), digest)
    assert kept.is_relative_to(tmp_path / "configured")
