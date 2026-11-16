# Copyright 2026 Maxence Faldor

import hashlib
import json
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from jolty import (
    Builder,
    Component,
    DimensionError,
    DownloadError,
    JoltyError,
    OperatingPoint,
    Role,
    Simulation,
    SpiceError,
    _core,
    circuit,
    component,
    devices,
    ir,
    spice,
)
from jolty.spice import sky130
from jolty.spice.export import netlist
from jolty.spice.expressions import evaluate, number, parse, translate
from jolty.spice.netlist import statements, tokens
from jolty.units import (
    VOLTAGE,
    A,
    C,
    H,
    J,
    K,
    V,
    W,
    celsius,
    kohm,
    m,
    mA,
    nm,
    ns,
    ohm,
    s,
    uF,
    um,
)
from jolty.waveforms import Exponential, Piecewise, Pulse, Sine


def library(text: str) -> spice.Library:
    read = spice.Library()
    read.read(text, "parts.lib")
    return read


def driven(part: Component, level: float, pins: tuple[str, str]) -> Component:
    """`part` driven at its first pin by `level` volts, its second pin `out`."""
    c = Builder("bench")
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(level * V), name="V1")
    placed = c.place(part, name="U1")
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
        ("LIMIT(5, 2) + AGAUSS(1, 0.1, 3) + AUNIF(2, 1)", 8.0),
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
        ("LIMIT(1)", "`limit` takes 3 arguments, not 1"),
        ("AGAUSS(1, 2)", "`agauss` takes 3 arguments, not 2"),
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


def waveforms(text: str) -> dict[str, JsonValue]:
    """The waveform of each source of the one subcircuit `text` defines, by its id."""
    read = library(text)
    (name,) = read.subcircuits
    return {
        instance.name: instance.parameters["waveform"]
        for instance in read.component(name).definition.instances
        if instance.name[0] in "VI"
    }


def test_sources_read_their_waveforms_over_time() -> None:
    read = waveforms(
        ".subckt drive a b\n"
        "V1 a 0 PULSE(0 5 1n 2n 3n 4n 20n)\n"
        "V2 b 0 dc 1 pwl(0, 1, 1u, 2.5)\n"
        "I1 a b sin (1m 2m 1meg 5n 1e6 90)\n"
        "I2 a b DC 1m AC 1 SIN(1m 1m 1k)\n"
        "V3 c 0 pulse -1 1 0 1n 1n 1n\n"
        "R1 c 0 1k\n"
        "V4 d 0 EXP(0 4.5 1n 2n 600n 3n)\n"
        "R2 d 0 1k\n"
        ".ends"
    )
    pulse: dict[str, float | None] = {"initial": 0.0, "pulsed": 5.0, "delay": 1e-9}
    # SPICE2 wrote a waveform's arguments without parentheses, to the end: V3's.
    pulse |= {"rise": 2e-9, "fall": number("3n"), "width": 4e-9, "period": 2e-8}
    once = {"initial": -1.0, "pulsed": 1.0, "delay": 0.0, "rise": 1e-9}
    once |= {"fall": 1e-9, "width": 1e-9}
    sine = {"offset": 1e-3, "amplitude": 2e-3, "frequency": 1e6, "delay": 5e-9}
    sine |= {"damping": 1e6, "phase": math.pi / 2}
    # Its DC level is its level at zero, the sine's offset at no phase.
    level = {"offset": 1e-3, "amplitude": 1e-3, "frequency": 1e3, "delay": 0.0}
    level |= {"damping": 0.0, "phase": 0.0}
    assert read == {
        "V1": {"pulse": pulse},
        "V2": {"piecewise": {"points": [[0.0, 1.0], [1e-6, 2.5]]}},
        "I1": {"sine": sine},
        "I2": {"sine": level},
        "V3": {"pulse": once},
        "V4": {
            "exponential": {
                "initial": 0.0,
                "pulsed": 4.5,
                "rise_delay": 1e-9,
                "rise_time_constant": 2e-9,
                "fall_delay": number("600n"),
                "fall_time_constant": number("3n"),
            }
        },
    }


def test_waveforms_read_back_as_the_exporter_writes_them() -> None:
    written = {
        "P": Pulse(
            initial=-1 * V,
            pulsed=2 * V,
            delay=1 * ns,
            rise=2 * ns,
            fall=3 * ns,
            width=4 * ns,
            period=20 * ns,
        ),
        "S": Sine(
            offset=0.5 * V,
            amplitude=1 * V,
            frequency=1e6 / s,
            delay=1 * ns,
            damping=1e3 / s,
            phase=math.pi / 2,
        ),
        "W": Piecewise([(0 * s, 0 * V), (1 * ns, 3.3 * V), (2 * ns, 1 * V)]),
        "E": Exponential(
            initial=0.5 * V,
            pulsed=2 * V,
            rise_delay=1 * ns,
            rise_time_constant=2 * ns,
            fall_delay=10 * ns,
            fall_time_constant=3 * ns,
        ),
    }

    @component
    def sources(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        for name, waveform in written.items():
            placed = c.place(devices.voltage_source(waveform), name=name)
            load = c.place(devices.resistor(1 * kohm), name=f"R{name}")
            c.connect(placed.positive, load.a, name=f"n{name}")
            c.connect(placed.negative, load.b, ground)

    lines = netlist(circuit(sources()), transient=True)[1:]
    read = waveforms(".subckt sources\n" + "\n".join(lines) + "\n.ends")
    assert read == {
        f"V{name}": waveform.to_json(VOLTAGE) for name, waveform in written.items()
    }


@pytest.mark.parametrize(
    ("element", "message"),
    [
        ("V1 a 0 PULSE(0 5)", "a PULSE without a positive rise, fall and width"),
        ("V1 a 0 PULSE(0 5 0 0 1n 1n)", "a PULSE without a positive rise"),
        ("V1 a 0 SIN(0 1)", "a SIN without a positive frequency"),
        ("V1 a 0 EXP(0 1 0 1n 1u 1n)", "an EXP without a positive rise delay"),
        ("V1 a 0 EXP(0 1 1n 1n)", "an EXP without a positive rise delay"),
        ("V1 a 0 SFFM(0 1 1k 2 1meg)", "jolty has no `SFFM` waveform"),
        ("V1 a 0 PWL(0 1 1n)", "jolty has no `PWL` waveform"),
        ("V1 a 0 DC 1 PULSE(0 5 0 1n 1n 1n)", "a DC level of 1.0 beside a waveform"),
        ("V1 a 0 PULSE(0 5 0 1n 1n 1n) 0 1", "PULSE, SIN, PWL or EXP waveform only"),
    ],
)
def test_sources_refuse_what_their_waveforms_leave_out(
    element: str, message: str
) -> None:
    with pytest.raises(SpiceError, match=re.escape(message)):
        library(f".subckt drive a\n{element}\nR1 a 0 1k\n.ends").component("drive")


def test_an_area_scales_the_defaults_a_card_leaves_out() -> None:
    parts = library(
        ".model q npn bf=100\n.model d d\n.model j njf\n"
        ".subckt parts c b e\n"
        "Q1 c b e q area=10\nQ2 c b e q 2\nQ3 c b e q\nD1 c e d 3\nJ1 c b e j area=4\n"
        ".ends"
    )
    read = {
        instance.name: instance.parameters
        for instance in parts.component("parts").definition.instances
    }
    # ngspice multiplies the saturation current, 1e-16 A unless the card gives one,
    # by the area (bjttemp.c:168), and a diode's 1e-14 A, a JFET's 1e-14 A and its
    # transconductance parameter, 1e-4 A/V², alike.
    assert read["Q1"]["saturation_current"] == pytest.approx(1e-15)
    assert read["Q2"]["saturation_current"] == pytest.approx(2e-16)
    assert "saturation_current" not in read["Q3"]
    assert read["D1"]["saturation_current"] == pytest.approx(3e-14)
    assert read["J1"]["saturation_current"] == pytest.approx(4e-14)
    assert read["J1"]["transconductance_parameter"] == pytest.approx(4e-4)


@pytest.mark.parametrize("name", ["vaf", "var", "ikf", "ikr", "irb", "vtf"])
def test_a_bipolar_parameter_infinite_at_zero_reads_as_infinite(name: str) -> None:
    # ngspice takes each as infinite where it is 0 (bjttemp.c:51-54 and 83-99,
    # bjtload.c:620), as jolty's default is: the card reads as one without it.
    card = f".model q npn bf=100 rb=10 tf=1n xtf=1 {name}=0\n"
    parts = library(card + ".subckt one c b\nQ1 c b 0 q\nV1 c 0 5\nV2 b 0 0.7\n.ends")
    bare = library(card.replace(f"{name}=0", "") + ".subckt one c b\nQ1 c b 0 q\n.ends")
    read = parts.component("one").definition.instances[0].parameters
    assert read == bare.component("one").definition.instances[0].parameters
    point = Simulation(parts.component("one")).run(OperatingPoint())
    assert point.current("V1") < 0


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
        (
            ".model P VDMOS\n.subckt a d g s t c\nM1 d g s t c P thermal\n.ends",
            "no `THERMAL` VDMOS",
        ),
        (".model P VDMOS(PHI=0)\n.subckt a x\nM1 x x 0 P\n.ends", "`PHI`"),
        (".model P VDMOS(VP=1)\n.subckt a x\nM1 x x 0 P\n.ends", "counterpart to `VP`"),
        (".model P VDMOS\n.subckt a x\nM1 x x 0 P 2\n.ends", "a VDMOS takes no `2`"),
        (
            ".model P VDMOS\n.subckt a x\nM1 x x 0 x x x P\n.ends",
            "a VDMOS joins at most 5 nodes",
        ),
        (".model P VDMOS\n.subckt a x\nM1 x x 0 P W=1u\n.ends", "counterpart to `W`"),
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
    assert (resistor.name, coupled.name, alone.name) == ("R1", "K1", "L4")
    assert coupled.of == ir.DeviceOf(model="coupled_inductors")
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


def test_transmission_lines_read_their_delay_as_ngspice_does() -> None:
    parts = library(
        ".subckt a p q r\nT1 p 0 q 0 Z0=50 TD=1n IC=1,0,1,0\n"
        "T2 q 0 r 0 ZO=75 F=100meg NL=0.5\nT3 r 0 0 0 Z0=50\n.ends"
    )
    timed, normalized, defaulted = parts.component("a").definition.instances
    assert timed.of == ir.DeviceOf(model="transmission_line")
    assert timed.parameters == {"characteristic_impedance": 50.0, "delay": 1e-9}
    assert timed.terminals == {
        "positive1": "p",
        "negative1": spice.GROUND,
        "positive2": "q",
        "negative2": spice.GROUND,
    }
    # A half wavelength at 100 MHz, and ngspice's quarter wavelength at 1 GHz.
    assert normalized.parameters["delay"] == pytest.approx(5e-9, rel=1e-15)
    assert defaulted.parameters["delay"] == pytest.approx(0.25e-9, rel=1e-15)

    @component
    def matched(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(1 * V), name="V1")
        line = c.place(devices.transmission_line(50 * ohm, 1 * ns), name="T1")
        load = c.place(devices.resistor(50 * ohm), name="R1")
        c.connect(source.positive, line.positive1, name="a")
        c.connect(line.positive2, load.a, name="b")
        c.connect(source.negative, line.negative1, line.negative2, load.b, ground)

    written = [line for line in netlist(circuit(matched())) if line.startswith("T")]
    assert written == ["TT1 a 0 b 0 Z0=50.0 TD=1e-09"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("T1 x 0 y 0 TD=1n", "gives its characteristic impedance, `Z0`"),
        ("T1 x 0 y Z0=50 TD=1n", "joins four nodes"),
        ("T1 x 0 y 0 Z0=50 TD=1n LEN=2", "jolty reads no `LEN` here"),
        ("O1 x 0 y 0 LOSSY", "of letter `O`, ngspice's lossy transmission line, LTRA"),
    ],
)
def test_a_transmission_line_is_read_whole_and_a_lossy_one_refused(
    text: str, message: str
) -> None:
    with pytest.raises(SpiceError, match=message):
        library(f".subckt a x y\n{text}\n.ends").component("a")


def test_a_current_controlled_switch_senses_its_source_in_series() -> None:
    parts = library(
        ".model SW ISWITCH(ION=2m IOFF=1m RON=10)\n"
        ".subckt a x y\nV1 x 0 1\nW1 y 0 V1 SW OFF\n.ends"
    )
    source, switch = parts.component("a").definition.instances
    assert switch.of == ir.DeviceOf(model="current_controlled_switch")
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


def test_a_vdmos_card_reads_as_ngspice_reads_it() -> None:
    """Its polarity from `pchan` or its type, its parameters, and three to five nodes.

    ngspice reads past a card's annotations of its part, its `PHI` and its thermal
    network, and an element's fourth and fifth nodes, its thermal ones, where it does
    not heat.
    """
    parts = library(
        ".model PX VDMOS(pchan Rg=3 Vto=-4 mfg=Some-Maker_Inc. Vds=-60 Ron=10m Qg=20n\n"
        "+ PHI=0.7 RTHJC=1 CTHJ=1m KSUBTHRES=120m VTOTC=-2m BEX=-1.2 M=0.4)\n"
        ".model NX VDMOS Nchan Vds=100\n+ Rd=44m RQ=0.5 VQ=20 TCVTH=-3m MU=-1.4\n"
        ".model PY VDMOSP(Vto=-2)\n"
        ".subckt a d g s t\nM1 d g s PX\nM2 d g s s NX m=2\nM3 d g s t s PY off\n.ends"
    )
    first, second, third = parts.component("a").definition.instances
    assert [instance.of for instance in (first, second, third)] == [
        ir.DeviceOf(model="vdmos")
    ] * 3
    assert first.parameters == {
        "gate_resistance": 3.0,
        "threshold_voltage": -4.0,
        "subthreshold_slope": 0.12,
        "threshold_voltage_temperature_coefficient": -2e-3,
        "transconductance_parameter_temperature_exponent": -1.2,
        "grading_coefficient": 0.4,
        "channel": "p",
    }
    assert second.parameters == {
        "drain_resistance": 0.044,
        "quasi_saturation_resistance": 0.5,
        "quasi_saturation_voltage": 20.0,
        "threshold_voltage_temperature_coefficient": -3e-3,
        "transconductance_parameter_temperature_exponent": -1.4,
        "multiplier": 2.0,
        "channel": "n",
    }
    assert third.parameters == {"threshold_voltage": -2.0, "channel": "p"}
    for instance in (first, second, third):
        assert instance.terminals == {"drain": "d", "gate": "g", "source": "s"}


def test_a_vdmos_is_written_for_ngspice_as_its_card_reads_back() -> None:
    @component
    def bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        supply = c.place(devices.voltage_source(-5 * V), name="V1")
        transistor = c.place(
            devices.vdmos(
                channel="p",
                threshold_voltage=-4 * V,
                drain_resistance=0.02 * ohm,
                multiplier=2,
                junction_capacitance=1e-9 * C / V,
            ).warmer_by(10 * K),
            name="M1",
        )
        c.connect(supply.positive, transistor.drain, transistor.gate, name="d")
        c.connect(supply.negative, transistor.source, ground)

    lines = netlist(circuit(bench()))
    card = next(line for line in lines if line.startswith(".model"))
    assert card == (".model MM1 VDMOS(pchan vto=-4.0 rd=0.02 cjo=1e-09)"), card
    element = next(line for line in lines if line.startswith("MM1"))
    assert element.startswith("MM1 d d 0 MM1 m=2 dtemp=")
    read = library(f"{card}\n.subckt a d\nM1 d d 0 MM1 m=2\n.ends")
    (instance,) = read.component("a").definition.instances
    assert instance.parameters == {
        "threshold_voltage": -4.0,
        "drain_resistance": 0.02,
        "junction_capacitance": 1e-9,
        "multiplier": 2.0,
        "channel": "p",
    }


def test_a_vbic_card_reads_as_ngspice_reads_it() -> None:
    """Its levels 4 and 9, within its parentheses or before, and three to five nodes.

    ngspice raises an intrinsic collector resistance below 10 mΩ to 10 mΩ, and reads
    past `VERS` and `VREF`; an element's fifth node, its thermal one, is left unjoined
    where its card does not heat.
    """
    parts = library(
        ".model QN NPN(level=4 rcx=10 rbx=5 is=1e-16 vef=50 tf=10p rci=0\n"
        "+ vers=1.2 vref=0)\n"
        ".model QP PNP Level=9(is=2e-17 nf=1.02)\n"
        ".subckt a c b e s\nQ1 c b e s QN\nQ2 c b e QP\nQ3 c b e s s QN\n.ends"
    )
    first, second, third = parts.component("a").definition.instances
    assert [instance.of for instance in (first, second, third)] == [
        ir.DeviceOf(model="vbic")
    ] * 3
    assert first.parameters == {
        "extrinsic_collector_resistance": 10.0,
        "extrinsic_base_resistance": 5.0,
        "saturation_current": 1e-16,
        "forward_early_voltage": 50.0,
        "forward_transit_time": 1e-11,
        "intrinsic_collector_resistance": 0.01,
        "polarity": "npn",
    }
    assert second.parameters == {
        "saturation_current": 2e-17,
        "forward_emission_coefficient": 1.02,
        "polarity": "pnp",
    }
    assert third.parameters == first.parameters


def test_a_vbic_whose_card_heats_reads_with_its_thermal_network() -> None:
    """A positive `RTH` heats it on its fifth node, unless `SELFT` is 0.

    Its thermal resistance and capacitance hold that node to the ground, the area times
    the copies dividing the resistance and multiplying the capacitance, as they multiply
    its currents; ngspice raises a capacitance below 1 pJ/K to 1 pJ/K.
    """
    parts = library(
        ".model QH NPN(level=4 is=1e-16 rth=200 cth=1n)\n"
        ".model QL NPN(level=4 is=1e-16 rth=200)\n"
        ".model QC NPN(level=4 is=1e-16 rth=200 cth=1n selft=0)\n"
        ".subckt a c b e s\nQ1 c b e s t QH 2 m=2\nQ2 c b e s u QL\nQ3 c b e s w QC\n"
        ".ends"
    )
    placed = {
        instance.name: (instance.of, instance.parameters, instance.terminals)
        for instance in parts.component("a").definition.instances
    }
    joined = {"collector": "c", "base": "b", "emitter": "e", "substrate": "s"}
    vbic, resistance = ir.DeviceOf(model="vbic"), "thermal_resistance"
    capacitance = "thermal_capacitance"
    # The area times the copies multiplies the currents the card leaves out at their
    # defaults too, and divides the resistances, as ngspice multiplies every current of
    # a VBIC by it (vbicload.c:4017-4052).
    scaled = {
        "saturation_current": 4e-16,
        "ideal_base_emitter_current": 4e-18,
        "ideal_base_collector_current": 4e-16,
        "base_emitter_breakdown_current": 4e-6,
        "intrinsic_collector_resistance": 0.025,
        "intrinsic_base_resistance": 0.025,
        "parasitic_base_resistance": 0.025,
    }
    assert placed == {
        "Q1": (
            vbic,
            {**scaled, "polarity": "npn", "thermal": True},
            {**joined, "thermal": "t"},
        ),
        "Q1.rth": (
            ir.DeviceOf(model=resistance),
            {"resistance": 50.0},
            {"a": "t", "b": "ground"},
        ),
        "Q1.cth": (
            ir.DeviceOf(model=capacitance),
            {"capacitance": 4e-9},
            {"a": "t", "b": "ground"},
        ),
        "Q2": (
            vbic,
            {"saturation_current": 1e-16, "polarity": "npn", "thermal": True},
            {**joined, "thermal": "u"},
        ),
        "Q2.rth": (
            ir.DeviceOf(model=resistance),
            {"resistance": 200.0},
            {"a": "u", "b": "ground"},
        ),
        "Q2.cth": (
            ir.DeviceOf(model=capacitance),
            {"capacitance": 1e-12},
            {"a": "u", "b": "ground"},
        ),
        "Q3": (vbic, {"saturation_current": 1e-16, "polarity": "npn"}, joined),
    }


def test_a_bipolar_card_of_another_level_is_refused() -> None:
    """A bipolar card of a level but 1, Gummel-Poon's, and VBIC's 4 and 9 is refused."""
    parts = library(
        ".model QX NPN(level=2 is=1e-16)\n.subckt a c b e\nQ1 c b e QX\n.ends"
    )
    with pytest.raises(SpiceError, match="of level 2; jolty models levels 1, 4 and 9"):
        parts.component("a")


def test_a_vbic_is_written_for_ngspice_as_its_card_reads_back() -> None:
    transistor = devices.vbic(
        polarity="pnp",
        saturation_current=2e-17 * A,
        extrinsic_collector_resistance=10 * ohm,
        forward_transit_time=0.01 * ns,
    )

    @component
    def bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        supply = c.place(devices.voltage_source(-5 * V), name="V1")
        placed = c.place(transistor, name="Q1")
        c.connect(supply.positive, placed.collector, placed.base, name="c")
        c.connect(supply.negative, placed.emitter, placed.substrate, ground)

    lines = netlist(circuit(bench()))
    card = next(line for line in lines if line.startswith(".model"))
    assert card.startswith(".model MQ1 PNP(level=4 "), card
    element = next(line for line in lines if line.startswith("QQ1"))
    read = library(f"{card}\n.subckt a c\n{element.replace('QQ1', 'Q1', 1)}\n.ends")
    (instance,) = read.component("a").definition.instances
    assert instance.parameters == transistor.parameters


def test_a_bsim3_card_reads_in_si_units_as_ngspice_reads_it() -> None:
    """A level 8 or 49 card makes a BSIM3, each parameter by its name in SI units.

    As ngspice reads them, `NCH` is per cubic centimeter up to 1e20 and per cubic meter
    above, `NSUB` per cubic centimeter, a `U0` above 1 in cm²/(V·s) with its
    dependences, and with `BINUNIT` 1, its default, each length and width dependence
    per micrometer; its bins' ranges, its safe operating area and the area
    calculation's parameters, which change nothing, are read past.
    """
    parts = library(
        ".model N1 NMOS(level=8 version=3.3.0 tnom=50 nch=2.5e17 lnch=1e16 nsub=1e17\n"
        "+ u0=400 lu0=-20 vth0=0.6 lvth0=0.02 pvth0=0.003 capmod=2 mobmod=3\n"
        "+ paramchk=1 xpart=0.5 lmin=0.1u vgs_max=4 ld=1n elm=3)\n"
        ".model P1 PMOS(level=49 binunit=2 nch=3e23 u0=0.02 lvth0=1e-8)\n"
        ".subckt a d g\nM1 d g 0 0 N1 W=2u L=0.5u nrd=2 delvto=0.1\n"
        "M2 d g 0 0 P1 L=1u off\n.ends"
    )
    first, second = parts.component("a").definition.instances
    assert first.of == second.of == ir.DeviceOf(model="bsim3")
    assert first.parameters == pytest.approx(
        {
            "channel": "n",
            "nominal_temperature": celsius(50).value,
            "nch": 2.5e23,
            "lnch": 1e16,
            "nsub": 1e23,
            "u0": 0.04,
            "lu0": -2e-9,
            "vth0": 0.6,
            "lvth0": 2e-8,
            "pvth0": 3e-15,
            "capacitance_model": 2,
            "mobility_model": 3,
            "check_parameters": True,
            "charge_partition": 0.5,
            "width": 2e-6,
            "length": 5e-7,
            "drain_squares": 2.0,
            "threshold_adjustment": 0.1,
        }
    )
    assert second.parameters == {
        "channel": "p",
        "nch": 3e23,
        "u0": 0.02,
        "lvth0": 1e-8,
        "length": 1e-6,
    }


@pytest.mark.parametrize(
    ("card", "error"),
    [
        ("level=49 version=3.1", "`N1` is a BSIM3 card of version 3.1"),
        ("level=8 nqsmod=1", "jolty models BSIM3 with `NQSMOD` 0 alone"),
        ("level=8 acnqsmod=1", "jolty models BSIM3 with `ACNQSMOD` 0 alone"),
        ("level=8 acm=2", "jolty models BSIM3 with `ACM` 0 alone"),
        ("level=8 capmod=1.5", "`CAPMOD` of `N1` is a whole number, not 1.5"),
        ("level=8 vto=1", "no counterpart to `VTO` of `N1`"),
        ("level=8 lvto=1", "no counterpart to `LVTO` of `N1`"),
    ],
)
def test_a_bsim3_card_jolty_cannot_run_as_ngspice_does_is_refused(
    card: str, error: str
) -> None:
    parts = library(f".model N1 NMOS({card})\n.subckt a d\nM1 d d 0 0 N1\n.ends")
    with pytest.raises(SpiceError, match=re.escape(error)):
        parts.component("a")


def test_an_element_takes_the_bin_of_its_length_and_width() -> None:
    """As ngspice's INPgetModBin takes a bin, the last defined that holds them.

    A bin is a card named after its model and a dot and a number, whose ranges hold the
    length and the width given on the element, each within 1 nm of a bound counted in.
    """
    parts = library(
        ".model nch.1 nmos(level=49 lmin=0.1u lmax=1u wmin=1u wmax=10u vth0=0.4)\n"
        ".model nch.2 nmos(level=49 lmin=1u lmax=10u wmin=1u wmax=10u vth0=0.5)\n"
        ".model nch.x nmos(level=49 lmin=0 lmax=1 wmin=0 wmax=1 vth0=0.6)\n"
        ".subckt a d\nM1 d d 0 0 nch L=0.5u W=2u\nM2 d d 0 0 nch L=2u W=2u\n"
        "M3 d d 0 0 nch L=1.0000000005u W=2u\n.ends\n"
        ".subckt wide d\nM1 d d 0 0 nch L=0.5u W=20u\n.ends\n"
        ".subckt unsized d\nM1 d d 0 0 nch\n.ends"
    )
    thresholds = [
        instance.parameters["vth0"]
        for instance in parts.component("a").definition.instances
    ]
    assert thresholds == [0.4, 0.5, 0.5]
    for name in ("wide", "unsized"):
        with pytest.raises(SpiceError, match="no model is named `nch`"):
            parts.component(name)


def test_a_bsim3_is_written_for_ngspice_as_its_card_reads_back() -> None:
    @component
    def bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        supply = c.place(devices.voltage_source(3 * V), name="V1")
        transistor = c.place(
            devices.bsim3(
                channel="p",
                width=2 * um,
                length=0.5 * um,
                capacitance_model=1,
                check_parameters=True,
                nominal_temperature=celsius(50),
                card={
                    "vth0": -0.6 * V,
                    "nch": 1.7e23 / m**3,
                    "lnch": 1e30 / m**2,
                    "nsub": 6e22 / m**3,
                    "lvth0": 1e-8 * m * V,
                    "k2": -0.02,
                },
            ),
            name="M1",
        )
        c.connect(supply.positive, transistor.drain, transistor.gate, name="d")
        c.connect(supply.negative, transistor.source, transistor.bulk, ground)

    lines = netlist(circuit(bench()))
    card = next(line for line in lines if line.startswith(".model"))
    assert card == (
        ".model MM1 PMOS(level=8 version=3.3.0 binunit=2 tnom=50.0 capmod=1 "
        "paramchk=1 vth0=-0.6 nch=1.7e+17 lnch=1e+30 nsub=6e+16 lvth0=1e-08 k2=-0.02)"
    ), card
    element = next(line for line in lines if line.startswith("MM1"))
    assert element == "MM1 d d 0 0 MM1 w=2e-06 l=5e-07", element
    read = library(
        f"{card}\n.subckt a d\n{element.replace('MM1 d d', 'M1 d d', 1)}\n.ends"
    )
    (instance,) = read.component("a").definition.instances
    assert instance.parameters == pytest.approx(
        {
            "channel": "p",
            "nominal_temperature": celsius(50).value,
            "capacitance_model": 1,
            "check_parameters": True,
            "vth0": -0.6,
            "nch": 1.7e23,
            "lnch": 1e30,
            "nsub": 6e22,
            "lvth0": 1e-8,
            "k2": -0.02,
            "width": 2e-6,
            "length": 5e-7,
        }
    )
    with pytest.raises(JoltyError, match="at its own temperature"):
        netlist(circuit(bench_held()))


@component
def bench_held(c: Builder) -> None:
    """A BSIM3 held at 85 °C, which ngspice's has no temperature of its own for."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(3 * V), name="V1")
    transistor = c.place(devices.bsim3().held_at(celsius(85)), name="M1")
    c.connect(supply.positive, transistor.drain, transistor.gate, name="d")
    c.connect(supply.negative, transistor.source, transistor.bulk, ground)


def test_a_bsim3s_parameters_take_their_dimensions() -> None:
    with pytest.raises(DimensionError, match="vth0 takes a quantity in V"):
        devices.bsim3(card={"vth0": 0.5})
    with pytest.raises(DimensionError, match="k2 takes a plain number"):
        devices.bsim3(card={"k2": 0.5 * V})
    with pytest.raises(ValueError, match="BSIM3 has no parameter `vto`"):
        devices.bsim3(card={"vto": 0.5 * V})
    assert (
        devices.bsim3(mobility_model=3, card={"uc": -0.05 / V}).parameters["uc"]
        == -0.05
    )
    with pytest.raises(DimensionError, match="uc takes a quantity in kg"):
        devices.bsim3(card={"uc": -0.05 / V})


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
        source = c.place(devices.voltage_source(1 * V).warmer_by(5 * K), name="V1")
        load = c.place(devices.resistor(1 * kohm).held_at(celsius(85)), name="R1")
        diode = c.place(devices.diode().warmer_by(10 * K), name="D1")
        placed = c.place(switch, name="S1")
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


def test_a_behavioral_expression_is_written_as_jolty_computes_it() -> None:
    """Each operation in parentheses, a power, a limit and a table as jolty reads them.

    ngspice groups `^` to the left, raises a negative base's magnitude and reads
    another `limit`, and its behavioral sources hold no table.
    """

    @component
    def shaped(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        expression = "limit(v(a) ^ 3, 2 ^ 3 ^ 2, -1) + table(v(a), 0, 1, 1, 3)"
        source = c.place(devices.behavioral_voltage_source(expression), name="B1")
        c.connect(source.a, name="a")
        c.connect(source.positive, name="out")
        c.connect(source.negative, ground)

    (line,) = [line for line in netlist(circuit(shaped())) if line.startswith("B")]
    limited = "min(max(pwr(v(a),3.0),min(512.0,(-1.0))),max(512.0,(-1.0)))"
    held = "(1.0+(v(a)-(-1.0))*0.0)"
    rising = "(1.0+(v(a)-(0.0))*2.0)"
    table = f"(v(a)<0.0?{held}:(v(a)<1.0?{rising}:(3.0+(v(a)-(1.0))*0.0)))"
    assert line == f"BB1 out 0 V={{({limited}+{table})}}"


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


def _every_model() -> list[devices.Device]:
    """A device of every model the engine holds, as a script places it."""
    rate = 1e6 / s
    return [
        devices.behavioral_current_source("1e-3 * v(a, b)"),
        devices.behavioral_voltage_source("2 * i(p)"),
        devices.bipolar_transistor(polarity="pnp", forward_early_voltage=50 * V),
        devices.capacitor(1 * uF),
        devices.coupled_inductors(
            [devices.inductor(1 * H), devices.inductor(2 * H)], {}
        ),
        devices.current_controlled_current_source(2),
        devices.current_controlled_memristor(
            on_resistance=100 * ohm, off_resistance=16 * kohm, charge=2e-6 * C
        ),
        devices.current_controlled_switch(on_current=2 * mA),
        devices.current_controlled_voltage_source(1 * kohm),
        devices.current_source(1 * A),
        devices.diode().heated(),
        sky130.mosfet("nfet_01v8", 1 * um, 0.15 * um),
        devices.heat_source(1 * W),
        devices.inductor(1 * H),
        devices.jfet(channel="p"),
        devices.logic_gate("nand", rise=1 * ns, transition=1 * ns, electrical=True),
        devices.logic_source(0, [(1 * ns, 1)], transition=1 * ns),
        devices.mosfet(level=3, oxide_thickness=10 * nm),
        devices.vdmos(channel="p", breakdown_voltage=60 * V),
        devices.vbic(polarity="pnp", saturation_current=2e-17 * A),
        devices.bsim3(
            channel="p",
            capacitance_model=0,
            card={"vth0": -0.6 * V, "lvth0": 1e-8 * m * V},
        ),
        devices.photocell(10 * kohm),
        devices.resistor(1 * kohm).held_at(celsius(85)),
        devices.thermal_capacitance(1 * J / K),
        devices.thermal_resistance(10 * K / W),
        devices.transmission_line(50 * ohm, 1 * ns),
        devices.voltage_controlled_current_source(1e-3 * A / V),
        devices.voltage_controlled_memristor(
            on_resistance=1 * kohm,
            off_resistance=100 * kohm,
            on_voltage=-1 * V,
            off_voltage=0.5 * V,
            on_rate=rate,
            off_rate=rate,
        ),
        devices.voltage_controlled_switch(),
        devices.voltage_controlled_voltage_source(10),
        devices.voltage_source(5 * V),
    ]


def _within(given: JsonValue, defaulted: JsonValue) -> bool:
    """Whether `defaulted` holds `given`, each object within it with defaults added."""
    match given, defaulted:
        case dict(), dict():
            return all(
                key in defaulted and _within(value, defaulted[key])
                for key, value in given.items()
            )
        case list(), list():
            return len(given) == len(defaulted) and all(
                _within(*pair) for pair in zip(given, defaulted, strict=True)
            )
    return given == defaulted


def test_the_engine_gives_every_models_parameters_with_their_defaults() -> None:
    placed = {device.model: device.parameters for device in _every_model()}
    assert sorted(placed) == _core.models()
    for model, given in placed.items():
        defaulted = json.loads(_core.defaulted(model, json.dumps(given)))
        # The parameters with their defaults read back as they are.
        assert json.loads(_core.defaulted(model, json.dumps(defaulted))) == defaulted
        schema = json.loads(_core.schema(model))
        for name, field in schema["properties"].items():
            if name in given:
                assert _within(given[name], defaulted[name]), (model, name)
            elif "default" in field:
                assert defaulted[name] == field["default"], (model, name)
            else:
                assert name in {"temperature", "temperature_offset"}, (model, name)


def test_the_engine_says_what_it_cannot_default() -> None:
    with pytest.raises(ValueError, match="no model is named `transistor`"):
        _core.defaulted("transistor", "{}")
    with pytest.raises(ValueError, match="a resistor's parameters: missing field"):
        _core.defaulted("resistor", "{}")


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


def test_a_file_fetched_at_once_by_several_is_kept_whole_for_each(
    tmp_path: Path,
) -> None:
    published = tmp_path / "part.lib"
    published.write_bytes(bytes(range(256)) * 4096)
    digest = hashlib.sha256(published.read_bytes()).hexdigest()
    cache = tmp_path / "cache"
    # Fetches of a mebibyte, started together, write it at the same time.
    fetching = 8
    together = threading.Barrier(fetching)

    def fetch(_: int) -> bytes:
        together.wait()
        return spice.fetch(published.as_uri(), digest, cache).read_bytes()

    with ThreadPoolExecutor(fetching) as pool:
        kept = list(pool.map(fetch, range(fetching)))
    assert kept == [published.read_bytes()] * fetching


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


# A process kit's constructs, read as ngspice reads them.


def _devices(part: Component) -> dict[str, tuple[str, dict[str, JsonValue]]]:
    """Each device `part` holds, by name: its model and its parameters."""
    return {
        instance.name: (instance.of.model, dict(instance.parameters))
        for instance in part.definition.instances
        if isinstance(instance.of, ir.DeviceOf)
    }


def test_a_library_reads_a_section_and_the_files_it_includes(tmp_path: Path) -> None:
    (tmp_path / "corners.lib").write_text(
        ".lib typical\n.param sheet=1k\n.include 'parts/divider.lib'\n.endl typical\n"
        ".lib fast\n.param sheet=900\n.lib corners.lib shared\n.endl fast\n"
        '.lib shared\n.option scale=1u\n.include "parts/divider.lib"\n.endl shared\n',
        encoding="utf-8",
    )
    (tmp_path / "parts").mkdir()
    (tmp_path / "parts" / "divider.lib").write_text(
        ".subckt divider a b\nR1 a b {2 * sheet}\n.ends\n", encoding="utf-8"
    )
    typical, fast = spice.Library(), spice.Library()
    typical.include(tmp_path / "corners.lib", "typical")
    fast.include(tmp_path / "corners.lib", "FAST")
    resistances = [
        _devices(read.component("divider"))["R1"][1]["resistance"]
        for read in (typical, fast)
    ]
    assert resistances == [2e3, 1.8e3]
    assert (typical.options, fast.options) == ({}, {"scale": 1e-6})
    with pytest.raises(SpiceError, match="has no section `slow`"):
        spice.Library().include(tmp_path / "corners.lib", "slow")
    (tmp_path / "loop.lib").write_text(".include loop.lib\n", encoding="utf-8")
    with pytest.raises(SpiceError, match="includes itself"):
        spice.load(tmp_path / "loop.lib")


def test_a_subcircuits_if_blocks_choose_by_its_parameters() -> None:
    read = library(
        ".subckt pick a b mode=0\n.if (mode == 0)\nR1 a b 1k\n.elseif (mode == 1)\n"
        "R1 a b 2k\n.else\n.if (mode > 5)\nR1 a b 4k\n.else\nR1 a b 3k\n.endif\n"
        ".endif\n.ends"
    )
    chosen = [
        _devices(read.component("pick", mode=mode))["R1"][1]["resistance"]
        for mode in (0, 1, 2, 9)
    ]
    assert chosen == [1e3, 2e3, 3e3, 4e3]


def test_an_instance_gives_its_subcircuits_own_parameters() -> None:
    read = library(
        ".subckt load a b\n.param r = 1k + extra extra = 0\nR1 a b {r}\n.ends\n"
        ".subckt pair a b\nX1 a b r = 2k load\nX2 a b load\n.ends"
    )
    assert _devices(read.component("load", r=5e3))["R1"][1]["resistance"] == 5e3
    pair = read.component("pair")
    loads = [child for child in pair.children if child.name.startswith("load")]
    resistances = {_devices(load)["R1"][1]["resistance"] for load in loads}
    assert resistances == {1e3, 2e3}


def test_temper_is_the_temperature_a_library_reads_at() -> None:
    hot = spice.Library(celsius(127).value)
    hot.read(".subckt warm a b\nR1 a b {1k * (1 + temper / 100)}\n.ends", "kit.lib")
    assert _devices(hot.component("warm"))["R1"][1]["resistance"] == pytest.approx(
        2.27e3
    )


MISMATCHED: Final = (
    ".param global = agauss(0, 1, 1)\n"
    ".subckt matched a b\n.param x = agauss(1, 0.1, 1)\n"
    "R1 a b {1k * x * (1 + global)}\n.ends"
)
"""A resistor drawn per placement and per sample, and scaled by a global draw."""


def test_a_placements_draws_are_its_own_and_its_samples() -> None:
    first, second = library(MISMATCHED), library(MISMATCHED)

    def drawn(read: spice.Library, sample: spice.Sample, path: str) -> JsonValue:
        return _devices(read.drawn("matched", sample, path))["R1"][1]["resistance"]

    sample = spice.Sample(seed=3, index=8)
    assert drawn(first, sample, "X1") == drawn(second, sample, "X1")
    assert drawn(first, sample, "X1") != drawn(first, sample, "X2")
    assert drawn(first, sample, "X1") != drawn(first, spice.Sample(3, 9), "X1")
    assert _devices(first.component("matched"))["R1"][1]["resistance"] == 1e3


def test_a_resistor_whose_value_reads_a_voltage_is_refused() -> None:
    read = library(".subckt bent a b\nR1 a b r='1k * (1 + abs(v(a, b)))'\n.ends")
    with pytest.raises(SpiceError, match="whose value reads a voltage or a current"):
        read.component("bent")


def test_a_diodes_level_is_its_cards_and_its_area_positive() -> None:
    read = library(
        ".model plain d level=1 is=1e-14\n.model geometric d level=3 is=1e-14\n"
        ".subckt one a b\nD1 a b plain\n.ends\n.subckt three a b\nD1 a b geometric\n"
        ".ends\n.subckt none a b\nD1 a b plain area=0\n.ends"
    )
    assert _devices(read.component("one"))["D1"][0] == "diode"
    with pytest.raises(SpiceError, match="a diode of level 3; jolty models level 1"):
        read.component("three")
    with pytest.raises(SpiceError, match="area times its copies is positive, not 0"):
        read.component("none")


def test_a_resistor_on_a_vbics_thermal_node_is_a_thermal_resistance() -> None:
    read = library(
        ".model hot npn level=9 is=1e-16 rth=300 cth=1e-12\n"
        ".subckt heated c b e t\nQ1 c b e 0 t hot\nRT t 0 1e9\nCT t 0 1p\n.ends\n"
        ".subckt mixed c b e t\nQ1 c b e 0 t hot\nRT t c 1k\n.ends"
    )
    found = _devices(read.component("heated"))
    assert found["RT"] == ("thermal_resistance", {"resistance": 1e9})
    assert found["CT"] == ("thermal_capacitance", {"capacitance": 1e-12})
    with pytest.raises(SpiceError, match="joins a thermal node to another"):
        read.component("mixed")


def test_a_library_reads_initial_conditions_and_nodesets() -> None:
    read = library(".param top=2\n.ic v(out)=0.5 V(In)={top}\n.nodeset v(x)=-1m")
    assert read.initial_conditions == {"out": 0.5, "In": 2.0}
    assert read.nodesets == {"x": -1e-3}
    with pytest.raises(
        SpiceError, match=re.escape("written `V(node)=value`, not `out")
    ):
        library(".ic out=1")


@pytest.mark.parametrize(
    ("element", "message"),
    [
        ("Rx/r1 a 0 1k", "an element named `Rx/r1`, whose `/` separates the levels"),
        ("R1 a/b 0 1k", "a node named `a/b`, whose `/` separates the levels"),
    ],
)
def test_a_name_that_holds_a_path_separator_is_refused(
    element: str, message: str
) -> None:
    with pytest.raises(SpiceError, match=re.escape(message)):
        library(f".subckt part a\n{element}\nR2 a 0 1k\n.ends").component("part")


def test_user_functions_are_written_out_in_every_expression() -> None:
    parts = library(
        ".func twice(x) {2*x}\n"
        ".func Quad(x) = {twice(twice(x))}\n"
        ".param p={quad(1)}\n"
        ".subckt parts a\n"
        "R1 a 0 {twice(p)}\n"
        "B1 b 0 V={twice(V(a))}\n"
        "R2 b 0 1k\n"
        ".ends"
    )
    read = {
        instance.name: instance.parameters
        for instance in parts.component("parts").definition.instances
    }
    assert parts.values()["p"] == 4.0
    assert read["R1"]["resistance"] == 8.0
    assert read["B1"]["expression"] == "(2.0 * v(a))"


@pytest.mark.parametrize(
    ("functions", "message"),
    [
        (".func f(x) {f(x)}", "a function calls itself: `f` calls `f`"),
        (
            ".func f(x) {g(x)}\n.func g(x) {1 + f(x)}",
            "a function calls itself: `f` calls `g` calls `f`",
        ),
        (".func f(x, y) {x*y}", "`f` takes 2 arguments, not 1"),
    ],
)
def test_a_function_that_calls_itself_is_refused_by_name(
    functions: str, message: str
) -> None:
    with pytest.raises(SpiceError, match=re.escape(message)):
        library(f"{functions}\n.subckt part a\nR1 a 0 {{f(2)}}\n.ends").component(
            "part"
        )
