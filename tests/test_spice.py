# Copyright 2026 Maxence Faldor

import hashlib
import json
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from jolty import (
    Ac,
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
    results,
    spice,
)
from jolty.plan import TIGHT
from jolty.spice import models, sky130
from jolty.spice.export import netlist
from jolty.spice.expressions import evaluate, number, parse, translate
from jolty.spice.netlist import statements, tokens
from jolty.spice.ngspice import Compatibility
from jolty.units import (
    VOLTAGE,
    A,
    C,
    H,
    Hz,
    J,
    K,
    V,
    W,
    celsius,
    kohm,
    m,
    mA,
    nF,
    nm,
    ns,
    ohm,
    s,
    uF,
    uH,
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
        ("POW(-2, 3) + (-2) ** 3 + PWR(-2, 3)", -8.0),
        ("POW(-2, 2) + POW(-4, 0.5)", 6.0),
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


def test_a_diode_is_sized_by_its_area_perimeter_and_copies_as_ngspice_sizes_it() -> (
    None
):
    parts = library(
        ".option scale=1u\n"
        ".model d3 d level=3 js=1e-3 rs=2e-10 cj=1e-3 jsw=1e-9 cjsw=1e-10 rsw=1e-4"
        " ik=1e5 ikr=0 ibv=1e3 bv=5 xw=0.5 tref=30 tlev=2 tlevc=1 scalm=1e-6\n"
        ".model d1 d is=1e-14 jsw=1e-15 ibv=1e-3 bv=5 area=3 pj=2\n"
        ".subckt parts a c\nD1 a c d3 w=1.5 l=3.5 m=2\nD2 a c d1 m=2\nD3 a c d1 w=1\n"
        ".ends"
    )
    read = {
        instance.name: instance.parameters
        for instance in parts.component("parts").definition.instances
    }
    # Level 3: an area of (1.5 + 0.5)(3.5 + 0.5) µm², a perimeter of 2·2 + 2·4 µm,
    # each times the copies once (`diosetup.c`, lines 336 and 337, where ngspice
    # multiplies them again, as the guide's departures state); `IKR` of 0 is off, and
    # the breakdown current scales with the area.
    area, perimeter = 8e-12 * 2, 12e-6 * 2
    assert read["D1"] == pytest.approx(
        {
            "saturation_current": 1e-3 * area,
            "series_resistance": 2e-10 / area,
            "junction_capacitance": 1e-3 * area,
            "sidewall_saturation_current": 1e-9 * perimeter,
            "sidewall_junction_capacitance": 1e-10 * perimeter,
            "sidewall_series_resistance": 1e-4 / perimeter,
            "forward_knee_current": 1e5 * area,
            "breakdown_current": 1e3 * area,
            "breakdown_voltage": 5.0,
            "nominal_temperature": celsius(30).value,
            "temperature_model": 2,
            "capacitance_temperature_model": 1,
        }
    )
    # Level 1: the card's area and perimeter where the element gives no size, and 1 and
    # 0 where it gives a width; the breakdown current scales with the copies alone.
    assert read["D2"] == pytest.approx(
        {
            "saturation_current": 1e-14 * 6,
            "sidewall_saturation_current": 1e-15 * 4,
            "breakdown_current": 2e-3,
            "breakdown_voltage": 5.0,
        }
    )
    assert read["D3"] == pytest.approx(
        {
            "saturation_current": 1e-14,
            "breakdown_current": 1e-3,
            "breakdown_voltage": 5.0,
        }
    )


def test_a_level_3_diodes_metal_and_polysilicon_are_refused() -> None:
    read = library(".model d d level=3 wm=1 lm=1\n.subckt one a c\nD1 a c d\n.ends")
    with pytest.raises(SpiceError, match="no metal or polysilicon"):
        read.component("one")


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


def test_a_bipolar_card_reads_hspices_temperature_terms_as_ngspice_does() -> None:
    """`TREF`, `TLEV`, `TLEVC`, `NKF` and the coefficients, and what ngspice reads past.

    ngspice reads `TREF` as `TNOM`, a lone `IBC` not at all, HSPICE's `DCAP` and kin
    nowhere, and `TLEV` 3's coefficients only at `TLEV` 3 (`bjttemp.c:166-197`); it
    limits `NKF` to 1 (`bjtsetup.c:332-338`), and takes an `ISS` of 0 as none.
    """
    parts = library(
        ".model q pnp tref=25 tlev=1 tlevc=1 nkf=1.2 trc1=-0.01 trc2=3e-5 tbf1=6e-3\n"
        "+ cte=1e-3 tvje=2e-3 tis1=1 tise2=1 ibc=0 dcap=2 gap1=0 cbcp=0 iss=0 rc=10\n"
        ".subckt one c b\nQ1 c b 0 q\n.ends"
    )
    (transistor,) = parts.component("one").definition.instances
    assert transistor.parameters == {
        "polarity": "pnp",
        "substrate": "lateral",
        "temperature_model": 1,
        "capacitance_temperature_model": 1,
        "nominal_temperature": pytest.approx(298.15),
        "high_injection_exponent": 1.0,
        "collector_resistance_temperature_coefficient": -0.01,
        "collector_resistance_quadratic_temperature_coefficient": 3e-5,
        "forward_beta_temperature_coefficient": 6e-3,
        "base_emitter_capacitance_temperature_coefficient": 1e-3,
        "base_emitter_potential_temperature_coefficient": 2e-3,
        "collector_resistance": 10.0,
    }


@pytest.mark.parametrize(
    ("given", "refusal"),
    [
        ("tlev=3", "as `TLEV` 3 says; jolty models 0 and 1"),
        ("tlevc=2", "as `TLEVC` 2 says; jolty models 0 and 1"),
        ("ibe=1e-16 ibc=2e-16", "no counterpart to `IBE`"),
    ],
)
def test_a_bipolar_card_jolty_does_not_model_is_refused_by_name(
    given: str, refusal: str
) -> None:
    parts = library(f".model q npn {given}\n.subckt one c b\nQ1 c b 0 q\n.ends")
    with pytest.raises(SpiceError, match=re.escape(refusal)):
        parts.component("one")


def test_a_bipolar_transistors_temperature_terms_are_written_back() -> None:
    """The exporter writes `TLEV`, `TLEVC` and the coefficients, which read back."""
    given = {
        "temperature_model": 1,
        "capacitance_temperature_model": 1,
        "high_injection_exponent": 0.4,
        "collector_resistance": 10.0,
        "collector_resistance_temperature_coefficient": -0.01,
        "collector_resistance_quadratic_temperature_coefficient": 3e-5,
        "base_emitter_potential_temperature_coefficient": 2e-3,
    }

    @component
    def bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        supply = c.place(devices.voltage_source(5 * V), name="V1")
        transistor = c.place(
            devices.bipolar_transistor(
                temperature_model=1,
                capacitance_temperature_model=1,
                high_injection_exponent=0.4,
                collector_resistance=10 * ohm,
                collector_resistance_temperature_coefficient=-0.01 / K,
                collector_resistance_quadratic_temperature_coefficient=3e-5 / K**2,
                base_emitter_potential_temperature_coefficient=2e-3 * V / K,
            ),
            name="Q1",
        )
        c.connect(supply.positive, transistor.collector, transistor.base, name="c")
        c.connect(supply.negative, transistor.emitter, ground)

    lines = netlist(circuit(bench()))
    card = next(line for line in lines if line.startswith(".model"))
    element = next(line for line in lines if line.startswith("QQ1"))
    read = library(f"{card}\n.subckt a c\n{element.replace('QQ1', 'Q1', 1)}\n.ends")
    (instance,) = read.component("a").definition.instances
    assert instance.parameters == {**given, "polarity": "npn", "substrate": "vertical"}


def admittance(part: Component, bias: float, frequency: float) -> complex:
    """The current into `part`'s first pin, its second grounded, per volt at AC.

    The first pin is held at `bias` volts.
    """

    @component
    def bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(bias * V), name="V1")
        placed = c.place(part, name="X1")
        first, second, *_ = part.definition.ports
        c.connect(source.positive, placed[first.name], name="in")
        c.connect(source.negative, placed[second.name], ground)

    simulation = Simulation(bench())
    response = simulation.run(Ac("V1", [frequency * Hz]), TIGHT)
    column = simulation.unknowns.index(results.Unknown("V1:current", "current"))
    return -complex(response.values[0, column])


CHARGED: Final = {
    "q='1e-12 * v(p, n) + 1e-13 * v(p, n) * v(p, n)' m=2": (
        lambda v: 2 * (1e-12 + 2e-13 * v)
    ),
    "c='1e-12 * (1 + 0.1 * v(p, n) + 0.01 * v(p, n) ^ 2) * k' dtemp=5": (
        lambda v: 2e-12 * (1 + 0.1 * v + 0.01 * v**2)
    ),
    "c='k * 1e-12 * (1 + 0.5 * tanh(2 * v(p, n) - 1))'": (
        lambda v: 2e-12 * (1 + 0.5 * math.tanh(2 * v - 1))
    ),
    "c='1e-12 * (exp(-v(n, p)) + cosh(v(p, n) / 2) - sin(3 * v(p, n)))'": (
        lambda v: 1e-12 * (math.exp(v) + math.cosh(v / 2) - math.sin(3 * v))
    ),
}
"""Capacitors of a charge and of a value that reads their voltage, by what their lines
give, and the capacitance each should have at a voltage: the charge's derivative, or
the value, times the copies `m`; `k` is the subcircuit's parameter, 2."""


@pytest.mark.parametrize("given", CHARGED)
def test_a_capacitor_of_a_charge_holds_it_as_a_behavioral_capacitor(given: str) -> None:
    """Its admittance at a bias is `jωC` of the capacitance there, its charge's slope.

    A value that reads the capacitor's voltage is integrated in closed form into its
    charge, so its slope is the value itself; at DC no current flows. The tolerance,
    1e-9 of the admittance, is the engine's Newton tolerance at most, far above the
    closed forms' rounding.
    """
    parts = library(f".subckt cap p n k=2\nC1 p n {given}\n.ends")
    part = parts.component("cap")
    (instance,) = part.definition.instances
    assert instance.of == ir.DeviceOf(model="behavioral_capacitor")
    frequency = 1e6
    for bias in (-1.5, 0.0, 0.7):
        expected = 2j * math.pi * frequency * CHARGED[given](bias)
        assert admittance(part, bias, frequency) == pytest.approx(expected, rel=1e-9)


def test_a_grounded_capacitors_value_reads_its_node_as_its_voltage() -> None:
    """`V(p)` of a capacitor from `p` to the ground is its voltage, as `V(p, 0)`."""
    parts = library(".subckt cap p\nC1 p 0 c='1e-12 * (2 + v(p))'\n.ends")
    part = parts.component("cap")
    expected = 2j * math.pi * 1e6 * 1e-12 * (2 + 0.5)
    assert admittance(part, 0.5, 1e6) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(
    ("given", "refusal"),
    [
        ("c='1e-12 * v(x)'", "reads another voltage or a current"),
        ("c='1e-12 * abs(v(p, n))'", "has no integral in closed form"),
        ("c='1e-12 * tanh(v(p, n) * v(p, n))'", "has no integral in closed form"),
        ("q='1e-12 * v(p, n)' tc1=1e-3", "jolty reads no `TC1` of a capacitor"),
        ("q='1e-12 * v(p, n)' q='2'", "a capacitor's charge `Q` is one expression"),
    ],
)
def test_a_capacitor_jolty_cannot_charge_is_refused(given: str, refusal: str) -> None:
    parts = library(f".subckt cap p n x\nC1 p n {given}\n.ends")
    with pytest.raises(SpiceError, match=re.escape(refusal)):
        parts.component("cap")


def test_a_behavioral_capacitor_is_written_back_as_a_charge() -> None:
    """The exporter writes its charge `Q=`, which reads back as the same capacitor."""
    charge = "1e-12 * v(positive, negative) + 5e-13 * tanh(2 * v(positive, negative))"

    @component
    def bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(1 * V), name="V1")
        placed = c.place(devices.behavioral_capacitor(charge), name="B1")
        c.connect(source.positive, placed.positive, name="p")
        c.connect(source.negative, placed.negative, ground)

    lines = netlist(circuit(bench()))
    element = next(line for line in lines if line.startswith("CB1"))
    assert element.startswith("CB1 p 0 Q={")
    read = library(f".subckt a p\n{element.replace('CB1', 'C1', 1)}\n.ends")
    capacitance = 1e-12 + 5e-13 * 2 * (1 - math.tanh(2 * 0.3) ** 2)
    expected = 2j * math.pi * 1e6 * capacitance
    assert admittance(read.component("a"), 0.3, 1e6) == pytest.approx(
        expected, rel=1e-9
    )


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
        (".model D1 D(VP=0.5)\n.subckt a x\nD1 x 0 D1\n.ends", "counterpart to `VP`"),
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


def test_a_bsim4_card_reads_in_si_units_as_ngspice_reads_it() -> None:
    """A level 14 or 54 card of version 4.5 makes a BSIM4, each parameter in SI units.

    As ngspice reads them, `NDEP` is per cubic centimeter up to 1e20 and per cubic
    meter above, `NSD` and `NGATE` up to 1e24, `NSUB` per cubic centimeter, a `U0`
    above 1 in cm²/(V·s) with its dependences, `VTHO` is `VTH0`, and with `BINUNIT` 1,
    its default, each length and width dependence per micrometer; its bins' ranges,
    which only select it, are read past, and an instance's gate contacts, `XGW` and
    `NGCON`, stand for its card's.
    """
    parts = library(
        ".model N1 NMOS(level=54 version=4.5 tnom=25 ndep=1.7e17 lndep=1e16 nsd=1e20\n"
        "+ ngate=3e25 nsub=6e16 u0=400 lu0=-20 vtho=0.5 lvth0=0.02 rbodymod=1\n"
        "+ permod=0 paramchk=0 xpart=0.5 lmin=0.1u tnoimod=1 capmod=2 mobmod=0\n"
        "+ rgatemod=3 xgw=0.5u)\n"
        ".subckt a d g\nM1 d g 0 0 N1 W=2u L=0.15u nf=2 sa=1u sb=2u sd=0.5u\n"
        "+ nrd=2 delvto=0.1 rbpb=20 xgw=1u ngcon=2\n.ends"
    )
    (instance,) = parts.component("a").definition.instances
    assert instance.of == ir.DeviceOf(model="bsim4")
    assert instance.parameters == pytest.approx(
        {
            "channel": "n",
            "nominal_temperature": celsius(25).value,
            "ndep": 1.7e23,
            "lndep": 1e16,
            "nsd": 1e26,
            "ngate": 3e25,
            "nsub": 6e22,
            "u0": 0.04,
            "lu0": -2e-9,
            "vth0": 0.5,
            "lvth0": 2e-8,
            "body_resistance_model": 1,
            "gate_resistance_model": 3,
            "perimeter_model": 0,
            "thermal_noise_model": 1,
            "check_parameters": False,
            "charge_partition": 0.5,
            "width": 2e-6,
            "length": 1.5e-7,
            "fingers": 2.0,
            "diffusion_edge_a": 1e-6,
            "diffusion_edge_b": 2e-6,
            "finger_spacing": 5e-7,
            "drain_squares": 2.0,
            "threshold_adjustment": 0.1,
            "rbpb": 20.0,
            "xgw": 1e-6,
            "ngcon": 2.0,
        }
    )


@pytest.mark.parametrize(
    ("card", "error"),
    [
        ("level=54 version=4.6", "`N1` is a BSIM4 card of version 4.6"),
        ("level=14", "`N1` is a BSIM4 card of version 4.8"),
        ("level=54 version=4.5 capmod=1", "jolty models BSIM4 with `CAPMOD` 2 alone"),
        ("level=54 version=4.5 igcmod=1", "jolty models BSIM4 with `IGCMOD` 0 alone"),
        (
            "level=54 version=4.5 rgatemod=4",
            "gate_resistance_model is 0, 1, 2 or 3, not 4",
        ),
        ("level=54 version=4.5 rbodymod=2", "body_resistance_model 0 or 1, not 2"),
        ("level=54 version=4.5 vto=1", "no counterpart to `VTO` of `N1`"),
    ],
)
def test_a_bsim4_card_jolty_cannot_run_as_ngspice_does_is_refused(
    card: str, error: str
) -> None:
    parts = library(f".model N1 NMOS({card})\n.subckt a d\nM1 d d 0 0 N1\n.ends")
    with pytest.raises((SpiceError, ValueError), match=re.escape(error)):
        parts.component("a")


def test_a_bsim4_is_written_for_ngspice_as_its_card_reads_back() -> None:
    @component
    def bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        supply = c.place(devices.voltage_source(1.8 * V), name="V1")
        transistor = c.place(
            devices.bsim4(
                channel="p",
                width=2 * um,
                length=0.15 * um,
                fingers=2,
                body_resistance_model=1,
                nominal_temperature=celsius(50),
                card={
                    "vth0": -0.5 * V,
                    "ndep": 1.7e23 / m**3,
                    "ngate": 1e29 / m**3,
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
        ".model MM1 PMOS(level=54 version=4.5 binunit=2 tnom=50.0 rbodymod=1 "
        "vth0=-0.5 ndep=1.7e+17 ngate=1e+23 nsub=6e+16 lvth0=1e-08 k2=-0.02)"
    ), card
    element = next(line for line in lines if line.startswith("MM1"))
    assert element == "MM1 d d 0 0 MM1 w=2e-06 l=1.5e-07 nf=2.0", element
    read = library(
        f"{card}\n.subckt a d\n{element.replace('MM1 d d', 'M1 d d', 1)}\n.ends"
    )
    (instance,) = read.component("a").definition.instances
    assert instance.parameters == pytest.approx(
        {
            "channel": "p",
            "nominal_temperature": celsius(50).value,
            "body_resistance_model": 1,
            "vth0": -0.5,
            "ndep": 1.7e23,
            "ngate": 1e29,
            "nsub": 6e22,
            "lvth0": 1e-8,
            "k2": -0.02,
            "width": 2e-6,
            "length": 1.5e-7,
            "fingers": 2.0,
        }
    )


@pytest.mark.parametrize("model", models.BSIMS)
def test_a_bsims_tables_name_the_engines_parameters(model: str) -> None:
    """The reader's tables of a BSIM model name the parameters the engine's model has.

    Its card's, its bins' and its instance's parameters, and its choices and settings,
    are every parameter of the engine's model but its own temperature and its heating,
    which no card gives.
    """
    bsim = models.BSIMS[model]
    named = {
        "channel",
        "nominal_temperature",
        *bsim.card,
        *(prefix + name for name in bsim.binned for prefix in "lwp"),
        *(jolty for jolty, _ in bsim.instance.values()),
        *bsim.selectors.values(),
        *bsim.settings.values(),
    }
    schema = json.loads(_core.schema(model))
    unread = {"temperature", "temperature_offset", "thermal"}
    assert named == set(schema["properties"]) - unread


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
        ".subckt wide d\nM1 d d 0 0 nch L=0.5u W=20e-6\n.ends\n"
        ".subckt unsized d\nM1 d d 0 0 nch\n.ends"
    )
    thresholds = [
        instance.parameters["vth0"]
        for instance in parts.component("a").definition.instances
    ]
    assert thresholds == [0.4, 0.5, 0.5]
    with pytest.raises(
        SpiceError,
        match="no bin of `nch` holds a length of 5e-07 m and a width of 2e-05 m",
    ):
        parts.component("wide")
    with pytest.raises(SpiceError, match="no bin of `nch` is selected without a size"):
        parts.component("unsized")


FINGERED: Final = (
    ".model nch.1 nmos(level=54 version=4.5 lmin=0.1u lmax=1u wmin=1u wmax=3u "
    "vth0=0.4)\n"
    ".model nch.2 nmos(level=54 version=4.5 lmin=0.1u lmax=1u wmin=3u wmax=9u "
    "vth0=0.5)\n"
    ".subckt fingers d\nM1 d d 0 0 nch L=0.5u W=4u nf=2\n.ends\n"
)
"""Two bins of a BSIM4, 1 µm to 3 µm wide and 3 µm to 9 µm, and a transistor 4 µm wide
in two fingers."""


@pytest.mark.parametrize(
    ("compatibility", "option", "threshold"),
    [
        (None, "", 0.5),
        ("hsa", "", 0.4),
        ("hsa", ".option wnflag=0\n", 0.5),
        (None, ".option wnflag=1\n", 0.4),
    ],
)
def test_a_bin_holds_one_fingers_width_where_ngspice_divides_it(
    compatibility: Compatibility | None, option: str, threshold: float
) -> None:
    """A bin holds the width over `nf` as INPgetModBin selects it.

    It does in HSPICE's compatibility, or where the option `wnflag` asks.
    """
    parts = spice.Library(compatibility=compatibility)
    parts.read(option + FINGERED, "fingers.lib")
    (instance,) = parts.component("fingers").definition.instances
    assert instance.parameters["vth0"] == threshold


def test_the_scale_option_scales_a_bsim4s_sizes_and_distances() -> None:
    """The option `scale` multiplies a BSIM4's lengths, and its areas by its square.

    A BSIM4's length, width and perimeters, and its distances to its diffusion's edges
    and between its fingers, as `b4v5par.c` scales them, all in micrometres here.
    """
    parts = spice.Library()
    parts.read(
        ".option scale=1e-6\n.model nch nmos(level=54 version=4.5)\n"
        ".subckt one d\nM1 d d 0 0 nch L=0.5 W=2 ad=1 pd=4 sa=0.25 sb=0.5 sd=0.125\n"
        ".ends\n",
        "scaled.lib",
    )
    (instance,) = parts.component("one").definition.instances
    assert {
        name: instance.parameters[name]
        for name in (
            "length",
            "width",
            "drain_area",
            "drain_perimeter",
            "diffusion_edge_a",
            "diffusion_edge_b",
            "finger_spacing",
        )
    } == pytest.approx(
        {
            "length": 5e-7,
            "width": 2e-6,
            "drain_area": 1e-12,
            "drain_perimeter": 4e-6,
            "diffusion_edge_a": 2.5e-7,
            "diffusion_edge_b": 5e-7,
            "finger_spacing": 1.25e-7,
        },
        rel=1e-15,
    )


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


def test_a_resistor_of_a_voltage_is_ngspices_behavioral_current() -> None:
    """As `inpcom.c` makes `R = {f(V)}` a current `m V / (f(V) (1 + TC1 dT))`.

    Two copies of 1 kΩ that grow by 10% a volt, a temperature coefficient of 1e-3 per
    kelvin, read at 77 °C, 50 K above ngspice's 27 °C: at 2 V, 2 · 2 V / (1.2 kΩ ·
    1.05).
    """
    parts = spice.Library(celsius(77).value)
    parts.read(
        ".subckt grown a\nR1 a 0 r={1k * (1 + 0.1 * abs(v(a)))} tc1=1e-3 m=2\n.ends",
        "grown.lib",
    )

    @component
    def bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(2 * V), name="V1")
        placed = c.place(parts.component("grown"), name="X1")
        c.connect(source.positive, placed["a"], name="a")
        c.connect(source.negative, placed["ground"], ground)

    simulation = Simulation(bench())
    point = simulation.run(OperatingPoint())
    current = point.values[
        simulation.unknowns.index(results.Unknown("V1:current", "current"))
    ]
    assert -current == pytest.approx(4 / (1.2e3 * 1.05), rel=1e-12)


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
        devices.behavioral_capacitor("1e-12 * tanh(v(positive, negative))"),
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
        devices.bsim4(
            channel="p",
            body_resistance_model=1,
            card={"vth0": -0.5 * V, "lvth0": 1e-8 * m * V},
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


MULTIPLIED: Final = (
    ".model DX D(IS=1e-14 KF=0)\n"
    ".model RN RES(KF=1e-12 AF=2)\n"
    ".subckt cell a b\nR1 a b 3k\nR2 a b RN 3k m=2\nC1 a b 2p\nL1 a b 6n\n"
    "D1 a b DX\nX1 a b inner m=2\n.ends\n"
    ".subckt inner a b\nR1 a b 4k\n.ends\n"
    ".subckt own a b m=1\nR1 a b {1k*m}\n.ends\n"
    ".subckt sourced a b\nI1 a b 1m\n.ends\n"
)
"""Subcircuits a multiplier `m` multiplies, and one that reads `m` as its parameter."""


def test_a_subcircuits_multiplier_multiplies_every_element_it_holds() -> None:
    """As ngspice's `inp_fix_subckt_multiplier` adds `m` to each element.

    Three copies of `cell` divide its resistors and its inductor by 3 and multiply its
    capacitor and its diode by 3, its resistor's own 2 copies times 3, whose flicker
    noise is `m KF (I / m)^AF` (`resnoise.c`), and its instance's 2 copies of `inner`
    times 3; a subcircuit with a parameter `m` reads it as its own, and a current
    source under a multiplier is refused.
    """
    parts = library(MULTIPLIED)
    cell = parts.component("cell", m=3)
    found = {
        instance.name: instance.parameters for instance in cell.definition.instances
    }
    assert found["R1"]["resistance"] == pytest.approx(1e3)
    assert found["R2"]["resistance"] == pytest.approx(500)
    assert found["R2"]["flicker_coefficient"] == pytest.approx(1e-12 / 6)
    assert found["C1"]["capacitance"] == pytest.approx(6e-12)
    assert found["L1"]["inductance"] == pytest.approx(2e-9)
    assert found["D1"]["saturation_current"] == pytest.approx(3e-14)
    (inner,) = cell.children
    (resistor,) = inner.definition.instances
    assert resistor.parameters["resistance"] == pytest.approx(4e3 / 6)
    (own,) = parts.component("own", m=2).definition.instances
    assert own.parameters["resistance"] == pytest.approx(2e3)
    with pytest.raises(SpiceError, match="multiplies no current source by its"):
        parts.component("sourced", m=2)


GEOMETRIC: Final = (
    ".option scale=1e-6\n"
    ".model rs r rsh=100 short=0.5u narrow=0.25u tc1r=1e-3 tnom=30\n"
    ".model rd r r=7\n.model rn r\n"
    ".model cc c cox=1e-3 capsw=1e-10 narrow=0.2u del=0.1u\n"
    ".model ct c thick=1e-8 di=7.8 defw=2u\n.model cv c cap=3p cj=1\n"
    ".subckt cards a\nR1 a 0 rs w=2 l=11\nR2 a 0 rs\nR3 a 0 rs 5k scale=2 m=2\n"
    "R4 a 0 rd\nR5 a 0 rn\nC1 a 0 cc w=3 l=4\nC2 a 0 ct l=5\nC3 a 0 cv w=3 l=4\n"
    ".ends\n"
)
"""SPICE3's resistor and capacitor cards, which size their elements by geometry."""


def test_spice3_cards_size_resistors_and_capacitors_as_ngspice_does() -> None:
    """As `restemp.c` and `captemp.c` take their values, the element's sizes scaled.

    A resistor is `(L - 2 SHORT) / (W - 2 NARROW) RSH`, of 10 µm by 10 µm unless given,
    or else its card's `R`, or else 1 mΩ; a capacitor `CJ (W - NARROW) (L - SHORT) + 2
    CJSW (L - SHORT + W - NARROW)`, `DEL` the default of `SHORT` halved, `CJ` of `THICK`
    and `DI` where the card gives them, 0 long unless given, or its card's `CAP`.
    """
    parts = library(GEOMETRIC)
    found = {
        instance.name: instance.parameters
        for instance in parts.component("cards").definition.instances
    }
    resistances = [found[f"R{index}"]["resistance"] for index in range(1, 6)]
    assert resistances == pytest.approx([2e3 / 3, 900 / 9.5, 5e3, 7, 1e-3], rel=1e-14)
    assert found["R1"]["temperature_coefficient"] == 1e-3
    assert found["R1"]["nominal_temperature"] == pytest.approx(303.15)
    epsilon = 7.8 / (4e-7 * math.pi * 299792458.0**2)
    capacitances = [found[f"C{index}"]["capacitance"] for index in range(1, 4)]
    expected = [
        1e-3 * 2.8e-6 * 3.8e-6 + 2e-10 * (3.8e-6 + 2.8e-6),
        epsilon / 1e-8 * 2e-6 * 5e-6,
        3e-12,
    ]
    assert capacitances == pytest.approx(expected, rel=1e-14)


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


def test_an_inductor_whose_value_reads_a_voltage_is_refused() -> None:
    read = library(".subckt bent a b\nL1 a b {1u * (1 + abs(v(a, b)))}\n.ends")
    with pytest.raises(SpiceError, match="inductor whose value reads a voltage"):
        read.component("bent")


def test_a_diodes_level_is_its_cards_and_its_area_positive() -> None:
    read = library(
        ".model plain d level=1 is=1e-14\n.model geometric d level=2 is=1e-14\n"
        ".subckt one a b\nD1 a b plain\n.ends\n.subckt two a b\nD1 a b geometric\n"
        ".ends\n.subckt none a b\nD1 a b plain area=0\n.ends"
    )
    assert _devices(read.component("one"))["D1"][0] == "diode"
    # ngspice refuses level 2 too (`diosetup.c`, lines 38 to 42).
    with pytest.raises(SpiceError, match="a diode of level 2; jolty models 1 and 3"):
        read.component("two")
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


@dataclass(frozen=True, slots=True)
class _AsItself:
    """Model exports and reads back as itself, optionally with auxiliary elements."""

    auxiliary: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class _AsMapped:
    """Model exports by documented design as different SPICE elements."""

    models: frozenset[str]
    reason: str


@dataclass(frozen=True, slots=True)
class _AsSubcircuit:
    """Model exports as a foundry subcircuit call (`X`) rather than a primitive."""

    subcircuit: str
    stub_header: str


@dataclass(frozen=True, slots=True)
class _Refused:
    """Model has no SPICE element and is refused by name during export."""

    message: str


type _RoundTripOutcome = _AsItself | _AsMapped | _AsSubcircuit | _Refused


@dataclass(frozen=True, slots=True)
class _TerminalDrive:
    """How a bench terminal is connected: ground, resistor load, or biased source."""

    kind: str
    value: float = 0.0
    series: float = 0.0


@dataclass(frozen=True, slots=True)
class _RoundTripCase:
    """One model's round-trip fixture and expected SPICE outcome."""

    device: devices.Device
    outcome: _RoundTripOutcome
    drives: tuple[_TerminalDrive, ...] = ()
    frequency: float | None = None
    probe_index: int = 0


def _ground() -> _TerminalDrive:
    return _TerminalDrive("ground")


def _voltage(volts: float, *, series: float = 0.0) -> _TerminalDrive:
    return _TerminalDrive("voltage", volts, series)


def _load(ohms: float) -> _TerminalDrive:
    return _TerminalDrive("load", ohms)


_SPICE_ROUND_TRIP: Final[dict[str, _RoundTripCase]] = {
    "resistor": _RoundTripCase(
        devices.resistor(1 * kohm),
        _AsItself(),
        (_voltage(1.5), _ground()),
    ),
    "capacitor": _RoundTripCase(
        devices.capacitor(1 * nF),
        _AsItself(),
        (_voltage(1.5), _ground()),
        frequency=1e6,
    ),
    "inductor": _RoundTripCase(
        devices.inductor(1 * uH),
        _AsItself(),
        (_voltage(0.0, series=50.0), _ground()),
        frequency=1e6,
    ),
    "coupled_inductors": _RoundTripCase(
        devices.coupled_inductors(
            [devices.inductor(1 * uH), devices.inductor(2 * uH)],
            {(1, 2): 0.8},
        ),
        _AsItself(),
        (_voltage(0.0, series=50.0), _ground(), _load(50.0), _ground()),
        frequency=1e6,
    ),
    "transmission_line": _RoundTripCase(
        devices.transmission_line(50 * ohm, 5 * ns),
        _AsItself(),
        (_voltage(1.0, series=50.0), _ground(), _load(50.0), _ground()),
        frequency=1e7,
    ),
    "voltage_source": _RoundTripCase(
        devices.voltage_source(3.3 * V),
        _AsItself(),
        (_load(1000.0), _ground()),
    ),
    "current_source": _RoundTripCase(
        devices.current_source(1 * mA),
        _AsItself(),
        (_voltage(1.0), _ground()),
    ),
    "voltage_controlled_voltage_source": _RoundTripCase(
        devices.voltage_controlled_voltage_source(2.0),
        _AsItself(),
        (_load(1000.0), _ground(), _voltage(1.2), _ground()),
    ),
    "voltage_controlled_current_source": _RoundTripCase(
        devices.voltage_controlled_current_source(1e-3 * A / V),
        _AsItself(),
        (_voltage(0.5), _ground(), _voltage(1.2), _ground()),
    ),
    "current_controlled_voltage_source": _RoundTripCase(
        devices.current_controlled_voltage_source(500 * ohm),
        _AsItself(auxiliary=frozenset({"voltage_source"})),
        (_load(1000.0), _ground(), _voltage(1.0, series=1000.0), _ground()),
    ),
    "current_controlled_current_source": _RoundTripCase(
        devices.current_controlled_current_source(3.0),
        _AsItself(auxiliary=frozenset({"voltage_source"})),
        (_voltage(0.5), _ground(), _voltage(1.0, series=1000.0), _ground()),
    ),
    "behavioral_voltage_source": _RoundTripCase(
        devices.behavioral_voltage_source("2.0 * v(control) + 1.5"),
        _AsItself(),
        (_load(1000.0), _ground(), _voltage(0.75)),
    ),
    "behavioral_current_source": _RoundTripCase(
        devices.behavioral_current_source("1e-3 * v(positive, negative)"),
        _AsItself(),
        (_voltage(1.2), _ground()),
    ),
    "behavioral_capacitor": _RoundTripCase(
        devices.behavioral_capacitor("1e-12 * v(positive, negative)"),
        _AsItself(),
        (_voltage(1.2), _ground()),
        frequency=1e6,
    ),
    "voltage_controlled_switch": _RoundTripCase(
        devices.voltage_controlled_switch(on_voltage=2 * V, off_voltage=0.5 * V),
        _AsMapped(
            models=frozenset({"behavioral_current_source"}),
            reason=(
                "smooth cubic transition exported as behavioral current B source "
                "rather than ngspice's discontinuous S switch"
            ),
        ),
        (_voltage(1.0), _ground(), _voltage(1.3), _ground()),
    ),
    "current_controlled_switch": _RoundTripCase(
        devices.current_controlled_switch(on_current=2 * mA, off_current=0.5 * mA),
        _AsMapped(
            models=frozenset({"behavioral_current_source", "voltage_source"}),
            reason=(
                "smooth cubic transition exported as behavioral current B source "
                "with 0 V sense voltage source"
            ),
        ),
        (_voltage(1.0), _ground(), _voltage(1.3, series=1000.0), _ground()),
    ),
    "diode": _RoundTripCase(
        devices.diode(saturation_current=1e-14 * A, emission_coefficient=1.2),
        _AsItself(),
        (_voltage(0.7), _ground()),
    ),
    "bipolar_transistor": _RoundTripCase(
        devices.bipolar_transistor(saturation_current=1e-15 * A, forward_beta=100),
        _AsItself(),
        (_voltage(2.0), _voltage(0.7), _ground()),
    ),
    "vbic": _RoundTripCase(
        devices.vbic(saturation_current=1e-16 * A),
        _AsItself(),
        (_voltage(2.0), _voltage(0.7), _ground(), _ground()),
    ),
    "jfet": _RoundTripCase(
        devices.jfet(
            threshold_voltage=-2 * V, transconductance_parameter=1e-3 * A / V**2
        ),
        _AsItself(),
        (_voltage(5.0), _voltage(-1.0), _ground()),
    ),
    "mosfet": _RoundTripCase(
        devices.mosfet(
            threshold_voltage=0.7 * V,
            transconductance_parameter=1e-4 * A / V**2,
            width=10 * um,
            length=2 * um,
        ),
        _AsItself(),
        (_voltage(3.0), _voltage(2.0), _ground(), _ground()),
    ),
    "vdmos": _RoundTripCase(
        devices.vdmos(
            threshold_voltage=2 * V,
            transconductance_parameter=0.5 * A / V**2,
        ),
        _AsItself(),
        (_voltage(5.0), _voltage(4.0), _ground()),
    ),
    "bsim3": _RoundTripCase(
        devices.bsim3(channel="n", width=10 * um, length=1 * um),
        _AsItself(),
        (_voltage(1.5), _voltage(1.2), _ground(), _ground()),
    ),
    "bsim4": _RoundTripCase(
        devices.bsim4(channel="n", width=10 * um, length=1 * um),
        _AsItself(),
        (_voltage(1.5), _voltage(1.2), _ground(), _ground()),
    ),
    "foundry_mosfet": _RoundTripCase(
        sky130.mosfet("nfet_01v8", width=1 * um, length=0.15 * um),
        _AsSubcircuit(
            subcircuit="sky130_fd_pr__nfet_01v8",
            stub_header=(
                ".subckt sky130_fd_pr__nfet_01v8 d g s b "
                "L=0.15 W=1 nf=1 ad=0 as=0 pd=0 ps=0 nrd=0 nrs=0 "
                "sa=0 sb=0 sd=0 mult=1 m=1"
            ),
        ),
    ),
    "voltage_controlled_memristor": _RoundTripCase(
        devices.voltage_controlled_memristor(
            on_resistance=100 * ohm,
            off_resistance=16 * kohm,
            on_voltage=-1 * V,
            off_voltage=1 * V,
            on_rate=1e3 / s,
            off_rate=1e3 / s,
            state=0.25,
        ),
        _AsMapped(
            models=frozenset({"behavioral_current_source", "voltage_source"}),
            reason="ngspice lacks jolty's VTEAM memristor law; exported as B source",
        ),
        (_voltage(0.5), _ground()),
    ),
    "current_controlled_memristor": _RoundTripCase(
        devices.current_controlled_memristor(
            on_resistance=100 * ohm,
            off_resistance=16 * kohm,
            charge=1e-6 * A * s,
            state=0.25,
        ),
        _AsMapped(
            models=frozenset({"behavioral_current_source", "voltage_source"}),
            reason="ngspice lacks jolty's HP ion-drift memristor; exported as B source",
        ),
        (_voltage(0.5), _ground()),
    ),
    "photocell": _RoundTripCase(
        devices.photocell(10 * kohm),
        _AsMapped(
            models=frozenset({"behavioral_current_source"}),
            reason="ngspice has no native photocell; exported as B source",
        ),
        (_voltage(1.0), _ground(), _voltage(5.0), _ground()),
    ),
    "heat_source": _RoundTripCase(
        devices.heat_source(1 * W),
        _Refused("a heat_source has no SPICE element"),
    ),
    "thermal_resistance": _RoundTripCase(
        devices.thermal_resistance(50 * K / W),
        _Refused("a thermal_resistance has no SPICE element"),
    ),
    "thermal_capacitance": _RoundTripCase(
        devices.thermal_capacitance(1e-3 * J / K),
        _Refused("a thermal_capacitance has no SPICE element"),
    ),
    "logic_source": _RoundTripCase(
        devices.logic_source(0, [(1 * ns, 1)], transition=1 * ns),
        _Refused("a logic_source has no SPICE element"),
    ),
    "logic_cell": _RoundTripCase(
        devices.logic_cell(["a"], {"y": "10"}, rise=1 * ns, transition=1 * ns),
        _Refused("a logic_cell has no SPICE element"),
    ),
}
"""Every model in `Library::standard()` with its SPICE round-trip expectation."""


def _isolated_circuit(case: _RoundTripCase) -> tuple[ir.Circuit, list[str]]:
    """Minimal circuit holding one instance of `case.device` and its port nets."""
    c = Builder("isolated")
    ground = c.port("ground", Role.GROUND)
    placed = c.place(case.device, name="U1")
    terminals = case.device.terminals
    drives = case.drives or tuple(
        _ground() if index == len(terminals) - 1 else _load(1000.0)
        for index in range(len(terminals))
    )
    ports: list[str] = []
    for terminal, drive in zip(terminals, drives, strict=True):
        if drive.kind == "ground":
            c.connect(placed[terminal], ground)
        else:
            net_name = f"net_{terminal}"
            c.connect(placed[terminal], name=net_name)
            ports.append(net_name)
    return circuit(c.finish()), ports


def _driven_bench(
    subject: devices.Device | Component,
    terminals: tuple[str, ...],
    drives: tuple[_TerminalDrive, ...],
    *,
    port_prefix: str = "",
) -> tuple[Component, str, str]:
    """Wraps `subject` with sources and loads matching `drives`."""
    c = Builder("bench")
    ground = c.port("ground", Role.GROUND)
    placed = c.place(subject, name="DUT")
    if isinstance(subject, Component) and spice.GROUND in (
        port.name for port in subject.definition.ports
    ):
        c.connect(placed[spice.GROUND], ground)
    first_source = ""
    first_net = ""
    for index, (terminal, drive) in enumerate(zip(terminals, drives, strict=True)):
        pin = placed[f"{port_prefix}{terminal}"]
        if drive.kind == "ground":
            if not port_prefix:
                c.connect(pin, ground)
            continue
        net_name = f"net_{terminal}"
        if not first_net:
            first_net = net_name
        if drive.kind == "load":
            resistor = c.place(devices.resistor(drive.value * ohm), name=f"RL{index}")
            c.connect(pin, resistor.a, name=net_name)
            c.connect(resistor.b, ground)
        else:
            source_name = f"VS{index}"
            if not first_source:
                first_source = source_name
            source = c.place(devices.voltage_source(drive.value * V), name=source_name)
            c.connect(source.negative, ground)
            if drive.series > 0.0:
                series = c.place(
                    devices.resistor(drive.series * ohm), name=f"RS{index}"
                )
                c.connect(source.positive, series.a, name=f"src_{index}")
                c.connect(series.b, pin, name=net_name)
            else:
                c.connect(source.positive, pin, name=net_name)
    return c.finish(), first_source, first_net


def test_every_model_has_a_spice_round_trip_case() -> None:
    """Every model in `Library::standard()` is accounted for in `_SPICE_ROUND_TRIP`."""
    assert set(_SPICE_ROUND_TRIP) == set(_core.models())


@pytest.mark.parametrize("model_name", sorted(_SPICE_ROUND_TRIP))
def test_every_model_spice_round_trip(model_name: str) -> None:
    """Every model round-trips electrically, maps by design, or refuses by name."""
    case = _SPICE_ROUND_TRIP[model_name]
    built, ports = _isolated_circuit(case)
    outcome = case.outcome

    if isinstance(outcome, _Refused):
        with pytest.raises(JoltyError, match=re.escape(outcome.message)):
            netlist(built)
        return

    lines = netlist(built)
    body = [line for line in lines[1:] if line.strip()]
    cards = [line for line in body if line.lower().startswith(".model")]
    elements = [line for line in body if not line.lower().startswith(".model")]

    if isinstance(outcome, _AsSubcircuit):
        assert any(
            line.startswith("XU1 ") and outcome.subcircuit in line for line in elements
        ), body
        missing = "\n".join([f".subckt dut {' '.join(ports)}", *elements, ".ends"])
        with pytest.raises(SpiceError, match=re.escape(outcome.subcircuit)):
            library(missing).component("dut")
        stubbed = "\n".join(
            [
                outcome.stub_header,
                "R1 d s 1k",
                ".ends",
                f".subckt dut {' '.join(ports)}",
                *elements,
                ".ends",
            ]
        )
        dut = library(stubbed).component("dut")
        subcircuits = {
            instance.of.component.partition("[")[0]
            for instance in dut.definition.instances
            if isinstance(instance.of, ir.ComponentOf)
        }
        assert subcircuits == {outcome.subcircuit}
        return

    deck = "\n".join([*cards, f".subckt dut {' '.join(ports)}", *elements, ".ends"])
    dut = library(deck).component("dut")
    got_models = {
        instance.of.model
        for instance in dut.definition.instances
        if isinstance(instance.of, ir.DeviceOf)
    }

    if isinstance(outcome, _AsItself):
        assert got_models == {model_name, *outcome.auxiliary}
    else:
        assert got_models == outcome.models

    terminals = case.device.terminals
    original, source_name, probe_net = _driven_bench(
        case.device, terminals, case.drives
    )
    round_trip_terminals = tuple(
        terminal
        for terminal, drive in zip(terminals, case.drives, strict=True)
        if drive.kind != "ground"
    )
    round_trip_drives = tuple(drive for drive in case.drives if drive.kind != "ground")
    round_trip_bench, _, _ = _driven_bench(
        dut, round_trip_terminals, round_trip_drives, port_prefix="net_"
    )

    # Compare electrical response between original and round-tripped device.
    # Tolerance 1e-9 relative covers IEEE-754 decimal serialization rounding in SPICE
    # netlists and TIGHT Newton convergence (reltol=1e-6, abstol=1e-14).
    if case.frequency is not None:
        analysis = Ac(source_name, [case.frequency * Hz])
        expected = complex(
            Simulation(original).run(analysis, TIGHT).current(source_name)[0]
        )
        actual = complex(
            Simulation(round_trip_bench).run(analysis, TIGHT).current(source_name)[0]
        )
    elif source_name and case.drives[0].kind == "voltage":
        expected = complex(
            Simulation(original).run(OperatingPoint(), TIGHT).current(source_name)
        )
        actual = complex(
            Simulation(round_trip_bench)
            .run(OperatingPoint(), TIGHT)
            .current(source_name)
        )
    else:
        expected = complex(
            Simulation(original).run(OperatingPoint(), TIGHT).voltage(probe_net)
        )
        actual = complex(
            Simulation(round_trip_bench).run(OperatingPoint(), TIGHT).voltage(probe_net)
        )
    scale = max(abs(expected), abs(actual), 1e-12)
    assert abs(expected - actual) <= 1e-9 * scale, (
        f"{model_name}: expected {expected!r}, round-tripped {actual!r}"
    )
