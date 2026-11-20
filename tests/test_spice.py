# Copyright 2026 Maxence Faldor

import hashlib
import json
import math
import re
import struct
import threading
import zlib
from collections.abc import Mapping, MutableMapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np
import pytest
from pydantic import JsonValue

from jolty import (
    Ac,
    Builder,
    Component,
    DimensionError,
    DownloadError,
    JoltyError,
    Noise,
    OperatingPoint,
    Options,
    PeriodicSteadyState,
    Role,
    Simulation,
    SpiceError,
    Transient,
    _core,
    circuit,
    component,
    devices,
    function,
    ir,
    jvp,
    results,
    spice,
    vjp,
)
from jolty.plan import TIGHT, Mismatch
from jolty.spice import (
    _cache,
    download,
    export,
    models,
    ngspice,
    sky130,
    xyce,
)
from jolty.spice import (
    library as spice_library,
)
from jolty.spice.export import netlist
from jolty.spice.expressions import (
    Binary,
    Call,
    Conditional,
    Dual,
    Expression,
    Function,
    Name,
    Number,
    Unary,
    Voltage,
    evaluate,
    evaluate_dual,
    inlined,
    integral,
    is_finite,
    nesting_depth,
    number,
    parse,
    translate,
)
from jolty.spice.netlist import Statement, statements, tokens
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
    ms,
    nF,
    nm,
    ns,
    ohm,
    s,
    uF,
    uH,
    um,
    us,
)
from jolty.waveforms import Exponential, Piecewise, Pulse, Sine

DECIMAL_SCALE_RTOL: Final = 1e-6
"""Relative tolerance (`1e-6`) for decimal scale-suffix and round-trip rounding."""

KELVIN_ROUNDING_RTOL: Final = 1e-6
"""Relative tolerance (`1e-6`) for Celsius-to-Kelvin addition rounding."""

LINEAR_DC_RTOL: Final = 1e-6
"""Relative tolerance (`1e-6`) for linear DC LU solve rounding."""

LU_ROUNDING_RTOL: Final = 1e-12
"""Relative tolerance (`1e-12`) for exact linear LU and IEEE-754 scaling rounding."""


def library(text: str) -> spice.Library:
    read = spice.Library()
    read.read(text, "parts.lib")
    return read


def driven(part: Component, level: float, pins: tuple[str, str]) -> Component:
    """Builds a bench driving `part`'s first pin at `level` V with second pin `out`."""
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


def _diode_connected_netlist(
    transistor: devices.Device,
    supply_voltage: float,
    *,
    instance_name: str = "M1",
) -> list[str]:
    """Exports a diode-connected single-transistor bench driven by `supply_voltage`."""
    terminals = transistor.terminals
    if "drain" in terminals:
        high, net_name = ("drain", "gate"), "d"
    else:
        high, net_name = ("collector", "base"), "c"
    low = tuple(pin for pin in terminals if pin not in high)
    c = Builder("bench")
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(supply_voltage * V), name="V1")
    placed = c.place(transistor, name=instance_name)
    c.connect(supply.positive, *(placed[pin] for pin in high), name=net_name)
    c.connect(supply.negative, *(placed[pin] for pin in low), ground)
    return netlist(circuit(c.finish()))


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
    assert number(text) == pytest.approx(value, rel=DECIMAL_SCALE_RTOL)


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
    # Default pytest.approx (rel=1e-6, abs=1e-12) covers expression arithmetic rounding.
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
    assert written == "v(n_a) * i(vsense) - (-v(n_b))"


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
    assert output(part, 3.0) == pytest.approx(1.0, rel=LINEAR_DC_RTOL)
    assert output(parts.component("divider", top=1e3), 3.0) == pytest.approx(
        1.5, rel=LINEAR_DC_RTOL
    )


def test_subcircuits_place_one_another() -> None:
    parts = library(
        ".subckt half a b\nR1 a b 1k\nR2 b 0 1k\n.ends\n"
        ".subckt quarter in out\nX1 in mid half\nX2 mid out half\n.ends"
    )
    # The second divider loads the first: 4 V across 1k + (1k ∥ 2k).
    assert output(parts.component("quarter"), 5.0) == pytest.approx(
        1.0, rel=LINEAR_DC_RTOL
    )


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
    # Default pytest.approx (rel=1e-6) matches Newton voltage tolerance (vntol=1e-6 V).
    assert point.voltage("U1/a") == pytest.approx(5.0)
    assert point.voltage("U1/b") == pytest.approx(2.0)
    assert point.voltage("U1/c") == pytest.approx(10.0)
    assert point.voltage("out") == pytest.approx(2.0)


def waveforms(text: str) -> dict[str, JsonValue]:
    """Returns source waveforms from the single subcircuit in `text`, keyed by name."""
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
    # The DC level of a zero-phase sine equals the offset value at t = 0.
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
    # transconductance parameter, 1e-4 A/V², alike. Explicit abs=0 prevents default
    # abs=1e-12 from masking sub-picoampere values; rel=1e-12 covers scaling rounding.
    assert read["Q1"]["saturation_current"] == pytest.approx(
        1e-15, rel=LU_ROUNDING_RTOL, abs=0
    )
    assert read["Q2"]["saturation_current"] == pytest.approx(
        2e-16, rel=LU_ROUNDING_RTOL, abs=0
    )
    assert "saturation_current" not in read["Q3"]
    assert read["D1"]["saturation_current"] == pytest.approx(
        3e-14, rel=LU_ROUNDING_RTOL, abs=0
    )
    assert read["J1"]["saturation_current"] == pytest.approx(
        4e-14, rel=LU_ROUNDING_RTOL, abs=0
    )
    assert read["J1"]["transconductance_parameter"] == pytest.approx(
        4e-4, rel=1e-12, abs=0
    )


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
    # Level 3 area (1.5+0.5)(3.5+0.5) µm² and perimeter (2*2+2*4) µm scale
    # once by `m` (ngspice `diosetup.c` lines 336 to 337 double-multiplies).
    # `IKR=0` disables high-injection knee, and breakdown current scales with
    # area; explicit `abs=0` avoids masking femtocurrent/femtofarad entries.
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
        },
        rel=LU_ROUNDING_RTOL,
        abs=0,
    )
    # Level 1 uses card area/perimeter when unsized, or 1/0 when width is set;
    # breakdown current scales with `m` alone.
    assert read["D2"] == pytest.approx(
        {
            "saturation_current": 1e-14 * 6,
            "sidewall_saturation_current": 1e-15 * 4,
            "breakdown_current": 2e-3,
            "breakdown_voltage": 5.0,
        },
        rel=LU_ROUNDING_RTOL,
        abs=0,
    )
    assert read["D3"] == pytest.approx(
        {
            "saturation_current": 1e-14,
            "breakdown_current": 1e-3,
            "breakdown_voltage": 5.0,
        },
        rel=LU_ROUNDING_RTOL,
        abs=0,
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
    """Verifies BJT cards parse HSPICE temperature terms identically to ngspice.

    ngspice treats `TREF` as `TNOM`, ignores standalone `IBC` and HSPICE `DCAP`
    variants, applies `TLEV=3` terms only at `TLEV=3` (`bjttemp.c:166-197`),
    clamps `NKF` to 1 (`bjtsetup.c:332-338`), and treats `ISS=0` as absent.
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
        "nominal_temperature": pytest.approx(298.15, rel=KELVIN_ROUNDING_RTOL),
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
    """Verifies BJT `TLEV`, `TLEVC`, and temperature terms round-trip through SPICE."""
    given = {
        "temperature_model": 1,
        "capacitance_temperature_model": 1,
        "high_injection_exponent": 0.4,
        "collector_resistance": 10.0,
        "collector_resistance_temperature_coefficient": -0.01,
        "collector_resistance_quadratic_temperature_coefficient": 3e-5,
        "base_emitter_potential_temperature_coefficient": 2e-3,
    }
    transistor = devices.bipolar_transistor(
        temperature_model=1,
        capacitance_temperature_model=1,
        high_injection_exponent=0.4,
        collector_resistance=10 * ohm,
        collector_resistance_temperature_coefficient=-0.01 / K,
        collector_resistance_quadratic_temperature_coefficient=3e-5 / K**2,
        base_emitter_potential_temperature_coefficient=2e-3 * V / K,
    )
    lines = _diode_connected_netlist(transistor, 5.0, instance_name="Q1")
    card = next(line for line in lines if line.startswith(".model"))
    element = next(line for line in lines if line.startswith("QQ1"))
    read = library(f"{card}\n.subckt a c\n{element.replace('QQ1', 'Q1', 1)}\n.ends")
    (instance,) = read.component("a").definition.instances
    assert instance.parameters == {**given, "polarity": "npn", "substrate": "vertical"}


def admittance(part: Component, bias: float, frequency: float) -> complex:
    """Returns AC admittance into pin 0 of `part` at `bias` V with pin 1 grounded."""

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
"""Charge- and voltage-defined capacitors paired with expected capacitance formulas.

Expected capacitance equals the charge derivative or voltage expression times
multiplier `m`, with subcircuit parameter `k = 2`.
"""


@pytest.mark.parametrize("given", CHARGED)
def test_a_capacitor_of_a_charge_holds_it_as_a_behavioral_capacitor(given: str) -> None:
    """Verifies AC admittance equals `jωC` for charge- and voltage-defined capacitors.

    Voltage-dependent capacitances integrate analytically into charge expressions
    with zero DC current. The `1e-9` relative tolerance matches tight Newton
    convergence, well above closed-form rounding.
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
    """Verifies grounded capacitors treat `V(p)` identically to `V(p, 0)`."""
    parts = library(".subckt cap p\nC1 p 0 c='1e-12 * (2 + v(p))'\n.ends")
    part = parts.component("cap")
    expected = 2j * math.pi * 1e6 * 1e-12 * (2 + 0.5)
    # AC linearization at TIGHT options (vntol=1e-9 V), far above float rounding.
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
    """Verifies behavioral capacitors export `Q=` and reload identically."""
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
    # AC linearization at TIGHT options (vntol=1e-9 V) plus netlist decimal rounding.
    assert admittance(read.component("a"), 0.3, 1e6) == pytest.approx(
        expected, rel=1e-9
    )


def test_a_table_source_holds_beyond_its_ends() -> None:
    parts = library(".subckt clamp in out\nE1 out 0 TABLE {V(in)} = (0,0) (1,2)\n.ends")
    part = parts.component("clamp")
    assert output(part, 0.5) == pytest.approx(1.0, rel=LINEAR_DC_RTOL)
    assert output(part, 3.0) == pytest.approx(2.0, rel=LINEAR_DC_RTOL)


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
        (".subckt a x\nR1 x 0 1k", r"parts\.lib:1: unterminated \.subckt block"),
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
    # rel=1e-15 (~4.5 eps) covers single floating-point division nl / f rounding.
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
    """Verifies VDMOS cards parse polarity, parameters, and 3 to 5 nodes like ngspice.

    Ignores part annotations, `PHI`, thermal network parameters, and optional
    4th and 5th thermal nodes when self-heating is disabled.
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
    transistor = devices.vdmos(
        channel="p",
        threshold_voltage=-4 * V,
        drain_resistance=0.02 * ohm,
        multiplier=2,
        junction_capacitance=1e-9 * C / V,
    ).warmer_by(10 * K)
    lines = _diode_connected_netlist(transistor, -5.0, instance_name="M1")
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
    """Verifies VBIC levels 4 and 9 parse parameters and 3 to 5 nodes like ngspice.

    Clamps intrinsic collector resistance to at least 10 mΩ, ignores `VERS` and
    `VREF`, and leaves an unused fifth thermal node unconnected when unheated.
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
    """Verifies heated VBIC cards attach a thermal network when `RTH > 0`.

    Requires `SELFT != 0`. Thermal resistance and capacitance connect the fifth
    node to ground, scaled by area times multiplier `m`; ngspice floors thermal
    capacitance at 1 pJ/K.
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
    # `area * m` scales defaulted currents and inverse resistances alike,
    # matching ngspice VBIC scaling (`vbicload.c:4017-4052`).
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
    """Verifies BJT cards outside levels 1, 4, and 9 are rejected."""
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
    lines = _diode_connected_netlist(transistor, -5.0, instance_name="Q1")
    card = next(line for line in lines if line.startswith(".model"))
    assert card.startswith(".model MQ1 PNP(level=4 "), card
    element = next(line for line in lines if line.startswith("QQ1"))
    read = library(f"{card}\n.subckt a c\n{element.replace('QQ1', 'Q1', 1)}\n.ends")
    (instance,) = read.component("a").definition.instances
    assert instance.parameters == transistor.parameters


@pytest.mark.parametrize(
    ("model", "deck", "expected_instances"),
    [
        pytest.param(
            "bsim3",
            (
                ".model N1 NMOS(level=8 version=3.3.0 tnom=50 nch=2.5e17 lnch=1e16 "
                "nsub=1e17\n"
                "+ u0=400 lu0=-20 vth0=0.6 lvth0=0.02 pvth0=0.003 capmod=2 mobmod=3\n"
                "+ paramchk=1 xpart=0.5 lmin=0.1u vgs_max=4 ld=1n elm=3)\n"
                ".model P1 PMOS(level=49 binunit=2 nch=3e23 u0=0.02 lvth0=1e-8)\n"
                ".subckt a d g\nM1 d g 0 0 N1 W=2u L=0.5u nrd=2 delvto=0.1\n"
                "M2 d g 0 0 P1 L=1u off\n.ends"
            ),
            (
                (
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
                    },
                    False,
                ),
                (
                    {
                        "channel": "p",
                        "nch": 3e23,
                        "u0": 0.02,
                        "lvth0": 1e-8,
                        "length": 1e-6,
                    },
                    True,
                ),
            ),
            id="bsim3",
        ),
        pytest.param(
            "bsim4",
            (
                ".model N1 NMOS(level=54 version=4.5 tnom=25 ndep=1.7e17 lndep=1e16 "
                "nsd=1e20\n"
                "+ ngate=3e25 nsub=6e16 u0=400 lu0=-20 vtho=0.5 lvth0=0.02 rbodymod=1\n"
                "+ permod=0 paramchk=0 xpart=0.5 lmin=0.1u tnoimod=1 capmod=2 "
                "mobmod=0\n"
                "+ rgatemod=3 xgw=0.5u)\n"
                ".subckt a d g\nM1 d g 0 0 N1 W=2u L=0.15u nf=2 sa=1u sb=2u sd=0.5u\n"
                "+ nrd=2 delvto=0.1 rbpb=20 xgw=1u ngcon=2\n.ends"
            ),
            (
                (
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
                    },
                    False,
                ),
            ),
            id="bsim4",
        ),
    ],
)
def test_bsim_cards_read_in_si_units_as_ngspice_reads_them(
    model: str,
    deck: str,
    expected_instances: tuple[tuple[Mapping[str, object], bool], ...],
) -> None:
    """Verifies BSIM3 (8/49) and BSIM4 (14/54) cards convert into SI units.

    Default `pytest.approx` (`rel=1e-6`, `abs=1e-12`) covers SI unit scaling
    rounding.
    """
    instances = library(deck).component("a").definition.instances
    assert len(instances) == len(expected_instances)
    for instance, (expected, exact) in zip(instances, expected_instances, strict=True):
        assert instance.of == ir.DeviceOf(model=model)
        if exact:
            assert instance.parameters == expected
        else:
            assert instance.parameters == pytest.approx(expected)


_BSIM_REFUSALS: Final[
    list[tuple[str, str, type[Exception] | tuple[type[Exception], ...]]]
] = [
    *[
        (card, error, SpiceError)
        for card, error in (
            ("level=49 version=3.1", "`N1` is a BSIM3 card of version 3.1"),
            ("level=8 nqsmod=1", "jolty models BSIM3 with `NQSMOD` 0 alone"),
            ("level=8 acnqsmod=1", "jolty models BSIM3 with `ACNQSMOD` 0 alone"),
            ("level=8 acm=2", "jolty models BSIM3 with `ACM` 0 alone"),
            ("level=8 capmod=1.5", "`CAPMOD` of `N1` is a whole number, not 1.5"),
            ("level=8 vto=1", "no counterpart to `VTO` of `N1`"),
            ("level=8 lvto=1", "no counterpart to `LVTO` of `N1`"),
        )
    ],
    *[
        (card, error, (SpiceError, ValueError))
        for card, error in (
            ("level=54 version=4.6", "`N1` is a BSIM4 card of version 4.6"),
            ("level=14", "`N1` is a BSIM4 card of version 4.8"),
            (
                "level=54 version=4.5 capmod=1",
                "jolty models BSIM4 with `CAPMOD` 2 alone",
            ),
            (
                "level=54 version=4.5 igcmod=1",
                "jolty models BSIM4 with `IGCMOD` 0 alone",
            ),
            (
                "level=54 version=4.5 rgatemod=4",
                "gate_resistance_model is 0, 1, 2 or 3, not 4",
            ),
            ("level=54 version=4.5 rbodymod=2", "body_resistance_model 0 or 1, not 2"),
            ("level=54 version=4.5 vto=1", "no counterpart to `VTO` of `N1`"),
        )
    ],
]


@pytest.mark.parametrize(("card", "error", "expected_error"), _BSIM_REFUSALS)
def test_bsim_cards_jolty_cannot_run_as_ngspice_does_are_refused(
    card: str,
    error: str,
    expected_error: type[Exception] | tuple[type[Exception], ...],
) -> None:
    parts = library(f".model N1 NMOS({card})\n.subckt a d\nM1 d d 0 0 N1\n.ends")
    with pytest.raises(expected_error, match=re.escape(error)):
        parts.component("a")


@pytest.mark.parametrize(
    ("transistor", "supply", "expected_card", "expected_element", "expected_params"),
    [
        pytest.param(
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
            3.0,
            (
                ".model MM1 PMOS(level=8 version=3.3.0 binunit=2 tnom=50.0 capmod=1 "
                "paramchk=1 vth0=-0.6 nch=1.7e+17 lnch=1e+30 nsub=6e+16 lvth0=1e-08 "
                "k2=-0.02)"
            ),
            "MM1 d d 0 0 MM1 w=2e-06 l=5e-07",
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
            },
            id="bsim3",
        ),
        pytest.param(
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
            1.8,
            (
                ".model MM1 PMOS(level=54 version=4.5 binunit=2 tnom=50.0 rbodymod=1 "
                "vth0=-0.5 ndep=1.7e+17 ngate=1e+23 nsub=6e+16 lvth0=1e-08 k2=-0.02)"
            ),
            "MM1 d d 0 0 MM1 w=2e-06 l=1.5e-07 nf=2.0",
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
            },
            id="bsim4",
        ),
    ],
)
def test_bsim_is_written_for_ngspice_as_its_card_reads_back(
    transistor: devices.Device,
    supply: float,
    expected_card: str,
    expected_element: str,
    expected_params: Mapping[str, object],
) -> None:
    lines = _diode_connected_netlist(transistor, supply, instance_name="M1")
    card = next(line for line in lines if line.startswith(".model"))
    assert card == expected_card, card
    element = next(line for line in lines if line.startswith("MM1"))
    assert element == expected_element, element
    read = library(
        f"{card}\n.subckt a d\n{element.replace('MM1 d d', 'M1 d d', 1)}\n.ends"
    )
    (instance,) = read.component("a").definition.instances
    # Decimal card formatting and SI scaling (1e-6 / 1e6) round within default 1e-6.
    assert instance.parameters == pytest.approx(expected_params)


def test_a_bsim3_held_at_its_own_temperature_cannot_be_exported() -> None:
    """Verifies BSIM3 instances with per-instance temperature are refused on export."""
    with pytest.raises(JoltyError, match="at its own temperature"):
        _diode_connected_netlist(devices.bsim3().held_at(celsius(85)), 3.0)


def _assert_matches_engine_schema(model: str, named: set[str]) -> None:
    schema = json.loads(_core.schema(model))
    unread = {"temperature", "temperature_offset", "thermal"}
    assert named == set(schema["properties"]) - unread


@pytest.mark.parametrize("model", models.BSIMS)
def test_a_bsims_tables_name_the_engines_parameters(model: str) -> None:
    """Verifies BSIM parser tables cover all engine model parameters.

    Card, bin, and instance parameters plus choices and settings match every
    engine BSIM parameter except per-instance temperature and self-heating.
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
    _assert_matches_engine_schema(model, named)


@pytest.mark.parametrize(
    ("model", "named"),
    [
        (
            "diode",
            {*(jolty for jolty, _ in models.DIODE.values()), "nominal_temperature"},
        ),
        (
            "jfet",
            {
                *(jolty for jolty, _ in models.JFET.values()),
                "channel",
                "nominal_temperature",
            },
        ),
        (
            "bipolar_transistor",
            {
                *(jolty for jolty, _ in models.BIPOLAR.values()),
                "polarity",
                "substrate",
                "nominal_temperature",
                *models.BIPOLAR_TEMPERATURE_MODELS.values(),
            },
        ),
        (
            "mosfet",
            {
                *(jolty for jolty, _ in models.MOSFET.values()),
                *(jolty for jolty, _ in models.MOSFET_INSTANCE.values()),
                "channel",
                "level",
                "gate_material",
                "nominal_temperature",
            },
        ),
        (
            "vdmos",
            {
                *(jolty for jolty, _ in models.VDMOS.values()),
                *(jolty for jolty, _ in models.VDMOS_INSTANCE.values()),
                "channel",
                "nominal_temperature",
            },
        ),
        (
            "vbic",
            {
                *(jolty for jolty, _ in models.VBIC.values()),
                "polarity",
                "base_charge",
                "nominal_temperature",
            },
        ),
        (
            "voltage_controlled_switch",
            {jolty for jolty, _ in models.SWITCH.values()},
        ),
        (
            "current_controlled_switch",
            {jolty for jolty, _ in models.CURRENT_SWITCH.values()},
        ),
        (
            "resistor",
            {
                "resistance",
                "nominal_temperature",
                *models.PASSIVE_DRIFT.values(),
                *models.PASSIVE_CURVATURE.values(),
                *(jolty for jolty, _ in models.FLICKER.values()),
            },
        ),
        (
            "capacitor",
            {
                "capacitance",
                "nominal_temperature",
                models.PASSIVE_DRIFT["tc1"],
                models.PASSIVE_DRIFT["tc2"],
                *models.PASSIVE_CURVATURE.values(),
            },
        ),
        (
            "inductor",
            {
                "inductance",
                "nominal_temperature",
                models.PASSIVE_DRIFT["tc1"],
                models.PASSIVE_DRIFT["tc2"],
            },
        ),
    ],
)
def test_compact_switch_and_passive_tables_match_engine_schema(
    model: str, named: set[str]
) -> None:
    """Verifies compact, switch, and passive tables match engine schema properties."""
    _assert_matches_engine_schema(model, named)


def test_model_tables_have_canonical_entries_and_valid_scaling_subsets() -> None:
    """Verifies model tables define valid SI factors and consistent scaling subsets."""
    tables = (
        models.DIODE,
        models.JFET,
        models.BIPOLAR,
        models.MOSFET,
        models.MOSFET_INSTANCE,
        models.VDMOS,
        models.VDMOS_INSTANCE,
        models.VBIC,
        models.SWITCH,
        models.CURRENT_SWITCH,
    )
    for table in tables:
        assert table
        for spice_name, (jolty_name, factor) in table.items():
            assert spice_name == spice_name.casefold()
            assert spice_name
            assert jolty_name
            assert factor != 0.0
            assert math.isfinite(factor)
    assert set(models.DIODE_BY_AREA) <= set(models.DIODE)
    assert set(models.DIODE_BY_PERIMETER) <= set(models.DIODE)
    assert set(models.VBIC_MULTIPLIED) <= set(models.VBIC)
    assert set(models.VBIC_RESISTANCES) <= set(models.VBIC)


def test_an_element_takes_the_bin_of_its_length_and_width() -> None:
    """Verifies binned model selection matches ngspice `INPgetModBin` (last match).

    Each bin is a numbered card whose length and width ranges include the
    element dimensions within a 1 nm boundary tolerance.
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
    """Verifies bin selection uses per-finger width `W / nf` like `INPgetModBin`.

    Applies under HSPICE compatibility mode or when the `wnflag` option is set.
    """
    parts = spice.Library(compatibility=compatibility)
    parts.read(option + FINGERED, "fingers.lib")
    (instance,) = parts.component("fingers").definition.instances
    assert instance.parameters["vth0"] == threshold


def test_the_scale_option_scales_a_bsim4s_sizes_and_distances() -> None:
    """Verifies `scale` multiplies BSIM4 lengths linearly and areas quadratically.

    Scales length, width, perimeters, and diffusion distances like `b4v5par.c`.
    `rel=1e-15, abs=0` (~4.5 eps) covers `scale` and `scale**2` rounding
    without default `abs=1e-12` masking `drain_area=1e-12`.
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
        abs=0,
    )


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
        "nominal_temperature": pytest.approx(323.15, rel=KELVIN_ROUNDING_RTOL),
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
    assert hot.parameters == {
        "temperature": pytest.approx(353.15, rel=KELVIN_ROUNDING_RTOL)
    }
    assert capacitor.parameters == {"capacitance": 1e-9, "temperature_offset": -10.0}
    assert transistor.parameters["temperature"] == pytest.approx(
        323.15, rel=KELVIN_ROUNDING_RTOL
    )
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
    """Verifies exported behavioral expressions match jolty operator and call semantics.

    Handles parentheses, `^` associativity, negative-base powers, `limit`, and
    piecewise tables where ngspice differs from jolty.
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


def test_a_resistor_of_a_voltage_tracks_simulation_temperature_dynamically() -> None:
    """Verifies voltage-dependent resistors lower to `behavioral_resistor`.

    Sweeps a single built circuit across -40, 27, 77, and 125 °C without rebuilding the
    circuit or SPICE library, checking `I = m * V / (R(V) * (1 + TC1*dT + TC2*dT^2))`.
    """
    parts = spice.Library()
    parts.read(
        ".subckt grown a\n"
        "R1 a 0 r={1k * (1 + 0.1 * abs(v(a)))} tc1=1e-3 tc2=2e-5 m=2\n"
        ".ends",
        "grown.lib",
    )
    (instance,) = parts.component("grown").definition.instances
    assert instance.of == ir.DeviceOf(model="behavioral_resistor")

    @component
    def bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(2 * V), name="V1")
        placed = c.place(parts.component("grown"), name="X1")
        c.connect(source.positive, placed["a"], name="a")
        c.connect(source.negative, placed["ground"], ground)

    built = bench()
    for degrees in (-40.0, 27.0, 77.0, 125.0):
        simulation = Simulation(built, temperature=celsius(degrees))
        point = simulation.run(OperatingPoint())
        current = point.values[
            simulation.unknowns.index(results.Unknown("V1:current", "current"))
        ]
        delta = degrees - 27.0
        factor = 1.0 + 1e-3 * delta + 2e-5 * delta * delta
        # Single-iteration solve at fixed V=2 V; rel=1e-12 (~4500 eps) covers rounding.
        assert -current == pytest.approx(4 / (1.2e3 * factor), rel=1e-12)


def test_behavioral_resistor_noise_and_derivatives_across_bias_and_temperature() -> (
    None
):
    """Verifies `behavioral_resistor` thermal noise `4kT / R(V, T)` and JVP/VJP.

    Checks both direct `devices.behavioral_resistor` construction and SPICE-imported
    `R={...}` elements with `tc1`, `tc2`, `tnom`, and `dtemp` across biases and
    temperatures.
    """
    nominal_degrees = 25.0
    linear_coefficient = 2e-3
    quadratic_coefficient = 1e-5
    parts = spice.Library()
    parts.read(
        ".subckt nonlinear a\n"
        "R1 a 0 r={1k * (1 + 0.2 * v(a))} tc1=2e-3 tc2=1e-5 tnom=25\n"
        ".ends",
        "nonlinear.lib",
    )

    for bias in (0.0, 2.0, 5.0):

        @component
        def direct_bench(c: Builder, voltage: float = bias) -> None:
            ground = c.port("ground", Role.GROUND)
            source = c.place(devices.voltage_source(voltage * V), name="V1")
            resistor = c.place(
                devices.behavioral_resistor(
                    "1e3 * (1 + 0.2 * v(positive, negative))",
                    temperature_coefficient=linear_coefficient / K,
                    quadratic_temperature_coefficient=quadratic_coefficient / K**2,
                    nominal_temperature=celsius(nominal_degrees),
                ),
                name="BR1",
            )
            c.connect(source.positive, resistor.positive, name="a")
            c.connect(source.negative, resistor.negative, ground)

        @component
        def spice_bench(c: Builder, voltage: float = bias) -> None:
            ground = c.port("ground", Role.GROUND)
            source = c.place(devices.voltage_source(voltage * V), name="V1")
            placed = c.place(parts.component("nonlinear"), name="X1")
            c.connect(source.positive, placed["a"], name="a")
            c.connect(source.negative, placed["ground"], ground)

        direct_built = direct_bench()
        spice_built = spice_bench()
        for degrees in (-40.0, 27.0, 125.0):
            temperature = celsius(degrees)
            delta = degrees - nominal_degrees
            factor = (
                1.0 + linear_coefficient * delta + quadratic_coefficient * delta * delta
            )
            effective = 1e3 * (1.0 + 0.2 * bias) * factor
            expected_psd = 4.0 * _core.BOLTZMANN * temperature.value / effective
            for candidate in (direct_built, spice_built):
                spectrum = Simulation(candidate, temperature=temperature).run(
                    Noise("V1:current", [1e3 * Hz]), TIGHT
                )
                # Linear adjoint noise solve at fixed bias; rel=1e-9 covers rounding.
                assert spectrum.densities[0] == pytest.approx(expected_psd, rel=1e-9)

    hot = celsius(125.0)

    @component
    def differentiable(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(2 * V), name="V1")
        resistor = c.place(
            devices.behavioral_resistor(
                "1e3 * (1 + 0.2 * v(positive, negative))",
                temperature_coefficient=linear_coefficient / K,
                quadratic_temperature_coefficient=quadratic_coefficient / K**2,
                nominal_temperature=celsius(nominal_degrees),
            ).held_at(hot),
            name="BR1",
        )
        c.connect(source.positive, resistor.positive, name="a")
        c.connect(source.negative, resistor.negative, ground)

    delta = hot.value - celsius(nominal_degrees).value
    factor = 1.0 + linear_coefficient * delta + quadratic_coefficient * delta * delta
    nominal_resistance = 1e3 * (1.0 + 0.2 * 2.0)
    # Current leaving V1 positive terminal is -I_resistor = -2 / (R_expr * factor).
    d_factor_d_nominal = -(linear_coefficient + 2.0 * quadratic_coefficient * delta)
    expected_partials = {
        "temperature_coefficient": 2.0 * delta / (nominal_resistance * factor**2),
        "quadratic_temperature_coefficient": (
            2.0 * delta**2 / (nominal_resistance * factor**2)
        ),
        "nominal_temperature": (
            2.0 * d_factor_d_nominal / (nominal_resistance * factor**2)
        ),
    }
    point_parameters = {
        "BR1": {
            "temperature_coefficient": linear_coefficient,
            "quadratic_temperature_coefficient": quadratic_coefficient,
            "nominal_temperature": celsius(nominal_degrees).value,
        }
    }
    tangents = {
        "BR1": {
            "temperature_coefficient": 1e-4,
            "quadratic_temperature_coefficient": 1e-6,
            "nominal_temperature": 0.5,
        }
    }
    evaluated = function(differentiable(), OperatingPoint())
    outputs, pushed = jvp(evaluated, point_parameters, tangents)
    same, pull = vjp(evaluated, point_parameters)
    assert outputs["V1:current"] == same["V1:current"]
    pulled = pull({"V1:current": 1.0})
    for name, expected_partial in expected_partials.items():
        # Exact forward/adjoint derivative against closed form; rel=1e-11 covers LU.
        assert pulled["BR1"][name] == pytest.approx(expected_partial, rel=1e-11)
    expected_push = sum(
        expected_partials[name] * tangents["BR1"][name] for name in expected_partials
    )
    # Exact directional derivative against closed form; rel=1e-11 covers LU rounding.
    assert pushed["V1:current"] == pytest.approx(expected_push, rel=1e-11)


def test_behavioral_resistor_spice_export_preserves_drift_and_temperature() -> None:
    """Verifies SPICE export and re-import preserve `behavioral_resistor` settings."""

    @component
    def held(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(2 * V), name="V1")
        resistor = c.place(
            devices.behavioral_resistor(
                "1e3 * (1 + 0.1 * v(positive, negative))",
                temperature_coefficient=2e-3 / K,
                quadratic_temperature_coefficient=1e-5 / K**2,
            ).warmer_by(15 * K),
            name="R1",
        )
        c.connect(source.positive, resistor.positive, name="a")
        c.connect(source.negative, resistor.negative, ground)

    lines = netlist(circuit(held()))
    (resistor_line,) = [line for line in lines if line.startswith("RR1 ")]
    assert "R={" in resistor_line
    assert "tc1=0.002" in resistor_line
    assert "tc2=1e-05" in resistor_line
    assert "dtemp=" in resistor_line
    with pytest.raises(JoltyError, match="behavioral_resistor"):
        netlist(circuit(held()), dialect=export.XYCE)

    deck = f".subckt dut a\n{resistor_line}\n.ends"
    reimported = library(deck).component("dut")
    (instance,) = reimported.definition.instances
    assert instance.of == ir.DeviceOf(model="behavioral_resistor")
    # Default pytest.approx covers decimal netlist formatting round-trip rounding.
    assert instance.parameters["temperature_coefficient"] == pytest.approx(2e-3)
    # Default pytest.approx covers decimal netlist formatting round-trip rounding.
    assert instance.parameters["quadratic_temperature_coefficient"] == pytest.approx(
        1e-5
    )
    # SPICE dtemp conversion scales by thermal-voltage ratio vs ngspice CODATA 2014.
    assert instance.parameters["temperature_offset"] == pytest.approx(15.0, rel=1e-5)


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
        },
        rel=DECIMAL_SCALE_RTOL,
    )
    assert capacitor.parameters == pytest.approx(
        {
            "capacitance": 1e-9,
            "voltage_coefficient": -1e-3,
            "quadratic_voltage_coefficient": 4e-6,
        },
        rel=DECIMAL_SCALE_RTOL,
    )
    with pytest.raises(SpiceError, match="jolty reads no `VC1` here"):
        library(".subckt a x\nL1 x 0 1u VC1=1m\n.ends").component("a")


def _every_model() -> list[devices.Device]:
    """Builds a circuit containing one instance of every device model in the engine."""
    rate = 1e6 / s
    return [
        devices.behavioral_capacitor("1e-12 * tanh(v(positive, negative))"),
        devices.behavioral_current_source("1e-3 * v(a, b)"),
        devices.behavioral_resistor(
            "1e3 * (1 + 0.1 * v(positive, negative))",
            temperature_coefficient=1e-3 / K,
        ),
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
    """Returns whether `defaulted` contains `given` after applying nested defaults."""
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
    """Returns `(model, parameters)` for every device instance in `part`, by name."""
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
    """Verifies subcircuit multiplier `m` propagates like `inp_fix_subckt_multiplier`.

    Three copies of `cell` divide resistances and inductance by 3, multiply
    capacitance and diode parameters by 3, and combine nested multipliers
    (`resnoise.c` flicker scaling `m*KF*(I/m)^AF`). A subcircuit declaring
    parameter `m` binds it locally, and current sources reject `m`.
    `rel=1e-12, abs=0` covers IEEE-754 scaling rounding without masking `3e-14`.
    """
    parts = library(MULTIPLIED)
    cell = parts.component("cell", m=3)
    found = {
        instance.name: instance.parameters for instance in cell.definition.instances
    }
    assert found["R1"]["resistance"] == pytest.approx(1e3, rel=1e-12, abs=0)
    assert found["R2"]["resistance"] == pytest.approx(500, rel=1e-12, abs=0)
    assert found["R2"]["flicker_coefficient"] == pytest.approx(
        1e-12 / 6, rel=1e-12, abs=0
    )
    assert found["C1"]["capacitance"] == pytest.approx(6e-12, rel=1e-12, abs=0)
    assert found["L1"]["inductance"] == pytest.approx(2e-9, rel=1e-12, abs=0)
    assert found["D1"]["saturation_current"] == pytest.approx(3e-14, rel=1e-12, abs=0)
    (inner,) = cell.children
    (resistor,) = inner.definition.instances
    assert resistor.parameters["resistance"] == pytest.approx(4e3 / 6, rel=1e-12, abs=0)
    (own,) = parts.component("own", m=2).definition.instances
    assert own.parameters["resistance"] == pytest.approx(2e3, rel=1e-12, abs=0)
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
    """Verifies SPICE3 resistor and capacitor geometry sizing matches ngspice.

    Resistors compute `(L - 2*SHORT) / (W - 2*NARROW) * RSH` (defaulting to
    10 µm square, card `R`, or 1 mΩ). Capacitors compute
    `CJ*(W - NARROW)*(L - SHORT) + 2*CJSW*(L - SHORT + W - NARROW)` with `DEL`
    halved for `SHORT` and `CJ` derived from `THICK` and `DI`.
    """
    parts = library(GEOMETRIC)
    found = {
        instance.name: instance.parameters
        for instance in parts.component("cards").definition.instances
    }
    resistances = [found[f"R{index}"]["resistance"] for index in range(1, 6)]
    # rel=1e-14 (~45 eps) covers geometry-formula floating-point arithmetic rounding.
    assert resistances == pytest.approx([2e3 / 3, 900 / 9.5, 5e3, 7, 1e-3], rel=1e-14)
    assert found["R1"]["temperature_coefficient"] == 1e-3
    assert found["R1"]["nominal_temperature"] == pytest.approx(
        303.15, rel=KELVIN_ROUNDING_RTOL
    )
    epsilon = 7.8 / (4e-7 * math.pi * 299792458.0**2)
    capacitances = [found[f"C{index}"]["capacitance"] for index in range(1, 4)]
    expected = [
        1e-3 * 2.8e-6 * 3.8e-6 + 2e-10 * (3.8e-6 + 2.8e-6),
        epsilon / 1e-8 * 2e-6 * 5e-6,
        3e-12,
    ]
    # rel=1e-14 (~45 eps) covers geometry-formula and vacuum-permittivity rounding.
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
        2.27e3, rel=KELVIN_ROUNDING_RTOL
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
    assert read["B1"]["expression"] == "2.0 * v(a)"


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
    """Bench terminal connection specification: ground, load resistor, or source."""

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
    "behavioral_resistor": _RoundTripCase(
        devices.behavioral_resistor(
            "1e3 * (1 + 0.1 * v(positive, negative))",
            temperature_coefficient=1e-3 / K,
        ),
        _AsItself(),
        (_voltage(1.2), _ground()),
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
    """Builds a minimal circuit holding one instance of `case.device` and port nets."""
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
    """Verifies every model in `Library::standard()` appears in `_SPICE_ROUND_TRIP`."""
    assert set(_SPICE_ROUND_TRIP) == set(_core.models())


@pytest.mark.parametrize("model_name", sorted(_SPICE_ROUND_TRIP))
def test_every_model_spice_round_trip(model_name: str) -> None:
    """Verifies every model round-trips electrically, maps by design, or refuses."""
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
    # 1e-9 relative covers SPICE decimal serialization rounding and TIGHT convergence.
    assert abs(expected - actual) <= 1e-9 * scale, (
        f"{model_name}: expected {expected!r}, round-tripped {actual!r}"
    )


def test_library_caches_invalidate_on_mutations_and_temperature() -> None:
    """Verifies library caches stay coherent across mutations and temperatures.

    Every `rel=1e-15` (~4.5 eps) check covers exact expression arithmetic rounding.
    """
    lib = library(
        ".param base=1k\n"
        ".param scaled={base + 10 * temper}\n"
        ".param noisy={agauss(base, 50, 1)}\n"
        ".param derived={2 * noisy}\n"
        ".func boost(x) {x}\n"
        ".subckt load a\n"
        "R1 a 0 {boost(scaled)}\n"
        ".ends"
    )
    first_nominal = lib.values()
    assert first_nominal["scaled"] == pytest.approx(1270.0, rel=1e-15)
    assert first_nominal["derived"] == pytest.approx(2000.0, rel=1e-15)
    assert _devices(lib.component("load"))["R1"][1]["resistance"] == pytest.approx(
        1270.0, rel=1e-15
    )

    # Distinct samples genuinely draw distinct values for stochastic globals and their
    # transitive dependents, while repeating a sample is bit-identical.
    sample_a = spice.Sample(7, 0)
    sample_b = spice.Sample(7, 1)
    drawn_a = lib.values(sample_a)
    drawn_b = lib.values(sample_b)
    assert drawn_a["noisy"] != drawn_b["noisy"]
    assert drawn_a["derived"] == pytest.approx(2.0 * drawn_a["noisy"], rel=1e-15)
    assert drawn_b["derived"] == pytest.approx(2.0 * drawn_b["noisy"], rel=1e-15)
    assert lib.values(sample_a) == drawn_a

    # Adding a .param invalidates plan, scopes and built nominal components.
    lib.read(".param base=2k", "override.lib")
    assert lib.values()["scaled"] == pytest.approx(2270.0, rel=1e-15)
    assert _devices(lib.component("load"))["R1"][1]["resistance"] == pytest.approx(
        2270.0, rel=1e-15
    )
    assert lib.values(sample_a)["noisy"] != drawn_a["noisy"]

    # Temperature change invalidates both values() and nominal component().
    lib.temperature = 327.15
    assert lib.values()["scaled"] == pytest.approx(2540.0, rel=1e-15)
    assert _devices(lib.component("load"))["R1"][1]["resistance"] == pytest.approx(
        2540.0, rel=1e-15
    )

    # Mutating library.functions directly invalidates plan and built components.
    lib.functions["boost"] = Function(("x",), parse("2 * x"))
    assert _devices(lib.component("load"))["R1"][1]["resistance"] == pytest.approx(
        5080.0, rel=1e-15
    )

    # Adding a .func via read() invalidates as well.
    lib.read(".func boost(x) {3 * x}", "fn.lib")
    assert _devices(lib.component("load"))["R1"][1]["resistance"] == pytest.approx(
        7620.0, rel=1e-15
    )


def test_ako_local_models_are_not_crosstalked_by_model_cache() -> None:
    """Verifies derived `AKO:` cards resolve against the active local base card.

    `rel=1e-15, abs=0` (~4.5 eps) covers IEEE-754 decimal literal parsing rounding
    without default `abs=1e-12` masking `1e-14` or `1e-12`.
    """
    lib = library(
        ".subckt switched a k sel=0\n"
        ".if (sel == 0)\n"
        ".model base D(IS=1e-14 N=1.0)\n"
        ".else\n"
        ".model base D(IS=1e-12 N=1.5)\n"
        ".endif\n"
        ".model derived ako:base D(RS=5)\n"
        "D1 a k derived\n"
        ".ends"
    )
    zero = _devices(lib.component("switched", sel=0))["D1"][1]
    one = _devices(lib.component("switched", sel=1))["D1"][1]
    assert zero["saturation_current"] == pytest.approx(1e-14, rel=1e-15, abs=0)
    assert zero["emission_coefficient"] == pytest.approx(1.0, rel=1e-15, abs=0)
    assert one["saturation_current"] == pytest.approx(1e-12, rel=1e-15, abs=0)
    assert one["emission_coefficient"] == pytest.approx(1.5, rel=1e-15, abs=0)


def test_model_card_parameters_are_read_only() -> None:
    """Verifies cached `ModelCard` parameter mappings are read-only proxy views."""
    (statement,) = statements(".model d1 D(IS=1e-14)", "card.lib")
    card = models.read(statement, {})
    assert isinstance(card.parameters, MappingProxyType)
    assert not isinstance(card.parameters, MutableMapping)
    assert not hasattr(card.parameters, "__setitem__")


def test_drawn_is_thread_safe_and_bit_identical_under_contention() -> None:
    """Verifies concurrent `drawn` calls sharing a `Library` cache match bit-for-bit."""
    lib = library(
        ".param g={agauss(1k, 100, 1)}\n"
        ".subckt stage a\n"
        ".model dloc D(IS=1e-14 RS=10)\n"
        ".param r={agauss(g, 50, 1)}\n"
        "R1 a 0 {r}\n"
        "D1 a 0 dloc\n"
        ".ends"
    )
    expected = {
        index: _devices(lib.drawn("stage", spice.Sample(3, index), "X1"))["R1"][1][
            "resistance"
        ]
        for index in range(16)
    }
    with ThreadPoolExecutor(max_workers=8) as pool:
        results_list = list(
            pool.map(
                lambda idx: (
                    idx,
                    _devices(lib.drawn("stage", spice.Sample(3, idx), "X1"))["R1"][1][
                        "resistance"
                    ],
                ),
                list(range(16)) * 4,
            )
        )
    for idx, value in results_list:
        assert value == expected[idx]


def test_evaluate_fast_path_handles_literals_unary_and_shadowed_constants() -> None:
    """Verifies fast-path evaluation matches `_core.constant` on all edge cases."""
    assert evaluate(Number(3.5), {}) == 3.5
    assert evaluate(Unary("-", Number(2.25)), {}) == -2.25
    assert evaluate(Unary("+", Number(2.25)), {}) == 2.25
    assert evaluate(Name("pi"), {}) == math.pi
    assert evaluate(Name("pi"), {"pi": 4.0}) == 4.0
    # Shadowing a built-in constant with non-finite value must NOT fall back to pi.
    with pytest.raises(SpiceError):
        evaluate(Name("pi"), {"pi": float("inf")})
    with pytest.raises(SpiceError):
        evaluate(Number(float("nan")), {})


# `.include` and `.lib` expansion inside subcircuits.


def test_include_and_lib_expand_inside_subcircuits_from_file_and_text(
    tmp_path: Path,
) -> None:
    parts_dir = tmp_path / "my parts"
    parts_dir.mkdir()
    nested = parts_dir / "nested"
    nested.mkdir()

    (nested / "leaf.inc").write_text(
        "* Nested relative include\n"
        ".func boost(x) {x + 1k}\n"
        ".param r_bottom = boost(2k)\n",
        encoding="utf-8",
    )
    (parts_dir / "body.inc").write_text(
        "* Body inside subcircuit\n"
        ".include nested/leaf.inc\n"
        ".model d_inc D(IS=1e-14 N=1.0)\n"
        "R2 out 0 {r_bottom}\n",
        encoding="utf-8",
    )
    (tmp_path / "sections.lib").write_text(
        ".lib fast\n"
        ".param r_top = 1k\n"
        ".endl fast\n"
        ".lib slow\n"
        ".param r_top = 3k\n"
        ".endl slow\n",
        encoding="utf-8",
    )
    deck = tmp_path / "top.cir"
    deck.write_text(
        "* Top deck with .include and .lib inside subcircuits\n"
        ".subckt fast_div in out\n"
        ".lib sections.lib fast\n"
        '.include "my parts/body.inc"\n'
        "R1 in out {r_top}\n"
        ".ends\n"
        ".subckt slow_div in out\n"
        '.lib "sections.lib" slow\n'
        '.include "my parts/body.inc"\n'
        "R1 in out {r_top}\n"
        ".ends\n",
        encoding="utf-8",
    )

    via_include = spice.load(deck)
    via_read = spice.Library()
    via_read.read(deck.read_text(encoding="utf-8"), str(deck))
    for loaded in (via_include, via_read):
        assert output(loaded.component("fast_div"), 4.0) == pytest.approx(
            3.0, rel=LU_ROUNDING_RTOL, abs=0
        )
        assert output(loaded.component("slow_div"), 4.0) == pytest.approx(
            2.0, rel=LU_ROUNDING_RTOL, abs=0
        )


def test_include_inside_if_block_within_subcircuit(tmp_path: Path) -> None:
    (tmp_path / "lo.inc").write_text("R2 out 0 1k\n", encoding="utf-8")
    (tmp_path / "hi.inc").write_text("R2 out 0 3k\n", encoding="utf-8")
    deck = tmp_path / "cond.cir"
    deck.write_text(
        ".subckt stage in out mode=0\n"
        "R1 in out 1k\n"
        ".if (mode == 0)\n"
        ".include lo.inc\n"
        ".else\n"
        ".include hi.inc\n"
        ".endif\n"
        ".ends\n",
        encoding="utf-8",
    )
    lib = spice.load(deck)
    assert output(lib.component("stage", mode=0), 4.0) == pytest.approx(
        2.0, rel=LU_ROUNDING_RTOL, abs=0
    )
    assert output(lib.component("stage", mode=1), 4.0) == pytest.approx(
        3.0, rel=LU_ROUNDING_RTOL, abs=0
    )


def test_malformed_two_word_lib_inside_subcircuit_is_refused(tmp_path: Path) -> None:
    deck = tmp_path / "bad.cir"
    deck.write_text(
        ".subckt stage in out\n.lib orphan_section\nR1 in out 1k\n.ends\n",
        encoding="utf-8",
    )
    with pytest.raises(
        SpiceError,
        match="a file is included by its path, a section by the file's and name",
    ):
        spice.load(deck)
    read_lib = spice.Library()
    with pytest.raises(
        SpiceError,
        match="a file is included by its path, a section by the file's and name",
    ):
        read_lib.read(deck.read_text(encoding="utf-8"), str(deck))


def test_include_and_lib_cycles_inside_subcircuits_are_refused(tmp_path: Path) -> None:
    self_file = tmp_path / "self.cir"
    self_file.write_text(
        ".subckt loop a\n.include self.cir\nR1 a 0 1k\n.ends\n",
        encoding="utf-8",
    )
    with pytest.raises(SpiceError, match="includes itself"):
        spice.load(self_file)
    read_lib = spice.Library()
    with pytest.raises(SpiceError, match="includes itself"):
        read_lib.read(self_file.read_text(encoding="utf-8"), str(self_file))

    first = tmp_path / "a.inc"
    second = tmp_path / "b.inc"
    first.write_text(".include b.inc\n", encoding="utf-8")
    second.write_text(".include a.inc\n", encoding="utf-8")
    host = tmp_path / "host.cir"
    host.write_text(".subckt top a\n.include a.inc\n.ends\n", encoding="utf-8")
    with pytest.raises(SpiceError, match="includes itself"):
        spice.load(host)

    sec_lib = tmp_path / "cycle.lib"
    sec_lib.write_text(
        ".lib sec_a\n.lib cycle.lib sec_b\n.endl sec_a\n"
        ".lib sec_b\n.lib cycle.lib sec_a\n.endl sec_b\n",
        encoding="utf-8",
    )
    host_sec = tmp_path / "host_sec.cir"
    host_sec.write_text(
        ".subckt top a\n.lib cycle.lib sec_a\n.ends\n", encoding="utf-8"
    )
    with pytest.raises(SpiceError, match="includes itself"):
        spice.load(host_sec)


def test_coverage_lib_and_include_probe_decks_solve_in_subcircuit() -> None:
    decks = Path(__file__).parents[1] / "benchmarks" / "coverage" / "decks"
    lib_section = decks / "sections.lib"
    inc_file = decks / "included.cir"

    for directive in (f'.lib "{lib_section}" low', f'.include "{inc_file}"'):
        lib = spice.Library()
        lib.read(
            f".subckt deck\n"
            f"{directive}\n"
            f"V1 in 0 4\n"
            f"R1 in out 3k\n"
            f"R2 out 0 {{lower}}\n"
            f".ends\n",
            "fixture.cir",
        )
        point = Simulation(circuit(lib.component("deck"))).run(OperatingPoint())
        assert point.voltage("out") == pytest.approx(2.0, rel=LU_ROUNDING_RTOL, abs=0)


# `.func` definitions inside subcircuits and model cards.


def test_local_func_scopes_in_subcircuits_and_elements() -> None:
    """Verifies local `.func` scopes shadow globals without leaking out.

    `rel=1e-15, abs=0` (~4.5 eps) covers IEEE-754 expression scaling rounding.
    """
    lib = library(
        ".func g(x) {x + 1}\n"
        ".subckt inner a b k=1\n"
        "R1 a b {g(k) * 1k}\n"
        ".ends\n"
        ".subckt outer in out base=100\n"
        ".func g(x) {x * 10}\n"
        ".func pick(flag) {flag > 0}\n"
        ".param r_val = g(base)\n"
        ".if (pick(base))\n"
        "Rtop in mid {r_val}\n"
        ".else\n"
        "Rtop in mid 50k\n"
        ".endif\n"
        "E1 tbl 0 TABLE {V(in)} = (0, 0) (1, {g(0.2)})\n"
        "Rtbl tbl 0 1k\n"
        "B1 bnode 0 V = {g(0.5)}\n"
        "Rb bnode 0 1k\n"
        "X1 mid out inner k={g(0.2)}\n"
        ".ends\n"
        ".subckt sibling in out\n"
        "R1 in out {g(1) * 1k}\n"
        ".ends\n"
    )
    outer_devices = _devices(lib.component("outer"))
    assert outer_devices["Rtop"][1]["resistance"] == pytest.approx(
        1000.0, rel=1e-15, abs=0
    )
    # Local shadow must NOT mutate global library.functions or leak to sibling or child.
    sibling_devices = _devices(lib.component("sibling"))
    assert sibling_devices["R1"][1]["resistance"] == pytest.approx(
        2000.0, rel=1e-15, abs=0
    )
    # In outer, X1 gets k = local g(0.2) = 2.0; inside inner, global g(2.0) = 3.0 -> 3k.
    outer_comp = lib.component("outer")
    inner_comp = next(c for c in outer_comp.children if c.name.startswith("inner"))
    assert _devices(inner_comp)["R1"][1]["resistance"] == pytest.approx(
        3000.0, rel=1e-15, abs=0
    )
    assert set(lib.functions) == {"g"}


def test_conditional_local_func_overrides_unconditional_in_any_order() -> None:
    """Verifies conditional `.func` definitions override unconditional ones.

    `rel=1e-15, abs=0` (~4.5 eps) covers IEEE-754 scaling rounding.
    """
    lib = library(
        ".subckt cell a b mode=0\n"
        ".if (mode == 1)\n"
        ".func f(x) {x * 5}\n"
        ".param r_cond = f(100)\n"
        ".endif\n"
        ".func f(x) {x * 2}\n"
        "R1 a b {f(100)}\n"
        ".ends\n"
    )
    assert _devices(lib.component("cell", mode=0))["R1"][1]["resistance"] == (
        pytest.approx(200.0, rel=1e-15, abs=0)
    )
    assert _devices(lib.component("cell", mode=1))["R1"][1]["resistance"] == (
        pytest.approx(500.0, rel=1e-15, abs=0)
    )


def test_model_cards_evaluate_global_and_local_func_and_bare_calls() -> None:
    """Verifies global, local, and bare `.func` calls evaluate in `.model` cards.

    `rel=1e-15, abs=0` (~4.5 eps) covers IEEE-754 scaling rounding at sub-pA scale.
    """
    lib = library(
        # Global .model defined BEFORE global .func.
        ".model d_glob D IS=scale_is(1e-14) N=bump(1.0)\n"
        ".func scale_is(x) {x * 2}\n"
        ".func bump(x) {x + 0.25}\n"
        ".option scale = {scale_is(0.5)}\n"
        ".subckt use_bare a k\n"
        ".func local_is(x) {x * 3}\n"
        ".model d_bare1 D N=1.2 IS=local_is(1e-14)\n"
        ".model d_bare2 D IS=local_is(2e-14) N=bump(0.5)\n"
        ".model d_ako ako:d_glob D(RS=bump(4.75))\n"
        ".model d_loc_ako ako:d_bare1 D(RS=local_is(2))\n"
        "D1 a k d_glob\n"
        "D2 a k d_bare1\n"
        "D3 a k d_bare2\n"
        "D4 a k d_ako\n"
        "D5 a k d_loc_ako\n"
        ".ends\n"
    )
    assert lib.options["scale"] == pytest.approx(1.0, rel=1e-15, abs=0)
    devs = _devices(lib.component("use_bare"))
    assert devs["D1"][1]["saturation_current"] == pytest.approx(2e-14, rel=1e-15, abs=0)
    assert devs["D1"][1]["emission_coefficient"] == pytest.approx(
        1.25, rel=1e-15, abs=0
    )
    assert devs["D2"][1]["saturation_current"] == pytest.approx(3e-14, rel=1e-15, abs=0)
    assert devs["D2"][1]["emission_coefficient"] == pytest.approx(1.2, rel=1e-15, abs=0)
    assert devs["D3"][1]["saturation_current"] == pytest.approx(6e-14, rel=1e-15, abs=0)
    assert devs["D3"][1]["emission_coefficient"] == pytest.approx(
        0.75, rel=1e-15, abs=0
    )
    assert devs["D4"][1]["saturation_current"] == pytest.approx(2e-14, rel=1e-15, abs=0)
    assert devs["D4"][1]["series_resistance"] == pytest.approx(5.0, rel=1e-15, abs=0)
    assert devs["D5"][1]["saturation_current"] == pytest.approx(3e-14, rel=1e-15, abs=0)
    assert devs["D5"][1]["series_resistance"] == pytest.approx(6.0, rel=1e-15, abs=0)
    # survey_model also resolves global .func.
    surveyed = lib.survey_model("d_glob")
    assert surveyed.problem is None


def test_passive_geometric_binned_and_vbic_model_cards_use_func() -> None:
    """Verifies passive, geometric, binned, and VBIC `.model` cards evaluate `.func`.

    `rel=1e-15, abs=0` (~4.5 eps) covers IEEE-754 scaling rounding.
    """
    lib = library(
        ".func twice(x) {2 * x}\n"
        ".model r_ps RES(R=twice(1.5) TC1=twice(1m))\n"
        ".model r_geo R(RSH=twice(50))\n"
        ".model c_geo C(CJ=twice(1e-3))\n"
        ".model mbin.0 NMOS(LEVEL=49 VTH0=twice(0.35)"
        " LMIN=twice(0.25u) LMAX=twice(1u) WMIN=twice(0.5u) WMAX=twice(5u))\n"
        ".model qheat NPN(LEVEL=4 IS=1e-16 RTH=twice(50) CTH=twice(1u))\n"
        ".subckt deck a b\n"
        "R1 a b r_ps 1k\n"
        "R2 a b r_geo W=2u L=4u\n"
        "C1 a b c_geo W=10u L=10u\n"
        "M1 a b 0 0 mbin W=2u L=1u\n"
        "Q1 a b 0 0 t qheat\n"
        ".ends\n"
    )
    devs = _devices(lib.component("deck"))
    assert devs["R1"][1]["resistance"] == pytest.approx(3000.0, rel=1e-15, abs=0)
    assert devs["R2"][1]["resistance"] == pytest.approx(200.0, rel=1e-15, abs=0)
    assert devs["C1"][1]["capacitance"] == pytest.approx(2e-13, rel=1e-15, abs=0)
    assert devs["M1"][1]["vth0"] == pytest.approx(0.7, rel=1e-15, abs=0)
    assert "Q1.rth" in devs
    assert devs["Q1.rth"][1]["resistance"] == pytest.approx(100.0, rel=1e-15, abs=0)
    assert devs["Q1.cth"][1]["capacitance"] == pytest.approx(2e-6, rel=1e-15, abs=0)
    bin_card = lib.card("mbin", {})
    assert bin_card is not None
    assert models.center(bin_card, lib.values(), functions=lib.functions) == (
        pytest.approx((1.25e-6, 5.5e-6), rel=1e-15, abs=0)
    )


@pytest.mark.parametrize(
    ("deck", "match"),
    [
        (
            ".subckt a n\n.func f(x) {f(x) + 1}\nR1 n 0 {f(1)}\n.ends\n",
            "a function calls itself: `f` calls `f`",
        ),
        (
            ".subckt a n\n.func p(x) {q(x)}\n.func q(x) {p(x)}\nR1 n 0 {p(1)}\n.ends\n",
            "a function calls itself: `p` calls `q` calls `p`",
        ),
        (
            (
                ".func g(x) {loc(x)}\n.subckt a n\n.func loc(x) {g(x)}\n"
                "R1 n 0 {loc(1)}\n.ends\n"
            ),
            "a function calls itself: `loc` calls `g` calls `loc`",
        ),
        (
            (
                ".subckt a n\n.func bad(x) {bad(x)}\n.model dm D(IS=bad(1e-14))\n"
                "D1 n 0 dm\n.ends\n"
            ),
            "a function calls itself: `bad` calls `bad`",
        ),
        (
            ".subckt a n\n.func pair(x, y) {x + y}\nR1 n 0 {pair(1)}\n.ends\n",
            "`pair` takes 2 arguments, not 1",
        ),
    ],
)
def test_recursive_local_func_and_wrong_arity_are_refused(
    deck: str, match: str
) -> None:
    with pytest.raises(SpiceError, match=match):
        library(deck).component("a")


def test_local_func_model_cache_has_no_crosstalk() -> None:
    """Verifies subcircuit-local `.func` calls do not crosstalk across model caches.

    `rel=1e-15, abs=0` (~4.5 eps) covers IEEE-754 scaling rounding at sub-pA scale.
    """
    lib = library(
        ".func scale(x) {x}\n"
        ".model d_shared D(IS=scale(1e-14))\n"
        ".subckt cell_a a k\n"
        ".func scale(x) {x * 2}\n"
        "D1 a k d_shared\n"
        ".ends\n"
        ".subckt cell_b a k\n"
        ".func scale(x) {x * 7}\n"
        "D1 a k d_shared\n"
        ".ends\n"
        ".subckt switched a k sel=0\n"
        ".if (sel == 0)\n"
        ".func gain(x) {x * 3}\n"
        ".else\n"
        ".func gain(x) {x * 9}\n"
        ".endif\n"
        ".model d_local D(IS=gain(1e-14))\n"
        "D1 a k d_local\n"
        ".ends\n"
    )
    assert _devices(lib.component("cell_a"))["D1"][1]["saturation_current"] == (
        pytest.approx(2e-14, rel=1e-15, abs=0)
    )
    assert _devices(lib.component("cell_b"))["D1"][1]["saturation_current"] == (
        pytest.approx(7e-14, rel=1e-15, abs=0)
    )
    assert _devices(lib.component("switched", sel=0))["D1"][1][
        "saturation_current"
    ] == pytest.approx(3e-14, rel=1e-15, abs=0)
    assert _devices(lib.component("switched", sel=1))["D1"][1][
        "saturation_current"
    ] == pytest.approx(9e-14, rel=1e-15, abs=0)


# `.ic` / `.nodeset` inside subcircuits and device `IC=` under UIC.


def test_ic_and_nodeset_inside_subcircuits_remap_ports_and_hierarchy() -> None:
    lib = library(
        ".ic v(OUT)=9\n"
        ".nodeset v(OUT)=8\n"
        ".subckt leaf a b vinit=2\n"
        ".func half(x) {x / 2}\n"
        "R1 a mid 1k\n"
        "R2 mid b 1k\n"
        ".ic v(MID)={vinit} v(a)={half(vinit)}\n"
        ".nodeset v(mid)={vinit + 0.5}\n"
        ".ends\n"
        ".subckt mid_stage in out\n"
        "X1 in inner leaf vinit=2\n"
        "X2 inner out leaf vinit=4\n"
        ".ends\n"
        ".subckt top out\n"
        "Xleft p1 out mid_stage\n"
        "Xright p2 0 leaf vinit=2\n"
        "R1 p1 0 1k\n"
        "R2 p2 0 1k\n"
        ".ends\n"
    )
    ic_top = lib.initial_conditions_of("top")
    assert ic_top == {
        "Xleft/X1/mid": 2.0,
        "p1": 1.0,
        "Xleft/X2/mid": 4.0,
        "Xleft/inner": 2.0,
        "Xright/mid": 2.0,
        "p2": 1.0,
        # Top-level .ic v(OUT)=9 canonicalizes to `out` and overrides inner values.
        "out": 9.0,
    }
    ns_top = lib.nodesets_of("top")
    assert ns_top == {
        "Xleft/X1/mid": 2.5,
        "Xleft/X2/mid": 4.5,
        "Xright/mid": 2.5,
        "out": 8.0,
    }

    # Subcircuit instantiated with explicit path and connections mapping pins to parent
    # nets and ground.
    ic_placed = lib.initial_conditions_of(
        "leaf", "U1", connections={"a": "bus", "b": "0"}, vinit=6
    )
    assert ic_placed == {"U1/mid": 6.0, "bus": 3.0}
    with pytest.raises(SpiceError, match="has no pin `bogus`"):
        lib.initial_conditions_of("leaf", "U1", connections={"bogus": "bus"})

    # Non-zero ground .ic and .nodeset are refused (direct and via pin bound to ground).
    with pytest.raises(SpiceError, match="ground"):
        lib.initial_conditions_of("leaf", "U1", connections={"a": "0"})
    bad_ns = library(".subckt s a\nR1 a 0 1k\n.nodeset v(0)=1\n.ends\n")
    with pytest.raises(SpiceError, match="ground"):
        bad_ns.nodesets_of("s")


def test_identical_cached_subcircuit_instances_preserve_each_ic() -> None:
    lib = library(
        ".subckt cell a\n"
        "R1 a mid 1k\n"
        "C1 mid 0 1u IC=3.5\n"
        ".ic v(mid)=3.5\n"
        ".ends\n"
        ".subckt top a b\n"
        "X1 a cell\n"
        "X2 b cell\n"
        ".ends\n"
    )
    assert lib.initial_conditions_of("top") == {"X1/mid": 3.5, "X2/mid": 3.5}
    assert lib.initial_conditions_of("top", operating_point=False) == {
        "X1/mid": 3.5,
        "X2/mid": 3.5,
    }


def test_capacitor_ic_topologies_under_uic_and_ignored_under_op() -> None:
    lib = library(
        ".subckt topologies a b c d e\n"
        "Cpos a 0 1u IC=2.5\n"
        "Cneg 0 b 1u IC=1.5\n"
        "Cchain1 c d 1u IC=2.0\n"
        "Cchain2 d 0 1u IC=1.0\n"
        "Cfloat fpos fneg 1u IC=3.25\n"
        "R1 fpos 0 10k\n"
        "R2 fneg 0 10k\n"
        "Canchor e anc 1u IC=1.25\n"
        ".ic v(anc)=0.75\n"
        "Cq qnode 0 Q={1u * V(qnode)} IC=4.0\n"
        "Cexpr cnode 0 C={1u + 0.1u * V(cnode)} IC=-2.0\n"
        ".ends\n"
    )
    # With operating_point=True, device IC= is ignored; only .ic remains.
    assert lib.initial_conditions_of("topologies") == {"anc": 0.75}
    # With operating_point=False (UIC), capacitor IC= voltages solve into node voltages.
    uic = lib.initial_conditions_of("topologies", operating_point=False)
    assert uic == {
        "anc": 0.75,
        "a": 2.5,
        "b": -1.5,
        "d": 1.0,
        "c": 3.0,
        "e": 2.0,
        "qnode": 4.0,
        "cnode": -2.0,
        "fneg": 0.0,
        "fpos": 3.25,
    }


def test_subcircuit_pin_capacitor_ic_grounded_at_instantiation() -> None:
    lib = library(
        ".subckt capcell p n\n"
        "C1 p n 1u IC=2.5\n"
        ".ends\n"
        ".subckt top pos neg\n"
        "X1 pos 0 capcell\n"
        "X2 0 neg capcell\n"
        ".ends\n"
    )
    uic = lib.initial_conditions_of("top", operating_point=False)
    assert uic == {"pos": 2.5, "neg": -2.5}


def test_consistent_parallel_and_loop_capacitor_ic_succeed() -> None:
    lib = library(
        ".subckt ok a b\n"
        "C1 a 0 1u IC=3\n"
        "C2 a 0 2u IC=3\n"
        "C3 b a 1u IC=2\n"
        "C4 b 0 1u IC=5\n"
        "Cshort a a 1u IC=0\n"
        ".ends\n"
    )
    assert lib.initial_conditions_of("ok", operating_point=False) == {
        "a": 3.0,
        "b": 5.0,
    }


@pytest.mark.parametrize(
    ("deck", "via_component", "match"),
    [
        (
            ".subckt s a\nC1 a 0 1u IC=1\nC2 a 0 1u IC=2\n.ends\n",
            False,
            "contradictory initial conditions",
        ),
        (
            ".subckt s a\nC1 a 0 1u IC=1\n.ic v(a)=2\n.ends\n",
            False,
            "contradictory initial conditions",
        ),
        (
            ".subckt s a b c\nC1 a b 1u IC=1\nC2 b c 1u IC=1\nC3 c a 1u IC=1\n.ends\n",
            False,
            "contradictory initial conditions",
        ),
        (".subckt s a\nC1 a a 1u IC=1.5\n.ends\n", False, "shorted capacitor `C1`"),
        (".subckt s a\nC1 a 0 1u IC=1,2\n.ends\n", True, "`IC` accepts one value"),
        (".subckt s a\nR1 a 0 1k IC=1\n.ends\n", True, "no `IC` here"),
    ],
)
def test_contradictory_and_malformed_ic_are_refused(
    deck: str, *, via_component: bool, match: str
) -> None:
    lib = library(deck)
    action = (
        (lambda: lib.component("s"))
        if via_component
        else (lambda: lib.initial_conditions_of("s", operating_point=False))
    )
    with pytest.raises(SpiceError, match=match):
        action()


def test_inductor_ic_currents_multiplier_and_coupled_windings() -> None:
    """Verifies inductor `IC=` currents scale by `m` (`rel=1e-15, abs=0` IEEE-754)."""
    lib = library(
        ".subckt coil a b\n"
        "L1 a b 1m M=2 IC=0.25\n"
        ".ends\n"
        ".subckt top a b c d\n"
        "X1 a b coil m=3\n"
        "Lp a 0 1m IC=0.1\n"
        "Ls c 0 1m IC=-0.2\n"
        "K1 Lp Ls 0.9\n"
        "Lzero d 0 1m IC=0\n"
        ".ends\n"
    )
    currents = lib.initial_currents_of("top")
    assert currents == {
        "X1/L1:current": pytest.approx(1.5, rel=1e-15, abs=0),
        "K1:current1": pytest.approx(0.1, rel=1e-15, abs=0),
        "K1:current2": pytest.approx(-0.2, rel=1e-15, abs=0),
        "Lzero:current": 0.0,
    }
    prefixed = lib.initial_currents_of("top", "U1")
    assert prefixed["U1/X1/L1:current"] == pytest.approx(1.5, rel=1e-15, abs=0)
    assert prefixed["U1/K1:current1"] == pytest.approx(0.1, rel=1e-15, abs=0)

    # Non-zero inductor IC= is refused under operating_point=False.
    with pytest.raises(
        SpiceError,
        match=r"inductor initial current `X1/L1:current`.*accepts net voltages alone",
    ):
        lib.initial_conditions_of("top", operating_point=False)

    # Zero inductor IC= is allowed under operating_point=False.
    zero_ind = library(".subckt z a\nL1 a 0 1m IC=0\nC1 a 0 1u IC=2.0\n.ends\n")
    assert zero_ind.initial_conditions_of("z", operating_point=False) == {"a": 2.0}


def test_transient_rc_decay_from_subcircuit_ic_and_capacitor_ic() -> None:
    lib = library(
        ".subckt rc_ic\n"
        "R1 mid 0 1k\n"
        "C1 mid 0 1u\n"
        ".ic v(mid)=3.0\n"
        ".ends\n"
        ".subckt rc_cap_ic\n"
        "R1 mid 0 1k\n"
        "C1 mid 0 1u IC=4.0\n"
        ".ends\n"
        ".subckt wrapper\n"
        "X1 rc_cap_ic\n"
        ".ends\n"
    )
    tau = 1e-3  # R * C = 1k * 1u = 1 ms

    # 1) Subcircuit .ic and nested capacitor IC= under UIC match V0 * exp(-t / tau).
    for subcircuit, probe, expected_initial in (
        ("rc_ic", "mid", 3.0),
        ("wrapper", "X1/mid", 4.0),
    ):
        ic_map = {
            net: value * V
            for net, value in lib.initial_conditions_of(
                subcircuit, operating_point=False
            ).items()
        }
        assert ic_map == {probe: expected_initial * V}
        res = Simulation(circuit(lib.component(subcircuit))).run(
            Transient(
                stop=2 * ms,
                max_step=20 * us,
                initial_conditions=ic_map,
                operating_point=False,
            ),
            TIGHT,
        )
        # Initial condition at t=0 matches within LU rounding (rel=1e-12, abs=0);
        # trapezoidal discretization with max_step=20 us on tau=1 ms (h/tau=0.02) gives
        # O(h^2/tau^2) ~ 1e-4 relative error at t=2 ms (bounded by rel=1e-4, abs=0).
        assert res.voltage(probe)[0] == pytest.approx(
            expected_initial, rel=1e-12, abs=0
        )
        assert res.voltage(probe)[-1] == pytest.approx(
            expected_initial * math.exp(-2e-3 / tau), rel=1e-4, abs=0
        )

    # 3) Subcircuit .nodeset guides a bistable tanh circuit to its chosen root.
    bistable = library(
        ".subckt latch\n"
        "R1 x 0 1\n"
        "B1 0 x I = {2 * tanh(V(x))}\n"
        ".nodeset v(x)=2\n"
        ".ends\n"
        ".subckt top\n"
        "X1 latch\n"
        ".ends\n"
    )
    nodesets = {net: value * V for net, value in bistable.nodesets_of("top").items()}
    assert nodesets == {"X1/x": 2.0 * V}
    op = Simulation(circuit(bistable.component("top"))).run(
        OperatingPoint(), Options(nodesets=nodesets)
    )
    # Closed-form fixed point of v = 2 * tanh(v) is 1.915008046... V; rel=1e-6 covers
    # default Newton operating-point reltol=1e-3 / vntol=1e-6 convergence tolerance.
    assert op.voltage("X1/x") == pytest.approx(1.915008046, rel=1e-6, abs=0)


def _assert_pss_fields(
    card: object, expected: Mapping[str, object], *, tolerance: float = 1e-12
) -> None:
    """Asserts `.pss` card fields match `expected` within IEEE-754 rounding."""
    for field, target in expected.items():
        actual = getattr(card, field)
        if isinstance(target, float):
            assert actual == pytest.approx(target, rel=tolerance, abs=0), field
        else:
            assert actual == target, field


def test_library_reads_driven_pss_statements() -> None:
    """Verifies driven `.pss` cards parse within IEEE-754 rounding (`rel=1e-12`)."""
    read = library(
        ".param f0=1meg\n"
        ".func half(x) {x / 2}\n"
        ".pss 1meg\n"
        ".pss {f0} {half(20u)}\n"
        ".pss period=1u max_step=5n method=gear2\n"
        ".pss fund=1k points=100 method=be\n"
        ".pss freq=10k tstep=1u uic\n"
    )
    cards = read.periodic_steady_states
    assert len(cards) == 5
    assert cards[0].statement is not None
    assert cards[0].statement.line == 3
    expected: tuple[dict[str, object], ...] = (
        {
            "autonomous": False,
            "frequency": 1e6,
            "period": 1e-6,
            "stabilization": None,
            "operating_point": True,
        },
        {"autonomous": False, "frequency": 1e6, "period": 1e-6, "stabilization": 10e-6},
        {
            "autonomous": False,
            "period": 1e-6,
            "frequency": 1e6,
            "max_step": 5e-9,
            "method": "gear2",
        },
        {
            "autonomous": False,
            "period": 1e-3,
            "points": 100,
            "method": "backward_euler",
        },
        {
            "autonomous": False,
            "period": 1e-4,
            "max_step": 1e-6,
            "operating_point": False,
        },
    )
    for card, fields in zip(cards, expected, strict=True):
        _assert_pss_fields(card, fields)


def test_library_reads_autonomous_pss_statements() -> None:
    """Verifies autonomous `.pss` cards parse within IEEE-754 rounding (`rel=1e-12`)."""
    read = library(
        ".pss\n"
        ".pss 1meg 10u out\n"
        ".pss 1meg 10u V(out) 1024 10 50 5e-3 uic\n"
        ".pss 1meg 0 2 512\n"
        ".pss fund=2meg tstab=5u oscnode=v(tank) harms=8 sc_iter=40 steady_coeff=1e-3\n"
        ".pss autonomous max_step=2n\n"
        ".pss oscnob=node_a\n"
    )
    cards = read.periodic_steady_states
    assert len(cards) == 7
    assert cards[0].statement is not None
    assert cards[0].statement.line == 1
    expected: tuple[dict[str, object], ...] = (
        {"autonomous": True, "period": None, "frequency": None},
        {
            "autonomous": True,
            "period": None,
            "frequency": 1e6,
            "stabilization": 10e-6,
            "oscillator_node": "out",
        },
        {
            "autonomous": True,
            "period": None,
            "frequency": 1e6,
            "stabilization": 10e-6,
            "oscillator_node": "out",
            "points": 1024,
            "harmonics": 10,
            "max_shooting_iterations": 50,
            "steady_coefficient": 5e-3,
            "operating_point": False,
        },
        {
            "autonomous": True,
            "oscillator_node": "2",
            "stabilization": 0.0,
            "points": 512,
        },
        {
            "autonomous": True,
            "period": None,
            "frequency": 2e6,
            "stabilization": 5e-6,
            "oscillator_node": "tank",
            "harmonics": 8,
            "max_shooting_iterations": 40,
            "steady_coefficient": 1e-3,
        },
        {"autonomous": True, "period": None, "max_step": 2e-9},
        {"autonomous": True, "oscillator_node": "node_a"},
    )
    for card, fields in zip(cards, expected, strict=True):
        _assert_pss_fields(card, fields)


def test_pss_from_spice_simulates_driven_and_autonomous_decks() -> None:
    """Verifies `.pss` card conversion and shooting on driven and autonomous decks.

    Driven `period` and `max_step` match exact decimal parsing within IEEE-754
    rounding (`rel=1e-12, abs=0`). The autonomous tank period matches
    `2*pi*sqrt(L*C)` within 1% (`rel=1e-2`, matching the trapezoidal
    limit-cycle tolerance in `crates/jolty-engine/tests/periodic.rs`).
    """
    driven_deck = library(
        ".subckt rc_driven\n"
        "V1 in 0 SIN(0 1 1meg)\n"
        "R1 in out 1k\n"
        "C1 out 0 1n\n"
        ".ends\n"
        ".pss 1meg points=100\n"
    )
    (driven_card,) = driven_deck.periodic_steady_states
    driven_analysis = PeriodicSteadyState.from_spice(driven_card)
    assert driven_analysis.period is not None
    assert driven_analysis.period.value == pytest.approx(1e-6, rel=1e-12, abs=0)
    assert driven_analysis.max_step is not None
    assert driven_analysis.max_step.value == pytest.approx(1e-8, rel=1e-12, abs=0)
    assert driven_analysis.method == "trapezoidal"
    driven_result = Simulation(circuit(driven_deck.component("rc_driven"))).run(
        driven_analysis
    )
    assert isinstance(driven_result, results.PeriodicSteadyStateResult)
    assert driven_result.period == pytest.approx(1e-6, rel=1e-12, abs=0)

    # Autonomous regulated LC tank: L=1uH, C=1nF -> f0 ~ 5.033 MHz, T ~ 198.7 ns.
    inductance, capacitance = 1e-6, 1e-9
    impedance = math.sqrt(inductance / capacitance)
    conductance = 0.2 / impedance
    regulation = f"-{conductance} * V(t) * (1 - (V(t)^2 + ({impedance} * I(V1))^2))"
    osc_deck = library(
        ".subckt tank\n"
        f"C1 t 0 {capacitance}\n"
        f"L1 t l {inductance}\n"
        "V1 l 0 0\n"
        f"B1 t 0 I={{{regulation}}}\n"
        ".ends\n"
        ".pss 5meg 0 t\n"
    )
    (osc_card,) = osc_deck.periodic_steady_states
    osc_analysis = PeriodicSteadyState.from_spice(osc_card)
    assert osc_analysis.period is None
    assert osc_analysis.max_step is None
    assert osc_analysis.method == "trapezoidal"
    osc_result = Simulation(circuit(osc_deck.component("tank"))).run(osc_analysis)
    assert isinstance(osc_result, results.PeriodicSteadyStateResult)
    expected_period = 2.0 * math.pi * math.sqrt(inductance * capacitance)
    assert osc_result.period == pytest.approx(expected_period, rel=1e-2, abs=0)


@pytest.mark.parametrize(
    ("card", "fragment"),
    [
        (".pss 0", "frequency is positive"),
        (".pss -1meg", "frequency is positive"),
        (".pss period=0", "period is positive"),
        (".pss period=-1u", "period is positive"),
        (".pss 1meg -1u", "stabilization time is at least 0"),
        (".pss max_step=0", "max_step is positive"),
        (".pss max_step=-1n", "max_step is positive"),
        (".pss 1meg 0 out 0", "points is a positive integer"),
        (".pss 1meg 0 out 10.5", "points is a positive integer"),
        (".pss 1meg 0 out 64 0", "harmonics is a positive integer"),
        (".pss 1meg 0 out 64 5 0", "shooting iterations is a positive integer"),
        (".pss 1meg 0 out 64 5 10 0", "steady coefficient is positive"),
        (".pss 1meg 0 0", "ground node `0`"),
        (".pss 1meg 0 V(0)", "ground node `0`"),
        (".pss 1meg 0 V(a,b)", "written `node` or `V(node)`"),
        (".pss 1meg 0 V()", "written `node` or `V(node)`"),
        (".pss 1meg 0 a/b", "separates the levels"),
        (".pss 1meg period=1u", "sets period or frequency, not both"),
        (".pss fund=1meg period=1u", "sets period or frequency, not both"),
        (".pss autonomous period=1u", "takes no period"),
        (".pss oscnode=out period=1u", "takes no period"),
        (".pss period=1u period=2u", "duplicate .pss option"),
        (".pss 1meg fund=2meg", "duplicate .pss option"),
        (".pss 1meg 10u tstab=20u", "duplicate .pss option"),
        (".pss foo=1", "unknown .pss option `foo`"),
        (".pss method=rk4", "unknown .pss method `rk4`"),
        (".pss 1meg 0 out 64 5 10 1e-3 extra", "at most 7 positional"),
        (".pss =", "unexpected `=`"),
        (".pss period=", "missing value after `period=`"),
        (".pss period==1u", "missing value after `period=`"),
        (".pss {missing_param}", "invalid .pss frequency"),
        (".pss bogus", "invalid .pss frequency"),
    ],
)
def test_malformed_pss_statements_raise_factual_spice_errors(
    card: str, fragment: str
) -> None:
    advice = ("use ", "pass ", "try ", "instead", "should", "give ", "did you mean")
    with pytest.raises(SpiceError) as raised:
        library(card)
    message = str(raised.value)
    assert fragment in message, message
    lowered = message.casefold()
    for word in advice:
        assert word not in lowered, f"advice word {word!r} in {message}"


# Regression tests for netlist, library, expressions, models, export, download,
# sky130, ngspice, and xyce.


def test_netlist_preserves_tabs_as_separators() -> None:
    parsed = statements(
        ".subckt\tdiv\ta\tb\n.if\t(1)\nR1\ta\tb\t1k\n+\tTC1=1m\n.endif\n.ends\n",
        "tabs.lib",
    )
    assert [item.text for item in parsed] == [
        ".subckt div a b",
        ".if (1)",
        "R1 a b 1k TC1=1m",
        ".endif",
        ".ends",
    ]
    part = library(
        ".subckt\tdiv\ta\tb\n.if\t(1)\nR1\ta\tb\t1k\n.endif\n.ends\n"
    ).component("div")
    assert _devices(part) == {"R1": ("resistor", {"resistance": 1000.0})}


def test_netlist_preserves_semicolon_and_dollar_inside_quoted_strings(
    tmp_path: Path,
) -> None:
    lines = statements(
        '.include "dir;1/foo.lib" ; real trailing comment\n'
        ".include 'dir $1/bar.lib' $ real trailing comment\n"
        "R1 a b 1k ; comment after element\n"
        "R2 b 0 2k $ dollar comment\n",
        "quoted.lib",
    )
    assert [item.text for item in lines] == [
        '.include "dir;1/foo.lib"',
        ".include 'dir $1/bar.lib'",
        "R1 a b 1k",
        "R2 b 0 2k",
    ]
    semi_dir = tmp_path / "dir;1"
    semi_dir.mkdir()
    (semi_dir / "foo.lib").write_text(".subckt s1 a b\nR1 a b 1k\n.ends\n")
    dollar_dir = tmp_path / "dir $1"
    dollar_dir.mkdir()
    (dollar_dir / "bar.lib").write_text(".subckt s2 a b\nR2 a b 2k\n.ends\n")
    top = tmp_path / "top.lib"
    top.write_text(
        ".include \"dir;1/foo.lib\" ; comment\n.include 'dir $1/bar.lib' $ comment\n",
    )
    loaded = spice.load(top)
    assert set(loaded.subcircuits) == {"s1", "s2"}
    assert _devices(loaded.component("s1")) == {
        "R1": ("resistor", {"resistance": 1000.0})
    }
    assert _devices(loaded.component("s2")) == {
        "R2": ("resistor", {"resistance": 2000.0})
    }


def test_library_concurrent_survey_component_drawn_and_random_variables() -> None:
    lib = library(
        ".param base = agauss(1000, 100, 3)\n"
        ".subckt leaf a b\n"
        ".param rval = agauss(500, 50, 3)\n"
        "R1 a b {rval}\n"
        ".ends\n"
        ".subckt top in out\n"
        "R0 in mid {base}\n"
        "X1 mid out leaf\n"
        ".ends\n"
    )
    expected_nominal = _devices(lib.component("top"))
    expected_survey = lib.survey("top")
    expected_rv = lib.random_variables("top", "U1")
    workers = 8
    barrier = threading.Barrier(workers)

    def task(worker: int) -> None:
        barrier.wait()
        for step in range(4):
            kind = (worker + step) % 4
            if kind == 0:
                assert lib.survey("top") == expected_survey
                assert not lib.surveying
            elif kind == 1:
                assert _devices(lib.component("top")) == expected_nominal
            elif kind == 2:
                sample = spice.Sample(seed=17, index=worker)
                drawn = lib.drawn("top", sample, f"U{worker}")
                assert drawn.definition.instances
            else:
                assert lib.random_variables("top", "U1") == expected_rv

    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(task, worker) for worker in range(workers)]
        for future in futures:
            future.result()
    assert not lib.surveying


def test_library_cache_invalidates_on_options_compatibility_and_parsed_models() -> None:
    lib = library(
        ".model nch.1 nmos(level=54 version=4.5 lmin=0.1u lmax=10u "
        "wmin=0.1u wmax=1.2u vth0=0.4)\n"
        ".model nch.2 nmos(level=54 version=4.5 lmin=0.1u lmax=10u "
        "wmin=1.2u wmax=10u vth0=0.7)\n"
        ".subckt dut d g so b\n"
        ".model dloc d(is=1e-14)\n"
        "D1 d so dloc\n"
        "M1 d g so b nch l=1 w=2 nf=2\n"
        ".ends\n"
    )
    lib.options["scale"] = 1e-6
    first = _devices(lib.component("dut"))
    assert first["M1"][1]["vth0"] == 0.7
    assert lib._parsed_models  # noqa: SLF001 - verify local model cache populated

    # Changing scale invalidates cached nominal builds.
    lib.options["scale"] = 0.5e-6
    scaled = _devices(lib.component("dut"))
    assert scaled["M1"][1]["vth0"] == 0.4
    lib.options["scale"] = 1e-6

    # Changing compatibility from None to "hsa" turns on wnflag=1, halving w/nf to 1u.
    lib.compatibility = "hsa"
    second = _devices(lib.component("dut"))
    assert second["M1"][1]["vth0"] == 0.4

    # Explicit options["wnflag"] overrides compatibility and invalidates cache.
    lib.options["wnflag"] = 0.0
    third = _devices(lib.component("dut"))
    assert third["M1"][1]["vth0"] == 0.7

    # Adding a parameter invalidates _parsed_models so stale cards are not retained.
    lib.read(".param extra = 1\n", "extra.lib")
    assert not lib._parsed_models  # noqa: SLF001 - verify cache cleared on invalidation


def test_include_inside_subcircuit_preserves_winding_and_coupling_order(
    tmp_path: Path,
) -> None:
    # Included file restarts line numbers at 1, below L1's line number in main file.
    # Couplings satisfy 0.6**2 + 0.5**2 = 0.61 < 1 (strictly positive definite matrix).
    (tmp_path / "second.inc").write_text(
        "L2 b 0 2m\nK1 L1 L2 0.6\nL3 c 0 3m\nK2 L2 L3 0.5\n",
    )
    main = tmp_path / "main.lib"
    main.write_text(
        "* padding line 1\n"
        "* padding line 2\n"
        "* padding line 3\n"
        "* padding line 4\n"
        "* padding line 5\n"
        ".subckt xfmr a b c\n"
        "R0 a b 10k\n"
        "L1 a 0 1m\n"
        '.include "second.inc"\n'
        "R1 c 0 10k\n"
        ".ends\n",
    )
    comp = spice.load(main).component("xfmr")
    instance_names = [inst.name for inst in comp.definition.instances]
    assert instance_names == ["R0", "K1", "R1"]
    coupled = comp.definition.instances[1]
    assert coupled.of == ir.DeviceOf(model="coupled_inductors")
    assert coupled.terminals == {
        "a1": "a",
        "b1": "ground",
        "a2": "b",
        "b2": "ground",
        "a3": "c",
        "b3": "ground",
    }
    windings = coupled.parameters["windings"]
    assert windings == [
        {"inductance": 1e-3},
        {"inductance": 2e-3},
        {"inductance": 3e-3},
    ]


@pytest.mark.parametrize(
    "deck",
    [
        ".ic V(a)={1/0}\n",
        ".ic V(a)=1e500\n",
        ".nodeset V(a)={0/0}\n",
        ".nodeset V(a)=1e500\n",
        ".option scale={1/0}\n",
        ".pss period=1e-310\n",
        ".pss 1e-310\n",
    ],
)
def test_non_finite_top_level_directives_are_rejected(deck: str) -> None:
    with pytest.raises(SpiceError):
        library(deck)


@pytest.mark.parametrize(
    "body",
    [
        "R1 a 0 {1/0}\n",
        "R1 a 0 {0/0}\n",
        "V1 a 0 {1/0}\n",
        "C1 a 0 1u IC={1/0}\n",
        "L1 a 0 1m IC={1/0}\n",
        "L1 a 0 1m M=10 IC=1e308\n",
    ],
)
def test_non_finite_element_values_and_ic_are_rejected(body: str) -> None:
    lib = library(f".subckt s a\n{body}.ends\n")
    with pytest.raises(SpiceError):
        lib.component("s")


@pytest.mark.parametrize(
    "card",
    [
        ".pss 1meg 0 ground",
        ".pss 1meg 0 GROUND",
        ".pss 1meg 0 V(ground)",
        ".pss 1meg 0 V(GROUND)",
    ],
)
def test_pss_node_rejects_ground_port_name(card: str) -> None:
    with pytest.raises(SpiceError, match="ground port"):
        library(card)


@pytest.mark.parametrize(
    "element_line",
    [
        "C1 a 0 Q={1u * V(a)} M=0",
        "C1 a 0 Q={1u * V(a)} M=-2",
        "C1 a 0 Q={1u * V(a)} M={1/0}",
        "C1 a 0 C={1u + 0.1u * V(a)} M=0",
        "C1 a 0 C={1u + 0.1u * V(a)} M=-1",
    ],
)
def test_charge_of_voltage_validates_positive_finite_copies(
    element_line: str,
) -> None:
    lib = library(f".subckt s a\n{element_line}\n.ends\n")
    with pytest.raises(SpiceError):
        lib.component("s")


def test_binned_model_index_tracks_order_and_invalidation() -> None:
    lib = library(
        ".model nch.1 nmos(level=49 lmin=0.1u lmax=1u wmin=1u wmax=10u vth0=0.3)\n"
        ".model nch.2 nmos(level=49 lmin=0.1u lmax=1u wmin=1u wmax=10u vth0=0.5)\n"
        ".subckt s d g so b\n"
        "M1 d g so b nch l=0.5u w=2u\n"
        ".ends\n"
    )
    assert _devices(lib.component("s"))["M1"][1]["vth0"] == 0.5
    # Adding a parameter clears the bin index and rebuilds it accurately on next lookup.
    lib.read(".param k = 1\n", "k.lib")
    assert _devices(lib.component("s"))["M1"][1]["vth0"] == 0.5
    # Redefining an earlier bin keeps its dictionary position, while a new bin wins.
    lib.read(
        ".model nch.3 nmos(level=49 lmin=0.1u lmax=1u wmin=1u wmax=10u vth0=0.8)\n",
        "b3.lib",
    )
    assert _devices(lib.component("s"))["M1"][1]["vth0"] == 0.8
    assert lib.survey_model("nch").element == "nch.1"


def test_heat_propagates_across_multi_stage_foster_and_cauer_ladders() -> None:
    hot = ".model hot npn level=9 is=1e-16 rth=300 cth=1e-12\n"
    cauer = library(
        f"{hot}.subckt cauer c b e t\nQ1 c b e 0 t hot\nRload c 0 1k\n"
        "Rth1 t t1 100\nCth1 t1 0 10n\nRth2 t1 t2 200\nCth2 t2 0 20n\n"
        "Rth3 t2 0 500\n.ends\n"
    )
    found = _devices(cauer.component("cauer"))
    assert {k: found[k] for k in ("Rload", "Rth1", "Cth1", "Rth2", "Cth2", "Rth3")} == {
        "Rload": ("resistor", {"resistance": 1000.0}),
        "Rth1": ("thermal_resistance", {"resistance": 100.0}),
        "Cth1": ("thermal_capacitance", {"capacitance": 10e-9}),
        "Rth2": ("thermal_resistance", {"resistance": 200.0}),
        "Cth2": ("thermal_capacitance", {"capacitance": 20e-9}),
        "Rth3": ("thermal_resistance", {"resistance": 500.0}),
    }

    foster = library(
        f"{hot}.subckt foster c b e t\nQ1 c b e 0 t hot\n"
        "R1 t t1 40\nC1 t t1 5n\nR2 t1 0 60\nC2 t1 0 8n\n.ends\n"
    )
    foster_devices = _devices(foster.component("foster"))
    assert {k: foster_devices[k] for k in ("R1", "C1", "R2", "C2")} == {
        "R1": ("thermal_resistance", {"resistance": 40.0}),
        "C1": ("thermal_capacitance", {"capacitance": 5e-9}),
        "R2": ("thermal_resistance", {"resistance": 60.0}),
        "C2": ("thermal_capacitance", {"capacitance": 8e-9}),
    }

    for body, match in (
        ("Rth1 t t1 100\nRbad t1 c 1k\n", "joins a thermal node to another"),
        ("Rth1 t t1 100\nRth2 t1 0 200 TC1=1m\n", "drifts with temperature or voltage"),
    ):
        bad = library(f"{hot}.subckt s c b e t\nQ1 c b e 0 t hot\n{body}.ends\n")
        with pytest.raises(SpiceError, match=match):
            bad.component("s")


@pytest.mark.parametrize(
    "deck",
    [
        ".subckt s a\nR1 a 0 1k m=\n.ends",
        ".subckt s a\nR1 a 0 1k m=2 3\n.ends",
        ".subckt s a\nR1 a 0 1k temp=\n.ends",
        ".subckt s a\nR1 a 0 1k temp=25 extra\n.ends",
        ".subckt s a\nR1 a 0 1k r=\n.ends",
        ".subckt s a\nR1 a 0 1k tc=\n.ends",
        ".subckt s a\nR1 a 0 1k tc1=1m 2m\n.ends",
        ".subckt s a\nC1 a 0 Q={1u*V(a)} m=\n.ends",
        ".subckt s a\nC1 a 0 Q={1u*V(a)} m=2 extra\n.ends",
        ".subckt s a\nR1 a 0 {1k + V(a)} m=\n.ends",
        ".model mn nmos\n.subckt s d g so b\nM1 d g so b mn l=\n.ends",
        ".model mn nmos\n.subckt s d g so b\nM1 d g so b mn l=1u 2u\n.ends",
        ".model mv vdmos\n.subckt s d g so\nM1 d g so mv m=\n.ends",
        ".model mv vdmos\n.subckt s d g so\nM1 d g so mv m=2 thermal\n.ends",
        ".model mv vdmos\n.subckt s d g so\nM1 d g so mv m=2 extra\n.ends",
        ".model qn npn\n.subckt s c b e\nQ1 c b e qn area=\n.ends",
        ".model qn npn\n.subckt s c b e\nQ1 c b e qn area=2 3\n.ends",
    ],
)
def test_empty_and_extra_keyword_tokens_are_rejected(deck: str) -> None:
    lib = library(deck)
    with pytest.raises(SpiceError):
        lib.component("s")


def test_valid_spice_trailing_flags_and_multi_value_ic_remain_accepted() -> None:
    lib = library(
        ".model mn nmos\n"
        ".model mv vdmos\n"
        ".model qv npn(level=4 is=1e-16)\n"
        ".subckt s a b c d\n"
        "M1 a b c d mn l=1u off\n"
        "M2 a b c mv m=2 off\n"
        "Q1 a b c 0 qv area=2 ic=0.7,5.0 off\n"
        ".ends\n"
    )
    found = _devices(lib.component("s"))
    assert set(found) == {"M1", "M2", "Q1"}


def test_folded_behavioral_capacitor_handles_wide_expression_in_linear_time() -> None:
    terms = " + ".join(["1p"] * 200)
    lib = library(f".subckt s a\nC1 a 0 {{1n * V(a) + {terms}}}\n.ends\n")
    found = _devices(lib.component("s"))
    assert found["C1"][0] == "behavioral_capacitor"


def test_transmission_line_maps_four_terminals_in_port_order() -> None:
    lib = library(".subckt s p1 n1 p2 n2\nT1 p1 n1 p2 n2 Z0=50 TD=2n\n.ends\n")
    (inst,) = lib.component("s").definition.instances
    assert inst.of == ir.DeviceOf(model="transmission_line")
    assert inst.terminals == {
        "positive1": "p1",
        "negative1": "n1",
        "positive2": "p2",
        "negative2": "n2",
    }


def test_expressions_translate_guards_exponential_expansion() -> None:
    nested_int: Expression = Voltage("a", None)
    for _ in range(30):
        nested_int = Call("int", (nested_int,))
    with pytest.raises(SpiceError, match="exceeds the limit"):
        translate(nested_int, {}, str, str)

    nested_pwrs: Expression = Voltage("a", None)
    for _ in range(30):
        nested_pwrs = Call("pwrs", (nested_pwrs, Number(2.0)))
    with pytest.raises(SpiceError, match="exceeds the limit"):
        translate(nested_pwrs, {}, str, str)

    nested_odd_pow: Expression = Voltage("a", None)
    for _ in range(20):
        nested_odd_pow = Binary("^", nested_odd_pow, Number(3.0))
    with pytest.raises(SpiceError, match="exceeds the limit"):
        translate(nested_odd_pow, {}, str, str)


def test_expressions_translate_minimal_parentheses_allows_flat_300_term_sum() -> None:
    flat_sum = " + ".join(["1"] * 300)
    parsed = parse(flat_sum)
    written = translate(parsed, {}, str, str)
    assert nesting_depth(written) == 0
    assert evaluate(parsed, {}) == 300.0

    assert translate(parse("(v(a) + v(b)) * v(c)"), {}, str, str) == (
        "(v(a) + v(b)) * v(c)"
    )
    assert translate(parse("v(a) + v(b) * v(c)"), {}, str, str) == (
        "v(a) + v(b) * v(c)"
    )
    assert translate(parse("v(a) - (v(b) - v(c))"), {}, str, str) == (
        "v(a) - (v(b) - v(c))"
    )
    assert translate(parse("v(a) - v(b) - v(c)"), {}, str, str) == (
        "v(a) - v(b) - v(c)"
    )
    assert translate(parse("v(a) / (v(b) * v(c))"), {}, str, str) == (
        "v(a) / (v(b) * v(c))"
    )
    assert translate(parse("-(v(a) + v(b))"), {}, str, str) == "-(v(a) + v(b))"
    assert translate(parse("-(-v(a))"), {}, str, str) == "- -v(a)"
    assert translate(Number(-0.0), {}, str, str) == "(-0.0)"


@pytest.mark.parametrize(
    "text",
    [
        "1e500",
        "-1e500",
        "1e308k",
        "-1e308meg",
        "1m\u0130l",
        "1m\u0131l",
    ],
)
def test_expressions_number_rejects_non_finite_and_unicode_suffixes(text: str) -> None:
    assert number(text) is None


def test_expressions_number_underflow_and_parse_non_finite() -> None:
    assert number("1e-500") == 0.0
    with pytest.raises(SpiceError, match="not finite"):
        parse("1e500")


def test_expressions_iterative_repr_handles_deep_chain_and_matches_dataclass() -> None:
    chain: Expression = Number(1.0)
    for index in range(1500):
        chain = Binary("+", chain, Number(float(index)))
    text = repr(chain)
    assert text.startswith("Binary(operator='+', left=")
    assert text.endswith("right=Number(value=1499.0))")

    assert repr(Call("f", ())) == "Call(function='f', arguments=())"
    assert repr(Call("f", (Number(1.0),))) == (
        "Call(function='f', arguments=(Number(value=1.0),))"
    )
    assert repr(Call("f", (Number(1.0), Number(2.0)))) == (
        "Call(function='f', arguments=(Number(value=1.0), Number(value=2.0)))"
    )
    assert repr(Unary("-", Name("x"))) == "Unary(operator='-', operand=Name(name='x'))"
    assert repr(Conditional(Name("c"), Number(1.0), Number(0.0))) == (
        "Conditional(condition=Name(name='c'), then=Number(value=1.0), "
        "otherwise=Number(value=0.0))"
    )


def test_expressions_integral_handles_wide_non_polynomial_sum() -> None:
    var = Voltage("a", None)
    expr: Expression = Call("sin", (Binary("*", Number(2.0), var),))
    for _ in range(300):
        expr = Binary("+", expr, Number(1.0))
    antiderivative = integral(expr, var)
    assert antiderivative is not None
    # Finite-sum antiderivative evaluated via _core.constant: -cos(2x)/2 + 300x.
    at_one = translate(antiderivative, {}, lambda _n: "x", str)
    evaluated = _core.constant(at_one.replace("v(x)", "1.0"))
    expected = -math.cos(2.0) / 2.0 + 300.0
    # IEEE-754 floating-point summation of 300 terms matches closed form within 1e-12.
    assert math.isclose(evaluated, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_models_ako_handles_bare_overrides_attached_parens_and_spaces() -> None:
    lib = library(
        ".model D1 D(IS=1e-14 N=1.5)\n"
        ".model D2 AKO:D1 IS=1e-15\n"
        ".model D3 AKO:D1 IS = 2e-15\n"
        ".model D4 AKO:D1(IS=3e-15)\n"
        ".model D5 AKO:D1  D(IS=4e-15)\n"
        ".model D6 AKO: D1 IS=5e-15\n"
        ".model M1 VDMOS(VTO=2.0)\n"
        ".model M2 AKO:M1 pchan VTO=-2.5\n"
        ".model M3 AKO:M2 nchan VTO=1.8\n"
        ".subckt s a b\n"
        "D2 a b D2\n"
        "D3 a b D3\n"
        "D4 a b D4\n"
        "D5 a b D5\n"
        "D6 a b D6\n"
        "M2 a b 0 M2\n"
        "M3 a b 0 M3\n"
        ".ends\n"
    )
    found = _devices(lib.component("s"))
    assert found["D2"][1]["emission_coefficient"] == 1.5
    for name, expected_is in (
        ("D2", 1e-15),
        ("D3", 2e-15),
        ("D4", 3e-15),
        ("D5", 4e-15),
        ("D6", 5e-15),
    ):
        assert found[name][1]["saturation_current"] == expected_is
    assert found["M2"][1] == {"channel": "p", "threshold_voltage": -2.5}
    assert found["M3"][1] == {"channel": "n", "threshold_voltage": 1.8}

    with pytest.raises(SpiceError, match="AKO:"):
        library(".model Dbad AKO:\n")


@pytest.mark.parametrize(
    "card_line",
    [
        ".model m1 d(is=1e-14 nlev=2)\n.subckt s a b\nD1 a b m1\n.ends\n",
        ".model m1 d(is=1e-14 gdsnoi=1)\n.subckt s a b\nD1 a b m1\n.ends\n",
        ".model m1 npn(is=1e-16 nlev=2)\n.subckt s c b e\nQ1 c b e m1\n.ends\n",
        ".model m1 njf(nlev=2)\n.subckt s d g so\nJ1 d g so m1\n.ends\n",
        ".model m1 vdmos(gdsnoi=1)\n.subckt s d g so\nM1 d g so m1\n.ends\n",
        ".model m1 npn(level=4 nlev=2)\n.subckt s c b e\nQ1 c b e 0 m1\n.ends\n",
        ".model m1 vswitch(nlev=2)\n.subckt s a b c d\nS1 a b c d m1\n.ends\n",
        (".model m1 iswitch(gdsnoi=1)\n.subckt s a b\nV1 a 0 1\nW1 b 0 V1 m1\n.ends\n"),
    ],
)
def test_models_noise_parameters_rejected_outside_mosfet_and_bsim3(
    card_line: str,
) -> None:
    lib = library(card_line)
    with pytest.raises(SpiceError):
        lib.component("s")


def test_models_noise_parameters_allowed_on_mosfet_and_bsim3() -> None:
    lib = library(
        ".model m1 nmos(level=1 nlev=2 gdsnoi=1)\n"
        ".model m3 nmos(level=49 version=3.3.0 nlev=2 gdsnoi=1)\n"
        ".subckt s d g so b\n"
        "M1 d g so b m1 w=1u l=1u\n"
        "M3 d g so b m3 w=1u l=1u\n"
        ".ends\n"
    )
    found = _devices(lib.component("s"))
    assert set(found) == {"M1", "M3"}


@pytest.mark.parametrize(
    "card",
    [
        ".model m nmos(level=49 binunit=3)",
        ".model m nmos(level=49 binunit=0)",
        ".model m nmos(level=49 binunit=1.5)",
        ".model m nmos(level=49 paramchk=2)",
        ".model m nmos(level=49 paramchk=-1)",
        ".model m nmos(level=54 version=4.5 binunit=3)",
        ".model m nmos(level=54 version=4.5 paramchk=2)",
        ".model m nmos(level=49 binunit={1/0})",
        ".model m nmos(level=49 paramchk={0/0})",
    ],
)
def test_models_bsim_rejects_invalid_binunit_and_paramchk(card: str) -> None:
    lib = library(f"{card}\n.subckt s d g so b\nM1 d g so b m w=1u l=1u\n.ends\n")
    with pytest.raises(SpiceError):
        lib.component("s")


def test_models_version_strips_surrounding_quotes() -> None:
    single = models.read(Statement("m.lib", 1, ".model m1 nmos(version='3.3.0')"), {})
    double = models.read(Statement("m.lib", 1, '.model m2 nmos(version="3.3.0")'), {})
    assert single.version == "3.3.0"
    assert double.version == "3.3.0"


def test_export_flatten_handles_deep_hierarchy_iteratively() -> None:
    depth = 1200
    components: dict[str, ir.Component] = {
        "c0": ir.Component(
            ports=(
                ir.Port(name="p", role=Role.SIGNAL),
                ir.Port(name="ground", role=Role.GROUND),
            ),
            instances=(
                ir.Instance(
                    name="R1",
                    of=ir.DeviceOf(model="resistor"),
                    parameters={"resistance": 1000.0},
                    terminals={"a": "p", "b": "ground"},
                ),
            ),
        )
    }
    for level in range(1, depth):
        components[f"c{level}"] = ir.Component(
            ports=(
                ir.Port(name="p", role=Role.SIGNAL),
                ir.Port(name="ground", role=Role.GROUND),
            ),
            instances=(
                ir.Instance(
                    name="X1",
                    of=ir.ComponentOf(component=f"c{level - 1}"),
                    terminals={"p": "p", "ground": "ground"},
                ),
            ),
        )
    flat = export.flatten(ir.Circuit(top=f"c{depth - 1}", components=components))
    assert len(flat.components[flat.top].instances) == 1


def test_export_written_is_iterative_and_guards_exponential_blowup() -> None:
    def render(tree: JsonValue) -> str:
        return export._written(tree, lambda _node: "v(a)")  # noqa: SLF001 - test internal writer

    # 1200-deep left-leaning sum must not raise RecursionError.
    deep: JsonValue = 1.0
    for _ in range(1200):
        deep = {"operation": "+", "operands": [deep, 1.0]}
    assert render(deep).count("+") == 1200

    # Nested limit (2^d / 4^d) and nested tables blow up exponentially if unguarded.
    nested_limit: JsonValue = {"voltage": "a"}
    for _ in range(18):
        nested_limit = {"operation": "limit", "operands": [0.0, nested_limit, 1.0]}
    nested_both: JsonValue = {"voltage": "a"}
    for _ in range(10):
        nested_both = {
            "operation": "limit",
            "operands": [0.0, nested_both, nested_both],
        }
    nested_table: JsonValue = {"voltage": "a"}
    for _ in range(12):
        nested_table = {
            "table": nested_table,
            "abscissae": [0.0, 1.0, 2.0, 3.0],
            "ordinates": [0.0, 1.0, 0.5, 2.0],
            "rounding": 0.1,
        }
    short_table: JsonValue = {
        "table": {"voltage": "a"},
        "abscissae": [0.0],
        "ordinates": [1.0],
        "rounding": 0.0,
    }
    for tree, match in (
        (nested_limit, "exceeds the limit"),
        (nested_both, "exceeds the limit"),
        (nested_table, "exceeds the limit"),
        (short_table, "at least 2 points"),
    ):
        with pytest.raises(ValueError, match=match):
            render(tree)


def _single_component_circuit(
    *instances: ir.Instance,
    ports: tuple[ir.Port, ...] = (ir.Port(name="ground", role=Role.GROUND),),
) -> ir.Circuit:
    return ir.Circuit(
        top="top",
        components={"top": ir.Component(ports=ports, instances=instances)},
    )


def test_export_passive_uses_explicit_primary_parameter_regardless_of_key_order() -> (
    None
):
    design = _single_component_circuit(
        ir.Instance(
            name="R1",
            of=ir.DeviceOf(model="resistor"),
            parameters={"temperature_coefficient": 1e-3, "resistance": 1000.0},
            terminals={"a": "n1", "b": "ground"},
        ),
        ir.Instance(
            name="C1",
            of=ir.DeviceOf(model="capacitor"),
            parameters={"temperature_coefficient": 2e-4, "capacitance": 1e-6},
            terminals={"a": "n1", "b": "ground"},
        ),
        ir.Instance(
            name="L1",
            of=ir.DeviceOf(model="inductor"),
            parameters={"temperature_coefficient": 3e-4, "inductance": 1e-3},
            terminals={"a": "n1", "b": "ground"},
        ),
    )
    deck = netlist(design)
    assert any(line.startswith("RR1 n1 0 1000.0 tc1=0.001") for line in deck)
    assert any(line.startswith("CC1 n1 0 1e-06 tc1=0.0002") for line in deck)
    assert any(line.startswith("LL1 n1 0 0.001 tc1=0.0003") for line in deck)


def test_export_netlist_rejects_missing_ground_port_with_jolty_error() -> None:
    ungrounded = _single_component_circuit(
        ir.Instance(
            name="R1",
            of=ir.DeviceOf(model="resistor"),
            parameters={"resistance": 100.0},
            terminals={"a": "a", "b": "b"},
        ),
        ports=(ir.Port(name="a", role=Role.SIGNAL),),
    )
    with pytest.raises(JoltyError, match="no ground port"):
        netlist(ungrounded)


def test_export_diode_capacitance_singular_shift_raises_jolty_error() -> None:
    singular = _single_component_circuit(
        ir.Instance(
            name="D1",
            of=ir.DeviceOf(model="diode"),
            parameters={
                "capacitance_temperature_model": 1,
                "nominal_temperature": _core.REFERENCE_TEMPERATURE + 8.0,
                "junction_capacitance": 1e-12,
                "junction_capacitance_temperature_coefficient": 0.125,
            },
            terminals={"anode": "a", "cathode": "ground"},
        )
    )
    with pytest.raises(JoltyError, match="singular"):
        netlist(singular)


def test_export_node_voltages_flattens_once_for_kept_and_flattened_subcircuits() -> (
    None
):
    design = ir.Circuit(
        top="top",
        components={
            "stage": ir.Component(
                ports=(
                    ir.Port(name="inp", role=Role.SIGNAL),
                    ir.Port(name="ground", role=Role.GROUND),
                ),
                instances=(
                    ir.Instance(
                        name="R1",
                        of=ir.DeviceOf(model="resistor"),
                        parameters={"resistance": 1000.0},
                        terminals={"a": "inp", "b": "mid"},
                    ),
                    ir.Instance(
                        name="R2",
                        of=ir.DeviceOf(model="resistor"),
                        parameters={"resistance": 1000.0},
                        terminals={"a": "mid", "b": "ground"},
                    ),
                ),
            ),
            "top": ir.Component(
                ports=(ir.Port(name="ground", role=Role.GROUND),),
                instances=(
                    ir.Instance(
                        name="U1",
                        of=ir.ComponentOf(component="stage"),
                        terminals={"inp": "a", "ground": "ground"},
                    ),
                    ir.Instance(
                        name="R3",
                        of=ir.DeviceOf(model="resistor"),
                        parameters={"resistance": 500.0},
                        terminals={"a": "a", "b": "ground"},
                    ),
                ),
            ),
        },
    )
    flat_lines = export.node_voltages(".ic", design, {"a": 1.0, "U1/mid": 0.5})
    assert flat_lines == [".ic v(a)=1.0 v(u1_mid)=0.5"]
    kept_lines = export.node_voltages(
        ".nodeset", design, {"a": 1.0, "U1/mid": 0.5}, subcircuits=("stage",)
    )
    assert kept_lines == [".nodeset v(a)=1.0 v(xu1.mid)=0.5"]


def test_export_spice_value_rejects_booleans() -> None:
    for flag in (True, False):
        with pytest.raises(JoltyError, match="no SPICE value"):
            export._spice_value("level", flag, 1.0, float)  # noqa: SLF001 - bool guard test


@pytest.mark.parametrize(
    "bad_digest",
    [
        "../../etc/passwd",
        "abc",
        "0" * 63,
        "0" * 65,
        "g" * 64,
        f"{'0' * 64}\n",
    ],
)
def test_download_rejects_malformed_digest(tmp_path: Path, bad_digest: str) -> None:
    published = tmp_path / "part.lib"
    published.write_text("ok")
    with pytest.raises(DownloadError, match="64 hexadecimal digits"):
        spice.fetch(published.as_uri(), bad_digest, tmp_path / "cache")


@pytest.mark.parametrize(
    "bad_url",
    [
        "https://example.com/",
        "https://example.com/dir/",
        "https://example.com/.",
        "https://example.com/foo/.",
        "https://example.com/..",
        "https://example.com/foo/..",
    ],
)
def test_download_rejects_unsafe_url_filename(tmp_path: Path, bad_url: str) -> None:
    with pytest.raises(DownloadError, match="names no safe file"):
        spice.fetch(bad_url, "0" * 64, tmp_path / "cache")


def test_download_cleans_partial_files_on_digest_mismatch_and_size_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published = tmp_path / "part.lib"
    published.write_bytes(b"x" * 128)
    cache = tmp_path / "cache"
    with pytest.raises(DownloadError, match="has SHA-256 digest"):
        spice.fetch(published.as_uri(), "0" * 64, cache)
    assert not list(cache.rglob("*.partial"))

    digest = hashlib.sha256(published.read_bytes()).hexdigest()
    monkeypatch.setattr(download, "_MAX_DOWNLOAD_BYTES", 32)
    with pytest.raises(DownloadError, match="exceeds maximum download size"):
        spice.fetch(published.as_uri(), digest, cache)
    assert not list(cache.rglob("*.partial"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"fingers": 0},
        {"fingers": -1},
        {"fingers": True},
        {"width": 0.0},
        {"width": -1e-6},
        {"width": float("inf")},
        {"width": float("nan")},
        {"width": True},
        {"length": 0.0},
        {"length": float("inf")},
        {"length": False},
        {"multiplier": 0},
        {"multiplier": True},
    ],
)
def test_sky130_written_rejects_invalid_and_boolean_parameters(
    overrides: dict[str, JsonValue],
) -> None:
    base: dict[str, JsonValue] = {
        "pdk": "sky130",
        "model": "sky130_fd_pr__nfet_01v8",
        "width": 1e-6,
        "length": 1.5e-7,
        "fingers": 1,
        "multiplier": 1,
    }
    base.update(overrides)
    with pytest.raises(ValueError, match=r"SKY130|positive|1 or more"):
        sky130.written(base)


def test_sky130_mosfet_rejects_non_finite_size_and_boolean_counts() -> None:
    with pytest.raises(ValueError, match="finite"):
        sky130.mosfet("nfet_01v8", float("inf") * um, 0.15 * um)
    with pytest.raises(ValueError, match="1 or more"):
        sky130.mosfet("nfet_01v8", 1 * um, 0.15 * um, fingers=True)
    with pytest.raises(ValueError, match="1 or more"):
        sky130.mosfet("nfet_01v8", 1 * um, 0.15 * um, multiplier=False)


def test_ngspice_runs_rejects_empty_deck_and_quotes_include_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(JoltyError, match="deck is empty"):
        ngspice.runs([], ("op",))

    captured: list[str] = []
    spaced_dir = tmp_path / "dir with spaces"
    spaced_dir.mkdir()
    lib_file = spaced_dir / "models.lib"
    lib_file.write_text(".model d d\n")

    raw_bytes = (
        b"Title: test\n"
        b"Date: today\n"
        b"Plotname: Operating Point\n"
        b"Flags: real\n"
        b"No. Variables: 1\n"
        b"No. Points: 1\n"
        b"Variables:\n"
        b"\t0\tv(a)\tvoltage\n"
        b"Values:\n"
        b" 0\t1.0\n"
    )

    def fake_execute(
        deck: Sequence[str],
        *,
        compatibility: Compatibility | None = None,
        timeout: float | None = None,
    ) -> ngspice.Executed:
        del compatibility, timeout
        captured.extend(deck)
        return ngspice.Executed(printed="", code=0, seconds=0.01, values=raw_bytes)

    monkeypatch.setattr(ngspice, "execute", fake_execute)
    (result,) = ngspice.runs(
        ["test title", "R1 a 0 1k"], ("op",), libraries=(lib_file,)
    )
    assert result.plot == "Operating Point"
    assert f'.include "{lib_file.resolve()}"' in captured


def test_ngspice_parse_raw_checks_ascii_row_index_and_reads_binary_complex() -> None:
    bad_ascii = (
        "Title: test\n"
        "Plotname: Operating Point\n"
        "Flags: real\n"
        "No. Variables: 1\n"
        "No. Points: 2\n"
        "Variables:\n"
        "\t0\tv(a)\tvoltage\n"
        "Values:\n"
        " 0\t1.0\n"
        " 5\t2.0\n"
    )
    with pytest.raises(JoltyError, match="row index 5, not 1"):
        ngspice.parse_raw(bad_ascii)

    def binary_header(points: int) -> bytes:
        return (
            "Title: ac\nPlotname: AC Analysis\nFlags: complex\nNo. Variables: 2\n"
            f"No. Points: {points}\n"
            "Variables:\n\t0\tfrequency\tfrequency\n\t1\tv(out)\tvoltage\nBinary:\n"
        ).encode("ascii")

    payload = struct.pack("<8d", 100.0, 0.0, -0.0, -2.5, 1000.0, 0.0, 0.75, 0.25)
    (plot,) = ngspice.parse_raw(binary_header(2) + payload)
    assert plot.values.dtype == np.complex128
    assert plot.values.shape == (2, 2)
    assert plot.values[0, 0] == 100.0 + 0.0j
    assert math.copysign(1.0, float(plot.values[0, 1].real)) == -1.0
    assert plot.values[0, 1] == complex(-0.0, -2.5)
    assert plot.values[1, 1] == 0.75 + 0.25j

    (empty_plot,) = ngspice.parse_raw(binary_header(0))
    assert empty_plot.values.dtype == np.complex128
    assert empty_plot.values.shape == (0, 2)


def test_xyce_run_rejects_empty_analysis_and_parses_complex_csv_with_blank_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for blank in ("", "   \t\n"):
        with pytest.raises(JoltyError, match="analysis command is empty"):
            xyce.run(["* title"], blank)

    printed = (
        "Total Simulation Solvers Run Time: 0.12 seconds\n"
        "Total Elapsed Run Time: 0.25 seconds\n"
    )

    def stub(values: str) -> None:
        def fake_execute(
            deck: Sequence[str],
            *,
            launch: Sequence[str] | None = None,
            timeout: float | None = None,
        ) -> xyce.Executed:
            del deck, launch, timeout
            return xyce.Executed(printed=printed, code=0, seconds=0.3, values=values)

        monkeypatch.setattr(xyce, "execute", fake_execute)

    stub("\nFREQ,Re(V(OUT)),Im(V(OUT))\n\n100.0,0.5,-0.25\n1000.0,0.25,0.75\n\n")
    result = xyce.run(["* title", "R1 out 0 1k"], ".ac dec 10 100 1k")
    assert list(result.vectors["frequency"]) == [100.0 + 0j, 1000.0 + 0j]
    assert list(result.vectors["v(out)"]) == [0.5 - 0.25j, 0.25 + 0.75j]

    # Blank-only CSV triggers "printed no values" instead of ValueError.
    stub("\n\n")
    with pytest.raises(JoltyError, match="printed no values"):
        xyce.run(["* title", "R1 out 0 1k"], ".op")


def test_memristor_transient_export_emits_numeric_state_capacitance() -> None:
    @component
    def memristor_bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(1 * V), name="V1")
        memristor = c.place(
            devices.voltage_controlled_memristor(
                on_resistance=100 * ohm,
                off_resistance=10 * kohm,
                on_voltage=-0.5 * V,
                off_voltage=0.5 * V,
                on_rate=10 / s,
                off_rate=10 / s,
            ),
            name="M1",
        )
        c.connect(source.positive, memristor.positive, name="in")
        c.connect(memristor.negative, source.negative, ground)

    lines = export.netlist(circuit(memristor_bench()), transient=True)
    rendered = "\n".join(lines)
    assert "Quantity" not in rendered
    assert any(
        line.startswith("CM1_state _M1_state 0 ")
        and float(line.split()[-1]) == export.STATE_CAPACITANCE
        for line in lines
    )


NESTED_MISMATCHED: Final = (
    ".subckt inner a b\n"
    ".param spread = agauss(1, 0.1, 1)\n"
    "R1 a b {1k * spread}\n"
    ".ends\n"
    ".subckt outer a b\n"
    "Xinner a b inner\n"
    ".ends\n"
)
"""Nested subcircuit whose inner placement draws from `agauss`."""


def test_nested_subcircuit_drawn_and_mismatch_encode_paths_without_collision() -> None:
    """Verifies nested `.subckt` draws elaborate cleanly and match closed-form mismatch.

    Inner placement paths contain `'/'` for statistical variate keying while component
    names encode `'/'` so `jolty_ir::Name` validates every nested subcircuit under both
    `Sample` and `Probe`. Linear divider tolerances cover LU and central-difference
    rounding (`LU_ROUNDING_RTOL` and `rel=1e-9` for `O(eps^(2/3))` finite differences).
    """

    def bench(part: Component) -> Component:
        builder = Builder("nested_bench")
        ground = builder.port("ground", Role.GROUND)
        source = builder.place(devices.voltage_source(1 * V), name="V1")
        upper = builder.place(devices.resistor(1 * kohm), name="Rtop")
        lower = builder.place(part, name="X1")
        builder.connect(source.positive, upper.a, name="in")
        builder.connect(upper.b, lower.a, name="out")
        builder.connect(lower.b, source.negative, ground)
        return builder.finish()

    first, second = library(NESTED_MISMATCHED), library(NESTED_MISMATCHED)
    sample = spice.Sample(seed=13, index=5)
    placed = first.drawn("outer", sample, "X1")
    (child,) = placed.children
    assert "/" not in child.name
    assert "X1%2FXinner" in child.name
    assert all("/" not in key for key in circuit(bench(placed)).components)

    # Statistical draws continue to key on the raw `'/'`-separated path bit-identically.
    expected_normal = _core.variate("normal", 13, 5, "X1/Xinner", "spread#0")
    expected_resistance = 1e3 * (1.0 + 0.1 * expected_normal)
    (inner_resistor,) = child.definition.instances
    assert inner_resistor.parameters["resistance"] == expected_resistance

    point = Simulation(bench(placed)).run(OperatingPoint())
    expected_voltage = expected_resistance / (1e3 + expected_resistance)
    assert point.voltage("out") == pytest.approx(expected_voltage, rel=LU_ROUNDING_RTOL)
    repeated = second.drawn("outer", sample, "X1")
    assert Simulation(bench(repeated)).run(OperatingPoint()).voltage(
        "out"
    ) == point.voltage("out")
    sibling = first.drawn("outer", sample, "X2")
    assert Simulation(bench(sibling)).run(OperatingPoint()).voltage(
        "out"
    ) != point.voltage("out")
    next_sample = first.drawn("outer", spice.Sample(seed=13, index=6), "X1")
    assert Simulation(bench(next_sample)).run(OperatingPoint()).voltage(
        "out"
    ) != point.voltage("out")

    deep = first.drawn("outer", sample, "stage1/X1")
    assert all("/" not in key for key in circuit(bench(deep)).components)
    assert math.isfinite(Simulation(bench(deep)).run(OperatingPoint()).voltage("out"))

    (variable,) = first.random_variables("outer", "X1")
    assert variable.instance == "X1/Xinner"
    assert variable.name == "spread#0"
    assert variable.mean == 1.0
    assert variable.standard_deviation == 0.1
    assert variable.derivatives == {
        "X1/Xinner/R1": {"resistance": pytest.approx(1e3, rel=1e-9)}
    }
    mismatch = Simulation(bench(first.component("outer"))).run(
        Mismatch(["out"], random_variables=[variable])
    )
    assert mismatch.standard_deviation("out") == pytest.approx(0.025, rel=1e-9)


def test_drawn_component_names_never_collide_across_special_paths() -> None:
    """Verifies percent-encoded `Drawing` paths stay distinct and valid IR names."""
    deterministic = library(".subckt cell a b\nR1 a b 1k\n.ends\n")
    sample = spice.Sample(seed=1, index=0)
    special_paths = ("a/b", "a_b", "a.b", "a%2Fb", "a b", "a\tb")
    drawn_cells = [deterministic.drawn("cell", sample, path) for path in special_paths]
    names = [part.name for part in drawn_cells]
    assert len(set(names)) == len(special_paths)
    assert all(
        "/" not in name and not any(c.isspace() or not c.isprintable() for c in name)
        for name in names
    )

    builder = Builder("special_paths")
    ground = builder.port("ground", Role.GROUND)
    source = builder.place(devices.voltage_source(1 * V), name="V1")
    for index, part in enumerate(drawn_cells):
        placed = builder.place(part, name=f"U{index}")
        builder.connect(source.positive, placed.a)
        builder.connect(placed.b, source.negative, ground)
    assert math.isfinite(
        Simulation(builder.finish()).run(OperatingPoint()).current("V1")
    )

    # Nested path `X1/X2` and flat path `X1_X2` with different draws never collide.
    nested = library(
        ".subckt leaf a b\n.param k = agauss(1, 0.1, 1)\nR1 a b {1k * k}\n.ends\n"
        ".subckt wrap a b\nX2 a b leaf\n.ends\n"
    )
    pair_builder = Builder("nested_vs_flat")
    pair_ground = pair_builder.port("ground", Role.GROUND)
    pair_source = pair_builder.place(devices.voltage_source(1 * V), name="V1")
    wrapped = pair_builder.place(nested.drawn("wrap", sample, "X1"), name="X1")
    flat = pair_builder.place(nested.drawn("leaf", sample, "X1_X2"), name="X1_X2")
    pair_builder.connect(pair_source.positive, wrapped.a, flat.a)
    pair_builder.connect(wrapped.b, flat.b, pair_source.negative, pair_ground)
    built_pair = pair_builder.finish()
    assert all("/" not in key for key in circuit(built_pair).components)
    assert math.isfinite(Simulation(built_pair).run(OperatingPoint()).current("V1"))

    # Probe path encoding works when caller supplies a `'/'`-separated placement path.
    (probed,) = nested.random_variables("wrap", "stage/X1")
    assert probed.instance == "stage/X1/X2"
    assert set(probed.derivatives) == {"stage/X1/X2/R1"}


def test_resistor_cards_accept_tref_as_tnom_alias_and_scale_temperature() -> None:
    """Verifies SPICE3 `R` and PSpice `RES` cards map `TREF` identically to `TNOM`.

    Temperature-scaling simulation uses `LU_ROUNDING_RTOL` (1e-12) for exact linear
    conductance rounding and `KELVIN_ROUNDING_RTOL` for Celsius-to-Kelvin addition.
    """
    parts = library(
        ".model rm_tref r rsh=100 tc1r=2e-3 tref=30\n"
        ".model rm_tnom r rsh=100 tc1r=2e-3 tnom=30\n"
        ".model rm_monte r tref=30.0\n"
        ".model rx_tref RES(TC1=2e-3 TREF=30)\n"
        ".model rx_tnom RES(TC1=2e-3 TNOM=30)\n"
        ".subckt cards a\n"
        "R1 a 0 rm_tref w=2u l=10u\n"
        "R2 a 0 rm_tnom w=2u l=10u\n"
        "R3 a 0 rm_monte r=1k tc1=2e-3\n"
        "R4 a 0 rx_tref 1k\n"
        "R5 a 0 rx_tnom 1k\n"
        ".ends\n"
        ".subckt single_tref a\nR1 a 0 rm_monte r=1k tc1=2e-3\n.ends\n"
        ".subckt single_tnom a\nR1 a 0 rm_tnom r=1k tc1=2e-3\n.ends\n"
    )
    found = {
        instance.name: instance.parameters
        for instance in parts.component("cards").definition.instances
    }
    assert found["R1"] == found["R2"]
    assert found["R1"]["nominal_temperature"] == pytest.approx(
        celsius(30).value, rel=KELVIN_ROUNDING_RTOL
    )
    assert found["R3"] == {
        "resistance": 1e3,
        "temperature_coefficient": 2e-3,
        "nominal_temperature": pytest.approx(
            celsius(30).value, rel=KELVIN_ROUNDING_RTOL
        ),
    }
    assert found["R4"] == found["R5"]

    def drive_single(subcircuit_name: str) -> float:
        builder = Builder(f"drive_{subcircuit_name}")
        ground = builder.port("ground", Role.GROUND)
        source = builder.place(devices.voltage_source(1 * V), name="V1")
        placed = builder.place(parts.component(subcircuit_name), name="U1")
        builder.connect(source.positive, placed.a, name="drive")
        builder.connect(source.negative, placed[spice.GROUND], ground)
        point = Simulation(builder.finish(), temperature=celsius(80)).run(
            OperatingPoint()
        )
        return point.current("V1")

    current_tref = drive_single("single_tref")
    current_tnom = drive_single("single_tnom")
    assert current_tref == current_tnom
    # At 80 C with TREF = 30 C and tc1 = 2e-3 / K, R = 1000 * (1 + 0.1) = 1100 ohm.
    assert -current_tref == pytest.approx(1.0 / 1100.0, rel=LU_ROUNDING_RTOL)


@pytest.mark.parametrize(
    ("deck", "match"),
    [
        (
            ".model rmod r tnom=25 tref=30\n.subckt a x\nR1 x 0 rmod r=1k\n.ends",
            "sets its nominal temperature twice: `TNOM`, `TREF`",
        ),
        (
            ".model rmod r tref=30 tnom=25\n.subckt a x\nR1 x 0 rmod r=1k\n.ends",
            "sets its nominal temperature twice: `TREF`, `TNOM`",
        ),
        (
            ".model rmod RES(TNOM=25 TREF=30)\n.subckt a x\nR1 x 0 rmod 1k\n.ends",
            "sets its nominal temperature twice: `TNOM`, `TREF`",
        ),
        (
            (
                ".model rmod RES(T_MEASURED=25 TREF=30)\n"
                ".subckt a x\nR1 x 0 rmod 1k\n.ends"
            ),
            "sets its nominal temperature twice: `T_MEASURED`, `TREF`",
        ),
        (
            ".model qmod npn tnom=25 tref=30\n.subckt a c\nQ1 c 0 0 qmod\n.ends",
            "sets its nominal temperature twice: `TNOM`, `TREF`",
        ),
        (
            ".model dmod d tnom=25 tref=30\n.subckt a x\nD1 x 0 dmod\n.ends",
            "sets its nominal temperature twice: `TNOM`, `TREF`",
        ),
        (
            ".model cmod c tref=30\n.subckt a x\nC1 x 0 cmod 1p\n.ends",
            "counterpart to `TREF`",
        ),
        (
            ".model cmod CAP(TREF=30)\n.subckt a x\nC1 x 0 cmod 1p\n.ends",
            "counterpart to `TREF`",
        ),
        (
            ".subckt a x\nR1 x 0 0\n.ends",
            "resistance is positive, not 0.0",
        ),
        (
            ".subckt a x\nR1 x 0 -10\n.ends",
            "resistance is positive, not -10.0",
        ),
        (
            ".subckt a x\nL1 x 0 0\n.ends",
            "inductance is positive, not 0.0",
        ),
        (
            ".subckt a x\nC1 x 0 -1p\n.ends",
            "capacitance is at least 0",
        ),
    ],
)
def test_nominal_temperature_conflicts_and_invalid_passive_values_are_rejected(
    deck: str, match: str
) -> None:
    with pytest.raises(SpiceError, match=match):
        library(deck).component("a")


def test_zero_capacitance_remains_accepted_in_spice() -> None:
    found = _devices(library(".subckt a x\nC1 x 0 0\n.ends").component("a"))
    assert found["C1"] == ("capacitor", {"capacitance": 0.0})


def _write_cached_library_fixture(directory: Path) -> tuple[Path, Path, Path]:
    """Creates a multi-file SPICE library exercising includes, sections, and cards."""
    included = directory / "leaf.inc"
    included.write_text(
        ".param leaf_resistance = 1k\n"
        ".func double(x) {2 * x}\n"
        ".model base_diode D(IS=1e-14 RS=10)\n"
        ".model derived_diode AKO:base_diode D(N=1.5)\n"
        ".model nch.1 nmos(level=54 version=4.5 lmin=0.1u lmax=10u "
        "wmin=0.1u wmax=1.2u vth0=0.4)\n"
        ".model nch.2 nmos(level=54 version=4.5 lmin=0.1u lmax=10u "
        "wmin=1.2u wmax=10u vth0=0.7)\n",
        encoding="utf-8",
    )
    sections_file = directory / "corners.lib"
    sections_file.write_text(
        ".lib nominal\n"
        ".param corner_scale = 1.0\n"
        ".include 'leaf.inc'\n"
        ".endl nominal\n"
        ".lib fast\n"
        ".param corner_scale = 0.8\n"
        ".include 'leaf.inc'\n"
        ".endl fast\n",
        encoding="utf-8",
    )
    main = directory / "main.lib"
    main.write_text(
        ".option scale=1u\n"
        ".lib 'corners.lib' nominal\n"
        ".subckt inner a b\n"
        ".func half(x) {x / 2}\n"
        "R1 a b {half(double(leaf_resistance) * corner_scale)}\n"
        ".ends\n"
        ".subckt stage d g s b flag=1\n"
        ".if (flag == 1)\n"
        "X1 d s inner\n"
        ".else\n"
        "Ralt d s 500\n"
        ".endif\n"
        "D1 d s derived_diode\n"
        "M1 d g s b nch l=1 w=2 nf=2\n"
        "Vsense d sensed 0\n"
        "Rsense sensed s 1k\n"
        "B1 g 0 I={V(d, s) > 0 ? 1e-6 * (-V(d) + abs(I(Vsense))) : 0}\n"
        ".ic V(d)=0.5\n"
        ".nodeset V(g)=0.3\n"
        ".ends\n",
        encoding="utf-8",
    )
    return main, sections_file, included


def test_disk_cache_hit_is_bit_identical_to_cold_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main, _, _ = _write_cached_library_fixture(tmp_path)
    cache_directory = tmp_path / "cache"

    cold = spice.load(main, compatibility="hsa", cache=cache_directory)
    cache_entries = list((cache_directory / "spice").glob("*.zlib"))
    assert len(cache_entries) == 1
    assert not list((cache_directory / "spice").glob("*.tmp"))

    def fail_on_reparse(*_args: object, **_kwargs: object) -> object:
        message = "expected disk cache hit without re-expanding SPICE files"
        raise AssertionError(message)

    monkeypatch.setattr(spice_library, "expanded", fail_on_reparse)
    warm = spice.load(main, compatibility="hsa", cache=cache_directory)

    assert warm.subcircuits == cold.subcircuits
    assert warm.models == cold.models
    assert warm.options == cold.options
    assert warm.values() == cold.values()
    assert circuit(warm.component("stage")) == circuit(cold.component("stage"))
    assert circuit(warm.component("stage", flag=0)) == circuit(
        cold.component("stage", flag=0)
    )
    assert warm.initial_conditions_of("stage") == cold.initial_conditions_of("stage")
    assert warm.nodesets_of("stage") == cold.nodesets_of("stage")
    for card_name in warm.models:
        card = warm.card(card_name, {})
        assert card is not None
        assert isinstance(card.parameters, MappingProxyType)


def test_disk_cache_invalidates_on_transitive_include_and_lib_edits(
    tmp_path: Path,
) -> None:
    main, sections_file, included = _write_cached_library_fixture(tmp_path)
    cache_directory = tmp_path / "cache"

    initial = spice.load(main, cache=cache_directory)
    assert _devices(initial.component("inner"))["R1"][1]["resistance"] == 1000.0

    # Same byte length edit in a transitively included file (`1k` -> `2k`).
    included.write_text(
        included.read_text(encoding="utf-8").replace("1k", "2k", 1),
        encoding="utf-8",
    )
    after_include_edit = spice.load(main, cache=cache_directory)
    assert (
        _devices(after_include_edit.component("inner"))["R1"][1]["resistance"] == 2000.0
    )

    # Edit inside the `.lib` section file (`1.0` -> `3.0`).
    sections_file.write_text(
        sections_file.read_text(encoding="utf-8").replace("1.0", "3.0", 1),
        encoding="utf-8",
    )
    after_section_edit = spice.load(main, cache=cache_directory)
    assert (
        _devices(after_section_edit.component("inner"))["R1"][1]["resistance"] == 6000.0
    )

    # Deleting a transitively included file invalidates and raises SpiceError.
    included.unlink()
    with pytest.raises(SpiceError, match="cannot read"):
        spice.load(main, cache=cache_directory)


def test_disk_cache_invalidates_on_symlink_retarget(tmp_path: Path) -> None:
    first_target = tmp_path / "first.inc"
    second_target = tmp_path / "second.inc"
    first_target.write_text(".param value = 100\n", encoding="utf-8")
    second_target.write_text(".param value = 250\n", encoding="utf-8")
    link = tmp_path / "active.inc"
    link.symlink_to(first_target)
    main = tmp_path / "main.lib"
    main.write_text(
        ".include 'active.inc'\n.subckt load a\nR1 a 0 {value}\n.ends\n",
        encoding="utf-8",
    )
    cache_directory = tmp_path / "cache"

    loaded_first = spice.load(main, cache=cache_directory)
    assert _devices(loaded_first.component("load"))["R1"][1]["resistance"] == 100.0

    link.unlink()
    link.symlink_to(second_target)
    loaded_second = spice.load(main, cache=cache_directory)
    assert _devices(loaded_second.component("load"))["R1"][1]["resistance"] == 250.0


def test_disk_cache_separates_sections_and_compatibility_modes(
    tmp_path: Path,
) -> None:
    main, sections_file, _ = _write_cached_library_fixture(tmp_path)
    cache_directory = tmp_path / "cache"

    default_mode = spice.load(main, compatibility=None, cache=cache_directory)
    hspice_mode = spice.load(main, compatibility="hsa", cache=cache_directory)
    assert _devices(default_mode.component("stage"))["M1"][1]["vth0"] == 0.7
    assert _devices(hspice_mode.component("stage"))["M1"][1]["vth0"] == 0.4

    nominal_section = spice.Library.from_sources(
        ((sections_file, "nominal"),), cache=cache_directory
    )
    fast_section = spice.Library.from_sources(
        ((sections_file, "fast"),), cache=cache_directory
    )
    assert nominal_section.values()["corner_scale"] == 1.0
    assert fast_section.values()["corner_scale"] == 0.8

    entries = list((cache_directory / "spice").glob("*.zlib"))
    assert len(entries) == 4


@pytest.mark.parametrize(
    "corrupt_bytes",
    [
        b"",
        b"not a cache file at all",
        b"JLTYSPC1truncated",
        b"JLTYSPC1" + zlib.compress(b"{not valid json"),
        b"JLTYSPC1" + zlib.compress(b"[]"),
        b"JLTYSPC1" + zlib.compress(b'{"format": 999}'),
        b"JLTYSPC1" + zlib.compress(b'{"format": 1, "manifest": [true]}'),
        b"JLTYSPC1" + zlib.compress(b'{"format": 1, "evaluated_temperature": NaN}'),
    ],
)
def test_disk_cache_recovers_from_corrupt_or_truncated_entries(
    tmp_path: Path, corrupt_bytes: bytes
) -> None:
    main, _, _ = _write_cached_library_fixture(tmp_path)
    cache_directory = tmp_path / "cache"
    key = _cache.cache_key(((main, None),), None)
    entry = _cache.cache_file(cache_directory, key)
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_bytes(corrupt_bytes)

    recovered = spice.load(main, cache=cache_directory)
    assert _devices(recovered.component("inner"))["R1"][1]["resistance"] == 1000.0

    # Healing overwrite wrote a valid entry that now passes read_cached directly.
    healed = _cache.read_cached(
        cache_directory,
        key,
        compatibility=None,
        temperature=_core.REFERENCE_TEMPERATURE,
    )
    assert healed is not None


def test_disk_cache_handles_top_level_evaluated_directives_across_temperatures(
    tmp_path: Path,
) -> None:
    deck = tmp_path / "top_ic.lib"
    deck.write_text(
        ".ic V(out)={temper / 100}\n"
        ".nodeset V(bias)={0.1 + temper / 1000}\n"
        ".pss {1e6 + 1e3 * temper} 1u\n"
        ".subckt load out bias\n"
        "R1 out 0 1k\n"
        "R2 bias 0 1k\n"
        ".ends\n",
        encoding="utf-8",
    )
    cache_directory = tmp_path / "cache"

    at_nominal = spice.load(deck, temperature=celsius(27).value, cache=cache_directory)
    at_hot = spice.load(deck, temperature=celsius(85).value, cache=cache_directory)
    at_hot_cached = spice.load(
        deck, temperature=celsius(85).value, cache=cache_directory
    )

    # Celsius conversion (`celsius(T) - 273.15`) introduces <= 1e-12 floating rounding.
    assert at_nominal.initial_conditions["out"] == pytest.approx(0.27, rel=1e-12)
    assert at_hot.initial_conditions["out"] == pytest.approx(0.85, rel=1e-12)
    assert at_hot_cached.initial_conditions == at_hot.initial_conditions
    assert at_hot_cached.nodesets == at_hot.nodesets
    assert at_hot_cached.periodic_steady_states == at_hot.periodic_steady_states


def test_disk_cache_shares_entry_across_temperatures_without_top_level_directives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deck = tmp_path / "thermal.lib"
    deck.write_text(
        ".subckt warm a b\nR1 a b {1k * (1 + temper / 100)}\n.ends\n",
        encoding="utf-8",
    )
    cache_directory = tmp_path / "cache"

    cold = spice.load(deck, temperature=celsius(27).value, cache=cache_directory)
    # Exact arithmetic `1000 * (1 + 27 / 100)`; rel=1e-12 covers Kelvin/Celsius offset.
    assert _devices(cold.component("warm"))["R1"][1]["resistance"] == pytest.approx(
        1270.0, rel=1e-12
    )

    def fail_on_reparse(*_args: object, **_kwargs: object) -> object:
        message = "expected shared disk cache hit across temperatures"
        raise AssertionError(message)

    monkeypatch.setattr(spice_library, "expanded", fail_on_reparse)
    hot = spice.load(deck, temperature=celsius(85).value, cache=cache_directory)
    # Exact arithmetic `1000 * (1 + 85 / 100)`; rel=1e-12 covers Kelvin/Celsius offset.
    assert _devices(hot.component("warm"))["R1"][1]["resistance"] == pytest.approx(
        1850.0, rel=1e-12
    )
    assert len(list((cache_directory / "spice").glob("*.zlib"))) == 1


def test_disk_cache_is_race_free_under_concurrent_loads(tmp_path: Path) -> None:
    main, _, _ = _write_cached_library_fixture(tmp_path)
    cache_directory = tmp_path / "cache"
    expected = circuit(
        spice.load(main, cache=tmp_path / "reference").component("stage")
    )

    workers = 8
    barrier = threading.Barrier(workers)

    def load_concurrent(_worker: int) -> ir.Circuit:
        barrier.wait()
        return circuit(spice.load(main, cache=cache_directory).component("stage"))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        loaded = list(pool.map(load_concurrent, range(workers)))

    assert all(built == expected for built in loaded)
    assert not list((cache_directory / "spice").glob("*.tmp"))


def test_disk_cache_uses_configured_jolty_cache_and_ignores_unwritable_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main, _, _ = _write_cached_library_fixture(tmp_path)
    configured = tmp_path / "configured_cache"
    monkeypatch.setenv("JOLTY_CACHE", str(configured))

    loaded = spice.load(main)
    assert _devices(loaded.component("inner"))["R1"][1]["resistance"] == 1000.0
    assert len(list((configured / "spice").glob("*.zlib"))) == 1

    # Passing a regular file as the cache root forces OSError on write; load succeeds.
    blocked = tmp_path / "not_a_directory"
    blocked.write_text("file", encoding="utf-8")
    fallback = spice.load(main, cache=blocked)
    assert _devices(fallback.component("inner"))["R1"][1]["resistance"] == 1000.0


def test_built_cache_keeps_temper_free_subcircuits_and_updates_temper_dependents() -> (
    None
):
    """Verifies `_built` decouples library temperature from `temper`-free subcircuits.

    Covers direct `temper`, transitive global parameters, subcircuit defaults,
    nested children, `.if` conditions, `.model` expressions, `.ic` expressions,
    voltage-dependent resistors with and without `TC1`, and dynamic MOSFET bin bounds.
    `rel=1e-12` covers Celsius/Kelvin conversion rounding (`T - 273.15`).
    """
    source = (
        ".param base = 1k\n"
        ".param drifted = {base + 10 * temper}\n"
        ".model d_fixed D(IS=1e-14 RS=5)\n"
        ".model d_warm D(IS=1e-14 RS={5 + 0.1 * temper})\n"
        ".model n_dyn.1 nmos(level=54 version=4.5 lmin=0.1u lmax=10u "
        "wmin=0.1u wmax={1u + 1e-8 * temper} vth0=0.4)\n"
        ".model n_dyn.2 nmos(level=54 version=4.5 lmin=0.1u lmax=10u "
        "wmin={1u + 1e-8 * temper} wmax=10u vth0=0.8)\n"
        ".subckt fixed_load a\n"
        "R1 a 0 {base}\n"
        "D1 a 0 d_fixed\n"
        "Rbeh a 0 r='1k * (1 + 0.1 * V(a, 0))'\n"
        ".ends\n"
        ".subckt direct_warm a\n"
        "R1 a 0 {1k * (1 + temper / 100)}\n"
        ".ends\n"
        ".subckt global_warm a\n"
        "R1 a 0 {drifted}\n"
        ".ends\n"
        ".subckt default_warm a r={base + temper}\n"
        "R1 a 0 {r}\n"
        ".ends\n"
        ".subckt parent_of_default_warm a\n"
        "X1 a default_warm\n"
        ".ends\n"
        ".subckt parent_of_overridden_default a\n"
        "X1 a default_warm r=750\n"
        ".ends\n"
        ".subckt branch_warm a\n"
        ".if (temper > 50)\n"
        "R1 a 0 2k\n"
        ".else\n"
        "R1 a 0 1k\n"
        ".endif\n"
        ".ends\n"
        ".subckt card_warm a\n"
        "D1 a 0 d_warm\n"
        ".ends\n"
        ".subckt ic_warm a\n"
        "R1 a 0 1k\n"
        ".ic V(a)={temper / 100}\n"
        ".ends\n"
        ".subckt vres_tc a\n"
        "R1 a 0 r='1k * (1 + 0.1 * V(a, 0))' tc1=1m\n"
        ".ends\n"
        ".subckt binned_warm d g s b\n"
        "M1 d g s b n_dyn l=1u w=1.5u\n"
        ".ends\n"
    )
    loaded = library(source)

    fixed_at_27 = loaded.component("fixed_load")
    overridden_parent_at_27 = loaded.component("parent_of_overridden_default")
    assert _devices(loaded.component("direct_warm"))["R1"][1][
        "resistance"
    ] == pytest.approx(1270.0, rel=1e-12)
    assert _devices(loaded.component("global_warm"))["R1"][1][
        "resistance"
    ] == pytest.approx(1270.0, rel=1e-12)
    parent_default_at_27 = loaded.component("parent_of_default_warm")
    assert _devices(parent_default_at_27.children[0])["R1"][1][
        "resistance"
    ] == pytest.approx(1027.0, rel=1e-12)
    assert _devices(loaded.component("branch_warm"))["R1"][1]["resistance"] == 1000.0
    assert _devices(loaded.component("card_warm"))["D1"][1][
        "series_resistance"
    ] == pytest.approx(7.7, rel=1e-12)
    assert loaded.initial_conditions_of("ic_warm")["a"] == pytest.approx(
        0.27, rel=1e-12
    )
    vres_27 = circuit(loaded.component("vres_tc"))
    assert _devices(loaded.component("binned_warm"))["M1"][1]["vth0"] == 0.8

    # Raise temperature to 85 C (358.15 K) and compare every subcircuit against a fresh
    # uncached Library constructed at 85 C.
    hot_kelvins = celsius(85).value
    loaded.temperature = hot_kelvins
    fresh_hot = spice.Library(hot_kelvins)
    fresh_hot.read(source, "parts.lib")

    # Temperature-independent subcircuits stay cached (exact object identity).
    assert loaded.component("fixed_load") is fixed_at_27
    assert loaded.component("parent_of_overridden_default") is overridden_parent_at_27

    # Every temperature-dependent subcircuit re-evaluates bit-identically to fresh.
    for subcircuit_name in (
        "fixed_load",
        "direct_warm",
        "global_warm",
        "default_warm",
        "parent_of_default_warm",
        "parent_of_overridden_default",
        "branch_warm",
        "card_warm",
        "ic_warm",
        "vres_tc",
        "binned_warm",
    ):
        assert circuit(loaded.component(subcircuit_name)) == circuit(
            fresh_hot.component(subcircuit_name)
        )

    assert _devices(loaded.component("branch_warm"))["R1"][1]["resistance"] == 2000.0
    assert _devices(loaded.component("parent_of_default_warm").children[0])["R1"][1][
        "resistance"
    ] == pytest.approx(1085.0, rel=1e-12)
    assert _devices(loaded.component("binned_warm"))["M1"][1]["vth0"] == 0.4
    assert loaded.initial_conditions_of("ic_warm")["a"] == pytest.approx(
        0.85, rel=1e-12
    )
    assert circuit(loaded.component("vres_tc")) != vres_27

    # Returning to 27 C reuses the cached 27 C temperature-dependent entries too.
    loaded.temperature = celsius(27).value
    assert _devices(loaded.component("branch_warm"))["R1"][1]["resistance"] == 1000.0
    assert circuit(loaded.component("vres_tc")) == vres_27


def test_global_plan_retargets_without_rebuild_when_no_param_uses_temper() -> None:
    """Verifies switching library temperature keeps temper-free builds cached."""
    loaded = library(
        ".param r_base = 2k\n"
        ".subckt load a\n"
        "R1 a 0 {r_base}\n"
        ".ends\n"
        ".subckt sensor a\n"
        "R1 a 0 {1k + temper}\n"
        ".ends\n"
    )
    fixed_before = loaded.component("load")
    # Exact arithmetic `1000 + 27`; rel=1e-12 covers Kelvin/Celsius offset rounding.
    assert _devices(loaded.component("sensor"))["R1"][1]["resistance"] == pytest.approx(
        1027.0, rel=1e-12
    )

    loaded.temperature = celsius(100).value
    assert loaded.values()["r_base"] == 2000.0
    # Celsius conversion (`celsius(100) - 273.15`) introduces <= 1e-12 float rounding.
    assert loaded.values()["temper"] == pytest.approx(100.0, rel=1e-12)
    assert loaded.component("load") is fixed_before
    assert _devices(loaded.component("sensor"))["R1"][1]["resistance"] == pytest.approx(
        1100.0, rel=1e-12
    )


def test_dual_arithmetic_and_zero_pruning() -> None:
    """Verifies `Dual` arithmetic rules, zero-slope pruning, and constant guards.

    Exact algebraic operations carry only IEEE-754 floating-point rounding (`rel=1e-14`,
    ~45 eps).
    """
    first = Dual.seed(3.0, "x")
    second = Dual.seed(4.0, "y")

    assert (+first) == first
    assert (-first).value == -3.0
    assert (-first).derivative("x") == -1.0

    added = first + second + 2.0
    assert added.value == 9.0
    assert (added.derivative("x"), added.derivative("y")) == (1.0, 1.0)
    assert (2.0 + first).value == 5.0
    assert (2.0 + first).derivative("x") == 1.0

    subtracted = first - second - 1.0
    assert subtracted.value == -2.0
    assert (subtracted.derivative("x"), subtracted.derivative("y")) == (1.0, -1.0)
    assert (5.0 - first).value == 2.0
    assert (5.0 - first).derivative("x") == -1.0

    # Canceling terms drop exact zero derivatives rather than retaining empty entries.
    assert (first - first).derivatives == {}
    assert (0.0 * first).derivatives == {}
    assert (first * 0.0).derivatives == {}

    product = first * second
    assert product.value == 12.0
    assert (product.derivative("x"), product.derivative("y")) == (4.0, 3.0)
    assert (2.5 * first).derivative("x") == 2.5
    assert (first * 2.5).derivative("x") == 2.5

    quotient = first / second
    assert quotient.value == pytest.approx(0.75, rel=1e-14)
    assert quotient.derivative("x") == pytest.approx(0.25, rel=1e-14)
    assert quotient.derivative("y") == pytest.approx(-3.0 / 16.0, rel=1e-14)
    assert (6.0 / first).value == 2.0
    assert (6.0 / first).derivative("x") == pytest.approx(-6.0 / 9.0, rel=1e-14)
    assert (first / 2.0).derivative("x") == 0.5

    # Unseeded constants never multiply zero derivatives by infinities or NaNs.
    zero_constant = Dual[str].constant(0.0)
    infinite_constant = Dual[str].constant(math.inf)
    assert (zero_constant / zero_constant).derivatives == {}
    assert (Dual.constant(1.0) / zero_constant).derivatives == {}
    assert (zero_constant * infinite_constant).derivatives == {}


def test_evaluate_dual_operators_and_all_builtin_functions() -> None:
    """Verifies `evaluate_dual` across every SPICE operator, built-in, and `.func`.

    Primals agree with `evaluate` and partials agree with closed-form calculus rules
    within IEEE-754 floating-point rounding (`rel=1e-14`, ~45 eps).
    """
    x_value = 1.2
    y_value = 0.7
    unit_value = 0.4
    hyperbolic_value = 2.0
    seeded: dict[str, float | Dual[str]] = {
        "x": Dual.seed(x_value, "x"),
        "y": Dual.seed(y_value, "y"),
        "u": Dual.seed(unit_value, "u"),
        "v": Dual.seed(hyperbolic_value, "v"),
    }
    plain = {"x": x_value, "y": y_value, "u": unit_value, "v": hyperbolic_value}

    cases: list[tuple[str, dict[str, float]]] = [
        ("x + 2 * y - 3", {"x": 1.0, "y": 2.0}),
        (
            "x * y / (x + y)",
            {
                "x": y_value**2 / (x_value + y_value) ** 2,
                "y": x_value**2 / (x_value + y_value) ** 2,
            },
        ),
        ("-x + (+y)", {"x": -1.0, "y": 1.0}),
        (
            "x ^ y",
            {
                "x": y_value * x_value ** (y_value - 1.0),
                "y": (x_value**y_value) * math.log(x_value),
            },
        ),
        ("(-x) ^ 3", {"x": -3.0 * x_value**2}),
        ("(-x) ^ 2", {"x": 2.0 * x_value}),
        ("x ^ 0", {}),
        (
            "pow(x, y)",
            {
                "x": y_value * x_value ** (y_value - 1.0),
                "y": (x_value**y_value) * math.log(x_value),
            },
        ),
        (
            "pwr(-x, y)",
            {
                "x": y_value * x_value ** (y_value - 1.0),
                "y": (x_value**y_value) * math.log(x_value),
            },
        ),
        (
            "pwrs(-x, y)",
            {
                "x": -y_value * x_value ** (y_value - 1.0),
                "y": -(x_value**y_value) * math.log(x_value),
            },
        ),
        ("abs(x)", {"x": 1.0}),
        ("abs(-x)", {"x": 1.0}),
        ("sqrt(x)", {"x": 0.5 / math.sqrt(x_value)}),
        ("exp(x)", {"x": math.exp(x_value)}),
        ("ln(x)", {"x": 1.0 / x_value}),
        ("log(x)", {"x": 1.0 / x_value}),
        ("log10(x)", {"x": 1.0 / (x_value * math.log(10.0))}),
        ("sin(x)", {"x": math.cos(x_value)}),
        ("cos(x)", {"x": -math.sin(x_value)}),
        ("tan(x)", {"x": 1.0 + math.tan(x_value) ** 2}),
        ("asin(u)", {"u": 1.0 / math.sqrt(1.0 - unit_value**2)}),
        ("acos(u)", {"u": -1.0 / math.sqrt(1.0 - unit_value**2)}),
        ("atan(x)", {"x": 1.0 / (1.0 + x_value**2)}),
        ("sinh(x)", {"x": math.cosh(x_value)}),
        ("cosh(x)", {"x": math.sinh(x_value)}),
        ("tanh(x)", {"x": 1.0 - math.tanh(x_value) ** 2}),
        ("asinh(x)", {"x": 1.0 / math.hypot(x_value, 1.0)}),
        ("acosh(v)", {"v": 1.0 / math.sqrt(hyperbolic_value**2 - 1.0)}),
        ("atanh(u)", {"u": 1.0 / (1.0 - unit_value**2)}),
        ("floor(x)", {}),
        ("ceil(x)", {}),
        ("int(-x)", {}),
        ("sgn(x)", {}),
        ("sign(-x)", {}),
        ("u(x)", {}),
        ("stp(-x)", {}),
        ("uramp(x)", {"x": 1.0}),
        ("uramp(-x)", {}),
        ("min(x, y)", {"y": 1.0}),
        ("max(x, y)", {"x": 1.0}),
        ("limit(x, y, v)", {"x": 1.0}),
        ("limit(u, y, v)", {"y": 1.0}),
        ("limit(v + 1, y, v)", {"v": 1.0}),
        ("limit(u, v, y)", {"y": 1.0}),
        ("if(x > y, x * x, 3 * y)", {"x": 2.0 * x_value}),
        ("if(x < y, x * x, 3 * y)", {"y": 3.0}),
        ("x > y ? x * y : 5 * x", {"x": y_value, "y": x_value}),
        ("(x > y) && (y > 0) || !(x == y)", {}),
        ("(x <= y) + (x >= y) + (x != y)", {}),
        ("table(x, 0, 1, 2, 5, 4, 3)", {"x": 2.0}),
        ("pwl(v + 3, 0, 1, 2, 5, 4, 3)", {"v": -1.0}),
        (
            "table(u - 1, 0, y, 2, 5)",
            {"u": (5.0 - y_value) / 2.0, "y": 1.0 - (unit_value - 1.0) / 2.0},
        ),
        ("rounded_table(x, 0.25, 0, 0, 2, 4, 4, 2)", {"x": 2.0}),
        ("rounded_table(v, 0.25, 0, 0, 2, 4, 4, 2)", {"v": 0.5}),
        ("rounded_table(v - 0.25, 0.25, 0, 0, 2, 4, 4, 2)", {"v": 1.25}),
    ]
    for source, expected_slopes in cases:
        tree = parse(source)
        dual = evaluate_dual(tree, seeded)
        reference = evaluate(tree, plain)
        assert dual.value == pytest.approx(reference, rel=1e-14, abs=1e-15), source
        for key in ("x", "y", "u", "v"):
            assert dual.derivative(key) == pytest.approx(
                expected_slopes.get(key, 0.0), rel=1e-14, abs=1e-15
            ), (source, key)

    # Smooth zero-base powers and `abs(0)` produce clean finite zero derivatives.
    at_two = {"x": Dual.seed(2.0, "x")}
    assert evaluate_dual(parse("(x - 2) ^ 2"), at_two) == Dual(0.0)
    assert evaluate_dual(parse("abs(x - 2)"), at_two) == Dual(0.0)
    assert evaluate_dual(parse("0 ^ 2"), at_two) == Dual(0.0)
    assert evaluate_dual(parse("sqrt(0)"), at_two) == Dual(0.0)
    assert not is_finite(evaluate_dual(parse("sqrt(x - 2)"), at_two))

    # User-defined `.func` calls propagate exact derivatives iteratively and via inline.
    user_functions = {
        "inner": Function(("a", "b"), parse("a * b + sin(a)")),
        "outer": Function(("z",), parse("inner(z * z, z + 1)")),
    }
    direct = evaluate_dual(parse("outer(x)"), seeded, functions=user_functions)
    expanded = evaluate_dual(inlined(parse("outer(x)"), user_functions), seeded)
    expected_outer_slope = (
        3.0 * x_value**2 + 2.0 * x_value + 2.0 * x_value * math.cos(x_value**2)
    )
    assert direct.value == pytest.approx(expanded.value, rel=1e-14)
    assert direct.derivative("x") == pytest.approx(expected_outer_slope, rel=1e-14)
    assert expanded.derivative("x") == pytest.approx(expected_outer_slope, rel=1e-14)


def test_library_derivatives_geometric_passives_and_diode() -> None:
    """Verifies exact Jacobian propagation through geometric passives and level-3 diode.

    Every closed-form slope agrees with `Library.derivatives` within IEEE-754
    floating-point rounding (`rel=1e-14`, ~45 eps).
    """
    loaded = library(
        ".option scale=1e-6\n"
        ".model rpoly r rsh=100 narrow=0.1u short=0.2u kf=1e-20 af=2 lf=1.5 wf=0.5\n"
        ".model cmim c cj=2e-3 cjsw=1e-10 narrow=0.1u short=0.05u\n"
        ".model cox c thick=10n di=4.0\n"
        ".model d3 d level=3 js=1e-4 jsw=1e-10 xw=0.1\n"
        ".subckt passive_block a b w=2 l=5\n"
        "R1 a b rpoly w={w} l={l}\n"
        "C1 a b cmim w={w} l={l}\n"
        "C2 a b cox w={w} l={l}\n"
        "D1 a b d3 w={w} l={l}\n"
        ".ends\n"
    )
    scale = 1e-6
    width_microns = 2.0
    length_microns = 5.0
    width = width_microns * scale
    length = length_microns * scale
    copies = 2.0
    default_evaluated = loaded.derivatives("passive_block", "U1")
    assert set(default_evaluated.arguments) == {"w", "l"}
    override_only = loaded.derivatives("passive_block", "U1", m=copies)
    assert list(override_only.arguments) == ["m"]
    evaluated = loaded.derivatives(
        "passive_block", "U1", with_respect_to=("w", "l", "m"), m=copies
    )

    # Geometric resistor: R = ((L - 2*short) / (W - 2*narrow)) * rsh / m.
    resistor_length = length - 4e-7
    resistor_width = width - 2e-7
    resistance_single = resistor_length / resistor_width * 100.0
    resistor_slopes = evaluated.derivatives["U1/R1"]
    assert resistor_slopes["resistance"]["l"] == pytest.approx(
        scale * 100.0 / (resistor_width * copies), rel=1e-14
    )
    assert resistor_slopes["resistance"]["w"] == pytest.approx(
        -scale * resistance_single / (resistor_width * copies), rel=1e-14
    )
    assert resistor_slopes["resistance"]["m"] == pytest.approx(
        -resistance_single / (copies**2), rel=1e-14
    )
    # Flicker noise coefficient: kf / (long^lf * wide^wf) * m^(1 - af).
    flicker_area = (resistor_length**1.5) * (resistor_width**0.5)
    flicker_value = 1e-20 / flicker_area * (copies ** (1.0 - 2.0))
    assert resistor_slopes["flicker_coefficient"]["l"] == pytest.approx(
        scale * (-1.5 * flicker_value / resistor_length), rel=1e-14
    )
    assert resistor_slopes["flicker_coefficient"]["w"] == pytest.approx(
        scale * (-0.5 * flicker_value / resistor_width), rel=1e-14
    )
    assert resistor_slopes["flicker_coefficient"]["m"] == pytest.approx(
        -flicker_value / copies, rel=1e-14
    )

    # Geometric capacitor: C = m * (cj * wide * long + 2 * cjsw * (wide + long)).
    cap_width = width - 1e-7
    cap_length = length - 5e-8
    cap_single = 2e-3 * cap_width * cap_length + 2e-10 * (cap_width + cap_length)
    cap_slopes = evaluated.derivatives["U1/C1"]["capacitance"]
    assert cap_slopes["w"] == pytest.approx(
        scale * copies * (2e-3 * cap_length + 2e-10), rel=1e-14
    )
    assert cap_slopes["l"] == pytest.approx(
        scale * copies * (2e-3 * cap_width + 2e-10), rel=1e-14
    )
    assert cap_slopes["m"] == pytest.approx(cap_single, rel=1e-14)

    # Thickness dielectric capacitor: cj = di * eps0 / thick.
    oxide_cj = 4.0 * models.VACUUM_PERMITTIVITY / 10e-9
    assert evaluated.derivatives["U1/C2"]["capacitance"]["w"] == pytest.approx(
        scale * copies * oxide_cj * length, rel=1e-14
    )
    assert evaluated.derivatives["U1/C2"]["capacitance"]["l"] == pytest.approx(
        scale * copies * oxide_cj * width, rel=1e-14
    )

    # Level-3 diode under `.option scale=1e-6`: width_eff = w + xw, scale = 1e-6.
    diode_width = width_microns + 0.1
    diode_length = length_microns + 0.1
    diode_slopes = evaluated.derivatives["U1/D1"]
    assert diode_slopes["saturation_current"]["w"] == pytest.approx(
        copies * 1e-4 * diode_length * (scale**2), rel=1e-14
    )
    assert diode_slopes["saturation_current"]["l"] == pytest.approx(
        copies * 1e-4 * diode_width * (scale**2), rel=1e-14
    )
    assert diode_slopes["sidewall_saturation_current"]["w"] == pytest.approx(
        copies * 1e-10 * 2.0 * scale, rel=1e-14
    )


def test_library_derivatives_coupled_sources_line_vbic_and_nested() -> None:
    """Verifies exact Jacobian across coupled inductors, sources, lines, VBIC, and `X`.

    Every closed-form slope and push/pull transpose agrees within IEEE-754 rounding
    (`rel=1e-14`, ~45 eps).
    """
    loaded = library(
        ".model qvbic npn level=4 is=1e-16 rth={rth_val} cth={cth_val}\n"
        ".subckt child a b rval=2k\n"
        "Rchild a b {rval}\n"
        ".ends\n"
        ".subckt mixed_block a b c e s dt w=2 l1=1u l2=2u k=0.5 "
        "off=0.5 amp=1.0 freq=1meg deg=30 z0=50 nl=0.25 rth_val=100 cth_val=1n\n"
        "L1 a 0 {l1}\n"
        "L2 b 0 {l2}\n"
        "K1 L1 L2 {k}\n"
        "V1 a 0 SIN({off} {amp} {freq} 0 0 {deg})\n"
        "T1 a 0 b 0 Z0={z0} F={freq} NL={nl}\n"
        "Q1 c b e s dt qvbic\n"
        "X1 a b child rval={w * 1k}\n"
        ".ends\n"
    )
    copies = 2.0
    evaluated = loaded.derivatives(
        "mixed_block",
        "U1",
        with_respect_to=(
            "w",
            "l1",
            "l2",
            "k",
            "off",
            "amp",
            "freq",
            "deg",
            "z0",
            "nl",
            "rth_val",
            "cth_val",
            "m",
        ),
        m=copies,
    )

    # Coupled inductors remap L1/L2/K1 onto K1 windings and couplings.
    coupled_slopes = evaluated.derivatives["U1/K1"]
    assert coupled_slopes["windings.0.inductance"]["l1"] == pytest.approx(
        1.0 / copies, rel=1e-14
    )
    assert coupled_slopes["windings.1.inductance"]["l2"] == pytest.approx(
        1.0 / copies, rel=1e-14
    )
    assert coupled_slopes["couplings.0.coefficient"]["k"] == pytest.approx(
        1.0, rel=1e-14
    )

    # Independent sine source converts degrees to radians.
    source_slopes = evaluated.derivatives["U1/V1"]
    assert source_slopes["waveform.sine.offset"]["off"] == pytest.approx(1.0, rel=1e-14)
    assert source_slopes["waveform.sine.amplitude"]["amp"] == pytest.approx(
        1.0, rel=1e-14
    )
    assert source_slopes["waveform.sine.frequency"]["freq"] == pytest.approx(
        1.0, rel=1e-14
    )
    assert source_slopes["waveform.sine.phase"]["deg"] == pytest.approx(
        math.pi / 180.0, rel=1e-14
    )

    # Transmission line: delay = nl / freq.
    line_slopes = evaluated.derivatives["U1/T1"]
    assert line_slopes["characteristic_impedance"]["z0"] == pytest.approx(
        1.0, rel=1e-14
    )
    assert line_slopes["delay"]["nl"] == pytest.approx(1.0 / 1e6, rel=1e-14)
    assert line_slopes["delay"]["freq"] == pytest.approx(-0.25 / 1e12, rel=1e-14)

    # VBIC self-heating network: rth / m and cth * m.
    assert evaluated.derivatives["U1/Q1.rth"]["resistance"]["rth_val"] == pytest.approx(
        1.0 / copies, rel=1e-14
    )
    assert evaluated.derivatives["U1/Q1.rth"]["resistance"]["m"] == pytest.approx(
        -100.0 / (copies**2), rel=1e-14
    )
    assert evaluated.derivatives["U1/Q1.cth"]["capacitance"][
        "cth_val"
    ] == pytest.approx(copies, rel=1e-14)
    assert evaluated.derivatives["U1/Q1.cth"]["capacitance"]["m"] == pytest.approx(
        1e-9, rel=1e-14
    )

    # Nested X subcircuit propagates parent parameter and multiplier `m`.
    assert evaluated.derivatives["U1/X1/Rchild"]["resistance"]["w"] == pytest.approx(
        1000.0 / copies, rel=1e-14
    )
    assert evaluated.derivatives["U1/X1/Rchild"]["resistance"]["m"] == pytest.approx(
        -2000.0 / (copies**2), rel=1e-14
    )

    # `push` and `pull` are exact transposes on arbitrary tangents and cotangents.
    tangents = {"w": 0.13, "l1": 1e-7, "m": 0.25, "deg": 5.0}
    pushed = evaluated.push(tangents)
    cotangents = {
        device_path: {
            parameter: float(index + 1)
            for index, parameter in enumerate(device_parameters)
        }
        for device_path, device_parameters in evaluated.derivatives.items()
    }
    pulled = evaluated.pull(cotangents)
    forward = sum(
        cotangents[device_path][parameter] * delta
        for device_path, device_deltas in pushed.items()
        for parameter, delta in device_deltas.items()
    )
    backward = sum(pulled[name] * delta for name, delta in tangents.items())
    assert forward == pytest.approx(backward, rel=1e-14)


def test_library_derivatives_binned_mosfet_and_bin_bounds() -> None:
    """Verifies BSIM4 `BinBounds`, `contains`, `placement`, and `with_respect_to`.

    Exact scale-factor slopes agree within IEEE-754 rounding (`rel=1e-14`).
    """
    source = (
        ".option scale=1e-6\n"
        ".model nmos.0 nmos level=54 version=4.5 "
        "lmin=0.1u lmax=0.2u wmin=0.5u wmax=1.0u vth0=0.4\n"
        ".model nmos.1 nmos level=54 version=4.5 "
        "lmin=0.2u lmax=0.5u wmin=0.5u wmax=1.0u vth0=0.5\n"
        ".subckt fet d g s b w=0.8 l=0.15 nf=2\n"
        "M1 d g s b nmos w={w} l={l} nf={nf} "
        "ad={w * 0.27} as={w * 0.27} pd={2 * (w + 0.27)} ps={2 * (w + 0.27)}\n"
        ".ends\n"
    )
    loaded = library(source)
    placement = loaded.placement("fet", w=0.8, l=0.15, nf=2.0)
    assert isinstance(placement, spice.SubcircuitPlacement)
    assert circuit(placement.component()) == circuit(
        loaded.component("fet", w=0.8, l=0.15, nf=2.0)
    )

    evaluated = placement.derivatives("M1", with_respect_to=("W", "L", "nf"))
    assert isinstance(evaluated, spice.SubcircuitDerivatives)
    assert list(evaluated.arguments) == ["W", "L", "nf"]
    assert evaluated.bins == {
        "M1/M1": spice.BinBounds(
            model="nmos.0",
            min_length=1e-7,
            max_length=2e-7,
            min_width=5e-7,
            max_width=1e-6,
            finger_divisor=1.0,
        )
    }
    box = evaluated.bins["M1/M1"]
    assert box.contains(0.15 * um, 0.8 * um)
    assert box.contains(1.5e-7, 8e-7)
    assert not box.contains(0.3 * um, 0.8 * um)
    assert not box.contains(0.15 * um, 1.2 * um)

    slopes = evaluated.derivatives["M1/M1"]
    assert slopes["width"]["W"] == pytest.approx(1e-6, rel=1e-14)
    assert slopes["length"]["L"] == pytest.approx(1e-6, rel=1e-14)
    assert slopes["fingers"]["nf"] == pytest.approx(1.0, rel=1e-14)
    assert slopes["drain_area"]["W"] == pytest.approx(0.27e-12, rel=1e-14)
    assert slopes["source_area"]["W"] == pytest.approx(0.27e-12, rel=1e-14)
    assert slopes["drain_perimeter"]["W"] == pytest.approx(2e-6, rel=1e-14)
    assert slopes["source_perimeter"]["W"] == pytest.approx(2e-6, rel=1e-14)

    # Stepping length into bin 1 updates active `BinBounds` and model parameters.
    stepped = placement.derivatives("M1", with_respect_to=(), l=0.3)
    assert stepped.arguments == {}
    assert stepped.derivatives == {}
    assert stepped.parameters["M1/M1"]["vth0"] == 0.5
    assert stepped.bins["M1/M1"].model == "nmos.1"
    assert stepped.bins["M1/M1"].contains(0.3 * um, 0.8 * um)

    # HSPICE compatibility divides width by `nf` for bin selection (`finger_divisor=2`).
    hspice_library = spice.Library(compatibility="hsa")
    hspice_library.read(source, "parts.lib")
    hspice_eval = hspice_library.derivatives("fet", "M1", w=1.6, l=0.15, nf=2.0)
    hspice_box = hspice_eval.bins["M1/M1"]
    assert hspice_box.finger_divisor == 2.0
    assert hspice_box.contains(0.15 * um, 1.6 * um)
    assert not hspice_box.contains(0.15 * um, 0.8 * um)


def test_library_derivatives_errors_and_thread_safety() -> None:
    """Verifies error handling and thread safety for `Library.derivatives`."""
    loaded = library(".subckt cell a b w=1k\nR1 a b {w * 2}\n.ends\n")
    with pytest.raises(SpiceError, match="no subcircuit is named `missing`"):
        loaded.derivatives("missing")
    with pytest.raises(SpiceError, match="has no parameter `bogus`"):
        loaded.derivatives("cell", bogus=1.0)
    with pytest.raises(SpiceError, match="has no parameter `bogus`"):
        loaded.derivatives("cell", with_respect_to=("bogus",))
    with pytest.raises(SpiceError, match="duplicate parameter"):
        loaded.derivatives("cell", w=1.0, W=2.0)
    with pytest.raises(SpiceError, match="duplicate parameter"):
        loaded.derivatives("cell", with_respect_to=("w", "W"))
    with pytest.raises(SpiceError, match="must be finite"):
        loaded.derivatives("cell", w=math.nan)
    with pytest.raises(SpiceError, match="must be finite"):
        loaded.derivatives("cell", w=math.inf)

    evaluated = loaded.derivatives("cell", "U1", with_respect_to=("w",))
    with pytest.raises(ValueError, match="unknown subcircuit argument `bogus`"):
        evaluated.push({"bogus": 1.0})

    # Concurrent calls across threads return bit-identical parameters and Jacobians.
    def worker(index: int) -> tuple[float, float]:
        result = loaded.derivatives("cell", f"U{index}", w=100.0 + index)
        return (
            result.parameters[f"U{index}/R1"]["resistance"],
            result.derivatives[f"U{index}/R1"]["resistance"]["w"],
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(worker, range(32)))
    assert outcomes == [(2.0 * (100.0 + index), 2.0) for index in range(32)]
