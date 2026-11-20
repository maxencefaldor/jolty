# Copyright 2026 Maxence Faldor
"""Hypothesis fuzz tests for the SPICE netlist, model card, and library reader."""

import contextlib
from pathlib import Path
from typing import Final

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from fuzz.helpers import FAST_FUZZ, FUZZ
from jolty import Builder, Role, Simulation, _core, circuit, component, devices
from jolty.errors import ElaborationError, SpiceError
from jolty.spice import Library, Sample, export, models, netlist
from jolty.units import K, V

_SPICE_NODES: Final = ("0", "a", "b", "c", "d", "n1", "n2", "t", "ground", "a/b")
_SPICE_NUMBERS: Final = (
    "0",
    "1",
    "-1",
    "1k",
    "1u",
    "10u",
    "1n",
    "1m",
    "0.5",
    "1.5",
    "1e-14",
    "1e500",
    "1e308",
    "{1/0}",
    "{0/0}",
    "{p}",
    "{missing}",
)
_SPICE_CARD_TEMPLATES: Final = (
    ".subckt s1 a b",
    ".subckt s1 a b params: p=1",
    ".subckt s1 a b c d",
    ".subckt s2 a b",
    ".subckt",
    ".subckt s1 0 a",
    ".subckt s1 ground a",
    ".subckt s1 a/b c",
    ".ends",
    ".ends s1",
    ".end",
    ".param p=1k",
    ".param p = 1 + 2 q = 3",
    ".param a=b b=a",
    ".param = 1",
    ".param p=",
    ".param p=agauss(1k, 100, 3)",
    ".param p=agauss(1k, 1e308, 1e-10)",
    ".params x=1",
    ".func double(x) {2 * x}",
    ".func bad(x) {bad(x)}",
    ".func",
    ".option scale=1u",
    ".option scale=1e308",
    ".option wnflag=1",
    ".option",
    ".options a=1 flag",
    ".ic V(a)=1",
    ".ic V(a)=",
    ".ic",
    ".nodeset V(a)=0",
    ".if (1)",
    ".if (0)",
    ".if (p > 0)",
    ".if ()",
    ".if",
    ".elseif (1)",
    ".else",
    ".endif",
    ".model mr r rsh=10 narrow=5u short=5u kf=1e-12 af=1.5",
    ".model mc c cap=1n thick=0",
    ".model md d is=1e-14 level=3",
    ".model md d is=1e-14 level=1e500",
    ".model mq npn bf=100",
    ".model mvbic npn level=4 rth=10 kfn=1e-12 afn=1.5",
    ".model mn nmos level=1",
    ".model mb3.0 nmos level=8 lmin=0 lmax=1 wmin=0 wmax=1",
    ".model mb4.0 nmos level=54 version=4.5 lmin=0 lmax=1 wmin=0 wmax=1",
    ".model mvd vdmos pchan mfg=Vendor",
    ".model msw vswitch ron=1 roff=1meg von=2 voff=1",
    ".model mcsw iswitch ron=1 roff=1meg ion=1m ioff=0",
    ".model derived ako:md rs=10",
    ".model derived ako:missing",
    ".model",
    ".model empty",
    "R1 a b 1k",
    "R1 a b mr w=10u l=10u",
    "R1 a b mr w=0 l=10u",
    "R1 a b 1k m=0",
    "R1 a b 1k m=-1",
    "R1 a b 1k tc1=",
    "R1 a b {v(a,b)} tc1=-1",
    "R1 a b",
    "R1",
    "C1 a b 1n",
    "C1 a b mc w=1u l=1u",
    "C1 a b Q={v(a,b)*1n}",
    "C1 a b C={v(a,b)^500}",
    "C1 a b 1n m=0",
    "C1 a b 1n tc1=",
    "C1",
    "L1 a b 1u",
    "L2 a b 1u",
    "L1 a b 1u m=0",
    "L1",
    "K1 L1 L2 0.5",
    "K1 L1 L2 1.5",
    "K1 L1 L1 0.5",
    "K1 L1 L2 1e500",
    "K1",
    "D1 a b md",
    "D1 a b md area=0",
    "D1 a b md m=0",
    "D1 a b md w=0 l=0",
    "D1 a b",
    "Q1 a b c mq",
    "Q1 a b c 0 t mvbic",
    "Q1 a b c mvbic area=0",
    "Q1 a b c mvbic m=0",
    "Q1 a b c mvbic m=-1",
    "Q1",
    "J1 a b c md",
    "M1 a b c d mn w=1u l=1u",
    "M1 a b c d mb3 w=1u l=1u nf=0",
    "M1 a b c d mb4 w=1u l=1u nf=0",
    "M1 a b c mvd",
    "M1",
    "V1 a 0 5",
    "V1 a 0 PULSE(0 5 0 1n 1n 5n 10n)",
    "V1 a 0 PULSE(0 1 0 0 0 0 0)",
    "V1 a 0 PULSE()",
    "V1 a 0 SIN(0 1 1k)",
    "V1 a 0 SIN(0 1 1e500)",
    "V1 a 0 PWL(1 0 0 1)",
    "V1 a 0 PWL(1 0 0 1) DC 0",
    "V1 a 0 EXP(0 1)",
    "V1 a 0 1e500",
    "V1",
    "I1 a 0 1m",
    "I1 a",
    "I1",
    "E1 a b c d 2",
    "E1 a b VALUE {V(c,d)}",
    "E1 a b VALUE",
    "E1 a b VALUE =",
    "E1 a b TABLE {V(c,d)} = (0,0) (1,1)",
    "E1 a b TABLE {V(c,d)} = (0,0) (0,1)",
    "E1 a b POLY(1) c d 0 1",
    "E1 a b POLY(0) 1 2",
    "E1 a b POLY(9999999) c d 1",
    "E1",
    "G1 a b c d 1m",
    "G1 a b TABLE {V(c,d)} = (0,0)",
    "F1 a b V1 2",
    "F1 a b POLY(0) 1 2",
    "H1 a b V1 2",
    "B1 a b V={V(a)}",
    "B1 a b I={I(V1)}",
    "B1 a b V=",
    "B1",
    "S1 a b c d msw",
    "S1 a b c d md",
    "S1",
    "W1 a b V1 mcsw",
    "W1 a b V1 md",
    "W1",
    "T1 a b c d Z0=50 TD=1n",
    "T1 a b c d Z0=50 F=0",
    "T1 a b c d Z0=50 TD=1e500",
    "T1",
    "X1 a b s1",
    "X1 a b s2",
    "X1 a b s1 m=0",
    "X1 a b s1 p=2",
    "X1",
    "+ extra=1",
    "* comment line",
    "; inline comment",
    ",",
    "=",
    "(",
    "'unterminated",
)


def _random_spice_card(draw: st.DrawFn, index: int) -> str:
    """Synthesizes a randomized SPICE card from edge node names and numbers."""
    node_one = draw(st.sampled_from(_SPICE_NODES))
    node_two = draw(st.sampled_from(_SPICE_NODES))
    value = draw(st.sampled_from(_SPICE_NUMBERS))
    multiplier = draw(st.sampled_from(_SPICE_NUMBERS))
    kind = draw(st.sampled_from(["R", "C", "L", "V", "I", "D", "T", "param"]))
    if kind in {"R", "C", "L"}:
        return f"{kind}fuzz{index} {node_one} {node_two} {value} m={multiplier}"
    if kind in {"V", "I"}:
        return f"{kind}fuzz{index} {node_one} {node_two} {value}"
    if kind == "D":
        return f"Dfuzz{index} {node_one} {node_two} md area={value} m={multiplier}"
    if kind == "T":
        node_three = draw(st.sampled_from(_SPICE_NODES))
        node_four = draw(st.sampled_from(_SPICE_NODES))
        return (
            f"Tfuzz{index} {node_one} {node_two} {node_three} {node_four} "
            f"Z0={value} F={multiplier}"
        )
    return f".param p{index}={value}"


@st.composite
def _spice_netlists(draw: st.DrawFn) -> str:
    """Generates grammar-guided adversarial SPICE netlists with cards and directives."""
    lines = draw(
        st.lists(st.sampled_from(_SPICE_CARD_TEMPLATES), min_size=1, max_size=12)
    )
    extra_count = draw(st.integers(min_value=0, max_value=3))
    lines.extend(_random_spice_card(draw, index) for index in range(extra_count))
    if draw(st.booleans()):
        # Ensure at least one closed subcircuit wraps body cards so element readers run.
        return "\n".join([".subckt s1 a b c d", *lines, ".ends"])
    return "\n".join(lines)


@st.composite
def _valid_spice_subcircuits(draw: st.DrawFn) -> str:
    """Generates valid SPICE subcircuits that elaborate and export cleanly."""
    scale = draw(
        st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    resistance_one = draw(
        st.floats(min_value=100.0, max_value=1e5, allow_nan=False, allow_infinity=False)
    )
    resistance_two = draw(
        st.floats(min_value=100.0, max_value=1e5, allow_nan=False, allow_infinity=False)
    )
    capacitance = draw(
        st.floats(
            min_value=1e-12, max_value=1e-6, allow_nan=False, allow_infinity=False
        )
    )
    inductance = draw(
        st.floats(min_value=1e-9, max_value=1e-3, allow_nan=False, allow_infinity=False)
    )
    lines = [
        f".param factor = {scale:.4g}",
        ".func scaled(x) {x * factor}",
        ".model d_fast d is=1e-14 rs=10 cjo=1p",
        ".subckt stage inp out",
        f"R1 inp mid {{scaled({resistance_one:.4g})}}",
        f"R2 mid out {resistance_two:.4g}",
        f"R3 out 0 {resistance_two:.4g}",
        f"C1 mid 0 {capacitance:.4g}",
        f"L1 mid lnode {inductance:.4g}",
        f"R4 lnode 0 {resistance_one:.4g}",
    ]
    if draw(st.booleans()):
        lines.append("D1 out 0 d_fast")
    if draw(st.booleans()):
        lines.append("G1 out 0 inp 0 1e-4")
    if draw(st.booleans()):
        lines.extend(
            [".if (factor > 5)", "R5 inp 0 100k", ".else", "R5 inp 0 200k", ".endif"]
        )
    lines.append(".ends")
    return "\n".join(lines)


def _exercise_spice_library(
    scratch_directory: Path,
    text: str,
    *,
    filename: str,
    temperature: float = _core.REFERENCE_TEMPERATURE,
    exercise_drawn: bool = False,
) -> None:
    library = Library(temperature=temperature)
    try:
        library.read(text, "fuzz.lib")
    except SpiceError:
        return
    with contextlib.suppress(SpiceError):
        library.values()
    sample = Sample(seed=1, index=0)
    for subcircuit in library.subcircuits:
        assert isinstance(library.survey(subcircuit), tuple)
        with contextlib.suppress(SpiceError):
            library.component(subcircuit)
            if exercise_drawn:
                library.drawn(subcircuit, sample, "X1")
                library.random_variables(subcircuit, "X1")
    for model_name in library.models:
        with contextlib.suppress(SpiceError):
            library.survey_model(model_name)
    path = scratch_directory / filename
    path.write_text(text, encoding="utf-8")
    included = Library(temperature=temperature)
    with contextlib.suppress(SpiceError):
        included.include(path)


@FUZZ
@example("")
@example("+ orphaned continuation")
@example(",")
@example("* comment\n; trailing\nR1 a b 1k ; comment\n+ tc1=1m")
@example("(((unclosed")
@example("'unclosed quote")
@given(st.text(max_size=250))
def test_netlist_statements_and_tokens_arbitrary_text(text: str) -> None:
    direct = netlist.tokens(text)
    assert all(token != "" for token in direct)
    try:
        found = netlist.statements(text, "fuzz.lib")
    except SpiceError:
        return
    for statement in found:
        words = netlist.tokens(statement.text)
        assert all(word != "" for word in words)


@FUZZ
@example(".model")
@example(".model m1")
@example(".model m1 vdmos pchan mfg=Vendor")
@example(".model m1 ako:missing")
@example(".model m1 nmos (version=)")
@given(st.text(max_size=150))
def test_models_read_arbitrary_text_only_raises_spice_error(text: str) -> None:
    statement = netlist.Statement("fuzz.lib", 1, text)
    try:
        card = models.read(statement, {})
    except SpiceError:
        return
    assert isinstance(card, models.ModelCard)


@FUZZ
@example("")
@example(",")
@example("+ bad")
@example(".subckt s a b\nR1 a b 1k")
@example(".ends")
@given(st.text(max_size=200))
def test_spice_library_arbitrary_text_only_raises_spice_error(
    scratch_directory: Path, text: str
) -> None:
    _exercise_spice_library(scratch_directory, text, filename="arbitrary.lib")


@FUZZ
@given(_spice_netlists(), st.sampled_from([_core.REFERENCE_TEMPERATURE, 301.15]))
def test_spice_library_grammar_guided_netlists_only_raise_spice_error(
    scratch_directory: Path, text: str, temperature: float
) -> None:
    _exercise_spice_library(
        scratch_directory,
        text,
        filename="deck.lib",
        temperature=temperature,
        exercise_drawn=True,
    )


@FAST_FUZZ
@given(_valid_spice_subcircuits())
def test_valid_spice_subcircuits_round_trip_through_export(text: str) -> None:
    library = Library()
    library.read(text, "valid.lib")
    stage = library.component("stage")

    @component
    def harness(builder: Builder) -> None:
        ground = builder.port("ground", Role.GROUND)
        source = builder.place(devices.voltage_source(1 * V), name="V1")
        placed = builder.place(stage, name="X1")
        builder.connect(source.positive, placed.inp)
        builder.connect(source.negative, placed.ground, ground)

    built = circuit(harness())
    Simulation(built)
    exported = export.netlist(built)
    assert exported
    assert netlist.statements("\n".join(exported), "exported.cir")


@pytest.mark.parametrize(
    ("text", "temperature", "match"),
    [
        # Self-recursive and mutually recursive subcircuits (was RecursionError).
        (".subckt s a b\nX1 a b s\n.ends", 300.15, "places itself"),
        (
            ".subckt s1 a b\nX1 a b s2\n.ends\n.subckt s2 a b\nX1 a b s1\n.ends",
            300.15,
            "places itself",
        ),
        # Geometric resistor effective width == 0 (was ZeroDivisionError).
        (
            (
                ".subckt s a b\n.model mr r rsh=10 narrow=5u short=5u\n"
                "R1 a b mr w=10u l=10u\n.ends"
            ),
            300.15,
            "effective width",
        ),
        # Geometric resistor flicker noise zero width (was ZeroDivisionError).
        (
            ".subckt s a b\n.model mr r rsh=10 kf=1e-12\nR1 a b mr w=0 l=10u\n.ends",
            300.15,
            "effective length and width",
        ),
        # Negative effective length with fractional exponent (was complex TypeError).
        (
            (
                ".subckt s a b\n.model mr r rsh=10 short=1u kf=1e-12 lf=0.5\n"
                "R1 a b mr w=1u l=1u\n.ends"
            ),
            300.15,
            "effective length and width",
        ),
        # Passive copies m=0 and m=-1 with flicker noise.
        (".subckt s a b\nR1 a b 1k m=0\n.ends", 300.15, "positive and finite"),
        (".subckt s a b\nL1 a b 1u m=0\n.ends", 300.15, "positive and finite"),
        (
            (
                ".subckt s a b\n.model mr r rsh=10 kf=1e-12 af=1.5\n"
                "R1 a b mr w=1u l=1u m=-1\n.ends"
            ),
            300.15,
            "positive and finite",
        ),
        # Subcircuit multiplier m=0 (was ZeroDivisionError).
        (
            (
                ".subckt inner a b\nR1 a b 1k\n.ends\n"
                ".subckt s a b\nX1 a b inner m=0\n.ends"
            ),
            300.15,
            "positive and finite",
        ),
        # Capacitor huge exponent integral (was RecursionError / hang).
        (
            ".subckt s a b\nC1 a b C={v(a,b)^500}\n.ends",
            300.15,
            "integral in closed form",
        ),
        # VBIC area=0, m=0, m=-1 (was ZeroDivisionError / complex TypeError).
        (
            ".subckt s a b c\n.model mq npn level=4\nQ1 a b c mq area=0\n.ends",
            300.15,
            "area times its copies",
        ),
        (
            (
                ".subckt s a b c\n.model mq npn level=4 rth=10\n"
                "Q1 a b c 0 t mq m=0\n.ends"
            ),
            300.15,
            "positive and finite",
        ),
        (
            (
                ".subckt s a b c\n.model mq npn level=4 kfn=1e-12 afn=1.5\n"
                "Q1 a b c mq m=-1\n.ends"
            ),
            300.15,
            "positive and finite",
        ),
        # Binned MOSFET nf=0 with wnflag=1 (was ZeroDivisionError).
        (
            (
                ".option wnflag=1\n.subckt s d g so b\n"
                ".model m.0 nmos level=8 lmin=0 lmax=1 wmin=0 wmax=1\n"
                "M1 d g so b m w=1u l=1u nf=0\n.ends"
            ),
            300.15,
            "fingers `NF`",
        ),
        # Scale option overflow on MOSFET ad scaling (was OverflowError).
        (
            (
                ".option scale=1e308\n.subckt s d g so b\n.model m nmos level=1\n"
                "M1 d g so b m w=1u l=1u ad=1\n.ends"
            ),
            300.15,
            "scale option overflows",
        ),
        # Missing expression after VALUE / VALUE = (was IndexError).
        (".subckt s a b c d\nE1 a b VALUE\n.ends", 300.15, "VALUE source"),
        (".subckt s a b c d\nE1 a b VALUE =\n.ends", 300.15, "VALUE source"),
        # POLY(0) infinite loop and POLY(9999999) allocation hang.
        (".subckt s a b\nE1 a b POLY(0) 1 2\n.ends", 300.15, "one or more controls"),
        (".subckt s a b\nE1 a b POLY(9999999) a b 1\n.ends", 300.15, "controls"),
        # Independent current source with missing nodes (was unpacking ValueError).
        (".subckt s a b\nI1\n.ends", 300.15, "joins two nodes"),
        (".subckt s a b\nI1 a\n.ends", 300.15, "joins two nodes"),
        # Empty drift keyword assignment tc1= (was IndexError).
        (".subckt s a b\nR1 a b 1k tc1=\n.ends", 300.15, "not assigned a value"),
        # Switch referencing non-switch model card (was KeyError).
        (
            ".subckt s a b c d\n.model md d\nS1 a b c d md\n.ends",
            300.15,
            "model of type `d`",
        ),
        (
            ".subckt s a b\n.model md d\nV1 a 0 1\nW1 a b V1 md\n.ends",
            300.15,
            "model of type `d`",
        ),
        # Transmission line F=0 (was ZeroDivisionError).
        (
            ".subckt s a b c d\nT1 a b c d Z0=50 F=0\n.ends",
            300.15,
            "frequency `F`",
        ),
        # Non-finite values and Rust device validation errors (were raw ValueError).
        (".subckt s a b\nR1 a b 1k temp=1e500\n.ends", 300.15, "not finite"),
        (".subckt s a b\nV1 a b PWL(1 0 0 1)\n.ends", 300.15, "strictly increase"),
        (".subckt s a b\nV1 a b PWL(1 0 0 1) DC 0\n.ends", 300.15, "strictly increase"),
        (
            ".subckt s a b c d\nE1 a b TABLE {V(c,d)} = (0,0) (0,1)\n.ends",
            300.15,
            "ascending abscissae",
        ),
        (
            (
                ".subckt s d g so b\n.model m nmos level=8 mobmod=1e308\n"
                "M1 d g so b m w=1u l=1u\n.ends"
            ),
            300.15,
            "u8",
        ),
        (
            ".subckt s a b\nL1 a b 1u\nL2 a b 1u\nK1 L1 L2 1e500\n.ends",
            300.15,
            "not finite",
        ),
    ],
)
def test_spice_component_regression_bugs(
    text: str, temperature: float, match: str
) -> None:
    library = Library(temperature=temperature)
    library.read(text, "regression.lib")
    target = "s1" if "s1" in library.subcircuits else "s"
    with pytest.raises(SpiceError, match=match):
        library.component(target)
    # survey() must report the SpiceError cleanly and never crash.
    assert library.survey(target)


@pytest.mark.parametrize(
    ("text", "temperature", "match"),
    [
        # Behavioral resistor temperature factor validates dynamically at elaboration.
        (
            ".subckt s a b\nR1 a b {v(a,b)} tc1=-1\n.ends",
            301.15,
            "temperature coefficients",
        ),
        (
            ".subckt s a b\nR1 a b {v(a,b)} tc2=1\n.ends",
            1e200,
            "temperature coefficients",
        ),
    ],
)
def test_behavioral_resistor_temperature_factor_errors_at_elaboration(
    text: str, temperature: float, match: str
) -> None:
    library = Library(temperature=temperature)
    library.read(text, "regression.lib")
    part = library.component("s")
    assert library.survey("s")

    @component
    def bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        placed = c.place(part, name="X1")
        source = c.place(devices.voltage_source(1 * V), name="V1")
        c.connect(source.positive, placed["a"])
        c.connect(source.negative, placed["b"], ground)

    with pytest.raises(ElaborationError, match=match):
        Simulation(bench(), temperature=temperature * K)


def test_spice_include_and_survey_model_regression_bugs(tmp_path: Path) -> None:
    # Comma-only statement inside include() previously crashed with IndexError.
    comma_file = tmp_path / "comma.lib"
    comma_file.write_text(",\n", encoding="utf-8")
    with pytest.raises(SpiceError):
        Library().include(comma_file)

    # Non-finite model level previously crashed survey_model with OverflowError.
    for text in (".model m nmos level=1/0\n", ".model m nmos level=0/0\n"):
        library = Library()
        library.read(text, "level.lib")
        assert library.survey_model("m").problem is not None

    # Non-finite spread in random_variables raises SpiceError, not ValidationError.
    library = Library()
    library.read(
        ".subckt s a b\n.param p=agauss(1, 1e308, 1e-10)\nR1 a b {p}\n.ends",
        "random_variables.lib",
    )
    with pytest.raises(SpiceError, match="non-finite"):
        library.random_variables("s", "X1")
