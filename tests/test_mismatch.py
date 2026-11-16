# Copyright 2026 Maxence Faldor
"""Mismatch analyses against closed forms, and against Monte Carlo on one circuit.

Monte Carlo's standard deviation over `n` samples has a relative standard error of
``sqrt((kurtosis - 1) / n) / 2``, the delta method's on the sample variance's, whose
variance is ``σ⁴ (kurtosis - 1) / n``; each comparison takes the kurtosis the samples
show, and holds the two standard deviations within five such standard errors, as
`test_kits.py` holds its draws. The draws are fixed by their seeds, so the tests are
deterministic.
"""

import math
from collections.abc import Callable
from typing import Final

import numpy as np
import pytest

from jolty import Builder, Component, Role, Simulation, component, devices, ir, results
from jolty.plan import Ac, Mismatch, MonteCarlo, OperatingPoint, decade
from jolty.report import report
from jolty.settings import Settings
from jolty.spice import Library, Sample, kits, sg13g2, sky130, standard_deviation
from jolty.units import A, GHz, MHz, V, kHz, kohm, nF, uA, um

STANDARD_ERRORS: Final = 5
"""How many of Monte Carlo's standard errors the two standard deviations lie within."""


def agrees(first_order: float, samples: np.ndarray) -> None:
    """Checks that `samples` spread with the standard deviation `first_order`."""
    count = len(samples)
    centered = samples - samples.mean()
    kurtosis = np.mean(centered**4) / np.mean(centered**2) ** 2
    error = math.sqrt((kurtosis - 1) / count) / 2
    found = samples.std(ddof=1)
    assert abs(found / first_order - 1) < STANDARD_ERRORS * error, (
        f"Monte Carlo {found!r} against {first_order!r}, a standard error {error:.3g}"
    )


# Closed forms.


@component
def divider(c: Builder) -> None:
    """1 kΩ over 3 kΩ, fed 1 V, both from one lot that scales them alike."""
    ground = c.port("ground", Role.GROUND)
    lot = ir.Lot(name="die", relative=0.2, distribution="normal")
    upper = devices.resistor(1 * kohm).toleranced("resistance", 0.03, "normal", lot)
    lower = devices.resistor(3 * kohm).toleranced("resistance", 0.01, lot=lot)
    source = c.place(devices.voltage_source(1 * V), name="V1")
    placed_upper = c.place(upper, name="R1")
    placed_lower = c.place(lower, name="R2")
    c.connect(source.positive, placed_upper.a, name="in")
    c.connect(placed_upper.b, placed_lower.a, name="out")
    c.connect(placed_lower.b, source.negative, ground)


def test_a_dividers_spread_is_its_closed_form() -> None:
    """Each resistor's own factor spreads the ratio, and the lot that scales both not.

    `out = R2 / (R1 + R2)` moves by `-R2 / (R1 + R2)²` with `R1` and `R1 / (R1 + R2)²`
    with `R2`; a normal tolerance of 3 % is a standard deviation of 1 %, a uniform one
    of 1 % is one of `1 % / √3`, and a common factor moves the ratio by nothing. The
    divider's two unknowns solve exactly but for rounding, so the two agree within
    1e-12, about 4500 ε.
    """
    found = Simulation(divider()).run(Mismatch(["out"]))
    named = [(each.instance, each.lot, each.name) for each in found.random_variables]
    assert named == [
        ("R1", None, "resistance"),
        ("R2", None, "resistance"),
        (None, "die", "resistance"),
    ]
    upper, lower = 1e3, 3e3
    contributions = [
        -lower / (upper + lower) ** 2 * upper * 0.01,
        upper / (upper + lower) ** 2 * lower * 0.01 / math.sqrt(3),
        0.0,
    ]
    assert found.contributions("out") == pytest.approx(contributions, abs=1e-15)
    deviation = math.hypot(*contributions)
    assert found.standard_deviation("out") == pytest.approx(deviation, rel=1e-12)
    assert found.shares("out") == pytest.approx(
        [(each / deviation) ** 2 for each in contributions], abs=1e-12
    )
    # The lot's contribution is 0 but for rounding, which its line would show.
    assert report("spread", found, 10).describe().splitlines()[1:4] == [
        "  out = 750 mV, standard deviation 2.165 mV",
        "    R1       resistance  -1.875 mV  75.0%",
        "    R2       resistance  1.083 mV   25.0%",
    ]
    assert found.random_variables[2].derivatives == {
        "R1": {"resistance": pytest.approx(upper)},
        "R2": {"resistance": pytest.approx(lower)},
    }


@component
def low_pass(c: Builder) -> None:
    """A 10 kΩ, 1 nF low-pass, each part within 3 %, normally."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(0 * V), name="V1")
    resistor = devices.resistor(10 * kohm).toleranced("resistance", 0.03, "normal")
    capacitor = devices.capacitor(1 * nF).toleranced("capacitance", 0.03, "normal")
    placed_resistor = c.place(resistor, name="R1")
    placed_capacitor = c.place(capacitor, name="C1")
    c.connect(source.positive, placed_resistor.a, name="in")
    c.connect(placed_resistor.b, placed_capacitor.a, name="out")
    c.connect(placed_capacitor.b, source.negative, ground)


def test_a_responses_magnitude_and_phase_spread_as_their_closed_forms() -> None:
    """A low-pass's magnitude and phase spread with `x = ωRC`, each part's 1 % of it.

    `|H| = (1 + x²)^(-1/2)` moves by `-x (1 + x²)^(-3/2)` and its phase `-atan(x)` by
    `-1 / (1 + x²)` per unit of `x`, which each part's standard deviation of 1 % moves
    by `x / 100`, the two apart. The circuit is linear and its operating point does not
    move, so the derivatives are the adjoint's alone, exact but for rounding: 1e-10
    holds them, about 450,000 ε, as each frequency's solve rounds.
    """
    frequencies = decade(1 * kHz, 1 * MHz, 3)
    found = Simulation(low_pass()).run(Mismatch(["out"], of=Ac("V1", frequencies)))
    assert isinstance(found.result, results.AcResult)
    x = 2 * math.pi * found.result.frequencies * 1e4 * 1e-9
    moved = x * 0.01 * math.sqrt(2)
    magnitude = x * (1 + x**2) ** -1.5 * moved
    phase = moved / (1 + x**2)
    spread = found.standard_deviation("out", "magnitude")
    assert spread == pytest.approx(magnitude, rel=1e-10)
    assert found.standard_deviation("out", "phase") == pytest.approx(phase, rel=1e-10)
    with pytest.raises(ValueError, match="magnitude or its phase"):
        found.standard_deviation("out")
    text = report("spread", found, 10).describe()
    assert "out's magnitude: standard deviation largest" in text
    assert "out's phase" in text


# A process kit's statistical functions.


@pytest.fixture(scope="session")
def kept(pytestconfig: pytest.Config) -> Settings:
    """Settings whose cache, pytest's, holds the kits' pinned files."""
    assert pytestconfig.cache is not None
    return Settings(cache=pytestconfig.cache.mkdir("downloads"))


HBT: Final = (sg13g2.DIRECTORY / "cornerHBT.lib", "hbt_typ_mismatch")
"""SG13G2's bipolar transistors at their typical corner, drawing their mismatch."""


@pytest.fixture(scope="session")
def hbt(kept: Settings) -> Library:
    """SG13G2's bipolar transistors, read where they draw their mismatch."""
    return kits.read(sg13g2.KIT, (HBT,), settings=kept)


def test_a_placements_random_variables_are_its_kits_calls(hbt: Library) -> None:
    """Each `agauss` of `npn13G2` that moves its transistor is a random variable.

    The card's `is` is `4.81e-17 (Nx/4) vbic_is (1 + mm_ok (is_mm - 1) / √(m Nx))`, so
    the saturation current moves by 1.2025e-17 A per unit of `is_mm`, of standard
    deviation `is_var_mm`, 0.038. The central difference rounds by about ε times the
    number over its step, 0.038 times the cube root of ε, so by about 1e-9: 1e-8 holds
    it with room for the reading's own rounding.
    """
    found = {each.name: each for each in hbt.random_variables("npn13G2", "Q1", mm_ok=1)}
    assert set(found) == {
        f"{name}_mm#0"
        for name in ("is", "ibei", "cje", "cjc", "cjcp", "re", "rcx", "rbx", "tf")
    }
    saturation = found["is_mm#0"]
    assert (saturation.instance, saturation.mean) == ("Q1", 1.0)
    assert saturation.standard_deviation == pytest.approx(0.038, rel=1e-15)
    moved = saturation.derivatives["Q1/Qnpn13G2"]["saturation_current"]
    assert moved == pytest.approx(4.81e-17 / 4, rel=1e-8)
    assert set(found["rbx_mm#0"].derivatives["Q1/Qnpn13G2"]) == {
        "extrinsic_base_resistance",
        "intrinsic_base_resistance",
    }
    # Without `mm_ok`, every call's spread is 0, and no random variable is drawn.
    assert hbt.random_variables("npn13G2", "Q1") == ()


def test_each_statistical_function_spreads_as_it_draws() -> None:
    """A function's standard deviation is the one its draws show over many samples.

    The bounds are five standard errors of the standard deviation over `SAMPLES`
    draws of each, the draws fixed by their seed.
    """
    calls = {
        "agauss": (2.0, 0.3, 3.0),
        "gauss": (-2.0, 0.15, 3.0),
        "aunif": (2.0, -0.1),
        "unif": (-2.0, 0.05),
        "limit": (2.0, 0.1),
    }
    for function, arguments in calls.items():
        values = np.array(
            [
                Sample(13, index).draw("X1", "x")(function, arguments)
                for index in range(SAMPLES)
            ]
        )
        agrees(standard_deviation(function, arguments), values)
    assert standard_deviation("agauss", (2.0, 0.3, 0.0)) == 0


SAMPLES: Final = 4000
"""The samples of each Monte Carlo analysis whose runs are one batch of the engine's."""

KIT_SAMPLES: Final = 1000
"""The samples of each Monte Carlo analysis that reads a kit's placements anew, each
one reading of a placement taking about 4 ms."""


# Against Monte Carlo: tolerances, drawn by the engine.


def transistor() -> devices.Device:
    """A level 1 MOSFET, its threshold and transconductance within 3 %, normally."""
    made = devices.mosfet(
        channel="n",
        width=10 * um,
        length=1 * um,
        threshold_voltage=0.7 * V,
        transconductance_parameter=200e-6 * A / V**2,
        channel_length_modulation=0.02 / V,
    )
    spread = made.toleranced("threshold_voltage", 0.03, "normal")
    return spread.toleranced("transconductance_parameter", 0.03, "normal")


GAIN: Final = 1e4
"""The gain that feeds a pair of MOSFETs' output back to its second input, so that the
input follows the first, off by the pair's input offset times `GAIN / (1 + GAIN)`."""


@component
def mosfet_pair(c: Builder) -> None:
    """A differential pair of `transistor`s, its second input following its first."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(3 * V), name="VDD")
    held = c.place(devices.voltage_source(1.5 * V), name="VIN")
    tail = c.place(devices.current_source(100 * uA), name="IT")
    first = c.place(transistor(), name="M1")
    second = c.place(transistor(), name="M2")
    loads = [c.place(devices.resistor(10 * kohm), name=name) for name in ("R1", "R2")]
    gain = c.place(devices.voltage_controlled_voltage_source(GAIN), name="E1")
    c.connect(supply.positive, loads[0].a, loads[1].a, name="vdd")
    c.connect(loads[0].b, first.drain, gain.control_negative, name="d1")
    c.connect(loads[1].b, second.drain, gain.control_positive, name="d2")
    c.connect(held.positive, first.gate, name="in")
    c.connect(second.gate, gain.positive, name="out")
    c.connect(first.source, second.source, tail.positive, name="tail")
    c.connect(supply.negative, held.negative, tail.negative, gain.negative, ground)
    c.connect(first.bulk, second.bulk, ground)


@component
def mosfet_mirror(c: Builder) -> None:
    """A current mirror of `transistor`s, 100 µA in, its output held at 1.5 V."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(3 * V), name="VDD")
    driven = c.place(devices.current_source(100 * uA), name="IIN")
    held = c.place(devices.voltage_source(1.5 * V), name="VOUT")
    first = c.place(transistor(), name="M1")
    second = c.place(transistor(), name="M2")
    c.connect(supply.positive, driven.positive, name="vdd")
    c.connect(driven.negative, first.drain, first.gate, second.gate, name="gate")
    c.connect(held.positive, second.drain, name="out")
    c.connect(supply.negative, held.negative, first.source, second.source, ground)
    c.connect(first.bulk, second.bulk, ground)


@pytest.mark.parametrize(
    ("circuit", "output"),
    [(mosfet_pair, "out"), (mosfet_mirror, "VOUT:current")],
    ids=["pair-offset", "mirror-output"],
)
def test_tolerances_spread_as_monte_carlo_finds(
    circuit: Callable[[], Component], output: str
) -> None:
    """A pair's input offset and a mirror's output current, from 3 % thresholds.

    Each transistor's threshold and transconductance parameter is within 3 %, a
    standard deviation of 1 %; the threshold's 7 mV of each transistor makes the
    pair's offset, and the mirror's output, the Monte Carlo analysis's own within its
    statistics.
    """
    simulation = Simulation(circuit())
    first_order = float(simulation.run(Mismatch([output])).standard_deviation(output))
    batch = simulation.batch(
        {"op": OperatingPoint()}, MonteCarlo(SAMPLES, seed=1), traces=[output]
    )
    agrees(first_order, batch.traces[0].values_of(output)[:, 0])


# Against Monte Carlo: a kit's mismatch, each sample read anew.


def bipolar(library: Library, path: str, index: int | None) -> Component:
    """SG13G2's `npn13G2` drawing its mismatch, nominal or in the sample `index`."""
    if index is None:
        return library.component("npn13G2", mm_ok=1)
    return library.drawn("npn13G2", Sample(5, index), path, mm_ok=1)


@component
def bipolar_pair(c: Builder, library: Library, index: int | None) -> None:
    """A differential pair of `npn13G2`s, both bases at 1.2 V.

    Its output is its collectors' difference: its input offset times its gain, about 19.
    """
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(2.5 * V), name="VDD")
    held = c.place(devices.voltage_source(1.2 * V), name="VIN")
    tail = c.place(devices.current_source(200 * uA), name="IT")
    first = c.place(bipolar(library, "Q1", index), name="Q1")
    second = c.place(bipolar(library, "Q2", index), name="Q2")
    loads = [c.place(devices.resistor(5 * kohm), name=name) for name in ("R1", "R2")]
    sensed = c.place(devices.voltage_controlled_voltage_source(1), name="E1")
    c.connect(supply.positive, loads[0].a, loads[1].a, name="vdd")
    c.connect(loads[0].b, first.c, sensed.control_negative, name="c1")
    c.connect(loads[1].b, second.c, sensed.control_positive, name="c2")
    c.connect(held.positive, first.b, second.b, name="in")
    c.connect(sensed.positive, name="out")
    c.connect(first.e, second.e, tail.positive, name="tail")
    c.connect(supply.negative, held.negative, tail.negative, sensed.negative, ground)
    c.connect(first.bn, second.bn, first.ground, second.ground, ground)


@component
def bipolar_mirror(c: Builder, library: Library, index: int | None) -> None:
    """A current mirror of `npn13G2`s, 100 µA in, its output held at 1.5 V."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(2.5 * V), name="VDD")
    driven = c.place(devices.current_source(100 * uA), name="IIN")
    held = c.place(devices.voltage_source(1.5 * V), name="VOUT")
    first = c.place(bipolar(library, "Q1", index), name="Q1")
    second = c.place(bipolar(library, "Q2", index), name="Q2")
    c.connect(supply.positive, driven.positive, name="vdd")
    c.connect(driven.negative, first.c, first.b, second.b, name="base")
    c.connect(held.positive, second.c, name="out")
    c.connect(supply.negative, held.negative, first.e, second.e, ground)
    c.connect(first.bn, second.bn, first.ground, second.ground, ground)


@pytest.mark.parametrize(
    ("circuit", "output"),
    [(bipolar_pair, "out"), (bipolar_mirror, "VOUT:current")],
    ids=["pair-offset", "mirror-output"],
)
def test_sg13g2s_mismatch_spreads_as_monte_carlo_finds(
    hbt: Library,
    circuit: Callable[[Library, int | None], Component],
    output: str,
) -> None:
    """SG13G2's VBIC pair and mirror, from the kit's own mismatch of each transistor.

    The saturation current's 3.8 % of each transistor makes nearly all of either
    spread, its emitter resistance's 5.5 % the rest; Monte Carlo reads each placement
    anew in each sample.
    """
    given = [
        variable
        for path in ("Q1", "Q2")
        for variable in hbt.random_variables("npn13G2", path, mm_ok=1)
    ]
    found = Simulation(circuit(hbt, None)).run(
        Mismatch([output], random_variables=given)
    )
    largest = np.argsort(-found.shares(output))[:2]
    named = {found.random_variables[index].name for index in largest}
    assert named == {"is_mm#0"}
    column = results.column(found.unknowns, output)
    samples = np.array(
        [
            Simulation(circuit(hbt, index)).run(OperatingPoint()).values[column]
            for index in range(KIT_SAMPLES)
        ]
    )
    agrees(float(found.standard_deviation(output)), samples)


@pytest.fixture(scope="session")
def mim(kept: Settings) -> Library:
    """SKY130 at its typical corner, drawing its mismatch."""
    section = f"{sky130.TYPICAL}{sky130.MISMATCH}"
    return kits.read(sky130.KIT, ((sky130.LIBRARY, section),), settings=kept)


MIM: Final = "sky130_fd_pr__cap_mim_m3_1"
"""SKY130's MIM capacitor, between its third and fourth metals."""


def capacitor(library: Library, index: int | None) -> Component:
    """A 10 µm by 10 µm MIM capacitor, nominal or in the sample `index`."""
    if index is None:
        return library.component(MIM, w=10, l=10)
    return library.drawn(MIM, Sample(9, index), "C1", w=10, l=10)


@component
def mim_low_pass(c: Builder, library: Library, index: int | None) -> None:
    """A 10 kΩ resistor into a MIM capacitor of about 0.2 pF, a pole near 80 MHz."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(0 * V), name="V1")
    resistor = c.place(devices.resistor(10 * kohm), name="R1")
    placed = c.place(capacitor(library, index), name="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, placed.c0, name="out")
    c.connect(placed.c1, source.negative, ground)


def test_sky130s_mim_capacitor_spreads_a_response_as_monte_carlo_finds(
    mim: Library,
) -> None:
    """A low-pass's magnitude and phase over its pole, from the capacitor's mismatch.

    The kit spreads the capacitance by 2.8 % over the square root of its area in
    square micrometres, its one random variable, whose derivative is that of the
    capacitance `czero` the subcircuit's parameters compute.
    """
    (variable,) = mim.random_variables(MIM, "C1", w=10, l=10)
    assert (variable.instance, variable.name) == ("C1", "czero#0")
    analysis = Ac("V1", decade(10 * MHz, 1 * GHz, 2))
    found = Simulation(mim_low_pass(mim, None)).run(
        Mismatch(["out"], of=analysis, random_variables=[variable])
    )
    phasors = np.array(
        [
            Simulation(mim_low_pass(mim, index)).run(analysis).voltage("out")
            for index in range(KIT_SAMPLES)
        ]
    )
    for part, samples in (("magnitude", np.abs(phasors)), ("phase", np.angle(phasors))):
        spreads = found.standard_deviation("out", part)
        for at, spread in enumerate(np.atleast_1d(spreads)):
            agrees(float(spread), samples[:, at])
