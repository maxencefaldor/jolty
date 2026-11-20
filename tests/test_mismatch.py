# Copyright 2026 Maxence Faldor
"""Mismatch analyses verified against closed forms and Monte Carlo simulation.

By the delta method, a Monte Carlo standard deviation over `n` samples has relative
standard error ``sqrt((kurtosis - 1) / n) / 2``, derived from sample-variance variance
``σ⁴ (kurtosis - 1) / n``. Each comparison uses the sample kurtosis and checks
agreement within five standard errors, matching `test_kits.py`. Fixed seeds keep all
draws deterministic.
"""

import math
from collections.abc import Callable
from typing import Final

import numpy as np
import pytest

from jolty import (
    Builder,
    Component,
    Role,
    Simulation,
    circuit,
    component,
    devices,
    ir,
    results,
)
from jolty.plan import (
    Ac,
    DcSweep,
    Mismatch,
    MonteCarlo,
    OperatingPoint,
    Options,
    PeriodicSteadyState,
    decade,
)
from jolty.report import report
from jolty.settings import Settings
from jolty.spice import Library, Sample, kits, sg13g2, sky130, standard_deviation
from jolty.units import (
    A,
    F,
    GHz,
    H,
    MHz,
    V,
    celsius,
    format_si,
    kHz,
    kohm,
    nF,
    uA,
    um,
)
from jolty.variation import sample, toleranced

STANDARD_ERRORS: Final = 5
"""Number of Monte Carlo standard errors allowed between standard deviations."""


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
    """Builds a 1 kΩ / 3 kΩ divider driven at 1 V with a shared lot scale factor."""
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
    """Verifies that divider mismatch matches closed-form sensitivity expressions.

    The ratio `out = R2 / (R1 + R2)` has derivatives `-R2 / (R1 + R2)²` with respect to
    `R1` and `R1 / (R1 + R2)²` with respect to `R2`. A 3 % normal tolerance gives a 1 %
    standard deviation, a 1 % uniform tolerance gives `1 % / √3`, and the common lot
    factor cancels. The linear solve is exact apart from rounding, so both agree within
    1e-12 (about 4500 ε).
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
    # The lot contribution is zero apart from rounding, which its line would show.
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
    """Builds a 10 kΩ, 1 nF low-pass filter with 3 % normal tolerances."""
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
    """Verifies low-pass AC magnitude and phase mismatch against closed-form formulas.

    With `x = ωRC`, magnitude `|H| = (1 + x²)^(-1/2)` changes by `-x (1 + x²)^(-3/2)`
    and phase `-atan(x)` changes by `-1 / (1 + x²)` per unit `x`. Each component has a
    1 % standard deviation, contributing `x / 100` independently. The circuit is linear
    with a fixed operating point, so adjoint derivatives are exact apart from rounding
    and agree within 1e-10 (about 450,000 ε).
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


def test_a_sweeps_spread_scales_with_its_level() -> None:
    """Verifies that swept divider output spread scales linearly with supply voltage.

    With `out = V R2 / (R1 + R2)`, each contribution equals the 1 V operating-point
    value scaled by the sweep level, up to floating-point rounding.
    """
    levels = [0.5 * V, 1 * V, 2 * V]
    simulation = Simulation(divider())
    swept = simulation.run(Mismatch(["out"], of=DcSweep("V1", levels)))
    point = simulation.run(Mismatch(["out"]))
    expected = np.outer([0.5, 1, 2], point.contributions("out"))
    assert swept.contributions("out") == pytest.approx(expected, rel=1e-12, abs=1e-15)
    text = report("swept", swept, 10).describe()
    assert "out: standard deviation largest, 4.33 mV, at 2 V" in text


def test_an_oscillators_period_spreads_with_its_tank() -> None:
    """Verifies that oscillator period spread scales with tank L and C.

    Each tank random variable shifts the period by its value times the corresponding
    period derivative. The two contributions sum to the period (`L ∂T/∂L + C ∂T/∂C =
    T`), matching `test_periodic.py` sensitivities within 1e-12 at reltol 1e-6.
    """
    inductance, capacitance = 1e-6, 1e-9
    impedance = math.sqrt(inductance / capacitance)
    expression = (
        f"-{0.2 / impedance!r} * v(x) * (1 - (v(x) ^ 2 + ({impedance!r} * i(p)) ^ 2))"
    )

    @component
    def tank(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        coil = devices.inductor(inductance * H).toleranced("inductance", 0.03, "normal")
        plates = devices.capacitor(capacitance * F)
        capacitor = c.place(plates.toleranced("capacitance", 0.03, "normal"), name="C1")
        inductor = c.place(coil, name="L1")
        regulation = c.place(devices.behavioral_current_source(expression), name="B1")
        c.connect(capacitor.a, inductor.a, regulation.positive, regulation.x, name="t")
        c.connect(inductor.b, regulation.p_positive, name="l")
        c.connect(capacitor.b, regulation.negative, regulation.p_negative, ground)

    found = Simulation(tank()).run(
        Mismatch(["t"], of=PeriodicSteadyState()), Options(relative_tolerance=1e-6)
    )
    assert isinstance(found.result, results.PeriodicSteadyStateResult)
    assert found.period_derivatives is not None
    period = found.result.period
    assert abs(sum(found.period_derivatives) / period - 1) <= 1e-12
    spread = math.hypot(*found.period_derivatives) * 0.01
    assert found.period_standard_deviation() == pytest.approx(spread, rel=1e-15)
    deviation = format_si(found.period_standard_deviation(), "s")
    assert f"standard deviation {deviation}" in report("ring", found, 10).describe()


# A process kit's statistical functions.


@pytest.fixture(scope="session")
def kept(pytestconfig: pytest.Config) -> Settings:
    """Returns settings backed by the pytest cache directory for pinned PDK files."""
    assert pytestconfig.cache is not None
    return Settings(cache=pytestconfig.cache.mkdir("downloads"))


HBT: Final = (sg13g2.DIRECTORY / "cornerHBT.lib", "hbt_typ_mismatch")
"""SG13G2's bipolar transistors at their typical corner, drawing their mismatch."""


@pytest.fixture(scope="session")
def hbt(kept: Settings) -> Library:
    """Loads the SG13G2 bipolar library configured to evaluate mismatch draws."""
    return kits.read(sg13g2.KIT, (HBT,), settings=kept)


def test_a_placements_random_variables_are_its_kits_calls(hbt: Library) -> None:
    """Verifies that each effective `agauss` in `npn13G2` becomes a random variable.

    The card defines `is` as `4.81e-17 (Nx/4) vbic_is (1 + mm_ok (is_mm - 1) / √(m
    Nx))`, so saturation current changes by 1.2025e-17 A per unit of `is_mm` with
    standard deviation `is_var_mm` (0.038). Central-difference rounding scales as ε
    divided by step `0.038 * ε^(1/3)` (about 1e-9), so 1e-8 bounds the error.
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
    """Verifies that each statistical function's standard deviation matches large draws.

    Bounds equal five standard errors of the standard deviation across `SAMPLES` seeded
    draws per function.
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
"""The samples of each Monte Carlo analysis that reads SG13G2's placements anew, each
reading about 4 ms."""

MIM_SAMPLES: Final = 400
"""The samples of each Monte Carlo analysis that reads SKY130's placements anew:
five standard errors are 18 % of the standard deviation."""


# Against Monte Carlo: tolerances, drawn by the engine.


def transistor() -> devices.Device:
    """Builds a level-1 MOSFET with 3 % normal tolerances on threshold and beta."""
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
    """Builds a differential pair of `transistor` devices with matched gate bias."""
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
    """Builds a current mirror of `transistor` devices, 100 µA in and 1.5 V out."""
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
    """Verifies MOSFET pair offset and mirror current spread against Monte Carlo.

    Each transistor threshold and transconductance parameter varies by 3 % (1 %
    standard deviation). Threshold variation (7 mV per device) drives both pair offset
    and mirror output spread within Monte Carlo statistical bounds.
    """
    simulation = Simulation(circuit())
    first_order = float(simulation.run(Mismatch([output])).standard_deviation(output))
    batch = simulation.batch(
        {"op": OperatingPoint()}, MonteCarlo(SAMPLES, seed=1), traces=[output]
    )
    agrees(first_order, batch.traces[0].values_of(output)[:, 0])


# Against Monte Carlo: a kit's mismatch, each sample read anew.


def bipolar(library: Library, path: str, index: int | None) -> Component:
    """Builds an SG13G2 `npn13G2` device drawing nominal or indexed mismatch."""
    if index is None:
        return library.component("npn13G2", mm_ok=1)
    return library.drawn("npn13G2", Sample(5, index), path, mm_ok=1)


@component
def bipolar_pair(c: Builder, library: Library, index: int | None) -> None:
    """Builds a differential pair of `npn13G2` transistors with both bases at 1.2 V.

    The differential collector output equals the input offset times a gain of about 19.
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
    """Builds an `npn13G2` current mirror with 100 µA input and 1.5 V output."""
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
    """Verifies SG13G2 VBIC pair and mirror mismatch spread against Monte Carlo.

    Saturation-current variation (3.8 % per transistor) accounts for most of the
    spread, and emitter-resistance variation (5.5 %) accounts for the rest. Monte Carlo
    re-evaluates each placement in every sample.
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
    """Loads the SKY130 typical-corner library configured to draw mismatch."""
    section = f"{sky130.TYPICAL}{sky130.MISMATCH}"
    return kits.read(sky130.KIT, ((sky130.LIBRARY, section),), settings=kept)


MIM: Final = "sky130_fd_pr__cap_mim_m3_1"
"""SKY130's MIM capacitor, between its third and fourth metals."""


def capacitor(library: Library, index: int | None) -> Component:
    """Builds a 10 µm by 10 µm MIM capacitor drawing nominal or indexed mismatch."""
    if index is None:
        return library.component(MIM, w=10, l=10)
    return library.drawn(MIM, Sample(9, index), "C1", w=10, l=10)


@component
def mim_low_pass(c: Builder, library: Library, index: int | None) -> None:
    """Builds a 10 kΩ resistor into an ~0.2 pF MIM capacitor with a pole near 80 MHz."""
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
    """Verifies MIM low-pass AC magnitude and phase spread against Monte Carlo.

    The kit spreads capacitance by 2.8 % divided by the square root of area in square
    micrometers. The single random variable acts through the `czero` capacitance
    computed by the subcircuit parameters.
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
            for index in range(MIM_SAMPLES)
        ]
    )
    for part, samples in (("magnitude", np.abs(phasors)), ("phase", np.angle(phasors))):
        spreads = found.standard_deviation("out", part)
        for at, spread in enumerate(np.atleast_1d(spreads)):
            agrees(float(spread), samples[:, at])


NFET: Final = "sky130_fd_pr__nfet_01v8"
"""SKY130's 1.8 V n-channel BSIM4 MOSFET."""


def sky130_nfet(library: Library, path: str, index: int | None) -> Component:
    """Builds a 1 µm by 0.15 µm SKY130 BSIM4 MOSFET with nominal or indexed mismatch."""
    if index is None:
        return library.component(NFET, w=1.0, l=0.15)
    return library.drawn(NFET, Sample(11, index), path, w=1.0, l=0.15)


@component
def sky130_nfet_pair(c: Builder, library: Library, index: int | None) -> None:
    """Builds a SKY130 BSIM4 differential pair with both gates biased at 1.0 V.

    The differential drain output equals the input offset times the voltage gain.
    """
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(1.8 * V), name="VDD")
    held = c.place(devices.voltage_source(1.0 * V), name="VIN")
    tail = c.place(devices.current_source(100 * uA), name="IT")
    first = c.place(sky130_nfet(library, "M1", index), name="M1")
    second = c.place(sky130_nfet(library, "M2", index), name="M2")
    loads = [c.place(devices.resistor(5 * kohm), name=name) for name in ("R1", "R2")]
    sensed = c.place(devices.voltage_controlled_voltage_source(1), name="E1")
    c.connect(supply.positive, loads[0].a, loads[1].a, name="vdd")
    c.connect(loads[0].b, first.d, sensed.control_negative, name="d1")
    c.connect(loads[1].b, second.d, sensed.control_positive, name="d2")
    c.connect(held.positive, first.g, second.g, name="in")
    c.connect(sensed.positive, name="out")
    c.connect(first.s, second.s, tail.positive, name="tail")
    c.connect(supply.negative, held.negative, tail.negative, sensed.negative, ground)
    c.connect(first.b, second.b, ground)


@component
def sky130_nfet_mirror(c: Builder, library: Library, index: int | None) -> None:
    """Builds a SKY130 BSIM4 current mirror with 100 µA input and 1.2 V output."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(1.8 * V), name="VDD")
    driven = c.place(devices.current_source(100 * uA), name="IIN")
    held = c.place(devices.voltage_source(1.2 * V), name="VOUT")
    first = c.place(sky130_nfet(library, "M1", index), name="M1")
    second = c.place(sky130_nfet(library, "M2", index), name="M2")
    c.connect(supply.positive, driven.positive, name="vdd")
    c.connect(driven.negative, first.d, first.g, second.g, name="gate")
    c.connect(held.positive, second.d, name="out")
    c.connect(supply.negative, held.negative, first.s, second.s, ground)
    c.connect(first.b, second.b, ground)


@pytest.mark.parametrize(
    ("circuit", "output"),
    [(sky130_nfet_pair, "out"), (sky130_nfet_mirror, "VOUT:current")],
    ids=["pair-offset", "mirror-output"],
)
def test_sky130s_bsim4_mosfet_mismatch_spreads_as_monte_carlo_finds(
    mim: Library,
    circuit: Callable[[Library, int | None], Component],
    output: str,
) -> None:
    """Verifies SKY130 BSIM4 pair and mirror mismatch spread against Monte Carlo.

    Each placement draws three independent mismatch variables (`toxe`, `vth0`, `voff`)
    inside bin 71; first-order adjoint sensitivities agree with Monte Carlo over
    `MIM_SAMPLES` draws within five standard errors.
    """
    given = [
        variable
        for path in ("M1", "M2")
        for variable in mim.random_variables(NFET, path, w=1.0, l=0.15)
    ]
    assert {variable.name.rsplit(".", 1)[-1] for variable in given} == {
        "toxe#0",
        "vth0#0",
        "voff#0",
    }
    assert len(given) == 6
    found = Simulation(circuit(mim, None)).run(
        Mismatch([output], random_variables=given)
    )
    shares = found.shares(output)
    top_four = np.argsort(-shares)[:4]
    named = {
        found.random_variables[index].name.rsplit(".", 1)[-1] for index in top_four
    }
    # Both `vth0` and `toxe` dominate strong-inversion mismatch (>90% of variance);
    # `voff` contributes the subthreshold/moderate-inversion remainder.
    assert named == {"vth0#0", "toxe#0"}
    assert float(shares[top_four].sum()) > 0.9
    column = results.column(found.unknowns, output)
    samples = np.array(
        [
            Simulation(circuit(mim, index)).run(OperatingPoint()).values[column]
            for index in range(MIM_SAMPLES)
        ]
    )
    agrees(float(found.standard_deviation(output)), samples)


XHIGH_PO: Final = "sky130_fd_pr__res_xhigh_po_0p69"
"""SKY130 nested precision polysilicon resistor wrapping `res_xhigh_po__base`."""

GENERIC_PO: Final = "sky130_fd_pr__res_generic_po"
"""SKY130 generic polysilicon resistor whose local model card sets `TREF = 30.0`."""


def sky130_xhigh_po_bench(part: Component) -> Component:
    """Builds a 1 V operating-point bench for a SKY130 xhigh-poly resistor."""
    builder = Builder("sky130_xhigh_po_bench")
    ground = builder.port("ground", Role.GROUND)
    source = builder.place(devices.voltage_source(1 * V), name="V1")
    resistor = builder.place(part, name="R1")
    builder.connect(source.positive, resistor.r0, name="drive")
    builder.connect(resistor.r1, resistor.b, source.negative, ground)
    return builder.finish()


def test_sky130s_nested_resistor_draws_and_simulates(mim: Library) -> None:
    """Verifies SKY130 nested `res_xhigh_po_0p69` draws and elaborates natively.

    `sky130_fd_pr__res_xhigh_po_0p69` places `sky130_fd_pr__res_xhigh_po__base`,
    whose inner `agauss` draws produce a child path containing `'/'`. Encoding `'/'`
    in the qualified component name lets the circuit elaborate and simulate cleanly
    while keeping statistical variate keys bit-identical.
    """
    sample = Sample(17, 0)
    first = mim.drawn(XHIGH_PO, sample, "R1")
    assert first.children
    assert all(
        "/" not in key for key in circuit(sky130_xhigh_po_bench(first)).components
    )
    nominal_current = (
        Simulation(sky130_xhigh_po_bench(mim.component(XHIGH_PO)))
        .run(OperatingPoint())
        .current("V1")
    )
    first_current = (
        Simulation(sky130_xhigh_po_bench(first)).run(OperatingPoint()).current("V1")
    )
    repeated_current = (
        Simulation(sky130_xhigh_po_bench(mim.drawn(XHIGH_PO, sample, "R1")))
        .run(OperatingPoint())
        .current("V1")
    )
    sibling_current = (
        Simulation(sky130_xhigh_po_bench(mim.drawn(XHIGH_PO, sample, "R2")))
        .run(OperatingPoint())
        .current("V1")
    )
    next_current = (
        Simulation(sky130_xhigh_po_bench(mim.drawn(XHIGH_PO, Sample(17, 1), "R1")))
        .run(OperatingPoint())
        .current("V1")
    )
    assert first_current == repeated_current
    assert first_current != nominal_current
    assert first_current != sibling_current
    assert first_current != next_current
    assert isinstance(mim.random_variables(XHIGH_PO, "R1"), tuple)


def sky130_generic_po_bench(part: Component) -> Component:
    """Builds a 1 V operating-point bench for a SKY130 generic poly resistor."""
    builder = Builder("sky130_generic_po_bench")
    ground = builder.port("ground", Role.GROUND)
    source = builder.place(devices.voltage_source(1 * V), name="V1")
    resistor = builder.place(part, name="R1")
    builder.connect(source.positive, resistor.r0, name="drive")
    builder.connect(resistor.r1, source.negative, ground)
    return builder.finish()


def test_sky130s_generic_poly_resistor_reads_tref_and_draws_mismatch(
    mim: Library,
) -> None:
    """Verifies SKY130 `res_generic_po` reads `TREF = 30` and draws mismatch natively.

    Nominal-temperature assertion uses `rel=1e-12` for Celsius-to-Kelvin IEEE-754
    addition rounding (`30 + 273.15`).
    """
    nominal = mim.component(GENERIC_PO, w=1, l=10)
    (element,) = nominal.definition.instances
    assert element.parameters["nominal_temperature"] == pytest.approx(
        celsius(30).value, rel=1e-12
    )
    (variable,) = mim.random_variables(GENERIC_PO, "R1", w=1, l=10)
    assert (variable.instance, variable.name) == ("R1", "rp1_mm#0")
    spread = Simulation(sky130_generic_po_bench(nominal)).run(
        Mismatch(["V1:current"], random_variables=[variable])
    )
    assert float(spread.standard_deviation("V1:current")) > 0
    nominal_current = (
        Simulation(sky130_generic_po_bench(nominal)).run(OperatingPoint()).current("V1")
    )
    drawn_current = (
        Simulation(
            sky130_generic_po_bench(
                mim.drawn(GENERIC_PO, Sample(19, 0), "R1", w=1, l=10)
            )
        )
        .run(OperatingPoint())
        .current("V1")
    )
    assert drawn_current != nominal_current


@pytest.mark.parametrize(
    ("bench", "output"),
    [(bipolar_pair, "out"), (bipolar_mirror, "VOUT:current")],
    ids=["pair-offset", "mirror-output"],
)
def test_native_monte_carlo_matches_sg13g2_drawn_and_mismatch(
    hbt: Library,
    bench: Callable[[Library, int | None], Component],
    output: str,
) -> None:
    """Verifies native Monte Carlo matches SG13G2 `Library.drawn` and `Mismatch`.

    Operating-point outputs match `Library.drawn` sample-by-sample within `rel=1e-10,
    abs=1e-12` (Newton DC reltol `1e-6` convergence and floating-point expression
    evaluation order), and native batch standard deviation over `SAMPLES` draws agrees
    with first-order `Mismatch` within five standard errors.
    """
    given = [
        variable
        for path in ("Q1", "Q2")
        for variable in hbt.random_variables("npn13G2", path, mm_ok=1)
    ]
    nominal = bench(hbt, None)
    simulation = Simulation(nominal)
    first_order = simulation.run(Mismatch([output], random_variables=given))
    column = results.column(first_order.unknowns, output)

    checked = 16
    batch = simulation.batch(
        {"op": OperatingPoint()}, MonteCarlo(checked, seed=5), traces=[output]
    )
    assert not batch.failed.any()
    native = batch.traces[0].values_of(output)[:, 0]
    for index in range(checked):
        expected = Simulation(bench(hbt, index)).run(OperatingPoint()).values[column]
        assert native[index] == pytest.approx(expected, rel=1e-10, abs=1e-12)

    large = simulation.batch(
        {"op": OperatingPoint()}, MonteCarlo(SAMPLES, seed=5), traces=[output]
    )
    assert not large.failed.any()
    agrees(
        float(first_order.standard_deviation(output)),
        large.traces[0].values_of(output)[:, 0],
    )


def test_native_monte_carlo_matches_sky130_mosfet_drawn_and_mismatch(
    mim: Library,
) -> None:
    """Verifies native Monte Carlo matches SKY130 BSIM4 `Library.drawn` and `Mismatch`.

    Sample-by-sample agreement uses `rel=1e-10, abs=1e-12` for Newton convergence and
    floating-point evaluation order; batch spread agrees with first-order `Mismatch`
    within five standard errors.
    """
    given = [
        variable
        for path in ("M1", "M2")
        for variable in mim.random_variables(NFET, path, w=1.0, l=0.15)
    ]
    simulation = Simulation(sky130_nfet_pair(mim, None))
    first_order = simulation.run(Mismatch(["out"], random_variables=given))
    column = results.column(first_order.unknowns, "out")

    checked = 12
    batch = simulation.batch(
        {"op": OperatingPoint()}, MonteCarlo(checked, seed=11), traces=["out"]
    )
    assert not batch.failed.any()
    native = batch.traces[0].values_of("out")[:, 0]
    for index in range(checked):
        expected = (
            Simulation(sky130_nfet_pair(mim, index))
            .run(OperatingPoint())
            .values[column]
        )
        assert native[index] == pytest.approx(expected, rel=1e-10, abs=1e-12)

    large = simulation.batch(
        {"op": OperatingPoint()}, MonteCarlo(KIT_SAMPLES, seed=11), traces=["out"]
    )
    assert not large.failed.any()
    agrees(
        float(first_order.standard_deviation("out")),
        large.traces[0].values_of("out")[:, 0],
    )


def test_native_monte_carlo_matches_sky130_passives_drawn_and_mismatch(
    mim: Library,
) -> None:
    """Verifies native Monte Carlo on SKY130 MIM cap, nested poly, and generic poly.

    Each device matches `Library.drawn` sample-by-sample within `rel=1e-10, abs=1e-12`
    (Newton / linear-solve rounding and floating-point expression order), and nested
    `res_xhigh_po_0p69` exposes behavioral resistor mismatch variables agreeing with
    native Monte Carlo spread within five standard errors.
    """
    ac = Ac("V1", decade(10 * MHz, 1 * GHz, 2))
    mim_batch = Simulation(mim_low_pass(mim, None)).batch(
        {"ac": ac}, MonteCarlo(10, seed=9), traces=["out"]
    )
    assert not mim_batch.failed.any()
    mim_phasors = mim_batch.traces[0].values_of("out")
    for index in range(10):
        expected = Simulation(mim_low_pass(mim, index)).run(ac).voltage("out")
        assert np.abs(mim_phasors[index]) == pytest.approx(
            np.abs(expected), rel=1e-10, abs=1e-12
        )
        assert np.angle(mim_phasors[index]) == pytest.approx(
            np.angle(expected), rel=1e-10, abs=1e-12
        )

    xhigh_variables = mim.random_variables(XHIGH_PO, "R1")
    assert {variable.name for variable in xhigh_variables} == {
        "res_match#0",
        "rend#0",
    }
    xhigh_simulation = Simulation(sky130_xhigh_po_bench(mim.component(XHIGH_PO)))
    xhigh_mismatch = xhigh_simulation.run(
        Mismatch(["V1:current"], random_variables=xhigh_variables)
    )
    xhigh_batch = xhigh_simulation.batch(
        {"op": OperatingPoint()}, MonteCarlo(10, seed=17), traces=["V1:current"]
    )
    assert not xhigh_batch.failed.any()
    xhigh_currents = xhigh_batch.traces[0].values_of("V1:current")[:, 0]
    for index in range(10):
        expected_current = (
            Simulation(
                sky130_xhigh_po_bench(mim.drawn(XHIGH_PO, Sample(17, index), "R1"))
            )
            .run(OperatingPoint())
            .current("V1")
        )
        assert xhigh_currents[index] == pytest.approx(
            expected_current, rel=1e-10, abs=1e-12
        )
    xhigh_large = xhigh_simulation.batch(
        {"op": OperatingPoint()},
        MonteCarlo(KIT_SAMPLES, seed=17),
        traces=["V1:current"],
    )
    assert not xhigh_large.failed.any()
    agrees(
        float(xhigh_mismatch.standard_deviation("V1:current")),
        xhigh_large.traces[0].values_of("V1:current")[:, 0],
    )

    generic_simulation = Simulation(
        sky130_generic_po_bench(mim.component(GENERIC_PO, w=1, l=10))
    )
    generic_batch = generic_simulation.batch(
        {"op": OperatingPoint()}, MonteCarlo(10, seed=19), traces=["V1:current"]
    )
    assert not generic_batch.failed.any()
    generic_currents = generic_batch.traces[0].values_of("V1:current")[:, 0]
    for index in range(10):
        expected_current = (
            Simulation(
                sky130_generic_po_bench(
                    mim.drawn(GENERIC_PO, Sample(19, index), "R1", w=1, l=10)
                )
            )
            .run(OperatingPoint())
            .current("V1")
        )
        assert generic_currents[index] == pytest.approx(
            expected_current, rel=1e-10, abs=1e-12
        )


def test_native_monte_carlo_combines_sky130_and_toleranced_resistor(
    mim: Library,
) -> None:
    """Verifies a circuit mixing SKY130 BSIM4 and a toleranced resistor varies both.

    Operating-point drain voltage matches `Library.drawn` plus standalone tolerance
    draws sample-by-sample within `rel=1e-10, abs=1e-12` (Newton convergence rounding).
    """

    def mixed_bench(drawn_sample: Sample | None = None) -> Component:
        builder = Builder("mixed_sky130_bench")
        ground = builder.port("ground", Role.GROUND)
        supply = builder.place(devices.voltage_source(1.8 * V), name="VDD")
        gate = builder.place(devices.voltage_source(1.0 * V), name="VG")
        load = builder.place(devices.resistor(5 * kohm, tolerance=0.05), name="R1")
        mosfet_part = (
            mim.component(NFET, w=1.0, l=0.15)
            if drawn_sample is None
            else mim.drawn(NFET, drawn_sample, "M1", w=1.0, l=0.15)
        )
        mosfet = builder.place(mosfet_part, name="M1")
        builder.connect(supply.positive, load.a, name="vdd")
        builder.connect(load.b, mosfet.d, name="drain")
        builder.connect(gate.positive, mosfet.g, name="gate")
        builder.connect(mosfet.s, mosfet.b, supply.negative, gate.negative, ground)
        return builder.finish()

    nominal_built = circuit(mixed_bench())
    toleranced_load = toleranced(nominal_built)
    seed, count = 29, 12
    batch = Simulation(nominal_built).batch(
        {"op": OperatingPoint()}, MonteCarlo(count, seed=seed), traces=["drain"]
    )
    assert not batch.failed.any()
    assert "R1 resistance" in batch.parameters
    assert any(label.startswith("M1/") for label in batch.parameters)
    assert np.std(batch.parameter("R1 resistance")) > 0
    drain_voltages = batch.traces[0].values_of("drain")[:, 0]
    for index in range(count):
        drawn_load = sample(toleranced_load, seed, index)
        expected = (
            Simulation(mixed_bench(Sample(seed, index)), overrides=drawn_load)
            .run(OperatingPoint())
            .voltage("drain")
        )
        assert drain_voltages[index] == pytest.approx(expected, rel=1e-10, abs=1e-12)
