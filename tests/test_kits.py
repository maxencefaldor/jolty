# Copyright 2026 Maxence Faldor
"""Tests open process design kits across corners and against ngspice.

Each kit's pinned files are cached inside pytest's cache on first use. Devices
supported natively by jolty are simulated against ngspice at the same corner,
skipping when ngspice is unavailable.
"""

import math
import re
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np
import pytest
from pydantic import JsonValue

from differential import SKIP, closest, compare
from jolty import (
    Builder,
    Component,
    Role,
    Simulation,
    _core,
    circuit,
    component,
    devices,
    ir,
    results,
)
from jolty.plan import (
    TIGHT,
    Ac,
    DcSweep,
    Noise,
    OperatingPoint,
    Transient,
    decade,
    linear,
)
from jolty.settings import Settings
from jolty.spice import (
    gf180mcu,
    kits,
    ngspice,
    sg13g2,
    sky130,
)
from jolty.spice import (
    library as spice_library,
)
from jolty.spice.expressions import STATISTICAL
from jolty.spice.library import Library, Sample
from jolty.units import GHz, Hz, MHz, V, celsius, fF, kHz, kohm, ns, ps
from jolty.waveforms import Pulse

KITS: Final = (sky130.KIT, gf180mcu.KIT, sg13g2.KIT)
"""The open process kits."""

GUIDE: Final = Path(__file__).resolve().parents[1] / "docs" / "guide" / "foundry.md"
"""The guide's page whose tables state each kit's devices."""

AGREEMENT: Final = 2 * ngspice.RELATIVE_TOLERANCE
"""Relative agreement bound between jolty and ngspice (twice `reltol`).

Each engine's Newton iteration stops within one relative tolerance of the
solution, so the two solutions agree within twice that relative tolerance.
"""


@pytest.fixture(scope="session")
def kept(pytestconfig: pytest.Config) -> Settings:
    """Returns `Settings` pointing at pytest's persistent kit download cache."""
    assert pytestconfig.cache is not None
    settings = Settings(cache=pytestconfig.cache.mkdir("downloads"))
    for kit in KITS:
        kits.root(kit, settings)
    return settings


@pytest.mark.parametrize(
    ("kit", "section"),
    [(kit, section) for kit in KITS for section in kit.sections],
    ids=[f"{kit.name}-{name}" for kit in KITS for _, name in kit.sections],
)
def test_a_kit_reads_at_each_of_its_sections(
    kept: Settings, kit: kits.Kit, section: kits.Section
) -> None:
    """Verifies that every kit section parses finite parameters and devices."""
    library = kits.read(kit, (section,), settings=kept)
    values = library.values()
    assert library.subcircuits or library.models
    assert all(math.isfinite(value) for value in values.values())
    # Default SPICE nominal temperature 27 C (300.15 - 273.15); float rounding.
    assert values["temper"] == pytest.approx(27.0)


def test_sky130_scales_its_sizes_to_micrometres(kept: Settings) -> None:
    """Verifies that SKY130 sets the SPICE `scale` option to micrometers."""
    library = kits.read(sky130.KIT, settings=kept)
    assert library.options["scale"] == sky130.SCALE


def test_temper_is_the_temperature_a_kit_reads_at(kept: Settings) -> None:
    """Verifies that SG13G2 transit time evaluates `temper` at read temperature."""
    hot = kits.read(sg13g2.KIT, settings=kept, temperature=celsius(85).value)
    # Celsius round-trip (85 + 273.15 - 273.15); default pytest.approx covers rounding.
    assert hot.values()["temper"] == pytest.approx(85.0)
    times = [
        _parameters(library.component("npn13G2"), "vbic")["forward_transit_time"]
        for library in (kits.read(sg13g2.KIT, settings=kept), hot)
    ]
    # Card's `tf` is 2.08e-13 s * ((temper + 273) / 300) ** 0.7; f64 power rounding.
    expected = [2.08e-13 * ((temper + 273) / 300) ** 0.7 for temper in (27.0, 85.0)]
    assert times == pytest.approx(expected, rel=1e-12)


MISMATCH: Final = (sg13g2.DIRECTORY / "cornerHBT.lib", "hbt_typ_mismatch")
"""SG13G2 bipolar typical-corner section with statistical mismatch enabled."""


def test_a_samples_mismatch_is_the_same_bits_on_any_shard(kept: Settings) -> None:
    """Verifies that mismatch draws are bit-identical across separate kit loads.

    Distinct placements and sample indices draw independently, while nominal
    evaluation without a sample leaves statistical functions at nominal values.
    """
    first, second = (kits.read(sg13g2.KIT, (MISMATCH,), settings=kept) for _ in "ab")

    def drawn(library: Library, sample: Sample, path: str) -> Mapping[str, object]:
        return _parameters(library.drawn("npn13G2", sample, path, mm_ok=1), "vbic")

    sample = Sample(seed=7, index=3)
    assert drawn(first, sample, "X1") == drawn(second, sample, "X1")
    assert drawn(first, sample, "X1") != drawn(first, sample, "X2")
    assert drawn(first, sample, "X1") != drawn(first, Sample(7, 4), "X1")
    nominal = _parameters(first.component("npn13G2", mm_ok=1), "vbic")
    unmatched = _parameters(first.component("npn13G2"), "vbic")
    assert nominal == unmatched
    assert drawn(first, sample, "X1") != nominal


SAMPLES: Final = 4000
"""The samples whose draws the statistical functions' test takes."""


def test_statistical_functions_draw_as_ngspice_defines_them() -> None:
    """Verifies that SPICE statistical functions match expected means and spreads.

    Checks sample means and standard deviations within five standard errors over
    `SAMPLES` deterministic draws.
    """
    calls = {
        "agauss": (2.0, 0.3, 3.0),
        "gauss": (2.0, 0.15, 3.0),
        "aunif": (2.0, 0.1),
        "unif": (2.0, 0.05),
        "limit": (2.0, 0.1),
    }
    assert calls.keys() == STATISTICAL.keys()
    deviation = {
        "agauss": 0.1,
        "gauss": 0.1,
        "aunif": 0.1 / 3**0.5,
        "unif": 0.1 / 3**0.5,
    }
    for function, arguments in calls.items():
        values = np.array(
            [
                Sample(11, index).draw("X1", "x")(function, arguments)
                for index in range(SAMPLES)
            ]
        )
        spread = deviation.get(function, 0.1)
        # 5 standard errors of mean (sigma/sqrt(N)) and std (sqrt(2/N)) over SAMPLES.
        assert abs(values.mean() - 2.0) < 5 * spread / SAMPLES**0.5
        assert abs(values.std() / spread - 1) < 5 * (2 / SAMPLES) ** 0.5
        if function == "limit":
            assert set(np.round(values, 12)) == {1.9, 2.1}
    assert Sample(11, 0).draw("X1", "x")("agauss", (2.0, 0.3, 0.0)) == 2.0


def _parameters(placed: Component, model: str) -> Mapping[str, object]:
    """Returns parameters of the single device of `model` inside `placed`."""
    (found,) = (
        instance.parameters
        for instance in placed.definition.instances
        if instance.of == ir.DeviceOf(model=model)
    )
    return found


# Kit device inventory compared against the foundry guide tables.


STATUSES: Final = {
    "native": "native",
    "verilog-a": "Verilog-A",
    "neither": "neither",
    "unread": "unread",
    "unvalued": "unvalued",
}
"""Display labels used for kit inventory statuses in the foundry guide."""


PROBLEM: Final = re.compile(r"^(?:[^:]+:\d+: )?(?P<problem>.*?)(?:: `[^`]*`)?$")
"""Pattern extracting the core diagnostic text without location or statement."""


def table(devices: tuple[kits.Device, ...]) -> list[str]:
    """Formats Markdown table rows grouping kit devices by status and models.

    Each row lists the status, required models, first diagnostic message
    (omitting source location and statement, prioritizing model issues), and
    defined devices in declaration order.
    """
    groups: dict[tuple[str, tuple[str, ...]], list[kits.Device]] = {}
    for device in devices:
        groups.setdefault((device.status, device.models), []).append(device)
    rows = [
        "| Status | Models | First problem | Devices |",
        "| --- | --- | --- | --- |",
    ]
    for (status, models), grouped in groups.items():
        problems = sorted(
            (use for use in grouped[0].uses if use.problem is not None),
            key=lambda use: use.undefined is not None or not use.model,
        )
        found = PROBLEM.fullmatch(problems[0].problem or "") if problems else None
        problem = found["problem"] if found else ""
        names = ", ".join(
            f"`{device.name}`" + (" (model)" if device.kind == "model" else "")
            for device in grouped
        )
        rows.append(
            f"| {STATUSES[status]} | {'; '.join(models) or 'none'} | {problem} | "
            f"{len(grouped)}: {names} |"
        )
    return rows


@pytest.mark.parametrize("kit", KITS, ids=[kit.name for kit in KITS])
def test_the_guide_states_each_kits_devices(kept: Settings, kit: kits.Kit) -> None:
    """Verifies that the foundry guide table matches the typical-corner inventory."""
    text = GUIDE.read_text(encoding="utf-8")
    marker = f"<!-- devices of: {kit.name} -->"
    assert marker in text, f"{GUIDE.name} has no `{marker}`"
    written = text.split(marker, 1)[1].strip().splitlines()
    rows = table(kits.inventory(kit, settings=kept))
    assert written[: len(rows)] == rows
    assert len(written) == len(rows) or not written[len(rows)].startswith("|")


# Native kit devices compared against ngspice reading the same kit.


@component
def gummel(c: Builder, part: Component) -> None:
    """Builds a Gummel bench with base source `VB` and collector source `VC`."""
    ground = c.port("ground", Role.GROUND)
    base = c.place(devices.voltage_source(0.8 * V), name="VB")
    collector = c.place(devices.voltage_source(1.5 * V), name="VC")
    placed = c.place(part, name="X1")
    c.connect(base.positive, placed["b"], name="base")
    c.connect(collector.positive, placed["c"], name="collector")
    c.connect(
        base.negative,
        collector.negative,
        placed["e"],
        placed["bn"],
        placed["ground"],
        ground,
    )


GUMMEL: Final = ["* gummel", "VB base 0 0.8", "VC collector 0 1.5"]
"""The deck of `gummel`, but its transistor."""

HBTS: Final = ("hbt_typ", "hbt_bcs", "hbt_wcs")
"""SG13G2's corners of its bipolar transistors: typical, best and worst case."""


@SKIP
@pytest.mark.parametrize("corner", HBTS)
def test_an_sg13g2_bipolar_transistor_agrees_with_ngspice(
    kept: Settings, corner: str
) -> None:
    """Verifies IHP `npn13G2` self-heating VBIC DC sweep agreement with ngspice.

    Sweeps the base from 0.4 V to 1 V (heating the device by 60 to 130 K) and
    compares terminal currents, substrate voltage, and thermal node at matching
    kelvins. Runs at the nominal 27 °C card temperature where ngspice maps
    heated VBIC parameters identically.
    """
    section = (sg13g2.DIRECTORY / "cornerHBT.lib", corner)
    library = kits.read(sg13g2.KIT, (section,), settings=kept)
    simulation = Simulation(gummel(library.component("npn13G2")))
    sweep = simulation.run(DcSweep("VB", linear(0.4 * V, 1.0 * V, 31)), TIGHT)
    root = kits.root(sg13g2.KIT, kept)
    raw = ngspice.run(
        [*GUMMEL, "X1 collector base 0 0 npn13G2"],
        "dc VB 0.4 1.0 0.02",
        sections=((root / section[0], corner),),
        temperature=ngspice.same_kelvins(celsius(27).value),
    )
    vectors = {
        results.Unknown("VB:current", "current"): "i(vb)",
        results.Unknown("VC:current", "current"): "i(vc)",
        results.Unknown("X1/s1", "voltage"): "v(x1.s1)",
        results.Unknown("X1/t", "temperature"): "v(x1.t)",
    }
    for unknown, name in vectors.items():
        spice = raw.real(name)
        found = sweep.values[:, simulation.unknowns.index(unknown)]
        compare(
            found, spice, unknown, scale=float(np.abs(spice).max()), tolerance=AGREEMENT
        )
    rise = raw.real("v(x1.t)").max()
    assert rise > 50.0  # The bench heats: no vacuous agreement.


@SKIP
def test_an_sg13g2_transistor_reads_its_transit_time_where_it_runs(
    kept: Settings,
) -> None:
    """Verifies SG13G2 AC gain to 10 GHz at 85 °C with `temper`-evaluated `tf`.

    Reading the library at 27 °C instead shifts the gain by more than 1000
    tolerances. Self-heating is disabled (`selft=0`) because ngspice maps
    heated VBIC parameters from ambient temperature away from the card nominal.
    """
    section = (sg13g2.DIRECTORY / "cornerHBT.lib", "hbt_typ")
    hot = celsius(85)
    gains = {}
    for read in (hot, celsius(27)):
        library = kits.read(sg13g2.KIT, (section,), read.value, kept)
        cool = library.component("npn13G2", selft=0)
        simulation = Simulation(gummel(cool), temperature=hot)
        response = simulation.run(Ac("VB", decade(1 * MHz, 10 * GHz, 1)), TIGHT)
        gains[read.value] = response.values[
            :, simulation.unknowns.index(results.Unknown("VC:current", "current"))
        ]
    raw = ngspice.runs(
        [
            "* gain",
            "VB base 0 0.8 ac 1",
            "VC collector 0 1.5",
            "X1 collector base 0 0 npn13G2 selft=0",
        ],
        ("ac dec 1 1meg 10g",),
        sections=((kits.root(sg13g2.KIT, kept) / section[0], "hbt_typ"),),
        temperature=ngspice.same_kelvins(hot.value),
    )[0]
    spice = raw.vectors["i(vc)"]
    unknown = results.Unknown("VC:current", "current")
    scale = float(np.abs(spice).max())
    compare(gains[hot.value], spice, unknown, scale=scale, tolerance=AGREEMENT)
    away = np.abs(gains[celsius(27).value] - spice).max() / scale
    assert away > 1000 * AGREEMENT


BIPOLARS: Final = (
    (sky130.KIT, "sky130_fd_pr__npn_05v5_W1p00L1p00", 1.0),
    (sky130.KIT, "sky130_fd_pr__pnp_05v5_W0p68L0p68", -1.0),
    (gf180mcu.KIT, "vnpn_10x10", 1.0),
    (gf180mcu.KIT, "vpnp_0p42x10", -1.0),
    (sg13g2.KIT, "pnpMPA", -1.0),
)
"""Gummel-Poon BJTs `(kit, name, sign)` spanning SKY130, GF180MCU, and SG13G2.

Covers SKY130 (`TLEV=0`, `TLEVC=0`, `TREF=30 °C`, HSPICE temperature terms),
GF180MCU (`TLEVC=1`, `NKF=0.4`, `BF`/`IKF` coefficients, `TREF=25 °C`), and
SG13G2 (`TRM1` and `TRC1`).
"""

BIPOLAR_TEMPERATURES: Final = (-40.0, 27.0, 125.0)
"""Temperatures in degrees Celsius covering kit range endpoints and nominal."""

CONSTANTS: Final = 1 - ngspice.same_kelvins(1.0)
"""Relative CODATA 2014 vs 2019 SI `k_B/q` offset (`3.4e-7`) vs ngspice.

At matching kelvins, ngspice's thermal voltage sits `3.4e-7` below jolty's, so
junction exponents (`V / Vt` and `(T/Tn - 1) * Eg / Vt`) differ by `CONSTANTS`
times their sum.
"""

SPICE_THERMAL_VOLTAGE: Final = ("Gummel-Poon",)
"""Device models computing thermal voltage with each engine's physical constants.

At matching kelvins, junction currents diverge by `CONSTANTS` times their
exponents, bounded by `junction_tolerance`.
"""


def junction_tolerance(voltage: float, temperature: float) -> float:
    """Returns relative current tolerance vs ngspice at `voltage` and `temperature`.

    Adds `AGREEMENT` to `CONSTANTS` times twice the junction voltage exponent
    `voltage / Vt`. For kit bandgaps (1.17 to 1.19 V) and `TREF` (25 to 30 °C),
    the activation term `Eg * |T/Tn - 1|` stays below 0.40 V across -40 to
    125 °C, keeping total exponent shift within `2 * voltage / Vt`.
    """
    thermal_voltage = temperature * _core.BOLTZMANN / _core.ELEMENTARY_CHARGE
    return AGREEMENT + CONSTANTS * 2 * voltage / thermal_voltage


@SKIP
@pytest.mark.parametrize(
    ("kit", "name", "sign"), BIPOLARS, ids=[b[1] for b in BIPOLARS]
)
def test_a_kits_bipolar_transistor_follows_the_temperature_as_in_ngspice(
    kept: Settings, kit: kits.Kit, name: str, sign: float
) -> None:
    """Verifies kit Gummel-Poon temperature tracking against ngspice across corners.

    Holds the collector at 1.5 V, sweeps the base from 0.4 V to 0.9 V across
    `BIPOLAR_TEMPERATURES` (signed by polarity), and checks collector and base
    currents within `junction_tolerance` at 0.9 V.
    """
    library = kits.read(kit, settings=kept)
    part = library.component(name)
    collector, base, *grounded = part.definition.ports

    @component
    def gummel_at(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        drive = c.place(devices.voltage_source(sign * 0.4 * V), name="VB")
        held = c.place(devices.voltage_source(sign * 1.5 * V), name="VC")
        placed = c.place(part, name="X1")
        c.connect(drive.positive, placed[base.name], name="base")
        c.connect(held.positive, placed[collector.name], name="collector")
        c.connect(
            drive.negative,
            held.negative,
            *(placed[port.name] for port in grounded),
            ground,
        )

    pins = " 0" * sum(port.role == Role.SIGNAL for port in grounded)
    deck = [
        "* gummel",
        *NOMINAL.get(kit.name, ()),
        f"VB base 0 {sign * 0.4!r}",
        f"VC collector 0 {sign * 1.5!r}",
        f"X1 collector base{pins} {name}",
    ]
    root = kits.root(kit, kept)
    for degrees in BIPOLAR_TEMPERATURES:
        temperature = celsius(degrees)
        simulation = Simulation(gummel_at(), temperature=temperature)
        swept = linear(sign * 0.4 * V, sign * 0.9 * V, 26)
        sweep = simulation.run(DcSweep("VB", swept), TIGHT)
        raw = ngspice.run(
            deck,
            f"dc VB {sign * 0.4!r} {sign * 0.9!r} {sign * 0.02!r}",
            libraries=tuple(root / path for path in kit.includes),
            sections=tuple((root / path, section) for path, section in kit.typical),
            compatibility=kit.compatibility,
            temperature=ngspice.same_kelvins(temperature.value),
        )
        for source in ("VB", "VC"):
            unknown = results.Unknown(f"{source}:current", "current")
            spice = raw.real(f"i({source.casefold()})")
            found = sweep.values[:, simulation.unknowns.index(unknown)]
            scale = float(np.abs(spice).max())
            tolerance = junction_tolerance(0.9, temperature.value)
            # Derived thermal-voltage tolerance covering CODATA kT/q and diode stepping.
            compare(found, spice, unknown, scale=scale, tolerance=tolerance)


SKY130_CAPACITORS: Final = ("tt", "ll", "hh")
"""SKY130's corners of its capacitors: typical, low and high."""


@SKIP
@pytest.mark.parametrize("corner", SKY130_CAPACITORS)
def test_a_sky130_mim_capacitor_agrees_with_ngspice(
    kept: Settings, corner: str
) -> None:
    """Verifies SKY130 10 µm MIM capacitor AC admittance against ngspice at 85 °C.

    Compares admittance from 1 MHz to 100 GHz across corners, exercising
    temperature-drifting plate and via series resistances.
    """
    library = kits.read(
        sky130.KIT,
        ((sky130.LIBRARY, corner),),
        celsius(85).value,
        kept,
    )

    @component
    def bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(0 * V), name="V1")
        placed = c.place(
            library.component("sky130_fd_pr__cap_mim_m3_1", w=10, l=10), name="X1"
        )
        c.connect(source.positive, placed["c0"], name="plate")
        c.connect(source.negative, placed["c1"], ground)

    simulation = Simulation(bench(), temperature=celsius(85))
    response = simulation.run(Ac("V1", decade(1 * MHz, 100 * GHz, 2)), TIGHT)
    raw = ngspice.runs(
        [
            "* mim",
            "V1 plate 0 0 ac 1",
            "X1 plate 0 sky130_fd_pr__cap_mim_m3_1 w=10 l=10",
        ],
        ("ac dec 2 1meg 100g",),
        sections=((kits.root(sky130.KIT, kept) / sky130.LIBRARY, corner),),
        compatibility=sky130.KIT.compatibility,
        temperature=celsius(85).value,
    )[0]
    unknown = results.Unknown("V1:current", "current")
    spice = raw.vectors["i(v1)"]
    found = response.values[:, simulation.unknowns.index(unknown)]
    compare(
        found, spice, unknown, scale=float(np.abs(spice).max()), tolerance=AGREEMENT
    )


SKY130_MOSFET_CORNERS: Final = ("tt", "ff", "ss")
"""SKY130's corners of its MOSFETs: typical, and fast and slow in both channels."""

SKY130_SIZES: Final = tuple(
    (width, length)
    for width in (0.42, 0.55, 0.7, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0)
    for length in (0.15, 0.2, 0.3, 0.75, 1.5, 3.0, 6.0, 10.0, 50.0)
)
"""Widths and lengths, in micrometers as SKY130's netlists write them, one in each of
the nine lengths' bins of `nfet_01v8` and `pfet_01v8` and in nine of their widths'."""

SUPPLY: Final = 1.8
"""The volts SKY130's 1.8 V transistors are biased up to."""


@SKIP
@pytest.mark.parametrize("corner", SKY130_MOSFET_CORNERS)
@pytest.mark.parametrize("channel", ["n", "p"])
def test_sky130_transistors_agree_with_ngspice_across_their_bins(
    kept: Settings, corner: str, channel: str
) -> None:
    """Verifies SKY130 BSIM4 4.5 `nfet_01v8` and `pfet_01v8` DC sweeps across bins.

    Places one transistor per `(width, length)` bin in `SKY130_SIZES` with
    grounded source and body, sensing each drain current through a 0 V source.
    Sweeps `VD` and `VG` to the supply and compares all bins against ngspice.
    """
    name = f"sky130_fd_pr__{channel}fet_01v8"
    sign = 1.0 if channel == "n" else -1.0
    library = kits.read(sky130.KIT, ((sky130.LIBRARY, corner),), settings=kept)

    @component
    def curves(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        drain = c.place(devices.voltage_source(sign * SUPPLY * V), name="VD")
        gate = c.place(devices.voltage_source(sign * SUPPLY * V), name="VG")
        c.connect(drain.negative, gate.negative, ground)
        for number, (width, length) in enumerate(SKY130_SIZES):
            sense = c.place(devices.voltage_source(0 * V), name=f"A{number}")
            placed = c.place(
                library.component(name, w=width, l=length), name=f"X{number}"
            )
            c.connect(sense.positive, drain.positive)
            c.connect(sense.negative, placed["d"], name=f"n{number}")
            c.connect(placed["g"], gate.positive)
            c.connect(placed["s"], placed["b"], ground)

    simulation = Simulation(curves())
    deck = [
        f"* {name}",
        f"VD drain 0 {sign * SUPPLY!r}",
        f"VG gate 0 {sign * SUPPLY!r}",
    ]
    for number, (width, length) in enumerate(SKY130_SIZES):
        deck.append(f"VA{number} drain n{number} 0")
        deck.append(f"X{number} n{number} gate 0 0 {name} w={width!r} l={length!r}")
    swept = linear(0 * V, sign * SUPPLY * V, 10)
    sweeps = [simulation.run(DcSweep(source, swept), TIGHT) for source in ("VD", "VG")]
    step = sign * SUPPLY / 9
    raws = ngspice.runs(
        deck,
        tuple(f"dc {source} 0 {sign * SUPPLY!r} {step!r}" for source in ("VD", "VG")),
        sections=((kits.root(sky130.KIT, kept) / sky130.LIBRARY, corner),),
        compatibility=sky130.KIT.compatibility,
        temperature=ngspice.same_kelvins(celsius(27).value),
    )
    for sweep, raw in zip(sweeps, raws, strict=True):
        for number in range(len(SKY130_SIZES)):
            unknown = results.Unknown(f"A{number}:current", "current")
            spice = raw.real(f"i(va{number})")
            found = sweep.values[:, simulation.unknowns.index(unknown)]
            scale = float(np.abs(spice).max())
            compare(found, spice, unknown, scale=scale, tolerance=AGREEMENT)


@SKIP
@pytest.mark.parametrize("channel", ["n", "p"])
def test_sky130_transistors_admittances_agree_with_ngspice(
    kept: Settings, channel: str
) -> None:
    """Verifies SKY130 `nfet_01v8` and `pfet_01v8` AC admittances to 10 GHz.

    Biases a 1 µm by 0.15 µm device at half-supply with body tied to source,
    drives gate and drain in turn, and compares drain, gate, and body currents
    against ngspice at `tt`. Although ngspice body capacitances depart from
    charge derivatives (`AC_DEPARTED`), the 50 Ω body resistance couples only
    `ωRC ≈ 0.6%` at 10 GHz, keeping current differences within `AGREEMENT`.
    """
    name = f"sky130_fd_pr__{channel}fet_01v8"
    sign = 1.0 if channel == "n" else -1.0
    library = kits.read(sky130.KIT, settings=kept)
    half = sign * SUPPLY / 2

    @component
    def biased(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        drain = c.place(devices.voltage_source(half * V), name="VD")
        gate = c.place(devices.voltage_source(half * V), name="VG")
        body = c.place(devices.voltage_source(0 * V), name="VB")
        placed = c.place(library.component(name, w=1, l=0.15), name="X1")
        c.connect(drain.positive, placed["d"], name="drain")
        c.connect(gate.positive, placed["g"], name="gate")
        c.connect(body.positive, placed["b"], name="body")
        c.connect(drain.negative, gate.negative, body.negative, placed["s"], ground)

    simulation = Simulation(biased())
    deck = [
        "* admittances",
        f"VD drain 0 {half!r}",
        f"VG gate 0 {half!r}",
        "VB body 0 0",
        f"X1 drain gate 0 body {name} w=1 l=0.15",
    ]
    for driven in ("VD", "VG"):
        response = simulation.run(Ac(driven, decade(1 * MHz, 10 * GHz, 2)), TIGHT)
        excited = [
            f"{line} ac 1" if line.startswith(f"{driven} ") else line for line in deck
        ]
        raw = ngspice.run(
            excited,
            "ac dec 2 1meg 10g",
            sections=((kits.root(sky130.KIT, kept) / sky130.LIBRARY, "tt"),),
            compatibility=sky130.KIT.compatibility,
            temperature=ngspice.same_kelvins(celsius(27).value),
        )
        for source in ("VD", "VG", "VB"):
            unknown = results.Unknown(f"{source}:current", "current")
            spice = raw.vectors[f"i({source.lower()})"]
            found = response.values[:, simulation.unknowns.index(unknown)]
            scale = float(np.abs(spice).max())
            compare(found, spice, unknown, scale=scale, tolerance=AGREEMENT)


RF_TRANSISTORS: Final = (
    "sky130_fd_pr__rf_nfet_01v8_bM02W1p65L0p15",
    "sky130_fd_pr__rf_pfet_01v8_bM04W3p00L0p18",
    "sky130_fd_pr__rf_nfet_01v8_lvt_bM04W5p00L0p25",
    "sky130_fd_pr__rf_nfet_g5v0d10v5_bM04W5p00L0p50",
)
"""SKY130 RF BSIM4 4.5 transistors under test (`rgatemod=3`, `rdsmod=1`, `rbodymod=1`).

Covers 1.8 V n-channel and p-channel, 1.8 V low-threshold n-channel, and 5 V
n-channel cells laid out alongside dummy devices.
"""

RF_OUTPUT_DEPARTED: Final = frozenset({"sky130_fd_pr__rf_pfet_01v8_bM04W3p00L0p18"})
"""RF transistors whose ngspice output conductance differs from `dId/dVd` by 2.1e-4.

Measured against 0.1 mV central differences of drain current, whereas jolty
matches the exact derivative (documented in the foundry departures guide).
"""

RF_REVERSE: Final = 0.5
"""Reverse body-to-source bias in volts minimizing ngspice body-capacitance drift."""


@SKIP
@pytest.mark.parametrize("name", RF_TRANSISTORS)
def test_sky130_rf_transistors_admittances_agree_with_ngspice(
    kept: Settings, name: str
) -> None:
    """Verifies SKY130 RF BSIM4 operating points and AC admittances to 10 GHz.

    Biases gate and drain at half-supply with body reversed by `RF_REVERSE`,
    then compares operating-point currents (scaled by drain current) and AC
    currents from 1 MHz to 10 GHz against ngspice at `tt`. Picoampere body
    leakage lies below the 1 nV / kilohm resolution floor of both engines.

    With `rbodymod=1`, ngspice body-capacitance departures (`3.4e-5` under
    reverse bias) perturb currents by less than `1e-6` up to 10 GHz, staying
    within `AGREEMENT` (growing quadratically to `3.3e-6` at 30 GHz and
    `8.4e-6` at 100 GHz). Drain-driven drain current is skipped for
    `RF_OUTPUT_DEPARTED` where ngspice's output conductance is non-exact.
    """
    sign = -1.0 if "pfet" in name else 1.0
    library = kits.read(sky130.KIT, settings=kept)
    half = sign * SUPPLY / 2
    body = -sign * RF_REVERSE

    @component
    def biased(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        drain = c.place(devices.voltage_source(half * V), name="VD")
        gate = c.place(devices.voltage_source(half * V), name="VG")
        bulk = c.place(devices.voltage_source(body * V), name="VB")
        placed = c.place(library.component(name), name="X1")
        c.connect(drain.positive, placed["d"], name="drain")
        c.connect(gate.positive, placed["g"], name="gate")
        c.connect(bulk.positive, placed["b"], name="body")
        c.connect(drain.negative, gate.negative, bulk.negative, placed["s"], ground)

    simulation = Simulation(biased())
    deck = [
        "* rf admittances",
        f"VD drain 0 {half!r}",
        f"VG gate 0 {half!r}",
        f"VB body 0 {body!r}",
        f"X1 drain gate 0 body {name}",
    ]
    sections = ((kits.root(sky130.KIT, kept) / sky130.LIBRARY, "tt"),)
    kelvins = ngspice.same_kelvins(celsius(27).value)
    sources = ("VD", "VG", "VB")
    point = simulation.run(OperatingPoint(), TIGHT)
    raw = ngspice.run(
        deck,
        "op",
        sections=sections,
        compatibility=sky130.KIT.compatibility,
        temperature=kelvins,
    )
    drain = float(np.abs(raw.real("i(vd)")).max())
    for source in sources:
        unknown = results.Unknown(f"{source}:current", "current")
        spice = raw.real(f"i({source.lower()})")
        found = point.values[[simulation.unknowns.index(unknown)]]
        compare(found, spice, unknown, scale=drain, tolerance=AGREEMENT)
    for driven in ("VD", "VG"):
        response = simulation.run(Ac(driven, decade(1 * MHz, 10 * GHz, 2)), TIGHT)
        excited = [
            f"{line} ac 1" if line.startswith(f"{driven} ") else line for line in deck
        ]
        raw = ngspice.run(
            excited,
            "ac dec 2 1meg 10g",
            sections=sections,
            compatibility=sky130.KIT.compatibility,
            temperature=kelvins,
        )
        for source in sources:
            if name in RF_OUTPUT_DEPARTED and driven == source == "VD":
                continue
            unknown = results.Unknown(f"{source}:current", "current")
            spice = raw.vectors[f"i({source.lower()})"]
            found = response.values[:, simulation.unknowns.index(unknown)]
            scale = float(np.abs(spice).max())
            compare(found, spice, unknown, scale=scale, tolerance=AGREEMENT)


@SKIP
@pytest.mark.parametrize(
    "name", [name for name in RF_TRANSISTORS if name not in RF_OUTPUT_DEPARTED]
)
def test_a_sky130_rf_transistor_makes_the_noise_of_ngspices(
    kept: Settings, name: str
) -> None:
    """Verifies SKY130 RF common-source output noise against ngspice up to 100 MHz.

    Compares holistic channel thermal noise, flicker noise, and drain, source,
    body, and gate electrode thermal noise within `AGREEMENT`. Above 100 MHz,
    ngspice body-capacitance departures shift the density by up to `4.4e-6` at
    10 GHz; `RF_OUTPUT_DEPARTED` is excluded due to a `1e-5` stage gain shift.
    """
    library = kits.read(sky130.KIT, settings=kept)
    half = SUPPLY / 2

    @component
    def stage(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        drain = c.place(devices.voltage_source(SUPPLY * V), name="VDD")
        drive = c.place(devices.voltage_source(half * V), name="V1")
        load = c.place(devices.resistor(1 * kohm), name="RL")
        placed = c.place(library.component(name), name="X1")
        c.connect(drain.positive, load.a, name="vdd")
        c.connect(load.b, placed["d"], name="out")
        c.connect(drive.positive, placed["g"], name="in")
        c.connect(drain.negative, drive.negative, placed["s"], placed["b"], ground)

    simulation = Simulation(stage())
    spectrum = simulation.run(
        Noise("out", decade(1 * Hz, 100 * MHz, 2), source="V1"), TIGHT
    )
    raw = ngspice.run(
        [
            "* rf noise",
            f"VDD vdd 0 {SUPPLY!r}",
            f"V1 in 0 {half!r} ac 1",
            "RL vdd out 1k",
            f"X1 out in 0 0 {name}",
        ],
        "noise v(out) V1 dec 2 1 100meg\nsetplot noise1",
        sections=((kits.root(sky130.KIT, kept) / sky130.LIBRARY, "tt"),),
        compatibility=sky130.KIT.compatibility,
        temperature=ngspice.same_kelvins(celsius(27).value),
    )
    np.testing.assert_allclose(
        np.sqrt(spectrum.densities), raw.real("onoise_spectrum"), rtol=AGREEMENT
    )


NOISY: Final = {
    "sky130-n": (sky130.KIT, "sky130_fd_pr__nfet_01v8", "tt", 1.8, 0.8, (1, 0.15)),
    "sky130-p": (sky130.KIT, "sky130_fd_pr__pfet_01v8", "tt", -1.8, -0.9, (2, 0.5)),
    "gf180mcu-n": (gf180mcu.KIT, "nmos_3p3", "typical", 3.3, 1.2, (1e-6, 5e-7)),
    "gf180mcu-p": (gf180mcu.KIT, "pmos_3p3", "typical", -3.3, -1.4, (2e-6, 5e-7)),
}
"""Kit BSIM4 transistors for noise parity `(kit, model, corner, Vdd, Vg, (w, l))`.

Covers SKY130 holistic thermal noise (`tnoimod=1`), GF180MCU charge-based
thermal noise (`tnoimod=0`), and BSIM4 flicker noise (`fnoimod=1`).
"""


@SKIP
@pytest.mark.parametrize("name", NOISY)
def test_a_kits_transistor_makes_the_noise_of_ngspices(
    kept: Settings, name: str
) -> None:
    """Verifies kit BSIM4 common-source output noise against ngspice up to 1 GHz.

    Compares channel thermal, flicker, and resistor noise within `AGREEMENT`.
    The operating point agrees within `AGREEMENT`, and the Boltzmann constant
    offset (`1.38064852e-23` J/K in ngspice) shifts densities by `3.4e-7`.
    """
    kit, model, corner, supply, gate, (width, length) = NOISY[name]
    if kit is sky130.KIT:
        sections = ((sky130.LIBRARY, corner),)
    else:
        library_path = gf180mcu.DIRECTORY / "sm141064.ngspice"
        # The corner's section in place of the transistors' typical one.
        sections = (
            *(each for each in kit.typical if each != (library_path, "typical")),
            (library_path, corner),
        )
    library = kits.read(kit, sections, settings=kept)

    @component
    def stage(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        drain = c.place(devices.voltage_source(supply * V), name="VDD")
        drive = c.place(devices.voltage_source(gate * V), name="V1")
        load = c.place(devices.resistor(5 * kohm), name="RL")
        placed = c.place(library.component(model, w=width, l=length), name="X1")
        c.connect(drain.positive, load.a, name="vdd")
        c.connect(load.b, placed["d"], name="out")
        c.connect(drive.positive, placed["g"], name="in")
        c.connect(drain.negative, drive.negative, placed["s"], placed["b"], ground)

    simulation = Simulation(stage())
    spectrum = simulation.run(
        Noise("out", decade(1 * Hz, 1 * GHz, 2), source="V1"), TIGHT
    )
    root = kits.root(kit, kept)
    raw = ngspice.run(
        [
            "* noise",
            *NOMINAL.get(kit.name, ()),
            f"VDD vdd 0 {supply!r}",
            f"V1 in 0 {gate!r} ac 1",
            "RL vdd out 5k",
            f"X1 out in 0 0 {model} w={width!r} l={length!r}",
        ],
        "noise v(out) V1 dec 2 1 1g\nsetplot noise1",
        libraries=tuple(root / path for path in kit.includes),
        sections=tuple((root / path, section) for path, section in sections),
        compatibility=kit.compatibility,
        temperature=ngspice.same_kelvins(celsius(27).value),
    )
    np.testing.assert_allclose(
        np.sqrt(spectrum.densities), raw.real("onoise_spectrum"), rtol=AGREEMENT
    )


@SKIP
@pytest.mark.parametrize("corner", SKY130_MOSFET_CORNERS)
def test_a_sky130_inverter_switches_as_in_ngspice(kept: Settings, corner: str) -> None:
    """Verifies SKY130 CMOS inverter transient switching against ngspice.

    Drives a `nfet_01v8`/`pfet_01v8` inverter with a 50 ps rail-to-rail pulse
    into a 5 fF load and compares two switching periods against ngspice within
    the standard transient tolerance (`test_ngspice.test_transients_agree`).
    """
    library = kits.read(sky130.KIT, ((sky130.LIBRARY, corner),), settings=kept)
    pulse = Pulse(
        initial=0 * V,
        pulsed=SUPPLY * V,
        rise=50 * ps,
        fall=50 * ps,
        width=1 * ns,
        delay=0.5 * ns,
        period=2 * ns,
    )

    @component
    def inverter(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        supply = c.place(devices.voltage_source(SUPPLY * V), name="VDD")
        drive = c.place(devices.voltage_source(pulse), name="VIN")
        lower = c.place(
            library.component("sky130_fd_pr__nfet_01v8", w=1, l=0.15), name="XN"
        )
        upper = c.place(
            library.component("sky130_fd_pr__pfet_01v8", w=2, l=0.15), name="XP"
        )
        load = c.place(devices.capacitor(5 * fF), name="CL")
        c.connect(supply.negative, drive.negative, lower["s"], lower["b"], ground)
        c.connect(load.b, ground)
        c.connect(supply.positive, upper["s"], upper["b"], name="vdd")
        c.connect(drive.positive, lower["g"], upper["g"], name="in")
        c.connect(lower["d"], upper["d"], load.a, name="out")

    simulation = Simulation(inverter())
    stop, step = 4 * ns, 5 * ps
    waveforms = simulation.run(Transient(stop=stop, max_step=step), TIGHT)
    raw = ngspice.run(
        [
            "* inverter",
            f"VDD vdd 0 {SUPPLY!r}",
            f"VIN in 0 PULSE(0 {SUPPLY!r} 0.5n 50p 50p 1n 2n)",
            "XN out in 0 0 sky130_fd_pr__nfet_01v8 w=1 l=0.15",
            "XP out in vdd vdd sky130_fd_pr__pfet_01v8 w=2 l=0.15",
            "CL out 0 5f",
        ],
        f"tran {step.value!r} {stop.value!r} 0 {step.value!r}",
        sections=((kits.root(sky130.KIT, kept) / sky130.LIBRARY, corner),),
        compatibility=sky130.KIT.compatibility,
        temperature=ngspice.same_kelvins(celsius(27).value),
    )
    unknown = results.Unknown("out", "voltage")
    spice = raw.real("v(out)")
    found = closest(
        waveforms.times,
        waveforms.values[:, simulation.unknowns.index(unknown)],
        raw.real("time"),
        spice,
        step.value,
    )
    # Transient O(h^2) integration and window comparison vs ngspice (1e-3 of rail).
    compare(found, spice, unknown, scale=SUPPLY, tolerance=1e-3)


NOMINAL: Final = {gf180mcu.KIT.name: (".param sw_stat_global=0 sw_stat_mismatch=0",)}
"""SPICE parameter lines disabling kit-default statistical draws in ngspice.

GF180MCU's `design.ngspice` enables global and mismatch switches by default, so
disabling them aligns ngspice with nominal evaluation in jolty.
"""

AC_DEPARTED: Final = ("BSIM4", "resistor whose value reads a voltage")
"""Models whose AC response departs from ngspice at `DRIVE`.

In ngspice, BSIM4 body capacitances are not exact charge derivatives (worst
when a p-channel drain at `DRIVE` forward-biases the body). Behavioral `pow`
derivatives of a negative base to an even power also have the wrong sign
(`inpptree.c`, lines 613 to 632, affecting SKY130 `res_high_po`).
`test_sky130_transistors_admittances_agree_with_ngspice` checks BSIM4 with the
body tied to the source.
"""

DEPARTED: Final = {
    "pnpMPA": "its area, 2, by which ngspice multiplies its reverse saturation current "
    "twice",
    **dict.fromkeys(
        (
            "npolyf_u",
            "ppolyf_u",
            "npolyf_s",
            "ppolyf_s",
            "ppolyf_u_1k",
            "ppolyf_u_2k",
            "ppolyf_u_1k_6p0",
            "ppolyf_u_2k_6p0",
            "ppolyf_u_3k",
        ),
        "its body's resistance, which ngspice takes from its length, its line's first "
        "expression",
    ),
}
"""Native kit devices whose currents at `DRIVE` depart from ngspice.

SG13G2 `pnpMPA` forward-biases the base-collector junction at `DRIVE` (checked
by `test_a_kits_bipolar_transistor_follows_the_temperature_as_in_ngspice`).
GF180MCU poly resistors place length and width before resistance, causing
ngspice to parse the first braced expression (length in meters) as resistance
(`inpcom.c`, line 6841).
"""

LAGGED: Final = "capacitor whose value reads a voltage"
"""Inventory label for voltage-dependent capacitors delayed by `CAPACITANCE_LAG`."""

CAPACITANCE_LAG: Final = 1e-9 * 1e-6
"""Time constant in seconds by which ngspice 47 delays voltage-dependent capacitors.

ngspice differentiates voltage through 1e-9 Ω into 1 µF (`inpcom.c`, lines 6968
to 6977), yielding admittance `jωC / (1 + jωτ)` (offset by `6.3e-4` at 100 GHz).
"""


def lagged(part: Component) -> tuple[str, ...]:
    """Returns names of behavioral capacitors in `part` that ngspice delays.

    Each capacitor connects the first pin to another pin so the delayed current
    enters the first pin directly.
    """
    first, *rest = (port.name for port in part.definition.ports)
    names = []
    for instance in part.definition.instances:
        if instance.of == ir.DeviceOf(model="behavioral_capacitor"):
            assert instance.terminals["positive"] == first
            assert instance.terminals["negative"] in rest
            names.append(instance.name)
    return tuple(names)


DRIVE: Final = 0.5
"""Bias voltage applied to pin 0 of each native device (other pins grounded)."""

FLOOR: Final = 2 * ngspice.CURRENT_TOLERANCE.value
"""Absolute current agreement floor equal to twice ngspice `abstol`.

Each engine's Newton iteration stops within one absolute current tolerance, so
small currents agree within twice that floor regardless of magnitude.
"""

JUNCTION: Final = (1 - ngspice.same_kelvins(1.0)) * (
    DRIVE / (_core.BOLTZMANN * _core.REFERENCE_TEMPERATURE / _core.ELEMENTARY_CHARGE)
    + 1
)
"""Extra relative diode-current tolerance (`6.9e-6`) from CODATA `k_B/q` offset.

At matching kelvins (required for BSIM4 agreement), thermal voltage differs by
`3.4e-7` between jolty and ngspice. The junction exponent `v / (n*Vt)` reaches
at most `19.3` at 27 °C, while activation stays under `0.5` for `TREF` in 25 to
30 °C.
"""


def operating_point_departed(parameters: Mapping[str, JsonValue]) -> bool:
    """Returns whether a diode with `parameters` departs from ngspice at DC.

    A sidewall without a dedicated emission coefficient shares the junction
    emission coefficient in jolty, whereas ngspice scales sidewall saturation
    current with emission coefficient 1. At 27 °C, devices such as GF180MCU
    `nwp_3p3` (`TREF=25 °C`) therefore differ by `3.3e-4`.
    """
    nominal = parameters.get("nominal_temperature", _core.REFERENCE_TEMPERATURE)
    return (
        parameters.get("sidewall_saturation_current", 0.0) != 0.0
        and "sidewall_emission_coefficient" not in parameters
        and parameters.get("emission_coefficient", 1.0) != 1.0
        and nominal != _core.REFERENCE_TEMPERATURE
    )


def response_departed(parameters: Mapping[str, JsonValue]) -> bool:
    """Returns whether a diode with `parameters` departs from ngspice at AC.

    Departures occur whenever `operating_point_departed` holds or recombination
    is active (ngspice computes recombination conductance inexactly, shifting
    SG13G2 `dperim` susceptance by 1.4%). When `capacitance_temperature_model`
    is 1 with `TREF != 27 °C`, ngspice also shifts capacitances from 27 °C,
    moving SKY130 admittances by 0.3% to 1.4%.
    """
    nominal = parameters.get("nominal_temperature", _core.REFERENCE_TEMPERATURE)
    return (
        operating_point_departed(parameters)
        or "recombination_saturation_current" in parameters
        or (
            parameters.get("capacitance_temperature_model") == 1
            and nominal != _core.REFERENCE_TEMPERATURE
        )
    )


def diodes(part: Component) -> list[Mapping[str, JsonValue]]:
    """Returns parameter dictionaries of every diode nested inside `part`."""
    found = [
        instance.parameters
        for instance in part.definition.instances
        if instance.of == ir.DeviceOf(model="diode")
    ]
    return found + [each for child in part.children for each in diodes(child)]


@SKIP
@pytest.mark.parametrize("kit", KITS, ids=[kit.name for kit in KITS])
def test_every_native_device_agrees_with_ngspice(kept: Settings, kit: kits.Kit) -> None:
    """Verifies DC and AC agreement with ngspice for every native kit subcircuit.

    Places each inventory subcircuit with pin 0 driven at `DRIVE` through a
    dedicated 0 V sense source and remaining pins grounded. Compares DC and
    1 kHz to 100 GHz AC currents against ngspice at matching kelvins, skipping
    documented departures (`DEPARTED`, `AC_DEPARTED`, `operating_point_departed`,
    `response_departed`). Applies `JUNCTION` to diodes, `junction_tolerance` to
    Gummel-Poon BJTs, and `CAPACITANCE_LAG` compensation to `LAGGED` capacitors.
    """
    library = kits.read(kit, settings=kept)
    native = {
        device.name: device
        for device in kits.inventory(kit, settings=kept)
        if device.status == "native" and device.kind == "subcircuit"
    }
    driven = {
        name: part
        for name, device in native.items()
        if (part := library.component(name, **device.parameters))
        .definition.ports[0]
        .role
        == Role.SIGNAL
    }

    @component
    def bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(DRIVE * V), name="V0")
        c.connect(source.negative, ground)
        senses = []
        for number, part in enumerate(driven.values()):
            sense = c.place(devices.voltage_source(0 * V), name=f"A{number}")
            placed = c.place(part, name=f"X{number}")
            first, *rest = part.definition.ports
            c.connect(sense.negative, placed[first.name], name=f"n{number}")
            c.connect(*(placed[port.name] for port in rest), ground)
            senses.append(sense.positive)
        c.connect(source.positive, *senses, name="drive")

    delayed = {
        number: tuple(f"X{number}{ir.SEPARATOR}{each}" for each in lagged(part))
        for number, (name, part) in enumerate(driven.items())
        if LAGGED in native[name].models
    }
    simulation = Simulation(bench())
    point = simulation.run(OperatingPoint(), TIGHT)
    record = [path for paths in delayed.values() for path in paths]
    frequencies = decade(1 * kHz, 100 * GHz, 1)
    response = simulation.run(Ac("V0", frequencies, record=record), TIGHT)
    lag = 2j * np.pi * response.frequencies * CAPACITANCE_LAG
    root = kits.root(kit, kept)
    deck = [
        f"* every native device of {kit.name}",
        *NOMINAL.get(kit.name, ()),
        f"V0 drive 0 {DRIVE!r} ac 1",
    ]
    for number, (name, part) in enumerate(driven.items()):
        pins = sum(port.role == Role.SIGNAL for port in part.definition.ports)
        deck.append(f"VA{number} drive n{number} 0")
        given = "".join(
            f" {key}={value!r}" for key, value in native[name].parameters.items()
        )
        deck.append(f"X{number} n{number}{' 0' * (pins - 1)} {name}{given}")
    raw_point, raw_response = ngspice.runs(
        deck,
        ("op", "ac dec 1 1k 100g"),
        libraries=tuple(root / path for path in kit.includes),
        sections=tuple((root / path, section) for path, section in kit.typical),
        compatibility=kit.compatibility,
        temperature=ngspice.same_kelvins(celsius(27).value),
    )
    # A shared drive node sums every device current in ngspice. Rounding of
    # that sum perturbs each sensed current by at most (n - 1)*eps times the
    # sum of magnitudes (Higham, 2002, §4.2), reaching 2.4e-14 A on x86.
    sensed = [abs(raw_point.real(f"i(va{number})")[0]) for number in range(len(driven))]
    summed = (len(sensed) - 1) * np.finfo(float).eps * sum(sensed)
    for number, name in enumerate(driven):
        column = simulation.unknowns.index(
            results.Unknown(f"A{number}:current", "current")
        )
        if name in DEPARTED:
            continue
        held = diodes(driven[name])
        models = native[name].models
        tolerance = (
            junction_tolerance(DRIVE, celsius(27).value)
            if any(model.startswith(SPICE_THERMAL_VOLTAGE) for model in models)
            else AGREEMENT
        ) + (JUNCTION if held else 0.0)
        found, spice = point.values[column], raw_point.real(f"i(va{number})")[0]
        if not any(operating_point_departed(each) for each in held):
            assert abs(found - spice) <= tolerance * abs(spice) + FLOOR + summed, (
                name,
                summed,
            )
        if any(model.startswith(AC_DEPARTED) for model in models) or any(
            response_departed(each) for each in held
        ):
            continue
        spice_ac = raw_response.vectors[f"i(va{number})"]
        found_ac = response.values[:, column] - sum(
            (response.current(path, "positive") * lag / (1 + lag))
            for path in delayed.get(number, ())
        )
        # Cross-engine AC agreement using AGREEMENT (2 * ngspice reltol) plus junction
        # tolerance plus absolute current floor FLOOR.
        np.testing.assert_allclose(
            found_ac,
            spice_ac,
            rtol=0,
            atol=tolerance * float(np.abs(spice_ac).max()) + FLOOR,
        )


CHARGED: Final = ("capacitor of a charge", LAGGED)
"""The capacitors jolty reads as behavioral capacitors: of a charge, and of a value
that reads their voltage, integrated into one."""

BIASES: Final = (-2.0, -1.0, 0.0, 1.0, 2.0)
"""The volts across each capacitor of `CHARGED` in
`test_a_kits_charged_capacitors_follow_their_bias_as_in_ngspice`: within the kits' 3.3 V
and 5 V devices' ratings, and across the varactors' 2.3 V."""


@SKIP
@pytest.mark.parametrize("kit", KITS, ids=[kit.name for kit in KITS])
def test_a_kits_charged_capacitors_follow_their_bias_as_in_ngspice(
    kept: Settings, kit: kits.Kit
) -> None:
    """Verifies bias-dependent 1 MHz admittances of native kit capacitors.

    Drives one copy of each `CHARGED` device per bias in `BIASES` through a
    dedicated 0 V sense source with remaining pins grounded, compensating
    `LAGGED` capacitors by `CAPACITANCE_LAG` to match ngspice.
    """
    library = kits.read(kit, settings=kept)
    charged = {
        device.name: device
        for device in kits.inventory(kit, settings=kept)
        if device.status == "native"
        and device.kind == "subcircuit"
        and any(model in CHARGED for model in device.models)
    }
    placed = [
        (name, bias, library.component(name, **device.parameters))
        for name, device in charged.items()
        for bias in BIASES
    ]
    if not placed:
        return  # The kit has none, as SG13G2.

    @component
    def bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(0 * V), name="V0")
        c.connect(source.negative, ground)
        holds = []
        for number, (_, bias, part) in enumerate(placed):
            held = c.place(devices.voltage_source(bias * V), name=f"B{number}")
            sense = c.place(devices.voltage_source(0 * V), name=f"A{number}")
            device = c.place(part, name=f"X{number}")
            first, *rest = part.definition.ports
            holds.append(held.negative)
            c.connect(held.positive, sense.positive, name=f"b{number}")
            c.connect(sense.negative, device[first.name], name=f"n{number}")
            c.connect(*(device[port.name] for port in rest), ground)
        c.connect(source.positive, *holds, name="drive")

    delayed = {
        number: tuple(f"X{number}{ir.SEPARATOR}{each}" for each in lagged(part))
        for number, (name, _, part) in enumerate(placed)
        if LAGGED in charged[name].models
    }
    simulation = Simulation(bench())
    record = [path for paths in delayed.values() for path in paths]
    response = simulation.run(Ac("V0", [1 * MHz], record=record), TIGHT)
    lag = 2j * np.pi * response.frequencies * CAPACITANCE_LAG
    deck = [f"* charged capacitors of {kit.name}", *NOMINAL.get(kit.name, ())]
    deck.append("V0 drive 0 0 ac 1")
    for number, (name, bias, part) in enumerate(placed):
        pins = sum(port.role == Role.SIGNAL for port in part.definition.ports)
        given = "".join(
            f" {key}={value!r}" for key, value in charged[name].parameters.items()
        )
        deck += [
            f"VB{number} b{number} drive {bias!r}",
            f"VA{number} b{number} n{number} 0",
            f"X{number} n{number}{' 0' * (pins - 1)} {name}{given}",
        ]
    root = kits.root(kit, kept)
    raw = ngspice.run(
        deck,
        "ac lin 1 1meg 1meg",
        libraries=tuple(root / path for path in kit.includes),
        sections=tuple((root / path, section) for path, section in kit.typical),
        compatibility=kit.compatibility,
        temperature=ngspice.same_kelvins(celsius(27).value),
    )
    for number in range(len(placed)):
        column = simulation.unknowns.index(
            results.Unknown(f"A{number}:current", "current")
        )
        found = response.values[:, column] - sum(
            (response.current(path, "positive") * lag / (1 + lag))
            for path in delayed.get(number, ())
        )
        spice = raw.vectors[f"i(va{number})"]
        np.testing.assert_allclose(
            found, spice, rtol=0, atol=AGREEMENT * float(np.abs(spice).max()) + FLOOR
        )


QUOTED: Final = re.compile(r"=\s*\"")
"""Pattern for double-quoted unbraced card values ignored by ngspice 47.

ngspice warns of a model issue and falls back to defaults, whereas jolty
evaluates the expression (e.g., SKY130 `res_generic_po` narrowing, 0.56%).
"""

CARDS: Final = {"r": "R", "c": "C", "d": "D"}
"""Mapping from SPICE model-card kind (`r`, `c`, `d`) to bare element prefix."""


@SKIP
@pytest.mark.parametrize("kit", KITS, ids=[kit.name for kit in KITS])
def test_every_native_card_agrees_with_ngspice(kept: Settings, kit: kits.Kit) -> None:
    """Verifies DC and AC agreement with ngspice for bare native kit model cards.

    Instantiates each native resistor, capacitor, and diode model card using
    card defaults and drives it from DC to 100 GHz. Excludes `QUOTED` cards
    dropped by ngspice and documented diode departures.
    """
    library = kits.read(kit, settings=kept)
    cards = [
        (name, CARDS[card.kind])
        for device in kits.inventory(kit, settings=kept)
        if device.status == "native"
        and device.kind == "model"
        and (card := library.card(name := device.name, {})) is not None
        and card.kind in CARDS
        and not QUOTED.search(card.statement.text)
    ]
    if not cards:
        return  # The kit has none, as GF180MCU, whose cards hold voltages yet.
    elements = [
        f"{letter}{number} n{number} 0 {name}"
        for number, (name, letter) in enumerate(cards)
    ]
    pins = " ".join(f"n{number}" for number in range(len(cards)))
    library.read(
        f".subckt bare {pins}\n" + "\n".join(elements) + "\n.ends\n", "bare.lib"
    )
    bare = library.component("bare")

    @component
    def bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(DRIVE * V), name="V0")
        placed = c.place(bare, name="X0")
        c.connect(source.negative, placed["ground"], ground)
        senses = []
        for number in range(len(cards)):
            sense = c.place(devices.voltage_source(0 * V), name=f"A{number}")
            c.connect(sense.negative, placed[f"n{number}"], name=f"n{number}")
            senses.append(sense.positive)
        c.connect(source.positive, *senses, name="drive")

    simulation = Simulation(bench())
    point = simulation.run(OperatingPoint(), TIGHT)
    response = simulation.run(Ac("V0", decade(1 * kHz, 100 * GHz, 1)), TIGHT)
    root = kits.root(kit, kept)
    deck = [
        f"* every native card of {kit.name}",
        *NOMINAL.get(kit.name, ()),
        f"V0 drive 0 {DRIVE!r} ac 1",
    ]
    for number, element in enumerate(elements):
        deck += [f"VA{number} drive n{number} 0", element]
    raw_point, raw_response = ngspice.runs(
        deck,
        ("op", "ac dec 1 1k 100g"),
        libraries=tuple(root / path for path in kit.includes),
        sections=tuple((root / path, section) for path, section in kit.typical),
        compatibility=kit.compatibility,
        temperature=ngspice.same_kelvins(celsius(27).value),
    )
    held = {
        instance.name: instance.parameters
        for instance in bare.definition.instances
        if instance.of == ir.DeviceOf(model="diode")
    }
    for number, (_, letter) in enumerate(cards):
        column = simulation.unknowns.index(
            results.Unknown(f"A{number}:current", "current")
        )
        diode = held.get(f"{letter}{number}")
        tolerance = AGREEMENT + (JUNCTION if diode is not None else 0.0)
        found, spice = point.values[column], raw_point.real(f"i(va{number})")[0]
        if diode is None or not operating_point_departed(diode):
            assert abs(found - spice) <= tolerance * abs(spice) + FLOOR
        if diode is not None and response_departed(diode):
            continue
        spice_ac = raw_response.vectors[f"i(va{number})"]
        # Cross-engine AC agreement using AGREEMENT (+ JUNCTION for diodes) + FLOOR.
        np.testing.assert_allclose(
            response.values[:, column],
            spice_ac,
            rtol=0,
            atol=tolerance * float(np.abs(spice_ac).max()) + FLOOR,
        )


EVERY_PART: Final = (
    (
        ".model sized d level=3 js=1e-2 n=1.3 rs=2e-10 cj=1e-3 vj=0.8 m=0.4 fc=0.6"
        " tt=1e-9 ikf=1e9 ikr=1e3 jsw=1e-9 ns=1.1 cjsw=1e-10 php=0.6 mjsw=0.3 fcs=0.4"
        " ikp=1e2 jtun=5e-2 jtunsw=1e-9 ntun=20 xtitun=2 keg=0.8 bv=5 ibv=1e-3"
        " tcv=1e-3 tlev=2 eg=1.17 gap1=5e-4 gap2=1000 xti=2.5 trs=2e-3 trs2=1e-5"
        " ttt1=1e-3 ttt2=1e-6 tm1=1e-3 tm2=1e-6 tlevc=1 cta=1e-3 ctp=2e-3 tpb=2e-3"
        " tphp=1e-3"
    ),
    (
        ".model behind d is=1e-14 n=1.1 rs=5 cjo=1e-12 jsw=2e-15 ns=1.4 rsw=40"
        " cjsw=2e-13 ikf=1e-2 jtunsw=1e-15 bv=7 ibv=1e-4 nbv=1.5 tcv=2e-4 tlev=1"
        " trs=1e-3 xti=2"
    ),
    (
        ".model recombining d is=1e-14 rs=10 isr=1e-12 nr=2.2 m=0.35 vj=0.7 jsw=1e-15"
        " ikr=1e-9 bv=8 ibv=1e-6"
    ),
)
"""Diode cards exercising every ngspice diode feature across three models.

Covers a level-3 geometry-sized diode with parallel sidewall, knees, tunneling,
`TLEV=2`, and `TLEVC=1` at 27 °C; a diode with a dedicated sidewall series
resistance at `TLEV=1`; and a recombining diode whose breakdown lies outside
the sweep.
"""

EVERY_ELEMENT: Final = (
    "D0 n0 0 sized w=1u l=2u",
    "D1 n1 0 behind pj=1",
    "D2 n2 0 recombining pj=1",
)
"""Element lines instantiating each `EVERY_PART` diode card on nodes `n0` to `n2`."""


@SKIP
@pytest.mark.parametrize("hot", [False, True], ids=["27C", "85C"])
def test_a_diode_of_every_part_agrees_with_ngspice(hot: bool) -> None:  # noqa: FBT001 - a parameter of the test
    """Verifies DC and AC agreement with ngspice for every `EVERY_PART` diode.

    Drives each diode through a dedicated 0 V sense source at 27 °C and 85 °C,
    sweeping from -6 V through breakdown to 0.75 V within `AGREEMENT + reltol`.
    Compares 1 kHz to 100 GHz AC admittance at 0.5 V except for the recombining
    diode, where ngspice computes recombination conductance inexactly.
    """
    temperature = celsius(85 if hot else 27)
    library = Library(temperature.value)
    library.read(
        "\n".join([*EVERY_PART, ".subckt every n0 n1 n2", *EVERY_ELEMENT, ".ends", ""]),
        "every.lib",
    )
    part = library.component("every")

    @component
    def bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(DRIVE * V), name="V0")
        placed = c.place(part, name="X0")
        c.connect(source.negative, placed["ground"], ground)
        senses = []
        for number in range(len(EVERY_ELEMENT)):
            sense = c.place(devices.voltage_source(0 * V), name=f"A{number}")
            c.connect(sense.negative, placed[f"n{number}"], name=f"n{number}")
            senses.append(sense.positive)
        c.connect(source.positive, *senses, name="drive")

    simulation = Simulation(bench(), temperature=temperature)
    sweep = simulation.run(DcSweep("V0", linear(-6 * V, 0.75 * V, 28)), TIGHT)
    response = simulation.run(Ac("V0", decade(1 * kHz, 100 * GHz, 1)), TIGHT)
    deck = ["* every part of a diode", *EVERY_PART, f"V0 drive 0 {DRIVE!r} ac 1"]
    for number, element in enumerate(EVERY_ELEMENT):
        deck += [f"VA{number} drive n{number} 0", element]
    raw_sweep, raw_response = ngspice.runs(
        deck,
        ("dc V0 -6 0.75 0.25", "ac dec 1 1k 100g"),
        temperature=temperature.value,
    )
    for number in range(len(EVERY_ELEMENT)):
        column = simulation.unknowns.index(
            results.Unknown(f"A{number}:current", "current")
        )
        found, spice = sweep.values[:, column], raw_sweep.real(f"i(va{number})")
        # In breakdown, ngspice's current carries its breakdown voltage's match too,
        # within its relative tolerance of the breakdown current (`diotemp.c`, lines 230
        # to 238), where jolty matches it to rounding.
        np.testing.assert_allclose(
            found, spice, rtol=AGREEMENT + ngspice.RELATIVE_TOLERANCE, atol=FLOOR
        )
        if number == len(EVERY_ELEMENT) - 1:
            continue  # The recombining diode's conductance departs.
        spice_ac = raw_response.vectors[f"i(va{number})"]
        np.testing.assert_allclose(
            response.values[:, column],
            spice_ac,
            rtol=0,
            atol=AGREEMENT * float(np.abs(spice_ac).max()) + FLOOR,
        )


def test_the_tables_name_every_status() -> None:
    """Verifies that `STATUSES` covers every status defined by `kits.Status`."""
    assert set(STATUSES) == set(typing.get_args(kits.Status.__value__))


def test_kits_read_uses_disk_cache_and_matches_cold_parse_bit_identically(
    kept: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies `kits.read` hits disk cache across temperatures with bit-exact IR."""
    sg13g2_settings = Settings(
        cache=tmp_path / "sg13g2_cache",
        pdk_root=kits.root(sg13g2.KIT, kept),
    )
    sky130_settings = Settings(
        cache=tmp_path / "sky130_cache",
        pdk_root=kits.root(sky130.KIT, kept),
    )
    cold_sg13g2 = kits.read(sg13g2.KIT, settings=sg13g2_settings)
    cold_sky130 = kits.read(sky130.KIT, settings=sky130_settings)
    assert len(list((sg13g2_settings.cache / "spice").glob("*.zlib"))) == 1
    assert len(list((sky130_settings.cache / "spice").glob("*.zlib"))) == 1

    def fail_on_reparse(*_args: object, **_kwargs: object) -> object:
        message = "expected kits.read disk cache hit without re-expanding PDK files"
        raise AssertionError(message)

    monkeypatch.setattr(spice_library, "expanded", fail_on_reparse)
    warm_sg13g2 = kits.read(sg13g2.KIT, settings=sg13g2_settings)
    warm_sg13g2_hot = kits.read(
        sg13g2.KIT, settings=sg13g2_settings, temperature=celsius(85).value
    )
    warm_sky130 = kits.read(sky130.KIT, settings=sky130_settings)

    assert circuit(warm_sg13g2.component("npn13G2")) == circuit(
        cold_sg13g2.component("npn13G2")
    )
    assert circuit(warm_sg13g2.component("pnpMPA")) == circuit(
        cold_sg13g2.component("pnpMPA")
    )
    assert circuit(
        warm_sky130.component("sky130_fd_pr__nfet_01v8", w=1.0, l=0.15)
    ) == circuit(cold_sky130.component("sky130_fd_pr__nfet_01v8", w=1.0, l=0.15))
    assert len(list((sg13g2_settings.cache / "spice").glob("*.zlib"))) == 1
    assert (
        _parameters(warm_sg13g2_hot.component("npn13G2"), "vbic")[
            "forward_transit_time"
        ]
        != _parameters(warm_sg13g2.component("npn13G2"), "vbic")["forward_transit_time"]
    )


def test_kits_library_temperature_change_reuses_temper_free_devices(
    kept: Settings,
) -> None:
    """Verifies `pnpMPA` stays cached across temperatures while `npn13G2` updates."""
    loaded = kits.read(sg13g2.KIT, settings=kept)
    pnp_at_27 = loaded.component("pnpMPA")
    npn_at_27 = loaded.component("npn13G2")

    hot_kelvins = celsius(85).value
    loaded.temperature = hot_kelvins
    fresh_hot = kits.read(sg13g2.KIT, settings=kept, temperature=hot_kelvins)

    assert loaded.component("pnpMPA") is pnp_at_27
    npn_at_85 = loaded.component("npn13G2")
    assert npn_at_85 is not npn_at_27
    assert circuit(npn_at_85) == circuit(fresh_hot.component("npn13G2"))
