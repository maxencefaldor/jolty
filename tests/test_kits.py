# Copyright 2026 Maxence Faldor
"""The open process kits, read at every corner, and their devices jolty runs natively.

The tests fetch each kit's pinned files into pytest's cache the first time. A device
whose models jolty has runs natively, read from the kit, against ngspice reading the
same kit at the same corner, which those tests skip where ngspice is not installed.
"""

import math
import re
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np
import pytest

from differential import SKIP, closest, compare
from jolty import Builder, Component, Role, Simulation, component, devices, ir, results
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
from jolty.spice import gf180mcu, kits, ngspice, sg13g2, sky130
from jolty.spice.expressions import STATISTICAL
from jolty.spice.library import Library, Sample
from jolty.units import GHz, Hz, MHz, V, celsius, fF, kHz, kohm, ns, ps
from jolty.waveforms import Pulse

KITS: Final = (sky130.KIT, gf180mcu.KIT, sg13g2.KIT)
"""The open process kits."""

GUIDE: Final = Path(__file__).resolve().parents[1] / "docs" / "guide" / "foundry.md"
"""The guide's page whose tables state each kit's devices."""

AGREEMENT: Final = 2 * ngspice.RELATIVE_TOLERANCE
"""How closely jolty and ngspice agree, relative to a signal's size: each engine's
Newton iteration stops once its step lies within its relative tolerance, so each lies
about that far from the solution, and the two within twice it."""


@pytest.fixture(scope="session")
def kept(pytestconfig: pytest.Config) -> Settings:
    """Settings whose cache, pytest's, holds each kit's pinned files."""
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
    """Each section, its files included, defines devices and parameters of values."""
    library = kits.read(kit, (section,), settings=kept)
    values = library.values()
    assert library.subcircuits or library.models
    assert all(math.isfinite(value) for value in values.values())
    assert values["temper"] == pytest.approx(27.0)


def test_sky130_scales_its_sizes_to_micrometres(kept: Settings) -> None:
    """SKY130's library sets ngspice's `scale` to the size its netlists write."""
    library = kits.read(sky130.KIT, settings=kept)
    assert library.options["scale"] == sky130.SCALE


def test_temper_is_the_temperature_a_kit_reads_at(kept: Settings) -> None:
    """SG13G2's transit time reads `temper`, the temperature the reader reads at."""
    hot = kits.read(sg13g2.KIT, settings=kept, temperature=celsius(85).value)
    assert hot.values()["temper"] == pytest.approx(85.0)
    times = [
        _parameters(library.component("npn13G2"), "vbic")["forward_transit_time"]
        for library in (kits.read(sg13g2.KIT, settings=kept), hot)
    ]
    # The card's `tf` is 2.08e-13 s times ((temper + 273) / 300) ** 0.7.
    expected = [2.08e-13 * ((temper + 273) / 300) ** 0.7 for temper in (27.0, 85.0)]
    assert times == pytest.approx(expected, rel=1e-12)


MISMATCH: Final = (sg13g2.DIRECTORY / "cornerHBT.lib", "hbt_typ_mismatch")
"""SG13G2's bipolar transistors at their typical corner, drawing their mismatch."""


def test_a_samples_mismatch_is_the_same_bits_on_any_shard(kept: Settings) -> None:
    """Two readings of the kit, as two shards make, draw a sample's mismatch alike.

    A placement draws apart from another, and a sample from another; without a sample,
    each statistical function takes its nominal value.
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
    """Over many samples, each function spreads about its nominal value as it should.

    The bounds are five standard errors of the mean and of the variance over
    `SAMPLES`; the draws are fixed by their seed, so the test is deterministic.
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
        assert abs(values.mean() - 2.0) < 5 * spread / SAMPLES**0.5
        assert abs(values.std() / spread - 1) < 5 * (2 / SAMPLES) ** 0.5
        if function == "limit":
            assert set(np.round(values, 12)) == {1.9, 2.1}
    assert Sample(11, 0).draw("X1", "x")("agauss", (2.0, 0.3, 0.0)) == 2.0


def _parameters(placed: Component, model: str) -> Mapping[str, object]:
    """The parameters of the one device of `model` that `placed` holds."""
    (found,) = (
        instance.parameters
        for instance in placed.definition.instances
        if instance.of == ir.DeviceOf(model=model)
    )
    return found


# The kits' devices and what jolty has of them, as the guide states them.


STATUSES: Final = {
    "native": "native",
    "verilog-a": "Verilog-A",
    "neither": "neither",
    "unread": "unread",
    "unvalued": "unvalued",
}
"""How the guide's tables write each status."""


PROBLEM: Final = re.compile(r"^(?:[^:]+:\d+: )?(?P<problem>.*?)(?:: `[^`]*`)?$")
"""A survey's problem, its file and line before it and its statement after it."""


def table(devices: tuple[kits.Device, ...]) -> list[str]:
    """The rows of a kit's table in the guide: its devices grouped by what they need.

    Each row states a status, the models its devices need, the first problem of its
    first device, one of a model before any other, its place and statement left out,
    and the devices, in the order they are defined, a model marked as one.
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
    """The guide's table of a kit's devices is what the inventory finds at typical."""
    text = GUIDE.read_text(encoding="utf-8")
    marker = f"<!-- devices of: {kit.name} -->"
    assert marker in text, f"{GUIDE.name} has no `{marker}`"
    written = text.split(marker, 1)[1].strip().splitlines()
    rows = table(kits.inventory(kit, settings=kept))
    assert written[: len(rows)] == rows
    assert len(written) == len(rows) or not written[len(rows)].startswith("|")


# Devices jolty has the models of, against ngspice reading the same kit.


@component
def gummel(c: Builder, part: Component) -> None:
    """A bipolar transistor's base driven by `VB` and its collector held by `VC`."""
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
    """IHP's `npn13G2`, a VBIC heated through its fifth node, read from the kit.

    Its base sweeps from off to 1 V, where it warms by 60 to 130 K; its currents, its
    substrate and its thermal node are compared with ngspice's, which reads the same
    library at the same corner, at the same kelvins, as VBIC computes its thermal
    voltage with constants of its own in both engines. It runs at its card's nominal
    27 °C, where ngspice's mapping of a heated VBIC's parameters is the model's, as
    the guide's departures state.
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
    """At 85 °C, the transistor's gain to 10 GHz, set by its transit time, as ngspice's.

    The transit time reads `temper`; read at 27 °C instead, the gain lies a thousand
    tolerances away. The transistor does not heat here: away from its card's nominal
    temperature, ngspice maps a heated VBIC's parameters from the ambient rather than
    from its card's, a departure the guide states.
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


SKY130_CAPACITORS: Final = ("tt", "ll", "hh")
"""SKY130's corners of its capacitors: typical, low and high."""


@SKIP
@pytest.mark.parametrize("corner", SKY130_CAPACITORS)
def test_a_sky130_mim_capacitor_agrees_with_ngspice(
    kept: Settings, corner: str
) -> None:
    """SKY130's MIM capacitor, 10 µm square, its plate and vias' resistances behind it.

    Its admittance from 1 MHz to 100 GHz, where its resistances take over, at 85 °C,
    where they drift, is compared with ngspice's reading the kit at the same corner.
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
    """SKY130's `nfet_01v8` and `pfet_01v8`, BSIM4 4.5 binned by size, read natively.

    A transistor of each of `SKY130_SIZES` selects a bin of its own, its drain and gate
    on two sources at the supply, its source and body grounded; the drain current of
    each, sensed by a source of its own, swept by the drain at a gate at the supply and
    by the gate at a drain at the supply, is compared with ngspice's, which reads the
    same library at the same corner.
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
    """SKY130's `nfet_01v8` and `pfet_01v8`, read natively, from 1 MHz to 10 GHz.

    A transistor 1 µm wide and 0.15 µm long, its gate and drain at half the supply and
    its body at its source, is driven at its gate and at its drain in turn; the
    currents of its drain, gate and body are compared with ngspice's, which reads the
    same library at `tt`. ngspice's capacitances with respect to the body are not its
    charges' derivatives (`AC_DEPARTED`), and the kit's 50 Ω body resistances let the
    body's nodes follow the drive by `ωRC`, about 2π · 10 GHz · 50 Ω · 2 fF = 0.6% at
    10 GHz: the departure, 1e-4 of those capacitances, then moves the currents by under
    1e-6 of themselves, within `AGREEMENT`; at 100 GHz, by 2e-6.
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


NOISY: Final = {
    "sky130-n": (sky130.KIT, "sky130_fd_pr__nfet_01v8", "tt", 1.8, 0.8, (1, 0.15)),
    "sky130-p": (sky130.KIT, "sky130_fd_pr__pfet_01v8", "tt", -1.8, -0.9, (2, 0.5)),
    "gf180mcu-n": (gf180mcu.KIT, "nmos_3p3", "typical", 3.3, 1.2, (1e-6, 5e-7)),
    "gf180mcu-p": (gf180mcu.KIT, "pmos_3p3", "typical", -3.3, -1.4, (2e-6, 5e-7)),
}
"""BSIM4 transistors of the kits whose noise is compared, each with its kit, its
corner, its stage's supply and gate, and its width and length as its kit writes them:
SKY130's of the holistic thermal noise, `tnoimod` 1, GF180MCU's of the charge-based,
`tnoimod` 0, both of BSIM4's flicker noise, `fnoimod` 1."""


@SKIP
@pytest.mark.parametrize("name", NOISY)
def test_a_kits_transistor_makes_the_noise_of_ngspices(
    kept: Settings, name: str
) -> None:
    """A common-source stage of a kit's BSIM4 transistor and 5 kΩ, from 1 Hz to 1 GHz.

    Its output's noise, the channel's thermal and flicker noise and its resistances',
    is compared with ngspice's, which reads the same library at the same corner, within
    `AGREEMENT`: the operating point the noise is computed at agrees within it, and
    the two engines' Boltzmann constants, ngspice's 1.38064852e-23 J/K, move the
    densities by 3.4e-7 of themselves.
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
    """An inverter of SKY130's `nfet_01v8` and `pfet_01v8`, read natively, and its load.

    Its input pulses from 0 to the supply in 50 ps; its output, which the transistors'
    charges and 5 fF load, is compared over two periods with ngspice's, which reads the
    same library at the same corner, within the tolerance of the engine's transients
    against ngspice's (`test_ngspice.test_transients_agree`).
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
    compare(found, spice, unknown, scale=SUPPLY, tolerance=1e-3)


NOMINAL: Final = {gf180mcu.KIT.name: (".param sw_stat_global=0 sw_stat_mismatch=0",)}
"""The parameters that turn each kit's statistical draws off in ngspice, where its files
turn them on, so ngspice computes at the nominal values jolty reads without a sample:
GF180MCU's `design.ngspice` sets both switches to 1, drawing each instance's mismatch
and the process's spread."""

AC_DEPARTED: Final = "BSIM4"
"""The models whose frequency response departs from ngspice's: its BSIM4's capacitances
with respect to the body are not the derivatives of its charges, as the guide's
departures state, and a p-channel's drain at `DRIVE` biases its body forward, where
they depart most. `test_sky130_transistors_admittances_agree_with_ngspice` compares
BSIM4's with the body at the source."""

DRIVE: Final = 0.5
"""The volts each native device's first pin is held at, its others grounded, in
`test_every_native_device_agrees_with_ngspice`: within every device's ratings, and
enough to bias a junction."""

FLOOR: Final = 2 * ngspice.CURRENT_TOLERANCE
"""How far apart two currents the engines find may lie however small they are: each
engine's Newton iteration stops once its step on a current lies within its absolute
tolerance, so the two lie within twice it."""


@SKIP
@pytest.mark.parametrize("kit", KITS, ids=[kit.name for kit in KITS])
def test_every_native_device_agrees_with_ngspice(kept: Settings, kit: kits.Kit) -> None:
    """Each subcircuit the inventory finds native, at its typical corner, as ngspice's.

    One source holds every device's first pin at `DRIVE` and drives it at AC, through
    a source of no volts of each device's own, which senses its current, its other
    pins grounded; each device's current is compared at the operating point and from
    1 kHz to 100 GHz with ngspice's, which reads the same libraries and places the same
    subcircuits, at the same kelvins, but for the frequency response of a device of
    `AC_DEPARTED`.
    """
    library = kits.read(kit, settings=kept)
    native = {
        device.name: device.models
        for device in kits.inventory(kit, settings=kept)
        if device.status == "native" and device.kind == "subcircuit"
    }
    driven = {
        name: part
        for name in native
        if (part := library.component(name)).definition.ports[0].role == Role.SIGNAL
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

    simulation = Simulation(bench())
    point = simulation.run(OperatingPoint(), TIGHT)
    response = simulation.run(Ac("V0", decade(1 * kHz, 100 * GHz, 1)), TIGHT)
    root = kits.root(kit, kept)
    deck = [
        f"* every native device of {kit.name}",
        *NOMINAL.get(kit.name, ()),
        f"V0 drive 0 {DRIVE!r} ac 1",
    ]
    for number, (name, part) in enumerate(driven.items()):
        pins = sum(port.role == Role.SIGNAL for port in part.definition.ports)
        deck.append(f"VA{number} drive n{number} 0")
        deck.append(f"X{number} n{number}{' 0' * (pins - 1)} {name}")
    raw_point, raw_response = ngspice.runs(
        deck,
        ("op", "ac dec 1 1k 100g"),
        libraries=tuple(root / path for path in kit.includes),
        sections=tuple((root / path, section) for path, section in kit.typical),
        compatibility=kit.compatibility,
        temperature=ngspice.same_kelvins(celsius(27).value),
    )
    for number, name in enumerate(driven):
        column = simulation.unknowns.index(
            results.Unknown(f"A{number}:current", "current")
        )
        found, spice = point.values[column], raw_point.real(f"i(va{number})")[0]
        assert abs(found - spice) <= AGREEMENT * abs(spice) + FLOOR
        if any(model.startswith(AC_DEPARTED) for model in native[name]):
            continue
        spice_ac = raw_response.vectors[f"i(va{number})"]
        np.testing.assert_allclose(
            response.values[:, column],
            spice_ac,
            rtol=0,
            atol=AGREEMENT * float(np.abs(spice_ac).max()) + FLOOR,
        )


def test_the_tables_name_every_status() -> None:
    """The guide's names of the statuses are those `kits.Status` holds."""
    assert set(STATUSES) == set(typing.get_args(kits.Status.__value__))
