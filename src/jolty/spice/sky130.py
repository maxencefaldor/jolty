# Copyright 2026 Maxence Faldor
"""SkyWater SKY130 foundry transistor primitives and ngspice kit configuration.

The [`mosfet`][jolty.spice.sky130.mosfet] helper creates a foundry device that the
native engine checks structurally while delegating simulation to ngspice with SKY130
models. Constant [`KIT`][jolty.spice.sky130.KIT] pins one open_pdks release published
by ciel. Calling [`kits.root`][jolty.spice.kits.root] downloads and verifies each
archive against its SHA-256 digest into the cache or uses a local ``pdk_root``
installation.

Examples:
    >>> from jolty.units import um
    >>> mosfet("nfet_01v8", 1 * um, 0.15 * um).parameters["model"]
    'sky130_fd_pr__nfet_01v8'
"""

import math
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Final

from pydantic import JsonValue

from jolty.devices import Device, device
from jolty.errors import SimulationError
from jolty.spice.kits import Corners, Fetched, Kit, root
from jolty.spice.ngspice import Running
from jolty.units import LENGTH, Quantity

VERSION: Final = "bdc9412b3e468c102d01b7cf6337be06ec6e9c9a"
"""Pinned open_pdks commit hash for the SKY130 release."""

LIBRARY: Final = PurePosixPath("sky130A/libs.tech/ngspice/sky130.lib.spice")
"""Relative path of the top-level ngspice model library within the PDK root."""

TYPICAL: Final = "tt"
"""Default typical process corner when no corner is specified."""

CORNERS: Final = ("tt", "ff", "ss", "sf", "fs", "ll", "hh", "hl", "lh")
"""Supported SKY130 process corners (transistor fast/slow and RC low/high corners)."""

MISMATCH: Final = "_mm"
"""Section suffix enabling per-transistor statistical mismatch."""

_RELEASES: Final = "https://github.com/fossi-foundation/ciel-releases/releases/download"
"""Base download URL for ciel prebuilt open PDK release archives."""

KIT: Final = Kit(
    name="sky130",
    version=VERSION,
    files=(
        Fetched(
            f"{_RELEASES}/sky130-{VERSION}/common.tar.zst",
            "8094c39d69f79b16c301ce995d9a365f0163e9258984844ad29741652980daa7",
            PurePosixPath("sky130A/libs.tech/ngspice"),
            archive=True,
        ),
        Fetched(
            f"{_RELEASES}/sky130-{VERSION}/sky130_fd_pr.tar.zst",
            "fb5cdee699e04c23623ad58d015e14fe6e577d63d6e49ac97ccbcb5723f13138",
            PurePosixPath("sky130A/libs.ref/sky130_fd_pr/spice"),
            archive=True,
        ),
    ),
    includes=(),
    corners=(
        Corners(
            LIBRARY,
            tuple(
                f"{corner}{mismatch}"
                for corner in CORNERS
                for mismatch in ("", MISMATCH)
            ),
        ),
    ),
    compatibility="hsa",
)
"""Pinned SKY130 ciel release archives, digests, corner sections, and HSPICE mode."""


TRANSISTORS: Final = (
    "nfet_01v8",
    "nfet_01v8_lvt",
    "pfet_01v8",
    "pfet_01v8_lvt",
    "pfet_01v8_hvt",
    "nfet_03v3_nvt",
    "nfet_05v0_nvt",
    "nfet_g5v0d10v5",
    "pfet_g5v0d10v5",
    "nfet_g5v0d16v0",
    "pfet_g5v0d16v0",
)
"""Supported SKY130 MOSFET model suffixes without the ``sky130_fd_pr__`` prefix.

Covers 1.8 V core transistors (standard, low, and high threshold), 3.3 V and 5 V native
devices, and 5 V gate devices rated for 10.5 V or 16 V drains.
"""

PREFIX: Final = "sky130_fd_pr__"
"""The prefix of the names of the primitive devices' models."""

SCALE: Final = 1e-6
"""The length, in meters, that a size of 1 in the kit's netlists stands for: its library
sets ngspice's ``scale`` to a micrometer."""

DIFFUSION: Final = 0.29
"""Diffusion length adjacent to each finger in micrometers for area estimation."""


def mosfet(
    transistor: str,
    width: Quantity,
    length: Quantity,
    *,
    fingers: int = 1,
    multiplier: int = 1,
) -> Device:
    """Creates a SKY130 `transistor` device with total `width`, `length`, and `fingers`.

    The device exposes ``drain``, ``gate``, ``source``, and ``bulk`` terminals with
    `multiplier` parallel copies. ngspice simulates the device using binned SKY130
    models; dimensions outside every model bin fail in ngspice.

    Raises:
        ValueError: When `transistor` is unknown, dimensions are non-positive, or
            `fingers` or `multiplier` is less than 1.
        DimensionError: When `width` or `length` does not carry length dimensions.
    """
    if transistor not in TRANSISTORS:
        message = (
            f"SKY130 has no transistor {transistor!r}; "
            f"available transistors: {TRANSISTORS}"
        )
        raise ValueError(message)
    sizes: dict[str, float] = {
        "width": width.to(LENGTH, "a transistor's width"),
        "length": length.to(LENGTH, "a transistor's length"),
    }
    for name, size in sizes.items():
        if not (size > 0 and math.isfinite(size)):
            message = f"a transistor's {name} is positive, not {size!r} m"
            raise ValueError(message)
    for name, count in (("fingers", fingers), ("multiplier", multiplier)):
        if isinstance(count, bool) or count < 1:
            message = f"a transistor's {name} is 1 or more, not {count}"
            raise ValueError(message)
    return device(
        "foundry_mosfet",
        {
            "pdk": "sky130",
            "model": PREFIX + transistor,
            **sizes,
            "fingers": fingers,
            "multiplier": multiplier,
        },
    )


def written(parameters: Mapping[str, JsonValue]) -> str:
    """Formats the SPICE model name and instance parameters for a SKY130 transistor.

    Scales dimensions to micrometers to match the PDK ``scale`` option and estimates
    drain and source diffusion areas, perimeters, and squares from `DIFFUSION` as in
    SKY130 xschem symbols. Emits both ``mult`` (for PDK mismatch) and ``m`` (for
    ngspice multiplicity).

    Raises:
        ValueError: When `parameters` are invalid for a SKY130 transistor.
    """
    match parameters:
        case {
            "pdk": "sky130",
            "model": str(model),
            "width": int() | float() as width,
            "length": int() | float() as length,
            "fingers": int(fingers),
            "multiplier": int(multiplier),
        } if not (
            isinstance(width, bool)
            or isinstance(length, bool)
            or isinstance(fingers, bool)
            or isinstance(multiplier, bool)
        ):
            pass
        case _:
            message = f"not a SKY130 transistor's parameters: {dict(parameters)}"
            raise ValueError(message)
    if not (width > 0 and math.isfinite(width)):
        message = f"a transistor's width is positive and finite, not {width!r} m"
        raise ValueError(message)
    if not (length > 0 and math.isfinite(length)):
        message = f"a transistor's length is positive and finite, not {length!r} m"
        raise ValueError(message)
    if fingers < 1:
        message = f"a transistor's fingers is 1 or more, not {fingers}"
        raise ValueError(message)
    if multiplier < 1:
        message = f"a transistor's multiplier is 1 or more, not {multiplier}"
        raise ValueError(message)
    wide, long = _micrometers(width), _micrometers(length)
    if not (wide > 0 and math.isfinite(wide) and long > 0 and math.isfinite(long)):
        message = (
            f"a transistor's scaled size is positive and finite, not "
            f"width={width!r} m, length={length!r} m"
        )
        raise ValueError(message)
    finger = wide / fingers
    drains, sources = (fingers + 1) // 2, (fingers + 2) // 2
    values = {
        "L": long,
        "W": wide,
        "nf": fingers,
        "ad": drains * finger * DIFFUSION,
        "as": sources * finger * DIFFUSION,
        "pd": 2 * drains * (finger + DIFFUSION),
        "ps": 2 * sources * (finger + DIFFUSION),
        "nrd": DIFFUSION / wide,
        "nrs": DIFFUSION / wide,
        "sa": 0,
        "sb": 0,
        "sd": 0,
        "mult": multiplier,
        "m": multiplier,
    }
    listed = " ".join(f"{name}={_rounded(value)}" for name, value in values.items())
    return f"{model} {listed}"


def running(process: str | None, *, mismatched: bool) -> Running:
    """Builds ngspice library section and compatibility options for SKY130 `process`.

    Defaults to the typical corner when `process` is None, appending the mismatch
    section suffix when `mismatched` is true.

    Raises:
        SimulationError: When `process` is not a supported SKY130 corner.
    """
    corner = TYPICAL if process is None else process
    if corner not in CORNERS:
        message = (
            f"SKY130 has no process corner {corner!r}; available corners: {CORNERS}"
        )
        raise SimulationError(message)
    section = f"{corner}{MISMATCH}" if mismatched else corner
    return {
        "sections": ((root(KIT) / LIBRARY, section),),
        "compatibility": KIT.compatibility,
    }


def inner(element: str, model: str) -> str:
    """Returns the internal ngspice MOSFET path inside SKY130 subcircuit `element`.

    SKY130 subcircuits name the inner transistor after its model, so ``XM1`` of
    ``nfet_01v8`` maps to ``m.xm1.msky130_fd_pr__nfet_01v8``.
    """
    return f"m.{element}.m{model}".lower()


def _micrometers(size: float) -> float:
    """Converts `size` from meters to micrometers with twelve significant digits."""
    return float(f"{size / SCALE:.12g}")


def _rounded(value: float) -> str:
    """Formats `value` to twelve significant digits for SPICE netlist output."""
    return repr(float(f"{value:.12g}")) if isinstance(value, float) else str(value)
