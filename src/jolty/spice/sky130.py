# Copyright 2026 Maxence Faldor
"""SkyWater's SKY130 process: its transistors, which ngspice simulates with its models.

[`mosfet`][jolty.spice.sky130.mosfet] places a transistor of the process, a foundry's
device, which the engine checks but does not solve: a plan of a circuit that holds one
runs in ngspice, with the process design kit's models. jolty pins one release of the
kit, as open_pdks builds it and ciel publishes it, [`KIT`][jolty.spice.sky130.KIT],
which [`kits.root`][jolty.spice.kits.root] fetches into the cache the first time, each
archive checked against its SHA-256 digest, or finds installed where the settings'
``pdk_root`` says.

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
"""The commit of open_pdks whose build of SKY130 jolty pins."""

LIBRARY: Final = PurePosixPath("sky130A/libs.tech/ngspice/sky130.lib.spice")
"""Where the model library ngspice reads lies within the kit's root."""

TYPICAL: Final = "tt"
"""The typical process corner, which a foundry's devices run at unless another is
named."""

CORNERS: Final = ("tt", "ff", "ss", "sf", "fs", "ll", "hh", "hl", "lh")
"""The process corners of the model library: typical, then fast and slow, n-channel
first where they differ, and the corners of its resistors and capacitors, low and high,
with the transistors at theirs."""

MISMATCH: Final = "_mm"
"""What a corner's name takes after it to name its section that draws each transistor's
mismatch."""

_RELEASES: Final = "https://github.com/fossi-foundation/ciel-releases/releases/download"
"""Where ciel publishes the releases of the kits it builds."""

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
"""The release's archives, each with its digest and the directory it keeps: the model
library, and the models of the primitive devices it includes; the library's corners,
each with its mismatch after it; and HSPICE's compatibility, which the library asks
ngspice for."""


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
"""The transistors of the process jolty places, by the names of their models without
the library's prefix: the 1.8 V core transistors, of standard, low or high threshold,
the native ones of 3.3 V and 5 V, and those of 5 V gates whose drains stand 10.5 V or
16 V."""

PREFIX: Final = "sky130_fd_pr__"
"""The prefix of the names of the primitive devices' models."""

SCALE: Final = 1e-6
"""The length, in meters, that a size of 1 in the kit's netlists stands for: its library
sets ngspice's ``scale`` to a micrometer."""

DIFFUSION: Final = 0.29
"""The length of the diffusion beside each finger, in micrometers, from which the kit's
xschem symbols estimate a transistor's diffusion areas and perimeters."""


def mosfet(
    transistor: str,
    width: Quantity,
    length: Quantity,
    *,
    fingers: int = 1,
    multiplier: int = 1,
) -> Device:
    """The kit's `transistor`, of `width`, all its `fingers` together, and `length`.

    Its terminals are ``drain``, ``gate``, ``source`` and ``bulk``, and `multiplier`
    transistors stand in parallel. ngspice simulates it with the kit's model, which
    bins its sizes: a size that no bin of the model holds fails in ngspice.

    Raises:
        ValueError: When no transistor of the process is named `transistor`, a size is
            not positive, or the fingers or the multiplier are fewer than one.
        DimensionError: When a size is not a length.
    """
    if transistor not in TRANSISTORS:
        message = f"SKY130 has no transistor {transistor!r}: it has {TRANSISTORS}"
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
    """The model and the instance parameters a netlist writes for a SKY130 transistor.

    Its sizes are in micrometers, as the kit's library scales them, and its diffusions'
    areas and perimeters, and their squares, are estimated from `DIFFUSION`, as the
    kit's xschem symbols estimate them; the multiplier is the kit's ``mult``, for its
    mismatch, and ngspice's ``m``.

    Raises:
        ValueError: When the parameters are not a SKY130 transistor's.
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
    """How ngspice reads the kit's model library: at the corner `process`.

    The corner is the library's typical one unless given, and each transistor draws its
    mismatch when `mismatched`.

    Raises:
        SimulationError: When the library has no such corner.
    """
    corner = TYPICAL if process is None else process
    if corner not in CORNERS:
        message = f"SKY130 has no process corner {corner!r}: it has {CORNERS}"
        raise SimulationError(message)
    section = f"{corner}{MISMATCH}" if mismatched else corner
    return {
        "sections": ((root(KIT) / LIBRARY, section),),
        "compatibility": KIT.compatibility,
    }


def inner(element: str, model: str) -> str:
    """The name ngspice gives the transistor within the kit's subcircuit `element`.

    The kit's subcircuit names it after its model, so the transistor of ``XM1``, an
    ``nfet_01v8``, is ``m.xm1.msky130_fd_pr__nfet_01v8``.
    """
    return f"m.{element}.m{model}".lower()


def _micrometers(size: float) -> float:
    """`size`, in meters, in micrometers, to twelve significant digits."""
    return float(f"{size / SCALE:.12g}")


def _rounded(value: float) -> str:
    """`value` to twelve significant digits, as SPICE reads it."""
    return repr(float(f"{value:.12g}")) if isinstance(value, float) else str(value)
