# Copyright 2026 Maxence Faldor
"""GlobalFoundries' GF180MCU process: its model files, as published for ngspice.

Its primitives' repository publishes them, and [`KIT`][jolty.spice.gf180mcu.KIT] pins
them, as [`kits.read`][jolty.spice.kits.read] reads them.
"""

from pathlib import PurePosixPath
from typing import Final

from jolty.spice.kits import Corners, Fetched, Kit

VERSION: Final = "9f992d5a9186d1f7820c58f039c484ad35b2edea"
"""The commit of GF180MCU's primitives' repository whose ngspice models jolty pins, its
last."""

_SOURCES: Final = (
    "https://raw.githubusercontent.com/google/globalfoundries-pdk-libs-gf180mcu_fd_pr/"
    f"{VERSION}/models/ngspice/"
)
"""Where GlobalFoundries publishes GF180MCU's ngspice models."""

DIRECTORY: Final = PurePosixPath("gf180mcuD/libs.tech/ngspice")
"""Where open_pdks installs GF180MCU's ngspice models, for its five-metal stack."""

_FILES: Final = {
    "design.ngspice": (
        "8d9721a5bf8f079d3fddbd03339af9a0c84d4feb06db8e06465fbd02c7500508"
    ),
    "sm141064.ngspice": (
        "73fc67d38747d95ce03f3c2ba5f0a25c98f56a293363a9df4c971a3a28a3dcda"
    ),
    "smbb000149.ngspice": (
        "3b0995ba269b6cbc468f34fe977c4d21b0faa2935e02f4b2a5d24f58ea66affe"
    ),
}
"""Each model file, by name, and its SHA-256 digest."""

_LIBRARY: Final = DIRECTORY / "sm141064.ngspice"
"""The model library of every device but the 10 V asymmetric transistors."""

KIT: Final = Kit(
    name="gf180mcu",
    version=VERSION,
    files=tuple(
        Fetched(_SOURCES + name, digest, DIRECTORY / name)
        for name, digest in _FILES.items()
    ),
    includes=(DIRECTORY / "design.ngspice",),
    corners=(
        Corners(_LIBRARY, ("typical", "ff", "ss", "fs", "sf")),
        Corners(_LIBRARY, ("bjt_typical", "bjt_ss", "bjt_ff")),
        Corners(_LIBRARY, ("diode_typical", "diode_ss", "diode_ff")),
        Corners(_LIBRARY, ("res_typical", "res_ss", "res_ff")),
        Corners(_LIBRARY, ("mimcap_typical", "mimcap_ss", "mimcap_ff")),
        Corners(_LIBRARY, ("moscap_typical", "moscap_ff", "moscap_ss")),
        Corners(DIRECTORY / "smbb000149.ngspice", ("typical", "ss", "ff", "sf", "fs")),
    ),
    statistics=((_LIBRARY, "statistical"),),
)
"""GF180MCU's model files: the corners of its transistors, typical, fast and slow, the
n-channel first where they differ; of its bipolar transistors, diodes, resistors, MIM
and MOS capacitors; and of its 10 V asymmetric transistors. ``design.ngspice``, which
switches its statistics and sets their skews, comes first; the statistical section
draws the process's variation."""
