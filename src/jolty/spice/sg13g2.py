# Copyright 2026 Maxence Faldor
"""IHP SG13G2 open process design kit model definitions for ngspice.

[`KIT`][jolty.spice.sg13g2.KIT] pins the published model files for loading through
[`kits.read`][jolty.spice.kits.read]. Transistors, resistors, and varactors use
Verilog-A models compiled by OpenVAF for ngspice, while HBTs use VBIC models.
"""

from pathlib import PurePosixPath
from typing import Final

from jolty.spice.kits import Corners, Fetched, Kit

VERSION: Final = "e73adb4045731ddec21634146d536380057f6a26"
"""Pinned IHP-Open-PDK commit matching `tests/test_veriloga.py`."""

_SOURCES: Final = (
    "https://raw.githubusercontent.com/IHP-GmbH/IHP-Open-PDK/"
    f"{VERSION}/ihp-sg13g2/libs.tech/ngspice/models/"
)
"""Base raw GitHub URL for published SG13G2 ngspice model files."""

DIRECTORY: Final = PurePosixPath("ihp-sg13g2/libs.tech/ngspice/models")
"""Installation path of SG13G2 ngspice models under open_pdks."""

_FILES: Final = {
    "cap_cmomf.lib": "410001def252a236f221442efcea6353874d1614db593db4142ea3b8a499a207",
    "cap_cmomi.lib": "ae6270ea4abf6ad409065d6d24df2a9aa5b121cba3d6facbc14d6a3ceb7b424f",
    "capacitors_mod.lib": (
        "ff9eb4cc9fdccc0645d9db0bd37e25e9fee2a633765910a7f55ac2084c49695f"
    ),
    "capacitors_mod_mismatch.lib": (
        "6c550362c762362e276f0ed0a4f875616ab6a9de853f1a8f6851b52676ea6f77"
    ),
    "capacitors_stat.lib": (
        "68d2ef41174bcfae5997b355c4fd3dbe4f2b8ec4be30ff79d474052b3be1715a"
    ),
    "cornerCAP.lib": "4fc4a0f17ca2aef66ca38c2d2e7cead3307b69508d304ed073129565ec1c3e56",
    "cornerDIO.lib": "28e747e8f7437a3f539c5cc36f50c9d17c57c03ed463a4c9ee8f0a93a823fe0b",
    "cornerHBT.lib": "b907e1b9681d4ca389e511525dda0babe4193e9893d1ba07fad5751e6a08eddd",
    "cornerMOSCAP.lib": (
        "68f3fc5450d6185cc45360662a0ca9d2640a31db3a082bc2af7ec589d2bfd807"
    ),
    "cornerMOShv.lib": (
        "5a1f862dc304fdfe2fb9758b15d54809a688d83c3ce981dad2e8473b3e93481a"
    ),
    "cornerMOSlv.lib": (
        "03d505847c880d233b341be115a1e5460edf4d8e9b3e8a7df791a52fa4455d67"
    ),
    "cornerRES.lib": "e0432415be26b0e530ea1acea44000c42d65904b406955464e4a7f97e95bfa2c",
    "diodes.lib": "e6211bd0318122fbeb707498ede893223c332098cbb21c441b28b9d8c365e83d",
    "resistors_mod.lib": (
        "b79812f31c7b757013a872fe419ad8c1cea3416f77f82826ac3ba34a4506e60d"
    ),
    "resistors_stat.lib": (
        "15b87c1af76d3a010ec28bf56f9a6f2f0c2874f95bc8f42dafba91f580380b02"
    ),
    "sg13g2_bondpad.lib": (
        "19b3edd62d507bfaf03633b3dc3e89b4ab4d43187a309e8fc665cc5af28a0a4f"
    ),
    "sg13g2_dschottky_nbl1_mod.lib": (
        "c3552c165acd21524043ff9a9b3938c71565722466e9568cf6b109408d5ff6da"
    ),
    "sg13g2_dschottky_nbl1_stat.lib": (
        "730c857bbaabb0c7876eb21b9b60a6882955db388956d082615763559c61d262"
    ),
    "sg13g2_esd.lib": (
        "4a4fb26a87f64bc6ed807f161cc7b3afaeb8d5ca7230b08ebba728e805caa925"
    ),
    "sg13g2_hbt_mod.lib": (
        "370f15b1a125a58cc44964cc7291f7d018e7b74c6731bd332a00fab53f94503d"
    ),
    "sg13g2_hbt_stat.lib": (
        "badafaca780a89ab2a7a2ebb4b3e04d0f482e0f52f6c6677f29c419fd99fccc2"
    ),
    "sg13g2_moscap_mismatch.lib": (
        "347786491f820dc1b74720f543dea2ec2d84ee0c3c8bafd8bb334563bd4487da"
    ),
    "sg13g2_moscap_mod.lib": (
        "4873c91eb09fbd167d3239683535d7cb72c6a66ff7419a08ef2817e80c4d2f8d"
    ),
    "sg13g2_moscap_mod_mismatch.lib": (
        "a8f5c79bb5327569e6fb7fdc3c4e64995f817fbdbc00931f4e3b19151d0cf112"
    ),
    "sg13g2_moscap_parm.lib": (
        "52e62c52e7097b3da032ba44127dd478821ee5263b065621f5cf45fc7ffc0490"
    ),
    "sg13g2_moscap_stat.lib": (
        "38bb882ac751c3e33da6deeb55d00a7526f2057ad3baab9b4f91945361dff817"
    ),
    "sg13g2_moshv_mismatch.lib": (
        "2e9612bdb8252c281f422511039dc9b3ea983892aec89f8ef3361592e68fed09"
    ),
    "sg13g2_moshv_mod.lib": (
        "58ce3084c253f2d3753c3b95386e8f4b81dcadfed4f4503dce56862a556c3328"
    ),
    "sg13g2_moshv_mod_mismatch.lib": (
        "75c30f6975aad4cfe95af7c80fcea142ab0df6c9d7667c09120c7d0c9b814801"
    ),
    "sg13g2_moshv_parm.lib": (
        "eea5bbed39184cb1e2654ae29425a9ed6ee2f2341d6cb76c472591c75237bd35"
    ),
    "sg13g2_moshv_stat.lib": (
        "cd30f22637e41fdbbaba506ebe0f17746b8438027e4651a7a87b241a39d68903"
    ),
    "sg13g2_moslv_mismatch.lib": (
        "2d3251e2d42eadb0f8af152a4915fdc60a68bceb95a2d17a52f90cfb34b7e63e"
    ),
    "sg13g2_moslv_mod.lib": (
        "84ec57080b9dd4666417f05db6f7b34c8dd3f12118929f9b308018613bece16e"
    ),
    "sg13g2_moslv_mod_mismatch.lib": (
        "fac66b4344da4e31dece9a95c32d7af5aac7b40a60b4e029201e1904fb4e8acd"
    ),
    "sg13g2_moslv_parm.lib": (
        "cbf10b8453a18bb70b8ad0d6e40faa0937338e699b5460947a94900db6c92a6e"
    ),
    "sg13g2_moslv_stat.lib": (
        "8d807e0ae41be381e655bf51f0eecf72269c8eaa022a2da59a1c90b0772e229a"
    ),
    "sg13g2_svaricaphv_mod.lib": (
        "b6ae39c7334c75c222ecdee9d6b04c9be48657e91351429aebac47d1192a9abb"
    ),
    "sg13g2_svaricaphv_mod_mismatch.lib": (
        "064e1c751508149f2674914c1791c418195c55fff7d82236d20c2c79f49030fe"
    ),
}
"""Each model file, by name, and its SHA-256 digest."""

_TRANSISTORS: Final = tuple(
    f"mos_{corner}{mismatch}"
    for corner in ("tt", "ss", "ff", "sf", "fs")
    for mismatch in ("", "_mismatch")
)
"""The corners of the transistors, typical, slow and fast, the n-channel first where
they differ, each with its mismatch after it."""


def _corner_triplet(prefix: str) -> tuple[str, ...]:
    """Returns typical, best-case, and worst-case section names with mismatch."""
    return tuple(
        f"{prefix}_{corner}{mismatch}"
        for corner in ("typ", "bcs", "wcs")
        for mismatch in ("", "_mismatch")
    )


KIT: Final = Kit(
    name="sg13g2",
    version=VERSION,
    files=tuple(
        Fetched(_SOURCES + name, digest, DIRECTORY / name)
        for name, digest in _FILES.items()
    ),
    includes=(DIRECTORY / "sg13g2_bondpad.lib",),
    corners=(
        Corners(DIRECTORY / "cornerMOSlv.lib", _TRANSISTORS),
        Corners(DIRECTORY / "cornerMOShv.lib", _TRANSISTORS),
        Corners(DIRECTORY / "cornerHBT.lib", _corner_triplet("hbt")),
        Corners(DIRECTORY / "cornerRES.lib", _corner_triplet("res")),
        Corners(DIRECTORY / "cornerCAP.lib", _corner_triplet("cap")),
        Corners(DIRECTORY / "cornerDIO.lib", ("dio_tt", "dio_ss", "dio_ff")),
        Corners(DIRECTORY / "cornerMOSCAP.lib", ("moscap_tt", "moscap_tt_mismatch")),
    ),
    statistics=(
        (DIRECTORY / "cornerMOSlv.lib", "mos_tt_stat"),
        (DIRECTORY / "cornerMOShv.lib", "mos_tt_stat"),
        (DIRECTORY / "cornerHBT.lib", "hbt_typ_stat"),
        (DIRECTORY / "cornerRES.lib", "res_stat"),
        (DIRECTORY / "cornerRES.lib", "res_stat_mismatch"),
        (DIRECTORY / "cornerCAP.lib", "cap_typ_stat"),
        (DIRECTORY / "cornerDIO.lib", "dio_tt_stat"),
        (DIRECTORY / "cornerMOSCAP.lib", "moscap_tt_stat"),
    ),
)
"""SG13G2 kit definition covering MOS, HBT, passive, diode, and MOSCAP corners.

Includes standalone ``sg13g2_bondpad.lib`` first, followed by corner sections with
mismatch variants and statistical process-variation sections.
"""
