# Copyright 2026 Maxence Faldor
"""The CMC's models, compiled from their published sources, are their closed forms.

R3_CMC 1.1.2, the CMC's three-terminal resistor, MOSVAR 1.3.0, its MOS varactor, and
PSP 103.8.2, its surface-potential MOSFET with JUNCAP 200.6.2, are as IHP's open
process kit publishes them, R3_CMC under the Educational Community License 2.0 and the
others under the notices their sources carry. DIODE_CMC 2.0.0, the CMC's junction
diode, is as the VA-Models collection publishes it, under the CMC's in-code statement.
Neither the sources nor the kernels compiled from them are in the repository: each
source is fetched by its digests and compiled by jolty's Verilog-A compiler into
``crates/jolty-veriloga/tests/fetched/kernels/``, which git ignores, and the crate's
``fetched`` tests, built then, check each kernel against closed forms of the model.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Final

import pytest

from jolty import spice

ROOT: Final = Path(__file__).resolve().parent.parent
"""The repository."""

IHP: Final = (
    "https://raw.githubusercontent.com/IHP-GmbH/IHP-Open-PDK/"
    "e73adb4045731ddec21634146d536380057f6a26/ihp-sg13g2/libs.tech/verilog-a/"
)
"""Where IHP publishes its kit's Verilog-A models, at a commit of the kit."""

VA_MODELS: Final = (
    "https://raw.githubusercontent.com/dwarning/VA-Models/"
    "f04299a330591a591f309f4be1e208f83883ba2a/code/"
)
"""Where VA-Models publishes its models, at a commit of the collection."""

SOURCES: Final = {
    "r3_cmc": (
        IHP + "r3_cmc/",
        {
            "r3_cmc.va": (
                "398746f45048a9e075913e85a982258b929c9042257339bf1fc47d7e25303551"
            ),
            "r3_cmc_macros.include": (
                "66e1bf0b43495a9c17454a1ac203171db5488edc7e19563f4f645389ba7eebb4"
            ),
        },
    ),
    "diode_cmc": (
        VA_MODELS + "diode_cmc/vacode/",
        {
            "diode_cmc.va": (
                "ab816362ac8c5315917c694dffc06f9394041a2735410a7970e19afa88d366b1"
            ),
            "DIODE_CMC_InitModel.include": (
                "d4ffdb9802b07c2a08f73932d0688c0fa8e3e91c7d7dc2490c4d0307a0797a36"
            ),
            "DIODE_CMC_SIMKIT_macrodefs.include": (
                "be1e7ccebc6d638798b3c90434be05017ba8cc74fbc2b92e0353bce0096cb9c8"
            ),
            "DIODE_CMC_macrodefs.include": (
                "88e81fe172c1fcf76daee3bb18acf8bea4422378f0ac79a1ade8f8c0e413be7a"
            ),
            "DIODE_CMC_parlist.include": (
                "d633f167e93af48b368b9f5a530830507bdb4bf79c7bd20c45e87a5a58994b59"
            ),
            "DIODE_CMC_varlist1.include": (
                "bd9ec585c8eb5ae163ad3630baf92b0c3a04c29f333b841d4da0a1b6e91a0ea2"
            ),
            "DIODE_CMC_varlist2.include": (
                "e3522448c7bf5753d9107e45cf53acc4550128a0e8490a3f697417da02614395"
            ),
        },
    ),
    "mosvar": (
        IHP + "mosvar/",
        {
            "mosvar.va": (
                "146eec0c2a9c3c437c53109f8ec42af68f75bd9d341e79797d2feda0340a19f6"
            ),
            "macrosAndDefines_cmc.va": (
                "dda168152714f25ec1c92f75166f22627ff568a3aebe73581c95cef3caaf655a"
            ),
            "mosvarpspMacros.va": (
                "c981ee834d00424c43fef2638b6e924d412f29115d2c8f40f8854efafe154cf5"
            ),
        },
    ),
    "psp103": (
        IHP + "psp103/",
        {
            "psp103.va": (
                "9bab5086f8e203a820043538ede898d0812d07371b1aa6f081d806aaad95cd50"
            ),
            "Common103_macrodefs.include": (
                "14fb6781da0b440a7627aa605c3c59fbf9f755c8eca2b641765bb8d6248029e7"
            ),
            "JUNCAP200_InitModel.include": (
                "335c2b2e8d1ffc9a9c7d820cbe82308e4a91894455f6bbae0e62cdfa9049c690"
            ),
            "JUNCAP200_macrodefs.include": (
                "00d5153fe0a5f1da6523e67354e05c1ae08db2196bf711534fe44b65fb907b65"
            ),
            "JUNCAP200_parlist.include": (
                "d4350de852c966098f74adbd24f70b3e2ed8fec87c7f7d3569ee18168ece5901"
            ),
            "JUNCAP200_varlist.include": (
                "ee30317a2f4a03bfb6bdca471d72feacfa02d5f7cce664fe6ede99482fed37e7"
            ),
            "PSP103_macrodefs.include": (
                "fcb3d0830353c28710645cd28c9caf7610ebc0a05ed1ef89e0ef3e0799581d2b"
            ),
            "PSP103_module.include": (
                "1a37803aea033a823a472d13c9a4fcd98fd8c0e069d8b95474e01a49647cc069"
            ),
            "PSP103_nqs_macrodefs.include": (
                "d4e44d3c8634ce08502f9b25b4f6a56fbe75ce5b8a7d7be987af535b83f16746"
            ),
            "PSP103_parlist.include": (
                "7adbc8e7010fdc6b9c906dace441f2a443c4079693733387dad0d3e3a737ec6f"
            ),
            "PSP103_scaling.include": (
                "1a560ceffd722940f3aa67d64bcfc8f7925d4aa55a0c2caf0c933cf7a47dd54b"
            ),
        },
    ),
}
"""Each model, by the name of its source and its kernel: where its source is published,
and each file of it with the SHA-256 digest of its contents, the model's own first."""

KERNELS: Final = ROOT / "crates/jolty-veriloga/tests/fetched/kernels"
"""Where the kernels compiled from the sources go, for the crate's ``fetched`` tests."""

CARGO: Final = ("cargo", "--quiet")
"""Cargo, quietly, on the repository's own crates."""


@pytest.fixture(scope="session")
def downloads(pytestconfig: pytest.Config) -> Path:
    """Where the sources' files are kept between runs."""
    assert pytestconfig.cache is not None
    return pytestconfig.cache.mkdir("downloads")


def test_each_model_compiles_and_is_its_closed_forms(
    downloads: Path, tmp_path: Path
) -> None:
    """Each source, fetched by its digests, compiles into a kernel its tests pass."""
    KERNELS.mkdir(exist_ok=True)
    for name, (url, files) in SOURCES.items():
        source = tmp_path / name
        source.mkdir()
        for file, digest in files.items():
            shutil.copy(spice.fetch(url + file, digest, downloads), source / file)
        command = [
            *(*CARGO, "run", "--package", "jolty-veriloga"),
            *("--no-default-features", "--bin", "compile", "--"),
            *(
                str(source / f"{name}.va"),
                str(KERNELS / f"{name}.rs"),
                f"{url}{name}.va",
            ),
        ]
        subprocess.run(command, check=True, cwd=ROOT)  # noqa: S603 - cargo, on the repository's own compiler
    command = [
        *(*CARGO, "test", "--package", "jolty-veriloga"),
        *("--features", "fetched", "--test", "fetched"),
    ]
    tested = subprocess.run(  # noqa: S603 - cargo, on the repository's own tests
        command, check=False, capture_output=True, text=True, cwd=ROOT
    )
    assert tested.returncode == 0, tested.stdout + tested.stderr
