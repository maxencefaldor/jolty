# Copyright 2026 Maxence Faldor
"""The gallery's pages state how far the two engines' readings lie apart."""

import math
import re
import runpy
from typing import Final

from jolty.specs import Verdict
from workspace import ROOT

GALLERY = runpy.run_path(str(ROOT / "docs" / "gallery.py"), run_name="gallery")
"""The script that builds the docs' gallery, loaded without building it."""

_PNG_LINK: Final = re.compile(r"!\[[^\]]*\]\(([^)\s]+\.png)\)")


def _gain(value: float) -> Verdict:
    """A gain read in decibels that passed its bounds."""
    return Verdict(requirement="gain", status="passed", value=value, unit="dB")


def test_a_reading_of_an_exact_zero_is_stated_rather_than_differenced() -> None:
    difference = GALLERY["_difference"](_gain(-math.inf), _gain(-321.276))
    assert difference == "jolty reads an exact zero"


def test_finite_readings_differ_relative_to_the_larger() -> None:
    # |1 - 1.0001| / 1.0001 is 9.999e-5, written to two digits.
    assert GALLERY["_difference"](_gain(1.0), _gain(1.0001)) == "1.0e-04"


def test_missing_gallery_values_use_ascii_hyphen_not_em_dash() -> None:
    assert GALLERY["_si"](None, "V") == "-"
    assert GALLERY["_reading"](None) == "-"
    assert GALLERY["_difference"](_gain(1.0), None) == "-"


def test_every_docs_image_is_referenced_by_markdown() -> None:
    docs = ROOT / "docs"
    images = docs / "images"
    markdown_files = [ROOT / "README.md", *sorted(docs.rglob("*.md"))]
    referenced = {
        (markdown.parent / target).resolve()
        for markdown in markdown_files
        for target in _PNG_LINK.findall(markdown.read_text(encoding="utf-8"))
        if not target.startswith(("http://", "https://"))
    }
    on_disk = {image.resolve() for image in images.rglob("*.png")}
    orphans = sorted(path.relative_to(ROOT).as_posix() for path in on_disk - referenced)
    missing = sorted(
        path.relative_to(ROOT).as_posix() for path in referenced if not path.is_file()
    )
    assert not orphans, f"unreferenced PNGs under docs/images: {orphans}"
    assert not missing, f"missing PNGs referenced in docs: {missing}"
