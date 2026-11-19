# Copyright 2026 Maxence Faldor
"""The circuits the engine checks its kernels on are those the corpus and examples run.

``crates/jolty-engine/tests/data/kernels.json`` holds every circuit of the differential
corpus and of the examples, with the analyses run on each, as ``tests/kernel_corpus.py``
records them; the engine's tests read it. The file is checked here, recorded afresh and
compared, so that it cannot fall behind the corpus it was recorded from.
"""

import subprocess
import sys
from pathlib import Path
from typing import Final

from differential import SKIP

SCRIPT: Final = Path(__file__).resolve().parent / "kernel_corpus.py"
"""The script that records the circuits."""


@SKIP
def test_the_kernels_circuits_are_the_corpus_and_the_examples() -> None:
    """Verifies that re-recording the corpus and examples matches the committed JSON."""
    checked = subprocess.run(  # noqa: S603 - this interpreter, on the repository's script
        [sys.executable, str(SCRIPT), "--check"],
        capture_output=True,
        text=True,
        check=False,
        cwd=SCRIPT.parent.parent,
    )
    assert checked.returncode == 0, checked.stderr[-2000:]
