# Copyright 2026 Maxence Faldor
"""Loads, before any test runs, every module whose constants Hypothesis could draw on.

Now and then Hypothesis draws a constant it found in the source of a local module
loaded so far, one neither of the standard library nor installed as a package: every
module of jolty, installed from its source, and the benchmarks' scripts. Its constants
only grow as modules load, so the same choices would make another random circuit after
other tests loaded more modules, and a module loaded while a test runs would make
Hypothesis find the test's data generation flaky. Loaded here, they are the same in
every run, whichever tests it runs, so the random circuits are too.
"""

import importlib
import pkgutil
import sys
from pathlib import Path

import jolty

BENCHMARKS = Path(__file__).parent.parent / "benchmarks"
"""The benchmarks, whose shared scripts their tests import."""

for found in pkgutil.walk_packages(jolty.__path__, f"{jolty.__name__}."):
    # The command line's entry point runs the command line as it loads.
    if not found.name.endswith(".__main__"):
        importlib.import_module(found.name)

sys.path.insert(0, str(BENCHMARKS))
for script in ("arms", "verdict", "scoring", "score", "run", "stub", "transcript"):
    importlib.import_module(script)
sys.path.remove(str(BENCHMARKS))
