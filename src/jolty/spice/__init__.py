# Copyright 2026 Maxence Faldor
r"""SPICE library import, external simulator runners, and open PDK definitions.

[`load`][jolty.spice.load] reads SPICE files into a [`Library`][jolty.spice.Library],
converting each subcircuit into a placeable component whose ports match its pin list in
order. Subcircuits referencing SPICE global ground (node ``0``) automatically expose a
``ground`` port. Elements follow PSpice and ngspice syntax, supporting parameters,
``POLY``, ``VALUE``, and ``TABLE`` sources, and switches; unsupported constructs raise
errors with exact file and line numbers rather than being silently ignored.
[`fetch`][jolty.spice.fetch] downloads published model files and verifies their SHA-256
digests.

Examples:
    >>> library = Library()
    >>> library.read(".subckt buffer in out\nE1 out 0 in 0 1\n.ends", "buffer.lib")
    >>> library.subcircuits
    ('buffer',)
"""

__lazy_modules__ = ["jolty.spice.download", "jolty.spice.library"]

from jolty.spice.download import fetch
from jolty.spice.library import (
    GROUND,
    BinBounds,
    Library,
    PeriodicSteadyStateCard,
    Sample,
    SubcircuitDerivatives,
    SubcircuitPlacement,
    Use,
    load,
    standard_deviation,
)

__all__ = [
    "GROUND",
    "BinBounds",
    "Library",
    "PeriodicSteadyStateCard",
    "Sample",
    "SubcircuitDerivatives",
    "SubcircuitPlacement",
    "Use",
    "fetch",
    "load",
    "standard_deviation",
]
