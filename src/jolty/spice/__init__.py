# Copyright 2026 Maxence Faldor
r"""Read SPICE libraries, such as a vendor's macromodel of an op-amp, as components.

[`load`][jolty.spice.load] reads SPICE files into a [`Library`][jolty.spice.Library],
whose subcircuits each become a component to place: its pins are its ports, in order,
and a subcircuit that reaches SPICE's global ground, node ``0``, has a ``ground`` port
too. Elements read as PSpice and ngspice read them, with their parameters, ``POLY``,
``VALUE`` and ``TABLE`` sources, and switches; whatever jolty does not model is an error
that names the file and the line, never silently dropped. [`fetch`][jolty.spice.fetch]
downloads a model file from where its maker publishes it, checked against its digest.

Examples:
    >>> library = Library()
    >>> library.read(".subckt buffer in out\nE1 out 0 in 0 1\n.ends", "buffer.lib")
    >>> library.subcircuits
    ('buffer',)
"""

from jolty.spice.download import fetch
from jolty.spice.library import (
    GROUND,
    Library,
    Sample,
    Use,
    load,
    standard_deviation,
)

__all__ = ["GROUND", "Library", "Sample", "Use", "fetch", "load", "standard_deviation"]
