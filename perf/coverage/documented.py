# Copyright 2026 Maxence Faldor
"""The cells of the simulators that cannot run here, from their documents alone.

Each cell is ``documented`` where the simulator's document lists the row, by the
statement it names, ``not offered`` where the document's list of its analyses holds
none that asks it, and ``unknown`` where no document read says either way. None is a
probe: these simulators are not installed, or cannot be run here, and their cells
follow their documents.

The documents, as read for ``research/benchmarks.md`` §3.1 of the lead's notes on 15
November 2026:

- LTspice: Analog Devices' list of its simulator directives, in its help, "Dot
  Commands" (``.op``, ``.dc``, ``.ac``, ``.tran``, ``.noise``, ``.tf``, ``.four``,
  ``.meas``, ``.step``, ``.fra``);
- Spectre 23.1: Cadence's *Spectre Circuit Simulator Reference*, version 23.1,
  September 2023, its analyses by statement;
- HSPICE: Synopsys's *HSPICE Quick Reference Guide*, June 2015, the text that could be
  extracted from it;
- AFS: Siemens's Analog FastSPICE fact sheet and white paper, which list a transient,
  transient noise and harmonic balance, and no other analysis.
"""

from typing import Final

TITLES: Final = {
    "ltspice": "LTspice",
    "spectre": "Spectre 23.1",
    "hspice": "HSPICE 2015",
    "afs": "AFS",
}
"""The documented simulators, by name, and the title of each one's column."""

LTSPICE: Final = "LTspice's help, Dot Commands"
SPECTRE: Final = "Spectre Circuit Simulator Reference 23.1"
HSPICE: Final = "HSPICE Quick Reference Guide, June 2015"
AFS: Final = "Analog FastSPICE fact sheet"

_D: Final = "documented"
_N: Final = "not offered"

CELLS: Final[dict[str, dict[str, tuple[str, str]]]] = {
    "op": {
        "ltspice": (_D, f"{LTSPICE}: .op"),
        "spectre": (_D, f"{SPECTRE}: dc"),
        "hspice": (_D, f"{HSPICE}: .OP"),
    },
    "dc-source": {
        "ltspice": (_D, f"{LTSPICE}: .dc, up to three sources"),
        "spectre": (_D, f"{SPECTRE}: dc"),
        "hspice": (_D, f"{HSPICE}: .DC"),
    },
    "dc-two-sources": {
        "ltspice": (_D, f"{LTSPICE}: .dc, up to three sources"),
    },
    "ac": {
        "ltspice": (_D, f"{LTSPICE}: .ac"),
        "spectre": (_D, f"{SPECTRE}: ac"),
        "hspice": (_D, f"{HSPICE}: .AC"),
    },
    "noise": {
        "ltspice": (_D, f"{LTSPICE}: .noise"),
        "spectre": (_D, f"{SPECTRE}: noise"),
        "hspice": (_D, f"{HSPICE}: .NOISE"),
    },
    "tran-trapezoidal": {
        "ltspice": (_D, f"{LTSPICE}: .tran"),
        "spectre": (_D, f"{SPECTRE}: tran"),
        "hspice": (_D, f"{HSPICE}: .TRAN"),
        "afs": (_D, f"{AFS}: transient"),
    },
    "tf": {
        "ltspice": (_D, f"{LTSPICE}: .tf"),
        "spectre": (_D, f"{SPECTRE}: xf"),
        "hspice": (_D, f"{HSPICE}: .TF"),
    },
    "pz": {
        "spectre": (_D, f"{SPECTRE}: pz"),
    },
    "stability": {
        "ltspice": (_D, f"{LTSPICE}: .fra, through a transient"),
        "spectre": (_D, f"{SPECTRE}: stb"),
        "hspice": (_D, f"{HSPICE}: .LSTB"),
    },
    "sens-dc": {
        "spectre": (_D, f"{SPECTRE}: sens, DC"),
    },
    "sens-ac": {
        "spectre": (_D, f"{SPECTRE}: sens, AC"),
    },
    "pss": {
        "ltspice": (_N, f"{LTSPICE}: no periodic steady state among them"),
        "spectre": (_D, f"{SPECTRE}: pss, hb"),
        "hspice": (_D, f"{HSPICE}: .SN, .HB"),
        "afs": (_D, f"{AFS}: harmonic balance"),
    },
    "monte-carlo": {
        "ltspice": (_D, f"{LTSPICE}: .step over random functions"),
        "spectre": (_D, f"{SPECTRE}: montecarlo"),
        "hspice": (_D, f"{HSPICE}: .VARIATION"),
    },
    "measure": {
        "ltspice": (_D, f"{LTSPICE}: .meas"),
    },
}
"""The documented cells, by row then simulator: each status and its source."""


def cells(row: str) -> dict[str, tuple[str, str]]:
    """Each documented simulator's cell of `row`: its status and its source."""
    given = CELLS.get(row, {})
    return {
        name: given.get(name, ("unknown", "no document read says")) for name in TITLES
    }
