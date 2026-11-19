# Copyright 2026 Maxence Faldor
"""Documentation-sourced coverage cells for commercial simulators unavailable locally.

Each cell is marked ``documented`` when the reference manual lists the corresponding
statement, ``unsupported`` when the manual's analysis list omits the capability, and
``unknown`` when no consulted manual addresses the row. Commercial simulators are not
executed locally, so all entries rely solely on vendor manuals.

Reference manuals consulted on 15 November 2026:

- LTspice: Analog Devices help topic "Dot Commands" (``.op``, ``.dc``, ``.ac``,
  ``.tran``, ``.noise``, ``.tf``, ``.four``, ``.meas``, ``.step``, ``.fra``).
- Spectre 23.1: Cadence *Spectre Circuit Simulator Reference*, version 23.1,
  September 2023.
- HSPICE: Synopsys *HSPICE Quick Reference Guide*, June 2015.
- AFS: Siemens Analog FastSPICE fact sheet and white paper (transient, transient
  noise, and harmonic balance).
"""

from typing import Final

TITLES: Final = {
    "ltspice": "LTspice",
    "spectre": "Spectre 23.1",
    "hspice": "HSPICE 2015",
    "afs": "AFS",
}
"""Mapping from documented simulator identifier to matrix column heading."""

LTSPICE: Final = "LTspice's help, Dot Commands"
SPECTRE: Final = "Spectre Circuit Simulator Reference 23.1"
HSPICE: Final = "HSPICE Quick Reference Guide, June 2015"
AFS: Final = "Analog FastSPICE fact sheet"

_D: Final = "documented"
_N: Final = "unsupported"

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
    "corners": {
        "ltspice": (_D, f"{LTSPICE}: .step over parameters and temp"),
        "spectre": (_D, f"{SPECTRE}: sweep"),
        "hspice": (_D, f"{HSPICE}: .ALTER, .DATA"),
    },
    "disto": {
        "ltspice": (_N, f"{LTSPICE}: no .disto; only .four after a transient"),
        "spectre": (_D, f"{SPECTRE}: distortion through hb"),
    },
    "sp": {
        "spectre": (_D, f"{SPECTRE}: sp"),
        "hspice": (_D, f"{HSPICE}: .LIN"),
    },
    "trnoise": {
        "hspice": (_D, f"{HSPICE}: .TRANNOISE"),
        "afs": (_D, f"{AFS}: transient noise"),
    },
    "measure": {
        "ltspice": (_D, f"{LTSPICE}: .meas"),
    },
    "four": {
        "ltspice": (_D, f"{LTSPICE}: .four"),
    },
}
"""Documentation-sourced cell statuses and citations keyed by row and simulator."""


def cells(row: str) -> dict[str, tuple[str, str]]:
    """Returns documentation status and citation pairs for `row` across manuals."""
    given = CELLS.get(row, {})
    return {
        name: given.get(name, ("unknown", "no document read says")) for name in TITLES
    }
