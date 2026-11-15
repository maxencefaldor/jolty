# Copyright 2026 Maxence Faldor
"""The simulators that run the probes: jolty, ngspice and Xyce.

Each runs a row's probe and returns what it read, the numbers the row names, or raises
`NotOffered` with the simulator's own message when it refuses the request, or
`Failed` when it takes the request and does not finish. ngspice runs the deck through
`jolty.spice.ngspice`, in batch mode, writing an ASCII raw file per command; Xyce runs
it with a ``.print`` of each read in CSV, found on the path or named by ``--xyce``.
"""

import csv
import io
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np
from probes import ROOM, Documented, NotOffered, Read, Row, Spice, Values

from jolty import _core
from jolty.errors import JoltyError
from jolty.spice import ngspice

TIMEOUT: Final = 60
"""The seconds a SPICE simulator may run a probe; each takes well under one."""


class Failed(Exception):  # noqa: N818 - a finding, not a failure of the suite
    """A simulator took a probe's request and did not finish it, with its message."""


NGSPICE_REFUSALS: Final = re.compile(
    r"unknown|not (?:yet )?supported|no such|unrecognized|could not find|"
    r"undefined|not available|cannot",
    re.IGNORECASE,
)
"""ngspice's words when it refuses what a deck asks rather than failing to solve it:
"unknown subckt", "Unknown model type", "not (yet) supported with option KLU", "no
such vector", "could not find a valid modelname"; any other message is a failure."""

XYCE_REFUSALS: Final = re.compile(
    r"Unrecognized|not supported|unknown|undefined|No analysis specified|Netlist error|"
    r"Unable to find|not a valid|is not allowed",
    re.IGNORECASE,
)
"""Xyce's words when it refuses what a deck asks: "Unrecognized dot line will be
ignored", "No analysis specified", "Unable to find", "Netlist error", an error in
reading the deck; any other message is a failure."""


def _said(printed: str, refusals: re.Pattern[str]) -> str:
    """The lines of a simulator's output that say why it stopped, or its last ones."""
    lines = [" ".join(line.split()) for line in printed.splitlines() if line.strip()]
    said = [line for line in lines if refusals.search(line) or "rror" in line]
    return " / ".join(dict.fromkeys(said or lines[-3:]))[:600]


def _refused(printed: str, refusals: re.Pattern[str]) -> Exception:
    """A refusal when the simulator's words say so, else a failure, with its words."""
    said = _said(printed, refusals)
    return NotOffered(said) if refusals.search(printed) else Failed(said)


def jolty(row: Row) -> Values:
    """Runs a row's probe in jolty.

    Raises:
        NotOffered: When jolty offers no way to ask the row, or refuses the probe.
        Failed: When jolty takes the probe and does not finish it.
    """
    if isinstance(row.jolty, NotOffered):
        raise row.jolty
    try:
        return row.jolty()
    except JoltyError as error:
        raise Failed(" ".join(str(error).split())) from None


def _at(scale: np.ndarray, values: np.ndarray, read: Read) -> complex:
    """The value a read names: at its time, interpolated, or at its index."""
    if read.time is not None:
        if values.size == 1:
            return complex(values[0])
        real = np.interp(read.time, scale.real, values.real)
        imaginary = np.interp(read.time, scale.real, values.imag)
        return complex(real, imaginary)
    return complex(values[read.index])


PRINTED: Final = "printed:"
"""What begins a read of a number the simulator prints, such as a measurement's
``tcross = 1.69e-06``, rather than writes as a vector."""


def _printed(printed: str, read: str) -> complex:
    """The number a simulator printed as ``name = value``.

    Raises:
        Failed: When it printed none.
    """
    name = read.removeprefix(PRINTED)
    found = re.search(
        rf"^\s*{re.escape(name)}\s*=\s*(\S+)", printed, re.IGNORECASE | re.MULTILINE
    )
    if found is None:
        message = f"printed no {name}"
        raise Failed(message)
    return complex(float(found.group(1)))


def _spice(row: Row) -> Spice:
    if isinstance(row.spice, NotOffered):
        raise row.spice
    return row.spice


def run_ngspice(row: Row, options: str = "") -> Values:
    """Runs a row's probe in ngspice, with `options` added, such as KLU's.

    Raises:
        NotOffered: When ngspice has no statement for the row, or refuses the probe.
        Failed: When ngspice takes the probe and does not finish it.
    """
    probe = _spice(row)
    if isinstance(probe.ngspice, NotOffered | Documented):
        raise probe.ngspice
    given = " ".join(part for part in (probe.options, options) if part)
    tight = ngspice.OPTIONS + (f"\n{given}" if given else "")
    try:
        plots = ngspice.runs(
            list(probe.deck),
            probe.ngspice,
            options=tight,
            temperature=probe.temperature,
        )
    except JoltyError as error:
        raise _refused(str(error), NGSPICE_REFUSALS) from None
    values: Values = {}
    for name, read in probe.reads.items():
        plot = plots[read.plot]
        if read.ngspice.startswith(PRINTED):
            values[name] = _printed(plot.printed, read.ngspice)
            continue
        vector = plot.vectors.get(read.ngspice.lower())
        if vector is None:
            message = (
                f"ngspice wrote no {read.ngspice}; its plot {plot.plot!r} holds "
                f"{', '.join(sorted(plot.vectors))}"
            )
            raise Failed(message)
        scale = next(iter(plot.vectors.values()))
        values[name] = _at(scale, vector, read)
    return values if probe.derive is None else probe.derive(values)


XYCE_BOLTZMANN: Final = 1.3806226e-23
"""Xyce 7.10's Boltzmann constant (``CONSTboltz``, ``N_DEV_Const.h:45``)."""

XYCE_CHARGE: Final = 1.6021918e-19
"""Xyce 7.10's elementary charge (``CONSTQ``, ``N_DEV_Const.h:43``)."""


def xyce_celsius(temperature: float) -> float:
    """Where Xyce's thermal voltage is jolty's at `temperature`, in degrees Celsius."""
    thermal_voltage = temperature * _core.BOLTZMANN / _core.ELEMENTARY_CHARGE
    return thermal_voltage * XYCE_CHARGE / XYCE_BOLTZMANN - 273.15


def _columns(reads: Mapping[str, Read]) -> list[str]:
    """The expressions Xyce prints for the reads, each complex one as its two parts."""
    columns: list[str] = []
    for read in reads.values():
        if read.xyce is None or not read.printed or read.xyce.startswith(PRINTED):
            continue
        parts = (
            [f"{read.xyce[0]}R{read.xyce[1:]}", f"{read.xyce[0]}I{read.xyce[1:]}"]
            if read.complex
            else [read.xyce]
        )
        columns.extend(part for part in parts if part not in columns)
    return columns


def run_xyce(row: Row, executable: str) -> Values:
    """Runs a row's probe in Xyce, the program at `executable`.

    The deck's lines go to Xyce as written, with the probe's statements, its
    temperature where Xyce's thermal voltage is jolty's, and a ``.print`` of every
    read in CSV.

    Raises:
        NotOffered: When Xyce has no statement for the row, or refuses the probe.
        Failed: When Xyce takes the probe and does not finish it.
    """
    probe = _spice(row)
    if isinstance(probe.xyce, NotOffered | Documented):
        raise probe.xyce
    statements, kind = probe.xyce
    columns = _columns(probe.reads)
    if probe.thermal:
        nominal, temperature = xyce_celsius(ROOM), xyce_celsius(probe.temperature)
    else:
        nominal, temperature = ROOM - 273.15, probe.temperature - 273.15
    deck = [
        *probe.deck,
        *statements,
        f".options device temp={temperature!r} tnom={nominal!r}",
        ".options nonlin reltol=1e-6 abstol=1e-14",
        f".print {kind} format=csv file=probe.csv {' '.join(columns)}",
        ".end",
    ]
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        (directory / "probe.cir").write_text("\n".join(deck) + "\n", encoding="utf-8")
        try:
            done = subprocess.run(  # noqa: S603 - Xyce runs the deck written here
                [executable, "probe.cir"],
                capture_output=True,
                check=False,
                text=True,
                timeout=TIMEOUT,
                cwd=directory,
            )
        except subprocess.TimeoutExpired:
            message = f"Xyce ran for more than {TIMEOUT} s"
            raise Failed(message) from None
        printed = done.stdout + done.stderr
        table = directory / "probe.csv"
        if done.returncode != 0 or not table.is_file():
            raise _refused(printed, XYCE_REFUSALS)
        text = table.read_text(encoding="utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    header = [cell.strip().upper() for cell in rows[0]]
    data = np.array([[float(cell) for cell in line] for line in rows[1:] if line])
    found = {column: data[:, index] for index, column in enumerate(header)}
    scale = data[:, 0]
    values: Values = {}
    for name, read in probe.reads.items():
        if read.xyce is None:
            continue
        if read.xyce.startswith(PRINTED):
            values[name] = _printed(printed, read.xyce)
            continue
        key = read.xyce.upper()
        if read.complex:
            real = found[f"{key[0]}R{key[1:]}"]
            imaginary = found[f"{key[0]}I{key[1:]}"]
            vector = real + 1j * imaginary
        else:
            vector = found[key].astype(complex)
        values[name] = _at(scale, vector, read) ** read.power
    return values if probe.derive is None else probe.derive(values)


def xyce_version(executable: str) -> str:
    """The version Xyce prints of itself."""
    done = subprocess.run(  # noqa: S603 - Xyce prints its version
        [executable, "-v"], capture_output=True, check=False, text=True
    )
    return done.stdout.strip().splitlines()[0] if done.stdout.strip() else ""


def find_xyce(given: str | None) -> str | None:
    """The Xyce program to run: the one given, else the one on the path."""
    return given or shutil.which("Xyce")
