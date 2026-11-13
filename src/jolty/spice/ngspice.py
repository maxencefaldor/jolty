# Copyright 2026 Maxence Faldor
"""ngspice, run on a deck of SPICE lines, and the raw files it writes, read.

[`run`][jolty.spice.ngspice.run] runs one analysis of a deck in ngspice, with tolerances
as tight as jolty's, and reads the raw file it writes, and
[`runs`][jolty.spice.ngspice.runs] runs several in one ngspice. Each runs at the
temperature where ngspice's thermal voltage is jolty's,
[`spice_celsius`][jolty.spice.ngspice.spice_celsius], so the two engines differ by their
devices rather than their physical constants.
"""

import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypedDict, Unpack

import numpy as np
import numpy.typing as npt

from jolty import _core
from jolty.errors import JoltyError

EXECUTABLE: Final = shutil.which("ngspice")
"""The ngspice executable on the path, or None when it is not installed."""

VOLTAGE_TOLERANCE: Final = 1e-9
"""The absolute tolerance on voltages ngspice runs with, in volts."""

CURRENT_TOLERANCE: Final = 1e-14
"""The absolute tolerance on currents ngspice runs with, in amperes."""

OPTIONS: Final = (
    ".options reltol=1e-6"
    f" vntol={VOLTAGE_TOLERANCE!r} abstol={CURRENT_TOLERANCE!r} chgtol=1e-18"
)
"""ngspice's options, as tight as jolty's tightest."""

_SPICE_THERMAL_VOLTAGE_PER_KELVIN: Final = 1.38064852e-23 / 1.6021766208e-19
"""ngspice's Boltzmann constant over the elementary charge, from CODATA 2014."""

type Real = npt.NDArray[np.float64]


def spice_celsius(temperature: float) -> float:
    """Where ngspice's thermal voltage is jolty's at `temperature`, in degrees Celsius.

    The temperature is in kelvins. Running ngspice there, and its models' nominal
    temperature where it is jolty's, compares the devices rather than the physical
    constants.
    """
    thermal_voltage = temperature * _core.BOLTZMANN / _core.ELEMENTARY_CHARGE
    return thermal_voltage / _SPICE_THERMAL_VOLTAGE_PER_KELVIN - 273.15


@dataclass(frozen=True)
class Raw:
    """The vectors of one ngspice analysis, by lowercase name.

    It keeps what ngspice printed while it ran the deck the analysis is of.
    """

    plot: str
    vectors: dict[str, npt.NDArray[np.complex128]]
    printed: str

    def real(self, name: str) -> Real:
        """The real part of the vector `name`."""
        return self.vectors[name].real


_TRANSIENT_OPERATING_POINT: Final = "Transient op started"
"""What ngspice prints when Newton iteration, gmin stepping and source stepping all fail
to find an operating point, and it takes the end of a transient from zero instead."""

_ABORTED: Final = "simulation(s) aborted"
"""What ngspice prints when an analysis stops short, as a transient whose step falls
below its smallest; the results it writes then end where it stopped."""

_SHORTEST_TRANSIENT_OPERATING_POINT: Final = "optran 1 1 1 1n 1n 0"
"""Keeps ngspice's first Newton iteration, gmin stepping and source stepping, and makes
the transient it falls back to a single nanosecond: its end is refused anyway, and the
default 10 µs can take minutes on a circuit that defeats the others."""


TIMEOUT: Final = 300
"""The seconds ngspice may run an analysis."""


type Compatibility = Literal["ps", "hsa"]
"""A compatibility mode of ngspice's: PSpice's, ``ps``, or HSPICE's, ``hsa``, which
foundries' process design kits ask for."""


class Running(TypedDict, total=False):
    """How ngspice runs a deck."""

    libraries: tuple[Path, ...]
    """The files it includes."""
    sections: tuple[tuple[Path, str], ...]
    """The sections of libraries it reads, each a file and a section's name, such as a
    foundry's model library at a process corner."""
    compatibility: Compatibility | None
    """The mode it reads the deck and the files in, its own unless given."""
    options: str
    """The options line, as tight as jolty's tightest unless given."""
    temperature: float
    """jolty's temperature, in kelvins, 27 °C unless given."""
    seed: int | None
    """The seed of ngspice's random numbers, such as a kit's mismatch draws, if any."""


def run(
    lines: list[str],
    analysis: str,
    **running: Unpack[Running],
) -> Raw:
    """Runs one analysis command in ngspice and reads the raw file it writes.

    ngspice runs as [`runs`][jolty.spice.ngspice.runs] says.

    Raises:
        JoltyError: As `runs` does.
    """
    (result,) = runs(lines, (analysis,), **running)
    return result


def runs(
    lines: list[str],
    analyses: Sequence[str],
    **running: Unpack[Running],
) -> tuple[Raw, ...]:
    """Runs analysis commands in turn in one ngspice, and reads the plot each writes.

    ngspice includes the `libraries`, reads the `sections` of others, and reads them
    and the deck in its `compatibility` mode, its random numbers drawn from `seed`.
    `options` replaces the tight options. It runs at jolty's `temperature`, in kelvins,
    its models' nominal temperature at jolty's 27 °C. Each command may run several
    analyses, and a result is read from the plot of each.

    Raises:
        JoltyError: When ngspice is not installed, runs out of time, writes no results,
            runs no analysis, aborts one, or finds an operating point only as the end
            of a transient of fixed length, which it does not check to be one.
    """
    if EXECUTABLE is None:
        message = "ngspice is not installed"
        raise JoltyError(message)
    nominal = spice_celsius(_core.REFERENCE_TEMPERATURE)
    temperature = running.get("temperature", _core.REFERENCE_TEMPERATURE)
    deck = [
        lines[0],
        *(f".include {library.resolve()}" for library in running.get("libraries", ())),
        *(
            f'.lib "{library.resolve()}" {section}'
            for library, section in running.get("sections", ())
        ),
        *lines[1:],
        running.get("options", OPTIONS),
        f".options temp={spice_celsius(temperature)!r} tnom={nominal!r}",
        *(
            f".options seed={seed}"
            for seed in [running.get("seed")]
            if seed is not None
        ),
        ".control",
        "set filetype=ascii",
        "set appendwrite",
        _SHORTEST_TRANSIENT_OPERATING_POINT,
        *(f"{analysis}\nwrite result.raw all" for analysis in analyses),
        ".endc",
        ".end",
    ]
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        compatibility = running.get("compatibility")
        if compatibility is not None:
            settings = directory / ".spiceinit"
            settings.write_text(f"set ngbehavior={compatibility}\n", encoding="utf-8")
        (directory / "circuit.cir").write_text("\n".join(deck) + "\n", encoding="utf-8")
        try:
            done = subprocess.run(  # noqa: S603 - ngspice runs the deck written here
                [EXECUTABLE, "-b", "circuit.cir"],
                capture_output=True,
                check=False,  # ngspice exits with 0 on most errors: the raw file tells
                text=True,
                timeout=TIMEOUT,
                cwd=directory,
            )
        except subprocess.TimeoutExpired:
            message = f"ngspice ran for more than {TIMEOUT} s"
            raise JoltyError(message) from None
        raw = directory / "result.raw"
        printed = done.stdout + done.stderr
        if not raw.is_file():
            message = f"ngspice wrote no results:\n{printed}"
            raise JoltyError(message)
        found = _read_raws(raw.read_text(encoding="utf-8"), printed)
    if len(found) != len(analyses) or any(
        result.plot == "Constant values" for result in found
    ):
        message = f"ngspice ran no analysis:\n{printed}"
        raise JoltyError(message)
    if _ABORTED in printed:
        message = f"ngspice aborted the analysis:\n{printed}"
        raise JoltyError(message)
    if _TRANSIENT_OPERATING_POINT in printed:
        message = (
            "ngspice found no operating point but by a transient of fixed length, "
            f"whose end it does not check to be one:\n{printed}"
        )
        raise JoltyError(message)
    return found


def _read_raws(text: str, printed: str) -> tuple[Raw, ...]:
    """Each plot of a raw file, which begins at its title, with what ngspice printed."""
    plots = re.split(r"^(?=Title:)", text, flags=re.MULTILINE)
    return tuple(_read_raw(plot, printed) for plot in plots if plot.strip())


def _read_raw(text: str, printed: str) -> Raw:
    header, values = text.split("Values:\n")
    names = []
    plot = ""
    within = False
    for line in header.splitlines():
        if line.startswith("Plotname:"):
            plot = line.removeprefix("Plotname:").strip()
        elif line.startswith("Variables:"):
            within = True
        elif within:
            names.append(_spice_name(line.split()[1].lower()))
    # Each point starts with its index, a bare integer; values always have an exponent.
    numbers = [
        complex(*map(float, token.split(","))) if "," in token else complex(token)
        for token in values.split()
        if not token.isdigit()
    ]
    table = np.array(numbers).reshape(-1, len(names))
    return Raw(
        plot,
        {name: table[:, column] for column, name in enumerate(names)},
        printed,
    )


_SCALES: Final = frozenset({"time", "frequency"})
"""The scales of a raw file's vectors."""

_NOISES: Final = ("onoise", "inoise")
"""The prefixes of the vectors of noise densities, at the output and referred to the
input, in all and of each device's sources, a device within a subcircuit after a dot
and any other after an underscore."""


def _spice_name(name: str) -> str:
    """A raw file's vector name in the ``v(node)`` and ``i(device)`` forms."""
    if name.endswith("#branch"):
        return f"i({name.removesuffix('#branch')})"
    if name.startswith(("v(", "i(", *_NOISES)) or name in _SCALES:
        return name
    return f"v({name})"
