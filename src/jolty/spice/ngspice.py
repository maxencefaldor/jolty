# Copyright 2026 Maxence Faldor
"""ngspice, run on a deck of SPICE lines, and the raw files it writes, read.

[`run`][jolty.spice.ngspice.run] runs one analysis of a deck in ngspice, with tolerances
as tight as jolty's, and reads the raw file it writes;
[`runs`][jolty.spice.ngspice.runs] runs several in one ngspice; and
[`read`][jolty.spice.ngspice.read] reads the plots of any raw file. Each runs at the
temperature where ngspice's thermal voltage is jolty's,
[`option_temperature`][jolty.spice.ngspice.option_temperature], so the two engines
differ by their devices rather than their physical constants.
"""

import functools
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal, TypedDict, Unpack

import numpy as np
import numpy.typing as npt

from jolty import _core
from jolty.errors import JoltyError

EXECUTABLE: Final = shutil.which("ngspice")
"""The ngspice executable on the path, or None when it is not installed."""

RELATIVE_TOLERANCE: Final = 1e-6
"""The relative tolerance ngspice runs with."""

VOLTAGE_TOLERANCE: Final = 1e-9
"""The absolute tolerance on voltages ngspice runs with, in volts."""

CURRENT_TOLERANCE: Final = 1e-14
"""The absolute tolerance on currents ngspice runs with, in amperes."""

OPTIONS: Final = (
    f".options reltol={RELATIVE_TOLERANCE!r} vntol={VOLTAGE_TOLERANCE!r}"
    f" abstol={CURRENT_TOLERANCE!r}"
)
"""ngspice's options, as tight as jolty's tightest, the charge tolerance both engines'
default."""

NO_OPTIONS: Final = "*"
"""The options line that leaves every option at ngspice's default: a comment."""

_SPICE_THERMAL_VOLTAGE_PER_KELVIN: Final = 1.38064852e-23 / 1.6021766208e-19
"""ngspice's Boltzmann constant over the elementary charge, from CODATA 2014."""

type Real = npt.NDArray[np.float64]


def option_temperature(temperature: float) -> float:
    """Where ngspice's thermal voltage is jolty's at `temperature`, in degrees Celsius.

    The temperature is in kelvins. Running ngspice there, and its models' nominal
    temperature where it is jolty's, compares the devices rather than the physical
    constants.
    """
    thermal_voltage = temperature * _core.BOLTZMANN / _core.ELEMENTARY_CHARGE
    return thermal_voltage / _SPICE_THERMAL_VOLTAGE_PER_KELVIN - 273.15


def same_kelvins(temperature: float) -> float:
    """The temperature to give [`run`][jolty.spice.ngspice.run] to run at `temperature`.

    `run` runs ngspice where its thermal voltage is jolty's at the temperature it is
    given; given this one, it runs it at `temperature` itself, in kelvins. A model that
    computes with constants of its own, as BSIM3 does with Boltzmann's over the
    electron's charge, agrees with ngspice's at the same kelvins.

    Examples:
        >>> round(option_temperature(same_kelvins(300.15)), 9)
        27.0
    """
    jolty = _core.BOLTZMANN / _core.ELEMENTARY_CHARGE
    return temperature * _SPICE_THERMAL_VOLTAGE_PER_KELVIN / jolty


@dataclass(frozen=True)
class Result:
    """The vectors of one ngspice analysis, by lowercase name.

    It keeps what ngspice printed while it ran the deck the analysis is of.
    """

    plot: str
    vectors: dict[str, npt.NDArray[np.complex128]]
    printed: str
    seconds: float | None = None
    """The time from starting the ngspice that ran the analysis to its exit, in
    seconds, as jolty measures it, the same for every analysis that ngspice ran; None
    for a raw file read without running it."""

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
    timeout: float | None
    """The seconds it may run, `TIMEOUT` unless given, or as long as it takes if
    None."""


class NgspiceTimeoutError(JoltyError):
    """ngspice ran for longer than it was given."""


@dataclass(frozen=True, slots=True)
class Executed:
    """What one run of ngspice on a deck left, and how long it took."""

    printed: str
    """Its standard output, then its standard error."""
    code: int
    """Its exit status, 0 on most errors too: what it printed tells them."""
    seconds: float
    """Its time from its start to its exit, by the wall clock, writing the deck left
    out."""
    values: bytes | None
    """The file of values the deck writes, `VALUES`, ASCII or binary, or None if it
    wrote none."""


VALUES: Final = "result.raw"
"""The raw file a deck run by [`execute`][jolty.spice.ngspice.execute] writes its values
to, which it reads back."""


@functools.cache
def version() -> str | None:
    """Ngspice's version and build, as it states them, or None when it is not installed.

    Examples:
        >>> version() is None or version().startswith("ngspice")
        True
    """
    if EXECUTABLE is None:
        return None
    done = subprocess.run(  # noqa: S603 - ngspice states its version
        [EXECUTABLE, "--version"], capture_output=True, check=False, text=True
    )
    lines = [line.strip("* ") for line in done.stdout.splitlines()]
    return "; ".join(line for line in lines if line.startswith(("ngspice", "Compiled")))


def execute(
    deck: Sequence[str],
    *,
    compatibility: Compatibility | None = None,
    timeout: float | None = TIMEOUT,
) -> Executed:
    """Runs ngspice in batch mode on the lines `deck`, in a directory of its own.

    The deck is written to ``circuit.cir`` in a temporary directory, which the run
    takes as its own, with a ``.spiceinit`` that sets the `compatibility` mode if one
    is given, so the raw file its ``write`` names, `VALUES`, is read back from there;
    the run is stopped after `timeout` seconds, or never if None.

    It is the one way jolty runs ngspice, as
    [`xyce.execute`][jolty.spice.xyce.execute] is Xyce's:
    [`runs`][jolty.spice.ngspice.runs] writes its deck and reads its results around
    it, and the benchmarks run their decks by it.

    Raises:
        JoltyError: When ngspice is not installed.
        NgspiceTimeoutError: When it runs for more than `timeout` seconds, with what it
            had printed.
    """
    if EXECUTABLE is None:
        message = "ngspice is not installed"
        raise JoltyError(message)
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        if compatibility is not None:
            settings = directory / ".spiceinit"
            settings.write_text(f"set ngbehavior={compatibility}\n", encoding="utf-8")
        (directory / "circuit.cir").write_text("\n".join(deck) + "\n", encoding="utf-8")
        started = time.perf_counter()
        try:
            done = subprocess.run(  # noqa: S603 - ngspice runs the deck written here
                [EXECUTABLE, "-b", "circuit.cir"],
                capture_output=True,
                # ngspice exits with 0 on most errors: what it printed tells them.
                check=False,
                text=True,
                timeout=timeout,
                cwd=directory,
            )
        except subprocess.TimeoutExpired as expired:
            # The output so far: bytes on POSIX, text where communicate gathered it.
            printed = "".join(
                part.decode(errors="replace") if isinstance(part, bytes) else part or ""
                for part in (expired.stdout, expired.stderr)
            )
            message = f"ngspice ran for more than {timeout:g} s:\n{printed}"
            raise NgspiceTimeoutError(message) from None
        seconds = time.perf_counter() - started
        values = directory / VALUES
        written = values.read_bytes() if values.is_file() else None
    return Executed(done.stdout + done.stderr, done.returncode, seconds, written)


def run(
    deck: list[str],
    analysis: str,
    **running: Unpack[Running],
) -> Result:
    """Runs one analysis command in ngspice and reads the raw file it writes.

    ngspice runs as [`runs`][jolty.spice.ngspice.runs] says.

    Raises:
        JoltyError: As `runs` does.
    """
    (result,) = runs(deck, (analysis,), **running)
    return result


def runs(
    deck: list[str],
    analyses: Sequence[str],
    **running: Unpack[Running],
) -> tuple[Result, ...]:
    """Runs analysis commands in turn in one ngspice, and reads the plot each writes.

    ngspice includes the `libraries`, reads the `sections` of others, and reads them
    and the deck in its `compatibility` mode, its random numbers drawn from `seed`.
    `options` replaces the tight options. It runs at jolty's `temperature`, in kelvins,
    its models' nominal temperature at jolty's 27 °C, for at most `timeout` seconds.
    Each command may run several analyses, and a result is read from the plot of each,
    with the time the run took.

    Raises:
        JoltyError: When ngspice is not installed, runs out of time, writes no results,
            runs no analysis, aborts one, or finds an operating point only as the end
            of a transient of fixed length, which it does not check to be one.
    """
    nominal = option_temperature(_core.REFERENCE_TEMPERATURE)
    temperature = running.get("temperature", _core.REFERENCE_TEMPERATURE)
    whole = [
        deck[0],
        *(f".include {library.resolve()}" for library in running.get("libraries", ())),
        *(
            f'.lib "{library.resolve()}" {section}'
            for library, section in running.get("sections", ())
        ),
        *deck[1:],
        running.get("options", OPTIONS),
        f".options temp={option_temperature(temperature)!r} tnom={nominal!r}",
        *(
            f".options seed={seed}"
            for seed in [running.get("seed")]
            if seed is not None
        ),
        ".control",
        "set filetype=ascii",
        "set appendwrite",
        _SHORTEST_TRANSIENT_OPERATING_POINT,
        *(f"{analysis}\nwrite {VALUES} all" for analysis in analyses),
        ".endc",
        ".end",
    ]
    executed = execute(
        whole,
        compatibility=running.get("compatibility"),
        timeout=running.get("timeout", TIMEOUT),
    )
    printed = executed.printed
    # What ngspice printed is read first: a transient it aborted still writes every
    # step it took, which can be hundreds of megabytes of results it has disowned.
    if _ABORTED in printed:
        message = f"ngspice aborted the analysis:\n{printed}"
        raise JoltyError(message)
    if _TRANSIENT_OPERATING_POINT in printed:
        message = (
            "ngspice found no operating point but by a transient of fixed length, "
            f"whose end it does not check to be one:\n{printed}"
        )
        raise JoltyError(message)
    if executed.values is None:
        message = f"ngspice wrote no results:\n{printed}"
        raise JoltyError(message)
    found = read(executed.values, printed)
    if len(found) != len(analyses) or any(
        result.plot == "Constant values" for result in found
    ):
        message = f"ngspice ran no analysis:\n{printed}"
        raise JoltyError(message)
    return tuple(replace(result, seconds=executed.seconds) for result in found)


def read(data: str | bytes, printed: str = "") -> tuple[Result, ...]:
    """Each plot of the raw file `data`, with what ngspice `printed` as it ran.

    The file is SPICE's raw format, ASCII, as text or bytes, or binary, as bytes, its
    numbers little-endian doubles: each plot a header, from its ``Title:`` to its
    ``Values:`` or ``Binary:``, then its points, as many as its ``No. Points:`` says,
    each a value of every variable, a complex one's real part then its imaginary.
    ngspice writes a plot per ``write``, each appended to the last with
    ``appendwrite``.

    Raises:
        JoltyError: When a plot's header lacks what it must state, or its values end
            short of its points.
    """
    raw = data.encode() if isinstance(data, str) else data
    plots: list[Result] = []
    position = raw.find(b"Title:")
    while 0 <= position < len(raw):
        found = _VALUES.search(raw, position)
        if found is None:
            message = "a raw file's plot has no `Values:` or `Binary:`"
            raise JoltyError(message)
        plot, names, points, complex_values = _header(
            raw[position : found.start()].decode()
        )
        width = len(names) * (2 if complex_values else 1)
        if found[1] == b"Binary":
            count = points * width
            table = np.frombuffer(raw, "<f8", count, found.end()).reshape(-1, width)
            position = found.end() + 8 * count
        else:
            table, position = _ascii(raw, found.end(), points, width)
        values = (
            table[:, 0::2] + 1j * table[:, 1::2] if complex_values else table + 0j
        ).astype(np.complex128)
        plots.append(
            Result(
                plot,
                {name: values[:, column] for column, name in enumerate(names)},
                printed,
            )
        )
        position = raw.find(b"Title:", position)
    return tuple(plots)


_VALUES: Final = re.compile(rb"^(Values|Binary):[ \t]*\r?\n", re.MULTILINE)
"""The line that ends a raw file's plot's header, saying how its values are written."""


def _header(header: str) -> tuple[str, list[str], int, bool]:
    """A plot's name, its vectors' names, its points and whether they are complex.

    Raises:
        JoltyError: When the header does not state its points.
    """
    names: list[str] = []
    plot = ""
    points: int | None = None
    complex_values = False
    within = False
    for line in header.splitlines():
        if line.startswith("Plotname:"):
            plot = line.removeprefix("Plotname:").strip()
        elif line.startswith("Flags:"):
            complex_values = "complex" in line.casefold()
        elif line.startswith("No. Points:"):
            points = int(line.removeprefix("No. Points:"))
        elif line.startswith("Variables:"):
            within = True
        elif within and line.strip():
            names.append(_spice_name(line.split()[1].lower()))
    if points is None:
        message = f"a raw file's plot `{plot}` states no `No. Points:`"
        raise JoltyError(message)
    return plot, names, points, complex_values


def _ascii(
    raw: bytes, start: int, points: int, width: int
) -> tuple[npt.NDArray[np.float64], int]:
    """The `points` an ASCII plot's values write from `start`, and where they end.

    Each point is its index, a bare integer, then each variable's value, ``re,im`` for
    a complex one; the table has `width` columns, a complex value's parts two.

    Raises:
        JoltyError: When the values end short of the points.
    """
    numbers: list[float] = []
    position = start
    for _ in range(points):
        index = _TOKEN.search(raw, position)
        position = len(raw) if index is None else index.end()
        row: list[float] = []
        while len(row) < width:
            token = _TOKEN.search(raw, position)
            if token is None:
                message = f"a raw file's values end short of its {points} points"
                raise JoltyError(message)
            position = token.end()
            row.extend(float(part) for part in token[0].split(b","))
        numbers.extend(row)
    return np.array(numbers, dtype=np.float64).reshape(points, width), position


_TOKEN: Final = re.compile(rb"\S+")
"""A word of an ASCII raw file's values: a point's index, or a value."""


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
