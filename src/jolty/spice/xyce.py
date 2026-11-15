# Copyright 2026 Maxence Faldor
"""Xyce, run on a deck of SPICE lines, and the values it prints, read.

Xyce, Sandia's simulator, is under the GPL, so jolty runs it as a separate program and
never links it. [`run`][jolty.spice.xyce.run] runs one analysis of a deck written in
Xyce's dialect, [`XYCE`][jolty.spice.export.XYCE], with tolerances as tight as jolty's,
and reads the values Xyce prints, with the times it reports. Each runs at the
temperature where Xyce's thermal voltage is jolty's,
[`xyce_celsius`][jolty.spice.xyce.xyce_celsius], so the two engines differ by their
devices rather than their physical constants.
"""

import csv
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from jolty import _core
from jolty.errors import JoltyError

EXECUTABLE: Final = shutil.which("Xyce")
"""The Xyce executable on the path, or None when it is not installed."""

_XYCE_BOLTZMANN: Final = 1.3806226e-23
"""Xyce's Boltzmann constant, in joules per kelvin: ``CONSTboltz`` in Xyce 7.10.0's
``src/DeviceModelPKG/Core/N_DEV_Const.h``, line 45, SPICE3's value."""

_XYCE_CHARGE: Final = 1.6021918e-19
"""Xyce's elementary charge, in coulombs: ``CONSTQ`` in the same file, line 43."""

_ZERO_CELSIUS: Final = 273.15
"""Zero degrees Celsius in kelvins, as Xyce takes it: ``CONSTCtoK``, line 44."""

RELATIVE_TOLERANCE: Final = 1e-6
"""The relative tolerance of Xyce's Newton iterations and of its time steps' error, as
jolty's tightest."""

ABSOLUTE_TOLERANCE: Final = 1e-12
"""The absolute tolerance on every unknown of Xyce's Newton iterations, voltages and
currents alike: its own default at an operating point, ``NONLIN``'s ``ABSTOL``
(``N_NLS_NLParams.C``, line 444), below jolty's tightest on voltages."""

RESIDUAL_TOLERANCE: Final = 1e-9
"""The tolerance on the norm of the residual of Xyce's Newton iterations, its
``RHSTOL``, in amperes: the tightest at which every transient of `test_xyce.py`
converges, measured on `dogwood` with Xyce 7.10.0; at 1e-14, the transformer's and the
bipolar amplifier's iterations stop on their rounding, and their steps shrink to
Xyce's least."""

STEP_TOLERANCE: Final = 1e-9
"""The absolute tolerance on every unknown of Xyce's time steps' error: jolty's
tightest on voltages, as Xyce takes one tolerance for voltages and currents."""

OPTIONS: Final = (
    (
        f".options nonlin reltol={RELATIVE_TOLERANCE!r} abstol={ABSOLUTE_TOLERANCE!r}"
        f" rhstol={RESIDUAL_TOLERANCE!r}"
    ),
    (
        f".options nonlin-tran reltol={RELATIVE_TOLERANCE!r}"
        f" abstol={ABSOLUTE_TOLERANCE!r} rhstol={RESIDUAL_TOLERANCE!r}"
    ),
    f".options timeint reltol={RELATIVE_TOLERANCE!r} abstol={STEP_TOLERANCE!r}",
)
"""Xyce's options, as tight as jolty's tightest: its Newton iterations, at an operating
point and in a transient, and its time steps' error.

Xyce's tolerances are not SPICE's: its Newton iteration stops on a weighted norm of
its update and of its residual, and its time steps' error is judged on the unknowns,
not on the charges. These make each as tight as jolty's tightest makes its own.
"""

TIMEOUT: Final = 300
"""The seconds Xyce may run an analysis."""

_SOLVERS: Final = re.compile(r"Total Simulation Solvers Run Time:\s*(\S+) seconds")
"""Where Xyce reports the time of its analyses alone."""

_ELAPSED: Final = re.compile(r"Total Elapsed Run Time:\s*(\S+) seconds")
"""Where Xyce reports the time from its start, parsing included, to its end."""

_PRINTED: Final = {
    ".op": ("dc", ("v(*)", "i(*)")),
    ".dc": ("dc", ("v(*)", "i(*)")),
    ".ac": ("ac", ("v(*)", "i(*)")),
    ".noise": ("noise", ("onoise", "inoise")),
    ".tran": ("tran", ("v(*)", "i(*)")),
}
"""For each analysis Xyce runs, what its ``.print`` line prints, and by default which
values: every node's voltage and every current, or the noise densities."""

_PRECISION: Final = 16
"""The digits Xyce prints after a value's point, 17 significant ones, so that each
number reads back as the same double."""


def xyce_celsius(temperature: float) -> float:
    """Where Xyce's thermal voltage is jolty's at `temperature`, in degrees Celsius.

    The temperature is in kelvins. Running Xyce there, and its models' nominal
    temperature where it is jolty's, compares the devices rather than the physical
    constants: Xyce keeps SPICE3's, which put its thermal voltage 29 parts in a million
    below jolty's at the same kelvins.

    Examples:
        >>> round(xyce_celsius(300.15), 3)
        27.009
    """
    thermal_voltage = temperature * _core.BOLTZMANN / _core.ELEMENTARY_CHARGE
    return thermal_voltage * _XYCE_CHARGE / _XYCE_BOLTZMANN - _ZERO_CELSIUS


type Values = npt.NDArray[np.complex128]


@dataclass(frozen=True)
class Result:
    """The values one Xyce analysis printed, by lowercase name, and its times.

    Its scale is ``time`` in a transient and ``frequency`` in a small-signal analysis;
    a voltage is ``v(node)``, a current ``i(device)``, and the noise densities
    ``onoise`` and ``inoise``, in V²/Hz and in the input's unit squared per hertz.
    """

    vectors: dict[str, Values]
    printed: str
    """What Xyce printed while it ran."""
    solvers: float
    """Xyce's time for its analyses alone, in seconds, as it reports it."""
    elapsed: float
    """Xyce's time from its start to its end, parsing included, in seconds, as it
    reports it."""
    seconds: float
    """The time from starting Xyce's process to its exit, in seconds, as jolty
    measures it."""

    def real(self, name: str) -> npt.NDArray[np.float64]:
        """The real part of the vector `name`."""
        return self.vectors[name].real


def run(  # noqa: PLR0913 - the deck, its analysis, and how Xyce runs it
    lines: list[str],
    analysis: str,
    *,
    printed: Sequence[str] | None = None,
    options: Sequence[str] = OPTIONS,
    temperature: float = _core.REFERENCE_TEMPERATURE,
    launch: Sequence[str] | None = None,
    timeout: float | None = TIMEOUT,
) -> Result:
    """Runs one analysis command in Xyce and reads the values it prints.

    The `analysis` is Xyce's own command, such as ``.dc V1 0 1 0.1``; Xyce prints the
    values `printed`, by default every node's voltage and every current, or a noise
    analysis's densities. `options` replace the tight options. It runs at jolty's
    `temperature`, in kelvins, its models' nominal temperature at jolty's 27 °C, by
    the command `launch`, the serial Xyce on the path unless given, such as
    ``mpirun -np 4 Xyce`` for its parallel build, for at most `timeout` seconds, or as
    long as it takes if None.

    Raises:
        JoltyError: When Xyce is not installed, runs out of time, fails, or prints no
            values.
    """
    command = analysis.split(maxsplit=1)[0].lower()
    if command not in _PRINTED:
        message = f"Xyce has no analysis {command}"
        raise JoltyError(message)
    kind, every = _PRINTED[command]
    if launch is None:
        if EXECUTABLE is None:
            message = "Xyce is not installed"
            raise JoltyError(message)
        launch = (EXECUTABLE,)
    nominal = xyce_celsius(_core.REFERENCE_TEMPERATURE)
    deck = [
        *lines,
        *options,
        f".options device temp={xyce_celsius(temperature)!r} tnom={nominal!r}",
        analysis,
        f".print {kind} format=csv precision={_PRECISION} file=result.csv "
        + " ".join(every if printed is None else printed),
        ".end",
    ]
    executed = execute(deck, [*launch], timeout=timeout)
    if executed.returncode != 0 or executed.values is None:
        message = (
            f"Xyce failed, exiting with {executed.returncode}:\n{executed.printed}"
        )
        raise JoltyError(message)
    vectors = _read(executed.values)
    solvers = _SOLVERS.search(executed.printed)
    elapsed = _ELAPSED.search(executed.printed)
    if not vectors or solvers is None or elapsed is None:
        message = f"Xyce printed no values:\n{executed.printed}"
        raise JoltyError(message)
    return Result(
        vectors,
        executed.printed,
        float(solvers.group(1)),
        float(elapsed.group(1)),
        executed.seconds,
    )


@dataclass(frozen=True)
class Executed:
    """What one run of Xyce on a deck left.

    What it printed, the values file it wrote, if any, its exit status, and its time
    from its start to its exit.
    """

    printed: str
    values: str | None
    returncode: int
    seconds: float


def execute(
    deck: Sequence[str], launch: Sequence[str], *, timeout: float | None
) -> Executed:
    """Runs Xyce by the command `launch` on the lines `deck`, in a directory of its own.

    The deck is written to ``circuit.cir`` in a temporary directory, which the run
    takes as its own, so the file its ``.print`` names, ``result.csv``, is read back
    from there; the run is stopped after `timeout` seconds, or never if None.

    Raises:
        JoltyError: When it runs out of time.
    """
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        (directory / "circuit.cir").write_text("\n".join(deck) + "\n", encoding="utf-8")
        started = time.perf_counter()
        try:
            done = subprocess.run(  # noqa: S603 - Xyce runs the deck written here
                [*launch, "circuit.cir"],
                capture_output=True,
                check=False,
                text=True,
                timeout=timeout,
                cwd=directory,
            )
        except subprocess.TimeoutExpired:
            message = f"Xyce ran for more than {timeout:g} s"
            raise JoltyError(message) from None
        seconds = time.perf_counter() - started
        values = directory / "result.csv"
        written = values.read_text(encoding="utf-8") if values.is_file() else None
    return Executed(done.stdout + done.stderr, written, done.returncode, seconds)


_SCALES: Final = {"time": "time", "freq": "frequency"}
"""The names of the scales Xyce prints, and jolty's."""


def _read(text: str) -> dict[str, Values]:
    """The columns of the values Xyce printed as CSV, by lowercase name.

    A complex value's real and imaginary parts, ``Re(V(OUT))`` and ``Im(V(OUT))``,
    make one vector.
    """
    rows = list(csv.reader(text.splitlines()))
    if len(rows) < 2:  # noqa: PLR2004 - a header and a row of values at least
        return {}
    header, *data = (row for row in rows if row)
    table = np.array([[float(value) for value in row] for row in data])
    vectors: dict[str, Values] = {}
    for column, title in enumerate(header):
        name = title.strip().lower()
        part = re.fullmatch(r"(re|im)\((.*)\)", name)
        values = table[:, column].astype(np.complex128)
        if part is None:
            vectors[_SCALES.get(name, name)] = values
            continue
        kind, quantity = part.groups()
        found = vectors.setdefault(quantity, np.zeros(len(data), np.complex128))
        found += values if kind == "re" else 1j * values
    return vectors
