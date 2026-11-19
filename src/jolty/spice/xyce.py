# Copyright 2026 Maxence Faldor
"""Subprocess runner and CSV output parser for Sandia Xyce SPICE decks.

Xyce is GPL-licensed, so jolty invokes it as an external executable and never links it.
[`run`][jolty.spice.xyce.run] executes one analysis in Xyce dialect
([`XYCE`][jolty.spice.export.XYCE]) with tolerances matched to jolty and parses the
printed CSV vectors and timing lines. Each run uses
[`option_temperature`][jolty.spice.xyce.option_temperature] to align thermal voltages
so differences reflect device models rather than physical constants.

Benchmark suites integrate Xyce through `run` using decks exported by
`jolty.spice.export.netlist` (with ``options=()`` for Xyce default tolerances). Failed
runs raise `JoltyError` containing Xyce output; successful runs return a `Result` with
vectors, wall time, and reported solver time. [`execute`][jolty.spice.xyce.execute]
runs arbitrary pre-assembled decks directly.
"""

import csv
import functools
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from jolty import _core
from jolty.errors import JoltyError
from jolty.spice import _process

EXECUTABLE: Final = shutil.which("Xyce")
"""The Xyce executable on the path, or None when it is not installed."""

_XYCE_THERMAL_VOLTAGE_PER_KELVIN: Final = 1.3806226e-23 / 1.6021918e-19
"""Xyce's Boltzmann constant over elementary charge: ``CONSTboltz`` / ``CONSTQ`` in
Xyce 7.10.0's ``src/DeviceModelPKG/Core/N_DEV_Const.h``, lines 45 and 43 (SPICE3)."""

RELATIVE_TOLERANCE: Final = 1e-6
"""The relative tolerance of Xyce's Newton iterations and of its time steps' error, as
jolty's tightest."""

ABSOLUTE_TOLERANCE: Final = 1e-12
"""Absolute tolerance on every unknown in Xyce Newton iterations.

Matches Xyce's default operating-point ``NONLIN`` ``ABSTOL`` (``N_NLS_NLParams.C``,
line 444), tighter than jolty's voltage tolerance.
"""

RESIDUAL_TOLERANCE: Final = 1e-9
"""Newton residual norm tolerance (``RHSTOL``) in amperes for Xyce runs.

Tightest setting at which every transient in `test_xyce.py` converges on Xyce 7.10.0;
at 1e-14, transformer and BJT amplifier steps stall on floating-point rounding.
"""

STEP_TOLERANCE: Final = 1e-9
"""Absolute time-integration error tolerance matching jolty's tightest voltage bound."""

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
"""Xyce tolerance options aligned with jolty's tightest Newton and time-step bounds.

Xyce stops Newton iterations on weighted update and residual norms and bounds time-step
error on unknowns rather than charges; the values above match jolty's tight tolerances.
"""

TIMEOUT: Final = _process.TIMEOUT
"""Default timeout in seconds for a Xyce subprocess."""

_ANALYSIS_SECONDS: Final = re.compile(
    r"Total Simulation Solvers Run Time:\s*(\S+) seconds"
)
"""Pattern extracting reported solver-only runtime from Xyce stdout."""

_ELAPSED: Final = re.compile(r"Total Elapsed Run Time:\s*(\S+) seconds")
"""Pattern extracting reported total elapsed runtime (including parsing) from stdout."""

_PRINTED: Final = {
    ".op": ("dc", ("v(*)", "i(*)")),
    ".dc": ("dc", ("v(*)", "i(*)")),
    ".ac": ("ac", ("v(*)", "i(*)")),
    ".noise": ("noise", ("onoise", "inoise")),
    ".tran": ("tran", ("v(*)", "i(*)")),
}
"""Mapping from Xyce analysis command to ``.print`` kind and default output columns."""

_PRECISION: Final = 16
"""Decimal fractional digits printed (17 significant digits for IEEE-754 doubles)."""


@functools.cache
def version() -> str | None:
    """Returns the installed Xyce version banner, or None when Xyce is unavailable.

    Examples:
        >>> version() is None or "Xyce" in version()
        True
    """
    if EXECUTABLE is None:
        return None
    lines = _process.probe_version(EXECUTABLE, "-v").strip().splitlines()
    return lines[0] if lines else ""


def option_temperature(temperature: float) -> float:
    """Computes Celsius temperature matching jolty thermal voltage at `temperature`.

    Accepts `temperature` in kelvins. Xyce uses SPICE3 constants whose thermal voltage
    sits 29 ppm below jolty at equal kelvins, so adjusting simulation and nominal
    temperatures isolates device-model differences from physical constants.

    Examples:
        >>> round(option_temperature(300.15), 3)
        27.009
    """
    return _process.thermal_option_temperature(
        temperature, _XYCE_THERMAL_VOLTAGE_PER_KELVIN
    )


type Values = npt.NDArray[np.complex128]


@dataclass(frozen=True)
class Result:
    """Vectors and timing measurements parsed from one Xyce analysis run.

    Scale vectors are ``time`` for transients and ``frequency`` for AC analyses. Node
    voltages use ``v(node)``, branch currents use ``i(device)``, and noise densities
    use ``onoise`` and ``inoise`` in squared amplitude per hertz.
    """

    vectors: dict[str, Values]
    printed: str
    """Combined stdout and stderr printed by Xyce during the run."""
    analysis_seconds: float
    """Solver-only runtime reported by Xyce, in seconds."""
    elapsed_seconds: float
    """Total runtime including parsing reported by Xyce, in seconds."""
    seconds: float
    """Wall-clock process runtime measured by jolty, in seconds."""

    def real(self, name: str) -> npt.NDArray[np.float64]:
        """Returns the real part of the vector `name`."""
        return self.vectors[name].real


def run(  # noqa: PLR0913 - the deck, its analysis, and how Xyce runs it
    deck: list[str],
    analysis: str,
    *,
    printed: Sequence[str] | None = None,
    options: Sequence[str] = OPTIONS,
    temperature: float = _core.REFERENCE_TEMPERATURE,
    launch: Sequence[str] | None = None,
    timeout: float | None = TIMEOUT,
) -> Result:
    """Runs one Xyce analysis command on `deck` and parses its printed CSV values.

    `analysis` is a Xyce analysis card such as ``.dc V1 0 1 0.1``. Xyce prints the
    columns in `printed` (defaulting to all node voltages and branch currents, or noise
    densities). `options` replaces the default tight tolerances. The run aligns thermal
    voltage with `temperature` (in kelvins) and nominal 27 °C, invokes `launch`
    (defaulting to serial ``Xyce``, or e.g. ``mpirun -np 4 Xyce``), and enforces
    `timeout` seconds unless None.

    Raises:
        JoltyError: When Xyce is unavailable, times out, fails, or prints no values.
    """
    parts = analysis.split(maxsplit=1)
    if not parts:
        message = "Xyce analysis command is empty"
        raise JoltyError(message)
    command = parts[0].lower()
    if command not in _PRINTED:
        message = f"Xyce has no analysis {command}"
        raise JoltyError(message)
    kind, every = _PRINTED[command]
    nominal = option_temperature(_core.REFERENCE_TEMPERATURE)
    whole = [
        *deck,
        *options,
        f".options device temp={option_temperature(temperature)!r} tnom={nominal!r}",
        analysis,
        f".print {kind} format=csv precision={_PRECISION} file={VALUES} "
        + " ".join(every if printed is None else printed),
        ".end",
    ]
    executed = execute(whole, launch=launch, timeout=timeout)
    if executed.code != 0 or executed.values is None:
        message = f"Xyce failed, exiting with {executed.code}:\n{executed.printed}"
        raise JoltyError(message)
    vectors = _read(executed.values)
    analysis_seconds = _ANALYSIS_SECONDS.search(executed.printed)
    elapsed_seconds = _ELAPSED.search(executed.printed)
    if not vectors or analysis_seconds is None or elapsed_seconds is None:
        message = f"Xyce printed no values:\n{executed.printed}"
        raise JoltyError(message)
    return Result(
        vectors,
        executed.printed,
        float(analysis_seconds.group(1)),
        float(elapsed_seconds.group(1)),
        executed.seconds,
    )


class XyceTimeoutError(JoltyError):
    """Xyce ran for longer than it was given."""


@dataclass(frozen=True, slots=True)
class Executed(_process.Executed[str]):
    """Raw output, exit code, elapsed wall time, and CSV text from one Xyce run."""


VALUES: Final = "result.csv"
"""CSV filename written by decks executed via [`execute`][jolty.spice.xyce.execute]."""


def execute(
    deck: Sequence[str],
    *,
    launch: Sequence[str] | None = None,
    timeout: float | None = TIMEOUT,
) -> Executed:
    """Runs Xyce via `launch` on `deck` inside a dedicated temporary directory.

    Writes `deck` to ``circuit.cir`` in a fresh temporary working directory and reads
    back `VALUES` written by ``.print``. Enforces `timeout` seconds unless None.
    Defaults `launch` to serial ``Xyce`` on PATH (supporting wrappers such as
    ``mpirun -np 4 Xyce``). Serves as the single Xyce process entry point parallel to
    [`ngspice.execute`][jolty.spice.ngspice.execute].

    Raises:
        JoltyError: When Xyce is unavailable and no `launch` command is supplied.
        XyceTimeoutError: When the process exceeds `timeout` seconds.
    """
    if launch is None:
        if EXECUTABLE is None:
            message = "Xyce is not installed"
            raise JoltyError(message)
        launch = (EXECUTABLE,)
    ran = _process.run_deck(
        "Xyce",
        launch,
        deck,
        VALUES,
        _process.read_text_utf8,
        timeout=timeout,
        timeout_error=XyceTimeoutError,
    )
    return Executed(ran.printed, ran.code, ran.seconds, ran.values)


_SCALES: Final = {"time": "time", "freq": "frequency"}
"""Mapping from Xyce CSV scale column headers to jolty scale names."""

_COMPLEX_PART: Final = re.compile(r"(re|im)\((.*)\)")
"""Pattern matching a case-folded complex CSV column header."""


def _read(text: str) -> dict[str, Values]:
    """Parses CSV columns printed by Xyce into complex vectors keyed by lowercase name.

    Combines matching ``Re(...)`` and ``Im(...)`` columns into a single complex vector.
    """
    rows = [row for row in csv.reader(text.splitlines()) if row]
    if len(rows) < 2:  # noqa: PLR2004 - a header and a row of values at least
        return {}
    header, *data = rows
    table = np.array([[float(value) for value in row] for row in data])
    vectors: dict[str, Values] = {}
    for column, title in enumerate(header):
        name = title.strip().lower()
        part = _COMPLEX_PART.fullmatch(name)
        values = table[:, column].astype(np.complex128)
        if part is None:
            vectors[_SCALES.get(name, name)] = values
            continue
        kind, quantity = part.groups()
        found = vectors.setdefault(quantity, np.zeros(len(data), np.complex128))
        found += values if kind == "re" else 1j * values
    return vectors
