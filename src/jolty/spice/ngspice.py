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

__lazy_modules__ = ["numpy", "numpy.typing"]

import functools
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal, TypedDict, Unpack

import numpy as np
import numpy.typing as npt

from jolty import _core
from jolty.errors import JoltyError
from jolty.spice import _process
from jolty.units import fA, nV

EXECUTABLE: Final = shutil.which("ngspice")
"""The ngspice executable on the path, or None when it is not installed."""

RELATIVE_TOLERANCE: Final = 1e-6
"""The relative tolerance ngspice runs with."""

VOLTAGE_TOLERANCE: Final = 1 * nV
"""The absolute tolerance on voltages ngspice runs with."""

CURRENT_TOLERANCE: Final = 10 * fA
"""The absolute tolerance on currents ngspice runs with."""

OPTIONS: Final = (
    f".options reltol={RELATIVE_TOLERANCE!r} vntol={VOLTAGE_TOLERANCE.value!r}"
    f" abstol={CURRENT_TOLERANCE.value!r}"
)
"""ngspice's options, as tight as jolty's tightest, the charge tolerance both engines'
default."""

NO_OPTIONS: Final = "*"
"""The options line that leaves every option at ngspice's default: a comment."""

_SPICE_THERMAL_VOLTAGE_PER_KELVIN: Final = 1.38064852e-23 / 1.6021766208e-19
"""ngspice's Boltzmann constant over the elementary charge, from CODATA 2014."""

type Real = npt.NDArray[np.float64]
type Complex = npt.NDArray[np.complex128]


def option_temperature(temperature: float) -> float:
    """Computes Celsius temperature matching jolty thermal voltage at `temperature`.

    Accepts `temperature` in kelvins. Adjusting simulation and nominal temperatures
    aligns thermal voltages so comparisons reflect device models rather than constants.

    Args:
        temperature: Target jolty temperature in kelvins.

    Returns:
        Celsius ``.options temp`` value for ngspice with matching thermal voltage.
    """
    return _process.thermal_option_temperature(
        temperature, _SPICE_THERMAL_VOLTAGE_PER_KELVIN
    )


def same_kelvins(temperature: float) -> float:
    """Computes the input to [`run`][jolty.spice.ngspice.run] running at `temperature`.

    Inverts `option_temperature` so ngspice executes at exact `temperature` in kelvins,
    which matches models such as BSIM3 that embed fixed Boltzmann and charge constants.

    Args:
        temperature: Desired exact ngspice temperature in kelvins.

    Returns:
        Adjusted kelvin value whose `option_temperature` equals `temperature` in
        Celsius.

    Examples:
        >>> round(option_temperature(same_kelvins(300.15)), 9)
        27.0
    """
    jolty = _core.BOLTZMANN / _core.ELEMENTARY_CHARGE
    return temperature * _SPICE_THERMAL_VOLTAGE_PER_KELVIN / jolty


@dataclass(frozen=True)
class Result:
    """Vectors from one ngspice analysis plot keyed by lowercase name.

    Retains combined stdout and stderr printed by ngspice during the run.
    """

    plot: str
    vectors: dict[str, Complex]
    printed: str
    seconds: float | None = None
    """Wall-clock runtime of the ngspice process in seconds, or None for file reads."""

    def real(self, name: str) -> Real:
        """Returns the real part of the vector `name`.

        Args:
            name: Lowercase vector name in `vectors` (such as ``"v(out)"``).

        Returns:
            Real-part `float64` array of the named vector.
        """
        return self.vectors[name].real


_TRANSIENT_OPERATING_POINT: Final = "Transient op started"
"""Marker printed when ngspice falls back from DC solvers to a transient op."""

_ABORTED: Final = "simulation(s) aborted"
"""Marker printed when ngspice aborts an analysis before completion."""

_SHORTEST_TRANSIENT_OPERATING_POINT: Final = "optran 1 1 1 1n 1n 0"
"""Control directive capping fallback transient op at 1 ns (fallback is rejected)."""


TIMEOUT: Final = _process.TIMEOUT
"""Default timeout in seconds for an ngspice subprocess."""


type Compatibility = Literal["ps", "psa", "hsa"]
"""ngspice compatibility mode: PSpice (``ps``/``psa``) or HSPICE (``hsa`` for PDKs)."""


class Running(TypedDict, total=False):
    """Keyword configuration options for ngspice deck execution."""

    libraries: tuple[Path, ...]
    """Library files included via ``.include``."""
    sections: tuple[tuple[Path, str], ...]
    """Library ``(path, section)`` pairs included via ``.lib``, such as PDK corners."""
    compatibility: Compatibility | None
    """ngspice compatibility mode, or None for default behavior."""
    options: str
    """SPICE ``.options`` line, defaulting to jolty's tightest tolerances."""
    temperature: float
    """Simulation temperature in kelvins, defaulting to 27 °C."""
    seed: int | None
    """Optional RNG seed for ngspice statistical mismatch draws."""
    timeout: float | None
    """Subprocess timeout in seconds (`TIMEOUT` by default, or None for unlimited)."""


class NgspiceTimeoutError(JoltyError):
    """Error raised when an ngspice subprocess exceeds its timeout."""


@dataclass(frozen=True, slots=True)
class Executed(_process.Executed[bytes]):
    """Raw output, exit code, wall time, and raw-file bytes from one ngspice run."""


VALUES: Final = "result.raw"
"""Raw filename written by decks run via [`execute`][jolty.spice.ngspice.execute]."""


@functools.cache
def version() -> str | None:
    """Returns the installed ngspice version and build banner, or None if unavailable.

    Returns:
        Semicolon-joined ngspice version and build lines, or `None` when uninstalled.

    Examples:
        >>> version() is None or version().startswith("ngspice")
        True
    """
    if EXECUTABLE is None:
        return None
    lines = [
        line.strip("* ")
        for line in _process.probe_version(EXECUTABLE, "--version").splitlines()
    ]
    return "; ".join(line for line in lines if line.startswith(("ngspice", "Compiled")))


def execute(
    deck: Sequence[str],
    *,
    compatibility: Compatibility | None = None,
    timeout: float | None = TIMEOUT,
) -> Executed:
    """Runs ngspice in batch mode on `deck` inside a dedicated temporary directory.

    Writes `deck` to ``circuit.cir`` (plus ``.spiceinit`` when `compatibility` is set),
    reads back `VALUES` written by ``write``, and enforces `timeout` seconds unless
    None. Serves as the single ngspice process entry point parallel to
    [`xyce.execute`][jolty.spice.xyce.execute].

    Args:
        deck: Complete SPICE deck lines written to ``circuit.cir``.
        compatibility: Optional ngspice compatibility mode written to ``.spiceinit``.
        timeout: Subprocess timeout in seconds, or `None` for no limit.

    Returns:
        `Executed` record containing stdout/stderr, exit code, wall time, and raw bytes.

    Raises:
        JoltyError: When ngspice is unavailable.
        NgspiceTimeoutError: When ngspice exceeds `timeout` seconds.
    """
    if EXECUTABLE is None:
        message = "ngspice is not installed"
        raise JoltyError(message)
    extra = (
        {".spiceinit": f"set ngbehavior={compatibility}\n"}
        if compatibility is not None
        else {}
    )
    ran = _process.run_deck(
        "ngspice",
        (EXECUTABLE, "-b"),
        deck,
        VALUES,
        Path.read_bytes,
        timeout=timeout,
        timeout_error=NgspiceTimeoutError,
        extra_files=extra,
    )
    return Executed(ran.printed, ran.code, ran.seconds, ran.values)


def run(
    deck: list[str],
    analysis: str,
    **running: Unpack[Running],
) -> Result:
    """Runs one analysis command in ngspice and parses the resulting raw plot.

    Accepts the same `running` options as [`runs`][jolty.spice.ngspice.runs].

    Args:
        deck: SPICE netlist lines starting with a title line.
        analysis: Interactive ngspice analysis command (such as ``"op"`` or
            ``"tran 1n 1u"``).
        **running: Optional library, corner, tolerance, temperature, seed, and timeout
            settings.

    Returns:
        Parsed `Result` plot and runtime for `analysis`.

    Raises:
        JoltyError: Under the same failure conditions as `runs`.
    """
    (result,) = runs(deck, (analysis,), **running)
    return result


def runs(
    deck: list[str],
    analyses: Sequence[str],
    **running: Unpack[Running],
) -> tuple[Result, ...]:
    """Runs multiple analysis commands sequentially in one ngspice process.

    Includes configured `libraries` and `sections`, sets `compatibility` and `seed`,
    replaces default tight tolerances with `options`, aligns thermal voltage with
    `temperature` (in kelvins) and nominal 27 °C, and enforces `timeout`. Returns one
    parsed `Result` per plot along with the process runtime.

    Args:
        deck: SPICE netlist lines starting with a title line.
        analyses: Interactive ngspice analysis commands executed in sequence.
        **running: Optional library, corner, tolerance, temperature, seed, and timeout
            settings.

    Returns:
        Tuple of parsed `Result` plots in execution order.

    Raises:
        JoltyError: When `deck` is empty, ngspice is unavailable, times out, writes no
            results, runs no analysis, aborts, or falls back to transient op.
    """
    if not deck:
        message = "ngspice deck is empty; a SPICE deck begins with a title line"
        raise JoltyError(message)
    nominal = option_temperature(_core.REFERENCE_TEMPERATURE)
    temperature = running.get("temperature", _core.REFERENCE_TEMPERATURE)
    whole = [
        deck[0],
        *(
            f'.include "{library.resolve()}"'
            for library in running.get("libraries", ())
        ),
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


@dataclass(frozen=True, slots=True)
class RawVariable:
    """One variable declared in a SPICE raw file header, exact spelling preserved."""

    name: str
    kind: str


@dataclass(frozen=True, slots=True)
class RawPlot:
    """One plot parsed verbatim from a SPICE raw file.

    `title` is the text after ``Title:``, `plot` after ``Plotname:``, `variables` the
    declared variables in column order, `complex_values` whether ``Flags:`` marks the
    plot complex, and `values` a ``(points, len(variables))`` array of `float64` or
    `complex128` preserving exact bits for binary files.
    """

    title: str
    plot: str
    variables: tuple[RawVariable, ...]
    complex_values: bool
    values: Real | Complex


@dataclass(frozen=True, slots=True)
class _ParsedHeader:
    """Parsed header fields of one SPICE raw plot."""

    title: str
    plot: str
    variables: tuple[RawVariable, ...]
    points: int
    complex_values: bool


def parse_raw(data: str | bytes) -> tuple[RawPlot, ...]:
    """Parses every plot of the SPICE raw file `data`, preserving case and types.

    Supports both ASCII (``Values:``) and binary (``Binary:``, little-endian IEEE-754
    doubles) plots, including multi-plot files written with ``appendwrite`` or
    [`save_raw`][jolty.files.save_raw].

    Args:
        data: Raw file payload as UTF-8 text or raw bytes.

    Returns:
        Parsed `RawPlot` records in file order.

    Raises:
        JoltyError: When `data` holds malformed headers, mismatched variable counts,
            truncated binary or ASCII values, non-numeric ASCII tokens, or trailing
            non-whitespace data outside plots.
    """
    raw = data.encode() if isinstance(data, str) else data
    plots: list[RawPlot] = []
    position = _skip_whitespace(raw, 0)
    while position < len(raw):
        if not raw.startswith(b"Title:", position):
            message = "a raw file's plot begins with `Title:`"
            raise JoltyError(message)
        found = _VALUES.search(raw, position)
        if found is None:
            message = "a raw file's plot has no `Values:` or `Binary:`"
            raise JoltyError(message)
        try:
            header_text = raw[position : found.start()].decode("utf-8")
        except UnicodeDecodeError:
            message = "a raw file's header is not valid UTF-8 text"
            raise JoltyError(message) from None
        parsed = _header(header_text)
        variable_count = len(parsed.variables)
        if found[1] == b"Binary":
            values, position = _binary(
                raw,
                found.end(),
                parsed.points,
                variable_count,
                complex_values=parsed.complex_values,
            )
        else:
            table, position = _ascii(
                raw,
                found.end(),
                parsed.points,
                variable_count,
                complex_values=parsed.complex_values,
            )
            values = _typed_values(
                table,
                parsed.points,
                variable_count,
                complex_values=parsed.complex_values,
            )
        plots.append(
            RawPlot(
                title=parsed.title,
                plot=parsed.plot,
                variables=parsed.variables,
                complex_values=parsed.complex_values,
                values=values,
            )
        )
        position = _skip_whitespace(raw, position)
    return tuple(plots)


def _skip_whitespace(raw: bytes, position: int) -> int:
    """Returns the first non-whitespace byte index in `raw` at or after `position`."""
    match = _NON_WHITESPACE.search(raw, position)
    return len(raw) if match is None else match.start()


_NON_WHITESPACE: Final = re.compile(rb"\S")
"""Matches the next non-whitespace byte in a raw file."""


_DOUBLE_BYTES: Final = 8
"""Byte width of a little-endian IEEE-754 double (`<f8`) in a binary raw file."""

_COMPLEX_BYTES: Final = 16
"""Byte width of a little-endian complex double pair (`<c16`) in a binary raw file."""


def _binary(
    raw: bytes,
    start: int,
    points: int,
    variable_count: int,
    *,
    complex_values: bool,
) -> tuple[npt.NDArray[np.float64] | npt.NDArray[np.complex128], int]:
    """Reads `points` binary little-endian rows from `raw` at `start`."""
    count = points * variable_count
    item_bytes = _COMPLEX_BYTES if complex_values else _DOUBLE_BYTES
    expected_bytes = item_bytes * count
    if start + expected_bytes > len(raw):
        message = (
            f"a raw file's binary values end short of its {points} points "
            f"({len(raw) - start} bytes available, {expected_bytes} required)"
        )
        raise JoltyError(message)
    if complex_values:
        if count == 0:
            return np.empty((points, variable_count), dtype=np.complex128), start
        values: npt.NDArray[np.float64] | npt.NDArray[np.complex128] = (
            np.frombuffer(raw, "<c16", count, start)
            .reshape(points, variable_count)
            .astype(np.complex128, copy=True)
        )
    else:
        if count == 0:
            return np.empty((points, variable_count), dtype=np.float64), start
        values = (
            np.frombuffer(raw, "<f8", count, start)
            .reshape(points, variable_count)
            .astype(np.float64, copy=True)
        )
    return values, start + expected_bytes


def _typed_values(
    table: npt.NDArray[np.float64],
    points: int,
    variable_count: int,
    *,
    complex_values: bool,
) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
    """Packs flat real columns into `float64` or exact-bit `complex128` arrays."""
    if not complex_values:
        return table
    packed = np.empty((points, variable_count), dtype=np.complex128)
    packed.real = table[:, 0::2]
    packed.imag = table[:, 1::2]
    return packed


def read(data: str | bytes, printed: str = "") -> tuple[Result, ...]:
    """Parses all plots in SPICE raw `data` into `Result` objects with `printed` text.

    Supports both ASCII and binary (little-endian double) SPICE raw formats, including
    multi-plot files appended by ``appendwrite``.

    Args:
        data: Raw file payload as UTF-8 text or raw bytes.
        printed: Combined stdout and stderr text attached to each returned `Result`.

    Returns:
        Parsed `Result` objects with normalized SPICE vector names in file order.

    Raises:
        JoltyError: When a plot header is malformed or values end before ``No. Points``.
    """
    plots: list[Result] = []
    for raw_plot in parse_raw(data):
        values = raw_plot.values.astype(np.complex128)
        vectors = {
            _spice_name(variable.name.lower()): values[:, column]
            for column, variable in enumerate(raw_plot.variables)
        }
        plots.append(Result(raw_plot.plot, vectors, printed))
    return tuple(plots)


_VALUES: Final = re.compile(rb"^(Values|Binary):[ \t]*(?:\r?\n|$)", re.MULTILINE)
"""Pattern matching the header terminator marking ASCII or binary plot values."""


def _parse_count(fields: dict[str, str], key: str, label: str, plot: str) -> int:
    """Parses a required non-negative integer header field, or raises `JoltyError`."""
    if key not in fields:
        message = f"a raw file's plot `{plot}` states no `{label}`"
        raise JoltyError(message)
    value = fields[key].strip()
    try:
        count = int(value)
    except ValueError:
        message = f"a raw file's plot `{plot}` has invalid `{label}`: `{value}`"
        raise JoltyError(message) from None
    if count < 0:
        message = f"a raw file's plot `{plot}` has negative `{label}`: {count}"
        raise JoltyError(message)
    return count


def _parse_variable(line: str, plot: str, expected_index: int) -> RawVariable:
    """Parses one line under ``Variables:``, ``index name type``."""
    parts = line.split()
    if len(parts) < 3:  # noqa: PLR2004 - index, name, type
        message = (
            f"a raw file's plot `{plot}` has a malformed variable entry: "
            f"`{line.strip()}`"
        )
        raise JoltyError(message)
    try:
        index = int(parts[0])
    except ValueError:
        message = (
            f"a raw file's plot `{plot}` has non-integer variable index `{parts[0]}`"
        )
        raise JoltyError(message) from None
    if index != expected_index:
        message = (
            f"a raw file's plot `{plot}` lists variable index {index} at position "
            f"{expected_index}"
        )
        raise JoltyError(message)
    return RawVariable(name=parts[1], kind=parts[2].casefold())


def _header(header: str) -> _ParsedHeader:
    """Parses a plot's header, checking counts and flags.

    Raises:
        JoltyError: When required fields are missing or counts disagree.
    """
    fields, variables = _split_header(header)
    plot = fields.get("plotname", "")
    if not plot:
        message = "a raw file's plot states no `Plotname:`"
        raise JoltyError(message)
    flag_text = fields.get("flags")
    modes = (
        set(flag_text.casefold().split()) & {"real", "complex"}
        if flag_text is not None
        else set()
    )
    if len(modes) != 1:
        message = (
            f"a raw file's plot `{plot}` states neither `real` nor `complex` "
            "exclusively in `Flags:`"
        )
        raise JoltyError(message)
    expected_variables = _parse_count(fields, "no. variables", "No. Variables:", plot)
    points = _parse_count(fields, "no. points", "No. Points:", plot)
    parsed_variables = tuple(
        _parse_variable(line, plot, index) for index, line in enumerate(variables)
    )
    if len(parsed_variables) != expected_variables:
        message = (
            f"a raw file's plot `{plot}` declares {expected_variables} variables, "
            f"but lists {len(parsed_variables)}"
        )
        raise JoltyError(message)
    return _ParsedHeader(
        title=fields.get("title", ""),
        plot=plot,
        variables=parsed_variables,
        points=points,
        complex_values="complex" in modes,
    )


def _split_header(header: str) -> tuple[dict[str, str], list[str]]:
    """Splits a raw header into key-value header lines and ``Variables:`` entries."""
    fields: dict[str, str] = {}
    variables: list[str] = []
    within = False
    for line in header.splitlines():
        if within:
            if line.strip():
                variables.append(line)
            continue
        if line.startswith("Variables:"):
            within = True
            continue
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip().casefold()] = value.strip()
    return fields, variables


def _ascii(
    raw: bytes,
    start: int,
    points: int,
    variable_count: int,
    *,
    complex_values: bool,
) -> tuple[npt.NDArray[np.float64], int]:
    """Reads `points` rows of ASCII raw values from `raw` starting at `start`.

    Each row starts with an integer point index followed by `variable_count` values
    (``re,im`` pairs when `complex_values` is true). Returns the table and end offset.

    Raises:
        JoltyError: When values end early or contain malformed numeric tokens.
    """
    width = variable_count * (2 if complex_values else 1)
    numbers: list[float] = []
    position = start
    for point_index in range(points):
        index_token = _TOKEN.search(raw, position)
        if index_token is None:
            message = f"a raw file's values end short of its {points} points"
            raise JoltyError(message)
        try:
            row_index = int(index_token[0])
        except ValueError:
            found_text = index_token[0].decode("utf-8", errors="replace")
            message = (
                f"a raw file's point {point_index} begins with `{found_text}`, "
                "not an integer index"
            )
            raise JoltyError(message) from None
        if row_index != point_index:
            message = (
                f"a raw file's point {point_index} has row index {row_index}, "
                f"not {point_index}"
            )
            raise JoltyError(message)
        position = index_token.end()
        row, position = _ascii_row(
            raw, position, points, variable_count, complex_values=complex_values
        )
        numbers.extend(row)
    return np.array(numbers, dtype=np.float64).reshape(points, width), position


def _ascii_row(
    raw: bytes,
    position: int,
    points: int,
    variable_count: int,
    *,
    complex_values: bool,
) -> tuple[list[float], int]:
    """Reads one point's `variable_count` ASCII values from `raw` at `position`."""
    row: list[float] = []
    for _ in range(variable_count):
        token = _TOKEN.search(raw, position)
        if token is None:
            message = f"a raw file's values end short of its {points} points"
            raise JoltyError(message)
        position = token.end()
        word = token[0]
        parts = word.split(b",")
        if complex_values:
            if len(parts) != 2:  # noqa: PLR2004 - real and imaginary parts
                text = word.decode("utf-8", errors="replace")
                message = f"a complex raw value is `real,imag`, not `{text}`"
                raise JoltyError(message)
        elif len(parts) != 1:
            text = word.decode("utf-8", errors="replace")
            message = f"a real raw value is a single number, not `{text}`"
            raise JoltyError(message)
        for part in parts:
            try:
                row.append(float(part))
            except ValueError:
                text = part.decode("utf-8", errors="replace")
                message = f"a raw file's value `{text}` is not a number"
                raise JoltyError(message) from None
    return row, position


_TOKEN: Final = re.compile(rb"\S+")
"""A word of an ASCII raw file's values: a point's index, or a value."""


_SCALES: Final = frozenset({"time", "frequency"})
"""The scales of a raw file's vectors."""

_NOISES: Final = ("onoise", "inoise")
"""The prefixes of the vectors of noise densities, at the output and referred to the
input, in all and of each device's sources, a device within a subcircuit after a dot
and any other after an underscore."""


def _spice_name(name: str) -> str:
    """Normalizes a raw-file vector `name` into ``v(node)`` or ``i(device)`` form."""
    if name.endswith("#branch"):
        return f"i({name.removesuffix('#branch')})"
    if name.startswith(("v(", "i(", *_NOISES)) or name in _SCALES:
        return name
    return f"v({name})"
