# Copyright 2026 Maxence Faldor
"""SPICE `.raw` file reading and writing (binary and ASCII)."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from jolty import results, unknowns
from jolty.errors import JoltyError
from jolty.spice import ngspice

type RawAnalysisResult = (
    results.OperatingPointResult
    | results.DcSweepResult
    | results.AcResult
    | results.TransientResult
)
"""An analysis result that SPICE's raw format represents."""

_SWEEP_SCALES: Final[Mapping[results.Nature, tuple[str, str]]] = {
    "voltage": ("v-sweep", "voltage"),
    "current": ("i-sweep", "current"),
    "temperature": ("temp-sweep", "temperature"),
}
"""Scale vector name and SPICE type for each DC sweep nature."""

_WRAPPED_MIN_LENGTH: Final = 3
"""Minimum length of a SPICE ``v(x)`` or ``i(x)`` wrapper with a non-empty name."""


@dataclass(frozen=True, slots=True)
class _Plot:
    """Data prepared for writing one plot of a SPICE raw file."""

    title: str
    plot: str
    unknowns: tuple[results.Unknown, ...]
    scale: tuple[str, str, np.ndarray] | None
    table: np.ndarray


def save_raw(
    path: Path,
    analyses: Mapping[str, results.AnalysisResult],
    *,
    binary: bool = True,
) -> None:
    """Writes the results of `analyses` to `path` in SPICE raw format, one plot each.

    Each plot follows the ngspice 47 raw format with a header (analysis title, scale,
    and vector names and types) followed by data points. Points are encoded as
    little-endian IEEE-754 doubles when `binary` is true, or as ASCII otherwise.

    An operating point writes a single point with no scale vector. A DC sweep uses
    ``v-sweep``, ``i-sweep``, or ``temp-sweep`` as its scale, falling back to type
    ``notype`` for parameter sweeps without a physical nature. An AC analysis uses a
    ``frequency`` scale with complex values, and a transient analysis uses a ``time``
    scale.

    Net vectors use the net path, and branch currents such as ``V1:current`` use
    ngspice ``V1#branch`` notation; all other unknowns keep their names. Thermal
    unknowns use type ``temperature`` and state unknowns use ``notype``. Output headers
    omit timestamps so identical results produce identical bytes, and
    [`load_raw`][jolty.files.load_raw] reads the file back.

    Raises:
        ValueError: When `analyses` is empty, an analysis name contains a line break,
            or an analysis type is unsupported in SPICE raw format.
    """
    if not analyses:
        message = "a raw file holds one or more analyses"
        raise ValueError(message)
    written = bytearray()
    for name, result in analyses.items():
        if "\n" in name or "\r" in name:
            message = f"a raw file's analysis name holds no line break: {name!r}"
            raise ValueError(message)
        written += _raw_plot(_prepare_plot(name, result), binary=binary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(written))


def load_raw(path: Path) -> dict[str, RawAnalysisResult]:
    """Reads the SPICE raw file at `path` into jolty results keyed by plot title.

    Supports both binary (little-endian IEEE-754 doubles) and ASCII raw files written by
    [`save_raw`][jolty.files.save_raw] or ngspice, including multi-plot files. Each
    plot's ``Title:`` becomes its mapping key; an empty title falls back to the analysis
    kind (``operating_point``, ``dc_sweep``, ``ac``, or ``transient``). Duplicate titles
    receive ``#2``, ``#3``, and subsequent suffixes so no plot is dropped.

    SPICE raw files store only headers and vector arrays, so jolty fills missing
    metadata with defaults. Loaded operating points carry ``strategy="raw"`` and
    ``iterations=None``, while AC results carry ``source=""`` and an all-NaN
    ``operating_point`` placeholder. Loaded DC sweeps use the scale vector name as
    ``source`` with ``parameter=None``, and transients set step count to
    ``max(len(times) - 1, 0)`` with zero rejected steps. Binary raw files preserve exact
    IEEE-754 bits, including ``-0.0``.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> import numpy as np
        >>> from jolty import results
        >>> point = results.OperatingPointResult(
        ...     (results.Unknown("out", "voltage"),), np.array([2.5]), "direct"
        ... )
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     path = Path(directory) / "divider.raw"
        ...     save_raw(path, {"op": point})
        ...     load_raw(path)["op"].voltage("out")
        2.5

    Raises:
        JoltyError: When the file contains no plots, malformed headers, truncated
            values, invalid numbers, an unsupported ``Plotname:``, or mismatched
            real/complex flags.
    """
    data = path.read_bytes()
    try:
        plots = ngspice.parse_raw(data)
    except JoltyError as error:
        message = f"{path}: {error}"
        raise JoltyError(message) from None
    if not plots:
        message = f"{path} holds no SPICE raw plots"
        raise JoltyError(message)
    loaded: dict[str, RawAnalysisResult] = {}
    for plot in plots:
        default_name, result = _result_from_raw(path, plot)
        base = plot.title or default_name
        key = base
        suffix = 2
        while key in loaded:
            key = f"{base}#{suffix}"
            suffix += 1
        loaded[key] = result
    return loaded


def _prepare_plot(name: str, result: results.AnalysisResult) -> _Plot:
    """Packs `result` into a `_Plot` ready for SPICE raw serialization."""
    match result:
        case results.TransientResult(times=times, values=values):
            return _Plot(
                name,
                "Transient Analysis",
                result.unknowns,
                ("time", "time", times),
                values,
            )
        case results.AcResult(frequencies=frequencies, values=values):
            return _Plot(
                name,
                "AC Analysis",
                result.unknowns,
                ("frequency", "frequency", frequencies),
                np.asarray(values, dtype=np.complex128),
            )
        case results.DcSweepResult(swept=swept, values=values, nature=nature):
            scale = (*_raw_sweep_scale(nature), swept)
            return _Plot(
                name, "DC transfer characteristic", result.unknowns, scale, values
            )
        case results.OperatingPointResult(values=values):
            table = np.asarray(values, dtype=np.float64).reshape(
                1, len(result.unknowns)
            )
            return _Plot(name, "Operating Point", result.unknowns, None, table)
        case _:
            message = (
                "a raw file holds operating points, DC sweeps, small-signal "
                f"responses and transients, not {name}'s {type(result).__name__}"
            )
            raise ValueError(message)


def _raw_sweep_scale(nature: results.Nature | None) -> tuple[str, str]:
    """Returns the scale vector name and SPICE type for a DC sweep of `nature`."""
    if nature is None:
        return ("v-sweep", "notype")
    return _SWEEP_SCALES.get(nature, ("v-sweep", "notype"))


def _raw_sweep_nature(scale: ngspice.RawVariable) -> results.Nature | None:
    """Infers the DC sweep physical nature from its raw scale vector."""
    if scale.kind == "current" or scale.name.casefold() == "i-sweep":
        return "current"
    found = unknowns.parse_nature(scale.kind)
    return None if found == "state" else found


def _result_from_raw(
    path: Path, plot: ngspice.RawPlot
) -> tuple[str, RawAnalysisResult]:
    """Converts one parsed `RawPlot` into `(default_name, result)`."""
    match plot.plot.casefold():
        case "operating point":
            return "operating_point", _raw_operating_point(path, plot)
        case "dc transfer characteristic":
            return "dc_sweep", _raw_dc_sweep(path, plot)
        case "ac analysis":
            return "ac", _raw_ac(path, plot)
        case "transient analysis":
            return "transient", _raw_transient(path, plot)
        case _:
            message = f"{path}: unsupported SPICE raw plot `{plot.plot}`"
            raise JoltyError(message)


def _raw_scaled(
    path: Path,
    plot: ngspice.RawPlot,
    kind: str,
    scale_kind: str,
    *,
    expect_complex: bool = False,
) -> tuple[ngspice.RawVariable, tuple[results.Unknown, ...], np.ndarray, np.ndarray]:
    """Validates scale and real/complex flags of a scaled raw plot and extracts data."""
    label = plot.title or plot.plot
    if plot.complex_values != expect_complex:
        found = "real values, not complex" if expect_complex else "complex values"
        message = f"{path}: {kind} plot `{label}` holds {found}"
        raise JoltyError(message)
    if not plot.variables:
        message = f"{path}: {kind} plot `{label}` has no {scale_kind} variable"
        raise JoltyError(message)
    scale = plot.variables[0]
    if scale_kind not in {"sweep", scale.kind} and scale.name.casefold() != scale_kind:
        message = (
            f"{path}: {kind} plot `{label}` begins with `{scale.name}` ({scale.kind}), "
            f"not {scale_kind}"
        )
        raise JoltyError(message)
    parsed = tuple(_unknown_from_raw(variable) for variable in plot.variables[1:])
    axis = np.array(plot.values[:, 0].real, dtype=np.float64, copy=True)
    values = (
        np.array(plot.values[:, 1:], dtype=np.complex128, copy=True)
        if expect_complex
        else np.array(plot.values[:, 1:].real, dtype=np.float64, copy=True)
    )
    return scale, parsed, axis, values


def _raw_operating_point(
    path: Path, plot: ngspice.RawPlot
) -> results.OperatingPointResult:
    """Converts an ``Operating Point`` raw plot into `OperatingPointResult`."""
    label = plot.title or plot.plot
    if plot.complex_values:
        message = f"{path}: operating point plot `{label}` holds complex values"
        raise JoltyError(message)
    if plot.values.shape[0] != 1:
        message = (
            f"{path}: operating point plot `{label}` holds "
            f"{plot.values.shape[0]} points, not 1"
        )
        raise JoltyError(message)
    parsed = tuple(_unknown_from_raw(variable) for variable in plot.variables)
    values = np.array(plot.values[0].real, dtype=np.float64, copy=True)
    return results.OperatingPointResult(parsed, values, strategy="raw")


def _raw_dc_sweep(path: Path, plot: ngspice.RawPlot) -> results.DcSweepResult:
    """Converts a ``DC transfer characteristic`` raw plot into `DcSweepResult`."""
    scale, parsed, swept, values = _raw_scaled(path, plot, "DC sweep", "sweep")
    return results.DcSweepResult(
        parsed,
        source=scale.name,
        nature=_raw_sweep_nature(scale),
        swept=swept,
        values=values,
    )


def _raw_ac(path: Path, plot: ngspice.RawPlot) -> results.AcResult:
    """Converts an ``AC Analysis`` raw plot into `AcResult`."""
    _, parsed, frequencies, values = _raw_scaled(
        path, plot, "AC", "frequency", expect_complex=True
    )
    operating_point = results.OperatingPointResult(
        parsed, np.full(len(parsed), np.nan, dtype=np.float64), strategy="raw"
    )
    return results.AcResult(
        parsed,
        "",
        frequencies=frequencies,
        values=values,
        operating_point=operating_point,
    )


def _raw_transient(path: Path, plot: ngspice.RawPlot) -> results.TransientResult:
    """Converts a ``Transient Analysis`` raw plot into `TransientResult`."""
    _, parsed, times, values = _raw_scaled(path, plot, "transient", "time")
    statistics = results.Statistics(None, max(len(times) - 1, 0), 0, 0, 0)
    return results.TransientResult(
        parsed, times=times, values=values, statistics=statistics
    )


def _unknown_from_raw(variable: ngspice.RawVariable) -> results.Unknown:
    """Converts a raw file vector into a jolty `Unknown`, reversing `_raw_name`."""
    name = variable.name
    nature = unknowns.parse_nature(variable.kind)
    if name.endswith("#branch") and len(name) > len("#branch"):
        return results.Unknown(f"{name.removesuffix('#branch')}:current", "current")
    if name.endswith(")") and len(name) > _WRAPPED_MIN_LENGTH:
        lowered = name.casefold()
        inner = name[2:-1]
        if nature in {"voltage", "state"} and lowered.startswith("v("):
            return results.Unknown(inner, "voltage")
        if nature in {"current", "state"} and lowered.startswith("i("):
            return results.Unknown(f"{inner}:current", "current")
    return results.Unknown(name, nature)


def _raw_header(
    spec: _Plot,
    named: list[tuple[str, str]],
    points: int,
    *,
    complex_values: bool,
    binary: bool,
) -> bytes:
    """Formats the header bytes for one plot of a SPICE raw file."""
    lines = [
        f"Title: {spec.title}",
        f"Plotname: {spec.plot}",
        f"Flags: {'complex' if complex_values else 'real'}",
        f"No. Variables: {len(named)}",
        f"No. Points: {points}",
        "Variables:",
        *(f"\t{index}\t{name}\t{kind}" for index, (name, kind) in enumerate(named)),
        "Binary:" if binary else "Values:",
    ]
    return ("\n".join(lines) + "\n").encode()


def _raw_values(matrix: np.ndarray, *, complex_values: bool, binary: bool) -> bytes:
    """Encodes the point matrix bytes for one plot of a SPICE raw file."""
    if complex_values:
        typed = matrix.astype(np.complex128)
        if binary:
            points, width = typed.shape
            parts = np.stack([typed.real, typed.imag], axis=-1).reshape(
                points, 2 * width
            )
            return parts.astype("<f8").tobytes()
        formatted = (
            "\n\t".join(f"{value.real:.17e},{value.imag:.17e}" for value in row)
            for row in typed
        )
    else:
        typed = matrix.astype(np.float64)
        if binary:
            return typed.astype("<f8").tobytes()
        formatted = ("\n\t".join(f"{value:.17e}" for value in row) for row in typed)
    return "".join(f" {index}\t{row}\n" for index, row in enumerate(formatted)).encode()


def _raw_plot(spec: _Plot, *, binary: bool) -> bytes:
    """Serializes one SPICE raw plot as header bytes plus ASCII or `binary` values."""
    named = [
        (_raw_name(unknown), "notype" if unknown.nature == "state" else unknown.nature)
        for unknown in spec.unknowns
    ]
    columns = [np.asarray(spec.table[:, index]) for index in range(len(spec.unknowns))]
    if spec.scale is not None:
        named.insert(0, spec.scale[:2])
        columns.insert(0, np.asarray(spec.scale[2]))
    complex_values = np.iscomplexobj(spec.table)
    points = len(columns[0]) if columns else spec.table.shape[0]
    header = _raw_header(
        spec, named, points, complex_values=complex_values, binary=binary
    )
    matrix = np.column_stack(columns) if columns else np.zeros((points, 0))
    return header + _raw_values(matrix, complex_values=complex_values, binary=binary)


def _raw_name(unknown: results.Unknown) -> str:
    """Formats the SPICE raw vector name for `unknown` using ngspice conventions."""
    path, separator, label = unknown.name.rpartition(":")
    if separator and label == "current" and unknown.nature == "current":
        return f"{path}#branch"
    return unknown.name
