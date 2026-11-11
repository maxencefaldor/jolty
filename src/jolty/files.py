# Copyright 2026 Maxence Faldor
"""Circuit and results files.

A circuit file holds a [`Circuit`][jolty.ir.Circuit] as JSON. A results file holds the
results of named analyses as a NumPy ``.npz`` archive: a JSON header that names the
circuit's unknowns and says what each analysis is, then the arrays of each analysis
under keys such as ``step.times`` and ``step.values``.

Examples:
    >>> import tempfile
    >>> from pathlib import Path
    >>> from jolty import results
    >>> point = results.OperatingPoint(
    ...     (results.Unknown("out", "voltage"),), np.array([2.5]), "direct"
    ... )
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     path = Path(directory) / "divider.results.npz"
    ...     save_results(path, {"op": point})
    ...     load_results(path)["op"].voltage("out")
    2.5
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jolty import ir, results
from jolty.errors import JoltyError

CIRCUIT_SUFFIX: Final = ".jolty.json"
"""The suffix of circuit files, such as ``rc_filter.jolty.json``."""

RESULTS_SUFFIX: Final = ".results.npz"
"""The suffix of results files, such as ``rc_filter.results.npz``."""

_HEADER: Final = "header"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _Point(_Model):
    kind: Literal["operating_point"] = "operating_point"
    strategy: str


class _Sweep(_Model):
    kind: Literal["dc_sweep"] = "dc_sweep"
    source: str
    quantity: Literal["voltage", "current"]


class _Response(_Model):
    kind: Literal["ac"] = "ac"
    source: str
    strategy: str


class _Waveforms(_Model):
    kind: Literal["transient"] = "transient"
    statistics: results.Statistics


type _Entry = Annotated[
    _Point | _Sweep | _Response | _Waveforms, Field(discriminator="kind")
]


class _Header(_Model):
    format: Literal["jolty-results"] = "jolty-results"
    version: Literal[1] = 1
    unknowns: tuple[results.Unknown, ...]
    analyses: dict[str, _Entry]


def circuit_path(script: Path) -> Path:
    """Where a script's circuit is saved by default: as ``<stem>.jolty.json``."""
    return script.with_name(script.stem + CIRCUIT_SUFFIX)


def results_path(script: Path) -> Path:
    """Where a script's results are saved by default: as ``<stem>.results.npz``."""
    return script.with_name(script.stem + RESULTS_SUFFIX)


def plot_path(results: Path, analysis: str) -> Path:
    """Where the plot of `analysis` goes beside its results, as ``<stem>.<name>.png``.

    Examples:
        >>> plot_path(Path("rc.results.npz"), "step").name
        'rc.step.png'
    """
    stem = results.name.removesuffix(RESULTS_SUFFIX).removesuffix(".npz")
    return results.with_name(f"{stem}.{analysis}.png")


def save_circuit(path: Path, circuit: ir.Circuit) -> None:
    """Writes `circuit` to `path` as JSON."""
    path.write_text(circuit.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_circuit(path: Path) -> ir.Circuit:
    """Reads the circuit file at `path`."""
    return ir.Circuit.model_validate_json(path.read_bytes())


def save_results(path: Path, analyses: Mapping[str, results.Result]) -> None:
    """Writes the results of `analyses`, by name, to `path`.

    Raises:
        ValueError: When the analyses solve different unknowns, or none is given.
    """
    unknowns = {result.unknowns for result in analyses.values()}
    if len(unknowns) != 1:
        message = "a results file holds analyses of one circuit, one or more"
        raise ValueError(message)
    entries: dict[str, _Entry] = {}
    arrays: dict[str, np.ndarray] = {}
    for name, result in analyses.items():
        match result:
            case results.OperatingPoint():
                entries[name] = _Point(strategy=result.strategy)
                arrays[f"{name}.values"] = result.values
            case results.Sweep():
                entries[name] = _Sweep(source=result.source, quantity=result.quantity)
                arrays[f"{name}.levels"] = result.levels
                arrays[f"{name}.values"] = result.values
            case results.Response():
                entries[name] = _Response(
                    source=result.source, strategy=result.operating_point.strategy
                )
                arrays[f"{name}.frequencies"] = result.frequencies
                arrays[f"{name}.values"] = result.values
                arrays[f"{name}.operating_point"] = result.operating_point.values
            case results.Waveforms():
                entries[name] = _Waveforms(statistics=result.statistics)
                arrays[f"{name}.times"] = result.times
                arrays[f"{name}.values"] = result.values
    header = _Header(unknowns=unknowns.pop(), analyses=entries)
    arrays[_HEADER] = np.array(header.model_dump_json())
    with path.open("wb") as file:
        np.savez_compressed(file, allow_pickle=False, **arrays)


def load_results(path: Path) -> dict[str, results.Result]:
    """Reads the results file at `path`: the results of each analysis, by name.

    Raises:
        JoltyError: When the file is not a jolty results file.
    """
    with np.load(path, allow_pickle=False) as data:
        try:
            header = _Header.model_validate_json(str(data[_HEADER]))
        except KeyError, ValidationError:
            message = f"{path} is not a jolty results file"
            raise JoltyError(message) from None
        unknowns = header.unknowns
        loaded: dict[str, results.Result] = {}
        for name, entry in header.analyses.items():
            values = data[f"{name}.values"]
            match entry:
                case _Point(strategy=strategy):
                    loaded[name] = results.OperatingPoint(unknowns, values, strategy)
                case _Sweep(source=source, quantity=quantity):
                    levels = data[f"{name}.levels"]
                    loaded[name] = results.Sweep(
                        unknowns, source, quantity, levels, values
                    )
                case _Response(source=source, strategy=strategy):
                    point = results.OperatingPoint(
                        unknowns, data[f"{name}.operating_point"], strategy
                    )
                    frequencies = data[f"{name}.frequencies"]
                    loaded[name] = results.Response(
                        unknowns, source, frequencies, values, point
                    )
                case _Waveforms(statistics=statistics):
                    times = data[f"{name}.times"]
                    loaded[name] = results.Waveforms(
                        unknowns, times, values, statistics
                    )
    return loaded
