# Copyright 2026 Maxence Faldor
"""Circuit and results files.

A circuit file holds a [`Circuit`][jolty.ir.Circuit] as JSON. A results file holds the
results of named analyses as a NumPy ``.npz`` archive: a JSON header that names the
circuit's unknowns, logic nets and device terminals and says what each analysis is, then
the arrays of each analysis under keys such as ``step.times``, ``step.values`` and
``step.currents``.

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


class _Sensitivities(_Model):
    kind: Literal["sensitivity"] = "sensitivity"
    outputs: tuple[str, ...]
    parameters: tuple[tuple[str, str], ...]
    strategy: str


type _Entry = Annotated[
    _Point | _Sweep | _Response | _Waveforms | _Sensitivities,
    Field(discriminator="kind"),
]


class _Header(_Model):
    format: Literal["jolty-results"] = "jolty-results"
    version: Literal[1] = 1
    unknowns: tuple[results.Unknown, ...]
    nets: tuple[str, ...] = ()
    terminals: tuple[str, ...] = ()
    analyses: dict[str, _Entry]


def circuit_path(script: Path) -> Path:
    """Where a script's circuit is saved by default: as ``<stem>.jolty.json``."""
    return script.with_name(script.stem + CIRCUIT_SUFFIX)


def results_path(script: Path) -> Path:
    """Where a script's results are saved by default: as ``<stem>.results.npz``."""
    return script.with_name(script.stem + RESULTS_SUFFIX)


def plot_path(results: Path, analysis: str) -> Path:
    """Where the plot of `analysis` goes beside its results: ``<stem>.<name>.plot.png``.

    Examples:
        >>> plot_path(Path("rc.results.npz"), "step").name
        'rc.step.plot.png'
    """
    stem = results.name.removesuffix(RESULTS_SUFFIX).removesuffix(".npz")
    return results.with_name(f"{stem}.{analysis}.plot.png")


def schematic_path(circuit: Path, instance: str = "") -> Path:
    """Where the schematic of the component at `instance` goes beside its circuit.

    The top component's is ``<stem>.schematic.png``; another's carries the ids of its
    instance's path, joined by dots, as ``<stem>.X1.X2.schematic.png``. The suffixes
    keep schematics and plots apart, whatever their names.

    Examples:
        >>> schematic_path(Path("rc.jolty.json")).name
        'rc.schematic.png'
        >>> schematic_path(Path("rc.jolty.json"), "X1/X2").name
        'rc.X1.X2.schematic.png'
    """
    stem = circuit.name.removesuffix(CIRCUIT_SUFFIX).removesuffix(".json")
    parts = [stem, *instance.split("/")] if instance else [stem]
    return circuit.with_name(".".join([*parts, "schematic.png"]))


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
    circuits = {
        (result.unknowns, _nets(result), _terminals(result))
        for result in analyses.values()
    }
    if len(circuits) != 1:
        message = "a results file holds analyses of one circuit, one or more"
        raise ValueError(message)
    entries: dict[str, _Entry] = {}
    arrays: dict[str, np.ndarray] = {}
    for name, result in analyses.items():
        match result:
            case results.OperatingPoint():
                entries[name] = _Point(strategy=result.strategy)
                arrays[f"{name}.values"] = result.values
                arrays[f"{name}.logic"] = result.logic
                arrays[f"{name}.currents"] = result.currents
                arrays[f"{name}.charges"] = result.charges
            case results.Sweep():
                entries[name] = _Sweep(source=result.source, quantity=result.quantity)
                arrays[f"{name}.levels"] = result.levels
                arrays[f"{name}.values"] = result.values
                arrays[f"{name}.logic"] = result.logic
                arrays[f"{name}.currents"] = result.currents
                arrays[f"{name}.charges"] = result.charges
            case results.Response():
                entries[name] = _Response(
                    source=result.source, strategy=result.operating_point.strategy
                )
                arrays[f"{name}.frequencies"] = result.frequencies
                arrays[f"{name}.values"] = result.values
                arrays[f"{name}.currents"] = result.currents
                arrays.update(_point_arrays(name, result.operating_point))
            case results.Waveforms():
                entries[name] = _Waveforms(statistics=result.statistics)
                arrays[f"{name}.times"] = result.times
                arrays[f"{name}.values"] = result.values
                arrays[f"{name}.currents"] = result.currents
                arrays[f"{name}.charges"] = result.charges
                traces = result.logic
                arrays[f"{name}.logic.counts"] = np.array(
                    [len(trace.times) for trace in traces], dtype=np.int64
                )
                arrays[f"{name}.logic.times"] = np.concatenate(
                    [np.empty(0), *(trace.times for trace in traces)]
                )
                arrays[f"{name}.logic.levels"] = np.concatenate(
                    [np.empty(0), *(trace.levels for trace in traces)]
                )
            case results.Sensitivities():
                entries[name] = _Sensitivities(
                    outputs=result.outputs,
                    parameters=tuple(
                        (parameter.instance, parameter.name)
                        for parameter in result.parameters
                    ),
                    strategy=result.operating_point.strategy,
                )
                arrays[f"{name}.parameters"] = np.array(
                    [parameter.value for parameter in result.parameters], dtype=float
                )
                arrays[f"{name}.values"] = result.derivatives
                arrays.update(_point_arrays(name, result.operating_point))
    unknowns, nets, terminals = circuits.pop()
    header = _Header(
        unknowns=unknowns, nets=nets, terminals=terminals, analyses=entries
    )
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
        unknowns, nets, terminals = header.unknowns, header.nets, header.terminals
        loaded: dict[str, results.Result] = {}
        for name, entry in header.analyses.items():
            values = data[f"{name}.values"]
            match entry:
                case _Point(strategy=strategy):
                    loaded[name] = results.OperatingPoint(
                        unknowns,
                        values,
                        strategy,
                        nets,
                        data[f"{name}.logic"],
                        terminals,
                        data[f"{name}.currents"],
                        data[f"{name}.charges"],
                    )
                case _Sweep(source=source, quantity=quantity):
                    levels = data[f"{name}.levels"]
                    loaded[name] = results.Sweep(
                        unknowns,
                        source,
                        quantity,
                        levels,
                        values,
                        nets,
                        data[f"{name}.logic"],
                        terminals,
                        data[f"{name}.currents"],
                        data[f"{name}.charges"],
                    )
                case _Response(source=source, strategy=strategy):
                    frequencies = data[f"{name}.frequencies"]
                    loaded[name] = results.Response(
                        unknowns,
                        source,
                        frequencies,
                        values,
                        _point(data, name, strategy, header),
                        terminals,
                        data[f"{name}.currents"],
                    )
                case _Waveforms(statistics=statistics):
                    times = data[f"{name}.times"]
                    loaded[name] = results.Waveforms(
                        unknowns,
                        times,
                        values,
                        statistics,
                        nets,
                        _traces(data, name),
                        terminals,
                        data[f"{name}.currents"],
                        data[f"{name}.charges"],
                    )
                case _Sensitivities(
                    outputs=outputs, parameters=named, strategy=strategy
                ):
                    parameters = tuple(
                        results.Parameter(instance, parameter, float(value))
                        for (instance, parameter), value in zip(
                            named, data[f"{name}.parameters"], strict=True
                        )
                    )
                    point = _point(data, name, strategy, header)
                    loaded[name] = results.Sensitivities(
                        unknowns, outputs, parameters, values, point
                    )
    return loaded


def _nets(result: results.Result) -> tuple[str, ...]:
    """The logic nets of `result`."""
    if isinstance(result, results.Response | results.Sensitivities):
        return result.operating_point.nets
    return result.nets


def _terminals(result: results.Result) -> tuple[str, ...]:
    """The device terminals of `result`."""
    if isinstance(result, results.Sensitivities):
        return result.operating_point.terminals
    return result.terminals


def _point_arrays(name: str, point: results.OperatingPoint) -> dict[str, np.ndarray]:
    """The arrays of the operating point that the analysis `name` holds."""
    return {
        f"{name}.operating_point": point.values,
        f"{name}.operating_point.logic": point.logic,
        f"{name}.operating_point.currents": point.currents,
        f"{name}.operating_point.charges": point.charges,
    }


def _point(
    data: Mapping[str, np.ndarray], name: str, strategy: str, header: _Header
) -> results.OperatingPoint:
    """The operating point that the analysis `name` holds in `data`."""
    return results.OperatingPoint(
        header.unknowns,
        data[f"{name}.operating_point"],
        strategy,
        header.nets,
        data[f"{name}.operating_point.logic"],
        header.terminals,
        data[f"{name}.operating_point.currents"],
        data[f"{name}.operating_point.charges"],
    )


def _traces(data: Mapping[str, np.ndarray], name: str) -> tuple[results.Trace, ...]:
    """The logic traces of the transient `name`, stored end to end in `data`."""
    counts = data[f"{name}.logic.counts"]
    ends = np.cumsum(counts)
    times = np.split(data[f"{name}.logic.times"], ends[:-1])
    levels = np.split(data[f"{name}.logic.levels"], ends[:-1])
    return tuple(
        results.Trace(times[index], levels[index]) for index in range(len(counts))
    )
