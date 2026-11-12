# Copyright 2026 Maxence Faldor
"""Circuit and results files.

A circuit file holds a [`Circuit`][jolty.ir.Circuit] as JSON. A results file holds the
results of named analyses as a NumPy ``.npz`` archive: a JSON header that names the
circuit's unknowns, logic nets, device terminals, derivatives and variables, and says
what each analysis is, then the arrays of each analysis under keys such as
``step.times``, ``step.values``, ``step.currents`` and ``step.conductances``, or
``loop.frequencies`` and ``loop.gains``.

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


class _LoopGain(_Model):
    kind: Literal["stability"] = "stability"
    probe: str
    strategy: str


type _Entry = Annotated[
    _Point | _Sweep | _Response | _Waveforms | _Sensitivities | _LoopGain,
    Field(discriminator="kind"),
]


class _Names(_Model):
    """The names of a circuit's columns, which every analysis of a file shares."""

    unknowns: tuple[results.Unknown, ...]
    nets: tuple[str, ...] = ()
    terminals: tuple[str, ...] = ()
    composites: tuple[tuple[str, tuple[int, ...]], ...] = ()
    derivatives: tuple[tuple[str, str], ...] = ()
    variables: tuple[tuple[str, str], ...] = ()


class _Header(_Names):
    format: Literal["jolty-results"] = "jolty-results"
    version: Literal[1] = 1
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


def viewer_path(circuit: Path) -> Path:
    """Where the 3D viewer of a circuit goes beside it: ``<stem>.view.html``.

    Examples:
        >>> viewer_path(Path("rc.jolty.json")).name
        'rc.view.html'
    """
    stem = circuit.name.removesuffix(CIRCUIT_SUFFIX).removesuffix(".json")
    return circuit.with_name(f"{stem}.view.html")


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
        ValueError: When the analyses are of different circuits, or none is given.
    """
    circuits = {_names(result) for result in analyses.values()}
    if len(circuits) != 1:
        message = "a results file holds analyses of one circuit, one or more"
        raise ValueError(message)
    entries: dict[str, _Entry] = {}
    arrays: dict[str, np.ndarray] = {}
    for name, result in analyses.items():
        match result:
            case results.OperatingPoint():
                entries[name] = _Point(strategy=result.strategy)
                arrays.update(_point_arrays(name, result))
            case results.Sweep():
                entries[name] = _Sweep(source=result.source, quantity=result.quantity)
                arrays[f"{name}.levels"] = result.levels
                arrays[f"{name}.values"] = result.values
                arrays[f"{name}.logic"] = result.logic
                arrays.update(_observed_arrays(name, result))
            case results.Response():
                entries[name] = _Response(
                    source=result.source, strategy=result.operating_point.strategy
                )
                arrays[f"{name}.frequencies"] = result.frequencies
                arrays[f"{name}.values"] = result.values
                arrays[f"{name}.currents"] = result.currents
                arrays.update(
                    _point_arrays(f"{name}.operating_point", result.operating_point)
                )
            case results.Waveforms():
                entries[name] = _Waveforms(statistics=result.statistics)
                arrays[f"{name}.times"] = result.times
                arrays[f"{name}.values"] = result.values
                arrays.update(_observed_arrays(name, result))
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
                arrays[f"{name}.derivatives"] = result.derivatives
                arrays.update(
                    _point_arrays(f"{name}.operating_point", result.operating_point)
                )
            case results.LoopGain():
                entries[name] = _LoopGain(
                    probe=result.probe, strategy=result.operating_point.strategy
                )
                arrays[f"{name}.frequencies"] = result.frequencies
                arrays[f"{name}.gains"] = result.gains
                arrays.update(
                    _point_arrays(f"{name}.operating_point", result.operating_point)
                )
    names = circuits.pop()
    header = _Header(**names.model_dump(), analyses=entries)
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
        unknowns, nets = header.unknowns, header.nets
        loaded: dict[str, results.Result] = {}
        for name, entry in header.analyses.items():
            match entry:
                case _Point(strategy=strategy):
                    loaded[name] = _point(data, name, strategy, header)
                case _Sweep(source=source, quantity=quantity):
                    loaded[name] = results.Sweep(
                        unknowns,
                        source,
                        quantity,
                        data[f"{name}.levels"],
                        data[f"{name}.values"],
                        nets,
                        data[f"{name}.logic"],
                        **_observed(data, name, header),
                    )
                case _Response(source=source, strategy=strategy):
                    point = _point(data, f"{name}.operating_point", strategy, header)
                    loaded[name] = results.Response(
                        unknowns,
                        source,
                        data[f"{name}.frequencies"],
                        data[f"{name}.values"],
                        point,
                        terminals=header.terminals,
                        currents=data[f"{name}.currents"],
                        composites=header.composites,
                    )
                case _Waveforms(statistics=statistics):
                    loaded[name] = results.Waveforms(
                        unknowns,
                        data[f"{name}.times"],
                        data[f"{name}.values"],
                        statistics,
                        nets,
                        _traces(data, name),
                        **_observed(data, name, header),
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
                    point = _point(data, f"{name}.operating_point", strategy, header)
                    loaded[name] = results.Sensitivities(
                        unknowns,
                        outputs,
                        parameters,
                        data[f"{name}.derivatives"],
                        point,
                    )
                case _LoopGain(probe=probe, strategy=strategy):
                    point = _point(data, f"{name}.operating_point", strategy, header)
                    loaded[name] = results.LoopGain(
                        unknowns,
                        probe,
                        data[f"{name}.frequencies"],
                        data[f"{name}.gains"],
                        point,
                    )
    return loaded


def _names(result: results.Result) -> _Names:
    """The names of the columns of `result`'s circuit."""
    if isinstance(result, results.Response | results.Sensitivities | results.LoopGain):
        return _names(result.operating_point)
    return _Names(
        unknowns=result.unknowns,
        nets=result.nets,
        terminals=result.terminals,
        composites=result.composites,
        derivatives=result.derivatives,
        variables=result.variables,
    )


_OBSERVED: Final = ("currents", "charges", "conductances", "capacitances", "readings")
"""The arrays of what every device did, by the names of their fields."""


def _observed_arrays(
    name: str, result: results.OperatingPoint | results.Sweep | results.Waveforms
) -> dict[str, np.ndarray]:
    """The arrays of what every device did in the analysis `name`."""
    return {f"{name}.{field}": getattr(result, field) for field in _OBSERVED}


def _point_arrays(name: str, point: results.OperatingPoint) -> dict[str, np.ndarray]:
    """The arrays of the operating point stored under `name`."""
    return {
        f"{name}.values": point.values,
        f"{name}.logic": point.logic,
        **_observed_arrays(name, point),
    }


def _observed(
    data: Mapping[str, np.ndarray], name: str, header: _Header
) -> results.Observed:
    """What every device did in the analysis stored under `name`, as fields."""
    return {
        "terminals": header.terminals,
        "currents": data[f"{name}.currents"],
        "charges": data[f"{name}.charges"],
        "composites": header.composites,
        "derivatives": header.derivatives,
        "conductances": data[f"{name}.conductances"],
        "capacitances": data[f"{name}.capacitances"],
        "variables": header.variables,
        "readings": data[f"{name}.readings"],
    }


def _point(
    data: Mapping[str, np.ndarray], name: str, strategy: str, header: _Header
) -> results.OperatingPoint:
    """The operating point stored under `name` in `data`."""
    return results.OperatingPoint(
        header.unknowns,
        data[f"{name}.values"],
        strategy,
        header.nets,
        data[f"{name}.logic"],
        **_observed(data, name, header),
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
