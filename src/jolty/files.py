# Copyright 2026 Maxence Faldor
"""Circuit and results files.

A circuit file holds a [`Circuit`][jolty.ir.Circuit] as JSON. A results file holds the
results of named analyses as a NumPy ``.npz`` archive: a JSON header that names the
circuit's unknowns and logic nets, says what the analyses recorded of its devices, each
recording once however many analyses share it, and says what each analysis is, then
the arrays of each analysis under keys such as ``step.times``, ``step.values``,
``step.currents`` and ``step.conductances``, ``loop.frequencies`` and ``loop.gains``, or
``hiss.densities`` and ``hiss.contributions``. The header is in version 3 of the format;
a file of an earlier version, written before results recorded selectively, is not read.

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

_VERSION: Final = 3
"""The version of the results file's format that jolty writes and reads."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _Point(_Model):
    kind: Literal["operating_point"] = "operating_point"
    strategy: str
    recording: int


class _Points(_Model):
    kind: Literal["operating_points"] = "operating_points"
    strategies: tuple[str, ...]
    recording: int


class _Sweep(_Model):
    kind: Literal["dc_sweep"] = "dc_sweep"
    source: str
    quantity: results.Quantity
    recording: int


class _Response(_Model):
    kind: Literal["ac"] = "ac"
    source: str
    strategy: str
    recording: int


class _Waveforms(_Model):
    kind: Literal["transient"] = "transient"
    statistics: results.Statistics
    recording: int


class _Sensitivities(_Model):
    kind: Literal["sensitivity"] = "sensitivity"
    outputs: tuple[str, ...]
    parameters: tuple[tuple[str, str], ...]
    result: _Entry


class _LoopGain(_Model):
    kind: Literal["stability"] = "stability"
    probe: str
    strategy: str
    recording: int


class _NoiseInput(_Model):
    source: str
    quantity: results.Quantity


class _NoiseSpectrum(_Model):
    kind: Literal["noise"] = "noise"
    output: str
    reference: str | None
    quantity: results.Quantity
    sources: tuple[tuple[str, str], ...]
    input: _NoiseInput | None
    strategy: str
    recording: int


class _PolesZeros(_Model):
    kind: Literal["pole_zero"] = "pole_zero"
    source: str | None
    output: str | None
    reference: str | None
    strategy: str
    recording: int


type _Entry = Annotated[
    _Point
    | _Points
    | _Sweep
    | _Response
    | _Waveforms
    | _Sensitivities
    | _LoopGain
    | _NoiseSpectrum
    | _PolesZeros,
    Field(discriminator="kind"),
]


class _Names(_Model):
    """The names of a circuit's columns, which every analysis of a file shares."""

    unknowns: tuple[results.Unknown, ...]
    nets: tuple[str, ...] = ()


class _Device(_Model):
    """A device a result recorded, as a `RecordedDevice`."""

    path: str
    terminals: tuple[str, ...]
    internals: tuple[str, ...] = ()
    variables: tuple[str, ...] = ()


class _Recording(_Model):
    """What results recorded, as a [`Recording`][jolty.results.Recording]."""

    devices: tuple[_Device, ...] = ()
    composites: tuple[str, ...] = ()
    thermal: tuple[str, ...] = ()

    @classmethod
    def of(cls, recording: results.Recording) -> _Recording:
        """The header's entry of `recording`."""
        return cls(
            devices=tuple(
                _Device(
                    path=device.path,
                    terminals=device.terminals,
                    internals=device.internals,
                    variables=device.variables,
                )
                for device in recording.devices
            ),
            composites=recording.composites,
            thermal=tuple(sorted(recording.thermal)),
        )

    def recording(self) -> results.Recording:
        """The recording the entry describes."""
        return results.Recording(
            tuple(
                results.RecordedDevice(
                    device.path, device.terminals, device.internals, device.variables
                )
                for device in self.devices
            ),
            self.composites,
            frozenset(self.thermal),
        )


class _Header(_Names):
    format: Literal["jolty-results"] = "jolty-results"
    version: Literal[3] = 3
    recordings: tuple[_Recording, ...] = ()
    analyses: dict[str, _Entry]


class _Format(BaseModel):
    """A results file's format and version, whatever else its header holds."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    format: Literal["jolty-results"]
    version: int


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


def netlist_path(circuit: Path, instance: str = "") -> Path:
    """Where the KiCad netlist of the component at `instance` goes beside its circuit.

    The top component's is ``<stem>.net``; another's carries the ids of its instance's
    path, joined by dots, as ``<stem>.X1.X2.net``.

    Examples:
        >>> netlist_path(Path("amp.jolty.json")).name
        'amp.net'
        >>> netlist_path(Path("amp.jolty.json"), "X1/X2").name
        'amp.X1.X2.net'
    """
    stem = circuit.name.removesuffix(CIRCUIT_SUFFIX).removesuffix(".json")
    parts = [stem, *instance.split("/")] if instance else [stem]
    return circuit.with_name(".".join([*parts, "net"]))


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
    arrays: dict[str, np.ndarray] = {}
    recordings: dict[results.Recording, int] = {}
    entries = {
        name: _save(name, result, arrays, recordings)
        for name, result in analyses.items()
    }
    names = circuits.pop()
    header = _Header(
        **names.model_dump(),
        recordings=tuple(_Recording.of(recording) for recording in recordings),
        analyses=entries,
    )
    arrays[_HEADER] = np.array(header.model_dump_json())
    with path.open("wb") as file:
        np.savez_compressed(file, allow_pickle=False, **arrays)


def load_results(path: Path) -> dict[str, results.Result]:
    """Reads the results file at `path`: the results of each analysis, by name.

    Raises:
        JoltyError: When the file is not a jolty results file, or one in an earlier
            version of the format.
    """
    with np.load(path, allow_pickle=False) as data:
        try:
            header = _Header.model_validate_json(str(data[_HEADER]))
        except KeyError, ValidationError:
            raise _unreadable(path, data) from None
        recordings = tuple(recording.recording() for recording in header.recordings)
        return {
            name: _load(data, name, entry, header, recordings)
            for name, entry in header.analyses.items()
        }


def _unreadable(path: Path, data: Mapping[str, np.ndarray]) -> JoltyError:
    """Why the file at `path`, holding `data`, cannot be read as results."""
    try:
        version = _Format.model_validate_json(str(data[_HEADER])).version
    except KeyError, ValidationError:
        return JoltyError(f"{path} is not a jolty results file")
    if version == _VERSION:
        return JoltyError(f"{path} is not a jolty results file")
    return JoltyError(
        f"{path} holds results in version {version} of the format, which jolty reads "
        f"no more; simulate again to write them in version {_VERSION}"
    )


def _save(
    name: str,
    result: results.Result,
    arrays: dict[str, np.ndarray],
    recordings: dict[results.Recording, int],
) -> _Entry:
    """The header's entry of `result`, its arrays added to `arrays` under `name`.

    Its recording joins `recordings`, each recording by its index in the header, unless
    another result's is the same.
    """
    match result:
        case results.OperatingPoint():
            arrays.update(_point_arrays(name, result))
            recording = _indexed(recordings, result.recording)
            return _Point(strategy=result.strategy, recording=recording)
        case results.Sweep():
            arrays[f"{name}.levels"] = result.levels
            arrays[f"{name}.values"] = result.values
            arrays[f"{name}.logic"] = result.logic
            arrays.update(_observed_arrays(name, result))
            return _Sweep(
                source=result.source,
                quantity=result.quantity,
                recording=_indexed(recordings, result.recording),
            )
        case results.Response():
            arrays[f"{name}.frequencies"] = result.frequencies
            arrays[f"{name}.values"] = result.values
            arrays[f"{name}.currents"] = result.currents
            point = result.operating_point
            arrays.update(_point_arrays(f"{name}.operating_point", point))
            return _Response(
                source=result.source,
                strategy=point.strategy,
                recording=_indexed(recordings, result.recording),
            )
        case results.Waveforms():
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
            return _Waveforms(
                statistics=result.statistics,
                recording=_indexed(recordings, result.recording),
            )
        case results.OperatingPoints():
            for index, point in enumerate(result.points):
                arrays.update(_point_arrays(f"{name}.{index}", point))
            return _Points(
                strategies=tuple(point.strategy for point in result.points),
                recording=_indexed(recordings, result.points[0].recording),
            )
        case _:
            return _save_linear(name, result, arrays, recordings)


def _indexed(
    recordings: dict[results.Recording, int], recording: results.Recording
) -> int:
    """The index of `recording` among `recordings`, which it joins if it is new."""
    return recordings.setdefault(recording, len(recordings))


def _save_linear(
    name: str,
    result: results.Sensitivities
    | results.LoopGain
    | results.NoiseSpectrum
    | results.PolesZeros,
    arrays: dict[str, np.ndarray],
    recordings: dict[results.Recording, int],
) -> _Entry:
    """The header's entry of sensitivities, a loop gain, a noise spectrum or poles."""
    match result:
        case results.Sensitivities():
            arrays[f"{name}.parameters"] = np.array(
                [parameter.value for parameter in result.parameters], dtype=float
            )
            arrays[f"{name}.derivatives"] = result.derivatives
            return _Sensitivities(
                outputs=result.outputs,
                parameters=tuple(
                    (parameter.instance, parameter.name)
                    for parameter in result.parameters
                ),
                result=_save(f"{name}.result", result.result, arrays, recordings),
            )
        case results.LoopGain():
            arrays[f"{name}.frequencies"] = result.frequencies
            arrays[f"{name}.gains"] = result.gains
            point = result.operating_point
            arrays.update(_point_arrays(f"{name}.operating_point", point))
            return _LoopGain(
                probe=result.probe,
                strategy=point.strategy,
                recording=_indexed(recordings, point.recording),
            )
        case results.NoiseSpectrum():
            arrays[f"{name}.frequencies"] = result.frequencies
            arrays[f"{name}.densities"] = result.densities
            arrays[f"{name}.contributions"] = result.contributions
            point = result.operating_point
            arrays.update(_point_arrays(f"{name}.operating_point", point))
            found = None
            if result.input is not None:
                arrays[f"{name}.gains"] = result.input.gains
                found = _NoiseInput(
                    source=result.input.source, quantity=result.input.quantity
                )
            return _NoiseSpectrum(
                output=result.output,
                reference=result.reference,
                quantity=result.quantity,
                sources=tuple(
                    (source.instance, source.name) for source in result.sources
                ),
                input=found,
                strategy=point.strategy,
                recording=_indexed(recordings, point.recording),
            )
        case results.PolesZeros():
            arrays[f"{name}.poles"] = result.poles
            if result.zeros is not None:
                arrays[f"{name}.zeros"] = result.zeros
            point = result.operating_point
            arrays.update(_point_arrays(f"{name}.operating_point", point))
            return _PolesZeros(
                source=result.source,
                output=result.output,
                reference=result.reference,
                strategy=point.strategy,
                recording=_indexed(recordings, point.recording),
            )


def _load(
    data: Mapping[str, np.ndarray],
    name: str,
    entry: _Entry,
    header: _Header,
    recordings: tuple[results.Recording, ...],
) -> results.Result:
    """The result stored under `name` in `data`, as `entry` says it is.

    It recorded the one of `recordings` the entry names.
    """
    unknowns, nets = header.unknowns, header.nets
    match entry:
        case _Point(strategy=strategy, recording=recording):
            return _point(data, name, strategy, header, recordings[recording])
        case _Points(strategies=strategies, recording=recording):
            points = tuple(
                _point(data, f"{name}.{index}", strategy, header, recordings[recording])
                for index, strategy in enumerate(strategies)
            )
            return results.OperatingPoints(unknowns, points)
        case _Sweep(source=source, quantity=quantity, recording=recording):
            return results.Sweep(
                unknowns,
                source,
                quantity,
                data[f"{name}.levels"],
                data[f"{name}.values"],
                nets,
                data[f"{name}.logic"],
                **_observed(data, name, recordings[recording]),
            )
        case _Response(source=source, strategy=strategy, recording=recording):
            point = f"{name}.operating_point"
            return results.Response(
                unknowns,
                source,
                data[f"{name}.frequencies"],
                data[f"{name}.values"],
                _point(data, point, strategy, header, recordings[recording]),
                recording=recordings[recording],
                currents=data[f"{name}.currents"],
            )
        case _Waveforms(statistics=statistics, recording=recording):
            return results.Waveforms(
                unknowns,
                data[f"{name}.times"],
                data[f"{name}.values"],
                statistics,
                nets,
                _traces(data, name),
                **_observed(data, name, recordings[recording]),
            )
        case _:
            return _load_linear(data, name, entry, header, recordings)


def _load_linear(
    data: Mapping[str, np.ndarray],
    name: str,
    entry: _Sensitivities | _LoopGain | _NoiseSpectrum | _PolesZeros,
    header: _Header,
    recordings: tuple[results.Recording, ...],
) -> results.Result:
    """The sensitivities, loop gain, noise spectrum or poles stored under `name`."""
    unknowns = header.unknowns
    match entry:
        case _Sensitivities(outputs=outputs, parameters=named, result=differentiated):
            parameters = tuple(
                results.Parameter(instance, parameter, float(value))
                for (instance, parameter), value in zip(
                    named, data[f"{name}.parameters"], strict=True
                )
            )
            found = _load(data, f"{name}.result", differentiated, header, recordings)
            if isinstance(
                found,
                results.Sensitivities
                | results.LoopGain
                | results.NoiseSpectrum
                | results.PolesZeros
                | results.OperatingPoints,
            ):
                message = (
                    f"the sensitivities {name} differentiate neither an operating "
                    "point, a sweep, a response nor a transient"
                )
                raise JoltyError(message)
            return results.Sensitivities(
                unknowns, outputs, parameters, data[f"{name}.derivatives"], found
            )
        case _LoopGain(probe=probe, strategy=strategy, recording=recording):
            point = f"{name}.operating_point"
            return results.LoopGain(
                unknowns,
                probe,
                data[f"{name}.frequencies"],
                data[f"{name}.gains"],
                _point(data, point, strategy, header, recordings[recording]),
            )
        case _NoiseSpectrum():
            found_input = None
            if entry.input is not None:
                found_input = results.NoiseInput(
                    entry.input.source, entry.input.quantity, data[f"{name}.gains"]
                )
            return results.NoiseSpectrum(
                unknowns,
                entry.output,
                entry.reference,
                entry.quantity,
                data[f"{name}.frequencies"],
                data[f"{name}.densities"],
                tuple(
                    results.NoiseSource(instance, source)
                    for instance, source in entry.sources
                ),
                data[f"{name}.contributions"],
                found_input,
                _point(
                    data,
                    f"{name}.operating_point",
                    entry.strategy,
                    header,
                    recordings[entry.recording],
                ),
            )
        case _PolesZeros(
            source=source,
            output=output,
            reference=reference,
            strategy=strategy,
            recording=recording,
        ):
            point = f"{name}.operating_point"
            return results.PolesZeros(
                unknowns,
                data[f"{name}.poles"],
                data.get(f"{name}.zeros"),
                source,
                output,
                reference,
                _point(data, point, strategy, header, recordings[recording]),
            )


def _names(result: results.Result) -> _Names:
    """The names of the columns of `result`'s circuit."""
    if isinstance(result, results.Sensitivities):
        return _names(result.result)
    if isinstance(result, results.OperatingPoints):
        return _names(result.points[0])
    if isinstance(
        result,
        results.Response
        | results.LoopGain
        | results.NoiseSpectrum
        | results.PolesZeros,
    ):
        return _names(result.operating_point)
    return _Names(unknowns=result.unknowns, nets=result.nets)


_OBSERVED: Final = ("currents", "charges", "conductances", "capacitances", "readings")
"""The arrays of what the recorded devices did, by the names of their fields."""


def _observed_arrays(
    name: str, result: results.OperatingPoint | results.Sweep | results.Waveforms
) -> dict[str, np.ndarray]:
    """The arrays of what the recorded devices did in the analysis `name`."""
    return {f"{name}.{field}": getattr(result, field) for field in _OBSERVED}


def _point_arrays(name: str, point: results.OperatingPoint) -> dict[str, np.ndarray]:
    """The arrays of the operating point stored under `name`."""
    return {
        f"{name}.values": point.values,
        f"{name}.logic": point.logic,
        **_observed_arrays(name, point),
    }


def _observed(
    data: Mapping[str, np.ndarray], name: str, recording: results.Recording
) -> results.Observed:
    """What the devices `recording` names did in the analysis stored under `name`."""
    return {
        "recording": recording,
        "currents": data[f"{name}.currents"],
        "charges": data[f"{name}.charges"],
        "conductances": data[f"{name}.conductances"],
        "capacitances": data[f"{name}.capacitances"],
        "readings": data[f"{name}.readings"],
    }


def _point(
    data: Mapping[str, np.ndarray],
    name: str,
    strategy: str,
    header: _Header,
    recording: results.Recording,
) -> results.OperatingPoint:
    """The operating point stored under `name` in `data`, which recorded `recording`."""
    return results.OperatingPoint(
        header.unknowns,
        data[f"{name}.values"],
        strategy,
        header.nets,
        data[f"{name}.logic"],
        **_observed(data, name, recording),
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
