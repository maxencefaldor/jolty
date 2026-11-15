# Copyright 2026 Maxence Faldor
"""Circuit, results, batch and shard files.

A circuit file holds a [`Circuit`][jolty.ir.Circuit] as JSON. A batch file holds a
plan's batches of runs, each a [`Batch`][jolty.files.Batch]: the arrays of its
[`BatchResult`][jolty.results.BatchResult] under keys such as ``monte-carlo.measured``,
and a JSON header of the rest. A shard file holds a [`Shard`][jolty.files.Shard], a
range of a plan's Monte Carlo samples: the arrays of its batch at each of the plan's
conditions under keys such as ``1.measured``, for the plan's first corner, and a JSON
header of the rest, the requirements it measured among it. Every file but a circuit's
is a NumPy ``.npz`` archive whose bytes are a function of what it holds alone, so a
shard run again writes the same file. A results file holds the
results of named analyses as a NumPy ``.npz`` archive: a JSON header that names the
circuit's unknowns and logic nets, says what the analyses recorded of its devices, each
recording once however many analyses share it, and says what each analysis is, then
the arrays of each analysis under keys such as ``step.times``, ``step.values``,
``step.currents`` and ``step.conductances``, ``loop.frequencies`` and ``loop.gains``, or
``hiss.densities`` and ``hiss.contributions``, and where each analysis used logic
devices outside their domains. Its header also states the results'
[`Provenance`][jolty.files.Provenance]: the jolty, the build and the platform that
simulated them, and the hashes of the circuit and the plan they are of. The header is
in version 10 of the format, which holds that provenance and the regions of operation
each recorded transistor's model distinguishes; a file of another version is not
read.

Examples:
    >>> import tempfile
    >>> from pathlib import Path
    >>> from jolty import results
    >>> point = results.OperatingPointResult(
    ...     (results.Unknown("out", "voltage"),), np.array([2.5]), "direct"
    ... )
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     path = Path(directory) / "divider.results.npz"
    ...     save_results(path, {"op": point})
    ...     load_results(path)["op"].voltage("out")
    2.5
"""

import functools
import hashlib
import platform as platforms
import sys
import sysconfig
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Final, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jolty import _core, ir, results
from jolty.errors import JoltyError
from jolty.plan import Plan
from jolty.specs import Measurement, MeasurementError, Reading, Requirement, Specs
from jolty.units import Dimension, Quantity

CIRCUIT_SUFFIX: Final = ".jolty.json"
"""The suffix of circuit files, such as ``rc_filter.jolty.json``."""

RESULTS_SUFFIX: Final = ".results.npz"
"""The suffix of results files, such as ``rc_filter.results.npz``."""

BATCHES_SUFFIX: Final = ".batches.npz"
"""The suffix of batch files, such as ``rc_filter.batches.npz``."""

SHARDS_SUFFIX: Final = ".shards"
"""The suffix of the directories that hold a script's shards, such as ``amp.shards``."""

SHARD_PATTERN: Final = "shard-*-*.npz"
"""The names of shard files, which a merge reads in their directory."""

_EPOCH: Final = (1980, 1, 1, 0, 0, 0)
"""The date a ZIP archive's entries carry: the earliest it can, the same every time."""

_HEADER: Final = "header"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _OperatingPointResult(_Model):
    kind: Literal["operating_point"] = "operating_point"
    strategy: str
    recording: int


class _OperatingPointsResult(_Model):
    kind: Literal["operating_points"] = "operating_points"
    strategies: tuple[str, ...]
    recording: int


class _DcSweepResult(_Model):
    kind: Literal["dc_sweep"] = "dc_sweep"
    source: str
    quantity: results.Quantity | None
    parameter: str | None = None
    recording: int


class _AcResult(_Model):
    kind: Literal["ac"] = "ac"
    source: str
    strategy: str
    recording: int


class _TransientResult(_Model):
    kind: Literal["transient"] = "transient"
    statistics: results.Statistics
    recording: int


class _SensitivityResult(_Model):
    kind: Literal["sensitivity"] = "sensitivity"
    outputs: tuple[str, ...]
    parameters: tuple[tuple[str, str], ...]
    result: _Entry


class _StabilityResult(_Model):
    kind: Literal["stability"] = "stability"
    probe: str
    strategy: str
    recording: int


class _NoiseInput(_Model):
    source: str
    quantity: results.Quantity


class _NoiseResult(_Model):
    kind: Literal["noise"] = "noise"
    output: str
    reference: str | None
    quantity: results.Quantity
    sources: tuple[tuple[str, str], ...]
    input: _NoiseInput | None
    strategy: str
    recording: int


class _PoleZeroResult(_Model):
    kind: Literal["pole_zero"] = "pole_zero"
    source: str | None
    output: str | None
    reference: str | None
    strategy: str
    recording: int


class _TransferFunctionResult(_Model):
    kind: Literal["transfer_function"] = "transfer_function"
    source: str
    input: results.Quantity
    output: str
    reference: str | None
    strategy: str
    recording: int


type _Entry = Annotated[
    _OperatingPointResult
    | _OperatingPointsResult
    | _DcSweepResult
    | _AcResult
    | _TransientResult
    | _SensitivityResult
    | _StabilityResult
    | _NoiseResult
    | _PoleZeroResult
    | _TransferFunctionResult,
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
    regions: tuple[str, ...] = ()
    margins: tuple[str, ...] = ()


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
                    regions=device.regions,
                    margins=device.margins,
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
                    device.path,
                    device.terminals,
                    device.internals,
                    device.variables,
                    device.regions,
                    device.margins,
                )
                for device in self.devices
            ),
            self.composites,
            frozenset(self.thermal),
        )


_PACKAGE: Final = Path(__file__).resolve().parent
"""jolty's own directory, whose modules and engine a build is."""


def circuit_hash(circuit: ir.Circuit) -> str:
    """A SHA-256 hash, in hexadecimal, of what a simulation reads of `circuit`.

    It hashes the circuit in the circuit format without each instance's
    [`Source`][jolty.ir.Source], the script line that placed it, which names where a
    problem lies and which no simulation reads: the same circuit built by scripts in two
    places, or on two machines, hashes alike.

    Examples:
        >>> from jolty.ir import Circuit, Component, DeviceOf, Instance, Source
        >>> def placed(file: str, resistance: float) -> Circuit:
        ...     resistor = Instance(
        ...         id="R1",
        ...         of=DeviceOf(device="resistor"),
        ...         parameters={"resistance": resistance},
        ...         terminals={"a": "in", "b": "out"},
        ...         source=Source(file=file, line=3),
        ...     )
        ...     return Circuit(
        ...         top="top", components={"top": Component(instances=(resistor,))}
        ...     )
        >>> circuit_hash(placed("a.py", 1e3)) == circuit_hash(placed("b/a.py", 1e3))
        True
        >>> circuit_hash(placed("a.py", 1e3)) == circuit_hash(placed("a.py", 2e3))
        False
    """
    sourceless = {"components": {"__all__": {"instances": {"__all__": {"source"}}}}}
    text = circuit.model_dump_json(exclude=sourceless)
    return hashlib.sha256(text.encode()).hexdigest()


@functools.cache
def build() -> str:
    """A SHA-256 hash of jolty's build: its engine, its Python modules and the Python.

    It hashes the bytes of the engine's library and of every module of the package, by
    their paths within it, and the interpreter's version, since a measurement a script
    writes computes in Python.
    """
    hashed = hashlib.sha256(Path(_core.__file__).read_bytes())
    for module in sorted(_PACKAGE.rglob("*.py")):
        hashed.update(module.relative_to(_PACKAGE).as_posix().encode())
        hashed.update(module.read_bytes())
    hashed.update(sys.version.encode())
    return hashed.hexdigest()


def platform() -> str:
    """The platform: the one Python was built for, and the C library's version.

    The engine's ``exp``, ``ln`` and ``pow`` come from the platform's C library: glibc
    on most Linux systems, whose version names it, and the system's own on macOS and
    Windows, whose versions name it.

    Examples:
        >>> platform() != ""
        True
    """
    library, release = platforms.libc_ver()
    if not library:
        library = platforms.system()
        release = platforms.mac_ver()[0] or platforms.version()
    return f"{sysconfig.get_platform()} {library} {release}"


def plan_hash(plan: Plan) -> str:
    """A SHA-256 hash, in hexadecimal, of `plan`'s analyses, options and conditions.

    It hashes them as their representations spell them.

    Examples:
        >>> from jolty.plan import OperatingPoint
        >>> plan_hash(Plan({"op": OperatingPoint()})) == plan_hash(
        ...     Plan({"op": OperatingPoint()})
        ... )
        True
    """
    return hashlib.sha256(repr(plan).encode()).hexdigest()


class Provenance(BaseModel):
    """Where results come from, as a results file states it.

    The jolty, the build and the platform that simulated them, and the circuit and the
    plan they are of, by their hashes. Every results file states it, and
    [`load_results`][jolty.files.load_results] given a circuit refuses a file whose
    results are of another.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    jolty: str = Field(description="jolty's version.")
    build: str = Field(
        description="The [`build`][jolty.files.build] that simulated them."
    )
    platform: str = Field(
        description="The [`platform`][jolty.files.platform] they were simulated on."
    )
    circuit: str | None = Field(
        default=None,
        description="The [`circuit_hash`][jolty.files.circuit_hash] of the circuit "
        "they are of, when it was given.",
    )
    plan: str | None = Field(
        default=None,
        description="The [`plan_hash`][jolty.files.plan_hash] of the plan they are "
        "of, when it was given.",
    )

    @classmethod
    def of(cls, circuit: ir.Circuit | None = None, plan: Plan | None = None) -> Self:
        """The provenance of results of `circuit` and `plan`, simulated here and now.

        Examples:
            >>> Provenance.of().circuit is None
            True
        """
        return cls(
            jolty=version("jolty"),
            build=build(),
            platform=platform(),
            circuit=None if circuit is None else circuit_hash(circuit),
            plan=None if plan is None else plan_hash(plan),
        )


class _Header(_Names):
    format: Literal["jolty-results"] = "jolty-results"
    version: Literal[10] = 10
    provenance: Provenance
    recordings: tuple[_Recording, ...] = ()
    analyses: dict[str, _Entry]
    excursions: dict[str, tuple[results.Excursion, ...]] = Field(
        default_factory=dict,
        description="Where each analysis the engine watched used logic devices outside "
        "their domains, by its name.",
    )


_VERSION: Final = _Header.model_construct().version
"""The version of the results file's format that jolty writes and reads."""


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


def shards_path(script: Path) -> Path:
    """Where a script's shards are written by default: in ``<stem>.shards``."""
    return script.with_name(script.stem + SHARDS_SUFFIX)


def shard_name(start: int, stop: int) -> str:
    """The name of the file of the samples from `start` up to `stop`.

    Examples:
        >>> shard_name(0, 1000)
        'shard-0-1000.npz'
    """
    return f"shard-{start}-{stop}.npz"


def merged_path(shards: Path) -> Path:
    """Where the batch merged from the shards in `shards` goes: beside the directory.

    Examples:
        >>> merged_path(Path("amp.shards")).name
        'amp.batches.npz'
    """
    return shards.with_name(shards.name.removesuffix(SHARDS_SUFFIX) + BATCHES_SUFFIX)


def batches_path(results: Path) -> Path:
    """Where the batches go beside their plan's results: ``<stem>.batches.npz``.

    Examples:
        >>> batches_path(Path("rc.results.npz")).name
        'rc.batches.npz'
    """
    stem = results.name.removesuffix(RESULTS_SUFFIX).removesuffix(".npz")
    return results.with_name(stem + BATCHES_SUFFIX)


def plot_path(results: Path, analysis: str) -> Path:
    """Where the plot of `analysis` goes beside its results or its batches.

    It is ``<stem>.<name>.plot.png``.

    Examples:
        >>> plot_path(Path("rc.results.npz"), "step").name
        'rc.step.plot.png'
        >>> plot_path(Path("rc.batches.npz"), "monte-carlo").name
        'rc.monte-carlo.plot.png'
    """
    name = results.name
    stem = name.removesuffix(RESULTS_SUFFIX).removesuffix(BATCHES_SUFFIX)
    stem = stem.removesuffix(".npz")
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


def save_results(
    path: Path,
    analyses: Mapping[str, results.AnalysisResult],
    *,
    circuit: ir.Circuit | None = None,
    plan: Plan | None = None,
) -> None:
    """Writes the results of `analyses`, by name, to `path`, of `circuit` and `plan`.

    The file states their [`Provenance`][jolty.files.Provenance]: this jolty, its
    build and the platform, and the hashes of `circuit` and `plan` when they are given.

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
        provenance=Provenance.of(circuit, plan),
        recordings=tuple(_Recording.of(recording) for recording in recordings),
        analyses=entries,
        excursions={
            name: result.excursions
            for name, result in analyses.items()
            if result.excursions is not None
        },
    )
    arrays[_HEADER] = np.array(header.model_dump_json())
    _write(path, arrays)


def _write(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Writes `arrays` to `path` as ``numpy.savez`` does, in bytes they alone decide.

    Each array is an uncompressed ``.npy`` entry of the archive, in the order given, as
    ``numpy.savez`` writes it, but dated as every entry is, where ``numpy.savez`` dates
    it with the time it is written.
    """
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, array in arrays.items():
            entry = zipfile.ZipInfo(f"{name}.npy", date_time=_EPOCH)
            with archive.open(entry, "w", force_zip64=True) as file:
                np.lib.format.write_array(
                    file, np.asanyarray(array), allow_pickle=False
                )


def load_results(
    path: Path, *, circuit: ir.Circuit | None = None
) -> dict[str, results.AnalysisResult]:
    """Reads the results file at `path`: the results of each analysis, by name.

    Given the `circuit` they are to be of, it refuses a file whose
    [`Provenance`][jolty.files.Provenance] names another circuit.

    Raises:
        JoltyError: When the file is not a jolty results file, or one in another
            version of the format, or its results are of a circuit other than `circuit`.
    """
    with np.load(path, allow_pickle=False) as data:
        header = _header(path, data)
        of = header.provenance.circuit
        if circuit is not None and of is not None and of != circuit_hash(circuit):
            message = (
                f"{path} holds the results of another circuit than the one given: "
                f"its circuit's hash is {of[:12]}, the one given's "
                f"{circuit_hash(circuit)[:12]}; simulate the circuit again"
            )
            raise JoltyError(message)
        recordings = tuple(recording.recording() for recording in header.recordings)
        found = {
            name: _load(data, name, entry, header, recordings)
            for name, entry in header.analyses.items()
        }
    return {
        name: result
        if name not in header.excursions
        else replace(result, excursions=header.excursions[name])
        for name, result in found.items()
    }


def load_provenance(path: Path) -> Provenance:
    """The [`Provenance`][jolty.files.Provenance] the results file at `path` states.

    Raises:
        JoltyError: When the file is not a jolty results file, or one in another
            version of the format.
    """
    with np.load(path, allow_pickle=False) as data:
        return _header(path, data).provenance


def _header(path: Path, data: Mapping[str, np.ndarray]) -> _Header:
    """The header of the results file at `path`, which holds `data`.

    Raises:
        JoltyError: When the file is not a jolty results file, or one in another
            version of the format.
    """
    try:
        return _Header.model_validate_json(str(data[_HEADER]))
    except KeyError, ValidationError:
        raise _unreadable(path, data) from None


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
    result: results.AnalysisResult,
    arrays: dict[str, np.ndarray],
    recordings: dict[results.Recording, int],
) -> _Entry:
    """The header's entry of `result`, its arrays added to `arrays` under `name`.

    Its recording joins `recordings`, each recording by its index in the header, unless
    another result's is the same.
    """
    match result:
        case results.OperatingPointResult():
            arrays.update(_point_arrays(name, result))
            recording = _indexed(recordings, result.recording)
            return _OperatingPointResult(strategy=result.strategy, recording=recording)
        case results.DcSweepResult():
            arrays[f"{name}.levels"] = result.levels
            arrays[f"{name}.values"] = result.values
            arrays[f"{name}.logic"] = result.logic
            arrays.update(_observed_arrays(name, result))
            return _DcSweepResult(
                source=result.source,
                quantity=result.quantity,
                parameter=result.parameter,
                recording=_indexed(recordings, result.recording),
            )
        case results.AcResult():
            arrays[f"{name}.frequencies"] = result.frequencies
            arrays[f"{name}.values"] = result.values
            arrays[f"{name}.currents"] = result.currents
            point = result.operating_point
            arrays.update(_point_arrays(f"{name}.operating_point", point))
            return _AcResult(
                source=result.source,
                strategy=point.strategy,
                recording=_indexed(recordings, result.recording),
            )
        case results.TransientResult():
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
            return _TransientResult(
                statistics=result.statistics,
                recording=_indexed(recordings, result.recording),
            )
        case results.OperatingPointsResult():
            for index, point in enumerate(result.points):
                arrays.update(_point_arrays(f"{name}.{index}", point))
            return _OperatingPointsResult(
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
    result: results.SensitivityResult
    | results.StabilityResult
    | results.NoiseResult
    | results.PoleZeroResult
    | results.TransferFunctionResult,
    arrays: dict[str, np.ndarray],
    recordings: dict[results.Recording, int],
) -> _Entry:
    """The header's entry of a small-signal result or of sensitivities."""
    match result:
        case results.TransferFunctionResult():
            arrays[f"{name}.transfer"] = np.array(
                [result.gain, result.input_resistance, result.output_resistance]
            )
            point = result.operating_point
            arrays.update(_point_arrays(f"{name}.operating_point", point))
            return _TransferFunctionResult(
                source=result.source,
                input=result.input,
                output=result.output,
                reference=result.reference,
                strategy=point.strategy,
                recording=_indexed(recordings, point.recording),
            )
        case results.SensitivityResult():
            arrays[f"{name}.parameters"] = np.array(
                [parameter.value for parameter in result.parameters], dtype=float
            )
            arrays[f"{name}.derivatives"] = result.derivatives
            return _SensitivityResult(
                outputs=result.outputs,
                parameters=tuple(
                    (parameter.instance, parameter.name)
                    for parameter in result.parameters
                ),
                result=_save(f"{name}.result", result.result, arrays, recordings),
            )
        case results.StabilityResult():
            arrays[f"{name}.frequencies"] = result.frequencies
            arrays[f"{name}.gains"] = result.gains
            point = result.operating_point
            arrays.update(_point_arrays(f"{name}.operating_point", point))
            return _StabilityResult(
                probe=result.probe,
                strategy=point.strategy,
                recording=_indexed(recordings, point.recording),
            )
        case results.NoiseResult():
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
            return _NoiseResult(
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
        case results.PoleZeroResult():
            arrays[f"{name}.poles"] = result.poles
            if result.zeros is not None:
                arrays[f"{name}.zeros"] = result.zeros
            point = result.operating_point
            arrays.update(_point_arrays(f"{name}.operating_point", point))
            return _PoleZeroResult(
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
) -> results.AnalysisResult:
    """The result stored under `name` in `data`, as `entry` says it is.

    It recorded the one of `recordings` the entry names.
    """
    unknowns, nets = header.unknowns, header.nets
    match entry:
        case _OperatingPointResult(strategy=strategy, recording=recording):
            return _point(data, name, strategy, header, recordings[recording])
        case _OperatingPointsResult(strategies=strategies, recording=recording):
            points = tuple(
                _point(data, f"{name}.{index}", strategy, header, recordings[recording])
                for index, strategy in enumerate(strategies)
            )
            return results.OperatingPointsResult(unknowns, points)
        case _DcSweepResult(source=source, quantity=quantity, recording=recording):
            return results.DcSweepResult(
                unknowns,
                source,
                quantity,
                data[f"{name}.levels"],
                data[f"{name}.values"],
                nets,
                data[f"{name}.logic"],
                **_observed(data, name, recordings[recording]),
                parameter=entry.parameter,
            )
        case _AcResult(source=source, strategy=strategy, recording=recording):
            point = f"{name}.operating_point"
            return results.AcResult(
                unknowns,
                source,
                data[f"{name}.frequencies"],
                data[f"{name}.values"],
                _point(data, point, strategy, header, recordings[recording]),
                recording=recordings[recording],
                currents=data[f"{name}.currents"],
            )
        case _TransientResult(statistics=statistics, recording=recording):
            return results.TransientResult(
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
    entry: _SensitivityResult
    | _StabilityResult
    | _NoiseResult
    | _PoleZeroResult
    | _TransferFunctionResult,
    header: _Header,
    recordings: tuple[results.Recording, ...],
) -> results.AnalysisResult:
    """The small-signal result or the sensitivities stored under `name`."""
    unknowns = header.unknowns
    match entry:
        case _TransferFunctionResult():
            gain, input_resistance, output_resistance = (
                float(value) for value in data[f"{name}.transfer"]
            )
            return results.TransferFunctionResult(
                unknowns,
                entry.source,
                entry.input,
                entry.output,
                entry.reference,
                gain,
                input_resistance,
                output_resistance,
                _point(
                    data,
                    f"{name}.operating_point",
                    entry.strategy,
                    header,
                    recordings[entry.recording],
                ),
            )
        case _SensitivityResult(
            outputs=outputs, parameters=named, result=differentiated
        ):
            parameters = tuple(
                results.Parameter(instance, parameter, float(value))
                for (instance, parameter), value in zip(
                    named, data[f"{name}.parameters"], strict=True
                )
            )
            found = _load(data, f"{name}.result", differentiated, header, recordings)
            if isinstance(
                found,
                results.SensitivityResult
                | results.StabilityResult
                | results.NoiseResult
                | results.PoleZeroResult
                | results.TransferFunctionResult
                | results.OperatingPointsResult,
            ):
                message = (
                    f"the sensitivities {name} differentiate neither an operating "
                    "point, a sweep, a response nor a transient"
                )
                raise JoltyError(message)
            return results.SensitivityResult(
                unknowns, outputs, parameters, data[f"{name}.derivatives"], found
            )
        case _StabilityResult(probe=probe, strategy=strategy, recording=recording):
            point = f"{name}.operating_point"
            return results.StabilityResult(
                unknowns,
                probe,
                data[f"{name}.frequencies"],
                data[f"{name}.gains"],
                _point(data, point, strategy, header, recordings[recording]),
            )
        case _NoiseResult():
            found_input = None
            if entry.input is not None:
                found_input = results.NoiseInput(
                    entry.input.source, entry.input.quantity, data[f"{name}.gains"]
                )
            return results.NoiseResult(
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
        case _PoleZeroResult(
            source=source,
            output=output,
            reference=reference,
            strategy=strategy,
            recording=recording,
        ):
            point = f"{name}.operating_point"
            return results.PoleZeroResult(
                unknowns,
                data[f"{name}.poles"],
                data.get(f"{name}.zeros"),
                source,
                output,
                reference,
                _point(data, point, strategy, header, recordings[recording]),
            )


def _names(result: results.AnalysisResult) -> _Names:
    """The names of the columns of `result`'s circuit."""
    if isinstance(result, results.SensitivityResult):
        return _names(result.result)
    if isinstance(result, results.OperatingPointsResult):
        return _names(result.points[0])
    if isinstance(
        result,
        results.AcResult
        | results.StabilityResult
        | results.NoiseResult
        | results.PoleZeroResult
        | results.TransferFunctionResult,
    ):
        return _names(result.operating_point)
    return _Names(unknowns=result.unknowns, nets=result.nets)


_OBSERVED: Final = ("currents", "charges", "conductances", "capacitances", "readings")
"""The arrays of what the recorded devices did, by the names of their fields."""


def _observed_arrays(
    name: str,
    result: results.OperatingPointResult
    | results.DcSweepResult
    | results.TransientResult,
) -> dict[str, np.ndarray]:
    """The arrays of what the recorded devices did in the analysis `name`."""
    return {f"{name}.{field}": getattr(result, field) for field in _OBSERVED}


def _point_arrays(
    name: str, point: results.OperatingPointResult
) -> dict[str, np.ndarray]:
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
) -> results.OperatingPointResult:
    """The operating point stored under `name` in `data`, which recorded `recording`."""
    return results.OperatingPointResult(
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


@dataclass(frozen=True, slots=True)
class Batch:
    """A batch of runs as a batch file holds it, with the requirements it measured.

    Each of its measurements is a requirement's, which `requirements` states in words,
    bounded by the `bounds` of each, its least and its most value, in SI base units,
    None for a bound it does not state.
    """

    result: results.BatchResult
    requirements: tuple[str, ...]
    bounds: tuple[tuple[float | None, float | None], ...]


class _Traced(_Model):
    analysis: str
    along: Literal["point", "level", "frequency", "time"]
    unknowns: tuple[results.Unknown, ...]
    source: str | None = None
    quantity: results.Quantity | None = None
    parameter: str | None = None


type _Exponents = tuple[str, str, str, str, str]
"""A dimension's exponents of the base units, as fractions."""


type _Unit = _Exponents | results.Plain | None
"""A measurement's unit: a dimension's exponents of the base units, as fractions, a
plain unit, or none."""


def _unit(unit: Dimension | results.Plain | None) -> _Unit:
    """`unit` as a batch file's header holds it."""
    return _exponents(unit) if isinstance(unit, Dimension) else unit


def _dimension(unit: _Unit) -> Dimension | results.Plain | None:
    """The unit a batch file's header holds as `unit`."""
    return _of_exponents(unit) if isinstance(unit, tuple) else unit


def _exponents(dimension: Dimension) -> _Exponents:
    """The exponents of `dimension`, as a file's header holds them."""
    kilogram, metre, second, ampere, kelvin = (
        str(Fraction(exponent))
        for exponent in (
            dimension.kilogram,
            dimension.metre,
            dimension.second,
            dimension.ampere,
            dimension.kelvin,
        )
    )
    return kilogram, metre, second, ampere, kelvin


def _of_exponents(exponents: _Exponents) -> Dimension:
    """The dimension whose exponents a file's header holds as `exponents`."""
    fractions = [Fraction(exponent) for exponent in exponents]
    return Dimension(
        *(
            int(exponent) if exponent.denominator == 1 else exponent
            for exponent in fractions
        )
    )


class _Columns(_Model):
    """What a batch file's header holds of a batch's result besides its arrays."""

    analyses: tuple[str, ...]
    parameters: tuple[str, ...]
    measurements: tuple[str, ...]
    units: tuple[_Unit, ...]
    reasons: tuple[str, ...]
    problems: dict[int, tuple[str, ...]]
    errors: dict[int, dict[str, str]]
    traced: tuple[_Traced, ...]

    @classmethod
    def of(cls, found: results.BatchResult) -> _Columns:
        """What the header holds of `found`."""
        return cls(
            analyses=found.analyses,
            parameters=found.parameters,
            measurements=found.measurements,
            units=tuple(_unit(unit) for unit in found.units),
            reasons=found.reasons,
            problems=found.problems,
            errors=found.errors,
            traced=tuple(
                _Traced(
                    analysis=traced.analysis,
                    along=traced.along,
                    unknowns=traced.unknowns,
                    source=traced.source,
                    quantity=traced.quantity,
                    parameter=traced.parameter,
                )
                for traced in found.traced
            ),
        )

    def result(self, name: str, data: Mapping[str, np.ndarray]) -> results.BatchResult:
        """The result whose arrays `data` holds under keys that start with `name`."""
        traced = tuple(
            results.Traced(
                each.analysis,
                each.along,
                data[f"{name}.axis.{index}"],
                each.unknowns,
                data[f"{name}.traced.{index}"],
                each.source,
                each.quantity,
                each.parameter,
            )
            for index, each in enumerate(self.traced)
        )
        return results.BatchResult(
            self.analyses,
            self.parameters,
            data[f"{name}.values"],
            data[f"{name}.ended"],
            self.measurements,
            tuple(_dimension(unit) for unit in self.units),
            data[f"{name}.measured"],
            data[f"{name}.causes"],
            self.reasons,
            {int(index): tuple(each) for index, each in self.problems.items()},
            {int(index): dict(each) for index, each in self.errors.items()},
            traced,
        )


def _arrays(name: str, found: results.BatchResult) -> dict[str, np.ndarray]:
    """The arrays of `found`, under keys that start with `name`."""
    arrays = {f"{name}.{key}": getattr(found, key) for key in _ARRAYS}
    for index, traced in enumerate(found.traced):
        arrays[f"{name}.traced.{index}"] = traced.values
        arrays[f"{name}.axis.{index}"] = traced.axis
    return arrays


_ARRAYS: Final = ("values", "ended", "measured", "causes")
"""The arrays of a batch's result that every batch has."""


class _Batch(_Columns):
    requirements: tuple[str, ...]
    bounds: tuple[tuple[float | None, float | None], ...]


class _Batches(_Model):
    format: Literal["jolty-batches"] = "jolty-batches"
    version: Literal[1] = 1
    batches: dict[str, _Batch]


def save_batches(path: Path, batches: Mapping[str, Batch]) -> None:
    """Writes `batches`, by name, to `path`."""
    arrays: dict[str, np.ndarray] = {}
    entries = {}
    for name, batch in batches.items():
        arrays |= _arrays(name, batch.result)
        entries[name] = _Batch(
            **_Columns.of(batch.result).model_dump(),
            requirements=batch.requirements,
            bounds=batch.bounds,
        )
    arrays[_HEADER] = np.array(_Batches(batches=entries).model_dump_json())
    _write(path, arrays)


def load_batches(path: Path) -> dict[str, Batch]:
    """Reads the batch file at `path`: each batch, by name.

    Raises:
        JoltyError: When the file is not a jolty batch file of this version.
    """
    with np.load(path, allow_pickle=False) as data:
        try:
            header = _Batches.model_validate_json(str(data[_HEADER]))
        except KeyError, ValidationError:
            message = f"{path} is not a jolty batch file of version 1"
            raise JoltyError(message) from None
        return {
            name: Batch(entry.result(name, data), entry.requirements, entry.bounds)
            for name, entry in header.batches.items()
        }


@dataclass(frozen=True, slots=True)
class Shard:
    """The samples from `start` up to `stop` of a plan's Monte Carlo analysis, run.

    The analysis draws `samples` samples from its `seed`, each of the circuit's
    `toleranced` parameters and the kit's mismatch of its `mismatched` foundry's
    devices, and runs them at the plan's own conditions, then at its `corners`, by
    name. `conditions` holds the batch of the shard's samples at each, row 0 its
    sample `start`, or None where the circuit does not elaborate. `analyses` names
    every analysis of the plan, and `specs` are those its samples were measured
    against, if any, each requirement's measurement the batch's measurement at its
    place. Shards of one run of the analysis share their `fingerprint`, which
    [`jolty.shards`][jolty.shards] makes.
    """

    start: int
    stop: int
    fingerprint: str
    samples: int
    seed: int
    toleranced: int
    mismatched: int
    corners: tuple[str, ...]
    analyses: tuple[str, ...]
    specs: Specs | None
    conditions: tuple[results.BatchResult | None, ...]


class _Bound(_Model):
    value: float
    unit: _Exponents


class _Requirement(_Model):
    """A requirement as a shard's header states it: its measurement and its bounds."""

    analysis: str
    description: str
    plain: results.Plain | None
    minimum: _Bound | None
    maximum: _Bound | None
    margin: _Bound | None

    @classmethod
    def of(cls, requirement: Requirement) -> _Requirement:
        """How the header states `requirement`."""
        measurement = requirement.measurement
        return cls(
            analysis=measurement.analysis,
            description=measurement.description,
            plain=measurement.plain,
            minimum=_bound(requirement.minimum),
            maximum=_bound(requirement.maximum),
            margin=_bound(requirement.margin),
        )

    def requirement(self) -> Requirement:
        """The requirement, whose measurement its shard read: it reads no result."""
        measurement = Measurement(
            self.analysis, self.description, self.plain, _read_by_its_shard
        )
        return Requirement(
            measurement,
            _quantity(self.minimum),
            _quantity(self.maximum),
            _quantity(self.margin),
        )


def _read_by_its_shard(result: results.AnalysisResult) -> Reading:
    """Reads nothing of `result`: a shard's measurement was read where the shard ran.

    Raises:
        MeasurementError: Always.
    """
    del result
    message = "a shard's file holds what its measurements read, and reads no result"
    raise MeasurementError(message)


def _bound(quantity: Quantity | None) -> _Bound | None:
    """`quantity` as a shard's header holds a bound."""
    if quantity is None:
        return None
    return _Bound(value=quantity.value, unit=_exponents(quantity.dimension))


def _quantity(bound: _Bound | None) -> Quantity | None:
    """The quantity a shard's header holds as `bound`."""
    return None if bound is None else Quantity(bound.value, _of_exponents(bound.unit))


class _Shard(_Model):
    format: Literal["jolty-shard"] = "jolty-shard"
    version: Literal[1] = 1
    start: int
    stop: int
    fingerprint: str
    samples: int
    seed: int
    toleranced: int
    mismatched: int
    corners: tuple[str, ...]
    analyses: tuple[str, ...]
    requirements: tuple[_Requirement, ...] | None
    yield_target: float | None
    conditions: tuple[_Columns | None, ...]


def save_shard(path: Path, shard: Shard) -> None:
    """Writes `shard` to `path`."""
    arrays: dict[str, np.ndarray] = {}
    for place, found in enumerate(shard.conditions):
        if found is not None:
            arrays |= _arrays(str(place), found)
    header = _Shard(
        start=shard.start,
        stop=shard.stop,
        fingerprint=shard.fingerprint,
        samples=shard.samples,
        seed=shard.seed,
        toleranced=shard.toleranced,
        mismatched=shard.mismatched,
        corners=shard.corners,
        analyses=shard.analyses,
        requirements=None
        if shard.specs is None
        else tuple(_Requirement.of(each) for each in shard.specs.requirements),
        yield_target=None if shard.specs is None else shard.specs.yield_target,
        conditions=tuple(
            None if found is None else _Columns.of(found) for found in shard.conditions
        ),
    )
    arrays[_HEADER] = np.array(header.model_dump_json())
    _write(path, arrays)


def load_shard(path: Path) -> Shard:
    """Reads the shard file at `path`.

    Raises:
        JoltyError: When the file is not a jolty shard file of this version.
    """
    unreadable = f"{path} is not a jolty shard file of version 1"
    try:
        archive = np.load(path, allow_pickle=False)
    except ValueError, OSError, zipfile.BadZipFile:
        raise JoltyError(unreadable) from None
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise JoltyError(unreadable)
    with archive as data:
        try:
            header = _Shard.model_validate_json(str(data[_HEADER]))
        except KeyError, ValidationError:
            raise JoltyError(unreadable) from None
        requirements = header.requirements
        return Shard(
            start=header.start,
            stop=header.stop,
            fingerprint=header.fingerprint,
            samples=header.samples,
            seed=header.seed,
            toleranced=header.toleranced,
            mismatched=header.mismatched,
            corners=header.corners,
            analyses=header.analyses,
            specs=None
            if requirements is None
            else Specs(
                [each.requirement() for each in requirements], header.yield_target
            ),
            conditions=tuple(
                None if entry is None else entry.result(str(place), data)
                for place, entry in enumerate(header.conditions)
            ),
        )
