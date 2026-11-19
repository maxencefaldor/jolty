# Copyright 2026 Maxence Faldor
"""Circuit, results, batch and shard files, and SPICE's raw files.

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
simulated them, and the digests of the circuit and the plan they are of. The header is
in version 20 of the format, which holds that provenance, the regions of operation
each recorded transistor's model distinguishes, the Newton iterations that found each
operating point, each recording's summed terminals as its ``summed_terminals``, a
periodic steady state as an analysis of its own, a mismatch analysis's random
variables, and each transient's integrated residuals (`charge_leak`); a file of another
version is not read.

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

import contextlib
import enum
import functools
import hashlib
import json
import math
import platform as platforms
import struct
import sys
import sysconfig
import zipfile
import zlib
from collections.abc import Generator, Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path, PurePath
from typing import Annotated, Final, Literal, Self, override

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jolty import _core, ir, results
from jolty.errors import JoltyError
from jolty.plan import Plan
from jolty.specs import Measure, MeasureError, Reading, Requirement, Specs
from jolty.spice import ngspice
from jolty.units import Dimension, Quantity

CIRCUIT_SUFFIX: Final = ".jolty.json"
"""The suffix of circuit files, such as ``rc_filter.jolty.json``."""

RESULTS_SUFFIX: Final = ".results.npz"
"""The suffix of results files, such as ``rc_filter.results.npz``."""

RAW_SUFFIX: Final = ".raw"
"""The suffix of SPICE raw files, such as ``rc_filter.raw``."""

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


class _Found(_Model):
    """How an operating point was found: its strategy, and its Newton iterations."""

    strategy: str
    iterations: int | None = None

    @classmethod
    def of(cls, point: results.OperatingPointResult) -> Self:
        """How `point` was found."""
        return cls(strategy=point.strategy, iterations=point.iterations)


class _OperatingPointResult(_Model):
    kind: Literal["operating_point"] = "operating_point"
    found: _Found
    recording: int


class _OperatingPointsResult(_Model):
    kind: Literal["operating_points"] = "operating_points"
    found: tuple[_Found, ...]
    recording: int


class _DcSweepResult(_Model):
    kind: Literal["dc_sweep"] = "dc_sweep"
    source: str
    nature: results.Nature | None
    parameter: str | None = None
    strategies: tuple[str, ...] = ()
    iterations: tuple[int, ...] = ()
    recording: int


class _AcResult(_Model):
    kind: Literal["ac"] = "ac"
    source: str
    found: _Found
    recording: int


class _Statistics(_Model):
    """How a transient or periodic steady state went, as the header stores it."""

    initial: str | None
    accepted: int
    rejected: int
    landed: int
    iterations: int
    work: results.Work | None = None
    profile: results.Profile | None = None
    charge_leak: bool = Field(
        default=False,
        description="Whether the arrays hold each unknown's integrated residual.",
    )

    @classmethod
    def from_statistics(cls, statistics: results.Statistics) -> Self:
        """The header's entry of `statistics`."""
        return cls(
            initial=statistics.initial,
            accepted=statistics.accepted,
            rejected=statistics.rejected,
            landed=statistics.landed,
            iterations=statistics.iterations,
            work=statistics.work,
            profile=statistics.profile,
            charge_leak=statistics.charge_leak is not None,
        )


class _TransientResult(_Model):
    kind: Literal["transient"] = "transient"
    statistics: _Statistics
    recording: int


class _PeriodicSteadyStateResult(_Model):
    kind: Literal["periodic_steady_state"] = "periodic_steady_state"
    period: float
    periods: int
    statistics: _Statistics
    recording: int


class _SensitivityResult(_Model):
    kind: Literal["sensitivity"] = "sensitivity"
    outputs: tuple[str, ...]
    parameters: tuple[tuple[str, str], ...]
    period: bool = Field(
        default=False,
        description="Whether the arrays hold an oscillator's period's derivatives.",
    )
    result: _Entry


class _MismatchResult(_Model):
    kind: Literal["mismatch"] = "mismatch"
    outputs: tuple[str, ...]
    random_variables: tuple[ir.RandomVariable, ...]
    period: bool = Field(
        default=False,
        description="Whether the arrays hold an oscillator's period's derivatives.",
    )
    result: _Entry


class _StabilityResult(_Model):
    kind: Literal["stability"] = "stability"
    probe: str
    found: _Found
    recording: int


class _NoiseInput(_Model):
    source: str
    nature: results.Nature


class _NoiseResult(_Model):
    kind: Literal["noise"] = "noise"
    output: str
    reference: str | None
    nature: results.Nature
    sources: tuple[tuple[str, str], ...]
    input: _NoiseInput | None
    found: _Found
    recording: int


class _PoleZeroResult(_Model):
    kind: Literal["pole_zero"] = "pole_zero"
    source: str | None
    output: str | None
    reference: str | None
    found: _Found
    recording: int


class _TransferFunctionResult(_Model):
    kind: Literal["transfer_function"] = "transfer_function"
    source: str
    input: results.Nature
    output: str
    reference: str | None
    found: _Found
    recording: int


type _Entry = Annotated[
    _OperatingPointResult
    | _OperatingPointsResult
    | _DcSweepResult
    | _AcResult
    | _TransientResult
    | _PeriodicSteadyStateResult
    | _SensitivityResult
    | _MismatchResult
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
    summed_terminals: tuple[str, ...] = ()
    thermal: tuple[str, ...] = ()

    @classmethod
    def from_recording(cls, recording: results.Recording) -> _Recording:
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
            summed_terminals=recording.summed_terminals,
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
            self.summed_terminals,
            frozenset(self.thermal),
        )


_PACKAGE: Final = Path(__file__).resolve().parent
"""jolty's own directory, whose modules and engine a build is."""


def circuit_digest(circuit: ir.Circuit) -> str:
    """A SHA-256 digest, in hexadecimal, of what a simulation reads of `circuit`.

    It hashes the circuit in the circuit format without each instance's
    [`SourceLocation`][jolty.ir.SourceLocation], the script line that placed it, which
    names where a problem lies and which no simulation reads: the same circuit built by
    scripts in two places, or on two machines, hashes alike.

    Examples:
        >>> from jolty.ir import Circuit, Component, DeviceOf, Instance, SourceLocation
        >>> def placed(file: str, resistance: float) -> Circuit:
        ...     resistor = Instance(
        ...         name="R1",
        ...         of=DeviceOf(model="resistor"),
        ...         parameters={"resistance": resistance},
        ...         terminals={"a": "in", "b": "out"},
        ...         source_location=SourceLocation(file=file, line=3),
        ...     )
        ...     return Circuit(
        ...         top="top", components={"top": Component(instances=(resistor,))}
        ...     )
        >>> circuit_digest(placed("a.py", 1e3)) == circuit_digest(placed("b/a.py", 1e3))
        True
        >>> circuit_digest(placed("a.py", 1e3)) == circuit_digest(placed("a.py", 2e3))
        False
    """
    sourceless = {
        "components": {"__all__": {"instances": {"__all__": {"source_location"}}}}
    }
    text = circuit.model_dump_json(exclude=sourceless)
    return hashlib.sha256(text.encode()).hexdigest()


@functools.cache
def build() -> str:
    """A SHA-256 digest of jolty's build: its engine, its Python modules and the Python.

    It hashes the bytes of the engine's library and of every module of the package, by
    their paths within it, and the interpreter's version, since a measure a script
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


def _type_name(value: object) -> str:
    """Returns the module-qualified type name of `value`."""
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _canonical_json(value: object) -> str:
    """Serializes a canonicalized value into compact, key-sorted JSON text."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_float(value: float) -> object:
    """Canonicalizes a Python float, encoding NaN and infinities deterministically."""
    if math.isnan(value):
        return {"__float__": "nan"}
    if math.isinf(value):
        return {"__float__": "inf" if value > 0 else "-inf"}
    return 0.0 if value == 0 else value


def _canonical_scalar(value: object) -> tuple[bool, object]:
    """Canonicalizes primitive scalars, returning `(matched, converted)`."""
    if value is None or isinstance(value, bool | int | str):
        return True, value
    if isinstance(value, float):
        return True, _canonical_float(value)
    if isinstance(value, complex):
        return True, {
            "__type__": "complex",
            "real": _canonical(value.real),
            "imag": _canonical(value.imag),
        }
    if isinstance(value, np.generic):
        return True, _canonical(value.item())
    return False, None


def _canonical_special(value: object) -> tuple[bool, object]:
    """Canonicalizes arrays, binary buffers, paths, fractions, and enums."""
    if isinstance(value, np.ndarray):
        return True, {
            "__type__": "numpy.ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": _canonical(value.tolist()),
        }
    if isinstance(value, bytes | bytearray | memoryview):
        return True, {"__type__": "bytes", "hex": bytes(value).hex()}
    if isinstance(value, enum.Enum):
        return True, {"__type__": _type_name(value), "value": _canonical(value.value)}
    if isinstance(value, PurePath):
        return True, {"__type__": "pathlib.PurePath", "value": value.as_posix()}
    if isinstance(value, Fraction):
        return True, {
            "__type__": "fractions.Fraction",
            "numerator": value.numerator,
            "denominator": value.denominator,
        }
    return False, None


def _canonical_mapping[K, V](value: Mapping[K, V]) -> object:
    """Canonicalizes any mapping independently of key insertion order."""
    if all(isinstance(key, str) for key in value):
        return {
            str(key): _canonical(entry)
            for key, entry in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    entries = [[_canonical(key), _canonical(entry)] for key, entry in value.items()]
    entries.sort(key=lambda pair: _canonical_json(pair[0]))
    return {"__type__": "mapping", "items": entries}


def _canonical_collection(value: object) -> object:
    """Canonicalizes mappings, sequences, sets, and arbitrary fallback objects."""
    if isinstance(value, Mapping):
        return _canonical_mapping(value)
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_canonical(item) for item in value]}
    if isinstance(value, set | frozenset):
        items = [_canonical(item) for item in value]
        items.sort(key=_canonical_json)
        return {"__type__": "set", "items": items}
    return {"__type__": _type_name(value), "repr": repr(value)}


def _canonical(value: object) -> object:
    """Converts `value` into a deterministic, JSON-serializable structure."""
    matched, converted = _canonical_scalar(value)
    if matched:
        return converted
    matched, converted = _canonical_special(value)
    if matched:
        return converted
    if isinstance(value, BaseModel):
        return {
            "__type__": _type_name(value),
            "fields": {
                name: _canonical(getattr(value, name))
                for name in sorted(type(value).model_fields)
            },
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": _type_name(value),
            "fields": {
                item.name: _canonical(getattr(value, item.name))
                for item in sorted(fields(value), key=lambda entry: entry.name)
            },
        }
    return _canonical_collection(value)


def plan_digest(plan: Plan) -> str:
    """Returns a SHA-256 hex digest of `plan`'s analyses, options, and conditions.

    Canonical JSON serialization with sorted mapping keys makes the digest independent
    of dictionary insertion order across analyses, corners, fidelity scopes, overrides,
    and corner arguments.

    Examples:
        >>> from jolty.plan import OperatingPoint
        >>> plan_digest(Plan({"op": OperatingPoint()})) == plan_digest(
        ...     Plan({"op": OperatingPoint()})
        ... )
        True
    """
    text = _canonical_json(_canonical(plan))
    return hashlib.sha256(text.encode()).hexdigest()


class Provenance(BaseModel):
    """Where results come from, as a results file states it.

    The jolty, the build and the platform that simulated them, and the circuit and the
    plan they are of, by their digests. Every results file states it, and
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
        description="The [`circuit_digest`][jolty.files.circuit_digest] of the circuit "
        "they are of, when it was given.",
    )
    plan: str | None = Field(
        default=None,
        description="The [`plan_digest`][jolty.files.plan_digest] of the plan they are "
        "of, when it was given.",
    )

    @classmethod
    def from_circuit_and_plan(
        cls, circuit: ir.Circuit | None = None, plan: Plan | None = None
    ) -> Self:
        """The provenance of results of `circuit` and `plan`, simulated here and now.

        Examples:
            >>> Provenance.from_circuit_and_plan().circuit is None
            True
        """
        return cls(
            jolty=version("jolty"),
            build=build(),
            platform=platform(),
            circuit=None if circuit is None else circuit_digest(circuit),
            plan=None if plan is None else plan_digest(plan),
        )


class _Header(_Names):
    format: Literal["jolty-results"] = "jolty-results"
    version: Literal[20] = 20
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


def raw_path(script: Path) -> Path:
    """Where a script's SPICE raw file goes by default: as ``<stem>.raw``."""
    return script.with_name(script.stem + RAW_SUFFIX)


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
        provenance=Provenance.from_circuit_and_plan(circuit, plan),
        recordings=tuple(
            _Recording.from_recording(recording) for recording in recordings
        ),
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


_ARCHIVE_ERRORS: Final = (
    ValueError,
    EOFError,
    zipfile.BadZipFile,
    zlib.error,
    struct.error,
    OSError,
    NotImplementedError,
    RuntimeError,
)
"""Exceptions raised by NumPy or zipfile when reading a corrupt `.npz` archive."""

_LOAD_ERRORS: Final = (
    *_ARCHIVE_ERRORS,
    KeyError,
    IndexError,
    TypeError,
    ZeroDivisionError,
    OverflowError,
)
"""Exceptions raised when decoding arrays or header metadata from a corrupt archive."""


class _Archive(Mapping[str, np.ndarray]):
    """Read-only view of an `.npz` archive requiring every entry to be an `ndarray`.

    `NpzFile.__getitem__` silently returns raw `bytes` for zip entries that do not hold
    a valid `.npy` header. Rejecting non-array entries here prevents raw bytes from
    escaping as `AttributeError` later during decoding or merging.
    """

    def __init__(self, archive: np.lib.npyio.NpzFile) -> None:
        self._archive = archive

    @override
    def __getitem__(self, key: str) -> np.ndarray:
        value = self._archive[key]
        if not isinstance(value, np.ndarray):
            message = f"archive entry `{key}` is {type(value).__name__}, not ndarray"
            raise TypeError(message)
        return value

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._archive.files)

    @override
    def __len__(self) -> int:
        return len(self._archive.files)


@contextlib.contextmanager
def _open_archive(path: Path, unreadable: str) -> Generator[Mapping[str, np.ndarray]]:
    """Opens the NumPy `.npz` archive at `path`, raising `JoltyError` if corrupt.

    Opening the file handle explicitly ensures no file descriptor leaks when `np.load`
    fails midway through parsing a truncated or corrupt archive. File-system errors
    indicating `path` is missing, a directory, or unreadable due to permissions
    propagate as `OSError` so callers can distinguish missing files from corrupt data.
    """
    try:
        handle = path.open("rb")
    except FileNotFoundError, IsADirectoryError, PermissionError:
        raise
    except OSError:
        raise JoltyError(unreadable) from None
    with handle:
        try:
            archive = np.load(handle, allow_pickle=False)
        except _ARCHIVE_ERRORS:
            raise JoltyError(unreadable) from None
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise JoltyError(unreadable)
        try:
            yield _Archive(archive)
        finally:
            archive.close()


def _header_text(data: Mapping[str, np.ndarray], unreadable: str) -> str:
    """Extracts the JSON header string from `data`, raising `JoltyError` if corrupt."""
    try:
        raw = data[_HEADER]
    except (KeyError, TypeError, *_ARCHIVE_ERRORS):
        raise JoltyError(unreadable) from None
    if raw.ndim != 0 or raw.dtype.kind not in {"U", "S"}:
        raise JoltyError(unreadable)
    value = raw.item()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            raise JoltyError(unreadable) from None
    return str(value)


def _recording_at(
    recordings: tuple[results.Recording, ...], index: int
) -> results.Recording:
    """Returns `recordings[index]` with bounds checking against negative indices."""
    if not 0 <= index < len(recordings):
        message = f"recording index {index} is out of range for {len(recordings)}"
        raise IndexError(message)
    return recordings[index]


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
    unreadable = f"{path} is not a jolty results file"
    with _open_archive(path, unreadable) as data:
        header = _header(path, data)
        of = header.provenance.circuit
        if circuit is not None and of is not None and of != circuit_digest(circuit):
            message = (
                f"{path} holds the results of another circuit than the one given: "
                f"its circuit's hash is {of[:12]}, the one given's "
                f"{circuit_digest(circuit)[:12]}; simulate the circuit again"
            )
            raise JoltyError(message)
        recordings = tuple(recording.recording() for recording in header.recordings)
        try:
            found = {
                name: _load(data, name, entry, header, recordings)
                for name, entry in header.analyses.items()
            }
        except JoltyError:
            raise
        except _LOAD_ERRORS:
            raise JoltyError(unreadable) from None
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
    with _open_archive(path, f"{path} is not a jolty results file") as data:
        return _header(path, data).provenance


def _header(path: Path, data: Mapping[str, np.ndarray]) -> _Header:
    """The header of the results file at `path`, which holds `data`.

    Raises:
        JoltyError: When the file is not a jolty results file, or one in another
            version of the format.
    """
    unreadable = f"{path} is not a jolty results file"
    text = _header_text(data, unreadable)
    try:
        return _Header.model_validate_json(text)
    except ValidationError:
        raise _unreadable(path, text) from None


def _unreadable(path: Path, text: str) -> JoltyError:
    """Why the file at `path`, holding header `text`, cannot be read as results."""
    try:
        version = _Format.model_validate_json(text).version
    except ValidationError:
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
            return _OperatingPointResult(found=_Found.of(result), recording=recording)
        case results.DcSweepResult():
            arrays[f"{name}.swept"] = result.swept
            arrays[f"{name}.values"] = result.values
            arrays[f"{name}.logic"] = result.logic
            arrays.update(_observed_arrays(name, result))
            return _DcSweepResult(
                source=result.source,
                nature=result.nature,
                parameter=result.parameter,
                strategies=result.strategies,
                iterations=result.iterations,
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
                found=_Found.of(point),
                recording=_indexed(recordings, result.recording),
            )
        case results.TransientResult():
            return _save_waveforms(name, result, arrays, recordings)
        case results.OperatingPointsResult():
            for index, point in enumerate(result.points):
                arrays.update(_point_arrays(f"{name}.{index}", point))
            return _OperatingPointsResult(
                found=tuple(_Found.of(point) for point in result.points),
                recording=_indexed(recordings, result.points[0].recording),
            )
        case _:
            return _save_linear(name, result, arrays, recordings)


def _save_waveforms(
    name: str,
    result: results.TransientResult,
    arrays: dict[str, np.ndarray],
    recordings: dict[results.Recording, int],
) -> _TransientResult | _PeriodicSteadyStateResult:
    """The entry of `result`, a transient's or a period's waveforms, and its arrays."""
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
    if result.statistics.charge_leak is not None:
        arrays[f"{name}.charge_leak.signed"] = result.statistics.charge_leak.signed
        arrays[f"{name}.charge_leak.absolute"] = result.statistics.charge_leak.absolute
    statistics = _Statistics.from_statistics(result.statistics)
    recording = _indexed(recordings, result.recording)
    if isinstance(result, results.PeriodicSteadyStateResult):
        return _PeriodicSteadyStateResult(
            period=result.period,
            periods=result.periods,
            statistics=statistics,
            recording=recording,
        )
    return _TransientResult(statistics=statistics, recording=recording)


def _indexed(
    recordings: dict[results.Recording, int], recording: results.Recording
) -> int:
    """The index of `recording` among `recordings`, which it joins if it is new."""
    return recordings.setdefault(recording, len(recordings))


def _save_linear(  # noqa: C901 - one case per kind of result
    name: str,
    result: results.SensitivityResult
    | results.MismatchResult
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
                found=_Found.of(point),
                recording=_indexed(recordings, point.recording),
            )
        case results.SensitivityResult():
            arrays[f"{name}.parameters"] = np.array(
                [parameter.value for parameter in result.parameters], dtype=float
            )
            arrays[f"{name}.derivatives"] = result.derivatives
            if result.period_derivatives is not None:
                arrays[f"{name}.period_derivatives"] = result.period_derivatives
            return _SensitivityResult(
                outputs=result.outputs,
                parameters=tuple(
                    (parameter.instance, parameter.name)
                    for parameter in result.parameters
                ),
                period=result.period_derivatives is not None,
                result=_save(f"{name}.result", result.result, arrays, recordings),
            )
        case results.MismatchResult():
            arrays[f"{name}.derivatives"] = result.derivatives
            if result.period_derivatives is not None:
                arrays[f"{name}.period_derivatives"] = result.period_derivatives
            return _MismatchResult(
                outputs=result.outputs,
                random_variables=result.random_variables,
                period=result.period_derivatives is not None,
                result=_save(f"{name}.result", result.result, arrays, recordings),
            )
        case results.StabilityResult():
            arrays[f"{name}.frequencies"] = result.frequencies
            arrays[f"{name}.gains"] = result.gains
            point = result.operating_point
            arrays.update(_point_arrays(f"{name}.operating_point", point))
            return _StabilityResult(
                probe=result.probe,
                found=_Found.of(point),
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
                    source=result.input.source, nature=result.input.nature
                )
            return _NoiseResult(
                output=result.output,
                reference=result.reference,
                nature=result.nature,
                sources=tuple(
                    (source.instance, source.name) for source in result.sources
                ),
                input=found,
                found=_Found.of(point),
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
                found=_Found.of(point),
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
        case _OperatingPointResult(found=found, recording=recording):
            return _point(
                data, name, found, header, _recording_at(recordings, recording)
            )
        case _OperatingPointsResult(found=found, recording=recording):
            resolved = _recording_at(recordings, recording)
            points = tuple(
                _point(data, f"{name}.{index}", each, header, resolved)
                for index, each in enumerate(found)
            )
            return results.OperatingPointsResult(unknowns, points)
        case _DcSweepResult(source=source, nature=nature, recording=recording):
            return results.DcSweepResult(
                unknowns,
                source,
                nature,
                data[f"{name}.swept"],
                data[f"{name}.values"],
                nets,
                data[f"{name}.logic"],
                **_observed(data, name, _recording_at(recordings, recording)),
                parameter=entry.parameter,
                strategies=entry.strategies,
                iterations=entry.iterations,
            )
        case _AcResult(source=source, found=found, recording=recording):
            point = f"{name}.operating_point"
            resolved = _recording_at(recordings, recording)
            return results.AcResult(
                unknowns,
                source,
                data[f"{name}.frequencies"],
                data[f"{name}.values"],
                _point(data, point, found, header, resolved),
                recording=resolved,
                currents=data[f"{name}.currents"],
            )
        case _TransientResult() | _PeriodicSteadyStateResult():
            return _load_waveforms(data, name, entry, header, recordings)
        case _:
            return _load_linear(data, name, entry, header, recordings)


def _load_waveforms(
    data: Mapping[str, np.ndarray],
    name: str,
    entry: _TransientResult | _PeriodicSteadyStateResult,
    header: _Header,
    recordings: tuple[results.Recording, ...],
) -> results.TransientResult:
    """The waveforms stored under `name` in `data`, a transient's or a period's."""
    leak: results.ChargeLeak | None = None
    if entry.statistics.charge_leak:
        signed = data[f"{name}.charge_leak.signed"]
        absolute = data[f"{name}.charge_leak.absolute"]
        expected = (len(header.unknowns),)
        if (
            signed.shape != expected
            or absolute.shape != expected
            or not np.issubdtype(signed.dtype, np.floating)
            or not np.issubdtype(absolute.dtype, np.floating)
        ):
            message = f"charge leak arrays do not match {expected}"
            raise ValueError(message)
        leak = results.ChargeLeak(signed, absolute)
    statistics = results.Statistics(
        entry.statistics.initial,
        entry.statistics.accepted,
        entry.statistics.rejected,
        entry.statistics.landed,
        entry.statistics.iterations,
        entry.statistics.work,
        entry.statistics.profile,
        leak,
    )
    fields = (
        header.unknowns,
        data[f"{name}.times"],
        data[f"{name}.values"],
        statistics,
        header.nets,
        _traces(data, name),
    )
    observed = _observed(data, name, _recording_at(recordings, entry.recording))
    if isinstance(entry, _PeriodicSteadyStateResult):
        return results.PeriodicSteadyStateResult(
            *fields, **observed, period=entry.period, periods=entry.periods
        )
    return results.TransientResult(*fields, **observed)


def _load_linear(
    data: Mapping[str, np.ndarray],
    name: str,
    entry: _SensitivityResult
    | _MismatchResult
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
            transfer = data[f"{name}.transfer"]
            if transfer.ndim != 1 or len(transfer) != 3:  # noqa: PLR2004 - gain, input and output resistance
                message = f"{name}.transfer has shape {transfer.shape}, not (3,)"
                raise ValueError(message)
            gain, input_resistance, output_resistance = (
                float(value) for value in transfer
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
                    entry.found,
                    header,
                    _recording_at(recordings, entry.recording),
                ),
            )
        case _SensitivityResult(
            outputs=outputs, parameters=named, period=period, result=differentiated
        ):
            raw_parameters = data[f"{name}.parameters"]
            if raw_parameters.ndim != 1 or len(raw_parameters) != len(named):
                message = (
                    f"{name}.parameters has shape {raw_parameters.shape}, not "
                    f"({len(named)},)"
                )
                raise ValueError(message)
            parameters = tuple(
                results.Parameter(instance, parameter, float(value))
                for (instance, parameter), value in zip(
                    named, raw_parameters, strict=True
                )
            )
            found = _differentiated(
                _load(data, f"{name}.result", differentiated, header, recordings), name
            )
            return results.SensitivityResult(
                unknowns,
                outputs,
                parameters,
                data[f"{name}.derivatives"],
                found,
                data[f"{name}.period_derivatives"] if period else None,
            )
        case _MismatchResult(
            outputs=outputs,
            random_variables=random_variables,
            period=period,
            result=differentiated,
        ):
            found = _differentiated(
                _load(data, f"{name}.result", differentiated, header, recordings), name
            )
            return results.MismatchResult(
                unknowns,
                outputs,
                random_variables,
                data[f"{name}.derivatives"],
                found,
                data[f"{name}.period_derivatives"] if period else None,
            )
        case _StabilityResult(probe=probe, found=found, recording=recording):
            point = f"{name}.operating_point"
            return results.StabilityResult(
                unknowns,
                probe,
                data[f"{name}.frequencies"],
                data[f"{name}.gains"],
                _point(
                    data, point, found, header, _recording_at(recordings, recording)
                ),
            )
        case _NoiseResult():
            found_input = None
            if entry.input is not None:
                found_input = results.NoiseInput(
                    entry.input.source, entry.input.nature, data[f"{name}.gains"]
                )
            return results.NoiseResult(
                unknowns,
                entry.output,
                entry.reference,
                entry.nature,
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
                    entry.found,
                    header,
                    _recording_at(recordings, entry.recording),
                ),
            )
        case _PoleZeroResult(
            source=source,
            output=output,
            reference=reference,
            found=found,
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
                _point(
                    data, point, found, header, _recording_at(recordings, recording)
                ),
            )


def _differentiated(
    found: results.AnalysisResult, name: str
) -> (
    results.OperatingPointResult
    | results.DcSweepResult
    | results.AcResult
    | results.TransientResult
):
    """`found`, the result the analysis `name` differentiates.

    Raises:
        JoltyError: When it is no operating point, sweep, response or transient.
    """
    if isinstance(
        found,
        results.OperatingPointResult
        | results.DcSweepResult
        | results.AcResult
        | results.TransientResult,
    ):
        return found
    message = (
        f"the analysis {name} differentiates neither an operating point, a sweep, a "
        "response nor a transient"
    )
    raise JoltyError(message)


def _names(result: results.AnalysisResult) -> _Names:
    """The names of the columns of `result`'s circuit."""
    if isinstance(result, results.SensitivityResult | results.MismatchResult):
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


_OBSERVED: Final = ("currents", "charges", "conductances", "capacitances", "variables")
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
) -> results.RecordedValues:
    """What the devices `recording` names did in the analysis stored under `name`."""
    return {
        "recording": recording,
        "currents": data[f"{name}.currents"],
        "charges": data[f"{name}.charges"],
        "conductances": data[f"{name}.conductances"],
        "capacitances": data[f"{name}.capacitances"],
        "variables": data[f"{name}.variables"],
    }


def _point(
    data: Mapping[str, np.ndarray],
    name: str,
    found: _Found,
    header: _Header,
    recording: results.Recording,
) -> results.OperatingPointResult:
    """The operating point stored under `name` in `data`, found as `found` says.

    It recorded `recording`.
    """
    return results.OperatingPointResult(
        header.unknowns,
        data[f"{name}.values"],
        found.strategy,
        found.iterations,
        header.nets,
        data[f"{name}.logic"],
        **_observed(data, name, recording),
    )


def _traces(
    data: Mapping[str, np.ndarray], name: str
) -> tuple[results.LogicTrace, ...]:
    """The logic traces of the transient `name`, stored end to end in `data`."""
    counts = data[f"{name}.logic.counts"]
    if (
        counts.ndim != 1
        or not np.issubdtype(counts.dtype, np.integer)
        or bool(np.any(counts < 0))
    ):
        message = f"{name}.logic.counts is not a 1-D array of non-negative integers"
        raise ValueError(message)
    times_raw = data[f"{name}.logic.times"]
    levels_raw = data[f"{name}.logic.levels"]
    if times_raw.ndim != 1 or levels_raw.ndim != 1:
        message = f"{name}.logic traces are not 1-D arrays"
        raise ValueError(message)
    total = sum(int(count) for count in counts)
    if len(times_raw) != total or len(levels_raw) != total:
        message = (
            f"{name}.logic traces lengths ({len(times_raw)}, {len(levels_raw)}) "
            f"do not match counts sum {total}"
        )
        raise ValueError(message)
    ends = np.cumsum(counts, dtype=np.intp)
    times = np.split(times_raw, ends[:-1])
    levels = np.split(levels_raw, ends[:-1])
    return tuple(
        results.LogicTrace(times[index], levels[index]) for index in range(len(counts))
    )


@dataclass(frozen=True, slots=True)
class Batch:
    """A batch of runs as a batch file holds it, with the requirements it measured.

    Each of its measures is a requirement's, which `requirements` states in words,
    bounded by the `bounds` of each, its least and its most value, in SI base units,
    None for a bound it does not state.
    """

    result: results.BatchResult
    requirements: tuple[str, ...]
    bounds: tuple[tuple[float | None, float | None], ...]


class _Traces(_Model):
    analysis: str
    along: Literal["point", "swept", "frequency", "time"]
    unknowns: tuple[results.Unknown, ...]
    source: str | None = None
    nature: results.Nature | None = None
    parameter: str | None = None


type _Exponents = tuple[str, str, str, str, str]
"""A dimension's exponents of the base units, as fractions."""


type _Unit = _Exponents | results.Plain | None
"""A measure's unit: a dimension's exponents of the base units, as fractions, a
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
    measures: tuple[str, ...]
    units: tuple[_Unit, ...]
    reasons: tuple[str, ...]
    problems: dict[int, tuple[str, ...]]
    errors: dict[int, dict[str, str]]
    traces: tuple[_Traces, ...]

    @classmethod
    def from_batch(cls, found: results.BatchResult) -> _Columns:
        """What the header holds of `found`."""
        return cls(
            analyses=found.analyses,
            parameters=found.parameters,
            measures=found.measures,
            units=tuple(_unit(unit) for unit in found.units),
            reasons=found.reasons,
            problems=found.problems,
            errors=found.errors,
            traces=tuple(
                _Traces(
                    analysis=traces.analysis,
                    along=traces.along,
                    unknowns=traces.unknowns,
                    source=traces.source,
                    nature=traces.nature,
                    parameter=traces.parameter,
                )
                for traces in found.traces
            ),
        )

    def result(self, name: str, data: Mapping[str, np.ndarray]) -> results.BatchResult:
        """The result whose arrays `data` holds under keys that start with `name`."""
        values = data[f"{name}.values"]
        ended = data[f"{name}.ended"]
        measured = data[f"{name}.measured"]
        causes = data[f"{name}.causes"]
        if values.ndim != 2:  # noqa: PLR2004 - samples by parameters
            message = f"{name}.values has shape {values.shape}, not 2-D"
            raise ValueError(message)
        runs = values.shape[0]
        expected = (
            ("values", values, (runs, len(self.parameters))),
            ("ended", ended, (runs, len(self.analyses))),
            ("measured", measured, (runs, len(self.measures))),
            ("causes", causes, (runs, len(self.measures))),
        )
        for label, array, shape in expected:
            if array.shape != shape:
                message = f"{name}.{label} has shape {array.shape}, not {shape}"
                raise ValueError(message)
        traces = tuple(
            results.Traces(
                each.analysis,
                each.along,
                data[f"{name}.axis.{index}"],
                each.unknowns,
                data[f"{name}.traces.{index}"],
                each.source,
                each.nature,
                each.parameter,
            )
            for index, each in enumerate(self.traces)
        )
        return results.BatchResult(
            self.analyses,
            self.parameters,
            values,
            ended,
            self.measures,
            tuple(_dimension(unit) for unit in self.units),
            measured,
            causes,
            self.reasons,
            {int(index): tuple(each) for index, each in self.problems.items()},
            {int(index): dict(each) for index, each in self.errors.items()},
            traces,
        )


def _arrays(name: str, found: results.BatchResult) -> dict[str, np.ndarray]:
    """The arrays of `found`, under keys that start with `name`."""
    arrays = {f"{name}.{key}": getattr(found, key) for key in _ARRAYS}
    for index, traces in enumerate(found.traces):
        arrays[f"{name}.traces.{index}"] = traces.values
        arrays[f"{name}.axis.{index}"] = traces.axis
    return arrays


_ARRAYS: Final = ("values", "ended", "measured", "causes")
"""The arrays of a batch's result that every batch has."""


class _Batch(_Columns):
    requirements: tuple[str, ...]
    bounds: tuple[tuple[float | None, float | None], ...]


class _Batches(_Model):
    format: Literal["jolty-batches"] = "jolty-batches"
    version: Literal[4] = 4
    batches: dict[str, _Batch]


_BATCHES_VERSION: Final = _Batches.model_construct().version
"""The version of the batch file's format that jolty writes and reads."""


def save_batches(path: Path, batches: Mapping[str, Batch]) -> None:
    """Writes `batches`, by name, to `path`."""
    arrays: dict[str, np.ndarray] = {}
    entries = {}
    for name, batch in batches.items():
        arrays |= _arrays(name, batch.result)
        entries[name] = _Batch(
            **_Columns.from_batch(batch.result).model_dump(),
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
    unreadable = f"{path} is not a jolty batch file of version {_BATCHES_VERSION}"
    with _open_archive(path, unreadable) as data:
        text = _header_text(data, unreadable)
        try:
            header = _Batches.model_validate_json(text)
        except ValidationError:
            raise JoltyError(unreadable) from None
        try:
            return {
                name: Batch(entry.result(name, data), entry.requirements, entry.bounds)
                for name, entry in header.batches.items()
            }
        except JoltyError:
            raise
        except _LOAD_ERRORS:
            raise JoltyError(unreadable) from None


@dataclass(frozen=True, slots=True)
class Shard:
    """The samples from `start` up to `stop` of a plan's Monte Carlo analysis, run.

    The analysis draws `samples` samples from its `seed`, each of the circuit's
    `toleranced` parameters and the kit's mismatch of its `mismatched` foundry's
    devices, and runs them at the plan's own conditions, then at its `corners`, by
    name. `conditions` holds the batch of the shard's samples at each, row 0 its
    sample `start`, or None where the circuit does not elaborate. `analyses` names
    every analysis of the plan, and `specs` are those its samples were measured
    against, if any, each requirement's measure the batch's measure at its
    place. Shards of one run of the analysis share their `digest`, which
    [`jolty.shards`][jolty.shards] makes.
    """

    start: int
    stop: int
    digest: str
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
    """A requirement as a shard's header states it: its measure and its bounds."""

    analysis: str
    description: str
    plain: results.Plain | None
    minimum: _Bound | None
    maximum: _Bound | None
    margin: _Bound | None

    @classmethod
    def from_requirement(cls, requirement: Requirement) -> _Requirement:
        """How the header states `requirement`."""
        measure = requirement.measure
        return cls(
            analysis=measure.analysis,
            description=measure.description,
            plain=measure.plain,
            minimum=_bound(requirement.minimum),
            maximum=_bound(requirement.maximum),
            margin=_bound(requirement.margin),
        )

    def requirement(self) -> Requirement:
        """The requirement, whose measure its shard read: it reads no result."""
        measure = Measure(
            self.analysis, self.description, self.plain, _read_by_its_shard
        )
        return Requirement(
            measure,
            _quantity(self.minimum),
            _quantity(self.maximum),
            _quantity(self.margin),
        )


def _read_by_its_shard(result: results.AnalysisResult) -> Reading:
    """Reads nothing of `result`: a shard's measure was read where the shard ran.

    Raises:
        MeasureError: Always.
    """
    del result
    message = "a shard's file holds what its measures read, and reads no result"
    raise MeasureError(message)


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
    version: Literal[3] = 3
    start: int
    stop: int
    digest: str
    samples: int
    seed: int
    toleranced: int
    mismatched: int
    corners: tuple[str, ...]
    analyses: tuple[str, ...]
    requirements: tuple[_Requirement, ...] | None
    yield_target: float | None
    conditions: tuple[_Columns | None, ...]


_SHARD_VERSION: Final = _Shard.model_construct().version
"""The version of the shard file's format that jolty writes and reads."""


def save_shard(path: Path, shard: Shard) -> None:
    """Writes `shard` to `path`."""
    arrays: dict[str, np.ndarray] = {}
    for place, found in enumerate(shard.conditions):
        if found is not None:
            arrays |= _arrays(str(place), found)
    header = _Shard(
        start=shard.start,
        stop=shard.stop,
        digest=shard.digest,
        samples=shard.samples,
        seed=shard.seed,
        toleranced=shard.toleranced,
        mismatched=shard.mismatched,
        corners=shard.corners,
        analyses=shard.analyses,
        requirements=None
        if shard.specs is None
        else tuple(
            _Requirement.from_requirement(each) for each in shard.specs.requirements
        ),
        yield_target=None if shard.specs is None else shard.specs.yield_target,
        conditions=tuple(
            None if found is None else _Columns.from_batch(found)
            for found in shard.conditions
        ),
    )
    arrays[_HEADER] = np.array(header.model_dump_json())
    _write(path, arrays)


def load_shard(path: Path) -> Shard:
    """Reads the shard file at `path`.

    Raises:
        JoltyError: When the file is not a jolty shard file of this version.
    """
    unreadable = f"{path} is not a jolty shard file of version {_SHARD_VERSION}"
    with _open_archive(path, unreadable) as data:
        text = _header_text(data, unreadable)
        try:
            header = _Shard.model_validate_json(text)
        except ValidationError:
            raise JoltyError(unreadable) from None
        if (
            not (0 <= header.start <= header.stop <= header.samples)
            or len(header.conditions) != len(header.corners) + 1
        ):
            raise JoltyError(unreadable)
        requirements = header.requirements
        try:
            return Shard(
                start=header.start,
                stop=header.stop,
                digest=header.digest,
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
        except JoltyError:
            raise
        except _LOAD_ERRORS:
            raise JoltyError(unreadable) from None


type RawAnalysisResult = (
    results.OperatingPointResult
    | results.DcSweepResult
    | results.AcResult
    | results.TransientResult
)
"""An analysis result that SPICE's raw format represents."""

_RAW_TYPES: Final[dict[results.Nature, str]] = {
    "voltage": "voltage",
    "current": "current",
    "temperature": "temperature",
}
"""The type a raw file gives an unknown of each nature it has one for; ``notype`` for
a device's state."""

_RAW_NATURES: Final[dict[str, results.Nature]] = {
    "voltage": "voltage",
    "current": "current",
    "temperature": "temperature",
}
"""The unknown nature for each SPICE raw vector type; ``notype`` and any other maps to
``state``."""


def save_raw(
    path: Path,
    analyses: Mapping[str, results.AnalysisResult],
    *,
    binary: bool = True,
) -> None:
    """Writes the results of `analyses` to `path` as SPICE's raw file, a plot each.

    The file is ngspice's raw format, which other tools read: each plot a header, its
    title the analysis's name, its scale and its vectors' names and types, then its
    points, in `binary`, little-endian doubles, or ASCII, each plot named as ngspice 47
    names its analysis's. An operating point is a plot of one point and no scale, a DC
    sweep's scale is ``v-sweep``, ``i-sweep`` or ``temp-sweep`` (with type ``notype``
    when the sweep has no nature, as for a parameter), a small-signal response's
    ``frequency`` and its values complex, and a transient's ``time``. A net's vector is
    named by its path, and a source's or an inductor's current, such as ``V1:current``,
    as ngspice names it, ``V1#branch``; every other unknown by its name. A thermal
    unknown carries type ``temperature`` and a state unknown ``notype``. It holds no
    date, so the same results give the same bytes. [`load_raw`][jolty.files.load_raw]
    reads the file back.

    Raises:
        ValueError: When `analyses` is empty, a name holds a line break, or an analysis
            is of a kind a raw file holds no plot of.
    """
    if not analyses:
        message = "a raw file holds one or more analyses"
        raise ValueError(message)
    written = bytearray()
    for name, result in analyses.items():
        if "\n" in name or "\r" in name:
            message = f"a raw file's analysis name holds no line break: {name!r}"
            raise ValueError(message)
        match result:
            case results.TransientResult(times=times, values=values):
                plot = "Transient Analysis"
                scale: tuple[str, str, np.ndarray] | None = ("time", "time", times)
                table: np.ndarray = values
            case results.AcResult(frequencies=frequencies, values=values):
                plot = "AC Analysis"
                scale = ("frequency", "frequency", frequencies)
                table = np.asarray(values, dtype=np.complex128)
            case results.DcSweepResult(swept=swept, values=values, nature=nature):
                plot = "DC transfer characteristic"
                scale = (*_raw_sweep_scale(nature), swept)
                table = values
            case results.OperatingPointResult(values=values):
                plot = "Operating Point"
                scale = None
                table = np.asarray(values, dtype=np.float64).reshape(
                    1, len(result.unknowns)
                )
            case _:
                message = (
                    "a raw file holds operating points, DC sweeps, small-signal "
                    f"responses and transients, not {name}'s {type(result).__name__}"
                )
                raise ValueError(message)
        written += _raw_plot(name, plot, result.unknowns, scale, table, binary=binary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(written))


def load_raw(path: Path) -> dict[str, RawAnalysisResult]:
    """Reads the SPICE raw file at `path` as jolty analysis results, by plot title.

    Supports both binary (little-endian IEEE-754 doubles) and ASCII raw files written by
    [`save_raw`][jolty.files.save_raw] or by ngspice, including multi-plot files. Each
    plot's ``Title:`` becomes its key in the returned mapping; an empty title falls back
    to the analysis kind (``operating_point``, ``dc_sweep``, ``ac`` or ``transient``),
    and duplicate titles receive ``#2``, ``#3`` and so on so no plot is dropped.

    SPICE's raw format stores only headers and vector arrays: an operating point loaded
    from raw carries ``strategy="raw"`` and ``iterations=None``, an AC analysis carries
    ``source=""`` and an all-NaN ``operating_point`` placeholder, a DC sweep carries the
    scale vector's name as ``source`` and ``parameter=None``, and a transient carries
    step count ``max(len(times) - 1, 0)`` with zero rejected steps. Binary raw files
    preserve exact IEEE-754 bits, including ``-0.0``.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
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
        JoltyError: When the file holds no plots, malformed headers, truncated values,
            invalid numbers, an unsupported ``Plotname:``, or mismatched real/complex
            flags.
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


def _raw_sweep_scale(nature: results.Nature | None) -> tuple[str, str]:
    """Scale vector name and SPICE type for a DC sweep of `nature`."""
    if nature == "current":
        return ("i-sweep", "current")
    if nature == "voltage":
        return ("v-sweep", "voltage")
    if nature == "temperature":
        return ("temp-sweep", "temperature")
    return ("v-sweep", "notype")


def _raw_sweep_nature(scale: ngspice.RawVariable) -> results.Nature | None:
    """DC sweep nature inferred from its raw scale vector."""
    if scale.kind == "current" or scale.name.casefold() == "i-sweep":
        return "current"
    if scale.kind == "voltage":
        return "voltage"
    if scale.kind == "temperature":
        return "temperature"
    return None


def _result_from_raw(
    path: Path, plot: ngspice.RawPlot
) -> tuple[str, RawAnalysisResult]:
    """Converts one parsed `RawPlot` into `(default_name, result)`."""
    plot_kind = plot.plot.casefold()
    if plot_kind == "operating point":
        return "operating_point", _raw_operating_point(path, plot)
    if plot_kind == "dc transfer characteristic":
        return "dc_sweep", _raw_dc_sweep(path, plot)
    if plot_kind == "ac analysis":
        return "ac", _raw_ac(path, plot)
    if plot_kind == "transient analysis":
        return "transient", _raw_transient(path, plot)
    message = f"{path}: unsupported SPICE raw plot `{plot.plot}`"
    raise JoltyError(message)


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
    unknowns = tuple(_unknown_from_raw(variable) for variable in plot.variables)
    values = np.array(plot.values[0].real, dtype=np.float64, copy=True)
    return results.OperatingPointResult(unknowns, values, strategy="raw")


def _raw_dc_sweep(path: Path, plot: ngspice.RawPlot) -> results.DcSweepResult:
    """Converts a ``DC transfer characteristic`` raw plot into `DcSweepResult`."""
    label = plot.title or plot.plot
    if plot.complex_values:
        message = f"{path}: DC sweep plot `{label}` holds complex values"
        raise JoltyError(message)
    if not plot.variables:
        message = f"{path}: DC sweep plot `{label}` has no sweep variable"
        raise JoltyError(message)
    scale = plot.variables[0]
    unknowns = tuple(_unknown_from_raw(variable) for variable in plot.variables[1:])
    swept = np.array(plot.values[:, 0].real, dtype=np.float64, copy=True)
    values = np.array(plot.values[:, 1:].real, dtype=np.float64, copy=True)
    return results.DcSweepResult(
        unknowns,
        source=scale.name,
        nature=_raw_sweep_nature(scale),
        swept=swept,
        values=values,
    )


def _raw_ac(path: Path, plot: ngspice.RawPlot) -> results.AcResult:
    """Converts an ``AC Analysis`` raw plot into `AcResult`."""
    label = plot.title or plot.plot
    if not plot.complex_values:
        message = f"{path}: AC plot `{label}` holds real values, not complex"
        raise JoltyError(message)
    if not plot.variables:
        message = f"{path}: AC plot `{label}` has no frequency variable"
        raise JoltyError(message)
    scale = plot.variables[0]
    if scale.kind != "frequency" and scale.name.casefold() != "frequency":
        message = (
            f"{path}: AC plot `{label}` begins with `{scale.name}` ({scale.kind}), "
            "not frequency"
        )
        raise JoltyError(message)
    unknowns = tuple(_unknown_from_raw(variable) for variable in plot.variables[1:])
    frequencies = np.array(plot.values[:, 0].real, dtype=np.float64, copy=True)
    values = np.array(plot.values[:, 1:], dtype=np.complex128, copy=True)
    operating_point = results.OperatingPointResult(
        unknowns,
        np.full(len(unknowns), np.nan, dtype=np.float64),
        strategy="raw",
    )
    return results.AcResult(
        unknowns,
        source="",
        frequencies=frequencies,
        values=values,
        operating_point=operating_point,
    )


def _raw_transient(path: Path, plot: ngspice.RawPlot) -> results.TransientResult:
    """Converts a ``Transient Analysis`` raw plot into `TransientResult`."""
    label = plot.title or plot.plot
    if plot.complex_values:
        message = f"{path}: transient plot `{label}` holds complex values"
        raise JoltyError(message)
    if not plot.variables:
        message = f"{path}: transient plot `{label}` has no time variable"
        raise JoltyError(message)
    scale = plot.variables[0]
    if scale.kind != "time" and scale.name.casefold() != "time":
        message = (
            f"{path}: transient plot `{label}` begins with `{scale.name}` "
            f"({scale.kind}), not time"
        )
        raise JoltyError(message)
    unknowns = tuple(_unknown_from_raw(variable) for variable in plot.variables[1:])
    times = np.array(plot.values[:, 0].real, dtype=np.float64, copy=True)
    values = np.array(plot.values[:, 1:].real, dtype=np.float64, copy=True)
    statistics = results.Statistics(
        initial=None,
        accepted=max(len(times) - 1, 0),
        rejected=0,
        landed=0,
        iterations=0,
    )
    return results.TransientResult(
        unknowns, times=times, values=values, statistics=statistics
    )


def _unknown_from_raw(variable: ngspice.RawVariable) -> results.Unknown:
    """The jolty unknown for a raw file's vector, reversing `_raw_name`."""
    name = variable.name
    nature = _RAW_NATURES.get(variable.kind, "state")
    if name.endswith("#branch") and len(name) > len("#branch"):
        return results.Unknown(f"{name.removesuffix('#branch')}:current", "current")
    lowered = name.casefold()
    if (
        nature in {"voltage", "state"}
        and lowered.startswith("v(")
        and name.endswith(")")
        and len(name) > 3  # noqa: PLR2004 - 'v(' plus a name plus ')'
    ):
        return results.Unknown(name[2:-1], "voltage")
    if (
        nature in {"current", "state"}
        and lowered.startswith("i(")
        and name.endswith(")")
        and len(name) > 3  # noqa: PLR2004 - 'i(' plus a name plus ')'
    ):
        return results.Unknown(f"{name[2:-1]}:current", "current")
    return results.Unknown(name, nature)


def _raw_plot(  # noqa: PLR0913 - a plot's names, its scale, its values and its format
    title: str,
    plot: str,
    unknowns: tuple[results.Unknown, ...],
    scale: tuple[str, str, np.ndarray] | None,
    table: np.ndarray,
    *,
    binary: bool,
) -> bytes:
    """One plot of a raw file: its header, then its values, ASCII or `binary`."""
    named = [
        (_raw_name(unknown), _RAW_TYPES.get(unknown.nature, "notype"))
        for unknown in unknowns
    ]
    columns = [np.asarray(table[:, index]) for index in range(len(unknowns))]
    if scale is not None:
        named.insert(0, scale[:2])
        columns.insert(0, np.asarray(scale[2]))
    complex_values = np.iscomplexobj(table)
    points = len(columns[0]) if columns else table.shape[0]
    lines = [
        f"Title: {title}",
        f"Plotname: {plot}",
        f"Flags: {'complex' if complex_values else 'real'}",
        f"No. Variables: {len(named)}",
        f"No. Points: {points}",
        "Variables:",
        *(f"\t{index}\t{name}\t{kind}" for index, (name, kind) in enumerate(named)),
        "Binary:" if binary else "Values:",
    ]
    header = ("\n".join(lines) + "\n").encode()
    matrix = np.column_stack(columns) if columns else np.zeros((points, 0))
    if complex_values:
        matrix = matrix.astype(np.complex128)
        if binary:
            parts = np.stack([matrix.real, matrix.imag], axis=-1).reshape(
                points, 2 * len(named)
            )
            return header + parts.astype("<f8").tobytes()
        rows = (
            "\t".join(f"{value.real:.17e},{value.imag:.17e}" for value in row)
            for row in matrix
        )
    else:
        matrix = matrix.astype(np.float64)
        if binary:
            return header + matrix.astype("<f8").tobytes()
        rows = ("\t".join(f"{value:.17e}" for value in row) for row in matrix)
    values = "".join(
        f" {index}\t" + "\n\t".join(row.split("\t")) + "\n"
        for index, row in enumerate(rows)
    )
    return header + values.encode()


def _raw_name(unknown: results.Unknown) -> str:
    """The name of `unknown`'s vector in a raw file, as ngspice names it."""
    path, separator, label = unknown.name.rpartition(":")
    if separator and label == "current" and unknown.nature == "current":
        return f"{path}#branch"
    return unknown.name
