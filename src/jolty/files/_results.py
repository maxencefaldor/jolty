# Copyright 2026 Maxence Faldor
"""Results archive serialization, provenance, and shared `.npz` archive primitives."""

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
from dataclasses import fields, is_dataclass, replace
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path, PurePath
from typing import Annotated, Final, Literal, Self, override

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jolty import _core, ir, results
from jolty.errors import JoltyError
from jolty.plan import Plan

HEADER: Final = "header"
"""Archive entry holding the JSON header of a results, batch, or shard file."""

_EPOCH: Final = (1980, 1, 1, 0, 0, 0)
"""The date a ZIP archive's entries carry: the earliest it can, the same every time."""

_PACKAGE: Final = Path(__file__).resolve().parent.parent
"""jolty's own directory, whose modules and engine a build is."""

_RESULTS_VERSION: Final = 20
"""The version of the results file's format that jolty writes and reads."""

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

LOAD_ERRORS: Final = (
    *_ARCHIVE_ERRORS,
    KeyError,
    IndexError,
    TypeError,
    ZeroDivisionError,
    OverflowError,
)
"""Exceptions raised when decoding arrays or header metadata from a corrupt archive."""

_RECORDED: Final = ("currents", "charges", "conductances", "capacitances", "variables")
"""The arrays of what the recorded devices did, by the names of their fields."""

_TRANSFER_LENGTH: Final = 3
"""Scalars stored in a transfer function array: gain, input and output resistance."""


class _Archive(Mapping[str, np.ndarray]):
    """Read-only view of an `.npz` archive requiring every entry to be an `ndarray`."""

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


def write_archive(
    path: Path, arrays: Mapping[str, np.ndarray], header: BaseModel | None = None
) -> None:
    """Writes `arrays` (and JSON `header`) to `path` deterministically."""
    entries = (
        arrays
        if header is None
        else {**arrays, HEADER: np.array(header.model_dump_json())}
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, array in entries.items():
            entry = zipfile.ZipInfo(f"{name}.npy", date_time=_EPOCH)
            with archive.open(entry, "w", force_zip64=True) as file:
                np.lib.format.write_array(
                    file, np.asanyarray(array), allow_pickle=False
                )


@contextlib.contextmanager
def open_archive(path: Path, unreadable: str) -> Generator[Mapping[str, np.ndarray]]:
    """Opens the NumPy `.npz` archive at `path`, raising `JoltyError` if corrupt."""
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


def header_text(data: Mapping[str, np.ndarray], unreadable: str) -> str:
    """Extracts the JSON header string from `data`, raising `JoltyError` if corrupt."""
    try:
        raw = data[HEADER]
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


@contextlib.contextmanager
def open_validated[M: BaseModel](
    path: Path, model: type[M], unreadable: str
) -> Generator[tuple[M, Mapping[str, np.ndarray]]]:
    """Opens `path`, validates its header against `model`, and wraps load errors."""
    with open_archive(path, unreadable) as data:
        text = header_text(data, unreadable)
        try:
            header = model.model_validate_json(text)
        except ValidationError:
            raise JoltyError(unreadable) from None
        try:
            yield header, data
        except JoltyError:
            raise
        except LOAD_ERRORS:
            raise JoltyError(unreadable) from None


def circuit_digest(circuit: ir.Circuit) -> str:
    """A SHA-256 digest, in hexadecimal, of what a simulation reads of `circuit`.

    It digests the circuit in the circuit format without each instance's
    [`SourceLocation`][jolty.ir.SourceLocation], the script line that placed it, which
    names where a problem lies and which no simulation reads: the same circuit built by
    scripts in two places, or on two machines, digests alike.

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

    It digests the bytes of the engine's library and of every module of the package, by
    their paths within it, and the interpreter's version, since a measure a script
    writes computes in Python.
    """
    digester = hashlib.sha256(Path(_core.__file__).read_bytes())
    for module in sorted(_PACKAGE.rglob("*.py")):
        digester.update(module.relative_to(_PACKAGE).as_posix().encode())
        digester.update(module.read_bytes())
    digester.update(sys.version.encode())
    return digester.hexdigest()


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
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_float(value: float) -> object:
    if math.isnan(value):
        return {"__float__": "nan"}
    if math.isinf(value):
        return {"__float__": "inf" if value > 0 else "-inf"}
    return 0.0 if value == 0 else value


def _canonical_mapping[K, V](value: Mapping[K, V]) -> object:
    if all(isinstance(key, str) for key in value):
        return {
            str(key): _canonical(entry)
            for key, entry in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    entries = [[_canonical(key), _canonical(entry)] for key, entry in value.items()]
    entries.sort(key=lambda pair: _canonical_json(pair[0]))
    return {"__type__": "mapping", "items": entries}


def _canonical_special(value: object) -> tuple[bool, object]:
    match value:
        case np.ndarray():
            converted: object = {
                "__type__": "numpy.ndarray",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": _canonical(value.tolist()),
            }
        case bytes() | bytearray() | memoryview():
            converted = {"__type__": "bytes", "hex": bytes(value).hex()}
        case enum.Enum():
            converted = {
                "__type__": _type_name(value),
                "value": _canonical(value.value),
            }
        case PurePath():
            converted = {"__type__": "pathlib.PurePath", "value": value.as_posix()}
        case Fraction():
            converted = {
                "__type__": "fractions.Fraction",
                "numerator": value.numerator,
                "denominator": value.denominator,
            }
        case _:
            return False, None
    return True, converted


def _canonical_collection(value: object) -> object:
    match value:
        case Mapping():
            return _canonical_mapping(value)
        case list():
            return [_canonical(item) for item in value]
        case tuple():
            return {"__type__": "tuple", "items": [_canonical(item) for item in value]}
        case set() | frozenset():
            items = sorted((_canonical(item) for item in value), key=_canonical_json)
            return {"__type__": "set", "items": items}
        case _:
            return {"__type__": _type_name(value), "repr": repr(value)}


def _canonical_scalar(value: object) -> tuple[bool, object]:
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


def _canonical(value: object) -> object:
    """Converts `value` into a deterministic, JSON-serializable structure."""
    for helper in (_canonical_scalar, _canonical_special):
        matched, converted = helper(value)
        if matched:
            return converted
    if isinstance(value, BaseModel):
        names = sorted(type(value).model_fields)
    elif is_dataclass(value) and not isinstance(value, type):
        names = sorted(item.name for item in fields(value))
    else:
        return _canonical_collection(value)
    return {
        "__type__": _type_name(value),
        "fields": {name: _canonical(getattr(value, name)) for name in names},
    }


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
    return hashlib.sha256(_canonical_json(_canonical(plan)).encode()).hexdigest()


class Provenance(ir.FrozenModel):
    """Where results come from, as a results file states it.

    The jolty, the build and the platform that simulated them, and the circuit and the
    plan they are of, by their digests. Every results file states it, and
    [`load_results`][jolty.files.load_results] given a circuit refuses a file whose
    results are of another.
    """

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


class _Found(ir.FrozenModel):
    strategy: str
    iterations: int | None = None

    @classmethod
    def from_point(cls, point: results.OperatingPointResult) -> Self:
        return cls(strategy=point.strategy, iterations=point.iterations)


class _LinearizedEntry(ir.FrozenModel):
    found: _Found
    recording: int


class _OperatingPointResult(_LinearizedEntry):
    kind: Literal["operating_point"] = "operating_point"


class _OperatingPointsResult(ir.FrozenModel):
    kind: Literal["operating_points"] = "operating_points"
    found: tuple[_Found, ...]
    recording: int


class _DcSweepResult(ir.FrozenModel):
    kind: Literal["dc_sweep"] = "dc_sweep"
    source: str
    nature: results.Nature | None
    parameter: str | None = None
    strategies: tuple[str, ...] = ()
    iterations: tuple[int, ...] = ()
    recording: int


class _AcResult(_LinearizedEntry):
    kind: Literal["ac"] = "ac"
    source: str


class _Statistics(ir.FrozenModel):
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

    def to_statistics(self, leak: results.ChargeLeak | None) -> results.Statistics:
        return results.Statistics(
            self.initial,
            self.accepted,
            self.rejected,
            self.landed,
            self.iterations,
            self.work,
            self.profile,
            leak,
        )


class _WaveformEntry(ir.FrozenModel):
    statistics: _Statistics
    recording: int


class _TransientResult(_WaveformEntry):
    kind: Literal["transient"] = "transient"


class _PeriodicSteadyStateResult(_WaveformEntry):
    kind: Literal["periodic_steady_state"] = "periodic_steady_state"
    period: float
    periods: int


class _DifferentiatedEntry(ir.FrozenModel):
    outputs: tuple[str, ...]
    period: bool = Field(
        default=False,
        description="Whether the arrays hold an oscillator's period's derivatives.",
    )
    result: _Entry


class _SensitivityResult(_DifferentiatedEntry):
    kind: Literal["sensitivity"] = "sensitivity"
    parameters: tuple[tuple[str, str], ...]


class _MismatchResult(_DifferentiatedEntry):
    kind: Literal["mismatch"] = "mismatch"
    random_variables: tuple[ir.RandomVariable, ...]


class _StabilityResult(_LinearizedEntry):
    kind: Literal["stability"] = "stability"
    probe: str


class _NoiseInput(ir.FrozenModel):
    source: str
    nature: results.Nature


class _NoiseResult(_LinearizedEntry):
    kind: Literal["noise"] = "noise"
    output: str
    reference: str | None
    nature: results.Nature
    sources: tuple[tuple[str, str], ...]
    input: _NoiseInput | None


class _PoleZeroResult(_LinearizedEntry):
    kind: Literal["pole_zero"] = "pole_zero"
    source: str | None
    output: str | None
    reference: str | None


class _TransferFunctionResult(_LinearizedEntry):
    kind: Literal["transfer_function"] = "transfer_function"
    source: str
    input: results.Nature
    output: str
    reference: str | None


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


class _Recording(ir.FrozenModel):
    devices: tuple[results.RecordedDevice, ...] = ()
    summed_terminals: tuple[str, ...] = ()
    thermal: tuple[str, ...] = ()

    @classmethod
    def from_recording(cls, recording: results.Recording) -> _Recording:
        return cls(
            devices=recording.devices,
            summed_terminals=recording.summed_terminals,
            thermal=tuple(sorted(recording.thermal)),
        )

    def recording(self) -> results.Recording:
        return results.Recording(
            self.devices, self.summed_terminals, frozenset(self.thermal)
        )


class _Header(ir.FrozenModel):
    unknowns: tuple[results.Unknown, ...]
    nets: tuple[str, ...] = ()
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


class _Format(BaseModel):
    """A results file's format and version, whatever else its header holds."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    format: Literal["jolty-results"]
    version: int


def save_results(
    path: Path,
    analyses: Mapping[str, results.AnalysisResult],
    *,
    circuit: ir.Circuit | None = None,
    plan: Plan | None = None,
) -> None:
    """Writes the results of `analyses`, by name, to `path`, of `circuit` and `plan`.

    The file states their [`Provenance`][jolty.files.Provenance]: this jolty, its
    build and the platform, and the digests of `circuit` and `plan` when they are given.

    Raises:
        ValueError: When the analyses are of different circuits, or none is given.
    """
    circuits = {_names(result) for result in analyses.values()}
    if len(circuits) != 1:
        message = "a results file holds analyses of one circuit, one or more"
        raise ValueError(message)
    saver = _Saver()
    entries = {name: saver.save(name, result) for name, result in analyses.items()}
    unknowns, nets = circuits.pop()
    header = _Header(
        unknowns=unknowns,
        nets=nets,
        provenance=Provenance.from_circuit_and_plan(circuit, plan),
        recordings=tuple(_Recording.from_recording(each) for each in saver.recordings),
        analyses=entries,
        excursions={
            name: result.excursions
            for name, result in analyses.items()
            if result.excursions is not None
        },
    )
    write_archive(path, saver.arrays, header)


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
    with open_archive(path, unreadable) as data:
        header = _header(path, data)
        of = header.provenance.circuit
        if circuit is not None and of is not None and of != circuit_digest(circuit):
            message = (
                f"{path} holds the results of another circuit than the one given: "
                f"its circuit's digest is {of[:12]}, the one given's "
                f"{circuit_digest(circuit)[:12]}; simulate the circuit again"
            )
            raise JoltyError(message)
        loader = _Loader(data, header)
        try:
            found = {
                name: loader.load(name, entry)
                for name, entry in header.analyses.items()
            }
        except JoltyError:
            raise
        except LOAD_ERRORS:
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
    with open_archive(path, f"{path} is not a jolty results file") as data:
        return _header(path, data).provenance


def _header(path: Path, data: Mapping[str, np.ndarray]) -> _Header:
    unreadable = f"{path} is not a jolty results file"
    text = header_text(data, unreadable)
    try:
        return _Header.model_validate_json(text)
    except ValidationError:
        raise _unreadable(path, text) from None


def _unreadable(path: Path, text: str) -> JoltyError:
    try:
        found_version = _Format.model_validate_json(text).version
    except ValidationError:
        found_version = _RESULTS_VERSION
    if found_version == _RESULTS_VERSION:
        return JoltyError(f"{path} is not a jolty results file")
    return JoltyError(
        f"{path} holds results in version {found_version} of the format, which jolty "
        f"reads no more; simulate again to write them in version {_RESULTS_VERSION}"
    )


class _Saver:
    """Collects arrays and deduplicates recordings while saving one results file."""

    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.recordings: dict[results.Recording, int] = {}

    def index(self, recording: results.Recording) -> int:
        return self.recordings.setdefault(recording, len(self.recordings))

    def put_recorded(
        self,
        name: str,
        result: results.OperatingPointResult
        | results.DcSweepResult
        | results.TransientResult,
    ) -> None:
        for field in _RECORDED:
            self.arrays[f"{name}.{field}"] = getattr(result, field)

    def put_point(self, name: str, point: results.OperatingPointResult) -> _Found:
        self.arrays[f"{name}.values"] = point.values
        self.arrays[f"{name}.logic"] = point.logic
        self.put_recorded(name, point)
        return _Found.from_point(point)

    def save(self, name: str, result: results.AnalysisResult) -> _Entry:
        match result:
            case results.OperatingPointResult():
                return _OperatingPointResult(
                    found=self.put_point(name, result),
                    recording=self.index(result.recording),
                )
            case results.OperatingPointsResult():
                found = tuple(
                    self.put_point(f"{name}.{index}", point)
                    for index, point in enumerate(result.points)
                )
                return _OperatingPointsResult(
                    found=found, recording=self.index(result.points[0].recording)
                )
            case results.DcSweepResult():
                self.arrays[f"{name}.swept"] = result.swept
                self.arrays[f"{name}.values"] = result.values
                self.arrays[f"{name}.logic"] = result.logic
                self.put_recorded(name, result)
                return _DcSweepResult(
                    source=result.source,
                    nature=result.nature,
                    parameter=result.parameter,
                    strategies=result.strategies,
                    iterations=result.iterations,
                    recording=self.index(result.recording),
                )
            case results.TransientResult():
                return self.save_waveforms(name, result)
            case results.SensitivityResult() | results.MismatchResult():
                return self.save_differentiated(name, result)
            case _:
                return self.save_linear(name, result)

    def save_waveforms(
        self, name: str, result: results.TransientResult
    ) -> _TransientResult | _PeriodicSteadyStateResult:
        self.arrays[f"{name}.times"] = result.times
        self.arrays[f"{name}.values"] = result.values
        self.put_recorded(name, result)
        traces = result.logic
        self.arrays[f"{name}.logic.counts"] = np.array(
            [len(trace.times) for trace in traces], dtype=np.int64
        )
        self.arrays[f"{name}.logic.times"] = np.concatenate(
            [np.empty(0), *(trace.times for trace in traces)]
        )
        self.arrays[f"{name}.logic.levels"] = np.concatenate(
            [np.empty(0), *(trace.levels for trace in traces)]
        )
        if result.statistics.charge_leak is not None:
            self.arrays[f"{name}.charge_leak.signed"] = (
                result.statistics.charge_leak.signed
            )
            self.arrays[f"{name}.charge_leak.absolute"] = (
                result.statistics.charge_leak.absolute
            )
        statistics = _Statistics.from_statistics(result.statistics)
        recording = self.index(result.recording)
        if isinstance(result, results.PeriodicSteadyStateResult):
            return _PeriodicSteadyStateResult(
                period=result.period,
                periods=result.periods,
                statistics=statistics,
                recording=recording,
            )
        return _TransientResult(statistics=statistics, recording=recording)

    def save_differentiated(
        self, name: str, result: results.SensitivityResult | results.MismatchResult
    ) -> _SensitivityResult | _MismatchResult:
        self.arrays[f"{name}.derivatives"] = result.derivatives
        if result.period_derivatives is not None:
            self.arrays[f"{name}.period_derivatives"] = result.period_derivatives
        nested = self.save(f"{name}.result", result.result)
        has_period = result.period_derivatives is not None
        if isinstance(result, results.SensitivityResult):
            self.arrays[f"{name}.parameters"] = np.array(
                [parameter.value for parameter in result.parameters], dtype=float
            )
            return _SensitivityResult(
                outputs=result.outputs,
                parameters=tuple(
                    (parameter.instance, parameter.name)
                    for parameter in result.parameters
                ),
                period=has_period,
                result=nested,
            )
        return _MismatchResult(
            outputs=result.outputs,
            random_variables=result.random_variables,
            period=has_period,
            result=nested,
        )

    def save_linear(
        self,
        name: str,
        result: results.AcResult
        | results.StabilityResult
        | results.NoiseResult
        | results.PoleZeroResult
        | results.TransferFunctionResult,
    ) -> _Entry:
        point = result.operating_point
        found = self.put_point(f"{name}.operating_point", point)
        chosen = (
            result.recording
            if isinstance(result, results.AcResult)
            else point.recording
        )
        recording = self.index(chosen)
        match result:
            case results.AcResult():
                self.arrays[f"{name}.frequencies"] = result.frequencies
                self.arrays[f"{name}.values"] = result.values
                self.arrays[f"{name}.currents"] = result.currents
                return _AcResult(source=result.source, found=found, recording=recording)
            case results.TransferFunctionResult():
                self.arrays[f"{name}.transfer"] = np.array(
                    [result.gain, result.input_resistance, result.output_resistance]
                )
                return _TransferFunctionResult(
                    source=result.source,
                    input=result.input,
                    output=result.output,
                    reference=result.reference,
                    found=found,
                    recording=recording,
                )
            case results.StabilityResult():
                self.arrays[f"{name}.frequencies"] = result.frequencies
                self.arrays[f"{name}.gains"] = result.gains
                return _StabilityResult(
                    probe=result.probe, found=found, recording=recording
                )
            case results.NoiseResult():
                self.arrays[f"{name}.frequencies"] = result.frequencies
                self.arrays[f"{name}.densities"] = result.densities
                self.arrays[f"{name}.contributions"] = result.contributions
                noise_input = None
                if result.input is not None:
                    self.arrays[f"{name}.gains"] = result.input.gains
                    noise_input = _NoiseInput(
                        source=result.input.source, nature=result.input.nature
                    )
                return _NoiseResult(
                    output=result.output,
                    reference=result.reference,
                    nature=result.nature,
                    sources=tuple(
                        (source.instance, source.name) for source in result.sources
                    ),
                    input=noise_input,
                    found=found,
                    recording=recording,
                )
            case results.PoleZeroResult():
                self.arrays[f"{name}.poles"] = result.poles
                if result.zeros is not None:
                    self.arrays[f"{name}.zeros"] = result.zeros
                return _PoleZeroResult(
                    source=result.source,
                    output=result.output,
                    reference=result.reference,
                    found=found,
                    recording=recording,
                )


class _Loader:
    """Decodes analysis results from one `.npz` archive and its validated header."""

    def __init__(self, data: Mapping[str, np.ndarray], header: _Header) -> None:
        self.data = data
        self.header = header
        self.recordings = tuple(each.recording() for each in header.recordings)

    def recording(self, index: int) -> results.Recording:
        if not 0 <= index < len(self.recordings):
            count = len(self.recordings)
            message = f"recording index {index} is out of range for {count}"
            raise IndexError(message)
        return self.recordings[index]

    def recorded(
        self, name: str, recording: results.Recording
    ) -> results.RecordedValues:
        data = self.data
        return {
            "recording": recording,
            "currents": data[f"{name}.currents"],
            "charges": data[f"{name}.charges"],
            "conductances": data[f"{name}.conductances"],
            "capacitances": data[f"{name}.capacitances"],
            "variables": data[f"{name}.variables"],
        }

    def point(
        self, name: str, found: _Found, recording: results.Recording
    ) -> results.OperatingPointResult:
        return results.OperatingPointResult(
            self.header.unknowns,
            self.data[f"{name}.values"],
            found.strategy,
            found.iterations,
            self.header.nets,
            self.data[f"{name}.logic"],
            **self.recorded(name, recording),
        )

    def load(self, name: str, entry: _Entry) -> results.AnalysisResult:
        unknowns, nets = self.header.unknowns, self.header.nets
        match entry:
            case _OperatingPointResult():
                return self.point(name, entry.found, self.recording(entry.recording))
            case _OperatingPointsResult():
                resolved = self.recording(entry.recording)
                points = tuple(
                    self.point(f"{name}.{index}", each, resolved)
                    for index, each in enumerate(entry.found)
                )
                return results.OperatingPointsResult(unknowns, points)
            case _DcSweepResult():
                return results.DcSweepResult(
                    unknowns,
                    entry.source,
                    entry.nature,
                    self.data[f"{name}.swept"],
                    self.data[f"{name}.values"],
                    nets,
                    self.data[f"{name}.logic"],
                    **self.recorded(name, self.recording(entry.recording)),
                    parameter=entry.parameter,
                    strategies=entry.strategies,
                    iterations=entry.iterations,
                )
            case _TransientResult() | _PeriodicSteadyStateResult():
                return self.load_waveforms(name, entry)
            case _SensitivityResult() | _MismatchResult():
                return self.load_differentiated(name, entry)
            case _:
                return self.load_linear(name, entry)

    def load_waveforms(
        self, name: str, entry: _TransientResult | _PeriodicSteadyStateResult
    ) -> results.TransientResult:
        leak: results.ChargeLeak | None = None
        if entry.statistics.charge_leak:
            signed = self.data[f"{name}.charge_leak.signed"]
            absolute = self.data[f"{name}.charge_leak.absolute"]
            expected = (len(self.header.unknowns),)
            if (
                signed.shape != expected
                or absolute.shape != expected
                or not np.issubdtype(signed.dtype, np.floating)
                or not np.issubdtype(absolute.dtype, np.floating)
            ):
                message = f"charge leak arrays do not match {expected}"
                raise ValueError(message)
            leak = results.ChargeLeak(signed, absolute)
        waveform_fields = (
            self.header.unknowns,
            self.data[f"{name}.times"],
            self.data[f"{name}.values"],
            entry.statistics.to_statistics(leak),
            self.header.nets,
            _traces(self.data, name),
        )
        recorded = self.recorded(name, self.recording(entry.recording))
        if isinstance(entry, _PeriodicSteadyStateResult):
            return results.PeriodicSteadyStateResult(
                *waveform_fields, **recorded, period=entry.period, periods=entry.periods
            )
        return results.TransientResult(*waveform_fields, **recorded)

    def load_differentiated(
        self, name: str, entry: _SensitivityResult | _MismatchResult
    ) -> results.SensitivityResult | results.MismatchResult:
        found = _differentiated(self.load(f"{name}.result", entry.result), name)
        period_derivatives = (
            self.data[f"{name}.period_derivatives"] if entry.period else None
        )
        if isinstance(entry, _SensitivityResult):
            raw_parameters = self.data[f"{name}.parameters"]
            if raw_parameters.ndim != 1 or len(raw_parameters) != len(entry.parameters):
                message = (
                    f"{name}.parameters has shape {raw_parameters.shape}, not "
                    f"({len(entry.parameters)},)"
                )
                raise ValueError(message)
            parameters = tuple(
                results.Parameter(instance, parameter, float(value))
                for (instance, parameter), value in zip(
                    entry.parameters, raw_parameters, strict=True
                )
            )
            return results.SensitivityResult(
                self.header.unknowns,
                entry.outputs,
                parameters,
                self.data[f"{name}.derivatives"],
                found,
                period_derivatives,
            )
        return results.MismatchResult(
            self.header.unknowns,
            entry.outputs,
            entry.random_variables,
            self.data[f"{name}.derivatives"],
            found,
            period_derivatives,
        )

    def load_linear(
        self,
        name: str,
        entry: _AcResult
        | _StabilityResult
        | _NoiseResult
        | _PoleZeroResult
        | _TransferFunctionResult,
    ) -> results.AnalysisResult:
        unknowns = self.header.unknowns
        resolved = self.recording(entry.recording)
        point = self.point(f"{name}.operating_point", entry.found, resolved)
        data = self.data
        match entry:
            case _AcResult():
                return results.AcResult(
                    unknowns,
                    entry.source,
                    data[f"{name}.frequencies"],
                    data[f"{name}.values"],
                    point,
                    recording=resolved,
                    currents=data[f"{name}.currents"],
                )
            case _TransferFunctionResult():
                transfer = data[f"{name}.transfer"]
                if transfer.ndim != 1 or len(transfer) != _TRANSFER_LENGTH:
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
                    point,
                )
            case _StabilityResult():
                return results.StabilityResult(
                    unknowns,
                    entry.probe,
                    data[f"{name}.frequencies"],
                    data[f"{name}.gains"],
                    point,
                )
            case _NoiseResult():
                found_input = (
                    None
                    if entry.input is None
                    else results.NoiseInput(
                        entry.input.source, entry.input.nature, data[f"{name}.gains"]
                    )
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
                    point,
                )
            case _PoleZeroResult():
                return results.PoleZeroResult(
                    unknowns,
                    data[f"{name}.poles"],
                    data.get(f"{name}.zeros"),
                    entry.source,
                    entry.output,
                    entry.reference,
                    point,
                )


def _differentiated(
    found: results.AnalysisResult, name: str
) -> (
    results.OperatingPointResult
    | results.DcSweepResult
    | results.AcResult
    | results.TransientResult
):
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


def _names(
    result: results.AnalysisResult,
) -> tuple[tuple[results.Unknown, ...], tuple[str, ...]]:
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
    return result.unknowns, result.nets


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
