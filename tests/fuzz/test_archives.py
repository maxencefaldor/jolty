# Copyright 2026 Maxence Faldor
"""Hypothesis fuzz tests for results, batches, and shard `.npz` archive loaders."""

import contextlib
import io
import json
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from fuzz.helpers import FUZZ, ArchiveEntries, JsonObject
from jolty import JoltyError, files, results, shards

type ArchiveMutator = Callable[[ArchiveEntries, JsonObject, st.DataObject], bool]


@dataclass(frozen=True, slots=True)
class _BaselineArchives:
    """Serialized `.npz` entry maps for valid results, batches, and shard archives."""

    results: dict[str, np.ndarray]
    batches: dict[str, np.ndarray]
    shard: dict[str, np.ndarray]

    def copy_for(self, kind: str) -> ArchiveEntries:
        """Returns an independent copy of the entry map for `kind`."""
        mapping = {
            "results": self.results,
            "batches": self.batches,
            "shard": self.shard,
        }[kind]
        return {key: value.copy() for key, value in mapping.items()}


@dataclass(frozen=True, slots=True)
class _ZipCorruption:
    """Parameters describing a single truncation or byte-flip corruption."""

    kind: str
    fraction: float
    compression: int
    xor_mask: int


def _write_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray | bytes],
    *,
    compression: int = zipfile.ZIP_STORED,
) -> None:
    """Writes raw `.npy` entries or arrays into a `.npz` archive at `path`."""
    with zipfile.ZipFile(path, "w", compression) as archive:
        for name, value in arrays.items():
            if isinstance(value, bytes):
                archive.writestr(f"{name}.npy", value)
            else:
                buffer = io.BytesIO()
                np.save(buffer, value, allow_pickle=False)
                archive.writestr(f"{name}.npy", buffer.getvalue())


def _sample_results() -> dict[str, results.AnalysisResult]:
    """Representative results across multiple analysis kinds."""
    unknowns = (results.Unknown("out", "voltage"),)
    point = results.OperatingPointResult(unknowns, np.array([2.5]), "direct", 3)
    statistics = results.Statistics(
        initial="direct",
        accepted=1,
        rejected=0,
        landed=1,
        iterations=2,
        work=results.Work(
            iterations=2, factorizations=1, refactorizations=1, fallbacks=0, reuses=0
        ),
        profile=None,
    )
    transient = results.TransientResult(
        unknowns,
        np.array([0.0, 1e-6]),
        np.array([[2.5], [2.5]]),
        statistics,
    )
    transfer = results.TransferFunctionResult(
        unknowns, "V1", "voltage", "out", None, 0.5, 2e3, 500.0, point
    )
    sensitivity = results.SensitivityResult(
        unknowns,
        ("out",),
        (results.Parameter("R1", "resistance", 1e3),),
        np.array([[-1e-3]]),
        point,
    )
    return {
        "operating_point": point,
        "step": transient,
        "transfer": transfer,
        "sensitivity": sensitivity,
    }


def _sample_batch() -> results.BatchResult:
    """Minimal valid two-sample `BatchResult`."""
    return results.BatchResult(
        analyses=("operating_point",),
        parameters=(),
        values=np.empty((2, 0)),
        ended=np.ones((2, 1), dtype=np.bool_),
        measures=("out in operating_point",),
        units=(None,),
        measured=np.array([[2.5], [2.4]]),
        causes=np.full((2, 1), -1, dtype=np.int32),
        reasons=(),
    )


def _sample_shard(
    start: int = 0, stop: int = 2, samples: int = 2, digest: str | None = None
) -> files.Shard:
    """Minimal valid `Shard` for mutation testing."""
    return files.Shard(
        start=start,
        stop=stop,
        digest="a" * 64 if digest is None else digest,
        samples=samples,
        seed=1,
        toleranced=1,
        mismatched=0,
        corners=(),
        analyses=("operating_point",),
        specs=None,
        conditions=(_sample_batch(),),
    )


def _read_npz_dict(path: Path) -> dict[str, np.ndarray]:
    """Reads every entry of a valid `.npz` archive into a plain dict."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _pickled_npy_bytes() -> bytes:
    """Returns a `.npy` payload containing a pickled Python object array."""
    buffer = io.BytesIO()
    np.save(buffer, np.array([{"header": 1}], dtype=object), allow_pickle=True)
    return buffer.getvalue()


@st.composite
def _zip_corruptions(draw: st.DrawFn) -> _ZipCorruption:
    """Generates archive truncation or single-byte corruption parameters."""
    kind = draw(st.sampled_from(["results", "batches", "shard"]))
    fraction = draw(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    compression = draw(st.sampled_from([zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED]))
    xor_mask = draw(st.integers(min_value=0, max_value=255))
    return _ZipCorruption(kind, fraction, compression, xor_mask)


@pytest.fixture(scope="module")
def baseline_archives(scratch_directory: Path) -> _BaselineArchives:
    """Baseline valid results, batches, and shard archive dictionaries."""
    results_path = scratch_directory / "base.results.npz"
    files.save_results(results_path, _sample_results())
    batches_path = scratch_directory / "base.batches.npz"
    files.save_batches(
        batches_path,
        {"monte_carlo": files.Batch(_sample_batch(), ("out >= 0 V",), ((0.0, None),))},
    )
    shard_path = scratch_directory / "base.shard.npz"
    files.save_shard(shard_path, _sample_shard())
    return _BaselineArchives(
        results=_read_npz_dict(results_path),
        batches=_read_npz_dict(batches_path),
        shard=_read_npz_dict(shard_path),
    )


def _drop_array(
    arrays: ArchiveEntries, _header: JsonObject, data: st.DataObject
) -> bool:
    candidates = [key for key in arrays if key != "header"]
    if candidates:
        arrays.pop(data.draw(st.sampled_from(candidates)))
    return False


def _bad_version(extra: Sequence[object] = ()) -> ArchiveMutator:
    def mutate(
        _arrays: ArchiveEntries, header: JsonObject, data: st.DataObject
    ) -> bool:
        current = header["version"]
        choices: list[object] = [0, *extra]
        if isinstance(current, int):
            choices.extend((current - 1, current + 1, str(current)))
        else:
            choices.append("bad-version")
        header["version"] = data.draw(st.sampled_from(choices))
        return False

    return mutate


def _bad_format(value: str = "other") -> ArchiveMutator:
    def mutate(
        _arrays: ArchiveEntries, header: JsonObject, _data: st.DataObject
    ) -> bool:
        header["format"] = value
        return False

    return mutate


def _set_array(
    key: str,
    value: np.ndarray | bytes | Callable[[], bytes],
    *,
    rewrites_header: bool = False,
) -> ArchiveMutator:
    def mutate(
        arrays: ArchiveEntries, _header: JsonObject, _data: st.DataObject
    ) -> bool:
        if isinstance(value, np.ndarray):
            arrays[key] = value.copy()
        elif isinstance(value, bytes):
            arrays[key] = value
        else:
            arrays[key] = value()
        return rewrites_header

    return mutate


def _set_header(**updates: object) -> ArchiveMutator:
    def mutate(
        _arrays: ArchiveEntries, header: JsonObject, _data: st.DataObject
    ) -> bool:
        header.update(updates)
        return False

    return mutate


def _set_nested(
    section: str, entry: str | None, field: str, value: object
) -> ArchiveMutator:
    def mutate(
        _arrays: ArchiveEntries, header: JsonObject, _data: st.DataObject
    ) -> bool:
        target = cast("JsonObject", header[section])
        if entry is not None:
            target = cast("JsonObject", target[entry])
        target[field] = value
        return False

    return mutate


def _results_negative_recording(
    _arrays: ArchiveEntries, header: JsonObject, data: st.DataObject
) -> bool:
    analyses = cast("dict[str, JsonObject]", header["analyses"])
    analyses["operating_point"]["recording"] = data.draw(st.sampled_from([-1, -99]))
    return False


def _shard_requirement(
    unit: list[str], *, minimum_value: float = 1.0
) -> list[JsonObject]:
    return [
        {
            "analysis": "operating_point",
            "description": "bound",
            "plain": None,
            "minimum": {"value": minimum_value, "unit": unit},
            "maximum": None,
            "margin": None,
        }
    ]


def _shard_bad_yield_target(
    _arrays: ArchiveEntries, header: JsonObject, data: st.DataObject
) -> bool:
    header["requirements"] = _shard_requirement(
        ["0", "0", "0", "0", "0"], minimum_value=0.0
    )
    header["yield_target"] = data.draw(st.sampled_from([-0.5, 0.0, 2.0]))
    return False


_COMMON_MUTATORS: Final[dict[str, ArchiveMutator]] = {
    "drop_array": _drop_array,
    "bad_format": _bad_format(),
    "pickled_header": _set_array("header", _pickled_npy_bytes, rewrites_header=True),
}

_ARCHIVE_MUTATORS: Final[dict[str, dict[str, ArchiveMutator]]] = {
    "results": {
        **_COMMON_MUTATORS,
        "bad_version": _bad_version(extra=(2,)),
        "bad_format": _bad_format("other-format"),
        "bad_json": _set_array("header", b"corrupt", rewrites_header=True),
        "numeric_header": _set_array(
            "header", np.array([1.0, 2.0]), rewrites_header=True
        ),
        "corrupt_npy_array": _set_array("operating_point.values", b"\x93NUMPYcorrupt"),
        "out_of_bounds_recording": _set_nested(
            "analyses", "operating_point", "recording", 99
        ),
        "negative_recording": _results_negative_recording,
        "bad_transfer_shape": _set_array("transfer.transfer", np.array([1.0])),
        "bad_transfer_scalar": _set_array("transfer.transfer", np.array(1.0)),
        "float_logic_counts": _set_array("step.logic.counts", np.array([0.5, 1.5])),
        "negative_logic_counts": _set_array(
            "step.logic.counts", np.array([-1], dtype=np.int64)
        ),
        "mismatched_logic_counts": _set_array(
            "step.logic.counts", np.array([5], dtype=np.int64)
        ),
        "sensitivity_parameter_mismatch": _set_array(
            "sensitivity.parameters", np.array([1.0, 2.0, 3.0])
        ),
        "wrong_circuit_digest": _set_nested("provenance", None, "circuit", "0" * 64),
    },
    "batches": {
        **_COMMON_MUTATORS,
        "bad_version": _bad_version(),
        "bad_fraction_zero_denominator": _set_nested(
            "batches", "monte_carlo", "units", [["1/0", "0", "0", "0", "0"]]
        ),
        "bad_fraction_text": _set_nested(
            "batches", "monte_carlo", "units", [["not_a_fraction", "0", "0", "0", "0"]]
        ),
        "corrupt_array": _set_array("monte_carlo.values", b"corrupt"),
        "missing_trace_axis": _set_nested(
            "batches",
            "monte_carlo",
            "traces",
            [
                {
                    "analysis": "operating_point",
                    "along": "point",
                    "unknowns": [{"name": "out", "nature": "voltage"}],
                }
            ],
        ),
    },
    "shard": {
        **_COMMON_MUTATORS,
        "bad_version": _bad_version(),
        "start_greater_than_stop": _set_header(start=2, stop=1),
        "negative_start": _set_header(start=-1),
        "stop_greater_than_samples": _set_header(stop=5, samples=2),
        "empty_conditions": _set_header(conditions=[]),
        "extra_condition": _set_header(conditions=[None, None]),
        "bad_fraction_zero_denominator": _set_header(
            requirements=_shard_requirement(["1/0", "0", "0", "0", "0"])
        ),
        "bad_fraction_text": _set_header(
            requirements=_shard_requirement(["bad", "0", "0", "0", "0"])
        ),
        "empty_requirements": _set_header(requirements=[]),
        "bad_yield_target": _shard_bad_yield_target,
        "corrupt_array": _set_array("0.values", b"corrupt"),
        "wrong_shape": _set_array("0.measured", np.zeros((2, 3))),
    },
}

_ARCHIVE_LOADERS: Final[tuple[tuple[str, Callable[[Path], object]], ...]] = (
    ("load_results", files.load_results),
    ("load_provenance", files.load_provenance),
    ("load_batches", files.load_batches),
    ("load_shard", files.load_shard),
)


@FUZZ
@example(b"")
@example(b"not a zip")
@example(b"PK\x03\x04")
@example(b"\x93NUMPY\x01\x00")
@given(st.binary(max_size=512))
def test_archive_loaders_reject_arbitrary_bytes(
    scratch_directory: Path, payload: bytes
) -> None:
    path = scratch_directory / "arbitrary.npz"
    path.write_bytes(payload)
    for _, loader in _ARCHIVE_LOADERS:
        with pytest.raises(JoltyError):
            loader(path)


@FUZZ
@given(_zip_corruptions())
def test_archive_loaders_reject_truncated_or_bitflipped_zips(
    scratch_directory: Path,
    baseline_archives: _BaselineArchives,
    corruption: _ZipCorruption,
) -> None:
    source = scratch_directory / f"corrupt_source_{corruption.kind}.npz"
    _write_npz(
        source,
        baseline_archives.copy_for(corruption.kind),
        compression=corruption.compression,
    )
    raw = source.read_bytes()
    target = scratch_directory / "corrupt_target.npz"
    if corruption.xor_mask == 0:
        cut = min(len(raw) - 1, max(0, int(len(raw) * corruption.fraction)))
        target.write_bytes(raw[:cut])
    else:
        mutated = bytearray(raw)
        position = min(len(mutated) - 1, int(len(mutated) * corruption.fraction))
        mutated[position] ^= corruption.xor_mask
        target.write_bytes(bytes(mutated))
    for _, loader in _ARCHIVE_LOADERS:
        with contextlib.suppress(JoltyError):
            loader(target)


@pytest.mark.parametrize("kind", ["results", "batches", "shard"])
@FUZZ
@given(st.data())
def test_mutated_archives_only_raise_jolty_error(
    scratch_directory: Path,
    baseline_archives: _BaselineArchives,
    kind: str,
    data: st.DataObject,
) -> None:
    mutators = _ARCHIVE_MUTATORS[kind]
    mutations = data.draw(
        st.lists(st.sampled_from(tuple(mutators)), min_size=1, max_size=3)
    )
    arrays = baseline_archives.copy_for(kind)
    header = json.loads(str(arrays["header"]))
    header_rewritten = False
    for mutation in mutations:
        rewritten = mutators[mutation](arrays, header, data)
        header_rewritten = header_rewritten or rewritten
    if not header_rewritten:
        arrays["header"] = np.array(json.dumps(header))
    path = scratch_directory / f"mutated.{kind}.npz"
    _write_npz(path, arrays)
    if kind == "results":
        with contextlib.suppress(JoltyError):
            files.load_provenance(path)
        with contextlib.suppress(JoltyError):
            files.load_results(path)
    elif kind == "batches":
        with contextlib.suppress(JoltyError):
            files.load_batches(path)
    else:
        try:
            loaded = files.load_shard(path)
        except JoltyError:
            return
        with contextlib.suppress(JoltyError):
            merged, _ = shards.merge([path])
            shards.monte_carlo(merged)
        assert loaded.start <= loaded.stop <= loaded.samples


def test_archive_missing_file_still_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.npz"
    for _, loader in _ARCHIVE_LOADERS:
        with pytest.raises(FileNotFoundError):
            loader(missing)


def test_archive_format_regression_bugs(tmp_path: Path) -> None:
    # Empty file, plain .npy, pickled object array, corrupt DEFLATE, raw bytes header.
    empty = tmp_path / "empty.npz"
    empty.write_bytes(b"")
    plain_array = tmp_path / "plain.npy"
    np.save(plain_array, np.array([1.0]))
    pickled = tmp_path / "pickled.npz"
    _write_npz(pickled, {"header": _pickled_npy_bytes()})
    corrupt_deflate = tmp_path / "deflate.npz"
    with zipfile.ZipFile(corrupt_deflate, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("header.npy", b"hello world" * 100)
    raw = bytearray(corrupt_deflate.read_bytes())
    for offset in range(40, min(80, len(raw) - 40)):
        raw[offset] ^= 0xFF
    corrupt_deflate.write_bytes(bytes(raw))
    raw_header = tmp_path / "raw_header.npz"
    _write_npz(raw_header, {"header": b"corrupt"})

    for bad_path in (empty, plain_array, pickled, corrupt_deflate, raw_header):
        for _, loader in _ARCHIVE_LOADERS:
            with pytest.raises(JoltyError):
                loader(bad_path)

    # Negative recording index previously wrapped silently instead of failing.
    base_results = tmp_path / "base.results.npz"
    files.save_results(base_results, _sample_results())
    results_arrays = _read_npz_dict(base_results)
    header = json.loads(str(results_arrays["header"]))
    header["analyses"]["operating_point"]["recording"] = -1
    negative_recording = tmp_path / "negative_recording.results.npz"
    _write_npz(
        negative_recording,
        {**results_arrays, "header": np.array(json.dumps(header))},
    )
    with pytest.raises(JoltyError, match="is not a jolty results file"):
        files.load_results(negative_recording)


def test_shard_regression_bugs(tmp_path: Path) -> None:
    # Non-.npy array entry and mismatched 2-D shape must raise JoltyError at load time.
    base_shard = tmp_path / "base.shard.npz"
    files.save_shard(base_shard, _sample_shard())
    shard_arrays = _read_npz_dict(base_shard)
    overrides: tuple[tuple[str, str, np.ndarray | bytes], ...] = (
        ("raw_values.shard.npz", "0.values", b"corrupt"),
        ("bad_shape.shard.npz", "0.measured", np.zeros((2, 3))),
    )
    for filename, key, value in overrides:
        bad_path = tmp_path / filename
        entries: ArchiveEntries = dict(shard_arrays)
        entries[key] = value
        _write_npz(bad_path, entries)
        with pytest.raises(JoltyError):
            files.load_shard(bad_path)

    # shards.merge with mismatched condition counts previously raised IndexError.
    batch = _sample_batch()
    digest = "b" * 64
    shard_one = files.Shard(
        0, 2, digest, 4, 1, 0, 0, ("hot",), ("operating_point",), None, (batch, batch)
    )
    shard_two = files.Shard(
        2, 4, digest, 4, 1, 0, 0, (), ("operating_point",), None, (batch,)
    )
    path_one, path_two = tmp_path / "s1.npz", tmp_path / "s2.npz"
    files.save_shard(path_one, shard_one)
    files.save_shard(path_two, shard_two)
    with pytest.raises(JoltyError, match="same number of conditions"):
        shards.merge([path_one, path_two])

    # shards.monte_carlo with empty conditions previously raised IndexError.
    empty_conditions = files.Shard(
        0, 2, digest, 2, 1, 0, 0, (), ("operating_point",), None, ()
    )
    with pytest.raises(JoltyError, match="no conditions"):
        shards.monte_carlo(empty_conditions)
