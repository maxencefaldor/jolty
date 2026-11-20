# Copyright 2026 Maxence Faldor
"""Persistent disk cache for parsed SPICE libraries.

Stores parsed library definitions (subcircuits, model cards, parameters, functions,
options, and initial-state cards) as zlib-compressed UTF-8 JSON inside the configured
cache directory. Cache entries validate their format version, parser source fingerprint,
compatibility mode, and the resolved path, byte size, and SHA-256 digest of every
transitively included SPICE file before reuse.

Serialization uses an explicit typed JSON codec without `pickle` or `marshal`. Writes
stage to a unique temporary file in the same directory and publish via atomic rename so
concurrent readers never observe partial files. Corrupt, stale, or unwritable cache
entries fall back cleanly to fresh parsing.
"""

import contextlib
import functools
import hashlib
import json
import math
import os
import uuid
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, cast

from pydantic import JsonValue

from jolty import _core

from jolty.spice._params import NO_FUNCTIONS, Subcircuit
from jolty.spice._preprocess import SourceFile
from jolty.spice._pss import IntegrationMethod, PeriodicSteadyStateCard
from jolty.spice.expressions import (
    Binary,
    Call,
    Conditional,
    Current,
    Expression,
    Function,
    Name,
    Number,
    Unary,
    Voltage,
    fold,
)
from jolty.spice.models import ModelCard
from jolty.spice.netlist import Statement
from jolty.spice.ngspice import Compatibility

CACHE_FORMAT_VERSION: Final = 1
"""On-disk SPICE library cache schema version."""

_MAGIC: Final = b"JLTYSPC1"
"""Eight-byte header identifying a jolty SPICE library cache v1 payload."""

_COMPRESSION_LEVEL: Final = 1
"""Fast zlib compression level balancing write latency and disk size."""

_MAX_CACHE_BYTES: Final = 128 * 1024 * 1024
"""Maximum decompressed cache payload size in bytes (128 MiB).

Bounds memory usage when opening a corrupt or malformed cache entry while comfortably
exceeding full open-PDK library snapshots (roughly 8 MiB uncompressed for SKY130).
"""

_CACHE_SUBDIRECTORY: Final = "spice"
"""Subdirectory under `Settings.cache` holding parsed SPICE library cache entries."""

_PARSER_MODULES: Final = (
    "netlist.py",
    "expressions.py",
    "_preprocess.py",
    "_params.py",
    "models.py",
    "_pss.py",
    "_cache.py",
)
"""Parser and codec source files hashed into the cache fingerprint."""

_VALID_UNARY_OPERATORS: Final = frozenset({"+", "-", "!"})
_VALID_BINARY_OPERATORS: Final = frozenset(
    {"||", "&&", "==", "!=", "<", "<=", ">", ">=", "+", "-", "*", "/", "^"}
)
_VALID_INTEGRATION_METHODS: Final = frozenset(
    {"trapezoidal", "gear2", "backward_euler"}
)

_MANIFEST_FIELDS: Final = 4
_STATEMENT_FIELDS: Final = 3
_FUNCTION_FIELDS: Final = 2
_PARAMETER_PAIR_FIELDS: Final = 2
_MODEL_FIELDS: Final = 5
_SUBCIRCUIT_FIELDS: Final = 6
_VOLTAGE_SINGLE_FIELDS: Final = 2
_VOLTAGE_PAIR_FIELDS: Final = 3
_CURRENT_FIELDS: Final = 2
_UNARY_FIELDS: Final = 3
_BINARY_FIELDS: Final = 4
_CONDITIONAL_FIELDS: Final = 4
_CALL_FIELDS: Final = 3


class _CacheCorruptError(Exception):
    """Internal signal raised when a cache file fails structural validation."""


@dataclass(frozen=True, slots=True)
class LibrarySnapshot:
    """Immutable snapshot of all parsed definitions inside a `Library`."""

    subcircuits: tuple[Subcircuit, ...]
    models: tuple[ModelCard, ...]
    parameters: tuple[tuple[str, Expression], ...]
    options: tuple[tuple[str, float], ...]
    functions: tuple[tuple[str, Function], ...]
    initial_conditions: tuple[tuple[str, float], ...]
    nodesets: tuple[tuple[str, float], ...]
    periodic_steady_states: tuple[PeriodicSteadyStateCard, ...]
    has_evaluated_directives: bool
    evaluated_temperature: float | None


@functools.cache
def _parser_fingerprint() -> str:
    """Returns a SHA-256 fingerprint of format version, package version, and parser."""
    hasher = hashlib.sha256()
    hasher.update(f"format={CACHE_FORMAT_VERSION}\n".encode())
    hasher.update(f"version={_core.__version__}\n".encode())
    directory = Path(__file__).resolve().parent
    for module_name in _PARSER_MODULES:
        hasher.update(f"module={module_name}\n".encode())
        try:
            source_bytes = (directory / module_name).read_bytes()
        except OSError:
            continue
        hasher.update(source_bytes)
    return hasher.hexdigest()


def cache_key(
    sources: Sequence[tuple[Path, str | None]],
    compatibility: Compatibility | None,
) -> str:
    """Computes the deterministic SHA-256 cache key for `sources` and `compatibility`.

    Args:
        sources: Ordered `(path, section)` SPICE sources to load.
        compatibility: Optional ngspice compatibility mode applied to the library.

    Returns:
        Lowercase 64-character hexadecimal SHA-256 cache key.
    """
    canonical_sources: list[list[str | None]] = []
    for path, section in sources:
        folded_section = None if section is None else section.casefold()
        canonical_sources.append(
            [str(path.absolute()), str(path.resolve()), folded_section]
        )
    key_payload = {
        "format": CACHE_FORMAT_VERSION,
        "fingerprint": _parser_fingerprint(),
        "compatibility": compatibility,
        "sources": canonical_sources,
    }
    encoded = json.dumps(key_payload, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def cache_file(cache_directory: Path, key: str) -> Path:
    """Returns the on-disk cache path for `key` inside `cache_directory`."""
    return cache_directory / _CACHE_SUBDIRECTORY / f"{key}.zlib"


def read_cached(
    cache_directory: Path,
    key: str,
    *,
    compatibility: Compatibility | None,
    temperature: float,
) -> LibrarySnapshot | None:
    """Reads and validates a cached `LibrarySnapshot`, returning `None` on miss/error.

    Args:
        cache_directory: Root cache directory (`Settings.cache`).
        key: Deterministic cache key produced by `cache_key`.
        compatibility: Expected ngspice compatibility mode.
        temperature: Requested library temperature in kelvins.

    Returns:
        Decoded `LibrarySnapshot` when present and valid, or `None` otherwise.
    """
    path = cache_file(cache_directory, key)
    try:
        raw_bytes = path.read_bytes()
    except OSError:
        return None
    try:
        payload = _decompress_payload(raw_bytes)
        if not _header_matches(payload, compatibility, temperature):
            return None
        if not _manifest_matches(payload.get("manifest")):
            return None
        return _decode_snapshot(payload)
    except (
        _CacheCorruptError,
        OSError,
        OverflowError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        zlib.error,
        RecursionError,
    ):
        return None


def write_cached(
    cache_directory: Path,
    key: str,
    snapshot: LibrarySnapshot,
    manifest: Sequence[SourceFile],
    *,
    compatibility: Compatibility | None,
) -> None:
    """Atomically writes `snapshot` and `manifest` to `cache_directory` under `key`.

    Silently skips caching when the payload cannot be encoded or the destination is
    not writable.

    Args:
        cache_directory: Root cache directory (`Settings.cache`).
        key: Deterministic cache key produced by `cache_key`.
        snapshot: Immutable parsed library definitions to persist.
        manifest: All transitively included SPICE source files read during parsing.
        compatibility: ngspice compatibility mode recorded in the cache header.
    """
    try:
        payload = _encode_payload(snapshot, manifest, compatibility)
        encoded = json.dumps(
            payload,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        compressed = _MAGIC + zlib.compress(encoded, level=_COMPRESSION_LEVEL)
    except ValueError, TypeError, RecursionError, zlib.error:
        return
    target = cache_file(cache_directory, key)
    temporary = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(compressed)
        temporary.replace(target)
    except OSError:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise


def _reject_non_finite(token: str) -> float:
    """Rejects JSON `NaN`/`Infinity` tokens during decoding."""
    message = f"non-finite JSON constant: {token}"
    raise _CacheCorruptError(message)


def _decompress_payload(raw_bytes: bytes) -> dict[str, object]:
    """Validates `_MAGIC`, decompresses under `_MAX_CACHE_BYTES`, and parses JSON."""
    if not raw_bytes.startswith(_MAGIC):
        raise _CacheCorruptError
    decompressor = zlib.decompressobj()
    decompressed = decompressor.decompress(raw_bytes[len(_MAGIC) :], _MAX_CACHE_BYTES)
    if decompressor.unconsumed_tail or not decompressor.eof or decompressor.unused_data:
        raise _CacheCorruptError
    decoded: object = json.loads(
        decompressed.decode("utf-8"),
        parse_constant=_reject_non_finite,
    )
    if not isinstance(decoded, dict):
        raise _CacheCorruptError
    return cast("dict[str, object]", decoded)


def _header_matches(
    payload: Mapping[str, object],
    compatibility: Compatibility | None,
    temperature: float,
) -> bool:
    """Validates format version, parser fingerprint, compatibility, and temperature."""
    if (
        payload.get("format") != CACHE_FORMAT_VERSION
        or payload.get("fingerprint") != _parser_fingerprint()
        or payload.get("compatibility") != compatibility
    ):
        return False
    has_evaluated = payload.get("has_evaluated_directives")
    if type(has_evaluated) is not bool:
        return False
    evaluated_temperature = payload.get("evaluated_temperature")
    if has_evaluated:
        return (
            type(evaluated_temperature) is float or type(evaluated_temperature) is int
        ) and float(evaluated_temperature) == temperature
    return evaluated_temperature is None


def _manifest_entry_matches(entry: object) -> bool:
    """Returns whether one manifest entry `[opened, resolved, size, digest]` matches."""
    if not isinstance(entry, list) or len(entry) != _MANIFEST_FIELDS:
        return False
    opened, resolved, expected_size, expected_digest = entry
    if (
        type(opened) is not str
        or type(resolved) is not str
        or type(expected_size) is not int
        or expected_size < 0
        or type(expected_digest) is not str
    ):
        return False
    path = Path(opened)
    try:
        if str(path.resolve()) != resolved:
            return False
        with path.open("rb") as stream:
            return (
                os.fstat(stream.fileno()).st_size == expected_size
                and hashlib.file_digest(stream, "sha256").hexdigest() == expected_digest
            )
    except OSError:
        return False


def _manifest_matches(raw_manifest: object) -> bool:
    """Returns whether every file recorded in `raw_manifest` matches on disk."""
    if not isinstance(raw_manifest, list) or not raw_manifest:
        return False
    return all(_manifest_entry_matches(entry) for entry in raw_manifest)


def _encode_leaf(node: Expression) -> JsonValue | None:
    """Encodes leaf `Expression` nodes for `expressions.fold`."""
    match node:
        case Number(value):
            return float(value)
        case Name(name):
            return name
        case Voltage(positive, None):
            return ["v", positive]
        case Voltage(positive, negative):
            return ["v", positive, negative]
        case Current(source):
            return ["i", source]
    return None


def _encode_compound(node: Expression, built: list[JsonValue]) -> JsonValue:
    """Encodes compound `Expression` nodes from already-encoded child values."""
    match node:
        case Unary(operator, _):
            return ["u", operator, built[0]]
        case Binary(operator, _, _):
            return ["b", operator, built[0], built[1]]
        case Conditional():
            return ["?", built[0], built[1], built[2]]
        case Call(function, _):
            return ["c", function, built]
    message = f"unexpected expression node: {type(node).__name__}"
    raise ValueError(message)


def _encode_expression(expression: Expression) -> JsonValue:
    """Encodes `expression` into a compact JSON representation iteratively."""
    if type(expression) is Number:
        return float(expression.value)
    if type(expression) is Name:
        return expression.name
    return fold(expression, _encode_leaf, _encode_compound)


def _encode_expression_map(
    mapping: Mapping[str, Expression],
) -> dict[str, JsonValue]:
    """Encodes an ordered mapping of expressions to a JSON object."""
    return {
        key: float(expression.value)
        if type(expression) is Number
        else expression.name
        if type(expression) is Name
        else fold(expression, _encode_leaf, _encode_compound)
        for key, expression in mapping.items()
    }


def _encode_functions(
    functions: Mapping[str, Function],
) -> dict[str, JsonValue]:
    """Encodes user-defined `.func` mappings to a JSON object."""
    return {
        name: [list(function.parameters), _encode_expression(function.body)]
        for name, function in functions.items()
    }


class _FileTable:
    """Interns statement file names to integer indices during cache encoding."""

    __slots__ = ("_indices", "files")

    def __init__(self) -> None:
        self._indices: dict[str, int] = {}
        self.files: list[str] = []

    def index(self, file_name: str) -> int:
        """Returns the integer table index for `file_name`, adding it if new."""
        existing = self._indices.get(file_name)
        if existing is not None:
            return existing
        position = len(self.files)
        self._indices[file_name] = position
        self.files.append(file_name)
        return position

    def statement(self, statement: Statement) -> list[JsonValue]:
        """Encodes `statement` as `[file_index, line, text]`."""
        return [self.index(statement.file), statement.line, statement.text]


def _encode_periodic_steady_state(
    card: PeriodicSteadyStateCard,
    file_table: _FileTable,
) -> dict[str, JsonValue]:
    """Encodes one `PeriodicSteadyStateCard` into a JSON object."""
    return {
        "autonomous": card.autonomous,
        "period": card.period,
        "frequency": card.frequency,
        "stabilization": card.stabilization,
        "oscillator_node": card.oscillator_node,
        "points": card.points,
        "harmonics": card.harmonics,
        "max_shooting_iterations": card.max_shooting_iterations,
        "steady_coefficient": card.steady_coefficient,
        "max_step": card.max_step,
        "method": card.method,
        "operating_point": card.operating_point,
        "statement": (
            None if card.statement is None else file_table.statement(card.statement)
        ),
    }


def _encode_payload(
    snapshot: LibrarySnapshot,
    manifest: Sequence[SourceFile],
    compatibility: Compatibility | None,
) -> dict[str, JsonValue]:
    """Encodes `snapshot` and `manifest` into the top-level JSON dictionary."""
    file_table = _FileTable()
    encoded_models: list[JsonValue] = [
        [
            card.name,
            card.kind,
            card.version,
            file_table.statement(card.statement),
            _encode_expression_map(card.parameters),
        ]
        for card in snapshot.models
    ]
    encoded_subcircuits: list[JsonValue] = [
        [
            subcircuit.name,
            list(subcircuit.pins),
            file_table.statement(subcircuit.statement),
            _encode_expression_map(subcircuit.defaults),
            _encode_functions(subcircuit.functions),
            [file_table.statement(statement) for statement in subcircuit.body],
        ]
        for subcircuit in snapshot.subcircuits
    ]
    encoded_parameters: list[JsonValue] = [
        [name, _encode_expression(expression)]
        for name, expression in snapshot.parameters
    ]
    encoded_manifest: list[JsonValue] = [
        [entry.opened, entry.resolved, entry.size, entry.digest] for entry in manifest
    ]
    encoded_periodic: list[JsonValue] = [
        _encode_periodic_steady_state(card, file_table)
        for card in snapshot.periodic_steady_states
    ]
    return {
        "format": CACHE_FORMAT_VERSION,
        "fingerprint": _parser_fingerprint(),
        "compatibility": compatibility,
        "has_evaluated_directives": snapshot.has_evaluated_directives,
        "evaluated_temperature": snapshot.evaluated_temperature,
        "manifest": encoded_manifest,
        "files": list(file_table.files),
        "options": {name: float(value) for name, value in snapshot.options},
        "functions": _encode_functions(dict(snapshot.functions)),
        "parameters": encoded_parameters,
        "models": encoded_models,
        "subcircuits": encoded_subcircuits,
        "initial_conditions": {
            node: float(voltage) for node, voltage in snapshot.initial_conditions
        },
        "nodesets": {node: float(voltage) for node, voltage in snapshot.nodesets},
        "periodic_steady_states": encoded_periodic,
    }


def _as_finite_float(raw: object) -> float:
    """Validates `raw` as a finite numeric JSON value (excluding booleans)."""
    if type(raw) is float:
        if not math.isfinite(raw):
            raise _CacheCorruptError
        return raw
    if type(raw) is int:
        value = float(raw)
        if not math.isfinite(value):
            raise _CacheCorruptError
        return value
    raise _CacheCorruptError


def _as_optional_float(raw: object) -> float | None:
    """Validates `raw` as `None` or a finite float."""
    return None if raw is None else _as_finite_float(raw)


def _as_optional_int(raw: object) -> int | None:
    """Validates `raw` as `None` or an integer (excluding booleans)."""
    if raw is None:
        return None
    if type(raw) is not int:
        raise _CacheCorruptError
    return raw


def _as_optional_str(raw: object) -> str | None:
    """Validates `raw` as `None` or a string."""
    if raw is None:
        return None
    if type(raw) is not str:
        raise _CacheCorruptError
    return raw


def _as_bool(raw: object) -> bool:
    """Validates `raw` as a strict boolean."""
    if type(raw) is not bool:
        raise _CacheCorruptError
    return raw


def _as_str(raw: object) -> str:
    """Validates `raw` as a string."""
    if type(raw) is not str:
        raise _CacheCorruptError
    return raw


def _decode_compound_frame(
    current: list[object],
    frames: list[tuple[object, bool]],
    results: list[Expression],
) -> None:
    """Validates one compound expression list and schedules its children."""
    if not current or type(current[0]) is not str:
        raise _CacheCorruptError
    tag = current[0]
    length = len(current)
    if tag == "v":
        _decode_voltage_frame(current, length, results)
        return
    if tag == "i":
        if length != _CURRENT_FIELDS or type(current[1]) is not str:
            raise _CacheCorruptError
        results.append(Current(current[1]))
        return
    if (
        tag == "u"
        and length == _UNARY_FIELDS
        and type(current[1]) is str
        and current[1] in _VALID_UNARY_OPERATORS
    ):
        frames.append((current, True))
        frames.append((current[2], False))
        return
    if (
        tag == "b"
        and length == _BINARY_FIELDS
        and type(current[1]) is str
        and current[1] in _VALID_BINARY_OPERATORS
    ):
        frames.append((current, True))
        frames.append((current[3], False))
        frames.append((current[2], False))
        return
    if tag == "?" and length == _CONDITIONAL_FIELDS:
        frames.append((current, True))
        frames.append((current[3], False))
        frames.append((current[2], False))
        frames.append((current[1], False))
        return
    if (
        tag == "c"
        and length == _CALL_FIELDS
        and type(current[1]) is str
        and isinstance(current[2], list)
    ):
        frames.append((current, True))
        frames.extend((argument, False) for argument in reversed(current[2]))
        return
    raise _CacheCorruptError


def _decode_voltage_frame(
    current: list[object], length: int, results: list[Expression]
) -> None:
    """Decodes a `["v", positive]` or `["v", positive, negative]` expression."""
    if length == _VOLTAGE_SINGLE_FIELDS and type(current[1]) is str:
        results.append(Voltage(current[1], None))
        return
    if (
        length == _VOLTAGE_PAIR_FIELDS
        and type(current[1]) is str
        and type(current[2]) is str
    ):
        results.append(Voltage(current[1], current[2]))
        return
    raise _CacheCorruptError


def _finish_compound_frame(current: list[object], results: list[Expression]) -> None:
    """Combines decoded child expressions for `current` onto `results`."""
    tag = current[0]
    if tag == "u" and type(current[1]) is str:
        operand = results.pop()
        results.append(Unary(current[1], operand))
        return
    if tag == "b" and type(current[1]) is str:
        right = results.pop()
        left = results.pop()
        results.append(Binary(current[1], left, right))
        return
    if tag == "?":
        otherwise = results.pop()
        then_branch = results.pop()
        condition = results.pop()
        results.append(Conditional(condition, then_branch, otherwise))
        return
    if tag == "c" and type(current[1]) is str and isinstance(current[2], list):
        arity = len(current[2])
        if arity:
            arguments = tuple(results[-arity:])
            del results[-arity:]
        else:
            arguments = ()
        results.append(Call(current[1], arguments))
        return
    raise _CacheCorruptError


def _decode_atom(raw: object) -> Expression | None:
    """Decodes scalar `Number` or `Name` JSON atoms, returning `None` for compounds."""
    if type(raw) is float:
        if not math.isfinite(raw):
            raise _CacheCorruptError
        return Number(raw)
    if type(raw) is str:
        return Name(raw)
    if type(raw) is int:
        value = float(raw)
        if not math.isfinite(value):
            raise _CacheCorruptError
        return Number(value)
    return None


def _decode_expression(raw: object) -> Expression:
    """Decodes an encoded JSON value back into an `Expression` iteratively."""
    direct = _decode_atom(raw)
    if direct is not None:
        return direct
    frames: list[tuple[object, bool]] = [(raw, False)]
    results: list[Expression] = []
    while frames:
        current, ready = frames.pop()
        if not ready:
            atom = _decode_atom(current)
            if atom is not None:
                results.append(atom)
                continue
            if not isinstance(current, list):
                raise _CacheCorruptError
            _decode_compound_frame(current, frames, results)
            continue
        if not isinstance(current, list):
            raise _CacheCorruptError
        _finish_compound_frame(current, results)
    if len(results) != 1:
        raise _CacheCorruptError
    return results[0]


def _decode_expression_map(raw: object) -> Mapping[str, Expression]:
    """Decodes a JSON object into an immutable mapping of expressions."""
    if not isinstance(raw, dict):
        raise _CacheCorruptError
    isfinite = math.isfinite
    decoded: dict[str, Expression] = {}
    for key, value in raw.items():
        if type(key) is not str:
            raise _CacheCorruptError
        if type(value) is float and isfinite(value):
            decoded[key] = Number(value)
        elif type(value) is str:
            decoded[key] = Name(value)
        else:
            decoded[key] = _decode_expression(value)
    return MappingProxyType(decoded)


def _decode_float_pairs(raw: object) -> tuple[tuple[str, float], ...]:
    """Decodes a JSON object of finite floats into an ordered `(name, value)` tuple."""
    if not isinstance(raw, dict):
        raise _CacheCorruptError
    return tuple((_as_str(key), _as_finite_float(value)) for key, value in raw.items())


def _decode_functions(raw: object) -> dict[str, Function]:
    """Decodes a JSON object of `.func` definitions."""
    if not isinstance(raw, dict):
        raise _CacheCorruptError
    decoded: dict[str, Function] = {}
    for name, entry in raw.items():
        if (
            type(name) is not str
            or not isinstance(entry, list)
            or len(entry) != _FUNCTION_FIELDS
            or not isinstance(entry[0], list)
        ):
            raise _CacheCorruptError
        parameters = tuple(_as_str(parameter) for parameter in entry[0])
        decoded[name] = Function(parameters, _decode_expression(entry[1]))
    return decoded


def _decode_statement(raw: object, files: Sequence[str]) -> Statement:
    """Decodes `[file_index, line, text]` using interned `files`."""
    if not isinstance(raw, list) or len(raw) != _STATEMENT_FIELDS:
        raise _CacheCorruptError
    file_index, line, text = raw
    if (
        type(file_index) is not int
        or not (0 <= file_index < len(files))
        or type(line) is not int
        or line < 1
        or type(text) is not str
    ):
        raise _CacheCorruptError
    return Statement(files[file_index], line, text)


def _decode_models(raw: object, files: Sequence[str]) -> tuple[ModelCard, ...]:
    """Decodes the ordered sequence of global `ModelCard` entries."""
    if not isinstance(raw, list):
        raise _CacheCorruptError
    cards: list[ModelCard] = []
    for entry in raw:
        if not isinstance(entry, list) or len(entry) != _MODEL_FIELDS:
            raise _CacheCorruptError
        name, kind, card_version, raw_statement, raw_parameters = entry
        cards.append(
            ModelCard(
                name=_as_str(name),
                kind=_as_str(kind),
                parameters=_decode_expression_map(raw_parameters),
                statement=_decode_statement(raw_statement, files),
                version=_as_optional_str(card_version),
            )
        )
    return tuple(cards)


def _decode_subcircuits(raw: object, files: Sequence[str]) -> tuple[Subcircuit, ...]:
    """Decodes the ordered sequence of `Subcircuit` definitions."""
    if not isinstance(raw, list):
        raise _CacheCorruptError
    decoded: list[Subcircuit] = []
    for entry in raw:
        if not isinstance(entry, list) or len(entry) != _SUBCIRCUIT_FIELDS:
            raise _CacheCorruptError
        (
            name,
            raw_pins,
            raw_statement,
            raw_defaults,
            raw_functions,
            raw_body,
        ) = entry
        if not isinstance(raw_pins, list) or not isinstance(raw_body, list):
            raise _CacheCorruptError
        local_functions = _decode_functions(raw_functions)
        decoded.append(
            Subcircuit(
                name=_as_str(name),
                pins=tuple(_as_str(pin) for pin in raw_pins),
                defaults=_decode_expression_map(raw_defaults),
                body=tuple(_decode_statement(item, files) for item in raw_body),
                statement=_decode_statement(raw_statement, files),
                functions=(
                    MappingProxyType(local_functions)
                    if local_functions
                    else NO_FUNCTIONS
                ),
            )
        )
    return tuple(decoded)


def _decode_parameters(raw: object) -> tuple[tuple[str, Expression], ...]:
    """Decodes ordered global `.param` `(name, expression)` pairs."""
    if not isinstance(raw, list):
        raise _CacheCorruptError
    pairs: list[tuple[str, Expression]] = []
    for entry in raw:
        if not isinstance(entry, list) or len(entry) != _PARAMETER_PAIR_FIELDS:
            raise _CacheCorruptError
        pairs.append((_as_str(entry[0]), _decode_expression(entry[1])))
    return tuple(pairs)


def _decode_integration_method(raw: object) -> IntegrationMethod | None:
    """Validates an optional `.pss` integration method literal."""
    if raw is None:
        return None
    if type(raw) is not str or raw not in _VALID_INTEGRATION_METHODS:
        raise _CacheCorruptError
    return cast(
        "Literal['trapezoidal', 'gear2', 'backward_euler']",
        raw,
    )


def _decode_periodic_steady_states(
    raw: object, files: Sequence[str]
) -> tuple[PeriodicSteadyStateCard, ...]:
    """Decodes ordered `.pss` `PeriodicSteadyStateCard` entries."""
    if not isinstance(raw, list):
        raise _CacheCorruptError
    cards: list[PeriodicSteadyStateCard] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise _CacheCorruptError
        raw_statement = entry.get("statement")
        statement = (
            None if raw_statement is None else _decode_statement(raw_statement, files)
        )
        cards.append(
            PeriodicSteadyStateCard(
                autonomous=_as_bool(entry["autonomous"]),
                period=_as_optional_float(entry["period"]),
                frequency=_as_optional_float(entry["frequency"]),
                stabilization=_as_optional_float(entry["stabilization"]),
                oscillator_node=_as_optional_str(entry["oscillator_node"]),
                points=_as_optional_int(entry["points"]),
                harmonics=_as_optional_int(entry["harmonics"]),
                max_shooting_iterations=_as_optional_int(
                    entry["max_shooting_iterations"]
                ),
                steady_coefficient=_as_optional_float(entry["steady_coefficient"]),
                max_step=_as_optional_float(entry["max_step"]),
                method=_decode_integration_method(entry["method"]),
                operating_point=_as_bool(entry["operating_point"]),
                statement=statement,
            )
        )
    return tuple(cards)


def _decode_snapshot(payload: Mapping[str, object]) -> LibrarySnapshot:
    """Decodes a validated cache JSON `payload` into a `LibrarySnapshot`."""
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise _CacheCorruptError
    files = tuple(_as_str(name) for name in raw_files)
    has_evaluated = _as_bool(payload["has_evaluated_directives"])
    evaluated_temperature = _as_optional_float(payload["evaluated_temperature"])
    return LibrarySnapshot(
        subcircuits=_decode_subcircuits(payload["subcircuits"], files),
        models=_decode_models(payload["models"], files),
        parameters=_decode_parameters(payload["parameters"]),
        options=_decode_float_pairs(payload["options"]),
        functions=tuple(_decode_functions(payload["functions"]).items()),
        initial_conditions=_decode_float_pairs(payload["initial_conditions"]),
        nodesets=_decode_float_pairs(payload["nodesets"]),
        periodic_steady_states=_decode_periodic_steady_states(
            payload["periodic_steady_states"], files
        ),
        has_evaluated_directives=has_evaluated,
        evaluated_temperature=evaluated_temperature,
    )
