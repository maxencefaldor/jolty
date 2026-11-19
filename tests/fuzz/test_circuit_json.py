# Copyright 2026 Maxence Faldor
"""Hypothesis fuzz tests for the circuit format JSON and Rust core parsers."""

import contextlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final, cast

import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from pydantic import ValidationError

from fuzz.helpers import FAST_FUZZ, FUZZ, JsonObject
from jolty import (
    Builder,
    ElaborationError,
    JoltyError,
    Role,
    Simulation,
    _core,
    circuit,
    component,
    devices,
    files,
    ir,
    variation,
)
from jolty.spice import export
from jolty.units import V, kohm

type CircuitMutator = Callable[[JsonObject, st.DrawFn], None]


def _base_circuit_document() -> JsonObject:
    """Returns a fresh valid v5 circuit JSON document with hierarchy and ratings."""
    return {
        "format": "jolty-circuit",
        "version": 5,
        "top": "top",
        "components": {
            "stage": {
                "ports": [
                    {"name": "in", "role": "input"},
                    {"name": "out", "role": "output"},
                    {"name": "gnd", "role": "ground"},
                ],
                "instances": [
                    {
                        "name": "R1",
                        "of": {"model": "resistor"},
                        "parameters": {"resistance": 1000.0},
                        "terminals": {"a": "in", "b": "out"},
                        "tolerances": {
                            "resistance": {"relative": 0.05, "distribution": "uniform"}
                        },
                        "ratings": [{"stress": "power", "maximum": 0.25}],
                    }
                ],
            },
            "top": {
                "ports": [{"name": "0", "role": "ground"}],
                "instances": [
                    {
                        "name": "V1",
                        "of": {"model": "voltage_source"},
                        "parameters": {"waveform": {"dc": 5.0}},
                        "terminals": {"positive": "vin", "negative": "0"},
                    },
                    {
                        "name": "X1",
                        "of": {"component": "stage"},
                        "terminals": {"in": "vin", "out": "vout", "gnd": "0"},
                    },
                    {
                        "name": "Rload",
                        "of": {"model": "resistor"},
                        "parameters": {"resistance": 1000.0},
                        "terminals": {"a": "vout", "b": "0"},
                    },
                ],
            },
        },
    }


def _component_instances(document: JsonObject, name: str) -> list[JsonObject]:
    components = cast("dict[str, JsonObject]", document["components"])
    return cast("list[JsonObject]", components[name]["instances"])


def _set_root(key: str, choices: Sequence[object]) -> CircuitMutator:
    def mutate(document: JsonObject, draw: st.DrawFn) -> None:
        document[key] = draw(st.sampled_from(choices))

    return mutate


def _set_instance(
    component_name: str, index: int, field: str, choices: Sequence[object]
) -> CircuitMutator:
    def mutate(document: JsonObject, draw: st.DrawFn) -> None:
        _component_instances(document, component_name)[index][field] = draw(
            st.sampled_from(choices)
        )

    return mutate


def _append_instance(
    component_name: str, instance: Mapping[str, object]
) -> CircuitMutator:
    def mutate(document: JsonObject, _draw: st.DrawFn) -> None:
        _component_instances(document, component_name).append(dict(instance))

    return mutate


def _mutate_drop_key(document: JsonObject, draw: st.DrawFn) -> None:
    document.pop(
        draw(st.sampled_from(["format", "version", "top", "components"])), None
    )


def _mutate_duplicate_instance(document: JsonObject, _draw: st.DrawFn) -> None:
    top_instances = _component_instances(document, "top")
    top_instances.append(dict(top_instances[0]))


def _mutate_bad_tolerance(document: JsonObject, draw: st.DrawFn) -> None:
    parameter = draw(st.sampled_from(["resistance", "missing", "a.b.c", ""]))
    _component_instances(document, "stage")[0]["tolerances"] = {
        parameter: draw(
            st.sampled_from(
                [
                    {"relative": -0.1, "distribution": "uniform"},
                    {"relative": 1.0, "distribution": "uniform"},
                    {"relative": 2.0, "distribution": "normal"},
                    {"relative": 0.1, "distribution": "invalid"},
                    {"relative": 0.1, "lot": {"name": "a/b", "relative": -1.0}},
                ]
            )
        )
    }


def _mutate_bad_role(document: JsonObject, draw: st.DrawFn) -> None:
    components = cast("dict[str, JsonObject]", document["components"])
    ports = cast("list[JsonObject]", components["top"]["ports"])
    ports[0]["role"] = draw(st.sampled_from(["unknown", "", 1, None]))


_CIRCUIT_MUTATORS: Final[dict[str, CircuitMutator]] = {
    "wrong_format": _set_root("format", ("", "other", "jolty", 123, None)),
    "wrong_version": _set_root(
        "version", (-1, 0, 1, 3, 4, 6, 99, 2**40, "5", 5.5, None)
    ),
    "drop_key": _mutate_drop_key,
    "extra_key": _set_root("unexpected_field", (1, "x", {}, [])),
    "bad_top": _set_root("top", ("missing", "", "a/b", " stage ", 42, None)),
    "self_cycle": _append_instance(
        "stage",
        {
            "name": "Xloop",
            "of": {"component": "stage"},
            "terminals": {"in": "in", "out": "out", "gnd": "gnd"},
        },
    ),
    "mutual_cycle": _append_instance(
        "stage",
        {"name": "Xback", "of": {"component": "top"}, "terminals": {"0": "gnd"}},
    ),
    "duplicate_instance": _mutate_duplicate_instance,
    "bad_model": _set_instance(
        "top",
        2,
        "of",
        tuple({"model": name} for name in ("", "no_such_model", "a/b", "res/1")),
    ),
    "bad_parameter_number": _set_instance(
        "top",
        2,
        "parameters",
        tuple(
            {"resistance": value}
            for value in (-1.0, 0.0, 1e308, -1e308, "1k", None, True, [], {})
        ),
    ),
    "bad_terminals": _set_instance(
        "top",
        2,
        "terminals",
        (
            {},
            {"a": "vout"},
            {"wrong": "0", "extra": "vout"},
            {"a": "", "b": "0"},
            {"a": "v/out", "b": "0"},
        ),
    ),
    "bad_tolerance": _mutate_bad_tolerance,
    "bad_rating": _set_instance(
        "stage",
        0,
        "ratings",
        tuple(
            [rating]
            for rating in (
                {"stress": "power", "maximum": 0.0},
                {"stress": "power", "minimum": 0.1, "maximum": 1.0},
                {"stress": "temperature", "minimum": -10.0},
                {"stress": "voltage", "terminals": ["a"], "maximum": 5.0},
                {"stress": "voltage", "terminals": ["a", "a"], "maximum": 5.0},
                {
                    "stress": "current",
                    "terminals": ["a"],
                    "minimum": 5.0,
                    "maximum": 1.0,
                },
                {"stress": "voltage", "terminals": ["a", "b"]},
            )
        ),
    ),
    "bad_part": _set_instance("stage", 0, "part", ("", " 2N3904", "X ")),
    "bad_role": _mutate_bad_role,
    "bad_source_location": _set_instance(
        "top",
        0,
        "source_location",
        (
            {"file": "x.py", "line": -1},
            {"file": "x.py", "line": 2**40},
            {"file": 123, "line": 1},
        ),
    ),
}


@st.composite
def _mutated_circuit_json(draw: st.DrawFn) -> str:
    """Produces valid-JSON or NaN/Inf-corrupted mutations of a circuit document."""
    document = _base_circuit_document()
    mutations = draw(
        st.lists(st.sampled_from(tuple(_CIRCUIT_MUTATORS)), min_size=1, max_size=3)
    )
    for kind in mutations:
        if "components" in document and isinstance(document["components"], dict):
            _CIRCUIT_MUTATORS[kind](document, draw)
    encoded = json.dumps(document)
    if draw(st.booleans()) and "1000.0" in encoded:
        replacement = draw(st.sampled_from(["1000.0", "NaN", "Infinity", "-Infinity"]))
        encoded = encoded.replace("1000.0", replacement, 1)
    return encoded


@FUZZ
@example(b"")
@example(b"\x00\xff\xfe")
@example(b"{}")
@example(b'{"format": "jolty-circuit", "version": 99}')
@given(st.binary(max_size=256))
def test_circuit_json_arbitrary_bytes_only_raise_validation_error(
    scratch_directory: Path, payload: bytes
) -> None:
    path = scratch_directory / "fuzz_bytes.jolty.json"
    path.write_bytes(payload)
    with contextlib.suppress(ValidationError):
        ir.Circuit.model_validate_json(payload)
    with contextlib.suppress(ValidationError):
        files.load_circuit(path)
    decoded = payload.decode("utf-8", errors="replace")
    with contextlib.suppress(ValueError, _core.ElaborationError):
        _core.Elaborated(decoded, "{}")
    with contextlib.suppress(ValueError):
        _core.toleranced(decoded)


@FUZZ
@example("{" * 200 + "}" * 200)
@example("[" * 200 + "]" * 200)
@given(st.text(max_size=256))
def test_circuit_json_arbitrary_text_and_core_auxiliary_parsers(text: str) -> None:
    with contextlib.suppress(ValueError, _core.ElaborationError):
        _core.Elaborated(text, "{}")
    with contextlib.suppress(ValueError, _core.ElaborationError):
        _core.Simulator(text, 300.15, "{}", "{}")
    with contextlib.suppress(ValueError):
        _core.waveform_value(text, 0.0)
    with contextlib.suppress(ValueError):
        _core.terminals("resistor", text)
    with contextlib.suppress(ValueError):
        _core.defaulted("resistor", text)
    with contextlib.suppress(ValueError):
        _core.sample(text, 0, 0)


@FUZZ
@given(_mutated_circuit_json())
def test_mutated_circuit_json_raises_only_documented_errors(
    scratch_directory: Path, encoded: str
) -> None:
    path = scratch_directory / "mutated.jolty.json"
    path.write_text(encoded, encoding="utf-8")
    try:
        parsed = files.load_circuit(path)
    except ValidationError:
        parsed = None
    if parsed is not None:
        reparsed = ir.Circuit.model_validate_json(encoded)
        assert parsed.model_dump_json() == reparsed.model_dump_json()
        assert parsed.find("X1/R1") is None or isinstance(
            parsed.find("X1/R1"), ir.Instance
        )
        for definition in parsed.components.values():
            for instance in definition.instances:
                assert isinstance(ir.numbers(instance.parameters), dict)
        with contextlib.suppress(ValueError):
            variation.toleranced(parsed)
        with contextlib.suppress(JoltyError):
            export.flatten(parsed)
        with contextlib.suppress(ElaborationError, JoltyError, ValueError):
            Simulation(parsed)
    with contextlib.suppress(ValueError, _core.ElaborationError):
        _core.Elaborated(encoded, "{}")
    with contextlib.suppress(ValueError):
        _core.toleranced(encoded)


def test_circuit_cycle_and_missing_component_regression_bugs() -> None:
    # Self-recursive and mutually recursive circuits previously segfaulted in
    # Rust variants::walk and hit RecursionError in export.flatten / export.netlist.
    self_cycle_document = _base_circuit_document()
    _component_instances(self_cycle_document, "stage").append(
        {
            "name": "Xloop",
            "of": {"component": "stage"},
            "terminals": {"in": "in", "out": "out", "gnd": "gnd"},
        }
    )
    self_cycle_json = json.dumps(self_cycle_document)
    self_cycle = ir.Circuit.model_validate_json(self_cycle_json)
    with pytest.raises(ValueError, match="contains itself: stage -> stage"):
        _core.toleranced(self_cycle_json)
    with pytest.raises(ValueError, match="contains itself: stage -> stage"):
        variation.toleranced(self_cycle)
    with pytest.raises(JoltyError, match="contains itself: stage -> stage"):
        export.flatten(self_cycle)
    with pytest.raises(JoltyError, match="contains itself: stage -> stage"):
        export.netlist(self_cycle)
    with pytest.raises(_core.ElaborationError):
        _core.Elaborated(self_cycle_json, "{}")

    mutual_document = _base_circuit_document()
    _component_instances(mutual_document, "stage").append(
        {"name": "Xback", "of": {"component": "top"}, "terminals": {"0": "gnd"}}
    )
    mutual = ir.Circuit.model_validate_json(json.dumps(mutual_document))
    with pytest.raises(ValueError, match="contains itself: top -> stage -> top"):
        variation.toleranced(mutual)
    with pytest.raises(JoltyError, match="contains itself: top -> stage -> top"):
        export.flatten(mutual)

    missing_top_document = _base_circuit_document()
    missing_top_document["top"] = "missing"
    missing_top = ir.Circuit.model_validate_json(json.dumps(missing_top_document))
    assert variation.toleranced(missing_top) == ()
    with pytest.raises(JoltyError, match="no component `missing`"):
        export.flatten(missing_top)


@FAST_FUZZ
@given(
    st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_valid_circuit_json_round_trips_cleanly(
    scratch_directory: Path, resistance: float, voltage: float
) -> None:
    @component
    def divider(builder: Builder) -> None:
        ground = builder.port("ground", Role.GROUND)
        source = builder.place(devices.voltage_source(voltage * V), name="V1")
        upper = builder.place(devices.resistor(resistance * kohm), name="R1")
        lower = builder.place(devices.resistor(resistance * kohm), name="R2")
        builder.connect(source.positive, upper.a)
        builder.connect(upper.b, lower.a, name="out")
        builder.connect(lower.b, source.negative, ground)

    written = circuit(divider())
    path = scratch_directory / "roundtrip.jolty.json"
    files.save_circuit(path, written)
    loaded = files.load_circuit(path)
    assert loaded == written
    assert files.circuit_digest(loaded) == files.circuit_digest(written)
    raw = path.read_text(encoding="utf-8")
    assert _core.Elaborated(raw, "{}") is not None
