# Copyright 2026 Maxence Faldor
"""Hypothesis fuzz suite for the expression, circuit JSON, SPICE, and archive parsers.

Covers the four Python-exposed parsers named in `plans/quality.md` item 18:

1. Expression parsers: Rust `jolty._core.constant` and `jolty._core.expression_tree`,
   plus Python `jolty.spice.expressions` (`number`, `parse`, `evaluate`, `translate`,
   `inlined`, `substituted`, `names`, `integral`).
2. Circuit format JSON parser: `jolty.ir.Circuit.model_validate_json`,
   `jolty.files.load_circuit`, and Rust `jolty._core` (`Elaborated`, `Simulator`,
   `toleranced`, `level`, `terminals`, `defaulted`, `sample`).
3. SPICE netlist and library reader: `jolty.spice.netlist` (`statements`, `tokens`),
   `jolty.spice.models.read`, `jolty.spice.Library` (`read`, `include`, `values`,
   `component`, `drawn`, `survey`, `survey_model`, `random_variables`), and
   `jolty.spice.export.netlist` round-trip.
4. Archive loaders: `jolty.files` (`load_results`, `load_provenance`, `load_batches`,
   `load_shard`) and `jolty.shards` (`merge`, `monte_carlo`).

Each test runs deterministically with a fast example budget in default CI runs so the
file finishes well under ten seconds. Set `JOLTY_FUZZ_EXAMPLES` or pass
`--hypothesis-profile=nightly` for longer searches.
"""

import contextlib
import io
import json
import math
import os
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

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
    results,
    shards,
    variation,
)
from jolty.errors import SpiceError
from jolty.spice import Library, Sample, export, expressions, models, netlist
from jolty.spice.expressions import nesting_depth
from jolty.units import V, kohm

DEFAULT_EXAMPLES: Final = 30
"""Default per-property example budget keeping CI runtime fast and deterministic."""

SEARCH: Final = os.environ.get("JOLTY_FUZZ_EXAMPLES")
"""Override example count for ad-hoc fuzzing searches."""


def _fuzz_settings(default_examples: int = DEFAULT_EXAMPLES) -> settings:
    """Returns Hypothesis settings tuned for CI speed, profile overrides, or search."""
    suppressed = [HealthCheck.too_slow, HealthCheck.filter_too_much]
    if SEARCH is not None:
        return settings(
            max_examples=int(SEARCH),
            derandomize=False,
            deadline=None,
            suppress_health_check=suppressed,
        )
    if settings.get_current_profile_name() != "default":
        return settings(parent=settings.default, suppress_health_check=suppressed)
    return settings(
        max_examples=default_examples,
        derandomize=True,
        deadline=None,
        suppress_health_check=suppressed,
    )


FUZZ: Final = _fuzz_settings()
FAST_FUZZ: Final = _fuzz_settings(default_examples=20)

type ArchiveEntries = dict[str, np.ndarray | bytes]
type JsonObject = dict[str, object]
type CircuitMutator = Callable[[JsonObject, st.DrawFn], None]
type ResultsMutator = Callable[[ArchiveEntries, JsonObject, st.DataObject], None]
type BatchMutator = Callable[[ArchiveEntries, JsonObject, st.DataObject], bool]
type ShardMutator = Callable[[ArchiveEntries, JsonObject, st.DataObject], bool]


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


# ---------------------------------------------------------------------------
# Strategies: 1. Expression generators (SPICE and Rust `_core`)
# ---------------------------------------------------------------------------

_SPICE_SUFFIXES: Final = (
    "",
    "t",
    "g",
    "meg",
    "k",
    "mil",
    "m",
    "u",
    "n",
    "p",
    "f",
    "kohm",
    "uF",
    "MEG",
)
_SPICE_BINARY_OPERATORS: Final = (
    "+",
    "-",
    "*",
    "/",
    "^",
    "**",
    "<",
    "<=",
    ">",
    ">=",
    "==",
    "!=",
    "<>",
    "&",
    "&&",
    "|",
    "||",
)
_SPICE_UNARY_OPERATORS: Final = ("-", "+", "!", "~")
_SPICE_UNARY_FUNCTIONS: Final = (
    "abs",
    "sqrt",
    "exp",
    "ln",
    "log",
    "log10",
    "floor",
    "ceil",
    "sgn",
    "sign",
    "u",
    "stp",
    "uramp",
    "int",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
)
_SPICE_BINARY_FUNCTIONS: Final = ("min", "max", "pow", "pwr", "pwrs")
_SPICE_PARAMETERS: Final[dict[str, float]] = {
    "alpha": 2.5,
    "beta": -1.25,
    "gain": 10.0,
    "offset": 0.5,
}
_CORE_UNARY_FUNCTIONS: Final = (
    "abs",
    "sqrt",
    "exp",
    "ln",
    "log",
    "log10",
    "floor",
    "ceil",
    "sgn",
    "u",
    "uramp",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
)
_CORE_BINARY_OPERATORS: Final = (
    "+",
    "-",
    "*",
    "/",
    "^",
    "<",
    "<=",
    ">",
    ">=",
    "==",
    "!=",
    "&&",
    "||",
)


def _one_of_strings(
    strategies: list[st.SearchStrategy[str]],
) -> st.SearchStrategy[str]:
    """Combines string strategies while preserving strict static typing."""
    combined = strategies[0]
    for strategy in strategies[1:]:
        combined = combined | strategy
    return combined


@st.composite
def _spice_number_literals(draw: st.DrawFn) -> str:
    """Generates syntactically valid SPICE numeric literals with scale suffixes."""
    mantissa = draw(
        st.floats(
            min_value=0.0,
            max_value=1e4,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    suffix = draw(st.sampled_from(_SPICE_SUFFIXES))
    return f"{mantissa:.4g}{suffix}"


def _valid_spice_expressions(*, with_variables: bool) -> st.SearchStrategy[str]:
    """Grammar-guided valid SPICE expression strings."""
    base_leaves: list[st.SearchStrategy[str]] = [
        _spice_number_literals(),
        st.sampled_from(["pi", "e", "temp", "vt", "gmin", *_SPICE_PARAMETERS]),
    ]
    if with_variables:
        base_leaves.append(
            st.sampled_from(
                [
                    "V(in)",
                    "V(out, ref)",
                    "V(0, ref)",
                    "V(node_a, 0)",
                    "V(vc+, com)",
                    "I(V1)",
                ]
            )
        )
    leaves = _one_of_strings(base_leaves)

    def extend(children: st.SearchStrategy[str]) -> st.SearchStrategy[str]:
        unary = st.tuples(st.sampled_from(_SPICE_UNARY_OPERATORS), children).map(
            lambda pair: f"({pair[0]}{pair[1]})"
        )
        binary = st.tuples(
            children, st.sampled_from(_SPICE_BINARY_OPERATORS), children
        ).map(lambda triple: f"({triple[0]} {triple[1]} {triple[2]})")
        ternary = st.tuples(children, children, children).map(
            lambda triple: f"({triple[0]} ? {triple[1]} : {triple[2]})"
        )
        unary_call = st.tuples(st.sampled_from(_SPICE_UNARY_FUNCTIONS), children).map(
            lambda pair: f"{pair[0]}({pair[1]})"
        )
        binary_call = st.tuples(
            st.sampled_from(_SPICE_BINARY_FUNCTIONS), children, children
        ).map(lambda triple: f"{triple[0]}({triple[1]}, {triple[2]})")
        limit_call = st.tuples(children, children, children).map(
            lambda triple: f"limit({triple[0]}, {triple[1]}, {triple[2]})"
        )
        if_call = st.tuples(children, children, children).map(
            lambda triple: f"if({triple[0]}, {triple[1]}, {triple[2]})"
        )
        braced = children.map(lambda inner: f"{{{inner}}}")
        table_call = children.map(lambda inner: f"table({inner}, -1, -1, 0, 0, 1, 1)")
        branches: list[st.SearchStrategy[str]] = [
            unary,
            binary,
            ternary,
            unary_call,
            binary_call,
            limit_call,
            if_call,
            braced,
            table_call,
        ]
        if not with_variables:
            branches.append(
                st.sampled_from(
                    [
                        "agauss(1.0, 0.1, 3.0)",
                        "gauss(2.0, 0.05, 3.0)",
                        "aunif(1.0, 0.2)",
                        "unif(2.0, 0.1)",
                        "limit(1.0, 0.1)",
                    ]
                )
            )
        return _one_of_strings(branches)

    return st.recursive(leaves, extend, max_leaves=8)


def _valid_core_expressions(*, with_variables: bool) -> st.SearchStrategy[str]:
    """Grammar-guided valid expressions in jolty's Rust expression language."""
    numbers = st.floats(
        min_value=0.0,
        max_value=1e3,
        allow_nan=False,
        allow_infinity=False,
    ).map(lambda value: f"{value:.6g}")
    base_leaves: list[st.SearchStrategy[str]] = [numbers, st.just("pi")]
    if with_variables:
        base_leaves.append(st.sampled_from(["time", "v(a)", "v(a, b)", "i(p1)"]))
    leaves = _one_of_strings(base_leaves)

    def extend(children: st.SearchStrategy[str]) -> st.SearchStrategy[str]:
        unary = st.tuples(st.sampled_from(("-", "+", "!")), children).map(
            lambda pair: f"({pair[0]}{pair[1]})"
        )
        binary = st.tuples(
            children, st.sampled_from(_CORE_BINARY_OPERATORS), children
        ).map(lambda triple: f"({triple[0]} {triple[1]} {triple[2]})")
        ternary = st.tuples(children, children, children).map(
            lambda triple: f"({triple[0]} ? {triple[1]} : {triple[2]})"
        )
        unary_call = st.tuples(st.sampled_from(_CORE_UNARY_FUNCTIONS), children).map(
            lambda pair: f"{pair[0]}({pair[1]})"
        )
        binary_call = st.tuples(
            st.sampled_from(("min", "max", "pow")), children, children
        ).map(lambda triple: f"{triple[0]}({triple[1]}, {triple[2]})")
        limit_call = st.tuples(children, children, children).map(
            lambda triple: f"limit({triple[0]}, {triple[1]}, {triple[2]})"
        )
        table_call = children.map(
            lambda inner: f"rounded_table({inner}, 0.25, -2, -1, 0, 0, 2, 1)"
        )
        return _one_of_strings(
            [unary, binary, ternary, unary_call, binary_call, limit_call, table_call]
        )

    return st.recursive(leaves, extend, max_leaves=8)


_EXPRESSION_SOUP_TOKENS: Final = (
    "0",
    "1",
    "-1",
    "1e500",
    "1e-500",
    "1e308",
    "1k",
    "2.5MEG",
    "3mil",
    "4uF",
    "1\u212a",
    "1\u0660",
    "alpha",
    "unknown_name",
    "pi",
    "time",
    "inf",
    "nan",
    "v(a)",
    "v(a,b)",
    "v()",
    "v(,)",
    "i(v1)",
    "i()",
    "i(a,b)",
    "agauss(1,2,3)",
    "agauss(1)",
    "limit(1,2,3)",
    "limit(1,2)",
    "limit(1)",
    "pow(1,2)",
    "pow(-2, 1e30)",
    "table(1,0,0,1,1)",
    "table(1,0,0,0,1)",
    "pwl(1)",
    "rounded_table(1,0.25,0,0,1,1)",
    "user_double(1)",
    "user_loop(1)",
    "user_zero()",
    "{1}",
    "'1'",
    '"1"',
    "+",
    "-",
    "*",
    "/",
    "^",
    "**",
    "?",
    ":",
    "(",
    ")",
    "{",
    "}",
    "'",
    '"',
    ",",
    "!",
    "~",
    "&",
    "&&",
    "|",
    "||",
    "==",
    "!=",
    "<>",
    "<",
    "<=",
    ">",
    ">=",
    " ",
    "\t",
    "\n",
    "$",
    ";",
)


def _expression_token_soup() -> st.SearchStrategy[str]:
    """Adversarial sequences of expression tokens and punctuation."""
    return st.lists(
        st.sampled_from(_EXPRESSION_SOUP_TOKENS), min_size=0, max_size=35
    ).map("".join)


# ---------------------------------------------------------------------------
# Strategies: 2. Circuit format JSON mutations
# ---------------------------------------------------------------------------


def _base_circuit_document() -> JsonObject:
    """Returns a fresh valid v4 circuit JSON document with hierarchy and ratings."""
    return {
        "format": "jolty-circuit",
        "version": 4,
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
                        "ratings": [
                            {"stress": "power", "maximum": 0.25},
                        ],
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


def _mutate_wrong_format(document: JsonObject, draw: st.DrawFn) -> None:
    document["format"] = draw(st.sampled_from(["", "other", "jolty", 123, None]))


def _mutate_wrong_version(document: JsonObject, draw: st.DrawFn) -> None:
    document["version"] = draw(
        st.sampled_from([-1, 0, 1, 3, 5, 99, 2**40, "4", 4.5, None])
    )


def _mutate_drop_key(document: JsonObject, draw: st.DrawFn) -> None:
    key = draw(st.sampled_from(["format", "version", "top", "components"]))
    document.pop(key, None)


def _mutate_extra_key(document: JsonObject, draw: st.DrawFn) -> None:
    document["unexpected_field"] = draw(st.sampled_from([1, "x", {}, []]))


def _mutate_bad_top(document: JsonObject, draw: st.DrawFn) -> None:
    document["top"] = draw(st.sampled_from(["missing", "", "a/b", " stage ", 42, None]))


def _mutate_self_cycle(document: JsonObject, _draw: st.DrawFn) -> None:
    _component_instances(document, "stage").append(
        {
            "name": "Xloop",
            "of": {"component": "stage"},
            "terminals": {"in": "in", "out": "out", "gnd": "gnd"},
        }
    )


def _mutate_mutual_cycle(document: JsonObject, _draw: st.DrawFn) -> None:
    _component_instances(document, "stage").append(
        {
            "name": "Xback",
            "of": {"component": "top"},
            "terminals": {"0": "gnd"},
        }
    )


def _mutate_duplicate_instance(document: JsonObject, _draw: st.DrawFn) -> None:
    top_instances = _component_instances(document, "top")
    top_instances.append(dict(top_instances[0]))


def _mutate_bad_model(document: JsonObject, draw: st.DrawFn) -> None:
    _component_instances(document, "top")[2]["of"] = {
        "model": draw(st.sampled_from(["", "no_such_model", "a/b", "res/1"]))
    }


def _mutate_bad_parameter_number(document: JsonObject, draw: st.DrawFn) -> None:
    _component_instances(document, "top")[2]["parameters"] = {
        "resistance": draw(
            st.sampled_from([-1.0, 0.0, 1e308, -1e308, "1k", None, True, [], {}])
        )
    }


def _mutate_bad_terminals(document: JsonObject, draw: st.DrawFn) -> None:
    _component_instances(document, "top")[2]["terminals"] = draw(
        st.sampled_from(
            [
                {},
                {"a": "vout"},
                {"wrong": "0", "extra": "vout"},
                {"a": "", "b": "0"},
                {"a": "v/out", "b": "0"},
            ]
        )
    )


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


def _mutate_bad_rating(document: JsonObject, draw: st.DrawFn) -> None:
    _component_instances(document, "stage")[0]["ratings"] = [
        draw(
            st.sampled_from(
                [
                    {"stress": "power", "maximum": 0.0},
                    {"stress": "power", "minimum": 0.1, "maximum": 1.0},
                    {"stress": "temperature", "minimum": -10.0},
                    {"stress": "voltage", "terminals": ["a"], "maximum": 5.0},
                    {
                        "stress": "voltage",
                        "terminals": ["a", "a"],
                        "maximum": 5.0,
                    },
                    {
                        "stress": "current",
                        "terminals": ["a"],
                        "minimum": 5.0,
                        "maximum": 1.0,
                    },
                    {"stress": "voltage", "terminals": ["a", "b"]},
                ]
            )
        )
    ]


def _mutate_bad_part(document: JsonObject, draw: st.DrawFn) -> None:
    _component_instances(document, "stage")[0]["part"] = draw(
        st.sampled_from(["", " 2N3904", "X "])
    )


def _mutate_bad_role(document: JsonObject, draw: st.DrawFn) -> None:
    components = cast("dict[str, JsonObject]", document["components"])
    ports = cast("list[JsonObject]", components["top"]["ports"])
    ports[0]["role"] = draw(st.sampled_from(["unknown", "", 1, None]))


def _mutate_bad_source_location(document: JsonObject, draw: st.DrawFn) -> None:
    _component_instances(document, "top")[0]["source_location"] = draw(
        st.sampled_from(
            [
                {"file": "x.py", "line": -1},
                {"file": "x.py", "line": 2**40},
                {"file": 123, "line": 1},
            ]
        )
    )


_CIRCUIT_MUTATORS: Final[dict[str, CircuitMutator]] = {
    "wrong_format": _mutate_wrong_format,
    "wrong_version": _mutate_wrong_version,
    "drop_key": _mutate_drop_key,
    "extra_key": _mutate_extra_key,
    "bad_top": _mutate_bad_top,
    "self_cycle": _mutate_self_cycle,
    "mutual_cycle": _mutate_mutual_cycle,
    "duplicate_instance": _mutate_duplicate_instance,
    "bad_model": _mutate_bad_model,
    "bad_parameter_number": _mutate_bad_parameter_number,
    "bad_terminals": _mutate_bad_terminals,
    "bad_tolerance": _mutate_bad_tolerance,
    "bad_rating": _mutate_bad_rating,
    "bad_part": _mutate_bad_part,
    "bad_role": _mutate_bad_role,
    "bad_source_location": _mutate_bad_source_location,
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


# ---------------------------------------------------------------------------
# Strategies: 3. SPICE netlist generators (adversarial and valid)
# ---------------------------------------------------------------------------

_SPICE_NODES: Final = ("0", "a", "b", "c", "d", "n1", "n2", "t", "ground", "a/b")
_SPICE_NUMBERS: Final = (
    "0",
    "1",
    "-1",
    "1k",
    "1u",
    "10u",
    "1n",
    "1m",
    "0.5",
    "1.5",
    "1e-14",
    "1e500",
    "1e308",
    "{1/0}",
    "{0/0}",
    "{p}",
    "{missing}",
)
_SPICE_CARD_TEMPLATES: Final = (
    ".subckt s1 a b",
    ".subckt s1 a b params: p=1",
    ".subckt s1 a b c d",
    ".subckt s2 a b",
    ".subckt",
    ".subckt s1 0 a",
    ".subckt s1 ground a",
    ".subckt s1 a/b c",
    ".ends",
    ".ends s1",
    ".end",
    ".param p=1k",
    ".param p = 1 + 2 q = 3",
    ".param a=b b=a",
    ".param = 1",
    ".param p=",
    ".param p=agauss(1k, 100, 3)",
    ".param p=agauss(1k, 1e308, 1e-10)",
    ".params x=1",
    ".func double(x) {2 * x}",
    ".func bad(x) {bad(x)}",
    ".func",
    ".option scale=1u",
    ".option scale=1e308",
    ".option wnflag=1",
    ".option",
    ".options a=1 flag",
    ".ic V(a)=1",
    ".ic V(a)=",
    ".ic",
    ".nodeset V(a)=0",
    ".if (1)",
    ".if (0)",
    ".if (p > 0)",
    ".if ()",
    ".if",
    ".elseif (1)",
    ".else",
    ".endif",
    ".model mr r rsh=10 narrow=5u short=5u kf=1e-12 af=1.5",
    ".model mc c cap=1n thick=0",
    ".model md d is=1e-14 level=3",
    ".model md d is=1e-14 level=1e500",
    ".model mq npn bf=100",
    ".model mvbic npn level=4 rth=10 kfn=1e-12 afn=1.5",
    ".model mn nmos level=1",
    ".model mb3.0 nmos level=8 lmin=0 lmax=1 wmin=0 wmax=1",
    ".model mb4.0 nmos level=54 version=4.5 lmin=0 lmax=1 wmin=0 wmax=1",
    ".model mvd vdmos pchan mfg=Vendor",
    ".model msw vswitch ron=1 roff=1meg von=2 voff=1",
    ".model mcsw iswitch ron=1 roff=1meg ion=1m ioff=0",
    ".model derived ako:md rs=10",
    ".model derived ako:missing",
    ".model",
    ".model empty",
    "R1 a b 1k",
    "R1 a b mr w=10u l=10u",
    "R1 a b mr w=0 l=10u",
    "R1 a b 1k m=0",
    "R1 a b 1k m=-1",
    "R1 a b 1k tc1=",
    "R1 a b {v(a,b)} tc1=-1",
    "R1 a b",
    "R1",
    "C1 a b 1n",
    "C1 a b mc w=1u l=1u",
    "C1 a b Q={v(a,b)*1n}",
    "C1 a b C={v(a,b)^500}",
    "C1 a b 1n m=0",
    "C1 a b 1n tc1=",
    "C1",
    "L1 a b 1u",
    "L2 a b 1u",
    "L1 a b 1u m=0",
    "L1",
    "K1 L1 L2 0.5",
    "K1 L1 L2 1.5",
    "K1 L1 L1 0.5",
    "K1 L1 L2 1e500",
    "K1",
    "D1 a b md",
    "D1 a b md area=0",
    "D1 a b md m=0",
    "D1 a b md w=0 l=0",
    "D1 a b",
    "Q1 a b c mq",
    "Q1 a b c 0 t mvbic",
    "Q1 a b c mvbic area=0",
    "Q1 a b c mvbic m=0",
    "Q1 a b c mvbic m=-1",
    "Q1",
    "J1 a b c md",
    "M1 a b c d mn w=1u l=1u",
    "M1 a b c d mb3 w=1u l=1u nf=0",
    "M1 a b c d mb4 w=1u l=1u nf=0",
    "M1 a b c mvd",
    "M1",
    "V1 a 0 5",
    "V1 a 0 PULSE(0 5 0 1n 1n 5n 10n)",
    "V1 a 0 PULSE(0 1 0 0 0 0 0)",
    "V1 a 0 PULSE()",
    "V1 a 0 SIN(0 1 1k)",
    "V1 a 0 SIN(0 1 1e500)",
    "V1 a 0 PWL(1 0 0 1)",
    "V1 a 0 PWL(1 0 0 1) DC 0",
    "V1 a 0 EXP(0 1)",
    "V1 a 0 1e500",
    "V1",
    "I1 a 0 1m",
    "I1 a",
    "I1",
    "E1 a b c d 2",
    "E1 a b VALUE {V(c,d)}",
    "E1 a b VALUE",
    "E1 a b VALUE =",
    "E1 a b TABLE {V(c,d)} = (0,0) (1,1)",
    "E1 a b TABLE {V(c,d)} = (0,0) (0,1)",
    "E1 a b POLY(1) c d 0 1",
    "E1 a b POLY(0) 1 2",
    "E1 a b POLY(9999999) c d 1",
    "E1",
    "G1 a b c d 1m",
    "G1 a b TABLE {V(c,d)} = (0,0)",
    "F1 a b V1 2",
    "F1 a b POLY(0) 1 2",
    "H1 a b V1 2",
    "B1 a b V={V(a)}",
    "B1 a b I={I(V1)}",
    "B1 a b V=",
    "B1",
    "S1 a b c d msw",
    "S1 a b c d md",
    "S1",
    "W1 a b V1 mcsw",
    "W1 a b V1 md",
    "W1",
    "T1 a b c d Z0=50 TD=1n",
    "T1 a b c d Z0=50 F=0",
    "T1 a b c d Z0=50 TD=1e500",
    "T1",
    "X1 a b s1",
    "X1 a b s2",
    "X1 a b s1 m=0",
    "X1 a b s1 p=2",
    "X1",
    "+ extra=1",
    "* comment line",
    "; inline comment",
    ",",
    "=",
    "(",
    "'unterminated",
)


def _random_spice_card(draw: st.DrawFn, index: int) -> str:
    """Synthesizes a randomized SPICE card from edge node names and numbers."""
    node_one = draw(st.sampled_from(_SPICE_NODES))
    node_two = draw(st.sampled_from(_SPICE_NODES))
    value = draw(st.sampled_from(_SPICE_NUMBERS))
    multiplier = draw(st.sampled_from(_SPICE_NUMBERS))
    kind = draw(st.sampled_from(["R", "C", "L", "V", "I", "D", "T", "param"]))
    if kind in {"R", "C", "L"}:
        return f"{kind}fuzz{index} {node_one} {node_two} {value} m={multiplier}"
    if kind in {"V", "I"}:
        return f"{kind}fuzz{index} {node_one} {node_two} {value}"
    if kind == "D":
        return f"Dfuzz{index} {node_one} {node_two} md area={value} m={multiplier}"
    if kind == "T":
        node_three = draw(st.sampled_from(_SPICE_NODES))
        node_four = draw(st.sampled_from(_SPICE_NODES))
        return (
            f"Tfuzz{index} {node_one} {node_two} {node_three} {node_four} "
            f"Z0={value} F={multiplier}"
        )
    return f".param p{index}={value}"


@st.composite
def _spice_netlists(draw: st.DrawFn) -> str:
    """Grammar-guided adversarial SPICE netlists mixing cards and directives."""
    lines = draw(
        st.lists(st.sampled_from(_SPICE_CARD_TEMPLATES), min_size=1, max_size=12)
    )
    extra_count = draw(st.integers(min_value=0, max_value=3))
    lines.extend(_random_spice_card(draw, index) for index in range(extra_count))
    if draw(st.booleans()):
        # Ensure at least one closed subcircuit wraps body cards so element readers run.
        wrapped = [".subckt s1 a b c d", *lines, ".ends"]
        return "\n".join(wrapped)
    return "\n".join(lines)


@st.composite
def _valid_spice_subcircuits(draw: st.DrawFn) -> str:
    """Generates valid SPICE subcircuits that elaborate and export cleanly."""
    scale = draw(
        st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    resistance_one = draw(
        st.floats(min_value=100.0, max_value=1e5, allow_nan=False, allow_infinity=False)
    )
    resistance_two = draw(
        st.floats(min_value=100.0, max_value=1e5, allow_nan=False, allow_infinity=False)
    )
    capacitance = draw(
        st.floats(
            min_value=1e-12, max_value=1e-6, allow_nan=False, allow_infinity=False
        )
    )
    inductance = draw(
        st.floats(min_value=1e-9, max_value=1e-3, allow_nan=False, allow_infinity=False)
    )
    use_diode = draw(st.booleans())
    use_controlled = draw(st.booleans())
    use_conditional = draw(st.booleans())
    lines = [
        f".param factor = {scale:.4g}",
        ".func scaled(x) {x * factor}",
        ".model d_fast d is=1e-14 rs=10 cjo=1p",
        ".subckt stage inp out",
        f"R1 inp mid {{scaled({resistance_one:.4g})}}",
        f"R2 mid out {resistance_two:.4g}",
        f"R3 out 0 {resistance_two:.4g}",
        f"C1 mid 0 {capacitance:.4g}",
        f"L1 mid lnode {inductance:.4g}",
        f"R4 lnode 0 {resistance_one:.4g}",
    ]
    if use_diode:
        lines.append("D1 out 0 d_fast")
    if use_controlled:
        lines.append("G1 out 0 inp 0 1e-4")
    if use_conditional:
        lines.extend(
            [
                ".if (factor > 5)",
                "R5 inp 0 100k",
                ".else",
                "R5 inp 0 200k",
                ".endif",
            ]
        )
    lines.append(".ends")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers: 4. Archive builders and mutators (results, batches, shards)
# ---------------------------------------------------------------------------


def _write_npz(
    path: Path,
    arrays: ArchiveEntries,
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
    batch = _sample_batch()
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
        conditions=(batch,),
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
def scratch_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Shared temporary directory reused across fuzz iterations to avoid churn."""
    return tmp_path_factory.mktemp("fuzz_parsers")


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


def _drop_random_array(arrays: ArchiveEntries, data: st.DataObject) -> None:
    candidates = [key for key in arrays if key != "header"]
    if candidates:
        arrays.pop(data.draw(st.sampled_from(candidates)))


def _results_drop_array(
    arrays: ArchiveEntries, _header: JsonObject, data: st.DataObject
) -> None:
    _drop_random_array(arrays, data)


def _results_bad_version(
    _arrays: ArchiveEntries, header: JsonObject, data: st.DataObject
) -> None:
    current = header["version"]
    assert isinstance(current, int)
    header["version"] = data.draw(
        st.sampled_from([0, 2, current - 1, current + 1, str(current)])
    )


def _results_bad_format(
    _arrays: ArchiveEntries, header: JsonObject, _data: st.DataObject
) -> None:
    header["format"] = "other-format"


def _results_bad_json(
    arrays: ArchiveEntries, _header: JsonObject, _data: st.DataObject
) -> None:
    arrays["header"] = b"corrupt"


def _results_numeric_header(
    arrays: ArchiveEntries, _header: JsonObject, _data: st.DataObject
) -> None:
    arrays["header"] = np.array([1.0, 2.0])


def _results_pickled_header(
    arrays: ArchiveEntries, _header: JsonObject, _data: st.DataObject
) -> None:
    arrays["header"] = _pickled_npy_bytes()


def _results_corrupt_npy_array(
    arrays: ArchiveEntries, _header: JsonObject, _data: st.DataObject
) -> None:
    arrays["operating_point.values"] = b"\x93NUMPYcorrupt"


def _results_out_of_bounds_recording(
    _arrays: ArchiveEntries, header: JsonObject, _data: st.DataObject
) -> None:
    analyses = cast("dict[str, JsonObject]", header["analyses"])
    analyses["operating_point"]["recording"] = 99


def _results_negative_recording(
    _arrays: ArchiveEntries, header: JsonObject, data: st.DataObject
) -> None:
    analyses = cast("dict[str, JsonObject]", header["analyses"])
    analyses["operating_point"]["recording"] = data.draw(st.sampled_from([-1, -99]))


def _results_bad_transfer_shape(
    arrays: ArchiveEntries, _header: JsonObject, _data: st.DataObject
) -> None:
    arrays["transfer.transfer"] = np.array([1.0])


def _results_bad_transfer_scalar(
    arrays: ArchiveEntries, _header: JsonObject, _data: st.DataObject
) -> None:
    arrays["transfer.transfer"] = np.array(1.0)


def _results_float_logic_counts(
    arrays: ArchiveEntries, _header: JsonObject, _data: st.DataObject
) -> None:
    arrays["step.logic.counts"] = np.array([0.5, 1.5])


def _results_negative_logic_counts(
    arrays: ArchiveEntries, _header: JsonObject, _data: st.DataObject
) -> None:
    arrays["step.logic.counts"] = np.array([-1], dtype=np.int64)


def _results_mismatched_logic_counts(
    arrays: ArchiveEntries, _header: JsonObject, _data: st.DataObject
) -> None:
    arrays["step.logic.counts"] = np.array([5], dtype=np.int64)


def _results_sensitivity_parameter_mismatch(
    arrays: ArchiveEntries, _header: JsonObject, _data: st.DataObject
) -> None:
    arrays["sensitivity.parameters"] = np.array([1.0, 2.0, 3.0])


def _results_wrong_circuit_digest(
    _arrays: ArchiveEntries, header: JsonObject, _data: st.DataObject
) -> None:
    provenance = cast("JsonObject", header["provenance"])
    provenance["circuit"] = "0" * 64


_RESULTS_MUTATORS: Final[dict[str, ResultsMutator]] = {
    "drop_array": _results_drop_array,
    "bad_version": _results_bad_version,
    "bad_format": _results_bad_format,
    "bad_json": _results_bad_json,
    "numeric_header": _results_numeric_header,
    "pickled_header": _results_pickled_header,
    "corrupt_npy_array": _results_corrupt_npy_array,
    "out_of_bounds_recording": _results_out_of_bounds_recording,
    "negative_recording": _results_negative_recording,
    "bad_transfer_shape": _results_bad_transfer_shape,
    "bad_transfer_scalar": _results_bad_transfer_scalar,
    "float_logic_counts": _results_float_logic_counts,
    "negative_logic_counts": _results_negative_logic_counts,
    "mismatched_logic_counts": _results_mismatched_logic_counts,
    "sensitivity_parameter_mismatch": _results_sensitivity_parameter_mismatch,
    "wrong_circuit_digest": _results_wrong_circuit_digest,
}


def _batch_drop_array(
    arrays: ArchiveEntries, _header: JsonObject, data: st.DataObject
) -> bool:
    _drop_random_array(arrays, data)
    return False


def _batch_bad_version(
    _arrays: ArchiveEntries, header: JsonObject, data: st.DataObject
) -> bool:
    header["version"] = data.draw(st.sampled_from([0, 3, 5, "4"]))
    return False


def _batch_bad_format(
    _arrays: ArchiveEntries, header: JsonObject, _data: st.DataObject
) -> bool:
    header["format"] = "other"
    return False


def _batch_bad_fraction_zero_denominator(
    _arrays: ArchiveEntries, header: JsonObject, _data: st.DataObject
) -> bool:
    batches = cast("dict[str, JsonObject]", header["batches"])
    batches["monte_carlo"]["units"] = [["1/0", "0", "0", "0", "0"]]
    return False


def _batch_bad_fraction_text(
    _arrays: ArchiveEntries, header: JsonObject, _data: st.DataObject
) -> bool:
    batches = cast("dict[str, JsonObject]", header["batches"])
    batches["monte_carlo"]["units"] = [["not_a_fraction", "0", "0", "0", "0"]]
    return False


def _batch_pickled_header(
    arrays: ArchiveEntries, _header: JsonObject, _data: st.DataObject
) -> bool:
    arrays["header"] = _pickled_npy_bytes()
    return True


def _batch_corrupt_array(
    arrays: ArchiveEntries, _header: JsonObject, _data: st.DataObject
) -> bool:
    arrays["monte_carlo.values"] = b"corrupt"
    return False


def _batch_missing_trace_axis(
    _arrays: ArchiveEntries, header: JsonObject, _data: st.DataObject
) -> bool:
    batches = cast("dict[str, JsonObject]", header["batches"])
    batches["monte_carlo"]["traces"] = [
        {
            "analysis": "operating_point",
            "along": "point",
            "unknowns": [{"name": "out", "nature": "voltage"}],
        }
    ]
    return False


_BATCH_MUTATORS: Final[dict[str, BatchMutator]] = {
    "drop_array": _batch_drop_array,
    "bad_version": _batch_bad_version,
    "bad_format": _batch_bad_format,
    "bad_fraction_zero_denominator": _batch_bad_fraction_zero_denominator,
    "bad_fraction_text": _batch_bad_fraction_text,
    "pickled_header": _batch_pickled_header,
    "corrupt_array": _batch_corrupt_array,
    "missing_trace_axis": _batch_missing_trace_axis,
}


def _shard_drop_array(
    arrays: ArchiveEntries, _header: JsonObject, data: st.DataObject
) -> bool:
    _drop_random_array(arrays, data)
    return False


def _shard_bad_version(
    _arrays: ArchiveEntries, header: JsonObject, data: st.DataObject
) -> bool:
    header["version"] = data.draw(st.sampled_from([0, 2, 4, "3"]))
    return False


def _shard_bad_format(
    _arrays: ArchiveEntries, header: JsonObject, _data: st.DataObject
) -> bool:
    header["format"] = "other"
    return False


def _shard_start_greater_than_stop(
    _arrays: ArchiveEntries, header: JsonObject, _data: st.DataObject
) -> bool:
    header["start"], header["stop"] = 2, 1
    return False


def _shard_negative_start(
    _arrays: ArchiveEntries, header: JsonObject, _data: st.DataObject
) -> bool:
    header["start"] = -1
    return False


def _shard_stop_greater_than_samples(
    _arrays: ArchiveEntries, header: JsonObject, _data: st.DataObject
) -> bool:
    header["stop"], header["samples"] = 5, 2
    return False


def _shard_empty_conditions(
    _arrays: ArchiveEntries, header: JsonObject, _data: st.DataObject
) -> bool:
    header["conditions"] = []
    return False


def _shard_extra_condition(
    _arrays: ArchiveEntries, header: JsonObject, _data: st.DataObject
) -> bool:
    header["conditions"] = [None, None]
    return False


def _shard_bad_fraction_zero_denominator(
    _arrays: ArchiveEntries, header: JsonObject, _data: st.DataObject
) -> bool:
    header["requirements"] = [
        {
            "analysis": "operating_point",
            "description": "bound",
            "plain": None,
            "minimum": {"value": 1.0, "unit": ["1/0", "0", "0", "0", "0"]},
            "maximum": None,
            "margin": None,
        }
    ]
    return False


def _shard_bad_fraction_text(
    _arrays: ArchiveEntries, header: JsonObject, _data: st.DataObject
) -> bool:
    header["requirements"] = [
        {
            "analysis": "operating_point",
            "description": "bound",
            "plain": None,
            "minimum": {"value": 1.0, "unit": ["bad", "0", "0", "0", "0"]},
            "maximum": None,
            "margin": None,
        }
    ]
    return False


def _shard_empty_requirements(
    _arrays: ArchiveEntries, header: JsonObject, _data: st.DataObject
) -> bool:
    header["requirements"] = []
    return False


def _shard_bad_yield_target(
    _arrays: ArchiveEntries, header: JsonObject, data: st.DataObject
) -> bool:
    header["requirements"] = [
        {
            "analysis": "operating_point",
            "description": "bound",
            "plain": None,
            "minimum": {"value": 0.0, "unit": ["0", "0", "0", "0", "0"]},
            "maximum": None,
            "margin": None,
        }
    ]
    header["yield_target"] = data.draw(st.sampled_from([-0.5, 0.0, 2.0]))
    return False


def _shard_pickled_header(
    arrays: ArchiveEntries, _header: JsonObject, _data: st.DataObject
) -> bool:
    arrays["header"] = _pickled_npy_bytes()
    return True


def _shard_corrupt_array(
    arrays: ArchiveEntries, _header: JsonObject, _data: st.DataObject
) -> bool:
    arrays["0.values"] = b"corrupt"
    return False


def _shard_wrong_shape(
    arrays: ArchiveEntries, _header: JsonObject, _data: st.DataObject
) -> bool:
    arrays["0.measured"] = np.zeros((2, 3))
    return False


_SHARD_MUTATORS: Final[dict[str, ShardMutator]] = {
    "drop_array": _shard_drop_array,
    "bad_version": _shard_bad_version,
    "bad_format": _shard_bad_format,
    "start_greater_than_stop": _shard_start_greater_than_stop,
    "negative_start": _shard_negative_start,
    "stop_greater_than_samples": _shard_stop_greater_than_samples,
    "empty_conditions": _shard_empty_conditions,
    "extra_condition": _shard_extra_condition,
    "bad_fraction_zero_denominator": _shard_bad_fraction_zero_denominator,
    "bad_fraction_text": _shard_bad_fraction_text,
    "empty_requirements": _shard_empty_requirements,
    "bad_yield_target": _shard_bad_yield_target,
    "pickled_header": _shard_pickled_header,
    "corrupt_array": _shard_corrupt_array,
    "wrong_shape": _shard_wrong_shape,
}


# ---------------------------------------------------------------------------
# 1. Expression Parser Fuzz Tests
# ---------------------------------------------------------------------------

_USER_FUNCTIONS: Final[dict[str, expressions.Function]] = {
    "user_double": expressions.Function(("x",), expressions.parse("x + x")),
    "user_loop": expressions.Function(("x",), expressions.parse("user_loop(x)")),
    "user_zero": expressions.Function((), expressions.parse("42")),
}


@FUZZ
@example("")
@example("4.7k")
@example("1\u212a")
@example("1\u0660")
@example("1e500")
@example("1e-500")
@given(st.text(max_size=120))
def test_number_never_raises_on_arbitrary_text(text: str) -> None:
    first = expressions.number(text)
    second = expressions.number(text)
    assert first is None or isinstance(first, float)
    if first is None:
        assert second is None
    elif math.isnan(first):
        assert second is not None
        assert math.isnan(second)
    else:
        assert first == second


@FUZZ
@example("")
@example("(" * 120 + "1" + ")" * 120)
@example("-" * 1100 + "1")
@example("{1} + {2}")
@example("'1' + '2'")
@example("v()")
@example("i()")
@example("v(a, )")
@example("1e500")
@example("1/0")
@example("0/0")
@example("1" + "+1" * 500)
@given(st.text(max_size=200))
def test_spice_expression_arbitrary_text_only_raises_spice_error(text: str) -> None:
    try:
        parsed = expressions.parse(text)
    except SpiceError:
        return
    assert isinstance(expressions.names(parsed), set)
    assert expressions.inlined(parsed, {}) == parsed
    with contextlib.suppress(SpiceError):
        expressions.inlined(parsed, _USER_FUNCTIONS)
    expressions.substituted(parsed, expressions.Name("alpha"), expressions.Number(1.0))
    expressions.integral(parsed, expressions.Voltage("a", None))
    with contextlib.suppress(SpiceError):
        expressions.evaluate(parsed, _SPICE_PARAMETERS)
    with contextlib.suppress(SpiceError):
        expressions.translate(parsed, _SPICE_PARAMETERS, str.lower, str.lower)


@FUZZ
@given(_expression_token_soup())
def test_spice_expression_token_soup_only_raises_spice_error(text: str) -> None:
    try:
        parsed = expressions.parse(text)
    except SpiceError:
        return
    assert isinstance(expressions.names(parsed), set)
    with contextlib.suppress(SpiceError):
        expressions.inlined(parsed, _USER_FUNCTIONS)
    expressions.integral(parsed, expressions.Voltage("a", None))
    with contextlib.suppress(SpiceError):
        expressions.evaluate(parsed, _SPICE_PARAMETERS)
    with contextlib.suppress(SpiceError):
        expressions.translate(parsed, _SPICE_PARAMETERS, str.lower, str.lower)


@FUZZ
@given(_valid_spice_expressions(with_variables=False))
def test_valid_spice_expressions_parse_and_evaluate_deterministically(
    text: str,
) -> None:
    parsed = expressions.parse(text)
    assert expressions.parse(f"{{{text}}}") == parsed
    try:
        first = expressions.evaluate(parsed, _SPICE_PARAMETERS)
    except SpiceError:
        return
    second = expressions.evaluate(parsed, _SPICE_PARAMETERS)
    if math.isnan(first):
        assert math.isnan(second)
    else:
        assert first == second


@FUZZ
@given(_valid_spice_expressions(with_variables=True))
def test_valid_spice_expressions_with_variables_translate_to_core(text: str) -> None:
    parsed = expressions.parse(text)
    node_map = {
        "in": "n_in",
        "out": "n_out",
        "ref": "n_ref",
        "node_a": "n_a",
        "vc+": "n_vc_plus",
        "com": "n_com",
    }
    translated = expressions.translate(
        parsed,
        _SPICE_PARAMETERS,
        lambda node: node_map[node.casefold()],
        lambda source: source.casefold(),
    )
    error_message: str | None = None
    try:
        tree_json = _core.expression_tree(translated)
    except ValueError as error:
        error_message = str(error)
    else:
        assert json.loads(tree_json) is not None or tree_json == "null"
    if error_message is not None:
        assert "levels deep" in error_message


@FUZZ
@example("")
@example("1/0")
@example("0/0")
@example("table(1, 0, 0, 1, 1)")
@example("table(1, 0, 0, 0, 1)")
@example("table(0, 0/0, 1, 1, 2)")
@example("rounded_table(1, 0.25, 0, 0, 1, 1)")
@example("(" * 1000 + "1" + ")" * 1000)
@example("-" * 1000 + "1")
@given(st.text(max_size=150))
def test_core_expression_arbitrary_text_only_raises_value_error(text: str) -> None:
    with contextlib.suppress(ValueError):
        _core.constant(text)
    try:
        tree = _core.expression_tree(text)
    except ValueError:
        return
    assert json.loads(tree) is not None or tree == "null"


@FUZZ
@given(_expression_token_soup())
def test_core_expression_token_soup_only_raises_value_error(text: str) -> None:
    with contextlib.suppress(ValueError):
        _core.constant(text)
    try:
        tree = _core.expression_tree(text)
    except ValueError:
        return
    json.loads(tree)


@FUZZ
@given(_valid_core_expressions(with_variables=False))
def test_valid_core_expressions_evaluate_deterministically(text: str) -> None:
    first = _core.constant(text)
    second = _core.constant(text)
    if math.isnan(first):
        assert math.isnan(second)
    else:
        assert first == second
    tree = _core.expression_tree(text)
    json.loads(tree)


@FUZZ
@given(_valid_core_expressions(with_variables=True))
def test_valid_core_expressions_with_variables_produce_json_tree(text: str) -> None:
    tree = _core.expression_tree(text)
    assert tree
    json.loads(tree)


def test_expression_parser_regression_bugs() -> None:
    # Deeply nested parentheses and unaries raise SpiceError, not RecursionError.
    with pytest.raises(SpiceError, match="maximum nesting depth"):
        expressions.parse("(" * 120 + "1" + ")" * 120)
    with pytest.raises(SpiceError, match="maximum nesting depth"):
        expressions.parse("-" * 1100 + "1")
    with pytest.raises(SpiceError, match="maximum nesting depth"):
        expressions.parse("1 ? " * 120 + "1" + " : 2" * 120)
    with pytest.raises(SpiceError, match="maximum nesting depth"):
        expressions.parse("1||1&&1==1<1+1*(" * 105 + "1" + ")" * 105)

    # Multiple braced sub-expressions must not strip mismatched outer braces.
    assert expressions.evaluate(expressions.parse("{1} + {2}"), {}) == 3.0
    assert expressions.evaluate(expressions.parse("{ {1} + {2} }"), {}) == 3.0
    assert expressions.evaluate(expressions.parse('{"-1/2"}'), {}) == -0.5

    # Empty V() / I() node names are rejected at parse time.
    for invalid in ("v()", "V( )", "v(a, )", "v(, b)", "i()", "I( )"):
        with pytest.raises(SpiceError):
            expressions.parse(invalid)

    # Flat long binary chains never raise RecursionError in AST walkers, and evaluate
    # guards parenthesis depth before crossing into the Rust parser.
    long_sum = expressions.parse("1" + "+1" * 1500)
    second_sum = expressions.parse("1" + "+1" * 1500)
    assert long_sum == second_sum
    assert hash(long_sum) == hash(second_sum)
    assert expressions.names(long_sum) == set()
    assert expressions.inlined(long_sum, {}) is long_sum
    substituted = expressions.substituted(
        long_sum, expressions.Name("x"), expressions.Number(0.0)
    )
    assert substituted is long_sum
    assert substituted == second_sum
    empty_call = expressions.Call("f", ())
    assert empty_call == expressions.Call("f", ())
    assert empty_call != expressions.Call("g", ())
    assert hash(empty_call) == hash(expressions.Call("f", ()))
    translated = expressions.translate(long_sum, {}, str.lower, str.lower)
    assert nesting_depth(translated) == 1500
    with pytest.raises(SpiceError, match="parentheses deep"):
        expressions.evaluate(long_sum, {})

    # Zero-argument call AST inlines cleanly; empty parse syntax raises SpiceError.
    with pytest.raises(SpiceError):
        expressions.parse("user_zero()")
    assert (
        expressions.evaluate(
            expressions.inlined(expressions.Call("user_zero", ()), _USER_FUNCTIONS), {}
        )
        == 42.0
    )
    with pytest.raises(SpiceError):
        expressions.inlined(expressions.parse("user_loop(1)"), _USER_FUNCTIONS)

    # Exponential `.func` expansion stops at the node budget instead of hanging.
    chain: dict[str, expressions.Function] = {
        "f0": expressions.Function(("x",), expressions.parse("x + x"))
    }
    for index in range(1, 24):
        previous = f"f{index - 1}"
        chain[f"f{index}"] = expressions.Function(
            ("x",), expressions.parse(f"{previous}(x) + {previous}(x)")
        )
    with pytest.raises(SpiceError, match="inlining user functions exceeds"):
        expressions.inlined(expressions.parse("f23(1)"), chain)

    # High polynomial exponents return None instead of hanging or blowing recursion.
    variable = expressions.Voltage("a", None)
    assert expressions.integral(expressions.parse("v(a)^100000"), variable) is None
    assert expressions.integral(expressions.parse("v(a)^1e20"), variable) is None


def test_core_expression_depth_and_non_finite_table_regression_bugs() -> None:
    # Rust expression parser guards recursion depth and non-finite table points.
    with pytest.raises(ValueError, match="levels deep"):
        _core.constant("(" * 1000 + "1" + ")" * 1000)
    with pytest.raises(ValueError, match="levels deep"):
        _core.expression_tree("-" * 1000 + "v(a)")
    with pytest.raises(ValueError, match="levels deep"):
        _core.constant("2 ^ " * 300 + "2")
    with pytest.raises(ValueError, match="ascending abscissae"):
        _core.constant("table(0, 0/0, 1, 1, 2)")
    with pytest.raises(ValueError, match="ascending abscissae"):
        _core.expression_tree("table(v(a), 0, 1/0, 1, 2)")


# ---------------------------------------------------------------------------
# 2. Circuit Format JSON Parser Fuzz Tests
# ---------------------------------------------------------------------------


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
        _core.level(text, 0.0)
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
        {
            "name": "Xback",
            "of": {"component": "top"},
            "terminals": {"0": "gnd"},
        }
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
    elaborated = _core.Elaborated(raw, "{}")
    assert elaborated is not None


# ---------------------------------------------------------------------------
# 3. SPICE Netlist & Library Reader Fuzz Tests
# ---------------------------------------------------------------------------


@FUZZ
@example("")
@example("+ orphaned continuation")
@example(",")
@example("* comment\n; trailing\nR1 a b 1k ; comment\n+ tc1=1m")
@example("(((unclosed")
@example("'unclosed quote")
@given(st.text(max_size=250))
def test_netlist_statements_and_tokens_arbitrary_text(text: str) -> None:
    direct = netlist.tokens(text)
    assert all(token != "" for token in direct)
    try:
        found = netlist.statements(text, "fuzz.lib")
    except SpiceError:
        return
    for statement in found:
        words = netlist.tokens(statement.text)
        assert all(word != "" for word in words)


@FUZZ
@example(".model")
@example(".model m1")
@example(".model m1 vdmos pchan mfg=Vendor")
@example(".model m1 ako:missing")
@example(".model m1 nmos (version=)")
@given(st.text(max_size=150))
def test_models_read_arbitrary_text_only_raises_spice_error(text: str) -> None:
    statement = netlist.Statement("fuzz.lib", 1, text)
    try:
        card = models.read(statement, {})
    except SpiceError:
        return
    assert isinstance(card, models.ModelCard)


@FUZZ
@example("")
@example(",")
@example("+ bad")
@example(".subckt s a b\nR1 a b 1k")
@example(".ends")
@given(st.text(max_size=200))
def test_spice_library_arbitrary_text_only_raises_spice_error(
    scratch_directory: Path, text: str
) -> None:
    library = Library()
    try:
        library.read(text, "fuzz.lib")
    except SpiceError:
        return
    with contextlib.suppress(SpiceError):
        library.values()
    for subcircuit in library.subcircuits:
        assert isinstance(library.survey(subcircuit), tuple)
        with contextlib.suppress(SpiceError):
            library.component(subcircuit)
    for model_name in library.models:
        with contextlib.suppress(SpiceError):
            library.survey_model(model_name)
    path = scratch_directory / "arbitrary.lib"
    path.write_text(text, encoding="utf-8")
    included = Library()
    with contextlib.suppress(SpiceError):
        included.include(path)


@FUZZ
@given(_spice_netlists(), st.sampled_from([_core.REFERENCE_TEMPERATURE, 301.15]))
def test_spice_library_grammar_guided_netlists_only_raise_spice_error(
    scratch_directory: Path, text: str, temperature: float
) -> None:
    library = Library(temperature=temperature)
    try:
        library.read(text, "fuzz.lib")
    except SpiceError:
        return
    with contextlib.suppress(SpiceError):
        library.values()
    sample = Sample(seed=1, index=0)
    for subcircuit in library.subcircuits:
        uses = library.survey(subcircuit)
        assert isinstance(uses, tuple)
        with contextlib.suppress(SpiceError):
            library.component(subcircuit)
            library.drawn(subcircuit, sample, "X1")
            library.random_variables(subcircuit, "X1")
    for model_name in library.models:
        with contextlib.suppress(SpiceError):
            library.survey_model(model_name)
    path = scratch_directory / "deck.lib"
    path.write_text(text, encoding="utf-8")
    included = Library(temperature=temperature)
    with contextlib.suppress(SpiceError):
        included.include(path)


@FAST_FUZZ
@given(_valid_spice_subcircuits())
def test_valid_spice_subcircuits_round_trip_through_export(text: str) -> None:
    library = Library()
    library.read(text, "valid.lib")
    stage = library.component("stage")

    @component
    def harness(builder: Builder) -> None:
        ground = builder.port("ground", Role.GROUND)
        source = builder.place(devices.voltage_source(1 * V), name="V1")
        placed = builder.place(stage, name="X1")
        builder.connect(source.positive, placed.inp)
        builder.connect(source.negative, placed.ground, ground)

    built = circuit(harness())
    Simulation(built)
    exported = export.netlist(built)
    assert exported
    parsed_lines = netlist.statements("\n".join(exported), "exported.cir")
    assert parsed_lines


@pytest.mark.parametrize(
    ("text", "temperature", "match"),
    [
        # Self-recursive and mutually recursive subcircuits (was RecursionError).
        (".subckt s a b\nX1 a b s\n.ends", 300.15, "places itself"),
        (
            ".subckt s1 a b\nX1 a b s2\n.ends\n.subckt s2 a b\nX1 a b s1\n.ends",
            300.15,
            "places itself",
        ),
        # Geometric resistor effective width == 0 (was ZeroDivisionError).
        (
            (
                ".subckt s a b\n.model mr r rsh=10 narrow=5u short=5u\n"
                "R1 a b mr w=10u l=10u\n.ends"
            ),
            300.15,
            "effective width",
        ),
        # Geometric resistor flicker noise zero width (was ZeroDivisionError).
        (
            ".subckt s a b\n.model mr r rsh=10 kf=1e-12\nR1 a b mr w=0 l=10u\n.ends",
            300.15,
            "effective length and width",
        ),
        # Negative effective length with fractional exponent (was complex TypeError).
        (
            (
                ".subckt s a b\n.model mr r rsh=10 short=1u kf=1e-12 lf=0.5\n"
                "R1 a b mr w=1u l=1u\n.ends"
            ),
            300.15,
            "effective length and width",
        ),
        # Passive copies m=0 and m=-1 with flicker noise.
        (".subckt s a b\nR1 a b 1k m=0\n.ends", 300.15, "positive and finite"),
        (".subckt s a b\nL1 a b 1u m=0\n.ends", 300.15, "positive and finite"),
        (
            (
                ".subckt s a b\n.model mr r rsh=10 kf=1e-12 af=1.5\n"
                "R1 a b mr w=1u l=1u m=-1\n.ends"
            ),
            300.15,
            "positive and finite",
        ),
        # Subcircuit multiplier m=0 (was ZeroDivisionError).
        (
            (
                ".subckt inner a b\nR1 a b 1k\n.ends\n"
                ".subckt s a b\nX1 a b inner m=0\n.ends"
            ),
            300.15,
            "positive and finite",
        ),
        # Behavioral resistor temperature factor zero / overflow.
        (".subckt s a b\nR1 a b {v(a,b)} tc1=-1\n.ends", 301.15, "temperature factor"),
        (".subckt s a b\nR1 a b {v(a,b)} tc2=1\n.ends", 1e200, "temperature factor"),
        # Capacitor huge exponent integral (was RecursionError / hang).
        (
            ".subckt s a b\nC1 a b C={v(a,b)^500}\n.ends",
            300.15,
            "integral in closed form",
        ),
        # VBIC area=0, m=0, m=-1 (was ZeroDivisionError / complex TypeError).
        (
            ".subckt s a b c\n.model mq npn level=4\nQ1 a b c mq area=0\n.ends",
            300.15,
            "area times its copies",
        ),
        (
            (
                ".subckt s a b c\n.model mq npn level=4 rth=10\n"
                "Q1 a b c 0 t mq m=0\n.ends"
            ),
            300.15,
            "positive and finite",
        ),
        (
            (
                ".subckt s a b c\n.model mq npn level=4 kfn=1e-12 afn=1.5\n"
                "Q1 a b c mq m=-1\n.ends"
            ),
            300.15,
            "positive and finite",
        ),
        # Binned MOSFET nf=0 with wnflag=1 (was ZeroDivisionError).
        (
            (
                ".option wnflag=1\n.subckt s d g so b\n"
                ".model m.0 nmos level=8 lmin=0 lmax=1 wmin=0 wmax=1\n"
                "M1 d g so b m w=1u l=1u nf=0\n.ends"
            ),
            300.15,
            "fingers `NF`",
        ),
        # Scale option overflow on MOSFET ad scaling (was OverflowError).
        (
            (
                ".option scale=1e308\n.subckt s d g so b\n.model m nmos level=1\n"
                "M1 d g so b m w=1u l=1u ad=1\n.ends"
            ),
            300.15,
            "scale option overflows",
        ),
        # Missing expression after VALUE / VALUE = (was IndexError).
        (".subckt s a b c d\nE1 a b VALUE\n.ends", 300.15, "VALUE source"),
        (".subckt s a b c d\nE1 a b VALUE =\n.ends", 300.15, "VALUE source"),
        # POLY(0) infinite loop and POLY(9999999) allocation hang.
        (".subckt s a b\nE1 a b POLY(0) 1 2\n.ends", 300.15, "one or more controls"),
        (".subckt s a b\nE1 a b POLY(9999999) a b 1\n.ends", 300.15, "controls"),
        # Independent current source with missing nodes (was unpacking ValueError).
        (".subckt s a b\nI1\n.ends", 300.15, "joins two nodes"),
        (".subckt s a b\nI1 a\n.ends", 300.15, "joins two nodes"),
        # Empty drift keyword assignment tc1= (was IndexError).
        (".subckt s a b\nR1 a b 1k tc1=\n.ends", 300.15, "not assigned a value"),
        # Switch referencing non-switch model card (was KeyError).
        (
            ".subckt s a b c d\n.model md d\nS1 a b c d md\n.ends",
            300.15,
            "model of type `d`",
        ),
        (
            ".subckt s a b\n.model md d\nV1 a 0 1\nW1 a b V1 md\n.ends",
            300.15,
            "model of type `d`",
        ),
        # Transmission line F=0 (was ZeroDivisionError).
        (
            ".subckt s a b c d\nT1 a b c d Z0=50 F=0\n.ends",
            300.15,
            "frequency `F`",
        ),
        # Non-finite values and Rust device validation errors (were raw ValueError).
        (".subckt s a b\nR1 a b 1k temp=1e500\n.ends", 300.15, "expected value"),
        (".subckt s a b\nV1 a b PWL(1 0 0 1)\n.ends", 300.15, "must increase"),
        (".subckt s a b\nV1 a b PWL(1 0 0 1) DC 0\n.ends", 300.15, "must increase"),
        (
            ".subckt s a b c d\nE1 a b TABLE {V(c,d)} = (0,0) (0,1)\n.ends",
            300.15,
            "ascending abscissae",
        ),
        (
            (
                ".subckt s d g so b\n.model m nmos level=8 mobmod=1e308\n"
                "M1 d g so b m w=1u l=1u\n.ends"
            ),
            300.15,
            "u8",
        ),
        (
            ".subckt s a b\nL1 a b 1u\nL2 a b 1u\nK1 L1 L2 1e500\n.ends",
            300.15,
            "expected value",
        ),
    ],
)
def test_spice_component_regression_bugs(
    text: str, temperature: float, match: str
) -> None:
    library = Library(temperature=temperature)
    library.read(text, "regression.lib")
    target = "s1" if "s1" in library.subcircuits else "s"
    with pytest.raises(SpiceError, match=match):
        library.component(target)
    # survey() must report the SpiceError cleanly and never crash.
    uses = library.survey(target)
    assert uses


def test_spice_include_and_survey_model_regression_bugs(tmp_path: Path) -> None:
    # Comma-only statement inside include() previously crashed with IndexError.
    comma_file = tmp_path / "comma.lib"
    comma_file.write_text(",\n", encoding="utf-8")
    with pytest.raises(SpiceError):
        Library().include(comma_file)

    # Non-finite model level previously crashed survey_model with OverflowError.
    for text in (".model m nmos level=1/0\n", ".model m nmos level=0/0\n"):
        library = Library()
        library.read(text, "level.lib")
        found = library.survey_model("m")
        assert found.problem is not None

    # Non-finite spread in random_variables raises SpiceError, not ValidationError.
    library = Library()
    library.read(
        ".subckt s a b\n.param p=agauss(1, 1e308, 1e-10)\nR1 a b {p}\n.ends",
        "random_variables.lib",
    )
    with pytest.raises(SpiceError, match="non-finite"):
        library.random_variables("s", "X1")


# ---------------------------------------------------------------------------
# 4. Shard / Results / Batch Archive Fuzz Tests
# ---------------------------------------------------------------------------

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


@FUZZ
@given(
    st.lists(st.sampled_from(tuple(_RESULTS_MUTATORS)), min_size=1, max_size=3),
    st.data(),
)
def test_mutated_results_archives_only_raise_jolty_error(
    scratch_directory: Path,
    baseline_archives: _BaselineArchives,
    mutations: list[str],
    data: st.DataObject,
) -> None:
    arrays = baseline_archives.copy_for("results")
    header = json.loads(str(arrays["header"]))
    for kind in mutations:
        _RESULTS_MUTATORS[kind](arrays, header, data)
    header_entry = arrays.get("header")
    if (
        isinstance(header_entry, np.ndarray)
        and header_entry.dtype.kind == "U"
        and header_entry.ndim == 0
    ):
        arrays["header"] = np.array(json.dumps(header))
    path = scratch_directory / "mutated.results.npz"
    _write_npz(path, arrays)
    with contextlib.suppress(JoltyError):
        files.load_provenance(path)
    with contextlib.suppress(JoltyError):
        files.load_results(path)


@FUZZ
@given(
    st.lists(st.sampled_from(tuple(_BATCH_MUTATORS)), min_size=1, max_size=3),
    st.data(),
)
def test_mutated_batch_archives_only_raise_jolty_error(
    scratch_directory: Path,
    baseline_archives: _BaselineArchives,
    mutations: list[str],
    data: st.DataObject,
) -> None:
    arrays = baseline_archives.copy_for("batches")
    header = json.loads(str(arrays["header"]))
    header_rewritten = False
    for kind in mutations:
        rewritten = _BATCH_MUTATORS[kind](arrays, header, data)
        header_rewritten = header_rewritten or rewritten
    if not header_rewritten:
        arrays["header"] = np.array(json.dumps(header))
    path = scratch_directory / "mutated.batches.npz"
    _write_npz(path, arrays)
    with contextlib.suppress(JoltyError):
        files.load_batches(path)


@FUZZ
@given(
    st.lists(st.sampled_from(tuple(_SHARD_MUTATORS)), min_size=1, max_size=3),
    st.data(),
)
def test_mutated_shard_archives_and_merge_only_raise_jolty_error(
    scratch_directory: Path,
    baseline_archives: _BaselineArchives,
    mutations: list[str],
    data: st.DataObject,
) -> None:
    arrays = baseline_archives.copy_for("shard")
    header = json.loads(str(arrays["header"]))
    header_rewritten = False
    for kind in mutations:
        rewritten = _SHARD_MUTATORS[kind](arrays, header, data)
        header_rewritten = header_rewritten or rewritten
    if not header_rewritten:
        arrays["header"] = np.array(json.dumps(header))
    path = scratch_directory / "mutated.shard.npz"
    _write_npz(path, arrays)
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
    # Empty file previously raised EOFError in load_shard and others.
    empty = tmp_path / "empty.npz"
    empty.write_bytes(b"")
    for _, loader in _ARCHIVE_LOADERS:
        with pytest.raises(JoltyError):
            loader(empty)

    # Plain `.npy` file previously raised TypeError (ndarray lacks __exit__).
    plain_array = tmp_path / "plain.npy"
    np.save(plain_array, np.array([1.0]))
    for _, loader in _ARCHIVE_LOADERS:
        with pytest.raises(JoltyError):
            loader(plain_array)

    # Pickled object array inside zip previously raised ValueError.
    pickled = tmp_path / "pickled.npz"
    _write_npz(pickled, {"header": _pickled_npy_bytes()})
    for _, loader in _ARCHIVE_LOADERS:
        with pytest.raises(JoltyError):
            loader(pickled)

    # Corrupt DEFLATE stream previously raised BadZipFile / zlib.error and leaked
    # file descriptors.
    corrupt_deflate = tmp_path / "deflate.npz"
    with zipfile.ZipFile(corrupt_deflate, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("header.npy", b"hello world" * 100)
    raw = bytearray(corrupt_deflate.read_bytes())
    for offset in range(40, min(80, len(raw) - 40)):
        raw[offset] ^= 0xFF
    corrupt_deflate.write_bytes(bytes(raw))
    for _, loader in _ARCHIVE_LOADERS:
        with pytest.raises(JoltyError):
            loader(corrupt_deflate)

    # Non-.npy bytes inside a zip entry previously escaped as raw `bytes` and raised
    # AttributeError on `.ndim` or `.shape`.
    raw_header = tmp_path / "raw_header.npz"
    _write_npz(raw_header, {"header": b"corrupt"})
    for _, loader in _ARCHIVE_LOADERS:
        with pytest.raises(JoltyError):
            loader(raw_header)

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
    raw_values = tmp_path / "raw_values.shard.npz"
    _write_npz(raw_values, {**shard_arrays, "0.values": b"corrupt"})
    with pytest.raises(JoltyError):
        files.load_shard(raw_values)

    bad_shape = tmp_path / "bad_shape.shard.npz"
    _write_npz(bad_shape, {**shard_arrays, "0.measured": np.zeros((2, 3))})
    with pytest.raises(JoltyError):
        files.load_shard(bad_shape)

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
