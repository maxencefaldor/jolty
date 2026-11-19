# Copyright 2026 Maxence Faldor
"""Consistency tests verifying that `_core.pyi` accurately describes the Rust extension.

`ty` reads the stub rather than the compiled extension, so these checks keep both in
sync. Every runtime symbol, class method, and `@final` annotation must match between
`_core.pyi` and `_core`. The stub also types every dictionary returned by the Rust
extension so renamed keys cannot drift silently. Tests execute the stub as Python to
inspect its `TypedDict` definitions, run every analysis kind via `Simulation.run`, and
validate every returned dictionary against those annotations.
"""

import ast
import runpy
import types
import typing
from collections.abc import Mapping
from pathlib import Path

import pytest

import jolty
from jolty import Builder, Role, Simulation, _core, circuit, component, devices
from jolty.plan import (
    Ac,
    Analysis,
    DcSweep,
    Mismatch,
    Noise,
    OperatingPoint,
    OperatingPoints,
    PeriodicSteadyState,
    PoleZero,
    Sensitivity,
    Stability,
    TransferFunction,
    Transient,
    decade,
)
from jolty.units import MHz, V, kHz, kohm, nF, us

STUB_PATH: typing.Final = Path(jolty.__file__).with_name("_core.pyi")
"""The extension's stub."""

STUB: typing.Final = runpy.run_path(str(STUB_PATH))
"""The stub's names, run as Python."""

TREE: typing.Final = ast.parse(STUB_PATH.read_text(encoding="utf-8"))
"""The stub's statements."""


def _type_only(statement: ast.stmt) -> bool:
    """Returns whether `statement` declares a typing-only construct absent at runtime.

    `TypedDict` classes describe extension-built dictionaries, and type aliases name
    static types.
    """
    if isinstance(statement, ast.TypeAlias):
        return True
    return isinstance(statement, ast.ClassDef) and typing.is_typeddict(
        STUB[statement.name]
    )


def _declared(statements: list[ast.stmt]) -> dict[str, ast.stmt]:
    """Maps declared names in `statements` to AST nodes, skipping typing-only items."""
    found: dict[str, ast.stmt] = {}
    for statement in statements:
        if _type_only(statement):
            continue
        if isinstance(statement, ast.ClassDef | ast.FunctionDef):
            found[statement.name] = statement
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            found[statement.target.id] = statement
    return found


def _public(names: typing.Iterable[str]) -> set[str]:
    """Filters `names` to public attributes plus `__new__` and `__version__`."""
    return {
        name
        for name in names
        if not name.startswith("_") or name in {"__new__", "__version__"}
    }


def test_the_stub_declares_every_name_of_the_module_and_no_other() -> None:
    declared = set(_declared(TREE.body))
    assert declared == _public(vars(_core))


@pytest.mark.parametrize(
    "name",
    [s.name for s in TREE.body if isinstance(s, ast.ClassDef) and not _type_only(s)],
)
def test_each_class_of_the_stub_is_the_module_s(name: str) -> None:
    statement = _declared(TREE.body)[name]
    assert isinstance(statement, ast.ClassDef)
    runtime = getattr(_core, name)
    methods = _public(_declared(statement.body)) - {"__new__"}
    # Every stubbed attribute exists on the runtime class (own or inherited, such as
    # exception `args`), and every own runtime attribute appears in the stub.
    assert methods <= _public(dir(runtime)), name
    assert _public(vars(runtime)) - {"__new__"} <= methods, name
    final = any(
        isinstance(decorator, ast.Name) and decorator.id == "final"
        for decorator in statement.decorator_list
    )
    try:
        type("Derived", (runtime,), {})
    except TypeError:
        subclassed = False
    else:
        subclassed = True
    assert final != subclassed, f"{name} is final in the stub: {final}"


def _hints(typed: object) -> dict[str, object]:
    """Returns resolved field type hints for `TypedDict` class `typed`."""
    # The hints of a TypedDict are types, which `get_type_hints` gives as `Any`.
    return typing.cast("dict[str, object]", typing.get_type_hints(typed, globalns=STUB))


def _kind(typed: object) -> object:
    """Returns the literal `kind` discriminator of `typed`, if declared."""
    hint = _hints(typed).get("kind")
    return None if hint is None else typing.get_args(hint)[0]


def _mismatches(value: object, hint: object, where: str) -> list[str]:
    """Collects key mismatches between runtime `value` and `TypedDict` `hint`."""
    if typing.is_typeddict(hint):
        return _keys(value, hint, where)
    if isinstance(hint, typing.TypeAliasType):
        return _mismatches(value, hint.__value__, where)
    origin, arguments = typing.get_origin(hint), typing.get_args(hint)
    if origin in {typing.Union, types.UnionType} and isinstance(value, Mapping):
        typed = [argument for argument in arguments if typing.is_typeddict(argument)]
        kinds = {_kind(argument): argument for argument in typed}
        chosen = kinds.get(value.get("kind"), typed[0]) if typed else None
        return [] if chosen is None else _keys(value, chosen, where)
    if origin is list and isinstance(value, list):
        return [
            problem
            for index, each in enumerate(value)
            for problem in _mismatches(each, arguments[0], f"{where}[{index}]")
        ]
    return []


def _keys(value: object, typed: object, where: str) -> list[str]:
    """Collects missing or extra keys in `value` relative to `TypedDict` `typed`."""
    if not isinstance(value, Mapping):
        return [f"{where} is a {type(value).__name__}, not a dictionary"]
    named = _hints(typed)
    found: list[str] = []
    if set(value) != set(named):
        extra, missing = (
            sorted(set(value) - set(named)),
            sorted(set(named) - set(value)),
        )
        found.append(f"{where}: {extra} not in the stub, {missing} not returned")
    for key in named.keys() & value.keys():
        found += _mismatches(value[key], named[key], f"{where}.{key}")
    return found


@component
def _probed(c: Builder) -> None:
    """Builds a voltage source, current probe, and RC low-pass filter."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1 * V), name="V1")
    probe = c.place(devices.voltage_source(0 * V), name="VP")
    resistor = c.place(devices.resistor(1 * kohm), name="R1")
    capacitor = c.place(devices.capacitor(1 * nF), name="C1")
    c.connect(source.positive, probe.positive)
    c.connect(probe.negative, resistor.a)
    c.connect(resistor.b, capacitor.a, c.net("out"))
    c.connect(source.negative, capacitor.b, ground)


FREQUENCIES: typing.Final = decade(1 * kHz, 10 * MHz, 2)

ANALYSES: typing.Final[Mapping[str, Analysis]] = {
    "operating_point": OperatingPoint(),
    "operating_points": OperatingPoints(),
    "dc_sweep": DcSweep("V1", [0 * V, 1 * V]),
    "ac": Ac("V1", FREQUENCIES),
    "transient": Transient(stop=5 * us),
    "periodic_steady_state": PeriodicSteadyState(period=5 * us),
    "sensitivity": Sensitivity(["out"]),
    "mismatch": Mismatch(["out"], of=Ac("V1", FREQUENCIES)),
    "stability": Stability("VP", FREQUENCIES),
    "noise": Noise("out", FREQUENCIES, source="V1"),
    "pole_zero": PoleZero(source="V1", output="out"),
    "transfer_function": TransferFunction("V1", "out"),
}


@pytest.mark.parametrize("name", list(ANALYSES))
def test_each_outcome_holds_the_keys_its_stub_names(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    returned: list[object] = []
    resulted = vars(Simulation)["_result"]

    def capturing(self: Simulation, outcome: object, recording: object) -> object:
        returned.append(outcome)
        return resulted(self, outcome, recording)

    monkeypatch.setattr(Simulation, "_result", capturing)
    Simulation(circuit(_probed())).run(ANALYSES[name])
    # The analysis's own outcome comes first, before any it holds within it.
    outcome = returned[0]
    assert isinstance(outcome, Mapping)
    assert outcome["kind"] == name
    assert _mismatches(outcome, STUB["AnalysisResult"], name) == []


def test_every_analysis_is_run() -> None:
    analyses = set(typing.get_args(Analysis.__value__))
    assert {type(analysis) for analysis in ANALYSES.values()} == analyses


def test_every_kind_of_outcome_is_run() -> None:
    kinds = {
        _kind(member) for member in typing.get_args(STUB["AnalysisResult"].__value__)
    }
    assert kinds == set(ANALYSES)
