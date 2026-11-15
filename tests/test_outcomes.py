# Copyright 2026 Maxence Faldor
"""Each outcome the extension returns holds exactly the keys its stub's TypedDict names.

The stub, ``_core.pyi``, types the dictionaries the extension builds, which no type
checker sees built: a key renamed in the Rust code and not in the stub would type a
value that is never there. The stub's TypedDicts exist for the type checker alone, so
the test runs the stub as Python to get them, runs every kind of analysis through
`Simulation.run`, and reads each outcome the extension returned against them.
"""

import runpy
import types
import typing
from collections.abc import Mapping
from pathlib import Path

import pytest

import jolty
from jolty import Builder, Role, Simulation, circuit, component, devices
from jolty.plan import (
    Ac,
    Analysis,
    DcSweep,
    Noise,
    OperatingPoint,
    OperatingPoints,
    PoleZero,
    Sensitivity,
    Stability,
    Transient,
    decade,
)
from jolty.units import MHz, V, kHz, kohm, nF, us

STUB: typing.Final = runpy.run_path(str(Path(jolty.__file__).with_name("_core.pyi")))
"""The stub's names, run as Python."""


def _hints(typed: object) -> dict[str, object]:
    """The keys of the TypedDict `typed` and their types, the stub's names resolved."""
    # The hints of a TypedDict are types, which `get_type_hints` gives as `Any`.
    return typing.cast("dict[str, object]", typing.get_type_hints(typed, globalns=STUB))


def _kind(typed: object) -> object:
    """The kind a TypedDict's ``kind`` key names, if it has one."""
    hint = _hints(typed).get("kind")
    return None if hint is None else typing.get_args(hint)[0]


def _mismatches(value: object, hint: object, where: str) -> list[str]:
    """Where `value`'s dictionaries hold other keys than `hint`'s TypedDicts name."""
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
    """Where `value` holds other keys than the TypedDict `typed` names, or within."""
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
    """A source, a probe and an RC low-pass."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1 * V), id="V1")
    probe = c.place(devices.voltage_source(0 * V), id="VP")
    resistor = c.place(devices.resistor(1 * kohm), id="R1")
    capacitor = c.place(devices.capacitor(1 * nF), id="C1")
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
    "sensitivity": Sensitivity(["out"]),
    "stability": Stability("VP", FREQUENCIES),
    "noise": Noise("out", FREQUENCIES, source="V1"),
    "pole_zero": PoleZero(source="V1", output="out"),
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
    assert _mismatches(outcome, STUB["Outcome"], name) == []


def test_every_kind_of_outcome_is_run() -> None:
    kinds = {_kind(member) for member in typing.get_args(STUB["Outcome"].__value__)}
    assert kinds == set(ANALYSES)
