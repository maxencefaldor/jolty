# Copyright 2026 Maxence Faldor

import json
from pathlib import Path

import pytest

from jolty import (
    Builder,
    Component,
    JoltyError,
    Role,
    Simulation,
    circuit,
    component,
    devices,
)
from jolty.checks import Diagnostic, Instance, levels, report, rules
from jolty.cli import main
from jolty.files import save_circuit
from jolty.ir import Source
from jolty.settings import Level
from jolty.units import V, kohm, mA, mH, nF, ns, ps


def raised(top: Component) -> list[str]:
    """The rules the structural checks raise on the circuit with `top` at its top."""
    return [diagnostic.rule for diagnostic in Simulation(top).check()]


@component
def divider(c: Builder, *, floating: bool = False) -> None:
    """A source across two resistors, the lower one's end left floating if asked."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1 * V), id="V1")
    upper = c.place(devices.resistor(1 * kohm), id="R1")
    lower = c.place(devices.resistor(1 * kohm), id="R2")
    c.connect(source.positive, upper.a)
    c.connect(upper.b, lower.a, name="out")
    if floating:
        c.connect(source.negative, ground)
    else:
        c.connect(lower.b, source.negative, ground)


@component
def shorted(c: Builder, model: str) -> None:
    """A source, a resistor and a capacitor in parallel; `model` names one to short."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1 * V), id="V1")
    resistor = c.place(devices.resistor(1 * kohm), id="R1")
    capacitor = c.place(devices.capacitor(1 * nF), id="C1")
    c.connect(source.positive, resistor.a, capacitor.a, name="in")
    c.connect(resistor.b, ground)
    match model:
        case "source":
            c.connect(source.negative, source.positive)
            c.connect(capacitor.b, ground)
        case "capacitor":
            c.connect(source.negative, ground)
            c.connect(capacitor.b, capacitor.a)
        case _:
            c.connect(source.negative, capacitor.b, ground)


@component
def parallel_sources(c: Builder, *, inductor: bool) -> None:
    """A source in parallel with a second source, or with an inductor."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1 * V), id="V1")
    other = devices.inductor(1 * mH) if inductor else devices.voltage_source(2 * V)
    second = c.place(other, id="X")
    load = c.place(devices.resistor(1 * kohm), id="R1")
    first, last = ("a", "b") if inductor else ("positive", "negative")
    c.connect(source.positive, second[first], load.a, name="in")
    c.connect(source.negative, second[last], load.b, ground)


@component
def coupled(c: Builder, *, grounded: bool) -> None:
    """A source coupled into a resistor, grounded or through a second capacitor."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1 * V), id="V1")
    coupling = c.place(devices.capacitor(1 * nF), id="C1")
    load = c.place(devices.resistor(1 * kohm), id="R1")
    c.connect(source.positive, coupling.a, name="in")
    c.connect(coupling.b, load.a, name="mid")
    c.connect(source.negative, ground)
    if grounded:
        c.connect(load.b, ground)
    else:
        other = c.place(devices.capacitor(1 * nF), id="C2")
        c.connect(load.b, other.a, name="out")
        c.connect(other.b, ground)


@component
def charged(c: Builder, *, load: bool) -> None:
    """A current source into a capacitor, loaded by a resistor if asked."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.current_source(1 * mA), id="I1")
    capacitor = c.place(devices.capacitor(1 * nF), id="C1")
    c.connect(source.negative, capacitor.a, name="top")
    c.connect(source.positive, capacitor.b, ground)
    if load:
        resistor = c.place(devices.resistor(1 * kohm), id="R1")
        c.connect(resistor.a, capacitor.a)
        c.connect(resistor.b, ground)


@component
def latch(c: Builder, *, sources: int) -> None:
    """A NAND of the net ``a`` and the inverse of its own output.

    `sources` logic sources drive ``a``; with none, a second such NAND reads it too.
    """
    c.port("ground", Role.GROUND)
    a = c.net("a")
    for index in range(sources):
        source = c.place(devices.logic_source(0, transition=100 * ps), id=f"A{index}")
        c.connect(source.output, a)
    for index in range(1 if sources else 2):
        nand = c.place(
            devices.logic_gate("nand", rise=1 * ns, transition=100 * ps), id=f"U{index}"
        )
        inverter = c.place(
            devices.logic_gate("not", rise=1 * ns, transition=100 * ps), id=f"I{index}"
        )
        c.connect(a, nand.a)
        c.connect(nand.y, inverter.a)
        c.connect(inverter.y, nand.b)


def test_a_sound_circuit_raises_nothing() -> None:
    assert raised(divider()) == []


def test_unconnected_terminal() -> None:
    assert raised(divider(floating=True)) == ["unconnected-terminal"]
    assert raised(divider(floating=False)) == []


def test_shorted_source_and_shorted_device() -> None:
    assert raised(shorted("source")) == ["shorted-source"]
    assert raised(shorted("capacitor")) == ["shorted-device"]
    assert raised(shorted("neither")) == []


def test_voltage_loop() -> None:
    assert raised(parallel_sources(inductor=False)) == ["voltage-loop"]
    assert raised(parallel_sources(inductor=True)) == ["voltage-loop"]
    assert raised(divider()) == []


def test_no_dc_path() -> None:
    assert raised(coupled(grounded=True)) == []
    (diagnostic,) = Simulation(coupled(grounded=False)).check()
    assert diagnostic.rule == "no-dc-path"
    assert diagnostic.message == "`mid` and `out` have no DC path to `ground`"
    assert diagnostic.nets == ("mid", "out")
    assert [instance.path for instance in diagnostic.instances] == ["C1", "R1", "C2"]
    source = diagnostic.instances[0].source
    assert source is not None
    assert source.file == "tests/test_checks.py"


def test_current_cutset() -> None:
    assert raised(charged(load=False)) == ["current-cutset"]
    assert raised(charged(load=True)) == []


def test_driver_conflict_and_floating_input() -> None:
    assert raised(latch(sources=1)) == []
    assert raised(latch(sources=2)) == ["driver-conflict"]
    (diagnostic,) = Simulation(latch(sources=0)).check()
    assert diagnostic.rule == "floating-input"
    assert (
        diagnostic.message
        == "nothing drives the logic net `a`, which `U0` and `U1` read"
    )


def test_every_rule_has_a_case_that_raises_it() -> None:
    cases = [
        divider(floating=True),
        shorted("source"),
        shorted("capacitor"),
        parallel_sources(inductor=False),
        coupled(grounded=False),
        charged(load=False),
        latch(sources=2),
        latch(sources=0),
    ]
    found = {rule for case in cases for rule in raised(case)}
    assert found == set(rules())


def test_levels_apply_settings_then_options() -> None:
    defaults = levels({})
    assert defaults["no-dc-path"] is Level.ERROR
    assert defaults["shorted-device"] is Level.WARN
    configured = {"all": Level.WARN, "voltage-loop": Level.IGNORE}
    chosen = levels(configured, error=["shorted-device"])
    assert chosen["no-dc-path"] is Level.WARN
    assert chosen["voltage-loop"] is Level.IGNORE
    assert chosen["shorted-device"] is Level.ERROR
    assert set(levels({}, ignore=["all"]).values()) == {Level.IGNORE}


def test_levels_name_only_rules_and_one_level_each() -> None:
    with pytest.raises(JoltyError, match="the options name no-such-rule, which no"):
        levels({}, warn=["no-such-rule"])
    with pytest.raises(JoltyError, match="the settings name typo, which no rule is"):
        levels({"typo": Level.WARN})
    with pytest.raises(JoltyError, match="--error and --ignore both name no-dc-path"):
        levels({}, error=["no-dc-path"], ignore=["no-dc-path"])


def finding(rule: str, level: Level = Level.ERROR) -> Diagnostic:
    return Diagnostic(
        rule=rule,
        level=level,
        message=f"a fact of {rule}",
        instances=(
            Instance(path="X1/C1", source=Source(file="f.py", line=3, function="rc")),
        ),
        nets=(),
    )


def test_a_report_lists_errors_first_up_to_its_limit() -> None:
    found = [
        finding("shorted-device", Level.WARN),
        finding("no-dc-path"),
        finding("voltage-loop"),
    ]
    summary = report(found, levels({}), limit=2)
    assert [diagnostic.rule for diagnostic in summary.diagnostics] == [
        "no-dc-path",
        "voltage-loop",
    ]
    assert (summary.errors, summary.warnings, summary.unlisted) == (2, 1, 1)
    assert not summary.ok
    assert summary.describe() == (
        "error[no-dc-path]: a fact of no-dc-path\n"
        "  = X1/C1, f.py:3 in rc()\n\n"
        "error[voltage-loop]: a fact of voltage-loop\n"
        "  = X1/C1, f.py:3 in rc()\n\n"
        "2 errors, 1 warning (1 not listed)"
    )
    quiet = report(found, levels({}, ignore=["all"]), limit=20)
    assert quiet.ok
    assert quiet.describe() == "0 errors, 0 warnings"


def run(*arguments: str) -> int:
    """Runs the command line and returns its exit status."""
    try:
        main(arguments)
    except SystemExit as exit_:
        return exit_.code if isinstance(exit_.code, int) else 0
    return 0


def test_check_reads_a_saved_circuit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "coupled.jolty.json"
    save_circuit(path, circuit(coupled(grounded=False)))
    assert run("check", str(path)) == 1
    assert capsys.readouterr().out.startswith("error[no-dc-path]: `mid` and `out`")
    assert run("check", str(path), "--warn", "no-dc-path") == 0
    assert capsys.readouterr().out.startswith("warn[no-dc-path]")
    assert run("check", str(path), "--output-format", "json", "--ignore", "all") == 0
    result = json.loads(capsys.readouterr().out)
    assert result["report"]["errors"] == 0
    assert run("check", str(tmp_path / "missing.jolty.json")) == 2


def test_explain_states_each_rule_and_its_level(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.jolty.rules]\nno-dc-path = "warn"\n'
    )
    assert run("explain", "no-dc-path") == 0
    out = capsys.readouterr().out
    assert out.startswith("no-dc-path (warn, default error): A group of nodes")
    assert run("explain") == 0
    assert capsys.readouterr().out.count("\n\n") == len(rules()) - 1
    assert run("explain", "bogus") == 2
