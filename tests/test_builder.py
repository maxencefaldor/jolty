# Copyright 2026 Maxence Faldor

from pathlib import Path

import pytest

from jolty import Builder, BuildError, Role, circuit, component, devices, ir
from jolty.units import Quantity, V, kohm, nF


@component
def low_pass(c: Builder, resistance: Quantity, capacitance: Quantity) -> None:
    source = c.port("input")
    output = c.port("output")
    ground = c.port("ground", Role.GROUND)
    resistor = c.place(devices.resistor(resistance), id="R")
    capacitor = c.place(devices.capacitor(capacitance), id="C")
    c.connect(source, resistor.a)
    c.connect(resistor.b, capacitor.a, output)
    c.connect(capacitor.b, ground)


def line_of(text: str) -> int:
    """The number of the line of this file that holds `text`."""
    lines = Path(__file__).read_text(encoding="utf-8").splitlines()
    return next(number for number, line in enumerate(lines, 1) if text in line)


@component
def two_stages(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1 * V), id="V1")
    first = c.place(low_pass(1 * kohm, 1 * nF), id="X1")
    second = c.place(low_pass(1 * kohm, 1 * nF), id="X2")
    c.connect(source.positive, first.input)
    c.connect(first.output, second.input, name="mid")
    c.connect(source.negative, first.ground, second.ground, ground)


def test_a_component_declares_ports_and_connects_terminals_to_nets() -> None:
    built = low_pass(1 * kohm, 1 * nF)
    assert built.name == "low_pass[resistance=1kΩ,capacitance=1nF]"
    definition = built.definition
    assert [(port.name, port.role) for port in definition.ports] == [
        ("input", Role.SIGNAL),
        ("output", Role.SIGNAL),
        ("ground", Role.GROUND),
    ]
    resistor, capacitor = definition.instances
    assert resistor.of == ir.DeviceOf(device="resistor")
    assert resistor.parameters == {"resistance": 1000.0}
    assert resistor.terminals == {"a": "input", "b": "output"}
    assert capacitor.terminals == {"a": "output", "b": "ground"}


def test_each_instance_records_the_line_that_placed_it() -> None:
    resistor, capacitor = low_pass(1 * kohm, 1 * nF).definition.instances
    assert resistor.source == ir.Source(
        file="tests/test_builder.py", line=line_of('id="R")'), function="low_pass"
    )
    assert capacitor.source is not None
    assert capacitor.source.line == line_of('id="C")')


def test_the_same_arguments_build_the_same_component_once() -> None:
    assert low_pass(1 * kohm, 1 * nF) is low_pass(1 * kohm, 1 * nF)
    assert low_pass(1 * kohm, 1 * nF) is not low_pass(2 * kohm, 1 * nF)


@component
def amplifier(c: Builder, transistor: devices.Device) -> None:
    c.port("ground", Role.GROUND)
    c.place(transistor, id="Q1")


def test_arguments_name_components_exactly_and_without_whitespace() -> None:
    near = low_pass(1.00001 * kohm, 4.7 * nF)
    assert near.name == "low_pass[resistance=1.00001kΩ,capacitance=4.7nF]"
    assert near is not low_pass(1 * kohm, 4.7 * nF)
    npn = amplifier(devices.bipolar_transistor(polarity="npn"))
    pnp = amplifier(devices.bipolar_transistor(polarity="pnp"))
    assert npn.name.startswith("amplifier[transistor=bipolar_transistor~")
    assert npn.name != pnp.name
    assert not any(character.isspace() for character in npn.name)
    assert circuit(npn).top == npn.name


def test_a_circuit_holds_each_component_once() -> None:
    built = circuit(two_stages())
    assert built.top == "two_stages"
    assert sorted(built.components) == [
        "low_pass[resistance=1kΩ,capacitance=1nF]",
        "two_stages",
    ]
    top = built.components["two_stages"]
    assert [instance.id for instance in top.instances] == ["V1", "X1", "X2"]
    assert top.instances[1].terminals == {
        "input": "V1.positive",
        "output": "mid",
        "ground": "ground",
    }


def test_unnamed_nets_take_the_name_of_their_first_terminal() -> None:
    @component
    def chain(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        first = c.place(devices.resistor(1 * kohm), id="R1")
        second = c.place(devices.resistor(1 * kohm), id="R2")
        c.connect(first.b, second.a)
        c.connect(first.a, second.b, ground)

    first, second = chain().definition.instances
    assert first.terminals == {"a": "ground", "b": "R1.b"}
    assert second.terminals == {"a": "R1.b", "b": "ground"}


def test_joining_two_named_nets_is_a_build_error() -> None:
    @component
    def shorted(c: Builder) -> None:
        supply = c.port("supply", Role.SUPPLY)
        ground = c.port("ground", Role.GROUND)
        resistor = c.place(devices.resistor(1 * kohm), id="R1")
        c.connect(resistor.a, supply)
        c.connect(resistor.a, ground)

    with pytest.raises(BuildError, match="joins the nets supply and ground"):
        shorted()


def test_names_ids_and_terminals_are_checked() -> None:
    builder = Builder("checked")
    resistor = builder.place(devices.resistor(1 * kohm), id="R1")
    with pytest.raises(BuildError, match="already places an instance named 'R1'"):
        builder.place(devices.resistor(1 * kohm), id="R1")
    with pytest.raises(BuildError, match="an instance id uses letters"):
        builder.place(devices.resistor(1 * kohm), id="R/2")
    builder.net("mid")
    with pytest.raises(BuildError, match="already declares a net or port named 'mid'"):
        builder.port("mid")
    with pytest.raises(AttributeError, match="R1 has no terminal 'c'; it has a, b"):
        _ = resistor.c
    other = Builder("other").place(devices.resistor(1 * kohm), id="R9")
    with pytest.raises(BuildError, match=r"R9\.a does not belong to checked"):
        builder.connect(resistor.a, other.a)


def test_two_different_components_cannot_share_a_name() -> None:
    first, second = Builder("twin").finish(), Builder("twin").finish()
    parent = Builder("parent")
    parent.place(first, id="A")
    with pytest.raises(BuildError, match="two different components are named 'twin'"):
        parent.place(second, id="B")
    left, right = Builder("left"), Builder("right")
    left.place(first, id="A")
    right.place(second, id="B")
    top = Builder("top")
    top.place(left.finish(), id="L")
    top.place(right.finish(), id="R")
    with pytest.raises(BuildError, match="two different components are named 'twin'"):
        circuit(top.finish())
