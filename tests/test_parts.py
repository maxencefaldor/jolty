# Copyright 2026 Maxence Faldor
"""Real parts: their models, their numbers and their ratings, held to every result."""

import math
import re
from collections.abc import Callable

import pytest

from jolty import (
    Ac,
    Builder,
    Component,
    OperatingPoint,
    Role,
    Simulation,
    _core,
    component,
    devices,
    ir,
    parts,
)
from jolty.units import CURRENT, A, MHz, Quantity, V, W, celsius, fF, ohm, ps


def test_small_signal_parts_carry_their_numbers_and_ratings() -> None:
    npn, pnp, diode = parts.q2n3904(), parts.q2n3906(), parts.d1n4148()
    assert (npn.part, pnp.part, diode.part) == ("2N3904", "2N3906", "1N4148")
    assert npn.ratings[0] == ir.Rating(
        stress="voltage", terminals=("collector", "emitter"), maximum=40.0
    )
    assert pnp.ratings[3] == ir.Rating(
        stress="current", terminals=("collector",), minimum=-0.2
    )
    assert diode.ratings[0] == ir.Rating(
        stress="voltage", terminals=("anode", "cathode"), minimum=-100.0
    )
    junctions = [part.ratings[-1] for part in (npn, pnp, diode)]
    assert [(rating.stress, rating.maximum) for rating in junctions] == [
        ("temperature", celsius(150).value),
        ("temperature", celsius(150).value),
        ("temperature", celsius(175).value),
    ]
    assert all(rating.minimum == celsius(-55).value for rating in junctions)


def test_a_heated_part_joins_its_junction_to_a_thermal_net() -> None:
    assert parts.q2n3904(heated=True).terminals[-1] == "thermal"
    assert parts.q2n3906(heated=True).ratings == parts.q2n3906().ratings
    heated = parts.irfz44(heated=True)
    assert heated.ports == ("drain", "gate", "source", "thermal")
    assert heated.ratings == parts.irfz44().ratings


@pytest.mark.parametrize("make", [parts.irf511, parts.irf9230])
def test_a_power_mosfet_is_rated_as_its_channel_conducts(
    make: Callable[[], Component],
) -> None:
    built = make()
    sheet = getattr(parts, built.part or "")
    assert built.ports == ("drain", "gate", "source")
    drain_source, gate_source, drain, power, junction = built.ratings
    sign = 1.0 if sheet.channel == "n" else -1.0
    bound = sign * sheet.drain_source_voltage
    assert (drain_source.maximum, drain_source.minimum) == (
        (bound, None) if sign > 0 else (None, bound)
    )
    assert (gate_source.minimum, gate_source.maximum) == (-20.0, 20.0)
    assert drain.terminals == ("drain",)
    assert power.maximum == sheet.power
    assert junction.maximum == celsius(sheet.junction_temperature).value


@pytest.mark.parametrize("name", ["IRF511", "IRF9230", "IRFZ44", "MTP50P03HDL"])
def test_power_mosfets_meet_their_datasheets(name: str) -> None:
    sheet = getattr(parts, name)
    sign = 1.0 if sheet.channel == "n" else -1.0

    def bench(gate: float | None, drain: Quantity) -> Simulation:
        """The transistor, its gate at `gate` volts, its drain held or fed.

        The drain takes a voltage or a current, both signed as its channel conducts.
        Without a gate voltage, the gate is tied to the drain, which leaves the
        symmetric channel off whichever way the current flows.
        """

        @component
        def fixture(c: Builder) -> None:
            ground = c.port("ground", Role.GROUND)
            placed = c.place(parts.power_mosfet(sheet), name="X1")
            c.connect(placed.source, ground)
            if gate is None:
                c.connect(placed.gate, placed.drain)
            else:
                gate_source = c.place(
                    devices.voltage_source(sign * gate * V), name="VG"
                )
                c.connect(gate_source.positive, placed.gate, name="gate")
                c.connect(gate_source.negative, ground)
            if drain.dimension == CURRENT:
                fed = c.place(devices.current_source(sign * drain), name="ID")
                c.connect(fed.negative, placed.drain, name="drain")
                c.connect(fed.positive, ground)
            else:
                held = c.place(devices.voltage_source(sign * drain), name="VD")
                c.connect(held.positive, placed.drain, name="drain")
                c.connect(held.negative, ground)

        return Simulation(fixture())

    on = bench(sheet.on_gate, sheet.on_current * A).run(OperatingPoint())
    resistance = abs(on.voltage("drain")) / sheet.on_current
    # Smooth triode DC solve fits datasheet Rds(on) within 1e-6 (Newton convergence).
    assert resistance == pytest.approx(sheet.on_resistance, rel=1e-6)
    drop = bench(None, -sheet.diode_current * A).run(OperatingPoint())
    # Exponential body-diode drop fits datasheet Vsd within default reltol=1e-3.
    assert abs(drop.voltage("drain")) == pytest.approx(sheet.diode_drop, rel=1e-3)
    beta = parts.transconductance_parameter(sheet)
    overdrive = (2 * sheet.transconductance_current / beta) ** 0.5
    currents = [
        abs(
            bench(sheet.threshold + overdrive + step, 20 * V)
            .run(OperatingPoint())
            .current("VD")
        )
        for step in (-1e-3, 1e-3)
    ]
    # Central difference over 2 mV matches datasheet gm within 1e-3 (Newton reltol).
    assert (currents[1] - currents[0]) / 2e-3 >= sheet.transconductance * (1 - 1e-3)
    still = bench(0.0, sheet.capacitance_voltage * V)
    omega = 2 * math.pi * 1e6
    through_gate = still.run(Ac("VG", [1 * MHz]))
    through_drain = still.run(Ac("VD", [1 * MHz]))
    found = {
        "input": abs(through_gate.current("VG")[0]) / omega,
        "reverse": abs(through_drain.current("VG")[0]) / omega,
        "output": abs(through_drain.current("VD")[0]) / omega,
    }
    for kind, capacitance in found.items():
        stated = getattr(sheet, f"{kind}_capacitance")
        # 1 MHz AC extraction vs datasheet capacitance at default reltol=1e-3.
        assert capacitance == pytest.approx(stated, rel=1e-3), kind


@component
def switched(c: Builder, *, rated_power: float) -> None:
    """An IRF511 fully on, fed 6 A, beyond its 5.6, its power rated anew."""
    ground = c.port("ground", Role.GROUND)
    switch = c.place(parts.irf511().rated("power", rated_power * W), name="X1")
    gate = c.place(devices.voltage_source(10 * V), name="VG")
    fed = c.place(devices.current_source(6 * A), name="ID")
    c.connect(gate.positive, switch.gate, name="gate")
    c.connect(fed.negative, switch.drain, name="drain")
    c.connect(switch.source, gate.negative, fed.positive, ground)


def test_a_component_is_held_to_its_ratings_at_its_ports() -> None:
    simulation = Simulation(switched(rated_power=10))
    point = simulation.run(OperatingPoint(record=["X1", "X1/*"]))
    found = simulation.check_results({"op": point})
    assert [(d.rule, d.instances[0].path) for d in found] == [
        ("over-rating", "X1"),
        ("over-rating", "X1"),
    ]
    current = re.fullmatch(
        r"the current into `X1` at `drain` reaches (\S+) A at the operating point of "
        r"`op`, beyond its rating of 5\.6 A",
        found[0].message,
    )
    assert current is not None
    # Ideal 6 A source minus body-diode/gmin leakage (~1e-12 S * V) and rounding.
    assert float(current[1]) == pytest.approx(6.0, rel=1e-9)
    stated = re.search(r"dissipates reaches (\S+) W", found[1].message)
    assert stated is not None
    inside = sum(
        point.voltage(net) * point.current(device, terminal)
        for device, terminal, net in (
            ("X1/M1", "drain", "drain"),
            ("X1/M1", "gate", "gate"),
            ("X1/D1", "cathode", "drain"),
        )
    )
    # Tellegen port power equals sum of internal terminal powers within f64 rounding.
    assert float(stated[1]) == pytest.approx(inside, rel=1e-12)


@component
def wrapped(c: Builder) -> None:
    """A resistor of 1 Ω between two ports, rated to dissipate 0.1 W."""
    a, b = c.port("a"), c.port("b")
    inner = c.place(devices.resistor(1 * ohm).rated("power", 0.1 * W), name="R1")
    c.connect(inner.a, a)
    c.connect(inner.b, b)


@component
def across(c: Builder) -> None:
    """1 V across the wrapped resistor, rated as a whole to dissipate 2 W."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1 * V), name="V1")
    placed = c.place(wrapped().rated("power", 2 * W), name="X1")
    c.connect(source.positive, placed.a, name="top")
    c.connect(source.negative, placed.b, ground)


def test_a_substituted_component_is_held_to_its_own_ratings_alone() -> None:
    def raised(resistance: float | None) -> list[tuple[str, str]]:
        substitutes = (
            {} if resistance is None else {"X1": devices.resistor(resistance * ohm)}
        )
        simulation = Simulation(across(), substitutes=substitutes)
        found = simulation.check_results({"op": simulation.run(OperatingPoint())})
        return [(d.rule, d.instances[0].path) for d in found]

    assert raised(None) == [("over-rating", "X1/R1")]
    assert raised(10.0) == []
    assert raised(0.4) == [("over-rating", "X1")]


@component
def gated(c: Builder) -> None:
    """An electrical inverter on 1.8 V, its supply rated to 1.5 V."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(1.8 * V), name="VDD")
    gate = c.place(
        devices.logic_gate(
            "not",
            rise=500 * ps,
            transition=100 * ps,
            electrical=True,
            input_capacitance=2 * fF,
        ).rated("voltage", 1.5 * V, terminals=("supply", "ground")),
        name="U1",
    )
    driven = c.place(devices.voltage_source(0 * V), name="V1")
    c.connect(supply.positive, gate.supply, name="vdd")
    c.connect(driven.positive, gate.a, name="in")
    c.connect(gate.y, name="out")
    c.connect(supply.negative, driven.negative, gate.ground, ground)


def test_a_logic_device_is_held_to_its_ratings_at_its_terminals() -> None:
    simulation = Simulation(gated())
    found = simulation.check_results({"op": simulation.run(OperatingPoint())})
    assert [(d.rule, d.message) for d in found] == [
        (
            "over-rating",
            (
                "the voltage of `U1` at `supply` over `ground` reaches 1.8 V at the "
                "operating point of `op`, beyond its rating of 1.5 V"
            ),
        )
    ]


def test_thermal_voltage_uses_core_physical_constants() -> None:
    expected = _core.BOLTZMANN * _core.REFERENCE_TEMPERATURE / _core.ELEMENTARY_CHARGE
    assert parts.THERMAL_VOLTAGE == expected
