# Copyright 2026 Maxence Faldor
"""Ratings: the limits a device's maker states.

A simulation holds them to every operating point and transient it solves.
"""

import re
from pathlib import Path

import pytest

from jolty import (
    Builder,
    Component,
    DcSweep,
    OperatingPoint,
    Role,
    Simulation,
    Transient,
    circuit,
    component,
    devices,
    ir,
    results,
)
from jolty.errors import DimensionError
from jolty.files import load_circuit, save_circuit
from jolty.units import V, W, mA, ohm, s, uF, us
from jolty.waveforms import Pulse


def test_a_device_states_its_ratings() -> None:
    resistor = devices.resistor(100 * ohm).rated("power", 0.25 * W)
    assert resistor.ratings == (ir.Rating(stress="power", maximum=0.25),)
    capacitor = devices.capacitor(1 * uF).rated("voltage", 16 * V, minimum=-1 * V)
    (rating,) = capacitor.ratings
    assert (rating.terminals, rating.minimum, rating.maximum) == (
        ("a", "b"),
        -1.0,
        16.0,
    )
    forward = devices.resistor(1 * ohm).rated("current", 10 * mA).ratings[0]
    assert (forward.terminals, forward.minimum, forward.maximum) == (("a",), None, 0.01)
    reverse = devices.diode().rated("voltage", minimum=-100 * V).ratings[0]
    assert (reverse.terminals, reverse.minimum, reverse.maximum) == (
        ("anode", "cathode"),
        -100.0,
        None,
    )
    transistor = devices.bipolar_transistor(polarity="npn").rated(
        "voltage", 40 * V, terminals=("collector", "emitter")
    )
    assert transistor.ratings[0].terminals == ("collector", "emitter")


def test_a_rating_names_the_devices_terminals_in_its_unit() -> None:
    resistor = devices.resistor(100 * ohm)
    with pytest.raises(
        ValueError, match="names collector, which is not among the terminals a, b"
    ):
        resistor.rated("voltage", 1 * V, terminals=("collector", "a"))
    with pytest.raises(ValueError, match="names 2 distinct terminals, not"):
        resistor.rated("voltage", 1 * V, terminals=("a",))
    with pytest.raises(DimensionError):
        resistor.rated("power", 1 * V)
    with pytest.raises(ValueError, match="a power rating states only a maximum"):
        resistor.rated("power", 0 * W)
    with pytest.raises(ValueError, match="a power rating states only a maximum"):
        resistor.rated("power", 1 * W, minimum=0 * W)
    with pytest.raises(ValueError, match="states a maximum, a minimum or both"):
        resistor.rated("voltage")
    with pytest.raises(ValueError, match="exceeds its maximum"):
        resistor.rated("voltage", 1 * V, minimum=2 * V)


@component
def supply(c: Builder, *, rated_power: float, rated_voltage: float) -> None:
    """10 V that pulses to 20 V for 2 µs, across 100 Ω and 1 µF, each rated."""
    ground = c.port("ground", Role.GROUND)
    pulse = Pulse(
        initial=10 * V,
        pulsed=20 * V,
        delay=1 * us,
        rise=1 * us,
        fall=1 * us,
        width=2 * us,
    )
    source = c.place(devices.voltage_source(pulse), name="V1")
    load = c.place(
        devices.resistor(100 * ohm).rated("power", rated_power * W), name="R1"
    )
    store = c.place(
        devices.capacitor(1 * uF).rated("voltage", rated_voltage * V), name="C1"
    )
    c.connect(source.positive, load.a, store.a, name="top")
    c.connect(source.negative, load.b, store.b, ground)


def raised(built: Component) -> list[tuple[str, str]]:
    simulation = Simulation(built)
    solved = {
        "op": simulation.run(OperatingPoint()),
        "tran": simulation.run(Transient(stop=6e-6 * s, max_step=0.1e-6 * s)),
    }
    return [(d.rule, d.message) for d in simulation.check_results(solved)]


def stated(pattern: str, message: str) -> float:
    """The stress `message` states where `pattern` holds `{}`.

    A transient's stress carries the rounding of its Newton iteration in its last
    digits, which the message states exactly.
    """
    head, tail = (re.escape(part) for part in pattern.split("{}"))
    found = re.fullmatch(rf"{head}(\S+){tail}", message)
    assert found is not None, message
    return float(found[1])


def test_a_stress_beyond_its_rating_is_an_error() -> None:
    found = raised(supply(rated_power=1.5, rated_voltage=16))
    assert [rule for rule, _ in found] == ["over-rating", "over-rating"]
    power = stated(
        "the power `R1` dissipates reaches {} W at 2e-06 s in `tran`, "
        "beyond its rating of 1.5 W",
        found[0][1],
    )
    voltage = stated(
        "the voltage of `C1` at `a` over `b` reaches {} V at 2e-06 s in `tran`, "
        "beyond its rating of 16.0 V",
        found[1][1],
    )
    # Exact linear pulse peak (20 V, 20^2 / 100 = 4 W) at breakpoint; LU rounding.
    assert power == pytest.approx(4.0, rel=1e-12)
    assert voltage == pytest.approx(20.0, rel=1e-12)


def test_a_stress_past_its_derating_is_a_warning() -> None:
    # 1 W at rest is 80 % of 1.25 W, and 20 V within 25 V is 80 % of it.
    found = raised(supply(rated_power=4.5, rated_voltage=24))
    assert [rule for rule, _ in found] == ["near-rating", "near-rating"]
    power = stated(
        "the power `R1` dissipates reaches {} W at 2e-06 s in `tran`, "
        "past 80% of its rating of 4.5 W",
        found[0][1],
    )
    # Exact linear pulse peak power (20^2 / 100 = 4 W) at breakpoint; LU rounding.
    assert power == pytest.approx(4.0, rel=1e-12)
    assert raised(supply(rated_power=10, rated_voltage=50)) == []


def test_a_stress_below_a_minimum_and_at_rest() -> None:
    @component
    def reversed_store(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(-2 * V), name="V1")
        load = c.place(devices.resistor(100 * ohm), name="R1")
        store = c.place(
            devices.capacitor(1 * uF).rated("voltage", 16 * V, minimum=-1 * V),
            name="C1",
        )
        c.connect(source.positive, load.a, store.a, name="top")
        c.connect(source.negative, load.b, store.b, ground)

    simulation = Simulation(reversed_store())
    found = simulation.check_results({"op": simulation.run(OperatingPoint())})
    assert [(d.rule, d.message) for d in found] == [
        (
            "over-rating",
            (
                "the voltage of `C1` at `a` over `b` falls to -2.0 V at the operating "
                "point of `op`, below its rating of -1.0 V"
            ),
        )
    ]
    assert found[0].instances[0].path == "C1"


def test_by_default_results_record_the_instances_rated_for_a_current_or_power() -> None:
    simulation = Simulation(supply(rated_power=0.5, rated_voltage=50))
    point = simulation.run(OperatingPoint())
    # R1's power reads its currents; C1's voltage reads its nets alone.
    assert point.recording.terminals == ("R1.a", "R1.b")
    assert [d.rule for d in simulation.check_results({"op": point})] == ["over-rating"]
    step = simulation.run(Transient(stop=1e-6 * s))
    assert step.recording.terminals == ("R1.a", "R1.b")
    sweep = simulation.run(DcSweep("V1", [0 * V, 10 * V]))
    assert sweep.recording == results.Recording()
    # A record that names nothing still records what the checks read.
    bare = simulation.run(OperatingPoint(record=[]))
    assert bare.recording.terminals == ("R1.a", "R1.b")
    assert [d.rule for d in simulation.check_results({"op": bare})] == ["over-rating"]


@component
def divided(c: Builder) -> None:
    """Two resistors of 100 Ω in series, a component rated for 0.5 W."""
    top, bottom = c.port("top"), c.port("bottom")
    upper = c.place(devices.resistor(100 * ohm), name="R1")
    lower = c.place(devices.resistor(100 * ohm), name="R2")
    c.connect(top, upper.a)
    c.connect(upper.b, lower.a)
    c.connect(lower.b, bottom)


def test_a_component_s_rating_reads_its_ports_without_its_devices() -> None:
    @component
    def bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(20 * V), name="V1")
        load = c.place(divided().rated("power", 1.5 * W), name="X1")
        c.connect(source.positive, load.top)
        c.connect(source.negative, load.bottom, ground)

    simulation = Simulation(bench())
    point = simulation.run(OperatingPoint())
    assert point.recording.devices == ()
    assert point.recording.summed_terminals == ("X1.top", "X1.bottom")
    ((rule, message),) = [
        (d.rule, d.message) for d in simulation.check_results({"op": point})
    ]
    assert rule == "over-rating"
    assert message.startswith("the power `X1` dissipates reaches 2.0 W")


def test_ratings_travel_with_the_circuit(tmp_path: Path) -> None:
    built = circuit(supply(rated_power=1.5, rated_voltage=16))
    path = tmp_path / "supply.jolty.json"
    save_circuit(path, built)
    loaded = load_circuit(path)
    assert loaded == built
    stored = loaded.find("C1")
    assert stored is not None
    assert stored.ratings[0].maximum == 16.0
