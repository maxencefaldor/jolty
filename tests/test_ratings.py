# Copyright 2026 Maxence Faldor
"""Ratings: the limits a device's maker states.

A simulation holds them to every operating point and transient it solves.
"""

from pathlib import Path

import pytest

from jolty import (
    Builder,
    Component,
    OperatingPoint,
    Role,
    Simulation,
    Transient,
    circuit,
    component,
    devices,
    ir,
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
    source = c.place(devices.voltage_source(pulse), id="V1")
    load = c.place(devices.resistor(100 * ohm).rated("power", rated_power * W), id="R1")
    store = c.place(
        devices.capacitor(1 * uF).rated("voltage", rated_voltage * V), id="C1"
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


def test_a_stress_beyond_its_rating_is_an_error() -> None:
    found = raised(supply(rated_power=1.5, rated_voltage=16))
    assert found == [
        (
            "over-rating",
            (
                "the power `R1` dissipates reaches 4.0 W at 2e-06 s in `tran`, beyond "
                "its rating of 1.5 W"
            ),
        ),
        (
            "over-rating",
            (
                "the voltage of `C1` at `a` over `b` reaches 20.0 V at 2e-06 s in "
                "`tran`, beyond its rating of 16.0 V"
            ),
        ),
    ]


def test_a_stress_past_its_derating_is_a_warning() -> None:
    # 1 W at rest is 80 % of 1.25 W, and 20 V within 25 V is 80 % of it.
    found = raised(supply(rated_power=4.5, rated_voltage=24))
    assert [rule for rule, _ in found] == ["derating", "derating"]
    assert found[0][1] == (
        "the power `R1` dissipates reaches 4.0 W at 2e-06 s in `tran`, past 80% of its "
        "rating of 4.5 W"
    )
    assert raised(supply(rated_power=10, rated_voltage=50)) == []


def test_a_stress_below_a_minimum_and_at_rest() -> None:
    @component
    def reversed_store(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(-2 * V), id="V1")
        load = c.place(devices.resistor(100 * ohm), id="R1")
        store = c.place(
            devices.capacitor(1 * uF).rated("voltage", 16 * V, minimum=-1 * V), id="C1"
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


def test_ratings_travel_with_the_circuit(tmp_path: Path) -> None:
    built = circuit(supply(rated_power=1.5, rated_voltage=16))
    path = tmp_path / "supply.jolty.json"
    save_circuit(path, built)
    loaded = load_circuit(path)
    assert loaded == built
    stored = loaded.find("C1")
    assert stored is not None
    assert stored.ratings[0].maximum == 16.0
