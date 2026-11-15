# Copyright 2026 Maxence Faldor
"""Each transistor's region of operation at an operating point, as its model states it.

Every transistor of the library, each model a part of IEEE 315's letter Q is made of,
is biased into each region its model's equations distinguish, and an operating point
that records it reads that region back by its path; the margins that place it there
are its variables, and the report lists them, those nearest a boundary first, at the
plan's own operating points that `jolty simulate` runs.
"""

import pytest
from pydantic import JsonValue

from jolty import (
    Builder,
    Component,
    OperatingPoint,
    Role,
    Simulation,
    devices,
    kicad,
    report,
    results,
)
from jolty.units import V

type Bias = tuple[str, dict[str, JsonValue], dict[str, float]]
"""A region, the parameters of a transistor, and its terminals' voltages, in volts,
that put it there; a terminal left out is on the ground."""

LEVEL_1: dict[str, JsonValue] = {"threshold_voltage": 1}
WEAK: dict[str, JsonValue] = {
    "level": 3,
    "threshold_voltage": 1,
    "fast_surface_state_density": 1e15,
}

BIASES: dict[str, tuple[Bias, ...]] = {
    "mosfet": (
        ("off", LEVEL_1, {"drain": 3, "gate": 0.5}),
        ("weak_inversion", WEAK, {"drain": 3, "gate": 0.5}),
        ("linear", LEVEL_1, {"drain": 0.5, "gate": 3}),
        ("saturated", LEVEL_1, {"drain": 3, "gate": 2}),
    ),
    "bsim3": (
        ("weak_inversion", {}, {"drain": 1, "gate": 0.2}),
        ("linear", {}, {"drain": 0.05, "gate": 2}),
        ("saturated", {}, {"drain": 3, "gate": 3}),
    ),
    "vdmos": (
        ("linear", {}, {"drain": 0.5, "gate": 6}),
        ("saturated", {}, {"drain": 10, "gate": 4}),
    ),
    "jfet": (
        ("off", {}, {"drain": 5, "gate": -3}),
        ("linear", {}, {"drain": 0.5}),
        ("saturated", {}, {"drain": 5}),
    ),
    "bipolar_transistor": (
        ("cut_off", {}, {"collector": 5, "base": -0.5}),
        ("active", {}, {"collector": 5, "base": 0.7}),
        ("saturated", {}, {"collector": 0.2, "base": 0.7}),
        ("reversed", {}, {"emitter": 5, "base": 0.7}),
    ),
    "vbic": (
        ("cut_off", {}, {"collector": 5, "base": -0.5}),
        ("active", {}, {"collector": 5, "base": 0.7}),
        ("saturated", {}, {"collector": 0.2, "base": 0.7}),
        ("reversed", {}, {"emitter": 5, "base": 0.7}),
    ),
}
"""A bias of every transistor model into each region its equations distinguish."""


def biased(
    model: str, parameters: dict[str, JsonValue], voltages: dict[str, float]
) -> Component:
    """A transistor `T1` of `model` with each terminal held at its voltage."""
    c = Builder("biased")
    ground = c.port("ground", Role.GROUND)
    device = devices.device(model, parameters)
    transistor = c.place(device, id="T1")
    for terminal in device.terminals:
        if terminal not in voltages:
            c.connect(transistor[terminal], ground)
            continue
        source = c.place(devices.voltage_source(voltages[terminal] * V), id=terminal)
        c.connect(source.positive, transistor[terminal], name=terminal)
        c.connect(source.negative, ground)
    return c.finish()


def test_every_transistor_reaches_each_region_its_model_distinguishes() -> None:
    transistors = {model for model, prefix in kicad.PREFIXES.items() if prefix == "Q"}
    assert set(BIASES) == transistors, "a bias of every transistor model, and no more"
    for model, biases in BIASES.items():
        distinguished = set()
        for region, parameters, voltages in biases:
            simulation = Simulation(biased(model, parameters, voltages))
            assert simulation.transistors == ("T1",), model
            point = simulation.run(OperatingPoint(record=simulation.transistors))
            assert point.region("T1") == region, (model, voltages)
            (device,) = point.recording.devices
            distinguished.update(device.regions)
        reached = {region for region, _, _ in biases}
        assert reached == distinguished, f"{model} reaches each of its regions"


def test_a_region_is_read_only_of_a_transistor_recorded() -> None:
    simulation = Simulation(biased("jfet", {}, {"drain": 5}))
    point = simulation.run(OperatingPoint(record=["drain"]))
    with pytest.raises(KeyError, match="'drain' distinguishes no regions"):
        point.region("drain")
    unrecorded = results.OperatingPoint(point.unknowns, point.values, point.strategy)
    with pytest.raises(KeyError, match="recorded no device named 'T1'; record its"):
        unrecorded.region("T1")


def test_the_report_lists_each_transistor_nearest_a_boundary_first() -> None:
    c = Builder("pair")
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(5 * V), id="VDD")
    high = c.place(devices.voltage_source(3 * V), id="VH")
    low = c.place(devices.voltage_source(2 * V), id="VL")
    far = c.place(devices.device("mosfet", LEVEL_1), id="M1")
    near = c.place(devices.device("mosfet", LEVEL_1), id="M2")
    c.connect(supply.positive, far.drain, name="vdd")
    c.connect(high.positive, far.gate, name="high")
    c.connect(low.positive, near.gate, near.drain, name="low")
    c.connect(
        ground,
        *(source.negative for source in (supply, high, low)),
        *(
            transistor[terminal]
            for transistor in (far, near)
            for terminal in ("source", "bulk")
        ),
    )
    simulation = Simulation(c.finish())
    point = simulation.run(OperatingPoint(record=simulation.transistors))
    summary = report.report("op", point, listed=40)
    assert isinstance(summary, report.PointReport)
    # Above a threshold of 1 V, M1's gate at 3 V and drain at 5 V leave it 2 V of
    # overdrive and 3 V past saturation; M2's gate and drain at 2 V, 1 V of each.
    assert [transistor.path for transistor in summary.transistors] == ["M2", "M1"]
    (m1_overdrive, m1_margin) = summary.transistors[1].margins
    assert (m1_overdrive.variable, m1_margin.variable) == (
        "overdrive_voltage",
        "saturation_margin",
    )
    assert (m1_overdrive.value, m1_margin.value) == pytest.approx((2, 3))
    assert summary.describe().splitlines()[-3:] == [
        "  regions, nearest a boundary first:",
        "    M2  saturated  overdrive_voltage 1 V  saturation_margin 1 V",
        "    M1  saturated  overdrive_voltage 2 V  saturation_margin 3 V",
    ]
    shortened = report.report("op", point, listed=1)
    assert isinstance(shortened, report.PointReport)
    assert [transistor.path for transistor in shortened.transistors] == ["M2"]
    assert shortened.unlisted_transistors == 1
    assert shortened.describe().endswith("  ... and 1 more in the results file")
