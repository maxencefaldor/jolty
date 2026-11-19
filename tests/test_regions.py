# Copyright 2026 Maxence Faldor
"""Operating-region classification and boundary margin checks for library transistors.

Every transistor model (IEEE 315 letter Q) is biased into each region distinguished by
its model equations, and operating-point recording reads back the region by path.
Boundary margins appear as device variables and sort closest-to-boundary first in
`jolty simulate` operating-point reports.
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
P_CHANNEL: dict[str, JsonValue] = {"channel": "p", "threshold_voltage": -1}
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
    "bsim4": (
        ("weak_inversion", {"vth0": 0.5}, {"drain": 1, "gate": 0.1}),
        ("linear", {"vth0": 0.5}, {"drain": 0.05, "gate": 1.8}),
        ("saturated", {"vth0": 0.5}, {"drain": 1.8, "gate": 1.8}),
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
    """Builds a circuit biasing `T1` of `model` at fixed terminal voltages."""
    c = Builder("biased")
    ground = c.port("ground", Role.GROUND)
    device = devices.device(model, parameters)
    transistor = c.place(device, name="T1")
    for terminal in device.terminals:
        if terminal not in voltages:
            c.connect(transistor[terminal], ground)
            continue
        source = c.place(devices.voltage_source(voltages[terminal] * V), name=terminal)
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
            # Region-only recording returns the same region as full device recording.
            alone = simulation.run(OperatingPoint(regions=True))
            assert alone.region("T1") == region, (model, voltages)
            for margin in device.margins:
                found = alone.variable("T1", margin)
                assert found == point.variable("T1", margin), (model, margin)
            assert alone.currents.size == alone.conductances.size == 0, model
        reached = {region for region, _, _ in biases}
        assert reached == distinguished, f"{model} reaches each of its regions"


def test_a_region_is_read_only_of_a_transistor_recorded() -> None:
    simulation = Simulation(biased("jfet", {}, {"drain": 5}))
    point = simulation.run(OperatingPoint(record=["drain"]))
    with pytest.raises(KeyError, match="'drain' distinguishes no regions"):
        point.region("drain")
    unrecorded = results.OperatingPointResult(
        point.unknowns, point.values, point.strategy
    )
    with pytest.raises(KeyError, match="recorded no device named 'T1'; record its"):
        unrecorded.region("T1")


def test_the_report_lists_each_transistor_nearest_a_boundary_first() -> None:
    c = Builder("pair")
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(5 * V), name="VDD")
    high = c.place(devices.voltage_source(3 * V), name="VH")
    low = c.place(devices.voltage_source(-2 * V), name="VL")
    far = c.place(devices.device("mosfet", LEVEL_1), name="M1")
    near = c.place(devices.device("mosfet", P_CHANNEL), name="M2")
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
    summary = report.report("op", point, max_unknowns=40)
    assert isinstance(summary, report.OperatingPointReport)
    # With threshold 1 V, M1 (Vg=3 V, Vd=5 V) has 2 V overdrive and 3 V saturation
    # margin. P-channel M2 (Vth=-1 V, Vg=Vd=-2 V) has 1 V of each, both positive for
    # saturated transistors.
    assert [transistor.path for transistor in summary.transistors] == ["M2", "M1"]
    (m1_overdrive, m1_margin) = summary.transistors[1].margins
    assert (m1_overdrive.variable, m1_margin.variable) == (
        "overdrive_voltage",
        "saturation_margin",
    )
    # Fixed gate/drain diffs (3 - 1 = 2 V, 5 - 2 = 3 V); default pytest.approx
    # (rel=1e-6, abs=1e-12) covers operating-point Newton rounding.
    assert (m1_overdrive.value, m1_margin.value) == pytest.approx((2, 3))
    assert summary.describe().splitlines()[-3:] == [
        "  regions, nearest a boundary first:",
        "    M2  saturated  overdrive_voltage 1 V  saturation_margin 1 V",
        "    M1  saturated  overdrive_voltage 2 V  saturation_margin 3 V",
    ]
    shortened = report.report("op", point, max_unknowns=1)
    assert isinstance(shortened, report.OperatingPointReport)
    assert [transistor.path for transistor in shortened.transistors] == ["M2"]
    assert shortened.unlisted_transistors == 1
    assert shortened.describe().endswith("  ... and 1 more in the results file")
