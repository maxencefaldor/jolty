# Copyright 2026 Maxence Faldor

import inspect
import math
from collections.abc import Callable
from typing import Literal

import pytest

from jolty import _core, devices
from jolty.errors import DimensionError
from jolty.units import (
    LENGTH,
    VOLTAGE,
    K,
    S,
    V,
    fA,
    kohm,
    m,
    mA,
    mH,
    ms,
    nm,
    ns,
    ohm,
    pF,
    ps,
    s,
    uC,
    uF,
    um,
)

ANNOTATED_BUILDERS: tuple[tuple[str, Callable[..., devices.Device], set[str]], ...] = (
    ("resistor", devices.resistor, {"tolerance"}),
    ("capacitor", devices.capacitor, {"tolerance"}),
    ("inductor", devices.inductor, {"tolerance"}),
    ("transmission_line", devices.transmission_line, set()),
    (
        "voltage_controlled_voltage_source",
        devices.voltage_controlled_voltage_source,
        set(),
    ),
    (
        "voltage_controlled_current_source",
        devices.voltage_controlled_current_source,
        set(),
    ),
    (
        "current_controlled_current_source",
        devices.current_controlled_current_source,
        set(),
    ),
    (
        "current_controlled_voltage_source",
        devices.current_controlled_voltage_source,
        set(),
    ),
    ("voltage_controlled_switch", devices.voltage_controlled_switch, set()),
    ("current_controlled_switch", devices.current_controlled_switch, set()),
    ("photocell", devices.photocell, set()),
    ("voltage_controlled_memristor", devices.voltage_controlled_memristor, set()),
    ("current_controlled_memristor", devices.current_controlled_memristor, set()),
    ("diode", devices.diode, set()),
    ("jfet", devices.jfet, set()),
    ("bipolar_transistor", devices.bipolar_transistor, set()),
    ("vbic", devices.vbic, set()),
    ("mosfet", devices.mosfet, set()),
    ("vdmos", devices.vdmos, set()),
    ("bsim3", devices.bsim3, {"card"}),
    ("bsim4", devices.bsim4, {"card"}),
)


def test_parameter_metadata_is_queryable_for_every_registered_model() -> None:
    for model in _core.models():
        dimensions = _core.parameter_dimensions(model)
        binned = _core.binned_parameters(model)
        assert isinstance(dimensions, dict)
        assert isinstance(binned, list)
        assert set(binned) <= set(dimensions)

    assert _core.parameter_dimensions("logic_cell") == {}
    assert _core.binned_parameters("logic_cell") == []

    with pytest.raises(ValueError, match="no model"):
        _core.parameter_dimensions("nonexistent_model")
    with pytest.raises(ValueError, match="no model"):
        _core.binned_parameters("nonexistent_model")


@pytest.mark.parametrize(("model", "builder", "excluded"), ANNOTATED_BUILDERS)
def test_builder_signatures_match_engine_parameter_dimensions(
    model: str,
    builder: Callable[..., devices.Device],
    excluded: set[str],
) -> None:
    parameters = set(inspect.signature(builder).parameters) - excluded
    exported = set(_core.parameter_dimensions(model))
    if model in {"bsim3", "bsim4"}:
        assert parameters <= exported
    else:
        assert parameters == exported


def test_bsim_cards_and_binned_sets_derive_from_engine() -> None:
    bsim3_instance = set(inspect.signature(devices.bsim3).parameters) - {"card"}
    bsim4_instance = set(inspect.signature(devices.bsim4).parameters) - {"card"}

    assert set(devices.BSIM3_CARD) == (
        set(_core.parameter_dimensions("bsim3")) - bsim3_instance
    )
    assert set(devices.BSIM4_CARD) == (
        set(_core.parameter_dimensions("bsim4")) - bsim4_instance
    )
    assert devices.BSIM3_BINNED == frozenset(_core.binned_parameters("bsim3"))
    assert devices.BSIM4_BINNED == frozenset(_core.binned_parameters("bsim4"))
    assert devices.BSIM3_BINNED <= set(devices.BSIM3_CARD)
    assert devices.BSIM4_BINNED <= set(devices.BSIM4_CARD)
    assert devices.BSIM3_CARD["uc"] == "uc"
    assert devices.BSIM3_CARD["uc1"] == "uc"
    assert devices.bsim3_dimension("uc", mobility_model=3) == devices.PER_VOLT
    assert devices.bsim3_dimension("uc", mobility_model=1) == LENGTH / VOLTAGE**2
    assert devices.bsim4_dimension("lvth0") == VOLTAGE * LENGTH


def test_diode_breakdown_temperature_coefficient_follows_temperature_model() -> None:
    linear = devices.diode(
        temperature_model=0,
        breakdown_temperature_coefficient=2e-3 * V / K,
    )
    assert linear.parameters["breakdown_temperature_coefficient"] == pytest.approx(2e-3)
    with pytest.raises(DimensionError, match="breakdown_temperature_coefficient"):
        devices.diode(
            temperature_model=0,
            breakdown_temperature_coefficient=2e-3 / K,
        )

    relative_models: tuple[Literal[1, 2], ...] = (1, 2)
    for model in relative_models:
        relative = devices.diode(
            temperature_model=model,
            breakdown_temperature_coefficient=2e-3 / K,
        )
        assert relative.parameters["breakdown_temperature_coefficient"] == (
            pytest.approx(2e-3)
        )
        with pytest.raises(DimensionError, match="breakdown_temperature_coefficient"):
            devices.diode(
                temperature_model=model,
                breakdown_temperature_coefficient=2e-3 * V / K,
            )


def test_passive_and_analog_builders_validate_parameters() -> None:
    made_resistor = devices.resistor(
        1 * kohm,
        tolerance=0.01,
        temperature_coefficient=1e-4 / K,
        flicker_coefficient=1e-12,
    )
    assert made_resistor.parameters["resistance"] == 1000.0
    assert made_resistor.tolerances["resistance"].relative == 0.01

    made_capacitor = devices.capacitor(1 * uF, tolerance=0.05)
    assert made_capacitor.tolerances["capacitance"].relative == 0.05
    made_inductor = devices.inductor(1 * mH, tolerance=0.1)
    assert made_inductor.tolerances["inductance"].relative == 0.1
    assert devices.transmission_line(50 * ohm, 1 * ns).parameters["delay"] == 1e-9
    assert devices.voltage_controlled_voltage_source(2.5).parameters["gain"] == 2.5
    transconductance_source = devices.voltage_controlled_current_source(0.01 * S)
    assert transconductance_source.parameters["transconductance"] == 0.01
    assert devices.current_controlled_current_source(3.0).parameters["gain"] == 3.0
    transresistance_source = devices.current_controlled_voltage_source(1 * kohm)
    assert transresistance_source.parameters["transresistance"] == 1000.0
    voltage_switch = devices.voltage_controlled_switch(on_voltage=1 * V)
    assert voltage_switch.parameters["on_voltage"] == 1.0
    current_switch = devices.current_controlled_switch(on_current=1 * mA)
    assert current_switch.parameters["on_current"] == 1e-3
    cell = devices.photocell(10 * kohm, gamma=0.8, response_time=20 * ms)
    assert cell.parameters["gamma"] == 0.8
    voltage_memristor = devices.voltage_controlled_memristor(
        on_resistance=1 * kohm,
        off_resistance=100 * kohm,
        on_voltage=-1 * V,
        off_voltage=0.5 * V,
        on_rate=1e6 / s,
        off_rate=1e5 / s,
        law="exponential",
    )
    assert voltage_memristor.parameters["law"] == "exponential"
    current_memristor = devices.current_controlled_memristor(
        on_resistance=100 * ohm,
        off_resistance=16 * kohm,
        charge=2 * uC,
        law="linear",
    )
    assert current_memristor.parameters["charge"] == 2e-6
    electrical_cell = devices.logic_cell(
        ["a"],
        {"y": "10"},
        rise=20 * ps,
        transition=10 * ps,
        electrical=True,
        threshold=0.4,
    )
    assert electrical_cell.parameters["electrical"] == {
        "input_capacitances": [0.0],
        "threshold": 0.4,
    }


def test_semiconductor_builders_validate_parameters() -> None:
    made_jfet = devices.jfet(channel="p", transconductance_parameter=1 * mA / V**2)
    assert made_jfet.parameters["channel"] == "p"
    bipolar = devices.bipolar_transistor(
        polarity="pnp", saturation_current=10 * fA, forward_beta=150
    )
    assert bipolar.parameters["polarity"] == "pnp"
    made_vbic = devices.vbic(
        polarity="npn", base_charge="vbic", saturation_current=10 * fA
    )
    assert made_vbic.parameters["base_charge"] == "vbic"
    made_mosfet = devices.mosfet(
        level=3,
        channel="n",
        width=2 * um,
        length=1 * um,
        body_effect=0.5,
        gate_material="aluminum",
    )
    assert made_mosfet.parameters["body_effect"] == 0.5
    made_vdmos = devices.vdmos(
        channel="n", threshold_voltage=4 * V, gate_source_capacitance=100 * pF
    )
    assert made_vdmos.parameters["gate_source_capacitance"] == 1e-10
    made_bsim3 = devices.bsim3(
        width=2 * um,
        length=0.35 * um,
        check_parameters=False,
        mobility_model=1,
        card={"uc": -4.65e-11 * m / V**2, "k2": -0.02},
    )
    assert made_bsim3.parameters["check_parameters"] is False
    made_bsim4 = devices.bsim4(
        width=2 * um,
        length=0.15 * um,
        check_parameters=True,
        card={"toxe": 4 * nm, "vth0": 0.5 * V},
    )
    assert made_bsim4.parameters["check_parameters"] is True


def test_device_builders_reject_wrong_dimension_and_non_finite_values() -> None:
    with pytest.raises(DimensionError, match="resistance"):
        devices.resistor(1 * uF)
    with pytest.raises(DimensionError, match="saturation_current"):
        devices.bipolar_transistor(saturation_current=1 * V)
    with pytest.raises(ValueError, match="finite"):
        devices.bipolar_transistor(forward_beta=math.inf)
    with pytest.raises(DimensionError, match="uc"):
        devices.bsim3(mobility_model=3, card={"uc": -4.65e-11 * m / V**2})
