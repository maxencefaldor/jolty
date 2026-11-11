# Copyright 2026 Maxence Faldor
"""The devices a build script places, each a model, its terminals and its parameters.

Each function checks the dimension of every parameter it reads, so a capacitance
passed where a resistance belongs fails on the line that passed it. Parameters left out
take the engine's defaults.

Examples:
    >>> from jolty.units import kohm
    >>> resistor(10 * kohm).parameters
    {'resistance': 10000.0}
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import JsonValue

from jolty import _core
from jolty.units import (
    CAPACITANCE,
    CONDUCTANCE,
    CURRENT,
    DIMENSIONLESS,
    INDUCTANCE,
    LENGTH,
    RESISTANCE,
    TIME,
    VOLTAGE,
    Dimension,
    Quantity,
    finite,
)
from jolty.waveforms import Dc, Waveform


@dataclass(frozen=True, slots=True)
class Device:
    """A device ready to place: its model's name, its terminals and its parameters."""

    model: str
    terminals: tuple[str, ...]
    parameters: dict[str, JsonValue]


def resistor(resistance: Quantity) -> Device:
    """A linear resistor between terminals ``a`` and ``b``."""
    return _device("resistor", {"resistance": resistance.to(RESISTANCE, "resistance")})


def capacitor(capacitance: Quantity) -> Device:
    """A linear capacitor between terminals ``a`` and ``b``."""
    value = capacitance.to(CAPACITANCE, "capacitance")
    return _device("capacitor", {"capacitance": value})


def inductor(inductance: Quantity) -> Device:
    """A linear inductor between terminals ``a`` and ``b``.

    Its current, from ``a`` through the inductor to ``b``, is the unknown ``current``.
    """
    value = inductance.to(INDUCTANCE, "inductance")
    return _device("inductor", {"inductance": value})


def voltage_source(waveform: Waveform | Quantity) -> Device:
    """An independent voltage source, ``positive`` minus ``negative`` at its level.

    A quantity is a constant level. The source's current, from ``positive`` through
    the source to ``negative``, is the unknown ``current``: negative while it delivers
    power.
    """
    level = Dc(waveform) if isinstance(waveform, Quantity) else waveform
    parameters: dict[str, JsonValue] = {"waveform": level.to_json(VOLTAGE)}
    return _device("voltage_source", parameters)


def current_source(waveform: Waveform | Quantity) -> Device:
    """An independent current source, driving its level from ``positive`` to the other.

    The current flows through the source from ``positive`` to ``negative``, so it enters
    the circuit at the ``negative`` terminal. A quantity is a constant level.
    """
    level = Dc(waveform) if isinstance(waveform, Quantity) else waveform
    parameters: dict[str, JsonValue] = {"waveform": level.to_json(CURRENT)}
    return _device("current_source", parameters)


def voltage_controlled_voltage_source(gain: float) -> Device:
    """A source holding ``positive`` minus ``negative`` at `gain` times a voltage.

    That control voltage is ``control_positive`` minus ``control_negative``, and the
    control terminals draw no current.
    """
    parameters: dict[str, JsonValue] = {"gain": finite(gain, "gain")}
    return _device("voltage_controlled_voltage_source", parameters)


def voltage_controlled_current_source(transconductance: Quantity) -> Device:
    """A source driving `transconductance` times the control voltage through itself.

    The current flows from ``positive`` through the source to ``negative``; the control
    voltage is ``control_positive`` minus ``control_negative``.
    """
    value = transconductance.to(CONDUCTANCE, "transconductance")
    parameters: dict[str, JsonValue] = {"transconductance": value}
    return _device("voltage_controlled_current_source", parameters)


def current_controlled_current_source(gain: float) -> Device:
    """A source driving `gain` times a sensed current through itself.

    The control port, from ``control_positive`` to ``control_negative``, is a short
    circuit that senses the current through it, the unknown ``control``; placed in
    series with a branch, it senses that branch's current. The output current flows
    from ``positive`` through the source to ``negative``.
    """
    parameters: dict[str, JsonValue] = {"gain": finite(gain, "gain")}
    return _device("current_controlled_current_source", parameters)


def current_controlled_voltage_source(transresistance: Quantity) -> Device:
    """A source holding its output at `transresistance` times a sensed current.

    The control port, from ``control_positive`` to ``control_negative``, is a short
    circuit that senses the current through it, the unknown ``control``. The output,
    ``positive`` minus ``negative``, is held at the transresistance times that current;
    its current, from ``positive`` through the source to ``negative``, is the unknown
    ``current``, as a voltage source's is.
    """
    value = transresistance.to(RESISTANCE, "transresistance")
    parameters: dict[str, JsonValue] = {"transresistance": value}
    return _device("current_controlled_voltage_source", parameters)


def behavioral_voltage_source(expression: str) -> Device:
    """A source holding its output, ``positive`` minus ``negative``, at `expression`.

    The expression reads voltages and currents in SI units: ``v(a)`` is the voltage
    of terminal ``a``, ``v(a, b)`` that of ``a`` over ``b``, ``i(p)`` the current
    through the sensing port ``p``, and ``time`` the time. Its terminals are
    ``positive`` and ``negative``, then each terminal it reads, then ``p_positive`` and
    ``p_negative`` for each port ``p``: a short circuit whose current, from
    ``p_positive`` to ``p_negative``, is the unknown ``p``. The source's own current,
    from ``positive`` through the source to ``negative``, is the unknown ``current``.
    The reference's expressions page gives the language, its operators and its
    functions.

    Raises:
        ValueError: When `expression` is not an expression.

    Examples:
        >>> behavioral_voltage_source("1e5 * v(inp, inn)").terminals
        ('positive', 'negative', 'inp', 'inn')
    """
    return _device("behavioral_voltage_source", {"expression": expression})


def behavioral_current_source(expression: str) -> Device:
    """A source driving `expression` from ``positive`` through itself to ``negative``.

    The expression and the terminals are a behavioral voltage source's.

    Raises:
        ValueError: When `expression` is not an expression.

    Examples:
        >>> behavioral_current_source("limit(1e-3 * v(c), -1e-3, 1e-3)").terminals
        ('positive', 'negative', 'c')
    """
    return _device("behavioral_current_source", {"expression": expression})


def voltage_controlled_switch(
    *,
    on_resistance: Quantity | None = None,
    off_resistance: Quantity | None = None,
    on_voltage: Quantity | None = None,
    off_voltage: Quantity | None = None,
) -> Device:
    """A switch between ``positive`` and ``negative``, in PSpice's model.

    The voltage of ``control_positive`` over ``control_negative`` moves its resistance
    smoothly, logarithmically, from the off resistance at the off voltage to the on
    resistance at the on voltage; beyond either, the resistance holds. The control
    terminals draw no current. Parameters left out take PSpice's defaults: 1 Ω on, 1 MΩ
    off, on at 1 V and off at 0 V.
    """
    parameters = _parameters(
        {
            "on_resistance": (on_resistance, RESISTANCE),
            "off_resistance": (off_resistance, RESISTANCE),
            "on_voltage": (on_voltage, VOLTAGE),
            "off_voltage": (off_voltage, VOLTAGE),
        },
        {},
    )
    return _device("voltage_controlled_switch", parameters)


def diode(  # noqa: PLR0913 - the SPICE junction model has eight parameters
    *,
    saturation_current: Quantity | None = None,
    emission_coefficient: float | None = None,
    series_resistance: Quantity | None = None,
    junction_capacitance: Quantity | None = None,
    junction_potential: Quantity | None = None,
    grading_coefficient: float | None = None,
    depletion_coefficient: float | None = None,
    transit_time: Quantity | None = None,
) -> Device:
    """A junction diode from ``anode`` to ``cathode``, in the SPICE model.

    Parameters left out take SPICE's defaults: a saturation current of 10 fA, an
    emission coefficient of 1, no series resistance, junction capacitance or transit
    time, a junction potential of 1 V, and grading and depletion coefficients of 0.5.
    """
    parameters = _parameters(
        {
            "saturation_current": (saturation_current, CURRENT),
            "series_resistance": (series_resistance, RESISTANCE),
            "junction_capacitance": (junction_capacitance, CAPACITANCE),
            "junction_potential": (junction_potential, VOLTAGE),
            "transit_time": (transit_time, TIME),
        },
        {
            "emission_coefficient": emission_coefficient,
            "grading_coefficient": grading_coefficient,
            "depletion_coefficient": depletion_coefficient,
        },
    )
    return _device("diode", parameters)


TRANSCONDUCTANCE_PARAMETER: Final = CURRENT / VOLTAGE**2
"""The dimension of a field-effect transistor's transconductance parameter, A/V²."""

PER_VOLT: Final = DIMENSIONLESS / VOLTAGE
"""The dimension of a channel-length modulation, V⁻¹."""

MOBILITY: Final = LENGTH**2 / (VOLTAGE * TIME)
"""The dimension of a carrier mobility, m²/(V·s)."""


def jfet(  # noqa: PLR0913 - the SPICE JFET model has twelve parameters
    *,
    channel: Literal["n", "p"] | None = None,
    threshold_voltage: Quantity | None = None,
    transconductance_parameter: Quantity | None = None,
    channel_length_modulation: Quantity | None = None,
    drain_resistance: Quantity | None = None,
    source_resistance: Quantity | None = None,
    gate_source_capacitance: Quantity | None = None,
    gate_drain_capacitance: Quantity | None = None,
    junction_potential: Quantity | None = None,
    saturation_current: Quantity | None = None,
    depletion_coefficient: float | None = None,
    doping_tail: float | None = None,
) -> Device:
    """A junction field-effect transistor between ``drain``, ``gate`` and ``source``.

    It follows the SPICE model with Sydney University's doping tail, as ngspice does:
    two gate junctions, and a drain current that follows the square law above the
    threshold voltage. Parameters left out take SPICE's defaults: an n-channel, a
    threshold of -2 V, a transconductance parameter of 100 µA/V², no channel-length
    modulation, resistance or capacitance, a junction potential of 1 V, a saturation
    current of 10 fA, a depletion coefficient of 0.5, and a doping tail of 1, the plain
    square law. A drain or source resistance adds an internal node, the unknown
    ``drain`` or ``source``.

    Examples:
        >>> from jolty.units import V, mA
        >>> jfet(channel="p", transconductance_parameter=1 * mA / V**2).parameters
        {'channel': 'p', 'transconductance_parameter': 0.001}
    """
    parameters: dict[str, JsonValue] = {} if channel is None else {"channel": channel}
    parameters |= _parameters(
        {
            "threshold_voltage": (threshold_voltage, VOLTAGE),
            "transconductance_parameter": (
                transconductance_parameter,
                TRANSCONDUCTANCE_PARAMETER,
            ),
            "channel_length_modulation": (channel_length_modulation, PER_VOLT),
            "drain_resistance": (drain_resistance, RESISTANCE),
            "source_resistance": (source_resistance, RESISTANCE),
            "gate_source_capacitance": (gate_source_capacitance, CAPACITANCE),
            "gate_drain_capacitance": (gate_drain_capacitance, CAPACITANCE),
            "junction_potential": (junction_potential, VOLTAGE),
            "saturation_current": (saturation_current, CURRENT),
        },
        {"depletion_coefficient": depletion_coefficient, "doping_tail": doping_tail},
    )
    return _device("jfet", parameters)


def bipolar_transistor(  # noqa: PLR0913 - the Gummel-Poon model has 32 parameters
    *,
    polarity: Literal["npn", "pnp"] | None = None,
    saturation_current: Quantity | None = None,
    forward_beta: float | None = None,
    reverse_beta: float | None = None,
    forward_emission_coefficient: float | None = None,
    reverse_emission_coefficient: float | None = None,
    forward_early_voltage: Quantity | None = None,
    reverse_early_voltage: Quantity | None = None,
    forward_knee_current: Quantity | None = None,
    reverse_knee_current: Quantity | None = None,
    base_emitter_leakage_current: Quantity | None = None,
    base_emitter_leakage_emission_coefficient: float | None = None,
    base_collector_leakage_current: Quantity | None = None,
    base_collector_leakage_emission_coefficient: float | None = None,
    base_resistance: Quantity | None = None,
    minimum_base_resistance: Quantity | None = None,
    base_resistance_current: Quantity | None = None,
    emitter_resistance: Quantity | None = None,
    collector_resistance: Quantity | None = None,
    base_emitter_capacitance: Quantity | None = None,
    base_emitter_potential: Quantity | None = None,
    base_emitter_grading: float | None = None,
    forward_transit_time: Quantity | None = None,
    transit_time_bias_coefficient: float | None = None,
    transit_time_voltage: Quantity | None = None,
    transit_time_current: Quantity | None = None,
    base_collector_capacitance: Quantity | None = None,
    base_collector_potential: Quantity | None = None,
    base_collector_grading: float | None = None,
    internal_base_fraction: float | None = None,
    reverse_transit_time: Quantity | None = None,
    depletion_coefficient: float | None = None,
) -> Device:
    """A bipolar junction transistor between ``collector``, ``base`` and ``emitter``.

    It follows the Gummel-Poon model of SPICE, as ngspice does. Parameters left out take
    SPICE's defaults: an NPN with a saturation current of 0.1 fA, a forward beta of 100
    and a reverse beta of 1, emission coefficients of 1, infinite Early voltages and
    knee currents, no leakage, resistance, capacitance or transit time, and junction
    potentials of 0.75 V with grading coefficients of 0.33. A base, collector or emitter
    resistance adds an internal node, the unknown ``base``, ``collector`` or
    ``emitter``.

    Examples:
        >>> from jolty.units import V, fA
        >>> transistor = bipolar_transistor(
        ...     saturation_current=10 * fA,
        ...     forward_beta=200,
        ...     forward_early_voltage=50 * V,
        ... )
        >>> transistor.parameters["forward_early_voltage"]
        50.0
    """
    parameters: dict[str, JsonValue] = (
        {} if polarity is None else {"polarity": polarity}
    )
    parameters |= _parameters(
        {
            "saturation_current": (saturation_current, CURRENT),
            "forward_early_voltage": (forward_early_voltage, VOLTAGE),
            "reverse_early_voltage": (reverse_early_voltage, VOLTAGE),
            "forward_knee_current": (forward_knee_current, CURRENT),
            "reverse_knee_current": (reverse_knee_current, CURRENT),
            "base_emitter_leakage_current": (base_emitter_leakage_current, CURRENT),
            "base_collector_leakage_current": (base_collector_leakage_current, CURRENT),
            "base_resistance": (base_resistance, RESISTANCE),
            "minimum_base_resistance": (minimum_base_resistance, RESISTANCE),
            "base_resistance_current": (base_resistance_current, CURRENT),
            "emitter_resistance": (emitter_resistance, RESISTANCE),
            "collector_resistance": (collector_resistance, RESISTANCE),
            "base_emitter_capacitance": (base_emitter_capacitance, CAPACITANCE),
            "base_emitter_potential": (base_emitter_potential, VOLTAGE),
            "forward_transit_time": (forward_transit_time, TIME),
            "transit_time_voltage": (transit_time_voltage, VOLTAGE),
            "transit_time_current": (transit_time_current, CURRENT),
            "base_collector_capacitance": (base_collector_capacitance, CAPACITANCE),
            "base_collector_potential": (base_collector_potential, VOLTAGE),
            "reverse_transit_time": (reverse_transit_time, TIME),
        },
        {
            "forward_beta": forward_beta,
            "reverse_beta": reverse_beta,
            "forward_emission_coefficient": forward_emission_coefficient,
            "reverse_emission_coefficient": reverse_emission_coefficient,
            "base_emitter_leakage_emission_coefficient": (
                base_emitter_leakage_emission_coefficient
            ),
            "base_collector_leakage_emission_coefficient": (
                base_collector_leakage_emission_coefficient
            ),
            "base_emitter_grading": base_emitter_grading,
            "transit_time_bias_coefficient": transit_time_bias_coefficient,
            "base_collector_grading": base_collector_grading,
            "internal_base_fraction": internal_base_fraction,
            "depletion_coefficient": depletion_coefficient,
        },
    )
    return _device("bipolar_transistor", parameters)


def mosfet(  # noqa: PLR0913 - SPICE's level 1 MOSFET has 37 parameters
    *,
    channel: Literal["n", "p"] | None = None,
    width: Quantity | None = None,
    length: Quantity | None = None,
    multiplier: float | None = None,
    threshold_voltage: Quantity | None = None,
    transconductance_parameter: Quantity | None = None,
    body_effect: float | None = None,
    surface_potential: Quantity | None = None,
    channel_length_modulation: Quantity | None = None,
    lateral_diffusion: Quantity | None = None,
    oxide_thickness: Quantity | None = None,
    surface_mobility: Quantity | None = None,
    substrate_doping: Quantity | None = None,
    surface_state_density: Quantity | None = None,
    gate_material: Literal["opposite_doping", "same_doping", "aluminum"] | None = None,
    drain_resistance: Quantity | None = None,
    source_resistance: Quantity | None = None,
    sheet_resistance: Quantity | None = None,
    drain_squares: float | None = None,
    source_squares: float | None = None,
    saturation_current: Quantity | None = None,
    saturation_current_density: Quantity | None = None,
    drain_area: Quantity | None = None,
    source_area: Quantity | None = None,
    drain_perimeter: Quantity | None = None,
    source_perimeter: Quantity | None = None,
    junction_potential: Quantity | None = None,
    bulk_drain_capacitance: Quantity | None = None,
    bulk_source_capacitance: Quantity | None = None,
    bottom_capacitance: Quantity | None = None,
    sidewall_capacitance: Quantity | None = None,
    bottom_grading: float | None = None,
    sidewall_grading: float | None = None,
    depletion_coefficient: float | None = None,
    gate_source_overlap: Quantity | None = None,
    gate_drain_overlap: Quantity | None = None,
    gate_bulk_overlap: Quantity | None = None,
) -> Device:
    """A MOSFET between ``drain``, ``gate``, ``source`` and ``bulk``.

    It follows SPICE's level 1 model, Shichman and Hodges's square law with Meyer's gate
    capacitances, as ngspice does. Above the threshold, the drain current is
    ``kp · w / (l - 2 · ld) · (1 + lambda · vds)`` times ``(vgs - von)² / 2`` in
    saturation and ``vds · (vgs - von - vds / 2)`` below it, where the threshold
    ``von = vto + gamma · (√(phi - vbs) - √phi)`` rises with the body effect. Two
    junctions join the bulk to the drain and the source. Parameters left out take
    SPICE's defaults: an n-channel 100 µm wide and long, a threshold of 0 V, a
    transconductance parameter of 20 µA/V², no body effect, a surface potential of
    0.6 V, a saturation current of 10 fA, a junction potential of 0.8 V, grading and
    depletion coefficients of 0.5, and no channel-length modulation, resistance or
    capacitance. An oxide thickness gives the gate Meyer's capacitances; with a
    substrate doping, it derives the threshold, the body effect and the surface
    potential left out, as SPICE derives them. The body effect is a number of
    square-root volts.

    A drain or source resistance adds an internal node, the unknown ``drain`` or
    ``source``. Meyer's capacitances add three currents of the transistor's own, the
    displacement currents through the whole oxide towards the source, the drain and the
    bulk: ``gate_source_displacement``, ``gate_drain_displacement`` and
    ``gate_bulk_displacement``.

    Examples:
        >>> from jolty.units import nm, um
        >>> mosfet(width=2 * um, length=1 * um, oxide_thickness=20 * nm).parameters
        {'width': 2e-06, 'length': 1e-06, 'oxide_thickness': 2e-08}
    """
    parameters: dict[str, JsonValue] = {} if channel is None else {"channel": channel}
    if gate_material is not None:
        parameters["gate_material"] = gate_material
    capacitance_per_length = CAPACITANCE / LENGTH
    parameters |= _parameters(
        {
            "width": (width, LENGTH),
            "length": (length, LENGTH),
            "threshold_voltage": (threshold_voltage, VOLTAGE),
            "transconductance_parameter": (
                transconductance_parameter,
                TRANSCONDUCTANCE_PARAMETER,
            ),
            "surface_potential": (surface_potential, VOLTAGE),
            "channel_length_modulation": (channel_length_modulation, PER_VOLT),
            "lateral_diffusion": (lateral_diffusion, LENGTH),
            "oxide_thickness": (oxide_thickness, LENGTH),
            "surface_mobility": (surface_mobility, MOBILITY),
            "substrate_doping": (substrate_doping, DIMENSIONLESS / LENGTH**3),
            "surface_state_density": (surface_state_density, DIMENSIONLESS / LENGTH**2),
            "drain_resistance": (drain_resistance, RESISTANCE),
            "source_resistance": (source_resistance, RESISTANCE),
            "sheet_resistance": (sheet_resistance, RESISTANCE),
            "saturation_current": (saturation_current, CURRENT),
            "saturation_current_density": (
                saturation_current_density,
                CURRENT / LENGTH**2,
            ),
            "drain_area": (drain_area, LENGTH**2),
            "source_area": (source_area, LENGTH**2),
            "drain_perimeter": (drain_perimeter, LENGTH),
            "source_perimeter": (source_perimeter, LENGTH),
            "junction_potential": (junction_potential, VOLTAGE),
            "bulk_drain_capacitance": (bulk_drain_capacitance, CAPACITANCE),
            "bulk_source_capacitance": (bulk_source_capacitance, CAPACITANCE),
            "bottom_capacitance": (bottom_capacitance, CAPACITANCE / LENGTH**2),
            "sidewall_capacitance": (sidewall_capacitance, capacitance_per_length),
            "gate_source_overlap": (gate_source_overlap, capacitance_per_length),
            "gate_drain_overlap": (gate_drain_overlap, capacitance_per_length),
            "gate_bulk_overlap": (gate_bulk_overlap, capacitance_per_length),
        },
        {
            "multiplier": multiplier,
            "body_effect": body_effect,
            "drain_squares": drain_squares,
            "source_squares": source_squares,
            "bottom_grading": bottom_grading,
            "sidewall_grading": sidewall_grading,
            "depletion_coefficient": depletion_coefficient,
        },
    )
    return _device("mosfet", parameters)


def _device(model: str, parameters: dict[str, JsonValue]) -> Device:
    """A device of `model` with `parameters`, with the terminals the engine gives it.

    Raises:
        ValueError: When the engine cannot read the parameters.
    """
    terminals = _core.terminals(model, json.dumps(parameters))
    return Device(model, tuple(terminals), parameters)


def _parameters(
    quantities: Mapping[str, tuple[Quantity | None, Dimension]],
    numbers: Mapping[str, float | None],
) -> dict[str, JsonValue]:
    """The parameters given, by name, each checked and in SI base units.

    A quantity's dimension is checked, and a number is checked finite.

    Raises:
        DimensionError: When a quantity has the wrong dimension.
        ValueError: When a value is not finite.
    """
    parameters: dict[str, JsonValue] = {
        name: value.to(dimension, name)
        for name, (value, dimension) in quantities.items()
        if value is not None
    }
    parameters |= {
        name: finite(value, name)
        for name, value in numbers.items()
        if value is not None
    }
    return parameters
