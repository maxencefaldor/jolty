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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import JsonValue

from jolty.units import (
    CAPACITANCE,
    CONDUCTANCE,
    CURRENT,
    DIMENSIONLESS,
    INDUCTANCE,
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
    return Device(
        "resistor", ("a", "b"), {"resistance": resistance.to(RESISTANCE, "resistance")}
    )


def capacitor(capacitance: Quantity) -> Device:
    """A linear capacitor between terminals ``a`` and ``b``."""
    value = capacitance.to(CAPACITANCE, "capacitance")
    return Device("capacitor", ("a", "b"), {"capacitance": value})


def inductor(inductance: Quantity) -> Device:
    """A linear inductor between terminals ``a`` and ``b``.

    Its current, from ``a`` through the inductor to ``b``, is the unknown ``current``.
    """
    value = inductance.to(INDUCTANCE, "inductance")
    return Device("inductor", ("a", "b"), {"inductance": value})


def voltage_source(waveform: Waveform | Quantity) -> Device:
    """An independent voltage source, ``positive`` minus ``negative`` at its level.

    A quantity is a constant level. The source's current, from ``positive`` through
    the source to ``negative``, is the unknown ``current``: negative while it delivers
    power.
    """
    level = Dc(waveform) if isinstance(waveform, Quantity) else waveform
    parameters: dict[str, JsonValue] = {"waveform": level.to_json(VOLTAGE)}
    return Device("voltage_source", ("positive", "negative"), parameters)


def current_source(waveform: Waveform | Quantity) -> Device:
    """An independent current source, driving its level from ``positive`` to the other.

    The current flows through the source from ``positive`` to ``negative``, so it enters
    the circuit at the ``negative`` terminal. A quantity is a constant level.
    """
    level = Dc(waveform) if isinstance(waveform, Quantity) else waveform
    parameters: dict[str, JsonValue] = {"waveform": level.to_json(CURRENT)}
    return Device("current_source", ("positive", "negative"), parameters)


_CONTROLLED = ("positive", "negative", "control_positive", "control_negative")


def voltage_controlled_voltage_source(gain: float) -> Device:
    """A source holding ``positive`` minus ``negative`` at `gain` times a voltage.

    That control voltage is ``control_positive`` minus ``control_negative``, and the
    control terminals draw no current.
    """
    parameters: dict[str, JsonValue] = {"gain": finite(gain, "gain")}
    return Device("voltage_controlled_voltage_source", _CONTROLLED, parameters)


def voltage_controlled_current_source(transconductance: Quantity) -> Device:
    """A source driving `transconductance` times the control voltage through itself.

    The current flows from ``positive`` through the source to ``negative``; the control
    voltage is ``control_positive`` minus ``control_negative``.
    """
    value = transconductance.to(CONDUCTANCE, "transconductance")
    parameters: dict[str, JsonValue] = {"transconductance": value}
    return Device("voltage_controlled_current_source", _CONTROLLED, parameters)


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
    return Device("diode", ("anode", "cathode"), parameters)


TRANSCONDUCTANCE_PARAMETER: Final = CURRENT / VOLTAGE**2
"""The dimension of a field-effect transistor's transconductance parameter, A/V²."""

PER_VOLT: Final = DIMENSIONLESS / VOLTAGE
"""The dimension of a channel-length modulation, V⁻¹."""


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
    return Device("jfet", ("drain", "gate", "source"), parameters)


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
    return Device("bipolar_transistor", ("collector", "base", "emitter"), parameters)


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
