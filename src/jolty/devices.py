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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Final, Literal

from pydantic import JsonValue

from jolty import _core, ir
from jolty.units import (
    CAPACITANCE,
    CONDUCTANCE,
    CURRENT,
    DIMENSIONLESS,
    INDUCTANCE,
    LENGTH,
    RESISTANCE,
    TEMPERATURE,
    TIME,
    VOLTAGE,
    Dimension,
    Quantity,
    finite,
)
from jolty.waveforms import Dc, Waveform


@dataclass(frozen=True, slots=True)
class Device:
    """A device ready to place: its model's name, its terminals and its parameters.

    Some parameters may have a tolerance, which Monte Carlo analyses sample.
    """

    model: str
    terminals: tuple[str, ...]
    parameters: dict[str, JsonValue]
    tolerances: dict[str, ir.Tolerance] = field(default_factory=dict)

    def toleranced(
        self,
        parameter: str,
        relative: float,
        distribution: ir.Distribution = "uniform",
        lot: ir.Lot | None = None,
    ) -> Device:
        """The device, `parameter` given a `relative` tolerance, such as 0.01 for 1 %.

        The parameter is named by [path][jolty.ir.parameter], so ``waveform.dc`` is a
        source's DC level, and values spread by their
        [distribution][jolty.ir.Distribution]: evenly or normally within the tolerance,
        or in the logarithm within a factor of ``1 + relative``. Monte Carlo analyses
        sample them, scaled by the factor of the `lot` the part comes from, if any,
        which every part in the lot shares.

        Examples:
            >>> from jolty.units import V
            >>> supply = voltage_source(5 * V).toleranced("waveform.dc", 0.05)
            >>> supply.tolerances["waveform.dc"].relative
            0.05

        Raises:
            ValueError: When the device gives the parameter no number, or its
                distribution does not take the tolerance.
        """
        match ir.parameter(self.parameters, parameter):
            case float() | int():
                pass
            case _:
                message = f"a {self.model} gives {parameter} no number to tolerate"
                raise ValueError(message)
        tolerance = ir.Tolerance(relative=relative, distribution=distribution, lot=lot)
        return replace(self, tolerances={**self.tolerances, parameter: tolerance})


PER_KELVIN: Final = DIMENSIONLESS / TEMPERATURE
"""The dimension of a linear temperature coefficient, K⁻¹."""

PER_KELVIN_SQUARED: Final = DIMENSIONLESS / TEMPERATURE**2
"""The dimension of a quadratic temperature coefficient, K⁻²."""

PER_VOLT: Final = DIMENSIONLESS / VOLTAGE
"""The dimension of a linear voltage coefficient, V⁻¹."""

PER_VOLT_SQUARED: Final = DIMENSIONLESS / VOLTAGE**2
"""The dimension of a quadratic voltage coefficient, V⁻²."""


def resistor(  # noqa: PLR0913 - the value, its tolerance, its drift and its curvature
    resistance: Quantity,
    *,
    tolerance: float | None = None,
    temperature_coefficient: Quantity | None = None,
    quadratic_temperature_coefficient: Quantity | None = None,
    exponential_temperature_coefficient: float | None = None,
    nominal_temperature: Quantity | None = None,
    voltage_coefficient: Quantity | None = None,
    quadratic_voltage_coefficient: Quantity | None = None,
) -> Device:
    """A resistor between terminals ``a`` and ``b``.

    A tolerance, such as 0.01 for 1 %, spreads its resistance evenly within it, as Monte
    Carlo analyses sample it. Its resistance drifts with temperature, by
    ``1 + tc1 · ΔT + tc2 · ΔT²`` from its value at the nominal temperature, 27 °C unless
    given, with the linear and quadratic temperature coefficients, none unless given.
    An exponential temperature coefficient, in percent per kelvin as SPICE's ``TCE``,
    drifts it by ``1.01^(tce · ΔT)`` instead.

    Its resistance also changes with the voltage ``v`` from ``a`` to ``b``, by
    ``1 + vc1 · v + vc2 · v²``, with the linear and quadratic voltage coefficients, none
    unless given; where they bend it below zero, it is negative there, as in ngspice.

    Examples:
        >>> from jolty.units import K, V, kohm
        >>> resistor(1 * kohm, temperature_coefficient=1e-4 / K).parameters
        {'resistance': 1000.0, 'temperature_coefficient': 0.0001}
        >>> resistor(1 * kohm, voltage_coefficient=-2e-6 / V).parameters
        {'resistance': 1000.0, 'voltage_coefficient': -2e-06}
    """
    parameters: dict[str, JsonValue] = {
        "resistance": resistance.to(RESISTANCE, "resistance")
    }
    parameters |= _drift(
        temperature_coefficient, quadratic_temperature_coefficient, nominal_temperature
    )
    parameters |= _parameters(
        {}, {"exponential_temperature_coefficient": exponential_temperature_coefficient}
    )
    parameters |= _curvature(voltage_coefficient, quadratic_voltage_coefficient)
    made = device("resistor", parameters)
    return made if tolerance is None else made.toleranced("resistance", tolerance)


def capacitor(  # noqa: PLR0913 - the value, its tolerance, its drift and its curvature
    capacitance: Quantity,
    *,
    tolerance: float | None = None,
    temperature_coefficient: Quantity | None = None,
    quadratic_temperature_coefficient: Quantity | None = None,
    nominal_temperature: Quantity | None = None,
    voltage_coefficient: Quantity | None = None,
    quadratic_voltage_coefficient: Quantity | None = None,
) -> Device:
    """A capacitor between terminals ``a`` and ``b``.

    Its capacitance drifts with temperature, and spreads within its tolerance, as a
    resistor's resistance does. It also changes with the voltage ``v`` from ``a`` to
    ``b``, by ``1 + vc1 · v + vc2 · v²``, and its charge is that capacitance's integral
    from 0 V.
    """
    parameters: dict[str, JsonValue] = {
        "capacitance": capacitance.to(CAPACITANCE, "capacitance")
    }
    parameters |= _drift(
        temperature_coefficient, quadratic_temperature_coefficient, nominal_temperature
    )
    parameters |= _curvature(voltage_coefficient, quadratic_voltage_coefficient)
    made = device("capacitor", parameters)
    return made if tolerance is None else made.toleranced("capacitance", tolerance)


def inductor(
    inductance: Quantity,
    *,
    tolerance: float | None = None,
    temperature_coefficient: Quantity | None = None,
    quadratic_temperature_coefficient: Quantity | None = None,
    nominal_temperature: Quantity | None = None,
) -> Device:
    """A linear inductor between terminals ``a`` and ``b``.

    Its current, from ``a`` through the inductor to ``b``, is the unknown ``current``.
    Its inductance drifts with temperature, and spreads within its tolerance, as a
    resistor's resistance does.
    """
    parameters: dict[str, JsonValue] = {
        "inductance": inductance.to(INDUCTANCE, "inductance")
    }
    parameters |= _drift(
        temperature_coefficient, quadratic_temperature_coefficient, nominal_temperature
    )
    made = device("inductor", parameters)
    return made if tolerance is None else made.toleranced("inductance", tolerance)


def _drift(
    linear: Quantity | None, quadratic: Quantity | None, nominal: Quantity | None
) -> dict[str, JsonValue]:
    """A passive device's temperature coefficients and nominal temperature, as given."""
    return _parameters(
        {
            "temperature_coefficient": (linear, PER_KELVIN),
            "quadratic_temperature_coefficient": (quadratic, PER_KELVIN_SQUARED),
            "nominal_temperature": (nominal, TEMPERATURE),
        },
        {},
    )


def _curvature(
    linear: Quantity | None, quadratic: Quantity | None
) -> dict[str, JsonValue]:
    """A passive device's voltage coefficients, as given."""
    return _parameters(
        {
            "voltage_coefficient": (linear, PER_VOLT),
            "quadratic_voltage_coefficient": (quadratic, PER_VOLT_SQUARED),
        },
        {},
    )


def voltage_source(waveform: Waveform | Quantity) -> Device:
    """An independent voltage source, ``positive`` minus ``negative`` at its level.

    A quantity is a constant level. The source's current, from ``positive`` through
    the source to ``negative``, is the unknown ``current``: negative while it delivers
    power.
    """
    level = Dc(waveform) if isinstance(waveform, Quantity) else waveform
    parameters: dict[str, JsonValue] = {"waveform": level.to_json(VOLTAGE)}
    return device("voltage_source", parameters)


def current_source(waveform: Waveform | Quantity) -> Device:
    """An independent current source, driving its level from ``positive`` to the other.

    The current flows through the source from ``positive`` to ``negative``, so it enters
    the circuit at the ``negative`` terminal. A quantity is a constant level.
    """
    level = Dc(waveform) if isinstance(waveform, Quantity) else waveform
    parameters: dict[str, JsonValue] = {"waveform": level.to_json(CURRENT)}
    return device("current_source", parameters)


def voltage_controlled_voltage_source(gain: float) -> Device:
    """A source holding ``positive`` minus ``negative`` at `gain` times a voltage.

    That control voltage is ``control_positive`` minus ``control_negative``, and the
    control terminals draw no current.
    """
    parameters: dict[str, JsonValue] = {"gain": finite(gain, "gain")}
    return device("voltage_controlled_voltage_source", parameters)


def voltage_controlled_current_source(transconductance: Quantity) -> Device:
    """A source driving `transconductance` times the control voltage through itself.

    The current flows from ``positive`` through the source to ``negative``; the control
    voltage is ``control_positive`` minus ``control_negative``.
    """
    value = transconductance.to(CONDUCTANCE, "transconductance")
    parameters: dict[str, JsonValue] = {"transconductance": value}
    return device("voltage_controlled_current_source", parameters)


def current_controlled_current_source(gain: float) -> Device:
    """A source driving `gain` times a sensed current through itself.

    The control port, from ``control_positive`` to ``control_negative``, is a short
    circuit that senses the current through it, the unknown ``control``; placed in
    series with a branch, it senses that branch's current. The output current flows
    from ``positive`` through the source to ``negative``.
    """
    parameters: dict[str, JsonValue] = {"gain": finite(gain, "gain")}
    return device("current_controlled_current_source", parameters)


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
    return device("current_controlled_voltage_source", parameters)


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
    return device("behavioral_voltage_source", {"expression": expression})


def behavioral_current_source(expression: str) -> Device:
    """A source driving `expression` from ``positive`` through itself to ``negative``.

    The expression and the terminals are a behavioral voltage source's.

    Raises:
        ValueError: When `expression` is not an expression.

    Examples:
        >>> behavioral_current_source("limit(1e-3 * v(c), -1e-3, 1e-3)").terminals
        ('positive', 'negative', 'c')
    """
    return device("behavioral_current_source", {"expression": expression})


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
    return device("voltage_controlled_switch", parameters)


def photocell(
    resistance: Quantity,
    *,
    illuminance: float | None = None,
    gamma: float | None = None,
    dark_resistance: Quantity | None = None,
    response_time: Quantity | None = None,
) -> Device:
    """A photocell, a light-dependent resistor, between ``a`` and ``b``.

    The voltage of ``light_positive`` over ``light_negative`` is the illuminance on the
    cell, 1 V for each lux, and those terminals draw no current. Its resistance is
    `resistance` at the reference `illuminance`, in lux, 10 unless given, as datasheets
    state it, and falls as the illuminance to the power `gamma`, 0.7 unless given, in
    parallel with its dark resistance, 1 MΩ unless given. With a response time, its
    conductance follows the light with that time constant, as a cadmium sulphide cell's
    does, over milliseconds; without one, at once.

    Examples:
        >>> from jolty.units import kohm, ms
        >>> cell = photocell(10 * kohm, gamma=0.8, response_time=20 * ms)
        >>> cell.parameters["response_time"]
        0.02
    """
    parameters = _parameters(
        {
            "resistance": (resistance, RESISTANCE),
            "dark_resistance": (dark_resistance, RESISTANCE),
            "response_time": (response_time, TIME),
        },
        {"illuminance": illuminance, "gamma": gamma},
    )
    return device("photocell", parameters)


def diode(  # noqa: PLR0913 - the SPICE junction model has fifteen parameters
    *,
    saturation_current: Quantity | None = None,
    emission_coefficient: float | None = None,
    series_resistance: Quantity | None = None,
    junction_capacitance: Quantity | None = None,
    junction_potential: Quantity | None = None,
    grading_coefficient: float | None = None,
    depletion_coefficient: float | None = None,
    transit_time: Quantity | None = None,
    breakdown_voltage: Quantity | None = None,
    breakdown_current: Quantity | None = None,
    breakdown_emission_coefficient: float | None = None,
    breakdown_temperature_coefficient: Quantity | None = None,
    energy_gap: Quantity | None = None,
    saturation_current_temperature_exponent: float | None = None,
    nominal_temperature: Quantity | None = None,
) -> Device:
    """A junction diode from ``anode`` to ``cathode``, in the SPICE model.

    Parameters left out take SPICE's defaults: a saturation current of 10 fA, an
    emission coefficient of 1, no series resistance, junction capacitance or transit
    time, a junction potential of 1 V, and grading and depletion coefficients of 0.5.
    Given a breakdown voltage, the reverse current grows exponentially beyond it, with
    a breakdown current of 1 mA there and the emission coefficient unless given.

    Away from the nominal temperature, 27 °C unless given, the saturation current scales
    with the energy gap, 1.11 V unless given, and the temperature exponent, 3 unless
    given; the junction potential and capacitance follow silicon's band gap, and the
    breakdown voltage falls by its temperature coefficient, none unless given, as in
    ngspice.
    """
    parameters = _parameters(
        {
            "saturation_current": (saturation_current, CURRENT),
            "series_resistance": (series_resistance, RESISTANCE),
            "junction_capacitance": (junction_capacitance, CAPACITANCE),
            "junction_potential": (junction_potential, VOLTAGE),
            "transit_time": (transit_time, TIME),
            "breakdown_voltage": (breakdown_voltage, VOLTAGE),
            "breakdown_current": (breakdown_current, CURRENT),
            "breakdown_temperature_coefficient": (
                breakdown_temperature_coefficient,
                VOLTS_PER_KELVIN,
            ),
            "energy_gap": (energy_gap, VOLTAGE),
            "nominal_temperature": (nominal_temperature, TEMPERATURE),
        },
        {
            "emission_coefficient": emission_coefficient,
            "grading_coefficient": grading_coefficient,
            "depletion_coefficient": depletion_coefficient,
            "breakdown_emission_coefficient": breakdown_emission_coefficient,
            "saturation_current_temperature_exponent": (
                saturation_current_temperature_exponent
            ),
        },
    )
    return device("diode", parameters)


TRANSCONDUCTANCE_PARAMETER: Final = CURRENT / VOLTAGE**2
"""The dimension of a field-effect transistor's transconductance parameter, A/V²."""

PER_VOLT: Final = DIMENSIONLESS / VOLTAGE
"""The dimension of a channel-length modulation, V⁻¹."""

MOBILITY: Final = LENGTH**2 / (VOLTAGE * TIME)
"""The dimension of a carrier mobility, m²/(V·s)."""

VOLTS_PER_KELVIN: Final = VOLTAGE / TEMPERATURE
"""The dimension of a voltage's temperature coefficient, V/K."""


def jfet(  # noqa: PLR0913 - the SPICE JFET model has seventeen parameters
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
    threshold_temperature_coefficient: Quantity | None = None,
    beta_temperature_exponent: float | None = None,
    energy_gap: Quantity | None = None,
    saturation_current_temperature_exponent: float | None = None,
    nominal_temperature: Quantity | None = None,
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

    Away from the nominal temperature, 27 °C unless given, the gate junctions'
    saturation current scales with the energy gap, 1.11 V unless given, and the
    temperature exponent, none unless given; their potential and capacitances follow
    silicon's band gap; the threshold moves by its temperature coefficient and the
    transconductance parameter scales by its temperature exponent, neither unless given,
    as in ngspice.

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
            "threshold_temperature_coefficient": (
                threshold_temperature_coefficient,
                VOLTS_PER_KELVIN,
            ),
            "energy_gap": (energy_gap, VOLTAGE),
            "nominal_temperature": (nominal_temperature, TEMPERATURE),
        },
        {
            "depletion_coefficient": depletion_coefficient,
            "doping_tail": doping_tail,
            "beta_temperature_exponent": beta_temperature_exponent,
            "saturation_current_temperature_exponent": (
                saturation_current_temperature_exponent
            ),
        },
    )
    return device("jfet", parameters)


def bipolar_transistor(  # noqa: PLR0913 - the Gummel-Poon model has 42 parameters
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
    substrate: Literal["vertical", "lateral"] | None = None,
    substrate_saturation_current: Quantity | None = None,
    substrate_emission_coefficient: float | None = None,
    substrate_capacitance: Quantity | None = None,
    substrate_potential: Quantity | None = None,
    substrate_grading: float | None = None,
    energy_gap: Quantity | None = None,
    saturation_current_temperature_exponent: float | None = None,
    beta_temperature_exponent: float | None = None,
    nominal_temperature: Quantity | None = None,
) -> Device:
    """A bipolar junction transistor between ``collector``, ``base`` and ``emitter``.

    It follows the Gummel-Poon model of SPICE, as ngspice does. Parameters left out take
    SPICE's defaults: an NPN with a saturation current of 0.1 fA, a forward beta of 100
    and a reverse beta of 1, emission coefficients of 1, infinite Early voltages and
    knee currents, no leakage, resistance, capacitance or transit time, and junction
    potentials of 0.75 V with grading coefficients of 0.33. A base, collector or emitter
    resistance adds an internal node, the unknown ``base``, ``collector`` or
    ``emitter``.

    A transistor built ``vertical`` or ``lateral`` has a fourth terminal, ``substrate``,
    joined by a junction to its collector if vertical or to its base if lateral. The
    junction conducts the minimum conductance, and a diode's current with a substrate
    saturation current; its depletion capacitance grows linearly in forward bias. The
    substrate parameters need a substrate.

    Away from the nominal temperature, 27 °C unless given, the saturation currents scale
    with the energy gap, 1.11 V unless given, and the temperature exponent, 3 unless
    given; the betas scale by their temperature exponent, none unless given, and the
    junctions' potentials and capacitances follow silicon's band gap, as in ngspice.

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
    parameters: dict[str, JsonValue] = {
        name: value
        for name, value in (("polarity", polarity), ("substrate", substrate))
        if value is not None
    }
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
            "substrate_saturation_current": (substrate_saturation_current, CURRENT),
            "substrate_capacitance": (substrate_capacitance, CAPACITANCE),
            "substrate_potential": (substrate_potential, VOLTAGE),
            "energy_gap": (energy_gap, VOLTAGE),
            "nominal_temperature": (nominal_temperature, TEMPERATURE),
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
            "substrate_emission_coefficient": substrate_emission_coefficient,
            "substrate_grading": substrate_grading,
            "saturation_current_temperature_exponent": (
                saturation_current_temperature_exponent
            ),
            "beta_temperature_exponent": beta_temperature_exponent,
        },
    )
    return device("bipolar_transistor", parameters)


def mosfet(  # noqa: PLR0913 - SPICE's level 1 and 3 MOSFETs have 50 parameters
    *,
    level: Literal[1, 3] | None = None,
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
    nominal_temperature: Quantity | None = None,
    max_drift_velocity: Quantity | None = None,
    junction_depth: Quantity | None = None,
    fast_surface_state_density: Quantity | None = None,
    static_feedback: float | None = None,
    narrow_width_factor: float | None = None,
    mobility_modulation: Quantity | None = None,
    saturation_field_factor: float | None = None,
    width_narrowing: Quantity | None = None,
    length_adjustment: Quantity | None = None,
    width_adjustment: Quantity | None = None,
    threshold_adjustment: Quantity | None = None,
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

    Away from the nominal temperature, 27 °C unless given, the transconductance
    parameter and the mobility fall as the temperature to the power -1.5, the surface
    potential follows silicon's band gap and the threshold with it, and the junctions'
    saturation currents, potential and capacitances follow the band gap too, as in
    ngspice.

    At ``level=3``, the channel follows SPICE's semi-empirical law of short channels
    instead, and the parameters from ``max_drift_velocity`` on, level 3's alone, model
    velocity saturation, short and narrow channels' thresholds, the drain's static
    feedback, the gate field's degradation of the mobility, the channel's shortening in
    saturation, weak inversion, and the mask's adjustments to the channel. A level 3
    transistor has no channel-length modulation, an oxide 0.1 µm thick unless given, and
    a sidewall grading coefficient of 0.33.

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
    parameters: dict[str, JsonValue] = {
        name: value
        for name, value in (
            ("level", level),
            ("channel", channel),
            ("gate_material", gate_material),
        )
        if value is not None
    }
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
            "nominal_temperature": (nominal_temperature, TEMPERATURE),
            "max_drift_velocity": (max_drift_velocity, LENGTH / TIME),
            "junction_depth": (junction_depth, LENGTH),
            "fast_surface_state_density": (
                fast_surface_state_density,
                DIMENSIONLESS / LENGTH**2,
            ),
            "mobility_modulation": (mobility_modulation, PER_VOLT),
            "width_narrowing": (width_narrowing, LENGTH),
            "length_adjustment": (length_adjustment, LENGTH),
            "width_adjustment": (width_adjustment, LENGTH),
            "threshold_adjustment": (threshold_adjustment, VOLTAGE),
        },
        {
            "multiplier": multiplier,
            "static_feedback": static_feedback,
            "narrow_width_factor": narrow_width_factor,
            "saturation_field_factor": saturation_field_factor,
            "body_effect": body_effect,
            "drain_squares": drain_squares,
            "source_squares": source_squares,
            "bottom_grading": bottom_grading,
            "sidewall_grading": sidewall_grading,
            "depletion_coefficient": depletion_coefficient,
        },
    )
    return device("mosfet", parameters)


type LogicLevel = Literal[0, 1] | None
"""A logic level: 0, 1, or None for an unknown level."""

type Gate = Literal["and", "or", "nand", "nor", "xor", "xnor", "not", "buffer"]
"""The function of a logic gate."""

_GATES: Final[dict[str, Callable[[Sequence[bool]], bool]]] = {
    "and": all,
    "or": any,
    "nand": lambda inputs: not all(inputs),
    "nor": lambda inputs: not any(inputs),
    "xor": lambda inputs: sum(inputs) % 2 == 1,
    "xnor": lambda inputs: sum(inputs) % 2 == 0,
    "not": lambda inputs: not inputs[0],
    "buffer": lambda inputs: inputs[0],
}


def logic_cell(  # noqa: PLR0913 - the function, its timing and its electrical pins
    inputs: Sequence[str],
    outputs: Mapping[str, str | Callable[..., bool]],
    *,
    rise: Quantity,
    fall: Quantity | None = None,
    transition: Quantity,
    electrical: bool = False,
    input_capacitance: Quantity | None = None,
    threshold: float = 0.5,
) -> Device:
    """A combinational logic cell: each output a function of the inputs.

    Each output is its truth table, a ``0`` or a ``1`` for each row of the inputs, the
    first input the most significant bit, so a NAND of ``a`` and ``b`` is ``"1110"``;
    or a function of the inputs' levels as booleans, such as ``lambda a, b: not (a and
    b)``. An output fed unknown inputs is known only when every level they could take
    gives it the same level.

    An output reaches a new level `rise` after an input changes when it rises, and
    `fall` when it falls, `rise` unless given, and its edges take `transition` to swing
    from ground to supply. Its delay is inertial: a pulse at the inputs shorter than
    the delay never reaches it.

    The cell's pins join logic nets, which an event scheduler simulates. An
    `electrical` cell has the terminals ``supply`` and ``ground`` too, and its pins may
    join electrical nets: each input then reads its net against `threshold` of the
    supply, from ground, and loads it with `input_capacitance`, zero unless given, and
    each output drives its net between ground and supply in a ramp, crossing half the
    supply when its level changes.

    Examples:
        >>> from jolty.units import ps
        >>> nand = logic_cell(
        ...     ["a", "b"],
        ...     {"y": lambda a, b: not (a and b)},
        ...     rise=20 * ps,
        ...     transition=10 * ps,
        ... )
        >>> nand.terminals, nand.parameters["outputs"]
        (('a', 'b', 'y'), [{'name': 'y', 'table': '1110'}])

    Raises:
        DimensionError: When a delay, the transition or the capacitance is not one.
        ValueError: When a capacitance or a threshold is given to a cell that is not
            electrical, or the engine cannot read the cell.
    """
    tables: list[JsonValue] = [
        {"name": name, "table": _table(function, len(inputs))}
        for name, function in outputs.items()
    ]
    timing: dict[str, JsonValue] = {
        "rise": rise.to(TIME, "a rise delay"),
        "fall": (rise if fall is None else fall).to(TIME, "a fall delay"),
        "transition": transition.to(TIME, "a transition"),
    }
    parameters: dict[str, JsonValue] = {
        "inputs": list(inputs),
        "outputs": tables,
        "timing": {"fixed": timing},
    }
    if electrical:
        capacitance = (
            0.0
            if input_capacitance is None
            else input_capacitance.to(CAPACITANCE, "an input capacitance")
        )
        parameters["electrical"] = {
            "threshold": finite(threshold, "threshold"),
            "input_capacitances": [capacitance] * len(inputs),
        }
    elif input_capacitance is not None or threshold != 0.5:  # noqa: PLR2004 - the default
        message = "only an electrical cell has an input capacitance and a threshold"
        raise ValueError(message)
    return device("logic_cell", parameters)


def _table(function: str | Callable[..., bool], inputs: int) -> str:
    """The truth table of `function`: a table, or a function of `inputs` levels."""
    if isinstance(function, str):
        return function
    rows = (
        [bool(row >> (inputs - 1 - bit) & 1) for bit in range(inputs)]
        for row in range(1 << inputs)
    )
    return "".join("1" if function(*row) else "0" for row in rows)


def logic_gate(  # noqa: PLR0913 - a logic cell's parameters
    gate: Gate,
    inputs: int = 2,
    *,
    rise: Quantity,
    fall: Quantity | None = None,
    transition: Quantity,
    electrical: bool = False,
    input_capacitance: Quantity | None = None,
    threshold: float = 0.5,
) -> Device:
    """A logic gate of `gate`'s function, a [`logic_cell`][jolty.devices.logic_cell].

    Its inputs are ``a``, ``b`` and on, `inputs` of them, or one for ``not`` and
    ``buffer``, and its output is ``y``. The other parameters are a logic cell's.

    Examples:
        >>> from jolty.units import ps
        >>> logic_gate("xor", rise=20 * ps, transition=10 * ps).parameters["outputs"]
        [{'name': 'y', 'table': '0110'}]
    """
    count = 1 if gate in {"not", "buffer"} else inputs
    names = [chr(ord("a") + index) for index in range(count)]
    function = _GATES[gate]
    return logic_cell(
        names,
        {"y": lambda *levels: function(levels)},
        rise=rise,
        fall=fall,
        transition=transition,
        electrical=electrical,
        input_capacitance=input_capacitance,
        threshold=threshold,
    )


def logic_source(
    initial: LogicLevel,
    changes: Sequence[tuple[Quantity, LogicLevel]] = (),
    *,
    transition: Quantity,
) -> Device:
    """A logic source: one output, ``output``, at `initial`, then at each of `changes`.

    Each change is a time and the level from then on, in time order; each edge takes
    `transition` to swing from ground to supply, and crosses half the supply at its
    time.

    Examples:
        >>> from jolty.units import ns, ps
        >>> clock = logic_source(0, [(1 * ns, 1), (2 * ns, 0)], transition=50 * ps)
        >>> clock.parameters["changes"]
        [[1e-09, '1'], [2e-09, '0']]
    """
    parameters: dict[str, JsonValue] = {
        "initial": _written(initial),
        "changes": [
            [time.to(TIME, "a change's time"), _written(level)]
            for time, level in changes
        ],
        "transition": transition.to(TIME, "a transition"),
    }
    return device("logic_source", parameters)


def _written(level: LogicLevel) -> str:
    """A logic level as the engine writes it: ``0``, ``1`` or ``x``."""
    return "x" if level is None else str(level)


def device(model: str, parameters: dict[str, JsonValue]) -> Device:
    """A device of `model` with `parameters`, and the terminals the engine gives it.

    The parameters are numbers in SI base units. The functions above check each
    parameter's dimension; this one takes numbers as they are, for devices read from
    elsewhere, such as a SPICE library.

    Raises:
        ValueError: When no model is named `model`, or the engine cannot read the
            parameters.
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
