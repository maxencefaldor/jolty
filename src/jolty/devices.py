# Copyright 2026 Maxence Faldor
"""The devices a build script places, each a model, its terminals and its parameters.

Each function checks the dimension of every parameter it reads, so a capacitance
passed where a resistance belongs fails on the line that passed it. Parameters left out
take the engine's defaults, which the reference's models page lists with every
parameter of every model.

Examples:
    >>> from jolty.units import kohm
    >>> resistor(10 * kohm).parameters
    {'resistance': 10000.0}
"""

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Final, Literal

from pydantic import JsonValue

from jolty import _core, ir
from jolty.errors import DimensionError
from jolty.units import (
    CAPACITANCE,
    CHARGE,
    CONDUCTANCE,
    CURRENT,
    DIMENSIONLESS,
    FREQUENCY,
    HEAT_CAPACITY,
    INDUCTANCE,
    LENGTH,
    POWER,
    RESISTANCE,
    TEMPERATURE,
    THERMAL_RESISTANCE,
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

    Some parameters may have a tolerance, which Monte Carlo analyses sample, and the
    device may state ratings, which a simulation checks, and the number of the part it
    stands for, which a bill of materials prices.
    """

    model: str
    terminals: tuple[str, ...]
    parameters: dict[str, JsonValue]
    tolerances: dict[str, ir.Tolerance] = field(default_factory=dict)
    ratings: tuple[ir.Rating, ...] = ()
    part: str | None = None

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

    def rated(
        self,
        stress: ir.Stress,
        maximum: Quantity | None = None,
        *,
        minimum: Quantity | None = None,
        terminals: Sequence[str] | None = None,
    ) -> Device:
        """The device, with a rating its maker states on one of its `stress`es.

        A voltage is of the first of its two `terminals` over the second, a current is
        into its one terminal, and the power and the temperature name none; a voltage
        is of the device's first two terminals by default, in order, and a current into
        its first. The power is what its electrical terminals take, and the temperature
        the simulation's, raised by its thermal net's where it is
        [heated][jolty.devices.Device.heated]. A rating bounds its stress from above,
        from below or both: a capacitor rated 50 V either way takes
        ``maximum=50 * V, minimum=-50 * V``. A simulation checks every rating at the
        operating point and over a transient.

        Examples:
            >>> from jolty.units import V, W, celsius, kohm, uF
            >>> resistor(1 * kohm).rated("power", 0.25 * W).ratings[0].maximum
            0.25
            >>> capacitor(1 * uF).rated("voltage", 50 * V).ratings[0].terminals
            ('a', 'b')
            >>> diode().rated("temperature", celsius(175)).ratings[0].maximum
            448.15

        Raises:
            DimensionError: When a bound is not a voltage, a current, a power or a
                temperature as the stress is.
            ValueError: When the terminals are not the device's, or not as many as the
                stress needs, or the bounds are none or out of order.
        """
        rating = rate(
            self.terminals,
            stress,
            maximum=maximum,
            minimum=minimum,
            terminals=terminals,
        )
        return replace(self, ratings=(*self.ratings, rating))

    def heated(self) -> Device:
        """The device, heated by the power it dissipates.

        A terminal ``thermal`` joins it to a thermal net, or to the ground, the ambient.
        The device runs at the simulation's temperature raised by that net's, and the
        power its electrical terminals take, each one's voltage times the current into
        it, flows into the net as heat. A resistor, a diode, a JFET, a bipolar
        transistor, a MOSFET, a BSIM3 and a VDMOS heat.

        Examples:
            >>> diode().heated().terminals
            ('anode', 'cathode', 'thermal')

        Raises:
            ValueError: When the device's model does not heat, or the engine cannot
                read its parameters.
        """
        _core.terminals(self.model, json.dumps(self.parameters))
        parameters: dict[str, JsonValue] = {**self.parameters, "thermal": True}
        try:
            terminals = _core.terminals(self.model, json.dumps(parameters))
        except ValueError:
            message = f"a {self.model} does not heat"
            raise ValueError(message) from None
        return replace(self, terminals=tuple(terminals), parameters=parameters)

    def held_at(self, temperature: Quantity) -> Device:
        """The device, held at `temperature` whatever the simulation's.

        Its parameters scale to it as they scale to the simulation's, as SPICE's
        ``TEMP`` and PSpice's ``T_ABS`` hold a device, and a heated device warms from
        it: a resistor held at 0 K makes no noise, while a junction needs a temperature
        above 0 K. It is no parameter that sensitivities differentiate.

        Examples:
            >>> from jolty.units import celsius, kohm
            >>> resistor(1 * kohm).held_at(celsius(85)).parameters["temperature"]
            358.15

        Raises:
            DimensionError: When `temperature` is not one.
            ValueError: When it lies below 0 K, or the device is a logic device, whose
                timing holds whatever its temperature.
        """
        kelvins = temperature.to(TEMPERATURE, "a device's own temperature")
        if kelvins < 0:
            message = (
                f"a device's own temperature lies at 0 K or above, not {kelvins!r} K"
            )
            raise ValueError(message)
        return self._surrounded("temperature", kelvins)

    def warmer_by(self, offset: Quantity) -> Device:
        """The device, `offset` warmer than the simulation, or cooler where negative.

        It is as [`held_at`][jolty.devices.Device.held_at] holds it, but follows the
        simulation's temperature, as SPICE's ``DTEMP`` and PSpice's ``T_REL_GLOBAL``
        do.

        Examples:
            >>> from jolty.units import K, kohm
            >>> resistor(1 * kohm).warmer_by(20 * K).parameters["temperature_offset"]
            20.0

        Raises:
            DimensionError: When `offset` is not a temperature.
            ValueError: When the device is a logic device, whose timing holds whatever
                its temperature.
        """
        kelvins = offset.to(TEMPERATURE, "a device's temperature offset")
        return self._surrounded("temperature_offset", finite(kelvins, "offset"))

    def _surrounded(self, key: str, kelvins: float) -> Device:
        """The device at a temperature of its own, `kelvins` as `key` gives it.

        Raises:
            ValueError: When the device's model takes no temperature of its own.
        """
        kept = {
            name: value
            for name, value in self.parameters.items()
            if name not in {"temperature", "temperature_offset"}
        }
        parameters: dict[str, JsonValue] = {**kept, key: kelvins}
        _core.terminals(self.model, json.dumps(kept))
        try:
            _core.terminals(self.model, json.dumps(parameters))
        except ValueError:
            message = f"a {self.model} runs at no temperature of its own"
            raise ValueError(message) from None
        return replace(self, parameters=parameters)

    def numbered(self, part: str) -> Device:
        """The device, standing for the part its maker numbers `part`.

        Examples:
            >>> diode().numbered("1N4148").part
            '1N4148'

        Raises:
            ValueError: When the number is empty or spaced at either end.
        """
        return replace(self, part=ir.part_number(part))


def rate(
    own: Sequence[str],
    stress: ir.Stress,
    *,
    maximum: Quantity | None,
    minimum: Quantity | None,
    terminals: Sequence[str] | None,
) -> ir.Rating:
    """A rating of the `stress` of a device or a component whose terminals are `own`.

    It is as [`Device.rated`][jolty.devices.Device.rated] describes.

    Raises:
        DimensionError: When a bound is not a voltage, a current, a power or a
            temperature as the stress is.
        ValueError: When the terminals are not `own`, or not as many as the stress
            needs, or the bounds are none or out of order.
    """
    dimension, defaults = {
        "voltage": (VOLTAGE, own[:2]),
        "current": (CURRENT, own[:1]),
        "power": (POWER, ()),
        "temperature": (TEMPERATURE, ()),
    }[stress]
    named = tuple(defaults if terminals is None else terminals)
    unknown = [terminal for terminal in named if terminal not in own]
    if unknown:
        message = (
            f"a rating names {', '.join(unknown)}, which is not among the terminals "
            f"{', '.join(own)}"
        )
        raise ValueError(message)
    return ir.Rating(
        stress=stress,
        terminals=named,
        maximum=None
        if maximum is None
        else maximum.to(dimension, f"a {stress} rating's maximum"),
        minimum=None
        if minimum is None
        else minimum.to(dimension, f"a {stress} rating's minimum"),
    )


PER_KELVIN: Final = DIMENSIONLESS / TEMPERATURE
"""The dimension of a linear temperature coefficient, K⁻¹."""

PER_KELVIN_SQUARED: Final = DIMENSIONLESS / TEMPERATURE**2
"""The dimension of a quadratic temperature coefficient, K⁻²."""

PER_VOLT: Final = DIMENSIONLESS / VOLTAGE
"""The dimension of a linear voltage coefficient, V⁻¹."""

PER_VOLT_SQUARED: Final = DIMENSIONLESS / VOLTAGE**2
"""The dimension of a quadratic voltage coefficient, V⁻²."""


def resistor(  # noqa: PLR0913 - its value, tolerance, drift, curvature and noise
    resistance: Quantity,
    *,
    tolerance: float | None = None,
    temperature_coefficient: Quantity | None = None,
    quadratic_temperature_coefficient: Quantity | None = None,
    exponential_temperature_coefficient: float | None = None,
    nominal_temperature: Quantity | None = None,
    voltage_coefficient: Quantity | None = None,
    quadratic_voltage_coefficient: Quantity | None = None,
    flicker_coefficient: float | None = None,
    flicker_exponent: float | None = None,
) -> Device:
    """A resistor between terminals ``a`` and ``b``.

    A tolerance, such as 0.01 for 1 %, spreads its resistance evenly within it, as Monte
    Carlo analyses sample it. Its resistance drifts with temperature, by
    ``1 + tc1 · ΔT + tc2 · ΔT²`` from its value at the nominal temperature, with the
    linear and quadratic temperature coefficients. An exponential temperature
    coefficient, in percent per kelvin as SPICE's ``TCE``, drifts it by
    ``1.01^(tce · ΔT)`` instead.

    Its resistance also changes with the voltage ``v`` from ``a`` to ``b``, by
    ``1 + vc1 · v + vc2 · v²``, with the linear and quadratic voltage coefficients;
    where they bend it below zero, it is negative there, as in ngspice.

    Beside its thermal noise, with a flicker coefficient ``KF``, its current ``I`` makes
    flicker noise of ``KF · |I|^AF / f`` A²/Hz across it, ``AF`` the flicker exponent,
    as ngspice's resistor makes it.

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
    parameters |= _curvature(voltage_coefficient, quadratic_voltage_coefficient)
    parameters |= _parameters(
        {},
        {
            "exponential_temperature_coefficient": exponential_temperature_coefficient,
            "flicker_coefficient": flicker_coefficient,
            "flicker_exponent": flicker_exponent,
        },
    )
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


def coupled_inductors(
    windings: Sequence[Device], couplings: Mapping[tuple[int, int], float]
) -> Device:
    """Inductors whose fluxes link, as a transformer's windings or a choke's.

    Each winding is an [`inductor`][jolty.devices.inductor], with its tolerance and
    drift; winding ``k``, from 1, joins terminals ``a{k}`` and ``b{k}``, and its
    current, from ``a{k}`` through it to ``b{k}``, is the unknown ``current{k}``.
    `couplings` gives the coefficient that links two windings, by their numbers, from -1
    to 1: their mutual inductance is the coefficient times the root of their
    inductances' product, and currents into both ``a`` terminals add their fluxes where
    it is positive, as SPICE's ``K`` couples its inductors' first nodes. There are at
    least two windings, and a pair is coupled once at most.

    Examples:
        >>> from jolty.units import mH
        >>> transformer = coupled_inductors(
        ...     [inductor(1 * mH), inductor(4 * mH)], {(1, 2): 0.99}
        ... )
        >>> transformer.terminals
        ('a1', 'b1', 'a2', 'b2')

    Raises:
        ValueError: When a winding is no inductor, states ratings or a part number, or
            runs at a temperature of its own, which the coupled inductors take as a
            whole, as [`held_at`][jolty.devices.Device.held_at] sets it.
    """
    tolerances: dict[str, ir.Tolerance] = {}
    for number, winding in enumerate(windings, start=1):
        if winding.model != "inductor" or winding.ratings or winding.part is not None:
            message = f"winding {number} is an inductor that states no ratings or part"
            raise ValueError(message)
        if {"temperature", "temperature_offset"} & winding.parameters.keys():
            message = (
                f"winding {number} runs at a temperature of its own, which the coupled "
                "inductors take as a whole"
            )
            raise ValueError(message)
        tolerances |= {
            f"windings.{number - 1}.{path}": tolerance
            for path, tolerance in winding.tolerances.items()
        }
    coupled: list[JsonValue] = [
        {"windings": [first, second], "coefficient": finite(coefficient, "coupling")}
        for (first, second), coefficient in couplings.items()
    ]
    parameters: dict[str, JsonValue] = {
        "windings": [dict(winding.parameters) for winding in windings],
        "couplings": coupled,
    }
    made = device("coupled_inductors", parameters)
    return replace(made, tolerances=tolerances)


def transmission_line(characteristic_impedance: Quantity, delay: Quantity) -> Device:
    """A lossless transmission line, SPICE's ``T``, whose waves cross it in `delay`.

    Port ``k``, 1 or 2, joins ``positive{k}`` and ``negative{k}``, and its current,
    from ``positive{k}`` into the line and out at ``negative{k}``, is the unknown
    ``current{k}``. Each port is a voltage behind the `characteristic_impedance`, the
    wave that left the other port a delay before: at DC a short through from port to
    port, and in AC the delay's phase, ``e^{-jωτ}``. A transient steps onto each corner
    of its waveforms as it arrives at the other port.

    Examples:
        >>> from jolty.units import ns, ohm
        >>> line = transmission_line(50 * ohm, 1 * ns)
        >>> line.terminals
        ('positive1', 'negative1', 'positive2', 'negative2')
    """
    parameters = _parameters(
        {
            "characteristic_impedance": (characteristic_impedance, RESISTANCE),
            "delay": (delay, TIME),
        },
        {},
    )
    return device("transmission_line", parameters)


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


def thermal_resistance(resistance: Quantity) -> Device:
    """A thermal resistance between thermal terminals ``a`` and ``b``, in K/W.

    Heat flows through it from the hotter terminal to the colder at their difference
    over its resistance, as current flows through a resistor. A thermal terminal joins
    a thermal net, whose unknown is its temperature above the ambient, or the ground,
    which is the ambient itself.

    Examples:
        >>> from jolty.units import K, W
        >>> thermal_resistance(50 * K / W).terminals
        ('a', 'b')
    """
    return device(
        "thermal_resistance",
        {"resistance": resistance.to(THERMAL_RESISTANCE, "thermal resistance")},
    )


def thermal_capacitance(capacitance: Quantity) -> Device:
    """A heat capacity between thermal terminals ``a`` and ``b``, in J/K.

    It stores heat in proportion to their difference, as a capacitor stores charge, so
    it sets how fast a thermal net's temperature follows the heat into it.
    """
    return device(
        "thermal_capacitance",
        {"capacitance": capacitance.to(HEAT_CAPACITY, "heat capacity")},
    )


def heat_source(power: Quantity) -> Device:
    """A constant heat flow, in watts, from thermal terminal ``a`` into ``b``.

    It drives heat as a current source drives current: a part beside the circuit,
    heating the board under it.
    """
    return device("heat_source", {"power": power.to(POWER, "heat flow")})


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
    terminals draw no current. Parameters left out take PSpice's defaults.
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


def current_controlled_switch(
    *,
    on_resistance: Quantity | None = None,
    off_resistance: Quantity | None = None,
    on_current: Quantity | None = None,
    off_current: Quantity | None = None,
) -> Device:
    """A switch between ``positive`` and ``negative`` that a current moves, as PSpice's.

    Its control port, from ``control_positive`` to ``control_negative``, is a short
    circuit whose current, the unknown ``control``, moves the resistance as a
    [`voltage_controlled_switch`][jolty.devices.voltage_controlled_switch]'s voltage
    does, from the off resistance at the off current to the on resistance at the on
    current. Parameters left out take PSpice's defaults.
    """
    parameters = _parameters(
        {
            "on_resistance": (on_resistance, RESISTANCE),
            "off_resistance": (off_resistance, RESISTANCE),
            "on_current": (on_current, CURRENT),
            "off_current": (off_current, CURRENT),
        },
        {},
    )
    return device("current_controlled_switch", parameters)


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
    `resistance` at the reference `illuminance`, in lux, as datasheets state it, and
    falls as the illuminance to the power `gamma`, in parallel with its dark
    resistance. With a response time, its conductance follows the light with that time
    constant, as a cadmium sulphide cell's does, over milliseconds; without one, at
    once.

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


def voltage_controlled_memristor(  # noqa: PLR0913 - VTEAM's thresholds, rates and exponents
    *,
    on_resistance: Quantity,
    off_resistance: Quantity,
    on_voltage: Quantity,
    off_voltage: Quantity,
    on_rate: Quantity,
    off_rate: Quantity,
    on_exponent: float | None = None,
    off_exponent: float | None = None,
    law: Literal["linear", "exponential"] | None = None,
    state: float | None = None,
    window: float | None = None,
) -> Device:
    """A voltage-controlled memristor between ``positive`` and ``negative``, VTEAM's.

    Its resistance follows its state by its `law`, from `on_resistance` at a state of 0
    to `off_resistance` at 1. A voltage of ``positive`` over ``negative`` above
    `off_voltage` moves the state toward off at `off_rate` times ``(v/v_off - 1)`` to
    the power `off_exponent`, and one below `on_voltage`, a negative voltage, toward on
    at `on_rate` times ``(v/v_on - 1)`` to the power `on_exponent`; between them, the
    state holds. Each rate is the state's speed at twice its threshold, in full ranges
    per second. A `window`, a fraction of the range, slows the state at either bound,
    which it never passes. It is made in `state`.

    Examples:
        >>> from jolty.units import V, kohm, s
        >>> cell = voltage_controlled_memristor(
        ...     on_resistance=1 * kohm,
        ...     off_resistance=100 * kohm,
        ...     on_voltage=-1 * V,
        ...     off_voltage=0.5 * V,
        ...     on_rate=1e6 / s,
        ...     off_rate=1e5 / s,
        ... )
        >>> cell.parameters["off_rate"]
        100000.0
    """
    parameters = _parameters(
        {
            "on_resistance": (on_resistance, RESISTANCE),
            "off_resistance": (off_resistance, RESISTANCE),
            "on_voltage": (on_voltage, VOLTAGE),
            "off_voltage": (off_voltage, VOLTAGE),
            "on_rate": (on_rate, FREQUENCY),
            "off_rate": (off_rate, FREQUENCY),
        },
        {
            "on_exponent": on_exponent,
            "off_exponent": off_exponent,
            "state": state,
            "window": window,
        },
        {"law": law},
    )
    return device("voltage_controlled_memristor", parameters)


def current_controlled_memristor(  # noqa: PLR0913 - its resistances, charge, law and state
    *,
    on_resistance: Quantity,
    off_resistance: Quantity,
    charge: Quantity,
    law: Literal["linear", "exponential"] | None = None,
    state: float | None = None,
    window: float | None = None,
) -> Device:
    """A current-controlled memristor between ``positive`` and ``negative``, HP's.

    Its resistance follows its state by its `law`, from `on_resistance` at a state of 0
    to `off_resistance` at 1. The state moves with the charge through it, from
    ``positive`` to ``negative`` toward off, by the charge over `charge`, the charge
    that carries it across its whole range, as the dopants of HP's titanium dioxide
    cell drift. A `window`, a fraction of the range, slows the state at either bound,
    which it never passes. It is made in `state`.

    Examples:
        >>> from jolty.units import kohm, ohm, uC
        >>> cell = current_controlled_memristor(
        ...     on_resistance=100 * ohm, off_resistance=16 * kohm, charge=2 * uC
        ... )
        >>> cell.parameters["charge"]
        2e-06
    """
    parameters = _parameters(
        {
            "on_resistance": (on_resistance, RESISTANCE),
            "off_resistance": (off_resistance, RESISTANCE),
            "charge": (charge, CHARGE),
        },
        {"state": state, "window": window},
        {"law": law},
    )
    return device("current_controlled_memristor", parameters)


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
    flicker_coefficient: float | None = None,
    flicker_exponent: float | None = None,
) -> Device:
    """A junction diode from ``anode`` to ``cathode``, in the SPICE model.

    Parameters left out take SPICE's defaults. Given a breakdown voltage, the reverse
    current grows exponentially beyond it, with the breakdown current there and the
    breakdown emission coefficient, the emission coefficient unless given.

    Away from the nominal temperature, the saturation current scales with the energy
    gap and the temperature exponent; the junction potential and capacitance follow
    silicon's band gap, and the breakdown voltage falls by its temperature coefficient,
    as in ngspice.

    Its series resistance makes thermal noise and its current shot noise; with a
    flicker coefficient ``KF``, its current ``I`` also makes flicker noise of
    ``KF · |I|^AF / f`` A²/Hz, ``AF`` the flicker exponent, as SPICE models it.
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
            "flicker_coefficient": flicker_coefficient,
            "flicker_exponent": flicker_exponent,
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
    threshold_voltage_temperature_coefficient: Quantity | None = None,
    transconductance_parameter_temperature_exponent: float | None = None,
    energy_gap: Quantity | None = None,
    saturation_current_temperature_exponent: float | None = None,
    nominal_temperature: Quantity | None = None,
    flicker_coefficient: float | None = None,
    flicker_exponent: float | None = None,
) -> Device:
    """A junction field-effect transistor between ``drain``, ``gate`` and ``source``.

    It follows the SPICE model with Sydney University's doping tail, as ngspice does:
    two gate junctions, and a drain current that follows the square law above the
    threshold voltage. Parameters left out take SPICE's defaults, and a doping tail of 1
    is the plain square law. A drain or source resistance adds an internal node, the
    unknown ``drain`` or ``source``.

    Away from the nominal temperature, the gate junctions' saturation current scales
    with the energy gap and the temperature exponent; their potential and capacitances
    follow silicon's band gap; the threshold moves by its temperature coefficient and
    the transconductance parameter scales by its temperature exponent, as in ngspice.

    Its resistances make thermal noise, and its channel thermal noise of
    ``8kT · gm / 3`` A²/Hz; with a flicker coefficient ``KF``, its drain current ``Id``
    also makes flicker noise of ``KF · |Id|^AF / f`` A²/Hz, ``AF`` the flicker exponent,
    as SPICE models it.

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
            "threshold_voltage_temperature_coefficient": (
                threshold_voltage_temperature_coefficient,
                VOLTS_PER_KELVIN,
            ),
            "energy_gap": (energy_gap, VOLTAGE),
            "nominal_temperature": (nominal_temperature, TEMPERATURE),
        },
        {
            "depletion_coefficient": depletion_coefficient,
            "doping_tail": doping_tail,
            "transconductance_parameter_temperature_exponent": (
                transconductance_parameter_temperature_exponent
            ),
            "saturation_current_temperature_exponent": (
                saturation_current_temperature_exponent
            ),
            "flicker_coefficient": flicker_coefficient,
            "flicker_exponent": flicker_exponent,
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
    min_base_resistance: Quantity | None = None,
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
    excess_phase: float | None = None,
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
    flicker_coefficient: float | None = None,
    flicker_exponent: float | None = None,
) -> Device:
    """A bipolar junction transistor between ``collector``, ``base`` and ``emitter``.

    It follows the Gummel-Poon model of SPICE, as ngspice does. Parameters left out take
    SPICE's defaults, an Early voltage or a knee current left out infinite. A base,
    collector or emitter resistance adds an internal node, the unknown ``base``,
    ``collector`` or ``emitter``.

    A transistor built ``vertical`` or ``lateral`` has a fourth terminal, ``substrate``,
    joined by a junction to its collector if vertical or to its base if lateral. The
    junction conducts the minimum conductance, and a diode's current with a substrate
    saturation current; its depletion capacitance grows linearly in forward bias. The
    substrate parameters need a substrate.

    An ``excess_phase``, SPICE's ``PTF`` in degrees, delays the forward transport
    current by ``PTF · π / 180 · TF`` through a second-order Bessel filter, as
    ngspice's transient does, so the phase it adds at ``1 / (2π · TF)`` is nearly
    ``PTF``; the filter's two unknowns are ``delayed_transport`` and
    ``delayed_transport_slope``.

    Away from the nominal temperature, the saturation currents scale with the energy
    gap and the temperature exponent; the betas scale by their temperature exponent,
    and the junctions' potentials and capacitances follow silicon's band gap, as in
    ngspice.

    Its resistances make thermal noise, and its collector and base currents shot noise;
    with a flicker coefficient ``KF``, its base current ``Ib`` also makes flicker noise
    of ``KF · |Ib|^AF / f`` A²/Hz, ``AF`` the flicker exponent, as SPICE models it.

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
    parameters = _parameters(
        {
            "saturation_current": (saturation_current, CURRENT),
            "forward_early_voltage": (forward_early_voltage, VOLTAGE),
            "reverse_early_voltage": (reverse_early_voltage, VOLTAGE),
            "forward_knee_current": (forward_knee_current, CURRENT),
            "reverse_knee_current": (reverse_knee_current, CURRENT),
            "base_emitter_leakage_current": (base_emitter_leakage_current, CURRENT),
            "base_collector_leakage_current": (base_collector_leakage_current, CURRENT),
            "base_resistance": (base_resistance, RESISTANCE),
            "min_base_resistance": (min_base_resistance, RESISTANCE),
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
            "excess_phase": excess_phase,
            "substrate_emission_coefficient": substrate_emission_coefficient,
            "substrate_grading": substrate_grading,
            "saturation_current_temperature_exponent": (
                saturation_current_temperature_exponent
            ),
            "beta_temperature_exponent": beta_temperature_exponent,
            "flicker_coefficient": flicker_coefficient,
            "flicker_exponent": flicker_exponent,
        },
        {"polarity": polarity, "substrate": substrate},
    )
    return device("bipolar_transistor", parameters)


def vbic(  # noqa: PLR0913 - VBIC 1.2 has 103 parameters
    *,
    polarity: Literal["npn", "pnp"] | None = None,
    extrinsic_collector_resistance: Quantity | None = None,
    intrinsic_collector_resistance: Quantity | None = None,
    epi_saturation_voltage: Quantity | None = None,
    epi_doping_parameter: float | None = None,
    high_current_collector_factor: float | None = None,
    extrinsic_base_resistance: Quantity | None = None,
    intrinsic_base_resistance: Quantity | None = None,
    emitter_resistance: Quantity | None = None,
    substrate_resistance: Quantity | None = None,
    parasitic_base_resistance: Quantity | None = None,
    saturation_current: Quantity | None = None,
    forward_emission_coefficient: float | None = None,
    reverse_emission_coefficient: float | None = None,
    depletion_coefficient: float | None = None,
    base_emitter_overlap_capacitance: Quantity | None = None,
    base_emitter_capacitance: Quantity | None = None,
    base_emitter_potential: Quantity | None = None,
    base_emitter_grading: float | None = None,
    base_emitter_smoothing: Quantity | None = None,
    base_collector_overlap_capacitance: Quantity | None = None,
    base_collector_capacitance: Quantity | None = None,
    epi_charge: Quantity | None = None,
    extrinsic_base_collector_capacitance: Quantity | None = None,
    base_collector_potential: Quantity | None = None,
    base_collector_grading: float | None = None,
    base_collector_smoothing: Quantity | None = None,
    collector_substrate_capacitance: Quantity | None = None,
    collector_substrate_potential: Quantity | None = None,
    collector_substrate_grading: float | None = None,
    collector_substrate_smoothing: Quantity | None = None,
    collector_substrate_fixed_capacitance: Quantity | None = None,
    ideal_base_emitter_current: Quantity | None = None,
    intrinsic_base_emitter_fraction: float | None = None,
    ideal_base_emitter_emission_coefficient: float | None = None,
    nonideal_base_emitter_current: Quantity | None = None,
    nonideal_base_emitter_emission_coefficient: float | None = None,
    ideal_base_collector_current: Quantity | None = None,
    ideal_base_collector_emission_coefficient: float | None = None,
    nonideal_base_collector_current: Quantity | None = None,
    nonideal_base_collector_emission_coefficient: float | None = None,
    avalanche_coefficient: Quantity | None = None,
    avalanche_exponent_coefficient: float | None = None,
    parasitic_saturation_current: Quantity | None = None,
    parasitic_extrinsic_fraction: float | None = None,
    parasitic_forward_emission_coefficient: float | None = None,
    parasitic_ideal_base_emitter_current: Quantity | None = None,
    parasitic_nonideal_base_emitter_current: Quantity | None = None,
    parasitic_ideal_base_collector_current: Quantity | None = None,
    parasitic_ideal_base_collector_emission_coefficient: float | None = None,
    parasitic_nonideal_base_collector_current: Quantity | None = None,
    parasitic_nonideal_base_collector_emission_coefficient: float | None = None,
    forward_early_voltage: Quantity | None = None,
    reverse_early_voltage: Quantity | None = None,
    forward_knee_current: Quantity | None = None,
    reverse_knee_current: Quantity | None = None,
    parasitic_knee_current: Quantity | None = None,
    knee_rolloff: float | None = None,
    base_charge: Literal["vbic", "gummel_poon"] | None = None,
    forward_transit_time: Quantity | None = None,
    transit_time_base_width_coefficient: float | None = None,
    transit_time_bias_coefficient: float | None = None,
    transit_time_voltage: Quantity | None = None,
    transit_time_current: Quantity | None = None,
    reverse_transit_time: Quantity | None = None,
    excess_phase_delay: Quantity | None = None,
    flicker_coefficient: float | None = None,
    flicker_exponent: float | None = None,
    flicker_frequency_exponent: float | None = None,
    reach_through_voltage: Quantity | None = None,
    reach_through_smoothing: float | None = None,
    base_emitter_breakdown_voltage: Quantity | None = None,
    base_emitter_breakdown_emission_coefficient: float | None = None,
    base_emitter_breakdown_current: Quantity | None = None,
    reverse_saturation_current_ratio: float | None = None,
    nominal_temperature: Quantity | None = None,
    activation_energy: Quantity | None = None,
    ideal_base_emitter_activation_energy: Quantity | None = None,
    ideal_base_collector_activation_energy: Quantity | None = None,
    ideal_collector_substrate_activation_energy: Quantity | None = None,
    nonideal_base_emitter_activation_energy: Quantity | None = None,
    nonideal_base_collector_activation_energy: Quantity | None = None,
    nonideal_collector_substrate_activation_energy: Quantity | None = None,
    parasitic_activation_energy: Quantity | None = None,
    reverse_activation_energy_shift: Quantity | None = None,
    saturation_current_temperature_exponent: float | None = None,
    ideal_base_current_temperature_exponent: float | None = None,
    nonideal_base_current_temperature_exponent: float | None = None,
    reverse_saturation_current_temperature_exponent: float | None = None,
    emission_coefficient_temperature_coefficient: Quantity | None = None,
    avalanche_temperature_coefficient: Quantity | None = None,
    forward_knee_current_temperature_exponent: float | None = None,
    extrinsic_collector_resistance_temperature_exponent: float | None = None,
    intrinsic_collector_resistance_temperature_exponent: float | None = None,
    extrinsic_base_resistance_temperature_exponent: float | None = None,
    intrinsic_base_resistance_temperature_exponent: float | None = None,
    emitter_resistance_temperature_exponent: float | None = None,
    substrate_resistance_temperature_exponent: float | None = None,
    parasitic_base_resistance_temperature_exponent: float | None = None,
    epi_saturation_voltage_temperature_exponent: float | None = None,
    breakdown_voltage_temperature_coefficient: Quantity | None = None,
    breakdown_voltage_quadratic_temperature_coefficient: Quantity | None = None,
    breakdown_emission_temperature_coefficient: Quantity | None = None,
) -> Device:
    """A bipolar transistor of integrated circuits, VBIC 1.2, between four terminals.

    It follows the vertical bipolar inter-company model as ngspice 47 implements it,
    between ``collector``, ``base``, ``emitter`` and ``substrate`` (McAndrew et al.,
    "VBIC95, the vertical bipolar inter-company model", IEEE Journal of Solid-State
    Circuits, 1996, as the VBIC 1.2 definition extends it). Beside the Gummel-Poon
    transistor's transport current, Early effect, high injection and junction charges,
    it models the parasitic substrate transistor from the extrinsic base to the
    substrate, the weak avalanche of the base-collector junction, the quasi-saturation
    of the intrinsic collector by Kull's model, the base-emitter breakdown, the
    reach-through of the base-collector junction, and capacitances smoothed at the
    junctions' potentials. Parameters left out take ngspice's defaults, an Early
    voltage, a knee current, the epi saturation voltage and the transit time's voltage
    infinite.

    The extrinsic collector, extrinsic base, emitter and substrate resistances add the
    internal unknowns ``collector``, ``base``, ``emitter`` and ``substrate``; the
    intrinsic collector, the intrinsic base and the parasitic base are internal unknowns
    always, and an excess-phase delay adds ``transport_filter`` and
    ``delayed_transport``, the network that delays the forward transport current.

    Away from the nominal temperature, its saturation currents scale by their activation
    energies and temperature exponents, its resistances, forward knee current and epi
    saturation voltage by their exponents, its potentials and capacitances with the
    intrinsic carrier concentration, and its emission coefficients, avalanche and
    breakdown by their coefficients, as ngspice's VBIC scales them; its thermal voltage
    takes VBIC's own constants. A heated transistor runs at its thermal net's
    temperature, as VBIC's self-heating does, its thermal resistance and capacitance
    thermal elements on that net.

    Its resistances make thermal noise, the intrinsic ones at their slopes; its
    transport current, its intrinsic base-emitter current and the parasitic transistor's
    base-emitter and transport currents make shot noise; and with a flicker coefficient
    ``KFN``, both base-emitter currents ``Ib`` make flicker noise of ``KFN · |Ib|^AFN /
    f^BFN`` A²/Hz, as ngspice models them.

    Examples:
        >>> from jolty.units import V, fA, ohm
        >>> transistor = vbic(
        ...     saturation_current=10 * fA,
        ...     ideal_base_emitter_current=0.1 * fA,
        ...     forward_early_voltage=50 * V,
        ...     extrinsic_collector_resistance=10 * ohm,
        ... )
        >>> transistor.terminals
        ('collector', 'base', 'emitter', 'substrate')
    """
    parameters = _parameters(
        {
            "extrinsic_collector_resistance": (
                extrinsic_collector_resistance,
                RESISTANCE,
            ),
            "intrinsic_collector_resistance": (
                intrinsic_collector_resistance,
                RESISTANCE,
            ),
            "epi_saturation_voltage": (epi_saturation_voltage, VOLTAGE),
            "extrinsic_base_resistance": (extrinsic_base_resistance, RESISTANCE),
            "intrinsic_base_resistance": (intrinsic_base_resistance, RESISTANCE),
            "emitter_resistance": (emitter_resistance, RESISTANCE),
            "substrate_resistance": (substrate_resistance, RESISTANCE),
            "parasitic_base_resistance": (parasitic_base_resistance, RESISTANCE),
            "saturation_current": (saturation_current, CURRENT),
            "base_emitter_overlap_capacitance": (
                base_emitter_overlap_capacitance,
                CAPACITANCE,
            ),
            "base_emitter_capacitance": (base_emitter_capacitance, CAPACITANCE),
            "base_emitter_potential": (base_emitter_potential, VOLTAGE),
            "base_emitter_smoothing": (base_emitter_smoothing, VOLTAGE),
            "base_collector_overlap_capacitance": (
                base_collector_overlap_capacitance,
                CAPACITANCE,
            ),
            "base_collector_capacitance": (base_collector_capacitance, CAPACITANCE),
            "epi_charge": (epi_charge, CHARGE),
            "extrinsic_base_collector_capacitance": (
                extrinsic_base_collector_capacitance,
                CAPACITANCE,
            ),
            "base_collector_potential": (base_collector_potential, VOLTAGE),
            "base_collector_smoothing": (base_collector_smoothing, VOLTAGE),
            "collector_substrate_capacitance": (
                collector_substrate_capacitance,
                CAPACITANCE,
            ),
            "collector_substrate_potential": (collector_substrate_potential, VOLTAGE),
            "collector_substrate_smoothing": (collector_substrate_smoothing, VOLTAGE),
            "collector_substrate_fixed_capacitance": (
                collector_substrate_fixed_capacitance,
                CAPACITANCE,
            ),
            "ideal_base_emitter_current": (ideal_base_emitter_current, CURRENT),
            "nonideal_base_emitter_current": (nonideal_base_emitter_current, CURRENT),
            "ideal_base_collector_current": (ideal_base_collector_current, CURRENT),
            "nonideal_base_collector_current": (
                nonideal_base_collector_current,
                CURRENT,
            ),
            "avalanche_coefficient": (avalanche_coefficient, PER_VOLT),
            "parasitic_saturation_current": (parasitic_saturation_current, CURRENT),
            "parasitic_ideal_base_emitter_current": (
                parasitic_ideal_base_emitter_current,
                CURRENT,
            ),
            "parasitic_nonideal_base_emitter_current": (
                parasitic_nonideal_base_emitter_current,
                CURRENT,
            ),
            "parasitic_ideal_base_collector_current": (
                parasitic_ideal_base_collector_current,
                CURRENT,
            ),
            "parasitic_nonideal_base_collector_current": (
                parasitic_nonideal_base_collector_current,
                CURRENT,
            ),
            "forward_early_voltage": (forward_early_voltage, VOLTAGE),
            "reverse_early_voltage": (reverse_early_voltage, VOLTAGE),
            "forward_knee_current": (forward_knee_current, CURRENT),
            "reverse_knee_current": (reverse_knee_current, CURRENT),
            "parasitic_knee_current": (parasitic_knee_current, CURRENT),
            "forward_transit_time": (forward_transit_time, TIME),
            "transit_time_voltage": (transit_time_voltage, VOLTAGE),
            "transit_time_current": (transit_time_current, CURRENT),
            "reverse_transit_time": (reverse_transit_time, TIME),
            "excess_phase_delay": (excess_phase_delay, TIME),
            "reach_through_voltage": (reach_through_voltage, VOLTAGE),
            "base_emitter_breakdown_voltage": (base_emitter_breakdown_voltage, VOLTAGE),
            "base_emitter_breakdown_current": (base_emitter_breakdown_current, CURRENT),
            "nominal_temperature": (nominal_temperature, TEMPERATURE),
            "activation_energy": (activation_energy, VOLTAGE),
            "ideal_base_emitter_activation_energy": (
                ideal_base_emitter_activation_energy,
                VOLTAGE,
            ),
            "ideal_base_collector_activation_energy": (
                ideal_base_collector_activation_energy,
                VOLTAGE,
            ),
            "ideal_collector_substrate_activation_energy": (
                ideal_collector_substrate_activation_energy,
                VOLTAGE,
            ),
            "nonideal_base_emitter_activation_energy": (
                nonideal_base_emitter_activation_energy,
                VOLTAGE,
            ),
            "nonideal_base_collector_activation_energy": (
                nonideal_base_collector_activation_energy,
                VOLTAGE,
            ),
            "nonideal_collector_substrate_activation_energy": (
                nonideal_collector_substrate_activation_energy,
                VOLTAGE,
            ),
            "parasitic_activation_energy": (parasitic_activation_energy, VOLTAGE),
            "reverse_activation_energy_shift": (
                reverse_activation_energy_shift,
                VOLTAGE,
            ),
            "emission_coefficient_temperature_coefficient": (
                emission_coefficient_temperature_coefficient,
                PER_KELVIN,
            ),
            "avalanche_temperature_coefficient": (
                avalanche_temperature_coefficient,
                PER_KELVIN,
            ),
            "breakdown_voltage_temperature_coefficient": (
                breakdown_voltage_temperature_coefficient,
                PER_KELVIN,
            ),
            "breakdown_voltage_quadratic_temperature_coefficient": (
                breakdown_voltage_quadratic_temperature_coefficient,
                PER_KELVIN_SQUARED,
            ),
            "breakdown_emission_temperature_coefficient": (
                breakdown_emission_temperature_coefficient,
                PER_KELVIN,
            ),
        },
        {
            "epi_doping_parameter": epi_doping_parameter,
            "high_current_collector_factor": high_current_collector_factor,
            "forward_emission_coefficient": forward_emission_coefficient,
            "reverse_emission_coefficient": reverse_emission_coefficient,
            "depletion_coefficient": depletion_coefficient,
            "base_emitter_grading": base_emitter_grading,
            "base_collector_grading": base_collector_grading,
            "collector_substrate_grading": collector_substrate_grading,
            "intrinsic_base_emitter_fraction": intrinsic_base_emitter_fraction,
            "ideal_base_emitter_emission_coefficient": (
                ideal_base_emitter_emission_coefficient
            ),
            "nonideal_base_emitter_emission_coefficient": (
                nonideal_base_emitter_emission_coefficient
            ),
            "ideal_base_collector_emission_coefficient": (
                ideal_base_collector_emission_coefficient
            ),
            "nonideal_base_collector_emission_coefficient": (
                nonideal_base_collector_emission_coefficient
            ),
            "avalanche_exponent_coefficient": avalanche_exponent_coefficient,
            "parasitic_extrinsic_fraction": parasitic_extrinsic_fraction,
            "parasitic_forward_emission_coefficient": (
                parasitic_forward_emission_coefficient
            ),
            "parasitic_ideal_base_collector_emission_coefficient": (
                parasitic_ideal_base_collector_emission_coefficient
            ),
            "parasitic_nonideal_base_collector_emission_coefficient": (
                parasitic_nonideal_base_collector_emission_coefficient
            ),
            "knee_rolloff": knee_rolloff,
            "transit_time_base_width_coefficient": transit_time_base_width_coefficient,
            "transit_time_bias_coefficient": transit_time_bias_coefficient,
            "flicker_coefficient": flicker_coefficient,
            "flicker_exponent": flicker_exponent,
            "flicker_frequency_exponent": flicker_frequency_exponent,
            "reach_through_smoothing": reach_through_smoothing,
            "base_emitter_breakdown_emission_coefficient": (
                base_emitter_breakdown_emission_coefficient
            ),
            "reverse_saturation_current_ratio": reverse_saturation_current_ratio,
            "saturation_current_temperature_exponent": (
                saturation_current_temperature_exponent
            ),
            "ideal_base_current_temperature_exponent": (
                ideal_base_current_temperature_exponent
            ),
            "nonideal_base_current_temperature_exponent": (
                nonideal_base_current_temperature_exponent
            ),
            "reverse_saturation_current_temperature_exponent": (
                reverse_saturation_current_temperature_exponent
            ),
            "forward_knee_current_temperature_exponent": (
                forward_knee_current_temperature_exponent
            ),
            "extrinsic_collector_resistance_temperature_exponent": (
                extrinsic_collector_resistance_temperature_exponent
            ),
            "intrinsic_collector_resistance_temperature_exponent": (
                intrinsic_collector_resistance_temperature_exponent
            ),
            "extrinsic_base_resistance_temperature_exponent": (
                extrinsic_base_resistance_temperature_exponent
            ),
            "intrinsic_base_resistance_temperature_exponent": (
                intrinsic_base_resistance_temperature_exponent
            ),
            "emitter_resistance_temperature_exponent": (
                emitter_resistance_temperature_exponent
            ),
            "substrate_resistance_temperature_exponent": (
                substrate_resistance_temperature_exponent
            ),
            "parasitic_base_resistance_temperature_exponent": (
                parasitic_base_resistance_temperature_exponent
            ),
            "epi_saturation_voltage_temperature_exponent": (
                epi_saturation_voltage_temperature_exponent
            ),
        },
        {"polarity": polarity, "base_charge": base_charge},
    )
    return device("vbic", parameters)


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
    flicker_coefficient: float | None = None,
    flicker_exponent: float | None = None,
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
    SPICE's defaults. An oxide thickness gives the gate Meyer's capacitances; with a
    substrate doping, it derives the threshold, the body effect and the surface
    potential left out, as SPICE derives them. The body effect is a number of
    square-root volts.

    Away from the nominal temperature, the transconductance parameter and the mobility
    fall as the temperature to the power -1.5, the surface potential follows silicon's
    band gap and the threshold with it, and the junctions' saturation currents,
    potential and capacitances follow the band gap too, as in ngspice.

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

    Its resistances make thermal noise, and its channel thermal noise of
    ``8kT · gm / 3`` A²/Hz; with a flicker coefficient ``KF``, its drain current ``Id``
    also makes flicker noise of ``KF · |Id|^AF / (f · W · Leff · Cox²)`` A²/Hz over its
    effective gate, the widths of its parallel copies summed, with an oxide 0.1 µm thick
    unless given, ``AF`` the flicker exponent, as ngspice models it.

    Examples:
        >>> from jolty.units import nm, um
        >>> mosfet(width=2 * um, length=1 * um, oxide_thickness=20 * nm).parameters
        {'width': 2e-06, 'length': 1e-06, 'oxide_thickness': 2e-08}
    """
    capacitance_per_length = CAPACITANCE / LENGTH
    parameters = _parameters(
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
            "flicker_coefficient": flicker_coefficient,
            "flicker_exponent": flicker_exponent,
        },
        {"level": level, "channel": channel, "gate_material": gate_material},
    )
    return device("mosfet", parameters)


def vdmos(  # noqa: PLR0913 - ngspice's VDMOS has 49 parameters
    *,
    channel: Literal["n", "p"] | None = None,
    multiplier: float | None = None,
    threshold_voltage: Quantity | None = None,
    transconductance_parameter: Quantity | None = None,
    channel_length_modulation: Quantity | None = None,
    mobility_modulation: Quantity | None = None,
    triode_conductance_multiplier: float | None = None,
    subthreshold_slope: Quantity | None = None,
    subthreshold_shift: Quantity | None = None,
    drain_resistance: Quantity | None = None,
    source_resistance: Quantity | None = None,
    gate_resistance: Quantity | None = None,
    quasi_saturation_resistance: Quantity | None = None,
    quasi_saturation_voltage: Quantity | None = None,
    drain_source_resistance: Quantity | None = None,
    gate_source_capacitance: Quantity | None = None,
    max_gate_drain_capacitance: Quantity | None = None,
    min_gate_drain_capacitance: Quantity | None = None,
    gate_drain_capacitance_coefficient: Quantity | None = None,
    saturation_current: Quantity | None = None,
    emission_coefficient: float | None = None,
    body_resistance: Quantity | None = None,
    junction_capacitance: Quantity | None = None,
    junction_potential: Quantity | None = None,
    grading_coefficient: float | None = None,
    depletion_coefficient: float | None = None,
    transit_time: Quantity | None = None,
    breakdown_voltage: Quantity | None = None,
    breakdown_current: Quantity | None = None,
    breakdown_emission_coefficient: float | None = None,
    energy_gap: Quantity | None = None,
    saturation_current_temperature_exponent: float | None = None,
    nominal_temperature: Quantity | None = None,
    threshold_voltage_temperature_coefficient: Quantity | None = None,
    transconductance_parameter_temperature_exponent: float | None = None,
    subthreshold_slope_temperature_coefficient: Quantity | None = None,
    subthreshold_slope_quadratic_temperature_coefficient: Quantity | None = None,
    drain_resistance_temperature_exponent: float | None = None,
    drain_resistance_temperature_coefficient: Quantity | None = None,
    drain_resistance_quadratic_temperature_coefficient: Quantity | None = None,
    quasi_saturation_temperature_exponent: float | None = None,
    source_resistance_temperature_coefficient: Quantity | None = None,
    source_resistance_quadratic_temperature_coefficient: Quantity | None = None,
    gate_resistance_temperature_coefficient: Quantity | None = None,
    gate_resistance_quadratic_temperature_coefficient: Quantity | None = None,
    body_resistance_temperature_coefficient: Quantity | None = None,
    body_resistance_quadratic_temperature_coefficient: Quantity | None = None,
    flicker_coefficient: float | None = None,
    flicker_exponent: float | None = None,
) -> Device:
    """A vertical power MOSFET between ``drain``, ``gate`` and ``source``: VDMOS.

    It follows ngspice's VDMOS model, which reads the power MOSFETs' cards of LTspice:
    Shichman and Hodges's square law without a body effect, its overdrive softened below
    the threshold by the subthreshold slope, the mobility degraded by the gate's field,
    and given both quasi-saturation parameters, a drain resistance that grows with the
    drain voltage. The gate holds a constant capacitance to the source and one to the
    drain that runs from its minimum, far below the drain, to its maximum, far above
    it. A body diode with parameters of its own, its breakdown the transistor's, joins
    the source to the drain. Parameters left out take ngspice's defaults, an IRF540's
    or an IRF9540's where ngspice takes them from one.

    Away from the nominal temperature, the transconductance parameter scales by its
    temperature exponent, the threshold, the subthreshold slope and the resistances by
    their temperature coefficients, and the body diode as a diode does, as in ngspice.

    A drain, source, gate or body resistance adds an internal node, the unknown
    ``drain``, ``source``, ``gate`` or ``body_diode``. The drain and source resistances
    make thermal noise, and the channel thermal noise of ``8kT · gm / 3`` A²/Hz; with a
    flicker coefficient ``KF``, its drain current ``Id`` also makes flicker noise of
    ``KF · |Id|^AF / (f · Cox²)`` A²/Hz, ``Cox`` an oxide's of 0.1 µm per area, as
    ngspice models it.

    Examples:
        >>> from jolty.units import V, ohm
        >>> vdmos(threshold_voltage=4 * V, drain_resistance=0.044 * ohm).parameters
        {'threshold_voltage': 4.0, 'drain_resistance': 0.044}
    """
    parameters = _parameters(
        {
            "threshold_voltage": (threshold_voltage, VOLTAGE),
            "transconductance_parameter": (
                transconductance_parameter,
                TRANSCONDUCTANCE_PARAMETER,
            ),
            "channel_length_modulation": (channel_length_modulation, PER_VOLT),
            "mobility_modulation": (mobility_modulation, PER_VOLT),
            "subthreshold_slope": (subthreshold_slope, VOLTAGE),
            "subthreshold_shift": (subthreshold_shift, VOLTAGE),
            "drain_resistance": (drain_resistance, RESISTANCE),
            "source_resistance": (source_resistance, RESISTANCE),
            "gate_resistance": (gate_resistance, RESISTANCE),
            "quasi_saturation_resistance": (quasi_saturation_resistance, RESISTANCE),
            "quasi_saturation_voltage": (quasi_saturation_voltage, VOLTAGE),
            "drain_source_resistance": (drain_source_resistance, RESISTANCE),
            "gate_source_capacitance": (gate_source_capacitance, CAPACITANCE),
            "max_gate_drain_capacitance": (max_gate_drain_capacitance, CAPACITANCE),
            "min_gate_drain_capacitance": (min_gate_drain_capacitance, CAPACITANCE),
            "gate_drain_capacitance_coefficient": (
                gate_drain_capacitance_coefficient,
                PER_VOLT,
            ),
            "saturation_current": (saturation_current, CURRENT),
            "body_resistance": (body_resistance, RESISTANCE),
            "junction_capacitance": (junction_capacitance, CAPACITANCE),
            "junction_potential": (junction_potential, VOLTAGE),
            "transit_time": (transit_time, TIME),
            "breakdown_voltage": (breakdown_voltage, VOLTAGE),
            "breakdown_current": (breakdown_current, CURRENT),
            "energy_gap": (energy_gap, VOLTAGE),
            "nominal_temperature": (nominal_temperature, TEMPERATURE),
            "threshold_voltage_temperature_coefficient": (
                threshold_voltage_temperature_coefficient,
                VOLTS_PER_KELVIN,
            ),
            "subthreshold_slope_temperature_coefficient": (
                subthreshold_slope_temperature_coefficient,
                PER_KELVIN,
            ),
            "subthreshold_slope_quadratic_temperature_coefficient": (
                subthreshold_slope_quadratic_temperature_coefficient,
                PER_KELVIN_SQUARED,
            ),
            "drain_resistance_temperature_coefficient": (
                drain_resistance_temperature_coefficient,
                PER_KELVIN,
            ),
            "drain_resistance_quadratic_temperature_coefficient": (
                drain_resistance_quadratic_temperature_coefficient,
                PER_KELVIN_SQUARED,
            ),
            "source_resistance_temperature_coefficient": (
                source_resistance_temperature_coefficient,
                PER_KELVIN,
            ),
            "source_resistance_quadratic_temperature_coefficient": (
                source_resistance_quadratic_temperature_coefficient,
                PER_KELVIN_SQUARED,
            ),
            "gate_resistance_temperature_coefficient": (
                gate_resistance_temperature_coefficient,
                PER_KELVIN,
            ),
            "gate_resistance_quadratic_temperature_coefficient": (
                gate_resistance_quadratic_temperature_coefficient,
                PER_KELVIN_SQUARED,
            ),
            "body_resistance_temperature_coefficient": (
                body_resistance_temperature_coefficient,
                PER_KELVIN,
            ),
            "body_resistance_quadratic_temperature_coefficient": (
                body_resistance_quadratic_temperature_coefficient,
                PER_KELVIN_SQUARED,
            ),
        },
        {
            "multiplier": multiplier,
            "triode_conductance_multiplier": triode_conductance_multiplier,
            "emission_coefficient": emission_coefficient,
            "grading_coefficient": grading_coefficient,
            "depletion_coefficient": depletion_coefficient,
            "breakdown_emission_coefficient": breakdown_emission_coefficient,
            "saturation_current_temperature_exponent": (
                saturation_current_temperature_exponent
            ),
            "transconductance_parameter_temperature_exponent": (
                transconductance_parameter_temperature_exponent
            ),
            "drain_resistance_temperature_exponent": (
                drain_resistance_temperature_exponent
            ),
            "quasi_saturation_temperature_exponent": (
                quasi_saturation_temperature_exponent
            ),
            "flicker_coefficient": flicker_coefficient,
            "flicker_exponent": flicker_exponent,
        },
        {"channel": channel},
    )
    return device("vdmos", parameters)


_HALF: Final = Fraction(1, 2)

BSIM3_CARD: Final[Mapping[str, Dimension | Literal["uc"] | None]] = {
    "tox": LENGTH,
    "toxm": LENGTH,
    "xj": LENGTH,
    "nch": LENGTH**-3,
    "nsub": LENGTH**-3,
    "ngate": LENGTH**-3,
    "gamma1": VOLTAGE**_HALF,
    "gamma2": VOLTAGE**_HALF,
    "vbx": VOLTAGE,
    "vbm": VOLTAGE,
    "xt": LENGTH,
    "vth0": VOLTAGE,
    "vfb": VOLTAGE,
    "k1": VOLTAGE**_HALF,
    "k2": None,
    "k3": None,
    "k3b": PER_VOLT,
    "w0": LENGTH,
    "nlx": LENGTH,
    "dvt0": None,
    "dvt1": None,
    "dvt2": PER_VOLT,
    "dvt0w": None,
    "dvt1w": LENGTH**-1,
    "dvt2w": PER_VOLT,
    "dsub": None,
    "eta0": None,
    "etab": PER_VOLT,
    "u0": MOBILITY,
    "ua": LENGTH / VOLTAGE,
    "ub": LENGTH**2 / VOLTAGE**2,
    "uc": "uc",
    "vsat": LENGTH / TIME,
    "a0": None,
    "ags": PER_VOLT,
    "a1": PER_VOLT,
    "a2": None,
    "b0": LENGTH,
    "b1": LENGTH,
    "keta": PER_VOLT,
    "voff": VOLTAGE,
    "nfactor": None,
    "cdsc": CAPACITANCE / LENGTH**2,
    "cdscb": CAPACITANCE / (VOLTAGE * LENGTH**2),
    "cdscd": CAPACITANCE / (VOLTAGE * LENGTH**2),
    "cit": CAPACITANCE / LENGTH**2,
    "pclm": None,
    "pdiblc1": None,
    "pdiblc2": None,
    "pdiblcb": PER_VOLT,
    "drout": None,
    "pscbe1": VOLTAGE / LENGTH,
    "pscbe2": LENGTH / VOLTAGE,
    "pvag": None,
    "delta": VOLTAGE,
    "rdsw": RESISTANCE,
    "prwg": PER_VOLT,
    "prwb": VOLTAGE**-_HALF,
    "wr": None,
    "dwg": LENGTH / VOLTAGE,
    "dwb": LENGTH / VOLTAGE**_HALF,
    "alpha0": LENGTH / VOLTAGE,
    "alpha1": PER_VOLT,
    "beta0": VOLTAGE,
    "kt1": VOLTAGE,
    "kt1l": VOLTAGE * LENGTH,
    "kt2": None,
    "ute": None,
    "ua1": LENGTH / VOLTAGE,
    "ub1": LENGTH**2 / VOLTAGE**2,
    "uc1": "uc",
    "at": LENGTH / TIME,
    "prt": RESISTANCE,
    "cgsl": CAPACITANCE / LENGTH,
    "cgdl": CAPACITANCE / LENGTH,
    "ckappa": VOLTAGE,
    "cf": CAPACITANCE / LENGTH,
    "clc": LENGTH,
    "cle": None,
    "vfbcv": VOLTAGE,
    "acde": LENGTH / VOLTAGE,
    "moin": None,
    "noff": None,
    "voffcv": VOLTAGE,
    "cgso": CAPACITANCE / LENGTH,
    "cgdo": CAPACITANCE / LENGTH,
    "cgbo": CAPACITANCE / LENGTH,
    "lint": LENGTH,
    "ll": None,
    "lln": None,
    "lw": None,
    "lwn": None,
    "lwl": None,
    "llc": None,
    "lwc": None,
    "lwlc": None,
    "wint": LENGTH,
    "wl": None,
    "wln": None,
    "ww": None,
    "wwn": None,
    "wwl": None,
    "wlc": None,
    "wwc": None,
    "wwlc": None,
    "dlc": LENGTH,
    "dwc": LENGTH,
    "xl": LENGTH,
    "xw": LENGTH,
    "rsh": RESISTANCE,
    "js": CURRENT / LENGTH**2,
    "jsw": CURRENT / LENGTH,
    "ijth": CURRENT,
    "nj": None,
    "xti": None,
    "cj": CAPACITANCE / LENGTH**2,
    "cjsw": CAPACITANCE / LENGTH,
    "cjswg": CAPACITANCE / LENGTH,
    "mj": None,
    "mjsw": None,
    "mjswg": None,
    "pb": VOLTAGE,
    "pbsw": VOLTAGE,
    "pbswg": VOLTAGE,
    "tpb": VOLTAGE / TEMPERATURE,
    "tpbsw": VOLTAGE / TEMPERATURE,
    "tpbswg": VOLTAGE / TEMPERATURE,
    "tcj": TEMPERATURE**-1,
    "tcjsw": TEMPERATURE**-1,
    "tcjswg": TEMPERATURE**-1,
    "noia": None,
    "noib": None,
    "noic": None,
    "em": VOLTAGE / LENGTH,
    "ef": None,
    "af": None,
    "kf": None,
    "lintnoi": LENGTH,
}
"""The parameters of a BSIM3 card, by the BSIM3v3.3 manual's names, and the dimension of
each, in SI units: ``None`` for a plain number, as for ``ll``, whose unit follows
``lln``, and ``"uc"`` for ``uc`` and ``uc1``, whose unit follows ``mobility_model``."""

BSIM3_BINNED: Final = frozenset(
    {
        "xj",
        "nch",
        "nsub",
        "ngate",
        "gamma1",
        "gamma2",
        "vbx",
        "vbm",
        "xt",
        "vth0",
        "vfb",
        "k1",
        "k2",
        "k3",
        "k3b",
        "w0",
        "nlx",
        "dvt0",
        "dvt1",
        "dvt2",
        "dvt0w",
        "dvt1w",
        "dvt2w",
        "dsub",
        "eta0",
        "etab",
        "u0",
        "ua",
        "ub",
        "uc",
        "vsat",
        "a0",
        "ags",
        "a1",
        "a2",
        "b0",
        "b1",
        "keta",
        "voff",
        "nfactor",
        "cdsc",
        "cdscb",
        "cdscd",
        "cit",
        "pclm",
        "pdiblc1",
        "pdiblc2",
        "pdiblcb",
        "drout",
        "pscbe1",
        "pscbe2",
        "pvag",
        "delta",
        "rdsw",
        "prwg",
        "prwb",
        "wr",
        "dwg",
        "dwb",
        "alpha0",
        "alpha1",
        "beta0",
        "kt1",
        "kt1l",
        "kt2",
        "ute",
        "ua1",
        "ub1",
        "uc1",
        "at",
        "prt",
        "cgsl",
        "cgdl",
        "ckappa",
        "cf",
        "clc",
        "cle",
        "vfbcv",
        "acde",
        "moin",
        "noff",
        "voffcv",
    }
)
"""The BSIM3 parameters that bin: each has a length, a width and an area dependence,
its name after ``l``, ``w`` and ``p``, its unit times meters or square meters."""


def bsim3_dimension(name: str, mobility_model: int | None = None) -> Dimension | None:
    """The dimension of BSIM3's parameter `name`, ``None`` for a plain number.

    A binned dependence's is its parameter's times meters, or square meters for the
    area's, and ``uc``'s and ``uc1``'s per square volt times meters, or per volt at
    `mobility_model` 3.

    Examples:
        >>> bsim3_dimension("lvth0") == VOLTAGE * LENGTH, bsim3_dimension("k2")
        (True, None)

    Raises:
        ValueError: When BSIM3 has no parameter `name`.
    """
    dimension, extent = _bsim_parameter("BSIM3", name, BSIM3_CARD, BSIM3_BINNED)
    if dimension == "uc":
        dimension = PER_VOLT if mobility_model == 3 else LENGTH / VOLTAGE**2  # noqa: PLR2004 - mobmod 3's body effect
    return _extended(dimension, extent)


def _bsim_parameter[T](
    model: str, name: str, card: Mapping[str, T], binned: frozenset[str]
) -> tuple[T, Dimension]:
    """The entry in `card` of `model`'s parameter `name`, and its extent.

    The entry is the parameter's own, or that of the parameter it is a length, width or
    area dependence of; the extent is what its dimension is times, meters or square
    meters for a dependence.

    Raises:
        ValueError: When `model` has no parameter `name`.
    """
    if name in card:
        return card[name], DIMENSIONLESS
    prefix, base = name[:1], name[1:]
    if prefix not in {"l", "w", "p"} or base not in binned:
        message = f"{model} has no parameter `{name}`"
        raise ValueError(message)
    return card[base], LENGTH**2 if prefix == "p" else LENGTH


def _extended(dimension: Dimension | None, extent: Dimension) -> Dimension | None:
    """`dimension` times `extent`, ``None`` for a plain number."""
    if dimension is None:
        return None if extent == DIMENSIONLESS else extent
    return dimension * extent


def _card(
    parameters: dict[str, JsonValue],
    card: Mapping[str, Quantity | float],
    dimension_of: Callable[[str], Dimension | None],
) -> None:
    """Adds each entry of a model's `card` to `parameters`, its dimension checked.

    Raises:
        DimensionError: When a parameter has the wrong dimension.
        ValueError: When the model has no parameter of a name in `card`, or a value is
            not finite.
    """
    for name, value in card.items():
        dimension = dimension_of(name)
        match value:
            case Quantity() if dimension is not None:
                parameters[name] = value.to(dimension, name)
            case int() | float() if dimension is None:
                parameters[name] = finite(value, name)
            case _:
                unit = (
                    "a plain number"
                    if dimension is None
                    else f"a quantity in {dimension}"
                )
                message = f"{name} takes {unit}, not {value!r}"
                raise DimensionError(message)


def bsim3(  # noqa: PLR0913 - the instance's geometry and BSIM3's model selectors
    *,
    channel: Literal["n", "p"] | None = None,
    width: Quantity | None = None,
    length: Quantity | None = None,
    multiplier: float | None = None,
    drain_area: Quantity | None = None,
    source_area: Quantity | None = None,
    drain_perimeter: Quantity | None = None,
    source_perimeter: Quantity | None = None,
    drain_squares: float | None = None,
    source_squares: float | None = None,
    threshold_adjustment: Quantity | None = None,
    mobility_multiplier: float | None = None,
    mobility_model: Literal[1, 2, 3] | None = None,
    capacitance_model: Literal[0, 1, 2, 3] | None = None,
    noise_model: Literal[1, 2, 3, 4, 5, 6] | None = None,
    check_parameters: bool | None = None,
    charge_partition: float | None = None,
    nominal_temperature: Quantity | None = None,
    card: Mapping[str, Quantity | float] | None = None,
) -> Device:
    """A MOSFET in BSIM3v3.3 between ``drain``, ``gate``, ``source`` and ``bulk``.

    It follows the BSIM3v3.3 model of short channels (W. Liu, X. Xi and others, UC
    Berkeley, 2005), SPICE's level 8 and 49, as ngspice 47 implements it. The keywords
    before `card` are the instance's: its drawn geometry, its diffusions, a shift of
    its threshold and a factor of its mobility, then the card's models of the mobility,
    the charges and the noise, whether its sizes are checked and clamped, and the
    drain's share of the channel's charge. Each entry of `card` is a parameter of the
    BSIM3v3.3 manual by its name, such as ``vth0``, ``tox`` or ``lvth0``, in SI units:
    a quantity where it has a dimension, as
    [`bsim3_dimension`][jolty.devices.bsim3_dimension] says, and a number where it has
    none. Parameters left out take ngspice's defaults.

    The drain current folds in the threshold's short-channel, narrow-width and
    drain-induced barrier lowering terms, the poly gate's depletion, the mobility's
    degradation by the gate field, velocity saturation, the output resistance's
    channel-length modulation, barrier lowering and substrate current, and the source
    and drain resistance ``rdsw``, that of a channel 1 µm wide; the impact ionization's
    substrate current flows into the bulk. The charges under the gate follow
    `capacitance_model`, the overlaps' charges beside them, and two diodes join the bulk
    to the drain and the source. A binned card's ``l``, ``w`` and ``p`` dependences are
    on the reciprocal of the effective length, width and their product in meters, not
    micrometers, and every parameter scales with the temperature as ngspice scales it. A
    sheet resistance ``rsh`` adds an internal node behind the drain and the source, the
    unknowns ``drain`` and ``source``.

    Its sheet resistances make thermal noise, and its channel the thermal and flicker
    noise of `noise_model`: SPICE2's or BSIM3's.

    Examples:
        >>> from jolty.units import V, nm, um
        >>> transistor = bsim3(
        ...     width=2 * um,
        ...     length=0.35 * um,
        ...     card={"tox": 7.6 * nm, "vth0": 0.5 * V, "k2": -0.02},
        ... )
        >>> transistor.parameters["tox"], transistor.parameters["k2"]
        (7.6e-09, -0.02)

    Raises:
        DimensionError: When a parameter has the wrong dimension.
        ValueError: When BSIM3 has no parameter of a name in `card`, or a value is not
            finite.
    """
    parameters = _parameters(
        {
            "width": (width, LENGTH),
            "length": (length, LENGTH),
            "drain_area": (drain_area, LENGTH**2),
            "source_area": (source_area, LENGTH**2),
            "drain_perimeter": (drain_perimeter, LENGTH),
            "source_perimeter": (source_perimeter, LENGTH),
            "threshold_adjustment": (threshold_adjustment, VOLTAGE),
            "nominal_temperature": (nominal_temperature, TEMPERATURE),
        },
        {
            "multiplier": multiplier,
            "drain_squares": drain_squares,
            "source_squares": source_squares,
            "mobility_multiplier": mobility_multiplier,
            "charge_partition": charge_partition,
        },
        {
            "channel": channel,
            "mobility_model": mobility_model,
            "capacitance_model": capacitance_model,
            "noise_model": noise_model,
        },
    )
    if check_parameters is not None:
        parameters["check_parameters"] = check_parameters
    _card(parameters, card or {}, lambda name: bsim3_dimension(name, mobility_model))
    return device("bsim3", parameters)


BSIM4_CARD: Final[Mapping[str, Dimension | None]] = {
    "toxe": LENGTH,
    "toxp": LENGTH,
    "toxm": LENGTH,
    "toxref": LENGTH,
    "dtox": LENGTH,
    "epsrox": None,
    "cdsc": CAPACITANCE / LENGTH**2,
    "cdscb": CAPACITANCE / (VOLTAGE * LENGTH**2),
    "cdscd": CAPACITANCE / (VOLTAGE * LENGTH**2),
    "cit": CAPACITANCE / LENGTH**2,
    "nfactor": None,
    "xj": LENGTH,
    "vsat": LENGTH / TIME,
    "at": LENGTH / TIME,
    "a0": None,
    "ags": PER_VOLT,
    "a1": PER_VOLT,
    "a2": None,
    "keta": PER_VOLT,
    "nsub": LENGTH**-3,
    "ndep": LENGTH**-3,
    "nsd": LENGTH**-3,
    "phin": VOLTAGE,
    "ngate": LENGTH**-3,
    "gamma1": VOLTAGE**_HALF,
    "gamma2": VOLTAGE**_HALF,
    "vbx": VOLTAGE,
    "vbm": VOLTAGE,
    "xt": LENGTH,
    "k1": VOLTAGE**_HALF,
    "kt1": VOLTAGE,
    "kt1l": VOLTAGE * LENGTH,
    "kt2": None,
    "k2": None,
    "k3": None,
    "k3b": PER_VOLT,
    "w0": LENGTH,
    "dvtp0": LENGTH,
    "dvtp1": PER_VOLT,
    "lpe0": LENGTH,
    "lpeb": LENGTH,
    "dvt0": None,
    "dvt1": None,
    "dvt2": PER_VOLT,
    "dvt0w": LENGTH**-1,
    "dvt1w": LENGTH**-1,
    "dvt2w": PER_VOLT,
    "drout": None,
    "dsub": None,
    "vth0": VOLTAGE,
    "ua": LENGTH / VOLTAGE,
    "ua1": LENGTH / VOLTAGE,
    "ub": (LENGTH / VOLTAGE) ** 2,
    "ub1": (LENGTH / VOLTAGE) ** 2,
    "uc": LENGTH / VOLTAGE**2,
    "uc1": LENGTH / VOLTAGE**2,
    "ud": LENGTH**-2,
    "ud1": LENGTH**-2,
    "up": LENGTH**-2,
    "lp": LENGTH,
    "u0": MOBILITY,
    "eu": None,
    "ute": None,
    "voff": VOLTAGE,
    "minv": None,
    "voffl": VOLTAGE * LENGTH,
    "cgso": CAPACITANCE / LENGTH,
    "cgdo": CAPACITANCE / LENGTH,
    "cgbo": CAPACITANCE / LENGTH,
    "delta": VOLTAGE,
    "rsh": RESISTANCE,
    "rdsw": RESISTANCE,
    "rdswmin": RESISTANCE,
    "rsw": RESISTANCE,
    "rdw": RESISTANCE,
    "rdwmin": RESISTANCE,
    "rswmin": RESISTANCE,
    "prwg": PER_VOLT,
    "prwb": VOLTAGE**-_HALF,
    "prt": RESISTANCE,
    "eta0": None,
    "etab": PER_VOLT,
    "pclm": None,
    "pdiblc1": None,
    "pdiblc2": None,
    "pdiblcb": PER_VOLT,
    "fprout": VOLTAGE / LENGTH**_HALF,
    "pdits": PER_VOLT,
    "pditsl": LENGTH**-1,
    "pditsd": PER_VOLT,
    "pscbe1": VOLTAGE / LENGTH,
    "pscbe2": LENGTH / VOLTAGE,
    "pvag": None,
    "jss": CURRENT / LENGTH**2,
    "jsws": CURRENT / LENGTH,
    "jswgs": CURRENT / LENGTH,
    "pbs": VOLTAGE,
    "njs": None,
    "xtis": None,
    "mjs": None,
    "pbsws": VOLTAGE,
    "mjsws": None,
    "pbswgs": VOLTAGE,
    "mjswgs": None,
    "cjs": CAPACITANCE / LENGTH**2,
    "cjsws": CAPACITANCE / LENGTH,
    "cjswgs": CAPACITANCE / LENGTH,
    "jsd": CURRENT / LENGTH**2,
    "jswd": CURRENT / LENGTH,
    "jswgd": CURRENT / LENGTH,
    "pbd": VOLTAGE,
    "njd": None,
    "xtid": None,
    "mjd": None,
    "pbswd": VOLTAGE,
    "mjswd": None,
    "pbswgd": VOLTAGE,
    "mjswgd": None,
    "cjd": CAPACITANCE / LENGTH**2,
    "cjswd": CAPACITANCE / LENGTH,
    "cjswgd": CAPACITANCE / LENGTH,
    "vfbcv": VOLTAGE,
    "vfb": VOLTAGE,
    "tpb": VOLTAGE / TEMPERATURE,
    "tcj": TEMPERATURE**-1,
    "tpbsw": VOLTAGE / TEMPERATURE,
    "tcjsw": TEMPERATURE**-1,
    "tpbswg": VOLTAGE / TEMPERATURE,
    "tcjswg": TEMPERATURE**-1,
    "acde": LENGTH / VOLTAGE,
    "moin": None,
    "noff": None,
    "voffcv": VOLTAGE,
    "dmcg": LENGTH,
    "dmci": LENGTH,
    "dmdg": LENGTH,
    "dmcgt": LENGTH,
    "xgw": LENGTH,
    "xgl": LENGTH,
    "rshg": RESISTANCE,
    "ngcon": None,
    "xrcrg1": None,
    "xrcrg2": None,
    "lambda": None,
    "vtl": LENGTH / TIME,
    "lc": LENGTH,
    "xn": None,
    "vfbsdoff": VOLTAGE,
    "tvfbsdoff": TEMPERATURE**-1,
    "tvoff": TEMPERATURE**-1,
    "lintnoi": LENGTH,
    "lint": LENGTH,
    "ll": None,
    "llc": None,
    "lln": None,
    "lw": None,
    "lwc": None,
    "lwn": None,
    "lwl": None,
    "lwlc": None,
    "wr": None,
    "wint": LENGTH,
    "dwg": LENGTH / VOLTAGE,
    "dwb": LENGTH / VOLTAGE**_HALF,
    "wl": None,
    "wlc": None,
    "wln": None,
    "ww": None,
    "wwc": None,
    "wwn": None,
    "wwl": None,
    "wwlc": None,
    "b0": LENGTH,
    "b1": LENGTH,
    "cgsl": CAPACITANCE / LENGTH,
    "cgdl": CAPACITANCE / LENGTH,
    "ckappas": VOLTAGE,
    "ckappad": VOLTAGE,
    "cf": CAPACITANCE / LENGTH,
    "clc": LENGTH,
    "cle": None,
    "dwc": LENGTH,
    "dlc": LENGTH,
    "xw": LENGTH,
    "xl": LENGTH,
    "dlcig": LENGTH,
    "dwj": LENGTH,
    "alpha0": LENGTH / VOLTAGE,
    "alpha1": PER_VOLT,
    "beta0": VOLTAGE,
    "agidl": CONDUCTANCE,
    "bgidl": VOLTAGE / LENGTH,
    "cgidl": VOLTAGE**3,
    "egidl": VOLTAGE,
    "aigc": None,
    "bigc": None,
    "cigc": PER_VOLT,
    "aigsd": None,
    "bigsd": None,
    "cigsd": PER_VOLT,
    "aigbacc": None,
    "bigbacc": None,
    "cigbacc": PER_VOLT,
    "aigbinv": None,
    "bigbinv": None,
    "cigbinv": PER_VOLT,
    "nigc": None,
    "nigbinv": None,
    "nigbacc": None,
    "ntox": None,
    "eigbinv": VOLTAGE,
    "pigcd": None,
    "poxedge": None,
    "ijthdfwd": CURRENT,
    "ijthsfwd": CURRENT,
    "ijthdrev": CURRENT,
    "ijthsrev": CURRENT,
    "xjbvd": None,
    "xjbvs": None,
    "bvd": VOLTAGE,
    "bvs": VOLTAGE,
    "jtss": CURRENT / LENGTH**2,
    "jtsd": CURRENT / LENGTH**2,
    "jtssws": CURRENT / LENGTH,
    "jtsswd": CURRENT / LENGTH,
    "jtsswgs": CURRENT / LENGTH,
    "jtsswgd": CURRENT / LENGTH,
    "njts": None,
    "njtssw": None,
    "njtsswg": None,
    "xtss": None,
    "xtsd": None,
    "xtssws": None,
    "xtsswd": None,
    "xtsswgs": None,
    "xtsswgd": None,
    "tnjts": None,
    "tnjtssw": None,
    "tnjtsswg": None,
    "vtss": VOLTAGE,
    "vtsd": VOLTAGE,
    "vtssws": VOLTAGE,
    "vtsswd": VOLTAGE,
    "vtsswgs": VOLTAGE,
    "vtsswgd": VOLTAGE,
    "gbmin": CONDUCTANCE,
    "rbdb": RESISTANCE,
    "rbpb": RESISTANCE,
    "rbsb": RESISTANCE,
    "rbps": RESISTANCE,
    "rbpd": RESISTANCE,
    "rbps0": RESISTANCE,
    "rbpsl": None,
    "rbpsw": None,
    "rbpsnf": None,
    "rbpd0": RESISTANCE,
    "rbpdl": None,
    "rbpdw": None,
    "rbpdnf": None,
    "rbpbx0": RESISTANCE,
    "rbpbxl": None,
    "rbpbxw": None,
    "rbpbxnf": None,
    "rbpby0": RESISTANCE,
    "rbpbyl": None,
    "rbpbyw": None,
    "rbpbynf": None,
    "rbsbx0": RESISTANCE,
    "rbsby0": RESISTANCE,
    "rbdbx0": RESISTANCE,
    "rbdby0": RESISTANCE,
    "rbsdbxl": None,
    "rbsdbxw": None,
    "rbsdbxnf": None,
    "rbsdbyl": None,
    "rbsdbyw": None,
    "rbsdbynf": None,
    "saref": LENGTH,
    "sbref": LENGTH,
    "wlod": LENGTH,
    "ku0": LENGTH,
    "kvsat": None,
    "kvth0": VOLTAGE * LENGTH,
    "tku0": None,
    "llodku0": None,
    "wlodku0": None,
    "llodvth": None,
    "wlodvth": None,
    "lku0": None,
    "wku0": None,
    "pku0": None,
    "lkvth0": None,
    "wkvth0": None,
    "pkvth0": None,
    "stk2": LENGTH,
    "lodk2": None,
    "steta0": LENGTH,
    "lodeta0": None,
    "web": None,
    "wec": None,
    "kvth0we": VOLTAGE,
    "k2we": None,
    "ku0we": None,
    "scref": LENGTH,
    "noia": None,
    "noib": None,
    "noic": None,
    "tnoia": None,
    "tnoib": None,
    "rnoia": None,
    "rnoib": None,
    "ntnoi": None,
    "em": VOLTAGE / LENGTH,
    "ef": None,
    "af": None,
    "kf": None,
}
"""The parameters of a BSIM4 card, by the BSIM4.5 manual's names, and the dimension of
each, in SI units: ``None`` for a plain number, as for ``ll``, whose unit follows
``lln``, or for one in the manual's own units, as the gate tunneling's ``aigc``."""

BSIM4_BINNED: Final = frozenset(
    {
        "cdsc",
        "cdscb",
        "cdscd",
        "cit",
        "nfactor",
        "xj",
        "vsat",
        "at",
        "a0",
        "ags",
        "a1",
        "a2",
        "keta",
        "nsub",
        "ndep",
        "nsd",
        "phin",
        "ngate",
        "gamma1",
        "gamma2",
        "vbx",
        "vbm",
        "xt",
        "vfb",
        "k1",
        "kt1",
        "kt1l",
        "k2",
        "kt2",
        "k3",
        "k3b",
        "w0",
        "lpe0",
        "lpeb",
        "dvtp0",
        "dvtp1",
        "dvt0",
        "dvt1",
        "dvt2",
        "dvt0w",
        "dvt1w",
        "dvt2w",
        "drout",
        "dsub",
        "vth0",
        "ua",
        "ua1",
        "ub",
        "ub1",
        "uc",
        "uc1",
        "ud",
        "ud1",
        "up",
        "lp",
        "eu",
        "u0",
        "ute",
        "voff",
        "tvoff",
        "minv",
        "fprout",
        "pdits",
        "pditsd",
        "delta",
        "rdsw",
        "rdw",
        "rsw",
        "prwg",
        "prwb",
        "prt",
        "eta0",
        "etab",
        "pclm",
        "pdiblc1",
        "pdiblc2",
        "pdiblcb",
        "pscbe1",
        "pscbe2",
        "pvag",
        "wr",
        "dwg",
        "dwb",
        "b0",
        "b1",
        "alpha0",
        "alpha1",
        "beta0",
        "agidl",
        "bgidl",
        "cgidl",
        "egidl",
        "aigc",
        "bigc",
        "cigc",
        "aigsd",
        "bigsd",
        "cigsd",
        "aigbacc",
        "bigbacc",
        "cigbacc",
        "aigbinv",
        "bigbinv",
        "cigbinv",
        "nigc",
        "nigbacc",
        "nigbinv",
        "ntox",
        "eigbinv",
        "pigcd",
        "poxedge",
        "xrcrg1",
        "xrcrg2",
        "lambda",
        "vtl",
        "xn",
        "vfbsdoff",
        "tvfbsdoff",
        "cgsl",
        "cgdl",
        "ckappas",
        "ckappad",
        "cf",
        "clc",
        "cle",
        "vfbcv",
        "acde",
        "moin",
        "noff",
        "voffcv",
        "kvth0we",
        "k2we",
        "ku0we",
    }
)
"""The BSIM4 parameters that bin: each has a length, a width and an area dependence,
its name after ``l``, ``w`` and ``p``, its unit times meters or square meters."""


def bsim4_dimension(name: str) -> Dimension | None:
    """The dimension of BSIM4's parameter `name`, ``None`` for a plain number.

    A binned dependence's is its parameter's times meters, or square meters for the
    area's.

    Examples:
        >>> bsim4_dimension("lvth0") == VOLTAGE * LENGTH, bsim4_dimension("k2")
        (True, None)

    Raises:
        ValueError: When BSIM4 has no parameter `name`.
    """
    return _extended(*_bsim_parameter("BSIM4", name, BSIM4_CARD, BSIM4_BINNED))


def bsim4(  # noqa: PLR0913 - the instance's geometry and BSIM4's model selectors
    *,
    channel: Literal["n", "p"] | None = None,
    width: Quantity | None = None,
    length: Quantity | None = None,
    multiplier: float | None = None,
    fingers: float | None = None,
    drain_area: Quantity | None = None,
    source_area: Quantity | None = None,
    drain_perimeter: Quantity | None = None,
    source_perimeter: Quantity | None = None,
    drain_squares: float | None = None,
    source_squares: float | None = None,
    diffusion_edge_a: Quantity | None = None,
    diffusion_edge_b: Quantity | None = None,
    finger_spacing: Quantity | None = None,
    threshold_adjustment: Quantity | None = None,
    mobility_multiplier: float | None = None,
    body_resistance_model: Literal[0, 1] | None = None,
    perimeter_model: Literal[0, 1] | None = None,
    check_parameters: bool | None = None,
    charge_partition: float | None = None,
    nominal_temperature: Quantity | None = None,
    card: Mapping[str, Quantity | float] | None = None,
) -> Device:
    """A MOSFET in BSIM4.5 between ``drain``, ``gate``, ``source`` and ``bulk``.

    It follows the BSIM4.5.0 model of short channels (X. Xi, M. Dunga and others, UC
    Berkeley, 2005), SPICE's level 14 and 54 of version 4.5, as ngspice 47 runs it, in
    the modes SKY130's and GF180MCU's cards select. The keywords before `card` are the
    instance's: its drawn geometry and fingers, its diffusions, the distances that set
    the stress of its length of diffusion, a shift of its threshold and a factor of its
    mobility, then the card's body resistance network and perimeter model, whether its
    sizes are checked and clamped, and the drain's share of the channel's charge. Each
    entry of `card` is a parameter of the BSIM4.5 manual by its name, such as ``vth0``,
    ``toxe`` or ``lvth0``, in SI units: a quantity where it has a dimension, as
    [`bsim4_dimension`][jolty.devices.bsim4_dimension] says, and a number where it has
    none. Parameters left out take ngspice's defaults.

    The drain current folds in the threshold's short-channel, narrow-width, pocket and
    barrier lowering terms, the poly gate's depletion, the mobility's degradation,
    velocity saturation and overshoot, the source end's velocity limit, the output
    resistance's terms and the source and drain resistance ``rdsw``; the impact
    ionization's substrate current and the gate-induced leakages flow into the body,
    and two diodes, with their trap-assisted tunneling, join it to the drain and the
    source. A binned card's ``l``, ``w`` and ``p`` dependences are on the reciprocal of
    the effective length, width and their product in meters, not micrometers, and
    every parameter scales with the temperature as ngspice scales it. A sheet
    resistance ``rsh`` adds an internal node behind the drain and the source where
    ``drain_squares`` or ``source_squares`` is positive, and ``body_resistance_model``
    1 three under the channel and the junctions. The charges under the gate follow
    ``capmod`` 2, beside the overlaps' and the junctions'. Its noise is not modeled yet,
    so a noise analysis refuses a circuit that holds one.

    Examples:
        >>> from jolty.units import V, nm, um
        >>> transistor = bsim4(
        ...     width=2 * um,
        ...     length=0.15 * um,
        ...     card={"toxe": 4 * nm, "vth0": 0.5 * V, "k2": -0.02},
        ... )
        >>> transistor.parameters["toxe"], transistor.parameters["k2"]
        (4e-09, -0.02)

    Raises:
        DimensionError: When a parameter has the wrong dimension.
        ValueError: When BSIM4 has no parameter of a name in `card`, or a value is not
            finite.
    """
    parameters = _parameters(
        {
            "width": (width, LENGTH),
            "length": (length, LENGTH),
            "drain_area": (drain_area, LENGTH**2),
            "source_area": (source_area, LENGTH**2),
            "drain_perimeter": (drain_perimeter, LENGTH),
            "source_perimeter": (source_perimeter, LENGTH),
            "diffusion_edge_a": (diffusion_edge_a, LENGTH),
            "diffusion_edge_b": (diffusion_edge_b, LENGTH),
            "finger_spacing": (finger_spacing, LENGTH),
            "threshold_adjustment": (threshold_adjustment, VOLTAGE),
            "nominal_temperature": (nominal_temperature, TEMPERATURE),
        },
        {
            "multiplier": multiplier,
            "fingers": fingers,
            "drain_squares": drain_squares,
            "source_squares": source_squares,
            "mobility_multiplier": mobility_multiplier,
            "charge_partition": charge_partition,
        },
        {
            "channel": channel,
            "body_resistance_model": body_resistance_model,
            "perimeter_model": perimeter_model,
        },
    )
    if check_parameters is not None:
        parameters["check_parameters"] = check_parameters
    _card(parameters, card or {}, bsim4_dimension)
    return device("bsim4", parameters)


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
    threshold: float | None = None,
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
        pins: dict[str, JsonValue] = {"input_capacitances": [capacitance] * len(inputs)}
        parameters["electrical"] = pins | _parameters({}, {"threshold": threshold})
    elif input_capacitance is not None or threshold is not None:
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
    threshold: float | None = None,
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
        "initial": format_logic_level(initial),
        "changes": [
            [time.to(TIME, "a change's time"), format_logic_level(level)]
            for time, level in changes
        ],
        "transition": transition.to(TIME, "a transition"),
    }
    return device("logic_source", parameters)


def format_logic_level(level: LogicLevel | float) -> str:
    """A logic level as the engine writes it: ``0``, ``1`` or ``x``.

    A result holds a level as a number, NaN where it is unknown.

    Examples:
        >>> [format_logic_level(level) for level in (0, 1, None, 1.0, math.nan)]
        ['0', '1', 'x', '1', 'x']
    """
    if level is None or math.isnan(level):
        return "x"
    return str(int(level))


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
    choices: Mapping[str, str | int | None] | None = None,
) -> dict[str, JsonValue]:
    """The parameters given, by name, each checked and in SI base units.

    A quantity's dimension is checked, and a number is checked finite; a choice, such
    as a transistor's polarity, is taken as it is.

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
    parameters |= {
        name: value for name, value in (choices or {}).items() if value is not None
    }
    return parameters
