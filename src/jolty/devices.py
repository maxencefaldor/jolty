# Copyright 2026 Maxence Faldor
"""Primitive device constructors defining models, terminals, and validated parameters.

Each constructor checks the physical dimension of every parameter at call time, so
passing a capacitance where a resistance is expected fails immediately. Omitted
parameters use the engine defaults documented on the reference models page.

Examples:
    >>> from jolty.units import kohm
    >>> resistor(10 * kohm).parameters
    {'resistance': 10000.0}
"""

import functools
import inspect
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
    CURRENT,
    DIMENSIONLESS,
    HEAT_CAPACITY,
    LENGTH,
    POWER,
    TEMPERATURE,
    THERMAL_RESISTANCE,
    TIME,
    VOLTAGE,
    Dimension,
    Exponent,
    Quantity,
    finite,
)
from jolty.waveforms import Dc, Waveform


@dataclass(frozen=True, slots=True)
class Device:
    """Placeable device specification storing model name, terminals, and SI parameters.

    Device parameters may carry statistical tolerances for Monte Carlo sampling.
    Devices may also declare operating-limit ratings checked during simulation and a
    manufacturer part number for bill-of-materials pricing.
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
        """Returns a copy of the device with a `relative` tolerance on `parameter`.

        The `parameter` argument uses dotted [path][jolty.ir.parameter] notation (such
        as ``waveform.dc`` for a source DC level), with `relative` expressed as a
        fraction (such as 0.01 for 1 %). Sampled values follow
        [`Distribution`][jolty.ir.Distribution]: uniform or normal within the
        tolerance, or log-uniform within a factor of ``1 + relative``. Monte Carlo
        analyses multiply sampled values by any shared `lot` factor.

        Args:
            parameter: Dotted parameter path targeting a numeric model value.
            relative: Relative tolerance fraction (such as 0.01 for 1 %).
            distribution: Sampling distribution across the tolerance interval.
            lot: Optional shared manufacturing lot correlating multiple instances.

        Returns:
            Copy of the device carrying the added parameter tolerance.

        Raises:
            ValueError: If `parameter` does not resolve to a numeric value, or if
                `distribution` rejects the tolerance.

        Examples:
            >>> from jolty.units import V
            >>> supply = voltage_source(5 * V).toleranced("waveform.dc", 0.05)
            >>> supply.tolerances["waveform.dc"].relative
            0.05
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
        """Returns a copy of the device with an operating limit on `stress`.

        Voltage ratings measure the first terminal minus the second (defaulting to the
        first two terminals), while current ratings measure current into one terminal
        (defaulting to the first). Power and temperature ratings take no terminals.
        Electrical power equals total power entering electrical terminals, and device
        temperature equals simulation temperature plus any thermal net rise when
        [heated][jolty.devices.Device.heated].

        Ratings enforce `maximum`, `minimum`, or both bounds (for example,
        ``maximum=50 * V, minimum=-50 * V`` for a 50 V bipolar capacitor). Simulations
        check every rating at the operating point and throughout transient runs.

        Args:
            stress: Electrical or thermal stress category to bound.
            maximum: Upper operating limit in units matching `stress`.
            minimum: Lower operating limit in units matching `stress`.
            terminals: Explicit terminal names overriding default rating pins.

        Returns:
            Copy of the device with the appended operating-limit rating.

        Raises:
            DimensionError: If a bound does not match the physical dimension of
                `stress`.
            ValueError: If `terminals` are invalid for the device or `stress`, or if
                bounds are missing or inverted.

        Examples:
            >>> from jolty.units import V, W, celsius, kohm, uF
            >>> resistor(1 * kohm).rated("power", 0.25 * W).ratings[0].maximum
            0.25
            >>> capacitor(1 * uF).rated("voltage", 50 * V).ratings[0].terminals
            ('a', 'b')
            >>> diode().rated("temperature", celsius(175)).ratings[0].maximum
            448.15
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
        """Returns a copy of the device enabled for self-heating power dissipation.

        Self-heating adds a ``thermal`` terminal connecting to a thermal net or to
        ambient ground. The device operates at the simulation temperature plus the
        thermal net temperature rise, injecting total dissipated electrical power into
        that thermal net. Supported models are resistors, diodes, JFETs, bipolar
        transistors, VBICs, MOSFETs, VDMOS devices, BSIM3, and BSIM4.

        Returns:
            Copy of the device exposing an electrothermal ``thermal`` terminal.

        Raises:
            ValueError: If the device model does not support self-heating or has
                invalid parameters.

        Examples:
            >>> diode().heated().terminals
            ('anode', 'cathode', 'thermal')
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
        """Returns a copy of the device held at an absolute `temperature`.

        Temperature-dependent model parameters scale to `temperature` independently of
        the global simulation temperature, matching SPICE ``TEMP`` and PSpice ``T_ABS``.
        Heated devices accumulate thermal net rise above this base temperature.
        Resistors held at 0 K generate zero thermal noise, whereas semiconductor
        junctions require temperatures above 0 K. Device temperature overrides are not
        differentiated by sensitivity analyses.

        Args:
            temperature: Absolute operating temperature in kelvins.

        Returns:
            Copy of the device pinned to `temperature` regardless of ambient.

        Raises:
            DimensionError: If `temperature` does not have temperature dimension.
            ValueError: If `temperature` is below 0 K or the device is a logic model
                whose timing ignores temperature.

        Examples:
            >>> from jolty.units import celsius, kohm
            >>> resistor(1 * kohm).held_at(celsius(85)).parameters["temperature"]
            358.15
        """
        kelvins = temperature.to(TEMPERATURE, "a device's own temperature")
        if kelvins < 0:
            message = (
                f"a device's own temperature lies at 0 K or above, not {kelvins!r} K"
            )
            raise ValueError(message)
        return self._surrounded("temperature", kelvins)

    def warmer_by(self, offset: Quantity) -> Device:
        """Returns a copy of the device offset in temperature by `offset`.

        Positive offsets warm the device relative to the simulation temperature and
        negative offsets cool it, matching SPICE ``DTEMP`` and PSpice ``T_REL_GLOBAL``.
        Scaling semantics match [`held_at`][jolty.devices.Device.held_at] while
        tracking changes in global simulation temperature.

        Args:
            offset: Temperature delta relative to global simulation ambient.

        Returns:
            Copy of the device shifted in temperature by `offset`.

        Raises:
            DimensionError: If `offset` does not have temperature dimension.
            ValueError: If the device is a logic model whose timing ignores
                temperature.

        Examples:
            >>> from jolty.units import K, kohm
            >>> resistor(1 * kohm).warmer_by(20 * K).parameters["temperature_offset"]
            20.0
        """
        kelvins = offset.to(TEMPERATURE, "a device's temperature offset")
        return self._surrounded("temperature_offset", finite(kelvins, "offset"))

    def _surrounded(self, key: str, kelvins: float) -> Device:
        """Returns a copy of the device with temperature `key` set to `kelvins`.

        Raises:
            ValueError: If the device model does not support per-instance temperatures.
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
        """Returns a copy of the device tagged with manufacturer part number `part`.

        Args:
            part: Non-empty manufacturer part number without outer whitespace.

        Returns:
            Copy of the device annotated with manufacturer part number `part`.

        Raises:
            ValueError: If `part` is empty or has leading or trailing whitespace.

        Examples:
            >>> diode().numbered("1N4148").part
            '1N4148'
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
    """Creates a `stress` rating for a device or component with terminals `own`.

    Rating rules and default terminal selection match
    [`Device.rated`][jolty.devices.Device.rated].

    Args:
        own: Ordered terminal names belonging to the device or component.
        stress: Electrical or thermal stress category to bound.
        maximum: Upper operating limit in units matching `stress`.
        minimum: Lower operating limit in units matching `stress`.
        terminals: Explicit terminal subset overriding default rating pins.

    Returns:
        Validated `ir.Rating` converted to SI base units.

    Raises:
        DimensionError: If a bound does not match the physical dimension of `stress`.
        ValueError: If `terminals` are not in `own`, have the wrong count for `stress`,
            or have missing or inverted bounds.
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
"""Dimension of a linear temperature coefficient, K⁻¹."""

PER_KELVIN_SQUARED: Final = DIMENSIONLESS / TEMPERATURE**2
"""Dimension of a quadratic temperature coefficient, K⁻²."""

PER_VOLT: Final = DIMENSIONLESS / VOLTAGE
"""Dimension of a linear voltage coefficient or channel-length modulation, V⁻¹."""

PER_VOLT_SQUARED: Final = DIMENSIONLESS / VOLTAGE**2
"""Dimension of a quadratic voltage coefficient, V⁻²."""

TRANSCONDUCTANCE_PARAMETER: Final = CURRENT / VOLTAGE**2
"""Dimension of a field-effect transistor transconductance parameter, A/V²."""

MOBILITY: Final = LENGTH**2 / (VOLTAGE * TIME)
"""Dimension of carrier mobility, m²/(V·s)."""

VOLTS_PER_KELVIN: Final = VOLTAGE / TEMPERATURE
"""Dimension of a voltage temperature coefficient, V/K."""

type _ParameterKind = Dimension | str | None


def _exponent(doubled: int) -> Exponent:
    """Decodes an SI base exponent from the doubled integer exported by the engine."""
    return doubled // 2 if doubled % 2 == 0 else Fraction(doubled, 2)


def _dimension(exponents: tuple[int, int, int, int, int]) -> Dimension:
    """Decodes a dimension from five doubled SI base exponents from the engine."""
    kilogram, meter, second, ampere, kelvin = exponents
    return Dimension(
        kilogram=_exponent(kilogram),
        meter=_exponent(meter),
        second=_exponent(second),
        ampere=_exponent(ampere),
        kelvin=_exponent(kelvin),
    )


@functools.cache
def _model_kinds(model: str) -> Mapping[str, _ParameterKind]:
    """Returns parameter kinds declared by the engine for `model` as dimensions."""
    return {
        name: _dimension(raw) if isinstance(raw, tuple) else raw
        for name, raw in _core.parameter_dimensions(model).items()
    }


def _quantity(value: object, dimension: Dimension, name: str) -> float:
    """Converts `value` to SI base units after verifying `dimension`.

    Raises:
        DimensionError: If `value` is not a [`Quantity`][jolty.units.Quantity] with
            `dimension`.
        ValueError: If the numeric value is not finite.
    """
    if not isinstance(value, Quantity):
        message = f"{name} takes a quantity in {dimension}, not {value!r}"
        raise DimensionError(message)
    return value.to(dimension, name)


def _parameter_value(
    name: str,
    value: object,
    kind: _ParameterKind,
    conditional: Mapping[str, Dimension] | None,
) -> JsonValue:
    """Validates and converts one parameter value to SI base units for the engine.

    Raises:
        DimensionError: If a quantity or plain number has the wrong kind or dimension.
        TypeError: If a choice parameter is not a string, integer, or boolean.
        ValueError: If a numeric value is not finite or a conditional tag has no
            resolved dimension.
    """
    if kind == "choice":
        if isinstance(value, bool | int | str):
            return value
        message = f"{name} takes a choice, not {value!r}"
        raise TypeError(message)
    if isinstance(kind, str):
        if conditional is None or kind not in conditional:
            message = f"{name} has no resolved dimension for condition `{kind}`"
            raise ValueError(message)
        kind = conditional[kind]
    if isinstance(kind, Dimension):
        return _quantity(value, kind, name)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return finite(value, name)
    message = f"{name} takes a plain number, not {value!r}"
    raise DimensionError(message)


def _device(
    model: str,
    values: Mapping[str, object],
    *,
    conditional: Mapping[str, Dimension] | None = None,
    card: Mapping[str, Quantity | float] | None = None,
    card_dimension: Callable[[str], Dimension | None] | None = None,
) -> Device:
    """Builds a device of `model` after validating `values` against engine metadata.

    Raises:
        DimensionError: If a parameter has the wrong dimension or kind.
        TypeError: If a choice parameter is not a string, integer, or boolean.
        ValueError: If a value is not finite or the engine rejects the device.
    """
    kinds = _model_kinds(model)
    parameters: dict[str, JsonValue] = {
        name: _parameter_value(name, value, kinds[name], conditional)
        for name, value in values.items()
        if value is not None
    }
    if card is not None and card_dimension is not None:
        _card(parameters, card, card_dimension)
    return device(model, parameters)


def resistor(  # noqa: PLR0913 - resistor value, tolerance, drift, and noise
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
    """Creates a resistor between terminals ``a`` and ``b``.

    When `tolerance` is given (such as 0.01 for 1 %), Monte Carlo analyses sample the
    resistance uniformly within that fraction. Linear and quadratic temperature
    coefficients scale resistance from `nominal_temperature` by
    ``1 + tc1 · ΔT + tc2 · ΔT²``. An `exponential_temperature_coefficient` in percent
    per kelvin (SPICE ``TCE``) instead scales resistance by ``1.01^(tce · ΔT)``.

    Linear and quadratic voltage coefficients scale resistance with terminal voltage
    ``v`` from ``a`` to ``b`` by ``1 + vc1 · v + vc2 · v²``. Negative effective
    resistances produced by large voltage coefficients are preserved, matching ngspice.

    In addition to thermal noise, a flicker coefficient ``KF`` adds flicker noise of
    ``KF · |I|^AF / f`` A²/Hz from resistor current ``I`` with exponent ``AF``.

    Args:
        resistance: Nominal resistance at `nominal_temperature` in ohms.
        tolerance: Optional uniform relative tolerance fraction on resistance.
        temperature_coefficient: Linear resistance drift coefficient (K⁻¹).
        quadratic_temperature_coefficient: Quadratic resistance drift coefficient (K⁻²).
        exponential_temperature_coefficient: Exponential resistance drift rate in
            percent per kelvin (SPICE ``TCE``).
        nominal_temperature: Reference temperature for drift scaling in kelvins.
        voltage_coefficient: Linear voltage coefficient of resistance (V⁻¹).
        quadratic_voltage_coefficient: Quadratic voltage coefficient of resistance
            (V⁻²).
        flicker_coefficient: Flicker noise scaling coefficient (SPICE ``KF``).
        flicker_exponent: Flicker noise current exponent (SPICE ``AF``).

    Returns:
        Two-terminal resistor `Device` across ``a`` and ``b``.

    Examples:
        >>> from jolty.units import K, V, kohm
        >>> resistor(1 * kohm, temperature_coefficient=1e-4 / K).parameters
        {'resistance': 1000.0, 'temperature_coefficient': 0.0001}
        >>> resistor(1 * kohm, voltage_coefficient=-2e-6 / V).parameters
        {'resistance': 1000.0, 'voltage_coefficient': -2e-06}
    """
    values = locals()
    values.pop("tolerance")
    made = _device("resistor", values)
    return made if tolerance is None else made.toleranced("resistance", tolerance)


def capacitor(  # noqa: PLR0913 - capacitance, tolerance, drift, and curvature
    capacitance: Quantity,
    *,
    tolerance: float | None = None,
    temperature_coefficient: Quantity | None = None,
    quadratic_temperature_coefficient: Quantity | None = None,
    nominal_temperature: Quantity | None = None,
    voltage_coefficient: Quantity | None = None,
    quadratic_voltage_coefficient: Quantity | None = None,
) -> Device:
    """Creates a capacitor between terminals ``a`` and ``b``.

    Capacitance temperature drift and Monte Carlo tolerance scaling match
    [`resistor`][jolty.devices.resistor]. Voltage coefficients scale capacitance with
    terminal voltage ``v`` from ``a`` to ``b`` by ``1 + vc1 · v + vc2 · v²``, and
    stored charge equals the integral of that capacitance from 0 V.

    Args:
        capacitance: Nominal zero-bias capacitance in farads.
        tolerance: Optional uniform relative tolerance fraction on capacitance.
        temperature_coefficient: Linear capacitance drift coefficient (K⁻¹).
        quadratic_temperature_coefficient: Quadratic capacitance drift coefficient
            (K⁻²).
        nominal_temperature: Reference temperature for drift scaling in kelvins.
        voltage_coefficient: Linear voltage coefficient of capacitance (V⁻¹).
        quadratic_voltage_coefficient: Quadratic voltage coefficient of capacitance
            (V⁻²).

    Returns:
        Two-terminal capacitor `Device` across ``a`` and ``b``.
    """
    values = locals()
    values.pop("tolerance")
    made = _device("capacitor", values)
    return made if tolerance is None else made.toleranced("capacitance", tolerance)


def inductor(
    inductance: Quantity,
    *,
    tolerance: float | None = None,
    temperature_coefficient: Quantity | None = None,
    quadratic_temperature_coefficient: Quantity | None = None,
    nominal_temperature: Quantity | None = None,
) -> Device:
    """Creates a linear inductor between terminals ``a`` and ``b``.

    Branch current flowing from ``a`` through the inductor to ``b`` is exposed as the
    unknown ``current``. Inductance temperature drift and Monte Carlo tolerance scaling
    match [`resistor`][jolty.devices.resistor].

    Args:
        inductance: Nominal inductance at `nominal_temperature` in henries.
        tolerance: Optional uniform relative tolerance fraction on inductance.
        temperature_coefficient: Linear inductance drift coefficient (K⁻¹).
        quadratic_temperature_coefficient: Quadratic inductance drift coefficient (K⁻²).
        nominal_temperature: Reference temperature for drift scaling in kelvins.

    Returns:
        Two-terminal linear inductor `Device` across ``a`` and ``b``.
    """
    values = locals()
    values.pop("tolerance")
    made = _device("inductor", values)
    return made if tolerance is None else made.toleranced("inductance", tolerance)


def coupled_inductors(
    windings: Sequence[Device], couplings: Mapping[tuple[int, int], float]
) -> Device:
    """Creates magnetically coupled inductors such as transformer or choke windings.

    Each entry in `windings` is an [`inductor`][jolty.devices.inductor] retaining its
    tolerance and temperature drift. Winding ``k`` (1-indexed) connects terminals
    ``a{k}`` and ``b{k}``, with branch current from ``a{k}`` to ``b{k}`` exposed as
    ``current{k}``.

    The `couplings` mapping specifies pairwise coupling coefficients in ``[-1, 1]`` by
    1-based winding indices. Mutual inductance equals the coefficient times the square
    root of the two inductances' product. Positive coupling adds flux for currents
    entering both ``a`` terminals, matching SPICE ``K`` elements. At least two windings
    are required, and each pair may be coupled at most once.

    Args:
        windings: Sequence of at least two plain unrated `inductor` devices.
        couplings: Pairwise coupling coefficients in ``[-1, 1]`` keyed by 1-based
            winding indices.

    Returns:
        Multi-winding coupled-inductor `Device` preserving winding tolerances.

    Raises:
        ValueError: If a winding is not a plain inductor, declares ratings or a part
            number, or sets an individual temperature instead of applying
            [`held_at`][jolty.devices.Device.held_at] to the coupled device as a whole.

    Examples:
        >>> from jolty.units import mH
        >>> transformer = coupled_inductors(
        ...     [inductor(1 * mH), inductor(4 * mH)], {(1, 2): 0.99}
        ... )
        >>> transformer.terminals
        ('a1', 'b1', 'a2', 'b2')
    """
    tolerances: dict[str, ir.Tolerance] = {}
    for number, winding in enumerate(windings, start=1):
        if winding.model != "inductor" or winding.ratings or winding.part is not None:
            message = f"winding {number} is an inductor that states no ratings or part"
            raise ValueError(message)
        if {"temperature", "temperature_offset"} & winding.parameters.keys():
            message = (
                f"winding {number} sets an individual temperature, which the coupled "
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
    """Creates a lossless transmission line (SPICE ``T``) with propagation `delay`.

    Port ``k`` (1 or 2) connects ``positive{k}`` and ``negative{k}``, with port current
    exposed as ``current{k}``. Each port presents `characteristic_impedance` in series
    with the wave launched from the opposite port one `delay` earlier. The line acts as
    a DC short between ports and introduces phase shift ``e^{-jωτ}`` in AC analyses.
    Transient stepping lands on waveform corners as they arrive at the opposite port.

    Args:
        characteristic_impedance: Port characteristic impedance in ohms.
        delay: One-way propagation delay between ports in seconds.

    Returns:
        Four-terminal lossless transmission-line `Device`.

    Examples:
        >>> from jolty.units import ns, ohm
        >>> line = transmission_line(50 * ohm, 1 * ns)
        >>> line.terminals
        ('positive1', 'negative1', 'positive2', 'negative2')
    """
    return _device("transmission_line", locals())


def thermal_resistance(resistance: Quantity) -> Device:
    """Creates a thermal resistance between thermal terminals ``a`` and ``b``, in K/W.

    Heat flows from the hotter terminal to the colder terminal at the temperature
    difference divided by `resistance`, analogous to electrical resistance. Each
    thermal terminal connects either to a thermal net (solving for temperature rise
    above ambient) or to ground representing ambient temperature.

    Args:
        resistance: Thermal resistance between ``a`` and ``b`` in K/W.

    Returns:
        Two-terminal thermal-resistance `Device` across ``a`` and ``b``.

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
    """Creates a thermal capacitance between thermal terminals ``a`` and ``b``, in J/K.

    Stored thermal energy scales linearly with the terminal temperature difference,
    controlling how quickly a thermal net responds to injected heat flow.

    Args:
        capacitance: Heat capacity between ``a`` and ``b`` in J/K.

    Returns:
        Two-terminal thermal-capacitance `Device` across ``a`` and ``b``.
    """
    return device(
        "thermal_capacitance",
        {"capacitance": capacitance.to(HEAT_CAPACITY, "heat capacity")},
    )


def heat_source(power: Quantity) -> Device:
    """Creates a constant heat source driving `power` (watts) from ``a`` into ``b``.

    Heat sources drive thermal power into a thermal network just as current sources
    drive electrical current, modeling external heat dissipation into a substrate or
    board.

    Args:
        power: Constant thermal power injected from ``a`` into ``b`` in watts.

    Returns:
        Two-terminal constant heat-source `Device` across ``a`` and ``b``.
    """
    return device("heat_source", {"power": power.to(POWER, "heat flow")})


def voltage_source(waveform: Waveform | Quantity) -> Device:
    """Creates an independent voltage source holding ``positive`` minus ``negative``.

    Passing a [`Quantity`][jolty.units.Quantity] creates a constant DC level. Branch
    current from ``positive`` through the source to ``negative`` is exposed as
    ``current``, taking negative values while supplying power.

    Args:
        waveform: Time-varying `Waveform` or constant DC voltage `Quantity`.

    Returns:
        Independent voltage-source `Device` across ``positive`` and ``negative``.
    """
    level = Dc(waveform) if isinstance(waveform, Quantity) else waveform
    parameters: dict[str, JsonValue] = {"waveform": level.to_json(VOLTAGE)}
    return device("voltage_source", parameters)


def current_source(waveform: Waveform | Quantity) -> Device:
    """Creates an independent current source driving ``positive`` to ``negative``.

    Current flows internally from ``positive`` to ``negative``, exiting into the
    circuit at ``negative``. Passing a [`Quantity`][jolty.units.Quantity] creates a
    constant DC level.

    Args:
        waveform: Time-varying `Waveform` or constant DC current `Quantity`.

    Returns:
        Independent current-source `Device` across ``positive`` and ``negative``.
    """
    level = Dc(waveform) if isinstance(waveform, Quantity) else waveform
    parameters: dict[str, JsonValue] = {"waveform": level.to_json(CURRENT)}
    return device("current_source", parameters)


def voltage_controlled_voltage_source(gain: float) -> Device:
    """Creates a voltage-controlled voltage source with output scaling `gain`.

    Output voltage ``positive`` minus ``negative`` equals `gain` times the control
    voltage ``control_positive`` minus ``control_negative``. Control terminals draw
    zero current.

    Args:
        gain: Dimensionless voltage ratio from control port to output port.

    Returns:
        Four-terminal VCVS `Device` with voltage gain `gain`.
    """
    return _device("voltage_controlled_voltage_source", locals())


def voltage_controlled_current_source(transconductance: Quantity) -> Device:
    """Creates a voltage-controlled current source with scaling `transconductance`.

    Output current flows from ``positive`` through the source to ``negative`` at
    `transconductance` times the control voltage ``control_positive`` minus
    ``control_negative``.

    Args:
        transconductance: Voltage-to-current transfer ratio in siemens.

    Returns:
        Four-terminal VCCS `Device` with transconductance `transconductance`.
    """
    return _device("voltage_controlled_current_source", locals())


def current_controlled_current_source(gain: float) -> Device:
    """Creates a current-controlled current source with current scaling `gain`.

    The control port from ``control_positive`` to ``control_negative`` acts as a
    zero-volt short circuit exposing sensed branch current as ``control``. Output
    current flows from ``positive`` through the source to ``negative`` at `gain` times
    that sensed current.

    Args:
        gain: Dimensionless current gain from sensed control branch to output.

    Returns:
        Four-terminal CCCS `Device` with current gain `gain`.
    """
    return _device("current_controlled_current_source", locals())


def current_controlled_voltage_source(transresistance: Quantity) -> Device:
    """Creates a current-controlled voltage source with scaling `transresistance`.

    The control port from ``control_positive`` to ``control_negative`` acts as a
    zero-volt short circuit exposing sensed current as ``control``. Output voltage
    ``positive`` minus ``negative`` equals `transresistance` times that current, and
    output branch current is exposed as ``current``.

    Args:
        transresistance: Current-to-voltage transfer ratio in ohms.

    Returns:
        Four-terminal CCVS `Device` with transresistance `transresistance`.
    """
    return _device("current_controlled_voltage_source", locals())


def behavioral_voltage_source(expression: str) -> Device:
    """Creates a behavioral voltage source driving ``positive`` minus ``negative``.

    The `expression` evaluates in SI units: ``v(a)`` reads node voltage ``a``,
    ``v(a, b)`` reads differential voltage ``a`` minus ``b``, ``i(p)`` reads sensing
    port current ``p``, and ``time`` reads simulation time. Terminals are ordered as
    ``positive``, ``negative``, each referenced voltage terminal, and ``p_positive``
    followed by ``p_negative`` for each short-circuit sensing port ``p`` (exposing
    unknown ``p``). Output branch current from ``positive`` to ``negative`` is exposed
    as ``current``. Refer to the expressions reference page for operators and
    functions.

    Args:
        expression: SI voltage expression over node voltages, currents, and time.

    Returns:
        Behavioral voltage-source `Device` with expression-derived terminals.

    Raises:
        ValueError: If `expression` cannot be parsed as a valid behavioral expression.

    Examples:
        >>> behavioral_voltage_source("1e5 * v(inp, inn)").terminals
        ('positive', 'negative', 'inp', 'inn')
    """
    return device("behavioral_voltage_source", {"expression": expression})


def behavioral_current_source(expression: str) -> Device:
    """Creates a behavioral current source driving `expression` across its terminals.

    Expression syntax and terminal ordering match
    [`behavioral_voltage_source`][jolty.devices.behavioral_voltage_source].

    Args:
        expression: SI current expression over node voltages, currents, and time.

    Returns:
        Behavioral current-source `Device` with expression-derived terminals.

    Raises:
        ValueError: If `expression` cannot be parsed as a valid behavioral expression.

    Examples:
        >>> behavioral_current_source("limit(1e-3 * v(c), -1e-3, 1e-3)").terminals
        ('positive', 'negative', 'c')
    """
    return device("behavioral_current_source", {"expression": expression})


def behavioral_capacitor(expression: str) -> Device:
    """Creates a charge-conserving behavioral capacitor defined by `expression`.

    Expression syntax (in coulombs) and terminal ordering match
    [`behavioral_voltage_source`][jolty.devices.behavioral_voltage_source]. Capacitor
    current from ``positive`` through the device to ``negative`` equals the time
    derivative of charge using exact symbolic derivatives, conserving charge across
    all operating points.

    Args:
        expression: SI charge expression in coulombs differentiated over time.

    Returns:
        Charge-conserving behavioral capacitor `Device`.

    Raises:
        ValueError: If `expression` cannot be parsed as a valid behavioral expression.

    Examples:
        >>> behavioral_capacitor("1e-12 * tanh(v(positive, negative))").terminals
        ('positive', 'negative')
    """
    return device("behavioral_capacitor", {"expression": expression})


def voltage_controlled_switch(
    *,
    on_resistance: Quantity | None = None,
    off_resistance: Quantity | None = None,
    on_voltage: Quantity | None = None,
    off_voltage: Quantity | None = None,
) -> Device:
    """Creates a PSpice voltage-controlled switch across ``positive`` and ``negative``.

    Control voltage ``control_positive`` minus ``control_negative`` interpolates
    resistance logarithmically between `off_resistance` at `off_voltage` and
    `on_resistance` at `on_voltage`, clamping outside that range. Control terminals
    draw zero current, and omitted parameters use PSpice defaults.

    Args:
        on_resistance: Closed-state switch resistance at or above `on_voltage`.
        off_resistance: Open-state switch resistance at or below `off_voltage`.
        on_voltage: Control voltage threshold for full conduction.
        off_voltage: Control voltage threshold for open-state resistance.

    Returns:
        Four-terminal voltage-controlled switch `Device`.
    """
    return _device("voltage_controlled_switch", locals())


def current_controlled_switch(
    *,
    on_resistance: Quantity | None = None,
    off_resistance: Quantity | None = None,
    on_current: Quantity | None = None,
    off_current: Quantity | None = None,
) -> Device:
    """Creates a PSpice current-controlled switch across ``positive`` and ``negative``.

    The control port from ``control_positive`` to ``control_negative`` acts as a short
    circuit whose sensed current ``control`` interpolates resistance logarithmically
    from `off_resistance` at `off_current` to `on_resistance` at `on_current`, matching
    [`voltage_controlled_switch`][jolty.devices.voltage_controlled_switch]. Omitted
    parameters use PSpice defaults.

    Args:
        on_resistance: Closed-state switch resistance at or above `on_current`.
        off_resistance: Open-state switch resistance at or below `off_current`.
        on_current: Sensed control current threshold for full conduction.
        off_current: Sensed control current threshold for open-state resistance.

    Returns:
        Four-terminal current-controlled switch `Device`.
    """
    return _device("current_controlled_switch", locals())


def photocell(
    resistance: Quantity,
    *,
    illuminance: float | None = None,
    gamma: float | None = None,
    dark_resistance: Quantity | None = None,
    response_time: Quantity | None = None,
) -> Device:
    """Creates a light-dependent resistor (photocell) between ``a`` and ``b``.

    Control voltage ``light_positive`` minus ``light_negative`` represents illuminance
    at 1 V per lux with zero control current. Cell resistance equals `resistance` at
    the reference `illuminance` (in lux) and scales with illuminance to the power
    `gamma`, placed in parallel with `dark_resistance`. When `response_time` is
    specified, conductance tracks light changes with that first-order time constant;
    otherwise conductance responds instantaneously.

    Args:
        resistance: Cell resistance at reference `illuminance` in ohms.
        illuminance: Reference illuminance in lux corresponding to `resistance`.
        gamma: Power-law exponent relating illuminance to conductance.
        dark_resistance: Shunt resistance in complete darkness in ohms.
        response_time: First-order photoconductive time constant in seconds.

    Returns:
        Four-terminal light-dependent resistor `Device`.

    Examples:
        >>> from jolty.units import kohm, ms
        >>> cell = photocell(10 * kohm, gamma=0.8, response_time=20 * ms)
        >>> cell.parameters["response_time"]
        0.02
    """
    return _device("photocell", locals())


def voltage_controlled_memristor(  # noqa: PLR0913 - VTEAM model parameters
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
    """Creates a VTEAM voltage-controlled memristor between its two terminals.

    Device resistance follows internal state according to `law`, spanning
    `on_resistance` at state 0 to `off_resistance` at state 1. Voltage ``positive``
    minus ``negative`` above `off_voltage` drives state toward 1 at rate
    ``off_rate · (v/v_off - 1)^off_exponent``. Negative voltage below `on_voltage`
    drives state toward 0 at rate ``on_rate · (v/v_on - 1)^on_exponent``, while
    voltages between thresholds hold the state constant.

    Each rate specifies state velocity in full ranges per second at twice its
    threshold. A fractional `window` slows state evolution near 0 and 1 without
    crossing either bound, starting from initial `state`.

    Args:
        on_resistance: Low-resistance state value at state 0 in ohms.
        off_resistance: High-resistance state value at state 1 in ohms.
        on_voltage: Negative voltage threshold driving state toward 0.
        off_voltage: Positive voltage threshold driving state toward 1.
        on_rate: State transition velocity toward 0 in inverse seconds.
        off_rate: State transition velocity toward 1 in inverse seconds.
        on_exponent: Power-law nonlinearity exponent for state-0 switching.
        off_exponent: Power-law nonlinearity exponent for state-1 switching.
        law: Resistance interpolation law (`'linear'` or `'exponential'`).
        state: Initial normalized internal state in ``[0, 1]``.
        window: Boundary window fraction slowing drift near 0 and 1.

    Returns:
        Two-terminal VTEAM voltage-controlled memristor `Device`.

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
    return _device("voltage_controlled_memristor", locals())


def current_controlled_memristor(  # noqa: PLR0913 - HP memristor parameters
    *,
    on_resistance: Quantity,
    off_resistance: Quantity,
    charge: Quantity,
    law: Literal["linear", "exponential"] | None = None,
    state: float | None = None,
    window: float | None = None,
) -> Device:
    """Creates an HP current-controlled memristor across ``positive`` and ``negative``.

    Device resistance follows internal state according to `law`, spanning
    `on_resistance` at state 0 to `off_resistance` at state 1. Current from ``positive``
    to ``negative`` moves state toward 1 in proportion to integrated charge divided by
    `charge`, modeling dopant drift across the full device length. A fractional
    `window` slows state motion near 0 and 1 without crossing either bound, starting
    from initial `state`.

    Args:
        on_resistance: Low-resistance state value at state 0 in ohms.
        off_resistance: High-resistance state value at state 1 in ohms.
        charge: Full-scale dopant-drift charge in coulombs.
        law: Resistance interpolation law (`'linear'` or `'exponential'`).
        state: Initial normalized internal state in ``[0, 1]``.
        window: Boundary window fraction slowing drift near 0 and 1.

    Returns:
        Two-terminal HP current-controlled memristor `Device`.

    Examples:
        >>> from jolty.units import kohm, ohm, uC
        >>> cell = current_controlled_memristor(
        ...     on_resistance=100 * ohm, off_resistance=16 * kohm, charge=2 * uC
        ... )
        >>> cell.parameters["charge"]
        2e-06
    """
    return _device("current_controlled_memristor", locals())


def diode(  # noqa: PLR0913 - ngspice junction diode model parameters
    *,
    saturation_current: Quantity | None = None,
    emission_coefficient: float | None = None,
    series_resistance: Quantity | None = None,
    junction_capacitance: Quantity | None = None,
    junction_potential: Quantity | None = None,
    grading_coefficient: float | None = None,
    depletion_coefficient: float | None = None,
    transit_time: Quantity | None = None,
    forward_knee_current: Quantity | None = None,
    reverse_knee_current: Quantity | None = None,
    recombination_saturation_current: Quantity | None = None,
    recombination_emission_coefficient: float | None = None,
    sidewall_saturation_current: Quantity | None = None,
    sidewall_emission_coefficient: float | None = None,
    sidewall_series_resistance: Quantity | None = None,
    sidewall_junction_capacitance: Quantity | None = None,
    sidewall_junction_potential: Quantity | None = None,
    sidewall_grading_coefficient: float | None = None,
    sidewall_depletion_coefficient: float | None = None,
    sidewall_forward_knee_current: Quantity | None = None,
    tunneling_saturation_current: Quantity | None = None,
    sidewall_tunneling_saturation_current: Quantity | None = None,
    tunneling_emission_coefficient: float | None = None,
    tunneling_saturation_current_temperature_exponent: float | None = None,
    tunneling_energy_gap_factor: float | None = None,
    breakdown_voltage: Quantity | None = None,
    breakdown_current: Quantity | None = None,
    breakdown_emission_coefficient: float | None = None,
    breakdown_temperature_coefficient: Quantity | None = None,
    temperature_model: Literal[0, 1, 2] | None = None,
    capacitance_temperature_model: Literal[0, 1] | None = None,
    energy_gap: Quantity | None = None,
    energy_gap_alpha: Quantity | None = None,
    energy_gap_beta: Quantity | None = None,
    saturation_current_temperature_exponent: float | None = None,
    junction_capacitance_temperature_coefficient: Quantity | None = None,
    sidewall_junction_capacitance_temperature_coefficient: Quantity | None = None,
    junction_potential_temperature_coefficient: Quantity | None = None,
    sidewall_junction_potential_temperature_coefficient: Quantity | None = None,
    series_resistance_temperature_coefficient: Quantity | None = None,
    series_resistance_quadratic_temperature_coefficient: Quantity | None = None,
    transit_time_temperature_coefficient: Quantity | None = None,
    transit_time_quadratic_temperature_coefficient: Quantity | None = None,
    grading_coefficient_temperature_coefficient: Quantity | None = None,
    grading_coefficient_quadratic_temperature_coefficient: Quantity | None = None,
    nominal_temperature: Quantity | None = None,
    flicker_coefficient: float | None = None,
    flicker_exponent: float | None = None,
) -> Device:
    """Creates a junction diode from ``anode`` to ``cathode`` matching ngspice 47.

    Omitted parameters use ngspice defaults. The bottom junction conducts ideal
    diffusion, recombination, and tunneling currents softened by knee currents. The
    sidewall junction conducts in parallel with the bottom junction, or behind a
    dedicated sidewall series resistance when specified. Above `breakdown_voltage`,
    reverse current grows exponentially using `breakdown_current` and
    `breakdown_emission_coefficient` (defaulting to `emission_coefficient`).

    Away from `nominal_temperature`, saturation currents scale with the energy gap
    (constant, or Varshni's law at `temperature_model` 2, matching SPICE ``TLEV``).
    Junction potentials and capacitances follow the silicon band gap at
    `capacitance_temperature_model` 0 (SPICE ``TLEVC``) and explicit coefficients at 1.
    Breakdown voltage shifts with temperature in V/K at temperature model 0 and
    relatively in 1/K at models 1 and 2. Series resistances, transit time, and grading
    coefficients follow their explicit ngspice temperature coefficients.

    Series resistances generate thermal noise and junction currents generate shot noise.
    With flicker coefficient ``KF`` and exponent ``AF``, each junction current ``I``
    adds flicker noise of ``KF · |I|^AF / f`` A²/Hz.

    Args:
        saturation_current: Bottom-junction reverse saturation current (``IS``).
        emission_coefficient: Bottom-junction forward ideality factor (``N``).
        series_resistance: Ohmic bottom series resistance in ohms (``RS``).
        junction_capacitance: Zero-bias bottom depletion capacitance (``CJO``).
        junction_potential: Bottom-junction built-in potential in volts (``VJ``).
        grading_coefficient: Bottom-junction depletion grading exponent (``M``).
        depletion_coefficient: Forward-bias depletion capacitance linearization fraction
            (``FC``).
        transit_time: Minority-carrier diffusion transit time in seconds (``TT``).
        forward_knee_current: High-injection forward knee current (``IKF``).
        reverse_knee_current: Reverse high-injection knee current (``IKR``).
        recombination_saturation_current: Depletion-region recombination saturation
            current (``ISR``).
        recombination_emission_coefficient: Recombination current ideality factor
            (``NR``).
        sidewall_saturation_current: Sidewall-junction reverse saturation current
            (``ISS``).
        sidewall_emission_coefficient: Sidewall-junction forward ideality factor
            (``NS``).
        sidewall_series_resistance: Dedicated sidewall ohmic series resistance in ohms
            (``RSW``).
        sidewall_junction_capacitance: Zero-bias sidewall depletion capacitance
            (``CJSW``).
        sidewall_junction_potential: Sidewall-junction built-in potential in volts
            (``VJSW``).
        sidewall_grading_coefficient: Sidewall-junction depletion grading exponent
            (``MJSW``).
        sidewall_depletion_coefficient: Sidewall forward-bias capacitance linearization
            fraction (``FCS``).
        sidewall_forward_knee_current: Sidewall high-injection forward knee current
            (``IKP``).
        tunneling_saturation_current: Bottom trap-assisted tunneling saturation current
            (``JTUN``).
        sidewall_tunneling_saturation_current: Sidewall trap-assisted tunneling
            saturation current (``JTUNSW``).
        tunneling_emission_coefficient: Tunneling exponential slope factor (``NTUN``).
        tunneling_saturation_current_temperature_exponent: Temperature exponent for
            tunneling saturation current (``XTITUN``).
        tunneling_energy_gap_factor: Band-gap scaling factor for tunneling temperature
            drift (``KEG``).
        breakdown_voltage: Reverse avalanche breakdown voltage in volts (``BV``).
        breakdown_current: Reverse current at `breakdown_voltage` (``IBV``).
        breakdown_emission_coefficient: Reverse breakdown ideality factor (``NBV``).
        breakdown_temperature_coefficient: Breakdown voltage temperature coefficient in
            V/K or K⁻¹ (``TCV``).
        temperature_model: Saturation and breakdown temperature selector (``TLEV``).
        capacitance_temperature_model: Junction capacitance temperature selector
            (``TLEVC``).
        energy_gap: Semiconductor band-gap energy in joules or eV-volts (``EG``).
        energy_gap_alpha: Varshni band-gap temperature slope coefficient (``GAP1``).
        energy_gap_beta: Varshni band-gap reference temperature offset (``GAP2``).
        saturation_current_temperature_exponent: Saturation current temperature exponent
            (``XTI``).
        junction_capacitance_temperature_coefficient: Linear temperature coefficient of
            `junction_capacitance` (``CTA``).
        sidewall_junction_capacitance_temperature_coefficient: Linear temperature
            coefficient of sidewall capacitance (``CTP``).
        junction_potential_temperature_coefficient: Linear temperature coefficient of
            `junction_potential` (``PTA``).
        sidewall_junction_potential_temperature_coefficient: Linear temperature
            coefficient of sidewall potential (``PTP``).
        series_resistance_temperature_coefficient: Linear temperature coefficient of
            series resistance (``TRS1``).
        series_resistance_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of series resistance (``TRS2``).
        transit_time_temperature_coefficient: Linear temperature coefficient of transit
            time (``TTT1``).
        transit_time_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of transit time (``TTT2``).
        grading_coefficient_temperature_coefficient: Linear temperature coefficient of
            grading exponent (``TM1``).
        grading_coefficient_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of grading exponent (``TM2``).
        nominal_temperature: Model extraction temperature in kelvins (``TNOM``).
        flicker_coefficient: Flicker noise scaling coefficient (``KF``).
        flicker_exponent: Flicker noise current exponent (``AF``).

    Returns:
        Junction diode `Device` across ``anode`` and ``cathode``.

    Raises:
        DimensionError: If a parameter has the wrong physical dimension; breakdown
            voltage temperature coefficient requires V/K at temperature model 0 and 1/K
            at models 1 and 2.
    """
    values = locals()
    breakdown = PER_KELVIN if temperature_model in {1, 2} else VOLTS_PER_KELVIN
    return _device("diode", values, conditional={"diode_tbv": breakdown})


def jfet(  # noqa: PLR0913 - SPICE JFET model parameters
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
    """Creates a JFET across ``drain``, ``gate``, and ``source`` matching ngspice.

    The model implements SPICE JFET equations with the Sydney University doping tail
    used in ngspice: two gate junctions and square-law drain conduction above threshold.
    Omitted parameters use SPICE defaults, with `doping_tail` equal to 1 giving the
    standard square law. Nonzero drain or source resistance adds internal node unknown
    ``drain`` or ``source``.

    Away from `nominal_temperature`, gate-junction saturation current scales with the
    energy gap and temperature exponent, while junction potential and capacitances
    follow the silicon band gap. Threshold voltage shifts by its linear coefficient and
    transconductance scales by its temperature exponent, matching ngspice.

    Series resistances generate thermal noise, and the channel produces thermal noise
    of ``8kT · gm / 3`` A²/Hz. With flicker coefficient ``KF`` and exponent ``AF``,
    drain current ``Id`` adds flicker noise of ``KF · |Id|^AF / f`` A²/Hz.

    Args:
        channel: Channel carrier polarity (`'n'` or `'p'`).
        threshold_voltage: Pinch-off threshold voltage in volts (``VTO``).
        transconductance_parameter: Square-law transconductance in A/V² (``BETA``).
        channel_length_modulation: Output conductance slope in V⁻¹ (``LAMBDA``).
        drain_resistance: Extrinsic ohmic drain resistance in ohms (``RD``).
        source_resistance: Extrinsic ohmic source resistance in ohms (``RS``).
        gate_source_capacitance: Zero-bias gate-source capacitance (``CGS``).
        gate_drain_capacitance: Zero-bias gate-drain capacitance (``CGD``).
        junction_potential: Gate-junction built-in potential in volts (``PB``).
        saturation_current: Gate-junction reverse saturation current (``IS``).
        depletion_coefficient: Forward-bias capacitance linearization fraction (``FC``).
        doping_tail: Sydney University power-law conduction exponent (``B``).
        threshold_voltage_temperature_coefficient: Linear temperature shift of
            `threshold_voltage` in V/K (``TCV``).
        transconductance_parameter_temperature_exponent: Temperature exponent of
            `transconductance_parameter` (``BEX``).
        energy_gap: Gate-junction band-gap energy at 0 K (``EG``).
        saturation_current_temperature_exponent: Gate-junction saturation current
            temperature exponent (``XTI``).
        nominal_temperature: Model extraction temperature in kelvins (``TNOM``).
        flicker_coefficient: Flicker noise scaling coefficient (``KF``).
        flicker_exponent: Flicker noise drain-current exponent (``AF``).

    Returns:
        Three-terminal JFET `Device` across ``drain``, ``gate``, and ``source``.

    Examples:
        >>> from jolty.units import V, mA
        >>> jfet(channel="p", transconductance_parameter=1 * mA / V**2).parameters
        {'channel': 'p', 'transconductance_parameter': 0.001}
    """
    return _device("jfet", locals())


def bipolar_transistor(  # noqa: PLR0913 - Gummel-Poon BJT model parameters
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
    temperature_model: Literal[0, 1] | None = None,
    capacitance_temperature_model: Literal[0, 1] | None = None,
    high_injection_exponent: float | None = None,
    forward_beta_temperature_coefficient: Quantity | None = None,
    forward_beta_quadratic_temperature_coefficient: Quantity | None = None,
    reverse_beta_temperature_coefficient: Quantity | None = None,
    reverse_beta_quadratic_temperature_coefficient: Quantity | None = None,
    forward_knee_current_temperature_coefficient: Quantity | None = None,
    forward_knee_current_quadratic_temperature_coefficient: Quantity | None = None,
    reverse_knee_current_temperature_coefficient: Quantity | None = None,
    reverse_knee_current_quadratic_temperature_coefficient: Quantity | None = None,
    forward_early_voltage_temperature_coefficient: Quantity | None = None,
    forward_early_voltage_quadratic_temperature_coefficient: Quantity | None = None,
    reverse_early_voltage_temperature_coefficient: Quantity | None = None,
    reverse_early_voltage_quadratic_temperature_coefficient: Quantity | None = None,
    base_resistance_temperature_coefficient: Quantity | None = None,
    base_resistance_quadratic_temperature_coefficient: Quantity | None = None,
    min_base_resistance_temperature_coefficient: Quantity | None = None,
    min_base_resistance_quadratic_temperature_coefficient: Quantity | None = None,
    base_resistance_current_temperature_coefficient: Quantity | None = None,
    base_resistance_current_quadratic_temperature_coefficient: Quantity | None = None,
    collector_resistance_temperature_coefficient: Quantity | None = None,
    collector_resistance_quadratic_temperature_coefficient: Quantity | None = None,
    emitter_resistance_temperature_coefficient: Quantity | None = None,
    emitter_resistance_quadratic_temperature_coefficient: Quantity | None = None,
    forward_emission_coefficient_temperature_coefficient: Quantity | None = None,
    forward_emission_coefficient_quadratic_temperature_coefficient: Quantity
    | None = None,
    reverse_emission_coefficient_temperature_coefficient: Quantity | None = None,
    reverse_emission_coefficient_quadratic_temperature_coefficient: Quantity
    | None = None,
    base_emitter_leakage_emission_coefficient_temperature_coefficient: Quantity
    | None = None,
    base_emitter_leakage_emission_coefficient_quadratic_temperature_coefficient: Quantity  # noqa: E501 - the coefficient named in full, as the glossary names coefficients
    | None = None,
    base_collector_leakage_emission_coefficient_temperature_coefficient: Quantity
    | None = None,
    base_collector_leakage_emission_coefficient_quadratic_temperature_coefficient: Quantity  # noqa: E501 - the coefficient named in full, as the glossary names coefficients
    | None = None,
    substrate_emission_coefficient_temperature_coefficient: Quantity | None = None,
    substrate_emission_coefficient_quadratic_temperature_coefficient: Quantity
    | None = None,
    transit_time_current_temperature_coefficient: Quantity | None = None,
    transit_time_current_quadratic_temperature_coefficient: Quantity | None = None,
    forward_transit_time_temperature_coefficient: Quantity | None = None,
    forward_transit_time_quadratic_temperature_coefficient: Quantity | None = None,
    reverse_transit_time_temperature_coefficient: Quantity | None = None,
    reverse_transit_time_quadratic_temperature_coefficient: Quantity | None = None,
    base_emitter_grading_temperature_coefficient: Quantity | None = None,
    base_emitter_grading_quadratic_temperature_coefficient: Quantity | None = None,
    base_collector_grading_temperature_coefficient: Quantity | None = None,
    base_collector_grading_quadratic_temperature_coefficient: Quantity | None = None,
    substrate_grading_temperature_coefficient: Quantity | None = None,
    substrate_grading_quadratic_temperature_coefficient: Quantity | None = None,
    base_emitter_capacitance_temperature_coefficient: Quantity | None = None,
    base_collector_capacitance_temperature_coefficient: Quantity | None = None,
    substrate_capacitance_temperature_coefficient: Quantity | None = None,
    base_emitter_potential_temperature_coefficient: Quantity | None = None,
    base_collector_potential_temperature_coefficient: Quantity | None = None,
    substrate_potential_temperature_coefficient: Quantity | None = None,
    nominal_temperature: Quantity | None = None,
    flicker_coefficient: float | None = None,
    flicker_exponent: float | None = None,
) -> Device:
    """Creates a Gummel-Poon BJT across ``collector``, ``base``, and ``emitter``.

    The model follows ngspice Gummel-Poon equations with SPICE defaults for omitted
    parameters (infinite Early voltages and knee currents by default). Nonzero base,
    collector, or emitter resistance adds internal node unknown ``base``, ``collector``,
    or ``emitter``.

    Setting `substrate` to ``"vertical"`` or ``"lateral"`` adds a fourth ``substrate``
    terminal connected through a junction to the collector (vertical) or base
    (lateral). The substrate junction conducts minimum conductance plus diode current
    when `substrate_saturation_current` is nonzero, with depletion capacitance growing
    linearly in forward bias. Substrate parameters require `substrate` to be set.

    When `excess_phase` (SPICE ``PTF``, in degrees) is set, a second-order Bessel
    filter delays forward transport current by ``PTF · π / 180 · TF``, adding
    approximately ``PTF`` degrees of phase at ``1 / (2π · TF)``. The filter exposes
    internal unknowns ``delayed_transport`` and ``delayed_transport_slope``.

    Away from `nominal_temperature`, saturation currents scale with energy gap and
    temperature exponent. Forward and reverse betas scale by ``(T/Tn)^XTB`` at
    ``temperature_model=0`` (SPICE ``TLEV``) and ``1 + XTB · ΔT`` at
    ``temperature_model=1``. Junction potentials and capacitances follow silicon band
    gap at ``capacitance_temperature_model=0`` (SPICE ``TLEVC``) and explicit linear
    coefficients at model 1.

    Parameters with linear and quadratic temperature coefficients (HSPICE ``TRC1`` and
    ``TRC2`` family) scale by ``1 + c1 · ΔT + c2 · ΔT²``, overriding the beta exponent
    when supplied on beta. The ``high_injection_exponent`` parameter (SPICE ``NKF``)
    shapes the base-charge high-injection term ``(1 + 4 · q2)^NKF``.

    Series resistances generate thermal noise, while collector and base currents
    generate shot noise. With flicker coefficient ``KF`` and exponent ``AF``, base
    current ``Ib`` adds flicker noise of ``KF · |Ib|^AF / f`` A²/Hz.

    Args:
        polarity: Transistor doping polarity (`'npn'` or `'pnp'`).
        saturation_current: Transport saturation current in amperes (``IS``).
        forward_beta: Ideal maximum forward current gain (``BF``).
        reverse_beta: Ideal maximum reverse current gain (``BR``).
        forward_emission_coefficient: Forward transport ideality factor (``NF``).
        reverse_emission_coefficient: Reverse transport ideality factor (``NR``).
        forward_early_voltage: Forward Early voltage in volts (``VAF``).
        reverse_early_voltage: Reverse Early voltage in volts (``VAR``).
        forward_knee_current: Forward beta high-current roll-off knee (``IKF``).
        reverse_knee_current: Reverse beta high-current roll-off knee (``IKR``).
        base_emitter_leakage_current: Base-emitter nonideal leakage saturation current
            (``ISE``).
        base_emitter_leakage_emission_coefficient: Base-emitter leakage ideality factor
            (``NE``).
        base_collector_leakage_current: Base-collector nonideal leakage saturation
            current (``ISC``).
        base_collector_leakage_emission_coefficient: Base-collector leakage ideality
            factor (``NC``).
        base_resistance: Zero-bias base resistance in ohms (``RB``).
        min_base_resistance: High-current minimum base resistance in ohms (``RBM``).
        base_resistance_current: Base current where base resistance halves toward
            minimum (``IRB``).
        emitter_resistance: Extrinsic emitter series resistance in ohms (``RE``).
        collector_resistance: Extrinsic collector series resistance in ohms (``RC``).
        base_emitter_capacitance: Zero-bias base-emitter depletion capacitance
            (``CJE``).
        base_emitter_potential: Base-emitter built-in potential in volts (``VJE``).
        base_emitter_grading: Base-emitter depletion grading exponent (``MJE``).
        forward_transit_time: Ideal forward base transit time in seconds (``TF``).
        transit_time_bias_coefficient: Forward transit-time bias-dependence multiplier
            (``XTF``).
        transit_time_voltage: Base-collector voltage scaling `forward_transit_time`
            (``VTF``).
        transit_time_current: High-current onset for `forward_transit_time` rise
            (``ITF``).
        base_collector_capacitance: Zero-bias total base-collector depletion capacitance
            (``CJC``).
        base_collector_potential: Base-collector built-in potential in volts (``VJC``).
        base_collector_grading: Base-collector depletion grading exponent (``MJC``).
        internal_base_fraction: Fraction of `base_collector_capacitance` on intrinsic
            base (``XCJC``).
        reverse_transit_time: Ideal reverse transit time in seconds (``TR``).
        depletion_coefficient: Forward-bias depletion capacitance linearization fraction
            (``FC``).
        excess_phase: Excess phase in degrees at `1 / (2π · TF)` (``PTF``).
        substrate: Substrate junction orientation (`'vertical'` or `'lateral'`).
        substrate_saturation_current: Substrate-junction saturation current in amperes
            (``ISS``).
        substrate_emission_coefficient: Substrate-junction forward ideality factor
            (``NS``).
        substrate_capacitance: Zero-bias substrate depletion capacitance (``CJS``).
        substrate_potential: Substrate built-in potential in volts (``VJS``).
        substrate_grading: Substrate depletion grading exponent (``MJS``).
        energy_gap: Band-gap energy for saturation scaling (``EG``).
        saturation_current_temperature_exponent: Saturation current temperature exponent
            (``XTI``).
        beta_temperature_exponent: Forward and reverse beta exponent (``XTB``).
        temperature_model: Beta temperature scaling law selector (``TLEV``).
        capacitance_temperature_model: Junction capacitance temperature law selector
            (``TLEVC``).
        high_injection_exponent: High-injection base-charge roll-off exponent (``NKF``).
        forward_beta_temperature_coefficient: Linear temperature coefficient of
            `forward_beta` (K⁻¹).
        forward_beta_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of `forward_beta` (K⁻²).
        reverse_beta_temperature_coefficient: Linear temperature coefficient of
            `reverse_beta` (K⁻¹).
        reverse_beta_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of `reverse_beta` (K⁻²).
        forward_knee_current_temperature_coefficient: Linear temperature coefficient of
            `forward_knee_current` (K⁻¹).
        forward_knee_current_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of `forward_knee_current` (K⁻²).
        reverse_knee_current_temperature_coefficient: Linear temperature coefficient of
            `reverse_knee_current` (K⁻¹).
        reverse_knee_current_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of `reverse_knee_current` (K⁻²).
        forward_early_voltage_temperature_coefficient: Linear temperature coefficient of
            `forward_early_voltage` (K⁻¹).
        forward_early_voltage_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of `forward_early_voltage` (K⁻²).
        reverse_early_voltage_temperature_coefficient: Linear temperature coefficient of
            `reverse_early_voltage` (K⁻¹).
        reverse_early_voltage_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of `reverse_early_voltage` (K⁻²).
        base_resistance_temperature_coefficient: Linear temperature coefficient of
            `base_resistance` (K⁻¹).
        base_resistance_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of `base_resistance` (K⁻²).
        min_base_resistance_temperature_coefficient: Linear temperature coefficient of
            `min_base_resistance` (K⁻¹).
        min_base_resistance_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of `min_base_resistance` (K⁻²).
        base_resistance_current_temperature_coefficient: Linear temperature coefficient
            of `base_resistance_current` (K⁻¹).
        base_resistance_current_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of `base_resistance_current` (K⁻²).
        collector_resistance_temperature_coefficient: Linear temperature coefficient of
            `collector_resistance` (K⁻¹).
        collector_resistance_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of `collector_resistance` (K⁻²).
        emitter_resistance_temperature_coefficient: Linear temperature coefficient of
            `emitter_resistance` (K⁻¹).
        emitter_resistance_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of `emitter_resistance` (K⁻²).
        forward_emission_coefficient_temperature_coefficient: Linear temperature
            coefficient of `forward_emission_coefficient` (K⁻¹).
        forward_emission_coefficient_quadratic_temperature_coefficient: Quadratic
            temperature coefficient of ``NF`` (K⁻²).
        reverse_emission_coefficient_temperature_coefficient: Linear temperature
            coefficient of `reverse_emission_coefficient` (K⁻¹).
        reverse_emission_coefficient_quadratic_temperature_coefficient: Quadratic
            temperature coefficient of ``NR`` (K⁻²).
        base_emitter_leakage_emission_coefficient_temperature_coefficient: Linear
            temperature coefficient of ``NE`` (K⁻¹).
        base_emitter_leakage_emission_coefficient_quadratic_temperature_coefficient:
            Quadratic temperature coefficient of ``NE`` (K⁻²).
        base_collector_leakage_emission_coefficient_temperature_coefficient: Linear
            temperature coefficient of ``NC`` (K⁻¹).
        base_collector_leakage_emission_coefficient_quadratic_temperature_coefficient:
            Quadratic temperature coefficient of ``NC`` (K⁻²).
        substrate_emission_coefficient_temperature_coefficient: Linear temperature
            coefficient of `substrate_emission_coefficient` (K⁻¹).
        substrate_emission_coefficient_quadratic_temperature_coefficient: Quadratic
            temperature coefficient of ``NS`` (K⁻²).
        transit_time_current_temperature_coefficient: Linear temperature coefficient of
            `transit_time_current` (K⁻¹).
        transit_time_current_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of `transit_time_current` (K⁻²).
        forward_transit_time_temperature_coefficient: Linear temperature coefficient of
            `forward_transit_time` (K⁻¹).
        forward_transit_time_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of `forward_transit_time` (K⁻²).
        reverse_transit_time_temperature_coefficient: Linear temperature coefficient of
            `reverse_transit_time` (K⁻¹).
        reverse_transit_time_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of `reverse_transit_time` (K⁻²).
        base_emitter_grading_temperature_coefficient: Linear temperature coefficient of
            `base_emitter_grading` (K⁻¹).
        base_emitter_grading_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of `base_emitter_grading` (K⁻²).
        base_collector_grading_temperature_coefficient: Linear temperature coefficient
            of `base_collector_grading` (K⁻¹).
        base_collector_grading_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of `base_collector_grading` (K⁻²).
        substrate_grading_temperature_coefficient: Linear temperature coefficient of
            `substrate_grading` (K⁻¹).
        substrate_grading_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of `substrate_grading` (K⁻²).
        base_emitter_capacitance_temperature_coefficient: Linear temperature coefficient
            of `base_emitter_capacitance` (K⁻¹).
        base_collector_capacitance_temperature_coefficient: Linear temperature
            coefficient of `base_collector_capacitance` (K⁻¹).
        substrate_capacitance_temperature_coefficient: Linear temperature coefficient of
            `substrate_capacitance` (K⁻¹).
        base_emitter_potential_temperature_coefficient: Linear temperature coefficient
            of `base_emitter_potential` (V/K).
        base_collector_potential_temperature_coefficient: Linear temperature coefficient
            of `base_collector_potential` (V/K).
        substrate_potential_temperature_coefficient: Linear temperature coefficient of
            `substrate_potential` (V/K).
        nominal_temperature: Model extraction temperature in kelvins (``TNOM``).
        flicker_coefficient: Flicker noise scaling coefficient (``KF``).
        flicker_exponent: Flicker noise base-current exponent (``AF``).

    Returns:
        Gummel-Poon BJT `Device` with optional fourth ``substrate`` terminal.

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
    return _device("bipolar_transistor", locals())


def vbic(  # noqa: PLR0913 - VBIC 1.2 model parameters
    *,
    polarity: Literal["npn", "pnp"] | None = None,
    extrinsic_collector_resistance: Quantity | None = None,
    intrinsic_collector_resistance: Quantity | None = None,
    epitaxial_saturation_voltage: Quantity | None = None,
    epitaxial_doping_parameter: float | None = None,
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
    epitaxial_charge: Quantity | None = None,
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
    epitaxial_saturation_voltage_temperature_exponent: float | None = None,
    breakdown_voltage_temperature_coefficient: Quantity | None = None,
    breakdown_voltage_quadratic_temperature_coefficient: Quantity | None = None,
    breakdown_emission_temperature_coefficient: Quantity | None = None,
) -> Device:
    """Creates a four-terminal VBIC 1.2 bipolar transistor matching ngspice 47.

    The model implements the vertical bipolar inter-company formulation (McAndrew et
    al., IEEE JSSC 1996, extended to VBIC 1.2) across ``collector``, ``base``,
    ``emitter``, and ``substrate``. In addition to Gummel-Poon transport, Early effect,
    high injection, and junction charges, VBIC models the parasitic substrate
    transistor, base-collector weak avalanche, and Kull quasi-saturation. VBIC also
    includes base-emitter breakdown, base-collector reach-through, and smoothed
    junction capacitances. Omitted parameters use ngspice defaults, with infinite Early
    voltages, knee currents, epitaxial saturation voltage, and transit-time voltage.

    Extrinsic collector, extrinsic base, emitter, and substrate resistances add
    internal unknowns ``collector``, ``base``, ``emitter``, and ``substrate``. The
    intrinsic collector, intrinsic base, and parasitic base nodes are always present as
    internal unknowns. Excess-phase delay adds ``transport_filter`` and
    ``delayed_transport`` for the transport-delay network.

    Away from `nominal_temperature`, saturation currents scale by activation energies
    and temperature exponents, while resistances, forward knee current, and epitaxial
    saturation voltage scale by their exponents. Junction potentials and capacitances
    scale with intrinsic carrier concentration, and emission, avalanche, and breakdown
    parameters follow their coefficients using VBIC thermal-voltage constants. A heated
    VBIC runs at its thermal net temperature, representing VBIC self-heating through
    external thermal elements.

    Extrinsic and intrinsic resistances generate thermal noise (intrinsic resistances
    at their local slopes). Transport, intrinsic base-emitter, and parasitic currents
    generate shot noise. With flicker coefficient ``KFN``, base-emitter currents ``Ib``
    add flicker noise of ``KFN · |Ib|^AFN / f^BFN`` A²/Hz.

    Args:
        polarity: Transistor doping polarity (`'npn'` or `'pnp'`).
        extrinsic_collector_resistance: Extrinsic collector series resistance in ohms
            (``RCX``).
        intrinsic_collector_resistance: Intrinsic epitaxial collector resistance in ohms
            (``RCI``).
        epitaxial_saturation_voltage: Epitaxial carrier velocity-saturation voltage
            (``VO``).
        epitaxial_doping_parameter: Epitaxial doping parameter for Kull quasi-saturation
            (``GAMM``).
        high_current_collector_factor: High-current intrinsic collector
            current-spreading factor (``HRCF``).
        extrinsic_base_resistance: Extrinsic base resistance in ohms (``RBX``).
        intrinsic_base_resistance: Zero-bias intrinsic modulated base resistance in ohms
            (``RBI``).
        emitter_resistance: Extrinsic emitter resistance in ohms (``RE``).
        substrate_resistance: Substrate contact resistance in ohms (``RS``).
        parasitic_base_resistance: Parasitic PNP/NPN base resistance in ohms (``RBP``).
        saturation_current: Main transport saturation current in amperes (``IS``).
        forward_emission_coefficient: Forward transport ideality factor (``NF``).
        reverse_emission_coefficient: Reverse transport ideality factor (``NR``).
        depletion_coefficient: Forward-bias depletion capacitance offset fraction
            (``FC``).
        base_emitter_overlap_capacitance: Fixed base-emitter overlap capacitance in
            farads (``CBEO``).
        base_emitter_capacitance: Zero-bias base-emitter depletion capacitance
            (``CJE``).
        base_emitter_potential: Base-emitter built-in potential in volts (``PE``).
        base_emitter_grading: Base-emitter depletion grading exponent (``ME``).
        base_emitter_smoothing: Smooth-capacitance transition parameter for B-E junction
            (``AJE``).
        base_collector_overlap_capacitance: Fixed base-collector overlap capacitance in
            farads (``CBCO``).
        base_collector_capacitance: Zero-bias intrinsic base-collector capacitance
            (``CJC``).
        epitaxial_charge: Quasi-saturation epitaxial charge parameter (``QCO``).
        extrinsic_base_collector_capacitance: Zero-bias extrinsic base-collector
            capacitance (``CJEP``).
        base_collector_potential: Base-collector built-in potential in volts (``PC``).
        base_collector_grading: Base-collector depletion grading exponent (``MC``).
        base_collector_smoothing: Smooth-capacitance transition parameter for B-C
            junction (``AJC``).
        collector_substrate_capacitance: Zero-bias collector-substrate depletion
            capacitance (``CJCP``).
        collector_substrate_potential: Collector-substrate built-in potential in volts
            (``PS``).
        collector_substrate_grading: Collector-substrate depletion grading exponent
            (``MS``).
        collector_substrate_smoothing: Smooth-capacitance transition parameter for C-S
            junction (``AJS``).
        collector_substrate_fixed_capacitance: Fixed collector-substrate parasitic
            capacitance (``CCSO``).
        ideal_base_emitter_current: Ideal base-emitter diode saturation current
            (``IBEI``).
        intrinsic_base_emitter_fraction: Sidewall/intrinsic partition fraction for B-E
            current (``WBE``).
        ideal_base_emitter_emission_coefficient: Ideal base-emitter diode ideality
            factor (``NEI``).
        nonideal_base_emitter_current: Nonideal base-emitter recombination current
            (``IBEN``).
        nonideal_base_emitter_emission_coefficient: Nonideal base-emitter recombination
            ideality factor (``NEN``).
        ideal_base_collector_current: Ideal intrinsic base-collector saturation current
            (``IBCI``).
        ideal_base_collector_emission_coefficient: Ideal base-collector diode ideality
            factor (``NCI``).
        nonideal_base_collector_current: Nonideal base-collector recombination current
            (``IBCN``).
        nonideal_base_collector_emission_coefficient: Nonideal base-collector
            recombination ideality factor (``NCN``).
        avalanche_coefficient: Base-collector weak-avalanche multiplier (``AVC1``).
        avalanche_exponent_coefficient: Base-collector weak-avalanche exponential factor
            (``AVC2``).
        parasitic_saturation_current: Parasitic substrate-transistor transport
            saturation current (``ISP``).
        parasitic_extrinsic_fraction: Extrinsic collector partition of parasitic
            transport (``WSP``).
        parasitic_forward_emission_coefficient: Parasitic transport ideality factor
            (``NFP``).
        parasitic_ideal_base_emitter_current: Parasitic extrinsic B-C ideal diode
            current (``IBEIP``).
        parasitic_nonideal_base_emitter_current: Parasitic extrinsic B-C nonideal diode
            current (``IBENP``).
        parasitic_ideal_base_collector_current: Parasitic B-S ideal diode saturation
            current (``IBCIP``).
        parasitic_ideal_base_collector_emission_coefficient: Parasitic B-S ideal diode
            ideality factor (``NCIP``).
        parasitic_nonideal_base_collector_current: Parasitic B-S nonideal diode
            saturation current (``IBCNP``).
        parasitic_nonideal_base_collector_emission_coefficient: Parasitic B-S nonideal
            diode ideality factor (``NCNP``).
        forward_early_voltage: Forward Early voltage in volts (``VEF``).
        reverse_early_voltage: Reverse Early voltage in volts (``VER``).
        forward_knee_current: Forward high-injection knee current (``IKF``).
        reverse_knee_current: Reverse high-injection knee current (``IKR``).
        parasitic_knee_current: Parasitic transistor high-injection knee current
            (``IKP``).
        knee_rolloff: High-injection base-charge exponent (``NKF``).
        base_charge: Base-charge formulation (`'vbic'` or `'gummel_poon'`, ``QBM``).
        forward_transit_time: Nominal forward transit time in seconds (``TF``).
        transit_time_base_width_coefficient: Base-width modulation weight on
            `forward_transit_time` (``QTF``).
        transit_time_bias_coefficient: Bias-dependent multiplier on
            `forward_transit_time` (``XTF``).
        transit_time_voltage: B-C voltage scaling transit-time bias (``VTF``).
        transit_time_current: High-current threshold for transit time (``ITF``).
        reverse_transit_time: Reverse transit time in seconds (``TR``).
        excess_phase_delay: Excess-phase transport delay in seconds (``TD``).
        flicker_coefficient: Base-emitter flicker noise coefficient (``KFN``).
        flicker_exponent: Flicker noise current exponent (``AFN``).
        flicker_frequency_exponent: Flicker noise frequency exponent (``BFN``).
        reach_through_voltage: B-C punch-through limiting voltage (``VRT``).
        reach_through_smoothing: Reach-through capacitance smoothing (``ART``).
        base_emitter_breakdown_voltage: Base-emitter reverse breakdown voltage in volts
            (``VBBE``).
        base_emitter_breakdown_emission_coefficient: Base-emitter breakdown ideality
            factor (``NBBE``).
        base_emitter_breakdown_current: Reverse current at
            `base_emitter_breakdown_voltage` (``IBBE``).
        reverse_saturation_current_ratio: Reverse-to-forward transport saturation ratio
            (``ISRR``).
        nominal_temperature: Reference extraction temperature in kelvins (``TNOM``).
        activation_energy: Main transport activation energy (``EA``).
        ideal_base_emitter_activation_energy: Ideal B-E diode activation energy
            (``EAIE``).
        ideal_base_collector_activation_energy: Ideal B-C diode activation energy
            (``EAIC``).
        ideal_collector_substrate_activation_energy: Ideal C-S diode activation energy
            (``EAIS``).
        nonideal_base_emitter_activation_energy: Nonideal B-E recombination activation
            energy (``EANE``).
        nonideal_base_collector_activation_energy: Nonideal B-C recombination activation
            energy (``EANC``).
        nonideal_collector_substrate_activation_energy: Nonideal C-S recombination
            activation energy (``EANS``).
        parasitic_activation_energy: Parasitic transport activation energy (``EAP``).
        reverse_activation_energy_shift: Activation-energy offset for `ISRR` scaling
            (``DEAR``).
        saturation_current_temperature_exponent: Transport saturation temperature
            exponent (``XIS``).
        ideal_base_current_temperature_exponent: Ideal base diode temperature exponent
            (``XII``).
        nonideal_base_current_temperature_exponent: Nonideal recombination diode
            temperature exponent (``XIN``).
        reverse_saturation_current_temperature_exponent: Temperature exponent of
            `reverse_saturation_current_ratio` (``XISR``).
        emission_coefficient_temperature_coefficient: Linear temperature coefficient of
            `NF` and `NR` (``TNF``).
        avalanche_temperature_coefficient: Linear temperature coefficient of `AVC2`
            (``TAVC``).
        forward_knee_current_temperature_exponent: Temperature exponent of
            `forward_knee_current` (``XIKF``).
        extrinsic_collector_resistance_temperature_exponent: Temperature exponent of
            `extrinsic_collector_resistance` (``XRCX``).
        intrinsic_collector_resistance_temperature_exponent: Temperature exponent of
            `intrinsic_collector_resistance` (``XRCI``).
        extrinsic_base_resistance_temperature_exponent: Temperature exponent of
            `extrinsic_base_resistance` (``XRBX``).
        intrinsic_base_resistance_temperature_exponent: Temperature exponent of
            `intrinsic_base_resistance` (``XRBI``).
        emitter_resistance_temperature_exponent: Temperature exponent of
            `emitter_resistance` (``XRE``).
        substrate_resistance_temperature_exponent: Temperature exponent of
            `substrate_resistance` (``XRS``).
        parasitic_base_resistance_temperature_exponent: Temperature exponent of
            `parasitic_base_resistance` (``XRBP``).
        epitaxial_saturation_voltage_temperature_exponent: Temperature exponent of
            `epitaxial_saturation_voltage` (``XVO``).
        breakdown_voltage_temperature_coefficient: Linear temperature coefficient of
            `VBBE` (``TVBBE1``).
        breakdown_voltage_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of `VBBE` (``TVBBE2``).
        breakdown_emission_temperature_coefficient: Linear temperature coefficient of
            `NBBE` (``TNBBE``).

    Returns:
        Four-terminal VBIC 1.2 bipolar transistor `Device`.

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
    return _device("vbic", locals())


def mosfet(  # noqa: PLR0913 - SPICE level 1 and 3 MOSFET parameters
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
    """Creates a SPICE level 1 or level 3 MOSFET with four terminals.

    Terminals are ``drain``, ``gate``, ``source``, and ``bulk``. At ``level=1``
    (default), the transistor follows the Shichman-Hodges square law with Meyer gate
    capacitances, matching ngspice. The effective threshold voltage is
    ``von = vto + gamma · (√(phi - vbs) - √phi)``. Above threshold, channel prefactor
    ``kp · w / (l - 2 · ld) · (1 + lambda · vds)`` drives the drain current. Drain
    current multiplies that prefactor by ``(vgs - von)² / 2`` in saturation and by
    ``vds · (vgs - von - vds / 2)`` in triode. Bulk-drain and bulk-source junctions
    connect the body to the diffusions.

    Omitted parameters use SPICE defaults, with `body_effect` specified in square-root
    volts. Supplying `oxide_thickness` enables Meyer gate capacitances; combining
    `oxide_thickness` with `substrate_doping` derives omitted threshold voltage, body
    effect, and surface potential. Away from `nominal_temperature`, transconductance
    and mobility scale as ``T^-1.5``, while surface potential, threshold voltage, and
    junction parameters follow the silicon band gap.

    At ``level=3``, channel conduction uses the SPICE semi-empirical short-channel
    model. Level-3-only parameters from `max_drift_velocity` onward model velocity
    saturation, short- and narrow-channel threshold shifts, drain-induced barrier
    lowering, mobility degradation, channel-length shortening, weak inversion, and mask
    adjustments. Level-3 MOSFETs disable channel-length modulation and default to
    0.1 µm oxide thickness and 0.33 sidewall grading.

    Nonzero drain or source resistance adds internal node unknown ``drain`` or
    ``source``. Meyer gate capacitances add three displacement-current unknowns:
    ``gate_source_displacement``, ``gate_drain_displacement``, and
    ``gate_bulk_displacement``.

    Series resistances produce thermal noise, and the channel generates thermal noise
    of ``8kT · gm / 3`` A²/Hz. With flicker coefficient ``KF`` and exponent ``AF``,
    drain current ``Id`` adds flicker noise of ``KF · |Id|^AF / (f · W · Leff · Cox²)``
    A²/Hz. Gate area sums parallel multiplier widths and assumes a 0.1 µm oxide
    thickness when omitted.

    Args:
        level: SPICE MOSFET equation level (`1` Shichman-Hodges or `3` short-channel).
        channel: Channel carrier polarity (`'n'` or `'p'`).
        width: Drawn channel width in meters (``W``).
        length: Drawn channel length in meters (``L``).
        multiplier: Parallel finger or device count multiplier (``M``).
        threshold_voltage: Zero-bias threshold voltage in volts (``VTO``).
        transconductance_parameter: Intrinsic transconductance factor in A/V² (``KP``).
        body_effect: Bulk threshold parameter in V^{1/2} (``GAMMA``).
        surface_potential: Strong-inversion surface potential in volts (``PHI``).
        channel_length_modulation: Saturation output slope in V⁻¹ (``LAMBDA``).
        lateral_diffusion: Source/drain lateral under-diffusion in meters (``LD``).
        oxide_thickness: Gate oxide thickness in meters (``TOX``).
        surface_mobility: Low-field carrier surface mobility (``UO``).
        substrate_doping: Channel substrate doping concentration (``NSUB``).
        surface_state_density: Fixed oxide interface charge density (``NSS``).
        gate_material: Gate electrode work-function type (``TPG``: opposite, same, or
            aluminum).
        drain_resistance: Extrinsic drain series resistance in ohms (``RD``).
        source_resistance: Extrinsic source series resistance in ohms (``RS``).
        sheet_resistance: Diffusion sheet resistance in ohms per square (``RSH``).
        drain_squares: Number of sheet squares in the drain diffusion (``NRD``).
        source_squares: Number of sheet squares in the source diffusion (``NRS``).
        saturation_current: Bulk-junction lumped saturation current (``IS``).
        saturation_current_density: Bulk-junction areal saturation current density
            (``JS``).
        drain_area: Drain diffusion area in square meters (``AD``).
        source_area: Source diffusion area in square meters (``AS``).
        drain_perimeter: Drain diffusion perimeter in meters (``PD``).
        source_perimeter: Source diffusion perimeter in meters (``PS``).
        junction_potential: Bulk-junction built-in potential in volts (``PB``).
        bulk_drain_capacitance: Fixed zero-bias bulk-drain capacitance (``CBD``).
        bulk_source_capacitance: Fixed zero-bias bulk-source capacitance (``CBS``).
        bottom_capacitance: Zero-bias bottom areal junction capacitance (``CJ``).
        sidewall_capacitance: Zero-bias sidewall perimetric junction capacitance
            (``CJSW``).
        bottom_grading: Bottom-junction grading exponent (``MJ``).
        sidewall_grading: Sidewall-junction grading exponent (``MJSW``).
        depletion_coefficient: Forward-bias junction capacitance linearization fraction
            (``FC``).
        gate_source_overlap: Gate-source overlap capacitance per meter (``CGSO``).
        gate_drain_overlap: Gate-drain overlap capacitance per meter (``CGDO``).
        gate_bulk_overlap: Gate-bulk overlap capacitance per meter (``CGBO``).
        nominal_temperature: Model extraction temperature in kelvins (``TNOM``).
        flicker_coefficient: Flicker noise scaling coefficient (``KF``).
        flicker_exponent: Flicker noise drain-current exponent (``AF``).
        max_drift_velocity: Level-3 carrier saturation velocity in m/s (``VMAX``).
        junction_depth: Level-3 metallurgical junction depth in meters (``XJ``).
        fast_surface_state_density: Level-3 subthreshold fast surface-state density
            (``NFS``).
        static_feedback: Level-3 DIBL static feedback factor (``ETA``).
        narrow_width_factor: Level-3 narrow-channel threshold factor (``DELTA``).
        mobility_modulation: Level-3 gate-field mobility degradation in V⁻¹ (``THETA``).
        saturation_field_factor: Level-3 channel-length shortening factor (``KAPPA``).
        width_narrowing: Level-3 lateral width reduction per side (``WD``).
        length_adjustment: Level-3 drawn-to-mask length offset (``XL``).
        width_adjustment: Level-3 drawn-to-mask width offset (``XW``).
        threshold_adjustment: Per-instance threshold voltage shift (``DELVTO``).

    Returns:
        Four-terminal SPICE level 1 or level 3 MOSFET `Device`.

    Examples:
        >>> from jolty.units import nm, um
        >>> mosfet(width=2 * um, length=1 * um, oxide_thickness=20 * nm).parameters
        {'width': 2e-06, 'length': 1e-06, 'oxide_thickness': 2e-08}
    """
    return _device("mosfet", locals())


def vdmos(  # noqa: PLR0913 - ngspice VDMOS model parameters
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
    quasi_saturation_resistance_temperature_exponent: float | None = None,
    source_resistance_temperature_coefficient: Quantity | None = None,
    source_resistance_quadratic_temperature_coefficient: Quantity | None = None,
    gate_resistance_temperature_coefficient: Quantity | None = None,
    gate_resistance_quadratic_temperature_coefficient: Quantity | None = None,
    body_resistance_temperature_coefficient: Quantity | None = None,
    body_resistance_quadratic_temperature_coefficient: Quantity | None = None,
    flicker_coefficient: float | None = None,
    flicker_exponent: float | None = None,
) -> Device:
    """Creates a VDMOS power MOSFET across ``drain``, ``gate``, and ``source``.

    The model implements ngspice VDMOS equations compatible with LTspice power MOSFET
    cards. Conduction follows the Shichman-Hodges square law without body effect,
    softened below threshold by `subthreshold_slope` and degraded by gate-field
    mobility modulation. Supplying both quasi-saturation parameters adds a
    drain-voltage-dependent drain resistance.

    Gate capacitances include a fixed gate-source capacitance and a voltage-dependent
    gate-drain capacitance transitioning smoothly between its minimum and maximum
    values. An integrated body diode with dedicated junction parameters and shared
    breakdown connects ``source`` to ``drain``. Omitted parameters use ngspice defaults
    based on IRF540 and IRF9540 devices.

    Away from `nominal_temperature`, transconductance scales by its temperature
    exponent, while threshold voltage, subthreshold slope, and resistances follow their
    temperature coefficients. The body diode follows standard diode temperature scaling
    matching ngspice.

    Nonzero drain, source, gate, or body resistance adds internal unknown ``drain``,
    ``source``, ``gate``, or ``body_diode``. Drain and source resistances generate
    thermal noise, and the channel generates thermal noise of ``8kT · gm / 3`` A²/Hz.
    With flicker coefficient ``KF`` and exponent ``AF``, drain current ``Id`` adds
    flicker noise of ``KF · |Id|^AF / (f · Cox²)`` A²/Hz using a 0.1 µm reference
    oxide capacitance per unit area.

    Args:
        channel: Channel carrier polarity (`'n'` or `'p'`).
        multiplier: Parallel cell or die multiplier (``M``).
        threshold_voltage: Gate-source turn-on threshold voltage (``VTO``).
        transconductance_parameter: Channel transconductance factor (``KP``).
        channel_length_modulation: Saturation output slope in V⁻¹ (``LAMBDA``).
        mobility_modulation: Gate-field mobility degradation in V⁻¹ (``THETA``).
        triode_conductance_multiplier: Linear-region conductance compression multiplier
            (``MTRIODE``).
        subthreshold_slope: Subthreshold conduction slope voltage (``KSUBVT``).
        subthreshold_shift: Subthreshold threshold voltage offset (``VSUBSC``).
        drain_resistance: Extrinsic drift/drain resistance in ohms (``RD``).
        source_resistance: Extrinsic source lead resistance in ohms (``RS``).
        gate_resistance: Internal polysilicon gate resistance in ohms (``RG``).
        quasi_saturation_resistance: Quasi-saturation JFET-region resistance in ohms
            (``RQ``).
        quasi_saturation_voltage: Quasi-saturation onset voltage (``VQ``).
        drain_source_resistance: Parallel drain-source leakage resistance (``RDS``).
        gate_source_capacitance: Fixed gate-source capacitance in farads (``CGS``).
        max_gate_drain_capacitance: Accumulation-region maximum Miller capacitance
            (``CGDMAX``).
        min_gate_drain_capacitance: High-voltage minimum Miller capacitance
            (``CGDMIN``).
        gate_drain_capacitance_coefficient: Miller capacitance voltage transition slope
            in V⁻¹ (``A``).
        saturation_current: Body-diode reverse saturation current (``IS``).
        emission_coefficient: Body-diode forward ideality factor (``N``).
        body_resistance: Body-diode series resistance in ohms (``RB``).
        junction_capacitance: Zero-bias body-diode capacitance in farads (``CJO``).
        junction_potential: Body-diode built-in potential in volts (``VJ``).
        grading_coefficient: Body-diode depletion grading exponent (``M``).
        depletion_coefficient: Body-diode forward capacitance linearization fraction
            (``FC``).
        transit_time: Body-diode reverse-recovery transit time (``TT``).
        breakdown_voltage: Drain-source avalanche breakdown voltage (``BV``).
        breakdown_current: Reverse current at `breakdown_voltage` (``IBV``).
        breakdown_emission_coefficient: Avalanche breakdown ideality factor (``NBV``).
        energy_gap: Body-diode band-gap energy (``EG``).
        saturation_current_temperature_exponent: Body-diode saturation current
            temperature exponent (``XTI``).
        nominal_temperature: Model extraction temperature in kelvins (``TNOM``).
        threshold_voltage_temperature_coefficient: Linear temperature shift of
            `threshold_voltage` in V/K (``VTC``).
        transconductance_parameter_temperature_exponent: Temperature exponent of
            `transconductance_parameter` (``BEX``).
        subthreshold_slope_temperature_coefficient: Linear temperature coefficient of
            `subthreshold_slope` (``TKVT1``).
        subthreshold_slope_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of `subthreshold_slope` (``TKVT2``).
        drain_resistance_temperature_exponent: Power-law temperature exponent of
            `drain_resistance` (``TRD``).
        drain_resistance_temperature_coefficient: Linear temperature coefficient of
            `drain_resistance` (``TRD1``).
        drain_resistance_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of `drain_resistance` (``TRD2``).
        quasi_saturation_resistance_temperature_exponent: Temperature exponent of
            `quasi_saturation_resistance` (``TRQ``).
        source_resistance_temperature_coefficient: Linear temperature coefficient of
            `source_resistance` (``TRS1``).
        source_resistance_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of `source_resistance` (``TRS2``).
        gate_resistance_temperature_coefficient: Linear temperature coefficient of
            `gate_resistance` (``TRG1``).
        gate_resistance_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of `gate_resistance` (``TRG2``).
        body_resistance_temperature_coefficient: Linear temperature coefficient of
            `body_resistance` (``TRB1``).
        body_resistance_quadratic_temperature_coefficient: Quadratic temperature
            coefficient of `body_resistance` (``TRB2``).
        flicker_coefficient: Flicker noise scaling coefficient (``KF``).
        flicker_exponent: Flicker noise drain-current exponent (``AF``).

    Returns:
        Three-terminal VDMOS power MOSFET `Device`.

    Examples:
        >>> from jolty.units import V, ohm
        >>> vdmos(threshold_voltage=4 * V, drain_resistance=0.044 * ohm).parameters
        {'threshold_voltage': 4.0, 'drain_resistance': 0.044}
    """
    return _device("vdmos", locals())


def bsim3(  # noqa: PLR0913 - instance geometry and BSIM3 model selectors
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
    """Creates a BSIM3v3.3 MOSFET across ``drain``, ``gate``, ``source``, and ``bulk``.

    The model follows BSIM3v3.3 short-channel equations (Liu, Xi et al., UC Berkeley,
    2005; SPICE levels 8 and 49) as implemented in ngspice 47. Keyword arguments before
    `card` configure drawn instance geometry, diffusion dimensions, threshold shift,
    mobility multiplier, model selectors, parameter clamping, and charge partitioning.
    Each `card` entry maps a BSIM3v3.3 manual name (such as ``vth0``, ``tox``, or
    ``lvth0``) to an SI [`Quantity`][jolty.units.Quantity] matching
    [`bsim3_dimension`][jolty.devices.bsim3_dimension] or a plain float. Omitted
    parameters use ngspice defaults.

    Drain current incorporates short-channel, narrow-width, and DIBL threshold terms,
    poly-gate depletion, field mobility degradation, velocity saturation, output
    conductance effects, and normalized source-drain resistance ``rdsw``. Impact
    ionization current flows into the bulk. Gate charges follow `capacitance_model`
    alongside overlap capacitances and bulk-drain and bulk-source diodes.

    Binned card terms prefixed with ``l``, ``w``, and ``p`` scale with reciprocal
    effective length, width, and length-width product in meters rather than
    micrometers. All parameters scale with temperature matching ngspice. Nonzero sheet
    resistance ``rsh`` adds internal node unknowns ``drain`` and ``source``.

    Sheet resistances generate thermal noise, while the channel produces thermal and
    flicker noise selected by `noise_model` (SPICE2 or BSIM3 formulations).

    Args:
        channel: Channel carrier polarity (`'n'` or `'p'`).
        width: Drawn channel width in meters (``W``).
        length: Drawn channel length in meters (``L``).
        multiplier: Parallel instance multiplier (``M``).
        drain_area: Drain diffusion area in square meters (``AD``).
        source_area: Source diffusion area in square meters (``AS``).
        drain_perimeter: Drain diffusion perimeter in meters (``PD``).
        source_perimeter: Source diffusion perimeter in meters (``PS``).
        drain_squares: Sheet-resistance squares in drain diffusion (``NRD``).
        source_squares: Sheet-resistance squares in source diffusion (``NRS``).
        threshold_adjustment: Instance threshold voltage offset (``DELVTO``).
        mobility_multiplier: Instance low-field mobility scale factor (``MULU0``).
        mobility_model: BSIM3 mobility degradation formulation (``MOBMOD`` 1..3).
        capacitance_model: Intrinsic gate-charge formulation (``CAPMOD`` 0..3).
        noise_model: Channel thermal/flicker noise selector (``NOIMOD`` 1..6).
        check_parameters: When ``True``, clamps out-of-range model parameters.
        charge_partition: Channel-charge drain/source split selector (``XPART``).
        nominal_temperature: Model extraction temperature in kelvins (``TNOM``).
        card: Mapping of BSIM3v3.3 manual parameter names to SI quantities or floats.

    Returns:
        Four-terminal BSIM3v3.3 MOSFET `Device`.

    Raises:
        DimensionError: If a parameter has the wrong physical dimension.
        ValueError: If `card` names an unknown BSIM3 parameter or a non-finite value.

    Examples:
        >>> from jolty.units import V, nm, um
        >>> transistor = bsim3(
        ...     width=2 * um,
        ...     length=0.35 * um,
        ...     card={"tox": 7.6 * nm, "vth0": 0.5 * V, "k2": -0.02},
        ... )
        >>> transistor.parameters["tox"], transistor.parameters["k2"]
        (7.6e-09, -0.02)
    """
    values = locals()
    values.pop("card")
    return _device(
        "bsim3",
        values,
        card=card,
        card_dimension=lambda name: bsim3_dimension(name, mobility_model),
    )


def _bsim3_card() -> dict[str, Dimension | Literal["uc"] | None]:
    """Derives BSIM3 card parameter names and SI dimensions from the engine."""
    excluded = inspect.signature(bsim3).parameters
    card: dict[str, Dimension | Literal["uc"] | None] = {}
    for name, kind in _model_kinds("bsim3").items():
        if name in excluded:
            continue
        match kind:
            case Dimension() | None:
                card[name] = kind
            case "uc":
                card[name] = "uc"
            case _:
                message = f"unexpected BSIM3 card kind for `{name}`: {kind!r}"
                raise ValueError(message)
    return card


BSIM3_CARD: Final[Mapping[str, Dimension | Literal["uc"] | None]] = _bsim3_card()
"""Mapping from BSIM3v3.3 card parameter names to expected SI dimensions.

Plain numeric parameters map to ``None`` (such as ``ll``, whose unit depends on
``lln``). Parameters ``uc`` and ``uc1`` map to ``"uc"`` because their dimension depends
on ``mobility_model``."""

BSIM3_BINNED: Final = frozenset(_core.binned_parameters("bsim3"))
"""BSIM3 parameters supporting ``l``, ``w``, and ``p`` binning prefixes, scaled by
meters or square meters."""


def bsim3_dimension(name: str, mobility_model: int | None = None) -> Dimension | None:
    """Returns the expected SI dimension of BSIM3 parameter `name`, or ``None``.

    Binned prefixes multiply the base parameter dimension by meters (``l``, ``w``) or
    square meters (``p``). Parameters ``uc`` and ``uc1`` use meters per square volt by
    default, or inverse volts when `mobility_model` is 3.

    Args:
        name: BSIM3v3.3 direct or ``l``/``w``/``p``-binned card parameter name.
        mobility_model: Optional ``MOBMOD`` value resolving ``uc`` and ``uc1``.

    Returns:
        Expected SI `Dimension`, or ``None`` for plain dimensionless numbers.

    Raises:
        ValueError: If BSIM3 defines no parameter named `name`.

    Examples:
        >>> bsim3_dimension("lvth0") == VOLTAGE * LENGTH, bsim3_dimension("k2")
        (True, None)
    """
    dimension, extent = _bsim_parameter("BSIM3", name, BSIM3_CARD, BSIM3_BINNED)
    if dimension == "uc":
        dimension = PER_VOLT if mobility_model == 3 else LENGTH / VOLTAGE**2  # noqa: PLR2004 - mobmod 3's body effect
    return _extended(dimension, extent)


def _bsim_parameter[T](
    model: str, name: str, card: Mapping[str, T], binned: frozenset[str]
) -> tuple[T, Dimension]:
    """Resolves the `card` entry and binning extent for parameter `name` in `model`.

    Direct parameters return their matching `card` entry and dimensionless extent.
    Binned ``l``, ``w``, or ``p`` prefixed parameters return the base parameter entry
    with length or area extent.

    Raises:
        ValueError: If `model` defines no parameter named `name`.
    """
    if name in card:
        return card[name], DIMENSIONLESS
    prefix, base = name[:1], name[1:]
    if prefix not in {"l", "w", "p"} or base not in binned:
        message = f"{model} has no parameter `{name}`"
        raise ValueError(message)
    return card[base], LENGTH**2 if prefix == "p" else LENGTH


def _extended(dimension: Dimension | None, extent: Dimension) -> Dimension | None:
    """Multiplies `dimension` by `extent`, returning ``None`` for plain floats."""
    if dimension is None:
        return None if extent == DIMENSIONLESS else extent
    return dimension * extent


def _card(
    parameters: dict[str, JsonValue],
    card: Mapping[str, Quantity | float],
    dimension_of: Callable[[str], Dimension | None],
) -> None:
    """Validates and merges BSIM `card` entries into `parameters` in SI base units.

    Raises:
        DimensionError: If a parameter has the wrong physical dimension.
        ValueError: If `card` names an unknown parameter or contains a non-finite
            value.
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


def bsim4(  # noqa: PLR0913 - instance geometry and BSIM4 model selectors
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
    gate_resistance_model: Literal[0, 1, 2, 3] | None = None,
    perimeter_model: Literal[0, 1] | None = None,
    drain_source_resistance_model: Literal[0, 1] | None = None,
    flicker_noise_model: Literal[0, 1] | None = None,
    thermal_noise_model: Literal[0, 1] | None = None,
    check_parameters: bool | None = None,
    charge_partition: float | None = None,
    nominal_temperature: Quantity | None = None,
    card: Mapping[str, Quantity | float] | None = None,
) -> Device:
    """Creates a BSIM4.5 MOSFET across ``drain``, ``gate``, ``source``, and ``bulk``.

    The model follows BSIM4.5.0 short-channel equations (Xi, Dunga et al., UC Berkeley,
    2005; SPICE levels 14 and 54) matching ngspice 47 for SKY130 and GF180MCU modes.
    Keyword arguments before `card` configure instance geometry, finger count,
    diffusions, length-of-diffusion stress distances, threshold and mobility shifts,
    resistance and noise model selectors, clamping, and charge partitioning. Each
    `card` entry maps a BSIM4.5 manual name to an SI [`Quantity`][jolty.units.Quantity]
    matching [`bsim4_dimension`][jolty.devices.bsim4_dimension] or a plain float.
    Omitted parameters use ngspice defaults.

    Drain current models short-channel, narrow-width, pocket-implant, and DIBL
    threshold terms, poly-gate depletion, mobility degradation, velocity saturation and
    overshoot, source velocity limit, output conductance, and ``rdsw``. Impact
    ionization and gate-induced leakage currents flow into the body, while two diodes
    with trap-assisted tunneling connect the body to drain and source. Gate charges
    use ``capmod`` 2 alongside overlap and junction capacitances.

    Binned ``l``, ``w``, and ``p`` parameters scale with reciprocal effective length,
    width, and area in meters, with full ngspice temperature scaling. Positive
    `drain_squares` or `source_squares` with sheet resistance ``rsh`` adds internal
    drain or source nodes, and ``body_resistance_model=1`` adds three substrate nodes.
    Settings ``gate_resistance_model`` 1 through 3 insert gate-electrode resistance
    ``rshg``, adding intrinsic input resistance (``xrcrg1``, ``xrcrg2``) in series at
    mode 2 or after a mid-gate overlap node at mode 3.

    Resistances produce thermal noise. Channel thermal noise follows
    `thermal_noise_model` (``tnoimod``, charge-based or holistic), while flicker noise
    follows `flicker_noise_model` (``fnoimod``, SPICE2 or BSIM4).

    Args:
        channel: Channel carrier polarity (`'n'` or `'p'`).
        width: Total drawn channel width across fingers in meters (``W``).
        length: Drawn channel length in meters (``L``).
        multiplier: Parallel instance multiplier (``M``).
        fingers: Number of gate fingers dividing `width` (``NF``).
        drain_area: Explicit drain diffusion area in square meters (``AD``).
        source_area: Explicit source diffusion area in square meters (``AS``).
        drain_perimeter: Explicit drain diffusion perimeter in meters (``PD``).
        source_perimeter: Explicit source diffusion perimeter in meters (``PS``).
        drain_squares: Sheet-resistance squares in drain diffusion (``NRD``).
        source_squares: Sheet-resistance squares in source diffusion (``NRS``).
        diffusion_edge_a: LOD stress distance from gate to left STI edge (``SA``).
        diffusion_edge_b: LOD stress distance from gate to right STI edge (``SB``).
        finger_spacing: Gate-to-gate spacing between adjacent fingers (``SD``).
        threshold_adjustment: Instance threshold voltage offset (``DELVTO``).
        mobility_multiplier: Instance low-field mobility scale factor (``MULU0``).
        body_resistance_model: Substrate resistance network selector (``RBODYMOD``).
        gate_resistance_model: Gate electrode resistance selector (``RGATEMOD``).
        perimeter_model: Layout source/drain geometry selector (``GEOMOD``/``PERMOD``).
        drain_source_resistance_model: Lumped vs extrinsic S/D resistance selector
            (``RDSMOD``).
        flicker_noise_model: Flicker noise formulation selector (``FNOIMOD``).
        thermal_noise_model: Channel thermal noise selector (``TNOIMOD``).
        check_parameters: When ``True``, clamps out-of-range model parameters.
        charge_partition: Channel-charge drain/source split selector (``XPART``).
        nominal_temperature: Model extraction temperature in kelvins (``TNOM``).
        card: Mapping of BSIM4.5 manual parameter names to SI quantities or floats.

    Returns:
        Four-terminal BSIM4.5 MOSFET `Device`.

    Raises:
        DimensionError: If a parameter has the wrong physical dimension.
        ValueError: If `card` names an unknown BSIM4 parameter or a non-finite value.

    Examples:
        >>> from jolty.units import V, nm, um
        >>> transistor = bsim4(
        ...     width=2 * um,
        ...     length=0.15 * um,
        ...     card={"toxe": 4 * nm, "vth0": 0.5 * V, "k2": -0.02},
        ... )
        >>> transistor.parameters["toxe"], transistor.parameters["k2"]
        (4e-09, -0.02)
    """
    values = locals()
    values.pop("card")
    return _device("bsim4", values, card=card, card_dimension=bsim4_dimension)


def _bsim4_card() -> dict[str, Dimension | None]:
    """Derives BSIM4 card parameter names and SI dimensions from the engine."""
    excluded = inspect.signature(bsim4).parameters
    card: dict[str, Dimension | None] = {}
    for name, kind in _model_kinds("bsim4").items():
        if name in excluded:
            continue
        match kind:
            case Dimension() | None:
                card[name] = kind
            case _:
                message = f"unexpected BSIM4 card kind for `{name}`: {kind!r}"
                raise ValueError(message)
    return card


BSIM4_CARD: Final[Mapping[str, Dimension | None]] = _bsim4_card()
"""Mapping from BSIM4.5 card parameter names to expected SI dimensions.

Plain numeric parameters map to ``None``, including exponent-dependent parameters such
as ``ll`` and manual-unit gate tunneling parameters such as ``aigc``."""

BSIM4_BINNED: Final = frozenset(_core.binned_parameters("bsim4"))
"""BSIM4 parameters supporting ``l``, ``w``, and ``p`` binning prefixes, scaled by
meters or square meters."""


def bsim4_dimension(name: str) -> Dimension | None:
    """Returns the expected SI dimension of BSIM4 parameter `name`, or ``None``.

    Binned prefixes multiply the base parameter dimension by meters (``l``, ``w``) or
    square meters (``p``).

    Args:
        name: BSIM4.5 direct or ``l``/``w``/``p``-binned card parameter name.

    Returns:
        Expected SI `Dimension`, or ``None`` for plain dimensionless numbers.

    Raises:
        ValueError: If BSIM4 defines no parameter named `name`.

    Examples:
        >>> bsim4_dimension("lvth0") == VOLTAGE * LENGTH, bsim4_dimension("k2")
        (True, None)
    """
    return _extended(*_bsim_parameter("BSIM4", name, BSIM4_CARD, BSIM4_BINNED))


type LogicLevel = Literal[0, 1] | None
"""Digital logic level: 0, 1, or ``None`` for an unknown state."""

type Gate = Literal["and", "or", "nand", "nor", "xor", "xnor", "not", "buffer"]
"""Supported standard logic gate function names."""

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


def logic_cell(  # noqa: PLR0913 - logic function, timing, and electrical pins
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
    """Creates a combinational logic cell mapping boolean inputs to outputs.

    Each entry in `outputs` provides either a truth-table string of ``0`` and ``1``
    characters (with the first input as MSB, such as ``"1110"`` for NAND) or a boolean
    callable over input levels. Unknown inputs propagate an unknown output unless every
    possible binary completion produces the same output level.

    Rising outputs switch after `rise` and falling outputs after `fall` (defaulting to
    `rise`), with edges taking `transition` to cross between rails. Propagation delay
    is inertial, filtering out input pulses shorter than the delay.

    Pure logic cells connect to event-driven logic nets. Setting `electrical=True` adds
    ``supply`` and ``ground`` terminals so pins may join electrical nets. Electrical
    inputs compare node voltages against `threshold` of supply and present
    `input_capacitance` (defaulting to zero), while electrical outputs drive ramped
    waveforms crossing half supply at transition times.

    Args:
        inputs: Ordered boolean input terminal names (first input is MSB).
        outputs: Mapping from output pin names to truth-table strings or callables.
        rise: Inertial propagation delay for rising output transitions.
        fall: Inertial propagation delay for falling edges, defaulting to `rise`.
        transition: Output ramp duration between logic rails.
        electrical: When ``True``, adds ``supply`` and ``ground`` electrical rails.
        input_capacitance: Shunt capacitance per input pin when `electrical` is true.
        threshold: Fractional supply threshold slicing electrical inputs.

    Returns:
        Combinational `logic_cell` `Device` with Pure-Logic or mixed-signal pins.

    Raises:
        DimensionError: If `rise`, `fall`, `transition`, or `input_capacitance` has the
            wrong physical dimension.
        ValueError: If `input_capacitance` or `threshold` is passed with
            ``electrical=False``, or if the engine rejects the cell.

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
        if threshold is not None:
            pins["threshold"] = finite(threshold, "threshold")
        parameters["electrical"] = pins
    elif input_capacitance is not None or threshold is not None:
        message = "only an electrical cell has an input capacitance and a threshold"
        raise ValueError(message)
    return device("logic_cell", parameters)


def _table(function: str | Callable[..., bool], inputs: int) -> str:
    """Builds a truth-table string from a literal table or a boolean callable."""
    if isinstance(function, str):
        return function
    rows = (
        [bool(row >> (inputs - 1 - bit) & 1) for bit in range(inputs)]
        for row in range(1 << inputs)
    )
    return "".join("1" if function(*row) else "0" for row in rows)


def logic_gate(  # noqa: PLR0913 - logic cell gate parameters
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
    """Creates a standard [`logic_cell`][jolty.devices.logic_cell] implementing `gate`.

    Input terminals are named sequentially from ``a`` (count given by `inputs`, or 1
    for ``not`` and ``buffer``), and the output terminal is ``y``. All timing and
    electrical options forward directly to [`logic_cell`][jolty.devices.logic_cell].

    Args:
        gate: Standard boolean gate function name.
        inputs: Input pin count (forced to 1 for `'not'` and `'buffer'`).
        rise: Inertial propagation delay for rising output transitions.
        fall: Inertial propagation delay for falling edges, defaulting to `rise`.
        transition: Output ramp duration between logic rails.
        electrical: When ``True``, adds ``supply`` and ``ground`` electrical rails.
        input_capacitance: Shunt capacitance per input pin when `electrical` is true.
        threshold: Fractional supply threshold slicing electrical inputs.

    Returns:
        Standard combinational gate `Device` with inputs ``a..`` and output ``y``.

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
    """Creates a piecewise-event logic source driving terminal ``output``.

    Output starts at `initial` and transitions at each time-ordered `(time, level)`
    pair in `changes`. Each edge ramps between ground and supply over `transition`,
    crossing half supply at the scheduled event time.

    Args:
        initial: Initial logic state (`0`, `1`, or ``None``) at time zero.
        changes: Time-ordered `(time, level)` transition events.
        transition: Edge ramp duration crossing half supply at event times.

    Returns:
        Single-terminal event-driven `logic_source` `Device`.

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
    """Formats a logic level as an engine character: ``0``, ``1``, or ``x``.

    Simulation results store numeric logic levels using NaN for unknown states.

    Args:
        level: Discrete `LogicLevel` or floating-point trace reading (NaN for unknown).

    Returns:
        Single-character engine state string: ``'0'``, ``'1'``, or ``'x'``.

    Examples:
        >>> [format_logic_level(level) for level in (0, 1, None, 1.0, math.nan)]
        ['0', '1', 'x', '1', 'x']
    """
    if level is None or math.isnan(level):
        return "x"
    return str(int(level))


def device(model: str, parameters: dict[str, JsonValue]) -> Device:
    """Creates a device of `model` with raw SI `parameters` and resolved terminals.

    Unlike the typed constructors above, this low-level helper accepts raw SI base-unit
    values without dimension checking, supporting imported models such as SPICE
    libraries.

    Args:
        model: Primitive engine model identifier.
        parameters: Validated parameter dictionary already expressed in SI base units.

    Returns:
        Constructed `Device` with engine-resolved terminal names.

    Raises:
        ValueError: If `model` is unknown or the engine rejects `parameters`.
    """
    terminals = _core.terminals(model, json.dumps(parameters))
    return Device(model, tuple(terminals), parameters)
