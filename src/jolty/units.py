# Copyright 2026 Maxence Faldor
"""Physical quantities, so a value in farads cannot go where ohms belong.

Build scripts write values as numbers times units, such as ``10 * kohm`` or ``4.7 *
uF``. A [`Quantity`][jolty.units.Quantity] holds its value in SI base units and its
[`Dimension`][jolty.units.Dimension]. Arithmetic combines dimensions, and every device
checks the dimension of each parameter as it reads it.

Examples:
    >>> from jolty.units import V, kohm
    >>> print(5 * V / (1 * kohm))
    5 mA
"""

import math
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Final, override

from jolty.errors import DimensionError

_BASES: Final = ("kg", "m", "s", "A", "K")
_PREFIXES: Final = {
    -15: "f",
    -12: "p",
    -9: "n",
    -6: "µ",
    -3: "m",
    0: "",
    3: "k",
    6: "M",
    9: "G",
    12: "T",
}


type Exponent = int | Fraction
"""Exponent of an SI base unit (integer or `Fraction`, as in V/√Hz)."""


@dataclass(frozen=True, slots=True)
class Dimension:
    """Exponents of the SI base units kilogram, meter, second, ampere, and kelvin."""

    kilogram: Exponent = 0
    meter: Exponent = 0
    second: Exponent = 0
    ampere: Exponent = 0
    kelvin: Exponent = 0

    def _exponents(self) -> tuple[Exponent, ...]:
        return (self.kilogram, self.meter, self.second, self.ampere, self.kelvin)

    def __mul__(self, other: Dimension) -> Dimension:
        """Returns the product of two dimensions."""
        exponents = zip(self._exponents(), other._exponents(), strict=True)
        return Dimension(*(a + b for a, b in exponents))

    def __truediv__(self, other: Dimension) -> Dimension:
        """Returns the quotient of two dimensions."""
        exponents = zip(self._exponents(), other._exponents(), strict=True)
        return Dimension(*(a - b for a, b in exponents))

    def __pow__(self, power: Exponent) -> Dimension:
        """Returns this dimension raised to `power`."""
        return Dimension(*(exponent * power for exponent in self._exponents()))

    @override
    def __str__(self) -> str:
        """Formats the named unit symbol (``Ω``, ``V⁻¹``) or a base-unit product.

        Examples:
            >>> str(VOLTAGE), str(DIMENSIONLESS / VOLTAGE), str(LENGTH * TIME)
            ('V', 'V⁻¹', 'm·s')
            >>> str(LENGTH ** Fraction(1, 2))
            'm^1/2'
        """
        symbol = _SYMBOLS.get(self)
        if symbol is not None:
            return symbol
        inverse = _SYMBOLS.get(DIMENSIONLESS / self)
        if inverse is not None:
            return f"{inverse}\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}"
        factors = [
            base if exponent == 1 else f"{base}^{exponent}"
            for base, exponent in zip(_BASES, self._exponents(), strict=True)
            if exponent
        ]
        return "·".join(factors) or "1"


DIMENSIONLESS: Final = Dimension()
MASS: Final = Dimension(kilogram=1)
LENGTH: Final = Dimension(meter=1)
TIME: Final = Dimension(second=1)
CURRENT: Final = Dimension(ampere=1)
TEMPERATURE: Final = Dimension(kelvin=1)
FREQUENCY: Final = DIMENSIONLESS / TIME
CHARGE: Final = CURRENT * TIME
ENERGY: Final = MASS * LENGTH**2 / TIME**2
POWER: Final = ENERGY / TIME
VOLTAGE: Final = POWER / CURRENT
RESISTANCE: Final = VOLTAGE / CURRENT
CONDUCTANCE: Final = DIMENSIONLESS / RESISTANCE
CAPACITANCE: Final = CHARGE / VOLTAGE
INDUCTANCE: Final = VOLTAGE * TIME / CURRENT
THERMAL_RESISTANCE: Final = TEMPERATURE / POWER
HEAT_CAPACITY: Final = ENERGY / TEMPERATURE

_SYMBOLS: Final = {
    LENGTH: "m",
    TIME: "s",
    CURRENT: "A",
    TEMPERATURE: "K",
    FREQUENCY: "Hz",
    CHARGE: "C",
    ENERGY: "J",
    POWER: "W",
    VOLTAGE: "V",
    RESISTANCE: "Ω",
    CONDUCTANCE: "S",
    CAPACITANCE: "F",
    INDUCTANCE: "H",
    THERMAL_RESISTANCE: "K/W",
    HEAT_CAPACITY: "J/K",
    CURRENT / VOLTAGE**2: "A/V\N{SUPERSCRIPT TWO}",
    FREQUENCY ** Fraction(1, 2): "\N{SQUARE ROOT}Hz",
    VOLTAGE / FREQUENCY ** Fraction(1, 2): "V/\N{SQUARE ROOT}Hz",
    CURRENT / FREQUENCY ** Fraction(1, 2): "A/\N{SQUARE ROOT}Hz",
}
"""Mapping from named dimensions to their SI unit symbols."""


@dataclass(frozen=True, slots=True)
class Quantity:
    """Physical value stored in SI base units paired with its dimension."""

    value: float
    dimension: Dimension

    def to(self, dimension: Dimension, name: str) -> float:
        """Validates dimension and finiteness and returns the scalar SI value.

        Args:
            dimension: Expected physical dimension for conversion.
            name: Parameter label included in error messages on mismatch.

        Returns:
            Finite scalar value expressed in SI base units.

        Raises:
            DimensionError: When the quantity dimension differs from `dimension`;
                the message names `name`, the expected unit, and the given quantity.
            ValueError: When the numeric value is infinite or NaN.
        """
        if self.dimension != dimension:
            message = f"{name} takes a quantity in {dimension}, not {self}"
            raise DimensionError(message)
        return finite(self.value, name)

    def __mul__(self, other: float | Quantity) -> Quantity:
        """Returns the product of this quantity and scalar or quantity `other`."""
        if isinstance(other, Quantity):
            return Quantity(self.value * other.value, self.dimension * other.dimension)
        return Quantity(self.value * other, self.dimension)

    def __rmul__(self, other: float) -> Quantity:
        """Returns the product of scalar `other` and this quantity."""
        return Quantity(other * self.value, self.dimension)

    def __truediv__(self, other: float | Quantity) -> Quantity:
        """Returns the quotient of this quantity and scalar or quantity `other`."""
        if isinstance(other, Quantity):
            return Quantity(self.value / other.value, self.dimension / other.dimension)
        return Quantity(self.value / other, self.dimension)

    def __rtruediv__(self, other: float) -> Quantity:
        """Returns the quotient of scalar `other` divided by this quantity."""
        return Quantity(other / self.value, DIMENSIONLESS / self.dimension)

    def compact(self) -> str:
        """Formats the quantity to 15 significant digits with an SI prefix, no space.

        Fifteen significant digits preserve script literals while eliminating binary
        floating-point noise in generated component names.

        Returns:
            Space-free SI-prefixed string rounded to 15 significant digits.

        Examples:
            >>> (1.00001 * kohm).compact(), (4.7 * nF).compact(), (0.02 / V).compact()
            ('1.00001kΩ', '4.7nF', '0.02V⁻¹')
        """
        value = float(f"{self.value:.15g}")
        symbol = _SYMBOLS.get(self.dimension)
        if symbol is None:
            unit = "" if self.dimension == DIMENSIONLESS else str(self.dimension)
            return f"{value!r}{unit}"
        if value == 0 or not math.isfinite(value):
            return f"{value!r}{symbol}"
        exponent = 3 * math.floor(math.log10(abs(value)) / 3)
        exponent = min(max(exponent, min(_PREFIXES)), max(_PREFIXES))
        # Shifting the shortest decimal form of the value keeps its digits exact.
        mantissa = Decimal(repr(value)).scaleb(-exponent).normalize()
        return f"{mantissa:f}{_PREFIXES[exponent]}{symbol}"

    def __pow__(self, power: Exponent) -> Quantity:
        """Returns this quantity raised to whole or fractional `power`.

        Examples:
            >>> str(1 * mA / V**2)
            '1 mA/V²'
            >>> str(4 * nV / Hz ** Fraction(1, 2))
            '4 nV/√Hz'
        """
        return Quantity(math.pow(self.value, power), self.dimension**power)

    def __add__(self, other: Quantity) -> Quantity:
        """Returns the sum of two quantities with matching dimensions."""
        return Quantity(self.value + self._same(other, "add"), self.dimension)

    def __sub__(self, other: Quantity) -> Quantity:
        """Returns the difference of two quantities with matching dimensions."""
        return Quantity(self.value - self._same(other, "subtract"), self.dimension)

    def __neg__(self) -> Quantity:
        """Returns the negated quantity."""
        return Quantity(-self.value, self.dimension)

    def __lt__(self, other: Quantity) -> bool:
        """Returns whether this quantity is strictly less than `other`."""
        return self.value < self._same(other, "compare")

    def __le__(self, other: Quantity) -> bool:
        """Returns whether this quantity is less than or equal to `other`."""
        return self.value <= self._same(other, "compare")

    def __gt__(self, other: Quantity) -> bool:
        """Returns whether this quantity is strictly greater than `other`."""
        return self.value > self._same(other, "compare")

    def __ge__(self, other: Quantity) -> bool:
        """Returns whether this quantity is greater than or equal to `other`."""
        return self.value >= self._same(other, "compare")

    def _same(self, other: Quantity, verb: str) -> float:
        if not isinstance(other, Quantity) or other.dimension != self.dimension:
            message = f"cannot {verb} {self} and {other}"
            raise DimensionError(message)
        return other.value

    @override
    def __str__(self) -> str:
        """Formats the quantity with an SI prefix and unit symbol, such as ``4.7 kΩ``.

        Compound units without a dedicated symbol omit prefixes to avoid binding only
        to the first factor, and dimensionless values print as plain numbers.

        Examples:
            >>> str(0.02 / V), str(0.3 * V / V)
            ('0.02 V⁻¹', '0.3')
        """
        if self.dimension == DIMENSIONLESS:
            return f"{float(f'{self.value:.4g}'):g}"
        if self.dimension in _SYMBOLS:
            return format_si(self.value, _SYMBOLS[self.dimension])
        return f"{float(f'{self.value:.4g}'):g} {self.dimension}"


ZERO_CELSIUS: Final = 273.15
"""Zero degrees Celsius expressed in kelvins (273.15 K)."""


def celsius(degrees: float) -> Quantity:
    """Converts `degrees` Celsius into a temperature `Quantity` in kelvins.

    Args:
        degrees: Temperature on the Celsius scale.

    Returns:
        Temperature `Quantity` in kelvins (`degrees + 273.15`).

    Examples:
        >>> celsius(27).value
        300.15
    """
    return Quantity(degrees + ZERO_CELSIUS, TEMPERATURE)


def finite(value: float, name: str) -> float:
    """Validates that `value` is finite and returns `value`.

    Args:
        value: Numeric scalar to check for finiteness.
        name: Parameter label included in the `ValueError` message when non-finite.

    Returns:
        Validated finite floating-point `value`.

    Raises:
        ValueError: When `value` is infinite or NaN; the message names `name`.
    """
    if not math.isfinite(value):
        message = f"{name} is finite, not {value}"
        raise ValueError(message)
    return value


def format_si(value: float, symbol: str) -> str:
    """Formats `value` to four significant digits with an SI prefix in [1, 1000).

    Args:
        value: Numeric magnitude in SI base units.
        symbol: Base unit symbol (such as `"V"` or `"Ω"`) placed after the SI prefix.

    Returns:
        Human-readable string formatted to four significant digits with an SI prefix.

    Examples:
        >>> format_si(4700.0, "Ω")
        '4.7 kΩ'
        >>> format_si(-2.5e-9, "F")
        '-2.5 nF'
        >>> format_si(0.99999, "V")
        '1 V'
    """
    rounded = float(f"{value:.4g}")
    if rounded == 0 or not math.isfinite(rounded):
        return f"{rounded:g} {symbol}".rstrip()
    exponent = 3 * math.floor(math.log10(abs(rounded)) / 3)
    exponent = min(max(exponent, min(_PREFIXES)), max(_PREFIXES))
    mantissa = rounded / 10.0**exponent
    return f"{mantissa:.4g} {_PREFIXES[exponent]}{symbol}".rstrip()


# Units to multiply numbers by. The micro prefix is written "u" in names.
m: Final = Quantity(1.0, LENGTH)
mm: Final = Quantity(1e-3, LENGTH)
um: Final = Quantity(1e-6, LENGTH)
nm: Final = Quantity(1e-9, LENGTH)
s: Final = Quantity(1.0, TIME)
ms: Final = Quantity(1e-3, TIME)
us: Final = Quantity(1e-6, TIME)
ns: Final = Quantity(1e-9, TIME)
ps: Final = Quantity(1e-12, TIME)
Hz: Final = Quantity(1.0, FREQUENCY)
kHz: Final = Quantity(1e3, FREQUENCY)
MHz: Final = Quantity(1e6, FREQUENCY)
GHz: Final = Quantity(1e9, FREQUENCY)
V: Final = Quantity(1.0, VOLTAGE)
kV: Final = Quantity(1e3, VOLTAGE)
mV: Final = Quantity(1e-3, VOLTAGE)
uV: Final = Quantity(1e-6, VOLTAGE)
nV: Final = Quantity(1e-9, VOLTAGE)
A: Final = Quantity(1.0, CURRENT)
mA: Final = Quantity(1e-3, CURRENT)
uA: Final = Quantity(1e-6, CURRENT)
nA: Final = Quantity(1e-9, CURRENT)
pA: Final = Quantity(1e-12, CURRENT)
fA: Final = Quantity(1e-15, CURRENT)
C: Final = Quantity(1.0, CHARGE)
mC: Final = Quantity(1e-3, CHARGE)
uC: Final = Quantity(1e-6, CHARGE)
nC: Final = Quantity(1e-9, CHARGE)
pC: Final = Quantity(1e-12, CHARGE)
fC: Final = Quantity(1e-15, CHARGE)
ohm: Final = Quantity(1.0, RESISTANCE)
kohm: Final = Quantity(1e3, RESISTANCE)
Mohm: Final = Quantity(1e6, RESISTANCE)
S: Final = Quantity(1.0, CONDUCTANCE)
mS: Final = Quantity(1e-3, CONDUCTANCE)
uS: Final = Quantity(1e-6, CONDUCTANCE)
nS: Final = Quantity(1e-9, CONDUCTANCE)
pS: Final = Quantity(1e-12, CONDUCTANCE)
F: Final = Quantity(1.0, CAPACITANCE)
mF: Final = Quantity(1e-3, CAPACITANCE)
uF: Final = Quantity(1e-6, CAPACITANCE)
nF: Final = Quantity(1e-9, CAPACITANCE)
pF: Final = Quantity(1e-12, CAPACITANCE)
fF: Final = Quantity(1e-15, CAPACITANCE)
H: Final = Quantity(1.0, INDUCTANCE)
mH: Final = Quantity(1e-3, INDUCTANCE)
uH: Final = Quantity(1e-6, INDUCTANCE)
nH: Final = Quantity(1e-9, INDUCTANCE)
W: Final = Quantity(1.0, POWER)
mW: Final = Quantity(1e-3, POWER)
uW: Final = Quantity(1e-6, POWER)
K: Final = Quantity(1.0, TEMPERATURE)
J: Final = Quantity(1.0, ENERGY)
mJ: Final = Quantity(1e-3, ENERGY)
rtHz: Final = Hz ** Fraction(1, 2)
"""Square root of hertz, used for noise densities such as ``4 * nV / rtHz``."""
