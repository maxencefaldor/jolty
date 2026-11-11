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


@dataclass(frozen=True, slots=True)
class Dimension:
    """The exponents of the SI base units kilogram, metre, second, ampere and kelvin."""

    kilogram: int = 0
    metre: int = 0
    second: int = 0
    ampere: int = 0
    kelvin: int = 0

    def _exponents(self) -> tuple[int, ...]:
        return (self.kilogram, self.metre, self.second, self.ampere, self.kelvin)

    def __mul__(self, other: Dimension) -> Dimension:
        """The dimension of a product."""
        exponents = zip(self._exponents(), other._exponents(), strict=True)
        return Dimension(*(a + b for a, b in exponents))

    def __truediv__(self, other: Dimension) -> Dimension:
        """The dimension of a quotient."""
        exponents = zip(self._exponents(), other._exponents(), strict=True)
        return Dimension(*(a - b for a, b in exponents))

    def __pow__(self, power: int) -> Dimension:
        """The dimension of a power."""
        return Dimension(*(exponent * power for exponent in self._exponents()))

    @override
    def __str__(self) -> str:
        """The unit's symbol, such as ``Ω``, or a product of base units, or ``1``."""
        symbol = _SYMBOLS.get(self)
        if symbol is not None:
            return symbol
        factors = [
            base if exponent == 1 else f"{base}^{exponent}"
            for base, exponent in zip(_BASES, self._exponents(), strict=True)
            if exponent
        ]
        return "·".join(factors) or "1"


DIMENSIONLESS: Final = Dimension()
MASS: Final = Dimension(kilogram=1)
LENGTH: Final = Dimension(metre=1)
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

_SYMBOLS: Final = {
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
}


@dataclass(frozen=True, slots=True)
class Quantity:
    """A value in SI base units and its dimension."""

    value: float
    dimension: Dimension

    def to(self, dimension: Dimension, name: str) -> float:
        """The value in SI base units, once its dimension is checked and it is finite.

        Raises:
            DimensionError: When the quantity has another dimension than `dimension`;
                the message names `name`, the expected unit and the given quantity.
            ValueError: When the value is infinite or not a number.
        """
        if self.dimension != dimension:
            message = f"{name} takes a quantity in {dimension}, not {self}"
            raise DimensionError(message)
        return finite(self.value, name)

    def __mul__(self, other: float | Quantity) -> Quantity:
        """The product, whose dimension is the product of the dimensions."""
        if isinstance(other, Quantity):
            return Quantity(self.value * other.value, self.dimension * other.dimension)
        return Quantity(self.value * other, self.dimension)

    def __rmul__(self, other: float) -> Quantity:
        """The product of a number and the quantity."""
        return Quantity(other * self.value, self.dimension)

    def __truediv__(self, other: float | Quantity) -> Quantity:
        """The quotient, whose dimension is the quotient of the dimensions."""
        if isinstance(other, Quantity):
            return Quantity(self.value / other.value, self.dimension / other.dimension)
        return Quantity(self.value / other, self.dimension)

    def __rtruediv__(self, other: float) -> Quantity:
        """The quotient of a number by the quantity, as a frequency from a period."""
        return Quantity(other / self.value, DIMENSIONLESS / self.dimension)

    def __add__(self, other: Quantity) -> Quantity:
        """The sum of two quantities of the same dimension."""
        return Quantity(self.value + self._same(other, "add"), self.dimension)

    def __sub__(self, other: Quantity) -> Quantity:
        """The difference of two quantities of the same dimension."""
        return Quantity(self.value - self._same(other, "subtract"), self.dimension)

    def __neg__(self) -> Quantity:
        """The opposite quantity."""
        return Quantity(-self.value, self.dimension)

    def _same(self, other: Quantity, verb: str) -> float:
        if not isinstance(other, Quantity) or other.dimension != self.dimension:
            message = f"cannot {verb} {self} and {other}"
            raise DimensionError(message)
        return other.value

    @override
    def __str__(self) -> str:
        """The value with an SI prefix and the unit's symbol, such as ``4.7 kΩ``."""
        symbol = "" if self.dimension == DIMENSIONLESS else str(self.dimension)
        return format_si(self.value, symbol)


def finite(value: float, name: str) -> float:
    """`value`, after checking it is finite.

    Raises:
        ValueError: When `value` is infinite or not a number; the message names `name`.
    """
    if not math.isfinite(value):
        message = f"{name} must be finite, not {value}"
        raise ValueError(message)
    return value


def format_si(value: float, symbol: str) -> str:
    """`value` to four significant digits, with the SI prefix that puts it in [1, 1000).

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
A: Final = Quantity(1.0, CURRENT)
mA: Final = Quantity(1e-3, CURRENT)
uA: Final = Quantity(1e-6, CURRENT)
nA: Final = Quantity(1e-9, CURRENT)
pA: Final = Quantity(1e-12, CURRENT)
ohm: Final = Quantity(1.0, RESISTANCE)
kohm: Final = Quantity(1e3, RESISTANCE)
Mohm: Final = Quantity(1e6, RESISTANCE)
S: Final = Quantity(1.0, CONDUCTANCE)
mS: Final = Quantity(1e-3, CONDUCTANCE)
uS: Final = Quantity(1e-6, CONDUCTANCE)
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
