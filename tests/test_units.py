# Copyright 2026 Maxence Faldor

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jolty import DimensionError
from jolty.units import (
    CAPACITANCE,
    CURRENT,
    DIMENSIONLESS,
    FREQUENCY,
    RESISTANCE,
    TIME,
    VOLTAGE,
    Dimension,
    Quantity,
    V,
    format_si,
    kohm,
    mA,
    ms,
    nF,
    uF,
)

_PREFIXES = {"f": -15, "p": -12, "n": -9, "µ": -6, "m": -3, "k": 3, "M": 6, "G": 9}


def test_dimensions_combine_like_their_units() -> None:
    assert VOLTAGE / CURRENT == RESISTANCE
    assert RESISTANCE * CAPACITANCE == TIME
    assert DIMENSIONLESS / TIME == FREQUENCY
    assert (VOLTAGE**2 / RESISTANCE) / (VOLTAGE * CURRENT) == DIMENSIONLESS


def test_dimensions_print_as_symbols_or_base_units() -> None:
    assert str(RESISTANCE) == "Ω"
    assert str(DIMENSIONLESS) == "1"
    assert str(Dimension(metre=1, second=-1)) == "m·s^-1"


def test_quantities_carry_their_dimension_through_arithmetic() -> None:
    current = 5 * V / (1 * kohm)
    assert current.dimension == CURRENT
    assert current.to(CURRENT, "a current") == pytest.approx(5e-3)
    assert (1 * kohm * (1 * uF)).dimension == TIME
    assert (1 / (1 * ms)).to(FREQUENCY, "a frequency") == pytest.approx(1e3)
    assert (2 * mA - 1 * mA).to(CURRENT, "a current") == pytest.approx(1e-3)
    assert str(-(4.7 * nF)) == "-4.7 nF"


def test_a_quantity_of_the_wrong_dimension_names_what_it_was_for() -> None:
    with pytest.raises(DimensionError, match=r"resistance takes a quantity in Ω, not"):
        (1 * uF).to(RESISTANCE, "resistance")
    with pytest.raises(DimensionError, match=r"cannot add 1 V and 1 mA"):
        _ = 1 * V + 1 * mA


def test_a_quantity_must_be_finite() -> None:
    with pytest.raises(ValueError, match="a level must be finite, not inf"):
        Quantity(math.inf, VOLTAGE).to(VOLTAGE, "a level")


def test_format_si_picks_the_prefix_after_rounding() -> None:
    assert format_si(4700.0, "Ω") == "4.7 kΩ"
    assert format_si(0.0, "V") == "0 V"
    assert format_si(0.99999, "V") == "1 V"
    assert format_si(999.95e-6, "A") == "1 mA"
    assert format_si(2.5, "") == "2.5"
    assert format_si(1e-20, "F") == "1e-05 fF"


@given(
    st.floats(min_value=1e-15, max_value=1e11)
    | st.floats(min_value=-1e11, max_value=-1e-15)
)
def test_format_si_keeps_four_significant_digits(value: float) -> None:
    text = format_si(value, "V")
    mantissa, unit = text.split(" ")
    exponent = _PREFIXES.get(unit[:-1], 0)
    assert 1 <= abs(float(mantissa)) < 1000
    assert float(mantissa) * 10.0**exponent == pytest.approx(value, rel=5e-4)
