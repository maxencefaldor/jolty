//! Physical dimensions and parameter metadata for device models.
//!
//! Each device parameter is either a physical [`Dimension`] in SI base units, a plain
//! dimensionless number, a discrete choice, or a parameter whose dimension depends on
//! another parameter of the same device. Exponents are stored doubled so integer and
//! half-integer powers (such as square-root volts) are represented exactly without
//! floating point.

/// The exponents of the five SI base units (kilogram, meter, second, ampere, kelvin),
/// each stored doubled so integer and half-integer powers are exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dimension([i8; 5]);

impl Dimension {
    /// A dimension from doubled SI base exponents `[2*kg, 2*m, 2*s, 2*A, 2*K]`.
    #[must_use]
    pub const fn new(exponents: [i8; 5]) -> Self {
        Self(exponents)
    }

    /// The doubled SI base exponents `[2*kg, 2*m, 2*s, 2*A, 2*K]`.
    #[must_use]
    pub const fn exponents(self) -> [i8; 5] {
        self.0
    }

    /// The dimension of the product `self * other`.
    #[must_use]
    pub const fn mul(self, other: Self) -> Self {
        let [
            left_kilogram,
            left_meter,
            left_second,
            left_ampere,
            left_kelvin,
        ] = self.0;
        let [
            right_kilogram,
            right_meter,
            right_second,
            right_ampere,
            right_kelvin,
        ] = other.0;
        Self([
            left_kilogram + right_kilogram,
            left_meter + right_meter,
            left_second + right_second,
            left_ampere + right_ampere,
            left_kelvin + right_kelvin,
        ])
    }

    /// The dimension of the quotient `self / other`.
    #[must_use]
    pub const fn div(self, other: Self) -> Self {
        let [
            left_kilogram,
            left_meter,
            left_second,
            left_ampere,
            left_kelvin,
        ] = self.0;
        let [
            right_kilogram,
            right_meter,
            right_second,
            right_ampere,
            right_kelvin,
        ] = other.0;
        Self([
            left_kilogram - right_kilogram,
            left_meter - right_meter,
            left_second - right_second,
            left_ampere - right_ampere,
            left_kelvin - right_kelvin,
        ])
    }

    /// The dimension of `self` raised to the integer power `exponent`.
    #[must_use]
    pub const fn pow(self, exponent: i8) -> Self {
        let [kilogram, meter, second, ampere, kelvin] = self.0;
        Self([
            kilogram * exponent,
            meter * exponent,
            second * exponent,
            ampere * exponent,
            kelvin * exponent,
        ])
    }

    /// The reciprocal dimension `1 / self`.
    #[must_use]
    pub const fn recip(self) -> Self {
        self.pow(-1)
    }

    /// The square root of a dimension whose base exponents are whole numbers.
    #[must_use]
    pub const fn sqrt(self) -> Self {
        let [kilogram, meter, second, ampere, kelvin] = self.0;
        Self([kilogram / 2, meter / 2, second / 2, ampere / 2, kelvin / 2])
    }
}

/// Dimensionless quantity (`1`).
pub const DIMENSIONLESS: Dimension = Dimension::new([0, 0, 0, 0, 0]);
/// Mass (`kg`).
pub const MASS: Dimension = Dimension::new([2, 0, 0, 0, 0]);
/// Length (`m`).
pub const LENGTH: Dimension = Dimension::new([0, 2, 0, 0, 0]);
/// Time (`s`).
pub const TIME: Dimension = Dimension::new([0, 0, 2, 0, 0]);
/// Electric current (`A`).
pub const CURRENT: Dimension = Dimension::new([0, 0, 0, 2, 0]);
/// Thermodynamic temperature (`K`).
pub const TEMPERATURE: Dimension = Dimension::new([0, 0, 0, 0, 2]);
/// Frequency (`Hz = s^-1`).
pub const FREQUENCY: Dimension = DIMENSIONLESS.div(TIME);
/// Electric charge (`C = A * s`).
pub const CHARGE: Dimension = CURRENT.mul(TIME);
/// Energy (`J = kg * m^2 / s^2`).
pub const ENERGY: Dimension = MASS.mul(LENGTH.pow(2)).div(TIME.pow(2));
/// Power (`W = J / s`).
pub const POWER: Dimension = ENERGY.div(TIME);
/// Electric potential (`V = W / A`).
pub const VOLTAGE: Dimension = POWER.div(CURRENT);
/// Electrical resistance (`ohm = V / A`).
pub const RESISTANCE: Dimension = VOLTAGE.div(CURRENT);
/// Electrical conductance (`S = 1 / ohm`).
pub const CONDUCTANCE: Dimension = DIMENSIONLESS.div(RESISTANCE);
/// Electrical capacitance (`F = C / V`).
pub const CAPACITANCE: Dimension = CHARGE.div(VOLTAGE);
/// Electrical inductance (`H = V * s / A`).
pub const INDUCTANCE: Dimension = VOLTAGE.mul(TIME).div(CURRENT);
/// Thermal resistance (`K / W`).
pub const THERMAL_RESISTANCE: Dimension = TEMPERATURE.div(POWER);
/// Heat capacity (`J / K`).
pub const HEAT_CAPACITY: Dimension = ENERGY.div(TEMPERATURE);
/// Linear temperature coefficient (`K^-1`).
pub const PER_KELVIN: Dimension = DIMENSIONLESS.div(TEMPERATURE);
/// Quadratic temperature coefficient (`K^-2`).
pub const PER_KELVIN_SQUARED: Dimension = DIMENSIONLESS.div(TEMPERATURE.pow(2));
/// Linear voltage coefficient or channel-length modulation (`V^-1`).
pub const PER_VOLT: Dimension = DIMENSIONLESS.div(VOLTAGE);
/// Quadratic voltage coefficient (`V^-2`).
pub const PER_VOLT_SQUARED: Dimension = DIMENSIONLESS.div(VOLTAGE.pow(2));
/// Field-effect transistor transconductance parameter (`A / V^2`).
pub const TRANSCONDUCTANCE_PARAMETER: Dimension = CURRENT.div(VOLTAGE.pow(2));
/// Carrier mobility (`m^2 / (V * s)`).
pub const MOBILITY: Dimension = LENGTH.pow(2).div(VOLTAGE.mul(TIME));
/// Voltage temperature coefficient (`V / K`).
pub const VOLTS_PER_KELVIN: Dimension = VOLTAGE.div(TEMPERATURE);

/// What kind of value a model parameter expects in Python constructors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterKind {
    /// A physical quantity with an SI dimension.
    Quantity(Dimension),
    /// A plain dimensionless finite real number.
    Number,
    /// A discrete option (enum, boolean, or string) passed through unchanged.
    Choice,
    /// A physical quantity whose dimension depends on another parameter, identified by tag.
    Conditional(&'static str),
}

/// The derive of [`ParameterMetadata`](trait@ParameterMetadata), for a kernel's parameter
/// struct.
pub use jolty_derive::ParameterMetadata;

/// Static parameter dimension and binning metadata for a kernel's parameter struct.
pub trait ParameterMetadata {
    /// Every base parameter of the model and its expected kind, in declaration order.
    const PARAMETER_KINDS: &'static [(&'static str, ParameterKind)];

    /// Base names of parameters that expand into `l`, `w`, and `p` geometry-binning
    /// dependences.
    const BINNED_PARAMETERS: &'static [&'static str] = &[];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_si_dimensions_have_expected_doubled_exponents() {
        assert_eq!(DIMENSIONLESS.exponents(), [0, 0, 0, 0, 0]);
        assert_eq!(VOLTAGE.exponents(), [2, 4, -6, -2, 0]);
        assert_eq!(RESISTANCE.exponents(), [2, 4, -6, -4, 0]);
        assert_eq!(CONDUCTANCE.exponents(), [-2, -4, 6, 4, 0]);
        assert_eq!(CAPACITANCE.exponents(), [-2, -4, 8, 4, 0]);
        assert_eq!(INDUCTANCE.exponents(), [2, 4, -4, -4, 0]);
        assert_eq!(THERMAL_RESISTANCE.exponents(), [-2, -4, 6, 0, 2]);
        assert_eq!(HEAT_CAPACITY.exponents(), [2, 4, -4, 0, -2]);
        assert_eq!(PER_KELVIN.exponents(), [0, 0, 0, 0, -2]);
        assert_eq!(PER_KELVIN_SQUARED.exponents(), [0, 0, 0, 0, -4]);
        assert_eq!(PER_VOLT.exponents(), [-2, -4, 6, 2, 0]);
        assert_eq!(PER_VOLT_SQUARED.exponents(), [-4, -8, 12, 4, 0]);
        assert_eq!(TRANSCONDUCTANCE_PARAMETER.exponents(), [-4, -8, 12, 6, 0]);
        assert_eq!(MOBILITY.exponents(), [-2, 0, 4, 2, 0]);
        assert_eq!(VOLTS_PER_KELVIN.exponents(), [2, 4, -6, -2, -2]);
        assert_eq!(VOLTAGE.sqrt().exponents(), [1, 2, -3, -1, 0]);
        assert_eq!(VOLTAGE.sqrt().recip().exponents(), [-1, -2, 3, 1, 0]);
        assert_eq!(RESISTANCE.mul(CAPACITANCE), TIME);
        assert_eq!(VOLTAGE.div(CURRENT), RESISTANCE);
    }
}
