//! The numbers a model computes with: plain floats, or floats carried with their exact
//! derivative with respect to one parameter, so the same equations give both.

use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

use serde::Deserialize;

/// A real number a model computes with.
///
/// `f64` is the number itself; [`Gradient`] carries its derivative with respect to one
/// parameter alongside, by forward-mode automatic differentiation. Comparisons and
/// branches read the value alone, so a model takes the same branch with either.
pub trait Real:
    Copy
    + fmt::Debug
    + Default
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
    + AddAssign
    + SubAssign
    + for<'de> Deserialize<'de>
{
    /// A number with no derivative.
    fn constant(value: f64) -> Self;

    /// The number's value.
    fn value(self) -> f64;

    /// `f(self)`, given `f`'s value and slope at `self`.
    #[must_use]
    fn chain(self, value: f64, slope: f64) -> Self;

    /// `e` raised to the number.
    #[must_use]
    fn exp(self) -> Self {
        let exponential = self.value().exp();
        self.chain(exponential, exponential)
    }

    /// The natural logarithm.
    #[must_use]
    fn ln(self) -> Self {
        self.chain(self.value().ln(), self.value().recip())
    }

    /// The square root.
    #[must_use]
    fn sqrt(self) -> Self {
        let root = self.value().sqrt();
        self.chain(root, 0.5 / root)
    }

    /// The reciprocal.
    #[must_use]
    fn recip(self) -> Self {
        let value = self.value();
        self.chain(value.recip(), -(value * value).recip())
    }

    /// The number raised to the constant `exponent`.
    #[must_use]
    fn powf(self, exponent: f64) -> Self {
        let value = self.value();
        self.chain(value.powf(exponent), exponent * value.powf(exponent - 1.0))
    }

    /// The number raised to `exponent`, itself a number with a derivative.
    #[must_use]
    fn pow(self, exponent: Self) -> Self {
        (self.ln() * exponent).exp()
    }

    /// The magnitude.
    #[must_use]
    fn abs(self) -> Self {
        if self.value() < 0.0 { -self } else { self }
    }

    /// The sine, of an angle in radians.
    #[must_use]
    fn sin(self) -> Self {
        self.chain(self.value().sin(), self.value().cos())
    }

    /// The tangent, of an angle in radians.
    #[must_use]
    fn tan(self) -> Self {
        let tangent = self.value().tan();
        self.chain(tangent, tangent.mul_add(tangent, 1.0))
    }

    /// The least non-negative remainder of division by `divisor`: the number less
    /// `divisor` times the floor of their quotient.
    #[must_use]
    fn rem_euclid(self, divisor: Self) -> Self {
        let quotient = self.value().div_euclid(divisor.value());
        self - divisor * quotient
    }

    /// The larger of the two, by value.
    #[must_use]
    fn max(self, other: Self) -> Self {
        if other.value() > self.value() {
            other
        } else {
            self
        }
    }

    /// The smaller of the two, by value.
    #[must_use]
    fn min(self, other: Self) -> Self {
        if other.value() < self.value() {
            other
        } else {
            self
        }
    }
}

impl Real for f64 {
    fn constant(value: f64) -> Self {
        value
    }

    fn value(self) -> f64 {
        self
    }

    fn chain(self, value: f64, _: f64) -> Self {
        value
    }

    fn exp(self) -> Self {
        f64::exp(self)
    }

    fn ln(self) -> Self {
        f64::ln(self)
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    fn recip(self) -> Self {
        f64::recip(self)
    }

    fn powf(self, exponent: f64) -> Self {
        f64::powf(self, exponent)
    }

    fn pow(self, exponent: Self) -> Self {
        f64::powf(self, exponent)
    }

    fn abs(self) -> Self {
        f64::abs(self)
    }

    fn sin(self) -> Self {
        f64::sin(self)
    }

    fn tan(self) -> Self {
        f64::tan(self)
    }

    fn rem_euclid(self, divisor: Self) -> Self {
        f64::rem_euclid(self, divisor)
    }

    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }

    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }
}

/// A number and its exact derivative with respect to one parameter.
///
/// In a parameter file, a number reads as a constant, and the object
/// `{"value": v, "derivative": 1}` as the parameter differentiated against.
#[derive(Debug, Clone, Copy, Default, PartialEq, Deserialize)]
#[serde(from = "Written")]
pub struct Gradient {
    /// The number.
    pub value: f64,
    /// Its derivative with respect to the parameter.
    pub derivative: f64,
}

/// How a [`Gradient`] is written: a plain number, or a number and its derivative.
#[derive(Deserialize)]
#[serde(untagged)]
enum Written {
    Constant(f64),
    Seeded { value: f64, derivative: f64 },
}

impl From<Written> for Gradient {
    fn from(written: Written) -> Self {
        match written {
            Written::Constant(value) => Self::constant(value),
            Written::Seeded { value, derivative } => Self { value, derivative },
        }
    }
}

impl Real for Gradient {
    fn constant(value: f64) -> Self {
        Self {
            value,
            derivative: 0.0,
        }
    }

    fn value(self) -> f64 {
        self.value
    }

    fn chain(self, value: f64, slope: f64) -> Self {
        Self {
            value,
            derivative: slope * self.derivative,
        }
    }
}

impl Add for Gradient {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            value: self.value + other.value,
            derivative: self.derivative + other.derivative,
        }
    }
}

impl Sub for Gradient {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            value: self.value - other.value,
            derivative: self.derivative - other.derivative,
        }
    }
}

impl Mul for Gradient {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            value: self.value * other.value,
            derivative: self.derivative * other.value + self.value * other.derivative,
        }
    }
}

impl Div for Gradient {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let value = self.value / other.value;
        Self {
            value,
            derivative: (self.derivative - value * other.derivative) / other.value,
        }
    }
}

impl Neg for Gradient {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            value: -self.value,
            derivative: -self.derivative,
        }
    }
}

impl Add<f64> for Gradient {
    type Output = Self;

    fn add(self, other: f64) -> Self {
        Self {
            value: self.value + other,
            derivative: self.derivative,
        }
    }
}

impl Sub<f64> for Gradient {
    type Output = Self;

    fn sub(self, other: f64) -> Self {
        Self {
            value: self.value - other,
            derivative: self.derivative,
        }
    }
}

impl Mul<f64> for Gradient {
    type Output = Self;

    fn mul(self, other: f64) -> Self {
        Self {
            value: self.value * other,
            derivative: self.derivative * other,
        }
    }
}

impl Div<f64> for Gradient {
    type Output = Self;

    fn div(self, other: f64) -> Self {
        Self {
            value: self.value / other,
            derivative: self.derivative / other,
        }
    }
}

impl AddAssign for Gradient {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl SubAssign for Gradient {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The derivative of `f` at `x` by central differences, to check exact ones against.
    fn central(f: &impl Fn(f64) -> f64, x: f64) -> f64 {
        let step = 1e-6 * x.abs().max(1.0);
        (f(x + step) - f(x - step)) / (2.0 * step)
    }

    /// Checks `exact`, computed with gradients, against `plain` and its central
    /// difference at 0.7.
    fn check(exact: impl Fn(Gradient) -> Gradient, plain: impl Fn(f64) -> f64) {
        let x = 0.7;
        let found = exact(Gradient {
            value: x,
            derivative: 1.0,
        });
        assert!((found.value - plain(x)).abs() < 1e-12);
        assert!((found.derivative - central(&plain, x)).abs() < 1e-6);
    }

    #[test]
    fn gradients_follow_the_rules_of_calculus() {
        check(
            |g| (g * g + g * 3.0) / (g - 2.0),
            |v| (v * v + 3.0 * v) / (v - 2.0),
        );
        check(|g| g.exp().ln().sqrt(), |v| v.exp().ln().sqrt());
        check(|g| g.powf(2.5).recip(), |v| v.powf(2.5).recip());
        check(|g| g.pow(g), |v| v.powf(v));
        check(
            |g| (-g).abs().max(Gradient::constant(0.1)),
            |v| v.abs().max(0.1),
        );
        check(
            |g| g.min(Gradient::constant(2.0)) / 4.0 - 1.0,
            |v| v.min(2.0) / 4.0 - 1.0,
        );
        check(|g| (g * 3.0).sin(), |v| (3.0 * v).sin());
        check(|g| (g * 0.5).tan(), |v| (0.5 * v).tan());
        check(
            |g| (g * 10.0).rem_euclid(g + 1.0),
            |v| (10.0 * v).rem_euclid(v + 1.0),
        );
    }

    #[test]
    fn a_seeded_number_reads_as_the_parameter_differentiated_against() {
        let constant: Gradient = serde_json::from_str("2.5").unwrap();
        assert_eq!(constant, Gradient::constant(2.5));
        let seeded: Gradient = serde_json::from_str(r#"{"value": 2.5, "derivative": 1}"#).unwrap();
        assert_eq!(
            seeded,
            Gradient {
                value: 2.5,
                derivative: 1.0
            }
        );
    }
}
