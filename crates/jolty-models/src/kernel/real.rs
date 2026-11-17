//! The numbers a model computes with: plain floats, or floats carried with their exact
//! derivatives with respect to some of its parameters, so the same equations give both.

use std::borrow::Cow;
use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::Lift;

/// A real number a model computes with.
///
/// `f64` is the number itself; [`Gradient`] carries its derivatives with respect to some
/// parameters alongside, by forward-mode automatic differentiation. Comparisons and
/// branches read the value alone, so a model takes the same branch with either. A
/// parameter file reads it, it writes as it reads, a JSON schema describes it, and a twin
/// makes it of a plain number by the path the number is written at.
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
    + Serialize
    + JsonSchema
    + Lift<Self, Plain = f64>
{
    /// A number with no derivative.
    fn constant(value: f64) -> Self;

    /// The number's value.
    fn value(self) -> f64;

    /// `f(self)`, given `f`'s value, slope and curvature at `self`: a number carrying
    /// derivatives of the second order, such as [`Hyper`], reads the curvature.
    #[must_use]
    fn chain(self, value: f64, slope: f64, curvature: f64) -> Self;

    /// `e` raised to the number.
    #[must_use]
    fn exp(self) -> Self {
        let exponential = self.value().exp();
        self.chain(exponential, exponential, exponential)
    }

    /// The natural logarithm.
    #[must_use]
    fn ln(self) -> Self {
        let value = self.value();
        self.chain(value.ln(), value.recip(), -(value * value).recip())
    }

    /// `ln(1 + x)` of the number `x`, to its last digit where `x` is near zero, where
    /// `ln` of the rounded `1 + x` would lose them.
    #[must_use]
    fn ln_1p(self) -> Self {
        let value = self.value();
        let base = 1.0 + value;
        self.chain(value.ln_1p(), base.recip(), -(base * base).recip())
    }

    /// `eˣ − 1` of the number `x`, to its last digit where `x` is near zero, where the
    /// difference of `eˣ` and 1 would lose them.
    #[must_use]
    fn exp_m1(self) -> Self {
        let value = self.value();
        let exponential = value.exp();
        self.chain(value.exp_m1(), exponential, exponential)
    }

    /// The square root.
    #[must_use]
    fn sqrt(self) -> Self {
        let value = self.value();
        let root = value.sqrt();
        self.chain(root, 0.5 / root, -0.25 / (root * value))
    }

    /// The reciprocal.
    #[must_use]
    fn recip(self) -> Self {
        let value = self.value();
        let reciprocal = value.recip();
        self.chain(
            reciprocal,
            -(value * value).recip(),
            2.0 * reciprocal * reciprocal * reciprocal,
        )
    }

    /// The number raised to the constant `exponent`.
    #[must_use]
    fn powf(self, exponent: f64) -> Self {
        let value = self.value();
        self.chain(
            value.powf(exponent),
            exponent * value.powf(exponent - 1.0),
            exponent * (exponent - 1.0) * value.powf(exponent - 2.0),
        )
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
        let sine = self.value().sin();
        self.chain(sine, self.value().cos(), -sine)
    }

    /// The tangent, of an angle in radians.
    #[must_use]
    fn tan(self) -> Self {
        let tangent = self.value().tan();
        let slope = tangent.mul_add(tangent, 1.0);
        self.chain(tangent, slope, 2.0 * tangent * slope)
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

    fn chain(self, value: f64, _: f64, _: f64) -> Self {
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

/// A number and its exact derivatives with respect to `K` parameters, one by default.
///
/// In a parameter file, a number reads as a constant, and the object
/// `{"value": v, "derivatives": [d₁, …, d_K]}` as a number moving with each parameter at
/// those rates, as a twin seeds the parameters it differentiates against. It writes as it
/// reads: a constant as a number, any other as that object.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gradient<const K: usize = 1> {
    /// The number.
    pub value: f64,
    /// Its derivative with respect to each parameter.
    pub derivatives: [f64; K],
}

/// How a [`Gradient`] is written: a plain number, or a number and its derivatives.
#[derive(Deserialize, Serialize, JsonSchema)]
#[serde(untagged)]
enum Written {
    /// A constant.
    Constant(f64),
    /// A number and its derivative with respect to each parameter.
    Seeded { value: f64, derivatives: Vec<f64> },
}

impl<const K: usize> Gradient<K> {
    /// The number `value` moving at the rates `derivatives`.
    #[must_use]
    pub const fn new(value: f64, derivatives: [f64; K]) -> Self {
        Self { value, derivatives }
    }

    /// Each derivative, as `combine` makes it of the same derivative of `self` and of
    /// `other`.
    fn zip(self, other: Self, combine: impl Fn(f64, f64) -> f64) -> [f64; K] {
        zip(self.derivatives, other.derivatives, combine)
    }

    /// Each derivative, as `map` makes it of the same derivative of `self`.
    fn map(self, map: impl Fn(f64) -> f64) -> [f64; K] {
        self.derivatives.map(map)
    }

    /// The derivatives of `self · other`, by the product rule.
    fn product(self, other: Self) -> [f64; K] {
        self.zip(other, |own, others| own * other.value + self.value * others)
    }

    /// The derivatives of `self / other`, whose value is `value`, by the quotient rule.
    fn quotient(self, other: Self, value: f64) -> [f64; K] {
        self.zip(other, |own, others| (own - value * others) / other.value)
    }
}

/// Each element of `own`, as `combine` makes it of that element and the same one of
/// `others`.
fn zip<const K: usize>(
    own: [f64; K],
    others: [f64; K],
    combine: impl Fn(f64, f64) -> f64,
) -> [f64; K] {
    let mut combined = own;
    for (element, other) in combined.iter_mut().zip(others) {
        *element = combine(*element, other);
    }
    combined
}

impl<const K: usize> Default for Gradient<K> {
    fn default() -> Self {
        Self::constant(0.0)
    }
}

impl<const K: usize> Serialize for Gradient<K> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let written = if self.derivatives.iter().all(|&derivative| derivative == 0.0) {
            Written::Constant(self.value)
        } else {
            Written::Seeded {
                value: self.value,
                derivatives: self.derivatives.to_vec(),
            }
        };
        written.serialize(serializer)
    }
}

impl<'de, const K: usize> Deserialize<'de> for Gradient<K> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match Written::deserialize(deserializer)? {
            Written::Constant(value) => Ok(Self::constant(value)),
            Written::Seeded { value, derivatives } => {
                let count = derivatives.len();
                let derivatives = derivatives.try_into().map_err(|_| {
                    D::Error::custom(format!(
                        "a number of {K} parameters has {K} derivatives, not {count}"
                    ))
                })?;
                Ok(Self { value, derivatives })
            }
        }
    }
}

impl<const K: usize> JsonSchema for Gradient<K> {
    fn schema_name() -> Cow<'static, str> {
        Cow::Borrowed("Gradient")
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        Written::json_schema(generator)
    }
}

impl<const K: usize> Real for Gradient<K> {
    fn constant(value: f64) -> Self {
        Self {
            value,
            derivatives: [0.0; K],
        }
    }

    fn value(self) -> f64 {
        self.value
    }

    fn chain(self, value: f64, slope: f64, _: f64) -> Self {
        Self {
            value,
            derivatives: self.map(|derivative| slope * derivative),
        }
    }
}

impl<const K: usize> Add for Gradient<K> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            value: self.value + other.value,
            derivatives: self.zip(other, |own, others| own + others),
        }
    }
}

impl<const K: usize> Sub for Gradient<K> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            value: self.value - other.value,
            derivatives: self.zip(other, |own, others| own - others),
        }
    }
}

impl<const K: usize> Mul for Gradient<K> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            value: self.value * other.value,
            derivatives: self.product(other),
        }
    }
}

impl<const K: usize> Div for Gradient<K> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let value = self.value / other.value;
        Self {
            value,
            derivatives: self.quotient(other, value),
        }
    }
}

impl<const K: usize> Neg for Gradient<K> {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            value: -self.value,
            derivatives: self.map(|derivative| -derivative),
        }
    }
}

impl<const K: usize> Add<f64> for Gradient<K> {
    type Output = Self;

    fn add(self, other: f64) -> Self {
        Self {
            value: self.value + other,
            derivatives: self.derivatives,
        }
    }
}

impl<const K: usize> Sub<f64> for Gradient<K> {
    type Output = Self;

    fn sub(self, other: f64) -> Self {
        Self {
            value: self.value - other,
            derivatives: self.derivatives,
        }
    }
}

impl<const K: usize> Mul<f64> for Gradient<K> {
    type Output = Self;

    fn mul(self, other: f64) -> Self {
        Self {
            value: self.value * other,
            derivatives: self.map(|derivative| derivative * other),
        }
    }
}

impl<const K: usize> Div<f64> for Gradient<K> {
    type Output = Self;

    fn div(self, other: f64) -> Self {
        Self {
            value: self.value / other,
            derivatives: self.map(|derivative| derivative / other),
        }
    }
}

impl<const K: usize> AddAssign for Gradient<K> {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl<const K: usize> SubAssign for Gradient<K> {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

/// A number, its derivatives with respect to `K` parameters and to a second variable, and
/// its derivatives with respect to each parameter and that variable both: second-order
/// forward-mode differentiation, as a device that heats computes when its derivatives with
/// respect to its parameters are asked for, the second variable its temperature.
///
/// In a parameter file it reads as a [`Gradient`] does, its derivatives with respect to the
/// parameters, and writes as one. No parameter has a derivative with respect to the second
/// variable, so a number with one does not write.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hyper<const K: usize = 1> {
    /// The number.
    pub value: f64,
    /// Its derivative with respect to each parameter.
    pub first: [f64; K],
    /// Its derivative with respect to the second variable.
    pub second: f64,
    /// Its derivative with respect to each parameter and the second variable.
    pub both: [f64; K],
}

impl<const K: usize> Hyper<K> {
    /// The number `value` moving at the rates `first` along the parameters, and not with
    /// the second variable.
    #[must_use]
    pub const fn new(value: f64, first: [f64; K]) -> Self {
        Self {
            value,
            first,
            second: 0.0,
            both: [0.0; K],
        }
    }

    /// `self · other`, each derivative by the product rule.
    fn product(self, other: Self) -> Self {
        let first = zip(self.first, other.first, |own, others| {
            own * other.value + self.value * others
        });
        let mut both = self.both;
        let firsts = self.first.into_iter().zip(other.first);
        for ((both, (own, others)), other_both) in both.iter_mut().zip(firsts).zip(other.both) {
            *both = *both * other.value
                + own * other.second
                + self.second * others
                + self.value * other_both;
        }
        Self {
            value: self.value * other.value,
            first,
            second: self.second * other.value + self.value * other.second,
            both,
        }
    }

    /// `self / other`, its derivatives those of `self = q · other` solved for `q`'s.
    fn quotient(self, other: Self) -> Self {
        let value = self.value / other.value;
        let first = zip(self.first, other.first, |own, others| {
            (own - value * others) / other.value
        });
        let second = (self.second - value * other.second) / other.value;
        let mut both = self.both;
        let firsts = first.into_iter().zip(other.first);
        for ((both, (quotient, others)), other_both) in both.iter_mut().zip(firsts).zip(other.both)
        {
            *both = (*both - quotient * other.second - second * others - value * other_both)
                / other.value;
        }
        Self {
            value,
            first,
            second,
            both,
        }
    }
}

impl<const K: usize> Default for Hyper<K> {
    fn default() -> Self {
        Self::constant(0.0)
    }
}

impl<const K: usize> Serialize for Hyper<K> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if self.second != 0.0 || self.both.iter().any(|&both| both != 0.0) {
            return Err(serde::ser::Error::custom(
                "a number with a derivative with respect to a second variable does not write",
            ));
        }
        Gradient::new(self.value, self.first).serialize(serializer)
    }
}

impl<'de, const K: usize> Deserialize<'de> for Hyper<K> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let Gradient { value, derivatives } = Gradient::<K>::deserialize(deserializer)?;
        Ok(Self::new(value, derivatives))
    }
}

impl<const K: usize> JsonSchema for Hyper<K> {
    fn schema_name() -> Cow<'static, str> {
        Gradient::<K>::schema_name()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        Gradient::<K>::json_schema(generator)
    }
}

impl<const K: usize> Real for Hyper<K> {
    fn constant(value: f64) -> Self {
        Self {
            value,
            first: [0.0; K],
            second: 0.0,
            both: [0.0; K],
        }
    }

    fn value(self) -> f64 {
        self.value
    }

    fn chain(self, value: f64, slope: f64, curvature: f64) -> Self {
        Self {
            value,
            first: self.first.map(|first| slope * first),
            second: slope * self.second,
            both: zip(self.both, self.first, |both, first| {
                slope * both + curvature * first * self.second
            }),
        }
    }
}

impl<const K: usize> Add for Hyper<K> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            value: self.value + other.value,
            first: zip(self.first, other.first, |own, others| own + others),
            second: self.second + other.second,
            both: zip(self.both, other.both, |own, others| own + others),
        }
    }
}

impl<const K: usize> Sub for Hyper<K> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + -other
    }
}

impl<const K: usize> Mul for Hyper<K> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.product(other)
    }
}

impl<const K: usize> Div for Hyper<K> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        self.quotient(other)
    }
}

impl<const K: usize> Neg for Hyper<K> {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            value: -self.value,
            first: self.first.map(|first| -first),
            second: -self.second,
            both: self.both.map(|both| -both),
        }
    }
}

impl<const K: usize> Add<f64> for Hyper<K> {
    type Output = Self;

    fn add(self, other: f64) -> Self {
        Self {
            value: self.value + other,
            ..self
        }
    }
}

impl<const K: usize> Sub<f64> for Hyper<K> {
    type Output = Self;

    fn sub(self, other: f64) -> Self {
        Self {
            value: self.value - other,
            ..self
        }
    }
}

impl<const K: usize> Mul<f64> for Hyper<K> {
    type Output = Self;

    fn mul(self, other: f64) -> Self {
        Self {
            value: self.value * other,
            first: self.first.map(|first| first * other),
            second: self.second * other,
            both: self.both.map(|both| both * other),
        }
    }
}

impl<const K: usize> Div<f64> for Hyper<K> {
    type Output = Self;

    fn div(self, other: f64) -> Self {
        Self {
            value: self.value / other,
            first: self.first.map(|first| first / other),
            second: self.second / other,
            both: self.both.map(|both| both / other),
        }
    }
}

impl<const K: usize> AddAssign for Hyper<K> {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl<const K: usize> SubAssign for Hyper<K> {
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
        let found = exact(Gradient::new(x, [1.0]));
        let [derivative] = found.derivatives;
        assert!((found.value - plain(x)).abs() < 1e-12);
        assert!((derivative - central(&plain, x)).abs() < 1e-6);
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

    /// A function of two numbers, each seeded on a parameter of its own, that exercises
    /// every operation a number has.
    fn exercise<R: Real>(x: R, y: R) -> R {
        let z = (x * y + x * 3.0) / (y - 2.0) - (x / y).exp().ln_1p().sqrt();
        (z.powf(1.5).recip() + z.pow(y) * 0.5 - x.abs().max(y).min(z)) / 4.0 + z.sin().tan()
    }

    #[test]
    fn each_slot_of_a_chunk_is_the_number_of_one_parameter_bit_for_bit() {
        let (x, y) = (0.7, 2.6);
        let chunk = exercise(
            Gradient::new(x, [1.0, 0.0, 0.0]),
            Gradient::new(y, [0.0, 1.0, 0.0]),
        );
        let alone = [
            exercise(Gradient::new(x, [1.0]), Gradient::constant(y)),
            exercise(Gradient::constant(x), Gradient::new(y, [1.0])),
        ];
        assert_eq!(chunk.value.to_bits(), alone[0].value.to_bits());
        for (slot, one) in alone.iter().enumerate() {
            assert_eq!(
                chunk.derivatives[slot].to_bits(),
                one.derivatives[0].to_bits()
            );
        }
        assert_eq!(chunk.derivatives[2].to_bits(), 0.0_f64.to_bits());
        let warm = |first: [f64; 2], second: f64| Hyper {
            value: x,
            first,
            second,
            both: [0.0; 2],
        };
        let hot = exercise(warm([1.0, 0.0], 1.0), Hyper::constant(y));
        let one = exercise(
            Hyper {
                value: x,
                first: [1.0],
                second: 1.0,
                both: [0.0],
            },
            Hyper::constant(y),
        );
        assert_eq!(hot.first[0].to_bits(), one.first[0].to_bits());
        assert_eq!(hot.both[0].to_bits(), one.both[0].to_bits());
        assert_eq!(hot.second.to_bits(), one.second.to_bits());
    }

    #[test]
    fn a_seeded_number_reads_as_the_parameters_differentiated_against() {
        let constant: Gradient = serde_json::from_str("2.5").unwrap();
        assert_eq!(constant, Gradient::constant(2.5));
        let seeded: Gradient =
            serde_json::from_str(r#"{"value": 2.5, "derivatives": [1]}"#).unwrap();
        assert_eq!(seeded, Gradient::new(2.5, [1.0]));
        let chunk: Gradient<2> =
            serde_json::from_str(r#"{"value": 2.5, "derivatives": [0, 3]}"#).unwrap();
        assert_eq!(chunk, Gradient::new(2.5, [0.0, 3.0]));
        let mismatched =
            serde_json::from_str::<Gradient<2>>(r#"{"value": 2.5, "derivatives": [1]}"#);
        assert!(mismatched.is_err());
    }

    #[test]
    fn a_number_with_derivatives_writes_as_it_reads() {
        let seeded = Gradient::new(2.5, [1.0, 0.0]);
        let written = serde_json::to_value(seeded).unwrap();
        assert_eq!(
            written,
            serde_json::json!({"value": 2.5, "derivatives": [1.0, 0.0]})
        );
        assert_eq!(
            serde_json::from_value::<Gradient<2>>(written).unwrap(),
            seeded
        );
        let constant = serde_json::to_value(Gradient::<2>::constant(-3.0)).unwrap();
        assert_eq!(constant, serde_json::json!(-3.0));
        assert_eq!(
            serde_json::from_value::<Gradient<2>>(constant).unwrap(),
            Gradient::constant(-3.0)
        );
        let hyper: Hyper = serde_json::from_str(r#"{"value": 2.5, "derivatives": [1]}"#).unwrap();
        let written = serde_json::to_value(hyper).unwrap();
        assert_eq!(serde_json::from_value::<Hyper>(written).unwrap(), hyper);
        let warm = Hyper {
            second: 1.0,
            ..hyper
        };
        assert!(serde_json::to_value(warm).is_err());
    }
}
