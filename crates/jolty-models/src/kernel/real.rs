//! The numbers a model computes with: plain floats, or floats carried with their exact
//! derivatives with respect to some of its parameters, so the same equations give both.

use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

use crate::Lift;

/// A real number a model computes with.
///
/// `f64` is the number itself; [`Gradient`] carries its derivatives with respect to some
/// parameters alongside, by forward-mode automatic differentiation. Comparisons and
/// branches read the value alone, so a model takes the same branch with either. Only plain
/// `f64` parameters are read from or written to JSON; a twin lifts them into any `Real`
/// field by field through [`Lift`].
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
    + Lift<Self, Plain = f64>
{
    /// How many first-order derivative slots the number carries: none for `f64`.
    const SEEDS: usize;

    /// A number with no derivative.
    fn constant(value: f64) -> Self;

    /// The number `value` moving at rate one along the seed `slot`, and along no other: a
    /// constant when the number carries fewer seeds, as `f64` carries none.
    fn unit(value: f64, slot: usize) -> Self;

    /// The number `value` moving at `rates` along each seed, padded with zeros or truncated
    /// to [`Real::SEEDS`].
    fn seeded(value: f64, rates: &[f64]) -> Self;

    /// The number's value.
    fn value(self) -> f64;

    /// The number's first-order derivative along the seed `slot`, or zero when it carries
    /// fewer seeds.
    fn derivative(self, slot: usize) -> f64;

    /// The number `value`, moving as this one does: its derivatives with another value,
    /// such as a voltage's limited value, which moves as the voltage applied.
    #[must_use]
    fn with_value(self, value: f64) -> Self;

    /// A function of the number, given `function`, its value, slope and curvature at
    /// the number: a number carrying derivatives of the second order, such as [`Hyper`],
    /// reads the curvature.
    #[must_use]
    fn chain(self, function: (f64, f64, f64)) -> Self;

    /// `e` raised to the number.
    #[must_use]
    fn exp(self) -> Self {
        let exponential = self.value().exp();
        self.chain((exponential, exponential, exponential))
    }

    /// The natural logarithm.
    #[must_use]
    fn ln(self) -> Self {
        let value = self.value();
        self.chain((value.ln(), value.recip(), -(value * value).recip()))
    }

    /// `ln(1 + x)` of the number `x`, to its last digit where `x` is near zero, where
    /// `ln` of the rounded `1 + x` would lose them.
    #[must_use]
    fn ln_1p(self) -> Self {
        let value = self.value();
        let base = 1.0 + value;
        self.chain((value.ln_1p(), base.recip(), -(base * base).recip()))
    }

    /// `eˣ − 1` of the number `x`, to its last digit where `x` is near zero, where the
    /// difference of `eˣ` and 1 would lose them.
    #[must_use]
    fn exp_m1(self) -> Self {
        let value = self.value();
        let exponential = value.exp();
        self.chain((value.exp_m1(), exponential, exponential))
    }

    /// The square root.
    #[must_use]
    fn sqrt(self) -> Self {
        let value = self.value();
        let root = value.sqrt();
        self.chain((root, 0.5 / root, -0.25 / (root * value)))
    }

    /// The reciprocal.
    #[must_use]
    fn recip(self) -> Self {
        let value = self.value();
        let reciprocal = value.recip();
        self.chain((
            reciprocal,
            -(value * value).recip(),
            2.0 * reciprocal * reciprocal * reciprocal,
        ))
    }

    /// The number raised to the constant `exponent`.
    #[must_use]
    fn powf(self, exponent: f64) -> Self {
        let value = self.value();
        let slope = if exponent == 0.0 {
            0.0
        } else if (exponent - 1.0) == 0.0 {
            1.0
        } else {
            exponent * value.powf(exponent - 1.0)
        };
        let curvature_factor = exponent * (exponent - 1.0);
        let curvature = if curvature_factor == 0.0 {
            0.0
        } else {
            curvature_factor * value.powf(exponent - 2.0)
        };
        self.chain((value.powf(exponent), slope, curvature))
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
        self.chain((sine, self.value().cos(), -sine))
    }

    /// The tangent, of an angle in radians.
    #[must_use]
    fn tan(self) -> Self {
        let tangent = self.value().tan();
        let slope = tangent.mul_add(tangent, 1.0);
        self.chain((tangent, slope, 2.0 * tangent * slope))
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
    const SEEDS: usize = 0;

    fn constant(value: f64) -> Self {
        value
    }

    fn unit(value: f64, _: usize) -> Self {
        value
    }

    fn seeded(value: f64, _: &[f64]) -> Self {
        value
    }

    fn value(self) -> f64 {
        self
    }

    fn derivative(self, _: usize) -> f64 {
        0.0
    }

    fn with_value(self, value: f64) -> Self {
        value
    }

    fn chain(self, (value, _, _): (f64, f64, f64)) -> Self {
        value
    }

    fn exp(self) -> Self {
        f64::exp(self)
    }

    fn ln(self) -> Self {
        f64::ln(self)
    }

    fn ln_1p(self) -> Self {
        f64::ln_1p(self)
    }

    fn exp_m1(self) -> Self {
        f64::exp_m1(self)
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

/// A number and its exact derivatives with respect to `K` parameters or seeded unknowns,
/// one by default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gradient<const K: usize = 1> {
    /// The number.
    pub value: f64,
    /// Its derivative with respect to each seed.
    pub derivatives: [f64; K],
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

/// `factor * rate`, yielding `0.0` when `rate` is zero and `factor` is not `NaN` so an
/// infinite slope or curvature along an unmoved direction does not produce `NaN`.
#[inline]
fn scaled(factor: f64, rate: f64) -> f64 {
    if rate == 0.0 && !factor.is_nan() {
        0.0
    } else {
        factor * rate
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

impl<const K: usize> Real for Gradient<K> {
    const SEEDS: usize = K;

    fn constant(value: f64) -> Self {
        Self {
            value,
            derivatives: [0.0; K],
        }
    }

    fn unit(value: f64, slot: usize) -> Self {
        let mut unit = Self::constant(value);
        if let Some(derivative) = unit.derivatives.get_mut(slot) {
            *derivative = 1.0;
        }
        unit
    }

    fn seeded(value: f64, rates: &[f64]) -> Self {
        Self {
            value,
            derivatives: std::array::from_fn(|slot| rates.get(slot).copied().unwrap_or(0.0)),
        }
    }

    fn value(self) -> f64 {
        self.value
    }

    fn derivative(self, slot: usize) -> f64 {
        self.derivatives.get(slot).copied().unwrap_or(0.0)
    }

    fn with_value(self, value: f64) -> Self {
        Self { value, ..self }
    }

    fn chain(self, (value, slope, _): (f64, f64, f64)) -> Self {
        Self {
            value,
            derivatives: self.map(|derivative| scaled(slope, derivative)),
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

/// A number, its derivatives with respect to `K` parameters or seeded unknowns and to a
/// second variable, and its mixed derivatives with respect to both: second-order
/// forward-mode differentiation, as a device that heats computes when differentiated by a
/// twin, the second variable its temperature.
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

impl<const K: usize> Real for Hyper<K> {
    const SEEDS: usize = K;

    fn constant(value: f64) -> Self {
        Self {
            value,
            first: [0.0; K],
            second: 0.0,
            both: [0.0; K],
        }
    }

    fn unit(value: f64, slot: usize) -> Self {
        let mut unit = Self::constant(value);
        if let Some(first) = unit.first.get_mut(slot) {
            *first = 1.0;
        }
        unit
    }

    fn seeded(value: f64, rates: &[f64]) -> Self {
        Self::new(
            value,
            std::array::from_fn(|slot| rates.get(slot).copied().unwrap_or(0.0)),
        )
    }

    fn value(self) -> f64 {
        self.value
    }

    fn derivative(self, slot: usize) -> f64 {
        self.first.get(slot).copied().unwrap_or(0.0)
    }

    fn with_value(self, value: f64) -> Self {
        Self { value, ..self }
    }

    fn chain(self, (value, slope, curvature): (f64, f64, f64)) -> Self {
        Self {
            value,
            first: self.first.map(|first| scaled(slope, first)),
            second: scaled(slope, self.second),
            both: zip(self.both, self.first, |both, first| {
                let cross = if first == 0.0 || self.second == 0.0 {
                    0.0
                } else {
                    first * self.second
                };
                scaled(slope, both) + scaled(curvature, cross)
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
        Self {
            value: self.value - other.value,
            first: zip(self.first, other.first, |own, others| own - others),
            second: self.second - other.second,
            both: zip(self.both, other.both, |own, others| own - others),
        }
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
    fn powf_and_chain_avoid_zero_times_infinity_nans() {
        let zero_gradient = Gradient::new(0.0, [1.0, 0.0]);
        assert_eq!(zero_gradient.powf(0.0), Gradient::new(1.0, [0.0, 0.0]));
        assert_eq!(zero_gradient.powf(1.0), Gradient::new(0.0, [1.0, 0.0]));
        let sqrt_zero = Gradient::new(0.0, [0.0, 0.0]).sqrt();
        assert_eq!(sqrt_zero, Gradient::new(0.0, [0.0, 0.0]));

        let zero_hyper = Hyper {
            value: 0.0,
            first: [1.0, 0.0],
            second: 0.0,
            both: [0.0, 0.0],
        };
        assert_eq!(
            zero_hyper.powf(0.0),
            Hyper {
                value: 1.0,
                first: [0.0, 0.0],
                second: 0.0,
                both: [0.0, 0.0],
            }
        );
        assert_eq!(
            zero_hyper.powf(1.0),
            Hyper {
                value: 0.0,
                first: [1.0, 0.0],
                second: 0.0,
                both: [0.0, 0.0],
            }
        );
        // `0.0.powf(1.5)` has finite slope 0 and infinite curvature, which must not poison
        // `both` when `first * second == 0`.
        let three_halves = zero_hyper.powf(1.5);
        assert_eq!(three_halves.value, 0.0);
        assert_eq!(three_halves.first, [0.0, 0.0]);
        assert_eq!(three_halves.second, 0.0);
        assert_eq!(three_halves.both, [0.0, 0.0]);

        let subtracted = Hyper {
            value: 3.0,
            first: [2.0, -1.0],
            second: 4.0,
            both: [5.0, -6.0],
        } - Hyper {
            value: 1.0,
            first: [0.5, 2.0],
            second: 1.5,
            both: [-1.0, 3.0],
        };
        assert_eq!(
            subtracted,
            Hyper {
                value: 2.0,
                first: [1.5, -3.0],
                second: 2.5,
                both: [6.0, -9.0],
            }
        );
    }
}
