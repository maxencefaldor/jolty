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
    /// Number of first-order derivative lanes carried by the scalar (`0` for `f64`).
    const SEEDS: usize;

    /// Constructs a constant scalar with zero derivatives.
    fn constant(value: f64) -> Self;

    /// Constructs a scalar at `value` with unit derivative along seed `slot` (or a constant
    /// when `slot >= Self::SEEDS`).
    fn unit(value: f64, slot: usize) -> Self;

    /// Constructs a scalar at `value` carrying directional derivatives `rates`, zero-padded
    /// or truncated to [`Real::SEEDS`].
    fn seeded(value: f64, rates: &[f64]) -> Self;

    /// Returns the primal scalar value.
    fn value(self) -> f64;

    /// Returns the first-order derivative along seed `slot`, or `0.0` if out of range.
    fn derivative(self, slot: usize) -> f64;

    /// Replaces the primal value with `value` while preserving all derivatives (for example,
    /// clamping a voltage while retaining sensitivity to the applied voltage).
    #[must_use]
    fn with_value(self, value: f64) -> Self;

    /// Applies the chain rule using `(value, slope, curvature)` of a univariate function at
    /// `self`; second-order types such as [`Hyper`] propagate `curvature`.
    #[must_use]
    fn chain(self, function: (f64, f64, f64)) -> Self;

    /// Computes `e^self`.
    #[must_use]
    fn exp(self) -> Self {
        let exponential = self.value().exp();
        self.chain((exponential, exponential, exponential))
    }

    /// Computes the natural logarithm `ln(self)`.
    #[must_use]
    fn ln(self) -> Self {
        let value = self.value();
        self.chain((value.ln(), value.recip(), -(value * value).recip()))
    }

    /// Computes `ln(1 + self)` accurately near zero without catastrophic cancellation.
    #[must_use]
    fn ln_1p(self) -> Self {
        let value = self.value();
        let base = 1.0 + value;
        self.chain((value.ln_1p(), base.recip(), -(base * base).recip()))
    }

    /// Computes `e^self − 1` accurately near zero without catastrophic cancellation.
    #[must_use]
    fn exp_m1(self) -> Self {
        let value = self.value();
        let exponential = value.exp();
        self.chain((value.exp_m1(), exponential, exponential))
    }

    /// Computes the square root `√self`.
    #[must_use]
    fn sqrt(self) -> Self {
        let value = self.value();
        let root = value.sqrt();
        self.chain((root, 0.5 / root, -0.25 / (root * value)))
    }

    /// Computes the reciprocal `1 / self`.
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

    /// Raises `self` to constant power `exponent`.
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

    /// Raises `self` to differentiable power `exponent`.
    #[must_use]
    fn pow(self, exponent: Self) -> Self {
        (self.ln() * exponent).exp()
    }

    /// Computes the absolute value `|self|`.
    #[must_use]
    fn abs(self) -> Self {
        if self.value() < 0.0 { -self } else { self }
    }

    /// Computes the sine of an angle in radians.
    #[must_use]
    fn sin(self) -> Self {
        let sine = self.value().sin();
        self.chain((sine, self.value().cos(), -sine))
    }

    /// Computes the tangent of an angle in radians.
    #[must_use]
    fn tan(self) -> Self {
        let tangent = self.value().tan();
        let slope = tangent.mul_add(tangent, 1.0);
        self.chain((tangent, slope, 2.0 * tangent * slope))
    }

    /// Computes the arctangent, returning an angle in radians.
    #[must_use]
    fn atan(self) -> Self {
        let value = self.value();
        let slope = 1.0 / (1.0 + value * value);
        self.chain((value.atan(), slope, -2.0 * value * slope * slope))
    }

    /// Computes the least non-negative Euclidean remainder
    /// `self − divisor · floor(self / divisor)`.
    #[must_use]
    fn rem_euclid(self, divisor: Self) -> Self {
        let quotient = self.value().div_euclid(divisor.value());
        self - divisor * quotient
    }

    /// Returns the larger of `self` and `other` by primal value.
    #[must_use]
    fn max(self, other: Self) -> Self {
        if other.value() > self.value() {
            other
        } else {
            self
        }
    }

    /// Returns the smaller of `self` and `other` by primal value.
    #[must_use]
    fn min(self, other: Self) -> Self {
        if other.value() < self.value() {
            other
        } else {
            self
        }
    }
}

/// Forwards [`Real`] methods on `f64` directly to the inherent `f64` operations.
macro_rules! forward_f64 {
    ($($method:ident),+ $(,)?) => {
        $(
            fn $method(self) -> Self {
                f64::$method(self)
            }
        )+
    };
    ($($method:ident($arg:ident: $ty:ty) => $target:ident),+ $(,)?) => {
        $(
            fn $method(self, $arg: $ty) -> Self {
                f64::$target(self, $arg)
            }
        )+
    };
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

    forward_f64!(exp, ln, ln_1p, exp_m1, sqrt, recip, abs, sin, tan, atan);
    forward_f64!(
        powf(exponent: f64) => powf,
        pow(exponent: Self) => powf,
        rem_euclid(divisor: Self) => rem_euclid,
        max(other: Self) => max,
        min(other: Self) => min,
    );
}

/// Primal value paired with first-order derivatives along `K` parameter or unknown seeds
/// (defaulting to 1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gradient<const K: usize = 1> {
    /// Primal scalar value.
    pub value: f64,
    /// First-order derivatives with respect to each seed.
    pub derivatives: [f64; K],
}

impl<const K: usize> Gradient<K> {
    /// Constructs a gradient number from primal `value` and directional `derivatives`.
    #[must_use]
    pub const fn new(value: f64, derivatives: [f64; K]) -> Self {
        Self { value, derivatives }
    }

    /// Multiplies `self` by `other` using the product rule.
    fn product(self, other: Self) -> Self {
        Self {
            value: self.value * other.value,
            derivatives: zip(self.derivatives, other.derivatives, |own, others| {
                own * other.value + self.value * others
            }),
        }
    }

    /// Divides `self` by `other` using the quotient rule.
    fn quotient(self, other: Self) -> Self {
        let value = self.value / other.value;
        Self {
            value,
            derivatives: zip(self.derivatives, other.derivatives, |own, others| {
                (own - value * others) / other.value
            }),
        }
    }
}

/// Multiplies `factor * rate`, returning `0.0` when `rate == 0.0` and `factor` is not `NaN`
/// so infinite slopes along unseeded directions do not produce `NaN`.
#[inline]
fn scaled(factor: f64, rate: f64) -> f64 {
    if rate == 0.0 && !factor.is_nan() {
        0.0
    } else {
        factor * rate
    }
}

/// Combines corresponding lanes of `own` and `others` elementwise using `combine`.
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

/// Implements `Default`, `Add`, `Sub`, `Mul`, `Div`, `Neg`, scalar `f64` arithmetic and
/// assignment operators for a forward-mode number type with scalar fields and `[f64; K]`
/// derivative arrays, delegating `Mul` and `Div` to its `product` and `quotient` methods.
macro_rules! impl_number_ops {
    (
        @linear $name:ident, [$($scalar:ident),*], [$($array:ident),*],
        $trait:ident, $method:ident, $op:tt
    ) => {
        impl<const K: usize> $trait for $name<K> {
            type Output = Self;

            fn $method(self, other: Self) -> Self {
                Self {
                    value: self.value $op other.value,
                    $($scalar: self.$scalar $op other.$scalar,)*
                    $($array: zip(self.$array, other.$array, |own, others| own $op others),)*
                }
            }
        }
    };
    (@shift $name:ident, $(($trait:ident, $method:ident, $op:tt)),+) => {$(
        impl<const K: usize> $trait<f64> for $name<K> {
            type Output = Self;

            fn $method(self, other: f64) -> Self {
                Self {
                    value: self.value $op other,
                    ..self
                }
            }
        }
    )+};
    (
        @scale $name:ident, [$($scalar:ident),*], [$($array:ident),*],
        $trait:ident, $method:ident, $op:tt
    ) => {
        impl<const K: usize> $trait<f64> for $name<K> {
            type Output = Self;

            fn $method(self, other: f64) -> Self {
                Self {
                    value: self.value $op other,
                    $($scalar: self.$scalar $op other,)*
                    $($array: self.$array.map(|derivative| derivative $op other),)*
                }
            }
        }
    };
    (@assign $name:ident, $(($trait:ident, $method:ident, $op:tt)),+) => {$(
        impl<const K: usize> $trait for $name<K> {
            fn $method(&mut self, other: Self) {
                *self = *self $op other;
            }
        }
    )+};
    ($name:ident, scalars: [$($scalar:ident),*], arrays: [$($array:ident),*]) => {
        impl<const K: usize> Default for $name<K> {
            fn default() -> Self {
                Self::constant(0.0)
            }
        }

        impl_number_ops!(@linear $name, [$($scalar),*], [$($array),*], Add, add, +);
        impl_number_ops!(@linear $name, [$($scalar),*], [$($array),*], Sub, sub, -);
        impl_number_ops!(@shift $name, (Add, add, +), (Sub, sub, -));
        impl_number_ops!(@scale $name, [$($scalar),*], [$($array),*], Mul, mul, *);
        impl_number_ops!(@scale $name, [$($scalar),*], [$($array),*], Div, div, /);
        impl_number_ops!(
            @assign $name, (AddAssign, add_assign, +), (SubAssign, sub_assign, -)
        );

        impl<const K: usize> Mul for $name<K> {
            type Output = Self;

            fn mul(self, other: Self) -> Self {
                self.product(other)
            }
        }

        impl<const K: usize> Div for $name<K> {
            type Output = Self;

            fn div(self, other: Self) -> Self {
                self.quotient(other)
            }
        }

        impl<const K: usize> Neg for $name<K> {
            type Output = Self;

            fn neg(self) -> Self {
                Self {
                    value: -self.value,
                    $($scalar: -self.$scalar,)*
                    $($array: self.$array.map(|derivative| -derivative),)*
                }
            }
        }
    };
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
            derivatives: self.derivatives.map(|derivative| scaled(slope, derivative)),
        }
    }
}

impl_number_ops!(Gradient, scalars: [], arrays: [derivatives]);

/// Second-order hyper-dual scalar tracking primal value, first derivatives along `K` seeds
/// and one secondary variable (temperature), plus all mixed second derivatives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hyper<const K: usize = 1> {
    /// Primal scalar value.
    pub value: f64,
    /// First-order derivatives with respect to each parameter or unknown seed.
    pub first: [f64; K],
    /// First-order derivative with respect to the secondary variable.
    pub second: f64,
    /// Mixed second-order derivatives with respect to each seed and the secondary variable.
    pub both: [f64; K],
}

impl<const K: usize> Hyper<K> {
    /// Constructs a hyper-dual number at `value` with first-order seed rates `first` and
    /// zero secondary or mixed derivatives.
    #[must_use]
    pub const fn new(value: f64, first: [f64; K]) -> Self {
        Self {
            value,
            first,
            second: 0.0,
            both: [0.0; K],
        }
    }

    /// Multiplies `self` by `other`, propagating first and mixed second derivatives via the
    /// product rule.
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

    /// Divides `self` by `other`, solving `self = q · other` for all first and mixed second
    /// derivatives of the quotient `q`.
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

impl_number_ops!(Hyper, scalars: [second], arrays: [first, both]);

#[cfg(test)]
mod tests {
    use super::*;

    /// Approximates the derivative of `f` at `x` by central differences as a reference check.
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
        check(|g| (g * 0.5).atan(), |v| (0.5 * v).atan());
        check(
            |g| (g * 10.0).rem_euclid(g + 1.0),
            |v| (10.0 * v).rem_euclid(v + 1.0),
        );
        let hyper = Hyper {
            value: 0.7,
            first: [2.0],
            second: 3.0,
            both: [5.0],
        }
        .atan();
        let slope = 1.0 / (1.0 + 0.7 * 0.7);
        let curvature = -2.0 * 0.7 * slope * slope;
        assert!((hyper.value - 0.7_f64.atan()).abs() < 1e-15);
        assert!((hyper.first[0] - 2.0 * slope).abs() < 1e-15);
        assert!((hyper.second - 3.0 * slope).abs() < 1e-15);
        assert!((hyper.both[0] - (5.0 * slope + 6.0 * curvature)).abs() < 1e-14);
    }

    /// Evaluates a composite test expression exercising every [`Real`] operation on `x` and
    /// `y`.
    fn exercise<R: Real>(x: R, y: R) -> R {
        let z = (x * y + x * 3.0) / (y - 2.0) - (x / y).exp().ln_1p().sqrt();
        (z.powf(1.5).recip() + z.pow(y) * 0.5 - x.abs().max(y).min(z)) / 4.0
            + z.sin().tan()
            + x.atan()
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
