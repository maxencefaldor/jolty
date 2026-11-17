//! Forward-mode automatic differentiation: a value carried with its gradient, so a model
//! writes its equations once and gets their exact derivatives with them.
//!
//! The value and the derivatives are any [`Real`], so a model that differentiates with
//! respect to its voltages, for its Jacobian, also differentiates each of them with respect
//! to a parameter when it computes with [`crate::Gradient`].

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::Real;

/// A value and its derivatives with respect to `N` variables.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual<R: Real, const N: usize> {
    /// The value.
    pub value: R,
    /// Its derivative with respect to each variable.
    pub gradient: [R; N],
}

impl<R: Real, const N: usize> Dual<R, N> {
    /// A constant, whose derivatives are all zero.
    #[must_use]
    pub fn constant(value: R) -> Self {
        Self {
            value,
            gradient: [R::constant(0.0); N],
        }
    }

    /// The constant `value`, a plain number.
    #[must_use]
    pub fn number(value: f64) -> Self {
        Self::constant(R::constant(value))
    }

    /// The `N` variables at `values`, each the derivative of itself only.
    #[must_use]
    pub fn variables(values: [f64; N]) -> [Self; N] {
        let mut variables = values.map(|value| Self::constant(R::constant(value)));
        for (index, variable) in variables.iter_mut().enumerate() {
            if let Some(derivative) = variable.gradient.get_mut(index) {
                *derivative = R::constant(1.0);
            }
        }
        variables
    }

    /// `f(self)`, given `f`'s value and derivative at `self`.
    #[must_use]
    pub fn chain(self, (value, slope): (R, R)) -> Self {
        Self {
            value,
            gradient: self.gradient.map(|derivative| slope * derivative),
        }
    }

    /// `e` raised to the value.
    #[must_use]
    pub fn exp(self) -> Self {
        let exponential = self.value.exp();
        self.chain((exponential, exponential))
    }

    /// The natural logarithm.
    pub(crate) fn ln(self) -> Self {
        self.chain((self.value.ln(), self.value.recip()))
    }

    /// `ln(1 + x)` of the value `x`, as [`Real::ln_1p`] computes it.
    pub(crate) fn ln_1p(self) -> Self {
        self.chain((self.value.ln_1p(), (self.value + 1.0).recip()))
    }

    /// `eˣ − 1` of the value `x`, as [`Real::exp_m1`] computes it.
    pub(crate) fn exp_m1(self) -> Self {
        self.chain((self.value.exp_m1(), self.value.exp()))
    }

    /// The value, which must be positive, raised to `exponent`, a number that the
    /// variables do not move: the derivative, `exponent · value^exponent / value`, divides
    /// the power by the value rather than raising the value again.
    pub(crate) fn pow(self, exponent: R) -> Self {
        let power = self.value.pow(exponent);
        self.chain((power, exponent * power / self.value))
    }

    /// The reciprocal.
    #[must_use]
    pub fn recip(self) -> Self {
        let reciprocal = self.value.recip();
        self.chain((reciprocal, -(reciprocal * reciprocal)))
    }

    /// The square root.
    #[must_use]
    pub fn sqrt(self) -> Self {
        let root = self.value.sqrt();
        self.chain((root, R::constant(0.5) / root))
    }

    /// The larger of `self` and the constant `floor`: the floor itself, with no
    /// derivatives, where `self` falls below it.
    #[must_use]
    pub fn max(self, floor: f64) -> Self {
        if self.value.value() < floor {
            Self::number(floor)
        } else {
            self
        }
    }

    /// The product with `factor`, a number that the variables do not move.
    #[must_use]
    pub fn scale(self, factor: R) -> Self {
        Self {
            value: self.value * factor,
            gradient: self.gradient.map(|derivative| derivative * factor),
        }
    }

    /// The quotient by `divisor`, a number that the variables do not move: the value
    /// divided, and the derivatives multiplied by the divisor's reciprocal, so that one
    /// division serves them all.
    #[must_use]
    pub fn divide(self, divisor: R) -> Self {
        let reciprocal = divisor.recip();
        Self {
            value: self.value / divisor,
            gradient: self.gradient.map(|derivative| derivative * reciprocal),
        }
    }

    /// The sum with `term`, a number that the variables do not move.
    #[must_use]
    pub fn plus(self, term: R) -> Self {
        Self {
            value: self.value + term,
            ..self
        }
    }

    /// The value moved by the derivatives along `step`: how a value evaluated at limited
    /// variables extends linearly to the applied ones.
    #[must_use]
    pub fn extend(&self, step: [f64; N]) -> R {
        self.value
            + self
                .gradient
                .iter()
                .zip(step)
                .fold(R::constant(-0.0), |sum, (&derivative, step)| {
                    sum + derivative * step
                })
    }

    /// The same number among `M` variables, each of its own `N` at its position among
    /// them in `positions`, its derivative with respect to every other zero: how a law of
    /// a few of a model's variables joins a law of more, so that each computes at the
    /// width of the variables it reads. A position beyond the `M` variables is dropped.
    #[must_use]
    pub fn embed<const M: usize>(self, positions: [usize; N]) -> Dual<R, M> {
        let mut gradient = [R::constant(0.0); M];
        for (position, derivative) in positions.into_iter().zip(self.gradient) {
            if let Some(slot) = gradient.get_mut(position) {
                *slot = derivative;
            }
        }
        Dual {
            value: self.value,
            gradient,
        }
    }
}

impl<R: Real> Dual<R, 1> {
    /// The one variable at `value`, the derivative of itself.
    #[must_use]
    pub fn variable(value: f64) -> Self {
        Self {
            value: R::constant(value),
            gradient: [R::constant(1.0)],
        }
    }
}

impl<R: Real, const N: usize> Add for Dual<R, N> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut gradient = self.gradient;
        for (derivative, other) in gradient.iter_mut().zip(other.gradient) {
            *derivative += other;
        }
        Self {
            value: self.value + other.value,
            gradient,
        }
    }
}

impl<R: Real, const N: usize> Sub for Dual<R, N> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + -other
    }
}

impl<R: Real, const N: usize> Neg for Dual<R, N> {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            value: -self.value,
            gradient: self.gradient.map(|derivative| -derivative),
        }
    }
}

impl<R: Real, const N: usize> Mul for Dual<R, N> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let mut gradient = [R::constant(0.0); N];
        let pairs = self.gradient.into_iter().zip(other.gradient);
        for (derivative, (own, others)) in gradient.iter_mut().zip(pairs) {
            *derivative = own * other.value + self.value * others;
        }
        Self {
            value: self.value * other.value,
            gradient,
        }
    }
}

impl<R: Real, const N: usize> Div for Dual<R, N> {
    type Output = Self;

    /// The quotient `a / b`, with the derivatives `(a′ − (a / b) · b′) / b`, each
    /// multiplied by the reciprocal of `b`, so that one division serves them all.
    fn div(self, other: Self) -> Self {
        let value = self.value / other.value;
        let reciprocal = other.value.recip();
        let mut gradient = self.gradient;
        for (derivative, others) in gradient.iter_mut().zip(other.gradient) {
            *derivative = (*derivative - value * others) * reciprocal;
        }
        Self { value, gradient }
    }
}

impl<R: Real, const N: usize> Add<f64> for Dual<R, N> {
    type Output = Self;

    fn add(self, other: f64) -> Self {
        Self {
            value: self.value + other,
            ..self
        }
    }
}

impl<R: Real, const N: usize> Sub<f64> for Dual<R, N> {
    type Output = Self;

    fn sub(self, other: f64) -> Self {
        self + -other
    }
}

impl<R: Real, const N: usize> Mul<f64> for Dual<R, N> {
    type Output = Self;

    fn mul(self, other: f64) -> Self {
        Self {
            value: self.value * other,
            gradient: self.gradient.map(|derivative| derivative * other),
        }
    }
}

impl<R: Real, const N: usize> Div<f64> for Dual<R, N> {
    type Output = Self;

    #[expect(
        clippy::suspicious_arithmetic_impl,
        reason = "the derivatives multiply by the divisor's reciprocal, one division for all"
    )]
    fn div(self, other: f64) -> Self {
        let reciprocal = other.recip();
        Self {
            value: self.value / other,
            gradient: self.gradient.map(|derivative| derivative * reciprocal),
        }
    }
}

impl<R: Real, const N: usize> Mul<Dual<R, N>> for f64 {
    type Output = Dual<R, N>;

    fn mul(self, other: Dual<R, N>) -> Dual<R, N> {
        other * self
    }
}

impl<R: Real, const N: usize> Sub<Dual<R, N>> for f64 {
    type Output = Dual<R, N>;

    fn sub(self, other: Dual<R, N>) -> Dual<R, N> {
        -other + self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivatives_follow_the_rules_of_calculus() {
        let [x, y] = Dual::<f64, 2>::variables([3.0, 2.0]);
        // f = x² y / (1 + y) − √x
        let f = x * x * y / (y + 1.0) - x.sqrt();
        let expected_x = 2.0 * 3.0 * 2.0 / 3.0 - 0.5 / 3.0_f64.sqrt();
        let expected_y = 9.0 / 9.0;
        assert!((f.value - (6.0 - 3.0_f64.sqrt())).abs() < 1e-15);
        assert!((f.gradient[0] - expected_x).abs() < 1e-15);
        assert!((f.gradient[1] - expected_y).abs() < 1e-15);
        assert!((f.extend([1e-3, -2e-3]) - (f.value + 1e-3 * expected_x - 2e-3)).abs() < 1e-15);
        let floored = (2.0 - x).max(0.0);
        assert!(floored.value == 0.0 && floored.gradient.iter().all(|slope| *slope == 0.0));
        let scaled = x.scale(2.0);
        assert!((scaled.value - 6.0).abs() < 1e-15 && (scaled.gradient[0] - 2.0).abs() < 1e-15);
    }

    #[test]
    fn a_number_of_a_few_variables_embeds_among_more() {
        let [x, y] = Dual::<f64, 2>::variables([3.0, 2.0]);
        let embedded: Dual<f64, 3> = (x * y).embed([2, 0]);
        assert_eq!(embedded.value, 6.0);
        assert_eq!(embedded.gradient, [3.0, 0.0, 2.0]);
        let alone = Dual::<f64, 1>::variable(4.0);
        assert_eq!(alone.embed::<2>([1]).gradient, [0.0, 1.0]);
        assert_eq!(alone.embed::<2>([5]).gradient, [0.0, 0.0]);
    }
}
