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
pub(crate) struct Dual<R: Real, const N: usize> {
    pub(crate) value: R,
    pub(crate) gradient: [R; N],
}

impl<R: Real, const N: usize> Dual<R, N> {
    /// A constant, whose derivatives are all zero.
    pub(crate) fn constant(value: R) -> Self {
        Self {
            value,
            gradient: [R::constant(0.0); N],
        }
    }

    /// The constant `value`, a plain number.
    pub(crate) fn number(value: f64) -> Self {
        Self::constant(R::constant(value))
    }

    /// The `N` variables at `values`, each the derivative of itself only.
    pub(crate) fn variables(values: [f64; N]) -> [Self; N] {
        let mut index = 0;
        values.map(|value| {
            let mut gradient = [R::constant(0.0); N];
            gradient[index] = R::constant(1.0);
            index += 1;
            Self {
                value: R::constant(value),
                gradient,
            }
        })
    }

    /// `f(self)`, given `f`'s value and derivative at `self`.
    pub(crate) fn chain(self, (value, slope): (R, R)) -> Self {
        Self {
            value,
            gradient: self.gradient.map(|derivative| slope * derivative),
        }
    }

    /// `e` raised to the value.
    pub(crate) fn exp(self) -> Self {
        let exponential = self.value.exp();
        self.chain((exponential, exponential))
    }

    /// The reciprocal.
    pub(crate) fn recip(self) -> Self {
        self.chain((self.value.recip(), -(self.value * self.value).recip()))
    }

    /// The square root.
    pub(crate) fn sqrt(self) -> Self {
        let root = self.value.sqrt();
        self.chain((root, R::constant(0.5) / root))
    }

    /// The larger of `self` and the constant `floor`: the floor itself, with no
    /// derivatives, where `self` falls below it.
    pub(crate) fn max(self, floor: f64) -> Self {
        if self.value.value() < floor {
            Self::number(floor)
        } else {
            self
        }
    }

    /// The product with `factor`, a number that the variables do not move.
    pub(crate) fn scale(self, factor: R) -> Self {
        Self {
            value: self.value * factor,
            gradient: self.gradient.map(|derivative| derivative * factor),
        }
    }

    /// The quotient by `divisor`, a number that the variables do not move.
    pub(crate) fn divide(self, divisor: R) -> Self {
        Self {
            value: self.value / divisor,
            gradient: self.gradient.map(|derivative| derivative / divisor),
        }
    }

    /// The sum with `term`, a number that the variables do not move.
    pub(crate) fn plus(self, term: R) -> Self {
        Self {
            value: self.value + term,
            ..self
        }
    }

    /// The value moved by the derivatives along `step`: how a value evaluated at limited
    /// variables extends linearly to the applied ones.
    pub(crate) fn extend(&self, step: [f64; N]) -> R {
        self.value
            + self
                .gradient
                .iter()
                .zip(step)
                .fold(R::constant(-0.0), |sum, (&derivative, step)| {
                    sum + derivative * step
                })
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
        for (index, derivative) in gradient.iter_mut().enumerate() {
            *derivative = self.gradient[index] * other.value + self.value * other.gradient[index];
        }
        Self {
            value: self.value * other.value,
            gradient,
        }
    }
}

impl<R: Real, const N: usize> Div for Dual<R, N> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let mut gradient = [R::constant(0.0); N];
        for (index, derivative) in gradient.iter_mut().enumerate() {
            *derivative = (self.gradient[index] * other.value - self.value * other.gradient[index])
                / (other.value * other.value);
        }
        Self {
            value: self.value / other.value,
            gradient,
        }
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

    fn div(self, other: f64) -> Self {
        Self {
            value: self.value / other,
            gradient: self.gradient.map(|derivative| derivative / other),
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
}
