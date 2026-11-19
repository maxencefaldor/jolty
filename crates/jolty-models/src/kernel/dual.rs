//! Forward-mode automatic differentiation: a value carried with its gradient, so a model
//! writes its equations once and gets their exact derivatives with them.
//!
//! The value and the derivatives are any [`Real`], so a model that differentiates with
//! respect to its voltages, for its Jacobian, also differentiates each of them with respect
//! to a parameter when it computes with [`crate::Gradient`].

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::Real;

/// Scalar value paired with its first derivatives with respect to `N` independent variables.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual<R: Real, const N: usize> {
    /// Primal value.
    pub value: R,
    /// Partial derivatives with respect to each variable.
    pub gradient: [R; N],
}

impl<R: Real, const N: usize> Dual<R, N> {
    /// Creates a constant dual number with zero gradient.
    #[must_use]
    pub fn constant(value: R) -> Self {
        Self {
            value,
            gradient: [R::constant(0.0); N],
        }
    }

    /// Creates a constant dual number from plain `f64` scalar `value`.
    #[must_use]
    pub fn number(value: f64) -> Self {
        Self::constant(R::constant(value))
    }

    /// Constructs `N` independent dual variables initialized from `values`, each with unit
    /// self-derivative and propagating any underlying unknown seeds in `R`.
    #[must_use]
    pub fn variables(values: [R; N]) -> [Self; N] {
        let mut variables = values.map(Self::constant);
        for (index, variable) in variables.iter_mut().enumerate() {
            if let Some(derivative) = variable.gradient.get_mut(index) {
                *derivative = R::constant(1.0);
            }
        }
        variables
    }

    /// Applies the chain rule to `self` given output `value` and scalar derivative `slope`.
    #[must_use]
    pub fn chain(self, (value, slope): (R, R)) -> Self {
        Self {
            value,
            gradient: self.gradient.map(|derivative| slope * derivative),
        }
    }

    /// Evaluates univariate `function` on a width-1 dual number and chains its slope into all
    /// `N` gradient lanes of `self`.
    pub(crate) fn apply(self, function: impl FnOnce(Dual<R, 1>) -> Dual<R, 1>) -> Self {
        let Dual {
            value,
            gradient: [slope],
        } = function(Dual {
            value: self.value,
            gradient: [R::constant(1.0)],
        });
        self.chain((value, slope))
    }

    /// Computes `e^self`.
    #[must_use]
    pub fn exp(self) -> Self {
        let exponential = self.value.exp();
        self.chain((exponential, exponential))
    }

    /// Computes the natural logarithm `ln(self)`.
    pub(crate) fn ln(self) -> Self {
        self.chain((self.value.ln(), self.value.recip()))
    }

    /// Computes `ln(1 + self)` accurately near zero using [`Real::ln_1p`].
    pub(crate) fn ln_1p(self) -> Self {
        self.chain((self.value.ln_1p(), (self.value + 1.0).recip()))
    }

    /// Computes `e^self − 1` accurately near zero using [`Real::exp_m1`].
    pub(crate) fn exp_m1(self) -> Self {
        self.chain((self.value.exp_m1(), self.value.exp()))
    }

    /// Raises positive `self` to a variable-independent `exponent`, reusing `value^exponent`
    /// to compute the derivative `exponent · power / value` with one power call.
    pub(crate) fn pow(self, exponent: R) -> Self {
        let power = self.value.pow(exponent);
        self.chain((power, exponent * power / self.value))
    }

    /// Computes the reciprocal `1 / self`.
    #[must_use]
    pub fn recip(self) -> Self {
        let reciprocal = self.value.recip();
        self.chain((reciprocal, -(reciprocal * reciprocal)))
    }

    /// Computes the square root `√self`.
    #[must_use]
    pub fn sqrt(self) -> Self {
        let root = self.value.sqrt();
        self.chain((root, R::constant(0.5) / root))
    }

    /// Computes the tangent `tan(self)`.
    pub(crate) fn tan(self) -> Self {
        let tangent = self.value.tan();
        self.chain((tangent, tangent * tangent + 1.0))
    }

    /// Returns the maximum of `self` and constant `floor`, returning constant `floor` with
    /// zero gradient when `self` lies below `floor`.
    #[must_use]
    pub fn max(self, floor: f64) -> Self {
        if self.value.value() < floor {
            Self::number(floor)
        } else {
            self
        }
    }

    /// Scales `self` by variable-independent scalar `factor`.
    #[must_use]
    pub fn scale(self, factor: R) -> Self {
        Self {
            value: self.value * factor,
            gradient: self.gradient.map(|derivative| derivative * factor),
        }
    }

    /// Divides `self` by variable-independent scalar `divisor`, multiplying gradient lanes by
    /// the reciprocal to share a single division.
    #[must_use]
    pub fn divide(self, divisor: R) -> Self {
        let reciprocal = divisor.recip();
        Self {
            value: self.value / divisor,
            gradient: self.gradient.map(|derivative| derivative * reciprocal),
        }
    }

    /// Adds variable-independent scalar `term` to `self` without changing the gradient.
    #[must_use]
    pub fn plus(self, term: R) -> Self {
        Self {
            value: self.value + term,
            ..self
        }
    }

    /// Extends `self.value` linearly along `step` using the gradient (used to project
    /// voltage-limited evaluations back to applied voltages).
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

    /// Embeds an `N`-variable dual number into an `M`-variable dual number by placing each
    /// derivative at the target lane in `positions` (ignoring out-of-range indices).
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
    /// Creates a single independent dual variable at `value` with unit derivative.
    #[must_use]
    pub fn variable(value: R) -> Self {
        Self {
            value,
            gradient: [R::constant(1.0)],
        }
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

/// Implements binary operators on [`Dual`] returning `Dual<R, N>`.
macro_rules! dual_ops {
    ($(
        $trait:ident :: $method:ident ($self:ident : $lhs:ty, $other:ident : $rhs:ty) $body:block
    )*) => {$(
        impl<R: Real, const N: usize> $trait<$rhs> for $lhs {
            type Output = Dual<R, N>;

            fn $method($self, $other: $rhs) -> Dual<R, N> $body
        }
    )*};
}

dual_ops! {
    Add::add(self: Dual<R, N>, other: Dual<R, N>) {
        let mut gradient = self.gradient;
        for (derivative, other) in gradient.iter_mut().zip(other.gradient) {
            *derivative += other;
        }
        Self {
            value: self.value + other.value,
            gradient,
        }
    }

    Sub::sub(self: Dual<R, N>, other: Dual<R, N>) {
        self + -other
    }

    Mul::mul(self: Dual<R, N>, other: Dual<R, N>) {
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

    // The quotient `a / b`, with the derivatives `(a′ − (a / b) · b′) / b`, each
    // multiplied by the reciprocal of `b`, so that one division serves them all.
    Div::div(self: Dual<R, N>, other: Dual<R, N>) {
        let value = self.value / other.value;
        let reciprocal = other.value.recip();
        let mut gradient = self.gradient;
        for (derivative, others) in gradient.iter_mut().zip(other.gradient) {
            *derivative = (*derivative - value * others) * reciprocal;
        }
        Self { value, gradient }
    }

    Add::add(self: Dual<R, N>, other: f64) {
        Self {
            value: self.value + other,
            ..self
        }
    }

    Sub::sub(self: Dual<R, N>, other: f64) {
        self + -other
    }

    Mul::mul(self: Dual<R, N>, other: f64) {
        Self {
            value: self.value * other,
            gradient: self.gradient.map(|derivative| derivative * other),
        }
    }

    Mul::mul(self: f64, other: Dual<R, N>) {
        other * self
    }

    Sub::sub(self: f64, other: Dual<R, N>) {
        -other + self
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
        let tangent = (x * y).tan();
        let secant_squared = 1.0 / 6.0_f64.cos().powi(2);
        assert!((tangent.value - 6.0_f64.tan()).abs() < 1e-15);
        assert!((tangent.gradient[0] - 2.0 * secant_squared).abs() < 1e-14);
        assert!((tangent.gradient[1] - 3.0 * secant_squared).abs() < 1e-14);
        // Applying a univariate function at width 1 matches direct composition.
        let product = x * y;
        assert_eq!(
            product.apply(|value| value * 2.0 + 1.0),
            product * 2.0 + 1.0
        );
        assert_eq!(product.apply(Dual::tan), tangent);
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
