//! Forward-mode automatic differentiation: a value carried with its gradient, so a model
//! writes its equations once and gets their exact derivatives with them.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// A value and its derivatives with respect to `N` variables.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Dual<const N: usize> {
    pub(crate) value: f64,
    pub(crate) gradient: [f64; N],
}

impl<const N: usize> Dual<N> {
    /// A constant, whose derivatives are all zero.
    pub(crate) const fn constant(value: f64) -> Self {
        Self {
            value,
            gradient: [0.0; N],
        }
    }

    /// The `N` variables at `values`, each the derivative of itself only.
    pub(crate) fn variables(values: [f64; N]) -> [Self; N] {
        let mut index = 0;
        values.map(|value| {
            let mut gradient = [0.0; N];
            gradient[index] = 1.0;
            index += 1;
            Self { value, gradient }
        })
    }

    /// `f(self)`, given `f`'s value and derivative at `self`.
    pub(crate) fn chain(self, (value, slope): (f64, f64)) -> Self {
        Self {
            value,
            gradient: self.gradient.map(|derivative| slope * derivative),
        }
    }

    /// The square root.
    pub(crate) fn sqrt(self) -> Self {
        let root = self.value.sqrt();
        self.chain((root, 0.5 / root))
    }

    /// The larger of `self` and the constant `floor`: the floor itself, with no
    /// derivatives, where `self` falls below it.
    pub(crate) fn max(self, floor: f64) -> Self {
        if self.value < floor {
            Self::constant(floor)
        } else {
            self
        }
    }

    /// The value moved by the derivatives along `step`: how a value evaluated at limited
    /// variables extends linearly to the applied ones.
    pub(crate) fn extend(&self, step: [f64; N]) -> f64 {
        self.value
            + self
                .gradient
                .iter()
                .zip(step)
                .map(|(derivative, step)| derivative * step)
                .sum::<f64>()
    }
}

impl<const N: usize> Add for Dual<N> {
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

impl<const N: usize> Sub for Dual<N> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + -other
    }
}

impl<const N: usize> Neg for Dual<N> {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            value: -self.value,
            gradient: self.gradient.map(|derivative| -derivative),
        }
    }
}

impl<const N: usize> Mul for Dual<N> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let mut gradient = [0.0; N];
        for (index, derivative) in gradient.iter_mut().enumerate() {
            *derivative = self.gradient[index] * other.value + self.value * other.gradient[index];
        }
        Self {
            value: self.value * other.value,
            gradient,
        }
    }
}

impl<const N: usize> Div for Dual<N> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let mut gradient = [0.0; N];
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

impl<const N: usize> Add<f64> for Dual<N> {
    type Output = Self;

    fn add(self, other: f64) -> Self {
        Self {
            value: self.value + other,
            ..self
        }
    }
}

impl<const N: usize> Sub<f64> for Dual<N> {
    type Output = Self;

    fn sub(self, other: f64) -> Self {
        self + -other
    }
}

impl<const N: usize> Mul<f64> for Dual<N> {
    type Output = Self;

    fn mul(self, other: f64) -> Self {
        Self {
            value: self.value * other,
            gradient: self.gradient.map(|derivative| derivative * other),
        }
    }
}

impl<const N: usize> Div<f64> for Dual<N> {
    type Output = Self;

    fn div(self, other: f64) -> Self {
        Self {
            value: self.value / other,
            gradient: self.gradient.map(|derivative| derivative / other),
        }
    }
}

impl<const N: usize> Mul<Dual<N>> for f64 {
    type Output = Dual<N>;

    fn mul(self, other: Dual<N>) -> Dual<N> {
        other * self
    }
}

impl<const N: usize> Sub<Dual<N>> for f64 {
    type Output = Dual<N>;

    fn sub(self, other: Dual<N>) -> Dual<N> {
        -other + self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivatives_follow_the_rules_of_calculus() {
        let [x, y] = Dual::variables([3.0, 2.0]);
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
    }
}
