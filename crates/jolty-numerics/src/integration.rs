//! Integration formulas: how a time step turns charges into currents.
//!
//! The reactive part of a circuit is a vector of charges and fluxes `q(x)` whose time
//! derivative enters its equations. Over a step from `t_n` to `t_{n+1} = t_n + h`, a formula
//! approximates the derivative at the new point from charges at the new and earlier points:
//!
//! ```text
//! dq/dt(t_{n+1}) ≈ α q_{n+1} + β₀ q_n + β₁ q_{n-1} + γ q̇_n
//! ```
//!
//! The Jacobian of a step's Newton system is therefore `G + α C`, where `G` and `C` are the
//! derivatives of the resistive and reactive parts.

/// An integration formula.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    /// The first-order backward Euler formula: stable and damped, used to start and to restart
    /// after a discontinuity.
    BackwardEuler,
    /// The second-order trapezoidal formula, the default: the most accurate of the three, and
    /// undamped.
    #[default]
    Trapezoidal,
    /// The second-order backward differentiation formula: damped, for stiff circuits whose
    /// trapezoidal solution rings.
    Gear2,
}

impl Method {
    /// The order of accuracy `p`: the error of one step is proportional to `h^{p+1}`.
    #[must_use]
    pub const fn order(self) -> usize {
        match self {
            Self::BackwardEuler => 1,
            Self::Trapezoidal | Self::Gear2 => 2,
        }
    }

    /// The local truncation error of a step of length `step`, estimated from the divided
    /// difference of order `p + 1` of the charge over the step's last `p + 2` points.
    ///
    /// The error of a step is `C h^{p+1} q^{(p+1)}`, and the derivative is estimated as
    /// `q^{(p+1)} ≈ (p+1)! · q[t_{n-p}, …, t_{n+1}]`, where `C` is ½ for backward Euler, 1/12
    /// for the trapezoidal formula and 2/9 for Gear2.
    #[must_use]
    pub fn truncation_error(self, step: f64, divided_difference: f64) -> f64 {
        let (constant_times_factorial, power) = match self {
            Self::BackwardEuler => (1.0 / 2.0 * 2.0, step * step),
            Self::Trapezoidal => (1.0 / 12.0 * 6.0, step * step * step),
            Self::Gear2 => (2.0 / 9.0 * 6.0, step * step * step),
        };
        constant_times_factorial * power * divided_difference.abs()
    }

    /// The factor that scales a step whose error was `error_ratio` times its tolerance to one
    /// whose error would just meet it: `error_ratio^{−1/(p+1)}`, as the error grows with
    /// `h^{p+1}`.
    #[must_use]
    pub fn step_factor(self, error_ratio: f64) -> f64 {
        match self {
            Self::BackwardEuler => error_ratio.sqrt().recip(),
            Self::Trapezoidal | Self::Gear2 => error_ratio.cbrt().recip(),
        }
    }
}

/// The coefficients of one step: `dq/dt(t_{n+1}) ≈ α q_{n+1} + β₀ q_n + β₁ q_{n-1} + γ q̇_n`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Formula {
    /// The coefficient of the charge at the new point.
    pub alpha: f64,
    /// The coefficients of the charges at the two previous points.
    pub beta: [f64; 2],
    /// The coefficient of the derivative at the previous point.
    pub gamma: f64,
}

impl Formula {
    /// The coefficients of `method` for a step of length `step` that follows a step of length
    /// `previous_step`, which only [`Method::Gear2`] reads.
    ///
    /// # Panics
    ///
    /// Panics if `step` is not positive, or if the method is [`Method::Gear2`] and
    /// `previous_step` is not positive.
    #[must_use]
    pub fn new(method: Method, step: f64, previous_step: f64) -> Self {
        assert!(step > 0.0, "a step has a positive length, not {step}");
        match method {
            Method::BackwardEuler => Self {
                alpha: 1.0 / step,
                beta: [-1.0 / step, 0.0],
                gamma: 0.0,
            },
            Method::Trapezoidal => Self {
                alpha: 2.0 / step,
                beta: [-2.0 / step, 0.0],
                gamma: -1.0,
            },
            Method::Gear2 => {
                assert!(
                    previous_step > 0.0,
                    "Gear2 reads a previous step of positive length"
                );
                let ratio = step / previous_step;
                Self {
                    alpha: (1.0 + 2.0 * ratio) / (step * (1.0 + ratio)),
                    beta: [
                        -(1.0 + ratio) / step,
                        ratio * ratio / (step * (1.0 + ratio)),
                    ],
                    gamma: 0.0,
                }
            }
        }
    }

    /// The part of the derivative that earlier points contribute: `β₀ q_n + β₁ q_{n-1} + γ q̇_n`.
    #[must_use]
    pub fn history(&self, previous: [f64; 2], previous_derivative: f64) -> f64 {
        self.beta[0] * previous[0] + self.beta[1] * previous[1] + self.gamma * previous_derivative
    }
}

/// The divided difference `f[t_0, …, t_k]` of `values` sampled at the distinct `times`.
///
/// For a polynomial of degree `k`, it is the leading coefficient; in general, it approximates
/// the `k`-th derivative divided by `k!`, which is how truncation errors are estimated.
///
/// # Panics
///
/// Panics if `times` and `values` differ in length or are empty.
#[must_use]
pub fn divided_difference(times: &[f64], values: &[f64]) -> f64 {
    assert_eq!(times.len(), values.len(), "one value per time");
    assert!(
        !values.is_empty(),
        "a divided difference needs at least one point"
    );
    let mut table = values.to_vec();
    for level in 1..table.len() {
        for index in 0..table.len() - level {
            table[index] =
                (table[index + 1] - table[index]) / (times[index + level] - times[index]);
        }
    }
    table[0]
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    /// A cubic polynomial and its derivative.
    fn cubic(c: [f64; 4], t: f64) -> (f64, f64) {
        let value = c[0] + t * (c[1] + t * (c[2] + t * c[3]));
        let derivative = c[1] + t * (2.0 * c[2] + t * 3.0 * c[3]);
        (value, derivative)
    }

    proptest! {
        /// Each formula is exact on polynomials up to its order, whatever the step lengths.
        #[test]
        fn formulas_are_exact_up_to_their_order(
            c in proptest::array::uniform4(-5.0..5.0f64),
            step in 0.01..1.0f64,
            previous_step in 0.01..1.0f64,
        ) {
            for method in [Method::BackwardEuler, Method::Trapezoidal, Method::Gear2] {
                let mut coefficients = c;
                coefficients[method.order() + 1..].fill(0.0);
                let q = |t| cubic(coefficients, t);
                let (t1, t0) = (previous_step + step, previous_step);
                let formula = Formula::new(method, step, previous_step);
                let history = formula.history([q(t0).0, q(0.0).0], q(t0).1);
                let estimate = formula.alpha * q(t1).0 + history;
                let exact = q(t1).1;
                prop_assert!((estimate - exact).abs() < 1e-8, "{method:?}: {estimate} vs {exact}");
            }
        }

        /// The divided difference of a polynomial of degree k over k + 1 points is its leading
        /// coefficient.
        #[test]
        fn divided_differences_recover_the_leading_coefficient(
            c in proptest::array::uniform4(-5.0..5.0f64),
            gaps in proptest::array::uniform3(0.1..1.0f64),
        ) {
            let times = [0.0, gaps[0], gaps[0] + gaps[1], gaps[0] + gaps[1] + gaps[2]];
            let values = times.map(|t| cubic(c, t).0);
            prop_assert!((divided_difference(&times, &values) - c[3]).abs() < 1e-8);
        }
    }

    /// The estimate of a step's error matches the error the formula actually makes on a cubic,
    /// to leading order in the step.
    #[test]
    fn truncation_errors_predict_the_actual_error() {
        let q = |t: f64| t * t * t;
        let derivative = |t: f64| 3.0 * t * t;
        for method in [Method::BackwardEuler, Method::Trapezoidal, Method::Gear2] {
            let step = 1e-3;
            let times = [1.0 - 2.0 * step, 1.0 - step, 1.0, 1.0 + step];
            let formula = Formula::new(method, step, step);
            let estimate = formula.alpha * q(times[3])
                + formula.history([q(times[2]), q(times[1])], derivative(times[2]));
            // An error e in the derivative becomes an error e / α in the solved charge.
            let actual = (estimate - derivative(times[3])).abs() / formula.alpha;
            let start = times.len() - (method.order() + 2);
            let values: Vec<f64> = times[start..].iter().map(|&t| q(t)).collect();
            let dd = divided_difference(&times[start..], &values);
            let predicted = method.truncation_error(step, dd);
            assert!(
                (actual / predicted - 1.0).abs() < 0.05,
                "{method:?}: {actual} vs {predicted}"
            );
        }
    }

    #[test]
    fn gear2_at_a_constant_step_is_the_textbook_formula() {
        let formula = Formula::new(Method::Gear2, 0.5, 0.5);
        assert!((formula.alpha - 3.0).abs() < 1e-15);
        assert!((formula.beta[0] + 4.0).abs() < 1e-15);
        assert!((formula.beta[1] - 1.0).abs() < 1e-15);
    }
}
