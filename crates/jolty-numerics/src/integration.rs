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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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
    /// Returns the order of accuracy `p` where single-step local error scales as `h^{p+1}`.
    #[must_use]
    pub const fn order(self) -> usize {
        match self {
            Self::BackwardEuler => 1,
            Self::Trapezoidal | Self::Gear2 => 2,
        }
    }

    /// Returns whether the formula and error estimator depend on the previous step length (true
    /// for Gear2, false for one-step methods).
    #[must_use]
    pub const fn is_multistep(self) -> bool {
        matches!(self, Self::Gear2)
    }

    /// Estimates local truncation error for `step` following `previous_step` from the
    /// `(p + 1)`-th divided difference of charge over the last `p + 2` time points.
    ///
    /// Local error is `C h^{p+1} q^{(p+1)}`, with `q^{(p+1)} ≈ (p+1)! · q[t_{n-p}, …, t_{n+1}]`.
    /// Error constant `C` equals `1/2` for backward Euler and `1/12` for trapezoidal
    /// regardless of step ratio.
    ///
    /// For Gear2 with step ratio `ω = step / previous_step`, `C = (1 + ω)² / (6 ω (1 + 2ω))`,
    /// combining the quadratic derivative error `h (h + h/ω) q''' / 6` with the formula factor
    /// `h (1 + ω) / (1 + 2ω)`. For Gear2, `C` equals `2/9` at a constant step (SPICE's fixed
    /// constant) and rises to `7.6` times that after a 10x step cut.
    #[must_use]
    pub fn truncation_error(self, step: f64, previous_step: f64, divided_difference: f64) -> f64 {
        self.error_per_difference(step, previous_step) * divided_difference.abs()
    }

    /// Computes the [truncation error](Method::truncation_error) per unit divided difference,
    /// `C (p+1)! h^{p+1}`, shared across all charges in a step.
    #[must_use]
    pub fn error_per_difference(self, step: f64, previous_step: f64) -> f64 {
        let power = step.powi(self.exponent());
        self.constant(step / previous_step) * self.factorial() * power
    }

    /// Computes the maximum step following `previous_step` whose truncation error stays within
    /// `max(charge, rate * step)`.
    ///
    /// Equivalent to [`Method::longest_common_step`] for a single charge; returns infinity
    /// when `divided_difference` is zero.
    #[must_use]
    pub fn longest_step(
        self,
        previous_step: f64,
        divided_difference: f64,
        charge: f64,
        rate: f64,
    ) -> f64 {
        let unit = self.unit_steps(divided_difference, charge, rate);
        let constant_step = longest_at(unit, CONSTANT_STEP);
        self.longest_common_step(previous_step, constant_step, |scales| {
            longest_at(unit, scales)
        })
    }

    /// Computes constant-step candidate bounds `((charge / K)^{1/(p+1)}, (rate / K)^{1/p})`
    /// where `K = C₁ (p+1)! |divided_difference|`.
    ///
    /// Returns `(inf, inf)` when `divided_difference` is zero. Multiplying by
    /// [`Method::scales`] adjusts both bounds for non-unit step ratios.
    #[must_use]
    pub fn unit_steps(self, divided_difference: f64, charge: f64, rate: f64) -> (f64, f64) {
        let coefficient = self.constant(1.0) * self.factorial() * divided_difference.abs();
        if coefficient == 0.0 {
            return (f64::INFINITY, f64::INFINITY);
        }
        let (by_charge, by_current) = self.powers();
        (
            (charge / coefficient).powf(by_charge),
            (rate / coefficient).powf(by_current),
        )
    }

    /// Computes the step-ratio scaling factors `((C/C₁)^{-1/(p+1)}, (C/C₁)^{-1/p})` for
    /// [`Method::unit_steps`] at step ratio `ratio`.
    ///
    /// Both factors equal `1.0` at `ratio = 1.0` and at all ratios for one-step methods.
    #[must_use]
    pub fn scales(self, ratio: f64) -> (f64, f64) {
        let relative = self.constant(ratio) / self.constant(1.0);
        let (by_charge, by_current) = self.powers();
        (relative.powf(-by_charge), relative.powf(-by_current))
    }

    /// Solves for the longest step following `previous_step` that satisfies every charge's
    /// truncation tolerance.
    ///
    /// Takes `constant_step` (the minimum [longest step](longest_at) across charges at
    /// constant-step [scales](Method::scales)) and `shortest` (which recomputes that minimum
    /// at arbitrary scales). One-step methods return `constant_step.0` directly without
    /// calling `shortest`.
    ///
    /// For Gear2, the error constant depends on `u = h / h₀` ([`Method::truncation_error`]),
    /// so the step solves the fixed point `h = Λ(C(h/h₀))`. In `x = ln h`, `ln Λ - x` has
    /// slope between `-1` and `-1/2`, so Newton iteration from `previous_step` contracts
    /// until the update reaches float precision or stops shrinking.
    pub fn longest_common_step(
        self,
        previous_step: f64,
        constant_step: (f64, ErrorBound),
        mut shortest: impl FnMut((f64, f64)) -> (f64, ErrorBound),
    ) -> f64 {
        if !self.is_multistep() {
            return constant_step.0;
        }
        let (by_charge, by_current) = self.powers();
        let mut step = previous_step;
        let mut found = constant_step;
        let mut moved = f64::INFINITY;
        loop {
            let ratio = step / previous_step;
            let (reach, bound) = found;
            if !reach.is_finite() || reach <= 0.0 {
                return reach;
            }
            let power = match bound {
                ErrorBound::Charge => by_charge,
                ErrorBound::Current => by_current,
            };
            // Compute the slope of ln Λ(C(h/h₀)) in ln h and take a Newton step on ln Λ - ln h.
            let slope = -power * gear2_elasticity(ratio);
            let next = step * (reach / step).powf((1.0 - slope).recip());
            let change = (next - step).abs();
            if change.is_nan() || change <= f64::EPSILON * step || change >= moved {
                return next.min(step);
            }
            moved = change;
            step = next;
            found = shortest(self.scales(step / previous_step));
        }
    }

    /// Returns `(p + 1)!`, converting an order-`(p + 1)` divided difference into the
    /// `(p + 1)`-th derivative estimate.
    #[must_use]
    pub const fn factorial(self) -> f64 {
        match self {
            Self::BackwardEuler => 2.0,
            Self::Trapezoidal | Self::Gear2 => 6.0,
        }
    }

    /// Returns the leading truncation-error constant at step ratio `ratio`
    /// ([`Method::truncation_error`]): `1/2` for backward Euler, `1/12` for trapezoidal, and
    /// `(1 + ω)² / (6 ω (1 + 2ω))` for Gear2.
    fn constant(self, ratio: f64) -> f64 {
        match self {
            Self::BackwardEuler => 1.0 / 2.0,
            Self::Trapezoidal => 1.0 / 12.0,
            Self::Gear2 => gear2_constant(ratio),
        }
    }

    /// Returns the reciprocal error-constant exponents `(1 / (p + 1), 1 / p)` for charge and
    /// current tolerances.
    fn powers(self) -> (f64, f64) {
        let order = f64::from(self.exponent() - 1);
        ((order + 1.0).recip(), order.recip())
    }

    /// Returns the step-size exponent `p + 1` in the single-step truncation error.
    fn exponent(self) -> i32 {
        match self {
            Self::BackwardEuler => 2,
            Self::Trapezoidal | Self::Gear2 => 3,
        }
    }
}

/// Computes Gear2's truncation-error constant `(1 + ω)² / (6 ω (1 + 2ω))` at step ratio
/// `ratio`, rescaling by `1 / ω` when `ω > 1` to avoid overflow.
///
/// Ranges from `1/12` as `ω → ∞`, through `2/9` at `ω = 1`, to `1 / (6ω)` as `ω → 0`.
fn gear2_constant(ratio: f64) -> f64 {
    if ratio > 1.0 {
        let inverse = ratio.recip();
        let factor = 1.0 + inverse;
        (factor * factor) / (6.0 * (2.0 + inverse))
    } else {
        let factor = 1.0 + ratio;
        (factor * factor) / (6.0 * ratio * (1.0 + 2.0 * ratio))
    }
}

/// Computes the logarithmic derivative `d ln C / d ln u = -(1 + 3u) / ((1 + u)(1 + 2u))` of
/// Gear2's error constant, rescaling by `1 / u` when `u > 1` to avoid cancellation and
/// overflow.
///
/// Ranges from `-1` as `u → 0`, through `-2/3` at `u = 1`, to `0` as `u → ∞`.
fn gear2_elasticity(ratio: f64) -> f64 {
    if ratio > 1.0 {
        let inverse = ratio.recip();
        -(inverse * (3.0 + inverse)) / ((1.0 + inverse) * (2.0 + inverse))
    } else {
        -(1.0 + 3.0 * ratio) / ((1.0 + ratio) * (1.0 + 2.0 * ratio))
    }
}

/// Active tolerance governing a charge's maximum allowable time step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorBound {
    /// Absolute charge error tolerance.
    Charge,
    /// Current error tolerance, which scales linearly with step length.
    Current,
}

/// Constant-step [scales](Method::scales) `(1.0, 1.0)` for a unit step ratio (`C = C₁`).
pub const CONSTANT_STEP: (f64, f64) = (1.0, 1.0);

/// Returns the larger of the scaled charge and current [unit steps](Method::unit_steps) under
/// [scales](Method::scales), together with the governing [`ErrorBound`].
#[must_use]
pub fn longest_at((by_charge, by_current): (f64, f64), scales: (f64, f64)) -> (f64, ErrorBound) {
    let (charge, current) = (by_charge * scales.0, by_current * scales.1);
    if charge >= current {
        (charge, ErrorBound::Charge)
    } else {
        (current, ErrorBound::Current)
    }
}

/// Time-step integration coefficients in `dq/dt(t_{n+1}) ≈ α q_{n+1} + β₀ q_n + β₁ q_{n-1} + γ q̇_n`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Formula {
    /// Coefficient multiplying the new charge `q_{n+1}`.
    pub alpha: f64,
    /// Coefficients multiplying the two prior charges `[q_n, q_{n-1}]`.
    pub beta: [f64; 2],
    /// Coefficient multiplying the prior time derivative `q̇_n`.
    pub gamma: f64,
}

impl Formula {
    /// Builds integration coefficients for `method` over `step` following `previous_step`
    /// (used only by [`Method::Gear2`]).
    ///
    /// # Panics
    ///
    /// Panics if `step` is not positive, or if `method` is [`Method::Gear2`] and
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

    /// Evaluates the history contribution `β₀ q_n + β₁ q_{n-1} + γ q̇_n` to the time
    /// derivative.
    ///
    /// Zero coefficients skip reading their corresponding entries so restarts do not propagate
    /// `0.0 * NaN` or `0.0 * inf` from unused history slots.
    #[must_use]
    pub fn history(&self, previous: [f64; 2], previous_derivative: f64) -> f64 {
        let mut total = self.beta[0] * previous[0];
        if self.beta[1] != 0.0 {
            total += self.beta[1] * previous[1];
        }
        if self.gamma != 0.0 {
            total += self.gamma * previous_derivative;
        }
        total
    }

    /// Computes the discrete charge-leak increment
    /// `Δℓ_n = (residual - γ · previous_residual + β₁ · previous_increment) / α`.
    ///
    /// The identity `α + β₀ + β₁ = 0` rearranges the step formula
    /// `q̇_n = α q_n + β₀ q_{n-1} + β₁ q_{n-2} + γ q̇_{n-1}` into
    /// `α Δq_n = q̇_n - γ q̇_{n-1} + β₁ Δq_{n-1}`. Substituting `q̇_n = r_n - f_n` splits
    /// each step's charge change linearly into the discrete integral of `-f` plus the integral
    /// of the residual `r`:
    /// - backward Euler (`β₁ = 0, γ = 0`): `h · r_n`,
    /// - trapezoidal (`β₁ = 0, γ = -1`): `½ h · (r_n + r_{n-1})`,
    /// - Gear2 (`γ = 0`): `h · (ω r̄_{n-1} + (1 + ω) r_n) / (1 + 2ω)`, a convex combination
    ///   of the previous step's mean leak rate `r̄_{n-1} = Δℓ_{n-1} / h_{n-1}` and `r_n`.
    ///
    /// Zero coefficients skip reading earlier steps, so backward Euler and post-restart
    /// steps do not depend on `previous_residual` or `previous_increment`.
    #[must_use]
    pub fn integral_increment(
        &self,
        residual: f64,
        previous_residual: f64,
        previous_increment: f64,
    ) -> f64 {
        let mut rate = residual;
        if self.gamma != 0.0 {
            rate -= self.gamma * previous_residual;
        }
        if self.beta[1] != 0.0 {
            rate += self.beta[1] * previous_increment;
        }
        rate / self.alpha
    }
}

/// Computes the Newton divided difference `f[t_0, …, t_k]` of `values` at distinct `times` in
/// place without allocation, overwriting `values`.
///
/// Recovers the leading coefficient for degree-`k` polynomials and approximates `f^{(k)} / k!`
/// for truncation-error estimation.
///
/// # Panics
///
/// Panics if `times` and `values` differ in length or are empty.
#[must_use]
pub fn divided_difference(times: &[f64], values: &mut [f64]) -> f64 {
    assert_eq!(times.len(), values.len(), "one value per time");
    assert!(
        !values.is_empty(),
        "a divided difference needs at least one point"
    );
    for level in 1..values.len() {
        for index in 0..values.len() - level {
            values[index] =
                (values[index + 1] - values[index]) / (times[index + level] - times[index]);
        }
    }
    values[0]
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    /// Evaluates a cubic polynomial and its first derivative at `t`.
    fn cubic(c: [f64; 4], t: f64) -> (f64, f64) {
        let value = c[0] + t * (c[1] + t * (c[2] + t * c[3]));
        let derivative = c[1] + t * (2.0 * c[2] + t * 3.0 * c[3]);
        (value, derivative)
    }

    proptest! {
        /// Verifies that every integration formula is exact on polynomials up to its order
        /// across arbitrary step lengths.
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

        /// Verifies that the divided difference of a degree-`k` polynomial over `k + 1`
        /// points recovers its leading coefficient.
        #[test]
        fn divided_differences_recover_the_leading_coefficient(
            c in proptest::array::uniform4(-5.0..5.0f64),
            gaps in proptest::array::uniform3(0.1..1.0f64),
        ) {
            let times = [0.0, gaps[0], gaps[0] + gaps[1], gaps[0] + gaps[1] + gaps[2]];
            let mut values = times.map(|t| cubic(c, t).0);
            prop_assert!((divided_difference(&times, &mut values) - c[3]).abs() < 1e-8);
        }
    }

    /// Verifies that estimated truncation error matches actual cubic-polynomial error across
    /// step ratios from 0.1 to 10 (including Gear2's 7.6x growth on a 10x step cut).
    #[test]
    fn truncation_errors_predict_the_actual_error() {
        let q = |t: f64| t * t * t;
        let derivative = |t: f64| 3.0 * t * t;
        for method in [Method::BackwardEuler, Method::Trapezoidal, Method::Gear2] {
            for ratio in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0] {
                let step = 1e-3;
                let previous_step = step / ratio;
                let times = [
                    1.0 - 2.0 * previous_step,
                    1.0 - previous_step,
                    1.0,
                    1.0 + step,
                ];
                let formula = Formula::new(method, step, previous_step);
                let estimate = formula.alpha * q(times[3])
                    + formula.history([q(times[2]), q(times[1])], derivative(times[2]));
                // An error e in the derivative becomes an error e / α in the solved charge.
                let actual = (estimate - derivative(times[3])).abs() / formula.alpha;
                let start = times.len() - (method.order() + 2);
                let mut values: Vec<f64> = times[start..].iter().map(|&t| q(t)).collect();
                let dd = divided_difference(&times[start..], &mut values);
                let predicted = method.truncation_error(step, previous_step, dd);
                assert!(
                    (actual / predicted - 1.0).abs() < 0.05,
                    "{method:?} at a ratio of {ratio}: {actual} vs {predicted}"
                );
            }
        }
    }

    /// Verifies that Gear2's longest step meets its charge or current tolerance across step
    /// ratios spanning 10⁻¹⁰ to 10¹⁰.
    #[test]
    fn gear2s_longest_step_meets_its_tolerance_at_every_scale() {
        let method = Method::Gear2;
        for exponent in (-30..=30).step_by(3) {
            let tolerance = 10_f64.powi(exponent);
            for (charge, rate) in [(tolerance, 0.0), (0.0, tolerance)] {
                let longest = method.longest_step(1.0, 1.0, charge, rate);
                let error = |step: f64| method.truncation_error(step, 1.0, 1.0);
                let allowed = |step: f64| charge.max(rate * step);
                let met = error(longest) / allowed(longest);
                assert!(
                    (met - 1.0).abs() < 1e-12,
                    "{charge:e} and {rate:e}: {longest:e} meets {met}"
                );
                let longer = longest * (1.0 + 1e-9);
                assert!(error(longer) > allowed(longer));
            }
        }
    }

    proptest! {
        /// Verifies that the computed longest step satisfies `max(charge, rate * step)`
        /// tightly and any slightly longer step exceeds it.
        #[test]
        fn the_longest_step_just_meets_its_tolerance(
            divided_difference in 1e-6..1e6f64,
            charge in 1e-15..1e-6f64,
            rate in 1e-12..1e-3f64,
            previous_step in 1e-12..1e-3f64,
        ) {
            for method in [Method::BackwardEuler, Method::Trapezoidal, Method::Gear2] {
                let tolerance = |step: f64| charge.max(rate * step);
                let error =
                    |step: f64| method.truncation_error(step, previous_step, divided_difference);
                let longest = method.longest_step(previous_step, divided_difference, charge, rate);
                let met = error(longest) / tolerance(longest);
                prop_assert!((met - 1.0).abs() < 1e-9, "{method:?}: {met}");
                let longer = longest * (1.0 + 1e-6);
                prop_assert!(error(longer) > tolerance(longer), "{method:?}");
            }
        }
    }

    proptest! {
        /// Verifies that [`Method::longest_common_step`] equals the minimum individual longest
        /// step across multiple charges.
        #[test]
        fn the_common_longest_step_is_the_shortest_of_each_charges(
            charges in proptest::collection::vec(
                (1e-6..1e6f64, 1e-15..1e-6f64, 1e-12..1e-3f64),
                1..6,
            ),
            previous_step in 1e-12..1e-3f64,
        ) {
            for method in [Method::BackwardEuler, Method::Trapezoidal, Method::Gear2] {
                let units: Vec<(f64, f64)> = charges
                    .iter()
                    .map(|&(dd, charge, rate)| method.unit_steps(dd, charge, rate))
                    .collect();
                let shortest = |scales| {
                    units
                        .iter()
                        .map(|&unit| longest_at(unit, scales))
                        .min_by(|a: &(f64, ErrorBound), b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
                        .unwrap_or((f64::INFINITY, ErrorBound::Charge))
                };
                let constant_step = shortest(method.scales(1.0));
                let common = method.longest_common_step(previous_step, constant_step, shortest);
                let each = charges
                    .iter()
                    .map(|&(dd, charge, rate)| method.longest_step(previous_step, dd, charge, rate))
                    .fold(f64::INFINITY, f64::min);
                prop_assert!((common / each - 1.0).abs() < 1e-12, "{method:?}: {common} vs {each}");
            }
        }
    }

    #[test]
    fn a_constant_steps_scales_are_one_for_every_method() {
        for method in [Method::BackwardEuler, Method::Trapezoidal, Method::Gear2] {
            let (by_charge, by_current) = method.scales(1.0);
            assert_eq!(by_charge.to_bits(), CONSTANT_STEP.0.to_bits(), "{method:?}");
            assert_eq!(
                by_current.to_bits(),
                CONSTANT_STEP.1.to_bits(),
                "{method:?}"
            );
        }
    }

    #[test]
    fn a_vanishing_divided_difference_bounds_no_step() {
        for method in [Method::BackwardEuler, Method::Trapezoidal, Method::Gear2] {
            let longest = method.longest_step(1e-9, 0.0, 1e-14, 1e-9);
            assert!(longest.is_infinite(), "{method:?}");
        }
    }

    #[test]
    fn gear2_at_a_constant_step_is_the_textbook_formula() {
        let formula = Formula::new(Method::Gear2, 0.5, 0.5);
        assert!((formula.alpha - 3.0).abs() < 1e-15);
        assert!((formula.beta[0] + 4.0).abs() < 1e-15);
        assert!((formula.beta[1] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn integral_increment_ignores_earlier_steps_on_backward_euler() {
        let formula = Formula::new(Method::BackwardEuler, 2e-6, 1e-6);
        let increment = formula.integral_increment(3e-9, f64::NAN, f64::INFINITY);
        let expected = 3e-9 / formula.alpha;
        assert_eq!(increment.to_bits(), expected.to_bits());
    }

    #[test]
    fn history_skips_unused_non_finite_slots_when_coefficients_are_zero() {
        let euler = Formula::new(Method::BackwardEuler, 2e-6, 1e-6);
        let expected_euler = euler.beta[0] * 3.0;
        assert_eq!(
            euler.history([3.0, f64::NAN], f64::INFINITY).to_bits(),
            expected_euler.to_bits()
        );
        assert_eq!(
            euler.history([3.0, f64::INFINITY], f64::NAN).to_bits(),
            expected_euler.to_bits()
        );

        let trap = Formula::new(Method::Trapezoidal, 2e-6, 1e-6);
        let expected_trap = trap.beta[0] * 3.0 + trap.gamma * 5.0;
        assert_eq!(
            trap.history([3.0, f64::NAN], 5.0).to_bits(),
            expected_trap.to_bits()
        );

        let gear2 = Formula::new(Method::Gear2, 2e-6, 1e-6);
        let expected_gear2 = gear2.beta[0] * 3.0 + gear2.beta[1] * 2.0;
        assert_eq!(
            gear2.history([3.0, 2.0], f64::NAN).to_bits(),
            expected_gear2.to_bits()
        );
        assert_eq!(
            gear2.history([3.0, 2.0], f64::INFINITY).to_bits(),
            expected_gear2.to_bits()
        );
    }

    #[test]
    fn gear2_constant_and_elasticity_are_accurate_and_overflow_safe() {
        for ratio in [0.01_f64, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0] {
            let reference_constant = (1.0 + ratio).powi(2) / (6.0 * ratio * (1.0 + 2.0 * ratio));
            let got_constant = gear2_constant(ratio);
            assert!(
                (got_constant - reference_constant).abs()
                    <= 4.0 * f64::EPSILON * reference_constant,
                "constant mismatch at ratio {ratio}: {got_constant} vs {reference_constant}"
            );

            let reference_elasticity =
                2.0 * ratio / (1.0 + ratio) - 1.0 - 2.0 * ratio / (1.0 + 2.0 * ratio);
            let got_elasticity = gear2_elasticity(ratio);
            assert!(
                (got_elasticity - reference_elasticity).abs() <= 8.0 * f64::EPSILON,
                "elasticity mismatch at ratio {ratio}: {got_elasticity} vs {reference_elasticity}"
            );
        }

        // Exact closed-form checks at large ratios where the subtracted form cancels or overflows.
        let large = 1e8_f64;
        let expected_elasticity = -(3.0 * large + 1.0) / ((large + 1.0) * (2.0 * large + 1.0));
        // Expand -u * (1 + 3u) / ((1 + u)(1 + 2u)) = -1.5 + 1.75 / u + O(1 / u^2).
        let got_large = gear2_elasticity(large);
        let asymptotic = (-1.5 + 1.75e-8) / large;
        assert!(
            (got_large - asymptotic).abs() < 1e-15 * asymptotic.abs(),
            "ratio 1e8 elasticity: {got_large:e} vs {asymptotic:e} (naive {expected_elasticity:e})"
        );

        let huge = 1e200_f64;
        let got_huge_constant = gear2_constant(huge);
        assert!(got_huge_constant.is_finite());
        assert!((got_huge_constant - 1.0 / 12.0).abs() <= f64::EPSILON);

        let got_huge_elasticity = gear2_elasticity(huge);
        let expected_huge_elasticity = -1.5e-200;
        assert!(got_huge_elasticity.is_finite());
        assert!(
            (got_huge_elasticity - expected_huge_elasticity).abs()
                <= 4.0 * f64::EPSILON * expected_huge_elasticity.abs(),
            "ratio 1e200 elasticity: {got_huge_elasticity:e}"
        );

        assert!((gear2_constant(f64::MAX) - 1.0 / 12.0).abs() <= f64::EPSILON);
        assert!(gear2_elasticity(f64::MAX).is_finite() && gear2_elasticity(f64::MAX) < 0.0);
        let tiny_constant = gear2_constant(1e-308);
        assert!(tiny_constant.is_finite() && tiny_constant > 0.0);
    }

    proptest! {
        /// Verifies that summing `integral_increment` over arbitrary step sequences and
        /// restarts equals `q_N - q_0` when static current `f == 0`.
        #[test]
        fn integral_increment_matches_charge_change_for_every_formula(
            initial_charge in -1e-9..1e-9f64,
            exponents in proptest::collection::vec(-10.0..-8.0f64, 4..32),
            targets in proptest::collection::vec(-1e-3..1e-3f64, 4..32),
        ) {
            let count = exponents.len().min(targets.len());
            let steps: Vec<f64> = exponents[..count]
                .iter()
                .map(|&exponent| 10_f64.powf(exponent))
                .collect();
            for mode in 0..4u8 {
                let mut charge = initial_charge;
                let mut previous_charge = initial_charge;
                let mut previous_derivative = 0.0;
                let mut leak = 0.0;
                let mut previous_residual = 0.0;
                let mut previous_increment = 0.0;
                let mut previous_step = steps[0];
                let mut charge_scale = initial_charge.abs();
                let mut ratio_sum = 0.0;
                for (index, (&step, &target)) in steps.iter().zip(&targets[..count]).enumerate() {
                    let method = if index < 2 {
                        Method::BackwardEuler
                    } else {
                        match mode {
                            0 => Method::BackwardEuler,
                            1 => Method::Trapezoidal,
                            2 => Method::Gear2,
                            _ => match index % 7 {
                                0 | 1 => Method::BackwardEuler,
                                2 | 4 => Method::Trapezoidal,
                                _ => Method::Gear2,
                            },
                        }
                    };
                    let formula = Formula::new(method, step, previous_step);
                    let history = formula.history([charge, previous_charge], previous_derivative);
                    let next_charge = (target - history) / formula.alpha;
                    let residual = formula.alpha * next_charge + history;
                    let increment = formula.integral_increment(
                        residual,
                        previous_residual,
                        previous_increment,
                    );
                    leak += increment;
                    charge_scale = charge_scale.max(next_charge.abs());
                    let ratio_weight = if method == Method::Gear2 {
                        step / previous_step
                    } else {
                        1.0
                    };
                    ratio_sum += 1.0 + ratio_weight;
                    previous_charge = charge;
                    charge = next_charge;
                    previous_derivative = residual;
                    previous_residual = residual;
                    previous_increment = increment;
                    previous_step = step;
                }
                let actual = charge - initial_charge;
                // Per-step rounding scales with `(1 + ratio) * eps * max(|q|)` when Gear2
                // divides by `alpha` and combines `beta_0` and `beta_1`.
                let bound =
                    32.0 * f64::EPSILON * ratio_sum * charge_scale.max(leak.abs()).max(1e-30);
                prop_assert!(
                    (actual - leak).abs() <= bound,
                    "mode {mode}: actual {actual:e} vs leak {leak:e} (bound {bound:e})"
                );
            }
        }

        /// Verifies that integrating a constant residual `c` over arbitrary non-uniform steps
        /// spanning total time `T` equals `c * T` for backward Euler, trapezoidal, and Gear2.
        #[test]
        fn integral_increment_integrates_a_constant_residual_exactly(
            constant_residual in -1e-3..1e-3f64,
            exponents in proptest::collection::vec(-10.0..-7.0f64, 4..32),
        ) {
            let steps: Vec<f64> = exponents
                .iter()
                .map(|&exponent| 10_f64.powf(exponent))
                .collect();
            let total_time: f64 = steps.iter().sum();
            let expected = constant_residual * total_time;
            for method in [Method::BackwardEuler, Method::Trapezoidal, Method::Gear2] {
                let mut total = 0.0;
                let mut previous_residual = 0.0;
                let mut previous_increment = 0.0;
                let mut previous_step = steps[0];
                let mut ratio_sum = 0.0;
                for (index, &step) in steps.iter().enumerate() {
                    let chosen = if index < 2 {
                        Method::BackwardEuler
                    } else {
                        method
                    };
                    let formula = Formula::new(chosen, step, previous_step);
                    let increment = formula.integral_increment(
                        constant_residual,
                        previous_residual,
                        previous_increment,
                    );
                    total += increment;
                    let ratio_weight = if chosen == Method::Gear2 {
                        step / previous_step
                    } else {
                        1.0
                    };
                    ratio_sum += 1.0 + ratio_weight;
                    previous_residual = constant_residual;
                    previous_increment = increment;
                    previous_step = step;
                }
                let bound = 64.0 * f64::EPSILON * ratio_sum * expected.abs().max(1e-30);
                prop_assert!(
                    (total - expected).abs() <= bound,
                    "{method:?}: {total:e} vs {expected:e} (bound {bound:e})"
                );
            }
        }
    }
}
