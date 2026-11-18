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

    /// Whether a step's formula and its error read the step before it, as Gear2's do; a
    /// one-step formula's read the step alone.
    #[must_use]
    pub const fn is_multistep(self) -> bool {
        matches!(self, Self::Gear2)
    }

    /// The local truncation error of a step of length `step` that follows one of length
    /// `previous_step`, estimated from the divided difference of order `p + 1` of the charge
    /// over the step's last `p + 2` points.
    ///
    /// The error of a step is `C h^{p+1} q^{(p+1)}`, and the derivative is estimated as
    /// `q^{(p+1)} ≈ (p+1)! · q[t_{n-p}, …, t_{n+1}]`, the divided difference taken at the
    /// points' own times. `C` is ½ for backward Euler and 1/12 for the trapezoidal formula,
    /// whatever the previous step. For Gear2, `C = (1 + ω)² / (6 ω (1 + 2ω))`, where `ω` is
    /// the step's length over the previous one's: the derivative of the quadratic through
    /// the step's three points misses `h (h + h/ω) q'''/6`, which the formula's coefficient
    /// of the derivative, `h (1 + ω)/(1 + 2ω)`, turns into an error in the charge. It is 2/9
    /// at a constant step, SPICE's constant at any step, and 7.6 times that after a step
    /// cut to a tenth.
    #[must_use]
    pub fn truncation_error(self, step: f64, previous_step: f64, divided_difference: f64) -> f64 {
        self.error_per_difference(step, previous_step) * divided_difference.abs()
    }

    /// The [truncation error](Method::truncation_error) of a step of length `step` that
    /// follows one of length `previous_step`, per unit of its divided difference:
    /// `C (p+1)! h^{p+1}`, the same for every charge of the step, so that a step's judgement
    /// computes it once.
    #[must_use]
    pub fn error_per_difference(self, step: f64, previous_step: f64) -> f64 {
        let power = step.powi(self.exponent());
        self.constant(step / previous_step) * self.factorial() * power
    }

    /// The longest step after one of length `previous_step` whose truncation error, by
    /// `divided_difference`, stays within the larger of a tolerance `charge` on the charge
    /// and one on the current it carries, `rate` times the step: the
    /// [common longest step](Method::longest_common_step) of that one charge, unbounded when
    /// the divided difference vanishes.
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

    /// A charge's longest steps at a constant step's error constant: where its error
    /// `K h^{p+1}`, `K` that constant times its divided difference times `(p+1)!`, meets its
    /// tolerance `charge` on the charge, at `(charge / K)^{1/(p+1)}`, and meets its tolerance
    /// on the current it carries, `rate` times the step, at `(rate / K)^{1/p}`; both
    /// unbounded when the divided difference vanishes. For a step of another ratio to the
    /// one before it, they are these times its [scales](Method::scales).
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

    /// The factors a step `ratio` times as long as the one before it scales a charge's
    /// [unit steps](Method::unit_steps) by: `(C/C₁)^{-1/(p+1)}`, for the step its tolerance on
    /// the charge allows, and `(C/C₁)^{-1/p}`, for the one its tolerance on the current
    /// allows, `C` the error constant of such a step and `C₁` a constant step's. Both are 1 at
    /// a ratio of 1, and at any ratio for a one-step formula, whose constant reads no step
    /// before.
    #[must_use]
    pub fn scales(self, ratio: f64) -> (f64, f64) {
        let relative = self.constant(ratio) / self.constant(1.0);
        let (by_charge, by_current) = self.powers();
        (relative.powf(-by_charge), relative.powf(-by_current))
    }

    /// The longest step after one of length `previous_step` that keeps every charge's
    /// truncation error within its tolerance, given `constant_step`, the shortest of the
    /// charges' [longest steps](longest_at) at a constant step's [scales](Method::scales) and
    /// the tolerance that holds it, and `shortest`, which returns the same at the scales of
    /// any ratio.
    ///
    /// A one-step formula's error constant does not depend on the step, so its longest step is
    /// the constant step's, and `shortest` is never called. Gear2's depends on the step's ratio `u = h/h₀` to the
    /// previous one ([`Method::truncation_error`]), the same for every charge, so its longest
    /// step is the fixed point of `h = Λ(C(h/h₀))`, `Λ` the shortest: in `x = ln h`, the
    /// difference `ln Λ − x` falls with a slope between −1 and −½, since `ln C` falls with
    /// `ln u` with a slope between −1 and 0 and `Λ` grows as `C^{-1/3}` or `C^{-1/2}`. Newton's
    /// iteration on it, from the previous step's length, contracts to the fixed point, and
    /// stops where a step moves it by less than a float's resolution, or no longer shrinks.
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
            // The slope of ln Λ(C(h/h₀)) in ln h, and Newton's step on ln Λ − ln h.
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

    /// `(p+1)!`, the factorial that turns a divided difference of order `p + 1` into a
    /// derivative: how many times SPICE's estimate of a step's error, the error constant
    /// times the divided difference, the error is.
    #[must_use]
    pub const fn factorial(self) -> f64 {
        match self {
            Self::BackwardEuler => 2.0,
            Self::Trapezoidal | Self::Gear2 => 6.0,
        }
    }

    /// The constant of the leading error term of a step `ratio` times as long as the one
    /// before it: ½ for backward Euler and 1/12 for the trapezoidal formula, one-step
    /// formulas, and `(1 + ω)² / (6 ω (1 + 2ω))` for Gear2 ([`Method::truncation_error`]).
    fn constant(self, ratio: f64) -> f64 {
        match self {
            Self::BackwardEuler => 1.0 / 2.0,
            Self::Trapezoidal => 1.0 / 12.0,
            Self::Gear2 => gear2_constant(ratio),
        }
    }

    /// The powers of the error constant's reciprocal in a charge's longest steps,
    /// `1/(p+1)` by its tolerance on the charge and `1/p` by the one on its current.
    fn powers(self) -> (f64, f64) {
        let order = f64::from(self.exponent() - 1);
        ((order + 1.0).recip(), order.recip())
    }

    /// The power of the step in the error of one step, `p + 1`.
    fn exponent(self) -> i32 {
        match self {
            Self::BackwardEuler => 2,
            Self::Trapezoidal | Self::Gear2 => 3,
        }
    }
}

/// Gear2's error constant for a step `ratio` times as long as the one before it,
/// `(1 + ω)² / (6 ω (1 + 2ω))`, written as `(1 + 1/ω)(1 + ω) / (6 (1 + 2ω))` so that a long
/// step overflows nothing: from 1/12 for a step far longer than the last, through 2/9 at a
/// constant step, to `1/(6ω)` for a step far shorter.
fn gear2_constant(ratio: f64) -> f64 {
    (1.0 + ratio.recip()) * (1.0 + ratio) / (6.0 * (1.0 + 2.0 * ratio))
}

/// The logarithmic slope of Gear2's error constant in its step's ratio `u` to the previous
/// one, `d ln C / d ln u = 2u/(1 + u) − 1 − 2u/(1 + 2u)`: from −1 for a step far shorter than
/// the last, through −⅔ at a constant step, to 0 for one far longer.
fn gear2_elasticity(ratio: f64) -> f64 {
    2.0 * ratio / (1.0 + ratio) - 1.0 - 2.0 * ratio / (1.0 + 2.0 * ratio)
}

/// Which of a charge's two tolerances holds its longest step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorBound {
    /// The tolerance on the charge.
    Charge,
    /// The tolerance on the current it carries, which grows with the step.
    Current,
}

/// The [scales](Method::scales) of a constant step, a ratio of 1, for every method: both 1, as
/// its error constant is `C₁` itself; written out, so that a step's judgement multiplies by
/// nothing.
pub const CONSTANT_STEP: (f64, f64) = (1.0, 1.0);

/// A charge's longest step at the error constant whose [scales](Method::scales) are `scales`,
/// from its [unit steps](Method::unit_steps) `unit`: the longer of the two, and the tolerance
/// that holds it, its charge's where they are equal.
#[must_use]
pub fn longest_at((by_charge, by_current): (f64, f64), scales: (f64, f64)) -> (f64, ErrorBound) {
    let (charge, current) = (by_charge * scales.0, by_current * scales.1);
    if charge >= current {
        (charge, ErrorBound::Charge)
    } else {
        (current, ErrorBound::Current)
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

    /// The charge increment `Δℓ_n` a converged residual `residual` at the new point adds,
    /// given `previous_residual` at the previous point and `previous_increment` of the step
    /// before: `(residual − γ · previous_residual + β₁ · previous_increment) / α`.
    ///
    /// Because `α + β₀ + β₁ = 0`, the step formula
    /// `q̇_n = α q_n + β₀ q_{n-1} + β₁ q_{n-2} + γ q̇_{n-1}` rearranges to
    /// `α Δq_n = q̇_n − γ q̇_{n-1} + β₁ Δq_{n-1}`. Substituting `q̇_n = r_n − f_n` splits
    /// each step's charge change linearly into the discrete integral of `−f` plus that of
    /// the residual `r`:
    /// - backward Euler (`β₁ = 0, γ = 0`): `h · r_n`,
    /// - trapezoidal (`β₁ = 0, γ = −1`): `½ h · (r_n + r_{n-1})`,
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

/// The divided difference `f[t_0, …, t_k]` of `values` sampled at the distinct `times`,
/// found in place: the table of differences overwrites `values`, so finding one allocates
/// nothing.
///
/// For a polynomial of degree `k`, it is the leading coefficient; in general, it approximates
/// the `k`-th derivative divided by `k!`, which is how truncation errors are estimated.
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
            let mut values = times.map(|t| cubic(c, t).0);
            prop_assert!((divided_difference(&times, &mut values) - c[3]).abs() < 1e-8);
        }
    }

    /// The estimate of a step's error matches the error the formula actually makes on a cubic,
    /// to leading order in the step, for a step from a tenth to ten times as long as the one
    /// before it: Gear2's error grows to 7.6 times its constant step's when its step is cut
    /// to a tenth.
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

    /// Gear2's longest step after a step of unit length meets its tolerance on the charge,
    /// or on the current, exactly, for tolerances that put it from 10⁻¹⁰ to 10¹⁰ times as
    /// long, and a step a little longer exceeds it.
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
        /// The longest step meets the larger of its two tolerances exactly, and a step a
        /// little longer exceeds both.
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
        /// The longest step common to several charges is the shortest of each one's own,
        /// whichever charge, and whichever of its tolerances, holds it at the constant step's
        /// error constant.
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

    proptest! {
        /// For any sequence of steps and residuals across backward Euler, trapezoidal,
        /// Gear2, and mixed sequences with restarts, summing `integral_increment` equals
        /// the exact discrete charge change `q_N - q_0` when static current `f == 0`.
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

        /// Integrating a constant residual `c` over arbitrary non-uniform steps spanning
        /// total time `T` equals `c * T` for backward Euler, trapezoidal, and Gear2.
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
