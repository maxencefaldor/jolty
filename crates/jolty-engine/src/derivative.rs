//! Derivatives of every analysis with respect to the parameters named: its JVP, which
//! pushes tangents on the parameters forward to every unknown at every point of the
//! analysis, and its VJP, whose pullback carries cotangents on those outputs back to the
//! parameters.
//!
//! Each point of an analysis solves the circuit's equations `F(x, p) = 0`, and its solution
//! moves with the parameters by `dx/dp = −J⁻¹ ∂F/∂p`, where `J = ∂F/∂x` (the implicit
//! function theorem). Both factors are exact: `J` from the devices' dual numbers, `∂F/∂p`
//! from each parameter's twin, its instance prepared again computing with the derivative
//! of every number with respect to the parameter and evaluated where the instance is, into
//! a block of its own, so that a parameter costs what its instance does. The JVP evaluates
//! the product from the right, one solve with `J` per tangent `t`, `ẋ = −J⁻¹ (∂F/∂p) t`;
//! the VJP from the left, one solve with `Jᵀ` per cotangent `v`, `p̄ = −(J⁻ᵀ v)ᵀ ∂F/∂p`,
//! which serves every parameter however many (the adjoint method). At each point one
//! factorization serves every tangent and every cotangent. The two are transposes of one
//! linear map, so `⟨v, ẋ⟩ = ⟨p̄, t⟩` to the rounding of their solves.
//!
//! At the operating point and at each level of a DC sweep, `F` is the DC equations; a
//! number the sweep holds moves nothing there. Over frequency, the response to an
//! excitation `b` solves `(G + jωC) x = b`, so it moves by `−(G + jωC)⁻¹ (dG + jω dC) x`,
//! where `G` and `C` move with each parameter directly, exactly, and through the operating
//! point, which moves by `s = −G⁻¹ ∂F/∂p`: a central difference of the devices' `G` and `C`
//! along each parameter's own `s`, its largest voltage a microvolt either way, since the
//! devices give their derivatives with respect to their voltages, not their second
//! derivatives; the JVP and the VJP read the same difference, so they stay each other's
//! transposes. A phasor's cotangent pulls back to a complex number, `vᵀ dx/dp`, bilinear;
//! a real loss of the phasors `x = a + jb` takes the real part, with `v = ∂L/∂a − j ∂L/∂b`.
//!
//! Over time, each step solves `f(x) + α q(x) + h = 0`, where the integration formula's
//! `α` and history `h` stand for the charges' derivative. The JVP is the direct method: it
//! carries each tangent's derivatives of `x`, of the charges and of their rate from step
//! to step, `J ẋ = −(∂f/∂p t + α ∂q/∂p t + ḣ)`, each step's Jacobian evaluated again at its
//! solution (D. E. Hocevar, P. Yang, T. N. Trick and B. D. Epler, "Transient sensitivity
//! computation for MOSFET circuits", *IEEE Transactions on Computer-Aided Design* 4(4),
//! 1985). The VJP is its transpose step for step, the discrete adjoint, one backward pass
//! from the last point to the first for every cotangent at once, one transposed solve a
//! step and cotangent whatever the number of parameters. A derivative moves no step and no
//! logic event. The start's derivatives are those of the state the transient starts from:
//! the operating point's, through the DC Jacobian, its rows of the nets the initial
//! conditions hold while it is solved replaced by theirs, `v − v₀ = 0`, which no parameter
//! moves; and none for a transient from its initial conditions alone, whose state is given.
//!
//! Over a periodic steady state, a period from the start `x₀` it repeats carries each
//! tangent's derivatives the same way, the start held, to its end `∂φ/∂p t`. The steady
//! state moves so that it still repeats: `(Φ − I) ẋ₀ = −∂φ/∂p t`, by the same monodromy
//! matrix and in the same units as shooting, or for an oscillator, whose period `T` moves
//! too, the bordered system `[[Φ − I, ∂φ/∂T], [∂φ/∂Tᵀ, 0]] [ẋ₀, Ṫ] = [−∂φ/∂p t, 0]`, its
//! start still orthogonal to the flow. A second pass over the period then carries the
//! tangent from `ẋ₀`, each step's charge terms also scaled by `−Ṫ / T`, every point at its
//! fraction of the period. Where `Φ − I` is singular, a neutral direction, `ẋ₀` leaves
//! every quantity a period conserves as it was, as shooting does, no singular value
//! counting as zero but within the rounding. The VJP takes the same three steps backward:
//! the adjoint over the period, the start held, gives the cotangent's own terms and its
//! derivatives with respect to the start and the period; the transpose of the bordered
//! solve carries those to the period's end; and the adjoint of the period from there gives
//! the rest. Both passes read the steps the converged period took, evaluating the circuit
//! and factoring its Jacobian again at each point.

use std::collections::BTreeMap;

use jolty_models::{Gradient, Input, UnknownIndex};
use jolty_numerics::integration::Formula;
use jolty_numerics::{Complex64, Pattern, solve_each_in_range};

use crate::ac::over_frequencies;
use crate::newton::{add_product, force_row, step_matrix};
use crate::sensitivity::{Derivatives, Differentiated, Parameter};
use crate::state::Held;
use crate::transient::{Carried, Tangent};
use crate::{
    AcResult, AnalysisResult, DcSweep, DcSweepResult, Options, PeriodicSteadyStateResult,
    SimulationError, Simulator, Transient, TransientResult,
};

/// How far the operating point moves, along its derivative, in the central difference
/// of `G` and `C`: its largest voltage moves this many volts either way.
const BIAS_STEP: f64 = 1e-6;

/// A derivative over frequency, as a delay line's refusal names it.
const OVER_FREQUENCY: &str = "derivative over frequency";
/// Why a derivative over frequency takes no delay line.
const LEFT_OUT_OVER_FREQUENCY: &str = "it leaves out those of the terms its lines delay";
/// A derivative over time, as a delay line's refusal names it.
const OVER_TIME: &str = "derivative over time";
/// Why a derivative over time takes no delay line.
const LEFT_OUT_OVER_TIME: &str = "it leaves out those of the past its lines read";

/// What [`Simulator::jvp`] finds: the analysis's result, and the derivative of every
/// unknown at each of its points along each tangent.
#[derive(Debug, Clone, PartialEq)]
pub struct Jvp {
    /// The parameters the tangents move, in their order, each at its value.
    pub parameters: Vec<Parameter>,
    /// The derivatives, by point, then unknown, then tangent: of values, or over
    /// frequency of phasors.
    pub tangents: Derivatives,
    /// For an oscillator's periodic steady state, its period's derivative along each
    /// tangent.
    pub period: Option<Vec<f64>>,
    /// The result of the analysis differentiated.
    pub result: AnalysisResult,
}

/// A cotangent on an analysis's outputs, every unknown at every point: a linear function
/// of them, such as a loss's derivative with respect to each.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Cotangent<T> {
    /// By the index of each point where it is not zero, its value on every unknown there,
    /// in their order.
    pub points: BTreeMap<usize, Vec<T>>,
    /// Its value on an oscillator's period; zero for any other analysis.
    pub period: f64,
}

/// Cotangents on an analysis's outputs: on values, or over frequency on phasors.
#[derive(Debug, Clone, PartialEq)]
pub enum Cotangents {
    /// On values: at an operating point, along a sweep or a period, or over time.
    Real(Vec<Cotangent<f64>>),
    /// On phasors, over frequency.
    Complex(Vec<Cotangent<Complex64>>),
}

/// What a pullback finds of cotangents: each one's value on every parameter, by cotangent
/// then parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterCotangents {
    /// Of cotangents on values.
    Real(Vec<Vec<f64>>),
    /// Of cotangents on phasors.
    Complex(Vec<Vec<Complex64>>),
}

/// What [`Simulator::vjp`] finds: the analysis's result, and what
/// [`Simulator::pull_back`] reads to carry cotangents on it back to the parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Pullback {
    /// The parameters the cotangents pull back to, in their order, each at its value.
    pub parameters: Vec<Parameter>,
    /// The result of the analysis differentiated.
    pub result: AnalysisResult,
    /// What else of the analysis the pullback reads.
    saved: Saved,
}

/// What a pullback reads of its analysis besides its result.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Saved {
    /// Nothing, at an operating point or over frequency.
    Nothing,
    /// The sweep, whose levels it holds again.
    DcSweep(DcSweep),
    /// The transient's start, and each step's formula and whether the integration
    /// restarts after it.
    Transient(Start, Vec<(Formula, bool)>),
    /// The converged period's steps, and shooting's bordered system.
    Period(Vec<(Formula, bool)>, System),
}

/// Shooting's bordered system, its unknowns' tolerances and its period's scale, as
/// [`crate::periodic::Shot::system`] gives them.
pub(crate) type System = (Vec<f64>, Vec<f64>, Option<f64>);

/// A periodic steady state's period, the steps it took and its shooting's bordered system.
pub(crate) type Period<'a> = (
    &'a PeriodicSteadyStateResult,
    &'a [(Formula, bool)],
    &'a System,
);

/// How `G` and `C` move, by stored position, as the operating point moves along one
/// parameter's derivative of it; none when no voltage moves.
type Bias = Option<(Vec<f64>, Vec<f64>)>;

/// Where a transient's derivatives start.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Start {
    /// At the operating point, solved with the unknowns given forced, as a transient's
    /// initial conditions force theirs.
    Solved(Vec<UnknownIndex>),
    /// At a state held, given: its derivatives along each direction, or none, zero.
    Held(Option<Vec<Vec<f64>>>),
}

/// A device's instance computing with the derivative of every number with respect to one
/// number of its parameters.
pub(crate) struct Twin {
    /// The index of the device it is made from.
    pub(crate) device: usize,
    pub(crate) parameter: Parameter,
    pub(crate) made: jolty_models::Twin,
    /// Whether a sweep holds its number, which then moves nothing: the number swept, or
    /// any of a source the sweep holds at a level, which replaces its waveform.
    pub(crate) fixed: bool,
}

impl Twin {
    /// Evaluates the twin where the device is: at the solution `x`, at `time` in a
    /// transient analysis or, with `None`, at an operating point or about one.
    pub(crate) fn evaluate(&mut self, x: &[f64], time: Option<f64>, gmin: f64) {
        let mut input = Input::new(x, time, gmin);
        input.limiting = false;
        self.made.evaluate(&input);
    }

    /// `aᵀ ∂f/∂p` for the vector `a` of the unknowns, as last evaluated: the instance's own
    /// terms alone.
    pub(crate) fn residual_dot(&self, a: &[f64]) -> f64 {
        self.made
            .residual()
            .map(|(unknown, residual)| a[unknown] * residual.derivative)
            .sum()
    }

    /// `aᵀ ∂q/∂p` for the vector `a` of the unknowns, as last evaluated: the instance's own
    /// terms alone.
    pub(crate) fn charge_dot(&self, a: &[f64]) -> f64 {
        self.made
            .charge()
            .map(|(unknown, charge)| a[unknown] * charge.derivative)
            .sum()
    }

    /// `aᵀ (∂G/∂p + jω ∂C/∂p) x` at `(ω, x)`, for the vector `a` of the equations, as last
    /// evaluated: the instance's own entries alone, each at its row and column of
    /// `positions`.
    fn product(
        &self,
        positions: &[(usize, usize)],
        a: &[Complex64],
        (omega, x): (f64, &[Complex64]),
    ) -> Complex64 {
        self.made
            .entries()
            .map(|(slot, conductance, capacitance)| {
                let (row, column) = positions[slot];
                let derivative =
                    Complex64::new(conductance.derivative, omega * capacitance.derivative);
                a[row] * derivative * x[column]
            })
            .sum()
    }

    /// Adds `scale (∂G/∂p + jω ∂C/∂p) x` at `(ω, x)` to `into`, as last evaluated.
    fn add_product(
        &self,
        positions: &[(usize, usize)],
        scale: f64,
        (omega, x): (f64, &[Complex64]),
        into: &mut [Complex64],
    ) {
        for (slot, conductance, capacitance) in self.made.entries() {
            let (row, column) = positions[slot];
            let derivative = Complex64::new(conductance.derivative, omega * capacitance.derivative);
            into[row] += derivative * x[column] * scale;
        }
    }
}

/// Which of a twin's derivatives a direction combines.
#[derive(Clone, Copy)]
enum Part {
    /// `∂f/∂p`.
    Residual,
    /// `∂q/∂p`.
    Charge,
}

/// `Σ t_j ∂f/∂p_j` or `Σ t_j ∂q/∂p_j` along `direction`, a number per twin, by unknown,
/// among `dimension`, as the twins were last evaluated; a twin a sweep holds moves nothing.
fn combined(twins: &[Twin], direction: &[f64], dimension: usize, part: Part) -> Vec<f64> {
    let mut values = vec![0.0; dimension];
    for (twin, &along) in twins.iter().zip(direction) {
        if along == 0.0 || twin.fixed {
            continue;
        }
        let derivatives: Box<dyn Iterator<Item = (usize, Gradient)>> = match part {
            Part::Residual => Box::new(twin.made.residual()),
            Part::Charge => Box::new(twin.made.charge()),
        };
        for (unknown, value) in derivatives {
            values[unknown] += along * value.derivative;
        }
    }
    values
}

/// The unit directions, one along each of `count` parameters.
pub(crate) fn units(count: usize) -> Vec<Vec<f64>> {
    (0..count)
        .map(|index| {
            let mut unit = vec![0.0; count];
            unit[index] = 1.0;
            unit
        })
        .collect()
}

/// Adds `scale` times `x` to `into`.
fn add(into: &mut [f64], scale: f64, x: &[f64]) {
    for (value, term) in into.iter_mut().zip(x) {
        *value += scale * term;
    }
}

/// Adds to `into` the product of the transpose of the matrix of `pattern` whose stored
/// values are `values` and the vector `x`, the matrix's rows in order.
fn add_transposed_product(pattern: &Pattern, values: &[f64], x: &[f64], into: &mut [f64]) {
    for (row, &factor) in x.iter().enumerate() {
        for &(slot, column) in pattern.row(row) {
            into[column] += values[slot] * factor;
        }
    }
}

/// `rows` by `columns` turned to `columns` by `rows`.
fn transposed<T: Copy>(rows: Vec<Vec<T>>, columns: usize) -> Vec<Vec<T>> {
    let mut turned = vec![Vec::with_capacity(rows.len()); columns];
    for row in rows {
        for (column, value) in turned.iter_mut().zip(row) {
            column.push(value);
        }
    }
    turned
}

/// The direct method's state along a transient: each direction's derivatives of the
/// charges, and those of the unknowns kept at every accepted point.
pub(crate) struct Forward {
    pub(crate) twins: Vec<Twin>,
    /// The directions, by direction then twin.
    directions: Vec<Vec<f64>>,
    /// The unknowns whose derivatives it keeps, by index.
    kept: Vec<usize>,
    start: Start,
    tangents: Vec<Tangent>,
    /// By point, then unknown kept, then direction.
    pub(crate) found: Vec<Vec<Vec<f64>>>,
    /// An oscillator's period, and its derivative along each direction, which scales
    /// every step.
    pub(crate) stretch: Option<(f64, Vec<f64>)>,
    /// The charges themselves, while the period stretches, whose formula's terms the
    /// steps' lengths divide.
    charges: Option<Tangent>,
    /// Each direction's derivatives of the unknowns at the last point, kept for a period
    /// whose end they are.
    ends: Option<Vec<Vec<f64>>>,
}

impl Forward {
    /// The direct method along `directions` of `twins`, keeping the derivatives of the
    /// unknowns `kept`, from `start`.
    pub(crate) fn new(
        twins: Vec<Twin>,
        directions: Vec<Vec<f64>>,
        kept: Vec<usize>,
        start: Start,
    ) -> Self {
        Self {
            twins,
            directions,
            kept,
            start,
            tangents: Vec::new(),
            found: Vec::new(),
            stretch: None,
            charges: None,
            ends: None,
        }
    }

    /// The derivatives of the unknowns kept in `moved`, by direction then unknown, by
    /// unknown kept then direction.
    fn keep(&self, moved: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.kept
            .iter()
            .map(|&unknown| moved.iter().map(|shift| shift[unknown]).collect())
            .collect()
    }
}

/// What a JVP finds: the twins, as they ended the analysis, the derivatives of the
/// unknowns kept, by point, then unknown kept, then direction, an oscillator's period's
/// along each direction, and the analysis's result.
pub(crate) struct Pushed {
    pub(crate) twins: Vec<Twin>,
    pub(crate) tangents: Derivatives,
    pub(crate) period: Option<Vec<f64>>,
    pub(crate) result: AnalysisResult,
}

/// Each cotangent's derivatives, as the adjoint method carries them backward along a
/// transient, with respect to the charges at the last point, at the one before it and to
/// their rate, and while a period stretches the same of the stretch's terms, and the
/// period's derivative found so far.
struct Adjoint {
    charge: Vec<f64>,
    before: Vec<f64>,
    rate: Vec<f64>,
    stretch: Option<(Vec<f64>, Vec<f64>)>,
    period: f64,
    /// The last point its cotangent is not zero at, before which it carries nothing.
    last: Option<usize>,
}

impl Adjoint {
    /// Carries the derivatives back over the step `formula` took to a point where the
    /// charges are `charge`, `restarted` saying whether the integration restarts after it,
    /// from `held`, those with respect to the charges at the point, `onward`, what goes on
    /// to the point before, and `solution`, the step's `λ`; while a period of length
    /// `stretch` stretches, its terms too. Returns `μ − α λ`, which the charges' partial
    /// derivatives take.
    fn step_back(
        &mut self,
        (held, onward, solution): (&[f64], &[f64], &[f64]),
        (formula, restarted, charge): (Formula, bool, &[f64]),
        stretch: Option<f64>,
    ) -> Vec<f64> {
        let mut through = std::mem::take(&mut self.rate);
        add(&mut through, -1.0, solution);
        if let (Some((stretch_charge, stretch_before)), Some(period)) = (&mut self.stretch, stretch)
        {
            let mut read = std::mem::take(stretch_charge);
            let onward = if restarted {
                add(&mut read, 1.0, stretch_before);
                vec![0.0; read.len()]
            } else {
                std::mem::take(stretch_before)
            };
            add(&mut read, formula.alpha, &through);
            let dot: f64 = read.iter().zip(charge).map(|(a, q)| a * q).sum();
            self.period -= dot / period;
            *stretch_charge = onward;
            add(stretch_charge, formula.beta[0], &through);
            *stretch_before = through
                .iter()
                .map(|value| formula.beta[1] * value)
                .collect();
        }
        self.charge.clear();
        self.charge.extend_from_slice(onward);
        add(&mut self.charge, formula.beta[0], &through);
        self.before = through
            .iter()
            .map(|value| formula.beta[1] * value)
            .collect();
        self.rate = through.iter().map(|value| formula.gamma * value).collect();
        let mut moved = held.to_vec();
        add(&mut moved, -formula.alpha, solution);
        moved
    }
}

/// What the adjoint method leaves of each cotangent at a transient's start: its derivative
/// with respect to the state held there, and with respect to an oscillator's period.
type Ends = Vec<(Vec<f64>, f64)>;

/// What takes each term of a cotangent's value on every parameter, by its index, every
/// parameter's at once; the terms sum to the values.
pub(crate) type Fold<'a> = &'a mut dyn FnMut(usize, &[f64]);

impl Simulator {
    /// The derivative of every unknown at each point of `analysis` along each of
    /// `tangents`, each a number per parameter of `parameters`, every parameter by its
    /// device's path and its name, such as `R1` and `resistance` or `V1` and
    /// `waveform.dc`: the JVP of the analysis, by the direct method, one solve per tangent
    /// at each point.
    ///
    /// The derivatives are exact, but for a response's through its operating point, a
    /// central difference along each parameter's own derivative of the operating point.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownParameters`] for every parameter no device holds
    /// as a real number, [`SimulationError::InvalidAnalysis`] for a tangent that is not a
    /// finite number per parameter, [`SimulationError::Singular`] when a Jacobian is, and
    /// the errors of the analysis.
    pub fn jvp(
        &mut self,
        analysis: &Differentiated,
        parameters: &[(String, String)],
        tangents: &[Vec<f64>],
        options: &Options,
    ) -> Result<Jvp, SimulationError> {
        options.parallelism.confined(|| {
            self.solvable()?;
            let twins = self.named_twins(parameters)?;
            if let Some(tangent) = tangents.iter().find(|tangent| {
                tangent.len() != parameters.len() || !tangent.iter().all(|value| value.is_finite())
            }) {
                return Err(SimulationError::InvalidAnalysis(format!(
                    "a tangent is a finite number for each of the {} parameters, not {tangent:?}",
                    parameters.len()
                )));
            }
            let kept: Vec<usize> = (0..self.elaborated.unknowns.len()).collect();
            let pushed = self.push_forward(analysis, twins, tangents.to_vec(), &kept, options)?;
            Ok(Jvp {
                parameters: pushed
                    .twins
                    .into_iter()
                    .map(|twin| twin.parameter)
                    .collect(),
                tangents: pushed.tangents,
                period: pushed.period,
                result: pushed.result,
            })
        })
    }

    /// The result of `analysis`, and its pullback: what carries a cotangent on every
    /// unknown at each of its points back to each parameter of `parameters`, each by its
    /// device's path and its name, through [`Simulator::pull_back`]: the VJP of the
    /// analysis, by the adjoint method, one transposed solve per cotangent at each point
    /// whatever the number of parameters.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownParameters`] for every parameter no device holds
    /// as a real number, [`SimulationError::Delayed`] over frequency or time for a circuit
    /// with delay lines, whose terms the derivatives would leave out,
    /// [`SimulationError::InvalidAnalysis`] over time for a circuit with logic, whose
    /// drivers a point evaluated again after the run does not see as they were, and the
    /// errors of the analysis.
    pub fn vjp(
        &mut self,
        analysis: &Differentiated,
        parameters: &[(String, String)],
        options: &Options,
    ) -> Result<Pullback, SimulationError> {
        options.parallelism.confined(|| {
            self.solvable()?;
            let twins = self.named_twins(parameters)?;
            let (result, saved) = self.run_saving(analysis, options)?;
            Ok(Pullback {
                parameters: twins.into_iter().map(|twin| twin.parameter).collect(),
                result,
                saved,
            })
        })
    }

    /// The value of each of `cotangents` on every parameter of `pullback`, by cotangent
    /// then parameter: its pullback, by the adjoint method.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidAnalysis`] for cotangents of another kind than
    /// the analysis's outputs, or at a point it does not have, or of another number of
    /// unknowns, or not finite; [`SimulationError::Singular`] when a Jacobian is.
    pub fn pull_back(
        &mut self,
        pullback: &Pullback,
        cotangents: &Cotangents,
        options: &Options,
    ) -> Result<ParameterCotangents, SimulationError> {
        options.parallelism.confined(|| {
            self.solvable()?;
            self.prepare(options)?;
            let names: Vec<(String, String)> = pullback
                .parameters
                .iter()
                .map(|parameter| (parameter.instance.clone(), parameter.name.clone()))
                .collect();
            let mut twins = self.named_twins(&names)?;
            self.checked(&pullback.result, cotangents)?;
            let count = twins.len();
            match (&pullback.saved, &pullback.result, cotangents) {
                (_, AnalysisResult::Ac(response), Cotangents::Complex(cotangents)) => {
                    Ok(ParameterCotangents::Complex(self.pull_frequencies(
                        &mut twins, response, cotangents, options,
                    )?))
                }
                (saved, result, Cotangents::Real(cotangents)) => {
                    let mut found = vec![vec![0.0; count]; cotangents.len()];
                    let mut fold = |index: usize, terms: &[f64]| {
                        for (total, term) in found[index].iter_mut().zip(terms) {
                            *total += term;
                        }
                    };
                    self.pull(&mut twins, (saved, result), cotangents, &mut fold, options)?;
                    Ok(ParameterCotangents::Real(found))
                }
                _ => Err(SimulationError::InvalidAnalysis(
                    "cotangents on a response's phasors are complex, and on any other \
                     analysis's values real"
                        .to_owned(),
                )),
            }
        })
    }

    /// The twins of `parameters`, each by its device's path and its name, in their order.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownParameters`] naming every one no device holds as
    /// a real number.
    pub(crate) fn named_twins(
        &self,
        parameters: &[(String, String)],
    ) -> Result<Vec<Twin>, SimulationError> {
        let mut twins = Vec::with_capacity(parameters.len());
        let mut missing = Vec::new();
        for (path, name) in parameters {
            match self.twin_at(path, name) {
                Some(twin) => twins.push(twin),
                None => missing.push((path.clone(), name.clone())),
            }
        }
        if missing.is_empty() {
            Ok(twins)
        } else {
            Err(SimulationError::UnknownParameters(missing))
        }
    }

    /// Checks that `cotangents` have the points and unknowns of `result` and are finite.
    fn checked(
        &self,
        result: &AnalysisResult,
        cotangents: &Cotangents,
    ) -> Result<(), SimulationError> {
        let points = match result {
            AnalysisResult::DcSweep(sweep) => sweep.values.len(),
            AnalysisResult::Ac(response) => response.values.len(),
            AnalysisResult::Transient(waveforms) => waveforms.values.len(),
            AnalysisResult::PeriodicSteadyState(found) => found.waveforms.values.len(),
            _ => 1,
        };
        let unknowns = self.elaborated.unknowns.len();
        let fits = |at: usize, values: usize, finite: bool, period: f64| {
            at < points && values == unknowns && finite && period.is_finite()
        };
        let fit = match cotangents {
            Cotangents::Real(each) => each.iter().all(|cotangent| {
                cotangent.points.iter().all(|(&at, values)| {
                    let finite = values.iter().all(|value| value.is_finite());
                    fits(at, values.len(), finite, cotangent.period)
                })
            }),
            Cotangents::Complex(each) => each.iter().all(|cotangent| {
                cotangent.points.iter().all(|(&at, values)| {
                    let finite = values.iter().all(|value| value.is_finite());
                    fits(at, values.len(), finite, cotangent.period)
                })
            }),
        };
        if fit {
            Ok(())
        } else {
            Err(SimulationError::InvalidAnalysis(format!(
                "a cotangent is finite, at points from 0 to {} of {unknowns} unknowns each",
                points.saturating_sub(1)
            )))
        }
    }

    /// The JVP of `analysis` along `directions`, each a number per twin of `twins`,
    /// keeping the derivatives of the unknowns `kept`.
    ///
    /// # Errors
    ///
    /// As [`Simulator::jvp`].
    pub(crate) fn push_forward(
        &mut self,
        analysis: &Differentiated,
        mut twins: Vec<Twin>,
        directions: Vec<Vec<f64>>,
        kept: &[usize],
        options: &Options,
    ) -> Result<Pushed, SimulationError> {
        let mut period = None;
        let (tangents, result) = match analysis {
            Differentiated::OperatingPoint => {
                let point = self.operating_point(options)?;
                let moved = self.dc_shifts(&mut twins, &point.values, &directions, &[], options)?;
                let at = kept_of(kept, &moved);
                (
                    Derivatives::Real(vec![at]),
                    AnalysisResult::OperatingPoint(point),
                )
            }
            Differentiated::DcSweep(sweep) => {
                let result = self.dc_sweep(sweep, options)?;
                let found =
                    self.along_levels(sweep, &result, &mut twins, |simulator, twins, x| {
                        let moved = simulator.dc_shifts(twins, x, &directions, &[], options)?;
                        Ok(kept_of(kept, &moved))
                    })?;
                (Derivatives::Real(found), AnalysisResult::DcSweep(result))
            }
            Differentiated::Ac(ac) => {
                self.undelayed(OVER_FREQUENCY, LEFT_OUT_OVER_FREQUENCY)?;
                let response = self.ac(ac, options)?;
                let found =
                    self.frequency_tangents(&mut twins, &response, &directions, kept, options)?;
                (Derivatives::Complex(found), AnalysisResult::Ac(response))
            }
            Differentiated::Transient(transient) => {
                self.undelayed(OVER_TIME, LEFT_OUT_OVER_TIME)?;
                let start = self.start_of(transient)?;
                let mut forward = Forward::new(twins, directions, kept.to_vec(), start);
                let carried = Carried::Forward(&mut forward);
                let waveforms = self.transient_with(transient, options, None, carried)?;
                twins = forward.twins;
                (
                    Derivatives::Real(forward.found),
                    AnalysisResult::Transient(waveforms),
                )
            }
            Differentiated::PeriodicSteadyState(analysis) => {
                let (found, mut shot) = self.shot_steady_state(analysis, options)?;
                let steps = shot.take_steps();
                let system = shot.system();
                let forward = Forward::new(twins, directions, kept.to_vec(), Start::Held(None));
                let forward = self.along_period((&found, &steps, &system), forward, options)?;
                twins = forward.twins;
                period = forward.stretch.map(|(_, derivatives)| derivatives);
                (
                    Derivatives::Real(forward.found),
                    AnalysisResult::PeriodicSteadyState(found),
                )
            }
        };
        Ok(Pushed {
            twins,
            tangents,
            period,
            result,
        })
    }

    /// The result of `analysis`, and what its pullback reads of it besides.
    ///
    /// # Errors
    ///
    /// As [`Simulator::vjp`].
    pub(crate) fn run_saving(
        &mut self,
        analysis: &Differentiated,
        options: &Options,
    ) -> Result<(AnalysisResult, Saved), SimulationError> {
        Ok(match analysis {
            Differentiated::OperatingPoint => (
                AnalysisResult::OperatingPoint(self.operating_point(options)?),
                Saved::Nothing,
            ),
            Differentiated::DcSweep(sweep) => (
                AnalysisResult::DcSweep(self.dc_sweep(sweep, options)?),
                Saved::DcSweep(sweep.clone()),
            ),
            Differentiated::Ac(ac) => {
                self.undelayed(OVER_FREQUENCY, LEFT_OUT_OVER_FREQUENCY)?;
                (AnalysisResult::Ac(self.ac(ac, options)?), Saved::Nothing)
            }
            Differentiated::Transient(transient) => {
                self.replayable()?;
                let start = self.start_of(transient)?;
                let mut steps = Vec::new();
                let waveforms =
                    self.transient_with(transient, options, None, Carried::Steps(&mut steps))?;
                (
                    AnalysisResult::Transient(waveforms),
                    Saved::Transient(start, steps),
                )
            }
            Differentiated::PeriodicSteadyState(analysis) => {
                let (found, mut shot) = self.shot_steady_state(analysis, options)?;
                let steps = shot.take_steps();
                (
                    AnalysisResult::PeriodicSteadyState(found),
                    Saved::Period(steps, shot.system()),
                )
            }
        })
    }

    /// Fails for a circuit whose points the adjoint method cannot evaluate again after its
    /// run as they were: one with delay lines, whose past the derivatives would leave out,
    /// or with logic, whose drivers it does not see as they were.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Delayed`] and [`SimulationError::InvalidAnalysis`].
    pub(crate) fn replayable(&self) -> Result<(), SimulationError> {
        self.undelayed(OVER_TIME, LEFT_OUT_OVER_TIME)?;
        if self.elaborated.logic.nets.is_empty() {
            return Ok(());
        }
        Err(SimulationError::InvalidAnalysis(
            "a derivative over time by the adjoint method is found for circuits without \
             logic, whose drivers a point evaluated again after the run does not see as \
             they were"
                .to_owned(),
        ))
    }

    /// Where the derivatives of `transient` start: at its operating point, with the nets
    /// its initial conditions force there, or at its initial conditions alone, held.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidAnalysis`] for initial conditions that name no
    /// net, or a voltage that is not finite.
    pub(crate) fn start_of(&self, transient: &Transient) -> Result<Start, SimulationError> {
        if !transient.operating_point {
            return Ok(Start::Held(None));
        }
        let [initial] = self.forcings([(&transient.initial_conditions, "initial condition")])?;
        Ok(Start::Solved(
            initial
                .forced
                .into_iter()
                .map(|(unknown, _)| unknown)
                .collect(),
        ))
    }

    /// The real cotangents `cotangents` of the analysis whose result is `result`, which
    /// left `saved`, pulled back to the twins: each term of each cotangent's values, by
    /// its index, to `fold`.
    pub(crate) fn pull(
        &mut self,
        twins: &mut [Twin],
        (saved, result): (&Saved, &AnalysisResult),
        cotangents: &[Cotangent<f64>],
        fold: Fold<'_>,
        options: &Options,
    ) -> Result<(), SimulationError> {
        let seeds: Vec<&BTreeMap<usize, Vec<f64>>> = cotangents
            .iter()
            .map(|cotangent| &cotangent.points)
            .collect();
        match (saved, result) {
            (Saved::Nothing, AnalysisResult::OperatingPoint(point)) => {
                let at: Vec<(usize, &[f64])> = seeds
                    .iter()
                    .enumerate()
                    .filter_map(|(index, seed)| Some((index, seed.get(&0)?.as_slice())))
                    .collect();
                self.dc_cotangents(twins, &point.values, &at, fold, options)
            }
            (Saved::DcSweep(sweep), AnalysisResult::DcSweep(found)) => {
                let mut level = 0;
                self.along_levels(sweep, found, twins, |simulator, twins, x| {
                    let at: Vec<(usize, &[f64])> = seeds
                        .iter()
                        .enumerate()
                        .filter_map(|(index, seed)| Some((index, seed.get(&level)?.as_slice())))
                        .collect();
                    level += 1;
                    simulator.dc_cotangents(twins, x, &at, &mut *fold, options)
                })?;
                Ok(())
            }
            (Saved::Transient(start, steps), AnalysisResult::Transient(waveforms)) => {
                self.backward(
                    twins,
                    (waveforms, steps),
                    &seeds,
                    (start, None),
                    fold,
                    options,
                )?;
                Ok(())
            }
            (Saved::Period(steps, system), AnalysisResult::PeriodicSteadyState(found)) => {
                self.pull_period(twins, (found, steps, system), cotangents, fold, options)
            }
            _ => Err(SimulationError::InvalidAnalysis(
                "cotangents on a response's phasors are complex, and on any other analysis's \
                 values real"
                    .to_owned(),
            )),
        }
    }

    /// Each direction's derivative of the DC solution `x`, `−J⁻¹ Σ t_j ∂F/∂p_j`, by
    /// direction then unknown, the rows of the unknowns `forced` held at their voltages,
    /// `v − v₀ = 0`, which no parameter moves; the twins left evaluated at `x`.
    fn dc_shifts(
        &mut self,
        twins: &mut [Twin],
        x: &[f64],
        directions: &[Vec<f64>],
        forced: &[UnknownIndex],
        options: &Options,
    ) -> Result<Vec<Vec<f64>>, SimulationError> {
        self.evaluate(x, None, (1.0, options.gmin), false);
        let mut forced_matrix = None;
        if !forced.is_empty() {
            let mut matrix = self.state.workspace.contributions.conductance.clone();
            for unknown in forced {
                force_row(&self.elaborated, &mut matrix, unknown.index());
            }
            forced_matrix = Some(matrix);
        }
        let matrix = forced_matrix
            .as_deref()
            .unwrap_or(&self.state.workspace.contributions.conductance);
        let factors = self.state.solver.factor(matrix)?;
        let shared = factors.shared();
        let parallelism = &options.parallelism;
        parallelism.spread(
            twins,
            || (),
            |(), twin| twin.evaluate(x, None, options.gmin),
        );
        let dimension = x.len();
        let twins = &*twins;
        let mut directions: Vec<&Vec<f64>> = directions.iter().collect();
        let moved = parallelism.spread(
            &mut directions,
            || shared.workspace(),
            |workspace, direction| {
                let mut moved: Vec<f64> = combined(twins, direction, dimension, Part::Residual)
                    .iter()
                    .map(|value| -value)
                    .collect();
                for unknown in forced {
                    moved[unknown.index()] = 0.0;
                }
                shared.solve(&mut moved, workspace)?;
                Ok(moved)
            },
        );
        moved.into_iter().collect()
    }

    /// Pulls back the cotangents `at`, each its index and its value on every unknown, at
    /// the DC solution `x`: `−λᵀ ∂F/∂p` for each twin, where `Jᵀ λ = v`, to `fold`.
    fn dc_cotangents(
        &mut self,
        twins: &mut [Twin],
        x: &[f64],
        at: &[(usize, &[f64])],
        fold: Fold<'_>,
        options: &Options,
    ) -> Result<(), SimulationError> {
        if at.is_empty() {
            return Ok(());
        }
        self.evaluate(x, None, (1.0, options.gmin), false);
        let factors = self
            .state
            .solver
            .factor(&self.state.workspace.contributions.conductance)?;
        let shared = factors.shared();
        let parallelism = &options.parallelism;
        let mut given: Vec<&[f64]> = at.iter().map(|&(_, values)| values).collect();
        let adjoints = parallelism.spread(
            &mut given,
            || shared.workspace(),
            |workspace, values| {
                let mut adjoint = values.to_vec();
                shared.solve_transpose(&mut adjoint, workspace)?;
                Ok(adjoint)
            },
        );
        let adjoints = adjoints
            .into_iter()
            .collect::<Result<Vec<Vec<f64>>, SimulationError>>()?;
        let by_twin = parallelism.spread(
            twins,
            || (),
            |(), twin| {
                if twin.fixed {
                    return vec![0.0; adjoints.len()];
                }
                twin.evaluate(x, None, options.gmin);
                adjoints
                    .iter()
                    .map(|adjoint| -twin.residual_dot(adjoint))
                    .collect::<Vec<f64>>()
            },
        );
        for (&(index, _), terms) in at.iter().zip(transposed(by_twin, at.len())) {
            fold(index, &terms);
        }
        Ok(())
    }

    /// What `each` finds at each level of the sweep `result`, of `sweep`, given the
    /// solution there, its source held there, and the twins of its device prepared again
    /// there, from what its instance is made of: a number the sweep holds moves nothing.
    fn along_levels<R>(
        &mut self,
        sweep: &DcSweep,
        result: &DcSweepResult,
        twins: &mut [Twin],
        mut each: impl FnMut(&mut Self, &mut [Twin], &[f64]) -> Result<R, SimulationError>,
    ) -> Result<Vec<R>, SimulationError> {
        let swept = self.swept(sweep)?;
        let mut found = Vec::with_capacity(result.swept.len());
        let mut run = || {
            for (&level, x) in result.swept.iter().zip(&result.values) {
                self.hold(swept, Some(level))?;
                for twin in twins.iter_mut().filter(|twin| twin.device == swept.device) {
                    twin.fixed = match swept.held {
                        Held::Level(_) => true,
                        Held::Number(_) => sweep.parameter.as_ref() == Some(&twin.parameter.name),
                    };
                    if !twin.fixed
                        && let Some(made) = self.twin(swept.device, twin.parameter.clone())
                    {
                        *twin = made;
                    }
                }
                found.push(each(self, twins, x)?);
            }
            Ok(())
        };
        let ran = run();
        self.hold(swept, None)?;
        ran.map(|()| found)
    }

    /// Each twin's derivative of the operating point `x` and of `G` and `C` along it, by
    /// stored position, a central difference whose largest voltage moves [`BIAS_STEP`]
    /// either way, none when no voltage moves; the devices left evaluated at `x`, and the
    /// twins too.
    fn biases(
        &mut self,
        twins: &mut [Twin],
        x: &[f64],
        options: &Options,
    ) -> Result<Vec<Bias>, SimulationError> {
        let units = units(twins.len());
        let moved = self.dc_shifts(twins, x, &units, &[], options)?;
        let mut biases = Vec::with_capacity(twins.len());
        for shift in &moved {
            biases.push(self.bias_derivative(x, shift, options));
        }
        self.evaluate(x, None, (1.0, options.gmin), false);
        Ok(biases)
    }

    /// The derivatives of `G` and `C`, by stored position, as the operating point `x` moves
    /// along `shift`: a central difference whose largest voltage moves [`BIAS_STEP`] either
    /// way. `None` when no voltage moves.
    fn bias_derivative(
        &mut self,
        x: &[f64],
        shift: &[f64],
        options: &Options,
    ) -> Option<(Vec<f64>, Vec<f64>)> {
        let largest = self
            .elaborated
            .unknowns
            .iter()
            .zip(shift)
            .filter(|(unknown, _)| !unknown.nature.is_current())
            .fold(0.0_f64, |most, (_, value)| most.max(value.abs()));
        if largest == 0.0 || !largest.is_finite() {
            return None;
        }
        let step = BIAS_STEP / largest;
        let mut ends = [(); 2].map(|()| (Vec::new(), Vec::new()));
        for (end, sign) in ends.iter_mut().zip([1.0, -1.0]) {
            let moved: Vec<f64> = x
                .iter()
                .zip(shift)
                .map(|(value, along)| value + sign * step * along)
                .collect();
            self.evaluate(&moved, None, (1.0, options.gmin), false);
            let contributions = &self.state.workspace.contributions;
            *end = (
                contributions.conductance.clone(),
                contributions.capacitance.clone(),
            );
        }
        let [(g_up, c_up), (g_down, c_down)] = ends;
        let difference = |up: &[f64], down: &[f64]| -> Vec<f64> {
            up.iter()
                .zip(down)
                .map(|(above, below)| (above - below) / (2.0 * step))
                .collect()
        };
        Some((difference(&g_up, &g_down), difference(&c_up, &c_down)))
    }

    /// The derivatives of the phasors of the unknowns `kept` in `response` along each of
    /// `directions`, by frequency, then unknown kept, then direction.
    fn frequency_tangents(
        &mut self,
        twins: &mut [Twin],
        response: &AcResult,
        directions: &[Vec<f64>],
        kept: &[usize],
        options: &Options,
    ) -> Result<Vec<Vec<Vec<Complex64>>>, SimulationError> {
        let biases = self.biases(twins, &response.operating_point.values, options)?;
        let positions: Vec<(usize, usize)> = self.elaborated.pattern.positions().collect();
        let twins = &*twins;
        let dimension = self.elaborated.unknowns.len();
        over_frequencies(
            self.linearization(),
            &self.state.workspace.contributions,
            &response.frequencies,
            &options.parallelism,
            &|factors, index, frequency| {
                let phasors = &response.values[index];
                let omega = std::f64::consts::TAU * frequency;
                let mut at = Vec::with_capacity(directions.len());
                for direction in directions {
                    let mut moved = vec![Complex64::default(); dimension];
                    for ((twin, bias), &along) in twins.iter().zip(&biases).zip(direction) {
                        if along == 0.0 {
                            continue;
                        }
                        twin.add_product(&positions, along, (omega, phasors), &mut moved);
                        if let Some((conductance, capacitance)) = bias {
                            let entries = positions.iter().zip(conductance.iter().zip(capacitance));
                            for (&(row, column), (g, c)) in entries {
                                moved[row] +=
                                    Complex64::new(*g, omega * c) * phasors[column] * along;
                            }
                        }
                    }
                    for value in &mut moved {
                        *value = -*value;
                    }
                    factors.solve(&mut moved)?;
                    at.push(kept.iter().map(|&unknown| moved[unknown]).collect());
                }
                Ok(transposed(at, kept.len()))
            },
        )
    }

    /// The values of `cotangents` on the phasors of `response` on every twin, by
    /// cotangent then twin, summed over the frequencies in order.
    pub(crate) fn pull_frequencies(
        &mut self,
        twins: &mut [Twin],
        response: &AcResult,
        cotangents: &[Cotangent<Complex64>],
        options: &Options,
    ) -> Result<Vec<Vec<Complex64>>, SimulationError> {
        let biases = self.biases(twins, &response.operating_point.values, options)?;
        let positions: Vec<(usize, usize)> = self.elaborated.pattern.positions().collect();
        let twins = &*twins;
        let found = over_frequencies(
            self.linearization(),
            &self.state.workspace.contributions,
            &response.frequencies,
            &options.parallelism,
            &|factors, index, frequency| {
                let phasors = &response.values[index];
                let omega = std::f64::consts::TAU * frequency;
                let mut at = Vec::new();
                for (cotangent, given) in cotangents.iter().enumerate() {
                    let Some(values) = given.points.get(&index) else {
                        continue;
                    };
                    let mut adjoint = values.clone();
                    factors.solve_transpose(&mut adjoint)?;
                    let mut by_twin = Vec::with_capacity(twins.len());
                    for (twin, moving) in twins.iter().zip(&biases) {
                        let mut total = twin.product(&positions, &adjoint, (omega, phasors));
                        if let Some((conductance, capacitance)) = moving {
                            let entries = positions.iter().zip(conductance.iter().zip(capacitance));
                            total += entries
                                .map(|(&(row, column), (g, c))| {
                                    adjoint[row] * Complex64::new(*g, omega * c) * phasors[column]
                                })
                                .sum::<Complex64>();
                        }
                        by_twin.push(-total);
                    }
                    at.push((cotangent, by_twin));
                }
                Ok(at)
            },
        )?;
        let mut sums = vec![vec![Complex64::default(); twins.len()]; cotangents.len()];
        for (cotangent, terms) in found.into_iter().flatten() {
            for (total, term) in sums[cotangent].iter_mut().zip(terms) {
                *total += term;
            }
        }
        Ok(sums)
    }

    /// Each direction's derivatives along the period `found`, which took `steps`, whose
    /// shooting's bordered system is `system`: `forward`, carried over the period's steps
    /// once from the start held to the period's end, then again from the start's own
    /// derivatives, and with an oscillator's period's.
    pub(crate) fn along_period(
        &mut self,
        (found, steps, system): Period<'_>,
        mut forward: Forward,
        options: &Options,
    ) -> Result<Forward, SimulationError> {
        let waveforms = (&found.waveforms, steps);
        let unknowns = self.elaborated.unknowns.len();
        forward.ends = Some(Vec::new());
        self.replay(&mut forward, waveforms, options)?;
        // (Φ − I) ẋ₀ + ∂φ/∂T Ṫ = −∂φ/∂p t, in shooting's units, bordered for an
        // oscillator by the phase condition's row.
        let (matrix, tolerances, scale) = system;
        let mut columns: Vec<Vec<f64>> = forward
            .ends
            .take()
            .unwrap_or_default()
            .into_iter()
            .map(|end| {
                let mut column: Vec<f64> = end
                    .iter()
                    .zip(tolerances)
                    .map(|(derivative, tolerance)| -derivative / tolerance)
                    .collect();
                column.extend(scale.map(|_| 0.0));
                column
            })
            .collect();
        solve_each_in_range(matrix, &mut columns).map_err(|error| self.state.solver.name(error))?;
        forward.stretch = scale.map(|scale| {
            let derivatives = columns.iter().map(|column| column[unknowns] * scale);
            (found.period, derivatives.collect())
        });
        forward.start = Start::Held(Some(
            columns
                .iter()
                .map(|column| {
                    column
                        .iter()
                        .zip(tolerances)
                        .map(|(scaled, tolerance)| scaled * tolerance)
                        .collect()
                })
                .collect(),
        ));
        self.replay(&mut forward, waveforms, options)?;
        Ok(forward)
    }

    /// Carries `forward` over the points of `result`, which `steps` reached, as the
    /// transient carried it when it took them.
    pub(crate) fn replay(
        &mut self,
        forward: &mut Forward,
        (result, steps): (&TransientResult, &[(Formula, bool)]),
        options: &Options,
    ) -> Result<(), SimulationError> {
        self.start_forward(forward, &result.values[0], options)?;
        for (point, (formula, restarted)) in steps.iter().enumerate() {
            let at = (result.times[point + 1], result.values[point + 1].as_slice());
            self.advance_forward(forward, at, (formula, *restarted), options)?;
        }
        Ok(())
    }

    /// The values on every twin of `cotangents` on the period `found`, which took
    /// `steps`, whose shooting's bordered system is `system`, each term to `fold`: the
    /// adjoint over the period, the start held, the transpose of the bordered solve, and
    /// the adjoint over the period again from its end.
    fn pull_period(
        &mut self,
        twins: &mut [Twin],
        (found, steps, system): Period<'_>,
        cotangents: &[Cotangent<f64>],
        fold: Fold<'_>,
        options: &Options,
    ) -> Result<(), SimulationError> {
        let (matrix, tolerances, scale) = system;
        let waveforms = (&found.waveforms, steps);
        let seeds: Vec<&BTreeMap<usize, Vec<f64>>> = cotangents
            .iter()
            .map(|cotangent| &cotangent.points)
            .collect();
        let held = Start::Held(None);
        let stretch = scale.map(|_| found.period);
        let ends = self.backward(twins, waveforms, &seeds, (&held, stretch), fold, options)?;
        // The bordered solve's map, column by column, whose transpose carries each
        // cotangent on the start and the period to the period's end.
        let rows = tolerances.len() + usize::from(scale.is_some());
        let mut map = units(rows);
        solve_each_in_range(matrix, &mut map).map_err(|error| self.state.solver.name(error))?;
        let last = found.waveforms.values.len() - 1;
        let ended: Vec<BTreeMap<usize, Vec<f64>>> = ends
            .iter()
            .zip(cotangents)
            .map(|((start, period), cotangent)| {
                let mut scaled: Vec<f64> = start
                    .iter()
                    .zip(tolerances)
                    .map(|(value, tolerance)| value * tolerance)
                    .collect();
                if let Some(scale) = scale {
                    scaled.push((period + cotangent.period) * scale);
                }
                let end = tolerances
                    .iter()
                    .zip(&map)
                    .map(|(tolerance, column)| {
                        let dot: f64 = column.iter().zip(&scaled).map(|(m, w)| m * w).sum();
                        -dot / tolerance
                    })
                    .collect();
                BTreeMap::from([(last, end)])
            })
            .collect();
        let seeds: Vec<&BTreeMap<usize, Vec<f64>>> = ended.iter().collect();
        self.backward(twins, waveforms, &seeds, (&held, None), fold, options)?;
        Ok(())
    }

    /// Starts the direct method at the state `x` a transient starts from: each
    /// direction's derivative of it, by the DC Jacobian at the operating point, with the
    /// rows of the nets forced there held, or as given for a state held; and of the
    /// charges there.
    pub(crate) fn start_forward(
        &mut self,
        forward: &mut Forward,
        x: &[f64],
        options: &Options,
    ) -> Result<(), SimulationError> {
        let shifts = match &forward.start {
            Start::Held(Some(shifts)) => shifts.clone(),
            Start::Held(None) => vec![vec![0.0; x.len()]; forward.directions.len()],
            Start::Solved(forced) => {
                self.dc_shifts(&mut forward.twins, x, &forward.directions, forced, options)?
            }
        };
        let at = forward.keep(&shifts);
        // A state that only time moves holds at the operating point, and has a charge
        // only once time passes.
        self.evaluate(x, Some(0.0), (1.0, options.gmin), false);
        forward.charges = forward
            .stretch
            .as_ref()
            .map(|_| Tangent::new(self.state.workspace.contributions.charge.clone()));
        forward.tangents.clear();
        let inside = options.parallelism.inside(x.len());
        for twin in &mut forward.twins {
            twin.evaluate(x, Some(0.0), options.gmin);
        }
        for (direction, moved) in forward.directions.iter().zip(&shifts) {
            let mut charge = combined(&forward.twins, direction, x.len(), Part::Charge);
            let capacitances = &self.state.workspace.contributions.capacitance;
            add_product(
                &self.elaborated.pattern,
                (capacitances, moved),
                &mut charge,
                inside,
            );
            forward.tangents.push(Tangent::new(charge));
        }
        forward.found = vec![at];
        Ok(())
    }

    /// Carries the direct method over the step a transient accepted at `time`, its solution
    /// `x`, by `formula`; `restarted` says whether the integration restarts after it. While
    /// an oscillator's period stretches, each direction's history also takes the charges'
    /// terms the steps divide, `α q_{n+1} + β₀ q_n + β₁ q_{n-1}`, times `−Ṫ / T`.
    pub(crate) fn advance_forward(
        &mut self,
        forward: &mut Forward,
        (time, x): (f64, &[f64]),
        (formula, restarted): (&Formula, bool),
        options: &Options,
    ) -> Result<(), SimulationError> {
        self.evaluate(x, Some(time), (1.0, options.gmin), false);
        let inside = options.parallelism.inside(x.len());
        let workspace = &mut self.state.workspace;
        let contributions = &workspace.contributions;
        step_matrix(
            &mut workspace.jacobian,
            contributions,
            formula.alpha,
            inside,
        );
        let mut factors = self.state.solver.factor(&self.state.workspace.jacobian)?;
        let charge_now = &self.state.workspace.contributions.charge;
        let stretched = match (&forward.stretch, &forward.charges) {
            (Some((period, derivatives)), Some(charges)) => {
                let divided = charges.divided(charge_now, formula);
                Some((divided, period, derivatives))
            }
            _ => None,
        };
        for twin in &mut forward.twins {
            twin.evaluate(x, Some(time), options.gmin);
        }
        let mut solutions = Vec::with_capacity(forward.directions.len());
        for (index, (direction, tangent)) in forward
            .directions
            .iter()
            .zip(&mut forward.tangents)
            .enumerate()
        {
            let residual = combined(&forward.twins, direction, x.len(), Part::Residual);
            let partial_charge = combined(&forward.twins, direction, x.len(), Part::Charge);
            let mut history = tangent.history(formula);
            if let Some((divided, period, derivatives)) = &stretched {
                let by = derivatives[index] / *period;
                for (term, divided) in history.iter_mut().zip(divided) {
                    *term -= by * divided;
                }
            }
            let mut moved: Vec<f64> = residual
                .iter()
                .zip(&partial_charge)
                .zip(&history)
                .map(|((residual, charge), term)| -(residual + formula.alpha * charge + term))
                .collect();
            factors.solve(&mut moved)?;
            let mut charge = partial_charge;
            let capacitances = &self.state.workspace.contributions.capacitance;
            add_product(
                &self.elaborated.pattern,
                (capacitances, &moved),
                &mut charge,
                inside,
            );
            tangent.advance(charge, &history, formula, restarted);
            solutions.push(moved);
        }
        if let Some(charges) = &mut forward.charges {
            let history = charges.history(formula);
            let charge = self.state.workspace.contributions.charge.clone();
            charges.advance(charge, &history, formula, restarted);
        }
        forward.found.push(forward.keep(&solutions));
        if forward.ends.is_some() {
            forward.ends = Some(solutions);
        }
        Ok(())
    }

    /// The adjoint method over the waveforms `result`, which `steps` reached after their
    /// first point, from the last point back to the first, for every cotangent of `seeds`
    /// at once, each a value on every unknown at the points where it is not zero: each
    /// term of each one's value on every twin to `fold`, by its index. The derivatives
    /// start at `start`; while an oscillator's period `stretch` stretches, each step's
    /// charge terms scale with it. Returns, for each cotangent, its derivatives with
    /// respect to the state held at the start, zero for a start solved, and to the period.
    ///
    /// At the point `k`, where the cotangent is `c_k`, with `μ = λ_q + α λ_r`, and `+ λ_q'`
    /// where the integration restarts after the point:
    ///
    /// ```text
    /// (G + α C)ᵀ λ = c_k + Cᵀ μ,          G and C evaluated at x_k
    /// p̄ += −λᵀ ∂f/∂p + (μ − α λ)ᵀ ∂q/∂p
    /// λ_q ← β₀ (λ_r − λ), and + λ_q' where it does not restart
    /// λ_q' ← β₁ (λ_r − λ),  λ_r ← γ (λ_r − λ)
    /// ```
    ///
    /// At the start, where the charges' derivative is zero and the point before is the
    /// point itself, `μ = λ_q + λ_q'` and `p̄ += μᵀ ∂q/∂p`; at an operating point, the DC
    /// Jacobian, its forced rows held, takes the step's place, `Jᵀ λ = c_0 + Cᵀ μ`, and
    /// `p̄ += −λᵀ ∂F/∂p`; a state held instead has the derivative `c_0 + Cᵀ μ`. A
    /// stretching period's terms `−(Ṫ / T) (α q_k + β₀ q_{k-1} + β₁ q'_{k-1})` in each
    /// step's history take `T̄ += −(λ_r − λ)ᵀ (…) / T`, carried back to the charges they
    /// read as `λ_q` and `λ_q'` are.
    pub(crate) fn backward(
        &mut self,
        twins: &mut [Twin],
        (result, steps): (&TransientResult, &[(Formula, bool)]),
        seeds: &[&BTreeMap<usize, Vec<f64>>],
        (start, stretch): (&Start, Option<f64>),
        fold: Fold<'_>,
        options: &Options,
    ) -> Result<Ends, SimulationError> {
        let dimension = self.elaborated.unknowns.len();
        let gmin = options.gmin;
        let inside = options.parallelism.inside(dimension);
        let zeros = || vec![0.0; dimension];
        let mut adjoints: Vec<Adjoint> = seeds
            .iter()
            .map(|seed| Adjoint {
                charge: zeros(),
                before: zeros(),
                rate: zeros(),
                stretch: stretch.map(|_| (zeros(), zeros())),
                period: 0.0,
                last: seed.keys().next_back().copied(),
            })
            .collect();
        for (point, &(formula, restarted)) in steps.iter().enumerate().rev() {
            let point = point + 1;
            let (time, x) = (result.times[point], &result.values[point]);
            self.evaluate(x, Some(time), (1.0, gmin), false);
            let workspace = &mut self.state.workspace;
            let capacitance = &workspace.contributions.capacitance;
            let active: Vec<usize> = (0..adjoints.len())
                .filter(|&index| adjoints[index].last.is_some_and(|last| last >= point))
                .collect();
            let mut solved = Vec::with_capacity(active.len());
            for &index in &active {
                let adjoint = &mut adjoints[index];
                // The derivative with respect to the charges at the point, all told, and
                // what goes on to those at the point before it.
                let mut held = std::mem::take(&mut adjoint.charge);
                let onward = if restarted {
                    add(&mut held, 1.0, &adjoint.before);
                    zeros()
                } else {
                    std::mem::replace(&mut adjoint.before, zeros())
                };
                add(&mut held, formula.alpha, &adjoint.rate);
                let mut solution = zeros();
                if let Some(values) = seeds[index].get(&point) {
                    add(&mut solution, 1.0, values);
                }
                add_transposed_product(&self.elaborated.pattern, capacitance, &held, &mut solution);
                solved.push((held, onward, solution));
            }
            step_matrix(
                &mut workspace.jacobian,
                &workspace.contributions,
                formula.alpha,
                inside,
            );
            let mut factors = self.state.solver.factor(&self.state.workspace.jacobian)?;
            for (_, _, solution) in &mut solved {
                factors.solve_transpose(solution)?;
            }
            let charge = &self.state.workspace.contributions.charge;
            let mut shifted = Vec::with_capacity(active.len());
            for (&index, (held, onward, solution)) in active.iter().zip(&solved) {
                let stepped = (formula, restarted, charge.as_slice());
                shifted.push(adjoints[index].step_back((held, onward, solution), stepped, stretch));
            }
            let terms = options.parallelism.spread(
                twins,
                || (),
                |(), twin| {
                    twin.evaluate(x, Some(time), gmin);
                    shifted
                        .iter()
                        .zip(&solved)
                        .map(|(moved, (_, _, solution))| {
                            (twin.charge_dot(moved), -twin.residual_dot(solution))
                        })
                        .collect::<Vec<(f64, f64)>>()
                },
            );
            for (column, &index) in active.iter().enumerate() {
                let (charged, resistive): (Vec<f64>, Vec<f64>) =
                    terms.iter().map(|by_active| by_active[column]).unzip();
                fold(index, &charged);
                fold(index, &resistive);
            }
        }
        self.backward_start(
            twins,
            result,
            seeds,
            adjoints,
            (start, stretch),
            fold,
            options,
        )
    }

    /// The adjoint method at a transient's start, the point `0` of `result`, from what
    /// `adjoints` carried back to it, as [`Simulator::backward`] says.
    #[expect(
        clippy::too_many_arguments,
        reason = "the start reads the twins, the waveforms, the cotangents and what the \
                  steps carried back, where the derivatives start, and the options, and \
                  folds the terms, as the backward pass it ends does"
    )]
    fn backward_start(
        &mut self,
        twins: &mut [Twin],
        result: &TransientResult,
        seeds: &[&BTreeMap<usize, Vec<f64>>],
        adjoints: Vec<Adjoint>,
        (start, stretch): (&Start, Option<f64>),
        fold: Fold<'_>,
        options: &Options,
    ) -> Result<Ends, SimulationError> {
        let gmin = options.gmin;
        let x = &result.values[0];
        self.evaluate(x, Some(0.0), (1.0, gmin), false);
        let capacitance = &self.state.workspace.contributions.capacitance;
        let charge = &self.state.workspace.contributions.charge;
        let mut held_all = Vec::with_capacity(adjoints.len());
        let mut solutions = Vec::with_capacity(adjoints.len());
        let mut periods = Vec::with_capacity(adjoints.len());
        for (seed, adjoint) in seeds.iter().zip(adjoints) {
            let mut held = adjoint.charge;
            add(&mut held, 1.0, &adjoint.before);
            let mut solution = vec![0.0; x.len()];
            if let Some(values) = seed.get(&0) {
                add(&mut solution, 1.0, values);
            }
            add_transposed_product(&self.elaborated.pattern, capacitance, &held, &mut solution);
            let mut period = adjoint.period;
            if let (Some((mut read, before)), Some(stretched)) = (adjoint.stretch, stretch) {
                add(&mut read, 1.0, &before);
                let dot: f64 = read.iter().zip(charge).map(|(a, q)| a * q).sum();
                period -= dot / stretched;
            }
            held_all.push(held);
            solutions.push(solution);
            periods.push(period);
        }
        let charged = options.parallelism.spread(
            twins,
            || (),
            |(), twin| {
                twin.evaluate(x, Some(0.0), gmin);
                held_all
                    .iter()
                    .map(|held| twin.charge_dot(held))
                    .collect::<Vec<f64>>()
            },
        );
        for (index, terms) in transposed(charged, seeds.len()).iter().enumerate() {
            fold(index, terms);
        }
        let Start::Solved(forced) = start else {
            return Ok(solutions.into_iter().zip(periods).collect());
        };
        self.evaluate(x, None, (1.0, gmin), false);
        let mut matrix = self.state.workspace.contributions.conductance.clone();
        for unknown in forced {
            force_row(&self.elaborated, &mut matrix, unknown.index());
        }
        let mut factors = self.state.solver.factor(&matrix)?;
        for solution in &mut solutions {
            factors.solve_transpose(solution)?;
            for unknown in forced {
                solution[unknown.index()] = 0.0;
            }
        }
        let resistive = options.parallelism.spread(
            twins,
            || (),
            |(), twin| {
                twin.evaluate(x, None, gmin);
                solutions
                    .iter()
                    .map(|solution| -twin.residual_dot(solution))
                    .collect::<Vec<f64>>()
            },
        );
        for (index, terms) in transposed(resistive, seeds.len()).iter().enumerate() {
            fold(index, terms);
        }
        let zeros = vec![0.0; x.len()];
        Ok(periods
            .into_iter()
            .map(|period| (zeros.clone(), period))
            .collect())
    }
}

/// The derivatives of the unknowns `kept` of `moved`, by direction then unknown, by
/// unknown kept then direction.
fn kept_of(kept: &[usize], moved: &[Vec<f64>]) -> Vec<Vec<f64>> {
    kept.iter()
        .map(|&unknown| moved.iter().map(|shift| shift[unknown]).collect())
        .collect()
}
