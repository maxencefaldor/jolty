//! Newton iteration on the circuit's equations.

use jolty_models::{Contributions, Input, Nature};
use jolty_numerics::{Pattern, reduce};
use rayon::prelude::*;

use crate::elaborated::Elaborated;
use crate::options::UnknownGrain;
use crate::profile::{Activity, Meter};
use crate::solver::{Factors, Solver};
use crate::{Options, SimulationError, Simulator};

/// Multiplier on `f64::EPSILON * (|J| * |x|)` below which a residual counts as numerical
/// zero.
///
/// Grounded in floating-point inner-product rounding analysis (Higham, *Accuracy and
/// Stability of Numerical Algorithms*, 2nd ed., 2002, ch. 3). A factor of `8` absorbs KCL
/// stencil cancellation without masking unconverged residuals.
const RESIDUAL_ROUNDINGS: f64 = 8.0;

/// Maximum number of worst-offending unknowns named in convergence or singularity errors.
///
/// Chosen as a concise diagnostic formatting limit (source: `docs/guide/simulation.md`).
pub(crate) const NAMED_UNKNOWNS: usize = 3;

/// Constant shift `sigma = 1` in the shifted deflation multiplier `1 / ||x - r||^2 + sigma`,
/// recovering unmodified equations far from known roots `r`.
///
/// Source: Farrell, Birkisson, and Funke, "Deflation techniques for finding distinct
/// solutions of nonlinear partial differential equations", *SIAM Journal on Scientific
/// Computing* 37(4), 2015 (`p = 2`, `sigma = 1`).
const SHIFT: f64 = 1.0;

/// Computes the floating-point rounding floor for a residual whose row magnitude is `resolution`
/// (`|J| * |x|`).
pub(crate) fn rounding(resolution: f64) -> f64 {
    RESIDUAL_ROUNDINGS * f64::EPSILON * resolution
}

/// Computes the magnitude of `step` for an unknown of `nature` ending at `value`, normalized
/// by its tolerance.
fn step_error(value: f64, step: f64, nature: Nature, options: &Options) -> f64 {
    step.abs() / options.unknown_tolerance(nature, value, value - step)
}

/// Applies `each` to contiguous chunks of `values` with each chunk's starting index.
///
/// Runs sequentially as a single chunk when `grain` is `None`; otherwise dispatches chunks
/// of [`reduce::CHUNK`] elements across the current Rayon pool with at least
/// [`UnknownGrain::chunks`] chunks per task.
pub(crate) fn each_chunk<T: Send>(
    values: &mut [T],
    grain: Option<UnknownGrain>,
    each: impl Fn(usize, &mut [T]) + Sync,
) {
    let Some(grain) = grain else {
        each(0, values);
        return;
    };
    values
        .par_chunks_mut(reduce::CHUNK)
        .with_min_len(grain.chunks)
        .enumerate()
        .for_each(|(index, chunk)| each(index * reduce::CHUNK, chunk));
}

/// Applies `each` across aligned chunks of `N` equal-length `planes` and reduces the chunk
/// results with `join` starting from identity `none`.
///
/// The `join` operation must be associative and commutative (such as `&&` or [`largest`]) so
/// parallel reductions stay bit-identical across thread counts.
fn each_chunk_of<const N: usize, R: Copy + Send + Sync>(
    planes: [&mut [f64]; N],
    grain: Option<UnknownGrain>,
    none: R,
    join: impl Fn(R, R) -> R + Sync + Send,
    each: impl Fn(usize, [&mut [f64]; N]) -> R + Sync,
) -> R {
    let Some(grain) = grain else {
        return each(0, planes);
    };
    let whole = Planes { first: 0, planes };
    rayon::iter::split(whole, |part| part.halve(grain.chunks))
        .map(|part| each(part.first, part.planes))
        .reduce(|| none, join)
}

/// Returns the larger of `one` and `other`, propagating NaN deterministically as a
/// commutative-associative reduction for [`each_chunk_of`].
fn largest(one: f64, other: f64) -> f64 {
    if one.is_nan() || one >= other {
        one
    } else {
        other
    }
}

/// Aligned slices across `N` planes starting at index `first`.
struct Planes<'a, const N: usize> {
    first: usize,
    planes: [&'a mut [f64]; N],
}

impl<const N: usize> Planes<'_, N> {
    /// Splits the planes at a multiple of [`reduce::CHUNK`] when both halves hold at least
    /// `least` chunks, returning `(self, None)` otherwise.
    fn halve(self, least: usize) -> (Self, Option<Self>) {
        let count = self.planes.first().map_or(0, |plane| plane.len());
        let chunks = count.div_ceil(reduce::CHUNK);
        if chunks < 2 * least.max(1) {
            return (self, None);
        }
        let at = chunks / 2 * reduce::CHUNK;
        let Self { first, mut planes } = self;
        let later = planes.each_mut().map(|plane| {
            let (head, tail) = std::mem::take(plane).split_at_mut(at);
            *plane = head;
            tail
        });
        (
            Self { first, planes },
            Some(Self {
                first: first + at,
                planes: later,
            }),
        )
    }
}

/// Checks whether `holds` returns true for every index in `0..count`.
///
/// Runs sequentially when `grain` is `None`, or in Rayon tasks of at least
/// [`UnknownGrain::chunks`] chunks of [`reduce::CHUNK`] indices otherwise.
pub(crate) fn all(
    count: usize,
    grain: Option<UnknownGrain>,
    holds: impl Fn(usize) -> bool + Sync + Send,
) -> bool {
    match grain {
        None => (0..count).all(holds),
        Some(grain) => (0..count)
            .into_par_iter()
            .with_min_len(grain.chunks * reduce::CHUNK)
            .all(holds),
    }
}

/// Evaluates `each` on every fixed [`reduce::CHUNK`]-sized subrange covering `0..count` in
/// index order.
///
/// Runs sequentially when `grain` is `None`, or across the current Rayon pool with at least
/// [`UnknownGrain::chunks`] ranges per task. Fixed chunk boundaries keep subsequent folds
/// bit-identical across thread counts.
pub(crate) fn pieces<R: Send>(
    count: usize,
    grain: Option<UnknownGrain>,
    each: impl Fn(std::ops::Range<usize>) -> R + Sync + Send,
) -> Vec<R> {
    let range = |piece: usize| piece * reduce::CHUNK..count.min((piece + 1) * reduce::CHUNK);
    let pieces = count.div_ceil(reduce::CHUNK);
    match grain {
        None => (0..pieces).map(|piece| each(range(piece))).collect(),
        Some(grain) => (0..pieces)
            .into_par_iter()
            .with_min_len(grain.chunks)
            .map(|piece| each(range(piece)))
            .collect(),
    }
}

/// Writes the transient step Jacobian `G + αC` from `contributions` into `matrix`,
/// parallelized according to `grain`.
pub(crate) fn step_matrix(
    matrix: &mut [f64],
    contributions: &Contributions,
    alpha: f64,
    grain: Option<UnknownGrain>,
) {
    let (conductance, capacitance) = (&contributions.conductance, &contributions.capacitance);
    each_chunk(matrix, grain, |start, chunk| {
        let stored = conductance[start..].iter().zip(&capacitance[start..]);
        for (entry, (conductance, capacitance)) in chunk.iter_mut().zip(stored) {
            *entry = conductance + alpha * capacitance;
        }
    });
}

/// Builds `G + αC` from `workspace.contributions` into `workspace.jacobian` and factors
/// the matrix with `solver`.
pub(crate) fn factor_step_jacobian<'a>(
    solver: &'a mut Solver<f64>,
    workspace: &mut Workspace,
    alpha: f64,
    grain: Option<UnknownGrain>,
) -> Result<Factors<'a, f64>, SimulationError> {
    step_matrix(
        &mut workspace.jacobian,
        &workspace.contributions,
        alpha,
        grain,
    );
    solver.factor(&workspace.jacobian)
}

/// Accumulates the matrix-vector product `(pattern, values) * x` into `into` using fixed
/// column order per row, parallelized according to `grain`.
pub(crate) fn add_product(
    pattern: &Pattern,
    values: &[f64],
    x: &[f64],
    into: &mut [f64],
    grain: Option<UnknownGrain>,
) {
    each_chunk(into, grain, |start, chunk| {
        pattern.add_product_from(start, values, x, chunk);
    });
}

/// Replaces row `index` in `jacobian` with the forced-voltage equation `v - v0 = 0`
/// (unit diagonal, zero off-diagonals).
pub(crate) fn force_row(elaborated: &Elaborated, jacobian: &mut [f64], index: usize) {
    for &(slot, _) in elaborated.pattern.row(index) {
        jacobian[slot] = 0.0;
    }
    let (_, entry) = elaborated.node_diagonals[index];
    jacobian[elaborated.pattern.slot(entry)] = 1.0;
}

/// Sums `count` terms produced by `term` in [`reduce::sum`]'s fixed tree order, using
/// Rayon when `grain` is set.
fn sum(count: usize, term: impl Fn(usize) -> f64 + Sync, grain: Option<UnknownGrain>) -> f64 {
    match grain {
        None => reduce::sum(count, term),
        Some(grain) => reduce::sum_in_parallel(count, term, grain.chunks),
    }
}

/// Reusable evaluation buffers allocated once per simulator alongside its work meter.
#[derive(Debug)]
pub(crate) struct Workspace {
    /// Device contributions; after [`Simulator::combine`], `residual` holds the full system
    /// residual.
    pub(crate) contributions: Contributions,
    /// System Jacobian indexed by pattern slot, overwritten each Newton iteration and reused
    /// for step-control factorizations between iterations.
    pub(crate) jacobian: Vec<f64>,
    /// Per-equation sensitivity `|J| * |x|` bounding floating-point residual resolution.
    resolution: Vec<f64>,
    /// Most recent Newton step vector.
    step: Vec<f64>,
    /// Reusable buffer for transient divided charge differences and their filtered form.
    pub(crate) differences: Vec<f64>,
    /// Reusable buffer holding scaled divided differences solved through the step Jacobian.
    pub(crate) moved: Vec<f64>,
    /// Iteration counter and optional phase profiler.
    pub(crate) meter: Meter,
}

impl Workspace {
    /// Allocates zeroed evaluation buffers for `elaborated` with a fresh work meter.
    pub(crate) fn new(elaborated: &Elaborated) -> Self {
        let (unknowns, stored) = (elaborated.unknowns.len(), elaborated.pattern.len());
        Self {
            contributions: Contributions::new(&elaborated.pattern),
            jacobian: vec![0.0; stored],
            resolution: vec![0.0; unknowns],
            step: vec![0.0; unknowns],
            differences: Vec::new(),
            moved: Vec::new(),
            meter: Meter::default(),
        }
    }

    /// Zeroes all workspace buffers and resets the work meter.
    pub(crate) fn reset(&mut self) {
        let Contributions {
            residual,
            scale,
            charge,
            conductance,
            capacitance,
        } = &mut self.contributions;
        for values in [
            residual,
            scale,
            charge,
            conductance,
            capacitance,
            &mut self.jacobian,
            &mut self.resolution,
            &mut self.step,
        ] {
            values.fill(0.0);
        }
        self.differences.clear();
        self.moved.clear();
        self.meter = Meter::default();
    }

    /// Constructs a test workspace pre-filled with `conductance`, `capacitance`, and
    /// `jacobian` for `pattern`.
    #[cfg(test)]
    pub(crate) fn from_matrices(
        pattern: &Pattern,
        conductance: Vec<f64>,
        capacitance: Vec<f64>,
        jacobian: Vec<f64>,
    ) -> Self {
        let unknowns = pattern.dimension();
        let mut contributions = Contributions::new(pattern);
        contributions.conductance = conductance;
        contributions.capacitance = capacitance;
        Self {
            contributions,
            jacobian,
            resolution: vec![0.0; unknowns],
            step: vec![0.0; unknowns],
            differences: Vec::new(),
            moved: Vec::new(),
            meter: Meter::default(),
        }
    }
}

/// Equation system solved by a Newton iteration.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Mode<'a> {
    /// DC system `f(x) = 0` with source scaling `source_scale`, optional node shunt `shunt`
    /// toward `anchor` (or ground), and deflation against `deflated` roots.
    Dc {
        source_scale: f64,
        shunt: f64,
        anchor: Option<&'a [f64]>,
        deflated: &'a [Vec<f64>],
    },
    /// Transient step system `f(x) + α q(x) + history = 0` at `time`.
    Step {
        time: f64,
        alpha: f64,
        history: &'a [f64],
    },
}

impl<'a> Mode<'a> {
    /// Builds an unshunted DC system at full source scale deflated by `deflated`.
    pub(crate) fn direct(deflated: &'a [Vec<f64>]) -> Self {
        Self::Dc {
            source_scale: 1.0,
            shunt: 0.0,
            anchor: None,
            deflated,
        }
    }
}

/// Reason a Newton iteration terminated without converging.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Failure {
    /// Sparse solver failure during factorization or solve (singular Jacobian or memory limit).
    Solver(SimulationError),
    /// Iteration limit exhausted before meeting convergence tolerances.
    Diverged,
}

impl Simulator {
    /// Evaluates every device at `x` (at `time` for transient runs, or `None` for DC),
    /// populating workspace residuals, charges, and derivatives.
    ///
    /// Applies voltage limiting when `limiting` is true and reports whether any device
    /// limited its step. Kernel groups accumulate contributions in a fixed order regardless
    /// of thread count.
    pub(crate) fn evaluate(
        &mut self,
        x: &[f64],
        time: Option<f64>,
        source_scale: f64,
        gmin: f64,
        limiting: bool,
    ) -> bool {
        let previous = self.state.workspace.meter.enter(Activity::Evaluating);
        let state = &mut self.state;
        let mut input = Input::new(x, time, gmin);
        input.source_scale = source_scale;
        input.limiting = limiting;
        if time.is_some() {
            input.past = state.past.as_ref();
        }
        // The check evaluates each instance alone, from what it remembered before.
        #[cfg(test)]
        if let Some(check) = &mut state.check {
            check.last = Some((x.to_vec(), time, source_scale, gmin));
        }
        #[cfg(test)]
        let before = state.check.is_some().then(|| state.runs.clone());
        let limited = state.runs.evaluate(
            &self.elaborated.groups,
            &input,
            &mut state.workspace.contributions,
            state.tasks,
        );
        #[cfg(test)]
        if let (Some(check), Some(mut alone)) = (&mut state.check, before) {
            let evaluated = crate::kernels::Evaluated {
                input: &input,
                limited,
                assembled: &state.workspace.contributions,
                slab: state.runs.slab(),
            };
            check.compare(&self.elaborated, &mut alone, evaluated);
        }
        state.workspace.meter.leave(previous);
        limited
    }

    /// Solves the equation system specified by `mode` via Newton iteration from `x`,
    /// returning the iteration count and leaving the solution and workspace state in place.
    ///
    /// Convergence requires the latest step to satisfy unknown tolerances, every residual to
    /// meet equation tolerances, and no device voltage limiting on the final evaluation.
    ///
    /// Each run takes at most `limit` steps and evaluates the final iterate under the same
    /// criteria. Step size is the maximum ratio of any unknown update to its tolerance; DC
    /// solves accept steps up to 1.0 as in SPICE. Transient iterations also estimate the
    /// contraction rate `θ` between consecutive unlimited steps. Stopping then requires the
    /// Deuflhard bound `θ / (1 - θ) * step <= 1.0` (Deuflhard, *Newton Methods for Nonlinear
    /// Problems*, 2004, §2.1; Hairer and Wanner, *Solving Ordinary Differential Equations II*,
    /// 2nd ed., 1996, §IV.8). First steps, limited steps, and non-contracting steps fall back
    /// to the direct DC step criterion.
    pub(crate) fn newton(
        &mut self,
        x: &mut [f64],
        mode: Mode<'_>,
        options: &Options,
        limit: usize,
    ) -> Result<usize, Failure> {
        let (time, source_scale) = match mode {
            Mode::Dc { source_scale, .. } => (None, source_scale),
            Mode::Step { time, .. } => (Some(time), 1.0),
        };
        // The devices evaluate, and the work on each unknown and each stored entry splits,
        // as the options spread them, the same bits either way.
        let instances = self.elaborated.groups.evaluated();
        self.state.tasks = options.parallelism.evaluation(instances);
        let grain = options.parallelism.unknown_grain(x.len());
        // A DC solution starts afresh from its guess; a time step goes on from the last.
        let afresh = matches!(mode, Mode::Dc { .. });
        let mut stepped_within_tolerance = false;
        // The size of the last step that moved, which a time step's contraction is
        // measured against.
        let mut moved: Option<f64> = None;
        let mut iteration = 0;
        loop {
            // The iterate after the last step allowed is evaluated only to be judged.
            if iteration == limit && !stepped_within_tolerance {
                return Err(Failure::Diverged);
            }
            let limiting = iteration > 0 || !afresh;
            let limited = self.evaluate(x, time, source_scale, options.gmin, limiting);
            // The residuals are judged only once the last step was within tolerance and no
            // device limited its voltage.
            let judged = (stepped_within_tolerance && !limited).then_some(options);
            if self.combine(x, mode, judged, grain) {
                return Ok(iteration);
            }
            if iteration == limit {
                return Err(Failure::Diverged);
            }
            // The solver keeps the factors of a Jacobian bitwise unchanged since the last, as
            // a linear circuit's is from one iteration to the next, and refactors a changed
            // one with the last pivots.
            let Workspace {
                jacobian,
                step,
                meter,
                ..
            } = &mut self.state.workspace;
            meter.iterate();
            let previous = meter.enter(Activity::Factoring);
            let solved = self.state.solver.factor(jacobian).and_then(|mut factors| {
                meter.enter(Activity::Solving);
                factors.solve(step)
            });
            meter.leave(previous);
            solved.map_err(Failure::Solver)?;
            if let Mode::Dc { deflated, .. } = mode {
                self.deflate(x, deflated, grain);
            }
            let largest = self.take_step(x, options, grain);
            iteration += 1;
            // An iteration that contracts by `rate` from one step to the next lies within
            // `rate/(1 - rate)` times its last step of the solution. A DC solution's steps,
            // a time step's first, a step from a limited evaluation or after one, and a
            // step no shorter than the one before are judged alone, as SPICE judges them.
            let rate = moved
                .filter(|_| !afresh && !limited)
                .map(|before| largest / before);
            let distance = match rate {
                Some(rate) if rate < 1.0 => rate / (1.0 - rate) * largest,
                _ => largest,
            };
            stepped_within_tolerance = distance <= 1.0;
            // A step that did not move gives no contraction to measure the next by.
            moved = (largest > 0.0 && !limited).then_some(largest);
        }
    }

    /// Scales the Newton step of `f(x) = 0` into the deflated step of `m(x) * f(x) = 0`, where
    /// `m(x) = Π (1 / ||x - r||^2 + SHIFT)` over `known` roots `r` in node-voltage space.
    ///
    /// Applying the scalar factor `1 / (1 - ∇m·δ / m)` repels iteration from known roots
    /// (Farrell, Birkisson, and Funke, *SIAM Journal on Scientific Computing* 37(4), 2015).
    fn deflate(&mut self, x: &[f64], known: &[Vec<f64>], grain: Option<UnknownGrain>) {
        let step = &mut self.state.workspace.step;
        let natures = &self.elaborated.natures;
        let mut slope = 0.0;
        for root in known {
            // Each term over the node voltages, zero for the others.
            let apart = |index: usize| match natures[index] {
                Nature::Voltage => x[index] - root[index],
                _ => 0.0,
            };
            let distance = sum(x.len(), |index| apart(index) * apart(index), grain);
            let along = sum(x.len(), |index| apart(index) * step[index], grain);
            if distance > 0.0 {
                // ∇(1/d² + SHIFT)·δ = -2 (x - r)·δ / d⁴.
                slope += -2.0 * along / (distance * distance) / (distance.recip() + SHIFT);
            }
        }
        let scale = (1.0 - slope).recip();
        if scale.is_finite() {
            each_chunk(step, grain, |_, chunk| {
                for value in chunk {
                    *value *= scale;
                }
            });
        }
    }

    /// Assembles the residual, Jacobian, and Newton right-hand side for `mode`, parallelized
    /// according to `grain`.
    ///
    /// When `judged` provides options, checks whether every residual meets its equation
    /// tolerance or floating-point rounding floor, returning `false` otherwise. Row residuals,
    /// `|J| * |x|` resolution bounds, convergence checks, and negated right-hand sides run in
    /// one pass after matrix assembly.
    fn combine(
        &mut self,
        x: &[f64],
        mode: Mode<'_>,
        judged: Option<&Options>,
        grain: Option<UnknownGrain>,
    ) -> bool {
        let Workspace {
            contributions,
            jacobian,
            resolution,
            step,
            meter,
            ..
        } = &mut self.state.workspace;
        let previous = meter.enter(Activity::Assembling);
        let reactive = match mode {
            Mode::Dc {
                source_scale,
                shunt,
                anchor,
                ..
            } => {
                let conductance = &contributions.conductance;
                each_chunk(jacobian, grain, |start, chunk| {
                    chunk.copy_from_slice(&conductance[start..start + chunk.len()]);
                });
                if shunt > 0.0 {
                    for &(unknown, entry) in &self.elaborated.node_diagonals {
                        let index = unknown.index();
                        jacobian[self.elaborated.pattern.slot(entry)] += shunt;
                        let held = anchor.map_or(0.0, |anchor| anchor[index]);
                        let current = shunt * (x[index] - held);
                        contributions.residual[index] += current;
                        let scale = &mut contributions.scale[index];
                        *scale = scale.max(current.abs());
                    }
                }
                // A forced node's equation is its voltage's, `v - v₀ = 0`, in place of its
                // currents' sum: an ideal source holds it, whose current the sum would be.
                for &(unknown, voltage) in &self.state.forced {
                    let index = unknown.index();
                    force_row(&self.elaborated, jacobian, index);
                    let target = source_scale * voltage;
                    contributions.residual[index] = x[index] - target;
                    contributions.scale[index] = x[index].abs().max(target.abs());
                }
                None
            }
            Mode::Step { alpha, history, .. } => {
                step_matrix(jacobian, contributions, alpha, grain);
                Some((alpha, history))
            }
        };
        let Contributions {
            residual,
            scale,
            charge,
            ..
        } = contributions;
        let (pattern, jacobian, natures) = (
            &self.elaborated.pattern,
            &*jacobian,
            &self.elaborated.natures,
        );
        let charge = &*charge;
        let within = each_chunk_of(
            [residual, scale, resolution, step],
            grain,
            true,
            |one, other| one && other,
            |start, [residual, scale, resolution, step]| {
                let mut within = true;
                let rows = residual.iter_mut().zip(scale.iter_mut());
                let rows = rows.zip(resolution.iter_mut().zip(step.iter_mut()));
                for (offset, ((residual, scale), (resolution, step))) in rows.enumerate() {
                    let index = start + offset;
                    if let Some((alpha, history)) = reactive {
                        let reactive = alpha * charge[index] + history[index];
                        *residual += reactive;
                        *scale = scale.max(reactive.abs());
                    }
                    // Each row's terms add in the order of their columns, as the pattern
                    // stores them.
                    *resolution = pattern
                        .row(index)
                        .iter()
                        .fold(0.0, |total, &(slot, column)| {
                            total + (jacobian[slot] * x[column]).abs()
                        });
                    let rounding = rounding(*resolution);
                    if let Some(options) = judged {
                        let absolute = options.equation_tolerance(natures[index]);
                        within &= residual.abs()
                            <= options.relative_tolerance * *scale + absolute + rounding;
                    }
                    // Residuals within the floating-point rounding floor of the row are
                    // numerical noise. Chasing them drives unbounded tiny steps when a small
                    // source current sits alongside megavolt node terms.
                    *step = if residual.abs() <= rounding {
                        0.0
                    } else {
                        -*residual
                    };
                }
                within
            },
        );
        meter.leave(previous);
        judged.is_some() && within
    }

    /// Advances `x` by the solved Newton step and returns the largest tolerance-normalized
    /// unknown update (or `NaN` if any entry is `NaN`), parallelized according to `grain`.
    fn take_step(&self, x: &mut [f64], options: &Options, grain: Option<UnknownGrain>) -> f64 {
        let (step, natures) = (&self.state.workspace.step, &self.elaborated.natures);
        each_chunk_of([x], grain, 0.0, largest, |start, [chunk]| {
            let mut most = 0.0;
            for (offset, value) in chunk.iter_mut().enumerate() {
                let index = start + offset;
                *value += step[index];
                let error = step_error(*value, step[index], natures[index], options);
                most = largest(most, error);
            }
            most
        })
    }

    /// Returns the names of the `count` unknowns whose last step was largest relative to
    /// tolerance.
    pub(crate) fn furthest_from_converging(
        &self,
        x: &[f64],
        options: &Options,
        count: usize,
    ) -> Vec<String> {
        let mut errors: Vec<(f64, usize)> = (0..x.len())
            .map(|index| {
                let (step, nature) = (
                    self.state.workspace.step[index],
                    self.elaborated.natures[index],
                );
                (step_error(x[index], step, nature, options), index)
            })
            .collect();
        errors.sort_by(|a, b| b.0.total_cmp(&a.0));
        errors
            .into_iter()
            .take(count)
            .map(|(_, index)| self.elaborated.unknowns[index].name.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::Path;
    use std::time::Duration;

    use jolty_numerics::Work;
    use serde_json::{Value, json};

    use super::{Failure, Mode};
    use crate::test_helpers::{grounded_simulator, simulator_from_json};
    use crate::{Method, Options, Simulator, Transient};

    /// Builds a simulator for a voltage source with `waveform` driving `stages` RC sections
    /// (1 kΩ series, 1 nF to ground), optionally terminated by a clamping diode if `clamped`.
    fn ladder(stages: usize, waveform: &Value, clamped: bool) -> Simulator {
        let node = |index: usize| format!("n{index}");
        let mut instances = vec![json!({
            "name": "V1", "of": {"model": "voltage_source"},
            "parameters": {"waveform": waveform},
            "terminals": {"positive": node(0), "negative": "ground"}
        })];
        for stage in 0..stages {
            instances.push(json!({
                "name": format!("R{stage}"), "of": {"model": "resistor"},
                "parameters": {"resistance": 1e3},
                "terminals": {"a": node(stage), "b": node(stage + 1)}
            }));
            instances.push(json!({
                "name": format!("C{stage}"), "of": {"model": "capacitor"},
                "parameters": {"capacitance": 1e-9},
                "terminals": {"a": node(stage + 1), "b": "ground"}
            }));
        }
        if clamped {
            instances.push(json!({
                "name": "D1", "of": {"model": "diode"}, "parameters": {},
                "terminals": {"anode": node(stages), "cathode": "ground"}
            }));
        }
        grounded_simulator("ladder", "ground", &Value::Array(instances))
    }

    /// Builds a simulator with a behavioral current source (`expression`) from `a` to ground
    /// reading `v(a)` and a grounded node `c` at 0 V.
    fn root(expression: &str) -> Simulator {
        grounded_simulator(
            "root",
            "ground",
            &json!([
                {"name": "V1", "of": {"model": "voltage_source"},
                 "parameters": {"waveform": {"dc": 0.0}},
                 "terminals": {"positive": "c", "negative": "ground"}},
                {"name": "B1", "of": {"model": "behavioral_current_source"},
                 "parameters": {"expression": expression},
                 "terminals": {"positive": "a", "negative": "ground", "a": "a", "c": "c"}}
            ]),
        )
    }

    /// Runs a single transient Newton solve on `simulator` from `v(a) = 1 V` with iteration
    /// cap `limit`, returning the iteration result and final `v(a)`.
    fn stepped(simulator: &mut Simulator, limit: usize) -> (Result<usize, Failure>, f64) {
        let a = simulator.unknown("a").expect("a node").index();
        let unknowns = simulator.elaborated.unknowns.len();
        let mut x = vec![0.0; unknowns];
        x[a] = 1.0;
        // Nothing holds a charge, so the formula's weight and history change nothing.
        let history = vec![0.0; unknowns];
        let mode = Mode::Step {
            time: 0.0,
            alpha: 1.0,
            history: &history,
        };
        let found = simulator.newton(&mut x, mode, &Options::default(), limit);
        (found, x[a])
    }

    #[test]
    fn a_time_step_takes_no_more_steps_than_its_limit() {
        // At the double root of `1e-3 * v(a)^2`, Newton halves `a` from 1 V each step and
        // falls within tolerance after 20 steps. A limit of 10 stops early despite steady
        // contraction.
        let mut simulator = root("1e-3 * v(a)^2 - v(c)");
        assert_eq!(stepped(&mut simulator, 10).0, Err(Failure::Diverged));
        assert_eq!(stepped(&mut simulator, 20).0, Ok(20));
    }

    #[test]
    fn a_time_step_converges_within_its_tolerance_of_a_slowly_contracting_root() {
        // At the triple root of `1e-3 * v(a)^3`, Newton contracts by 2/3 each step, leaving
        // twice the last step to go. SPICE's step-only test would stop at twice tolerance
        // from the root at 0 V.
        let mut simulator = root("1e-3 * v(a)^3 - v(c)");
        let (found, a) = stepped(&mut simulator, 100);
        assert!(found.is_ok(), "{found:?}");
        let options = Options::default();
        let tolerance = options.voltage_tolerance + options.relative_tolerance * a.abs();
        assert!(a.abs() <= tolerance, "{a} V from the root");
    }

    /// Returns `(pivot_searches, refactorizations, reuses)` elapsed between `before` and `after`.
    fn work(before: Work, after: Work) -> (usize, usize, usize) {
        (
            after.factorizations + after.fallbacks - before.factorizations - before.fallbacks,
            after.refactorizations - before.refactorizations,
            after.reuses - before.reuses,
        )
    }

    /// Verifies that static condensation eliminates MOSFET displacements and low-resistance
    /// branch currents before factorization, matching ngspice's node-only matrix dimensions
    /// (504 of 3,504 unknowns for the 500-inverter chain, and 13 of 23 for the buck converter).
    #[test]
    fn the_solver_factors_the_nodes_alone() {
        let circuits =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/speed/circuits");
        for (name, unknowns, factored) in [("inverter_chain_500", 3_504, 504), ("buck", 23, 13)] {
            let text = fs::read_to_string(circuits.join(format!("{name}.json"))).expect("a file");
            let simulator = simulator_from_json(&text);
            let elaborated = &simulator.elaborated;
            assert_eq!(elaborated.unknowns.len(), unknowns, "{name}");
            let kept = elaborated.ordering.factored().dimension();
            assert_eq!(kept, factored, "{name}");
        }
    }

    #[test]
    fn a_linear_operating_point_factors_its_jacobian_once() {
        // The first iteration steps onto the solution and the second, confirming it, finds
        // the same Jacobian: of the two factorizations every iteration once made, the
        // second keeps the factors of the first.
        let mut simulator = ladder(5, &json!({"dc": 1.0}), false);
        simulator.operating_point(&Options::default()).unwrap();
        let statistics = simulator.state.solver.statistics();
        assert_eq!(work(Work::default(), statistics), (1, 0, 1));
    }

    #[test]
    fn a_nonlinear_operating_point_refactors_with_its_first_pivots() {
        let mut simulator = ladder(3, &json!({"dc": 1.0}), true);
        simulator.operating_point(&Options::default()).unwrap();
        let (searched, refactored, _) = work(Work::default(), simulator.state.solver.statistics());
        assert_eq!(searched, 1);
        assert!(refactored >= 2, "{refactored} refactorizations");
    }

    #[test]
    fn a_linear_transient_factors_only_as_its_step_or_formula_changes() {
        // A sine wave keeps the ladder changing so every step runs at least one iteration.
        // The Jacobian `G + αC` changes only when `α` changes (step size or integration
        // formula). Factorizations therefore occur during initial step growth up to `max_step`,
        // on the final truncated step, on rejected steps, on the backward-Euler-to-trapezoidal
        // transition, and twice per attempt at the second step when judging the first.
        let sine = json!({"sine": {"offset": 0.0, "amplitude": 1.0, "frequency": 1e3}});
        let mut simulator = ladder(5, &sine, false);
        let options = Options::default();
        simulator.operating_point(&options).unwrap();
        let before = simulator.state.solver.statistics();
        let transient = Transient {
            stop: 1e-3,
            max_step: Some(1e-6),
            method: Method::Trapezoidal,
            breakpoints: Vec::new(),
            initial_conditions: BTreeMap::new(),
            operating_point: true,
        };
        let waveforms = simulator.transient(&transient, &options).unwrap();
        let (searched, refactored, reused) = work(before, simulator.state.solver.statistics());
        let steps = &waveforms.statistics;
        assert!(steps.accepted >= 1000, "{} steps", steps.accepted);
        assert!(steps.iterations >= steps.accepted);
        assert!(searched + refactored + reused >= steps.iterations);
        // Each step's length, read from the times to within their rounding, a few units in
        // the last place of the stop.
        let (longest, rounding) = (1e-6, 4.0 * f64::EPSILON * 1e-3);
        let lengths: Vec<f64> = waveforms
            .times
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .collect();
        let at_longest = |length: &f64| (length - longest).abs() < rounding;
        let ramp = lengths.iter().position(at_longest).unwrap_or(lengths.len());
        assert!(lengths[ramp..lengths.len() - 1].iter().all(at_longest));
        // Step lengths up to `max_step`, the `max_step` plateau, and the final step.
        let changes = ramp + 2;
        let formulas = 1;
        // Each attempt at the second step judges the first step under its formula and then
        // judges itself, requiring two factorizations per attempt (including rejections).
        let judgements = 2 * (1 + steps.rejected);
        assert!(
            searched + refactored <= changes + steps.rejected + formulas + judgements,
            "{searched} + {refactored} factorizations over {} steps, {ramp} before the longest",
            steps.accepted
        );
    }

    #[test]
    fn a_transient_counts_its_work_and_measures_its_time_only_when_asked() {
        let sine = json!({"sine": {"offset": 0.0, "amplitude": 1.0, "frequency": 1e3}});
        let transient = Transient {
            stop: 1e-3,
            max_step: Some(1e-5),
            method: Method::Trapezoidal,
            breakpoints: Vec::new(),
            initial_conditions: BTreeMap::new(),
            operating_point: true,
        };
        let plain = ladder(3, &sine, true)
            .transient(&transient, &Options::default())
            .unwrap();
        let profiling = Options {
            profile: true,
            ..Options::default()
        };
        let mut profiled = ladder(3, &sine, true)
            .transient(&transient, &profiling)
            .unwrap();
        // Total work includes operating-point iterations and rejected steps in addition to
        // accepted steps, with one factorization or reuse per iteration.
        let (steps, work) = (&plain.statistics, plain.statistics.work);
        assert!(work.iterations > steps.iterations, "{work:?} for {steps:?}");
        let factored = work.factorizations + work.refactorizations + work.fallbacks;
        assert!(factored + work.reuses >= work.iterations, "{work:?}");
        assert_eq!(work.factorizations, 1);
        // Measuring the time changes nothing else, and the activities sum to the whole.
        assert_eq!(plain.statistics.profile, None);
        let profile = profiled.statistics.profile.take().expect("a profile");
        assert_eq!(profiled, plain);
        assert!(profile.evaluating > Duration::ZERO, "{profile:?}");
        assert!(profile.factoring > Duration::ZERO, "{profile:?}");
        assert!(profile.step_control > Duration::ZERO, "{profile:?}");
        let activities = [
            profile.preparing,
            profile.evaluating,
            profile.assembling,
            profile.factoring,
            profile.solving,
            profile.step_control,
            profile.recording,
            profile.other,
        ];
        assert_eq!(activities.iter().sum::<Duration>(), profile.total());
    }
}
