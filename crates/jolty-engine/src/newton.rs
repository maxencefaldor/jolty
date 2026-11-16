//! Newton iteration on the circuit's equations.

use jolty_models::{Contributions, Input, Nature};
use jolty_numerics::{Pattern, reduce};
use rayon::prelude::*;

use crate::elaborated::Elaborated;
use crate::options::Inside;
use crate::profile::{Activity, Meter};
use crate::{Options, SimulationError, Simulator};

/// How many roundings of the unknowns a residual may show and still count as zero.
///
/// Each unknown is stored to half a unit in its last place, so a residual cannot resolve
/// changes finer than `ε·(|J|·|x|)` of its row, and each term it sums adds a rounding or two
/// of its own. Without this floor, an equation whose terms cancel, such as the current
/// through a milliohm between two nodes at hundreds of volts, could never converge.
const RESIDUAL_ROUNDINGS: f64 = 8.0;

/// How many unknowns a failure to converge names.
pub(crate) const NAMED_UNKNOWNS: usize = 3;

/// The shift of the deflation `1/‖x − r‖² + SHIFT` of each known root `r`: far from every
/// root, where the distance in volts grows, the deflated equations become the plain ones.
const SHIFT: f64 = 1.0;

/// The floating-point resolution of a residual whose equation's `|J|·|x|` is `resolution`:
/// the rounding of the unknowns it reads, which no step can resolve.
fn rounding(resolution: f64) -> f64 {
    RESIDUAL_ROUNDINGS * f64::EPSILON * resolution
}

/// The step `step` of an unknown of `nature` that the step left at `value`, as a multiple
/// of its tolerance.
fn step_error((value, step): (f64, f64), nature: Nature, options: &Options) -> f64 {
    let magnitude = value.abs().max((value - step).abs());
    let absolute = options.value_tolerance(nature);
    step.abs() / (options.relative_tolerance * magnitude + absolute)
}

/// Does `each` for every chunk of `values` and the index of its first value: for one chunk
/// of them all on the calling thread without `inside`, and else for each chunk of
/// [`reduce::CHUNK`] values, on the threads of the current rayon pool as they take them,
/// at least [`Inside::chunks`] a task.
pub(crate) fn each_chunk<T: Send>(
    values: &mut [T],
    inside: Option<Inside>,
    each: impl Fn(usize, &mut [T]) + Sync,
) {
    let Some(inside) = inside else {
        each(0, values);
        return;
    };
    values
        .par_chunks_mut(reduce::CHUNK)
        .with_min_len(inside.chunks)
        .enumerate()
        .for_each(|(index, chunk)| each(index * reduce::CHUNK, chunk));
}

/// Does `each` for every chunk of the `N` `planes`, of one length, and the index of their
/// first value, as [`each_chunk`] does for one, and returns what it returned of every
/// chunk joined by `join`, from `none`, each chunk done whatever the others return.
///
/// `join` must give the same whatever order and grouping it joins the chunks' findings
/// in, as `&&` and the largest do, so that the finding is the same bits on any number of
/// threads.
fn each_chunk_of<const N: usize, R: Copy + Send + Sync>(
    planes: [&mut [f64]; N],
    inside: Option<Inside>,
    (none, join): (R, impl Fn(R, R) -> R + Sync + Send),
    each: impl Fn(usize, [&mut [f64]; N]) -> R + Sync,
) -> R {
    let Some(inside) = inside else {
        return each(0, planes);
    };
    let whole = Planes { first: 0, planes };
    rayon::iter::split(whole, |part| part.halve(inside.chunks))
        .map(|part| each(part.first, part.planes))
        .reduce(|| none, join)
}

/// The larger of `one` and `other`, or NaN if either is: a join of
/// [`each_chunk_of`]'s, as it gives the same in any order and grouping.
fn largest(one: f64, other: f64) -> f64 {
    if one.is_nan() || one >= other {
        one
    } else {
        other
    }
}

/// Consecutive values of `N` planes, from the index `first` on.
struct Planes<'a, const N: usize> {
    first: usize,
    planes: [&'a mut [f64]; N],
}

impl<const N: usize> Planes<'_, N> {
    /// The values split in two at a multiple of [`reduce::CHUNK`] values from the first,
    /// each half at least `least` chunks, or whole with `None` when they fill fewer than
    /// twice as many.
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

/// Whether `holds` holds of every index below `count`, asked on the calling thread without
/// `inside`, and else in tasks of at least [`Inside::chunks`] chunks of [`reduce::CHUNK`]
/// indices on the threads of the current rayon pool: the same answer either way.
pub(crate) fn all(
    count: usize,
    inside: Option<Inside>,
    holds: impl Fn(usize) -> bool + Sync + Send,
) -> bool {
    match inside {
        None => (0..count).all(holds),
        Some(inside) => (0..count)
            .into_par_iter()
            .with_min_len(inside.chunks * reduce::CHUNK)
            .all(holds),
    }
}

/// What `each` finds of each of the ranges [`reduce::CHUNK`] long that the indices below
/// `count` fall into, in their order, on the calling thread without `inside`, and else on
/// the threads of the current rayon pool, at least [`Inside::chunks`] ranges a task: the
/// ranges are the same however many threads take them, so a fold of the findings in their
/// order is the same bits on any number.
pub(crate) fn pieces<R: Send>(
    count: usize,
    inside: Option<Inside>,
    each: impl Fn(std::ops::Range<usize>) -> R + Sync + Send,
) -> Vec<R> {
    let range = |piece: usize| piece * reduce::CHUNK..count.min((piece + 1) * reduce::CHUNK);
    let pieces = count.div_ceil(reduce::CHUNK);
    match inside {
        None => (0..pieces).map(|piece| each(range(piece))).collect(),
        Some(inside) => (0..pieces)
            .into_par_iter()
            .with_min_len(inside.chunks)
            .map(|piece| each(range(piece)))
            .collect(),
    }
}

/// Writes into `matrix` the matrix `G + αC` of the equations of a time step whose formula
/// weighs the new charges by `alpha`, from the conductances and capacitances of
/// `contributions`, its stored entries split as `inside` says.
pub(crate) fn step_matrix(
    matrix: &mut [f64],
    contributions: &Contributions,
    alpha: f64,
    inside: Option<Inside>,
) {
    let (conductance, capacitance) = (&contributions.conductance, &contributions.capacitance);
    each_chunk(matrix, inside, |start, chunk| {
        let stored = conductance[start..].iter().zip(&capacitance[start..]);
        for (entry, (conductance, capacitance)) in chunk.iter_mut().zip(stored) {
            *entry = conductance + alpha * capacitance;
        }
    });
}

/// Adds to `into` the product of the matrix of `pattern` whose stored values are `values`
/// and the vector `x`, each row's terms in the order of their columns, its rows split as
/// `inside` says: the same bits however they are.
pub(crate) fn add_product(
    pattern: &Pattern,
    (values, x): (&[f64], &[f64]),
    into: &mut [f64],
    inside: Option<Inside>,
) {
    each_chunk(into, inside, |start, chunk| {
        pattern.add_product_from(start, values, x, chunk);
    });
}

/// The sum of the `count` terms `term` gives, by [`reduce::sum`]'s fixed order, on the
/// calling thread without `inside`, and else on the threads of the current rayon pool, at
/// least [`Inside::chunks`] chunks a task.
fn sum(count: usize, term: impl Fn(usize) -> f64 + Sync, inside: Option<Inside>) -> f64 {
    match inside {
        None => reduce::sum(count, term),
        Some(inside) => reduce::sum_in_parallel(count, term, inside.chunks),
    }
}

/// The buffers one evaluation of the circuit fills, allocated once per simulator, and the
/// meter of the work they see done.
#[derive(Debug)]
pub(crate) struct Workspace {
    /// What the devices added; after [`Simulator::combine`], the residual is the full
    /// residual of the system being solved.
    pub(crate) contributions: Contributions,
    /// The Jacobian of the system being solved, by pattern slot; once a Newton iteration
    /// has ended, the space where the matrix of a step's equations is written to be
    /// factored, as the next iteration writes it again.
    pub(crate) jacobian: Vec<f64>,
    /// `|J|·|x|` of each equation: how much its residual moves when every unknown moves by
    /// its own magnitude, which bounds the residual's floating-point resolution.
    resolution: Vec<f64>,
    /// The last Newton step.
    step: Vec<f64>,
    /// The charges' divided differences of the step a transient judges last, and then
    /// those filtered through its equations, kept so that no step allocates them.
    pub(crate) differences: Vec<f64>,
    /// The differences times the formula's coefficient, solved through the step's
    /// equations, as the filter of the last judged step left them.
    pub(crate) moved: Vec<f64>,
    /// The Newton iterations run, and the time spent while an analysis profiles.
    pub(crate) meter: Meter,
}

impl Workspace {
    /// Zeroed buffers of the system of `elaborated`, and a meter that has counted nothing.
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

    /// Zeroes every buffer and the meter, as [`Workspace::new`] makes them.
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
}

/// The system a Newton iteration solves.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Mode<'a> {
    /// The DC equations `f(x) = 0`, with sources scaled by `source_scale` and a conductance
    /// `shunt` from every node to the voltage `anchor` holds it at, or to the reference
    /// without one, deflated by the solutions `deflated` already found, whose basins the
    /// iteration is pushed out of.
    Dc {
        source_scale: f64,
        shunt: f64,
        anchor: Option<&'a [f64]>,
        deflated: &'a [Vec<f64>],
    },
    /// The equations of a time step to `time`, `f(x) + α q(x) + history = 0`.
    Step {
        time: f64,
        alpha: f64,
        history: &'a [f64],
    },
}

/// Why a Newton iteration stopped without converging.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Failure {
    /// The solver could not factor the Jacobian, or solve for its step: the error says
    /// whether it was singular, naming the unknowns it leaves undetermined, or too large.
    Solver(SimulationError),
    /// The iteration took every step its limit allows without converging.
    Diverged,
}

impl Simulator {
    /// Evaluates every device at `x`, at `time` in a transient analysis or, with `None`, at
    /// an operating point or about one, filling the workspace's residuals, charges and
    /// their derivatives, with their steps limited if `limiting`. Returns whether a device
    /// limited its voltage.
    ///
    /// The groups of the instances add their contributions in the order of the groups, of
    /// their instances and of each instance's local unknowns and entries, however many
    /// threads evaluate them.
    pub(crate) fn evaluate(
        &mut self,
        x: &[f64],
        time: Option<f64>,
        (source_scale, gmin): (f64, f64),
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
            let assembled = (&state.workspace.contributions, state.runs.slab());
            check.compare(&self.elaborated, &mut alone, (&input, limited), assembled);
        }
        state.workspace.meter.leave(previous);
        limited
    }

    /// Solves the system of `mode` by Newton iteration from `x`, leaving the solution in `x`
    /// and the workspace evaluated there. Returns the number of iterations.
    ///
    /// The iteration has converged when the last step was within tolerance of the unknowns,
    /// every residual is within tolerance of the terms it sums, and no device limited its
    /// voltage.
    ///
    /// It takes at most `limit` steps, the iterate after the last judged as any other. A
    /// step's size is its largest move of an unknown against that unknown's tolerance. A
    /// DC solution's step is within tolerance when its size is at most 1, as SPICE judges
    /// it. A time step's iteration measures its contraction `θ`, each step's size over the
    /// one before it, between steps from evaluations no device limited, as a limited step
    /// is the limiter's rather than Newton's: as an iteration that contracts by `θ` lies
    /// within `θ/(1 − θ)` times its last step of the solution (Deuflhard, "Newton Methods
    /// for Nonlinear Problems", 2004, section 2.1), its step is within tolerance once that
    /// distance is, as Hairer and Wanner stop the Newton iteration of an implicit step
    /// ("Solving Ordinary Differential Equations II", 2nd edition, 1996, section IV.8);
    /// its first step, one from a limited evaluation or after one, and one no shorter
    /// than the one before, are judged as a DC solution's.
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
        let inside = options.parallelism.inside(x.len());
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
            let limited = self.evaluate(x, time, (source_scale, options.gmin), limiting);
            // The residuals are judged only once the last step was within tolerance and no
            // device limited its voltage.
            let judged = (stepped_within_tolerance && !limited).then_some(options);
            if self.combine(x, mode, judged, inside) {
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
                self.deflate(x, deflated, inside);
            }
            let largest = self.take_step(x, options, inside);
            iteration += 1;
            // An iteration that contracts by `rate` from one step to the next lies within
            // `rate/(1 − rate)` times its last step of the solution. A DC solution's steps,
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

    /// Turns the Newton step of `f(x) = 0` at `x` into that of the deflated equations
    /// `m(x)·f(x) = 0`, with `m(x) = Π (1/‖x − r‖² + SHIFT)` over the `known` solutions
    /// `r`, the distance taken over the node voltages: the step scaled by
    /// `1 / (1 − ∇m·δ / m)`, which is infinite at each known solution, so the iteration
    /// leaves it for another (Farrell, Birkisson and Funke, "Deflation techniques for
    /// finding distinct solutions of nonlinear partial differential equations", SIAM
    /// Journal on Scientific Computing 37(4), 2015).
    fn deflate(&mut self, x: &[f64], known: &[Vec<f64>], inside: Option<Inside>) {
        let step = &mut self.state.workspace.step;
        let natures = &self.elaborated.natures;
        let mut slope = 0.0;
        for root in known {
            // Each term over the node voltages, zero for the others.
            let apart = |index: usize| match natures[index] {
                Nature::Voltage => x[index] - root[index],
                _ => 0.0,
            };
            let distance = sum(x.len(), |index| apart(index) * apart(index), inside);
            let along = sum(x.len(), |index| apart(index) * step[index], inside);
            if distance > 0.0 {
                // ∇(1/d² + SHIFT)·δ = −2 (x − r)·δ / d⁴.
                slope += -2.0 * along / (distance * distance) / (distance.recip() + SHIFT);
            }
        }
        let scale = (1.0 - slope).recip();
        if scale.is_finite() {
            each_chunk(step, inside, |_, chunk| {
                for value in chunk {
                    *value *= scale;
                }
            });
        }
    }

    /// Turns the evaluated device contributions into the residual and Jacobian of `mode`,
    /// and the residual into the right-hand side of the Newton step, its work split as
    /// `inside` says. Returns whether every residual is within tolerance of the terms it
    /// sums, or within the rounding of the unknowns it depends on, when `judged` gives the
    /// options to judge them by, and else `false`: a node's residual is a current, a
    /// branch's a voltage.
    ///
    /// The matrix is found first, then each row's residual, the resolution `|J|·|x|` of its
    /// equation, its judgement and its right-hand side, in one pass over the rows.
    fn combine(
        &mut self,
        x: &[f64],
        mode: Mode<'_>,
        judged: Option<&Options>,
        inside: Option<Inside>,
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
                each_chunk(jacobian, inside, |start, chunk| {
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
                // A forced node's equation is its voltage's, `v − v₀ = 0`, in place of its
                // currents' sum: an ideal source holds it, whose current the sum would be.
                for &(unknown, voltage) in &self.state.forced {
                    let index = unknown.index();
                    for &(slot, _) in self.elaborated.pattern.row(index) {
                        jacobian[slot] = 0.0;
                    }
                    let (_, entry) = self.elaborated.node_diagonals[index];
                    jacobian[self.elaborated.pattern.slot(entry)] = 1.0;
                    let target = source_scale * voltage;
                    contributions.residual[index] = x[index] - target;
                    contributions.scale[index] = x[index].abs().max(target.abs());
                }
                None
            }
            Mode::Step { alpha, history, .. } => {
                step_matrix(jacobian, contributions, alpha, inside);
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
            inside,
            (true, |one, other| one && other),
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
                    // A residual within the rounding of the unknowns its equation reads is
                    // noise: the step it would drive is one the unknowns cannot record, and
                    // chasing it moves the others without end, as a source's current beside
                    // a node at megavolts.
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

    /// Moves `x` by the step solved, and returns the largest of its moves of the unknowns,
    /// each as a multiple of its tolerance, or NaN if one is, in one pass over the unknowns
    /// split as `inside` says: the step is within tolerance when it is at most 1.
    fn take_step(&self, x: &mut [f64], options: &Options, inside: Option<Inside>) -> f64 {
        let (step, natures) = (&self.state.workspace.step, &self.elaborated.natures);
        each_chunk_of([x], inside, (0.0, largest), |start, [chunk]| {
            let mut most = 0.0;
            for (offset, value) in chunk.iter_mut().enumerate() {
                let index = start + offset;
                *value += step[index];
                let error = step_error((*value, step[index]), natures[index], options);
                most = largest(most, error);
            }
            most
        })
    }

    /// The names of the `count` unknowns whose last step was largest against its tolerance.
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
                (step_error((x[index], step), nature, options), index)
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

    use jolty_ir::Circuit;
    use jolty_models::Library;
    use jolty_numerics::Work;
    use serde_json::{Value, json};

    use super::{Failure, Mode};
    use crate::{Conditions, Method, Options, Simulator, Transient};

    /// A simulator of a voltage source of `waveform` driving `stages` sections of 1 kΩ in
    /// series and 1 nF to the ground, with a diode from the last to the ground if `clamped`.
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
        let document = json!({
            "format": "jolty-circuit", "version": 4, "top": "ladder",
            "components": {"ladder": {
                "ports": [{"name": "ground", "role": "ground"}],
                "instances": instances
            }}
        });
        let circuit = Circuit::from_json(&document.to_string()).expect("a circuit");
        Simulator::new(&circuit, &Library::standard(), &Conditions::default())
            .expect("an elaborated circuit")
    }

    /// A simulator of a behavioral current source of `expression` from `a` to the ground,
    /// which reads `v(a)` and the voltage of `c`, which a voltage source holds at 0 V.
    fn root(expression: &str) -> Simulator {
        let document = json!({
            "format": "jolty-circuit", "version": 4, "top": "root",
            "components": {"root": {
                "ports": [{"name": "ground", "role": "ground"}],
                "instances": [
                    {"name": "V1", "of": {"model": "voltage_source"},
                     "parameters": {"waveform": {"dc": 0.0}},
                     "terminals": {"positive": "c", "negative": "ground"}},
                    {"name": "B1", "of": {"model": "behavioral_current_source"},
                     "parameters": {"expression": expression},
                     "terminals": {"positive": "a", "negative": "ground", "a": "a", "c": "c"}}
                ]
            }}
        });
        let circuit = Circuit::from_json(&document.to_string()).expect("a circuit");
        Simulator::new(&circuit, &Library::standard(), &Conditions::default())
            .expect("an elaborated circuit")
    }

    /// A time step's Newton iteration on the circuit of `simulator`, from `a` at 1 V,
    /// allowed `limit` steps: what it returned, and where it left `a`.
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
        // At the double root of 1e-3·v(a)², Newton's iteration halves `a` at each step,
        // from 1 V, and its step falls within its tolerance, 1e-6 V and a thousandth of
        // `a`, after 20 steps: a limit of 10 stops it, though it contracts throughout.
        let mut simulator = root("1e-3 * v(a)^2 - v(c)");
        assert_eq!(stepped(&mut simulator, 10).0, Err(Failure::Diverged));
        assert_eq!(stepped(&mut simulator, 20).0, Ok(20));
    }

    #[test]
    fn a_time_step_converges_within_its_tolerance_of_a_slowly_contracting_root() {
        // At the triple root of 1e-3·v(a)³, Newton's iteration contracts by two thirds at
        // each step, so a step within tolerance leaves twice as much to go: SPICE's test
        // on the step alone would stop with `a` twice its tolerance from the root, 0 V.
        let mut simulator = root("1e-3 * v(a)^3 - v(c)");
        let (found, a) = stepped(&mut simulator, 100);
        assert!(found.is_ok(), "{found:?}");
        let options = Options::default();
        let tolerance = options.voltage_tolerance + options.relative_tolerance * a.abs();
        assert!(a.abs() <= tolerance, "{a} V from the root");
    }

    /// The factorizations that searched for pivots, from the first or from a pivot too
    /// small, the refactorizations and the reuses of factors, from `before` to `after`.
    fn work(before: Work, after: Work) -> (usize, usize, usize) {
        (
            after.factorizations + after.fallbacks - before.factorizations - before.fallbacks,
            after.refactorizations - before.refactorizations,
            after.reuses - before.reuses,
        )
    }

    /// The sparse solver eliminates a MOSFET's three displacements before it factors, and
    /// the currents through its series resistances below 1 Ω, so it factors the circuit's
    /// nodes alone, as ngspice's matrix holds them: 504 of the 3,504 unknowns of the chain
    /// of 500 inverters, and 13 of the buck's 23, its switches' internal nodes among them.
    #[test]
    fn the_solver_factors_the_nodes_alone() {
        let circuits =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/speed/circuits");
        for (name, unknowns, factored) in [("inverter_chain_500", 3_504, 504), ("buck", 23, 13)] {
            let text = fs::read_to_string(circuits.join(format!("{name}.json"))).expect("a file");
            let circuit = Circuit::from_json(&text).expect("a circuit");
            let simulator = Simulator::new(&circuit, &Library::standard(), &Conditions::default())
                .expect("an elaborated circuit");
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
        // A sine wave keeps the ladder moving, so that every step takes an iteration at
        // least, each of which once made a factorization. The Jacobian, `G + αC`, now changes
        // only with `α`, as the step's length or formula changes: while the first steps grow
        // to the longest, a millionth of a second, on the last, landing on the stop, on each
        // step tried and rejected, once as the formula turns from backward Euler, which
        // starts the run, to the trapezoidal rule, and twice on each attempt at the second
        // step, as it judges the first.
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
        let steps = waveforms.statistics;
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
        // The lengths up to the longest, the longest's own, and the last's.
        let changes = ramp + 2;
        let formulas = 1;
        // The start is the run's one corner: each attempt at the second step judges the
        // first through the first's formula, then itself through its own, each with a
        // factorization, and a rejected step may be one of those attempts.
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
        // The work counts the operating point's iterations and the rejected steps' beside
        // the accepted steps', and each iteration factors its Jacobian once.
        let (steps, work) = (plain.statistics, plain.statistics.work);
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
