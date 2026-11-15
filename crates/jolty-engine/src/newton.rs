//! Newton iteration on the circuit's equations.

use jolty_models::{Contributions, Input, Quantity};
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

/// The step `step` of an unknown of `quantity` that the step left at `value`, as a multiple
/// of its tolerance.
fn step_error((value, step): (f64, f64), quantity: Quantity, options: &Options) -> f64 {
    let magnitude = value.abs().max((value - step).abs());
    let absolute = options.value_tolerance(quantity);
    step.abs() / (options.relative_tolerance * magnitude + absolute)
}

/// Does `each` for every chunk of `values` and the index of its first value: for one chunk
/// of them all on the calling thread without `inside`, and else for each chunk of
/// [`reduce::CHUNK`] values, on the threads of the current rayon pool as they take them.
pub(crate) fn each_chunk<T: Send>(
    values: &mut [T],
    inside: Option<Inside>,
    each: impl Fn(usize, &mut [T]) + Sync,
) {
    if inside.is_none() {
        each(0, values);
        return;
    }
    values
        .par_chunks_mut(reduce::CHUNK)
        .enumerate()
        .for_each(|(index, chunk)| each(index * reduce::CHUNK, chunk));
}

/// Does `each` for every chunk of the `N` `planes`, of one length, and the index of their
/// first value, as [`each_chunk`] does for one, and returns whether it returned `true` of
/// every chunk, each chunk done whatever the others return.
fn each_chunk_of<const N: usize>(
    planes: [&mut [f64]; N],
    inside: Option<Inside>,
    each: impl Fn(usize, [&mut [f64]; N]) -> bool + Sync,
) -> bool {
    if inside.is_none() {
        return each(0, planes);
    }
    let whole = Planes { first: 0, planes };
    rayon::iter::split(whole, Planes::halve)
        .map(|part| each(part.first, part.planes))
        .reduce(|| true, |one, other| one && other)
}

/// Consecutive values of `N` planes, from the index `first` on.
struct Planes<'a, const N: usize> {
    first: usize,
    planes: [&'a mut [f64]; N],
}

impl<const N: usize> Planes<'_, N> {
    /// The values split in two at a multiple of [`reduce::CHUNK`] values from the first, or
    /// whole with `None` when they fill one chunk or less.
    fn halve(self) -> (Self, Option<Self>) {
        let count = self.planes.first().map_or(0, |plane| plane.len());
        let chunks = count.div_ceil(reduce::CHUNK);
        if chunks < 2 {
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
/// `inside`, and else in chunks of [`reduce::CHUNK`] indices on the threads of the current
/// rayon pool: the same answer either way.
pub(crate) fn all(
    count: usize,
    inside: Option<Inside>,
    holds: impl Fn(usize) -> bool + Sync + Send,
) -> bool {
    match inside {
        None => (0..count).all(holds),
        Some(_) => (0..count)
            .into_par_iter()
            .with_min_len(reduce::CHUNK)
            .all(holds),
    }
}

/// What `each` finds of each of the ranges [`reduce::CHUNK`] long that the indices below
/// `count` fall into, in their order, on the calling thread without `inside`, and else on
/// the threads of the current rayon pool: the ranges are the same however many threads
/// take them, so a fold of the findings in their order is the same bits on any number.
pub(crate) fn pieces<R: Send>(
    count: usize,
    inside: Option<Inside>,
    each: impl Fn(std::ops::Range<usize>) -> R + Sync + Send,
) -> Vec<R> {
    let range = |piece: usize| piece * reduce::CHUNK..count.min((piece + 1) * reduce::CHUNK);
    let pieces = count.div_ceil(reduce::CHUNK);
    match inside {
        None => (0..pieces).map(|piece| each(range(piece))).collect(),
        Some(_) => (0..pieces)
            .into_par_iter()
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
/// calling thread without `inside`, and else on the threads of the current rayon pool.
fn sum(count: usize, term: impl Fn(usize) -> f64 + Sync, inside: Option<Inside>) -> f64 {
    match inside {
        None => reduce::sum(count, term),
        Some(_) => reduce::sum_in_parallel(count, term),
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
    /// The iteration limit was reached.
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
        let instances = self.elaborated.devices.len();
        self.state.tasks = options.parallelism.evaluation(instances);
        let inside = options.parallelism.inside(x.len());
        // A DC solution starts afresh from its guess; a time step goes on from the last.
        let afresh = matches!(mode, Mode::Dc { .. });
        let mut stepped_within_tolerance = false;
        for iteration in 0..limit {
            let limiting = iteration > 0 || !afresh;
            let limited = self.evaluate(x, time, (source_scale, options.gmin), limiting);
            // The residuals are judged only once the last step was within tolerance and no
            // device limited its voltage.
            let judged = (stepped_within_tolerance && !limited).then_some(options);
            if self.combine(x, mode, judged, inside) {
                return Ok(iteration);
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
            stepped_within_tolerance = self.take_step(x, options, inside);
        }
        Err(Failure::Diverged)
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
        let quantities = &self.elaborated.quantities;
        let mut slope = 0.0;
        for root in known {
            // Each term over the node voltages, zero for the others.
            let apart = |index: usize| match quantities[index] {
                Quantity::Voltage => x[index] - root[index],
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
        } = &mut self.state.workspace;
        let previous = meter.enter(Activity::Assembling);
        let reactive = match mode {
            Mode::Dc { shunt, anchor, .. } => {
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
        let (pattern, jacobian, quantities) = (
            &self.elaborated.pattern,
            &*jacobian,
            &self.elaborated.quantities,
        );
        let charge = &*charge;
        let within = each_chunk_of(
            [residual, scale, resolution, step],
            inside,
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
                        let absolute = options.equation_tolerance(quantities[index]);
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

    /// Moves `x` by the step solved, and returns whether the step was within tolerance of
    /// every unknown it moved, in one pass over the unknowns split as `inside` says.
    fn take_step(&self, x: &mut [f64], options: &Options, inside: Option<Inside>) -> bool {
        let (step, quantities) = (&self.state.workspace.step, &self.elaborated.quantities);
        each_chunk_of([x], inside, |start, [chunk]| {
            let mut within = true;
            for (offset, value) in chunk.iter_mut().enumerate() {
                let index = start + offset;
                *value += step[index];
                let error = step_error((*value, step[index]), quantities[index], options);
                within &= error <= 1.0;
            }
            within
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
                let (step, quantity) = (
                    self.state.workspace.step[index],
                    self.elaborated.quantities[index],
                );
                (step_error((x[index], step), quantity, options), index)
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
    use std::time::Duration;

    use jolty_ir::Circuit;
    use jolty_models::Library;
    use jolty_numerics::Statistics;
    use serde_json::{Value, json};

    use crate::{Conditions, Method, Options, Simulator, Transient};

    /// A simulator of a voltage source of `waveform` driving `stages` sections of 1 kΩ in
    /// series and 1 nF to the ground, with a diode from the last to the ground if `clamped`.
    fn ladder(stages: usize, waveform: &Value, clamped: bool) -> Simulator {
        let node = |index: usize| format!("n{index}");
        let mut instances = vec![json!({
            "id": "V1", "of": {"device": "voltage_source"},
            "parameters": {"waveform": waveform},
            "terminals": {"positive": node(0), "negative": "ground"}
        })];
        for stage in 0..stages {
            instances.push(json!({
                "id": format!("R{stage}"), "of": {"device": "resistor"},
                "parameters": {"resistance": 1e3},
                "terminals": {"a": node(stage), "b": node(stage + 1)}
            }));
            instances.push(json!({
                "id": format!("C{stage}"), "of": {"device": "capacitor"},
                "parameters": {"capacitance": 1e-9},
                "terminals": {"a": node(stage + 1), "b": "ground"}
            }));
        }
        if clamped {
            instances.push(json!({
                "id": "D1", "of": {"device": "diode"}, "parameters": {},
                "terminals": {"anode": node(stages), "cathode": "ground"}
            }));
        }
        let document = json!({
            "format": "jolty-circuit", "version": 1, "top": "ladder",
            "components": {"ladder": {
                "ports": [{"name": "ground", "role": "ground"}],
                "instances": instances
            }}
        });
        let circuit = Circuit::from_json(&document.to_string()).expect("a circuit");
        Simulator::new(&circuit, &Library::standard(), &Conditions::default())
            .expect("an elaborated circuit")
    }

    /// The factorizations that searched for pivots, from the first or from a pivot too
    /// small, the refactorizations and the reuses of factors, from `before` to `after`.
    fn work(before: Statistics, after: Statistics) -> (usize, usize, usize) {
        (
            after.factorizations + after.fallbacks - before.factorizations - before.fallbacks,
            after.refactorizations - before.refactorizations,
            after.reuses - before.reuses,
        )
    }

    #[test]
    fn a_linear_operating_point_factors_its_jacobian_once() {
        // The first iteration steps onto the solution and the second, confirming it, finds
        // the same Jacobian: of the two factorizations every iteration once made, the
        // second keeps the factors of the first.
        let mut simulator = ladder(5, &json!({"dc": 1.0}), false);
        simulator.operating_point(&Options::default()).unwrap();
        let statistics = simulator.state.solver.statistics();
        assert_eq!(work(Statistics::default(), statistics), (1, 0, 1));
    }

    #[test]
    fn a_nonlinear_operating_point_refactors_with_its_first_pivots() {
        let mut simulator = ladder(3, &json!({"dc": 1.0}), true);
        simulator.operating_point(&Options::default()).unwrap();
        let (searched, refactored, _) =
            work(Statistics::default(), simulator.state.solver.statistics());
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
            times: Vec::new(),
            periodic: false,
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
            times: Vec::new(),
            periodic: false,
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
