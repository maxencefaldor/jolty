//! DC operating-point solver with gmin stepping (node shunt and junction), source stepping,
//! pseudo-transient continuation, and deflated multi-solution search.
//!
//! Each simulator caches its latest operating point and reuses it across analyses whenever
//! held sources and DC-relevant options match. Options used only by transient runs and result
//! recording filters do not invalidate the cache because recorded quantities are evaluated at
//! the cached solution. Any future option is treated as DC-relevant until proven otherwise.
//! When only Newton tolerances or iteration limits differ, a direct-Newton solve from zero
//! continues from the cached iterate along the exact same trajectory.

use std::collections::BTreeMap;
use std::f64::consts::TAU;
use std::sync::Arc;

use jolty_models::{Input, LogicLevel, Nature, UnknownIndex};
use jolty_numerics::Complex64;
use serde::Deserialize;

use crate::logic::{Bounds, Events, ExcursionKind, detected};
use crate::newton::{Failure, Mode, NAMED_UNKNOWNS};
use crate::solver::Solver;
use crate::state::{Held, Warm};
use crate::{Options, ParameterId, Partition, RecordedValues, SimulationError, Simulator, Values};

/// The largest conductance gmin stepping starts from, in siemens (`10 mS`, i.e. `100` ohms).
///
/// Source: ngspice 47 (`src/spicelib/analysis/cktop.c`, line 189 in `dynamic_gmin` and line
/// 378 in `new_gmin`: `OldGmin = 1e-2;`).
const FIRST_STEPPED: f64 = 1e-2;
/// The largest factor gmin stepping divides the conductance by from one stage to the next.
///
/// Source: ngspice 47 (`src/spicelib/analysis/cktntask.c`, line 122:
/// `tsk->TSKgminFactor = 10;`, read in `cktop.c`, lines 188, 218 to 219, 377, 407 to 408).
/// Candidate for a future `Options` knob per `AGENTS.md`, "Knobs over opinions" (ngspice
/// exposes this as `gminfactor`).
const STEPPED_DIVISOR: f64 = 10.0;
/// The factor below which gmin stepping, having shrunk its divisor after each failure or
/// each stage that barely converged, gives up.
///
/// Source: ngspice 47 (`src/spicelib/analysis/cktop.c`, line 223:
/// `factor = MAX(sqrt (factor), 1.00005);` and line 234: `if (factor < 1.00005)`).
const MIN_STEPPED_DIVISOR: f64 = 1.00005;
/// Initial source-scale increment in source stepping (`10%` of full scale).
///
/// Doubles on success and quarters on failure, reaching `1.0` in four unconstrained steps
/// (source: continuation step schedule; compare ngspice 47 `gillespie_src` in
/// `src/spicelib/analysis/cktop.c`).
const FIRST_SOURCE_INCREMENT: f64 = 0.1;
/// Minimum source-scale increment before source stepping aborts (`0.01%` of full scale).
///
/// Allows five consecutive quarterings from `FIRST_SOURCE_INCREMENT = 0.1` and caps progress
/// at 10,000 steps across `[0, 1]` (source: continuation floor; compare ngspice 47
/// `gillespie_src` in `src/spicelib/analysis/cktop.c`).
const MIN_SOURCE_INCREMENT: f64 = 1e-4;
/// Initial diagonal damping conductance in siemens (`1 F / 1 s = 1 S`) holding each node to
/// its previous voltage in pseudo-transient continuation.
///
/// Source: Kelley and Keyes, "Convergence analysis of pseudo-transient continuation",
/// *SIAM Journal on Numerical Analysis* 35(2), 1998.
const FIRST_HELD: f64 = 1.0;
/// Maximum diagonal damping conductance in siemens (`1 F / 1 ns = 1e9 S`) before
/// pseudo-transient continuation aborts, nine decade backoffs above `FIRST_HELD`.
///
/// Source: Kelley and Keyes (1998) pseudo-transient continuation ceiling.
const MAX_HELD: f64 = 1e9;
/// Maximum step count for pseudo-transient continuation before aborting.
///
/// Bounds runtime with margin past the ~24 half-decade (`sqrt(10)`) steps from
/// `FIRST_HELD = 1 S` to default `gmin = 1e-12 S` (source: Kelley and Keyes, 1998).
const MAX_HELD_STEPS: usize = 1000;
/// Quasi-static AC probe frequency in hertz (`omega = 2 * pi rad/s`) used to measure each
/// logic driver's capacitive load via `|Im(I)| / omega`.
///
/// Source: `docs/guide/simulation.md`, chosen far below logic switching poles so driven
/// capacitances charge fully.
const LOAD_FREQUENCY: f64 = 1.0;
/// Separation threshold in multiples of unknown tolerance required for two DC solutions to
/// count as distinct.
///
/// Source: root-deduplication margin providing one decade beyond worst-case `~2x`
/// opposite-side convergence error.
const DISTINCT: f64 = 10.0;
/// Alternating node-voltage perturbation in volts (`~4 * kT/q` at `300 K`) applied to known
/// roots when searching for additional DC operating points.
///
/// Source: deflated-Newton symmetry-breaking restart perturbation large enough to escape a
/// known root without deeply saturating semiconductor junctions.
const NUDGE: f64 = 0.1;
/// Maximum analog-logic DC settling rounds before still-toggling detectors resolve to
/// `LogicLevel::Unknown`.
///
/// Source: `docs/guide/simulation.md` ("Logic"), matching `SETTLE_CHANGES = 4` in `logic.rs`
/// (two complete high/low cycles).
const SETTLE_ROUNDS: usize = 4;

/// DC operating-point solution: unknown values, recorded quantities, logic levels, and
/// convergence metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct OperatingPointResult {
    /// Unknown values in simulator order.
    pub values: Vec<f64>,
    /// Terminal currents, charges, device derivatives, and variables requested by options.
    pub recorded: RecordedValues,
    /// Settled logic net levels in simulator order.
    pub logic: Vec<LogicLevel>,
    /// Continuation strategy that converged.
    pub strategy: Strategy,
    /// Cumulative Newton iterations across all attempted strategies and logic settling rounds,
    /// matching a fresh solve regardless of cache reuse.
    pub iterations: usize,
}

/// Configuration for searching multiple DC solutions via deflated Newton iteration, up to
/// `max_solutions`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatingPoints {
    /// Maximum number of distinct DC solutions to find; at least 1.
    pub max_solutions: usize,
}

/// Converged analog DC solution with logic drivers held fixed.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Solved {
    pub(crate) values: Vec<f64>,
    pub(crate) strategy: Strategy,
    /// Total Newton iterations across all attempted continuation strategies.
    pub(crate) iterations: usize,
    /// Flag indicating whether direct Newton iteration from the initial guess succeeded alone.
    pub(crate) from_guess: bool,
}

/// Partition of requested node voltages (such as transient initial conditions) into directly
/// forced nets and nets already fixed by ideal branches.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Forcing {
    /// Independently forced `(node_unknown, voltage)` pairs.
    pub(crate) forced: Vec<(UnknownIndex, f64)>,
    /// Requested `(node_unknown, voltage)` pairs left unforced because voltage sources or
    /// inductors already constrain them against ground or an earlier forced net.
    pub(crate) fixed: Vec<(UnknownIndex, f64)>,
}

/// Cached DC operating point with its solver conditions, logic event state, and factored
/// sparse solver.
#[derive(Debug, Clone)]
pub(crate) struct Remembered {
    /// Swept source or parameter overrides active when the operating point was solved.
    held: BTreeMap<Held, f64>,
    /// DC-relevant options under which the operating point was solved.
    options: Options,
    /// Analog DC solution from zero prior to mixed-signal logic settling.
    solved: Solved,
    /// Settled operating point with recorded values from the original solve.
    point: OperatingPointResult,
    events: Events,
    /// Sparse solver state (pivots and factors) reused by subsequent analyses at this point.
    solver: Solver<f64>,
}

impl Remembered {
    /// Returns the settled operating-point solution vector.
    pub(crate) fn solution(&self) -> &[f64] {
        &self.point.values
    }

    /// Reports whether this cache entry matches `held`, `solver`'s ordering, and all
    /// DC-relevant fields of `options`.
    fn solved_under(
        &self,
        held: &BTreeMap<Held, f64>,
        solver: &Solver<f64>,
        options: &Options,
    ) -> bool {
        self.solves_alike(held, solver, options) && self.options == read_at_dc(options)
    }

    /// Reports whether this cache entry solves the same DC equations under `held`, `solver`'s
    /// ordering, and `options`, permitting differences only in Newton stopping criteria.
    fn solves_alike(
        &self,
        held: &BTreeMap<Held, f64>,
        solver: &Solver<f64>,
        options: &Options,
    ) -> bool {
        self.held == *held
            && Arc::ptr_eq(self.solver.ordering(), solver.ordering())
            && read_by_equations(&self.options) == read_by_equations(options)
    }
}

/// Normalizes `options` by resetting fields ignored by DC solving (transient controls,
/// recording filters, profiling, and parallelism) to defaults so DC cache keys compare equal.
fn read_at_dc(options: &Options) -> Options {
    let unread = Options::default();
    Options {
        charge_tolerance: unread.charge_tolerance,
        max_transient_iterations: unread.max_transient_iterations,
        truncation_tolerance: unread.truncation_tolerance,
        error_control: unread.error_control,
        max_memory: unread.max_memory,
        record: unread.record,
        regions: unread.regions,
        profile: unread.profile,
        parallelism: unread.parallelism,
        ..options.clone()
    }
}

/// Normalizes `options` to fields that affect the DC equations themselves, resetting Newton
/// tolerances and iteration limits to defaults.
fn read_by_equations(options: &Options) -> Options {
    let unread = Options::default();
    Options {
        relative_tolerance: unread.relative_tolerance,
        voltage_tolerance: unread.voltage_tolerance,
        current_tolerance: unread.current_tolerance,
        max_dc_iterations: unread.max_dc_iterations,
        ..read_at_dc(options)
    }
}

/// Conductance path reduced during gmin stepping.
#[derive(Debug, Clone, Copy)]
enum Stepped {
    /// Shunt conductance from every node to ground, removed once reduced to `gmin`.
    Shunt,
    /// Junction conductance (`gmin`) reduced directly toward the configured target.
    Junctions,
}

/// Strategy that converged to the DC operating point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Direct Newton iteration from the initial guess.
    Direct,
    /// Newton iteration while shrinking a node-to-ground shunt conductance to zero.
    GminStepping,
    /// Newton iteration while shrinking junction conductance down to `gmin`.
    JunctionGminStepping,
    /// Newton iteration while ramping independent sources from zero to full scale.
    SourceStepping,
    /// Pseudo-transient continuation anchoring each node to its previous voltage via a
    /// shrinking conductance.
    PseudoTransient,
}

impl Strategy {
    /// Returns the strategy name in `snake_case`, such as `gmin_stepping`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::GminStepping => "gmin_stepping",
            Self::JunctionGminStepping => "junction_gmin_stepping",
            Self::SourceStepping => "source_stepping",
            Self::PseudoTransient => "pseudo_transient",
        }
    }
}

impl Simulator {
    /// Finds the DC operating point with every source evaluated at time zero.
    ///
    /// Threshold detectors and logic drivers settle iteratively with the analog circuit until
    /// their states agree. The simulator caches the latest operating point and reuses it
    /// whenever held sources and DC-relevant options match, re-evaluating recorded quantities
    /// as needed. When only Newton tolerances or iteration limits differ, direct Newton
    /// iteration resumes from the cached iterate; [`Simulator::operating_point_solves`] tracks
    /// full solves.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::NoOperatingPoint`] or [`SimulationError::Singular`] when no
    /// strategy converges, and [`SimulationError::InvalidOptions`] for out-of-range options.
    pub fn operating_point(
        &mut self,
        options: &Options,
    ) -> Result<OperatingPointResult, SimulationError> {
        options.parallelism.confined(|| {
            self.solvable()?;
            self.prepare(options)?;
            Ok(self.settled_from_zero(options)?.1)
        })
    }

    /// Returns the analog DC solution from zero and the settled operating point, reusing the
    /// cache when held sources and DC-relevant options match.
    fn settled_from_zero(
        &mut self,
        options: &Options,
    ) -> Result<(Solved, OperatingPointResult), SimulationError> {
        if let Some(remembered) = &self.state.remembered
            && remembered.solved_under(&self.state.held, &self.state.solver, options)
        {
            let (solved, mut point) = (remembered.solved.clone(), remembered.point.clone());
            // Restoring evaluates every device at the cached point without voltage limiting.
            self.restore(options);
            point.recorded = self.record_evaluated();
            return Ok((solved, point));
        }
        self.reset_logic();
        let solved = self.solve_from_zero(options)?;
        let point = self.settle(solved.clone(), options)?;
        self.state.remembered = Some(Remembered {
            held: self.state.held.clone(),
            options: read_at_dc(options),
            solved: solved.clone(),
            point: point.clone(),
            events: self.state.events.clone(),
            solver: self.state.solver.clone(),
        });
        Ok((solved, point))
    }

    /// Restores logic event state, driver targets, device evaluation state, and sparse solver
    /// factors to match the cached operating point.
    ///
    /// Downstream analyses then factor from the exact same pivot and device state as a fresh
    /// operating-point solve.
    fn restore(&mut self, options: &Options) {
        let Some(remembered) = &self.state.remembered else {
            return;
        };
        let values = remembered.point.values.clone();
        self.state.events = remembered.events.clone();
        self.state.solver.adopt(&remembered.solver);
        for driver in 0..self.elaborated.logic.drivers.len() {
            let (element, level) = (
                self.elaborated.logic.drivers[driver].0,
                self.state.events.driven()[driver],
            );
            self.drive(element, 0.0, level, 0.0);
        }
        self.evaluate(&values, None, 1.0, options.gmin, false);
    }

    /// Solves the analog DC operating point from zero with all logic drivers at
    /// `LogicLevel::Unknown`, leaving devices evaluated at the solution.
    ///
    /// Reuses the cached solve when held sources and DC-relevant options match, or resumes
    /// direct Newton iteration when only tolerances or iteration limits differ. Otherwise
    /// solves from zero or warm-starts via [`Simulator::solve_warm`] in a chained batch.
    pub(crate) fn solve_from_zero(&mut self, options: &Options) -> Result<Solved, SimulationError> {
        let start = self
            .state
            .remembered
            .as_ref()
            .filter(|remembered| {
                remembered.solves_alike(&self.state.held, &self.state.solver, options)
            })
            .map(|remembered| {
                let same = remembered.solved_under(&self.state.held, &self.state.solver, options);
                (same, remembered.solved.clone())
            });
        if let Some((true, solved)) = &start {
            // Re-evaluate without limiting so stateful limiters match the end of the solve.
            self.evaluate(&solved.values, None, 1.0, options.gmin, false);
            return Ok(solved.clone());
        }
        self.state.solves += 1;
        if let Some((_, solved)) = start
            && solved.from_guess
            && let Some(left) = options.max_dc_iterations.checked_sub(solved.iterations)
        {
            let taken = solved.iterations;
            let mut values = solved.values;
            if let Ok(more) = self.newton(&mut values, Mode::direct(&[]), options, left) {
                return Ok(Solved {
                    values,
                    strategy: Strategy::Direct,
                    iterations: taken + more,
                    from_guess: true,
                });
            }
        }
        if let Some(warm) = self.state.warm.take() {
            return self.solve_warm(&warm, options);
        }
        let [nodesets] = self.forcings([(&options.nodesets, "nodeset")])?;
        self.solve_forced(&[], &nodesets.forced, options)
    }

    /// Validates each `(voltages, label)` set (such as `nodeset`) and splits nets into forced
    /// constraints versus nets already fixed by voltage sources or inductors.
    ///
    /// Nets connected by DC-voltage-fixing branches to ground or to an earlier forced net are
    /// recorded in [`Forcing::fixed`] to avoid over-constraining ideal sources.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidAnalysis`] when a set names an unknown net or a
    /// non-finite voltage.
    pub(crate) fn forcings<const N: usize>(
        &self,
        sets: [(&BTreeMap<String, f64>, &str); N],
    ) -> Result<[Forcing; N], SimulationError> {
        let nodes = self.elaborated.node_diagonals.len();
        let reference = nodes;
        let mut partition = Partition::new(nodes + 1);
        let end = |node: jolty_models::Node| node.map_or(reference, UnknownIndex::index);
        for device in &self.elaborated.devices {
            let ends = device.nodes();
            for branch in device.branches() {
                if branch.kind.fixes_dc_voltage() {
                    partition.union(end(ends[branch.from]), end(ends[branch.to]));
                }
            }
        }
        let mut forcings = std::array::from_fn(|_| Forcing::default());
        for ((voltages, what), forcing) in sets.into_iter().zip(&mut forcings) {
            for (name, &voltage) in voltages {
                let index = self
                    .unknown(name)
                    .filter(|unknown| {
                        unknown.index() < nodes
                            && self.elaborated.natures[unknown.index()] == Nature::Voltage
                    })
                    .ok_or_else(|| {
                        SimulationError::InvalidAnalysis(format!(
                            "the {what} names `{name}`, which is no net of the circuit"
                        ))
                    })?;
                if !voltage.is_finite() {
                    return Err(SimulationError::InvalidAnalysis(format!(
                        "the {what} of `{name}` is {voltage:?} V, not a finite voltage"
                    )));
                }
                if partition.union(index.index(), reference) {
                    forcing.forced.push((index, voltage));
                } else {
                    forcing.fixed.push((index, voltage));
                }
            }
        }
        Ok(forcings)
    }

    /// Solves the DC system from zero with `held` nets forced throughout, optionally using
    /// `released` nets as SPICE-style nodesets first.
    ///
    /// When `released` is non-empty, solves with both `held` and `released` forced and then
    /// re-solves with only `held` forced (falling back to zero if the nodeset solve fails).
    /// Leaves `self.state.forced` equal to `held` for subsequent logic settling.
    pub(crate) fn solve_forced(
        &mut self,
        held: &[(UnknownIndex, f64)],
        released: &[(UnknownIndex, f64)],
        options: &Options,
    ) -> Result<Solved, SimulationError> {
        let zero = vec![0.0; self.elaborated.unknowns.len()];
        if released.is_empty() {
            self.state.forced = held.to_vec();
            return self.solve_dc(&zero, options);
        }
        let before = self.state.workspace.meter.iterations();
        self.state.forced = [held, released].concat();
        let guess = self.solve_dc(&zero, options);
        self.state.forced = held.to_vec();
        let solved = self.solve_dc(
            guess.as_ref().map_or(&zero, |solved| &solved.values),
            options,
        )?;
        Ok(Solved {
            iterations: self.iterations_since(before),
            from_guess: false,
            ..solved
        })
    }

    /// Computes a transient starting operating point with `initial` node voltages forced and
    /// `options.nodesets` applied as releasable hints, without updating the DC cache.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidAnalysis`] if an initial condition names an unknown
    /// net or conflicts with voltage sources or inductors beyond tolerance, plus any error
    /// from [`Simulator::operating_point`].
    pub(crate) fn forced_operating_point(
        &mut self,
        initial: &BTreeMap<String, f64>,
        options: &Options,
    ) -> Result<OperatingPointResult, SimulationError> {
        let [initial, nodesets] = self.forcings([
            (initial, "initial condition"),
            (&options.nodesets, "nodeset"),
        ])?;
        self.state.solves += 1;
        self.reset_logic();
        let point = self
            .solve_forced(&initial.forced, &nodesets.forced, options)
            .and_then(|solved| self.settle(solved, options));
        self.state.forced.clear();
        let point = point?;
        for &(unknown, voltage) in &initial.fixed {
            let found = point.values[unknown.index()];
            let tolerance = options.unknown_tolerance(Nature::Voltage, found, voltage);
            if (found - voltage).abs() > tolerance {
                let name = &self.elaborated.unknowns[unknown.index()].name;
                return Err(SimulationError::InvalidAnalysis(format!(
                    "the initial condition of `{name}` is {voltage:?} V, but the voltage \
                     sources and inductors that fix it hold it at {found:?} V"
                )));
            }
        }
        Ok(point)
    }

    /// Solves a chained batch run starting from the preceding run's `warm` operating point.
    ///
    /// Evaluates devices at `warm` to initialize limiters, continues along all differing
    /// parameters and temperature, and falls back to full DC strategies from `warm` if
    /// continuation stalls.
    fn solve_warm(&mut self, warm: &Warm, options: &Options) -> Result<Solved, SimulationError> {
        let before = self.state.workspace.meter.iterations();
        let to = self.values.clone();
        self.evaluate(&warm.solution, None, 1.0, options.gmin, false);
        let coordinates = self.coordinates(&warm.values, &to);
        let spans: Vec<(f64, f64)> = coordinates
            .iter()
            .map(|&(_, from, to)| (from, to))
            .collect();
        let mut values = warm.solution.clone();
        let mut set = |simulator: &mut Self, at: &[f64]| {
            let mut point = to.clone();
            for (&(coordinate, _, target), &value) in coordinates.iter().zip(at) {
                if value.to_bits() != target.to_bits() {
                    match coordinate {
                        Some(parameter) => point.set(parameter, value),
                        None => point.temperature = Some(value),
                    }
                }
            }
            matches!(simulator.move_to(point), Ok(true))
        };
        if self.continue_along(&spans, &mut set, &mut values, options) {
            return Ok(Solved {
                values,
                strategy: Strategy::Direct,
                iterations: self.iterations_since(before),
                from_guess: false,
            });
        }
        if !self
            .move_to(to)
            .map_err(|error| SimulationError::InvalidAnalysis(error.to_string()))?
        {
            return Err(SimulationError::InvalidAnalysis(
                "a run of a chain has another structure than the run before it".to_owned(),
            ));
        }
        let continued = self.iterations_since(before);
        let solved = self.solve_dc(&warm.solution, options)?;
        Ok(Solved {
            iterations: continued + solved.iterations,
            from_guess: false,
            ..solved
        })
    }

    /// Returns the number of Newton iterations executed since the `before` counter snapshot.
    pub(crate) fn iterations_since(&self, before: usize) -> usize {
        self.state.workspace.meter.iterations() - before
    }

    /// Lists every parameter (and temperature as `None`) that differs between `from` and `to`,
    /// paired with `(from_value, to_value)` in parameter-index order followed by temperature.
    fn coordinates(&self, from: &Values, to: &Values) -> Vec<(Option<ParameterId>, f64, f64)> {
        let nominal = self.elaborated.nominal();
        let value = |values: &Values, parameter: ParameterId| {
            values.get(parameter).unwrap_or(nominal[parameter.index()])
        };
        let mut parameters: Vec<ParameterId> = from
            .patches()
            .iter()
            .chain(to.patches())
            .map(|&(parameter, _)| parameter)
            .collect();
        parameters.sort_unstable();
        parameters.dedup();
        let mut found: Vec<(Option<ParameterId>, f64, f64)> = parameters
            .into_iter()
            .map(|parameter| {
                (
                    Some(parameter),
                    value(from, parameter),
                    value(to, parameter),
                )
            })
            .filter(|&(_, a, b)| a.to_bits() != b.to_bits())
            .collect();
        let (a, b) = (
            self.elaborated.temperature_of(from),
            self.elaborated.temperature_of(to),
        );
        if a.to_bits() != b.to_bits() {
            found.push((None, a, b));
        }
        found
    }

    /// Finds the primary DC operating point followed by up to `search.max_solutions - 1`
    /// additional distinct DC solutions in discovery order.
    ///
    /// Uses deflated continuation from zero, reflections through pairs of known solutions, and
    /// symmetry-breaking node perturbations to locate bistable latch states and metastable
    /// equilibria.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidAnalysis`] when `max_solutions` is zero, plus any
    /// error from [`Simulator::operating_point`].
    pub fn operating_points(
        &mut self,
        search: &OperatingPoints,
        options: &Options,
    ) -> Result<Vec<OperatingPointResult>, SimulationError> {
        options.parallelism.confined(|| {
            self.solvable()?;
            self.prepare(options)?;
            if search.max_solutions == 0 {
                return Err(SimulationError::InvalidAnalysis(
                    "a search for operating points finds one or more".to_owned(),
                ));
            }
            let (first, point) = self.settled_from_zero(options)?;
            if search.max_solutions == 1 {
                return Ok(vec![point]);
            }
            // Start with every logic driver unknown, then settle logic afresh for each root.
            self.reset_logic();
            let mut known = vec![first.values];
            let mut others = Vec::new();
            while known.len() < search.max_solutions {
                let Some(solved) = self.another(&known, options) else {
                    break;
                };
                known.push(solved.values.clone());
                others.push(solved);
            }
            if others.is_empty() {
                self.restore(options);
            }
            let mut points = vec![point];
            for solved in others {
                self.reset_logic();
                points.push(self.settle(solved, options)?);
            }
            Ok(points)
        })
    }

    /// Searches for a DC solution distinct from `known` using deflated continuation from zero,
    /// pairwise reflections, and alternating node perturbations.
    fn another(&mut self, known: &[Vec<f64>], options: &Options) -> Option<Solved> {
        let before = self.state.workspace.meter.iterations();
        let zero = vec![0.0; self.elaborated.unknowns.len()];
        if let Ok(solved) = self.solve_deflated(&zero, options, known)
            && self.distinct(known, &solved.values, options)
        {
            return Some(solved);
        }
        let mut starts = Vec::new();
        for (index, mirror) in known.iter().enumerate().rev() {
            for (other, through) in known.iter().enumerate().rev() {
                if other != index {
                    starts.push(
                        mirror
                            .iter()
                            .zip(through)
                            .map(|(mirror, through)| 2.0 * through - mirror)
                            .collect::<Vec<f64>>(),
                    );
                }
            }
        }
        for root in known.iter().rev() {
            for sign in [1.0, -1.0] {
                let mut neighbor = root.clone();
                let voltages = self
                    .elaborated
                    .unknowns
                    .iter()
                    .zip(&mut neighbor)
                    .filter(|(unknown, _)| unknown.nature == Nature::Voltage);
                for (turn, (_, value)) in voltages.enumerate() {
                    let alternate = if turn % 2 == 0 { 1.0 } else { -1.0 };
                    *value += sign * alternate * NUDGE;
                }
                starts.push(neighbor);
            }
        }
        let direct = Mode::direct(known);
        for mut values in starts {
            if self
                .newton(&mut values, direct, options, options.max_dc_iterations)
                .is_ok()
                && self.distinct(known, &values, options)
            {
                return Some(Solved {
                    values,
                    strategy: Strategy::Direct,
                    iterations: self.iterations_since(before),
                    from_guess: false,
                });
            }
        }
        None
    }

    /// Checks whether `values` differs by more than [`DISTINCT`] tolerances from every root in
    /// `known` in at least one unknown.
    fn distinct(&self, known: &[Vec<f64>], values: &[f64], options: &Options) -> bool {
        !known
            .iter()
            .any(|root| self.indistinct(root, values, options))
    }

    /// Checks whether `first` and `second` agree within [`DISTINCT`] tolerances across all
    /// unknowns.
    fn indistinct(&self, first: &[f64], second: &[f64], options: &Options) -> bool {
        self.elaborated
            .unknowns
            .iter()
            .zip(first.iter().zip(second))
            .all(|(unknown, (a, b))| {
                let tolerance = options.unknown_tolerance(unknown.nature, *a, *b);
                (a - b).abs() <= DISTINCT * tolerance
            })
    }

    /// Resets all logic nets and analog logic drivers to `LogicLevel::Unknown`.
    pub(crate) fn reset_logic(&mut self) {
        self.state.events = Events::new(&self.elaborated.logic);
        for index in 0..self.elaborated.logic.drivers.len() {
            let element = self.elaborated.logic.drivers[index].0;
            self.drive(element, 0.0, LogicLevel::Unknown, 0.0);
        }
    }

    /// Measures each logic driver's total driven capacitance at `values` using a quasi-static
    /// small-signal probe at [`LOAD_FREQUENCY`].
    fn measure_loads(&mut self, values: &[f64], options: &Options) {
        if self.elaborated.logic.drivers.is_empty() {
            return;
        }
        self.evaluate(values, None, 1.0, options.gmin, true);
        let omega = TAU * LOAD_FREQUENCY;
        let contributions = &self.state.workspace.contributions;
        let matrix: Vec<Complex64> = contributions
            .conductance
            .iter()
            .zip(&contributions.capacitance)
            .map(|(&conductance, &capacitance)| Complex64::new(conductance, omega * capacitance))
            .collect();
        let mut solver = self.state.solver.sharing::<Complex64>();
        let Ok(mut factors) = solver.factor(&matrix) else {
            return;
        };
        let mut phasors = vec![Complex64::default(); values.len()];
        for &(element, net) in &self.elaborated.logic.drivers {
            let Some(current) = self.state.runs.driven(&self.elaborated.groups, element) else {
                continue;
            };
            // Raising the driver's voltage by a volt adds -1 to its equation's residual.
            phasors.fill(Complex64::default());
            phasors[current.index()] = Complex64::new(1.0, 0.0);
            if factors.solve(&mut phasors).is_ok() {
                self.state
                    .events
                    .load(net, phasors[current.index()].im.abs() / omega);
            }
        }
    }

    /// Records each characterized logic device whose supply at `values`, at `time`, lies
    /// further from the supply it was measured at than the relative tolerance.
    pub(crate) fn watch_supplies(&mut self, values: &[f64], time: f64, options: &Options) {
        let input = Input::new(values, Some(time), 0.0);
        for (index, placed) in self.elaborated.logic.devices.iter().enumerate() {
            let (Some(nominal), &[supply, ground]) = (
                placed.device.domain().and_then(|domain| domain.supply),
                placed.nodes(),
            ) else {
                continue;
            };
            let voltage = input.value(supply) - input.value(ground);
            let slack = options.relative_tolerance * nominal;
            let bounds = Bounds {
                minimum: nominal - slack,
                maximum: nominal + slack,
            };
            self.state
                .events
                .watch(index, ExcursionKind::Supply, voltage, bounds, time);
        }
    }

    /// Settles the logic with the analog solution `solved`, holding each driver at its net's
    /// level and solving again until the detectors read what the logic settled with.
    pub(crate) fn settle(
        &mut self,
        mut solved: Solved,
        options: &Options,
    ) -> Result<OperatingPointResult, SimulationError> {
        let mut iterations = solved.iterations;
        let detectors = self.elaborated.logic.detectors.len();
        let mut previous = vec![LogicLevel::Unknown; detectors];
        let mut unsettled = vec![false; detectors];
        for round in 0.. {
            let mut levels: Vec<LogicLevel> = self
                .elaborated
                .logic
                .detectors
                .iter()
                .map(|&(element, _)| {
                    let input = Input::new(&solved.values, None, 0.0);
                    let sensed = self
                        .state
                        .runs
                        .sense(&self.elaborated.groups, element, &input);
                    sensed.map_or(LogicLevel::Unknown, |(margin, _)| {
                        detected(margin, LogicLevel::Unknown)
                    })
                })
                .collect();
            for ((level, before), unsettled) in levels.iter_mut().zip(&previous).zip(&mut unsettled)
            {
                *unsettled |= round >= SETTLE_ROUNDS && level != before;
                if *unsettled {
                    *level = LogicLevel::Unknown;
                }
            }
            self.state.events.settle(&self.elaborated.logic, &levels);
            previous = levels;
            let changed = self.state.events.retarget(&self.elaborated.logic);
            if changed.is_empty() {
                break;
            }
            for (driver, level) in changed {
                let element = self.elaborated.logic.drivers[driver].0;
                self.drive(element, 0.0, level, 0.0);
            }
            solved = self.solve_dc(&solved.values, options)?;
            iterations += solved.iterations;
        }
        self.measure_loads(&solved.values, options);
        self.watch_supplies(&solved.values, 0.0, options);
        Ok(OperatingPointResult {
            recorded: self.record_dc(&solved.values, options),
            values: solved.values,
            logic: self.state.events.levels().to_vec(),
            strategy: solved.strategy,
            iterations,
        })
    }

    /// Finds a DC solution starting from `guess`, trying Newton iteration, then gmin stepping,
    /// then source stepping.
    pub(crate) fn solve_dc(
        &mut self,
        guess: &[f64],
        options: &Options,
    ) -> Result<Solved, SimulationError> {
        self.solve_deflated(guess, options, &[])
    }

    /// Solves the DC equations from `guess` using [`Simulator::solve_dc`]'s strategy ladder
    /// deflated against `deflated` roots.
    pub(crate) fn solve_deflated(
        &mut self,
        guess: &[f64],
        options: &Options,
        deflated: &[Vec<f64>],
    ) -> Result<Solved, SimulationError> {
        let before = self.state.workspace.meter.iterations();
        let finish = |simulator: &Self, values, strategy| Solved {
            values,
            strategy,
            iterations: simulator.iterations_since(before),
            from_guess: false,
        };
        let mut values = guess.to_owned();
        if let Ok(iterations) = self.newton(
            &mut values,
            Mode::direct(deflated),
            options,
            options.max_dc_iterations,
        ) {
            return Ok(Solved {
                values,
                strategy: Strategy::Direct,
                iterations,
                from_guess: true,
            });
        }
        for (stepped, strategy) in [
            (Stepped::Shunt, Strategy::GminStepping),
            (Stepped::Junctions, Strategy::JunctionGminStepping),
        ] {
            let mut values = guess.to_owned();
            if self
                .gmin_stepping(&mut values, options, deflated, stepped)
                .is_ok()
            {
                return Ok(finish(self, values, strategy));
            }
        }
        let zero = vec![0.0; guess.len()];
        let mut values = zero.clone();
        let stepping = match self.source_stepping(&mut values, options, deflated) {
            Ok(()) => return Ok(finish(self, values, Strategy::SourceStepping)),
            Err(failure) => failure,
        };
        let stepped = values;
        let mut values = zero;
        // Equations that fail as singular during source stepping stay singular even if
        // pseudo-transient shunts temporarily stabilize them before stalling.
        match (
            stepping,
            self.pseudo_transient(&mut values, options, deflated),
        ) {
            (_, Ok(())) => Ok(finish(self, values, Strategy::PseudoTransient)),
            (Failure::Solver(error), _) | (_, Err(Failure::Solver(error))) => Err(error),
            (Failure::Diverged, Err(Failure::Diverged)) => Err(SimulationError::NoOperatingPoint {
                worst: self.furthest_from_converging(&stepped, options, NAMED_UNKNOWNS),
            }),
        }
    }

    /// Solves the DC system by shrinking conductance `stepped` down to `gmin`, matching
    /// ngspice's dynamic and junction gmin stepping, deflated against `deflated`.
    ///
    /// The divisor grows up to [`STEPPED_DIVISOR`] after quick stages, shrinks to its square
    /// root after slow stages, and shrinks to its fourth root on failure before aborting
    /// below [`MIN_STEPPED_DIVISOR`].
    fn gmin_stepping(
        &mut self,
        values: &mut [f64],
        options: &Options,
        deflated: &[Vec<f64>],
        stepped: Stepped,
    ) -> Result<(), Failure> {
        let mut divisor = STEPPED_DIVISOR;
        let mut conductance = FIRST_STEPPED.max(options.gmin);
        let mut last: Option<(f64, Vec<f64>)> = None;
        loop {
            let (shunt, gmin) = match stepped {
                Stepped::Shunt => (conductance, options.gmin),
                Stepped::Junctions => (0.0, conductance),
            };
            let mode = Mode::Dc {
                source_scale: 1.0,
                shunt,
                anchor: None,
                deflated,
            };
            let stepping = Options {
                gmin,
                ..options.clone()
            };
            match self.newton(values, mode, &stepping, options.max_dc_iterations) {
                Ok(_) if conductance <= options.gmin => break,
                Ok(iterations) => {
                    last = Some((conductance, values.to_vec()));
                    if iterations < options.max_dc_iterations / 4 {
                        divisor = (divisor * divisor.sqrt()).min(STEPPED_DIVISOR);
                    } else if iterations > 3 * options.max_dc_iterations / 4 {
                        divisor = divisor.sqrt();
                        // Repeated slow stages indicate stalled conductance reduction.
                        if divisor < MIN_STEPPED_DIVISOR {
                            return Err(Failure::Diverged);
                        }
                    }
                    conductance = (conductance / divisor).max(options.gmin);
                }
                Err(failure) => {
                    divisor = divisor.sqrt().sqrt();
                    let Some((solved, solution)) = &last else {
                        return Err(failure);
                    };
                    if divisor < MIN_STEPPED_DIVISOR {
                        return Err(failure);
                    }
                    values.copy_from_slice(solution);
                    conductance = (solved / divisor).max(options.gmin);
                }
            }
        }
        match stepped {
            // The final junction stage already solved the target `gmin` system.
            Stepped::Junctions => Ok(()),
            Stepped::Shunt => self
                .newton(
                    values,
                    Mode::direct(deflated),
                    options,
                    options.max_dc_iterations,
                )
                .map(|_| ()),
        }
    }

    /// Solves the DC system via pseudo-transient continuation (Kelley and Keyes, *SIAM Journal
    /// on Numerical Analysis* 35(2), 1998) deflated against `deflated`.
    ///
    /// Anchors each node to its previous voltage through a shrinking diagonal conductance,
    /// bypassing continuation folds where gmin and source stepping stall. Finishes with a
    /// direct Newton solve once node changes settle or the conductance reaches `gmin`.
    fn pseudo_transient(
        &mut self,
        values: &mut [f64],
        options: &Options,
        deflated: &[Vec<f64>],
    ) -> Result<(), Failure> {
        let mut held = FIRST_HELD;
        let mut last = values.to_vec();
        for _ in 0..MAX_HELD_STEPS {
            let mode = Mode::Dc {
                source_scale: 1.0,
                shunt: held,
                anchor: Some(&last),
                deflated,
            };
            match self.newton(values, mode, options, options.max_dc_iterations) {
                Ok(iterations) => {
                    let settled = values.iter().zip(&last).zip(&self.elaborated.unknowns).all(
                        |((value, before), unknown)| {
                            (value - before).abs()
                                <= options.unknown_tolerance(unknown.nature, *value, *before)
                        },
                    );
                    last.copy_from_slice(values);
                    if settled || held <= options.gmin {
                        return self
                            .newton(
                                values,
                                Mode::direct(deflated),
                                options,
                                options.max_dc_iterations,
                            )
                            .map(|_| ());
                    }
                    held /= if iterations < options.max_dc_iterations / 4 {
                        STEPPED_DIVISOR
                    } else {
                        STEPPED_DIVISOR.sqrt()
                    };
                }
                Err(failure) => {
                    values.copy_from_slice(&last);
                    held *= STEPPED_DIVISOR;
                    if held > MAX_HELD {
                        return Err(failure);
                    }
                }
            }
        }
        Err(Failure::Diverged)
    }

    /// Ramps every source from zero to its value, growing the increment after each success
    /// and shrinking it after each failure; every solve is deflated by the solutions
    /// `deflated`.
    fn source_stepping(
        &mut self,
        values: &mut [f64],
        options: &Options,
        deflated: &[Vec<f64>],
    ) -> Result<(), Failure> {
        let mut scale = 0.0;
        let mut increment = FIRST_SOURCE_INCREMENT;
        let mut last = values.to_vec();
        while scale < 1.0 {
            let next = (scale + increment).min(1.0);
            match self.newton(
                values,
                Mode::Dc {
                    source_scale: next,
                    shunt: 0.0,
                    anchor: None,
                    deflated,
                },
                options,
                options.max_dc_iterations,
            ) {
                Ok(_) => {
                    scale = next;
                    last.copy_from_slice(values);
                    increment *= 2.0;
                }
                Err(failure) => {
                    values.copy_from_slice(&last);
                    increment /= 4.0;
                    if increment < MIN_SOURCE_INCREMENT {
                        return Err(failure);
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::test_helpers::{grounded_simulator, simulator_from_json};

    /// Builds a folded-characteristic test circuit: 3 V through 1 kΩ into a node with a
    /// negative 2 mS conductance and antiparallel diodes to ground, placing the operating
    /// point past a source-stepping fold on D1's branch.
    fn folded() -> Simulator {
        grounded_simulator(
            "test",
            "0",
            &json!([
                {"name": "V1", "of": {"model": "voltage_source"},
                 "parameters": {"waveform": {"dc": 3.0}},
                 "terminals": {"positive": "in", "negative": "0"}},
                {"name": "R1", "of": {"model": "resistor"}, "parameters": {"resistance": 1e3},
                 "terminals": {"a": "in", "b": "x"}},
                {"name": "G1", "of": {"model": "voltage_controlled_current_source"},
                 "parameters": {"transconductance": 2e-3},
                 "terminals": {"positive": "0", "negative": "x",
                               "control_positive": "x", "control_negative": "0"}},
                {"name": "D1", "of": {"model": "diode"}, "parameters": {},
                 "terminals": {"anode": "x", "cathode": "0"}},
                {"name": "D2", "of": {"model": "diode"}, "parameters": {},
                 "terminals": {"anode": "0", "cathode": "x"}},
            ]),
        )
    }

    #[test]
    fn pseudo_transient_continuation_follows_a_circuit_past_its_fold() {
        let mut simulator = folded();
        let options = Options::default();
        let mut values = vec![0.0; simulator.elaborated.unknowns.len()];
        simulator
            .pseudo_transient(&mut values, &options, &[])
            .expect("pseudo-transient continuation converges");
        let x = values[simulator.unknown("x").expect("a node").index()];
        // Past the fold, the solution lies where D1 conducts: (3 - x)/1 kΩ + 2 mS · x
        // flows into it.
        let slope = jolty_models::thermal_voltage(jolty_models::REFERENCE_TEMPERATURE);
        let diode = 1e-14 * ((x / slope).exp() - 1.0);
        assert!(x > 0.5, "the node at {x} V");
        assert!(((3.0 - x) / 1e3 + 2e-3 * x - diode).abs() < 1e-9);
    }

    #[test]
    fn pseudo_transient_continuation_finds_the_point_source_stepping_folds_short_of() {
        // Reduced 47-device circuit where direct Newton, gmin stepping, and source stepping
        // stall at a 51% fold (as in ngspice, which requires transient analysis to locate
        // the operating point).
        let mut simulator = simulator_from_json(include_str!("../tests/data/folded_sources.json"));
        let options = Options::default();
        let unknowns = simulator.elaborated.unknowns.len();
        let mut values = vec![0.0; unknowns];
        assert!(
            simulator
                .source_stepping(&mut values, &options, &[])
                .is_err()
        );
        let mut values = vec![0.0; unknowns];
        simulator
            .pseudo_transient(&mut values, &options, &[])
            .expect("pseudo-transient continuation converges");
        let voltage = |name: &str| values[simulator.unknown(name).expect("a node").index()];
        // ngspice's transient operating point, to its printed digits.
        for (node, expected) in [
            ("n20", -54.634_247),
            ("n16", -1.998_355),
            ("n7", -1.765_709),
        ] {
            let found = voltage(node);
            assert!((found - expected).abs() < 1e-5, "{node} at {found} V");
        }
    }
}
