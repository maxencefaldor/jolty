//! Distinguishing numerical trapezoidal ringing from genuine physical oscillation.
//!
//! Under the trapezoidal rule on a step of length `h` (`α = 2/h`), the charge derivatives
//! `y_k = dq/dt` of a linear circuit `G x + C x' = s` propagate through the amplification
//! operator `R = 2 F − I = (α C − G)(α C + G)⁻¹`, where `F = α C (G + α C)⁻¹` is Shampine's
//! filter (Shampine, 1980; Hairer and Wanner, *Solving Ordinary Differential Equations II*,
//! 1996, sections IV.3 and IV.8):
//!
//! - A **physical lossless oscillation** `s = ±jω` maps to the complex unit circle
//!   `R = exp(∓jθ)` with `θ = 2 atan(ω h / 2)`. When `ω h > 2` (fewer than four steps per
//!   period), `θ > π/2` and the first differences `y_{k+1} − y_k` alternate in sign every
//!   step even though the oscillation is real and energy-conserving.
//! - A **stiff real decaying mode** `s = −σ` with `σ > α = 2/h` maps to the negative real
//!   interval `R ∈ (−1, 0)`, so `y_k` suffers genuine numerical `(-1)^k` ringing.
//! - An **algebraic or DAE constraint mode** `s = ∞` (`C u = 0`, `G u ≠ 0`, such as a
//!   capacitor driven by a grounded or floating voltage source with an unlocated corner, or
//!   a Meyer gate-oxide displacement while its capacitance is zero) maps to `R = −1` (`F = 0`),
//!   carrying any derivative error from step to step undamped with alternating sign.
//!
//! A four-point alternating derivative check is therefore refined in two stages, keeping the
//! fast path free of allocations and linear solves:
//!
//! 1. **Variable-step scalar restoring-force identity**: on any single-mode harmonic
//!    oscillator `q'' = −ω² (q − q_eq)` integrated by the trapezoidal rule with arbitrary
//!    variable steps `h_k = t_{k+1} − t_k`, the midpoint acceleration `a_k = (y_{k+1} − y_k)/h_k`
//!    satisfies `a_k = −ω² (q_{k+1} + q_k)/2 + const` identically. Therefore
//!    `p_1 = (a_1 − a_0) · (q_2 − q_0)/2 ≤ 0` and `p_2 = (a_2 − a_1) · (q_3 − q_1)/2 ≤ 0`,
//!    with at least one strictly negative outside rounding. By contrast, a real decaying mode
//!    `q' = −λ (q − q_eq)` gives `p_1 ≥ 0` and `p_2 ≥ 0` identically, and decoupled algebraic
//!    ringing gives `p_1 = p_2 = 0` (flat `q`) or `p_1 · p_2 < 0` (monotone `q`).
//! 2. **Linearized passivity and exact algebraic projection**: for unknowns not explained by
//!    stage 1 (such as multi-mode coupled lossless LC ladders beating across frequencies, or
//!    genuine ringing), each physical domain (`Electrical`, `Thermal`, `Filter`, `State`,
//!    `Displacement`) is checked using the already-factored step Jacobian `J = G + α C`:
//!    - A `Displacement` unknown rings when its filter diagonal `F_{i,i} < 1/2` (`R_{i,i} < 0`).
//!    - With sign matrix `S` flipping `Current` rows that carry `C_{i,i} ≤ 0` so that `S C ≥ 0`
//!      and lossless `S G` is skew-symmetric in watts, a domain whose `S G` is skew-symmetric
//!      and decoupled from other domains has `uᵀ S G u = 0` identically for every `u`, skipping
//!      all linear solves for that domain. Otherwise the cached `LU` factors check the quadratic
//!      passivity ratio `uᵀ S G u ≥ uᵀ S (α C) u` (exact boundary `σ ≥ α = 2/h`), first on the
//!      collective Nyquist third difference `y_3 − 3 y_2 + 3 y_1 − y_0` of the domain's suspects
//!      (catching critically damped and overdamped RLC modes at every `ω h`), then per suspect
//!      unit vector `e_i` (preventing a large lossless tank from hiding a small stiff RC).
//!    - Finally, for any capacitor-voltage-source loop (grounded or floating ideal voltage
//!      sources where `C` is zero on row and column and `G` touches only voltage nodes), the
//!      local saddle-point projection `[C_SS / c_scale, B_S; A_S, 0] [z; μ] = [v_S; 0]` isolates
//!      the exact algebraic loop current `μ_y` and charge `μ_q` for every source in the
//!      component, cancelling all lossless LC modes (even multi-mode ladders, bypass capacitors
//!      to DC supplies, and ungrounded capacitive islands) to machine precision.

#[cfg(test)]
mod tests;

use std::collections::VecDeque;
use std::sync::Arc;

use jolty_models::Nature;
use jolty_numerics::{Entry, Ordering, Pattern, PatternBuilder, SparseLu};

use crate::Options;
use crate::newton::{Workspace, all, rounding};
use crate::profile::{Activity, Meter};
use crate::solver::{Factors, Solver};

/// The threshold of Shampine's filter diagonal `F = α C (G + α C)⁻¹` below which a real mode
/// rings under the trapezoidal rule.
///
/// For a real pole `s = −σ`, the trapezoidal amplification factor is `R = 2 F − 1`, where
/// `F = α / (α + σ)`. Thus `R < 0` (alternating sign) holds if and only if `σ > α = 2/h`,
/// equivalently `F < 0.5` (Dahlquist, 1963; Hairer and Wanner, *Solving Ordinary Differential
/// Equations II*, 1996, section IV.3; Shampine, 1980).
const TRAPEZOIDAL_REAL_FILTER_BOUND: f64 = 0.5;

/// Sentinel for an unknown absent from a local component index map.
const ABSENT: usize = usize::MAX;

/// The four consecutive trapezoidal points tested for numerical ringing, oldest first.
pub(crate) struct Run<'a> {
    /// The four accepted times `[t_0, t_1, t_2, t_3]`, strictly increasing.
    pub(crate) times: [f64; 4],
    /// The charges at each of the four times.
    pub(crate) charges: [&'a [f64]; 4],
    /// The charge derivatives `dq/dt` at each of the four times.
    pub(crate) derivatives: [&'a [f64]; 4],
    /// The trapezoidal step coefficient `α = 2 / (t_3 − t_2)` of the latest step.
    pub(crate) alpha: f64,
}

/// Borrowed immutable view of the linearized circuit matrices `(G, C)` and unknown natures.
#[derive(Clone, Copy)]
struct Linearized<'a> {
    pattern: &'a Pattern,
    conductance: &'a [f64],
    capacitance: &'a [f64],
    natures: &'a [Nature],
}

impl Linearized<'_> {
    fn count(self) -> usize {
        self.natures.len()
    }

    fn entry(self, values: &[f64], row: usize, column: usize) -> f64 {
        let entries = self.pattern.row(row);
        entries
            .binary_search_by_key(&column, |&(_, entry_column)| entry_column)
            .map_or(0.0, |position| values[entries[position].0])
    }

    fn row_dot(self, values: &[f64], row: usize, vector: &[f64]) -> f64 {
        self.pattern
            .row(row)
            .iter()
            .fold(0.0, |total, &(slot, column)| {
                total + values[slot] * vector[column]
            })
    }
}

/// The physical domain of an unknown, keeping incompatible units apart in quadratic forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Domain {
    /// Node voltages (`+1`) and electrical branch currents with `C_{i,i} ≤ 0` (`−1`).
    Electrical,
    /// Thermal network temperatures (`K · W`).
    Thermal,
    /// Internal excess-phase current filters (`C_{i,i} > 0`, `A²`).
    Filter,
    /// Dimensionless internal device state variables (memristor, photocell).
    State,
    /// Meyer gate-oxide displacement unknowns (`A²`, diagonal filter check).
    Displacement,
}

/// An ideal voltage-domain branch unknown (`Nature::Current`, zero capacitance row and column,
/// and conductance entries touching only `Nature::Voltage` unknowns).
#[derive(Clone)]
struct IdealSource {
    unknown: usize,
    terminals: Vec<(usize, f64)>,
}

/// Guard on one current-touching conductance slot for cached ideal-source validity.
#[derive(Clone, Copy)]
enum ConductanceGuard {
    NonZero(bool),
    Exact(u64),
}

impl ConductanceGuard {
    fn exact_bits(value: f64) -> u64 {
        if value == 0.0 { 0 } else { value.to_bits() }
    }

    fn matches(self, value: f64) -> bool {
        match self {
            Self::NonZero(nonzero) => (value != 0.0) == nonzero,
            Self::Exact(bits) => Self::exact_bits(value) == bits,
        }
    }
}

/// Cached ideal voltage sources and the current-touching slots guarding their validity.
struct CachedSources {
    sources: Vec<IdealSource>,
    watched: Vec<(usize, bool, ConductanceGuard)>,
}

impl CachedSources {
    fn build(system: Linearized<'_>) -> Self {
        let count = system.count();
        let mut has_capacitance = vec![false; count];
        let mut has_non_voltage = vec![false; count];
        let mut terminals: Vec<Vec<(usize, f64)>> = vec![Vec::new(); count];
        let mut watched = Vec::new();

        for (slot, (row, column)) in system.pattern.positions().enumerate() {
            let (row_nature, column_nature) = (system.natures[row], system.natures[column]);
            if row_nature != Nature::Current && column_nature != Nature::Current {
                continue;
            }
            let capacitance_nonzero = system.capacitance[slot] != 0.0;
            let conductance = system.conductance[slot];
            let guard = if column_nature == Nature::Current && row_nature == Nature::Voltage {
                ConductanceGuard::Exact(ConductanceGuard::exact_bits(conductance))
            } else {
                ConductanceGuard::NonZero(conductance != 0.0)
            };
            watched.push((slot, capacitance_nonzero, guard));
            if capacitance_nonzero {
                has_capacitance[row] = true;
                has_capacitance[column] = true;
            }
            if conductance != 0.0 {
                if column_nature == Nature::Current {
                    if row_nature == Nature::Voltage {
                        terminals[column].push((row, conductance));
                    } else {
                        has_non_voltage[column] = true;
                    }
                }
                if row_nature == Nature::Current && column_nature != Nature::Voltage {
                    has_non_voltage[row] = true;
                }
            }
        }

        let sources = terminals
            .into_iter()
            .enumerate()
            .filter_map(|(unknown, source_terminals)| {
                (system.natures[unknown] == Nature::Current
                    && !has_capacitance[unknown]
                    && !has_non_voltage[unknown]
                    && !source_terminals.is_empty())
                .then_some(IdealSource {
                    unknown,
                    terminals: source_terminals,
                })
            })
            .collect();
        Self { sources, watched }
    }

    fn matches(&self, system: Linearized<'_>) -> bool {
        self.watched
            .iter()
            .all(|&(slot, capacitance_nonzero, guard)| {
                (system.capacitance[slot] != 0.0) == capacitance_nonzero
                    && guard.matches(system.conductance[slot])
            })
    }
}

/// Whether a capacitive connected component is grounded or an ungrounded floating island.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Grounding {
    Grounded,
    Island,
}

/// Source of one numerical entry in a cached saddle-point matrix `K_S`.
#[derive(Clone, Copy)]
enum EntrySource {
    Capacitance(usize),
    Constant(f64),
    Conductance(usize),
}

/// Cached sparse pattern, ordering, entry map, and LU solver for one component's saddle matrix.
struct CachedSaddle {
    grounding: Grounding,
    pattern: Arc<Pattern>,
    entries: Vec<(usize, EntrySource)>,
    values: Vec<f64>,
    previous: Vec<f64>,
    right_hand_side: Vec<f64>,
    solver: SparseLu<f64>,
    factored: bool,
}

impl CachedSaddle {
    fn build(
        system: Linearized<'_>,
        component: &[usize],
        position: &[usize],
        sources: &[IdealSource],
        grounding: Grounding,
    ) -> Option<Self> {
        let (node_count, source_count) = (component.len(), sources.len());
        let is_island = grounding == Grounding::Island;
        let dimension = node_count + source_count + usize::from(is_island);
        let mut builder = PatternBuilder::new();
        let mut raw: Vec<(Entry, EntrySource)> = Vec::new();
        let mut push = |row, column, source| raw.push((builder.entry(row, column), source));

        for (local_row, &node) in component.iter().enumerate() {
            for &(slot, column) in system.pattern.row(node) {
                let local_column = position[column];
                if local_column != ABSENT && system.capacitance[slot] != 0.0 {
                    push(local_row, local_column, EntrySource::Capacitance(slot));
                }
            }
        }
        for (source_index, source) in sources.iter().enumerate() {
            let branch = node_count + source_index;
            for &(node, stamp) in &source.terminals {
                push(position[node], branch, EntrySource::Constant(stamp));
            }
            for &(slot, column) in system.pattern.row(source.unknown) {
                let local_column = position[column];
                if local_column != ABSENT && system.conductance[slot] != 0.0 {
                    push(branch, local_column, EntrySource::Conductance(slot));
                }
            }
        }
        if is_island {
            let pin = node_count + source_count;
            for local in 0..node_count {
                push(local, pin, EntrySource::Constant(1.0));
                push(pin, local, EntrySource::Constant(1.0));
            }
        }

        let pattern = Arc::new(builder.finish(dimension));
        let ordering = Ordering::new(Arc::clone(&pattern)).ok()?;
        let entries = raw
            .into_iter()
            .map(|(entry, source)| (pattern.slot(entry), source))
            .collect();
        Some(Self {
            grounding,
            values: vec![0.0; pattern.len()],
            previous: vec![0.0; pattern.len()],
            right_hand_side: vec![0.0; dimension],
            pattern,
            entries,
            solver: SparseLu::new(Arc::new(ordering)),
            factored: false,
        })
    }

    fn fill(&mut self, system: Linearized<'_>, capacitance_scale: f64) {
        std::mem::swap(&mut self.previous, &mut self.values);
        self.values.fill(0.0);
        for &(saddle_slot, source) in &self.entries {
            self.values[saddle_slot] += match source {
                EntrySource::Capacitance(slot) => system.capacitance[slot] / capacitance_scale,
                EntrySource::Constant(value) => value,
                EntrySource::Conductance(slot) => system.conductance[slot],
            };
        }
        // Reset pivots when entries change so nonlinear steps factor bit-for-bit identically to
        // a fresh solver while linear steps still reuse the cached LU factors verbatim.
        if self.factored
            && self
                .previous
                .iter()
                .zip(&self.values)
                .any(|(&before, &after)| before.to_bits() != after.to_bits())
        {
            self.solver.reset();
            self.factored = false;
        }
    }
}

/// One voltage-capacitive connected component and its lazily cached saddle solver.
struct CachedComponent {
    nodes: Vec<usize>,
    sources: Vec<IdealSource>,
    saddle: Option<CachedSaddle>,
}

impl CachedComponent {
    fn check(
        &mut self,
        system: Linearized<'_>,
        position: &mut [usize],
        run: &Run<'_>,
        options: &Options,
    ) -> bool {
        for (local, &node) in self.nodes.iter().enumerate() {
            position[node] = local;
        }
        let (scale, grounding) =
            capacitance_scale_and_grounding(system, &self.nodes, position, &self.sources);
        if scale > 0.0
            && self
                .saddle
                .as_ref()
                .is_none_or(|saddle| saddle.grounding != grounding)
        {
            self.saddle =
                CachedSaddle::build(system, &self.nodes, position, &self.sources, grounding);
        }
        for &node in &self.nodes {
            position[node] = ABSENT;
        }
        let Some(saddle) = self.saddle.as_mut().filter(|_| scale > 0.0) else {
            return false;
        };
        saddle.fill(system, scale);

        let CachedSaddle {
            pattern,
            values,
            right_hand_side,
            solver,
            factored,
            ..
        } = saddle;
        let Ok(mut factors) = solver.factor(values) else {
            solver.reset();
            *factored = false;
            return false;
        };
        *factored = true;

        let mut loop_derivatives = vec![[0.0; 4]; self.sources.len()];
        let mut loop_charges = vec![[0.0; 4]; self.sources.len()];
        let mut rounding_floor = 0.0_f64;
        for (step, (&derivative_slice, &charge_slice)) in
            run.derivatives.iter().zip(&run.charges).enumerate()
        {
            if !solve_saddle_slice(
                &mut factors,
                right_hand_side,
                &self.nodes,
                derivative_slice,
                &mut loop_derivatives,
                step,
            ) {
                return false;
            }
            let floor = saddle_rounding_floor(pattern, values, self.nodes.len(), right_hand_side);
            rounding_floor = rounding_floor.max(floor);
            if !solve_saddle_slice(
                &mut factors,
                right_hand_side,
                &self.nodes,
                charge_slice,
                &mut loop_charges,
                step,
            ) {
                return false;
            }
        }

        let absolute = options.equation_tolerance(Nature::Voltage);
        loop_derivatives
            .iter()
            .zip(&loop_charges)
            .any(|(&derivatives, &charges)| {
                let tolerance =
                    alternating_tolerance(derivatives, absolute, options).max(rounding_floor);
                turns_alternate(derivatives, tolerance)
                    && !is_harmonic_scalar(run.times, charges, derivatives)
            })
    }
}

/// Cached capacitive connected components and reusable scratch maps for stage 2d.
struct CachedLoop {
    capacitance_mask: Vec<(usize, bool)>,
    components: Vec<CachedComponent>,
    active_marks: Vec<bool>,
    position: Vec<usize>,
}

impl CachedLoop {
    fn build(system: Linearized<'_>, ideal_sources: &[IdealSource]) -> Self {
        let count = system.count();
        let mut capacitance_mask = Vec::new();
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); count];
        for (slot, (row, column)) in system.pattern.positions().enumerate() {
            if system.natures[row] == Nature::Voltage && system.natures[column] == Nature::Voltage {
                let nonzero = system.capacitance[slot] != 0.0;
                capacitance_mask.push((slot, nonzero));
                if row != column && nonzero {
                    adjacency[row].push(column);
                    adjacency[column].push(row);
                }
            }
        }
        for source in ideal_sources {
            if let Some((&(anchor, _), rest)) = source.terminals.split_first() {
                for &(other, _) in rest {
                    adjacency[anchor].push(other);
                    adjacency[other].push(anchor);
                }
            }
        }

        let mut visited = vec![false; count];
        let mut position = vec![ABSENT; count];
        let mut queue = VecDeque::new();
        let mut components = Vec::new();
        for source in ideal_sources {
            let Some(&(start, _)) = source.terminals.first() else {
                continue;
            };
            if visited[start] {
                continue;
            }
            let mut nodes = Vec::new();
            visited[start] = true;
            queue.push_back(start);
            while let Some(node) = queue.pop_front() {
                nodes.push(node);
                for &neighbor in &adjacency[node] {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
            nodes.sort_unstable();
            for (local, &node) in nodes.iter().enumerate() {
                position[node] = local;
            }
            let sources: Vec<IdealSource> = ideal_sources
                .iter()
                .filter(|candidate| {
                    candidate
                        .terminals
                        .iter()
                        .all(|&(node, _)| position[node] != ABSENT)
                })
                .cloned()
                .collect();
            for &node in &nodes {
                position[node] = ABSENT;
            }
            if !sources.is_empty() {
                components.push(CachedComponent {
                    nodes,
                    sources,
                    saddle: None,
                });
            }
        }

        Self {
            capacitance_mask,
            components,
            active_marks: vec![false; count],
            position,
        }
    }

    fn matches(&self, system: Linearized<'_>) -> bool {
        self.capacitance_mask
            .iter()
            .all(|&(slot, nonzero)| (system.capacitance[slot] != 0.0) == nonzero)
    }
}

/// Stateful trapezoidal ringing detector for a single transient run.
///
/// Lazily caches ideal voltage sources and per-component saddle-point symbolic/LU data on
/// the first alternating trapezoidal step, reusing them across subsequent steps whenever the
/// underlying matrix sparsity structure is unchanged.
#[derive(Default)]
pub(crate) struct Detector {
    sources: Option<CachedSources>,
    loop_cache: Option<CachedLoop>,
}

impl Detector {
    /// Whether the trapezoidal formula is numerically ringing at the end of `run`.
    ///
    /// The step Jacobian `G + α C` and the device matrices `G`, `C` must remain in `workspace`
    /// from the step's error-filter pass; the cached factorization in `solver` is reused
    /// without refactoring whenever `workspace.jacobian` is bitwise unchanged.
    pub(crate) fn rings(
        &mut self,
        pattern: &Pattern,
        natures: &[Nature],
        solver: &mut Solver<f64>,
        workspace: &mut Workspace,
        run: &Run<'_>,
        options: &Options,
    ) -> bool {
        let count = natures.len();
        let inside = options.parallelism.inside(count);
        if all(count, inside, |index| {
            !unknown_alternates(natures, run, index, options)
        }) {
            return false;
        }
        let candidates: Vec<usize> = (0..count)
            .filter(|&index| unknown_alternates(natures, run, index, options))
            .collect();
        let suspects: Vec<usize> = candidates
            .iter()
            .copied()
            .filter(|&index| {
                !is_harmonic_scalar(
                    run.times,
                    run.charges.map(|slice| slice[index]),
                    run.derivatives.map(|slice| slice[index]),
                )
            })
            .collect();

        let Workspace {
            contributions,
            jacobian,
            moved,
            meter,
            ..
        } = workspace;
        let system = Linearized {
            pattern,
            conductance: &contributions.conductance,
            capacitance: &contributions.capacitance,
            natures,
        };

        let has_ideal_sources = self.refresh_sources(system);
        if suspects.is_empty() && !has_ideal_sources {
            return false;
        }
        if !suspects.is_empty()
            && has_dissipative_ringing(system, solver, jacobian, moved, meter, &suspects, run)
        {
            return true;
        }
        has_ideal_sources && self.has_voltage_source_loop_ringing(system, &candidates, run, options)
    }

    fn refresh_sources(&mut self, system: Linearized<'_>) -> bool {
        if self
            .sources
            .as_ref()
            .is_none_or(|cached| !cached.matches(system))
        {
            self.loop_cache = None;
            self.sources = Some(CachedSources::build(system));
        }
        self.sources
            .as_ref()
            .is_some_and(|cached| !cached.sources.is_empty())
    }

    /// Exact saddle-point projection isolating algebraic `R = −1` ringing inside
    /// capacitor-voltage-source loops (both grounded and floating). Callers must have
    /// refreshed `self.sources` for `system` via [`Detector::refresh_sources`].
    fn has_voltage_source_loop_ringing(
        &mut self,
        system: Linearized<'_>,
        active: &[usize],
        run: &Run<'_>,
        options: &Options,
    ) -> bool {
        let Some(ideal_sources) = self
            .sources
            .as_ref()
            .map(|cached| cached.sources.as_slice())
        else {
            return false;
        };
        if ideal_sources.is_empty() {
            return false;
        }
        let cache = self
            .loop_cache
            .get_or_insert_with(|| CachedLoop::build(system, ideal_sources));
        if !cache.matches(system) {
            *cache = CachedLoop::build(system, ideal_sources);
        }

        for &index in active {
            cache.active_marks[index] = true;
        }
        let active_marks = &cache.active_marks;
        let position = &mut cache.position;
        let detected = cache.components.iter_mut().any(|component| {
            let touched = component.nodes.iter().any(|&node| active_marks[node]);
            touched && component.check(system, position, run, options)
        });
        for &index in active {
            cache.active_marks[index] = false;
        }
        detected
    }
}

/// Solves `factors` with unit vector `e_{index}` written into `buffer`.
fn solve_unit(
    factors: &mut Factors<'_, f64>,
    buffer: &mut Vec<f64>,
    count: usize,
    index: usize,
) -> bool {
    buffer.clear();
    buffer.resize(count, 0.0);
    buffer[index] = 1.0;
    factors.solve(buffer).is_ok()
}

/// Runs stages 2a, 2b, and 2c using the cached `LU` factors of `G + α C`.
fn has_dissipative_ringing(
    system: Linearized<'_>,
    solver: &mut Solver<f64>,
    jacobian: &[f64],
    buffer: &mut Vec<f64>,
    meter: &mut Meter,
    suspects: &[usize],
    run: &Run<'_>,
) -> bool {
    let domains = classify_domains(system);
    let needs_displacement = suspects
        .iter()
        .any(|&index| domains[index].0 == Domain::Displacement);
    let mut non_skew: Vec<Domain> = suspects
        .iter()
        .map(|&index| domains[index].0)
        .filter(|&domain| domain != Domain::Displacement)
        .collect();
    non_skew.sort_unstable();
    non_skew.dedup();
    non_skew.retain(|&domain| !domain_is_skew_lossless(system, &domains, domain));
    if !needs_displacement && non_skew.is_empty() {
        return false;
    }

    let previous = meter.enter(Activity::Factoring);
    let Ok(mut factors) = solver.factor(jacobian) else {
        meter.leave(previous);
        return true;
    };
    meter.enter(Activity::Solving);
    let detected = check_dissipative_stages(
        system,
        &mut factors,
        buffer,
        &domains,
        &non_skew,
        suspects,
        run,
    );
    meter.leave(previous);
    detected
}

/// Runs stages 2a, 2b, and 2c against already-computed LU `factors`.
fn check_dissipative_stages(
    system: Linearized<'_>,
    factors: &mut Factors<'_, f64>,
    buffer: &mut Vec<f64>,
    domains: &[(Domain, f64)],
    non_skew: &[Domain],
    suspects: &[usize],
    run: &Run<'_>,
) -> bool {
    let count = system.count();
    for &index in suspects {
        if domains[index].0 != Domain::Displacement {
            continue;
        }
        if !solve_unit(factors, buffer, count, index)
            || run.alpha * system.row_dot(system.capacitance, index, buffer)
                < TRAPEZOIDAL_REAL_FILTER_BOUND
        {
            return true;
        }
    }
    for &domain in non_skew {
        buffer.clear();
        buffer.resize(count, 0.0);
        for &index in suspects {
            if domains[index].0 == domain {
                let [value_0, value_1, value_2, value_3] =
                    run.derivatives.map(|slice| slice[index]);
                buffer[index] = value_3 - 3.0 * value_2 + 3.0 * value_1 - value_0;
            }
        }
        if factors.solve(buffer).is_err()
            || is_dissipative_response(system, run.alpha, domains, domain, buffer)
        {
            return true;
        }
    }
    for &index in suspects {
        let domain = domains[index].0;
        if domain == Domain::Displacement || !non_skew.contains(&domain) {
            continue;
        }
        if !solve_unit(factors, buffer, count, index)
            || is_dissipative_response(system, run.alpha, domains, domain, buffer)
        {
            return true;
        }
    }
    false
}

/// The threshold on consecutive derivative differences over `values`.
fn alternating_tolerance(values: [f64; 4], absolute: f64, options: &Options) -> f64 {
    let largest = values
        .iter()
        .fold(0.0_f64, |most, value| most.max(value.abs()));
    options.truncation_tolerance * (options.relative_tolerance * largest + absolute)
}

/// Whether unknown `index`'s derivative over `run` turns back and forth twice beyond tolerance.
fn unknown_alternates(natures: &[Nature], run: &Run<'_>, index: usize, options: &Options) -> bool {
    let values = run.derivatives.map(|slice| slice[index]);
    let tolerance =
        alternating_tolerance(values, options.equation_tolerance(natures[index]), options);
    turns_alternate(values, tolerance)
}

/// Whether `values` turns back and forth twice in a row, each turn exceeding `tolerance`.
#[must_use]
fn turns_alternate(values: [f64; 4], tolerance: f64) -> bool {
    let turns = [
        values[1] - values[0],
        values[2] - values[1],
        values[3] - values[2],
    ];
    turns[0] * turns[1] < 0.0
        && turns[1] * turns[2] < 0.0
        && turns.iter().all(|turn| turn.abs() > tolerance)
}

/// Exact variable-step scalar restoring-force check for a single-mode linear oscillator.
///
/// On `q'' = −ω² (q − q_eq)` integrated by the trapezoidal rule over arbitrary step sizes
/// `h_k = t_{k+1} − t_k > 0`, the midpoint acceleration `a_k = (y_{k+1} − y_k) / h_k` equals
/// `−ω² (q_{k+1} + q_k) / 2 + ω² q_eq` identically. Subtracting adjacent midpoints yields
/// `a_k − a_{k−1} = −ω² (q_{k+1} − q_{k−1}) / 2`, so both products
/// `p_1 = (a_1 − a_0) · (q_2 − q_0) / 2` and `p_2 = (a_2 − a_1) · (q_3 − q_1) / 2` are
/// non-positive and at least one is strictly negative outside floating-point rounding.
#[must_use]
fn is_harmonic_scalar(times: [f64; 4], charges: [f64; 4], derivatives: [f64; 4]) -> bool {
    let steps = [
        times[1] - times[0],
        times[2] - times[1],
        times[3] - times[2],
    ];
    if steps.iter().any(|&step| step <= 0.0) {
        return false;
    }
    let accelerations = [
        (derivatives[1] - derivatives[0]) / steps[0],
        (derivatives[2] - derivatives[1]) / steps[1],
        (derivatives[3] - derivatives[2]) / steps[2],
    ];
    let pair = |index: usize| {
        let charge_shift = 0.5 * (charges[index + 2] - charges[index]);
        let product = (accelerations[index + 1] - accelerations[index]) * charge_shift;
        let floor = rounding(
            f64::midpoint(accelerations[index + 1].abs(), accelerations[index].abs())
                * (charges[index + 2].abs() + charges[index].abs()),
        );
        (product, floor)
    };
    let (product_1, floor_1) = pair(0);
    let (product_2, floor_2) = pair(1);
    product_1 <= floor_1 && product_2 <= floor_2 && (product_1 < -floor_1 || product_2 < -floor_2)
}

/// Classifies every unknown into its physical domain and sign `s_i` (`+1` or `−1`).
///
/// Current rows with `C_{i,i} ≤ 0` take sign `−1` so `S C ≥ 0` and lossless `S G` is
/// skew-symmetric in watts. Internal BJT/VBIC excess-phase filters stamp `C_{i,i} = +delay > 0`
/// in units of `A²`, so they are kept in [`Domain::Filter`] with sign `+1`.
fn classify_domains(system: Linearized<'_>) -> Vec<(Domain, f64)> {
    system
        .natures
        .iter()
        .enumerate()
        .map(|(index, &nature)| match nature {
            Nature::Voltage => (Domain::Electrical, 1.0),
            Nature::Current if system.entry(system.capacitance, index, index) > 0.0 => {
                (Domain::Filter, 1.0)
            }
            Nature::Current => (Domain::Electrical, -1.0),
            Nature::Temperature => (Domain::Thermal, 1.0),
            Nature::State => (Domain::State, 1.0),
            Nature::Displacement => (Domain::Displacement, 1.0),
        })
        .collect()
}

/// Whether `S G` restricted to `domain` is strictly skew-symmetric within rounding and
/// decoupled from unknowns outside `domain`.
fn domain_is_skew_lossless(
    system: Linearized<'_>,
    domains: &[(Domain, f64)],
    domain: Domain,
) -> bool {
    for (row, &(row_domain, row_sign)) in domains.iter().enumerate() {
        if row_domain != domain {
            continue;
        }
        for &(slot, column) in system.pattern.row(row) {
            let value = system.conductance[slot];
            let (column_domain, column_sign) = domains[column];
            if column_domain != domain {
                if value != 0.0 {
                    return false;
                }
                continue;
            }
            let transpose = system.entry(system.conductance, column, row);
            if (row_sign * value + column_sign * transpose).abs()
                > rounding(value.abs() + transpose.abs())
            {
                return false;
            }
        }
    }
    true
}

/// Whether the linearized response `response = (G + α C)⁻¹ v` exhibits stiff dissipative
/// decay (`σ ≥ α = 2/h`) inside `domain`.
///
/// Accumulates `dissipation = ∑ s_i u_i (G u)_i` and `reactance = ∑ s_i u_i (α C u)_i` in
/// strict ascending index order `0..n` for thread-count bitwise determinism. A lossless
/// algebraic nullspace where both vanish is deferred to
/// [`Detector::has_voltage_source_loop_ringing`].
fn is_dissipative_response(
    system: Linearized<'_>,
    alpha: f64,
    domains: &[(Domain, f64)],
    domain: Domain,
    response: &[f64],
) -> bool {
    let (mut dissipation, mut dissipation_scale) = (0.0, 0.0);
    let (mut reactance, mut reactance_scale) = (0.0, 0.0);
    for (row, &(row_domain, sign)) in domains.iter().enumerate() {
        if row_domain != domain {
            continue;
        }
        let (unknown, unknown_magnitude) = (response[row], response[row].abs());
        let (mut conductance_dot, mut conductance_magnitude) = (0.0, 0.0);
        let (mut capacitance_dot, mut capacitance_magnitude) = (0.0, 0.0);
        for &(slot, column) in system.pattern.row(row) {
            let (entry_unknown, entry_magnitude) = (response[column], response[column].abs());
            let (entry_conductance, entry_capacitance) =
                (system.conductance[slot], system.capacitance[slot]);
            conductance_dot += entry_conductance * entry_unknown;
            conductance_magnitude += entry_conductance.abs() * entry_magnitude;
            capacitance_dot += entry_capacitance * entry_unknown;
            capacitance_magnitude += entry_capacitance.abs() * entry_magnitude;
        }
        dissipation += sign * unknown * conductance_dot;
        dissipation_scale += unknown_magnitude * conductance_magnitude;
        reactance += sign * unknown * (alpha * capacitance_dot);
        reactance_scale += unknown_magnitude * (alpha * capacitance_magnitude);
    }
    if reactance > rounding(reactance_scale) {
        dissipation >= reactance
    } else {
        dissipation > rounding(dissipation_scale)
    }
}

/// Whether `values` sum to zero within floating-point rounding of their magnitudes.
fn sums_to_zero(values: impl IntoIterator<Item = f64>) -> bool {
    let (sum, magnitude) = values
        .into_iter()
        .fold((0.0, 0.0), |(total, scale), value| {
            (total + value, scale + value.abs())
        });
    sum.abs() <= rounding(magnitude)
}

/// Computes the largest capacitance magnitude in `component` (to scale `C_SS` to O(1) without
/// changing `μ`) and whether `component` is an ungrounded capacitive island.
fn capacitance_scale_and_grounding(
    system: Linearized<'_>,
    component: &[usize],
    position: &[usize],
    sources: &[IdealSource],
) -> (f64, Grounding) {
    let node_count = component.len();
    let mut capacitance_scale = 0.0_f64;
    let (mut row_sums, mut row_magnitudes) = (vec![0.0; node_count], vec![0.0; node_count]);
    let (mut column_sums, mut column_magnitudes) = (vec![0.0; node_count], vec![0.0; node_count]);
    for (local_row, &node) in component.iter().enumerate() {
        for &(slot, column) in system.pattern.row(node) {
            let local_column = position[column];
            let entry = system.capacitance[slot];
            if local_column == ABSENT || entry == 0.0 {
                continue;
            }
            let magnitude = entry.abs();
            capacitance_scale = capacitance_scale.max(magnitude);
            row_sums[local_row] += entry;
            row_magnitudes[local_row] += magnitude;
            column_sums[local_column] += entry;
            column_magnitudes[local_column] += magnitude;
        }
    }
    if capacitance_scale == 0.0 {
        return (0.0, Grounding::Grounded);
    }
    let zero_within_rounding = |sums: &[f64], magnitudes: &[f64]| {
        sums.iter()
            .zip(magnitudes)
            .all(|(&sum, &magnitude)| sum.abs() <= rounding(magnitude))
    };
    let sources_balanced = sources.iter().all(|source| {
        sums_to_zero(source.terminals.iter().map(|&(_, stamp)| stamp))
            && sums_to_zero(system.pattern.row(source.unknown).iter().filter_map(
                |&(slot, column)| (position[column] != ABSENT).then_some(system.conductance[slot]),
            ))
    });
    let grounding = if zero_within_rounding(&row_sums, &row_magnitudes)
        && zero_within_rounding(&column_sums, &column_magnitudes)
        && sources_balanced
    {
        Grounding::Island
    } else {
        Grounding::Grounded
    };
    (capacitance_scale, grounding)
}

/// Solves one saddle right-hand side populated from `slice` over `component` and records the
/// resulting Lagrange multipliers into `series[source][step]`.
fn solve_saddle_slice(
    factors: &mut jolty_numerics::Factors<'_, f64>,
    right_hand_side: &mut [f64],
    component: &[usize],
    slice: &[f64],
    series: &mut [[f64; 4]],
    step: usize,
) -> bool {
    right_hand_side.fill(0.0);
    for (local_row, &node) in component.iter().enumerate() {
        right_hand_side[local_row] = slice[node];
    }
    if factors.solve(right_hand_side).is_err() {
        return false;
    }
    let node_count = component.len();
    for (source_index, target) in series.iter_mut().enumerate() {
        target[step] = right_hand_side[node_count + source_index];
    }
    true
}

/// Maximum componentwise residual rounding floor `8 ε · (|K_S| |solution|)` over the
/// capacitive rows `0..node_count` (Higham, 2002, §12.3), in amperes.
fn saddle_rounding_floor(
    pattern: &Pattern,
    values: &[f64],
    node_count: usize,
    solution: &[f64],
) -> f64 {
    (0..node_count).fold(0.0_f64, |floor, row_index| {
        let resolution = pattern
            .row(row_index)
            .iter()
            .fold(0.0, |total, &(slot, column)| {
                total + values[slot].abs() * solution[column].abs()
            });
        floor.max(rounding(resolution))
    })
}
