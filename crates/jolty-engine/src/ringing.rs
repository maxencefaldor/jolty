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
    /// The number of unknowns in the system.
    fn count(self) -> usize {
        self.natures.len()
    }

    /// The value at `(row, column)` in `values`, or `0.0` when structurally absent.
    fn entry(self, values: &[f64], row: usize, column: usize) -> f64 {
        let entries = self.pattern.row(row);
        entries
            .binary_search_by_key(&column, |&(_, entry_column)| entry_column)
            .map_or(0.0, |position| values[entries[position].0])
    }

    /// Dot product of `row` of `values` with `vector`, in ascending column order.
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
    /// Node voltages (with sign `+1`) and electrical branch currents carrying `C_{i,i} ≤ 0`
    /// (with sign `−1`), so `uᵀ S G u` and `uᵀ S (α C) u` both have units of watts.
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
struct IdealSource {
    unknown: usize,
    /// Voltage nodes appearing in column `unknown` of `G` (`G[row, unknown] ≠ 0`).
    terminals: Vec<(usize, f64)>,
}

/// Assembled saddle-point system for one capacitive connected component.
struct SaddleSystem {
    pattern: Arc<Pattern>,
    values: Vec<f64>,
    dimension: usize,
}

/// Mutable borrows needed to solve against the cached step Jacobian `G + α C`.
struct CachedSolve<'a> {
    solver: &'a mut Solver<f64>,
    jacobian: &'a [f64],
    buffer: &'a mut Vec<f64>,
    meter: &'a mut Meter,
}

/// Suspects and domain classifications shared across the passive-probe checks.
struct ProbeContext<'a> {
    domains: &'a [(Domain, f64)],
    non_skew_domains: &'a [Domain],
    suspects: &'a [usize],
}

/// Whether the trapezoidal formula is numerically ringing at the end of `run`.
///
/// The step Jacobian `G + α C` and the device matrices `G`, `C` must remain in `workspace`
/// from the step's error-filter pass; the cached factorization in `solver` is reused without
/// refactoring whenever `workspace.jacobian` is bitwise unchanged.
pub(crate) fn rings(
    pattern: &Pattern,
    natures: &[Nature],
    solver: &mut Solver<f64>,
    workspace: &mut Workspace,
    run: &Run<'_>,
    options: &Options,
) -> bool {
    let count = natures.len();
    let inside = options.parallelism.inside(count);

    // Fast path: match the original zero-allocation scan across all unknowns.
    if all(count, inside, |index| {
        !unknown_alternates(natures, run, index, options)
    }) {
        return false;
    }

    // Collect every alternating unknown in strict ascending index order for determinism.
    let candidates: Vec<usize> = (0..count)
        .filter(|&index| unknown_alternates(natures, run, index, options))
        .collect();

    // Stage 1: exact variable-step scalar restoring-force identity (O(1), zero solves).
    let suspects: Vec<usize> = candidates
        .iter()
        .copied()
        .filter(|&index| {
            let charges = run.charges.map(|slice| slice[index]);
            let derivatives = run.derivatives.map(|slice| slice[index]);
            !is_harmonic_scalar(run.times, charges, derivatives)
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

    // Cheap early exit for circuits with no ideal voltage sources (such as pure LC tanks
    // driven by current sources): when every alternating unknown satisfies the scalar
    // harmonic identity, no algebraic capacitor-voltage-source loop exists to hide beneath
    // the oscillation, so return without factoring or solving.
    let ideal_sources = find_ideal_sources(system);
    if suspects.is_empty() && ideal_sources.is_empty() {
        return false;
    }

    if !suspects.is_empty() {
        let mut cached = CachedSolve {
            solver,
            jacobian,
            buffer: moved,
            meter,
        };
        if has_dissipative_ringing(system, &mut cached, &suspects, run) {
            return true;
        }
    }

    // Stage 2d: exact saddle-point projection for capacitor-voltage-source loops (both
    // grounded and floating), isolating algebraic `R = −1` ringing without touching `solver`.
    // Runs on all alternating `candidates` (not only `suspects`) because a large single-mode
    // LC tank coupled through a capacitor to either a grounded or floating source can make
    // that source's terminal satisfy `is_harmonic_scalar` while hiding smaller corner-induced
    // ringing.
    !ideal_sources.is_empty()
        && has_voltage_source_loop_ringing(system, &ideal_sources, &candidates, run, options)
}

/// Runs stages 2a, 2b, and 2c (displacement diagonal, per-domain Nyquist third difference,
/// and per-suspect unit probes) using the cached `LU` factors of `G + α C`.
fn has_dissipative_ringing(
    system: Linearized<'_>,
    cached: &mut CachedSolve<'_>,
    suspects: &[usize],
    run: &Run<'_>,
) -> bool {
    let domains = classify_domains(system);
    let needs_displacement = suspects
        .iter()
        .any(|&index| domains[index].0 == Domain::Displacement);

    let mut active_domains: Vec<Domain> = suspects
        .iter()
        .map(|&index| domains[index].0)
        .filter(|&domain| domain != Domain::Displacement)
        .collect();
    active_domains.sort_unstable();
    active_domains.dedup();

    // A domain whose signed conductance `S G` is strictly skew-symmetric and decoupled from
    // every other domain has `uᵀ S G u = 0` identically, so skip all linear solves for it.
    let non_skew_domains: Vec<Domain> = active_domains
        .into_iter()
        .filter(|&domain| !domain_is_skew_lossless(system, &domains, domain))
        .collect();

    if !needs_displacement && non_skew_domains.is_empty() {
        return false;
    }

    let previous = cached.meter.enter(Activity::Factoring);
    let Ok(mut factors) = cached.solver.factor(cached.jacobian) else {
        cached.meter.leave(previous);
        // Conservative fallback when the matrix cannot be factored.
        return true;
    };
    cached.meter.enter(Activity::Solving);

    let context = ProbeContext {
        domains: &domains,
        non_skew_domains: &non_skew_domains,
        suspects,
    };
    let detected = (needs_displacement
        && check_displacement_suspects(system, &mut factors, cached.buffer, &context, run.alpha))
        || check_domain_nyquist(system, &mut factors, cached.buffer, &context, run)
        || check_suspect_probes(system, &mut factors, cached.buffer, &context, run.alpha);
    cached.meter.leave(previous);
    detected
}

/// Stage 2a: checks Meyer displacement suspects (`R_{i,i} = 2 F_{i,i} − 1 < 0` iff `F_{i,i} < 0.5`).
fn check_displacement_suspects(
    system: Linearized<'_>,
    factors: &mut Factors<'_, f64>,
    buffer: &mut Vec<f64>,
    context: &ProbeContext<'_>,
    alpha: f64,
) -> bool {
    let count = system.count();
    for &index in context.suspects {
        if context.domains[index].0 != Domain::Displacement {
            continue;
        }
        buffer.clear();
        buffer.resize(count, 0.0);
        buffer[index] = 1.0;
        if factors.solve(buffer).is_err() {
            return true;
        }
        let filtered_entry = alpha * system.row_dot(system.capacitance, index, buffer);
        if filtered_entry < TRAPEZOIDAL_REAL_FILTER_BOUND {
            return true;
        }
    }
    false
}

/// Stage 2b: probes each non-skew domain with the collective Nyquist third difference
/// `y_3 − 3 y_2 + 3 y_1 − y_0` restricted to its suspects, catching critically damped and
/// overdamped RLC modes at every `ω h`.
fn check_domain_nyquist(
    system: Linearized<'_>,
    factors: &mut Factors<'_, f64>,
    buffer: &mut Vec<f64>,
    context: &ProbeContext<'_>,
    run: &Run<'_>,
) -> bool {
    let count = system.count();
    for &domain in context.non_skew_domains {
        buffer.clear();
        buffer.resize(count, 0.0);
        for &index in context.suspects {
            if context.domains[index].0 == domain {
                let value_0 = run.derivatives[0][index];
                let value_1 = run.derivatives[1][index];
                let value_2 = run.derivatives[2][index];
                let value_3 = run.derivatives[3][index];
                buffer[index] = value_3 - 3.0 * value_2 + 3.0 * value_1 - value_0;
            }
        }
        if factors.solve(buffer).is_err() {
            return true;
        }
        if is_dissipative_response(system, run.alpha, context.domains, domain, buffer) {
            return true;
        }
    }
    false
}

/// Stage 2c: probes each suspect individually with unit vector `e_i`, preventing a large
/// multi-mode lossless tank in the same domain from swamping a small stiff RC branch.
fn check_suspect_probes(
    system: Linearized<'_>,
    factors: &mut Factors<'_, f64>,
    buffer: &mut Vec<f64>,
    context: &ProbeContext<'_>,
    alpha: f64,
) -> bool {
    let count = system.count();
    for &index in context.suspects {
        let domain = context.domains[index].0;
        if domain == Domain::Displacement || !context.non_skew_domains.contains(&domain) {
            continue;
        }
        buffer.clear();
        buffer.resize(count, 0.0);
        buffer[index] = 1.0;
        if factors.solve(buffer).is_err() {
            return true;
        }
        if is_dissipative_response(system, alpha, context.domains, domain, buffer) {
            return true;
        }
    }
    false
}

/// Whether unknown `index`'s derivative over the four points of `run` turns back and forth
/// twice in a row by more than its tolerance.
fn unknown_alternates(natures: &[Nature], run: &Run<'_>, index: usize, options: &Options) -> bool {
    let values = run.derivatives.map(|slice| slice[index]);
    let absolute = options.equation_tolerance(natures[index]);
    let largest = values
        .iter()
        .fold(0.0_f64, |most, value| most.max(value.abs()));
    let tolerance =
        options.truncation_tolerance * (options.relative_tolerance * largest + absolute);
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
    let step_0 = times[1] - times[0];
    let step_1 = times[2] - times[1];
    let step_2 = times[3] - times[2];
    if step_0 <= 0.0 || step_1 <= 0.0 || step_2 <= 0.0 {
        return false;
    }
    let acceleration_0 = (derivatives[1] - derivatives[0]) / step_0;
    let acceleration_1 = (derivatives[2] - derivatives[1]) / step_1;
    let acceleration_2 = (derivatives[3] - derivatives[2]) / step_2;
    let charge_shift_1 = 0.5 * (charges[2] - charges[0]);
    let charge_shift_2 = 0.5 * (charges[3] - charges[1]);
    let product_1 = (acceleration_1 - acceleration_0) * charge_shift_1;
    let product_2 = (acceleration_2 - acceleration_1) * charge_shift_2;
    let floor_1 = rounding(
        (acceleration_1.abs() + acceleration_0.abs()) * 0.5 * (charges[2].abs() + charges[0].abs()),
    );
    let floor_2 = rounding(
        (acceleration_2.abs() + acceleration_1.abs()) * 0.5 * (charges[3].abs() + charges[1].abs()),
    );
    product_1 <= floor_1 && product_2 <= floor_2 && (product_1 < -floor_1 || product_2 < -floor_2)
}

/// Classifies every unknown into its physical domain and sign `s_i` (`+1` or `−1`).
///
/// Current rows with `C_{i,i} ≤ 0` (inductors, coupled inductors, voltage sources, small
/// resistors, transmission lines) take sign `−1` so `S C ≥ 0` and lossless `S G` is
/// skew-symmetric in watts. Internal BJT/VBIC excess-phase filters stamp `C_{i,i} = +delay > 0`
/// in units of `A²`, so they are kept in [`Domain::Filter`] with sign `+1`.
fn classify_domains(system: Linearized<'_>) -> Vec<(Domain, f64)> {
    system
        .natures
        .iter()
        .enumerate()
        .map(|(index, &nature)| match nature {
            Nature::Voltage => (Domain::Electrical, 1.0),
            Nature::Current => {
                if system.entry(system.capacitance, index, index) > 0.0 {
                    (Domain::Filter, 1.0)
                } else {
                    (Domain::Electrical, -1.0)
                }
            }
            Nature::Temperature => (Domain::Thermal, 1.0),
            Nature::State => (Domain::State, 1.0),
            Nature::Displacement => (Domain::Displacement, 1.0),
        })
        .collect()
}

/// Whether `S G` restricted to `domain` is strictly skew-symmetric (within floating-point
/// rounding) and receives zero coupling from unknowns outside `domain`.
///
/// When true, `uᵀ S G u = 0` identically for every real vector `u`, guaranteeing that no
/// dissipative decay mode can exist in `domain`.
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
            let symmetric_part = row_sign * value + column_sign * transpose;
            if symmetric_part.abs() > rounding(value.abs() + transpose.abs()) {
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
/// strict ascending index order `0..n` for thread-count bitwise determinism:
///
/// - When `reactance > rounding(reactance_scale)`: dissipative ringing holds iff
///   `dissipation ≥ reactance` (`σ ≥ α`).
/// - When `reactance ≤ rounding(reactance_scale)`: dissipative iff
///   `dissipation > rounding(dissipation_scale)` (a lossless algebraic nullspace where both
///   `reactance` and `dissipation` vanish is deferred to the exact saddle-point check
///   [`has_voltage_source_loop_ringing`]).
fn is_dissipative_response(
    system: Linearized<'_>,
    alpha: f64,
    domains: &[(Domain, f64)],
    domain: Domain,
    response: &[f64],
) -> bool {
    let mut dissipation = 0.0;
    let mut dissipation_scale = 0.0;
    let mut reactance = 0.0;
    let mut reactance_scale = 0.0;

    for (row, &(row_domain, sign)) in domains.iter().enumerate() {
        if row_domain != domain {
            continue;
        }
        let unknown = response[row];
        let unknown_magnitude = unknown.abs();
        let mut conductance_dot = 0.0;
        let mut conductance_magnitude = 0.0;
        let mut capacitance_dot = 0.0;
        let mut capacitance_magnitude = 0.0;
        for &(slot, column) in system.pattern.row(row) {
            let entry_unknown = response[column];
            let entry_unknown_magnitude = entry_unknown.abs();
            let entry_conductance = system.conductance[slot];
            let entry_capacitance = system.capacitance[slot];
            conductance_dot += entry_conductance * entry_unknown;
            conductance_magnitude += entry_conductance.abs() * entry_unknown_magnitude;
            capacitance_dot += entry_capacitance * entry_unknown;
            capacitance_magnitude += entry_capacitance.abs() * entry_unknown_magnitude;
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

/// Identifies all ideal voltage-domain branches (`Nature::Current`, zero capacitance row and
/// column, and conductance entries touching only `Nature::Voltage` unknowns).
fn find_ideal_sources(system: Linearized<'_>) -> Vec<IdealSource> {
    let count = system.count();
    let mut has_capacitance = vec![false; count];
    let mut has_non_voltage_conductance = vec![false; count];
    let mut terminals: Vec<Vec<(usize, f64)>> = vec![Vec::new(); count];

    for (slot, (row, column)) in system.pattern.positions().enumerate() {
        if system.capacitance[slot] != 0.0 {
            has_capacitance[row] = true;
            has_capacitance[column] = true;
        }
        let entry_conductance = system.conductance[slot];
        if entry_conductance == 0.0 {
            continue;
        }
        if system.natures[column] == Nature::Current {
            if system.natures[row] == Nature::Voltage {
                terminals[column].push((row, entry_conductance));
            } else {
                has_non_voltage_conductance[column] = true;
            }
        }
        if system.natures[row] == Nature::Current && system.natures[column] != Nature::Voltage {
            has_non_voltage_conductance[row] = true;
        }
    }

    terminals
        .into_iter()
        .enumerate()
        .filter_map(|(unknown, source_terminals)| {
            (system.natures[unknown] == Nature::Current
                && !has_capacitance[unknown]
                && !has_non_voltage_conductance[unknown]
                && !source_terminals.is_empty())
            .then_some(IdealSource {
                unknown,
                terminals: source_terminals,
            })
        })
        .collect()
}

/// Exact saddle-point projection isolating algebraic `R = −1` ringing inside
/// capacitor-voltage-source loops (both grounded and floating).
///
/// On each connected component `S` of `Nature::Voltage` unknowns joined by capacitance edges
/// and incident ideal voltage sources `V_S`, solves:
///
/// ```text
/// [ C_SS / c_scale   B_S ] [ z ]   [ v_S ]
/// [     A_S           0  ] [ μ ] = [  0  ]
/// ```
///
/// where `B_S` holds the KCL branch stamps `G[u, j]` and `A_S` holds the KVL linearized
/// voltage constraints `G[j, u]` restricted to `S`. Because `C_SS` is scaled without changing
/// the Lagrange multiplier `μ`, every lossless LC mode (single- or multi-mode, bypass
/// capacitors to DC supplies, ungrounded capacitive islands, and controlled voltage sources
/// whose control terminals lie in `S`) produces `μ = 0` to machine precision, while any
/// unlocated corner or algebraic `(-1)^k` ringing on a source in `V_S` yields exact
/// alternating `μ_y` against smooth `μ_q`.
fn has_voltage_source_loop_ringing(
    system: Linearized<'_>,
    ideal_sources: &[IdealSource],
    active: &[usize],
    run: &Run<'_>,
    options: &Options,
) -> bool {
    let count = system.count();
    let mut is_active = vec![false; count];
    for &index in active {
        is_active[index] = true;
    }

    let adjacency = voltage_capacitive_adjacency(system, ideal_sources);
    let mut visited = vec![false; count];
    let mut position_in_component = vec![ABSENT; count];
    let mut queue = VecDeque::new();

    for source in ideal_sources {
        let Some(&(start, _)) = source.terminals.first() else {
            continue;
        };
        if visited[start] {
            continue;
        }
        let mut component = Vec::new();
        visited[start] = true;
        queue.push_back(start);
        while let Some(node) = queue.pop_front() {
            component.push(node);
            for &neighbor in &adjacency[node] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        if !component.iter().any(|&node| is_active[node]) {
            continue;
        }
        component.sort_unstable();
        for (local_index, &node) in component.iter().enumerate() {
            position_in_component[node] = local_index;
        }

        let sources_in_component: Vec<&IdealSource> = ideal_sources
            .iter()
            .filter(|candidate| {
                candidate
                    .terminals
                    .iter()
                    .all(|&(node, _)| position_in_component[node] != ABSENT)
            })
            .collect();

        let rings_here = check_component_saddle(
            system,
            &component,
            &position_in_component,
            &sources_in_component,
            run,
            options,
        );

        for &node in &component {
            position_in_component[node] = ABSENT;
        }
        if rings_here {
            return true;
        }
    }

    false
}

/// Builds symmetric adjacency among `Nature::Voltage` unknowns from capacitance nonzeros and
/// incident ideal voltage source terminals.
fn voltage_capacitive_adjacency(
    system: Linearized<'_>,
    ideal_sources: &[IdealSource],
) -> Vec<Vec<usize>> {
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); system.count()];
    for (slot, (row, column)) in system.pattern.positions().enumerate() {
        if row != column
            && system.capacitance[slot] != 0.0
            && system.natures[row] == Nature::Voltage
            && system.natures[column] == Nature::Voltage
        {
            adjacency[row].push(column);
            adjacency[column].push(row);
        }
    }
    for source in ideal_sources {
        let Some(&(anchor, _)) = source.terminals.first() else {
            continue;
        };
        for &(other, _) in &source.terminals[1..] {
            adjacency[anchor].push(other);
            adjacency[other].push(anchor);
        }
    }
    adjacency
}

/// Builds the scaled sparse saddle-point matrix `K_S` for `component` and its incident
/// `sources_in_component`, returning `None` if `component` holds no nonzero capacitance.
fn build_saddle_system(
    system: Linearized<'_>,
    component: &[usize],
    position_in_component: &[usize],
    sources_in_component: &[&IdealSource],
) -> Option<SaddleSystem> {
    let node_count = component.len();
    let source_count = sources_in_component.len();
    if node_count == 0 || source_count == 0 {
        return None;
    }

    let (capacitance_scale, is_island) = capacitance_scale_and_island(
        system,
        component,
        position_in_component,
        sources_in_component,
    );
    if capacitance_scale == 0.0 {
        return None;
    }

    let dimension = node_count + source_count + usize::from(is_island);
    let mut builder = PatternBuilder::new();
    let mut entries: Vec<(Entry, f64)> = Vec::new();

    for (local_row, &node) in component.iter().enumerate() {
        for &(slot, column) in system.pattern.row(node) {
            let local_column = position_in_component[column];
            if local_column == ABSENT {
                continue;
            }
            let entry = system.capacitance[slot];
            if entry != 0.0 {
                entries.push((
                    builder.entry(local_row, local_column),
                    entry / capacitance_scale,
                ));
            }
        }
    }
    for (source_index, source) in sources_in_component.iter().enumerate() {
        let branch_index = node_count + source_index;
        for &(node, stamp) in &source.terminals {
            let local_row = position_in_component[node];
            entries.push((builder.entry(local_row, branch_index), stamp));
        }
        for &(slot, column) in system.pattern.row(source.unknown) {
            let local_column = position_in_component[column];
            if local_column == ABSENT {
                continue;
            }
            let stamp = system.conductance[slot];
            if stamp != 0.0 {
                entries.push((builder.entry(branch_index, local_column), stamp));
            }
        }
    }
    if is_island {
        let pin_index = node_count + source_count;
        for local_index in 0..node_count {
            entries.push((builder.entry(local_index, pin_index), 1.0));
            entries.push((builder.entry(pin_index, local_index), 1.0));
        }
    }

    let pattern = Arc::new(builder.finish(dimension));
    let mut values = vec![0.0; pattern.len()];
    for (entry, value) in entries {
        values[pattern.slot(entry)] += value;
    }

    Some(SaddleSystem {
        pattern,
        values,
        dimension,
    })
}

/// Computes the largest capacitance magnitude in `component` (used to scale `C_SS` to O(1)
/// without changing `μ`) and whether `component` is an ungrounded capacitive island (`1` in
/// both left and right nullspaces of `C_SS`, `B_S`, and `A_S`).
fn capacitance_scale_and_island(
    system: Linearized<'_>,
    component: &[usize],
    position_in_component: &[usize],
    sources_in_component: &[&IdealSource],
) -> (f64, bool) {
    let node_count = component.len();
    let mut capacitance_scale = 0.0_f64;
    let mut row_sums = vec![0.0; node_count];
    let mut row_magnitude_sums = vec![0.0; node_count];
    let mut column_sums = vec![0.0; node_count];
    let mut column_magnitude_sums = vec![0.0; node_count];

    for (local_row, &node) in component.iter().enumerate() {
        for &(slot, column) in system.pattern.row(node) {
            let local_column = position_in_component[column];
            if local_column == ABSENT {
                continue;
            }
            let entry = system.capacitance[slot];
            if entry == 0.0 {
                continue;
            }
            let magnitude = entry.abs();
            capacitance_scale = capacitance_scale.max(magnitude);
            row_sums[local_row] += entry;
            row_magnitude_sums[local_row] += magnitude;
            column_sums[local_column] += entry;
            column_magnitude_sums[local_column] += magnitude;
        }
    }
    if capacitance_scale == 0.0 {
        return (0.0, false);
    }

    let capacitance_rows_zero = row_sums
        .iter()
        .zip(&row_magnitude_sums)
        .all(|(&sum, &magnitude_sum)| sum.abs() <= rounding(magnitude_sum));
    let capacitance_columns_zero = column_sums
        .iter()
        .zip(&column_magnitude_sums)
        .all(|(&sum, &magnitude_sum)| sum.abs() <= rounding(magnitude_sum));
    let sources_balanced = sources_in_component.iter().all(|source| {
        let mut branch_sum = 0.0;
        let mut branch_magnitude = 0.0;
        for &(_, stamp) in &source.terminals {
            branch_sum += stamp;
            branch_magnitude += stamp.abs();
        }
        let mut constraint_sum = 0.0;
        let mut constraint_magnitude = 0.0;
        for &(slot, column) in system.pattern.row(source.unknown) {
            if position_in_component[column] != ABSENT {
                let stamp = system.conductance[slot];
                constraint_sum += stamp;
                constraint_magnitude += stamp.abs();
            }
        }
        branch_sum.abs() <= rounding(branch_magnitude)
            && constraint_sum.abs() <= rounding(constraint_magnitude)
    });

    (
        capacitance_scale,
        capacitance_rows_zero && capacitance_columns_zero && sources_balanced,
    )
}

/// Maximum componentwise residual rounding floor `8 ε · (|K_S| |solution|)` over the
/// capacitive rows `0..node_count` (Higham, 2002, §12.3), in amperes, automatically
/// accounting for any ill-conditioning of `C_SS`.
fn saddle_rounding_floor(saddle: &SaddleSystem, node_count: usize, solution: &[f64]) -> f64 {
    let mut floor = 0.0_f64;
    for row_index in 0..node_count {
        let resolution = saddle
            .pattern
            .row(row_index)
            .iter()
            .fold(0.0, |total, &(slot, column)| {
                total + saddle.values[slot].abs() * solution[column].abs()
            });
        floor = floor.max(rounding(resolution));
    }
    floor
}

/// Factors and solves the saddle-point system for one capacitive connected component.
fn check_component_saddle(
    system: Linearized<'_>,
    component: &[usize],
    position_in_component: &[usize],
    sources_in_component: &[&IdealSource],
    run: &Run<'_>,
    options: &Options,
) -> bool {
    let Some(saddle) = build_saddle_system(
        system,
        component,
        position_in_component,
        sources_in_component,
    ) else {
        return false;
    };
    let Ok(ordering) = Ordering::new(Arc::clone(&saddle.pattern)) else {
        return false;
    };
    let mut saddle_solver = SparseLu::new(Arc::new(ordering));
    let Ok(mut factors) = saddle_solver.factor(&saddle.values) else {
        return false;
    };

    let node_count = component.len();
    let source_count = sources_in_component.len();
    let mut loop_derivatives = vec![[0.0; 4]; source_count];
    let mut loop_charges = vec![[0.0; 4]; source_count];
    let mut rounding_floor = 0.0_f64;
    let mut right_hand_side = vec![0.0; saddle.dimension];

    for (step, (&derivative_slice, &charge_slice)) in
        run.derivatives.iter().zip(&run.charges).enumerate()
    {
        right_hand_side.fill(0.0);
        for (local_row, &node) in component.iter().enumerate() {
            right_hand_side[local_row] = derivative_slice[node];
        }
        if factors.solve(&mut right_hand_side).is_err() {
            return false;
        }
        rounding_floor =
            rounding_floor.max(saddle_rounding_floor(&saddle, node_count, &right_hand_side));
        for (source_index, series) in loop_derivatives.iter_mut().enumerate() {
            series[step] = right_hand_side[node_count + source_index];
        }

        right_hand_side.fill(0.0);
        for (local_row, &node) in component.iter().enumerate() {
            right_hand_side[local_row] = charge_slice[node];
        }
        if factors.solve(&mut right_hand_side).is_err() {
            return false;
        }
        for (source_index, series) in loop_charges.iter_mut().enumerate() {
            series[step] = right_hand_side[node_count + source_index];
        }
    }

    let absolute = options.equation_tolerance(Nature::Voltage);
    for (&derivatives, &charges) in loop_derivatives.iter().zip(&loop_charges) {
        let largest = derivatives
            .iter()
            .fold(0.0_f64, |most, value| most.max(value.abs()));
        let tolerance = (options.truncation_tolerance
            * (options.relative_tolerance * largest + absolute))
            .max(rounding_floor);
        if turns_alternate(derivatives, tolerance)
            && !is_harmonic_scalar(run.times, charges, derivatives)
        {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;
    use std::sync::Arc;

    use jolty_models::Nature;
    use jolty_numerics::{Ordering, Pattern, PatternBuilder};

    use super::{
        Domain, Linearized, Run, classify_domains, domain_is_skew_lossless, find_ideal_sources,
        has_voltage_source_loop_ringing, is_harmonic_scalar, rings, turns_alternate,
    };
    use crate::Options;
    use crate::newton::Workspace;
    use crate::solver::Solver;

    /// Sparse matrices built from dense test arrays.
    struct LinearTestSystem {
        pattern: Arc<Pattern>,
        conductance: Vec<f64>,
        capacitance: Vec<f64>,
    }

    impl LinearTestSystem {
        /// Builds a sparse `Pattern` and value arrays for dense `conductance` and `capacitance`.
        fn new(conductance: &[Vec<f64>], capacitance: &[Vec<f64>]) -> Self {
            let dimension = conductance.len();
            let mut builder = PatternBuilder::new();
            let mut conductance_entries = Vec::new();
            let mut capacitance_entries = Vec::new();
            for (row, (conductance_row, capacitance_row)) in
                conductance.iter().zip(capacitance).enumerate()
            {
                for (column, (&entry_conductance, &entry_capacitance)) in
                    conductance_row.iter().zip(capacitance_row).enumerate()
                {
                    if entry_conductance != 0.0 || entry_capacitance != 0.0 || row == column {
                        let entry = builder.entry(row, column);
                        conductance_entries.push((entry, entry_conductance));
                        capacitance_entries.push((entry, entry_capacitance));
                    }
                }
            }
            let pattern = Arc::new(builder.finish(dimension));
            let mut conductance_values = vec![0.0; pattern.len()];
            let mut capacitance_values = vec![0.0; pattern.len()];
            for (entry, value) in conductance_entries {
                conductance_values[pattern.slot(entry)] += value;
            }
            for (entry, value) in capacitance_entries {
                capacitance_values[pattern.slot(entry)] += value;
            }
            Self {
                pattern,
                conductance: conductance_values,
                capacitance: capacitance_values,
            }
        }

        /// Builds `G + alpha * C` in storage order.
        fn jacobian(&self, alpha: f64) -> Vec<f64> {
            self.conductance
                .iter()
                .zip(&self.capacitance)
                .map(|(&entry_conductance, &entry_capacitance)| {
                    entry_conductance + alpha * entry_capacitance
                })
                .collect()
        }

        /// Creates a solver sharing this system's sparsity pattern.
        fn solver(&self) -> Solver<f64> {
            let dimension = self.pattern.dimension();
            let ordering = Arc::new(Ordering::new(Arc::clone(&self.pattern)).unwrap());
            let names: Arc<[String]> = (0..dimension).map(|index| format!("u{index}")).collect();
            Solver::new(ordering, names)
        }
    }

    /// Evaluates [`rings`] directly on synthetic linear matrices `(G, C)` over four snapshots.
    fn evaluate_rings(
        (conductance, capacitance): (&[Vec<f64>], &[Vec<f64>]),
        natures: &[Nature],
        (times, charges, derivatives): ([f64; 4], [&[f64]; 4], [&[f64]; 4]),
        options: &Options,
    ) -> bool {
        let matrices = LinearTestSystem::new(conductance, capacitance);
        let mut solver = matrices.solver();
        let alpha = 2.0 / (times[3] - times[2]);
        let jacobian = matrices.jacobian(alpha);
        let mut workspace = Workspace::from_matrices(
            &matrices.pattern,
            matrices.conductance,
            matrices.capacitance,
            jacobian,
        );
        let run = Run {
            times,
            charges,
            derivatives,
            alpha,
        };
        rings(
            &matrices.pattern,
            natures,
            &mut solver,
            &mut workspace,
            &run,
            options,
        )
    }

    /// Advances one fixed-step linear circuit step (`G x + C x' = source_values`).
    fn advance_linear_step(
        matrices: &LinearTestSystem,
        solver: &mut Solver<f64>,
        jacobian: &[f64],
        (alpha, use_backward_euler): (f64, bool),
        (charges, derivatives, source_values): (&[f64], &[f64], &[f64]),
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let dimension = charges.len();
        let mut solution: Vec<f64> = if use_backward_euler {
            (0..dimension)
                .map(|index| alpha * charges[index] + source_values[index])
                .collect()
        } else {
            (0..dimension)
                .map(|index| alpha * charges[index] + derivatives[index] + source_values[index])
                .collect()
        };
        solver
            .factor(jacobian)
            .unwrap()
            .solve(&mut solution)
            .unwrap();
        let mut next_charges = vec![0.0; dimension];
        matrices
            .pattern
            .add_product(&matrices.capacitance, &solution, &mut next_charges);
        let next_derivatives: Vec<f64> = if use_backward_euler {
            (0..dimension)
                .map(|index| alpha * (next_charges[index] - charges[index]))
                .collect()
        } else {
            (0..dimension)
                .map(|index| alpha * (next_charges[index] - charges[index]) - derivatives[index])
                .collect()
        };
        (solution, next_charges, next_derivatives)
    }

    /// Computes initial charges and charge derivatives `q'(0) = source(0) − G x(0)` (or zero
    /// derivatives for backward-Euler startup).
    fn initial_state_derivatives(
        matrices: &LinearTestSystem,
        initial_state: &[f64],
        backward_euler_startup: bool,
        initial_source: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let dimension = initial_state.len();
        let mut charges = vec![0.0; dimension];
        matrices
            .pattern
            .add_product(&matrices.capacitance, initial_state, &mut charges);
        let derivatives = if backward_euler_startup {
            vec![0.0; dimension]
        } else {
            let mut resistive = vec![0.0; dimension];
            matrices
                .pattern
                .add_product(&matrices.conductance, initial_state, &mut resistive);
            (0..dimension)
                .map(|index| initial_source[index] - resistive[index])
                .collect()
        };
        (charges, derivatives)
    }

    /// Rolling four-point history tracking accepted times, charges, and consecutive
    /// trapezoidal derivatives.
    struct RollingHistory {
        times: Vec<f64>,
        charges: Vec<Vec<f64>>,
        ringing_derivatives: Vec<Vec<f64>>,
    }

    impl RollingHistory {
        fn new(initial_charges: Vec<f64>) -> Self {
            Self {
                times: vec![0.0],
                charges: vec![initial_charges],
                ringing_derivatives: Vec::new(),
            }
        }

        fn build_run<'a>(
            &'a self,
            time: f64,
            next_charges: &'a [f64],
            derivatives: &'a [f64],
            next_derivatives: &'a [f64],
            alpha: f64,
        ) -> Option<Run<'a>> {
            if self.ringing_derivatives.len() < 2 || self.times.len() < 3 {
                return None;
            }
            let count = self.times.len();
            Some(Run {
                times: [
                    self.times[count - 3],
                    self.times[count - 2],
                    self.times[count - 1],
                    time,
                ],
                charges: [
                    &self.charges[count - 3],
                    &self.charges[count - 2],
                    &self.charges[count - 1],
                    next_charges,
                ],
                derivatives: [
                    &self.ringing_derivatives[0],
                    &self.ringing_derivatives[1],
                    derivatives,
                    next_derivatives,
                ],
                alpha,
            })
        }

        fn push(
            &mut self,
            time: f64,
            next_charges: Vec<f64>,
            previous_derivatives: Vec<f64>,
            use_backward_euler: bool,
        ) {
            if use_backward_euler {
                self.ringing_derivatives.clear();
            } else {
                if self.ringing_derivatives.len() == 2 {
                    self.ringing_derivatives.remove(0);
                }
                self.ringing_derivatives.push(previous_derivatives);
            }
            self.times.push(time);
            self.charges.push(next_charges);
            if self.times.len() > 3 {
                self.times.remove(0);
                self.charges.remove(0);
            }
        }
    }

    /// Runs a fixed-step trapezoidal simulation on linear `G x + C x' = source(t)` with the
    /// ringing detector active (taking one backward Euler step whenever it fires) and returns
    /// `(damp_count, solver_reuses_count)`.
    fn simulate_linear(
        conductance: &[Vec<f64>],
        capacitance: &[Vec<f64>],
        natures: &[Nature],
        initial_state: &[f64],
        (step_size, steps, backward_euler_startup): (f64, usize, bool),
        mut source: impl FnMut(f64) -> Vec<f64>,
    ) -> (usize, usize) {
        let matrices = LinearTestSystem::new(conductance, capacitance);
        let mut step_solver = matrices.solver();
        let mut ring_solver = matrices.solver();
        let options = Options::default();
        let initial_source = source(0.0);
        let (mut charges, mut derivatives) = initial_state_derivatives(
            &matrices,
            initial_state,
            backward_euler_startup,
            &initial_source,
        );
        let (alpha_trapezoidal, alpha_backward_euler) = (2.0 / step_size, 1.0 / step_size);
        let jacobian_trapezoidal = matrices.jacobian(alpha_trapezoidal);
        let jacobian_backward_euler = matrices.jacobian(alpha_backward_euler);

        // Pre-seed ring_solver cache with trapezoidal Jacobian, as `filter()` does.
        ring_solver.factor(&jacobian_trapezoidal).unwrap();
        let reuses_before = ring_solver.statistics().reuses;

        let mut workspace = Workspace::from_matrices(
            &matrices.pattern,
            matrices.conductance.clone(),
            matrices.capacitance.clone(),
            jacobian_trapezoidal.clone(),
        );
        let mut history = RollingHistory::new(charges.clone());
        let mut damp = false;
        let mut damps = 0;

        for step in 1..=steps {
            let time = f64::from(u32::try_from(step).unwrap()) * step_size;
            let use_backward_euler = damp || (backward_euler_startup && step < 3);
            let (alpha, jacobian) = if use_backward_euler {
                (alpha_backward_euler, &jacobian_backward_euler)
            } else {
                (alpha_trapezoidal, &jacobian_trapezoidal)
            };
            let source_values = source(time);
            let (_, next_charges, next_derivatives) = advance_linear_step(
                &matrices,
                &mut step_solver,
                jacobian,
                (alpha, use_backward_euler),
                (&charges, &derivatives, &source_values),
            );
            let next_damp = if use_backward_euler {
                false
            } else if let Some(run) =
                history.build_run(time, &next_charges, &derivatives, &next_derivatives, alpha)
            {
                rings(
                    &matrices.pattern,
                    natures,
                    &mut ring_solver,
                    &mut workspace,
                    &run,
                    &options,
                )
            } else {
                false
            };
            damps += usize::from(next_damp);
            damp = next_damp;
            history.push(time, next_charges.clone(), derivatives, use_backward_euler);
            charges = next_charges;
            derivatives = next_derivatives;
        }
        (damps, ring_solver.statistics().reuses - reuses_before)
    }

    #[test]
    fn scalar_harmonic_identity_passes_for_arbitrary_variable_steps() {
        let omega = 1.0e7;
        for &base_step in &[2.0e-8, 2.5e-7, 5.0e-7, 2.0e-6] {
            let steps = [base_step, 1.7 * base_step, 0.43 * base_step];
            let mut times = [0.0; 4];
            let mut charges = [0.0; 4];
            let mut derivatives = [0.0; 4];
            charges[0] = 1.0e-9;
            derivatives[0] = 0.003;
            for step in 0..3 {
                let length = steps[step];
                times[step + 1] = times[step] + length;
                let alpha = 2.0 / length;
                let denominator = alpha * alpha + omega * omega;
                let numerator = alpha * alpha - omega * omega;
                charges[step + 1] =
                    (numerator * charges[step] + 2.0 * alpha * derivatives[step]) / denominator;
                derivatives[step + 1] = (numerator * derivatives[step]
                    - 2.0 * alpha * omega * omega * charges[step])
                    / denominator;
            }
            assert!(
                is_harmonic_scalar(times, charges, derivatives),
                "base_step={base_step}, omega*h={}",
                omega * base_step
            );
        }
    }

    #[test]
    fn scalar_harmonic_identity_rejects_stiff_real_decay_on_variable_steps() {
        let lambda = 1.0e7;
        let steps = [5.0e-7, 3.0e-7, 8.0e-7];
        let mut times = [0.0; 4];
        let mut charges = [0.0; 4];
        let mut derivatives = [0.0; 4];
        charges[0] = 1.0e-9;
        derivatives[0] = -lambda * charges[0];
        for step in 0..3 {
            let length = steps[step];
            times[step + 1] = times[step] + length;
            let factor = (2.0 - lambda * length) / (2.0 + lambda * length);
            charges[step + 1] = factor * charges[step];
            derivatives[step + 1] = -lambda * charges[step + 1];
        }
        assert!(turns_alternate(derivatives, 1e-12));
        assert!(!is_harmonic_scalar(times, charges, derivatives));
    }

    #[test]
    fn scalar_harmonic_identity_rejects_flat_and_monotone_ringing() {
        let times = [0.0, 1e-5, 2e-5, 3e-5];
        let flat_charges = [1e-12, 1e-12, 1e-12, 1e-12];
        let ringing_derivatives = [1e-3, -1e-3, 1e-3, -1e-3];
        assert!(!is_harmonic_scalar(
            times,
            flat_charges,
            ringing_derivatives
        ));

        let slope = 6e-3;
        let ramp_charges = [0.0, slope * 1e-5, slope * 2e-5, slope * 3e-5];
        let noisy_derivatives = [slope + 0.01, slope - 0.01, slope + 0.01, slope - 0.01];
        assert!(!is_harmonic_scalar(times, ramp_charges, noisy_derivatives));
    }

    #[test]
    fn classifies_excess_phase_filter_separately_from_electrical_inductors() {
        let conductance = vec![
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let capacitance = vec![
            vec![1e-9, 0.0, 0.0],
            vec![0.0, -1e-6, 0.0],
            vec![0.0, 0.0, 1e-11],
        ];
        let natures = [Nature::Voltage, Nature::Current, Nature::Current];
        let matrices = LinearTestSystem::new(&conductance, &capacitance);
        let system = Linearized {
            pattern: &matrices.pattern,
            conductance: &matrices.conductance,
            capacitance: &matrices.capacitance,
            natures: &natures,
        };
        let domains = classify_domains(system);
        assert_eq!(domains[0], (Domain::Electrical, 1.0));
        assert_eq!(domains[1], (Domain::Electrical, -1.0));
        assert_eq!(domains[2], (Domain::Filter, 1.0));
        assert!(domain_is_skew_lossless(
            system,
            &domains,
            Domain::Electrical
        ));
        assert!(!domain_is_skew_lossless(system, &domains, Domain::Filter));
    }

    #[test]
    fn lossless_lc_ladder_with_dc_bypass_and_floating_zero_volt_source_never_damps() {
        let (bypass_capacitance, floating_capacitance, inductance_1, inductance_2) =
            (1e-6, 0.8e-6, 1e-5, 2e-5);
        let conductance = vec![
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let capacitance = vec![
            vec![
                bypass_capacitance,
                -bypass_capacitance,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            vec![
                -bypass_capacitance,
                bypass_capacitance + floating_capacitance,
                -floating_capacitance,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            vec![
                0.0,
                -floating_capacitance,
                floating_capacitance,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            vec![0.0, 0.0, 0.0, -inductance_1, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, -inductance_2, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let natures = [
            Nature::Voltage,
            Nature::Voltage,
            Nature::Voltage,
            Nature::Current,
            Nature::Current,
            Nature::Current,
            Nature::Current,
        ];
        let (damps_clean, reuses_clean) = simulate_linear(
            &conductance,
            &capacitance,
            &natures,
            &[5.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            (10e-6, 120, false),
            |_| vec![0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0],
        );
        assert_eq!(damps_clean, 0);
        assert_eq!(reuses_clean, 0);

        let omega = TAU * 1e3;
        let phase = 0.5;
        let (damps_ringing, _) = simulate_linear(
            &conductance,
            &capacitance,
            &natures,
            &[5.0, 0.5, 0.5 + phase.sin().abs(), 0.0, 0.0, 0.0, 0.0],
            (10e-6, 200, true),
            |time| {
                vec![
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    5.0,
                    (omega * time + phase).sin().abs(),
                ]
            },
        );
        assert!(
            damps_ringing >= 4,
            "expected damping on |sin| corners, got {damps_ringing}"
        );
    }

    #[test]
    fn ungrounded_capacitive_island_distinguishes_zero_volt_source_from_corner() {
        let conductance = vec![
            vec![0.0, 0.0, 0.0, 1.0, 0.0, -1.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![-1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let capacitance = vec![
            vec![1e-6, -1e-6, 0.0, 0.0, 0.0, 0.0],
            vec![-1e-6, 1.5e-6, -0.5e-6, 0.0, 0.0, 0.0],
            vec![0.0, -0.5e-6, 0.5e-6, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, -1e-5, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, -2e-5, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let natures = [
            Nature::Voltage,
            Nature::Voltage,
            Nature::Voltage,
            Nature::Current,
            Nature::Current,
            Nature::Current,
        ];
        let (damps_clean, _) = simulate_linear(
            &conductance,
            &capacitance,
            &natures,
            &[0.5, 0.5, -0.2, 0.0, 0.0, 0.0],
            (10e-6, 120, false),
            |_| vec![0.0; 6],
        );
        assert_eq!(damps_clean, 0);

        let omega = TAU * 1e3;
        let phase = 0.5;
        let (damps_ringing, _) = simulate_linear(
            &conductance,
            &capacitance,
            &natures,
            &[0.5, 0.5 + phase.sin().abs(), -0.2, 0.0, 0.0, 0.0],
            (10e-6, 200, true),
            |time| vec![0.0, 0.0, 0.0, 0.0, 0.0, (omega * time + phase).sin().abs()],
        );
        assert!(damps_ringing >= 4, "got {damps_ringing}");
    }

    #[test]
    fn parallel_rlc_separates_underdamped_from_critically_and_overdamped() {
        let capacitance = 1e-9;
        let inductance = 1e-6;
        let omega = 1.0 / (inductance * capacitance).sqrt();
        let impedance = (inductance / capacitance).sqrt();
        let capacitance_matrix = vec![vec![capacitance, 0.0], vec![0.0, -inductance]];
        let natures = [Nature::Voltage, Nature::Current];

        for &damping_ratio in &[0.0, 0.01] {
            let diagonal = if damping_ratio == 0.0 {
                0.0
            } else {
                2.0 * damping_ratio / impedance
            };
            let conductance_matrix = vec![vec![diagonal, 1.0], vec![1.0, 0.0]];
            let (damps, _) = simulate_linear(
                &conductance_matrix,
                &capacitance_matrix,
                &natures,
                &[1.0, 0.0],
                (5.0 / omega, 40, false),
                |_| vec![0.0, 0.0],
            );
            assert_eq!(damps, 0, "unexpected damping at zeta={damping_ratio}");
        }

        for &damping_ratio in &[1.0, 1.5, 3.0] {
            let conductance_matrix =
                vec![vec![2.0 * damping_ratio / impedance, 1.0], vec![1.0, 0.0]];
            let (damps, _) = simulate_linear(
                &conductance_matrix,
                &capacitance_matrix,
                &natures,
                &[1.0, 0.0],
                (5.0 / omega, 40, false),
                |_| vec![0.0, 0.0],
            );
            assert!(
                damps >= 1,
                "missed stiff RLC ringing at zeta={damping_ratio}"
            );
        }
    }

    #[test]
    fn mixed_lossless_two_mode_lc_and_small_stiff_rc_catches_rc_ringing() {
        let conductance = vec![
            vec![0.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, -1.0, 1.0, 0.0],
            vec![1.0, -1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.01],
        ];
        let capacitance = vec![
            vec![1e-6, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 2e-6, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, -1e-5, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, -2e-5, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 1e-9],
        ];
        let natures = [
            Nature::Voltage,
            Nature::Voltage,
            Nature::Current,
            Nature::Current,
            Nature::Voltage,
        ];
        let (damps, _) = simulate_linear(
            &conductance,
            &capacitance,
            &natures,
            &[1.0, -1.0, 0.0, 0.0, 1e-3],
            (3e-6, 40, false),
            |_| vec![0.0; 5],
        );
        assert!(
            damps >= 1,
            "stiff RC hidden by large LC ladder must be caught"
        );
    }

    #[test]
    fn mixed_lossless_lc_with_meyer_displacement_catches_ringing() {
        let conductance = vec![
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let capacitance = vec![
            vec![1e-6, 0.0, 0.0],
            vec![0.0, -1e-5, 0.0],
            vec![0.0, 0.0, 0.0],
        ];
        let natures = [Nature::Voltage, Nature::Current, Nature::Displacement];
        let times = [0.0, 1e-5, 2e-5, 3e-5];
        let charges = [1e-6, 0.0, 0.0];
        let derivative_0 = [0.0, 0.0, 1e-3];
        let derivative_1 = [0.0, 0.0, -1e-3];
        let derivative_2 = [0.0, 0.0, 1e-3];
        let derivative_3 = [0.0, 0.0, -1e-3];
        assert!(evaluate_rings(
            (&conductance, &capacitance),
            &natures,
            (
                times,
                [&charges, &charges, &charges, &charges],
                [&derivative_0, &derivative_1, &derivative_2, &derivative_3],
            ),
            &Options::default(),
        ));
    }

    #[test]
    fn thermal_and_state_stiff_modes_ring_above_stability_boundary() {
        let time_constant = 1.0e-6;
        let conductance = vec![vec![1.0]];
        let capacitance = vec![vec![time_constant]];
        let step = 10.0 * time_constant;
        let times = [0.0, step, 2.0 * step, 3.0 * step];
        let ratio = (2.0 - step / time_constant) / (2.0 + step / time_constant);
        let charge_series = [
            vec![time_constant],
            vec![time_constant * ratio],
            vec![time_constant * ratio * ratio],
            vec![time_constant * ratio * ratio * ratio],
        ];
        let derivative_series = [
            vec![-1.0],
            vec![-ratio],
            vec![-ratio * ratio],
            vec![-ratio * ratio * ratio],
        ];
        for nature in [Nature::Temperature, Nature::State] {
            assert!(evaluate_rings(
                (&conductance, &capacitance),
                &[nature],
                (
                    times,
                    [
                        &charge_series[0],
                        &charge_series[1],
                        &charge_series[2],
                        &charge_series[3],
                    ],
                    [
                        &derivative_series[0],
                        &derivative_series[1],
                        &derivative_series[2],
                        &derivative_series[3],
                    ],
                ),
                &Options::default(),
            ));
        }
    }

    /// Collects `steps` fixed-step trapezoidal charge and derivative trajectories for `matrices`.
    fn collect_trapezoidal_trajectory(
        matrices: &LinearTestSystem,
        initial_state: &[f64],
        step_size: f64,
        steps: usize,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let dimension = initial_state.len();
        let mut solver = matrices.solver();
        let alpha = 2.0 / step_size;
        let jacobian = matrices.jacobian(alpha);
        let zero_source = vec![0.0; dimension];
        let (mut charges, mut derivatives) =
            initial_state_derivatives(matrices, initial_state, false, &zero_source);
        derivatives[dimension - 1] = 0.0;
        let mut charge_history = vec![charges.clone()];
        let mut derivative_history = vec![derivatives.clone()];
        for _ in 0..steps {
            let (_, next_charges, next_derivatives) = advance_linear_step(
                matrices,
                &mut solver,
                &jacobian,
                (alpha, false),
                (&charges, &derivatives, &zero_source),
            );
            charge_history.push(next_charges.clone());
            derivative_history.push(next_derivatives.clone());
            charges = next_charges;
            derivatives = next_derivatives;
        }
        (charge_history, derivative_history)
    }

    #[test]
    fn vcvs_inside_two_mode_lc_has_zero_loop_multiplier() {
        let (inductance_1, inductance_2) = (1e-5, 2e-5);
        let (capacitance_1, capacitance_2, coupling_capacitance) = (1e-6, 0.7e-6, 0.5e-6);
        let conductance = vec![
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, -2.0, 1.0, 0.0, 0.0, 0.0],
        ];
        let capacitance = vec![
            vec![
                capacitance_1 + coupling_capacitance,
                -coupling_capacitance,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            vec![
                -coupling_capacitance,
                capacitance_2 + 2.0 * coupling_capacitance,
                -coupling_capacitance,
                0.0,
                0.0,
                0.0,
            ],
            vec![
                0.0,
                -coupling_capacitance,
                coupling_capacitance,
                0.0,
                0.0,
                0.0,
            ],
            vec![0.0, 0.0, 0.0, -inductance_1, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, -inductance_2, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let natures = [
            Nature::Voltage,
            Nature::Voltage,
            Nature::Voltage,
            Nature::Current,
            Nature::Current,
            Nature::Current,
        ];
        let step_size = 10e-6;
        let matrices = LinearTestSystem::new(&conductance, &capacitance);
        let (charge_history, derivative_history) = collect_trapezoidal_trajectory(
            &matrices,
            &[1.0, -0.5, -1.0, 0.0, 0.0, 0.0],
            step_size,
            20,
        );
        let system = Linearized {
            pattern: &matrices.pattern,
            conductance: &matrices.conductance,
            capacitance: &matrices.capacitance,
            natures: &natures,
        };
        let ideal = find_ideal_sources(system);
        assert_eq!(ideal.len(), 1);
        assert_eq!(ideal[0].unknown, 5);
        let options = Options::default();
        let alpha = 2.0 / step_size;
        for step in 3..derivative_history.len() {
            let run = Run {
                times: [0.0, step_size, 2.0 * step_size, 3.0 * step_size],
                charges: [
                    &charge_history[step - 3],
                    &charge_history[step - 2],
                    &charge_history[step - 1],
                    &charge_history[step],
                ],
                derivatives: [
                    &derivative_history[step - 3],
                    &derivative_history[step - 2],
                    &derivative_history[step - 1],
                    &derivative_history[step],
                ],
                alpha,
            };
            assert!(!has_voltage_source_loop_ringing(
                system,
                &ideal,
                &[0, 1, 2],
                &run,
                &options,
            ));
        }
    }

    #[test]
    fn grounded_corner_source_coupled_through_capacitor_to_large_lc_tank_is_caught() {
        // A grounded source at node 0 couples via `coupling_capacitance` to a large single-mode
        // lossless LC tank at node 1 (`v = 3 V`, `omega * h = 2.17 > 2`). The tank dominates
        // node 0's charge derivative so `is_harmonic_scalar` passes on every alternating unknown,
        // yet stage 2d cancels the tank exactly and catches the 6.28 mA `|sin|` ringing while
        // producing zero false damps when the grounded source is 5 V DC.
        let (coupling_capacitance, tank_capacitance, inductance) = (1e-6, 1e-6, 10e-6);
        let conductance = vec![
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
        ];
        let capacitance = vec![
            vec![coupling_capacitance, -coupling_capacitance, 0.0, 0.0],
            vec![
                -coupling_capacitance,
                coupling_capacitance + tank_capacitance,
                0.0,
                0.0,
            ],
            vec![0.0, 0.0, -inductance, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
        ];
        let natures = [
            Nature::Voltage,
            Nature::Voltage,
            Nature::Current,
            Nature::Current,
        ];
        let (damps_dc, reuses_dc) = simulate_linear(
            &conductance,
            &capacitance,
            &natures,
            &[5.0, 3.0, 0.0, 0.0],
            (9.7e-6, 120, false),
            |_| vec![0.0, 0.0, 0.0, 5.0],
        );
        assert_eq!(damps_dc, 0);
        assert_eq!(reuses_dc, 0);

        let omega = TAU * 1e3;
        let (damps_corner, _) = simulate_linear(
            &conductance,
            &capacitance,
            &natures,
            &[0.0, 3.0, 0.0, 0.0],
            (9.7e-6, 200, true),
            |time| vec![0.0, 0.0, 0.0, (omega * time).sin().abs()],
        );
        assert!(
            damps_corner >= 4,
            "expected damping on grounded |sin| coupled to large LC tank, got {damps_corner}"
        );
    }
}
