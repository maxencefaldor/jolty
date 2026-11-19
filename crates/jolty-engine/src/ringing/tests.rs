use std::collections::VecDeque;
use std::f64::consts::TAU;
use std::sync::Arc;

use jolty_models::Nature;
use jolty_numerics::{Ordering, Pattern, PatternBuilder};

use super::{
    CachedSources, Detector, Domain, Linearized, StepWindow, classify_domains,
    domain_is_skew_lossless, is_harmonic_scalar, turns_alternate,
};
use crate::Options;
use crate::newton::Workspace;
use crate::solver::Solver;

/// Integration rule used for one linear test step.
#[derive(Clone, Copy, PartialEq, Eq)]
enum StepRule {
    Trapezoidal,
    BackwardEuler,
}

/// Dense MNA stamp builder for synthetic linear ringing tests.
struct Circuit {
    conductance: Vec<Vec<f64>>,
    capacitance: Vec<Vec<f64>>,
    natures: Vec<Nature>,
}

impl Circuit {
    fn new(natures: &[Nature]) -> Self {
        let dimension = natures.len();
        Self {
            conductance: vec![vec![0.0; dimension]; dimension],
            capacitance: vec![vec![0.0; dimension]; dimension],
            natures: natures.to_vec(),
        }
    }

    fn electrical(voltages: usize, currents: usize) -> Self {
        let mut natures = vec![Nature::Voltage; voltages];
        natures.resize(voltages + currents, Nature::Current);
        Self::new(&natures)
    }

    fn stamp_conductance(&mut self, row: usize, column: usize, value: f64) {
        self.conductance[row][column] += value;
    }

    fn stamp_capacitance(&mut self, row: usize, column: usize, value: f64) {
        self.capacitance[row][column] += value;
    }

    fn capacitor(&mut self, first: usize, second: Option<usize>, value: f64) {
        self.stamp_capacitance(first, first, value);
        if let Some(other) = second {
            self.stamp_capacitance(first, other, -value);
            self.stamp_capacitance(other, first, -value);
            self.stamp_capacitance(other, other, value);
        }
    }

    fn voltage_source(&mut self, positive: usize, negative: Option<usize>, branch: usize) {
        self.stamp_conductance(positive, branch, 1.0);
        self.stamp_conductance(branch, positive, 1.0);
        if let Some(other) = negative {
            self.stamp_conductance(other, branch, -1.0);
            self.stamp_conductance(branch, other, -1.0);
        }
    }

    fn inductor(&mut self, positive: usize, negative: Option<usize>, branch: usize, value: f64) {
        self.voltage_source(positive, negative, branch);
        self.stamp_capacitance(branch, branch, -value);
    }

    fn build(&self) -> LinearSystem {
        let dimension = self.conductance.len();
        let mut builder = PatternBuilder::new();
        let mut entries = Vec::new();
        for (row, (conductance_row, capacitance_row)) in
            self.conductance.iter().zip(&self.capacitance).enumerate()
        {
            for (column, (&conductance, &capacitance)) in
                conductance_row.iter().zip(capacitance_row).enumerate()
            {
                if conductance != 0.0 || capacitance != 0.0 || row == column {
                    entries.push((builder.entry(row, column), conductance, capacitance));
                }
            }
        }
        let pattern = Arc::new(builder.finish(dimension));
        let mut conductance = vec![0.0; pattern.len()];
        let mut capacitance = vec![0.0; pattern.len()];
        for (entry, entry_conductance, entry_capacitance) in entries {
            let slot = pattern.slot(entry);
            conductance[slot] += entry_conductance;
            capacitance[slot] += entry_capacitance;
        }
        LinearSystem {
            pattern,
            conductance,
            capacitance,
        }
    }
}

/// Sparse matrices built from a [`Circuit`].
struct LinearSystem {
    pattern: Arc<Pattern>,
    conductance: Vec<f64>,
    capacitance: Vec<f64>,
}

impl LinearSystem {
    fn jacobian(&self, alpha: f64) -> Vec<f64> {
        self.conductance
            .iter()
            .zip(&self.capacitance)
            .map(|(&conductance, &capacitance)| conductance + alpha * capacitance)
            .collect()
    }

    fn solver(&self) -> Solver<f64> {
        let dimension = self.pattern.dimension();
        let ordering = Arc::new(Ordering::new(Arc::clone(&self.pattern)).unwrap());
        let names: Arc<[String]> = (0..dimension).map(|index| format!("u{index}")).collect();
        Solver::new(ordering, names)
    }

    fn linearized<'a>(&'a self, natures: &'a [Nature]) -> Linearized<'a> {
        Linearized {
            pattern: &self.pattern,
            conductance: &self.conductance,
            capacitance: &self.capacitance,
            natures,
        }
    }
}

/// Fixed-step linear circuit integrator holding charges and derivatives across steps.
struct LinearRunner<'a> {
    matrices: &'a LinearSystem,
    solver: Solver<f64>,
    charges: Vec<f64>,
    derivatives: Vec<f64>,
}

impl<'a> LinearRunner<'a> {
    fn new(
        matrices: &'a LinearSystem,
        initial_state: &[f64],
        startup: StepRule,
        initial_source: &[f64],
    ) -> Self {
        let dimension = initial_state.len();
        let mut charges = vec![0.0; dimension];
        matrices
            .pattern
            .add_product(&matrices.capacitance, initial_state, &mut charges);
        let derivatives = match startup {
            StepRule::BackwardEuler => vec![0.0; dimension],
            StepRule::Trapezoidal => {
                let mut resistive = vec![0.0; dimension];
                matrices
                    .pattern
                    .add_product(&matrices.conductance, initial_state, &mut resistive);
                (0..dimension)
                    .map(|index| initial_source[index] - resistive[index])
                    .collect()
            }
        };
        Self {
            matrices,
            solver: matrices.solver(),
            charges,
            derivatives,
        }
    }

    fn step(
        &mut self,
        jacobian: &[f64],
        alpha: f64,
        rule: StepRule,
        source_values: &[f64],
    ) -> Vec<f64> {
        let dimension = self.charges.len();
        let mut solution: Vec<f64> = (0..dimension)
            .map(|index| {
                let history = if rule == StepRule::Trapezoidal {
                    self.derivatives[index]
                } else {
                    0.0
                };
                alpha * self.charges[index] + history + source_values[index]
            })
            .collect();
        self.solver
            .factor(jacobian)
            .unwrap()
            .solve(&mut solution)
            .unwrap();
        let mut next_charges = vec![0.0; dimension];
        self.matrices
            .pattern
            .add_product(&self.matrices.capacitance, &solution, &mut next_charges);
        let next_derivatives: Vec<f64> = (0..dimension)
            .map(|index| {
                let prior = if rule == StepRule::Trapezoidal {
                    self.derivatives[index]
                } else {
                    0.0
                };
                alpha * (next_charges[index] - self.charges[index]) - prior
            })
            .collect();
        self.charges = next_charges;
        std::mem::replace(&mut self.derivatives, next_derivatives)
    }
}

fn evaluate_rings(
    circuit: &Circuit,
    times: [f64; 4],
    charges: [&[f64]; 4],
    derivatives: [&[f64]; 4],
) -> bool {
    let matrices = circuit.build();
    let mut solver = matrices.solver();
    let alpha = 2.0 / (times[3] - times[2]);
    let jacobian = matrices.jacobian(alpha);
    let mut workspace = Workspace::from_matrices(
        &matrices.pattern,
        matrices.conductance,
        matrices.capacitance,
        jacobian,
    );
    let window = StepWindow {
        times,
        charges,
        derivatives,
        alpha,
    };
    let mut detector = Detector::default();
    detector.rings(
        &matrices.pattern,
        &circuit.natures,
        &mut solver,
        &mut workspace,
        &window,
        &Options::default(),
    )
}

fn source_vector(dimension: usize, entries: &[(usize, f64)]) -> Vec<f64> {
    let mut values = vec![0.0; dimension];
    for &(index, value) in entries {
        values[index] = value;
    }
    values
}

fn update_history(
    times: &mut VecDeque<f64>,
    charges: &mut VecDeque<Vec<f64>>,
    ringing_derivatives: &mut VecDeque<Vec<f64>>,
    time: f64,
    next_charges: Vec<f64>,
    previous_derivatives: Vec<f64>,
    rule: StepRule,
) {
    if rule == StepRule::BackwardEuler {
        ringing_derivatives.clear();
    } else {
        if ringing_derivatives.len() == 2 {
            ringing_derivatives.pop_front();
        }
        ringing_derivatives.push_back(previous_derivatives);
    }
    times.push_back(time);
    charges.push_back(next_charges);
    if times.len() > 3 {
        times.pop_front();
        charges.pop_front();
    }
}

/// Runs a fixed-step trapezoidal simulation on `G x + C x' = source(t)` with the ringing
/// detector active, returning `(damp_count, solver_reuses_count, detector)`.
fn simulate_linear(
    circuit: &Circuit,
    initial_state: &[f64],
    step_size: f64,
    steps: usize,
    startup: StepRule,
    mut source: impl FnMut(f64) -> Vec<f64>,
) -> (usize, usize, Detector) {
    let matrices = circuit.build();
    let mut ring_solver = matrices.solver();
    let options = Options::default();
    let mut runner = LinearRunner::new(&matrices, initial_state, startup, &source(0.0));
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
    let mut times = VecDeque::from([0.0]);
    let mut charges = VecDeque::from([runner.charges.clone()]);
    let mut ringing_derivatives: VecDeque<Vec<f64>> = VecDeque::new();
    let mut detector = Detector::default();
    let mut damp = false;
    let mut damps = 0;

    for step in 1..=steps {
        let time = f64::from(u32::try_from(step).unwrap()) * step_size;
        let rule = if damp || (startup == StepRule::BackwardEuler && step < 3) {
            StepRule::BackwardEuler
        } else {
            StepRule::Trapezoidal
        };
        let (alpha, jacobian) = match rule {
            StepRule::BackwardEuler => (alpha_backward_euler, &jacobian_backward_euler),
            StepRule::Trapezoidal => (alpha_trapezoidal, &jacobian_trapezoidal),
        };
        let previous_derivatives = runner.step(jacobian, alpha, rule, &source(time));
        let next_damp = if rule == StepRule::Trapezoidal
            && ringing_derivatives.len() == 2
            && times.len() == 3
        {
            let window = StepWindow {
                times: [times[0], times[1], times[2], time],
                charges: [&charges[0], &charges[1], &charges[2], &runner.charges],
                derivatives: [
                    &ringing_derivatives[0],
                    &ringing_derivatives[1],
                    &previous_derivatives,
                    &runner.derivatives,
                ],
                alpha,
            };
            detector.rings(
                &matrices.pattern,
                &circuit.natures,
                &mut ring_solver,
                &mut workspace,
                &window,
                &options,
            )
        } else {
            false
        };
        damps += usize::from(next_damp);
        damp = next_damp;
        update_history(
            &mut times,
            &mut charges,
            &mut ringing_derivatives,
            time,
            runner.charges.clone(),
            previous_derivatives,
            rule,
        );
    }
    (
        damps,
        ring_solver.statistics().reuses - reuses_before,
        detector,
    )
}

#[test]
fn scalar_harmonic_identity_passes_for_arbitrary_variable_steps() {
    let omega = 1.0e7;
    for &base_step in &[2.0e-8, 2.5e-7, 5.0e-7, 2.0e-6] {
        let steps = [base_step, 1.7 * base_step, 0.43 * base_step];
        let mut times = [0.0; 4];
        let mut charges = [1.0e-9, 0.0, 0.0, 0.0];
        let mut derivatives = [0.003, 0.0, 0.0, 0.0];
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
    let mut charges = [1.0e-9, 0.0, 0.0, 0.0];
    let mut derivatives = [-lambda * charges[0], 0.0, 0.0, 0.0];
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
    let flat = [1e-12; 4];
    let ringing = [1e-3, -1e-3, 1e-3, -1e-3];
    assert!(!is_harmonic_scalar(times, flat, ringing));
    let slope = 6e-3;
    let ramp = [0.0, slope * 1e-5, slope * 2e-5, slope * 3e-5];
    let noisy = [slope + 0.01, slope - 0.01, slope + 0.01, slope - 0.01];
    assert!(!is_harmonic_scalar(times, ramp, noisy));
}

#[test]
fn classifies_excess_phase_filter_separately_from_electrical_inductors() {
    let mut circuit = Circuit::new(&[Nature::Voltage, Nature::Current, Nature::Current]);
    circuit.capacitor(0, None, 1e-9);
    circuit.inductor(0, None, 1, 1e-6);
    circuit.stamp_conductance(2, 2, 1.0);
    circuit.stamp_capacitance(2, 2, 1e-11);
    let matrices = circuit.build();
    let system = matrices.linearized(&circuit.natures);
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
    let mut circuit = Circuit::electrical(3, 4);
    circuit.capacitor(0, Some(1), 1e-6);
    circuit.capacitor(1, Some(2), 0.8e-6);
    circuit.inductor(1, None, 3, 1e-5);
    circuit.inductor(2, None, 4, 2e-5);
    circuit.voltage_source(0, None, 5);
    circuit.voltage_source(2, Some(1), 6);

    let (damps_clean, reuses_clean, detector) = simulate_linear(
        &circuit,
        &[5.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        10e-6,
        120,
        StepRule::Trapezoidal,
        |_| source_vector(7, &[(5, 5.0)]),
    );
    assert_eq!(damps_clean, 0);
    assert_eq!(reuses_clean, 0);
    // Verify the saddle cache factored once and reused across subsequent alternating steps.
    let saddle = detector
        .loop_cache
        .as_ref()
        .and_then(|cache| cache.components[0].saddle.as_ref())
        .expect("cached saddle populated");
    assert_eq!(saddle.solver.statistics().factorizations, 1);
    assert!(saddle.solver.statistics().reuses > 0);

    let omega = TAU * 1e3;
    let phase: f64 = 0.5;
    let (damps_ringing, _, _) = simulate_linear(
        &circuit,
        &[5.0, 0.5, 0.5 + phase.sin().abs(), 0.0, 0.0, 0.0, 0.0],
        10e-6,
        200,
        StepRule::BackwardEuler,
        |time| source_vector(7, &[(5, 5.0), (6, (omega * time + phase).sin().abs())]),
    );
    assert!(
        damps_ringing >= 4,
        "expected damping on |sin| corners, got {damps_ringing}"
    );
}

#[test]
fn ungrounded_capacitive_island_distinguishes_zero_volt_source_from_corner() {
    let mut circuit = Circuit::electrical(3, 3);
    circuit.capacitor(0, Some(1), 1e-6);
    circuit.capacitor(1, Some(2), 0.5e-6);
    circuit.inductor(0, None, 3, 1e-5);
    circuit.inductor(2, None, 4, 2e-5);
    circuit.voltage_source(1, Some(0), 5);

    let (damps_clean, _, _) = simulate_linear(
        &circuit,
        &[0.5, 0.5, -0.2, 0.0, 0.0, 0.0],
        10e-6,
        120,
        StepRule::Trapezoidal,
        |_| vec![0.0; 6],
    );
    assert_eq!(damps_clean, 0);

    let omega = TAU * 1e3;
    let phase: f64 = 0.5;
    let (damps_ringing, _, _) = simulate_linear(
        &circuit,
        &[0.5, 0.5 + phase.sin().abs(), -0.2, 0.0, 0.0, 0.0],
        10e-6,
        200,
        StepRule::BackwardEuler,
        |time| source_vector(6, &[(5, (omega * time + phase).sin().abs())]),
    );
    assert!(damps_ringing >= 4, "got {damps_ringing}");
}

#[test]
fn parallel_rlc_separates_underdamped_from_critically_and_overdamped() {
    let capacitance: f64 = 1e-9;
    let inductance: f64 = 1e-6;
    let omega = 1.0 / (inductance * capacitance).sqrt();
    let impedance = (inductance / capacitance).sqrt();
    let build_rlc = |damping_ratio: f64| {
        let mut circuit = Circuit::electrical(1, 1);
        circuit.capacitor(0, None, capacitance);
        circuit.inductor(0, None, 1, inductance);
        if damping_ratio != 0.0 {
            circuit.stamp_conductance(0, 0, 2.0 * damping_ratio / impedance);
        }
        circuit
    };

    for &damping_ratio in &[0.0, 0.01] {
        let (damps, _, _) = simulate_linear(
            &build_rlc(damping_ratio),
            &[1.0, 0.0],
            5.0 / omega,
            40,
            StepRule::Trapezoidal,
            |_| vec![0.0, 0.0],
        );
        assert_eq!(damps, 0, "unexpected damping at zeta={damping_ratio}");
    }
    for &damping_ratio in &[1.0, 1.5, 3.0] {
        let (damps, _, _) = simulate_linear(
            &build_rlc(damping_ratio),
            &[1.0, 0.0],
            5.0 / omega,
            40,
            StepRule::Trapezoidal,
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
    let mut circuit = Circuit::new(&[
        Nature::Voltage,
        Nature::Voltage,
        Nature::Current,
        Nature::Current,
        Nature::Voltage,
    ]);
    circuit.capacitor(0, None, 1e-6);
    circuit.capacitor(1, None, 2e-6);
    circuit.inductor(0, Some(1), 2, 1e-5);
    circuit.inductor(1, None, 3, 2e-5);
    circuit.stamp_conductance(4, 4, 0.01);
    circuit.capacitor(4, None, 1e-9);

    let (damps, _, _) = simulate_linear(
        &circuit,
        &[1.0, -1.0, 0.0, 0.0, 1e-3],
        3e-6,
        40,
        StepRule::Trapezoidal,
        |_| vec![0.0; 5],
    );
    assert!(
        damps >= 1,
        "stiff RC hidden by large LC ladder must be caught"
    );
}

#[test]
fn mixed_lossless_lc_with_meyer_displacement_catches_ringing() {
    let mut circuit = Circuit::new(&[Nature::Voltage, Nature::Current, Nature::Displacement]);
    circuit.capacitor(0, None, 1e-6);
    circuit.inductor(0, None, 1, 1e-5);
    circuit.stamp_conductance(2, 2, 1.0);
    let times = [0.0, 1e-5, 2e-5, 3e-5];
    let charges = [1e-6, 0.0, 0.0];
    let derivatives: [&[f64]; 4] = [
        &[0.0, 0.0, 1e-3],
        &[0.0, 0.0, -1e-3],
        &[0.0, 0.0, 1e-3],
        &[0.0, 0.0, -1e-3],
    ];
    let rings = evaluate_rings(&circuit, times, [&charges; 4], derivatives);
    assert!(rings);
}

#[test]
fn thermal_and_state_stiff_modes_ring_above_stability_boundary() {
    let time_constant = 1.0e-6;
    let step = 10.0 * time_constant;
    let times = [0.0, step, 2.0 * step, 3.0 * step];
    let ratio = (2.0 - step / time_constant) / (2.0 + step / time_constant);
    let powers = [1.0, ratio, ratio * ratio, ratio * ratio * ratio];
    let [q0, q1, q2, q3] = powers.map(|power| [time_constant * power]);
    let [d0, d1, d2, d3] = powers.map(|power| [-power]);
    let charge_slices: [&[f64]; 4] = [&q0, &q1, &q2, &q3];
    let derivative_slices: [&[f64]; 4] = [&d0, &d1, &d2, &d3];
    for nature in [Nature::Temperature, Nature::State] {
        let mut circuit = Circuit::new(&[nature]);
        circuit.stamp_conductance(0, 0, 1.0);
        circuit.stamp_capacitance(0, 0, time_constant);
        let rings = evaluate_rings(&circuit, times, charge_slices, derivative_slices);
        assert!(rings);
    }
}

#[test]
fn vcvs_inside_two_mode_lc_has_zero_loop_multiplier() {
    let mut circuit = Circuit::electrical(3, 3);
    circuit.capacitor(0, None, 1e-6);
    circuit.capacitor(1, None, 0.7e-6);
    circuit.capacitor(0, Some(1), 0.5e-6);
    circuit.capacitor(1, Some(2), 0.5e-6);
    circuit.inductor(0, None, 3, 1e-5);
    circuit.inductor(1, None, 4, 2e-5);
    circuit.voltage_source(2, None, 5);
    circuit.stamp_conductance(5, 1, -2.0);

    let step_size = 10e-6;
    let alpha = 2.0 / step_size;
    let matrices = circuit.build();
    let jacobian = matrices.jacobian(alpha);
    let zero_source = vec![0.0; 6];
    let mut runner = LinearRunner::new(
        &matrices,
        &[1.0, -0.5, -1.0, 0.0, 0.0, 0.0],
        StepRule::Trapezoidal,
        &zero_source,
    );
    runner.derivatives[5] = 0.0;
    let mut charge_history = vec![runner.charges.clone()];
    let mut derivative_history = vec![runner.derivatives.clone()];
    for _ in 0..20 {
        runner.step(&jacobian, alpha, StepRule::Trapezoidal, &zero_source);
        charge_history.push(runner.charges.clone());
        derivative_history.push(runner.derivatives.clone());
    }

    let system = matrices.linearized(&circuit.natures);
    let ideal = CachedSources::build(system).sources;
    assert_eq!(ideal.len(), 1);
    assert_eq!(ideal[0].unknown, 5);
    let options = Options::default();
    let mut detector = Detector::default();
    assert!(detector.refresh_sources(system));
    for step in 3..derivative_history.len() {
        let window = StepWindow {
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
        let rings = detector.has_voltage_source_loop_ringing(system, &[0, 1, 2], &window, &options);
        assert!(!rings);
    }
    let saddle = detector
        .loop_cache
        .as_ref()
        .and_then(|cache| cache.components[0].saddle.as_ref())
        .expect("cached saddle populated");
    assert_eq!(saddle.solver.statistics().factorizations, 1);
    assert_eq!(
        saddle.solver.statistics().reuses,
        derivative_history.len() - 4
    );
}

#[test]
fn grounded_corner_source_coupled_through_capacitor_to_large_lc_tank_is_caught() {
    // A grounded source at node 0 couples via `coupling_capacitance` to a large single-mode
    // lossless LC tank at node 1 (`v = 3 V`, `omega * h = 2.17 > 2`). The tank dominates
    // node 0's charge derivative so `is_harmonic_scalar` passes on every alternating unknown.
    // Stage 2d cancels the tank exactly, catching 6.28 mA `|sin|` ringing while producing
    // zero false damps on 5 V DC.
    let mut circuit = Circuit::electrical(2, 2);
    circuit.capacitor(0, Some(1), 1e-6);
    circuit.capacitor(1, None, 1e-6);
    circuit.inductor(1, None, 2, 10e-6);
    circuit.voltage_source(0, None, 3);

    let (damps_dc, reuses_dc, _) = simulate_linear(
        &circuit,
        &[5.0, 3.0, 0.0, 0.0],
        9.7e-6,
        120,
        StepRule::Trapezoidal,
        |_| source_vector(4, &[(3, 5.0)]),
    );
    assert_eq!(damps_dc, 0);
    assert_eq!(reuses_dc, 0);

    let omega = TAU * 1e3;
    let (damps_corner, _, _) = simulate_linear(
        &circuit,
        &[0.0, 3.0, 0.0, 0.0],
        9.7e-6,
        200,
        StepRule::BackwardEuler,
        |time| source_vector(4, &[(3, (omega * time).sin().abs())]),
    );
    assert!(
        damps_corner >= 4,
        "expected damping on grounded |sin| coupled to large LC tank, got {damps_corner}"
    );
}
