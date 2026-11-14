//! Newton iteration on the circuit's equations.

use jolty_models::{Contributions, Input, Quantity};

use crate::elaborate::System;
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

/// The buffers one evaluation of the circuit fills, allocated once per simulator.
#[derive(Debug)]
pub(crate) struct Workspace {
    /// What the devices added; after [`Simulator::combine`], the residual is the full
    /// residual of the system being solved.
    pub(crate) contributions: Contributions,
    /// The Jacobian of the system being solved, by pattern slot.
    jacobian: Vec<f64>,
    /// `|J|·|x|` of each equation: how much its residual moves when every unknown moves by
    /// its own magnitude, which bounds the residual's floating-point resolution.
    resolution: Vec<f64>,
    /// The last Newton step.
    step: Vec<f64>,
}

impl Workspace {
    pub(crate) fn new(system: &System) -> Self {
        let (unknowns, stored) = (system.unknowns.len(), system.pattern.len());
        Self {
            contributions: Contributions::new(&system.pattern),
            jacobian: vec![0.0; stored],
            resolution: vec![0.0; unknowns],
            step: vec![0.0; unknowns],
        }
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
    pub(crate) fn evaluate(
        &mut self,
        x: &[f64],
        time: Option<f64>,
        (source_scale, gmin): (f64, f64),
        limiting: bool,
    ) -> bool {
        let mut output = self.workspace.contributions.output(&self.system.pattern);
        let mut input = Input::new(x, time, gmin);
        input.source_scale = source_scale;
        input.limiting = limiting;
        for placed in &mut self.system.devices {
            placed.device.evaluate(&input, &mut output);
        }
        output.is_limited()
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
        // A DC solution starts afresh from its guess; a time step goes on from the last.
        let afresh = matches!(mode, Mode::Dc { .. });
        let mut stepped_within_tolerance = false;
        for iteration in 0..limit {
            let limiting = iteration > 0 || !afresh;
            let limited = self.evaluate(x, time, (source_scale, options.gmin), limiting);
            self.combine(x, mode);
            if stepped_within_tolerance && !limited && self.residual_within_tolerance(options) {
                return Ok(iteration);
            }
            // A residual within the rounding of the unknowns its equation reads is noise: the
            // step it would drive is one the unknowns cannot record, and chasing it moves
            // the others without end, as a source's current beside a node at megavolts.
            for index in 0..x.len() {
                let residual = self.workspace.contributions.residual[index];
                self.workspace.step[index] = if residual.abs() <= self.rounding(index) {
                    0.0
                } else {
                    -residual
                };
            }
            // The solver keeps the factors of a Jacobian bitwise unchanged since the last, as
            // a linear circuit's is from one iteration to the next, and refactors a changed
            // one with the last pivots.
            let workspace = &mut self.workspace;
            self.solver
                .factor(&workspace.jacobian)
                .and_then(|mut factors| factors.solve(&mut workspace.step))
                .map_err(Failure::Solver)?;
            if let Mode::Dc { deflated, .. } = mode {
                self.deflate(x, deflated);
            }
            for (value, step) in x.iter_mut().zip(&self.workspace.step) {
                *value += step;
            }
            stepped_within_tolerance = self.step_within_tolerance(x, options);
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
    fn deflate(&mut self, x: &[f64], known: &[Vec<f64>]) {
        let step = &mut self.workspace.step;
        let mut slope = 0.0;
        for root in known {
            let (mut distance, mut along) = (0.0, 0.0);
            for (index, unknown) in self.system.unknowns.iter().enumerate() {
                if unknown.quantity == Quantity::Voltage {
                    let apart = x[index] - root[index];
                    distance += apart * apart;
                    along += apart * step[index];
                }
            }
            if distance > 0.0 {
                // ∇(1/d² + SHIFT)·δ = −2 (x − r)·δ / d⁴.
                slope += -2.0 * along / (distance * distance) / (distance.recip() + SHIFT);
            }
        }
        let scale = (1.0 - slope).recip();
        if scale.is_finite() {
            for value in step.iter_mut() {
                *value *= scale;
            }
        }
    }

    /// Turns the evaluated device contributions into the residual and Jacobian of `mode`.
    fn combine(&mut self, x: &[f64], mode: Mode<'_>) {
        let Workspace {
            contributions,
            jacobian,
            resolution,
            ..
        } = &mut self.workspace;
        match mode {
            Mode::Dc { shunt, anchor, .. } => {
                jacobian.copy_from_slice(&contributions.conductance);
                if shunt > 0.0 {
                    for &(unknown, entry) in &self.system.node_diagonals {
                        let index = unknown.index();
                        jacobian[self.system.pattern.slot(entry)] += shunt;
                        let held = anchor.map_or(0.0, |anchor| anchor[index]);
                        let current = shunt * (x[index] - held);
                        contributions.residual[index] += current;
                        let scale = &mut contributions.scale[index];
                        *scale = scale.max(current.abs());
                    }
                }
            }
            Mode::Step { alpha, history, .. } => {
                for ((jacobian, conductance), capacitance) in jacobian
                    .iter_mut()
                    .zip(&contributions.conductance)
                    .zip(&contributions.capacitance)
                {
                    *jacobian = conductance + alpha * capacitance;
                }
                let rows = contributions
                    .residual
                    .iter_mut()
                    .zip(&mut contributions.scale);
                for ((residual, scale), (&charge, &term)) in
                    rows.zip(contributions.charge.iter().zip(history))
                {
                    let reactive = alpha * charge + term;
                    *residual += reactive;
                    *scale = scale.max(reactive.abs());
                }
            }
        }
        resolution.fill(0.0);
        for ((row, column), &value) in self.system.pattern.positions().zip(jacobian.iter()) {
            resolution[row] += (value * x[column]).abs();
        }
    }

    /// Whether the last step was within tolerance of every unknown it moved.
    fn step_within_tolerance(&self, x: &[f64], options: &Options) -> bool {
        (0..x.len()).all(|index| self.step_error(x, index, options) <= 1.0)
    }

    /// The last step of unknown `index` as a multiple of its tolerance.
    fn step_error(&self, x: &[f64], index: usize, options: &Options) -> f64 {
        let step = self.workspace.step[index];
        let magnitude = x[index].abs().max((x[index] - step).abs());
        let absolute = options.value_tolerance(self.system.unknowns[index].quantity);
        step.abs() / (options.relative_tolerance * magnitude + absolute)
    }

    /// Whether every residual is within tolerance of the terms it sums, or within the
    /// rounding of the unknowns it depends on. A node's residual is a current; a branch's is a
    /// voltage.
    fn residual_within_tolerance(&self, options: &Options) -> bool {
        let workspace = &self.workspace;
        self.system
            .unknowns
            .iter()
            .enumerate()
            .all(|(index, unknown)| {
                let absolute = options.equation_tolerance(unknown.quantity);
                let contributions = &workspace.contributions;
                contributions.residual[index].abs()
                    <= options.relative_tolerance * contributions.scale[index]
                        + absolute
                        + self.rounding(index)
            })
    }

    /// The floating-point resolution of equation `index`'s residual: the rounding of the
    /// unknowns it reads, which no step can resolve.
    fn rounding(&self, index: usize) -> f64 {
        RESIDUAL_ROUNDINGS * f64::EPSILON * self.workspace.resolution[index]
    }

    /// The names of the `count` unknowns whose last step was largest against its tolerance.
    pub(crate) fn furthest_from_converging(
        &self,
        x: &[f64],
        options: &Options,
        count: usize,
    ) -> Vec<String> {
        let mut errors: Vec<(f64, usize)> = (0..x.len())
            .map(|index| (self.step_error(x, index, options), index))
            .collect();
        errors.sort_by(|a, b| b.0.total_cmp(&a.0));
        errors
            .into_iter()
            .take(count)
            .map(|(_, index)| self.system.unknowns[index].name.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
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

    /// The factorizations that searched for pivots, the refactorizations and the reuses of
    /// factors, from `before` to `after`.
    fn work(before: Statistics, after: Statistics) -> (usize, usize, usize) {
        (
            after.factorizations - before.factorizations,
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
        let statistics = simulator.solver.statistics();
        assert_eq!(work(Statistics::default(), statistics), (1, 0, 1));
    }

    #[test]
    fn a_nonlinear_operating_point_refactors_with_its_first_pivots() {
        let mut simulator = ladder(3, &json!({"dc": 1.0}), true);
        simulator.operating_point(&Options::default()).unwrap();
        let (searched, refactored, _) = work(Statistics::default(), simulator.solver.statistics());
        assert_eq!(searched, 1);
        assert!(refactored >= 2, "{refactored} refactorizations");
    }

    #[test]
    fn a_linear_transient_factors_only_as_its_step_or_formula_changes() {
        // A sine wave keeps the ladder moving, so that every step takes an iteration at
        // least, each of which once made a factorization. The Jacobian, `G + αC`, now changes
        // only with `α`, as the step's length or formula changes: while the first steps grow
        // to the longest, a millionth of a second, on the last, landing on the stop, on each
        // step tried and rejected, and once as the formula turns from backward Euler, which
        // starts the run, to the trapezoidal rule.
        let sine = json!({"sine": {"offset": 0.0, "amplitude": 1.0, "frequency": 1e3}});
        let mut simulator = ladder(5, &sine, false);
        let options = Options::default();
        simulator.operating_point(&options).unwrap();
        let before = simulator.solver.statistics();
        let transient = Transient {
            stop: 1e-3,
            max_step: Some(1e-6),
            method: Method::Trapezoidal,
            times: Vec::new(),
            periodic: false,
        };
        let waveforms = simulator.transient(&transient, &options).unwrap();
        let (searched, refactored, reused) = work(before, simulator.solver.statistics());
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
        assert!(
            searched + refactored <= changes + steps.rejected + formulas,
            "{searched} + {refactored} factorizations over {} steps, {ramp} before the longest",
            steps.accepted
        );
    }
}
