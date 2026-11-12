//! Newton iteration on the circuit's equations.

use jolty_models::{Contributions, Input, Quantity};

use crate::elaborate::System;
use crate::{Options, Simulator};

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
    /// `shunt` from every node to the reference, deflated by the solutions `deflated`
    /// already found, whose basins the iteration is pushed out of.
    Dc {
        source_scale: f64,
        shunt: f64,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Failure {
    /// The Jacobian was singular.
    Singular,
    /// The iteration limit was reached.
    Diverged,
}

impl Simulator {
    /// Evaluates every device at `x`, filling the workspace's residuals, charges and their
    /// derivatives, with their steps limited if `limiting`. Returns whether a device limited
    /// its voltage.
    pub(crate) fn evaluate(
        &mut self,
        x: &[f64],
        time: f64,
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
            Mode::Dc { source_scale, .. } => (0.0, source_scale),
            Mode::Step { time, .. } => (time, 1.0),
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
            let workspace = &mut self.workspace;
            let residual = &workspace.contributions.residual;
            for (step, residual) in workspace.step.iter_mut().zip(residual) {
                *step = -residual;
            }
            self.solver
                .factor(&workspace.jacobian)
                .and_then(|mut factors| factors.solve(&mut workspace.step))
                .map_err(|_| Failure::Singular)?;
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
            Mode::Dc { shunt, .. } => {
                jacobian.copy_from_slice(&contributions.conductance);
                if shunt > 0.0 {
                    for &(unknown, entry) in &self.system.node_diagonals {
                        let index = unknown.index();
                        jacobian[self.system.pattern.slot(entry)] += shunt;
                        let current = shunt * x[index];
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
                let rounding = RESIDUAL_ROUNDINGS * f64::EPSILON * workspace.resolution[index];
                let contributions = &workspace.contributions;
                contributions.residual[index].abs()
                    <= options.relative_tolerance * contributions.scale[index] + absolute + rounding
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
