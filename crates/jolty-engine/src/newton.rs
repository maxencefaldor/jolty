//! Newton iteration on the circuit's equations.

use jolty_models::{Input, Output, Quantity};

use crate::elaborate::System;
use crate::{Options, Simulator};

/// How many roundings of the unknowns a residual may show and still count as zero.
///
/// Each unknown is stored to half a unit in its last place, so a residual cannot resolve
/// changes finer than `ε·(|J|·|x|)` of its row, and each term it sums adds a rounding or two
/// of its own. Without this floor, an equation whose terms cancel, such as the current
/// through a milliohm between two nodes at hundreds of volts, could never converge.
const RESIDUAL_ROUNDINGS: f64 = 8.0;

/// The buffers one evaluation of the circuit fills, allocated once per simulator.
#[derive(Debug)]
pub(crate) struct Workspace {
    /// The residual of every equation; after [`Simulator::combine`], the full residual of the
    /// system being solved.
    pub(crate) residual: Vec<f64>,
    /// The largest magnitude of a term of each residual.
    pub(crate) scale: Vec<f64>,
    /// The charge of every equation.
    pub(crate) charge: Vec<f64>,
    /// `∂f/∂x`, by pattern slot.
    pub(crate) conductance: Vec<f64>,
    /// `∂q/∂x`, by pattern slot.
    pub(crate) capacitance: Vec<f64>,
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
            residual: vec![0.0; unknowns],
            scale: vec![0.0; unknowns],
            charge: vec![0.0; unknowns],
            conductance: vec![0.0; stored],
            capacitance: vec![0.0; stored],
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
    /// `shunt` from every node to the reference.
    Dc { source_scale: f64, shunt: f64 },
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
    /// derivatives. Returns whether a device limited its voltage.
    pub(crate) fn evaluate(&mut self, x: &[f64], time: f64, source_scale: f64, gmin: f64) -> bool {
        let workspace = &mut self.workspace;
        for buffer in [
            &mut workspace.residual,
            &mut workspace.scale,
            &mut workspace.charge,
            &mut workspace.conductance,
            &mut workspace.capacitance,
        ] {
            buffer.fill(0.0);
        }
        let mut output = Output::new(
            &self.system.pattern,
            &mut workspace.residual,
            &mut workspace.scale,
            &mut workspace.charge,
            &mut workspace.conductance,
            &mut workspace.capacitance,
        );
        let mut input = Input::new(x, time, gmin);
        input.source_scale = source_scale;
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
        let mut stepped_within_tolerance = false;
        for iteration in 0..limit {
            let limited = self.evaluate(x, time, source_scale, options.gmin);
            self.combine(x, mode);
            if stepped_within_tolerance && !limited && self.residual_within_tolerance(options) {
                return Ok(iteration);
            }
            let workspace = &mut self.workspace;
            for (step, residual) in workspace.step.iter_mut().zip(&workspace.residual) {
                *step = -residual;
            }
            self.solver
                .factor(&workspace.jacobian)
                .and_then(|mut factors| factors.solve(&mut workspace.step))
                .map_err(|_| Failure::Singular)?;
            for (value, step) in x.iter_mut().zip(&workspace.step) {
                *value += step;
            }
            stepped_within_tolerance = self.step_within_tolerance(x, options);
        }
        Err(Failure::Diverged)
    }

    /// Turns the evaluated device contributions into the residual and Jacobian of `mode`.
    fn combine(&mut self, x: &[f64], mode: Mode<'_>) {
        let workspace = &mut self.workspace;
        match mode {
            Mode::Dc { shunt, .. } => {
                workspace.jacobian.copy_from_slice(&workspace.conductance);
                if shunt > 0.0 {
                    for &(unknown, entry) in &self.system.node_diagonals {
                        let index = unknown.index();
                        workspace.jacobian[self.system.pattern.slot(entry)] += shunt;
                        let current = shunt * x[index];
                        workspace.residual[index] += current;
                        workspace.scale[index] = workspace.scale[index].max(current.abs());
                    }
                }
            }
            Mode::Step { alpha, history, .. } => {
                for ((jacobian, conductance), capacitance) in workspace
                    .jacobian
                    .iter_mut()
                    .zip(&workspace.conductance)
                    .zip(&workspace.capacitance)
                {
                    *jacobian = conductance + alpha * capacitance;
                }
                let rows = workspace.residual.iter_mut().zip(&mut workspace.scale);
                for ((residual, scale), (&charge, &term)) in
                    rows.zip(workspace.charge.iter().zip(history))
                {
                    let reactive = alpha * charge + term;
                    *residual += reactive;
                    *scale = scale.max(reactive.abs());
                }
            }
        }
        workspace.resolution.fill(0.0);
        for ((row, column), &value) in self.system.pattern.positions().zip(&workspace.jacobian) {
            workspace.resolution[row] += (value * x[column]).abs();
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
        let absolute = match self.system.unknowns[index].quantity {
            Quantity::Voltage => options.voltage_tolerance,
            Quantity::Current => options.current_tolerance,
        };
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
                let absolute = match unknown.quantity {
                    Quantity::Voltage => options.current_tolerance,
                    Quantity::Current => options.voltage_tolerance,
                };
                let rounding = RESIDUAL_ROUNDINGS * f64::EPSILON * workspace.resolution[index];
                workspace.residual[index].abs()
                    <= options.relative_tolerance * workspace.scale[index] + absolute + rounding
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
