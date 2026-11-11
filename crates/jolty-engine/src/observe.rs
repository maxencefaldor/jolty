//! What every device does at every solution: the current into each of its terminals and the
//! charge each holds.
//!
//! The engine solves for node voltages and a few branch currents, so a device's own currents
//! are not among its unknowns. To read them exactly, each device is made a second time with
//! every terminal on a node of its own, and evaluated at the solution: the residual of a
//! terminal's own node is then the current into that terminal alone, even where two of its
//! terminals share a net or one joins the reference, and the charge of that node the charge
//! the terminal holds. In a transient, the current adds the charge's derivative, taken by
//! the very integration formula the step was solved with, so the currents into a node sum to
//! zero as the node's equation does.

use jolty_models::{Binding, Device, Input, Node, Output, Quantity, Unknown, UnknownInfo};
use jolty_numerics::integration::Formula;
use jolty_numerics::{Complex64, Pattern, PatternBuilder};

use crate::elaborate::System;
use crate::{ElaborationError, ElaborationProblem};

/// The terminals of every device and what they did at one solution.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Terminals {
    /// The current into each terminal, in amperes.
    pub currents: Vec<f64>,
    /// The charge each terminal holds, in coulombs.
    pub charges: Vec<f64>,
}

/// A device made again with each terminal on a node of its own.
#[derive(Debug)]
struct Probe {
    /// The device's index among the simulator's devices.
    placed: usize,
    device: Box<dyn Device>,
    pattern: Pattern,
    /// The solution's node or unknown for each of the copy's unknowns: its terminals, then
    /// its internal unknowns.
    sources: Vec<Node>,
    /// The number of terminals.
    terminals: usize,
    /// The index of its first terminal among every device's.
    first: usize,
    x: Vec<f64>,
    residual: Vec<f64>,
    scale: Vec<f64>,
    charge: Vec<f64>,
    conductance: Vec<f64>,
    capacitance: Vec<f64>,
}

impl Probe {
    /// Evaluates the copy at the solution `x`, at `time`.
    fn evaluate(&mut self, x: &[f64], time: f64, gmin: f64) {
        for (value, source) in self.x.iter_mut().zip(&self.sources) {
            *value = source.map_or(0.0, |unknown| x[unknown.index()]);
        }
        for buffer in [
            &mut self.residual,
            &mut self.scale,
            &mut self.charge,
            &mut self.conductance,
            &mut self.capacitance,
        ] {
            buffer.fill(0.0);
        }
        let mut output = Output::new(
            &self.pattern,
            &mut self.residual,
            &mut self.scale,
            &mut self.charge,
            &mut self.conductance,
            &mut self.capacitance,
        );
        let mut input = Input::new(&self.x, time, gmin);
        input.limiting = false;
        self.device.evaluate(&input, &mut output);
    }
}

/// The charges of every terminal at the last accepted point of a transient, and their
/// derivatives, as the transient's own history keeps the nodes'.
#[derive(Debug, Default)]
struct History {
    last: Vec<f64>,
    before: Vec<f64>,
    derivative: Vec<f64>,
    /// Whether the last point began the integration afresh, so has no point before it.
    restarted: bool,
}

/// Every device that is made from a model, made again to be read.
#[derive(Debug)]
pub(crate) struct Observer {
    probes: Vec<Probe>,
    /// Each terminal's name: its device's path and its own, such as `X1/R1.a`.
    names: Vec<String>,
    history: History,
}

impl Observer {
    /// Makes every device of `system` that a model made again, each terminal on its own node.
    pub(crate) fn new(system: &System) -> Result<Self, ElaborationError> {
        let mut probes = Vec::new();
        let mut names = Vec::new();
        let mut problems = Vec::new();
        for (placed, device) in system.devices.iter().enumerate() {
            let Some(made) = &device.made else {
                continue;
            };
            let terminals = device.nodes().len();
            let own: Vec<Node> = (0..terminals)
                .map(|index| Some(Unknown::new(index)))
                .collect();
            let mut unknowns: Vec<UnknownInfo> = (0..terminals)
                .map(|_| UnknownInfo {
                    name: String::new(),
                    quantity: Quantity::Voltage,
                })
                .collect();
            let mut builder = PatternBuilder::new();
            let mut binding = Binding::new(
                device.path(),
                &own,
                &mut unknowns,
                &mut builder,
                system.temperature,
            );
            let copy = match made.model.instantiate(&made.parameters, &mut binding) {
                Ok(copy) => copy,
                Err(error) => {
                    problems.push(ElaborationProblem::Parameters {
                        at: device.location().clone(),
                        message: error.0,
                    });
                    continue;
                }
            };
            let count = unknowns.len();
            let mut sources = device.nodes().to_vec();
            sources.extend(
                (device.first_internal..device.first_internal + count - terminals)
                    .map(|index| Some(Unknown::new(index))),
            );
            let pattern = builder.finish(count);
            let stored = pattern.len();
            names.extend(
                device
                    .terminals()
                    .iter()
                    .map(|terminal| format!("{}.{terminal}", device.path())),
            );
            probes.push(Probe {
                placed,
                device: copy,
                pattern,
                sources,
                terminals,
                first: names.len() - terminals,
                x: vec![0.0; count],
                residual: vec![0.0; count],
                scale: vec![0.0; count],
                charge: vec![0.0; count],
                conductance: vec![0.0; stored],
                capacitance: vec![0.0; stored],
            });
        }
        if !problems.is_empty() {
            return Err(ElaborationError::new(problems));
        }
        Ok(Self {
            probes,
            names,
            history: History::default(),
        })
    }

    /// Each terminal's name: its device's path and its own, such as `X1/R1.a`.
    pub(crate) fn names(&self) -> &[String] {
        &self.names
    }

    /// Holds the copy of the source at `placed` at `level`, or releases it, as the source
    /// itself is.
    pub(crate) fn hold(&mut self, placed: usize, level: Option<f64>) {
        for probe in &mut self.probes {
            if probe.placed == placed
                && let Some(source) = probe.device.as_source()
            {
                source.hold(level);
            }
        }
    }

    /// The terminals at the DC solution `x`: each current, without a charge's derivative.
    pub(crate) fn dc(&mut self, x: &[f64], gmin: f64) -> Terminals {
        self.at(x, 0.0, gmin)
    }

    /// The terminals at the DC solution `x` that begins a transient, whose charges stand
    /// still.
    pub(crate) fn start(&mut self, x: &[f64], gmin: f64) -> Terminals {
        let found = self.dc(x, gmin);
        self.history = History {
            last: found.charges.clone(),
            before: found.charges.clone(),
            derivative: vec![0.0; found.charges.len()],
            restarted: true,
        };
        found
    }

    /// The terminals at the transient's accepted point `x`, at `time`, solved by
    /// `formula`; `restart` when the integration begins afresh from it.
    pub(crate) fn step(
        &mut self,
        x: &[f64],
        time: f64,
        gmin: f64,
        (formula, restart): (&Formula, bool),
    ) -> Terminals {
        let mut found = self.at(x, time, gmin);
        let history = &mut self.history;
        let derivative: Vec<f64> = found
            .charges
            .iter()
            .enumerate()
            .map(|(index, &charge)| {
                let before = if history.restarted {
                    history.last[index]
                } else {
                    history.before[index]
                };
                formula.alpha * charge
                    + formula.history([history.last[index], before], history.derivative[index])
            })
            .collect();
        for (current, change) in found.currents.iter_mut().zip(&derivative) {
            *current += change;
        }
        history.before = std::mem::replace(&mut history.last, found.charges.clone());
        history.derivative = derivative;
        history.restarted = restart;
        found
    }

    /// The terminals at `x` at `time`, without a charge's derivative.
    fn at(&mut self, x: &[f64], time: f64, gmin: f64) -> Terminals {
        let mut found = Terminals {
            currents: vec![0.0; self.names.len()],
            charges: vec![0.0; self.names.len()],
        };
        for probe in &mut self.probes {
            probe.evaluate(x, time, gmin);
            let range = probe.first..probe.first + probe.terminals;
            found.currents[range.clone()].copy_from_slice(&probe.residual[..probe.terminals]);
            found.charges[range].copy_from_slice(&probe.charge[..probe.terminals]);
        }
        found
    }

    /// Readies the small-signal currents around the DC solution `x`.
    pub(crate) fn linearize(&mut self, x: &[f64], gmin: f64) {
        for probe in &mut self.probes {
            probe.evaluate(x, 0.0, gmin);
        }
    }

    /// The small-signal current into each terminal at angular frequency `omega`, for the
    /// phasors `values` of the unknowns, with the source at `excited` driven by a unit
    /// excitation, after [`Observer::linearize`].
    pub(crate) fn phasors(
        &mut self,
        values: &[Complex64],
        omega: f64,
        excited: usize,
    ) -> Vec<Complex64> {
        let mut found = vec![Complex64::default(); self.names.len()];
        for probe in &mut self.probes {
            let phasor = |column: usize| {
                probe.sources[column]
                    .map_or(Complex64::default(), |unknown| values[unknown.index()])
            };
            let positions = probe
                .pattern
                .positions()
                .zip(probe.conductance.iter().zip(&probe.capacitance));
            for ((row, column), (&conductance, &capacitance)) in positions {
                if row < probe.terminals {
                    found[probe.first + row] +=
                        Complex64::new(conductance, omega * capacitance) * phasor(column);
                }
            }
            if probe.placed == excited
                && let Some(source) = probe.device.as_source()
            {
                let mut derivative = vec![0.0; probe.x.len()];
                source.level_derivative(&mut derivative);
                for (terminal, &value) in derivative[..probe.terminals].iter().enumerate() {
                    found[probe.first + terminal] += Complex64::new(value, 0.0);
                }
            }
        }
        found
    }
}
