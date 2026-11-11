//! What every device does at every solution: the current into each of its terminals and the
//! charge each holds, the derivatives of its equations with respect to its unknowns, and the
//! variables it computes on its way.
//!
//! The engine solves for node voltages and a few branch currents, so a device's own currents
//! are not among its unknowns. To read them exactly, each device is made a second time with
//! every terminal on a node of its own, and evaluated at the solution: the residual of a
//! terminal's own node is then the current into that terminal alone, even where two of its
//! terminals share a net or one joins the reference, and the charge of that node the charge
//! the terminal holds. In a transient, the current adds the charge's derivative, taken by
//! the very integration formula the step was solved with, so the currents into a node sum to
//! zero as the node's equation does. The same evaluation gives the derivatives of the
//! device's residuals and charges, its conductances and capacitances, over its terminals
//! apart and its internal unknowns: the device's own small-signal model, exactly.

use jolty_models::{
    Binding, Device, Input, Level, LogicDriver, LogicDriverParameters, NativeDevice, Node, Output,
    ParameterError, Quantity, ThresholdDetector, Unknown, UnknownInfo,
};
use jolty_numerics::integration::Formula;
use jolty_numerics::{Complex64, Pattern, PatternBuilder};

use crate::elaborate::{Origin, Placed, System};
use crate::{ElaborationError, ElaborationProblem};

/// What every device did at one solution.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Observation {
    /// The current into each terminal, in amperes, in the order of
    /// [`Simulator::terminals`](crate::Simulator::terminals).
    pub currents: Vec<f64>,
    /// The charge each terminal holds, in coulombs, in the same order.
    pub charges: Vec<f64>,
    /// The derivative of each equation's residual with respect to each unknown, in the
    /// order of [`Simulator::derivatives`](crate::Simulator::derivatives): a conductance
    /// where the equation sums currents and the unknown is a voltage.
    pub conductances: Vec<f64>,
    /// The derivative of each equation's charge with respect to each unknown, in the same
    /// order: a capacitance where the equation sums currents and the unknown is a voltage.
    pub capacitances: Vec<f64>,
    /// The value of each variable, in the order of
    /// [`Simulator::variables`](crate::Simulator::variables).
    pub variables: Vec<f64>,
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
    /// The index of its first derivative among every device's: its equations' by its
    /// unknowns', row by row, every pair of them.
    first_derivative: usize,
    /// Its variables' range among every device's.
    variables: std::ops::Range<usize>,
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

/// Every device, made again to be read.
#[derive(Debug)]
pub(crate) struct Observer {
    probes: Vec<Probe>,
    /// Each terminal's name: its device's path and its own, such as `X1/R1.a`.
    names: Vec<String>,
    /// Each derivative's equation and unknown, by name.
    derivatives: Vec<(String, String)>,
    /// Each variable, by its device's path and its own name.
    variables: Vec<(String, &'static str)>,
    /// Each terminal of a component instance or a logic device, with the terminals whose
    /// currents and charges sum to its own.
    composites: Vec<(String, Vec<usize>)>,
    history: History,
}

impl Observer {
    /// Makes every device of `system` again, each terminal on its own node.
    pub(crate) fn new(system: &System) -> Result<Self, ElaborationError> {
        let mut observer = Self {
            probes: Vec::new(),
            names: Vec::new(),
            derivatives: Vec::new(),
            variables: Vec::new(),
            composites: Vec::new(),
            history: History::default(),
        };
        let mut problems = Vec::new();
        for (placed, device) in system.devices.iter().enumerate() {
            if let Err(error) = observer.probe(system, placed, device) {
                problems.push(ElaborationProblem::Parameters {
                    at: device.location().clone(),
                    message: error.0,
                });
            }
        }
        if !problems.is_empty() {
            return Err(ElaborationError::new(problems));
        }
        observer.composites = system
            .composites
            .iter()
            .map(|composite| {
                let members = composite
                    .members
                    .iter()
                    .map(|&(placed, terminal)| observer.probes[placed].first + terminal)
                    .collect();
                (composite.name.clone(), members)
            })
            .collect();
        Ok(observer)
    }

    /// Makes `device`, at index `placed` of `system`'s devices, again with each terminal on
    /// its own node, and names its terminals, derivatives and variables.
    fn probe(
        &mut self,
        system: &System,
        placed: usize,
        device: &Placed,
    ) -> Result<(), ParameterError> {
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
        let copy: Box<dyn Device> = match &device.origin {
            Origin::Model(made) => made.model.instantiate(&made.parameters, &mut binding)?,
            Origin::Detector(parameters) => {
                Box::new(ThresholdDetector::new(*parameters, &mut binding)?)
            }
            Origin::Driver => Box::new(LogicDriver::new(LogicDriverParameters {}, &mut binding)?),
        };
        let count = unknowns.len();
        let mut sources = device.nodes().to_vec();
        sources.extend(
            (device.first_internal..device.first_internal + count - terminals)
                .map(|index| Some(Unknown::new(index))),
        );
        let local: Vec<String> = device
            .terminals()
            .iter()
            .map(|terminal| format!("{}.{terminal}", device.path()))
            .chain(
                sources[terminals..]
                    .iter()
                    .flatten()
                    .map(|unknown| system.unknowns[unknown.index()].name.clone()),
            )
            .collect();
        let pattern = builder.finish(count);
        let stored = pattern.len();
        let first = self.names.len();
        let first_derivative = self.derivatives.len();
        self.names.extend(local[..terminals].iter().cloned());
        self.derivatives.extend(local.iter().flat_map(|row| {
            local
                .iter()
                .map(move |column| (row.clone(), column.clone()))
        }));
        let start = self.variables.len();
        self.variables.extend(
            copy.variable_names()
                .iter()
                .map(|&name| (device.path().to_owned(), name)),
        );
        self.probes.push(Probe {
            placed,
            device: copy,
            pattern,
            sources,
            terminals,
            first,
            first_derivative,
            variables: start..self.variables.len(),
            x: vec![0.0; count],
            residual: vec![0.0; count],
            scale: vec![0.0; count],
            charge: vec![0.0; count],
            conductance: vec![0.0; stored],
            capacitance: vec![0.0; stored],
        });
        Ok(())
    }

    /// Each terminal's name: its device's path and its own, such as `X1/R1.a`.
    pub(crate) fn names(&self) -> &[String] {
        &self.names
    }

    /// Each derivative's equation and unknown, by name.
    pub(crate) fn derivatives(&self) -> &[(String, String)] {
        &self.derivatives
    }

    /// Each variable, by its device's path and its own name.
    pub(crate) fn variables(&self) -> &[(String, &'static str)] {
        &self.variables
    }

    /// Each terminal of a component instance or a logic device, with the terminals whose
    /// currents and charges sum to its own.
    pub(crate) fn composites(&self) -> &[(String, Vec<usize>)] {
        &self.composites
    }

    /// Holds the copy of the source at `placed` at `level`, or releases it, as the source
    /// itself is.
    pub(crate) fn hold(&mut self, placed: usize, level: Option<f64>) {
        if let Some(source) = self.probes[placed].device.as_source() {
            source.hold(level);
        }
    }

    /// Starts the copy of the logic driver at `placed` moving to `level` at `time`, as the
    /// driver itself does.
    pub(crate) fn drive(&mut self, placed: usize, time: f64, level: Level, transition: f64) {
        if let Some(driver) = self.probes[placed].device.as_driver() {
            driver.drive(time, level, transition);
        }
    }

    /// What every device does at the DC solution `x`: each current without a charge's
    /// derivative.
    pub(crate) fn dc(&mut self, x: &[f64], gmin: f64) -> Observation {
        self.at(x, 0.0, gmin)
    }

    /// What every device does at the DC solution `x` that begins a transient, whose charges
    /// stand still.
    pub(crate) fn start(&mut self, x: &[f64], gmin: f64) -> Observation {
        let found = self.dc(x, gmin);
        self.history = History {
            last: found.charges.clone(),
            before: found.charges.clone(),
            derivative: vec![0.0; found.charges.len()],
            restarted: true,
        };
        found
    }

    /// What every device does at the transient's accepted point `x`, at `time`, solved by
    /// `formula`; `restart` when the integration begins afresh from it.
    pub(crate) fn step(
        &mut self,
        x: &[f64],
        time: f64,
        gmin: f64,
        (formula, restart): (&Formula, bool),
    ) -> Observation {
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

    /// What every device does at `x` at `time`, without a charge's derivative.
    fn at(&mut self, x: &[f64], time: f64, gmin: f64) -> Observation {
        let derivatives = self.derivatives.len();
        let mut found = Observation {
            currents: vec![0.0; self.names.len()],
            charges: vec![0.0; self.names.len()],
            conductances: vec![0.0; derivatives],
            capacitances: vec![0.0; derivatives],
            variables: vec![0.0; self.variables.len()],
        };
        for probe in &mut self.probes {
            probe.evaluate(x, time, gmin);
            let range = probe.first..probe.first + probe.terminals;
            found.currents[range.clone()].copy_from_slice(&probe.residual[..probe.terminals]);
            found.charges[range].copy_from_slice(&probe.charge[..probe.terminals]);
            let count = probe.x.len();
            let positions = probe
                .pattern
                .positions()
                .zip(probe.conductance.iter().zip(&probe.capacitance));
            for ((row, column), (&conductance, &capacitance)) in positions {
                let index = probe.first_derivative + row * count + column;
                found.conductances[index] = conductance;
                found.capacitances[index] = capacitance;
            }
            probe
                .device
                .variables(&mut found.variables[probe.variables.clone()]);
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
