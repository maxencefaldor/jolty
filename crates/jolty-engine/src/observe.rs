//! What the recorded devices do at every solution: the current into each of their terminals
//! and the charge each holds, the derivatives of their equations with respect to their
//! unknowns, and the variables they compute on their way.
//!
//! The engine solves for node voltages and a few branch currents, so a device's own currents
//! are not among its unknowns. To read them exactly, each device an analysis records is made
//! a second time with every terminal on a node of its own, and evaluated at the solution: the
//! residual of a terminal's own node is then the current into that terminal alone, even
//! where two of its terminals share a net or one joins the reference, and the charge of that
//! node the charge the terminal holds. In a transient, the current adds the charge's
//! derivative, taken by the very integration formula the step was solved with, so the
//! currents into a node sum to zero as the node's equation does. The same evaluation gives
//! the derivatives of the device's residuals and charges, its conductances and capacitances,
//! over its terminals apart and its internal unknowns: the device's own small-signal model,
//! exactly.
//!
//! An analysis records only what its options' [`Record`] names: a device's terminals,
//! derivatives and variables, and a component instance's or a logic device's terminals,
//! each summed from the device terminals behind it, which are evaluated but not kept. With
//! nothing named, no device is made again or evaluated a second time.

use std::ops::{Add, Range};

use jolty_models::{
    Binding, Contributions, Device, Input, Level, LogicDriver, LogicDriverParameters, NativeDevice,
    Node, ParameterError, Quantity, ThresholdDetector, Unknown, UnknownInfo,
};
use jolty_numerics::integration::Formula;
use jolty_numerics::{Complex64, Pattern, PatternBuilder};

use crate::elaborate::{Origin, System};
use crate::{Options, Record, SimulationError, Simulator};

/// What the recorded devices did at one solution, laid out as [`Recorded`] says.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Observation {
    /// The current into each recorded terminal, in amperes: each terminal of each recorded
    /// device, then each recorded terminal of a component instance or a logic device.
    pub currents: Vec<f64>,
    /// The charge each recorded terminal holds, in coulombs, in the same order.
    pub charges: Vec<f64>,
    /// For each recorded device, the derivative of each of its equations' residuals with
    /// respect to each of its unknowns, row by row: a conductance where the equation sums
    /// currents and the unknown is a voltage.
    pub conductances: Vec<f64>,
    /// The derivative of each equation's charge with respect to each unknown, in the same
    /// order: a capacitance where the equation sums currents and the unknown is a voltage.
    pub capacitances: Vec<f64>,
    /// The value of each variable of each recorded device, in order.
    pub variables: Vec<f64>,
}

/// What the recorded devices did at each point of a sweep or a transient.
///
/// Each quantity's rows lie end to end, one row per point, so a long analysis holds each
/// quantity in one allocation; [`Observations::point`] views one point's rows.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Observations {
    /// The number of points.
    pub points: usize,
    /// Each point's [`Observation::currents`], one after another.
    pub currents: Vec<f64>,
    /// Each point's [`Observation::charges`], one after another.
    pub charges: Vec<f64>,
    /// Each point's [`Observation::conductances`], one after another.
    pub conductances: Vec<f64>,
    /// Each point's [`Observation::capacitances`], one after another.
    pub capacitances: Vec<f64>,
    /// Each point's [`Observation::variables`], one after another.
    pub variables: Vec<f64>,
}

/// What the recorded devices did at one point of [`Observations`], as views of its rows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observed<'a> {
    /// The current into each recorded terminal, as [`Observation::currents`].
    pub currents: &'a [f64],
    /// The charge each recorded terminal holds, as [`Observation::charges`].
    pub charges: &'a [f64],
    /// The derivatives of the residuals, as [`Observation::conductances`].
    pub conductances: &'a [f64],
    /// The derivatives of the charges, as [`Observation::capacitances`].
    pub capacitances: &'a [f64],
    /// The value of each variable, as [`Observation::variables`].
    pub variables: &'a [f64],
}

impl Observations {
    /// Appends what the recorded devices did at one more point.
    pub fn push(&mut self, point: &Observation) {
        self.points += 1;
        self.currents.extend_from_slice(&point.currents);
        self.charges.extend_from_slice(&point.charges);
        self.conductances.extend_from_slice(&point.conductances);
        self.capacitances.extend_from_slice(&point.capacitances);
        self.variables.extend_from_slice(&point.variables);
    }

    /// What the recorded devices did at the point `index`.
    ///
    /// # Panics
    ///
    /// When `index` is not below [`Observations::points`].
    #[must_use]
    pub fn point(&self, index: usize) -> Observed<'_> {
        assert!(index < self.points, "no point {index} of {}", self.points);
        let points = self.points;
        let row = |values: &[f64]| -> Range<usize> {
            let width = values.len() / points;
            index * width..(index + 1) * width
        };
        Observed {
            currents: &self.currents[row(&self.currents)],
            charges: &self.charges[row(&self.charges)],
            conductances: &self.conductances[row(&self.conductances)],
            capacitances: &self.capacitances[row(&self.capacitances)],
            variables: &self.variables[row(&self.variables)],
        }
    }

    /// What the recorded devices did at each point, in order.
    pub fn iter(&self) -> impl Iterator<Item = Observed<'_>> {
        (0..self.points).map(|index| self.point(index))
    }
}

/// What a [`Record`] keeps of a circuit's devices, and where every [`Observation`] of an
/// analysis run with it holds each value.
///
/// An observation's currents and charges hold each terminal of each recorded device, in
/// order, then each recorded composite terminal, the sum of the device terminals behind it.
/// Its conductances and capacitances hold, for each recorded device, the derivative of each
/// of its equations with respect to each of its unknowns, row by row: its terminals, each
/// on a node of its own, then its [internal unknowns](crate::Placed::internals). Its variables
/// hold each recorded device's [variables](crate::Placed::variable_names).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Recorded {
    /// The devices whose terminals, derivatives and variables are recorded, by index among
    /// [`Simulator::devices`], in order.
    pub devices: Vec<usize>,
    /// The terminals of component instances and logic devices recorded, by index among
    /// [`Simulator::composites`], in order.
    pub composites: Vec<usize>,
}

impl Recorded {
    /// What `record` keeps of `system`'s devices.
    fn of(system: &System, record: &Record) -> Self {
        Self {
            devices: (0..system.devices.len())
                .filter(|&index| record.names(system.devices[index].path()))
                .collect(),
            composites: (0..system.composites.len())
                .filter(|&index| record.names(&system.composites[index].owner))
                .collect(),
        }
    }
}

impl Simulator {
    /// What `record` keeps of the circuit's devices, and where every [`Observation`] of an
    /// analysis run with it holds each value.
    #[must_use]
    pub fn recorded(&self, record: &Record) -> Recorded {
        Recorded::of(&self.system, record)
    }

    /// Checks `options`, and readies the devices made again to record what they name.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidOptions`] for options out of range, and
    /// [`SimulationError::Unrecordable`] when a device to record cannot be made again.
    pub(crate) fn prepare(&mut self, options: &Options) -> Result<(), SimulationError> {
        options.check()?;
        self.observer.select(&self.system, &options.record)
    }
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
    x: Vec<f64>,
    contributions: Contributions,
}

impl Probe {
    /// Makes the device at index `placed` of `system`'s devices again, each terminal on its
    /// own node.
    fn new(system: &System, placed: usize) -> Result<Self, ParameterError> {
        let device = &system.devices[placed];
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
            Origin::Model(made) => made.specification.instantiate(&mut binding)?,
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
        let pattern = builder.finish(count);
        Ok(Self {
            placed,
            device: copy,
            sources,
            terminals,
            x: vec![0.0; count],
            contributions: Contributions::new(&pattern),
            pattern,
        })
    }

    /// Evaluates the copy at the solution `x`, at `time` in a transient analysis or, with
    /// `None`, at an operating point or about one.
    fn evaluate(&mut self, x: &[f64], time: Option<f64>, gmin: f64) {
        for (value, source) in self.x.iter_mut().zip(&self.sources) {
            *value = source.map_or(0.0, |unknown| x[unknown.index()]);
        }
        let mut input = Input::new(&self.x, time, gmin);
        input.limiting = false;
        self.device
            .evaluate(&input, &mut self.contributions.output(&self.pattern));
    }
}

/// A recorded device: its copy, and where its values lie.
#[derive(Debug)]
struct Slot {
    /// Its copy, by index among the observer's.
    probe: usize,
    /// Its terminals, by index among the evaluated devices' terminals.
    terminals: Range<usize>,
    /// The index of its first derivative among an observation's.
    first_derivative: usize,
    /// Its variables' range among an observation's.
    variables: Range<usize>,
}

/// What the analyses record, and where each value comes from.
#[derive(Debug, Default)]
struct Selection {
    /// The record it was made for.
    record: Record,
    /// The devices evaluated at each point, those recorded and those behind a recorded
    /// composite terminal, in order: each one's copy, by index among the observer's, and
    /// the index of its first terminal among the evaluated devices' terminals.
    evaluated: Vec<(usize, usize)>,
    /// The number of the evaluated devices' terminals.
    tracked: usize,
    /// The recorded devices, in order.
    recorded: Vec<Slot>,
    /// The terminals of each recorded composite terminal, by index among the evaluated
    /// devices' terminals.
    sums: Vec<Vec<usize>>,
    /// The terminals an observation holds: the recorded devices' and the composites'.
    terminals: usize,
    /// The derivatives an observation holds.
    derivatives: usize,
    /// The variables an observation holds.
    variables: usize,
}

impl Selection {
    /// The values of the evaluated terminals' `tracked` that an observation holds: the
    /// recorded devices' terminals, then each recorded composite's sum.
    fn gather<T: Copy + Default + Add<Output = T>>(&self, tracked: &[T]) -> Vec<T> {
        let mut found = Vec::with_capacity(self.terminals);
        for slot in &self.recorded {
            found.extend_from_slice(&tracked[slot.terminals.clone()]);
        }
        found.extend(self.sums.iter().map(|members| {
            members
                .iter()
                .fold(T::default(), |sum, &member| sum + tracked[member])
        }));
        found
    }
}

/// The charges of every evaluated terminal at the last accepted point of a transient, and
/// their derivatives, as the transient's own history keeps the nodes'.
#[derive(Debug, Default)]
struct History {
    last: Vec<f64>,
    before: Vec<f64>,
    derivative: Vec<f64>,
    /// Whether the last point began the integration afresh, so has no point before it.
    restarted: bool,
}

/// The devices made again to be read, and what the analyses record of them.
#[derive(Debug)]
pub(crate) struct Observer {
    /// Each device made again, once a record first named it or a composite terminal it is
    /// behind; each copy is held and driven as its device is from then on.
    probes: Vec<Probe>,
    /// The index among `probes` of each device's copy, by the device's index, if made.
    made: Vec<Option<usize>>,
    selection: Selection,
    history: History,
}

impl Observer {
    /// An observer of `system`'s devices that records nothing yet.
    pub(crate) fn new(system: &System) -> Self {
        Self {
            probes: Vec::new(),
            made: vec![None; system.devices.len()],
            selection: Selection::default(),
            history: History::default(),
        }
    }

    /// Records what `record` names of `system` from the next point on, making the devices
    /// it reads again where they were not.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Unrecordable`] naming every device that cannot be made
    /// again.
    fn select(&mut self, system: &System, record: &Record) -> Result<(), SimulationError> {
        if self.selection.record == *record {
            return Ok(());
        }
        let recorded = Recorded::of(system, record);
        let mut kept = vec![false; system.devices.len()];
        for &device in &recorded.devices {
            kept[device] = true;
        }
        let mut needed = kept.clone();
        for &composite in &recorded.composites {
            for &(device, _) in &system.composites[composite].members {
                needed[device] = true;
            }
        }
        let mut problems = Vec::new();
        for placed in (0..system.devices.len()).filter(|&placed| needed[placed]) {
            if self.made[placed].is_some() {
                continue;
            }
            match Probe::new(system, placed) {
                Ok(probe) => {
                    self.made[placed] = Some(self.probes.len());
                    self.probes.push(probe);
                }
                Err(error) => problems.push(format!(
                    "{}: {}",
                    system.devices[placed].location(),
                    error.0
                )),
            }
        }
        if !problems.is_empty() {
            return Err(SimulationError::Unrecordable(problems));
        }
        let mut firsts = vec![0; system.devices.len()];
        let mut evaluated = Vec::new();
        let mut slots = Vec::with_capacity(recorded.devices.len());
        let (mut tracked, mut derivatives, mut variables) = (0, 0, 0);
        let mut terminals = recorded.composites.len();
        for (placed, made) in self.made.iter().enumerate() {
            let (true, Some(probe)) = (needed[placed], *made) else {
                continue;
            };
            let device = &system.devices[placed];
            let count = device.nodes().len();
            evaluated.push((probe, tracked));
            firsts[placed] = tracked;
            if kept[placed] {
                let unknowns = count + device.internals().len();
                let named = device.variable_names().len();
                slots.push(Slot {
                    probe,
                    terminals: tracked..tracked + count,
                    first_derivative: derivatives,
                    variables: variables..variables + named,
                });
                terminals += count;
                derivatives += unknowns * unknowns;
                variables += named;
            }
            tracked += count;
        }
        let sums = recorded
            .composites
            .iter()
            .map(|&composite| {
                system.composites[composite]
                    .members
                    .iter()
                    .map(|&(device, terminal)| firsts[device] + terminal)
                    .collect()
            })
            .collect();
        self.selection = Selection {
            record: record.clone(),
            evaluated,
            tracked,
            recorded: slots,
            sums,
            terminals,
            derivatives,
            variables,
        };
        Ok(())
    }

    /// The numbers an observation holds: the currents and the charges of the recorded
    /// terminals, and the recorded devices' derivatives and variables.
    pub(crate) fn width(&self) -> usize {
        let selection = &self.selection;
        2 * (selection.terminals + selection.derivatives) + selection.variables
    }

    /// Holds the copy of the source at `placed` at `level`, or releases it, as the source
    /// itself is.
    pub(crate) fn hold(&mut self, placed: usize, level: Option<f64>) {
        if let Some(probe) = self.made[placed]
            && let Some(source) = self.probes[probe].device.as_source()
        {
            source.hold(level);
        }
    }

    /// Starts the copy of the logic driver at `placed` moving to `level` at `time`, as the
    /// driver itself does.
    pub(crate) fn drive(&mut self, placed: usize, time: f64, level: Level, transition: f64) {
        if let Some(probe) = self.made[placed]
            && let Some(driver) = self.probes[probe].device.as_driver()
        {
            driver.drive(time, level, transition);
        }
    }

    /// What the recorded devices do at the DC solution `x`: each current without a charge's
    /// derivative.
    pub(crate) fn dc(&mut self, x: &[f64], gmin: f64) -> Observation {
        let (currents, charges) = self.evaluate(x, None, gmin);
        self.observation(&currents, &charges)
    }

    /// What the recorded devices do at the DC solution `x` that begins a transient, whose
    /// charges stand still.
    pub(crate) fn start(&mut self, x: &[f64], gmin: f64) -> Observation {
        let (currents, charges) = self.evaluate(x, Some(0.0), gmin);
        let found = self.observation(&currents, &charges);
        self.history = History {
            last: charges.clone(),
            before: charges.clone(),
            derivative: vec![0.0; charges.len()],
            restarted: true,
        };
        found
    }

    /// What the recorded devices do at the transient's accepted point `x`, at `time`,
    /// solved by `formula`; `restart` when the integration begins afresh from it.
    pub(crate) fn step(
        &mut self,
        x: &[f64],
        time: f64,
        gmin: f64,
        (formula, restart): (&Formula, bool),
    ) -> Observation {
        let (mut currents, charges) = self.evaluate(x, Some(time), gmin);
        let history = &mut self.history;
        let derivative: Vec<f64> = charges
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
        for (current, change) in currents.iter_mut().zip(&derivative) {
            *current += change;
        }
        let found = self.observation(&currents, &charges);
        let history = &mut self.history;
        history.before = std::mem::replace(&mut history.last, charges);
        history.derivative = derivative;
        history.restarted = restart;
        found
    }

    /// Evaluates every device the recording reads at `x`, at `time` in a transient analysis
    /// or, with `None`, at an operating point; returns the current into each of their
    /// terminals, without a charge's derivative, and the charge each holds.
    fn evaluate(&mut self, x: &[f64], time: Option<f64>, gmin: f64) -> (Vec<f64>, Vec<f64>) {
        let tracked = self.selection.tracked;
        let (mut currents, mut charges) = (vec![0.0; tracked], vec![0.0; tracked]);
        for &(index, first) in &self.selection.evaluated {
            let probe = &mut self.probes[index];
            probe.evaluate(x, time, gmin);
            let terminals = first..first + probe.terminals;
            currents[terminals.clone()]
                .copy_from_slice(&probe.contributions.residual[..probe.terminals]);
            charges[terminals].copy_from_slice(&probe.contributions.charge[..probe.terminals]);
        }
        (currents, charges)
    }

    /// What the recorded devices did at their last evaluation, given the current into each
    /// evaluated terminal and the charge it holds.
    fn observation(&self, currents: &[f64], charges: &[f64]) -> Observation {
        let selection = &self.selection;
        let mut found = Observation {
            currents: selection.gather(currents),
            charges: selection.gather(charges),
            conductances: vec![0.0; selection.derivatives],
            capacitances: vec![0.0; selection.derivatives],
            variables: vec![0.0; selection.variables],
        };
        for slot in &selection.recorded {
            let probe = &self.probes[slot.probe];
            let count = probe.x.len();
            let positions = probe.pattern.positions().zip(
                probe
                    .contributions
                    .conductance
                    .iter()
                    .zip(&probe.contributions.capacitance),
            );
            for ((row, column), (&conductance, &capacitance)) in positions {
                let index = slot.first_derivative + row * count + column;
                found.conductances[index] = conductance;
                found.capacitances[index] = capacitance;
            }
            probe
                .device
                .variables(&mut found.variables[slot.variables.clone()]);
        }
        found
    }

    /// Readies the small-signal currents around the DC solution `x`.
    pub(crate) fn linearize(&mut self, x: &[f64], gmin: f64) {
        for &(index, _) in &self.selection.evaluated {
            self.probes[index].evaluate(x, None, gmin);
        }
    }

    /// The small-signal current into each recorded terminal at angular frequency `omega`,
    /// for the phasors `values` of the unknowns, with the source at `excited` driven by a
    /// unit excitation, after [`Observer::linearize`].
    pub(crate) fn phasors(
        &mut self,
        values: &[Complex64],
        omega: f64,
        excited: usize,
    ) -> Vec<Complex64> {
        let mut tracked = vec![Complex64::default(); self.selection.tracked];
        for &(index, first) in &self.selection.evaluated {
            let probe = &mut self.probes[index];
            let phasor = |column: usize| {
                probe.sources[column]
                    .map_or(Complex64::default(), |unknown| values[unknown.index()])
            };
            let positions = probe.pattern.positions().zip(
                probe
                    .contributions
                    .conductance
                    .iter()
                    .zip(&probe.contributions.capacitance),
            );
            for ((row, column), (&conductance, &capacitance)) in positions {
                if row < probe.terminals {
                    tracked[first + row] +=
                        Complex64::new(conductance, omega * capacitance) * phasor(column);
                }
            }
            if probe.placed == excited
                && let Some(source) = probe.device.as_source()
            {
                let mut derivative = vec![0.0; probe.x.len()];
                source.level_derivative(&mut derivative);
                for (terminal, &value) in derivative[..probe.terminals].iter().enumerate() {
                    tracked[first + terminal] += Complex64::new(value, 0.0);
                }
            }
        }
        self.selection.gather(&tracked)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use jolty_ir::Circuit;
    use jolty_models::Library;
    use serde_json::json;

    use super::*;
    use crate::Conditions;

    /// A source driving a resistor into a MOSFET, inside a component instance `X1`.
    fn amplifier() -> Simulator {
        let resistor = json!({"id": "R1", "of": {"device": "resistor"},
                              "parameters": {"resistance": 1e3},
                              "terminals": {"a": "vdd", "b": "d"}});
        let mosfet = json!({"id": "M1", "of": {"device": "mosfet"},
                            "parameters": {"width": 2e-6, "length": 1e-6},
                            "terminals": {"drain": "d", "gate": "vdd", "source": "gnd",
                                          "bulk": "gnd"}});
        let document = json!({
            "format": "jolty-circuit", "version": 1, "top": "test",
            "components": {
                "test": {"ports": [{"name": "0", "role": "ground"}], "instances": [
                    {"id": "V1", "of": {"device": "voltage_source"},
                     "parameters": {"waveform": {"dc": 3.0}},
                     "terminals": {"positive": "in", "negative": "0"}},
                    {"id": "X1", "of": {"component": "stage"},
                     "terminals": {"vdd": "in", "gnd": "0"}}
                ]},
                "stage": {"ports": [{"name": "vdd", "role": "supply"},
                                    {"name": "gnd", "role": "ground"}],
                          "instances": [resistor, mosfet]}
            }
        });
        let circuit = Circuit::from_json(&document.to_string()).expect("a valid circuit");
        Simulator::new(&circuit, &Library::standard(), &Conditions::default())
            .expect("the circuit elaborates")
    }

    /// Options that record `record`.
    fn recording(record: Record) -> Options {
        Options {
            record,
            ..Options::default()
        }
    }

    #[test]
    fn a_point_holds_the_unknowns_and_only_what_is_recorded() {
        let mut simulator = amplifier();
        let unknowns = simulator.system.unknowns.len();
        simulator.prepare(&Options::default()).unwrap();
        assert_eq!(simulator.observer.width(), 0);
        assert!(
            simulator.observer.probes.is_empty(),
            "nothing is made again"
        );
        let everything = recording(Record::All);
        simulator.prepare(&everything).unwrap();
        // V1: 2 terminals and its current, 9 derivatives; R1: 2 terminals, 4 derivatives;
        // M1: 4 terminals, 16 derivatives and 5 variables; X1: 2 ports.
        assert_eq!(
            simulator.observer.width(),
            2 * (8 + 2) + 2 * (9 + 4 + 16) + 5
        );
        let mosfet = recording(Record::Instances(BTreeSet::from(["X1/M1".to_owned()])));
        simulator.prepare(&mosfet).unwrap();
        assert_eq!(simulator.observer.width(), 2 * 4 + 2 * 16 + 5);
        let ports = recording(Record::Instances(BTreeSet::from(["X1".to_owned()])));
        simulator.prepare(&ports).unwrap();
        assert_eq!(simulator.observer.width(), 2 * 2);
        let width = f64::from(u32::try_from(unknowns + 4).unwrap());
        assert_eq!(simulator.point_width().to_bits(), width.to_bits());
    }

    #[test]
    fn devices_are_made_again_only_when_recorded_or_behind_a_recorded_port() {
        let mut simulator = amplifier();
        let ports = recording(Record::Instances(BTreeSet::from(["X1".to_owned()])));
        let point = simulator.operating_point(&ports).unwrap();
        assert_eq!(point.observation.currents.len(), 2);
        let made: Vec<&str> = simulator
            .observer
            .probes
            .iter()
            .map(|probe| simulator.system.devices[probe.placed].path())
            .collect();
        assert_eq!(made, ["X1/R1", "X1/M1"]);
        let source = recording(Record::Instances(BTreeSet::from(["V1".to_owned()])));
        simulator.operating_point(&source).unwrap();
        assert_eq!(simulator.observer.probes.len(), 3);
        assert_eq!(simulator.observer.selection.evaluated.len(), 1);
    }
}
