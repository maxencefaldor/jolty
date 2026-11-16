//! What the recorded devices do at every solution: the current into each of their terminals
//! and the charge each holds, the derivatives of their equations with respect to their
//! unknowns, and the variables they compute on their way.
//!
//! The engine solves for node voltages and a few branch currents, so a device's own currents
//! are not among its unknowns. Each device's instance writes its contributions into a block
//! of its own of the run's slab: a residual and a charge for each of its local unknowns, its
//! terminals first, and a conductance and a capacitance for each pair of them it claims,
//! every terminal its own even where two share a net or one joins the reference. So the
//! current into a terminal is its block's residual there, alone, the charge the terminal
//! holds its block's charge, and the derivatives of the device's equations its block's
//! entries, over its terminals apart and its internal unknowns: the device's own
//! small-signal model, exactly, read from the evaluation that solved the circuit. In a
//! transient, the current adds the charge's derivative, taken by the very integration
//! formula the step was solved with, so the currents into a node sum to zero as the node's
//! equation does.
//!
//! An analysis records only what its options' [`Record`] names: a device's terminals,
//! derivatives and variables, and a component instance's or a logic device's terminals,
//! each summed from the device terminals behind it, which are read but not kept.

use std::ops::{Add, Range};

use jolty_models::{Groups, Runs};
use jolty_numerics::Complex64;
use jolty_numerics::integration::Formula;

use crate::elaborated::Elaborated;
use crate::profile::Activity;
use crate::{Options, Record, SimulationError, Simulator};

/// What the recorded devices did at one solution, laid out as [`Recording`] says.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RecordedValues {
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
/// quantity in one allocation; [`RecordedSeries::point`] views one point's rows.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RecordedSeries {
    /// The number of points.
    pub points: usize,
    /// Each point's [`RecordedValues::currents`], one after another.
    pub currents: Vec<f64>,
    /// Each point's [`RecordedValues::charges`], one after another.
    pub charges: Vec<f64>,
    /// Each point's [`RecordedValues::conductances`], one after another.
    pub conductances: Vec<f64>,
    /// Each point's [`RecordedValues::capacitances`], one after another.
    pub capacitances: Vec<f64>,
    /// Each point's [`RecordedValues::variables`], one after another.
    pub variables: Vec<f64>,
}

/// What the recorded devices did at one point of [`RecordedSeries`], as views of its rows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecordedView<'a> {
    /// The current into each recorded terminal, as [`RecordedValues::currents`].
    pub currents: &'a [f64],
    /// The charge each recorded terminal holds, as [`RecordedValues::charges`].
    pub charges: &'a [f64],
    /// The derivatives of the residuals, as [`RecordedValues::conductances`].
    pub conductances: &'a [f64],
    /// The derivatives of the charges, as [`RecordedValues::capacitances`].
    pub capacitances: &'a [f64],
    /// The value of each variable, as [`RecordedValues::variables`].
    pub variables: &'a [f64],
}

impl RecordedSeries {
    /// Appends what the recorded devices did at one more point.
    pub fn push(&mut self, point: &RecordedValues) {
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
    /// When `index` is not below [`RecordedSeries::points`].
    #[must_use]
    pub fn point(&self, index: usize) -> RecordedView<'_> {
        assert!(index < self.points, "no point {index} of {}", self.points);
        let points = self.points;
        let row = |values: &[f64]| -> Range<usize> {
            let width = values.len() / points;
            index * width..(index + 1) * width
        };
        RecordedView {
            currents: &self.currents[row(&self.currents)],
            charges: &self.charges[row(&self.charges)],
            conductances: &self.conductances[row(&self.conductances)],
            capacitances: &self.capacitances[row(&self.capacitances)],
            variables: &self.variables[row(&self.variables)],
        }
    }

    /// What the recorded devices did at each point, in order.
    pub fn iter(&self) -> impl Iterator<Item = RecordedView<'_>> {
        (0..self.points).map(|index| self.point(index))
    }
}

/// What a [`Record`] keeps of a circuit's devices, and where every [`RecordedValues`] of an
/// analysis run with it holds each value.
///
/// An observation's currents and charges hold each terminal of each recorded device, in
/// order, then each recorded composite terminal, the sum of the device terminals behind it.
/// Its conductances and capacitances hold, for each recorded device, the derivative of each
/// of its equations with respect to each of its unknowns, row by row: its terminals, each
/// on a node of its own, then its [internal unknowns](crate::Placed::internals). Its variables
/// hold each recorded device's [variables](crate::Placed::variable_names).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Recording {
    /// The devices whose terminals, derivatives and variables are recorded, by index among
    /// [`Simulator::devices`], in order.
    pub devices: Vec<usize>,
    /// The terminals of component instances and logic devices recorded, by index among
    /// [`Simulator::composites`], in order.
    pub composites: Vec<usize>,
}

impl Recording {
    /// What `record` keeps of `system`'s devices.
    fn of(system: &Elaborated, record: &Record) -> Self {
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
    /// What `record` keeps of the circuit's devices, and where every [`RecordedValues`] of an
    /// analysis run with it holds each value.
    #[must_use]
    pub fn recorded(&self, record: &Record) -> Recording {
        Recording::of(&self.elaborated, record)
    }

    /// Checks `options`, gives the solver the ordering and the threads their parallelism
    /// asks for, and readies the observer to record what they name.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidOptions`] for options out of range, and
    /// [`SimulationError::TooLarge`] when a dissected ordering does not fit in memory.
    pub(crate) fn prepare(&mut self, options: &Options) -> Result<(), SimulationError> {
        let previous = self.state.workspace.meter.enter(Activity::Preparing);
        let checked = options.check().and_then(|()| {
            let (interiors, threads) = options.parallelism.factorization();
            let ordering = self.elaborated.ordering(interiors)?;
            self.state.solver.order(&ordering, threads);
            Ok(())
        });
        if checked.is_ok() {
            self.state
                .observer
                .select(&self.elaborated, &options.record);
        }
        self.state.workspace.meter.leave(previous);
        checked
    }

    /// What the recorded devices do at the DC solution `x`, every device evaluated there
    /// first, its steps not limited, when the observer records any.
    pub(crate) fn observe_dc(&mut self, x: &[f64], options: &Options) -> RecordedValues {
        if self.state.observer.records() {
            self.evaluate(x, None, (1.0, options.gmin), false);
        }
        let read = self.read_recorded();
        self.state.observer.dc(&read)
    }

    /// What the recorded devices did at the last evaluation, as the slab holds it.
    pub(crate) fn read_recorded(&mut self) -> Read {
        // The check compares what is read with each device's instance evaluated alone.
        #[cfg(test)]
        if let Some(check) = &mut self.state.check {
            let devices = self.state.observer.selection.evaluated.iter();
            let devices = devices.map(|&(device, _, _)| device);
            check.compare_read(&self.elaborated, &self.state.runs, devices);
        }
        self.state
            .observer
            .read(&self.elaborated.groups, &self.state.runs)
    }
}

/// A recorded device, and where its values lie.
#[derive(Debug)]
struct Slot {
    /// The device, by its index among the circuit's.
    device: usize,
    /// Its terminals, by index among the evaluated devices' terminals.
    terminals: Range<usize>,
    /// Its local unknowns: its terminals, then its internal unknowns.
    unknowns: usize,
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
    /// The devices read at each point, those recorded and those behind a recorded composite
    /// terminal, in order: each one's index among the circuit's devices, the index of its
    /// first terminal among the read devices' terminals, and its number of terminals.
    evaluated: Vec<(usize, usize, usize)>,
    /// The number of the read devices' terminals.
    tracked: usize,
    /// The recorded devices, in order.
    recorded: Vec<Slot>,
    /// The terminals of each recorded composite terminal, by index among the read devices'
    /// terminals.
    sums: Vec<Vec<usize>>,
    /// The terminals an observation holds: the recorded devices' and the composites'.
    terminals: usize,
    /// The derivatives an observation holds.
    derivatives: usize,
    /// The variables an observation holds.
    variables: usize,
}

impl Selection {
    /// The values of the read terminals' `tracked` that an observation holds: the recorded
    /// devices' terminals, then each recorded composite's sum.
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

/// What the recorded devices did at the last evaluation, as their blocks of the slab hold
/// it: the current into each read terminal, without a charge's derivative, and the charge
/// each holds, and the recorded devices' derivatives and variables, laid out as an
/// [`RecordedValues`]'s.
#[derive(Debug, Clone, Default)]
pub(crate) struct Read {
    currents: Vec<f64>,
    charges: Vec<f64>,
    conductances: Vec<f64>,
    capacitances: Vec<f64>,
    variables: Vec<f64>,
}

/// The charges of every read terminal at the last accepted point of a transient, and their
/// derivatives, as the transient's own history keeps the nodes'.
#[derive(Debug, Default)]
struct History {
    last: Vec<f64>,
    before: Vec<f64>,
    derivative: Vec<f64>,
    /// Whether the last point began the integration afresh, so has no point before it.
    restarted: bool,
}

/// What the analyses record of the devices, and the history of the charges a transient
/// carries.
#[derive(Debug, Default)]
pub(crate) struct Observer {
    selection: Selection,
    history: History,
}

impl Observer {
    /// An observer that records nothing yet.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Forgets what was recorded, as [`Observer::new`] makes it.
    pub(crate) fn reset(&mut self) {
        self.selection = Selection::default();
        self.history = History::default();
    }

    /// Whether the observer reads any device.
    pub(crate) fn records(&self) -> bool {
        self.selection.tracked > 0
    }

    /// Records what `record` names of `system` from the next point on.
    fn select(&mut self, system: &Elaborated, record: &Record) {
        if self.selection.record == *record {
            return;
        }
        let recorded = Recording::of(system, record);
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
        let mut firsts = vec![0; system.devices.len()];
        let mut evaluated = Vec::new();
        let mut slots = Vec::with_capacity(recorded.devices.len());
        let (mut tracked, mut derivatives, mut variables) = (0, 0, 0);
        let mut terminals = recorded.composites.len();
        for (index, device) in system.devices.iter().enumerate() {
            if !needed[index] {
                continue;
            }
            let count = device.nodes().len();
            evaluated.push((index, tracked, count));
            firsts[index] = tracked;
            if kept[index] {
                let unknowns = count + device.internals().len();
                let named = device.variable_names().len();
                slots.push(Slot {
                    device: index,
                    terminals: tracked..tracked + count,
                    unknowns,
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
    }

    /// The numbers an observation holds: the currents and the charges of the recorded
    /// terminals, and the recorded devices' derivatives and variables.
    pub(crate) fn width(&self) -> usize {
        let selection = &self.selection;
        2 * (selection.terminals + selection.derivatives) + selection.variables
    }

    /// What the recorded devices did at the last evaluation of `runs`, the run of `groups`,
    /// as their blocks of its slab hold it.
    pub(crate) fn read(&self, groups: &Groups, runs: &Runs) -> Read {
        let selection = &self.selection;
        let tracked = selection.tracked;
        let mut read = Read {
            currents: vec![0.0; tracked],
            charges: vec![0.0; tracked],
            conductances: vec![0.0; selection.derivatives],
            capacitances: vec![0.0; selection.derivatives],
            variables: vec![0.0; selection.variables],
        };
        for &(device, first, terminals) in &selection.evaluated {
            if let Some(block) = runs.block(groups, device) {
                let span = first..first + terminals;
                read.currents[span.clone()].copy_from_slice(&block.residual[..terminals]);
                read.charges[span].copy_from_slice(&block.charge[..terminals]);
            }
        }
        for slot in &selection.recorded {
            if let Some(block) = runs.block(groups, slot.device) {
                let entries = block
                    .entries
                    .iter()
                    .zip(block.conductance.iter().zip(block.capacitance));
                for (&(row, column), (&conductance, &capacitance)) in entries {
                    let index = slot.first_derivative + row * slot.unknowns + column;
                    read.conductances[index] = conductance;
                    read.capacitances[index] = capacitance;
                }
            }
            runs.variables(
                groups,
                slot.device,
                &mut read.variables[slot.variables.clone()],
            );
        }
        read
    }

    /// What the recorded devices do at a DC solution, as `read` reads them there: each
    /// current without a charge's derivative.
    pub(crate) fn dc(&self, read: &Read) -> RecordedValues {
        self.observation(read, &read.currents)
    }

    /// What the recorded devices do at the DC solution that begins a transient, as `read`
    /// reads them there, whose charges stand still.
    pub(crate) fn start(&mut self, read: &Read) -> RecordedValues {
        let found = self.observation(read, &read.currents);
        self.history = History {
            last: read.charges.clone(),
            before: read.charges.clone(),
            derivative: vec![0.0; read.charges.len()],
            restarted: true,
        };
        found
    }

    /// What the recorded devices do at a point a transient accepted, as `read` reads them
    /// there, the point solved by `formula`; `restart` when the integration begins afresh
    /// from it.
    pub(crate) fn step(
        &mut self,
        read: &Read,
        (formula, restart): (&Formula, bool),
    ) -> RecordedValues {
        let history = &mut self.history;
        let derivative: Vec<f64> = read
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
        let currents: Vec<f64> = read
            .currents
            .iter()
            .zip(&derivative)
            .map(|(current, change)| current + change)
            .collect();
        let found = self.observation(read, &currents);
        let history = &mut self.history;
        history.before = std::mem::replace(&mut history.last, read.charges.clone());
        history.derivative = derivative;
        history.restarted = restart;
        found
    }

    /// What the recorded devices did, as `read` reads them, with the current into each
    /// read terminal `currents`.
    fn observation(&self, read: &Read, currents: &[f64]) -> RecordedValues {
        let selection = &self.selection;
        RecordedValues {
            currents: selection.gather(currents),
            charges: selection.gather(&read.charges),
            conductances: read.conductances.clone(),
            capacitances: read.capacitances.clone(),
            variables: read.variables.clone(),
        }
    }

    /// The small-signal currents into the recorded terminals about the DC solution `runs`,
    /// the run of `groups`, was last evaluated at, with the source at `excited` driven by a
    /// unit excitation: ready at any frequency, and read by any number of threads at once.
    pub(crate) fn currents(&self, groups: &Groups, runs: &Runs, excited: usize) -> Currents<'_> {
        let mut excitation = Vec::new();
        let mut linear = Vec::with_capacity(self.selection.evaluated.len());
        for &(device, first, terminals) in &self.selection.evaluated {
            let Some(block) = runs.block(groups, device) else {
                continue;
            };
            if device == excited
                && let Some(local) = runs.excitation(groups, device)
            {
                excitation.extend(
                    local
                        .iter()
                        .take(terminals)
                        .enumerate()
                        .map(|(terminal, &value)| (first + terminal, value)),
                );
            }
            let entries = block
                .entries
                .iter()
                .zip(block.conductance.iter().zip(block.capacitance));
            linear.push(Linear {
                first,
                entries: entries
                    .filter(|&(&(row, _), _)| row < terminals)
                    .map(|(&(row, column), (&conductance, &capacitance))| {
                        (row, column, conductance, capacitance)
                    })
                    .collect(),
                nodes: block.nodes.to_vec(),
            });
        }
        Currents {
            linear,
            excitation,
            selection: &self.selection,
        }
    }
}

/// The linearized equations of a read device's terminals: the index of its first terminal
/// among the read devices' terminals, each entry of a terminal's row with its column, its
/// conductance and its capacitance, and the circuit's unknown of each local unknown.
struct Linear {
    first: usize,
    entries: Vec<(usize, usize, f64, f64)>,
    nodes: Vec<u32>,
}

/// The small-signal currents into the recorded terminals about a DC solution, with a
/// source driven by a unit excitation, at any frequency.
pub(crate) struct Currents<'a> {
    linear: Vec<Linear>,
    /// What the excited source adds to each of its terminals, by index among the read
    /// devices' terminals.
    excitation: Vec<(usize, f64)>,
    selection: &'a Selection,
}

impl Currents<'_> {
    /// The small-signal current into each recorded terminal at angular frequency `omega`,
    /// for the phasors `values` of the unknowns.
    pub(crate) fn at(&self, values: &[Complex64], omega: f64) -> Vec<Complex64> {
        let mut tracked = vec![Complex64::default(); self.selection.tracked];
        for linear in &self.linear {
            let phasor = |column: usize| {
                linear
                    .nodes
                    .get(column)
                    .and_then(|&unknown| values.get(unknown as usize))
                    .copied()
                    .unwrap_or_default()
            };
            for &(row, column, conductance, capacitance) in &linear.entries {
                tracked[linear.first + row] +=
                    Complex64::new(conductance, omega * capacitance) * phasor(column);
            }
        }
        // Each of the excited source's terminals takes its excitation after its own
        // conductances, which only its own block adds to it.
        for &(terminal, value) in &self.excitation {
            tracked[terminal] += Complex64::new(value, 0.0);
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
        let resistor = json!({"name": "R1", "of": {"model": "resistor"},
                              "parameters": {"resistance": 1e3},
                              "terminals": {"a": "vdd", "b": "d"}});
        let mosfet = json!({"name": "M1", "of": {"model": "mosfet"},
                            "parameters": {"width": 2e-6, "length": 1e-6},
                            "terminals": {"drain": "d", "gate": "vdd", "source": "gnd",
                                          "bulk": "gnd"}});
        let document = json!({
            "format": "jolty-circuit", "version": 4, "top": "test",
            "components": {
                "test": {"ports": [{"name": "0", "role": "ground"}], "instances": [
                    {"name": "V1", "of": {"model": "voltage_source"},
                     "parameters": {"waveform": {"dc": 3.0}},
                     "terminals": {"positive": "in", "negative": "0"}},
                    {"name": "X1", "of": {"component": "stage"},
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
        let unknowns = simulator.elaborated.unknowns.len();
        simulator.prepare(&Options::default()).unwrap();
        assert_eq!(simulator.state.observer.width(), 0);
        assert!(!simulator.state.observer.records(), "nothing is read");
        let everything = recording(Record::All);
        simulator.prepare(&everything).unwrap();
        // V1: 2 terminals and its current, 9 derivatives; R1: 2 terminals, 4 derivatives;
        // M1: 4 terminals, 16 derivatives and 8 variables; X1: 2 ports.
        assert_eq!(
            simulator.state.observer.width(),
            2 * (8 + 2) + 2 * (9 + 4 + 16) + 8
        );
        let mosfet = recording(Record::Instances(BTreeSet::from(["X1/M1".to_owned()])));
        simulator.prepare(&mosfet).unwrap();
        assert_eq!(simulator.state.observer.width(), 2 * 4 + 2 * 16 + 8);
        let ports = recording(Record::Instances(BTreeSet::from(["X1".to_owned()])));
        simulator.prepare(&ports).unwrap();
        assert_eq!(simulator.state.observer.width(), 2 * 2);
        let width = f64::from(u32::try_from(unknowns + 4).unwrap());
        assert_eq!(simulator.point_width().to_bits(), width.to_bits());
    }

    #[test]
    fn devices_are_read_only_when_recorded_or_behind_a_recorded_port() {
        let mut simulator = amplifier();
        let read = |simulator: &Simulator| -> Vec<String> {
            let selection = &simulator.state.observer.selection;
            selection
                .evaluated
                .iter()
                .map(|&(device, _, _)| simulator.elaborated.devices[device].path().to_owned())
                .collect()
        };
        let ports = recording(Record::Instances(BTreeSet::from(["X1".to_owned()])));
        let point = simulator.operating_point(&ports).unwrap();
        assert_eq!(point.observation.currents.len(), 2);
        assert_eq!(read(&simulator), ["X1/R1", "X1/M1"]);
        let source = recording(Record::Instances(BTreeSet::from(["V1".to_owned()])));
        simulator.operating_point(&source).unwrap();
        assert_eq!(read(&simulator), ["V1"]);
    }
}
