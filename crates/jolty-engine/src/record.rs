//! Device recording capturing terminal currents, charges, equation derivatives, and
//! internal variables at each solution point.
//!
//! Individual device terminal currents are not circuit unknowns because the engine solves
//! for node voltages and select branch currents. Instead, each device writes
//! residuals, charges, conductances, and capacitances into a dedicated slab block over
//! unmerged local terminals and internal unknowns. Terminal current equals the local
//! terminal residual (plus `dq/dt` from the active integration formula in transient runs,
//! preserving KCL), while local matrix entries yield exact small-signal parameters.
//!
//! Analyses record only devices and summed component or logic terminals matched by
//! [`Record`], plus operating regions and boundary margins for unselected transistors when
//! [`regions`](Options::regions) is enabled.

use std::iter;
use std::ops::{Add, Range};

use jolty_models::{Groups, REGION, Runs};
use jolty_numerics::Complex64;

use crate::elaborated::Elaborated;
use crate::profile::Activity;
use crate::transient::StepFormula;
use crate::{Options, Placed, Record, SimulationError, Simulator};

/// Recorded device quantities at a single solution point, laid out according to
/// [`Recording`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RecordedValues {
    /// Current entering each recorded terminal in amperes: individual device terminals
    /// followed by summed component or logic terminals.
    pub currents: Vec<f64>,
    /// Charge stored at each recorded terminal in coulombs, matching `currents` order.
    pub charges: Vec<f64>,
    /// Row-major partial derivatives `∂f/∂x` of each recorded device's local equations with
    /// respect to its local unknowns (conductances for current-voltage pairs).
    pub conductances: Vec<f64>,
    /// Row-major partial derivatives `∂q/∂x` of each recorded device's local charges with
    /// respect to its local unknowns (capacitances for current-voltage pairs).
    pub capacitances: Vec<f64>,
    /// Recorded internal variable values across all recorded devices and region-only
    /// transistors.
    pub variables: Vec<f64>,
}

/// Recorded device quantities across all points of a DC sweep or transient analysis.
///
/// Rows for consecutive points are stored contiguously in a single allocation per quantity;
/// [`RecordedSeries::point`] borrows one point's slice view.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RecordedSeries {
    /// Number of recorded points.
    pub points: usize,
    /// Concatenated [`RecordedValues::currents`] across all points.
    pub currents: Vec<f64>,
    /// Concatenated [`RecordedValues::charges`] across all points.
    pub charges: Vec<f64>,
    /// Concatenated [`RecordedValues::conductances`] across all points.
    pub conductances: Vec<f64>,
    /// Concatenated [`RecordedValues::capacitances`] across all points.
    pub capacitances: Vec<f64>,
    /// Concatenated [`RecordedValues::variables`] across all points.
    pub variables: Vec<f64>,
}

/// Borrowed slice view of one solution point inside a [`RecordedSeries`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecordedView<'a> {
    /// Terminal currents matching [`RecordedValues::currents`].
    pub currents: &'a [f64],
    /// Terminal charges matching [`RecordedValues::charges`].
    pub charges: &'a [f64],
    /// Residual derivatives matching [`RecordedValues::conductances`].
    pub conductances: &'a [f64],
    /// Charge derivatives matching [`RecordedValues::capacitances`].
    pub capacitances: &'a [f64],
    /// Device variables matching [`RecordedValues::variables`].
    pub variables: &'a [f64],
}

impl RecordedSeries {
    /// Appends one recorded solution point to the series.
    pub fn push(&mut self, point: &RecordedValues) {
        self.points += 1;
        self.currents.extend_from_slice(&point.currents);
        self.charges.extend_from_slice(&point.charges);
        self.conductances.extend_from_slice(&point.conductances);
        self.capacitances.extend_from_slice(&point.capacitances);
        self.variables.extend_from_slice(&point.variables);
    }

    /// Returns a borrowed [`RecordedView`] for solution point `index`.
    ///
    /// # Panics
    ///
    /// Panics when `index` is greater than or equal to [`RecordedSeries::points`].
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

    /// Returns an iterator yielding a [`RecordedView`] for each recorded point in order.
    pub fn iter(&self) -> impl Iterator<Item = RecordedView<'_>> {
        (0..self.points).map(|index| self.point(index))
    }
}

/// Layout map describing which devices and summed terminals a [`Record`] selection stores
/// in each [`RecordedValues`].
///
/// Current and charge buffers list each recorded device's terminals in order, followed by
/// recorded summed terminals. Conductance and capacitance buffers store row-major local
/// equation derivatives for each recorded device over its unmerged terminals and
/// [internal unknowns](crate::Placed::internals). Variable buffers list full device
/// [variables](crate::Placed::variable_names) followed by
/// [region variables](Simulator::region_variables) for region-only transistors.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Recording {
    /// Device indices in [`Simulator::devices`] recorded in full (terminals, derivatives,
    /// and variables).
    pub devices: Vec<usize>,
    /// Summed terminal indices in [`Simulator::summed_terminals`] recorded for component
    /// instances and logic devices.
    pub summed_terminals: Vec<usize>,
    /// Device indices in [`Simulator::devices`] recorded for operating region and boundary
    /// margins only (transistors excluded from `devices` when [`regions`](Options::regions)
    /// is enabled).
    pub regions: Vec<usize>,
}

impl Recording {
    /// Builds a [`Recording`] layout from `record` and `regions` for `system`.
    fn from_record(system: &Elaborated, record: &Record, regions: bool) -> Self {
        let devices = &system.devices;
        let recorded = |index: usize| record.names(devices[index].path());
        Self {
            devices: (0..devices.len())
                .filter(|&index| recorded(index))
                .collect(),
            summed_terminals: (0..system.summed_terminals.len())
                .filter(|&index| record.names(&system.summed_terminals[index].owner))
                .collect(),
            regions: (0..devices.len())
                .filter(|&index| {
                    regions && !recorded(index) && !devices[index].regions().names.is_empty()
                })
                .collect(),
        }
    }
}

/// Returns indices into [`Placed::variable_names`] for the operating region and region
/// boundary margins of `device`.
fn region_variables(device: &Placed) -> Vec<usize> {
    let names = device.variable_names();
    iter::once(REGION)
        .chain(device.regions().margins.iter().copied())
        .filter_map(|name| names.iter().position(|&each| each == name))
        .collect()
}

impl Simulator {
    /// Computes the [`Recording`] layout selected by `record` and `regions`, matching the
    /// element ordering in [`RecordedValues`].
    #[must_use]
    pub fn recorded(&self, record: &Record, regions: bool) -> Recording {
        Recording::from_record(&self.elaborated, record, regions)
    }

    /// Returns indices into [`Placed::variable_names`] for the region and margin variables
    /// of the device at `index`, or an empty vector for non-transistors or out-of-range
    /// indices.
    #[must_use]
    pub fn region_variables(&self, index: usize) -> Vec<usize> {
        self.elaborated
            .devices
            .get(index)
            .map_or_else(Vec::new, region_variables)
    }

    /// Validates `options`, configures the solver ordering and parallelism thread count,
    /// and updates the recorder selection.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidOptions`] for invalid options, or
    /// [`SimulationError::TooLarge`] when a dissected ordering exceeds memory limits.
    pub(crate) fn prepare(&mut self, options: &Options) -> Result<(), SimulationError> {
        let previous = self.state.workspace.meter.enter(Activity::Preparing);
        let checked = options.check().and_then(|()| {
            let parallelism = &options.parallelism;
            let dissecting = parallelism.dissecting(self.elaborated.largest_block());
            let ordering = self.elaborated.ordering(dissecting)?;
            self.state
                .solver
                .order(&ordering, parallelism.factorization(&ordering));
            Ok(())
        });
        if checked.is_ok() {
            self.state
                .recorder
                .select(&self.elaborated, &options.record, options.regions);
        }
        self.state.workspace.meter.leave(previous);
        checked
    }

    /// Evaluates devices without voltage limiting at DC solution `x` (when recording is
    /// active) and returns the [`RecordedValues`].
    pub(crate) fn record_dc(&mut self, x: &[f64], options: &Options) -> RecordedValues {
        if self.state.recorder.records() {
            self.evaluate(x, None, 1.0, options.gmin, false);
        }
        self.record_evaluated()
    }

    /// Extracts DC [`RecordedValues`] from the most recent unlimited device evaluation.
    pub(crate) fn record_evaluated(&mut self) -> RecordedValues {
        let read = self.read_recorded();
        self.state.recorder.dc(&read)
    }

    /// Reads raw slab values for all recorded devices from the most recent evaluation.
    pub(crate) fn read_recorded(&mut self) -> Read {
        // The check compares what is read with each device's instance evaluated alone.
        #[cfg(test)]
        if let Some(check) = &mut self.state.check {
            let devices = self.state.recorder.selection.evaluated.iter();
            let devices = devices.map(|&(device, _, _)| device);
            check.compare_read(
                &self.elaborated,
                &self.state.runs,
                self.state.past.as_ref(),
                devices,
            );
        }
        self.state
            .recorder
            .read(&self.elaborated.groups, &self.state.runs)
    }
}

/// Output buffer offsets for a device recorded in full.
#[derive(Debug)]
struct Slot {
    /// Index of the device in the elaborated device list.
    device: usize,
    /// Index range of the device's terminals within the evaluated terminal buffer.
    terminals: Range<usize>,
    /// Local unknown count: terminals followed by internal unknowns.
    unknowns: usize,
    /// Starting offset of this device's dense local derivatives in the derivative buffer.
    first_derivative: usize,
    /// Index range of this device's variables in the recorded variable buffer.
    variables: Range<usize>,
}

/// Output buffer offsets for a transistor recorded for operating region and margins only.
#[derive(Debug)]
struct RegionSlot {
    /// Index of the transistor in the elaborated device list.
    device: usize,
    /// Total number of variables computed by the transistor model.
    computed: usize,
    /// Indices of region and margin variables within the transistor's computed variables.
    picked: Vec<usize>,
    /// Index range of region variables within the recorded variable buffer.
    variables: Range<usize>,
}

/// Precomputed index tables mapping slab entries to recorded output buffers.
#[derive(Debug, Default)]
struct Selection {
    /// Active [`Record`] filter used to build this selection.
    record: Record,
    /// Flag indicating whether unselected transistors record regions and margins.
    regions: bool,
    /// Devices read at each point (recorded directly or contributing to a summed terminal)
    /// as `(device_index, first_terminal_offset, terminal_count)`.
    evaluated: Vec<(usize, usize, usize)>,
    /// Total number of evaluated device terminals tracked across `evaluated`.
    tracked: usize,
    /// Full recording slots in output order.
    recorded: Vec<Slot>,
    /// Region-only transistor slots in output order.
    transistors: Vec<RegionSlot>,
    /// Member indices in the tracked terminal buffer for each recorded summed terminal.
    sums: Vec<Vec<usize>>,
    /// Total output terminal count (recorded device terminals plus summed terminals).
    terminals: usize,
    /// Total scalar entries in conductance and capacitance output buffers.
    derivatives: usize,
    /// Total scalar entries in the variable output buffer.
    variables: usize,
}

impl Selection {
    /// Gathers per-terminal values from `tracked` into output order (recorded device
    /// terminals followed by summed terminal totals).
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

/// Raw device slab snapshot from the latest evaluation before adding transient `dq/dt`
/// currents, laid out to populate [`RecordedValues`].
#[derive(Debug, Clone, Default)]
pub(crate) struct Read {
    currents: Vec<f64>,
    charges: Vec<f64>,
    conductances: Vec<f64>,
    capacitances: Vec<f64>,
    variables: Vec<f64>,
}

/// Per-terminal charge history maintained across accepted transient steps to compute
/// terminal `dq/dt` currents.
#[derive(Debug, Default)]
struct History {
    last: Vec<f64>,
    before: Vec<f64>,
    derivative: Vec<f64>,
    /// Flag indicating that the last accepted point restarted integration without prior
    /// step history.
    restarted: bool,
}

/// Stateful recorder managing device selection and transient terminal charge history.
#[derive(Debug, Default)]
pub(crate) struct Recorder {
    selection: Selection,
    history: History,
}

impl Recorder {
    /// Creates an empty [`Recorder`] selecting no devices.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Clears the active selection and transient history, restoring [`Recorder::new`] state.
    pub(crate) fn reset(&mut self) {
        self.selection = Selection::default();
        self.history = History::default();
    }

    /// Returns whether the recorder currently tracks any devices or transistor regions.
    pub(crate) fn records(&self) -> bool {
        self.selection.tracked > 0 || !self.selection.transistors.is_empty()
    }

    /// Updates the recording selection for `system` from `record` and `regions`.
    fn select(&mut self, system: &Elaborated, record: &Record, regions: bool) {
        if self.selection.record == *record && self.selection.regions == regions {
            return;
        }
        let recorded = Recording::from_record(system, record, regions);
        let mut kept = vec![false; system.devices.len()];
        for &device in &recorded.devices {
            kept[device] = true;
        }
        let mut needed = kept.clone();
        for &summed_terminal in &recorded.summed_terminals {
            for &(device, _) in &system.summed_terminals[summed_terminal].members {
                needed[device] = true;
            }
        }
        let mut firsts = vec![0; system.devices.len()];
        let mut evaluated = Vec::new();
        let mut slots = Vec::with_capacity(recorded.devices.len());
        let (mut tracked, mut derivatives, mut variables) = (0, 0, 0);
        let mut terminals = recorded.summed_terminals.len();
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
        let transistors = recorded
            .regions
            .iter()
            .map(|&device| {
                let picked = region_variables(&system.devices[device]);
                let start = variables;
                variables += picked.len();
                RegionSlot {
                    device,
                    computed: system.devices[device].variable_names().len(),
                    picked,
                    variables: start..variables,
                }
            })
            .collect();
        let sums = recorded
            .summed_terminals
            .iter()
            .map(|&summed_terminal| {
                system.summed_terminals[summed_terminal]
                    .members
                    .iter()
                    .map(|&(device, terminal)| firsts[device] + terminal)
                    .collect()
            })
            .collect();
        self.selection = Selection {
            record: record.clone(),
            regions,
            evaluated,
            tracked,
            recorded: slots,
            transistors,
            sums,
            terminals,
            derivatives,
            variables,
        };
    }

    /// Returns the total scalar count stored per recorded point across currents, charges,
    /// derivatives, and variables.
    pub(crate) fn width(&self) -> usize {
        let selection = &self.selection;
        2 * (selection.terminals + selection.derivatives) + selection.variables
    }

    /// Reads raw currents, charges, derivatives, and variables for all selected devices
    /// from the latest slab evaluation of `runs`.
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
        let widest = selection.transistors.iter().map(|slot| slot.computed).max();
        let mut computed = vec![0.0; widest.unwrap_or(0)];
        for slot in &selection.transistors {
            let computed = &mut computed[..slot.computed];
            runs.variables(groups, slot.device, computed);
            for (kept, &picked) in read.variables[slot.variables.clone()]
                .iter_mut()
                .zip(&slot.picked)
            {
                *kept = computed[picked];
            }
        }
        read
    }

    /// Assembles DC [`RecordedValues`] from `read` without charge time derivatives.
    pub(crate) fn dc(&self, read: &Read) -> RecordedValues {
        self.recorded(read, &read.currents)
    }

    /// Initializes transient terminal charge history from `read` at the initial DC point
    /// and returns the initial [`RecordedValues`].
    pub(crate) fn start(&mut self, read: &Read) -> RecordedValues {
        let found = self.recorded(read, &read.currents);
        self.history = History {
            last: read.charges.clone(),
            before: read.charges.clone(),
            derivative: vec![0.0; read.charges.len()],
            restarted: true,
        };
        found
    }

    /// Advances transient terminal charge history using `step` and returns the accepted
    /// point's [`RecordedValues`] with `dq/dt` included in terminal currents.
    pub(crate) fn step(&mut self, read: &Read, step: StepFormula) -> RecordedValues {
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
                step.formula.alpha * charge
                    + step
                        .formula
                        .history([history.last[index], before], history.derivative[index])
            })
            .collect();
        let currents: Vec<f64> = read
            .currents
            .iter()
            .zip(&derivative)
            .map(|(current, change)| current + change)
            .collect();
        let found = self.recorded(read, &currents);
        let history = &mut self.history;
        history.before = std::mem::replace(&mut history.last, read.charges.clone());
        history.derivative = derivative;
        history.restarted = step.restarted;
        found
    }

    /// Gathers terminal `currents` and `read` quantities into a [`RecordedValues`] buffer.
    fn recorded(&self, read: &Read, currents: &[f64]) -> RecordedValues {
        let selection = &self.selection;
        RecordedValues {
            currents: selection.gather(currents),
            charges: selection.gather(&read.charges),
            conductances: read.conductances.clone(),
            capacitances: read.capacitances.clone(),
            variables: read.variables.clone(),
        }
    }

    /// Builds a thread-safe [`Currents`] evaluator for small-signal AC terminal currents
    /// around the last operating point with unit excitation on device `excited`.
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

/// Linearized terminal rows extracted from one evaluated device block for AC current
/// calculation.
struct Linear {
    first: usize,
    entries: Vec<(usize, usize, f64, f64)>,
    nodes: Vec<u32>,
}

/// Linearized small-signal terminal-current evaluator for recorded devices at any AC
/// frequency.
pub(crate) struct Currents<'a> {
    linear: Vec<Linear>,
    /// Direct source excitation terms `(tracked_terminal, value)` for the excited source.
    excitation: Vec<(usize, f64)>,
    selection: &'a Selection,
}

impl Currents<'_> {
    /// Computes complex small-signal currents entering each recorded terminal at angular
    /// frequency `omega` for solution phasors `values`.
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
        // Apply direct excitation after the excited source's own conductance stamps.
        for &(terminal, value) in &self.excitation {
            tracked[terminal] += Complex64::new(value, 0.0);
        }
        self.selection.gather(&tracked)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use serde_json::json;

    use super::*;
    use crate::test_helpers::simulator_from_json;

    /// Builds a test simulator with a voltage source driving a resistor and MOSFET inside
    /// component instance `X1`.
    fn amplifier() -> Simulator {
        let resistor = json!({"name": "R1", "of": {"model": "resistor"},
                              "parameters": {"resistance": 1e3},
                              "terminals": {"a": "vdd", "b": "d"}});
        let mosfet = json!({"name": "M1", "of": {"model": "mosfet"},
                            "parameters": {"width": 2e-6, "length": 1e-6},
                            "terminals": {"drain": "d", "gate": "vdd", "source": "gnd",
                                          "bulk": "gnd"}});
        let document = json!({
            "format": "jolty-circuit", "version": 5, "top": "test",
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
        simulator_from_json(&document.to_string())
    }

    /// Returns default [`Options`] configured with `record`.
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
        assert_eq!(simulator.state.recorder.width(), 0);
        assert!(!simulator.state.recorder.records(), "nothing is read");
        let everything = recording(Record::All);
        simulator.prepare(&everything).unwrap();
        // V1: 2 terminals and its current, 9 derivatives; R1: 2 terminals, 4 derivatives;
        // M1: 4 terminals, 16 derivatives and 8 variables; X1: 2 ports.
        assert_eq!(
            simulator.state.recorder.width(),
            2 * (8 + 2) + 2 * (9 + 4 + 16) + 8
        );
        let mosfet = recording(Record::Instances(BTreeSet::from(["X1/M1".to_owned()])));
        simulator.prepare(&mosfet).unwrap();
        assert_eq!(simulator.state.recorder.width(), 2 * 4 + 2 * 16 + 8);
        let ports = recording(Record::Instances(BTreeSet::from(["X1".to_owned()])));
        simulator.prepare(&ports).unwrap();
        assert_eq!(simulator.state.recorder.width(), 2 * 2);
        let width = f64::from(u32::try_from(unknowns + 4).unwrap());
        assert_eq!(simulator.point_width().to_bits(), width.to_bits());
    }

    #[test]
    fn devices_are_read_only_when_recorded_or_behind_a_recorded_port() {
        let mut simulator = amplifier();
        let read = |simulator: &Simulator| -> Vec<String> {
            let selection = &simulator.state.recorder.selection;
            selection
                .evaluated
                .iter()
                .map(|&(device, _, _)| simulator.elaborated.devices[device].path().to_owned())
                .collect()
        };
        let ports = recording(Record::Instances(BTreeSet::from(["X1".to_owned()])));
        let point = simulator.operating_point(&ports).unwrap();
        assert_eq!(point.recorded.currents.len(), 2);
        assert_eq!(read(&simulator), ["X1/R1", "X1/M1"]);
        let source = recording(Record::Instances(BTreeSet::from(["V1".to_owned()])));
        simulator.operating_point(&source).unwrap();
        assert_eq!(read(&simulator), ["V1"]);
    }
}
