//! A run's state: everything a run of an elaborated circuit writes.
//!
//! A [`State`] holds a run's devices, made for its values, the prepared numbers and the
//! memory of the instances of kernels among them and the slab they write, and every buffer
//! its analyses fill: the sparse solver's factors, Newton iteration's workspace, the
//! logic's events, the devices made again to be read, and the operating point last solved.
//! Resetting it for other values keeps every buffer, makes every device again, prepares
//! again the instances of kernels whose numbers or temperature change, starts every one
//! from its initial memory, and forgets everything a run leaves behind, so that a run never
//! depends on the run before it: a state reset equals a state made afresh for the same
//! values, bit for bit.

use std::collections::BTreeMap;
use std::sync::Arc;

use jolty_models::{Changed, Device, Runs, Tasks};

use crate::elaborate::ElaborationError;
use crate::elaborated::{Devices, Elaborated, Origin, Origins};
use crate::logic::Events;
use crate::newton::Workspace;
use crate::observe::Observer;
use crate::operating_point::Remembered;
use crate::solver::Solver;
use crate::{ParameterId, Values};

/// What a DC sweep holds away from the run's own values: an independent source, at a
/// level, or a number of a device's parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Held {
    /// The independent source at this index among the devices.
    Level(usize),
    /// This number of a device's parameters.
    Number(ParameterId),
}

/// Where a run of a chain starts its first operating point: the operating point the run
/// before it solved, and the values it solved it at.
#[derive(Debug, Clone)]
pub(crate) struct Warm {
    pub(crate) values: Values,
    pub(crate) solution: Vec<f64>,
}

/// What one run of an elaborated circuit writes.
#[derive(Debug)]
pub(crate) struct State {
    /// The devices' temperature, in kelvins.
    pub(crate) temperature: f64,
    /// The devices, made for the run, in the order of the circuit's. The run evaluates
    /// those of no kernel one by one; of the instances of kernels, which their groups
    /// evaluate, it reads what an analysis reads of a device apart from its group, such as
    /// its noise.
    pub(crate) devices: Vec<Box<dyn Device>>,
    /// The groups of the instances of kernels: their prepared numbers, their memory and
    /// the slab they write.
    pub(crate) runs: Runs,
    /// How the groups' evaluation splits among threads, or `None` to evaluate them on the
    /// calling thread alone, the same bits either way.
    pub(crate) tasks: Option<Tasks>,
    /// The comparison, at every evaluation, of the groups' assembly with the instances
    /// evaluated as devices, when a test asks for it.
    #[cfg(test)]
    pub(crate) check: Option<crate::kernels::Check>,
    /// What each device whose numbers the run's values replace is made from, by device,
    /// and `None` for the others, made from their own; empty when they replace none.
    pub(crate) patched: Vec<Option<Origin>>,
    /// What each device a DC sweep holds away from the run's values is made from, by the
    /// device's index.
    pub(crate) swept: BTreeMap<usize, Origin>,
    /// Where a reset makes the devices, before they take the place of the run's.
    spare: Vec<Box<dyn Device>>,
    pub(crate) solver: Solver<f64>,
    pub(crate) workspace: Workspace,
    /// The state of the logic nets, as the last analysis left it.
    pub(crate) events: Events,
    /// The devices made again, to read their terminals.
    pub(crate) observer: Observer,
    /// The last operating point solved, what it was solved under and the state it left,
    /// which an analysis about the same operating point takes instead of solving it again.
    /// Once the devices are made, only what a sweep holds and driving the logic change
    /// their DC equations: what is held is part of what it was solved under, and an
    /// operating point starts from the logic reset and leaves the logic and its drivers
    /// as the one remembered left them.
    pub(crate) remembered: Option<Remembered>,
    /// What a DC sweep holds away from the run's own values, at the level or the number
    /// each is held at.
    pub(crate) held: BTreeMap<Held, f64>,
    /// Where the run's first operating point starts, in a chain of runs: none for a run
    /// that starts from zero.
    pub(crate) warm: Option<Warm>,
    /// How many times the run has solved for an operating point, or a DC sweep's first
    /// level, rather than taken the last one's.
    pub(crate) solves: usize,
}

impl State {
    /// A state of `elaborated` made afresh, whose `devices` were made at `temperature`, the
    /// instances of kernels among them prepared, those whose numbers a run's values replace
    /// from `patched`.
    pub(crate) fn new(
        elaborated: &Elaborated,
        (devices, runs): Devices,
        patched: Vec<Option<Origin>>,
        temperature: f64,
    ) -> Self {
        Self {
            temperature,
            devices,
            runs,
            tasks: None,
            #[cfg(test)]
            check: None,
            patched,
            swept: BTreeMap::new(),
            spare: Vec::new(),
            solver: Solver::new(
                Arc::clone(&elaborated.ordering),
                Arc::clone(&elaborated.names),
            ),
            workspace: Workspace::new(elaborated),
            events: Events::new(&elaborated.logic),
            observer: Observer::new(elaborated),
            remembered: None,
            held: BTreeMap::new(),
            warm: None,
            solves: 0,
        }
    }

    /// A state of `elaborated` for `values`, its devices made through bindings that replay
    /// the unknowns and entries the circuit holds, or `None` when the values give some
    /// device others, so another structure.
    ///
    /// # Errors
    ///
    /// Returns every problem the values make, as [`Elaborated::structure_of`] does.
    pub(crate) fn of(
        elaborated: &Elaborated,
        values: &Values,
    ) -> Result<Option<Self>, ElaborationError> {
        let temperature = elaborated.temperature_of(values);
        let patched = elaborated.patch(values)?;
        let mut devices = Vec::with_capacity(elaborated.devices.len());
        if !made(elaborated, &patched, temperature, &mut devices)? {
            return Ok(None);
        }
        let mut runs = elaborated.groups.runs();
        let changed = |device| changes(elaborated, &patched, device, true);
        if !prepared(elaborated, (&mut runs, &patched), temperature, changed)? {
            return Ok(None);
        }
        runs.commit();
        Ok(Some(Self::new(
            elaborated,
            (devices, runs),
            patched,
            temperature,
        )))
    }

    /// Makes the run's devices again for `values`, prepares again the instances of kernels
    /// whose numbers or temperature they change, and forgets everything the runs before it
    /// left: the instances' memory and the slab, the operating point remembered, the
    /// sources held, the sparse solver's pivots and factors, the meter's counts, the logic's
    /// events and traces, and the devices made again to be read. Every buffer is kept.
    /// Returns whether the devices of `values` have the structure of `elaborated`; when they
    /// do not, or the values make a problem, the state is left as it was.
    ///
    /// # Errors
    ///
    /// Returns every problem the values make, as [`Elaborated::structure_of`] does.
    pub(crate) fn reset(
        &mut self,
        elaborated: &Elaborated,
        values: &Values,
    ) -> Result<bool, ElaborationError> {
        let temperature = elaborated.temperature_of(values);
        let patched = elaborated.patch(values)?;
        let mut devices = std::mem::take(&mut self.spare);
        let mut replayed = made(elaborated, &patched, temperature, &mut devices);
        if matches!(replayed, Ok(true)) {
            // An instance keeps its numbers unless they or its temperature change.
            let moved = temperature.to_bits() != self.temperature.to_bits();
            let old = &self.patched;
            let changed = |device| {
                let again = moved || old.get(device).is_some_and(Option::is_some);
                changes(elaborated, &patched, device, again)
            };
            replayed = prepared(elaborated, (&mut self.runs, &patched), temperature, changed);
        }
        let made_again = matches!(replayed, Ok(true));
        if made_again {
            std::mem::swap(&mut self.devices, &mut devices);
            self.patched = patched;
            self.runs.commit();
        }
        // The devices not taken, the run's last or those of other values, go.
        devices.clear();
        self.spare = devices;
        if !made_again {
            return replayed;
        }
        self.temperature = temperature;
        self.swept.clear();
        self.solver.reset();
        self.workspace.reset();
        self.events.reset(&elaborated.logic);
        self.observer.reset();
        self.remembered = None;
        self.held.clear();
        self.warm = None;
        self.solves = 0;
        Ok(true)
    }

    /// Forgets what a run leaves but its devices, the sparse solver's pivots and the
    /// meter's counts: the operating point remembered, what a sweep holds, the logic's
    /// events and traces, and the devices made again to be read.
    pub(crate) fn forget(&mut self, elaborated: &Elaborated) {
        self.swept.clear();
        self.events.reset(&elaborated.logic);
        self.observer.reset();
        self.remembered = None;
        self.held.clear();
        self.warm = None;
    }

    /// Makes again, for `to`, the devices whose numbers or temperature differ from those
    /// of `from`, the values the run's devices are made for, and the copies of them the
    /// observer reads; leaves every other device as it is, its memory among it. Returns
    /// whether each of them claims the unknowns and entries of `elaborated`; when one does
    /// not, or the values make a problem, the state is left as it was.
    ///
    /// # Errors
    ///
    /// Returns every problem the values make for the devices made again.
    pub(crate) fn move_to(
        &mut self,
        elaborated: &Elaborated,
        from: &Values,
        to: &Values,
    ) -> Result<bool, ElaborationError> {
        let temperature = elaborated.temperature_of(to);
        let every = temperature.to_bits() != self.temperature.to_bits();
        let mut problems = Vec::new();
        let mut made = Vec::new();
        for index in 0..elaborated.devices.len() {
            let patches = elaborated.patches_of(to, index);
            if !every && same(patches, elaborated.patches_of(from, index)) {
                continue;
            }
            let origin = if patches.is_empty() {
                None
            } else {
                match elaborated.patched(index, patches) {
                    Ok(origin) => Some(origin),
                    Err(problem) => {
                        problems.push(*problem);
                        continue;
                    }
                }
            };
            let template = elaborated.devices[index].origin();
            match elaborated.replay(index, origin.as_ref().unwrap_or(template), temperature) {
                Ok(Some(device)) => made.push((index, origin, device)),
                Ok(None) => return Ok(false),
                Err(error) => problems.push(elaborated.problem(index, error)),
            }
        }
        if !problems.is_empty() {
            return Err(ElaborationError::new(problems));
        }
        self.temperature = temperature;
        if self.patched.is_empty() && made.iter().any(|(_, origin, _)| origin.is_some()) {
            self.patched.resize_with(elaborated.devices.len(), || None);
        }
        for (index, origin, device) in made {
            self.devices[index] = device;
            if let Some(patched) = self.patched.get_mut(index) {
                *patched = origin;
            }
            let origins = Origins {
                elaborated,
                patched: &self.patched,
                swept: &self.swept,
                temperature,
            };
            self.observer.remake(origins, index);
        }
        Ok(true)
    }
}

/// Whether `a` and `b` replace the same numbers with the same bits.
fn same(a: &[(ParameterId, f64)], b: &[(ParameterId, f64)]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|((p, x), (q, y))| p == q && x.to_bits() == y.to_bits())
}

/// What has changed of the device at `device` of `elaborated` for a run whose values replace
/// the numbers `patched` gives: its parameters, when the values replace them or the
/// device's group was made of others, else its numbers when they are `again` prepared, and
/// otherwise nothing.
fn changes(
    elaborated: &Elaborated,
    patched: &[Option<Origin>],
    device: usize,
    again: bool,
) -> Changed {
    let replaced = patched.get(device).is_some_and(Option::is_some);
    let made_otherwise = elaborated
        .devices
        .get(device)
        .is_some_and(|placed| !placed.nominal());
    if replaced || made_otherwise {
        Changed::Parameters
    } else if again {
        Changed::Numbers
    } else {
        Changed::Nothing
    }
}

/// Prepares, in `runs`, each instance of a kernel of `elaborated` whose device `changed`
/// says has changed,
/// for a run at `temperature`, from its origin of `patched` or its own, keeping the numbers
/// apart until `runs` commits them. Returns whether each has its group's kernel and shape.
///
/// # Errors
///
/// Returns the problem of each device whose parameters are out of range.
fn prepared(
    elaborated: &Elaborated,
    (runs, patched): (&mut Runs, &[Option<Origin>]),
    temperature: f64,
    changed: impl Fn(usize) -> Changed,
) -> Result<bool, ElaborationError> {
    let origins = Origins {
        elaborated,
        patched,
        temperature,
    };
    runs.prepare(
        &elaborated.groups,
        |device| origins.of(device).kernel(),
        temperature,
        changed,
    )
    .map_err(|problems| {
        let problems = problems.into_iter();
        ElaborationError::new(
            problems
                .map(|(device, error)| elaborated.problem(device, error))
                .collect(),
        )
    })
}

/// Makes every device of `elaborated` for a run at `temperature`, each from its origin of
/// `patched` or its own, through a binding that replays its claims, into `devices`.
/// Returns whether every one replayed them; it stops at the first that does not.
///
/// # Errors
///
/// Returns the problem of each device whose parameters are out of range.
fn made(
    elaborated: &Elaborated,
    patched: &[Option<Origin>],
    temperature: f64,
    devices: &mut Vec<Box<dyn Device>>,
) -> Result<bool, ElaborationError> {
    let swept = BTreeMap::new();
    let origins = Origins {
        elaborated,
        patched,
        swept: &swept,
        temperature,
    };
    let mut problems = Vec::new();
    for index in 0..elaborated.devices.len() {
        match elaborated.replay(index, origins.of(index), temperature) {
            Ok(Some(device)) => devices.push(device),
            Ok(None) => return Ok(false),
            Err(error) => problems.push(elaborated.problem(index, error)),
        }
    }
    if problems.is_empty() {
        Ok(true)
    } else {
        Err(ElaborationError::new(problems))
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use jolty_ir::Circuit;
    use jolty_models::Library;
    use serde_json::json;

    use crate::{Analysis, Elaborated, Options, Record, Simulator, Structure, Values};

    /// The circuit of the benchmark `bjt_amp`, whose transistors' junctions remember their
    /// last voltages, and a ring of a logic inverter, a resistor and a capacitor.
    fn circuits() -> [Circuit; 2] {
        let perf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../perf");
        let text = std::fs::read_to_string(perf.join("circuits/bjt_amp.json")).unwrap();
        let amplifier = Circuit::from_json(&text).unwrap();
        let device = |id: &str, model: &str, parameters, terminals| {
            json!({"id": id, "of": {"device": model}, "parameters": parameters,
                   "terminals": terminals})
        };
        let ring = json!({
            "format": "jolty-circuit", "version": 1, "top": "ring",
            "components": {"ring": {"ports": [{"name": "0", "role": "ground"}], "instances": [
                device("VDD", "voltage_source", json!({"waveform": {"dc": 1.0}}),
                       json!({"positive": "vdd", "negative": "0"})),
                device("U1", "logic_cell", json!({
                        "inputs": ["a"], "outputs": [{"name": "y", "table": "10"}],
                        "timing": {"fixed": {"rise": 1e-6, "fall": 1e-6, "transition": 1e-12}},
                        "electrical": {"input_capacitances": [0.0]}}),
                       json!({"a": "c", "y": "out", "supply": "vdd", "ground": "0"})),
                device("R1", "resistor", json!({"resistance": 1e3}),
                       json!({"a": "out", "b": "c"})),
                device("C1", "capacitor", json!({"capacitance": 1e-9}),
                       json!({"a": "c", "b": "0"})),
                device("V1", "voltage_source",
                       json!({"waveform": {"piecewise": {"points": [[0, 0], [1e-9, 1]]}}}),
                       json!({"positive": "kick", "negative": "0"})),
                device("R2", "resistor", json!({"resistance": 1e9}),
                       json!({"a": "kick", "b": "c"}))
            ]}}
        });
        [amplifier, Circuit::from_json(&ring.to_string()).unwrap()]
    }

    /// A state reset for some values is the state made afresh for them: its devices and
    /// their memory, the sparse solver's counts, the workspace and its meter, the logic's
    /// events, the devices made again to be read, the operating point remembered and the
    /// sources held, every one as a fresh state has it.
    #[test]
    fn a_state_reset_is_the_state_made_afresh() {
        let analyses: Vec<Analysis> = [
            json!({"transient": {"stop": 2e-6}}),
            json!("operating_point"),
            json!({"dc_sweep": {"source": "V1", "levels": [0.0, 0.5]}}),
        ]
        .into_iter()
        .map(|analysis| serde_json::from_value(analysis).unwrap())
        .collect();
        let options = Options {
            record: Record::All,
            ..Options::default()
        };
        for circuit in circuits() {
            let elaborated =
                Elaborated::new(&circuit, &Library::standard(), &Structure::default()).unwrap();
            let fresh =
                Simulator::from_elaborated(Arc::clone(&elaborated), Values::at(320.0)).unwrap();
            let mut reused =
                Simulator::from_elaborated(Arc::clone(&elaborated), Values::default()).unwrap();
            for analysis in &analyses {
                // A run may fail, as the ring's sweep of its kick does not; what it leaves
                // is forgotten all the same.
                let _ = reused.run(analysis, &options);
            }
            assert_ne!(format!("{:?}", reused.state), format!("{:?}", fresh.state));
            reused.reset(Values::at(320.0)).unwrap();
            assert_eq!(format!("{:?}", reused.state), format!("{:?}", fresh.state));
        }
    }
}
