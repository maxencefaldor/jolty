//! A run's state: everything a run of an elaborated circuit writes.
//!
//! A [`State`] holds a run's devices, made for its values, and every buffer its analyses
//! fill: the sparse solver's factors, Newton iteration's workspace, the logic's events, the
//! devices made again to be read, and the operating point last solved. Resetting it for
//! other values keeps every buffer and makes every device again, from its initial memory,
//! and forgets everything a run leaves behind, so that a run never depends on the run
//! before it: a state reset equals a state made afresh for the same values, bit for bit.

use std::collections::BTreeMap;
use std::sync::Arc;

use jolty_models::Device;

use crate::elaborate::ElaborationError;
use crate::elaborated::{Elaborated, Origin, Origins};
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

/// What one run of an elaborated circuit writes.
#[derive(Debug)]
pub(crate) struct State {
    /// The devices' temperature, in kelvins.
    pub(crate) temperature: f64,
    /// The devices, made for the run, in the order of the circuit's.
    pub(crate) devices: Vec<Box<dyn Device>>,
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
    /// How many times the run has solved for an operating point, or a DC sweep's first
    /// level, rather than taken the last one's.
    pub(crate) solves: usize,
}

impl State {
    /// A state of `elaborated` made afresh, whose `devices` were made at `temperature`,
    /// those whose numbers a run's values replace from `patched`.
    pub(crate) fn new(
        elaborated: &Elaborated,
        devices: Vec<Box<dyn Device>>,
        patched: Vec<Option<Origin>>,
        temperature: f64,
    ) -> Self {
        Self {
            temperature,
            devices,
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
        Ok(Some(Self::new(elaborated, devices, patched, temperature)))
    }

    /// Makes the run's devices again for `values` and forgets everything the runs before
    /// it left: the operating point remembered, the sources held, the sparse solver's
    /// pivots and factors, the meter's counts, the logic's events and traces, and the
    /// devices made again to be read. Every buffer is kept. Returns whether the devices of
    /// `values` have the structure of `elaborated`; when they do not, or the values make a
    /// problem, the state is left as it was.
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
        let replayed = made(elaborated, &patched, temperature, &mut devices);
        let made_again = matches!(replayed, Ok(true));
        if made_again {
            std::mem::swap(&mut self.devices, &mut devices);
            self.patched = patched;
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
        self.solves = 0;
        Ok(true)
    }
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
