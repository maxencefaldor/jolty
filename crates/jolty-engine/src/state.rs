//! A run's state: everything a run of an elaborated circuit writes.
//!
//! A [`State`] holds the prepared numbers and the memory of a run's devices' instances,
//! each made once, in its group, for the run's values, and the slab they write; and every
//! buffer its analyses fill: the sparse solver's factors, Newton iteration's workspace, the
//! logic's events, what the analyses record of the devices, and the operating point last
//! solved. Resetting it for other values keeps every buffer, prepares again the instances
//! whose numbers or temperature change, starts every one from its initial memory, and
//! forgets everything a run leaves behind, so that a run never depends on the run before
//! it: a state reset equals a state made afresh for the same values, bit for bit.

use std::collections::BTreeMap;
use std::sync::Arc;

use jolty_models::{Changed, Past, Runs, Tasks, UnknownIndex};

use crate::elaborate::ElaborationError;
use crate::elaborated::{Devices, Elaborated, Origin, Origins};
use crate::logic::Events;
use crate::newton::Workspace;
use crate::operating_point::Remembered;
use crate::record::Observer;
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
    pub(crate) solver: Solver<f64>,
    pub(crate) workspace: Workspace,
    /// The state of the logic nets, as the last analysis left it.
    pub(crate) events: Events,
    /// What the analyses record of the devices, and the charges a transient carries for
    /// them.
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
    /// The nets a DC solution forces to a voltage while it is solved, each a node's
    /// unknown and its voltage, which source stepping ramps as it ramps the sources: a
    /// transient's initial conditions, or the nodesets until they are released; none
    /// outside such a solution.
    pub(crate) forced: Vec<(UnknownIndex, f64)>,
    /// Where the run's first operating point starts, in a chain of runs: none for a run
    /// that starts from zero.
    pub(crate) warm: Option<Warm>,
    /// How many times the run has solved for an operating point, or a DC sweep's first
    /// level, rather than taken the last one's.
    pub(crate) solves: usize,
    /// What the unknowns its delay lines read took at the points the transient being run
    /// accepted, which each evaluation at a time reads: made afresh as a transient starts,
    /// none in one of a circuit without a delay line, and forgotten as one ends.
    pub(crate) past: Option<Past>,
}

impl State {
    /// A state of `elaborated` made afresh, whose instances `runs` were prepared at
    /// `temperature`, those whose numbers a run's values replace from `patched`.
    pub(crate) fn new(
        elaborated: &Elaborated,
        runs: Devices,
        patched: Vec<Option<Origin>>,
        temperature: f64,
    ) -> Self {
        Self {
            temperature,
            runs,
            tasks: None,
            #[cfg(test)]
            check: None,
            patched,
            swept: BTreeMap::new(),
            solver: Solver::new(
                Arc::clone(&elaborated.ordering),
                Arc::clone(&elaborated.names),
            ),
            workspace: Workspace::new(elaborated),
            events: Events::new(&elaborated.logic),
            observer: Observer::new(),
            remembered: None,
            held: BTreeMap::new(),
            forced: Vec::new(),
            warm: None,
            solves: 0,
            past: None,
        }
    }

    /// A state of `elaborated` for `values`, its instances prepared for them, or `None`
    /// when the values give some instance another shape, so the circuit another
    /// structure.
    ///
    /// # Errors
    ///
    /// Returns every problem the values make, as [`Elaborated::structure_of`] does.
    pub(crate) fn from_values(
        elaborated: &Elaborated,
        values: &Values,
    ) -> Result<Option<Self>, ElaborationError> {
        let temperature = elaborated.temperature_of(values);
        let patched = elaborated.patch(values)?;
        let mut runs = elaborated.groups.runs();
        let changed = |device| changes(elaborated, (&patched, temperature), device, true);
        let swept = BTreeMap::new();
        if !prepared(
            elaborated,
            (&mut runs, &patched, &swept),
            temperature,
            changed,
        )? {
            return Ok(None);
        }
        // Every instance is prepared, so each starts from its initial memory.
        runs.commit();
        Ok(Some(Self::new(elaborated, runs, patched, temperature)))
    }

    /// Prepares again the run's instances whose numbers or temperature `values` change,
    /// and forgets everything the runs before it left: the instances' memory and the slab,
    /// the operating point remembered, the sources held, the sparse solver's pivots and
    /// factors, the meter's counts, the logic's events and traces, and what the analyses
    /// record. Every buffer is kept. Returns whether the instances of `values`
    /// have the shapes of `elaborated`'s; when they do not, or the values make a problem,
    /// the state is left as it was.
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
        // An instance keeps its numbers unless they or its temperature change.
        let moved = temperature.to_bits() != self.temperature.to_bits();
        let old = &self.patched;
        let changed = |device| {
            let again = moved || old.get(device).is_some_and(Option::is_some);
            changes(elaborated, (&patched, temperature), device, again)
        };
        let replayed = prepared(
            elaborated,
            (&mut self.runs, &patched, &self.swept),
            temperature,
            changed,
        );
        if !matches!(replayed, Ok(true)) {
            return replayed;
        }
        self.patched = patched;
        self.runs.commit();
        self.runs.forget();
        self.temperature = temperature;
        self.swept.clear();
        self.solver.reset();
        self.workspace.reset();
        self.events.reset(&elaborated.logic);
        self.observer.reset();
        self.remembered = None;
        self.held.clear();
        self.forced.clear();
        self.warm = None;
        self.solves = 0;
        self.past = None;
        Ok(true)
    }

    /// Forgets what a run leaves but its devices, the sparse solver's pivots and the
    /// meter's counts: the operating point remembered, what a sweep holds, the logic's
    /// events and traces, and what the analyses record.
    pub(crate) fn forget(&mut self, elaborated: &Elaborated) {
        self.swept.clear();
        self.events.reset(&elaborated.logic);
        self.observer.reset();
        self.remembered = None;
        self.held.clear();
        self.forced.clear();
        self.warm = None;
        self.past = None;
    }

    /// Prepares again, for `to`, the instances whose numbers or temperature differ from
    /// those of `from`, the values the run's instances are prepared for, each from its
    /// initial memory; leaves every other instance as it is, its memory among it. Returns whether each of them keeps
    /// its shape in `elaborated`; when one does not, or the values make a problem, the
    /// state is left as it was.
    ///
    /// # Errors
    ///
    /// Returns every problem the values make for the instances prepared again.
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
            made.push((index, origin));
        }
        if !problems.is_empty() {
            return Err(ElaborationError::new(problems));
        }
        // The origins of the instances prepared again replace their old ones, which come
        // back if the instances refuse them.
        let was_empty = self.patched.is_empty();
        if was_empty && made.iter().any(|(_, origin)| origin.is_some()) {
            self.patched.resize_with(elaborated.devices.len(), || None);
        }
        let mut previous = Vec::with_capacity(made.len());
        for (index, origin) in &mut made {
            if let Some(slot) = self.patched.get_mut(*index) {
                previous.push((*index, std::mem::replace(slot, origin.take())));
            }
        }
        let again: Vec<usize> = made.iter().map(|(index, _)| *index).collect();
        let patched = &self.patched;
        let changed = |device: usize| {
            if again.binary_search(&device).is_ok() {
                changes(elaborated, (patched, temperature), device, true)
            } else {
                Changed::Nothing
            }
        };
        let kept = prepared(
            elaborated,
            (&mut self.runs, &self.patched, &self.swept),
            temperature,
            changed,
        );
        if !matches!(kept, Ok(true)) {
            for (index, origin) in previous {
                self.patched[index] = origin;
            }
            if was_empty {
                self.patched.clear();
            }
            return kept;
        }
        self.runs.commit();
        self.temperature = temperature;
        Ok(true)
    }

    /// Prepares again the instance of a kernel the device at `device` is, if it is one,
    /// from its origin as the run now makes it: swept, patched or its own. Returns whether
    /// it keeps its group's kernel and shape; when it does not, or its numbers make a
    /// problem, it keeps the numbers it had.
    ///
    /// # Errors
    ///
    /// Returns the problem the device's numbers make.
    pub(crate) fn prepare_again(
        &mut self,
        elaborated: &Elaborated,
        device: usize,
    ) -> Result<bool, ElaborationError> {
        let swept = self.swept.contains_key(&device);
        let changed = |index: usize| match (index == device, swept) {
            (false, _) => Changed::Nothing,
            (true, true) => Changed::Parameters,
            (true, false) => changes(elaborated, (&self.patched, self.temperature), index, true),
        };
        let kept = prepared(
            elaborated,
            (&mut self.runs, &self.patched, &self.swept),
            self.temperature,
            changed,
        )?;
        if kept {
            self.runs.commit();
        }
        Ok(kept)
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
/// the numbers `patched` gives, at `temperature`: its parameters, when the values replace
/// them or the device's group was made of others; else, when its numbers are `again`
/// prepared, its parameters at another temperature than its group's, which may give it
/// another shape, as a resistor's crossing 1 Ω does, and its numbers at its group's; and
/// otherwise nothing.
fn changes(
    elaborated: &Elaborated,
    (patched, temperature): (&[Option<Origin>], f64),
    device: usize,
    again: bool,
) -> Changed {
    let replaced = patched.get(device).is_some_and(Option::is_some);
    let made_otherwise = elaborated
        .devices
        .get(device)
        .is_some_and(|placed| !placed.nominal());
    let warmer = temperature.to_bits() != elaborated.temperature.to_bits();
    if replaced || made_otherwise || (again && warmer) {
        Changed::Parameters
    } else if again {
        Changed::Numbers
    } else {
        Changed::Nothing
    }
}

/// Prepares, in `runs`, each instance of a kernel of `elaborated` whose device `changed`
/// says has changed, for a run at `temperature`, from its origin of `swept`, of `patched`
/// or its own, keeping the numbers apart until `runs` commits them. Returns whether each
/// has its group's kernel and shape.
///
/// # Errors
///
/// Returns the problem of each device whose parameters are out of range.
fn prepared(
    elaborated: &Elaborated,
    (runs, patched, swept): (&mut Runs, &[Option<Origin>], &BTreeMap<usize, Origin>),
    temperature: f64,
    changed: impl Fn(usize) -> Changed,
) -> Result<bool, ElaborationError> {
    let origins = Origins {
        elaborated,
        patched,
        swept,
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
        let speed = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/speed");
        let text = std::fs::read_to_string(speed.join("circuits/bjt_amp.json")).unwrap();
        let amplifier = Circuit::from_json(&text).unwrap();
        let device = |id: &str, model: &str, parameters, terminals| {
            json!({"name": id, "of": {"model": model}, "parameters": parameters,
                   "terminals": terminals})
        };
        let ring = json!({
            "format": "jolty-circuit", "version": 4, "top": "ring",
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
    /// events, what the analyses record, the operating point remembered and the
    /// sources held, every one as a fresh state has it.
    #[test]
    fn a_state_reset_is_the_state_made_afresh() {
        let analyses: Vec<Analysis> = [
            json!({"transient": {"stop": 2e-6}}),
            json!("operating_point"),
            json!({"dc_sweep": {"source": "V1", "values": [0.0, 0.5]}}),
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
