//! Mutable per-run simulation state for an elaborated circuit.
//!
//! [`State`] owns prepared device kernel instances, evaluation slabs, sparse LU factors,
//! the Newton workspace, logic event queues, device recording history, and the cached
//! operating point. Resetting for new parameter values reuses allocations while
//! re-preparing changed devices, resetting device memory, and clearing analysis history so
//! a reset state matches a fresh state bit for bit.

use std::collections::BTreeMap;
use std::sync::Arc;

use jolty_models::{Changed, Past, Runs, Tasks, UnknownIndex};

use crate::elaborate::ElaborationError;
use crate::elaborated::{Devices, Elaborated, Origin, Origins};
use crate::logic::Events;
use crate::newton::Workspace;
use crate::operating_point::Remembered;
use crate::record::Recorder;
use crate::solver::Solver;
use crate::{ParameterId, Values};

/// Target temporarily overridden by a DC sweep: either an independent source value or a
/// numeric device parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Held {
    /// Independent source identified by its device index.
    Value(usize),
    /// Numeric device parameter identified by [`ParameterId`].
    Number(ParameterId),
}

/// Warm-start operating point and parameter values carried from the previous run in a
/// chain.
#[derive(Debug, Clone)]
pub(crate) struct Warm {
    pub(crate) values: Values,
    pub(crate) solution: Vec<f64>,
}

/// Mutable state owned by a single simulation run of an elaborated circuit.
#[derive(Debug)]
pub(crate) struct State {
    /// Active device operating temperature in kelvins.
    pub(crate) temperature: f64,
    /// Prepared device kernel groups, per-instance memory, and evaluation output slab.
    pub(crate) runs: Runs,
    /// Parallel device-evaluation task plan, or `None` for single-threaded evaluation.
    pub(crate) tasks: Option<Tasks>,
    /// Optional test harness comparing grouped kernel assembly against standalone device
    /// evaluation.
    #[cfg(test)]
    pub(crate) check: Option<crate::kernels::Check>,
    /// Patched device origins by device index (`None` for nominal devices; empty when no
    /// parameters are patched).
    pub(crate) patched: Vec<Option<Origin>>,
    /// Temporary device origins installed by an active DC sweep, keyed by device index.
    pub(crate) swept: BTreeMap<usize, Origin>,
    pub(crate) solver: Solver<f64>,
    pub(crate) workspace: Workspace,
    /// Logic net state and event queue left by the last analysis.
    pub(crate) events: Events,
    /// Active device recorder and transient terminal charge history.
    pub(crate) recorder: Recorder,
    /// Cached operating point and validity metadata reused by subsequent linearized
    /// analyses under matching conditions.
    ///
    /// After device preparation, only active sweep holds and logic driver state alter DC
    /// equations, so both are tracked in the cache key.
    pub(crate) remembered: Option<Remembered>,
    /// Active DC sweep overrides away from nominal run values, mapping [`Held`] keys to
    /// their held numbers.
    pub(crate) held: BTreeMap<Held, f64>,
    /// Node voltages forced during DC solves (initial conditions or active nodesets),
    /// ramped alongside sources during source stepping.
    pub(crate) forced: Vec<(UnknownIndex, f64)>,
    /// Optional warm-start state from the previous run in a chain (`None` starts from zero).
    pub(crate) warm: Option<Warm>,
    /// Cumulative operating-point solves performed (excluding cached reuses).
    pub(crate) solves: usize,
    /// Accepted transient waveform history read by delay lines during transient evaluation,
    /// absent when no delay lines exist or outside transient runs.
    pub(crate) past: Option<Past>,
}

impl State {
    /// Creates a fresh [`State`] for `elaborated` from prepared `runs`, `patched` origins,
    /// and `temperature`.
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
            recorder: Recorder::new(),
            remembered: None,
            held: BTreeMap::new(),
            forced: Vec::new(),
            warm: None,
            solves: 0,
            past: None,
        }
    }

    /// Creates a [`State`] for `elaborated` prepared at `values`, returning `None` if
    /// `values` alter any device kernel shape and require restructuring.
    ///
    /// # Errors
    ///
    /// Returns every elaboration problem triggered by `values`, matching
    /// [`Elaborated::structure_of`].
    pub(crate) fn from_values(
        elaborated: &Elaborated,
        values: &Values,
    ) -> Result<Option<Self>, ElaborationError> {
        let temperature = elaborated.temperature_of(values);
        let patched = elaborated.patch(values)?;
        let mut runs = elaborated.groups.runs();
        let swept = BTreeMap::new();
        let origins = Origins {
            elaborated,
            patched: &patched,
            swept: &swept,
            temperature,
        };
        let changed = |device| changes(&origins, device, Refresh::Always);
        if !prepared(&mut runs, &origins, changed)? {
            return Ok(None);
        }
        // Every instance is prepared, so each starts from its initial memory.
        runs.commit();
        Ok(Some(Self::new(elaborated, runs, patched, temperature)))
    }

    /// Resets the state in place for `values`, re-preparing changed devices and clearing
    /// all cached solver, logic, recording, and meter state while keeping allocations.
    ///
    /// Returns `true` if all devices keep their elaborated kernel shapes; returns `false`
    /// without mutating state when restructuring is required.
    ///
    /// # Errors
    ///
    /// Returns every elaboration problem triggered by `values`, matching
    /// [`Elaborated::structure_of`], leaving state unchanged.
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
        let origins = Origins {
            elaborated,
            patched: &patched,
            swept: &self.swept,
            temperature,
        };
        let changed = |device| {
            let refresh = if moved || old.get(device).is_some_and(Option::is_some) {
                Refresh::Always
            } else {
                Refresh::IfPatched
            };
            changes(&origins, device, refresh)
        };
        let replayed = prepared(&mut self.runs, &origins, changed);
        if !matches!(replayed, Ok(true)) {
            return replayed;
        }
        self.patched = patched;
        self.runs.commit();
        self.runs.forget();
        self.temperature = temperature;
        self.solver.reset();
        self.workspace.reset();
        self.solves = 0;
        self.forget(elaborated);
        Ok(true)
    }

    /// Clears transient analysis caches (cached operating point, sweep holds, logic events,
    /// and recorder state) while preserving devices, sparse pivots, and meter counts.
    pub(crate) fn forget(&mut self, elaborated: &Elaborated) {
        self.swept.clear();
        self.events.reset(&elaborated.logic);
        self.recorder.reset();
        self.remembered = None;
        self.held.clear();
        self.forced.clear();
        self.warm = None;
        self.past = None;
    }

    /// Re-prepares devices whose parameter patches or temperature differ between `from` and
    /// `to`, leaving unchanged devices and their memory intact.
    ///
    /// Returns `false` without modifying state if any updated device changes kernel shape.
    ///
    /// # Errors
    ///
    /// Returns elaboration problems produced by out-of-range updated parameters.
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
        // Swap updated origins in, keeping previous values to roll back on shape mismatch.
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
        let origins = Origins {
            elaborated,
            patched: &self.patched,
            swept: &self.swept,
            temperature,
        };
        let changed = |device: usize| {
            if again.binary_search(&device).is_ok() {
                changes(&origins, device, Refresh::Always)
            } else {
                Changed::Nothing
            }
        };
        let kept = prepared(&mut self.runs, &origins, changed);
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

    /// Re-prepares the kernel instance at `device` from its current swept, patched, or
    /// nominal origin.
    ///
    /// Returns `true` if the device keeps its group's kernel and shape; returns `false`
    /// without committing changes otherwise.
    ///
    /// # Errors
    ///
    /// Returns the elaboration problem produced by out-of-range device parameters.
    pub(crate) fn prepare_again(
        &mut self,
        elaborated: &Elaborated,
        device: usize,
    ) -> Result<bool, ElaborationError> {
        let swept = self.swept.contains_key(&device);
        let origins = Origins {
            elaborated,
            patched: &self.patched,
            swept: &self.swept,
            temperature: self.temperature,
        };
        let changed = |index: usize| match (index == device, swept) {
            (false, _) => Changed::Nothing,
            (true, true) => Changed::Parameters,
            (true, false) => changes(&origins, index, Refresh::Always),
        };
        let kept = prepared(&mut self.runs, &origins, changed)?;
        if kept {
            self.runs.commit();
        }
        Ok(kept)
    }
}

/// Policy controlling whether an unpatched device must be re-prepared.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Refresh {
    IfPatched,
    Always,
}

/// Returns whether patch slices `a` and `b` override identical parameters with bit-equal
/// values.
fn same(a: &[(ParameterId, f64)], b: &[(ParameterId, f64)]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|((p, x), (q, y))| p == q && x.to_bits() == y.to_bits())
}

/// Classifies what changed for `device` under `origins` and `refresh`.
///
/// Patched devices, non-nominal devices, or [`Refresh::Always`] updates at a changed
/// temperature re-read parameters (because temperature drift can alter kernel shape, such
/// as a resistor crossing 1 Ω); [`Refresh::Always`] at the nominal temperature refreshes
/// numbers only.
fn changes(origins: &Origins<'_>, device: usize, refresh: Refresh) -> Changed {
    let replaced = origins.patched.get(device).is_some_and(Option::is_some);
    let made_otherwise = origins
        .elaborated
        .devices
        .get(device)
        .is_some_and(|placed| !placed.nominal());
    let warmer = origins.temperature.to_bits() != origins.elaborated.temperature.to_bits();
    let again = refresh == Refresh::Always;
    if replaced || made_otherwise || (again && warmer) {
        Changed::Parameters
    } else if again {
        Changed::Numbers
    } else {
        Changed::Nothing
    }
}

/// Prepares changed kernel instances in `runs` from `origins` into staging storage until
/// committed, returning whether every instance keeps its group's kernel and shape.
///
/// # Errors
///
/// Returns elaboration problems for every device with out-of-range parameters.
fn prepared(
    runs: &mut Runs,
    origins: &Origins<'_>,
    changed: impl Fn(usize) -> Changed,
) -> Result<bool, ElaborationError> {
    runs.prepare(
        &origins.elaborated.groups,
        |device| origins.of(device).kernel(),
        origins.temperature,
        changed,
    )
    .map_err(|problems| {
        let problems = problems.into_iter();
        ElaborationError::new(
            problems
                .map(|(device, error)| origins.elaborated.problem(device, error))
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

    /// Loads `bjt_amp` and constructs a mixed-signal logic-inverter ring circuit for reset
    /// equivalence testing.
    fn circuits() -> [Circuit; 2] {
        let speed = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/speed");
        let text = std::fs::read_to_string(speed.join("circuits/bjt_amp.json")).unwrap();
        let amplifier = Circuit::from_json(&text).unwrap();
        let device = |name: &str, model: &str, parameters, terminals| {
            json!({"name": name, "of": {"model": model}, "parameters": parameters,
                   "terminals": terminals})
        };
        let ring = json!({
            "format": "jolty-circuit", "version": 5, "top": "ring",
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

    /// Verifies that resetting a used state matches a freshly constructed state across
    /// device memory, solver counts, workspace, logic events, recordings, and holds.
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
