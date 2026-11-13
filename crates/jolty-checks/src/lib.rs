//! The checks of jolty: facts about a circuit's topology, found before any analysis, and
//! about the circuit linearized at its operating point.
//!
//! Each [`Rule`] names one fact a check can state, with its default [`Level`] and the exact
//! condition it checks. [`check`] reads an elaborated circuit through the branches each
//! device declares, and returns a [`Diagnostic`] for every instance of every structural
//! rule, each naming the instances and nets involved; [`check_operating_point`] returns
//! those of the rules on the circuit linearized at its operating point, such as a natural
//! mode that grows away from it. A diagnostic states a fact and never how to change the
//! circuit.
//!
//! ```
//! use jolty_engine::{Conditions, Simulator};
//! use jolty_ir::Circuit;
//! use jolty_models::Library;
//!
//! let circuit = Circuit::from_json(r#"{
//!     "format": "jolty-circuit", "version": 1, "top": "top",
//!     "components": {"top": {
//!         "ports": [{"name": "ground", "role": "ground"}],
//!         "instances": [
//!             {"id": "I1", "of": {"device": "current_source"},
//!              "parameters": {"waveform": {"dc": 1e-3}},
//!              "terminals": {"positive": "ground", "negative": "top"}},
//!             {"id": "C1", "of": {"device": "capacitor"},
//!              "parameters": {"capacitance": 1e-9},
//!              "terminals": {"a": "top", "b": "ground"}}
//!         ]
//!     }}
//! }"#)?;
//! let simulator = Simulator::new(&circuit, &Library::standard(), &Conditions::default())?;
//! let diagnostics = jolty_checks::check(&simulator);
//! assert_eq!(diagnostics[0].rule, "current-cutset");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod point;
pub mod preferred;
mod rules;
mod structure;

use std::fmt;

use jolty_engine::{Location, Simulator};

pub use point::{MOST_LINEARIZED, check_operating_point};
pub use rules::RULES;

/// How a check reports a rule's diagnostics, by the names ty gives its levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Level {
    /// The fact surely breaks the circuit.
    Error,
    /// The fact may break the circuit.
    Warn,
    /// The check leaves the rule out.
    Ignore,
}

impl Level {
    /// The level's name: `error`, `warn` or `ignore`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warn => "warn",
            Self::Ignore => "ignore",
        }
    }
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// One fact a check can state about a circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rule {
    /// The rule's name: lowercase words joined by hyphens, such as `no-dc-path`.
    pub name: &'static str,
    /// The level a check reports the rule at, unless configured otherwise.
    pub level: Level,
    /// The exact condition the rule checks.
    pub condition: &'static str,
}

/// One instance of a rule's fact in a circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    /// The name of the rule.
    pub rule: &'static str,
    /// What is true, naming the instances and nets involved.
    pub message: String,
    /// The instances involved, each with the script line that placed it.
    pub instances: Vec<Location>,
    /// The nets involved, by path.
    pub nets: Vec<String>,
}

/// Every diagnostic of every rule in the circuit `simulator` elaborated, rule by rule in
/// the order of [`RULES`], and within a rule in the order the instances were placed, analog
/// devices before logic ones.
#[must_use]
pub fn check(simulator: &Simulator) -> Vec<Diagnostic> {
    let structure = structure::Structure::new(simulator);
    let mut diagnostics = Vec::new();
    rules::unconnected_terminals(&structure, &mut diagnostics);
    rules::shorted_sources(&structure, &mut diagnostics);
    rules::shorted_devices(&structure, &mut diagnostics);
    rules::voltage_loops(&structure, &mut diagnostics);
    let floating = rules::floating_groups(&structure);
    rules::no_dc_paths(&structure, &floating, &mut diagnostics);
    rules::current_cutsets(&structure, &floating, &mut diagnostics);
    rules::driver_conflicts(simulator, &mut diagnostics);
    rules::floating_inputs(simulator, &mut diagnostics);
    rules::non_standard_values(simulator, &mut diagnostics);
    diagnostics
}
