//! Circuit diagnostics covering pre-analysis topology checks and operating-point checks.
//!
//! Each [`Rule`] defines a diagnostic condition and default [`Level`]. [`check`] inspects the
//! branches declared by devices in an elaborated circuit and emits a [`Diagnostic`] for each
//! structural rule violation, naming the affected instances and nets. [`check_operating_point`]
//! evaluates linearized operating-point rules such as growing natural modes. Every diagnostic
//! states a factual condition without prescribing circuit edits.
//!
//! ```
//! use jolty_engine::{Conditions, Simulator};
//! use jolty_ir::Circuit;
//! use jolty_models::Library;
//!
//! let circuit = Circuit::from_json(r#"{
//!     "format": "jolty-circuit", "version": 5, "top": "top",
//!     "components": {"top": {
//!         "ports": [{"name": "ground", "role": "ground"}],
//!         "instances": [
//!             {"name": "I1", "of": {"model": "current_source"},
//!              "parameters": {"waveform": {"dc": 1e-3}},
//!              "terminals": {"positive": "ground", "negative": "top"}},
//!             {"name": "C1", "of": {"model": "capacitor"},
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

pub use point::{MAX_LINEARIZED, check_operating_point};
pub use rules::RULES;

/// Reporting severity for a rule's diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Level {
    /// Severity indicating a definite circuit error.
    Error,
    /// Severity indicating a likely or conditional circuit defect.
    Warn,
    /// Severity suppressing diagnostics for the rule.
    Ignore,
}

impl Level {
    /// Returns the lowercase level name: `error`, `warn`, or `ignore`.
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

/// Definition of a circuit check rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rule {
    /// Kebab-case rule identifier, such as `no-dc-path`.
    pub name: &'static str,
    /// Default reporting severity unless overridden by configuration.
    pub level: Level,
    /// Complete factual description of the condition detected by the rule.
    pub condition: &'static str,
}

/// Single diagnostic reported for a rule violation in a circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    /// Identifier of the triggered rule.
    pub rule: &'static str,
    /// Factual diagnostic message naming the involved instances and nets.
    pub message: String,
    /// Involved device instances with their source locations.
    pub instances: Vec<Location>,
    /// Involved net paths.
    pub nets: Vec<String>,
}

/// Runs all structural rules on the circuit elaborated by `simulator` and returns the
/// resulting diagnostics in [`RULES`] order.
///
/// Within each rule, diagnostics follow instance placement order with analog devices before
/// logic devices.
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

#[cfg(test)]
pub(crate) mod testing {
    use jolty_engine::{Conditions, Simulator};
    use jolty_ir::Circuit;
    use jolty_models::Library;
    use serde_json::{Value, json};

    /// Builds a simulator whose top component declares ground port `0` and places `instances`.
    pub(crate) fn simulator(instances: &Value) -> Simulator {
        let document = json!({
            "format": "jolty-circuit", "version": 5, "top": "test",
            "components": {"test": {"ports": [{"name": "0", "role": "ground"}],
                                    "instances": instances}}
        });
        let circuit = Circuit::from_json(&document.to_string()).expect("a valid circuit");
        Simulator::new(&circuit, &Library::standard(), &Conditions::default())
            .expect("an elaborated circuit")
    }
}
