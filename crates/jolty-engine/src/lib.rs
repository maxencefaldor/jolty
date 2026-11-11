//! The simulation engine of jolty.
//!
//! A [`Simulator`] elaborates a circuit once, flattening its hierarchy into devices bound to
//! unknowns and claiming the sparsity pattern of its matrix, then runs any number of analyses
//! on it: an [operating point](Simulator::operating_point), a [DC sweep](Simulator::dc_sweep),
//! a [small-signal frequency response](Simulator::ac) and a [transient](Simulator::transient).
//! Every analysis solves the charge-oriented equations `f(x) + d/dt q(x) = 0` by Newton
//! iteration with one sparse LU solver, whose symbolic analysis is shared by all of them.
//!
//! ```
//! use jolty_engine::{Conditions, Options, Simulator};
//! use jolty_ir::Circuit;
//! use jolty_models::Library;
//!
//! let circuit = Circuit::from_json(r#"{
//!     "format": "jolty-circuit", "version": 1, "top": "divider",
//!     "components": {"divider": {
//!         "ports": [{"name": "ground", "role": "ground"}],
//!         "instances": [
//!             {"id": "V1", "of": {"device": "voltage_source"},
//!              "parameters": {"waveform": {"dc": 5.0}},
//!              "terminals": {"positive": "top", "negative": "ground"}},
//!             {"id": "R1", "of": {"device": "resistor"},
//!              "parameters": {"resistance": 1000.0}, "terminals": {"a": "top", "b": "out"}},
//!             {"id": "R2", "of": {"device": "resistor"},
//!              "parameters": {"resistance": 1000.0}, "terminals": {"a": "out", "b": "ground"}}
//!         ]
//!     }}
//! }"#)?;
//! let mut simulator = Simulator::new(&circuit, &Library::standard(), &Conditions::default())?;
//! let solution = simulator.operating_point(&Options::default())?;
//! let out = simulator.unknown("out").expect("a net named out");
//! assert!((solution.values[out.index()] - 2.5).abs() < 1e-9);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod ac;
mod analysis;
mod elaborate;
mod newton;
mod operating_point;
mod options;
mod sweep;
mod transient;

use std::collections::BTreeMap;

use jolty_ir::{Circuit, Parameters};
use jolty_models::{Library, REFERENCE_TEMPERATURE, Unknown, UnknownInfo};
use jolty_numerics::{LinearError, SparseLu};

pub use ac::{Ac, Response};
pub use analysis::{Analysis, Outcome};
pub use elaborate::{ElaborationError, ElaborationProblem, Location, Placed};
pub use jolty_numerics::integration::Method;
pub use operating_point::{OperatingPoint, Strategy};
pub use options::Options;
pub use sweep::{DcSweep, Sweep};
pub use transient::{Statistics, Transient, Waveforms};

use elaborate::System;
use newton::Workspace;

/// What a circuit is simulated in, apart from its analyses: its devices' temperature,
/// and parameters that replace the circuit's own for some device instances, as a Monte
/// Carlo sample's do.
#[derive(Debug, Clone, PartialEq)]
pub struct Conditions {
    /// The devices' temperature, in kelvins.
    pub temperature: f64,
    /// Parameters that replace an instance's own, by the instance's path, such as `X1/R1`.
    pub overrides: BTreeMap<String, Parameters>,
}

impl Conditions {
    /// The conditions at `temperature`, in kelvins, with no parameter replaced.
    #[must_use]
    pub const fn at(temperature: f64) -> Self {
        Self {
            temperature,
            overrides: BTreeMap::new(),
        }
    }
}

impl Default for Conditions {
    /// The conditions at 27 °C, with no parameter replaced.
    fn default() -> Self {
        Self::at(REFERENCE_TEMPERATURE)
    }
}

/// A circuit elaborated for simulation, ready to run analyses.
#[derive(Debug)]
pub struct Simulator {
    system: System,
    solver: SparseLu<f64>,
    workspace: Workspace,
}

impl Simulator {
    /// Elaborates `circuit` with the models of `library`, in `conditions`.
    ///
    /// # Errors
    ///
    /// Returns every [`ElaborationProblem`] found: an invalid structure, a missing reference
    /// node, an unknown model, wrongly connected terminals, invalid parameters, or
    /// parameters replaced for a device the circuit does not have.
    pub fn new(
        circuit: &Circuit,
        library: &Library,
        conditions: &Conditions,
    ) -> Result<Self, ElaborationError> {
        let system = elaborate::elaborate(circuit, library, conditions)?;
        let solver = SparseLu::new(system.pattern.clone())
            .map_err(|_| ElaborationError::from(ElaborationProblem::TooLarge))?;
        let workspace = Workspace::new(&system);
        Ok(Self {
            system,
            solver,
            workspace,
        })
    }

    /// The unknowns, in the order of every solution's values.
    #[must_use]
    pub fn unknowns(&self) -> &[UnknownInfo] {
        &self.system.unknowns
    }

    /// The devices, with every component instance unfolded, in the order they were placed.
    #[must_use]
    pub fn devices(&self) -> &[Placed] {
        &self.system.devices
    }

    /// The name of the reference node, the net of the top component's ground port.
    #[must_use]
    pub fn reference(&self) -> &str {
        self.system.reference.as_str()
    }

    /// The unknown named `name`: a net's path such as `amplifier/mid`, or an instance's path
    /// and label such as `V1:current`.
    #[must_use]
    pub fn unknown(&self, name: &str) -> Option<Unknown> {
        self.system
            .unknowns
            .iter()
            .position(|unknown| unknown.name == name)
            .map(Unknown::new)
    }
}

/// Why an analysis failed.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SimulationError {
    /// No operating point was found, even with gmin and source stepping.
    #[error("no operating point found; furthest from converging: {}", .worst.join(", "))]
    NoOperatingPoint {
        /// The unknowns furthest from converging at the last attempt, worst first.
        worst: Vec<String>,
    },
    /// The circuit's equations are singular at every attempt.
    #[error(
        "the circuit's equations are singular: a node may have no DC path to the reference, \
         or sources may fix conflicting values"
    )]
    Singular,
    /// The circuit's system is too large to factor.
    #[error("the circuit's system is too large to factor")]
    TooLarge,
    /// The transient analysis shortened its step below the smallest it allows.
    #[error("the time step fell below {minimum:e} s at t = {time:e} s")]
    StepTooSmall {
        /// The time the analysis could not step past, in seconds.
        time: f64,
        /// The smallest step allowed, in seconds.
        minimum: f64,
    },
    /// An analysis names a source that does not exist.
    #[error("there is no instance at `{0}`")]
    UnknownSource(String),
    /// An analysis names an instance that is not an independent source.
    #[error("`{0}` is not an independent source")]
    NotASource(String),
    /// An analysis's settings are out of range.
    #[error("invalid analysis: {0}")]
    InvalidAnalysis(String),
    /// The options are out of range.
    #[error("invalid options: {0}")]
    InvalidOptions(String),
}

impl From<LinearError> for SimulationError {
    fn from(error: LinearError) -> Self {
        match error {
            LinearError::Singular => Self::Singular,
            LinearError::TooLarge => Self::TooLarge,
        }
    }
}
