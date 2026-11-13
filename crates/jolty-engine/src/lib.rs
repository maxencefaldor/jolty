//! The simulation engine of jolty.
//!
//! A [`Simulator`] elaborates a circuit once, flattening its hierarchy into devices bound to
//! unknowns and claiming the sparsity pattern of its matrix, then runs any number of analyses
//! on it: an [operating point](Simulator::operating_point), a [DC sweep](Simulator::dc_sweep),
//! a [small-signal frequency response](Simulator::ac), a [transient](Simulator::transient),
//! [sensitivities](Simulator::sensitivity), the [loop gain](Simulator::stability) through a
//! probe, the [noise](Simulator::noise) at an output, and the [poles and
//! zeros](Simulator::pole_zero).
//! Every analysis solves the charge-oriented equations `f(x) + d/dt q(x) = 0` by Newton
//! iteration with one sparse LU solver, whose symbolic analysis is shared by all of them.
//! Logic devices join logic nets, which an event scheduler simulates beside the analog
//! circuit, joined to it by connect elements where a logic pin meets an electrical net.
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
mod logic;
mod newton;
mod noise;
mod observe;
mod operating_point;
mod options;
mod periodic;
mod pole_zero;
mod sensitivity;
mod stability;
mod sweep;
mod transient;

use std::collections::BTreeMap;

use jolty_ir::{Circuit, Name, Parameters};
use jolty_models::{Level, Library, REFERENCE_TEMPERATURE, Unknown, UnknownInfo};
use jolty_numerics::{LinearError, SparseLu};
use serde::Deserialize;

pub use ac::{Ac, Response};
pub use analysis::{Analysis, Outcome};
pub use elaborate::{ElaborationError, ElaborationProblem, Location, Placed};
pub use jolty_numerics::integration::Method;
pub use logic::{Bound, Endpoint, Excursion, LogicNet, PlacedLogic};
pub use noise::{Noise, NoiseContribution, NoiseInput, NoiseSpectrum};
pub use observe::{Observation, Observations, Observed};
pub use operating_point::{OperatingPoint, OperatingPoints, Strategy};
pub use options::Options;
pub use pole_zero::{Mode, PoleZero, PolesZeros};
pub use sensitivity::{Derivatives, Differentiated, Parameter, Sensitivities, Sensitivity};
pub use stability::{LoopGain, Stability};
pub use sweep::{DcSweep, Sweep};
pub use transient::{Statistics, Transient, Waveforms};

use elaborate::System;
use logic::Events;
use newton::Workspace;
use observe::Observer;

/// What a circuit is simulated in, apart from its analyses: its devices' temperature,
/// parameters that replace the circuit's own for some device instances, as a Monte Carlo
/// sample's do, and models that stand in for some component instances, as a plan's
/// fidelity chooses.
#[derive(Debug, Clone, PartialEq)]
pub struct Conditions {
    /// The devices' temperature, in kelvins.
    pub temperature: f64,
    /// Parameters that replace an instance's own, by the instance's path, such as `X1/R1`,
    /// then by the parameter's [path](jolty_ir::parameter()), such as `waveform.dc`.
    pub overrides: BTreeMap<String, Parameters>,
    /// Models that stand in for component instances, by the instance's path, such as
    /// `adder/U1`.
    pub substitutes: BTreeMap<String, Substitute>,
}

/// A model that stands in for a component instance: a device whose terminals are the
/// component's ports, joined to the same nets.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Substitute {
    /// The model's name, such as `logic_cell`.
    pub model: Name,
    /// The device's parameters.
    pub parameters: Parameters,
}

impl Conditions {
    /// The conditions at `temperature`, in kelvins, with no parameter replaced and no
    /// model standing in.
    #[must_use]
    pub const fn at(temperature: f64) -> Self {
        Self {
            temperature,
            overrides: BTreeMap::new(),
            substitutes: BTreeMap::new(),
        }
    }
}

impl Default for Conditions {
    /// The conditions at 27 °C, with no parameter replaced and no model standing in.
    fn default() -> Self {
        Self::at(REFERENCE_TEMPERATURE)
    }
}

/// A circuit elaborated for simulation, ready to run analyses.
///
/// Every analysis returns [`SimulationError::Foundry`] for a circuit that holds a foundry's
/// device, which ngspice simulates in the engine's place.
#[derive(Debug)]
pub struct Simulator {
    system: System,
    solver: SparseLu<f64>,
    workspace: Workspace,
    /// The state of the logic nets, as the last analysis left it.
    events: Events,
    /// The devices made again, to read their terminals.
    observer: Observer,
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
        let events = Events::new(&system.logic);
        let observer = Observer::new(&system)?;
        Ok(Self {
            system,
            solver,
            workspace,
            events,
            observer,
        })
    }

    /// Every terminal of every device, by its device's path and its own name, such as
    /// `X1/R1.a`, in the order of every [`Observation`]'s currents and charges.
    #[must_use]
    pub fn terminals(&self) -> &[String] {
        self.observer.names()
    }

    /// The unknown that is each terminal's voltage, by name, in the order of
    /// [`Simulator::terminals`], or `None` for a terminal on the reference.
    #[must_use]
    pub fn terminal_nodes(&self) -> Vec<Option<&str>> {
        self.system
            .devices
            .iter()
            .flat_map(|device| {
                device.nodes().iter().map(|node| {
                    node.map(|unknown| self.system.unknowns[unknown.index()].name.as_str())
                })
            })
            .collect()
    }

    /// Whether each terminal, in the order of [`Simulator::terminals`], is thermal: its
    /// voltage a temperature above the ambient, and its current the heat flowing into it.
    #[must_use]
    pub fn thermal_terminals(&self) -> Vec<bool> {
        self.system
            .devices
            .iter()
            .flat_map(|device| {
                let thermal = device.thermal();
                (0..device.nodes().len()).map(move |terminal| thermal.contains(&terminal))
            })
            .collect()
    }

    /// The equation and the unknown of every derivative of every device, in the order of
    /// every [`Observation`]'s conductances and capacitances: for each device, every pair
    /// of its terminals and internal unknowns, row by row. Each is named as a terminal, for
    /// the current into it and the voltage at it, or as one of the device's internal
    /// unknowns, such as `X1/Q1:base`, for its equation and its value.
    #[must_use]
    pub fn derivatives(&self) -> &[(String, String)] {
        self.observer.derivatives()
    }

    /// Every variable of every device, by its device's path and its own name, such as
    /// `threshold_voltage`, in the order of every [`Observation`]'s variables.
    #[must_use]
    pub fn variables(&self) -> &[(String, &'static str)] {
        self.observer.variables()
    }

    /// Every terminal of every component instance and logic device on an electrical or a
    /// thermal net, such as `X1.vdd`, with the indices among [`Simulator::terminals`] of
    /// the device terminals whose currents and charges sum to its own. One on the
    /// reference, which is the ambient as well as the ground, sums its electrical
    /// terminals only.
    #[must_use]
    pub fn composites(&self) -> &[(String, Vec<usize>)] {
        self.observer.composites()
    }

    /// Whether each composite terminal, in the order of [`Simulator::composites`], is on a
    /// thermal net.
    #[must_use]
    pub fn thermal_composites(&self) -> Vec<bool> {
        self.system
            .composites
            .iter()
            .map(|composite| composite.thermal)
            .collect()
    }

    /// The unknown that is each composite terminal's voltage, by name, in the order of
    /// [`Simulator::composites`], or `None` for one on the reference.
    #[must_use]
    pub fn composite_nodes(&self) -> Vec<Option<&str>> {
        self.system
            .composites
            .iter()
            .map(|composite| {
                composite
                    .node
                    .map(|unknown| self.system.unknowns[unknown.index()].name.as_str())
            })
            .collect()
    }

    /// Starts the logic driver at `element` among the devices on a `(time, level,
    /// transition)` ramp, and its copy that the observer reads; returns when the ramp
    /// ends.
    fn drive(
        &mut self,
        element: usize,
        (time, level, transition): (f64, Level, f64),
    ) -> Option<f64> {
        self.observer.drive(element, time, level, transition);
        let driver = self.system.devices[element].device.as_driver()?;
        driver.drive(time, level, transition);
        Some(driver.settles_at())
    }

    /// The unknowns, in the order of every solution's values.
    #[must_use]
    pub fn unknowns(&self) -> &[UnknownInfo] {
        &self.system.unknowns
    }

    /// The analog devices, with every component instance unfolded, in the order they were
    /// placed, then the connect elements of the logic devices' pins that join electrical
    /// nets.
    #[must_use]
    pub fn devices(&self) -> &[Placed] {
        &self.system.devices
    }

    /// The paths of the foundry's devices the circuit holds, which no analysis solves, in
    /// the order they were placed.
    #[must_use]
    pub fn foundry_devices(&self) -> Vec<&str> {
        self.system
            .devices
            .iter()
            .filter(|placed| placed.foundry())
            .map(Placed::path)
            .collect()
    }

    /// Checks that the engine solves every device of the circuit.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Foundry`] naming the foundry's devices it holds.
    pub(crate) fn solvable(&self) -> Result<(), SimulationError> {
        let foundry = self.foundry_devices();
        if foundry.is_empty() {
            Ok(())
        } else {
            Err(SimulationError::Foundry(
                foundry.into_iter().map(str::to_owned).collect(),
            ))
        }
    }

    /// Where the last analysis used logic devices outside their domains: each device and
    /// bound once, with the value furthest outside and when it first left.
    #[must_use]
    pub fn excursions(&self) -> Vec<Excursion> {
        self.events.excursions()
    }

    /// The number of connect elements among [`Simulator::devices`], all at its end.
    #[must_use]
    pub fn elements(&self) -> usize {
        self.system.logic.detectors.len() + self.system.logic.drivers.len()
    }

    /// The logic devices, with every component instance unfolded, in the order they were
    /// placed.
    #[must_use]
    pub fn logic_devices(&self) -> &[PlacedLogic] {
        &self.system.logic.devices
    }

    /// The logic nets, in the order of every result's levels: the nets only logic pins join,
    /// in the order they were first met, then the nets between logic pins and their connect
    /// elements.
    #[must_use]
    pub fn logic_nets(&self) -> &[LogicNet] {
        &self.system.logic.nets
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

/// Why the transient analysis rejected a step.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Rejection {
    /// Newton iteration did not converge.
    #[error("Newton iteration did not converge; furthest from converging: {}", .worst.join(", "))]
    Unconverged {
        /// The unknowns furthest from converging at the last iteration, worst first.
        worst: Vec<String>,
    },
    /// The estimated truncation error exceeded its tolerance.
    #[error("the truncation error of `{0}` exceeded its tolerance")]
    Inaccurate(String),
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
    #[error("the time step fell below {minimum:e} s at t = {time:e} s: {rejection}")]
    StepTooSmall {
        /// The time the analysis could not step past, in seconds.
        time: f64,
        /// The smallest step allowed, in seconds.
        minimum: f64,
        /// Why the analysis rejected its last step.
        rejection: Rejection,
    },
    /// A transient's results would hold more numbers than the engine keeps.
    #[error(
        "the transient's results would pass {} GiB before it stops, its steps {step:?} s \
         long at t = {time:?} s",
        crate::transient::MOST_HELD_GIB
    )]
    TooManyPoints {
        /// The time the analysis reached, in seconds.
        time: f64,
        /// The length of its steps there, in seconds.
        step: f64,
    },
    /// A sensitivity analysis names an output that is no unknown.
    #[error("there is no unknown named `{0}`")]
    UnknownOutput(String),
    /// An analysis names a source that does not exist.
    #[error("there is no instance at `{0}`")]
    UnknownSource(String),
    /// An analysis names an instance that is not an independent source.
    #[error("`{0}` is not an independent source")]
    NotASource(String),
    /// A stability analysis's probe is not an independent voltage source between two nodes
    /// other than the reference.
    #[error("`{0}` is not a voltage source off the reference, as a probe must be")]
    NotAProbe(String),
    /// An analysis's settings are out of range.
    #[error("invalid analysis: {0}")]
    InvalidAnalysis(String),
    /// The QZ iteration of a pole-zero analysis did not converge.
    #[error("the eigenvalues of the linearized circuit did not converge")]
    Unconverged,
    /// Shooting found no state that a period carries back to itself.
    #[error(
        "no periodic steady state found in {periods} periods; furthest from repeating: {}",
        .worst.join(", ")
    )]
    NoSteadyState {
        /// The periods shooting simulated.
        periods: usize,
        /// The unknowns furthest from repeating over the last period, worst first.
        worst: Vec<String>,
    },
    /// The options are out of range.
    #[error("invalid options: {0}")]
    InvalidOptions(String),
    /// The circuit holds a foundry's devices, which ngspice simulates with the foundry's
    /// models, and which the engine does not solve.
    #[error(
        "the engine solves no foundry's device, which ngspice simulates with its foundry's \
         models: {}",
        .0.join(", ")
    )]
    Foundry(Vec<String>),
}

impl From<LinearError> for SimulationError {
    fn from(error: LinearError) -> Self {
        match error {
            LinearError::Singular => Self::Singular,
            LinearError::TooLarge => Self::TooLarge,
        }
    }
}
