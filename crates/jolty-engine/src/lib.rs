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
//! iteration with one sparse LU solver, whose symbolic analysis is shared by all of them,
//! and every analysis about the operating point takes the one the simulator solved last.
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
use std::sync::{Arc, OnceLock};

use jolty_ir::{Circuit, Name, Parameters};
use jolty_models::{Level, Library, Node, REFERENCE_TEMPERATURE, Unknown, UnknownInfo};
use jolty_numerics::{LinearError, Ordering, SparseLu};
use serde::Deserialize;

pub use ac::{Ac, Response};
pub use analysis::{Analysis, Outcome};
pub use elaborate::{Composite, ElaborationError, ElaborationProblem, Location, Placed};
pub use jolty_numerics::integration::Method;
pub use logic::{Bound, Endpoint, Excursion, LogicNet, PlacedLogic};
pub use noise::{Noise, NoiseContribution, NoiseInput, NoiseSpectrum};
pub use observe::{Observation, Observations, Observed, Recorded};
pub use operating_point::{OperatingPoint, OperatingPoints, Strategy};
pub use options::{Options, Record};
pub use pole_zero::{Mode, PoleZero, PolesZeros};
pub use sensitivity::{Derivatives, Differentiated, Parameter, Sensitivities, Sensitivity};
pub use stability::{LoopGain, Stability};
pub use sweep::{DcSweep, Sweep};
pub use transient::{Statistics, Transient, Waveforms};

use elaborate::System;
use logic::Events;
use newton::Workspace;
use observe::Observer;
use operating_point::Remembered;

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
    /// The devices and the composite terminals in the order of their paths, made the first
    /// time a terminal is looked up.
    index: OnceLock<Index>,
    /// The last operating point solved, what it was solved under and the state it left,
    /// which an analysis about the same operating point takes instead of solving it again.
    /// Once the devices are made, only holding a source and driving the logic change their
    /// DC equations: the sources held are part of what it was solved under, and an
    /// operating point starts from the logic reset and leaves the logic and its drivers
    /// as the one remembered left them.
    remembered: Option<Remembered>,
    /// The sources held away from their own levels, at the level each is held at, by the
    /// index of its device.
    held: BTreeMap<usize, f64>,
    /// How many times the simulator has solved for an operating point, or a DC sweep's
    /// first level, rather than taken the last one's.
    solves: usize,
}

/// The devices and the composite terminals of a circuit, each by index, sorted by path and
/// by name, to find one by either.
#[derive(Debug)]
struct Index {
    devices: Vec<usize>,
    composites: Vec<usize>,
}

impl Index {
    /// The index of `system`'s devices and composite terminals.
    fn new(system: &System) -> Self {
        let mut devices: Vec<usize> = (0..system.devices.len()).collect();
        devices.sort_by(|&a, &b| system.devices[a].path().cmp(system.devices[b].path()));
        let mut composites: Vec<usize> = (0..system.composites.len()).collect();
        composites.sort_by(|&a, &b| system.composites[a].name.cmp(&system.composites[b].name));
        Self {
            devices,
            composites,
        }
    }
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
        let ordering = Ordering::new(system.pattern.clone())
            .map_err(|_| ElaborationError::from(ElaborationProblem::TooLarge))?;
        let solver = SparseLu::new(Arc::new(ordering));
        let workspace = Workspace::new(&system);
        let events = Events::new(&system.logic);
        let observer = Observer::new(&system);
        Ok(Self {
            system,
            solver,
            workspace,
            events,
            observer,
            index: OnceLock::new(),
            remembered: None,
            held: BTreeMap::new(),
            solves: 0,
        })
    }

    /// How many times the simulator has solved for an operating point, or for a DC sweep's
    /// first level, since it was made. An analysis about the operating point the circuit
    /// was last solved at, with the same sources held and under the same options, save
    /// those only a transient reads, takes that solution again instead, and solves for
    /// none.
    #[must_use]
    pub fn operating_point_solves(&self) -> usize {
        self.solves
    }

    /// The terminals of every component instance and logic device on an electrical or a
    /// thermal net, such as `X1.vdd`, each with the device terminals whose currents and
    /// charges sum to its own. One on the reference, which is the ambient as well as the
    /// ground, sums its electrical terminals only.
    #[must_use]
    pub fn composites(&self) -> &[Composite] {
        &self.system.composites
    }

    /// The node that the terminal `terminal` of the device, component instance or logic
    /// device at the path `instance` joins, such as `X1/R1` and `a`, and whether it is
    /// thermal: its voltage a temperature above the ambient, and its current the heat
    /// flowing into it. `None` when the instance has no such terminal on an electrical or a
    /// thermal net.
    #[must_use]
    pub fn terminal(&self, instance: &str, terminal: &str) -> Option<(Node, bool)> {
        let index = self.index.get_or_init(|| Index::new(&self.system));
        let devices = &self.system.devices;
        if let Ok(found) = index
            .devices
            .binary_search_by(|&device| devices[device].path().cmp(instance))
        {
            let device = &devices[index.devices[found]];
            let position = device
                .terminals()
                .iter()
                .position(|name| name == terminal)?;
            return Some((
                device.nodes()[position],
                device.thermal().contains(&position),
            ));
        }
        let name = format!("{instance}.{terminal}");
        let composites = &self.system.composites;
        let found = index
            .composites
            .binary_search_by(|&composite| composites[composite].name.cmp(&name))
            .ok()?;
        let composite = &composites[index.composites[found]];
        Some((composite.node, composite.thermal))
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
    /// A transient's results would hold more numbers than its options let it keep.
    #[error(
        "the transient's results would pass {memory:?} GiB before it stops, its steps \
         {step:?} s long at t = {time:?} s"
    )]
    TooManyPoints {
        /// The time the analysis reached, in seconds.
        time: f64,
        /// The length of its steps there, in seconds.
        step: f64,
        /// The gibibytes its results may hold: its `memory` option, less what a periodic
        /// steady state's dense matrices hold beside them.
        memory: f64,
    },
    /// An analysis's dense matrices would hold more numbers than its options let it.
    #[error(
        "the {analysis}'s dense matrices of {rows} rows would hold {needed:?} GiB, more than \
         its `memory` option's {memory:?} GiB"
    )]
    TooMuchMemory {
        /// The analysis, such as `pole-zero analysis`.
        analysis: &'static str,
        /// The rows of its matrices, one per unknown, and one more where it borders them.
        rows: usize,
        /// The gibibytes it would hold at once.
        needed: f64,
        /// The gibibytes it may hold, its `memory` option.
        memory: f64,
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
    /// Devices an analysis records cannot be made again, each terminal on a node of its own,
    /// to be read.
    #[error("cannot record {}", .0.join("; "))]
    Unrecordable(Vec<String>),
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
