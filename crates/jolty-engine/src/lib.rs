//! The simulation engine of jolty.
//!
//! A circuit is elaborated once into an [`Elaborated`] circuit, its hierarchy flattened into
//! devices bound to unknowns, the sparsity pattern of its matrix claimed and its sparse
//! solver's ordering found, and its devices' numbers listed in a table of parameters. It
//! never changes after, so threads share it. A [`Simulator`] is one run of it: the
//! [`Values`] it runs at, which replace some numbers and the temperature, and the state the
//! run writes, its devices made for those values among it. A simulator
//! [reset](Simulator::reset) for other values makes its devices again and keeps every
//! buffer, and runs as one made afresh would, bit for bit.
//!
//! A simulator runs any number of analyses: an [operating
//! point](Simulator::operating_point), a [DC sweep](Simulator::dc_sweep), a [small-signal
//! frequency response](Simulator::ac), a [transient](Simulator::transient),
//! [sensitivities](Simulator::sensitivity), the [loop gain](Simulator::stability) through a
//! probe, the [noise](Simulator::noise) at an output, and the [poles and
//! zeros](Simulator::pole_zero).
//! Every analysis solves the charge-oriented equations `f(x) + d/dt q(x) = 0` by Newton
//! iteration with one sparse LU solver, whose ordering all of them share, real and complex,
//! and which refactors each new Jacobian with the pivots of the last; every analysis about
//! the operating point takes the one the simulator solved last.
//! Logic devices join logic nets, which an event scheduler simulates beside the analog
//! circuit, joined to it by connect elements where a logic pin meets an electrical net.
//!
//! Many runs of one circuit, its Monte Carlo samples, corners, sweeps and candidate
//! designs, run as one [batch](batch::Batch) across the threads of the caller's rayon pool,
//! their values the [`variants`] of each run's index; a run spreads its own
//! frequencies and right-hand sides across threads as its options'
//! [parallelism](Parallelism) allows. Every number is the same bits however many threads
//! run it.
//!
//! ```
//! use jolty_engine::{Conditions, Elaborated, Options, Simulator, Structure, Values};
//! use jolty_ir::Circuit;
//! use jolty_models::Library;
//!
//! let circuit = Circuit::from_json(r#"{
//!     "format": "jolty-circuit", "version": 3, "top": "divider",
//!     "components": {"divider": {
//!         "ports": [{"name": "ground", "role": "ground"}],
//!         "instances": [
//!             {"name": "V1", "of": {"device": "voltage_source"},
//!              "parameters": {"waveform": {"dc": 5.0}},
//!              "terminals": {"positive": "top", "negative": "ground"}},
//!             {"name": "R1", "of": {"device": "resistor"},
//!              "parameters": {"resistance": 1000.0}, "terminals": {"a": "top", "b": "out"}},
//!             {"name": "R2", "of": {"device": "resistor"},
//!              "parameters": {"resistance": 1000.0}, "terminals": {"a": "out", "b": "ground"}}
//!         ]
//!     }}
//! }"#)?;
//! let mut simulator = Simulator::new(&circuit, &Library::standard(), &Conditions::default())?;
//! let solution = simulator.operating_point(&Options::default())?;
//! let out = simulator.unknown("out").expect("a net named out");
//! assert!((solution.values[out.index()] - 2.5).abs() < 1e-9);
//!
//! // Elaborated once, the divider runs at the values of each sample of its upper resistor.
//! let elaborated = Elaborated::new(&circuit, &Library::standard(), &Structure::default())?;
//! let upper = elaborated.parameter("R1", "resistance").expect("a resistance");
//! let mut values = Values::default();
//! values.set(upper, 3000.0);
//! let mut sample = Simulator::from_elaborated(elaborated, values)?;
//! let solution = sample.operating_point(&Options::default())?;
//! assert!((solution.values[out.index()] - 1.25).abs() < 1e-9);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod ac;
mod analysis;
pub mod batch;
mod elaborate;
mod elaborated;
#[cfg(test)]
mod kernels;
mod logic;
pub mod measure;
mod newton;
mod noise;
mod observe;
mod operating_point;
mod options;
mod periodic;
mod pole_zero;
mod profile;
mod sensitivity;
mod solver;
mod stability;
mod state;
mod sweep;
mod transfer;
mod transient;
mod values;
pub mod variants;

use std::collections::BTreeMap;
use std::sync::Arc;

use jolty_ir::{Circuit, Name, Parameters};
use jolty_models::{Level, Library, Node, REFERENCE_TEMPERATURE, Unknown, UnknownInfo};
use serde::Deserialize;

pub use ac::{Ac, AcResult};
pub use analysis::{Analysis, AnalysisResult};
pub use elaborate::{Composite, ElaborationError, ElaborationProblem, Location};
pub use elaborated::{Elaborated, Placed, Structure, StructureKey};
pub use jolty_numerics::integration::Method;
pub use logic::{Bound, Endpoint, Excursion, LogicNet, PlacedLogic};
pub use noise::{Noise, NoiseContribution, NoiseInput, NoiseResult};
pub use observe::{Observation, Observations, Observed, Recorded};
pub use operating_point::{OperatingPointResult, OperatingPoints, Strategy};
pub use options::{Options, Parallelism, Record};
pub use pole_zero::{Mode, PoleZero, PoleZeroResult};
pub use profile::{Profile, Work};
pub use sensitivity::{Derivatives, Differentiated, Parameter, Sensitivity, SensitivityResult};
pub use stability::{Stability, StabilityResult};
pub use sweep::{DcSweep, DcSweepResult};
pub use transfer::{TransferFunction, TransferFunctionResult};
pub use transient::{Statistics, Transient, TransientResult};
pub use values::{ParameterId, Values};

use state::State;

// An elaborated circuit is shared by every thread that runs it, and a simulator, holding a
// run's state, moves to the thread that runs it.
const _: () = {
    const fn shared<T: Send + Sync>() {}
    const fn sent<T: Send>() {}
    shared::<Elaborated>();
    sent::<State>();
    sent::<Simulator>();
};

/// What a circuit is simulated in, apart from its analyses: its devices' temperature,
/// parameters that replace the circuit's own for some device instances, as a Monte Carlo
/// sample's do, and models that stand in for some component instances, as a plan's
/// fidelity chooses.
///
/// The models that stand in are the circuit's [`Structure`], and the temperature and the
/// numbers the [`Values`] of a run of it; [`Simulator::new`] elaborates the circuit with
/// all of them, its parameters those the conditions give, for one run.
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

/// A run of an elaborated circuit, ready to run analyses: the circuit, the values it runs
/// at, and what the run writes, its devices made for those values among it.
///
/// Every analysis returns [`SimulationError::Foundry`] for a circuit that holds a foundry's
/// device, which ngspice simulates in the engine's place.
#[derive(Debug)]
pub struct Simulator {
    elaborated: Arc<Elaborated>,
    values: Values,
    state: State,
}

impl Simulator {
    /// Elaborates `circuit` with the models of `library`, in `conditions`, for one run: the
    /// parameters the conditions replace are the elaborated circuit's own.
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
        let replaced = (&conditions.substitutes, &conditions.overrides);
        let (elaborated, devices) =
            elaborate::elaborate(circuit, library, replaced, conditions.temperature)?;
        let state = State::new(&elaborated, devices, Vec::new(), conditions.temperature);
        Ok(Self {
            elaborated: Arc::new(elaborated),
            values: Values::default(),
            state,
        })
    }

    /// A run of `elaborated` at `values`: its devices made from the numbers the values
    /// give, each claiming the unknowns and entries the circuit holds. Where the values
    /// give a device others, as a resistance below 1 Ω carries its current as an unknown,
    /// the run is of the circuit [restructured](Elaborated::restructured) for them.
    ///
    /// # Errors
    ///
    /// Returns every [`ElaborationProblem`] the values make: each parameter the circuit
    /// does not have, and each device whose numbers they put out of range.
    pub fn from_elaborated(
        elaborated: Arc<Elaborated>,
        values: Values,
    ) -> Result<Self, ElaborationError> {
        if let Some(state) = State::of(&elaborated, &values)? {
            return Ok(Self {
                elaborated,
                values,
                state,
            });
        }
        let (restructured, devices, patched) = elaborated.rebind(&values)?;
        let temperature = restructured.temperature_of(&values);
        let state = State::new(&restructured, devices, patched, temperature);
        Ok(Self {
            elaborated: Arc::new(restructured),
            values,
            state,
        })
    }

    /// Readies the simulator to run at `values`, as one made [from its elaborated
    /// circuit](Simulator::from_elaborated) at them would: its instances prepared again for
    /// the numbers they give, and everything the runs before left forgotten, its buffers
    /// kept.
    ///
    /// # Errors
    ///
    /// Returns every [`ElaborationProblem`] the values make, and leaves the simulator as it
    /// was.
    pub fn reset(&mut self, values: Values) -> Result<(), ElaborationError> {
        if !self.state.reset(&self.elaborated, &values)? {
            let (restructured, devices, patched) = self.elaborated.rebind(&values)?;
            let temperature = restructured.temperature_of(&values);
            self.state = State::new(&restructured, devices, patched, temperature);
            self.elaborated = Arc::new(restructured);
        }
        self.values = values;
        Ok(())
    }

    /// Readies the simulator to run at `values`, as [`Simulator::reset`] does, but only on
    /// the circuit it runs: returns false, and leaves the simulator as it was, when the
    /// values give that circuit another structure.
    ///
    /// # Errors
    ///
    /// Returns every [`ElaborationProblem`] the values make, and leaves the simulator as it
    /// was.
    pub(crate) fn try_reset(&mut self, values: Values) -> Result<bool, ElaborationError> {
        if !self.state.reset(&self.elaborated, &values)? {
            return Ok(false);
        }
        self.values = values;
        Ok(true)
    }

    /// Readies the simulator to run at `values` from where its last run left it, as the
    /// next run of a chain: the instances whose numbers or temperature differ prepared
    /// again, every other instance as it is, and everything else forgotten as a reset forgets it,
    /// but for the sparse solver's pivots and the last operating point solved, which the
    /// next operating point starts from. Returns false, and leaves the simulator as it
    /// was, when the values give the circuit another structure, or the last run solved no
    /// operating point to start from.
    ///
    /// # Errors
    ///
    /// Returns every [`ElaborationProblem`] the values make, and leaves the simulator as it
    /// was.
    pub(crate) fn advance(&mut self, values: Values) -> Result<bool, ElaborationError> {
        let Some(solution) = self
            .state
            .remembered
            .as_ref()
            .map(|remembered| remembered.solution().to_vec())
        else {
            return Ok(false);
        };
        let from = std::mem::take(&mut self.values);
        let moved = self.state.move_to(&self.elaborated, &from, &values);
        if !matches!(moved, Ok(true)) {
            self.values = from;
            return moved;
        }
        self.state.forget(&self.elaborated);
        self.state.warm = Some(state::Warm {
            values: from,
            solution,
        });
        self.values = values;
        Ok(true)
    }

    /// Prepares again the instances whose numbers or temperature differ at `values` from
    /// the run's, as [`Simulator::advance`] does, and runs at `values` from now on,
    /// forgetting nothing. Returns false, and leaves the simulator as it was, when the
    /// values give the circuit another structure.
    ///
    /// # Errors
    ///
    /// Returns every [`ElaborationProblem`] the values make.
    pub(crate) fn move_to(&mut self, values: Values) -> Result<bool, ElaborationError> {
        if !self
            .state
            .move_to(&self.elaborated, &self.values, &values)?
        {
            return Ok(false);
        }
        self.values = values;
        Ok(true)
    }

    /// The elaborated circuit the simulator runs.
    #[must_use]
    pub fn elaborated(&self) -> &Arc<Elaborated> {
        &self.elaborated
    }

    /// The values the simulator runs at.
    #[must_use]
    pub fn values(&self) -> &Values {
        &self.values
    }

    /// How many times the simulator has solved for an operating point, or for a DC sweep's
    /// first level, since it was made. An analysis about the operating point the circuit
    /// was last solved at, with the same sources held and under the same options, save
    /// those only a transient reads, takes that solution again instead, and solves for
    /// none.
    #[must_use]
    pub fn operating_point_solves(&self) -> usize {
        self.state.solves
    }

    /// The terminals of every component instance and logic device on an electrical or a
    /// thermal net, such as `X1.vdd`, each with the device terminals whose currents and
    /// charges sum to its own. One on the reference, which is the ambient as well as the
    /// ground, sums its electrical terminals only.
    #[must_use]
    pub fn composites(&self) -> &[Composite] {
        &self.elaborated.composites
    }

    /// The node that the terminal `terminal` of the device, component instance or logic
    /// device at the path `instance` joins, such as `X1/R1` and `a`, and whether it is
    /// thermal: its voltage a temperature above the ambient, and its current the heat
    /// flowing into it. `None` when the instance has no such terminal on an electrical or a
    /// thermal net.
    #[must_use]
    pub fn terminal(&self, instance: &str, terminal: &str) -> Option<(Node, bool)> {
        self.elaborated.terminal(instance, terminal)
    }

    /// Starts the logic driver at `element` among the devices on a `(time, level,
    /// transition)` ramp; returns when the ramp ends.
    fn drive(
        &mut self,
        element: usize,
        (time, level, transition): (f64, Level, f64),
    ) -> Option<f64> {
        let groups = &self.elaborated.groups;
        self.state
            .runs
            .drive(groups, element, (time, level, transition))
    }

    /// The unknowns, in the order of every solution's values.
    #[must_use]
    pub fn unknowns(&self) -> &[UnknownInfo] {
        &self.elaborated.unknowns
    }

    /// The analog devices, with every component instance unfolded, in the order they were
    /// placed, then the connect elements of the logic devices' pins that join electrical
    /// nets.
    #[must_use]
    pub fn devices(&self) -> &[Placed] {
        &self.elaborated.devices
    }

    /// The paths of the foundry's devices the circuit holds, which no analysis solves, in
    /// the order they were placed.
    #[must_use]
    pub fn foundry_devices(&self) -> Vec<&str> {
        self.elaborated
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
        self.state.events.excursions()
    }

    /// The number of connect elements among [`Simulator::devices`], all at its end.
    #[must_use]
    pub fn elements(&self) -> usize {
        self.elaborated.logic.detectors.len() + self.elaborated.logic.drivers.len()
    }

    /// The logic devices, with every component instance unfolded, in the order they were
    /// placed.
    #[must_use]
    pub fn logic_devices(&self) -> &[PlacedLogic] {
        &self.elaborated.logic.devices
    }

    /// The logic nets, in the order of every result's levels: the nets only logic pins join,
    /// in the order they were first met, then the nets between logic pins and their connect
    /// elements.
    #[must_use]
    pub fn logic_nets(&self) -> &[LogicNet] {
        &self.elaborated.logic.nets
    }

    /// The name of the reference node, the net of the top component's ground port.
    #[must_use]
    pub fn reference(&self) -> &str {
        self.elaborated.reference.as_str()
    }

    /// The unknown named `name`: a net's path such as `amplifier/mid`, or an instance's path
    /// and label such as `V1:current`.
    #[must_use]
    pub fn unknown(&self, name: &str) -> Option<Unknown> {
        self.elaborated
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
    /// The circuit's equations are singular at every attempt, or a system made from them
    /// is: the circuit linearized at its operating point at a frequency, or the shooting
    /// equations of a periodic steady state.
    #[error(
        "the circuit's equations are singular{}: a node may have no DC path to the reference, \
         or sources may fix conflicting values",
        undetermined(.unknowns)
    )]
    Singular {
        /// Up to three of the unknowns the equations leave undetermined, as the sparse
        /// solver found them: a set of unknowns that appear in fewer equations than there
        /// are of them, the unknown whose column elimination left with no nonzero pivot, or
        /// those whose values in a solution are not finite. None when no unknown stands
        /// out, as when the pencil of a pole-zero analysis is singular at every frequency.
        unknowns: Vec<String>,
    },
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
    /// The pool of the threads a run's parallelism confines it to could not be made.
    #[error("the run's {threads} threads could not be started: {reason}")]
    Threads {
        /// The threads the run may use, its parallelism's `threads`.
        threads: usize,
        /// Why they could not be started.
        reason: String,
    },
    /// The circuit holds a foundry's devices, which ngspice simulates with the foundry's
    /// models, and which the engine does not solve.
    #[error(
        "the engine solves no foundry's device, which ngspice simulates with its foundry's \
         models: {}",
        .0.join(", ")
    )]
    Foundry(Vec<String>),
}

/// The clause of a singular error's message naming the `unknowns` its equations leave
/// undetermined, if any.
fn undetermined(unknowns: &[String]) -> String {
    if unknowns.is_empty() {
        String::new()
    } else {
        format!(", leaving {} undetermined", unknowns.join(", "))
    }
}
