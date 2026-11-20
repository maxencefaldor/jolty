//! Simulation engine for jolty.
//!
//! A circuit elaborates once into an immutable [`Elaborated`] representation that flattens
//! hierarchy into devices bound to unknowns, builds the matrix sparsity pattern and sparse
//! solver ordering, and indexes device parameters in a table. Threads share [`Elaborated`]
//! directly because it never mutates after construction. A [`Simulator`] executes runs
//! against an [`Elaborated`] circuit at specific [`Values`] (parameter overrides and
//! temperature), owning per-run devices and solver state. Calling
//! [`Simulator::reset`] rebuilds devices for new values while reusing buffers, producing
//! bit-identical results to a freshly constructed simulator.
//!
//! Supported analyses include [operating point](Simulator::operating_point), [operating
//! points](Simulator::operating_points), [DC sweep](Simulator::dc_sweep), [small-signal
//! frequency response](Simulator::ac), [transient](Simulator::transient), [periodic steady
//! state](Simulator::periodic_steady_state), [sensitivities](Simulator::sensitivity),
//! [mismatch](Simulator::mismatch), [loop gain](Simulator::stability),
//! [noise](Simulator::noise), [poles and zeros](Simulator::pole_zero), and [transfer
//! function](Simulator::transfer_function).
//! Every analysis solves the charge-oriented equations `f(x) + d/dt q(x) = 0` via Newton
//! iteration with a shared sparse LU ordering (real and complex) that reuses pivots from
//! previous factorizations. Linearized analyses reuse the most recently solved operating
//! point when conditions match.
//! Logic devices communicate across logic nets driven by an event scheduler running
//! alongside the analog solver, bridged by connect elements where logic pins meet
//! electrical nets.
//!
//! Multi-run studies (Monte Carlo samples, corners, sweeps, and design sweeps) execute as
//! a [batch](batch::Batch) across the caller's rayon pool using [`variants`] indexed by
//! run number. Individual runs can also parallelize frequencies and right-hand sides
//! according to [`Parallelism`]. All results remain bit-identical regardless of thread
//! count.
//!
//! ```
//! use jolty_engine::{Conditions, Elaborated, Options, Simulator, Structure, Values};
//! use jolty_ir::Circuit;
//! use jolty_models::Library;
//!
//! let circuit = Circuit::from_json(r#"{
//!     "format": "jolty-circuit", "version": 5, "top": "divider",
//!     "components": {"divider": {
//!         "ports": [{"name": "ground", "role": "ground"}],
//!         "instances": [
//!             {"name": "V1", "of": {"model": "voltage_source"},
//!              "parameters": {"waveform": {"dc": 5.0}},
//!              "terminals": {"positive": "top", "negative": "ground"}},
//!             {"name": "R1", "of": {"model": "resistor"},
//!              "parameters": {"resistance": 1000.0}, "terminals": {"a": "top", "b": "out"}},
//!             {"name": "R2", "of": {"model": "resistor"},
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
mod derivative;
mod elaborate;
mod elaborated;
#[cfg(test)]
mod kernels;
mod logic;
pub mod measure;
mod mismatch;
mod newton;
mod noise;
mod objective;
mod operating_point;
mod options;
mod periodic;
mod pole_zero;
mod profile;
mod record;
mod ringing;
mod sensitivity;
mod solver;
mod stability;
mod state;
mod sweep;
mod transfer;
mod transient;
mod twins;
mod values;
pub mod variants;

use std::collections::BTreeMap;
use std::sync::Arc;

use jolty_ir::{Circuit, Name, Parameters};
use jolty_models::{Library, LogicLevel, Node, REFERENCE_TEMPERATURE, Unknown, UnknownIndex};
use serde::Deserialize;

pub use ac::{Ac, AcResult};
pub use analysis::{Analysis, AnalysisResult};
pub use derivative::{Cotangent, Cotangents, JvpResult, ParameterCotangents, Pullback};
pub use elaborate::{ElaborationError, ElaborationProblem, Location, SummedTerminal};
pub use elaborated::{Elaborated, Partition, Placed, Structure, StructureKey};
pub use jolty_numerics::integration::Method;
pub use logic::{Endpoint, Excursion, ExcursionKind, LogicNet, PlacedLogic};
pub use mismatch::{Mismatch, MismatchResult, RandomVariable};
pub use noise::{NOISE_DENSITY, Noise, NoiseContribution, NoiseInput, NoiseResult, REFERRED_NOISE};
pub use objective::{Functional, Objective, ObjectiveResult};
pub use operating_point::{OperatingPointResult, OperatingPoints, Strategy};
pub use options::{BREAK_EVENS, BreakEvens, Options, Parallelism, Record, Sizes};
pub use periodic::{PeriodicSteadyState, PeriodicSteadyStateResult};
pub use pole_zero::{Mode, PoleZero, PoleZeroResult};
pub use profile::{Profile, Work};
pub use record::{RecordedSeries, RecordedValues, RecordedView, Recording};
pub use sensitivity::{Derivatives, Differentiated, Parameter, Sensitivity, SensitivityResult};
pub use stability::{LOOP_GAIN, Stability, StabilityResult};
pub use sweep::{DcSweep, DcSweepResult};
pub use transfer::{TransferFunction, TransferFunctionResult};
pub use transient::{ChargeLeak, Statistics, Transient, TransientResult};
pub use values::{ParameterId, Values};

use state::State;

/// Remote `serde` deserialization specification for [`Method`].
#[derive(Deserialize)]
#[serde(remote = "Method", rename_all = "snake_case")]
pub(crate) enum MethodSerde {
    BackwardEuler,
    Trapezoidal,
    Gear2,
}

// An elaborated circuit is shared by every thread that runs it, and a simulator, holding a
// run's state, moves to the thread that runs it.
const _: () = {
    const fn shared<T: Send + Sync>() {}
    const fn sent<T: Send>() {}
    shared::<Elaborated>();
    sent::<State>();
    sent::<Simulator>();
};

/// Simulation environment for a single run: operating temperature, per-instance parameter
/// overrides, and component model substitutes.
///
/// Substitutes define the circuit [`Structure`], while temperature and numeric overrides
/// define run [`Values`]. [`Simulator::new`] elaborates the circuit with all of these
/// applied for a single run.
#[derive(Debug, Clone, PartialEq)]
pub struct Conditions {
    /// Device operating temperature in kelvins.
    pub temperature: f64,
    /// Parameter overrides indexed by instance path (such as `X1/R1`) and parameter
    /// [path](jolty_ir::parameter()), such as `waveform.dc`.
    pub overrides: BTreeMap<String, Parameters>,
    /// Model substitutes replacing component instances, indexed by instance path (such as
    /// `adder/U1`).
    pub substitutes: BTreeMap<String, Substitute>,
}

/// Model substituting for a component instance, mapping component ports onto device
/// terminals connected to the same nets.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Substitute {
    /// Model name, such as `logic_cell`.
    pub model: Name,
    /// Device parameters for the substitute model.
    pub parameters: Parameters,
}

impl Conditions {
    /// Creates conditions at `temperature` in kelvins with no overrides or substitutes.
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
    /// Returns default conditions at 27 °C with no overrides or substitutes.
    fn default() -> Self {
        Self::at(REFERENCE_TEMPERATURE)
    }
}

/// Simulator instance holding an elaborated circuit, active parameter values, and mutable
/// solver and device state.
///
/// Every analysis returns [`SimulationError::Foundry`] when the circuit contains foundry
/// devices delegated to ngspice.
#[derive(Debug)]
pub struct Simulator {
    elaborated: Arc<Elaborated>,
    values: Values,
    state: State,
}

impl Simulator {
    /// Elaborates `circuit` with models from `library` under `conditions` for a single run,
    /// baking condition overrides into the elaborated circuit.
    ///
    /// # Errors
    ///
    /// Returns every [`ElaborationProblem`] found: invalid structure, missing ground,
    /// unknown model, invalid terminal connections, invalid parameters, or overrides
    /// targeting nonexistent devices.
    pub fn new(
        circuit: &Circuit,
        library: &Library,
        conditions: &Conditions,
    ) -> Result<Self, ElaborationError> {
        let replaced = elaborate::Replaced {
            substitutes: &conditions.substitutes,
            overrides: &conditions.overrides,
        };
        let (elaborated, devices) =
            elaborate::elaborate(circuit, library, replaced, conditions.temperature)?;
        let state = State::new(&elaborated, devices, Vec::new(), conditions.temperature);
        Ok(Self {
            elaborated: Arc::new(elaborated),
            values: Values::default(),
            state,
        })
    }

    /// Creates a simulator from `elaborated` at `values`, instantiating devices for those
    /// parameter values.
    ///
    /// When `values` alter device unknowns or matrix entries (such as a resistor dropping
    /// below 1 Ω and adding a branch current), the simulator builds on the
    /// [restructured](Elaborated::restructured) circuit.
    ///
    /// # Errors
    ///
    /// Returns every [`ElaborationProblem`] triggered by `values`, including unknown
    /// parameters and out-of-range device parameters.
    pub fn from_elaborated(
        elaborated: Arc<Elaborated>,
        values: Values,
    ) -> Result<Self, ElaborationError> {
        if let Some(state) = State::from_values(&elaborated, &values)? {
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

    /// Resets the simulator to `values`, rebuilding devices and clearing cached solutions
    /// while reusing buffers to match [`Simulator::from_elaborated`].
    ///
    /// # Errors
    ///
    /// Returns every [`ElaborationProblem`] triggered by `values` without modifying the
    /// simulator.
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

    /// Resets the simulator to `values` in place like [`Simulator::reset`], returning
    /// `false` without modification if `values` require restructuring.
    ///
    /// # Errors
    ///
    /// Returns every [`ElaborationProblem`] triggered by `values` without modifying the
    /// simulator.
    pub(crate) fn try_reset(&mut self, values: Values) -> Result<bool, ElaborationError> {
        if !self.state.reset(&self.elaborated, &values)? {
            return Ok(false);
        }
        self.values = values;
        Ok(true)
    }

    /// Advances the simulator to `values` as the next step in a warm-started chain.
    ///
    /// Re-prepares only devices whose parameters or temperature changed, preserving sparse
    /// solver pivots and the last operating point as a warm start. Returns `false` without
    /// modification if `values` require restructuring or if no prior operating point exists.
    ///
    /// # Errors
    ///
    /// Returns every [`ElaborationProblem`] triggered by `values` without modifying the
    /// simulator.
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

    /// Updates changed devices to `values` like [`Simulator::advance`] without clearing
    /// cached state, returning `false` if restructuring is required.
    ///
    /// # Errors
    ///
    /// Returns every [`ElaborationProblem`] triggered by `values`.
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

    /// Returns the elaborated circuit backing this simulator.
    #[must_use]
    pub fn elaborated(&self) -> &Arc<Elaborated> {
        &self.elaborated
    }

    /// Returns the active parameter and temperature values of this simulator.
    #[must_use]
    pub fn values(&self) -> &Values {
        &self.values
    }

    /// Returns the number of operating-point solves (including initial DC sweep points)
    /// performed since construction.
    ///
    /// Subsequent analyses matching the cached operating point, source settings, and
    /// non-transient options reuse that solution without re-solving.
    #[must_use]
    pub fn operating_point_solves(&self) -> usize {
        self.state.solves
    }

    /// Returns summed terminals for component instances and logic devices on electrical or
    /// thermal nets (such as `X1.vdd`).
    ///
    /// Each entry aggregates currents and charges of its constituent device terminals; a
    /// terminal on the shared ground/ambient reference aggregates electrical terminals only.
    #[must_use]
    pub fn summed_terminals(&self) -> &[SummedTerminal] {
        &self.elaborated.summed_terminals
    }

    /// Resolves `terminal` of `instance` (such as `X1/R1` and `a`) to `(node, is_thermal)`.
    ///
    /// For thermal nodes, voltage represents temperature rise above ambient and current
    /// represents heat flow into the terminal. Returns `None` when no matching electrical
    /// or thermal terminal exists.
    #[must_use]
    pub fn terminal(&self, instance: &str, terminal: &str) -> Option<(Node, bool)> {
        self.elaborated.terminal(instance, terminal)
    }

    /// Starts the logic driver at device index `element` ramping to `level` from `time`
    /// over `transition`, returning the ramp completion time.
    fn drive(
        &mut self,
        element: usize,
        time: f64,
        level: LogicLevel,
        transition: f64,
    ) -> Option<f64> {
        let groups = &self.elaborated.groups;
        self.state
            .runs
            .drive(groups, element, (time, level, transition))
    }

    /// Returns all circuit unknowns in solution vector order.
    #[must_use]
    pub fn unknowns(&self) -> &[Unknown] {
        &self.elaborated.unknowns
    }

    /// Returns unfolded analog devices in placement order, followed by logic connect
    /// elements attached to electrical nets.
    #[must_use]
    pub fn devices(&self) -> &[Placed] {
        &self.elaborated.devices
    }

    /// Returns paths of foundry devices delegated to ngspice, in placement order.
    #[must_use]
    pub fn foundry_devices(&self) -> Vec<&str> {
        self.elaborated
            .devices
            .iter()
            .filter(|placed| placed.foundry())
            .map(Placed::path)
            .collect()
    }

    /// Checks that the engine can solve every device in the circuit.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Foundry`] listing any foundry devices present.
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

    /// Returns logic-device domain excursions recorded during the last analysis, one entry
    /// per `(device, bound)` with peak excursion and first occurrence time.
    #[must_use]
    pub fn excursions(&self) -> Vec<Excursion> {
        self.state.events.excursions()
    }

    /// Returns the number of logic connect elements appended at the end of
    /// [`Simulator::devices`].
    #[must_use]
    pub fn elements(&self) -> usize {
        self.elaborated.logic.detectors.len() + self.elaborated.logic.drivers.len()
    }

    /// Returns unfolded logic devices in placement order.
    #[must_use]
    pub fn logic_devices(&self) -> &[PlacedLogic] {
        &self.elaborated.logic.devices
    }

    /// Returns logic nets in result logic-level order: pure logic nets first, followed by
    /// nets bridging logic pins to connect elements.
    #[must_use]
    pub fn logic_nets(&self) -> &[LogicNet] {
        &self.elaborated.logic.nets
    }

    /// Returns the ground net name declared by the top component's ground port.
    #[must_use]
    pub fn reference(&self) -> &str {
        self.elaborated.reference.as_str()
    }

    /// Looks up the [`UnknownIndex`] named `name` (such as net path `amplifier/mid` or
    /// branch label `V1:current`).
    #[must_use]
    pub fn unknown(&self, name: &str) -> Option<UnknownIndex> {
        self.elaborated
            .unknowns
            .iter()
            .position(|unknown| unknown.name == name)
            .map(UnknownIndex::new)
    }
}

/// Reason a transient analysis rejected a time step.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Rejection {
    /// Newton iteration did not converge.
    #[error("Newton iteration did not converge; furthest from converging: {}", .worst.join(", "))]
    Unconverged {
        /// Unknowns furthest from converging at the final iteration, worst first.
        worst: Vec<String>,
    },
    /// Estimated local truncation error exceeded tolerance.
    #[error("the truncation error of `{0}` exceeded its tolerance")]
    Inaccurate(String),
}

/// Failure returned by a simulation analysis.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SimulationError {
    /// No operating point was found, even with gmin and source stepping.
    #[error("no operating point found; furthest from converging: {}", .worst.join(", "))]
    NoOperatingPoint {
        /// Unknowns furthest from converging on the final attempt, worst first.
        worst: Vec<String>,
    },
    /// Singular circuit equations, linearized small-signal system, or periodic shooting
    /// equations.
    #[error(
        "the circuit's equations are singular{}: a node may have no DC path to the reference, \
         or sources may fix conflicting values",
        undetermined(.unknowns)
    )]
    Singular {
        /// Up to three undetermined unknowns reported by the sparse solver (structural
        /// under-determination, zero pivot column, or non-finite solution entries). Empty
        /// when no specific unknown stands out, such as an identically singular pole-zero
        /// pencil.
        unknowns: Vec<String>,
    },
    /// Circuit system exceeds sparse factorization size limits.
    #[error("the circuit's system is too large to factor")]
    TooLarge,
    /// Transient analysis reduced the time step below the allowed minimum.
    #[error("the time step fell below {minimum:e} s at t = {time:e} s: {rejection}")]
    StepTooSmall {
        /// Simulation time at which the solver stalled, in seconds.
        time: f64,
        /// Minimum permitted time step, in seconds.
        minimum: f64,
        /// Reason the final step was rejected.
        rejection: Rejection,
    },
    /// Transient waveform storage would exceed configured memory budget.
    #[error(
        "the transient's results would pass {max_memory:?} GiB before it stops, its steps \
         {step:?} s long at t = {time:?} s"
    )]
    TooManyPoints {
        /// Simulation time reached before aborting, in seconds.
        time: f64,
        /// Step size at the point of failure, in seconds.
        step: f64,
        /// Available memory budget in gibibytes (`max_memory` minus dense matrix storage
        /// held by periodic steady state).
        max_memory: f64,
    },
    /// Dense matrices required by an analysis would exceed configured memory budget.
    #[error(
        "the {analysis}'s dense matrices of {rows} rows would hold {needed:?} GiB, more than \
         its `max_memory` option's {max_memory:?} GiB"
    )]
    TooMuchMemory {
        /// Name of the analysis, such as `pole-zero analysis`.
        analysis: &'static str,
        /// Dense matrix dimension (one per unknown, plus one when bordered).
        rows: usize,
        /// Estimated memory requirement in gibibytes.
        needed: f64,
        /// Configured `max_memory` limit in gibibytes.
        max_memory: f64,
    },
    /// Sensitivity analysis references an unknown name not present in the circuit.
    #[error("there is no unknown named `{0}`")]
    UnknownOutput(String),
    /// Analysis references a source instance path not present in the circuit.
    #[error("there is no instance at `{0}`")]
    UnknownSource(String),
    /// Derivative or mismatch analysis references device parameters that do not exist as
    /// real-valued numbers, listed as `(device_path, parameter_path)`.
    #[error("no device holds a real number at {}", numbered(.0))]
    UnknownParameters(Vec<(String, String)>),
    /// Analysis references an instance that is not an independent source.
    #[error("`{0}` is not an independent source")]
    NotASource(String),
    /// Stability analysis probe is not a floating independent voltage source.
    #[error(
        "`{0}` is not a floating voltage source (probe requires a voltage source not connected \
         across ground)"
    )]
    NotAProbe(String),
    /// Analysis parameters are invalid or out of range.
    #[error("invalid analysis: {0}")]
    InvalidAnalysis(String),
    /// QZ eigenvalue iteration for pole-zero analysis failed to converge.
    #[error("the eigenvalues of the linearized circuit did not converge")]
    Unconverged,
    /// Periodic steady-state shooting failed to converge within the iteration limit.
    #[error(
        "no periodic steady state found in {periods} periods; furthest from repeating: {}",
        .worst.join(", ")
    )]
    NoSteadyState {
        /// Number of shooting periods simulated.
        periods: usize,
        /// Unknowns furthest from periodicity on the final period, worst first.
        worst: Vec<String>,
    },
    /// Linearized autonomous circuit has no growing mode to initiate oscillation.
    #[error(
        "no natural mode of the circuit linearized at its operating point grows, so no \
         oscillation starts from it"
    )]
    NoGrowingMode,
    /// Autonomous shooting converged to a stationary equilibrium rather than an
    /// oscillation.
    #[error(
        "shooting found a state that does not move: no unknown changes by its tolerance as \
         the period changes by its relative tolerance, so it is an equilibrium the \
         oscillation left, not an oscillation"
    )]
    Stationary,
    /// Autonomous oscillator startup transient failed to settle into a bounded orbit.
    #[error(
        "the oscillation from the operating point, kicked along its fastest-growing mode, \
         did not stop growing in {time:e} s: none of its {returns} returns to the hyperplane \
         through the operating point across the kick converged closer to the preceding \
         crossing"
    )]
    Unsettled {
        /// Elapsed startup transient duration, in seconds.
        time: f64,
        /// Number of Poincaré hyperplane crossings observed during startup.
        returns: usize,
    },
    /// Analysis does not support circuits containing transmission-line delay elements.
    #[error("a {analysis} takes no delay line, as {reason}: {}", .devices.join(", "))]
    Delayed {
        /// Name of the analysis, such as `pole-zero analysis`.
        analysis: &'static str,
        /// Explanation of why delay lines are unsupported.
        reason: &'static str,
        /// Paths of the delay-line instances present in the circuit.
        devices: Vec<String>,
    },
    /// Simulation options contain invalid or out-of-range values.
    #[error("invalid options: {0}")]
    InvalidOptions(String),
    /// Worker thread pool configured by parallelism options could not be spawned.
    #[error("the run's {threads} threads could not be started: {reason}")]
    Threads {
        /// Requested worker thread count from parallelism options.
        threads: usize,
        /// Operating-system error message describing thread creation failure.
        reason: String,
    },
    /// Circuit contains foundry devices delegated to ngspice rather than solved natively.
    #[error(
        "the engine solves no foundry's device, which ngspice simulates with its foundry's \
         models: {}",
        .0.join(", ")
    )]
    Foundry(Vec<String>),
}

/// Formats `(device_path, parameter_path)` pairs for error messages, such as
/// `` `X1/R1` `resistance` ``.
fn numbered(numbers: &[(String, String)]) -> String {
    numbers
        .iter()
        .map(|(path, name)| format!("`{path}` `{name}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Formats the optional undetermined-unknowns clause for singular-system error messages.
fn undetermined(unknowns: &[String]) -> String {
    if unknowns.is_empty() {
        String::new()
    } else {
        format!(", leaving {} undetermined", unknowns.join(", "))
    }
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use jolty_ir::Circuit;
    use jolty_models::Library;
    use serde_json::{Value, json};

    use crate::{Conditions, Simulator};

    /// Parses a circuit from `json` and elaborates it with the standard library and defaults.
    pub(crate) fn simulator_from_json(json: &str) -> Simulator {
        let circuit = Circuit::from_json(json).expect("a valid circuit");
        Simulator::new(&circuit, &Library::standard(), &Conditions::default())
            .expect("the circuit elaborates")
    }

    /// Elaborates a single-component grounded test circuit named `top` with ground `ground`
    /// and `instances`.
    pub(crate) fn grounded_simulator(top: &str, ground: &str, instances: &Value) -> Simulator {
        let document = json!({
            "format": "jolty-circuit",
            "version": jolty_ir::VERSION,
            "top": top,
            "components": {
                top: {
                    "ports": [{"name": ground, "role": "ground"}],
                    "instances": instances,
                }
            }
        });
        simulator_from_json(&document.to_string())
    }

    /// Builds a minimal 1 V DC source driving a 1 kΩ resistor to ground (`"0"`).
    pub(crate) fn source_resistor_simulator() -> Simulator {
        grounded_simulator(
            "test",
            "0",
            &json!([
                {"name": "V1", "of": {"model": "voltage_source"},
                 "parameters": {"waveform": {"dc": 1.0}},
                 "terminals": {"positive": "in", "negative": "0"}},
                {"name": "R1", "of": {"model": "resistor"},
                 "parameters": {"resistance": 1000.0},
                 "terminals": {"a": "in", "b": "0"}}
            ]),
        )
    }
}
