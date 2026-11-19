//! Small-signal DC transfer function computing gain, input resistance, and output
//! resistance (matching SPICE `.tf`).
//!
//! Linearizes the circuit at the operating point into Jacobian `J` and runs two solves:
//!
//! - Input excitation (`J x = -∂f/∂s`, matching AC at 0 Hz) yields transfer gain and input
//!   resistance seen by the source.
//! - Output excitation yields output resistance with the input source held at its operating
//!   value: a unit current (`J x = e_out - e_ref`) for voltage outputs, or a unit series
//!   voltage (`J x = -∂f/∂s_out`) for voltage-source current outputs.
//!
//! Independent voltage sources act as shorts and current sources as opens in the second
//! solve, matching SPICE `TFanal`. Open-circuit resistances return exact infinity rather
//! than ngspice's `1e20` Ω cap.

use jolty_models::{Nature, UnknownIndex};
use serde::Deserialize;

use crate::ac::selector_vector;
use crate::{OperatingPointResult, Options, SimulationError, Simulator};

/// A transfer function analysis: from an independent source to an output, at DC.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransferFunction {
    /// The path of the input, an independent voltage or current source, such as `V1`.
    pub source: String,
    /// The output, a node voltage by its net's name, such as `out`, or a voltage source's
    /// current, such as `V2:current`.
    pub output: String,
    /// The node the output voltage is taken over, for a differential output; by default
    /// ground. A current output has none.
    #[serde(default)]
    pub reference: Option<String>,
}

/// The small-signal transfer function at DC.
#[derive(Debug, Clone, PartialEq)]
pub struct TransferFunctionResult {
    /// The path of the input source.
    pub source: String,
    /// Physical quantity of the input source (voltage or current).
    pub input: Nature,
    /// Output node or current unknown name.
    pub output: String,
    /// Optional negative reference net for differential voltage outputs.
    pub reference: Option<String>,
    /// Operating point around which the circuit was linearized.
    pub operating_point: OperatingPointResult,
    /// Small-signal transfer gain (output units per input unit).
    pub gain: f64,
    /// Small-signal resistance seen by the input source, in ohms.
    pub input_resistance: f64,
    /// Small-signal resistance seen at the output with the input source fixed, in ohms.
    pub output_resistance: f64,
}

impl Simulator {
    /// Computes the small-signal DC transfer function (gain, input resistance, and output
    /// resistance) for `analysis`.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownSource`] or [`SimulationError::NotASource`] for
    /// invalid sources, [`SimulationError::UnknownOutput`] for missing outputs or
    /// references, [`SimulationError::InvalidAnalysis`] for invalid output types or
    /// differential references, [`SimulationError::Singular`] when linearized equations
    /// are singular, and any error from [`Simulator::operating_point`].
    pub fn transfer_function(
        &mut self,
        analysis: &TransferFunction,
        options: &Options,
    ) -> Result<TransferFunctionResult, SimulationError> {
        options.parallelism.confined(|| {
            self.solvable()?;
            options.check()?;
            let input = self.source_index(&analysis.source)?;
            let output = self.output_of(analysis)?;
            let operating_point = self.linearized(options)?;
            let size = self.elaborated.unknowns.len();
            // Prepare all right-hand sides and indices before borrowing the solver.
            let mut driven = self.excitation(input);
            // Independent sources either constrain voltage or inject current.
            let seen = if self.source_nature(input) == Nature::Current {
                Input::Current {
                    terminals: self.terminal_unknowns(input),
                }
            } else {
                Input::Voltage {
                    branch: self.elaborated.devices[input].first_internal,
                }
            };
            let mut probe = match output {
                Output::Voltage { node, reference } => Some(selector_vector(size, node, reference)),
                Output::Current { device, .. } if device == input => None,
                Output::Current { device, .. } => Some(self.excitation(device)),
            };
            let mut factors = self
                .state
                .solver
                .factor(&self.state.workspace.contributions.conductance)?;
            factors.solve(&mut driven)?;
            let gain = output.read(&driven);
            let input_resistance = seen.resistance(&driven);
            let output_resistance = match (output, probe.as_mut()) {
                (_, None) => input_resistance,
                (Output::Voltage { .. }, Some(injected)) => {
                    factors.solve(injected)?;
                    output.read(injected)
                }
                (Output::Current { branch, .. }, Some(series)) => {
                    factors.solve(series)?;
                    resistance(1.0, -series[branch])
                }
            };
            Ok(TransferFunctionResult {
                source: analysis.source.clone(),
                input: self.source_nature(input),
                output: analysis.output.clone(),
                reference: analysis.reference.clone(),
                operating_point,
                gain,
                input_resistance,
                output_resistance,
            })
        })
    }

    /// Resolves and validates the requested output quantity for `analysis`.
    fn output_of(&self, analysis: &TransferFunction) -> Result<Output, SimulationError> {
        let (node, reference, nature) =
            self.measured(&analysis.output, analysis.reference.as_deref())?;
        match (nature, node) {
            (Nature::Voltage, node) => Ok(Output::Voltage { node, reference }),
            (Nature::Current, Some(unknown)) => {
                let index = unknown.index();
                // Devices without internal unknowns share `first_internal` with the next
                // device, so filter explicitly to independent voltage sources.
                let device = (0..self.elaborated.devices.len())
                    .find(|&device| {
                        self.elaborated.devices[device].first_internal == index
                            && self
                                .state
                                .runs
                                .dc_value(&self.elaborated.groups, device)
                                .is_some()
                            && self.source_nature(device) == Nature::Voltage
                    })
                    .ok_or_else(|| {
                        SimulationError::InvalidAnalysis(format!(
                            "a transfer function's current output is a voltage source's, \
                             not {}",
                            analysis.output
                        ))
                    })?;
                Ok(Output::Current {
                    device,
                    branch: index,
                })
            }
            (Nature::Current, None)
            | (Nature::Displacement | Nature::Temperature | Nature::State, _) => {
                Err(SimulationError::InvalidAnalysis(format!(
                    "a transfer function's output is a node voltage or a voltage source's \
                     current, not {}",
                    analysis.output
                )))
            }
        }
    }

    /// Returns the negated excitation vector `-∂f/∂s` for the independent source at `device`.
    fn excitation(&self, device: usize) -> Vec<f64> {
        let mut excitation = self.source_excitation(device);
        for value in &mut excitation {
            *value = -*value;
        }
        excitation
    }

    /// Returns unknown indices for the first two terminals of `device`, using `None` for
    /// ground.
    fn terminal_unknowns(&self, device: usize) -> [Option<usize>; 2] {
        let nodes = self.elaborated.devices[device].nodes();
        [0, 1].map(|terminal| {
            nodes
                .get(terminal)
                .copied()
                .flatten()
                .map(UnknownIndex::index)
        })
    }
}

/// Measurement handle for reading input resistance from the input-excited solution.
#[derive(Debug, Clone, Copy)]
enum Input {
    /// Voltage source measured via its branch current unknown.
    Voltage { branch: usize },
    /// Current source measured via the voltage across its two terminal unknowns.
    Current { terminals: [Option<usize>; 2] },
}

impl Input {
    /// Computes input resistance from the unit-excited solution `x`.
    fn resistance(self, x: &[f64]) -> f64 {
        match self {
            Self::Voltage { branch } => resistance(1.0, -x[branch]),
            Self::Current {
                terminals: [positive, negative],
            } => {
                let at = |node: Option<usize>| node.map_or(0.0, |node| x[node]);
                at(negative) - at(positive)
            }
        }
    }
}

/// Measurement handle for extracting the transfer-function output from a solution vector.
#[derive(Debug, Clone, Copy)]
enum Output {
    /// Node voltage relative to an optional negative reference net or ground.
    Voltage {
        node: Option<UnknownIndex>,
        reference: Option<UnknownIndex>,
    },
    /// Branch current through an independent voltage source.
    Current { device: usize, branch: usize },
}

impl Output {
    /// Extracts the output scalar from solution vector `x`.
    fn read(self, x: &[f64]) -> f64 {
        match self {
            Self::Voltage { node, reference } => {
                node.map_or(0.0, |at| x[at.index()]) - reference.map_or(0.0, |at| x[at.index()])
            }
            Self::Current { branch, .. } => x[branch],
        }
    }
}

/// Computes `voltage / current`, returning signed infinity with `voltage`'s sign when
/// `current` is zero.
fn resistance(voltage: f64, current: f64) -> f64 {
    if current == 0.0 {
        voltage.signum() * f64::INFINITY
    } else {
        voltage / current
    }
}
