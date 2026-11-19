//! The small-signal transfer function at DC: a gain, an input resistance and an output
//! resistance, as SPICE's `.tf` reports them.
//!
//! The circuit is linearized at its operating point, `J`, and solved twice:
//!
//! - excited by its input source, `J x = −∂f/∂s`, as a small-signal response is at 0 Hz,
//!   which gives the gain at the output and the input resistance the source sees;
//! - excited at its output, which gives the resistance the output sees with the input
//!   source at its operating value: a unit current injected into the output node and drawn
//!   from its reference, `J x = e_out − e_ref`, for a voltage; a volt in series with the
//!   voltage source whose current is the output, `J x = −∂f/∂s_out`, for a current.
//!
//! An independent voltage source holds its voltage, so it is a short to the second solve,
//! and a current source is open to it, as SPICE's `TFanal` has them. A resistance the
//! solves find infinite, such as a current source's own, is infinite, where ngspice writes
//! 10²⁰ Ω.

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
    /// What the input source's value is: a voltage, or a current.
    pub input: Nature,
    /// The output.
    pub output: String,
    /// The node the output voltage is taken over, if any.
    pub reference: Option<String>,
    /// The operating point the circuit was linearized at.
    pub operating_point: OperatingPointResult,
    /// The output per unit of the source's value: volts or amperes of output per volt or
    /// ampere of input.
    pub gain: f64,
    /// The resistance the input source sees, in ohms.
    pub input_resistance: f64,
    /// The resistance the output sees, in ohms, with the input source at its operating
    /// value.
    pub output_resistance: f64,
}

impl Simulator {
    /// The small-signal transfer function from `analysis`'s source to its output, at DC.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownSource`] or [`SimulationError::NotASource`] when
    /// the input or a current output's device is not an independent source,
    /// [`SimulationError::UnknownOutput`] for an output or a reference that is no unknown,
    /// [`SimulationError::InvalidAnalysis`] for a current output that is no voltage
    /// source's, for an output taken over itself, or for an output over a reference that is
    /// not a voltage, [`SimulationError::Singular`] when the linearized circuit is, and the
    /// errors of [`Simulator::operating_point`].
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
            // Every right side and every index the solves read, before the factors borrow the
            // solver.
            let mut driven = self.excitation(input);
            // An independent source holds a voltage or drives a current.
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

    /// The output `analysis` names, checked.
    fn output_of(&self, analysis: &TransferFunction) -> Result<Output, SimulationError> {
        let (node, reference, nature) =
            self.measured(&analysis.output, analysis.reference.as_deref())?;
        match (nature, node) {
            (Nature::Voltage, node) => Ok(Output::Voltage { node, reference }),
            (Nature::Current, Some(unknown)) => {
                let index = unknown.index();
                // A device that adds no unknown of its own counts the same unknowns before
                // it as the next device does, so the source is sought among sources alone.
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

    /// `−∂f/∂s` of the source at `device`: the right side of a solve it excites.
    fn excitation(&self, device: usize) -> Vec<f64> {
        let mut excitation = self.source_excitation(device);
        for value in &mut excitation {
            *value = -*value;
        }
        excitation
    }

    /// The unknowns of the first two terminals of the device at `device`, none for
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

/// What a transfer function reads its input resistance from.
#[derive(Debug, Clone, Copy)]
enum Input {
    /// A voltage source, by the current through it, its branch unknown.
    Voltage { branch: usize },
    /// A current source, by the voltage across it, its terminals' unknowns.
    Current { terminals: [Option<usize>; 2] },
}

impl Input {
    /// The resistance the source sees in a solution `x` the source drove with one volt or
    /// ampere: the volt over the current it drives out of its positive terminal, or the
    /// voltage the ampere it drives develops across it.
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

/// What a transfer function reads its output from.
#[derive(Debug, Clone, Copy)]
enum Output {
    /// A node voltage, over another node or the reference.
    Voltage {
        node: Option<UnknownIndex>,
        reference: Option<UnknownIndex>,
    },
    /// The current through an independent voltage source, its branch unknown.
    Current { device: usize, branch: usize },
}

impl Output {
    /// The output's value in a solution `x`.
    fn read(self, x: &[f64]) -> f64 {
        match self {
            Self::Voltage { node, reference } => {
                node.map_or(0.0, |at| x[at.index()]) - reference.map_or(0.0, |at| x[at.index()])
            }
            Self::Current { branch, .. } => x[branch],
        }
    }
}

/// The resistance across which `voltage` drives `current`: infinite, of the voltage's
/// sign, where no current flows, whichever sign of zero the solve left it.
fn resistance(voltage: f64, current: f64) -> f64 {
    if current == 0.0 {
        voltage.signum() * f64::INFINITY
    } else {
        voltage / current
    }
}
