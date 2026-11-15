//! The loop gain of a feedback loop, by Middlebrook's double injection.
//!
//! A voltage source in series with the loop, the probe, is where the loop is broken. Its
//! `negative` terminal faces the side that drives the loop's signal into it, `y`, and its
//! `positive` terminal the side the signal enters, `x`. Two small-signal solves at each
//! frequency share one factorization: a unit voltage injected in series, `v_x − v_y = 1`,
//! and a unit current injected from the reference into `x`, of which the probe carries `i`
//! on to `y`. With `u = v_x − i`, the loop gain is
//!
//! ```text
//! T = −u / (1 + u),
//! ```
//!
//! which is Middlebrook's `T = (T_v T_i − 1) / (T_v + T_i + 2)`, where `T_v = −v_y / v_x`
//! and `T_i = i / (1 − i)`, cleared of their fractions. It holds however the two sides
//! load each other (R. D. Middlebrook, "Measurement of loop gain in feedback systems",
//! *International Journal of Electronics* 38(4), 1975), and is the loop gain Tian's method
//! and Spectre's `stb` analysis find. The injected current returns through the reference,
//! which must be the two sides' common return, as a ground is, and so neither of the
//! probe's terminals. Negative feedback makes `T` positive at low frequencies, and divides
//! each response it encloses by `1 + T`.

use jolty_models::Quantity;
use jolty_numerics::Complex64;
use serde::Deserialize;

use crate::ac::{Linearization, over_frequencies};
use crate::{OperatingPointResult, Options, SimulationError, Simulator};

/// A stability analysis: the loop gain through a probe, at each frequency.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stability {
    /// The path of the probe, an independent voltage source in series with the loop, its
    /// `positive` terminal toward the side the loop's signal enters, and neither terminal
    /// on the reference.
    pub probe: String,
    /// The frequencies, in hertz.
    pub frequencies: Vec<f64>,
}

/// The loop gain through a probe, at each frequency.
#[derive(Debug, Clone, PartialEq)]
pub struct StabilityResult {
    /// The path of the probe.
    pub probe: String,
    /// The operating point the circuit was linearized at.
    pub operating_point: OperatingPointResult,
    /// The frequencies, in hertz.
    pub frequencies: Vec<f64>,
    /// The loop gain `T` at each frequency, positive at low frequencies under negative
    /// feedback.
    pub gains: Vec<Complex64>,
}

impl Simulator {
    /// Linearizes the circuit at its operating point and finds the loop gain through the
    /// probe at each frequency.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownSource`] or [`SimulationError::NotAProbe`] when the
    /// probe is not an independent voltage source with both terminals off the reference,
    /// [`SimulationError::InvalidAnalysis`] for
    /// no frequency or a negative one, [`SimulationError::Singular`] when the linearized
    /// circuit is singular at a frequency, and the errors of [`Simulator::operating_point`].
    pub fn stability(
        &mut self,
        stability: &Stability,
        options: &Options,
    ) -> Result<StabilityResult, SimulationError> {
        options.parallelism.confined(|| {
            self.solvable()?;
            let not_a_probe = || SimulationError::NotAProbe(stability.probe.clone());
            let index = self
                .source_index(&stability.probe)
                .map_err(|error| match error {
                    SimulationError::NotASource(_) => not_a_probe(),
                    other => other,
                })?;
            if self.source_quantity(index) != Quantity::Voltage {
                return Err(not_a_probe());
            }
            let &[Some(entered), Some(_)] = self.elaborated.devices[index].nodes() else {
                return Err(not_a_probe());
            };
            let carried = self
                .unknown(&format!("{}:current", stability.probe))
                .ok_or_else(not_a_probe)?;
            let unknowns = self.elaborated.unknowns.len();
            let mut level = vec![0.0; unknowns];
            self.state
                .runs
                .excite(&self.elaborated.groups, index, &mut level);
            let operating_point = self.linearize(&stability.frequencies, options)?;
            let gains = over_frequencies(
                Linearization::new(&self.state.solver),
                &self.state.workspace.contributions,
                &stability.frequencies,
                &options.parallelism,
                &|factors, _, _| {
                    let mut series: Vec<Complex64> =
                        level.iter().map(|&d| Complex64::new(-d, 0.0)).collect();
                    factors.solve(&mut series)?;
                    let mut shunt = vec![Complex64::default(); unknowns];
                    shunt[entered.index()] = Complex64::new(1.0, 0.0);
                    factors.solve(&mut shunt)?;
                    let u = series[entered.index()] - shunt[carried.index()];
                    Ok(-u / (1.0 + u))
                },
            )?;
            Ok(StabilityResult {
                probe: stability.probe.clone(),
                operating_point,
                frequencies: stability.frequencies.clone(),
                gains,
            })
        })
    }
}
