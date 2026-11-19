//! Feedback loop gain computed via Middlebrook's double-injection method.
//!
//! A series floating voltage source acts as the loop-breaking probe. The probe's `negative`
//! terminal faces the driving side `y`, and the `positive` terminal faces the driven side
//! `x`. Two small-signal solves at each frequency share one factorization: a unit series
//! voltage (`v_x - v_y = 1`) and a unit shunt current injected from ground into `x` (of
//! which the probe carries `i` to `y`). Setting `u = v_x - i` gives the loop gain
//!
//! ```text
//! T = -u / (1 + u),
//! ```
//!
//! equivalent to Middlebrook's `T = (T_v T_i - 1) / (T_v + T_i + 2)` with
//! `T_v = -v_y / v_x` and `T_i = i / (1 - i)`. The identity holds regardless of mutual
//! loading (R. D. Middlebrook, "Measurement of loop gain in feedback systems",
//! *International Journal of Electronics* 38(4), 1975) and matches Tian's method and
//! Spectre `stb`. The injected current returns through the ground reference, so neither
//! probe terminal may sit on ground. Negative feedback yields positive `T` at low
//! frequencies and divides enclosed closed-loop responses by `1 + T`.

use jolty_models::Nature;
use jolty_numerics::Complex64;
use serde::Deserialize;

use crate::ac::over_frequencies;
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
    /// Returns [`SimulationError::UnknownSource`] or [`SimulationError::NotAProbe`] for
    /// invalid probes, [`SimulationError::InvalidAnalysis`] for empty or negative
    /// frequencies, [`SimulationError::Singular`] for singular linearizations, and any
    /// error from [`Simulator::operating_point`].
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
            if self.source_nature(index) != Nature::Voltage {
                return Err(not_a_probe());
            }
            let &[Some(entered), Some(_)] = self.elaborated.devices[index].nodes() else {
                return Err(not_a_probe());
            };
            let carried = self
                .unknown(&format!("{}:current", stability.probe))
                .ok_or_else(not_a_probe)?;
            let unknowns = self.elaborated.unknowns.len();
            let level = self.source_excitation(index);
            let operating_point = self.linearize(&stability.frequencies, options)?;
            let gains = over_frequencies(
                self.linearization(),
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
