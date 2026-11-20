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

use jolty_models::{Nature, UnknownIndex};
use jolty_numerics::Complex64;
use serde::Deserialize;

use crate::ac::over_frequencies;
use crate::solver::Factors;
use crate::{OperatingPointResult, Options, SimulationError, Simulator};

/// Output name for the loop gain `T` in sensitivity, mismatch, and derivative analyses.
pub const LOOP_GAIN: &str = "loop_gain";

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

/// Validated loop-breaking voltage-source probe resolved to its device and unknown indices.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Probe {
    /// Placed device index of the floating voltage source acting as the probe.
    pub(crate) device: usize,
    /// Node unknown on the probe's `positive` terminal (driven side of the loop).
    pub(crate) entered: UnknownIndex,
    /// Branch current unknown carried by the probe (`<probe>:current`).
    pub(crate) carried: UnknownIndex,
}

/// Small-signal double-injection responses at a single frequency.
pub(crate) struct Injection {
    /// Solution `x_s` to `(G + jωC) x_s = -b_series` for a unit series voltage.
    pub(crate) series: Vec<Complex64>,
    /// Solution `x_i` to `(G + jωC) x_i = e_entered` for a unit shunt current into `entered`.
    pub(crate) shunt: Vec<Complex64>,
    /// Middlebrook combined ratio `u = x_s[entered] - x_i[carried]`.
    pub(crate) ratio: Complex64,
}

impl Probe {
    /// Performs both unit injections against a single factorization `factors` of `G + jωC`.
    pub(crate) fn inject(
        self,
        factors: &mut Factors<'_, Complex64>,
        level: &[f64],
    ) -> Result<Injection, SimulationError> {
        let mut series: Vec<Complex64> = level
            .iter()
            .map(|&entry| Complex64::new(-entry, 0.0))
            .collect();
        factors.solve(&mut series)?;
        let mut shunt = vec![Complex64::default(); level.len()];
        shunt[self.entered.index()] = Complex64::new(1.0, 0.0);
        factors.solve(&mut shunt)?;
        let ratio = series[self.entered.index()] - shunt[self.carried.index()];
        Ok(Injection {
            series,
            shunt,
            ratio,
        })
    }
}

impl Simulator {
    /// Validates `name` as a floating independent voltage source and resolves its probe
    /// indices.
    pub(crate) fn probe(&self, name: &str) -> Result<Probe, SimulationError> {
        let not_a_probe = || SimulationError::NotAProbe(name.to_owned());
        let device = self.source_index(name).map_err(|error| match error {
            SimulationError::NotASource(_) => not_a_probe(),
            other => other,
        })?;
        if self.source_nature(device) != Nature::Voltage {
            return Err(not_a_probe());
        }
        let &[Some(entered), Some(_)] = self.elaborated.devices[device].nodes() else {
            return Err(not_a_probe());
        };
        let carried = self
            .unknown(&format!("{name}:current"))
            .ok_or_else(not_a_probe)?;
        Ok(Probe {
            device,
            entered,
            carried,
        })
    }

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
            let probe = self.probe(&stability.probe)?;
            let level = self.source_excitation(probe.device);
            let operating_point = self.linearize(&stability.frequencies, options)?;
            let gains = over_frequencies(
                self.linearization(),
                &self.state.workspace.contributions,
                &stability.frequencies,
                &options.parallelism,
                &|factors, _, _| {
                    let injection = probe.inject(factors, &level)?;
                    Ok(-injection.ratio / (1.0 + injection.ratio))
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
