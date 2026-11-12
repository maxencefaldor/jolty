//! Any analysis, as a simulation plan names it.

use serde::Deserialize;

use crate::{
    Ac, DcSweep, LoopGain, Noise, NoiseSpectrum, OperatingPoint, OperatingPoints, Options,
    PoleZero, PolesZeros, Response, Sensitivities, Sensitivity, SimulationError, Simulator,
    Stability, Sweep, Transient, Waveforms,
};

/// An analysis to run.
///
/// In JSON, the operating point is the string `"operating_point"`, and the others are objects
/// with one key, such as `{"transient": {"stop": 1e-3}}`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Analysis {
    /// The DC operating point.
    OperatingPoint,
    /// A DC sweep of one source.
    DcSweep(DcSweep),
    /// A small-signal frequency response.
    Ac(Ac),
    /// A transient analysis.
    Transient(Transient),
    /// The DC sensitivities of some unknowns to every device parameter.
    Sensitivity(Sensitivity),
    /// The loop gain through a probe.
    Stability(Stability),
    /// The noise at an output.
    Noise(Noise),
    /// The poles, and the zeros of a transfer function.
    PoleZero(PoleZero),
    /// Every DC solution, up to a number.
    OperatingPoints(OperatingPoints),
}

/// The result of an [`Analysis`].
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// The DC operating point.
    OperatingPoint(OperatingPoint),
    /// The operating points of a DC sweep.
    DcSweep(Sweep),
    /// A small-signal frequency response.
    Ac(Response),
    /// The waveforms of a transient analysis.
    Transient(Waveforms),
    /// The DC sensitivities.
    Sensitivity(Sensitivities),
    /// The loop gain.
    Stability(LoopGain),
    /// The noise at an output.
    Noise(NoiseSpectrum),
    /// The poles and zeros.
    PoleZero(PolesZeros),
    /// The DC solutions, the operating point first.
    OperatingPoints(Vec<OperatingPoint>),
}

impl Simulator {
    /// Runs `analysis` with `options`.
    ///
    /// # Errors
    ///
    /// Returns the errors of the analysis run.
    pub fn run(
        &mut self,
        analysis: &Analysis,
        options: &Options,
    ) -> Result<Outcome, SimulationError> {
        Ok(match analysis {
            Analysis::OperatingPoint => Outcome::OperatingPoint(self.operating_point(options)?),
            Analysis::DcSweep(sweep) => Outcome::DcSweep(self.dc_sweep(sweep, options)?),
            Analysis::Ac(ac) => Outcome::Ac(self.ac(ac, options)?),
            Analysis::Transient(transient) => {
                Outcome::Transient(self.transient(transient, options)?)
            }
            Analysis::Sensitivity(sensitivity) => {
                Outcome::Sensitivity(self.sensitivity(sensitivity, options)?)
            }
            Analysis::Stability(stability) => {
                Outcome::Stability(self.stability(stability, options)?)
            }
            Analysis::Noise(noise) => Outcome::Noise(self.noise(noise, options)?),
            Analysis::PoleZero(analysis) => Outcome::PoleZero(self.pole_zero(analysis, options)?),
            Analysis::OperatingPoints(search) => {
                Outcome::OperatingPoints(self.operating_points(search, options)?)
            }
        })
    }
}
