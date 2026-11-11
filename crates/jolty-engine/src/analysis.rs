//! Any analysis, as a simulation plan names it.

use serde::Deserialize;

use crate::{
    Ac, DcSweep, OperatingPoint, Options, Response, SimulationError, Simulator, Sweep, Transient,
    Waveforms,
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
        })
    }
}
