//! Any analysis, as a simulation plan names it.

use serde::Deserialize;

use crate::{
    Ac, AcResult, DcSweep, DcSweepResult, Mismatch, MismatchResult, Noise, NoiseResult,
    OperatingPointResult, OperatingPoints, Options, PeriodicSteadyState, PeriodicSteadyStateResult,
    PoleZero, PoleZeroResult, Sensitivity, SensitivityResult, SimulationError, Simulator, Sizes,
    Stability, StabilityResult, TransferFunction, TransferFunctionResult, Transient,
    TransientResult,
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
    /// One period of a periodic steady state.
    PeriodicSteadyState(PeriodicSteadyState),
    /// Sensitivities of some unknowns or noise densities to every device parameter across
    /// operating point, DC sweep, AC, noise, transient, or periodic steady state analyses.
    Sensitivity(Sensitivity),
    /// Mismatch variance analysis with respect to random parameter draws.
    Mismatch(Mismatch),
    /// The loop gain through a probe.
    Stability(Stability),
    /// The noise at an output.
    Noise(Noise),
    /// The poles, and the zeros of a transfer function.
    PoleZero(PoleZero),
    /// Every DC solution, up to a number.
    OperatingPoints(OperatingPoints),
    /// The small-signal transfer function at DC: a gain and two resistances.
    TransferFunction(TransferFunction),
}

/// The result of an [`Analysis`].
#[derive(Debug, Clone, PartialEq)]
pub enum AnalysisResult {
    /// The DC operating point.
    OperatingPoint(OperatingPointResult),
    /// The operating points of a DC sweep.
    DcSweep(DcSweepResult),
    /// A small-signal frequency response.
    Ac(AcResult),
    /// The waveforms of a transient analysis.
    Transient(TransientResult),
    /// One period of a periodic steady state.
    PeriodicSteadyState(PeriodicSteadyStateResult),
    /// Sensitivities across an operating point, DC sweep, AC, noise, transient, or periodic
    /// steady state analysis.
    Sensitivity(SensitivityResult),
    /// The derivatives of a mismatch analysis.
    Mismatch(MismatchResult),
    /// The loop gain.
    Stability(StabilityResult),
    /// The noise at an output.
    Noise(NoiseResult),
    /// The poles and zeros.
    PoleZero(PoleZeroResult),
    /// The DC solutions, the operating point first.
    OperatingPoints(Vec<OperatingPointResult>),
    /// The transfer function at DC.
    TransferFunction(TransferFunctionResult),
}

impl Simulator {
    /// Returns whether `analysis` under `options` can parallelize work on a pool of
    /// `threads` workers, according to [`crate::Parallelism::fills_threads`].
    ///
    /// Considers device instance count, unknown count, whether the nested-dissection
    /// ordering splits a block (built lazily here if needed), and small-signal frequency
    /// counts. Sensitivity and mismatch analyses always allow parallel right-hand sides.
    /// When `false`, execution stays on the caller thread outside any pool.
    #[must_use]
    pub fn spreads(&self, analysis: &Analysis, options: &Options, threads: usize) -> bool {
        let solves = match analysis {
            Analysis::Ac(ac) => ac.frequencies.len(),
            Analysis::Noise(noise) => noise.frequencies.len(),
            Analysis::Stability(stability) => stability.frequencies.len(),
            Analysis::Sensitivity(_) | Analysis::Mismatch(_) => usize::MAX,
            Analysis::OperatingPoint
            | Analysis::DcSweep(_)
            | Analysis::Transient(_)
            | Analysis::PeriodicSteadyState(_)
            | Analysis::PoleZero(_)
            | Analysis::OperatingPoints(_)
            | Analysis::TransferFunction(_) => 0,
        };
        let elaborated = &self.elaborated;
        let parallelism = &options.parallelism;
        let usable = parallelism
            .max_threads
            .map_or(threads, |most| most.min(threads));
        let dissected = parallelism.spreads()
            && elaborated
                .ordering(parallelism.dissecting_on(usable))
                .is_ok_and(|ordering| ordering.is_dissected());
        let sizes = Sizes {
            instances: elaborated.groups.evaluated(),
            unknowns: elaborated.unknowns().len(),
            dissected,
            solves,
        };
        parallelism.fills_threads(sizes, threads)
    }

    /// Runs `analysis` with `options`.
    ///
    /// # Errors
    ///
    /// Returns the errors of the analysis run.
    pub fn run(
        &mut self,
        analysis: &Analysis,
        options: &Options,
    ) -> Result<AnalysisResult, SimulationError> {
        Ok(match analysis {
            Analysis::OperatingPoint => {
                AnalysisResult::OperatingPoint(self.operating_point(options)?)
            }
            Analysis::DcSweep(sweep) => AnalysisResult::DcSweep(self.dc_sweep(sweep, options)?),
            Analysis::Ac(ac) => AnalysisResult::Ac(self.ac(ac, options)?),
            Analysis::Transient(transient) => {
                AnalysisResult::Transient(self.transient(transient, options)?)
            }
            Analysis::PeriodicSteadyState(analysis) => {
                AnalysisResult::PeriodicSteadyState(self.periodic_steady_state(analysis, options)?)
            }
            Analysis::Sensitivity(sensitivity) => {
                AnalysisResult::Sensitivity(self.sensitivity(sensitivity, options)?)
            }
            Analysis::Mismatch(mismatch) => {
                AnalysisResult::Mismatch(self.mismatch(mismatch, options)?)
            }
            Analysis::Stability(stability) => {
                AnalysisResult::Stability(self.stability(stability, options)?)
            }
            Analysis::Noise(noise) => AnalysisResult::Noise(self.noise(noise, options)?),
            Analysis::PoleZero(analysis) => {
                AnalysisResult::PoleZero(self.pole_zero(analysis, options)?)
            }
            Analysis::OperatingPoints(search) => {
                AnalysisResult::OperatingPoints(self.operating_points(search, options)?)
            }
            Analysis::TransferFunction(analysis) => {
                AnalysisResult::TransferFunction(self.transfer_function(analysis, options)?)
            }
        })
    }
}
