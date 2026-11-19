//! The small-signal frequency response around the operating point.

use std::f64::consts::TAU;

use jolty_models::{Contributions, DelayedEntry, Nature, UnknownIndex};
use jolty_numerics::Complex64;
use rayon::prelude::*;
use serde::Deserialize;

use crate::solver::{Factors, Refinement, Solver};
use crate::{OperatingPointResult, Options, Parallelism, SimulationError, Simulator};

/// A small-signal analysis: the response to a unit excitation of one independent source.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ac {
    /// The path of the excited source, such as `V1`.
    pub source: String,
    /// The frequencies, in hertz.
    pub frequencies: Vec<f64>,
}

/// The small-signal response of every unknown to a unit excitation, at each frequency.
#[derive(Debug, Clone, PartialEq)]
pub struct AcResult {
    /// The path of the excited source.
    pub source: String,
    /// The operating point the circuit was linearized at.
    pub operating_point: OperatingPointResult,
    /// The frequencies, in hertz.
    pub frequencies: Vec<f64>,
    /// The phasor of every unknown at each frequency, per volt or ampere of excitation.
    pub values: Vec<Vec<Complex64>>,
    /// The phasor of the current into each terminal its options record at each frequency,
    /// per volt or ampere of excitation, in the order of a [`RecordedValues`](crate::RecordedValues)'s
    /// currents.
    pub currents: Vec<Vec<Complex64>>,
}

impl Simulator {
    /// Linearizes the circuit at its operating point and solves `(G + jωC) x = −∂f/∂s` at each
    /// frequency, where `s` is the excited source's value.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownSource`] or [`SimulationError::NotASource`] when the
    /// excited instance is not an independent source, [`SimulationError::InvalidAnalysis`]
    /// for no frequency or a negative or non-finite one, [`SimulationError::Singular`] when
    /// the linearized circuit is singular at a frequency, and the errors of
    /// [`Simulator::operating_point`].
    pub fn ac(&mut self, ac: &Ac, options: &Options) -> Result<AcResult, SimulationError> {
        options.parallelism.confined(|| {
            self.solvable()?;
            let index = self.source_index(&ac.source)?;
            let operating_point = self.linearize(&ac.frequencies, options)?;
            let excitation = self.source_excitation(index);
            let currents =
                self.state
                    .recorder
                    .currents(&self.elaborated.groups, &self.state.runs, index);
            let found = over_frequencies(
                self.linearization(),
                &self.state.workspace.contributions,
                &ac.frequencies,
                &options.parallelism,
                &|factors, _, frequency| {
                    let mut phasors: Vec<Complex64> = excitation
                        .iter()
                        .map(|&d| Complex64::new(-d, 0.0))
                        .collect();
                    factors.solve(&mut phasors)?;
                    let recorded = currents.at(&phasors, TAU * frequency);
                    Ok((phasors, recorded))
                },
            )?;
            let (values, currents) = found.into_iter().unzip();
            Ok(AcResult {
                source: ac.source.clone(),
                operating_point,
                frequencies: ac.frequencies.clone(),
                values,
                currents,
            })
        })
    }

    /// Checks a small-signal analysis's `frequencies`, solves the operating point and
    /// evaluates the devices there, so the workspace holds `G` and `C`.
    pub(crate) fn linearize(
        &mut self,
        frequencies: &[f64],
        options: &Options,
    ) -> Result<OperatingPointResult, SimulationError> {
        options.check()?;
        if frequencies.is_empty() {
            return Err(SimulationError::InvalidAnalysis(
                "a small-signal analysis needs at least one frequency".to_owned(),
            ));
        }
        if let Some(frequency) = frequencies
            .iter()
            .find(|&&frequency| !(frequency >= 0.0 && frequency.is_finite()))
        {
            return Err(SimulationError::InvalidAnalysis(format!(
                "frequencies must be at least 0 and finite, not {frequency:?}"
            )));
        }
        self.linearized(options)
    }

    /// Solves the operating point and evaluates the devices there, so the workspace holds
    /// `G` and `C`.
    pub(crate) fn linearized(
        &mut self,
        options: &Options,
    ) -> Result<OperatingPointResult, SimulationError> {
        let operating_point = self.operating_point(options)?;
        self.evaluate(&operating_point.values, None, 1.0, options.gmin, true);
        Ok(operating_point)
    }

    /// The matrices of the circuit linearized at the operating point the workspace was
    /// last evaluated at, to factor at any frequency: `G + sC`, and the parts of `G` its
    /// delay lines read in the past each times `e^{−s·delay}`.
    pub(crate) fn linearization(&self) -> Linearization {
        let delayed = self.state.runs.delayed(&self.elaborated.groups);
        Linearization::new(&self.state.solver, delayed)
    }

    /// Fails with [`SimulationError::Delayed`], for `analysis` and the `reason` it gives,
    /// when the circuit holds a delay line.
    pub(crate) fn undelayed(
        &self,
        analysis: &'static str,
        reason: &'static str,
    ) -> Result<(), SimulationError> {
        let mut devices: Vec<usize> = self
            .state
            .runs
            .delayed(&self.elaborated.groups)
            .iter()
            .map(|part| part.device)
            .collect();
        if devices.is_empty() {
            return Ok(());
        }
        devices.sort_unstable();
        devices.dedup();
        Err(SimulationError::Delayed {
            analysis,
            reason,
            devices: devices
                .into_iter()
                .map(|device| format!("`{}`", self.elaborated.devices[device].path()))
                .collect(),
        })
    }

    /// The unknown named `output`, and the one named `reference` it is taken over, if
    /// any, with what the difference is: a voltage over a node, or any unknown alone.
    /// Ground's name names no unknown.
    pub(crate) fn measured(
        &self,
        output: &str,
        reference: Option<&str>,
    ) -> Result<(Option<UnknownIndex>, Option<UnknownIndex>, Nature), SimulationError> {
        let found = self.node(output)?;
        let base = reference.map(|name| self.node(name)).transpose()?.flatten();
        if found == base {
            return Err(SimulationError::InvalidAnalysis(format!(
                "the output `{output}` is taken over itself"
            )));
        }
        let nature = |node: Option<UnknownIndex>| {
            node.map_or(Nature::Voltage, |unknown| {
                self.elaborated.unknowns[unknown.index()].nature
            })
        };
        let nature = match (nature(found), nature(base), reference) {
            (nature, _, None) => nature,
            (Nature::Voltage, Nature::Voltage, Some(_)) => Nature::Voltage,
            (_, _, Some(name)) => {
                return Err(SimulationError::InvalidAnalysis(format!(
                    "an output over a node is a voltage, but `{output}` over `{name}` is not"
                )));
            }
        };
        Ok((found, base, nature))
    }

    /// `∂f/∂s` of the independent source at `device`, with one entry per unknown.
    pub(crate) fn source_excitation(&self, device: usize) -> Vec<f64> {
        let mut excitation = vec![0.0; self.elaborated.unknowns.len()];
        self.state
            .runs
            .excite(&self.elaborated.groups, device, &mut excitation);
        excitation
    }

    /// The node or unknown named `name`: none for ground.
    fn node(&self, name: &str) -> Result<Option<UnknownIndex>, SimulationError> {
        if name == self.reference() {
            return Ok(None);
        }
        self.unknown(name)
            .map(Some)
            .ok_or_else(|| SimulationError::UnknownOutput(name.to_owned()))
    }
}

/// A vector of length `unknowns` with `+1` at `positive` and `−1` at `negative`.
pub(crate) fn selector_vector<T>(
    unknowns: usize,
    positive: Option<UnknownIndex>,
    negative: Option<UnknownIndex>,
) -> Vec<T>
where
    T: Default + Clone + std::ops::AddAssign + From<f64>,
{
    let mut values = vec![T::default(); unknowns];
    for (node, sign) in [(positive, 1.0), (negative, -1.0)] {
        if let Some(unknown) = node {
            values[unknown.index()] += T::from(sign);
        }
    }
    values
}

/// What a small-signal analysis finds at one frequency, from the factors of `G + jωC`
/// there, the frequency's index among the analysis's, and the frequency.
pub(crate) type At<'a, R> =
    dyn Fn(&mut Factors<'_, Complex64>, usize, f64) -> Result<R, SimulationError> + Sync + 'a;

/// What `at` finds at each of `frequencies`, in order, from the matrix `G + jωC` of
/// `contributions` factored there: as `start` factors them going through the frequencies
/// in order, each factorization keeping the pivots the one before it left, while they
/// hold.
///
/// Spread over the tasks `parallelism` allows, it finds the same bits. The first frequency
/// is factored first, then each task takes a run of the others from the pivots it left.
/// A run's results stand when it started from the pivots the run before it left, which
/// compare alike unless a factorization of that run searched for new ones; every run that
/// did not is taken again, from the pivots the run before it left that time, until each
/// stands: as many rounds as the frequencies have runs whose pivots change, and one where
/// they hold throughout.
///
/// # Errors
///
/// Returns the first error, by frequency, of a factorization or of `at`.
pub(crate) fn over_frequencies<R: Send>(
    start: Linearization,
    contributions: &Contributions,
    frequencies: &[f64],
    parallelism: &Parallelism,
    at: &At<'_, R>,
) -> Result<Vec<R>, SimulationError> {
    let mut first = start;
    let Some((&lowest, rest)) = frequencies.split_first() else {
        return Ok(Vec::new());
    };
    let mut found = Vec::with_capacity(frequencies.len());
    found.push(at(&mut first.factor(contributions, lowest)?, 0, lowest)?);
    let tasks = parallelism.tasks(rest.len());
    if tasks == 1 {
        let chunk = FrequencyChunk {
            first: 1,
            frequencies: rest,
        };
        return FrequencyRun::from(first, chunk, contributions, at).finish(found);
    }
    let size = rest.len().div_ceil(tasks);
    let parts: Vec<FrequencyChunk<'_>> = rest
        .chunks(size)
        .enumerate()
        .map(|(part, frequencies)| FrequencyChunk {
            first: 1 + part * size,
            frequencies,
        })
        .collect();
    // Where each run starts, none for where the first frequency left `first`, and its
    // results from there, once taken.
    let mut starts: Vec<Option<Linearization>> = (0..parts.len()).map(|_| None).collect();
    let mut runs: Vec<Option<FrequencyRun<R>>> = (0..parts.len()).map(|_| None).collect();
    let mut standing = 0;
    while standing < parts.len() {
        let taken: Vec<(usize, FrequencyRun<R>)> = (standing..parts.len())
            .filter(|&part| runs[part].is_none())
            .collect::<Vec<usize>>()
            .into_par_iter()
            .map(|part| {
                let start = starts[part].as_ref().unwrap_or(&first).clone();
                (
                    part,
                    FrequencyRun::from(start, parts[part], contributions, at),
                )
            })
            .collect();
        for (part, run) in taken {
            runs[part] = Some(run);
        }
        // The runs stand in order while each started from the pivots the run before it
        // left; the first error of one that stands is the first error of all.
        while standing < parts.len()
            && starts[standing]
                .as_ref()
                .unwrap_or(&first)
                .factors_alike(end_before(&runs, standing, &first))
        {
            if let Some(run) = runs[standing].as_mut()
                && let Some(error) = run.error.take()
            {
                return Err(error);
            }
            standing += 1;
        }
        // Every run that does not stand starts again from where the run before it ended,
        // unless it started from pivots alike already.
        let ends: Vec<Option<Linearization>> = (standing..parts.len())
            .map(|part| {
                let end = end_before(&runs, part, &first);
                let start = starts[part].as_ref().unwrap_or(&first);
                (!start.factors_alike(end)).then(|| end.clone())
            })
            .collect();
        for (part, end) in (standing..parts.len()).zip(ends) {
            if end.is_some() {
                starts[part] = end;
                runs[part] = None;
            }
        }
    }
    for run in runs.into_iter().flatten() {
        found.extend(run.found);
    }
    Ok(found)
}

/// Where the run before `part` ended: where the first frequency left `first`, for the
/// first run.
fn end_before<'a, R>(
    runs: &'a [Option<FrequencyRun<R>>],
    part: usize,
    first: &'a Linearization,
) -> &'a Linearization {
    part.checked_sub(1)
        .and_then(|before| runs[before].as_ref())
        .map_or(first, |run| &run.end)
}

/// A contiguous slice of frequencies and the index of its first frequency.
#[derive(Clone, Copy)]
struct FrequencyChunk<'a> {
    first: usize,
    frequencies: &'a [f64],
}

/// What `at` found at a run of frequencies, from one linearization going through them in
/// order: each frequency's, up to the first error, and the linearization as it ended.
struct FrequencyRun<R> {
    found: Vec<R>,
    error: Option<SimulationError>,
    end: Linearization,
}

impl<R> FrequencyRun<R> {
    /// `at` at each frequency in `chunk` in order, from `linearization`, up to the first error.
    fn from(
        mut linearization: Linearization,
        chunk: FrequencyChunk<'_>,
        contributions: &Contributions,
        at: &At<'_, R>,
    ) -> Self {
        let mut found = Vec::with_capacity(chunk.frequencies.len());
        let mut error = None;
        for (index, &frequency) in (chunk.first..).zip(chunk.frequencies) {
            match linearization
                .factor(contributions, frequency)
                .and_then(|mut factors| at(&mut factors, index, frequency))
            {
                Ok(value) => found.push(value),
                Err(failed) => {
                    error = Some(failed);
                    break;
                }
            }
        }
        Self {
            found,
            error,
            end: linearization,
        }
    }

    /// `found` with the run's results after it, or the run's error.
    fn finish(self, mut found: Vec<R>) -> Result<Vec<R>, SimulationError> {
        if let Some(error) = self.error {
            return Err(error);
        }
        found.extend(self.found);
        Ok(found)
    }
}

/// The matrix `G + jωC` of a circuit linearized at its operating point, factored at any
/// frequency, each part of `G` a delay line reads in the past multiplied by `e^{−jωτ}`,
/// `τ` its delay.
#[derive(Clone)]
pub(crate) struct Linearization {
    solver: Solver<Complex64>,
    matrix: Vec<Complex64>,
    refinement: Refinement<Complex64>,
    /// Each part of `G` that a delay line reads in the past, which `G` holds whole, as
    /// at DC.
    delayed: Vec<DelayedEntry>,
}

impl Linearization {
    /// A linearization over the matrices of `solver`, the simulator's, whose ordering it
    /// shares, of a circuit whose delay lines read the parts `delayed` of its `G` in the
    /// past.
    pub(crate) fn new(solver: &Solver<f64>, delayed: Vec<DelayedEntry>) -> Self {
        Self {
            solver: solver.sharing(),
            matrix: vec![Complex64::default(); solver.pattern().len()],
            refinement: Refinement::default(),
            delayed,
        }
    }

    /// `G + jωC` at `frequency`, from the `contributions` at the operating point, factored.
    pub(crate) fn factor(
        &mut self,
        contributions: &Contributions,
        frequency: f64,
    ) -> Result<Factors<'_, Complex64>, SimulationError> {
        self.factor_at(contributions, Complex64::new(0.0, TAU * frequency))
    }

    /// Whether this linearization factors every matrix from now on as `other` would, bit
    /// for bit: they keep the same pivots.
    fn factors_alike(&self, other: &Self) -> bool {
        self.solver.factors_alike(&other.solver)
    }

    /// `G + sC` at the complex frequency `s`, from the `contributions` at the operating
    /// point, factored.
    pub(crate) fn factor_at(
        &mut self,
        contributions: &Contributions,
        s: Complex64,
    ) -> Result<Factors<'_, Complex64>, SimulationError> {
        for ((value, &conductance), &capacitance) in self
            .matrix
            .iter_mut()
            .zip(&contributions.conductance)
            .zip(&contributions.capacitance)
        {
            *value = conductance + s * capacitance;
        }
        for part in &self.delayed {
            if let Some(value) = self.matrix.get_mut(part.position as usize) {
                *value += ((-s * part.delay).exp() - 1.0) * part.conductance;
            }
        }
        self.solver
            .factor_refined(&self.matrix, &mut self.refinement)
    }
}

#[cfg(test)]
mod tests {
    use crate::test_helpers::source_resistor_simulator;
    use crate::{Ac, Options, SimulationError};

    #[test]
    fn ac_rejects_non_finite_and_negative_frequencies() {
        let mut simulator = source_resistor_simulator();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            let result = simulator.ac(
                &Ac {
                    source: "V1".to_owned(),
                    frequencies: vec![1e3, bad],
                },
                &Options::default(),
            );
            assert!(
                matches!(result, Err(SimulationError::InvalidAnalysis(_))),
                "expected InvalidAnalysis for frequency={bad:?}, got {result:?}"
            );
        }
        let ok = simulator.ac(
            &Ac {
                source: "V1".to_owned(),
                frequencies: vec![0.0, 1e3],
            },
            &Options::default(),
        );
        assert!(ok.is_ok(), "{ok:?}");
    }
}
