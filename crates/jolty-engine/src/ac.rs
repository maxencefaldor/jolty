//! The small-signal frequency response around the operating point.

use std::f64::consts::TAU;

use jolty_models::{Contributions, Quantity, Unknown};
use jolty_numerics::Complex64;
use rayon::prelude::*;
use serde::Deserialize;

use crate::solver::{Factors, Solver};
use crate::{OperatingPoint, Options, Parallelism, SimulationError, Simulator};

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
pub struct Response {
    /// The path of the excited source.
    pub source: String,
    /// The operating point the circuit was linearized at.
    pub operating_point: OperatingPoint,
    /// The frequencies, in hertz.
    pub frequencies: Vec<f64>,
    /// The phasor of every unknown at each frequency, per volt or ampere of excitation.
    pub values: Vec<Vec<Complex64>>,
    /// The phasor of the current into each terminal its options record at each frequency,
    /// per volt or ampere of excitation, in the order of an [`Observation`](crate::Observation)'s
    /// currents.
    pub currents: Vec<Vec<Complex64>>,
}

impl Simulator {
    /// Linearizes the circuit at its operating point and solves `(G + jωC) x = −∂f/∂s` at each
    /// frequency, where `s` is the excited source's level.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownSource`] or [`SimulationError::NotASource`] when the
    /// excited instance is not an independent source, [`SimulationError::InvalidAnalysis`]
    /// for no frequency or a negative one, [`SimulationError::Singular`] when the linearized
    /// circuit is singular at a frequency, and the errors of [`Simulator::operating_point`].
    pub fn ac(&mut self, ac: &Ac, options: &Options) -> Result<Response, SimulationError> {
        self.solvable()?;
        let index = self.source_index(&ac.source)?;
        let operating_point = self.linearize(&ac.frequencies, options)?;
        let mut excitation = vec![0.0; self.elaborated.unknowns.len()];
        if let Some(source) = self
            .state
            .device(index)
            .and_then(|device| device.as_source())
        {
            source.level_derivative(&mut excitation);
        }
        self.state
            .observer
            .linearize(&operating_point.values, options.gmin);
        let currents = self.state.observer.currents(index);
        let found = over_frequencies(
            Linearization::new(&self.state.solver),
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
        Ok(Response {
            source: ac.source.clone(),
            operating_point,
            frequencies: ac.frequencies.clone(),
            values,
            currents,
        })
    }

    /// Checks a small-signal analysis's `frequencies`, solves the operating point and
    /// evaluates the devices there, so the workspace holds `G` and `C`.
    pub(crate) fn linearize(
        &mut self,
        frequencies: &[f64],
        options: &Options,
    ) -> Result<OperatingPoint, SimulationError> {
        options.check()?;
        if frequencies.is_empty() {
            return Err(SimulationError::InvalidAnalysis(
                "a small-signal analysis needs at least one frequency".to_owned(),
            ));
        }
        if let Some(frequency) = frequencies.iter().find(|&&frequency| frequency < 0.0) {
            return Err(SimulationError::InvalidAnalysis(format!(
                "frequencies must be at least 0, not {frequency:?}"
            )));
        }
        self.linearized(options)
    }

    /// Solves the operating point and evaluates the devices there, so the workspace holds
    /// `G` and `C`.
    pub(crate) fn linearized(
        &mut self,
        options: &Options,
    ) -> Result<OperatingPoint, SimulationError> {
        let operating_point = self.operating_point(options)?;
        self.evaluate(&operating_point.values, None, (1.0, options.gmin), true);
        Ok(operating_point)
    }

    /// The unknown named `output`, and the one named `reference` it is taken over, if
    /// any, with what the difference is: a voltage over a node, or any unknown alone. The
    /// reference node's name names no unknown.
    pub(crate) fn measured(
        &self,
        output: &str,
        reference: Option<&str>,
    ) -> Result<(Option<Unknown>, Option<Unknown>, Quantity), SimulationError> {
        let found = self.node(output)?;
        let base = reference.map(|name| self.node(name)).transpose()?.flatten();
        if found == base {
            return Err(SimulationError::InvalidAnalysis(format!(
                "the output `{output}` is taken over itself"
            )));
        }
        let quantity = |node: Option<Unknown>| {
            node.map_or(Quantity::Voltage, |unknown| {
                self.elaborated.unknowns[unknown.index()].quantity
            })
        };
        let quantity = match (quantity(found), quantity(base), reference) {
            (quantity, _, None) => quantity,
            (Quantity::Voltage, Quantity::Voltage, Some(_)) => Quantity::Voltage,
            (_, _, Some(name)) => {
                return Err(SimulationError::InvalidAnalysis(format!(
                    "an output over a node is a voltage, but `{output}` over `{name}` is not"
                )));
            }
        };
        Ok((found, base, quantity))
    }

    /// The node or unknown named `name`: none for the reference node.
    fn node(&self, name: &str) -> Result<Option<Unknown>, SimulationError> {
        if name == self.reference() {
            return Ok(None);
        }
        self.unknown(name)
            .map(Some)
            .ok_or_else(|| SimulationError::UnknownOutput(name.to_owned()))
    }
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
        return Run::from(first, (1, rest), contributions, at).finish(found);
    }
    let size = rest.len().div_ceil(tasks);
    let parts: Vec<(usize, &[f64])> = rest
        .chunks(size)
        .enumerate()
        .map(|(part, frequencies)| (1 + part * size, frequencies))
        .collect();
    // Where each run starts, none for where the first frequency left `first`, and its
    // results from there, once taken.
    let mut starts: Vec<Option<Linearization>> = (0..parts.len()).map(|_| None).collect();
    let mut runs: Vec<Option<Run<R>>> = (0..parts.len()).map(|_| None).collect();
    let mut standing = 0;
    while standing < parts.len() {
        let taken: Vec<(usize, Run<R>)> = (standing..parts.len())
            .filter(|&part| runs[part].is_none())
            .collect::<Vec<usize>>()
            .into_par_iter()
            .map(|part| {
                let start = starts[part].as_ref().unwrap_or(&first).clone();
                (part, Run::from(start, parts[part], contributions, at))
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
    runs: &'a [Option<Run<R>>],
    part: usize,
    first: &'a Linearization,
) -> &'a Linearization {
    part.checked_sub(1)
        .and_then(|before| runs[before].as_ref())
        .map_or(first, |run| &run.end)
}

/// What `at` found at a run of frequencies, from one linearization going through them in
/// order: each frequency's, up to the first error, and the linearization as it ended.
struct Run<R> {
    found: Vec<R>,
    error: Option<SimulationError>,
    end: Linearization,
}

impl<R> Run<R> {
    /// `at` at each of `frequencies` in order, the first of them the frequency at `first`
    /// among all of them, from `linearization`, up to the first error.
    fn from(
        mut linearization: Linearization,
        (first, frequencies): (usize, &[f64]),
        contributions: &Contributions,
        at: &At<'_, R>,
    ) -> Self {
        let mut found = Vec::with_capacity(frequencies.len());
        let mut error = None;
        for (index, &frequency) in (first..).zip(frequencies) {
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
/// frequency.
#[derive(Clone)]
pub(crate) struct Linearization {
    solver: Solver<Complex64>,
    matrix: Vec<Complex64>,
}

impl Linearization {
    /// A linearization over the matrices of `solver`, the simulator's, whose ordering it
    /// shares.
    pub(crate) fn new(solver: &Solver<f64>) -> Self {
        Self {
            solver: solver.sharing(),
            matrix: vec![Complex64::default(); solver.pattern().len()],
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
        self.solver.factor(&self.matrix)
    }
}
