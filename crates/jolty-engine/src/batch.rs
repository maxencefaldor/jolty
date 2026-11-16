//! The batch executor: many runs of one circuit in one call, spread across the threads of
//! the rayon pool it runs in.
//!
//! A [`Batch`] holds a circuit elaborated once, the analyses each run runs in order, each
//! with its options, and the [measures](crate::measure) each run reads of their results. Its
//! [`Batch::run`] takes [`Variants`], the values of each run as a pure function of the
//! run's index, and a range of indices, and hands each run's [`Run`] to a [`Sink`] at
//! its index. Every outcome is a pure function of the circuit, the analyses, the options and
//! the index: the same bits however many threads run the batch, however its range is split,
//! and in whatever order its runs end.
//!
//! - **Chains.** The runs are taken in chains of [`Variants::chain`] runs, each chain on one
//!   thread, every run of it but the first starting from the operating point of the run
//!   before it, as a DC sweep does from one level to the next. A range that starts within a
//!   chain runs the chain from its start, and hands on only the runs within the range.
//! - **States.** Each thread keeps a run's state for each structure it meets, and resets it
//!   for every run, so a run never depends on the run its thread ran before, but within
//!   its chain.
//! - **Failures.** A run whose values make a problem, or one of whose analyses fails, is a
//!   failed outcome that says why, and the batch goes on.
//! - **Structures.** A run whose values give the circuit another structure, as a resistance
//!   crossing 1 Ω carries its current as an unknown, runs on the circuit elaborated for that
//!   structure, built once for the batch, whichever thread meets it first, behind the one
//!   lock of the engine. Its unknowns may lie elsewhere among its results, so its measures
//!   read the unknowns of the names they read of the batch's circuit, and its outcome says
//!   where each lies.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use rayon::prelude::*;

use crate::elaborate::ElaborationError;
use crate::measure::{AnalysisResult, Measure, MeasureError};
use crate::{Analysis, Elaborated, Options, SimulationError, Simulator, StructureKey, Values};

/// The values of the runs of a batch.
pub trait Variants: Sync {
    /// How many runs there are.
    fn len(&self) -> u64;

    /// Whether there are no runs.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The values of the run at `index`, below [`Variants::len`]: a pure function of the
    /// index.
    fn values(&self, index: u64) -> Values;

    /// How many consecutive runs make a chain, each run of it starting from the solution of
    /// the run before it, as the points of a sweep do; 1, by default, for runs that each
    /// start afresh. A chain starts at each multiple of it, and the runs of one differ one
    /// from the next only in the numbers its line steps.
    fn chain(&self) -> u64 {
        1
    }
}

/// Where a batch hands each run's outcome.
pub trait Sink: Sync {
    /// Takes the outcome of the run at `index`. The runs of a batch end in any order, and
    /// each is handed on once.
    fn accept(&self, index: u64, outcome: Run);
}

/// A measure a batch reads of each run: of the result of the analysis at `analysis`, by
/// its index among the batch's analyses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measured {
    /// The analysis whose result the measure reads, by its index.
    pub analysis: usize,
    /// The measure.
    pub measure: Measure,
}

/// What one run of a batch found.
#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    /// The values the run ran at.
    pub values: Values,
    /// The circuit the run ran on, when its values gave the batch's circuit another
    /// structure; none when they did not, or made problems.
    pub structure: Option<Arc<Restructured>>,
    /// The problems the run's values make, which no analysis runs past; none when they make
    /// none.
    pub problems: Option<ElaborationError>,
    /// What each analysis found, in the batch's order, or why it failed; none when the
    /// values make problems.
    pub results: Vec<Result<crate::AnalysisResult, SimulationError>>,
    /// What each measure read, in the batch's order, or why it read nothing.
    pub measured: Vec<Result<f64, Unmeasured>>,
}

impl Run {
    /// The index among the run's results' values of the batch circuit's unknown at
    /// `unknown`: the same index on the batch's own structure, and that of the unknown of
    /// the same name on another; none when the run's circuit has no unknown of that name.
    #[must_use]
    pub fn column(&self, unknown: usize) -> Option<usize> {
        match &self.structure {
            None => Some(unknown),
            Some(structure) => structure.column(unknown),
        }
    }

    /// Whether the run failed: its values made problems, or an analysis failed.
    #[must_use]
    pub fn failed(&self) -> bool {
        self.problems.is_some() || self.results.iter().any(Result::is_err)
    }

    /// Why the run failed: the problems its values made, or each analysis that failed, by
    /// its index, and why; none when it did not.
    #[must_use]
    pub fn failure(&self) -> Option<String> {
        if let Some(problems) = &self.problems {
            return Some(problems.to_string());
        }
        let failures: Vec<String> = self
            .results
            .iter()
            .enumerate()
            .filter_map(|(index, result)| {
                result
                    .as_ref()
                    .err()
                    .map(|error| format!("analysis {index}: {error}"))
            })
            .collect();
        (!failures.is_empty()).then(|| failures.join("; "))
    }
}

/// Why a measure of a run read no number.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum Unmeasured {
    /// The run's values made problems, or the analysis it reads failed.
    #[error("the analysis it reads did not run to its end")]
    Failed,
    /// The measure could not read the analysis's result.
    #[error(transparent)]
    Measure(#[from] MeasureError),
}

/// What the runs of a batch found together, each number exact, so that the summaries of the
/// parts of a range [merge](Summary::merge) into the whole range's, bit for bit.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    /// The runs.
    pub runs: u64,
    /// The runs that failed.
    pub failed: u64,
    /// Each measure's extremes over the runs, in the batch's order.
    pub measures: Vec<Extremes>,
}

/// The extremes of a measure over the runs of a batch.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Extremes {
    /// The runs whose measure read a number.
    pub measured: u64,
    /// The least number read, in the total order of `f64`; none when none was.
    pub minimum: Option<f64>,
    /// The greatest number read, in the total order of `f64`; none when none was.
    pub maximum: Option<f64>,
}

impl Extremes {
    /// The extremes with `value` read too.
    fn with(self, value: f64) -> Self {
        let least = |a: f64, b: f64| if b.total_cmp(&a).is_lt() { b } else { a };
        let most = |a: f64, b: f64| if b.total_cmp(&a).is_gt() { b } else { a };
        Self {
            measured: self.measured + 1,
            minimum: Some(self.minimum.map_or(value, |minimum| least(minimum, value))),
            maximum: Some(self.maximum.map_or(value, |maximum| most(maximum, value))),
        }
    }

    /// The extremes of the runs of both.
    fn merge(self, other: Self) -> Self {
        let mut merged = Self {
            measured: self.measured,
            ..self
        };
        for value in [other.minimum, other.maximum].into_iter().flatten() {
            merged = merged.with(value);
        }
        merged.measured = self.measured + other.measured;
        merged
    }
}

impl Summary {
    /// The summary of no run, of `measures` measures.
    #[must_use]
    pub fn empty(measures: usize) -> Self {
        Self {
            runs: 0,
            failed: 0,
            measures: vec![Extremes::default(); measures],
        }
    }

    /// The summary of the runs of both, as one summary of them all: every count added, and
    /// the extremes of both.
    #[must_use]
    pub fn merge(mut self, other: Self) -> Self {
        self.runs += other.runs;
        self.failed += other.failed;
        for (mine, theirs) in self.measures.iter_mut().zip(other.measures) {
            *mine = mine.merge(theirs);
        }
        self
    }

    /// Counts the run that found `outcome`.
    fn add(&mut self, outcome: &Run) {
        self.runs += 1;
        self.failed += u64::from(outcome.failed());
        for (extremes, measured) in self.measures.iter_mut().zip(&outcome.measured) {
            if let Ok(value) = measured {
                *extremes = extremes.with(*value);
            }
        }
    }
}

/// Why a batch cannot run.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BatchError {
    /// A measure reads an analysis the batch does not have.
    #[error("measure {measure} reads analysis {analysis}, but the batch runs {analyses}")]
    NoAnalysis {
        /// The measure, by its index.
        measure: usize,
        /// The analysis it reads, by its index.
        analysis: usize,
        /// How many analyses the batch runs.
        analyses: usize,
    },
    /// The range of runs does not lie within the variants.
    #[error("runs {start} to {end} do not lie within the {len} variants")]
    Range {
        /// Where the range starts.
        start: u64,
        /// Where the range ends, past its last run.
        end: u64,
        /// How many variants there are.
        len: u64,
    },
}

/// The circuit elaborated for another structure than its batch's, which the batch's runs of
/// that structure run on, built once for the batch: where its unknowns lie, and the
/// batch's measures as they read its results.
#[derive(Debug)]
pub struct Restructured {
    key: StructureKey,
    elaborated: Arc<Elaborated>,
    /// The index among this circuit's unknowns of each of the batch circuit's, by its
    /// index; none where this one has none of its name.
    columns: Vec<Option<usize>>,
    /// Each of the batch's measures, reading the unknowns of the names it reads of the
    /// batch's circuit; an error where this circuit has none of a name.
    measures: Vec<Result<Measured, MeasureError>>,
}

impl PartialEq for Restructured {
    /// Whether two are of the same structure, which makes them the same circuit.
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Restructured {
    /// The structure's key against the batch's circuit.
    #[must_use]
    pub fn key(&self) -> &StructureKey {
        &self.key
    }

    /// The circuit elaborated for the structure.
    #[must_use]
    pub fn elaborated(&self) -> &Arc<Elaborated> {
        &self.elaborated
    }

    /// The index among this circuit's unknowns of the batch circuit's unknown at
    /// `unknown`; none when this one has no unknown of its name.
    #[must_use]
    pub fn column(&self, unknown: usize) -> Option<usize> {
        self.columns.get(unknown).copied().flatten()
    }
}

/// A circuit elaborated for each structure its runs meet, built once, by the key of the
/// structure against the batch's own circuit.
type Structures =
    BTreeMap<StructureKey, Arc<OnceLock<Result<Arc<Restructured>, ElaborationError>>>>;

/// Many runs of one circuit: the circuit, elaborated once, the analyses each run runs, in
/// order, each with its options, and the measures each run reads.
#[derive(Debug)]
pub struct Batch {
    elaborated: Arc<Elaborated>,
    analyses: Vec<(Analysis, Options)>,
    measures: Vec<Measured>,
    /// The circuit elaborated for each other structure the runs have met: the engine's one
    /// lock, held only to find or add a structure's place, never while it is built.
    structures: Mutex<Structures>,
}

/// What a thread keeps from one run to the next: a run of each structure it has met, by
/// the structure's key, with the circuit of each structure other than the batch's own, and
/// the structure of the run it ran last.
#[derive(Default)]
struct Worker {
    simulators: BTreeMap<StructureKey, (Simulator, Option<Arc<Restructured>>)>,
    last: Option<StructureKey>,
}

impl Batch {
    /// A batch of runs of `elaborated`, each running `analyses` in order, each with its
    /// options, and reading `measures`.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::NoAnalysis`] for a measure of an analysis the batch does not
    /// have.
    pub fn new(
        elaborated: Arc<Elaborated>,
        analyses: Vec<(Analysis, Options)>,
        measures: Vec<Measured>,
    ) -> Result<Self, BatchError> {
        if let Some((measure, measured)) = measures
            .iter()
            .enumerate()
            .find(|(_, measured)| measured.analysis >= analyses.len())
        {
            return Err(BatchError::NoAnalysis {
                measure,
                analysis: measured.analysis,
                analyses: analyses.len(),
            });
        }
        Ok(Self {
            elaborated,
            analyses,
            measures,
            structures: Mutex::new(BTreeMap::new()),
        })
    }

    /// How many structures other than the circuit's own the batch's runs have met, each
    /// elaborated once.
    #[must_use]
    pub fn structures(&self) -> usize {
        self.structures
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// Runs the runs `range` of `variants`, on the threads of the rayon pool it is called
    /// in, and hands each run's outcome to `sink` at its index; returns their summary.
    ///
    /// A run spreads its own work across threads, as its options' parallelism allows, only
    /// when the range holds fewer runs than the pool has threads; it orders its matrix by
    /// the minimum degree unless the options set the interiors of a dissection, whose
    /// default depends on the threads a run takes, so that every run finds the same bits
    /// however many threads the pool has and however the runs split into ranges.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::Range`] when the range does not lie within the variants.
    pub fn run(
        &self,
        variants: &dyn Variants,
        range: Range<u64>,
        sink: &dyn Sink,
    ) -> Result<Summary, BatchError> {
        let len = variants.len();
        if range.start > range.end || range.end > len {
            return Err(BatchError::Range {
                start: range.start,
                end: range.end,
                len,
            });
        }
        let measures = self.measures.len();
        if range.is_empty() {
            return Ok(Summary::empty(measures));
        }
        let threads = rayon::current_num_threads() as u64;
        let options: Vec<Options> = self
            .analyses
            .iter()
            .map(|(_, options)| {
                let mut options = options.clone();
                if range.end - range.start >= threads {
                    options.parallelism.max_threads = Some(1);
                }
                let interiors = &mut options.parallelism.max_unknowns_per_interior;
                *interiors = interiors.or(Some(usize::MAX));
                options
            })
            .collect();
        let chain = variants.chain().max(1);
        let chains = range.start / chain..range.end.div_ceil(chain);
        let summary = chains
            .into_par_iter()
            .map_init(Worker::default, |worker, chained| {
                let mut summary = Summary::empty(measures);
                let first = chained * chain;
                let mut warm = false;
                // The runs of the chain past the range's end are of no run within it.
                for index in first..(first + chain).min(range.end) {
                    let outcome = self.one(worker, variants.values(index), warm, &options);
                    warm = outcome.problems.is_none();
                    if range.contains(&index) {
                        summary.add(&outcome);
                        sink.accept(index, outcome);
                    }
                }
                summary
            })
            .reduce(|| Summary::empty(measures), Summary::merge);
        Ok(summary)
    }

    /// The outcome of one run at `values`, on a state `worker` keeps; `warm` when it starts
    /// from the operating point of the run before it, in its chain. Each analysis runs
    /// with its own `options`.
    fn one(&self, worker: &mut Worker, values: Values, warm: bool, options: &[Options]) -> Run {
        let (simulator, structure) = match self.ready(worker, &values, warm) {
            Ok(ready) => ready,
            Err(problems) => {
                return Run {
                    values,
                    structure: None,
                    problems: Some(problems),
                    results: Vec::new(),
                    measured: vec![Err(Unmeasured::Failed); self.measures.len()],
                };
            }
        };
        let results: Vec<Result<crate::AnalysisResult, SimulationError>> = self
            .analyses
            .iter()
            .zip(options)
            .map(|((analysis, _), options)| simulator.run(analysis, options))
            .collect();
        let read = |measured: &Measured| match &results[measured.analysis] {
            Ok(found) => {
                let read = AnalysisResult::from_result(found).ok_or(MeasureError::Kind)?;
                Ok(measured.measure.measure(&read)?)
            }
            Err(_) => Err(Unmeasured::Failed),
        };
        let measured = match &structure {
            None => self.measures.iter().map(read).collect(),
            Some(structure) => structure
                .measures
                .iter()
                .zip(&self.measures)
                .map(|(moved, measured)| match moved {
                    Ok(moved) => read(moved),
                    Err(error) => match results[measured.analysis] {
                        Ok(_) => Err(Unmeasured::Measure(*error)),
                        Err(_) => Err(Unmeasured::Failed),
                    },
                })
                .collect(),
        };
        Run {
            values,
            structure,
            problems: None,
            results,
            measured,
        }
    }

    /// A run at `values`, of the circuit elaborated for their structure, and that circuit
    /// when it is not the batch's own: the last run of `worker` gone on from its operating
    /// point when `warm` and it can, and else a run of the structure reset for them.
    ///
    /// # Errors
    ///
    /// Returns the problems the values make, as the batch's circuit finds them.
    fn ready<'a>(
        &self,
        worker: &'a mut Worker,
        values: &Values,
        warm: bool,
    ) -> Result<(&'a mut Simulator, Option<Arc<Restructured>>), ElaborationError> {
        // The problems of the values as the batch's own circuit finds them, every one, so
        // that a run's message does not depend on the state its thread kept.
        let problems =
            |error: ElaborationError| self.elaborated.structure_of(values).err().unwrap_or(error);
        let mut readied = None;
        if let Some(last) = &worker.last
            && let Some((simulator, _)) = worker.simulators.get_mut(last)
        {
            let gone_on = if warm {
                simulator.advance(values.clone())
            } else {
                Ok(false)
            };
            let ready = match gone_on {
                Ok(false) => simulator.try_reset(values.clone()),
                done => done,
            };
            match ready {
                Ok(true) => readied = Some(last.clone()),
                Ok(false) => {}
                Err(error) => return Err(problems(error)),
            }
        }
        let reset = readied.is_some();
        let key = match readied {
            Some(key) => key,
            None => self.elaborated.structure_of(values)?,
        };
        worker.last = Some(key.clone());
        match worker.simulators.entry(key) {
            Entry::Occupied(occupied) => {
                let (simulator, structure) = occupied.into_mut();
                if !reset {
                    // The simulator runs the structure the values give, so they reset it.
                    if !simulator.try_reset(values.clone()).map_err(problems)? {
                        return Err(problems(ElaborationError::new(Vec::new())));
                    }
                }
                Ok((simulator, structure.clone()))
            }
            Entry::Vacant(vacant) => {
                let key = vacant.key();
                let (elaborated, structure) = if key.is_empty() {
                    (Arc::clone(&self.elaborated), None)
                } else {
                    let structure = self.structure(key, values)?;
                    (Arc::clone(&structure.elaborated), Some(structure))
                };
                let simulator = Simulator::from_elaborated(elaborated, values.clone())?;
                let (simulator, structure) = vacant.insert((simulator, structure));
                Ok((simulator, structure.clone()))
            }
        }
    }

    /// The circuit elaborated for the structure `key` names, which `values` give: built
    /// once for the batch, by the first run that meets it, while the others that meet it
    /// wait for it.
    fn structure(
        &self,
        key: &StructureKey,
        values: &Values,
    ) -> Result<Arc<Restructured>, ElaborationError> {
        let cell = {
            let mut structures = self
                .structures
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            Arc::clone(structures.entry(key.clone()).or_default())
        };
        cell.get_or_init(|| {
            let elaborated = self.elaborated.restructured(values)?;
            Ok(Arc::new(self.restructured(key.clone(), elaborated)))
        })
        .clone()
    }

    /// The circuit `elaborated` for the structure `key`, with where each of the batch
    /// circuit's unknowns lies among its own, by name, and the measures that read them.
    fn restructured(&self, key: StructureKey, elaborated: Arc<Elaborated>) -> Restructured {
        let own: HashMap<&str, usize> = elaborated
            .unknowns()
            .iter()
            .enumerate()
            .map(|(index, unknown)| (unknown.name.as_str(), index))
            .collect();
        let columns: Vec<Option<usize>> = self
            .elaborated
            .unknowns()
            .iter()
            .map(|unknown| own.get(unknown.name.as_str()).copied())
            .collect();
        let measures = self
            .measures
            .iter()
            .map(|measured| {
                let column = |unknown: usize| columns.get(unknown).copied().flatten();
                Ok(Measured {
                    analysis: measured.analysis,
                    measure: measured.measure.moved(column)?,
                })
            })
            .collect();
        Restructured {
            key,
            elaborated,
            columns,
            measures,
        }
    }
}
