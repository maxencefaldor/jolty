//! The batch executor: many runs of one circuit in one call, spread across the threads of
//! the rayon pool it runs in.
//!
//! A [`Batch`] holds a circuit elaborated once, the analyses each run runs in order, their
//! options, and the [measures](crate::measure) each run reads of their results. Its
//! [`Batch::run`] takes [`Variants`], the values of each run as a pure function of the
//! run's index, and a range of indices, and hands each run's [`Outcome`] to a [`Sink`] at
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
//!   lock of the engine.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::ops::Range;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use rayon::prelude::*;

use crate::elaborate::ElaborationError;
use crate::measure::{Measure, MeasureError, Results};
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
    fn accept(&self, index: u64, outcome: Outcome);
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
pub struct Outcome {
    /// The problems the run's values make, which no analysis runs past; none when they make
    /// none.
    pub problems: Option<ElaborationError>,
    /// What each analysis found, in the batch's order, or why it failed; none when the
    /// values make problems.
    pub results: Vec<Result<crate::Outcome, SimulationError>>,
    /// What each measure read, in the batch's order, or why it read nothing.
    pub measured: Vec<Result<f64, Unmeasured>>,
}

impl Outcome {
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
    fn add(&mut self, outcome: &Outcome) {
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

/// A circuit elaborated for each structure its runs meet, built once, by the key of the
/// structure against the batch's own circuit.
type Structures = BTreeMap<StructureKey, Arc<OnceLock<Result<Arc<Elaborated>, ElaborationError>>>>;

/// Many runs of one circuit: the circuit, elaborated once, the analyses each run runs, in
/// order, their options, and the measures each run reads.
#[derive(Debug)]
pub struct Batch {
    elaborated: Arc<Elaborated>,
    analyses: Vec<Analysis>,
    options: Options,
    measures: Vec<Measured>,
    /// The circuit elaborated for each other structure the runs have met: the engine's one
    /// lock, held only to find or add a structure's place, never while it is built.
    structures: Mutex<Structures>,
}

/// What a thread keeps from one run to the next: a run of each structure it has met, by
/// the structure's key, and the structure of the run it ran last.
#[derive(Default)]
struct Worker {
    simulators: BTreeMap<StructureKey, Simulator>,
    last: Option<StructureKey>,
}

impl Batch {
    /// A batch of runs of `elaborated`, each running `analyses` in order with `options`, and
    /// reading `measures`.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::NoAnalysis`] for a measure of an analysis the batch does not
    /// have.
    pub fn new(
        elaborated: Arc<Elaborated>,
        analyses: Vec<Analysis>,
        options: Options,
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
            options,
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
    /// when the range holds fewer runs than the pool has threads.
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
        let mut options = self.options.clone();
        let threads = rayon::current_num_threads() as u64;
        if range.end - range.start >= threads {
            options.parallelism.threads = Some(1);
        }
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
                    let outcome = self.one(worker, &variants.values(index), warm, &options);
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
    /// from the operating point of the run before it, in its chain.
    fn one(&self, worker: &mut Worker, values: &Values, warm: bool, options: &Options) -> Outcome {
        let simulator = match self.ready(worker, values, warm) {
            Ok(simulator) => simulator,
            Err(problems) => {
                return Outcome {
                    problems: Some(problems),
                    results: Vec::new(),
                    measured: vec![Err(Unmeasured::Failed); self.measures.len()],
                };
            }
        };
        let results: Vec<Result<crate::Outcome, SimulationError>> = self
            .analyses
            .iter()
            .map(|analysis| simulator.run(analysis, options))
            .collect();
        let measured = self
            .measures
            .iter()
            .map(|measured| match &results[measured.analysis] {
                Ok(found) => {
                    let read = Results::of(found).ok_or(MeasureError::Kind)?;
                    Ok(measured.measure.measure(&read)?)
                }
                Err(_) => Err(Unmeasured::Failed),
            })
            .collect();
        Outcome {
            problems: None,
            results,
            measured,
        }
    }

    /// A run at `values`, of the circuit elaborated for their structure: the last run of
    /// `worker` gone on from its operating point when `warm` and it can, and else a run of
    /// the structure reset for them.
    ///
    /// # Errors
    ///
    /// Returns the problems the values make, as the batch's circuit finds them.
    fn ready<'a>(
        &self,
        worker: &'a mut Worker,
        values: &Values,
        warm: bool,
    ) -> Result<&'a mut Simulator, ElaborationError> {
        // The problems of the values as the batch's own circuit finds them, every one, so
        // that a run's message does not depend on the state its thread kept.
        let problems =
            |error: ElaborationError| self.elaborated.structure_of(values).err().unwrap_or(error);
        let mut readied = None;
        if let Some(last) = &worker.last
            && let Some(simulator) = worker.simulators.get_mut(last)
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
                let simulator = occupied.into_mut();
                if !reset {
                    // The simulator runs the structure the values give, so they reset it.
                    if !simulator.try_reset(values.clone()).map_err(problems)? {
                        return Err(problems(ElaborationError::new(Vec::new())));
                    }
                }
                Ok(simulator)
            }
            Entry::Vacant(vacant) => {
                let key = vacant.key();
                let elaborated = if key.is_empty() {
                    Arc::clone(&self.elaborated)
                } else {
                    self.structure(key, values)?
                };
                Ok(vacant.insert(Simulator::from_elaborated(elaborated, values.clone())?))
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
    ) -> Result<Arc<Elaborated>, ElaborationError> {
        let cell = {
            let mut structures = self
                .structures
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            Arc::clone(structures.entry(key.clone()).or_default())
        };
        cell.get_or_init(|| self.elaborated.restructured(values))
            .clone()
    }
}
