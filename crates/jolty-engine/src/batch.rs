//! Parallel batch executor running many parameter variants of one elaborated circuit across
//! the active rayon thread pool.
//!
//! A [`Batch`] binds a shared elaborated circuit, ordered `(analysis, options)` pairs, and
//! [measures](crate::measure). Calling [`Batch::run`] evaluates [`Variants`] over an index
//! range and delivers each [`Run`] to a [`Sink`]. Every run outcome is a pure function of
//! circuit, analyses, options, and run index, producing bit-identical results across thread
//! counts, range splits, and completion orders.
//!
//! - **Chains:** Consecutive blocks of [`Variants::chain`] runs execute sequentially on one
//!   thread, warm-starting each run from the prior operating point. Ranges starting
//!   mid-chain run from chain start and emit only indices inside the requested range.
//! - **States:** Each worker caches one simulator per encountered circuit structure and
//!   resets state between independent runs so unrelated runs never leak state.
//! - **Failures:** Invalid parameter values or failing analyses produce a failed [`Run`]
//!   describing the cause while the remaining runs continue.
//! - **Structures:** Runs whose parameter values alter circuit topology (such as a resistor
//!   crossing 1 Ω) execute on a cached restructured circuit built once per batch under a
//!   single lock. Unknown indices are remapped by name so measures and results stay aligned
//!   with the base circuit.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use rayon::prelude::*;

use crate::elaborate::ElaborationError;
use crate::measure::{AnalysisResult, Measure, MeasureError};
use crate::{Analysis, Elaborated, Options, SimulationError, Simulator, StructureKey, Values};

/// Pure index-addressed source of per-run parameter [`Values`] for a batch.
pub trait Variants: Sync {
    /// Returns the total number of runs.
    fn len(&self) -> u64;

    /// Returns whether the variant set contains zero runs.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the [`Values`] for run `index` (below [`Variants::len`]) as a pure function
    /// of `index`.
    fn values(&self, index: u64) -> Values;

    /// Returns the consecutive warm-start chain length (default `1` for independent runs).
    ///
    /// Chains start at multiples of this length, stepping smoothly along a single axis.
    fn chain(&self) -> u64 {
        1
    }
}

/// Callback receiving completed [`Run`] outcomes from a batch.
pub trait Sink: Sync {
    /// Consumes the completed `outcome` for run `index` (invoked once per run in arbitrary
    /// completion order).
    fn accept(&self, index: u64, outcome: Run);
}

/// Binding of a [`Measure`] to a target analysis index within a batch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measured {
    /// Zero-based index into the batch's analysis list.
    pub analysis: usize,
    /// Measure evaluated on that analysis's result.
    pub measure: Measure,
}

/// Outcome of a single batch run.
#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    /// Parameter and temperature values applied to this run.
    pub values: Values,
    /// Restructured circuit used when `values` altered topology, or `None` when the base
    /// circuit applied or elaboration failed.
    pub structure: Option<Arc<Restructured>>,
    /// Elaboration error preventing analysis execution, if any.
    pub problems: Option<ElaborationError>,
    /// Per-analysis results or errors in batch order (empty if `problems` is `Some`).
    pub results: Vec<Result<crate::AnalysisResult, SimulationError>>,
    /// Evaluated measure scalars or [`Unmeasured`] causes in batch order.
    pub measured: Vec<Result<f64, Unmeasured>>,
}

impl Run {
    /// Maps base-circuit unknown index `unknown` to the corresponding column index in this
    /// run's results, returning `None` if a restructured run lacks that unknown name.
    #[must_use]
    pub fn column(&self, unknown: usize) -> Option<usize> {
        column(self.structure.as_deref(), unknown)
    }

    /// Returns whether this run encountered elaboration problems or any analysis failure.
    #[must_use]
    pub fn failed(&self) -> bool {
        self.problems.is_some() || self.results.iter().any(Result::is_err)
    }

    /// Formats the elaboration problems or failing analysis errors for this run, returning
    /// `None` when all analyses succeeded.
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

/// Outcome of one run produced by [`Batch::map`].
#[derive(Debug, Clone, PartialEq)]
pub struct Mapped<T> {
    /// Parameter and temperature values applied to this run.
    pub values: Values,
    /// Restructured circuit used when `values` altered topology, or `None` otherwise.
    pub structure: Option<Arc<Restructured>>,
    /// Mapped return value or elaboration error for this run.
    pub found: Result<T, ElaborationError>,
}

impl<T> Mapped<T> {
    /// Maps base-circuit unknown index `unknown` to this run's unknown index, matching
    /// [`Run::column`].
    #[must_use]
    pub fn column(&self, unknown: usize) -> Option<usize> {
        column(self.structure.as_deref(), unknown)
    }
}

/// Maps base-circuit unknown index `unknown` through optional `structure`.
fn column(structure: Option<&Restructured>, unknown: usize) -> Option<usize> {
    match structure {
        None => Some(unknown),
        Some(structure) => structure.column(unknown),
    }
}

/// Validates that `range` lies within `0..variants.len()`.
///
/// # Errors
///
/// Returns [`BatchError::Range`] when `range` is reversed or exceeds `variants.len()`.
fn within(variants: &dyn Variants, range: &Range<u64>) -> Result<(), BatchError> {
    let length = variants.len();
    if range.start > range.end || range.end > length {
        return Err(BatchError::Range {
            start: range.start,
            stop: range.end,
            length,
        });
    }
    Ok(())
}

/// Adapts `options` for a batch of `runs` so results stay bit-identical across thread
/// counts and range splits.
///
/// Forces single-threaded per-run execution once `runs` saturate the pool, and disables
/// automatic nested dissection unless explicitly configured.
fn batched(options: &Options, runs: u64) -> Options {
    let mut options = options.clone();
    if runs >= rayon::current_num_threads() as u64 {
        options.parallelism.max_threads = Some(1);
    }
    let interiors = &mut options.parallelism.max_unknowns_per_interior;
    *interiors = interiors.or(Some(usize::MAX));
    options
}

/// Reason a batch measure produced no numeric result.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum Unmeasured {
    /// Elaboration failed or the target analysis returned an error.
    #[error("the analysis it reads did not run to its end")]
    Failed,
    /// Measure evaluation failed on the completed analysis result.
    #[error(transparent)]
    Measure(#[from] MeasureError),
}

/// Exact aggregate statistics across batch runs that [merge](Summary::merge) across
/// sub-ranges bit for bit.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    /// Total runs counted.
    pub runs: u64,
    /// Number of runs that failed elaboration or analysis.
    pub failed: u64,
    /// Per-measure extrema across counted runs in batch order.
    pub measures: Vec<Extremes>,
}

/// Count and exact min/max extrema of one measure across batch runs.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Extremes {
    /// Number of runs where the measure succeeded.
    pub measured: u64,
    /// Minimum measured value under `f64::total_cmp`, or `None` if unmeasured.
    pub minimum: Option<f64>,
    /// Maximum measured value under `f64::total_cmp`, or `None` if unmeasured.
    pub maximum: Option<f64>,
}

impl Extremes {
    /// Returns updated extrema incorporating one additional measurement `value`.
    fn with(self, value: f64) -> Self {
        let least = |a: f64, b: f64| if b.total_cmp(&a).is_lt() { b } else { a };
        let most = |a: f64, b: f64| if b.total_cmp(&a).is_gt() { b } else { a };
        Self {
            measured: self.measured + 1,
            minimum: Some(self.minimum.map_or(value, |minimum| least(minimum, value))),
            maximum: Some(self.maximum.map_or(value, |maximum| most(maximum, value))),
        }
    }

    /// Merges two [`Extremes`] accumulators into combined count and bounds.
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
    /// Creates an empty [`Summary`] initialized for `measures` measure slots.
    #[must_use]
    pub fn empty(measures: usize) -> Self {
        Self {
            runs: 0,
            failed: 0,
            measures: vec![Extremes::default(); measures],
        }
    }

    /// Combines two [`Summary`] instances by summing counts and merging measure extrema.
    #[must_use]
    pub fn merge(mut self, other: Self) -> Self {
        self.runs += other.runs;
        self.failed += other.failed;
        for (mine, theirs) in self.measures.iter_mut().zip(other.measures) {
            *mine = mine.merge(theirs);
        }
        self
    }

    /// Incorporates one completed `outcome` into the summary counters and extrema.
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

/// Configuration or range error preventing batch execution.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BatchError {
    /// Measure references an out-of-range analysis index.
    #[error("measure {measure} reads analysis {analysis}, but the batch runs {analyses}")]
    NoAnalysis {
        /// Index of the offending measure.
        measure: usize,
        /// Referenced analysis index.
        analysis: usize,
        /// Total analyses configured on the batch.
        analyses: usize,
    },
    /// Requested run range falls outside the available variants.
    #[error("runs {start} to {stop} do not lie within the {length} variants")]
    Range {
        /// Inclusive start run index.
        start: u64,
        /// Exclusive stop run index.
        stop: u64,
        /// Total available variants.
        length: u64,
    },
}

/// Cached restructured circuit built once per distinct [`StructureKey`] encountered in a
/// batch, together with unknown-index and measure remappings.
#[derive(Debug)]
pub struct Restructured {
    key: StructureKey,
    elaborated: Arc<Elaborated>,
    /// Mapping from base-circuit unknown index to restructured unknown index (`None` if
    /// absent).
    columns: Vec<Option<usize>>,
    /// Batch measures remapped to this structure's unknown indices, or errors for missing
    /// unknowns.
    measures: Vec<Result<Measured, MeasureError>>,
}

impl PartialEq for Restructured {
    /// Returns whether two restructured circuits share the same [`StructureKey`].
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Restructured {
    /// Returns the [`StructureKey`] relative to the base batch circuit.
    #[must_use]
    pub fn key(&self) -> &StructureKey {
        &self.key
    }

    /// Returns the [`Elaborated`] circuit built for this structure.
    #[must_use]
    pub fn elaborated(&self) -> &Arc<Elaborated> {
        &self.elaborated
    }

    /// Maps base-circuit unknown index `unknown` to this structure's unknown index, or
    /// `None` if absent.
    #[must_use]
    pub fn column(&self, unknown: usize) -> Option<usize> {
        self.columns.get(unknown).copied().flatten()
    }
}

/// Cache of lazily constructed restructured circuits keyed by [`StructureKey`].
type Structures =
    BTreeMap<StructureKey, Arc<OnceLock<Result<Arc<Restructured>, ElaborationError>>>>;

/// Reusable batch executor holding an elaborated circuit, ordered analyses with options,
/// measures, and a shared restructured-circuit cache.
#[derive(Debug)]
pub struct Batch {
    elaborated: Arc<Elaborated>,
    analyses: Vec<(Analysis, Options)>,
    measures: Vec<Measured>,
    /// Single mutex protecting the map of [`OnceLock`] slots (never held during
    /// elaboration).
    structures: Mutex<Structures>,
}

/// Flag indicating whether a chain run starts fresh or warm-starts from the prior run.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Start {
    Cold,
    Warm,
}

/// Prepared simulator and optional restructured circuit metadata for a single batch run.
type Ready<'a> = Result<(&'a mut Simulator, Option<Arc<Restructured>>), ElaborationError>;

/// Per-worker simulator cache keyed by [`StructureKey`] together with the last executed
/// key.
#[derive(Default)]
struct Worker {
    simulators: BTreeMap<StructureKey, (Simulator, Option<Arc<Restructured>>)>,
    last: Option<StructureKey>,
}

impl Batch {
    /// Creates a [`Batch`] for `elaborated` with ordered `analyses` and `measures`.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::NoAnalysis`] when any measure references an out-of-range
    /// analysis index.
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

    /// Returns the number of distinct restructured circuits built during execution.
    #[must_use]
    pub fn structures(&self) -> usize {
        self.structures
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// Executes `range` of `variants` across the active rayon pool, delivering each [`Run`]
    /// to `sink` and returning the combined [`Summary`].
    ///
    /// Per-run parallelism activates only when `range` has fewer runs than pool threads,
    /// and automatic dissection is disabled unless explicitly set so results stay
    /// bit-identical across thread counts and range splits.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::Range`] when `range` lies outside `variants`.
    pub fn run(
        &self,
        variants: &dyn Variants,
        range: Range<u64>,
        sink: &dyn Sink,
    ) -> Result<Summary, BatchError> {
        within(variants, &range)?;
        let measures = self.measures.len();
        if range.is_empty() {
            return Ok(Summary::empty(measures));
        }
        let options: Vec<Options> = self
            .analyses
            .iter()
            .map(|(_, options)| batched(options, range.end - range.start))
            .collect();
        let summary = self
            .run_chains(
                variants,
                range,
                || Summary::empty(measures),
                |target, index, values, ready| {
                    let outcome = self.one(values, ready, &options);
                    if let Some(summary) = target {
                        summary.add(&outcome);
                        sink.accept(index, outcome);
                    }
                },
            )
            .reduce(|| Summary::empty(measures), Summary::merge);
        Ok(summary)
    }

    /// Executes custom `work` on every run in `range` of `variants` in parallel, returning
    /// ordered [`Mapped`] results.
    ///
    /// Prepares simulators in warm-started chains and restructured circuits exactly as
    /// [`Batch::run`] does, passing adjusted `options` for thread-count invariance. Used for
    /// custom batch operations such as per-run derivatives and pullbacks.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::Range`] when `range` lies outside `variants`.
    pub fn map<T: Send>(
        &self,
        variants: &dyn Variants,
        range: Range<u64>,
        options: &Options,
        work: &(dyn Fn(u64, &mut Simulator, &Options) -> T + Sync),
    ) -> Result<Vec<Mapped<T>>, BatchError> {
        within(variants, &range)?;
        if range.is_empty() {
            return Ok(Vec::new());
        }
        let options = batched(options, range.end - range.start);
        let mapped: Vec<Vec<Mapped<T>>> = self
            .run_chains(variants, range, Vec::new, |target, index, values, ready| {
                let (structure, found) = match ready {
                    Ok((simulator, structure)) => (structure, Ok(work(index, simulator, &options))),
                    Err(problems) => (None, Err(problems)),
                };
                if let Some(collected) = target {
                    collected.push(Mapped {
                        values,
                        structure,
                        found,
                    });
                }
            })
            .collect();
        Ok(mapped.into_iter().flatten().collect())
    }

    /// Runs every chain covering `range` of `variants` in parallel, warm-starting each run
    /// from the previous valid run in its chain, and yields each chain's accumulated value.
    fn run_chains<'a, A>(
        &'a self,
        variants: &'a dyn Variants,
        range: Range<u64>,
        init: impl Fn() -> A + Send + Sync + 'a,
        each: impl Fn(Option<&mut A>, u64, Values, Ready<'_>) + Send + Sync + 'a,
    ) -> impl ParallelIterator<Item = A> + 'a
    where
        A: Send + 'a,
    {
        let chain = variants.chain().max(1);
        let chains = range.start / chain..range.end.div_ceil(chain);
        chains
            .into_par_iter()
            .map_init(Worker::default, move |worker, chained| {
                let mut accumulator = init();
                let first = chained * chain;
                let mut start = Start::Cold;
                // Stop at `range.end` because trailing runs in the final chain are unused.
                for index in first..(first + chain).min(range.end) {
                    let values = variants.values(index);
                    let ready = self.ready(worker, &values, start);
                    start = if ready.is_ok() {
                        Start::Warm
                    } else {
                        Start::Cold
                    };
                    each(
                        range.contains(&index).then_some(&mut accumulator),
                        index,
                        values,
                        ready,
                    );
                }
                accumulator
            })
    }

    /// Executes all batch analyses and measures for one prepared run at `values`.
    fn one(&self, values: Values, ready: Ready<'_>, options: &[Options]) -> Run {
        let (simulator, structure) = match ready {
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

    /// Prepares a simulator on `worker` for `values` using warm advance when `start` is
    /// [`Start::Warm`] and falling back to reset or restructuring as needed.
    ///
    /// # Errors
    ///
    /// Returns elaboration problems reported against the base batch circuit for
    /// thread-independent diagnostics.
    fn ready<'a>(&self, worker: &'a mut Worker, values: &Values, start: Start) -> Ready<'a> {
        // Compute problems against the base circuit so diagnostics never depend on worker
        // cache history.
        let problems =
            |error: ElaborationError| self.elaborated.structure_of(values).err().unwrap_or(error);
        let mut readied = None;
        if let Some(last) = &worker.last
            && let Some((simulator, _)) = worker.simulators.get_mut(last)
        {
            let gone_on = match start {
                Start::Warm => simulator.advance(values.clone()),
                Start::Cold => Ok(false),
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
                    // The cached simulator already matches this structure, so reset in place.
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

    /// Returns the shared [`Restructured`] circuit for `key`, elaborating it once on first
    /// encounter while concurrent runs wait on the per-key [`OnceLock`].
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

    /// Builds a [`Restructured`] wrapper for `key` and `elaborated`, mapping base unknown
    /// indices and batch measures by unknown name.
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
