use std::collections::{BTreeMap, BTreeSet};

use jolty_models::{Nature, Tasks};
use jolty_numerics::{Dissecting, Ordering, reduce};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;

use crate::SimulationError;

/// Number of 8-byte IEEE 754 `f64` values per gibibyte (`2^30 / 8 = 134_217_728`, IEC
/// 80000-13 binary prefix).
pub(crate) const NUMBERS_PER_GIB: f64 = 134_217_728.0;

/// Converts element count `size` to `f64` for memory-limit calculations (exact up to `2^53`).
#[expect(
    clippy::cast_precision_loss,
    reason = "memory-limit counts need only floating-point precision"
)]
pub(crate) fn float_count(size: usize) -> f64 {
    size as f64
}

/// Measured break-even thresholds for [`Parallelism`] work-splitting options across thread
/// counts.
///
/// For each count in [`BreakEvens::threads`], entries give the minimum task size where
/// parallel execution outperformed serial execution across speed-suite benchmark circuits,
/// or `None` when no measured size benefited. Measurements come from
/// `scripts/bench --break-even` on a 32-core / 64-thread AMD Ryzen Threadripper 3970X.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BreakEvens {
    /// Thread counts at which break-even thresholds were measured, in ascending order.
    pub threads: [usize; 4],
    /// Measured defaults for [`Parallelism::min_solves_per_task`] across 1 to 1,111
    /// frequencies per decade on the amplifier fixture.
    pub min_solves_per_task: [Option<usize>; 4],
    /// Measured defaults for [`Parallelism::min_instances_per_task`] on chains of 1 to
    /// 1,000 inverters.
    pub min_instances_per_task: [Option<usize>; 4],
    /// Measured defaults for [`Parallelism::min_unknowns_per_task`] on 100 to 2,000
    /// inverter chains and 10x10 to 300x300 meshes.
    pub min_unknowns_per_task: [Option<usize>; 4],
    /// Measured defaults for [`Parallelism::max_unknowns_per_interior`] on 10x10 to
    /// 300x300 meshes, giving the maximum interior size where block dissection paid off.
    pub max_unknowns_per_interior: [Option<usize>; 4],
}

impl BreakEvens {
    /// Returns the break-even threshold from `measured` for `threads`, picking the highest
    /// measured thread count not exceeding `threads`, or `None` on one thread.
    #[must_use]
    pub fn on(&self, measured: [Option<usize>; 4], threads: usize) -> Option<usize> {
        self.threads
            .iter()
            .zip(measured)
            .rev()
            .find(|&(&each, _)| each <= threads)
            .and_then(|(_, least)| least)
    }

    /// Returns the smallest defined threshold across all measured thread counts.
    fn least(measured: [Option<usize>; 4]) -> Option<usize> {
        measured.into_iter().flatten().min()
    }
}

/// Default parallelism break-even thresholds measured on 2, 4, 8, and 64 threads.
///
/// Source: `scripts/bench --break-even` on `cedar` (AMD Ryzen Threadripper 3970X, 32 cores
/// / 64 threads, 16 November 2026); see `docs/guide/simulation.md` and
/// `benchmarks/speed/README.md`.
pub const BREAK_EVENS: BreakEvens = BreakEvens {
    threads: [2, 4, 8, 64],
    min_solves_per_task: [Some(4_999), Some(11), Some(11), Some(4)],
    min_instances_per_task: [None, Some(250), Some(125), None],
    min_unknowns_per_task: [None, None, None, None],
    max_unknowns_per_interior: [Some(5_001), Some(2_501), Some(1_251), Some(157)],
};

/// Selection of devices whose terminal currents, charges, derivatives, and internal
/// variables are recorded alongside circuit unknowns.
///
/// Serializes as `"all"` or `{"instances": ["M1", "X1"]}` in JSON.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Record {
    /// Records every analog device and every summed component or logic terminal.
    All,
    /// Records devices and component/logic terminals at the listed hierarchical paths (such
    /// as `X1/M1` or `X1`); unmatched paths record nothing.
    Instances(BTreeSet<String>),
}

impl Default for Record {
    /// Returns an empty recording selection capturing circuit unknowns only.
    fn default() -> Self {
        Self::Instances(BTreeSet::new())
    }
}

impl Record {
    /// Returns whether this recording configuration selects `path`.
    #[must_use]
    pub fn names(&self, path: &str) -> bool {
        match self {
            Self::All => true,
            Self::Instances(paths) => paths.contains(path),
        }
    }
}

/// Configuration for distributing work within a single simulation across threads in the
/// active rayon pool.
///
/// Covers small-signal frequencies, sensitivity right-hand sides, device evaluation,
/// Newton/step-control vector passes, and nested-dissection sparse factorization. Every
/// setting except [`Parallelism::max_unknowns_per_interior`] preserves bit-identical output
/// across thread counts; changing interior size alters elimination order and agrees within
/// floating-point rounding.
///
/// The caller owns the rayon pool and enters it with `ThreadPool::install`. When
/// `max_threads` is smaller than the caller pool, the engine runs in a dedicated sub-pool
/// of that size; a [batch](crate::batch::Batch) runs single-threaded per run when runs
/// already saturate the pool. Each work category parallelizes only once items fill a full
/// threshold task per thread, defaulting to [`BREAK_EVENS`].
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Parallelism {
    /// Maximum worker threads for a single run (`None` uses all pool threads; `Some(1)`
    /// forces serial execution). Values of two or more below the caller pool size run in a
    /// dedicated sub-pool of that thread count.
    pub max_threads: Option<usize>,
    /// Minimum frequencies, right-hand sides, or parameter derivatives per parallel task
    /// (at most one task per thread); defaults to [`BREAK_EVENS`].
    pub min_solves_per_task: Option<usize>,
    /// Minimum nonlinear device instances per parallel evaluation task (at most one task
    /// per thread); linear groups run in one task. Defaults to [`BREAK_EVENS`].
    pub min_instances_per_task: Option<usize>,
    /// Minimum unknowns or matrix nonzeros per task for parallel Newton and step-control
    /// passes, scheduled in [`jolty_numerics::reduce::CHUNK`] blocks; defaults to
    /// [`BREAK_EVENS`].
    pub min_unknowns_per_task: Option<usize>,
    /// Maximum interior unknown count for nested dissection of large block-triangular
    /// matrix blocks. Blocks exceeding this size use nested dissection when
    /// [`Dissecting::Cheaper`] estimates fewer multiply-adds than minimum degree; otherwise
    /// serial minimum degree is used. Changing dissection alters elimination order within
    /// factorization rounding while staying bit-identical across thread counts for a fixed
    /// ordering. Defaults to [`BREAK_EVENS`].
    pub max_unknowns_per_interior: Option<usize>,
}

impl Default for Parallelism {
    /// Returns default parallelism using all pool threads and measured break-even
    /// thresholds.
    fn default() -> Self {
        Self {
            max_threads: None,
            min_solves_per_task: None,
            min_instances_per_task: None,
            min_unknowns_per_task: None,
            max_unknowns_per_interior: None,
        }
    }
}

impl Parallelism {
    /// Computes the number of parallel tasks for `items` independent solves, returning `1`
    /// unless `items` fill one [`Parallelism::min_solves_per_task`] task per usable thread.
    #[must_use]
    pub fn tasks(&self, items: usize) -> usize {
        self.split(Kind::Solves, items)
            .map_or(1, |tasks| tasks.count(items))
    }

    /// Returns the minimum items per task for `kind` on `threads`, falling back to
    /// [`BREAK_EVENS`], or `None` to stay serial.
    fn least(&self, kind: Kind, threads: usize) -> Option<usize> {
        let (set, measured) = self.threshold(kind);
        set.or_else(|| BREAK_EVENS.on(measured, threads))
    }

    /// Returns the explicit threshold and measured break-even table for `kind`.
    const fn threshold(&self, kind: Kind) -> (Option<usize>, [Option<usize>; 4]) {
        match kind {
            Kind::Solves => (self.min_solves_per_task, BREAK_EVENS.min_solves_per_task),
            Kind::Instances => (
                self.min_instances_per_task,
                BREAK_EVENS.min_instances_per_task,
            ),
            Kind::Unknowns => (
                self.min_unknowns_per_task,
                BREAK_EVENS.min_unknowns_per_task,
            ),
        }
    }

    /// Builds a [`Tasks`] plan of at least `least` items per thread when `threads >= 2` and
    /// `items` fill every thread, returning `None` otherwise.
    fn filled(least: usize, items: usize, threads: usize) -> Option<Tasks> {
        let tasks = tasks_of(least, threads);
        (threads >= 2 && items / tasks.least >= threads).then_some(tasks)
    }

    /// Computes a [`Tasks`] split for `kind` and `items` across usable threads, or `None`
    /// to run serially.
    ///
    /// Small workloads that cannot fill two tasks return early without querying rayon,
    /// avoiding lazy initialization of rayon's global pool.
    fn split(&self, kind: Kind, items: usize) -> Option<Tasks> {
        let (set, measured) = self.threshold(kind);
        let fewest = set.or_else(|| BreakEvens::least(measured))?;
        Self::filled(fewest, items, 2)?;
        let threads = self.usable();
        Self::filled(self.least(kind, threads)?, items, threads)
    }

    /// Returns whether any work category in `sizes` meets its parallel threshold on
    /// `threads` (capped by [`Parallelism::max_threads`]).
    #[must_use]
    pub fn fills_threads(&self, sizes: Sizes, threads: usize) -> bool {
        let threads = self.max_threads.map_or(threads, |most| most.min(threads));
        let fills = |kind, items| {
            self.least(kind, threads)
                .and_then(|least| Self::filled(least, items, threads))
                .is_some()
        };
        self.spreads()
            && threads >= 2
            && (fills(Kind::Instances, sizes.instances)
                || fills(Kind::Unknowns, sizes.unknowns)
                || sizes.dissected
                || fills(Kind::Solves, sizes.solves))
    }

    /// Returns the usable thread count: [`Parallelism::max_threads`] capped by the active
    /// rayon pool size.
    fn usable(&self) -> usize {
        let threads = rayon::current_num_threads();
        self.max_threads.map_or(threads, |most| most.min(threads))
    }

    /// Executes `run` confined to at most [`Parallelism::max_threads`] threads.
    ///
    /// When `max_threads` is at least 2 and smaller than the enclosing rayon pool, builds a
    /// dedicated pool of that size so nested rayon tasks stay bounded. Otherwise runs on
    /// the current thread or enclosing pool.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Threads`] when worker threads cannot start, or any error
    /// returned by `run`.
    pub(crate) fn confined<T: Send>(
        &self,
        run: impl FnOnce() -> Result<T, SimulationError> + Send,
    ) -> Result<T, SimulationError> {
        // Serial runs return directly without querying rayon, avoiding global pool startup.
        let Some(most) = self.max_threads.filter(|&most| most >= 2) else {
            return run();
        };
        if most >= rayon::current_num_threads() {
            return run();
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(most)
            .build()
            .map_err(|error| SimulationError::Threads {
                threads: most,
                reason: error.to_string(),
            })?;
        pool.install(run)
    }

    /// Returns whether this configuration permits multi-threaded execution (true unless
    /// [`Parallelism::max_threads`] is 1).
    #[must_use]
    pub const fn spreads(&self) -> bool {
        !matches!(self.max_threads, Some(..2))
    }

    /// Chooses the [`Dissecting`] policy for a matrix whose largest block has
    /// `largest_block` unknowns, returning `None` for serial minimum degree.
    ///
    /// Blocks below every possible interior threshold return `None` without querying rayon,
    /// avoiding lazy initialization of rayon's global pool.
    #[must_use]
    pub(crate) fn dissecting(&self, largest_block: usize) -> Option<Dissecting> {
        let measured = BREAK_EVENS.max_unknowns_per_interior;
        let most = self
            .max_unknowns_per_interior
            .or_else(|| measured.into_iter().flatten().min())?;
        if largest_block <= most {
            return None;
        }
        self.dissecting_on(self.usable())
    }

    /// Chooses the [`Dissecting`] policy for `threads` workers, matching
    /// [`Parallelism::dissecting`].
    #[must_use]
    pub(crate) fn dissecting_on(&self, threads: usize) -> Option<Dissecting> {
        self.max_unknowns_per_interior
            .or_else(|| BREAK_EVENS.on(BREAK_EVENS.max_unknowns_per_interior, threads))
            .map(Dissecting::Cheaper)
    }

    /// Returns the thread count used to factor `ordering`: all usable threads when
    /// dissected, otherwise `1`.
    #[must_use]
    pub(crate) fn factorization(&self, ordering: &Ordering) -> usize {
        if self.spreads() && ordering.is_dissected() {
            self.usable()
        } else {
            1
        }
    }

    /// Returns a [`Tasks`] split for parallel device evaluation over `instances` nonlinear
    /// devices, or `None` to evaluate serially.
    ///
    /// Nonlinear groups split into tasks of at least [`Parallelism::min_instances_per_task`]
    /// instances via [`Tasks::count`] with balanced gather partitions, producing
    /// bit-identical results to serial evaluation.
    #[must_use]
    pub fn evaluation(&self, instances: usize) -> Option<Tasks> {
        self.split(Kind::Instances, instances)
    }

    /// Returns an [`UnknownGrain`] when `unknowns` meet [`Parallelism::min_unknowns_per_task`]
    /// across usable threads, or `None` for serial Newton and step-control loops.
    #[must_use]
    pub(crate) fn unknown_grain(&self, unknowns: usize) -> Option<UnknownGrain> {
        self.split(Kind::Unknowns, unknowns)
            .map(|tasks| UnknownGrain {
                chunks: tasks.least.get().div_ceil(reduce::CHUNK),
            })
    }

    /// Applies `each` to every item in `items` in order across [`Parallelism::tasks`]
    /// contiguous chunks, providing per-task `scratch` state while preserving bit-identical
    /// results.
    pub(crate) fn spread<T: Send, S, R: Send>(
        &self,
        items: &mut [T],
        scratch: impl Fn() -> S + Sync,
        each: impl Fn(&mut S, &mut T) -> R + Sync,
    ) -> Vec<R> {
        let tasks = self.tasks(items.len());
        if tasks == 1 {
            let mut space = scratch();
            return items
                .iter_mut()
                .map(|item| each(&mut space, item))
                .collect();
        }
        let size = items.len().div_ceil(tasks).max(1);
        let parts: Vec<Vec<R>> = items
            .par_chunks_mut(size)
            .map(|part| {
                let mut space = scratch();
                part.iter_mut().map(|item| each(&mut space, item)).collect()
            })
            .collect();
        parts.into_iter().flatten().collect()
    }
}

/// Work category governed by a threshold in [`Parallelism`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Frequencies, right-hand sides, or parameter derivatives.
    Solves,
    /// Nonlinear device instances.
    Instances,
    /// Unknowns and matrix nonzeros.
    Unknowns,
}

/// Constructs a [`Tasks`] descriptor with at least `least` items per task and `threads`
/// workers (clamping both to at least 1).
fn tasks_of(least: usize, threads: usize) -> Tasks {
    Tasks {
        least: NonZeroUsize::new(least).unwrap_or(NonZeroUsize::MIN),
        threads: NonZeroUsize::new(threads).unwrap_or(NonZeroUsize::MIN),
    }
}

/// Workload sizes inspected by [`Parallelism::fills_threads`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sizes {
    /// Count of evaluated nonlinear device instances.
    pub instances: usize,
    /// Count of circuit unknowns.
    pub unknowns: usize,
    /// Flag indicating whether the matrix ordering dissects at least one BTF block.
    pub dissected: bool,
    /// Count of independent solves (frequencies or right-hand sides).
    pub solves: usize,
}

/// Granularity for parallel unknown and matrix-entry loops, dividing work into
/// [`jolty_numerics::reduce::CHUNK`] blocks with at least `chunks` blocks per task while
/// combining reductions in deterministic order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UnknownGrain {
    /// Minimum number of chunks per parallel task.
    pub(crate) chunks: usize,
}

/// Domain in which transient local truncation error is judged against tolerances.
///
/// Both modes solve `(G + αC) moved = α d` to filter fast decaying modes through the step
/// equations (Shampine, 1980). [`ErrorControl::Charge`] multiplies back by `C` and bounds
/// filtered charge error against SPICE charge and step-scaled current tolerances.
/// [`ErrorControl::Unknown`] skips that product and bounds `moved` directly against
/// [`Options::relative_tolerance`] and each unknown's voltage or current tolerance, free of
/// [`Options::charge_tolerance`]. Index-2 algebraic current unknowns (current unknowns whose
/// column in `C` is zero, such as non-inductive voltage-source currents and MOSFET Meyer
/// displacement currents) are excluded from the unknown-space error norm because their
/// filtered error scales as `O(1/h)` (Hairer & Wanner, 1996, VII.4; Brenan, Campbell, and
/// Petzold, 1996, §5.2).
///
/// Serializes as `"charge"` or `"unknown"` in JSON.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorControl {
    /// Judge filtered charge error `C (G + αC)⁻¹ α d` against SPICE charge and step-scaled
    /// current tolerances (default).
    #[default]
    Charge,
    /// Judge filtered unknown error `(G + αC)⁻¹ α d` directly against unknown tolerances,
    /// excluding index-2 algebraic current unknowns and free of any charge floor.
    Unknown,
}

/// Numerical solver settings, iteration limits, recording options, and parallelism
/// configuration.
///
/// Defaults match SPICE so simulations compare directly against SPICE defaults.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    /// Relative tolerance applied to unknowns and residuals (default `1e-3`).
    pub relative_tolerance: f64,
    /// Absolute voltage tolerance in volts (default `1e-6`).
    pub voltage_tolerance: f64,
    /// Absolute current tolerance in amperes (default `1e-12`).
    pub current_tolerance: f64,
    /// Minimum reference charge floor in coulombs, matching SPICE `CHGTOL` (default
    /// `1e-14`).
    ///
    /// Under [`ErrorControl::Charge`], charge truncation tolerance scales
    /// `relative_tolerance * max(|q|, charge_tolerance)` (or the equivalent current tolerance
    /// over the step when larger) by `truncation_tolerance`. Ignored under
    /// [`ErrorControl::Unknown`].
    pub charge_tolerance: f64,
    /// Minimum shunt conductance placed across nonlinear junctions in siemens (default
    /// `1e-12`).
    pub gmin: f64,
    /// Maximum Newton iterations permitted per DC solve (default `100`).
    pub max_dc_iterations: usize,
    /// Maximum Newton iterations permitted per transient time step, matching SPICE `ITL4`
    /// (default `10`).
    pub max_transient_iterations: usize,
    /// Multiplier on local truncation error bounds matching SPICE `TRTOL` (default `7.0`).
    ///
    /// Under [`ErrorControl::Charge`], matches SPICE's omitted Taylor factorial so true error
    /// may reach twice this bound for backward Euler and six times for second-order methods.
    /// Under [`ErrorControl::Unknown`], scales the true one-step local truncation error bound
    /// directly without the factorial factor. Setting `f64::INFINITY` disables
    /// truncation-error rejection for fixed-step integration.
    pub truncation_tolerance: f64,
    /// Domain in which transient local truncation error is judged (default
    /// [`ErrorControl::Charge`]).
    pub error_control: ErrorControl,
    /// Maximum memory budget in gibibytes for transient waveforms or dense analysis
    /// matrices (default `2.0`).
    ///
    /// Analyses exceeding this budget abort cleanly instead of exhausting RAM: dense
    /// analyses fail upfront and transients fail if step shrinkage exceeds the limit.
    pub max_memory: f64,
    /// Device recording selection beyond circuit unknowns (empty by default).
    pub record: Record,
    /// Flag enabling operating-region and boundary-margin recording for transistors not
    /// already included in `record` (default `false`).
    pub regions: bool,
    /// Flag enabling per-activity timing instrumentation in transient statistics (default
    /// `false`).
    ///
    /// Clock reads add roughly 10% overhead on small transients while leaving all numeric
    /// results unchanged.
    pub profile: bool,
    /// Multi-threading policy for work inside a single run (defaults to all pool threads
    /// with measured break-even thresholds).
    ///
    /// Preserves bit-identical results except for minor rounding differences when nested
    /// dissection changes elimination order.
    pub parallelism: Parallelism,
    /// Initial DC node-voltage hints in volts by net path (such as `amplifier/mid`),
    /// matching SPICE `.nodeset` (empty by default).
    ///
    /// Hints hold target nodes during the first Newton solve and are then released to find
    /// the true operating point. Nets already constrained by voltage sources or inductors
    /// are ignored.
    pub nodesets: BTreeMap<String, f64>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            relative_tolerance: 1e-3,
            voltage_tolerance: 1e-6,
            current_tolerance: 1e-12,
            charge_tolerance: 1e-14,
            gmin: 1e-12,
            max_dc_iterations: 100,
            max_transient_iterations: 10,
            truncation_tolerance: 7.0,
            error_control: ErrorControl::default(),
            max_memory: 2.0,
            record: Record::default(),
            regions: false,
            profile: false,
            parallelism: Parallelism::default(),
            nodesets: BTreeMap::new(),
        }
    }
}

impl Options {
    /// Returns the absolute value tolerance for `nature`: `current_tolerance` for currents
    /// and `voltage_tolerance` for voltages, temperatures, and states.
    pub(crate) fn value_tolerance(&self, nature: Nature) -> f64 {
        if nature.is_current() {
            self.current_tolerance
        } else {
            self.voltage_tolerance
        }
    }

    /// Returns the absolute equation-residual tolerance for `nature`: `current_tolerance`
    /// for current-summing equations and `voltage_tolerance` for voltage or state
    /// equations.
    pub(crate) fn equation_tolerance(&self, nature: Nature) -> f64 {
        if nature.sums_currents() {
            self.current_tolerance
        } else {
            self.voltage_tolerance
        }
    }

    /// Returns the convergence tolerance comparing `first` and `second` for an unknown of
    /// `nature`: `relative_tolerance * max(|first|, |second|) + value_tolerance(nature)`.
    pub(crate) fn unknown_tolerance(&self, nature: Nature, first: f64, second: f64) -> f64 {
        self.relative_tolerance * first.abs().max(second.abs()) + self.value_tolerance(nature)
    }

    /// Checks that `analysis`, whose dense matrices have `rows` rows, fits `numbers` `f64`
    /// values within `max_memory`.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::TooMuchMemory`] when the memory estimate exceeds
    /// `max_memory`.
    pub(crate) fn hold(
        &self,
        analysis: &'static str,
        rows: usize,
        numbers: f64,
    ) -> Result<(), SimulationError> {
        let needed = numbers / NUMBERS_PER_GIB;
        if needed > self.max_memory {
            return Err(SimulationError::TooMuchMemory {
                analysis,
                rows,
                needed,
                max_memory: self.max_memory,
            });
        }
        Ok(())
    }

    /// Checks that `analysis`, holding `matrices` dense `rows × rows` matrices plus
    /// `extra_columns` vectors of length `rows`, fits within `max_memory`, returning the
    /// number of `f64` values held.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::TooMuchMemory`] when the memory estimate exceeds
    /// `max_memory`.
    pub(crate) fn hold_square(
        &self,
        analysis: &'static str,
        rows: usize,
        matrices: f64,
        extra_columns: usize,
    ) -> Result<f64, SimulationError> {
        let size = float_count(rows);
        let numbers = matrices * size * size + float_count(extra_columns) * size;
        self.hold(analysis, rows, numbers)?;
        Ok(numbers)
    }

    /// Checks that `held` transient result numbers fit within `max_memory`.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::TooManyPoints`] when `held` exceeds
    /// `max_memory * NUMBERS_PER_GIB`.
    pub(crate) fn hold_points(
        &self,
        held: f64,
        time: f64,
        step: f64,
    ) -> Result<(), SimulationError> {
        if held > self.max_memory * NUMBERS_PER_GIB {
            return Err(SimulationError::TooManyPoints {
                time,
                step,
                max_memory: self.max_memory,
            });
        }
        Ok(())
    }

    /// Validates that all tolerances, memory limits, iteration counts, and parallelism
    /// thresholds lie in range.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidOptions`] naming the first out-of-range setting.
    pub fn check(&self) -> Result<(), SimulationError> {
        let tolerances = [
            ("relative_tolerance", self.relative_tolerance),
            ("voltage_tolerance", self.voltage_tolerance),
            ("current_tolerance", self.current_tolerance),
            ("charge_tolerance", self.charge_tolerance),
            ("max_memory", self.max_memory),
        ];
        if let Some((name, value)) = tolerances
            .into_iter()
            .find(|&(_, value)| !(value > 0.0 && value.is_finite()))
        {
            return Err(SimulationError::InvalidOptions(format!(
                "{name} requires a positive finite value; got {value:?}"
            )));
        }
        if self.truncation_tolerance.is_nan() || self.truncation_tolerance <= 0.0 {
            return Err(SimulationError::InvalidOptions(format!(
                "truncation_tolerance requires a positive value; got {:?}",
                self.truncation_tolerance
            )));
        }
        if !(self.gmin >= 0.0 && self.gmin.is_finite()) {
            return Err(SimulationError::InvalidOptions(format!(
                "gmin requires a non-negative finite value; got {:?}",
                self.gmin
            )));
        }
        if self.max_dc_iterations == 0 || self.max_transient_iterations == 0 {
            return Err(SimulationError::InvalidOptions(
                "iteration limits require at least 1".to_owned(),
            ));
        }
        let parallelism = &self.parallelism;
        let counts = [
            ("max_threads", parallelism.max_threads),
            ("min_solves_per_task", parallelism.min_solves_per_task),
            ("min_instances_per_task", parallelism.min_instances_per_task),
            ("min_unknowns_per_task", parallelism.min_unknowns_per_task),
            (
                "max_unknowns_per_interior",
                parallelism.max_unknowns_per_interior,
            ),
        ];
        if let Some((name, _)) = counts.into_iter().find(|&(_, count)| count == Some(0)) {
            return Err(SimulationError::InvalidOptions(format!(
                "parallelism `{name}` requires at least 1; got 0"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{ErrorControl, Options};
    use crate::SimulationError;

    #[test]
    fn error_control_round_trips_through_json_and_defaults_to_charge() {
        assert_eq!(Options::default().error_control, ErrorControl::Charge);
        assert_eq!(
            serde_json::to_string(&ErrorControl::Charge).expect("serializes"),
            "\"charge\""
        );
        assert_eq!(
            serde_json::to_string(&ErrorControl::Unknown).expect("serializes"),
            "\"unknown\""
        );
        let parsed: Options =
            serde_json::from_str(r#"{"error_control": "unknown"}"#).expect("deserializes");
        assert_eq!(parsed.error_control, ErrorControl::Unknown);
        assert!(parsed.check().is_ok());
        assert!(serde_json::from_str::<Options>(r#"{"error_control": "flux"}"#).is_err());
    }

    #[test]
    fn check_rejects_non_finite_tolerances_and_gmin() {
        type OptionSetter = fn(&mut Options, f64);
        let finite_cases: [(&str, OptionSetter); 5] = [
            ("relative_tolerance", |options, value| {
                options.relative_tolerance = value;
            }),
            ("voltage_tolerance", |options, value| {
                options.voltage_tolerance = value;
            }),
            ("current_tolerance", |options, value| {
                options.current_tolerance = value;
            }),
            ("charge_tolerance", |options, value| {
                options.charge_tolerance = value;
            }),
            ("max_memory", |options, value| options.max_memory = value),
        ];
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
            for (name, setter) in finite_cases {
                let mut options = Options::default();
                setter(&mut options, bad);
                let result = options.check();
                assert!(
                    matches!(
                        &result,
                        Err(SimulationError::InvalidOptions(message)) if message.contains(name)
                    ),
                    "expected InvalidOptions for {name}={bad:?}, got {result:?}"
                );
            }
        }
        for bad in [f64::NAN, f64::NEG_INFINITY, 0.0, -1.0] {
            let options = Options {
                truncation_tolerance: bad,
                ..Options::default()
            };
            let result = options.check();
            assert!(
                matches!(
                    &result,
                    Err(SimulationError::InvalidOptions(message))
                        if message.contains("truncation_tolerance")
                ),
                "expected InvalidOptions for truncation_tolerance={bad:?}, got {result:?}"
            );
        }
        let fixed_step = Options {
            truncation_tolerance: f64::INFINITY,
            ..Options::default()
        };
        assert!(fixed_step.check().is_ok());
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1e-12] {
            let options = Options {
                gmin: bad,
                ..Options::default()
            };
            let result = options.check();
            assert!(
                matches!(
                    &result,
                    Err(SimulationError::InvalidOptions(message)) if message.contains("gmin")
                ),
                "expected InvalidOptions for gmin={bad:?}, got {result:?}"
            );
        }
        let zero_gmin = Options {
            gmin: 0.0,
            ..Options::default()
        };
        assert!(zero_gmin.check().is_ok());
    }
}
