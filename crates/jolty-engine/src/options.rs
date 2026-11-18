use std::collections::{BTreeMap, BTreeSet};

use jolty_models::{Nature, Tasks};
use jolty_numerics::{Dissecting, Ordering, reduce};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;

use crate::SimulationError;

/// The numbers of eight bytes a gibibyte holds.
///
/// Derivation: `2^30 / 8 = 1_073_741_824 / 8 = 134_217_728` IEEE 754 `f64` values per
/// gibibyte (IEC 80000-13 binary prefix).
pub(crate) const NUMBERS_PER_GIB: f64 = 134_217_728.0;

/// The break-evens of the thresholds of [`Parallelism`] that spread a kind of work in
/// tasks: on each number of [`BreakEvens::threads`], the least size of a task from which
/// spreading that work over every thread made a run faster than a serial one, on circuits
/// of the speed suite's fixtures' scripts at several sizes; none where no size measured
/// did. Measured by `scripts/bench --break-even` on an AMD Ryzen Threadripper 3970X of 32
/// cores and 64 threads, as the guide states each measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BreakEvens {
    /// The numbers of threads they were measured on, in increasing order.
    pub threads: [usize; 4],
    /// [`Parallelism::min_solves_per_task`]'s, on the amplifier's response at 1 to 1,111
    /// frequencies a decade.
    pub min_solves_per_task: [Option<usize>; 4],
    /// [`Parallelism::min_instances_per_task`]'s, on chains of 1 to 1,000 inverters.
    pub min_instances_per_task: [Option<usize>; 4],
    /// [`Parallelism::min_unknowns_per_task`]'s, on chains of 100 to 2,000 inverters and
    /// meshes of 10 to 300 nodes a side.
    pub min_unknowns_per_task: [Option<usize>; 4],
    /// [`Parallelism::max_unknowns_per_interior`]'s, the most unknowns of an interior from
    /// which dissecting a block into a thread's interior or more made a run faster, on
    /// meshes of 10 to 300 nodes a side.
    pub max_unknowns_per_interior: [Option<usize>; 4],
}

impl BreakEvens {
    /// The break-even, of those `measured` on each of [`BreakEvens::threads`], that a run
    /// on `threads` threads takes: the one measured on the most threads no more than
    /// `threads`; none on one thread, which spreads nothing.
    #[must_use]
    pub fn on(&self, measured: [Option<usize>; 4], threads: usize) -> Option<usize> {
        self.threads
            .iter()
            .zip(measured)
            .rev()
            .find(|&(&each, _)| each <= threads)
            .and_then(|(_, least)| least)
    }

    /// The least of those `measured`, which no run's break-even is below.
    fn least(measured: [Option<usize>; 4]) -> Option<usize> {
        measured.into_iter().flatten().min()
    }
}

/// The break-evens each threshold of [`Parallelism`] left unset takes, on 2, 4, 8 and 64
/// threads.
///
/// Source: measured by `scripts/bench --break-even` on `cedar` (AMD Ryzen Threadripper 3970X,
/// 32 cores / 64 threads, 16 November 2026); see `docs/guide/simulation.md` and
/// `benchmarks/speed/README.md` for each fixture and load average.
pub const BREAK_EVENS: BreakEvens = BreakEvens {
    threads: [2, 4, 8, 64],
    min_solves_per_task: [Some(4_999), Some(11), Some(11), Some(4)],
    min_instances_per_task: [None, Some(250), Some(125), None],
    min_unknowns_per_task: [None, None, None, None],
    max_unknowns_per_interior: [Some(5_001), Some(2_501), Some(1_251), Some(157)],
};

/// What an analysis records of the circuit's devices beside the unknowns: the current into
/// each terminal and the charge it holds, and each device's derivatives and variables.
///
/// In JSON, every instance is the string `"all"`, and some instances an object with one
/// key, such as `{"instances": ["M1", "X1"]}`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Record {
    /// Every device, and the terminals of every component instance and logic device.
    All,
    /// The devices at these paths, such as `X1/M1`, and the terminals of the component
    /// instances and logic devices at them, such as `X1`'s ports; a path that names none
    /// records nothing.
    Instances(BTreeSet<String>),
}

impl Default for Record {
    /// Nothing beyond the unknowns.
    fn default() -> Self {
        Self::Instances(BTreeSet::new())
    }
}

impl Record {
    /// Whether the record names the device or the component instance at `path`.
    #[must_use]
    pub fn names(&self, path: &str) -> bool {
        match self {
            Self::All => true,
            Self::Instances(paths) => paths.contains(path),
        }
    }
}

/// How a run spreads its own work across the threads of the rayon pool it runs in: the
/// frequencies of its small-signal analyses, the right-hand sides of its sensitivities, its
/// devices' instances, its Newton iteration's and step control's work on each unknown, and
/// the factorization of its matrix's large blocks. Every result is the same bits however
/// the work is split.
/// No setting but [`Parallelism::max_unknowns_per_interior`] changes any result: that one
/// orders the factorization's operations otherwise, so results agree with those of another
/// to within its rounding rather than bit for bit, as they do on another number of threads
/// where its default dissects a block.
///
/// The caller owns the pool and runs the engine inside it, with `ThreadPool::install`; a
/// run that spreads its work uses at most its `max_threads`, on a pool of exactly that
/// many threads of its own when they are fewer than its caller's pool has, and a
/// [batch](crate::batch::Batch) runs each of its runs on one thread whenever it has a run
/// for every thread of its pool. Spreading work costs the time of waking threads, which a
/// small run's work does not repay, so each kind of work spreads only in tasks of at least
/// its threshold, its least size per task, once it fills such a task for every thread the
/// run may use, as the break-evens were measured: one task, on the run's own thread, for
/// less. Each threshold defaults to its break-even on the threads
/// the run may use, [`BREAK_EVENS`], the least size of a task from which spreading made a
/// run faster, measured on an AMD Ryzen Threadripper 3970X of 32 cores and 64 threads;
/// the guide states each measurement.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Parallelism {
    /// The most threads a run spreads its own work over; none for every thread of the pool
    /// it runs in, and 1 to run it on one thread alone. A run that may spread its work over
    /// two or more, fewer than the pool it is called in has, runs on a pool of exactly that
    /// many threads of its own, which every task it spreads runs on.
    pub max_threads: Option<usize>,
    /// The fewest frequencies, right-hand sides or parameters' derivatives a task takes
    /// when a run spreads them, at most one task a thread; none, by default, for its
    /// break-even on the threads the run may use, [`BREAK_EVENS`].
    pub min_solves_per_task: Option<usize>,
    /// The fewest device instances of a group a task evaluates when a run spreads its
    /// devices' evaluation, at most one task of a group a thread, a linear group's
    /// instances, evaluated once a run, for their entries, taking their residuals and
    /// charges from them in one task; none, by default, for its break-even on the threads
    /// the run may use, [`BREAK_EVENS`].
    pub min_instances_per_task: Option<usize>,
    /// The fewest unknowns, or stored entries of the matrix, a task takes when a run
    /// spreads its Newton iteration's and its transient step control's work on each, in
    /// whole chunks of [`jolty_numerics::reduce::CHUNK`], which the threads take as they
    /// free; none, by default, for its break-even on the threads the run may use,
    /// [`BREAK_EVENS`].
    pub min_unknowns_per_task: Option<usize>,
    /// The most unknowns of an interior when a run's ordering dissects a block of its
    /// matrix's block triangular form: each block of more is ordered by nested dissection
    /// into interiors of at most as many, unless one cannot split, and the separators
    /// between them, whose factorizations and solves spread over the threads a run may
    /// use, wherever that costs the block's elimination fewer multiply-adds than its
    /// minimum degree ordering, as [`Dissecting::Cheaper`] counts them, and is otherwise
    /// ordered by the minimum degree, on the run's thread. It changes the order of the
    /// factorization's operations, so the last digits of results, which it gives the same
    /// bits on any number of threads; none, by default, for its break-even on the threads
    /// the run may use, [`BREAK_EVENS`], and the minimum degree's ordering on one, so that
    /// results agree across numbers of threads to within the factorization's rounding.
    pub max_unknowns_per_interior: Option<usize>,
}

impl Default for Parallelism {
    /// Every thread of the pool, and each threshold at its break-even.
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
    /// How many tasks a run splits `items` independent solves into: one for each thread it
    /// may use, each of at least [`Parallelism::min_solves_per_task`] of them, when they
    /// fill a task for every one, and else one.
    #[must_use]
    pub fn tasks(&self, items: usize) -> usize {
        self.split(Kind::Solves, items)
            .map_or(1, |tasks| tasks.count(items))
    }

    /// The least items a task of `kind` takes on `threads` threads: its threshold, if set,
    /// and else its break-even there; none, never to spread it.
    fn least(&self, kind: Kind, threads: usize) -> Option<usize> {
        let (set, measured) = self.threshold(kind);
        set.or_else(|| BREAK_EVENS.on(measured, threads))
    }

    /// The threshold of `kind`, if set, and its break-evens.
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

    /// Tasks of at least `least` items, one for each of `threads`, two or more, when
    /// `items` fill a task for every one; none else.
    fn filled(least: usize, items: usize, threads: usize) -> Option<Tasks> {
        let tasks = tasks_of(least, threads);
        (threads >= 2 && items / tasks.least >= threads).then_some(tasks)
    }

    /// Tasks of at least the least items of `kind`, one for each thread a run may use,
    /// when `items` fill a task for every one of them; none else. Items that do not fill
    /// two however many threads there are never ask for the pool, which would start
    /// rayon's global one when a run is called in none.
    fn split(&self, kind: Kind, items: usize) -> Option<Tasks> {
        let (set, measured) = self.threshold(kind);
        let fewest = set.or_else(|| BreakEvens::least(measured))?;
        Self::filled(fewest, items, 2)?;
        let threads = self.usable();
        Self::filled(self.least(kind, threads)?, items, threads)
    }

    /// Whether some kind of work of a run of `sizes` fills a task of its threshold for
    /// every one of `threads` threads, those of the pool it would run in, or as many as
    /// its [`Parallelism::max_threads`] if fewer, so that the run spreads it: its instances
    /// or its unknowns, its ordering dissects a block, or its solves fill them.
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

    /// The threads a run may spread its work over: its [`Parallelism::max_threads`], but no
    /// more than the rayon pool it runs in has.
    fn usable(&self) -> usize {
        let threads = rayon::current_num_threads();
        self.max_threads.map_or(threads, |most| most.min(threads))
    }

    /// Runs `run`, a run's work, on at most its [`Parallelism::max_threads`]. When the run
    /// may spread its work, and its threads are two or more but fewer than the pool it is
    /// called in has, it runs on a pool of exactly that many threads, made for it: rayon
    /// runs every task that a task of a pool spawns on that pool, so every task the run
    /// spreads runs on one of them. Else it runs where it is called, and a run that spreads
    /// nothing on the calling thread alone.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Threads`] when the pool's threads cannot be started, and
    /// else the errors of `run`.
    pub(crate) fn confined<T: Send>(
        &self,
        run: impl FnOnce() -> Result<T, SimulationError> + Send,
    ) -> Result<T, SimulationError> {
        // A run that spreads nothing never asks for the pool it is called in, which would
        // start rayon's global one when it is called in none.
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

    /// Whether a run may spread any of its work across threads, and so must run inside
    /// its caller's pool: unless its [`Parallelism::max_threads`] is one. A run that may
    /// not touches no thread but its own.
    #[must_use]
    pub const fn spreads(&self) -> bool {
        !matches!(self.max_threads, Some(..2))
    }

    /// How a run whose matrix's largest block holds `largest_block` unknowns orders it:
    /// by [`Dissecting::Cheaper`] with [`Parallelism::max_unknowns_per_interior`], or its
    /// break-even on the threads the run may use; none, for the minimum degree's ordering,
    /// where it has no break-even or the block holds no more. A block no break-even could
    /// dissect never asks for the pool, which would start rayon's global one when a run
    /// is called in none.
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

    /// How a run on `threads` threads orders its matrix's blocks, as
    /// [`Parallelism::dissecting`] says.
    #[must_use]
    pub(crate) fn dissecting_on(&self, threads: usize) -> Option<Dissecting> {
        self.max_unknowns_per_interior
            .or_else(|| BREAK_EVENS.on(BREAK_EVENS.max_unknowns_per_interior, threads))
            .map(Dissecting::Cheaper)
    }

    /// The threads a run's factorization of `ordering` spreads over: all it may use, for
    /// one that dissects a block, and one for a run that spreads nothing.
    #[must_use]
    pub(crate) fn factorization(&self, ordering: &Ordering) -> usize {
        if self.spreads() && ordering.is_dissected() {
            self.usable()
        } else {
            1
        }
    }

    /// How a run's devices split their evaluation among threads: every group at once, each
    /// that is not linear in tasks of at least [`Parallelism::min_instances_per_task`] of
    /// its instances, as [`Tasks::count`] says, then the values of the circuit's equations gathered in parts
    /// where their contributions balance, as many as the threads ask for as they take
    /// them; none, to evaluate them on the run's thread, when the `instances` it evaluates,
    /// those of its groups that are not linear, fill no task for every thread it may use,
    /// or it may use one alone. The bits found are the same either way.
    #[must_use]
    pub fn evaluation(&self, instances: usize) -> Option<Tasks> {
        self.split(Kind::Instances, instances)
    }

    /// Whether a run spreads its work on each of the circuit's `unknowns` unknowns and on
    /// each stored entry of its matrix among threads, in Newton iteration and in a
    /// transient's step control, as [`Inside`] does: once its unknowns fill a task of at
    /// least [`Parallelism::min_unknowns_per_task`] for every thread the run may use, two
    /// or more; none, to do it all on the run's thread. The bits found are the same either
    /// way.
    #[must_use]
    pub(crate) fn inside(&self, unknowns: usize) -> Option<Inside> {
        self.split(Kind::Unknowns, unknowns).map(|tasks| Inside {
            chunks: tasks.least.get().div_ceil(reduce::CHUNK),
        })
    }

    /// `each` of every one of `items`, in their order, spread over [`Parallelism::tasks`]
    /// tasks of consecutive items, each task with scratch space of its own that `scratch`
    /// makes; on the calling thread alone for one task. Each item's result is the same
    /// whichever task finds it.
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

/// A kind of work a run spreads in tasks, each with a threshold of [`Parallelism`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Frequencies, right-hand sides or parameters' derivatives.
    Solves,
    /// Device instances.
    Instances,
    /// Unknowns and stored entries.
    Unknowns,
}

/// Tasks of at least `least` items, one a thread of `threads`, one at least of each.
fn tasks_of(least: usize, threads: usize) -> Tasks {
    Tasks {
        least: NonZeroUsize::new(least).unwrap_or(NonZeroUsize::MIN),
        threads: NonZeroUsize::new(threads).unwrap_or(NonZeroUsize::MIN),
    }
}

/// What a run's work counts, which [`Parallelism::fills_threads`] reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sizes {
    /// The device instances it evaluates, those of the groups that are not linear.
    pub instances: usize,
    /// The circuit's unknowns.
    pub unknowns: usize,
    /// Whether its ordering dissects a block of its matrix's block triangular form.
    pub dissected: bool,
    /// The independent solves of its analysis: its frequencies, or its right-hand sides.
    pub solves: usize,
}

/// That a run spreads its work on each unknown and each stored entry over the threads of
/// its pool: in chunks of [`jolty_numerics::reduce::CHUNK`] of them, whatever the number of
/// threads, which rayon splits among the threads as they take them, so that a fast thread
/// takes more chunks than a slow one, each task at least `chunks` of them; a sum adds the
/// chunks' totals in their order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Inside {
    /// The fewest chunks a task takes.
    pub(crate) chunks: usize,
}

/// The numerical settings of a simulation: tolerances and iteration limits, and what its
/// results record.
///
/// The defaults are SPICE's, so results compare with SPICE simulators run at their defaults.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    /// The tolerance on every unknown and residual, relative to its magnitude; default 1e-3.
    pub relative_tolerance: f64,
    /// The absolute tolerance on voltages, in volts; default 1e-6.
    pub voltage_tolerance: f64,
    /// The absolute tolerance on currents, in amperes; default 1e-12.
    pub current_tolerance: f64,
    /// The least charge a time step's tolerance on a charge is relative to, in coulombs, as
    /// SPICE's `CHGTOL`: the tolerance on a charge `q` is `relative_tolerance` times the
    /// larger of `|q|` and this, or the tolerance on the current it carries over the step
    /// where that is larger, either times the truncation tolerance; default 1e-14.
    pub charge_tolerance: f64,
    /// The conductance placed across every nonlinear junction, in siemens; default 1e-12.
    pub gmin: f64,
    /// The Newton iterations allowed for one DC solution; default 100.
    pub max_dc_iterations: usize,
    /// The most Newton iterations one time step takes, SPICE's `ITL4`, the iterate after
    /// the last judged as any other; default 10.
    pub max_transient_iterations: usize,
    /// How many times the tolerances a step's truncation error may reach, as SPICE estimates
    /// the error, before the step is rejected; default 7, SPICE's `TRTOL`. SPICE's estimate
    /// leaves out a factorial, so the error itself may reach twice as many after a step of
    /// backward Euler, and six times as many after one of the second-order formulas.
    pub truncation_tolerance: f64,
    /// The gibibytes of numbers an analysis may hold at once: a transient's results, every
    /// unknown's and every device's at each point, and the dense matrices of a pole-zero
    /// analysis or a periodic steady state, of a row and a column per unknown. An analysis
    /// that would hold more fails instead of filling the memory: a transient whose steps
    /// shrink to a crawl, once it passes the limit, and a dense analysis before it begins;
    /// default 2.
    pub max_memory: f64,
    /// What the results record of the devices beside the unknowns; by default nothing.
    pub record: Record,
    /// Whether the results record, of every transistor `record` leaves out, its region of
    /// operation and its margins to the regions' boundaries alone; by default not.
    pub regions: bool,
    /// Whether a transient measures where its time goes, into its statistics' profile; by
    /// default not, as the clock adds about a tenth to a small circuit's transient. It
    /// changes no other result.
    pub profile: bool,
    /// How a run spreads its own work across threads, which changes no result but for
    /// the last digits a dissected factorization's order of operations gives; by default
    /// over every thread of its pool, each kind of work from its break-even on.
    pub parallelism: Parallelism,
    /// The voltage, in volts, each net named, such as `amplifier/mid`, is held at while
    /// a DC solution is first solved from zero, which is then released and solved again
    /// from there, as SPICE's `.nodeset` starts an operating point: a guess that leads
    /// Newton iteration to one solution of several. A net that voltage sources and
    /// inductors join to the reference, or to a net named before it, is not held, as
    /// they fix its voltage. None by default.
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
    /// The absolute tolerance on the value of an unknown of `nature`: the current
    /// tolerance for a current, the voltage tolerance for a voltage, a temperature or a
    /// device's state.
    pub(crate) fn value_tolerance(&self, nature: Nature) -> f64 {
        if nature.is_current() {
            self.current_tolerance
        } else {
            self.voltage_tolerance
        }
    }

    /// The absolute tolerance on the residual of an unknown of `nature`'s equation: the
    /// current tolerance for an equation that sums currents, the voltage tolerance for one
    /// that constrains a voltage or moves a device's state.
    pub(crate) fn equation_tolerance(&self, nature: Nature) -> f64 {
        if nature.sums_currents() {
            self.current_tolerance
        } else {
            self.voltage_tolerance
        }
    }

    /// Checks that `analysis`, whose dense matrices have `rows` rows, may hold `numbers`
    /// numbers at once.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::TooMuchMemory`] when they would pass its `max_memory`.
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

    /// Checks that every tolerance is positive and every limit allows an iteration.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidOptions`] naming the first setting out of range.
    pub fn check(&self) -> Result<(), SimulationError> {
        let tolerances = [
            ("relative_tolerance", self.relative_tolerance),
            ("voltage_tolerance", self.voltage_tolerance),
            ("current_tolerance", self.current_tolerance),
            ("charge_tolerance", self.charge_tolerance),
            ("truncation_tolerance", self.truncation_tolerance),
            ("max_memory", self.max_memory),
        ];
        if let Some((name, value)) = tolerances.into_iter().find(|&(_, value)| value <= 0.0) {
            return Err(SimulationError::InvalidOptions(format!(
                "{name} must be positive, not {value:?}"
            )));
        }
        if self.gmin < 0.0 {
            return Err(SimulationError::InvalidOptions(format!(
                "gmin must be at least 0, not {:?}",
                self.gmin
            )));
        }
        if self.max_dc_iterations == 0 || self.max_transient_iterations == 0 {
            return Err(SimulationError::InvalidOptions(
                "iteration limits must be at least 1".to_owned(),
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
                "parallelism's {name} must be at least 1, not 0"
            )));
        }
        Ok(())
    }
}
