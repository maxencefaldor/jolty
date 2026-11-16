use std::collections::BTreeSet;

use jolty_models::Nature;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;

use crate::SimulationError;

/// The numbers of eight bytes a gibibyte holds.
pub(crate) const NUMBERS_PER_GIB: f64 = 134_217_728.0;

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
/// orders the factorization's operations otherwise, so results agree with those of the
/// factorization on one thread to within its rounding rather than bit for bit.
///
/// The caller owns the pool and runs the engine inside it, with `ThreadPool::install`; a
/// run that spreads its work uses at most its `threads`, on a pool of exactly that many
/// threads of its own when they are fewer than its caller's pool has, and a
/// [batch](crate::batch::Batch) runs each of its runs on one thread whenever it has a run
/// for every thread of its pool. Spreading work costs the time of waking threads, which a
/// small run's work does not repay, so each kind of work spreads only once its threshold
/// is set: none is by default, and a run does all its work on its own thread.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Parallelism {
    /// The most threads a run spreads its own work over; none for every thread of the pool
    /// it runs in, and 1 to run it on one thread alone. A run that may spread its work over
    /// two or more, fewer than the pool it is called in has, runs on a pool of exactly that
    /// many threads of its own, which every task it spreads runs on.
    pub threads: Option<usize>,
    /// The fewest frequencies, right-hand sides or parameters' derivatives a task takes
    /// when a run spreads them; none, by default, to take them all on the run's thread.
    pub min_solves_per_task: Option<usize>,
    /// The device instances a task evaluates when a run evaluates its devices in
    /// parallel; none, by default, to evaluate them on the run's thread.
    pub min_instances_per_task: Option<usize>,
    /// The fewest unknowns a circuit has for its Newton iteration's and its transient step
    /// control's work on each unknown and each stored entry of its matrix to spread over the
    /// threads a run may use; none, by default, to keep that work on the run's thread.
    pub min_unknowns_inside: Option<usize>,
    /// The most unknowns of an interior when a run factors its matrix in parallel: each
    /// block of its block triangular form with more is ordered by nested dissection into
    /// interiors of at most as many, unless one cannot split, and the separators between
    /// them, whose factorizations and solves spread over the threads a run may use; none,
    /// by default, to factor every block on the run's thread, ordered by the minimum
    /// degree. It changes the order of the factorization's operations, so the last digits
    /// of results, which it gives the same bits on any number of threads.
    pub max_unknowns_per_interior: Option<usize>,
}

impl Parallelism {
    /// How many tasks a run splits `items` independent solves into: one per thread it may
    /// use, each taking at least [`Parallelism::min_solves_per_task`] of them, and one when
    /// that is unset or there are too few.
    #[must_use]
    pub fn tasks(&self, items: usize) -> usize {
        let Some(least) = self.min_solves_per_task else {
            return 1;
        };
        self.usable().min(items / least.max(1)).max(1)
    }

    /// The threads a run may spread its work over: its [`Parallelism::threads`], but no
    /// more than the rayon pool it runs in has.
    fn usable(&self) -> usize {
        let threads = rayon::current_num_threads();
        self.threads.map_or(threads, |most| most.min(threads))
    }

    /// Runs `run`, a run's work, on at most its [`Parallelism::threads`]. When the run may
    /// spread its work, and its threads are two or more but fewer than the pool it is
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
        let Some(most) = self.threads.filter(|&most| self.spreads() && most >= 2) else {
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
    /// its caller's pool: once a threshold of [`Parallelism`] lets one kind of work spread.
    /// A run that may not touches no thread but its own.
    #[must_use]
    pub const fn spreads(&self) -> bool {
        self.min_solves_per_task.is_some()
            || self.min_instances_per_task.is_some()
            || self.min_unknowns_inside.is_some()
            || self.max_unknowns_per_interior.is_some()
    }

    /// How a run factors its matrix: the most unknowns of an interior its ordering
    /// dissects its large blocks into, [`Parallelism::max_unknowns_per_interior`], and the
    /// threads it may use for them, all it may use; none and one when that is unset.
    #[must_use]
    pub(crate) fn factorization(&self) -> (Option<usize>, usize) {
        match self.max_unknowns_per_interior {
            Some(interiors) => (Some(interiors), self.usable()),
            None => (None, 1),
        }
    }

    /// How a run's devices split their evaluation among threads: every group at once, each
    /// in tasks of [`Parallelism::min_instances_per_task`] of its instances, then the
    /// values of the circuit's equations gathered in parts where their contributions
    /// balance, as many as the threads ask for as they take them; none, to evaluate them on
    /// the run's thread, when that is unset, the circuit's `instances` fill no more than one
    /// task, or the run may use one thread alone. The bits found are the same either way.
    #[must_use]
    pub fn evaluation(&self, instances: usize) -> Option<jolty_models::Tasks> {
        let per_task = NonZeroUsize::new(self.min_instances_per_task?)?;
        if self.usable() < 2 || instances <= per_task.get() {
            return None;
        }
        Some(jolty_models::Tasks {
            instances: per_task,
        })
    }

    /// Whether a run spreads its work on each of the circuit's `unknowns` unknowns and on
    /// each stored entry of its matrix among threads, in Newton iteration and in a
    /// transient's step control, as [`Inside`] does: once the circuit has at least
    /// [`Parallelism::min_unknowns_inside`] unknowns and the run may use two threads or
    /// more; none, to do it all on the run's thread. The bits found are the same either
    /// way.
    #[must_use]
    pub(crate) fn inside(&self, unknowns: usize) -> Option<Inside> {
        let least = self.min_unknowns_inside?;
        (self.usable() >= 2 && unknowns >= least.max(1)).then_some(Inside)
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

/// That a run spreads its work on each unknown and each stored entry over the threads of
/// its pool: in chunks of [`jolty_numerics::reduce::CHUNK`] of them, whatever the number of
/// threads, which rayon splits among the threads as they take them, so that a fast thread
/// takes more chunks than a slow one; a sum adds the chunks' totals in their order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Inside;

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
    /// The Newton iterations one time step takes before it goes on only while each step is
    /// shorter than the one before, and is shortened once one is not; default 10.
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
    /// How a run spreads its own work across threads, which changes no result; by default
    /// it does all its work on its own thread.
    pub parallelism: Parallelism,
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
            ("threads", parallelism.threads),
            ("min_solves_per_task", parallelism.min_solves_per_task),
            ("min_instances_per_task", parallelism.min_instances_per_task),
            ("min_unknowns_inside", parallelism.min_unknowns_inside),
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
