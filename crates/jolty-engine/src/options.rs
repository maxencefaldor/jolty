use std::collections::BTreeSet;

use jolty_models::Quantity;
use rayon::prelude::*;
use serde::Deserialize;

use crate::SimulationError;

/// The numbers of eight bytes a gibibyte holds.
pub(crate) const NUMBERS_PER_GIB: f64 = 134_217_728.0;

/// What an analysis records of the circuit's devices beside the unknowns: the current into
/// each terminal and the charge it holds, and each device's derivatives and variables.
///
/// In JSON, every instance is the string `"all"`, and some instances an object with one
/// key, such as `{"instances": ["M1", "X1"]}`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
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
/// frequencies of its small-signal analyses, the right-hand sides of its sensitivities, and,
/// once its devices evaluate in parallel, their instances and its unknowns. No setting
/// changes any result: every one of them is the same bits however the work is split.
///
/// The caller owns the pool and runs the engine inside it, with `ThreadPool::install`; a
/// run uses at most its `threads`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Parallelism {
    /// The most threads a run spreads its own work over; none for every thread of the pool
    /// it runs in, and 1 to run it on one thread alone.
    pub threads: Option<usize>,
    /// The fewest device instances a task evaluates, when devices evaluate in parallel;
    /// none until they do.
    pub min_instances_per_task: Option<usize>,
    /// The fewest unknowns a circuit has for its Newton iteration's work on each unknown to
    /// run in parallel; none until it does.
    pub min_unknowns_inside: Option<usize>,
}

impl Default for Parallelism {
    /// Every thread of the pool, and nothing that does not yet run in parallel.
    fn default() -> Self {
        Self {
            threads: None,
            min_instances_per_task: None,
            min_unknowns_inside: None,
        }
    }
}

impl Parallelism {
    /// How many tasks a run splits work of `items` independent items into: one per thread
    /// it may use, no more than there are items, and at least one.
    #[must_use]
    pub fn tasks(&self, items: usize) -> usize {
        let threads = rayon::current_num_threads();
        self.threads
            .map_or(threads, |most| most.min(threads))
            .min(items)
            .max(1)
    }

    /// `each` of every one of `items`, in their order, spread over [`Parallelism::tasks`]
    /// tasks of consecutive items, each task with scratch space of its own that `scratch`
    /// makes. Each item's result is the same whichever task finds it.
    pub(crate) fn spread<T: Send, S, R: Send>(
        &self,
        items: &mut [T],
        scratch: impl Fn() -> S + Sync,
        each: impl Fn(&mut S, &mut T) -> R + Sync,
    ) -> Vec<R> {
        let size = items.len().div_ceil(self.tasks(items.len())).max(1);
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

/// The numerical settings of a simulation: tolerances and iteration limits, and what its
/// results record.
///
/// The defaults are SPICE's, so results compare with SPICE simulators run at their defaults.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    /// The tolerance on every unknown and residual, relative to its magnitude; default 1e-3.
    pub relative_tolerance: f64,
    /// The absolute tolerance on voltages, in volts; default 1e-6.
    pub voltage_tolerance: f64,
    /// The absolute tolerance on currents, in amperes; default 1e-12.
    pub current_tolerance: f64,
    /// The absolute tolerance on charges, in coulombs; default 1e-14.
    pub charge_tolerance: f64,
    /// The conductance placed across every nonlinear junction, in siemens; default 1e-12.
    pub gmin: f64,
    /// The Newton iterations allowed for one DC solution; default 100.
    pub dc_iterations: usize,
    /// The Newton iterations allowed for one time step before the step is shortened; default 10.
    pub transient_iterations: usize,
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
    pub memory: f64,
    /// What the results record of the devices beside the unknowns; by default nothing.
    pub record: Record,
    /// Whether a transient measures where its time goes, into its statistics' profile; by
    /// default not, as the clock adds about a tenth to a small circuit's transient. It
    /// changes no other result.
    pub profile: bool,
    /// How a run spreads its own work across threads, which changes no result; by default
    /// over every thread of the pool it runs in.
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
            dc_iterations: 100,
            transient_iterations: 10,
            truncation_tolerance: 7.0,
            memory: 2.0,
            record: Record::default(),
            profile: false,
            parallelism: Parallelism::default(),
        }
    }
}

impl Options {
    /// The absolute tolerance on the value of an unknown of `quantity`: the current
    /// tolerance for a current, the voltage tolerance for a voltage, a temperature or a
    /// device's state.
    pub(crate) fn value_tolerance(&self, quantity: Quantity) -> f64 {
        if quantity.is_current() {
            self.current_tolerance
        } else {
            self.voltage_tolerance
        }
    }

    /// The absolute tolerance on the residual of an unknown of `quantity`'s equation: the
    /// current tolerance for an equation that sums currents, the voltage tolerance for one
    /// that constrains a voltage or moves a device's state.
    pub(crate) fn equation_tolerance(&self, quantity: Quantity) -> f64 {
        if quantity.sums_currents() {
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
    /// Returns [`SimulationError::TooMuchMemory`] when they would pass its `memory`.
    pub(crate) fn hold(
        &self,
        analysis: &'static str,
        rows: usize,
        numbers: f64,
    ) -> Result<(), SimulationError> {
        let needed = numbers / NUMBERS_PER_GIB;
        if needed > self.memory {
            return Err(SimulationError::TooMuchMemory {
                analysis,
                rows,
                needed,
                memory: self.memory,
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
            ("memory", self.memory),
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
        if self.dc_iterations == 0 || self.transient_iterations == 0 {
            return Err(SimulationError::InvalidOptions(
                "iteration limits must be at least 1".to_owned(),
            ));
        }
        let parallelism = &self.parallelism;
        let counts = [
            ("threads", parallelism.threads),
            ("min_instances_per_task", parallelism.min_instances_per_task),
            ("min_unknowns_inside", parallelism.min_unknowns_inside),
        ];
        if let Some((name, _)) = counts.into_iter().find(|&(_, count)| count == Some(0)) {
            return Err(SimulationError::InvalidOptions(format!(
                "parallelism's {name} must be at least 1, not 0"
            )));
        }
        Ok(())
    }
}
