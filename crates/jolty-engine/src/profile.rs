//! Solver work counters (always enabled) and optional per-activity wall-clock profiling.
//!
//! Each simulator tracks cumulative Newton iterations and sparse factorization statistics
//! from creation, taking differences across each analysis. When `options.profile` is
//! enabled, an activity clock attributes non-overlapping durations so all buckets sum to
//! total runtime. Clock sampling adds roughly 0.23 µs per iteration on Apple M2 Max (about
//! 10% on small transients), so profiling stays disabled by default and never alters
//! numeric results.

use std::time::{Duration, Instant};

use crate::Simulator;

/// The work an analysis did, counted whatever its options.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Work {
    /// Total Newton iterations run across the operating point (unless cached) and all
    /// attempted steps, both accepted and rejected.
    pub iterations: usize,
    /// The factorizations of its sparse solver that searched for every pivot: the first the
    /// simulator made, and any after a matrix the solver could not factor.
    pub factorizations: usize,
    /// The factorizations that kept every pivot of the last, with no search.
    pub refactorizations: usize,
    /// The factorizations that kept the pivots of the last up to one their matrix made too
    /// small, and searched for the pivots from that one on.
    pub fallbacks: usize,
    /// The factorizations given the very matrix factored last, which kept its factors.
    pub reuses: usize,
}

/// Per-activity wall-clock time breakdown collected when `options.profile` is enabled.
///
/// Activities are mutually exclusive so all fields sum to [`Profile::total`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Profile {
    /// Option validation, recording setup, and source breakpoint collection.
    pub preparing: Duration,
    /// Device kernel evaluation.
    pub evaluating: Duration,
    /// Matrix and right-hand-side assembly plus residual tolerance checks.
    pub assembling: Duration,
    /// Sparse matrix factorization.
    pub factoring: Duration,
    /// Linear triangular solves with existing factors.
    pub solving: Duration,
    /// Transient error estimation, step selection, and event/discontinuity localization.
    pub step_control: Duration,
    /// Recording device currents, charges, derivatives, and variables at accepted points.
    pub recording: Duration,
    /// Remaining solver work: Newton updates, integration terms, predictions, and
    /// sensitivity propagation.
    pub other: Duration,
}

impl Profile {
    /// Returns the total profiled wall-clock duration across all activities.
    #[must_use]
    pub fn total(&self) -> Duration {
        self.preparing
            + self.evaluating
            + self.assembling
            + self.factoring
            + self.solving
            + self.step_control
            + self.recording
            + self.other
    }

    /// Returns a mutable reference to the duration bucket for `activity`.
    fn spent(&mut self, activity: Activity) -> &mut Duration {
        match activity {
            Activity::Preparing => &mut self.preparing,
            Activity::Evaluating => &mut self.evaluating,
            Activity::Assembling => &mut self.assembling,
            Activity::Factoring => &mut self.factoring,
            Activity::Solving => &mut self.solving,
            Activity::StepControl => &mut self.step_control,
            Activity::Recording => &mut self.recording,
            Activity::Other => &mut self.other,
        }
    }

    /// Computes the per-activity elapsed durations since `earlier`.
    fn since(&self, earlier: &Self) -> Self {
        Self {
            preparing: self.preparing.saturating_sub(earlier.preparing),
            evaluating: self.evaluating.saturating_sub(earlier.evaluating),
            assembling: self.assembling.saturating_sub(earlier.assembling),
            factoring: self.factoring.saturating_sub(earlier.factoring),
            solving: self.solving.saturating_sub(earlier.solving),
            step_control: self.step_control.saturating_sub(earlier.step_control),
            recording: self.recording.saturating_sub(earlier.recording),
            other: self.other.saturating_sub(earlier.other),
        }
    }
}

/// Active simulator phase tracked by the profiling clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Activity {
    Preparing,
    Evaluating,
    Assembling,
    Factoring,
    Solving,
    StepControl,
    Recording,
    Other,
}

/// Wall-clock accumulator tracking the current [`Activity`], start timestamp, and
/// cumulative [`Profile`].
#[derive(Debug)]
struct Clock {
    activity: Activity,
    since: Instant,
    spent: Profile,
}

impl Clock {
    /// Charges elapsed wall-clock time since the last transition to the active phase.
    fn charge(&mut self) {
        let now = Instant::now();
        *self.spent.spent(self.activity) += now.saturating_duration_since(self.since);
        self.since = now;
    }
}

/// Cumulative Newton iteration counter and optional wall-clock [`Clock`] for a simulator.
#[derive(Debug, Default)]
pub(crate) struct Meter {
    iterations: usize,
    clock: Option<Clock>,
}

/// Snapshot of meter and sparse-solver counters captured at analysis start.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Reading {
    iterations: usize,
    solver: jolty_numerics::Work,
    spent: Option<Profile>,
}

impl Meter {
    /// Increments the cumulative Newton iteration counter.
    pub(crate) fn iterate(&mut self) {
        self.iterations += 1;
    }

    /// Returns the total Newton iterations counted since creation.
    pub(crate) fn iterations(&self) -> usize {
        self.iterations
    }

    /// Switches profiling to `activity` and returns the prior activity for [`Meter::leave`].
    pub(crate) fn enter(&mut self, activity: Activity) -> Activity {
        match &mut self.clock {
            Some(clock) => {
                clock.charge();
                std::mem::replace(&mut clock.activity, activity)
            }
            None => activity,
        }
    }

    /// Restores the prior profiling phase `previous` returned by [`Meter::enter`].
    pub(crate) fn leave(&mut self, previous: Activity) {
        self.enter(previous);
    }

    /// Captures starting meter and `solver` counters, starting or stopping the clock
    /// according to `profile`.
    pub(crate) fn read(&mut self, profile: bool, solver: jolty_numerics::Work) -> Reading {
        match (&mut self.clock, profile) {
            (Some(clock), true) => clock.charge(),
            (None, true) => {
                self.clock = Some(Clock {
                    activity: Activity::Other,
                    since: Instant::now(),
                    spent: Profile::default(),
                });
            }
            (_, false) => self.clock = None,
        }
        Reading {
            iterations: self.iterations,
            solver,
            spent: self.clock.as_ref().map(|clock| clock.spent),
        }
    }

    /// Computes delta [`Work`] and optional [`Profile`] elapsed since `start`.
    pub(crate) fn since(
        &mut self,
        start: &Reading,
        solver: jolty_numerics::Work,
    ) -> (Work, Option<Profile>) {
        let work = Work {
            iterations: self.iterations - start.iterations,
            factorizations: solver.factorizations - start.solver.factorizations,
            refactorizations: solver.refactorizations - start.solver.refactorizations,
            fallbacks: solver.fallbacks - start.solver.fallbacks,
            reuses: solver.reuses - start.solver.reuses,
        };
        let profile = match (&mut self.clock, &start.spent) {
            (Some(clock), Some(earlier)) => {
                clock.charge();
                Some(clock.spent.since(earlier))
            }
            _ => None,
        };
        (work, profile)
    }
}

impl Simulator {
    /// Captures starting meter and solver counters for an analysis, toggling profiling via
    /// `profile`.
    pub(crate) fn read_meter(&mut self, profile: bool) -> Reading {
        self.state
            .workspace
            .meter
            .read(profile, self.state.solver.statistics())
    }

    /// Computes [`Work`] and optional [`Profile`] accumulated since `reading`.
    pub(crate) fn metered_since(&mut self, reading: &Reading) -> (Work, Option<Profile>) {
        self.state
            .workspace
            .meter
            .since(reading, self.state.solver.statistics())
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use jolty_numerics::Work;

    use super::{Activity, Meter};

    #[test]
    fn a_profile_charges_each_span_to_the_activity_it_was_spent_in() {
        let mut meter = Meter::default();
        let before = Instant::now();
        let start = meter.read(true, Work::default());
        let outer = meter.enter(Activity::StepControl);
        let inner = meter.enter(Activity::Factoring);
        let entered = Instant::now();
        while entered.elapsed() < Duration::from_millis(1) {}
        meter.leave(inner);
        meter.leave(outer);
        meter.iterate();
        let solver = Work {
            refactorizations: 2,
            ..Work::default()
        };
        let (work, profile) = meter.since(&start, solver);
        let elapsed = before.elapsed();
        assert_eq!((work.iterations, work.refactorizations), (1, 2));
        let profile = profile.expect("a profile");
        assert!(profile.factoring >= Duration::from_millis(1), "{profile:?}");
        assert_eq!(profile.evaluating, Duration::ZERO);
        assert!(profile.total() <= elapsed, "{profile:?} in {elapsed:?}");
        // An analysis that does not profile stops the clock, and measures no time.
        let start = meter.read(false, solver);
        meter.enter(Activity::Evaluating);
        assert_eq!(meter.since(&start, solver), (super::Work::default(), None));
    }
}
