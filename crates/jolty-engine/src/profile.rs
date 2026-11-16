//! What an analysis did: its work, counted always, and where its time went, measured when
//! its options ask.
//!
//! A simulator's meter counts every Newton iteration it runs, and its sparse solver every
//! factorization, from the moment it is made; an analysis reads both as it starts and as it
//! ends, and its work is the difference. While an analysis profiles, the meter's clock
//! charges each span of time to the one activity the simulator is in, the activities nested
//! within another charged to themselves, so the activities sum to the whole. The clock, read
//! some ten times a Newton iteration, adds about 0.23 µs to each on an Apple M2 Max, a tenth
//! of a small circuit's transient, such as a buck converter's, so it runs only for an
//! analysis whose options' `profile` is on; nothing it measures changes a result.

use std::time::{Duration, Instant};

use crate::Simulator;

/// The work an analysis did, counted whatever its options.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Work {
    /// The Newton iterations it ran, each solving for one step of the unknowns: those of
    /// the operating point it solved, unless the simulator took the one it remembered, and
    /// those of every step it tried, rejected or accepted.
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

/// Where an analysis spent its time, measured when its options' `profile` is on.
///
/// Each span of the analysis counts in the one activity it was spent in, so the activities
/// sum to its whole time, [`Profile::total`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Profile {
    /// Checking the options, choosing the devices the analysis records, and gathering the
    /// sources' breakpoints.
    pub preparing: Duration,
    /// Evaluating the devices.
    pub evaluating: Duration,
    /// Turning what the devices contributed into each Newton iteration's matrix and
    /// right-hand side, and judging its residuals against their tolerances.
    pub assembling: Duration,
    /// Factoring matrices.
    pub factoring: Duration,
    /// Solving with the factors.
    pub solving: Duration,
    /// Judging each step of a transient by its error, choosing the next, and finding the
    /// events a step lands on: the logic's changes and the devices' jumps.
    pub step_control: Duration,
    /// Reading the recorded devices at each point kept.
    pub recording: Duration,
    /// Everything else: Newton iteration's updates and its tests of their steps, the
    /// integration formula's terms and the prediction each step starts from, and what a
    /// transient carries along its steps, such as sensitivities.
    pub other: Duration,
}

impl Profile {
    /// The whole time: the sum of every activity's.
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

    /// The time spent in `activity`.
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

    /// The time spent since `earlier`, activity by activity: none is less, since time
    /// is only ever added.
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

/// What a simulator is doing, for its clock.
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

/// The clock of a meter: the activity the simulator is in, since when, and the time spent
/// in each activity before.
#[derive(Debug)]
struct Clock {
    activity: Activity,
    since: Instant,
    spent: Profile,
}

impl Clock {
    /// Charges the time since the last charge to the current activity.
    fn charge(&mut self) {
        let now = Instant::now();
        *self.spent.spent(self.activity) += now.saturating_duration_since(self.since);
        self.since = now;
    }
}

/// The Newton iterations a simulator has run since it was made, and, while an analysis
/// profiles, the time it has spent in each activity.
#[derive(Debug, Default)]
pub(crate) struct Meter {
    iterations: usize,
    clock: Option<Clock>,
}

/// A meter's and a sparse solver's readings as an analysis starts.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Reading {
    iterations: usize,
    solver: jolty_numerics::Work,
    spent: Option<Profile>,
}

impl Meter {
    /// Counts a Newton iteration.
    pub(crate) fn iterate(&mut self) {
        self.iterations += 1;
    }

    /// Enters `activity`, and returns the activity left, to [`Meter::leave`] for it.
    pub(crate) fn enter(&mut self, activity: Activity) -> Activity {
        match &mut self.clock {
            Some(clock) => {
                clock.charge();
                std::mem::replace(&mut clock.activity, activity)
            }
            None => activity,
        }
    }

    /// Leaves the current activity for `previous`, which [`Meter::enter`] returned.
    pub(crate) fn leave(&mut self, previous: Activity) {
        self.enter(previous);
    }

    /// Reads the meter and the sparse solver's `solver` statistics as an analysis starts,
    /// starting the clock if the analysis will `profile` and stopping it if not.
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

    /// The work done since `start`, the sparse solver's statistics now `solver`, and the
    /// time spent since, if the clock has run since.
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
    /// Reads the meter and the sparse solver as an analysis starts, starting the clock if
    /// the analysis will `profile` and stopping it if not.
    pub(crate) fn read_meter(&mut self, profile: bool) -> Reading {
        self.state
            .workspace
            .meter
            .read(profile, self.state.solver.statistics())
    }

    /// The work done since `reading`, and the time spent since, if the clock has run.
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
