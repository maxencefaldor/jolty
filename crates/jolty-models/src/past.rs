//! Historical unknown trajectories read by delay lines during transient analysis, retained
//! back to the longest delay referenced by any device instance.

use std::collections::VecDeque;

use jolty_numerics::UNMAPPED;

/// Maximum number of interpolation nodes: one more than the highest supported integration
/// order (Gear2 and trapezoidal, order 2; [`jolty_numerics::integration::Method::order`]).
const MAX_NODES: usize = 3;

/// Delayed conductance contribution reported by [`crate::Kernel::delayed`] that samples its
/// column unknown `delay` seconds in the past.
///
/// In small-signal analysis at complex frequency `s`, the contribution equals
/// `conductance · e^{−s·delay}`. At DC, past equals present, so the normal evaluation
/// includes `conductance` directly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Delayed {
    /// Local entry index within the device instance.
    pub entry: usize,
    /// Positive propagation delay, in seconds.
    pub delay: f64,
    /// Conductance weight applied to the delayed unknown, in the entry's units.
    pub conductance: f64,
}

/// Circuit-level placement of a [`Delayed`] conductance contribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DelayedEntry {
    /// Device instance index within the circuit.
    pub device: usize,
    /// Stored matrix position in the circuit sparsity pattern.
    pub position: u32,
    /// Circuit unknown index sampled in the past.
    pub column: u32,
    /// Propagation delay, in seconds.
    pub delay: f64,
    /// Conductance weight applied to the delayed unknown.
    pub conductance: f64,
}

/// Recorded history of selected circuit unknowns and integration restart corners for delay
/// line interpolation.
///
/// Interpolation uses polynomials matching the transient integration order and stays within
/// the smooth segment between corners so no polynomial crosses a waveform discontinuity.
/// Transient runs retain samples covering the longest device delay plus `order` preceding
/// points via [`Past::forget_before`].
#[derive(Debug, Clone, Default)]
pub struct Past {
    /// Degree of the interpolating polynomials, at most [`MAX_NODES`] − 1.
    order: usize,
    /// Packed column index for each circuit unknown, or [`UNMAPPED`] when not recorded.
    columns: Vec<u32>,
    /// Number of recorded unknowns per sample point.
    width: usize,
    /// Sample timestamps in strictly increasing order.
    times: VecDeque<f64>,
    /// Flag per sample indicating whether integration restarted at a waveform corner.
    corners: VecDeque<bool>,
    /// Packed unknown values (`width` values per sample), oldest first.
    values: VecDeque<f64>,
}

impl Past {
    /// Creates an empty history buffer for a circuit with `unknowns` unknowns, recording the
    /// indices in `kept` for polynomial interpolation of degree `order` (clamped to at most 2).
    ///
    /// Matching `order` to the integration method keeps interpolation error `O(h^{order+1})`
    /// consistent with step local truncation error.
    #[must_use]
    pub fn new(unknowns: usize, kept: &[usize], order: usize) -> Self {
        let mut columns = vec![UNMAPPED; unknowns];
        let mut width: u32 = 0;
        for &unknown in kept {
            if let Some(column) = columns.get_mut(unknown)
                && *column == UNMAPPED
            {
                *column = width;
                width += 1;
            }
        }
        Self {
            order: order.clamp(1, MAX_NODES - 1),
            columns,
            width: width as usize,
            times: VecDeque::new(),
            corners: VecDeque::new(),
            values: VecDeque::new(),
        }
    }

    /// Appends an accepted sample at `time` (later than all stored samples) with solution
    /// `x`; `corner` indicates an integration restart point.
    pub fn push(&mut self, time: f64, x: &[f64], corner: bool) {
        let start = self.values.len();
        self.values.resize(start + self.width, 0.0);
        for (unknown, &column) in self.columns.iter().enumerate() {
            if column != UNMAPPED
                && let (Some(value), Some(&found)) =
                    (self.values.get_mut(start + column as usize), x.get(unknown))
            {
                *value = found;
            }
        }
        self.times.push_back(time);
        self.corners.push_back(corner);
    }

    /// Discards every stored sample after `time` when a transient step rolls back.
    pub fn back_to(&mut self, time: f64) {
        let kept = self.times.partition_point(|&each| each <= time);
        self.times.truncate(kept);
        self.corners.truncate(kept);
        self.values.truncate(kept * self.width);
    }

    /// Discards samples older than any interpolation at or after `horizon`, retaining the
    /// last `order` samples at or before `horizon`.
    ///
    /// Interpolating at `τ` uses `order + 1` points: the bracketing pair around `τ` plus
    /// `order − 1` neighboring points inside the same corner-free segment. The earliest
    /// reachable node is `order − 1` samples before the latest point at or before `horizon`.
    pub fn forget_before(&mut self, horizon: f64) {
        let before = self.times.partition_point(|&each| each <= horizon);
        let forgotten = before.saturating_sub(self.order);
        self.times.drain(..forgotten);
        self.corners.drain(..forgotten);
        self.values.drain(..forgotten * self.width);
    }

    /// Returns the number of stored sample points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Returns whether no sample points are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Interpolates the value of `unknown` at `time` and its derivative with respect to
    /// `present` at the current step time `now`.
    ///
    /// Includes `(now, present)` as an extra interpolation node when `now` exceeds all stored
    /// timestamps. Times at or before the first stored point return the initial rest value
    /// with zero derivative; returns `None` if `unknown` is unrecorded or the history is
    /// empty.
    #[must_use]
    pub fn at(&self, unknown: usize, time: f64, (now, present): (f64, f64)) -> Option<(f64, f64)> {
        let column = *self.columns.get(unknown)?;
        if column == UNMAPPED {
            return None;
        }
        let kept = self.times.len();
        let last = *self.times.back()?;
        let nodes = if now > last { kept + 1 } else { kept };
        let node = |index: usize| -> (f64, f64) {
            if index < kept {
                let value = self.values.get(index * self.width + column as usize);
                (self.times[index], value.copied().unwrap_or(0.0))
            } else {
                (now, present)
            }
        };
        if nodes == 1 || time <= self.times[0] {
            return Some((node(0).1, 0.0));
        }
        // The node at or before `time` and the one after it, the last two past the end.
        let at_or_before = self.times.partition_point(|&each| each <= time)
            + usize::from(nodes > kept && now <= time);
        let below = at_or_before.clamp(1, nodes - 1) - 1;
        // The nodes an interpolation may read, `order` either way at most, within the run
        // between the corners about `time`.
        let corner = |index: usize| index < kept && self.corners[index];
        let lowest = below.saturating_sub(self.order);
        let first = (lowest..=below)
            .rev()
            .find(|&index| corner(index))
            .unwrap_or(lowest);
        let highest = (below + 1 + self.order).min(nodes - 1);
        let final_node = (below + 1..=highest)
            .find(|&index| corner(index))
            .unwrap_or(highest);
        let (mut start, mut end) = (below, below + 1);
        while end - start < self.order {
            let earlier = (start > first).then(|| start - 1);
            let later = (end < final_node).then_some(end + 1);
            match (earlier, later) {
                (Some(earlier), Some(later)) => {
                    if (time - node(earlier).0).abs() <= (node(later).0 - time).abs() {
                        start = earlier;
                    } else {
                        end = later;
                    }
                }
                (Some(earlier), None) => start = earlier,
                (None, Some(later)) => end = later,
                (None, None) => break,
            }
        }
        let mut times = [0.0; MAX_NODES];
        let mut values = [0.0; MAX_NODES];
        for (offset, index) in (start..=end).enumerate() {
            if let (Some(at), Some(value)) = (times.get_mut(offset), values.get_mut(offset)) {
                (*at, *value) = node(index);
            }
        }
        let count = end - start + 1;
        let (mut value, mut weight) = (0.0, 0.0);
        for own in 0..count {
            let mut basis = 1.0;
            for other in (0..count).filter(|&other| other != own) {
                basis *= (time - times[other]) / (times[own] - times[other]);
            }
            value += basis * values[own];
            if start + own == kept {
                weight = basis;
            }
        }
        Some((value, weight))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a single-unknown history buffer from `(time, value, corner)` tuples for
    /// polynomial interpolation of degree `order`.
    fn past(order: usize, points: &[(f64, f64, bool)]) -> Past {
        let mut past = Past::new(1, &[0], order);
        for &(time, value, corner) in points {
            past.push(time, &[value], corner);
        }
        past
    }

    #[test]
    fn a_quadratic_is_read_exactly_and_a_line_at_first_order() {
        let square = |time: f64| time * time;
        let points: Vec<(f64, f64, bool)> = [0.0, 0.3, 0.7, 1.2, 2.0]
            .iter()
            .map(|&time| (time, square(time), time == 0.0))
            .collect();
        let quadratic = past(2, &points);
        for time in [0.1, 0.5, 0.95, 1.6] {
            let (value, weight) = quadratic.at(0, time, (3.0, 9.0)).unwrap();
            // Exact except for floating-point rounding of products near 1.
            assert!((value - square(time)).abs() < 1e-14, "{time}: {value}");
            assert_eq!(weight, 0.0);
        }
        let line = past(1, &points);
        let (value, _) = line.at(0, 0.5, (3.0, 9.0)).unwrap();
        // Linear between 0.3 and 0.7, whose squares are 0.09 and 0.49.
        assert!((value - 0.29).abs() < 1e-15, "{value}");
    }

    #[test]
    fn no_interpolation_spans_a_corner() {
        // |t − 1|, a corner at 1: each side is a line, read exactly at second order.
        let points: Vec<(f64, f64, bool)> = [
            (0.0, true),
            (0.4, false),
            (0.8, false),
            (1.0, true),
            (1.5, false),
            (2.0, false),
        ]
        .iter()
        .map(|&(time, corner): &(f64, bool)| (time, (time - 1.0_f64).abs(), corner))
        .collect();
        let read = past(2, &points);
        for time in [0.5, 0.9, 1.1, 1.7] {
            let (value, _) = read.at(0, time, (3.0, 2.0)).unwrap();
            assert!(
                (value - (time - 1.0_f64).abs()).abs() < 1e-15,
                "{time}: {value}"
            );
        }
    }

    #[test]
    fn the_present_is_a_node_when_the_time_lies_within_the_step() {
        let read = past(1, &[(0.0, 0.0, true), (1.0, 1.0, false)]);
        // Halfway from the last point, 1 at 1, to the present, 3 at 2.
        let (value, weight) = read.at(0, 1.5, (2.0, 3.0)).unwrap();
        assert_eq!((value, weight), (2.0, 0.5));
    }

    #[test]
    fn before_the_first_point_the_circuit_rests() {
        let read = past(2, &[(0.0, 4.0, true), (1.0, 5.0, false)]);
        assert_eq!(read.at(0, -0.5, (2.0, 6.0)), Some((4.0, 0.0)));
        assert_eq!(read.at(1, 0.5, (2.0, 6.0)), None);
    }

    #[test]
    fn forgetting_keeps_what_an_interpolation_after_the_horizon_reads() {
        let points: Vec<(f64, f64, bool)> = (0..10)
            .map(|step| (f64::from(step), f64::from(step * step), step == 0))
            .collect();
        let mut read = past(2, &points);
        let before = read.at(0, 5.5, (10.0, 100.0));
        read.forget_before(5.5);
        // The points at 4 and 5, then those after.
        assert_eq!(read.len(), 6);
        assert_eq!(read.at(0, 5.5, (10.0, 100.0)), before);
        read.back_to(6.0);
        assert_eq!(read.len(), 3);
    }
}
