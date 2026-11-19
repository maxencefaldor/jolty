//! The past a delay line reads: the values its unknowns took at the points a transient
//! accepted, back to the longest delay any instance reads them at.

use std::collections::VecDeque;

use jolty_numerics::UNMAPPED;

/// The maximum number of points an interpolation reads: one more than the highest order of an
/// integration method, Gear2's and the trapezoidal formula's 2
/// ([`jolty_numerics::integration::Method::order`]).
const MAX_NODES: usize = 3;

/// A part of an instance's conductance that reads the local unknown of its column `delay`
/// seconds in the past, which [`crate::Kernel::delayed`] states: in a small-signal analysis
/// at the complex frequency `s` it is `conductance · e^{−s·delay}`, and at DC, where the
/// past is the present, `conductance` itself, which the instance's evaluation includes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Delayed {
    /// The local entry, by its index among the instance's.
    pub entry: usize,
    /// The delay, in seconds; positive.
    pub delay: f64,
    /// The part of the entry's conductance that reads the past, in its own units.
    pub conductance: f64,
}

/// A [`Delayed`] part of the circuit's conductance, placed among its equations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DelayedEntry {
    /// The device whose instance reads the past, by its index among the circuit's.
    pub device: usize,
    /// The stored position of the entry in the circuit's pattern.
    pub position: u32,
    /// The circuit's unknown the entry's column is, which the instance reads in the past.
    pub column: u32,
    /// The delay, in seconds.
    pub delay: f64,
    /// The part of the entry's conductance that reads the past.
    pub conductance: f64,
}

/// The values some of a circuit's unknowns took at the points a transient accepted, and
/// where its integration restarted, as a delay line reads them: by interpolation of the
/// order of the transient's integration method, within the run of points between two
/// corners that holds the time read, so that no interpolation spans the corner a
/// waveform turns at.
///
/// A transient keeps the points of the last delay before its last accepted point, and the
/// `order` points before them that an interpolation there reads: of the longest delay its
/// instances read, as [`Past::forget_before`] says.
#[derive(Debug, Clone, Default)]
pub struct Past {
    /// The degree of the interpolating polynomials, at most [`MAX_NODES`] − 1.
    order: usize,
    /// The column of each of the circuit's unknowns among those kept, by the unknown's
    /// index, or [`UNMAPPED`] for one not kept.
    columns: Vec<u32>,
    /// The number of unknowns kept.
    width: usize,
    /// Each point's time, oldest first.
    times: VecDeque<f64>,
    /// Whether the integration restarted at each point, a corner in the waveforms.
    corners: VecDeque<bool>,
    /// The values of the unknowns kept at each point, `width` a point, oldest first.
    values: VecDeque<f64>,
}

impl Past {
    /// An empty past of a circuit of `unknowns` unknowns that keeps those of `kept`, by
    /// index, read by polynomials of degree `order`, the order of the integration method,
    /// so that an interpolation's error, `O(h^{order+1})` in the steps `h`, is of the
    /// order of a step's local truncation error; at most 2.
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

    /// Appends the point at `time`, later than every point kept, where the unknowns are
    /// `x`; `corner` says whether the integration restarts there.
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

    /// Forgets every point after `time`, as a transient going back to it does.
    pub fn back_to(&mut self, time: f64) {
        let kept = self.times.partition_point(|&each| each <= time);
        self.times.truncate(kept);
        self.corners.truncate(kept);
        self.values.truncate(kept * self.width);
    }

    /// Forgets the points no interpolation at `horizon` or later reads: every point but
    /// the last `order` at or before it.
    ///
    /// An interpolation at a time `τ` reads `order + 1` points: the two around `τ`,
    /// the last point at or before it and the first after, and `order − 1` more within
    /// the same run between corners. The earliest it can read is `order − 1` before the
    /// last point at or before `horizon`, the `order`th last at or before it.
    pub fn forget_before(&mut self, horizon: f64) {
        let before = self.times.partition_point(|&each| each <= horizon);
        let forgotten = before.saturating_sub(self.order);
        self.times.drain(..forgotten);
        self.corners.drain(..forgotten);
        self.values.drain(..forgotten * self.width);
    }

    /// The number of points kept.
    #[must_use]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Whether it keeps no point.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// The value the unknown `unknown` took at `time`, and its derivative with respect to
    /// the unknown's value `present` at `now`, the time a transient solves at: an
    /// interpolation of the points kept and of the present, when `now` is later than
    /// them all, as a node of its own. Before the first point, the unknown holds the
    /// value it took there, as the circuit rests before a transient starts. `None` for
    /// an unknown not kept, or before any point is.
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

    /// A past of one unknown at `points`, each a time, a value and whether it is a corner,
    /// read by polynomials of degree `order`.
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
            // Exact but for the rounding of a few products of numbers near 1.
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
