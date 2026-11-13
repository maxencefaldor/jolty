//! An expression as a list of instructions, each reading earlier ones: evaluated forwards
//! for its value, then backwards for its gradient.

use crate::operations::{Binary, Unary};
use crate::parser::Node;
use crate::{Choice, Workspace};

/// One instruction, whose operands are the indices of earlier instructions.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Instruction {
    Constant(f64),
    Variable(usize),
    Unary(Unary, usize),
    Binary(Binary, usize, usize),
    Limit([usize; 3]),
    Conditional([usize; 3]),
    Table(usize, Points),
}

/// The instructions of `node`, its root last.
pub(crate) fn compile(node: &Node) -> Vec<Instruction> {
    let mut tape = Vec::new();
    emit(node, &mut tape);
    tape
}

/// Appends the instructions of `node` to `tape` and returns its root's index.
fn emit(node: &Node, tape: &mut Vec<Instruction>) -> usize {
    let instruction = match node {
        Node::Constant(value) => Instruction::Constant(*value),
        Node::Variable(index) => Instruction::Variable(*index),
        Node::Unary(operation, x) => Instruction::Unary(*operation, emit(x, tape)),
        Node::Binary(operation, a, b) => {
            let a = emit(a, tape);
            Instruction::Binary(*operation, a, emit(b, tape))
        }
        Node::Limit(arguments) => Instruction::Limit(arguments.each_ref().map(|x| emit(x, tape))),
        Node::Conditional(arguments) => {
            Instruction::Conditional(arguments.each_ref().map(|x| emit(x, tape)))
        }
        Node::Table(x, points) => Instruction::Table(emit(x, tape), points.clone()),
    };
    tape.push(instruction);
    tape.len() - 1
}

/// Evaluates `tape` at `variables` in `workspace`, then, backwards, the derivatives of its
/// root with respect to each variable into `gradient`. Returns the root's value.
pub(crate) fn evaluate(
    tape: &[Instruction],
    variables: &[f64],
    workspace: &mut Workspace,
    gradient: &mut [f64],
) -> f64 {
    let Workspace {
        values,
        adjoints,
        inputs,
        slopes,
        limited,
    } = workspace;
    *limited = false;
    for (index, instruction) in tape.iter().enumerate() {
        values[index] = match instruction {
            Instruction::Constant(value) => *value,
            Instruction::Variable(variable) => variables[*variable],
            Instruction::Unary(operation, x) => operation.value(values[*x]),
            Instruction::Binary(operation, a, b) => operation.value(values[*a], values[*b]),
            Instruction::Limit([x, a, b]) => limit(values[*x], values[*a], values[*b]).0,
            Instruction::Conditional([condition, then, otherwise]) => {
                values[if values[*condition] == 0.0 {
                    *otherwise
                } else {
                    *then
                }]
            }
            Instruction::Table(x, points) => {
                let applied = values[*x];
                let used = if points.limits {
                    let step = inputs[index].and_then(|last| points.limit(applied, last));
                    *limited |= step.is_some();
                    let used = step.unwrap_or(applied);
                    inputs[index] = Some(used);
                    used
                } else {
                    applied
                };
                // Evaluated at the limited input, and extended linearly to the applied.
                let (value, slope) = points.interpolate(used);
                slopes[index] = slope;
                value + slope * (applied - used)
            }
        };
    }
    gradient.fill(0.0);
    adjoints.fill(0.0);
    let root = tape.len() - 1;
    adjoints[root] = 1.0;
    for (index, instruction) in tape.iter().enumerate().rev() {
        let adjoint = adjoints[index];
        // A part the root does not depend on here, such as a conditional's other branch,
        // passes nothing on, even where its own derivatives are not finite.
        if adjoint == 0.0 {
            continue;
        }
        match instruction {
            Instruction::Constant(_) => {}
            Instruction::Variable(variable) => gradient[*variable] += adjoint,
            Instruction::Unary(operation, x) => {
                adjoints[*x] += adjoint * operation.slope(values[*x], values[index]);
            }
            Instruction::Binary(operation, a, b) => {
                let (slope_a, slope_b) = operation.slopes(values[*a], values[*b], values[index]);
                adjoints[*a] += adjoint * slope_a;
                adjoints[*b] += adjoint * slope_b;
            }
            Instruction::Limit([x, a, b]) => {
                let chosen = limit(values[*x], values[*a], values[*b]).1;
                adjoints[[*x, *a, *b][chosen]] += adjoint;
            }
            Instruction::Conditional([condition, then, otherwise]) => {
                let chosen = if values[*condition] == 0.0 {
                    *otherwise
                } else {
                    *then
                };
                adjoints[chosen] += adjoint;
            }
            Instruction::Table(x, _) => adjoints[*x] += adjoint * slopes[index],
        }
    }
    values[root]
}

/// How a choice's value turns on its input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// A comparison, a step or a sign: it changes where its input crosses zero.
    Crossing,
    /// A floor: it changes where its input crosses an integer, which it then becomes.
    Floor,
    /// A ceiling: it changes where its input crosses an integer, which it was before.
    Ceil,
}

/// Appends to `found` the choices of `tape` whose `values` its root read, through the
/// branches its conditionals took: comparisons, steps, signs, floors and ceilings.
pub(crate) fn choices(tape: &[Instruction], values: &[f64], found: &mut Vec<Choice>) {
    let mut read = vec![false; tape.len()];
    if let Some(root) = read.last_mut() {
        *root = true;
    }
    for (index, instruction) in tape.iter().enumerate().rev() {
        if !read[index] {
            continue;
        }
        let made = |input: f64, kind: Kind| Choice {
            instruction: index,
            input,
            chosen: values[index],
            kind,
        };
        match instruction {
            Instruction::Constant(_) | Instruction::Variable(_) => {}
            Instruction::Unary(operation, x) => {
                read[*x] = true;
                let kind = match operation {
                    Unary::Step | Unary::Sign => Some(Kind::Crossing),
                    Unary::Floor => Some(Kind::Floor),
                    Unary::Ceil => Some(Kind::Ceil),
                    _ => None,
                };
                found.extend(kind.map(|kind| made(values[*x], kind)));
            }
            Instruction::Binary(operation, a, b) => {
                read[*a] = true;
                read[*b] = true;
                if operation.compares() {
                    found.push(made(values[*a] - values[*b], Kind::Crossing));
                }
            }
            Instruction::Limit(arguments) => arguments.iter().for_each(|&x| read[x] = true),
            Instruction::Conditional([condition, then, otherwise]) => {
                read[*condition] = true;
                read[if values[*condition] == 0.0 {
                    *otherwise
                } else {
                    *then
                }] = true;
            }
            Instruction::Table(x, _) => read[*x] = true,
        }
    }
}

/// `x` held between `a` and `b`, in either order, as SPICE's `limit` holds it, and which of
/// the three it is: 0 for `x`, 1 for `a`, 2 for `b`.
pub(crate) fn limit(x: f64, a: f64, b: f64) -> (f64, usize) {
    let (low, high, low_index, high_index) = if a > b { (b, a, 2, 1) } else { (a, b, 1, 2) };
    if x > high {
        (high, high_index)
    } else if x < low {
        (low, low_index)
    } else {
        (x, 0)
    }
}

/// Points with ascending abscissae, interpolated linearly between, held or extended beyond
/// the first and the last, with each corner rounded by a parabola over `rounding` times the
/// shorter of the segments beside it, and Newton's steps through them limited if `limits`:
/// the interpolation of XSPICE's `pwl`.
///
/// The points are stored with one more at each end, along which the interpolation runs
/// beyond them: the end's own ordinate to hold it, or the end segment's to extend it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Points {
    abscissae: Box<[f64]>,
    ordinates: Box<[f64]>,
    rounding: f64,
    pub(crate) limits: bool,
}

/// What the interpolation of points is beyond the first and the last.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Beyond {
    /// The first or the last point's ordinate.
    Hold,
    /// The first or the last segment extended.
    Extend,
}

impl Points {
    /// The interpolation of `points`, at least one, with ascending abscissae, rounded and
    /// its input's steps limited, as XSPICE's `pwl` does, if `rounding` is positive.
    pub(crate) fn new(points: &[(f64, f64)], beyond: Beyond, rounding: f64) -> Self {
        let mut abscissae: Vec<f64> = points.iter().map(|point| point.0).collect();
        let mut ordinates: Vec<f64> = points.iter().map(|point| point.1).collect();
        let last = abscissae.len() - 1;
        // A lone point is held, one unit away on each side.
        let (before, after) = if last == 0 {
            (abscissae[0] - 1.0, abscissae[0] + 1.0)
        } else {
            (
                2.0 * abscissae[0] - abscissae[1],
                2.0 * abscissae[last] - abscissae[last - 1],
            )
        };
        let (first, final_) = match beyond {
            Beyond::Extend if last > 0 => (
                2.0 * ordinates[0] - ordinates[1],
                2.0 * ordinates[last] - ordinates[last - 1],
            ),
            _ => (ordinates[0], ordinates[last]),
        };
        abscissae.insert(0, before);
        abscissae.push(after);
        ordinates.insert(0, first);
        ordinates.push(final_);
        Self {
            abscissae: abscissae.into(),
            ordinates: ordinates.into(),
            rounding,
            limits: rounding > 0.0,
        }
    }

    /// The input a Newton step from `last` to `x` may reach, as XSPICE's `pwl` limits it:
    /// no further than 30% of the segment `last` lies in, and not past its ends. Returns
    /// `None` when the step needs no limiting.
    pub(crate) fn limit(&self, x: f64, last: f64) -> Option<f64> {
        const FRACTION: f64 = 0.3;
        const EPSILON: f64 = 1e-9;
        let step = |lower: f64, upper: f64| {
            let most = FRACTION * (upper - lower);
            (most < (x - last).abs()).then(|| last + most.copysign(x - last))
        };
        let xs = &self.abscissae;
        let last_index = xs.len() - 1;
        if last < xs[0] {
            let limited = if xs[0] - x > xs[1] - xs[0] {
                step(x, xs[0])
            } else {
                step(xs[0], xs[1])
            };
            return if limited.unwrap_or(x) >= xs[0] {
                Some(xs[0])
            } else {
                limited
            };
        }
        if last >= xs[last_index] {
            let limited = if x - xs[last_index] > xs[last_index] - xs[last_index - 1] {
                step(xs[last_index], x)
            } else {
                step(xs[last_index - 1], xs[last_index])
            };
            return if limited.unwrap_or(x) < xs[last_index] {
                Some(xs[last_index] - EPSILON)
            } else {
                limited
            };
        }
        let i = xs.partition_point(|&abscissa| abscissa <= last);
        let limited = step(xs[i - 1], xs[i]);
        let reached = limited.unwrap_or(x);
        if reached > xs[i] {
            Some(xs[i])
        } else if reached < xs[i - 1] {
            Some(xs[i - 1] - EPSILON)
        } else {
            limited
        }
    }

    /// The interpolation at `x`, and its slope there.
    pub(crate) fn interpolate(&self, x: f64) -> (f64, f64) {
        let (xs, ys) = (&self.abscissae, &self.ordinates);
        let last = xs.len() - 1;
        let line = |anchor: usize, from: usize, to: usize| {
            let slope = (ys[to] - ys[from]) / (xs[to] - xs[from]);
            (ys[anchor] + (x - xs[anchor]) * slope, slope)
        };
        if x <= f64::midpoint(xs[0], xs[1]) {
            return line(0, 0, 1);
        }
        if x >= f64::midpoint(xs[last - 1], xs[last]) {
            return line(last, last - 1, last);
        }
        // The corner at `xs[i]` holds the inputs between the midpoints beside it.
        let i = 1
            + (1..last)
                .take_while(|&k| x >= f64::midpoint(xs[k], xs[k + 1]))
                .count();
        let (below, above) = (xs[i] - xs[i - 1], xs[i + 1] - xs[i]);
        let domain = self.rounding * below.min(above);
        if x < xs[i] - domain {
            return line(i, i - 1, i);
        }
        if x >= xs[i] + domain {
            return line(i, i, i + 1);
        }
        // XSPICE's parabola, which meets both segments with their slopes.
        let (lower, upper) = ((ys[i] - ys[i - 1]) / below, (ys[i + 1] - ys[i]) / above);
        let x_upper = xs[i] + domain;
        let y_upper = ys[i] + upper * domain;
        let curvature = (upper - lower) / 4.0 * (1.0 / domain);
        let linear = upper - 2.0 * curvature * x_upper;
        let constant = y_upper - curvature * x_upper * x_upper - linear * x_upper;
        (
            curvature * x * x + linear * x + constant,
            2.0 * curvature * x + linear,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_holds_between_bounds_in_either_order() {
        assert_eq!(limit(5.0, 0.0, 1.0), (1.0, 2));
        assert_eq!(limit(5.0, 1.0, 0.0), (1.0, 1));
        assert_eq!(limit(-5.0, 1.0, 0.0), (0.0, 2));
        assert_eq!(limit(0.25, 1.0, 0.0), (0.25, 0));
    }

    #[test]
    fn interpolation_holds_or_extends_beyond_the_points() {
        let points = [(0.0, 1.0), (1.0, 3.0), (3.0, 4.0)];
        let (table, pwl) = (
            Points::new(&points, Beyond::Hold, 0.0),
            Points::new(&points, Beyond::Extend, 0.0),
        );
        for (x, held, extended) in [
            (-1.0, (1.0, 0.0), (-1.0, 2.0)),
            (0.5, (2.0, 2.0), (2.0, 2.0)),
            (2.0, (3.5, 0.5), (3.5, 0.5)),
            (9.0, (4.0, 0.0), (7.0, 0.5)),
        ] {
            assert!(table.interpolate(x) == held, "table at {x}");
            assert!(pwl.interpolate(x) == extended, "pwl at {x}");
        }
        let single = Points::new(&[(1.0, 7.0)], Beyond::Extend, 0.0);
        assert_eq!(single.interpolate(0.0), (7.0, 0.0));
    }

    #[test]
    fn rounding_meets_both_segments_smoothly() {
        let points = Points::new(&[(0.0, 0.0), (1.0, 1.0), (3.0, 1.0)], Beyond::Hold, 0.1);
        // The corner at 1 is rounded over 0.1, a tenth of the shorter segment beside it.
        for x in [0.9, 1.1] {
            let (rounded, slope) = points.interpolate(x);
            let (sharp, sharp_slope) =
                Points::new(&[(0.0, 0.0), (1.0, 1.0), (3.0, 1.0)], Beyond::Hold, 0.0)
                    .interpolate(x);
            assert!((rounded - sharp).abs() < 1e-15 && (slope - sharp_slope).abs() < 1e-12);
        }
        let (value, slope) = points.interpolate(1.0);
        assert!((value - (1.0 - 0.1 / 4.0)).abs() < 1e-15 && (slope - 0.5).abs() < 1e-12);
        // The ends are corners too, held beyond: the first rounded over a tenth of 1.
        assert!(points.interpolate(-0.05).0 > 0.0 && points.interpolate(-0.2).0 == 0.0);
    }
}
