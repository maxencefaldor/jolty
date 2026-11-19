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
    Table(usize, Box<Points>),
}

/// The kind of three-operand instruction emitted by [`compile`].
#[derive(Clone, Copy)]
enum Ternary {
    Limit,
    Conditional,
}

/// Explicit continuation frame for iterative post-order compilation of [`Node`].
enum Build<'a> {
    Unary(Unary),
    BinaryLeft(Binary, &'a Node),
    BinaryRight(Binary, usize),
    TernaryFirst(Ternary, &'a Node, &'a Node),
    TernarySecond(Ternary, usize, &'a Node),
    TernaryThird(Ternary, usize, usize),
    Table(&'a Points),
}

/// The instructions of `node`, its root last, compiled iteratively in constant stack space.
pub(crate) fn compile(node: &Node) -> Vec<Instruction> {
    let mut tape = Vec::new();
    let mut frames: Vec<Build<'_>> = Vec::new();
    let mut current = node;
    'visit: loop {
        let mut index = loop {
            match current {
                Node::Constant(value) => {
                    tape.push(Instruction::Constant(*value));
                    break tape.len() - 1;
                }
                Node::Variable(variable) => {
                    tape.push(Instruction::Variable(*variable));
                    break tape.len() - 1;
                }
                Node::Unary(operation, operand) => {
                    frames.push(Build::Unary(*operation));
                    current = operand;
                }
                Node::Binary(operation, left, right) => {
                    frames.push(Build::BinaryLeft(*operation, right));
                    current = left;
                }
                Node::Limit(arguments) => {
                    let [first, second, third] = arguments.as_ref();
                    frames.push(Build::TernaryFirst(Ternary::Limit, second, third));
                    current = first;
                }
                Node::Conditional(arguments) => {
                    let [first, second, third] = arguments.as_ref();
                    frames.push(Build::TernaryFirst(Ternary::Conditional, second, third));
                    current = first;
                }
                Node::Table(table) => {
                    let (input, points) = table.as_ref();
                    frames.push(Build::Table(points));
                    current = input;
                }
            }
        };
        while let Some(frame) = frames.pop() {
            let instruction = match frame {
                Build::Unary(operation) => Instruction::Unary(operation, index),
                Build::BinaryLeft(operation, right) => {
                    frames.push(Build::BinaryRight(operation, index));
                    current = right;
                    continue 'visit;
                }
                Build::BinaryRight(operation, left) => Instruction::Binary(operation, left, index),
                Build::TernaryFirst(kind, second, third) => {
                    frames.push(Build::TernarySecond(kind, index, third));
                    current = second;
                    continue 'visit;
                }
                Build::TernarySecond(kind, first, third) => {
                    frames.push(Build::TernaryThird(kind, first, index));
                    current = third;
                    continue 'visit;
                }
                Build::TernaryThird(Ternary::Limit, first, second) => {
                    Instruction::Limit([first, second, index])
                }
                Build::TernaryThird(Ternary::Conditional, first, second) => {
                    Instruction::Conditional([first, second, index])
                }
                Build::Table(points) => Instruction::Table(index, Box::new(points.clone())),
            };
            tape.push(instruction);
            index = tape.len() - 1;
        }
        return tape;
    }
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
        curvatures,
        limited,
        ..
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
                let (used, stepped) = if points.limits {
                    let step = inputs[index].and_then(|last| points.limit(applied, last));
                    *limited |= step.is_some();
                    let used = step.unwrap_or(applied);
                    inputs[index] = Some(used);
                    (used, step.is_some())
                } else {
                    (applied, false)
                };
                // Evaluated at the limited input, and extended linearly to the applied.
                let (value, slope, curvature) = points.interpolate(used);
                slopes[index] = slope;
                curvatures[index] = if stepped { 0.0 } else { curvature };
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

/// Differentiates the value and gradient that [`evaluate`] last left in `workspace` along
/// `tangent`, one rate per variable: forward for each instruction's tangent, then backward
/// for each adjoint's tangent, writing the gradient's directional derivative into
/// `gradient_tangent` and returning the root's.
pub(crate) fn along(
    tape: &[Instruction],
    tangent: &[f64],
    workspace: &mut Workspace,
    gradient_tangent: &mut [f64],
) -> f64 {
    let Workspace {
        values,
        adjoints,
        slopes,
        curvatures,
        tangents,
        adjoint_tangents,
        ..
    } = workspace;
    along_forward(tape, tangent, values, slopes, tangents);
    along_backward(
        tape,
        (values, slopes, curvatures),
        (adjoints, tangents),
        adjoint_tangents,
        gradient_tangent,
    );
    tangents[tape.len() - 1]
}

/// Forward tangent pass of [`along`], guarding zero tangents so `0 * inf` never becomes `NaN`.
fn along_forward(
    tape: &[Instruction],
    tangent: &[f64],
    values: &[f64],
    slopes: &[f64],
    tangents: &mut [f64],
) {
    for (index, instruction) in tape.iter().enumerate() {
        tangents[index] = match instruction {
            Instruction::Constant(_) => 0.0,
            Instruction::Variable(variable) => tangent[*variable],
            Instruction::Unary(operation, x) => {
                if tangents[*x] == 0.0 {
                    0.0
                } else {
                    operation.slope(values[*x], values[index]) * tangents[*x]
                }
            }
            Instruction::Binary(operation, a, b) => {
                let (tangent_a, tangent_b) = (tangents[*a], tangents[*b]);
                if tangent_a == 0.0 && tangent_b == 0.0 {
                    0.0
                } else {
                    let (slope_a, slope_b) =
                        operation.slopes(values[*a], values[*b], values[index]);
                    let mut sum = 0.0;
                    if tangent_a != 0.0 {
                        sum += slope_a * tangent_a;
                    }
                    if tangent_b != 0.0 {
                        sum += slope_b * tangent_b;
                    }
                    sum
                }
            }
            Instruction::Limit([x, a, b]) => {
                let chosen = limit(values[*x], values[*a], values[*b]).1;
                tangents[[*x, *a, *b][chosen]]
            }
            Instruction::Conditional([condition, then, otherwise]) => {
                tangents[if values[*condition] == 0.0 {
                    *otherwise
                } else {
                    *then
                }]
            }
            Instruction::Table(x, _) => {
                if tangents[*x] == 0.0 {
                    0.0
                } else {
                    slopes[index] * tangents[*x]
                }
            }
        };
    }
}

/// Reverse adjoint-tangent pass of [`along`], skipping zero adjoints and zero tangents so
/// infinite second derivatives along unmoved directions do not produce `NaN`.
fn along_backward(
    tape: &[Instruction],
    (values, slopes, curvatures): (&[f64], &[f64], &[f64]),
    (adjoints, tangents): (&[f64], &[f64]),
    adjoint_tangents: &mut [f64],
    gradient_tangent: &mut [f64],
) {
    gradient_tangent.fill(0.0);
    adjoint_tangents.fill(0.0);
    for (index, instruction) in tape.iter().enumerate().rev() {
        let (adjoint, adjoint_tangent) = (adjoints[index], adjoint_tangents[index]);
        if adjoint == 0.0 && adjoint_tangent == 0.0 {
            continue;
        }
        match instruction {
            Instruction::Constant(_) => {}
            Instruction::Variable(variable) => gradient_tangent[*variable] += adjoint_tangent,
            Instruction::Unary(operation, x) => {
                let has_curvature = adjoint != 0.0 && tangents[*x] != 0.0;
                if adjoint_tangent != 0.0 || has_curvature {
                    let slope = operation.slope(values[*x], values[index]);
                    let mut delta = 0.0;
                    if adjoint_tangent != 0.0 {
                        delta += adjoint_tangent * slope;
                    }
                    if has_curvature {
                        let curvature = operation.curvature(values[*x], values[index], slope);
                        delta += adjoint * curvature * tangents[*x];
                    }
                    adjoint_tangents[*x] += delta;
                }
            }
            Instruction::Binary(operation, a, b) => {
                let (tangent_a, tangent_b) = (tangents[*a], tangents[*b]);
                let has_curvature = adjoint != 0.0 && (tangent_a != 0.0 || tangent_b != 0.0);
                if adjoint_tangent != 0.0 || has_curvature {
                    let (slope_a, slope_b) =
                        operation.slopes(values[*a], values[*b], values[index]);
                    if adjoint_tangent != 0.0 {
                        adjoint_tangents[*a] += adjoint_tangent * slope_a;
                        adjoint_tangents[*b] += adjoint_tangent * slope_b;
                    }
                    if has_curvature {
                        let [by_a, mixed, by_b] =
                            operation.curvatures(values[*a], values[*b], (slope_a, slope_b));
                        let mut delta_a = 0.0;
                        let mut delta_b = 0.0;
                        if tangent_a != 0.0 {
                            delta_a += by_a * tangent_a;
                            delta_b += mixed * tangent_a;
                        }
                        if tangent_b != 0.0 {
                            delta_a += mixed * tangent_b;
                            delta_b += by_b * tangent_b;
                        }
                        if delta_a != 0.0 {
                            adjoint_tangents[*a] += adjoint * delta_a;
                        }
                        if delta_b != 0.0 {
                            adjoint_tangents[*b] += adjoint * delta_b;
                        }
                    }
                }
            }
            Instruction::Limit([x, a, b]) => {
                let chosen = limit(values[*x], values[*a], values[*b]).1;
                adjoint_tangents[[*x, *a, *b][chosen]] += adjoint_tangent;
            }
            Instruction::Conditional([condition, then, otherwise]) => {
                let chosen = if values[*condition] == 0.0 {
                    *otherwise
                } else {
                    *then
                };
                adjoint_tangents[chosen] += adjoint_tangent;
            }
            Instruction::Table(x, _) => {
                let mut delta = 0.0;
                if adjoint_tangent != 0.0 {
                    delta += adjoint_tangent * slopes[index];
                }
                if adjoint != 0.0 && tangents[*x] != 0.0 {
                    delta += adjoint * curvatures[index] * tangents[*x];
                }
                adjoint_tangents[*x] += delta;
            }
        }
    }
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
        let last = points.len() - 1;
        // A lone point is held, one unit away on each side.
        let (before, after) = if last == 0 {
            (points[0].0 - 1.0, points[0].0 + 1.0)
        } else {
            (
                2.0 * points[0].0 - points[1].0,
                2.0 * points[last].0 - points[last - 1].0,
            )
        };
        let (first, final_) = match beyond {
            Beyond::Extend if last > 0 => (
                2.0 * points[0].1 - points[1].1,
                2.0 * points[last].1 - points[last - 1].1,
            ),
            _ => (points[0].1, points[last].1),
        };
        let mut abscissae = Vec::with_capacity(points.len() + 2);
        let mut ordinates = Vec::with_capacity(points.len() + 2);
        abscissae.push(before);
        ordinates.push(first);
        for &(abscissa, ordinate) in points {
            abscissae.push(abscissa);
            ordinates.push(ordinate);
        }
        abscissae.push(after);
        ordinates.push(final_);
        Self {
            abscissae: abscissae.into(),
            ordinates: ordinates.into(),
            rounding,
            limits: rounding > 0.0,
        }
    }

    /// The abscissae, with one added before the first point and one after the last.
    pub(crate) fn abscissae(&self) -> &[f64] {
        &self.abscissae
    }

    /// The ordinates, with the added points' holding or extending the ends.
    pub(crate) fn ordinates(&self) -> &[f64] {
        &self.ordinates
    }

    /// The fraction of the shorter segment beside each corner its parabola spans.
    pub(crate) const fn rounding(&self) -> f64 {
        self.rounding
    }

    /// The input a Newton step from `last` to `x` may reach, as XSPICE's `pwl` limits it:
    /// no further than 30% of the segment `last` lies in, and not past its ends. Returns
    /// `None` when the step needs no limiting.
    pub(crate) fn limit(&self, x: f64, last: f64) -> Option<f64> {
        /// Maximum fraction of a segment crossed in one Newton step.
        ///
        /// Origin: ngspice XSPICE `pwl` step limiter limits each step to 30% of the active segment.
        const FRACTION: f64 = 0.3;
        /// Small offset below a breakpoint keeping a downward-limited step inside the preceding
        /// segment.
        ///
        /// Origin: ngspice XSPICE `pwl` step limiter boundary offset (`1e-9`).
        const EPSILON: f64 = 1e-9;
        if !x.is_finite() || !last.is_finite() {
            return None;
        }
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

    /// The interpolation at `x`, its slope there, and its second derivative there.
    pub(crate) fn interpolate(&self, x: f64) -> (f64, f64, f64) {
        if x.is_nan() {
            return (f64::NAN, f64::NAN, f64::NAN);
        }
        let (xs, ys) = (&self.abscissae, &self.ordinates);
        let last = xs.len() - 1;
        let line = |anchor: usize, from: usize, to: usize| {
            let slope = (ys[to] - ys[from]) / (xs[to] - xs[from]);
            (ys[anchor] + (x - xs[anchor]) * slope, slope, 0.0)
        };
        if x <= f64::midpoint(xs[0], xs[1]) {
            return line(0, 0, 1);
        }
        if x >= f64::midpoint(xs[last - 1], xs[last]) {
            return line(last, last - 1, last);
        }
        // The corner at `xs[i]` holds the inputs between the midpoints beside it.
        let j = xs.partition_point(|&abscissa| abscissa <= x);
        let i = if x < f64::midpoint(xs[j - 1], xs[j]) {
            j - 1
        } else {
            j
        };
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
            2.0 * curvature,
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
            (-1.0, (1.0, 0.0, 0.0), (-1.0, 2.0, 0.0)),
            (0.5, (2.0, 2.0, 0.0), (2.0, 2.0, 0.0)),
            (2.0, (3.5, 0.5, 0.0), (3.5, 0.5, 0.0)),
            (9.0, (4.0, 0.0, 0.0), (7.0, 0.5, 0.0)),
        ] {
            assert!(table.interpolate(x) == held, "table at {x}");
            assert!(pwl.interpolate(x) == extended, "pwl at {x}");
        }
        let single = Points::new(&[(1.0, 7.0)], Beyond::Extend, 0.0);
        assert_eq!(single.interpolate(0.0), (7.0, 0.0, 0.0));
    }

    #[test]
    fn rounding_meets_both_segments_smoothly() {
        let points = Points::new(&[(0.0, 0.0), (1.0, 1.0), (3.0, 1.0)], Beyond::Hold, 0.1);
        // The corner at 1 is rounded over 0.1, a tenth of the shorter segment beside it.
        for x in [0.9, 1.1] {
            let (rounded, slope, _) = points.interpolate(x);
            let (sharp, sharp_slope, _) =
                Points::new(&[(0.0, 0.0), (1.0, 1.0), (3.0, 1.0)], Beyond::Hold, 0.0)
                    .interpolate(x);
            assert!((rounded - sharp).abs() < 1e-15 && (slope - sharp_slope).abs() < 1e-12);
        }
        let (value, slope, curvature) = points.interpolate(1.0);
        assert!(
            (value - (1.0 - 0.1 / 4.0)).abs() < 1e-15
                && (slope - 0.5).abs() < 1e-12
                && (curvature - -5.0).abs() < 1e-12
        );
        // The ends are corners too, held beyond: the first rounded over a tenth of 1.
        assert!(points.interpolate(-0.05).0 > 0.0 && points.interpolate(-0.2).0 == 0.0);
    }

    #[test]
    fn interpolate_returns_nan_tuple_for_nan_input() {
        let pairs = [(0.0, 0.0), (1.0, 2.0), (3.0, 1.0)];
        for rounding in [0.0, 0.25] {
            let points = Points::new(&pairs, Beyond::Hold, rounding);
            let (value, slope, curvature) = points.interpolate(f64::NAN);
            assert!(value.is_nan() && slope.is_nan() && curvature.is_nan());
        }
    }

    #[test]
    fn limit_returns_none_for_non_finite_inputs() {
        let points = Points::new(&[(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)], Beyond::Hold, 0.1);
        for non_finite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(points.limit(non_finite, 0.5), None);
            assert_eq!(points.limit(0.5, non_finite), None);
            assert_eq!(points.limit(non_finite, non_finite), None);
        }
    }

    #[test]
    fn interpolate_binary_search_matches_linear_scan() {
        let raw = [
            (-5.0, 2.0),
            (-1.0, -3.0),
            (0.0, 0.0),
            (1.0, 4.0),
            (1.0_f64.next_up(), 4.0),
            (2.5, 1.5),
            (10.0, -2.0),
        ];
        let points = Points::new(&raw, Beyond::Hold, 0.2);
        let xs = points.abscissae();
        let last = xs.len() - 1;
        let mut samples = vec![-10.0, 15.0];
        for pair in xs.windows(2) {
            let mid = f64::midpoint(pair[0], pair[1]);
            samples.extend([
                pair[0],
                pair[0].next_up(),
                mid.next_down(),
                mid,
                mid.next_up(),
                pair[1].next_down(),
                pair[1],
            ]);
        }
        for x in samples {
            if x <= f64::midpoint(xs[0], xs[1]) || x >= f64::midpoint(xs[last - 1], xs[last]) {
                continue;
            }
            let linear = 1
                + (1..last)
                    .take_while(|&k| x >= f64::midpoint(xs[k], xs[k + 1]))
                    .count();
            let j = xs.partition_point(|&abscissa| abscissa <= x);
            let binary = if x < f64::midpoint(xs[j - 1], xs[j]) {
                j - 1
            } else {
                j
            };
            assert_eq!(binary, linear, "mismatch at x = {x:?}");
            let (value, slope, curvature) = points.interpolate(x);
            assert!(value.is_finite() && slope.is_finite() && curvature.is_finite());
        }
        assert!(std::mem::size_of::<Instruction>() <= 32);
    }
}
