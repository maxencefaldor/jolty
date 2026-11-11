//! An expression as a list of instructions, each reading earlier ones: evaluated forwards
//! for its value, then backwards for its gradient.

use crate::operations::{Binary, Unary};
use crate::parser::Node;

/// One instruction, whose operands are the indices of earlier instructions.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Instruction {
    Constant(f64),
    Variable(usize),
    Unary(Unary, usize),
    Binary(Binary, usize, usize),
    Limit([usize; 3]),
    Conditional([usize; 3]),
    Table(usize, Box<[(f64, f64)]>),
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
        Node::Table(x, points) => Instruction::Table(emit(x, tape), points.clone().into()),
    };
    tape.push(instruction);
    tape.len() - 1
}

/// Evaluates `tape` at `variables` into `values`, then, backwards, the derivatives of its
/// root with respect to each variable into `gradient`, using `adjoints`. Returns the
/// root's value.
pub(crate) fn evaluate(
    tape: &[Instruction],
    variables: &[f64],
    (values, adjoints): (&mut [f64], &mut [f64]),
    gradient: &mut [f64],
) -> f64 {
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
            Instruction::Table(x, points) => interpolate(points, values[*x]).0,
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
            Instruction::Table(x, points) => {
                adjoints[*x] += adjoint * interpolate(points, values[*x]).1;
            }
        }
    }
    values[root]
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

/// The piecewise-linear interpolation of `points`, at least one, at `x`, constant beyond
/// the first and the last, and its slope there.
pub(crate) fn interpolate(points: &[(f64, f64)], x: f64) -> (f64, f64) {
    let after = points.partition_point(|&(abscissa, _)| abscissa <= x);
    if after == 0 {
        return (points[0].1, 0.0);
    }
    let (x0, y0) = points[after - 1];
    match points.get(after) {
        Some(&(x1, y1)) => {
            let slope = (y1 - y0) / (x1 - x0);
            (y0 + slope * (x - x0), slope)
        }
        None => (y0, 0.0),
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
    fn interpolation_is_constant_beyond_the_points() {
        let points = [(0.0, 1.0), (1.0, 3.0), (3.0, 4.0)];
        assert_eq!(interpolate(&points, -1.0), (1.0, 0.0));
        assert_eq!(interpolate(&points, 0.5), (2.0, 2.0));
        assert_eq!(interpolate(&points, 2.0), (3.5, 0.5));
        assert_eq!(interpolate(&points, 9.0), (4.0, 0.0));
        assert_eq!(interpolate(&[(1.0, 7.0)], 0.0), (7.0, 0.0));
    }
}
