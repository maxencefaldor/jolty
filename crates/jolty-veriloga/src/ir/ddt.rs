//! Single-pass `O(N)` additive splitting and `ddt(...)` factor extraction for Verilog-A
//! contribution expressions.

use crate::ast::Expression;

/// One multiplicative or divisive factor wrapped around a contribution term.
#[derive(Clone, Copy)]
pub(super) enum Factor<'a> {
    LeftMul(&'a Expression),
    RightMul(&'a Expression),
    RightDiv(&'a Expression),
}

impl<'a> Factor<'a> {
    pub(super) const fn expression(self) -> &'a Expression {
        match self {
            Self::LeftMul(expr) | Self::RightMul(expr) | Self::RightDiv(expr) => expr,
        }
    }

    pub(super) fn apply(
        self,
        scale: super::Expression,
        inner: super::Expression,
    ) -> super::Expression {
        match self {
            Self::LeftMul(_) => super::Expression::Binary("*", Box::new(scale), Box::new(inner)),
            Self::RightMul(_) => super::Expression::Binary("*", Box::new(inner), Box::new(scale)),
            Self::RightDiv(_) => super::Expression::Binary("/", Box::new(inner), Box::new(scale)),
        }
    }
}

/// One additive term produced by splitting a contribution expression, with its sign and
/// any multiplicative or divisive factors ordered innermost first.
pub(super) struct Term<'a> {
    pub(super) negative: bool,
    pub(super) expression: &'a Expression,
    pub(super) factors: Vec<Factor<'a>>,
}

/// Returns whether `expression` contains `ddt(...)`, for subtrees that are never split.
fn has_ddt(expression: &Expression) -> bool {
    match expression {
        Expression::Call(name, arguments) => name == "ddt" || arguments.iter().any(has_ddt),
        Expression::Unary(_, operand) => has_ddt(operand),
        Expression::Binary(_, left, right) => has_ddt(left) || has_ddt(right),
        Expression::Conditional(condition, then, otherwise) => {
            has_ddt(condition) || has_ddt(then) || has_ddt(otherwise)
        }
        _ => false,
    }
}

fn into_terms<'a>(terms: Option<Vec<Term<'a>>>, fallback: &'a Expression) -> Vec<Term<'a>> {
    terms.unwrap_or_else(|| {
        vec![Term {
            negative: false,
            expression: fallback,
            factors: Vec::new(),
        }]
    })
}

fn scale_terms<'a>(
    terms: Option<Vec<Term<'a>>>,
    fallback: &'a Expression,
    factor: Factor<'a>,
) -> (bool, Option<Vec<Term<'a>>>) {
    let mut terms = into_terms(terms, fallback);
    for term in &mut terms {
        term.factors.push(factor);
    }
    (true, Some(terms))
}

/// Computes `(contains_ddt, split_terms)` bottom-up in a single `O(N)` pass.
fn split_node(expression: &Expression) -> (bool, Option<Vec<Term<'_>>>) {
    match expression {
        Expression::Unary("-", operand) => {
            let (ddt, terms) = split_node(operand);
            if !ddt && !matches!(**operand, Expression::Call(..)) {
                return (false, None);
            }
            let mut terms = into_terms(terms, operand);
            for term in &mut terms {
                term.negative = !term.negative;
            }
            (ddt, Some(terms))
        }
        Expression::Binary(operator @ ("+" | "-"), left, right) => {
            let ((left_ddt, left_terms), (right_ddt, right_terms)) =
                (split_node(left), split_node(right));
            let mut terms = into_terms(left_terms, left);
            for mut term in into_terms(right_terms, right) {
                if *operator == "-" {
                    term.negative = !term.negative;
                }
                terms.push(term);
            }
            (left_ddt || right_ddt, Some(terms))
        }
        Expression::Binary("*", left, right) => {
            let ((left_ddt, left_terms), (right_ddt, right_terms)) =
                (split_node(left), split_node(right));
            match (left_ddt, right_ddt) {
                (true, false) => scale_terms(left_terms, left, Factor::RightMul(right)),
                (false, true) => scale_terms(right_terms, right, Factor::LeftMul(left)),
                (both, _) => (both, None),
            }
        }
        Expression::Binary("/", left, right) => {
            let ((left_ddt, left_terms), right_ddt) = (split_node(left), has_ddt(right));
            if left_ddt && !right_ddt {
                scale_terms(left_terms, left, Factor::RightDiv(right))
            } else {
                (left_ddt || right_ddt, None)
            }
        }
        _ => (has_ddt(expression), None),
    }
}

/// Splits `expression` into additive terms, each with whether it is subtracted and any
/// multiplicative or divisive factors wrapped around a `ddt(...)` subexpression.
pub(super) fn split_contribution(expression: &Expression) -> Vec<Term<'_>> {
    into_terms(split_node(expression).1, expression)
}
