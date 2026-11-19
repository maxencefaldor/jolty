//! Compile-time constant folding and static expression typing over resolved IR
//! expressions, shared by IR lowering, linearity analysis, and code emission.

use std::cmp::Ordering;

use crate::ast::{self, Type};

use super::{Builtin, Expression, LocalParameter};

/// A compile-time constant folded from an IR expression.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ConstVal {
    Bool(bool),
    Int(i32),
    Real(f64),
}

impl ConstVal {
    /// Interprets the constant as a Verilog-A boolean (nonzero is `true`).
    pub(crate) fn as_bool(self) -> bool {
        match self {
            Self::Bool(truth) => truth,
            Self::Int(value) => value != 0,
            Self::Real(value) => value != 0.0,
        }
    }

    /// Converts the constant to a 32-bit integer, returning `None` for real values.
    pub(crate) fn as_int(self) -> Option<i32> {
        match self {
            Self::Bool(truth) => Some(i32::from(truth)),
            Self::Int(value) => Some(value),
            Self::Real(_) => None,
        }
    }

    /// Promotes the constant to a finite `f64`.
    pub(crate) fn as_real(self) -> f64 {
        match self {
            Self::Bool(truth) => f64::from(u8::from(truth)),
            Self::Int(value) => f64::from(value),
            Self::Real(value) => value,
        }
    }
}

/// Evaluates binary arithmetic `left operator right` on two `f64` constants.
pub(crate) fn fold(operator: &str, left: f64, right: f64) -> f64 {
    match operator {
        "+" => left + right,
        "-" => left - right,
        "*" => left * right,
        _ => left / right,
    }
}

/// Evaluates a relational comparison `left operator right` on two real numbers.
pub(crate) fn compare_order(operator: &str, left: f64, right: f64) -> bool {
    let order = left.partial_cmp(&right);
    match operator {
        "==" => order == Some(Ordering::Equal),
        "!=" => order != Some(Ordering::Equal),
        "<" => order == Some(Ordering::Less),
        "<=" => matches!(order, Some(Ordering::Less | Ordering::Equal)),
        ">" => order == Some(Ordering::Greater),
        _ => matches!(order, Some(Ordering::Greater | Ordering::Equal)),
    }
}

/// Checks that `value` is finite, returning an error phrase otherwise.
fn finite(value: f64) -> Result<f64, String> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err("evaluates to a non-finite value".to_owned())
    }
}

/// Evaluates a built-in function on constant `f64` arguments.
fn evaluate_builtin(builtin: Builtin, arguments: &[f64]) -> Option<f64> {
    Some(match (builtin, arguments) {
        (Builtin::Exp, &[x]) => x.exp(),
        (Builtin::Expm1, &[x]) => x.exp_m1(),
        (Builtin::Ln, &[x]) => x.ln(),
        (Builtin::Ln1p, &[x]) => x.ln_1p(),
        (Builtin::Log, &[x]) => x.log10(),
        (Builtin::Sqrt, &[x]) => x.sqrt(),
        (Builtin::Pow, &[base, exponent]) => base.powf(exponent),
        (Builtin::Abs, &[x]) => x.abs(),
        (Builtin::Min, &[first, second]) => first.min(second),
        (Builtin::Max, &[first, second]) => first.max(second),
        (Builtin::Sin, &[x]) => x.sin(),
        (Builtin::Cos, &[x]) => x.cos(),
        (Builtin::Tan, &[x]) => x.tan(),
        (Builtin::Sinh, &[x]) => x.sinh(),
        (Builtin::Cosh, &[x]) => x.cosh(),
        (Builtin::Tanh, &[x]) => x.tanh(),
        (Builtin::Asin, &[x]) => x.asin(),
        (Builtin::Acos, &[x]) => x.acos(),
        (Builtin::Atan, &[x]) => x.atan(),
        (Builtin::Floor, &[x]) => x.floor(),
        (Builtin::Ceil, &[x]) => x.ceil(),
        _ => return None,
    })
}

/// Evaluates `expr` as a compile-time constant (`Ok(None)` when non-constant, `Err` on
/// integer overflow, integer divide-by-zero, or a non-finite real result).
pub(crate) fn fold_const(expr: &Expression) -> Result<Option<ConstVal>, String> {
    let overflow = || "overflows an integer".to_owned();
    match expr {
        Expression::Real(value) => Ok(Some(ConstVal::Real(finite(*value)?))),
        Expression::Integer(value) => Ok(Some(ConstVal::Int(*value))),
        Expression::Unary("!", operand) => {
            Ok(fold_const(operand)?.map(|value| ConstVal::Bool(!value.as_bool())))
        }
        Expression::Unary("+", operand) => fold_const(operand),
        Expression::Unary("-", operand) => {
            let Some(value) = fold_const(operand)? else {
                return Ok(None);
            };
            match value.as_int() {
                Some(integer) => Ok(Some(ConstVal::Int(
                    integer.checked_neg().ok_or_else(overflow)?,
                ))),
                None => Ok(Some(ConstVal::Real(finite(-value.as_real())?))),
            }
        }
        Expression::Binary("&&", left, right) => match fold_const(left)? {
            Some(value) if !value.as_bool() => Ok(Some(ConstVal::Bool(false))),
            Some(_) => Ok(fold_const(right)?.map(|value| ConstVal::Bool(value.as_bool()))),
            None => Ok(None),
        },
        Expression::Binary("||", left, right) => match fold_const(left)? {
            Some(value) if value.as_bool() => Ok(Some(ConstVal::Bool(true))),
            Some(_) => Ok(fold_const(right)?.map(|value| ConstVal::Bool(value.as_bool()))),
            None => Ok(None),
        },
        Expression::Binary(operator @ ("<" | "<=" | ">" | ">=" | "==" | "!="), left, right) => {
            let (Some(left), Some(right)) = (fold_const(left)?, fold_const(right)?) else {
                return Ok(None);
            };
            let result = compare_order(operator, left.as_real(), right.as_real());
            Ok(Some(ConstVal::Bool(result)))
        }
        Expression::Binary(operator @ ("+" | "-" | "*" | "/"), left, right) => {
            let (Some(left), Some(right)) = (fold_const(left)?, fold_const(right)?) else {
                return Ok(None);
            };
            if let (Some(left), Some(right)) = (left.as_int(), right.as_int()) {
                if *operator == "/" && right == 0 {
                    return Err("divides by zero".to_owned());
                }
                let result = match *operator {
                    "+" => left.checked_add(right),
                    "-" => left.checked_sub(right),
                    "*" => left.checked_mul(right),
                    _ => left.checked_div(right),
                };
                return Ok(Some(ConstVal::Int(result.ok_or_else(overflow)?)));
            }
            let value = fold(operator, left.as_real(), right.as_real());
            Ok(Some(ConstVal::Real(finite(value)?)))
        }
        Expression::Conditional(condition, then, otherwise) => match fold_const(condition)? {
            Some(value) if value.as_bool() => fold_const(then),
            Some(_) => fold_const(otherwise),
            None => Ok(None),
        },
        Expression::Call(builtin, arguments) => {
            let mut evaluated = Vec::with_capacity(arguments.len());
            for argument in arguments {
                let Some(value) = fold_const(argument)? else {
                    return Ok(None);
                };
                evaluated.push(value.as_real());
            }
            let Some(value) = evaluate_builtin(*builtin, &evaluated) else {
                return Ok(None);
            };
            Ok(Some(ConstVal::Real(finite(value)?)))
        }
        _ => Ok(None),
    }
}

/// Returns the `f64` value of `expr` if it folds to a finite compile-time constant.
pub(crate) fn constant(expr: &Expression) -> Option<f64> {
    fold_const(expr).ok().flatten().map(ConstVal::as_real)
}

/// Infers the Verilog-A numeric [`Type`] of `expr`, treating boolean expressions as
/// [`Type::Integer`].
pub(super) fn expr_type(
    expr: &Expression,
    module: &ast::Module,
    local_parameters: &[LocalParameter],
) -> Type {
    match expr {
        Expression::Integer(_)
        | Expression::Given(_)
        | Expression::Shape(_)
        | Expression::Unary("!", _)
        | Expression::Binary("<" | "<=" | ">" | ">=" | "==" | "!=" | "&&" | "||", ..) => {
            Type::Integer
        }
        Expression::Parameter(index) => module
            .parameters
            .get(*index)
            .map_or(Type::Real, |parameter| parameter.kind),
        Expression::LocalParameter(index) => local_parameters
            .get(*index)
            .map_or(Type::Real, |local| local.kind),
        Expression::Unary(_, operand) => expr_type(operand, module, local_parameters),
        Expression::Binary(_, left, right) | Expression::Conditional(_, left, right) => {
            if expr_type(left, module, local_parameters) == Type::Real
                || expr_type(right, module, local_parameters) == Type::Real
            {
                Type::Real
            } else {
                Type::Integer
            }
        }
        _ => Type::Real,
    }
}
