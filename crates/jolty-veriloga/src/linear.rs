//! Linearity analysis determining when a kernel instance satisfies
//! [`jolty_models::Kernel::linear`], meaning all equations depend linearly on local unknowns
//! with coefficients fixed by parameters and temperature.
//!
//! Each expression is classified into a probe dependence form (`Zero`, `Constant`, `Linear`,
//! `Affine`, or `Other`), together with any static expressions required to vanish for that
//! form to hold. An instance is linear whenever every active contribution can take form `Zero`
//! or `Linear` under its zero-conditions.

use std::collections::BTreeMap;

use crate::ir::{Expression, Module, Probe, Statement, constant};

/// Probe dependence classification of an expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Form {
    Zero,
    Constant,
    /// Sum of constants times probes with no constant term.
    Linear,
    /// Linear combination of probes plus a constant offset.
    Affine,
    Other,
}

/// Reachable [`Form`] variants for an expression, each mapped to the shortest set of static
/// expressions required to vanish.
pub(crate) type TermOptions = BTreeMap<Form, Vec<Expression>>;

/// Constructs [`TermOptions`] holding a single unconditional `form`.
fn only(form: Form) -> TermOptions {
    TermOptions::from([(form, Vec::new())])
}

/// Combines every pair of options from `left` and `right` via `combine`, keeping the shortest
/// zero-condition list per resulting [`Form`].
fn product(
    left: &TermOptions,
    right: &TermOptions,
    combine: impl Fn(Form, Form) -> Form,
) -> TermOptions {
    let mut options = TermOptions::new();
    for (&first, first_zero) in left {
        for (&second, second_zero) in right {
            let mut zero = first_zero.clone();
            for expr in second_zero {
                if !zero.contains(expr) {
                    zero.push(expr.clone());
                }
            }
            let form = combine(first, second);
            let fewer = options
                .get(&form)
                .is_none_or(|found| zero.len() < found.len());
            if fewer {
                options.insert(form, zero);
            }
        }
    }
    options
}

/// Returns the combined [`Form`] of a sum or difference of `left` and `right`.
fn sum(left: Form, right: Form) -> Form {
    match (left, right) {
        (Form::Other, _) | (_, Form::Other) => Form::Other,
        (Form::Zero, form) | (form, Form::Zero) => form,
        (Form::Constant, Form::Constant) => Form::Constant,
        (Form::Linear, Form::Linear) => Form::Linear,
        _ => Form::Affine,
    }
}

/// Returns the combined [`Form`] of a product of `left` and `right`.
fn times(left: Form, right: Form) -> Form {
    match (left, right) {
        (Form::Zero, _) | (_, Form::Zero) => Form::Zero,
        (Form::Constant, form) | (form, Form::Constant) => form,
        _ => Form::Other,
    }
}

/// Returns the joined [`Form`] across assignments or branches yielding `left` and `right`.
fn either(left: Form, right: Form) -> Form {
    sum(left, right)
}

/// Fixed-point probe-dependence options for every dynamic variable in a module.
pub(crate) struct Linearity<'m> {
    module: &'m Module,
    variables: Vec<TermOptions>,
}

impl<'m> Linearity<'m> {
    /// Computes the fixed-point [`Form`] options for every dynamic variable in `module`,
    /// returning `None` if iteration does not converge.
    pub(crate) fn new(module: &'m Module) -> Option<Self> {
        let mut linearity = Self {
            module,
            variables: vec![only(Form::Zero); module.variables.len()],
        };
        // Bound iteration count by variables times lattice height (five forms).
        let rounds = (module.variables.len() + 1) * 5;
        for _ in 0..rounds {
            let before = linearity.variables.clone();
            linearity.flow(&module.analog, false);
            if linearity.variables == before {
                return Some(linearity);
            }
        }
        None
    }

    fn flow(&mut self, statements: &[Statement], dynamic: bool) {
        for statement in statements {
            match statement {
                Statement::Assign(variable, value) if self.module.dynamic[*variable] => {
                    let value = if dynamic {
                        only(Form::Other)
                    } else {
                        self.options(value)
                    };
                    let joined = product(&self.variables[*variable], &value, either);
                    self.variables[*variable] = joined;
                }
                Statement::If(_, condition, then, otherwise) => {
                    let dynamic = dynamic || self.module.is_dynamic(condition);
                    self.flow(then, dynamic);
                    self.flow(otherwise, dynamic);
                }
                Statement::While(condition, body) => {
                    self.flow(body, dynamic || self.module.is_dynamic(condition));
                }
                Statement::Assign(..)
                | Statement::Contribute { .. }
                | Statement::Fail(..)
                | Statement::Warn(..) => {}
            }
        }
    }

    /// Returns the reachable [`Form`] options for `expr`.
    pub(crate) fn options(&self, expr: &Expression) -> TermOptions {
        if !self.module.is_dynamic(expr) {
            return match constant(expr) {
                Some(value) => only(if value == 0.0 {
                    Form::Zero
                } else {
                    Form::Constant
                }),
                None => TermOptions::from([
                    (Form::Constant, Vec::new()),
                    (Form::Zero, vec![expr.clone()]),
                ]),
            };
        }
        match expr {
            Expression::Probe(Probe::Voltage(..) | Probe::Current(_)) => only(Form::Linear),
            Expression::Variable(variable) => self.variables[*variable].clone(),
            Expression::Unary("-", operand) => self.options(operand),
            Expression::Binary("+" | "-", left, right) => {
                product(&self.options(left), &self.options(right), sum)
            }
            Expression::Binary("*", left, right) => {
                product(&self.options(left), &self.options(right), times)
            }
            // A quotient preserves its numerator form when the divisor is constant.
            Expression::Binary("/", left, right) => product(
                &self.options(left),
                &self.options(right),
                |numerator, divisor| {
                    if divisor == Form::Constant {
                        numerator
                    } else {
                        Form::Other
                    }
                },
            ),
            // A shape-guarded conditional joins the forms of its two branches.
            Expression::Conditional(condition, then, otherwise) if shaped(condition) => {
                product(&self.options(then), &self.options(otherwise), either)
            }
            // A function of constants stays constant; any dynamic argument yields `Other`.
            Expression::Call(_, arguments) => {
                arguments
                    .iter()
                    .fold(only(Form::Constant), |options, argument| {
                        product(&options, &self.options(argument), |first, second| {
                            if first <= Form::Constant && second <= Form::Constant {
                                Form::Constant
                            } else {
                                Form::Other
                            }
                        })
                    })
            }
            // Limited probes read memory, gmin reads simulator state, and remaining expressions
            // are non-real.
            _ => only(Form::Other),
        }
    }
}

/// Returns `true` when `condition` depends solely on kernel shape predicates.
fn shaped(condition: &Expression) -> bool {
    match condition {
        Expression::Shape(_) => true,
        Expression::Unary("!", operand) => shaped(operand),
        Expression::Binary("&&" | "||", left, right) => shaped(left) && shaped(right),
        _ => false,
    }
}
