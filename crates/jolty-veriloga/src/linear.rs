//! Which instances of a kernel are linear: whose equations are their entries times their
//! local unknowns, entries that the parameters and the temperature alone set, as
//! [`jolty_models::Kernel::linear`] asks.
//!
//! Each expression the evaluation computes has a form in the probes: zero, a constant, a
//! linear one with no constant term, an affine one, or any other. A form may hold only
//! when some static numbers are zero, as `vc · v² + 1` is a constant only when `vc` is:
//! each expression has the forms it can take, each with the static expressions that must
//! be zero for it. An instance is linear when each contribution its shape makes can be
//! zero or linear, the static expressions it requires zero.

use std::collections::BTreeMap;

use crate::ir::{Expr, Module, Probe, Stmt, constant};

/// How an expression depends on the probes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Form {
    Zero,
    Constant,
    /// A sum of constants times probes, with no constant term.
    Linear,
    /// A linear form plus a constant.
    Affine,
    Other,
}

/// The forms an expression can take, each with the static expressions that must be zero
/// for it, the fewest for each form.
pub(crate) type Options = BTreeMap<Form, Vec<Expr>>;

/// The options of an expression that is `form` whatever the static numbers are.
fn only(form: Form) -> Options {
    Options::from([(form, Vec::new())])
}

/// The options of each pair of an option of `left` and one of `right`, combined by
/// `combine`, each requiring what both do.
fn product(left: &Options, right: &Options, combine: impl Fn(Form, Form) -> Form) -> Options {
    let mut options = Options::new();
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

/// The form of a sum or a difference of `left` and `right`.
fn sum(left: Form, right: Form) -> Form {
    match (left, right) {
        (Form::Other, _) | (_, Form::Other) => Form::Other,
        (Form::Zero, form) | (form, Form::Zero) => form,
        (Form::Constant, Form::Constant) => Form::Constant,
        (Form::Linear, Form::Linear) => Form::Linear,
        _ => Form::Affine,
    }
}

/// The form of a product of `left` and `right`.
fn times(left: Form, right: Form) -> Form {
    match (left, right) {
        (Form::Zero, _) | (_, Form::Zero) => Form::Zero,
        (Form::Constant, form) | (form, Form::Constant) => form,
        _ => Form::Other,
    }
}

/// The form a variable has when its assignments give it `left` and `right`.
fn either(left: Form, right: Form) -> Form {
    sum(left, right)
}

/// The forms of each dynamic variable of a module, found from its assignments.
pub(crate) struct Linearity<'m> {
    module: &'m Module,
    variables: Vec<Options>,
}

impl<'m> Linearity<'m> {
    /// The forms of `module`'s variables, each the join of its assignments' and zero,
    /// where Verilog-A's variables start; `None` when they do not settle within as many
    /// rounds as the forms' order allows each variable to rise.
    pub(crate) fn new(module: &'m Module) -> Option<Self> {
        let mut linearity = Self {
            module,
            variables: vec![only(Form::Zero); module.variables.len()],
        };
        // Each round raises a variable's form or adds what it requires, at most once per
        // form and per static expression it can require; a bound of the rounds that many.
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

    fn flow(&mut self, statements: &[Stmt], dynamic: bool) {
        for statement in statements {
            match statement {
                Stmt::Assign(variable, value) if self.module.dynamic[*variable] => {
                    let value = if dynamic {
                        only(Form::Other)
                    } else {
                        self.options(value)
                    };
                    let joined = product(&self.variables[*variable], &value, either);
                    self.variables[*variable] = joined;
                }
                Stmt::If(_, condition, then, otherwise) => {
                    let dynamic = dynamic || self.module.is_dynamic(condition);
                    self.flow(then, dynamic);
                    self.flow(otherwise, dynamic);
                }
                Stmt::While(condition, body) => {
                    self.flow(body, dynamic || self.module.is_dynamic(condition));
                }
                Stmt::Assign(..) | Stmt::Contribute { .. } | Stmt::Fail(..) | Stmt::Warn(..) => {}
            }
        }
    }

    /// The forms `expr` can take.
    pub(crate) fn options(&self, expr: &Expr) -> Options {
        if !self.module.is_dynamic(expr) {
            return match constant(expr) {
                Some(value) => only(if value == 0.0 {
                    Form::Zero
                } else {
                    Form::Constant
                }),
                None => Options::from([
                    (Form::Constant, Vec::new()),
                    (Form::Zero, vec![expr.clone()]),
                ]),
            };
        }
        match expr {
            Expr::Probe(Probe::Voltage(..) | Probe::Current(_)) => only(Form::Linear),
            Expr::Variable(variable) => self.variables[*variable].clone(),
            Expr::Unary("-", operand) => self.options(operand),
            Expr::Binary("+" | "-", left, right) => {
                product(&self.options(left), &self.options(right), sum)
            }
            Expr::Binary("*", left, right) => {
                product(&self.options(left), &self.options(right), times)
            }
            // A quotient keeps its numerator's form where its divisor is a constant.
            Expr::Binary("/", left, right) => product(
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
            // A choice the shape makes is one of its two forms.
            Expr::Conditional(condition, then, otherwise) if shaped(condition) => {
                product(&self.options(then), &self.options(otherwise), either)
            }
            // A function of constants is a constant, of anything else any form.
            Expr::Call(_, arguments) => {
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
            // A limited probe reads the memory, gmin the simulator's state, and the rest
            // are no real numbers.
            _ => only(Form::Other),
        }
    }
}

/// Whether `condition` reads the shape's conditions alone.
fn shaped(condition: &Expr) -> bool {
    match condition {
        Expr::Shape(_) => true,
        Expr::Unary("!", operand) => shaped(operand),
        Expr::Binary("&&" | "||", left, right) => shaped(left) && shaped(right),
        _ => false,
    }
}
