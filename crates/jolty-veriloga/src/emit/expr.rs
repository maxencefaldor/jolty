//! Expression lowering for the Verilog-A emitter: static constant folding and hoisting,
//! type conversions via [`jolty_models::Dual::embed`], arithmetic, comparisons, probes,
//! `ddx`, and smooth transcendental functions.

use std::collections::{BTreeMap, BTreeSet};

use crate::Error;
use crate::ast::Type;
use crate::ir::fold::{ConstVal, compare_order, fold_const};
use crate::ir::{Builtin, Expr, Limiter, Module, Probe, Stmt, fold};

use super::facts::Facts;
use super::liveness::{live, prune};
use super::stamp::{positive, print};
use super::types::{
    AND, ATOM, COMPARE, CodeWriter, Kind, Mode, OR, Operand, PRODUCT, Place, Prepared, Rust, SUM,
    Site, UNARY, float, join, parenthesize, rust_type, union,
};

/// A smooth function of one real number, as the emitter writes it: its plain floats'
/// method, its value, slope and curvature at the float `{a}`, and its slope as one of
/// jolty's numbers, of the argument `{x}` and the value `{r}`.
struct Smooth {
    method: &'static str,
    plain: [&'static str; 3],
    slope: &'static str,
}

/// The smooth function `builtin` is, if it is one that jolty's numbers compute through
/// their chain rule.
#[expect(
    clippy::too_many_lines,
    reason = "one table, a row for each smooth function"
)]
const fn smooth(builtin: Builtin) -> Option<Smooth> {
    let (method, plain, slope) = match builtin {
        Builtin::Expm1 => (
            "exp_m1",
            ["{a}.exp_m1()", "{a}.exp()", "{a}.exp()"],
            "{x}.exp()",
        ),
        Builtin::Ln1p => (
            "ln_1p",
            [
                "{a}.ln_1p()",
                "({a} + 1.0).recip()",
                "-(({a} + 1.0) * ({a} + 1.0)).recip()",
            ],
            "({x} + 1.0).recip()",
        ),
        Builtin::Log => (
            "log10",
            [
                "{a}.log10()",
                "({a} * std::f64::consts::LN_10).recip()",
                "-({a} * {a} * std::f64::consts::LN_10).recip()",
            ],
            "({x} * std::f64::consts::LN_10).recip()",
        ),
        Builtin::Sin => (
            "sin",
            ["{a}.sin()", "{a}.cos()", "-{a}.sin()"],
            "{x}.chain(({a}.cos(), -{a}.sin(), -{a}.cos()))",
        ),
        Builtin::Cos => (
            "cos",
            ["{a}.cos()", "-{a}.sin()", "-{a}.cos()"],
            "{x}.chain((-{a}.sin(), -{a}.cos(), {a}.sin()))",
        ),
        Builtin::Tan => (
            "tan",
            [
                "{a}.tan()",
                "{a}.tan() * {a}.tan() + 1.0",
                "2.0 * {a}.tan() * ({a}.tan() * {a}.tan() + 1.0)",
            ],
            "{r} * {r} + 1.0",
        ),
        Builtin::Sinh => (
            "sinh",
            ["{a}.sinh()", "{a}.cosh()", "{a}.sinh()"],
            "{x}.chain(({a}.cosh(), {a}.sinh(), {a}.cosh()))",
        ),
        Builtin::Cosh => (
            "cosh",
            ["{a}.cosh()", "{a}.sinh()", "{a}.cosh()"],
            "{x}.chain(({a}.sinh(), {a}.cosh(), {a}.sinh()))",
        ),
        Builtin::Tanh => (
            "tanh",
            [
                "{a}.tanh()",
                "1.0 - {a}.tanh() * {a}.tanh()",
                "-2.0 * {a}.tanh() * (1.0 - {a}.tanh() * {a}.tanh())",
            ],
            "R::constant(1.0) - {r} * {r}",
        ),
        Builtin::Asin => (
            "asin",
            [
                "{a}.asin()",
                "(1.0 - {a} * {a}).sqrt().recip()",
                "{a} * (1.0 - {a} * {a}).powf(-1.5)",
            ],
            "(R::constant(1.0) - {x} * {x}).sqrt().recip()",
        ),
        Builtin::Acos => (
            "acos",
            [
                "{a}.acos()",
                "-(1.0 - {a} * {a}).sqrt().recip()",
                "-{a} * (1.0 - {a} * {a}).powf(-1.5)",
            ],
            "-(R::constant(1.0) - {x} * {x}).sqrt().recip()",
        ),
        Builtin::Atan => (
            "atan",
            [
                "{a}.atan()",
                "({a} * {a} + 1.0).recip()",
                "-2.0 * {a} * ({a} * {a} + 1.0).powi(-2)",
            ],
            "({x} * {x} + 1.0).recip()",
        ),
        // A step function, whose slope is zero wherever it has one.
        Builtin::Floor => ("floor", ["{a}.floor()", "0.0", "0.0"], "R::constant(0.0)"),
        Builtin::Ceil => ("ceil", ["{a}.ceil()", "0.0", "0.0"], "R::constant(0.0)"),
        Builtin::Exp
        | Builtin::Ln
        | Builtin::Sqrt
        | Builtin::Pow
        | Builtin::Abs
        | Builtin::Min
        | Builtin::Max => {
            return None;
        }
    };
    Some(Smooth {
        method,
        plain,
        slope,
    })
}

/// The emitter of one function of the kernel.
pub(super) struct Emitter<'f, 'm> {
    pub(super) facts: &'f Facts<'m>,
    pub(super) mode: Mode,
    /// The analog block as this mode runs it: each assignment that no statement after it
    /// reads left out.
    pub(super) analog: Vec<Stmt>,
    /// Whether each variable is kept, in this mode.
    pub(super) live: Vec<bool>,
    /// The function being emitted, in [`Mode::Function`].
    pub(super) function: Option<usize>,
    temporaries: usize,
    pub(super) prepared: &'f mut Prepared,
    /// The probes read, which the function binds first.
    pub(super) probes: BTreeSet<Probe>,
    /// The fields of the shape read.
    pub(super) shape: BTreeSet<String>,
    /// The entries of the stamp, which the shape holds, each by its row and column.
    pub(super) entries: &'f mut BTreeMap<(Place, Place), BTreeSet<(usize, Site)>>,
    /// The limited probes whose applied and limited voltages the stamp extends along.
    pub(super) extended: BTreeSet<usize>,
    /// The parameters read, in preparation.
    pub(super) parameters_read: BTreeSet<usize>,
    /// The non-constant `localparam`s read, in preparation.
    pub(super) locals_read: BTreeSet<usize>,
    /// Whether `$temperature` was read, in preparation.
    pub(super) uses_temperature: bool,
    /// The kind of each branch's accumulated flow or potential, and of its charge or
    /// flux, by the branch and whether it is the latter.
    pub(super) accumulators: BTreeMap<(usize, Site, bool), Kind>,
}

impl<'f, 'm> Emitter<'f, 'm> {
    pub(super) fn new(
        facts: &'f Facts<'m>,
        mode: Mode,
        prepared: &'f mut Prepared,
        entries: &'f mut BTreeMap<(Place, Place), BTreeSet<(usize, Site)>>,
    ) -> Self {
        let module = facts.module;
        let roots: BTreeSet<usize> = if mode == Mode::Evaluate {
            module.outputs.iter().copied().collect()
        } else {
            BTreeSet::new()
        };
        let (analog, live) = if matches!(mode, Mode::Evaluate | Mode::Noise) {
            let analog = prune(&module.analog, module, mode, &mut roots.clone());
            let live = live(module, &analog, mode, &roots);
            (analog, live)
        } else {
            (Vec::new(), vec![false; module.variables.len()])
        };
        Self {
            facts,
            mode,
            analog,
            live,
            function: None,
            temporaries: 0,
            prepared,
            probes: BTreeSet::new(),
            shape: BTreeSet::new(),
            entries,
            extended: BTreeSet::new(),
            parameters_read: BTreeSet::new(),
            locals_read: BTreeSet::new(),
            uses_temperature: false,
            accumulators: BTreeMap::new(),
        }
    }
}

/// Returns the positive counterpart of `expr` when `expr` is a strictly negative numeric
/// literal (excluding `-0.0`, `NaN`, and `i32::MIN`).
fn negative_literal(expr: &Expr) -> Option<Expr> {
    match expr {
        Expr::Real(value) if *value < 0.0 => Some(Expr::Real(-*value)),
        Expr::Integer(value) if *value < 0 && *value != i32::MIN => Some(Expr::Integer(-*value)),
        Expr::Unary("-", inner) => match inner.as_ref() {
            Expr::Real(value) if *value > 0.0 => Some(Expr::Real(*value)),
            Expr::Integer(value) if *value > 0 => Some(Expr::Integer(*value)),
            _ => None,
        },
        _ => None,
    }
}

/// Returns whether `expr` is a numeric literal factor for canonicalizing `literal * expr` to
/// `expr * literal`.
const fn is_literal_factor(expr: &Expr) -> bool {
    matches!(expr, Expr::Real(_) | Expr::Integer(_))
}

/// Whether `code` names a value, so that reading it again costs nothing.
fn is_name(code: &str) -> bool {
    !code.is_empty()
        && code
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '.')
        && !code.starts_with(|c: char| c.is_ascii_digit())
}

impl Emitter<'_, '_> {
    pub(super) fn module(&self) -> &Module {
        self.facts.module
    }

    /// A new temporary's name, numbered in four digits, so that no two differ by a digit
    /// more, which clippy's `similar_names` reads as a slip.
    pub(super) fn temporary(&mut self) -> String {
        let name = format!("t{:04}", self.temporaries);
        self.temporaries += 1;
        name
    }

    /// `operand` bound to a temporary, unless it names a value already.
    fn bind(&mut self, operand: Operand, out: &mut Vec<Rust>) -> Operand {
        if is_name(&operand.code) {
            return operand;
        }
        let name = self.temporary();
        // A dual number made from a constant takes its size from where it is used, which
        // a comparison of its value does not tell, so its type is written.
        let typed = if matches!(operand.kind, Kind::Dual(_)) && operand.code.starts_with("Dual::") {
            format!(": {}", rust_type(&operand.kind))
        } else {
            String::new()
        };
        out.push(Rust::Line(format!("let {name}{typed} = {};", operand.code)));
        Operand::new(name, ATOM, operand.kind)
    }

    /// The `Locals` field name for `probe`, or `None` for limited probes (which bind inline).
    pub(super) fn probe_field(&self, probe: Probe) -> Option<String> {
        let module = self.facts.module;
        match probe {
            Probe::Voltage(plus, minus) => Some(format!(
                "voltage_{}_{}",
                module.node_name(plus),
                module.node_name(minus)
            )),
            Probe::Current(branch) => Some(format!("current_{}", module.branches[branch].name)),
            Probe::Limited(_) => None,
        }
    }

    /// The kind of the variable at `variable`, in this mode.
    pub(super) fn variable_kind(&self, variable: usize) -> Kind {
        if let Some(function) = self.function {
            return Kind::from_type(self.module().functions[function].variables[variable].1, true);
        }
        let module = self.module();
        match module.variables[variable].1 {
            Type::Integer => Kind::Integer,
            Type::Real
                if self.mode == Mode::Evaluate
                    && module.dynamic[variable]
                    && !module.probes[variable].is_empty() =>
            {
                Kind::Dual(module.probes[variable].iter().copied().collect())
            }
            Type::Real => Kind::Real,
        }
    }

    /// The emitted kind of a non-constant static expression `expr`.
    fn static_kind(&self, expr: &Expr) -> Kind {
        let module = self.module();
        match expr {
            Expr::Real(_) | Expr::Gmin | Expr::SmallResistance | Expr::Probe(_) => {
                Kind::Float(None)
            }
            Expr::Integer(_) => Kind::Integer,
            Expr::Parameter(parameter) => {
                Kind::from_type(module.parameters[*parameter].kind, false)
            }
            Expr::LocalParameter(local) => {
                Kind::from_type(module.local_parameters[*local].kind, false)
            }
            Expr::Variable(variable) => Kind::from_type(module.variables[*variable].1, false),
            Expr::Temperature | Expr::Ddx(..) => Kind::Real,
            Expr::Given(_)
            | Expr::Shape(_)
            | Expr::Unary("!", _)
            | Expr::Binary("<" | "<=" | ">" | ">=" | "==" | "!=" | "&&" | "||", ..) => Kind::Bool,
            Expr::Unary(_, operand) => self.static_kind(operand),
            Expr::Binary(_, left, right) => {
                join(&self.static_kind(left), &self.static_kind(right))
            }
            Expr::Conditional(_, then, otherwise) => {
                join(&self.static_kind(then), &self.static_kind(otherwise))
            }
            Expr::Call(_, arguments) => {
                if arguments
                    .iter()
                    .all(|argument| matches!(self.static_kind(argument), Kind::Float(_)))
                {
                    Kind::Float(None)
                } else {
                    Kind::Real
                }
            }
        }
    }

    /// A static expression read in a dynamic mode: a constant folded in place, a
    /// constant of jolty's, or a prepared number.
    pub(super) fn static_operand(&mut self, expr: &Expr) -> Operand {
        if let Ok(Some(value)) = fold_const(expr) {
            return match value {
                ConstVal::Real(number) => float(number),
                ConstVal::Int(number) => Operand::new(
                    number.to_string(),
                    if number < 0 { UNARY } else { ATOM },
                    Kind::Integer,
                ),
                ConstVal::Bool(truth) => Operand::new(truth.to_string(), ATOM, Kind::Bool),
            };
        }
        let kind = self.static_kind(expr);
        match expr {
            Expr::SmallResistance => {
                return Operand::new("SMALL_RESISTANCE", ATOM, Kind::Float(None));
            }
            Expr::Unary("-", inner)
                if matches!(
                    self.static_kind(inner),
                    Kind::Real | Kind::Float(_) | Kind::Integer
                ) =>
            {
                return negate(self.static_operand(inner));
            }
            Expr::Binary("*", left, right)
                if matches!(kind, Kind::Real | Kind::Float(_) | Kind::Integer) =>
            {
                if let Some(positive) = negative_literal(left) {
                    let product = Expr::Binary("*", Box::new(positive), right.clone());
                    return negate(self.static_operand(&product));
                }
                if let Some(positive) = negative_literal(right) {
                    let product = Expr::Binary("*", left.clone(), Box::new(positive));
                    return negate(self.static_operand(&product));
                }
                if is_literal_factor(left) && !is_literal_factor(right) {
                    let swapped = Expr::Binary("*", right.clone(), left.clone());
                    return self.static_operand(&swapped);
                }
                if matches!(kind, Kind::Real | Kind::Float(_))
                    && let Expr::Integer(value) = right.as_ref()
                {
                    let normalized =
                        Expr::Binary("*", left.clone(), Box::new(Expr::Real(f64::from(*value))));
                    return self.static_operand(&normalized);
                }
            }
            _ => {}
        }
        let module = self.facts.module;
        let name = match expr {
            Expr::Parameter(parameter) => {
                let name = module.parameters[*parameter].name.clone();
                self.prepared.field(|_| name, kind.clone(), expr)
            }
            Expr::LocalParameter(local) => {
                let name = module.local_parameters[*local].name.clone();
                self.prepared.field(|_| name, kind.clone(), expr)
            }
            Expr::Given(parameter) => {
                let name = format!("{}_given", module.parameters[*parameter].name);
                self.prepared.field(|_| name, kind.clone(), expr)
            }
            Expr::Variable(variable) => {
                let name = module.variables[*variable].0.clone();
                self.prepared.field(|_| name, kind.clone(), expr)
            }
            Expr::Temperature => {
                self.prepared
                    .field(|_| "temperature".to_owned(), kind.clone(), expr)
            }
            _ => self
                .prepared
                .field(|index| format!("h{index}"), kind.clone(), expr),
        };
        Operand::new(format!("prepared.{name}"), ATOM, kind)
    }

    /// The value of `operand` as a plain float.
    pub(super) fn value_of(operand: &Operand) -> Operand {
        match operand.kind {
            Kind::Float(_) => operand.clone(),
            Kind::Integer => match operand.code.parse::<i32>() {
                Ok(value) => float(f64::from(value)),
                Err(_) => Operand::new(
                    format!("f64::from({})", operand.code),
                    ATOM,
                    Kind::Float(None),
                ),
            },
            Kind::Real => {
                if let Some(inner) = operand.code.strip_prefix('-')
                    && operand.precedence == UNARY
                {
                    Operand::new(format!("-{inner}.value()"), UNARY, Kind::Float(None))
                } else {
                    Operand::new(
                        format!("{}.value()", operand.at(ATOM)),
                        ATOM,
                        Kind::Float(None),
                    )
                }
            }
            Kind::Dual(_) => Operand::new(
                format!("{}.value.value()", operand.at(ATOM)),
                ATOM,
                Kind::Float(None),
            ),
            Kind::Bool => Operand::new(
                format!("f64::from(u8::from({}))", operand.code),
                ATOM,
                Kind::Float(None),
            ),
        }
    }

    /// `operand` as a truth value, Verilog-A's: true when it is not zero.
    pub(super) fn truth(&self, operand: Operand) -> (String, u8) {
        let _ = self;
        match operand.kind {
            Kind::Bool => (operand.code, operand.precedence),
            Kind::Integer if operand.code == "0" => ("false".to_owned(), ATOM),
            Kind::Integer => (format!("{} != 0", operand.at(SUM)), COMPARE),
            Kind::Float(Some(0.0)) => ("false".to_owned(), ATOM),
            _ => (
                format!("{} != 0.0", Self::value_of(&operand).at(SUM)),
                COMPARE,
            ),
        }
    }

    /// `operand` converted to `kind`.
    pub(super) fn convert(
        &mut self,
        operand: Operand,
        kind: &Kind,
        out: &mut Vec<Rust>,
    ) -> Result<Operand, Error> {
        let unsupported = || Error::at(0, format!("a {:?} cannot become a {kind:?}", operand.kind));
        Ok(match (&operand.kind, kind) {
            (from, to) if from == to => operand,
            (Kind::Float(_), Kind::Float(None)) => operand,
            (Kind::Bool, Kind::Integer) => {
                Operand::new(format!("i32::from({})", operand.code), ATOM, Kind::Integer)
            }
            (Kind::Bool, Kind::Float(_) | Kind::Real | Kind::Dual(_)) => {
                let floated = Self::value_of(&operand);
                return self.convert(floated, kind, out);
            }
            (Kind::Integer, Kind::Float(_) | Kind::Real | Kind::Dual(_)) => {
                let floated = match operand.code.parse::<i32>() {
                    Ok(value) => float(f64::from(value)),
                    Err(_) => Self::value_of(&operand),
                };
                return self.convert(floated, kind, out);
            }
            (Kind::Float(_), Kind::Real) => {
                Operand::new(format!("R::constant({})", operand.code), ATOM, Kind::Real)
            }
            (Kind::Float(_), Kind::Dual(_)) => Operand::new(
                format!("Dual::number({})", operand.code),
                ATOM,
                kind.clone(),
            ),
            (Kind::Real, Kind::Dual(_)) => Operand::new(
                format!("Dual::constant({})", operand.code),
                ATOM,
                kind.clone(),
            ),
            (Kind::Dual(from), Kind::Dual(to)) => {
                let Some(positions) = from
                    .iter()
                    .map(|probe| {
                        to.iter()
                            .position(|target| target == probe)
                            .map(|index| index.to_string())
                    })
                    .collect::<Option<Vec<String>>>()
                else {
                    return Err(unsupported());
                };
                if operand.code.starts_with("Dual::number(")
                    || operand.code.starts_with("Dual::constant(")
                {
                    return Ok(Operand::new(operand.code, ATOM, kind.clone()));
                }
                Operand::new(
                    format!(
                        "{}.embed::<{}>([{}])",
                        operand.at(ATOM),
                        to.len(),
                        positions.join(", ")
                    ),
                    ATOM,
                    kind.clone(),
                )
            }
            _ => return Err(unsupported()),
        })
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one arm per construct of the language, which read best side by side"
    )]
    pub(super) fn expression(
        &mut self,
        expr: &Expr,
        out: &mut Vec<Rust>,
    ) -> Result<Operand, Error> {
        let module = self.facts.module;
        if matches!(self.mode, Mode::Evaluate | Mode::Noise) && !module.is_dynamic(expr) {
            return Ok(self.static_operand(expr));
        }
        Ok(match expr {
            Expr::Real(value) => float(*value),
            Expr::Integer(value) => Operand::new(
                value.to_string(),
                if *value < 0 { UNARY } else { ATOM },
                Kind::Integer,
            ),
            // A parameter is read where it is used: given, or else at its default.
            Expr::Parameter(parameter) => {
                self.parameters_read.insert(*parameter);
                let parameter = &module.parameters[*parameter];
                let kind = Kind::from_type(parameter.kind, false);
                let code = if parameter.optional {
                    let default = self.expression(&parameter.default, out)?;
                    let default = self.convert(default, &kind, out)?;
                    format!("parameters.{}.unwrap_or({})", parameter.name, default.code)
                } else {
                    format!("parameters.{}", parameter.name)
                };
                Operand::new(code, ATOM, kind)
            }
            Expr::LocalParameter(local) => {
                if self.mode == Mode::Function {
                    let external = module
                        .local_parameters
                        .get(*local)
                        .map_or("unknown", |entry| entry.external.as_str());
                    return Err(Error::at(
                        0,
                        format!(
                            "the limiting function reads localparam {external}, which depends on parameters and is not compiled yet"
                        ),
                    ));
                }
                self.locals_read.insert(*local);
                let local = &module.local_parameters[*local];
                Operand::new(
                    format!("c.{}", local.name),
                    ATOM,
                    Kind::from_type(local.kind, false),
                )
            }
            Expr::Given(parameter) => {
                self.parameters_read.insert(*parameter);
                Operand::new(
                    format!(
                        "parameters.{}.is_some()",
                        module.parameters[*parameter].name
                    ),
                    ATOM,
                    Kind::Bool,
                )
            }
            Expr::Variable(variable) => {
                let kind = self.variable_kind(*variable);
                match self.function {
                    Some(function) if *variable < module.functions[function].inputs => {
                        Operand::new(
                            module.functions[function].variables[*variable].0.clone(),
                            ATOM,
                            kind,
                        )
                    }
                    Some(function) => Operand::new(
                        format!("v.{}", module.functions[function].variables[*variable].0),
                        ATOM,
                        kind,
                    ),
                    None => {
                        Operand::new(format!("v.{}", module.variables[*variable].0), ATOM, kind)
                    }
                }
            }
            Expr::Temperature => {
                self.uses_temperature = true;
                Operand::new("temperature", ATOM, Kind::Real)
            }
            Expr::Gmin => Operand::new("at.gmin", ATOM, Kind::Float(None)),
            Expr::SmallResistance => Operand::new("SMALL_RESISTANCE", ATOM, Kind::Float(None)),
            Expr::Probe(probe) => self.probe(*probe, out)?,
            Expr::Ddx(operand, node) => self.ddx(operand, *node, out)?,
            Expr::Shape(id) => {
                let index = self
                    .facts
                    .shaped
                    .get(id)
                    .ok_or_else(|| Error::at(0, "a condition that shapes no kernel".into()))?;
                self.shape.insert("conditions".to_owned());
                Operand::new(format!("shape.conditions[{index}]"), ATOM, Kind::Bool)
            }
            Expr::Unary("-", operand) => {
                let operand = self.expression(operand, out)?;
                negate(operand)
            }
            Expr::Unary(_, operand) => self.negation(operand, out)?,
            Expr::Binary(operator @ ("&&" | "||"), left, right) => {
                let left = self.expression(left, out)?;
                let (left_code, left_precedence) = self.truth(left);
                let (absorb, identity) = if *operator == "&&" {
                    ("false", "true")
                } else {
                    ("true", "false")
                };
                if left_code == absorb {
                    return Ok(Operand::new(absorb, ATOM, Kind::Bool));
                }
                if left_code == identity {
                    let right = self.expression(right, out)?;
                    let (right_code, right_precedence) = self.truth(right);
                    return Ok(Operand::new(right_code, right_precedence, Kind::Bool));
                }
                let mut right_out = Vec::new();
                let right = self.expression(right, &mut right_out)?;
                let (right_code, right_precedence) = self.truth(right);
                if right_out.is_empty() && right_code == identity {
                    return Ok(Operand::new(left_code, left_precedence, Kind::Bool));
                }
                out.extend(right_out);
                let least = if *operator == "&&" { AND } else { OR };
                let left = parenthesize(&left_code, left_precedence, least);
                let right = parenthesize(&right_code, right_precedence, least + 1);
                Operand::new(format!("{left} {operator} {right}"), least, Kind::Bool)
            }
            Expr::Binary(operator @ ("<" | "<=" | ">" | ">=" | "==" | "!="), left, right) => {
                let left = self.expression(left, out)?;
                let right = self.expression(right, out)?;
                compare(operator, &left, &right)
            }
            Expr::Binary(operator, left, right) => {
                let left = self.expression(left, out)?;
                // A dual number divided by a number the probes do not move: its value
                // divided, as the source writes it, and its derivatives multiplied by the
                // reciprocal, which the preparation computes once, a product for each
                // quotient.
                if *operator == "/"
                    && matches!(left.kind, Kind::Dual(_))
                    && !module.is_dynamic(right)
                {
                    let x = self.bind(left, out);
                    let divisor = self.static_operand(right);
                    let divisor = if divisor.kind == Kind::Integer {
                        self.convert(divisor, &Kind::Float(None), out)?
                    } else {
                        divisor
                    };
                    let reciprocal = Expr::Binary("/", Box::new(Expr::Real(1.0)), right.clone());
                    let reciprocal = self.static_operand(&reciprocal);
                    let reciprocal = self.convert(reciprocal, &Kind::Real, out)?;
                    return Ok(Operand::new(
                        format!(
                            "{0}.chain(({0}.value / {1}, {2}))",
                            x.code,
                            divisor.at(PRODUCT + 1),
                            reciprocal.code
                        ),
                        ATOM,
                        x.kind,
                    ));
                }
                let right = self.expression(right, out)?;
                self.arithmetic(operator, left, right, out)?
            }
            Expr::Conditional(condition, then, otherwise) => {
                let condition = self.expression(condition, out)?;
                let (condition, _) = self.truth(condition);
                let (mut first, mut second) = (Vec::new(), Vec::new());
                let then = self.expression(then, &mut first)?;
                let otherwise = self.expression(otherwise, &mut second)?;
                let kind = join(&then.kind, &otherwise.kind);
                let then = self.convert(then, &kind, &mut first)?;
                let otherwise = self.convert(otherwise, &kind, &mut second)?;
                let (condition, then, otherwise, first, second) = match positive(&condition) {
                    Some((positive, _)) => (positive, otherwise, then, second, first),
                    None => (condition, then, otherwise, first, second),
                };
                let name = self.temporary();
                if first.is_empty() && second.is_empty() {
                    out.push(Rust::Line(format!(
                        "let {name} = if {condition} {{ {} }} else {{ {} }};",
                        then.code, otherwise.code
                    )));
                } else {
                    // What each branch needs is computed within it, its value the block's.
                    let mut block = CodeWriter::new();
                    let (first_statements, first_value) = block_value(first, then.code);
                    block.block(format!("let {name} = if {condition} {{"), "", |block| {
                        print(&first_statements, block);
                        block.line(first_value);
                    });
                    let (second_statements, second_value) = block_value(second, otherwise.code);
                    if second_statements.is_empty() && second_value.starts_with("if ") {
                        // `else { if .. }` is `else if ..`.
                        block.line(format!("}} else {second_value};"));
                    } else {
                        block.block("} else {", "};", |block| {
                            print(&second_statements, block);
                            block.line(second_value);
                        });
                    }
                    out.push(Rust::Line(
                        block.finish().trim_end_matches('\n').to_owned(),
                    ));
                }
                Operand::new(name, ATOM, kind)
            }
            Expr::Call(builtin, arguments) => {
                let mut operands = Vec::new();
                for argument in arguments {
                    let operand = self.expression(argument, out)?;
                    let operand = if operand.kind == Kind::Integer {
                        self.convert(operand, &Kind::Float(None), out)?
                    } else {
                        operand
                    };
                    operands.push(operand);
                }
                self.call(*builtin, operands, out)?
            }
        })
    }

    /// `ddx(operand, V(node))`: the derivative of `operand` by the node's potential, the
    /// sum of its derivatives by the probes that read the node, each by its side.
    fn ddx(&mut self, operand: &Expr, node: usize, out: &mut Vec<Rust>) -> Result<Operand, Error> {
        if self.mode != Mode::Evaluate {
            return Err(Error::at(
                0,
                "a ddx that the noise reads is not compiled yet".into(),
            ));
        }
        let operand = self.expression(operand, out)?;
        let Kind::Dual(probes) = operand.kind.clone() else {
            return Ok(Operand::new("R::constant(0.0)", ATOM, Kind::Real));
        };
        let operand = self.bind(operand, out);
        let module = self.facts.module;
        let side = |probe: &Probe| match *probe {
            Probe::Voltage(plus, minus) => i8::from(plus == node) - i8::from(minus == node),
            Probe::Limited(index) => {
                i8::from(module.limits[index].plus == node)
                    - i8::from(module.limits[index].minus == node)
            }
            Probe::Current(_) => 0,
        };
        let names: Vec<String> = probes
            .iter()
            .map(|probe| {
                if side(probe) == 0 {
                    "_".to_owned()
                } else {
                    self.temporary()
                }
            })
            .collect();
        out.push(Rust::Line(format!(
            "let [{}] = {}.gradient;",
            names.join(", "),
            operand.code
        )));
        let mut terms = Vec::new();
        for (probe, name) in probes.iter().zip(&names) {
            match side(probe) {
                1 => terms.push(format!("+ {name}")),
                -1 => terms.push(format!("- {name}")),
                _ => {}
            }
        }
        let Some(first) = terms.first() else {
            return Ok(Operand::new("R::constant(0.0)", ATOM, Kind::Real));
        };
        let mut code = first
            .strip_prefix("+ ")
            .map_or_else(|| first.replacen("- ", "-", 1), str::to_owned);
        for term in &terms[1..] {
            code = format!("{code} {term}");
        }
        let precedence = if terms.len() > 1 {
            SUM
        } else if code.starts_with('-') {
            UNARY
        } else {
            ATOM
        };
        Ok(Operand::new(code, precedence, Kind::Real))
    }

    /// The negation of `operand`, a truth value: a comparison with its operator flipped
    /// where the flip is exact, as equality's is and an integers' order's is, a number
    /// compared with zero, or the truth negated.
    fn negation(&mut self, operand: &Expr, out: &mut Vec<Rust>) -> Result<Operand, Error> {
        if let Expr::Given(parameter) = operand {
            self.parameters_read.insert(*parameter);
            let name = &self.facts.module.parameters[*parameter].name;
            return Ok(Operand::new(
                format!("parameters.{name}.is_none()"),
                ATOM,
                Kind::Bool,
            ));
        }
        if let Expr::Binary(operator @ ("==" | "!=" | "<" | "<=" | ">" | ">="), left, right) =
            operand
        {
            let left = self.expression(left, out)?;
            let right = self.expression(right, out)?;
            let integers = left.kind == Kind::Integer && right.kind == Kind::Integer;
            let flipped = match *operator {
                "==" => Some("!="),
                "!=" => Some("=="),
                "<" if integers => Some(">="),
                "<=" if integers => Some(">"),
                ">" if integers => Some("<="),
                ">=" if integers => Some("<"),
                _ => None,
            };
            if let Some(flipped) = flipped {
                return Ok(compare(flipped, &left, &right));
            }
            let compared = compare(operator, &left, &right);
            return Ok(Operand::new(
                format!("!{}", compared.at(ATOM)),
                UNARY,
                Kind::Bool,
            ));
        }
        let operand = self.expression(operand, out)?;
        Ok(match operand.kind {
            Kind::Integer => Operand::new(format!("{} == 0", operand.at(SUM)), COMPARE, Kind::Bool),
            Kind::Bool => Operand::new(format!("!{}", operand.at(ATOM)), UNARY, Kind::Bool),
            _ => Operand::new(
                format!("{} == 0.0", Self::value_of(&operand).at(SUM)),
                COMPARE,
                Kind::Bool,
            ),
        })
    }

    /// A probe's value: with its derivative in an evaluation, plain in a noise analysis.
    fn probe(&mut self, probe: Probe, out: &mut Vec<Rust>) -> Result<Operand, Error> {
        let module = self.facts.module;
        let voltage = |emitter: &mut Self, plus: usize, minus: usize, read: &str| {
            for node in [plus, minus] {
                if module.is_internal(node) {
                    emitter.shape.insert(module.nodes[node].clone());
                }
            }
            format!(
                "at.{read}({}) - at.{read}({})",
                emitter.facts.node(plus, "shape."),
                emitter.facts.node(minus, "shape.")
            )
        };
        let kind = if self.mode == Mode::Evaluate {
            Kind::Dual(vec![probe])
        } else {
            Kind::Real
        };
        if let Some(field) = self.probe_field(probe) {
            self.probes.insert(probe);
            match probe {
                Probe::Voltage(plus, minus) => {
                    let _ = voltage(self, plus, minus, "value");
                }
                Probe::Current(branch) => {
                    self.shape.insert(module.branches[branch].name.clone());
                }
                Probe::Limited(_) => {}
            }
            return Ok(Operand::new(format!("v.{field}"), ATOM, kind));
        }
        let Probe::Limited(index) = probe else {
            return Ok(Operand::new("R::constant(0.0)", ATOM, kind));
        };
        let limit = module.limits[index].clone();
        let applied = voltage(self, limit.plus, limit.minus, "value");
        if self.mode == Mode::Noise {
            let name = self.temporary();
            out.push(Rust::Line(format!("let {name} = R::constant({applied});")));
            return Ok(Operand::new(name, ATOM, kind));
        }
        let mut arguments = vec![format!("applied{index}"), format!("memory.limited{index}")];
        for argument in &limit.arguments {
            let operand = self.expression(argument, out)?;
            arguments.push(Self::value_of(&operand).code);
        }
        out.push(Rust::Line(format!("let applied{index} = {applied};")));
        match limit.function {
            Limiter::Function(function) => {
                let function = &module.functions[function].name;
                out.push(Rust::Line(format!(
                    "let limited{index} = if at.limiting {{ {function}({}) }} else {{ applied{index} }};",
                    arguments.join(", ")
                )));
                out.push(Rust::If(
                    (
                        format!("limited{index}.total_cmp(&applied{index}).is_ne()"),
                        ATOM,
                    ),
                    vec![Rust::Line("stamp.mark_limited();".to_owned())],
                    Vec::new(),
                ));
            }
            Limiter::Junction | Limiter::Gate => {
                let function = if limit.function == Limiter::Junction {
                    "limit_junction"
                } else {
                    "limit_gate"
                };
                out.push(Rust::Line(format!(
                    "let limited{index} = match at.limiting.then(|| {function}({})).flatten() {{\n    Some(limited) => {{\n        stamp.mark_limited();\n        limited\n    }}\n    None => applied{index},\n}};",
                    arguments.join(", ")
                )));
            }
        }
        out.push(Rust::Line(format!(
            "memory.limited{index} = limited{index};"
        )));
        if self.extended.contains(&index) {
            out.push(Rust::Line(format!("v.applied{index} = applied{index};")));
            out.push(Rust::Line(format!("v.limited{index} = limited{index};")));
        }
        let name = self.temporary();
        let moving = voltage(self, limit.plus, limit.minus, "unknown::<R>");
        out.push(Rust::Line(format!(
            "let {name} = Dual::variable(({moving}).with_value(limited{index}));"
        )));
        Ok(Operand::new(name, ATOM, kind))
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one arm per construct of the language, which read best side by side"
    )]
    fn arithmetic(
        &mut self,
        operator: &str,
        left: Operand,
        right: Operand,
        out: &mut Vec<Rust>,
    ) -> Result<Operand, Error> {
        // A truth value counts as the integer 1 or 0, as Verilog-A's arithmetic has it.
        let left = if left.kind == Kind::Bool {
            self.convert(left, &Kind::Integer, out)?
        } else {
            left
        };
        let right = if right.kind == Kind::Bool {
            self.convert(right, &Kind::Integer, out)?
        } else {
            right
        };
        if left.kind == Kind::Integer && right.kind == Kind::Integer {
            let method = match operator {
                "+" => "wrapping_add",
                "-" => "wrapping_sub",
                "*" => "wrapping_mul",
                _ => {
                    return Err(Error::at(
                        0,
                        format!("the integer operation {operator} is not compiled yet"),
                    ));
                }
            };
            if let Ok(left_value) = left.code.parse::<i32>() {
                if let Ok(right_value) = right.code.parse::<i32>() {
                    let value = match operator {
                        "+" => left_value.wrapping_add(right_value),
                        "-" => left_value.wrapping_sub(right_value),
                        _ => left_value.wrapping_mul(right_value),
                    };
                    return Ok(Operand::new(
                        value.to_string(),
                        if value < 0 { UNARY } else { ATOM },
                        Kind::Integer,
                    ));
                }
                return Ok(Operand::new(
                    format!("i32::{method}({}, {})", left.code, right.code),
                    ATOM,
                    Kind::Integer,
                ));
            }
            return Ok(Operand::new(
                format!("{}.{method}({})", left.at(ATOM), right.code),
                ATOM,
                Kind::Integer,
            ));
        }
        let left = if left.kind == Kind::Integer {
            self.convert(left, &Kind::Float(None), out)?
        } else {
            left
        };
        let right = if right.kind == Kind::Integer {
            self.convert(right, &Kind::Float(None), out)?
        } else {
            right
        };
        let precedence = if matches!(operator, "+" | "-") {
            SUM
        } else {
            PRODUCT
        };
        let plain = |left: &Operand, right: &Operand, kind: Kind| {
            Operand::new(
                format!(
                    "{} {operator} {}",
                    left.at(precedence),
                    right.at(precedence + 1)
                ),
                precedence,
                kind,
            )
        };
        Ok(match (left.kind.clone(), right.kind.clone()) {
            (Kind::Float(Some(a)), Kind::Float(Some(b))) => float(fold(operator, a, b)),
            (Kind::Float(_), Kind::Float(_)) => plain(&left, &right, Kind::Float(None)),
            (Kind::Dual(a), Kind::Dual(b)) => {
                let kind = Kind::Dual(union(&a, &b));
                let left = self.convert(left, &kind, out)?;
                let right = self.convert(right, &kind, out)?;
                plain(&left, &right, kind)
            }
            (Kind::Dual(_), Kind::Real) => {
                let code = match operator {
                    "+" => format!("{}.plus({})", left.at(ATOM), right.code),
                    "-" => format!("{}.plus({})", left.at(ATOM), negate(right).code),
                    "*" => format!("{}.scale({})", left.at(ATOM), right.code),
                    _ => format!("{}.divide({})", left.at(ATOM), right.code),
                };
                Operand::new(code, ATOM, left.kind)
            }
            (Kind::Real, Kind::Dual(_)) => {
                let code = match operator {
                    "+" => format!("{}.plus({})", right.at(ATOM), left.code),
                    "-" => format!("(-{}).plus({})", right.at(ATOM), left.code),
                    "*" => format!("{}.scale({})", right.at(ATOM), left.code),
                    _ => format!("Dual::constant({}) / {}", left.code, right.at(PRODUCT + 1)),
                };
                let precedence = if operator == "/" { PRODUCT } else { ATOM };
                Operand::new(code, precedence, right.kind)
            }
            (Kind::Float(_), right_kind @ (Kind::Dual(_) | Kind::Real)) => match operator {
                "+" => Operand::new(
                    format!("{} + {}", right.at(SUM), left.at(SUM + 1)),
                    SUM,
                    right_kind,
                ),
                "*" if matches!(right_kind, Kind::Real) => Operand::new(
                    format!("{} * {}", right.at(PRODUCT), left.at(PRODUCT + 1)),
                    PRODUCT,
                    right_kind,
                ),
                "*" | "-" if matches!(right_kind, Kind::Dual(_)) => {
                    plain(&left, &right, right_kind)
                }
                _ => {
                    let constant = if matches!(right_kind, Kind::Dual(_)) {
                        "Dual::number"
                    } else {
                        "R::constant"
                    };
                    let left = Operand::new(
                        format!("{constant}({})", left.code),
                        ATOM,
                        right_kind.clone(),
                    );
                    plain(&left, &right, right_kind)
                }
            },
            (Kind::Dual(_) | Kind::Real, Kind::Float(_)) | (Kind::Real, Kind::Real) => {
                let kind = left.kind.clone();
                plain(&left, &right, kind)
            }
            _ => {
                return Err(Error::at(
                    0,
                    format!(
                        "the operation {operator} on a {:?} and a {:?} is not compiled yet",
                        left.kind, right.kind
                    ),
                ));
            }
        })
    }

    fn call(
        &mut self,
        builtin: Builtin,
        mut operands: Vec<Operand>,
        out: &mut Vec<Rust>,
    ) -> Result<Operand, Error> {
        let first = operands.remove(0);
        let second = operands.pop();
        if let Some(smooth) = smooth(builtin) {
            return Ok(self.smooth(&smooth, first, out));
        }
        let kind = first.kind.clone();
        let method = |operand: &Operand, method: &str| {
            Operand::new(
                format!("{}.{method}()", operand.at(ATOM)),
                ATOM,
                operand.kind.clone(),
            )
        };
        Ok(match (builtin, second) {
            (Builtin::Exp, None) => method(&first, "exp"),
            (Builtin::Sqrt, None) => method(&first, "sqrt"),
            (Builtin::Ln, None) if matches!(kind, Kind::Dual(_)) => {
                let x = self.bind(first, out);
                Operand::new(
                    format!("{0}.chain(({0}.value.ln(), {0}.value.recip()))", x.code),
                    ATOM,
                    kind,
                )
            }
            (Builtin::Ln, None) => method(&first, "ln"),
            (Builtin::Abs, None) if matches!(kind, Kind::Dual(_)) => {
                let x = self.bind(first, out);
                let name = self.temporary();
                out.push(Rust::Line(format!(
                    "let {name} = if {0}.value.value() < 0.0 {{ -{0} }} else {{ {0} }};",
                    x.code
                )));
                Operand::new(name, ATOM, kind)
            }
            (Builtin::Abs, None) => method(&first, "abs"),
            (Builtin::Min | Builtin::Max, Some(second)) => {
                let kind = join(&kind, &second.kind);
                let first = self.convert(first, &kind, out)?;
                let second = self.convert(second, &kind, out)?;
                let name = if builtin == Builtin::Min {
                    "min"
                } else {
                    "max"
                };
                if matches!(kind, Kind::Dual(_)) {
                    let (a, b) = (self.bind(first, out), self.bind(second, out));
                    let order = if builtin == Builtin::Min { "<" } else { ">" };
                    let chosen = self.temporary();
                    out.push(Rust::Line(format!(
                        "let {chosen} = if {1}.value.value() {order} {0}.value.value() {{ {1} }} else {{ {0} }};",
                        a.code, b.code
                    )));
                    Operand::new(chosen, ATOM, kind)
                } else {
                    Operand::new(
                        format!("{}.{name}({})", first.at(ATOM), second.code),
                        ATOM,
                        kind,
                    )
                }
            }
            (Builtin::Pow, Some(exponent)) => self.power(first, exponent, out)?,
            _ => {
                return Err(Error::at(
                    0,
                    "a function with the wrong number of arguments".into(),
                ));
            }
        })
    }

    /// The function `smooth` of `x`: on plain floats by their own method, on jolty's
    /// numbers through their chain rule, to the second order, and on dual numbers through
    /// theirs, its slope one of jolty's numbers.
    fn smooth(&mut self, smooth: &Smooth, x: Operand, out: &mut Vec<Rust>) -> Operand {
        let fill = |template: &str, a: &str, x: &str, r: &str| {
            template
                .replace("{a}", a)
                .replace("{x}", x)
                .replace("{r}", r)
        };
        match x.kind {
            Kind::Float(_) | Kind::Integer | Kind::Bool => Operand::new(
                format!("{}.{}()", x.at(ATOM), smooth.method),
                ATOM,
                Kind::Float(None),
            ),
            Kind::Real => {
                let x = self.bind(x, out);
                let a = self.temporary();
                out.push(Rust::Line(format!("let {a} = {}.value();", x.code)));
                let [value, slope, curvature] =
                    smooth.plain.map(|template| fill(template, &a, &x.code, ""));
                Operand::new(
                    format!("{}.chain(({value}, {slope}, {curvature}))", x.code),
                    ATOM,
                    Kind::Real,
                )
            }
            Kind::Dual(_) => {
                let x = self.bind(x, out);
                let a = self.temporary();
                out.push(Rust::Line(format!("let {a} = {}.value.value();", x.code)));
                let [value, slope, curvature] =
                    smooth.plain.map(|template| fill(template, &a, "", ""));
                let r = self.temporary();
                out.push(Rust::Line(format!(
                    "let {r} = {}.value.chain(({value}, {slope}, {curvature}));",
                    x.code
                )));
                let inner = format!("{}.value", x.code);
                let slope = fill(smooth.slope, &a, &inner, &r);
                Operand::new(format!("{}.chain(({r}, {slope}))", x.code), ATOM, x.kind)
            }
        }
    }

    /// `base` raised to `exponent`.
    fn power(
        &mut self,
        base: Operand,
        exponent: Operand,
        out: &mut Vec<Rust>,
    ) -> Result<Operand, Error> {
        Ok(match (base.kind.clone(), exponent.kind.clone()) {
            (Kind::Float(_), Kind::Float(_)) => Operand::new(
                format!("{}.powf({})", base.at(ATOM), exponent.code),
                ATOM,
                Kind::Float(None),
            ),
            (Kind::Real, Kind::Float(_)) => Operand::new(
                format!("{}.powf({})", base.at(ATOM), exponent.code),
                ATOM,
                Kind::Real,
            ),
            (Kind::Real | Kind::Float(_), Kind::Real) => {
                let base = self.convert(base, &Kind::Real, out)?;
                Operand::new(
                    format!("{}.pow({})", base.at(ATOM), exponent.code),
                    ATOM,
                    Kind::Real,
                )
            }
            (Kind::Dual(_), Kind::Float(_) | Kind::Real) => {
                let x = self.bind(base, out);
                let e = self.convert(exponent, &Kind::Real, out)?;
                let e = self.bind(e, out);
                // The power's derivative from the power itself, `e · x^e / x`, but at zero.
                let power = self.temporary();
                out.push(Rust::Line(format!(
                    "let {power} = {}.value.pow({});",
                    x.code, e.code
                )));
                let slope = self.temporary();
                out.push(Rust::Line(format!(
                    "let {slope} = if {0}.value.value() == 0.0 {{ {1} * {0}.value.pow({1} - 1.0) }} else {{ {1} * {power} / {0}.value }};",
                    x.code, e.code
                )));
                Operand::new(
                    format!("{}.chain(({power}, {slope}))", x.code),
                    ATOM,
                    x.kind,
                )
            }
            (Kind::Float(_) | Kind::Real, Kind::Dual(_)) => {
                let base = self.convert(base, &Kind::Real, out)?;
                Operand::new(
                    format!("{}.scale({}.ln()).exp()", exponent.at(ATOM), base.at(ATOM)),
                    ATOM,
                    exponent.kind,
                )
            }
            (Kind::Dual(a), Kind::Dual(b)) => {
                let kind = Kind::Dual(union(&a, &b));
                let x = self.convert(base, &kind, out)?;
                let x = self.bind(x, out);
                let e = self.convert(exponent, &kind, out)?;
                Operand::new(
                    format!(
                        "({0}.chain(({0}.value.ln(), {0}.value.recip())) * {1}).exp()",
                        x.code,
                        e.at(PRODUCT + 1)
                    ),
                    ATOM,
                    kind,
                )
            }
            _ => {
                return Err(Error::at(
                    0,
                    "pow of these kinds is not compiled yet".into(),
                ));
            }
        })
    }
}

/// The statements of a block and the code of its value, `value`, the last statement's
/// expression in its place where that statement only binds it.
fn block_value(mut statements: Vec<Rust>, value: String) -> (Vec<Rust>, String) {
    let bound = format!("let {value} = ");
    if let Some(Rust::Line(last)) = statements.last()
        && let Some(expression) = last
            .strip_prefix(&bound)
            .and_then(|rest| rest.strip_suffix(';'))
    {
        let expression = expression.to_owned();
        statements.pop();
        return (statements, expression);
    }
    (statements, value)
}

/// The negation of `operand`: exact, so a double one cancels.
fn negate(operand: Operand) -> Operand {
    match operand.kind {
        Kind::Float(Some(value)) => float(-value),
        Kind::Integer => match operand.code.parse::<i32>() {
            Ok(value) => {
                let negated = value.wrapping_neg();
                Operand::new(
                    negated.to_string(),
                    if negated < 0 { UNARY } else { ATOM },
                    Kind::Integer,
                )
            }
            Err(_) => Operand::new(
                format!("{}.wrapping_neg()", operand.at(ATOM)),
                ATOM,
                Kind::Integer,
            ),
        },
        _ => match operand.code.strip_prefix('-') {
            Some(inner) if operand.precedence == UNARY => Operand::new(inner, ATOM, operand.kind),
            _ => Operand::new(format!("-{}", operand.at(ATOM)), UNARY, operand.kind),
        },
    }
}

/// The comparison `left operator right`, of their values.
fn compare(operator: &str, left: &Operand, right: &Operand) -> Operand {
    // Two constants compare as they do, which the code need not repeat.
    let constant = |operand: &Operand| match operand.kind {
        Kind::Float(Some(value)) => Some(value),
        Kind::Integer => operand.code.parse::<i32>().ok().map(f64::from),
        _ => None,
    };
    if let (Some(left), Some(right)) = (constant(left), constant(right)) {
        return Operand::new(
            compare_order(operator, left, right).to_string(),
            ATOM,
            Kind::Bool,
        );
    }
    let (left, right) = if left.kind == Kind::Integer && right.kind == Kind::Integer {
        (left.clone(), right.clone())
    } else {
        (Emitter::value_of(left), Emitter::value_of(right))
    };
    let zero = |operand: &Operand| operand.kind == Kind::Float(Some(0.0));
    let exact = left.kind == Kind::Integer || zero(&left) || zero(&right);
    match operator {
        "==" | "!=" if !exact => {
            let test = format!(
                "matches!({}.partial_cmp(&{}), Some(Ordering::Equal))",
                left.at(ATOM),
                right.at(UNARY)
            );
            if operator == "==" {
                Operand::new(test, ATOM, Kind::Bool)
            } else {
                Operand::new(format!("!{test}"), UNARY, Kind::Bool)
            }
        }
        _ => Operand::new(
            format!("{} {operator} {}", left.at(SUM), right.at(SUM)),
            COMPARE,
            Kind::Bool,
        ),
    }
}
