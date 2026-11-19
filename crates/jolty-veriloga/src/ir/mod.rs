//! Resolved intermediate representation of a Verilog-A module: bound names, split resistive,
//! reactive, and noise contributions, and static/dynamic variable dependence.
//!
//! A value is *static* when it depends only on parameters and temperature, and *dynamic* when
//! it depends on probes, limited probes, or simulator state such as `gmin` through its value
//! or enclosing control flow. Static values compute once during preparation; dynamic values
//! evaluate on every iteration with exact derivatives with respect to each dependent probe.

mod analysis;
mod checks;
mod currents;
mod ddt;
pub(crate) mod fold;
mod lower;

use std::collections::BTreeSet;

use jolty_models::Nature;

use crate::Error;
use crate::ast::{self, Type};

pub(crate) use analysis::visit_contributions;
pub(crate) use currents::probed_currents;
pub(crate) use fold::{constant, fold};

/// Differentiation variable for kernel evaluation: a node voltage difference, a
/// potential-branch current, or a limited voltage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Probe {
    /// Voltage difference from the first node index to the second.
    Voltage(usize, usize),
    /// Current through the branch at this index (carrying a potential contribution).
    Current(usize),
    /// Limited voltage associated with the `$limit` entry at this index.
    Limited(usize),
}

/// Built-in real-valued mathematical function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Builtin {
    Exp,
    /// Exponential minus one (`eˣ − 1`), accurate near zero (jolty extension matching C
    /// `expm1`).
    Expm1,
    Ln,
    /// Natural logarithm of one plus argument (`ln(1 + x)`), accurate near zero (jolty
    /// extension matching C `log1p`).
    Ln1p,
    /// Base-10 logarithm, Verilog-A's `log`.
    Log,
    Sqrt,
    Pow,
    Abs,
    Min,
    Max,
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    Asin,
    Acos,
    Atan,
    Floor,
    Ceil,
}

impl Builtin {
    /// Returns the [`Builtin`] corresponding to Verilog-A function `name`, if any.
    pub(super) fn named(name: &str) -> Option<Self> {
        Some(match name {
            "exp" => Self::Exp,
            "expm1" => Self::Expm1,
            "ln" => Self::Ln,
            "ln1p" => Self::Ln1p,
            "log" => Self::Log,
            "sqrt" => Self::Sqrt,
            "pow" => Self::Pow,
            "abs" => Self::Abs,
            "min" => Self::Min,
            "max" => Self::Max,
            "sin" => Self::Sin,
            "cos" => Self::Cos,
            "tan" => Self::Tan,
            "sinh" => Self::Sinh,
            "cosh" => Self::Cosh,
            "tanh" => Self::Tanh,
            "asin" => Self::Asin,
            "acos" => Self::Acos,
            "atan" => Self::Atan,
            "floor" => Self::Floor,
            "ceil" => Self::Ceil,
            _ => return None,
        })
    }

    /// Returns the required argument count of the built-in function.
    pub(super) const fn arity(self) -> usize {
        match self {
            Self::Pow | Self::Min | Self::Max => 2,
            _ => 1,
        }
    }
}

/// Name-resolved IR expression.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Expression {
    Real(f64),
    Integer(i32),
    Parameter(usize),
    /// Non-constant `localparam` referenced by its module index.
    LocalParameter(usize),
    /// Predicate testing whether the parameter at this index was supplied.
    Given(usize),
    Variable(usize),
    /// Device temperature `$temperature` in kelvins.
    Temperature,
    /// Simulator shunt conductance `$simparam("gmin")` across nonlinear junctions.
    Gmin,
    /// Resistance threshold `$simparam("small_resistance")` ([`jolty_models::SMALL_RESISTANCE`])
    /// below which a resistor introduces a branch current unknown.
    SmallResistance,
    Probe(Probe),
    Unary(&'static str, Box<Self>),
    Binary(&'static str, Box<Self>, Box<Self>),
    Conditional(Box<Self>, Box<Self>, Box<Self>),
    Call(Builtin, Vec<Self>),
    /// Partial derivative `ddx(f, V(node))` with respect to a node potential, carrying no second
    /// derivative.
    Ddx(Box<Self>, usize),
    /// Predicate indicating whether shape-determining conditional `id` holds.
    Shape(usize),
}

/// Noise source attached to a branch contribution.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Noise {
    pub(crate) name: String,
    /// Power spectral density at 1 Hz.
    pub(crate) power: Expression,
    /// Frequency roll-off exponent for flicker noise, or `None` for white noise.
    pub(crate) exponent: Option<Expression>,
}

/// Name-resolved IR statement.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Statement {
    Assign(usize, Expression),
    /// Flow or potential contribution to a branch.
    Contribute {
        branch: usize,
        potential: bool,
        resistive: Option<Expression>,
        /// Charge or flux expression whose time derivative is added.
        reactive: Option<Expression>,
        /// Expressions factored from outside `ddt(...)` (`k * ddt(x)` folded into
        /// `ddt(k * x)`), each with its source line; verified static after analysis so
        /// reactive contributions stay charge-conserving.
        reactive_factors: Vec<(Expression, usize)>,
        noise: Vec<Noise>,
    },
    /// Numbered conditional statement with condition, then-branch, and else-branch.
    If(usize, Expression, Vec<Self>, Vec<Self>),
    While(Expression, Vec<Self>),
    /// Fatal preparation diagnostic from `$error` or `$fatal`, formatting arguments into `%g`
    /// slots.
    Fail(String, Vec<Expression>),
    /// Diagnostic from `$warning`, `$strobe`, or similar tasks, formatting arguments into `%g`
    /// slots and flagging whether evaluation depends on probes.
    Warn(String, Vec<Expression>, bool),
}

/// Function limiting a probe step: either a module analog function or a built-in jolty limiter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Limiter {
    Function(usize),
    /// Built-in junction limiter `"pnjlim"` ([`jolty_models::limit_junction`]).
    Junction,
    /// Built-in FET gate limiter `"fetlim"` ([`jolty_models::limit_gate`]).
    Gate,
}

/// Limited voltage probe pairing its branch nodes, limiter function, and extra arguments after
/// `(v_new, v_old)`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Limit {
    pub(crate) plus: usize,
    pub(crate) minus: usize,
    pub(crate) function: Limiter,
    pub(crate) arguments: Vec<Expression>,
}

/// Resolved parameter range `(exclude, lower, upper)` where each bound pairs an expression with
/// an inclusivity flag (`None` denotes infinity).
pub(crate) type Bounds = (bool, Option<(Expression, bool)>, Option<(Expression, bool)>);

/// Resolved non-constant `localparam` evaluated once during preparation in declaration order so
/// parameter derivatives propagate wherever it is read.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LocalParameter {
    /// Sanitized Rust identifier in generated kernel code.
    pub(crate) name: String,
    /// Original identifier from Verilog-A source.
    pub(crate) external: String,
    pub(crate) kind: Type,
    pub(crate) value: Expression,
}

/// Resolved model parameter.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Parameter {
    /// Sanitized Rust identifier in generated kernel code.
    pub(crate) name: String,
    /// Original identifier from Verilog-A source and circuit parameter maps.
    pub(crate) external: String,
    pub(crate) kind: Type,
    /// Default value expression (a constant or expression over earlier parameters).
    pub(crate) default: Expression,
    /// Flag set when the parameter is queried by `$param_given` or defaults from other
    /// parameters.
    pub(crate) optional: bool,
    pub(crate) ranges: Vec<Bounds>,
    pub(crate) description: Option<String>,
    pub(crate) units: Option<String>,
    /// Additional parameter names introduced by `aliasparam`.
    pub(crate) aliases: Vec<String>,
}

/// Global reference node (electrical ground or thermal ambient).
///
/// Reference potential is always zero and contributions to it are discarded.
pub(crate) const GROUND: usize = usize::MAX;

/// Named or unnamed branch between two nodes (or one node and [`GROUND`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Branch {
    pub(crate) name: String,
    pub(crate) plus: usize,
    pub(crate) minus: usize,
}

/// Analog function used as a probe limiter, with local variable slots ordered as inputs,
/// locals, then result.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Function {
    pub(crate) name: String,
    pub(crate) inputs: usize,
    pub(crate) variables: Vec<(String, Type)>,
    pub(crate) body: Vec<Statement>,
}

/// Resolved and analyzed Verilog-A module ready for Rust kernel emission.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Module {
    pub(crate) name: String,
    /// Rust node names ordered with ports first, followed by internal nodes.
    pub(crate) nodes: Vec<String>,
    /// Source node names used to label unknowns.
    pub(crate) labels: Vec<String>,
    /// Physical nature measured by each node unknown: voltage, temperature, or an auxiliary
    /// branch current.
    pub(crate) natures: Vec<Nature>,
    pub(crate) ports: usize,
    pub(crate) branches: Vec<Branch>,
    pub(crate) parameters: Vec<Parameter>,
    /// Non-constant `localparam`s prepared before range checks and the analog block.
    pub(crate) local_parameters: Vec<LocalParameter>,
    pub(crate) variables: Vec<(String, Type)>,
    /// Indices of variables designated as output variables.
    pub(crate) outputs: Vec<usize>,
    pub(crate) functions: Vec<Function>,
    pub(crate) limits: Vec<Limit>,
    pub(crate) analog: Vec<Statement>,
    /// Per-variable flag indicating whether the variable depends on probes or simulator state.
    pub(crate) dynamic: Vec<bool>,
    /// Probes each variable depends on through its value.
    pub(crate) probes: Vec<BTreeSet<Probe>>,
}

/// Converts Verilog-A `name` into a valid Rust snake-case identifier, appending `_` when the
/// result collides with a Rust keyword.
pub(crate) fn identifier(name: &str) -> String {
    const KEYWORDS: [&str; 51] = [
        "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum",
        "extern", "false", "fn", "for", "gen", "if", "impl", "in", "let", "loop", "match", "mod",
        "move", "mut", "pub", "ref", "return", "self", "Self", "static", "struct", "super",
        "trait", "true", "try", "type", "unsafe", "use", "where", "while", "abstract", "become",
        "box", "do", "final", "macro", "override", "priv", "typeof", "unsized", "virtual",
    ];
    // Insert underscores before uppercase transitions following lowercase letters or digits.
    let mut snake = String::new();
    let mut previous: Option<char> = None;
    for c in name.chars() {
        if c.is_uppercase() && previous.is_some_and(|p| p.is_lowercase() || p.is_ascii_digit()) {
            snake.push('_');
        }
        snake.extend(c.to_lowercase());
        previous = Some(c);
    }
    if KEYWORDS.contains(&snake.as_str()) {
        format!("{snake}_")
    } else {
        snake
    }
}

/// Returns `name`, or the first `name2`, `name3`, ... for which `taken` returns `false`.
pub(crate) fn apart(name: &str, taken: impl Fn(&str) -> bool) -> String {
    let mut unique = name.to_owned();
    let mut count = 1;
    while taken(&unique) {
        count += 1;
        unique = format!("{name}{count}");
    }
    unique
}

/// Resolves names, splits contributions, and performs static/dynamic dependence analysis on
/// `module`.
///
/// # Errors
///
/// Returns an [`Error`] on undeclared identifiers or unsupported constructs.
pub(crate) fn resolve(module: &ast::Module) -> Result<Module, Error> {
    let (nodes, thermal) = checks::nodes_of(module)?;
    checks::check_module_duplicates(module)?;
    checks::check_aliases(module)?;
    let mut sink = lower::initialize_sink(module, &nodes)?;
    let context = lower::Context {
        module,
        nodes: &nodes,
        thermal: &thermal,
    };
    let (parameters, module_locals) = lower::resolve_declarations(context, &mut sink)?;
    let mut functions = Vec::with_capacity(module.functions.len());
    for function in &module.functions {
        functions.push(lower::resolve_function(
            context,
            &module_locals,
            &mut sink.local_parameters,
            function,
        )?);
    }
    let mut bindings = lower::local_bindings(&module_locals);
    bindings.extend(
        module
            .variables
            .iter()
            .enumerate()
            .map(|(index, (name, _))| (name.as_str(), lower::Binding::Variable(index))),
    );
    let scope = lower::Scope::block(context, &bindings, &module_locals, true);
    let mut analog = scope.statements(&module.analog, &mut sink)?;
    analysis::number(&mut analog, &mut 0);
    let (versions, origins) = analysis::version(&mut analog, &mut sink.variables);
    let outputs = module
        .outputs
        .iter()
        .map(|(name, _, _)| scope.variable(name, 0).map(|variable| versions[variable]))
        .collect::<Result<_, _>>()?;
    let mut resolved = Module {
        name: module.name.clone(),
        ports: module.ports.len(),
        nodes: checks::rust_names(&nodes),
        labels: nodes.clone(),
        natures: thermal
            .iter()
            .map(|&is_thermal| {
                if is_thermal {
                    Nature::Temperature
                } else {
                    Nature::Voltage
                }
            })
            .collect(),
        branches: checks::named_apart(sink.branches),
        parameters,
        local_parameters: sink.local_parameters,
        variables: sink.variables,
        outputs,
        functions,
        limits: sink.limits,
        analog,
        dynamic: Vec::new(),
        probes: Vec::new(),
    };
    analysis::analyze(&mut resolved);
    if currents::flow_currents(&mut resolved)? {
        analysis::analyze(&mut resolved);
    }
    checks::check_reactive_factors(&resolved)?;
    checks::check_initial_variables(&resolved, &sink.initial, &origins)?;
    analysis::mark_dynamic_warnings(&mut resolved);
    Ok(resolved)
}
