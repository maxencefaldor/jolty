//! The compiler's representation of a module: its names resolved, each contribution split
//! into its resistive, reactive and noise parts, and what each variable depends on.
//!
//! A number is *static* when it depends on the parameters and the temperature alone, and
//! *dynamic* when it depends on a probe, a limited probe or the simulator's state, such as
//! gmin, through its value or through a condition it was assigned under. Static numbers
//! are prepared once; dynamic ones are evaluated at every iteration, with their exact
//! derivatives with respect to each probe they depend on.

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

/// What an evaluation differentiates by: a voltage between two nodes, the current of a
/// branch that holds a voltage, or a voltage limited from its last evaluation's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Probe {
    /// The voltage of the first node over the second, by index among the nodes.
    Voltage(usize, usize),
    /// The current of the branch at that index, one with a potential contribution.
    Current(usize),
    /// The limited voltage of the `$limit` at that index.
    Limited(usize),
}

/// A built-in function of real numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Builtin {
    Exp,
    /// `eˣ − 1`, to its last digit where `x` is near zero: jolty's, as C's `expm1`; not
    /// in the Verilog-AMS standard.
    Expm1,
    Ln,
    /// `ln(1 + x)`, to its last digit where `x` is near zero: jolty's, as C's `log1p`;
    /// not in the Verilog-AMS standard.
    Ln1p,
    /// The logarithm to base 10, Verilog-A's `log`.
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
    /// The built-in function Verilog-A names `name`.
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

    /// How many arguments it takes.
    pub(super) const fn arity(self) -> usize {
        match self {
            Self::Pow | Self::Min | Self::Max => 2,
            _ => 1,
        }
    }
}

/// An expression, its names resolved.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Expression {
    Real(f64),
    Integer(i32),
    Parameter(usize),
    /// A non-constant `localparam` at that index among the module's.
    LocalParameter(usize),
    /// Whether the parameter at that index was given.
    Given(usize),
    Variable(usize),
    /// `$temperature`, the device's temperature in kelvins.
    Temperature,
    /// `$simparam("gmin")`, the conductance across every nonlinear junction.
    Gmin,
    /// `$simparam("small_resistance")`, jolty's resistance below which a resistance
    /// carries its current as an unknown, [`jolty_models::SMALL_RESISTANCE`].
    SmallResistance,
    Probe(Probe),
    Unary(&'static str, Box<Self>),
    Binary(&'static str, Box<Self>, Box<Self>),
    Conditional(Box<Self>, Box<Self>, Box<Self>),
    Call(Builtin, Vec<Self>),
    /// `ddx(f, V(node))`: the derivative of `f` by the potential of the node, which has
    /// no derivative of its own in turn.
    Ddx(Box<Self>, usize),
    /// Whether the condition of the conditional at that number, one that shapes the
    /// kernel, holds: as its shape decided it.
    Shape(usize),
}

/// A noise source a contribution adds.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Noise {
    pub(crate) name: String,
    /// Its power spectral density at 1 Hz.
    pub(crate) power: Expression,
    /// For flicker noise, the power of the frequency its density falls with; `None` for
    /// white noise.
    pub(crate) exponent: Option<Expression>,
}

/// A statement, its names resolved.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Statement {
    Assign(usize, Expression),
    /// A contribution to a branch: of its flow, or of its potential.
    Contribute {
        branch: usize,
        potential: bool,
        resistive: Option<Expression>,
        /// The charge or flux whose time derivative it adds.
        reactive: Option<Expression>,
        /// Expressions factored from outside `ddt(...)` (`k * ddt(x)` folded into
        /// `ddt(k * x)`), each with its source line; verified static after analysis so
        /// reactive contributions stay charge-conserving.
        reactive_factors: Vec<(Expression, usize)>,
        noise: Vec<Noise>,
    },
    /// A conditional, numbered in the order the analog block holds them, its condition
    /// and its two branches.
    If(usize, Expression, Vec<Self>, Vec<Self>),
    While(Expression, Vec<Self>),
    /// `$error` or `$fatal`: preparation fails with the message, its `%g`s the
    /// arguments' values.
    Fail(String, Vec<Expression>),
    /// `$warning`, `$strobe` and the like: a fact about the instance, as the message
    /// says, its `%g`s the arguments' values; and whether it depends on the probes, so
    /// that the evaluation, not the preparation, finds it.
    Warn(String, Vec<Expression>, bool),
}

/// The function that limits a probe's step: an analog function of the module's, by its
/// index, or one of the simulator's, jolty's own limiting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Limiter {
    Function(usize),
    /// `"pnjlim"`, a junction's: [`jolty_models::limit_junction`].
    Junction,
    /// `"fetlim"`, a field-effect transistor's gate's: [`jolty_models::limit_gate`].
    Gate,
}

/// A limited probe: the voltage it limits and the function that limits it, with the
/// arguments after the two voltages it takes, the new one and the last.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Limit {
    pub(crate) plus: usize,
    pub(crate) minus: usize,
    pub(crate) function: Limiter,
    pub(crate) arguments: Vec<Expression>,
}

/// A range of a parameter: whether it excludes its values rather than admitting them,
/// and its lower and upper bounds, each with whether it is included, `None` for an
/// infinite one: a constant, or an expression of the parameters.
pub(crate) type Bounds = (bool, Option<(Expression, bool)>, Option<(Expression, bool)>);

/// A non-constant `localparam`, resolved: evaluated once during preparation in
/// declaration order so its value and parameter derivatives propagate everywhere it is
/// read.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LocalParameter {
    /// Its name in the kernel's code.
    pub(crate) name: String,
    /// Its name in the source.
    pub(crate) external: String,
    pub(crate) kind: Type,
    pub(crate) value: Expression,
}

/// A parameter, resolved.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Parameter {
    /// Its name in the kernel's code.
    pub(crate) name: String,
    /// Its name in the source and in a circuit's parameters.
    pub(crate) external: String,
    pub(crate) kind: Type,
    /// Its default: a constant, or an expression of the parameters before it, which makes
    /// it optional.
    pub(crate) default: Expression,
    /// Whether it is optional: read as given or not, or defaulting to other parameters.
    pub(crate) optional: bool,
    pub(crate) ranges: Vec<Bounds>,
    pub(crate) description: Option<String>,
    pub(crate) units: Option<String>,
    /// The other names `aliasparam` gives it.
    pub(crate) aliases: Vec<String>,
}

/// The node of the global reference, the ground: the ambient for a thermal node. Its
/// value reads zero, and what it receives goes nowhere, as a kernel's reference does.
pub(crate) const GROUND: usize = usize::MAX;

/// A branch: a named one, or the unnamed one of a pair of nodes, or of one node and the
/// ground.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Branch {
    pub(crate) name: String,
    pub(crate) plus: usize,
    pub(crate) minus: usize,
}

/// An analog function that limits a probe, its body over its own variables: its inputs
/// first, then its locals, then its result.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Function {
    pub(crate) name: String,
    pub(crate) inputs: usize,
    pub(crate) variables: Vec<(String, Type)>,
    pub(crate) body: Vec<Statement>,
}

/// A module, resolved and analyzed.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Module {
    pub(crate) name: String,
    /// The nodes, each by its Rust name: the ports first, in their order, then the
    /// internal nodes.
    pub(crate) nodes: Vec<String>,
    /// The name each node has in the source, which labels its unknown.
    pub(crate) labels: Vec<String>,
    /// What each node's unknown measures: a voltage, a temperature, or the current of a
    /// branch that the compiler gives an unknown.
    pub(crate) natures: Vec<Nature>,
    pub(crate) ports: usize,
    pub(crate) branches: Vec<Branch>,
    pub(crate) parameters: Vec<Parameter>,
    /// Non-constant `localparam`s prepared before range checks and the analog block.
    pub(crate) local_parameters: Vec<LocalParameter>,
    pub(crate) variables: Vec<(String, Type)>,
    /// The variables that are output variables, by index.
    pub(crate) outputs: Vec<usize>,
    pub(crate) functions: Vec<Function>,
    pub(crate) limits: Vec<Limit>,
    pub(crate) analog: Vec<Statement>,
    /// Whether each variable is dynamic.
    pub(crate) dynamic: Vec<bool>,
    /// The probes each variable depends on, through its values.
    pub(crate) probes: Vec<BTreeSet<Probe>>,
}

/// The Rust identifier of the Verilog-A name `name`: the name in snake case, with an
/// underscore after it where that is one of Rust's keywords, as the `type` of `R3_CMC` is.
pub(crate) fn identifier(name: &str) -> String {
    const KEYWORDS: [&str; 51] = [
        "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum",
        "extern", "false", "fn", "for", "gen", "if", "impl", "in", "let", "loop", "match", "mod",
        "move", "mut", "pub", "ref", "return", "self", "Self", "static", "struct", "super",
        "trait", "true", "try", "type", "unsafe", "use", "where", "while", "abstract", "become",
        "box", "do", "final", "macro", "override", "priv", "typeof", "unsized", "virtual",
    ];
    // Rust's names are snake case: an upper-case letter after a lower-case one or a
    // digit starts a word.
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

/// `name`, or the first of `name2`, `name3` and on that `taken` says is not taken.
pub(crate) fn apart(name: &str, taken: impl Fn(&str) -> bool) -> String {
    let mut unique = name.to_owned();
    let mut count = 1;
    while taken(&unique) {
        count += 1;
        unique = format!("{name}{count}");
    }
    unique
}

/// The module `module`, resolved and analyzed.
///
/// # Errors
///
/// Returns an [`Error`] for a name that is not declared, and for a construct the
/// compiler does not compile yet.
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
