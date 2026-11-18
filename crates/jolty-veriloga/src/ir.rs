//! The compiler's representation of a module: its names resolved, each contribution split
//! into its resistive, reactive and noise parts, and what each variable depends on.
//!
//! A number is *static* when it depends on the parameters and the temperature alone, and
//! *dynamic* when it depends on a probe, a limited probe or the simulator's state, such as
//! gmin, through its value or through a condition it was assigned under. Static numbers
//! are prepared once; dynamic ones are evaluated at every iteration, with their exact
//! derivatives with respect to each probe they depend on.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use jolty_models::{BOLTZMANN, ELEMENTARY_CHARGE, Nature};

use crate::Error;
use crate::ast::{self, Expression, Type};

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
    fn named(name: &str) -> Option<Self> {
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
    const fn arity(self) -> usize {
        match self {
            Self::Pow | Self::Min | Self::Max => 2,
            _ => 1,
        }
    }
}

/// An expression, its names resolved.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Expr {
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
    pub(crate) power: Expr,
    /// For flicker noise, the power of the frequency its density falls with; `None` for
    /// white noise.
    pub(crate) exponent: Option<Expr>,
}

/// A statement, its names resolved.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Stmt {
    Assign(usize, Expr),
    /// A contribution to a branch: of its flow, or of its potential.
    Contribute {
        branch: usize,
        potential: bool,
        resistive: Option<Expr>,
        /// The charge or flux whose time derivative it adds.
        reactive: Option<Expr>,
        /// Expressions factored from outside `ddt(...)` (`k * ddt(x)` folded into
        /// `ddt(k * x)`), each with its source line; verified static after analysis so
        /// reactive contributions stay charge-conserving.
        reactive_factors: Vec<(Expr, usize)>,
        noise: Vec<Noise>,
    },
    /// A conditional, numbered in the order the analog block holds them, its condition
    /// and its two branches.
    If(usize, Expr, Vec<Self>, Vec<Self>),
    While(Expr, Vec<Self>),
    /// `$error` or `$fatal`: preparation fails with the message, its `%g`s the
    /// arguments' values.
    Fail(String, Vec<Expr>),
    /// `$warning`, `$strobe` and the like: a fact about the instance, as the message
    /// says, its `%g`s the arguments' values; and whether it depends on the probes, so
    /// that the evaluation, not the preparation, finds it.
    Warn(String, Vec<Expr>, bool),
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
    pub(crate) arguments: Vec<Expr>,
}

/// A range of a parameter: whether it excludes its values rather than admitting them,
/// and its lower and upper bounds, each with whether it is included, `None` for an
/// infinite one: a constant, or an expression of the parameters.
pub(crate) type Bounds = (bool, Option<(Expr, bool)>, Option<(Expr, bool)>);

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

/// Rejects duplicate `localparam` names and collisions between a `localparam` and a
/// `parameter`, `variable`, or `aliasparam` in the module scope.
fn check_duplicates(module: &ast::Module) -> Result<(), Error> {
    let mut seen_locals: BTreeSet<&str> = BTreeSet::new();
    for local in &module.local_parameters {
        if !seen_locals.insert(local.name.as_str()) {
            return Err(Error::at(
                local.line,
                format!("the localparam {} is declared more than once", local.name),
            ));
        }
        if module
            .parameters
            .iter()
            .any(|parameter| parameter.name == local.name)
        {
            return Err(Error::at(
                local.line,
                format!(
                    "{} is declared as both a parameter and a localparam",
                    local.name
                ),
            ));
        }
        if module
            .variables
            .iter()
            .any(|(variable, _)| *variable == local.name)
        {
            return Err(Error::at(
                local.line,
                format!(
                    "{} is declared as both a localparam and a variable",
                    local.name
                ),
            ));
        }
        if module.aliases.iter().any(|(alias, _)| *alias == local.name) {
            return Err(Error::at(
                local.line,
                format!(
                    "{} is declared as both a localparam and an aliasparam",
                    local.name
                ),
            ));
        }
    }
    Ok(())
}

/// Rejects duplicate `localparam` names and collisions with variables declared in the
/// same named block.
fn check_block_duplicates(
    local_parameters: &[ast::LocalParameter],
    variables: &[(String, Type)],
) -> Result<(), Error> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for local in local_parameters {
        if !seen.insert(local.name.as_str()) {
            return Err(Error::at(
                local.line,
                format!("the localparam {} is declared more than once", local.name),
            ));
        }
        if variables
            .iter()
            .any(|(variable, _)| *variable == local.name)
        {
            return Err(Error::at(
                local.line,
                format!(
                    "{} is declared as both a localparam and a variable",
                    local.name
                ),
            ));
        }
    }
    Ok(())
}

/// Rejects duplicate `localparam` names and collisions with inputs, variables, or the
/// result name of an analog function.
fn check_function_duplicates(function: &ast::Function) -> Result<(), Error> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for local in &function.local_parameters {
        if !seen.insert(local.name.as_str()) {
            return Err(Error::at(
                local.line,
                format!("the localparam {} is declared more than once", local.name),
            ));
        }
        if local.name == function.name
            || function.inputs.contains(&local.name)
            || function
                .variables
                .iter()
                .any(|(variable, _)| *variable == local.name)
        {
            return Err(Error::at(
                local.line,
                format!(
                    "{} is declared as both a localparam and a variable",
                    local.name
                ),
            ));
        }
    }
    Ok(())
}

/// Checks that every `aliasparam` targets a declared `parameter`, not a `localparam` or
/// an undeclared name.
fn check_aliases(module: &ast::Module) -> Result<(), Error> {
    for (alias, parameter) in &module.aliases {
        if module
            .parameters
            .iter()
            .any(|declared| declared.name == *parameter)
        {
            continue;
        }
        if module
            .local_parameters
            .iter()
            .any(|local| local.name == *parameter)
        {
            return Err(Error::at(
                0,
                format!("the alias {alias} targets localparam {parameter}, not a parameter"),
            ));
        }
        return Err(Error::at(
            0,
            format!("the alias {alias} names no parameter, {parameter}"),
        ));
    }
    Ok(())
}

/// Resolved module `parameter` declarations and module-scope `localparam` bindings.
type ResolvedDeclarations = (Vec<Parameter>, Vec<(String, LocalValue)>);

/// Resolves `module`'s `parameter` and `localparam` declarations in source order,
/// returning the resolved parameters and the module-scope `localparam` bindings.
fn resolve_declarations(
    module: &ast::Module,
    (nodes, thermal): (&[String], &[bool]),
    sink: &mut Sink,
) -> Result<ResolvedDeclarations, Error> {
    let given = given_parameters(module);
    let mut seen_parameters: BTreeSet<usize> = BTreeSet::new();
    let mut module_locals: Vec<(String, LocalValue)> = Vec::new();
    let mut resolved_defaults: Vec<Option<Expr>> = vec![None; module.parameters.len()];

    for (position, declaration) in module.declarations.iter().enumerate() {
        let later_locals: Vec<String> = module.declarations[position..]
            .iter()
            .filter_map(|later| match *later {
                ast::Declaration::LocalParameter(index) => {
                    Some(module.local_parameters[index].name.clone())
                }
                ast::Declaration::Parameter(_) => None,
            })
            .collect();
        let bindings: Vec<(String, Binding)> = module_locals
            .iter()
            .map(|(name, value)| (name.clone(), Binding::LocalParameter(*value)))
            .collect();
        match *declaration {
            ast::Declaration::Parameter(index) => {
                let parameter = &module.parameters[index];
                let scope = Scope {
                    module,
                    nodes,
                    thermal,
                    bindings,
                    module_locals: module_locals.clone(),
                    allowed_parameters: Some(seen_parameters.clone()),
                    later_locals,
                    constant_owner: Some(format!("{}'s default", parameter.name)),
                    probes: false,
                    initial: false,
                };
                let mut default = scope.expression(&parameter.default, parameter.line, sink)?;
                if !sink.prelude.is_empty() {
                    return Err(Error::at(
                        parameter.line,
                        format!("{}'s default calls a function", parameter.name),
                    ));
                }
                fold_parameter_default(&mut default, parameter, module, &sink.local_parameters)?;
                resolved_defaults[index] = Some(default);
                seen_parameters.insert(index);
            }
            ast::Declaration::LocalParameter(index) => {
                let local = &module.local_parameters[index];
                let scope = Scope {
                    module,
                    nodes,
                    thermal,
                    bindings,
                    module_locals: module_locals.clone(),
                    allowed_parameters: Some(seen_parameters.clone()),
                    later_locals,
                    constant_owner: Some(local.name.clone()),
                    probes: false,
                    initial: false,
                };
                let value = resolve_local_parameter(&scope, local, sink)?;
                module_locals.push((local.name.clone(), value));
            }
        }
    }

    let parameters = resolve_parameter_ranges(
        module,
        (nodes, thermal),
        &module_locals,
        &given,
        &mut resolved_defaults,
        sink,
    )?;
    Ok((parameters, module_locals))
}

/// Constant-folds a non-trivial parameter default expression when it has no runtime
/// dependence, propagating any integer overflow, divide-by-zero, or non-finite error.
fn fold_parameter_default(
    default: &mut Expr,
    parameter: &ast::Parameter,
    module: &ast::Module,
    local_parameters: &[LocalParameter],
) -> Result<(), Error> {
    if constant(default).is_some() {
        return Ok(());
    }
    let wrap = |message: String| {
        Error::at(
            parameter.line,
            format!("{}'s default {message}", parameter.name),
        )
    };
    if parameter.kind == Type::Integer {
        if let Some(folded) = fold_integer(default, module, local_parameters).map_err(wrap)? {
            *default = Expr::Integer(folded);
        }
    } else if let Some(folded) = fold_real(default, module, local_parameters).map_err(wrap)? {
        *default = Expr::Real(folded);
    }
    Ok(())
}

/// Resolves every parameter's `from`/`exclude` ranges once all module-scope
/// `parameter` and `localparam` declarations are known.
fn resolve_parameter_ranges(
    module: &ast::Module,
    (nodes, thermal): (&[String], &[bool]),
    module_locals: &[(String, LocalValue)],
    given: &BTreeSet<String>,
    resolved_defaults: &mut [Option<Expr>],
    sink: &mut Sink,
) -> Result<Vec<Parameter>, Error> {
    let range_bindings: Vec<(String, Binding)> = module_locals
        .iter()
        .map(|(name, value)| (name.clone(), Binding::LocalParameter(*value)))
        .collect();
    let mut parameters: Vec<Parameter> = Vec::with_capacity(module.parameters.len());
    for (index, parameter) in module.parameters.iter().enumerate() {
        let default = resolved_defaults[index].take().ok_or_else(|| {
            Error::at(
                parameter.line,
                format!("{} lacks a default", parameter.name),
            )
        })?;
        let scope = Scope {
            module,
            nodes,
            thermal,
            bindings: range_bindings.clone(),
            module_locals: module_locals.to_vec(),
            allowed_parameters: None,
            later_locals: Vec::new(),
            constant_owner: Some(format!("{}'s range", parameter.name)),
            probes: false,
            initial: false,
        };
        let mut ranges = Vec::new();
        for range in &parameter.ranges {
            let mut bound = |bound: &Option<(Expression, bool)>| -> Result<_, Error> {
                bound
                    .as_ref()
                    .map(|(value, included)| {
                        let mut resolved = scope.expression(value, parameter.line, sink)?;
                        if constant(&resolved).is_none()
                            && let Some(folded) =
                                fold_real(&resolved, module, &sink.local_parameters).map_err(
                                    |message| {
                                        Error::at(
                                            parameter.line,
                                            format!("{}'s range {message}", parameter.name),
                                        )
                                    },
                                )?
                        {
                            resolved = Expr::Real(folded);
                        }
                        Ok((resolved, *included))
                    })
                    .transpose()
            };
            let (lower, upper) = (bound(&range.lower)?, bound(&range.upper)?);
            ranges.push((range.exclude, lower, upper));
        }
        if !sink.prelude.is_empty() {
            return Err(Error::at(
                parameter.line,
                format!("{}'s range calls a function", parameter.name),
            ));
        }
        let mut name = identifier(&parameter.name);
        // Two parameters whose names meet in snake case keep apart.
        if parameters.iter().any(|other| other.name == name) {
            name = format!("{name}_{index}");
        }
        parameters.push(Parameter {
            name,
            external: parameter.name.clone(),
            kind: parameter.kind,
            optional: constant(&default).is_none() || given.contains(&parameter.name),
            default,
            ranges,
            description: parameter.description.clone(),
            units: parameter.units.clone(),
            aliases: module
                .aliases
                .iter()
                .filter(|(_, named)| *named == parameter.name)
                .map(|(alias, _)| alias.clone())
                .collect(),
        });
    }
    Ok(parameters)
}

/// `branches`, each by its Rust name.
fn named_apart(branches: Vec<Branch>) -> Vec<Branch> {
    let names: Vec<String> = branches.iter().map(|branch| branch.name.clone()).collect();
    branches
        .into_iter()
        .zip(rust_names(&names))
        .map(|(branch, name)| Branch { name, ..branch })
        .collect()
}

/// The Rust names of `names`, in snake case, each apart from those before it.
fn rust_names(names: &[String]) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for name in names {
        let unique = apart(&identifier(name), |taken| {
            found.iter().any(|name| name == taken)
        });
        found.push(unique);
    }
    found
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
    pub(crate) value: Expr,
}

/// What a resolved `localparam` evaluates to: either a compile-time constant folded
/// directly into expressions, or an index into [`Sink::local_parameters`].
#[derive(Debug, Clone, Copy, PartialEq)]
enum LocalValue {
    Real(f64),
    Integer(i32),
    Prepared(usize),
}

/// A lexical binding in a [`Scope`]: either a mutable variable or an immutable
/// `localparam`.
#[derive(Debug, Clone, Copy)]
enum Binding {
    Variable(usize),
    LocalParameter(LocalValue),
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
    pub(crate) default: Expr,
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
    pub(crate) body: Vec<Stmt>,
}

/// A module, resolved and analysed.
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
    pub(crate) analog: Vec<Stmt>,
    /// Whether each variable is dynamic.
    pub(crate) dynamic: Vec<bool>,
    /// The probes each variable depends on, through its values.
    pub(crate) probes: Vec<BTreeSet<Probe>>,
}

/// What resolution adds to as it reads: the limited probes, the branches, and the
/// variables, to which named blocks and inlined analog functions add their own.
#[derive(Default)]
struct Sink {
    limits: Vec<Limit>,
    branches: Vec<Branch>,
    local_parameters: Vec<LocalParameter>,
    variables: Vec<(String, Type)>,
    /// The statements the expression being read needs before it: the bodies of the
    /// analog functions it calls, inlined.
    prelude: Vec<Stmt>,
    /// How many calls have been inlined, which numbers their variables.
    calls: usize,
    /// The variables that code under an initial event assigns.
    initial: BTreeSet<usize>,
}

impl Sink {
    /// A new variable of `kind`, named `name` unless that is taken.
    fn variable(&mut self, name: &str, kind: Type) -> usize {
        let unique = apart(&identifier(name), |taken| {
            self.variables.iter().any(|(variable, _)| variable == taken)
        });
        self.variables.push((unique, kind));
        self.variables.len() - 1
    }

    /// A new non-constant `localparam` of `kind`, named `name` unless that is taken.
    fn local_parameter(&mut self, name: &str, kind: Type, value: Expr) -> usize {
        let unique = apart(&identifier(name), |taken| {
            self.local_parameters
                .iter()
                .any(|existing| existing.name == taken)
        });
        self.local_parameters.push(LocalParameter {
            name: unique,
            external: name.to_owned(),
            kind,
            value,
        });
        self.local_parameters.len() - 1
    }
}

/// The names a module's expressions may read, in a scope: the module's, a named
/// block's or an analog function's.
#[derive(Clone)]
struct Scope<'a> {
    module: &'a ast::Module,
    nodes: &'a [String],
    /// Whether each node is thermal rather than electrical.
    thermal: &'a [bool],
    /// Lexical bindings in scope, the innermost last.
    bindings: Vec<(String, Binding)>,
    /// Module-scope `localparam` bindings, available to inlined analog functions.
    module_locals: Vec<(String, LocalValue)>,
    /// Parameters visible when resolving a constant declaration (`None` means all module
    /// parameters are visible).
    allowed_parameters: Option<BTreeSet<usize>>,
    /// Later `localparam` names in the current declaration sequence, for forward
    /// reference diagnostics.
    later_locals: Vec<String>,
    /// Owner label when resolving a constant expression (`Some` rejects variables and
    /// probes).
    constant_owner: Option<String>,
    /// Whether the scope reads probes, as the analog block does and a function does not.
    probes: bool,
    /// Whether the scope lies under an initial event, whose code runs at an analysis's
    /// first step only and so may read only what does not change after it.
    initial: bool,
}

/// The module `module`, resolved and analysed.
///
/// # Errors
///
/// Returns an [`Error`] for a name that is not declared, and for a construct the
/// compiler does not compile yet.
pub(crate) fn resolve(module: &ast::Module) -> Result<Module, Error> {
    let (nodes, thermal) = nodes_of(module)?;
    check_duplicates(module)?;
    check_aliases(module)?;
    let mut sink = initialize_sink(module, &nodes)?;
    let (parameters, module_locals) = resolve_declarations(module, (&nodes, &thermal), &mut sink)?;
    let mut functions = Vec::with_capacity(module.functions.len());
    for function in &module.functions {
        functions.push(resolve_function(
            module,
            (&nodes, &thermal),
            &module_locals,
            &mut sink.local_parameters,
            function,
        )?);
    }
    let mut bindings: Vec<(String, Binding)> = module_locals
        .iter()
        .map(|(name, value)| (name.clone(), Binding::LocalParameter(*value)))
        .collect();
    bindings.extend(
        module
            .variables
            .iter()
            .enumerate()
            .map(|(index, (name, _))| (name.clone(), Binding::Variable(index))),
    );
    let scope = Scope {
        module,
        nodes: &nodes,
        thermal: &thermal,
        bindings,
        module_locals,
        allowed_parameters: None,
        later_locals: Vec::new(),
        constant_owner: None,
        probes: true,
        initial: false,
    };
    let mut analog = scope.statements(&module.analog, &mut sink)?;
    number(&mut analog, &mut 0);
    let (versions, origins) = version(&mut analog, &mut sink.variables);
    let outputs = module
        .outputs
        .iter()
        .map(|(name, _, _)| scope.variable(name, 0).map(|variable| versions[variable]))
        .collect::<Result<_, _>>()?;
    let mut resolved = Module {
        name: module.name.clone(),
        ports: module.ports.len(),
        nodes: rust_names(&nodes),
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
        branches: named_apart(sink.branches),
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
    analyse(&mut resolved);
    if flow_currents(&mut resolved)? {
        analyse(&mut resolved);
    }
    check_reactive_factors(&resolved)?;
    check_initial_variables(&resolved, &sink.initial, &origins)?;
    let analog = std::mem::take(&mut resolved.analog);
    resolved.analog = dynamic_warnings(&resolved, analog, false);
    Ok(resolved)
}

/// Creates a [`Sink`] pre-populated with `module`'s declared variables and named branches.
fn initialize_sink(module: &ast::Module, nodes: &[String]) -> Result<Sink, Error> {
    let mut sink = Sink::default();
    for (name, kind) in &module.variables {
        sink.variable(name, *kind);
    }
    for (name, plus, minus) in &module.branches {
        let plus = node_index(nodes, plus, 0)?;
        let minus = if minus.is_empty() {
            GROUND
        } else {
            node_index(nodes, minus, 0)?
        };
        sink.branches.push(Branch {
            name: name.clone(),
            plus,
            minus,
        });
    }
    Ok(sink)
}

/// Checks that no variable assigned under an initial event depends on the probes.
fn check_initial_variables(
    resolved: &Module,
    initial_variables: &BTreeSet<usize>,
    origins: &[usize],
) -> Result<(), Error> {
    let initial = (0..resolved.variables.len()).find(|&variable| {
        resolved.dynamic[variable] && initial_variables.contains(&origins[variable])
    });
    if let Some(variable) = initial {
        return Err(Error::at(
            0,
            format!(
                "{}, which code under an initial event assigns, depends on the probes, which is not compiled yet",
                resolved.variables[variable].0
            ),
        ));
    }
    Ok(())
}

/// Checks that every expression factored from outside `ddt(...)` (`k * ddt(x)` folded
/// into `ddt(k * x)`) is static, so every reactive contribution conserves charge or flux.
fn check_reactive_factors(module: &Module) -> Result<(), Error> {
    let mut failure = None;
    visit_contributions(&module.analog, &mut |statement| {
        if failure.is_some() {
            return;
        }
        if let Stmt::Contribute {
            reactive_factors, ..
        } = statement
        {
            for (factor, line) in reactive_factors {
                if module.is_dynamic(factor) {
                    failure = Some(Error::at(
                        *line,
                        "ddt scaled by a dynamic expression is not charge-conserving and is not compiled yet"
                            .into(),
                    ));
                    return;
                }
            }
        }
    });
    if let Some(error) = failure {
        return Err(error);
    }
    Ok(())
}

/// `statements` with each warning marked dynamic that depends on the probes, through
/// its values or a condition it lies under, `dynamic` whether they lie under one.
fn dynamic_warnings(module: &Module, statements: Vec<Stmt>, dynamic: bool) -> Vec<Stmt> {
    statements
        .into_iter()
        .map(|statement| match statement {
            Stmt::Warn(message, arguments, _) => {
                let depends =
                    dynamic || arguments.iter().any(|argument| module.is_dynamic(argument));
                Stmt::Warn(message, arguments, depends)
            }
            Stmt::If(id, condition, then, otherwise) => {
                let inner = dynamic || module.is_dynamic(&condition);
                let then = dynamic_warnings(module, then, inner);
                let otherwise = dynamic_warnings(module, otherwise, inner);
                Stmt::If(id, condition, then, otherwise)
            }
            Stmt::While(condition, body) => {
                let inner = dynamic || module.is_dynamic(&condition);
                Stmt::While(condition, dynamic_warnings(module, body, inner))
            }
            other => other,
        })
        .collect()
}

/// A contribution to a branch's flow or potential, and the conditions it lies under:
/// those that shape the kernel, each by its number and whether it holds, and whether
/// any other does.
struct Site {
    branch: usize,
    potential: bool,
    resistive: Option<Expr>,
    reactive: bool,
    shape: Vec<(usize, bool)>,
    dynamic: bool,
}

/// The sites of `statements`, under `shape` and `dynamic`.
fn sites(
    module: &Module,
    statements: &[Stmt],
    (shape, dynamic): (&[(usize, bool)], bool),
    found: &mut Vec<Site>,
) {
    for statement in statements {
        match statement {
            Stmt::Contribute {
                branch,
                potential,
                resistive,
                reactive,
                ..
            } if resistive.is_some() || reactive.is_some() => found.push(Site {
                branch: *branch,
                potential: *potential,
                resistive: resistive.clone(),
                reactive: reactive.is_some(),
                shape: shape.to_vec(),
                dynamic,
            }),
            Stmt::If(id, condition, then, otherwise) => {
                if module.is_dynamic(condition) {
                    sites(module, then, (shape, true), found);
                    sites(module, otherwise, (shape, true), found);
                } else {
                    for (inner, holds) in [(then, true), (otherwise, false)] {
                        let mut deeper = shape.to_vec();
                        deeper.push((*id, holds));
                        sites(module, inner, (&deeper, dynamic), found);
                    }
                }
            }
            Stmt::While(_, body) => sites(module, body, (shape, true), found),
            Stmt::Assign(..) | Stmt::Contribute { .. } | Stmt::Fail(..) | Stmt::Warn(..) => {}
        }
    }
}

/// The truth of the shape's conditions `shape`, all of them.
fn guard(shape: &[(usize, bool)]) -> Option<Expr> {
    shape
        .iter()
        .map(|&(id, holds)| {
            if holds {
                Expr::Shape(id)
            } else {
                Expr::Unary("!", Box::new(Expr::Shape(id)))
            }
        })
        .reduce(|left, right| Expr::Binary("&&", Box::new(left), Box::new(right)))
}

/// The current of `branch`, as a probe reads it: the sum of its flow's contributions,
/// each where the shape makes it, and its own unknown where the shape makes it hold a
/// potential; `None` when it carries no flow, so that the probe reads its unknown.
fn flow_current(module: &Module, branch: usize, found: &[Site]) -> Result<Option<Expr>, Error> {
    let name = &module.branches[branch].name;
    let of = |potential: bool| {
        found
            .iter()
            .filter(move |site| site.branch == branch && site.potential == potential)
    };
    if of(false).next().is_none() {
        return Ok(None);
    }
    let mut flow: Option<Expr> = None;
    for site in of(false) {
        if site.dynamic || site.reactive {
            return Err(Error::at(
                0,
                format!(
                    "the current of {name}, a flow under a condition on the probes or of a charge, is not compiled yet"
                ),
            ));
        }
        let value = site.resistive.clone().unwrap_or(Expr::Real(0.0));
        let term = match guard(&site.shape) {
            Some(condition) => Expr::Conditional(
                Box::new(condition),
                Box::new(value),
                Box::new(Expr::Real(0.0)),
            ),
            None => value,
        };
        flow = Some(match flow {
            Some(sum) => Expr::Binary("+", Box::new(sum), Box::new(term)),
            None => term,
        });
    }
    let flow = flow.unwrap_or(Expr::Real(0.0));
    let held: Option<Option<Expr>> =
        of(true)
            .map(|site| guard(&site.shape))
            .reduce(|left, right| match (left, right) {
                (Some(left), Some(right)) => {
                    Some(Expr::Binary("||", Box::new(left), Box::new(right)))
                }
                _ => None,
            });
    Ok(Some(match held {
        None => flow,
        Some(None) => Expr::Probe(Probe::Current(branch)),
        Some(Some(condition)) => Expr::Conditional(
            Box::new(condition),
            Box::new(Expr::Probe(Probe::Current(branch))),
            Box::new(flow),
        ),
    }))
}

/// Replaces each probe of the current of a branch that carries a flow with what is
/// contributed to it, and returns whether it replaced any. The contributions come later
/// in the analog block than the probe may, so nothing they read may be assigned after it.
fn flow_currents(module: &mut Module) -> Result<bool, Error> {
    let noisy = noisy_currents(module)?;
    let mut found = Vec::new();
    sites(module, &module.analog, (&[], false), &mut found);
    let read = probed_currents(module);
    let mut replacements = BTreeMap::new();
    for branch in read {
        if let Some(current) = flow_current(module, branch, &found)?
            && current != Expr::Probe(Probe::Current(branch))
        {
            replacements.insert(branch, current);
        }
    }
    if replacements.is_empty() {
        return Ok(noisy);
    }
    // Where each variable is last assigned, and where each statement lies, in order.
    let mut last = vec![0_usize; module.variables.len()];
    let mut position = 0;
    let mut probed: Vec<(usize, usize)> = Vec::new();
    order(&module.analog, &mut position, &mut last, &mut |at, expr| {
        expr.walk(&mut |inner| {
            if let Expr::Probe(Probe::Current(branch)) = inner {
                probed.push((*branch, at));
            }
        });
    });
    for &(branch, at) in &probed {
        let Some(current) = replacements.get(&branch) else {
            continue;
        };
        let mut late = None;
        current.walk(&mut |inner| {
            if let Expr::Variable(variable) = inner
                && last[*variable] >= at
            {
                late = Some(*variable);
            }
        });
        if let Some(variable) = late {
            return Err(Error::at(
                0,
                format!(
                    "the current of {} is read before {}, which its flow reads, is assigned",
                    module.branches[branch].name, module.variables[variable].0
                ),
            ));
        }
    }
    for statement in &mut module.analog {
        statement.exprs_mut(&mut |expr| replace_currents(expr, &replacements));
    }
    for limit in &mut module.limits {
        for argument in &mut limit.arguments {
            replace_currents(argument, &replacements);
        }
    }
    Ok(true)
}

/// The branches whose currents `module` probes.
pub(crate) fn probed_currents(module: &Module) -> BTreeSet<usize> {
    let mut read = BTreeSet::new();
    let mut visit = |expr: &Expr| {
        expr.walk(&mut |inner| {
            if let Expr::Probe(Probe::Current(branch)) = inner {
                read.insert(*branch);
            }
        });
    };
    for statement in &module.analog {
        statement.walk(&mut visit);
    }
    for limit in &module.limits {
        limit.arguments.iter().for_each(&mut visit);
    }
    read
}

/// Finds each noisy branch whose current is probed, verifying none holds a potential,
/// and creates a current-unknown node and auxiliary equation branch for each.
fn collect_noisy_branches(module: &mut Module) -> Result<BTreeMap<usize, (usize, usize)>, Error> {
    let mut noisy = BTreeSet::new();
    visit_contributions(&module.analog, &mut |statement| {
        if let Stmt::Contribute {
            branch,
            potential: false,
            noise,
            ..
        } = statement
            && !noise.is_empty()
        {
            noisy.insert(*branch);
        }
    });
    let read = probed_currents(module);
    let mut rewritten = BTreeMap::new();
    for &branch in noisy.intersection(&read) {
        let name = module.branches[branch].name.clone();
        let mut potential = false;
        visit_contributions(&module.analog, &mut |statement| {
            if let Stmt::Contribute {
                branch: found,
                potential: true,
                ..
            } = statement
            {
                potential |= *found == branch;
            }
        });
        if potential {
            return Err(Error::at(
                0,
                format!(
                    "the current of {name}, which carries noise and holds a potential, is not compiled yet"
                ),
            ));
        }
        module.nodes.push(name.clone());
        module.labels.push(name.clone());
        module.natures.push(Nature::Current);
        let node = module.nodes.len() - 1;
        module.branches.push(Branch {
            name: format!("{name}_ground"),
            plus: node,
            minus: GROUND,
        });
        rewritten.insert(branch, (node, module.branches.len() - 1));
    }
    Ok(rewritten)
}

/// Gives each branch whose current is probed and whose flow carries noise its current as
/// an unknown, so that what reads the current reads its noise too, as PSP's drain does
/// its induced gate noise's: a node of the branch's name, of a current's nature, whose
/// equation is its value less the branch's flow, the noise entering it, and through which
/// the branch's flow then passes. Returns whether it gave any.
///
/// The noise enters the equation as the branch's flow does, with the opposite sign, which
/// changes no response's power or correlation: a noise source and its negative are the
/// same random process.
fn noisy_currents(module: &mut Module) -> Result<bool, Error> {
    let rewritten = collect_noisy_branches(module)?;
    if rewritten.is_empty() {
        return Ok(false);
    }
    let negate = |expr: &mut Option<Expr>| {
        if let Some(value) = expr.take() {
            *expr = Some(Expr::Unary("-", Box::new(value)));
        }
    };
    let mut analog = std::mem::take(&mut module.analog);
    let mut contributions = |statement: &mut Stmt| {
        if let Stmt::Contribute {
            branch,
            potential: false,
            resistive,
            reactive,
            ..
        } = statement
            && let Some(&(_, equation)) = rewritten.get(branch)
        {
            *branch = equation;
            negate(resistive);
            negate(reactive);
        }
    };
    visit_contributions_mut(&mut analog, &mut contributions);
    let replacements: BTreeMap<usize, Expr> = rewritten
        .iter()
        .map(|(&branch, &(node, _))| (branch, Expr::Probe(Probe::Voltage(node, GROUND))))
        .collect();
    for statement in &mut analog {
        statement.exprs_mut(&mut |expr| replace_currents(expr, &replacements));
    }
    for limit in &mut module.limits {
        for argument in &mut limit.arguments {
            replace_currents(argument, &replacements);
        }
    }
    for (&branch, &(node, equation)) in &rewritten {
        let current = Expr::Probe(Probe::Voltage(node, GROUND));
        for target in [branch, equation] {
            analog.push(Stmt::Contribute {
                branch: target,
                potential: false,
                resistive: Some(current.clone()),
                reactive: None,
                reactive_factors: Vec::new(),
                noise: Vec::new(),
            });
        }
    }
    module.analog = analog;
    Ok(true)
}

/// Calls `visit` on each contribution of `statements`.
pub(crate) fn visit_contributions(statements: &[Stmt], visit: &mut impl FnMut(&Stmt)) {
    for statement in statements {
        match statement {
            Stmt::Contribute { .. } => visit(statement),
            Stmt::If(_, _, then, otherwise) => {
                visit_contributions(then, visit);
                visit_contributions(otherwise, visit);
            }
            Stmt::While(_, body) => visit_contributions(body, visit),
            Stmt::Assign(..) | Stmt::Fail(..) | Stmt::Warn(..) => {}
        }
    }
}

/// Calls `visit` on each contribution of `statements`, to change it.
fn visit_contributions_mut(statements: &mut [Stmt], visit: &mut impl FnMut(&mut Stmt)) {
    for statement in statements {
        match statement {
            Stmt::Contribute { .. } => visit(statement),
            Stmt::If(_, _, then, otherwise) => {
                visit_contributions_mut(then, visit);
                visit_contributions_mut(otherwise, visit);
            }
            Stmt::While(_, body) => visit_contributions_mut(body, visit),
            Stmt::Assign(..) | Stmt::Fail(..) | Stmt::Warn(..) => {}
        }
    }
}

/// Numbers each statement of `statements` from `position` on, in order, records where
/// each variable is last assigned in `last`, and calls `visit` on each expression with
/// the number of its statement.
fn order(
    statements: &[Stmt],
    position: &mut usize,
    last: &mut [usize],
    visit: &mut impl FnMut(usize, &Expr),
) {
    for statement in statements {
        *position += 1;
        let at = *position;
        match statement {
            Stmt::Assign(variable, value) => {
                visit(at, value);
                last[*variable] = at;
            }
            Stmt::If(_, condition, then, otherwise) => {
                visit(at, condition);
                order(then, position, last, visit);
                order(otherwise, position, last, visit);
            }
            Stmt::While(condition, body) => {
                visit(at, condition);
                order(body, position, last, visit);
            }
            Stmt::Contribute { .. } | Stmt::Fail(..) | Stmt::Warn(..) => {
                statement.walk(&mut |expr| visit(at, expr));
            }
        }
    }
}

/// `expr` with each probe of a current that `replacements` holds replaced, the
/// replacements themselves left as they are.
fn replace_currents(expr: &mut Expr, replacements: &BTreeMap<usize, Expr>) {
    if let Expr::Probe(Probe::Current(branch)) = expr
        && let Some(current) = replacements.get(branch)
    {
        *expr = current.clone();
        return;
    }
    match expr {
        Expr::Unary(_, operand) | Expr::Ddx(operand, _) => replace_currents(operand, replacements),
        Expr::Binary(_, left, right) => {
            replace_currents(left, replacements);
            replace_currents(right, replacements);
        }
        Expr::Conditional(condition, then, otherwise) => {
            for inner in [condition, then, otherwise] {
                replace_currents(inner, replacements);
            }
        }
        Expr::Call(_, arguments) => {
            for argument in arguments {
                replace_currents(argument, replacements);
            }
        }
        _ => {}
    }
}

/// The nodes of `module`, its ports first, and whether each is thermal.
fn nodes_of(module: &ast::Module) -> Result<(Vec<String>, Vec<bool>), Error> {
    let mut nodes: Vec<String> = module.ports.clone();
    for (node, _) in &module.nodes {
        if !nodes.contains(node) {
            nodes.push(node.clone());
        }
    }
    for port in &module.ports {
        if !module.nodes.iter().any(|(node, _)| node == port) {
            return Err(Error::at(0, format!("the port {port} has no discipline")));
        }
    }
    let thermal: Vec<bool> = nodes
        .iter()
        .map(|node| {
            module
                .nodes
                .iter()
                .any(|(name, discipline)| name == node && discipline == "thermal")
        })
        .collect();
    Ok((nodes, thermal))
}

/// Resolves one `localparam` declaration: checks type compatibility, constant-folds
/// compile-time values into `LocalValue::Real` or `LocalValue::Integer`, and records
/// parameter- or temperature-dependent ones in `sink.local_parameters`.
fn resolve_local_parameter(
    scope: &Scope<'_>,
    local: &ast::LocalParameter,
    sink: &mut Sink,
) -> Result<LocalValue, Error> {
    let value = scope.expression(&local.value, local.line, sink)?;
    if !sink.prelude.is_empty() {
        return Err(Error::at(
            local.line,
            format!(
                "{} calls an analog function, which is not compiled yet",
                local.name
            ),
        ));
    }
    let inferred = expr_type(&value, scope.module, &sink.local_parameters);
    let kind = match local.kind {
        Some(Type::Integer) => {
            if inferred == Type::Real {
                return Err(Error::at(
                    local.line,
                    format!(
                        "the integer localparam {} has a real initializer",
                        local.name
                    ),
                ));
            }
            Type::Integer
        }
        Some(Type::Real) => Type::Real,
        None => inferred,
    };
    if kind == Type::Integer {
        match fold_integer(&value, scope.module, &sink.local_parameters) {
            Ok(Some(folded)) => Ok(LocalValue::Integer(folded)),
            Ok(None) => Ok(LocalValue::Prepared(sink.local_parameter(
                &local.name,
                Type::Integer,
                value,
            ))),
            Err(message) => Err(Error::at(
                local.line,
                format!("the integer localparam {} {message}", local.name),
            )),
        }
    } else {
        match fold_real(&value, scope.module, &sink.local_parameters) {
            Ok(Some(folded)) => Ok(LocalValue::Real(folded)),
            Ok(None) => Ok(LocalValue::Prepared(sink.local_parameter(
                &local.name,
                Type::Real,
                value,
            ))),
            Err(message) => Err(Error::at(
                local.line,
                format!("the localparam {} {message}", local.name),
            )),
        }
    }
}

/// The Verilog-A numeric type of `expr` (boolean expressions count as `Type::Integer`).
fn expr_type(expr: &Expr, module: &ast::Module, local_parameters: &[LocalParameter]) -> Type {
    match expr {
        Expr::Integer(_)
        | Expr::Given(_)
        | Expr::Shape(_)
        | Expr::Unary("!", _)
        | Expr::Binary("<" | "<=" | ">" | ">=" | "==" | "!=" | "&&" | "||", ..) => Type::Integer,
        Expr::Real(_)
        | Expr::Variable(_)
        | Expr::Temperature
        | Expr::Gmin
        | Expr::SmallResistance
        | Expr::Probe(_)
        | Expr::Ddx(..)
        | Expr::Call(..) => Type::Real,
        Expr::Parameter(index) => module
            .parameters
            .get(*index)
            .map_or(Type::Real, |parameter| parameter.kind),
        Expr::LocalParameter(index) => local_parameters
            .get(*index)
            .map_or(Type::Real, |local| local.kind),
        Expr::Unary(_, operand) => expr_type(operand, module, local_parameters),
        Expr::Binary(_, left, right) | Expr::Conditional(_, left, right) => {
            if expr_type(left, module, local_parameters) == Type::Real
                || expr_type(right, module, local_parameters) == Type::Real
            {
                Type::Real
            } else {
                Type::Integer
            }
        }
    }
}

/// Evaluates a built-in function on constant `f64` arguments.
fn eval_builtin(builtin: Builtin, arguments: &[f64]) -> Option<f64> {
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

/// Checks that `value` is finite, returning an error phrase otherwise.
fn finite(value: f64) -> Result<Option<f64>, String> {
    if value.is_finite() {
        Ok(Some(value))
    } else {
        Err("evaluates to a non-finite value".to_owned())
    }
}

/// Evaluates `expr` as a compile-time truth value (`Ok(None)` when non-constant).
fn fold_truth(
    expr: &Expr,
    module: &ast::Module,
    local_parameters: &[LocalParameter],
) -> Result<Option<bool>, String> {
    match expr {
        Expr::Integer(value) => Ok(Some(*value != 0)),
        Expr::Real(value) => Ok(Some(*value != 0.0)),
        Expr::Unary("!", operand) => {
            Ok(fold_truth(operand, module, local_parameters)?.map(|truth| !truth))
        }
        Expr::Binary("&&", left, right) => match fold_truth(left, module, local_parameters)? {
            Some(false) => Ok(Some(false)),
            Some(true) => fold_truth(right, module, local_parameters),
            None => Ok(None),
        },
        Expr::Binary("||", left, right) => match fold_truth(left, module, local_parameters)? {
            Some(true) => Ok(Some(true)),
            Some(false) => fold_truth(right, module, local_parameters),
            None => Ok(None),
        },
        Expr::Binary(operator @ ("<" | "<=" | ">" | ">=" | "==" | "!="), left, right) => {
            let (Some(left), Some(right)) = (
                fold_real(left, module, local_parameters)?,
                fold_real(right, module, local_parameters)?,
            ) else {
                return Ok(None);
            };
            let order = left.partial_cmp(&right);
            Ok(Some(match *operator {
                "==" => order == Some(Ordering::Equal),
                "!=" => order != Some(Ordering::Equal),
                "<" => order == Some(Ordering::Less),
                "<=" => matches!(order, Some(Ordering::Less | Ordering::Equal)),
                ">" => order == Some(Ordering::Greater),
                _ => matches!(order, Some(Ordering::Greater | Ordering::Equal)),
            }))
        }
        _ => Ok(fold_real(expr, module, local_parameters)?.map(|value| value != 0.0)),
    }
}

/// Evaluates `expr` as a compile-time integer constant (`Ok(None)` when non-constant).
fn fold_integer(
    expr: &Expr,
    module: &ast::Module,
    local_parameters: &[LocalParameter],
) -> Result<Option<i32>, String> {
    match expr {
        Expr::Integer(value) => Ok(Some(*value)),
        Expr::Unary("-", operand) => {
            let Some(value) = fold_integer(operand, module, local_parameters)? else {
                return Ok(None);
            };
            value
                .checked_neg()
                .map(Some)
                .ok_or_else(|| "overflows an integer".to_owned())
        }
        Expr::Unary("!", operand) => {
            Ok(fold_truth(operand, module, local_parameters)?.map(|truth| i32::from(!truth)))
        }
        Expr::Binary("&&" | "||" | "<" | "<=" | ">" | ">=" | "==" | "!=", ..) => {
            Ok(fold_truth(expr, module, local_parameters)?.map(i32::from))
        }
        Expr::Binary(operator @ ("+" | "-" | "*" | "/"), left, right) => {
            let (Some(left), Some(right)) = (
                fold_integer(left, module, local_parameters)?,
                fold_integer(right, module, local_parameters)?,
            ) else {
                return Ok(None);
            };
            if *operator == "/" && right == 0 {
                return Err("divides by zero".to_owned());
            }
            let value = match *operator {
                "+" => left.checked_add(right),
                "-" => left.checked_sub(right),
                "*" => left.checked_mul(right),
                _ => left.checked_div(right),
            };
            value
                .map(Some)
                .ok_or_else(|| "overflows an integer".to_owned())
        }
        Expr::Conditional(condition, then, otherwise) => {
            match fold_truth(condition, module, local_parameters)? {
                Some(true) => fold_integer(then, module, local_parameters),
                Some(false) => fold_integer(otherwise, module, local_parameters),
                None => Ok(None),
            }
        }
        _ => Ok(None),
    }
}

/// Evaluates `expr` as a compile-time real constant (`Ok(None)` when non-constant).
fn fold_real(
    expr: &Expr,
    module: &ast::Module,
    local_parameters: &[LocalParameter],
) -> Result<Option<f64>, String> {
    match expr {
        Expr::Real(value) => finite(*value),
        Expr::Integer(value) => Ok(Some(f64::from(*value))),
        Expr::Unary("-", operand) => {
            let Some(value) = fold_real(operand, module, local_parameters)? else {
                return Ok(None);
            };
            finite(-value)
        }
        Expr::Unary("!", _)
        | Expr::Binary("&&" | "||" | "<" | "<=" | ">" | ">=" | "==" | "!=", ..) => {
            Ok(fold_truth(expr, module, local_parameters)?.map(|truth| f64::from(u8::from(truth))))
        }
        Expr::Binary(operator @ ("+" | "-" | "*" | "/"), left, right) => {
            if *operator == "/"
                && expr_type(left, module, local_parameters) == Type::Integer
                && expr_type(right, module, local_parameters) == Type::Integer
            {
                return Ok(fold_integer(expr, module, local_parameters)?.map(f64::from));
            }
            let (Some(left), Some(right)) = (
                fold_real(left, module, local_parameters)?,
                fold_real(right, module, local_parameters)?,
            ) else {
                return Ok(None);
            };
            finite(fold(operator, left, right))
        }
        Expr::Conditional(condition, then, otherwise) => {
            match fold_truth(condition, module, local_parameters)? {
                Some(true) => fold_real(then, module, local_parameters),
                Some(false) => fold_real(otherwise, module, local_parameters),
                None => Ok(None),
            }
        }
        Expr::Call(builtin, arguments) => {
            let mut evaluated = Vec::with_capacity(arguments.len());
            for argument in arguments {
                let Some(value) = fold_real(argument, module, local_parameters)? else {
                    return Ok(None);
                };
                evaluated.push(value);
            }
            let Some(value) = eval_builtin(*builtin, &evaluated) else {
                return Ok(None);
            };
            finite(value)
        }
        _ => Ok(None),
    }
}

/// The index of the node `name` among `nodes`.
fn node_index(nodes: &[String], name: &str, line: usize) -> Result<usize, Error> {
    nodes
        .iter()
        .position(|node| node == name)
        .ok_or_else(|| Error::at(line, format!("{name} is not a node")))
}

/// The variables of an analog function: its inputs first, then its locals, then its
/// result, which bears its name.
fn function_variables(function: &ast::Function) -> Vec<(String, Type)> {
    let mut variables: Vec<(String, Type)> = Vec::new();
    for input in &function.inputs {
        let kind = function
            .variables
            .iter()
            .find(|(name, _)| name == input)
            .map_or(Type::Real, |&(_, kind)| kind);
        variables.push((input.clone(), kind));
    }
    for variable in &function.variables {
        if !function.inputs.contains(&variable.0) {
            variables.push(variable.clone());
        }
    }
    variables.push((function.name.clone(), Type::Real));
    variables
}

/// An analog function that limits a probe, resolved in a scope of its own.
fn resolve_function(
    module: &ast::Module,
    (nodes, thermal): (&[String], &[bool]),
    module_locals: &[(String, LocalValue)],
    local_parameters: &mut Vec<LocalParameter>,
    function: &ast::Function,
) -> Result<Function, Error> {
    check_function_duplicates(function)?;
    let variables = function_variables(function);
    let mut bindings: Vec<(String, Binding)> = module_locals
        .iter()
        .map(|(name, value)| (name.clone(), Binding::LocalParameter(*value)))
        .collect();
    bindings.extend(
        variables
            .iter()
            .enumerate()
            .map(|(index, (name, _))| (name.clone(), Binding::Variable(index))),
    );
    let mut sink = Sink {
        variables,
        local_parameters: std::mem::take(local_parameters),
        ..Sink::default()
    };
    let mut scope = Scope {
        module,
        nodes,
        thermal,
        bindings,
        module_locals: module_locals.to_vec(),
        allowed_parameters: None,
        later_locals: Vec::new(),
        constant_owner: None,
        probes: false,
        initial: false,
    };
    let resolved = resolve_scoped_locals(&mut scope, &function.local_parameters, &mut sink)
        .and_then(|()| scope.statements(&function.body, &mut sink));
    *local_parameters = sink.local_parameters;
    let body = resolved?;
    Ok(Function {
        name: function.name.clone(),
        inputs: function.inputs.len(),
        variables: sink.variables,
        body,
    })
}

/// Resolves a sequence of `localparam` declarations inside a named block or analog
/// function, appending each binding into `scope`.
fn resolve_scoped_locals(
    scope: &mut Scope<'_>,
    local_parameters: &[ast::LocalParameter],
    sink: &mut Sink,
) -> Result<(), Error> {
    for (position, local) in local_parameters.iter().enumerate() {
        let later_locals = local_parameters[position..]
            .iter()
            .map(|later| later.name.clone())
            .collect();
        let constant_scope = Scope {
            later_locals,
            constant_owner: Some(local.name.clone()),
            probes: false,
            initial: false,
            ..scope.clone()
        };
        let value = resolve_local_parameter(&constant_scope, local, sink)?;
        scope
            .bindings
            .push((local.name.clone(), Binding::LocalParameter(value)));
    }
    Ok(())
}

/// The names of the parameters `$param_given` reads across `module`.
fn given_parameters(module: &ast::Module) -> BTreeSet<String> {
    let mut given = BTreeSet::new();
    let mut visit = |expression: &Expression| {
        expression.walk(&mut |inner| {
            if let Expression::Call(name, arguments) = inner
                && name == "$param_given"
                && let [Expression::Name(parameter)] = arguments.as_slice()
            {
                given.insert(parameter.clone());
            }
        });
    };
    module.analog.walk(&mut visit);
    for local in &module.local_parameters {
        visit(&local.value);
    }
    for function in &module.functions {
        for local in &function.local_parameters {
            visit(&local.value);
        }
        function.body.walk(&mut visit);
    }
    for parameter in &module.parameters {
        visit(&parameter.default);
        for range in &parameter.ranges {
            if let Some((lower, _)) = &range.lower {
                visit(lower);
            }
            if let Some((upper, _)) = &range.upper {
                visit(upper);
            }
        }
    }
    given
}

impl ast::Statement {
    /// Calls `visit` on each expression of the statement and of those within it.
    fn walk(&self, visit: &mut impl FnMut(&Expression)) {
        match self {
            Self::Block(statements) => {
                for inner in statements {
                    inner.walk(visit);
                }
            }
            Self::Named {
                local_parameters,
                statements,
                ..
            } => {
                for local in local_parameters {
                    visit(&local.value);
                }
                for inner in statements {
                    inner.walk(visit);
                }
            }
            Self::Assign { value, .. } | Self::Contribute { value, .. } => visit(value),
            Self::If {
                condition,
                then,
                otherwise,
            } => {
                visit(condition);
                then.walk(visit);
                if let Some(otherwise) = otherwise {
                    otherwise.walk(visit);
                }
            }
            Self::While { condition, body } => {
                visit(condition);
                body.walk(visit);
            }
            Self::Initial { body, .. } => body.walk(visit),
            Self::Task { arguments, .. } => arguments.iter().for_each(visit),
        }
    }
}

impl Expression {
    /// Calls `visit` on the expression and on each within it.
    fn walk(&self, visit: &mut impl FnMut(&Self)) {
        visit(self);
        match self {
            Self::Call(_, arguments) => arguments.iter().for_each(|inner| inner.walk(visit)),
            Self::Unary(_, operand) => operand.walk(visit),
            Self::Binary(_, left, right) => {
                left.walk(visit);
                right.walk(visit);
            }
            Self::Conditional(condition, then, otherwise) => {
                condition.walk(visit);
                then.walk(visit);
                otherwise.walk(visit);
            }
            Self::Real(_) | Self::Integer(_) | Self::Text(_) | Self::Name(_) => {}
        }
    }
}

impl Expr {
    /// Calls `visit` on the expression and on each within it.
    pub(crate) fn walk(&self, visit: &mut impl FnMut(&Self)) {
        visit(self);
        match self {
            Self::Call(_, arguments) => arguments.iter().for_each(|inner| inner.walk(visit)),
            Self::Unary(_, operand) | Self::Ddx(operand, _) => operand.walk(visit),
            Self::Binary(_, left, right) => {
                left.walk(visit);
                right.walk(visit);
            }
            Self::Conditional(condition, then, otherwise) => {
                condition.walk(visit);
                then.walk(visit);
                otherwise.walk(visit);
            }
            Self::Real(_)
            | Self::Integer(_)
            | Self::Parameter(_)
            | Self::LocalParameter(_)
            | Self::Given(_)
            | Self::Variable(_)
            | Self::Temperature
            | Self::Gmin
            | Self::SmallResistance
            | Self::Probe(_)
            | Self::Shape(_) => {}
        }
    }
}

impl Stmt {
    /// Calls `visit` on each variable the statements within the statement assign, to
    /// change, the statement's own included.
    fn targets_mut(&mut self, visit: &mut impl FnMut(&mut usize)) {
        match self {
            Self::Assign(variable, _) => visit(variable),
            Self::If(_, _, then, otherwise) => {
                for inner in then.iter_mut().chain(otherwise.iter_mut()) {
                    inner.targets_mut(visit);
                }
            }
            Self::While(_, body) => {
                for inner in body {
                    inner.targets_mut(visit);
                }
            }
            Self::Contribute { .. } | Self::Fail(..) | Self::Warn(..) => {}
        }
    }

    /// Calls `visit` on each expression the statement reads, and on those of the
    /// statements within it, each to change.
    fn exprs_mut(&mut self, visit: &mut impl FnMut(&mut Expr)) {
        match self {
            Self::Assign(_, value) => visit(value),
            Self::Contribute {
                resistive,
                reactive,
                reactive_factors,
                noise,
                ..
            } => {
                for expr in resistive.iter_mut().chain(reactive.iter_mut()) {
                    visit(expr);
                }
                for (factor, _) in reactive_factors {
                    visit(factor);
                }
                for noise in noise {
                    visit(&mut noise.power);
                    if let Some(exponent) = &mut noise.exponent {
                        visit(exponent);
                    }
                }
            }
            Self::If(_, condition, then, otherwise) => {
                visit(condition);
                for inner in then.iter_mut().chain(otherwise.iter_mut()) {
                    inner.exprs_mut(visit);
                }
            }
            Self::While(condition, body) => {
                visit(condition);
                for inner in body {
                    inner.exprs_mut(visit);
                }
            }
            Self::Fail(_, arguments) | Self::Warn(_, arguments, _) => {
                for argument in arguments {
                    visit(argument);
                }
            }
        }
    }

    /// Calls `visit` on each expression the statement reads, and on those of the
    /// statements within it.
    pub(crate) fn walk(&self, visit: &mut impl FnMut(&Expr)) {
        match self {
            Self::Assign(_, value) => visit(value),
            Self::Contribute {
                resistive,
                reactive,
                noise,
                ..
            } => {
                for expr in resistive.iter().chain(reactive) {
                    visit(expr);
                }
                for noise in noise {
                    visit(&noise.power);
                    if let Some(exponent) = &noise.exponent {
                        visit(exponent);
                    }
                }
            }
            Self::If(_, condition, then, otherwise) => {
                visit(condition);
                for inner in then.iter().chain(otherwise) {
                    inner.walk(visit);
                }
            }
            Self::While(condition, body) => {
                visit(condition);
                for inner in body {
                    inner.walk(visit);
                }
            }
            Self::Fail(_, arguments) | Self::Warn(_, arguments, _) => {
                arguments.iter().for_each(visit);
            }
        }
    }
}

/// `left operator right`, of two constants, as Rust computes it at run time.
pub(crate) fn fold(operator: &str, left: f64, right: f64) -> f64 {
    match operator {
        "+" => left + right,
        "-" => left - right,
        "*" => left * right,
        _ => left / right,
    }
}

/// The value of `expr` if it is a constant.
pub(crate) fn constant(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Real(value) => Some(*value),
        Expr::Integer(value) => Some(f64::from(*value)),
        Expr::Unary("-", operand) => constant(operand).map(|value| -value),
        Expr::Binary(operator @ ("+" | "-" | "*" | "/"), left, right) => {
            Some(fold(operator, constant(left)?, constant(right)?))
        }
        _ => None,
    }
}

impl Scope<'_> {
    #[expect(
        clippy::too_many_lines,
        reason = "one arm per statement of the language, which read best side by side"
    )]
    fn statements(&self, statement: &ast::Statement, sink: &mut Sink) -> Result<Vec<Stmt>, Error> {
        // The statements an expression's inlined calls need come before the statement.
        let before = |sink: &mut Sink, statement: Stmt| {
            let mut resolved = std::mem::take(&mut sink.prelude);
            resolved.push(statement);
            resolved
        };
        Ok(match statement {
            ast::Statement::Block(statements) => {
                let mut resolved = Vec::new();
                for inner in statements {
                    resolved.extend(self.statements(inner, sink)?);
                }
                resolved
            }
            ast::Statement::Named {
                local_parameters,
                variables,
                statements,
                ..
            } => {
                check_block_duplicates(local_parameters, variables)?;
                let mut inner = self.clone();
                for (variable, kind) in variables {
                    let index = sink.variable(variable, *kind);
                    inner
                        .bindings
                        .push((variable.clone(), Binding::Variable(index)));
                }
                resolve_scoped_locals(&mut inner, local_parameters, sink)?;
                let mut resolved = Vec::new();
                for statement in statements {
                    resolved.extend(inner.statements(statement, sink)?);
                }
                resolved
            }
            ast::Statement::Assign {
                target,
                value,
                line,
            } => {
                let variable = self.variable(target, *line)?;
                let value = self.expression(value, *line, sink)?;
                if self.initial {
                    sink.initial.insert(variable);
                }
                before(sink, Stmt::Assign(variable, value))
            }
            ast::Statement::Contribute {
                nature,
                operands,
                value,
                line,
            } => {
                if self.initial {
                    return Err(Error::at(
                        *line,
                        "a contribution under an initial event is not compiled yet".into(),
                    ));
                }
                let contribution = self.contribution(nature, operands, value, *line, sink)?;
                before(sink, contribution)
            }
            ast::Statement::Initial { body, .. } => {
                let inner = Scope {
                    probes: false,
                    initial: true,
                    ..self.clone()
                };
                inner.statements(body, sink)?
            }
            ast::Statement::If {
                condition,
                then,
                otherwise,
            } => {
                let condition = self.expression(condition, 0, sink)?;
                let prelude = std::mem::take(&mut sink.prelude);
                let then = self.statements(then, sink)?;
                let otherwise = match otherwise {
                    Some(otherwise) => self.statements(otherwise, sink)?,
                    None => Vec::new(),
                };
                sink.prelude = prelude;
                before(sink, Stmt::If(0, condition, then, otherwise))
            }
            ast::Statement::While { condition, body } => {
                let condition = self.expression(condition, 0, sink)?;
                if !sink.prelude.is_empty() {
                    return Err(Error::at(
                        0,
                        "a function called in a loop's condition is not compiled yet".into(),
                    ));
                }
                let body = self.statements(body, sink)?;
                if body.iter().any(contributes) {
                    return Err(Error::at(
                        0,
                        "a contribution within a loop is not compiled yet".into(),
                    ));
                }
                vec![Stmt::While(condition, body)]
            }
            ast::Statement::Task {
                name,
                arguments,
                line,
            } => {
                // `$finish` and `$stop` end the simulation, which jolty does by refusing
                // the instance; their arguments only say what to print.
                if matches!(name.as_str(), "$finish" | "$stop") {
                    return Ok(before(
                        sink,
                        Stmt::Fail(format!("the model calls {name}"), Vec::new()),
                    ));
                }
                let mut arguments = arguments.iter();
                let Some(Expression::Text(message)) = arguments.next() else {
                    return Err(Error::at(*line, format!("{name} takes a message first")));
                };
                let values = arguments
                    .map(|argument| self.expression(argument, *line, sink))
                    .collect::<Result<_, _>>()?;
                let task = match name.as_str() {
                    "$error" | "$fatal" => Stmt::Fail(message.clone(), values),
                    "$warning" | "$strobe" | "$display" | "$write" | "$debug" => {
                        Stmt::Warn(message.clone(), values, false)
                    }
                    _ => return Err(Error::at(*line, format!("{name} is not compiled yet"))),
                };
                before(sink, task)
            }
        })
    }

    fn variable(&self, name: &str, line: usize) -> Result<usize, Error> {
        match self
            .bindings
            .iter()
            .rev()
            .find(|(bound, _)| bound == name)
            .map(|(_, binding)| *binding)
        {
            Some(Binding::Variable(index)) => Ok(index),
            Some(Binding::LocalParameter(_)) => Err(Error::at(
                line,
                format!("{name} is a localparam, not a variable"),
            )),
            None => Err(Error::at(line, format!("{name} is not a variable"))),
        }
    }

    /// The branch an access function reads or contributes to: a named one, or the
    /// unnamed one of two nodes, added on first use; its nodes of the discipline that
    /// the access `nature` reads.
    fn branch(
        &self,
        nature: &str,
        operands: &[String],
        line: usize,
        sink: &mut Sink,
    ) -> Result<usize, Error> {
        // The module's declared branches come first, each found by its name; another
        // single node is its unnamed branch to the ground.
        let declared = self.module.branches.len().min(sink.branches.len());
        let named = match operands {
            [name] => sink.branches[..declared]
                .iter()
                .position(|branch| branch.name == *name),
            _ => None,
        };
        let index = match (named, operands) {
            (Some(found), _) => found,
            (None, [node]) => {
                let plus = node_index(self.nodes, node, line)?;
                // Named apart from the node, whose unknown its current's would meet.
                let name = format!("{node}_ground");
                if let Some(found) = sink
                    .branches
                    .iter()
                    .position(|branch| branch.name == name && branch.minus == GROUND)
                {
                    found
                } else {
                    sink.branches.push(Branch {
                        name,
                        plus,
                        minus: GROUND,
                    });
                    sink.branches.len() - 1
                }
            }
            (None, [plus, minus]) => {
                let plus = node_index(self.nodes, plus, line)?;
                let minus = node_index(self.nodes, minus, line)?;
                let name = format!("{}_{}", self.nodes[plus], self.nodes[minus]);
                if let Some(found) = sink
                    .branches
                    .iter()
                    .position(|branch| branch.name == name && branch.plus == plus)
                {
                    found
                } else {
                    sink.branches.push(Branch { name, plus, minus });
                    sink.branches.len() - 1
                }
            }
            _ => {
                return Err(Error::at(
                    line,
                    "an access takes a branch or two nodes".into(),
                ));
            }
        };
        let thermal = matches!(nature, "Temp" | "Pwr");
        let Branch { plus, minus, .. } = sink.branches[index];
        let discipline = |node: usize| node == GROUND || self.thermal[node] == thermal;
        if !discipline(plus) || !discipline(minus) {
            let discipline = if thermal { "thermal" } else { "electrical" };
            return Err(Error::at(
                line,
                format!("{nature} reads {discipline} nodes"),
            ));
        }
        Ok(index)
    }

    fn contribution(
        &self,
        nature: &str,
        operands: &[String],
        value: &Expression,
        line: usize,
        sink: &mut Sink,
    ) -> Result<Stmt, Error> {
        let potential = match nature {
            "V" | "Temp" => true,
            "I" | "Pwr" => false,
            _ => {
                return Err(Error::at(
                    line,
                    format!("{nature} is not an access function"),
                ));
            }
        };
        let branch = self.branch(nature, operands, line, sink)?;
        let mut factors = Vec::new();
        let mut terms = Vec::new();
        split_contribution(value, false, &mut factors, &mut terms);
        let (mut resistive, mut reactive, mut reactive_factors, mut noise) =
            (None, None, Vec::new(), Vec::new());
        let add = |sum: &mut Option<Expr>, negative: bool, term: Expr| {
            *sum = Some(match (sum.take(), negative) {
                (None, false) => term,
                (None, true) => Expr::Unary("-", Box::new(term)),
                (Some(sum), false) => Expr::Binary("+", Box::new(sum), Box::new(term)),
                (Some(sum), true) => Expr::Binary("-", Box::new(sum), Box::new(term)),
            });
        };
        for (negative, term, term_factors) in terms {
            match term {
                Expression::Call(name, arguments) if name == "ddt" => {
                    let [argument] = arguments.as_slice() else {
                        return Err(Error::at(line, "ddt takes one argument".into()));
                    };
                    let mut resolved = self.expression(argument, line, sink)?;
                    for factor in term_factors.iter().rev() {
                        let scale = self.expression(factor.expression(), line, sink)?;
                        reactive_factors.push((scale.clone(), line));
                        resolved = factor.apply(scale, resolved);
                    }
                    add(&mut reactive, negative, resolved);
                }
                Expression::Call(name, arguments)
                    if term_factors.is_empty()
                        && (name == "white_noise" || name == "flicker_noise") =>
                {
                    if negative {
                        return Err(Error::at(
                            line,
                            "a noise source is added, not subtracted".into(),
                        ));
                    }
                    noise.push(self.noise(name, arguments, line, sink)?);
                }
                term => {
                    let mut resolved = self.expression(term, line, sink)?;
                    for factor in term_factors.iter().rev() {
                        let scale = self.expression(factor.expression(), line, sink)?;
                        resolved = factor.apply(scale, resolved);
                    }
                    add(&mut resistive, negative, resolved);
                }
            }
        }
        Ok(Stmt::Contribute {
            branch,
            potential,
            resistive,
            reactive,
            reactive_factors,
            noise,
        })
    }

    /// The noise source `name(arguments)`, white or flicker.
    fn noise(
        &self,
        name: &str,
        arguments: &[Expression],
        line: usize,
        sink: &mut Sink,
    ) -> Result<Noise, Error> {
        let (power, rest) = arguments
            .split_first()
            .ok_or_else(|| Error::at(line, format!("{name} takes its power")))?;
        let (exponent, rest) = if name == "flicker_noise" {
            let (exponent, rest) = rest
                .split_first()
                .ok_or_else(|| Error::at(line, "flicker_noise takes its exponent".into()))?;
            let exponent = self.expression(exponent, line, sink)?;
            (Some(exponent), rest)
        } else {
            (None, rest)
        };
        let [Expression::Text(name)] = rest else {
            return Err(Error::at(line, "a noise source is named".into()));
        };
        Ok(Noise {
            name: name.clone(),
            power: self.expression(power, line, sink)?,
            exponent,
        })
    }

    fn expression(
        &self,
        expression: &Expression,
        line: usize,
        sink: &mut Sink,
    ) -> Result<Expr, Error> {
        let mut recurse = |inner: &Expression| self.expression(inner, line, sink);
        Ok(match expression {
            Expression::Real(value) => Expr::Real(*value),
            Expression::Integer(value) => Expr::Integer(
                i32::try_from(*value)
                    .map_err(|_| Error::at(line, format!("{value} exceeds an integer")))?,
            ),
            Expression::Text(_) => return Err(Error::at(line, "a string is not a number".into())),
            Expression::Unary(operator, operand) => {
                Expr::Unary(operator, Box::new(recurse(operand)?))
            }
            Expression::Binary(operator, left, right) => {
                let left = recurse(left)?;
                let right = recurse(right)?;
                if matches!(*operator, "%" | "**" | "&" | "|" | "^") {
                    return Err(Error::at(
                        line,
                        format!("the operator {operator} is not compiled yet"),
                    ));
                }
                Expr::Binary(operator, Box::new(left), Box::new(right))
            }
            Expression::Conditional(condition, then, otherwise) => Expr::Conditional(
                Box::new(recurse(condition)?),
                Box::new(recurse(then)?),
                Box::new(recurse(otherwise)?),
            ),
            Expression::Name(name) => return self.resolve_name(name, line, sink),
            Expression::Call(name, arguments) => return self.call(name, arguments, line, sink),
        })
    }

    /// Resolves an identifier `name` against the lexical scope, module parameters, and
    /// constant-context diagnostics.
    fn resolve_name(&self, name: &str, line: usize, sink: &Sink) -> Result<Expr, Error> {
        if let Some(owner) = &self.constant_owner
            && self.later_locals.iter().any(|later| later == name)
        {
            return Err(Error::at(
                line,
                format!("{owner} reads {name}, which is not declared before it"),
            ));
        }
        if let Some((_, binding)) = self.bindings.iter().rev().find(|(bound, _)| bound == name) {
            return match *binding {
                Binding::Variable(variable) => {
                    if let Some(owner) = &self.constant_owner {
                        return Err(Error::at(
                            line,
                            format!("{owner} reads the variable {name}, which is not a constant"),
                        ));
                    }
                    if self.initial && !sink.initial.contains(&variable) {
                        return Err(Error::at(
                            line,
                            format!(
                                "code under an initial event reads {name}, which no initial code assigned before"
                            ),
                        ));
                    }
                    Ok(Expr::Variable(variable))
                }
                Binding::LocalParameter(LocalValue::Real(value)) => Ok(Expr::Real(value)),
                Binding::LocalParameter(LocalValue::Integer(value)) => Ok(Expr::Integer(value)),
                Binding::LocalParameter(LocalValue::Prepared(index)) => {
                    Ok(Expr::LocalParameter(index))
                }
            };
        }
        if let Some(parameter) = self
            .module
            .parameters
            .iter()
            .position(|parameter| parameter.name == name)
        {
            if let Some(allowed) = &self.allowed_parameters
                && !allowed.contains(&parameter)
            {
                let owner = self.constant_owner.as_deref().unwrap_or("declaration");
                return Err(Error::at(
                    line,
                    format!("{owner} reads {name}, which is not declared before it"),
                ));
            }
            return Ok(Expr::Parameter(parameter));
        }
        if let Some(owner) = &self.constant_owner
            && self
                .module
                .variables
                .iter()
                .any(|(variable, _)| variable == name)
        {
            return Err(Error::at(
                line,
                format!("{owner} reads the variable {name}, which is not a constant"),
            ));
        }
        Err(Error::at(line, format!("{name} is not declared")))
    }

    /// The probe `nature(arguments)`: a potential or a flow.
    fn access(
        &self,
        nature: &str,
        arguments: &[Expression],
        line: usize,
        sink: &mut Sink,
    ) -> Result<Expr, Error> {
        let operands = arguments
            .iter()
            .map(|argument| match argument {
                Expression::Name(name) => Ok(name.clone()),
                _ => Err(Error::at(line, "an access takes names".into())),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let branch = self.branch(nature, &operands, line, sink)?;
        if matches!(nature, "V" | "Temp") {
            let Branch { plus, minus, .. } = sink.branches[branch];
            Ok(Expr::Probe(Probe::Voltage(plus, minus)))
        } else {
            Ok(Expr::Probe(Probe::Current(branch)))
        }
    }

    fn call(
        &self,
        name: &str,
        arguments: &[Expression],
        line: usize,
        sink: &mut Sink,
    ) -> Result<Expr, Error> {
        if let Some(builtin) = Builtin::named(name) {
            let arity = builtin.arity();
            if arguments.len() != arity {
                return Err(Error::at(line, format!("{name} takes {arity} arguments")));
            }
            let arguments = arguments
                .iter()
                .map(|argument| self.expression(argument, line, sink))
                .collect::<Result<_, _>>()?;
            return Ok(Expr::Call(builtin, arguments));
        }
        if let Some(function) = self
            .module
            .functions
            .iter()
            .find(|function| function.name == name)
        {
            if let Some(owner) = &self.constant_owner {
                return Err(Error::at(
                    line,
                    format!("{owner} calls the analog function {name}, which is not compiled yet"),
                ));
            }
            return self.inline(function, arguments, line, sink);
        }
        match name {
            "V" | "I" | "Temp" | "Pwr" | "ddx" | "$limit" if self.constant_owner.is_some() => {
                let owner = self.constant_owner.as_deref().unwrap_or("declaration");
                Err(Error::at(
                    line,
                    format!("{owner} reads the probes through {name}, which is not a constant"),
                ))
            }
            "V" | "I" | "Temp" | "Pwr" | "ddx" | "$limit" if self.initial => Err(Error::at(
                line,
                format!(
                    "code under an initial event reads the probes through {name}, which its first step alone fixes"
                ),
            )),
            "V" | "I" | "Temp" | "Pwr" if self.probes => self.access(name, arguments, line, sink),
            "ddx" if self.probes => self.resolve_ddx(arguments, line, sink),
            "$temperature" if arguments.is_empty() => Ok(Expr::Temperature),
            // jolty's circuits place each instance once: none has a multiplicity but 1.
            "$mfactor" if arguments.is_empty() => Ok(Expr::Real(1.0)),
            // A jolty instance connects each of its terminals.
            "$port_connected" => match arguments {
                [Expression::Name(port)] if self.module.ports.contains(port) => {
                    Ok(Expr::Integer(1))
                }
                _ => Err(Error::at(line, "$port_connected takes a port".into())),
            },
            "$vt" => {
                let temperature = match arguments {
                    [] => Expr::Temperature,
                    [temperature] => self.expression(temperature, line, sink)?,
                    _ => return Err(Error::at(line, "$vt takes at most a temperature".into())),
                };
                let product =
                    Expr::Binary("*", Box::new(temperature), Box::new(Expr::Real(BOLTZMANN)));
                Ok(Expr::Binary(
                    "/",
                    Box::new(product),
                    Box::new(Expr::Real(ELEMENTARY_CHARGE)),
                ))
            }
            "$param_given" => self.resolve_param_given(arguments, line),
            "$simparam" => self.resolve_simparam(arguments, line, sink),
            "$limit" if self.probes => self.limit(arguments, line, sink),
            _ => Err(Error::at(line, format!("{name} is not compiled yet"))),
        }
    }

    fn resolve_ddx(
        &self,
        arguments: &[Expression],
        line: usize,
        sink: &mut Sink,
    ) -> Result<Expr, Error> {
        let [operand, Expression::Call(nature, node)] = arguments else {
            return Err(Error::at(
                line,
                "ddx takes a value and a node's potential".into(),
            ));
        };
        let (true, [Expression::Name(node)]) =
            (matches!(nature.as_str(), "V" | "Temp"), node.as_slice())
        else {
            return Err(Error::at(
                line,
                "ddx differentiates by one node's potential".into(),
            ));
        };
        let operand = self.expression(operand, line, sink)?;
        Ok(Expr::Ddx(
            Box::new(operand),
            node_index(self.nodes, node, line)?,
        ))
    }

    fn resolve_param_given(&self, arguments: &[Expression], line: usize) -> Result<Expr, Error> {
        match arguments {
            [Expression::Name(parameter)] => {
                let index = self
                    .module
                    .parameters
                    .iter()
                    .position(|declared| declared.name == *parameter)
                    .ok_or_else(|| Error::at(line, format!("{parameter} is not a parameter")))?;
                if let Some(allowed) = &self.allowed_parameters
                    && !allowed.contains(&index)
                {
                    let owner = self.constant_owner.as_deref().unwrap_or("declaration");
                    return Err(Error::at(
                        line,
                        format!("{owner} reads {parameter}, which is not declared before it"),
                    ));
                }
                Ok(Expr::Given(index))
            }
            _ => Err(Error::at(line, "$param_given takes one parameter".into())),
        }
    }

    fn resolve_simparam(
        &self,
        arguments: &[Expression],
        line: usize,
        sink: &mut Sink,
    ) -> Result<Expr, Error> {
        match arguments {
            [Expression::Text(simparam), ..] if simparam == "gmin" => {
                if let Some(owner) = &self.constant_owner {
                    return Err(Error::at(
                        line,
                        format!("{owner} reads $simparam(\"gmin\"), which is not a constant"),
                    ));
                }
                Ok(Expr::Gmin)
            }
            [Expression::Text(simparam), ..] if simparam == "small_resistance" => {
                Ok(Expr::SmallResistance)
            }
            // A name the simulator does not define reads as the default given, as the
            // standard has it; jolty defines no other.
            [Expression::Text(_), default] => self.expression(default, line, sink),
            [Expression::Text(simparam)] => Err(Error::at(
                line,
                format!("jolty defines no $simparam {simparam:?}"),
            )),
            _ => Err(Error::at(line, "$simparam takes a name".into())),
        }
    }

    /// `$limit(arguments)`: a voltage limited from its last evaluation's, by an analog
    /// function of the module's or one of the simulator's, named as a string.
    fn limit(&self, arguments: &[Expression], line: usize, sink: &mut Sink) -> Result<Expr, Error> {
        let (access, rest) = arguments
            .split_first()
            .ok_or_else(|| Error::at(line, "$limit takes a probe".into()))?;
        let Expr::Probe(Probe::Voltage(plus, minus)) = self.expression(access, line, sink)? else {
            return Err(Error::at(line, "$limit limits a voltage".into()));
        };
        let (function, rest) = match rest.split_first() {
            Some((Expression::Text(name), rest)) => match name.as_str() {
                "pnjlim" => (Limiter::Junction, rest),
                "fetlim" => (Limiter::Gate, rest),
                _ => {
                    return Err(Error::at(
                        line,
                        format!("the limiting function {name:?} is not jolty's"),
                    ));
                }
            },
            Some((Expression::Name(name), rest)) => {
                let function = self
                    .module
                    .functions
                    .iter()
                    .position(|declared| declared.name == *name)
                    .ok_or_else(|| Error::at(line, format!("{name} is not an analog function")))?;
                (Limiter::Function(function), rest)
            }
            _ => return Err(Error::at(line, "$limit names its limiting function".into())),
        };
        let arity = match function {
            Limiter::Junction => Some(2),
            Limiter::Gate => Some(1),
            Limiter::Function(_) => None,
        };
        if arity.is_some_and(|arity| arity != rest.len()) {
            return Err(Error::at(
                line,
                "the limiting function takes other arguments".into(),
            ));
        }
        let arguments = rest
            .iter()
            .map(|argument| self.expression(argument, line, sink))
            .collect::<Result<_, _>>()?;
        sink.limits.push(Limit {
            plus,
            minus,
            function,
            arguments,
        });
        Ok(Expr::Probe(Probe::Limited(sink.limits.len() - 1)))
    }

    /// A call of the analog function `function` with `arguments`, inlined: its inputs
    /// assigned and its body run before the statement that calls it, in variables of
    /// its own, and its result read.
    fn inline(
        &self,
        function: &ast::Function,
        arguments: &[Expression],
        line: usize,
        sink: &mut Sink,
    ) -> Result<Expr, Error> {
        if arguments.len() != function.inputs.len() {
            return Err(Error::at(
                line,
                format!(
                    "{} takes {} arguments",
                    function.name,
                    function.inputs.len()
                ),
            ));
        }
        check_function_duplicates(function)?;
        sink.calls += 1;
        let call = sink.calls;
        let module_bindings: Vec<(String, Binding)> = self
            .module_locals
            .iter()
            .map(|(name, value)| (name.clone(), Binding::LocalParameter(*value)))
            .collect();
        let mut scope = Scope {
            bindings: module_bindings,
            allowed_parameters: None,
            later_locals: Vec::new(),
            constant_owner: None,
            probes: false,
            initial: false,
            ..self.clone()
        };
        let mut input_indices = Vec::with_capacity(function.inputs.len());
        for (position, (variable, kind)) in function_variables(function).into_iter().enumerate() {
            let index = sink.variable(&format!("{}{call}_{variable}", function.name), kind);
            if position < function.inputs.len() {
                input_indices.push(index);
            }
            scope.bindings.push((variable, Binding::Variable(index)));
        }
        resolve_scoped_locals(&mut scope, &function.local_parameters, sink)?;
        let mut inputs = Vec::new();
        for (argument, index) in arguments.iter().zip(input_indices) {
            inputs.push(Stmt::Assign(index, self.expression(argument, line, sink)?));
        }
        let outer = std::mem::take(&mut sink.prelude);
        let body = scope.statements(&function.body, sink)?;
        sink.prelude = outer;
        sink.prelude.extend(inputs);
        sink.prelude.extend(body);
        let result = scope.variable(&function.name, line)?;
        Ok(Expr::Variable(result))
    }
}

/// One multiplicative or divisive factor wrapped around a contribution term.
#[derive(Clone, Copy)]
enum Factor<'a> {
    LeftMul(&'a Expression),
    RightMul(&'a Expression),
    RightDiv(&'a Expression),
}

impl<'a> Factor<'a> {
    const fn expression(self) -> &'a Expression {
        match self {
            Self::LeftMul(expression) | Self::RightMul(expression) | Self::RightDiv(expression) => {
                expression
            }
        }
    }

    fn apply(self, scale: Expr, inner: Expr) -> Expr {
        match self {
            Self::LeftMul(_) => Expr::Binary("*", Box::new(scale), Box::new(inner)),
            Self::RightMul(_) => Expr::Binary("*", Box::new(inner), Box::new(scale)),
            Self::RightDiv(_) => Expr::Binary("/", Box::new(inner), Box::new(scale)),
        }
    }
}

/// Whether `expression` holds a `ddt(...)` call anywhere within it.
fn contains_ddt(expression: &Expression) -> bool {
    let mut found = false;
    expression.walk(&mut |inner| {
        if let Expression::Call(name, _) = inner
            && name == "ddt"
        {
            found = true;
        }
    });
    found
}

/// Splits `expression` into additive terms, each with whether it is subtracted and any
/// multiplicative/divisive factors wrapped around a `ddt(...)` subexpression.
fn split_contribution<'a>(
    expression: &'a Expression,
    negative: bool,
    factors: &mut Vec<Factor<'a>>,
    terms: &mut Vec<(bool, &'a Expression, Vec<Factor<'a>>)>,
) {
    match expression {
        Expression::Binary("+", left, right) => {
            split_contribution(left, negative, factors, terms);
            split_contribution(right, negative, factors, terms);
        }
        Expression::Binary("-", left, right) => {
            split_contribution(left, negative, factors, terms);
            split_contribution(right, !negative, factors, terms);
        }
        Expression::Unary("-", operand)
            if matches!(**operand, Expression::Call(..)) || contains_ddt(operand) =>
        {
            split_contribution(operand, !negative, factors, terms);
        }
        Expression::Binary("*", left, right) if contains_ddt(left) != contains_ddt(right) => {
            if contains_ddt(left) {
                factors.push(Factor::RightMul(right));
                split_contribution(left, negative, factors, terms);
            } else {
                factors.push(Factor::LeftMul(left));
                split_contribution(right, negative, factors, terms);
            }
            factors.pop();
        }
        Expression::Binary("/", left, right) if contains_ddt(left) && !contains_ddt(right) => {
            factors.push(Factor::RightDiv(right));
            split_contribution(left, negative, factors, terms);
            factors.pop();
        }
        _ => terms.push((negative, expression, factors.clone())),
    }
}

/// Whether `statement` contributes to a branch, or holds a statement that does.
pub(crate) fn contributes(statement: &Stmt) -> bool {
    match statement {
        Stmt::Contribute { .. } => true,
        Stmt::If(_, _, then, otherwise) => then.iter().chain(otherwise).any(contributes),
        Stmt::While(_, body) => body.iter().any(contributes),
        Stmt::Assign(..) | Stmt::Fail(..) | Stmt::Warn(..) => false,
    }
}

/// Gives each variable that `statements`, the analog block, assigns again where every
/// path does, at its top, a new version of it from there on, so that what is read before
/// keeps its own dependence, as a static number later made dynamic does; adds the new
/// versions to `variables`, and returns the last version of each variable and the
/// variable each version is of.
fn version(
    statements: &mut [Stmt],
    variables: &mut Vec<(String, Type)>,
) -> (Vec<usize>, Vec<usize>) {
    let mut current: Vec<usize> = (0..variables.len()).collect();
    let mut origins = current.clone();
    let mut assigned = vec![false; variables.len()];
    let rename = |expr: &mut Expr, current: &[usize]| {
        rename_reads(expr, current);
    };
    for statement in statements {
        if let Stmt::Assign(variable, value) = statement {
            rename(value, &current);
            let original = *variable;
            if assigned[original] {
                let (name, kind) = variables[original].clone();
                let unique = apart(&name, |taken| {
                    variables.iter().any(|(variable, _)| variable == taken)
                });
                variables.push((unique, kind));
                origins.push(original);
                current[original] = variables.len() - 1;
            }
            assigned[original] = true;
            *variable = current[original];
        } else {
            statement.exprs_mut(&mut |expr| rename(expr, &current));
            statement.targets_mut(&mut |target| {
                assigned[*target] = true;
                *target = current[*target];
            });
        }
    }
    let mut last: Vec<usize> = current;
    last.resize(variables.len(), 0);
    for (index, version) in last.iter_mut().enumerate().skip(assigned.len()) {
        *version = index;
    }
    (last, origins)
}

/// `expr` reading each variable's version in `current`.
fn rename_reads(expr: &mut Expr, current: &[usize]) {
    if let Expr::Variable(variable) = expr {
        if let Some(&version) = current.get(*variable) {
            *variable = version;
        }
        return;
    }
    match expr {
        Expr::Unary(_, operand) | Expr::Ddx(operand, _) => rename_reads(operand, current),
        Expr::Binary(_, left, right) => {
            rename_reads(left, current);
            rename_reads(right, current);
        }
        Expr::Conditional(condition, then, otherwise) => {
            for inner in [condition, then, otherwise] {
                rename_reads(inner, current);
            }
        }
        Expr::Call(_, arguments) => {
            for argument in arguments {
                rename_reads(argument, current);
            }
        }
        _ => {}
    }
}

/// Numbers each conditional of `statements`, from `next` on, in the order they hold them.
fn number(statements: &mut [Stmt], next: &mut usize) {
    for statement in statements {
        match statement {
            Stmt::If(id, _, then, otherwise) => {
                *id = *next;
                *next += 1;
                number(then, next);
                number(otherwise, next);
            }
            Stmt::While(_, body) => number(body, next),
            Stmt::Assign(..) | Stmt::Contribute { .. } | Stmt::Fail(..) | Stmt::Warn(..) => {}
        }
    }
}

/// Finds which variables are dynamic, and the probes each depends on.
fn analyse(module: &mut Module) {
    let count = module.variables.len();
    module.dynamic = vec![false; count];
    module.probes = vec![BTreeSet::new(); count];
    loop {
        let before = (module.dynamic.clone(), module.probes.clone());
        let analog = std::mem::take(&mut module.analog);
        flow(module, &analog, false);
        // A static variable that dynamic code reads keeps one value for it, which
        // preparation leaves; one assigned again after such a read is dynamic.
        let mut read = BTreeSet::new();
        reassigned(module, &analog, false, &mut read);
        module.analog = analog;
        if (module.dynamic.clone(), module.probes.clone()) == before {
            return;
        }
    }
}

impl Module {
    /// The name of `node`, `ground` for the ground.
    pub(crate) fn node_name(&self, node: usize) -> &str {
        if node == GROUND {
            "ground"
        } else {
            &self.nodes[node]
        }
    }

    /// Whether `node` is one of the internal nodes, neither a port nor the ground.
    pub(crate) fn is_internal(&self, node: usize) -> bool {
        node != GROUND && node >= self.ports
    }

    /// Whether `expr` is dynamic.
    pub(crate) fn is_dynamic(&self, expr: &Expr) -> bool {
        let mut dynamic = false;
        expr.walk(&mut |inner| {
            dynamic |= match inner {
                Expr::Probe(_) | Expr::Gmin | Expr::Ddx(..) | Expr::Shape(_) => true,
                Expr::Variable(variable) => self.dynamic[*variable],
                _ => false,
            };
        });
        dynamic
    }

    /// The probes `expr` depends on through its value.
    pub(crate) fn probes_of(&self, expr: &Expr) -> BTreeSet<Probe> {
        let mut probes = BTreeSet::new();
        self.collect_probes(expr, &mut probes);
        probes
    }

    fn collect_probes(&self, expr: &Expr, probes: &mut BTreeSet<Probe>) {
        match expr {
            Expr::Probe(probe) => {
                probes.insert(*probe);
            }
            Expr::Variable(variable) => probes.extend(self.probes[*variable].iter().copied()),
            // A comparison, a logical operation, a condition or a `ddx` has no derivative.
            Expr::Binary("<" | "<=" | ">" | ">=" | "==" | "!=" | "&&" | "||", ..)
            | Expr::Unary("!", _)
            | Expr::Ddx(..)
            | Expr::Shape(_)
            | Expr::Real(_)
            | Expr::Integer(_)
            | Expr::Parameter(_)
            | Expr::LocalParameter(_)
            | Expr::Given(_)
            | Expr::Temperature
            | Expr::Gmin
            | Expr::SmallResistance => {}
            Expr::Conditional(_, then, otherwise) => {
                self.collect_probes(then, probes);
                self.collect_probes(otherwise, probes);
            }
            Expr::Unary(_, operand) => self.collect_probes(operand, probes),
            Expr::Binary(_, left, right) => {
                self.collect_probes(left, probes);
                self.collect_probes(right, probes);
            }
            Expr::Call(_, arguments) => {
                for argument in arguments {
                    self.collect_probes(argument, probes);
                }
            }
        }
    }
}

/// Propagates dependence through `statements`, under a condition that is dynamic when
/// `control` is.
fn flow(module: &mut Module, statements: &[Stmt], control: bool) {
    for statement in statements {
        match statement {
            Stmt::Assign(variable, value) => {
                if control || module.is_dynamic(value) {
                    module.dynamic[*variable] = true;
                }
                if module.variables[*variable].1 == Type::Real {
                    let probes = module.probes_of(value);
                    module.probes[*variable].extend(probes);
                }
            }
            Stmt::If(_, condition, then, otherwise) => {
                let control = control || module.is_dynamic(condition);
                flow(module, then, control);
                flow(module, otherwise, control);
            }
            Stmt::While(condition, body) => {
                let control = control || module.is_dynamic(condition);
                flow(module, body, control);
            }
            Stmt::Contribute { .. } | Stmt::Fail(..) | Stmt::Warn(..) => {}
        }
    }
}

/// Marks dynamic each static variable assigned after dynamic code read it, `read` holding
/// those read so far; within a loop, the body runs twice, as a second iteration reads
/// what the first assigned.
fn reassigned(module: &mut Module, statements: &[Stmt], dynamic: bool, read: &mut BTreeSet<usize>) {
    let reads = |module: &Module, expr: &Expr, read: &mut BTreeSet<usize>| {
        expr.walk(&mut |inner| {
            if let Expr::Variable(variable) = inner
                && !module.dynamic[*variable]
            {
                read.insert(*variable);
            }
        });
    };
    for statement in statements {
        match statement {
            Stmt::Assign(variable, value) => {
                if module.dynamic[*variable] || dynamic {
                    reads(module, value, read);
                } else if read.contains(variable) {
                    module.dynamic[*variable] = true;
                }
            }
            Stmt::Contribute { .. } => statement.walk(&mut |expr| reads(module, expr, read)),
            Stmt::If(_, condition, then, otherwise) => {
                let inner = dynamic || module.is_dynamic(condition);
                if inner || then.iter().chain(otherwise).any(contributes) {
                    reads(module, condition, read);
                }
                reassigned(module, then, inner, read);
                reassigned(module, otherwise, inner, read);
            }
            Stmt::While(condition, body) => {
                let inner = dynamic || module.is_dynamic(condition);
                if inner {
                    reads(module, condition, read);
                }
                for _ in 0..2 {
                    reassigned(module, body, inner, read);
                }
            }
            Stmt::Fail(..) | Stmt::Warn(..) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Expr, Module, Stmt, constant, resolve};
    use crate::compile;
    use crate::lexer::tokens;
    use crate::parser::parse;

    const ADVICE_WORDS: [&str; 7] = [
        "use",
        "pass",
        "try",
        "instead",
        "should",
        "give",
        "did you mean",
    ];

    fn assert_factual(message: &str) {
        let lower = message.to_ascii_lowercase();
        for forbidden in ADVICE_WORDS {
            let matches_word = if forbidden.contains(' ') {
                lower.contains(forbidden)
            } else {
                lower
                    .split(|character: char| !character.is_ascii_alphanumeric())
                    .any(|word| word == forbidden)
            };
            assert!(
                !matches_word,
                "error message contains advice word {forbidden:?}: {message:?}"
            );
        }
    }

    /// The message the compiler refuses `analog`, the analog block of a two-terminal
    /// module with variables `x` and `y`, with.
    fn refusal(analog: &str) -> String {
        let source = format!(
            "`include \"disciplines.vams\"\nmodule m(a, b);\ninout a, b;\nelectrical a, b;\nreal x, y;\nanalog begin\n{analog}\nend\nendmodule\n"
        );
        let message = compile(&source, "m.va", &|_| None)
            .expect_err("refused")
            .message;
        assert_factual(&message);
        message
    }

    fn module_refusal(body: &str) -> String {
        let source = format!(
            "`include \"disciplines.vams\"\nmodule m(a, b);\ninout a, b;\nelectrical a, b;\n{body}\nendmodule\n"
        );
        let message = compile(&source, "m.va", &|_| None)
            .expect_err("refused")
            .message;
        assert_factual(&message);
        message
    }

    fn resolve_source(body: &str) -> Module {
        let source = format!(
            "`include \"disciplines.vams\"\nmodule m(a, b);\ninout a, b;\nelectrical a, b;\n{body}\nendmodule\n"
        );
        let tokens = tokens(&source, &|_| None).expect("tokens");
        let parsed = parse(&tokens).expect("parse");
        resolve(&parsed).expect("resolve")
    }

    /// Code under an initial event may read neither a probe nor what other code assigns,
    /// and no other code may make what it assigns depend on the probes.
    #[test]
    fn initial_code_reads_only_what_its_first_step_fixes() {
        assert_eq!(
            refusal("@(initial_step) x = V(a, b);\nI(a, b) <+ x;"),
            "code under an initial event reads the probes through V, which its first step alone fixes"
        );
        assert_eq!(
            refusal("y = 2;\n@(initial_step) x = y;\nI(a, b) <+ x * V(a, b);"),
            "code under an initial event reads y, which no initial code assigned before"
        );
        assert_eq!(
            refusal("@(initial_step) x = 1;\nif (V(a, b) > 0) x = 2;\nI(a, b) <+ x * V(a, b);"),
            "x, which code under an initial event assigns, depends on the probes, which is not compiled yet"
        );
        assert_eq!(
            refusal("@(final_step) x = 1;\nI(a, b) <+ x * V(a, b);"),
            "the event final_step is not compiled yet"
        );
    }

    #[test]
    fn localparam_duplicate_and_collision_errors() {
        assert_eq!(
            module_refusal(
                "localparam real k = 1.0;\nlocalparam real k = 2.0;\nanalog I(a, b) <+ V(a, b);"
            ),
            "the localparam k is declared more than once"
        );
        assert_eq!(
            module_refusal(
                "parameter real k = 1.0;\nlocalparam real k = 2.0;\nanalog I(a, b) <+ V(a, b);"
            ),
            "k is declared as both a parameter and a localparam"
        );
        assert_eq!(
            module_refusal(
                "localparam real k = 2.0;\nparameter real k = 1.0;\nanalog I(a, b) <+ V(a, b);"
            ),
            "k is declared as both a parameter and a localparam"
        );
        assert_eq!(
            module_refusal("localparam real k = 2.0;\nreal k;\nanalog I(a, b) <+ V(a, b);"),
            "k is declared as both a localparam and a variable"
        );
        assert_eq!(
            module_refusal(
                "parameter real p = 1.0;\naliasparam k = p;\nlocalparam real k = 2.0;\nanalog I(a, b) <+ V(a, b);"
            ),
            "k is declared as both a localparam and an aliasparam"
        );
        assert_eq!(
            module_refusal(
                "localparam real k = 2.0;\naliasparam alias_k = k;\nanalog I(a, b) <+ V(a, b);"
            ),
            "the alias alias_k targets localparam k, not a parameter"
        );
        assert_eq!(
            module_refusal(
                "analog begin : blk\nlocalparam real k = 1.0;\nlocalparam real k = 2.0;\nI(a, b) <+ V(a, b);\nend"
            ),
            "the localparam k is declared more than once"
        );
        assert_eq!(
            module_refusal(
                "analog begin : blk\nlocalparam real k = 1.0;\nreal k;\nI(a, b) <+ V(a, b);\nend"
            ),
            "k is declared as both a localparam and a variable"
        );
        assert_eq!(
            module_refusal(
                "analog function real f;\ninput x;\nlocalparam real k = 1.0;\nlocalparam real k = 2.0;\nf = x;\nendfunction\nanalog I(a, b) <+ f(V(a, b));"
            ),
            "the localparam k is declared more than once"
        );
        assert_eq!(
            module_refusal(
                "analog function real f;\ninput x;\nlocalparam real x = 1.0;\nf = x;\nendfunction\nanalog I(a, b) <+ f(V(a, b));"
            ),
            "x is declared as both a localparam and a variable"
        );
        assert_eq!(
            module_refusal(
                "analog function real f;\ninput x;\nlocalparam real f = 1.0;\nf = x;\nendfunction\nanalog I(a, b) <+ f(V(a, b));"
            ),
            "f is declared as both a localparam and a variable"
        );
    }

    #[test]
    fn localparam_ordering_scope_and_mutability_errors() {
        assert_eq!(
            module_refusal("localparam real k = missing;\nanalog I(a, b) <+ V(a, b);"),
            "missing is not declared"
        );
        assert_eq!(
            module_refusal("localparam real k = k + 1.0;\nanalog I(a, b) <+ V(a, b);"),
            "k reads k, which is not declared before it"
        );
        assert_eq!(
            module_refusal(
                "localparam real first = second;\nlocalparam real second = 1.0;\nanalog I(a, b) <+ V(a, b);"
            ),
            "first reads second, which is not declared before it"
        );
        assert_eq!(
            module_refusal(
                "localparam real k = p;\nparameter real p = 1.0;\nanalog I(a, b) <+ V(a, b);"
            ),
            "k reads p, which is not declared before it"
        );
        assert_eq!(
            module_refusal(
                "parameter real p = k;\nlocalparam real k = 1.0;\nanalog I(a, b) <+ V(a, b);"
            ),
            "p's default reads k, which is not declared before it"
        );
        assert_eq!(
            module_refusal(
                "localparam integer has_p = $param_given(p);\nparameter real p = 1.0;\nanalog I(a, b) <+ V(a, b);"
            ),
            "has_p reads p, which is not declared before it"
        );
        assert_eq!(
            module_refusal(
                "localparam real k = 2.0;\nanalog begin : blk\nlocalparam real a = k;\nlocalparam real k = 3.0;\nI(a, b) <+ V(a, b) * a;\nend"
            ),
            "a reads k, which is not declared before it"
        );
        assert_eq!(
            module_refusal(
                "localparam real k = 2.0;\nanalog begin\nk = 3.0;\nI(a, b) <+ V(a, b);\nend"
            ),
            "k is a localparam, not a variable"
        );
        assert_eq!(
            module_refusal(
                "localparam real k = 2.0;\nreal g;\nanalog begin\ng = $param_given(k);\nI(a, b) <+ V(a, b) * g;\nend"
            ),
            "k is not a parameter"
        );
    }

    #[test]
    fn localparam_constant_expression_and_type_errors() {
        assert_eq!(
            module_refusal("real x;\nlocalparam real k = x;\nanalog I(a, b) <+ V(a, b);"),
            "k reads the variable x, which is not a constant"
        );
        assert_eq!(
            module_refusal(
                "analog begin : blk\nlocalparam real k = y;\nreal y;\nI(a, b) <+ V(a, b) * k;\nend"
            ),
            "k reads the variable y, which is not a constant"
        );
        assert_eq!(
            module_refusal("localparam real k = V(a, b);\nanalog I(a, b) <+ V(a, b);"),
            "k reads the probes through V, which is not a constant"
        );
        assert_eq!(
            module_refusal("localparam real k = $simparam(\"gmin\");\nanalog I(a, b) <+ V(a, b);"),
            "k reads $simparam(\"gmin\"), which is not a constant"
        );
        assert_eq!(
            module_refusal(
                "analog function real f;\ninput x;\nf = x + 1.0;\nendfunction\nlocalparam real k = f(1.0);\nanalog I(a, b) <+ V(a, b);"
            ),
            "k calls the analog function f, which is not compiled yet"
        );
        assert_eq!(
            module_refusal("localparam label = \"hello\";\nanalog I(a, b) <+ V(a, b);"),
            "a string is not a number"
        );
        assert_eq!(
            module_refusal("localparam integer n = 1.5;\nanalog I(a, b) <+ V(a, b);"),
            "the integer localparam n has a real initializer"
        );
        assert_eq!(
            module_refusal(
                "parameter real p = 1.0;\nlocalparam integer n = p;\nanalog I(a, b) <+ V(a, b);"
            ),
            "the integer localparam n has a real initializer"
        );
        assert_eq!(
            module_refusal("localparam integer n = 0 ? 1.5 : 2;\nanalog I(a, b) <+ V(a, b);"),
            "the integer localparam n has a real initializer"
        );
        assert_eq!(
            module_refusal("localparam integer n = 1 / 0;\nanalog I(a, b) <+ V(a, b);"),
            "the integer localparam n divides by zero"
        );
        assert_eq!(
            module_refusal("localparam real x = 5 / 0;\nanalog I(a, b) <+ V(a, b);"),
            "the localparam x divides by zero"
        );
        assert_eq!(
            module_refusal("localparam integer n = 2147483647 + 1;\nanalog I(a, b) <+ V(a, b);"),
            "the integer localparam n overflows an integer"
        );
        assert_eq!(
            module_refusal(
                "localparam integer n = -(-2147483647 - 1);\nanalog I(a, b) <+ V(a, b);"
            ),
            "the integer localparam n overflows an integer"
        );
        assert_eq!(
            module_refusal("localparam real x = 1.0 / 0.0;\nanalog I(a, b) <+ V(a, b);"),
            "the localparam x evaluates to a non-finite value"
        );
        assert_eq!(
            module_refusal("localparam real x = sqrt(-1.0);\nanalog I(a, b) <+ V(a, b);"),
            "the localparam x evaluates to a non-finite value"
        );
        assert_eq!(
            module_refusal("parameter real p = sqrt(-1.0);\nanalog I(a, b) <+ V(a, b);"),
            "p's default evaluates to a non-finite value"
        );
        assert_eq!(
            module_refusal(
                "parameter real p = 1.0 from (0:sqrt(-1.0));\nanalog I(a, b) <+ V(a, b);"
            ),
            "p's range evaluates to a non-finite value"
        );
    }

    #[test]
    fn limiting_function_rejects_parameter_dependent_localparam_cleanly() {
        assert_eq!(
            module_refusal(
                "parameter real p = 2.0;\nlocalparam real k = 2.0 * p;\nanalog function real lim;\ninput vnew, vold;\nreal vnew, vold;\nlim = vnew * k;\nendfunction\nreal v;\nanalog begin\nv = $limit(V(a, b), lim);\nI(a, b) <+ v;\nend"
            ),
            "the limiting function reads localparam k, which depends on parameters and is not compiled yet"
        );
        assert_eq!(
            module_refusal(
                "parameter real p = 2.0;\nanalog function real lim;\ninput vnew, vold;\nlocalparam real scale = p * 2.0;\nreal vnew, vold;\nlim = vnew * scale;\nendfunction\nreal v;\nanalog begin\nv = $limit(V(a, b), lim);\nI(a, b) <+ v;\nend"
            ),
            "the limiting function reads localparam scale, which depends on parameters and is not compiled yet"
        );
    }

    #[test]
    fn localparam_constant_folding_handles_builtins_truncation_and_short_circuit() {
        let module = resolve_source(
            "localparam integer truncated = 5 / 2;\n\
             localparam real truncated_real = 5 / 2;\n\
             localparam real guarded = 0 ? (1.0 / 0.0) : (sqrt(4.0) + ln(exp(1.0)) + floor(3.7) + ceil(1.2));\n\
             localparam integer short_and = 0 && (1 / 0);\n\
             localparam integer short_or = 1 || (1 / 0);\n\
             localparam untyped_int = truncated + short_or;\n\
             localparam untyped_real = guarded + truncated_real;\n\
             analog I(a, b) <+ V(a, b) * (untyped_real + untyped_int + short_and);",
        );
        // Every pure constant folds directly with zero prepared localparams.
        assert_eq!(module.local_parameters, [] as [super::LocalParameter; 0]);
        // `truncated = 2`, `truncated_real = 2.0`, `guarded = 2 + 1 + 3 + 2 = 8.0`,
        // `short_and = 0`, `short_or = 1`, `untyped_int = 3`, `untyped_real = 10.0`.
        let Stmt::Contribute {
            resistive: Some(Expr::Binary("*", _, right)),
            ..
        } = &module.analog[0]
        else {
            panic!("expected resistive contribution");
        };
        let value = constant(right).expect("folded constant");
        assert!((value - 13.0).abs() < 1e-12, "got {value}");
    }

    #[test]
    fn static_scaled_ddt_accepts_static_and_refuses_dynamic_factors() {
        for valid in [
            "x = 1e-12;\nI(a, b) <+ x * ddt(V(a, b));",
            "x = 1e-12;\nI(a, b) <+ ddt(V(a, b)) * x;",
            "x = 2.0;\nI(a, b) <+ ddt(V(a, b)) / x;",
            "x = 1e-12;\nI(a, b) <+ -(x * ddt(V(a, b)));",
            "x = 1e-12;\nI(a, b) <+ x * (-ddt(V(a, b)));",
            "x = 1e-12;\nI(a, b) <+ x * (V(a, b) / 100.0 + ddt(V(a, b)));",
            "x = 1e-12;\ny = 2.0;\nI(a, b) <+ (ddt(V(a, b)) * x) / y;",
            "x = 1e-12;\nI(a, b) <+ x * ddt(V(a, b));\nx = V(a, b);",
            "x = 1e-9;\nV(a, b) <+ x * ddt(I(a, b));",
        ] {
            let source = format!(
                "`include \"disciplines.vams\"\nmodule m(a, b);\ninout a, b;\nelectrical a, b;\nreal x, y;\nanalog begin\n{valid}\nend\nendmodule\n"
            );
            compile(&source, "m.va", &|_| None)
                .unwrap_or_else(|error| panic!("expected {valid:?} to compile, got: {error}"));
        }

        for dynamic_scale in [
            "I(a, b) <+ (1.0 + V(a, b)) * ddt(V(a, b));",
            "x = V(a, b);\nI(a, b) <+ x * ddt(V(a, b));",
            "if (V(a, b) > 0) x = 2e-12; else x = 1e-12;\nI(a, b) <+ x * ddt(V(a, b));",
            "I(a, b) <+ $simparam(\"gmin\") * ddt(V(a, b));",
        ] {
            assert_eq!(
                refusal(dynamic_scale),
                "ddt scaled by a dynamic expression is not charge-conserving and is not compiled yet",
                "{dynamic_scale}"
            );
        }

        assert_eq!(
            refusal("I(a, b) <+ ddt(V(a, b)) * ddt(V(a, b));"),
            "ddt is not compiled yet"
        );
        assert_eq!(
            refusal("I(a, b) <+ 1.0 / ddt(V(a, b));"),
            "ddt is not compiled yet"
        );
        assert_eq!(
            refusal("x = 2.0;\nI(a, b) <+ x * white_noise(1.0);"),
            "white_noise is not compiled yet"
        );
    }

    #[test]
    fn stmt_walk_visits_flicker_noise_exponent() {
        let reassigned_module = resolve_source(
            "real e;\n\
             analog begin\n\
                 e = 1.0;\n\
                 I(a, b) <+ V(a, b) / 100.0;\n\
                 I(a, b) <+ flicker_noise(1e-20, e, \"flicker\");\n\
                 e = 2.0;\n\
             end",
        );
        let e_index = reassigned_module
            .variables
            .iter()
            .position(|(name, _)| name == "e")
            .expect("variable e");
        assert!(
            reassigned_module.dynamic[e_index],
            "reassigned variable read by flicker_noise exponent must be marked dynamic"
        );
        assert_eq!(
            refusal(
                "x = 1.0;\n\
                 I(a, b) <+ V(a, b) / 100.0;\n\
                 I(a, b) <+ flicker_noise(1e-20, x, \"flicker\");\n\
                 x = 2.0;"
            ),
            "a flicker exponent that depends on the probes is not compiled yet"
        );

        let probed_module = resolve_source(
            "branch (a, b) br;\n\
             analog begin\n\
                 V(br) <+ 1.0;\n\
                 I(a, b) <+ flicker_noise(1e-20, I(br), \"flicker\");\n\
             end",
        );
        let br_index = probed_module
            .branches
            .iter()
            .position(|branch| branch.name == "br")
            .expect("branch br");
        assert!(
            probed_currents(&probed_module).contains(&br_index),
            "probed current inside flicker_noise exponent must be discovered by Stmt::walk"
        );
    }
}
