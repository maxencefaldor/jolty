//! The emitter: a resolved module written as a plain Rust kernel, the same form a person
//! writes, generic over jolty's number types.
//!
//! Its parameters become a `Parameters<R>` struct with their defaults; what depends on no
//! probe is computed in `prepare`, whose numbers the evaluation reads; the branches its
//! parameters switch, collapse or leave out are decided in `layout`, which claims the
//! internal unknowns and the entries of the shape they give; and `evaluate` runs the
//! analog block on jolty's dual numbers, each variable carrying the derivatives of the
//! probes it depends on and no others, then stamps each branch once.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use jolty_models::Nature;

use crate::Error;
use crate::ast::Type;
use crate::ir::{
    Branch, Builtin, Expr, GROUND, Limiter, Module, Parameter, Probe, Stmt, apart, constant, fold,
    probed_currents, visit_contributions,
};
use crate::lexer::CONSTANTS;
use crate::linear::{Form, Linearity};

/// The precedences of Rust's operators, from the loosest, as the printer parenthesizes
/// them.
const OR: u8 = 1;
const AND: u8 = 2;
const COMPARE: u8 = 3;
const SUM: u8 = 4;
const PRODUCT: u8 = 5;
const UNARY: u8 = 6;
const ATOM: u8 = 7;

/// The clippy bound on a function's lines, its `too-many-lines-threshold`, beyond which a
/// generated function says why it is long.
const LONGEST: usize = 100;

/// What a part of the generated code computes for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// The static numbers, in `prepare`.
    Prepare,
    /// The static numbers and the facts `$warning` and its like state, in `warnings`.
    Warnings,
    /// The contributions and their derivatives, in `evaluate`.
    Evaluate,
    /// The noise sources, in `noise`.
    Noise,
    /// An analog function that limits a probe, on plain floats.
    Function,
}

impl Mode {
    /// Whether the mode computes the static numbers, as preparation does.
    const fn prepares(self) -> bool {
        matches!(self, Self::Prepare | Self::Warnings)
    }
}

/// The kind of a generated expression.
#[derive(Debug, Clone, PartialEq)]
enum Kind {
    Bool,
    Integer,
    /// A plain float: a constant, which the compiler folds, or a number of the
    /// simulator's, such as gmin.
    Float(Option<f64>),
    /// A number of the kernel's type `R`, with no derivative by a probe.
    Real,
    /// A dual number carrying the derivatives by the probes listed, in their order.
    Dual(Vec<Probe>),
}

/// A generated expression: its code, the precedence of its outermost operator, and its
/// kind.
#[derive(Debug, Clone)]
struct Operand {
    code: String,
    precedence: u8,
    kind: Kind,
}

impl Operand {
    fn new(code: impl Into<String>, precedence: u8, kind: Kind) -> Self {
        Self {
            code: code.into(),
            precedence,
            kind,
        }
    }

    /// The code, parenthesized unless its precedence is at least `least`.
    fn at(&self, least: u8) -> String {
        if self.precedence >= least {
            self.code.clone()
        } else {
            format!("({})", self.code)
        }
    }
}

/// A generated statement, before the printer simplifies and indents it.
#[derive(Debug, Clone)]
enum Rust {
    Line(String),
    If((String, u8), Vec<Self>, Vec<Self>),
    While(String, Vec<Self>),
}

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

/// What an instance requires to be linear: nothing, the impossible, or the truth of
/// some code, with the precedence of its outermost operator.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Requirement {
    Always,
    Never,
    When(String, u8),
}

impl Requirement {
    /// The code, parenthesized unless its precedence is at least `least`.
    fn at(code: &str, precedence: u8, least: u8) -> String {
        if precedence >= least {
            code.to_owned()
        } else {
            format!("({code})")
        }
    }

    /// Both this requirement and `other`.
    fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::Never, _) | (_, Self::Never) => Self::Never,
            (Self::Always, other) | (other, Self::Always) => other,
            (Self::When(first, a), Self::When(second, b)) => Self::When(
                format!(
                    "{} && {}",
                    Self::at(&first, a, AND),
                    Self::at(&second, b, AND + 1)
                ),
                AND,
            ),
        }
    }

    /// `first` where `condition` holds, and `second` where it does not.
    fn choose(condition: &str, first: Self, second: Self) -> Self {
        let not = format!("!{condition}");
        match (first, second) {
            (Self::Always, Self::Always) => Self::Always,
            (Self::Never, Self::Never) => Self::Never,
            (Self::Always, Self::Never) => Self::When(condition.to_owned(), ATOM),
            (Self::Never, Self::Always) => Self::When(not, UNARY),
            (Self::Always, Self::When(code, precedence)) => Self::When(
                format!("{condition} || {}", Self::at(&code, precedence, OR + 1)),
                OR,
            ),
            (Self::When(code, precedence), Self::Always) => Self::When(
                format!("{not} || {}", Self::at(&code, precedence, OR + 1)),
                OR,
            ),
            (Self::Never, Self::When(code, precedence)) => Self::When(
                format!("{not} && {}", Self::at(&code, precedence, AND + 1)),
                AND,
            ),
            (Self::When(code, precedence), Self::Never) => Self::When(
                format!("{condition} && {}", Self::at(&code, precedence, AND + 1)),
                AND,
            ),
            (Self::When(first, _), Self::When(second, _)) => Self::When(
                format!("if {condition} {{ {first} }} else {{ {second} }}"),
                0,
            ),
        }
    }
}

/// The rows and columns of a branch's stamp: a node, or the current of a branch that
/// holds a voltage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Place {
    Node(usize),
    Current(usize),
}

/// What a contribution makes its branch: a flow, a potential, or a collapse of its
/// internal node into the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Site {
    Flow,
    Potential,
    Collapse,
}

/// What the generated code reads of the prepared numbers: each field's name, the kind
/// it has, and the static expression `prepare` computes it as.
#[derive(Debug, Default)]
struct Prepared {
    fields: Vec<(String, Kind, Expr)>,
}

impl Prepared {
    /// The field for `expr`, added once, named apart from earlier fields and
    /// `conditions`.
    fn field(&mut self, name: impl FnOnce(usize) -> String, kind: Kind, expr: &Expr) -> String {
        if let Some((name, _, _)) = self.fields.iter().find(|(_, _, found)| found == expr) {
            return name.clone();
        }
        let base = name(self.fields.len());
        let name = apart(&base, |taken| {
            taken == "conditions" || self.fields.iter().any(|(existing, _, _)| existing == taken)
        });
        self.fields.push((name.clone(), kind, expr.clone()));
        name
    }
}

/// Whether a branch's flow or potential, or its collapse, is active in an instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Activity {
    Never,
    Always,
    /// As the flag at that index among the shape's says.
    Flag(usize),
}

impl Activity {
    /// The index of its flag, if the shape decides it.
    const fn index(self) -> Option<usize> {
        match self {
            Self::Flag(index) => Some(index),
            Self::Never | Self::Always => None,
        }
    }
}

/// A module's facts the emitter reads throughout.
struct Facts<'m> {
    module: &'m Module,
    /// The conditionals whose conditions are static and that hold a contribution to a
    /// branch's flow or potential, by number, with their index among `conditions`.
    shaped: BTreeMap<usize, usize>,
    /// The distinct static conditions that shape the kernel, in order of first appearance.
    conditions: Vec<Expr>,
    /// Whether each branch adds to the equations only under some of the shape's
    /// conditions, with its index among the shape's flags, for its flow or potential and
    /// for its collapse.
    flags: BTreeMap<(usize, Site), usize>,
    /// The message of each warning that depends on the probes, which the memory flags
    /// once an evaluation finds it, and the number of values it formats, which the
    /// memory holds as the last evaluation that found it read them.
    warned: Vec<(String, usize)>,
    /// The branches whose currents are read.
    probed: BTreeSet<usize>,
}

/// The emitter of one function of the kernel.
struct Emitter<'f, 'm> {
    facts: &'f Facts<'m>,
    mode: Mode,
    /// The analog block as this mode runs it: each assignment that no statement after it
    /// reads left out.
    analog: Vec<Stmt>,
    /// Whether each variable is kept, in this mode.
    live: Vec<bool>,
    /// The function being emitted, in [`Mode::Function`].
    function: Option<usize>,
    temporaries: usize,
    prepared: &'f mut Prepared,
    /// The probes read, which the function binds first.
    probes: BTreeSet<Probe>,
    /// The fields of the shape read.
    shape: BTreeSet<String>,
    /// The entries of the stamp, which the shape holds, each by its row and column.
    entries: &'f mut BTreeMap<(Place, Place), BTreeSet<(usize, Site)>>,
    /// The limited probes whose applied and limited voltages the stamp extends along.
    extended: BTreeSet<usize>,
    /// The parameters read, in preparation.
    parameters_read: BTreeSet<usize>,
    /// The non-constant `localparam`s read, in preparation.
    locals_read: BTreeSet<usize>,
    /// Whether `$temperature` was read, in preparation.
    uses_temperature: bool,
    /// The kind of each branch's accumulated flow or potential, and of its charge or
    /// flux, by the branch and whether it is the latter.
    accumulators: BTreeMap<(usize, Site, bool), Kind>,
}

/// The rendered body of a generated `evaluate` or `noise` method, before its signature
/// is formatted with the finalized `Prepared` type.
struct MethodBody {
    shape: &'static str,
    prepared: &'static str,
    at: &'static str,
    memory: String,
    lines: Vec<String>,
}

/// The kernel of `module`, compiled from `source`, as the text of a Rust module.
///
/// # Errors
///
/// Returns an [`Error`] for a construct the emitter does not compile yet.
pub(crate) fn emit(module: &Module, (source, header): (&str, &str)) -> Result<String, Error> {
    let facts = Facts::new(module)?;
    let mut prepared = Prepared::default();
    let mut entries = BTreeMap::new();
    let evaluate_body =
        Emitter::new(&facts, Mode::Evaluate, &mut prepared, &mut entries).evaluate_body()?;
    let noise_body = if module.analog.iter().any(makes_noise) {
        Some(Emitter::new(&facts, Mode::Noise, &mut prepared, &mut entries).noise_body()?)
    } else {
        None
    };
    let linear_body =
        Emitter::new(&facts, Mode::Evaluate, &mut prepared, &mut entries).linear_body();
    let parameters_generic = module
        .parameters
        .iter()
        .any(|parameter| parameter.kind == Type::Real);
    let prepared_generic = prepared
        .fields
        .iter()
        .any(|(_, kind, _)| matches!(kind, Kind::Real | Kind::Dual(_)));
    let names = Names::from_module(&module.name, parameters_generic, prepared_generic);
    let evaluate = format_evaluate(&evaluate_body, &names);
    let noise = noise_body.as_ref().map(|body| format_noise(body, &names));
    let linear = linear_body
        .as_deref()
        .map_or_else(String::new, |body| format_linear(body, &names));
    let mut out = vec![facts.header((source, header), &names)];
    out.push(facts.parameters(&names));
    let layout = facts.layout(&entries, &names);
    let prepare = facts.prepare(Mode::Prepare, &prepared, &names)?;
    let warnings = facts.warnings_impl(&prepared, &names)?;
    out.push(facts.prepared_struct(&prepared, &names));
    out.push(facts.shape_struct(&entries, &names));
    out.push(facts.memory_struct(&names));
    out.push(facts.functions()?);
    out.push(format!("impl Kernel for {} {{\n", names.kernel));
    out.push(format!(
        "    const NAME: &'static str = {:?};\n",
        module.name
    ));
    out.push(format!(
        "    type Parameters<R: Real> = {};\n",
        names.parameters_type("R")
    ));
    out.push(format!("    type Shape = {};\n", names.shape));
    out.push(format!(
        "    type Prepared<R: Real> = {};\n",
        names.prepared_type("R")
    ));
    let memory = if facts.remembers() {
        names.memory.clone()
    } else {
        "()".to_owned()
    };
    out.push(format!("    type Memory = {memory};\n\n"));
    out.push(facts.interface(&names));
    out.push(facts.outputs_fns(&names));
    out.push(linear);
    out.push(prepare);
    out.push(layout);
    out.push(evaluate);
    if let Some(noise) = noise {
        out.push(noise);
    }
    out.push("}\n".to_owned());
    if !warnings.is_empty() {
        out.push(format!("\n{warnings}"));
    }
    Ok(imports(&out.concat()))
}

fn format_linear(body: &str, names: &Names) -> String {
    let shape = if mentions(body, "shape") {
        "shape"
    } else {
        "_"
    };
    let prepared = if mentions(body, "prepared") {
        "prepared"
    } else {
        "_"
    };
    format!(
        "    fn linear({shape}: &{}, {prepared}: &{}) -> bool {{\n        {body}\n    }}\n\n",
        names.shape,
        names.prepared_type("f64")
    )
}

fn format_evaluate(body: &MethodBody, names: &Names) -> String {
    format!(
        "{}    fn evaluate<R: Real>(\n        {}: &{},\n        {}: &{},\n        {},\n        at: &Local<'_>,\n        stamp: &mut Stamp<'_, R>,\n    ) {{\n{}\n    }}\n\n",
        too_long(&body.lines, "    "),
        body.shape,
        names.shape,
        body.prepared,
        names.prepared_type("R"),
        body.memory,
        body.lines.join("\n")
    )
}

fn format_noise(body: &MethodBody, names: &Names) -> String {
    format!(
        "{}    fn noise<R: Real>(\n        {}: &{},\n        {}: &{},\n        {}: &Local<'_>,\n        sources: &mut Vec<NoiseSource>,\n    ) {{\n{}\n    }}\n",
        too_long(&body.lines, "    "),
        body.shape,
        names.shape,
        body.prepared,
        names.prepared_type("R"),
        body.at,
        body.lines.join("\n")
    )
}

/// `text` with the imports its code uses, after its module documentation.
fn imports(text: &str) -> String {
    let candidates = [
        "BOLTZMANN",
        "Branch",
        "BranchKind",
        "Dual",
        "ELEMENTARY_CHARGE",
        "Interface",
        "Kernel",
        "Layout",
        "Lift",
        "Local",
        "Nature",
        "NoiseSource",
        "ParameterError",
        "Real",
        "SMALL_RESISTANCE",
        "Stamp",
        "UnknownIndex",
        "limit_gate",
        "limit_junction",
    ];
    let end = ")]\n\n";
    let (documentation, rest) = text.split_at(text.find(end).map_or(0, |at| at + end.len()));
    let used: Vec<&str> = candidates
        .into_iter()
        .filter(|name| mentions(rest, name))
        .collect();
    let mut out = vec![documentation.to_owned()];
    if mentions(rest, "Ordering") {
        out.push("use std::cmp::Ordering;\n\n".to_owned());
    }
    out.push(format!("use jolty_models::{{{}}};\n", used.join(", ")));
    out.push("use schemars::JsonSchema;\nuse serde::{Deserialize, Serialize};\n\n".to_owned());
    if mentions(rest, "NOWHERE") {
        out.push("/// A local index no instance has: a write to it goes nowhere, and a read of it reads\n/// zero, as `Stamp` and `Local` promise.\nconst NOWHERE: usize = usize::MAX;\n\n".to_owned());
    }
    out.push(rest.to_owned());
    out.concat()
}

/// Whether `text` holds `name` as a whole word.
fn mentions(text: &str, name: &str) -> bool {
    text.match_indices(name).any(|(at, _)| {
        let before = text[..at].chars().next_back();
        let after = text[at + name.len()..].chars().next();
        let word = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '_');
        !word(before) && !word(after)
    })
}

/// Adds the message of each warning of `statements` that depends on the probes to
/// `warned`, once, with the number of values it formats.
fn collect_warned(statements: &[Stmt], warned: &mut Vec<(String, usize)>) {
    for statement in statements {
        match statement {
            Stmt::Warn(message, arguments, true) => {
                let found = (message.clone(), arguments.len());
                if !warned.contains(&found) {
                    warned.push(found);
                }
            }
            Stmt::If(_, _, then, otherwise) => {
                collect_warned(then, warned);
                collect_warned(otherwise, warned);
            }
            Stmt::While(_, body) => collect_warned(body, warned),
            Stmt::Assign(..) | Stmt::Contribute { .. } | Stmt::Fail(..) | Stmt::Warn(..) => {}
        }
    }
}

/// Whether `statement` is a `$warning` or its like, or holds one.
fn warns(statement: &Stmt) -> bool {
    match statement {
        Stmt::Warn(_, _, dynamic) => !dynamic,
        Stmt::If(_, _, then, otherwise) => then.iter().chain(otherwise).any(warns),
        Stmt::While(_, body) => body.iter().any(warns),
        Stmt::Assign(..) | Stmt::Contribute { .. } | Stmt::Fail(..) => false,
    }
}

/// Whether `statement` adds to a branch's flow or potential, or holds one that does.
fn stamps(statement: &Stmt) -> bool {
    match statement {
        Stmt::Contribute {
            resistive,
            reactive,
            ..
        } => resistive.is_some() || reactive.is_some(),
        Stmt::If(_, _, then, otherwise) => then.iter().chain(otherwise).any(stamps),
        Stmt::While(_, body) => body.iter().any(stamps),
        Stmt::Assign(..) | Stmt::Fail(..) | Stmt::Warn(..) => false,
    }
}

/// Whether `statement` adds a noise source, or holds one that does.
fn makes_noise(statement: &Stmt) -> bool {
    match statement {
        Stmt::Contribute { noise, .. } => !noise.is_empty(),
        Stmt::If(_, _, then, otherwise) => then.iter().chain(otherwise).any(makes_noise),
        Stmt::While(_, body) => body.iter().any(makes_noise),
        Stmt::Assign(..) | Stmt::Fail(..) | Stmt::Warn(..) => false,
    }
}

/// The names of a kernel's types.
struct Names {
    kernel: String,
    parameters: String,
    shape: String,
    prepared: String,
    memory: String,
    parameters_generic: bool,
    prepared_generic: bool,
}

impl Names {
    /// The names of the kernel of `module`, in upper camel case, each word of the
    /// module's name, which its underscores part, capitalized as Rust capitalizes an
    /// acronym.
    fn from_module(module: &str, parameters_generic: bool, prepared_generic: bool) -> Self {
        let kernel: String = module
            .split('_')
            .map(|word| {
                let mut characters = word.chars();
                characters.next().map_or_else(String::new, |first| {
                    first
                        .to_uppercase()
                        .chain(characters.flat_map(char::to_lowercase))
                        .collect()
                })
            })
            .collect();
        Self {
            parameters: format!("{kernel}Parameters"),
            shape: format!("{kernel}Shape"),
            prepared: format!("Prepared{kernel}"),
            memory: format!("{kernel}Memory"),
            kernel,
            parameters_generic,
            prepared_generic,
        }
    }

    fn parameters_type(&self, number: &str) -> String {
        if self.parameters_generic {
            format!("{}<{number}>", self.parameters)
        } else {
            self.parameters.clone()
        }
    }

    fn prepared_type(&self, number: &str) -> String {
        if self.prepared_generic {
            format!("{}<{number}>", self.prepared)
        } else {
            self.prepared.clone()
        }
    }
}

/// A float written as Rust writes it, its long runs of digits grouped by three, as
/// clippy's `unreadable_literal` asks.
fn literal(value: f64) -> String {
    let named = CONSTANTS.iter().find(|(_, name, constant)| {
        constant.to_bits() == value.abs().to_bits()
            && !name.starts_with(|c: char| c.is_ascii_digit())
    });
    if let Some((_, name, _)) = named {
        return if value.is_sign_negative() {
            format!("-{name}")
        } else {
            (*name).to_owned()
        };
    }
    let written = format!("{value:?}");
    let (mantissa, exponent) = match written.find('e') {
        Some(at) => written.split_at(at),
        None => (written.as_str(), ""),
    };
    let (sign, mantissa) = mantissa
        .strip_prefix('-')
        .map_or(("", mantissa), |rest| ("-", rest));
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    // A part of five digits or more is grouped, and then the other too, if it has more
    // than three, as clippy's `inconsistent_digit_grouping` asks.
    let grouped = whole.len() >= 5 || fraction.len() >= 5;
    let group = |digits: &str, from_left: bool| -> String {
        if !grouped || digits.len() <= 3 {
            return digits.to_owned();
        }
        let characters: Vec<char> = digits.chars().collect();
        let chunks: Vec<String> = if from_left {
            characters
                .chunks(3)
                .map(|chunk| chunk.iter().collect())
                .collect()
        } else {
            let mut chunks: Vec<String> = characters
                .rchunks(3)
                .map(|chunk| chunk.iter().collect())
                .collect();
            chunks.reverse();
            chunks
        };
        chunks.join("_")
    };
    let fraction = if fraction.is_empty() {
        String::new()
    } else {
        format!(".{}", group(fraction, true))
    };
    let fraction = if fraction.is_empty() && exponent.is_empty() {
        ".0".to_owned()
    } else {
        fraction
    };
    format!("{sign}{}{fraction}{exponent}", group(whole, false))
}

/// The operand of the float `value`.
fn float(value: f64) -> Operand {
    let precedence = if value.is_sign_negative() {
        UNARY
    } else {
        ATOM
    };
    Operand::new(literal(value), precedence, Kind::Float(Some(value)))
}

/// The probes of `left` and of `right`, together, in order.
fn union(left: &[Probe], right: &[Probe]) -> Vec<Probe> {
    let mut all: BTreeSet<Probe> = left.iter().copied().collect();
    all.extend(right.iter().copied());
    all.into_iter().collect()
}

impl<'m> Facts<'m> {
    fn new(module: &'m Module) -> Result<Self, Error> {
        let mut facts = Self {
            module,
            shaped: BTreeMap::new(),
            conditions: Vec::new(),
            flags: BTreeMap::new(),
            warned: Vec::new(),
            probed: probed_currents(module),
        };
        collect_warned(&module.analog, &mut facts.warned);
        facts.find_shapes(&module.analog);
        facts.find_roles(&module.analog, false)?;
        static_tasks(&module.analog, module, false)?;
        first_order(module)?;
        Ok(facts)
    }

    /// Numbers the conditionals of `statements` that shape the kernel, sharing an index
    /// among identical condition expressions.
    fn find_shapes(&mut self, statements: &[Stmt]) {
        for statement in statements {
            match statement {
                Stmt::If(id, condition, then, otherwise) => {
                    let holds = then.iter().chain(otherwise).any(stamps);
                    if holds && !self.module.is_dynamic(condition) {
                        let index = if let Some(index) = self
                            .conditions
                            .iter()
                            .position(|existing| existing == condition)
                        {
                            index
                        } else {
                            let index = self.conditions.len();
                            self.conditions.push(condition.clone());
                            index
                        };
                        self.shaped.insert(*id, index);
                    }
                    self.find_shapes(then);
                    self.find_shapes(otherwise);
                }
                Stmt::While(_, body) => self.find_shapes(body),
                Stmt::Assign(..) | Stmt::Contribute { .. } | Stmt::Fail(..) | Stmt::Warn(..) => {}
            }
        }
    }

    /// Whether a contribution collapses its branch: a potential of zero across a branch
    /// with an internal node.
    fn collapses(
        &self,
        branch: usize,
        potential: bool,
        resistive: Option<&Expr>,
        reactive: Option<&Expr>,
    ) -> bool {
        let Branch { plus, minus, .. } = self.module.branches[branch];
        // The node that collapses: the later of two, or the one node of a branch to the
        // ground, which the ground ranks before. A branch whose current is read keeps it
        // as an unknown instead.
        let collapsing = if minus == GROUND {
            plus
        } else {
            plus.max(minus)
        };
        potential
            && reactive.is_none()
            && resistive.and_then(constant) == Some(0.0)
            && collapsing >= self.module.ports
            && !self.probed.contains(&branch)
    }

    /// What a contribution to `branch` makes it: a collapse, a potential or a flow.
    fn site(
        &self,
        branch: usize,
        potential: bool,
        resistive: Option<&Expr>,
        reactive: Option<&Expr>,
    ) -> Site {
        if self.collapses(branch, potential, resistive, reactive) {
            Site::Collapse
        } else if potential {
            Site::Potential
        } else {
            Site::Flow
        }
    }

    /// Finds each branch's role and which branches the shape switches, `shaped` saying
    /// whether `statements` lie under a shape's condition.
    fn find_roles(&mut self, statements: &[Stmt], shaped: bool) -> Result<(), Error> {
        for statement in statements {
            match statement {
                Stmt::Contribute {
                    branch,
                    potential,
                    resistive,
                    reactive,
                    ..
                } if resistive.is_some() || reactive.is_some() => {
                    let site =
                        self.site(*branch, *potential, resistive.as_ref(), reactive.as_ref());
                    let Branch { plus, ref name, .. } = self.module.branches[*branch];
                    if site == Site::Potential && self.module.natures[plus] == Nature::Temperature {
                        return Err(Error::at(
                            0,
                            format!(
                                "the thermal branch {name} holds a temperature, which is not compiled yet"
                            ),
                        ));
                    }
                    let key = (*branch, site);
                    if shaped {
                        if !self.flags.contains_key(&key) {
                            let index = self.flags.len();
                            self.flags.insert(key, index);
                        }
                    } else {
                        self.flags.insert(key, usize::MAX);
                    }
                }
                Stmt::If(id, _, then, otherwise) => {
                    let inner = shaped || self.shaped.contains_key(id);
                    self.find_roles(then, inner)?;
                    self.find_roles(otherwise, inner)?;
                }
                Stmt::While(_, body) => self.find_roles(body, shaped)?,
                Stmt::Assign(..) | Stmt::Contribute { .. } | Stmt::Fail(..) | Stmt::Warn(..) => {}
            }
        }
        // An unconditional site makes its branch always active: no flag.
        if !shaped {
            let mut index = 0;
            for value in self.flags.values_mut() {
                if *value != usize::MAX {
                    *value = index;
                    index += 1;
                }
            }
        }
        Ok(())
    }

    /// Whether `branch`'s flow, its potential or its collapse is active: never, always,
    /// or as the shape's flag says.
    fn flag(&self, branch: usize, site: Site) -> Activity {
        match self.flags.get(&(branch, site)) {
            None => Activity::Never,
            Some(&usize::MAX) => Activity::Always,
            Some(&index) => Activity::Flag(index),
        }
    }

    /// Whether `branch` holds a potential in some shape, its current then an unknown.
    fn holds_potential(&self, branch: usize) -> bool {
        self.flags.contains_key(&(branch, Site::Potential))
    }

    /// Whether `branch` carries a flow in some shape.
    fn carries_flow(&self, branch: usize) -> bool {
        self.flags.contains_key(&(branch, Site::Flow))
    }

    fn flag_count(&self) -> usize {
        self.flags
            .values()
            .filter(|&&index| index != usize::MAX)
            .count()
    }

    /// The local index of node `node`, as code that reads it from `holder`, the shape or
    /// the layout's own bindings.
    fn node(&self, node: usize, holder: &str) -> String {
        if node == GROUND {
            "NOWHERE".to_owned()
        } else if node < self.module.ports {
            node.to_string()
        } else {
            format!("{holder}{}", self.module.nodes[node])
        }
    }

    /// The local index of `place`, as code that reads it from `holder`.
    fn place(&self, place: Place, holder: &str) -> String {
        match place {
            Place::Node(node) => self.node(node, holder),
            Place::Current(branch) => format!("{holder}{}", self.module.branches[branch].name),
        }
    }

    /// The name of the shape's field of the entry at `row` and `column`.
    fn entry_name(&self, (row, column): (Place, Place)) -> String {
        let name = |place: Place| match place {
            Place::Node(node) => self.module.node_name(node).to_owned(),
            Place::Current(branch) => self.module.branches[branch].name.clone(),
        };
        format!("{}_by_{}", name(row), name(column))
    }

    fn header(&self, (source, notice): (&str, &str), names: &Names) -> String {
        let notice: String = std::iter::once("```text")
            .chain(notice.lines())
            .chain(std::iter::once("```"))
            .map(|line| {
                if line.is_empty() {
                    "//!\n".to_owned()
                } else {
                    format!("//! {line}\n")
                }
            })
            .collect();
        format!(
            "//! The `{}` kernel, compiled by jolty-veriloga from `{source}`.\n//!\n//! Written by `cargo run -p jolty-veriloga --bin compile`, never by hand. Its source says:\n//!\n{notice}\n#![cfg_attr(\n    not(test),\n    deny(\n        clippy::indexing_slicing,\n        clippy::panic,\n        reason = \"a kernel never panics, and reads fixed arrays through `get`\"\n    )\n)]\n\n/// The `{}` model, compiled from its Verilog-A source.\n#[derive(Debug, Clone, Copy)]\npub struct {};\n\n",
            self.module.name, self.module.name, names.kernel
        )
    }

    fn parameters(&self, names: &Names) -> String {
        let derive_default = !names.parameters_generic
            && self.module.parameters.iter().all(|parameter| {
                parameter.optional
                    || matches!(constant(&parameter.default), Some(value) if value == 0.0)
            });
        let derives = if derive_default {
            "Debug, Clone, Default, Deserialize, Serialize, JsonSchema, Lift"
        } else {
            "Debug, Clone, Deserialize, Serialize, JsonSchema, Lift"
        };
        let serde_attribute = if self.module.parameters.is_empty() {
            "#[serde(deny_unknown_fields)]"
        } else {
            "#[serde(default, deny_unknown_fields)]"
        };
        let schemars_attribute = if names.parameters_generic {
            "#[schemars(bound = \"R: Serialize + JsonSchema\")]\n"
        } else {
            ""
        };
        let type_parameters = if names.parameters_generic {
            "<R: Real = f64>"
        } else {
            ""
        };
        if self.module.parameters.is_empty() {
            return format!(
                "/// The parameters of a [`{}`], with its Verilog-A source's defaults.\n#[derive({derives})]\n{serde_attribute}\n{schemars_attribute}pub struct {}{type_parameters} {{}}\n\n",
                names.kernel, names.parameters
            );
        }
        let mut out = vec![format!(
            "/// The parameters of a [`{}`], with its Verilog-A source's defaults.\n#[derive({derives})]\n{serde_attribute}\n{schemars_attribute}pub struct {}{type_parameters} {{\n",
            names.kernel, names.parameters
        )];
        let mut defaults = Vec::new();
        for parameter in &self.module.parameters {
            let (field, default_entry) = parameter_field(parameter);
            out.push(field);
            defaults.push(default_entry);
        }
        out.push("}\n\n".to_owned());
        if !derive_default {
            // The function's lines, as clippy counts them: its struct's two and a default each.
            let long = too_long(
                &[defaults.clone(), vec!["Self {".to_owned(), "}".to_owned()]].concat(),
                "    ",
            );
            let impl_header = if names.parameters_generic {
                format!("impl<R: Real> Default for {}<R>", names.parameters)
            } else {
                format!("impl Default for {}", names.parameters)
            };
            out.push(format!(
                "{impl_header} {{\n{long}    fn default() -> Self {{\n        Self {{\n{}        }}\n    }}\n}}\n\n",
                defaults.concat()
            ));
        }
        out.concat()
    }

    fn interface(&self, names: &Names) -> String {
        let ports = self.module.ports;
        // Which nodes each branch joins, and whether it conducts at DC.
        let mut edges: Vec<(usize, usize, bool)> = Vec::new();
        for (index, branch) in self.module.branches.iter().enumerate() {
            let collapses = self.flags.contains_key(&(index, Site::Collapse));
            let stamps = self.carries_flow(index) || self.holds_potential(index);
            let conducts =
                collapses || self.holds_potential(index) || (stamps && self.conducts(index));
            let joins = stamps || collapses;
            if joins && branch.minus != GROUND {
                edges.push((branch.plus, branch.minus, conducts));
            }
        }
        let mut branches = Vec::new();
        for from in 0..ports {
            for to in from + 1..ports {
                let mut kind = None;
                for conducting in [true, false] {
                    if kind.is_none() && Self::joined(&edges, (from, to), ports, conducting) {
                        kind = Some(if conducting {
                            "Conductive"
                        } else {
                            "Capacitive"
                        });
                    }
                }
                if let Some(kind) = kind {
                    branches.push(format!("Branch::new({from}, {to}, BranchKind::{kind})"));
                }
            }
        }
        let terminals: Vec<String> = self.module.labels[..ports]
            .iter()
            .map(|name| format!("{name:?}"))
            .collect();
        let thermal: Vec<String> = (0..ports)
            .filter(|&port| self.module.natures[port] == Nature::Temperature)
            .map(|port| port.to_string())
            .collect();
        let interface = format!(
            "Interface::new(&[{}], &[{}])",
            terminals.join(", "),
            branches.join(", ")
        );
        let interface = if thermal.is_empty() {
            interface
        } else {
            format!(
                "Interface {{\n            thermal: vec![{}],\n            ..{interface}\n        }}",
                thermal.join(", ")
            )
        };
        format!(
            "    fn interface<R: Real>(_: &{}) -> Interface {{\n        {interface}\n    }}\n\n",
            names.parameters_type("R")
        )
    }

    /// Whether `branch` has a resistive part, so that it conducts at DC.
    fn conducts(&self, branch: usize) -> bool {
        let mut found = false;
        visit_contributions(&self.module.analog, &mut |statement| {
            if let Stmt::Contribute {
                branch: to,
                resistive: Some(_),
                ..
            } = statement
                && *to == branch
            {
                found = true;
            }
        });
        found
    }

    /// Whether terminals `from` and `to` are joined through internal nodes alone, by
    /// edges that conduct at DC when `conducting` says so.
    fn joined(
        edges: &[(usize, usize, bool)],
        (from, to): (usize, usize),
        ports: usize,
        conducting: bool,
    ) -> bool {
        let mut reached = BTreeSet::from([from]);
        let mut frontier = vec![from];
        while let Some(node) = frontier.pop() {
            for &(plus, minus, conducts) in edges {
                if conducting && !conducts {
                    continue;
                }
                let other = if plus == node {
                    minus
                } else if minus == node {
                    plus
                } else {
                    continue;
                };
                if other == to {
                    return true;
                }
                if other >= ports && reached.insert(other) {
                    frontier.push(other);
                }
            }
        }
        false
    }

    /// The kernel's output variables: their names, and how they are read from its memory.
    fn outputs_fns(&self, names: &Names) -> String {
        let module = self.module;
        if module.outputs.is_empty() {
            return String::new();
        }
        let outputs: Vec<&str> = module
            .outputs
            .iter()
            .map(|&output| module.variables[output].0.as_str())
            .collect();
        let quoted: Vec<String> = outputs.iter().map(|name| format!("{name:?}")).collect();
        let kept: Vec<String> = outputs
            .iter()
            .map(|name| format!("memory.{name}"))
            .collect();
        format!(
            "    fn variable_names() -> &'static [&'static str] {{\n        &[{}]\n    }}\n\n    fn variables(memory: &{}, values: &mut [f64]) {{\n        for (value, kept) in values.iter_mut().zip([{}]) {{\n            *value = kept;\n        }}\n    }}\n\n",
            quoted.join(", "),
            names.memory,
            kept.join(", ")
        )
    }

    fn prepared_struct(&self, prepared: &Prepared, names: &Names) -> String {
        let truths = prepared
            .fields
            .iter()
            .filter(|(_, kind, _)| *kind == Kind::Bool)
            .count();
        let excused = if truths > 3 {
            "#[expect(\n    clippy::struct_excessive_bools,\n    reason = \"the source's conditions on its parameters, each prepared once\"\n)]\n"
        } else {
            ""
        };
        let type_parameters = if names.prepared_generic {
            "<R: Real>"
        } else {
            ""
        };
        if prepared.fields.is_empty() && self.conditions.is_empty() {
            return format!(
                "/// A [`{}`]'s numbers at its temperature: what depends on no probe.\n#[derive(Debug, Clone, Copy)]\npub struct {}{type_parameters} {{}}\n\n",
                names.kernel, names.prepared
            );
        }
        let mut out = vec![format!(
            "/// A [`{}`]'s numbers at its temperature: what depends on no probe.\n#[derive(Debug, Clone, Copy)]\n{excused}pub struct {}{type_parameters} {{\n",
            names.kernel, names.prepared
        )];
        for (name, kind, _) in &prepared.fields {
            out.push(format!("    {name}: {},\n", rust_type(kind)));
        }
        if !self.conditions.is_empty() {
            out.push(format!(
                "    /// The conditions that shape the kernel.\n    conditions: [bool; {}],\n",
                self.conditions.len()
            ));
        }
        out.push("}\n\n".to_owned());
        out.concat()
    }

    fn shape_struct(
        &self,
        entries: &BTreeMap<(Place, Place), BTreeSet<(usize, Site)>>,
        names: &Names,
    ) -> String {
        let mut out = vec![format!(
            "/// Where a [`{}`]'s contributions go among its local unknowns, and the conditions\n/// that chose them.\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct {} {{\n",
            names.kernel, names.shape
        )];
        for node in &self.module.nodes[self.module.ports..] {
            out.push(format!("    {node}: usize,\n"));
        }
        for (index, branch) in self.module.branches.iter().enumerate() {
            if self.holds_potential(index) {
                out.push(format!("    {}: usize,\n", branch.name));
            }
        }
        for key in entries.keys() {
            out.push(format!("    {}: usize,\n", self.entry_name(*key)));
        }
        if !self.conditions.is_empty() {
            out.push(format!(
                "    conditions: [bool; {}],\n",
                self.conditions.len()
            ));
        }
        if self.flag_count() > 0 {
            out.push(format!("    active: [bool; {}],\n", self.flag_count()));
        }
        out.push("}\n\n".to_owned());
        out.concat()
    }

    /// Whether the kernel remembers anything: limited voltages or output variables.
    fn remembers(&self) -> bool {
        !self.module.limits.is_empty() || !self.module.outputs.is_empty() || !self.warned.is_empty()
    }

    fn memory_struct(&self, names: &Names) -> String {
        if !self.remembers() {
            return String::new();
        }
        let mut parts = Vec::new();
        if !self.module.limits.is_empty() {
            parts.push("each limited voltage, which the next is limited from");
        }
        if !self.module.outputs.is_empty() {
            parts.push("each output variable");
        }
        if !self.warned.is_empty() {
            parts.push("each warning that depends on the probes, once found");
        }
        let held = parts.join("; ");
        let mut out = vec![format!(
            "/// What a [`{}`] remembers of its last evaluation: {held}.\n#[derive(Debug, Clone, Copy, Default)]\npub struct {} {{\n",
            names.kernel, names.memory
        )];
        for index in 0..self.module.limits.len() {
            out.push(format!("    limited{index}: f64,\n"));
        }
        for &output in &self.module.outputs {
            out.push(format!("    {}: f64,\n", self.module.variables[output].0));
        }
        if !self.warned.is_empty() {
            out.push(format!(
                "    /// Whether an evaluation found each warning that depends on the probes, as\n    /// [`{}::warned`] names them.\n    warned: [bool; {}],\n",
                names.kernel,
                self.warned.len()
            ));
        }
        let values: usize = self.warned.iter().map(|(_, count)| count).sum();
        if values > 0 {
            out.push(format!(
                "    /// The values each of those warnings formats, as the last evaluation that\n    /// found it read them.\n    warned_values: [f64; {values}],\n"
            ));
        }
        out.push("}\n\n".to_owned());
        out.concat()
    }

    /// The analog functions that limit probes, each a function of plain floats.
    fn functions(&self) -> Result<String, Error> {
        let mut out = Vec::new();
        let used: BTreeSet<usize> = self
            .module
            .limits
            .iter()
            .filter_map(|limit| match limit.function {
                Limiter::Function(function) => Some(function),
                Limiter::Junction | Limiter::Gate => None,
            })
            .collect();
        for &index in &used {
            let function = &self.module.functions[index];
            let mut prepared = Prepared::default();
            let mut entries = BTreeMap::new();
            let mut emitter = Emitter::new(self, Mode::Function, &mut prepared, &mut entries);
            emitter.function = Some(index);
            emitter.live = vec![true; function.variables.len()];
            let mut body = Vec::new();
            emitter.statements(&function.body, &mut body)?;
            let locals: Vec<(String, Kind)> = function.variables[function.inputs..]
                .iter()
                .map(|(name, kind)| {
                    (
                        name.clone(),
                        if *kind == Type::Integer {
                            Kind::Integer
                        } else {
                            Kind::Float(None)
                        },
                    )
                })
                .collect();
            let mut lines = locals_struct(&locals);
            print(&body, 0, &mut lines);
            lines.push(format!("v.{}", function.name));
            let lines: Vec<String> = lines
                .into_iter()
                .map(|line| format!("    {line}"))
                .collect();
            let inputs: Vec<String> = function.variables[..function.inputs]
                .iter()
                .map(|(name, kind)| {
                    format!(
                        "{name}: {}",
                        if *kind == Type::Integer { "i32" } else { "f64" }
                    )
                })
                .collect();
            out.push(format!(
                "/// The Verilog-A analog function `{}`, which limits a voltage's step.\n{}fn {}({}) -> f64 {{\n{}\n}}\n\n",
                function.name,
                too_long(&lines, ""),
                function.name,
                inputs.join(", "),
                lines.join("\n")
            ));
        }
        Ok(out.concat())
    }

    fn layout(
        &self,
        entries: &BTreeMap<(Place, Place), BTreeSet<(usize, Site)>>,
        names: &Names,
    ) -> String {
        let module = self.module;
        let mut lines = Vec::new();
        let flags = self.flag_count();
        if flags > 0 {
            lines.push(format!("    let mut active = [false; {flags}];"));
            let mut walked = Vec::new();
            self.walk_layout(&module.analog, &mut walked, &mut BTreeSet::new());
            print(&walked, 1, &mut lines);
        }
        let active = |(branch, site): (usize, Site)| {
            self.flag(branch, site)
                .index()
                .map(|index| format!("active[{index}]"))
        };
        lines.extend(self.switch_checks());
        lines.extend(self.internal_nodes());
        for (index, branch) in module.branches.iter().enumerate() {
            if self.holds_potential(index) {
                let claim = format!("layout.internal({:?}, Nature::Current)", branch.name);
                let code = match active((index, Site::Potential)) {
                    Some(flag) => format!("if {flag} {{ {claim} }} else {{ NOWHERE }}"),
                    None => claim,
                };
                lines.push(format!("    let {} = {code};", branch.name));
            }
        }
        let mut fields: Vec<String> = module.nodes[module.ports..].to_vec();
        fields.extend(
            module
                .branches
                .iter()
                .enumerate()
                .filter(|(index, _)| self.holds_potential(*index))
                .map(|(_, branch)| branch.name.clone()),
        );
        // Each entry claimed in the shape itself, where some contribution that writes it
        // is active.
        for (key, users) in entries {
            fields.push(format!(
                "{}: {}",
                self.entry_name(*key),
                self.entry_claim(*key, users)
            ));
        }
        if !self.conditions.is_empty() {
            fields.push("conditions: prepared.conditions".to_owned());
        }
        if flags > 0 {
            fields.push("active".to_owned());
        }
        lines.push(format!(
            "    Ok({} {{ {} }})",
            names.shape,
            fields.join(", ")
        ));
        let prepared = if self.conditions.is_empty() {
            "_"
        } else {
            "prepared"
        };
        let lines: Vec<String> = lines
            .into_iter()
            .map(|line| format!("    {line}"))
            .collect();
        format!(
            "{}    fn layout<R: Real>(\n        _: &{},\n        {prepared}: &{},\n        layout: &mut Layout,\n    ) -> Result<{}, ParameterError> {{\n{}\n    }}\n\n",
            too_long(&lines, "    "),
            names.parameters_type("R"),
            names.prepared_type("R"),
            names.shape,
            lines.join("\n")
        )
    }

    /// The branches that may collapse in some shape, each with its flag where the shape
    /// decides it.
    fn shorts(&self) -> Vec<(usize, Option<usize>)> {
        (0..self.module.branches.len())
            .filter_map(|branch| match self.flag(branch, Site::Collapse) {
                Activity::Never => None,
                Activity::Always => Some((branch, None)),
                Activity::Flag(flag) => Some((branch, Some(flag))),
            })
            .collect()
    }

    /// Whether the internal node `node` collapses into the ground in some shape, through
    /// the shorts that may join it to the ground.
    fn grounded(&self, node: usize) -> bool {
        let shorts: Vec<(usize, usize)> = self
            .shorts()
            .into_iter()
            .map(|(branch, _)| {
                let Branch { plus, minus, .. } = self.module.branches[branch];
                (plus, minus)
            })
            .collect();
        let mut reached = BTreeSet::from([node]);
        let mut grown = true;
        while grown {
            grown = false;
            for &(plus, minus) in &shorts {
                if reached.contains(&plus) != reached.contains(&minus) {
                    reached.extend([plus, minus]);
                    grown = true;
                }
            }
        }
        node >= self.module.ports && reached.contains(&GROUND)
    }

    /// The layout's claims of the internal nodes, each a new unknown unless a short the
    /// shape makes joins it to the ground, a port or an earlier node. The shorts join the
    /// nodes into groups first, each group led by its earliest member, the ground before
    /// every node and the ports before the internal nodes.
    fn internal_nodes(&self) -> Vec<String> {
        let module = self.module;
        let claim = |node: usize| {
            format!(
                "layout.internal({:?}, Nature::{:?})",
                module.labels[node], module.natures[node]
            )
        };
        let shorts = self.shorts();
        if shorts.is_empty() {
            return (module.ports..module.nodes.len())
                .map(|node| format!("    let {} = {};", module.nodes[node], claim(node)))
                .collect();
        }
        // Each node's place among the joined: the ground first, then the nodes in order.
        let place = |node: usize| if node == GROUND { 0 } else { node + 1 };
        let count = module.nodes.len() + 1;
        let places: Vec<String> = (0..count).map(|place| place.to_string()).collect();
        let mut lines = vec![
            "    // The leader of each node's group, the ground first, then the nodes in order."
                .to_owned(),
            format!("    let leader = |joined: &[usize; {count}], mut place: usize| {{"),
            "        while let Some(&next) = joined.get(place)".to_owned(),
            "            && next != place".to_owned(),
            "        {".to_owned(),
            "            place = next;".to_owned(),
            "        }".to_owned(),
            "        place".to_owned(),
            "    };".to_owned(),
            format!("    let mut joined = [{}];", places.join(", ")),
        ];
        for (branch, flag) in shorts {
            let Branch { plus, minus, .. } = module.branches[branch];
            let join = format!(
                "let (one, other) = (leader(&joined, {}), leader(&joined, {})); if let Some(slot) = joined.get_mut(one.max(other)) {{ *slot = one.min(other); }}",
                place(plus),
                place(minus)
            );
            lines.push(match flag {
                Some(flag) => format!("    if active[{flag}] {{ {join} }}"),
                None => format!("    {{ {join} }}"),
            });
        }
        // Each node's local index by its place, the ground's nowhere.
        let mut locals = vec!["NOWHERE".to_owned()];
        locals.extend((0..module.ports).map(|port| port.to_string()));
        locals.resize(count, "NOWHERE".to_owned());
        lines.push(format!("    let mut local = [{}];", locals.join(", ")));
        for node in module.ports..module.nodes.len() {
            let at = place(node);
            lines.push(format!(
                "    local[{at}] = match leader(&joined, {at}) {{ {at} => {}, leader => local.get(leader).copied().unwrap_or(NOWHERE) }};",
                claim(node)
            ));
        }
        let mut names: Vec<String> = vec!["_".to_owned(); module.ports + 1];
        names.extend(module.nodes[module.ports..].iter().cloned());
        lines.push(format!("    let [{}] = local;", names.join(", ")));
        lines
    }

    /// The layout's claim of the entry at `key`, where one of `users` is active and each
    /// branch current and each node it reads exists.
    fn entry_claim(&self, key: (Place, Place), users: &BTreeSet<(usize, Site)>) -> String {
        let active = |(branch, site): (usize, Site)| {
            self.flag(branch, site)
                .index()
                .map(|index| format!("active[{index}]"))
        };
        let claim = format!(
            "layout.entry({}, {})",
            self.place(key.0, ""),
            self.place(key.1, "")
        );
        let users: Option<Vec<String>> = users.iter().map(|&user| active(user)).collect();
        // A branch current the entry reads exists only where its branch holds a
        // potential, and a node only where it does not collapse into the ground.
        let currents: Vec<String> = [key.0, key.1]
            .into_iter()
            .filter_map(|place| match place {
                Place::Current(branch) => active((branch, Site::Potential)),
                Place::Node(node) if self.grounded(node) => {
                    Some(format!("{} != NOWHERE", self.node(node, "")))
                }
                Place::Node(_) => None,
            })
            .collect();
        let mut conditions: Vec<String> = Vec::new();
        if let Some(users) = users {
            let users =
                if users.len() > 1 && currents.iter().any(|current| !users.contains(current)) {
                    format!("({})", users.join(" || "))
                } else {
                    users.join(" || ")
                };
            conditions.push(users);
        }
        for current in currents {
            if !conditions.contains(&current) {
                conditions.push(current);
            }
        }
        match conditions.as_slice() {
            [] => claim,
            // `if x != NOWHERE { a } else { b }` reads as `if x == NOWHERE { b } else { a }`.
            [only] if only.ends_with(" != NOWHERE") => format!(
                "if {} == NOWHERE {{ NOWHERE }} else {{ {claim} }}",
                only.trim_end_matches(" != NOWHERE")
            ),
            _ => format!(
                "if {} {{ {claim} }} else {{ NOWHERE }}",
                conditions.join(" && ")
            ),
        }
    }

    /// The layout's checks that no branch is both a flow and a potential in one shape, a
    /// switch the shape does not decide.
    fn switch_checks(&self) -> Vec<String> {
        let module = self.module;
        let mut lines = Vec::new();
        for (index, branch) in module.branches.iter().enumerate() {
            let flags: Vec<String> = [Site::Flow, Site::Potential]
                .map(|site| self.flag(index, site))
                .iter()
                .filter_map(|activity| activity.index())
                .map(|flag| format!("active[{flag}]"))
                .collect();
            if self.carries_flow(index) && self.holds_potential(index) && !flags.is_empty() {
                lines.push(format!(
                    "    if {} {{\n        return Err(ParameterError(\"the branch {} both carries a flow and holds a potential\".to_owned()));\n    }}",
                    flags.join(" && "),
                    branch.name
                ));
            }
        }
        lines
    }

    /// The layout's walk of `statements`: each flag set once where the shape's conditions
    /// make its branch active.
    fn walk_layout(&self, statements: &[Stmt], out: &mut Vec<Rust>, active: &mut BTreeSet<usize>) {
        for statement in statements {
            match statement {
                Stmt::Contribute {
                    branch,
                    potential,
                    resistive,
                    reactive,
                    ..
                } if resistive.is_some() || reactive.is_some() => {
                    let site =
                        self.site(*branch, *potential, resistive.as_ref(), reactive.as_ref());
                    if let Activity::Flag(index) = self.flag(*branch, site)
                        && active.insert(index)
                    {
                        out.push(Rust::Line(format!("active[{index}] = true;")));
                    }
                }
                Stmt::If(id, _, then, otherwise) => {
                    if let Some(index) = self.shaped.get(id) {
                        let (mut inner_then, mut inner_otherwise) = (Vec::new(), Vec::new());
                        let mut then_active = active.clone();
                        let mut otherwise_active = active.clone();
                        self.walk_layout(then, &mut inner_then, &mut then_active);
                        self.walk_layout(otherwise, &mut inner_otherwise, &mut otherwise_active);
                        if !(inner_then.is_empty() && inner_otherwise.is_empty()) {
                            out.push(Rust::If(
                                (format!("prepared.conditions[{index}]"), ATOM),
                                inner_then,
                                inner_otherwise,
                            ));
                        }
                    } else {
                        self.walk_layout(then, out, active);
                        self.walk_layout(otherwise, out, active);
                    }
                }
                Stmt::While(_, body) => self.walk_layout(body, out, active),
                Stmt::Assign(..) | Stmt::Contribute { .. } | Stmt::Fail(..) | Stmt::Warn(..) => {}
            }
        }
    }

    /// The lines that check each parameter against its ranges.
    fn parameter_lines(&self, emitter: &mut Emitter<'_, '_>) -> Result<Vec<String>, Error> {
        let module = self.module;
        // Each range's bounds, read first, as they may read other parameters.
        let mut checks = Vec::new();
        for (index, parameter) in module.parameters.iter().enumerate() {
            for (exclude, lower, upper) in &parameter.ranges {
                let mut check_body = Vec::new();
                let mut bound = |bound: &Option<(Expr, bool)>| -> Result<Option<Bound>, Error> {
                    let Some((expr, included)) = bound else {
                        return Ok(None);
                    };
                    let constant = constant(expr);
                    let code = match constant {
                        Some(value) => literal(value),
                        None => Emitter::value_of(&emitter.expression(expr, &mut check_body)?).code,
                    };
                    Ok(Some(Bound {
                        code,
                        constant,
                        included: *included,
                    }))
                };
                let (lower, upper) = (bound(lower)?, bound(upper)?);
                let value = emitter.expression(&Expr::Parameter(index), &mut check_body)?;
                print(&check_body, 0, &mut checks);
                checks.push(check(
                    (&value.code, &parameter.external),
                    parameter.kind,
                    *exclude,
                    (lower.as_ref(), upper.as_ref()),
                ));
            }
        }
        Ok(checks)
    }

    /// The inherent `impl` holding `warnings` and `warned`, when the module defines any.
    fn warnings_impl(&self, prepared: &Prepared, names: &Names) -> Result<String, Error> {
        let mut inherent = Vec::new();
        if self.module.analog.iter().any(warns) {
            inherent.push(self.prepare(Mode::Warnings, prepared, names)?);
        }
        if !self.warned.is_empty() {
            let mut messages = Vec::new();
            let mut offset = 0;
            for (message, count) in &self.warned {
                let values: Vec<String> = (offset..offset + count)
                    .map(|index| format!("memory.warned_values[{index}]"))
                    .collect();
                offset += count;
                messages.push(format_message(message, &values));
            }
            inherent.push(format!(
                "    /// The message of each warning that depends on the probes that an evaluation\n    /// of an instance, whose memory is `memory`, has found since the memory was made,\n    /// with the values the last evaluation that found it read.\n    pub fn warned(memory: &{}) -> impl Iterator<Item = String> {{\n        [{}].into_iter().zip(memory.warned).filter_map(|(message, found)| found.then_some(message))\n    }}\n\n",
                names.memory,
                messages.join(", ")
            ));
        }
        if inherent.is_empty() {
            Ok(String::new())
        } else {
            let warnings = inherent.concat();
            Ok(format!(
                "impl {} {{\n{}}}\n",
                names.kernel,
                warnings.trim_end_matches('\n').to_owned() + "\n"
            ))
        }
    }

    /// The fields of the prepared numbers, as `prepare` writes them at its end, in
    /// [`Mode::Prepare`]: each of `fields`, and the conditions that shape the kernel.
    fn prepared_values(
        &self,
        emitter: &mut Emitter<'_, '_>,
        (mode, fields): (Mode, &[(String, Kind, Expr)]),
        body: &mut Vec<Rust>,
    ) -> Result<Vec<String>, Error> {
        let mut values = Vec::new();
        for (name, kind, expr) in fields {
            let operand = emitter.expression(expr, body)?;
            let operand = emitter.convert(operand, kind, body)?;
            values.push(if operand.code == *name {
                name.clone()
            } else {
                format!("{name}: {}", operand.code)
            });
        }
        if mode == Mode::Prepare && !self.conditions.is_empty() {
            let mut conditions = Vec::with_capacity(self.conditions.len());
            for condition in &self.conditions {
                let operand = emitter.expression(condition, body)?;
                conditions.push(emitter.truth(operand).0);
            }
            values.push(format!("conditions: [{}]", conditions.join(", ")));
        }
        Ok(values)
    }

    /// The function `prepare`, in [`Mode::Prepare`], or in [`Mode::Warnings`] the
    /// kernel's function `warnings`, which runs preparation again for what its
    /// `$warning`s and their like state.
    fn prepare(&self, mode: Mode, prepared: &Prepared, names: &Names) -> Result<String, Error> {
        let module = self.module;
        let mut entries = BTreeMap::new();
        let mut scratch = Prepared::default();
        let mut emitter = Emitter::new(self, mode, &mut scratch, &mut entries);
        let mut roots = BTreeSet::new();
        let fields: &[(String, Kind, Expr)] = if mode == Mode::Prepare {
            &prepared.fields
        } else {
            &[]
        };
        for (_, _, expr) in fields {
            reads(expr, module, &mut roots);
        }
        if mode == Mode::Prepare {
            for condition in &self.conditions {
                reads(condition, module, &mut roots);
            }
        }
        emitter.analog = prune(&module.analog, module, mode, &mut roots.clone());
        emitter.live = live(module, &emitter.analog, mode, &roots);
        let mut body = Vec::new();
        let analog = emitter.analog.clone();
        emitter.statements(&analog, &mut body)?;
        let values = self.prepared_values(&mut emitter, (mode, fields), &mut body)?;
        let checks = self.parameter_lines(&mut emitter)?;
        let (constant_fields, constant_body) = emit_local_parameters(&mut emitter)?;
        let locals = static_locals(module, &emitter.live);
        let declared_constants = fields_struct("Constants", "c", &constant_fields);
        let (constants_struct, constants_init) =
            declared_constants.split_at(declared_constants.len().saturating_sub(1));
        let declared = locals_struct(&locals);
        let (structure, initial) = declared.split_at(declared.len().saturating_sub(1));
        let mut lines = constants_struct.to_vec();
        lines.extend(structure.iter().cloned());
        lines.extend(constants_init.iter().cloned());
        print(&constant_body, 0, &mut lines);
        lines.extend(checks);
        lines.extend(initial.iter().cloned());
        if mode == Mode::Warnings {
            lines.push("let mut found = Vec::new();".to_owned());
        }
        print(&body, 0, &mut lines);
        lines.push(if mode == Mode::Prepare {
            if values.is_empty() {
                format!("Ok({} {{}})", names.prepared)
            } else {
                format!("Ok({} {{ {} }})", names.prepared, values.join(", "))
            }
        } else {
            "Ok(found)".to_owned()
        });
        Ok(format_prepare(
            mode,
            names,
            &lines,
            !emitter.parameters_read.is_empty(),
            emitter.uses_temperature,
        ))
    }
}

/// Renders one struct field declaration and its `Default` initializer entry for `parameter`.
fn parameter_field(parameter: &Parameter) -> (String, String) {
    let mut documentation = parameter
        .description
        .as_deref()
        .map_or_else(|| format!("The parameter `{}`", parameter.external), prose);
    if let Some(units) = parameter.units.as_deref().filter(|units| !units.is_empty()) {
        documentation = format!("{documentation}, in {}", prose(units));
    }
    let default = match constant(&parameter.default) {
        Some(value) => format!("default {value}"),
        None => "by default, as its source computes it from the parameters before it".to_owned(),
    };
    documentation = format!("{documentation}; {default}.");
    let number = if parameter.kind == Type::Integer {
        "i32"
    } else {
        "R"
    };
    let kind = if parameter.optional {
        format!("Option<{number}>")
    } else {
        number.to_owned()
    };
    let aliases = parameter
        .aliases
        .iter()
        .map(|alias| format!("    #[serde(alias = {alias:?})]\n"))
        .collect::<Vec<_>>()
        .concat();
    let renamed = if parameter.name == parameter.external {
        String::new()
    } else {
        format!("    #[serde(rename = {:?})]\n", parameter.external)
    };
    let field = format!(
        "    /// {documentation}\n{renamed}{aliases}    pub {}: {kind},\n",
        parameter.name
    );
    let value = match (
        parameter.optional,
        parameter.kind,
        constant(&parameter.default),
    ) {
        (true, _, _) | (false, _, None) => "None".to_owned(),
        (false, Type::Integer, Some(value)) => format!("{value}"),
        (false, Type::Real, Some(value)) => format!("R::constant({})", literal(value)),
    };
    (field, format!("            {}: {value},\n", parameter.name))
}

/// The static variables kept for preparation, each with its emitted kind.
fn static_locals(module: &Module, live: &[bool]) -> Vec<(String, Kind)> {
    module
        .variables
        .iter()
        .enumerate()
        .filter(|(index, _)| live[*index] && !module.dynamic[*index])
        .map(|(_, (name, kind))| {
            (
                name.clone(),
                if *kind == Type::Integer {
                    Kind::Integer
                } else {
                    Kind::Real
                },
            )
        })
        .collect()
}

/// Formats `prepare` or `warnings` with its signature, doc comment, and indented `lines`.
fn format_prepare(
    mode: Mode,
    names: &Names,
    lines: &[String],
    uses_parameters: bool,
    uses_temperature: bool,
) -> String {
    let temperature = if uses_temperature { "temperature" } else { "_" };
    let parameters = if uses_parameters { "parameters" } else { "_" };
    let indented: Vec<String> = lines
        .iter()
        .map(|line| {
            if line.is_empty() {
                String::new()
            } else {
                format!("        {line}")
            }
        })
        .collect();
    let (documentation, name, result) = if mode == Mode::Prepare {
        (String::new(), "fn prepare", names.prepared_type("R"))
    } else {
        (
            "    /// The facts the `$warning`s of the Verilog-A source, and their like, state of an\n    /// instance with `parameters` at `temperature`, in kelvins, in the order it states\n    /// them.\n    ///\n    /// # Errors\n    ///\n    /// Returns [`ParameterError`] when a parameter is out of range.\n".to_owned(),
            "pub fn warnings",
            "Vec<String>".to_owned(),
        )
    };
    format!(
        "{documentation}{}    {name}<R: Real>(\n        {parameters}: &{},\n        {temperature}: R,\n    ) -> Result<{result}, ParameterError> {{\n{}\n    }}\n\n",
        too_long(&indented, "    "),
        names.parameters_type("R"),
        indented.join("\n")
    )
}

/// Field descriptors and initialization statements emitted for live `localparam` constants.
type EmittedLocals = (Vec<(String, Kind)>, Vec<Rust>);

/// Emits the field descriptors and initialization statements for every non-constant
/// `localparam` read during preparation, in ascending declaration order.
///
/// Every non-constant `localparam` only references strictly earlier `localparam` indices,
/// so a single reverse pass captures transitive dependencies (including through optional
/// parameter defaults) with zero unread fields.
fn emit_local_parameters(emitter: &mut Emitter<'_, '_>) -> Result<EmittedLocals, Error> {
    let count = emitter.facts.module.local_parameters.len();
    let mut entries: Vec<(String, Kind, Vec<Rust>)> = Vec::new();
    for index in (0..count).rev() {
        if !emitter.locals_read.contains(&index) {
            continue;
        }
        let (name, kind, value) = {
            let local = &emitter.facts.module.local_parameters[index];
            (
                local.name.clone(),
                if local.kind == Type::Integer {
                    Kind::Integer
                } else {
                    Kind::Real
                },
                local.value.clone(),
            )
        };
        let mut statements = Vec::new();
        let operand = emitter.expression(&value, &mut statements)?;
        let operand = emitter.convert(operand, &kind, &mut statements)?;
        statements.push(Rust::Line(format!("c.{name} = {};", operand.code)));
        entries.push((name, kind, statements));
    }
    entries.reverse();
    let mut fields = Vec::with_capacity(entries.len());
    let mut body = Vec::new();
    for (name, kind, statements) in entries {
        fields.push((name, kind));
        body.extend(statements);
    }
    Ok((fields, body))
}

/// Checks that no contribution reads a `ddx`, whose own derivative would be a second
/// one, which the dual numbers do not carry: the variables a `ddx` reaches are found by
/// its assignments, until none is added.
fn first_order(module: &Module) -> Result<(), Error> {
    let mut reached = vec![false; module.variables.len()];
    let reads = |expr: &Expr, reached: &[bool]| {
        let mut found = false;
        expr.walk(&mut |inner| {
            found |= matches!(inner, Expr::Ddx(..))
                || matches!(inner, Expr::Variable(variable) if reached[*variable]);
        });
        found
    };
    loop {
        let mut changed = false;
        visit_assignments(&module.analog, &mut |variable, value| {
            if !reached[variable] && reads(value, &reached) {
                reached[variable] = true;
                changed = true;
            }
        });
        if !changed {
            break;
        }
    }
    let mut refused = false;
    visit_contributions(&module.analog, &mut |statement| {
        statement.walk(&mut |expr| refused |= reads(expr, &reached));
    });
    if refused {
        return Err(Error::at(
            0,
            "a contribution that reads a ddx needs second derivatives, which are not compiled yet"
                .into(),
        ));
    }
    Ok(())
}

/// Calls `visit` on each assignment of `statements`, its variable and its value.
fn visit_assignments(statements: &[Stmt], visit: &mut impl FnMut(usize, &Expr)) {
    for statement in statements {
        match statement {
            Stmt::Assign(variable, value) => visit(*variable, value),
            Stmt::If(_, _, then, otherwise) => {
                visit_assignments(then, visit);
                visit_assignments(otherwise, visit);
            }
            Stmt::While(_, body) => visit_assignments(body, visit),
            Stmt::Contribute { .. } | Stmt::Fail(..) | Stmt::Warn(..) => {}
        }
    }
}

/// Checks that each `$error` and `$warning` of `statements`, under a condition that is
/// dynamic when `dynamic` is, depends on no probe, so that preparation states it.
fn static_tasks(statements: &[Stmt], module: &Module, dynamic: bool) -> Result<(), Error> {
    for statement in statements {
        match statement {
            Stmt::Fail(message, arguments) | Stmt::Warn(message, arguments, false) => {
                if dynamic || arguments.iter().any(|argument| module.is_dynamic(argument)) {
                    return Err(Error::at(
                        0,
                        format!(
                            "the task {message:?} depends on the probes, which is not compiled yet"
                        ),
                    ));
                }
            }
            Stmt::If(_, condition, then, otherwise) => {
                let dynamic = dynamic || module.is_dynamic(condition);
                static_tasks(then, module, dynamic)?;
                static_tasks(otherwise, module, dynamic)?;
            }
            Stmt::While(condition, body) => {
                static_tasks(body, module, dynamic || module.is_dynamic(condition))?;
            }
            Stmt::Warn(_, _, true) | Stmt::Assign(..) | Stmt::Contribute { .. } => {}
        }
    }
    Ok(())
}

/// Adds the variables `expr` reads, and those the arguments of its limited probes read,
/// to `found`.
fn reads(expr: &Expr, module: &Module, found: &mut BTreeSet<usize>) {
    expr.walk(&mut |inner| match inner {
        Expr::Variable(variable) => {
            found.insert(*variable);
        }
        Expr::Probe(Probe::Limited(limit)) => {
            for argument in &module.limits[*limit].arguments {
                reads(argument, module, found);
            }
        }
        _ => {}
    });
}

/// Which variables `mode` keeps: those `roots` holds, and those a kept statement reads,
/// until none is added; in a dynamic mode only dynamic variables, in preparation only
/// static ones.
fn live(module: &Module, statements: &[Stmt], mode: Mode, roots: &BTreeSet<usize>) -> Vec<bool> {
    let wanted = |variable: usize| mode.prepares() != module.dynamic[variable];
    let mut live: Vec<bool> = (0..module.variables.len())
        .map(|variable| roots.contains(&variable) && wanted(variable))
        .collect();
    loop {
        let mut found = BTreeSet::new();
        mark(statements, module, mode, &live, &mut found);
        let mut changed = false;
        for variable in found {
            if wanted(variable) && !live[variable] {
                live[variable] = true;
                changed = true;
            }
        }
        if !changed {
            return live;
        }
    }
}

/// `statements` as `mode` runs them, each assignment that nothing after it reads left
/// out, `live` the variables read after them, which it leaves those read before: a
/// backward pass over the structured code, a loop's body to its fixed point. An
/// assignment of a variable to itself does nothing, and one that limits a voltage is
/// kept for the memory it writes.
fn prune(
    statements: &[Stmt],
    module: &Module,
    mode: Mode,
    live: &mut BTreeSet<usize>,
) -> Vec<Stmt> {
    let wanted = |variable: usize| mode.prepares() != module.dynamic[variable];
    let limits = |expr: &Expr| {
        let mut found = false;
        expr.walk(&mut |inner| found |= matches!(inner, Expr::Probe(Probe::Limited(_))));
        found
    };
    let mut kept_backward = Vec::new();
    for statement in statements.iter().rev() {
        match statement {
            Stmt::Assign(variable, value) => {
                let itself = matches!(value, Expr::Variable(read) if read == variable);
                if !wanted(*variable) || itself || (!live.contains(variable) && !limits(value)) {
                    continue;
                }
                live.remove(variable);
                reads(value, module, live);
                kept_backward.push(statement.clone());
            }
            Stmt::If(id, condition, then, otherwise) => {
                let (mut first, mut second) = (live.clone(), live.clone());
                let then = prune(then, module, mode, &mut first);
                let otherwise = prune(otherwise, module, mode, &mut second);
                if then.is_empty() && otherwise.is_empty() && !limits(condition) {
                    continue;
                }
                first.extend(second);
                reads(condition, module, &mut first);
                *live = first;
                kept_backward.push(Stmt::If(*id, condition.clone(), then, otherwise));
            }
            Stmt::While(condition, body) => {
                let mut entry = live.clone();
                reads(condition, module, &mut entry);
                loop {
                    let mut inner = entry.clone();
                    prune(body, module, mode, &mut inner);
                    let mut next = live.clone();
                    next.extend(inner);
                    reads(condition, module, &mut next);
                    if next == entry {
                        break;
                    }
                    entry = next;
                }
                let mut inner = entry.clone();
                let body = prune(body, module, mode, &mut inner);
                if body.is_empty() {
                    continue;
                }
                *live = entry;
                kept_backward.push(Stmt::While(condition.clone(), body));
            }
            Stmt::Contribute { .. } | Stmt::Fail(..) | Stmt::Warn(..) => {
                if kept(statement, module, mode, &[]) {
                    let mut found = BTreeSet::new();
                    mark(
                        std::slice::from_ref(statement),
                        module,
                        mode,
                        &[],
                        &mut found,
                    );
                    live.extend(found);
                    kept_backward.push(statement.clone());
                }
            }
        }
    }
    kept_backward.reverse();
    kept_backward
}

/// Adds what the statements of `statements` that `mode` keeps read to `found`.
fn mark(
    statements: &[Stmt],
    module: &Module,
    mode: Mode,
    live: &[bool],
    found: &mut BTreeSet<usize>,
) {
    for statement in statements {
        if !kept(statement, module, mode, live) {
            continue;
        }
        match statement {
            Stmt::Assign(_, value) => reads(value, module, found),
            Stmt::Contribute {
                resistive,
                reactive,
                noise,
                ..
            } => {
                if mode == Mode::Noise {
                    for noise in noise {
                        reads(&noise.power, module, found);
                        if let Some(exponent) = &noise.exponent {
                            reads(exponent, module, found);
                        }
                    }
                } else {
                    for expr in resistive.iter().chain(reactive) {
                        reads(expr, module, found);
                    }
                }
            }
            Stmt::If(_, condition, then, otherwise) => {
                reads(condition, module, found);
                mark(then, module, mode, live, found);
                mark(otherwise, module, mode, live, found);
            }
            Stmt::While(condition, body) => {
                reads(condition, module, found);
                mark(body, module, mode, live, found);
            }
            Stmt::Fail(_, arguments) | Stmt::Warn(_, arguments, _) => {
                for expr in arguments {
                    reads(expr, module, found);
                }
            }
        }
    }
}

/// Whether `mode` keeps `statement`.
fn kept(statement: &Stmt, module: &Module, mode: Mode, live: &[bool]) -> bool {
    match statement {
        Stmt::Assign(variable, _) => match mode {
            Mode::Function => true,
            Mode::Prepare | Mode::Warnings => !module.dynamic[*variable] && live[*variable],
            Mode::Evaluate | Mode::Noise => module.dynamic[*variable] && live[*variable],
        },
        Stmt::Contribute {
            resistive,
            reactive,
            noise,
            ..
        } => match mode {
            Mode::Evaluate => resistive.is_some() || reactive.is_some(),
            Mode::Noise => !noise.is_empty(),
            Mode::Prepare | Mode::Warnings | Mode::Function => false,
        },
        Stmt::If(_, _, then, otherwise) => then
            .iter()
            .chain(otherwise)
            .any(|inner| kept(inner, module, mode, live)),
        Stmt::While(_, body) => body.iter().any(|inner| kept(inner, module, mode, live)),
        Stmt::Fail(..) => mode.prepares(),
        Stmt::Warn(_, _, dynamic) => {
            if *dynamic {
                mode == Mode::Evaluate
            } else {
                mode == Mode::Warnings
            }
        }
    }
}

/// The Rust type of `kind`.
fn rust_type(kind: &Kind) -> String {
    match kind {
        Kind::Bool => "bool".to_owned(),
        Kind::Integer => "i32".to_owned(),
        Kind::Float(_) => "f64".to_owned(),
        Kind::Real => "R".to_owned(),
        Kind::Dual(probes) => format!("Dual<R, {}>", probes.len()),
    }
}

/// The zero of `kind`.
fn zero(kind: &Kind) -> String {
    match kind {
        Kind::Bool => "false".to_owned(),
        Kind::Integer => "0".to_owned(),
        Kind::Float(_) => "0.0".to_owned(),
        Kind::Real => "R::constant(0.0)".to_owned(),
        Kind::Dual(_) => "Dual::number(0.0)".to_owned(),
    }
}

/// The lines that declare a function's struct named `struct_name` bound to `binding`,
/// with `fields` each initialized to zero.
fn fields_struct(struct_name: &str, binding: &str, fields: &[(String, Kind)]) -> Vec<String> {
    if fields.is_empty() {
        return Vec::new();
    }
    let generic = fields
        .iter()
        .any(|(_, kind)| matches!(kind, Kind::Real | Kind::Dual(_)));
    let parameters = if generic { "<R: Real>" } else { "" };
    let mut lines = vec![format!("struct {struct_name}{parameters} {{")];
    for (name, kind) in fields {
        lines.push(format!("    {name}: {},", rust_type(kind)));
    }
    lines.push("}".to_owned());
    let values: Vec<String> = fields
        .iter()
        .map(|(name, kind)| format!("{name}: {}", zero(kind)))
        .collect();
    lines.push(format!(
        "let mut {binding} = {struct_name} {{ {} }};",
        values.join(", ")
    ));
    lines
}

/// The lines that declare the function's variables, `locals`, each at zero, as
/// Verilog-A's variables start: a struct of them, so that one a branch leaves
/// unassigned reads zero.
fn locals_struct(locals: &[(String, Kind)]) -> Vec<String> {
    fields_struct("Locals", "v", locals)
}

/// A bound of a parameter's range, as the check reads it: its code, its value when it is
/// a constant, and whether it is included.
struct Bound {
    code: String,
    constant: Option<f64>,
    included: bool,
}

/// The check that the parameter `name`, named `external` in its source, of `kind`, lies
/// in the range of `lower` and `upper`, or, when `exclude`, does not.
fn check(
    (name, external): (&str, &str),
    kind: Type,
    exclude: bool,
    (lower, upper): (Option<&Bound>, Option<&Bound>),
) -> String {
    let value = if kind == Type::Integer {
        format!("f64::from({name})")
    } else {
        format!("{name}.value()")
    };
    let shown = if kind == Type::Integer {
        name.to_owned()
    } else {
        format!("{name}.value()")
    };
    if !exclude
        && upper.is_none()
        && kind == Type::Real
        && lower.is_some_and(|low| low.constant == Some(0.0))
    {
        let check = if lower.is_some_and(|low| low.included) {
            "non_negative"
        } else {
            "positive"
        };
        return format!("ParameterError::{check}({external:?}, {name})?;");
    }
    let contains = match (lower, upper) {
        (Some(low), Some(high)) if low.included && !high.included => {
            format!("({}..{}).contains(&{value})", low.code, high.code)
        }
        (Some(low), Some(high)) if low.included => {
            format!("({}..={}).contains(&{value})", low.code, high.code)
        }
        (Some(low), None) if low.included => format!("({}..).contains(&{value})", low.code),
        (None, Some(high)) if !high.included => format!("(..{}).contains(&{value})", high.code),
        (None, Some(high)) => format!("(..={}).contains(&{value})", high.code),
        (Some(low), high) => {
            let above = format!(
                "matches!({value}.partial_cmp(&{}), Some(Ordering::Greater))",
                low.code
            );
            match high {
                Some(high) if high.included => format!("{above} && {value} <= {}", high.code),
                Some(high) => format!("{above} && {value} < {}", high.code),
                None => above,
            }
        }
        (None, None) => "true".to_owned(),
    };
    // Each bound in words, a constant written in place, another read when it fails.
    let mut arguments = Vec::new();
    let mut words = |bound: &Bound, low: bool| {
        let value = bound.constant.map_or_else(
            || {
                arguments.push(bound.code.clone());
                "{:?}".to_owned()
            },
            |value| format!("{value}"),
        );
        match (low, bound.included) {
            (true, true) => format!("at least {value}"),
            (true, false) => format!("above {value}"),
            (false, true) => format!("at most {value}"),
            (false, false) => format!("below {value}"),
        }
    };
    let mut described = Vec::new();
    if let Some(low) = lower {
        described.push(words(low, true));
    }
    if let Some(high) = upper {
        described.push(words(high, false));
    }
    arguments.push(shown);
    let (condition, verb) = if exclude {
        (contains, "must not be")
    } else if contains.contains("&&") {
        (format!("!({contains})"), "must be")
    } else {
        (format!("!{contains}"), "must be")
    };
    format!(
        "if {condition} {{\n    return Err(ParameterError(format!(\"{external} {verb} {}, not {{:?}}\", {})));\n}}",
        described.join(" and "),
        arguments.join(", ")
    )
}

/// The code that makes the message of a task, `message` with each of its `%g`s and
/// their like the value that code of `values` at its place makes, and each value past
/// them after it, as Verilog's `$display` prints an argument no format takes.
fn format_message(message: &str, values: &[String]) -> String {
    let mut format = message.replace('{', "{{").replace('}', "}}");
    let mut taken = 0;
    for placeholder in ["%g", "%e", "%f", "%d", "%r"] {
        taken += format.matches(placeholder).count();
        format = format.replace(placeholder, "{:?}");
    }
    for _ in taken..values.len() {
        format.push_str("{:?}");
    }
    if values.is_empty() {
        format!("{format:?}.to_owned()")
    } else {
        format!(
            "format!(\"{}\", {})",
            format.replace('"', "\\\""),
            values.join(", ")
        )
    }
}

/// `text`, the source's prose, as documentation writes it: each word that reads as code,
/// with an underscore or a capital after a small letter, in backticks, as clippy's
/// `doc_markdown` asks.
fn prose(text: &str) -> String {
    text.split(' ')
        .map(|word| {
            let core = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
            let camel = core
                .chars()
                .zip(core.chars().skip(1))
                .any(|(before, after)| before.is_lowercase() && after.is_uppercase());
            if core.is_empty() || !(core.contains('_') || camel) {
                word.to_owned()
            } else {
                word.replacen(core, &format!("`{core}`"), 1)
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The attribute that excuses a function of `lines` for its length, when clippy's bound
/// is exceeded, indented by `indent`.
fn too_long(lines: &[String], indent: &str) -> String {
    let counted = lines
        .iter()
        .flat_map(|line| line.lines())
        .filter(|line| !line.trim().is_empty())
        .count();
    if counted > LONGEST {
        format!(
            "{indent}#[expect(\n{indent}    clippy::too_many_lines,\n{indent}    reason = \"a kernel's function is its Verilog-A source's analog block, which is one block\"\n{indent})]\n"
        )
    } else {
        String::new()
    }
}

/// Simplifies conditionals in `statements`: drops empty conditionals and inverts empty
/// `if cond {} else { body }` to `if !cond { body }`.
fn simplify(statements: &[Rust]) -> Vec<Rust> {
    let mut out = Vec::with_capacity(statements.len());
    for statement in statements {
        match statement {
            Rust::Line(_) => out.push(statement.clone()),
            Rust::While(condition, body) => {
                out.push(Rust::While(condition.clone(), simplify(body)));
            }
            Rust::If((condition, precedence), then, otherwise) => {
                let then = simplify(then);
                let otherwise = simplify(otherwise);
                if then.is_empty() && otherwise.is_empty() {
                    continue;
                }
                if then.is_empty() {
                    out.push(Rust::If(
                        negate_condition(condition, *precedence),
                        otherwise,
                        Vec::new(),
                    ));
                } else {
                    out.push(Rust::If((condition.clone(), *precedence), then, otherwise));
                }
            }
        }
    }
    out
}

/// Returns the logical negation of `condition` at `precedence` and its resulting precedence.
fn negate_condition(condition: &str, precedence: u8) -> (String, u8) {
    if let Some(positive) = positive(condition) {
        return positive;
    }
    if let Some((left, right)) = condition.split_once(" == ")
        && single(left)
        && single(right)
    {
        return (format!("{left} != {right}"), COMPARE);
    }
    if precedence >= UNARY {
        (format!("!{condition}"), UNARY)
    } else {
        (format!("!({condition})"), UNARY)
    }
}

/// Prints `statements` at `depth` levels of indentation to `out`, line by line, as
/// clippy's style asks: nested conditions without branches of their own joined, an
/// `else` that holds only a conditional written `else if`.
fn print(statements: &[Rust], depth: usize, out: &mut Vec<String>) {
    let simplified = simplify(statements);
    print_simplified(&simplified, depth, out);
}

/// Prints pre-simplified `statements` at `depth` levels of indentation to `out`.
fn print_simplified(statements: &[Rust], depth: usize, out: &mut Vec<String>) {
    let indent = "    ".repeat(depth);
    for statement in statements {
        match statement {
            Rust::Line(line) => {
                for part in line.lines() {
                    out.push(format!("{indent}{part}"));
                }
            }
            Rust::While(condition, body) => {
                out.push(format!("{indent}while {condition} {{"));
                print_simplified(body, depth + 1, out);
                out.push(format!("{indent}}}"));
            }
            Rust::If(..) => print_if(statement, depth, out, false),
        }
    }
}

/// Whether `code` is one operand: a path, or one whose parentheses or brackets, opened
/// within it, close only at its end.
fn single(code: &str) -> bool {
    let mut depth = 0_i32;
    for (at, c) in code.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => {
                depth -= 1;
                if depth == 0
                    && at + 1 < code.len()
                    && !code[at + 1..].starts_with('[')
                    && !code[at + 1..].starts_with('.')
                {
                    return false;
                }
            }
            ' ' if depth == 0 => return false,
            _ => {}
        }
    }
    depth == 0
}

/// The positive form of `condition` and its precedence, when it is a negation: `!c` of
/// one operand reads `c`, and `a != b` of two, at its top, `a == b`; `None` for another.
fn positive(condition: &str) -> Option<(String, u8)> {
    if let Some(inner) = condition.strip_prefix('!')
        && single(inner)
    {
        let stripped = inner
            .strip_prefix('(')
            .and_then(|rest| rest.strip_suffix(')'))
            .filter(|_| inner.ends_with(')') && inner.starts_with('('));
        return Some(match stripped {
            Some(stripped) => (stripped.to_owned(), OR),
            None => (inner.to_owned(), ATOM),
        });
    }
    let (left, right) = condition.split_once(" != ")?;
    (single(left) && single(right)).then(|| (format!("{left} == {right}"), COMPARE))
}

fn print_if(statement: &Rust, depth: usize, out: &mut Vec<String>, chained: bool) {
    let indent = "    ".repeat(depth);
    let Rust::If((condition, precedence), then, otherwise) = statement else {
        return;
    };
    // `if !c { a } else { b }` reads as `if c { b } else { a }`.
    if !otherwise.is_empty()
        && let Some(positive) = positive(condition)
    {
        let swapped = Rust::If(positive, otherwise.clone(), then.clone());
        print_if(&swapped, depth, out, chained);
        return;
    }
    let (mut condition, mut then) = ((condition.clone(), *precedence), then.clone());
    // `if a { if b { .. } }` is `if a && b { .. }`.
    while otherwise.is_empty()
        && let [Rust::If((inner, inner_precedence), inner_then, inner_otherwise)] = then.as_slice()
        && inner_otherwise.is_empty()
    {
        let left = if condition.1 > OR {
            condition.0.clone()
        } else {
            format!("({})", condition.0)
        };
        let right = if *inner_precedence > OR {
            inner.clone()
        } else {
            format!("({inner})")
        };
        condition = (format!("{left} && {right}"), AND);
        then = inner_then.clone();
    }
    if chained {
        if let Some(last) = out.last_mut() {
            *last = format!("{last} if {} {{", condition.0);
        }
    } else {
        out.push(format!("{indent}if {} {{", condition.0));
    }
    print_simplified(&then, depth + 1, out);
    match otherwise.as_slice() {
        [] => out.push(format!("{indent}}}")),
        [inner @ Rust::If(..)] => {
            out.push(format!("{indent}}} else"));
            print_if(inner, depth, out, true);
        }
        _ => {
            out.push(format!("{indent}}} else {{"));
            print_simplified(otherwise, depth + 1, out);
            out.push(format!("{indent}}}"));
        }
    }
}

impl<'f, 'm> Emitter<'f, 'm> {
    fn new(
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

/// Returns the compile-time integer value of `expr` when it is built from integer literals
/// and wrapping `+`, `-`, `*`, and unary signs (never integer division).
fn integer_constant(expr: &Expr) -> Option<i32> {
    match expr {
        Expr::Integer(value) => Some(*value),
        Expr::Unary("-", operand) => integer_constant(operand).map(i32::wrapping_neg),
        Expr::Unary("+", operand) => integer_constant(operand),
        Expr::Binary("+", left, right) => {
            Some(integer_constant(left)?.wrapping_add(integer_constant(right)?))
        }
        Expr::Binary("-", left, right) => {
            Some(integer_constant(left)?.wrapping_sub(integer_constant(right)?))
        }
        Expr::Binary("*", left, right) => {
            Some(integer_constant(left)?.wrapping_mul(integer_constant(right)?))
        }
        _ => None,
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

/// The kinds of two operands joined: the kind that holds both.
fn join(left: &Kind, right: &Kind) -> Kind {
    match (left, right) {
        (Kind::Dual(left), Kind::Dual(right)) => Kind::Dual(union(left, right)),
        (Kind::Dual(probes), _) | (_, Kind::Dual(probes)) => Kind::Dual(probes.clone()),
        (Kind::Real, _) | (_, Kind::Real) => Kind::Real,
        (Kind::Float(_), _) | (_, Kind::Float(_)) => Kind::Float(None),
        (Kind::Integer, _) | (_, Kind::Integer) => Kind::Integer,
        (Kind::Bool, Kind::Bool) => Kind::Bool,
    }
}

impl Emitter<'_, '_> {
    fn module(&self) -> &Module {
        self.facts.module
    }

    /// A new temporary's name, numbered in four digits, so that no two differ by a digit
    /// more, which clippy's `similar_names` reads as a slip.
    fn temporary(&mut self) -> String {
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

    /// The kind of the variable at `variable`, in this mode.
    fn variable_kind(&self, variable: usize) -> Kind {
        if let Some(function) = self.function {
            let (_, kind) = &self.module().functions[function].variables[variable];
            return if *kind == Type::Integer {
                Kind::Integer
            } else {
                Kind::Float(None)
            };
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

    /// The kind of `expr`, a static expression, a constant folded.
    fn static_kind(&self, expr: &Expr) -> Kind {
        let module = self.module();
        match expr {
            Expr::Real(value) => Kind::Float(Some(*value)),
            Expr::Integer(_) => Kind::Integer,
            Expr::Parameter(parameter) => {
                if module.parameters[*parameter].kind == Type::Integer {
                    Kind::Integer
                } else {
                    Kind::Real
                }
            }
            Expr::LocalParameter(local) => {
                if module.local_parameters[*local].kind == Type::Integer {
                    Kind::Integer
                } else {
                    Kind::Real
                }
            }
            Expr::Variable(variable) => {
                if module.variables[*variable].1 == Type::Integer {
                    Kind::Integer
                } else {
                    Kind::Real
                }
            }
            Expr::Temperature | Expr::Ddx(..) => Kind::Real,
            Expr::Gmin | Expr::SmallResistance | Expr::Probe(_) => Kind::Float(None),
            Expr::Given(_)
            | Expr::Shape(_)
            | Expr::Unary("!", _)
            | Expr::Binary("<" | "<=" | ">" | ">=" | "==" | "!=" | "&&" | "||", ..) => Kind::Bool,
            Expr::Unary(operator, operand) => match self.static_kind(operand) {
                Kind::Float(Some(value)) if *operator == "-" => Kind::Float(Some(-value)),
                Kind::Float(Some(value)) if *operator == "+" => Kind::Float(Some(value)),
                kind => kind,
            },
            Expr::Binary(operator @ ("+" | "-" | "*" | "/"), left, right) => {
                let fold_finite = |a: f64, b: f64| {
                    let value = fold(operator, a, b);
                    if value.is_finite() {
                        Kind::Float(Some(value))
                    } else {
                        Kind::Float(None)
                    }
                };
                match (self.static_kind(left), self.static_kind(right)) {
                    (Kind::Float(Some(left)), Kind::Float(Some(right))) => fold_finite(left, right),
                    (Kind::Float(Some(value)), Kind::Integer) => integer_constant(right)
                        .map_or(Kind::Float(None), |right| {
                            fold_finite(value, f64::from(right))
                        }),
                    (Kind::Integer, Kind::Float(Some(value))) => integer_constant(left)
                        .map_or(Kind::Float(None), |left| {
                            fold_finite(f64::from(left), value)
                        }),
                    (left, right) => join(&left, &right),
                }
            }
            Expr::Binary(_, left, right) => join(&self.static_kind(left), &self.static_kind(right)),
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

    /// A static expression read in a dynamic mode: a constant written in place, a
    /// constant of jolty's, or a prepared number.
    fn static_operand(&mut self, expr: &Expr) -> Operand {
        let kind = self.static_kind(expr);
        if let Kind::Float(Some(value)) = kind {
            return float(value);
        }
        if kind == Kind::Integer
            && let Some(value) = integer_constant(expr)
        {
            return Operand::new(
                value.to_string(),
                if value < 0 { UNARY } else { ATOM },
                Kind::Integer,
            );
        }
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
    fn value_of(operand: &Operand) -> Operand {
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
    fn truth(&self, operand: Operand) -> (String, u8) {
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
    fn convert(
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
            (Kind::Dual(from), Kind::Dual(to)) if from.iter().all(|probe| to.contains(probe)) => {
                let operand = self.bind(operand.clone(), out);
                let names: Vec<String> = from.iter().map(|_| self.temporary()).collect();
                out.push(Rust::Line(format!(
                    "let [{}] = {}.gradient;",
                    names.join(", "),
                    operand.code
                )));
                let gradient: Vec<String> = to
                    .iter()
                    .map(|probe| {
                        from.iter().position(|found| found == probe).map_or_else(
                            || "R::constant(0.0)".to_owned(),
                            |index| names[index].clone(),
                        )
                    })
                    .collect();
                let widened = self.temporary();
                out.push(Rust::Line(format!(
                    "let {widened} = Dual {{ value: {}.value, gradient: [{}] }};",
                    operand.code,
                    gradient.join(", ")
                )));
                Operand::new(widened, ATOM, kind.clone())
            }
            _ => return Err(unsupported()),
        })
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one arm per construct of the language, which read best side by side"
    )]
    fn expression(&mut self, expr: &Expr, out: &mut Vec<Rust>) -> Result<Operand, Error> {
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
                let kind = if parameter.kind == Type::Integer {
                    Kind::Integer
                } else {
                    Kind::Real
                };
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
                let kind = if local.kind == Type::Integer {
                    Kind::Integer
                } else {
                    Kind::Real
                };
                Operand::new(format!("c.{}", local.name), ATOM, kind)
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
                let right = self.expression(right, out)?;
                let least = if *operator == "&&" { AND } else { OR };
                let wrap = |(code, precedence): (String, u8), least: u8| {
                    if precedence >= least {
                        code
                    } else {
                        format!("({code})")
                    }
                };
                let left = wrap(self.truth(left), least);
                let right = wrap(self.truth(right), least + 1);
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
                    let mut lines = vec![format!("let {name} = if {condition} {{")];
                    let (statements, value) = block_value(first, then.code);
                    print(&statements, 1, &mut lines);
                    for part in value.lines() {
                        lines.push(format!("    {part}"));
                    }
                    let (statements, value) = block_value(second, otherwise.code);
                    if statements.is_empty() && value.starts_with("if ") {
                        // `else { if .. }` is `else if ..`.
                        lines.push(format!("}} else {value};"));
                    } else {
                        lines.push("} else {".to_owned());
                        print(&statements, 1, &mut lines);
                        for part in value.lines() {
                            lines.push(format!("    {part}"));
                        }
                        lines.push("};".to_owned());
                    }
                    out.push(Rust::Line(lines.join("\n")));
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
        let name = |node: usize| module.node_name(node).to_owned();
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
        match probe {
            Probe::Voltage(plus, minus) => {
                self.probes.insert(probe);
                let _ = voltage(self, plus, minus, "value");
                Ok(Operand::new(
                    format!("v.voltage_{}_{}", name(plus), name(minus)),
                    ATOM,
                    kind,
                ))
            }
            Probe::Current(branch) => {
                self.probes.insert(probe);
                self.shape.insert(module.branches[branch].name.clone());
                Ok(Operand::new(
                    format!("v.current_{}", module.branches[branch].name),
                    ATOM,
                    kind,
                ))
            }
            Probe::Limited(index) => {
                let limit = module.limits[index].clone();
                let applied = voltage(self, limit.plus, limit.minus, "value");
                if self.mode == Mode::Noise {
                    let name = self.temporary();
                    out.push(Rust::Line(format!("let {name} = R::constant({applied});")));
                    return Ok(Operand::new(name, ATOM, kind));
                }
                let mut arguments =
                    vec![format!("applied{index}"), format!("memory.limited{index}")];
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
                    "let [{name}] = Dual::<R, 1>::variables([({moving}).with_value(limited{index})]);"
                )));
                Ok(Operand::new(name, ATOM, kind))
            }
        }
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

    #[expect(
        clippy::too_many_lines,
        reason = "one arm per construct of the language, which read best side by side"
    )]
    fn statements(&mut self, statements: &[Stmt], out: &mut Vec<Rust>) -> Result<(), Error> {
        let module = self.facts.module;
        for statement in statements {
            if !kept(statement, module, self.mode, &self.live) {
                continue;
            }
            match statement {
                Stmt::Assign(variable, value) => {
                    if self
                        .function
                        .is_some_and(|function| *variable < module.functions[function].inputs)
                    {
                        return Err(Error::at(0, "an analog function assigns its input".into()));
                    }
                    let kind = self.variable_kind(*variable);
                    let name = match self.function {
                        Some(function) => module.functions[function].variables[*variable].0.clone(),
                        None => module.variables[*variable].0.clone(),
                    };
                    // `x = x + e` adds in place, where the kind does: the same sum.
                    if let Expr::Binary(operator @ ("+" | "-"), left, right) = value
                        && matches!(**left, Expr::Variable(read) if read == *variable)
                        && matches!(kind, Kind::Real | Kind::Float(_))
                    {
                        let right = self.expression(right, out)?;
                        let right = self.convert(right, &kind, out)?;
                        out.push(Rust::Line(format!("v.{name} {operator}= {};", right.code)));
                        continue;
                    }
                    let operand = self.expression(value, out)?;
                    let operand = self.convert(operand, &kind, out)?;
                    out.push(Rust::Line(format!("v.{name} = {};", operand.code)));
                }
                Stmt::Contribute {
                    branch,
                    potential,
                    resistive,
                    reactive,
                    noise,
                    ..
                } => match self.mode {
                    Mode::Evaluate => {
                        let site = self.facts.site(
                            *branch,
                            *potential,
                            resistive.as_ref(),
                            reactive.as_ref(),
                        );
                        if site == Site::Collapse {
                            continue;
                        }
                        for (part, charge) in [(resistive, false), (reactive, true)] {
                            if let Some(part) = part {
                                self.accumulate((*branch, site), charge, part, out)?;
                            }
                        }
                    }
                    Mode::Noise => {
                        for source in noise {
                            self.noise_source((*branch, *potential), source, out)?;
                        }
                    }
                    Mode::Prepare | Mode::Warnings | Mode::Function => {}
                },
                Stmt::If(id, condition, then, otherwise) => {
                    let dynamic = module.is_dynamic(condition);
                    if self.mode.prepares() && dynamic {
                        self.statements(then, out)?;
                        self.statements(otherwise, out)?;
                        continue;
                    }
                    let condition = if matches!(self.mode, Mode::Evaluate | Mode::Noise) && !dynamic
                    {
                        if let Some(index) = self.facts.shaped.get(id) {
                            self.shape.insert("conditions".to_owned());
                            (format!("shape.conditions[{index}]"), ATOM)
                        } else {
                            let operand = self.static_operand(condition);
                            self.truth(operand)
                        }
                    } else {
                        let operand = self.expression(condition, out)?;
                        self.truth(operand)
                    };
                    let (mut first, mut second) = (Vec::new(), Vec::new());
                    self.statements(then, &mut first)?;
                    self.statements(otherwise, &mut second)?;
                    out.push(Rust::If(condition, first, second));
                }
                Stmt::While(condition, body) => {
                    if self.mode.prepares() && module.is_dynamic(condition) {
                        self.statements(body, out)?;
                        continue;
                    }
                    let mut before = Vec::new();
                    let operand = self.expression(condition, &mut before)?;
                    if !before.is_empty() {
                        return Err(Error::at(
                            0,
                            "a loop condition this complex is not compiled yet".into(),
                        ));
                    }
                    let (condition, _) = self.truth(operand);
                    let mut inner = Vec::new();
                    self.statements(body, &mut inner)?;
                    out.push(Rust::While(condition, inner));
                }
                Stmt::Fail(message, arguments) => {
                    let message = self.message(message, arguments, out)?;
                    out.push(Rust::Line(format!(
                        "return Err(ParameterError({message}));"
                    )));
                }
                Stmt::Warn(message, arguments, false) => {
                    let message = self.message(message, arguments, out)?;
                    out.push(Rust::Line(format!("found.push({message});")));
                }
                Stmt::Warn(message, arguments, true) => {
                    let index = self
                        .facts
                        .warned
                        .iter()
                        .position(|(warned, count)| warned == message && *count == arguments.len())
                        .unwrap_or_default();
                    out.push(Rust::Line(format!("memory.warned[{index}] = true;")));
                    let offset: usize = self.facts.warned[..index]
                        .iter()
                        .map(|(_, count)| count)
                        .sum();
                    for (place, argument) in arguments.iter().enumerate() {
                        let operand = self.expression(argument, out)?;
                        let value = Self::value_of(&operand).code;
                        out.push(Rust::Line(format!(
                            "memory.warned_values[{}] = {value};",
                            offset + place
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// The message of a task, `message` with each of its `%g`s and their like the value
    /// of the argument of `arguments` at its place, as code that makes the string.
    fn message(
        &mut self,
        message: &str,
        arguments: &[Expr],
        out: &mut Vec<Rust>,
    ) -> Result<String, Error> {
        let mut values = Vec::new();
        for argument in arguments {
            let operand = self.expression(argument, out)?;
            values.push(Self::value_of(&operand).code);
        }
        Ok(format_message(message, &values))
    }

    /// The field of the accumulator of `branch`'s flow or potential, or of its charge or
    /// flux.
    fn accumulator(&self, branch: usize, site: Site, charge: bool) -> String {
        let suffix = match (site, charge) {
            (Site::Potential, false) => "potential",
            (Site::Potential, true) => "flux",
            (_, false) => "flow",
            (_, true) => "charge",
        };
        format!("{}_{suffix}", self.module().branches[branch].name)
    }

    fn accumulate(
        &mut self,
        (branch, site): (usize, Site),
        charge: bool,
        part: &Expr,
        out: &mut Vec<Rust>,
    ) -> Result<(), Error> {
        let kind = self
            .accumulators
            .get(&(branch, site, charge))
            .cloned()
            .unwrap_or(Kind::Real);
        let operand = self.expression(part, out)?;
        let operand = self.convert(operand, &kind, out)?;
        let field = self.accumulator(branch, site, charge);
        let line = if kind == Kind::Real {
            format!("v.{field} += {};", operand.code)
        } else {
            format!("v.{field} = v.{field} + {};", operand.at(SUM + 1))
        };
        out.push(Rust::Line(line));
        Ok(())
    }

    fn noise_source(
        &mut self,
        (branch, potential): (usize, bool),
        source: &crate::ir::Noise,
        out: &mut Vec<Rust>,
    ) -> Result<(), Error> {
        let module = self.facts.module;
        let operand = self.expression(&source.power, out)?;
        let power = Self::value_of(&operand).code;
        let Branch {
            plus,
            minus,
            ref name,
        } = module.branches[branch];
        let equations = if potential {
            self.shape.insert(name.clone());
            format!("[Some(UnknownIndex::new(shape.{name})), None]")
        } else {
            for node in [plus, minus] {
                if module.is_internal(node) {
                    self.shape.insert(module.nodes[node].clone());
                }
            }
            let equation = |node: usize| {
                if node == GROUND {
                    "None".to_owned()
                } else {
                    format!(
                        "Some(UnknownIndex::new({}))",
                        self.facts.node(node, "shape.")
                    )
                }
            };
            format!("[{}, {}]", equation(plus), equation(minus))
        };
        let line = match &source.exponent {
            None => format!(
                "sources.push(NoiseSource::white({:?}, {equations}, {power}));",
                source.name
            ),
            Some(exponent) => {
                if module.is_dynamic(exponent) {
                    return Err(Error::at(
                        0,
                        "a flicker exponent that depends on the probes is not compiled yet".into(),
                    ));
                }
                let exponent = self.expression(exponent, out)?;
                format!(
                    "sources.push(NoiseSource {{\n    name: {:?},\n    equations: {equations},\n    white: 0.0,\n    flicker: {power},\n    exponent: {},\n}});",
                    source.name,
                    Self::value_of(&exponent).code
                )
            }
        };
        out.push(Rust::Line(line));
        Ok(())
    }

    /// The kinds of the accumulators, from the probes each contribution depends on.
    fn find_accumulators(&mut self) {
        let module = self.facts.module;
        let mut probes: BTreeMap<(usize, Site, bool), BTreeSet<Probe>> = BTreeMap::new();
        visit_contributions(&module.analog, &mut |statement| {
            if let Stmt::Contribute {
                branch,
                potential,
                resistive,
                reactive,
                ..
            } = statement
                && self
                    .facts
                    .site(*branch, *potential, resistive.as_ref(), reactive.as_ref())
                    != Site::Collapse
            {
                let site = if *potential {
                    Site::Potential
                } else {
                    Site::Flow
                };
                for (part, charge) in [(resistive, false), (reactive, true)] {
                    if let Some(part) = part {
                        probes
                            .entry((*branch, site, charge))
                            .or_default()
                            .extend(module.probes_of(part));
                    }
                }
            }
        });
        for (key, probes) in probes {
            for probe in &probes {
                if let Probe::Limited(index) = probe {
                    self.extended.insert(*index);
                }
            }
            let kind = if probes.is_empty() {
                Kind::Real
            } else {
                Kind::Dual(probes.into_iter().collect())
            };
            self.accumulators.insert(key, kind);
        }
    }

    /// The local index of `node` in the evaluation, its use recorded.
    fn index(&mut self, place: Place) -> String {
        let module = self.facts.module;
        match place {
            Place::Node(node) if module.is_internal(node) => {
                self.shape.insert(module.nodes[node].clone());
            }
            Place::Current(branch) => {
                self.shape.insert(module.branches[branch].name.clone());
            }
            Place::Node(_) => {}
        }
        self.facts.place(place, "shape.")
    }

    /// Adds `value`, the derivative of the rows' equations by the probe `probe`, to the
    /// stamp's entries: `signs` the sign of each row's.
    fn derivative(
        &mut self,
        rows: &[(Place, &str)],
        probe: Probe,
        value: &str,
        user: (usize, Site),
        capacitance: bool,
        out: &mut Vec<Rust>,
    ) {
        let module = self.facts.module;
        let columns: Vec<(Place, bool)> = match probe {
            Probe::Voltage(plus, minus) => {
                vec![(Place::Node(plus), true), (Place::Node(minus), false)]
            }
            Probe::Limited(index) => {
                let limit = &module.limits[index];
                vec![
                    (Place::Node(limit.plus), true),
                    (Place::Node(limit.minus), false),
                ]
            }
            Probe::Current(branch) => vec![(Place::Current(branch), true)],
        };
        let method = if capacitance {
            "add_capacitance"
        } else {
            "add_conductance"
        };
        let ground = Place::Node(GROUND);
        for &(row, sign) in rows {
            for &(column, positive) in &columns {
                // The ground's equation and value are no unknown's.
                if row == ground || column == ground {
                    continue;
                }
                self.entries.entry((row, column)).or_default().insert(user);
                let name = self.facts.entry_name((row, column));
                let negative = (sign == "-") == positive;
                let value = if negative {
                    format!("-{value}")
                } else {
                    value.to_owned()
                };
                out.push(Rust::Line(format!(
                    "stamp.{method}(shape.{name}, {value});"
                )));
            }
        }
        self.shape.insert("entries".to_owned());
    }

    /// The value of the accumulator `field`, of `kind`, extended linearly from its
    /// limited voltages to the applied ones.
    fn total(field: &str, kind: &Kind) -> String {
        match kind {
            Kind::Dual(probes)
                if probes
                    .iter()
                    .any(|probe| matches!(probe, Probe::Limited(_))) =>
            {
                let steps: Vec<String> = probes
                    .iter()
                    .map(|probe| match probe {
                        Probe::Limited(index) => format!("v.applied{index} - v.limited{index}"),
                        _ => "0.0".to_owned(),
                    })
                    .collect();
                format!("v.{field}.extend([{}])", steps.join(", "))
            }
            Kind::Dual(_) => format!("v.{field}.value"),
            _ => format!("v.{field}"),
        }
    }

    /// Stamps each branch's accumulated contributions, once.
    fn stamps(&mut self, out: &mut Vec<Rust>) {
        let module = self.facts.module;
        for (index, branch) in module.branches.iter().enumerate() {
            for site in [Site::Potential, Site::Flow] {
                if self.facts.flag(index, site) == Activity::Never {
                    continue;
                }
                let mut lines = Vec::new();
                let (plus, minus) = (Place::Node(branch.plus), Place::Node(branch.minus));
                let (p, n) = (self.index(plus), self.index(minus));
                if site == Site::Potential {
                    self.stamp_potential(index, (plus, minus), (&p, &n), &mut lines);
                } else {
                    self.stamp_flow(index, (plus, minus), (&p, &n), &mut lines);
                }
                match self.facts.flag(index, site).index() {
                    Some(flag) => {
                        self.shape.insert("active".to_owned());
                        out.push(Rust::If(
                            (format!("shape.active[{flag}]"), ATOM),
                            lines,
                            Vec::new(),
                        ));
                    }
                    None => out.extend(lines),
                }
            }
        }
    }

    /// The stamp of the branch at `index`, which holds a potential between `places`, at
    /// the local indices `indices`, its current an unknown of its own.
    fn stamp_potential(
        &mut self,
        index: usize,
        (plus, minus): (Place, Place),
        (p, n): (&str, &str),
        lines: &mut Vec<Rust>,
    ) {
        let user = (index, Site::Potential);
        let own = Place::Current(index);
        let c = self.index(own);
        let ground = Place::Node(GROUND);
        lines.push(Rust::Line(format!("let current = at.unknown::<R>({c});")));
        lines.push(Rust::Line(format!("stamp.add_residual({p}, current);")));
        if minus != ground {
            lines.push(Rust::Line(format!("stamp.add_residual({n}, -current);")));
        }
        for (row, value) in [(plus, "R::constant(1.0)"), (minus, "R::constant(-1.0)")] {
            if row == ground {
                continue;
            }
            self.entries.entry((row, own)).or_default().insert(user);
            lines.push(Rust::Line(format!(
                "stamp.add_conductance(shape.{}, {value});",
                self.facts.entry_name((row, own))
            )));
        }
        let potential = self
            .accumulators
            .get(&(index, Site::Potential, false))
            .cloned();
        let total = potential.as_ref().map_or_else(
            || "R::constant(0.0)".to_owned(),
            |kind| Self::total(&self.accumulator(index, Site::Potential, false), kind),
        );
        lines.push(Rust::Line(format!(
            "stamp.add_residual({c}, at.unknown::<R>({p}) - at.unknown::<R>({n}) - {total});"
        )));
        for (column, value) in [(plus, "R::constant(1.0)"), (minus, "R::constant(-1.0)")] {
            if column == ground {
                continue;
            }
            self.entries.entry((own, column)).or_default().insert(user);
            lines.push(Rust::Line(format!(
                "stamp.add_conductance(shape.{}, {value});",
                self.facts.entry_name((own, column))
            )));
        }
        for (charge, kind) in [
            (false, potential),
            (
                true,
                self.accumulators
                    .get(&(index, Site::Potential, true))
                    .cloned(),
            ),
        ] {
            let Some(kind) = kind else { continue };
            let field = self.accumulator(index, Site::Potential, charge);
            if charge {
                lines.push(Rust::Line(format!(
                    "stamp.add_charge({c}, -{});",
                    Self::total(&field, &kind)
                )));
            }
            if let Kind::Dual(probes) = &kind {
                let names: Vec<String> = probes.iter().map(|_| self.temporary()).collect();
                lines.push(Rust::Line(format!(
                    "let [{}] = v.{field}.gradient;",
                    names.join(", ")
                )));
                for (probe, name) in probes.iter().zip(&names) {
                    self.derivative(&[(own, "-")], *probe, name, user, charge, lines);
                }
            }
        }
    }

    /// The stamp of the branch at `index`, whose flow runs between `places`, at the local
    /// indices `indices`.
    fn stamp_flow(
        &mut self,
        index: usize,
        (plus, minus): (Place, Place),
        (p, n): (&str, &str),
        lines: &mut Vec<Rust>,
    ) {
        let user = (index, Site::Flow);
        for charge in [false, true] {
            let Some(kind) = self.accumulators.get(&(index, Site::Flow, charge)).cloned() else {
                continue;
            };
            let field = self.accumulator(index, Site::Flow, charge);
            let total = self.temporary();
            lines.push(Rust::Line(format!(
                "let {total} = {};",
                Self::total(&field, &kind)
            )));
            let method = if charge { "add_charge" } else { "add_residual" };
            lines.push(Rust::Line(format!("stamp.{method}({p}, {total});")));
            if minus != Place::Node(GROUND) {
                lines.push(Rust::Line(format!("stamp.{method}({n}, -{total});")));
            }
            if let Kind::Dual(probes) = &kind {
                let names: Vec<String> = probes.iter().map(|_| self.temporary()).collect();
                lines.push(Rust::Line(format!(
                    "let [{}] = v.{field}.gradient;",
                    names.join(", ")
                )));
                for (probe, name) in probes.iter().zip(&names) {
                    self.derivative(
                        &[(plus, "+"), (minus, "-")],
                        *probe,
                        name,
                        user,
                        charge,
                        lines,
                    );
                }
            }
        }
    }

    /// The body of `linear`, unless no instance is: whether the contributions of an
    /// instance's shape are each zero or linear in the probes, the static numbers each
    /// requires zero. An instance whose memory keeps what depends on the probes, a warning
    /// found or an output variable, is not, as a linear one is evaluated once a run.
    fn linear_body(&mut self) -> Option<String> {
        let module = self.facts.module;
        let remembers = !self.facts.warned.is_empty()
            || module.outputs.iter().any(|&output| module.dynamic[output]);
        let linearity = Linearity::new(module).filter(|_| !remembers)?;
        match self.linear_requirement(&module.analog, &linearity) {
            Requirement::Never => None,
            Requirement::Always => Some("true".to_owned()),
            Requirement::When(code, _) => Some(code),
        }
    }

    /// What the contributions of `statements` require to be linear.
    fn linear_requirement(
        &mut self,
        statements: &[Stmt],
        linearity: &Linearity<'_>,
    ) -> Requirement {
        let mut all = Requirement::Always;
        for statement in statements {
            let requirement = match statement {
                Stmt::Contribute {
                    branch,
                    potential,
                    resistive,
                    reactive,
                    ..
                } if !self.facts.collapses(
                    *branch,
                    *potential,
                    resistive.as_ref(),
                    reactive.as_ref(),
                ) =>
                {
                    let parts: Vec<&Expr> = resistive.iter().chain(reactive).collect();
                    self.site_requirement(&parts, linearity)
                }
                Stmt::If(id, _, then, otherwise) => match self.facts.shaped.get(id) {
                    Some(&index) => {
                        let first = self.linear_requirement(then, linearity);
                        let second = self.linear_requirement(otherwise, linearity);
                        if !matches!(
                            (&first, &second),
                            (Requirement::Always, Requirement::Always)
                        ) {
                            self.shape.insert("conditions".to_owned());
                        }
                        Requirement::choose(&format!("shape.conditions[{index}]"), first, second)
                    }
                    None if stamps(statement) => Requirement::Never,
                    None => Requirement::Always,
                },
                Stmt::Contribute { .. }
                | Stmt::Assign(..)
                | Stmt::While(..)
                | Stmt::Fail(..)
                | Stmt::Warn(..) => Requirement::Always,
            };
            all = all.and(requirement);
        }
        all
    }

    /// What a contribution of `parts`, its resistive and reactive ones, requires to be
    /// linear: one of the ways each part is zero or linear, each requiring its static
    /// numbers zero.
    fn site_requirement(&mut self, parts: &[&Expr], linearity: &Linearity<'_>) -> Requirement {
        let mut ways: Vec<Vec<Expr>> = vec![Vec::new()];
        for part in parts {
            let options = linearity.options(part);
            let acceptable: Vec<&Vec<Expr>> = options
                .iter()
                .filter(|&(form, _)| matches!(form, Form::Zero | Form::Linear))
                .map(|(_, required)| required)
                .collect();
            ways = ways
                .iter()
                .flat_map(|way| {
                    acceptable.iter().map(move |required| {
                        let mut both = way.clone();
                        both.extend(required.iter().filter(|expr| !way.contains(expr)).cloned());
                        both
                    })
                })
                .collect();
        }
        if ways.is_empty() {
            return Requirement::Never;
        }
        if ways.iter().any(Vec::is_empty) {
            return Requirement::Always;
        }
        let mut alternatives = Vec::new();
        for way in &ways {
            let zero: Vec<String> = way
                .iter()
                .map(|expr| {
                    let operand = self.static_operand(expr);
                    format!("{} == 0.0", Self::value_of(&operand).at(SUM))
                })
                .collect();
            alternatives.push(if zero.len() > 1 && ways.len() > 1 {
                format!("({})", zero.join(" && "))
            } else {
                zero.join(" && ")
            });
        }
        let precedence = if ways.len() > 1 {
            OR
        } else if alternatives[0].contains("&&") {
            AND
        } else {
            COMPARE
        };
        Requirement::When(alternatives.join(" || "), precedence)
    }

    /// The lines that bind each probe the body reads.
    fn probe_lines(&self) -> Vec<String> {
        let module = self.facts.module;
        let read = if self.mode == Mode::Evaluate {
            "unknown::<R>"
        } else {
            "value"
        };
        let mut lines = Vec::new();
        for probe in &self.probes {
            let (name, value) = match *probe {
                Probe::Voltage(plus, minus) => (
                    format!(
                        "voltage_{}_{}",
                        module.node_name(plus),
                        module.node_name(minus)
                    ),
                    format!(
                        "at.{read}({}) - at.{read}({})",
                        self.facts.node(plus, "shape."),
                        self.facts.node(minus, "shape.")
                    ),
                ),
                Probe::Current(branch) => (
                    format!("current_{}", module.branches[branch].name),
                    format!("at.{read}(shape.{})", module.branches[branch].name),
                ),
                Probe::Limited(_) => continue,
            };
            lines.push(if self.mode == Mode::Evaluate {
                format!("v.{name} = Dual {{ value: {value}, gradient: [R::constant(1.0)] }};")
            } else {
                format!("v.{name} = R::constant({value});")
            });
        }
        lines
    }

    /// The fields of the evaluation's variables: the dynamic variables kept, and in an
    /// evaluation the accumulators and the voltages limited.
    fn locals(&self) -> Vec<(String, Kind)> {
        let module = self.facts.module;
        let mut locals: Vec<(String, Kind)> = (0..module.variables.len())
            .filter(|&variable| self.live[variable] && module.dynamic[variable])
            .map(|variable| {
                (
                    module.variables[variable].0.clone(),
                    self.variable_kind(variable),
                )
            })
            .collect();
        // Each probe read, by its derivative alone, as a dual number of one.
        for probe in &self.probes {
            let name = match *probe {
                Probe::Voltage(plus, minus) => format!(
                    "voltage_{}_{}",
                    module.node_name(plus),
                    module.node_name(minus)
                ),
                Probe::Current(branch) => format!("current_{}", module.branches[branch].name),
                Probe::Limited(_) => continue,
            };
            let kind = if self.mode == Mode::Evaluate {
                Kind::Dual(vec![*probe])
            } else {
                Kind::Real
            };
            locals.push((name, kind));
        }
        if self.mode == Mode::Evaluate {
            for ((branch, site, charge), kind) in &self.accumulators {
                locals.push((self.accumulator(*branch, *site, *charge), kind.clone()));
            }
            for index in &self.extended {
                locals.push((format!("applied{index}"), Kind::Float(None)));
                locals.push((format!("limited{index}"), Kind::Float(None)));
            }
        }
        locals
    }

    /// The body of `evaluate`.
    fn evaluate_body(&mut self) -> Result<MethodBody, Error> {
        self.find_accumulators();
        let module = self.facts.module;
        let mut body = Vec::new();
        let analog = self.analog.clone();
        self.statements(&analog, &mut body)?;
        self.stamps(&mut body);
        for &output in &module.outputs {
            let operand = self.expression(&Expr::Variable(output), &mut body)?;
            let value = Self::value_of(&operand).code;
            body.push(Rust::Line(format!(
                "memory.{} = {value};",
                module.variables[output].0
            )));
        }
        let mut lines = locals_struct(&self.locals());
        lines.extend(self.probe_lines());
        print(&body, 0, &mut lines);
        let text = lines.join("\n");
        let shape = if mentions(&text, "shape") {
            "shape"
        } else {
            "_"
        };
        let prepared = if mentions(&text, "prepared") {
            "prepared"
        } else {
            "_"
        };
        let memory = if self.facts.remembers() {
            format!(
                "memory: &mut {}",
                Names::from_module(&module.name, false, false).memory
            )
        } else {
            "(): &mut ()".to_owned()
        };
        let lines: Vec<String> = lines
            .into_iter()
            .map(|line| format!("        {line}"))
            .collect();
        Ok(MethodBody {
            shape,
            prepared,
            at: "at",
            memory,
            lines,
        })
    }

    /// The body of `noise`.
    fn noise_body(&mut self) -> Result<MethodBody, Error> {
        let mut body = Vec::new();
        let analog = self.analog.clone();
        self.statements(&analog, &mut body)?;
        let mut lines = locals_struct(&self.locals());
        lines.extend(self.probe_lines());
        print(&body, 0, &mut lines);
        let text = lines.join("\n");
        let shape = if mentions(&text, "shape") {
            "shape"
        } else {
            "_"
        };
        let prepared = if mentions(&text, "prepared") {
            "prepared"
        } else {
            "_"
        };
        let at = if mentions(&text, "at") { "at" } else { "_" };
        let lines: Vec<String> = lines
            .into_iter()
            .map(|line| format!("        {line}"))
            .collect();
        Ok(MethodBody {
            shape,
            prepared,
            at,
            memory: String::new(),
            lines,
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
        let order = left.partial_cmp(&right);
        let holds = match operator {
            "==" => order == Some(Ordering::Equal),
            "!=" => order != Some(Ordering::Equal),
            "<" => order == Some(Ordering::Less),
            "<=" => matches!(order, Some(Ordering::Less | Ordering::Equal)),
            ">" => order == Some(Ordering::Greater),
            _ => matches!(order, Some(Ordering::Greater | Ordering::Equal)),
        };
        return Operand::new(holds.to_string(), ATOM, Kind::Bool);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile;
    use crate::ir::resolve;
    use crate::lexer::tokens;
    use crate::parser::parse;

    /// Returns the body of `pub struct {name}` up to its closing brace.
    fn struct_body<'a>(code: &'a str, header: &str) -> &'a str {
        let start = code
            .find(header)
            .unwrap_or_else(|| panic!("missing {header} in:\n{code}"));
        let rest = &code[start..];
        let end = rest
            .find("\n}\n")
            .unwrap_or_else(|| panic!("unterminated {header} in:\n{rest}"));
        &rest[..end]
    }

    #[test]
    fn integer_real_product_folds_without_prepared_field() {
        let source = "\
`include \"disciplines.vams\"
module m(a, b);
    inout a, b;
    electrical a, b;
    parameter real g = 1.0;
    analog begin
        I(a, b) <+ V(a, b) * g;
        I(a, b) <+ white_noise(2 * 1.602176634e-19 * abs(V(a, b)), \"shot\");
    end
endmodule
";
        let code = compile(source, "m.va", &|_| None).expect("compile");
        let prepared = struct_body(&code, "pub struct PreparedM<R: Real> {");
        assert!(
            !prepared.contains(": f64,"),
            "expected folded constant instead of hoisted f64 field:\n{prepared}"
        );
        assert!(
            code.contains("3.204_353_268e-19"),
            "expected folded constant inline in:\n{code}"
        );
    }

    #[test]
    fn negations_and_signed_products_reuse_one_prepared_field() {
        let source = "\
`include \"disciplines.vams\"
module m(a, b);
    inout a, b;
    electrical a, b;
    parameter real g = 1.0;
    real s;
    analog begin
        s = g * 2.0;
        if (V(a, b) < -3 * s)
            I(a, b) <+ -s * V(a, b);
        else if (V(a, b) >= -(s * 3))
            I(a, b) <+ (s * -3) * V(a, b);
        else
            I(a, b) <+ (s * 3) * V(a, b);
    end
endmodule
";
        let code = compile(source, "m.va", &|_| None).expect("compile");
        let prepared = struct_body(&code, "pub struct PreparedM<R: Real> {");
        let real_fields = prepared
            .lines()
            .filter(|line| line.trim_end().ends_with(": R,"))
            .count();
        assert_eq!(
            real_fields, 2,
            "expected only `s` and one shared `s * 3.0` field:\n{prepared}"
        );
        assert!(!code.contains("-v.s"), "unexpected negated field:\n{code}");
        assert!(
            !code.contains("v.s * -3.0"),
            "unexpected negatively-signed product field:\n{code}"
        );
        assert!(
            !code.contains("-(v.s * 3.0)"),
            "unexpected negated product field:\n{code}"
        );
    }

    #[test]
    fn deduplicates_shaped_conditions_and_active_flags_and_inverts_empty_if() {
        let source = "\
`include \"disciplines.vams\"
module m(a, b);
    inout a, b;
    electrical a, b, n;
    parameter integer mode = 0;
    analog begin
        if (mode != 0)
            V(a, n) <+ 0;
        else
            I(a, n) <+ V(a, n) / 10.0;
        if (mode != 0) begin
            I(n, b) <+ V(n, b) / 100.0;
            I(n, b) <+ ddt(1e-12 * V(n, b));
        end else begin
            I(n, b) <+ V(n, b) / 200.0;
        end
    end
endmodule
";
        let code = compile(source, "m.va", &|_| None).expect("compile");
        assert!(
            code.contains("conditions: [bool; 1],"),
            "identical shaped conditions must share one slot:\n{code}"
        );
        assert!(
            !code.contains("{\n        } else"),
            "empty if-then branch must be inverted:\n{code}"
        );
        assert!(
            code.contains("if !shape.conditions[0] {"),
            "expected inverted condition in evaluate:\n{code}"
        );
        let lines: Vec<&str> = code.lines().map(str::trim).collect();
        for pair in lines.windows(2) {
            assert!(
                !(pair[0].starts_with("active[") && pair[0] == pair[1]),
                "duplicate consecutive active flag assignment `{}`:\n{code}",
                pair[0]
            );
        }
    }

    #[test]
    fn diode_model_folds_constant_deduplicates_conditions_and_inverts_empty_if() {
        let source = include_str!("../models/diode.va");
        let code = compile(source, "crates/jolty-veriloga/models/diode.va", &|_| None)
            .expect("diode compiles");
        let prepared = struct_body(&code, "pub struct PreparedDiode<R: Real> {");
        assert!(
            prepared.contains("conditions: [bool; 5],"),
            "expected 5 deduplicated conditions in diode:\n{prepared}"
        );
        assert!(
            !prepared.contains(": f64,"),
            "expected 2 * P_Q folded without f64 prepared field:\n{prepared}"
        );
        assert!(
            !code.contains("{\n        } else"),
            "expected no empty if blocks in diode"
        );
        let lines: Vec<&str> = code.lines().map(str::trim).collect();
        for pair in lines.windows(2) {
            assert!(
                !(pair[0].starts_with("active[") && pair[0] == pair[1]),
                "duplicate consecutive active flag assignment `{}` in diode",
                pair[0]
            );
        }
    }

    #[test]
    fn mark_visits_noise_exponent_variables() {
        let source = "\
`include \"disciplines.vams\"
module m(a, b);
    inout a, b;
    electrical a, b;
    parameter real p = 1.0;
    real e;
    analog begin
        e = p + 0.5;
        I(a, b) <+ V(a, b) / 100.0;
        I(a, b) <+ flicker_noise(1e-20, e, \"flicker\");
    end
endmodule
";
        let stream = tokens(source, &|_| None).expect("tokens");
        let parsed = parse(&stream).expect("parse");
        let module = resolve(&parsed).expect("resolve");
        let e_index = module
            .variables
            .iter()
            .position(|(name, _)| name == "e")
            .expect("variable e");
        let mut found = BTreeSet::new();
        mark(
            &module.analog,
            &module,
            Mode::Noise,
            &vec![false; module.variables.len()],
            &mut found,
        );
        assert!(
            found.contains(&e_index),
            "mark in Mode::Noise must visit variables in flicker_noise exponent"
        );
    }
}
