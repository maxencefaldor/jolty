//! Shared emitter types, precedence constants, indented [`CodeWriter`], linearity
//! [`Requirement`] algebra, and numeric literal formatting.

use std::collections::BTreeSet;

use crate::ast::Type;
use crate::ir::{Expr, Probe, apart};
use crate::lexer::CONSTANTS;

/// The precedences of Rust's operators, from the loosest, as the printer parenthesizes
/// them.
pub(super) const OR: u8 = 1;
pub(super) const AND: u8 = 2;
pub(super) const COMPARE: u8 = 3;
pub(super) const SUM: u8 = 4;
pub(super) const PRODUCT: u8 = 5;
pub(super) const UNARY: u8 = 6;
pub(super) const ATOM: u8 = 7;

/// The clippy bound on a function's lines, its `too-many-lines-threshold`, beyond which a
/// generated function says why it is long.
const LONGEST: usize = 100;

/// Returns `code`, parenthesized unless `precedence` is at least `least`.
pub(super) fn parenthesize(code: &str, precedence: u8, least: u8) -> String {
    if precedence >= least {
        code.to_owned()
    } else {
        format!("({code})")
    }
}

/// Minimal indented writer accumulating generated Rust source into a single [`String`].
#[derive(Debug, Default)]
pub(super) struct CodeWriter {
    buffer: String,
    depth: usize,
}

impl CodeWriter {
    pub(super) const fn new() -> Self {
        Self {
            buffer: String::new(),
            depth: 0,
        }
    }

    pub(super) const fn with_depth(depth: usize) -> Self {
        Self {
            buffer: String::new(),
            depth,
        }
    }

    /// Writes `text` (splitting embedded newlines) indented at the current depth, leaving
    /// empty lines unindented, terminated by a newline.
    pub(super) fn line(&mut self, text: impl AsRef<str>) {
        let text = text.as_ref();
        if text.is_empty() {
            self.buffer.push('\n');
            return;
        }
        for part in text.lines() {
            if !part.is_empty() {
                for _ in 0..self.depth {
                    self.buffer.push_str("    ");
                }
                self.buffer.push_str(part);
            }
            self.buffer.push('\n');
        }
    }

    /// Writes `open`, runs `body` one level deeper, and closes with `close` (if non-empty).
    pub(super) fn block(
        &mut self,
        open: impl AsRef<str>,
        close: &str,
        body: impl FnOnce(&mut Self),
    ) {
        self.line(open);
        self.depth += 1;
        body(self);
        self.depth -= 1;
        if !close.is_empty() {
            self.line(close);
        }
    }

    /// Writes an empty line.
    pub(super) fn blank(&mut self) {
        self.buffer.push('\n');
    }

    /// Appends `text` verbatim without altering indentation.
    pub(super) fn raw(&mut self, text: &str) {
        self.buffer.push_str(text);
    }

    /// Writes a struct literal `prefix { fields }suffix`, keeping up to two short
    /// single-line fields inline and wrapping larger literals cleanly one field per line.
    pub(super) fn struct_literal(&mut self, prefix: &str, fields: &[String], suffix: &str) {
        if fields.is_empty() {
            self.line(format!("{prefix} {{}}{suffix}"));
            return;
        }
        let single_width: usize = fields.iter().map(String::len).sum::<usize>()
            + fields.len().saturating_sub(1) * 2
            + prefix.len()
            + suffix.len()
            + 4;
        if fields.len() <= 2
            && single_width <= 88
            && fields.iter().all(|field| !field.contains('\n'))
        {
            self.line(format!("{prefix} {{ {} }}{suffix}", fields.join(", ")));
            return;
        }
        self.block(format!("{prefix} {{"), &format!("}}{suffix}"), |writer| {
            for field in fields {
                writer.line(format!("{field},"));
            }
        });
    }

    pub(super) fn finish(self) -> String {
        self.buffer
    }
}

/// What a part of the generated code computes for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Mode {
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
    pub(super) const fn prepares(self) -> bool {
        matches!(self, Self::Prepare | Self::Warnings)
    }
}

/// The kind of a generated expression.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Kind {
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

impl Kind {
    /// The emitted kind of a scalar Verilog-A type, using plain `f64` when `plain` is set.
    pub(super) const fn from_type(kind: Type, plain: bool) -> Self {
        match (kind, plain) {
            (Type::Integer, _) => Self::Integer,
            (Type::Real, true) => Self::Float(None),
            (Type::Real, false) => Self::Real,
        }
    }
}

/// A generated expression: its code, the precedence of its outermost operator, and its
/// kind.
#[derive(Debug, Clone)]
pub(super) struct Operand {
    pub(super) code: String,
    pub(super) precedence: u8,
    pub(super) kind: Kind,
}

impl Operand {
    pub(super) fn new(code: impl Into<String>, precedence: u8, kind: Kind) -> Self {
        Self {
            code: code.into(),
            precedence,
            kind,
        }
    }

    /// The code, parenthesized unless its precedence is at least `least`.
    pub(super) fn at(&self, least: u8) -> String {
        parenthesize(&self.code, self.precedence, least)
    }
}

/// A generated statement, before the printer simplifies and indents it.
#[derive(Debug, Clone)]
pub(super) enum Rust {
    Line(String),
    If((String, u8), Vec<Self>, Vec<Self>),
    While(String, Vec<Self>),
}

/// What an instance requires to be linear: nothing, the impossible, or a disjunction
/// (`||`) of one or more clauses `(code, precedence)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Requirement {
    Always,
    Never,
    Or(Vec<(String, u8)>),
}

impl Requirement {
    /// Renders `clauses` (non-empty) as one Rust boolean expression and its precedence.
    fn render_clauses(clauses: &[(String, u8)]) -> (String, u8) {
        if let [only] = clauses {
            return only.clone();
        }
        let parts: Vec<String> = clauses
            .iter()
            .map(|(code, precedence)| parenthesize(code, *precedence, COMPARE))
            .collect();
        (parts.join(" || "), OR)
    }

    /// Renders this requirement as a Rust boolean expression when satisfiable.
    pub(super) fn render(&self) -> Option<String> {
        match self {
            Self::Never => None,
            Self::Always => Some("true".to_owned()),
            Self::Or(clauses) => Some(Self::render_clauses(clauses).0),
        }
    }

    /// Both this requirement and `other`.
    pub(super) fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::Never, _) | (_, Self::Never) => Self::Never,
            (Self::Always, other) | (other, Self::Always) => other,
            (Self::Or(left), Self::Or(right)) if left == right => Self::Or(left),
            (Self::Or(left), Self::Or(right)) => {
                let (first, first_precedence) = Self::render_clauses(&left);
                let (second, second_precedence) = Self::render_clauses(&right);
                Self::Or(vec![(
                    format!(
                        "{} && {}",
                        parenthesize(&first, first_precedence, AND),
                        parenthesize(&second, second_precedence, AND + 1)
                    ),
                    AND,
                )])
            }
        }
    }

    /// `first` where `condition` holds, and `second` where it does not, factoring any
    /// common disjuncts so `if c { a || b } else { b }` simplifies to `(c && a) || b`.
    pub(super) fn choose(condition: &str, first: Self, second: Self) -> Self {
        let not = format!("!{condition}");
        let prepend = |prefix: String, precedence: u8, mut clauses: Vec<(String, u8)>| {
            if !clauses.iter().any(|(code, _)| code == &prefix) {
                clauses.insert(0, (prefix, precedence));
            }
            Self::Or(clauses)
        };
        let guard = |prefix: &str, clauses: &[(String, u8)]| {
            let (code, precedence) = Self::render_clauses(clauses);
            Self::Or(vec![(
                format!("{prefix} && {}", parenthesize(&code, precedence, AND + 1)),
                AND,
            )])
        };
        match (first, second) {
            (Self::Always, Self::Always) => Self::Always,
            (Self::Never, Self::Never) => Self::Never,
            (Self::Always, Self::Never) => Self::Or(vec![(condition.to_owned(), ATOM)]),
            (Self::Never, Self::Always) => Self::Or(vec![(not, UNARY)]),
            (Self::Always, Self::Or(clauses)) => prepend(condition.to_owned(), ATOM, clauses),
            (Self::Or(clauses), Self::Always) => prepend(not, UNARY, clauses),
            (Self::Never, Self::Or(clauses)) => guard(&not, &clauses),
            (Self::Or(clauses), Self::Never) => guard(condition, &clauses),
            (Self::Or(first), Self::Or(second)) => {
                let (common, only_first): (Vec<_>, Vec<_>) = first
                    .into_iter()
                    .partition(|clause| second.contains(clause));
                let only_second: Vec<_> = second
                    .into_iter()
                    .filter(|clause| !common.contains(clause))
                    .collect();
                if common.is_empty() {
                    let (left, _) = Self::render_clauses(&only_first);
                    let (right, _) = Self::render_clauses(&only_second);
                    return Self::Or(vec![(
                        format!("if {condition} {{ {left} }} else {{ {right} }}"),
                        0,
                    )]);
                }
                let to_requirement = |clauses: Vec<_>| {
                    if clauses.is_empty() {
                        Self::Never
                    } else {
                        Self::Or(clauses)
                    }
                };
                match Self::choose(
                    condition,
                    to_requirement(only_first),
                    to_requirement(only_second),
                ) {
                    Self::Never => Self::Or(common),
                    Self::Always => Self::Always,
                    Self::Or(mut rest) => {
                        for clause in common {
                            if !rest.contains(&clause) {
                                rest.push(clause);
                            }
                        }
                        Self::Or(rest)
                    }
                }
            }
        }
    }
}

/// The rows and columns of a branch's stamp: a node, or the current of a branch that
/// holds a voltage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Place {
    Node(usize),
    Current(usize),
}

/// What a contribution makes its branch: a flow, a potential, or a collapse of its
/// internal node into the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Site {
    Flow,
    Potential,
    Collapse,
}

/// Whether a branch's flow or potential, or its collapse, is active in an instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Activity {
    Never,
    Always,
    /// As the flag at that index among the shape's says.
    Flag(usize),
}

impl Activity {
    /// The index of its flag, if the shape decides it.
    pub(super) const fn index(self) -> Option<usize> {
        match self {
            Self::Flag(index) => Some(index),
            Self::Never | Self::Always => None,
        }
    }
}

/// What the generated code reads of the prepared numbers: each field's name, the kind
/// it has, and the static expression `prepare` computes it as.
#[derive(Debug, Default)]
pub(super) struct Prepared {
    pub(super) fields: Vec<(String, Kind, Expr)>,
}

impl Prepared {
    /// The field for `expr`, added once, named apart from earlier fields and
    /// `conditions`.
    pub(super) fn field(
        &mut self,
        name: impl FnOnce(usize) -> String,
        kind: Kind,
        expr: &Expr,
    ) -> String {
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

/// The rendered body of a generated `evaluate` or `noise` method, before its signature
/// is formatted with the finalized `Prepared` type.
pub(super) struct MethodBody {
    pub(super) shape: &'static str,
    pub(super) prepared: &'static str,
    pub(super) at: &'static str,
    pub(super) memory: String,
    pub(super) body: String,
}

/// The names of a kernel's types.
pub(super) struct Names {
    pub(super) kernel: String,
    pub(super) parameters: String,
    pub(super) shape: String,
    pub(super) prepared: String,
    pub(super) memory: String,
    pub(super) parameters_generic: bool,
    pub(super) prepared_generic: bool,
}

impl Names {
    /// The names of the kernel of `module`, in upper camel case, each word of the
    /// module's name, which its underscores part, capitalized as Rust capitalizes an
    /// acronym.
    pub(super) fn from_module(
        module: &str,
        parameters_generic: bool,
        prepared_generic: bool,
    ) -> Self {
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

    pub(super) fn parameters_type(&self, number: &str) -> String {
        if self.parameters_generic {
            format!("{}<{number}>", self.parameters)
        } else {
            self.parameters.clone()
        }
    }

    pub(super) fn prepared_type(&self, number: &str) -> String {
        if self.prepared_generic {
            format!("{}<{number}>", self.prepared)
        } else {
            self.prepared.clone()
        }
    }
}

/// The Rust type of `kind`.
pub(super) fn rust_type(kind: &Kind) -> String {
    match kind {
        Kind::Bool => "bool".to_owned(),
        Kind::Integer => "i32".to_owned(),
        Kind::Float(_) => "f64".to_owned(),
        Kind::Real => "R".to_owned(),
        Kind::Dual(probes) => format!("Dual<R, {}>", probes.len()),
    }
}

/// The zero of `kind`.
pub(super) fn zero(kind: &Kind) -> String {
    match kind {
        Kind::Bool => "false".to_owned(),
        Kind::Integer => "0".to_owned(),
        Kind::Float(_) => "0.0".to_owned(),
        Kind::Real => "R::constant(0.0)".to_owned(),
        Kind::Dual(_) => "Dual::number(0.0)".to_owned(),
    }
}

/// The probes of `left` and of `right`, together, in order.
pub(super) fn union(left: &[Probe], right: &[Probe]) -> Vec<Probe> {
    let mut all: BTreeSet<Probe> = left.iter().copied().collect();
    all.extend(right.iter().copied());
    all.into_iter().collect()
}

/// The kinds of two operands joined: the kind that holds both.
pub(super) fn join(left: &Kind, right: &Kind) -> Kind {
    match (left, right) {
        (Kind::Dual(left), Kind::Dual(right)) => Kind::Dual(union(left, right)),
        (Kind::Dual(probes), _) | (_, Kind::Dual(probes)) => Kind::Dual(probes.clone()),
        (Kind::Real, _) | (_, Kind::Real) => Kind::Real,
        (Kind::Float(_), _) | (_, Kind::Float(_)) => Kind::Float(None),
        (Kind::Integer, _) | (_, Kind::Integer) => Kind::Integer,
        (Kind::Bool, Kind::Bool) => Kind::Bool,
    }
}

/// Whether `text` holds `name` as a whole word.
pub(super) fn mentions(text: &str, name: &str) -> bool {
    text.match_indices(name).any(|(at, _)| {
        let before = text[..at].chars().next_back();
        let after = text[at + name.len()..].chars().next();
        let word = |character: Option<char>| {
            character.is_some_and(|character| character.is_alphanumeric() || character == '_')
        };
        !word(before) && !word(after)
    })
}

/// The attribute that excuses a function of `body` for its length, when clippy's bound
/// is exceeded, indented by `indent`.
pub(super) fn too_long(body: &str, indent: &str) -> String {
    let counted = body
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty() && !trimmed.starts_with("//")
        })
        .count();
    if counted > LONGEST {
        format!(
            "{indent}#[expect(\n{indent}    clippy::too_many_lines,\n{indent}    reason = \"a kernel's function is its Verilog-A source's analog block, which is one block\"\n{indent})]\n"
        )
    } else {
        String::new()
    }
}

/// A float written as Rust writes it, its long runs of digits grouped by three, as
/// clippy's `unreadable_literal` asks.
pub(super) fn literal(value: f64) -> String {
    let named = CONSTANTS.iter().find(|(_, name, constant)| {
        constant.to_bits() == value.abs().to_bits()
            && !name.starts_with(|character: char| character.is_ascii_digit())
    });
    if let Some((_, name, _)) = named {
        return if value.is_sign_negative() {
            format!("-{name}")
        } else {
            (*name).to_owned()
        };
    }
    let written = format!("{value:?}");
    let (mantissa, exponent) = written
        .find('e')
        .map_or((written.as_str(), ""), |at| written.split_at(at));
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
        let mut chunks: Vec<String> = if from_left {
            characters
                .chunks(3)
                .map(|chunk| chunk.iter().collect())
                .collect()
        } else {
            characters
                .rchunks(3)
                .map(|chunk| chunk.iter().collect())
                .collect()
        };
        if !from_left {
            chunks.reverse();
        }
        chunks.join("_")
    };
    let fraction = if fraction.is_empty() {
        if exponent.is_empty() {
            ".0".to_owned()
        } else {
            String::new()
        }
    } else {
        format!(".{}", group(fraction, true))
    };
    format!("{sign}{}{fraction}{exponent}", group(whole, false))
}

/// The operand of the float `value`.
pub(super) fn float(value: f64) -> Operand {
    let precedence = if value.is_sign_negative() {
        UNARY
    } else {
        ATOM
    };
    Operand::new(literal(value), precedence, Kind::Float(Some(value)))
}
