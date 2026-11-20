//! The expression language of jolty's behavioral sources, compiled once and evaluated with
//! its exact gradient.
//!
#![doc = include_str!("language.md")]
//!
//! ## Example
//!
//! ```
//! use jolty_expressions::{Expression, Variable, Workspace};
//!
//! let expression: Expression = "limit(1e3 * v(in, ref), -1, 1)".parse()?;
//! assert_eq!(
//!     expression.variables(),
//!     [Variable::Voltage("in".to_owned(), Some("ref".to_owned()))]
//! );
//! let mut workspace = Workspace::new(&expression);
//! let mut gradient = [0.0];
//! let value = expression.evaluate(&[0.25e-3], &mut workspace, &mut gradient);
//! assert!((value - 0.25).abs() < 1e-15 && (gradient[0] - 1e3).abs() < 1e-12);
//! # Ok::<(), jolty_expressions::SyntaxError>(())
//! ```

mod lexer;
mod operations;
mod parser;
mod tape;

use std::fmt::{self, Write as _};
use std::str::FromStr;

use tape::Instruction;

/// The language's description, in Markdown, as the reference documents it.
pub const LANGUAGE: &str = include_str!("language.md");

/// A quantity an expression reads.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Variable {
    /// The voltage of a terminal, `v(a)`, or of one over another, `v(a, b)`.
    Voltage(String, Option<String>),
    /// The current through a sensing port, `i(p)`.
    Current(String),
    /// The time, `time`, in seconds.
    Time,
}

/// A term of an expression's [tree](Expression::tree).
pub enum Term {
    /// A number.
    Constant(f64),
    /// A variable the expression reads.
    Variable(Variable),
    /// Operation applied to child operands, named by its expression-language spelling.
    ///
    /// Covers binary operators (`+`, `^`, `<=`), unary operators (`-`, `!`), conditionals
    /// (`if`), and built-in functions (`sqrt`, `min`, `limit`).
    Operation(&'static str, Vec<Term>),
    /// Piecewise-linear interpolation at `input` for `table`, `pwl`, and `rounded_table`,
    /// padded with synthetic endpoints to clamp or extend beyond the table range.
    ///
    /// Each corner (including endpoints) spans the region between adjacent segment midpoints
    /// and is rounded by a parabola over `rounding` times the shorter adjacent segment. Inputs
    /// outside the outer segment midpoints follow the outer segments linearly.
    Table {
        /// Interpolation input expression.
        input: Box<Term>,
        /// Ascending breakpoint abscissae including synthetic endpoints.
        abscissae: Vec<f64>,
        /// Breakpoint ordinates matching `abscissae`.
        ordinates: Vec<f64>,
        /// Fraction of the shorter adjacent segment spanned by each corner-rounding parabola.
        rounding: f64,
    },
}

/// Explicit frame for iterative post-order cloning of [`Term`].
enum CloneFrame<'a> {
    Operation {
        name: &'static str,
        remaining: std::slice::Iter<'a, Term>,
        cloned: Vec<Term>,
    },
    Table {
        abscissae: &'a [f64],
        ordinates: &'a [f64],
        rounding: f64,
    },
}

impl Clone for Term {
    fn clone(&self) -> Self {
        let mut frames: Vec<CloneFrame<'_>> = Vec::new();
        let mut current = self;
        'visit: loop {
            let mut built = loop {
                match current {
                    Self::Constant(value) => break Self::Constant(*value),
                    Self::Variable(variable) => break Self::Variable(variable.clone()),
                    Self::Operation(name, operands) => {
                        let mut remaining = operands.iter();
                        let Some(first) = remaining.next() else {
                            break Self::Operation(name, Vec::new());
                        };
                        frames.push(CloneFrame::Operation {
                            name,
                            remaining,
                            cloned: Vec::with_capacity(operands.len()),
                        });
                        current = first;
                    }
                    Self::Table {
                        input,
                        abscissae,
                        ordinates,
                        rounding,
                    } => {
                        frames.push(CloneFrame::Table {
                            abscissae,
                            ordinates,
                            rounding: *rounding,
                        });
                        current = input.as_ref();
                    }
                }
            };
            while let Some(frame) = frames.pop() {
                match frame {
                    CloneFrame::Operation {
                        name,
                        mut remaining,
                        mut cloned,
                    } => {
                        cloned.push(built);
                        if let Some(next) = remaining.next() {
                            frames.push(CloneFrame::Operation {
                                name,
                                remaining,
                                cloned,
                            });
                            current = next;
                            continue 'visit;
                        }
                        built = Self::Operation(name, cloned);
                    }
                    CloneFrame::Table {
                        abscissae,
                        ordinates,
                        rounding,
                    } => {
                        built = Self::Table {
                            input: Box::new(built),
                            abscissae: abscissae.to_vec(),
                            ordinates: ordinates.to_vec(),
                            rounding,
                        };
                    }
                }
            }
            return built;
        }
    }
}

impl PartialEq for Term {
    fn eq(&self, other: &Self) -> bool {
        let mut stack: Vec<(&Self, &Self)> = Vec::new();
        let mut current = (self, other);
        loop {
            match current {
                (Self::Constant(a), Self::Constant(b)) => {
                    if !a.eq(b) {
                        return false;
                    }
                }
                (Self::Variable(a), Self::Variable(b)) => {
                    if a != b {
                        return false;
                    }
                }
                (Self::Operation(name_a, operands_a), Self::Operation(name_b, operands_b)) => {
                    if name_a != name_b || operands_a.len() != operands_b.len() {
                        return false;
                    }
                    match (operands_a.as_slice(), operands_b.as_slice()) {
                        ([], []) => {}
                        ([only_a], [only_b]) => {
                            current = (only_a, only_b);
                            continue;
                        }
                        _ => stack.extend(operands_a.iter().zip(operands_b).rev()),
                    }
                }
                (
                    Self::Table {
                        input: input_a,
                        abscissae: abscissae_a,
                        ordinates: ordinates_a,
                        rounding: rounding_a,
                    },
                    Self::Table {
                        input: input_b,
                        abscissae: abscissae_b,
                        ordinates: ordinates_b,
                        rounding: rounding_b,
                    },
                ) => {
                    if !rounding_a.eq(rounding_b)
                        || abscissae_a != abscissae_b
                        || ordinates_a != ordinates_b
                    {
                        return false;
                    }
                    current = (input_a.as_ref(), input_b.as_ref());
                    continue;
                }
                _ => return false,
            }
            let Some(next) = stack.pop() else {
                return true;
            };
            current = next;
        }
    }
}

/// Explicit step for iterative [`fmt::Debug`] formatting of [`Term`].
enum DebugStep<'a> {
    Term(&'a Term),
    OperationTail(std::slice::Iter<'a, Term>),
    CloseOperation,
    TableRest {
        abscissae: &'a [f64],
        ordinates: &'a [f64],
        rounding: f64,
    },
}

/// Adapter that indents each line break by `depth` four-space levels when pretty-printing.
struct Indented<'a, 'b> {
    formatter: &'a mut fmt::Formatter<'b>,
    depth: usize,
}

impl fmt::Write for Indented<'_, '_> {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        let mut lines = text.split('\n');
        if let Some(first) = lines.next() {
            self.formatter.write_str(first)?;
        }
        for line in lines {
            self.formatter.write_str("\n")?;
            write_indent(self.formatter, self.depth)?;
            self.formatter.write_str(line)?;
        }
        Ok(())
    }
}

fn write_indent(f: &mut fmt::Formatter<'_>, depth: usize) -> fmt::Result {
    for _ in 0..depth {
        f.write_str("    ")?;
    }
    Ok(())
}

fn write_indented_debug(
    f: &mut fmt::Formatter<'_>,
    depth: usize,
    value: &(impl fmt::Debug + ?Sized),
) -> fmt::Result {
    let mut writer = Indented {
        formatter: f,
        depth,
    };
    write!(writer, "{value:#?}")
}

fn format_table_rest(
    f: &mut fmt::Formatter<'_>,
    pretty: bool,
    depth: &mut usize,
    abscissae: &[f64],
    ordinates: &[f64],
    rounding: f64,
) -> fmt::Result {
    if pretty {
        f.write_str(",\n")?;
        write_indent(f, *depth)?;
        f.write_str("abscissae: ")?;
        write_indented_debug(f, *depth, abscissae)?;
        f.write_str(",\n")?;
        write_indent(f, *depth)?;
        f.write_str("ordinates: ")?;
        write_indented_debug(f, *depth, ordinates)?;
        f.write_str(",\n")?;
        write_indent(f, *depth)?;
        writeln!(f, "rounding: {rounding:?},")?;
        *depth -= 1;
        write_indent(f, *depth)?;
        f.write_str("}")
    } else {
        write!(
            f,
            ", abscissae: {abscissae:?}, ordinates: {ordinates:?}, rounding: {rounding:?} }}"
        )
    }
}

fn format_term<'a>(
    term: &'a Term,
    f: &mut fmt::Formatter<'_>,
    pretty: bool,
    depth: &mut usize,
    stack: &mut Vec<DebugStep<'a>>,
) -> fmt::Result {
    match term {
        Term::Constant(value) => {
            if pretty {
                f.write_str("Constant(\n")?;
                write_indent(f, *depth + 1)?;
                writeln!(f, "{value:?},")?;
                write_indent(f, *depth)?;
                f.write_str(")")
            } else {
                write!(f, "Constant({value:?})")
            }
        }
        Term::Variable(variable) => {
            if pretty {
                f.write_str("Variable(\n")?;
                write_indent(f, *depth + 1)?;
                write_indented_debug(f, *depth + 1, variable)?;
                f.write_str(",\n")?;
                write_indent(f, *depth)?;
                f.write_str(")")
            } else {
                write!(f, "Variable({variable:?})")
            }
        }
        Term::Operation(name, operands) => {
            let mut remaining = operands.iter();
            if let Some(first) = remaining.next() {
                if pretty {
                    f.write_str("Operation(\n")?;
                    write_indent(f, *depth + 1)?;
                    writeln!(f, "{name:?},")?;
                    write_indent(f, *depth + 1)?;
                    f.write_str("[\n")?;
                    *depth += 2;
                    write_indent(f, *depth)?;
                } else {
                    write!(f, "Operation({name:?}, [")?;
                }
                stack.push(DebugStep::CloseOperation);
                stack.push(DebugStep::OperationTail(remaining));
                stack.push(DebugStep::Term(first));
                Ok(())
            } else if pretty {
                f.write_str("Operation(\n")?;
                write_indent(f, *depth + 1)?;
                writeln!(f, "{name:?},")?;
                write_indent(f, *depth + 1)?;
                f.write_str("[],\n")?;
                write_indent(f, *depth)?;
                f.write_str(")")
            } else {
                write!(f, "Operation({name:?}, [])")
            }
        }
        Term::Table {
            input,
            abscissae,
            ordinates,
            rounding,
        } => {
            if pretty {
                f.write_str("Table {\n")?;
                *depth += 1;
                write_indent(f, *depth)?;
                f.write_str("input: ")?;
            } else {
                f.write_str("Table { input: ")?;
            }
            stack.push(DebugStep::TableRest {
                abscissae,
                ordinates,
                rounding: *rounding,
            });
            stack.push(DebugStep::Term(input.as_ref()));
            Ok(())
        }
    }
}

impl fmt::Debug for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pretty = f.alternate();
        let mut depth = 0;
        let mut stack = vec![DebugStep::Term(self)];
        while let Some(step) = stack.pop() {
            match step {
                DebugStep::Term(term) => format_term(term, f, pretty, &mut depth, &mut stack)?,
                DebugStep::OperationTail(mut remaining) => {
                    if let Some(next) = remaining.next() {
                        if pretty {
                            f.write_str(",\n")?;
                            write_indent(f, depth)?;
                        } else {
                            f.write_str(", ")?;
                        }
                        stack.push(DebugStep::OperationTail(remaining));
                        stack.push(DebugStep::Term(next));
                    }
                }
                DebugStep::CloseOperation => {
                    if pretty {
                        f.write_str(",\n")?;
                        depth -= 1;
                        write_indent(f, depth)?;
                        f.write_str("],\n")?;
                        depth -= 1;
                        write_indent(f, depth)?;
                        f.write_str(")")?;
                    } else {
                        f.write_str("])")?;
                    }
                }
                DebugStep::TableRest {
                    abscissae,
                    ordinates,
                    rounding,
                } => format_table_rest(f, pretty, &mut depth, abscissae, ordinates, rounding)?,
            }
        }
        Ok(())
    }
}

impl Drop for Term {
    fn drop(&mut self) {
        let is_empty_leaf = match self {
            Self::Constant(_) | Self::Variable(_) => true,
            Self::Operation(_, operands) => operands.is_empty(),
            Self::Table { input, .. } => {
                matches!(input.as_ref(), Self::Constant(_) | Self::Variable(_))
            }
        };
        if is_empty_leaf {
            return;
        }
        let mut stack = Vec::new();
        detach_term_children(self, &mut stack);
        while let Some(mut term) = stack.pop() {
            detach_term_children(&mut term, &mut stack);
        }
    }
}

fn push_term_if_nested(slot: &mut Term, stack: &mut Vec<Term>) {
    if !matches!(slot, Term::Constant(_) | Term::Variable(_)) {
        stack.push(std::mem::replace(slot, Term::Constant(0.0)));
    }
}

fn detach_term_children(term: &mut Term, stack: &mut Vec<Term>) {
    match term {
        Term::Constant(_) | Term::Variable(_) => {}
        Term::Operation(_, operands) => {
            for operand in std::mem::take(operands) {
                if !matches!(operand, Term::Constant(_) | Term::Variable(_)) {
                    stack.push(operand);
                }
            }
        }
        Term::Table { input, .. } => push_term_if_nested(input, stack),
    }
}

/// A parsed expression, ready to evaluate.
#[derive(Debug, Clone, PartialEq)]
pub struct Expression {
    text: String,
    variables: Vec<Variable>,
    tape: Vec<Instruction>,
}

impl Expression {
    /// Returns the distinct variables referenced by this expression in order of first
    /// appearance, matching the input order of [`Expression::evaluate`].
    #[must_use]
    pub fn variables(&self) -> &[Variable] {
        &self.variables
    }

    /// Evaluates this expression at `variables` (in [`Expression::variables`] order) without
    /// step limiting or derivatives, returning the scalar result.
    ///
    /// # Panics
    ///
    /// Panics if `variables` does not hold one value per variable.
    #[must_use]
    pub fn value(&self, variables: &[f64]) -> f64 {
        assert_eq!(
            variables.len(),
            self.variables.len(),
            "one value per variable"
        );
        tape::value(&self.tape, variables)
    }

    /// Evaluates this expression at `variables` (in [`Expression::variables`] order), writes
    /// its partial derivatives into `gradient`, and returns the scalar result.
    ///
    /// # Panics
    ///
    /// Panics if `variables` or `gradient` does not hold one value per variable, or if
    /// `workspace` was made for another expression.
    pub fn evaluate(
        &self,
        variables: &[f64],
        workspace: &mut Workspace,
        gradient: &mut [f64],
    ) -> f64 {
        assert!(
            variables.len() == self.variables.len() && gradient.len() == self.variables.len(),
            "one value and one derivative per variable"
        );
        assert_eq!(
            workspace.values.len(),
            self.tape.len(),
            "a workspace made for this expression"
        );
        tape::evaluate(&self.tape, variables, workspace, gradient)
    }

    /// Computes directional derivatives along `tangent` (in [`Expression::variables`] order)
    /// at the point last evaluated by [`Expression::evaluate`] in `workspace`.
    ///
    /// Writes the directional derivative of the gradient into `gradient_tangent` and returns
    /// the directional derivative of the scalar value.
    ///
    /// # Panics
    ///
    /// Panics if `tangent` or `gradient_tangent` does not hold one value per variable, or
    /// if `workspace` was made for another expression.
    pub fn along(
        &self,
        tangent: &[f64],
        workspace: &mut Workspace,
        gradient_tangent: &mut [f64],
    ) -> f64 {
        assert!(
            tangent.len() == self.variables.len() && gradient_tangent.len() == self.variables.len(),
            "one rate and one derivative per variable"
        );
        assert_eq!(
            workspace.values.len(),
            self.tape.len(),
            "a workspace made for this expression"
        );
        tape::along(&self.tape, tangent, workspace, gradient_tangent)
    }

    /// Returns the constant-folded expression as a [`Term`] tree for export to external
    /// simulator formats.
    #[must_use]
    pub fn tree(&self) -> Term {
        let mut terms: Vec<Option<Term>> = Vec::with_capacity(self.tape.len());
        let take_term = |terms: &mut [Option<Term>], index: usize| {
            let slot = terms[index].take();
            debug_assert!(slot.is_some(), "tape instruction {index} already consumed");
            slot.unwrap_or(Term::Constant(f64::NAN))
        };
        for instruction in &self.tape {
            let term = match instruction {
                Instruction::Constant(value) => Term::Constant(*value),
                Instruction::Variable(index) => Term::Variable(self.variables[*index].clone()),
                Instruction::Unary(operation, x) => {
                    Term::Operation(operation.name(), vec![take_term(&mut terms, *x)])
                }
                Instruction::Binary(operation, a, b) => Term::Operation(
                    operation.name(),
                    vec![take_term(&mut terms, *a), take_term(&mut terms, *b)],
                ),
                Instruction::Limit(operands) => Term::Operation(
                    "limit",
                    operands
                        .iter()
                        .map(|&index| take_term(&mut terms, index))
                        .collect(),
                ),
                Instruction::Conditional(operands) => Term::Operation(
                    "if",
                    operands
                        .iter()
                        .map(|&index| take_term(&mut terms, index))
                        .collect(),
                ),
                Instruction::Table(x, points) => Term::Table {
                    input: Box::new(take_term(&mut terms, *x)),
                    abscissae: points.abscissae().to_vec(),
                    ordinates: points.ordinates().to_vec(),
                    rounding: points.rounding(),
                },
            };
            terms.push(Some(term));
        }
        terms.pop().flatten().unwrap_or(Term::Constant(f64::NAN))
    }

    /// Appends to `found` the choices the expression's value turned on at its last
    /// evaluation in `workspace`: each comparison, step, sign, floor and ceiling that the
    /// value read, through the branches it took. The value jumps where one of them changes.
    ///
    /// # Panics
    ///
    /// Panics if `workspace` was made for another expression.
    pub fn choices(&self, workspace: &Workspace, found: &mut Vec<Choice>) {
        assert_eq!(
            workspace.values.len(),
            self.tape.len(),
            "a workspace made for this expression"
        );
        tape::choices(&self.tape, &workspace.values, found);
    }
}

/// A choice an expression's value turned on at one evaluation: a comparison of its
/// operands, or a step, a sign, a floor or a ceiling of its argument. The value jumps
/// where the choice changes, its input crossing a threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Choice {
    /// Tape instruction index identifying this choice within its expression.
    instruction: usize,
    /// Tracked input quantity: left minus right for comparisons, or the function argument.
    input: f64,
    /// Discrete output value selected at evaluation time.
    chosen: f64,
    kind: tape::Kind,
}

impl Choice {
    /// Returns the earliest fractional position in `[0, 1]` between `before` and `after`
    /// where any shared choice changes value, using linear input interpolation.
    ///
    /// Returns `None` when no shared choice changed. Comparisons, steps, and signs switch at
    /// zero; `floor` and `ceil` switch at the adjacent integer boundary between the two
    /// evaluations.
    #[must_use]
    pub fn first_jump(before: &[Self], after: &[Self]) -> Option<f64> {
        before
            .iter()
            .filter_map(|first| {
                let last = after
                    .iter()
                    .find(|choice| choice.instruction == first.instruction)?;
                (first.chosen.to_bits() != last.chosen.to_bits()).then(|| first.crossing(last))
            })
            .min_by(f64::total_cmp)
    }

    /// Interpolates the fractional step from `self` to `later` at which the tracked input
    /// crosses the switching threshold.
    fn crossing(&self, later: &Self) -> f64 {
        let rising = later.chosen > self.chosen;
        let threshold = match self.kind {
            tape::Kind::Crossing => 0.0,
            tape::Kind::Floor if rising => self.chosen + 1.0,
            tape::Kind::Floor => self.chosen,
            tape::Kind::Ceil if rising => self.chosen,
            tape::Kind::Ceil => self.chosen - 1.0,
        };
        let fraction = (threshold - self.input) / (later.input - self.input);
        if fraction.is_finite() {
            fraction.clamp(0.0, 1.0)
        } else {
            1.0
        }
    }
}

impl FromStr for Expression {
    type Err = SyntaxError;

    fn from_str(text: &str) -> Result<Self, SyntaxError> {
        let (node, variables) = parser::parse(text)?;
        Ok(Self {
            text: text.to_owned(),
            variables,
            tape: tape::compile(&node),
        })
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

/// The memory one expression's evaluation works in, kept between evaluations: the last
/// inputs of its rounded tables, from which the next steps are limited.
#[derive(Debug, Clone)]
pub struct Workspace {
    values: Vec<f64>,
    adjoints: Vec<f64>,
    /// Each limited table's last input.
    inputs: Vec<Option<f64>>,
    /// Each table's slope at the input it was evaluated at.
    slopes: Vec<f64>,
    /// Each table's second derivative at the input it was evaluated at.
    curvatures: Vec<f64>,
    /// Directional derivative of each instruction's value during [`Expression::along`].
    tangents: Vec<f64>,
    /// Directional derivative of each instruction's adjoint during [`Expression::along`].
    adjoint_tangents: Vec<f64>,
    limited: bool,
}

impl Workspace {
    /// Allocates a reusable [`Workspace`] sized for `expression`.
    #[must_use]
    pub fn new(expression: &Expression) -> Self {
        let length = expression.tape.len();
        Self {
            values: vec![0.0; length],
            adjoints: vec![0.0; length],
            inputs: vec![None; length],
            slopes: vec![0.0; length],
            curvatures: vec![0.0; length],
            tangents: vec![0.0; length],
            adjoint_tangents: vec![0.0; length],
            limited: false,
        }
    }

    /// Forgets the last inputs of the rounded tables, so the next evaluation limits no
    /// step: as when an iteration starts afresh.
    pub fn forget(&mut self) {
        self.inputs.fill(None);
    }

    /// Returns whether the last evaluation clamped a rounded-table input, indicating an
    /// unconverged Newton step.
    #[must_use]
    pub fn limited(&self) -> bool {
        self.limited
    }
}

/// Error describing why expression text failed to parse and the one-based failure column.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message} at column {column}")]
pub struct SyntaxError {
    /// Human-readable description of the syntax error.
    pub message: String,
    /// One-based column of the offending token.
    pub column: usize,
}

impl SyntaxError {
    pub(crate) fn new(message: String, column: usize) -> Self {
        Self { message, column }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn a_tree_names_each_operation_and_folds_constants() {
        let expression: Expression = "-v(a) ^ 2 ^ 3 + limit(time, 2 * pi, 0) + log(i(p))"
            .parse()
            .expect("valid");
        let voltage = Term::Variable(Variable::Voltage("a".to_owned(), None));
        let power = Term::Operation("^", vec![voltage, Term::Constant(8.0)]);
        let limit = Term::Operation(
            "limit",
            vec![
                Term::Variable(Variable::Time),
                Term::Constant(2.0 * std::f64::consts::PI),
                Term::Constant(0.0),
            ],
        );
        let logarithm = Term::Operation(
            "ln",
            vec![Term::Variable(Variable::Current("p".to_owned()))],
        );
        let sum = Term::Operation("+", vec![Term::Operation("-", vec![power]), limit]);
        assert_eq!(
            expression.tree(),
            Term::Operation("+", vec![sum, logarithm])
        );
    }

    #[test]
    fn a_tables_tree_holds_its_points_with_their_ends() {
        let expression: Expression = "rounded_table(v(a), 0.25, 0, 1, 2, 3) + if(v(a), 1, 0)"
            .parse()
            .expect("valid");
        let input = Box::new(Term::Variable(Variable::Voltage("a".to_owned(), None)));
        let table = Term::Table {
            input: input.clone(),
            abscissae: vec![-2.0, 0.0, 2.0, 4.0],
            ordinates: vec![1.0, 1.0, 3.0, 3.0],
            rounding: 0.25,
        };
        let choice = Term::Operation("if", vec![*input, Term::Constant(1.0), Term::Constant(0.0)]);
        assert_eq!(expression.tree(), Term::Operation("+", vec![table, choice]));
    }

    /// Parses and evaluates `text` at `values`, returning `(value, gradient)`.
    fn evaluate(text: &str, values: &[f64]) -> (f64, Vec<f64>) {
        let expression: Expression = text.parse().expect("a valid expression");
        let mut workspace = Workspace::new(&expression);
        let mut gradient = vec![0.0; values.len()];
        let value = expression.evaluate(values, &mut workspace, &mut gradient);
        (value, gradient)
    }

    fn value(text: &str) -> f64 {
        evaluate(text, &[]).0
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn operators_bind_by_precedence() {
        for (text, expected) in [
            ("1 + 2 * 3", 7.0),
            ("(1 + 2) * 3", 9.0),
            ("-2 ^ 2", -4.0),
            ("2 ^ 3 ^ 2", 512.0),
            ("2 ^ -1", 0.5),
            ("8 / 4 / 2", 1.0),
            ("1 - 2 - 3", -4.0),
            ("1 < 2 == 1", 1.0),
            ("0 || 1 && 0", 0.0),
            ("1 ? 2 : 0 ? 3 : 4", 2.0),
            ("0 ? 2 : 0 ? 3 : 4", 4.0),
            ("!0 + !5", 1.0),
            ("u(0) + u(-1) + sgn(-3)", -0.5),
            ("limit(5, 2, -2) + limit(-5, -2, 2)", 0.0),
            ("table(2.5, 0, 0, 2, 10, 3, 20)", 15.0),
            ("min(2, 3) * max(2, 3) + pow(2, 10)", 1030.0),
        ] {
            assert!(close(value(text), expected), "{text} is {}", value(text));
        }
    }

    #[test]
    fn variables_are_numbered_once_in_order_first_read() {
        let expression: Expression = "v(a) * i(sense) + v(a) + time + v(b, a)"
            .parse()
            .expect("valid");
        assert_eq!(
            expression.variables(),
            [
                Variable::Voltage("a".to_owned(), None),
                Variable::Current("sense".to_owned()),
                Variable::Time,
                Variable::Voltage("b".to_owned(), Some("a".to_owned())),
            ]
        );
        assert_eq!(
            expression.to_string(),
            "v(a) * i(sense) + v(a) + time + v(b, a)"
        );
    }

    #[test]
    fn the_gradient_follows_the_branch_taken() {
        // The other branch's slope is infinite at zero, but it is not taken.
        let (value, gradient) = evaluate("v(x) > 0 ? sqrt(v(x)) : 2 * v(x)", &[0.0]);
        assert!(value == 0.0 && close(gradient[0], 2.0));
        let (_, gradient) = evaluate("limit(v(x), v(a), v(b))", &[3.0, 1.0, 2.0]);
        assert!(gradient[0] == 0.0 && gradient[1] == 0.0 && close(gradient[2], 1.0));
        let (_, gradient) = evaluate("table(v(x), 0, 0, 1, 4)", &[0.5]);
        assert!(close(gradient[0], 4.0));
    }

    #[test]
    fn reports_what_is_wrong_and_where() {
        for (text, message) in [
            ("1 +", "expected an operand, not the end at column 4"),
            ("(1", "expected `)`, not the end at column 3"),
            ("foo(1)", "unknown function `foo` at column 1"),
            ("2 * x", "unknown name `x` at column 5"),
            ("max(1)", "`max` takes 2 arguments, not 1 at column 1"),
            ("v(1)", "expected a name, not the number `1` at column 3"),
            (
                "1 2",
                "expected an operator, not the number `2` at column 3",
            ),
            (
                "table(v(a), 1, 0, 0, 1)",
                "`table` takes an input, then constant points, in pairs, with ascending \
                 abscissae at column 1",
            ),
            (
                "table(v(a), v(b), 0)",
                "`table` takes an input, then constant points, in pairs, with ascending \
                 abscissae at column 1",
            ),
            (
                "rounded_table(v(a), 0.7, 0, 0)",
                "`rounded_table` takes an input, a rounding from 0 to 0.5, then constant \
                 points, in pairs, with ascending abscissae at column 1",
            ),
        ] {
            let error = text.parse::<Expression>().expect_err(text);
            assert_eq!(error.to_string(), message);
        }
    }

    /// Evaluates `expression` at `values` and returns all active discontinuity choices.
    fn chosen(expression: &Expression, values: &[f64]) -> Vec<Choice> {
        let mut workspace = Workspace::new(expression);
        let mut gradient = vec![0.0; values.len()];
        expression.evaluate(values, &mut workspace, &mut gradient);
        let mut found = Vec::new();
        expression.choices(&workspace, &mut found);
        found
    }

    #[test]
    fn choices_are_those_the_branches_taken_read() {
        let expression: Expression = "v(x) > 0 ? sqrt(v(x)) : u(v(y)) + sgn(v(y))"
            .parse()
            .expect("valid");
        let taken = chosen(&expression, &[1.0, 2.0]);
        assert_eq!(taken.len(), 1);
        assert!(close(taken[0].input, 1.0) && close(taken[0].chosen, 1.0));
        assert_eq!(chosen(&expression, &[-1.0, 2.0]).len(), 3);
        let smooth: Expression = "min(v(x), 1) + limit(v(x), 0, 2) + abs(v(x))"
            .parse()
            .expect("valid");
        assert_eq!(chosen(&smooth, &[0.5]), []);
    }

    #[test]
    fn the_first_jump_is_where_the_earliest_choice_changes() {
        let compared: Expression = "v(x) > 0 ? 5 : v(y) > 1".parse().expect("valid");
        let jump = |expression: &Expression, before: &[f64], after: &[f64]| {
            Choice::first_jump(&chosen(expression, before), &chosen(expression, after))
        };
        assert_eq!(jump(&compared, &[-1.0, 0.0], &[3.0, 0.0]), Some(0.25));
        assert_eq!(jump(&compared, &[-1.0, 0.0], &[-0.5, 0.5]), None);
        // The second comparison is read at both ends, and changes earlier.
        assert_eq!(jump(&compared, &[-1.0, 0.0], &[-0.5, 4.0]), Some(0.25));
        let floor: Expression = "floor(v(x))".parse().expect("valid");
        let ceiling: Expression = "ceil(v(x))".parse().expect("valid");
        for expression in [&floor, &ceiling] {
            let rising = jump(expression, &[0.5], &[2.7]).expect("a jump");
            let falling = jump(expression, &[2.7], &[0.5]).expect("a jump");
            assert!(close(rising, 0.5 / 2.2) && close(falling, 0.7 / 2.2));
        }
        assert_eq!(jump(&floor, &[0.5], &[0.9]), None);
    }

    proptest! {
        /// Verifies that analytic gradients match central finite differences at smooth points.
        #[test]
        fn gradients_match_finite_differences(
            x in -2.0..2.0f64,
            y in 0.5..3.0f64,
            z in -1.0..1.0f64,
        ) {
            let text = "v(x) * exp(v(y) / 4) - sin(v(z)) / v(y) + v(y) ^ v(x) \
                        + atan(v(x) * v(z)) + sqrt(v(y)) * tanh(v(z)) + ln(v(y)) * v(x) ^ 2 \
                        + max(v(x), v(z)) + limit(v(z), -0.5, 0.5) \
                        + table(v(x), -1, 2, 0, -1, 1, 3) \
                        + rounded_table(v(z), 0.1, -0.5, 1, 0, 0, 0.5, 2)";
            let values = [x, y, z];
            let (_, gradient) = evaluate(text, &values);
            let value = evaluate(text, &values).0;
            for (index, derivative) in gradient.iter().enumerate() {
                let step = 1e-6;
                let mut above = values;
                above[index] += step;
                let mut below = values;
                below[index] -= step;
                let (high, low) = (evaluate(text, &above).0, evaluate(text, &below).0);
                // Corners from `max`, `limit`, and `table` create slope jumps of at least 1
                // between one-sided differences. At smooth points the one-sided difference
                // equals curvature times step (below 1e-2 when curvature < 1e4), so central
                // differences accurately approximate the derivative.
                let (forward, backward) = ((high - value) / step, (value - low) / step);
                if (forward - backward).abs() > 1e-2 * derivative.abs().max(1.0) {
                    continue;
                }
                let numeric = (high - low) / (2.0 * step);
                prop_assert!((numeric - derivative).abs() < 1e-5 * derivative.abs().max(1.0),
                    "variable {index}: analytic {derivative}, numeric {numeric}");
            }
        }

        /// Verifies that directional value and gradient derivatives match central finite
        /// differences wherever the gradient is smooth.
        #[test]
        fn second_derivatives_match_finite_differences(
            x in -0.8..0.8f64,
            y in 1.2..2.8f64,
            z in -0.8..0.8f64,
            dx in -1.0..1.0f64,
            dy in -1.0..1.0f64,
            dz in -1.0..1.0f64,
        ) {
            let text = "v(x) * exp(v(y) / 4) - sin(v(z)) / v(y) + cos(v(x)) * tan(v(z) / 2) \
                        + v(y) ^ v(x) + asin(v(x) / 2) + acos(v(z) / 2) + atan(v(x) * v(z)) \
                        + sinh(v(x)) + cosh(v(z)) + sqrt(v(y)) * tanh(v(z)) \
                        + asinh(v(x)) + acosh(v(y)) + atanh(v(z) / 2) \
                        + ln(v(y)) * v(x) ^ 2 + log10(v(y)) * v(z) \
                        + (v(y) > 0 ? v(x) * v(z) : 1 / v(x)) \
                        + rounded_table(v(z), 0.2, -0.5, 1, 0, 0, 0.5, 2)";
            let expression: Expression = text.parse().expect("a valid expression");
            let mut workspace = Workspace::new(&expression);
            let values = [x, y, z];
            let tangent = [dx, dy, dz];
            let mut gradient = [0.0; 3];
            let _ = expression.evaluate(&values, &mut workspace, &mut gradient);
            let mut gradient_tangent = [0.0; 3];
            let value_tangent = expression.along(&tangent, &mut workspace, &mut gradient_tangent);
            let expected_value_tangent: f64 = gradient
                .iter()
                .zip(&tangent)
                .map(|(slope, rate)| slope * rate)
                .sum();
            prop_assert!(
                (value_tangent - expected_value_tangent).abs()
                    < 1e-12 * expected_value_tangent.abs().max(1.0)
            );
            let step = 1e-6;
            let shifted = |sign: f64| {
                let point = [
                    values[0] + sign * step * tangent[0],
                    values[1] + sign * step * tangent[1],
                    values[2] + sign * step * tangent[2],
                ];
                let mut moved_gradient = [0.0; 3];
                let mut side = Workspace::new(&expression);
                let moved_value = expression.evaluate(&point, &mut side, &mut moved_gradient);
                (moved_value, moved_gradient)
            };
            let (_, high) = shifted(1.0);
            let (_, low) = shifted(-1.0);
            for index in 0..3 {
                let (forward, backward) = (
                    (high[index] - gradient[index]) / step,
                    (gradient[index] - low[index]) / step,
                );
                if (forward - backward).abs() > 1e-2 * gradient_tangent[index].abs().max(1.0) {
                    continue;
                }
                let numeric = (high[index] - low[index]) / (2.0 * step);
                prop_assert!(
                    (numeric - gradient_tangent[index]).abs()
                        < 1e-5 * gradient_tangent[index].abs().max(numeric.abs()).max(1.0),
                    "gradient[{index}]: exact {}, numeric {numeric}",
                    gradient_tangent[index]
                );
            }
        }
    }

    #[test]
    fn along_guards_zero_times_infinite_curvature_and_slope() {
        // `v(x)^1.5 + v(y)^2` at (0, 1): d²/dx² = +inf, but along [0, 1] x does not move.
        let power_sum: Expression = "v(x) ^ 1.5 + v(y) ^ 2".parse().expect("valid");
        let mut workspace = Workspace::new(&power_sum);
        let mut gradient = [0.0; 2];
        let _ = power_sum.evaluate(&[0.0, 1.0], &mut workspace, &mut gradient);
        let mut gradient_tangent = [0.0; 2];
        let value_tangent = power_sum.along(&[0.0, 1.0], &mut workspace, &mut gradient_tangent);
        assert_eq!(value_tangent, 2.0);
        assert_eq!(gradient_tangent, [0.0, 2.0]);

        // `v(x) * v(y)^1.5` at (0, 0): adjoint of `v(y)^1.5` is 0 while adjoint_tangent is 1.
        let power_product: Expression = "v(x) * v(y) ^ 1.5".parse().expect("valid");
        let mut workspace = Workspace::new(&power_product);
        let mut gradient = [0.0; 2];
        let _ = power_product.evaluate(&[0.0, 0.0], &mut workspace, &mut gradient);
        let mut gradient_tangent = [0.0; 2];
        let value_tangent = power_product.along(&[1.0, 0.0], &mut workspace, &mut gradient_tangent);
        assert_eq!(value_tangent, 0.0);
        assert_eq!(gradient_tangent, [0.0, 0.0]);

        // `sqrt(v(x)) + v(y)^2` at (0, 1): d/dx = +inf, but along [0, 1] x does not move.
        let root_sum: Expression = "sqrt(v(x)) + v(y) ^ 2".parse().expect("valid");
        let mut workspace = Workspace::new(&root_sum);
        let mut gradient = [0.0; 2];
        let _ = root_sum.evaluate(&[0.0, 1.0], &mut workspace, &mut gradient);
        let mut gradient_tangent = [0.0; 2];
        let value_tangent = root_sum.along(&[0.0, 1.0], &mut workspace, &mut gradient_tangent);
        assert_eq!(value_tangent, 2.0);
        assert_eq!(gradient_tangent, [0.0, 2.0]);
    }

    #[test]
    fn parser_rejects_excessive_nesting_depth_without_stack_overflow() {
        let limit = parser::MAX_DEPTH;
        let at_limit = format!("{}1{}", "(".repeat(limit), ")".repeat(limit));
        assert_eq!(value(&at_limit).to_bits(), 1.0_f64.to_bits());

        let deep_parens = format!("{}1{}", "(".repeat(limit + 1), ")".repeat(limit + 1));
        assert!(
            deep_parens
                .parse::<Expression>()
                .expect_err("depth above MAX_DEPTH must fail")
                .to_string()
                .contains("levels deep")
        );

        let very_deep_parens = format!("{}1{}", "(".repeat(1000), ")".repeat(1000));
        assert!(very_deep_parens.parse::<Expression>().is_err());

        let deep_unary = format!("{}1", "-".repeat(1000));
        assert!(deep_unary.parse::<Expression>().is_err());

        let deep_power = format!("{}2", "2 ^ ".repeat(1000));
        assert!(deep_power.parse::<Expression>().is_err());

        let deep_ternary = format!("{}1{}", "1 ? ".repeat(300), " : 2".repeat(300));
        assert!(deep_ternary.parse::<Expression>().is_err());

        let deep_calls = format!("{}0{}", "sin(".repeat(300), ")".repeat(300));
        assert!(deep_calls.parse::<Expression>().is_err());

        // Right-associative power with unary signs and chained ternary expressions match
        // the exact boundary at MAX_DEPTH, and worst-case mixed precedence inside every
        // parenthesized level runs in constant call-stack space.
        let at_limit_mixed = format!("{}1{}", "1||1&&1==1<1+1*(".repeat(limit), ")".repeat(limit));
        assert_eq!(value(&at_limit_mixed).to_bits(), 1.0_f64.to_bits());
        assert_eq!(
            value("2 ^ -3 ^ 2").to_bits(),
            (2.0_f64.powf(-9.0)).to_bits()
        );
        assert_eq!(value("-2 ^ -1").to_bits(), (-0.5_f64).to_bits());
        assert_eq!(value("1 ? 2 ? 3 : 4 : 5").to_bits(), 3.0_f64.to_bits());
        assert_eq!(value("0 ? 1 ? 2 : 3 : 4").to_bits(), 4.0_f64.to_bits());
        for (bad, expected_fragment) in [
            ("1 : 2", "expected an operator, not `:`"),
            ("(1 : 2)", "expected `)`, not `:`"),
            ("1 ? 2", "expected `:`, not the end"),
            ("(1 ? 2)", "expected `:`, not `)`"),
            ("max(1 ? 2, 3)", "expected `:`, not `,`"),
            ("(1, 2)", "expected `)`, not `,`"),
            ("sin()", "expected an operand, not `)`"),
        ] {
            let error = bad.parse::<Expression>().expect_err(bad);
            assert!(
                error.to_string().contains(expected_fragment),
                "`{bad}` gave `{error}`"
            );
        }

        // Flat binary chains use the iterative operator stack and do not hit MAX_DEPTH.
        let wide_sum = format!("1{}", " + 1".repeat(500));
        assert_eq!(value(&wide_sum).to_bits(), 501.0_f64.to_bits());

        // Long variable binary chains cannot be constant-folded; parsing, compilation,
        // evaluation, tree construction, and dropping must all stay iterative on a 2 MiB
        // debug stack.
        let terms = 5_000;
        let long_variable_sum = format!("v(a){}", " + v(a)".repeat(terms - 1));
        let expression: Expression = long_variable_sum.parse().expect("valid");
        let mut workspace = Workspace::new(&expression);
        let mut gradient = [0.0];
        let evaluated = expression.evaluate(&[2.0], &mut workspace, &mut gradient);
        assert_eq!(evaluated.to_bits(), 10_000.0_f64.to_bits());
        assert_eq!(gradient[0].to_bits(), 5_000.0_f64.to_bits());
        let tree = expression.tree();
        drop(tree);
    }

    #[test]
    fn table_rejects_non_finite_abscissae_and_ordinates() {
        for invalid in [
            "table(v(a), 0/0, 0, 1, 1)",
            "table(v(a), 0, 0, 0/0, 1)",
            "table(v(a), 0, 0/0, 1, 1)",
            "table(v(a), -1/0, 0, 1, 1)",
            "table(v(a), 0, 0, 1/0, 1)",
            "table(v(a), 0, 1/0, 1, 1)",
            "pwl(v(a), 0/0, 1, 2, 3)",
            "rounded_table(v(a), 0.25, 0/0, 0, 1, 1)",
            "rounded_table(v(a), 0/0, 0, 0, 1, 1)",
            "rounded_table(v(a), 1/0, 0, 0, 1, 1)",
            "table(0, 0/0, 1, 1, 2)",
        ] {
            let error = invalid.parse::<Expression>().unwrap_err();
            assert!(
                error.to_string().contains("ascending abscissae"),
                "unexpected error for `{invalid}`: {error}"
            );
        }
    }

    #[test]
    fn term_clone_eq_and_debug_stay_iterative_on_small_stack() {
        let handle = std::thread::Builder::new()
            .stack_size(2 * 1024 * 1024)
            .spawn(|| {
                let terms = 5_000;
                let long_sum = format!("v(a){}", " + v(a)".repeat(terms - 1));
                let expression: Expression = long_sum.parse().expect("valid");
                let tree = expression.tree();
                let cloned = tree.clone();
                assert_eq!(tree, cloned);
                let formatted = format!("{tree:?}");
                assert!(formatted.starts_with("Operation(\"+\", ["));
                assert!(formatted.contains("Voltage(\"a\", None)"));
            })
            .expect("spawn thread");
        handle
            .join()
            .expect("thread succeeded without stack overflow");
    }

    #[test]
    fn term_debug_matches_compact_and_pretty_layouts() {
        let constant = Term::Constant(1.5);
        assert_eq!(format!("{constant:?}"), "Constant(1.5)");
        assert_eq!(format!("{constant:#?}"), "Constant(\n    1.5,\n)");

        let time = Term::Variable(Variable::Time);
        assert_eq!(format!("{time:?}"), "Variable(Time)");
        assert_eq!(format!("{time:#?}"), "Variable(\n    Time,\n)");

        let empty = Term::Operation("f", Vec::new());
        assert_eq!(format!("{empty:?}"), "Operation(\"f\", [])");
        assert_eq!(format!("{empty:#?}"), "Operation(\n    \"f\",\n    [],\n)");

        let nested = Term::Operation(
            "+",
            vec![
                Term::Operation("-", vec![Term::Constant(1.0)]),
                Term::Variable(Variable::Voltage("a".to_owned(), None)),
            ],
        );
        assert_eq!(
            format!("{nested:?}"),
            "Operation(\"+\", [Operation(\"-\", [Constant(1.0)]), Variable(Voltage(\"a\", None))])"
        );
        assert_eq!(
            format!("{nested:#?}"),
            concat!(
                "Operation(\n",
                "    \"+\",\n",
                "    [\n",
                "        Operation(\n",
                "            \"-\",\n",
                "            [\n",
                "                Constant(\n",
                "                    1.0,\n",
                "                ),\n",
                "            ],\n",
                "        ),\n",
                "        Variable(\n",
                "            Voltage(\n",
                "                \"a\",\n",
                "                None,\n",
                "            ),\n",
                "        ),\n",
                "    ],\n",
                ")"
            )
        );

        let table = Term::Table {
            input: Box::new(Term::Variable(Variable::Current("p".to_owned()))),
            abscissae: vec![0.0, 1.0],
            ordinates: vec![2.0, 3.0],
            rounding: 0.25,
        };
        assert_eq!(
            format!("{table:?}"),
            "Table { input: Variable(Current(\"p\")), abscissae: [0.0, 1.0], \
             ordinates: [2.0, 3.0], rounding: 0.25 }"
        );
        assert_eq!(
            format!("{table:#?}"),
            concat!(
                "Table {\n",
                "    input: Variable(\n",
                "        Current(\n",
                "            \"p\",\n",
                "        ),\n",
                "    ),\n",
                "    abscissae: [\n",
                "        0.0,\n",
                "        1.0,\n",
                "    ],\n",
                "    ordinates: [\n",
                "        2.0,\n",
                "        3.0,\n",
                "    ],\n",
                "    rounding: 0.25,\n",
                "}"
            )
        );
    }

    #[test]
    fn term_partial_eq_handles_nan_signed_zero_and_mismatches() {
        assert_ne!(Term::Constant(f64::NAN), Term::Constant(f64::NAN));
        assert_eq!(Term::Constant(-0.0), Term::Constant(0.0));
        assert_ne!(Term::Constant(1.0), Term::Variable(Variable::Time));
        assert_eq!(
            Term::Operation("f", Vec::new()),
            Term::Operation("f", Vec::new())
        );
        assert_ne!(
            Term::Operation("f", Vec::new()),
            Term::Operation("g", Vec::new())
        );
        assert_ne!(
            Term::Operation("+", vec![Term::Constant(1.0)]),
            Term::Operation("+", vec![Term::Constant(1.0), Term::Constant(2.0)])
        );
        let base_table = Term::Table {
            input: Box::new(Term::Constant(0.0)),
            abscissae: vec![0.0, 1.0],
            ordinates: vec![0.0, 1.0],
            rounding: f64::NAN,
        };
        assert_ne!(base_table, base_table.clone());
    }

    #[test]
    fn rounded_table_recovers_after_nan_trial_input() {
        let expression: Expression = "rounded_table(sqrt(v(a)), 0.1, 0, 0, 1, 1)"
            .parse()
            .expect("valid");
        let mut workspace = Workspace::new(&expression);
        let mut gradient = [0.0];

        let first = expression.evaluate(&[-1.0], &mut workspace, &mut gradient);
        assert!(first.is_nan());
        assert!(gradient[0].is_nan());
        assert!(!workspace.limited());

        // Second evaluation after a NaN input must not underflow in `Points::limit`.
        let second = expression.evaluate(&[1.0], &mut workspace, &mut gradient);
        assert!(close(second, 0.975));
        assert!(close(gradient[0], 0.25));
        assert!(!workspace.limited());

        let third = expression.evaluate(&[-4.0], &mut workspace, &mut gradient);
        assert!(third.is_nan());
        assert!(!workspace.limited());

        let fourth = expression.evaluate(&[0.25], &mut workspace, &mut gradient);
        assert!(close(fourth, 0.5));
        assert!(close(gradient[0], 1.0));
    }

    #[test]
    fn value_matches_unlimited_evaluate_for_inline_and_heap_tapes() {
        let short = "limit(v(a), -1, 1) + if(v(b) > 0, rounded_table(v(a), 0.1, 0, 1, 2, 3), 2)";
        let parsed: Expression = short.parse().expect("valid");
        let (expected, _) = evaluate(short, &[0.5, 1.0]);
        assert_eq!(parsed.value(&[0.5, 1.0]).to_bits(), expected.to_bits());
        let (other, _) = evaluate(short, &[0.5, -1.0]);
        assert_eq!(parsed.value(&[0.5, -1.0]).to_bits(), other.to_bits());

        // Exceeds the 32-instruction inline stack buffer in `tape::value`.
        let long = (1..=40)
            .map(|index| format!("sin(v(a) + {index})"))
            .collect::<Vec<_>>()
            .join(" + ");
        let parsed_long: Expression = long.parse().expect("valid");
        let (expected_long, _) = evaluate(&long, &[0.25]);
        assert_eq!(
            parsed_long.value(&[0.25]).to_bits(),
            expected_long.to_bits()
        );
    }
}
