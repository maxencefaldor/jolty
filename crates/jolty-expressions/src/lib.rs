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

use std::fmt;
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
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    /// A number.
    Constant(f64),
    /// A variable the expression reads.
    Variable(Variable),
    /// An operation on its operands, named as the language names it: an operator, such as
    /// `+`, `^` or `<=`, of two operands, `-` or `!` of one, `if` of a condition and the
    /// two values it chooses between, or a function, such as `sqrt`, `min` or `limit`.
    Operation(&'static str, Vec<Term>),
    /// The piecewise-linear interpolation of points at `input`, as `table`, `pwl` and
    /// `rounded_table` read it: through the points given, with one more before the first
    /// and one after the last, which hold the ends or extend them. Each corner, the ends'
    /// included, holds the inputs between the midpoints of the segments beside it, and is
    /// rounded there by a parabola over `rounding` times the shorter of them; below the
    /// first segment's midpoint and above the last's, the input follows that segment.
    Table {
        /// The interpolation's input.
        input: Box<Term>,
        /// The points' abscissae, ascending.
        abscissae: Vec<f64>,
        /// The points' ordinates.
        ordinates: Vec<f64>,
        /// The fraction of the shorter segment beside each corner its parabola spans.
        rounding: f64,
    },
}

/// A parsed expression, ready to evaluate.
#[derive(Debug, Clone, PartialEq)]
pub struct Expression {
    text: String,
    variables: Vec<Variable>,
    tape: Vec<Instruction>,
}

impl Expression {
    /// The variables the expression reads, in the order it first reads them, which is the
    /// order [`Expression::evaluate`] takes their values in.
    #[must_use]
    pub fn variables(&self) -> &[Variable] {
        &self.variables
    }

    /// The expression's value with its `variables` at the values given, in the order of
    /// [`Expression::variables`], and its derivative with respect to each, written into
    /// `gradient`.
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

    /// The expression as a tree of terms, its constant parts folded: what another
    /// simulator's language needs to write it in its own.
    #[must_use]
    pub fn tree(&self) -> Term {
        let mut terms: Vec<Term> = Vec::with_capacity(self.tape.len());
        for instruction in &self.tape {
            let term = match instruction {
                Instruction::Constant(value) => Term::Constant(*value),
                Instruction::Variable(index) => Term::Variable(self.variables[*index].clone()),
                Instruction::Unary(operation, x) => {
                    Term::Operation(operation.name(), vec![terms[*x].clone()])
                }
                Instruction::Binary(operation, a, b) => {
                    Term::Operation(operation.name(), vec![terms[*a].clone(), terms[*b].clone()])
                }
                Instruction::Limit(operands) => Term::Operation(
                    "limit",
                    operands.iter().map(|&index| terms[index].clone()).collect(),
                ),
                Instruction::Conditional(operands) => Term::Operation(
                    "if",
                    operands.iter().map(|&index| terms[index].clone()).collect(),
                ),
                Instruction::Table(x, points) => Term::Table {
                    input: Box::new(terms[*x].clone()),
                    abscissae: points.abscissae().to_vec(),
                    ordinates: points.ordinates().to_vec(),
                    rounding: points.rounding(),
                },
            };
            terms.push(term);
        }
        terms.pop().unwrap_or(Term::Constant(f64::NAN))
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
    /// The instruction that makes it, which names it within its expression.
    instruction: usize,
    /// What it reads: a comparison's left operand less its right, or the argument.
    input: f64,
    /// What it chose: the comparison's truth, or the function's value.
    chosen: f64,
    kind: tape::Kind,
}

impl Choice {
    /// Where between two evaluations of one expression its value first jumps: the
    /// fraction of the way from the first evaluation's variables to the second's at which
    /// the earliest of the choices both took, and that changed, changes, interpolating its
    /// input linearly to the threshold between them. `None` if no choice both took
    /// changed.
    ///
    /// A comparison, a step and a sign change where their input crosses zero; a floor or a
    /// ceiling where its argument crosses the integer next to the first value towards the
    /// second.
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

    /// The fraction of the way from `self` to `later`, a change of the same choice, at
    /// which its input crosses the threshold between the two.
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
    limited: bool,
}

impl Workspace {
    /// A workspace for `expression`.
    #[must_use]
    pub fn new(expression: &Expression) -> Self {
        let length = expression.tape.len();
        Self {
            values: vec![0.0; length],
            adjoints: vec![0.0; length],
            inputs: vec![None; length],
            slopes: vec![0.0; length],
            limited: false,
        }
    }

    /// Forgets the last inputs of the rounded tables, so the next evaluation limits no
    /// step: as when an iteration starts afresh.
    pub fn forget(&mut self) {
        self.inputs.fill(None);
    }

    /// Whether the last evaluation limited a rounded table's input, so its value is not
    /// yet the expression's at the variables given: the step of an iteration that has
    /// not converged.
    #[must_use]
    pub fn limited(&self) -> bool {
        self.limited
    }
}

/// Why a text is not an expression, and the column, counted from 1, where it goes wrong.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message} at column {column}")]
pub struct SyntaxError {
    /// What is wrong.
    pub message: String,
    /// The column of the token where it goes wrong.
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

    /// The value and gradient of `text` at `values`.
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

    /// The choices `expression` turns on at `values`.
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
        /// The gradient matches central differences of the value, wherever the expression
        /// is smooth.
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
            for (index, derivative) in gradient.iter().enumerate() {
                let step = 1e-6;
                let mut above = values;
                above[index] += step;
                let mut below = values;
                below[index] -= step;
                let numeric = (evaluate(text, &above).0 - evaluate(text, &below).0) / (2.0 * step);
                prop_assert!((numeric - derivative).abs() < 1e-5 * derivative.abs().max(1.0),
                    "variable {index}: analytic {derivative}, numeric {numeric}");
            }
        }
    }
}
