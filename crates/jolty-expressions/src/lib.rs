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
