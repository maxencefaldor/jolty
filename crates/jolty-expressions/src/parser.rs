//! The grammar of expressions, read into a tree whose constant parts are folded.

use std::cmp::Ordering;

use crate::SyntaxError;
use crate::Variable;
use crate::lexer::{Kind, Symbol, Token, tokenize};
use crate::operations::{Binary, Unary};
use crate::tape::{Beyond, Points};

/// Maximum nesting depth accepted by the expression parser.
///
/// Origin: matches `_MAX_CORE_DEPTH` in `src/jolty/spice/expressions.py` (`256`).
/// The parser uses an explicit operator stack rather than call-stack recursion, so any
/// nesting up to `MAX_DEPTH` runs in constant call-stack space across Linux, macOS, and
/// Windows debug and release threads while rejecting pathological nesting deterministically.
pub(crate) const MAX_DEPTH: usize = 256;

/// An expression's tree. Every part without a variable is folded into a constant.
#[derive(Debug)]
pub(crate) enum Node {
    Constant(f64),
    Variable(usize),
    Unary(Unary, Box<Node>),
    Binary(Binary, Box<Node>, Box<Node>),
    /// Clamping operation `limit(x, a, b)` bounding `x` between `a` and `b` in either order.
    Limit(Box<[Node; 3]>),
    /// Ternary conditional `c ? a : b` (or `if(c, a, b)`).
    Conditional(Box<[Node; 3]>),
    /// Piecewise-linear table interpolation (`table`, `pwl`, or `rounded_table`) evaluated at
    /// `x`.
    Table(Box<(Node, Points)>),
}

impl Drop for Node {
    fn drop(&mut self) {
        if matches!(self, Self::Constant(_) | Self::Variable(_)) {
            return;
        }
        let mut stack = Vec::new();
        detach_children(self, &mut stack);
        while let Some(mut node) = stack.pop() {
            detach_children(&mut node, &mut stack);
        }
    }
}

fn push_if_nested(slot: &mut Node, stack: &mut Vec<Node>) {
    if !matches!(slot, Node::Constant(_) | Node::Variable(_)) {
        stack.push(std::mem::replace(slot, Node::Constant(0.0)));
    }
}

fn detach_children(node: &mut Node, stack: &mut Vec<Node>) {
    match node {
        Node::Constant(_) | Node::Variable(_) => {}
        Node::Unary(_, operand) => push_if_nested(operand, stack),
        Node::Binary(_, left, right) => {
            push_if_nested(left, stack);
            push_if_nested(right, stack);
        }
        Node::Limit(arguments) | Node::Conditional(arguments) => {
            for argument in arguments.as_mut() {
                push_if_nested(argument, stack);
            }
        }
        Node::Table(table) => push_if_nested(&mut table.0, stack),
    }
}

impl Node {
    /// Returns the numeric value if this node is a constant.
    fn constant(&self) -> Option<f64> {
        match self {
            Self::Constant(value) => Some(*value),
            _ => None,
        }
    }
}

/// Parses `text` into a constant-folded expression tree and ordered variable list.
pub(crate) fn parse(text: &str) -> Result<(Node, Vec<Variable>), SyntaxError> {
    let mut parser = Parser {
        tokens: tokenize(text)?,
        position: 0,
        variables: Vec::new(),
        depth: 0,
    };
    let node = parser.parse_all()?;
    Ok((node, parser.variables))
}

struct Parser {
    tokens: Vec<Token>,
    position: usize,
    variables: Vec<Variable>,
    depth: usize,
}

/// An open construct on the explicit shunting-yard stack, holding its already-parsed
/// left operand(s) so reductions require no separate value stack or fallible pops.
enum Pending {
    /// A prefix unary operator (`-`, `!`, or `+` as `None`); binds tighter than `*`/`/`
    /// and looser than `^`.
    Unary(Option<Unary>),
    /// A left-associative binary operator (`||` through `/`) at precedence `1..=6`.
    Binary(u8, Binary, Node),
    /// Right-associative exponentiation `^` (binds tighter than unary prefix signs).
    Power(Node),
    /// An open parenthesized group `(`.
    Group,
    /// An open function call `name(`, accumulating its already-completed arguments.
    Call {
        name: String,
        column: usize,
        arguments: Vec<Node>,
    },
    /// The `?` of `condition ? then : otherwise`, awaiting its matching `:`.
    Question(Node),
    /// The `:` of `condition ? then : otherwise`, awaiting its `otherwise` branch.
    Colon { condition: Node, then: Node },
}

/// Transition returned by [`Parser::step_operator`].
enum Step {
    NextOperand,
    Continue(Node),
    Done(Node),
}

/// Returns the precedence (`1..=6`, loosest first) and binary operation for left-associative
/// operator tokens.
const fn binary_operator(kind: &Kind) -> Option<(u8, Binary)> {
    let Kind::Symbol(symbol) = kind else {
        return None;
    };
    match symbol {
        Symbol::Or => Some((1, Binary::Or)),
        Symbol::And => Some((2, Binary::And)),
        Symbol::Equal => Some((3, Binary::Equal)),
        Symbol::NotEqual => Some((3, Binary::NotEqual)),
        Symbol::Less => Some((4, Binary::Less)),
        Symbol::LessEqual => Some((4, Binary::LessEqual)),
        Symbol::Greater => Some((4, Binary::Greater)),
        Symbol::GreaterEqual => Some((4, Binary::GreaterEqual)),
        Symbol::Plus => Some((5, Binary::Add)),
        Symbol::Minus => Some((5, Binary::Subtract)),
        Symbol::Times => Some((6, Binary::Multiply)),
        Symbol::Divide => Some((6, Binary::Divide)),
        _ => None,
    }
}

/// Reduces pending operators whose precedence is at least `min_precedence` against `current`.
///
/// Unary (`7`) and power (`8`) always reduce here because `min_precedence <= 6`.
/// `Colon` (precedence `0`) reduces only when `min_precedence == 0`, preserving right
/// associativity of chained `a ? b : c ? d : e`.
fn reduce(
    stack: &mut Vec<Pending>,
    mut current: Node,
    depth: &mut usize,
    min_precedence: u8,
) -> Node {
    while let Some(pending) = stack.pop() {
        match pending {
            Pending::Unary(operation) => {
                *depth -= 1;
                if let Some(operation) = operation {
                    current = unary(operation, current);
                }
            }
            Pending::Power(base) => {
                *depth -= 1;
                current = binary(Binary::Power, base, current);
            }
            Pending::Binary(precedence, operation, left) if precedence >= min_precedence => {
                current = binary(operation, left, current);
            }
            Pending::Colon { condition, then } if min_precedence == 0 => {
                *depth -= 1;
                current = conditional([condition, then, current]);
            }
            barrier => {
                stack.push(barrier);
                break;
            }
        }
    }
    current
}

fn depth_error(column: usize) -> SyntaxError {
    SyntaxError::new(
        format!("expression nests more than {MAX_DEPTH} levels deep"),
        column,
    )
}

fn expected_symbol(symbol: Symbol, token: &Token) -> SyntaxError {
    SyntaxError::new(
        format!(
            "expected `{}`, not {}",
            symbol.text(),
            describe(&token.kind)
        ),
        token.column,
    )
}

fn expected_operator(token: &Token) -> SyntaxError {
    SyntaxError::new(
        format!("expected an operator, not {}", describe(&token.kind)),
        token.column,
    )
}

impl Parser {
    fn peek(&self) -> &Token {
        &self.tokens[self.position]
    }

    fn next(&mut self) -> Token {
        let slot = &mut self.tokens[self.position];
        if slot.kind == Kind::End {
            return Token {
                kind: Kind::End,
                column: slot.column,
            };
        }
        let token = std::mem::replace(
            slot,
            Token {
                kind: Kind::End,
                column: 0,
            },
        );
        self.position += 1;
        token
    }

    /// Consumes `symbol` if it comes next.
    fn accept(&mut self, symbol: Symbol) -> bool {
        let found = self.peek().kind == Kind::Symbol(symbol);
        if found {
            self.position += 1;
        }
        found
    }

    fn expect(&mut self, symbol: Symbol) -> Result<(), SyntaxError> {
        if self.accept(symbol) {
            Ok(())
        } else {
            Err(expected_symbol(symbol, self.peek()))
        }
    }

    fn push_nested(
        &mut self,
        stack: &mut Vec<Pending>,
        pending: Pending,
        column: usize,
    ) -> Result<(), SyntaxError> {
        if self.depth >= MAX_DEPTH {
            return Err(depth_error(column));
        }
        self.depth += 1;
        stack.push(pending);
        Ok(())
    }

    /// Parses the complete token stream iteratively using an explicit operator stack.
    fn parse_all(&mut self) -> Result<Node, SyntaxError> {
        let mut stack: Vec<Pending> = Vec::new();
        loop {
            let mut current = self.read_operand(&mut stack)?;
            loop {
                match self.step_operator(&mut stack, current)? {
                    Step::NextOperand => break,
                    Step::Continue(node) => current = node,
                    Step::Done(node) => return Ok(node),
                }
            }
        }
    }

    /// Processes the next operator or delimiter after `current`.
    fn step_operator(
        &mut self,
        stack: &mut Vec<Pending>,
        mut current: Node,
    ) -> Result<Step, SyntaxError> {
        if let Some((precedence, operation)) = binary_operator(&self.peek().kind) {
            current = reduce(stack, current, &mut self.depth, precedence);
            self.position += 1;
            stack.push(Pending::Binary(precedence, operation, current));
            return Ok(Step::NextOperand);
        }
        if let Kind::Symbol(symbol @ (Symbol::Power | Symbol::Question)) = self.peek().kind {
            let column = self.peek().column;
            let pending = if symbol == Symbol::Power {
                Pending::Power(current)
            } else {
                Pending::Question(reduce(stack, current, &mut self.depth, 1))
            };
            self.push_nested(stack, pending, column)?;
            self.position += 1;
            return Ok(Step::NextOperand);
        }
        current = reduce(stack, current, &mut self.depth, 0);
        if let Some(Pending::Call { arguments, .. }) = stack.last_mut()
            && self.accept(Symbol::Comma)
        {
            arguments.push(current);
            return Ok(Step::NextOperand);
        }
        match stack.pop() {
            Some(Pending::Question(condition)) if self.accept(Symbol::Colon) => {
                stack.push(Pending::Colon {
                    condition,
                    then: current,
                });
                Ok(Step::NextOperand)
            }
            Some(Pending::Question(_)) => Err(expected_symbol(Symbol::Colon, self.peek())),
            Some(Pending::Group) if self.accept(Symbol::Close) => {
                self.depth -= 1;
                Ok(Step::Continue(current))
            }
            Some(Pending::Call {
                name,
                column,
                mut arguments,
            }) if self.accept(Symbol::Close) => {
                self.depth -= 1;
                arguments.push(current);
                Ok(Step::Continue(finish_call(&name, arguments, column)?))
            }
            Some(Pending::Group | Pending::Call { .. }) => {
                Err(expected_symbol(Symbol::Close, self.peek()))
            }
            _ if self.peek().kind == Kind::End => Ok(Step::Done(current)),
            _ => Err(expected_operator(self.peek())),
        }
    }

    /// Consumes any leading prefix operators and opening delimiters, pushing them onto
    /// `stack`, then returns the primary operand node.
    fn read_operand(&mut self, stack: &mut Vec<Pending>) -> Result<Node, SyntaxError> {
        loop {
            let token = self.next();
            match token.kind {
                Kind::Symbol(Symbol::Minus) => {
                    self.push_nested(stack, Pending::Unary(Some(Unary::Negate)), token.column)?;
                }
                Kind::Symbol(Symbol::Plus) => {
                    self.push_nested(stack, Pending::Unary(None), token.column)?;
                }
                Kind::Symbol(Symbol::Not) => {
                    self.push_nested(stack, Pending::Unary(Some(Unary::Not)), token.column)?;
                }
                Kind::Symbol(Symbol::Open) => {
                    self.push_nested(stack, Pending::Group, token.column)?;
                }
                Kind::Number(value) => return Ok(Node::Constant(value)),
                Kind::Name(name) if self.accept(Symbol::Open) => match name.as_str() {
                    "v" => {
                        let positive = self.name()?;
                        let negative = if self.accept(Symbol::Comma) {
                            Some(self.name()?)
                        } else {
                            None
                        };
                        self.expect(Symbol::Close)?;
                        return Ok(self.variable(Variable::Voltage(positive, negative)));
                    }
                    "i" => {
                        let port = self.name()?;
                        self.expect(Symbol::Close)?;
                        return Ok(self.variable(Variable::Current(port)));
                    }
                    "p" => {
                        let parameter = self.name()?;
                        self.expect(Symbol::Close)?;
                        return Ok(self.variable(Variable::Parameter(parameter)));
                    }
                    _ => {
                        self.push_nested(
                            stack,
                            Pending::Call {
                                name,
                                column: token.column,
                                arguments: Vec::new(),
                            },
                            token.column,
                        )?;
                    }
                },
                Kind::Name(name) => match name.as_str() {
                    "time" => return Ok(self.variable(Variable::Time)),
                    "pi" => return Ok(Node::Constant(std::f64::consts::PI)),
                    _ => {
                        return Err(SyntaxError::new(
                            format!("unknown name `{name}`"),
                            token.column,
                        ));
                    }
                },
                kind => {
                    return Err(SyntaxError::new(
                        format!("expected an operand, not {}", describe(&kind)),
                        token.column,
                    ));
                }
            }
        }
    }

    /// Parses a terminal or sensing-port identifier token.
    fn name(&mut self) -> Result<String, SyntaxError> {
        let token = self.next();
        match token.kind {
            Kind::Name(name) => Ok(name),
            kind => Err(SyntaxError::new(
                format!("expected a name, not {}", describe(&kind)),
                token.column,
            )),
        }
    }

    /// Returns the AST node for `variable`, interning its index on first occurrence.
    fn variable(&mut self, variable: Variable) -> Node {
        let index = self
            .variables
            .iter()
            .position(|known| *known == variable)
            .unwrap_or_else(|| {
                self.variables.push(variable);
                self.variables.len() - 1
            });
        Node::Variable(index)
    }
}

/// Builds the node for a completed function call `name(arguments...)` at `column`.
fn finish_call(name: &str, arguments: Vec<Node>, column: usize) -> Result<Node, SyntaxError> {
    if let Some(operation) = Unary::named(name) {
        let [x] = take(arguments, name, column)?;
        Ok(unary(operation, x))
    } else if let Some(operation) = Binary::named(name) {
        let [a, b] = take(arguments, name, column)?;
        Ok(binary(operation, a, b))
    } else if name == "limit" {
        Ok(limit(take(arguments, name, column)?))
    } else if name == "if" {
        Ok(conditional(take(arguments, name, column)?))
    } else if name == "table" {
        table(arguments, Beyond::Hold, false, column)
    } else if name == "pwl" {
        table(arguments, Beyond::Extend, false, column)
    } else if name == "rounded_table" {
        table(arguments, Beyond::Hold, true, column)
    } else {
        Err(SyntaxError::new(
            format!("unknown function `{name}`"),
            column,
        ))
    }
}

/// Validates that `arguments` contains exactly `N` expressions for `function` at `column`.
fn take<const N: usize>(
    arguments: Vec<Node>,
    function: &str,
    column: usize,
) -> Result<[Node; N], SyntaxError> {
    arguments.try_into().map_err(|arguments: Vec<Node>| {
        let plural = if N == 1 { "" } else { "s" };
        SyntaxError::new(
            format!(
                "`{function}` takes {N} argument{plural}, not {}",
                arguments.len()
            ),
            column,
        )
    })
}

fn unary(operation: Unary, x: Node) -> Node {
    match x.constant() {
        Some(x) => Node::Constant(operation.value(x)),
        None => Node::Unary(operation, Box::new(x)),
    }
}

fn binary(operation: Binary, a: Node, b: Node) -> Node {
    match (a.constant(), b.constant()) {
        (Some(a), Some(b)) => Node::Constant(operation.value(a, b)),
        _ => Node::Binary(operation, Box::new(a), Box::new(b)),
    }
}

fn limit(arguments: [Node; 3]) -> Node {
    match arguments.each_ref().map(Node::constant) {
        [Some(x), Some(a), Some(b)] => Node::Constant(crate::tape::limit(x, a, b).0),
        _ => Node::Limit(Box::new(arguments)),
    }
}

fn conditional(arguments: [Node; 3]) -> Node {
    match arguments[0].constant() {
        Some(condition) => {
            let [_, then, otherwise] = arguments;
            if condition == 0.0 { otherwise } else { then }
        }
        None => Node::Conditional(Box::new(arguments)),
    }
}

/// Builds a `table`, `pwl`, or `rounded_table` node from constant breakpoint pairs with
/// strictly ascending abscissae and (when `rounded` is set) a rounding fraction in `[0, 0.5]`.
fn table(
    arguments: Vec<Node>,
    beyond: Beyond,
    rounded: bool,
    column: usize,
) -> Result<Node, SyntaxError> {
    let function = match (beyond, rounded) {
        (_, true) => "rounded_table",
        (Beyond::Hold, false) => "table",
        (Beyond::Extend, false) => "pwl",
    };
    let invalid = || {
        let rounding = if rounded {
            " a rounding from 0 to 0.5,"
        } else {
            ""
        };
        SyntaxError::new(
            format!(
                "`{function}` takes an input,{rounding} then constant points, in pairs, \
                 with ascending abscissae"
            ),
            column,
        )
    };
    let mut arguments = arguments.into_iter();
    let x = arguments.next().ok_or_else(invalid)?;
    let numbers: Vec<f64> = arguments
        .map(|node| node.constant().ok_or_else(invalid))
        .collect::<Result<_, _>>()?;
    let (rounding, numbers) = if rounded {
        match numbers.split_first() {
            Some((&rounding, rest)) if (0.0..=0.5).contains(&rounding) => (rounding, rest),
            _ => return Err(invalid()),
        }
    } else {
        (0.0, numbers.as_slice())
    };
    let points: Vec<(f64, f64)> = numbers
        .chunks(2)
        .map(|pair| match *pair {
            [abscissa, ordinate] if abscissa.is_finite() && ordinate.is_finite() => {
                Ok((abscissa, ordinate))
            }
            _ => Err(invalid()),
        })
        .collect::<Result<_, _>>()?;
    if points.is_empty()
        || points
            .windows(2)
            .any(|pair| pair[0].0.partial_cmp(&pair[1].0) != Some(Ordering::Less))
    {
        return Err(invalid());
    }
    let points = Points::new(&points, beyond, rounding);
    Ok(match x.constant() {
        Some(x) => Node::Constant(points.interpolate(x).0),
        None => Node::Table(Box::new((x, points))),
    })
}

/// Formats a token kind for syntax error messages.
fn describe(kind: &Kind) -> String {
    match kind {
        Kind::Number(value) => format!("the number `{value}`"),
        Kind::Name(name) => format!("`{name}`"),
        Kind::Symbol(symbol) => format!("`{}`", symbol.text()),
        Kind::End => "the end".to_owned(),
    }
}
