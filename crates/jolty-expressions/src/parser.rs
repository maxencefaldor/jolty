//! The grammar of expressions, read into a tree whose constant parts are folded.

use crate::SyntaxError;
use crate::Variable;
use crate::lexer::{Kind, Symbol, Token, tokenize};
use crate::operations::{Binary, Unary};

/// An expression's tree. Every part without a variable is folded into a constant.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Node {
    Constant(f64),
    Variable(usize),
    Unary(Unary, Box<Node>),
    Binary(Binary, Box<Node>, Box<Node>),
    /// `limit(x, a, b)`: `x` held between `a` and `b`, in either order.
    Limit(Box<[Node; 3]>),
    /// `c ? a : b`.
    Conditional(Box<[Node; 3]>),
    /// `table(x, x1, y1, …)`: the points' piecewise-linear interpolation at `x`.
    Table(Box<Node>, Vec<(f64, f64)>),
}

impl Node {
    /// The node's value, if it is a constant.
    fn constant(&self) -> Option<f64> {
        match self {
            Self::Constant(value) => Some(*value),
            _ => None,
        }
    }
}

/// The tree of `text`, and the variables it reads, in the order first read.
pub(crate) fn parse(text: &str) -> Result<(Node, Vec<Variable>), SyntaxError> {
    let mut parser = Parser {
        tokens: tokenize(text)?,
        position: 0,
        variables: Vec::new(),
    };
    let node = parser.expression()?;
    let token = parser.peek();
    if token.kind != Kind::End {
        return Err(SyntaxError::new(
            format!("expected an operator, not {}", describe(&token.kind)),
            token.column,
        ));
    }
    Ok((node, parser.variables))
}

struct Parser {
    tokens: Vec<Token>,
    position: usize,
    variables: Vec<Variable>,
}

/// The binary operators at each level of precedence, loosest first, above the conditional.
const LEVELS: [&[(Symbol, Binary)]; 6] = [
    &[(Symbol::Or, Binary::Or)],
    &[(Symbol::And, Binary::And)],
    &[
        (Symbol::Equal, Binary::Equal),
        (Symbol::NotEqual, Binary::NotEqual),
    ],
    &[
        (Symbol::Less, Binary::Less),
        (Symbol::LessEqual, Binary::LessEqual),
        (Symbol::Greater, Binary::Greater),
        (Symbol::GreaterEqual, Binary::GreaterEqual),
    ],
    &[
        (Symbol::Plus, Binary::Add),
        (Symbol::Minus, Binary::Subtract),
    ],
    &[
        (Symbol::Times, Binary::Multiply),
        (Symbol::Divide, Binary::Divide),
    ],
];

impl Parser {
    fn peek(&self) -> &Token {
        &self.tokens[self.position]
    }

    fn next(&mut self) -> Token {
        let token = self.tokens[self.position].clone();
        if token.kind != Kind::End {
            self.position += 1;
        }
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
            let token = self.peek();
            Err(SyntaxError::new(
                format!(
                    "expected `{}`, not {}",
                    symbol.text(),
                    describe(&token.kind)
                ),
                token.column,
            ))
        }
    }

    /// `condition ? then : otherwise`, which groups to the right, or a looser operand.
    fn expression(&mut self) -> Result<Node, SyntaxError> {
        let condition = self.level(0)?;
        if !self.accept(Symbol::Question) {
            return Ok(condition);
        }
        let then = self.expression()?;
        self.expect(Symbol::Colon)?;
        let otherwise = self.expression()?;
        Ok(conditional([condition, then, otherwise]))
    }

    /// The operands of the binary operators at `level` and tighter, grouped to the left.
    fn level(&mut self, level: usize) -> Result<Node, SyntaxError> {
        let Some(operators) = LEVELS.get(level) else {
            return self.unary();
        };
        let mut node = self.level(level + 1)?;
        while let Some(&(_, operation)) = operators
            .iter()
            .find(|(symbol, _)| self.peek().kind == Kind::Symbol(*symbol))
        {
            self.position += 1;
            let right = self.level(level + 1)?;
            node = binary(operation, node, right);
        }
        Ok(node)
    }

    /// A signed or negated operand.
    fn unary(&mut self) -> Result<Node, SyntaxError> {
        if self.accept(Symbol::Minus) {
            Ok(unary(Unary::Negate, self.unary()?))
        } else if self.accept(Symbol::Plus) {
            self.unary()
        } else if self.accept(Symbol::Not) {
            Ok(unary(Unary::Not, self.unary()?))
        } else {
            self.power()
        }
    }

    /// `base ^ exponent`, which groups to the right and binds tighter than a sign.
    fn power(&mut self) -> Result<Node, SyntaxError> {
        let base = self.primary()?;
        if self.accept(Symbol::Power) {
            let exponent = self.unary()?;
            Ok(binary(Binary::Power, base, exponent))
        } else {
            Ok(base)
        }
    }

    fn primary(&mut self) -> Result<Node, SyntaxError> {
        let token = self.next();
        match token.kind {
            Kind::Number(value) => Ok(Node::Constant(value)),
            Kind::Symbol(Symbol::Open) => {
                let node = self.expression()?;
                self.expect(Symbol::Close)?;
                Ok(node)
            }
            Kind::Name(name) if self.accept(Symbol::Open) => self.call(&name, token.column),
            Kind::Name(name) => match name.as_str() {
                "time" => Ok(self.variable(Variable::Time)),
                "pi" => Ok(Node::Constant(std::f64::consts::PI)),
                _ => Err(SyntaxError::new(
                    format!("unknown name `{name}`"),
                    token.column,
                )),
            },
            kind => Err(SyntaxError::new(
                format!("expected an operand, not {}", describe(&kind)),
                token.column,
            )),
        }
    }

    /// The call of `name`, whose opening parenthesis is read.
    fn call(&mut self, name: &str, column: usize) -> Result<Node, SyntaxError> {
        match name {
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
            _ => {}
        }
        let mut arguments = vec![self.expression()?];
        while self.accept(Symbol::Comma) {
            arguments.push(self.expression()?);
        }
        self.expect(Symbol::Close)?;
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
            table(arguments, column)
        } else {
            Err(SyntaxError::new(
                format!("unknown function `{name}`"),
                column,
            ))
        }
    }

    /// A terminal's or a port's name.
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

    /// The node reading `variable`, numbered when first read.
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

/// The `N` arguments of a call of `function`, at `column`.
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

/// `table(x, x1, y1, …)`, whose points are constants in pairs, ascending.
fn table(arguments: Vec<Node>, column: usize) -> Result<Node, SyntaxError> {
    let invalid = || {
        SyntaxError::new(
            "`table` takes an input then constant points, in pairs, with ascending abscissae"
                .to_owned(),
            column,
        )
    };
    let mut arguments = arguments.into_iter();
    let x = arguments.next().ok_or_else(invalid)?;
    let numbers: Vec<f64> = arguments
        .map(|node| node.constant().ok_or_else(invalid))
        .collect::<Result<_, _>>()?;
    let points: Vec<(f64, f64)> = numbers
        .chunks(2)
        .map(|pair| match *pair {
            [abscissa, ordinate] => Ok((abscissa, ordinate)),
            _ => Err(invalid()),
        })
        .collect::<Result<_, _>>()?;
    if points.is_empty() || points.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
        return Err(invalid());
    }
    Ok(match x.constant() {
        Some(x) => Node::Constant(crate::tape::interpolate(&points, x).0),
        None => Node::Table(Box::new(x), points),
    })
}

/// A token as messages name it.
fn describe(kind: &Kind) -> String {
    match kind {
        Kind::Number(value) => format!("the number `{value}`"),
        Kind::Name(name) => format!("`{name}`"),
        Kind::Symbol(symbol) => format!("`{}`", symbol.text()),
        Kind::End => "the end".to_owned(),
    }
}
