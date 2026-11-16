//! A recursive-descent parser of the CMC's subset of Verilog-A: one module with its ports,
//! nodes, branches, parameters, variables and analog functions, and its analog block.

use std::collections::BTreeMap;

use crate::Error;
use crate::ast::{Expression, Function, Module, Parameter, Range, Statement, Type};
use crate::lexer::{Spanned, Token};

/// The disciplines the compiler knows, as the standard `disciplines.vams` declares them.
const DISCIPLINES: [&str; 2] = ["electrical", "thermal"];

/// The binary operators, by precedence from the loosest, as Verilog defines it.
const BINARY: [&[&str]; 10] = [
    &["||"],
    &["&&"],
    &["|"],
    &["^"],
    &["&"],
    &["==", "!="],
    &["<", "<=", ">", ">="],
    &["+", "-"],
    &["*", "/", "%"],
    &["**"],
];

/// The module of `tokens`.
///
/// # Errors
///
/// Returns an [`Error`] at the first token that does not fit the grammar.
pub(crate) fn parse(tokens: &[Spanned]) -> Result<Module, Error> {
    let mut parser = Parser { tokens, at: 0 };
    let module = parser.module()?;
    match parser.tokens.get(parser.at) {
        None => Ok(module),
        Some(_) => Err(parser.error("text after `endmodule`")),
    }
}

/// The tokens and the position of the next one.
struct Parser<'a> {
    tokens: &'a [Spanned],
    at: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.at).map(|spanned| &spanned.token)
    }

    fn peek_at(&self, ahead: usize) -> Option<&Token> {
        self.tokens
            .get(self.at + ahead)
            .map(|spanned| &spanned.token)
    }

    fn line(&self) -> usize {
        self.tokens
            .get(self.at)
            .or_else(|| self.tokens.last())
            .map_or(0, |spanned| spanned.line)
    }

    fn error(&self, message: &str) -> Error {
        let found = match self.peek() {
            Some(Token::Identifier(name) | Token::System(name)) => format!("`{name}`"),
            Some(Token::Symbol(symbol)) => format!("`{symbol}`"),
            Some(Token::Real(value)) => format!("{value:?}"),
            Some(Token::Integer(value)) => value.to_string(),
            Some(Token::Text(text)) => format!("{text:?}"),
            None => "the end".to_owned(),
        };
        Error::at(self.line(), format!("{message}, not {found}"))
    }

    fn next(&mut self) -> Option<Token> {
        let token = self.peek().cloned();
        self.at += 1;
        token
    }

    /// Whether the next token is the symbol `symbol`, which it then consumes.
    fn eat(&mut self, symbol: &str) -> bool {
        if matches!(self.peek(), Some(Token::Symbol(found)) if *found == symbol) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    /// Whether the next token is the keyword `keyword`, which it then consumes.
    fn keyword(&mut self, keyword: &str) -> bool {
        if matches!(self.peek(), Some(Token::Identifier(found)) if found == keyword) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, symbol: &str) -> Result<(), Error> {
        if self.eat(symbol) {
            Ok(())
        } else {
            Err(self.error(&format!("expected `{symbol}`")))
        }
    }

    fn expect_keyword(&mut self, keyword: &str) -> Result<(), Error> {
        if self.keyword(keyword) {
            Ok(())
        } else {
            Err(self.error(&format!("expected `{keyword}`")))
        }
    }

    fn name(&mut self) -> Result<String, Error> {
        match self.peek() {
            Some(Token::Identifier(name)) => {
                let name = name.clone();
                self.at += 1;
                Ok(name)
            }
            _ => Err(self.error("expected a name")),
        }
    }

    /// Names separated by commas, up to the `;` that ends them.
    fn names(&mut self) -> Result<Vec<String>, Error> {
        let mut names = vec![self.name()?];
        while self.eat(",") {
            names.push(self.name()?);
        }
        self.expect(";")?;
        Ok(names)
    }

    fn module(&mut self) -> Result<Module, Error> {
        self.expect_keyword("module")?;
        let name = self.name()?;
        let mut ports = Vec::new();
        if self.eat("(") && !self.eat(")") {
            ports.push(self.name()?);
            while self.eat(",") {
                ports.push(self.name()?);
            }
            self.expect(")")?;
        }
        self.expect(";")?;
        let mut module = Module {
            name,
            ports,
            nodes: Vec::new(),
            branches: Vec::new(),
            parameters: Vec::new(),
            aliases: Vec::new(),
            variables: Vec::new(),
            outputs: Vec::new(),
            functions: Vec::new(),
            analog: Statement::Block(Vec::new()),
        };
        let mut analog = None;
        while !self.keyword("endmodule") {
            let attributes = self.attributes()?;
            let keyword = self.name()?;
            match keyword.as_str() {
                "inout" | "input" | "output" => {
                    self.names()?;
                }
                "branch" => {
                    self.expect("(")?;
                    let plus = self.name()?;
                    // A branch of one node runs from it to the ground.
                    let minus = if self.eat(",") {
                        self.name()?
                    } else {
                        String::new()
                    };
                    self.expect(")")?;
                    for branch in self.names()? {
                        module.branches.push((branch, plus.clone(), minus.clone()));
                    }
                }
                "parameter" => self.parameters(&attributes, &mut module.parameters)?,
                "aliasparam" => {
                    let alias = self.name()?;
                    self.expect("=")?;
                    let parameter = self.name()?;
                    self.expect(";")?;
                    module.aliases.push((alias, parameter));
                }
                "real" | "integer" => {
                    let kind = if keyword == "real" {
                        Type::Real
                    } else {
                        Type::Integer
                    };
                    for variable in self.names()? {
                        if let Some(description) = attributes.get("desc") {
                            let units = attributes.get("units").cloned();
                            module
                                .outputs
                                .push((variable.clone(), description.clone(), units));
                        }
                        module.variables.push((variable, kind));
                    }
                }
                "analog" if self.keyword("function") => module.functions.push(self.function()?),
                "analog" => {
                    if analog.is_some() {
                        return Err(self.error("a second analog block"));
                    }
                    analog = Some(self.statement()?);
                }
                discipline if DISCIPLINES.contains(&discipline) => {
                    for node in self.names()? {
                        module.nodes.push((node, keyword.clone()));
                    }
                }
                _ => {
                    self.at -= 1;
                    return Err(self.error("expected a declaration or `analog`"));
                }
            }
        }
        module.analog = analog.ok_or_else(|| self.error("the module has no analog block"))?;
        Ok(module)
    }

    /// The attributes `(* name = "value", ... *)` before a declaration, if any.
    fn attributes(&mut self) -> Result<BTreeMap<String, String>, Error> {
        let mut attributes = BTreeMap::new();
        while self.eat("(*") {
            loop {
                let name = self.name()?;
                self.expect("=")?;
                let Some(Token::Text(value)) = self.next() else {
                    self.at -= 1;
                    return Err(self.error("expected an attribute's text"));
                };
                attributes.insert(name, value);
                if !self.eat(",") {
                    break;
                }
            }
            self.expect("*)")?;
        }
        Ok(attributes)
    }

    /// The parameters of one `parameter` declaration, each with its ranges.
    fn parameters(
        &mut self,
        attributes: &BTreeMap<String, String>,
        parameters: &mut Vec<Parameter>,
    ) -> Result<(), Error> {
        let kind = if self.keyword("integer") {
            Type::Integer
        } else {
            self.keyword("real");
            Type::Real
        };
        loop {
            let line = self.line();
            let name = self.name()?;
            self.expect("=")?;
            let default = self.expression()?;
            let mut ranges = Vec::new();
            loop {
                let exclude = if self.keyword("from") {
                    false
                } else if self.keyword("exclude") {
                    true
                } else {
                    break;
                };
                ranges.push(self.range(exclude)?);
            }
            parameters.push(Parameter {
                name,
                kind,
                default,
                ranges,
                description: attributes.get("desc").cloned(),
                units: attributes.get("units").cloned(),
                line,
            });
            if !self.eat(",") {
                break;
            }
        }
        self.expect(";")
    }

    /// A range `[a:b]`, `(a:b)` or the like, or for `exclude` a single value.
    fn range(&mut self, exclude: bool) -> Result<Range, Error> {
        let opened = if self.eat("[") {
            true
        } else if self.eat("(") {
            false
        } else if exclude {
            let value = self.expression()?;
            return Ok(Range {
                exclude,
                lower: Some((value.clone(), true)),
                upper: Some((value, true)),
            });
        } else {
            return Err(self.error("expected a range"));
        };
        let lower = self.bound()?.map(|value| (value, opened));
        self.expect(":")?;
        let upper = self.bound()?;
        let closed = if self.eat("]") {
            true
        } else {
            self.expect(")")?;
            false
        };
        Ok(Range {
            exclude,
            lower,
            upper: upper.map(|value| (value, closed)),
        })
    }

    /// A range's bound, `None` for `inf` or `-inf`.
    fn bound(&mut self) -> Result<Option<Expression>, Error> {
        let infinite =
            |token: Option<&Token>| matches!(token, Some(Token::Identifier(name)) if name == "inf");
        if infinite(self.peek()) {
            self.at += 1;
            return Ok(None);
        }
        if matches!(self.peek(), Some(Token::Symbol("-"))) && infinite(self.peek_at(1)) {
            self.at += 2;
            return Ok(None);
        }
        self.expression().map(Some)
    }

    /// An analog function, after `analog function`.
    fn function(&mut self) -> Result<Function, Error> {
        if !self.keyword("real") && self.keyword("integer") {
            return Err(self.error("an analog function returns a real number"));
        }
        let name = self.name()?;
        self.expect(";")?;
        let (mut inputs, mut variables) = (Vec::new(), Vec::new());
        loop {
            if self.keyword("input") {
                inputs.extend(self.names()?);
            } else if self.keyword("real") {
                variables.extend(self.names()?.into_iter().map(|name| (name, Type::Real)));
            } else if self.keyword("integer") {
                variables.extend(self.names()?.into_iter().map(|name| (name, Type::Integer)));
            } else {
                break;
            }
        }
        let body = self.statement()?;
        self.expect_keyword("endfunction")?;
        Ok(Function {
            name,
            inputs,
            variables,
            body,
        })
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one arm per statement of the language, which read best side by side"
    )]
    fn statement(&mut self) -> Result<Statement, Error> {
        let line = self.line();
        if self.keyword("begin") {
            let name = if self.eat(":") {
                Some(self.name()?)
            } else {
                None
            };
            let mut variables = Vec::new();
            while name.is_some() {
                let kind = if self.keyword("real") {
                    Type::Real
                } else if self.keyword("integer") {
                    Type::Integer
                } else {
                    break;
                };
                variables.extend(self.names()?.into_iter().map(|variable| (variable, kind)));
            }
            let mut statements = Vec::new();
            while !self.keyword("end") {
                statements.push(self.statement()?);
            }
            return Ok(match name {
                Some(name) => Statement::Named {
                    name,
                    variables,
                    statements,
                },
                None => Statement::Block(statements),
            });
        }
        if self.keyword("if") {
            self.expect("(")?;
            let condition = self.expression()?;
            self.expect(")")?;
            let then = Box::new(self.statement()?);
            let otherwise = if self.keyword("else") {
                Some(Box::new(self.statement()?))
            } else {
                None
            };
            return Ok(Statement::If {
                condition,
                then,
                otherwise,
            });
        }
        if self.keyword("while") {
            self.expect("(")?;
            let condition = self.expression()?;
            self.expect(")")?;
            let body = Box::new(self.statement()?);
            return Ok(Statement::While { condition, body });
        }
        if self.keyword("for") {
            self.expect("(")?;
            let start = self.assignment()?;
            self.expect(";")?;
            let condition = self.expression()?;
            self.expect(";")?;
            let step = self.assignment()?;
            self.expect(")")?;
            let body = self.statement()?;
            let body = Box::new(Statement::Block(vec![body, step]));
            return Ok(Statement::Block(vec![
                start,
                Statement::While { condition, body },
            ]));
        }
        if self.eat(";") {
            return Ok(Statement::Block(Vec::new()));
        }
        if self.eat("@") {
            self.expect("(")?;
            let event = self.name()?;
            // The analyses the event names, if any, are all the same to jolty, whose
            // every analysis starts from its own first step.
            if self.eat("(") && !self.eat(")") {
                self.arguments()?;
            }
            self.expect(")")?;
            if !matches!(
                event.as_str(),
                "initial_step" | "initial_model" | "initial_instance"
            ) {
                return Err(Error::at(
                    line,
                    format!("the event {event} is not compiled yet"),
                ));
            }
            let body = Box::new(self.statement()?);
            return Ok(Statement::Initial { body, line });
        }
        if let Some(Token::System(name)) = self.peek() {
            let name = name.clone();
            self.at += 1;
            let mut arguments = Vec::new();
            if self.eat("(") && !self.eat(")") {
                arguments = self.arguments()?;
            }
            self.expect(";")?;
            return Ok(Statement::Task {
                name,
                arguments,
                line,
            });
        }
        if matches!(self.peek_at(1), Some(Token::Symbol("("))) {
            let nature = self.name()?;
            self.expect("(")?;
            let mut operands = vec![self.name()?];
            while self.eat(",") {
                operands.push(self.name()?);
            }
            self.expect(")")?;
            self.expect("<+")?;
            let value = self.expression()?;
            self.expect(";")?;
            return Ok(Statement::Contribute {
                nature,
                operands,
                value,
                line,
            });
        }
        let assignment = self.assignment()?;
        self.expect(";")?;
        Ok(assignment)
    }

    fn assignment(&mut self) -> Result<Statement, Error> {
        let line = self.line();
        let target = self.name()?;
        self.expect("=")?;
        let value = self.expression()?;
        Ok(Statement::Assign {
            target,
            value,
            line,
        })
    }

    /// Expressions separated by commas, up to the `)` that ends them.
    fn arguments(&mut self) -> Result<Vec<Expression>, Error> {
        let mut arguments = vec![self.expression()?];
        while self.eat(",") {
            arguments.push(self.expression()?);
        }
        self.expect(")")?;
        Ok(arguments)
    }

    pub(crate) fn expression(&mut self) -> Result<Expression, Error> {
        let condition = self.binary(0)?;
        if self.eat("?") {
            let then = self.expression()?;
            self.expect(":")?;
            let otherwise = self.expression()?;
            return Ok(Expression::Conditional(
                Box::new(condition),
                Box::new(then),
                Box::new(otherwise),
            ));
        }
        Ok(condition)
    }

    /// The binary operations of precedence `level` and tighter, left-associative.
    fn binary(&mut self, level: usize) -> Result<Expression, Error> {
        let Some(operators) = BINARY.get(level) else {
            return self.unary();
        };
        let mut left = self.binary(level + 1)?;
        loop {
            let found = operators
                .iter()
                .find(|operator| matches!(self.peek(), Some(Token::Symbol(symbol)) if symbol == *operator));
            let Some(&operator) = found else {
                return Ok(left);
            };
            self.at += 1;
            let right = self.binary(level + 1)?;
            left = Expression::Binary(operator, Box::new(left), Box::new(right));
        }
    }

    fn unary(&mut self) -> Result<Expression, Error> {
        for operator in ["-", "+", "!"] {
            if self.eat(operator) {
                let operand = self.unary()?;
                return Ok(match (operator, operand) {
                    ("+", operand) => operand,
                    (operator, operand) => Expression::Unary(operator, Box::new(operand)),
                });
            }
        }
        self.primary()
    }

    fn primary(&mut self) -> Result<Expression, Error> {
        match self.next() {
            Some(Token::Real(value)) => Ok(Expression::Real(value)),
            Some(Token::Integer(value)) => Ok(Expression::Integer(value)),
            Some(Token::Text(text)) => Ok(Expression::Text(text)),
            Some(Token::Symbol("(")) => {
                let inner = self.expression()?;
                self.expect(")")?;
                Ok(inner)
            }
            Some(Token::Identifier(name) | Token::System(name)) => {
                if self.eat("(") {
                    let arguments = if self.eat(")") {
                        Vec::new()
                    } else {
                        self.arguments()?
                    };
                    Ok(Expression::Call(name, arguments))
                } else if name.starts_with('$') {
                    Ok(Expression::Call(name, Vec::new()))
                } else {
                    Ok(Expression::Name(name))
                }
            }
            _ => {
                self.at -= 1;
                Err(self.error("expected an expression"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokens;

    fn expression(source: &str) -> Expression {
        let tokens = tokens(source, &|_| None).expect("the source reads");
        Parser {
            tokens: &tokens,
            at: 0,
        }
        .expression()
        .expect("an expression")
    }

    #[test]
    fn operators_bind_by_verilog_s_precedence() {
        let parsed = expression("a + b * c ** 2 < d && !e");
        let name = |name: &str| Box::new(Expression::Name(name.to_owned()));
        let power = Expression::Binary("**", name("c"), Box::new(Expression::Integer(2)));
        let product = Expression::Binary("*", name("b"), Box::new(power));
        let sum = Expression::Binary("+", name("a"), Box::new(product));
        let less = Expression::Binary("<", Box::new(sum), name("d"));
        let not = Expression::Unary("!", name("e"));
        assert_eq!(
            parsed,
            Expression::Binary("&&", Box::new(less), Box::new(not))
        );
    }

    #[test]
    fn a_module_reads_with_its_declarations() {
        let source = r#"
            module r(a, b);
                inout a, b;
                electrical a, b;
                (* desc = "The resistance", units = "Ohm" *)
                parameter real r = 1 from (0:inf);
                analog I(a, b) <+ V(a, b) / r;
            endmodule
        "#;
        let module = parse(&tokens(source, &|_| None).expect("tokens")).expect("a module");
        assert_eq!(module.ports, ["a", "b"]);
        assert_eq!(
            module.parameters[0].description.as_deref(),
            Some("The resistance")
        );
        assert!(matches!(module.analog, Statement::Contribute { .. }));
    }
}
