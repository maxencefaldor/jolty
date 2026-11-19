//! The tokens of an expression.

use crate::SyntaxError;

/// One token, and the column, counted from 1, where it starts.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Token {
    pub(crate) kind: Kind,
    pub(crate) column: usize,
}

/// Lexical token kind.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Kind {
    Number(f64),
    Name(String),
    Symbol(Symbol),
    End,
}

/// An operator or a punctuation mark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Symbol {
    Open,
    Close,
    Comma,
    Plus,
    Minus,
    Times,
    Divide,
    Power,
    Not,
    Question,
    Colon,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
    And,
    Or,
}

impl Symbol {
    /// Returns the source spelling of this symbol.
    pub(crate) const fn text(self) -> &'static str {
        match self {
            Self::Open => "(",
            Self::Close => ")",
            Self::Comma => ",",
            Self::Plus => "+",
            Self::Minus => "-",
            Self::Times => "*",
            Self::Divide => "/",
            Self::Power => "^",
            Self::Not => "!",
            Self::Question => "?",
            Self::Colon => ":",
            Self::Less => "<",
            Self::LessEqual => "<=",
            Self::Greater => ">",
            Self::GreaterEqual => ">=",
            Self::Equal => "==",
            Self::NotEqual => "!=",
            Self::And => "&&",
            Self::Or => "||",
        }
    }
}

/// Every symbol, the two-character ones first, so each matches greedily.
const SYMBOLS: [Symbol; 19] = [
    Symbol::LessEqual,
    Symbol::GreaterEqual,
    Symbol::Equal,
    Symbol::NotEqual,
    Symbol::And,
    Symbol::Or,
    Symbol::Open,
    Symbol::Close,
    Symbol::Comma,
    Symbol::Plus,
    Symbol::Minus,
    Symbol::Times,
    Symbol::Divide,
    Symbol::Power,
    Symbol::Not,
    Symbol::Question,
    Symbol::Colon,
    Symbol::Less,
    Symbol::Greater,
];

/// Tokenizes `text` into a sequence ending with [`Kind::End`].
pub(crate) fn tokenize(text: &str) -> Result<Vec<Token>, SyntaxError> {
    let mut tokens = Vec::new();
    let mut rest = text;
    let mut column = 1;
    loop {
        let trimmed = rest.trim_start();
        column += rest[..rest.len() - trimmed.len()].chars().count();
        rest = trimmed;
        let Some(first) = rest.chars().next() else {
            tokens.push(Token {
                kind: Kind::End,
                column,
            });
            return Ok(tokens);
        };
        let (kind, length) = if first.is_ascii_digit() || first == '.' {
            number(rest, column)?
        } else if first.is_ascii_alphabetic() || first == '_' {
            let length = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(rest.len());
            (Kind::Name(rest[..length].to_owned()), length)
        } else if let Some(symbol) = SYMBOLS.into_iter().find(|s| rest.starts_with(s.text())) {
            (Kind::Symbol(symbol), symbol.text().len())
        } else {
            return Err(SyntaxError::new(format!("unexpected `{first}`"), column));
        };
        tokens.push(Token { kind, column });
        column += rest[..length].chars().count();
        rest = &rest[length..];
    }
}

/// Scans a floating-point literal (digits with optional fraction and exponent) at the start
/// of `text`.
fn number(text: &str, column: usize) -> Result<(Kind, usize), SyntaxError> {
    let bytes = text.as_bytes();
    let digits = |from: usize| {
        bytes[from..]
            .iter()
            .position(|byte| !byte.is_ascii_digit())
            .map_or(bytes.len(), |offset| from + offset)
    };
    let mut end = digits(0);
    if bytes.get(end) == Some(&b'.') {
        end = digits(end + 1);
    }
    if matches!(bytes.get(end), Some(b'e' | b'E')) {
        let sign = usize::from(matches!(bytes.get(end + 1), Some(b'+' | b'-')));
        let exponent = digits(end + 1 + sign);
        if exponent > end + 1 + sign {
            end = exponent;
        }
    }
    let written = &text[..end];
    let value = written
        .parse()
        .map_err(|_| SyntaxError::new(format!("`{written}` is not a number"), column))?;
    Ok((Kind::Number(value), end))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(text: &str) -> Vec<Kind> {
        tokenize(text)
            .expect("valid tokens")
            .into_iter()
            .map(|token| token.kind)
            .collect()
    }

    #[test]
    fn splits_numbers_names_and_symbols() {
        assert_eq!(
            kinds("2.5e-3*v(a_1)>=.5"),
            [
                Kind::Number(2.5e-3),
                Kind::Symbol(Symbol::Times),
                Kind::Name("v".to_owned()),
                Kind::Symbol(Symbol::Open),
                Kind::Name("a_1".to_owned()),
                Kind::Symbol(Symbol::Close),
                Kind::Symbol(Symbol::GreaterEqual),
                Kind::Number(0.5),
                Kind::End,
            ]
        );
        // An `e` without an exponent ends the number; the name follows.
        assert_eq!(
            kinds("2e"),
            [Kind::Number(2.0), Kind::Name("e".to_owned()), Kind::End]
        );
    }

    #[test]
    fn reports_where_a_token_is_unknown() {
        let error = tokenize("1 + $").expect_err("no `$` token");
        assert_eq!(error.to_string(), "unexpected `$` at column 5");
        let error = tokenize(" . ").expect_err("a lone point");
        assert_eq!(error.to_string(), "`.` is not a number at column 2");
    }

    #[test]
    fn preserves_one_based_char_columns_with_unicode_and_multiline_whitespace() {
        let empty = tokenize("").expect("empty token stream");
        assert_eq!(empty.len(), 1);
        assert_eq!(empty[0].column, 1);

        let unicode = "\u{00A0}v(a)\n\t\u{2003}+ 1.5\u{3000}";
        let tokens = tokenize(unicode).expect("valid tokens");
        let columns: Vec<usize> = tokens.iter().map(|token| token.column).collect();
        // Characters: [1:\u{00A0}, 2:v, 3:(, 4:a, 5:), 6:\n, 7:\t, 8:\u{2003}, 9:+, 10:' ',
        // 11:1, 12:., 13:5, 14:\u{3000}], End at 15.
        assert_eq!(columns, [2, 3, 4, 5, 9, 11, 15]);

        let error = tokenize("\u{00A0}\n  \u{03B1}").expect_err("unexpected Greek letter");
        assert_eq!(error.to_string(), "unexpected `α` at column 5");
    }
}
