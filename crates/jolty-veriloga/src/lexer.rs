//! Verilog-A's tokens, read from a source with its compiler directives carried out:
//! `` `include `` of the standard headers, `` `define `` with and without arguments, their
//! expansion, `` `undef ``, and `` `ifdef ``, `` `ifndef ``, `` `elsif ``, `` `else `` and
//! `` `endif ``.

use std::collections::BTreeMap;
use std::f64::consts;

use jolty_models::{BOLTZMANN, ELEMENTARY_CHARGE};

use crate::Error;

/// A token of the language, after preprocessing.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Token {
    /// A name, such as `resistance` or `V`.
    Identifier(String),
    /// A system function or task, such as `$temperature`, with its `$`.
    System(String),
    /// A real number, its scale factor applied.
    Real(f64),
    /// An integer.
    Integer(i64),
    /// A string, its quotes and escapes removed.
    Text(String),
    /// An operator or a punctuation mark.
    Symbol(&'static str),
}

/// A token and the line of the source it starts on.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Spanned {
    pub(crate) token: Token,
    pub(crate) line: usize,
}

/// The operators and punctuation marks, longest first, so that each matches before its
/// prefixes.
const SYMBOLS: [&str; 40] = [
    "<+", "**", "&&", "||", "==", "!=", "<=", ">=", "<<", ">>", "(*", "*)", "+", "-", "*", "/",
    "%", "!", "<", ">", "?", ":", "(", ")", "[", "]", "{", "}", ",", ";", "=", "@", "#", ".", "~",
    "&", "|", "^", "'", "$",
];

/// The standard headers a source includes, which the compiler knows: the disciplines,
/// whose natures it holds itself, under the standard's name and under `discipline.h`, the
/// name the CMC's code standard gives it, and the constants, defined below.
const HEADERS: [&str; 3] = ["disciplines.vams", "discipline.h", "constants.vams"];

/// What reads the text of a file a source includes, by the name it gives, `None` for
/// one it cannot read.
pub(crate) type Files<'a> = &'a dyn Fn(&str) -> Option<String>;

/// What raw lexing gives: a token, a directive's name, or the end of a line, which ends a
/// directive.
#[derive(Debug, Clone, PartialEq)]
enum Raw {
    Token(Token),
    Directive(String),
    Newline,
}

/// A raw item, its line, and whether a space came before it, which tells a macro's
/// arguments from its body.
#[derive(Debug, Clone)]
struct Item {
    raw: Raw,
    line: usize,
    spaced: bool,
}

/// A macro: its arguments' names, `None` for one without, and its body.
#[derive(Debug, Clone)]
struct Macro {
    arguments: Option<Vec<String>>,
    body: Vec<Spanned>,
}

/// The tokens of `source`, its directives carried out, the files it includes read by
/// `files`.
///
/// # Errors
///
/// Returns an [`Error`] at the line of the first token or directive it cannot read.
pub(crate) fn tokens(source: &str, files: Files<'_>) -> Result<Vec<Spanned>, Error> {
    let mut macros = constants();
    let mut out = Vec::new();
    preprocess(&raw(source)?, (&mut macros, files), &mut out)?;
    Ok(out)
}

/// The constants of the standard `constants.vams`: each one's macro, the Rust constant
/// the kernels name it by, and its value. Each comes from the definition it names: the
/// mathematical constants from Rust's, the elementary charge and Boltzmann's constant
/// from jolty's models, which take them from the SI's exact values, and 0 °C in kelvins,
/// which the SI defines.
pub(crate) const CONSTANTS: [(&str, &str, f64); 17] = [
    ("M_PI", "std::f64::consts::PI", consts::PI),
    ("M_TWO_PI", "std::f64::consts::TAU", consts::TAU),
    ("M_PI_2", "std::f64::consts::FRAC_PI_2", consts::FRAC_PI_2),
    ("M_PI_4", "std::f64::consts::FRAC_PI_4", consts::FRAC_PI_4),
    ("M_1_PI", "std::f64::consts::FRAC_1_PI", consts::FRAC_1_PI),
    ("M_2_PI", "std::f64::consts::FRAC_2_PI", consts::FRAC_2_PI),
    (
        "M_2_SQRTPI",
        "std::f64::consts::FRAC_2_SQRT_PI",
        consts::FRAC_2_SQRT_PI,
    ),
    ("M_E", "std::f64::consts::E", consts::E),
    ("M_LOG2E", "std::f64::consts::LOG2_E", consts::LOG2_E),
    ("M_LOG10E", "std::f64::consts::LOG10_E", consts::LOG10_E),
    ("M_LN2", "std::f64::consts::LN_2", consts::LN_2),
    ("M_LN10", "std::f64::consts::LN_10", consts::LN_10),
    ("M_SQRT2", "std::f64::consts::SQRT_2", consts::SQRT_2),
    (
        "M_SQRT1_2",
        "std::f64::consts::FRAC_1_SQRT_2",
        consts::FRAC_1_SQRT_2,
    ),
    ("P_Q", "ELEMENTARY_CHARGE", ELEMENTARY_CHARGE),
    ("P_K", "BOLTZMANN", BOLTZMANN),
    ("P_CELSIUS0", "273.15", 273.15),
];

/// The macros of [`CONSTANTS`].
fn constants() -> BTreeMap<String, Macro> {
    CONSTANTS
        .into_iter()
        .map(|(name, _, value)| {
            let body = vec![Spanned {
                token: Token::Real(value),
                line: 0,
            }];
            let defined = Macro {
                arguments: None,
                body,
            };
            (name.to_owned(), defined)
        })
        .collect()
}

/// The raw items of `source`: tokens, directives and ends of lines.
#[expect(
    clippy::too_many_lines,
    reason = "one arm per kind of token, which read best side by side"
)]
fn raw(source: &str) -> Result<Vec<Item>, Error> {
    let characters: Vec<char> = source.chars().collect();
    let mut items = Vec::new();
    let (mut at, mut line, mut spaced) = (0, 1, false);
    let error = Error::at;
    while let Some(&character) = characters.get(at) {
        let next = characters.get(at + 1).copied();
        let mut push = |raw: Raw, spaced: &mut bool| {
            items.push(Item {
                raw,
                line,
                spaced: *spaced,
            });
            *spaced = false;
        };
        match character {
            '\n' => {
                push(Raw::Newline, &mut spaced);
                line += 1;
                at += 1;
            }
            '\\' if next == Some('\n') => {
                line += 1;
                at += 2;
                spaced = true;
            }
            _ if character.is_whitespace() => {
                at += 1;
                spaced = true;
            }
            '/' if next == Some('/') => {
                while characters.get(at).is_some_and(|&c| c != '\n') {
                    at += 1;
                }
                spaced = true;
            }
            '/' if next == Some('*') => {
                let end = (at + 2..characters.len())
                    .find(|&end| {
                        characters.get(end) == Some(&'*') && characters.get(end + 1) == Some(&'/')
                    })
                    .map_or(characters.len(), |end| end + 2);
                line += characters
                    .get(at..end)
                    .unwrap_or_default()
                    .iter()
                    .filter(|&&c| c == '\n')
                    .count();
                at = end;
                spaced = true;
            }
            '`' => {
                let start = at + 1;
                at = start;
                while characters
                    .get(at)
                    .is_some_and(|&c| c.is_alphanumeric() || c == '_')
                {
                    at += 1;
                }
                let name: String = characters
                    .get(start..at)
                    .unwrap_or_default()
                    .iter()
                    .collect();
                push(Raw::Directive(name), &mut spaced);
            }
            '"' => {
                let (text, end) =
                    string(&characters, at + 1).map_err(|message| error(line, message))?;
                at = end;
                push(Raw::Token(Token::Text(text)), &mut spaced);
            }
            _ if character.is_ascii_digit()
                || (character == '.' && next.is_some_and(|c| c.is_ascii_digit())) =>
            {
                let (token, end) =
                    number(&characters, at).map_err(|message| error(line, message))?;
                at = end;
                push(Raw::Token(token), &mut spaced);
            }
            _ if character.is_alphabetic() || character == '_' || character == '$' => {
                let start = at;
                at += 1;
                while characters
                    .get(at)
                    .is_some_and(|&c| c.is_alphanumeric() || c == '_' || c == '$')
                {
                    at += 1;
                }
                let name: String = characters
                    .get(start..at)
                    .unwrap_or_default()
                    .iter()
                    .collect();
                let token = if character == '$' {
                    Token::System(name)
                } else {
                    Token::Identifier(name)
                };
                push(Raw::Token(token), &mut spaced);
            }
            _ => {
                let rest: String = characters
                    .get(at..(at + 2).min(characters.len()))
                    .unwrap_or_default()
                    .iter()
                    .collect();
                let symbol = SYMBOLS
                    .iter()
                    .find(|symbol| rest.starts_with(**symbol))
                    .ok_or_else(|| error(line, format!("an unexpected character {character:?}")))?;
                at += symbol.len();
                push(Raw::Token(Token::Symbol(symbol)), &mut spaced);
            }
        }
    }
    Ok(items)
}

/// The string whose text starts at `at` among `characters`, its escapes read, and where
/// it ends, after its closing quote.
fn string(characters: &[char], mut at: usize) -> Result<(String, usize), String> {
    let mut text = String::new();
    loop {
        match characters.get(at) {
            Some('"') => return Ok((text, at + 1)),
            Some('\\') => {
                text.push(match characters.get(at + 1) {
                    Some('n') => '\n',
                    Some('t') => '\t',
                    Some(&other) => other,
                    None => return Err("an unterminated string".into()),
                });
                at += 2;
            }
            Some('\n') | None => return Err("an unterminated string".into()),
            Some(&other) => {
                text.push(other);
                at += 1;
            }
        }
    }
}

/// The number that starts at `at` among `characters`, and where it ends: an integer, or
/// a real with a fraction, an exponent or one of Verilog-A's scale factors.
fn number(characters: &[char], mut at: usize) -> Result<(Token, usize), String> {
    let start = at;
    let digits = |at: &mut usize| {
        while characters
            .get(*at)
            .is_some_and(|c| c.is_ascii_digit() || *c == '_')
        {
            *at += 1;
        }
    };
    digits(&mut at);
    let mut real = false;
    if characters.get(at) == Some(&'.') {
        real = true;
        at += 1;
        digits(&mut at);
    }
    if matches!(characters.get(at), Some('e' | 'E')) {
        let sign = matches!(characters.get(at + 1), Some('+' | '-'));
        let first = characters.get(at + 1 + usize::from(sign));
        if first.is_some_and(char::is_ascii_digit) {
            real = true;
            at += 1 + usize::from(sign);
            digits(&mut at);
        }
    }
    let text: String = characters
        .get(start..at)
        .unwrap_or_default()
        .iter()
        .filter(|&&c| c != '_')
        .collect();
    // Verilog-A's scale factors, which multiply the number by a power of ten.
    let scale = match characters.get(at) {
        Some('T') => Some("e12"),
        Some('G') => Some("e9"),
        Some('M') => Some("e6"),
        Some('K' | 'k') => Some("e3"),
        Some('m') => Some("e-3"),
        Some('u') => Some("e-6"),
        Some('n') => Some("e-9"),
        Some('p') => Some("e-12"),
        Some('f') => Some("e-15"),
        Some('a') => Some("e-18"),
        _ => None,
    };
    let followed = |at: usize| {
        characters
            .get(at)
            .is_some_and(|c| c.is_alphanumeric() || *c == '_')
    };
    if let Some(exponent) = scale.filter(|_| !followed(at + 1) && !text.contains(['e', 'E'])) {
        let value: f64 = format!("{text}{exponent}")
            .parse()
            .map_err(|_| format!("an unreadable number {text}"))?;
        return Ok((Token::Real(value), at + 1));
    }
    if followed(at) {
        return Err(format!("an unreadable number starting {text}"));
    }
    let token = if real {
        Token::Real(
            text.parse()
                .map_err(|_| format!("an unreadable number {text}"))?,
        )
    } else {
        Token::Integer(
            text.parse()
                .map_err(|_| format!("an unreadable number {text}"))?,
        )
    };
    Ok((token, at))
}

/// Carries out the directives of `items`, appending the tokens they leave to `out`, with
/// the macros `macros` defines, which they change.
fn preprocess(
    items: &[Item],
    (macros, files): (&mut BTreeMap<String, Macro>, Files<'_>),
    out: &mut Vec<Spanned>,
) -> Result<(), Error> {
    // Whether each enclosing conditional's text is kept, and whether one of its branches
    // has been taken yet.
    let mut conditions: Vec<(bool, bool)> = Vec::new();
    let mut at = 0;
    while let Some(item) = items.get(at) {
        at += 1;
        let line = item.line;
        let kept = conditions.iter().all(|&(kept, _)| kept);
        let error = |message: String| Error::at(line, message);
        let name_after = |at: &mut usize| -> Result<String, Error> {
            match items.get(*at).map(|item| &item.raw) {
                Some(Raw::Token(Token::Identifier(name))) => {
                    *at += 1;
                    Ok(name.clone())
                }
                _ => Err(error("a directive lacks its name".into())),
            }
        };
        match &item.raw {
            Raw::Newline => {}
            Raw::Directive(directive) => match directive.as_str() {
                "ifdef" | "ifndef" => {
                    let name = name_after(&mut at)?;
                    let defined = macros.contains_key(&name) == (directive == "ifdef");
                    conditions.push((defined, defined));
                }
                "elsif" => {
                    let name = name_after(&mut at)?;
                    let Some((kept, taken)) = conditions.last_mut() else {
                        return Err(error("an `elsif without an `ifdef".into()));
                    };
                    *kept = !*taken && macros.contains_key(&name);
                    *taken |= *kept;
                }
                "else" => {
                    let Some((kept, taken)) = conditions.last_mut() else {
                        return Err(error("an `else without an `ifdef".into()));
                    };
                    *kept = !*taken;
                    *taken = true;
                }
                "endif" => {
                    conditions
                        .pop()
                        .ok_or_else(|| error("an `endif without an `ifdef".into()))?;
                }
                _ if !kept => {}
                "include" => {
                    let file = match items.get(at).map(|item| &item.raw) {
                        Some(Raw::Token(Token::Text(file))) => file.clone(),
                        _ => return Err(error("an `include lacks its file".into())),
                    };
                    at += 1;
                    if !HEADERS.contains(&file.as_str()) {
                        let text =
                            files(&file).ok_or_else(|| error(format!("{file} cannot be read")))?;
                        preprocess(&raw(&text)?, (macros, files), out)?;
                    }
                }
                "define" => at = define(items, at, macros)?,
                "undef" => {
                    let name = name_after(&mut at)?;
                    macros.remove(&name);
                }
                name => at = expand(name, (items, at, line), (macros, files), out)?,
            },
            Raw::Token(token) => {
                if kept {
                    out.push(Spanned {
                        token: token.clone(),
                        line,
                    });
                }
            }
        }
    }
    if conditions.is_empty() {
        Ok(())
    } else {
        Err(Error::at(
            items.last().map_or(0, |item| item.line),
            "an `ifdef without its `endif".into(),
        ))
    }
}

/// Reads the definition of a macro, from `at` on among `items`, after `` `define ``,
/// into `macros`, and returns where it ends.
fn define(
    items: &[Item],
    mut at: usize,
    macros: &mut BTreeMap<String, Macro>,
) -> Result<usize, Error> {
    let line = items.get(at).map_or(0, |item| item.line);
    let error = |message: String| Error::at(line, message);
    let name = match items.get(at).map(|item| &item.raw) {
        Some(Raw::Token(Token::Identifier(name))) => {
            at += 1;
            name.clone()
        }
        _ => return Err(error("a directive lacks its name".into())),
    };
    let mut arguments = None;
    if let Some(Item {
        raw: Raw::Token(Token::Symbol("(")),
        spaced: false,
        ..
    }) = items.get(at)
    {
        at += 1;
        let mut names = Vec::new();
        loop {
            match items.get(at).map(|item| &item.raw) {
                Some(Raw::Token(Token::Identifier(argument))) => {
                    names.push(argument.clone());
                }
                Some(Raw::Token(Token::Symbol(","))) => {}
                Some(Raw::Token(Token::Symbol(")"))) => break,
                _ => return Err(error(format!("{name}'s arguments are unclosed"))),
            }
            at += 1;
        }
        at += 1;
        arguments = Some(names);
    }
    let mut body = Vec::new();
    while let Some(item) = items.get(at) {
        match &item.raw {
            Raw::Newline => break,
            Raw::Token(token) => body.push(Spanned {
                token: token.clone(),
                line: item.line,
            }),
            Raw::Directive(inner) => body.push(Spanned {
                token: Token::Identifier(format!("`{inner}")),
                line: item.line,
            }),
        }
        at += 1;
    }
    macros.insert(name, Macro { arguments, body });
    Ok(at)
}

/// Expands the macro `name`, used at `at` among `items` on `line`, with its arguments,
/// appending the tokens it leaves to `out`, and returns where its use ends.
fn expand(
    name: &str,
    (items, mut at, line): (&[Item], usize, usize),
    (macros, files): (&mut BTreeMap<String, Macro>, Files<'_>),
    out: &mut Vec<Spanned>,
) -> Result<usize, Error> {
    let error = |message: String| Error::at(line, message);
    let Some(defined) = macros.get(name).cloned() else {
        return Err(error(format!("the macro `{name} is not defined")));
    };
    let mut body = defined.body.clone();
    if let Some(names) = &defined.arguments {
        let (values, end) =
            arguments(items, at).ok_or_else(|| error(format!("`{name} lacks its arguments")))?;
        at = end;
        if values.len() != names.len() {
            return Err(error(format!(
                "`{name} takes {} arguments, not {}",
                names.len(),
                values.len()
            )));
        }
        body = substitute(&body, names, &values);
    }
    let expanded: Vec<Item> = body
        .into_iter()
        .map(|spanned| Item {
            raw: match spanned.token {
                Token::Identifier(inner) if inner.starts_with('`') => {
                    Raw::Directive(inner.trim_start_matches('`').to_owned())
                }
                token => Raw::Token(token),
            },
            line,
            spaced: true,
        })
        .collect();
    preprocess(&expanded, (macros, files), out)?;
    Ok(at)
}

/// A macro's arguments, the items from `at` on, between balanced parentheses and split
/// at their top-level commas, and where they end.
fn arguments(items: &[Item], mut at: usize) -> Option<(Vec<Vec<Spanned>>, usize)> {
    while items.get(at)?.raw == Raw::Newline {
        at += 1;
    }
    if items.get(at)?.raw != Raw::Token(Token::Symbol("(")) {
        return None;
    }
    at += 1;
    let (mut values, mut current, mut depth) = (Vec::new(), Vec::new(), 0_usize);
    loop {
        let item = items.get(at)?;
        at += 1;
        let token = match &item.raw {
            Raw::Newline => continue,
            Raw::Directive(name) => Token::Identifier(format!("`{name}")),
            Raw::Token(token) => token.clone(),
        };
        match token {
            Token::Symbol("(" | "[") => depth += 1,
            Token::Symbol(")") if depth == 0 => {
                values.push(current);
                return Some((values, at));
            }
            Token::Symbol(")" | "]") => depth -= 1,
            Token::Symbol(",") if depth == 0 => {
                values.push(std::mem::take(&mut current));
                continue;
            }
            _ => {}
        }
        current.push(Spanned {
            token,
            line: item.line,
        });
    }
}

/// `body` with each identifier among `names` replaced by the tokens of its value.
fn substitute(body: &[Spanned], names: &[String], values: &[Vec<Spanned>]) -> Vec<Spanned> {
    body.iter()
        .flat_map(|spanned| {
            let found = match &spanned.token {
                Token::Identifier(name) => names.iter().position(|argument| argument == name),
                _ => None,
            };
            match found.and_then(|index| values.get(index)) {
                Some(value) => value.clone(),
                None => vec![spanned.clone()],
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(source: &str) -> Vec<Token> {
        tokens(source, &|_| None)
            .expect("the source reads")
            .into_iter()
            .map(|spanned| spanned.token)
            .collect()
    }

    #[test]
    fn numbers_read_with_their_scale_factors() {
        assert_eq!(
            kinds("1 2.5 1e-14 3k 4.7u .5 1_000"),
            [
                Token::Integer(1),
                Token::Real(2.5),
                Token::Real(1e-14),
                Token::Real(3e3),
                Token::Real(4.7e-6),
                Token::Real(0.5),
                Token::Integer(1000),
            ]
        );
    }

    #[test]
    fn macros_expand_with_their_arguments() {
        let source = "`define TWICE(x) (2 * (x))\n`define ONE 1\n`TWICE(`ONE + a)";
        assert_eq!(kinds(source), kinds("(2 * (1 + a))"));
    }

    #[test]
    fn conditionals_keep_the_branch_taken() {
        let source = "`define A\n`ifdef A\nkept\n`else\ndropped\n`endif\n`ifndef A\nno\n`endif";
        assert_eq!(kinds(source), [Token::Identifier("kept".into())]);
    }

    #[test]
    fn the_constants_are_jolty_s() {
        assert_eq!(kinds("`P_K"), [Token::Real(BOLTZMANN)]);
        assert_eq!(
            kinds("`include \"constants.vams\"\n`P_Q"),
            [Token::Real(ELEMENTARY_CHARGE)]
        );
    }

    #[test]
    fn an_included_file_is_read_and_its_macros_defined() {
        let files =
            |name: &str| (name == "macros.include").then(|| "`define HALF 0.5\n".to_owned());
        let included =
            tokens("`include \"macros.include\"\n`HALF", &files).expect("the source reads");
        assert_eq!(
            included
                .into_iter()
                .map(|spanned| spanned.token)
                .collect::<Vec<_>>(),
            [Token::Real(0.5)]
        );
        let missing = tokens("`include \"other.include\"", &files).expect_err("an unread file");
        assert_eq!(missing.to_string(), "line 1: other.include cannot be read");
    }

    #[test]
    fn an_unknown_macro_is_refused_by_name() {
        let error = tokens("\n`P_EPS0", &|_| None).expect_err("an unknown macro");
        assert_eq!(
            error.to_string(),
            "line 2: the macro `P_EPS0 is not defined"
        );
    }
}
