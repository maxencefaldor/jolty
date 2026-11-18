//! A compiler of Verilog-A into jolty's device kernels, and the kernels it compiled.
//!
//! Every model then comes from one source, its published Verilog-A, and its output is
//! plain Rust: a kernel implementing [`jolty_models::Kernel`], the same form a person
//! writes, generic over jolty's number types, so its parameter derivatives are exact
//! through [`jolty_models::Gradient`] and its Jacobian through [`jolty_models::Dual`].
//!
//! The compiler runs in four passes, each its own module:
//!
//! 1. **The lexer** reads the source into tokens, carrying out its compiler directives:
//!    the standard headers, whose constants it takes from jolty's own definitions,
//!    macros with and without arguments, and the conditionals.
//! 1. **The parser** reads one module of the CMC's subset by recursive descent: its
//!    ports, disciplines, named branches, parameters with their ranges and attributes,
//!    `localparam` constants, variables, analog functions and analog block.
//! 1. **The resolver** binds each name, splits each contribution into its resistive,
//!    `ddt` and noise parts, and decides by dependence what the kernel prepares and what
//!    it evaluates: a variable is dynamic when its value or a condition it is assigned
//!    under depends on a probe or the simulator's state, and static otherwise; it also
//!    finds the probes each variable depends on.
//! 1. **The emitter** writes the kernel. The parameters become a `Parameters<R>`
//!    struct; everything static is computed once in `prepare`; the conditions on
//!    parameters that switch, collapse or leave out branches become the kernel's shape,
//!    whose internal unknowns and entries `layout` claims; and `evaluate` runs the
//!    analog block on dual numbers, each variable carrying the derivatives of the probes
//!    it depends on alone, then stamps each branch's sums once. A `$limit` becomes the
//!    voltage limited from the last evaluation's, which the kernel's memory keeps, and
//!    every value is extended linearly from it to the applied voltage, as the
//!    hand-written kernels do; the noise functions become `noise`.
//!
//! The generated code follows the kernel rules, which are also what a GPU requires: it
//! is pure, allocates nothing, never panics and dispatches nothing at run time.
//!
//! The kernels compiled, in `kernels`, are checked in, written by
//! `cargo run -p jolty-veriloga --bin compile` and never by hand; a test fails when one
//! differs from what the compiler writes. They build with the default feature
//! `kernels`, so that the compiler runs without them, as with `--no-default-features`
//! once a change to it leaves a kernel that does not build.

mod ast;
mod emit;
mod ir;
#[cfg(feature = "kernels")]
pub mod kernels;
mod lexer;
mod linear;
mod parser;

use std::fmt;

/// The models compiled, each a Verilog-A source and the kernel module the compiler writes
/// from it, by their paths in the crate: the library's, in `models/` and `src/kernels/`,
/// and the tests' fixtures, which exercise a construct each, in `tests/fixtures/`.
pub const MODELS: [(&str, &str); 8] = [
    ("models/resistor.va", "src/kernels/resistor.rs"),
    ("models/diode.va", "src/kernels/diode.rs"),
    (
        "tests/fixtures/constructs.va",
        "tests/fixtures/constructs.rs",
    ),
    (
        "tests/fixtures/thermal_resistance.va",
        "tests/fixtures/thermal_resistance.rs",
    ),
    (
        "tests/fixtures/heated_resistor.va",
        "tests/fixtures/heated_resistor.rs",
    ),
    (
        "tests/fixtures/limited_diode.va",
        "tests/fixtures/limited_diode.rs",
    ),
    (
        "tests/fixtures/initial_events.va",
        "tests/fixtures/initial_events.rs",
    ),
    ("tests/fixtures/networks.va", "tests/fixtures/networks.rs"),
];

/// Why a source does not compile: the line it was read at and what was wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    /// The line of the source, from 1, or 0 for the module as a whole.
    pub line: usize,
    /// What was wrong.
    pub message: String,
}

impl Error {
    /// The error `message` at `line`.
    pub(crate) const fn at(line: usize, message: String) -> Self {
        Self { line, message }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for Error {}

/// The kernel `source`, one Verilog-A module read from the file `path`, compiles into,
/// as the text of a Rust module, the files it includes but the standard headers read by
/// `files`, by the names it gives them.
///
/// # Errors
///
/// Returns an [`Error`] at the first construct it cannot read or does not compile yet.
pub fn compile(
    source: &str,
    path: &str,
    files: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    let tokens = lexer::tokens(source, files)?;
    let module = parser::parse(&tokens)?;
    let module = ir::resolve(&module)?;
    // The comment that opens the source, which carries its notice, opens the kernel too.
    let header: Vec<&str> = source
        .lines()
        .map_while(|line| line.strip_prefix("//"))
        .map(|line| line.strip_prefix(' ').unwrap_or(line))
        .collect();
    emit::emit(&module, (path, &header.join("\n")))
}
