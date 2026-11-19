//! Verilog-A compiler producing device kernels for jolty, alongside the compiled kernels.
//!
//! Each model compiles directly from published Verilog-A source into plain Rust implementing
//! [`jolty_models::Kernel`]. Generated kernels are generic over jolty's number types so
//! parameter sensitivities propagate through [`jolty_models::Gradient`] and Jacobians propagate
//! through [`jolty_models::Dual`].
//!
//! Compilation proceeds in four passes, each in a dedicated module:
//!
//! 1. **Lexer** scans source into tokens while processing compiler directives: standard headers
//!    (using jolty's physical constant definitions), object-like and function-like macros, and
//!    conditional directives.
//! 2. **Parser** reads one module of the CMC Verilog-A subset by recursive descent: ports,
//!    disciplines, named branches, parameters with ranges and attributes, `localparam`
//!    constants, variables, analog functions, and the analog block.
//! 3. **Resolver** binds names, splits contributions into resistive, `ddt`, and noise terms,
//!    and classifies variables by dependence. Variables depending on probes or simulator state
//!    are dynamic; all others are static. The resolver also records the exact probe
//!    dependencies of every variable.
//! 4. **Emitter** generates the Rust kernel. Parameters map to `Parameters<R>`, static
//!    expressions compute once in `prepare`, and parameter conditions controlling branch
//!    topology form the kernel shape claimed in `layout`. During `evaluate`, the analog block
//!    runs on dual numbers carrying only the needed probe derivatives, then stamps branch sums
//!    once. Each `$limit` clamps against stored state and linearizes around the applied
//!    voltage, and noise functions lower to `noise`.
//!
//! Generated code obeys jolty's kernel constraints (compatible with GPU execution): pure,
//! allocation-free, panic-free, and free of dynamic dispatch.
//!
//! Compiled kernels under `kernels` are checked in and regenerated via
//! `cargo run -p jolty-veriloga --bin compile`; tests fail on any drift from compiler output.
//! The default `kernels` feature gates compilation of those generated modules so
//! `--no-default-features` keeps the compiler runnable even if a compiler change breaks a
//! generated kernel.

mod ast;
mod emit;
mod ir;
#[cfg(feature = "kernels")]
pub mod kernels;
mod lexer;
mod linear;
mod parser;

use std::fmt;

/// Compiled Verilog-A models and test fixtures paired with their generated Rust kernel paths
/// inside the crate.
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

/// Compilation error locating the failing source line and failure reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    /// Source line number (1-based), or 0 for module-wide errors.
    pub line: usize,
    /// Human-readable error description.
    pub message: String,
}

impl Error {
    /// Constructs an [`Error`] with `message` at `line`.
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

/// Compiles a single Verilog-A module from `source` (originating at `path`) into Rust kernel
/// source text.
///
/// Non-standard included files are resolved through `files`.
///
/// # Errors
///
/// Returns an [`Error`] at the first unsupported or malformed construct.
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
