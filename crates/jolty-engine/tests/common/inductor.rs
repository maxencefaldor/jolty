//! Inductor instance builder shared across engine integration tests.

use jolty_ir::Instance;

use crate::common::two_terminal;

/// Builds an inductor named `name` of `inductance` henries between `a` and `b`.
pub(crate) fn inductor(name: &str, inductance: f64, a: &str, b: &str) -> Instance {
    two_terminal(name, "inductor", "inductance", inductance, a, b)
}
