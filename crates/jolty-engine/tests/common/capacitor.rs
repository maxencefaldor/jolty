//! Capacitor instance builder shared across engine integration tests.

use jolty_ir::Instance;

use crate::common::two_terminal;

/// A capacitor named `name` of `capacitance` farads between `a` and `b`.
pub(crate) fn capacitor(name: &str, capacitance: f64, a: &str, b: &str) -> Instance {
    two_terminal(name, "capacitor", "capacitance", capacitance, a, b)
}
