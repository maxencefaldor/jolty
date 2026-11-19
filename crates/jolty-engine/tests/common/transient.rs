//! Transient analysis constructor shared across engine integration tests.

use std::collections::BTreeMap;

use jolty_engine::{Method, Transient};

/// A transient over `[0, stop]` with optional `max_step` and `method`, starting from the
/// operating point with no breakpoints or initial conditions.
pub(crate) fn transient(stop: f64, max_step: Option<f64>, method: Method) -> Transient {
    Transient {
        stop,
        max_step,
        method,
        breakpoints: Vec::new(),
        initial_conditions: BTreeMap::new(),
        operating_point: true,
    }
}
