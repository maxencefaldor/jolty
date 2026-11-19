//! DC waveform and grounded voltage source builders shared across engine tests.

use jolty_ir::Instance;
use serde_json::{Value, json};

use crate::common::voltage_source;

/// Builds a constant DC waveform at `value`.
pub(crate) fn dc(value: f64) -> Value {
    json!({"dc": value})
}

/// Builds a grounded DC voltage source named `name` holding `positive` at `value` volts over
/// `0`.
pub(crate) fn voltage_dc(name: &str, positive: &str, value: f64) -> Instance {
    voltage_source(name, positive, "0", dc(value))
}
