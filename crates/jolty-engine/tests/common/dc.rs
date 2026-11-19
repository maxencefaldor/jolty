//! DC waveform and grounded voltage source builders shared across engine tests.

use jolty_ir::Instance;
use serde_json::{Value, json};

use crate::common::voltage_source;

/// A constant DC waveform at `level`.
pub(crate) fn dc(level: f64) -> Value {
    json!({"dc": level})
}

/// A grounded DC voltage source named `name` holding `positive` at `level` volts over `0`.
pub(crate) fn voltage_dc(name: &str, positive: &str, level: f64) -> Instance {
    voltage_source(name, positive, "0", dc(level))
}
