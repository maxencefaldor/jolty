//! Sine waveform and grounded voltage source builders shared across engine tests.

use jolty_ir::Instance;
use serde_json::{Value, json};

use crate::common::voltage_source;

/// Builds a sinusoidal waveform of `offset`, `amplitude`, and `frequency`.
pub(crate) fn sine(offset: f64, amplitude: f64, frequency: f64) -> Value {
    json!({"sine": {"offset": offset, "amplitude": amplitude, "frequency": frequency}})
}

/// Builds a grounded sinusoidal voltage source named `name` at `positive` over `0`.
pub(crate) fn voltage_sine(
    name: &str,
    positive: &str,
    offset: f64,
    amplitude: f64,
    frequency: f64,
) -> Instance {
    voltage_source(name, positive, "0", sine(offset, amplitude, frequency))
}
