//! Current source builders shared across engine integration tests.

use jolty_ir::Instance;
use serde_json::{Value, json};

use crate::common::device;

/// A current source named `name` from `positive` to `negative` with `waveform`.
pub(crate) fn current_source(
    name: &str,
    positive: &str,
    negative: &str,
    waveform: Value,
) -> Instance {
    device(
        name,
        "current_source",
        json!({"waveform": waveform}),
        &[("positive", positive), ("negative", negative)],
    )
}

/// A DC current source named `name` pushing `current` amperes from `positive` to
/// `negative`.
pub(crate) fn current_dc(name: &str, positive: &str, negative: &str, current: f64) -> Instance {
    current_source(name, positive, negative, json!({"dc": current}))
}
