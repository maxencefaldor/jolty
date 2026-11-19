//! Current source builders shared across engine integration tests.

use jolty_ir::Instance;
use serde_json::{Value, json};

use crate::common::device;

/// Builds a current source named `name` from `positive` to `negative` with `waveform`.
pub(crate) fn current_source(
    name: &str,
    positive: &str,
    negative: &str,
    waveform: Value,
) -> Instance {
    let mut parameters = serde_json::Map::new();
    parameters.insert("waveform".to_owned(), waveform);
    device(
        name,
        "current_source",
        Value::Object(parameters),
        &[("positive", positive), ("negative", negative)],
    )
}

/// Builds a DC current source named `name` pushing `current` amperes from `positive` to
/// `negative`.
pub(crate) fn current_dc(name: &str, positive: &str, negative: &str, current: f64) -> Instance {
    current_source(name, positive, negative, json!({"dc": current}))
}
