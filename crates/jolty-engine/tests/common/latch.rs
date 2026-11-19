//! Cross-coupled CMOS latch fixture shared by `determinism.rs` and `operating_points.rs`.

use jolty_ir::Circuit;
use serde_json::json;

use crate::common::{circuit, device, voltage_source};

/// Two inverters in a loop, each a pair of MOSFETs: a latch, which holds either state and
/// balances between them.
pub(crate) fn latch() -> Circuit {
    let channel = |name: &str, kind: &str, (drain, gate, source): (&str, &str, &str)| {
        let threshold = if kind == "n" { 0.7 } else { -0.7 };
        device(
            name,
            "mosfet",
            json!({"channel": kind, "threshold_voltage": threshold,
                   "transconductance_parameter": 1e-4, "channel_length_modulation": 0.02,
                   "width": 1e-5, "length": 1e-6}),
            &[
                ("drain", drain),
                ("gate", gate),
                ("source", source),
                ("bulk", source),
            ],
        )
    };
    circuit([
        voltage_source("VDD", "vdd", "0", json!({"dc": 3.3})),
        channel("MP1", "p", ("b", "a", "vdd")),
        channel("MN1", "n", ("b", "a", "0")),
        channel("MP2", "p", ("a", "b", "vdd")),
        channel("MN2", "n", ("a", "b", "0")),
    ])
}
