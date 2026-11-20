//! Core circuit and device builders shared across engine integration tests.

use std::collections::BTreeMap;

use jolty_ir::{Circuit, Component, Instance, Name, Of, Parameters, Port, Role};
use serde_json::{Value, json};

/// Builds a circuit whose top component has ground port `0` and places `instances`.
pub(crate) fn circuit(instances: impl IntoIterator<Item = Instance>) -> Circuit {
    let top = Name::new("test").expect("valid top component name");
    let component = Component {
        ports: vec![Port {
            name: Name::new("0").expect("valid ground name"),
            role: Role::Ground,
        }],
        instances: instances.into_iter().collect(),
        statistical: None,
    };
    Circuit {
        components: BTreeMap::from([(top.clone(), component)]),
        top,
    }
}

/// Builds a device instance of `model` named `name`, with `parameters` and `terminals`.
pub(crate) fn device(
    name: &str,
    model: &str,
    parameters: Value,
    terminals: &[(&str, &str)],
) -> Instance {
    let parameters: Parameters = serde_json::from_value(parameters).expect("device parameters");
    let terminals = terminals
        .iter()
        .map(|&(terminal, net)| {
            (
                Name::new(terminal).expect("valid terminal name"),
                Name::new(net).expect("valid net name"),
            )
        })
        .collect();
    Instance {
        name: Name::new(name).expect("valid instance name"),
        of: Of::Device(Name::new(model).expect("valid model name")),
        parameters,
        terminals,
        tolerances: BTreeMap::new(),
        ratings: Vec::new(),
        part: None,
        source_location: None,
    }
}

/// Builds a two-terminal device of `model` named `name`, setting `parameter` to `value`
/// between `a` and `b`.
pub(crate) fn two_terminal(
    name: &str,
    model: &str,
    parameter: &str,
    value: f64,
    a: &str,
    b: &str,
) -> Instance {
    device(
        name,
        model,
        json!({parameter: value}),
        &[("a", a), ("b", b)],
    )
}

/// Builds a resistor named `name` of `resistance` ohms between `a` and `b`.
pub(crate) fn resistor(name: &str, resistance: f64, a: &str, b: &str) -> Instance {
    two_terminal(name, "resistor", "resistance", resistance, a, b)
}

/// Builds a voltage source named `name` driving `positive` over `negative` with `waveform`.
pub(crate) fn voltage_source(
    name: &str,
    positive: &str,
    negative: &str,
    waveform: Value,
) -> Instance {
    let mut parameters = serde_json::Map::new();
    parameters.insert("waveform".to_owned(), waveform);
    device(
        name,
        "voltage_source",
        Value::Object(parameters),
        &[("positive", positive), ("negative", negative)],
    )
}
