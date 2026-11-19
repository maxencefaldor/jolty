//! Rectifier fixture, finite-difference perturbation, and tight solver options shared by
//! `objective.rs` and `sensitivity.rs`.

use jolty_engine::{Conditions, Options, Simulator};
use jolty_ir::{Circuit, Instance};
use jolty_models::Library;
use serde_json::{Value, json};

use crate::common::{device, resistor, two_terminal, voltage_source};

/// Builds a diode rectifier driven by a 1 MHz sine wave through a resistor into an RC load.
pub(crate) fn rectifier() -> Vec<Instance> {
    vec![
        voltage_source(
            "V1",
            "in",
            "0",
            json!({"sine": {"offset": 0.2, "amplitude": 1.0, "frequency": 1e6}}),
        ),
        resistor("R1", 1e3, "in", "a"),
        device(
            "D1",
            "diode",
            json!({"saturation_current": 1e-14, "series_resistance": 10.0,
                   "transit_time": 1e-9, "junction_capacitance": 2e-12}),
            &[("anode", "a"), ("cathode", "out")],
        ),
        two_terminal("C1", "capacitor", "capacitance", 1e-10, "out", "0"),
        resistor("R2", 1e4, "out", "0"),
    ]
}

/// Returns tight solver options, disabling truncation step control when `fixed_steps` is
/// true so finite differences isolate parameter shifts rather than step placement.
pub(crate) fn tight_options(fixed_steps: bool) -> Options {
    Options {
        relative_tolerance: 1e-12,
        voltage_tolerance: 1e-15,
        current_tolerance: 1e-18,
        truncation_tolerance: if fixed_steps {
            f64::INFINITY
        } else {
            Options::default().truncation_tolerance
        },
        ..Options::default()
    }
}

/// Builds a simulator for `circuit` with parameter `name` of `device_name` overridden to
/// `value`.
pub(crate) fn moved(circuit: &Circuit, device_name: &str, name: &str, value: f64) -> Simulator {
    let mut parameters = circuit.components[&circuit.top]
        .instances
        .iter()
        .find(|instance| instance.name.as_str() == device_name)
        .expect("the device is placed")
        .parameters
        .clone();
    let slot = jolty_ir::parameter_mut(&mut parameters, name).expect("the parameter is set");
    *slot = Value::from(value);
    let mut conditions = Conditions::default();
    conditions
        .overrides
        .insert(device_name.to_owned(), parameters);
    Simulator::new(circuit, &Library::standard(), &conditions).expect("an elaborated circuit")
}
