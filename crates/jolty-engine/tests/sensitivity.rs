//! Adjoint sensitivities against finite differences of the whole circuit.

use jolty_engine::{Conditions, Options, Sensitivity, SimulationError, Simulator};
use jolty_ir::Circuit;
use jolty_models::Library;
use serde_json::{Value, json};

/// A circuit whose top component has a ground port `0` and places `instances`.
fn circuit_of(instances: &Value) -> Circuit {
    let document = json!({
        "format": "jolty-circuit", "version": 1, "top": "test",
        "components": {"test": {"ports": [{"name": "0", "role": "ground"}], "instances": instances}}
    });
    Circuit::from_json(&document.to_string()).expect("a valid circuit")
}

/// A source into a resistor and a diode with a series resistance, then a divider: nonlinear,
/// with an internal unknown.
fn diode() -> Value {
    json!([
        {"id": "V1", "of": {"device": "voltage_source"}, "parameters": {"waveform": {"dc": 5.0}},
         "terminals": {"positive": "in", "negative": "0"}},
        {"id": "R1", "of": {"device": "resistor"}, "parameters": {"resistance": 1e3},
         "terminals": {"a": "in", "b": "a"}},
        {"id": "D1", "of": {"device": "diode"},
         "parameters": {"saturation_current": 1e-14, "emission_coefficient": 1.2,
                        "series_resistance": 10.0},
         "terminals": {"anode": "a", "cathode": "0"}},
        {"id": "R2", "of": {"device": "resistor"}, "parameters": {"resistance": 2e3},
         "terminals": {"a": "a", "b": "out"}},
        {"id": "R3", "of": {"device": "resistor"}, "parameters": {"resistance": 3e3},
         "terminals": {"a": "out", "b": "0"}}
    ])
}

/// A common-source stage: an n-channel MOSFET biased into saturation, its drain loaded by
/// a resistor.
fn amplifier() -> Value {
    json!([
        {"id": "VDD", "of": {"device": "voltage_source"}, "parameters": {"waveform": {"dc": 3.3}},
         "terminals": {"positive": "vdd", "negative": "0"}},
        {"id": "VG", "of": {"device": "voltage_source"}, "parameters": {"waveform": {"dc": 1.2}},
         "terminals": {"positive": "gate", "negative": "0"}},
        {"id": "RD", "of": {"device": "resistor"}, "parameters": {"resistance": 1e4},
         "terminals": {"a": "vdd", "b": "drain"}},
        {"id": "M1", "of": {"device": "mosfet"},
         "parameters": {"width": 2e-6, "length": 1e-6, "threshold_voltage": 0.7,
                        "transconductance_parameter": 1e-4, "channel_length_modulation": 0.02},
         "terminals": {"drain": "drain", "gate": "gate", "source": "0", "bulk": "0"}}
    ])
}

fn tight() -> Options {
    Options {
        relative_tolerance: 1e-12,
        voltage_tolerance: 1e-15,
        current_tolerance: 1e-18,
        ..Options::default()
    }
}

/// The operating point of the circuit of `instances` with the parameter `name` of `device`
/// at `value`, and its simulator.
fn solved(instances: &Value, device: &str, name: &str, value: f64) -> (Simulator, Vec<f64>) {
    let mut parameters = instances
        .as_array()
        .and_then(|placed| placed.iter().find(|instance| instance["id"] == device))
        .expect("the device is placed")["parameters"]
        .clone();
    let slot = name
        .split('.')
        .try_fold(&mut parameters, |slot, key| slot.get_mut(key))
        .expect("the parameter is set");
    *slot = Value::from(value);
    let mut conditions = Conditions::default();
    conditions.overrides.insert(
        device.to_owned(),
        serde_json::from_value(parameters).expect("parameters"),
    );
    let mut simulator = Simulator::new(&circuit_of(instances), &Library::standard(), &conditions)
        .expect("an elaborated circuit");
    let values = simulator
        .operating_point(&tight())
        .expect("an operating point")
        .values;
    (simulator, values)
}

/// Checks that the adjoint sensitivities of `outputs` in the circuit of `instances` match
/// central finite differences of the whole circuit, solved again with each parameter moved,
/// and returns the parameters' names.
fn assert_matches(instances: &Value, outputs: &[&str]) -> Vec<String> {
    let circuit = circuit_of(instances);
    let mut simulator = Simulator::new(&circuit, &Library::standard(), &Conditions::default())
        .expect("an elaborated circuit");
    let outputs: Vec<String> = outputs.iter().map(|&output| output.to_owned()).collect();
    let sensitivity = Sensitivity {
        outputs: outputs.clone(),
    };
    let sensitivities = simulator
        .sensitivity(&sensitivity, &tight())
        .expect("sensitivities");
    for (index, parameter) in sensitivities.parameters.iter().enumerate() {
        let step = 1e-4 * parameter.value;
        let (device, name) = (&parameter.device, &parameter.name);
        let (moved, above) = solved(instances, device, name, parameter.value + step);
        let (_, below) = solved(instances, device, name, parameter.value - step);
        for (output, derivatives) in outputs.iter().zip(&sensitivities.derivatives) {
            let column = moved.unknown(output).expect("an unknown").index();
            let finite = (above[column] - below[column]) / (2.0 * step);
            let adjoint = derivatives[index];
            assert!(
                (adjoint - finite).abs() <= 1e-5 * finite.abs().max(1e-12),
                "d{output}/d{device}.{name}: adjoint {adjoint}, finite differences {finite}"
            );
        }
    }
    sensitivities
        .parameters
        .iter()
        .map(|parameter| format!("{}.{}", parameter.device, parameter.name))
        .collect()
}

#[test]
fn adjoint_sensitivities_match_finite_differences() {
    let names = assert_matches(&diode(), &["a", "out", "V1:current", "D1:junction"]);
    assert_eq!(
        names,
        [
            "V1.waveform.dc",
            "R1.resistance",
            "D1.emission_coefficient",
            "D1.saturation_current",
            "D1.series_resistance",
            "R2.resistance",
            "R3.resistance",
        ]
    );
    let names = assert_matches(&amplifier(), &["drain", "VDD:current"]);
    assert_eq!(names.len(), 8, "{names:?}");
}

#[test]
fn a_sensitivity_names_unknowns_that_exist() {
    let circuit = circuit_of(&diode());
    let mut simulator =
        Simulator::new(&circuit, &Library::standard(), &Conditions::default()).unwrap();
    let sensitivity = Sensitivity {
        outputs: vec!["nowhere".to_owned()],
    };
    let error = simulator
        .sensitivity(&sensitivity, &Options::default())
        .unwrap_err();
    assert_eq!(error, SimulationError::UnknownOutput("nowhere".to_owned()));
}
