//! Adjoint sensitivities against finite differences of the whole circuit.
#![cfg(test)]

use jolty_engine::{
    Ac, Conditions, DcSweep, Derivatives, Differentiated, Method, Options, Sensitivity,
    SimulationError, Simulator, Transient,
};
use jolty_ir::Circuit;
use jolty_models::Library;
use jolty_numerics::Complex64;
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

/// The common-source stage loaded by a capacitor and driven through a resistor, so its
/// response falls with frequency.
fn loaded_amplifier() -> Value {
    let mut instances = amplifier();
    let placed = instances.as_array_mut().expect("instances");
    placed[1]["terminals"]["positive"] = json!("source");
    placed.push(json!({"id": "RG", "of": {"device": "resistor"},
        "parameters": {"resistance": 1e4}, "terminals": {"a": "source", "b": "gate"}}));
    placed.push(json!({"id": "CL", "of": {"device": "capacitor"},
        "parameters": {"capacitance": 1e-12}, "terminals": {"a": "drain", "b": "0"}}));
    placed.push(json!({"id": "CG", "of": {"device": "capacitor"},
        "parameters": {"capacitance": 2e-13}, "terminals": {"a": "gate", "b": "drain"}}));
    instances
}

/// A sine through a resistor into a diode and a capacitor across it: a rectifier, whose
/// waveforms bend where the diode conducts.
fn rectifier() -> Value {
    json!([
        {"id": "V1", "of": {"device": "voltage_source"},
         "parameters": {"waveform": {"sine": {"offset": 0.2, "amplitude": 1.0,
                                              "frequency": 1e6}}},
         "terminals": {"positive": "in", "negative": "0"}},
        {"id": "R1", "of": {"device": "resistor"}, "parameters": {"resistance": 1e3},
         "terminals": {"a": "in", "b": "a"}},
        {"id": "D1", "of": {"device": "diode"},
         "parameters": {"saturation_current": 1e-14, "series_resistance": 10.0,
                        "transit_time": 1e-9, "junction_capacitance": 2e-12},
         "terminals": {"anode": "a", "cathode": "out"}},
        {"id": "C1", "of": {"device": "capacitor"}, "parameters": {"capacitance": 1e-10},
         "terminals": {"a": "out", "b": "0"}},
        {"id": "R2", "of": {"device": "resistor"}, "parameters": {"resistance": 1e4},
         "terminals": {"a": "out", "b": "0"}}
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

/// The options `of` runs at: tight, and for a transient, with a truncation tolerance no
/// step's error reaches, so every run takes the same steps, growing to the longest, and
/// finite differences see what the parameters move, not where the steps fall.
fn options(of: &Differentiated) -> Options {
    match of {
        Differentiated::Transient(_) => Options {
            truncation_tolerance: f64::INFINITY,
            ..tight()
        },
        _ => tight(),
    }
}

/// The simulator of the circuit of `instances` with the parameter `name` of `device` at
/// `value`.
fn moved(instances: &Value, device: &str, name: &str, value: f64) -> Simulator {
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
    Simulator::new(&circuit_of(instances), &Library::standard(), &conditions)
        .expect("an elaborated circuit")
}

/// Each output's value at each point of the analysis `of`, on `simulator`.
fn values(simulator: &mut Simulator, of: &Differentiated, outputs: &[&str]) -> Vec<Vec<Complex64>> {
    let columns: Vec<usize> = outputs
        .iter()
        .map(|output| simulator.unknown(output).expect("an unknown").index())
        .collect();
    let pick = |rows: Vec<Vec<f64>>| -> Vec<Vec<Complex64>> {
        rows.iter()
            .map(|row| columns.iter().map(|&column| row[column].into()).collect())
            .collect()
    };
    match of {
        Differentiated::OperatingPoint => pick(vec![
            simulator
                .operating_point(&tight())
                .expect("a solution")
                .values,
        ]),
        Differentiated::DcSweep(sweep) => {
            pick(simulator.dc_sweep(sweep, &tight()).expect("a sweep").values)
        }
        Differentiated::Ac(ac) => simulator
            .ac(ac, &tight())
            .expect("a response")
            .values
            .iter()
            .map(|row| columns.iter().map(|&column| row[column]).collect())
            .collect(),
        Differentiated::Transient(transient) => pick(
            simulator
                .transient(transient, &options(of))
                .expect("waveforms")
                .values,
        ),
    }
}

/// The derivatives of a sensitivity analysis, by point, output and parameter, as phasors.
fn complex(derivatives: &Derivatives) -> Vec<Vec<Vec<Complex64>>> {
    match derivatives {
        Derivatives::Real(points) => points
            .iter()
            .map(|outputs| {
                outputs
                    .iter()
                    .map(|row| row.iter().map(|&value| value.into()).collect())
                    .collect()
            })
            .collect(),
        Derivatives::Complex(points) => points.clone(),
    }
}

/// Checks that the sensitivities of `outputs` along `of` in the circuit of `instances` match
/// central finite differences of the whole circuit, solved again with each parameter moved,
/// within `tolerance` of the largest derivative of each output and parameter, and returns
/// the parameters' names.
fn assert_matches(
    instances: &Value,
    outputs: &[&str],
    of: &Differentiated,
    tolerance: f64,
) -> Vec<String> {
    let circuit = circuit_of(instances);
    let mut simulator = Simulator::new(&circuit, &Library::standard(), &Conditions::default())
        .expect("an elaborated circuit");
    let sensitivity = Sensitivity {
        outputs: outputs.iter().map(|&output| output.to_owned()).collect(),
        of: of.clone(),
    };
    let sensitivities = simulator
        .sensitivity(&sensitivity, &options(of))
        .expect("sensitivities");
    let found = complex(&sensitivities.derivatives);
    for (index, parameter) in sensitivities.parameters.iter().enumerate() {
        let step = 1e-4 * parameter.value;
        let (device, name) = (&parameter.device, &parameter.name);
        let above = values(
            &mut moved(instances, device, name, parameter.value + step),
            of,
            outputs,
        );
        let below = values(
            &mut moved(instances, device, name, parameter.value - step),
            of,
            outputs,
        );
        assert_eq!(above.len(), found.len(), "{device}.{name} moved the points");
        for (output_index, output) in outputs.iter().enumerate() {
            let finite: Vec<Complex64> = above
                .iter()
                .zip(&below)
                .map(|(up, down)| (up[output_index] - down[output_index]) / (2.0 * step))
                .collect();
            let exact: Vec<Complex64> = found
                .iter()
                .map(|point| point[output_index][index])
                .collect();
            let largest = finite
                .iter()
                .chain(&exact)
                .fold(0.0_f64, |most, value| most.max(value.norm()));
            for (point, (exact, finite)) in exact.iter().zip(&finite).enumerate() {
                assert!(
                    (exact - finite).norm() <= tolerance * largest.max(1e-12),
                    "d{output}/d{device}.{name} at point {point}: exact {exact}, finite \
                     differences {finite}"
                );
            }
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
    let at = Differentiated::OperatingPoint;
    let names = assert_matches(
        &diode(),
        &["a", "out", "V1:current", "D1:junction"],
        &at,
        1e-5,
    );
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
    let names = assert_matches(&amplifier(), &["drain", "VDD:current"], &at, 1e-5);
    assert_eq!(names.len(), 8, "{names:?}");
}

#[test]
fn sensitivities_along_a_sweep_match_finite_differences() {
    let sweep = Differentiated::DcSweep(DcSweep {
        source: "V1".to_owned(),
        parameter: None,
        levels: (0..=10).map(|step| f64::from(step) * 0.5).collect(),
    });
    let names = assert_matches(&diode(), &["a", "V1:current"], &sweep, 1e-5);
    assert_eq!(names[0], "V1.waveform.dc");
}

#[test]
fn sensitivities_over_frequency_match_finite_differences() {
    let ac = Differentiated::Ac(Ac {
        source: "VG".to_owned(),
        frequencies: vec![1e3, 1e6, 1e8, 1e9],
    });
    assert_matches(&loaded_amplifier(), &["drain"], &ac, 1e-5);
}

#[test]
fn sensitivities_over_time_match_finite_differences() {
    let transient = Differentiated::Transient(Transient {
        stop: 2e-6,
        max_step: Some(1e-8),
        method: Method::Trapezoidal,
        times: Vec::new(),
        periodic: false,
    });
    assert_matches(&rectifier(), &["a", "out"], &transient, 1e-4);
    let gear = Differentiated::Transient(Transient {
        stop: 2e-6,
        max_step: Some(1e-8),
        method: Method::Gear2,
        times: Vec::new(),
        periodic: false,
    });
    assert_matches(&rectifier(), &["out"], &gear, 1e-4);
}

#[test]
fn a_sensitivity_names_unknowns_that_exist() {
    let circuit = circuit_of(&diode());
    let mut simulator =
        Simulator::new(&circuit, &Library::standard(), &Conditions::default()).unwrap();
    let sensitivity = Sensitivity {
        outputs: vec!["nowhere".to_owned()],
        of: Differentiated::OperatingPoint,
    };
    let error = simulator
        .sensitivity(&sensitivity, &Options::default())
        .unwrap_err();
    assert_eq!(error, SimulationError::UnknownOutput("nowhere".to_owned()));
}

#[test]
fn a_sensitivity_differentiates_no_periodic_steady_state() {
    let circuit = circuit_of(&rectifier());
    let mut simulator =
        Simulator::new(&circuit, &Library::standard(), &Conditions::default()).unwrap();
    let sensitivity = Sensitivity {
        outputs: vec!["out".to_owned()],
        of: Differentiated::Transient(Transient {
            stop: 1e-6,
            max_step: None,
            method: Method::Trapezoidal,
            times: Vec::new(),
            periodic: true,
        }),
    };
    let error = simulator
        .sensitivity(&sensitivity, &Options::default())
        .unwrap_err();
    assert!(
        matches!(error, SimulationError::InvalidAnalysis(_)),
        "{error}"
    );
}
