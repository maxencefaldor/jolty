//! Every DC solution of circuits with one, three and a latch's.
#![cfg(test)]

use jolty_engine::{
    Conditions, OperatingPointResult, OperatingPoints, Options, SimulationError, Simulator,
};
use jolty_ir::Circuit;
use jolty_models::Library;
use serde_json::{Value, json};

/// A circuit of `instances` whose ground is `0`.
fn circuit(instances: &Value) -> Circuit {
    let document = json!({
        "format": "jolty-circuit", "version": 4, "top": "test",
        "components": {"test": {"ports": [{"name": "0", "role": "ground"}],
                                "instances": instances}}
    });
    Circuit::from_json(&document.to_string()).expect("a valid circuit")
}

fn source(id: &str, positive: &str, level: f64) -> Value {
    json!({"name": id, "of": {"model": "voltage_source"},
           "parameters": {"waveform": {"dc": level}},
           "terminals": {"positive": positive, "negative": "0"}})
}

fn resistor(id: &str, resistance: f64, a: &str, b: &str) -> Value {
    json!({"name": id, "of": {"model": "resistor"}, "parameters": {"resistance": resistance},
           "terminals": {"a": a, "b": b}})
}

fn search(
    circuit: &Circuit,
    most: usize,
) -> Result<(Simulator, Vec<OperatingPointResult>), SimulationError> {
    let mut simulator = Simulator::new(circuit, &Library::standard(), &Conditions::default())
        .expect("the circuit elaborates");
    let points = simulator.operating_points(&OperatingPoints { most }, &Options::default())?;
    Ok((simulator, points))
}

/// The value of `name` in each of `points`, sorted.
fn sorted(simulator: &Simulator, points: &[OperatingPointResult], name: &str) -> Vec<f64> {
    let unknown = simulator.unknown(name).expect("an unknown");
    let mut values: Vec<f64> = points
        .iter()
        .map(|point| point.values[unknown.index()])
        .collect();
    values.sort_by(f64::total_cmp);
    values
}

#[test]
fn a_divider_has_one_solution() {
    let divider = circuit(&json!([
        source("V1", "in", 5.0),
        resistor("R1", 1e3, "in", "out"),
        resistor("R2", 1e3, "out", "0"),
    ]));
    let (simulator, points) = search(&divider, 4).unwrap();
    assert_eq!(points.len(), 1);
    assert!((sorted(&simulator, &points, "out")[0] - 2.5).abs() < 1e-9);
}

#[test]
fn an_n_shaped_load_line_crosses_three_times() {
    // 1 mA/V³ · v (v − 1)(v − 2) = (1 V − v) / 10 kΩ, so (v − 1)(v² − 2v + 0.1) = 0.
    let cubic = circuit(&json!([
        source("V1", "in", 1.0),
        resistor("R1", 1e4, "in", "x"),
        {"name": "B1", "of": {"model": "behavioral_current_source"},
         "parameters": {"expression": "1e-3 * v(x) * (v(x) - 1) * (v(x) - 2)"},
         "terminals": {"positive": "x", "negative": "0", "x": "x"}},
    ]));
    let (simulator, points) = search(&cubic, 8).unwrap();
    let root = 0.9_f64.sqrt();
    let expected = [1.0 - root, 1.0, 1.0 + root];
    let found = sorted(&simulator, &points, "x");
    assert_eq!(found.len(), 3, "{found:?}");
    for (found, expected) in found.iter().zip(expected) {
        assert!((found - expected).abs() < 1e-6, "{found} vs {expected}");
    }
    // The first is the operating point, and a search stops at its most.
    let (_, first) = search(&cubic, 1).unwrap();
    assert_eq!(first.len(), 1);
    assert!(matches!(
        search(&cubic, 0),
        Err(SimulationError::InvalidAnalysis(_))
    ));
}

#[test]
fn a_latch_has_two_states_and_a_balance_between_them() {
    let inverter = |suffix: &str, input: &str, output: &str| {
        let channel = |id: String, kind: &str, drain: &str, source: &str| {
            let threshold = if kind == "n" { 0.7 } else { -0.7 };
            json!({"name": id, "of": {"model": "mosfet"},
                   "parameters": {"channel": kind, "threshold_voltage": threshold,
                                  "transconductance_parameter": 1e-4,
                                  "channel_length_modulation": 0.02,
                                  "width": 1e-5, "length": 1e-6},
                   "terminals": {"drain": drain, "gate": input, "source": source,
                                 "bulk": source}})
        };
        [
            channel(format!("MP{suffix}"), "p", output, "vdd"),
            channel(format!("MN{suffix}"), "n", output, "0"),
        ]
    };
    let [p1, n1] = inverter("1", "a", "b");
    let [p2, n2] = inverter("2", "b", "a");
    let latch = circuit(&json!([source("VDD", "vdd", 3.3), p1, n1, p2, n2]));
    let (simulator, points) = search(&latch, 8).unwrap();
    assert_eq!(points.len(), 3);
    let a = sorted(&simulator, &points, "a");
    let b = sorted(&simulator, &points, "b");
    for values in [&a, &b] {
        assert!(values[0].abs() < 1e-6, "{values:?}");
        assert!((values[1] - 1.65).abs() < 1e-6, "{values:?}");
        assert!((values[2] - 3.3).abs() < 1e-6, "{values:?}");
    }
}
