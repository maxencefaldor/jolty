//! Poles and zeros against closed forms.

use jolty_engine::{Conditions, Options, PoleZero, PolesZeros, SimulationError, Simulator};
use jolty_ir::Circuit;
use jolty_models::Library;
use jolty_numerics::Complex64;
use serde_json::{Value, json};

/// A circuit of `instances` whose ground is `0`.
fn circuit(instances: &Value) -> Circuit {
    let document = json!({
        "format": "jolty-circuit", "version": 1, "top": "test",
        "components": {"test": {"ports": [{"name": "0", "role": "ground"}],
                                "instances": instances}}
    });
    Circuit::from_json(&document.to_string()).expect("a valid circuit")
}

fn two_terminal(device: &str, id: &str, parameter: &str, value: f64, a: &str, b: &str) -> Value {
    json!({"id": id, "of": {"device": device}, "parameters": {parameter: value},
           "terminals": {"a": a, "b": b}})
}

fn resistor(id: &str, resistance: f64, a: &str, b: &str) -> Value {
    two_terminal("resistor", id, "resistance", resistance, a, b)
}

fn capacitor(id: &str, capacitance: f64, a: &str, b: &str) -> Value {
    two_terminal("capacitor", id, "capacitance", capacitance, a, b)
}

fn inductor(id: &str, inductance: f64, a: &str, b: &str) -> Value {
    two_terminal("inductor", id, "inductance", inductance, a, b)
}

fn source(id: &str, positive: &str) -> Value {
    json!({"id": id, "of": {"device": "voltage_source"},
           "parameters": {"waveform": {"dc": 1.0}},
           "terminals": {"positive": positive, "negative": "0"}})
}

fn pole_zero(
    circuit: &Circuit,
    source: Option<&str>,
    output: Option<&str>,
) -> Result<PolesZeros, SimulationError> {
    let mut simulator = Simulator::new(circuit, &Library::standard(), &Conditions::default())
        .expect("the circuit elaborates");
    let analysis = PoleZero {
        source: source.map(str::to_owned),
        output: output.map(str::to_owned),
        reference: None,
    };
    simulator.pole_zero(&analysis, &Options::default())
}

/// Asserts `found` are `expected`, in order, each within a relative 1e-9 of its size, and
/// a root at the origin exactly.
fn assert_roots(found: &[Complex64], expected: &[Complex64]) {
    assert_eq!(found.len(), expected.len(), "{found:?} vs {expected:?}");
    for (found, expected) in found.iter().zip(expected) {
        assert!(
            (found - expected).norm() <= 1e-9 * expected.norm(),
            "{found} vs {expected}"
        );
    }
}

fn real(value: f64) -> Complex64 {
    Complex64::new(value, 0.0)
}

#[test]
fn a_low_pass_has_one_pole_and_a_high_pass_a_zero_at_the_origin() {
    let (resistance, capacitance) = (1e3, 1e-9);
    let pole = real(-1.0 / (resistance * capacitance));
    let low = circuit(&json!([
        source("V1", "in"),
        resistor("R1", resistance, "in", "out"),
        capacitor("C1", capacitance, "out", "0"),
    ]));
    let found = pole_zero(&low, Some("V1"), Some("out")).unwrap();
    assert_roots(&found.poles, &[pole]);
    assert_roots(&found.zeros.unwrap(), &[]);
    let high = circuit(&json!([
        source("V1", "in"),
        capacitor("C1", capacitance, "in", "out"),
        resistor("R1", resistance, "out", "0"),
    ]));
    let found = pole_zero(&high, Some("V1"), Some("out")).unwrap();
    assert_roots(&found.poles, &[pole]);
    assert_roots(&found.zeros.unwrap(), &[real(0.0)]);
    // Without a transfer function, the poles alone.
    let alone = pole_zero(&high, None, None).unwrap();
    assert_roots(&alone.poles, &[pole]);
    assert!(alone.zeros.is_none());
}

#[test]
fn a_series_resonator_has_a_conjugate_pair() {
    let (resistance, inductance, capacitance) = (100.0, 1e-3, 1e-9);
    let resonator = circuit(&json!([
        source("V1", "in"),
        resistor("R1", resistance, "in", "a"),
        inductor("L1", inductance, "a", "out"),
        capacitor("C1", capacitance, "out", "0"),
    ]));
    // L·C·s² + R·C·s + 1 = 0.
    let damping = -resistance / (2.0 * inductance);
    let ringing = (1.0 / (inductance * capacitance) - damping * damping).sqrt();
    let pair = [
        Complex64::new(damping, ringing),
        Complex64::new(damping, -ringing),
    ];
    let across = pole_zero(&resonator, Some("V1"), Some("out")).unwrap();
    assert_roots(&across.poles, &pair);
    assert_roots(&across.zeros.unwrap(), &[]);
    // Across the resistor, the current's zero at the origin.
    let current = pole_zero(&resonator, Some("V1"), Some("V1:current")).unwrap();
    assert_roots(&current.zeros.unwrap(), &[real(0.0)]);
}

#[test]
fn a_ladder_matches_ngspice() {
    // ngspice 42's `pz in 0 mid 0 vol pz` on the same circuit.
    let ladder = circuit(&json!([
        source("V1", "in"),
        resistor("R1", 1e3, "in", "out"),
        capacitor("C1", 1e-9, "out", "0"),
        resistor("R2", 1e3, "out", "mid"),
        capacitor("C2", 1e-9, "mid", "0"),
        inductor("L1", 1e-3, "mid", "0"),
    ]));
    let found = pole_zero(&ladder, Some("V1"), Some("mid")).unwrap();
    let pair = [
        Complex64::new(-2.393_101_465_977_163e5, 8.578_736_265_951_786e5),
        Complex64::new(-2.393_101_465_977_163e5, -8.578_736_265_951_786e5),
    ];
    assert_roots(
        &found.poles,
        &[pair[0], pair[1], real(-2.521_379_706_804_567e6)],
    );
    assert_roots(&found.zeros.unwrap(), &[real(0.0)]);
}

#[test]
fn a_long_rc_ladder_has_every_pole_its_closed_form_gives() {
    // 120 stages of 1 kΩ into 1 nF: the poles are −(2/RC)(1 − cos((2k − 1)π / 241)).
    let stages: u32 = 120;
    let mut instances = vec![source("V1", "n0")];
    for stage in 1..=stages {
        let (from, to) = (format!("n{}", stage - 1), format!("n{stage}"));
        instances.push(resistor(&format!("R{stage}"), 1e3, &from, &to));
        instances.push(capacitor(&format!("C{stage}"), 1e-9, &to, "0"));
    }
    let found =
        pole_zero(&circuit(&Value::Array(instances)), None, None).expect("a pole-zero analysis");
    let mut expected: Vec<Complex64> = (1..=stages)
        .map(|k| {
            let angle = f64::from(2 * k - 1) * std::f64::consts::PI / f64::from(2 * stages + 1);
            real(-2e6 * (1.0 - angle.cos()))
        })
        .collect();
    expected.sort_by(|a, b| b.re.total_cmp(&a.re));
    assert_roots(&found.poles, &expected);
}

#[test]
fn a_resistive_circuit_has_no_poles() {
    let divider = circuit(&json!([
        source("V1", "in"),
        resistor("R1", 1e3, "in", "out"),
        resistor("R2", 1e3, "out", "0"),
    ]));
    let found = pole_zero(&divider, Some("V1"), Some("out")).unwrap();
    assert!(found.poles.is_empty() && found.zeros.unwrap().is_empty());
}

#[test]
fn rejects_what_it_cannot_measure() {
    let apart = circuit(&json!([
        source("V1", "in"),
        resistor("R1", 1e3, "in", "out"),
        capacitor("C1", 1e-9, "out", "0"),
        source("V2", "other"),
        resistor("R2", 1e3, "other", "far"),
        capacitor("C2", 1e-9, "far", "0"),
    ]));
    for (source, output) in [(Some("V1"), None), (None, Some("out"))] {
        assert!(matches!(
            pole_zero(&apart, source, output),
            Err(SimulationError::InvalidAnalysis(_))
        ));
    }
    assert!(matches!(
        pole_zero(&apart, Some("V1"), Some("far")),
        Err(SimulationError::InvalidAnalysis(message)) if message.contains("zero at every")
    ));
    assert_eq!(
        pole_zero(&apart, Some("R1"), Some("out")).unwrap_err(),
        SimulationError::NotASource("R1".to_owned())
    );
    assert_eq!(
        pole_zero(&apart, Some("V1"), Some("nowhere")).unwrap_err(),
        SimulationError::UnknownOutput("nowhere".to_owned())
    );
}
