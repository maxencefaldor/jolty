//! Poles and zeros against closed forms.
#![cfg(test)]

use jolty_engine::{Conditions, Options, PoleZero, PoleZeroResult, SimulationError, Simulator};
use jolty_ir::Circuit;
use jolty_models::Library;
use jolty_numerics::Complex64;
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

fn two_terminal(device: &str, id: &str, parameter: &str, value: f64, a: &str, b: &str) -> Value {
    json!({"name": id, "of": {"model": device}, "parameters": {parameter: value},
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
    json!({"name": id, "of": {"model": "voltage_source"},
           "parameters": {"waveform": {"dc": 1.0}},
           "terminals": {"positive": positive, "negative": "0"}})
}

fn pole_zero(
    circuit: &Circuit,
    source: Option<&str>,
    output: Option<&str>,
) -> Result<PoleZeroResult, SimulationError> {
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

/// A transconductance that feeds `node`'s own voltage back into it as current: a negative
/// conductance of `transconductance`.
fn negative_conductance(id: &str, transconductance: f64, node: &str) -> Value {
    json!({"name": id, "of": {"model": "voltage_controlled_current_source"},
           "parameters": {"transconductance": transconductance},
           "terminals": {"positive": "0", "negative": node,
                         "control_positive": node, "control_negative": "0"}})
}

fn growing_modes(instances: &Value) -> Vec<jolty_engine::Mode> {
    let mut simulator = Simulator::new(
        &circuit(instances),
        &Library::standard(),
        &Conditions::default(),
    )
    .expect("the circuit elaborates");
    simulator
        .growing_modes(&Options::default())
        .expect("the modes")
}

#[test]
fn a_negative_conductance_grows_and_a_lossless_resonator_does_not() {
    // 1 kΩ against a 2 mS negative conductance across 1 nF: s = (2e-3 − 1e-3) / 1e-9.
    let latch = growing_modes(&json!([
        resistor("R1", 1e3, "n", "0"),
        capacitor("C1", 1e-9, "n", "0"),
        negative_conductance("G1", 2e-3, "n"),
    ]));
    assert_eq!(latch.len(), 1, "{latch:?}");
    assert_roots(&[latch[0].pole], &[real(1e6)]);
    assert!((latch[0].shape[0].norm() - 1.0).abs() < 1e-12);
    // The same across 1 mH too: C s² − 1e-4 s + 1/L = 0, a pair that grows as it rings.
    let oscillator = growing_modes(&json!([
        resistor("R1", 1e4, "n", "0"),
        capacitor("C1", 1e-9, "n", "0"),
        inductor("L1", 1e-3, "n", "0"),
        negative_conductance("G1", 2e-4, "n"),
    ]));
    let (re, im) = (1e-4 / 2e-9, (4e-9_f64 / 1e-3 - 1e-8).sqrt() / 2e-9);
    assert_eq!(oscillator.len(), 1, "{oscillator:?}");
    assert_roots(&[oscillator[0].pole], &[Complex64::new(re, im)]);
    // Without the negative conductance and the resistor, the tank only rings.
    let tank = growing_modes(&json!([
        capacitor("C1", 1e-9, "n", "0"),
        inductor("L1", 1e-3, "n", "0"),
        source("V1", "s"),
        resistor("R1", 1e3, "s", "0"),
    ]));
    assert!(tank.is_empty(), "{tank:?}");
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

#[test]
fn a_qz_iteration_that_stalls_is_taken_again_on_the_reversed_pencil() {
    // A random circuit of 73 unknowns, of coupled windings among its devices, on whose
    // pencil faer 0.24's QZ iteration stops before its eigenvalues converge, without an
    // error; reversed, the pencil's iteration converges, as LAPACK's does on it.
    let circuit = Circuit::from_json(include_str!("data/stalled_qz.json")).expect("a circuit");
    let mut simulator = Simulator::new(&circuit, &Library::standard(), &Conditions::default())
        .expect("the circuit elaborates");
    assert_eq!(simulator.unknowns().len(), 73);
    // As tight as ngspice's, at whose operating point the iteration stalls.
    let tight = Options {
        relative_tolerance: 1e-6,
        voltage_tolerance: 1e-9,
        current_tolerance: 1e-14,
        ..Options::default()
    };
    assert_eq!(simulator.growing_modes(&tight).expect("the modes"), []);
}

#[test]
fn a_pencil_that_would_fill_the_memory_fails_at_once() {
    // A low-pass of three unknowns: in, V1:current and out, and the bordered pencil a row
    // more, of eight dense matrices of 16 numbers each, 1,024 bytes.
    let circuit = circuit(&json!([
        source("V1", "in"),
        resistor("R1", 1e3, "in", "out"),
        capacitor("C1", 1e-9, "out", "0"),
    ]));
    let mut simulator = Simulator::new(&circuit, &Library::standard(), &Conditions::default())
        .expect("the circuit elaborates");
    let options = Options {
        max_memory: 1e-9,
        ..Options::default()
    };
    let analysis = PoleZero {
        source: Some("V1".to_owned()),
        output: Some("out".to_owned()),
        reference: None,
    };
    let error = simulator
        .pole_zero(&analysis, &options)
        .expect_err("too little memory");
    assert_eq!(
        error,
        SimulationError::TooMuchMemory {
            analysis: "pole-zero analysis",
            rows: 4,
            needed: 128.0 / 134_217_728.0,
            max_memory: 1e-9,
        }
    );
    assert_eq!(
        error.to_string(),
        "the pole-zero analysis's dense matrices of 4 rows would hold 9.5367431640625e-7 \
         GiB, more than its `max_memory` option's 1e-9 GiB"
    );
    // The poles alone take the pencil of the unknowns, and so do the modes that grow.
    let poles = PoleZero::default();
    let rows = |error| match error {
        SimulationError::TooMuchMemory { rows, .. } => rows,
        other => panic!("{other:?}"),
    };
    assert_eq!(rows(simulator.pole_zero(&poles, &options).unwrap_err()), 3);
    assert_eq!(rows(simulator.growing_modes(&options).unwrap_err()), 3);
    assert_eq!(simulator.operating_point_solves(), 0);
    // A gibibyte holds it.
    assert!(simulator.pole_zero(&analysis, &Options::default()).is_ok());
}
