//! Circuits that Newton iteration alone does not solve, and the aid that does.
#![cfg(test)]

use jolty_engine::{Conditions, Options, Simulator, Strategy};
use jolty_ir::Circuit;
use jolty_models::Library;
use serde_json::{Value, json};

/// A circuit of `instances` whose ground is `0`.
fn circuit(instances: &Value) -> Circuit {
    let document = json!({
        "format": "jolty-circuit", "version": 3, "top": "test",
        "components": {"test": {"ports": [{"name": "0", "role": "ground"}],
                                "instances": instances}}
    });
    Circuit::from_json(&document.to_string()).expect("a valid circuit")
}

#[test]
fn gmin_stepping_across_the_junctions_saturates_a_transistor_into_a_negative_load() {
    // A transconductance from the emitter to itself pushes 97 µA per volt into it, and
    // another feeds back the collector's voltage; the diode keeps the emitter above the
    // ground. Newton iteration fails, and so does a shunt stepped down at every node, as
    // in ngspice, whose true gmin stepping finds the same point.
    let circuit = circuit(&json!([
        {"name": "V1", "of": {"device": "voltage_source"},
         "parameters": {"waveform": {"dc": 4.8}},
         "terminals": {"positive": "base", "negative": "0"}},
        {"name": "R1", "of": {"device": "resistor"}, "parameters": {"resistance": 230e3},
         "terminals": {"a": "low", "b": "base"}},
        {"name": "I1", "of": {"device": "current_source"},
         "parameters": {"waveform": {"dc": 0.2e-3}},
         "terminals": {"positive": "low", "negative": "0"}},
        {"name": "G1", "of": {"device": "voltage_controlled_current_source"},
         "parameters": {"transconductance": 1.75e-6},
         "terminals": {"positive": "0", "negative": "emitter",
                       "control_positive": "collector", "control_negative": "low"}},
        {"name": "D1", "of": {"device": "diode"}, "parameters": {},
         "terminals": {"anode": "0", "cathode": "emitter"}},
        {"name": "Q1", "of": {"device": "bipolar_transistor"}, "parameters": {"polarity": "pnp"},
         "terminals": {"collector": "collector", "base": "base", "emitter": "emitter"}},
        {"name": "G2", "of": {"device": "voltage_controlled_current_source"},
         "parameters": {"transconductance": 97e-6},
         "terminals": {"positive": "0", "negative": "emitter",
                       "control_positive": "emitter", "control_negative": "0"}},
    ]));
    let mut simulator = Simulator::new(&circuit, &Library::standard(), &Conditions::default())
        .expect("the circuit elaborates");
    let point = simulator
        .operating_point(&Options::default())
        .expect("an operating point");
    assert_eq!(point.strategy, Strategy::JunctionGminStepping);
    let voltage = |name: &str| point.values[simulator.unknown(name).expect("a node").index()];
    let (base, emitter, collector) = (voltage("base"), voltage("emitter"), voltage("collector"));
    // ngspice's operating point, to its printed digits.
    assert!(
        (emitter - 5.579_408).abs() < 1e-5,
        "the emitter at {emitter} V"
    );
    assert!(
        (collector - 5.561_480).abs() < 1e-5,
        "the collector at {collector} V"
    );
    assert!(
        collector > base,
        "a saturated transistor's collector junction conducts"
    );
}

#[test]
fn the_operating_point_source_stepping_folds_short_of_is_ngspices() {
    // A random circuit of 47 devices, shrunk from one whose operating point every other
    // strategy missed: source stepping stops at a fold of its solutions, at 51 % of the
    // sources, as ngspice's does, which finds the point only by its transient instead;
    // pseudo-transient continuation finds it too, as a unit test of it shows. Which of the
    // strategies before it converges first turns on the rounding of the sums of the
    // circuit's equations, which the order of their terms moves.
    let circuit = Circuit::from_json(include_str!("data/folded_sources.json")).expect("a circuit");
    let mut simulator = Simulator::new(&circuit, &Library::standard(), &Conditions::default())
        .expect("the circuit elaborates");
    let point = simulator
        .operating_point(&Options::default())
        .expect("an operating point");
    assert_ne!(point.strategy, Strategy::SourceStepping);
    let voltage = |name: &str| point.values[simulator.unknown(name).expect("a node").index()];
    // ngspice's transient operating point, to its printed digits.
    for (node, expected) in [
        ("n20", -54.634_247),
        ("n16", -1.998_355),
        ("n7", -1.765_709),
    ] {
        let found = voltage(node);
        assert!((found - expected).abs() < 1e-5, "{node} at {found} V");
    }
}
