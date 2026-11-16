//! The loop gain through a probe against closed forms.
#![cfg(test)]

use std::f64::consts::TAU;

use jolty_engine::{Conditions, Options, SimulationError, Simulator, Stability};
use jolty_ir::Circuit;
use jolty_models::Library;
use jolty_numerics::Complex64;
use serde_json::json;

const GAIN: f64 = 1e3;
const OUTPUT: f64 = 100.0;
const UPPER: f64 = 9e3;
const LOWER: f64 = 1e3;
const CAPACITANCE: f64 = 1e-9;

/// An inverting amplifier of gain `GAIN` behind `OUTPUT` ohms, its output fed back to its
/// input through a divider of `UPPER` over `LOWER` with `CAPACITANCE` across the lower,
/// the probe `VP`, at `level`, between the amplifier's output, `y`, and the divider, `x`.
/// The nets are renamed by `names`, so the reference can be any of them.
fn amplifier(names: &[(&str, &str)], level: f64) -> Circuit {
    let net = |name: &str| {
        names
            .iter()
            .find(|(from, _)| *from == name)
            .map_or(name.to_owned(), |(_, to)| (*to).to_owned())
    };
    let instances = json!([
        {"name": "E1", "of": {"device": "voltage_controlled_voltage_source"},
         "parameters": {"gain": -GAIN},
         "terminals": {"positive": net("e"), "negative": net("g"),
                       "control_positive": net("in"), "control_negative": net("g")}},
        {"name": "RO", "of": {"device": "resistor"}, "parameters": {"resistance": OUTPUT},
         "terminals": {"a": net("e"), "b": net("y")}},
        {"name": "VP", "of": {"device": "voltage_source"}, "parameters": {"waveform": {"dc": level}},
         "terminals": {"positive": net("x"), "negative": net("y")}},
        {"name": "R1", "of": {"device": "resistor"}, "parameters": {"resistance": UPPER},
         "terminals": {"a": net("x"), "b": net("in")}},
        {"name": "R2", "of": {"device": "resistor"}, "parameters": {"resistance": LOWER},
         "terminals": {"a": net("in"), "b": net("g")}},
        {"name": "C1", "of": {"device": "capacitor"}, "parameters": {"capacitance": CAPACITANCE},
         "terminals": {"a": net("in"), "b": net("g")}},
        {"name": "IS", "of": {"device": "current_source"}, "parameters": {"waveform": {"dc": 0.0}},
         "terminals": {"positive": net("g"), "negative": net("in")}}
    ]);
    let document = json!({
        "format": "jolty-circuit", "version": 3, "top": "test",
        "components": {"test": {"ports": [{"name": "0", "role": "ground"}], "instances": instances}}
    });
    Circuit::from_json(&document.to_string()).expect("a valid circuit")
}

/// The loop gain: the amplifier's gain times the divider's, loaded by the output resistance.
fn expected(frequency: f64) -> Complex64 {
    let lower = 1.0 / Complex64::new(1.0 / LOWER, TAU * frequency * CAPACITANCE);
    GAIN * lower / (OUTPUT + UPPER + lower)
}

fn stability(circuit: &Circuit, probe: &str) -> Result<Vec<Complex64>, SimulationError> {
    let mut simulator = Simulator::new(circuit, &Library::standard(), &Conditions::default())
        .expect("the circuit elaborates");
    let analysis = Stability {
        probe: probe.to_owned(),
        frequencies: FREQUENCIES.to_vec(),
    };
    Ok(simulator.stability(&analysis, &Options::default())?.gains)
}

const FREQUENCIES: [f64; 5] = [0.0, 1e3, 1e5, 1e6, 1e8];

fn assert_matches(gains: &[Complex64]) {
    for (&frequency, &gain) in FREQUENCIES.iter().zip(gains) {
        let wanted = expected(frequency);
        assert!(
            (gain - wanted).norm() <= 1e-9 * wanted.norm().max(1.0),
            "at {frequency} Hz: {gain} vs {wanted}"
        );
    }
}

#[test]
fn the_loop_gain_is_the_amplifiers_times_the_loaded_feedbacks() {
    assert_matches(&stability(&amplifier(&[("g", "0")], 0.0), "VP").unwrap());
}

#[test]
fn a_probe_must_be_a_voltage_source() {
    let circuit = amplifier(&[("g", "0")], 0.0);
    assert_eq!(
        stability(&circuit, "IS"),
        Err(SimulationError::NotAProbe("IS".to_owned()))
    );
    assert_eq!(
        stability(&circuit, "R1"),
        Err(SimulationError::NotAProbe("R1".to_owned()))
    );
    assert_eq!(
        stability(&circuit, "VX"),
        Err(SimulationError::UnknownSource("VX".to_owned()))
    );
}

#[test]
fn a_probe_must_leave_the_reference_to_the_injected_current() {
    for side in ["x", "y"] {
        assert_eq!(
            stability(&amplifier(&[(side, "0")], 0.0), "VP"),
            Err(SimulationError::NotAProbe("VP".to_owned()))
        );
    }
}

#[test]
fn the_probes_own_level_moves_only_the_operating_point() {
    assert_matches(&stability(&amplifier(&[("g", "0")], 0.5), "VP").unwrap());
}
