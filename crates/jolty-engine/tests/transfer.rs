//! The transfer function at DC against resistive circuits whose gains and resistances are
//! known in closed form.
#![cfg(test)]

use jolty_engine::{
    Conditions, Options, SimulationError, Simulator, TransferFunction, TransferFunctionResult,
};
use jolty_ir::Circuit;
use jolty_models::Library;
use serde_json::{Value, json};

/// A device instance: its name, model, parameters, and the net of each terminal.
type Device<'a> = (&'a str, &'a str, Value, &'a [(&'a str, &'a str)]);

/// A circuit whose top component has a ground port `0` and places `devices`.
fn circuit(devices: &[Device<'_>]) -> Circuit {
    let instances: Vec<Value> = devices
        .iter()
        .map(|(id, model, parameters, terminals)| {
            let terminals: serde_json::Map<String, Value> = terminals
                .iter()
                .map(|(terminal, net)| ((*terminal).to_owned(), json!(net)))
                .collect();
            json!({"name": id, "of": {"model": model}, "parameters": parameters, "terminals": terminals})
        })
        .collect();
    let document = json!({
        "format": "jolty-circuit", "version": 4, "top": "test",
        "components": {"test": {"ports": [{"name": "0", "role": "ground"}], "instances": instances}}
    });
    Circuit::from_json(&document.to_string()).expect("a valid circuit file")
}

fn dc(level: f64) -> Value {
    json!({"waveform": {"dc": level}})
}

fn resistor(id: &'static str, ohms: f64, a: &'static str, b: &'static str) -> Device<'static> {
    let terminals: &'static [(&str, &str)] = Box::leak(Box::new([("a", a), ("b", b)]));
    (id, "resistor", json!({"resistance": ohms}), terminals)
}

fn transfer(
    devices: &[Device<'_>],
    source: &str,
    output: &str,
    reference: Option<&str>,
) -> Result<TransferFunctionResult, SimulationError> {
    let mut simulator = Simulator::new(
        &circuit(devices),
        &Library::standard(),
        &Conditions::default(),
    )
    .expect("an elaborated circuit");
    simulator.transfer_function(
        &TransferFunction {
            source: source.to_owned(),
            output: output.to_owned(),
            reference: reference.map(str::to_owned),
        },
        &Options::default(),
    )
}

/// Agreement within rounding: each solve is a handful of operations on numbers of order 1
/// to 10⁴, so a few ulps of the result, 1e-12 of it, bound the difference.
fn close(found: f64, exact: f64) -> bool {
    (found - exact).abs() <= 1e-12 * exact.abs()
}

#[test]
fn a_divider_halves_by_its_ratio_and_shows_its_resistances() {
    let found = transfer(
        &[
            (
                "V1",
                "voltage_source",
                dc(1.0),
                &[("positive", "in"), ("negative", "0")],
            ),
            resistor("R1", 1e3, "in", "out"),
            resistor("R2", 3e3, "out", "0"),
        ],
        "V1",
        "out",
        None,
    )
    .unwrap();
    assert!(close(found.gain, 0.75), "{found:?}");
    assert!(close(found.input_resistance, 4e3), "{found:?}");
    // The source is a short to the output: R1 and R2 in parallel.
    assert!(close(found.output_resistance, 750.0), "{found:?}");
}

#[test]
fn a_current_source_sees_its_load_and_a_differential_output_its_branch() {
    // A current source drives `top`, through 1 kΩ to `mid` and 2 kΩ on to the ground.
    let devices = [
        (
            "I1",
            "current_source",
            dc(1e-3),
            &[("positive", "0"), ("negative", "top")][..],
        ),
        resistor("R1", 1e3, "top", "mid"),
        resistor("R2", 2e3, "mid", "0"),
    ];
    let found = transfer(&devices, "I1", "top", Some("mid")).unwrap();
    assert!(close(found.gain, 1e3), "{found:?}");
    assert!(close(found.input_resistance, 3e3), "{found:?}");
    // The current source is open to the output: R1 alone, R2 leading nowhere else.
    assert!(close(found.output_resistance, 1e3), "{found:?}");
}

#[test]
fn a_voltage_sources_current_is_an_output_with_the_resistance_it_sees() {
    // V2, at 0 V, measures the current through R1 into `n`, and sees R1 back to the short V1.
    let found = transfer(
        &[
            (
                "V1",
                "voltage_source",
                dc(1.0),
                &[("positive", "in"), ("negative", "0")],
            ),
            resistor("R1", 1e3, "in", "n"),
            (
                "V2",
                "voltage_source",
                dc(0.0),
                &[("positive", "n"), ("negative", "0")],
            ),
            resistor("R2", 1e6, "in", "0"),
        ],
        "V1",
        "V2:current",
        None,
    )
    .unwrap();
    assert!(close(found.gain, 1e-3), "{found:?}");
    assert!(
        close(found.input_resistance, 1e6 * 1e3 / (1e6 + 1e3)),
        "{found:?}"
    );
    assert!(close(found.output_resistance, 1e3), "{found:?}");
}

#[test]
fn an_output_the_analysis_cannot_read_is_an_error() {
    let devices = [
        (
            "V1",
            "voltage_source",
            dc(1.0),
            &[("positive", "in"), ("negative", "0")][..],
        ),
        resistor("R1", 1e3, "in", "0"),
        (
            "L1",
            "inductor",
            json!({"inductance": 1e-3}),
            &[("a", "in"), ("b", "0")][..],
        ),
    ];
    assert!(matches!(
        transfer(&devices, "V1", "nowhere", None),
        Err(SimulationError::UnknownOutput(name)) if name == "nowhere"
    ));
    assert!(matches!(
        transfer(&devices, "R1", "in", None),
        Err(SimulationError::NotASource(name)) if name == "R1"
    ));
    assert!(matches!(
        transfer(&devices, "V1", "V1:current", Some("in")),
        Err(SimulationError::InvalidAnalysis(_))
    ));
    assert!(matches!(
        transfer(&devices, "V1", "L1:current", None),
        Err(SimulationError::InvalidAnalysis(_))
    ));
}

#[test]
fn a_source_that_drives_nothing_sees_an_infinite_resistance() {
    // V1 holds `in`, which nothing else joins; R1 is a load of its own elsewhere.
    let found = transfer(
        &[
            (
                "V1",
                "voltage_source",
                dc(1.0),
                &[("positive", "in"), ("negative", "0")],
            ),
            (
                "I1",
                "current_source",
                dc(1e-3),
                &[("positive", "0"), ("negative", "load")],
            ),
            resistor("R1", 1e3, "load", "0"),
        ],
        "V1",
        "in",
        None,
    )
    .unwrap();
    assert_eq!(found.gain, 1.0, "{found:?}");
    assert_eq!(found.input_resistance, f64::INFINITY, "{found:?}");
    assert_eq!(found.output_resistance, 0.0, "{found:?}");
}
