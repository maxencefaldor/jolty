//! The transfer function at DC against resistive circuits whose gains and resistances are
//! known in closed form.
#![cfg(test)]

mod common;
#[path = "common/current.rs"]
mod current;
#[path = "common/dc.rs"]
mod dc;
#[path = "common/inductor.rs"]
mod inductor;
#[path = "common/sim.rs"]
mod sim;

use common::{circuit, device, resistor, two_terminal};
use current::current_dc;
use dc::voltage_dc;
use inductor::inductor;
use jolty_engine::{Options, SimulationError, TransferFunction, TransferFunctionResult};
use jolty_ir::Circuit;
use serde_json::json;
use sim::simulator;

fn transfer(
    circuit: &Circuit,
    source: &str,
    output: &str,
    reference: Option<&str>,
) -> Result<TransferFunctionResult, SimulationError> {
    simulator(circuit).transfer_function(
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
    let divider = circuit([
        voltage_dc("V1", "in", 1.0),
        resistor("R1", 1e3, "in", "out"),
        resistor("R2", 3e3, "out", "0"),
    ]);
    let found = transfer(&divider, "V1", "out", None).unwrap();
    assert!(close(found.gain, 0.75), "{found:?}");
    assert!(close(found.input_resistance, 4e3), "{found:?}");
    // The source is a short to the output: R1 and R2 in parallel.
    assert!(close(found.output_resistance, 750.0), "{found:?}");
}

#[test]
fn a_current_source_sees_its_load_and_a_differential_output_its_branch() {
    // A current source drives `top`, through 1 kΩ to `mid` and 2 kΩ on to the ground.
    let loaded = circuit([
        current_dc("I1", "0", "top", 1e-3),
        resistor("R1", 1e3, "top", "mid"),
        resistor("R2", 2e3, "mid", "0"),
    ]);
    let found = transfer(&loaded, "I1", "top", Some("mid")).unwrap();
    assert!(close(found.gain, 1e3), "{found:?}");
    assert!(close(found.input_resistance, 3e3), "{found:?}");
    // The current source is open to the output: R1 alone, R2 leading nowhere else.
    assert!(close(found.output_resistance, 1e3), "{found:?}");
}

#[test]
fn a_voltage_sources_current_is_an_output_with_the_resistance_it_sees() {
    // V2, at 0 V, measures the current through R1 into `n`, and sees R1 back to the short V1.
    let sensed = circuit([
        voltage_dc("V1", "in", 1.0),
        resistor("R1", 1e3, "in", "n"),
        voltage_dc("V2", "n", 0.0),
        resistor("R2", 1e6, "in", "0"),
    ]);
    let found = transfer(&sensed, "V1", "V2:current", None).unwrap();
    assert!(close(found.gain, 1e-3), "{found:?}");
    assert!(
        close(found.input_resistance, 1e6 * 1e3 / (1e6 + 1e3)),
        "{found:?}"
    );
    assert!(close(found.output_resistance, 1e3), "{found:?}");
}

#[test]
fn an_output_the_analysis_cannot_read_is_an_error() {
    let devices = circuit([
        voltage_dc("V1", "in", 1.0),
        resistor("R1", 1e3, "in", "0"),
        inductor("L1", 1e-3, "in", "0"),
    ]);
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
    let unloaded = circuit([
        voltage_dc("V1", "in", 1.0),
        current_dc("I1", "0", "load", 1e-3),
        resistor("R1", 1e3, "load", "0"),
    ]);
    let found = transfer(&unloaded, "V1", "in", None).unwrap();
    assert_eq!(found.gain, 1.0, "{found:?}");
    assert_eq!(found.input_resistance, f64::INFINITY, "{found:?}");
    assert_eq!(found.output_resistance, 0.0, "{found:?}");
}

#[test]
fn ground_node_and_reference_validation_match_other_small_signal_analyses() {
    let divider = circuit([
        voltage_dc("V1", "in", 1.0),
        resistor("R1", 1e3, "in", "out"),
        resistor("R2", 3e3, "out", "0"),
    ]);
    let over_ground = transfer(&divider, "V1", "out", Some("0")).unwrap();
    assert!(close(over_ground.gain, 0.75), "{over_ground:?}");
    assert!(close(over_ground.input_resistance, 4e3), "{over_ground:?}");
    assert!(
        close(over_ground.output_resistance, 750.0),
        "{over_ground:?}"
    );

    let from_ground = transfer(&divider, "V1", "0", Some("out")).unwrap();
    assert!(close(from_ground.gain, -0.75), "{from_ground:?}");
    assert!(close(from_ground.input_resistance, 4e3), "{from_ground:?}");
    assert!(
        close(from_ground.output_resistance, 750.0),
        "{from_ground:?}"
    );

    // Output taken over itself, whether named node or ground.
    assert!(matches!(
        transfer(&divider, "V1", "out", Some("out")),
        Err(SimulationError::InvalidAnalysis(_))
    ));
    assert!(matches!(
        transfer(&divider, "V1", "0", None),
        Err(SimulationError::InvalidAnalysis(_))
    ));
    assert!(matches!(
        transfer(&divider, "V1", "0", Some("0")),
        Err(SimulationError::InvalidAnalysis(_))
    ));

    // Non-voltage reference and current output over the ground node.
    assert!(matches!(
        transfer(&divider, "V1", "out", Some("V1:current")),
        Err(SimulationError::InvalidAnalysis(_))
    ));
    assert!(matches!(
        transfer(&divider, "V1", "V1:current", Some("0")),
        Err(SimulationError::InvalidAnalysis(_))
    ));

    let heated = circuit([
        voltage_dc("V1", "top", 10.0),
        device(
            "R1",
            "resistor",
            json!({"resistance": 100.0, "temperature_coefficient": 4e-3, "thermal": true}),
            &[("a", "top"), ("b", "0"), ("thermal", "junction")],
        ),
        two_terminal(
            "RTH",
            "thermal_resistance",
            "resistance",
            50.0,
            "junction",
            "0",
        ),
    ]);
    assert!(matches!(
        transfer(&heated, "V1", "top", Some("junction")),
        Err(SimulationError::InvalidAnalysis(_))
    ));
    assert!(matches!(
        transfer(&heated, "V1", "junction", None),
        Err(SimulationError::InvalidAnalysis(_))
    ));
}
