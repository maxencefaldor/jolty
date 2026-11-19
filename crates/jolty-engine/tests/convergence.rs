//! Circuits that Newton iteration alone does not solve, and the aid that does.
#![cfg(test)]

mod common;
#[path = "common/current.rs"]
mod current;
#[path = "common/dc.rs"]
mod dc;
#[path = "common/sim.rs"]
mod sim;

use common::{circuit, device, resistor};
use current::current_dc;
use dc::voltage_dc;
use jolty_engine::{Options, Strategy};
use jolty_ir::Circuit;
use serde_json::json;
use sim::simulator;

#[test]
fn gmin_stepping_across_the_junctions_saturates_a_transistor_into_a_negative_load() {
    // A transconductance from the emitter to itself pushes 97 µA per volt into it, and
    // another feeds back the collector's voltage; the diode keeps the emitter above the
    // ground. Newton iteration fails, and so does a shunt stepped down at every node, as
    // in ngspice, whose true gmin stepping finds the same point.
    let circuit = circuit([
        voltage_dc("V1", "base", 4.8),
        resistor("R1", 230e3, "low", "base"),
        current_dc("I1", "low", "0", 0.2e-3),
        device(
            "G1",
            "voltage_controlled_current_source",
            json!({"transconductance": 1.75e-6}),
            &[
                ("positive", "0"),
                ("negative", "emitter"),
                ("control_positive", "collector"),
                ("control_negative", "low"),
            ],
        ),
        device(
            "D1",
            "diode",
            json!({}),
            &[("anode", "0"), ("cathode", "emitter")],
        ),
        device(
            "Q1",
            "bipolar_transistor",
            json!({"polarity": "pnp"}),
            &[
                ("collector", "collector"),
                ("base", "base"),
                ("emitter", "emitter"),
            ],
        ),
        device(
            "G2",
            "voltage_controlled_current_source",
            json!({"transconductance": 97e-6}),
            &[
                ("positive", "0"),
                ("negative", "emitter"),
                ("control_positive", "emitter"),
                ("control_negative", "0"),
            ],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    let point = simulator
        .operating_point(&Options::default())
        .expect("an operating point");
    assert_eq!(point.strategy, Strategy::JunctionGminStepping);
    // Total iterations include strategies that failed before convergence: plain Newton runs
    // to its limit or hits a singular matrix.
    let options = Options::default();
    assert!(
        point.iterations > options.max_dc_iterations,
        "{} iterations",
        point.iterations
    );
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
    // Shrunk 47-device random circuit whose operating point defeats source stepping at a
    // solution fold around 51% of the sources (matching ngspice, falling back to transient).
    // Pseudo-transient continuation finds the operating point as well. Rounding in equation
    // sums determines which earlier strategy converges first.
    let circuit = Circuit::from_json(include_str!("data/folded_sources.json")).expect("a circuit");
    let mut simulator = simulator(&circuit);
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
