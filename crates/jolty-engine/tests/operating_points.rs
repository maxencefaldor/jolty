//! Every DC solution of circuits with one, three and a latch's.
#![cfg(test)]

mod common;
#[path = "common/dc.rs"]
mod dc;
#[path = "common/latch.rs"]
mod latch;
#[path = "common/sim.rs"]
mod sim;

use common::{circuit, device, resistor};
use dc::voltage_dc;
use jolty_engine::{OperatingPointResult, OperatingPoints, Options, SimulationError, Simulator};
use jolty_ir::Circuit;
use latch::latch;
use serde_json::json;
use sim::simulator;

fn search(
    circuit: &Circuit,
    most: usize,
) -> Result<(Simulator, Vec<OperatingPointResult>), SimulationError> {
    let mut simulator = simulator(circuit);
    let points = simulator.operating_points(
        &OperatingPoints {
            max_solutions: most,
        },
        &Options::default(),
    )?;
    Ok((simulator, points))
}

/// Returns the sorted values of unknown `name` across `points`.
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
    let divider = circuit([
        voltage_dc("V1", "in", 5.0),
        resistor("R1", 1e3, "in", "out"),
        resistor("R2", 1e3, "out", "0"),
    ]);
    let (simulator, points) = search(&divider, 4).unwrap();
    assert_eq!(points.len(), 1);
    assert!((sorted(&simulator, &points, "out")[0] - 2.5).abs() < 1e-9);
}

#[test]
fn an_n_shaped_load_line_crosses_three_times() {
    // 1 mA/V³ · v (v - 1)(v - 2) = (1 V - v) / 10 kΩ, so (v - 1)(v² - 2v + 0.1) = 0.
    let cubic = circuit([
        voltage_dc("V1", "in", 1.0),
        resistor("R1", 1e4, "in", "x"),
        device(
            "B1",
            "behavioral_current_source",
            json!({"expression": "1e-3 * v(x) * (v(x) - 1) * (v(x) - 2)"}),
            &[("positive", "x"), ("negative", "0"), ("x", "x")],
        ),
    ]);
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
    let (simulator, points) = search(&latch(), 8).unwrap();
    assert_eq!(points.len(), 3);
    let a = sorted(&simulator, &points, "a");
    let b = sorted(&simulator, &points, "b");
    for values in [&a, &b] {
        assert!(values[0].abs() < 1e-6, "{values:?}");
        assert!((values[1] - 1.65).abs() < 1e-6, "{values:?}");
        assert!((values[2] - 3.3).abs() < 1e-6, "{values:?}");
    }
}
