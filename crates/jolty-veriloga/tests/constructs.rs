//! The constructs of the CMC's models, each in a fixture of `tests/fixtures/` that the
//! compiler writes as the library's kernels: analog functions inlined, named blocks,
//! `tanh` and `atan`, `$warning`, output variables, the thermal discipline, the
//! simulator's own limiting, initial events, `floor` and `ceil`, `$finish`, shorts that
//! join nodes, and the noise of a branch whose current another reads.

#![cfg(test)]

#[path = "common/closed.rs"]
mod closed;
mod common;

#[rustfmt::skip]
#[expect(unreachable_pub, reason = "a fixture is a kernel as the compiler writes it for a library")]
#[path = "fixtures/constructs.rs"]
mod constructs;
#[rustfmt::skip]
#[expect(unreachable_pub, reason = "a fixture is a kernel as the compiler writes it for a library")]
#[path = "fixtures/heated_resistor.rs"]
mod heated_resistor;
#[rustfmt::skip]
#[expect(unreachable_pub, reason = "a fixture is a kernel as the compiler writes it for a library")]
#[path = "fixtures/initial_events.rs"]
mod initial_events;
#[rustfmt::skip]
#[expect(unreachable_pub, reason = "a fixture is a kernel as the compiler writes it for a library")]
#[path = "fixtures/limited_diode.rs"]
mod limited_diode;
#[rustfmt::skip]
#[expect(unreachable_pub, reason = "a fixture is a kernel as the compiler writes it for a library")]
#[path = "fixtures/networks.rs"]
mod networks;
#[rustfmt::skip]
#[expect(unreachable_pub, reason = "a fixture is a kernel as the compiler writes it for a library")]
#[path = "fixtures/thermal_resistance.rs"]
mod thermal_resistance;

use closed::Closed;
use common::Worst;
use jolty_models::testing::{
    Bench, Rig, assert_lift_matches_serde, assert_lift_matches_serde_with_optionals,
};
use jolty_models::{
    Differentiable, Diode, Input, Kernel, REFERENCE_TEMPERATURE, ThermalResistance,
};

/// How far a value computed by the compiled kernel may lie from its closed form computed
/// here, in units of the machine epsilon times its magnitude: each computes it in at most
/// eight operations, each rounding by half an epsilon, in its own order.
const CLOSED: Closed = Closed(8.0);

/// An inlined analog function and a named block's variables compute the saturating
/// conductance's closed form, `g · vs · tanh(v / vs) + b · g · vs · atan(v / vs)`, and its
/// derivative, `g · (1 − tanh²(v / vs)) + b · g / (1 + (v / vs)²)`.
#[test]
fn functions_blocks_tanh_and_atan_compute_their_closed_forms() {
    let (g, vs, b) = (2e-3, 0.8, 0.05);
    let parameters = format!(r#"{{"conductance": {g}, "saturation": {vs}, "bend": {b}}}"#);
    let mut bench = Bench::at::<constructs::Constructs>(&parameters, REFERENCE_TEMPERATURE);
    for v in [-2.0, -0.3, 0.0, 0.6, 1.5] {
        let evaluation = bench.evaluate(&[v, 0.0]);
        let ratio = v / vs;
        let current = g * vs * ratio.tanh() + b * g * vs * ratio.atan();
        let slope = g * (1.0 - ratio.tanh() * ratio.tanh()) + b * g / (1.0 + ratio * ratio);
        CLOSED.assert(&format!("current at {v}"), evaluation.residual[0], current);
        CLOSED.assert(
            &format!("conductance at {v}"),
            evaluation.conductance[0][0],
            slope,
        );
        // The output variables: the current the last evaluation computed, and its
        // derivative by the voltage of `a`, a `ddx`.
        assert_eq!(
            bench.variables(),
            [evaluation.residual[0], evaluation.conductance[0][0]]
        );
    }
    assert_eq!(
        constructs::Constructs::variable_names(),
        ["current", "small_signal"]
    );
}

/// The exact derivative by each parameter, through `tanh`, `atan` and the inlined
/// function, is the central difference of the plain values, to the difference's own
/// error: its step squared times the third derivative, and the values' rounding over the
/// step, below 1e-7 of the derivative here.
#[test]
fn tanh_and_atan_differentiate_by_their_parameters_exactly() {
    let model = Differentiable::<constructs::Constructs>::new();
    let rig = Rig { model: &model };
    let base = serde_json::json!({"conductance": 2e-3, "saturation": 0.8, "bend": 0.05});
    for parameter in ["conductance", "saturation", "bend"] {
        let at = [0.6, 0.1];
        let object = base.as_object().cloned().expect("an object");
        let exact = rig.exact(&object, parameter, (&at, None)).expect("a twin");
        let value = base[parameter].as_f64().expect("a number");
        let step = 1e-6 * value;
        let shifted = |by: f64| {
            let mut moved = object.clone();
            moved.insert(parameter.to_owned(), serde_json::json!(value + by));
            rig.plain(&moved, (&at, None))
        };
        let (above, below) = (shifted(step), shifted(-step));
        for (kind, index) in [(0, 0), (2, 0)] {
            let numeric = (above[kind][index] - below[kind][index]) / (2.0 * step);
            let [derivative] = exact[kind][index].derivatives;
            assert!(
                (numeric - derivative).abs() <= 1e-7 * derivative.abs(),
                "{parameter}: {derivative:e} exact, {numeric:e} by differences"
            );
        }
    }
}

/// A `$warning` states its fact for the parameters that raise it, and none for others.
#[test]
fn a_warning_states_its_fact() {
    let read = |text: &str| serde_json::from_str(text).expect("parameters");
    let wide = constructs::Constructs::warnings::<f64>(&read(r#"{"saturation": 20}"#), 300.0);
    assert_eq!(
        wide.expect("valid"),
        ["a saturation of 20.0 V lies beyond the device's range"]
    );
    let narrow = constructs::Constructs::warnings::<f64>(&read("{}"), 300.0);
    assert_eq!(narrow.expect("valid"), Vec::<String>::new());
}

/// The thermal resistance compiled from Verilog-A is jolty's own, to rounding, and its
/// terminals are thermal.
#[test]
fn the_thermal_resistance_is_its_twin() {
    let parameters = r#"{"resistance": 25}"#;
    let mut worst = Worst::default();
    let mut hand = Bench::at::<ThermalResistance>(parameters, REFERENCE_TEMPERATURE);
    let mut compiled =
        Bench::at::<thermal_resistance::ThermalResistance>(parameters, REFERENCE_TEMPERATURE);
    for x in [[0.0, 0.0], [5.0, 1.0], [-3.0, 40.0]] {
        worst.evaluations(
            "thermal resistance",
            &x,
            &hand.evaluate(&x),
            &compiled.evaluate(&x),
        );
        worst.noises("thermal resistance", &hand.noise(&x), &compiled.noise(&x));
    }
    let read = |text: &str| serde_json::from_str(text).expect("parameters");
    let interface = thermal_resistance::ThermalResistance::interface::<f64>(&read(parameters));
    assert_eq!(interface.thermal, [0, 1]);
}

/// A resistor heated through its thermal terminal reads the terminal's rise above the
/// ambient as `Temp(t)`, and its power heats the terminal's net as `Pwr(t)`: at 1 V
/// across it and 5 K above the ambient, `i = 1 / (r · (1 + α · 5))`, and the net takes
/// `i · 1 V` of heat, with the derivative of the current by the temperature,
/// `−r · α / (r · (1 + α · 5))²`.
#[test]
fn a_heated_resistor_reads_and_heats_its_thermal_terminal() {
    let (r, alpha) = (100.0, 1e-3);
    let parameters = format!(r#"{{"resistance": {r}, "temperature_coefficient": {alpha}}}"#);
    let mut bench =
        Bench::at::<heated_resistor::HeatedResistor>(&parameters, REFERENCE_TEMPERATURE);
    let evaluation = bench.evaluate(&[1.0, 0.0, 5.0]);
    let heated = r * (1.0 + alpha * 5.0);
    let current = 1.0 / heated;
    CLOSED.assert("current", evaluation.residual[0], current);
    CLOSED.assert("returning current", evaluation.residual[1], -current);
    CLOSED.assert("heat", evaluation.residual[2], -current);
    CLOSED.assert(
        "current by temperature",
        evaluation.conductance[0][2],
        -r * alpha / (heated * heated),
    );
    let read = |text: &str| serde_json::from_str(text).expect("parameters");
    let interface = heated_resistor::HeatedResistor::interface::<f64>(&read(&parameters));
    assert_eq!(interface.thermal, [2]);
}

/// A diode limited by the simulator's own `"pnjlim"` limits its steps as jolty's
/// hand-written diode does, which the same limiting holds: the same steps limited, and
/// the same currents and conductances, to rounding, along a sequence of large steps.
#[test]
fn the_simulators_limiting_is_jolty_s() {
    let mut worst = Worst::default();
    let mut hand = Bench::at::<Diode>("{}", REFERENCE_TEMPERATURE);
    let mut compiled = Bench::at::<limited_diode::Diode>("{}", REFERENCE_TEMPERATURE);
    let mut limited = 0;
    for v in [0.0, 10.0, 10.0, 10.0, 10.0, -10.0, -10.0, 0.6, 0.7, 0.65] {
        let x = [v, 0.0];
        let input = Input::new(&x, None, 1e-12);
        let (a, b) = (hand.evaluate_with(&input), compiled.evaluate_with(&input));
        limited += usize::from(a.limited);
        worst.evaluations(&format!("diode at {v}"), &x, &a, &b);
    }
    assert!(limited >= 4, "{limited} steps limited");
}

/// A parameter reads under the name `aliasparam` gives it, a range may be bounded by
/// another parameter, and a `$simparam` jolty does not define reads as the default its
/// source gives.
#[test]
fn aliases_bounds_and_defaults_read_as_the_source_says() {
    let mut by_name =
        Bench::at::<constructs::Constructs>(r#"{"conductance": 3e-3}"#, REFERENCE_TEMPERATURE);
    let mut by_alias = Bench::at::<constructs::Constructs>(r#"{"g": 3e-3}"#, REFERENCE_TEMPERATURE);
    assert_eq!(
        by_name.evaluate(&[0.4, 0.0]).residual,
        by_alias.evaluate(&[0.4, 0.0]).residual
    );
    let narrow = r#"{"saturation": 2, "saturation_max": 1}"#;
    let error = Bench::try_at::<constructs::Constructs>(narrow, REFERENCE_TEMPERATURE)
        .map(|_| ())
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "saturation_max must be at least 2.0, not 1.0"
    );
    assert_eq!(
        constructs::ConstructsParameters::<f64>::default()
            .scale
            .to_bits(),
        1.0_f64.to_bits()
    );
}

/// Code under an initial event computes the conductance of the fingers, rounded to the
/// nearest whole by `floor` and up by `ceil`: `(2 + 3) / r` for 2.4 fingers, and no
/// derivative by them, which are whole wherever they change nothing.
#[test]
fn initial_code_rounds_the_fingers() {
    let r = 1e3;
    let parameters = format!(r#"{{"r": {r}, "fingers": 2.4}}"#);
    let mut bench = Bench::at::<initial_events::InitialEvents>(&parameters, REFERENCE_TEMPERATURE);
    let evaluation = bench.evaluate(&[0.5, 0.0]);
    CLOSED.assert("current", evaluation.residual[0], 0.5 * 5.0 / r);
    CLOSED.assert("conductance", evaluation.conductance[0][0], 5.0 / r);
    let read = serde_json::from_str(&parameters).expect("parameters");
    let x = [0.5, 0.0];
    let rig = Rig {
        model: &Differentiable::<initial_events::InitialEvents>::default(),
    };
    let [residual, _, _] = rig.exact(&read, "fingers", (&x, None)).expect("a number");
    assert_eq!(residual[0].derivatives, [0.0]);
}

/// `$finish` refuses the instance, after the `$strobe` before it, and a value no format of
/// a message takes follows the message, as Verilog's `$display` prints it.
#[test]
fn finish_refuses_and_a_value_no_format_takes_follows() {
    let refused =
        Bench::try_at::<initial_events::InitialEvents>(r#"{"refuse": 1}"#, REFERENCE_TEMPERATURE);
    let message = refused.err().map(|error| error.0);
    assert_eq!(message.as_deref(), Some("the model calls $finish"));
    let read = |text: &str| serde_json::from_str(text).expect("parameters");
    let many = initial_events::InitialEvents::warnings::<f64>(
        &read(r#"{"fingers": 12}"#),
        REFERENCE_TEMPERATURE,
    );
    assert_eq!(many.expect("valid"), ["many fingers: 12.0"]);
    assert_eq!(
        initial_events::InitialEvents::warned(&initial_events::InitialEventsMemory::default())
            .count(),
        0
    );
}

/// The current of a short is read as its unknown, kept rather than collapsed: `2 · v / r`
/// through the two resistors from the inner node, and the node a short joins to it, to
/// `b`; and two nodes a short joins, the upper and the lower, are one node, on the ground
/// where the lower is shorted to it, so that `a` sends `v / r` into it, and `v / (2 · r)`
/// through the lower's resistance where it is not.
#[test]
fn shorts_join_nodes_and_keep_the_currents_read() {
    let (r, v) = (1e3, 0.6);
    for grounded in [0, 1] {
        let parameters = format!(r#"{{"r": {r}, "grounded": {grounded}}}"#);
        let mut bench = Bench::at::<networks::Networks>(&parameters, REFERENCE_TEMPERATURE);
        // The terminals, the inner node, the upper where it is not on the ground, the noise's
        // node and current, and the short's current.
        let unknowns = bench.unknowns().len();
        assert_eq!(unknowns, 7 - grounded, "{parameters}");
        let short = 2.0 * v / r;
        let mut x = vec![0.0; unknowns];
        x[0] = v;
        x[2] = v;
        if grounded == 0 {
            x[3] = v / 2.0;
        }
        x[unknowns - 1] = short;
        let evaluation = bench.evaluate(&x);
        let into_z = if grounded == 1 { v / r } else { v / (2.0 * r) };
        CLOSED.assert(
            &format!("{parameters}: a"),
            evaluation.residual[0],
            short + into_z,
        );
        CLOSED.assert(&format!("{parameters}: b"), evaluation.residual[1], -short);
        for (row, residual) in evaluation.residual.iter().enumerate().skip(2) {
            assert!(
                residual.abs() <= f64::EPSILON * short,
                "{parameters}: row {row}, {residual:e}"
            );
        }
        CLOSED.assert(
            &format!("{parameters}: shorted"),
            bench.variables()[0],
            short,
        );
    }
}

/// The noise current of a branch that another branch's current reads reaches that
/// current whole, its correlation kept: `a`'s current carries `k` times the source's
/// noise, of density `k² · s`, which the small-signal equations give with the terminals
/// held.
#[test]
fn a_noise_current_read_keeps_its_correlation() {
    let (k, s) = (0.1, 4e-20);
    let parameters = format!(r#"{{"k": {k}, "s": {s}}}"#);
    let mut bench = Bench::at::<networks::Networks>(&parameters, REFERENCE_TEMPERATURE);
    let x = vec![0.0; bench.unknowns().len()];
    let g = bench.evaluate(&x).conductance;
    let sources = bench.noise(&x);
    let [source] = sources.as_slice() else {
        panic!("one noise source, not {sources:?}");
    };
    // The excitation's response, the terminals held: `G δ = −e` over the internal unknowns.
    let mut excitation = vec![0.0; x.len()];
    for (equation, sign) in source.equations.iter().zip([1.0, -1.0]) {
        if let Some(unknown) = equation {
            excitation[unknown.index()] += sign;
        }
    }
    let internal: Vec<usize> = (2..x.len()).collect();
    let matrix: Vec<Vec<f64>> = internal
        .iter()
        .map(|&row| internal.iter().map(|&column| g[row][column]).collect())
        .collect();
    let right: Vec<f64> = internal.iter().map(|&row| -excitation[row]).collect();
    let response = solve(matrix, right);
    let current: f64 = internal
        .iter()
        .zip(&response)
        .map(|(&column, value)| g[0][column] * value)
        .sum::<f64>()
        + excitation[0];
    CLOSED.assert("a's noise", source.white * current * current, k * k * s);
}

/// Every compiled fixture kernel's derived `Lift` visits the exact real parameter paths
/// that `serde` writes, including integer parameters and optional parameters.
#[test]
fn every_compiled_fixture_lift_matches_serde() {
    assert_lift_matches_serde_with_optionals(
        &Differentiable::<constructs::Constructs>::new(),
        "{}",
        &[],
    );
    assert_lift_matches_serde(
        &Differentiable::<constructs::Constructs>::new(),
        r#"{"conductance": 3e-3, "saturation": 0.9, "bend": 0.1, "scale": 2.5}"#,
        &[],
    );
    assert_lift_matches_serde_with_optionals(
        &Differentiable::<heated_resistor::HeatedResistor>::new(),
        r#"{"resistance": 100, "temperature_coefficient": 1e-3}"#,
        &[],
    );
    assert_lift_matches_serde_with_optionals(
        &Differentiable::<initial_events::InitialEvents>::new(),
        r#"{"r": 1e3, "fingers": 2.4, "refuse": 0}"#,
        &[],
    );
    assert_lift_matches_serde_with_optionals(
        &Differentiable::<limited_diode::Diode>::new(),
        "{}",
        &[],
    );
    assert_lift_matches_serde_with_optionals(
        &Differentiable::<networks::Networks>::new(),
        r#"{"r": 1e3, "grounded": 1, "k": 0.2, "s": 4e-20}"#,
        &[],
    );
    assert_lift_matches_serde_with_optionals(
        &Differentiable::<thermal_resistance::ThermalResistance>::new(),
        r#"{"resistance": 25}"#,
        &[],
    );
}

/// The solution of `matrix · x = right`, by Gaussian elimination with partial pivoting.
fn solve(mut matrix: Vec<Vec<f64>>, mut right: Vec<f64>) -> Vec<f64> {
    let n = right.len();
    for column in 0..n {
        let pivot = (column..n)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))
            .expect("a row");
        matrix.swap(column, pivot);
        right.swap(column, pivot);
        for row in column + 1..n {
            let factor = matrix[row][column] / matrix[column][column];
            for inner in column..n {
                matrix[row][inner] -= factor * matrix[column][inner];
            }
            right[row] -= factor * right[column];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let known: f64 = (row + 1..n)
            .map(|column| matrix[row][column] * x[column])
            .sum();
        x[row] = (right[row] - known) / matrix[row][row];
    }
    x
}
