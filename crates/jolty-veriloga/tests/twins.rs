//! The kernels compiled from Verilog-A against their hand-written twins in
//! `jolty-models`, whose equations their sources follow operation for operation: the same
//! unknowns, and at every point the same residuals, charges, conductances, capacitances,
//! limiting, noise and exact gradient by every parameter, to rounding.

#![cfg(test)]

mod common;

use common::Worst;
use jolty_models::testing::{Bench, Rig, numbers};
use jolty_models::{Differentiable, Input, Kernel, Layout, REFERENCE_TEMPERATURE};
use jolty_veriloga::kernels::{diode, resistor};

/// Compares the twins `H` and `C` with `parameters`, a JSON object, at each point of
/// `points`, the values of the hand-written kernel's unknowns: evaluated in order, so
/// that each limits its step from the last, then settled, then each parameter's
/// gradient and the noise.
fn compare<H: Kernel, C: Kernel>(parameters: &str, points: &[Vec<f64>], worst: &mut Worst) {
    let mut hand = Bench::at::<H>(parameters, REFERENCE_TEMPERATURE);
    let mut compiled = Bench::at::<C>(parameters, REFERENCE_TEMPERATURE);
    assert_eq!(hand.unknowns(), compiled.unknowns(), "{parameters}");
    for x in points {
        let what = format!("{} {parameters} at {x:?}", H::NAME);
        let input = Input::new(x, Some(1e-9), 1e-12);
        worst.evaluations(
            &what,
            x,
            &hand.evaluate_with(&input),
            &compiled.evaluate_with(&input),
        );
        worst.noises(&what, &hand.noise(x), &compiled.noise(x));
    }
    for x in points {
        let what = format!("{} {parameters} settled at {x:?}", H::NAME);
        worst.evaluations(&what, x, &hand.settle(x), &compiled.settle(x));
    }
    let value: serde_json::Value = serde_json::from_str(parameters).expect("JSON");
    let object = value.as_object().cloned().expect("an object");
    let mut paths = Vec::new();
    numbers(&value, &mut Vec::new(), &mut paths);
    let (hand, compiled) = (Differentiable::<H>::new(), Differentiable::<C>::new());
    let (hand, compiled) = (Rig { model: &hand }, Rig { model: &compiled });
    for path in paths {
        let parameter = path.join(".");
        let number = path
            .iter()
            .fold(&value, |inner, key| &inner[key.as_str()])
            .as_f64()
            .expect("a number");
        for x in points {
            let what = format!("{} {parameters} by {parameter} at {x:?}", H::NAME);
            let a = hand
                .exact(&object, &parameter, (x, None))
                .expect("a twin by hand");
            let b = compiled
                .exact(&object, &parameter, (x, None))
                .expect("a twin compiled");
            for (kind, a, b) in [
                ("residual", &a[0], &b[0]),
                ("charge", &a[1], &b[1]),
                ("conductance", &a[2], &b[2]),
            ] {
                let values = a
                    .iter()
                    .fold(0.0_f64, |most, value| most.max(value.value.abs()));
                let derivatives = a
                    .iter()
                    .fold(0.0_f64, |most, value| most.max(value.derivative.abs()));
                for (index, (a, b)) in a.iter().zip(b).enumerate() {
                    worst.close(&format!("{what}: {kind} {index}"), a.value, b.value, values);
                    // A derivative sums terms of the size of the change a relative step of
                    // the parameter makes, the value over the parameter, which may cancel.
                    let terms = if number == 0.0 {
                        derivatives
                    } else {
                        derivatives.max(values / number.abs())
                    };
                    worst.close(
                        &format!("{what}: ∂{kind} {index}"),
                        a.derivative,
                        b.derivative,
                        terms,
                    );
                }
            }
        }
    }
}

/// Every point of a sweep of the first unknown from `low` to `high` in `steps`, the
/// others at `rest`.
fn sweep((low, high): (f64, f64), steps: u32, rest: &[f64]) -> Vec<Vec<f64>> {
    (0..=steps)
        .map(|step| {
            let mut x = vec![low + (high - low) * f64::from(step) / f64::from(steps)];
            x.extend_from_slice(rest);
            x
        })
        .collect()
}

#[test]
fn the_compiled_resistor_is_its_hand_written_twin() {
    let mut worst = Worst::default();
    let plain = r#"{"resistance": 1000}"#;
    compare::<jolty_models::Resistor, resistor::Resistor>(
        plain,
        &sweep((-5.0, 5.0), 20, &[0.0]),
        &mut worst,
    );
    let every = r#"{"resistance": 50, "temperature_coefficient": 3e-3,
        "quadratic_temperature_coefficient": 1e-5, "nominal_temperature": 290,
        "voltage_coefficient": 0.02, "quadratic_voltage_coefficient": 0.003,
        "flicker_coefficient": 1e-14, "flicker_exponent": 1.5}"#;
    compare::<jolty_models::Resistor, resistor::Resistor>(
        every,
        &sweep((-3.0, 3.0), 12, &[0.5]),
        &mut worst,
    );
    let exponential = r#"{"resistance": 50, "exponential_temperature_coefficient": 0.4}"#;
    compare::<jolty_models::Resistor, resistor::Resistor>(
        exponential,
        &sweep((-3.0, 3.0), 6, &[0.0]),
        &mut worst,
    );
    // Below 1 Ω, the current is an unknown of its own.
    let small = r#"{"resistance": 0.25, "voltage_coefficient": 0.1, "flicker_coefficient": 1e-14}"#;
    let points: Vec<Vec<f64>> = sweep((-1.0, 1.0), 10, &[0.1])
        .into_iter()
        .map(|mut x| {
            x.push(x[0] * 3.0);
            x
        })
        .collect();
    compare::<jolty_models::Resistor, resistor::Resistor>(small, &points, &mut worst);
}

#[test]
fn the_compiled_diode_is_its_hand_written_twin() {
    let mut worst = Worst::default();
    let junction = sweep((-10.0, 1.0), 44, &[0.0]);
    compare::<jolty_models::Diode, diode::Diode>("{}", &junction, &mut worst);
    let charged =
        r#"{"junction_capacitance": 2e-12, "transit_time": 1e-9, "grading_coefficient": 0.4}"#;
    compare::<jolty_models::Diode, diode::Diode>(charged, &junction, &mut worst);
    let every = r#"{"saturation_current": 2e-14, "emission_coefficient": 1.3,
        "junction_capacitance": 2e-12, "junction_potential": 0.8,
        "grading_coefficient": 0.4, "depletion_coefficient": 0.6, "transit_time": 1e-9,
        "breakdown_voltage": 5, "breakdown_current": 1e-3,
        "breakdown_emission_coefficient": 1.1, "breakdown_temperature_coefficient": 1e-3,
        "energy_gap": 1.2, "saturation_current_temperature_exponent": 2.5,
        "nominal_temperature": 290, "flicker_coefficient": 1e-12, "flicker_exponent": 1.5}"#;
    compare::<jolty_models::Diode, diode::Diode>(every, &junction, &mut worst);
    let resistive = r#"{"series_resistance": 10, "junction_capacitance": 1e-12}"#;
    let behind: Vec<Vec<f64>> = junction
        .iter()
        .map(|x| vec![x[0] + 0.01, 0.0, x[0]])
        .collect();
    compare::<jolty_models::Diode, diode::Diode>(resistive, &behind, &mut worst);
    let small = r#"{"series_resistance": 0.5, "junction_capacitance": 1e-12}"#;
    let carried: Vec<Vec<f64>> = junction
        .iter()
        .map(|x| vec![x[0] + 0.01, 0.0, x[0], 0.02])
        .collect();
    compare::<jolty_models::Diode, diode::Diode>(small, &carried, &mut worst);
}

#[test]
fn out_of_range_parameters_are_refused_as_by_hand() {
    for parameters in [
        r#"{"grading_coefficient": 1}"#,
        r#"{"saturation_current": -1}"#,
        r#"{"series_resistance": -1}"#,
        r#"{"breakdown_voltage": 0}"#,
    ] {
        let hand = Bench::try_at::<jolty_models::Diode>(parameters, 300.0)
            .map(|_| ())
            .unwrap_err();
        let compiled = Bench::try_at::<diode::Diode>(parameters, 300.0)
            .map(|_| ())
            .unwrap_err();
        assert_eq!(hand, compiled, "{parameters}");
    }
    let hand = Bench::try_at::<jolty_models::Diode>("{}", 0.0)
        .map(|_| ())
        .unwrap_err();
    let compiled = Bench::try_at::<diode::Diode>("{}", 0.0)
        .map(|_| ())
        .unwrap_err();
    assert_eq!(hand, compiled);
}

/// Whether an instance of `K` with `parameters`, a JSON object, says it is linear.
fn linear<K: Kernel>(parameters: &str) -> bool {
    let parameters: K::Parameters<f64> = serde_json::from_str(parameters).expect("parameters");
    let prepared = K::prepare(&parameters, REFERENCE_TEMPERATURE).expect("valid parameters");
    let mut layout = Layout::new(K::interface(&parameters).terminals.len());
    let shape = K::layout(&parameters, &prepared, &mut layout).expect("valid parameters");
    K::linear(&shape, &prepared)
}

/// The compiled kernels find the same instances linear as their twins do: a resistor
/// without voltage coefficients, whether its current is an unknown or not, and no diode.
#[test]
fn the_compiled_kernels_are_linear_where_their_twins_are() {
    for parameters in [
        r#"{"resistance": 1000}"#,
        r#"{"resistance": 1000, "voltage_coefficient": 0.01}"#,
        r#"{"resistance": 1000, "quadratic_voltage_coefficient": 0.01}"#,
        r#"{"resistance": 0.5}"#,
        r#"{"resistance": 0.5, "voltage_coefficient": 0.01}"#,
        r#"{"resistance": 50, "temperature_coefficient": 3e-3}"#,
    ] {
        let hand = linear::<jolty_models::Resistor>(parameters);
        assert_eq!(
            hand,
            linear::<resistor::Resistor>(parameters),
            "{parameters}"
        );
    }
    for parameters in ["{}", r#"{"series_resistance": 10}"#] {
        assert!(!linear::<jolty_models::Diode>(parameters));
        assert!(!linear::<diode::Diode>(parameters));
    }
}
