//! Differential tests comparing compiled Verilog-A kernels against their hand-written twins
//! in `jolty-models` across residuals, charges, Jacobians, limiting, noise, and exact
//! parameter gradients.

#![cfg(test)]

mod common;

use common::Worst;
use jolty_models::Gradient;
use jolty_models::testing::{
    Bench, Rig, assert_lift_matches_serde, assert_lift_matches_serde_with_optionals,
    assert_unknown_derivatives_match, numbers,
};
use jolty_models::{Differentiable, Input, Kernel, Layout, REFERENCE_TEMPERATURE, Thermal};
use jolty_veriloga::kernels::{diode, resistor};
use schemars::JsonSchema;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Compares the hand-written kernel `H` against compiled kernel `C` for JSON `parameters`
/// across `points`. Evaluates points in order to exercise step limiting, then checks
/// settled evaluations, parameter gradients, and noise.
fn compare<H: Kernel, C: Kernel>(parameters: &str, points: &[Vec<f64>], worst: &mut Worst)
where
    H::Parameters<f64>: DeserializeOwned + Serialize + JsonSchema,
    C::Parameters<f64>: DeserializeOwned + Serialize + JsonSchema,
{
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
    // Compiled kernels can claim entries they never stamp when a variable in one shape's
    // branch depends only on a probe in another shape. Compare conductances by matrix
    // position, treating unallocated entries in either kernel as zero.
    let places = [hand.positions(&object), compiled.positions(&object)];
    let mut union: Vec<(usize, usize)> = places.concat();
    union.sort_unstable();
    union.dedup();
    let by_place = |found: &[Gradient], places: &[(usize, usize)]| -> Vec<Gradient> {
        union
            .iter()
            .map(|place| {
                places
                    .iter()
                    .position(|other| other == place)
                    .map_or_else(Gradient::default, |slot| found[slot])
            })
            .collect()
    };
    for path in paths {
        let parameter = path.join(".");
        let number = path
            .iter()
            .fold(&value, |inner, key| &inner[key.as_str()])
            .as_f64()
            .expect("a number");
        for x in points {
            let what = format!("{} {parameters} by {parameter} at {x:?}", H::NAME);
            let a = hand.exact(&object, &parameter, (x, None));
            let b = compiled.exact(&object, &parameter, (x, None));
            // A selector, such as a temperature model, an integer, has no derivative in
            // either twin.
            let (a, b) = match (a, b) {
                (Some(a), Some(b)) => (a, b),
                (None, None) => continue,
                _ => panic!("{what}: a twin of one alone"),
            };
            let conductances = [by_place(&a[2], &places[0]), by_place(&b[2], &places[1])];
            for (kind, a, b) in [
                ("residual", &a[0], &b[0]),
                ("charge", &a[1], &b[1]),
                ("conductance", &conductances[0], &conductances[1]),
            ] {
                let values = a
                    .iter()
                    .fold(0.0_f64, |most, value| most.max(value.value.abs()));
                let derivatives = a
                    .iter()
                    .fold(0.0_f64, |most, value| most.max(value.derivatives[0].abs()));
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
                        a.derivatives[0],
                        b.derivatives[0],
                        terms,
                    );
                }
            }
        }
    }
}

/// Generates evenly spaced state vectors sweeping the first unknown from `low` to `high` in
/// `steps` while holding remaining unknowns at `rest`.
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
    // Below 1 ohm, the branch current becomes an explicit unknown.
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
    let parts = r#"{"saturation_current": 2e-14, "emission_coefficient": 1.3,
        "junction_capacitance": 2e-12, "junction_potential": 0.8, "grading_coefficient": 0.4,
        "depletion_coefficient": 0.6, "transit_time": 1e-9, "forward_knee_current": 1e-3,
        "reverse_knee_current": 1e-9, "recombination_saturation_current": 3e-13,
        "recombination_emission_coefficient": 1.8, "sidewall_saturation_current": 4e-15,
        "sidewall_emission_coefficient": 1.1, "sidewall_junction_capacitance": 5e-13,
        "sidewall_junction_potential": 0.6, "sidewall_grading_coefficient": 0.3,
        "sidewall_depletion_coefficient": 0.4, "sidewall_forward_knee_current": 2e-4,
        "tunneling_saturation_current": 1e-13, "sidewall_tunneling_saturation_current": 2e-14,
        "tunneling_emission_coefficient": 20,
        "tunneling_saturation_current_temperature_exponent": 2,
        "tunneling_energy_gap_factor": 0.8, "breakdown_voltage": 5, "breakdown_current": 1e-3,
        "breakdown_temperature_coefficient": 1e-3, "temperature_model": 2, "energy_gap": 1.17,
        "energy_gap_alpha": 5e-4, "energy_gap_beta": 1000,
        "saturation_current_temperature_exponent": 2.5,
        "transit_time_temperature_coefficient": 1e-3,
        "grading_coefficient_temperature_coefficient": 1e-3, "nominal_temperature": 290, "flicker_coefficient": 1e-12, "flicker_exponent": 1.5}"#;
    compare::<jolty_models::Diode, diode::Diode>(parts, &junction, &mut worst);
    let relative = parts.replace(r#""temperature_model": 2"#, r#""temperature_model": 1"#);
    compare::<jolty_models::Diode, diode::Diode>(&relative, &junction, &mut worst);
    let separate = parts.replace(
        '{',
        r#"{"series_resistance": 10, "sidewall_series_resistance": 40,
        "series_resistance_temperature_coefficient": 2e-3, "capacitance_temperature_model": 1,
        "junction_potential_temperature_coefficient": 2e-3,
        "junction_capacitance_temperature_coefficient": 1e-3,
        "sidewall_junction_potential_temperature_coefficient": 1e-3,
        "sidewall_junction_capacitance_temperature_coefficient": 2e-3, "#,
    );
    let beside: Vec<Vec<f64>> = junction
        .iter()
        .map(|x| vec![x[0] + 0.02, 0.0, x[0], x[0] + 0.01])
        .collect();
    compare::<jolty_models::Diode, diode::Diode>(&separate, &beside, &mut worst);
    let small = separate.replace(
        r#""sidewall_series_resistance": 40"#,
        r#""sidewall_series_resistance": 0.5"#,
    );
    let carried: Vec<Vec<f64>> = beside
        .iter()
        .map(|x| [x.as_slice(), &[0.02]].concat())
        .collect();
    compare::<jolty_models::Diode, diode::Diode>(&small, &carried, &mut worst);
}

#[test]
fn out_of_range_parameters_are_refused_as_by_hand() {
    for parameters in [
        r#"{"grading_coefficient": 1}"#,
        r#"{"saturation_current": -1}"#,
        r#"{"series_resistance": -1}"#,
        r#"{"breakdown_voltage": -1}"#,
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

/// Returns whether kernel `K` reports linearity for the JSON `parameters` object.
fn linear<K: Kernel>(parameters: &str) -> bool
where
    K::Parameters<f64>: DeserializeOwned,
{
    let parameters: K::Parameters<f64> = serde_json::from_str(parameters).expect("parameters");
    let prepared = K::prepare(&parameters, REFERENCE_TEMPERATURE).expect("valid parameters");
    let mut layout = Layout::new(K::interface(&parameters).terminals.len());
    let shape = K::layout(&parameters, &prepared, &mut layout).expect("valid parameters");
    K::linear(&shape, &prepared)
}

/// Verifies that compiled resistor and diode kernels match their hand-written twins on
/// `Kernel::linear`.
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

/// Verifies unknown derivatives of conductances and capacitances for compiled kernels,
/// including self-heating and small series resistances.
#[test]
fn the_compiled_kernels_differentiate_by_their_unknowns() {
    let resistor_model = Thermal::<resistor::Resistor>::new();
    assert_unknown_derivatives_match(
        &resistor_model,
        r#"{"resistance": 50, "voltage_coefficient": 0.02, "quadratic_voltage_coefficient": 0.003,
            "temperature_coefficient": 3e-3, "thermal": true}"#,
        &[2.0, 0.5, 4.0],
    );
    assert_unknown_derivatives_match(
        &resistor_model,
        r#"{"resistance": 0.25, "voltage_coefficient": 0.1}"#,
        &[0.8, 0.1, 2.5],
    );
    let diode_model = Thermal::<diode::Diode>::new();
    assert_unknown_derivatives_match(
        &diode_model,
        r#"{"series_resistance": 10, "junction_capacitance": 2e-12, "transit_time": 1e-9,
            "breakdown_voltage": 5, "thermal": true}"#,
        &[0.72, 0.0, 3.0, 0.68],
    );
    assert_unknown_derivatives_match(
        &diode_model,
        r#"{"series_resistance": 0.5, "junction_capacitance": 2e-12}"#,
        &[-5.5, 0.0, -5.3, -0.02],
    );
}

/// Verifies that derived `Lift` on compiled resistor and diode kernels visits the exact real
/// parameter paths serialized by `serde`.
#[test]
fn the_compiled_kernels_lift_matches_serde() {
    let resistor_model = Thermal::<resistor::Resistor>::new();
    let resistor_optionals =
        assert_lift_matches_serde_with_optionals(&resistor_model, r#"{"resistance": 1000}"#, &[]);
    assert!(
        resistor_optionals > 0,
        "resistor exposes optional parameters"
    );
    assert_lift_matches_serde(
        &resistor_model,
        r#"{"resistance": 50, "exponential_temperature_coefficient": 0.4,
            "voltage_coefficient": 0.02, "quadratic_voltage_coefficient": 0.003,
            "thermal": true}"#,
        &[],
    );

    let diode_model = Thermal::<diode::Diode>::new();
    let diode_optionals = assert_lift_matches_serde_with_optionals(&diode_model, "{}", &[]);
    assert!(diode_optionals > 0, "diode exposes optional parameters");
    assert_lift_matches_serde(
        &diode_model,
        r#"{"series_resistance": 10, "sidewall_series_resistance": 40,
            "forward_knee_current": 1e-3, "reverse_knee_current": 1e-9,
            "sidewall_forward_knee_current": 2e-4, "breakdown_voltage": 5,
            "temperature_model": 2, "capacitance_temperature_model": 1,
            "thermal": true}"#,
        &[],
    );
}
