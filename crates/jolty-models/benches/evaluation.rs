//! The time of one evaluation of a device's kernel, one instance on the shared bench, at a
//! step of a transient, in each of the shapes its parameters give it:
//! `<model>/<case>`. Name one after `--` to run it alone, as
//! `cargo bench -p jolty-models --bench evaluation -- vbic/`; save one build's times with
//! `--save-baseline NAME` and compare another's with `--baseline NAME`.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use jolty_models::testing::Bench;
use jolty_models::{
    BipolarTransistor, Capacitor, CurrentControlledMemristor, CurrentControlledSwitch, Input, Jfet,
    Kernel, Photocell, REFERENCE_TEMPERATURE, Resistor, Vbic, Vdmos, VoltageControlledMemristor,
    VoltageControlledSwitch,
};

/// Times the evaluation of an instance of `K` with `parameters` as `name`, each of its
/// unknowns at the voltage `bias` gives its name: the terminals' `n0`, `n1` and on, and
/// the device's own their labels, such as `collector`.
fn evaluation<K: Kernel>(
    criterion: &mut Criterion,
    name: &str,
    parameters: &str,
    bias: impl Fn(&str) -> f64,
) {
    let mut bench = Bench::at::<K>(parameters, REFERENCE_TEMPERATURE);
    let x: Vec<f64> = bench
        .unknowns()
        .into_iter()
        .map(|unknown| bias(unknown.strip_prefix("device:").unwrap_or(unknown)))
        .collect();
    let mut contributions = bench.contributions();
    let mut input = Input::new(&x, Some(1e-9), 1e-12);
    input.limiting = false;
    criterion.bench_function(name, |bencher| {
        bencher.iter(|| black_box(bench.evaluate_into(black_box(&input), &mut contributions)));
    });
}

/// A bipolar transistor's unknowns, forward active: its collector at 3 V, its base at
/// 0.75 V, its emitter grounded and its substrate at −1 V, each internal node a little
/// inside the terminal it lies behind, and each current unknown a milliampere.
fn forward_active(name: &str) -> f64 {
    match name {
        "n0" => 3.0,
        "collector" => 2.99,
        "intrinsic_collector" => 2.9,
        "n1" => 0.75,
        "base" => 0.746,
        "intrinsic_base" => 0.74,
        "parasitic_base" => 0.7,
        "emitter" => 0.002,
        "n3" => -1.0,
        "substrate" => -0.999,
        "n2" => 0.0,
        _ => 1e-3,
    }
}

/// VBIC with every part of its model in play, with series resistances above 1 Ω.
const VBIC_FULL: &str = r#"{"saturation_current": 2e-16, "forward_emission_coefficient": 1.01,
    "reverse_emission_coefficient": 1.02, "reverse_saturation_current_ratio": 0.8,
    "forward_early_voltage": 60, "reverse_early_voltage": 4,
    "forward_knee_current": 0.02, "reverse_knee_current": 0.005, "knee_rolloff": 0.6,
    "ideal_base_emitter_current": 2e-18, "intrinsic_base_emitter_fraction": 0.8,
    "nonideal_base_emitter_current": 3e-15, "ideal_base_collector_current": 4e-17,
    "nonideal_base_collector_current": 2e-15,
    "avalanche_coefficient": 2.5, "avalanche_exponent_coefficient": 12,
    "parasitic_saturation_current": 5e-18, "parasitic_extrinsic_fraction": 0.7,
    "parasitic_ideal_base_emitter_current": 1e-19,
    "parasitic_ideal_base_collector_current": 2e-19, "parasitic_knee_current": 1e-4,
    "intrinsic_collector_resistance": 20, "epi_saturation_voltage": 2,
    "epi_doping_parameter": 2e-11, "high_current_collector_factor": 0.5,
    "intrinsic_base_resistance": 30, "parasitic_base_resistance": 40,
    "base_emitter_capacitance": 2e-14, "base_collector_capacitance": 1e-14,
    "extrinsic_base_collector_capacitance": 5e-15, "epi_charge": 1e-14,
    "collector_substrate_capacitance": 3e-14, "forward_transit_time": 1e-11,
    "transit_time_bias_coefficient": 3, "transit_time_voltage": 2,
    "transit_time_current": 0.01, "reverse_transit_time": 1e-9,
    "base_emitter_breakdown_voltage": 5, "reach_through_voltage": 3,
    "reach_through_smoothing": 0.1, "extrinsic_collector_resistance": 5,
    "extrinsic_base_resistance": 8, "emitter_resistance": 2, "substrate_resistance": 3}"#;

/// A Gummel-Poon transistor with every part of its model in play: its base resistance
/// falling with its base current, its series resistances above 1 Ω, and a vertical
/// substrate junction.
const GUMMEL_POON_FULL: &str = r#"{"saturation_current": 1e-15, "forward_beta": 150,
    "forward_early_voltage": 60, "reverse_early_voltage": 20,
    "forward_knee_current": 0.05, "reverse_knee_current": 0.01,
    "base_emitter_leakage_current": 1e-14, "base_collector_leakage_current": 1e-13,
    "base_emitter_capacitance": 2e-12, "base_collector_capacitance": 1e-12,
    "base_collector_grading": 0.4, "internal_base_fraction": 0.7,
    "forward_transit_time": 4e-10, "transit_time_bias_coefficient": 2,
    "transit_time_voltage": 3, "transit_time_current": 0.02, "reverse_transit_time": 1e-8,
    "base_resistance": 200, "min_base_resistance": 20, "base_resistance_current": 1e-4,
    "collector_resistance": 10, "emitter_resistance": 2, "substrate": "vertical",
    "substrate_saturation_current": 1e-16, "substrate_capacitance": 3e-13}"#;

fn bipolar(criterion: &mut Criterion) {
    evaluation::<BipolarTransistor>(
        criterion,
        "bipolar_transistor/defaults",
        "{}",
        forward_active,
    );
    evaluation::<BipolarTransistor>(
        criterion,
        "bipolar_transistor/full",
        GUMMEL_POON_FULL,
        forward_active,
    );
    let delayed = GUMMEL_POON_FULL.replacen('{', r#"{"excess_phase": 30, "#, 1);
    evaluation::<BipolarTransistor>(
        criterion,
        "bipolar_transistor/delayed",
        &delayed,
        forward_active,
    );
    evaluation::<Vbic>(criterion, "vbic/defaults", "{}", forward_active);
    evaluation::<Vbic>(criterion, "vbic/full", VBIC_FULL, forward_active);
    let delayed = VBIC_FULL.replacen('{', r#"{"excess_phase_delay": 2e-12, "#, 1);
    evaluation::<Vbic>(criterion, "vbic/delayed", &delayed, forward_active);
}

/// A field-effect transistor's unknowns, saturated: its drain at 3 V, its gate at −0.5 V
/// and its source grounded, each internal node a little inside the terminal it lies
/// behind, and each current unknown a milliampere.
fn saturated(name: &str) -> f64 {
    match name {
        "n0" => 3.0,
        "drain" => 2.99,
        "n1" => -0.5,
        "n2" => 0.0,
        "source" => 0.002,
        _ => 1e-3,
    }
}

/// A JFET with every part of its model in play: charged gate junctions, channel-length
/// modulation, a doping tail and series resistances above 1 Ω.
const JFET_FULL: &str = r#"{"threshold_voltage": -1.8, "transconductance_parameter": 1e-3,
    "channel_length_modulation": 0.02, "doping_tail": 0.8, "gate_source_capacitance": 2e-12,
    "gate_drain_capacitance": 1e-12, "drain_resistance": 10, "source_resistance": 5}"#;

fn field_effect(criterion: &mut Criterion) {
    evaluation::<Jfet>(criterion, "jfet/defaults", "{}", saturated);
    evaluation::<Jfet>(criterion, "jfet/full", JFET_FULL, saturated);
    let quasi = r#"{"threshold_voltage": 3, "transconductance_parameter": 10,
        "drain_resistance": 2, "quasi_saturation_resistance": 5,
        "quasi_saturation_voltage": 20}"#;
    evaluation::<Vdmos>(criterion, "vdmos/quasi_saturated", quasi, quasi_saturated);
}

/// A power MOSFET's unknowns, its drain resistance quasi-saturated: its drain at 20 V, its
/// gate at 8 V and its source grounded, each internal node a little inside the terminal it
/// lies behind, and each current unknown a milliampere.
fn quasi_saturated(name: &str) -> f64 {
    match name {
        "n0" => 20.0,
        "drain" => 19.8,
        "n1" | "gate" => 8.0,
        _ => saturated(name),
    }
}

/// A two-terminal device's unknowns, with what controls it: 1.5 V across it, 1.2 V or
/// 1.2 mA or 50 lux on its control, its state at 0.3 and its lagging response at 0.5.
fn controlled(name: &str) -> f64 {
    match name {
        "n0" => 1.5,
        "n2" => 1.2,
        "state" => 0.3,
        "response" => 0.5,
        "control" => 1.2e-3,
        "n1" | "n3" => 0.0,
        _ => 50.0,
    }
}

fn short_laws(criterion: &mut Criterion) {
    let coefficients = r#""voltage_coefficient": 0.01, "quadratic_voltage_coefficient": 0.002"#;
    evaluation::<Resistor>(
        criterion,
        "resistor/coefficients",
        &format!(r#"{{"resistance": 1000, {coefficients}}}"#),
        controlled,
    );
    evaluation::<Capacitor>(
        criterion,
        "capacitor/coefficients",
        &format!(r#"{{"capacitance": 1e-12, {coefficients}}}"#),
        controlled,
    );
    evaluation::<VoltageControlledSwitch>(
        criterion,
        "voltage_controlled_switch/between",
        r#"{"on_voltage": 1.5, "off_voltage": 1}"#,
        controlled,
    );
    evaluation::<CurrentControlledSwitch>(
        criterion,
        "current_controlled_switch/between",
        r#"{"on_current": 1.5e-3, "off_current": 1e-3}"#,
        controlled,
    );
    evaluation::<VoltageControlledMemristor>(
        criterion,
        "voltage_controlled_memristor/driven",
        r#"{"on_resistance": 100, "off_resistance": 1e4, "on_voltage": -1, "off_voltage": 1,
        "on_rate": 1e3, "off_rate": 1e3}"#,
        controlled,
    );
    evaluation::<CurrentControlledMemristor>(
        criterion,
        "current_controlled_memristor/driven",
        r#"{"on_resistance": 100, "off_resistance": 1e4, "charge": 1e-6}"#,
        controlled,
    );
    evaluation::<Photocell>(
        criterion,
        "photocell/lagging",
        r#"{"resistance": 1e4, "response_time": 1e-3}"#,
        controlled,
    );
}

criterion_group!(benches, bipolar, field_effect, short_laws);
criterion_main!(benches);
