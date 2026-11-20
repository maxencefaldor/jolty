//! Representative device instances covering every analog model in the standard library
//! across operating regions and optional features. Shared test guards (zero-allocation
//! evaluation, Jacobian finite-difference checks, and `laws.rs` invariants) iterate over
//! these cases.
//!
//! Both unit tests and integration tests include this file by path, so it references no
//! crate-internal symbols.

/// A device of each model, by its model's name, its parameters and the values of its
/// first local unknowns, its terminals first, the rest at [`REST`]; every analog model has
/// one or more. A transistor's gives each parameter whose default depends on its polarity,
/// so that its mirror, of the other polarity, has its numbers.
pub(crate) const CASES: &[(&str, &str, &[f64])] = &[
    ("diode", "{}", &[0.7, 0.0]),
    (
        "diode",
        r#"{"series_resistance": 0.5, "junction_capacitance": 1e-12,
                "breakdown_voltage": 5}"#,
        &[-6.0, 0.0],
    ),
    (
        "diode",
        r#"{"junction_capacitance": 1e-12, "transit_time": 1e-9, "forward_knee_current": 1e-3,
                "recombination_saturation_current": 1e-13, "tunneling_saturation_current": 1e-13,
                "sidewall_saturation_current": 2e-15, "sidewall_junction_capacitance": 2e-13,
                "sidewall_forward_knee_current": 1e-4, "temperature_model": 2,
                "capacitance_temperature_model": 1, "nominal_temperature": 290}"#,
        &[0.65, 0.0],
    ),
    (
        "diode",
        r#"{"series_resistance": 20, "sidewall_series_resistance": 0.5,
                "sidewall_saturation_current": 2e-15, "sidewall_junction_capacitance": 2e-13,
                "sidewall_tunneling_saturation_current": 1e-14, "reverse_knee_current": 1e-12,
                "breakdown_voltage": 4}"#,
        &[-4.5, 0.0, -4.4, -4.45, 0.1],
    ),
    (
        "mosfet",
        r#"{"oxide_thickness": 2e-8}"#,
        &[3.0, 2.0, 0.0, 0.0],
    ),
    (
        "mosfet",
        r#"{"level": 3, "channel": "p", "drain_resistance": 10, "source_resistance": 0.5}"#,
        &[-3.0, -2.0, 0.0, 0.0],
    ),
    (
        "vdmos",
        r#"{"transconductance_parameter": 20}"#,
        &[12.0, 6.0, 0.0],
    ),
    (
        "vdmos",
        r#"{"channel": "p", "transconductance_parameter": 10, "drain_resistance": 0.05,
                "source_resistance": 2, "gate_resistance": 3, "body_resistance": 0.01,
                "quasi_saturation_resistance": 1, "quasi_saturation_voltage": 10,
                "breakdown_voltage": 30, "thermal": true}"#,
        &[-12.0, -6.0, 0.0, 5.0],
    ),
    ("bsim3", r#"{"u0": 0.067}"#, &[3.0, 2.0, 0.0, 0.0]),
    (
        "bsim4",
        r#"{"vth0": 0.5, "u0": 0.067, "eu": 1.67, "agidl": 1e-9, "alpha0": 1e-6,
                "beta0": 20, "drain_squares": 2, "rsh": 10}"#,
        &[1.8, 1.2, 0.0, 0.0],
    ),
    (
        "bsim4",
        r#"{"channel": "p", "vth0": -0.5, "u0": 0.025, "eu": 1, "body_resistance_model": 1,
                "ngate": 1e26, "lambda": 1e-10, "vtl": 2e5, "dvtp0": 1e-8, "dvtp1": 0.5,
                "charge_partition": 0.5}"#,
        &[-1.8, -1.2, 0.0, 0.0],
    ),
    (
        "bsim4",
        r#"{"vth0": 0.5, "u0": 0.067, "eu": 1.67, "gate_resistance_model": 3, "rshg": 50,
                "xgw": 7.7e-7, "ngcon": 2, "fingers": 2, "drain_source_resistance_model": 1,
                "rsh": 1, "drain_squares": 70, "source_squares": 35,
                "body_resistance_model": 1, "thermal_noise_model": 1}"#,
        &[1.8, 1.2, 0.0, 0.0],
    ),
    (
        "bsim4",
        r#"{"channel": "p", "vth0": -0.5, "u0": 0.025, "eu": 1, "gate_resistance_model": 2,
                "rshg": 50, "xgw": 7.7e-7}"#,
        &[-1.8, -1.2, 0.0, 0.0],
    ),
    (
        "bsim4",
        r#"{"vth0": 0.5, "u0": 0.067, "eu": 1.67, "gate_resistance_model": 1}"#,
        &[1.8, 1.2, 0.0, 0.0],
    ),
    (
        "bsim3",
        r#"{"channel": "p", "u0": 0.025, "rsh": 20, "source_squares": 0.02,
                "capacitance_model": 0, "noise_model": 2, "thermal": true}"#,
        &[-3.0, -2.0, 0.0, 0.0],
    ),
    ("vbic", "{}", &[5.0, 0.7, 0.0, 0.0, 4.9, 4.9, 0.69]),
    (
        "vbic",
        r#"{"polarity": "pnp", "extrinsic_collector_resistance": 0.5,
                "extrinsic_base_resistance": 20, "emitter_resistance": 0.2,
                "substrate_resistance": 3, "base_emitter_capacitance": 1e-13,
                "base_collector_capacitance": 5e-14, "base_emitter_smoothing": 0.2,
                "forward_transit_time": 1e-11, "excess_phase_delay": 1e-12,
                "avalanche_coefficient": 2, "parasitic_saturation_current": 1e-17,
                "base_emitter_breakdown_voltage": 5, "base_emitter_overlap_capacitance": 1e-15,
                "thermal": true}"#,
        &[-5.0, -0.7, 0.0, 0.0, 10.0],
    ),
    ("bipolar_transistor", "{}", &[5.0, 0.7, 0.0]),
    (
        "bipolar_transistor",
        r#"{"base_resistance": 50, "min_base_resistance": 0.5,
                "collector_resistance": 0.2, "emitter_resistance": 2,
                "base_emitter_capacitance": 1e-12, "forward_transit_time": 1e-10,
                "excess_phase": 10, "substrate": "vertical"}"#,
        &[5.0, 0.7, 0.0, -1.0],
    ),
    (
        "jfet",
        r#"{"drain_resistance": 10, "source_resistance": 0.5,
                "gate_source_capacitance": 1e-12}"#,
        &[5.0, -0.5, 0.0],
    ),
    (
        "resistor",
        r#"{"resistance": 1e3, "voltage_coefficient": 0.1, "thermal": true}"#,
        &[1.0, 0.0, 5.0],
    ),
    ("resistor", r#"{"resistance": 0.5}"#, &[1.0, 0.0, 2.0]),
    ("resistor", r#"{"resistance": 1e3}"#, &[1.0, 0.0]),
    ("capacitor", r#"{"capacitance": 1e-9}"#, &[1.0, 0.0]),
    (
        "coupled_inductors",
        r#"{"windings": [{"inductance": 1e-3}, {"inductance": 4e-3}],
                "couplings": [{"windings": [1, 2], "coefficient": 0.9}]}"#,
        &[1.0, 0.0, 0.5, 0.0, 1e-3, 2e-3],
    ),
    (
        "voltage_source",
        r#"{"waveform": {"sine": {"offset": 0, "amplitude": 1, "frequency": 1e6}}}"#,
        &[1.0, 0.0, 1e-3],
    ),
    (
        "current_controlled_switch",
        "{}",
        &[1.0, 0.0, 0.0, 0.0, 2e-3],
    ),
    (
        "behavioral_voltage_source",
        r#"{"expression": "if(v(a) > 0.5, tanh(v(a)) * i(s), table(v(b), 0, 0, 1, 1))"}"#,
        &[1.0, 0.0, 0.7, 0.3, 0.0, 0.0, 1e-3, 2e-3],
    ),
    (
        "voltage_controlled_memristor",
        r#"{"on_resistance": 1e3, "off_resistance": 1e5, "on_voltage": -1,
                "off_voltage": 0.5, "on_rate": 1e6, "off_rate": 1e5, "state": 0.5}"#,
        &[1.0, 0.0, 0.5],
    ),
    (
        "photocell",
        r#"{"resistance": 1e4, "response_time": 0.02}"#,
        &[1.0, 0.0, 5.0, 0.0, 0.5],
    ),
    ("heat_source", r#"{"power": 0.5}"#, &[1.0, 0.0]),
    (
        "behavioral_current_source",
        r#"{"expression": "1e-3 * v(a, b) + 1e-4 * tanh(i(s))"}"#,
        &[1.0, 0.0, 0.7, 0.3, 0.0, 0.0, 2e-3],
    ),
    (
        "behavioral_current_source",
        concat!(
            r#"{"expression": "v(positive, negative) / "#,
            r#"(p(rbody) * (1 + p(vc2) * v(positive, negative) ^ 2))","#,
            r#" "parameters": {"rbody": 1e3, "vc2": 0.05}}"#,
        ),
        &[1.0, 0.0],
    ),
    (
        "behavioral_capacitor",
        r#"{"expression": "1e-12 * v(positive, negative) + 1e-13 * tanh(v(a, b))"}"#,
        &[1.0, 0.0, 0.7, 0.3],
    ),
    (
        "behavioral_capacitor",
        r#"{"expression": "p(c0) * v(positive, negative) + p(c1) * tanh(v(a, b))",
                "parameters": {"c0": 1e-12, "c1": 1e-13}}"#,
        &[1.0, 0.0, 0.7, 0.3],
    ),
    (
        "behavioral_resistor",
        r#"{"expression": "1e3 * (1 + 0.2 * tanh(v(positive, negative)) + 0.1 * tanh(i(s)))",
                "temperature_coefficient": 2e-3, "quadratic_temperature_coefficient": 1e-5,
                "nominal_temperature": 290}"#,
        &[1.0, 0.0, 0.0, 0.0, 2e-3],
    ),
    (
        "behavioral_resistor",
        r#"{"expression": "p(rbody) * (1 + p(vc1) * tanh(v(positive, negative)))",
                "parameters": {"rbody": 1e3, "vc1": 0.2},
                "temperature_coefficient": 3e-3, "quadratic_temperature_coefficient": 1e-5,
                "thermal": true}"#,
        &[1.0, 0.0, 5.0],
    ),
    (
        "current_source",
        r#"{"waveform": {"pulse": {"initial": 0, "pulsed": 1e-3,
                "rise": 1e-9, "fall": 1e-9, "width": 1e-6, "period": 1e-5}}}"#,
        &[1.0, 0.0],
    ),
    ("inductor", r#"{"inductance": 1e-3}"#, &[1.0, 0.0, 1e-3]),
    (
        "current_controlled_current_source",
        r#"{"gain": 2}"#,
        &[1.0, 0.0, 0.5, 0.0, 1e-3],
    ),
    (
        "current_controlled_voltage_source",
        r#"{"transresistance": 1e3}"#,
        &[1.0, 0.0, 0.5, 0.0, 1e-3, 2e-3],
    ),
    (
        "voltage_controlled_voltage_source",
        r#"{"gain": 10}"#,
        &[1.0, 0.0, 0.5, 0.0, 1e-3],
    ),
    (
        "voltage_controlled_current_source",
        r#"{"transconductance": 1e-3}"#,
        &[1.0, 0.0, 0.5, 0.0],
    ),
    ("voltage_controlled_switch", "{}", &[1.0, 0.0, 0.5, 0.0]),
    (
        "current_controlled_memristor",
        r#"{"on_resistance": 100, "off_resistance": 16e3, "charge": 2e-6,
                "law": "exponential"}"#,
        &[1.0, 0.0, 0.5],
    ),
    (
        "transmission_line",
        r#"{"characteristic_impedance": 50, "delay": 1e-9}"#,
        &[1.0, 0.0, 0.5, 0.0, 2e-3, -1e-3],
    ),
    ("thermal_resistance", r#"{"resistance": 10}"#, &[5.0, 0.0]),
    ("thermal_capacitance", r#"{"capacitance": 2}"#, &[5.0, 0.0]),
    (
        "foundry_mosfet",
        r#"{"pdk": "sky130", "model": "sky130_fd_pr__nfet_01v8", "width": 1e-6,
                "length": 1.5e-7}"#,
        &[1.0, 1.0, 0.0, 0.0],
    ),
];

/// The value of each local unknown past those a case gives: a point near the ground, as
/// most of a small circuit's internal nodes and states lie.
pub(crate) const REST: f64 = 0.1;

/// Pads `given` up to `unknowns` entries with [`REST`].
pub(crate) fn values(given: &[f64], unknowns: usize) -> Vec<f64> {
    let mut values = given.to_vec();
    values.resize(unknowns, REST);
    values
}

/// Returns the sorted, deduplicated model names covered by [`CASES`].
pub(crate) fn covered() -> Vec<&'static str> {
    let mut names: Vec<&str> = CASES.iter().map(|&(name, _, _)| name).collect();
    names.sort_unstable();
    names.dedup();
    names
}
