//! Adjoint sensitivities against finite differences of the whole circuit.
#![cfg(test)]

use std::collections::BTreeMap;

use jolty_engine::{
    Ac, Conditions, Cotangent, Cotangents, DcSweep, Derivatives, Differentiated, Method, Options,
    ParameterCotangents, PeriodicSteadyState, Sensitivity, SimulationError, Simulator, Transient,
};
use jolty_ir::Circuit;
use jolty_models::Library;
use jolty_numerics::Complex64;
use serde_json::{Value, json};

/// A circuit whose top component has a ground port `0` and places `instances`.
fn circuit_of(instances: &Value) -> Circuit {
    let document = json!({
        "format": "jolty-circuit", "version": 4, "top": "test",
        "components": {"test": {"ports": [{"name": "0", "role": "ground"}], "instances": instances}}
    });
    Circuit::from_json(&document.to_string()).expect("a valid circuit")
}

/// A source into a resistor and a diode with a series resistance, then a divider: nonlinear,
/// with an internal unknown.
fn diode() -> Value {
    json!([
        {"name": "V1", "of": {"model": "voltage_source"}, "parameters": {"waveform": {"dc": 5.0}},
         "terminals": {"positive": "in", "negative": "0"}},
        {"name": "R1", "of": {"model": "resistor"}, "parameters": {"resistance": 1e3},
         "terminals": {"a": "in", "b": "a"}},
        {"name": "D1", "of": {"model": "diode"},
         "parameters": {"saturation_current": 1e-14, "emission_coefficient": 1.2,
                        "series_resistance": 10.0},
         "terminals": {"anode": "a", "cathode": "0"}},
        {"name": "R2", "of": {"model": "resistor"}, "parameters": {"resistance": 2e3},
         "terminals": {"a": "a", "b": "out"}},
        {"name": "R3", "of": {"model": "resistor"}, "parameters": {"resistance": 3e3},
         "terminals": {"a": "out", "b": "0"}}
    ])
}

/// A common-source stage: an n-channel MOSFET biased into saturation, its drain loaded by
/// a resistor.
fn amplifier() -> Value {
    json!([
        {"name": "VDD", "of": {"model": "voltage_source"}, "parameters": {"waveform": {"dc": 3.3}},
         "terminals": {"positive": "vdd", "negative": "0"}},
        {"name": "VG", "of": {"model": "voltage_source"}, "parameters": {"waveform": {"dc": 1.2}},
         "terminals": {"positive": "gate", "negative": "0"}},
        {"name": "RD", "of": {"model": "resistor"}, "parameters": {"resistance": 1e4},
         "terminals": {"a": "vdd", "b": "drain"}},
        {"name": "M1", "of": {"model": "mosfet"},
         "parameters": {"width": 2e-6, "length": 1e-6, "threshold_voltage": 0.7,
                        "transconductance_parameter": 1e-4, "channel_length_modulation": 0.02},
         "terminals": {"drain": "drain", "gate": "gate", "source": "0", "bulk": "0"}}
    ])
}

/// The common-source stage loaded by a capacitor and driven through a resistor, so its
/// response falls with frequency.
fn loaded_amplifier() -> Value {
    let mut instances = amplifier();
    let placed = instances.as_array_mut().expect("instances");
    placed[1]["terminals"]["positive"] = json!("source");
    placed.push(json!({"name": "RG", "of": {"model": "resistor"},
        "parameters": {"resistance": 1e4}, "terminals": {"a": "source", "b": "gate"}}));
    placed.push(json!({"name": "CL", "of": {"model": "capacitor"},
        "parameters": {"capacitance": 1e-12}, "terminals": {"a": "drain", "b": "0"}}));
    placed.push(json!({"name": "CG", "of": {"model": "capacitor"},
        "parameters": {"capacitance": 2e-13}, "terminals": {"a": "gate", "b": "drain"}}));
    instances
}

/// A sine through a resistor into a diode and a capacitor across it: a rectifier, whose
/// waveforms bend where the diode conducts.
fn rectifier() -> Value {
    json!([
        {"name": "V1", "of": {"model": "voltage_source"},
         "parameters": {"waveform": {"sine": {"offset": 0.2, "amplitude": 1.0,
                                              "frequency": 1e6}}},
         "terminals": {"positive": "in", "negative": "0"}},
        {"name": "R1", "of": {"model": "resistor"}, "parameters": {"resistance": 1e3},
         "terminals": {"a": "in", "b": "a"}},
        {"name": "D1", "of": {"model": "diode"},
         "parameters": {"saturation_current": 1e-14, "series_resistance": 10.0,
                        "transit_time": 1e-9, "junction_capacitance": 2e-12},
         "terminals": {"anode": "a", "cathode": "out"}},
        {"name": "C1", "of": {"model": "capacitor"}, "parameters": {"capacitance": 1e-10},
         "terminals": {"a": "out", "b": "0"}},
        {"name": "R2", "of": {"model": "resistor"}, "parameters": {"resistance": 1e4},
         "terminals": {"a": "out", "b": "0"}}
    ])
}

fn tight() -> Options {
    Options {
        relative_tolerance: 1e-12,
        voltage_tolerance: 1e-15,
        current_tolerance: 1e-18,
        ..Options::default()
    }
}

/// The options `of` runs at: tight, and for a transient or a periodic steady state, with a
/// truncation tolerance no step's error reaches, so every run takes the same steps,
/// growing to the longest, and finite differences see what the parameters move, not where
/// the steps fall.
fn options(of: &Differentiated) -> Options {
    match of {
        Differentiated::Transient(_) | Differentiated::PeriodicSteadyState(_) => Options {
            truncation_tolerance: f64::INFINITY,
            ..tight()
        },
        _ => tight(),
    }
}

/// The simulator of the circuit of `instances` with the parameter `name` of `device` at
/// `value`.
fn moved(instances: &Value, device: &str, name: &str, value: f64) -> Simulator {
    let mut parameters = instances
        .as_array()
        .and_then(|placed| placed.iter().find(|instance| instance["name"] == device))
        .expect("the device is placed")["parameters"]
        .clone();
    let slot = name
        .split('.')
        .try_fold(&mut parameters, |slot, key| slot.get_mut(key))
        .expect("the parameter is set");
    *slot = Value::from(value);
    let mut conditions = Conditions::default();
    conditions.overrides.insert(
        device.to_owned(),
        serde_json::from_value(parameters).expect("parameters"),
    );
    Simulator::new(&circuit_of(instances), &Library::standard(), &conditions)
        .expect("an elaborated circuit")
}

/// Each output's value at each point of the analysis `of`, on `simulator`.
fn values(simulator: &mut Simulator, of: &Differentiated, outputs: &[&str]) -> Vec<Vec<Complex64>> {
    let columns: Vec<usize> = outputs
        .iter()
        .map(|output| simulator.unknown(output).expect("an unknown").index())
        .collect();
    let pick = |rows: Vec<Vec<f64>>| -> Vec<Vec<Complex64>> {
        rows.iter()
            .map(|row| columns.iter().map(|&column| row[column].into()).collect())
            .collect()
    };
    match of {
        Differentiated::OperatingPoint => pick(vec![
            simulator
                .operating_point(&tight())
                .expect("a solution")
                .values,
        ]),
        Differentiated::DcSweep(sweep) => {
            pick(simulator.dc_sweep(sweep, &tight()).expect("a sweep").values)
        }
        Differentiated::Ac(ac) => simulator
            .ac(ac, &tight())
            .expect("a response")
            .values
            .iter()
            .map(|row| columns.iter().map(|&column| row[column]).collect())
            .collect(),
        Differentiated::Transient(transient) => pick(
            simulator
                .transient(transient, &options(of))
                .expect("waveforms")
                .values,
        ),
        Differentiated::PeriodicSteadyState(analysis) => pick(
            simulator
                .periodic_steady_state(analysis, &options(of))
                .expect("a periodic steady state")
                .waveforms
                .values,
        ),
    }
}

/// The derivatives of a sensitivity analysis or a JVP, by point, then row, then column, as
/// phasors.
fn complex(derivatives: &Derivatives) -> Vec<Vec<Vec<Complex64>>> {
    match derivatives {
        Derivatives::Real(points) => points
            .iter()
            .map(|outputs| {
                outputs
                    .iter()
                    .map(|row| row.iter().map(|&value| value.into()).collect())
                    .collect()
            })
            .collect(),
        Derivatives::Complex(points) => points.clone(),
    }
}

/// Checks that the sensitivities of `outputs` along `of` in the circuit of `instances` match
/// central finite differences of the whole circuit, solved again with each parameter moved,
/// within `tolerance` of the largest derivative of each output and parameter, and returns
/// the parameters' names.
fn assert_matches(
    instances: &Value,
    outputs: &[&str],
    of: &Differentiated,
    tolerance: f64,
) -> Vec<String> {
    let circuit = circuit_of(instances);
    let mut simulator = Simulator::new(&circuit, &Library::standard(), &Conditions::default())
        .expect("an elaborated circuit");
    let sensitivity = Sensitivity {
        outputs: outputs.iter().map(|&output| output.to_owned()).collect(),
        of: of.clone(),
    };
    let sensitivities = simulator
        .sensitivity(&sensitivity, &options(of))
        .expect("sensitivities");
    let found = complex(&sensitivities.derivatives);
    for (index, parameter) in sensitivities.parameters.iter().enumerate() {
        let step = 1e-4 * parameter.value;
        let (device, name) = (&parameter.instance, &parameter.name);
        let above = values(
            &mut moved(instances, device, name, parameter.value + step),
            of,
            outputs,
        );
        let below = values(
            &mut moved(instances, device, name, parameter.value - step),
            of,
            outputs,
        );
        assert_eq!(above.len(), found.len(), "{device}.{name} moved the points");
        for (output_index, output) in outputs.iter().enumerate() {
            let finite: Vec<Complex64> = above
                .iter()
                .zip(&below)
                .map(|(up, down)| (up[output_index] - down[output_index]) / (2.0 * step))
                .collect();
            let exact: Vec<Complex64> = found
                .iter()
                .map(|point| point[output_index][index])
                .collect();
            let largest = finite
                .iter()
                .chain(&exact)
                .fold(0.0_f64, |most, value| most.max(value.norm()));
            for (point, (exact, finite)) in exact.iter().zip(&finite).enumerate() {
                assert!(
                    (exact - finite).norm() <= tolerance * largest.max(1e-12),
                    "d{output}/d{device}.{name} at point {point}: exact {exact}, finite \
                     differences {finite}"
                );
            }
        }
    }
    sensitivities
        .parameters
        .iter()
        .map(|parameter| format!("{}.{}", parameter.instance, parameter.name))
        .collect()
}

#[test]
fn adjoint_sensitivities_match_finite_differences() {
    let at = Differentiated::OperatingPoint;
    let names = assert_matches(
        &diode(),
        &["a", "out", "V1:current", "D1:junction"],
        &at,
        1e-5,
    );
    assert_eq!(
        names,
        [
            "V1.waveform.dc",
            "R1.resistance",
            "D1.emission_coefficient",
            "D1.saturation_current",
            "D1.series_resistance",
            "R2.resistance",
            "R3.resistance",
        ]
    );
    let names = assert_matches(&amplifier(), &["drain", "VDD:current"], &at, 1e-5);
    assert_eq!(names.len(), 8, "{names:?}");
}

#[test]
fn sensitivities_along_a_sweep_match_finite_differences() {
    let sweep = Differentiated::DcSweep(DcSweep {
        source: "V1".to_owned(),
        parameter: None,
        values: (0..=10).map(|step| f64::from(step) * 0.5).collect(),
    });
    let names = assert_matches(&diode(), &["a", "V1:current"], &sweep, 1e-5);
    assert_eq!(names[0], "V1.waveform.dc");
}

#[test]
fn sensitivities_over_frequency_match_finite_differences() {
    let ac = Differentiated::Ac(Ac {
        source: "VG".to_owned(),
        frequencies: vec![1e3, 1e6, 1e8, 1e9],
    });
    assert_matches(&loaded_amplifier(), &["drain"], &ac, 1e-5);
}

#[test]
fn sensitivities_over_time_match_finite_differences() {
    let transient = Differentiated::Transient(Transient {
        stop: 2e-6,
        max_step: Some(1e-8),
        method: Method::Trapezoidal,
        breakpoints: Vec::new(),
        initial_conditions: BTreeMap::new(),
        operating_point: true,
    });
    assert_matches(&rectifier(), &["a", "out"], &transient, 1e-4);
    let gear = Differentiated::Transient(Transient {
        stop: 2e-6,
        max_step: Some(1e-8),
        method: Method::Gear2,
        breakpoints: Vec::new(),
        initial_conditions: BTreeMap::new(),
        operating_point: true,
    });
    assert_matches(&rectifier(), &["out"], &gear, 1e-4);
}

#[test]
fn sensitivities_along_a_periodic_steady_state_match_finite_differences() {
    // The rectifier's load holds its charge for a period, 10 kΩ across 100 pF, so each
    // parameter moves the state the period starts from as well as the period's own.
    let steady = Differentiated::PeriodicSteadyState(PeriodicSteadyState {
        period: Some(1e-6),
        max_step: Some(1e-8),
        ..PeriodicSteadyState::default()
    });
    assert_matches(&rectifier(), &["a", "out"], &steady, 1e-4);
}

#[test]
fn a_sensitivity_names_unknowns_that_exist() {
    let circuit = circuit_of(&diode());
    let mut simulator =
        Simulator::new(&circuit, &Library::standard(), &Conditions::default()).unwrap();
    let sensitivity = Sensitivity {
        outputs: vec!["nowhere".to_owned()],
        of: Differentiated::OperatingPoint,
    };
    let error = simulator
        .sensitivity(&sensitivity, &Options::default())
        .unwrap_err();
    assert_eq!(error, SimulationError::UnknownOutput("nowhere".to_owned()));
}

#[test]
fn sensitivities_through_an_inductor_and_a_port_match_finite_differences_over_a_period() {
    // A sine into a tank whose conductance reads its inductor's current through a port:
    // the inductor's flux, on its branch current, carries the state as a charge does, and
    // the conductance's derivative with respect to that current shapes every tangent.
    let instances = json!([
        {"name": "V1", "of": {"model": "voltage_source"},
         "parameters": {"waveform": {"sine": {"offset": 0.1, "amplitude": 1.0, "frequency": 5e6}}},
         "terminals": {"positive": "in", "negative": "0"}},
        {"name": "R1", "of": {"model": "resistor"}, "parameters": {"resistance": 100.0},
         "terminals": {"a": "in", "b": "t"}},
        {"name": "C1", "of": {"model": "capacitor"}, "parameters": {"capacitance": 1e-9},
         "terminals": {"a": "t", "b": "0"}},
        {"name": "L1", "of": {"model": "inductor"}, "parameters": {"inductance": 1e-6},
         "terminals": {"a": "t", "b": "l"}},
        {"name": "B1", "of": {"model": "behavioral_current_source"},
         "parameters": {"expression": "-6e-3 * v(x) * (1 - (v(x) ^ 2 + (31.6 * i(p)) ^ 2))"},
         "terminals": {"positive": "t", "negative": "0", "x": "t",
                       "p_positive": "l", "p_negative": "0"}},
    ]);
    let steady = Differentiated::PeriodicSteadyState(PeriodicSteadyState {
        period: Some(2e-7),
        max_step: Some(2e-9),
        ..PeriodicSteadyState::default()
    });
    assert_matches(&instances, &["t", "L1:current"], &steady, 1e-4);
}

/// The largest disagreement allowed between the JVP and the VJP, relative to the
/// magnitudes of the terms `vᵢ Jᵢⱼ tⱼ` they sum: the two sum the same terms in other
/// orders, from solves of the same matrices and their transposes, so they differ by the
/// rounding of those sums and solves, which the adjoint's check against the direct method
/// over the corpus measured at 4.8e-13 at worst (elm, 16 November 2026); a term either
/// leaves out shows by its own size.
const TRANSPOSED: f64 = 1e-11;

/// A number between −1 and 1 that `index` picks, the same on every run: tangents and
/// cotangents with no structure a mistake could hide in.
fn picked(index: usize) -> f64 {
    let wrapped = (index * 7919 + 13) % 1000;
    f64::from(u32::try_from(wrapped).expect("below 1000")) / 500.0 - 1.0
}

/// A cotangent of numbers [`picked`] on each of `unknowns` at each of `points`, on values
/// if `real` and else on phasors, and on an oscillator's period if it has one.
fn picked_cotangent(
    (points, unknowns): (usize, usize),
    real: bool,
    period: Option<f64>,
) -> Cotangents {
    let value = |point: usize, unknown: usize| picked(point * unknowns + unknown + 1);
    if real {
        return Cotangents::Real(vec![Cotangent {
            points: (0..points)
                .map(|point| {
                    (
                        point,
                        (0..unknowns).map(|unknown| value(point, unknown)).collect(),
                    )
                })
                .collect(),
            period: period.unwrap_or(0.0),
        }]);
    }
    Cotangents::Complex(vec![Cotangent {
        points: (0..points)
            .map(|point| {
                let row = (0..unknowns).map(|unknown| {
                    Complex64::new(value(point, unknown), value(point + 1, unknown))
                });
                (point, row.collect())
            })
            .collect(),
        period: 0.0,
    }])
}

/// Checks the dot-product test `⟨v, J t⟩ = ⟨Jᵀ v, t⟩` along `of` in the circuit of
/// `instances`, with respect to `parameters`, for a tangent and a cotangent of numbers
/// [`picked`], each parameter moved relative to its value, and that the JVP along the
/// tangent is the Jacobian's columns, which the JVP finds along each parameter, combined
/// by it; to [`TRANSPOSED`] of the magnitudes of the terms. Returns the worst relative
/// disagreement of the two.
fn assert_transposes(instances: &Value, parameters: &[(&str, &str)], of: &Differentiated) -> f64 {
    transposes_with(instances, parameters, of, &options(of))
}

/// [`assert_transposes`] with `options`.
fn transposes_with(
    instances: &Value,
    parameters: &[(&str, &str)],
    of: &Differentiated,
    options: &Options,
) -> f64 {
    let circuit = circuit_of(instances);
    let mut simulator = Simulator::new(&circuit, &Library::standard(), &Conditions::default())
        .expect("an elaborated circuit");
    let names: Vec<(String, String)> = parameters
        .iter()
        .map(|&(instance, name)| (instance.to_owned(), name.to_owned()))
        .collect();
    let units: Vec<Vec<f64>> = (0..names.len())
        .map(|index| {
            (0..names.len())
                .map(|at| f64::from(u8::from(at == index)))
                .collect()
        })
        .collect();
    let columns = simulator
        .jvp(of, &names, &units, options)
        .expect("the Jacobian's columns");
    let tangent: Vec<f64> = columns
        .parameters
        .iter()
        .enumerate()
        .map(|(index, parameter)| picked(index) * parameter.value)
        .collect();
    let along = simulator
        .jvp(of, &names, std::slice::from_ref(&tangent), options)
        .expect("a JVP");
    let pullback = simulator.vjp(of, &names, options).expect("a VJP");
    let jacobian = complex(&columns.tangents);
    let pushed = complex(&along.tangents);
    let unknowns = simulator.unknowns().len();
    let real = matches!(columns.tangents, Derivatives::Real(_));
    let period = columns.period.as_ref().map(|_| picked(0));
    let cotangents = picked_cotangent((jacobian.len(), unknowns), real, period);
    let pulled = simulator
        .pull_back(&pullback, &cotangents, options)
        .expect("a pullback");
    let pulled: Vec<Complex64> = match pulled {
        ParameterCotangents::Real(rows) => rows[0].iter().map(|&value| value.into()).collect(),
        ParameterCotangents::Complex(rows) => rows[0].clone(),
    };
    let cotangent = |point: usize, unknown: usize| match &cotangents {
        Cotangents::Real(each) => Complex64::from(each[0].points[&point][unknown]),
        Cotangents::Complex(each) => each[0].points[&point][unknown],
    };
    // ⟨v, J t⟩, the JVP along t against its columns combined, and the terms' magnitudes;
    // a solve's rounding is relative to the whole of the vector it solves for.
    let (mut forward, mut magnitude) = (Complex64::default(), 0.0);
    for (point, rows) in jacobian.iter().enumerate() {
        let terms_of = |row: &Vec<Complex64>| -> f64 {
            row.iter().zip(&tangent).map(|(j, t)| (j * t).norm()).sum()
        };
        let whole = rows.iter().map(terms_of).fold(f64::MIN_POSITIVE, f64::max);
        for (unknown, row) in rows.iter().enumerate() {
            let combined: Complex64 = row.iter().zip(&tangent).map(|(j, t)| j * t).sum();
            let terms = terms_of(row);
            let found = pushed[point][unknown][0];
            assert!(
                (found - combined).norm() <= TRANSPOSED * whole,
                "J t at point {point}, unknown {unknown}: {found} against {combined}"
            );
            let weight = cotangent(point, unknown);
            forward += weight * found;
            magnitude += weight.norm() * terms;
        }
    }
    if let (Some(weight), Some(derivatives)) = (period, &along.period) {
        forward += Complex64::from(weight * derivatives[0]);
        let columns = columns.period.as_ref().expect("the period's columns");
        let terms: f64 = columns
            .iter()
            .zip(&tangent)
            .map(|(j, t)| (j * t).abs())
            .sum();
        magnitude += weight.abs() * terms;
    }
    let backward: Complex64 = pulled.iter().zip(&tangent).map(|(p, t)| p * t).sum();
    let gap = (forward - backward).norm();
    assert!(
        gap <= TRANSPOSED * magnitude,
        "⟨v, J t⟩ = {forward} against ⟨Jᵀ v, t⟩ = {backward}, terms of {magnitude:e}"
    );
    gap / magnitude
}

#[test]
fn the_jvp_and_the_vjp_are_transposes_of_each_other_at_every_analysis() {
    let diode_parameters = [
        ("V1", "waveform.dc"),
        ("R1", "resistance"),
        ("D1", "saturation_current"),
        ("D1", "series_resistance"),
        ("R3", "resistance"),
    ];
    let mut worst = assert_transposes(&diode(), &diode_parameters, &Differentiated::OperatingPoint);
    let sweep = Differentiated::DcSweep(DcSweep {
        source: "V1".to_owned(),
        parameter: None,
        values: (0..=10).map(|step| f64::from(step) * 0.5).collect(),
    });
    worst = worst.max(assert_transposes(&diode(), &diode_parameters, &sweep));
    let ac = Differentiated::Ac(Ac {
        source: "VG".to_owned(),
        frequencies: vec![1e3, 1e6, 1e8, 1e9],
    });
    let amplifier_parameters = [
        ("RD", "resistance"),
        ("M1", "width"),
        ("M1", "threshold_voltage"),
        ("CL", "capacitance"),
        ("RG", "resistance"),
    ];
    worst = worst.max(assert_transposes(
        &loaded_amplifier(),
        &amplifier_parameters,
        &ac,
    ));
    let rectifier_parameters = [
        ("V1", "waveform.sine.amplitude"),
        ("R1", "resistance"),
        ("D1", "saturation_current"),
        ("D1", "junction_capacitance"),
        ("C1", "capacitance"),
        ("R2", "resistance"),
    ];
    for method in [Method::Trapezoidal, Method::Gear2] {
        let transient = Differentiated::Transient(Transient {
            stop: 2e-6,
            max_step: Some(1e-8),
            method,
            breakpoints: Vec::new(),
            initial_conditions: BTreeMap::new(),
            operating_point: true,
        });
        worst = worst.max(assert_transposes(
            &rectifier(),
            &rectifier_parameters,
            &transient,
        ));
    }
    let steady = Differentiated::PeriodicSteadyState(PeriodicSteadyState {
        period: Some(1e-6),
        max_step: Some(1e-8),
        ..PeriodicSteadyState::default()
    });
    worst = worst.max(assert_transposes(
        &rectifier(),
        &rectifier_parameters,
        &steady,
    ));
    // A tank whose conductance holds its radius through an impedance of its own: an
    // oscillator, whose period is one more output, at the tolerances its period's
    // derivatives are checked at in `periodic.rs`.
    let (inductance, capacitance): (f64, f64) = (1e-6, 1e-9);
    let impedance = (inductance / capacitance).sqrt();
    let expression = format!(
        "-{:e} * v(x) * (1 - (v(x) ^ 2 + ({impedance:e} * i(p)) ^ 2))",
        0.2 / impedance
    );
    let tank = json!([
        {"name": "C1", "of": {"model": "capacitor"}, "parameters": {"capacitance": capacitance},
         "terminals": {"a": "t", "b": "0"}},
        {"name": "L1", "of": {"model": "inductor"}, "parameters": {"inductance": inductance},
         "terminals": {"a": "t", "b": "l"}},
        {"name": "B1", "of": {"model": "behavioral_current_source"},
         "parameters": {"expression": expression},
         "terminals": {"positive": "t", "negative": "0", "x": "t",
                       "p_positive": "l", "p_negative": "0"}},
    ]);
    let oscillator = Differentiated::PeriodicSteadyState(PeriodicSteadyState::default());
    let tank_options = Options {
        relative_tolerance: 1e-10,
        voltage_tolerance: 1e-13,
        current_tolerance: 1e-16,
        truncation_tolerance: f64::INFINITY,
        ..Options::default()
    };
    let tank_parameters = [("C1", "capacitance"), ("L1", "inductance")];
    worst = worst.max(transposes_with(
        &tank,
        &tank_parameters,
        &oscillator,
        &tank_options,
    ));
    assert!(worst <= TRANSPOSED, "worst {worst:e}");
}

/// A VJP along more of one device's parameters than a twin carries, which it evaluates in
/// chunks, pulls a cotangent back to each parameter bit for bit as a VJP along that
/// parameter alone does, at an operating point and at every step of a transient.
#[test]
fn a_vjp_in_chunks_pulls_back_to_each_parameter_as_a_vjp_along_it_alone() {
    let mut instances = rectifier();
    instances[2]["parameters"] = json!({
        "saturation_current": 1e-14, "emission_coefficient": 1.05, "series_resistance": 10.0,
        "junction_capacitance": 2e-12, "junction_potential": 0.8, "grading_coefficient": 0.4,
        "depletion_coefficient": 0.45, "transit_time": 1e-9,
        "sidewall_saturation_current": 1e-16, "nominal_temperature": 300.15
    });
    let names: Vec<(String, String)> = instances[2]["parameters"]
        .as_object()
        .expect("the diode's parameters")
        .keys()
        .map(|name| ("D1".to_owned(), name.clone()))
        .chain([
            ("R1".to_owned(), "resistance".to_owned()),
            ("C1".to_owned(), "capacitance".to_owned()),
        ])
        .collect();
    let circuit = circuit_of(&instances);
    let transient = Differentiated::Transient(Transient {
        stop: 2e-6,
        max_step: Some(1e-8),
        method: Method::Trapezoidal,
        breakpoints: Vec::new(),
        initial_conditions: BTreeMap::new(),
        operating_point: true,
    });
    for (of, points) in [
        (Differentiated::OperatingPoint, vec![0]),
        (transient, vec![0, 50, 150]),
    ] {
        let options = options(&of);
        let mut simulator = Simulator::new(&circuit, &Library::standard(), &Conditions::default())
            .expect("an elaborated circuit");
        let unknowns = simulator.unknowns().len();
        let cotangents = Cotangents::Real(vec![Cotangent {
            points: points
                .iter()
                .map(|&point| {
                    let values = (0..unknowns)
                        .map(|unknown| picked(point * unknowns + unknown + 1))
                        .collect();
                    (point, values)
                })
                .collect(),
            period: 0.0,
        }]);
        let mut pulled = |names: &[(String, String)]| -> Vec<f64> {
            let pullback = simulator.vjp(&of, names, &options).expect("a VJP");
            let pulled = simulator
                .pull_back(&pullback, &cotangents, &options)
                .expect("a pullback");
            let ParameterCotangents::Real(rows) = pulled else {
                panic!("real cotangents");
            };
            rows[0].clone()
        };
        let together = pulled(&names);
        for (index, name) in names.iter().enumerate() {
            let [alone] = pulled(std::slice::from_ref(name))[..] else {
                panic!("one cotangent");
            };
            assert_eq!(
                together[index].to_bits(),
                alone.to_bits(),
                "{name:?}: {} among all, {alone} alone",
                together[index]
            );
        }
    }
}

/// A 1 V source through 1 kΩ into 1 nF: an RC of time constant 1 µs.
fn rc() -> Value {
    json!([
        {"name": "V1", "of": {"model": "voltage_source"}, "parameters": {"waveform": {"dc": 1.0}},
         "terminals": {"positive": "in", "negative": "0"}},
        {"name": "R1", "of": {"model": "resistor"}, "parameters": {"resistance": 1e3},
         "terminals": {"a": "in", "b": "out"}},
        {"name": "C1", "of": {"model": "capacitor"}, "parameters": {"capacitance": 1e-9},
         "terminals": {"a": "out", "b": "0"}}
    ])
}

#[test]
fn a_transient_from_initial_conditions_moves_as_the_state_it_starts_from() {
    // From `out` = 0 V, `out = V (1 − e^(−t/RC))`: `d out/dR = −V t/(R²C) e^(−t/RC)`,
    // `d out/dV = 1 − e^(−t/RC)` and `d out/dC = −V t/(RC²) e^(−t/RC)`, the same whether the
    // operating point is solved with `out` held there, or not solved at all, every other
    // unknown starting at zero. The steps' trapezoidal rule errs by `(h/τ)² (t/τ) / 12` of
    // the waveform for steps of `h`, 7e-7 by `t = 2τ` for `h = τ/500`, and its derivatives
    // alike; held to 1e-5 of each derivative's largest, which leaves room for the backward
    // Euler steps from the start, whose error is `h₀/2τ` for the first step's `h₀`.
    let (resistance, capacitance, level): (f64, f64, f64) = (1e3, 1e-9, 1.0);
    let tau = resistance * capacitance;
    let names = [
        ("R1".to_owned(), "resistance".to_owned()),
        ("V1".to_owned(), "waveform.dc".to_owned()),
        ("C1".to_owned(), "capacitance".to_owned()),
    ];
    let units = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    for operating_point in [false, true] {
        let transient = Differentiated::Transient(Transient {
            stop: 2.0 * tau,
            max_step: Some(tau / 500.0),
            method: Method::Trapezoidal,
            breakpoints: Vec::new(),
            initial_conditions: BTreeMap::from([("out".to_owned(), 0.0)]),
            operating_point,
        });
        let mut simulator = Simulator::new(
            &circuit_of(&rc()),
            &Library::standard(),
            &Conditions::default(),
        )
        .expect("an elaborated circuit");
        let options = options(&transient);
        let found = simulator
            .jvp(&transient, &names, &units, &options)
            .expect("the RC's JVP");
        let jolty_engine::AnalysisResult::Transient(waveforms) = &found.result else {
            panic!("a transient's waveforms");
        };
        let Derivatives::Real(points) = &found.tangents else {
            panic!("real derivatives");
        };
        let out = simulator.unknown("out").expect("out").index();
        let exact = |time: f64| {
            let decay = (-time / tau).exp();
            [
                -level * time / (resistance * resistance * capacitance) * decay,
                1.0 - decay,
                -level * time / (resistance * capacitance * capacitance) * decay,
            ]
        };
        let largest = [
            level / (resistance * std::f64::consts::E),
            1.0,
            level / (capacitance * std::f64::consts::E),
        ];
        for (time, at) in waveforms.times.iter().zip(points) {
            for ((found, exact), largest) in at[out].iter().zip(exact(*time)).zip(largest) {
                assert!(
                    (found - exact).abs() <= 1e-5 * largest,
                    "from the operating point {operating_point}, at {time:e} s: {found:e} \
                     against {exact:e}"
                );
            }
        }
        // The VJP of the cotangent on `out` at the middle and the last points is those rows
        // of the JVP's columns.
        let pullback = simulator
            .vjp(&transient, &names, &options)
            .expect("the RC's VJP");
        let unknowns = simulator.unknowns().len();
        let last = points.len() - 1;
        let rows: Vec<usize> = vec![last / 2, last];
        let cotangents = Cotangents::Real(
            rows.iter()
                .map(|&point| {
                    let mut values = vec![0.0; unknowns];
                    values[out] = 1.0;
                    Cotangent {
                        points: BTreeMap::from([(point, values)]),
                        period: 0.0,
                    }
                })
                .collect(),
        );
        let ParameterCotangents::Real(pulled) = simulator
            .pull_back(&pullback, &cotangents, &options)
            .expect("a pullback")
        else {
            panic!("real cotangents");
        };
        for (row, pulled) in rows.iter().zip(&pulled) {
            for (found, exact) in pulled.iter().zip(&points[*row][out]) {
                assert!(
                    (found - exact).abs() <= TRANSPOSED * exact.abs(),
                    "from the operating point {operating_point}, at point {row}: VJP {found:e} \
                     against JVP {exact:e}"
                );
            }
        }
    }
}
