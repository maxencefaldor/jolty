//! Adjoint sensitivities against finite differences of the whole circuit.
#![cfg(test)]

#[path = "common/capacitor.rs"]
mod capacitor;
mod common;
#[path = "common/dc.rs"]
mod dc;
#[path = "common/derivatives.rs"]
mod derivatives;
#[path = "common/inductor.rs"]
mod inductor;
#[path = "common/sim.rs"]
mod sim;
#[path = "common/sine.rs"]
mod sine;
#[path = "common/transient.rs"]
mod transient;

use std::collections::BTreeMap;
use std::f64::consts::TAU;

use capacitor::capacitor;
use common::{circuit, device, resistor, two_terminal, voltage_source};
use dc::voltage_dc;
use derivatives::{moved, rectifier, tight_options};
use inductor::inductor;
use jolty_engine::{
    Ac, Cotangent, Cotangents, DcSweep, Derivatives, Differentiated, LOOP_GAIN, Method, Mismatch,
    Options, Parallelism, ParameterCotangents, PeriodicSteadyState, RandomVariable, Sensitivity,
    SimulationError, Simulator, Stability, Transient,
};
use jolty_ir::{Circuit, Instance};
use jolty_numerics::Complex64;
use rayon::ThreadPoolBuilder;
use serde_json::json;
use sim::simulator;
use sine::voltage_sine;
use transient::transient;

/// Builds a nonlinear diode-divider circuit with series resistance and an internal unknown.
fn diode() -> Circuit {
    circuit([
        voltage_dc("V1", "in", 5.0),
        resistor("R1", 1e3, "in", "a"),
        device(
            "D1",
            "diode",
            json!({"saturation_current": 1e-14, "emission_coefficient": 1.2,
                   "series_resistance": 10.0}),
            &[("anode", "a"), ("cathode", "0")],
        ),
        resistor("R2", 2e3, "a", "out"),
        resistor("R3", 3e3, "out", "0"),
    ])
}

/// Builds a resistively loaded n-channel MOSFET common-source stage biased in saturation.
fn amplifier(gate_node: &str) -> Vec<Instance> {
    vec![
        voltage_dc("VDD", "vdd", 3.3),
        voltage_dc("VG", gate_node, 1.2),
        resistor("RD", 1e4, "vdd", "drain"),
        device(
            "M1",
            "mosfet",
            json!({"width": 2e-6, "length": 1e-6, "threshold_voltage": 0.7,
                   "transconductance_parameter": 1e-4, "channel_length_modulation": 0.02}),
            &[
                ("drain", "drain"),
                ("gate", "gate"),
                ("source", "0"),
                ("bulk", "0"),
            ],
        ),
    ]
}

/// Builds the common-source stage with source resistance and gate/load capacitances.
fn loaded_amplifier() -> Vec<Instance> {
    let mut instances = amplifier("source");
    instances.extend([
        resistor("RG", 1e4, "source", "gate"),
        capacitor("CL", 1e-12, "drain", "0"),
        capacitor("CG", 2e-13, "gate", "drain"),
    ]);
    instances
}

/// Returns tight solver options for `of`, disabling truncation step control on transients
/// and periodic steady states so finite differences hold step placement fixed.
fn options(of: &Differentiated) -> Options {
    tight_options(matches!(
        of,
        Differentiated::Transient(_) | Differentiated::PeriodicSteadyState(_)
    ))
}

/// Evaluates every output in `outputs` at each point of analysis `of` on `simulator`.
fn values(simulator: &mut Simulator, of: &Differentiated, outputs: &[&str]) -> Vec<Vec<Complex64>> {
    if let Differentiated::Stability(stability) = of {
        return simulator
            .stability(stability, &tight_options(false))
            .expect("loop gain")
            .gains
            .into_iter()
            .map(|gain| vec![gain; outputs.len()])
            .collect();
    }
    let columns: Vec<usize> = outputs
        .iter()
        .map(|output| simulator.unknown(output).expect("an unknown").index())
        .collect();
    let pick = |rows: Vec<Vec<f64>>| -> Vec<Vec<Complex64>> {
        rows.iter()
            .map(|row| columns.iter().map(|&column| row[column].into()).collect())
            .collect()
    };
    let tight = tight_options(false);
    match of {
        Differentiated::OperatingPoint => pick(vec![
            simulator
                .operating_point(&tight)
                .expect("a solution")
                .values,
        ]),
        Differentiated::DcSweep(sweep) => {
            pick(simulator.dc_sweep(sweep, &tight).expect("a sweep").values)
        }
        Differentiated::Ac(ac) => simulator
            .ac(ac, &tight)
            .expect("a response")
            .values
            .iter()
            .map(|row| columns.iter().map(|&column| row[column]).collect())
            .collect(),
        Differentiated::Stability(_) => Vec::new(),
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
                .transient
                .values,
        ),
    }
}

/// Converts sensitivity or JVP `derivatives` into complex phasor tables indexed by point,
/// row, and column.
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

/// Checks that the sensitivities of `outputs` along `of` in `circuit` match central finite
/// differences of the whole circuit, solved again with each parameter moved, within
/// `tolerance` of the largest derivative of each output and parameter, and returns the
/// parameters' names.
fn assert_matches(
    circuit: &Circuit,
    outputs: &[&str],
    of: &Differentiated,
    tolerance: f64,
) -> Vec<String> {
    let mut simulator = simulator(circuit);
    let sensitivity = Sensitivity {
        outputs: outputs.iter().map(|&output| output.to_owned()).collect(),
        of: of.clone(),
    };
    let sensitivities = simulator
        .sensitivity(&sensitivity, &options(of))
        .expect("sensitivities");
    let found = complex(&sensitivities.derivatives);
    for (index, parameter) in sensitivities.parameters.iter().enumerate() {
        let step = if parameter.value == 0.0 {
            1e-4
        } else {
            1e-4 * parameter.value
        };
        let (device, name) = (&parameter.instance, &parameter.name);
        let above = values(
            &mut moved(circuit, device, name, parameter.value + step),
            of,
            outputs,
        );
        let below = values(
            &mut moved(circuit, device, name, parameter.value - step),
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
    let names = assert_matches(
        &circuit(amplifier("gate")),
        &["drain", "VDD:current"],
        &at,
        1e-5,
    );
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
    assert_matches(&circuit(loaded_amplifier()), &["drain"], &ac, 1e-5);
}

#[test]
fn sensitivities_over_time_match_finite_differences() {
    let rectifier = circuit(rectifier());
    let trapezoidal = Differentiated::Transient(transient(2e-6, Some(1e-8), Method::Trapezoidal));
    assert_matches(&rectifier, &["a", "out"], &trapezoidal, 1e-4);
    let gear = Differentiated::Transient(transient(2e-6, Some(1e-8), Method::Gear2));
    assert_matches(&rectifier, &["out"], &gear, 1e-4);
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
    assert_matches(&circuit(rectifier()), &["a", "out"], &steady, 1e-4);
}

#[test]
fn a_sensitivity_names_unknowns_that_exist() {
    let mut simulator = simulator(&diode());
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
    let tank = circuit([
        voltage_sine("V1", "in", 0.1, 1.0, 5e6),
        resistor("R1", 100.0, "in", "t"),
        capacitor("C1", 1e-9, "t", "0"),
        inductor("L1", 1e-6, "t", "l"),
        device(
            "B1",
            "behavioral_current_source",
            json!({"expression": "-6e-3 * v(x) * (1 - (v(x) ^ 2 + (31.6 * i(p)) ^ 2))"}),
            &[
                ("positive", "t"),
                ("negative", "0"),
                ("x", "t"),
                ("p_positive", "l"),
                ("p_negative", "0"),
            ],
        ),
    ]);
    let steady = Differentiated::PeriodicSteadyState(PeriodicSteadyState {
        period: Some(2e-7),
        max_step: Some(2e-9),
        ..PeriodicSteadyState::default()
    });
    assert_matches(&tank, &["t", "L1:current"], &steady, 1e-4);
}

/// Relative tolerance for JVP-VJP transpose agreement (`1e-11`).
///
/// Forward and adjoint passes sum identical `vᵢ Jᵢⱼ tⱼ` terms in opposite orders using
/// transposed factorizations; corpus checks measure worst rounding at `4.8e-13` (elm,
/// 16 November 2026), well below any omitted term.
const TRANSPOSED: f64 = 1e-11;

/// Computes a deterministic pseudorandom value in `[-1, 1)` for `index` so accidental
/// cancellation cannot mask errors.
fn picked(index: usize) -> f64 {
    let wrapped = (index * 7919 + 13) % 1000;
    f64::from(u32::try_from(wrapped).expect("below 1000")) / 500.0 - 1.0
}

/// Builds deterministic [`picked`] cotangents across `points` and `unknowns`, real when
/// `real` is true and complex otherwise, with optional oscillator `period` weight.
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

/// Checks the dot-product identity `⟨v, J t⟩ = ⟨Jᵀ v, t⟩` along `of` in `circuit` with
/// respect to `parameters` using [`picked`] tangents and cotangents.
///
/// Verifies both transpose agreement within [`TRANSPOSED`] and consistency with columnwise
/// JVP evaluations, returning the worst relative disagreement.
fn assert_transposes(circuit: &Circuit, parameters: &[(&str, &str)], of: &Differentiated) -> f64 {
    transposes_with(circuit, parameters, of, &options(of))
}

/// Runs [`assert_transposes`] with explicit solver `options`.
fn transposes_with(
    circuit: &Circuit,
    parameters: &[(&str, &str)],
    of: &Differentiated,
    options: &Options,
) -> f64 {
    let mut simulator = simulator(circuit);
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
        .map(|(index, parameter)| {
            let scale = if parameter.value == 0.0 {
                1.0
            } else {
                parameter.value
            };
            picked(index) * scale
        })
        .collect();
    let along = simulator
        .jvp(of, &names, std::slice::from_ref(&tangent), options)
        .expect("a JVP");
    let pullback = simulator.vjp(of, &names, options).expect("a VJP");
    let jacobian = complex(&columns.tangents);
    let pushed = complex(&along.tangents);
    let outputs = if matches!(of, Differentiated::Stability(_)) {
        1
    } else {
        simulator.unknowns().len()
    };
    let real = matches!(columns.tangents, Derivatives::Real(_));
    let period = columns.period.as_ref().map(|_| picked(0));
    let cotangents = picked_cotangent((jacobian.len(), outputs), real, period);
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
    let diode = diode();
    let mut worst = assert_transposes(&diode, &diode_parameters, &Differentiated::OperatingPoint);
    let sweep = Differentiated::DcSweep(DcSweep {
        source: "V1".to_owned(),
        parameter: None,
        values: (0..=10).map(|step| f64::from(step) * 0.5).collect(),
    });
    worst = worst.max(assert_transposes(&diode, &diode_parameters, &sweep));
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
    let loaded = circuit(loaded_amplifier());
    worst = worst.max(assert_transposes(&loaded, &amplifier_parameters, &ac));
    let rectifier_parameters = [
        ("V1", "waveform.sine.amplitude"),
        ("R1", "resistance"),
        ("D1", "saturation_current"),
        ("D1", "junction_capacitance"),
        ("C1", "capacitance"),
        ("R2", "resistance"),
    ];
    let rectifier = circuit(rectifier());
    for method in [Method::Trapezoidal, Method::Gear2] {
        let transient = Differentiated::Transient(transient(2e-6, Some(1e-8), method));
        worst = worst.max(assert_transposes(
            &rectifier,
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
        &rectifier,
        &rectifier_parameters,
        &steady,
    ));
    // Regulated LC oscillator with fixed-impedance conductance; period becomes an extra
    // output checked at the tolerances from `periodic.rs`.
    let (inductance, capacitance): (f64, f64) = (1e-6, 1e-9);
    let impedance = (inductance / capacitance).sqrt();
    let expression = format!(
        "-{:e} * v(x) * (1 - (v(x) ^ 2 + ({impedance:e} * i(p)) ^ 2))",
        0.2 / impedance
    );
    let tank = circuit([
        capacitor("C1", capacitance, "t", "0"),
        inductor("L1", inductance, "t", "l"),
        device(
            "B1",
            "behavioral_current_source",
            json!({"expression": expression}),
            &[
                ("positive", "t"),
                ("negative", "0"),
                ("x", "t"),
                ("p_positive", "l"),
                ("p_negative", "0"),
            ],
        ),
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

/// Verifies that chunked VJP evaluation across more parameters than one twin carries matches
/// single-parameter VJP results bit for bit on operating points and transients.
#[test]
fn a_vjp_in_chunks_pulls_back_to_each_parameter_as_a_vjp_along_it_alone() {
    let mut instances = rectifier();
    instances[2].parameters = serde_json::from_value(json!({
        "saturation_current": 1e-14, "emission_coefficient": 1.05, "series_resistance": 10.0,
        "junction_capacitance": 2e-12, "junction_potential": 0.8, "grading_coefficient": 0.4,
        "depletion_coefficient": 0.45, "transit_time": 1e-9,
        "sidewall_saturation_current": 1e-16, "nominal_temperature": 300.15
    }))
    .expect("the diode's parameters");
    let names: Vec<(String, String)> = instances[2]
        .parameters
        .keys()
        .map(|name| ("D1".to_owned(), name.clone()))
        .chain([
            ("R1".to_owned(), "resistance".to_owned()),
            ("C1".to_owned(), "capacitance".to_owned()),
        ])
        .collect();
    let circuit = circuit(instances);
    let transient = Differentiated::Transient(transient(2e-6, Some(1e-8), Method::Trapezoidal));
    for (of, points) in [
        (Differentiated::OperatingPoint, vec![0]),
        (transient, vec![0, 50, 150]),
    ] {
        let options = options(&of);
        let mut simulator = simulator(&circuit);
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

/// Builds a 1 V RC low-pass circuit (1 kΩ, 1 nF) with a 1 µs time constant.
fn rc() -> Circuit {
    circuit([
        voltage_dc("V1", "in", 1.0),
        resistor("R1", 1e3, "in", "out"),
        capacitor("C1", 1e-9, "out", "0"),
    ])
}

#[test]
fn a_transient_from_initial_conditions_moves_as_the_state_it_starts_from() {
    // Starting from `out` = 0 V gives `out = V (1 - e^(-t/RC))`, regardless of whether the
    // DC operating point runs. Analytical derivatives are `d out/dR = -V t/(R²C) e^(-t/RC)`,
    // `d out/dV = 1 - e^(-t/RC)`, and `d out/dC = -V t/(RC²) e^(-t/RC)`.
    // Trapezoidal error scales as `(h/τ)² (t/τ) / 12` (7e-7 at `t = 2τ`, `h = τ/500`), and
    // initial backward Euler steps add `h₀/2τ`, staying within 1e-5 of peak derivatives.
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
            initial_conditions: BTreeMap::from([("out".to_owned(), 0.0)]),
            operating_point,
            ..transient(2.0 * tau, Some(tau / 500.0), Method::Trapezoidal)
        });
        let mut simulator = simulator(&rc());
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

/// Builds a self-heated diode biased through a resistor and coupled to a thermal RC network.
fn heated_diode(supply: f64, thermal_resistance: f64) -> Vec<Instance> {
    vec![
        voltage_dc("V1", "in", supply),
        resistor("R1", 50.0, "in", "anode"),
        device(
            "D1",
            "diode",
            json!({"saturation_current": 1e-14, "emission_coefficient": 1.2,
                   "series_resistance": 4.0, "junction_capacitance": 2e-9,
                   "transit_time": 1e-9, "thermal": true}),
            &[
                ("anode", "anode"),
                ("cathode", "0"),
                ("thermal", "temperature"),
            ],
        ),
        two_terminal(
            "RTH",
            "thermal_resistance",
            "resistance",
            thermal_resistance,
            "temperature",
            "0",
        ),
        two_terminal(
            "CTH",
            "thermal_capacitance",
            "capacitance",
            1e-8,
            "temperature",
            "0",
        ),
    ]
}

#[test]
fn sensitivities_over_frequency_of_a_heated_circuit_match_finite_differences() {
    // Frequencies spanning both the thermal pole (~80 kHz) and electrical pole (~2 MHz)
    // so central differences with relative step 1e-4 stay well above f64 rounding.
    let ac = Differentiated::Ac(Ac {
        source: "V1".to_owned(),
        frequencies: vec![1e5, 1e6, 1e7],
    });
    let parameters = [
        ("V1", "waveform.dc"),
        ("R1", "resistance"),
        ("D1", "saturation_current"),
        ("D1", "emission_coefficient"),
        ("D1", "series_resistance"),
        ("D1", "junction_capacitance"),
        ("RTH", "resistance"),
        ("CTH", "capacitance"),
    ];
    // Hot case: substantial self-heating shifts both electrical and thermal Jacobians.
    let hot = circuit(heated_diode(2.0, 200.0));
    assert_matches(&hot, &["anode", "temperature"], &ac, 1e-5);
    assert_transposes(&hot, &parameters, &ac);
    // Exact dot-product test also holds across six decades including low frequency.
    let wide_ac = Differentiated::Ac(Ac {
        source: "V1".to_owned(),
        frequencies: vec![1e2, 1e5, 1e8],
    });
    assert_transposes(&hot, &parameters, &wide_ac);
    // Exact ambient case (`ΔT == 0` bit-for-bit while forward-biased and conducting):
    // measure the diode's ambient power first, then sink that exact power with a
    // `heat_source` so `temperature == 0.0` at DC while remaining a live AC thermal unknown.
    let mut reference = simulator(&circuit([
        voltage_dc("V1", "in", 2.0),
        resistor("R1", 50.0, "in", "anode"),
        device(
            "D1",
            "diode",
            json!({"saturation_current": 1e-14, "emission_coefficient": 1.2,
                   "series_resistance": 4.0, "junction_capacitance": 2e-9,
                   "transit_time": 1e-9}),
            &[("anode", "anode"), ("cathode", "0")],
        ),
    ]));
    let tight = tight_options(false);
    let operating = reference.operating_point(&tight).expect("dc");
    let anode = operating.values[reference.unknown("anode").expect("anode").index()];
    let current = -operating.values[reference.unknown("V1:current").expect("V1:current").index()];
    let ambient_power = anode * current;
    let mut cooled = heated_diode(2.0, 200.0);
    cooled.push(two_terminal(
        "Q1",
        "heat_source",
        "power",
        ambient_power,
        "temperature",
        "0",
    ));
    let cooled = circuit(cooled);
    let mut cooled_simulator = simulator(&cooled);
    let cooled_operating = cooled_simulator.operating_point(&tight).expect("dc");
    let rise = cooled_operating.values[cooled_simulator
        .unknown("temperature")
        .expect("temperature")
        .index()];
    assert_eq!(
        (jolty_models::REFERENCE_TEMPERATURE + rise).to_bits(),
        jolty_models::REFERENCE_TEMPERATURE.to_bits(),
        "expected exact ambient temperature, got ΔT = {rise:e} K"
    );
    assert_matches(&cooled, &["anode", "temperature"], &ac, 1e-5);
    let cooled_parameters = [
        ("V1", "waveform.dc"),
        ("R1", "resistance"),
        ("D1", "saturation_current"),
        ("D1", "emission_coefficient"),
        ("D1", "series_resistance"),
        ("D1", "junction_capacitance"),
        ("RTH", "resistance"),
        ("CTH", "capacitance"),
        ("Q1", "power"),
    ];
    assert_transposes(&cooled, &cooled_parameters, &ac);
}

#[test]
fn sensitivities_over_frequency_of_a_behavioral_circuit_match_finite_differences() {
    let circuit = circuit([
        voltage_dc("V1", "in", 1.2),
        resistor("R1", 1e3, "in", "out"),
        device(
            "B1",
            "behavioral_current_source",
            json!({"expression": "2e-3 * tanh(3.0 * v(x)) + 1e-4 * (v(x) ^ 3)"}),
            &[("positive", "out"), ("negative", "0"), ("x", "out")],
        ),
        device(
            "BC1",
            "behavioral_capacitor",
            json!({"expression": "1e-12 * (v(x) + 0.3 * (v(x) ^ 2))"}),
            &[("positive", "out"), ("negative", "0"), ("x", "out")],
        ),
        device(
            "BR1",
            "behavioral_resistor",
            json!({
                "expression": "1.5e3 * (1 + 0.2 * tanh(v(x)))",
                "temperature_coefficient": 2e-3,
                "quadratic_temperature_coefficient": 1e-5,
                "nominal_temperature": 290.0
            }),
            &[("positive", "out"), ("negative", "0"), ("x", "out")],
        ),
        resistor("R2", 2e3, "out", "0"),
    ]);
    let ac = Differentiated::Ac(Ac {
        source: "V1".to_owned(),
        frequencies: vec![1e3, 1e7, 1e9],
    });
    assert_matches(&circuit, &["out"], &ac, 1e-5);
    assert_transposes(
        &circuit,
        &[
            ("V1", "waveform.dc"),
            ("R1", "resistance"),
            ("R2", "resistance"),
            ("BR1", "temperature_coefficient"),
            ("BR1", "quadratic_temperature_coefficient"),
            ("BR1", "nominal_temperature"),
        ],
        &ac,
    );
}

#[test]
fn sensitivities_over_frequency_of_a_linear_circuit_match_finite_differences() {
    let ac = Differentiated::Ac(Ac {
        source: "V1".to_owned(),
        frequencies: vec![1e2, 1e5, 1e7],
    });
    let rc = rc();
    assert_matches(&rc, &["out"], &ac, 1e-5);
    assert_transposes(
        &rc,
        &[
            ("V1", "waveform.dc"),
            ("R1", "resistance"),
            ("C1", "capacitance"),
        ],
        &ac,
    );
}

#[test]
fn ac_derivatives_are_deterministic_across_thread_counts() {
    let mut instances = loaded_amplifier();
    instances.extend([
        device(
            "D1",
            "diode",
            json!({"saturation_current": 1e-14, "series_resistance": 5.0,
                   "junction_capacitance": 1e-12, "thermal": true}),
            &[
                ("anode", "drain"),
                ("cathode", "0"),
                ("thermal", "temperature"),
            ],
        ),
        two_terminal(
            "RTH",
            "thermal_resistance",
            "resistance",
            150.0,
            "temperature",
            "0",
        ),
    ]);
    let circuit = circuit(instances);
    let ac = Differentiated::Ac(Ac {
        source: "VG".to_owned(),
        frequencies: vec![1e2, 1e4, 1e6, 1e7, 1e8, 1e9],
    });
    let names: Vec<(String, String)> = [
        ("RD", "resistance"),
        ("RG", "resistance"),
        ("CL", "capacitance"),
        ("CG", "capacitance"),
        ("M1", "width"),
        ("M1", "threshold_voltage"),
        ("M1", "transconductance_parameter"),
        ("D1", "saturation_current"),
        ("D1", "series_resistance"),
        ("RTH", "resistance"),
    ]
    .into_iter()
    .map(|(device, name)| (device.to_owned(), name.to_owned()))
    .collect();
    let tangent: Vec<f64> = (0..names.len()).map(picked).collect();
    let run = |threads: usize| -> (Vec<u64>, Vec<u64>) {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool");
        pool.install(|| {
            let options = Options {
                parallelism: Parallelism {
                    max_threads: Some(threads),
                    min_solves_per_task: Some(1),
                    ..Parallelism::default()
                },
                ..tight_options(false)
            };
            let mut simulator = simulator(&circuit);
            let jvp = simulator
                .jvp(&ac, &names, std::slice::from_ref(&tangent), &options)
                .expect("jvp");
            let Derivatives::Complex(points) = jvp.tangents else {
                panic!("complex tangents");
            };
            let forward_bits: Vec<u64> = points
                .iter()
                .flat_map(|rows| rows.iter().flat_map(|row| row.iter()))
                .flat_map(|value| [value.re.to_bits(), value.im.to_bits()])
                .collect();
            let cotangents = picked_cotangent((6, simulator.unknowns().len()), false, None);
            let pullback = simulator.vjp(&ac, &names, &options).expect("vjp");
            let ParameterCotangents::Complex(rows) = simulator
                .pull_back(&pullback, &cotangents, &options)
                .expect("pullback")
            else {
                panic!("complex cotangents");
            };
            let backward_bits: Vec<u64> = rows[0]
                .iter()
                .flat_map(|value| [value.re.to_bits(), value.im.to_bits()])
                .collect();
            (forward_bits, backward_bits)
        })
    };
    let baseline = run(1);
    for threads in [2, 4] {
        assert_eq!(run(threads), baseline, "threads={threads}");
    }
}

/// Builds the common-source stage driving a `rungs`-section resistive ladder, returning its
/// circuit, parameter names, and nominal values.
fn amplifier_with_ladder(rungs: usize) -> (Circuit, Vec<(String, String)>, Vec<f64>) {
    let mut instances = loaded_amplifier();
    let mut previous = "drain".to_owned();
    let mut names = vec![
        ("RD".to_owned(), "resistance".to_owned()),
        ("M1".to_owned(), "width".to_owned()),
        ("M1".to_owned(), "threshold_voltage".to_owned()),
        ("CL".to_owned(), "capacitance".to_owned()),
    ];
    let mut nominal = vec![1e4, 2e-6, 0.7, 1e-12];
    for rung in 0..rungs {
        let next = format!("tap{rung}");
        let series = format!("RS{rung}");
        let shunt = format!("RP{rung}");
        instances.push(resistor(&series, 100.0, &previous, &next));
        instances.push(resistor(&shunt, 1e5, &next, "0"));
        names.push((series, "resistance".to_owned()));
        names.push((shunt, "resistance".to_owned()));
        nominal.extend([100.0, 1e5]);
        previous = next;
    }
    (circuit(instances), names, nominal)
}

#[test]
fn ac_vjp_matches_jvp_and_finite_differences_with_many_parameters() {
    let (circuit, names, nominal) = amplifier_with_ladder(250);
    let ac_analysis = Ac {
        source: "VG".to_owned(),
        frequencies: vec![1e3, 1e6, 1e8],
    };
    let ac = Differentiated::Ac(ac_analysis.clone());
    let options = tight_options(false);
    let mut simulator = simulator(&circuit);
    let tangent: Vec<f64> = nominal
        .iter()
        .enumerate()
        .map(|(index, &scale)| picked(index) * scale)
        .collect();
    let jvp = simulator
        .jvp(&ac, &names, std::slice::from_ref(&tangent), &options)
        .expect("jvp");
    let Derivatives::Complex(pushed) = jvp.tangents else {
        panic!("complex tangents");
    };
    let cotangents = picked_cotangent((3, simulator.unknowns().len()), false, None);
    let pullback = simulator.vjp(&ac, &names, &options).expect("vjp");
    let ParameterCotangents::Complex(pulled) = simulator
        .pull_back(&pullback, &cotangents, &options)
        .expect("pullback")
    else {
        panic!("complex cotangents");
    };
    let Cotangents::Complex(weights) = &cotangents else {
        panic!("complex weights");
    };
    let (mut forward, mut magnitude) = (Complex64::default(), 0.0);
    for (point, rows) in pushed.iter().enumerate() {
        for (unknown, row) in rows.iter().enumerate() {
            let weight = weights[0].points[&point][unknown];
            forward += weight * row[0];
            magnitude += weight.norm() * row[0].norm();
        }
    }
    let backward: Complex64 = pulled[0].iter().zip(&tangent).map(|(p, t)| p * t).sum();
    let backward_magnitude: f64 = pulled[0]
        .iter()
        .zip(&tangent)
        .map(|(p, t)| (p * t).norm())
        .sum();
    let gap = (forward - backward).norm();
    assert!(
        gap <= TRANSPOSED * magnitude.max(backward_magnitude),
        "504-parameter AC dot-product test failed: forward={forward}, backward={backward}, \
         gap={gap:e}"
    );
    // Verify sampled VJP entries directly against central finite differences.
    for (sample_index, (device, parameter, base)) in [
        (0, ("RD", "resistance", 1e4)),
        (1, ("M1", "width", 2e-6)),
        (4, ("RS0", "resistance", 100.0)),
        (503, ("RP249", "resistance", 1e5)),
    ] {
        let step = 1e-4 * base;
        let run_objective = |value: f64| -> Complex64 {
            let mut moved_simulator = moved(&circuit, device, parameter, value);
            let response = moved_simulator.ac(&ac_analysis, &options).expect("ac");
            response
                .values
                .iter()
                .enumerate()
                .flat_map(|(point, row)| {
                    row.iter()
                        .enumerate()
                        .map(move |(unknown, &phasor)| weights[0].points[&point][unknown] * phasor)
                })
                .sum()
        };
        let finite = (run_objective(base + step) - run_objective(base - step)) / (2.0 * step);
        let exact = pulled[0][sample_index];
        let scale = exact.norm().max(finite.norm()).max(1e-12);
        assert!(
            (exact - finite).norm() <= 1e-5 * scale,
            "504-parameter AC VJP for {device}.{parameter}: exact {exact}, finite {finite}"
        );
    }
}

#[test]
fn jvp_with_empty_tangents_still_rejects_non_real_parameters() {
    let mut simulator = simulator(&circuit(loaded_amplifier()));
    let error = simulator
        .jvp(
            &Differentiated::OperatingPoint,
            &[("M1".to_owned(), "level".to_owned())],
            &[],
            &tight_options(false),
        )
        .unwrap_err();
    assert!(
        matches!(
            &error,
            SimulationError::UnknownParameters(missing)
                if missing.as_slice() == [("M1".to_owned(), "level".to_owned())]
        ),
        "expected UnknownParameters for M1.level, got {error:?}"
    );
}

#[test]
fn pull_back_rejects_non_finite_period_when_points_are_empty() {
    let mut simulator = simulator(&rc());
    let tight = tight_options(false);
    let pullback = simulator
        .vjp(
            &Differentiated::OperatingPoint,
            &[("R1".to_owned(), "resistance".to_owned())],
            &tight,
        )
        .expect("vjp");
    for bad_period in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let cotangent = Cotangent::<f64> {
            points: BTreeMap::new(),
            period: bad_period,
        };
        let error = simulator
            .pull_back(&pullback, &Cotangents::Real(vec![cotangent]), &tight)
            .unwrap_err();
        assert!(
            matches!(error, SimulationError::InvalidAnalysis(_)),
            "expected InvalidAnalysis for period={bad_period}, got {error:?}"
        );
    }
}

/// Builds a linear inverting feedback loop (VCVS gain -1000, output resistance 100 Ω,
/// capacitive feedback divider 9 kΩ / 1 kΩ || 1 nF) broken by floating probe `VP`.
fn linear_feedback_loop() -> Circuit {
    circuit([
        device(
            "E1",
            "voltage_controlled_voltage_source",
            json!({"gain": -1e3}),
            &[
                ("positive", "e"),
                ("negative", "0"),
                ("control_positive", "in"),
                ("control_negative", "0"),
            ],
        ),
        resistor("RO", 100.0, "e", "y"),
        voltage_source("VP", "x", "y", json!({"dc": 0.0})),
        resistor("R1", 9e3, "x", "in"),
        resistor("R2", 1e3, "in", "0"),
        capacitor("C1", 1e-9, "in", "0"),
    ])
}

/// Builds a nonlinear common-source MOSFET shunt-shunt feedback amplifier with a nonlinear
/// voltage-dependent capacitor on `drain` and floating probe `VP` in series with feedback.
fn mosfet_feedback_amplifier() -> Circuit {
    circuit([
        voltage_dc("VDD", "vdd", 3.3),
        resistor("RD", 5e3, "vdd", "drain"),
        device(
            "M1",
            "mosfet",
            json!({
                "width": 20e-6,
                "length": 1e-6,
                "threshold_voltage": 0.7,
                "transconductance_parameter": 1e-4,
                "channel_length_modulation": 0.02,
            }),
            &[
                ("drain", "drain"),
                ("gate", "gate"),
                ("source", "0"),
                ("bulk", "0"),
            ],
        ),
        voltage_source("VP", "fb", "drain", json!({"dc": 0.0})),
        resistor("RF", 20e3, "fb", "gate"),
        resistor("RG", 20e3, "gate", "0"),
        capacitor("CL", 2e-12, "drain", "0"),
        capacitor("CF", 5e-13, "fb", "gate"),
        device(
            "BC1",
            "behavioral_capacitor",
            json!({"expression": "1e-12 * (v(x) + 0.3 * (v(x) ^ 2))"}),
            &[("positive", "drain"), ("negative", "0"), ("x", "drain")],
        ),
    ])
}

#[test]
fn stability_sensitivities_match_closed_form_and_finite_differences_on_linear_loop() {
    let loop_circuit = linear_feedback_loop();
    let frequencies = vec![0.0, 1e3, 1e5, 1e6, 1e8];
    let stability = Differentiated::Stability(Stability {
        probe: "VP".to_owned(),
        frequencies: frequencies.clone(),
    });
    assert_matches(&loop_circuit, &[LOOP_GAIN], &stability, 1e-5);
    assert_transposes(
        &loop_circuit,
        &[
            ("E1", "gain"),
            ("RO", "resistance"),
            ("R1", "resistance"),
            ("R2", "resistance"),
            ("C1", "capacitance"),
            ("VP", "waveform.dc"),
        ],
        &stability,
    );
    let mut simulator = simulator(&loop_circuit);
    let found = simulator
        .sensitivity(
            &Sensitivity {
                outputs: vec![LOOP_GAIN.to_owned()],
                of: stability,
            },
            &tight_options(false),
        )
        .expect("stability sensitivity");
    let Derivatives::Complex(points) = found.derivatives else {
        panic!("expected complex stability derivatives");
    };
    let column_of = |instance: &str, name: &str| {
        found
            .parameters
            .iter()
            .position(|parameter| parameter.instance == instance && parameter.name == name)
            .expect("parameter column")
    };
    // Exact closed-form partial derivatives of T = A / (1 + (RO + R1) * (1/R2 + j*omega*C1))
    // with A = -E1.gain = 1e3; agreement holds within solver rounding (1e-9 relative).
    let (gain, output, upper, lower) = (1e3_f64, 100.0_f64, 9e3_f64, 1e3_f64);
    let capacitance = 1e-9_f64;
    for (point, &frequency) in frequencies.iter().enumerate() {
        let omega = TAU * frequency;
        let admittance = Complex64::new(1.0 / lower, omega * capacitance);
        let denominator = 1.0 + (output + upper) * admittance;
        let squared = denominator * denominator;
        let series_derivative = -gain * admittance / squared;
        let lower_derivative = Complex64::new(gain * (output + upper) / (lower * lower), 0.0);
        let cap_derivative = Complex64::new(0.0, -gain * (output + upper) * omega);
        let expected = [
            ("E1", "gain", Complex64::new(-1.0, 0.0) / denominator),
            ("RO", "resistance", series_derivative),
            ("R1", "resistance", series_derivative),
            ("R2", "resistance", lower_derivative / squared),
            ("C1", "capacitance", cap_derivative / squared),
            ("VP", "waveform.dc", Complex64::default()),
        ];
        for (instance, name, wanted) in expected {
            let exact = points[point][0][column_of(instance, name)];
            assert!(
                (exact - wanted).norm() <= 1e-9 * wanted.norm().max(1e-12),
                "dT/d{instance}.{name} at {frequency} Hz: got {exact}, wanted {wanted}"
            );
        }
    }
}

#[test]
fn stability_sensitivities_match_finite_differences_and_transpose_on_nonlinear_amplifier() {
    let amplifier = mosfet_feedback_amplifier();
    let stability = Differentiated::Stability(Stability {
        probe: "VP".to_owned(),
        frequencies: vec![0.0, 1e5, 1e7, 1e8],
    });
    assert_matches(&amplifier, &[LOOP_GAIN], &stability, 1e-5);
    let parameters = [
        ("VDD", "waveform.dc"),
        ("RD", "resistance"),
        ("M1", "width"),
        ("M1", "length"),
        ("M1", "threshold_voltage"),
        ("M1", "transconductance_parameter"),
        ("M1", "channel_length_modulation"),
        ("VP", "waveform.dc"),
        ("RF", "resistance"),
        ("RG", "resistance"),
        ("CL", "capacitance"),
        ("CF", "capacitance"),
    ];
    let worst = assert_transposes(&amplifier, &parameters, &stability);
    assert!(worst <= TRANSPOSED, "worst {worst:e}");
    // Confirm probe DC offset shifts nonlinear operating point and therefore loop gain.
    let mut simulator = simulator(&amplifier);
    let found = simulator
        .sensitivity(
            &Sensitivity {
                outputs: vec![LOOP_GAIN.to_owned()],
                of: stability,
            },
            &tight_options(false),
        )
        .expect("nonlinear stability sensitivity");
    let Derivatives::Complex(points) = found.derivatives else {
        panic!("expected complex derivatives");
    };
    let probe_column = found
        .parameters
        .iter()
        .position(|parameter| parameter.instance == "VP" && parameter.name == "waveform.dc")
        .expect("VP.waveform.dc");
    assert!(
        points[0][0][probe_column].norm() > 0.1,
        "expected nonzero bias-coupled dT/d(VP.waveform.dc), got {}",
        points[0][0][probe_column]
    );
}

#[test]
fn stability_mismatch_combines_loop_gain_sensitivities_by_chain_rule() {
    let amplifier = mosfet_feedback_amplifier();
    let stability = Differentiated::Stability(Stability {
        probe: "VP".to_owned(),
        frequencies: vec![1e3, 1e7, 1e8],
    });
    let options = tight_options(false);
    let mut simulator = simulator(&amplifier);
    let sensitivity = simulator
        .sensitivity(
            &Sensitivity {
                outputs: vec![LOOP_GAIN.to_owned()],
                of: stability.clone(),
            },
            &options,
        )
        .expect("stability sensitivity");
    let variable = RandomVariable {
        instance: None,
        lot: Some("feedback_match".to_owned()),
        name: "delta".to_owned(),
        mean: 0.0,
        standard_deviation: 0.01,
        derivatives: BTreeMap::from([
            (
                "RF".to_owned(),
                BTreeMap::from([("resistance".to_owned(), 2e3)]),
            ),
            (
                "RG".to_owned(),
                BTreeMap::from([("resistance".to_owned(), -1.5e3)]),
            ),
            (
                "M1".to_owned(),
                BTreeMap::from([("threshold_voltage".to_owned(), 0.02)]),
            ),
        ]),
    };
    let mismatch = simulator
        .mismatch(
            &Mismatch {
                outputs: vec![LOOP_GAIN.to_owned()],
                of: stability,
                toleranced: Vec::new(),
                random_variables: vec![variable],
            },
            &options,
        )
        .expect("stability mismatch");
    let (Derivatives::Complex(sensitivity_points), Derivatives::Complex(mismatch_points)) =
        (&sensitivity.derivatives, &mismatch.derivatives)
    else {
        panic!("expected complex derivatives");
    };
    let column_of = |instance: &str, name: &str| {
        sensitivity
            .parameters
            .iter()
            .position(|parameter| parameter.instance == instance && parameter.name == name)
            .expect("parameter column")
    };
    let feedback_column = column_of("RF", "resistance");
    let ground_column = column_of("RG", "resistance");
    let threshold_column = column_of("M1", "threshold_voltage");
    let pairs = sensitivity_points.iter().zip(mismatch_points).enumerate();
    for (point, (sensitivity_row, mismatch_row)) in pairs {
        let wanted = sensitivity_row[0][feedback_column] * 2e3
            - sensitivity_row[0][ground_column] * 1.5e3
            + sensitivity_row[0][threshold_column] * 0.02;
        let got = mismatch_row[0][0];
        // Chain rule combines identical adjoint sensitivities, so disagreement is rounding
        // only (within TRANSPOSED = 1e-11 relative).
        assert!(
            (got - wanted).norm() <= TRANSPOSED * wanted.norm().max(1e-12),
            "point {point}: got {got}, wanted {wanted}"
        );
    }
}

#[test]
fn stability_derivatives_are_deterministic_across_thread_counts() {
    let amplifier = mosfet_feedback_amplifier();
    let stability = Differentiated::Stability(Stability {
        probe: "VP".to_owned(),
        frequencies: vec![1e2, 1e4, 1e6, 1e7, 1e8, 1e9],
    });
    let names: Vec<(String, String)> = [
        ("VDD", "waveform.dc"),
        ("RD", "resistance"),
        ("M1", "width"),
        ("M1", "threshold_voltage"),
        ("VP", "waveform.dc"),
        ("RF", "resistance"),
        ("RG", "resistance"),
        ("CL", "capacitance"),
        ("CF", "capacitance"),
    ]
    .into_iter()
    .map(|(device, name)| (device.to_owned(), name.to_owned()))
    .collect();
    let tangent: Vec<f64> = (0..names.len()).map(picked).collect();
    let run = |threads: usize| -> (Vec<u64>, Vec<u64>) {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool");
        pool.install(|| {
            let options = Options {
                parallelism: Parallelism {
                    max_threads: Some(threads),
                    min_solves_per_task: Some(1),
                    ..Parallelism::default()
                },
                ..tight_options(false)
            };
            let mut simulator = simulator(&amplifier);
            let jvp = simulator
                .jvp(&stability, &names, std::slice::from_ref(&tangent), &options)
                .expect("jvp");
            let Derivatives::Complex(points) = jvp.tangents else {
                panic!("complex tangents");
            };
            let forward_bits: Vec<u64> = points
                .iter()
                .flat_map(|rows| rows.iter().flat_map(|row| row.iter()))
                .flat_map(|value| [value.re.to_bits(), value.im.to_bits()])
                .collect();
            let cotangents = picked_cotangent((6, 1), false, None);
            let pullback = simulator.vjp(&stability, &names, &options).expect("vjp");
            let ParameterCotangents::Complex(rows) = simulator
                .pull_back(&pullback, &cotangents, &options)
                .expect("pullback")
            else {
                panic!("complex cotangents");
            };
            let backward_bits: Vec<u64> = rows[0]
                .iter()
                .flat_map(|value| [value.re.to_bits(), value.im.to_bits()])
                .collect();
            (forward_bits, backward_bits)
        })
    };
    let baseline = run(1);
    for threads in [2, 4] {
        assert_eq!(run(threads), baseline, "threads={threads}");
    }
}

#[test]
fn stability_differentiation_validates_outputs_probes_and_cotangents() {
    let loop_circuit = linear_feedback_loop();
    let stability = Differentiated::Stability(Stability {
        probe: "VP".to_owned(),
        frequencies: vec![1e3, 1e6],
    });
    let options = tight_options(false);
    let mut loop_simulator = simulator(&loop_circuit);
    // Empty outputs retain one empty entry per frequency.
    let empty = loop_simulator
        .sensitivity(
            &Sensitivity {
                outputs: Vec::new(),
                of: stability.clone(),
            },
            &options,
        )
        .expect("empty-output stability sensitivity");
    let Derivatives::Complex(empty_points) = empty.derivatives else {
        panic!("expected complex derivatives");
    };
    assert_eq!(empty_points.len(), 2);
    assert!(empty_points.iter().all(Vec::is_empty));
    // Unrecognized output is rejected.
    assert_eq!(
        loop_simulator.sensitivity(
            &Sensitivity {
                outputs: vec!["in".to_owned()],
                of: stability.clone(),
            },
            &options,
        ),
        Err(SimulationError::UnknownOutput("in".to_owned()))
    );
    // Invalid probes propagate through jvp and vjp.
    let bad_probe = Differentiated::Stability(Stability {
        probe: "R1".to_owned(),
        frequencies: vec![1e3],
    });
    let names = vec![("R1".to_owned(), "resistance".to_owned())];
    assert_eq!(
        loop_simulator.jvp(&bad_probe, &names, &[vec![1.0]], &options),
        Err(SimulationError::NotAProbe("R1".to_owned()))
    );
    assert_eq!(
        loop_simulator.vjp(&bad_probe, &names, &options),
        Err(SimulationError::NotAProbe("R1".to_owned()))
    );
    // Pullback validates empty, domain-mismatched, and wrong-width cotangents.
    let pullback = loop_simulator
        .vjp(&stability, &names, &options)
        .expect("vjp");
    assert_eq!(
        loop_simulator
            .pull_back(&pullback, &Cotangents::Complex(Vec::new()), &options)
            .expect("empty pullback"),
        ParameterCotangents::Complex(Vec::new())
    );
    let wrong_domain = Cotangents::Real(vec![Cotangent {
        points: BTreeMap::from([(0, vec![1.0])]),
        period: 0.0,
    }]);
    assert!(matches!(
        loop_simulator.pull_back(&pullback, &wrong_domain, &options),
        Err(SimulationError::InvalidAnalysis(_))
    ));
    let wrong_width = Cotangents::Complex(vec![Cotangent {
        points: BTreeMap::from([(0, vec![Complex64::new(1.0, 0.0); 2])]),
        period: 0.0,
    }]);
    assert!(matches!(
        loop_simulator.pull_back(&pullback, &wrong_width, &options),
        Err(SimulationError::InvalidAnalysis(_))
    ));
}

#[test]
fn stability_differentiation_refuses_delay_lines() {
    let delayed_circuit = circuit([
        device(
            "E1",
            "voltage_controlled_voltage_source",
            json!({"gain": -1e2}),
            &[
                ("positive", "e"),
                ("negative", "0"),
                ("control_positive", "in"),
                ("control_negative", "0"),
            ],
        ),
        resistor("RO", 50.0, "e", "y"),
        voltage_source("VP", "x", "y", json!({"dc": 0.0})),
        device(
            "T1",
            "transmission_line",
            json!({"characteristic_impedance": 50.0, "delay": 1e-9}),
            &[
                ("positive1", "x"),
                ("negative1", "0"),
                ("positive2", "in"),
                ("negative2", "0"),
            ],
        ),
        resistor("R2", 50.0, "in", "0"),
    ]);
    let stability = Differentiated::Stability(Stability {
        probe: "VP".to_owned(),
        frequencies: vec![1e3, 1e6],
    });
    let options = tight_options(false);
    let mut delayed_simulator = simulator(&delayed_circuit);
    let names = vec![("RO".to_owned(), "resistance".to_owned())];
    assert!(matches!(
        delayed_simulator.jvp(&stability, &names, &[vec![1.0]], &options),
        Err(SimulationError::Delayed { .. })
    ));
    assert!(matches!(
        delayed_simulator.vjp(&stability, &names, &options),
        Err(SimulationError::Delayed { .. })
    ));
    assert!(matches!(
        delayed_simulator.sensitivity(
            &Sensitivity {
                outputs: vec![LOOP_GAIN.to_owned()],
                of: stability,
            },
            &options,
        ),
        Err(SimulationError::Delayed { .. })
    ));
}
