//! Measures read the engine's own results, and never panic on any numbers.
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
#[path = "common/transient.rs"]
mod transient;

use capacitor::capacitor;
use common::{circuit, device, resistor, voltage_source};
use dc::{dc, voltage_dc};
use derivatives::{moved, rectifier, tight_options};
use inductor::inductor;
use jolty_engine::measure::{
    AnalysisResult, Edge, Measure, MeasureError, Signal, Table, Transfer, Window, dampings,
    natural_frequencies,
};
use jolty_engine::{
    Ac, Analysis, DcSweep, Differentiated, Method, Options, ParameterCotangents,
    PeriodicSteadyState, Simulator,
};
use jolty_ir::Circuit;
use jolty_numerics::Complex64;
use proptest::prelude::*;
use serde_json::{Value, json};
use sim::simulator;
use transient::transient;

/// Builds a low-pass RC simulator (`R1` = 1 kΩ, `C1` = 1 µF) driven by 5 V source `V1`
/// through 0 V series probe `VP`.
fn low_pass() -> Simulator {
    simulator(&circuit([
        voltage_dc("V1", "in", 5.0),
        voltage_source("VP", "mid", "in", dc(0.0)),
        resistor("R1", 1e3, "mid", "out"),
        capacitor("C1", 1e-6, "out", "0"),
    ]))
}

fn run(simulator: &mut Simulator, analysis: &Value) -> jolty_engine::AnalysisResult {
    let analysis: Analysis = serde_json::from_value(analysis.clone()).expect("an analysis");
    simulator
        .run(&analysis, &Options::default())
        .expect("the analysis runs")
}

fn signal(simulator: &Simulator, name: &str) -> Signal {
    let unknown = simulator.unknown(name).expect("an unknown").index();
    Signal {
        unknown,
        reference: None,
    }
}

fn measure(measure: Measure, outcome: &jolty_engine::AnalysisResult) -> Result<f64, MeasureError> {
    measure.measure(&AnalysisResult::from_result(outcome).expect("a measurable result"))
}

#[test]
fn measures_read_a_point_a_sweep_and_a_transient() {
    let mut simulator = low_pass();
    let out = signal(&simulator, "out");
    let relative = Signal {
        reference: Some(signal(&simulator, "in").unknown),
        ..out
    };

    let point = run(&mut simulator, &json!("operating_point"));
    let jolty_engine::AnalysisResult::OperatingPoint(solution) = &point else {
        panic!("an operating point")
    };
    let read = measure(
        Measure::Value {
            signal: relative,
            at: None,
        },
        &point,
    );
    assert_eq!(
        read,
        Ok(solution.values[out.unknown] - solution.values[relative.reference.unwrap()])
    );

    let sweep = run(
        &mut simulator,
        &json!({"dc_sweep": {"source": "V1", "values": [3.0, 1.0, 2.0]}}),
    );
    let jolty_engine::AnalysisResult::DcSweep(swept) = &sweep else {
        panic!("a sweep")
    };
    let column: Vec<f64> = swept.values.iter().map(|row| row[out.unknown]).collect();
    let highest = column.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let window = Window::default();
    assert_eq!(
        measure(
            Measure::Maximum {
                signal: out,
                window
            },
            &sweep
        ),
        Ok(highest)
    );

    let transient = run(&mut simulator, &json!({"transient": {"stop": 5e-3}}));
    let jolty_engine::AnalysisResult::Transient(waveforms) = &transient else {
        panic!("a transient")
    };
    let last = waveforms.values.last().unwrap()[out.unknown];
    let at = Some(waveforms.times[waveforms.times.len() - 1]);
    assert_eq!(
        measure(Measure::Value { signal: out, at }, &transient),
        Ok(last)
    );
    // The capacitor charges from 5 V, its operating point, so the output stays put.
    assert_eq!(
        measure(
            Measure::Overshoot {
                signal: out,
                window
            },
            &transient
        ),
        Err(MeasureError::Flat(waveforms.values[0][out.unknown]))
    );
}

#[test]
fn measures_read_a_response_and_a_loop_gain() {
    let mut simulator = low_pass();
    let out = signal(&simulator, "out");
    let frequencies: Vec<f64> = (0..=60)
        .map(|index| 10f64.powf(f64::from(index) / 10.0))
        .collect();
    let ac = run(
        &mut simulator,
        &json!({"ac": {"source": "V1", "frequencies": frequencies}}),
    );
    let corner = 1.0 / (std::f64::consts::TAU * 1e3 * 1e-6);
    let output = Transfer::Signal {
        signal: out,
        over: None,
    };
    let bandwidth = measure(
        Measure::Bandwidth {
            transfer: output,
            drop: 3.0,
        },
        &ac,
    )
    .unwrap();
    // Interpolated over a tenth of a decade, the 3 dB point lies within 1 % of the corner,
    // where the gain is 3.0103 dB down.
    assert!(
        (bandwidth / corner - 1.0).abs() < 1e-2,
        "{bandwidth} vs {corner}"
    );
    let gain = measure(
        Measure::Gain {
            transfer: output,
            at: 1.0,
        },
        &ac,
    )
    .unwrap();
    let jolty_engine::AnalysisResult::Ac(response) = &ac else {
        panic!("a response")
    };
    let unit = 20.0 * response.values[0][out.unknown].norm().log10();
    // numpy's magnitude, √(1 + r²) times the larger part, rounds three times where `hypot`
    // rounds once, so the two lie within a few units in the last place.
    assert!(
        (gain - unit).abs() <= 8.0 * f64::EPSILON * unit.abs().max(1.0),
        "{gain} vs {unit}"
    );

    let stability = run(
        &mut simulator,
        &json!({"stability": {"probe": "VP", "frequencies": frequencies}}),
    );
    assert_eq!(
        measure(
            Measure::Gain {
                transfer: output,
                at: 1.0
            },
            &stability
        ),
        Err(MeasureError::Kind)
    );
    assert!(
        measure(
            Measure::Gain {
                transfer: Transfer::Loop,
                at: 1.0
            },
            &stability
        )
        .is_ok()
    );
}

#[test]
fn measures_read_noise_poles_and_solutions() {
    let mut simulator = low_pass();
    let frequencies: Vec<f64> = (0..=60)
        .map(|index| 10f64.powf(f64::from(index) / 10.0))
        .collect();
    let noise = run(
        &mut simulator,
        &json!({"noise": {"output": "out", "source": "V1", "frequencies": frequencies}}),
    );
    let jolty_engine::AnalysisResult::Noise(spectrum) = &noise else {
        panic!("a noise spectrum")
    };
    let density = measure(
        Measure::NoiseDensity {
            at: frequencies[10],
            referred: false,
        },
        &noise,
    );
    assert_eq!(density, Ok(spectrum.densities[10].sqrt()));
    let referred = spectrum.referred().unwrap();
    let density = measure(
        Measure::NoiseDensity {
            at: frequencies[10],
            referred: true,
        },
        &noise,
    );
    assert_eq!(density, Ok(referred[10].sqrt()));

    let poles = run(&mut simulator, &json!({"pole_zero": {}}));
    let jolty_engine::AnalysisResult::PoleZero(found) = &poles else {
        panic!("poles")
    };
    let smallest = |values: Vec<f64>| values.into_iter().fold(f64::INFINITY, f64::min);
    assert_eq!(
        measure(Measure::Damping, &poles),
        Ok(smallest(dampings(&found.poles)))
    );
    assert_eq!(
        measure(Measure::DominantPole, &poles),
        Ok(smallest(natural_frequencies(&found.poles)))
    );

    let solutions = run(
        &mut simulator,
        &json!({"operating_points": {"max_solutions": 3}}),
    );
    assert_eq!(measure(Measure::Solutions, &solutions), Ok(1.0));
}

#[test]
fn measures_read_exact_signals_exactly() {
    // A ramp of 2 V/s from -1 V, and the constant 0.5 V, at irregular times.
    let times = [0.0, 0.25, 0.5, 1.5, 2.0];
    let values: Vec<Vec<f64>> = times
        .iter()
        .map(|time| vec![2.0 * time - 1.0, 0.5])
        .collect();
    let results = AnalysisResult::Transient {
        times: &times,
        values: &values,
    };
    let ramp = Signal {
        unknown: 0,
        reference: None,
    };
    let relative = Signal {
        unknown: 0,
        reference: Some(1),
    };
    let window = Window {
        start: Some(0.125),
        stop: Some(1.75),
    };
    let read = |measure: Measure| measure.measure(&results);
    assert_eq!(
        read(Measure::Minimum {
            signal: ramp,
            window
        }),
        Ok(-0.75)
    );
    assert_eq!(
        read(Measure::PeakToPeak {
            signal: relative,
            window
        }),
        Ok(3.25)
    );
    assert_eq!(
        read(Measure::Mean {
            signal: ramp,
            window
        }),
        Ok(0.875)
    );
    let crossing = Measure::Crossing {
        signal: relative,
        window,
        threshold: 0.0,
        edge: Edge::Rising,
    };
    assert_eq!(read(crossing), Ok(0.75));
    let falling = Measure::Crossing {
        signal: ramp,
        window,
        threshold: 0.0,
        edge: Edge::Falling,
    };
    assert_eq!(read(falling), Err(MeasureError::Uncrossed));
    let past = Window {
        start: None,
        stop: Some(3.0),
    };
    assert_eq!(
        read(Measure::Maximum {
            signal: ramp,
            window: past
        }),
        Err(MeasureError::Outside {
            point: jolty_engine::measure::Point::Stop,
            first: 0.0,
            last: 2.0
        })
    );
    assert_eq!(read(Measure::Solutions), Err(MeasureError::Kind));
    let missing = Signal {
        unknown: 2,
        reference: None,
    };
    assert_eq!(
        read(Measure::Maximum {
            signal: missing,
            window
        }),
        Err(MeasureError::Unknown(2))
    );
}

/// Builds a proptest strategy for finite floats, signed zeros, infinities, and `NaN`.
fn number() -> impl Strategy<Value = f64> {
    prop_oneof![
        4 => -10.0..10.0f64,
        1 => any::<f64>(),
        1 => prop::sample::select(vec![0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]),
    ]
}

fn complex() -> impl Strategy<Value = Complex64> {
    (number(), number()).prop_map(|(re, im)| Complex64::new(re, im))
}

fn signal_strategy() -> impl Strategy<Value = Signal> {
    (0..3usize, prop::option::of(0..3usize))
        .prop_map(|(unknown, reference)| Signal { unknown, reference })
}

fn window_strategy() -> impl Strategy<Value = Window> {
    (prop::option::of(number()), prop::option::of(number()))
        .prop_map(|(start, stop)| Window { start, stop })
}

fn transfer_strategy() -> impl Strategy<Value = Transfer> {
    prop_oneof![
        Just(Transfer::Loop),
        (signal_strategy(), prop::option::of(signal_strategy()))
            .prop_map(|(signal, over)| Transfer::Signal { signal, over }),
    ]
}

fn measure_strategy() -> impl Strategy<Value = Measure> {
    let edge = prop::sample::select(vec![Edge::Rising, Edge::Falling, Edge::Either]);
    let signal = signal_strategy;
    let window = window_strategy;
    let transfer = transfer_strategy;
    prop_oneof![
        (signal(), prop::option::of(number()))
            .prop_map(|(signal, at)| Measure::Value { signal, at }),
        (transfer(), number()).prop_map(|(transfer, at)| Measure::Gain { transfer, at }),
        (transfer(), number()).prop_map(|(transfer, at)| Measure::Phase { transfer, at }),
        (transfer(), number()).prop_map(|(transfer, drop)| Measure::Bandwidth { transfer, drop }),
        transfer().prop_map(|transfer| Measure::Crossover { transfer }),
        transfer().prop_map(|transfer| Measure::PhaseMargin { transfer }),
        transfer().prop_map(|transfer| Measure::GainMargin { transfer }),
        (signal(), window()).prop_map(|(signal, window)| Measure::Maximum { signal, window }),
        (signal(), window()).prop_map(|(signal, window)| Measure::Minimum { signal, window }),
        (signal(), window()).prop_map(|(signal, window)| Measure::PeakToPeak { signal, window }),
        (signal(), window()).prop_map(|(signal, window)| Measure::Mean { signal, window }),
        (signal(), window()).prop_map(|(signal, window)| Measure::Rms { signal, window }),
        (signal(), window(), number(), number()).prop_map(|(signal, window, lower, upper)| {
            Measure::RiseTime {
                signal,
                window,
                lower,
                upper,
            }
        }),
        (signal(), window()).prop_map(|(signal, window)| Measure::Overshoot { signal, window }),
        (signal(), window(), number()).prop_map(|(signal, window, band)| Measure::SettlingTime {
            signal,
            window,
            band
        }),
        (signal(), window(), number(), edge).prop_map(|(signal, window, threshold, edge)| {
            Measure::Crossing {
                signal,
                window,
                threshold,
                edge,
            }
        }),
        (signal(), window(), prop::option::of(number())).prop_map(|(signal, window, threshold)| {
            Measure::Frequency {
                signal,
                window,
                threshold,
            }
        }),
        // A fundamental below 10 Hz over times within 10 s keeps a window's whole periods to a
        // few hundred, so a case runs in milliseconds.
        (signal(), window(), -10.0..10.0f64, 0..12usize).prop_map(
            |(signal, window, fundamental, harmonics)| Measure::Distortion {
                signal,
                window,
                fundamental,
                harmonics
            }
        ),
        (number(), any::<bool>()).prop_map(|(at, referred)| Measure::NoiseDensity { at, referred }),
        (
            prop::option::of(number()),
            prop::option::of(number()),
            any::<bool>()
        )
            .prop_map(|(start, stop, referred)| Measure::IntegratedNoise {
                start,
                stop,
                referred
            }),
        Just(Measure::Period),
        Just(Measure::Solutions),
        Just(Measure::Damping),
        Just(Measure::DominantPole),
    ]
}

/// The numbers of some result, of any kind, with as many or other numbers of points as of
/// rows, and two or three unknowns.
#[derive(Debug, Clone)]
struct Numbers {
    kind: usize,
    points: Vec<f64>,
    real: Vec<Vec<f64>>,
    complex: Vec<Vec<Complex64>>,
    densities: Vec<f64>,
    gains: Option<Vec<Complex64>>,
    loop_gains: Vec<Complex64>,
    period: f64,
    count: usize,
}

fn numbers() -> impl Strategy<Value = Numbers> {
    (0..9usize, 0..7usize, 2..4usize).prop_flat_map(|(kind, length, width)| {
        let rows = prop_oneof![4 => Just(length), 1 => 0..7usize];
        (
            Just(kind),
            prop::collection::vec(number(), length),
            rows,
            Just(width),
            prop::collection::vec(number(), length),
            prop::option::of(prop::collection::vec(complex(), length)),
            number(),
            0..5usize,
        )
            .prop_flat_map(
                |(kind, points, rows, width, densities, gains, period, count)| {
                    (
                        Just((kind, points, densities, gains, period, count)),
                        prop::collection::vec(prop::collection::vec(number(), width), rows),
                        prop::collection::vec(prop::collection::vec(complex(), width), rows),
                    )
                },
            )
            .prop_map(
                |((kind, points, densities, gains, period, count), real, complex)| Numbers {
                    kind,
                    points,
                    loop_gains: complex
                        .iter()
                        .filter_map(|row| row.first().copied())
                        .collect(),
                    real,
                    complex,
                    densities,
                    gains,
                    period,
                    count,
                },
            )
    })
}

impl Numbers {
    fn results(&self) -> AnalysisResult<'_> {
        let real: &dyn Table<f64> = &self.real;
        match self.kind {
            0 => AnalysisResult::OperatingPoint(self.real.first().map_or(&[], Vec::as_slice)),
            1 => AnalysisResult::OperatingPoints(self.count),
            2 => AnalysisResult::DcSweep {
                swept: &self.points,
                values: real,
            },
            3 => AnalysisResult::Transient {
                times: &self.points,
                values: real,
            },
            4 => AnalysisResult::PeriodicSteadyState {
                times: &self.points,
                values: real,
                period: self.period,
            },
            5 => AnalysisResult::Ac {
                frequencies: &self.points,
                values: &self.complex,
            },
            6 => AnalysisResult::Stability {
                frequencies: &self.points,
                gains: &self.loop_gains,
            },
            7 => AnalysisResult::Noise {
                frequencies: &self.points,
                densities: &self.densities,
                gains: self.gains.as_deref(),
            },
            _ => AnalysisResult::PoleZero(self.complex.first().map_or(&[], Vec::as_slice)),
        }
    }
}

/// Evaluates `measures` through one [`Simulator::vjp`] and [`Simulator::pull_back`] pass and
/// verifies every parameter derivative against central finite differences on `circuit`.
fn assert_measure_vjp_matches_finite_differences(
    circuit: &Circuit,
    analysis: &Differentiated,
    measures: &[(&str, Measure)],
    parameters: &[(&str, &str, f64)],
    options: &Options,
    tolerance: f64,
) {
    let names: Vec<(String, String)> = parameters
        .iter()
        .map(|&(device_name, param, _)| (device_name.to_owned(), param.to_owned()))
        .collect();
    let mut sim = simulator(circuit);
    let pullback = sim.vjp(analysis, &names, options).expect("vjp succeeds");
    let view = AnalysisResult::from_result(&pullback.result).expect("measurable result");
    let unknowns = sim.unknowns().len();
    let mut exact_rows: Vec<Vec<f64>> = Vec::with_capacity(measures.len());
    for &(label, measure) in measures {
        let (value, cotangent) = measure.cotangent(&view).expect(label);
        let primal = measure.measure(&view).expect(label);
        assert_eq!(
            value.to_bits(),
            primal.to_bits(),
            "{label}: primal mismatch"
        );
        let engine_cotangent = cotangent.to_cotangents(unknowns).expect(label);
        let pulled = sim
            .pull_back(&pullback, &engine_cotangent, options)
            .expect(label);
        let row = match pulled {
            ParameterCotangents::Real(rows) => rows.into_iter().next().expect(label),
            ParameterCotangents::Complex(rows) => rows
                .into_iter()
                .next()
                .expect(label)
                .into_iter()
                .map(|entry| entry.re)
                .collect(),
        };
        exact_rows.push(row);
    }
    let eval_all = |device_name: &str, param: &str, value: f64| -> Vec<f64> {
        let mut perturbed = moved(circuit, device_name, param, value);
        let ran = perturbed
            .vjp(analysis, &[], options)
            .expect("perturbed run");
        let perturbed_view = AnalysisResult::from_result(&ran.result).expect("measurable");
        measures
            .iter()
            .map(|&(label, measure)| measure.measure(&perturbed_view).expect(label))
            .collect()
    };
    for (param_index, &(device_name, param, nominal)) in parameters.iter().enumerate() {
        let step = 1e-4 * nominal.abs();
        let above = eval_all(device_name, param, nominal + step);
        let below = eval_all(device_name, param, nominal - step);
        for (measure_index, &(label, _)) in measures.iter().enumerate() {
            let exact = exact_rows[measure_index][param_index];
            let finite = (above[measure_index] - below[measure_index]) / (2.0 * step);
            let peak_value = above[measure_index].abs().max(below[measure_index].abs());
            // Roundoff floor when the true derivative is identically zero (e.g., AC phase wrt
            // real scalar VCVS gain): LU and central-difference subtraction scale as
            // `~1e3 * eps * |value| / (2 * step)`.
            let roundoff_floor = 1e3 * f64::EPSILON * peak_value.max(1.0) / (2.0 * step);
            let bound = (tolerance * exact.abs().max(finite.abs())).max(roundoff_floor);
            assert!(
                (exact - finite).abs() <= bound,
                "{label} d/d({device_name}.{param}): exact {exact:.9e} vs finite {finite:.9e}"
            );
        }
    }
}

/// Builds a three-pole buffered active RC filter with DC gain 5 V/V for AC measure VJP tests.
fn three_pole_filter() -> Circuit {
    let vcvs = |name: &str, gain: f64, out: &str, control: &str| {
        device(
            name,
            "voltage_controlled_voltage_source",
            json!({"gain": gain}),
            &[
                ("positive", out),
                ("negative", "0"),
                ("control_positive", control),
                ("control_negative", "0"),
            ],
        )
    };
    circuit([
        voltage_dc("V1", "in", 1.0),
        vcvs("E1", 5.0, "g", "in"),
        resistor("R1", 1e3, "g", "p1"),
        capacitor("C1", 1e-6, "p1", "0"),
        vcvs("E2", 1.0, "b1", "p1"),
        resistor("R2", 1e3, "b1", "p2"),
        capacitor("C2", 1e-6, "p2", "0"),
        vcvs("E3", 1.0, "b2", "p2"),
        resistor("R3", 1e3, "b2", "out"),
        capacitor("C3", 1e-6, "out", "0"),
    ])
}

#[test]
fn op_and_dc_measure_cotangents_pull_back_to_parameter_derivatives() {
    let diode_circuit = circuit([
        voltage_dc("V1", "in", 3.0),
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
    ]);
    let diode_sim = simulator(&diode_circuit);
    let out = signal(&diode_sim, "out");
    let tight = tight_options(false);
    let diode_params = [
        ("R1", "resistance", 1e3),
        ("R2", "resistance", 2e3),
        ("R3", "resistance", 3e3),
    ];
    // Central finite differences at relative step 1e-4 agree to 1e-5 on smooth circuits.
    let tolerance = 1e-5;
    assert_measure_vjp_matches_finite_differences(
        &diode_circuit,
        &Differentiated::OperatingPoint,
        &[(
            "op_value",
            Measure::Value {
                signal: out,
                at: None,
            },
        )],
        &diode_params,
        &tight,
        tolerance,
    );
    let sweep = Differentiated::DcSweep(DcSweep {
        source: "V1".to_owned(),
        parameter: None,
        values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
    });
    let window = Window {
        start: Some(1.5),
        stop: Some(4.5),
    };
    assert_measure_vjp_matches_finite_differences(
        &diode_circuit,
        &sweep,
        &[
            (
                "dc_value",
                Measure::Value {
                    signal: out,
                    at: Some(2.4),
                },
            ),
            (
                "dc_max",
                Measure::Maximum {
                    signal: out,
                    window,
                },
            ),
            (
                "dc_mean",
                Measure::Mean {
                    signal: out,
                    window,
                },
            ),
            (
                "dc_rms",
                Measure::Rms {
                    signal: out,
                    window,
                },
            ),
        ],
        &diode_params,
        &tight,
        tolerance,
    );
}

#[test]
fn ac_measure_cotangents_pull_back_to_parameter_derivatives() {
    let filter = three_pole_filter();
    let filter_sim = simulator(&filter);
    let filter_out = signal(&filter_sim, "out");
    let filter_in = signal(&filter_sim, "in");
    let direct = Transfer::Signal {
        signal: filter_out,
        over: None,
    };
    let ratio = Transfer::Signal {
        signal: filter_out,
        over: Some(filter_in),
    };
    let frequencies: Vec<f64> = (10..=40)
        .map(|index| 10f64.powf(f64::from(index) / 10.0))
        .collect();
    let ac = Differentiated::Ac(Ac {
        source: "V1".to_owned(),
        frequencies,
    });
    // Central finite differences at relative step 1e-4 agree to 1e-5 on smooth AC responses.
    let tolerance = 1e-5;
    assert_measure_vjp_matches_finite_differences(
        &filter,
        &ac,
        &[
            (
                "ac_gain",
                Measure::Gain {
                    transfer: direct,
                    at: 180.0,
                },
            ),
            (
                "ac_phase",
                Measure::Phase {
                    transfer: ratio,
                    at: 180.0,
                },
            ),
            (
                "ac_bw",
                Measure::Bandwidth {
                    transfer: direct,
                    drop: 3.0,
                },
            ),
            ("ac_crossover", Measure::Crossover { transfer: ratio }),
            ("ac_pm", Measure::PhaseMargin { transfer: direct }),
            ("ac_gm", Measure::GainMargin { transfer: ratio }),
        ],
        &[
            ("E1", "gain", 5.0),
            ("R1", "resistance", 1e3),
            ("C2", "capacitance", 1e-6),
        ],
        &tight_options(false),
        tolerance,
    );
}

fn rlc_transient_measures(rlc_out: Signal, window: Window) -> Vec<(&'static str, Measure)> {
    vec![
        (
            "tr_val",
            Measure::Value {
                signal: rlc_out,
                at: Some(1.3e-4),
            },
        ),
        (
            "tr_max",
            Measure::Maximum {
                signal: rlc_out,
                window,
            },
        ),
        (
            "tr_min",
            Measure::Minimum {
                signal: rlc_out,
                window,
            },
        ),
        (
            "tr_p2p",
            Measure::PeakToPeak {
                signal: rlc_out,
                window,
            },
        ),
        (
            "tr_mean",
            Measure::Mean {
                signal: rlc_out,
                window,
            },
        ),
        (
            "tr_rms",
            Measure::Rms {
                signal: rlc_out,
                window,
            },
        ),
        (
            "tr_rise",
            Measure::RiseTime {
                signal: rlc_out,
                window,
                lower: 0.1,
                upper: 0.9,
            },
        ),
        (
            "tr_overshoot",
            Measure::Overshoot {
                signal: rlc_out,
                window,
            },
        ),
        (
            "tr_settling",
            Measure::SettlingTime {
                signal: rlc_out,
                window,
                band: 0.05,
            },
        ),
        (
            "tr_crossing",
            Measure::Crossing {
                signal: rlc_out,
                window,
                threshold: 2.5,
                edge: Edge::Rising,
            },
        ),
    ]
}

#[test]
fn transient_measure_cotangents_pull_back_to_parameter_derivatives() {
    // Underdamped series RLC step response with ~35% overshoot and settling inside 600 us.
    let rlc = circuit([
        voltage_dc("V1", "in", 5.0),
        resistor("R1", 20.0, "in", "mid"),
        inductor("L1", 1e-3, "mid", "out"),
        capacitor("C1", 1e-6, "out", "0"),
    ]);
    let rlc_sim = simulator(&rlc);
    let rlc_out = signal(&rlc_sim, "out");
    let step_analysis = Differentiated::Transient(jolty_engine::Transient {
        operating_point: false,
        ..transient(6e-4, Some(2e-6), Method::Trapezoidal)
    });
    let fixed = tight_options(true);
    let measures = rlc_transient_measures(rlc_out, Window::default());
    // Central finite differences at relative step 1e-4 agree to 1e-4 across time-stepped
    // nonlinear crossings and settling thresholds.
    assert_measure_vjp_matches_finite_differences(
        &rlc,
        &step_analysis,
        &measures,
        &[
            ("R1", "resistance", 20.0),
            ("L1", "inductance", 1e-3),
            ("C1", "capacitance", 1e-6),
        ],
        &fixed,
        1e-4,
    );
}

#[test]
fn rectifier_measure_cotangents_pull_back_to_parameter_derivatives() {
    let fixed = tight_options(true);
    // Central finite differences at relative step 1e-4 agree to 1e-4 on nonlinear rectifier
    // harmonics.
    let rect = circuit(rectifier());
    let rect_sim = simulator(&rect);
    let rect_node = signal(&rect_sim, "a");
    let rect_transient =
        Differentiated::Transient(transient(3e-6, Some(1e-8), Method::Trapezoidal));
    assert_measure_vjp_matches_finite_differences(
        &rect,
        &rect_transient,
        &[
            (
                "rect_min",
                Measure::Minimum {
                    signal: rect_node,
                    window: Window::default(),
                },
            ),
            (
                "rect_freq",
                Measure::Frequency {
                    signal: rect_node,
                    window: Window::default(),
                    threshold: None,
                },
            ),
            (
                "rect_thd",
                Measure::Distortion {
                    signal: rect_node,
                    window: Window::default(),
                    fundamental: 1e6,
                    harmonics: 5,
                },
            ),
        ],
        &[("R1", "resistance", 1e3), ("C1", "capacitance", 1e-10)],
        &fixed,
        1e-4,
    );
}

#[test]
fn oscillator_measure_cotangents_pull_back_to_parameter_derivatives() {
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
    let tank_sim = simulator(&tank);
    let tank_node = signal(&tank_sim, "t");
    let oscillator = Differentiated::PeriodicSteadyState(PeriodicSteadyState::default());
    let tank_options = Options {
        relative_tolerance: 1e-10,
        voltage_tolerance: 1e-13,
        current_tolerance: 1e-16,
        truncation_tolerance: f64::INFINITY,
        ..Options::default()
    };
    // Central finite differences at relative step 1e-4 agree to 1e-4 on autonomous oscillator
    // period pullbacks.
    assert_measure_vjp_matches_finite_differences(
        &tank,
        &oscillator,
        &[
            ("osc_period", Measure::Period),
            (
                "osc_value",
                Measure::Value {
                    signal: tank_node,
                    at: Some(0.07e-6),
                },
            ),
            (
                "osc_crossing",
                Measure::Crossing {
                    signal: tank_node,
                    window: Window::default(),
                    threshold: 0.25,
                    edge: Edge::Rising,
                },
            ),
        ],
        &[
            ("C1", "capacitance", capacitance),
            ("L1", "inductance", inductance),
        ],
        &tank_options,
        1e-4,
    );
}

proptest! {
    #[test]
    fn a_measure_reads_any_numbers_without_panicking(measure in measure_strategy(), numbers in numbers()) {
        let _ = measure.measure(&numbers.results());
    }

    #[test]
    fn a_measure_cotangent_reads_any_numbers_without_panicking(
        measure in measure_strategy(),
        numbers in numbers(),
    ) {
        let results = numbers.results();
        if let Ok((value, _)) = measure.cotangent(&results) {
            let primal = measure.measure(&results).expect("cotangent implies primal");
            prop_assert_eq!(value.to_bits(), primal.to_bits());
        }
    }
}
