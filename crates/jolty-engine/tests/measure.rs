//! Measures read the engine's own results, and never panic on any numbers.
#![cfg(test)]

use jolty_engine::measure::{
    AnalysisResult, Edge, Measure, MeasureError, Signal, Table, Transfer, Window, dampings,
    natural_frequencies,
};
use jolty_engine::{Analysis, Conditions, Options, Simulator};
use jolty_ir::Circuit;
use jolty_models::Library;
use jolty_numerics::Complex64;
use proptest::prelude::*;
use serde_json::{Value, json};

/// A low pass of `R1`, 1 kΩ, from `in` to `out` and `C1`, 1 µF, from `out` to the ground,
/// driven at `in` by `V1`, 5 V, with `VP`, a probe of 0 V, in series with the resistor.
fn low_pass() -> Simulator {
    let instances = json!([
        {"name": "V1", "of": {"model": "voltage_source"},
         "parameters": {"waveform": {"dc": 5.0}},
         "terminals": {"positive": "in", "negative": "0"}},
        {"name": "VP", "of": {"model": "voltage_source"}, "parameters": {"waveform": {"dc": 0.0}},
         "terminals": {"positive": "mid", "negative": "in"}},
        {"name": "R1", "of": {"model": "resistor"}, "parameters": {"resistance": 1e3},
         "terminals": {"a": "mid", "b": "out"}},
        {"name": "C1", "of": {"model": "capacitor"}, "parameters": {"capacitance": 1e-6},
         "terminals": {"a": "out", "b": "0"}}
    ]);
    let document = json!({
        "format": "jolty-circuit", "version": 4, "top": "test",
        "components": {"test": {"ports": [{"name": "0", "role": "ground"}],
                                "instances": instances}}
    });
    let circuit = Circuit::from_json(&document.to_string()).expect("a valid circuit");
    Simulator::new(&circuit, &Library::standard(), &Conditions::default())
        .expect("the circuit elaborates")
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
    measure.measure(&AnalysisResult::of(outcome).expect("a measurable result"))
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
        &json!({"dc_sweep": {"source": "V1", "levels": [3.0, 1.0, 2.0]}}),
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
    let output = Transfer::Output {
        output: out,
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
        level: 0.0,
        edge: Edge::Rising,
    };
    assert_eq!(read(crossing), Ok(0.75));
    let falling = Measure::Crossing {
        signal: ramp,
        window,
        level: 0.0,
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

/// Any float: ordinary ones, and zeros of either sign, infinities and not a number.
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
            .prop_map(|(output, over)| Transfer::Output { output, over }),
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
        (signal(), window(), number(), number()).prop_map(|(signal, window, low, high)| {
            Measure::RiseTime {
                signal,
                window,
                low,
                high,
            }
        }),
        (signal(), window()).prop_map(|(signal, window)| Measure::Overshoot { signal, window }),
        (signal(), window(), number()).prop_map(|(signal, window, band)| Measure::SettlingTime {
            signal,
            window,
            band
        }),
        (signal(), window(), number(), edge).prop_map(|(signal, window, level, edge)| {
            Measure::Crossing {
                signal,
                window,
                level,
                edge,
            }
        }),
        (signal(), window(), prop::option::of(number())).prop_map(|(signal, window, level)| {
            Measure::Frequency {
                signal,
                window,
                level,
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
    count: usize,
}

fn numbers() -> impl Strategy<Value = Numbers> {
    (0..8usize, 0..7usize, 2..4usize).prop_flat_map(|(kind, length, width)| {
        let rows = prop_oneof![4 => Just(length), 1 => 0..7usize];
        (
            Just(kind),
            prop::collection::vec(number(), length),
            rows,
            Just(width),
            prop::collection::vec(number(), length),
            prop::option::of(prop::collection::vec(complex(), length)),
            0..5usize,
        )
            .prop_flat_map(|(kind, points, rows, width, densities, gains, count)| {
                (
                    Just((kind, points, densities, gains, count)),
                    prop::collection::vec(prop::collection::vec(number(), width), rows),
                    prop::collection::vec(prop::collection::vec(complex(), width), rows),
                )
            })
            .prop_map(
                |((kind, points, densities, gains, count), real, complex)| Numbers {
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
                levels: &self.points,
                values: real,
            },
            3 => AnalysisResult::Transient {
                times: &self.points,
                values: real,
            },
            4 => AnalysisResult::Ac {
                frequencies: &self.points,
                values: &self.complex,
            },
            5 => AnalysisResult::Stability {
                frequencies: &self.points,
                gains: &self.loop_gains,
            },
            6 => AnalysisResult::Noise {
                frequencies: &self.points,
                densities: &self.densities,
                gains: self.gains.as_deref(),
            },
            _ => AnalysisResult::PoleZero(self.complex.first().map_or(&[], Vec::as_slice)),
        }
    }
}

proptest! {
    #[test]
    fn a_measure_reads_any_numbers_without_panicking(measure in measure_strategy(), numbers in numbers()) {
        let _ = measure.measure(&numbers.results());
    }
}
