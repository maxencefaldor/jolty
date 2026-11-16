//! The operating point a simulator remembers, solved once for every analysis about it, and
//! the same as solved afresh.
#![cfg(test)]

use std::fs;
use std::path::Path;

use jolty_engine::{
    Ac, AcResult, Analysis, AnalysisResult, Conditions, DcSweep, DcSweepResult, Differentiated,
    Method, Noise, NoiseResult, OperatingPointResult, OperatingPoints, Options, PoleZero,
    PoleZeroResult, Record, RecordedSeries, RecordedValues, Sensitivity, SensitivityResult,
    Simulator, StabilityResult, TransferFunction, TransferFunctionResult, Transient,
    TransientResult, Work,
};
use jolty_ir::Circuit;
use jolty_models::Library;
use serde_json::{Value, json};

/// A circuit of `instances` whose ground is `0`.
fn circuit(instances: &Value) -> Circuit {
    let document = json!({
        "format": "jolty-circuit", "version": 4, "top": "test",
        "components": {"test": {"ports": [{"name": "0", "role": "ground"}],
                                "instances": instances}}
    });
    Circuit::from_json(&document.to_string()).expect("a valid circuit")
}

/// A sine of 1 V about 1 V into a diode through 1 kΩ, a microfarad across the diode: a
/// nonlinear circuit with a capacitance, which every analysis about an operating point can
/// run.
fn rectifier() -> Circuit {
    circuit(&json!([
        {"name": "V1", "of": {"model": "voltage_source"},
         "parameters": {"waveform": {"sine": {"offset": 1.0, "amplitude": 1.0,
                                              "frequency": 1e3}}},
         "terminals": {"positive": "in", "negative": "0"}},
        {"name": "R1", "of": {"model": "resistor"}, "parameters": {"resistance": 1e3},
         "terminals": {"a": "in", "b": "out"}},
        {"name": "D1", "of": {"model": "diode"}, "parameters": {},
         "terminals": {"anode": "out", "cathode": "0"}},
        {"name": "C1", "of": {"model": "capacitor"}, "parameters": {"capacitance": 1e-6},
         "terminals": {"a": "out", "b": "0"}},
    ]))
}

/// The benchmarks' synchronous buck converter, `benchmarks/speed/circuits/buck.json`: two transistors
/// switching an inductor, whose sparse solver searches for new pivots as they switch.
fn buck() -> Circuit {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/speed/circuits/buck.json");
    let text = fs::read_to_string(path).expect("the circuit reads");
    Circuit::from_json(&text).expect("a circuit")
}

fn simulator(circuit: &Circuit) -> Simulator {
    Simulator::new(circuit, &Library::standard(), &Conditions::default())
        .expect("the circuit elaborates")
}

/// `analysis` run with `options` on a simulator of its own.
fn afresh(circuit: &Circuit, analysis: &Analysis, options: &Options) -> AnalysisResult {
    simulator(circuit)
        .run(analysis, options)
        .expect("the analysis runs")
}

/// `outcome` apart from the work of its transients, which is less on a simulator that
/// remembers the operating point they start from.
fn apart_from_work(outcome: AnalysisResult) -> AnalysisResult {
    match outcome {
        AnalysisResult::Transient(mut waveforms) => {
            waveforms.statistics.work = Work::default();
            AnalysisResult::Transient(waveforms)
        }
        AnalysisResult::Sensitivity(sensitivities) => {
            AnalysisResult::Sensitivity(SensitivityResult {
                result: Box::new(apart_from_work(*sensitivities.result)),
                ..sensitivities
            })
        }
        other => other,
    }
}

/// `outcome` without what it records of the devices.
fn unrecorded(outcome: AnalysisResult) -> AnalysisResult {
    let point = |point: OperatingPointResult| OperatingPointResult {
        observation: RecordedValues::default(),
        ..point
    };
    match outcome {
        AnalysisResult::OperatingPoint(found) => AnalysisResult::OperatingPoint(point(found)),
        AnalysisResult::OperatingPoints(found) => {
            AnalysisResult::OperatingPoints(found.into_iter().map(point).collect())
        }
        AnalysisResult::DcSweep(sweep) => AnalysisResult::DcSweep(DcSweepResult {
            observations: RecordedSeries::default(),
            ..sweep
        }),
        AnalysisResult::Ac(response) => AnalysisResult::Ac(AcResult {
            operating_point: point(response.operating_point),
            currents: Vec::new(),
            ..response
        }),
        AnalysisResult::Transient(waveforms) => AnalysisResult::Transient(TransientResult {
            observations: RecordedSeries::default(),
            ..waveforms
        }),
        AnalysisResult::Sensitivity(found) => AnalysisResult::Sensitivity(SensitivityResult {
            result: Box::new(unrecorded(*found.result)),
            ..found
        }),
        AnalysisResult::Stability(gain) => AnalysisResult::Stability(StabilityResult {
            operating_point: point(gain.operating_point),
            ..gain
        }),
        AnalysisResult::Noise(spectrum) => AnalysisResult::Noise(NoiseResult {
            operating_point: point(spectrum.operating_point),
            ..spectrum
        }),
        AnalysisResult::PoleZero(found) => AnalysisResult::PoleZero(PoleZeroResult {
            operating_point: point(found.operating_point),
            ..found
        }),
        AnalysisResult::TransferFunction(found) => {
            AnalysisResult::TransferFunction(TransferFunctionResult {
                operating_point: point(found.operating_point),
                ..found
            })
        }
    }
}

/// The Newton iterations of the transient `outcome`, if it is one.
fn iterations(outcome: &AnalysisResult) -> Option<usize> {
    match outcome {
        AnalysisResult::Transient(waveforms) => Some(waveforms.statistics.work.iterations),
        _ => None,
    }
}

fn analyses() -> Vec<Analysis> {
    let frequencies = vec![1.0, 1e3, 1e6];
    let transient = Transient {
        stop: 2e-3,
        max_step: Some(2e-5),
        method: Method::Trapezoidal,
        breakpoints: Vec::new(),
        periodic: false,
    };
    vec![
        Analysis::OperatingPoint,
        Analysis::Ac(Ac {
            source: "V1".to_owned(),
            frequencies: frequencies.clone(),
        }),
        Analysis::TransferFunction(TransferFunction {
            source: "V1".to_owned(),
            output: "out".to_owned(),
            reference: None,
        }),
        Analysis::Transient(transient.clone()),
        Analysis::Noise(Noise {
            output: "out".to_owned(),
            reference: None,
            source: Some("V1".to_owned()),
            frequencies,
        }),
        Analysis::PoleZero(PoleZero {
            source: Some("V1".to_owned()),
            output: Some("out".to_owned()),
            reference: None,
        }),
        Analysis::Transient(Transient {
            periodic: true,
            stop: 1e-3,
            ..transient.clone()
        }),
        Analysis::Sensitivity(Sensitivity {
            outputs: vec!["out".to_owned()],
            of: Differentiated::Transient(transient),
        }),
        Analysis::OperatingPoints(OperatingPoints { max_solutions: 2 }),
    ]
}

#[test]
fn every_analysis_about_the_operating_point_solves_it_once_as_afresh() {
    // Before the simulator remembered its operating point, each of these analyses solved
    // it again from zero, eight times in all, and the checks at it three more.
    let circuit = rectifier();
    let options = Options::default();
    let mut simulator = simulator(&circuit);
    for analysis in analyses() {
        let outcome = simulator
            .run(&analysis, &options)
            .expect("the analysis runs");
        let fresh = afresh(&circuit, &analysis, &options);
        // A transient from the operating point remembered skips the iterations that solve
        // it, and takes the same steps.
        if let (Some(remembered), Some(solved)) = (iterations(&outcome), iterations(&fresh)) {
            assert!(
                remembered < solved,
                "{remembered} of {solved}: {analysis:?}"
            );
        }
        assert_eq!(
            apart_from_work(outcome),
            apart_from_work(fresh),
            "{analysis:?}"
        );
    }
    let (point, _) = simulator
        .gmin_dependence(&options)
        .expect("an operating point");
    simulator
        .growing_modes(&options)
        .expect("the poles are found");
    assert_eq!(simulator.operating_point_solves(), 1);
    assert_eq!(
        AnalysisResult::OperatingPoint(point),
        afresh(&circuit, &Analysis::OperatingPoint, &options)
    );
}

#[test]
fn options_that_change_the_dc_solution_solve_it_again() {
    let circuit = rectifier();
    let mut simulator = simulator(&circuit);
    let point = |simulator: &mut Simulator, options: &Options| {
        AnalysisResult::OperatingPoint(simulator.operating_point(options).expect("a solution"))
    };
    point(&mut simulator, &Options::default());
    // A transient's options and the memory leave the DC solution as it is.
    let transient = Options {
        max_transient_iterations: 50,
        truncation_tolerance: 1.0,
        charge_tolerance: 1e-18,
        max_memory: 4.0,
        ..Options::default()
    };
    assert_eq!(
        point(&mut simulator, &transient),
        afresh(&circuit, &Analysis::OperatingPoint, &transient)
    );
    assert_eq!(simulator.operating_point_solves(), 1);
    // So does what the results record, which is read of the devices there.
    let recording = Options {
        record: Record::All,
        ..Options::default()
    };
    assert_eq!(
        point(&mut simulator, &recording),
        afresh(&circuit, &Analysis::OperatingPoint, &recording)
    );
    assert_eq!(simulator.operating_point_solves(), 1);
    // Gmin is in the equations.
    let gmin = Options {
        gmin: 1e-6,
        ..Options::default()
    };
    assert_eq!(
        point(&mut simulator, &gmin),
        afresh(&circuit, &Analysis::OperatingPoint, &gmin)
    );
    assert_eq!(simulator.operating_point_solves(), 2);
    // The simulator remembers the last one only.
    point(&mut simulator, &Options::default());
    assert_eq!(simulator.operating_point_solves(), 3);
}

#[test]
fn tighter_tolerances_go_on_from_the_last_solution_to_the_one_from_zero() {
    // Newton iteration from zero takes the same iterates under any tolerances, so going
    // on from where looser ones stopped reaches the very solution tighter ones reach from
    // zero; looser ones after tighter accept a solution within their tolerance of it.
    let circuit = rectifier();
    let mut simulator = simulator(&circuit);
    let tight = Options {
        relative_tolerance: 1e-9,
        voltage_tolerance: 1e-12,
        ..Options::default()
    };
    simulator
        .operating_point(&Options::default())
        .expect("a solution");
    let continued = simulator.operating_point(&tight).expect("a solution");
    assert_eq!(simulator.operating_point_solves(), 2);
    assert_eq!(
        AnalysisResult::OperatingPoint(continued.clone()),
        afresh(&circuit, &Analysis::OperatingPoint, &tight)
    );
    let loose = Options {
        relative_tolerance: 1e-2,
        ..Options::default()
    };
    let AnalysisResult::OperatingPoint(from_zero) =
        afresh(&circuit, &Analysis::OperatingPoint, &loose)
    else {
        panic!("an operating point");
    };
    let loosened = simulator.operating_point(&loose).expect("a solution");
    for ((loosened, from_zero), tight) in loosened
        .values
        .iter()
        .zip(&from_zero.values)
        .zip(&continued.values)
    {
        assert!(
            (loosened - from_zero).abs() <= 1e-2 * from_zero.abs() + 1e-6,
            "{loosened} against {from_zero}"
        );
        assert!((loosened - tight).abs() <= 1e-9 * tight.abs() + 1e-12);
    }
}

#[test]
fn a_dc_sweep_leaves_the_operating_point_to_the_analyses_after_it() {
    let circuit = rectifier();
    let options = Options::default();
    let mut simulator = simulator(&circuit);
    let ac = Analysis::Ac(Ac {
        source: "V1".to_owned(),
        frequencies: vec![1e3],
    });
    simulator.run(&ac, &options).expect("a response");
    // From the source's own level at time zero, 1 V, the sweep starts at the operating
    // point's solution.
    let sweep = |levels: Vec<f64>| {
        Analysis::DcSweep(DcSweep {
            source: "V1".to_owned(),
            parameter: None,
            values: levels,
        })
    };
    let own = sweep(vec![1.0, 1.5, 2.0]);
    let swept = simulator.run(&own, &options).expect("a sweep");
    assert_eq!(swept, afresh(&circuit, &own, &options));
    assert_eq!(simulator.operating_point_solves(), 1);
    // From another level, it solves its first, and the source let go, the response is
    // the operating point's again.
    let other = sweep(vec![0.5, 1.5]);
    let swept = simulator.run(&other, &options).expect("a sweep");
    assert_eq!(swept, afresh(&circuit, &other, &options));
    assert_eq!(simulator.operating_point_solves(), 2);
    let response = simulator.run(&ac, &options).expect("a response");
    assert_eq!(response, afresh(&circuit, &ac, &options));
    assert_eq!(simulator.operating_point_solves(), 2);
}

/// What an analysis records is read of the devices at each solution, and changes nothing
/// else: a plan of analyses finds the same numbers, every unknown's and every one it finds
/// of them, its work counts among them, recording nothing, every device, or every device
/// in one of its analyses alone. Before, an analysis that recorded other than the one
/// before it solved the operating point again rather than take it: the rectifier's
/// transient after a response that recorded counted the iterations of solving it, and the
/// buck converter's second transient stepped from the pivots solving it left the sparse
/// solver, not from those the first transient left, and found other numbers.
#[test]
fn what_an_analysis_records_changes_none_of_its_numbers() {
    let periods = Analysis::Transient(Transient {
        stop: 2e-5,
        max_step: None,
        method: Method::Trapezoidal,
        breakpoints: Vec::new(),
        periodic: false,
    });
    let plans = [
        (rectifier(), analyses()),
        (
            buck(),
            vec![
                Analysis::OperatingPoint,
                periods.clone(),
                Analysis::OperatingPoint,
                periods,
            ],
        ),
    ];
    for (circuit, analyses) in plans {
        let found = |records: &[Record]| -> Vec<String> {
            let mut simulator = simulator(&circuit);
            analyses
                .iter()
                .zip(records)
                .map(|(analysis, record)| {
                    let options = Options {
                        record: record.clone(),
                        ..Options::default()
                    };
                    let outcome = simulator.run(analysis, &options);
                    // Debug writes each number in the shortest form that reads back as
                    // its bits.
                    format!("{:?}", outcome.map(unrecorded))
                })
                .collect()
        };
        let nothing = vec![Record::default(); analyses.len()];
        let unrecorded = found(&nothing);
        let mut ways = vec![vec![Record::All; analyses.len()]];
        for one in 0..analyses.len() {
            let mut way = nothing.clone();
            way[one] = Record::All;
            ways.push(way);
        }
        for way in ways {
            let recorded = found(&way);
            for (place, (recorded, unrecorded)) in recorded.iter().zip(&unrecorded).enumerate() {
                assert!(
                    recorded == unrecorded,
                    "{}: analysis {place} of {:?} differs, recording {way:?}",
                    circuit.top,
                    analyses[place]
                );
            }
        }
    }
}

#[test]
fn an_analysis_after_the_operating_point_again_factors_as_after_it_first() {
    // The buck's transient leaves its sparse solver other pivots than the operating
    // point's, which the second transient, from the operating point taken again, factored
    // from; it found other numbers in their last digits than the first.
    let periods = Analysis::Transient(Transient {
        stop: 2e-5,
        max_step: None,
        method: Method::Trapezoidal,
        breakpoints: Vec::new(),
        periodic: false,
    });
    let circuit = buck();
    let options = Options::default();
    let mut simulator = simulator(&circuit);
    let plan = [
        Analysis::OperatingPoint,
        periods.clone(),
        Analysis::OperatingPoint,
        periods,
    ];
    // Debug writes each number in the shortest form that reads back as its bits.
    let found: Vec<String> = plan
        .iter()
        .map(|analysis| format!("{:?}", simulator.run(analysis, &options)))
        .collect();
    assert_eq!(simulator.operating_point_solves(), 1);
    assert!(found[2] == found[0], "the operating point taken again");
    assert!(
        found[3] == found[1],
        "the transient after it, its work among it"
    );
}
