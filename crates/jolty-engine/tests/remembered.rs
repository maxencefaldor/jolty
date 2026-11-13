//! The operating point a simulator remembers, solved once for every analysis about it, and
//! the same as solved afresh.

use jolty_engine::{
    Ac, Analysis, Conditions, DcSweep, Differentiated, Method, Noise, OperatingPoints, Options,
    Outcome, PoleZero, Sensitivity, Simulator, Transient,
};
use jolty_ir::Circuit;
use jolty_models::Library;
use serde_json::{Value, json};

/// A circuit of `instances` whose ground is `0`.
fn circuit(instances: &Value) -> Circuit {
    let document = json!({
        "format": "jolty-circuit", "version": 1, "top": "test",
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
        {"id": "V1", "of": {"device": "voltage_source"},
         "parameters": {"waveform": {"sine": {"offset": 1.0, "amplitude": 1.0,
                                              "frequency": 1e3}}},
         "terminals": {"positive": "in", "negative": "0"}},
        {"id": "R1", "of": {"device": "resistor"}, "parameters": {"resistance": 1e3},
         "terminals": {"a": "in", "b": "out"}},
        {"id": "D1", "of": {"device": "diode"}, "parameters": {},
         "terminals": {"anode": "out", "cathode": "0"}},
        {"id": "C1", "of": {"device": "capacitor"}, "parameters": {"capacitance": 1e-6},
         "terminals": {"a": "out", "b": "0"}},
    ]))
}

fn simulator(circuit: &Circuit) -> Simulator {
    Simulator::new(circuit, &Library::standard(), &Conditions::default())
        .expect("the circuit elaborates")
}

/// `analysis` run with `options` on a simulator of its own.
fn afresh(circuit: &Circuit, analysis: &Analysis, options: &Options) -> Outcome {
    simulator(circuit)
        .run(analysis, options)
        .expect("the analysis runs")
}

fn analyses() -> Vec<Analysis> {
    let frequencies = vec![1.0, 1e3, 1e6];
    let transient = Transient {
        stop: 2e-3,
        max_step: Some(2e-5),
        method: Method::Trapezoidal,
        times: Vec::new(),
        periodic: false,
    };
    vec![
        Analysis::OperatingPoint,
        Analysis::Ac(Ac {
            source: "V1".to_owned(),
            frequencies: frequencies.clone(),
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
        Analysis::OperatingPoints(OperatingPoints { most: 2 }),
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
        assert_eq!(
            outcome,
            afresh(&circuit, &analysis, &options),
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
        Outcome::OperatingPoint(point),
        afresh(&circuit, &Analysis::OperatingPoint, &options)
    );
}

#[test]
fn options_that_change_the_dc_solution_solve_it_again() {
    let circuit = rectifier();
    let mut simulator = simulator(&circuit);
    let point = |simulator: &mut Simulator, options: &Options| {
        Outcome::OperatingPoint(simulator.operating_point(options).expect("a solution"))
    };
    point(&mut simulator, &Options::default());
    // A transient's options and the memory leave the DC solution as it is.
    let transient = Options {
        transient_iterations: 50,
        truncation_tolerance: 1.0,
        charge_tolerance: 1e-18,
        transient_memory: 4.0,
        ..Options::default()
    };
    assert_eq!(
        point(&mut simulator, &transient),
        afresh(&circuit, &Analysis::OperatingPoint, &transient)
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
        Outcome::OperatingPoint(continued.clone()),
        afresh(&circuit, &Analysis::OperatingPoint, &tight)
    );
    let loose = Options {
        relative_tolerance: 1e-2,
        ..Options::default()
    };
    let Outcome::OperatingPoint(from_zero) = afresh(&circuit, &Analysis::OperatingPoint, &loose)
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
            levels,
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
