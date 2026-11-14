//! The same circuits simulate the same however many threads run them: a run's small-signal
//! analyses and sensitivities find the same numbers however their frequencies and
//! right-hand sides spread over 1, 2, 3 and 8 threads. Every comparison is of every
//! number's bits, with `f64::to_bits`.
#![cfg(test)]

use std::fs;
use std::path::Path;
use std::sync::Arc;

use jolty_engine::{
    Analysis, Derivatives, Elaborated, Options, Parallelism, SimulationError, Simulator, Structure,
    Values,
};
use jolty_ir::Circuit;
use jolty_models::Library;
use jolty_numerics::Complex64;
use proptest::prelude::*;
use rayon::ThreadPoolBuilder;
use serde::Deserialize;
use serde_json::{Value, json};

/// The thread counts every comparison runs at.
const THREADS: [usize; 4] = [1, 2, 3, 8];

fn reals(values: &[f64], into: &mut Vec<u64>) {
    into.extend(values.iter().map(|value| value.to_bits()));
}

fn complexes(values: &[Complex64], into: &mut Vec<u64>) {
    for value in values {
        into.extend([value.re.to_bits(), value.im.to_bits()]);
    }
}

/// Every number of an analysis's result, as its bits, and the result as text, which holds
/// its counts of work and its messages.
fn bits(outcome: &Result<jolty_engine::Outcome, SimulationError>) -> Bits {
    use jolty_engine::Outcome as Found;
    let mut numbers = Vec::new();
    match outcome {
        Ok(Found::OperatingPoint(point)) => reals(&point.values, &mut numbers),
        Ok(Found::Ac(response)) => {
            reals(&response.operating_point.values, &mut numbers);
            for row in response.values.iter().chain(&response.currents) {
                complexes(row, &mut numbers);
            }
        }
        Ok(Found::Noise(spectrum)) => {
            reals(&spectrum.densities, &mut numbers);
            for contribution in &spectrum.contributions {
                reals(&contribution.densities, &mut numbers);
            }
            if let Some(input) = &spectrum.input {
                complexes(&input.gains, &mut numbers);
            }
        }
        Ok(Found::Stability(gain)) => complexes(&gain.gains, &mut numbers),
        Ok(Found::Transient(waveforms)) => {
            reals(&waveforms.times, &mut numbers);
            for row in &waveforms.values {
                reals(row, &mut numbers);
            }
            reals(&waveforms.observations.currents, &mut numbers);
        }
        Ok(Found::DcSweep(sweep)) => {
            for row in &sweep.values {
                reals(row, &mut numbers);
            }
            reals(&sweep.observations.currents, &mut numbers);
        }
        Ok(Found::Sensitivity(found)) => match &found.derivatives {
            Derivatives::Real(points) => {
                for row in points.iter().flatten() {
                    reals(row, &mut numbers);
                }
            }
            Derivatives::Complex(points) => {
                for row in points.iter().flatten() {
                    complexes(row, &mut numbers);
                }
            }
        },
        Ok(_) => panic!("no other analysis is compared here"),
        Err(_) => {}
    }
    (numbers, format!("{outcome:?}"))
}

/// An analysis's result as bits, and as text.
type Bits = (Vec<u64>, String);

/// A grounded network of resistors, capacitors, inductors and diodes, driven by a pulse
/// and by current sources.
#[derive(Debug, Clone)]
struct Network {
    resistors: Vec<(usize, usize, f64)>,
    capacitors: Vec<(usize, f64)>,
    inductors: Vec<(usize, usize, f64)>,
    diodes: Vec<usize>,
    sources: Vec<(usize, f64)>,
}

fn node(index: usize) -> String {
    if index == 0 {
        "0".to_owned()
    } else {
        format!("n{index}")
    }
}

/// A connected network on up to six nodes: a spanning tree of resistors from 0.1 Ω to
/// 100 kΩ, so that some carry their current as an unknown, extra resistors, a capacitor
/// at every node, inductors, diodes to the ground, the pulse at node 1, and current
/// sources.
fn network() -> impl Strategy<Value = Network> {
    (2usize..7).prop_flat_map(|nodes| {
        let tree =
            proptest::collection::vec((any::<prop::sample::Index>(), -1.0..5.0f64), nodes - 1);
        let extra = proptest::collection::vec((0..nodes, 0..nodes, -1.0..5.0f64), 0..nodes);
        let capacitors = proptest::collection::vec(-12.0..-9.0f64, nodes - 1);
        let inductors = proptest::collection::vec((0..nodes, 0..nodes, -7.0..-3.0f64), 0..3);
        let diodes = proptest::collection::vec(1..nodes, 0..3);
        let sources = proptest::collection::vec((1..nodes, -1e-3..1e-3f64), 0..3);
        (tree, extra, capacitors, inductors, diodes, sources).prop_map(
            |(tree, extra, capacitors, inductors, diodes, sources)| {
                let mut resistors: Vec<(usize, usize, f64)> = tree
                    .into_iter()
                    .enumerate()
                    .map(|(child, (parent, exponent))| {
                        (child + 1, parent.index(child + 1), 10f64.powf(exponent))
                    })
                    .collect();
                resistors.extend(
                    extra
                        .into_iter()
                        .filter(|(a, b, _)| a != b)
                        .map(|(a, b, exponent)| (a, b, 10f64.powf(exponent))),
                );
                Network {
                    resistors,
                    capacitors: capacitors
                        .into_iter()
                        .enumerate()
                        .map(|(index, exponent)| (index + 1, 10f64.powf(exponent)))
                        .collect(),
                    inductors: inductors
                        .into_iter()
                        .filter(|(a, b, _)| a != b)
                        .map(|(a, b, exponent)| (a, b, 10f64.powf(exponent)))
                        .collect(),
                    diodes,
                    sources,
                }
            },
        )
    })
}

impl Network {
    fn circuit(&self) -> Circuit {
        let device = |id: String, model: &str, parameters: Value, terminals: Value| {
            json!({"id": id, "of": {"device": model}, "parameters": parameters,
                   "terminals": terminals})
        };
        let mut instances = vec![
            device(
                "V1".to_owned(),
                "voltage_source",
                json!({"waveform": {"pulse": {"initial": 0.0, "pulsed": 1.0, "delay": 1e-9,
                    "rise": 1e-9, "fall": 1e-9, "width": 5e-9, "period": 20e-9}}}),
                json!({"positive": "drive", "negative": "0"}),
            ),
            device(
                "RD".to_owned(),
                "resistor",
                json!({"resistance": 50.0}),
                json!({"a": "drive", "b": node(1)}),
            ),
        ];
        for (index, &(a, b, resistance)) in self.resistors.iter().enumerate() {
            instances.push(device(
                format!("R{index}"),
                "resistor",
                json!({"resistance": resistance}),
                json!({"a": node(a), "b": node(b)}),
            ));
        }
        for &(at, capacitance) in &self.capacitors {
            instances.push(device(
                format!("C{at}"),
                "capacitor",
                json!({"capacitance": capacitance}),
                json!({"a": node(at), "b": "0"}),
            ));
        }
        for (index, &(a, b, inductance)) in self.inductors.iter().enumerate() {
            instances.push(device(
                format!("L{index}"),
                "inductor",
                json!({"inductance": inductance}),
                json!({"a": node(a), "b": node(b)}),
            ));
        }
        for (index, &at) in self.diodes.iter().enumerate() {
            instances.push(device(
                format!("D{index}"),
                "diode",
                json!({"saturation_current": 1e-14, "junction_capacitance": 1e-12}),
                json!({"anode": node(at), "cathode": "0"}),
            ));
        }
        for (index, &(at, current)) in self.sources.iter().enumerate() {
            instances.push(device(
                format!("I{index}"),
                "current_source",
                json!({"waveform": {"dc": current}}),
                json!({"positive": "0", "negative": node(at)}),
            ));
        }
        let document = json!({
            "format": "jolty-circuit", "version": 1, "top": "network",
            "components": {"network": {
                "ports": [{"name": "0", "role": "ground"}], "instances": instances,
            }}
        });
        Circuit::from_json(&document.to_string()).expect("a valid circuit")
    }
}

fn shared(circuit: &Circuit) -> Arc<Elaborated> {
    Elaborated::new(circuit, &Library::standard(), &Structure::default())
        .expect("the circuit elaborates")
}

/// A plan of the engine's benchmarks, `perf/plans/<name>.json`.
#[derive(Deserialize)]
struct Plan {
    circuit: String,
    analyses: Vec<(String, Analysis)>,
    options: Options,
}

/// The circuit, the analyses and the options of the benchmark `name`.
fn benchmark(name: &str) -> (Circuit, Vec<Analysis>, Options) {
    let perf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../perf");
    let plan: Plan = serde_json::from_str(
        &fs::read_to_string(perf.join(format!("plans/{name}.json"))).expect("the plan reads"),
    )
    .expect("a plan");
    let text = fs::read_to_string(perf.join(format!("circuits/{}.json", plan.circuit)))
        .expect("the circuit reads");
    let circuit = Circuit::from_json(&text).expect("a circuit");
    let analyses = plan
        .analyses
        .into_iter()
        .map(|(_, analysis)| analysis)
        .collect();
    (circuit, analyses, plan.options)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// A run's small-signal analyses and sensitivities find the same numbers whatever
    /// threads their frequencies and right-hand sides spread over.
    #[test]
    fn small_signal_analyses_find_the_same_however_their_frequencies_spread(
        network in network(),
    ) {
        let circuit = network.circuit();
        let frequencies: Vec<f64> =
            (0..64).map(|step| 10f64.powf(f64::from(step) / 6.0)).collect();
        let analyses: Vec<Analysis> = [
            json!({"ac": {"source": "V1", "frequencies": frequencies}}),
            json!({"noise": {"output": "n1", "source": "V1", "frequencies": frequencies}}),
            json!({"sensitivity": {"outputs": ["n1", "drive"]}}),
            json!({"sensitivity": {"outputs": ["n1"],
                   "of": {"ac": {"source": "V1", "frequencies": frequencies}}}}),
        ]
        .into_iter()
        .map(|analysis| serde_json::from_value(analysis).expect("an analysis"))
        .collect();
        let found = |threads: usize| {
            let pool = ThreadPoolBuilder::new().num_threads(threads).build().expect("a pool");
            pool.install(|| {
                let mut simulator = Simulator::from_elaborated(shared(&circuit), Values::default())
                    .expect("a run");
                analyses
                    .iter()
                    .map(|analysis| bits(&simulator.run(analysis, &Options::default())))
                    .collect::<Vec<_>>()
            })
        };
        let alone = found(1);
        for threads in THREADS {
            prop_assert_eq!(&found(threads), &alone, "at {} threads", threads);
        }
    }
}

/// Frequencies where the pivots of `G + jωC` change and change back, factored in any
/// spread of tasks, find what one solver going through them in order finds: inductors,
/// whose branch equations' own entries, `−jωL`, are below a tenth of the unit entries
/// beside them at low frequencies and above them at high ones, so that elimination takes
/// another pivot at each end, swept up and down.
#[test]
fn frequencies_whose_pivots_change_find_the_same_in_any_spread() {
    let document = json!({
        "format": "jolty-circuit", "version": 1, "top": "ladder",
        "components": {"ladder": {"ports": [{"name": "0", "role": "ground"}], "instances": [
            {"id": "V1", "of": {"device": "voltage_source"}, "parameters": {"waveform": {"dc": 0.0}},
             "terminals": {"positive": "in", "negative": "0"}},
            {"id": "R1", "of": {"device": "resistor"}, "parameters": {"resistance": 50.0},
             "terminals": {"a": "in", "b": "a"}},
            {"id": "L1", "of": {"device": "inductor"}, "parameters": {"inductance": 1e-3},
             "terminals": {"a": "a", "b": "b"}},
            {"id": "C1", "of": {"device": "capacitor"}, "parameters": {"capacitance": 1e-9},
             "terminals": {"a": "b", "b": "0"}},
            {"id": "L2", "of": {"device": "inductor"}, "parameters": {"inductance": 1e-6},
             "terminals": {"a": "b", "b": "c"}},
            {"id": "C2", "of": {"device": "capacitor"}, "parameters": {"capacitance": 1e-12},
             "terminals": {"a": "c", "b": "0"}},
            {"id": "L3", "of": {"device": "inductor"}, "parameters": {"inductance": 1e-5},
             "terminals": {"a": "c", "b": "0"}},
            {"id": "R2", "of": {"device": "resistor"}, "parameters": {"resistance": 1e3},
             "terminals": {"a": "c", "b": "0"}}
        ]}}
    });
    let circuit = Circuit::from_json(&document.to_string()).expect("a valid circuit");
    let up: Vec<f64> = (0..120)
        .map(|step| 10f64.powf(f64::from(step) / 12.0))
        .collect();
    let frequencies: Vec<f64> = up.iter().chain(up.iter().rev()).copied().collect();
    let analyses: Vec<Analysis> = [
        json!({"ac": {"source": "V1", "frequencies": frequencies}}),
        json!({"noise": {"output": "c", "source": "V1", "frequencies": frequencies}}),
    ]
    .into_iter()
    .map(|analysis| serde_json::from_value(analysis).expect("an analysis"))
    .collect();
    let found = |threads: usize| {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a pool");
        pool.install(|| {
            let mut simulator =
                Simulator::from_elaborated(shared(&circuit), Values::default()).expect("a run");
            analyses
                .iter()
                .map(|analysis| bits(&simulator.run(analysis, &Options::default())))
                .collect::<Vec<_>>()
        })
    };
    let alone = found(1);
    assert!(
        alone.iter().all(|(numbers, _)| !numbers.is_empty()),
        "{alone:?}"
    );
    for threads in [2, 3, 5, 8, 12] {
        assert_eq!(found(threads), alone, "at {threads} threads");
    }
}

/// A run's own work stays on one thread when its options say so, and finds the same.
#[test]
fn a_run_on_one_thread_finds_what_it_finds_on_many() {
    let (circuit, analyses, options) = benchmark("ac_wide");
    let serial = Options {
        parallelism: Parallelism {
            threads: Some(1),
            ..Parallelism::default()
        },
        ..options.clone()
    };
    let run = |options: &Options| {
        let pool = ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .expect("a pool");
        pool.install(|| {
            let mut simulator =
                Simulator::from_elaborated(shared(&circuit), Values::default()).expect("a run");
            analyses
                .iter()
                .map(|analysis| bits(&simulator.run(analysis, options)))
                .collect::<Vec<_>>()
        })
    };
    assert_eq!(run(&serial), run(&options));
}
