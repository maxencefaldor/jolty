//! The same circuits simulate the same however many threads or shards run them: a batch at
//! 1, 2, 3 and 8 threads, and split into ranges, hands on the same outcomes and summary,
//! and a run's small-signal analyses and sensitivities find the same numbers however
//! their frequencies and right-hand sides spread. Every comparison is of every number's
//! bits, with `f64::to_bits`.
#![cfg(test)]

use std::collections::BTreeMap;
use std::fs;
use std::ops::Range;
use std::path::Path;
use std::sync::{Arc, Mutex};

use jolty_engine::batch::{Batch, Measured, Outcome, Sink, Summary, Variants};
use jolty_engine::measure::{Measure, Signal};
use jolty_engine::variants::{Axis, Explicit, Grid, MonteCarlo, Product, Toleranced, toleranced};
use jolty_engine::{
    Analysis, Derivatives, Elaborated, Options, Parallelism, Record, SimulationError, Simulator,
    Structure, Values,
};
use jolty_ir::{Circuit, Distribution, Lot, Name, Tolerance};
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

/// A run's outcome as bits: each analysis's, each measure's, and the problems' text.
type OutcomeBits = (Vec<Bits>, Vec<Option<u64>>, String);

/// An outcome of a batch's run as bits: each analysis's, each measure's, and the
/// problems' text.
fn outcome_bits(outcome: &Outcome) -> OutcomeBits {
    (
        outcome.results.iter().map(bits).collect(),
        outcome
            .measured
            .iter()
            .map(|measured| measured.as_ref().ok().map(|value| value.to_bits()))
            .collect(),
        format!("{:?}", outcome.problems),
    )
}

/// A sink that keeps every outcome by its index, and fails a test that hands one on twice.
#[derive(Default)]
struct Kept(Mutex<BTreeMap<u64, Outcome>>);

impl Sink for Kept {
    fn accept(&self, index: u64, outcome: Outcome) {
        let mut kept = self.0.lock().expect("no test thread panicked");
        assert!(
            kept.insert(index, outcome).is_none(),
            "run {index} came twice"
        );
    }
}

/// What a batch found over some runs: each run's outcome as bits, by index, and the
/// summary.
type Found = (
    BTreeMap<u64, (Vec<(Vec<u64>, String)>, Vec<Option<u64>>, String)>,
    Summary,
);

/// The runs `ranges` of `variants`, each range a call of its own, on a pool of `threads`
/// threads, and their summaries merged in order.
fn run_on(batch: &Batch, variants: &dyn Variants, ranges: &[Range<u64>], threads: usize) -> Found {
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("a pool");
    let kept = Kept::default();
    let mut summary = None;
    for range in ranges {
        let found = pool
            .install(|| batch.run(variants, range.clone(), &kept))
            .expect("the range lies within the variants");
        summary = Some(match summary {
            None => found,
            Some(before) => Summary::merge(before, found),
        });
    }
    let outcomes = kept
        .0
        .into_inner()
        .expect("no test thread panicked")
        .iter()
        .map(|(&index, outcome)| (index, outcome_bits(outcome)))
        .collect();
    (outcomes, summary.expect("a range or more"))
}

/// The one range `range`, as the ranges of a run.
fn only(range: Range<u64>) -> [Range<u64>; 1] {
    [range]
}

/// Checks that `variants` find the same outcomes and the same summary at every thread
/// count of [`THREADS`], and split into `splits`, each a list of ranges that cover them.
fn assert_same_everywhere(batch: &Batch, variants: &dyn Variants, splits: &[Vec<Range<u64>>]) {
    let whole = only(0..variants.len());
    let reference = run_on(batch, variants, &whole, 1);
    assert_eq!(reference.0.len() as u64, variants.len());
    for threads in THREADS {
        assert!(
            run_on(batch, variants, &whole, threads) == reference,
            "at {threads} threads"
        );
        for split in splits {
            assert!(
                run_on(batch, variants, split, threads) == reference,
                "split into {split:?} at {threads} threads"
            );
        }
    }
}

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

    /// Each resistor's resistance, toleranced by 50%, so that samples cross 1 Ω.
    fn toleranced(&self) -> Vec<Toleranced> {
        self.resistors
            .iter()
            .enumerate()
            .map(|(index, &(_, _, resistance))| Toleranced {
                path: format!("R{index}"),
                name: "resistance".to_owned(),
                nominal: resistance,
                tolerance: Tolerance {
                    relative: 0.5,
                    distribution: Distribution::Uniform,
                    lot: None,
                },
            })
            .collect()
    }
}

/// The analyses a batch of networks runs: the operating point, 40 frequencies of the
/// pulse's response, the noise at node 1 referred to the pulse, and a transient.
fn network_analyses() -> Vec<Analysis> {
    let frequencies: Vec<f64> = (0..40)
        .map(|step| 10f64.powf(f64::from(step) / 4.0))
        .collect();
    [
        json!("operating_point"),
        json!({"ac": {"source": "V1", "frequencies": frequencies}}),
        json!({"noise": {"output": "n1", "source": "V1", "frequencies": frequencies}}),
        json!({"transient": {"stop": 4e-8}}),
    ]
    .into_iter()
    .map(|analysis| serde_json::from_value(analysis).expect("an analysis"))
    .collect()
}

/// A measure of node 1 at the operating point.
fn node_one(elaborated: &Elaborated) -> Vec<Measured> {
    let unknown = elaborated
        .unknowns()
        .iter()
        .position(|unknown| unknown.name == "n1")
        .expect("node 1");
    vec![Measured {
        analysis: 0,
        measure: Measure::Value {
            signal: Signal {
                unknown,
                reference: None,
            },
            at: None,
        },
    }]
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

    /// A Monte Carlo batch of a random network, whose samples cross 1 Ω and so change its
    /// structure, finds the same at every thread count, and split into ranges.
    #[test]
    fn a_batch_finds_the_same_however_its_runs_spread(
        network in network(),
        seed in any::<u64>(),
    ) {
        let circuit = network.circuit();
        let elaborated = shared(&circuit);
        let samples = 24;
        let variants = MonteCarlo::new(&elaborated, seed, samples, network.toleranced())
            .expect("the network's resistances");
        let options = Options { record: Record::All, ..Options::default() };
        let measures = node_one(&elaborated);
        let batch = Batch::new(elaborated, network_analyses(), options, measures)
            .expect("a batch");
        assert_same_everywhere(&batch, &variants, &[vec![0..7, 7..8, 8..24], vec![0..13, 13..24]]);
    }

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

/// The amplifier's resistors, each within 5%, the benchmark `monte_carlo`'s tolerances.
fn amplifier_samples(elaborated: &Elaborated, seed: u64, samples: u64) -> MonteCarlo {
    let (circuit, _, _) = benchmark("bjt_amp");
    let varied = toleranced(&circuit).expect("numbers");
    assert!(
        !varied.is_empty(),
        "the amplifier's resistors are toleranced"
    );
    MonteCarlo::new(elaborated, seed, samples, varied).expect("the amplifier's numbers")
}

/// The benchmark `bjt_amp`'s analyses on samples of its resistors find the same at every
/// thread count and split; each sample drawn alone, as a batch of one run, finds what it
/// finds among the others.
#[test]
fn amplifier_samples_find_the_same_alone_and_in_a_batch() {
    let (circuit, analyses, options) = benchmark("bjt_amp");
    let elaborated = shared(&circuit);
    let variants = amplifier_samples(&elaborated, 7, 16);
    let batch = Batch::new(elaborated, analyses, options, Vec::new()).expect("a batch");
    assert_same_everywhere(&batch, &variants, &[vec![0..5, 5..16]]);
    let (all, _) = run_on(&batch, &variants, &only(0..16), 8);
    for index in [0, 9, 15] {
        let (alone, summary) = run_on(&batch, &variants, &only(index..index + 1), 1);
        assert_eq!(summary.runs, 1);
        assert!(alone[&index] == all[&index], "sample {index}");
    }
}

/// A sample's draws depend on its seed, its index and each part's name alone: a part added
/// leaves every other part's draws as they were, a lot draws one factor for every part in
/// it, and a seed draws the same sample every time.
#[test]
fn a_part_added_leaves_every_other_draw_alone() {
    let part = |path: &str, lot: Option<&str>| Toleranced {
        path: path.to_owned(),
        name: "resistance".to_owned(),
        nominal: 1e3,
        tolerance: Tolerance {
            relative: 0.01,
            distribution: Distribution::Normal,
            lot: lot.map(|name| Lot {
                name: Name::new(name).expect("a name"),
                relative: 0.2,
                distribution: Distribution::LogUniform,
            }),
        },
    };
    let (first, second) = (part("X1/R1", None), part("X2/R1", None));
    for index in 0..100 {
        let alone = first.drawn(3, index);
        assert_eq!(alone.to_bits(), first.drawn(3, index).to_bits());
        assert_ne!(alone.to_bits(), second.drawn(3, index).to_bits());
        assert_ne!(alone.to_bits(), first.drawn(4, index).to_bits());
    }
    // Parts in one lot share its factor, so their ratio is their own factors'.
    let (a, b) = (part("R1", Some("die")), part("R2", Some("die")));
    let (own_a, own_b) = (part("R1", None), part("R2", None));
    for index in 0..100 {
        let shared = a.drawn(5, index) / own_a.drawn(5, index);
        let other = b.drawn(5, index) / own_b.drawn(5, index);
        assert!((shared / other - 1.0).abs() < 1e-14, "sample {index}");
    }
    // A network's samples, with a part added to it, draw the same for every other part.
    let network = Network {
        resistors: vec![(1, 0, 1e3), (2, 1, 2e3)],
        capacitors: vec![(1, 1e-12), (2, 1e-12)],
        inductors: Vec::new(),
        diodes: Vec::new(),
        sources: Vec::new(),
    };
    let mut larger = network.clone();
    larger.resistors.push((2, 0, 5e3));
    let (small, large) = (shared(&network.circuit()), shared(&larger.circuit()));
    let fewer = MonteCarlo::new(&small, 11, 50, network.toleranced()).expect("numbers");
    let more = MonteCarlo::new(&large, 11, 50, larger.toleranced()).expect("numbers");
    for index in 0..50 {
        let (fewer, more) = (fewer.values(index), more.values(index));
        assert_eq!(fewer.patches().len() + 1, more.patches().len());
        for (path, name) in [("R0", "resistance"), ("R1", "resistance")] {
            let value = |elaborated: &Elaborated, values: &Values| {
                values
                    .get(elaborated.parameter(path, name).expect("a resistance"))
                    .expect("drawn")
            };
            assert_eq!(
                value(&small, &fewer).to_bits(),
                value(&large, &more).to_bits()
            );
        }
    }
}

/// A divider whose upper resistance each run of `upper` replaces.
fn divider() -> Circuit {
    let document = json!({
        "format": "jolty-circuit", "version": 1, "top": "divider",
        "components": {"divider": {"ports": [{"name": "0", "role": "ground"}], "instances": [
            {"id": "V1", "of": {"device": "voltage_source"}, "parameters": {"waveform": {"dc": 5.0}},
             "terminals": {"positive": "top", "negative": "0"}},
            {"id": "R1", "of": {"device": "resistor"}, "parameters": {"resistance": 1e3},
             "terminals": {"a": "top", "b": "out"}},
            {"id": "R2", "of": {"device": "resistor"}, "parameters": {"resistance": 1e3},
             "terminals": {"a": "out", "b": "0"}},
            {"id": "C1", "of": {"device": "capacitor"}, "parameters": {"capacitance": 1e-9},
             "terminals": {"a": "out", "b": "0"}}
        ]}}
    });
    Circuit::from_json(&document.to_string()).expect("a valid circuit")
}

/// The values of `elaborated`, the divider, with its upper resistance `ohms`.
fn upper(elaborated: &Elaborated, ohms: f64) -> Values {
    let mut values = Values::default();
    values.set(
        elaborated
            .parameter("R1", "resistance")
            .expect("a resistance"),
        ohms,
    );
    values
}

/// A batch whose runs fail now and then, their values out of range, reports each failed
/// run, by its index, with why it failed, and runs every other.
#[test]
fn a_failed_run_is_an_outcome_and_the_batch_goes_on() {
    let elaborated = shared(&divider());
    let runs: Vec<Values> = (0..40)
        .map(|index| {
            upper(
                &elaborated,
                if index % 7 == 3 {
                    -1.0
                } else {
                    1e3 + f64::from(index)
                },
            )
        })
        .collect();
    let variants = Explicit(runs);
    let analyses = vec![Analysis::OperatingPoint];
    let measures = vec![Measured {
        analysis: 0,
        measure: Measure::Value {
            signal: Signal {
                unknown: 1,
                reference: None,
            },
            at: None,
        },
    }];
    let batch = Batch::new(
        Arc::clone(&elaborated),
        analyses,
        Options::default(),
        measures,
    )
    .expect("a batch");
    let pool = ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .expect("a pool");
    let kept = Kept::default();
    let summary = pool
        .install(|| batch.run(&variants, 0..40, &kept))
        .expect("the runs lie within the variants");
    let kept = kept.0.into_inner().expect("no test thread panicked");
    assert_eq!(kept.len(), 40);
    let failed: Vec<u64> = kept
        .iter()
        .filter(|(_, outcome)| outcome.failed())
        .map(|(&index, _)| index)
        .collect();
    assert_eq!(failed, [3, 10, 17, 24, 31, 38]);
    assert_eq!((summary.runs, summary.failed), (40, 6));
    assert_eq!(summary.measures[0].measured, 34);
    for index in &failed {
        let message = kept[index].failure().expect("a failure");
        assert!(message.contains("R1"), "{message}");
        assert!(message.contains("resistance"), "{message}");
    }
    // Every other run found its divider's output, alone and in the batch alike.
    for (&index, outcome) in &kept {
        if outcome.failed() {
            continue;
        }
        let mut alone = Simulator::from_elaborated(Arc::clone(&elaborated), variants.values(index))
            .expect("a run");
        let point = alone
            .operating_point(&Options::default())
            .expect("an operating point");
        assert_eq!(
            outcome.measured[0].as_ref().map(|value| value.to_bits()),
            Ok(point.values[1].to_bits())
        );
    }
}

/// A batch whose runs alternate between two structures, the upper resistance either side
/// of 1 Ω, elaborates the other structure once, and finds for each run what a run made for
/// its values alone finds.
#[test]
fn runs_that_alternate_between_structures_share_each_structure() {
    let elaborated = shared(&divider());
    let runs: Vec<Values> = (0..64)
        .map(|index| {
            upper(
                &elaborated,
                if index % 2 == 0 { 0.5 } else { 2.0 } + f64::from(index) * 1e-3,
            )
        })
        .collect();
    let variants = Explicit(runs);
    let analyses: Vec<Analysis> = [
        json!("operating_point"),
        json!({"ac": {"source": "V1", "frequencies": [1e3, 1e6]}}),
        json!({"transient": {"stop": 1e-6}}),
    ]
    .into_iter()
    .map(|analysis| serde_json::from_value(analysis).expect("an analysis"))
    .collect();
    let batch = Batch::new(
        Arc::clone(&elaborated),
        analyses.clone(),
        Options::default(),
        Vec::new(),
    )
    .expect("a batch");
    assert_same_everywhere(&batch, &variants, &[vec![0..31, 31..64]]);
    assert_eq!(
        batch.structures(),
        1,
        "the other structure is elaborated once"
    );
    let (found, summary) = run_on(&batch, &variants, &only(0..64), 8);
    assert_eq!((summary.runs, summary.failed), (64, 0));
    for index in [0, 1, 32, 63] {
        let mut alone = Simulator::from_elaborated(Arc::clone(&elaborated), variants.values(index))
            .expect("a run");
        let alone: Vec<_> = analyses
            .iter()
            .map(|analysis| bits(&alone.run(analysis, &Options::default())))
            .collect();
        assert!(found[&index].0 == alone, "run {index}");
    }
}

/// A warm grid's lines are chains, each point starting from the one before it, and find
/// the same at every thread count and split, a range that starts within a chain among
/// them; a chain of operating points along a source's level is the DC sweep of it.
#[test]
fn a_warm_grid_runs_its_lines_as_chains() {
    let (circuit, _, options) = benchmark("bjt_amp");
    let elaborated = shared(&circuit);
    let level = elaborated
        .parameter("V1", "waveform.sine.offset")
        .or_else(|| elaborated.parameter("V1", "waveform.dc"));
    let resistors: Vec<_> = elaborated
        .devices()
        .iter()
        .filter(|device| device.model() == "resistor")
        .map(|device| device.path().to_owned())
        .collect();
    let first = elaborated
        .parameter(&resistors[0], "resistance")
        .expect("a resistance");
    let nominal = elaborated.nominal()[first.index()];
    let levels: Vec<f64> = (0..9).map(|step| -0.02 + f64::from(step) * 0.005).collect();
    let mut axes = vec![(
        Axis::Parameter(first),
        vec![nominal * 0.9, nominal, nominal * 1.1],
    )];
    if let Some(level) = level {
        axes.push((Axis::Parameter(level), levels));
    } else {
        axes.push((Axis::Temperature, vec![290.0, 300.0, 310.0]));
    }
    let grid = Grid::new(axes, true).expect("a grid");
    assert!(grid.chain() > 1);
    let analyses = vec![Analysis::OperatingPoint];
    let batch =
        Batch::new(Arc::clone(&elaborated), analyses, options, Vec::new()).expect("a batch");
    let len = grid.len();
    assert_same_everywhere(
        &batch,
        &grid,
        &[vec![0..4, 4..13, 13..len], vec![0..len - 1, len - 1..len]],
    );
}

/// A warm grid of a source's DC level is the DC sweep of it, bit for bit: the chain of
/// operating points goes from one level to the next as the sweep does.
#[test]
fn a_chain_of_levels_is_the_dc_sweep_of_them() {
    let elaborated = shared(&divider());
    let level = elaborated.parameter("V1", "waveform.dc").expect("a level");
    let levels: Vec<f64> = (0..21).map(|step| f64::from(step) * 0.25).collect();
    let grid = Grid::new(vec![(Axis::Parameter(level), levels.clone())], true).expect("a grid");
    let batch = Batch::new(
        Arc::clone(&elaborated),
        vec![Analysis::OperatingPoint],
        Options::default(),
        Vec::new(),
    )
    .expect("a batch");
    let (found, _) = run_on(&batch, &grid, &only(0..grid.len()), 4);
    let sweep: Analysis =
        serde_json::from_value(json!({"dc_sweep": {"source": "V1", "levels": levels}}))
            .expect("a sweep");
    let mut simulator =
        Simulator::from_elaborated(Arc::clone(&elaborated), Values::default()).expect("a run");
    let Ok(jolty_engine::Outcome::DcSweep(swept)) = simulator.run(&sweep, &Options::default())
    else {
        panic!("a sweep");
    };
    for (index, values) in swept.values.iter().enumerate() {
        let mut expected = Vec::new();
        reals(values, &mut expected);
        assert_eq!(found[&(index as u64)].0[0].0, expected, "level {index}");
    }
}

/// A product of corners and samples runs every sample at every corner: run `i` is corner
/// `i / samples`, sample `i % samples`.
#[test]
fn a_product_runs_every_sample_at_every_corner() {
    let (circuit, analyses, options) = benchmark("bjt_amp");
    let elaborated = shared(&circuit);
    let corners = Explicit(vec![Values::at(260.0), Values::at(350.0)]);
    let samples = amplifier_samples(&elaborated, 1, 6);
    let product = Product::new(corners, samples.clone()).expect("a product");
    assert_eq!(product.len(), 12);
    let values = product.values(8);
    assert_eq!(values.temperature, Some(350.0));
    assert_eq!(values.patches(), samples.values(2).patches());
    let batch = Batch::new(elaborated, analyses, options, Vec::new()).expect("a batch");
    assert_same_everywhere(&batch, &product, &[vec![0..6, 6..12]]);
}
