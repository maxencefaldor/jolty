//! Bitwise determinism across thread counts, batch splits, and parallel task granularities.
//!
//! Verifies identical `f64::to_bits` outcomes and summaries at 1, 2, 3, and 8 threads across
//! split ranges, parallel frequency sweeps, sensitivity right-hand sides, device evaluation,
//! Newton unknown chunks, and nested-dissection factorizations.
#![cfg(test)]

#[path = "common/capacitor.rs"]
mod capacitor;
mod common;
#[path = "common/dc.rs"]
mod dc;
#[path = "common/inductor.rs"]
mod inductor;
#[path = "common/latch.rs"]
mod latch;
#[path = "common/network.rs"]
mod network;
#[path = "common/shared.rs"]
mod shared;

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use capacitor::capacitor;
use common::{circuit, device, resistor};
use dc::voltage_dc;
use inductor::inductor;
use jolty_engine::batch::{Batch, Measured, Run, Sink, Summary, Variants};
use jolty_engine::measure::{Measure, Signal};
use jolty_engine::variants::{
    Axis, Explicit, Grid, GridChain, MonteCarlo, Product, Toleranced, Variate, statistical,
    toleranced, variate,
};
use jolty_engine::{
    Analysis, AnalysisResult, BREAK_EVENS, Elaborated, OperatingPoints, Options, Parallelism,
    Record, Simulator, Values,
};
use jolty_ir::{Circuit, Distribution, Lot, Name, Tolerance};
use latch::latch;
use proptest::prelude::*;
use rayon::ThreadPoolBuilder;
use serde_json::json;
use shared::{Bits, Network, benchmark, bits, network, reals, shared, upper};

/// The thread counts every comparison runs at.
const THREADS: [usize; 4] = [1, 2, 3, 8];

/// Pairs every analysis in `analyses` with cloned `options` for batch execution.
fn with(analyses: Vec<Analysis>, options: &Options) -> Vec<(Analysis, Options)> {
    analyses
        .into_iter()
        .map(|analysis| (analysis, options.clone()))
        .collect()
}

/// Configures `options` to parallelize frequencies and sensitivity right-hand sides down to
/// one solve per task.
fn spreading(options: Options) -> Options {
    Options {
        parallelism: Parallelism {
            min_solves_per_task: Some(1),
            ..options.parallelism
        },
        ..options
    }
}

/// Raw bit representation of a batch run outcome: analysis bits, measure bits, and problems.
type OutcomeBits = (Vec<Bits>, Vec<Option<u64>>, String);

/// Extracts raw IEEE-754 bits and problem debug text from a batch [`Run`].
fn outcome_bits(outcome: &Run) -> OutcomeBits {
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
struct Kept(Mutex<BTreeMap<u64, Run>>);

impl Sink for Kept {
    fn accept(&self, index: u64, outcome: Run) {
        let mut kept = self.0.lock().expect("no test thread panicked");
        assert!(
            kept.insert(index, outcome).is_none(),
            "run {index} came twice"
        );
    }
}

/// Indexed batch outcomes as raw bits paired with the merged [`Summary`].
type Found = (BTreeMap<u64, OutcomeBits>, Summary);

/// Executes `ranges` of `variants` on a dedicated pool of `threads` threads, merging each
/// range summary in order.
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

/// Wraps a single `range` in a one-element slice array for [`run_on`].
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

/// Builds 50% uniform resistance tolerances for every resistor in `network` so samples cross
/// the 1 Ω restructuring boundary.
fn toleranced_resistors(network: &Network) -> Vec<Toleranced> {
    network
        .resistors
        .iter()
        .enumerate()
        .map(|(index, &(_, _, resistance))| Toleranced {
            instance: format!("R{index}"),
            parameter: "resistance".to_owned(),
            nominal: resistance,
            tolerance: Tolerance {
                relative: 0.5,
                distribution: Distribution::Uniform,
                lot: None,
            },
        })
        .collect()
}

/// Builds the operating point, 40-point AC, 40-point noise, and transient analyses for
/// network batch tests.
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

/// Builds a [`Measured`] entry reading node `n1` at the operating point.
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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// Verifies that a Monte Carlo batch with samples crossing 1 Ω restructuring boundaries
    /// yields identical bits across thread counts and range splits.
    #[test]
    fn a_batch_finds_the_same_however_its_runs_spread(
        network in network(true),
        seed in any::<u64>(),
    ) {
        let circuit = network.circuit();
        let elaborated = shared(&circuit);
        let samples = 24;
        let variants = MonteCarlo::new(&elaborated, seed, samples, toleranced_resistors(&network))
            .expect("the network's resistances");
        let options = Options { record: Record::All, ..Options::default() };
        let measures = node_one(&elaborated);
        let batch = Batch::new(elaborated, with(network_analyses(), &options), measures)
            .expect("a batch");
        assert_same_everywhere(&batch, &variants, &[vec![0..7, 7..8, 8..24], vec![0..13, 13..24]]);
    }

    /// Verifies that parallelized small-signal analyses and sensitivities match single-threaded
    /// execution bit for bit across all [`THREADS`] counts.
    #[test]
    fn small_signal_analyses_find_the_same_however_their_frequencies_spread(
        network in network(true),
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
                    .map(|analysis| bits(&simulator.run(analysis, &spreading(Options::default()))))
                    .collect::<Vec<_>>()
            })
        };
        let alone = found(1);
        for threads in THREADS {
            prop_assert_eq!(&found(threads), &alone, "at {} threads", threads);
        }
    }
}

/// Verifies that parallel frequency sweeps match sequential execution even when `G + jωC`
/// pivots switch between low and high frequencies.
///
/// Inductor diagonal entries `-jωL` cross the `0.1` pivot threshold relative to adjacent
/// unit entries as frequencies sweep up and back down.
#[test]
fn frequencies_whose_pivots_change_find_the_same_in_any_spread() {
    let circuit = circuit([
        voltage_dc("V1", "in", 0.0),
        resistor("R1", 50.0, "in", "a"),
        inductor("L1", 1e-3, "a", "b"),
        capacitor("C1", 1e-9, "b", "0"),
        inductor("L2", 1e-6, "b", "c"),
        capacitor("C2", 1e-12, "c", "0"),
        inductor("L3", 1e-5, "c", "0"),
        resistor("R2", 1e3, "c", "0"),
    ]);
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
                .map(|analysis| bits(&simulator.run(analysis, &spreading(Options::default()))))
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

/// Verifies that capping `max_threads` at 1 keeps work single-threaded while producing the
/// exact same bits as multi-threaded default and fine-grained execution.
#[test]
fn a_run_on_one_thread_finds_what_it_finds_on_many() {
    let (circuit, analyses, plan) = benchmark("ac_wide");
    let options = spreading(plan.clone());
    let serial = Options {
        parallelism: Parallelism {
            max_threads: Some(1),
            ..options.parallelism.clone()
        },
        ..options.clone()
    };
    let eight = ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .expect("a pool");
    let least = BREAK_EVENS
        .on(BREAK_EVENS.min_solves_per_task, 8)
        .expect("solves spread on 8 threads");
    let tasks = |parallelism: &Parallelism, items| eight.install(|| parallelism.tasks(items));
    assert_eq!(tasks(&Parallelism::default(), 8 * least - 1), 1);
    assert_eq!(tasks(&Parallelism::default(), 8 * least), 8);
    assert_eq!(tasks(&Parallelism::default(), 100 * least), 8);
    assert_eq!(tasks(&serial.parallelism, 10_000), 1);
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
    let found = run(&options);
    assert_eq!(run(&serial), found);
    assert_eq!(run(&plan), found);
}

/// Builds [`MonteCarlo`] resistor samples for `bjt_amp` using its 5% tolerances.
fn amplifier_samples(elaborated: &Elaborated, seed: u64, samples: u64) -> MonteCarlo {
    let (circuit, _, _) = benchmark("bjt_amp");
    let toleranced = toleranced(&circuit).expect("numbers");
    assert!(
        !toleranced.is_empty(),
        "the amplifier's resistors are toleranced"
    );
    MonteCarlo::new(elaborated, seed, samples, toleranced).expect("the amplifier's numbers")
}

/// Verifies that `bjt_amp` Monte Carlo samples match across all thread counts, range splits,
/// and isolated single-sample runs.
#[test]
fn amplifier_samples_find_the_same_alone_and_in_a_batch() {
    let (circuit, analyses, options) = benchmark("bjt_amp");
    let elaborated = shared(&circuit);
    let variants = amplifier_samples(&elaborated, 7, 16);
    // A range of fewer runs than threads spreads each run's frequencies too.
    let batch =
        Batch::new(elaborated, with(analyses, &spreading(options)), Vec::new()).expect("a batch");
    assert_same_everywhere(&batch, &variants, &[vec![0..5, 5..16]]);
    let (all, _) = run_on(&batch, &variants, &only(0..16), 8);
    for index in [0, 9, 15] {
        let (alone, summary) = run_on(&batch, &variants, &only(index..index + 1), 1);
        assert_eq!(summary.runs, 1);
        assert!(alone[&index] == all[&index], "sample {index}");
    }
}

/// Verifies that keyed Monte Carlo draws depend strictly on seed, index, and part path.
///
/// Adding an extra part leaves existing part draws unchanged, shared lots draw one common
/// factor, and identical seeds reproduce identical bits.
#[test]
fn a_part_added_leaves_every_other_draw_alone() {
    let part = |path: &str, lot: Option<&str>| Toleranced {
        instance: path.to_owned(),
        parameter: "resistance".to_owned(),
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
    let fewer = MonteCarlo::new(&small, 11, 50, toleranced_resistors(&network)).expect("numbers");
    let more = MonteCarlo::new(&large, 11, 50, toleranced_resistors(&larger)).expect("numbers");
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

/// Builds a 5 V RC voltage divider whose upper resistor `R1` is varied by [`upper`].
fn divider() -> Circuit {
    circuit([
        voltage_dc("V1", "top", 5.0),
        resistor("R1", 1e3, "top", "out"),
        resistor("R2", 1e3, "out", "0"),
        capacitor("C1", 1e-9, "out", "0"),
    ])
}

/// Verifies that invalid parameter runs record indexed failure outcomes without aborting
/// valid runs in the same batch.
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
        with(analyses, &Options::default()),
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

/// Verifies that batches alternating across the 1 Ω resistance boundary cache the alternate
/// structure once and match standalone runs bit for bit.
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
        with(analyses.clone(), &Options::default()),
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

/// Builds a shunted divider where lowering `R1` below 1 Ω inserts a branch current unknown
/// ahead of `V1:current`.
fn shunted() -> Circuit {
    circuit([
        resistor("R1", 2.0, "out", "0"),
        voltage_dc("V1", "top", 5.0),
        resistor("R2", 1e3, "top", "out"),
    ])
}

/// Verifies that measures remap shifted unknown columns accurately on restructured runs.
///
/// Whether `R1` sits above or below 1 Ω, `V1:current` matches standalone simulation bits and
/// [`Run::column`] maps nominal unknown indices to the run's layout.
#[test]
fn a_run_of_another_structure_measures_the_unknowns_of_its_names() {
    let elaborated = shared(&shunted());
    let named = |elaborated: &Elaborated, name: &str| {
        elaborated
            .unknowns()
            .iter()
            .position(|unknown| unknown.name == name)
            .expect("an unknown of the name")
    };
    let current = named(&elaborated, "V1:current");
    let measure = |unknown| Measure::Value {
        signal: Signal {
            unknown,
            reference: None,
        },
        at: None,
    };
    let measures = vec![
        Measured {
            analysis: 0,
            measure: measure(current),
        },
        Measured {
            analysis: 0,
            measure: measure(named(&elaborated, "out")),
        },
    ];
    let shunt = elaborated
        .parameter("R1", "resistance")
        .expect("a resistance");
    let runs: Vec<Values> = (0..16)
        .map(|index| {
            let mut values = Values::default();
            let ohms = if index % 2 == 0 { 0.5 } else { 2.0 };
            values.set(shunt, ohms + f64::from(index) * 1e-3);
            values
        })
        .collect();
    let variants = Explicit(runs);
    let batch = Batch::new(
        Arc::clone(&elaborated),
        with(vec![Analysis::OperatingPoint], &Options::default()),
        measures,
    )
    .expect("a batch");
    assert_same_everywhere(&batch, &variants, &[vec![0..5, 5..16]]);
    let pool = ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .expect("a pool");
    let kept = Kept::default();
    pool.install(|| batch.run(&variants, 0..16, &kept))
        .expect("the runs lie within the variants");
    for (index, outcome) in kept.0.into_inner().expect("no test thread panicked") {
        let mut alone = Simulator::from_elaborated(Arc::clone(&elaborated), variants.values(index))
            .expect("a run");
        let Ok(AnalysisResult::OperatingPoint(point)) =
            alone.run(&Analysis::OperatingPoint, &Options::default())
        else {
            panic!("an operating point");
        };
        let own = named(alone.elaborated(), "V1:current");
        assert_eq!(outcome.structure.is_some(), index % 2 == 0, "run {index}");
        assert_eq!(outcome.column(current), Some(own), "run {index}");
        assert_eq!(
            outcome.measured[0].as_ref().map(|value| value.to_bits()),
            Ok(point.values[own].to_bits()),
            "run {index}"
        );
        assert_eq!(outcome.values, variants.values(index), "run {index}");
    }
}

/// Verifies that warm-started [`Grid`] chains produce identical bits across thread counts
/// and mid-chain range splits.
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
    let grid = Grid::new(axes, GridChain::Warm).expect("a grid");
    assert!(grid.chain() > 1);
    let analyses = vec![Analysis::OperatingPoint];
    let batch = Batch::new(
        Arc::clone(&elaborated),
        with(analyses, &options),
        Vec::new(),
    )
    .expect("a batch");
    let len = grid.len();
    assert_same_everywhere(
        &batch,
        &grid,
        &[vec![0..4, 4..13, 13..len], vec![0..len - 1, len - 1..len]],
    );
}

/// Verifies that a warm [`Grid`] sweeping a source DC level matches [`Analysis::DcSweep`]
/// bit for bit.
#[test]
fn a_chain_of_levels_is_the_dc_sweep_of_them() {
    let elaborated = shared(&divider());
    let level = elaborated.parameter("V1", "waveform.dc").expect("a level");
    let levels: Vec<f64> = (0..21).map(|step| f64::from(step) * 0.25).collect();
    let grid = Grid::new(
        vec![(Axis::Parameter(level), levels.clone())],
        GridChain::Warm,
    )
    .expect("a grid");
    let batch = Batch::new(
        Arc::clone(&elaborated),
        with(vec![Analysis::OperatingPoint], &Options::default()),
        Vec::new(),
    )
    .expect("a batch");
    let (found, _) = run_on(&batch, &grid, &only(0..grid.len()), 4);
    let sweep: Analysis =
        serde_json::from_value(json!({"dc_sweep": {"source": "V1", "values": levels}}))
            .expect("a sweep");
    let mut simulator =
        Simulator::from_elaborated(Arc::clone(&elaborated), Values::default()).expect("a run");
    let Ok(AnalysisResult::DcSweep(swept)) = simulator.run(&sweep, &Options::default()) else {
        panic!("a sweep");
    };
    for (index, values) in swept.values.iter().enumerate() {
        let mut expected = Vec::new();
        reals(values, &mut expected);
        assert_eq!(found[&(index as u64)].0[0].0, expected, "level {index}");
    }
}

/// Verifies that [`Product`] runs every sample at every corner in row-major order
/// (`corner = i / samples`, `sample = i % samples`) deterministically across thread counts.
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
    let batch = Batch::new(elaborated, with(analyses, &options), Vec::new()).expect("a batch");
    assert_same_everywhere(&batch, &product, &[vec![0..6, 6..12]]);
}

/// Builds a forward-biased diode (10 fA saturation current) driven by 1 V through 1 kΩ.
fn forward_diode() -> Circuit {
    circuit([
        voltage_dc("V1", "top", 1.0),
        resistor("R1", 1e3, "top", "anode"),
        device(
            "D1",
            "diode",
            json!({"saturation_current": 1e-14}),
            &[("anode", "anode"), ("cathode", "0")],
        ),
    ])
}

/// Decade-spaced diode saturation currents shifting forward voltage by roughly 60 mV each.
const SATURATION_CURRENTS: [f64; 4] = [1e-15, 1e-14, 1e-13, 1e-12];

/// Computes the diode anode voltage from a fresh simulator constructed with `values`.
fn fresh_anode(elaborated: &Arc<Elaborated>, values: Values) -> f64 {
    let mut simulator = Simulator::from_elaborated(Arc::clone(elaborated), values).expect("a run");
    let Ok(AnalysisResult::OperatingPoint(point)) =
        simulator.run(&Analysis::OperatingPoint, &Options::default())
    else {
        panic!("an operating point");
    };
    let anode = simulator.unknown("anode").expect("the anode").index();
    point.values[anode]
}

/// Checks whether `found` matches `expected` within default SPICE Newton voltage tolerance.
fn within_tolerance(found: f64, expected: f64) -> bool {
    (found - expected).abs() <= 1e-6 + 1e-3 * expected.abs()
}

/// Verifies that a DC sweep over a diode kernel parameter re-prepares each value to match
/// fresh operating-point runs.
#[test]
fn a_sweep_of_a_kernels_number_evaluates_each_value() {
    let elaborated = shared(&forward_diode());
    let sweep: Analysis = serde_json::from_value(json!({"dc_sweep": {
        "source": "D1", "parameter": "saturation_current", "values": SATURATION_CURRENTS,
    }}))
    .expect("a sweep");
    let mut simulator =
        Simulator::from_elaborated(Arc::clone(&elaborated), Values::default()).expect("a run");
    let Ok(AnalysisResult::DcSweep(swept)) = simulator.run(&sweep, &Options::default()) else {
        panic!("a sweep");
    };
    let anode = simulator.unknown("anode").expect("the anode").index();
    let parameter = elaborated
        .parameter("D1", "saturation_current")
        .expect("a saturation current");
    for (values, current) in swept.values.iter().zip(SATURATION_CURRENTS) {
        let mut at = Values::default();
        at.set(parameter, current);
        let expected = fresh_anode(&elaborated, at);
        assert!(
            within_tolerance(values[anode], expected),
            "at {current:e} A the sweep finds {} V, a fresh run {expected} V",
            values[anode]
        );
    }
}

/// Verifies that a warm grid chain over a diode kernel parameter re-prepares each step to
/// match fresh operating-point runs.
#[test]
fn a_warm_chain_through_a_kernels_numbers_evaluates_each_runs() {
    let elaborated = shared(&forward_diode());
    let parameter = elaborated
        .parameter("D1", "saturation_current")
        .expect("a saturation current");
    let grid = Grid::new(
        vec![(Axis::Parameter(parameter), SATURATION_CURRENTS.to_vec())],
        GridChain::Warm,
    )
    .expect("a grid");
    assert_eq!(grid.chain(), SATURATION_CURRENTS.len() as u64);
    let batch = Batch::new(
        Arc::clone(&elaborated),
        with(vec![Analysis::OperatingPoint], &Options::default()),
        Vec::new(),
    )
    .expect("a batch");
    let (found, _) = run_on(&batch, &grid, &only(0..grid.len()), 1);
    let anode = elaborated
        .unknowns()
        .iter()
        .position(|unknown| unknown.name == "anode")
        .expect("the anode");
    for (index, current) in SATURATION_CURRENTS.into_iter().enumerate() {
        let mut at = Values::default();
        at.set(parameter, current);
        let expected = fresh_anode(&elaborated, at);
        let bits = found[&(index as u64)].0[0].0[anode];
        let chained = f64::from_bits(bits);
        assert!(
            within_tolerance(chained, expected),
            "at {current:e} A the chain finds {chained} V, a fresh run {expected} V"
        );
    }
}

/// Parallel `(min_instances_per_task, min_unknowns_per_task)` configurations exercised
/// below; `usize::MAX` disables parallel splitting along that axis.
const SPREADS: [(usize, usize); 5] = [
    (1, usize::MAX),
    (3, usize::MAX),
    (64, usize::MAX),
    (usize::MAX, 1),
    (3, 1),
];

/// Asserts that running `analyses` on `circuit` across 2, 3, and 8 threads under every
/// configuration in `spreads` matches single-threaded execution bit for bit.
fn spread_finds_what_one_thread_finds(
    circuit: &Circuit,
    analyses: &[Analysis],
    options: &Options,
    spreads: &[(usize, usize)],
) {
    let elaborated = shared(circuit);
    let run = |options: &Options| -> Vec<Bits> {
        let mut simulator =
            Simulator::from_elaborated(Arc::clone(&elaborated), Values::default()).expect("a run");
        analyses
            .iter()
            .map(|analysis| bits(&simulator.run(analysis, options)))
            .collect()
    };
    let serial = run(&Options {
        parallelism: Parallelism {
            max_threads: Some(1),
            ..options.parallelism.clone()
        },
        ..options.clone()
    });
    assert!(
        serial.iter().all(|(numbers, _)| !numbers.is_empty()),
        "{serial:?}"
    );
    for threads in [2, 3, 8] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a pool");
        for &(instances, unknowns) in spreads {
            let parallel = Options {
                parallelism: Parallelism {
                    min_instances_per_task: Some(instances),
                    min_unknowns_per_task: Some(unknowns),
                    ..options.parallelism.clone()
                },
                ..options.clone()
            };
            assert!(parallel.parallelism.spreads());
            let found = pool.install(|| run(&parallel));
            assert_eq!(
                found, serial,
                "{threads} threads, tasks of {instances} instances and {unknowns} unknowns"
            );
        }
    }
}

/// Verifies bitwise equivalence between single-threaded and parallel device/Newton
/// evaluation on `bjt_amp` and the first ten periods of the `buck` converter benchmark.
#[test]
fn a_run_spread_over_threads_finds_what_it_finds_on_one() {
    let (circuit, analyses, options) = benchmark("bjt_amp");
    spread_finds_what_one_thread_finds(&circuit, &analyses, &options, &SPREADS);
    let (circuit, _, options) = benchmark("buck");
    let periods: Analysis =
        serde_json::from_value(json!({"transient": {"stop": 2e-5, "method": "trapezoidal"}}))
            .expect("a transient");
    spread_finds_what_one_thread_finds(&circuit, &[periods], &options, &SPREADS);
}

/// Verifies bitwise equivalence across multi-chunk parallel reductions on the 3,504-unknown
/// `inverter_chain_500` transient (first 0.1 ns), covering divided differences, filtered
/// differences, and step-ratio reductions.
#[test]
fn a_transient_judged_over_many_chunks_finds_what_it_finds_on_one_thread() {
    let (circuit, _, options) = benchmark("inverter_chain_500");
    let start: Analysis =
        serde_json::from_value(json!({"transient": {"stop": 1e-10, "method": "trapezoidal"}}))
            .expect("a transient");
    let unknowns = shared(&circuit).unknowns().len();
    assert!(
        unknowns > 3 * jolty_numerics::reduce::CHUNK,
        "{unknowns} unknowns"
    );
    let spreads = [(usize::MAX, 1), (64, 1)];
    spread_finds_what_one_thread_finds(&circuit, &[start], &options, &spreads);
}

/// Verifies that deflated multi-solution DC search on [`latch`] produces identical bits on
/// one thread and across parallel thread spreads.
#[test]
fn a_deflated_search_spread_over_threads_finds_what_it_finds_on_one() {
    let search = Analysis::OperatingPoints(OperatingPoints { max_solutions: 8 });
    spread_finds_what_one_thread_finds(&latch(), &[search], &Options::default(), &SPREADS);
}

/// Configures `options` to enable nested-dissection factorization with interior blocks capped
/// at `interiors` unknowns when dissection reduces multiply-adds.
fn dissecting(options: &Options, interiors: usize) -> Options {
    Options {
        parallelism: Parallelism {
            max_unknowns_per_interior: Some(interiors),
            ..options.parallelism.clone()
        },
        ..options.clone()
    }
}

/// Checks that nested-dissection factorization with interior limit `interiors` produces
/// identical bits on 1, 2, 3, and 8 threads across `rc_mesh_100` operating point and
/// transient analyses.
///
/// Small blocks and chains skip dissection when minimum degree requires fewer multiply-adds;
/// delayed interior pivots and wide separator rows are covered directly by solver unit tests.
fn assert_dissected_factorization_matches_across_threads(interiors: usize) {
    let (mesh, _, mesh_options) = benchmark("rc_mesh_100");
    let start: Analysis =
        serde_json::from_value(json!({"transient": {"stop": 1e-7, "method": "trapezoidal"}}))
            .expect("a transient");
    let analyses = [Analysis::OperatingPoint, start];
    let options = dissecting(&mesh_options, interiors);
    let elaborated = shared(&mesh);
    let run = || -> Vec<Bits> {
        let mut simulator =
            Simulator::from_elaborated(Arc::clone(&elaborated), Values::default()).expect("a run");
        analyses
            .iter()
            .map(|analysis| bits(&simulator.run(analysis, &options)))
            .collect()
    };
    let mut found = Vec::new();
    for threads in THREADS {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a pool");
        found.push(pool.install(run));
    }
    assert!(
        found[0].iter().all(|(numbers, _)| !numbers.is_empty()),
        "{:?}",
        found[0]
    );
    for (threads, each) in THREADS.iter().zip(&found).skip(1) {
        assert_eq!(
            each, &found[0],
            "interiors {interiors} on {threads} threads"
        );
    }
}

#[test]
fn a_dissected_factorization_of_157_finds_the_same_on_any_number_of_threads() {
    assert_dissected_factorization_matches_across_threads(157);
}

#[test]
fn a_dissected_factorization_of_500_finds_the_same_on_any_number_of_threads() {
    assert_dissected_factorization_matches_across_threads(500);
}

/// Verifies that dissected and undissected operating points agree within twice Newton's
/// stopping tolerance (`2 * (reltol * max(|a|, |b|) + abstol)`), accounting for elimination
/// order differences.
#[test]
fn a_dissected_factorization_finds_the_operating_point_within_tolerance() {
    for (name, interiors) in [("rc_mesh_100", 157), ("rc_mesh_100", 500)] {
        let (circuit, _, options) = benchmark(name);
        let options = undissected(&options, &circuit);
        let elaborated = shared(&circuit);
        let point = |options: &Options| {
            let mut simulator =
                Simulator::from_elaborated(Arc::clone(&elaborated), Values::default())
                    .expect("a run");
            simulator
                .operating_point(options)
                .expect("an operating point")
        };
        let serial = point(&options);
        let dissected = point(&dissecting(&options, interiors));
        for ((unknown, &one), &other) in elaborated
            .unknowns()
            .iter()
            .zip(&serial.values)
            .zip(&dissected.values)
        {
            let absolute = if unknown.nature.is_current() {
                options.current_tolerance
            } else {
                options.voltage_tolerance
            };
            let tolerance = options.relative_tolerance * one.abs().max(other.abs()) + absolute;
            assert!(
                (one - other).abs() <= 2.0 * tolerance,
                "{name}: {} is {one} serially and {other} dissected",
                unknown.name
            );
        }
    }
}

/// Configures `options` to disable nested dissection on `circuit` by setting the interior
/// cap to the full unknown count.
fn undissected(options: &Options, circuit: &Circuit) -> Options {
    dissecting(options, shared(circuit).unknowns().len())
}

/// Verifies that switching between undissected and dissected options re-solves cleanly
/// instead of reusing incompatible cached solver state, matching a fresh simulator bit for
/// bit.
#[test]
fn a_dissected_run_solves_afresh_what_a_serial_one_left() {
    let (circuit, _, options) = benchmark("rc_mesh_100");
    let options = undissected(&options, &circuit);
    let elaborated = shared(&circuit);
    let dissected = dissecting(&options, 500);
    let start: Analysis =
        serde_json::from_value(json!({"transient": {"stop": 1e-7, "method": "trapezoidal"}}))
            .expect("a transient");
    let point =
        |simulator: &mut Simulator, options: &Options| bits(&simulator.run(&start, options));
    let mut fresh =
        Simulator::from_elaborated(Arc::clone(&elaborated), Values::default()).expect("a run");
    let alone = point(&mut fresh, &dissected);
    let mut used = Simulator::from_elaborated(elaborated, Values::default()).expect("a run");
    let serial = point(&mut used, &options);
    assert_ne!(serial.0, alone.0, "the orders of operations differ");
    assert_eq!(point(&mut used, &dissected), alone);
    assert_eq!(point(&mut used, &options), serial);
}

/// Builds a hierarchical circuit combining a component statistical model (targeting both a
/// linear resistor and a parameterized behavioral resistor) with a discrete toleranced load.
fn statistical_bench() -> Circuit {
    let stage_statistical = json!({
        "steps": [
            {"kind": "draw", "name": "lot", "shared": true, "call": "lot#0",
             "function": "agauss", "nominal": "0.0", "spread": "0.09", "sigma": "3.0"},
            {"kind": "draw", "name": "mm", "call": "mm#0",
             "function": "gauss", "nominal": "1000.0", "spread": "0.06", "sigma": "3.0"},
            {"kind": "value", "name": "rbody",
             "expression": "p(mm) * (1.0 + p(lot))"}
        ],
        "targets": [
            {"instance": "R1", "parameter": "resistance", "expression": "p(rbody)"},
            {"instance": "B1", "parameter": "parameters.rbody", "expression": "p(rbody) * 2.0"},
            {"instance": "B1", "parameter": "parameters.vc1",
             "expression": "0.05 * (1.0 + 0.5 * p(lot))"}
        ]
    });
    let document = json!({
        "format": "jolty-circuit", "version": jolty_ir::VERSION, "top": "top",
        "components": {
            "top": {
                "ports": [{"name": "0", "role": "ground"}],
                "instances": [
                    {"name": "V1", "of": {"model": "voltage_source"},
                     "parameters": {"waveform": {"dc": 1.8}},
                     "terminals": {"positive": "vdd", "negative": "0"}},
                    {"name": "X1", "of": {"component": "stage"},
                     "terminals": {"in": "vdd", "out": "out1"}},
                    {"name": "X2", "of": {"component": "stage"},
                     "terminals": {"in": "vdd", "out": "out2"}},
                    {"name": "RL1", "of": {"model": "resistor"},
                     "parameters": {"resistance": 1500.0},
                     "terminals": {"a": "out1", "b": "0"},
                     "tolerances": {
                         "resistance": {"relative": 0.05, "distribution": "normal"}
                     }},
                    {"name": "RL2", "of": {"model": "resistor"},
                     "parameters": {"resistance": 1500.0},
                     "terminals": {"a": "out2", "b": "0"}}
                ]
            },
            "stage": {
                "ports": [
                    {"name": "in", "role": "signal"},
                    {"name": "out", "role": "signal"}
                ],
                "instances": [
                    {"name": "R1", "of": {"model": "resistor"},
                     "parameters": {"resistance": 1000.0},
                     "terminals": {"a": "in", "b": "mid"}},
                    {"name": "B1", "of": {"model": "behavioral_current_source"},
                     "parameters": {
                         "expression": concat!(
                             "v(positive, negative) / ",
                             "(p(rbody) * (1 + p(vc1) * abs(v(positive, negative))))",
                         ),
                         "parameters": {"rbody": 2000.0, "vc1": 0.05}
                     },
                     "terminals": {"positive": "mid", "negative": "out"}}
                ],
                "statistical": stage_statistical
            }
        }
    });
    Circuit::from_json(&document.to_string()).expect("valid statistical circuit")
}

/// Verifies that [`MonteCarlo::with_statistical`] combined with tolerances and temperature
/// grids is bitwise deterministic across all thread counts and matches explicit single runs.
#[test]
fn statistical_monte_carlo_is_deterministic_and_matches_explicit_runs() {
    let circuit = statistical_bench();
    let elaborated = shared(&circuit);
    let toleranced_items = toleranced(&circuit).expect("toleranced items");
    let statistical_items = statistical(&circuit).expect("statistical placements");
    let seed = 41;
    let samples = 12;
    let monte_carlo = MonteCarlo::with_statistical(
        &elaborated,
        seed,
        samples,
        toleranced_items.clone(),
        statistical_items,
    )
    .expect("compiled statistical Monte Carlo");
    let batch = Batch::new(
        Arc::clone(&elaborated),
        with(vec![Analysis::OperatingPoint], &Options::default()),
        Vec::new(),
    )
    .expect("a batch");
    assert_same_everywhere(&batch, &monte_carlo, &[vec![0..4, 4..samples]]);

    let temperatures = Grid::new(
        vec![(Axis::Temperature, vec![273.15, 323.15])],
        GridChain::Independent,
    )
    .expect("a temperature grid");
    let product = Product::new(temperatures, monte_carlo.clone()).expect("a product");
    assert_same_everywhere(&batch, &product, &[vec![0..9, 9..24]]);

    let (all, _) = run_on(&batch, &monte_carlo, &only(0..samples), 4);
    for index in 0..samples {
        let lot = (0.09 / 3.0) * variate(Variate::Normal, seed, index, "", "lot#0");
        let mut overrides = BTreeMap::new();
        for scope in ["X1", "X2"] {
            let mm = 1000.0
                + (0.06 / 3.0) * 1000.0 * variate(Variate::Normal, seed, index, scope, "mm#0");
            let rbody = mm * (1.0 + lot);
            let vc1 = 0.05 * (1.0 + 0.5 * lot);
            overrides.insert(
                format!("{scope}/R1"),
                BTreeMap::from([("resistance".to_owned(), rbody)]),
            );
            overrides.insert(
                format!("{scope}/B1"),
                BTreeMap::from([
                    ("parameters.rbody".to_owned(), rbody * 2.0),
                    ("parameters.vc1".to_owned(), vc1),
                ]),
            );
        }
        overrides.insert(
            "RL1".to_owned(),
            BTreeMap::from([(
                "resistance".to_owned(),
                toleranced_items[0].drawn(seed, index),
            )]),
        );
        let values = elaborated
            .values(&overrides, None)
            .expect("valid overrides");
        let mut simulator =
            Simulator::from_elaborated(Arc::clone(&elaborated), values).expect("a run");
        let expected = bits(&simulator.run(&Analysis::OperatingPoint, &Options::default()));
        assert_eq!(all[&index].0, vec![expected], "sample {index}");
    }
}
