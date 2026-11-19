//! Verification tests comparing grouped kernel evaluation against standalone one-by-one
//! device evaluation and comparing parallel group assembly against serial execution.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::LazyLock;

use jolty_ir::{Circuit, Parameters};
use jolty_models::{Contributions, Input, Library, Past, Runs, Tasks};
use serde::Deserialize;

use crate::elaborated::Elaborated;
use crate::{Analysis, Conditions, Options, Record, Simulator, Substitute};

/// Accumulated comparison statistics between grouped kernel assembly and standalone device
/// evaluation.
#[derive(Debug, Clone, Default)]
pub(crate) struct Check {
    /// Total circuit evaluations compared.
    pub(crate) evaluations: usize,
    /// Total scalar assembly entries compared across all evaluations.
    pub(crate) values: usize,
    /// Worst observed difference in units of `ε · Σ|t|`, where `Σ|t|` sums term magnitudes.
    pub(crate) worst: f64,
    /// Maximum number of summed contributions for any single assembled entry.
    pub(crate) most: usize,
    /// Saved arguments from the latest evaluation: `(unknowns, time, source_scale, gmin)`.
    pub(crate) last: Option<(Vec<f64>, Option<f64>, f64, f64)>,
    /// Count of recorder checks and individual recorded scalars compared.
    pub(crate) reads: (usize, usize),
}

impl Check {
    /// Compares recorder slab reads for `devices` in `runs` against evaluating each device
    /// alone without voltage limiting at the last evaluation state (reading `past` during
    /// transients).
    ///
    /// Verifies bit-identical residuals, charges, conductances, and capacitances for every
    /// checked device.
    pub(crate) fn compare_read(
        &mut self,
        elaborated: &Elaborated,
        runs: &Runs,
        past: Option<&Past>,
        devices: impl Iterator<Item = usize>,
    ) {
        let Some((x, time, scale, gmin)) = &self.last else {
            return;
        };
        let mut input = Input::new(x, *time, *gmin);
        input.source_scale = *scale;
        input.limiting = false;
        if time.is_some() {
            input.past = past;
        }
        let groups = &elaborated.groups;
        for device in devices {
            let (Some(block), Some(alone)) = (
                runs.block(groups, device),
                runs.clone().evaluate_alone(groups, device, &input),
            ) else {
                continue;
            };
            let entries = block.entries.len();
            let bits = |values: &[f64]| {
                values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            };
            let path = elaborated.devices[device].path();
            assert_eq!(
                bits(block.residual),
                bits(&alone.rows[0]),
                "{path}'s residuals"
            );
            assert_eq!(bits(block.charge), bits(&alone.rows[2]), "{path}'s charges");
            assert_eq!(
                bits(block.conductance),
                bits(&alone.entries[0][..entries]),
                "{path}'s conductances"
            );
            assert_eq!(
                bits(block.capacitance),
                bits(&alone.entries[1][..entries]),
                "{path}'s capacitances"
            );
            self.reads.1 += 2 * (block.residual.len() + entries);
        }
        self.reads.0 += 1;
    }
}

/// Computes Higham's rounding bound `2 * (terms - 1) * u * magnitude` between reordered
/// sums of `terms` terms with total `magnitude` (Higham, *Accuracy and Stability of
/// Numerical Algorithms*, 2002, §4.2; `u = f64::EPSILON / 2`).
fn bound(terms: usize, magnitude: f64) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "a count of contributions is far below 2⁵³"
    )]
    let terms = terms as f64;
    (terms - 1.0).max(0.0) * f64::EPSILON * magnitude
}

/// Borrowed evaluation slab slices matching [`jolty_models::Runs::slab`].
pub(crate) type Slab<'a> = ([&'a [f64]; 3], [&'a [f64]; 2]);

/// Assembled evaluation output verified against standalone device evaluation.
#[derive(Clone, Copy)]
pub(crate) struct Evaluated<'a> {
    pub(crate) input: &'a Input<'a>,
    pub(crate) limited: bool,
    pub(crate) assembled: &'a Contributions,
    pub(crate) slab: Slab<'a>,
}

impl Check {
    /// Compares assembled contributions and slab entries in `evaluated` against standalone
    /// one-by-one device evaluation of `alone` in placement order.
    ///
    /// Both paths sum identical per-instance slab terms in different orders (group order vs.
    /// device placement order), so every finite entry agrees within [`bound`].
    pub(crate) fn compare(
        &mut self,
        elaborated: &Elaborated,
        alone: &mut Runs,
        evaluated: Evaluated<'_>,
    ) {
        let Evaluated {
            input,
            limited,
            assembled,
            slab,
        } = evaluated;
        let (sums, one_by_one) = alone_sums(elaborated, alone, input);
        assert_eq!(limited, one_by_one, "whether a voltage was limited");
        let bits = |values: &[f64]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            bits(&assembled.scale),
            bits(&sums.scale),
            "the residuals' scales"
        );
        let (rows, slots) = elaborated.groups.gathers();
        let ([residual, _, charge], [conductance, capacitance]) = slab;
        // Matrix entries include base linear sums gathered from all linear instances.
        let (first, summed) = elaborated.groups.base();
        let planes = [
            (
                "residual",
                (&assembled.residual, &sums.residual),
                (rows, residual, false),
            ),
            (
                "charge",
                (&assembled.charge, &sums.charge),
                (rows, charge, false),
            ),
            (
                "conductance",
                (&assembled.conductance, &sums.conductance),
                (slots, conductance, true),
            ),
            (
                "capacitance",
                (&assembled.capacitance, &sums.capacitance),
                (slots, capacitance, true),
            ),
        ];
        for (name, (assembled, sums), (map, slab, based)) in planes {
            for (index, (&assembled, &sum)) in assembled.iter().zip(sums).enumerate() {
                let sources: Vec<u32> = map
                    .sources_of(index)
                    .iter()
                    .flat_map(|source| match (*source as usize).checked_sub(first) {
                        Some(value) if based => summed.sources_of(value),
                        _ => std::slice::from_ref(source),
                    })
                    .copied()
                    .collect();
                let terms = sources.len();
                let magnitude = sources
                    .iter()
                    .map(|&source| slab[source as usize].abs())
                    .sum::<f64>();
                // A sum that overflows, as a junction's current can at a first guess, does
                // in both.
                if !magnitude.is_finite() {
                    assert_eq!(assembled.to_bits(), sum.to_bits(), "{name} {index}");
                    continue;
                }
                let difference = (assembled - sum).abs();
                assert!(
                    difference <= bound(terms, magnitude),
                    "{name} {index}: the groups assemble {assembled:e}, the devices {sum:e}, \
                     of {terms} terms summing {magnitude:e} in magnitude"
                );
                if magnitude > 0.0 {
                    self.worst = self.worst.max(difference / (f64::EPSILON * magnitude));
                }
                self.most = self.most.max(terms);
                self.values += 1;
            }
        }
        self.evaluations += 1;
    }
}

/// Evaluates every grouped device in `elaborated` individually in placement order at
/// `input`, returning summed [`Contributions`] and whether any device limited a voltage.
fn alone_sums(
    elaborated: &Elaborated,
    alone: &mut Runs,
    input: &Input<'_>,
) -> (Contributions, bool) {
    let mut members: Vec<usize> = elaborated.groups.members().flatten().copied().collect();
    members.sort_unstable();
    let mut sums: Contributions = Contributions::new(&elaborated.pattern);
    let mut one_by_one = false;
    for &member in &members {
        let Some(block) = alone.evaluate_alone(&elaborated.groups, member, input) else {
            continue;
        };
        one_by_one |= block.limited;
        let [residual, scale, charge] = &block.rows;
        let [conductance, capacitance] = &block.entries;
        let rows = residual.iter().zip(scale).zip(charge);
        for (&row, ((&residual, &scale), &charge)) in block.nodes.iter().zip(rows) {
            let row = row as usize;
            if let (Some(sum), Some(largest), Some(held)) = (
                sums.residual.get_mut(row),
                sums.scale.get_mut(row),
                sums.charge.get_mut(row),
            ) {
                *sum += residual;
                *largest = largest.max(scale);
                *held += charge;
            }
        }
        let stored = conductance.iter().zip(capacitance);
        for (&slot, (&conductance, &capacitance)) in block.slots.iter().zip(stored) {
            let slot = slot as usize;
            if let (Some(sum), Some(held)) = (
                sums.conductance.get_mut(slot),
                sums.capacitance.get_mut(slot),
            ) {
                *sum += conductance;
                *held += capacitance;
            }
        }
    }
    (sums, one_by_one)
}

/// Loads benchmark fixture `name`, returning its circuit, planned analyses, and options.
fn fixture(name: &str) -> (Circuit, Vec<Analysis>, Options) {
    #[derive(Deserialize)]
    struct Plan {
        circuit: String,
        analyses: Vec<(String, Analysis)>,
        options: Options,
    }
    let speed = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/speed");
    let plan: Plan = serde_json::from_str(
        &std::fs::read_to_string(speed.join(format!("plans/{name}.json"))).expect("a plan"),
    )
    .expect("a plan reads");
    let text = std::fs::read_to_string(speed.join(format!("circuits/{}.json", plan.circuit)))
        .expect("a circuit");
    let circuit = Circuit::from_json(&text).expect("a circuit reads");
    let analyses = plan
        .analyses
        .into_iter()
        .map(|(_, analysis)| analysis)
        .collect();
    (circuit, analyses, plan.options)
}

/// Runs `analyses` on `circuit` after applying `prepare` and returns formatted `Debug`
/// outcomes (exact shortest float representations) alongside the final simulator.
fn outcomes(
    circuit: &Circuit,
    analyses: &[Analysis],
    options: &Options,
    prepare: impl FnOnce(&mut Simulator),
) -> (Vec<String>, Simulator) {
    let mut simulator = Simulator::new(circuit, &Library::standard(), &Conditions::default())
        .expect("the circuit elaborates");
    prepare(&mut simulator);
    let outcomes = analyses
        .iter()
        .map(|analysis| format!("{:?}", simulator.run(analysis, options)))
        .collect();
    (outcomes, simulator)
}

/// Truncates transient stop times to `stop` and AC sweeps to `frequencies` entries for
/// faster test runs.
fn shortened(analyses: Vec<Analysis>, stop: f64, frequencies: usize) -> Vec<Analysis> {
    analyses
        .into_iter()
        .map(|analysis| match analysis {
            Analysis::Transient(mut transient) => {
                transient.stop = transient.stop.min(stop);
                Analysis::Transient(transient)
            }
            Analysis::Ac(mut ac) => {
                ac.frequencies.truncate(frequencies);
                Analysis::Ac(ac)
            }
            other => other,
        })
        .collect()
}

#[test]
fn the_groups_assemble_what_their_instances_do_as_devices() {
    for (name, stop) in [("bjt_amp", 2e-5), ("buck", 2e-6)] {
        let (circuit, analyses, options) = fixture(name);
        let analyses = shortened(analyses, stop, 5);
        let (_, simulator) = outcomes(&circuit, &analyses, &options, |simulator| {
            simulator.state.check = Some(Check::default());
        });
        let check = simulator.state.check.expect("a check");
        assert!(check.evaluations > 10, "{name}: {check:?}");
    }
}

/// Corpus circuit entry containing circuit JSON, elaboration conditions, and planned runs
/// generated by `tests/kernel_corpus.py` and verified by `tests/test_kernel_corpus.py`.
#[derive(Deserialize)]
pub(crate) struct Recording {
    circuit: serde_json::Value,
    temperature: f64,
    overrides: BTreeMap<String, Parameters>,
    substitutes: BTreeMap<String, Substitute>,
    pub(crate) runs: Vec<(Analysis, Options)>,
}

impl Recording {
    /// Returns the cached slice of recorded kernel corpus circuits, parsed once per process.
    pub(crate) fn corpus() -> &'static [Self] {
        static CORPUS: LazyLock<Vec<Recording>> = LazyLock::new(|| {
            let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/kernels.json");
            let text = std::fs::read_to_string(path).expect("the corpus");
            serde_json::from_str(&text).expect("the corpus reads")
        });
        &CORPUS
    }

    /// Parses the recorded circuit and builds a [`Simulator`] under its recorded conditions.
    pub(crate) fn simulator(&self) -> (Circuit, Simulator) {
        let circuit = Circuit::from_json(&self.circuit.to_string()).expect("a circuit");
        let conditions = Conditions {
            temperature: self.temperature,
            overrides: self.overrides.clone(),
            substitutes: self.substitutes.clone(),
        };
        let simulator = Simulator::new(&circuit, &Library::standard(), &conditions)
            .expect("the circuit elaborates");
        (circuit, simulator)
    }
}

/// Generates `#[test]` functions calling `$runner(shard, total)`, assigning consecutive
/// shard indices `0..total` automatically from the list of test names so no shard can be
/// skipped or duplicated.
macro_rules! sharded_tests {
    ($runner:path; $($name:ident),+ $(,)?) => {
        crate::kernels::sharded_tests!(
            @step $runner;
            [$(stringify!($name)),+].len();
            0usize;
            $($name),+
        );
    };
    (@step $runner:path; $total:expr; $index:expr; $name:ident $(, $rest:ident)*) => {
        #[test]
        fn $name() {
            $runner($index, $total);
        }
        crate::kernels::sharded_tests!(@step $runner; $total; $index + 1usize; $($rest),*);
    };
    (@step $runner:path; $total:expr; $index:expr;) => {};
}

pub(crate) use sharded_tests;

/// Checks one shard (`index % shards == shard`) of the kernel corpus.
fn check_corpus_shard(shard: usize, shards: usize) {
    let mut all = Check::default();
    let mut checked_circuits = 0_usize;
    for (index, record) in Recording::corpus().iter().enumerate() {
        if index % shards != shard {
            continue;
        }
        checked_circuits += 1;
        let (circuit, mut simulator) = record.simulator();
        simulator.state.check = Some(Check::default());
        for (analysis, options) in &record.runs {
            let options = Options {
                record: Record::All,
                ..options.clone()
            };
            // Failing analyses are still checked through every evaluation they perform.
            let _ = simulator.run(analysis, &options);
        }
        let check = simulator.state.check.expect("a check");
        assert!(check.evaluations > 0, "{}: no evaluation", circuit.top);
        all.evaluations += check.evaluations;
        all.values += check.values;
        all.worst = all.worst.max(check.worst);
        all.most = all.most.max(check.most);
        all.reads.0 += check.reads.0;
        all.reads.1 += check.reads.1;
    }
    assert!(checked_circuits > 0, "shard {shard}/{shards} was empty");
    // Each value sums at most `all.most` contributions, so the bound is at most 1e-13 of
    // their magnitudes for sums of up to 451.
    assert!(bound(all.most, 1.0) <= 1e-13, "{all:?}");
    assert!(all.reads.0 > 0, "{all:?}");
}

// Verifies every corpus evaluation with full recording enabled: grouped assembly matches
// standalone device sums within 1e-13 of total term magnitude, and recorder slab reads
// match standalone device evaluation bit for bit. Sharded across 16 tests for parallel
// nextest execution.
sharded_tests!(
    check_corpus_shard;
    every_corpus_circuit_assembles_as_its_devices_shard_00,
    every_corpus_circuit_assembles_as_its_devices_shard_01,
    every_corpus_circuit_assembles_as_its_devices_shard_02,
    every_corpus_circuit_assembles_as_its_devices_shard_03,
    every_corpus_circuit_assembles_as_its_devices_shard_04,
    every_corpus_circuit_assembles_as_its_devices_shard_05,
    every_corpus_circuit_assembles_as_its_devices_shard_06,
    every_corpus_circuit_assembles_as_its_devices_shard_07,
    every_corpus_circuit_assembles_as_its_devices_shard_08,
    every_corpus_circuit_assembles_as_its_devices_shard_09,
    every_corpus_circuit_assembles_as_its_devices_shard_10,
    every_corpus_circuit_assembles_as_its_devices_shard_11,
    every_corpus_circuit_assembles_as_its_devices_shard_12,
    every_corpus_circuit_assembles_as_its_devices_shard_13,
    every_corpus_circuit_assembles_as_its_devices_shard_14,
    every_corpus_circuit_assembles_as_its_devices_shard_15,
);

/// Builds a [`Tasks`] descriptor with at least `least` instances per task across `threads`.
fn tasks(least: usize, threads: usize) -> Tasks {
    Tasks {
        least: least.try_into().expect("a task holds an instance"),
        threads: threads.try_into().expect("a pool holds a thread"),
    }
}

#[test]
fn a_parallel_evaluation_is_the_serial_one_bit_for_bit() {
    for (name, stop) in [("bjt_amp", 2e-5), ("buck", 2e-6)] {
        let (circuit, analyses, options) = fixture(name);
        let analyses = shortened(analyses, stop, 5);
        let (serial, _) = outcomes(&circuit, &analyses, &options, |_| {});
        for threads in [1, 2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("a pool");
            for split in [tasks(1, threads), tasks(3, threads), tasks(64, threads)] {
                let (parallel, _) = pool.install(|| {
                    outcomes(&circuit, &analyses, &options, |simulator| {
                        simulator.state.tasks = Some(split);
                    })
                });
                assert!(
                    parallel == serial,
                    "{name} on {threads} threads in tasks of {split:?}"
                );
            }
        }
    }
}

#[test]
fn a_parallel_evaluation_of_a_thousand_transistors_is_the_serial_one_bit_for_bit() {
    let (circuit, _, _) = fixture("inverter_chain_500");
    let mut simulator = Simulator::new(&circuit, &Library::standard(), &Conditions::default())
        .expect("the circuit elaborates");
    let unknowns = simulator.unknowns().len();
    // Spread arbitrary voltages across all unknowns so transistors trigger limiting and
    // reverse-mode evaluation, exercising multi-chunk gather (2,000 contributions on supply).
    let points: Vec<Vec<f64>> = (1..=3)
        .map(|point| {
            (0..unknowns)
                .map(|index| {
                    #[expect(clippy::cast_precision_loss, reason = "a few thousand indices")]
                    let index = index as f64;
                    3.3 * (0.37 * index + f64::from(point)).sin()
                })
                .collect()
        })
        .collect();
    let mut evaluations = |tasks: Option<Tasks>| {
        simulator
            .reset(crate::Values::default())
            .expect("the values apply");
        simulator.state.tasks = tasks;
        points
            .iter()
            .map(|x| {
                let limited = simulator.evaluate(x, Some(1e-9), 1.0, 1e-12, true);
                let contributions = &simulator.state.workspace.contributions;
                format!("{limited} {contributions:?}")
            })
            .collect::<Vec<_>>()
    };
    let serial = evaluations(None);
    for threads in [1, 2, 3, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a pool");
        for instances in [1, 7, 64, 1000] {
            let split = tasks(instances, threads);
            let parallel = pool.install(|| evaluations(Some(split)));
            assert!(
                parallel == serial,
                "on {threads} threads in tasks of {split:?}"
            );
        }
    }
}

/// Builds a star network with `arms` 1 kΩ resistors from a hub to separate 1 nF ground
/// capacitors, plus a diode and 1 V / 1 Ω source at the hub.
fn star(arms: usize) -> Circuit {
    use serde_json::json;
    let mut instances = vec![
        json!({"name": "V1", "of": {"model": "voltage_source"},
            "parameters": {"waveform": {"dc": 1.0}},
            "terminals": {"positive": "supply", "negative": "ground"}}),
        json!({"name": "R0", "of": {"model": "resistor"}, "parameters": {"resistance": 1.0},
            "terminals": {"a": "supply", "b": "hub"}}),
        json!({"name": "D1", "of": {"model": "diode"}, "parameters": {},
            "terminals": {"anode": "hub", "cathode": "ground"}}),
    ];
    for arm in 1..=arms {
        let node = format!("n{arm}");
        instances.push(
            json!({"name": format!("R{arm}"), "of": {"model": "resistor"},
            "parameters": {"resistance": 1e3}, "terminals": {"a": "hub", "b": node}}),
        );
        instances.push(
            json!({"name": format!("C{arm}"), "of": {"model": "capacitor"},
            "parameters": {"capacitance": 1e-9}, "terminals": {"a": node, "b": "ground"}}),
        );
    }
    let document = json!({
        "format": "jolty-circuit", "version": 5, "top": "star",
        "components": {"star": {
            "ports": [{"name": "ground", "role": "ground"}],
            "instances": instances
        }}
    });
    Circuit::from_json(&document.to_string()).expect("a circuit")
}

/// Verifies bit-identical parallel and serial evaluation and transient results on a 2,000-arm
/// linear star with a hub diode (exercising multi-chunk gathers and base conductance sums).
#[test]
fn a_parallel_evaluation_of_a_star_of_linear_arms_is_the_serial_one_bit_for_bit() {
    let circuit = star(2000);
    let mut simulator = Simulator::new(&circuit, &Library::standard(), &Conditions::default())
        .expect("the circuit elaborates");
    let unknowns = simulator.unknowns().len();
    let points: Vec<Vec<f64>> = (1..=3)
        .map(|point| {
            (0..unknowns)
                .map(|index| {
                    #[expect(clippy::cast_precision_loss, reason = "a few thousand indices")]
                    let index = index as f64;
                    0.9 * (0.37 * index + f64::from(point)).sin()
                })
                .collect()
        })
        .collect();
    let mut evaluations = |tasks: Option<Tasks>| {
        simulator
            .reset(crate::Values::default())
            .expect("the values apply");
        simulator.state.tasks = tasks;
        points
            .iter()
            .map(|x| {
                let limited = simulator.evaluate(x, Some(1e-9), 1.0, 1e-12, true);
                let contributions = &simulator.state.workspace.contributions;
                format!("{limited} {contributions:?}")
            })
            .collect::<Vec<_>>()
    };
    let serial = evaluations(None);
    let transient = [Analysis::Transient(crate::Transient {
        stop: 2e-6,
        max_step: None,
        method: crate::Method::Trapezoidal,
        breakpoints: Vec::new(),
        initial_conditions: BTreeMap::new(),
        operating_point: true,
    })];
    let (waveforms, _) = outcomes(&circuit, &transient, &Options::default(), |_| {});
    for threads in [1, 2, 3, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a pool");
        for instances in [1, 7, 64, 1000] {
            let split = tasks(instances, threads);
            let parallel = pool.install(|| evaluations(Some(split)));
            assert!(
                parallel == serial,
                "on {threads} threads in tasks of {split:?}"
            );
            let options = Options {
                parallelism: crate::Parallelism {
                    min_instances_per_task: Some(instances),
                    ..crate::Parallelism::default()
                },
                ..Options::default()
            };
            let (spread, _) = pool.install(|| outcomes(&circuit, &transient, &options, |_| {}));
            assert!(
                spread == waveforms,
                "a transient on {threads} threads in tasks of {split:?}"
            );
        }
    }
}
