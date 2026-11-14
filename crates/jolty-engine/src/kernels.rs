//! The groups of the instances of kernels against the devices they were: at every
//! evaluation of a run, the circuit's equations as the groups assemble them, against the same
//! instances evaluated one by one as devices of their own, in the order of the circuit's
//! devices; and the groups' parallel evaluation against their serial one.

use std::collections::BTreeMap;
use std::path::Path;

use jolty_ir::{Circuit, Parameters};
use jolty_models::{Contributions, Input, Library, Runs, Tasks};
use serde::Deserialize;

use crate::elaborated::Elaborated;
use crate::{Analysis, Conditions, Options, Simulator, Substitute};

/// How far the groups' assembly of a run's equations has strayed from the devices'.
#[derive(Debug, Clone, Default)]
pub(crate) struct Check {
    /// The evaluations compared.
    pub(crate) evaluations: usize,
    /// The values compared, over every evaluation.
    pub(crate) values: usize,
    /// The largest difference between the two assemblies of a value, in units of
    /// `ε · Σ|t|`, where `Σ|t|` sums the magnitudes of the terms that add into it.
    pub(crate) worst: f64,
    /// The most terms a value sums.
    pub(crate) most: usize,
}

/// The bound on the difference between two sums of the same `terms` terms, of magnitudes
/// summing to `magnitude`, in different orders and groupings: each lies within
/// `(terms − 1) · u · magnitude` of the exact sum, with `u` half of `f64::EPSILON`
/// (Higham, "Accuracy and Stability of Numerical Algorithms", 2002, §4.2), so the two lie
/// within twice that of each other.
fn bound(terms: usize, magnitude: f64) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "a count of contributions is far below 2⁵³"
    )]
    let terms = terms as f64;
    (terms - 1.0).max(0.0) * f64::EPSILON * magnitude
}

/// A run's slab, as [`jolty_models::Runs::slab`] gives it.
pub(crate) type Slab<'a> = ([&'a [f64]; 3], [&'a [f64]; 2]);

impl Check {
    /// Compares `assembled`, the contributions the groups added from `slab` to `prefix`,
    /// those of the devices of no kernel, with the instances of `elaborated`'s groups
    /// evaluated at `input` each alone, from what `alone`, the groups' run before the
    /// evaluation, holds, one by one in the order of the circuit's devices, each one's
    /// local sums added to `prefix` as a device of its own adds them; `limited` says
    /// whether the groups' evaluation limited a voltage.
    ///
    /// Both sum the same terms: the value of `prefix`, and each instance's contributions
    /// to the value, which its block of the slab holds. The groups add them one by one in
    /// the order of the groups, of their instances and of each instance's local indices;
    /// the instances alone each one's own sum, in the order of the devices.
    pub(crate) fn compare(
        &mut self,
        elaborated: &Elaborated,
        alone: &mut Runs,
        (input, limited): (&Input<'_>, bool),
        prefix: &Contributions,
        (assembled, slab): (&Contributions, Slab<'_>),
    ) {
        let (sums, one_by_one) = alone_sums(elaborated, alone, input, prefix);
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
        let planes = [
            (
                "residual",
                (&assembled.residual, &sums.residual, &prefix.residual),
                (rows, residual),
            ),
            (
                "charge",
                (&assembled.charge, &sums.charge, &prefix.charge),
                (rows, charge),
            ),
            (
                "conductance",
                (
                    &assembled.conductance,
                    &sums.conductance,
                    &prefix.conductance,
                ),
                (slots, conductance),
            ),
            (
                "capacitance",
                (
                    &assembled.capacitance,
                    &sums.capacitance,
                    &prefix.capacitance,
                ),
                (slots, capacitance),
            ),
        ];
        for (name, (assembled, sums, prefix), (map, slab)) in planes {
            for (index, ((&assembled, &sum), &first)) in
                assembled.iter().zip(sums).zip(prefix).enumerate()
            {
                let sources = map.sources_of(index);
                let terms = sources.len() + 1;
                let magnitude = first.abs()
                    + sources
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

/// The contributions of the instances of `elaborated`'s groups evaluated at `input` each
/// alone, from what `alone` holds, one by one in the order of the circuit's devices, each
/// one's local sums added to `prefix` as a device of its own adds them, and whether one
/// limited its voltages.
fn alone_sums(
    elaborated: &Elaborated,
    alone: &mut Runs,
    input: &Input<'_>,
    prefix: &Contributions,
) -> (Contributions, bool) {
    let mut members: Vec<usize> = elaborated.groups.members().flatten().copied().collect();
    members.sort_unstable();
    let mut sums = prefix.clone();
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

/// A fixture of the benchmarks: its circuit and the analyses of its plan, with its
/// options.
fn fixture(name: &str) -> (Circuit, Vec<Analysis>, Options) {
    #[derive(Deserialize)]
    struct Plan {
        circuit: String,
        analyses: Vec<(String, Analysis)>,
        options: Options,
    }
    let perf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../perf");
    let plan: Plan = serde_json::from_str(
        &std::fs::read_to_string(perf.join(format!("plans/{name}.json"))).expect("a plan"),
    )
    .expect("a plan reads");
    let text = std::fs::read_to_string(perf.join(format!("circuits/{}.json", plan.circuit)))
        .expect("a circuit");
    let circuit = Circuit::from_json(&text).expect("a circuit reads");
    let analyses = plan
        .analyses
        .into_iter()
        .map(|(_, analysis)| analysis)
        .collect();
    (circuit, analyses, plan.options)
}

/// Every outcome of `analyses` run in order on a simulator of `circuit`, written out with
/// every number's shortest form, which tells any two bit patterns apart but NaNs', and the
/// simulator.
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

/// The analyses of a fixture shortened to run in a test's time: its transients cut to
/// `stop`, and its frequency responses to their first `frequencies`.
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

/// A circuit of the differential corpus or of an example that holds a kernel's devices,
/// what it is elaborated in, and the runs its tests or its plan make of it, as
/// `tests/kernel_corpus.py` writes them.
#[derive(Deserialize)]
struct Recorded {
    circuit: serde_json::Value,
    temperature: f64,
    overrides: BTreeMap<String, Parameters>,
    substitutes: BTreeMap<String, Substitute>,
    runs: Vec<(Analysis, Options)>,
}

#[test]
fn every_corpus_circuit_and_example_assembles_as_its_devices_to_within_1e_13() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/kernels.json");
    let text = std::fs::read_to_string(path).expect("the corpus");
    let recorded: Vec<Recorded> = serde_json::from_str(&text).expect("the corpus reads");
    let mut all = Check::default();
    for record in &recorded {
        let circuit = Circuit::from_json(&record.circuit.to_string()).expect("a circuit");
        let conditions = Conditions {
            temperature: record.temperature,
            overrides: record.overrides.clone(),
            substitutes: record.substitutes.clone(),
        };
        let mut simulator = Simulator::new(&circuit, &Library::standard(), &conditions)
            .expect("the circuit elaborates");
        simulator.state.check = Some(Check::default());
        for (analysis, options) in &record.runs {
            // An analysis that fails, as one a test expects to, was checked all the same.
            let _ = simulator.run(analysis, options);
        }
        let check = simulator.state.check.expect("a check");
        assert!(check.evaluations > 0, "{}: no evaluation", circuit.top);
        all.evaluations += check.evaluations;
        all.values += check.values;
        all.worst = all.worst.max(check.worst);
        all.most = all.most.max(check.most);
    }
    // Each value sums at most `all.most` contributions, so the bound is at most 1e-13 of
    // their magnitudes for sums of up to 451.
    assert!(bound(all.most, 1.0) <= 1e-13, "{all:?}");
}

#[test]
fn a_parallel_evaluation_is_the_serial_one_bit_for_bit() {
    let tasks = |instances: usize, targets: usize| Tasks {
        instances: instances.try_into().expect("a task holds an instance"),
        targets: targets.try_into().expect("a task gathers a target"),
    };
    for (name, stop) in [("bjt_amp", 2e-5), ("buck", 2e-6)] {
        let (circuit, analyses, options) = fixture(name);
        let analyses = shortened(analyses, stop, 5);
        let (serial, _) = outcomes(&circuit, &analyses, &options, |_| {});
        for threads in [1, 2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("a pool");
            for split in [tasks(1, 1), tasks(3, 7), tasks(64, 1024)] {
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
    // Voltages and currents all over, at which every transistor limits its steps from the
    // last, and some evaluate in reverse.
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
                let limited = simulator.evaluate(x, Some(1e-9), (1.0, 1e-12), true);
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
        for (instances, targets) in [(1, 1), (7, 100), (64, 1024), (1000, 4096)] {
            let split = Tasks {
                instances: instances.try_into().expect("a task holds an instance"),
                targets: targets.try_into().expect("a task gathers a target"),
            };
            let parallel = pool.install(|| evaluations(Some(split)));
            assert!(
                parallel == serial,
                "on {threads} threads in tasks of {split:?}"
            );
        }
    }
}
