//! A circuit elaborated once and run many times: a run of the shared circuit at some values
//! equals the circuit elaborated with those values, a run reset to other values equals one
//! made afresh for them, and values that change a device's unknowns run on the circuit
//! restructured for them. Every comparison is of every number's bits.
#![cfg(test)]

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use jolty_engine::{
    Analysis, Conditions, Derivatives, Elaborated, ElaborationProblem, Options, Outcome, Record,
    SimulationError, Simulator, Structure, Values,
};
use jolty_ir::{Circuit, Parameters};
use jolty_models::Library;
use jolty_numerics::Complex64;
use proptest::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};

/// Numbers that replace devices' own, by the device's path, then by the number's path.
type Overrides = BTreeMap<String, BTreeMap<String, f64>>;

/// Every number of an outcome, as its bits, and the whole outcome as text: Rust writes
/// each number in the shortest form that reads back as the same bits, so equal text with
/// equal bits is the same outcome, its counts of work among it.
#[derive(Debug, PartialEq, Eq)]
struct Bits {
    numbers: Vec<u64>,
    text: String,
}

fn reals(values: &[f64], into: &mut Vec<u64>) {
    into.extend(values.iter().map(|value| value.to_bits()));
}

fn complexes(values: &[Complex64], into: &mut Vec<u64>) {
    for value in values {
        into.extend([value.re.to_bits(), value.im.to_bits()]);
    }
}

fn point(point: &jolty_engine::OperatingPoint, into: &mut Vec<u64>) {
    reals(&point.values, into);
    let observation = &point.observation;
    for values in [
        &observation.currents,
        &observation.charges,
        &observation.conductances,
        &observation.capacitances,
        &observation.variables,
    ] {
        reals(values, into);
    }
}

/// The bits of the outcome of a run, or its error.
fn bits(outcome: &Result<Outcome, SimulationError>) -> Bits {
    let mut numbers = Vec::new();
    if let Ok(found) = outcome {
        numbers_of(found, &mut numbers);
    }
    Bits {
        numbers,
        text: format!("{outcome:?}"),
    }
}

/// Every number of `outcome`, as its bits.
fn numbers_of(outcome: &Outcome, numbers: &mut Vec<u64>) {
    match outcome {
        Outcome::OperatingPoint(found) => point(found, numbers),
        Outcome::Ac(response) => {
            point(&response.operating_point, numbers);
            reals(&response.frequencies, numbers);
            for row in response.values.iter().chain(&response.currents) {
                complexes(row, numbers);
            }
        }
        Outcome::Transient(waveforms) => {
            reals(&waveforms.times, numbers);
            for row in &waveforms.values {
                reals(row, numbers);
            }
            let observed = &waveforms.observations;
            for values in [
                &observed.currents,
                &observed.charges,
                &observed.conductances,
                &observed.capacitances,
                &observed.variables,
            ] {
                reals(values, numbers);
            }
            for trace in &waveforms.logic {
                numbers.extend(trace.iter().map(|(time, _)| time.to_bits()));
            }
        }
        Outcome::DcSweep(sweep) => {
            reals(&sweep.levels, numbers);
            for row in &sweep.values {
                reals(row, numbers);
            }
        }
        Outcome::Sensitivity(found) => {
            numbers.extend(
                found
                    .parameters
                    .iter()
                    .map(|parameter| parameter.value.to_bits()),
            );
            match &found.derivatives {
                Derivatives::Real(points) => {
                    for row in points.iter().flatten() {
                        reals(row, numbers);
                    }
                }
                Derivatives::Complex(points) => {
                    for row in points.iter().flatten() {
                        complexes(row, numbers);
                    }
                }
            }
            numbers_of(&found.result, numbers);
        }
        _ => panic!("no other analysis is compared here"),
    }
}

/// The bits of each analysis of `analyses`, run in turn on `simulator`.
fn run(simulator: &mut Simulator, analyses: &[Analysis], options: &Options) -> Vec<Bits> {
    analyses
        .iter()
        .map(|analysis| bits(&simulator.run(analysis, options)))
        .collect()
}

/// A plan of the engine's benchmarks, `perf/plans/<name>.json`.
#[derive(Deserialize)]
struct Plan {
    circuit: String,
    analyses: Vec<(String, Analysis)>,
    options: Options,
}

/// The circuit and the analyses of the benchmark `bjt_amp`: two 2N3904s and three 2N3906s,
/// whose junctions limit their steps from their last voltages, over 200 µs of a sine, at
/// an operating point and at 71 frequencies. The transient comes first, so that it solves
/// the operating point it starts from, and counts the factorizations of a solver that has
/// kept no pivots.
fn bjt_amp() -> (Circuit, Vec<Analysis>, Options) {
    let perf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../perf");
    let plan: Plan = serde_json::from_str(
        &fs::read_to_string(perf.join("plans/bjt_amp.json")).expect("the plan reads"),
    )
    .expect("a plan");
    let text = fs::read_to_string(perf.join(format!("circuits/{}.json", plan.circuit)))
        .expect("the circuit reads");
    let circuit = Circuit::from_json(&text).expect("a circuit");
    let mut analyses: Vec<Analysis> = plan
        .analyses
        .into_iter()
        .map(|(_, analysis)| analysis)
        .collect();
    analyses.rotate_right(1);
    assert!(matches!(analyses[0], Analysis::Transient(_)));
    (circuit, analyses, plan.options)
}

/// Every number of every device of `circuit`, by the device's path then the number's.
fn every_number(circuit: &Circuit) -> Vec<(String, String, f64)> {
    let elaborated = shared(circuit);
    let mut found = Vec::new();
    for device in elaborated.devices() {
        if let Some(parameters) = device.parameters() {
            for (name, value) in jolty_ir::numbers(parameters) {
                found.push((device.path().to_owned(), name, value));
            }
        }
    }
    found
}

/// `circuit` elaborated once, for runs to share.
fn shared(circuit: &Circuit) -> Arc<Elaborated> {
    Elaborated::new(circuit, &Library::standard(), &Structure::default())
        .expect("the circuit elaborates")
}

/// The circuit elaborated with `overrides` at `temperature`, as `Simulator::new` takes it.
fn elaborated_with(
    circuit: &Circuit,
    overrides: &Overrides,
    temperature: f64,
) -> Result<Simulator, jolty_engine::ElaborationError> {
    let mut conditions = Conditions::at(temperature);
    for (path, numbers) in overrides {
        // Each number by its path among the parameters, such as `waveform.dc`.
        let parameters: Parameters = numbers
            .iter()
            .map(|(name, &value)| (name.clone(), json!(value)))
            .collect();
        conditions.overrides.insert(path.clone(), parameters);
    }
    Simulator::new(circuit, &Library::standard(), &conditions)
}

/// The same run two ways: the circuit elaborated with `overrides` at `temperature`, and the
/// circuit elaborated once and run at the values of the same numbers.
fn both_ways(
    circuit: &Circuit,
    overrides: &Overrides,
    temperature: f64,
    (analyses, options): (&[Analysis], &Options),
) -> (Vec<Bits>, Vec<Bits>) {
    let direct = elaborated_with(circuit, overrides, temperature);
    let elaborated = shared(circuit);
    let values = elaborated
        .values(overrides, Some(temperature))
        .expect("values of the circuit's numbers");
    let from_shared = Simulator::from_elaborated(elaborated, values);
    match (direct, from_shared) {
        (Ok(mut direct), Ok(mut from_shared)) => {
            assert_eq!(direct.unknowns(), from_shared.unknowns());
            (
                run(&mut direct, analyses, options),
                run(&mut from_shared, analyses, options),
            )
        }
        (direct, from_shared) => {
            let problems = |result: Result<Simulator, jolty_engine::ElaborationError>| {
                result.err().map(|error| error.to_string())
            };
            let (direct, from_shared) = (problems(direct), problems(from_shared));
            assert_eq!(direct, from_shared, "the problems of the values differ");
            (Vec::new(), Vec::new())
        }
    }
}

/// A grounded network of resistors and capacitors, driven by a pulse and by current
/// sources: its resistors, as `(node, node, resistance)`, node 0 the ground.
#[derive(Debug, Clone)]
struct Network {
    resistors: Vec<(usize, usize, f64)>,
    capacitors: Vec<(usize, f64)>,
    sources: Vec<(usize, f64)>,
}

/// The numbers of a network's pulse, `V1`.
const PULSE_NUMBERS: usize = 7;

fn node(index: usize) -> String {
    if index == 0 {
        "0".to_owned()
    } else {
        format!("n{index}")
    }
}

/// A connected network on up to six nodes: a spanning tree, extra resistors, a capacitor
/// at every node, the pulse at node 1, and current sources. Resistances run from 0.1 Ω to
/// 100 kΩ, so that some lie below 1 Ω, where a resistor carries its current as an unknown.
fn network() -> impl Strategy<Value = Network> {
    (2usize..7).prop_flat_map(|nodes| {
        let tree =
            proptest::collection::vec((any::<prop::sample::Index>(), -1.0..5.0f64), nodes - 1);
        let extra = proptest::collection::vec((0..nodes, 0..nodes, -1.0..5.0f64), 0..nodes);
        let capacitors = proptest::collection::vec(-12.0..-9.0f64, nodes - 1);
        let sources = proptest::collection::vec((1..nodes, -1e-3..1e-3f64), 0..3);
        (tree, extra, capacitors, sources).prop_map(|(tree, extra, capacitors, sources)| {
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
            let capacitors = capacitors
                .into_iter()
                .enumerate()
                .map(|(index, exponent)| (index + 1, 10f64.powf(exponent)))
                .collect();
            Network {
                resistors,
                capacitors,
                sources,
            }
        })
    })
}

impl Network {
    fn circuit(&self) -> Circuit {
        let mut instances = vec![json!({
            "id": "V1", "of": {"device": "voltage_source"},
            "parameters": {"waveform": {"pulse": {"initial": 0.0, "pulsed": 1.0, "delay": 1e-9,
                "rise": 1e-9, "fall": 1e-9, "width": 5e-9, "period": 20e-9}}},
            "terminals": {"positive": "drive", "negative": "0"},
        })];
        instances.push(json!({
            "id": "RD", "of": {"device": "resistor"}, "parameters": {"resistance": 50.0},
            "terminals": {"a": "drive", "b": node(1)},
        }));
        for (index, &(a, b, resistance)) in self.resistors.iter().enumerate() {
            instances.push(json!({
                "id": format!("R{index}"), "of": {"device": "resistor"},
                "parameters": {"resistance": resistance},
                "terminals": {"a": node(a), "b": node(b)},
            }));
        }
        for &(at, capacitance) in &self.capacitors {
            instances.push(json!({
                "id": format!("C{at}"), "of": {"device": "capacitor"},
                "parameters": {"capacitance": capacitance},
                "terminals": {"a": node(at), "b": "0"},
            }));
        }
        for (index, &(at, current)) in self.sources.iter().enumerate() {
            instances.push(json!({
                "id": format!("I{index}"), "of": {"device": "current_source"},
                "parameters": {"waveform": {"dc": current}},
                "terminals": {"positive": "0", "negative": node(at)},
            }));
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

/// Every number of `circuit` that `factors` scales, each by its factor, by its place
/// among the numbers of every device but those at `kept`; a factor of 1 leaves its number
/// alone.
fn scaled(circuit: &Circuit, kept: &[&str], factors: &[f64]) -> Overrides {
    let mut overrides = Overrides::new();
    let varied = every_number(circuit)
        .into_iter()
        .filter(|(path, _, _)| !kept.contains(&path.as_str()));
    for ((path, name, value), &factor) in varied.zip(factors) {
        if factor.to_bits() != 1f64.to_bits() {
            overrides
                .entry(path)
                .or_default()
                .insert(name, value * factor);
        }
    }
    overrides
}

/// A factor for each of `count` numbers: 1 for about half of them, which keep their own,
/// and else one within a factor `spread` of it.
fn factors(count: usize, spread: f64) -> impl Strategy<Value = Vec<f64>> {
    proptest::collection::vec(
        prop_oneof![
            Just(1.0),
            (-1.0..1.0f64).prop_map(move |exponent| spread.powf(exponent))
        ],
        count..=count,
    )
}

/// A network's operating point, its sensitivities there, and its transient, recording
/// every device.
fn network_analyses() -> (Vec<Analysis>, Options) {
    let analyses = vec![
        Analysis::OperatingPoint,
        serde_json::from_value(json!({"sensitivity": {"outputs": ["n1"]}})).expect("a sensitivity"),
        serde_json::from_value(json!({"transient": {"stop": 4e-8}})).expect("a transient"),
    ];
    let options = Options {
        record: Record::All,
        ..Options::default()
    };
    (analyses, options)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// A run of a random network elaborated once, at the values of random numbers and a
    /// random temperature, equals the network elaborated with those numbers, bit for bit:
    /// its resistors, capacitors and current sources within a decade of their own, so that
    /// resistances cross 1 Ω, and its pulse as it is.
    #[test]
    fn overrides_run_the_same_on_a_shared_circuit(
        (network, factors) in network().prop_flat_map(|network| {
            let count = every_number(&network.circuit()).len() - PULSE_NUMBERS;
            (Just(network), factors(count, 10.0))
        }),
        temperature in 250.0..400.0f64,
    ) {
        let circuit = network.circuit();
        let overrides = scaled(&circuit, &["V1"], &factors);
        let (analyses, options) = network_analyses();
        let (direct, shared) = both_ways(&circuit, &overrides, temperature, (&analyses, &options));
        prop_assert_eq!(direct, shared);
    }

    /// So does a run of `bjt_amp`, every number of its transistors and its sources within a
    /// factor 2 of its own, recording every device.
    #[test]
    fn overrides_run_the_same_on_a_shared_amplifier(
        factors in factors(every_number(&bjt_amp().0).len(), 2.0),
        temperature in 250.0..400.0f64,
    ) {
        let (circuit, analyses, options) = bjt_amp();
        let options = Options {
            record: Record::All,
            ..options
        };
        let overrides = scaled(&circuit, &[], &factors);
        let (direct, shared) = both_ways(&circuit, &overrides, temperature, (&analyses, &options));
        prop_assert_eq!(direct, shared);
    }
}

/// Two sets of values of `bjt_amp`: its resistors scaled one way, then the other, at two
/// temperatures.
fn two_values(elaborated: &Elaborated) -> (Values, Values) {
    let resistors: Vec<(String, String, f64)> = elaborated
        .devices()
        .iter()
        .filter(|device| device.model() == "resistor")
        .filter_map(|device| {
            let resistance = device.parameters()?.get("resistance")?.as_f64()?;
            Some((
                device.path().to_owned(),
                "resistance".to_owned(),
                resistance,
            ))
        })
        .collect();
    let values = |factor: f64, temperature: f64| {
        let mut overrides = Overrides::new();
        for (index, (path, name, value)) in resistors.iter().enumerate() {
            let scale = if index % 2 == 0 { factor } else { 1.0 / factor };
            overrides
                .entry(path.clone())
                .or_default()
                .insert(name.clone(), value * scale);
        }
        elaborated
            .values(&overrides, Some(temperature))
            .expect("values of the circuit's numbers")
    };
    (values(1.05, 290.0), values(0.95, 330.0))
}

/// A run reset from values A to values B equals a run made afresh for B, bit for bit, its
/// work counted alike: the operating point remembered, the pivots kept, the meter's counts,
/// the junctions' last voltages and what the analyses recorded are forgotten.
#[test]
fn a_run_reset_to_other_values_equals_one_made_for_them() {
    let (circuit, analyses, options) = bjt_amp();
    let options = Options {
        record: Record::All,
        ..options
    };
    let elaborated = shared(&circuit);
    let (first, second) = two_values(&elaborated);
    let mut fresh =
        Simulator::from_elaborated(Arc::clone(&elaborated), second.clone()).expect("a run");
    let expected = run(&mut fresh, &analyses, &options);

    let mut reused =
        Simulator::from_elaborated(Arc::clone(&elaborated), first.clone()).expect("a run");
    let before = run(&mut reused, &analyses, &options);
    assert_ne!(before, expected, "the values make a difference");
    // A DC sweep holds a source away from its level; the reset releases it.
    let sweep: Analysis =
        serde_json::from_value(json!({"dc_sweep": {"source": "V1", "levels": [-0.1, 0.0, 0.1]}}))
            .expect("a sweep");
    run(&mut reused, &[sweep], &options);
    reused
        .reset(second)
        .expect("values of the circuit's numbers");
    assert_eq!(reused.operating_point_solves(), 0);
    assert_eq!(run(&mut reused, &analyses, &options), expected);
    // And back again: the first values run as they did.
    reused
        .reset(first)
        .expect("values of the circuit's numbers");
    assert_eq!(run(&mut reused, &analyses, &options), before);
    assert!(
        Arc::ptr_eq(reused.elaborated(), &elaborated),
        "no structure changed"
    );
}

/// A logic inverter in a ring with a resistor and a capacitor oscillates: a run reset
/// after its transient forgets the logic's events and traces, and the drivers' ramps.
#[test]
fn a_reset_forgets_the_logic() {
    let device = |id: &str, model: &str, parameters: Value, terminals: Value| {
        json!({"id": id, "of": {"device": model}, "parameters": parameters,
               "terminals": terminals})
    };
    let instances = vec![
        device(
            "VDD",
            "voltage_source",
            json!({"waveform": {"dc": 1.0}}),
            json!({"positive": "vdd", "negative": "0"}),
        ),
        device(
            "U1",
            "logic_cell",
            json!({
                "inputs": ["a"], "outputs": [{"name": "y", "table": "10"}],
                "timing": {"fixed": {"rise": 1e-6, "fall": 1e-6, "transition": 1e-12}},
                "electrical": {"input_capacitances": [0.0]}}),
            json!({"a": "c", "y": "out", "supply": "vdd", "ground": "0"}),
        ),
        device(
            "R1",
            "resistor",
            json!({"resistance": 1e3}),
            json!({"a": "out", "b": "c"}),
        ),
        device(
            "C1",
            "capacitor",
            json!({"capacitance": 1e-9}),
            json!({"a": "c", "b": "0"}),
        ),
        device(
            "V1",
            "voltage_source",
            json!({"waveform": {"piecewise": {"points": [[0, 0], [1e-9, 0], [2e-9, 1], [3e-9, 0]]}}}),
            json!({"positive": "kick", "negative": "0"}),
        ),
        device(
            "R2",
            "resistor",
            json!({"resistance": 1e9}),
            json!({"a": "kick", "b": "c"}),
        ),
    ];
    let document = json!({
        "format": "jolty-circuit", "version": 1, "top": "ring",
        "components": {"ring": {"ports": [{"name": "0", "role": "ground"}], "instances": instances}}
    });
    let circuit = Circuit::from_json(&document.to_string()).expect("a valid circuit");
    let elaborated = shared(&circuit);
    let analyses = [
        serde_json::from_value(json!({"transient": {"stop": 2e-5}})).expect("a transient"),
        Analysis::OperatingPoint,
    ];
    let options = Options::default();
    let resistance = |ohms: f64| {
        let overrides = Overrides::from([(
            "R1".to_owned(),
            BTreeMap::from([("resistance".to_owned(), ohms)]),
        )]);
        elaborated.values(&overrides, None).expect("values")
    };
    let mut fresh =
        Simulator::from_elaborated(Arc::clone(&elaborated), resistance(2e3)).expect("a run");
    let expected = run(&mut fresh, &analyses, &options);
    let mut reused =
        Simulator::from_elaborated(Arc::clone(&elaborated), resistance(1e3)).expect("a run");
    let before = run(&mut reused, &analyses, &options);
    assert_ne!(before, expected);
    reused.reset(resistance(2e3)).expect("values");
    assert_eq!(reused.excursions(), []);
    assert_eq!(run(&mut reused, &analyses, &options), expected);
}

/// A divider whose upper resistor is 2 Ω: a resistance below 1 Ω carries its current as an
/// unknown of its own.
fn divider() -> Circuit {
    let document = json!({
        "format": "jolty-circuit", "version": 1, "top": "divider",
        "components": {"divider": {"ports": [{"name": "0", "role": "ground"}], "instances": [
            {"id": "V1", "of": {"device": "voltage_source"},
             "parameters": {"waveform": {"dc": 1.0}},
             "terminals": {"positive": "in", "negative": "0"}},
            {"id": "R1", "of": {"device": "resistor"}, "parameters": {"resistance": 2.0},
             "terminals": {"a": "in", "b": "out"}},
            {"id": "R2", "of": {"device": "resistor"}, "parameters": {"resistance": 3.0},
             "terminals": {"a": "out", "b": "0"}}
        ]}}
    });
    Circuit::from_json(&document.to_string()).expect("a valid circuit")
}

fn upper(elaborated: &Elaborated, ohms: f64) -> Values {
    let overrides = Overrides::from([(
        "R1".to_owned(),
        BTreeMap::from([("resistance".to_owned(), ohms)]),
    )]);
    elaborated.values(&overrides, None).expect("values")
}

/// A resistor overridden across 1 Ω changes the circuit's structure: the values get a key
/// of their own, alike for every resistance below, and run on the circuit restructured for
/// them, as the circuit elaborated with the resistance does.
#[test]
fn a_resistor_overridden_across_one_ohm_changes_the_structure() {
    let circuit = divider();
    let elaborated = shared(&circuit);
    let above = elaborated
        .structure_of(&upper(&elaborated, 5.0))
        .expect("a structure");
    assert!(above.is_empty(), "{above:?}");
    let below = elaborated
        .structure_of(&upper(&elaborated, 0.5))
        .expect("a structure");
    assert_eq!(below.devices().collect::<Vec<_>>(), [1]);
    let lower = elaborated
        .structure_of(&upper(&elaborated, 0.25))
        .expect("a structure");
    assert_eq!(
        below, lower,
        "every resistance below 1 Ω shares a structure"
    );

    let values = upper(&elaborated, 0.5);
    let restructured = elaborated.restructured(&values).expect("a structure");
    let names: Vec<&str> = restructured
        .unknowns()
        .iter()
        .map(|unknown| unknown.name.as_str())
        .collect();
    assert!(names.contains(&"R1:current"), "{names:?}");
    assert!(
        restructured
            .structure_of(&values)
            .expect("a structure")
            .is_empty()
    );
    assert_eq!(restructured.nominal(), elaborated.nominal());

    let analyses = [Analysis::OperatingPoint];
    let options = Options::default();
    let overrides = Overrides::from([(
        "R1".to_owned(),
        BTreeMap::from([("resistance".to_owned(), 0.5)]),
    )]);
    let (direct, from_shared) = both_ways(&circuit, &overrides, 300.15, (&analyses, &options));
    assert_eq!(direct, from_shared);
    let mut simulator =
        Simulator::from_elaborated(Arc::clone(&elaborated), values.clone()).expect("a run");
    let out = simulator.unknown("out").expect("a node");
    let point = simulator.operating_point(&options).expect("a solution");
    // 1 V across 0.5 Ω then 3 Ω.
    assert!((point.values[out.index()] - 3.0 / 3.5).abs() < 1e-12);
    // A run reset back above 1 Ω takes the circuit's own structure again.
    simulator.reset(upper(&elaborated, 5.0)).expect("values");
    assert!(simulator.unknown("R1:current").is_none());
    let point = simulator.operating_point(&options).expect("a solution");
    assert!((point.values[out.index()] - 3.0 / 8.0).abs() < 1e-12);
    // And the restructured circuit runs its values without restructuring again.
    let on = Simulator::from_elaborated(Arc::clone(&restructured), values).expect("a run");
    assert!(Arc::ptr_eq(on.elaborated(), &restructured));
}

/// Values name their numbers by path and by name: every path the circuit has no device at,
/// every number a device does not write, and every number of a logic device are each a
/// problem, all of them reported.
#[test]
fn values_report_every_number_they_cannot_replace() {
    let document = json!({
        "format": "jolty-circuit", "version": 1, "top": "top",
        "components": {"top": {"ports": [{"name": "0", "role": "ground"}], "instances": [
            {"id": "V1", "of": {"device": "voltage_source"},
             "parameters": {"waveform": {"dc": 1.0}},
             "terminals": {"positive": "in", "negative": "0"}},
            {"id": "R1", "of": {"device": "resistor"}, "parameters": {"resistance": 2.0},
             "terminals": {"a": "in", "b": "0"}},
            {"id": "U1", "of": {"device": "logic_source"},
             "parameters": {"initial": "0", "changes": [[1e-9, "1"]], "transition": 1e-12},
             "terminals": {"output": "y"}}
        ]}}
    });
    let circuit = Circuit::from_json(&document.to_string()).expect("a valid circuit");
    let elaborated = shared(&circuit);
    let overrides = Overrides::from([
        (
            "R1".to_owned(),
            BTreeMap::from([
                ("resistance".to_owned(), 3.0),
                ("inductance".to_owned(), 1.0),
            ]),
        ),
        (
            "R9".to_owned(),
            BTreeMap::from([("resistance".to_owned(), 3.0)]),
        ),
        (
            "U1".to_owned(),
            BTreeMap::from([("transition".to_owned(), 1e-11)]),
        ),
        (
            "V1".to_owned(),
            BTreeMap::from([("waveform.sine.offset".to_owned(), 1.0)]),
        ),
    ]);
    let error = elaborated.values(&overrides, None).expect_err("problems");
    assert_eq!(
        error.to_string(),
        "instance `R1`: the override `inductance` names no number among the parameters\n\
         the circuit has no device at `R9`, whose parameters are replaced\n\
         instance `U1`: the override `transition` replaces a number of a logic device, which \
         only elaborating the circuit with it replaces\n\
         instance `V1`: the override `waveform.sine.offset` names no number among the \
         parameters"
    );
    // A number out of range is the problem of its device, when it is applied.
    let negative = upper(&elaborated, -1.0);
    let problem = Simulator::from_elaborated(Arc::clone(&elaborated), negative.clone())
        .expect_err("a problem");
    assert_eq!(
        problem.to_string(),
        "instance `R1`: resistance must be positive, not -1.0"
    );
    // A reset to it leaves the run as it was.
    let mut simulator =
        Simulator::from_elaborated(Arc::clone(&elaborated), upper(&elaborated, 4.0))
            .expect("a run");
    let options = Options::default();
    let before = bits(&simulator.run(&Analysis::OperatingPoint, &options));
    assert!(simulator.reset(negative).is_err());
    assert_eq!(
        bits(&simulator.run(&Analysis::OperatingPoint, &options)),
        before
    );
    // A parameter of another circuit is none of this one's.
    let (amplifier, _, _) = bjt_amp();
    let larger = shared(&amplifier);
    let last = larger.nominal().len() - 1;
    let mut far = Values::default();
    let mut beyond = None;
    for (path, name, _) in every_number(&amplifier) {
        if let Some(parameter) = larger.parameter(&path, &name)
            && parameter.index() == last
        {
            beyond = Some(parameter);
        }
    }
    far.set(beyond.expect("the last parameter"), 1.0);
    let problems = elaborated.structure_of(&far).expect_err("a problem");
    assert!(matches!(
        problems.problems(),
        [ElaborationProblem::NoParameter(index)] if *index == last
    ));
}

/// The table numbers every device's numbers in the order of the devices' paths, then of
/// the numbers' own, each at its nominal value.
#[test]
fn the_table_numbers_parameters_by_path_then_by_name() {
    let (circuit, _, _) = bjt_amp();
    let elaborated = shared(&circuit);
    let mut numbers = every_number(&circuit);
    numbers.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(numbers.len(), elaborated.nominal().len());
    for (index, (path, name, value)) in numbers.iter().enumerate() {
        let parameter = elaborated.parameter(path, name).expect("a parameter");
        assert_eq!(parameter.index(), index, "{path} {name}");
        assert_eq!(elaborated.nominal()[index].to_bits(), value.to_bits());
    }
    assert_eq!(elaborated.parameter("X1/Q1", "no_such_number"), None);
    assert_eq!(elaborated.parameter("X9", "resistance"), None);
}
