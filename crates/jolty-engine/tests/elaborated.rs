//! A circuit elaborated once and run many times: a run of the shared circuit at some values
//! equals the circuit elaborated with those values, a run reset to other values equals one
//! made afresh for them, and values that change a device's unknowns run on the circuit
//! restructured for them. Every comparison is of every number's bits.
#![cfg(test)]

#[path = "common/capacitor.rs"]
mod capacitor;
mod common;
#[path = "common/dc.rs"]
mod dc;
#[path = "common/network.rs"]
mod network;
#[path = "common/shared.rs"]
mod shared;

use std::collections::BTreeMap;
use std::sync::Arc;

use capacitor::capacitor;
use common::{circuit, device, resistor, voltage_source};
use dc::voltage_dc;
use jolty_engine::{
    Analysis, Conditions, Elaborated, ElaborationError, ElaborationProblem, Options, Record,
    Simulator, Values,
};
use jolty_ir::{Circuit, Parameters};
use jolty_models::Library;
use proptest::prelude::*;
use serde_json::json;
use shared::{Bits, benchmark, bits, network, shared, upper};

/// Numbers that replace devices' own, by the device's path, then by the number's path.
type Overrides = BTreeMap<String, BTreeMap<String, f64>>;

/// The bits of each analysis of `analyses`, run in turn on `simulator`.
fn run(simulator: &mut Simulator, analyses: &[Analysis], options: &Options) -> Vec<Bits> {
    analyses
        .iter()
        .map(|analysis| bits(&simulator.run(analysis, options)))
        .collect()
}

/// The circuit and the analyses of the benchmark `bjt_amp`: two 2N3904s and three 2N3906s,
/// whose junctions limit their steps from their last voltages, over 200 µs of a sine, at
/// an operating point and at 71 frequencies. The transient comes first, so that it solves
/// the operating point it starts from, and counts the factorizations of a solver that has
/// kept no pivots.
fn bjt_amp() -> (Circuit, Vec<Analysis>, Options) {
    let (circuit, mut analyses, options) = benchmark("bjt_amp");
    analyses.rotate_right(1);
    assert!(matches!(analyses[0], Analysis::Transient(_)));
    (circuit, analyses, options)
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

/// The circuit elaborated with `overrides` at `temperature`, as `Simulator::new` takes it.
fn elaborated_with(
    circuit: &Circuit,
    overrides: &Overrides,
    temperature: f64,
) -> Result<Simulator, ElaborationError> {
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
            let problems = |result: Result<Simulator, ElaborationError>| {
                result.err().map(|error| error.to_string())
            };
            let (direct, from_shared) = (problems(direct), problems(from_shared));
            assert_eq!(direct, from_shared, "the problems of the values differ");
            (Vec::new(), Vec::new())
        }
    }
}

/// The numbers of a network's pulse, `V1`.
const PULSE_NUMBERS: usize = 7;

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
        (network, factors) in network(false).prop_flat_map(|network| {
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
        serde_json::from_value(json!({"dc_sweep": {"source": "V1", "values": [-0.1, 0.0, 0.1]}}))
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
    let circuit = circuit([
        voltage_dc("VDD", "vdd", 1.0),
        device(
            "U1",
            "logic_cell",
            json!({
                "inputs": ["a"], "outputs": [{"name": "y", "table": "10"}],
                "timing": {"fixed": {"rise": 1e-6, "fall": 1e-6, "transition": 1e-12}},
                "electrical": {"input_capacitances": [0.0]}}),
            &[("a", "c"), ("y", "out"), ("supply", "vdd"), ("ground", "0")],
        ),
        resistor("R1", 1e3, "out", "c"),
        capacitor("C1", 1e-9, "c", "0"),
        voltage_source(
            "V1",
            "kick",
            "0",
            json!({"piecewise": {"points": [[0, 0], [1e-9, 0], [2e-9, 1], [3e-9, 0]]}}),
        ),
        resistor("R2", 1e9, "kick", "c"),
    ]);
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
    circuit([
        voltage_dc("V1", "in", 1.0),
        resistor("R1", 2.0, "in", "out"),
        resistor("R2", 3.0, "out", "0"),
    ])
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
    let circuit = circuit([
        voltage_dc("V1", "in", 1.0),
        resistor("R1", 2.0, "in", "0"),
        device(
            "U1",
            "logic_source",
            json!({"initial": "0", "changes": [[1e-9, "1"]], "transition": 1e-12}),
            &[("output", "y")],
        ),
    ]);
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
