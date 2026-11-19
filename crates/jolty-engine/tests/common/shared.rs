//! Shared bitwise comparison, benchmark loading, and random driven network helpers for
//! `determinism.rs` and `elaborated.rs`.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use jolty_engine::{
    Analysis, AnalysisResult, Derivatives, Elaborated, OperatingPointResult, Options,
    SimulationError, Structure, Values,
};
use jolty_ir::Circuit;
use jolty_models::Library;
use jolty_numerics::Complex64;
use proptest::prelude::*;
use serde::Deserialize;
use serde_json::json;

use crate::common::{circuit, device, resistor, two_terminal, voltage_source};
use crate::network::{connected_resistors, node};

/// Every number of an analysis's outcome as raw IEEE-754 bits, paired with its debug text,
/// which holds its counts of work and its messages.
pub(crate) type Bits = (Vec<u64>, String);

/// Appends the raw IEEE-754 bits of every value in `values` into `into`.
pub(crate) fn reals(values: &[f64], into: &mut Vec<u64>) {
    into.extend(values.iter().map(|value| value.to_bits()));
}

fn complexes(values: &[Complex64], into: &mut Vec<u64>) {
    for value in values {
        into.extend([value.re.to_bits(), value.im.to_bits()]);
    }
}

fn point(point: &OperatingPointResult, into: &mut Vec<u64>) {
    reals(&point.values, into);
    let recorded = &point.recorded;
    for values in [
        &recorded.currents,
        &recorded.charges,
        &recorded.conductances,
        &recorded.capacitances,
        &recorded.variables,
    ] {
        reals(values, into);
    }
}

fn numbers_of(outcome: &AnalysisResult, numbers: &mut Vec<u64>) {
    match outcome {
        AnalysisResult::OperatingPoint(found) => point(found, numbers),
        AnalysisResult::OperatingPoints(points) => {
            for found in points {
                point(found, numbers);
            }
        }
        AnalysisResult::Ac(response) => {
            point(&response.operating_point, numbers);
            reals(&response.frequencies, numbers);
            for row in response.values.iter().chain(&response.currents) {
                complexes(row, numbers);
            }
        }
        AnalysisResult::Noise(spectrum) => {
            point(&spectrum.operating_point, numbers);
            reals(&spectrum.densities, numbers);
            for contribution in &spectrum.contributions {
                reals(&contribution.densities, numbers);
            }
            if let Some(input) = &spectrum.input {
                complexes(&input.gains, numbers);
            }
        }
        AnalysisResult::Stability(gain) => {
            point(&gain.operating_point, numbers);
            complexes(&gain.gains, numbers);
        }
        AnalysisResult::Transient(waveforms) => {
            reals(&waveforms.times, numbers);
            for row in &waveforms.values {
                reals(row, numbers);
            }
            let recorded = &waveforms.recorded;
            for values in [
                &recorded.currents,
                &recorded.charges,
                &recorded.conductances,
                &recorded.capacitances,
                &recorded.variables,
            ] {
                reals(values, numbers);
            }
            for trace in &waveforms.logic {
                numbers.extend(trace.iter().map(|(time, _)| time.to_bits()));
            }
            reals(&waveforms.statistics.charge_leak.signed, numbers);
            reals(&waveforms.statistics.charge_leak.absolute, numbers);
        }
        AnalysisResult::DcSweep(sweep) => {
            reals(&sweep.swept, numbers);
            for row in &sweep.values {
                reals(row, numbers);
            }
            let recorded = &sweep.recorded;
            for values in [
                &recorded.currents,
                &recorded.charges,
                &recorded.conductances,
                &recorded.capacitances,
                &recorded.variables,
            ] {
                reals(values, numbers);
            }
        }
        AnalysisResult::Sensitivity(found) => {
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

/// Extracts every number of an analysis outcome as raw IEEE-754 bits alongside its debug
/// representation.
pub(crate) fn bits(outcome: &Result<AnalysisResult, SimulationError>) -> Bits {
    let mut numbers = Vec::new();
    if let Ok(found) = outcome {
        numbers_of(found, &mut numbers);
    }
    (numbers, format!("{outcome:?}"))
}

/// Elaborates `circuit` once with default structure for sharing across runs.
pub(crate) fn shared(circuit: &Circuit) -> Arc<Elaborated> {
    Elaborated::new(circuit, &Library::standard(), &Structure::default())
        .expect("the circuit elaborates")
}

/// Benchmark plan loaded from `benchmarks/speed/plans/<name>.json`.
#[derive(Deserialize)]
struct Plan {
    circuit: String,
    analyses: Vec<(String, Analysis)>,
    options: Options,
}

/// Loads the circuit, analyses, and solver options for speed benchmark `name`.
pub(crate) fn benchmark(name: &str) -> (Circuit, Vec<Analysis>, Options) {
    let speed = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/speed");
    let plan: Plan = serde_json::from_str(
        &fs::read_to_string(speed.join(format!("plans/{name}.json"))).expect("the plan reads"),
    )
    .expect("a plan");
    let text = fs::read_to_string(speed.join(format!("circuits/{}.json", plan.circuit)))
        .expect("the circuit reads");
    let circuit = Circuit::from_json(&text).expect("a circuit");
    let analyses = plan
        .analyses
        .into_iter()
        .map(|(_, analysis)| analysis)
        .collect();
    (circuit, analyses, plan.options)
}

/// A grounded network of resistors, capacitors, optional inductors and diodes, driven by a
/// pulse and by current sources.
#[derive(Debug, Clone)]
pub(crate) struct Network {
    pub(crate) resistors: Vec<(usize, usize, f64)>,
    pub(crate) capacitors: Vec<(usize, f64)>,
    pub(crate) inductors: Vec<(usize, usize, f64)>,
    pub(crate) diodes: Vec<usize>,
    pub(crate) sources: Vec<(usize, f64)>,
}

/// Builds a proptest strategy for connected driven networks on up to six nodes.
///
/// Each network pairs a 0.1 Ω to 100 kΩ resistor spanning tree with optional extra
/// resistors, per-node capacitors, a pulse source at node 1, DC current sources, and
/// optional inductors and diodes when `with_nonlinear` is true.
pub(crate) fn network(with_nonlinear: bool) -> impl Strategy<Value = Network> {
    let count = if with_nonlinear { 0..3 } else { 0..1 };
    (2usize..7).prop_flat_map(move |nodes| {
        let resistors = connected_resistors(nodes, -1.0..5.0);
        let capacitors = proptest::collection::vec(-12.0..-9.0f64, nodes - 1);
        let inductors =
            proptest::collection::vec((0..nodes, 0..nodes, -7.0..-3.0f64), count.clone());
        let diodes = proptest::collection::vec(1..nodes, count.clone());
        let sources = proptest::collection::vec((1..nodes, -1e-3..1e-3f64), 0..3);
        (resistors, capacitors, inductors, diodes, sources).prop_map(
            |(resistors, capacitors, inductors, diodes, sources)| Network {
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
            },
        )
    })
}

impl Network {
    /// Builds the network as a [`Circuit`] driven at node 1 through a 50 Ω source resistor.
    pub(crate) fn circuit(&self) -> Circuit {
        let mut instances = vec![
            voltage_source(
                "V1",
                "drive",
                "0",
                json!({"pulse": {"initial": 0.0, "pulsed": 1.0, "delay": 1e-9,
                    "rise": 1e-9, "fall": 1e-9, "width": 5e-9, "period": 20e-9}}),
            ),
            resistor("RD", 50.0, "drive", &node(1)),
        ];
        for (index, &(a, b, resistance)) in self.resistors.iter().enumerate() {
            instances.push(resistor(
                &format!("R{index}"),
                resistance,
                &node(a),
                &node(b),
            ));
        }
        for &(at, capacitance) in &self.capacitors {
            instances.push(two_terminal(
                &format!("C{at}"),
                "capacitor",
                "capacitance",
                capacitance,
                &node(at),
                "0",
            ));
        }
        for (index, &(a, b, inductance)) in self.inductors.iter().enumerate() {
            instances.push(two_terminal(
                &format!("L{index}"),
                "inductor",
                "inductance",
                inductance,
                &node(a),
                &node(b),
            ));
        }
        for (index, &at) in self.diodes.iter().enumerate() {
            instances.push(device(
                &format!("D{index}"),
                "diode",
                json!({"saturation_current": 1e-14, "junction_capacitance": 1e-12}),
                &[("anode", &node(at)), ("cathode", "0")],
            ));
        }
        for (index, &(at, current)) in self.sources.iter().enumerate() {
            instances.push(device(
                &format!("I{index}"),
                "current_source",
                json!({"waveform": {"dc": current}}),
                &[("positive", "0"), ("negative", &node(at))],
            ));
        }
        circuit(instances)
    }
}

/// Builds parameter [`Values`] for `elaborated` with `R1.resistance` set to `ohms`.
pub(crate) fn upper(elaborated: &Elaborated, ohms: f64) -> Values {
    let mut values = Values::default();
    values.set(
        elaborated
            .parameter("R1", "resistance")
            .expect("a resistance"),
        ohms,
    );
    values
}
