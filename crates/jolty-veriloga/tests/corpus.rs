//! The compiled kernels against their hand-written twins at every state the kernel corpus
//! visits: each circuit of `crates/jolty-engine/tests/data/kernels.json` runs every
//! analysis its tests run, its hand-written resistors and diodes recording each state
//! they are evaluated at, and both twins are evaluated again at each, to rounding.

#![cfg(test)]

mod common;

use std::any::TypeId;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;

use common::Worst;
use jolty_engine::{Analysis, Conditions, Options, Simulator, Substitute};
use jolty_ir::{Circuit, Parameters};
use jolty_models::testing::Bench;
use jolty_models::{
    Diode, Input, Interface, Kernel, Layout, Library, Local, NoiseSource, ParameterError, Real,
    Resistor, Stamp, Thermal,
};
use jolty_veriloga::kernels::{diode, resistor};
use serde::Deserialize;

/// A state a kernel was evaluated at: the kernel's name, its parameters as JSON, its
/// temperature, its local unknowns' values, and the analysis's time, whether time passes,
/// the sources' scale and gmin.
#[derive(Debug, Clone, PartialEq)]
struct State {
    kernel: &'static str,
    parameters: String,
    temperature: f64,
    x: Vec<f64>,
    time: Option<f64>,
    source_scale: f64,
    gmin: f64,
}

/// The states recorded, in the order of their evaluations.
static STATES: Mutex<Vec<State>> = Mutex::new(Vec::new());

/// The kernel `K`, recording each state it is evaluated at on plain numbers.
#[derive(Debug, Clone, Copy)]
struct StateRecorder<K>(K);

impl<K: Kernel> Kernel for StateRecorder<K> {
    const NAME: &'static str = K::NAME;
    type Parameters<R: Real> = K::Parameters<R>;
    /// The kernel's shape and its number of local unknowns.
    type Shape = (K::Shape, usize);
    /// The kernel's numbers, and its parameters and temperature, on plain numbers.
    type Prepared<R: Real> = (K::Prepared<R>, Option<(String, f64)>);
    type Memory = K::Memory;

    fn interface<R: Real>(parameters: &Self::Parameters<R>) -> Interface {
        K::interface(parameters)
    }

    fn prepare<R: Real>(
        parameters: &Self::Parameters<R>,
        temperature: R,
    ) -> Result<Self::Prepared<R>, ParameterError> {
        let plain = (TypeId::of::<R>() == TypeId::of::<f64>()).then(|| {
            let written = serde_json::to_string(parameters).expect("parameters write");
            (written, temperature.value())
        });
        Ok((K::prepare(parameters, temperature)?, plain))
    }

    fn layout<R: Real>(
        parameters: &Self::Parameters<R>,
        (prepared, _): &Self::Prepared<R>,
        layout: &mut Layout,
    ) -> Result<Self::Shape, ParameterError> {
        let shape = K::layout(parameters, prepared, layout)?;
        Ok((shape, layout.unknowns()))
    }

    fn memory<R: Real>((shape, _): &Self::Shape, (prepared, _): &Self::Prepared<R>) -> K::Memory {
        K::memory(shape, prepared)
    }

    fn evaluate<R: Real>(
        (shape, unknowns): &Self::Shape,
        (prepared, plain): &Self::Prepared<R>,
        memory: &mut K::Memory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        if let Some((parameters, temperature)) = plain {
            STATES.lock().expect("the states").push(State {
                kernel: K::NAME,
                parameters: parameters.clone(),
                temperature: *temperature,
                x: (0..*unknowns).map(|local| at.value(local)).collect(),
                time: at.transient.then_some(at.time),
                source_scale: at.source_scale,
                gmin: at.gmin,
            });
        }
        K::evaluate(shape, prepared, memory, at, stamp);
    }

    fn reheat(memory: &mut K::Memory) {
        K::reheat(memory);
    }

    fn noise<R: Real>(
        (shape, _): &Self::Shape,
        (prepared, _): &Self::Prepared<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        K::noise(shape, prepared, at, sources);
    }
}

/// A circuit of the corpus, what it is elaborated in, and the runs its tests make of it.
#[derive(Deserialize)]
struct CorpusCircuit {
    circuit: serde_json::Value,
    temperature: f64,
    overrides: BTreeMap<String, Parameters>,
    substitutes: BTreeMap<String, Substitute>,
    runs: Vec<(Analysis, Options)>,
}

/// Evaluates both twins, `H` by hand and `C` compiled, at each of `states`, all of one
/// instance's parameters and temperature, its steps not limited, and compares them, and
/// their noise at the first and the last.
fn replay<H: Kernel, C: Kernel>(states: &[&State], worst: &mut Worst) {
    let Some(first) = states.first() else { return };
    let mut hand = Bench::at::<H>(&first.parameters, first.temperature);
    let mut compiled = Bench::at::<C>(&first.parameters, first.temperature);
    for state in states {
        let mut input = Input::new(&state.x, state.time, state.gmin);
        input.source_scale = state.source_scale;
        input.limiting = false;
        let what = format!("{} {} at {:?}", H::NAME, state.parameters, state.x);
        if std::ptr::eq(*state, *first) || std::ptr::eq(*state, states[states.len() - 1]) {
            worst.noises(&what, &hand.noise(&state.x), &compiled.noise(&state.x));
        }
        worst.evaluations(
            &what,
            &state.x,
            &hand.evaluate_with(&input),
            &compiled.evaluate_with(&input),
        );
    }
}

/// At every state the corpus's runs evaluate a resistor or a diode at, the compiled
/// kernel is its hand-written twin, to rounding: on 16 November 2026, 1,423,039 states of
/// resistors and 823,769 of diodes, of 80 instances, all within 3.1 ε of each other.
#[test]
fn the_compiled_kernels_are_their_twins_across_the_corpus() {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../jolty-engine/tests/data/kernels.json");
    let text = std::fs::read_to_string(path).expect("the corpus");
    let corpus: Vec<CorpusCircuit> = serde_json::from_str(&text).expect("the corpus reads");
    let mut library = Library::standard();
    library.add(Thermal::<StateRecorder<Resistor>>::new());
    library.add(Thermal::<StateRecorder<Diode>>::new());
    for entry in &corpus {
        let circuit = Circuit::from_json(&entry.circuit.to_string()).expect("a circuit");
        let conditions = Conditions {
            temperature: entry.temperature,
            overrides: entry.overrides.clone(),
            substitutes: entry.substitutes.clone(),
        };
        let mut simulator =
            Simulator::new(&circuit, &library, &conditions).expect("the circuit elaborates");
        for (analysis, options) in &entry.runs {
            // An analysis that fails, as one a test expects to, was evaluated all the same.
            let _ = simulator.run(analysis, options);
        }
    }
    let states = std::mem::take(&mut *STATES.lock().expect("the states"));
    let mut instances: BTreeMap<(&str, &str, u64), Vec<&State>> = BTreeMap::new();
    for state in &states {
        let key = (
            state.kernel,
            state.parameters.as_str(),
            state.temperature.to_bits(),
        );
        instances.entry(key).or_default().push(state);
    }
    let mut worst = Worst::default();
    for ((kernel, _, _), states) in &instances {
        match *kernel {
            "resistor" => replay::<Resistor, resistor::Resistor>(states, &mut worst),
            _ => replay::<Diode, diode::Diode>(states, &mut worst),
        }
    }
    let count = |name: &str| states.iter().filter(|state| state.kernel == name).count();
    assert!(
        count("resistor") > 10_000 && count("diode") > 1_000,
        "{} resistor and {} diode states",
        count("resistor"),
        count("diode")
    );
}
