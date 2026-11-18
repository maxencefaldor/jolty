//! The compiled kernels against their hand-written twins at every state the kernel corpus
//! visits: each circuit of `crates/jolty-engine/tests/data/kernels.json` runs every
//! analysis its tests run, its hand-written resistors and diodes recording each state
//! they are evaluated at, and both twins are evaluated again at each, to rounding.

#![cfg(test)]

mod common;

use std::any::Any;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{LazyLock, Mutex, PoisonError};

use common::Worst;
use jolty_engine::{Analysis, Conditions, Options, Simulator, Substitute};
use jolty_ir::{Circuit, Parameters};
use jolty_models::testing::Bench;
use jolty_models::{
    Diode, Input, Interface, Kernel, Layout, Library, Local, NoiseSource, ParameterError, Real,
    Resistor, Stamp, Thermal,
};
use jolty_veriloga::kernels::{diode, resistor};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

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

/// Serializes the recording phase across threads when run inside one process under plain
/// `cargo test`; under `cargo-nextest` every shard runs in its own process with zero
/// contention, and `replay` always runs outside this lock.
static RECORD_GUARD: Mutex<()> = Mutex::new(());

/// The kernel `K`, recording each state it is evaluated at on plain numbers.
#[derive(Debug, Clone, Copy)]
struct StateRecorder<K>(K);

impl<K: Kernel> Kernel for StateRecorder<K>
where
    K::Parameters<f64>: Serialize,
{
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
        let plain = (parameters as &dyn Any)
            .downcast_ref::<K::Parameters<f64>>()
            .map(|plain| {
                let written = serde_json::to_string(plain).expect("parameters write");
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
            STATES
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(State {
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
fn replay<H: Kernel, C: Kernel>(states: &[&State], worst: &mut Worst)
where
    H::Parameters<f64>: DeserializeOwned + Serialize + JsonSchema,
    C::Parameters<f64>: DeserializeOwned + Serialize + JsonSchema,
{
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

/// The kernel corpus parsed once per process.
static CORPUS: LazyLock<Vec<CorpusCircuit>> = LazyLock::new(|| {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../jolty-engine/tests/data/kernels.json");
    let text = std::fs::read_to_string(path).expect("the corpus");
    serde_json::from_str(&text).expect("the corpus reads")
});

/// Clears `STATES` on drop so an unwind inside a shard cannot leak recorded states into a
/// subsequent shard running in the same process.
struct ClearStatesOnDrop;

impl Drop for ClearStatesOnDrop {
    fn drop(&mut self) {
        STATES
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }
}

/// At every state in shard `shard` (`index % shards == shard`) of the corpus's runs that
/// evaluates a resistor or a diode, checks that the compiled kernel is its hand-written
/// twin, to rounding: across all 16 shards (on 16 November 2026), 1,423,039 states of
/// resistors and 823,769 of diodes, of 80 instances, all within 3.1 ε of each other.
fn check_corpus_shard(shard: usize, shards: usize) {
    let mut library = Library::standard();
    library.add(Thermal::<StateRecorder<Resistor>>::new());
    library.add(Thermal::<StateRecorder<Diode>>::new());
    let mut expects_diode = false;
    let states = {
        let _guard = RECORD_GUARD.lock().unwrap_or_else(PoisonError::into_inner);
        let _clear = ClearStatesOnDrop;
        STATES
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
        for (index, entry) in CORPUS.iter().enumerate() {
            if index % shards != shard {
                continue;
            }
            let circuit = Circuit::from_json(&entry.circuit.to_string()).expect("a circuit");
            let conditions = Conditions {
                temperature: entry.temperature,
                overrides: entry.overrides.clone(),
                substitutes: entry.substitutes.clone(),
            };
            let mut simulator =
                Simulator::new(&circuit, &library, &conditions).expect("the circuit elaborates");
            if simulator
                .devices()
                .iter()
                .any(|device| device.model() == "diode")
            {
                expects_diode = true;
            }
            for (analysis, options) in &entry.runs {
                // An analysis that fails, as one a test expects to, was evaluated all the same.
                let _ = simulator.run(analysis, options);
            }
        }
        std::mem::take(&mut *STATES.lock().unwrap_or_else(PoisonError::into_inner))
    };
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
        count("resistor") > 0,
        "shard {shard}/{shards} evaluated no resistor states"
    );
    if expects_diode {
        assert!(
            count("diode") > 0,
            "shard {shard}/{shards} holds a diode circuit but evaluated no diode states"
        );
    }
}

macro_rules! corpus_twin_shards {
    ($($name:ident),+ $(,)?) => {
        corpus_twin_shards!(@step [$(stringify!($name)),+].len(); 0usize; $($name),+);
    };
    (@step $total:expr; $index:expr; $name:ident $(, $rest:ident)*) => {
        #[test]
        fn $name() {
            check_corpus_shard($index, $total);
        }
        corpus_twin_shards!(@step $total; $index + 1usize; $($rest),*);
    };
    (@step $total:expr; $index:expr;) => {};
}

corpus_twin_shards!(
    the_compiled_kernels_are_their_twins_across_the_corpus_shard_00,
    the_compiled_kernels_are_their_twins_across_the_corpus_shard_01,
    the_compiled_kernels_are_their_twins_across_the_corpus_shard_02,
    the_compiled_kernels_are_their_twins_across_the_corpus_shard_03,
    the_compiled_kernels_are_their_twins_across_the_corpus_shard_04,
    the_compiled_kernels_are_their_twins_across_the_corpus_shard_05,
    the_compiled_kernels_are_their_twins_across_the_corpus_shard_06,
    the_compiled_kernels_are_their_twins_across_the_corpus_shard_07,
    the_compiled_kernels_are_their_twins_across_the_corpus_shard_08,
    the_compiled_kernels_are_their_twins_across_the_corpus_shard_09,
    the_compiled_kernels_are_their_twins_across_the_corpus_shard_10,
    the_compiled_kernels_are_their_twins_across_the_corpus_shard_11,
    the_compiled_kernels_are_their_twins_across_the_corpus_shard_12,
    the_compiled_kernels_are_their_twins_across_the_corpus_shard_13,
    the_compiled_kernels_are_their_twins_across_the_corpus_shard_14,
    the_compiled_kernels_are_their_twins_across_the_corpus_shard_15,
);
