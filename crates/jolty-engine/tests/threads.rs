//! Thread-pool bounding and bitwise reproducibility under [`Parallelism`].
//!
//! Each run limits parallel work to at most `max_threads` using a dedicated pool when the
//! caller pool is wider, producing results bitwise identical to single-threaded execution.
#![cfg(test)]

use std::collections::BTreeSet;

use jolty_engine::{
    Analysis, AnalysisResult, Conditions, Options, Parallelism, Record, RecordedSeries,
    RecordedValues, Simulator, TransientResult,
};
use jolty_ir::Circuit;
use jolty_models::{
    Branch, BranchKind, Differentiable, Interface, Kernel, Layout, Library, Local, ParameterError,
    Real, Stamp, ThermalResistanceParameters,
};
use rayon::ThreadPoolBuilder;
use serde_json::json;

/// Number of spy resistors in the ladder, each dispatched as an individual task.
const SPIES: usize = 64;

/// Resistor test kernel that records the evaluating thread index and pool width as its two
/// variables, using a single scalar resistance parameter in ohms.
struct Spy;

impl Kernel for Spy {
    const NAME: &'static str = "spy";
    type Parameters<R: Real> = ThermalResistanceParameters<R>;
    /// Local matrix entry indices for `(a, a)`, `(a, b)`, `(b, a)`, and `(b, b)`.
    type Shape = [usize; 4];
    /// Prepared conductance value.
    type Prepared<R: Real> = R;
    /// Pool thread index and total pool thread count from the most recent evaluation.
    type Memory = Option<(usize, usize)>;

    fn interface<R: Real>(_: &ThermalResistanceParameters<R>) -> Interface {
        Interface::new(&["a", "b"], &[Branch::new(0, 1, BranchKind::Conductive)])
    }

    fn prepare<R: Real>(
        parameters: &ThermalResistanceParameters<R>,
        _: R,
    ) -> Result<R, ParameterError> {
        Ok(R::constant(1.0) / parameters.resistance)
    }

    fn layout<R: Real>(
        _: &ThermalResistanceParameters<R>,
        _: &R,
        layout: &mut Layout,
    ) -> Result<[usize; 4], ParameterError> {
        Ok([
            layout.entry(0, 0),
            layout.entry(0, 1),
            layout.entry(1, 0),
            layout.entry(1, 1),
        ])
    }

    fn evaluate<R: Real>(
        entries: &[usize; 4],
        &conductance: &R,
        memory: &mut Option<(usize, usize)>,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let current = conductance * (at.value(0) - at.value(1));
        stamp.add_residual(0, current);
        stamp.add_residual(1, -current);
        for (&entry, sign) in entries.iter().zip([1.0, -1.0, -1.0, 1.0]) {
            stamp.add_conductance(entry, conductance * sign);
        }
        *memory = rayon::current_thread_index().map(|index| (index, rayon::current_num_threads()));
    }

    fn variable_names() -> &'static [&'static str] {
        &["thread", "threads"]
    }

    fn variables(memory: &Option<(usize, usize)>, values: &mut [f64]) {
        let number = |count: usize| f64::from(u32::try_from(count).unwrap_or(u32::MAX));
        let (thread, threads) = memory.map_or((f64::NAN, f64::NAN), |(thread, threads)| {
            (number(thread), number(threads))
        });
        for (value, found) in values.iter_mut().zip([thread, threads]) {
            *value = found;
        }
    }
}

/// Builds a nonlinear ladder of [`SPIES`] spy resistors with a midpoint diode and capacitor
/// driven by a 1 kHz sine wave.
fn ladder() -> Circuit {
    let node = |rung: usize| match rung {
        0 => "in".to_owned(),
        SPIES => "0".to_owned(),
        _ => format!("n{rung}"),
    };
    let mut instances = vec![json!(
        {"name": "V1", "of": {"model": "voltage_source"},
         "parameters": {"waveform": {"sine": {"offset": 1.0, "amplitude": 1.0,
                                              "frequency": 1e3}}},
         "terminals": {"positive": "in", "negative": "0"}}
    )];
    for rung in 0..SPIES {
        instances.push(json!(
            {"name": format!("S{rung}"), "of": {"model": "spy"},
             "parameters": {"resistance": 100.0},
             "terminals": {"a": node(rung), "b": node(rung + 1)}}
        ));
    }
    let middle = node(SPIES / 2);
    instances.push(json!(
        {"name": "D1", "of": {"model": "diode"}, "parameters": {},
         "terminals": {"anode": middle, "cathode": "0"}}
    ));
    instances.push(json!(
        {"name": "C1", "of": {"model": "capacitor"}, "parameters": {"capacitance": 1e-6},
         "terminals": {"a": middle, "b": "0"}}
    ));
    let document = json!({
        "format": "jolty-circuit", "version": 5, "top": "ladder",
        "components": {"ladder": {"ports": [{"name": "0", "role": "ground"}],
                                  "instances": instances}}
    });
    Circuit::from_json(&document.to_string()).expect("a valid circuit")
}

/// Runs the ladder operating point and a 1 ms transient under `options`, returning debug
/// snapshots without recorded data plus the `(thread_index, pool_size)` pairs seen by spies.
fn run(options: &Options) -> (Vec<String>, BTreeSet<(u64, u64)>) {
    let mut library = Library::standard();
    library.add(Differentiable::<Spy>::new());
    let mut simulator =
        Simulator::new(&ladder(), &library, &Conditions::default()).expect("the ladder");
    let options = Options {
        record: Record::Instances((0..SPIES).map(|rung| format!("S{rung}")).collect()),
        ..options.clone()
    };
    let transient: Analysis =
        serde_json::from_value(json!({"transient": {"stop": 1e-3, "max_step": 2e-5}}))
            .expect("a transient");
    let mut threads = BTreeSet::new();
    let mut spied = |variables: &[f64]| {
        for pair in variables.chunks(2) {
            // Each is a count below a thousand, which a double holds exactly.
            #[expect(clippy::cast_possible_truncation, reason = "a count, held exactly")]
            #[expect(clippy::cast_sign_loss, reason = "a count, held exactly")]
            threads.insert((pair[0] as u64, pair[1] as u64));
        }
    };
    let found = [Analysis::OperatingPoint, transient]
        .iter()
        .map(|analysis| {
            match simulator
                .run(analysis, &options)
                .expect("the analysis runs")
            {
                AnalysisResult::OperatingPoint(mut point) => {
                    spied(&point.recorded.variables);
                    point.recorded = RecordedValues::default();
                    format!("{point:?}")
                }
                AnalysisResult::Transient(waveforms) => {
                    spied(&waveforms.recorded.variables);
                    let waveforms = TransientResult {
                        recorded: RecordedSeries::default(),
                        ..waveforms
                    };
                    format!("{waveforms:?}")
                }
                other => panic!("no other analysis runs here: {other:?}"),
            }
        })
        .collect();
    (found, threads)
}

/// Verifies that `max_threads` caps parallel evaluation to a dedicated sub-pool when the
/// caller pool is wider while reusing narrower caller pools.
///
/// Parallel results match single-threaded execution bit for bit.
#[test]
fn a_run_spreads_its_work_over_its_threads_alone() {
    let (serial, alone) = ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("a pool")
        .install(|| run(&Options::default()));
    assert_eq!(alone, BTreeSet::from([(0, 1)]));
    let capped = Options {
        parallelism: Parallelism {
            max_threads: Some(2),
            min_solves_per_task: Some(1),
            min_instances_per_task: Some(1),
            min_unknowns_per_task: Some(1),
            ..Parallelism::default()
        },
        ..Options::default()
    };
    for threads in [3, 8] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a pool");
        let (found, spied) = pool.install(|| run(&capped));
        assert!(
            spied
                .iter()
                .all(|&(thread, threads)| thread < 2 && threads == 2),
            "on a pool of {threads}, the spies were evaluated on {spied:?}"
        );
        assert!(found == serial, "on a pool of {threads}");
    }
    // A run capped above its caller's pool runs on that pool.
    let above = Options {
        parallelism: Parallelism {
            max_threads: Some(8),
            ..capped.parallelism
        },
        ..capped
    };
    let pool = ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .expect("a pool");
    let (found, spied) = pool.install(|| run(&above));
    assert!(
        spied
            .iter()
            .all(|&(thread, threads)| thread < 3 && threads == 3),
        "capped at 8 on a pool of 3, the spies were evaluated on {spied:?}"
    );
    assert!(found == serial, "capped at 8 on a pool of 3");
}
