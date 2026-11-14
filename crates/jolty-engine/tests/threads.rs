//! A run spreads its own work over at most its parallelism's threads, on a pool of exactly
//! that many of its own when its caller's has more, and finds the bits it finds on one.
#![cfg(test)]

use std::collections::BTreeSet;

use jolty_engine::{
    Analysis, Conditions, Observation, Observations, Options, Outcome, Parallelism, Record,
    Simulator, Waveforms,
};
use jolty_ir::Circuit;
use jolty_models::{
    Branch, BranchKind, Interface, Kernel, Layout, Library, Local, Native, ParameterError, Real,
    Stamp, ThermalResistanceParameters,
};
use rayon::ThreadPoolBuilder;
use serde_json::json;

/// The spies in the ladder, each a task of its own.
const SPIES: usize = 64;

/// A resistor that remembers the thread that evaluated it last: its index in its pool and
/// the pool's threads, as its two variables. Its parameters are a thermal resistance's, a
/// resistance alone, here in ohms.
struct Spy;

impl Kernel for Spy {
    const NAME: &'static str = "spy";
    type Parameters<R: Real> = ThermalResistanceParameters<R>;
    /// Its local entries, `(a, a)`, `(a, b)`, `(b, a)` and `(b, b)`.
    type Shape = [usize; 4];
    /// Its conductance.
    type Prepared<R: Real> = R;
    /// The index of the thread that evaluated it last in its pool, and the pool's threads.
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

/// A sine into a ladder of [`SPIES`] spies to the ground, a diode and a capacitor across
/// its middle rung: a nonlinear circuit whose spies are each a task of their own.
fn ladder() -> Circuit {
    let node = |rung: usize| match rung {
        0 => "in".to_owned(),
        SPIES => "0".to_owned(),
        _ => format!("n{rung}"),
    };
    let mut instances = vec![json!(
        {"id": "V1", "of": {"device": "voltage_source"},
         "parameters": {"waveform": {"sine": {"offset": 1.0, "amplitude": 1.0,
                                              "frequency": 1e3}}},
         "terminals": {"positive": "in", "negative": "0"}}
    )];
    for rung in 0..SPIES {
        instances.push(json!(
            {"id": format!("S{rung}"), "of": {"device": "spy"},
             "parameters": {"resistance": 100.0},
             "terminals": {"a": node(rung), "b": node(rung + 1)}}
        ));
    }
    let middle = node(SPIES / 2);
    instances.push(json!(
        {"id": "D1", "of": {"device": "diode"}, "parameters": {},
         "terminals": {"anode": middle, "cathode": "0"}}
    ));
    instances.push(json!(
        {"id": "C1", "of": {"device": "capacitor"}, "parameters": {"capacitance": 1e-6},
         "terminals": {"a": middle, "b": "0"}}
    ));
    let document = json!({
        "format": "jolty-circuit", "version": 1, "top": "ladder",
        "components": {"ladder": {"ports": [{"name": "0", "role": "ground"}],
                                  "instances": instances}}
    });
    Circuit::from_json(&document.to_string()).expect("a valid circuit")
}

/// The ladder's operating point and a millisecond of its transient, with `options`,
/// recording the spies: what each found, without what it recorded, as its debug text, and
/// the threads the spies recorded, as their index in their pool and the pool's threads.
fn run(options: &Options) -> (Vec<String>, BTreeSet<(u64, u64)>) {
    let mut library = Library::standard();
    library.add(Native::<Spy>::new());
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
                Outcome::OperatingPoint(mut point) => {
                    spied(&point.observation.variables);
                    point.observation = Observation::default();
                    format!("{point:?}")
                }
                Outcome::Transient(waveforms) => {
                    spied(&waveforms.observations.variables);
                    let waveforms = Waveforms {
                        observations: Observations::default(),
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

/// A run capped at two threads, spreading every kind of its work, however small, runs it
/// on a pool of two threads of its own when its caller's has three or eight: every spy,
/// each a task of its own, is evaluated on one of them. Capped at eight, it runs on its
/// caller's pool of three. It finds what it finds on one thread, spreading nothing, bit for
/// bit. Before, its instances' tasks ran on every thread of its caller's pool.
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
            threads: Some(2),
            min_solves_per_task: Some(1),
            min_instances_per_task: Some(1),
            min_unknowns_inside: Some(1),
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
            threads: Some(8),
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
