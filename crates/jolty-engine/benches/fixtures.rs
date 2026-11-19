//! Engine benchmarks covering every fixture in `benchmarks/speed/`.
//!
//! For fixtures without Monte Carlo analysis, `<fixture>/setup` times JSON parsing and
//! simulator construction, `<fixture>/shared` times simulator creation from a shared
//! [`Elaborated`] circuit, and `<fixture>/run` times setup plus every analysis in the plan.
//! For Monte Carlo fixtures, `<fixture>/batch/<threads>` times the first [`SAMPLES`]
//! samples as one batch across 1, 2, 4, 8, and full-core thread pools.
//! `scripts/bench` runs the suite fixtures and summarizes criterion measurements; pass a
//! filter after `--` to run others, such as
//! `cargo bench -p jolty-engine --bench fixtures -- inverter_chain_5000/`.

use std::error::Error;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, SamplingMode, criterion_group, criterion_main};
use jolty_engine::batch::{Batch, Sink};
use jolty_engine::variants::{MonteCarlo, toleranced};
use jolty_engine::{
    Analysis, AnalysisResult, Conditions, Elaborated, Options, SimulationError, Simulator,
    Structure, Values,
};
use jolty_ir::Circuit;
use jolty_models::Library;
use serde::Deserialize;

/// A fixture's plan, as `benchmarks/speed/plans/<fixture>.json` holds it in the engine's format.
#[derive(Deserialize)]
struct Plan {
    /// The circuit's file in `benchmarks/speed/circuits`, without its extension.
    circuit: String,
    /// The analyses, in order, each by name.
    analyses: Vec<(String, Analysis)>,
    options: Options,
    /// A Monte Carlo analysis's samples and seed.
    monte_carlo: Option<Samples>,
}

/// A Monte Carlo analysis of a fixture's plan.
#[derive(Deserialize)]
struct Samples {
    samples: u64,
    seed: u64,
}

/// The samples of a Monte Carlo fixture a batch runs: those the suite's Python timing runs.
const SAMPLES: u64 = 1000;

/// A sink that keeps no outcome, as a batch that only measures its runs keeps none.
struct Dropped;

impl Sink for Dropped {
    fn accept(&self, _: u64, outcome: jolty_engine::batch::Run) {
        black_box(outcome);
    }
}

/// A fixture: its name, its circuit's JSON text and its plan.
struct Fixture {
    name: String,
    text: String,
    plan: Plan,
}

impl Fixture {
    /// Parses the circuit JSON and constructs a fresh [`Simulator`].
    fn setup(&self, library: &Library) -> Result<Simulator, Box<dyn Error>> {
        let circuit = Circuit::from_json(&self.text)?;
        Ok(Simulator::new(&circuit, library, &Conditions::default())?)
    }

    /// Parses the circuit JSON and elaborates a shared [`Elaborated`] instance.
    fn elaborated(&self, library: &Library) -> Result<Arc<Elaborated>, Box<dyn Error>> {
        let circuit = Circuit::from_json(&self.text)?;
        Ok(Elaborated::new(&circuit, library, &Structure::default())?)
    }

    /// Runs every analysis in the fixture plan in order on a fresh [`Simulator`].
    fn run(&self, library: &Library) -> Result<Vec<AnalysisResult>, Box<dyn Error>> {
        let mut simulator = self.setup(library)?;
        let outcomes: Result<Vec<AnalysisResult>, SimulationError> = self
            .plan
            .analyses
            .iter()
            .map(|(_, analysis)| simulator.run(analysis, &self.plan.options))
            .collect();
        Ok(outcomes?)
    }
}

/// Returns the speed benchmark directory (`benchmarks/speed`).
fn speed() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/speed")
}

/// Loads every fixture from `benchmarks/speed/plans`, sorted by name.
fn fixtures() -> Result<Vec<Fixture>, Box<dyn Error>> {
    let mut plans: Vec<PathBuf> = fs::read_dir(speed().join("plans"))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<_, _>>()?;
    plans.sort();
    let mut fixtures = Vec::new();
    for path in plans {
        let name = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| format!("{} has no name", path.display()))?
            .to_owned();
        let plan: Plan = serde_json::from_str(&fs::read_to_string(&path)?)?;
        let circuit = speed()
            .join("circuits")
            .join(format!("{}.json", plan.circuit));
        let text = fs::read_to_string(&circuit)?;
        fixtures.push(Fixture { name, text, plan });
    }
    Ok(fixtures)
}

#[expect(
    clippy::expect_used,
    reason = "a benchmark whose fixtures cannot be read, or that fails, has nothing to measure"
)]
fn benchmark(c: &mut Criterion) {
    let library = Library::standard();
    let fixtures = fixtures().expect("the fixtures of benchmarks/speed/ read");
    for fixture in &fixtures {
        if let Some(samples) = &fixture.plan.monte_carlo {
            batches(c, fixture, samples, &library);
            continue;
        }
        let mut group = c.benchmark_group(&fixture.name);
        // A run of the larger fixtures takes seconds: each sample is one run or more, and
        // ten samples are the fewest criterion takes.
        group
            .sampling_mode(SamplingMode::Flat)
            .sample_size(10)
            .warm_up_time(Duration::from_secs(1))
            .measurement_time(Duration::from_secs(5));
        group.bench_function("setup", |b| b.iter(|| black_box(fixture.setup(&library))));
        // The circuit is elaborated once, when the benchmark is selected.
        let mut elaborated = None;
        group.bench_function("shared", |b| {
            let elaborated = elaborated.get_or_insert_with(|| {
                fixture
                    .elaborated(&library)
                    .expect("the fixture elaborates")
            });
            b.iter(|| {
                black_box(Simulator::from_elaborated(
                    Arc::clone(elaborated),
                    Values::default(),
                ))
            });
        });
        // A fixture selected runs once before it is measured, so that a failure stops the
        // benchmark rather than being measured.
        let mut checked = false;
        group.bench_function("run", |b| {
            if !checked {
                fixture.run(&library).expect("the fixture runs");
                checked = true;
            }
            b.iter(|| black_box(fixture.run(&library)));
        });
        group.finish();
    }
}

/// Measures the first [`SAMPLES`] Monte Carlo samples of `fixture` across 1, 2, 4, 8, and
/// full-core thread pools.
#[expect(
    clippy::expect_used,
    reason = "a benchmark whose fixture cannot be read, or that fails, has nothing to measure"
)]
fn batches(c: &mut Criterion, fixture: &Fixture, samples: &Samples, library: &Library) {
    let circuit = Circuit::from_json(&fixture.text).expect("the fixture reads");
    let elaborated = fixture.elaborated(library).expect("the fixture elaborates");
    let toleranced = toleranced(&circuit).expect("its tolerances are of numbers");
    let count = SAMPLES.min(samples.samples);
    let variants = MonteCarlo::new(&elaborated, samples.seed, count, toleranced)
        .expect("its tolerances are of its numbers");
    let analyses = fixture
        .plan
        .analyses
        .iter()
        .map(|(_, analysis)| (analysis.clone(), fixture.plan.options.clone()))
        .collect();
    let batch = Batch::new(elaborated, analyses, Vec::new()).expect("a batch");
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    let mut counts: Vec<usize> = [1, 2, 4, 8]
        .into_iter()
        .filter(|&threads| threads < cores)
        .collect();
    counts.push(cores);
    let mut group = c.benchmark_group(&fixture.name);
    group
        .sampling_mode(SamplingMode::Flat)
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5));
    for threads in counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a pool");
        group.bench_with_input(BenchmarkId::new("batch", threads), &threads, |b, _| {
            b.iter(|| {
                pool.install(|| batch.run(&variants, 0..count, &Dropped))
                    .expect("the samples lie within the variants")
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
