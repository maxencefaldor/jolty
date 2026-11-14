//! The engine's benchmarks: each fixture of `perf/`, its circuit read from its JSON text
//! and elaborated, then its analyses run in order, end to end.
//!
//! For each fixture whose plan runs no Monte Carlo analysis, `<fixture>/setup` measures
//! the setup alone, from the circuit's JSON text to a simulator ready to run, and
//! `<fixture>/run` the setup and then every analysis of the plan. `scripts/bench` runs
//! the suite's fixtures and reads what criterion measured; name others after `--` to run
//! them, such as `cargo bench -p jolty-engine --bench fixtures -- inverter_chain_5000/`.

use std::error::Error;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Duration;

use criterion::{Criterion, SamplingMode, criterion_group, criterion_main};
use jolty_engine::{Analysis, Conditions, Options, Outcome, SimulationError, Simulator};
use jolty_ir::Circuit;
use jolty_models::Library;
use serde::Deserialize;

/// A fixture's plan, as `perf/plans/<fixture>.json` holds it in the engine's format.
#[derive(Deserialize)]
struct Plan {
    /// The circuit's file in `perf/circuits`, without its extension.
    circuit: String,
    /// The analyses, in order, each by name.
    analyses: Vec<(String, Analysis)>,
    options: Options,
    /// A Monte Carlo analysis's samples and seed, which only the Python timing runs.
    monte_carlo: Option<serde_json::Value>,
}

/// A fixture: its name, its circuit's JSON text and its plan.
struct Fixture {
    name: String,
    text: String,
    plan: Plan,
}

impl Fixture {
    /// A simulator of the circuit, read from its JSON text.
    fn setup(&self, library: &Library) -> Result<Simulator, Box<dyn Error>> {
        let circuit = Circuit::from_json(&self.text)?;
        Ok(Simulator::new(&circuit, library, &Conditions::default())?)
    }

    /// The outcomes of the plan's analyses, run in order on a simulator set up afresh.
    fn run(&self, library: &Library) -> Result<Vec<Outcome>, Box<dyn Error>> {
        let mut simulator = self.setup(library)?;
        let outcomes: Result<Vec<Outcome>, SimulationError> = self
            .plan
            .analyses
            .iter()
            .map(|(_, analysis)| simulator.run(analysis, &self.plan.options))
            .collect();
        Ok(outcomes?)
    }
}

/// The `perf` directory.
fn perf() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../perf")
}

/// Every fixture whose plan runs no Monte Carlo analysis, in the order of their names.
fn fixtures() -> Result<Vec<Fixture>, Box<dyn Error>> {
    let mut plans: Vec<PathBuf> = fs::read_dir(perf().join("plans"))?
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
        if plan.monte_carlo.is_some() {
            continue;
        }
        let circuit = perf()
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
    let fixtures = fixtures().expect("the fixtures of perf/ read");
    for fixture in &fixtures {
        let mut group = c.benchmark_group(&fixture.name);
        // A run of the larger fixtures takes seconds: each sample is one run or more, and
        // ten samples are the fewest criterion takes.
        group
            .sampling_mode(SamplingMode::Flat)
            .sample_size(10)
            .warm_up_time(Duration::from_secs(1))
            .measurement_time(Duration::from_secs(5));
        group.bench_function("setup", |b| b.iter(|| black_box(fixture.setup(&library))));
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

criterion_group!(benches, benchmark);
criterion_main!(benches);
