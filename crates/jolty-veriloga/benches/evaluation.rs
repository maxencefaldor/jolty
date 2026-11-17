//! The time of one evaluation of each compiled kernel against its hand-written twin's,
//! one instance on the shared bench, in each of its shapes: `<model>/<case>/hand` and
//! `<model>/<case>/compiled`. Name either after `--` to run it alone, so that the two
//! run interleaved, as `cargo bench -p jolty-veriloga --bench evaluation -- /hand`.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use jolty_models::testing::Bench;
use jolty_models::{Input, Kernel, REFERENCE_TEMPERATURE};
use jolty_veriloga::kernels::{diode, resistor};
use schemars::JsonSchema;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Times the evaluation of an instance of `K` with `parameters` at `x`, at a step of a
/// transient, as `name`.
fn evaluation<K: Kernel>(criterion: &mut Criterion, name: &str, parameters: &str, x: &[f64])
where
    K::Parameters<f64>: DeserializeOwned + Serialize + JsonSchema,
{
    let mut bench = Bench::at::<K>(parameters, REFERENCE_TEMPERATURE);
    let mut contributions = bench.contributions();
    let mut input = Input::new(x, Some(1e-9), 1e-12);
    input.limiting = false;
    criterion.bench_function(name, |bencher| {
        bencher.iter(|| black_box(bench.evaluate_into(black_box(&input), &mut contributions)));
    });
}

fn twins(criterion: &mut Criterion) {
    let resistors: [(&str, &str, &[f64]); 3] = [
        ("linear", r#"{"resistance": 1000}"#, &[1.0, 0.0]),
        (
            "coefficients",
            r#"{"resistance": 1000, "voltage_coefficient": 0.01}"#,
            &[1.0, 0.0],
        ),
        (
            "small",
            r#"{"resistance": 0.5, "voltage_coefficient": 0.01}"#,
            &[1.0, 0.0, 2.0],
        ),
    ];
    for (case, parameters, x) in resistors {
        evaluation::<jolty_models::Resistor>(
            criterion,
            &format!("resistor/{case}/hand"),
            parameters,
            x,
        );
        evaluation::<resistor::Resistor>(
            criterion,
            &format!("resistor/{case}/compiled"),
            parameters,
            x,
        );
    }
    let diodes: [(&str, &str, &[f64]); 4] = [
        ("plain", "{}", &[0.6, 0.0]),
        (
            "charged",
            r#"{"junction_capacitance": 2e-12, "transit_time": 1e-9}"#,
            &[0.6, 0.0],
        ),
        (
            "breakdown",
            r#"{"breakdown_voltage": 5, "junction_capacitance": 2e-12}"#,
            &[-5.2, 0.0],
        ),
        (
            "resistive",
            r#"{"series_resistance": 10, "junction_capacitance": 1e-12}"#,
            &[0.7, 0.0, 0.65],
        ),
    ];
    for (case, parameters, x) in diodes {
        evaluation::<jolty_models::Diode>(criterion, &format!("diode/{case}/hand"), parameters, x);
        evaluation::<diode::Diode>(criterion, &format!("diode/{case}/compiled"), parameters, x);
    }
}

criterion_group!(benches, twins);
criterion_main!(benches);
