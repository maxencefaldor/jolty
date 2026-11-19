//! The kernels compiled from Verilog-A: each checked-in one is what the compiler writes
//! from its source, and each follows the kernel rules.

#![cfg(test)]

use std::path::Path;

use jolty_models::testing::Bench;
use jolty_models::{Input, Kernel, REFERENCE_TEMPERATURE};
use jolty_veriloga::kernels::{diode, resistor};
use jolty_veriloga::{MODELS, compile};
use schemars::JsonSchema;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Verifies that every checked-in kernel matches compiler output from its Verilog-A source.
#[test]
fn each_kernel_is_what_the_compiler_writes() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (source, kernel) in MODELS {
        let text = std::fs::read_to_string(root.join(source)).expect("the source");
        let compiled = compile(&text, &format!("crates/jolty-veriloga/{source}"), &|_| None)
            .expect("it compiles");
        let written = std::fs::read_to_string(root.join(kernel)).expect("the kernel");
        assert!(
            compiled == written,
            "{kernel} is not what the compiler writes from {source}: run `cargo run -p jolty-veriloga --bin compile`"
        );
    }
}

/// Counts heap allocations made by evaluating an instance of `K` with `parameters` across
/// operating-point and transient steps after warm-up.
fn allocations<K: Kernel>(parameters: &str, x: &[f64]) -> u64
where
    K::Parameters<f64>: DeserializeOwned + Serialize + JsonSchema,
{
    let mut bench = Bench::at::<K>(parameters, REFERENCE_TEMPERATURE);
    let mut contributions = bench.contributions();
    let moved: Vec<f64> = x.iter().map(|value| value * 1.5 - 0.2).collect();
    bench.evaluate_into(&Input::new(x, None, 1e-12), &mut contributions);
    let info = allocation_counter::measure(|| {
        for (step, at) in [x, &moved, x].into_iter().enumerate() {
            let mut input = Input::new(at, (step == 1).then_some(1e-9), 1e-12);
            input.limiting = step > 0;
            bench.evaluate_into(&input, &mut contributions);
        }
    });
    info.count_total
}

/// Verifies that compiled resistor and diode kernels allocate zero heap memory across all
/// shapes.
#[test]
fn a_compiled_kernel_allocates_nothing() {
    for (parameters, x) in [
        (r#"{"resistance": 50}"#, &[1.0, 0.0][..]),
        (r#"{"resistance": 0.5}"#, &[1.0, 0.0, 0.1]),
    ] {
        assert_eq!(
            allocations::<resistor::Resistor>(parameters, x),
            0,
            "{parameters}"
        );
    }
    for (parameters, x) in [
        (
            r#"{"junction_capacitance": 1e-12, "breakdown_voltage": 5}"#,
            &[0.6, 0.0][..],
        ),
        (r#"{"series_resistance": 10}"#, &[0.7, 0.0, 0.65]),
        (r#"{"series_resistance": 0.5}"#, &[0.7, 0.0, 0.65, 0.1]),
    ] {
        assert_eq!(
            allocations::<diode::Diode>(parameters, x),
            0,
            "{parameters}"
        );
    }
}
