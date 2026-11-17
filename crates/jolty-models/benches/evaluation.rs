//! The time of one evaluation of a device's kernel, in each of the cases of `cases/`.
//! Name one after `--` to run it alone, as
//! `cargo bench -p jolty-models --bench evaluation -- vbic/`; save one build's times with
//! `--save-baseline NAME` and compare another's with `--baseline NAME`. Times move with
//! the machine's load and with how a build lays out its code; `instructions.rs` counts
//! what they cannot resolve.

mod cases;

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

/// A case's name, and what makes it.
type Named = (&'static str, fn() -> cases::Case);

/// Every case, by name.
const CASES: [Named; 16] = [
    (
        "bipolar_transistor/defaults",
        cases::bipolar_transistor_defaults,
    ),
    ("bipolar_transistor/full", cases::bipolar_transistor_full),
    (
        "bipolar_transistor/delayed",
        cases::bipolar_transistor_delayed,
    ),
    ("vbic/defaults", cases::vbic_defaults),
    ("vbic/full", cases::vbic_full),
    ("vbic/delayed", cases::vbic_delayed),
    ("jfet/defaults", cases::jfet_defaults),
    ("jfet/full", cases::jfet_full),
    ("vdmos/quasi_saturated", cases::vdmos_quasi_saturated),
    ("resistor/coefficients", cases::resistor_coefficients),
    ("capacitor/coefficients", cases::capacitor_coefficients),
    (
        "voltage_controlled_switch/between",
        cases::voltage_controlled_switch_between,
    ),
    (
        "current_controlled_switch/between",
        cases::current_controlled_switch_between,
    ),
    (
        "voltage_controlled_memristor/driven",
        cases::voltage_controlled_memristor_driven,
    ),
    (
        "current_controlled_memristor/driven",
        cases::current_controlled_memristor_driven,
    ),
    ("photocell/lagging", cases::photocell_lagging),
];

fn evaluation(criterion: &mut Criterion) {
    for (name, case) in CASES {
        let mut case = case();
        criterion.bench_function(name, |bencher| bencher.iter(|| black_box(case.evaluate())));
    }
}

criterion_group!(benches, evaluation);
criterion_main!(benches);
