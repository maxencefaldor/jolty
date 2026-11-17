//! The instructions one evaluation of a device's kernel executes, in each of the cases of
//! `cases/`, counted by Valgrind's Callgrind through gungraun, the case's making left
//! out: deterministic, so two builds compare to the instruction whatever the machine's
//! load, where `evaluation.rs`'s times cannot. It needs Valgrind and the gungraun-runner
//! of gungraun's version, `cargo install gungraun-runner --version 0.20.0 --locked`; run
//! it as `cargo bench -p jolty-models --bench instructions`, which compares each count
//! with the last run's, or name a baseline with `-- --save-baseline=NAME` and
//! `-- --baseline=NAME`.

#![expect(
    missing_docs,
    unused_qualifications,
    reason = "gungraun's macros expand to items undocumented and paths qualified in full, which these lints read as the bench's own"
)]

mod cases;

use std::hint::black_box;

use gungraun::{library_benchmark, library_benchmark_group, main};

#[library_benchmark]
#[bench::bipolar_transistor_defaults(setup = cases::bipolar_transistor_defaults)]
#[bench::bipolar_transistor_full(setup = cases::bipolar_transistor_full)]
#[bench::bipolar_transistor_delayed(setup = cases::bipolar_transistor_delayed)]
#[bench::vbic_defaults(setup = cases::vbic_defaults)]
#[bench::vbic_full(setup = cases::vbic_full)]
#[bench::vbic_delayed(setup = cases::vbic_delayed)]
#[bench::jfet_defaults(setup = cases::jfet_defaults)]
#[bench::jfet_full(setup = cases::jfet_full)]
#[bench::vdmos_quasi_saturated(setup = cases::vdmos_quasi_saturated)]
#[bench::resistor_coefficients(setup = cases::resistor_coefficients)]
#[bench::capacitor_coefficients(setup = cases::capacitor_coefficients)]
#[bench::voltage_controlled_switch_between(setup = cases::voltage_controlled_switch_between)]
#[bench::current_controlled_switch_between(setup = cases::current_controlled_switch_between)]
#[bench::voltage_controlled_memristor_driven(setup = cases::voltage_controlled_memristor_driven)]
#[bench::current_controlled_memristor_driven(setup = cases::current_controlled_memristor_driven)]
#[bench::photocell_lagging(setup = cases::photocell_lagging)]
fn evaluation(mut case: cases::Case) -> bool {
    black_box(case.evaluate())
}

library_benchmark_group!(name = kernels, benchmarks = [evaluation]);

main!(library_benchmark_groups = kernels);
