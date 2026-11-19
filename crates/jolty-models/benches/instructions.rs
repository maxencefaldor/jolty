//! Instruction counts for single kernel and twin evaluations across all cases in
//! `cases/`, measured under Valgrind Callgrind via gungraun. Setup and drop are excluded
//! by returning the case to the harness, yielding deterministic instruction counts
//! unaffected by host load.
//!
//! Running the suite requires Valgrind and a matching `gungraun-runner`
//! (`cargo install gungraun-runner --version 0.20.0 --locked`). Invoke
//! `cargo bench -p jolty-models --bench instructions` to compare against the previous
//! run, or pass `-- --save-baseline=NAME` and `-- --baseline=NAME`.

#![expect(
    missing_docs,
    unused_qualifications,
    reason = "gungraun macros expand to undocumented items and qualified paths"
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
fn evaluation(mut case: cases::Case) -> cases::Case {
    black_box(case.evaluate());
    case
}

#[library_benchmark]
#[benches::bsim4(args = [1, 2, 4, 8], setup = cases::bsim4_twin)]
#[benches::diode(args = [1, 2, 4, 8], setup = cases::diode_twin)]
#[benches::resistor(args = [1, 2, 4, 8], setup = cases::resistor_twin)]
fn twin(mut case: cases::TwinCase) -> cases::TwinCase {
    case.evaluate();
    case
}

library_benchmark_group!(name = kernels, benchmarks = [evaluation, twin]);

main!(library_benchmark_groups = kernels);
