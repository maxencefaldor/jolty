//! Closed-form verification tests for externally fetched CMC Verilog-A models (`R3_CMC`,
//! `DIODE_CMC`, MOSVAR, and PSP).
//!
//! Running `tests/test_veriloga.py` downloads each verified source by digest, compiles it into
//! the git-ignored `kernels/` directory, and runs these tests under the `fetched` feature.

#[path = "../common/closed.rs"]
mod closed;
mod diode_cmc;
mod mosvar;
mod psp103;
mod r3_cmc;
