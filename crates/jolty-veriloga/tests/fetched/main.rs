//! The CMC's models whose sources the repository does not hold, against closed forms
//! where their equations reduce to them: `R3_CMC`, `DIODE_CMC`, MOSVAR and PSP.
//!
//! `tests/test_veriloga.py` fetches each source by its digest, compiles it into
//! `kernels/`, which git ignores, and runs these tests with the `fetched` feature, which
//! builds them; without it, or without the kernels, there is nothing to build.

#[path = "../common/closed.rs"]
mod closed;
mod diode_cmc;
mod mosvar;
mod psp103;
mod r3_cmc;
