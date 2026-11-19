//! Batched device-instance groups sharing one kernel and shape, and the sole parallel
//! execution layer in `jolty-models`.
//!
//! [`Groups`] stores immutable circuit topology built once and shared across runs: each
//! group's kernel, shape, member instances, local-to-global unknown and entry mappings, and
//! slab block offsets. [`Runs`] stores mutable per-run state: prepared parameters, instance
//! memory, and the evaluation slab.
//!
//! Each instance writes contributions into a dedicated slab block. Every global equation
//! sums contributing blocks in a fixed deterministic order: group by group, instance by
//! instance, and local index by local index. Serial evaluation adds each block immediately
//! upon writing, whereas parallel evaluation writes all blocks first and gathers via
//! [`GatherMap`](jolty_numerics::GatherMap) in identical order for bit-identical sums.
//!
//! High-fanout targets with more than [`CHUNK`](jolty_numerics::reduce::CHUNK) contributions
//! (such as supply rails) would otherwise force a single-threaded gather to stride across
//! nearly every block. Contributions to those targets are copied immediately after writing
//! into contiguous copy regions beyond the blocks, preserving exact summation order.
//!
//! Linear instances (where [`Kernel::linear`](crate::Kernel::linear) returns `true`) evaluate
//! Jacobian entries once per run into their blocks and sum them once into a shared base
//! region beyond the copies. Subsequent evaluations add the base value followed by nonlinear
//! entries, and compute linear residuals, scales, and charges directly from cached entries.

mod kernel_run;
mod kind;
mod layout;
mod linear;
mod runs;
mod twin;

pub use kind::Instance;
pub use layout::{Groups, TooLarge};
pub use runs::{Alone, Block, Changed, Runs, Tasks};
pub use twin::{Seeds, Twin, TwinNumber};

pub(crate) use kind::Kind;
