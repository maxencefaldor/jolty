//! Groups: the instances of one kernel and one shape, evaluated together, and the only code
//! that touches threads.
//!
//! [`Groups`] holds what a circuit's groups are, built once with its structure and shared:
//! each group's kernel and shape, its instances, the circuit's unknown of each of their
//! local unknowns and its stored entry of each of their local entries, and where each
//! instance's block lies in a slab. [`Runs`] holds what one run writes: each instance's
//! prepared numbers and memory, and the slab.
//!
//! Each instance writes its contributions into its own block, and every value of the
//! circuit's equations adds the blocks' contributions to it in one fixed order: group after
//! group, instance after instance, local index after local index. A run
//! evaluated serially adds each instance's block as soon as it is written; one evaluated in
//! parallel writes every block first, then gathers them through a
//! [`GatherMap`](jolty_numerics::GatherMap) in the same order. Either way, the sums are the
//! same bits.
//!
//! A value of more than [`CHUNK`](jolty_numerics::reduce::CHUNK) contributions, such as a
//! supply's, joined to every device, would make its gather read one place of nearly every
//! block, one after another on one thread. Each instance's contributions to such a value
//! are also copied, as soon as its block is written, to places of the slab beyond the
//! blocks, where every such value's contributions lie together in their order, and its
//! gather reads them there: the same values, in the same order.
//!
//! A linear instance, as [`Kernel::linear`](crate::Kernel::linear) says, is evaluated once
//! a run, for its entries, which its block keeps: they sum, once a run, in the order of the
//! groups and their instances, into the base, which lies beyond the copies, one value for
//! each stored entry they add to. Each evaluation then adds to each stored entry the base's
//! value first, then the entries of the instances that are not linear, in their order, and
//! writes each linear instance's residuals, their scales and its charges from its entries,
//! as its kernel's equations are its entries times its local unknowns.

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
