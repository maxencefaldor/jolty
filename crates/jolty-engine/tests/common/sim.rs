//! Default simulator constructor shared across engine integration tests.

use jolty_engine::{Conditions, Simulator};
use jolty_ir::Circuit;
use jolty_models::Library;

/// Elaborates `circuit` with the standard library and default conditions.
pub(crate) fn simulator(circuit: &Circuit) -> Simulator {
    Simulator::new(circuit, &Library::standard(), &Conditions::default())
        .expect("the circuit elaborates")
}
