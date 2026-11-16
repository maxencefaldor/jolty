//! A foundry's transistor, which ngspice simulates with the foundry's own models.
//!
//! The engine solves none: the device declares its terminals and branches, as a MOSFET's,
//! so the structural checks read it, and an analysis of a circuit that holds one stops and
//! names it. Its parameters name the process design kit and the kit's model, and give its
//! size, which a netlist written for ngspice hands the kit's model.

#![cfg_attr(
    not(test),
    deny(
        clippy::indexing_slicing,
        clippy::panic,
        reason = "a kernel never panics, and reads fixed arrays through `get`"
    )
)]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{Branch, BranchKind, Interface, Kernel, Layout, Local, ParameterError, Real, Stamp};

/// A transistor of a foundry's process, between `drain`, `gate`, `source` and `bulk`.
#[derive(Debug, Clone, Copy)]
pub struct FoundryMosfet;

/// The parameters of a [`FoundryMosfet`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields, bound = "")]
pub struct FoundryMosfetParameters<R: Real = f64> {
    /// The process design kit, such as `sky130`.
    pub pdk: String,
    /// The kit's model of the transistor, such as `sky130_fd_pr__nfet_01v8`.
    pub model: String,
    /// The channel's width, all its fingers together, in metres; positive.
    pub width: R,
    /// The channel's length, in metres; positive.
    pub length: R,
    /// The fingers the width is split into; 1 unless given.
    #[serde(default = "one")]
    pub fingers: u32,
    /// The transistors in parallel; 1 unless given.
    #[serde(default = "one")]
    pub multiplier: u32,
}

const fn one() -> u32 {
    1
}

impl Kernel for FoundryMosfet {
    const NAME: &'static str = "foundry_mosfet";
    const FOUNDRY: bool = true;
    type Parameters<R: Real> = FoundryMosfetParameters<R>;
    type Shape = ();
    type Prepared<R: Real> = ();
    type Memory = ();

    fn interface<R: Real>(_: &FoundryMosfetParameters<R>) -> Interface {
        Interface::new(
            &["drain", "gate", "source", "bulk"],
            &[
                Branch::new(0, 2, BranchKind::Conductive),
                Branch::new(3, 0, BranchKind::Conductive),
                Branch::new(3, 2, BranchKind::Conductive),
                Branch::new(1, 2, BranchKind::Capacitive),
                Branch::new(1, 0, BranchKind::Capacitive),
                Branch::new(1, 3, BranchKind::Capacitive),
            ],
        )
    }

    fn prepare<R: Real>(
        parameters: &FoundryMosfetParameters<R>,
        _: R,
    ) -> Result<(), ParameterError> {
        for (name, text) in [("pdk", &parameters.pdk), ("model", &parameters.model)] {
            if text.is_empty() {
                return Err(ParameterError(format!(
                    "{name} must name one, not be empty"
                )));
            }
        }
        ParameterError::positive("width", parameters.width)?;
        ParameterError::positive("length", parameters.length)?;
        for (name, count) in [
            ("fingers", parameters.fingers),
            ("multiplier", parameters.multiplier),
        ] {
            if count == 0 {
                return Err(ParameterError(format!("{name} must be at least 1, not 0")));
            }
        }
        Ok(())
    }

    /// Claims nothing: no analysis runs on a circuit that holds a foundry's device.
    fn layout<R: Real>(
        _: &FoundryMosfetParameters<R>,
        (): &(),
        _: &mut Layout,
    ) -> Result<(), ParameterError> {
        Ok(())
    }

    /// Adds nothing: no analysis runs on a circuit that holds a foundry's device.
    fn evaluate<R: Real>((): &(), (): &(), (): &mut (), _: &Local<'_>, _: &mut Stamp<'_, R>) {}
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{Differentiable, Model};

    fn parameters(value: serde_json::Value) -> jolty_ir::Parameters {
        let serde_json::Value::Object(map) = value else {
            panic!("an object");
        };
        map
    }

    #[test]
    fn a_foundry_mosfet_has_a_mosfets_terminals_and_is_a_foundrys() {
        let model = Differentiable::<FoundryMosfet>::new();
        assert!(model.foundry());
        assert!(!Differentiable::<crate::Resistor>::new().foundry());
        let interface = model
            .read(&parameters(json!({
                "pdk": "sky130",
                "model": "sky130_fd_pr__nfet_01v8",
                "width": 1e-6,
                "length": 1.5e-7,
            })))
            .expect("valid")
            .interface();
        assert_eq!(interface.terminals, ["drain", "gate", "source", "bulk"]);
        assert_eq!(interface.thermal, Vec::<usize>::new());
    }
}
