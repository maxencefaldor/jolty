//! The `thermal_resistance` kernel, compiled by jolty-veriloga from `crates/jolty-veriloga/tests/fixtures/thermal_resistance.va`.
//!
//! Written by `cargo run -p jolty-veriloga --bin compile`, never by hand. Its source says:
//!
//! ```text
//! jolty's thermal resistance in Verilog-A: a heat flow between two thermal nodes, in
//! proportion to the temperature between them.
//! ```

#![cfg_attr(
    not(test),
    deny(
        clippy::indexing_slicing,
        clippy::panic,
        reason = "a kernel never panics, and reads fixed arrays through `get`"
    )
)]

use jolty_models::{Branch, BranchKind, Dual, Interface, Kernel, Layout, Lift, Local, ParameterError, Real, Stamp};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The `thermal_resistance` model, compiled from its Verilog-A source.
#[derive(Debug, Clone, Copy)]
pub struct ThermalResistance;

/// The parameters of a [`ThermalResistance`], with its Verilog-A source's defaults.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift)]
#[serde(default, deny_unknown_fields, bound = "")]
pub struct ThermalResistanceParameters<R: Real = f64> {
    /// The resistance, in K/W; default 1.
    pub resistance: R,
}

impl<R: Real> Default for ThermalResistanceParameters<R> {
    fn default() -> Self {
        Self {
            resistance: R::constant(1.0),
        }
    }
}

/// A [`ThermalResistance`]'s numbers at its temperature: what depends on no probe.
#[derive(Debug, Clone, Copy)]
pub struct PreparedThermalResistance<R: Real> {
    resistance: R,
    h1: R,
}

/// Where a [`ThermalResistance`]'s contributions go among its local unknowns, and the conditions
/// that chose them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThermalResistanceShape {
    a_by_a: usize,
    a_by_b: usize,
    b_by_a: usize,
    b_by_b: usize,
}

impl Kernel for ThermalResistance {
    const NAME: &'static str = "thermal_resistance";
    type Parameters<R: Real> = ThermalResistanceParameters<R>;
    type Shape = ThermalResistanceShape;
    type Prepared<R: Real> = PreparedThermalResistance<R>;
    type Memory = ();

    fn interface<R: Real>(_: &ThermalResistanceParameters<R>) -> Interface {
        Interface {
            thermal: vec![0, 1],
            ..Interface::new(&["a", "b"], &[Branch::new(0, 1, BranchKind::Conductive)])
        }
    }

    fn linear(_: &ThermalResistanceShape, _: &PreparedThermalResistance<f64>) -> bool {
        true
    }

    fn prepare<R: Real>(
        parameters: &ThermalResistanceParameters<R>,
        _: R,
    ) -> Result<PreparedThermalResistance<R>, ParameterError> {
        ParameterError::positive("resistance", parameters.resistance)?;
        Ok(PreparedThermalResistance { resistance: parameters.resistance, h1: R::constant(1.0) / parameters.resistance })
    }

    fn layout<R: Real>(
        _: &ThermalResistanceParameters<R>,
        _: &PreparedThermalResistance<R>,
        layout: &mut Layout,
    ) -> Result<ThermalResistanceShape, ParameterError> {
        Ok(ThermalResistanceShape { a_by_a: layout.entry(0, 0), a_by_b: layout.entry(0, 1), b_by_a: layout.entry(1, 0), b_by_b: layout.entry(1, 1) })
    }

    fn evaluate<R: Real>(
        shape: &ThermalResistanceShape,
        prepared: &PreparedThermalResistance<R>,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        struct Locals<R: Real> {
            voltage_a_b: Dual<R, 1>,
            a_b_flow: Dual<R, 1>,
        }
        let mut v = Locals { voltage_a_b: Dual::number(0.0), a_b_flow: Dual::number(0.0) };
        v.voltage_a_b = Dual { value: at.unknown::<R>(0) - at.unknown::<R>(1), gradient: [R::constant(1.0)] };
        v.a_b_flow = v.a_b_flow + v.voltage_a_b.chain((v.voltage_a_b.value / prepared.resistance, prepared.h1));
        let t0000 = v.a_b_flow.value;
        stamp.add_residual(0, t0000);
        stamp.add_residual(1, -t0000);
        let [t0001] = v.a_b_flow.gradient;
        stamp.add_conductance(shape.a_by_a, t0001);
        stamp.add_conductance(shape.a_by_b, -t0001);
        stamp.add_conductance(shape.b_by_a, -t0001);
        stamp.add_conductance(shape.b_by_b, t0001);
    }

}
