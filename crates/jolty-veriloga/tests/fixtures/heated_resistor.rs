//! The `heated_resistor` kernel, compiled by jolty-veriloga from `crates/jolty-veriloga/tests/fixtures/heated_resistor.va`.
//!
//! Written by `cargo run -p jolty-veriloga --bin compile`, never by hand. Original source notice:
//!
//! ```text
//! A resistor whose resistance rises with the temperature of its thermal terminal, which
//! its power heats: a thermal node read alone, as `Temp(t)`, its rise above the ambient,
//! and written through a branch declared on it alone, to the ground.
//! ```

#![cfg_attr(
    not(test),
    deny(
        clippy::indexing_slicing,
        clippy::panic,
        reason = "a kernel never panics, and reads fixed arrays through `get`"
    )
)]

use jolty_models::{
    Branch, BranchKind, Dual, Interface, Kernel, Layout, Lift, Local, ParameterError, Real, Stamp,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A local index no instance has: a write to it goes nowhere, and a read of it reads
/// zero, as `Stamp` and `Local` promise.
const NOWHERE: usize = usize::MAX;

/// The `heated_resistor` model, compiled from its Verilog-A source.
#[derive(Debug, Clone, Copy)]
pub struct HeatedResistor;

/// The parameters of a [`HeatedResistor`], with its Verilog-A source's defaults.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift)]
#[serde(default, deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct HeatedResistorParameters<R: Real = f64> {
    /// The resistance at the ambient temperature, in ohms; default 100.
    pub resistance: R,
    /// The linear temperature coefficient, in 1/K; default 0.001.
    pub temperature_coefficient: R,
}

impl<R: Real> Default for HeatedResistorParameters<R> {
    fn default() -> Self {
        Self {
            resistance: R::constant(100.0),
            temperature_coefficient: R::constant(0.001),
        }
    }
}

/// A [`HeatedResistor`]'s numbers at its temperature: what depends on no probe.
#[derive(Debug, Clone, Copy)]
pub struct PreparedHeatedResistor<R: Real> {
    resistance: R,
    temperature_coefficient: R,
    /// The conditions that shape the kernel.
    conditions: [bool; 1],
}

/// Where a [`HeatedResistor`]'s contributions go among its local unknowns, and the conditions
/// that chose them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeatedResistorShape {
    a_by_a: usize,
    a_by_b: usize,
    a_by_t: usize,
    b_by_a: usize,
    b_by_b: usize,
    b_by_t: usize,
    t_by_a: usize,
    t_by_b: usize,
    t_by_t: usize,
    conditions: [bool; 1],
    active: [bool; 1],
}

impl Kernel for HeatedResistor {
    const NAME: &'static str = "heated_resistor";
    type Parameters<R: Real> = HeatedResistorParameters<R>;
    type Shape = HeatedResistorShape;
    type Prepared<R: Real> = PreparedHeatedResistor<R>;
    type Memory = ();

    fn interface<R: Real>(_: &HeatedResistorParameters<R>) -> Interface {
        Interface {
            thermal: vec![2],
            ..Interface::new(&["a", "b", "t"], &[Branch::new(0, 1, BranchKind::Conductive)])
        }
    }

    fn linear(shape: &HeatedResistorShape, prepared: &PreparedHeatedResistor<f64>) -> bool {
        prepared.temperature_coefficient == 0.0 && !shape.conditions[0]
    }

    fn prepare<R: Real>(
        parameters: &HeatedResistorParameters<R>,
        _: R,
    ) -> Result<PreparedHeatedResistor<R>, ParameterError> {
        ParameterError::positive("resistance", parameters.resistance)?;
        Ok(PreparedHeatedResistor {
            resistance: parameters.resistance,
            temperature_coefficient: parameters.temperature_coefficient,
            conditions: [1 != 0],
        })
    }

    fn layout<R: Real>(
        _: &HeatedResistorParameters<R>,
        prepared: &PreparedHeatedResistor<R>,
        layout: &mut Layout,
    ) -> Result<HeatedResistorShape, ParameterError> {
        let mut active = [false; 1];
        if prepared.conditions[0] {
            active[0] = true;
        }
        Ok(HeatedResistorShape {
            a_by_a: layout.entry(0, 0),
            a_by_b: layout.entry(0, 1),
            a_by_t: layout.entry(0, 2),
            b_by_a: layout.entry(1, 0),
            b_by_b: layout.entry(1, 1),
            b_by_t: layout.entry(1, 2),
            t_by_a: if active[0] { layout.entry(2, 0) } else { NOWHERE },
            t_by_b: if active[0] { layout.entry(2, 1) } else { NOWHERE },
            t_by_t: if active[0] { layout.entry(2, 2) } else { NOWHERE },
            conditions: prepared.conditions,
            active,
        })
    }

    fn evaluate<R: Real>(
        shape: &HeatedResistorShape,
        prepared: &PreparedHeatedResistor<R>,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        struct Locals<R: Real> {
            current: Dual<R, 2>,
            voltage_a_b: Dual<R, 1>,
            voltage_t_ground: Dual<R, 1>,
            heat_flow: Dual<R, 2>,
            a_b_flow: Dual<R, 2>,
        }
        let mut v = Locals {
            current: Dual::number(0.0),
            voltage_a_b: Dual::number(0.0),
            voltage_t_ground: Dual::number(0.0),
            heat_flow: Dual::number(0.0),
            a_b_flow: Dual::number(0.0),
        };
        v.voltage_a_b = Dual::variable(at.unknown::<R>(0) - at.unknown::<R>(1));
        v.voltage_t_ground = Dual::variable(at.unknown::<R>(2) - at.unknown::<R>(NOWHERE));
        v.current = v.voltage_a_b.embed::<2>([0]) / (v.voltage_t_ground.scale(prepared.temperature_coefficient) + 1.0).scale(prepared.resistance).embed::<2>([1]);
        v.a_b_flow = v.a_b_flow + v.current;
        if shape.conditions[0] {
            v.heat_flow = v.heat_flow + (-v.voltage_a_b).embed::<2>([0]) * v.current;
        }
        if shape.active[0] {
            let t0000 = v.heat_flow.value;
            stamp.add_residual(2, t0000);
            let [t0001, t0002] = v.heat_flow.gradient;
            stamp.add_conductance(shape.t_by_a, t0001);
            stamp.add_conductance(shape.t_by_b, -t0001);
            stamp.add_conductance(shape.t_by_t, t0002);
        }
        let t0003 = v.a_b_flow.value;
        stamp.add_residual(0, t0003);
        stamp.add_residual(1, -t0003);
        let [t0004, t0005] = v.a_b_flow.gradient;
        stamp.add_conductance(shape.a_by_a, t0004);
        stamp.add_conductance(shape.a_by_b, -t0004);
        stamp.add_conductance(shape.b_by_a, -t0004);
        stamp.add_conductance(shape.b_by_b, t0004);
        stamp.add_conductance(shape.a_by_t, t0005);
        stamp.add_conductance(shape.b_by_t, -t0005);
    }
}
