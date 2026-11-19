//! The `constructs` kernel, compiled by jolty-veriloga from `crates/jolty-veriloga/tests/fixtures/constructs.va`.
//!
//! Written by `cargo run -p jolty-veriloga --bin compile`, never by hand. Its source says:
//!
//! ```text
//! A conductance that saturates, written to exercise the constructs of the CMC's models:
//! an analog function called inline, a named block with variables of its own, tanh and
//! atan, a $warning, output variables, one of them a ddx, an aliasparam, a range bounded
//! by another parameter, a default jolty's $simparam leaves to the source, and $mfactor.
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

/// The `constructs` model, compiled from its Verilog-A source.
#[derive(Debug, Clone, Copy)]
pub struct Constructs;

/// The parameters of a [`Constructs`], with its Verilog-A source's defaults.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift)]
#[serde(default, deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct ConstructsParameters<R: Real = f64> {
    /// The conductance at zero volts, in S; default 0.001.
    #[serde(alias = "g")]
    pub conductance: R,
    /// The voltage the current saturates at, in V; default 1.
    pub saturation: R,
    /// The weight of the bend, a share of the conductance; default 0.001.
    pub bend: R,
    /// The largest saturation voltage the device allows, in V; default 100.
    pub saturation_max: R,
    /// The scale of the conductance, the simulator's own unless given; default 1.
    pub scale: R,
}

impl<R: Real> Default for ConstructsParameters<R> {
    fn default() -> Self {
        Self {
            conductance: R::constant(0.001),
            saturation: R::constant(1.0),
            bend: R::constant(0.001),
            saturation_max: R::constant(100.0),
            scale: R::constant(1.0),
        }
    }
}

/// A [`Constructs`]'s numbers at its temperature: what depends on no probe.
#[derive(Debug, Clone, Copy)]
pub struct PreparedConstructs<R: Real> {
    h0: R,
    saturation: R,
    h2: R,
    h3: R,
    saturated1_vs: R,
    h5: R,
}

/// Where a [`Constructs`]'s contributions go among its local unknowns, and the conditions
/// that chose them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstructsShape {
    a_by_a: usize,
    a_by_b: usize,
    b_by_a: usize,
    b_by_b: usize,
}

/// What a [`Constructs`] remembers of its last evaluation: each output variable.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConstructsMemory {
    current: f64,
    small_signal: f64,
}

impl Kernel for Constructs {
    const NAME: &'static str = "constructs";
    type Parameters<R: Real> = ConstructsParameters<R>;
    type Shape = ConstructsShape;
    type Prepared<R: Real> = PreparedConstructs<R>;
    type Memory = ConstructsMemory;

    fn interface<R: Real>(_: &ConstructsParameters<R>) -> Interface {
        Interface::new(&["a", "b"], &[Branch::new(0, 1, BranchKind::Conductive)])
    }

    fn variable_names() -> &'static [&'static str] {
        &["current", "small_signal"]
    }

    fn variables(memory: &ConstructsMemory, values: &mut [f64]) {
        for (value, kept) in values.iter_mut().zip([memory.current, memory.small_signal]) {
            *value = kept;
        }
    }

    fn prepare<R: Real>(
        parameters: &ConstructsParameters<R>,
        _: R,
    ) -> Result<PreparedConstructs<R>, ParameterError> {
        struct Locals<R: Real> {
            saturated1_g: R,
            saturated1_vs: R,
        }
        ParameterError::positive("conductance", parameters.conductance)?;
        ParameterError::positive("saturation", parameters.saturation)?;
        if !(parameters.saturation.value()..).contains(&parameters.saturation_max.value()) {
            return Err(ParameterError(format!("saturation_max must be at least {:?}, not {:?}", parameters.saturation.value(), parameters.saturation_max.value())));
        }
        ParameterError::positive("scale", parameters.scale)?;
        let mut v = Locals { saturated1_g: R::constant(0.0), saturated1_vs: R::constant(0.0) };
        v.saturated1_g = parameters.conductance;
        v.saturated1_vs = parameters.saturation;
        Ok(PreparedConstructs {
            h0: parameters.bend * parameters.conductance * parameters.saturation,
            saturation: parameters.saturation,
            h2: R::constant(1.0) / parameters.saturation,
            h3: v.saturated1_g * v.saturated1_vs,
            saturated1_vs: v.saturated1_vs,
            h5: R::constant(1.0) / v.saturated1_vs,
        })
    }

    fn layout<R: Real>(
        _: &ConstructsParameters<R>,
        _: &PreparedConstructs<R>,
        layout: &mut Layout,
    ) -> Result<ConstructsShape, ParameterError> {
        Ok(ConstructsShape {
            a_by_a: layout.entry(0, 0),
            a_by_b: layout.entry(0, 1),
            b_by_a: layout.entry(1, 0),
            b_by_b: layout.entry(1, 1),
        })
    }

    fn evaluate<R: Real>(
        shape: &ConstructsShape,
        prepared: &PreparedConstructs<R>,
        memory: &mut ConstructsMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        struct Locals<R: Real> {
            current: Dual<R, 1>,
            small_signal: R,
            v: Dual<R, 1>,
            bent: Dual<R, 1>,
            saturated1_v: Dual<R, 1>,
            saturated1_saturated: Dual<R, 1>,
            voltage_a_b: Dual<R, 1>,
            a_b_flow: Dual<R, 1>,
        }
        let mut v = Locals {
            current: Dual::number(0.0),
            small_signal: R::constant(0.0),
            v: Dual::number(0.0),
            bent: Dual::number(0.0),
            saturated1_v: Dual::number(0.0),
            saturated1_saturated: Dual::number(0.0),
            voltage_a_b: Dual::number(0.0),
            a_b_flow: Dual::number(0.0),
        };
        v.voltage_a_b = Dual::variable(at.unknown::<R>(0) - at.unknown::<R>(1));
        v.v = v.voltage_a_b;
        let t0000 = v.v.chain((v.v.value / prepared.saturation, prepared.h2));
        let t0001 = t0000.value.value();
        let t0002 = t0000.value.chain((t0001.atan(), (t0001 * t0001 + 1.0).recip(), -2.0 * t0001 * (t0001 * t0001 + 1.0).powi(-2)));
        v.bent = t0000.chain((t0002, (t0000.value * t0000.value + 1.0).recip())).scale(prepared.h0);
        v.saturated1_v = v.v;
        let t0003 = v.saturated1_v.chain((v.saturated1_v.value / prepared.saturated1_vs, prepared.h5));
        let t0004 = t0003.value.value();
        let t0005 = t0003.value.chain((t0004.tanh(), 1.0 - t0004.tanh() * t0004.tanh(), -2.0 * t0004.tanh() * (1.0 - t0004.tanh() * t0004.tanh())));
        v.saturated1_saturated = t0003.chain((t0005, R::constant(1.0) - t0005 * t0005)).scale(prepared.h3);
        v.current = (v.saturated1_saturated + v.bent) * 1.0;
        v.a_b_flow = v.a_b_flow + v.current;
        let [t0006] = v.current.gradient;
        v.small_signal = t0006;
        let t0007 = v.a_b_flow.value;
        stamp.add_residual(0, t0007);
        stamp.add_residual(1, -t0007);
        let [t0008] = v.a_b_flow.gradient;
        stamp.add_conductance(shape.a_by_a, t0008);
        stamp.add_conductance(shape.a_by_b, -t0008);
        stamp.add_conductance(shape.b_by_a, -t0008);
        stamp.add_conductance(shape.b_by_b, t0008);
        memory.current = v.current.value.value();
        memory.small_signal = v.small_signal.value();
    }
}

impl Constructs {
    /// The facts the `$warning`s of the Verilog-A source, and their like, state of an
    /// instance with `parameters` at `temperature`, in kelvins, in the order it states
    /// them.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when a parameter is out of range.
    pub fn warnings<R: Real>(
        parameters: &ConstructsParameters<R>,
        _: R,
    ) -> Result<Vec<String>, ParameterError> {
        ParameterError::positive("conductance", parameters.conductance)?;
        ParameterError::positive("saturation", parameters.saturation)?;
        if !(parameters.saturation.value()..).contains(&parameters.saturation_max.value()) {
            return Err(ParameterError(format!("saturation_max must be at least {:?}, not {:?}", parameters.saturation.value(), parameters.saturation_max.value())));
        }
        ParameterError::positive("scale", parameters.scale)?;
        let mut found = Vec::new();
        if parameters.saturation.value() > 10.0 {
            found.push(format!("a saturation of {:?} V lies beyond the device's range", parameters.saturation.value()));
        }
        Ok(found)
    }
}
