//! The `initial_events` kernel, compiled by jolty-veriloga from `crates/jolty-veriloga/tests/fixtures/initial_events.va`.
//!
//! Written by `cargo run -p jolty-veriloga --bin compile`, never by hand. Original source notice:
//!
//! ```text
//! A resistor of whole fingers, written to exercise the events and tasks of the CMC's
//! models: code under @(initial_step) that rounds with floor and ceil, $finish, $strobe
//! of a value no format takes, a $strobe of values that depend on the probes, and two
//! variables whose names meet in snake case.
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

/// The `initial_events` model, compiled from its Verilog-A source.
#[derive(Debug, Clone, Copy)]
pub struct InitialEvents;

/// The parameters of a [`InitialEvents`], with its Verilog-A source's defaults.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift)]
#[serde(default, deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct InitialEventsParameters<R: Real = f64> {
    /// The resistance of one finger, in Ohm; default 1000.
    pub r: R,
    /// The fingers, rounded to the nearest whole and up; default 2.4.
    pub fingers: R,
    /// The largest voltage the device allows, in V; default 1.
    pub vmax: R,
    /// Whether the device refuses to be made; default 0.
    pub refuse: i32,
}

impl<R: Real> Default for InitialEventsParameters<R> {
    fn default() -> Self {
        Self {
            r: R::constant(1000.0),
            fingers: R::constant(2.4),
            vmax: R::constant(1.0),
            refuse: 0,
        }
    }
}

/// A [`InitialEvents`]'s numbers at its temperature: what depends on no probe.
#[derive(Debug, Clone, Copy)]
pub struct PreparedInitialEvents<R: Real> {
    g: R,
    vmax: R,
}

/// Where a [`InitialEvents`]'s contributions go among its local unknowns, and the conditions
/// that chose them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InitialEventsShape {
    a_by_a: usize,
    a_by_b: usize,
    b_by_a: usize,
    b_by_b: usize,
}

/// What a [`InitialEvents`] remembers of its last evaluation: each warning that depends on the probes, once found.
#[derive(Debug, Clone, Copy, Default)]
pub struct InitialEventsMemory {
    /// Whether an evaluation found each warning that depends on the probes, as
    /// [`InitialEvents::warned`] names them.
    warned: [bool; 1],
    /// The values each of those warnings formats, as the last evaluation that
    /// found it read them.
    warned_values: [f64; 3],
}

impl Kernel for InitialEvents {
    const NAME: &'static str = "initial_events";
    type Parameters<R: Real> = InitialEventsParameters<R>;
    type Shape = InitialEventsShape;
    type Prepared<R: Real> = PreparedInitialEvents<R>;
    type Memory = InitialEventsMemory;

    fn interface<R: Real>(_: &InitialEventsParameters<R>) -> Interface {
        Interface::new(&["a", "b"], &[Branch::new(0, 1, BranchKind::Conductive)])
    }

    fn prepare<R: Real>(
        parameters: &InitialEventsParameters<R>,
        _: R,
    ) -> Result<PreparedInitialEvents<R>, ParameterError> {
        struct Locals<R: Real> {
            g: R,
        }
        ParameterError::positive("r", parameters.r)?;
        ParameterError::positive("fingers", parameters.fingers)?;
        ParameterError::positive("vmax", parameters.vmax)?;
        if !(0.0..=1.0).contains(&f64::from(parameters.refuse)) {
            return Err(ParameterError(format!("refuse is at least 0 and at most 1, not {:?}", parameters.refuse)));
        }
        let mut v = Locals { g: R::constant(0.0) };
        let t0000 = parameters.fingers + 0.5;
        let t0001 = t0000.value();
        let t0002 = parameters.fingers.value();
        v.g = (t0000.chain((t0001.floor(), 0.0, 0.0)) + parameters.fingers.chain((t0002.ceil(), 0.0, 0.0))) / parameters.r;
        if parameters.refuse != 0 {
            return Err(ParameterError("the model calls $finish".to_owned()));
        }
        Ok(PreparedInitialEvents { g: v.g, vmax: parameters.vmax })
    }

    fn layout<R: Real>(
        _: &InitialEventsParameters<R>,
        _: &PreparedInitialEvents<R>,
        layout: &mut Layout,
    ) -> Result<InitialEventsShape, ParameterError> {
        Ok(InitialEventsShape {
            a_by_a: layout.entry(0, 0),
            a_by_b: layout.entry(0, 1),
            b_by_a: layout.entry(1, 0),
            b_by_b: layout.entry(1, 1),
        })
    }

    fn evaluate<R: Real>(
        shape: &InitialEventsShape,
        prepared: &PreparedInitialEvents<R>,
        memory: &mut InitialEventsMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        struct Locals<R: Real> {
            vab: Dual<R, 1>,
            vab2: Dual<R, 1>,
            voltage_a_b: Dual<R, 1>,
            a_b_flow: Dual<R, 1>,
        }
        let mut v = Locals {
            vab: Dual::number(0.0),
            vab2: Dual::number(0.0),
            voltage_a_b: Dual::number(0.0),
            a_b_flow: Dual::number(0.0),
        };
        v.voltage_a_b = Dual::variable(at.unknown::<R>(0) - at.unknown::<R>(1));
        v.vab = v.voltage_a_b;
        v.vab2 = v.vab.scale(prepared.g);
        let t0000 = if v.vab.value.value() < 0.0 { -v.vab } else { v.vab };
        if t0000.value.value() > prepared.vmax.value() {
            memory.warned[0] = true;
            memory.warned_values[0] = v.vab.value.value();
            memory.warned_values[1] = prepared.vmax.value();
            let t0001 = if v.vab.value.value() < 0.0 { -v.vab } else { v.vab };
            memory.warned_values[2] = t0001.plus(-prepared.vmax).value.value();
        }
        v.a_b_flow = v.a_b_flow + v.vab2;
        let t0002 = v.a_b_flow.value;
        stamp.add_residual(0, t0002);
        stamp.add_residual(1, -t0002);
        let [t0003] = v.a_b_flow.gradient;
        stamp.add_conductance(shape.a_by_a, t0003);
        stamp.add_conductance(shape.a_by_b, -t0003);
        stamp.add_conductance(shape.b_by_a, -t0003);
        stamp.add_conductance(shape.b_by_b, t0003);
    }
}

impl InitialEvents {
    /// The facts the `$warning`s of the Verilog-A source, and their like, state of an
    /// instance with `parameters` at `temperature`, in kelvins, in the order it states
    /// them.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when a parameter is out of range.
    pub fn warnings<R: Real>(
        parameters: &InitialEventsParameters<R>,
        _: R,
    ) -> Result<Vec<String>, ParameterError> {
        ParameterError::positive("r", parameters.r)?;
        ParameterError::positive("fingers", parameters.fingers)?;
        ParameterError::positive("vmax", parameters.vmax)?;
        if !(0.0..=1.0).contains(&f64::from(parameters.refuse)) {
            return Err(ParameterError(format!("refuse is at least 0 and at most 1, not {:?}", parameters.refuse)));
        }
        let mut found = Vec::new();
        if parameters.fingers.value() > 10.0 {
            found.push(format!("many fingers: {:?}", parameters.fingers.value()));
        }
        if parameters.refuse != 0 {
            found.push("refused, as asked".to_owned());
            return Err(ParameterError("the model calls $finish".to_owned()));
        }
        Ok(found)
    }

    /// The message of each warning that depends on the probes that an evaluation
    /// of an instance, whose memory is `memory`, has found since the memory was made,
    /// with the values the last evaluation that found it read.
    pub fn warned(memory: &InitialEventsMemory) -> impl Iterator<Item = String> {
        [format!("V(a,b) = {:?} exceeds {:?} by {:?}", memory.warned_values[0], memory.warned_values[1], memory.warned_values[2])].into_iter().zip(memory.warned).filter_map(|(message, found)| found.then_some(message))
    }
}
