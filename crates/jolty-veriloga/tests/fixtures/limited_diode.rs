//! The `diode` kernel, compiled by jolty-veriloga from `crates/jolty-veriloga/tests/fixtures/limited_diode.va`.
//!
//! Written by `cargo run -p jolty-veriloga --bin compile`, never by hand. Original source notice:
//!
//! ```text
//! jolty's junction diode at its defaults, its steps limited by the simulator's own
//! `"pnjlim"`, jolty's limiting of a junction, as the hand-written diode in
//! `crates/jolty-models/src/kernel/diode.rs` limits them: Copyright 1985 - 2018,
//! Regents of the University of California and others. Used under ngspice's Modified BSD license, which
//! the repository's `NOTICE` reproduces.
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
    BOLTZMANN, Branch, BranchKind, Dual, ELEMENTARY_CHARGE, Interface, Kernel, Layout, Lift, Local,
    ParameterError, Real, Stamp, limit_junction,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The `diode` model, compiled from its Verilog-A source.
#[derive(Debug, Clone, Copy)]
pub struct Diode;

/// The parameters of a [`Diode`], with its Verilog-A source's defaults.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift)]
#[serde(default, deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct DiodeParameters<R: Real = f64> {
    /// The saturation current, in A; default 0.00000000000001.
    pub saturation_current: R,
}

impl<R: Real> Default for DiodeParameters<R> {
    fn default() -> Self {
        Self {
            saturation_current: R::constant(1e-14),
        }
    }
}

/// A [`Diode`]'s numbers at its temperature: what depends on no probe.
#[derive(Debug, Clone, Copy)]
pub struct PreparedDiode<R: Real> {
    slope: R,
    critical: R,
    h2: R,
    saturation_current: R,
    h4: R,
}

/// Where a [`Diode`]'s contributions go among its local unknowns, and the conditions
/// that chose them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiodeShape {
    anode_by_anode: usize,
    anode_by_cathode: usize,
    cathode_by_anode: usize,
    cathode_by_cathode: usize,
}

/// What a [`Diode`] remembers of its last evaluation: each limited voltage, which the next is limited from.
#[derive(Debug, Clone, Copy, Default)]
pub struct DiodeMemory {
    limited0: f64,
}

impl Kernel for Diode {
    const NAME: &'static str = "diode";
    type Parameters<R: Real> = DiodeParameters<R>;
    type Shape = DiodeShape;
    type Prepared<R: Real> = PreparedDiode<R>;
    type Memory = DiodeMemory;

    fn interface<R: Real>(_: &DiodeParameters<R>) -> Interface {
        Interface::new(&["anode", "cathode"], &[Branch::new(0, 1, BranchKind::Conductive)])
    }

    fn prepare<R: Real>(
        parameters: &DiodeParameters<R>,
        temperature: R,
    ) -> Result<PreparedDiode<R>, ParameterError> {
        struct Locals<R: Real> {
            slope: R,
            critical: R,
        }
        ParameterError::positive("saturation_current", parameters.saturation_current)?;
        let mut v = Locals { slope: R::constant(0.0), critical: R::constant(0.0) };
        v.slope = temperature * BOLTZMANN / ELEMENTARY_CHARGE;
        v.critical = v.slope * (v.slope / (parameters.saturation_current * std::f64::consts::SQRT_2)).ln();
        Ok(PreparedDiode {
            slope: v.slope,
            critical: v.critical,
            h2: v.slope * 3.0,
            saturation_current: parameters.saturation_current,
            h4: R::constant(1.0) / v.slope,
        })
    }

    fn layout<R: Real>(
        _: &DiodeParameters<R>,
        _: &PreparedDiode<R>,
        layout: &mut Layout,
    ) -> Result<DiodeShape, ParameterError> {
        Ok(DiodeShape {
            anode_by_anode: layout.entry(0, 0),
            anode_by_cathode: layout.entry(0, 1),
            cathode_by_anode: layout.entry(1, 0),
            cathode_by_cathode: layout.entry(1, 1),
        })
    }

    fn evaluate<R: Real>(
        shape: &DiodeShape,
        prepared: &PreparedDiode<R>,
        memory: &mut DiodeMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        struct Locals<R: Real> {
            vd: Dual<R, 1>,
            id: Dual<R, 1>,
            ratio: Dual<R, 1>,
            anode_cathode_flow: Dual<R, 1>,
            applied0: f64,
            limited0: f64,
        }
        let mut v = Locals {
            vd: Dual::number(0.0),
            id: Dual::number(0.0),
            ratio: Dual::number(0.0),
            anode_cathode_flow: Dual::number(0.0),
            applied0: 0.0,
            limited0: 0.0,
        };
        let applied0 = at.value(0) - at.value(1);
        let limited0 = match at.limiting.then(|| limit_junction(applied0, memory.limited0, prepared.slope.value(), prepared.critical.value())).flatten() {
            Some(limited) => {
                stamp.mark_limited();
                limited
            }
            None => applied0,
        };
        memory.limited0 = limited0;
        v.applied0 = applied0;
        v.limited0 = limited0;
        let t0000 = Dual::variable((at.unknown::<R>(0) - at.unknown::<R>(1)).with_value(limited0));
        v.vd = t0000;
        if v.vd.value.value() >= -prepared.h2.value() {
            v.id = (v.vd.chain((v.vd.value / prepared.slope, prepared.h4)).exp() - 1.0).scale(prepared.saturation_current);
        } else {
            v.ratio = Dual::constant(prepared.h2) / (v.vd * std::f64::consts::E);
            v.id = (v.ratio * v.ratio * v.ratio + 1.0).scale(-prepared.saturation_current);
        }
        v.anode_cathode_flow = v.anode_cathode_flow + (v.id + at.gmin * v.vd);
        let t0001 = v.anode_cathode_flow.extend([v.applied0 - v.limited0]);
        stamp.add_residual(0, t0001);
        stamp.add_residual(1, -t0001);
        let [t0002] = v.anode_cathode_flow.gradient;
        stamp.add_conductance(shape.anode_by_anode, t0002);
        stamp.add_conductance(shape.anode_by_cathode, -t0002);
        stamp.add_conductance(shape.cathode_by_anode, -t0002);
        stamp.add_conductance(shape.cathode_by_cathode, t0002);
    }
}
