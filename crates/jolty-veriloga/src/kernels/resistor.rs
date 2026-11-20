//! The `resistor` kernel, compiled by jolty-veriloga from `crates/jolty-veriloga/models/resistor.va`.
//!
//! Written by `cargo run -p jolty-veriloga --bin compile`, never by hand. Original source notice:
//!
//! ```text
//! The resistor of jolty's library, written in Verilog-A after its hand-written kernel in
//! `crates/jolty-models/src/kernel/passive.rs`, operation for operation: Ohm's law, its
//! resistance drifting with temperature and changing with its voltage, and its thermal
//! and flicker noise, as ngspice's resistor makes them.
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
    Branch, BranchKind, Dual, Interface, Kernel, Layout, Lift, Local, Nature, NoiseSource,
    ParameterError, Real, SMALL_RESISTANCE, Stamp, UnknownIndex,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A local index no instance has: a write to it goes nowhere, and a read of it reads
/// zero, as `Stamp` and `Local` promise.
const NOWHERE: usize = usize::MAX;

/// The `resistor` model, compiled from its Verilog-A source.
#[derive(Debug, Clone, Copy)]
pub struct Resistor;

/// The parameters of a [`Resistor`], with its Verilog-A source's defaults.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift)]
#[serde(default, deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct ResistorParameters<R: Real = f64> {
    /// The resistance at the nominal temperature, in ohms; default 1.
    pub resistance: R,
    /// The linear temperature coefficient tc1, in 1/K; default 0.
    pub temperature_coefficient: R,
    /// The quadratic temperature coefficient tc2, in 1/K^2; default 0.
    pub quadratic_temperature_coefficient: R,
    /// The exponential temperature coefficient tce: given, the resistance is multiplied by 1.01^(tce `dT`) instead of the polynomial, in %/K; default 0.
    pub exponential_temperature_coefficient: Option<R>,
    /// The temperature the resistance was measured at, 27 C by default, in K; default 300.15.
    pub nominal_temperature: R,
    /// The linear voltage coefficient vc1, in 1/V; default 0.
    pub voltage_coefficient: R,
    /// The quadratic voltage coefficient vc2, in 1/V^2; default 0.
    pub quadratic_voltage_coefficient: R,
    /// The flicker noise coefficient KF; default 0.
    pub flicker_coefficient: R,
    /// The power of the current in the flicker noise, AF; default 1.
    pub flicker_exponent: R,
}

impl<R: Real> Default for ResistorParameters<R> {
    fn default() -> Self {
        Self {
            resistance: R::constant(1.0),
            temperature_coefficient: R::constant(0.0),
            quadratic_temperature_coefficient: R::constant(0.0),
            exponential_temperature_coefficient: None,
            nominal_temperature: R::constant(300.15),
            voltage_coefficient: R::constant(0.0),
            quadratic_voltage_coefficient: R::constant(0.0),
            flicker_coefficient: R::constant(0.0),
            flicker_exponent: R::constant(1.0),
        }
    }
}

/// A [`Resistor`]'s numbers at its temperature: what depends on no probe.
#[derive(Debug, Clone, Copy)]
pub struct PreparedResistor<R: Real> {
    voltage_coefficient: R,
    quadratic_voltage_coefficient: R,
    r: R,
    h3: R,
    h4: bool,
    flicker_coefficient: R,
    flicker_exponent: R,
    /// The conditions that shape the kernel.
    conditions: [bool; 1],
}

/// Where a [`Resistor`]'s contributions go among its local unknowns, and the conditions
/// that chose them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResistorShape {
    current: usize,
    a_by_a: usize,
    a_by_b: usize,
    a_by_current: usize,
    b_by_a: usize,
    b_by_b: usize,
    b_by_current: usize,
    current_by_a: usize,
    current_by_b: usize,
    current_by_current: usize,
    conditions: [bool; 1],
    active: [bool; 2],
}

impl Kernel for Resistor {
    const NAME: &'static str = "resistor";
    type Parameters<R: Real> = ResistorParameters<R>;
    type Shape = ResistorShape;
    type Prepared<R: Real> = PreparedResistor<R>;
    type Memory = ();

    fn interface<R: Real>(_: &ResistorParameters<R>) -> Interface {
        Interface::new(&["a", "b"], &[Branch::new(0, 1, BranchKind::Conductive)])
    }

    fn linear(shape: &ResistorShape, prepared: &PreparedResistor<f64>) -> bool {
        (shape.conditions[0] && prepared.r == 0.0) || (prepared.voltage_coefficient == 0.0 && prepared.quadratic_voltage_coefficient == 0.0)
    }

    fn prepare<R: Real>(
        parameters: &ResistorParameters<R>,
        temperature: R,
    ) -> Result<PreparedResistor<R>, ParameterError> {
        struct Locals<R: Real> {
            difference: R,
            factor: R,
            r: R,
        }
        ParameterError::positive("resistance", parameters.resistance)?;
        ParameterError::positive("nominal_temperature", parameters.nominal_temperature)?;
        ParameterError::non_negative("flicker_coefficient", parameters.flicker_coefficient)?;
        ParameterError::positive("flicker_exponent", parameters.flicker_exponent)?;
        let mut v = Locals {
            difference: R::constant(0.0),
            factor: R::constant(0.0),
            r: R::constant(0.0),
        };
        v.difference = temperature - parameters.nominal_temperature;
        if parameters.exponential_temperature_coefficient.is_some() {
            if parameters.temperature_coefficient.value() != 0.0 || parameters.quadratic_temperature_coefficient.value() != 0.0 {
                return Err(ParameterError("an exponential temperature coefficient excludes the linear and quadratic ones".to_owned()));
            }
            v.factor = R::constant(1.01).pow(parameters.exponential_temperature_coefficient.unwrap_or(R::constant(0.0)) * v.difference);
        } else {
            v.factor = (parameters.temperature_coefficient + parameters.quadratic_temperature_coefficient * v.difference) * v.difference + 1.0;
        }
        if v.factor.value() == 0.0 {
            return Err(ParameterError(format!("the temperature coefficients scale the resistance to zero at {:?} K", temperature.value())));
        }
        v.r = parameters.resistance * v.factor;
        Ok(PreparedResistor {
            voltage_coefficient: parameters.voltage_coefficient,
            quadratic_voltage_coefficient: parameters.quadratic_voltage_coefficient,
            r: v.r,
            h3: temperature * 5.522_596e-23,
            h4: parameters.flicker_coefficient.value() > 0.0,
            flicker_coefficient: parameters.flicker_coefficient,
            flicker_exponent: parameters.flicker_exponent,
            conditions: [v.r.abs().value() < SMALL_RESISTANCE],
        })
    }

    fn layout<R: Real>(
        _: &ResistorParameters<R>,
        prepared: &PreparedResistor<R>,
        layout: &mut Layout,
    ) -> Result<ResistorShape, ParameterError> {
        let mut active = [false; 2];
        if prepared.conditions[0] {
            active[0] = true;
        } else {
            active[1] = true;
        }
        let current = if active[0] { layout.internal("current", Nature::Current) } else { NOWHERE };
        Ok(ResistorShape {
            current,
            a_by_a: if active[1] { layout.entry(0, 0) } else { NOWHERE },
            a_by_b: if active[1] { layout.entry(0, 1) } else { NOWHERE },
            a_by_current: if active[0] { layout.entry(0, current) } else { NOWHERE },
            b_by_a: if active[1] { layout.entry(1, 0) } else { NOWHERE },
            b_by_b: if active[1] { layout.entry(1, 1) } else { NOWHERE },
            b_by_current: if active[0] { layout.entry(1, current) } else { NOWHERE },
            current_by_a: if active[0] { layout.entry(current, 0) } else { NOWHERE },
            current_by_b: if active[0] { layout.entry(current, 1) } else { NOWHERE },
            current_by_current: if active[0] { layout.entry(current, current) } else { NOWHERE },
            conditions: prepared.conditions,
            active,
        })
    }

    fn evaluate<R: Real>(
        shape: &ResistorShape,
        prepared: &PreparedResistor<R>,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        struct Locals<R: Real> {
            v: Dual<R, 1>,
            f: Dual<R, 1>,
            voltage_a_b: Dual<R, 1>,
            current_current: Dual<R, 1>,
            current_potential: Dual<R, 2>,
            a_b_flow: Dual<R, 1>,
        }
        let mut v = Locals {
            v: Dual::number(0.0),
            f: Dual::number(0.0),
            voltage_a_b: Dual::number(0.0),
            current_current: Dual::number(0.0),
            current_potential: Dual::number(0.0),
            a_b_flow: Dual::number(0.0),
        };
        v.voltage_a_b = Dual::variable(at.unknown::<R>(0) - at.unknown::<R>(1));
        v.current_current = Dual::variable(at.unknown::<R>(shape.current));
        v.v = v.voltage_a_b;
        v.f = v.v.scale(prepared.voltage_coefficient) + (v.v * v.v).scale(prepared.quadratic_voltage_coefficient) + 1.0;
        if shape.conditions[0] {
            v.current_potential = v.current_potential + v.f.scale(prepared.r).embed::<2>([0]) * v.current_current.embed::<2>([1]);
        } else {
            v.a_b_flow = v.a_b_flow + Dual::number(1.0) / v.f.scale(prepared.r) * v.v;
        }
        if shape.active[0] {
            let current = at.unknown::<R>(shape.current);
            stamp.add_residual(0, current);
            stamp.add_residual(1, -current);
            stamp.add_conductance(shape.a_by_current, R::constant(1.0));
            stamp.add_conductance(shape.b_by_current, R::constant(-1.0));
            stamp.add_residual(shape.current, at.unknown::<R>(0) - at.unknown::<R>(1) - v.current_potential.value);
            stamp.add_conductance(shape.current_by_a, R::constant(1.0));
            stamp.add_conductance(shape.current_by_b, R::constant(-1.0));
            let [t0000, t0001] = v.current_potential.gradient;
            stamp.add_conductance(shape.current_by_a, -t0000);
            stamp.add_conductance(shape.current_by_b, t0000);
            stamp.add_conductance(shape.current_by_current, -t0001);
        }
        if shape.active[1] {
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

    fn noise<R: Real>(
        shape: &ResistorShape,
        prepared: &PreparedResistor<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource<R>>,
    ) {
        struct Locals<R: Real> {
            v: R,
            f: R,
            i: R,
            voltage_a_b: R,
            current_current: R,
        }
        let mut v = Locals {
            v: R::constant(0.0),
            f: R::constant(0.0),
            i: R::constant(0.0),
            voltage_a_b: R::constant(0.0),
            current_current: R::constant(0.0),
        };
        v.voltage_a_b = at.unknown::<R>(0) - at.unknown::<R>(1);
        v.current_current = at.unknown::<R>(shape.current);
        v.v = v.voltage_a_b;
        v.f = prepared.voltage_coefficient * v.v + prepared.quadratic_voltage_coefficient * (v.v * v.v) + 1.0;
        if shape.conditions[0] {
            sources.push(NoiseSource::white("thermal", [Some(UnknownIndex::new(shape.current)), None], prepared.h3 * (prepared.r * v.f).abs()));
            v.i = v.current_current;
        } else {
            sources.push(NoiseSource::white("thermal", [Some(UnknownIndex::new(0)), Some(UnknownIndex::new(1))], prepared.h3 * (R::constant(1.0) / (prepared.r * v.f)).abs()));
            v.i = R::constant(1.0) / (prepared.r * v.f) * v.v;
        }
        if prepared.h4 {
            sources.push(NoiseSource {
                name: "flicker",
                equations: [Some(UnknownIndex::new(0)), Some(UnknownIndex::new(1))],
                white: R::constant(0.0),
                flicker: prepared.flicker_coefficient * v.i.abs().pow(prepared.flicker_exponent),
                exponent: R::constant(1.0),
            });
        }
    }
}
