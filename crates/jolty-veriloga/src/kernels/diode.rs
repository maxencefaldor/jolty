//! The `diode` kernel, compiled by jolty-veriloga from `crates/jolty-veriloga/models/diode.va`.
//!
//! Written by `cargo run -p jolty-veriloga --bin compile`, never by hand. Its source says:
//!
//! ```text
//! The junction diode of jolty's library, written in Verilog-A after its hand-written
//! kernel in `crates/jolty-models/src/kernel/diode.rs` and `junction.rs`, operation
//! for operation. Those derive from ngspice 47's implementation of the diode, in
//! `src/spicelib/devices/dio/`, and its limiting, in `devsup.c`:
//! Copyright 1985 - 2018, Regents of the University of California and others. Used under
//! ngspice's Modified BSD license, which the repository's `NOTICE` reproduces.
//! ```

#![cfg_attr(
    not(test),
    deny(
        clippy::indexing_slicing,
        clippy::panic,
        reason = "a kernel never panics, and reads fixed arrays through `get`"
    )
)]

use std::cmp::Ordering;

use jolty_models::{BOLTZMANN, Branch, BranchKind, Dual, ELEMENTARY_CHARGE, Interface, Kernel, Layout, Local, Nature, NoiseSource, ParameterError, Real, SMALL_RESISTANCE, Stamp, UnknownIndex};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A local index no instance has: a write to it goes nowhere, and a read of it reads
/// zero, as `Stamp` and `Local` promise.
const NOWHERE: usize = usize::MAX;

/// The `diode` model, compiled from its Verilog-A source.
#[derive(Debug, Clone, Copy)]
pub struct Diode;

/// The parameters of a [`Diode`], with its Verilog-A source's defaults.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(default, deny_unknown_fields, bound = "")]
pub struct DiodeParameters<R: Real = f64> {
    /// The saturation current, in A; default 0.00000000000001.
    pub saturation_current: R,
    /// The emission coefficient; default 1.
    pub emission_coefficient: R,
    /// The series resistance, in ohms; default 0.
    pub series_resistance: R,
    /// The zero-bias junction capacitance, in F; default 0.
    pub junction_capacitance: R,
    /// The junction potential, in V; default 1.
    pub junction_potential: R,
    /// The grading coefficient of the junction; default 0.5.
    pub grading_coefficient: R,
    /// The fraction of the junction potential above which the depletion capacitance is extended linearly; default 0.5.
    pub depletion_coefficient: R,
    /// The transit time, in s; default 0.
    pub transit_time: R,
    /// The reverse breakdown voltage; without it, the junction never breaks down, in V; default 1.
    pub breakdown_voltage: Option<R>,
    /// The current at the breakdown voltage, in A; default 0.001.
    pub breakdown_current: R,
    /// The breakdown emission coefficient, the emission coefficient by default; by default, as its source computes it from the parameters before it.
    pub breakdown_emission_coefficient: Option<R>,
    /// How fast the breakdown voltage falls with temperature, in V/K; default 0.
    pub breakdown_temperature_coefficient: R,
    /// The energy gap that scales the saturation current with temperature, silicon's by default, in V; default 1.11.
    pub energy_gap: R,
    /// The exponent of the temperature ratio in the saturation current's scaling; default 3.
    pub saturation_current_temperature_exponent: R,
    /// The temperature the parameters were measured at, 27 C by default, in K; default 300.15.
    pub nominal_temperature: R,
    /// The flicker noise coefficient KF; default 0.
    pub flicker_coefficient: R,
    /// The power of the current in the flicker noise, AF; default 1.
    pub flicker_exponent: R,
}

impl<R: Real> Default for DiodeParameters<R> {
    fn default() -> Self {
        Self {
            saturation_current: R::constant(1e-14),
            emission_coefficient: R::constant(1.0),
            series_resistance: R::constant(0.0),
            junction_capacitance: R::constant(0.0),
            junction_potential: R::constant(1.0),
            grading_coefficient: R::constant(0.5),
            depletion_coefficient: R::constant(0.5),
            transit_time: R::constant(0.0),
            breakdown_voltage: None,
            breakdown_current: R::constant(0.001),
            breakdown_emission_coefficient: None,
            breakdown_temperature_coefficient: R::constant(0.0),
            energy_gap: R::constant(1.11),
            saturation_current_temperature_exponent: R::constant(3.0),
            nominal_temperature: R::constant(300.15),
            flicker_coefficient: R::constant(0.0),
            flicker_exponent: R::constant(1.0),
        }
    }
}

/// A [`Diode`]'s numbers at its temperature: what depends on no probe.
#[derive(Debug, Clone, Copy)]
pub struct PreparedDiode<R: Real> {
    series_resistance: R,
    h1: R,
    slope: R,
    critical: R,
    breakdown: i32,
    matched: R,
    breakdown_slope: R,
    h7: bool,
    h8: R,
    h9: R,
    h10: R,
    h11: R,
    h12: R,
    saturation: R,
    h14: R,
    h15: R,
    h16: bool,
    threshold: R,
    h18: R,
    h19: R,
    potential2: R,
    h21: R,
    h22: R,
    capacitance: R,
    charge_at_threshold: R,
    offset: R,
    h26: R,
    h27: R,
    divisor: R,
    h29: R,
    transit_time: R,
    h31: R,
    h32: R,
    h33: f64,
    h34: bool,
    flicker_coefficient: R,
    flicker_exponent: R,
    /// The conditions that shape the kernel.
    conditions: [bool; 2],
}

/// Where a [`Diode`]'s contributions go among its local unknowns, and the conditions
/// that chose them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiodeShape {
    junction: usize,
    junction_current: usize,
    anode_by_anode: usize,
    anode_by_junction: usize,
    anode_by_junction_current: usize,
    cathode_by_cathode: usize,
    cathode_by_junction: usize,
    junction_by_anode: usize,
    junction_by_cathode: usize,
    junction_by_junction: usize,
    junction_by_junction_current: usize,
    junction_current_by_anode: usize,
    junction_current_by_junction: usize,
    junction_current_by_junction_current: usize,
    conditions: [bool; 2],
    active: [bool; 3],
}

/// What a [`Diode`] remembers of its last evaluation: each limited voltage, which the next is limited from.
#[derive(Debug, Clone, Copy, Default)]
pub struct DiodeMemory {
    limited0: f64,
}

/// The Verilog-A analog function `junction_limit`, which limits a voltage's step.
fn junction_limit(new: f64, old: f64, slope: f64, critical: f64, breaks: f64, matched: f64, breakdown_slope: f64) -> f64 {
    struct Locals {
        reflected: f64,
        last: f64,
        step: f64,
        limited: f64,
        floor: f64,
        reverse: i32,
        junction_limit: f64,
    }
    let mut v = Locals { reflected: 0.0, last: 0.0, step: 0.0, limited: 0.0, floor: 0.0, reverse: 0, junction_limit: 0.0 };
    v.reverse = i32::from(breaks != 0.0 && new < (10.0 * breakdown_slope - matched).min(0.0));
    if v.reverse == 0 {
        v.reflected = new;
        v.last = old;
        v.step = slope;
    } else {
        v.reflected = -(new + matched);
        v.last = -(old + matched);
        v.step = breakdown_slope;
    }
    v.limited = v.reflected;
    if v.reflected > critical && (v.reflected - v.last).abs() > 2.0 * v.step {
        if v.last > 0.0 {
            if 1.0 + (v.reflected - v.last) / v.step > 0.0 {
                v.limited = v.last + v.step * (1.0 + (v.reflected - v.last) / v.step).ln();
            } else {
                v.limited = critical;
            }
        } else {
            v.limited = v.step * (v.reflected / v.step).ln();
        }
    } else if v.reflected < 0.0 {
        if v.last > 0.0 {
            v.floor = -v.last - 1.0;
        } else {
            v.floor = 2.0 * v.last - 1.0;
        }
        if v.reflected < v.floor {
            v.limited = v.floor;
        }
    }
    if matches!(v.limited.partial_cmp(&v.reflected), Some(Ordering::Equal)) {
        v.junction_limit = new;
    } else if v.reverse == 0 {
        v.junction_limit = v.limited;
    } else {
        v.junction_limit = -(v.limited + matched);
    }
    v.junction_limit
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
            vt: R,
            saturation: R,
            slope: R,
            critical: R,
            intrinsic: R,
            potential: R,
            capacitance: R,
            threshold: R,
            charge_at_threshold: R,
            divisor: R,
            offset: R,
            breakdown_slope: R,
            unmatched: R,
            matched: R,
            breakdown: i32,
            iteration: i32,
            settled: i32,
            potential2: R,
        }
        ParameterError::positive("saturation_current", parameters.saturation_current)?;
        ParameterError::positive("emission_coefficient", parameters.emission_coefficient)?;
        ParameterError::non_negative("series_resistance", parameters.series_resistance)?;
        ParameterError::non_negative("junction_capacitance", parameters.junction_capacitance)?;
        ParameterError::positive("junction_potential", parameters.junction_potential)?;
        if !(0.0..1.0).contains(&parameters.grading_coefficient.value()) {
    return Err(ParameterError(format!("grading_coefficient must be at least 0 and below 1, not {:?}", parameters.grading_coefficient.value())));
}
        if !(0.0..1.0).contains(&parameters.depletion_coefficient.value()) {
    return Err(ParameterError(format!("depletion_coefficient must be at least 0 and below 1, not {:?}", parameters.depletion_coefficient.value())));
}
        ParameterError::non_negative("transit_time", parameters.transit_time)?;
        ParameterError::positive("breakdown_voltage", parameters.breakdown_voltage.unwrap_or(R::constant(1.0)))?;
        ParameterError::positive("breakdown_current", parameters.breakdown_current)?;
        ParameterError::positive("breakdown_emission_coefficient", parameters.breakdown_emission_coefficient.unwrap_or(parameters.emission_coefficient))?;
        ParameterError::non_negative("energy_gap", parameters.energy_gap)?;
        ParameterError::positive("nominal_temperature", parameters.nominal_temperature)?;
        ParameterError::non_negative("flicker_coefficient", parameters.flicker_coefficient)?;
        ParameterError::positive("flicker_exponent", parameters.flicker_exponent)?;
        let mut v = Locals { vt: R::constant(0.0), saturation: R::constant(0.0), slope: R::constant(0.0), critical: R::constant(0.0), intrinsic: R::constant(0.0), potential: R::constant(0.0), capacitance: R::constant(0.0), threshold: R::constant(0.0), charge_at_threshold: R::constant(0.0), divisor: R::constant(0.0), offset: R::constant(0.0), breakdown_slope: R::constant(0.0), unmatched: R::constant(0.0), matched: R::constant(0.0), breakdown: 0, iteration: 0, settled: 0, potential2: R::constant(0.0) };
        if temperature.value() <= 0.0 {
            return Err(ParameterError(format!("a junction's temperature lies above 0 K, not at {:?} K", temperature.value())));
        }
        v.vt = temperature * BOLTZMANN / ELEMENTARY_CHARGE;
        v.saturation = parameters.saturation_current * (((temperature / parameters.nominal_temperature - 1.0) * parameters.energy_gap / v.vt + parameters.saturation_current_temperature_exponent * (temperature / parameters.nominal_temperature).ln()) / parameters.emission_coefficient).exp();
        v.intrinsic = (parameters.junction_potential - parameters.nominal_temperature * BOLTZMANN / ELEMENTARY_CHARGE * -2.0 * ((parameters.nominal_temperature / 300.15).ln() * 1.5 + ((parameters.nominal_temperature * 0.000_702 * parameters.nominal_temperature / (parameters.nominal_temperature + 1108.0) - 1.16) / (parameters.nominal_temperature * 2.761_298e-23) + 1.345_418_513_059_931_1e20) * ELEMENTARY_CHARGE)) / (parameters.nominal_temperature / 300.15);
        v.potential = v.intrinsic * (temperature / 300.15) + temperature * BOLTZMANN / ELEMENTARY_CHARGE * -2.0 * ((temperature / 300.15).ln() * 1.5 + ((temperature * 0.000_702 * temperature / (temperature + 1108.0) - 1.16) / (temperature * 2.761_298e-23) + 1.345_418_513_059_931_1e20) * ELEMENTARY_CHARGE);
        v.capacitance = parameters.junction_capacitance * (parameters.grading_coefficient * ((temperature - 300.15) * 0.0004 - (v.potential - v.intrinsic) / v.intrinsic) + 1.0) / (parameters.grading_coefficient * ((parameters.nominal_temperature - 300.15) * 0.0004 - (parameters.junction_potential - v.intrinsic) / v.intrinsic) + 1.0);
        v.potential2 = v.potential.min(R::constant(1.0) / parameters.depletion_coefficient);
        v.slope = parameters.emission_coefficient * v.vt;
        v.critical = v.slope * (v.slope / (v.saturation * std::f64::consts::SQRT_2)).ln();
        v.threshold = parameters.depletion_coefficient * v.potential2;
        v.charge_at_threshold = v.potential2 * (R::constant(1.0) - (R::constant(1.0) - parameters.depletion_coefficient).pow(R::constant(1.0) - parameters.grading_coefficient)) / (R::constant(1.0) - parameters.grading_coefficient);
        v.divisor = (R::constant(1.0) - parameters.depletion_coefficient).pow(parameters.grading_coefficient + 1.0);
        v.offset = R::constant(1.0) - parameters.depletion_coefficient * (parameters.grading_coefficient + 1.0);
        v.breakdown = i32::from(parameters.breakdown_voltage.is_some());
        v.breakdown_slope = parameters.breakdown_emission_coefficient.unwrap_or(parameters.emission_coefficient) * v.vt;
        v.matched = parameters.breakdown_voltage.unwrap_or(R::constant(1.0)) - parameters.breakdown_temperature_coefficient * (temperature - parameters.nominal_temperature);
        if v.breakdown != 0 && parameters.breakdown_current.value() >= (v.saturation * v.matched / v.vt).value() {
            v.unmatched = v.matched;
            v.matched = v.unmatched - v.breakdown_slope * (parameters.breakdown_current / v.saturation + 1.0).ln();
            v.iteration = 0;
            v.settled = 0;
            while v.iteration < 25 && v.settled == 0 {
                v.matched = v.unmatched - v.breakdown_slope * (parameters.breakdown_current / v.saturation + 1.0 - v.matched / v.vt).ln();
                v.settled = i32::from((v.saturation * (((v.unmatched - v.matched) / v.breakdown_slope).exp() - 1.0 + v.matched / v.vt) - parameters.breakdown_current).abs().value() <= (parameters.breakdown_current * 1e-12).value());
                v.iteration = v.iteration.wrapping_add(1);
            }
        }
        Ok(PreparedDiode { series_resistance: parameters.series_resistance, h1: R::constant(1.0) / parameters.series_resistance, slope: v.slope, critical: v.critical, breakdown: v.breakdown, matched: v.matched, breakdown_slope: v.breakdown_slope, h7: v.breakdown != 0, h8: v.slope * -3.0, h9: -v.matched, h10: R::constant(1.0) / v.breakdown_slope, h11: -v.saturation, h12: -(v.slope * 3.0), saturation: v.saturation, h14: R::constant(1.0) / v.slope, h15: v.slope * 3.0, h16: v.capacitance.value() > 0.0, threshold: v.threshold, h18: v.capacitance * v.potential2, h19: R::constant(1.0) - parameters.grading_coefficient, potential2: v.potential2, h21: R::constant(1.0) / v.potential2, h22: R::constant(1.0) / (R::constant(1.0) - parameters.grading_coefficient), capacitance: v.capacitance, charge_at_threshold: v.charge_at_threshold, offset: v.offset, h26: parameters.grading_coefficient / (v.potential2 * 2.0), h27: v.threshold * v.threshold, divisor: v.divisor, h29: R::constant(1.0) / v.divisor, transit_time: parameters.transit_time, h31: temperature * 5.522_596e-23 * parameters.series_resistance.abs(), h32: temperature * 5.522_596e-23 * (R::constant(1.0) / parameters.series_resistance).abs(), h33: 3.204_353_268e-19, h34: parameters.flicker_coefficient.value() > 0.0, flicker_coefficient: parameters.flicker_coefficient, flicker_exponent: parameters.flicker_exponent, conditions: [parameters.series_resistance.value() == 0.0, parameters.series_resistance.value() < SMALL_RESISTANCE] })
    }

    fn layout<R: Real>(
        _: &DiodeParameters<R>,
        prepared: &PreparedDiode<R>,
        layout: &mut Layout,
    ) -> Result<DiodeShape, ParameterError> {
        let mut active = [false; 3];
        if prepared.conditions[0] {
            active[2] = true;
        } else if prepared.conditions[1] {
            active[0] = true;
        } else {
            active[1] = true;
        }
        // The leader of each node's group, the ground first, then the nodes in order.
        let leader = |joined: &[usize; 4], mut place: usize| {
            while let Some(&next) = joined.get(place)
                && next != place
            {
                place = next;
            }
            place
        };
        let mut joined = [0, 1, 2, 3];
        if active[2] { let (one, other) = (leader(&joined, 1), leader(&joined, 3)); if let Some(slot) = joined.get_mut(one.max(other)) { *slot = one.min(other); } }
        let mut local = [NOWHERE, 0, 1, NOWHERE];
        local[3] = match leader(&joined, 3) { 3 => layout.internal("junction", Nature::Voltage), leader => local.get(leader).copied().unwrap_or(NOWHERE) };
        let [_, _, _, junction] = local;
        let junction_current = if active[0] { layout.internal("junction_current", Nature::Current) } else { NOWHERE };
        Ok(DiodeShape { junction, junction_current, anode_by_anode: if active[1] { layout.entry(0, 0) } else { NOWHERE }, anode_by_junction: if active[1] { layout.entry(0, junction) } else { NOWHERE }, anode_by_junction_current: if active[0] { layout.entry(0, junction_current) } else { NOWHERE }, cathode_by_cathode: layout.entry(1, 1), cathode_by_junction: layout.entry(1, junction), junction_by_anode: if active[1] { layout.entry(junction, 0) } else { NOWHERE }, junction_by_cathode: layout.entry(junction, 1), junction_by_junction: layout.entry(junction, junction), junction_by_junction_current: if active[0] { layout.entry(junction, junction_current) } else { NOWHERE }, junction_current_by_anode: if active[0] { layout.entry(junction_current, 0) } else { NOWHERE }, junction_current_by_junction: if active[0] { layout.entry(junction_current, junction) } else { NOWHERE }, junction_current_by_junction_current: if active[0] { layout.entry(junction_current, junction_current) } else { NOWHERE }, conditions: prepared.conditions, active })
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
            e: Dual<R, 1>,
            ratio: Dual<R, 1>,
            qd: Dual<R, 1>,
            voltage_anode_junction: Dual<R, 1>,
            current_junction_current: Dual<R, 1>,
            junction_current_potential: Dual<R, 1>,
            anode_junction_flow: Dual<R, 1>,
            junction_cathode_flow: Dual<R, 1>,
            junction_cathode_charge: Dual<R, 1>,
            applied0: f64,
            limited0: f64,
        }
        let mut v = Locals { vd: Dual::number(0.0), id: Dual::number(0.0), e: Dual::number(0.0), ratio: Dual::number(0.0), qd: Dual::number(0.0), voltage_anode_junction: Dual::number(0.0), current_junction_current: Dual::number(0.0), junction_current_potential: Dual::number(0.0), anode_junction_flow: Dual::number(0.0), junction_cathode_flow: Dual::number(0.0), junction_cathode_charge: Dual::number(0.0), applied0: 0.0, limited0: 0.0 };
        v.voltage_anode_junction = Dual { value: R::constant(at.value(0) - at.value(shape.junction)), gradient: [R::constant(1.0)] };
        v.current_junction_current = Dual { value: R::constant(at.value(shape.junction_current)), gradient: [R::constant(1.0)] };
        if shape.conditions[0] {
        } else if shape.conditions[1] {
            v.junction_current_potential = v.junction_current_potential + v.current_junction_current.scale(prepared.series_resistance);
        } else {
            v.anode_junction_flow = v.anode_junction_flow + v.voltage_anode_junction.scale(prepared.h1);
        }
        let applied0 = at.value(shape.junction) - at.value(1);
        let limited0 = if at.limiting { junction_limit(applied0, memory.limited0, prepared.slope.value(), prepared.critical.value(), f64::from(prepared.breakdown), prepared.matched.value(), prepared.breakdown_slope.value()) } else { applied0 };
        if limited0.total_cmp(&applied0).is_ne() {
            stamp.mark_limited();
        }
        memory.limited0 = limited0;
        v.applied0 = applied0;
        v.limited0 = limited0;
        let [t0000] = Dual::<R, 1>::variables([limited0]);
        v.vd = t0000;
        if prepared.h7 && v.vd.value.value() < prepared.h8.value() && v.vd.value.value() < prepared.h9.value() {
            let t0001 = -v.vd.plus(prepared.matched);
            v.e = t0001.chain((t0001.value / prepared.breakdown_slope, prepared.h10)).exp();
            v.id = v.e.scale(prepared.h11);
        } else if v.vd.value.value() >= prepared.h12.value() {
            v.id = (v.vd.chain((v.vd.value / prepared.slope, prepared.h14)).exp() - 1.0).scale(prepared.saturation);
        } else {
            v.ratio = Dual::constant(prepared.h15) / (v.vd * std::f64::consts::E);
            v.id = (v.ratio * v.ratio * v.ratio + 1.0).scale(prepared.h11);
        }
        if prepared.h16 {
            if v.vd.value.value() < prepared.threshold.value() {
                let t0002 = -v.vd.chain((v.vd.value / prepared.potential2, prepared.h21));
                let t0003 = t0002.value.value();
                let t0004 = t0002.value.chain(t0003.ln_1p(), (t0003 + 1.0).recip(), -((t0003 + 1.0) * (t0003 + 1.0)).recip());
                let t0005 = t0002.chain((t0004, (t0002.value + 1.0).recip())).scale(prepared.h19);
                let t0006 = t0005.value.value();
                let t0007 = t0005.value.chain(t0006.exp_m1(), t0006.exp(), t0006.exp());
                let t0008 = (-t0005.chain((t0007, t0005.value.exp()))).scale(prepared.h18);
                v.qd = t0008.chain((t0008.value / prepared.h19, prepared.h22));
            } else {
                let t0009 = v.vd.plus(-prepared.threshold).scale(prepared.offset) + (v.vd * v.vd).plus(-prepared.h27).scale(prepared.h26);
                v.qd = t0009.chain((t0009.value / prepared.divisor, prepared.h29)).plus(prepared.charge_at_threshold).scale(prepared.capacitance);
            }
        }
        v.junction_cathode_flow = v.junction_cathode_flow + (v.id + at.gmin * v.vd);
        v.junction_cathode_charge = v.junction_cathode_charge + (v.qd + v.id.scale(prepared.transit_time));
        if shape.active[0] {
            let current = at.value(shape.junction_current);
            stamp.add_residual(0, R::constant(current));
            stamp.add_residual(shape.junction, R::constant(-current));
            stamp.add_conductance(shape.anode_by_junction_current, R::constant(1.0));
            stamp.add_conductance(shape.junction_by_junction_current, R::constant(-1.0));
            stamp.add_residual(shape.junction_current, R::constant(at.value(0) - at.value(shape.junction)) - v.junction_current_potential.value);
            stamp.add_conductance(shape.junction_current_by_anode, R::constant(1.0));
            stamp.add_conductance(shape.junction_current_by_junction, R::constant(-1.0));
            let [t0010] = v.junction_current_potential.gradient;
            stamp.add_conductance(shape.junction_current_by_junction_current, -t0010);
        }
        if shape.active[1] {
            let t0011 = v.anode_junction_flow.value;
            stamp.add_residual(0, t0011);
            stamp.add_residual(shape.junction, -t0011);
            let [t0012] = v.anode_junction_flow.gradient;
            stamp.add_conductance(shape.anode_by_anode, t0012);
            stamp.add_conductance(shape.anode_by_junction, -t0012);
            stamp.add_conductance(shape.junction_by_anode, -t0012);
            stamp.add_conductance(shape.junction_by_junction, t0012);
        }
        let t0013 = v.junction_cathode_flow.extend([v.applied0 - v.limited0]);
        stamp.add_residual(shape.junction, t0013);
        stamp.add_residual(1, -t0013);
        let [t0014] = v.junction_cathode_flow.gradient;
        stamp.add_conductance(shape.junction_by_junction, t0014);
        stamp.add_conductance(shape.junction_by_cathode, -t0014);
        stamp.add_conductance(shape.cathode_by_junction, -t0014);
        stamp.add_conductance(shape.cathode_by_cathode, t0014);
        let t0015 = v.junction_cathode_charge.extend([v.applied0 - v.limited0]);
        stamp.add_charge(shape.junction, t0015);
        stamp.add_charge(1, -t0015);
        let [t0016] = v.junction_cathode_charge.gradient;
        stamp.add_capacitance(shape.junction_by_junction, t0016);
        stamp.add_capacitance(shape.junction_by_cathode, -t0016);
        stamp.add_capacitance(shape.cathode_by_junction, -t0016);
        stamp.add_capacitance(shape.cathode_by_cathode, t0016);
    }

    fn noise<R: Real>(
        shape: &DiodeShape,
        prepared: &PreparedDiode<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        struct Locals<R: Real> {
            vd: R,
            id: R,
            e: R,
            ratio: R,
        }
        let mut v = Locals { vd: R::constant(0.0), id: R::constant(0.0), e: R::constant(0.0), ratio: R::constant(0.0) };
        if shape.conditions[0] {
        } else if shape.conditions[1] {
            sources.push(NoiseSource::white("series thermal", [Some(UnknownIndex::new(shape.junction_current)), None], prepared.h31.value()));
        } else {
            sources.push(NoiseSource::white("series thermal", [Some(UnknownIndex::new(0)), Some(UnknownIndex::new(shape.junction))], prepared.h32.value()));
        }
        let t0000 = R::constant(at.value(shape.junction) - at.value(1));
        v.vd = t0000;
        if prepared.h7 && v.vd.value() < prepared.h8.value() && v.vd.value() < prepared.h9.value() {
            v.e = (-(prepared.matched + v.vd) / prepared.breakdown_slope).exp();
            v.id = prepared.h11 * v.e;
        } else if v.vd.value() >= prepared.h12.value() {
            v.id = prepared.saturation * ((v.vd / prepared.slope).exp() - 1.0);
        } else {
            v.ratio = prepared.h15 / (v.vd * std::f64::consts::E);
            v.id = prepared.h11 * (v.ratio * v.ratio * v.ratio + 1.0);
        }
        sources.push(NoiseSource::white("shot", [Some(UnknownIndex::new(shape.junction)), Some(UnknownIndex::new(1))], (v.id.abs() * prepared.h33).value()));
        if prepared.h34 {
            sources.push(NoiseSource {
                name: "flicker",
                equations: [Some(UnknownIndex::new(shape.junction)), Some(UnknownIndex::new(1))],
                white: 0.0,
                flicker: (prepared.flicker_coefficient * v.id.abs().pow(prepared.flicker_exponent)).value(),
                exponent: 1.0,
            });
        }
    }
}
