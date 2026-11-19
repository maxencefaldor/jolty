//! The `diode` kernel, compiled by jolty-veriloga from `crates/jolty-veriloga/models/diode.va`.
//!
//! Written by `cargo run -p jolty-veriloga --bin compile`, never by hand. Original source notice:
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

use jolty_models::{
    BOLTZMANN, Branch, BranchKind, Dual, ELEMENTARY_CHARGE, Interface, Kernel, Layout, Lift, Local,
    Nature, NoiseSource, ParameterError, Real, SMALL_RESISTANCE, Stamp, UnknownIndex,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A local index no instance has: a write to it goes nowhere, and a read of it reads
/// zero, as `Stamp` and `Local` promise.
const NOWHERE: usize = usize::MAX;

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
    /// The forward knee current; without it, none, in A; default 1.
    pub forward_knee_current: Option<R>,
    /// The reverse knee current; without it, none, in A; default 1.
    pub reverse_knee_current: Option<R>,
    /// The saturation current of the recombination; without it, none, in A; default 0.
    pub recombination_saturation_current: Option<R>,
    /// The recombination current's emission coefficient; default 2.
    pub recombination_emission_coefficient: R,
    /// The sidewall junction's saturation current, in A; default 0.
    pub sidewall_saturation_current: R,
    /// The sidewall junction's emission coefficient, the emission coefficient by default; by default, as its source computes it from the parameters before it.
    pub sidewall_emission_coefficient: Option<R>,
    /// The sidewall junction's own series resistance; without it, the sidewall sits beside the junction, in ohms; default 0.
    pub sidewall_series_resistance: Option<R>,
    /// The sidewall junction's zero-bias capacitance, in F; default 0.
    pub sidewall_junction_capacitance: R,
    /// The sidewall junction's potential, in V; default 1.
    pub sidewall_junction_potential: R,
    /// The sidewall junction's grading coefficient; default 0.33.
    pub sidewall_grading_coefficient: R,
    /// The fraction of the sidewall junction's potential above which its depletion capacitance is extended linearly; default 0.5.
    pub sidewall_depletion_coefficient: R,
    /// The sidewall junction's forward knee current; without it, none, in A; default 1.
    pub sidewall_forward_knee_current: Option<R>,
    /// The tunneling saturation current, in A; default 0.
    pub tunneling_saturation_current: R,
    /// The sidewall's tunneling saturation current, in A; default 0.
    pub sidewall_tunneling_saturation_current: R,
    /// The tunneling current's emission coefficient; default 30.
    pub tunneling_emission_coefficient: R,
    /// The exponent of the temperature ratio in the tunneling saturation currents' scaling; default 3.
    pub tunneling_saturation_current_temperature_exponent: R,
    /// The factor of the energy gap in the tunneling saturation currents' scaling; default 1.
    pub tunneling_energy_gap_factor: R,
    /// The reverse breakdown voltage; without it, the junction never breaks down, in V; default 1.
    pub breakdown_voltage: Option<R>,
    /// The current at the breakdown voltage, in A; default 0.001.
    pub breakdown_current: R,
    /// The breakdown emission coefficient, the emission coefficient by default; by default, as its source computes it from the parameters before it.
    pub breakdown_emission_coefficient: Option<R>,
    /// How fast the breakdown voltage falls with temperature, in V/K at temperature model 0 and 1/K at 1 and 2; default 0.
    pub breakdown_temperature_coefficient: R,
    /// How the saturation currents and the band gap follow the temperature, SPICE's TLEV; default 0.
    pub temperature_model: i32,
    /// How the junction capacitances and potentials follow the temperature, SPICE's TLEVC; default 0.
    pub capacitance_temperature_model: i32,
    /// The energy gap that scales the saturation currents with temperature, 1.11 by default, silicon's, and 1.16 at temperature model 2, in V; by default, as its source computes it from the parameters before it.
    pub energy_gap: Option<R>,
    /// Varshni's alpha of the energy gap at temperature model 2, in V/K; default 0.000702.
    pub energy_gap_alpha: R,
    /// Varshni's beta of the energy gap at temperature model 2, in K; default 1108.
    pub energy_gap_beta: R,
    /// The exponent of the temperature ratio in the saturation currents' scaling; default 3.
    pub saturation_current_temperature_exponent: R,
    /// The junction capacitance's linear temperature coefficient at capacitance temperature model 1, in 1/K; default 0.
    pub junction_capacitance_temperature_coefficient: R,
    /// The sidewall junction capacitance's linear temperature coefficient at capacitance temperature model 1, in 1/K; default 0.
    pub sidewall_junction_capacitance_temperature_coefficient: R,
    /// How fast the junction potential falls with temperature at capacitance temperature model 1, in V/K; default 0.
    pub junction_potential_temperature_coefficient: R,
    /// How fast the sidewall junction potential falls with temperature at capacitance temperature model 1, in V/K; default 0.
    pub sidewall_junction_potential_temperature_coefficient: R,
    /// The series resistances' linear temperature coefficient, in 1/K; default 0.
    pub series_resistance_temperature_coefficient: R,
    /// The series resistances' quadratic temperature coefficient, in 1/K^2; default 0.
    pub series_resistance_quadratic_temperature_coefficient: R,
    /// The transit time's linear temperature coefficient, in 1/K; default 0.
    pub transit_time_temperature_coefficient: R,
    /// The transit time's quadratic temperature coefficient, in 1/K^2; default 0.
    pub transit_time_quadratic_temperature_coefficient: R,
    /// The grading coefficient's linear temperature coefficient, in 1/K; default 0.
    pub grading_coefficient_temperature_coefficient: R,
    /// The grading coefficient's quadratic temperature coefficient, in 1/K^2; default 0.
    pub grading_coefficient_quadratic_temperature_coefficient: R,
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
            forward_knee_current: None,
            reverse_knee_current: None,
            recombination_saturation_current: None,
            recombination_emission_coefficient: R::constant(2.0),
            sidewall_saturation_current: R::constant(0.0),
            sidewall_emission_coefficient: None,
            sidewall_series_resistance: None,
            sidewall_junction_capacitance: R::constant(0.0),
            sidewall_junction_potential: R::constant(1.0),
            sidewall_grading_coefficient: R::constant(0.33),
            sidewall_depletion_coefficient: R::constant(0.5),
            sidewall_forward_knee_current: None,
            tunneling_saturation_current: R::constant(0.0),
            sidewall_tunneling_saturation_current: R::constant(0.0),
            tunneling_emission_coefficient: R::constant(30.0),
            tunneling_saturation_current_temperature_exponent: R::constant(3.0),
            tunneling_energy_gap_factor: R::constant(1.0),
            breakdown_voltage: None,
            breakdown_current: R::constant(0.001),
            breakdown_emission_coefficient: None,
            breakdown_temperature_coefficient: R::constant(0.0),
            temperature_model: 0,
            capacitance_temperature_model: 0,
            energy_gap: None,
            energy_gap_alpha: R::constant(0.000_702),
            energy_gap_beta: R::constant(1108.0),
            saturation_current_temperature_exponent: R::constant(3.0),
            junction_capacitance_temperature_coefficient: R::constant(0.0),
            sidewall_junction_capacitance_temperature_coefficient: R::constant(0.0),
            junction_potential_temperature_coefficient: R::constant(0.0),
            sidewall_junction_potential_temperature_coefficient: R::constant(0.0),
            series_resistance_temperature_coefficient: R::constant(0.0),
            series_resistance_quadratic_temperature_coefficient: R::constant(0.0),
            transit_time_temperature_coefficient: R::constant(0.0),
            transit_time_quadratic_temperature_coefficient: R::constant(0.0),
            grading_coefficient_temperature_coefficient: R::constant(0.0),
            grading_coefficient_quadratic_temperature_coefficient: R::constant(0.0),
            nominal_temperature: R::constant(300.15),
            flicker_coefficient: R::constant(0.0),
            flicker_exponent: R::constant(1.0),
        }
    }
}

/// A [`Diode`]'s numbers at its temperature: what depends on no probe.
#[derive(Debug, Clone, Copy)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the source's conditions on its parameters, each prepared once"
)]
pub struct PreparedDiode<R: Real> {
    resistance: R,
    h1: R,
    slope: R,
    critical: R,
    breaks: i32,
    matched: R,
    breakdown_slope: R,
    h7: bool,
    h8: R,
    saturation: R,
    h10: R,
    h11: R,
    recombination_saturation_current_given: bool,
    potential: R,
    h14: R,
    recombination: R,
    recombination_slope: R,
    h17: R,
    half_grading: R,
    h19: bool,
    tunneling: R,
    tunneling_slope: R,
    h22: R,
    forward_knee_current_given: bool,
    forward_knee_current: R,
    h25: R,
    reverse_knee_current_given: bool,
    reverse_knee_current: R,
    h28: R,
    h29: bool,
    threshold: R,
    h31: R,
    h32: R,
    h33: R,
    capacitance: R,
    charge_at_threshold: R,
    offset: R,
    h37: R,
    h38: R,
    divisor: R,
    h40: R,
    sidewall_slope: R,
    sidewall_critical: R,
    h43: bool,
    h44: bool,
    h45: R,
    sidewall_saturation: R,
    h47: R,
    h48: bool,
    sidewall_tunneling: R,
    sidewall_forward_knee_current_given: bool,
    sidewall_forward_knee_current: R,
    h52: R,
    h53: bool,
    sidewall_threshold: R,
    h55: R,
    h56: R,
    sidewall_potential: R,
    h58: R,
    h59: R,
    sidewall_capacitance: R,
    sidewall_charge_at_threshold: R,
    sidewall_offset: R,
    h63: R,
    h64: R,
    sidewall_divisor: R,
    h66: R,
    h67: R,
    sidewall_resistance: R,
    h69: R,
    h70: R,
    h71: R,
    h72: bool,
    flicker_coefficient: R,
    flicker_exponent: R,
    h75: R,
    h76: R,
    /// The conditions that shape the kernel.
    conditions: [bool; 5],
}

/// Where a [`Diode`]'s contributions go among its local unknowns, and the conditions
/// that chose them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiodeShape {
    junction: usize,
    sidewall: usize,
    junction_current: usize,
    sidewall_current: usize,
    anode_by_anode: usize,
    anode_by_junction: usize,
    anode_by_sidewall: usize,
    anode_by_junction_current: usize,
    anode_by_sidewall_current: usize,
    cathode_by_cathode: usize,
    cathode_by_junction: usize,
    cathode_by_sidewall: usize,
    junction_by_anode: usize,
    junction_by_cathode: usize,
    junction_by_junction: usize,
    junction_by_sidewall: usize,
    junction_by_junction_current: usize,
    sidewall_by_anode: usize,
    sidewall_by_cathode: usize,
    sidewall_by_junction: usize,
    sidewall_by_sidewall: usize,
    sidewall_by_sidewall_current: usize,
    junction_current_by_anode: usize,
    junction_current_by_junction: usize,
    junction_current_by_junction_current: usize,
    sidewall_current_by_anode: usize,
    sidewall_current_by_sidewall: usize,
    sidewall_current_by_sidewall_current: usize,
    conditions: [bool; 5],
    active: [bool; 9],
}

/// What a [`Diode`] remembers of its last evaluation: each limited voltage, which the next is limited from.
#[derive(Debug, Clone, Copy, Default)]
pub struct DiodeMemory {
    limited0: f64,
    limited1: f64,
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
    let mut v = Locals {
        reflected: 0.0,
        last: 0.0,
        step: 0.0,
        limited: 0.0,
        floor: 0.0,
        reverse: 0,
        junction_limit: 0.0,
    };
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

    #[expect(
        clippy::too_many_lines,
        reason = "a kernel's function is its Verilog-A source's analog block, which is one block"
    )]
    fn prepare<R: Real>(
        parameters: &DiodeParameters<R>,
        temperature: R,
    ) -> Result<PreparedDiode<R>, ParameterError> {
        struct Locals<R: Real> {
            vt: R,
            nominal_vt: R,
            ratio: R,
            difference: R,
            gap0: R,
            alpha: R,
            beta: R,
            activation: R,
            resistances: R,
            resistance: R,
            sidewall_resistance: R,
            transit: R,
            grading: R,
            saturation: R,
            sidewall_saturation: R,
            slope: R,
            sidewall_slope: R,
            critical: R,
            sidewall_critical: R,
            recombination: R,
            recombination_slope: R,
            half_grading: R,
            tunneling_scaling: R,
            tunneling_slope: R,
            tunneling: R,
            sidewall_tunneling: R,
            intrinsic: R,
            potential: R,
            capacitance: R,
            threshold: R,
            charge_at_threshold: R,
            divisor: R,
            offset: R,
            sidewall_intrinsic: R,
            sidewall_potential: R,
            sidewall_capacitance: R,
            sidewall_threshold: R,
            sidewall_charge_at_threshold: R,
            sidewall_divisor: R,
            sidewall_offset: R,
            breakdown_slope: R,
            unmatched: R,
            matched: R,
            fall: R,
            breaks: i32,
            sidewall_breaks: i32,
            separate: i32,
            varying: i32,
            iteration: i32,
            settled: i32,
        }
        ParameterError::positive("saturation_current", parameters.saturation_current)?;
        ParameterError::positive("emission_coefficient", parameters.emission_coefficient)?;
        ParameterError::non_negative("series_resistance", parameters.series_resistance)?;
        ParameterError::non_negative("junction_capacitance", parameters.junction_capacitance)?;
        ParameterError::positive("junction_potential", parameters.junction_potential)?;
        if !(0.0..1.0).contains(&parameters.grading_coefficient.value()) {
            return Err(ParameterError(format!("grading_coefficient is at least 0 and below 1, not {:?}", parameters.grading_coefficient.value())));
        }
        if !(0.0..1.0).contains(&parameters.depletion_coefficient.value()) {
            return Err(ParameterError(format!("depletion_coefficient is at least 0 and below 1, not {:?}", parameters.depletion_coefficient.value())));
        }
        ParameterError::non_negative("transit_time", parameters.transit_time)?;
        ParameterError::positive("forward_knee_current", parameters.forward_knee_current.unwrap_or(R::constant(1.0)))?;
        ParameterError::positive("reverse_knee_current", parameters.reverse_knee_current.unwrap_or(R::constant(1.0)))?;
        ParameterError::non_negative("recombination_saturation_current", parameters.recombination_saturation_current.unwrap_or(R::constant(0.0)))?;
        ParameterError::positive("recombination_emission_coefficient", parameters.recombination_emission_coefficient)?;
        ParameterError::non_negative("sidewall_saturation_current", parameters.sidewall_saturation_current)?;
        ParameterError::positive("sidewall_emission_coefficient", parameters.sidewall_emission_coefficient.unwrap_or(parameters.emission_coefficient))?;
        ParameterError::non_negative("sidewall_series_resistance", parameters.sidewall_series_resistance.unwrap_or(R::constant(0.0)))?;
        ParameterError::non_negative("sidewall_junction_capacitance", parameters.sidewall_junction_capacitance)?;
        ParameterError::positive("sidewall_junction_potential", parameters.sidewall_junction_potential)?;
        if !(0.0..1.0).contains(&parameters.sidewall_grading_coefficient.value()) {
            return Err(ParameterError(format!("sidewall_grading_coefficient is at least 0 and below 1, not {:?}", parameters.sidewall_grading_coefficient.value())));
        }
        if !(0.0..1.0).contains(&parameters.sidewall_depletion_coefficient.value()) {
            return Err(ParameterError(format!("sidewall_depletion_coefficient is at least 0 and below 1, not {:?}", parameters.sidewall_depletion_coefficient.value())));
        }
        ParameterError::positive("sidewall_forward_knee_current", parameters.sidewall_forward_knee_current.unwrap_or(R::constant(1.0)))?;
        ParameterError::non_negative("tunneling_saturation_current", parameters.tunneling_saturation_current)?;
        ParameterError::non_negative("sidewall_tunneling_saturation_current", parameters.sidewall_tunneling_saturation_current)?;
        ParameterError::positive("tunneling_emission_coefficient", parameters.tunneling_emission_coefficient)?;
        ParameterError::non_negative("breakdown_voltage", parameters.breakdown_voltage.unwrap_or(R::constant(1.0)))?;
        ParameterError::non_negative("breakdown_current", parameters.breakdown_current)?;
        ParameterError::positive("breakdown_emission_coefficient", parameters.breakdown_emission_coefficient.unwrap_or(parameters.emission_coefficient))?;
        if !(0.0..=2.0).contains(&f64::from(parameters.temperature_model)) {
            return Err(ParameterError(format!("temperature_model is at least 0 and at most 2, not {:?}", parameters.temperature_model)));
        }
        if !(0.0..=1.0).contains(&f64::from(parameters.capacitance_temperature_model)) {
            return Err(ParameterError(format!("capacitance_temperature_model is at least 0 and at most 1, not {:?}", parameters.capacitance_temperature_model)));
        }
        ParameterError::non_negative("energy_gap_beta", parameters.energy_gap_beta)?;
        ParameterError::positive("nominal_temperature", parameters.nominal_temperature)?;
        ParameterError::non_negative("flicker_coefficient", parameters.flicker_coefficient)?;
        ParameterError::positive("flicker_exponent", parameters.flicker_exponent)?;
        let mut v = Locals {
            vt: R::constant(0.0),
            nominal_vt: R::constant(0.0),
            ratio: R::constant(0.0),
            difference: R::constant(0.0),
            gap0: R::constant(0.0),
            alpha: R::constant(0.0),
            beta: R::constant(0.0),
            activation: R::constant(0.0),
            resistances: R::constant(0.0),
            resistance: R::constant(0.0),
            sidewall_resistance: R::constant(0.0),
            transit: R::constant(0.0),
            grading: R::constant(0.0),
            saturation: R::constant(0.0),
            sidewall_saturation: R::constant(0.0),
            slope: R::constant(0.0),
            sidewall_slope: R::constant(0.0),
            critical: R::constant(0.0),
            sidewall_critical: R::constant(0.0),
            recombination: R::constant(0.0),
            recombination_slope: R::constant(0.0),
            half_grading: R::constant(0.0),
            tunneling_scaling: R::constant(0.0),
            tunneling_slope: R::constant(0.0),
            tunneling: R::constant(0.0),
            sidewall_tunneling: R::constant(0.0),
            intrinsic: R::constant(0.0),
            potential: R::constant(0.0),
            capacitance: R::constant(0.0),
            threshold: R::constant(0.0),
            charge_at_threshold: R::constant(0.0),
            divisor: R::constant(0.0),
            offset: R::constant(0.0),
            sidewall_intrinsic: R::constant(0.0),
            sidewall_potential: R::constant(0.0),
            sidewall_capacitance: R::constant(0.0),
            sidewall_threshold: R::constant(0.0),
            sidewall_charge_at_threshold: R::constant(0.0),
            sidewall_divisor: R::constant(0.0),
            sidewall_offset: R::constant(0.0),
            breakdown_slope: R::constant(0.0),
            unmatched: R::constant(0.0),
            matched: R::constant(0.0),
            fall: R::constant(0.0),
            breaks: 0,
            sidewall_breaks: 0,
            separate: 0,
            varying: 0,
            iteration: 0,
            settled: 0,
        };
        if temperature.value() <= 0.0 {
            return Err(ParameterError(format!("a junction's temperature lies above 0 K, not at {:?} K", temperature.value())));
        }
        let t0000 = if parameters.temperature_model == 2 { 1.16 } else { 1.11 };
        if parameters.energy_gap.unwrap_or(R::constant(t0000)).value() < 0.0 {
            let t0001 = if parameters.temperature_model == 2 { 1.16 } else { 1.11 };
            return Err(ParameterError(format!("energy_gap is at least 0, not {:?}", parameters.energy_gap.unwrap_or(R::constant(t0001)).value())));
        }
        v.vt = temperature * BOLTZMANN / ELEMENTARY_CHARGE;
        v.ratio = temperature / parameters.nominal_temperature;
        v.difference = temperature - parameters.nominal_temperature;
        v.separate = i32::from(parameters.sidewall_series_resistance.is_some());
        v.resistances = (parameters.series_resistance_temperature_coefficient + parameters.series_resistance_quadratic_temperature_coefficient * v.difference) * v.difference + 1.0;
        v.resistance = parameters.series_resistance * v.resistances;
        v.sidewall_resistance = parameters.sidewall_series_resistance.unwrap_or(R::constant(0.0)) * v.resistances;
        v.transit = (parameters.transit_time_temperature_coefficient + parameters.transit_time_quadratic_temperature_coefficient * v.difference) * v.difference + 1.0;
        v.varying = i32::from(parameters.temperature_model == 2);
        if v.varying == 0 {
            v.gap0 = R::constant(1.16);
            v.alpha = R::constant(0.000_702);
            v.beta = R::constant(1108.0);
            let t0003 = if parameters.temperature_model == 2 { 1.16 } else { 1.11 };
            v.activation = (v.ratio - 1.0) * parameters.energy_gap.unwrap_or(R::constant(t0003)) / v.vt;
        } else {
            let t0002 = if parameters.temperature_model == 2 { 1.16 } else { 1.11 };
            v.gap0 = parameters.energy_gap.unwrap_or(R::constant(t0002));
            v.alpha = parameters.energy_gap_alpha;
            v.beta = parameters.energy_gap_beta;
            v.nominal_vt = parameters.nominal_temperature * BOLTZMANN / ELEMENTARY_CHARGE;
            v.activation = -(parameters.nominal_temperature * v.alpha * parameters.nominal_temperature / (parameters.nominal_temperature + v.beta) - v.gap0) / v.nominal_vt - -(temperature * v.alpha * temperature / (temperature + v.beta) - v.gap0) / v.vt;
        }
        v.saturation = parameters.saturation_current * ((v.activation * 1.0 + parameters.saturation_current_temperature_exponent * v.ratio.ln()) / parameters.emission_coefficient).exp();
        v.sidewall_saturation = parameters.sidewall_saturation_current * ((v.activation * 1.0 + parameters.saturation_current_temperature_exponent * v.ratio.ln()) / parameters.sidewall_emission_coefficient.unwrap_or(parameters.emission_coefficient)).exp();
        v.grading = parameters.grading_coefficient * ((parameters.grading_coefficient_temperature_coefficient + parameters.grading_coefficient_quadratic_temperature_coefficient * v.difference) * v.difference + 1.0);
        if parameters.capacitance_temperature_model == 0 {
            v.intrinsic = (parameters.junction_potential - parameters.nominal_temperature * BOLTZMANN / ELEMENTARY_CHARGE * -2.0 * ((parameters.nominal_temperature / 300.15).ln() * 1.5 + ((parameters.nominal_temperature * v.alpha * parameters.nominal_temperature / (parameters.nominal_temperature + v.beta) - v.gap0) / (parameters.nominal_temperature * 2.761_298e-23) + 1.345_418_513_059_931_1e20) * ELEMENTARY_CHARGE)) / (parameters.nominal_temperature / 300.15);
            v.potential = v.intrinsic * (temperature / 300.15) + temperature * BOLTZMANN / ELEMENTARY_CHARGE * -2.0 * ((temperature / 300.15).ln() * 1.5 + ((temperature * v.alpha * temperature / (temperature + v.beta) - v.gap0) / (temperature * 2.761_298e-23) + 1.345_418_513_059_931_1e20) * ELEMENTARY_CHARGE);
            v.capacitance = parameters.junction_capacitance * (v.grading * ((temperature - 300.15) * 0.0004 - (v.potential - v.intrinsic) / v.intrinsic) + 1.0) / (v.grading * ((parameters.nominal_temperature - 300.15) * 0.0004 - (parameters.junction_potential - v.intrinsic) / v.intrinsic) + 1.0);
            v.sidewall_intrinsic = (parameters.sidewall_junction_potential - parameters.nominal_temperature * BOLTZMANN / ELEMENTARY_CHARGE * -2.0 * ((parameters.nominal_temperature / 300.15).ln() * 1.5 + ((parameters.nominal_temperature * v.alpha * parameters.nominal_temperature / (parameters.nominal_temperature + v.beta) - v.gap0) / (parameters.nominal_temperature * 2.761_298e-23) + 1.345_418_513_059_931_1e20) * ELEMENTARY_CHARGE)) / (parameters.nominal_temperature / 300.15);
            v.sidewall_potential = v.sidewall_intrinsic * (temperature / 300.15) + temperature * BOLTZMANN / ELEMENTARY_CHARGE * -2.0 * ((temperature / 300.15).ln() * 1.5 + ((temperature * v.alpha * temperature / (temperature + v.beta) - v.gap0) / (temperature * 2.761_298e-23) + 1.345_418_513_059_931_1e20) * ELEMENTARY_CHARGE);
            v.sidewall_capacitance = parameters.sidewall_junction_capacitance * (parameters.sidewall_grading_coefficient * ((temperature - 300.15) * 0.0004 - (v.sidewall_potential - v.sidewall_intrinsic) / v.sidewall_intrinsic) + 1.0) / (parameters.sidewall_grading_coefficient * ((parameters.nominal_temperature - 300.15) * 0.0004 - (parameters.sidewall_junction_potential - v.sidewall_intrinsic) / v.sidewall_intrinsic) + 1.0);
        } else {
            v.potential = parameters.junction_potential - parameters.junction_potential_temperature_coefficient * v.difference;
            v.capacitance = parameters.junction_capacitance * (parameters.junction_capacitance_temperature_coefficient * v.difference + 1.0);
            v.sidewall_potential = parameters.sidewall_junction_potential - parameters.sidewall_junction_potential_temperature_coefficient * v.difference;
            v.sidewall_capacitance = parameters.sidewall_junction_capacitance * (parameters.sidewall_junction_capacitance_temperature_coefficient * v.difference + 1.0);
        }
        if v.potential.value() <= 0.0 {
            return Err(ParameterError(format!("junction_potential at the temperature is positive, not {:?}", v.potential.value())));
        }
        if v.sidewall_potential.value() <= 0.0 {
            return Err(ParameterError(format!("sidewall_junction_potential at the temperature is positive, not {:?}", v.sidewall_potential.value())));
        }
        v.threshold = parameters.depletion_coefficient * v.potential;
        v.charge_at_threshold = v.potential * (R::constant(1.0) - (R::constant(1.0) - parameters.depletion_coefficient).pow(R::constant(1.0) - v.grading)) / (R::constant(1.0) - v.grading);
        v.divisor = (R::constant(1.0) - parameters.depletion_coefficient).pow(v.grading + 1.0);
        v.offset = R::constant(1.0) - parameters.depletion_coefficient * (v.grading + 1.0);
        v.sidewall_threshold = parameters.sidewall_depletion_coefficient * v.sidewall_potential;
        v.sidewall_charge_at_threshold = v.sidewall_potential * (R::constant(1.0) - (R::constant(1.0) - parameters.sidewall_depletion_coefficient).pow(R::constant(1.0) - parameters.sidewall_grading_coefficient)) / (R::constant(1.0) - parameters.sidewall_grading_coefficient);
        v.sidewall_divisor = (R::constant(1.0) - parameters.sidewall_depletion_coefficient).pow(parameters.sidewall_grading_coefficient + 1.0);
        v.sidewall_offset = R::constant(1.0) - parameters.sidewall_depletion_coefficient * (parameters.sidewall_grading_coefficient + 1.0);
        v.slope = parameters.emission_coefficient * v.vt;
        v.sidewall_slope = parameters.sidewall_emission_coefficient.unwrap_or(parameters.emission_coefficient) * v.vt;
        if v.separate == 0 {
            v.critical = v.slope * (v.slope / ((v.saturation + v.sidewall_saturation) * std::f64::consts::SQRT_2)).ln();
        } else {
            v.critical = v.slope * (v.slope / (v.saturation * std::f64::consts::SQRT_2)).ln();
            v.sidewall_critical = v.sidewall_slope * (v.sidewall_slope / (v.sidewall_saturation * std::f64::consts::SQRT_2)).ln();
        }
        if parameters.recombination_saturation_current.is_some() {
            v.recombination = parameters.recombination_saturation_current.unwrap_or(R::constant(0.0)) * ((v.activation * 1.0 + parameters.saturation_current_temperature_exponent * v.ratio.ln()) / parameters.recombination_emission_coefficient).exp();
            v.recombination_slope = parameters.recombination_emission_coefficient * v.vt;
            v.half_grading = v.grading * 0.5;
        }
        if parameters.tunneling_saturation_current.value() > 0.0 || parameters.sidewall_tunneling_saturation_current.value() > 0.0 {
            v.tunneling_scaling = ((parameters.tunneling_energy_gap_factor * v.activation + parameters.tunneling_saturation_current_temperature_exponent * v.ratio.ln()) / parameters.tunneling_emission_coefficient).exp();
            v.tunneling_slope = parameters.tunneling_emission_coefficient * v.vt;
            v.tunneling = parameters.tunneling_saturation_current * v.tunneling_scaling;
            v.sidewall_tunneling = parameters.sidewall_tunneling_saturation_current * v.tunneling_scaling;
        }
        v.breaks = i32::from(parameters.breakdown_voltage.is_some());
        v.breakdown_slope = parameters.breakdown_emission_coefficient.unwrap_or(parameters.emission_coefficient) * v.vt;
        v.fall = parameters.breakdown_temperature_coefficient * v.difference;
        if parameters.temperature_model == 0 {
            v.matched = parameters.breakdown_voltage.unwrap_or(R::constant(1.0)) - v.fall;
        } else {
            v.matched = parameters.breakdown_voltage.unwrap_or(R::constant(1.0)) * -(v.fall - 1.0);
        }
        if v.breaks != 0 && parameters.breakdown_current.value() >= ((v.saturation + v.sidewall_saturation) * v.matched / v.vt).value() {
            v.unmatched = v.matched;
            v.matched = v.unmatched - v.breakdown_slope * (parameters.breakdown_current / (v.saturation + v.sidewall_saturation) + 1.0).ln();
            v.iteration = 0;
            v.settled = 0;
            while v.iteration < 25 && v.settled == 0 {
                v.matched = v.unmatched - v.breakdown_slope * (parameters.breakdown_current / (v.saturation + v.sidewall_saturation) + 1.0 - v.matched / v.vt).ln();
                v.settled = i32::from(((v.saturation + v.sidewall_saturation) * (((v.unmatched - v.matched) / v.breakdown_slope).exp() - 1.0 + v.matched / v.vt) - parameters.breakdown_current).abs().value() <= (parameters.breakdown_current * 1e-12).value());
                v.iteration = v.iteration.wrapping_add(1);
            }
        }
        v.sidewall_breaks = i32::from(v.breaks != 0 && v.separate == 0);
        Ok(PreparedDiode {
            resistance: v.resistance,
            h1: R::constant(1.0) / v.resistance,
            slope: v.slope,
            critical: v.critical,
            breaks: v.breaks,
            matched: v.matched,
            breakdown_slope: v.breakdown_slope,
            h7: v.breaks != 0,
            h8: v.slope * 3.0,
            saturation: v.saturation,
            h10: R::constant(1.0) / v.breakdown_slope,
            h11: R::constant(1.0) / v.slope,
            recombination_saturation_current_given: parameters.recombination_saturation_current.is_some(),
            potential: v.potential,
            h14: R::constant(1.0) / v.potential,
            recombination: v.recombination,
            recombination_slope: v.recombination_slope,
            h17: R::constant(1.0) / v.recombination_slope,
            half_grading: v.half_grading,
            h19: v.tunneling.value() > 0.0,
            tunneling: v.tunneling,
            tunneling_slope: v.tunneling_slope,
            h22: R::constant(1.0) / v.tunneling_slope,
            forward_knee_current_given: parameters.forward_knee_current.is_some(),
            forward_knee_current: parameters.forward_knee_current.unwrap_or(R::constant(1.0)),
            h25: R::constant(1.0) / parameters.forward_knee_current.unwrap_or(R::constant(1.0)),
            reverse_knee_current_given: parameters.reverse_knee_current.is_some(),
            reverse_knee_current: parameters.reverse_knee_current.unwrap_or(R::constant(1.0)),
            h28: R::constant(1.0) / parameters.reverse_knee_current.unwrap_or(R::constant(1.0)),
            h29: v.capacitance.value() > 0.0,
            threshold: v.threshold,
            h31: v.capacitance * v.potential,
            h32: R::constant(1.0) - v.grading,
            h33: R::constant(1.0) / (R::constant(1.0) - v.grading),
            capacitance: v.capacitance,
            charge_at_threshold: v.charge_at_threshold,
            offset: v.offset,
            h37: v.grading / (v.potential * 2.0),
            h38: v.threshold * v.threshold,
            divisor: v.divisor,
            h40: R::constant(1.0) / v.divisor,
            sidewall_slope: v.sidewall_slope,
            sidewall_critical: v.sidewall_critical,
            h43: v.sidewall_saturation.value() > 0.0,
            h44: v.sidewall_breaks != 0,
            h45: v.sidewall_slope * 3.0,
            sidewall_saturation: v.sidewall_saturation,
            h47: R::constant(1.0) / v.sidewall_slope,
            h48: v.sidewall_tunneling.value() > 0.0,
            sidewall_tunneling: v.sidewall_tunneling,
            sidewall_forward_knee_current_given: parameters.sidewall_forward_knee_current.is_some(),
            sidewall_forward_knee_current: parameters.sidewall_forward_knee_current.unwrap_or(R::constant(1.0)),
            h52: R::constant(1.0) / parameters.sidewall_forward_knee_current.unwrap_or(R::constant(1.0)),
            h53: v.sidewall_capacitance.value() > 0.0,
            sidewall_threshold: v.sidewall_threshold,
            h55: v.sidewall_capacitance * v.sidewall_potential,
            h56: R::constant(1.0) - parameters.sidewall_grading_coefficient,
            sidewall_potential: v.sidewall_potential,
            h58: R::constant(1.0) / v.sidewall_potential,
            h59: R::constant(1.0) / (R::constant(1.0) - parameters.sidewall_grading_coefficient),
            sidewall_capacitance: v.sidewall_capacitance,
            sidewall_charge_at_threshold: v.sidewall_charge_at_threshold,
            sidewall_offset: v.sidewall_offset,
            h63: parameters.sidewall_grading_coefficient / (v.sidewall_potential * 2.0),
            h64: v.sidewall_threshold * v.sidewall_threshold,
            sidewall_divisor: v.sidewall_divisor,
            h66: R::constant(1.0) / v.sidewall_divisor,
            h67: parameters.transit_time * v.transit,
            sidewall_resistance: v.sidewall_resistance,
            h69: R::constant(1.0) / v.sidewall_resistance,
            h70: temperature * 5.522_596e-23 * v.resistance.abs(),
            h71: temperature * 5.522_596e-23 * (R::constant(1.0) / v.resistance).abs(),
            h72: parameters.flicker_coefficient.value() > 0.0,
            flicker_coefficient: parameters.flicker_coefficient,
            flicker_exponent: parameters.flicker_exponent,
            h75: temperature * 5.522_596e-23 * v.sidewall_resistance.abs(),
            h76: temperature * 5.522_596e-23 * (R::constant(1.0) / v.sidewall_resistance).abs(),
            conditions: [
                parameters.series_resistance.value() == 0.0,
                parameters.series_resistance.value() < SMALL_RESISTANCE,
                v.separate != 0,
                parameters.sidewall_series_resistance.unwrap_or(R::constant(0.0)).value() == 0.0,
                parameters.sidewall_series_resistance.unwrap_or(R::constant(0.0)).value() < SMALL_RESISTANCE,
            ],
        })
    }

    fn layout<R: Real>(
        _: &DiodeParameters<R>,
        prepared: &PreparedDiode<R>,
        layout: &mut Layout,
    ) -> Result<DiodeShape, ParameterError> {
        let mut active = [false; 9];
        if prepared.conditions[0] {
            active[3] = true;
        } else if prepared.conditions[1] {
            active[0] = true;
        } else {
            active[2] = true;
        }
        if !prepared.conditions[2] {
            active[6] = true;
        }
        if prepared.conditions[2] {
            active[4] = true;
            if prepared.conditions[3] {
                active[8] = true;
            } else if prepared.conditions[4] {
                active[1] = true;
            } else {
                active[7] = true;
            }
            active[5] = true;
        } else {
            active[4] = true;
        }
        // The leader of each node's group, the ground first, then the nodes in order.
        let leader = |joined: &[usize; 5], mut place: usize| {
            while let Some(&next) = joined.get(place)
                && next != place
            {
                place = next;
            }
            place
        };
        let mut joined = [0, 1, 2, 3, 4];
        if active[3] { let (one, other) = (leader(&joined, 1), leader(&joined, 3)); if let Some(slot) = joined.get_mut(one.max(other)) { *slot = one.min(other); } }
        if active[6] { let (one, other) = (leader(&joined, 4), leader(&joined, 3)); if let Some(slot) = joined.get_mut(one.max(other)) { *slot = one.min(other); } }
        if active[8] { let (one, other) = (leader(&joined, 1), leader(&joined, 4)); if let Some(slot) = joined.get_mut(one.max(other)) { *slot = one.min(other); } }
        let mut local = [NOWHERE, 0, 1, NOWHERE, NOWHERE];
        local[3] = match leader(&joined, 3) { 3 => layout.internal("junction", Nature::Voltage), leader => local.get(leader).copied().unwrap_or(NOWHERE) };
        local[4] = match leader(&joined, 4) { 4 => layout.internal("sidewall", Nature::Voltage), leader => local.get(leader).copied().unwrap_or(NOWHERE) };
        let [_, _, _, junction, sidewall] = local;
        let junction_current = if active[0] { layout.internal("junction_current", Nature::Current) } else { NOWHERE };
        let sidewall_current = if active[1] { layout.internal("sidewall_current", Nature::Current) } else { NOWHERE };
        Ok(DiodeShape {
            junction,
            sidewall,
            junction_current,
            sidewall_current,
            anode_by_anode: if active[2] || active[7] { layout.entry(0, 0) } else { NOWHERE },
            anode_by_junction: if active[2] { layout.entry(0, junction) } else { NOWHERE },
            anode_by_sidewall: if active[7] { layout.entry(0, sidewall) } else { NOWHERE },
            anode_by_junction_current: if active[0] { layout.entry(0, junction_current) } else { NOWHERE },
            anode_by_sidewall_current: if active[1] { layout.entry(0, sidewall_current) } else { NOWHERE },
            cathode_by_cathode: if active[4] || active[5] { layout.entry(1, 1) } else { NOWHERE },
            cathode_by_junction: if active[4] || active[5] { layout.entry(1, junction) } else { NOWHERE },
            cathode_by_sidewall: if active[4] || active[5] { layout.entry(1, sidewall) } else { NOWHERE },
            junction_by_anode: if active[2] { layout.entry(junction, 0) } else { NOWHERE },
            junction_by_cathode: if active[4] { layout.entry(junction, 1) } else { NOWHERE },
            junction_by_junction: if active[2] || active[4] { layout.entry(junction, junction) } else { NOWHERE },
            junction_by_sidewall: if active[4] { layout.entry(junction, sidewall) } else { NOWHERE },
            junction_by_junction_current: if active[0] { layout.entry(junction, junction_current) } else { NOWHERE },
            sidewall_by_anode: if active[7] { layout.entry(sidewall, 0) } else { NOWHERE },
            sidewall_by_cathode: if active[5] { layout.entry(sidewall, 1) } else { NOWHERE },
            sidewall_by_junction: if active[5] { layout.entry(sidewall, junction) } else { NOWHERE },
            sidewall_by_sidewall: if active[5] || active[7] { layout.entry(sidewall, sidewall) } else { NOWHERE },
            sidewall_by_sidewall_current: if active[1] { layout.entry(sidewall, sidewall_current) } else { NOWHERE },
            junction_current_by_anode: if active[0] { layout.entry(junction_current, 0) } else { NOWHERE },
            junction_current_by_junction: if active[0] { layout.entry(junction_current, junction) } else { NOWHERE },
            junction_current_by_junction_current: if active[0] { layout.entry(junction_current, junction_current) } else { NOWHERE },
            sidewall_current_by_anode: if active[1] { layout.entry(sidewall_current, 0) } else { NOWHERE },
            sidewall_current_by_sidewall: if active[1] { layout.entry(sidewall_current, sidewall) } else { NOWHERE },
            sidewall_current_by_sidewall_current: if active[1] { layout.entry(sidewall_current, sidewall_current) } else { NOWHERE },
            conditions: prepared.conditions,
            active,
        })
    }

    #[expect(
        clippy::too_many_lines,
        reason = "a kernel's function is its Verilog-A source's analog block, which is one block"
    )]
    fn evaluate<R: Real>(
        shape: &DiodeShape,
        prepared: &PreparedDiode<R>,
        memory: &mut DiodeMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        struct Locals<R: Real> {
            vd: Dual<R, 1>,
            vs: Dual<R, 2>,
            vr: Dual<R, 1>,
            id: Dual<R, 2>,
            isw: Dual<R, 2>,
            cube: Dual<R, 2>,
            remaining: Dual<R, 1>,
            qd: Dual<R, 1>,
            qs: Dual<R, 2>,
            voltage_anode_junction: Dual<R, 1>,
            voltage_anode_sidewall: Dual<R, 1>,
            current_junction_current: Dual<R, 1>,
            current_sidewall_current: Dual<R, 1>,
            junction_current_potential: Dual<R, 1>,
            sidewall_current_potential: Dual<R, 1>,
            anode_junction_flow: Dual<R, 1>,
            junction_cathode_flow: Dual<R, 2>,
            junction_cathode_charge: Dual<R, 2>,
            sidewall_cathode_flow: Dual<R, 2>,
            sidewall_cathode_charge: Dual<R, 2>,
            anode_sidewall_flow: Dual<R, 1>,
            applied0: f64,
            limited0: f64,
            applied1: f64,
            limited1: f64,
        }
        let mut v = Locals {
            vd: Dual::number(0.0),
            vs: Dual::number(0.0),
            vr: Dual::number(0.0),
            id: Dual::number(0.0),
            isw: Dual::number(0.0),
            cube: Dual::number(0.0),
            remaining: Dual::number(0.0),
            qd: Dual::number(0.0),
            qs: Dual::number(0.0),
            voltage_anode_junction: Dual::number(0.0),
            voltage_anode_sidewall: Dual::number(0.0),
            current_junction_current: Dual::number(0.0),
            current_sidewall_current: Dual::number(0.0),
            junction_current_potential: Dual::number(0.0),
            sidewall_current_potential: Dual::number(0.0),
            anode_junction_flow: Dual::number(0.0),
            junction_cathode_flow: Dual::number(0.0),
            junction_cathode_charge: Dual::number(0.0),
            sidewall_cathode_flow: Dual::number(0.0),
            sidewall_cathode_charge: Dual::number(0.0),
            anode_sidewall_flow: Dual::number(0.0),
            applied0: 0.0,
            limited0: 0.0,
            applied1: 0.0,
            limited1: 0.0,
        };
        v.voltage_anode_junction = Dual::variable(at.unknown::<R>(0) - at.unknown::<R>(shape.junction));
        v.voltage_anode_sidewall = Dual::variable(at.unknown::<R>(0) - at.unknown::<R>(shape.sidewall));
        v.current_junction_current = Dual::variable(at.unknown::<R>(shape.junction_current));
        v.current_sidewall_current = Dual::variable(at.unknown::<R>(shape.sidewall_current));
        if !shape.conditions[0] {
            if shape.conditions[1] {
                v.junction_current_potential = v.junction_current_potential + v.current_junction_current.scale(prepared.resistance);
            } else {
                v.anode_junction_flow = v.anode_junction_flow + v.voltage_anode_junction.scale(prepared.h1);
            }
        }
        let applied0 = at.value(shape.junction) - at.value(1);
        let limited0 = if at.limiting { junction_limit(applied0, memory.limited0, prepared.slope.value(), prepared.critical.value(), f64::from(prepared.breaks), prepared.matched.value(), prepared.breakdown_slope.value()) } else { applied0 };
        if limited0.total_cmp(&applied0).is_ne() {
            stamp.mark_limited();
        }
        memory.limited0 = limited0;
        v.applied0 = applied0;
        v.limited0 = limited0;
        let t0000 = Dual::variable((at.unknown::<R>(shape.junction) - at.unknown::<R>(1)).with_value(limited0));
        v.vd = t0000;
        if prepared.h7 && v.vd.value.value() < -prepared.h8.value() && v.vd.value.value() < -prepared.matched.value() {
            let t0001 = -v.vd.plus(prepared.matched);
            v.id = t0001.chain((t0001.value / prepared.breakdown_slope, prepared.h10)).exp().scale(-prepared.saturation).embed::<2>([0]);
        } else if v.vd.value.value() >= -prepared.h8.value() {
            v.id = (v.vd.chain((v.vd.value / prepared.slope, prepared.h11)).exp() - 1.0).scale(prepared.saturation).embed::<2>([0]);
        } else {
            v.cube = (Dual::constant(prepared.h8) / (v.vd * std::f64::consts::E)).embed::<2>([0]);
            v.id = (v.cube * v.cube * v.cube + 1.0).scale(-prepared.saturation);
        }
        if prepared.recombination_saturation_current_given {
            if v.vd.value.value() >= -prepared.h8.value() {
                v.vr = v.vd;
            } else {
                v.vr = Dual::constant(-prepared.h8);
            }
            v.remaining = -(v.vr.chain((v.vr.value / prepared.potential, prepared.h14)) - 1.0);
            let t0002 = v.remaining * v.remaining + 0.005;
            let t0003 = t0002.value.pow(prepared.half_grading);
            let t0004 = if t0002.value.value() == 0.0 { prepared.half_grading * t0002.value.pow(prepared.half_grading - 1.0) } else { prepared.half_grading * t0003 / t0002.value };
            v.id = v.id + ((v.vr.chain((v.vr.value / prepared.recombination_slope, prepared.h17)).exp() - 1.0).scale(prepared.recombination) * t0002.chain((t0003, t0004))).embed::<2>([0]);
        }
        if prepared.h19 {
            let t0005 = -v.vd;
            v.id = v.id - (t0005.chain((t0005.value / prepared.tunneling_slope, prepared.h22)).exp() - 1.0).scale(prepared.tunneling).embed::<2>([0]);
        }
        if v.vd.value.value() >= -prepared.h8.value() {
            if prepared.forward_knee_current_given && v.id.value.value() > 1e-18 {
                v.id = v.id / (v.id.chain((v.id.value / prepared.forward_knee_current, prepared.h25)).sqrt() + 1.0);
            }
        } else if prepared.reverse_knee_current_given && v.id.value.value() < -1e-18 {
            let t0006 = -v.id;
            v.id = v.id / (t0006.chain((t0006.value / prepared.reverse_knee_current, prepared.h28)).sqrt() + 1.0);
        }
        if prepared.h29 {
            if v.vd.value.value() < prepared.threshold.value() {
                let t0007 = -v.vd.chain((v.vd.value / prepared.potential, prepared.h14));
                let t0008 = t0007.value.value();
                let t0009 = t0007.value.chain((t0008.ln_1p(), (t0008 + 1.0).recip(), -((t0008 + 1.0) * (t0008 + 1.0)).recip()));
                let t0010 = t0007.chain((t0009, (t0007.value + 1.0).recip())).scale(prepared.h32);
                let t0011 = t0010.value.value();
                let t0012 = t0010.value.chain((t0011.exp_m1(), t0011.exp(), t0011.exp()));
                let t0013 = (-t0010.chain((t0012, t0010.value.exp()))).scale(prepared.h31);
                v.qd = t0013.chain((t0013.value / prepared.h32, prepared.h33));
            } else {
                let t0014 = v.vd.plus(-prepared.threshold).scale(prepared.offset) + (v.vd * v.vd).plus(-prepared.h38).scale(prepared.h37);
                v.qd = t0014.chain((t0014.value / prepared.divisor, prepared.h40)).plus(prepared.charge_at_threshold).scale(prepared.capacitance);
            }
        }
        if shape.conditions[2] {
            let applied1 = at.value(shape.sidewall) - at.value(1);
            let limited1 = if at.limiting { junction_limit(applied1, memory.limited1, prepared.sidewall_slope.value(), prepared.sidewall_critical.value(), 0.0, 0.0, 0.0) } else { applied1 };
            if limited1.total_cmp(&applied1).is_ne() {
                stamp.mark_limited();
            }
            memory.limited1 = limited1;
            v.applied1 = applied1;
            v.limited1 = limited1;
            let t0015 = Dual::variable((at.unknown::<R>(shape.sidewall) - at.unknown::<R>(1)).with_value(limited1));
            v.vs = t0015.embed::<2>([1]);
        } else {
            v.vs = v.vd.embed::<2>([0]);
        }
        v.isw = Dual::number(0.0);
        if prepared.h43 {
            if prepared.h44 && v.vs.value.value() < -prepared.h45.value() && v.vs.value.value() < -prepared.matched.value() {
                let t0016 = -v.vs.plus(prepared.matched);
                v.isw = t0016.chain((t0016.value / prepared.breakdown_slope, prepared.h10)).exp().scale(-prepared.sidewall_saturation);
            } else if v.vs.value.value() >= -prepared.h45.value() {
                v.isw = (v.vs.chain((v.vs.value / prepared.sidewall_slope, prepared.h47)).exp() - 1.0).scale(prepared.sidewall_saturation);
            } else {
                v.cube = Dual::constant(prepared.h45) / (v.vs * std::f64::consts::E);
                v.isw = (v.cube * v.cube * v.cube + 1.0).scale(-prepared.sidewall_saturation);
            }
        }
        if prepared.h48 {
            let t0017 = -v.vs;
            v.isw = v.isw - (t0017.chain((t0017.value / prepared.tunneling_slope, prepared.h22)).exp() - 1.0).scale(prepared.sidewall_tunneling);
        }
        if prepared.sidewall_forward_knee_current_given && v.isw.value.value() > 1e-18 {
            v.isw = v.isw / (v.isw.chain((v.isw.value / prepared.sidewall_forward_knee_current, prepared.h52)).sqrt() + 1.0);
        }
        if prepared.h53 {
            if v.vs.value.value() < prepared.sidewall_threshold.value() {
                let t0018 = -v.vs.chain((v.vs.value / prepared.sidewall_potential, prepared.h58));
                let t0019 = t0018.value.value();
                let t0020 = t0018.value.chain((t0019.ln_1p(), (t0019 + 1.0).recip(), -((t0019 + 1.0) * (t0019 + 1.0)).recip()));
                let t0021 = t0018.chain((t0020, (t0018.value + 1.0).recip())).scale(prepared.h56);
                let t0022 = t0021.value.value();
                let t0023 = t0021.value.chain((t0022.exp_m1(), t0022.exp(), t0022.exp()));
                let t0024 = (-t0021.chain((t0023, t0021.value.exp()))).scale(prepared.h55);
                v.qs = t0024.chain((t0024.value / prepared.h56, prepared.h59));
            } else {
                let t0025 = v.vs.plus(-prepared.sidewall_threshold).scale(prepared.sidewall_offset) + (v.vs * v.vs).plus(-prepared.h64).scale(prepared.h63);
                v.qs = t0025.chain((t0025.value / prepared.sidewall_divisor, prepared.h66)).plus(prepared.sidewall_charge_at_threshold).scale(prepared.sidewall_capacitance);
            }
        }
        if shape.conditions[2] {
            v.junction_cathode_flow = v.junction_cathode_flow + (v.id + (at.gmin * v.vd).embed::<2>([0]));
            v.junction_cathode_charge = v.junction_cathode_charge + (v.qd.embed::<2>([0]) + v.id.scale(prepared.h67));
            if !shape.conditions[3] {
                if shape.conditions[4] {
                    v.sidewall_current_potential = v.sidewall_current_potential + v.current_sidewall_current.scale(prepared.sidewall_resistance);
                } else {
                    v.anode_sidewall_flow = v.anode_sidewall_flow + v.voltage_anode_sidewall.scale(prepared.h69);
                }
            }
            v.sidewall_cathode_flow = v.sidewall_cathode_flow + (v.isw + at.gmin * v.vs);
            v.sidewall_cathode_charge = v.sidewall_cathode_charge + v.qs;
        } else {
            v.junction_cathode_flow = v.junction_cathode_flow + (v.id + v.isw + (at.gmin * v.vd).embed::<2>([0]));
            v.junction_cathode_charge = v.junction_cathode_charge + (v.qd.embed::<2>([0]) + v.qs + (v.id + v.isw).scale(prepared.h67));
        }
        if shape.active[0] {
            let current = at.unknown::<R>(shape.junction_current);
            stamp.add_residual(0, current);
            stamp.add_residual(shape.junction, -current);
            stamp.add_conductance(shape.anode_by_junction_current, R::constant(1.0));
            stamp.add_conductance(shape.junction_by_junction_current, R::constant(-1.0));
            stamp.add_residual(shape.junction_current, at.unknown::<R>(0) - at.unknown::<R>(shape.junction) - v.junction_current_potential.value);
            stamp.add_conductance(shape.junction_current_by_anode, R::constant(1.0));
            stamp.add_conductance(shape.junction_current_by_junction, R::constant(-1.0));
            let [t0026] = v.junction_current_potential.gradient;
            stamp.add_conductance(shape.junction_current_by_junction_current, -t0026);
        }
        if shape.active[1] {
            let current = at.unknown::<R>(shape.sidewall_current);
            stamp.add_residual(0, current);
            stamp.add_residual(shape.sidewall, -current);
            stamp.add_conductance(shape.anode_by_sidewall_current, R::constant(1.0));
            stamp.add_conductance(shape.sidewall_by_sidewall_current, R::constant(-1.0));
            stamp.add_residual(shape.sidewall_current, at.unknown::<R>(0) - at.unknown::<R>(shape.sidewall) - v.sidewall_current_potential.value);
            stamp.add_conductance(shape.sidewall_current_by_anode, R::constant(1.0));
            stamp.add_conductance(shape.sidewall_current_by_sidewall, R::constant(-1.0));
            let [t0027] = v.sidewall_current_potential.gradient;
            stamp.add_conductance(shape.sidewall_current_by_sidewall_current, -t0027);
        }
        if shape.active[2] {
            let t0028 = v.anode_junction_flow.value;
            stamp.add_residual(0, t0028);
            stamp.add_residual(shape.junction, -t0028);
            let [t0029] = v.anode_junction_flow.gradient;
            stamp.add_conductance(shape.anode_by_anode, t0029);
            stamp.add_conductance(shape.anode_by_junction, -t0029);
            stamp.add_conductance(shape.junction_by_anode, -t0029);
            stamp.add_conductance(shape.junction_by_junction, t0029);
        }
        if shape.active[4] {
            let t0030 = v.junction_cathode_flow.extend([v.applied0 - v.limited0, v.applied1 - v.limited1]);
            stamp.add_residual(shape.junction, t0030);
            stamp.add_residual(1, -t0030);
            let [t0031, t0032] = v.junction_cathode_flow.gradient;
            stamp.add_conductance(shape.junction_by_junction, t0031);
            stamp.add_conductance(shape.junction_by_cathode, -t0031);
            stamp.add_conductance(shape.cathode_by_junction, -t0031);
            stamp.add_conductance(shape.cathode_by_cathode, t0031);
            stamp.add_conductance(shape.junction_by_sidewall, t0032);
            stamp.add_conductance(shape.junction_by_cathode, -t0032);
            stamp.add_conductance(shape.cathode_by_sidewall, -t0032);
            stamp.add_conductance(shape.cathode_by_cathode, t0032);
            let t0033 = v.junction_cathode_charge.extend([v.applied0 - v.limited0, v.applied1 - v.limited1]);
            stamp.add_charge(shape.junction, t0033);
            stamp.add_charge(1, -t0033);
            let [t0034, t0035] = v.junction_cathode_charge.gradient;
            stamp.add_capacitance(shape.junction_by_junction, t0034);
            stamp.add_capacitance(shape.junction_by_cathode, -t0034);
            stamp.add_capacitance(shape.cathode_by_junction, -t0034);
            stamp.add_capacitance(shape.cathode_by_cathode, t0034);
            stamp.add_capacitance(shape.junction_by_sidewall, t0035);
            stamp.add_capacitance(shape.junction_by_cathode, -t0035);
            stamp.add_capacitance(shape.cathode_by_sidewall, -t0035);
            stamp.add_capacitance(shape.cathode_by_cathode, t0035);
        }
        if shape.active[5] {
            let t0036 = v.sidewall_cathode_flow.extend([v.applied0 - v.limited0, v.applied1 - v.limited1]);
            stamp.add_residual(shape.sidewall, t0036);
            stamp.add_residual(1, -t0036);
            let [t0037, t0038] = v.sidewall_cathode_flow.gradient;
            stamp.add_conductance(shape.sidewall_by_junction, t0037);
            stamp.add_conductance(shape.sidewall_by_cathode, -t0037);
            stamp.add_conductance(shape.cathode_by_junction, -t0037);
            stamp.add_conductance(shape.cathode_by_cathode, t0037);
            stamp.add_conductance(shape.sidewall_by_sidewall, t0038);
            stamp.add_conductance(shape.sidewall_by_cathode, -t0038);
            stamp.add_conductance(shape.cathode_by_sidewall, -t0038);
            stamp.add_conductance(shape.cathode_by_cathode, t0038);
            let t0039 = v.sidewall_cathode_charge.extend([v.applied0 - v.limited0, v.applied1 - v.limited1]);
            stamp.add_charge(shape.sidewall, t0039);
            stamp.add_charge(1, -t0039);
            let [t0040, t0041] = v.sidewall_cathode_charge.gradient;
            stamp.add_capacitance(shape.sidewall_by_junction, t0040);
            stamp.add_capacitance(shape.sidewall_by_cathode, -t0040);
            stamp.add_capacitance(shape.cathode_by_junction, -t0040);
            stamp.add_capacitance(shape.cathode_by_cathode, t0040);
            stamp.add_capacitance(shape.sidewall_by_sidewall, t0041);
            stamp.add_capacitance(shape.sidewall_by_cathode, -t0041);
            stamp.add_capacitance(shape.cathode_by_sidewall, -t0041);
            stamp.add_capacitance(shape.cathode_by_cathode, t0041);
        }
        if shape.active[7] {
            let t0042 = v.anode_sidewall_flow.value;
            stamp.add_residual(0, t0042);
            stamp.add_residual(shape.sidewall, -t0042);
            let [t0043] = v.anode_sidewall_flow.gradient;
            stamp.add_conductance(shape.anode_by_anode, t0043);
            stamp.add_conductance(shape.anode_by_sidewall, -t0043);
            stamp.add_conductance(shape.sidewall_by_anode, -t0043);
            stamp.add_conductance(shape.sidewall_by_sidewall, t0043);
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "a kernel's function is its Verilog-A source's analog block, which is one block"
    )]
    fn noise<R: Real>(
        shape: &DiodeShape,
        prepared: &PreparedDiode<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        struct Locals<R: Real> {
            vd: R,
            vs: R,
            vr: R,
            id: R,
            isw: R,
            cube: R,
            remaining: R,
        }
        let mut v = Locals {
            vd: R::constant(0.0),
            vs: R::constant(0.0),
            vr: R::constant(0.0),
            id: R::constant(0.0),
            isw: R::constant(0.0),
            cube: R::constant(0.0),
            remaining: R::constant(0.0),
        };
        if !shape.conditions[0] {
            if shape.conditions[1] {
                sources.push(NoiseSource::white("series thermal", [Some(UnknownIndex::new(shape.junction_current)), None], prepared.h70.value()));
            } else {
                sources.push(NoiseSource::white("series thermal", [Some(UnknownIndex::new(0)), Some(UnknownIndex::new(shape.junction))], prepared.h71.value()));
            }
        }
        let t0000 = R::constant(at.value(shape.junction) - at.value(1));
        v.vd = t0000;
        if prepared.h7 && v.vd.value() < -prepared.h8.value() && v.vd.value() < -prepared.matched.value() {
            v.id = -prepared.saturation * (-(prepared.matched + v.vd) / prepared.breakdown_slope).exp();
        } else if v.vd.value() >= -prepared.h8.value() {
            v.id = prepared.saturation * ((v.vd / prepared.slope).exp() - 1.0);
        } else {
            v.cube = prepared.h8 / (v.vd * std::f64::consts::E);
            v.id = -prepared.saturation * (v.cube * v.cube * v.cube + 1.0);
        }
        if prepared.recombination_saturation_current_given {
            if v.vd.value() >= -prepared.h8.value() {
                v.vr = v.vd;
            } else {
                v.vr = -prepared.h8;
            }
            v.remaining = -(v.vr / prepared.potential - 1.0);
            v.id += prepared.recombination * ((v.vr / prepared.recombination_slope).exp() - 1.0) * (v.remaining * v.remaining + 0.005).pow(prepared.half_grading);
        }
        if prepared.h19 {
            v.id -= prepared.tunneling * ((-v.vd / prepared.tunneling_slope).exp() - 1.0);
        }
        if v.vd.value() >= -prepared.h8.value() {
            if prepared.forward_knee_current_given && v.id.value() > 1e-18 {
                v.id = v.id / ((v.id / prepared.forward_knee_current).sqrt() + 1.0);
            }
        } else if prepared.reverse_knee_current_given && v.id.value() < -1e-18 {
            v.id = v.id / ((-v.id / prepared.reverse_knee_current).sqrt() + 1.0);
        }
        if shape.conditions[2] {
            let t0001 = R::constant(at.value(shape.sidewall) - at.value(1));
            v.vs = t0001;
        } else {
            v.vs = v.vd;
        }
        v.isw = R::constant(0.0);
        if prepared.h43 {
            if prepared.h44 && v.vs.value() < -prepared.h45.value() && v.vs.value() < -prepared.matched.value() {
                v.isw = -prepared.sidewall_saturation * (-(prepared.matched + v.vs) / prepared.breakdown_slope).exp();
            } else if v.vs.value() >= -prepared.h45.value() {
                v.isw = prepared.sidewall_saturation * ((v.vs / prepared.sidewall_slope).exp() - 1.0);
            } else {
                v.cube = prepared.h45 / (v.vs * std::f64::consts::E);
                v.isw = -prepared.sidewall_saturation * (v.cube * v.cube * v.cube + 1.0);
            }
        }
        if prepared.h48 {
            v.isw -= prepared.sidewall_tunneling * ((-v.vs / prepared.tunneling_slope).exp() - 1.0);
        }
        if prepared.sidewall_forward_knee_current_given && v.isw.value() > 1e-18 {
            v.isw = v.isw / ((v.isw / prepared.sidewall_forward_knee_current).sqrt() + 1.0);
        }
        if shape.conditions[2] {
            sources.push(NoiseSource::white("shot", [Some(UnknownIndex::new(shape.junction)), Some(UnknownIndex::new(1))], (v.id.abs() * 3.204_353_268e-19).value()));
            if prepared.h72 {
                sources.push(NoiseSource {
                    name: "flicker",
                    equations: [Some(UnknownIndex::new(shape.junction)), Some(UnknownIndex::new(1))],
                    white: 0.0,
                    flicker: (prepared.flicker_coefficient * v.id.abs().pow(prepared.flicker_exponent)).value(),
                    exponent: 1.0,
                });
            }
            if !shape.conditions[3] {
                if shape.conditions[4] {
                    sources.push(NoiseSource::white("sidewall series thermal", [Some(UnknownIndex::new(shape.sidewall_current)), None], prepared.h75.value()));
                } else {
                    sources.push(NoiseSource::white("sidewall series thermal", [Some(UnknownIndex::new(0)), Some(UnknownIndex::new(shape.sidewall))], prepared.h76.value()));
                }
            }
            sources.push(NoiseSource::white("sidewall shot", [Some(UnknownIndex::new(shape.sidewall)), Some(UnknownIndex::new(1))], (v.isw.abs() * 3.204_353_268e-19).value()));
            if prepared.h72 {
                sources.push(NoiseSource {
                    name: "sidewall flicker",
                    equations: [Some(UnknownIndex::new(shape.sidewall)), Some(UnknownIndex::new(1))],
                    white: 0.0,
                    flicker: (prepared.flicker_coefficient * v.isw.abs().pow(prepared.flicker_exponent)).value(),
                    exponent: 1.0,
                });
            }
        } else {
            sources.push(NoiseSource::white("shot", [Some(UnknownIndex::new(shape.junction)), Some(UnknownIndex::new(1))], ((v.id + v.isw).abs() * 3.204_353_268e-19).value()));
            if prepared.h72 {
                sources.push(NoiseSource {
                    name: "flicker",
                    equations: [Some(UnknownIndex::new(shape.junction)), Some(UnknownIndex::new(1))],
                    white: 0.0,
                    flicker: (prepared.flicker_coefficient * (v.id + v.isw).abs().pow(prepared.flicker_exponent)).value(),
                    exponent: 1.0,
                });
            }
        }
    }
}
