//! The Gummel-Poon bipolar transistor, derived from ngspice 47's implementation of it in
//! `src/spicelib/devices/bjt/`:
//! Copyright 1985 - 2018, Regents of the University of California and others. Used under
//! ngspice's Modified BSD license, which the repository's `NOTICE` reproduces.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::kernel::dual::Dual;
use crate::kernel::junction::{
    Depletion, Temperatures, below_one, critical_voltage, depletion_charge, junction_current,
    limit_junction,
};
use crate::kernel::parts::SMALL_RESISTANCE;
use crate::kernel::parts::{
    self, CurrentBranch, Derivatives, Nodes, Oriented, Series, series_value,
};
use crate::kernel::region::Bipolar;
use crate::noise::Flicker;
use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Lift, Local, Nature, NoiseSource,
    ParameterError, REFERENCE_TEMPERATURE, Real, Regions, Stamp, UnknownIndex,
};

/// The doping order of a bipolar transistor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Polarity {
    /// An NPN transistor, conducting with its base above its emitter.
    Npn,
    /// A PNP transistor, the NPN's mirror: every voltage and current reversed.
    Pnp,
}

impl Polarity {
    /// `1` for NPN, `−1` for PNP: the factor that orients voltages and currents as an
    /// NPN's.
    #[must_use]
    pub const fn sign(self) -> f64 {
        match self {
            Self::Npn => 1.0,
            Self::Pnp => -1.0,
        }
    }
}

/// How a bipolar transistor is built, which sets where its junction to the substrate lies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Construction {
    /// Built vertically, with the junction to the substrate at the collector.
    Vertical,
    /// Built laterally, with the junction to the substrate at the base.
    Lateral,
}

/// How a bipolar transistor's saturation currents and betas follow the temperature:
/// SPICE's `TLEV`, as ngspice's `bjttemp.c` computes each (lines 162 to 253).
///
/// Either way, a saturation current `I` of emission coefficient `n` scales by
/// `exp(((T/Tn − 1) · Eg / Vt + XTI · ln(T/Tn)) / n)`, and the leakage currents, `ISE` and
/// `ISC`, are divided by the betas' factor. ngspice's `TLEV` 3, which raises each saturation
/// current to the power `1 + TIS1·ΔT + TIS2·ΔT²`, a power that does not commute with the
/// area a SPICE element scales the current by, is not modeled; nor is 2, for which ngspice
/// computes no saturation current.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize, JsonSchema)]
#[serde(try_from = "u8", into = "u8")]
pub enum BipolarTemperatureModel {
    /// `TLEV` 0: the betas scale by `(T/Tn)^XTB`.
    #[default]
    Zero,
    /// `TLEV` 1: the betas scale by `1 + XTB · ΔT`.
    One,
}

impl TryFrom<u8> for BipolarTemperatureModel {
    type Error = String;

    fn try_from(tlev: u8) -> Result<Self, String> {
        match tlev {
            0 => Ok(Self::Zero),
            1 => Ok(Self::One),
            _ => Err(format!(
                "a bipolar transistor's temperature_model is 0 or 1, not {tlev}"
            )),
        }
    }
}

impl From<BipolarTemperatureModel> for u8 {
    fn from(tlev: BipolarTemperatureModel) -> Self {
        match tlev {
            BipolarTemperatureModel::Zero => 0,
            BipolarTemperatureModel::One => 1,
        }
    }
}

/// How a bipolar transistor's depletion capacitances and built-in potentials follow the
/// temperature: SPICE's `TLEVC`, as ngspice's `bjttemp.c` computes each (lines 260 to 313).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize, JsonSchema)]
#[serde(try_from = "u8", into = "u8")]
pub enum BipolarCapacitanceTemperatureModel {
    /// `TLEVC` 0: through silicon's band gap and intrinsic concentration.
    #[default]
    Zero,
    /// `TLEVC` 1: linearly, each capacitance by `1 + CT · ΔT` and each potential falling
    /// by `TVJ · ΔT`, their own coefficients.
    One,
}

impl TryFrom<u8> for BipolarCapacitanceTemperatureModel {
    type Error = String;

    fn try_from(tlevc: u8) -> Result<Self, String> {
        match tlevc {
            0 => Ok(Self::Zero),
            1 => Ok(Self::One),
            _ => Err(format!(
                "a bipolar transistor's capacitance_temperature_model is 0 or 1, not {tlevc}"
            )),
        }
    }
}

impl From<BipolarCapacitanceTemperatureModel> for u8 {
    fn from(tlevc: BipolarCapacitanceTemperatureModel) -> Self {
        match tlevc {
            BipolarCapacitanceTemperatureModel::Zero => 0,
            BipolarCapacitanceTemperatureModel::One => 1,
        }
    }
}

/// The largest grading coefficient ngspice takes at a transistor's temperature: it limits
/// one its temperature coefficients raise beyond to it, with a warning, as a depletion
/// charge divides by `1 − m` (`bjttemp.c`, lines 133 to 146).
const MAX_GRADING: f64 = 0.999;

/// A bipolar junction transistor, in the Gummel-Poon model of SPICE, as ngspice implements
/// it.
///
/// Two junctions, base-emitter and base-collector, each carry an ideal
/// junction current and a leakage one. The transport current from
/// collector to emitter, `(Ibe − Ibc) / qb`, falls with the base charge `qb`, which models
/// the Early effect and high injection. The base resistance falls from `rb` towards `rbm`
/// as the base charge or the base current grows. Each junction holds a depletion charge,
/// and a diffusion charge set by its transit time; the forward transit time grows with
/// the current and the base-collector voltage. Part of the base-collector depletion charge
/// sits on the external base. A base, collector or emitter resistance adds an internal
/// node, labelled `base`, `collector` or `emitter`; one below [`SMALL_RESISTANCE`], the
/// base's at its minimum, also carries its current as an unknown, labelled `base_current`,
/// `collector_current` or `emitter_current`.
///
/// A transistor with a substrate has a fourth terminal, `substrate`, joined by a junction
/// to its internal collector if it is vertical or to its internal base if it is lateral.
/// The junction conducts the minimum conductance, and its diode's current if it has a
/// saturation current; its depletion capacitance grows linearly in forward bias.
///
/// An excess phase delays the forward transport current, `Ibe / qb`, through Weil's
/// approximation as SPICE makes it, a second-order Bessel filter,
/// `3 / (s² · td² + 3 · s · td + 3)`: its group delay at low frequencies is `td`, and its
/// phase at the frequency `1 / (2π · TF)` nearly the excess phase. Two unknowns carry the
/// filter, labelled `delayed_transport` and `delayed_transport_slope`.
///
/// Away from its nominal temperature, its saturation currents scale with the energy gap,
/// its betas by their temperature exponent, exponentially or linearly as its
/// [temperature model](BipolarTemperatureModel) says, and its junctions' potentials and
/// capacitances with silicon's band gap or linearly, as its
/// [capacitance temperature model](BipolarCapacitanceTemperatureModel) says; each other
/// parameter that HSPICE gives temperature coefficients, its resistances, Early voltages,
/// knee currents, emission and grading coefficients and transit times, and a beta given
/// coefficients of its own, scales by `1 + c1·ΔT + c2·ΔT²`, as ngspice 47's `bjttemp.c`
/// scales them.
#[derive(Debug, Clone, Copy)]
pub struct BipolarTransistor;

/// Where a transistor's contributions go among its local unknowns: through its collector
/// and emitter resistances and its base resistance, among the base terminal and the
/// collector, base and emitter behind the resistances, and through its excess phase's
/// filter and its substrate junction, if it has them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BipolarTransistorShape {
    stamp: Nodes<STAMPED>,
    collector_series: Series,
    emitter_series: Series,
    base: Option<BaseForm>,
    excess: Option<ExcessPhase>,
    /// The substrate, then the internal collector or base.
    substrate: Option<Nodes<2>>,
}

/// A transistor's numbers at its temperature.
#[derive(Debug, Clone, Copy)]
pub struct PreparedBipolarTransistor<R: Real> {
    sign: f64,
    /// The collector and the emitter resistances' conductances, or each resistance where
    /// its current is an unknown.
    series: [R; 2],
    base: Option<BaseResistance<R>>,
    junctions: Junctions<R>,
    charges: Option<Charges<R>>,
    /// The delay of the excess phase, in seconds.
    delay: R,
    substrate: Option<SubstrateJunction<R>>,
    /// The thermal voltage at the transistor's temperature.
    thermal_voltage: f64,
    critical_voltage: f64,
    temperature: f64,
    flicker: Flicker,
}

/// What a transistor remembers.
#[derive(Debug, Clone, Copy, Default)]
pub struct BipolarTransistorMemory {
    /// The base-emitter and base-collector voltages of the last evaluation, from which the
    /// next steps are limited.
    last: [f64; JUNCTIONS],
    /// The substrate junction's voltage of the last evaluation.
    substrate: f64,
    /// The values of [`Bipolar::VARIABLE_NAMES`] at the last evaluation.
    variables: [f64; 3],
}

/// The indices of the terminals a substrate junction may join: the collector and the base.
const COLLECTOR: usize = 0;
const BASE: usize = 1;

/// Local node indices: the base terminal, then the collector, base and emitter behind their
/// resistances, which are the terminals themselves without one.
const OUTER_BASE: usize = 0;
const INNER_COLLECTOR: usize = 1;
const INNER_BASE: usize = 2;
const INNER_EMITTER: usize = 3;

/// The number of stamped nodes.
const STAMPED: usize = 4;

/// The number of voltages the transistor's equations take as variables.
const VARIABLE_COUNT: usize = 4;

/// The number of those voltages, the first, across junctions, whose steps are limited.
const JUNCTIONS: usize = 2;

/// The pairs of stamped nodes across which the variables lie: the base-emitter and the
/// base-collector junctions, the base resistance, and the external base-collector
/// depletion, from the base terminal to the internal collector.
const VARIABLES: [(usize, usize); VARIABLE_COUNT] = [
    BASE_EMITTER,
    BASE_COLLECTOR,
    BASE_SERIES,
    EXTERNAL_BASE_COLLECTOR,
];
const BASE_EMITTER: (usize, usize) = (INNER_BASE, INNER_EMITTER);
const BASE_COLLECTOR: (usize, usize) = (INNER_BASE, INNER_COLLECTOR);
const BASE_SERIES: (usize, usize) = (OUTER_BASE, INNER_BASE);
const EXTERNAL_BASE_COLLECTOR: (usize, usize) = (OUTER_BASE, INNER_COLLECTOR);

/// The variables' positions among [`VARIABLES`], by ngspice's names, and `Vrb` for the
/// drop across the base resistance.
const VBE: usize = 0;
const VBC: usize = 1;
const VRB: usize = 2;
const VBX: usize = 3;

/// The variables each law of the transistor's equations reads, by their positions among
/// [`VARIABLES`], in the order its numbers carry their derivatives: each law computes at
/// the width of those alone.
///
/// The transport currents, the base resistance and the base-emitter charge read both
/// junctions.
const INTRINSIC: [usize; 2] = [VBE, VBC];
/// The base resistance's current, where it is a conductance, reads its own drop, then the
/// junctions the resistance falls with.
const BASE_RESISTANCE: [usize; 3] = [VRB, VBE, VBC];

/// The variables of one evaluation.
type Bias = parts::Bias<VARIABLE_COUNT>;

/// The parameters of a [`BipolarTransistor`], with SPICE's defaults. A parameter SPICE
/// treats as infinite at zero, such as an Early voltage, is infinite when left out.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift)]
#[serde(default, deny_unknown_fields, bound = "")]
pub struct BipolarTransistorParameters<R: Real = f64> {
    /// The doping order; default `npn`.
    pub polarity: Polarity,
    /// The transport saturation current, `IS`, in amperes; default 1e-16.
    pub saturation_current: R,
    /// The ideal maximum forward current gain, `BF`; default 100.
    pub forward_beta: R,
    /// The ideal maximum reverse current gain, `BR`; default 1.
    pub reverse_beta: R,
    /// The forward emission coefficient, `NF`; default 1.
    pub forward_emission_coefficient: R,
    /// The reverse emission coefficient, `NR`; default 1.
    pub reverse_emission_coefficient: R,
    /// The forward Early voltage, `VAF`, in volts; infinite by default.
    pub forward_early_voltage: Option<R>,
    /// The reverse Early voltage, `VAR`, in volts; infinite by default.
    pub reverse_early_voltage: Option<R>,
    /// The forward knee current, `IKF`, where high injection sets in, in amperes; infinite
    /// by default.
    pub forward_knee_current: Option<R>,
    /// The reverse knee current, `IKR`, in amperes; infinite by default.
    pub reverse_knee_current: Option<R>,
    /// The base-emitter leakage saturation current, `ISE`, in amperes; default 0.
    pub base_emitter_leakage_current: R,
    /// The base-emitter leakage emission coefficient, `NE`; default 1.5.
    pub base_emitter_leakage_emission_coefficient: R,
    /// The base-collector leakage saturation current, `ISC`, in amperes; default 0.
    pub base_collector_leakage_current: R,
    /// The base-collector leakage emission coefficient, `NC`; default 2.
    pub base_collector_leakage_emission_coefficient: R,
    /// The zero-bias base resistance, `RB`, in ohms; default 0.
    pub base_resistance: R,
    /// The base resistance at high current, `RBM`, in ohms; the base resistance by default.
    pub min_base_resistance: Option<R>,
    /// The base current, `IRB`, at which the base resistance falls halfway to its minimum,
    /// in amperes; without it, the base resistance falls with the base charge.
    pub base_resistance_current: Option<R>,
    /// The emitter resistance, `RE`, in ohms; default 0.
    pub emitter_resistance: R,
    /// The collector resistance, `RC`, in ohms; default 0.
    pub collector_resistance: R,
    /// The zero-bias base-emitter depletion capacitance, `CJE`, in farads; default 0.
    pub base_emitter_capacitance: R,
    /// The base-emitter built-in potential, `VJE`, in volts; default 0.75.
    pub base_emitter_potential: R,
    /// The base-emitter grading coefficient, `MJE`, below 1; default 0.33.
    pub base_emitter_grading: R,
    /// The ideal forward transit time, `TF`, in seconds; default 0.
    pub forward_transit_time: R,
    /// How much the forward transit time grows with bias, `XTF`; default 0.
    pub transit_time_bias_coefficient: R,
    /// The base-collector voltage, `VTF`, that sets how the forward transit time grows with
    /// it, in volts; infinite by default.
    pub transit_time_voltage: Option<R>,
    /// The current, `ITF`, above which the forward transit time grows, in amperes;
    /// default 0.
    pub transit_time_current: R,
    /// The zero-bias base-collector depletion capacitance, `CJC`, in farads; default 0.
    pub base_collector_capacitance: R,
    /// The base-collector built-in potential, `VJC`, in volts; default 0.75.
    pub base_collector_potential: R,
    /// The base-collector grading coefficient, `MJC`, below 1; default 0.33.
    pub base_collector_grading: R,
    /// The fraction of the base-collector capacitance on the internal base, `XCJC`, the
    /// rest on the external base; default 1.
    pub internal_base_fraction: R,
    /// The ideal reverse transit time, `TR`, in seconds; default 0.
    pub reverse_transit_time: R,
    /// The fraction of each junction's potential above which its depletion capacitance is
    /// extended linearly, `FC`, below 1; default 0.5.
    pub depletion_coefficient: R,
    /// The excess phase at the frequency `1 / (2π · TF)`, `PTF`, in degrees, which delays
    /// the forward transport current by `PTF · π / 180 · TF`; default 0.
    pub excess_phase: R,
    /// How the transistor is built, if it has a substrate, as ngspice's `SUBS` of 1 and −1
    /// builds it; without it, the transistor has none. The parameters below, of the
    /// junction to the substrate, need one.
    pub substrate: Option<Construction>,
    /// The substrate junction's saturation current, `ISS`, in amperes; without it, the
    /// junction carries the minimum conductance only.
    pub substrate_saturation_current: Option<R>,
    /// The substrate junction's emission coefficient, `NS`; default 1.
    pub substrate_emission_coefficient: Option<R>,
    /// The substrate junction's zero-bias depletion capacitance, `CJS`, in farads;
    /// default 0.
    pub substrate_capacitance: Option<R>,
    /// The substrate junction's built-in potential, `VJS`, in volts; default 0.75.
    pub substrate_potential: Option<R>,
    /// The substrate junction's grading coefficient, `MJS`, below 1; default 0.
    pub substrate_grading: Option<R>,
    /// The energy gap that scales the saturation currents with temperature, `EG`, in
    /// volts; default 1.11, silicon's.
    pub energy_gap: R,
    /// The exponent of the temperature ratio in the saturation currents' scaling, `XTI`;
    /// default 3.
    pub saturation_current_temperature_exponent: R,
    /// The exponent of the temperature ratio that scales the betas where `TLEV` is 0, or
    /// their linear temperature coefficient, per kelvin, where it is 1, `XTB`; default 0.
    pub beta_temperature_exponent: R,
    /// How the saturation currents and the betas follow the temperature, `TLEV`; default
    /// 0.
    pub temperature_model: BipolarTemperatureModel,
    /// How the junctions' depletion capacitances and built-in potentials follow the
    /// temperature, `TLEVC`; default 0.
    pub capacitance_temperature_model: BipolarCapacitanceTemperatureModel,
    /// The exponent of high injection in the base charge, `NKF`, above 0 and at most 1:
    /// `qb = q1 · (1 + (1 + 4 · q2)^NKF) / 2`; default 0.5, a square root.
    pub high_injection_exponent: R,
    /// The forward beta's linear temperature coefficient, `TBF1`, per kelvin: given,
    /// with `TBF2` or without, the beta scales by `1 + TBF1·ΔT + TBF2·ΔT²` from the
    /// nominal temperature instead of by `XTB`'s factor; without either, by `XTB`'s
    /// factor.
    pub forward_beta_temperature_coefficient: Option<R>,
    /// The forward beta's quadratic temperature coefficient, `TBF2`, per kelvin
    /// squared, as `TBF1` says.
    pub forward_beta_quadratic_temperature_coefficient: Option<R>,
    /// The reverse beta's linear temperature coefficient, `TBR1`, per kelvin: given,
    /// with `TBR2` or without, the beta scales by `1 + TBR1·ΔT + TBR2·ΔT²` from the
    /// nominal temperature instead of by `XTB`'s factor; without either, by `XTB`'s
    /// factor.
    pub reverse_beta_temperature_coefficient: Option<R>,
    /// The reverse beta's quadratic temperature coefficient, `TBR2`, per kelvin
    /// squared, as `TBR1` says.
    pub reverse_beta_quadratic_temperature_coefficient: Option<R>,
    /// The forward knee current's linear temperature coefficient, `TIKF1`, per kelvin:
    /// it scales by `1 + TIKF1·ΔT + TIKF2·ΔT²` from the nominal temperature; default 0.
    pub forward_knee_current_temperature_coefficient: R,
    /// The forward knee current's quadratic temperature coefficient, `TIKF2`, per
    /// kelvin squared; default 0.
    pub forward_knee_current_quadratic_temperature_coefficient: R,
    /// The reverse knee current's linear temperature coefficient, `TIKR1`, per kelvin:
    /// it scales by `1 + TIKR1·ΔT + TIKR2·ΔT²` from the nominal temperature; default 0.
    pub reverse_knee_current_temperature_coefficient: R,
    /// The reverse knee current's quadratic temperature coefficient, `TIKR2`, per
    /// kelvin squared; default 0.
    pub reverse_knee_current_quadratic_temperature_coefficient: R,
    /// The forward Early voltage's linear temperature coefficient, `TVAF1`, per kelvin:
    /// it scales by `1 + TVAF1·ΔT + TVAF2·ΔT²` from the nominal temperature; default 0.
    pub forward_early_voltage_temperature_coefficient: R,
    /// The forward Early voltage's quadratic temperature coefficient, `TVAF2`, per
    /// kelvin squared; default 0.
    pub forward_early_voltage_quadratic_temperature_coefficient: R,
    /// The reverse Early voltage's linear temperature coefficient, `TVAR1`, per kelvin:
    /// it scales by `1 + TVAR1·ΔT + TVAR2·ΔT²` from the nominal temperature; default 0.
    pub reverse_early_voltage_temperature_coefficient: R,
    /// The reverse Early voltage's quadratic temperature coefficient, `TVAR2`, per
    /// kelvin squared; default 0.
    pub reverse_early_voltage_quadratic_temperature_coefficient: R,
    /// The zero-bias base resistance's linear temperature coefficient, `TRB1`, per
    /// kelvin: it scales by `1 + TRB1·ΔT + TRB2·ΔT²` from the nominal temperature;
    /// default 0.
    pub base_resistance_temperature_coefficient: R,
    /// The zero-bias base resistance's quadratic temperature coefficient, `TRB2`, per
    /// kelvin squared; default 0.
    pub base_resistance_quadratic_temperature_coefficient: R,
    /// The minimum base resistance's linear temperature coefficient, `TRM1`, per
    /// kelvin: it scales by `1 + TRM1·ΔT + TRM2·ΔT²` from the nominal temperature;
    /// default 0.
    pub min_base_resistance_temperature_coefficient: R,
    /// The minimum base resistance's quadratic temperature coefficient, `TRM2`, per
    /// kelvin squared; default 0.
    pub min_base_resistance_quadratic_temperature_coefficient: R,
    /// The base resistance's halfway current's linear temperature coefficient, `TIRB1`,
    /// per kelvin: it scales by `1 + TIRB1·ΔT + TIRB2·ΔT²` from the nominal
    /// temperature; default 0.
    pub base_resistance_current_temperature_coefficient: R,
    /// The base resistance's halfway current's quadratic temperature coefficient,
    /// `TIRB2`, per kelvin squared; default 0.
    pub base_resistance_current_quadratic_temperature_coefficient: R,
    /// The collector resistance's linear temperature coefficient, `TRC1`, per kelvin:
    /// it scales by `1 + TRC1·ΔT + TRC2·ΔT²` from the nominal temperature; default 0.
    pub collector_resistance_temperature_coefficient: R,
    /// The collector resistance's quadratic temperature coefficient, `TRC2`, per kelvin
    /// squared; default 0.
    pub collector_resistance_quadratic_temperature_coefficient: R,
    /// The emitter resistance's linear temperature coefficient, `TRE1`, per kelvin: it
    /// scales by `1 + TRE1·ΔT + TRE2·ΔT²` from the nominal temperature; default 0.
    pub emitter_resistance_temperature_coefficient: R,
    /// The emitter resistance's quadratic temperature coefficient, `TRE2`, per kelvin
    /// squared; default 0.
    pub emitter_resistance_quadratic_temperature_coefficient: R,
    /// The forward emission coefficient's linear temperature coefficient, `TNF1`, per
    /// kelvin: it scales by `1 + TNF1·ΔT + TNF2·ΔT²` from the nominal temperature;
    /// default 0.
    pub forward_emission_coefficient_temperature_coefficient: R,
    /// The forward emission coefficient's quadratic temperature coefficient, `TNF2`,
    /// per kelvin squared; default 0.
    pub forward_emission_coefficient_quadratic_temperature_coefficient: R,
    /// The reverse emission coefficient's linear temperature coefficient, `TNR1`, per
    /// kelvin: it scales by `1 + TNR1·ΔT + TNR2·ΔT²` from the nominal temperature;
    /// default 0.
    pub reverse_emission_coefficient_temperature_coefficient: R,
    /// The reverse emission coefficient's quadratic temperature coefficient, `TNR2`,
    /// per kelvin squared; default 0.
    pub reverse_emission_coefficient_quadratic_temperature_coefficient: R,
    /// The base-emitter leakage emission coefficient's linear temperature coefficient,
    /// `TNE1`, per kelvin: it scales by `1 + TNE1·ΔT + TNE2·ΔT²` from the nominal
    /// temperature; default 0.
    pub base_emitter_leakage_emission_coefficient_temperature_coefficient: R,
    /// The base-emitter leakage emission coefficient's quadratic temperature
    /// coefficient, `TNE2`, per kelvin squared; default 0.
    pub base_emitter_leakage_emission_coefficient_quadratic_temperature_coefficient: R,
    /// The base-collector leakage emission coefficient's linear temperature
    /// coefficient, `TNC1`, per kelvin: it scales by `1 + TNC1·ΔT + TNC2·ΔT²` from the
    /// nominal temperature; default 0.
    pub base_collector_leakage_emission_coefficient_temperature_coefficient: R,
    /// The base-collector leakage emission coefficient's quadratic temperature
    /// coefficient, `TNC2`, per kelvin squared; default 0.
    pub base_collector_leakage_emission_coefficient_quadratic_temperature_coefficient: R,
    /// The substrate junction's emission coefficient's linear temperature coefficient,
    /// `TNS1`, per kelvin: it scales by `1 + TNS1·ΔT + TNS2·ΔT²` from the nominal
    /// temperature; default 0.
    pub substrate_emission_coefficient_temperature_coefficient: R,
    /// The substrate junction's emission coefficient's quadratic temperature
    /// coefficient, `TNS2`, per kelvin squared; default 0.
    pub substrate_emission_coefficient_quadratic_temperature_coefficient: R,
    /// The transit time's current's linear temperature coefficient, `TITF1`, per
    /// kelvin: it scales by `1 + TITF1·ΔT + TITF2·ΔT²` from the nominal temperature;
    /// default 0.
    pub transit_time_current_temperature_coefficient: R,
    /// The transit time's current's quadratic temperature coefficient, `TITF2`, per
    /// kelvin squared; default 0.
    pub transit_time_current_quadratic_temperature_coefficient: R,
    /// The forward transit time's linear temperature coefficient, `TTF1`, per kelvin:
    /// it scales by `1 + TTF1·ΔT + TTF2·ΔT²` from the nominal temperature; default 0.
    pub forward_transit_time_temperature_coefficient: R,
    /// The forward transit time's quadratic temperature coefficient, `TTF2`, per kelvin
    /// squared; default 0.
    pub forward_transit_time_quadratic_temperature_coefficient: R,
    /// The reverse transit time's linear temperature coefficient, `TTR1`, per kelvin:
    /// it scales by `1 + TTR1·ΔT + TTR2·ΔT²` from the nominal temperature; default 0.
    pub reverse_transit_time_temperature_coefficient: R,
    /// The reverse transit time's quadratic temperature coefficient, `TTR2`, per kelvin
    /// squared; default 0.
    pub reverse_transit_time_quadratic_temperature_coefficient: R,
    /// The base-emitter grading coefficient's linear temperature coefficient, `TMJE1`,
    /// per kelvin: it scales by `1 + TMJE1·ΔT + TMJE2·ΔT²` from the nominal
    /// temperature; default 0.
    pub base_emitter_grading_temperature_coefficient: R,
    /// The base-emitter grading coefficient's quadratic temperature coefficient,
    /// `TMJE2`, per kelvin squared; default 0.
    pub base_emitter_grading_quadratic_temperature_coefficient: R,
    /// The base-collector grading coefficient's linear temperature coefficient,
    /// `TMJC1`, per kelvin: it scales by `1 + TMJC1·ΔT + TMJC2·ΔT²` from the nominal
    /// temperature; default 0.
    pub base_collector_grading_temperature_coefficient: R,
    /// The base-collector grading coefficient's quadratic temperature coefficient,
    /// `TMJC2`, per kelvin squared; default 0.
    pub base_collector_grading_quadratic_temperature_coefficient: R,
    /// The substrate junction's grading coefficient's linear temperature coefficient,
    /// `TMJS1`, per kelvin: it scales by `1 + TMJS1·ΔT + TMJS2·ΔT²` from the nominal
    /// temperature; default 0.
    pub substrate_grading_temperature_coefficient: R,
    /// The substrate junction's grading coefficient's quadratic temperature
    /// coefficient, `TMJS2`, per kelvin squared; default 0.
    pub substrate_grading_quadratic_temperature_coefficient: R,
    /// The base-emitter depletion capacitance's temperature coefficient where `TLEVC`
    /// is 1, `CTE`, per kelvin; default 0.
    pub base_emitter_capacitance_temperature_coefficient: R,
    /// The base-collector depletion capacitance's temperature coefficient where `TLEVC`
    /// is 1, `CTC`, per kelvin; default 0.
    pub base_collector_capacitance_temperature_coefficient: R,
    /// The substrate junction's depletion capacitance's temperature coefficient where
    /// `TLEVC` is 1, `CTS`, per kelvin; default 0.
    pub substrate_capacitance_temperature_coefficient: R,
    /// The base-emitter built-in potential's temperature coefficient where `TLEVC` is
    /// 1, `TVJE`, in volts per kelvin; default 0.
    pub base_emitter_potential_temperature_coefficient: R,
    /// The base-collector built-in potential's temperature coefficient where `TLEVC` is
    /// 1, `TVJC`, in volts per kelvin; default 0.
    pub base_collector_potential_temperature_coefficient: R,
    /// The substrate junction's built-in potential's temperature coefficient where
    /// `TLEVC` is 1, `TVJS`, in volts per kelvin; default 0.
    pub substrate_potential_temperature_coefficient: R,
    /// The temperature the parameters were measured at, in kelvins; default 27 °C.
    pub nominal_temperature: R,
    /// The flicker noise coefficient of the base current, `KF`; default 0, no flicker
    /// noise.
    pub flicker_coefficient: R,
    /// The power of the base current in the flicker noise, `AF`; default 1.
    pub flicker_exponent: R,
}

impl<R: Real> Default for BipolarTransistorParameters<R> {
    #[expect(
        clippy::too_many_lines,
        reason = "one line for each of the Gummel-Poon model's parameters"
    )]
    fn default() -> Self {
        Self {
            polarity: Polarity::Npn,
            saturation_current: R::constant(1e-16),
            forward_beta: R::constant(100.0),
            reverse_beta: R::constant(1.0),
            forward_emission_coefficient: R::constant(1.0),
            reverse_emission_coefficient: R::constant(1.0),
            forward_early_voltage: None,
            reverse_early_voltage: None,
            forward_knee_current: None,
            reverse_knee_current: None,
            base_emitter_leakage_current: R::constant(0.0),
            base_emitter_leakage_emission_coefficient: R::constant(1.5),
            base_collector_leakage_current: R::constant(0.0),
            base_collector_leakage_emission_coefficient: R::constant(2.0),
            base_resistance: R::constant(0.0),
            min_base_resistance: None,
            base_resistance_current: None,
            emitter_resistance: R::constant(0.0),
            collector_resistance: R::constant(0.0),
            base_emitter_capacitance: R::constant(0.0),
            base_emitter_potential: R::constant(0.75),
            base_emitter_grading: R::constant(0.33),
            forward_transit_time: R::constant(0.0),
            transit_time_bias_coefficient: R::constant(0.0),
            transit_time_voltage: None,
            transit_time_current: R::constant(0.0),
            base_collector_capacitance: R::constant(0.0),
            base_collector_potential: R::constant(0.75),
            base_collector_grading: R::constant(0.33),
            internal_base_fraction: R::constant(1.0),
            reverse_transit_time: R::constant(0.0),
            depletion_coefficient: R::constant(0.5),
            excess_phase: R::constant(0.0),
            substrate: None,
            substrate_saturation_current: None,
            substrate_emission_coefficient: None,
            substrate_capacitance: None,
            substrate_potential: None,
            substrate_grading: None,
            energy_gap: R::constant(1.11),
            saturation_current_temperature_exponent: R::constant(3.0),
            beta_temperature_exponent: R::constant(0.0),
            temperature_model: BipolarTemperatureModel::Zero,
            capacitance_temperature_model: BipolarCapacitanceTemperatureModel::Zero,
            high_injection_exponent: R::constant(0.5),
            forward_beta_temperature_coefficient: None,
            forward_beta_quadratic_temperature_coefficient: None,
            reverse_beta_temperature_coefficient: None,
            reverse_beta_quadratic_temperature_coefficient: None,
            forward_knee_current_temperature_coefficient: R::constant(0.0),
            forward_knee_current_quadratic_temperature_coefficient: R::constant(0.0),
            reverse_knee_current_temperature_coefficient: R::constant(0.0),
            reverse_knee_current_quadratic_temperature_coefficient: R::constant(0.0),
            forward_early_voltage_temperature_coefficient: R::constant(0.0),
            forward_early_voltage_quadratic_temperature_coefficient: R::constant(0.0),
            reverse_early_voltage_temperature_coefficient: R::constant(0.0),
            reverse_early_voltage_quadratic_temperature_coefficient: R::constant(0.0),
            base_resistance_temperature_coefficient: R::constant(0.0),
            base_resistance_quadratic_temperature_coefficient: R::constant(0.0),
            min_base_resistance_temperature_coefficient: R::constant(0.0),
            min_base_resistance_quadratic_temperature_coefficient: R::constant(0.0),
            base_resistance_current_temperature_coefficient: R::constant(0.0),
            base_resistance_current_quadratic_temperature_coefficient: R::constant(0.0),
            collector_resistance_temperature_coefficient: R::constant(0.0),
            collector_resistance_quadratic_temperature_coefficient: R::constant(0.0),
            emitter_resistance_temperature_coefficient: R::constant(0.0),
            emitter_resistance_quadratic_temperature_coefficient: R::constant(0.0),
            forward_emission_coefficient_temperature_coefficient: R::constant(0.0),
            forward_emission_coefficient_quadratic_temperature_coefficient: R::constant(0.0),
            reverse_emission_coefficient_temperature_coefficient: R::constant(0.0),
            reverse_emission_coefficient_quadratic_temperature_coefficient: R::constant(0.0),
            base_emitter_leakage_emission_coefficient_temperature_coefficient: R::constant(0.0),
            base_emitter_leakage_emission_coefficient_quadratic_temperature_coefficient:
                R::constant(0.0),
            base_collector_leakage_emission_coefficient_temperature_coefficient: R::constant(0.0),
            base_collector_leakage_emission_coefficient_quadratic_temperature_coefficient:
                R::constant(0.0),
            substrate_emission_coefficient_temperature_coefficient: R::constant(0.0),
            substrate_emission_coefficient_quadratic_temperature_coefficient: R::constant(0.0),
            transit_time_current_temperature_coefficient: R::constant(0.0),
            transit_time_current_quadratic_temperature_coefficient: R::constant(0.0),
            forward_transit_time_temperature_coefficient: R::constant(0.0),
            forward_transit_time_quadratic_temperature_coefficient: R::constant(0.0),
            reverse_transit_time_temperature_coefficient: R::constant(0.0),
            reverse_transit_time_quadratic_temperature_coefficient: R::constant(0.0),
            base_emitter_grading_temperature_coefficient: R::constant(0.0),
            base_emitter_grading_quadratic_temperature_coefficient: R::constant(0.0),
            base_collector_grading_temperature_coefficient: R::constant(0.0),
            base_collector_grading_quadratic_temperature_coefficient: R::constant(0.0),
            substrate_grading_temperature_coefficient: R::constant(0.0),
            substrate_grading_quadratic_temperature_coefficient: R::constant(0.0),
            base_emitter_capacitance_temperature_coefficient: R::constant(0.0),
            base_collector_capacitance_temperature_coefficient: R::constant(0.0),
            substrate_capacitance_temperature_coefficient: R::constant(0.0),
            base_emitter_potential_temperature_coefficient: R::constant(0.0),
            base_collector_potential_temperature_coefficient: R::constant(0.0),
            substrate_potential_temperature_coefficient: R::constant(0.0),
            nominal_temperature: R::constant(REFERENCE_TEMPERATURE),
            flicker_coefficient: R::constant(0.0),
            flicker_exponent: R::constant(1.0),
        }
    }
}

/// The DC currents of the two junctions and the base charge they share.
#[derive(Debug, Clone, Copy)]
struct Junctions<R: Real> {
    saturation_current: R,
    forward_beta: R,
    reverse_beta: R,
    forward_slope: R,
    reverse_slope: R,
    emitter_leakage: R,
    emitter_leakage_slope: R,
    collector_leakage: R,
    collector_leakage_slope: R,
    /// `1 / VAF` and `1 / VAR`, zero for infinite Early voltages.
    inverse_early: [R; 2],
    /// `1 / IKF` and `1 / IKR`, zero for infinite knee currents.
    inverse_knee: [R; 2],
    /// `NKF`, the exponent of high injection in the base charge.
    high_injection_exponent: R,
}

/// The junctions' DC state at one bias, each number with its derivatives with respect to
/// the junction voltages it reads: its own junction's alone, or [`INTRINSIC`]'s.
#[derive(Debug, Clone, Copy)]
struct State<R: Real> {
    /// The ideal base-emitter and base-collector currents, `Ibe` and `Ibc`, each of its own
    /// junction's voltage.
    ideal: [Dual<R, 1>; 2],
    /// The base current through each junction, to the emitter and to the collector, of its
    /// own junction's voltage.
    base: [Dual<R, 1>; 2],
    /// The transport current from collector to emitter.
    transport: Dual<R, 2>,
    /// Its forward part, `Ibe / qb`, which an excess phase delays.
    forward_transport: Dual<R, 2>,
    /// Its reverse part, `Ibc / qb`, which it leaves: computed apart, not as the
    /// difference of the two above, which cancel.
    reverse_transport: Dual<R, 2>,
    /// The normalized base charge, `qb`.
    charge: Dual<R, 2>,
}

impl<R: Real> Junctions<R> {
    /// The junctions' state at the base-emitter and base-collector `voltages`, with the
    /// minimum conductance `gmin` across each junction.
    fn evaluate(&self, voltages: [R; 2], gmin: f64) -> State<R> {
        let [vbe, vbc] = voltages.map(Dual::<R, 1>::variable);
        let leakage = |voltage, current: R, slope| {
            if current.value() > 0.0 {
                junction_current(voltage, current, slope)
            } else {
                Dual::number(0.0)
            }
        };
        let forward = junction_current(vbe, self.saturation_current, self.forward_slope);
        let reverse = junction_current(vbc, self.saturation_current, self.reverse_slope);
        let emitter_leakage = leakage(vbe, self.emitter_leakage, self.emitter_leakage_slope);
        let collector_leakage = leakage(vbc, self.collector_leakage, self.collector_leakage_slope);
        let base = [
            forward.divide(self.forward_beta) + emitter_leakage + vbe * gmin,
            reverse.divide(self.reverse_beta) + collector_leakage + vbc * gmin,
        ];

        // The base charge: the Early effect through q1, high injection through q2. Within
        // its law, `vbe` comes first and `vbc` second.
        let [early_forward, early_reverse] = self.inverse_early;
        let [knee_forward, knee_reverse] = self.inverse_knee;
        let q1 = (1.0 - vbc.scale(early_forward).embed([1]) - vbe.scale(early_reverse).embed([0]))
            .recip();
        let qb = if knee_forward.value() == 0.0 && knee_reverse.value() == 0.0 {
            q1
        } else {
            let q2 =
                forward.scale(knee_forward).embed([0]) + reverse.scale(knee_reverse).embed([1]);
            let argument = (q2 * 4.0 + 1.0).max(0.0);
            let exponent = self.high_injection_exponent;
            // SPICE keeps the power at 1 where high injection would make it vanish, and
            // takes a square root where `NKF` is a half, its default (`bjtload.c`, lines
            // 571 to 586).
            #[expect(
                clippy::float_cmp,
                reason = "ngspice takes a square root where `NKF` is its default, a half"
            )]
            let power = if argument.value.value() == 0.0 {
                Dual::number(1.0)
            } else if exponent.value() == 0.5 {
                argument.sqrt()
            } else {
                argument.pow(exponent)
            };
            q1 * (power + 1.0) / 2.0
        };
        let [forward_current, reverse_current] = [forward.embed([0]), reverse.embed([1])];
        State {
            ideal: [forward, reverse],
            base,
            transport: (forward_current - reverse_current) / qb,
            forward_transport: forward_current / qb,
            reverse_transport: reverse_current / qb,
            charge: qb,
        }
    }
}

/// How the base resistance joins the base terminal to the internal base.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BaseForm {
    /// A conductance between the two.
    Conductance,
    /// With a minimum below [`SMALL_RESISTANCE`], the base current as a local unknown of
    /// its own, held by `drop − r · i = 0`.
    Current(BaseCurrent),
}

impl BaseForm {
    /// Adds the current through the base `resistance`, a number of the intrinsic
    /// transistor's law, `drop / r`, or the equation of the base current that crosses it,
    /// of the transistor `added` adds for, its derivatives among the stamped nodes summed
    /// into `derivatives`.
    fn add<R: Real>(
        &self,
        resistance: Dual<R, 2>,
        (added, derivatives): (
            &Oriented<'_, STAMPED, VARIABLE_COUNT>,
            &mut Derivatives<R, STAMPED>,
        ),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        match self {
            Self::Current(current) => current.add(resistance, added, at, stamp),
            Self::Conductance => {
                // Within its law, the drop comes first, then the junctions.
                let [_, _, drop, _] = added.moving(at);
                let current = Dual::<R, 1>::variable(drop).embed([0]) / resistance.embed([1, 2]);
                added.reading(BASE_RESISTANCE).add_current(
                    stamp,
                    derivatives,
                    BASE_SERIES,
                    current,
                );
            }
        }
    }
}

/// The base current as a local unknown of its own, and the local entries of its equation
/// at itself and at each stamped node, as the resistance depends on the junction
/// voltages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BaseCurrent {
    branch: CurrentBranch,
    own: usize,
    driven: [usize; STAMPED],
}

impl BaseCurrent {
    /// The current from the base terminal to the internal base, the `stamp`ed nodes'
    /// first and third, labelled `base_current`.
    fn new(layout: &mut Layout, stamp: &Nodes<STAMPED>) -> Self {
        let branch = CurrentBranch::new(
            layout,
            "base_current",
            [stamp.node(OUTER_BASE), stamp.node(INNER_BASE)],
        );
        let row = branch.unknown();
        Self {
            branch,
            own: layout.entry(row, row),
            driven: std::array::from_fn(|index| layout.entry(row, stamp.node(index))),
        }
    }

    /// Adds the current's equation for `resistance`, a number of the intrinsic
    /// transistor's law, of the transistor `added` adds for.
    fn add<R: Real>(
        &self,
        resistance: Dual<R, 2>,
        added: &Oriented<'_, STAMPED, VARIABLE_COUNT>,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let intrinsic = added.reading(INTRINSIC);
        let current = self.branch.current::<R>(at);
        let drop = self.branch.voltage::<R>(at);
        let residual = drop - intrinsic.extend(resistance) * current;
        self.branch.add(at, stamp, residual);
        stamp.add_conductance(self.own, -resistance.value);
        let gradient = resistance
            .gradient
            .map(|slope| slope * -(current * added.sign));
        intrinsic.add_conductances(stamp, &self.driven, gradient);
    }
}

/// The base resistance, which falls from its zero-bias value towards its minimum.
#[derive(Debug, Clone, Copy)]
struct BaseResistance<R: Real> {
    maximum: R,
    minimum: R,
    /// The base current at which the resistance falls halfway, or `None` for a resistance
    /// that falls with the base charge.
    halfway_current: Option<R>,
}

impl<R: Real> BaseResistance<R> {
    /// The resistance at the junctions' `state`, a number of the intrinsic transistor's
    /// law.
    fn evaluate(&self, state: &State<R>) -> Dual<R, 2> {
        let excess = self.maximum - self.minimum;
        let Some(halfway) = self.halfway_current else {
            return (Dual::constant(excess) / state.charge).plus(self.minimum);
        };
        let [base_emitter, base_collector] = state.base;
        // SPICE floors the ratio, below which the resistance no longer changes.
        let ratio = (base_emitter.embed([0]) + base_collector.embed([1]))
            .divide(halfway)
            .max(1e-9);
        ratio
            .apply(|ratio| shape(z_of(ratio)))
            .scale(excess * 3.0)
            .plus(self.minimum)
    }
}

/// SPICE's `z` for a base current `ratio` of the halfway current:
/// `z = (√(1 + 144 x / π²) − 1) / ((24 / π²) · √x)`.
fn z_of<R: Real>(ratio: Dual<R, 1>) -> Dual<R, 1> {
    const A: f64 = 14.590_25;
    const B: f64 = 2.4317;
    let root = (ratio * A + 1.0).sqrt();
    (root - 1.0) / (ratio.sqrt() * B)
}

/// `(tan z − z) / (z · tan² z)`.
///
/// Below `z = 0.25`, where the subtraction loses `3ε / z²` of the precision, a series in
/// `z²` takes over, accurate to within a few ulps there.
fn shape<R: Real>(z: Dual<R, 1>) -> Dual<R, 1> {
    const SERIES: [f64; 7] = [
        1.0 / 3.0,
        -4.0 / 45.0,
        -4.0 / 315.0,
        -8.0 / 4725.0,
        -4.0 / 18711.0,
        -5528.0 / 212_837_625.0,
        -8.0 / 2_606_175.0,
    ];
    if z.value.value() < 0.25 {
        // The series in `w = z²` by Horner's rule.
        let square = z * z;
        return SERIES
            .iter()
            .rev()
            .fold(Dual::number(0.0), |sum, &coefficient| {
                sum * square + coefficient
            });
    }
    let tan = z.tan();
    (tan - z) / (z * tan * tan)
}

/// The forward transport current `u` delayed through the filter of an excess phase: the
/// delayed current `x` and `y = td · dx/dt`, both in amperes, held by `td · dx/dt − y = 0`
/// and `td · dy/dt + 3 · y + 3 · x − 3 · u = 0`, so `td² · x'' + 3 · td · x' + 3 · x = 3 · u`.
/// At the operating point `x` is `u`. Each is a current from the internal collector to the
/// internal emitter, as the transistor's own currents are, so that a PNP's are its NPN
/// mirror's reversed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExcessPhase {
    /// The local unknowns `x` and `y`.
    unknowns: [usize; 2],
    /// The local entries of `x`'s equation at `x` and `y`, and of `y`'s at `x` and `y`.
    own: [[usize; 2]; 2],
    /// The local entries of `y`'s equation at each stamped node.
    driven: [usize; STAMPED],
    /// The local entries of the internal collector's and emitter's equations at `x`.
    delivered: [usize; 2],
}

impl ExcessPhase {
    /// The filter, driven by the transport current among the `stamp`ed nodes.
    fn new(layout: &mut Layout, stamp: &Nodes<STAMPED>) -> Self {
        let unknowns = [
            layout.internal("delayed_transport", Nature::Current),
            layout.internal("delayed_transport_slope", Nature::Current),
        ];
        let [x, y] = unknowns;
        Self {
            unknowns,
            own: [x, y].map(|row| [x, y].map(|column| layout.entry(row, column))),
            driven: std::array::from_fn(|index| layout.entry(y, stamp.node(index))),
            delivered: [INNER_COLLECTOR, INNER_EMITTER]
                .map(|index| layout.entry(stamp.node(index), x)),
        }
    }

    /// Adds the filter's two equations, of `delay`, driven by the forward transport
    /// current `forward` of the transistor `added` adds for, a number of the intrinsic
    /// transistor's law, and the delayed current from the internal collector to the
    /// internal emitter.
    fn add<R: Real>(
        &self,
        added: &Oriented<'_, STAMPED, VARIABLE_COUNT>,
        (forward, delay): (Dual<R, 2>, R),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let intrinsic = added.reading(INTRINSIC);
        let [x, y] = self.unknowns;
        let [delayed, slope] = [x, y].map(|unknown| at.unknown::<R>(unknown));
        let driven = intrinsic.extend(forward) * added.sign;
        stamp.add_residual(x, -slope);
        stamp.add_charge(x, delay * delayed);
        stamp.add_residual(y, (delayed + slope - driven) * 3.0);
        stamp.add_charge(y, delay * slope);
        let [[x_x, x_y], [y_x, y_y]] = self.own;
        stamp.add_conductance(x_y, R::constant(-1.0));
        stamp.add_capacitance(x_x, delay);
        stamp.add_conductance(y_x, R::constant(3.0));
        stamp.add_conductance(y_y, R::constant(3.0));
        stamp.add_capacitance(y_y, delay);
        let gradient = forward.gradient.map(|slope| slope * -3.0);
        intrinsic.add_conductances(stamp, &self.driven, gradient);
        stamp.add_residual(added.nodes.node(INNER_COLLECTOR), delayed);
        stamp.add_residual(added.nodes.node(INNER_EMITTER), -delayed);
        let [at_collector, at_emitter] = self.delivered;
        stamp.add_conductance(at_collector, R::constant(1.0));
        stamp.add_conductance(at_emitter, R::constant(-1.0));
    }
}

/// What the transistor stores: the junctions' depletion charges and transit times.
#[derive(Debug, Clone, Copy)]
struct Charges<R: Real> {
    base_emitter: Option<Depletion<R>>,
    /// The base-collector depletion on the internal base, and on the external one.
    base_collector: Option<Depletion<R>>,
    external_base_collector: Option<Depletion<R>>,
    forward_transit_time: R,
    reverse_transit_time: R,
    bias_coefficient: R,
    /// `1 / (1.44 · VTF)`, zero for an infinite `VTF`.
    voltage_factor: R,
    high_current: R,
}

impl<R: Real> Charges<R> {
    /// The charges at the junctions' `state`, at `voltages`, [`VARIABLES`]' values.
    fn evaluate(&self, voltages: [R; VARIABLE_COUNT], state: &State<R>) -> JunctionCharges<R> {
        let [vbe, vbc, _, vbx] = voltages.map(Dual::<R, 1>::variable);
        let [forward, reverse] = state.ideal;
        // The forward diffusion charge grows with the current and with the base-collector
        // voltage. Within its law, `vbe` comes first and `vbc` second.
        let mut diffusion = forward.embed([0]);
        if self.forward_transit_time.value() != 0.0 && vbe.value.value() > 0.0 {
            let mut factor = Dual::number(0.0);
            if self.bias_coefficient.value() != 0.0 {
                factor = vbc
                    .scale(self.voltage_factor)
                    .exp()
                    .scale(self.bias_coefficient)
                    .embed([1]);
                if self.high_current.value() != 0.0 {
                    let share = (forward / forward.plus(self.high_current)).embed([0]);
                    factor = factor * (share * share);
                }
            }
            diffusion = diffusion * (factor + 1.0) / state.charge;
        }
        let depletion = |depletion: &Option<Depletion<R>>, voltage| {
            depletion_charge(depletion.as_ref(), voltage)
        };
        JunctionCharges {
            qbe: diffusion.scale(self.forward_transit_time)
                + depletion(&self.base_emitter, vbe).embed([0]),
            qbc: reverse.scale(self.reverse_transit_time) + depletion(&self.base_collector, vbc),
            qbx: depletion(&self.external_base_collector, vbx),
        }
    }
}

/// The charges on the base-emitter junction, of [`INTRINSIC`], on the base-collector
/// junction, of `vbc`, and the external base-collector depletion, of `vbx`.
struct JunctionCharges<R: Real> {
    qbe: Dual<R, 2>,
    qbc: Dual<R, 1>,
    qbx: Dual<R, 1>,
}

impl<R: Real> JunctionCharges<R> {
    /// Adds each charge through `added`, by the law it belongs to, its derivatives summed
    /// into `derivatives`.
    fn add(
        self,
        added: &Oriented<'_, STAMPED, VARIABLE_COUNT>,
        derivatives: &mut Derivatives<R, STAMPED>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let Self { qbe, qbc, qbx } = self;
        added
            .reading(INTRINSIC)
            .add_charge(stamp, derivatives, BASE_EMITTER, qbe);
        added
            .reading([VBC])
            .add_charge(stamp, derivatives, BASE_COLLECTOR, qbc);
        added
            .reading([VBX])
            .add_charge(stamp, derivatives, EXTERNAL_BASE_COLLECTOR, qbx);
    }
}

/// The junction from the substrate to the internal collector of a vertical transistor or
/// to the internal base of a lateral one.
#[derive(Debug, Clone, Copy)]
struct SubstrateJunction<R: Real> {
    /// `±1`: orients the junction's voltage and current, from the substrate, as a vertical
    /// NPN's.
    sign: f64,
    /// The saturation current and exponential slope of the junction's diode, if it has one.
    diode: Option<(R, R)>,
    /// The thermal voltage at the transistor's temperature.
    thermal_voltage: f64,
    critical_voltage: f64,
    /// The depletion, whose capacitance grows linearly from zero bias.
    depletion: Option<Depletion<R>>,
}

/// The critical voltage of a substrate junction without a diode, from which ngspice limits
/// its steps all the same.
const DIODELESS_CRITICAL_VOLTAGE: f64 = 50.0;

impl<R: Real> SubstrateJunction<R> {
    /// Adds the junction's current and its charge, between the substrate and the internal
    /// node it joins, `nodes`, at `at`, its last voltage `last`, which it updates.
    fn evaluate(&self, nodes: &Nodes<2>, last: &mut f64, at: &Local<'_>, stamp: &mut Stamp<'_, R>) {
        let applied = self.sign * (nodes.voltage(at, 0) - nodes.voltage(at, 1));
        let limited = limit_junction(applied, *last, self.thermal_voltage, self.critical_voltage)
            .filter(|_| at.limiting);
        if limited.is_some() {
            stamp.mark_limited();
        }
        *last = limited.unwrap_or(applied);
        let beyond = [applied - *last];
        let voltage = Dual::<R, 1>::variable(nodes.across(at, (0, 1), (self.sign, *last)));
        let current = self.diode.map_or(Dual::number(0.0), |(saturation, slope)| {
            junction_current(voltage, saturation, slope)
        }) + voltage * at.gmin;
        let pair = (0, 1);
        let [conductance] = current.gradient;
        nodes.current(stamp, 0, 1, current.extend(beyond) * self.sign);
        nodes.conductance(stamp, pair, pair, conductance);
        if let Some(depletion) = &self.depletion {
            let charge = depletion.evaluate(voltage);
            let [capacitance] = charge.gradient;
            nodes.charge(stamp, 0, 1, charge.extend(beyond) * self.sign);
            nodes.capacitance(stamp, pair, pair, capacitance);
        }
    }
}

impl<R: Real> BipolarTransistorParameters<R> {
    /// The junctions' DC model at `temperatures`, once each of its parameters is checked.
    fn junctions(&self, temperatures: Temperatures<R>) -> Result<Junctions<R>, ParameterError> {
        let positive = ParameterError::positive;
        let non_negative = ParameterError::non_negative;
        positive("saturation_current", self.saturation_current)?;
        positive("forward_beta", self.forward_beta)?;
        positive("reverse_beta", self.reverse_beta)?;
        positive(
            "forward_emission_coefficient",
            self.forward_emission_coefficient,
        )?;
        positive(
            "reverse_emission_coefficient",
            self.reverse_emission_coefficient,
        )?;
        non_negative(
            "base_emitter_leakage_current",
            self.base_emitter_leakage_current,
        )?;
        positive(
            "base_emitter_leakage_emission_coefficient",
            self.base_emitter_leakage_emission_coefficient,
        )?;
        non_negative(
            "base_collector_leakage_current",
            self.base_collector_leakage_current,
        )?;
        positive(
            "base_collector_leakage_emission_coefficient",
            self.base_collector_leakage_emission_coefficient,
        )?;
        non_negative("energy_gap", self.energy_gap)?;
        let exponent = self.high_injection_exponent;
        if !(exponent.value() > 0.0 && exponent.value() <= 1.0) {
            return Err(ParameterError(format!(
                "high_injection_exponent must lie above 0 and at most 1, not {:?}",
                exponent.value()
            )));
        }
        let vt = temperatures.thermal_voltage();
        let scaling = |emission| {
            temperatures.saturation(
                self.energy_gap,
                self.saturation_current_temperature_exponent,
                emission,
            )
        };
        let [forward_beta, reverse_beta, beta_factor] = self.betas(temperatures)?;
        let [inverse_early, inverse_knee] = self.inverses(temperatures)?;
        let emission = |name, nominal: R, linear, quadratic| {
            positive_at(name, nominal * temperatures.polynomial(linear, quadratic))
        };
        let (emitter_emission, collector_emission) = (
            self.base_emitter_leakage_emission_coefficient,
            self.base_collector_leakage_emission_coefficient,
        );
        Ok(Junctions {
            saturation_current: self.saturation_current * scaling(R::constant(1.0)),
            forward_beta,
            reverse_beta,
            forward_slope: emission(
                "forward_emission_coefficient",
                self.forward_emission_coefficient,
                self.forward_emission_coefficient_temperature_coefficient,
                self.forward_emission_coefficient_quadratic_temperature_coefficient,
            )? * vt,
            reverse_slope: emission(
                "reverse_emission_coefficient",
                self.reverse_emission_coefficient,
                self.reverse_emission_coefficient_temperature_coefficient,
                self.reverse_emission_coefficient_quadratic_temperature_coefficient,
            )? * vt,
            // ngspice scales a leakage current by its nominal emission coefficient
            // (`bjttemp.c`, lines 246 to 249), its slope by the one at the temperature.
            emitter_leakage: self.base_emitter_leakage_current * scaling(emitter_emission)
                / beta_factor,
            emitter_leakage_slope: emission(
                "base_emitter_leakage_emission_coefficient",
                emitter_emission,
                self.base_emitter_leakage_emission_coefficient_temperature_coefficient,
                self.base_emitter_leakage_emission_coefficient_quadratic_temperature_coefficient,
            )? * vt,
            collector_leakage: self.base_collector_leakage_current * scaling(collector_emission)
                / beta_factor,
            collector_leakage_slope: emission(
                "base_collector_leakage_emission_coefficient",
                collector_emission,
                self.base_collector_leakage_emission_coefficient_temperature_coefficient,
                self.base_collector_leakage_emission_coefficient_quadratic_temperature_coefficient,
            )? * vt,
            inverse_early,
            inverse_knee,
            high_injection_exponent: exponent,
        })
    }

    /// The forward and reverse betas at `temperatures`, and the factor `TLEV` scales them
    /// by, which divides the leakage currents (`bjttemp.c`, lines 231 to 249).
    fn betas(&self, temperatures: Temperatures<R>) -> Result<[R; 3], ParameterError> {
        let factor = positive_at(
            "the betas' temperature factor",
            match self.temperature_model {
                BipolarTemperatureModel::Zero => {
                    temperatures.ratio().pow(self.beta_temperature_exponent)
                }
                BipolarTemperatureModel::One => {
                    temperatures.difference() * self.beta_temperature_exponent + 1.0
                }
            },
        )?;
        // A beta given a temperature coefficient scales by it alone (`bjttemp.c`, lines
        // 236 to 243).
        let beta = |name, nominal: R, linear: Option<R>, quadratic: Option<R>| {
            let scale = if linear.is_none() && quadratic.is_none() {
                factor
            } else {
                let zero = R::constant(0.0);
                temperatures.polynomial(linear.unwrap_or(zero), quadratic.unwrap_or(zero))
            };
            positive_at(name, nominal * scale)
        };
        Ok([
            beta(
                "forward_beta",
                self.forward_beta,
                self.forward_beta_temperature_coefficient,
                self.forward_beta_quadratic_temperature_coefficient,
            )?,
            beta(
                "reverse_beta",
                self.reverse_beta,
                self.reverse_beta_temperature_coefficient,
                self.reverse_beta_quadratic_temperature_coefficient,
            )?,
            factor,
        ])
    }

    /// The reciprocals of the Early voltages, forward and reverse, and of the knee
    /// currents, at `temperatures`, each zero where it is left out, infinite
    /// (`bjttemp.c`, lines 83 to 104).
    fn inverses(&self, temperatures: Temperatures<R>) -> Result<[[R; 2]; 2], ParameterError> {
        let inverse_at = |name, value: Option<R>, linear, quadratic| {
            value.map_or(Ok(R::constant(0.0)), |value| {
                ParameterError::positive(name, value)?;
                positive_at(name, value * temperatures.polynomial(linear, quadratic))
                    .map(Real::recip)
            })
        };
        Ok([
            [
                inverse_at(
                    "forward_early_voltage",
                    self.forward_early_voltage,
                    self.forward_early_voltage_temperature_coefficient,
                    self.forward_early_voltage_quadratic_temperature_coefficient,
                )?,
                inverse_at(
                    "reverse_early_voltage",
                    self.reverse_early_voltage,
                    self.reverse_early_voltage_temperature_coefficient,
                    self.reverse_early_voltage_quadratic_temperature_coefficient,
                )?,
            ],
            [
                inverse_at(
                    "forward_knee_current",
                    self.forward_knee_current,
                    self.forward_knee_current_temperature_coefficient,
                    self.forward_knee_current_quadratic_temperature_coefficient,
                )?,
                inverse_at(
                    "reverse_knee_current",
                    self.reverse_knee_current,
                    self.reverse_knee_current_temperature_coefficient,
                    self.reverse_knee_current_quadratic_temperature_coefficient,
                )?,
            ],
        ])
    }

    /// The base resistance, if the transistor has one, once its parameters are checked.
    fn base_resistance(&self) -> Result<Option<BaseResistance<R>>, ParameterError> {
        ParameterError::non_negative("base_resistance", self.base_resistance)?;
        let minimum = self.min_base_resistance.unwrap_or(self.base_resistance);
        ParameterError::non_negative("min_base_resistance", minimum)?;
        if let Some(current) = self.base_resistance_current {
            ParameterError::positive("base_resistance_current", current)?;
        }
        Ok(
            (self.base_resistance.value() > 0.0).then_some(BaseResistance {
                maximum: self.base_resistance,
                minimum,
                halfway_current: self.base_resistance_current,
            }),
        )
    }

    /// The base resistance at `temperatures`, if the transistor has one, once its
    /// parameters are checked: `RB`, `RBM` and `IRB` each scaled by its own coefficients,
    /// `RBM` left out by `RB`'s nominal value (`bjttemp.c`, lines 47 to 49 and 118 to
    /// 123).
    fn base_resistance_at(
        &self,
        temperatures: Temperatures<R>,
    ) -> Result<Option<BaseResistance<R>>, ParameterError> {
        let Some(nominal) = self.base_resistance()? else {
            return Ok(None);
        };
        let scale =
            |value: R, linear, quadratic| value * temperatures.polynomial(linear, quadratic);
        Ok(Some(BaseResistance {
            maximum: positive_at(
                "base_resistance",
                scale(
                    nominal.maximum,
                    self.base_resistance_temperature_coefficient,
                    self.base_resistance_quadratic_temperature_coefficient,
                ),
            )?,
            minimum: non_negative_at(
                "min_base_resistance",
                scale(
                    nominal.minimum,
                    self.min_base_resistance_temperature_coefficient,
                    self.min_base_resistance_quadratic_temperature_coefficient,
                ),
            )?,
            halfway_current: nominal
                .halfway_current
                .map(|current| {
                    positive_at(
                        "base_resistance_current",
                        scale(
                            current,
                            self.base_resistance_current_temperature_coefficient,
                            self.base_resistance_current_quadratic_temperature_coefficient,
                        ),
                    )
                })
                .transpose()?,
        }))
    }

    /// What the transistor stores at `temperatures`, if anything, once each of its
    /// parameters is checked.
    fn charges(&self, temperatures: Temperatures<R>) -> Result<Option<Charges<R>>, ParameterError> {
        self.check_charges()?;
        let voltage_factor = inverse("transit_time_voltage", self.transit_time_voltage)? / 1.44;
        let stores = [
            self.base_emitter_capacitance,
            self.base_collector_capacitance,
            self.forward_transit_time,
            self.reverse_transit_time,
        ]
        .iter()
        .any(|value| value.value() > 0.0);
        let [emitter_potential, emitter_capacitance, emitter_grading] = self.junction_at(
            temperatures,
            "base_emitter_potential",
            [
                self.base_emitter_potential,
                self.base_emitter_capacitance,
                self.base_emitter_grading,
            ],
            [
                self.base_emitter_grading_temperature_coefficient,
                self.base_emitter_grading_quadratic_temperature_coefficient,
            ],
            [
                self.base_emitter_capacitance_temperature_coefficient,
                self.base_emitter_potential_temperature_coefficient,
            ],
        )?;
        let [
            collector_potential,
            collector_capacitance,
            collector_grading,
        ] = self.junction_at(
            temperatures,
            "base_collector_potential",
            [
                self.base_collector_potential,
                self.base_collector_capacitance,
                self.base_collector_grading,
            ],
            [
                self.base_collector_grading_temperature_coefficient,
                self.base_collector_grading_quadratic_temperature_coefficient,
            ],
            [
                self.base_collector_capacitance_temperature_coefficient,
                self.base_collector_potential_temperature_coefficient,
            ],
        )?;
        let internal = collector_capacitance * self.internal_base_fraction;
        let collector_depletion = |capacitance| {
            Depletion::new(
                capacitance,
                collector_potential,
                collector_grading,
                self.depletion_coefficient,
            )
        };
        let scale =
            |value: R, linear, quadratic| value * temperatures.polynomial(linear, quadratic);
        Ok(stores.then(|| Charges {
            base_emitter: Depletion::new(
                emitter_capacitance,
                emitter_potential,
                emitter_grading,
                self.depletion_coefficient,
            ),
            base_collector: collector_depletion(internal),
            external_base_collector: collector_depletion(collector_capacitance - internal),
            forward_transit_time: scale(
                self.forward_transit_time,
                self.forward_transit_time_temperature_coefficient,
                self.forward_transit_time_quadratic_temperature_coefficient,
            ),
            reverse_transit_time: scale(
                self.reverse_transit_time,
                self.reverse_transit_time_temperature_coefficient,
                self.reverse_transit_time_quadratic_temperature_coefficient,
            ),
            bias_coefficient: self.transit_time_bias_coefficient,
            voltage_factor,
            high_current: scale(
                self.transit_time_current,
                self.transit_time_current_temperature_coefficient,
                self.transit_time_current_quadratic_temperature_coefficient,
            ),
        }))
    }

    /// Checks the parameters of the transistor's charges at their nominal temperature.
    fn check_charges(&self) -> Result<(), ParameterError> {
        let non_negative = ParameterError::non_negative;
        non_negative("base_emitter_capacitance", self.base_emitter_capacitance)?;
        ParameterError::positive("base_emitter_potential", self.base_emitter_potential)?;
        below_one("base_emitter_grading", self.base_emitter_grading)?;
        non_negative("forward_transit_time", self.forward_transit_time)?;
        non_negative(
            "transit_time_bias_coefficient",
            self.transit_time_bias_coefficient,
        )?;
        non_negative("transit_time_current", self.transit_time_current)?;
        non_negative(
            "base_collector_capacitance",
            self.base_collector_capacitance,
        )?;
        ParameterError::positive("base_collector_potential", self.base_collector_potential)?;
        below_one("base_collector_grading", self.base_collector_grading)?;
        if !(0.0..=1.0).contains(&self.internal_base_fraction.value()) {
            return Err(ParameterError(format!(
                "internal_base_fraction must be at least 0 and at most 1, not {:?}",
                self.internal_base_fraction.value()
            )));
        }
        non_negative("reverse_transit_time", self.reverse_transit_time)?;
        below_one("depletion_coefficient", self.depletion_coefficient)?;
        Ok(())
    }

    /// A junction's built-in `potential`, zero-bias `capacitance` and `grading` at
    /// `temperatures`, in that order (`bjttemp.c`, lines 132 to 146 and 260 to 313): the
    /// grading by its own `grading_coefficients`, linear and quadratic, at most
    /// [`MAX_GRADING`]; the potential and the capacitance through silicon's band gap at
    /// that grading, or where `TLEVC` is 1 by their own `linear` coefficients, the
    /// capacitance's and the potential's.
    ///
    /// # Errors
    ///
    /// Returns an error naming the `potential` parameter where the potential at the
    /// temperature is not positive.
    fn junction_at(
        &self,
        temperatures: Temperatures<R>,
        potential_name: &str,
        [potential, capacitance, grading]: [R; 3],
        [grading_linear, grading_quadratic]: [R; 2],
        [capacitance_linear, potential_linear]: [R; 2],
    ) -> Result<[R; 3], ParameterError> {
        let grading = (grading * temperatures.polynomial(grading_linear, grading_quadratic))
            .min(R::constant(MAX_GRADING));
        let (potential, capacitance) = match self.capacitance_temperature_model {
            BipolarCapacitanceTemperatureModel::Zero => {
                temperatures.junction(potential, capacitance, grading)
            }
            BipolarCapacitanceTemperatureModel::One => {
                let difference = temperatures.difference();
                (
                    potential - potential_linear * difference,
                    capacitance * (capacitance_linear * difference + 1.0),
                )
            }
        };
        Ok([
            positive_at(potential_name, potential)?,
            capacitance,
            grading,
        ])
    }
}

impl<R: Real> BipolarTransistorParameters<R> {
    /// The substrate junction at `temperatures`, if the transistor has a substrate, once
    /// each of its parameters is checked.
    fn substrate_junction(
        &self,
        temperatures: Temperatures<R>,
    ) -> Result<Option<SubstrateJunction<R>>, ParameterError> {
        let Some(construction) = self.substrate else {
            let given = [
                (
                    "substrate_saturation_current",
                    self.substrate_saturation_current,
                ),
                (
                    "substrate_emission_coefficient",
                    self.substrate_emission_coefficient,
                ),
                ("substrate_capacitance", self.substrate_capacitance),
                ("substrate_potential", self.substrate_potential),
                ("substrate_grading", self.substrate_grading),
            ]
            .into_iter()
            .find(|(_, value)| value.is_some());
            return match given {
                Some((name, _)) => Err(ParameterError(format!(
                    "{name} needs a substrate, vertical or lateral"
                ))),
                None => Ok(None),
            };
        };
        let emission = self
            .substrate_emission_coefficient
            .unwrap_or(R::constant(1.0));
        ParameterError::positive("substrate_emission_coefficient", emission)?;
        let capacitance = self.substrate_capacitance.unwrap_or(R::constant(0.0));
        ParameterError::non_negative("substrate_capacitance", capacitance)?;
        let potential = self.substrate_potential.unwrap_or(R::constant(0.75));
        ParameterError::positive("substrate_potential", potential)?;
        let grading = below_one(
            "substrate_grading",
            self.substrate_grading.unwrap_or(R::constant(0.0)),
        )?;
        let vt = temperatures.thermal_voltage();
        // ngspice scales the substrate's saturation current as the transport current's.
        let scaling = temperatures.saturation(
            self.energy_gap,
            self.saturation_current_temperature_exponent,
            R::constant(1.0),
        );
        let emission = positive_at(
            "substrate_emission_coefficient",
            emission
                * temperatures.polynomial(
                    self.substrate_emission_coefficient_temperature_coefficient,
                    self.substrate_emission_coefficient_quadratic_temperature_coefficient,
                ),
        )?;
        let diode = self
            .substrate_saturation_current
            .map(|current| {
                ParameterError::positive("substrate_saturation_current", current)
                    .map(|current| (current * scaling, emission * vt))
            })
            .transpose()?;
        let [potential, capacitance, grading] = self.junction_at(
            temperatures,
            "substrate_potential",
            [potential, capacitance, grading],
            [
                self.substrate_grading_temperature_coefficient,
                self.substrate_grading_quadratic_temperature_coefficient,
            ],
            [
                self.substrate_capacitance_temperature_coefficient,
                self.substrate_potential_temperature_coefficient,
            ],
        )?;
        let orientation = match construction {
            Construction::Vertical => 1.0,
            Construction::Lateral => -1.0,
        };
        Ok(Some(SubstrateJunction {
            sign: self.polarity.sign() * orientation,
            diode,
            thermal_voltage: vt.value(),
            critical_voltage: diode.map_or(DIODELESS_CRITICAL_VOLTAGE, |(current, _)| {
                critical_voltage(current.value(), vt.value())
            }),
            depletion: Depletion::new(capacitance, potential, grading, R::constant(0.0)),
        }))
    }
}

/// Checks that `value`, the parameter `name` at the transistor's temperature, is
/// positive.
fn positive_at<R: Real>(name: &str, value: R) -> Result<R, ParameterError> {
    if value.value() > 0.0 {
        Ok(value)
    } else {
        Err(ParameterError(format!(
            "{name} must be positive at the transistor's temperature, not {:?}",
            value.value()
        )))
    }
}

/// Checks that `value`, the parameter `name` at the transistor's temperature, is not
/// negative.
fn non_negative_at<R: Real>(name: &str, value: R) -> Result<R, ParameterError> {
    if value.value() >= 0.0 {
        Ok(value)
    } else {
        Err(ParameterError(format!(
            "{name} must be at least 0 at the transistor's temperature, not {:?}",
            value.value()
        )))
    }
}

/// A series resistance at the transistor's temperature, `scaled`, in the form its
/// `nominal` value lays it out in: its conductance, or the resistance itself where its
/// current is an unknown, as [`series_value`] gives the nominal one.
fn series_at<R: Real>(nominal: R, scaled: R) -> R {
    if nominal.value() < SMALL_RESISTANCE {
        scaled
    } else {
        scaled.recip()
    }
}

/// The reciprocal of an optional positive parameter, zero when it is left out, infinite.
fn inverse<R: Real>(name: &str, value: Option<R>) -> Result<R, ParameterError> {
    value.map_or(Ok(R::constant(0.0)), |value| {
        ParameterError::positive(name, value).map(Real::recip)
    })
}

impl Kernel for BipolarTransistor {
    const NAME: &'static str = "bipolar_transistor";
    type Parameters<R: Real> = BipolarTransistorParameters<R>;
    type Shape = BipolarTransistorShape;
    type Prepared<R: Real> = PreparedBipolarTransistor<R>;
    type Memory = BipolarTransistorMemory;

    fn interface<R: Real>(parameters: &BipolarTransistorParameters<R>) -> Interface {
        let mut terminals = vec!["collector", "base", "emitter"];
        let mut branches = vec![
            Branch::new(1, 2, BranchKind::Conductive),
            Branch::new(1, 0, BranchKind::Conductive),
            Branch::new(0, 2, BranchKind::Conductive),
        ];
        if let Some(construction) = parameters.substrate {
            let joined = match construction {
                Construction::Vertical => COLLECTOR,
                Construction::Lateral => BASE,
            };
            terminals.push("substrate");
            branches.push(Branch::new(3, joined, BranchKind::Conductive));
        }
        Interface::new(&terminals, &branches)
    }

    fn prepare<R: Real>(
        parameters: &BipolarTransistorParameters<R>,
        temperature: R,
    ) -> Result<PreparedBipolarTransistor<R>, ParameterError> {
        let Scaled {
            series,
            base,
            junctions,
            charges,
            substrate,
            thermal_voltage,
            critical_voltage,
            temperature,
        } = parameters.scaled(temperature)?;
        let flicker = Flicker::new(parameters.flicker_coefficient, parameters.flicker_exponent)?;
        // ngspice delays the excess phase by the nominal transit time (`bjttemp.c`, lines
        // 56 and 57).
        let delay = parameters.excess_phase_delay()?;
        Ok(PreparedBipolarTransistor {
            sign: parameters.polarity.sign(),
            series,
            base,
            junctions,
            charges,
            delay,
            substrate,
            thermal_voltage,
            critical_voltage,
            temperature,
            flicker,
        })
    }

    fn layout<R: Real>(
        parameters: &BipolarTransistorParameters<R>,
        _: &PreparedBipolarTransistor<R>,
        layout: &mut Layout,
    ) -> Result<BipolarTransistorShape, ParameterError> {
        let base_resistance = parameters.base_resistance()?;
        let delay = parameters.excess_phase_delay()?;
        let [collector_resistance, emitter_resistance] = parameters.series_resistances()?;
        let [collector, base_terminal, emitter] = [0, 1, 2];
        let (collector_series, inner_collector) = Series::behind(
            layout,
            collector,
            collector_resistance,
            ("collector", "collector_current"),
        );
        let inner_base = if parameters.base_resistance.value() > 0.0 {
            layout.internal("base", Nature::Voltage)
        } else {
            base_terminal
        };
        let (emitter_series, inner_emitter) = Series::behind(
            layout,
            emitter,
            emitter_resistance,
            ("emitter", "emitter_current"),
        );
        let substrate = parameters.substrate.map(|construction| {
            let joined = match construction {
                Construction::Vertical => inner_collector,
                Construction::Lateral => inner_base,
            };
            Nodes::new(layout, [3, joined])
        });
        let stamp = Nodes::new(
            layout,
            [base_terminal, inner_collector, inner_base, inner_emitter],
        );
        let base = base_resistance.map(|resistance| {
            if resistance.minimum.value() < SMALL_RESISTANCE {
                BaseForm::Current(BaseCurrent::new(layout, &stamp))
            } else {
                BaseForm::Conductance
            }
        });
        let excess = (delay.value() > 0.0).then(|| ExcessPhase::new(layout, &stamp));
        Ok(BipolarTransistorShape {
            stamp,
            collector_series,
            emitter_series,
            base,
            excess,
            substrate,
        })
    }

    fn evaluate<R: Real>(
        shape: &BipolarTransistorShape,
        prepared: &PreparedBipolarTransistor<R>,
        memory: &mut BipolarTransistorMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let [collector_series, emitter_series] = prepared.series;
        shape.collector_series.add(collector_series, at, stamp);
        shape.emitter_series.add(emitter_series, at, stamp);
        let bias = prepared.bias(shape, memory, at, stamp);
        let [vbe, vbc, ..] = bias.voltages;
        memory.variables = Bipolar::variables(vbe, vbc);
        let added = Oriented {
            nodes: &shape.stamp,
            sign: prepared.sign,
            bias: &bias,
            variables: VARIABLES,
        };
        let moving = added.moving(at);
        let [vbe, vbc, ..] = moving;
        let state = prepared.junctions.evaluate([vbe, vbc], at.gmin);
        let intrinsic = added.reading(INTRINSIC);
        // Every derivative among the stamped nodes is summed here, and written once.
        let mut derivatives = Derivatives::new();
        let sums = &mut derivatives;
        let collector_emitter = (INNER_COLLECTOR, INNER_EMITTER);
        match &shape.excess {
            Some(excess) => {
                let reverse = -state.reverse_transport;
                intrinsic.add_current(stamp, sums, collector_emitter, reverse);
                excess.add(&added, (state.forward_transport, prepared.delay), at, stamp);
            }
            None => intrinsic.add_current(stamp, sums, collector_emitter, state.transport),
        }
        let [base_emitter, base_collector] = state.base;
        added
            .reading([VBE])
            .add_current(stamp, sums, BASE_EMITTER, base_emitter);
        added
            .reading([VBC])
            .add_current(stamp, sums, BASE_COLLECTOR, base_collector);
        if let (Some(resistance), Some(form)) = (&prepared.base, &shape.base) {
            form.add(resistance.evaluate(&state), (&added, &mut *sums), at, stamp);
        }
        if let Some(charges) = &prepared.charges {
            charges.evaluate(moving, &state).add(&added, sums, stamp);
        }
        derivatives.stamp(&shape.stamp, stamp);
        if let (Some(junction), Some(nodes)) = (&prepared.substrate, &shape.substrate) {
            junction.evaluate(nodes, &mut memory.substrate, at, stamp);
        }
    }

    fn variable_names() -> &'static [&'static str] {
        &Bipolar::VARIABLE_NAMES
    }

    fn variables(memory: &BipolarTransistorMemory, values: &mut [f64]) {
        parts::write_variables(&memory.variables, values);
    }

    fn regions() -> Regions {
        Bipolar::REGIONS
    }

    /// The thermal noise of the series resistances, and the shot noise of the collector
    /// and base currents, with the base current's flicker noise, as SPICE models them.
    fn noise<R: Real>(
        shape: &BipolarTransistorShape,
        prepared: &PreparedBipolarTransistor<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        let temperature = prepared.temperature;
        let [collector_series, emitter_series] = prepared.series;
        shape.collector_series.noise(
            collector_series,
            ("collector thermal", temperature),
            sources,
        );
        shape
            .emitter_series
            .noise(emitter_series, ("emitter thermal", temperature), sources);
        let nodes = &shape.stamp;
        let voltages = [BASE_EMITTER, BASE_COLLECTOR].map(|(plus, minus)| {
            prepared.sign * (nodes.voltage(at, plus) - nodes.voltage(at, minus))
        });
        let state = prepared
            .junctions
            .evaluate(voltages.map(R::constant), at.gmin);
        let node = |index: usize| nodes.equation(index);
        if let (Some(resistance), Some(form)) = (&prepared.base, &shape.base) {
            let resistance = resistance.evaluate(&state).value.value();
            sources.push(match form {
                BaseForm::Conductance => NoiseSource::thermal(
                    "base thermal",
                    [node(OUTER_BASE), node(INNER_BASE)],
                    resistance.recip(),
                    temperature,
                ),
                BaseForm::Current(current) => NoiseSource::thermal_in_series(
                    "base thermal",
                    UnknownIndex::new(current.branch.unknown()),
                    resistance,
                    temperature,
                ),
            });
        }
        let [base_emitter, base_collector] = state.base.map(|current| current.value);
        let collector = state.transport.value - base_collector;
        let base_current = (base_emitter + base_collector).value();
        sources.push(NoiseSource::shot(
            "collector shot",
            [node(INNER_COLLECTOR), node(INNER_EMITTER)],
            collector.value(),
        ));
        let base_to_emitter = [node(INNER_BASE), node(INNER_EMITTER)];
        sources.push(NoiseSource::shot(
            "base shot",
            base_to_emitter,
            base_current,
        ));
        prepared.flicker.add(base_to_emitter, base_current, sources);
    }
}

/// What a transistor's temperature sets.
struct Scaled<R: Real> {
    /// The collector and emitter resistances, in the form [`series_at`] gives them.
    series: [R; 2],
    base: Option<BaseResistance<R>>,
    junctions: Junctions<R>,
    charges: Option<Charges<R>>,
    substrate: Option<SubstrateJunction<R>>,
    thermal_voltage: f64,
    critical_voltage: f64,
    temperature: f64,
}

impl<R: Real> BipolarTransistorParameters<R> {
    /// What the transistor's `temperature`, in kelvins, sets, once each parameter it
    /// reads is checked.
    fn scaled(&self, temperature: R) -> Result<Scaled<R>, ParameterError> {
        let temperatures = Temperatures::of_junction(temperature, self.nominal_temperature)?;
        let junctions = self.junctions(temperatures)?;
        let thermal_voltage = temperatures.thermal_voltage().value();
        let [collector, emitter] = self.series_resistances()?;
        let series_at_temperature = |name, nominal: R, linear, quadratic| {
            if nominal.value() > 0.0 {
                let scaled = nominal * temperatures.polynomial(linear, quadratic);
                positive_at(name, scaled).map(|scaled| series_at(nominal, scaled))
            } else {
                Ok(series_value(nominal))
            }
        };
        Ok(Scaled {
            // The collector's and the emitter's resistances scale by their coefficients
            // (`bjttemp.c`, lines 105 to 116).
            series: [
                series_at_temperature(
                    "collector_resistance",
                    collector,
                    self.collector_resistance_temperature_coefficient,
                    self.collector_resistance_quadratic_temperature_coefficient,
                )?,
                series_at_temperature(
                    "emitter_resistance",
                    emitter,
                    self.emitter_resistance_temperature_coefficient,
                    self.emitter_resistance_quadratic_temperature_coefficient,
                )?,
            ],
            base: self.base_resistance_at(temperatures)?,
            critical_voltage: critical_voltage(
                junctions.saturation_current.value(),
                thermal_voltage,
            ),
            junctions,
            charges: self.charges(temperatures)?,
            substrate: self.substrate_junction(temperatures)?,
            thermal_voltage,
            temperature: temperatures.actual.value(),
        })
    }
}

impl<R: Real> BipolarTransistorParameters<R> {
    /// The delay of the excess phase, `excess_phase · π / 180 · forward_transit_time`,
    /// once the excess phase is checked.
    fn excess_phase_delay(&self) -> Result<R, ParameterError> {
        let excess_phase = ParameterError::non_negative("excess_phase", self.excess_phase)?;
        Ok(excess_phase * (std::f64::consts::PI / 180.0) * self.forward_transit_time)
    }

    /// The collector and emitter resistances, once each is checked.
    fn series_resistances(&self) -> Result<[R; 2], ParameterError> {
        let non_negative = ParameterError::non_negative;
        Ok([
            non_negative("collector_resistance", self.collector_resistance)?,
            non_negative("emitter_resistance", self.emitter_resistance)?,
        ])
    }
}

impl<R: Real> PreparedBipolarTransistor<R> {
    /// The variables at `at`, the junction voltages limited from the last evaluation's,
    /// which `memory` keeps and takes the new ones.
    fn bias(
        &self,
        shape: &BipolarTransistorShape,
        memory: &mut BipolarTransistorMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) -> Bias {
        let nodes = &shape.stamp;
        let applied = VARIABLES
            .map(|(plus, minus)| self.sign * (nodes.voltage(at, plus) - nodes.voltage(at, minus)));
        let mut voltages = applied;
        let mut limited = false;
        for (voltage, last) in voltages.iter_mut().zip(&mut memory.last) {
            if at.limiting
                && let Some(step) =
                    limit_junction(*voltage, *last, self.thermal_voltage, self.critical_voltage)
            {
                *voltage = step;
                limited = true;
            }
            *last = *voltage;
        }
        if limited {
            stamp.mark_limited();
        }
        Bias::new(applied, voltages)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::Differentiable;
    use crate::Input;
    use crate::kernel::region::{BASE_COLLECTOR, BASE_EMITTER};
    use crate::testing::{
        Bench, assert_close, assert_jacobians_match, assert_mirrors,
        assert_parameter_gradients_match, assert_switches_where_margin_vanishes, source, thermal,
    };
    use crate::{ELEMENTARY_CHARGE, UnknownIndex, thermal_voltage};

    const THERMAL_VOLTAGE: f64 = thermal_voltage(REFERENCE_TEMPERATURE);

    /// A transistor with every effect switched on.
    const FULL: &str = r#"{"saturation_current": 1e-15, "forward_beta": 150,
        "forward_early_voltage": 60, "reverse_early_voltage": 20,
        "forward_knee_current": 0.05, "reverse_knee_current": 0.01,
        "base_emitter_leakage_current": 1e-14, "base_collector_leakage_current": 1e-13,
        "base_emitter_capacitance": 2e-12, "base_collector_capacitance": 1e-12,
        "base_collector_grading": 0.4, "internal_base_fraction": 0.7,
        "forward_transit_time": 4e-10, "transit_time_bias_coefficient": 2,
        "transit_time_voltage": 3, "transit_time_current": 0.02,
        "reverse_transit_time": 1e-8}"#;

    #[test]
    fn its_collector_current_is_beta_times_its_base_current() {
        let mut bench = Bench::new::<BipolarTransistor>(r#"{"saturation_current": 1e-15}"#);
        let evaluation = bench.settle(&[5.0, 0.65, 0.0]);
        let (collector, base) = (evaluation.residual[0], evaluation.residual[1]);
        let expected = 1e-15 * ((0.65 / THERMAL_VOLTAGE).exp() - 1.0);
        // The base-collector junction's reverse current adds a little to both.
        assert!(
            (collector / expected - 1.0).abs() < 1e-6,
            "{collector} vs {expected}"
        );
        assert!(
            (collector / base - 100.0).abs() < 1e-3,
            "{}",
            collector / base
        );
    }

    #[test]
    fn a_pnp_mirrors_an_npn() {
        let model = Differentiable::<BipolarTransistor>::new();
        assert_mirrors(&model, FULL, &[3.0, 0.7, 0.0]);
        assert_mirrors(&model, &excess(), &[3.0, 0.7, 0.0, 1e-3, 0.0]);
    }

    #[test]
    fn series_resistances_add_internal_nodes() {
        let bench = Bench::new::<BipolarTransistor>(
            r#"{"base_resistance": 100, "collector_resistance": 5, "emitter_resistance": 1}"#,
        );
        assert_eq!(
            bench.unknowns(),
            [
                "n0",
                "n1",
                "n2",
                "device:collector",
                "device:base",
                "device:emitter"
            ]
        );
    }

    #[test]
    fn small_series_resistances_carry_their_currents_as_unknowns() {
        let small = |polarity: &str, halfway: &str| {
            FULL.replace(
                '{',
                &format!(
                    r#"{{"polarity": "{polarity}", "base_resistance": 0.8,
                    "min_base_resistance": 0.2, {halfway} "collector_resistance": 0.5,
                    "emitter_resistance": 0.1, "#
                ),
            )
        };
        let bench = Bench::new::<BipolarTransistor>(&small("npn", ""));
        assert_eq!(
            bench.unknowns(),
            [
                "n0",
                "n1",
                "n2",
                "device:collector",
                "device:collector_current",
                "device:base",
                "device:emitter",
                "device:emitter_current",
                "device:base_current",
            ]
        );
        for (polarity, sign) in [("npn", 1.0), ("pnp", -1.0)] {
            for halfway in ["", r#""base_resistance_current": 1e-4,"#] {
                let parameters = small(polarity, halfway);
                for [collector, base] in [[3.0, 0.7], [0.4, 0.72], [-0.5, 0.1], [0.2, -0.6]] {
                    // Each current near what its resistance carries at these voltages, as
                    // near a solution: arbitrary currents can leave a node's residual a
                    // near-cancellation that finite differences cannot resolve.
                    let x = [
                        collector,
                        base,
                        0.0,
                        0.98 * collector,
                        0.02 * collector / 0.5,
                        0.97 * base,
                        0.01,
                        -0.01 / 0.1,
                        0.03 * base / 0.5,
                    ]
                    .map(|value| sign * value);
                    assert_jacobians_match::<BipolarTransistor>(&parameters, &x);
                    assert_parameter_gradients_match::<BipolarTransistor>(&parameters, &x);
                }
            }
        }
    }

    /// [`FULL`] with an excess phase of 30°, a delay of 30 · π / 180 · 0.4 ns.
    fn excess() -> String {
        FULL.replace('{', r#"{"excess_phase": 30, "#)
    }

    #[test]
    fn an_excess_phase_delays_the_forward_transport_current_through_two_unknowns() {
        let mut plain = Bench::new::<BipolarTransistor>(FULL);
        let mut delayed = Bench::new::<BipolarTransistor>(&excess());
        assert_eq!(
            delayed.unknowns(),
            [
                "n0",
                "n1",
                "n2",
                "device:delayed_transport",
                "device:delayed_transport_slope"
            ]
        );
        let at = [3.0, 0.7, 0.0];
        let rest = plain.settle(&at);
        // y's residual is 3 (x + y − u), so at x = y = 0 it gives the forward transport
        // current u, `Ibe / qb`; at rest x is u, and the transistor's currents are those
        // it has without the delay.
        let forward = -delayed.settle(&[at[0], at[1], at[2], 0.0, 0.0]).residual[4] / 3.0;
        let settled = delayed.settle(&[at[0], at[1], at[2], forward, 0.0]);
        assert_close(&settled.residual[..3], &rest.residual);
        assert!(settled.residual[3].abs() < 1e-30);
        assert!(
            settled.residual[4].abs() < 1e-15 * forward.abs(),
            "{} vs {forward}",
            settled.residual[4]
        );
        // The delayed current's equations hold charges of the delay times each unknown.
        let delay = 30.0 * std::f64::consts::PI / 180.0 * 4e-10;
        assert_close(&settled.charge[3..], &[delay * forward, 0.0]);
        assert_close(
            &[settled.capacitance[3][3], settled.capacitance[4][4]],
            &[delay, delay],
        );
    }

    #[test]
    fn its_delay_is_a_second_order_bessel_filter() {
        // Linearized, the delayed current responds to the forward transport current
        // through 3 / (s² td² + 3 s td + 3): solve the two rows at s = jω for a unit drive.
        let mut delayed = Bench::new::<BipolarTransistor>(&excess());
        let evaluation = delayed.settle(&[3.0, 0.7, 0.0, 1e-3, 0.0]);
        let delay = 30.0 * std::f64::consts::PI / 180.0 * 4e-10;
        for omega in [1e8, 1.0 / delay, 1e10] {
            let entry = |row: usize, column: usize| {
                (
                    evaluation.conductance[row][column],
                    omega * evaluation.capacitance[row][column],
                )
            };
            // The rows of x and y over the columns x and y, and y's drive by the forward
            // transport current, whose conductance at the internal base is −3 du/dvb.
            let [a, b, c, d] = [entry(3, 3), entry(3, 4), entry(4, 3), entry(4, 4)];
            let multiply =
                |p: (f64, f64), q: (f64, f64)| (p.0 * q.0 - p.1 * q.1, p.0 * q.1 + p.1 * q.0);
            let determinant = {
                let (ad, bc) = (multiply(a, d), multiply(b, c));
                (ad.0 - bc.0, ad.1 - bc.1)
            };
            // x = −b · (−3 u) / det for a drive u of 1 into y's row, as y's residual holds −3 u.
            let numerator = multiply(b, (3.0, 0.0));
            let numerator = (-numerator.0, -numerator.1);
            let norm = determinant.0 * determinant.0 + determinant.1 * determinant.1;
            let response = (
                (numerator.0 * determinant.0 + numerator.1 * determinant.1) / norm,
                (numerator.1 * determinant.0 - numerator.0 * determinant.1) / norm,
            );
            let wt = omega * delay;
            let denominator = (3.0 - wt * wt, 3.0 * wt);
            let dnorm = denominator.0 * denominator.0 + denominator.1 * denominator.1;
            let expected = (3.0 * denominator.0 / dnorm, -3.0 * denominator.1 / dnorm);
            assert!(
                (response.0 - expected.0).abs() < 1e-12 && (response.1 - expected.1).abs() < 1e-12,
                "{response:?} vs {expected:?} at {omega}"
            );
        }
    }

    #[test]
    fn an_excess_phase_needs_a_transit_time_and_is_not_negative() {
        let without = Bench::new::<BipolarTransistor>(r#"{"excess_phase": 30}"#);
        assert_eq!(without.unknowns(), ["n0", "n1", "n2"]);
        let error = Bench::try_new::<BipolarTransistor>(r#"{"excess_phase": -1}"#).unwrap_err();
        assert_eq!(
            error.to_string(),
            "excess_phase must be at least 0, not -1.0"
        );
    }

    #[test]
    fn a_heated_transistor_is_the_transistor_at_its_temperature() {
        let model = crate::Thermal::<BipolarTransistor>::new();
        let heated = FULL.replace('{', r#"{"thermal": true, "base_resistance": 50, "#);
        let mut bench = Bench::from_model(&model, &heated, REFERENCE_TEMPERATURE).expect("valid");
        let cool = FULL.replace('{', r#"{"base_resistance": 50, "#);
        for rise in [0.0, 35.0] {
            let mut plain = Bench::at::<BipolarTransistor>(&cool, REFERENCE_TEMPERATURE + rise);
            let hot = bench.settle(&[3.0, 0.7, 0.0, rise, 0.69]);
            let reference = plain.settle(&[3.0, 0.7, 0.0, 0.69]);
            let electrical = [0, 1, 2, 4].map(|row| hot.residual[row]);
            assert_close(&electrical, &reference.residual);
        }
        for x in [[3.0, 0.7, 0.0, 20.0, 0.69], [0.3, 0.75, 0.0, 70.0, 0.74]] {
            crate::testing::assert_model_jacobians_match(&model, &heated, &x);
            crate::testing::assert_model_parameter_gradients_match(&model, &heated, &x, None);
        }
        // Its temperature coefficients follow its own temperature, exactly differentiated.
        for coefficients in with_coefficients() {
            let heated = coefficients.replace('{', r#"{"thermal": true, "#);
            for x in [
                [3.0, 0.7, 0.0, -1.0, 20.0, 2.98, 0.69, 0.001],
                [0.3, 0.75, 0.0, 0.2, 70.0, 0.29, 0.74, 0.002],
            ] {
                crate::testing::assert_model_jacobians_match(&model, &heated, &x);
            }
        }
    }

    #[test]
    fn the_base_resistance_shape_is_continuous_where_its_series_hands_over() {
        let shape_at = |z: f64| {
            let Dual {
                value,
                gradient: [slope],
            } = shape(Dual::<f64, 1>::variable(z));
            (value, slope)
        };
        let z = 0.25 - 1e-12;
        let (below, below_slope) = shape_at(z);
        let tan = z.tan();
        let direct = (tan - z) / (z * tan * tan);
        assert!((below / direct - 1.0).abs() < 1e-14, "{below} vs {direct}");
        let (above, above_slope) = shape_at(0.25 + 1e-12);
        assert!((above / below - 1.0).abs() < 1e-11);
        assert!((above_slope / below_slope - 1.0).abs() < 1e-9);
        assert!((shape_at(1e-9).0 - 1.0 / 3.0).abs() < 1e-17);
    }

    #[test]
    fn a_substrate_is_a_fourth_terminal_joined_to_the_collector_or_the_base() {
        let parameters = |construction| {
            serde_json::from_str::<BipolarTransistorParameters>(&format!(
                r#"{{"substrate": "{construction}"}}"#
            ))
            .unwrap()
        };
        let vertical = BipolarTransistor::interface(&parameters("vertical"));
        assert_eq!(
            vertical.terminals,
            ["collector", "base", "emitter", "substrate"]
        );
        assert_eq!(
            vertical.branches[3],
            Branch::new(3, COLLECTOR, BranchKind::Conductive)
        );
        let lateral = BipolarTransistor::interface(&parameters("lateral"));
        assert_eq!(
            lateral.branches[3],
            Branch::new(3, BASE, BranchKind::Conductive)
        );
    }

    #[test]
    fn a_substrate_junction_without_a_diode_conducts_the_minimum_conductance() {
        let mut bench = Bench::new::<BipolarTransistor>(r#"{"substrate": "vertical"}"#);
        let (gmin, x) = (1e-12, [2.0, 0.5, 0.0, -3.0]);
        let evaluation = bench.evaluate_with(&Input::new(&x, None, gmin));
        assert!((evaluation.residual[3] - gmin * (x[3] - x[0])).abs() < 1e-24);
        assert!((evaluation.conductance[3][3] - gmin).abs() < 1e-24);
    }

    #[test]
    fn a_substrate_junction_conducts_from_its_p_side() {
        // A lateral PNP's substrate is its junction's p side, a vertical one's its n side.
        for (construction, joined, forward) in
            [("lateral", BASE, 1.0), ("vertical", COLLECTOR, -1.0)]
        {
            let mut bench = Bench::new::<BipolarTransistor>(&format!(
                r#"{{"polarity": "pnp", "substrate": "{construction}",
                    "substrate_saturation_current": 1e-15}}"#
            ));
            let mut x = [-2.0, -0.5, 0.0, 0.0];
            x[3] = x[joined] + forward * 0.6;
            let current = bench.settle(&x).residual[3];
            let expected = forward * 1e-15 * ((0.6 / THERMAL_VOLTAGE).exp() - 1.0);
            assert!(
                (current / expected - 1.0).abs() < 1e-12,
                "{current} vs {expected}"
            );
        }
    }

    #[test]
    fn substrate_parameters_need_a_substrate() {
        let error =
            Bench::try_new::<BipolarTransistor>(r#"{"substrate_capacitance": 1e-12}"#).unwrap_err();
        assert_eq!(
            error.to_string(),
            "substrate_capacitance needs a substrate, vertical or lateral"
        );
    }

    #[test]
    fn rejects_out_of_range_parameters() {
        let error =
            Bench::try_new::<BipolarTransistor>(r#"{"forward_early_voltage": 0}"#).unwrap_err();
        assert_eq!(
            error.to_string(),
            "forward_early_voltage must be positive, not 0.0"
        );
    }

    #[test]
    fn parameter_gradients_are_exact() {
        let temperature = r#""energy_gap": 1.2, "saturation_current_temperature_exponent": 3.5,
            "beta_temperature_exponent": 1.2, "nominal_temperature": 290, "#;
        let full = FULL.replace('{', &format!("{{{temperature}"));
        let resistive = |halfway: &str| {
            full.replace(
                '{',
                &format!(
                    r#"{{"base_resistance": 200, "min_base_resistance": 20, {halfway}
                    "collector_resistance": 10, "emitter_resistance": 2, "#
                ),
            )
        };
        let delayed = full.replace('{', r#"{"excess_phase": 30, "#);
        for [collector, base] in [[3.0, 0.7], [0.4, 0.72], [-0.5, 0.1], [0.2, -0.6]] {
            assert_parameter_gradients_match::<BipolarTransistor>(&full, &[collector, base, 0.0]);
            assert_parameter_gradients_match::<BipolarTransistor>(
                &delayed,
                &[collector, base, 0.0, 2e-3, -1e-4],
            );
            for halfway in ["", r#""base_resistance_current": 1e-4,"#] {
                let substrate = resistive(halfway).replace(
                    '{',
                    r#"{"substrate": "lateral", "substrate_saturation_current": 1e-16,
                        "substrate_emission_coefficient": 1.2, "substrate_capacitance": 3e-12,
                        "substrate_potential": 0.6, "substrate_grading": 0.4, "#,
                );
                assert_parameter_gradients_match::<BipolarTransistor>(
                    &substrate,
                    &[
                        collector,
                        base,
                        0.0,
                        -1.0,
                        collector - 0.01,
                        base - 0.02,
                        0.001,
                    ],
                );
            }
        }
    }

    /// Every temperature coefficient, each moving its parameter by a few percent over the
    /// 10 K between `parameter_gradients_are_exact`'s nominal temperature and the
    /// reference temperature, with a substrate and the base resistance.
    const COEFFICIENTS: &str = r#""nominal_temperature": 290, "high_injection_exponent": 0.4,
        "forward_beta_temperature_coefficient": 4e-3,
        "forward_beta_quadratic_temperature_coefficient": 2e-5,
        "reverse_beta_quadratic_temperature_coefficient": -1e-5,
        "forward_knee_current_temperature_coefficient": -3e-3,
        "forward_knee_current_quadratic_temperature_coefficient": 1e-5,
        "reverse_knee_current_temperature_coefficient": 2e-3,
        "reverse_knee_current_quadratic_temperature_coefficient": -1e-5,
        "forward_early_voltage_temperature_coefficient": 1e-3,
        "forward_early_voltage_quadratic_temperature_coefficient": 1e-6,
        "reverse_early_voltage_temperature_coefficient": -1e-3,
        "reverse_early_voltage_quadratic_temperature_coefficient": 2e-6,
        "base_resistance": 200, "min_base_resistance": 20, "base_resistance_current": 1e-4,
        "base_resistance_temperature_coefficient": 3e-3,
        "base_resistance_quadratic_temperature_coefficient": 1e-5,
        "min_base_resistance_temperature_coefficient": 2e-3,
        "min_base_resistance_quadratic_temperature_coefficient": -1e-5,
        "base_resistance_current_temperature_coefficient": -2e-3,
        "base_resistance_current_quadratic_temperature_coefficient": 1e-5,
        "collector_resistance": 10, "emitter_resistance": 2,
        "collector_resistance_temperature_coefficient": -1e-2,
        "collector_resistance_quadratic_temperature_coefficient": 3e-5,
        "emitter_resistance_temperature_coefficient": 2e-3,
        "emitter_resistance_quadratic_temperature_coefficient": 1e-6,
        "forward_emission_coefficient_temperature_coefficient": 4e-5,
        "forward_emission_coefficient_quadratic_temperature_coefficient": -3e-7,
        "reverse_emission_coefficient_temperature_coefficient": -1e-3,
        "reverse_emission_coefficient_quadratic_temperature_coefficient": -2e-6,
        "base_emitter_leakage_emission_coefficient_temperature_coefficient": 1e-3,
        "base_emitter_leakage_emission_coefficient_quadratic_temperature_coefficient": 1e-6,
        "base_collector_leakage_emission_coefficient_temperature_coefficient": -1e-3,
        "base_collector_leakage_emission_coefficient_quadratic_temperature_coefficient": 1e-6,
        "transit_time_current_temperature_coefficient": 2e-3,
        "transit_time_current_quadratic_temperature_coefficient": 1e-6,
        "forward_transit_time_temperature_coefficient": 1e-3,
        "forward_transit_time_quadratic_temperature_coefficient": 2e-6,
        "reverse_transit_time_temperature_coefficient": -1e-3,
        "reverse_transit_time_quadratic_temperature_coefficient": 1e-6,
        "base_emitter_grading_temperature_coefficient": 1e-3,
        "base_emitter_grading_quadratic_temperature_coefficient": 1e-6,
        "base_collector_grading_temperature_coefficient": -1e-3,
        "base_collector_grading_quadratic_temperature_coefficient": 1e-6,
        "base_emitter_capacitance_temperature_coefficient": 1e-3,
        "base_collector_capacitance_temperature_coefficient": 3e-3,
        "substrate_capacitance_temperature_coefficient": -2e-3,
        "base_emitter_potential_temperature_coefficient": 2e-3,
        "base_collector_potential_temperature_coefficient": 2.5e-3,
        "substrate_potential_temperature_coefficient": 1e-3,
        "substrate": "lateral", "substrate_saturation_current": 1e-16,
        "substrate_emission_coefficient": 1.2, "substrate_capacitance": 3e-12,
        "substrate_potential": 0.6, "substrate_grading": 0.4,
        "substrate_emission_coefficient_temperature_coefficient": 1e-3,
        "substrate_emission_coefficient_quadratic_temperature_coefficient": 1e-6,
        "substrate_grading_temperature_coefficient": 2e-3,
        "substrate_grading_quadratic_temperature_coefficient": 1e-6, "#;

    /// `COEFFICIENTS` with `FULL`, under each pair of temperature models.
    fn with_coefficients() -> Vec<String> {
        [(0, 0), (1, 0), (0, 1), (1, 1)]
            .map(|(tlev, tlevc)| {
                FULL.replace(
                    '{',
                    &format!(
                        "{{{COEFFICIENTS}\"temperature_model\": {tlev}, \
                         \"capacitance_temperature_model\": {tlevc}, \
                         \"beta_temperature_exponent\": 1.5e-3, "
                    ),
                )
            })
            .to_vec()
    }

    #[test]
    fn temperature_coefficients_scale_their_parameters_exactly() {
        for parameters in with_coefficients() {
            for x in [
                [3.0, 0.7, 0.0, -1.0, 2.98, 0.68, 0.001],
                [0.2, 0.75, 0.0, 0.3, 0.19, 0.73, 0.002],
                [-0.5, 0.1, 0.0, -2.0, -0.49, 0.09, 0.0],
            ] {
                assert_jacobians_match::<BipolarTransistor>(&parameters, &x);
                assert_parameter_gradients_match::<BipolarTransistor>(&parameters, &x);
            }
        }
    }

    #[test]
    fn linear_temperature_models_scale_as_ngspice() {
        let parameters = with_coefficients().swap_remove(3);
        let parsed: BipolarTransistorParameters = serde_json::from_str(&parameters).expect("valid");
        let prepared = BipolarTransistor::prepare(&parsed, REFERENCE_TEMPERATURE).expect("valid");
        let difference = REFERENCE_TEMPERATURE - 290.0;
        let polynomial =
            |linear: f64, quadratic: f64| 1.0 + (linear + quadratic * difference) * difference;
        // `TLEV` 1: the betas' factor `1 + XTB·ΔT` divides the leakage currents, and the
        // forward beta given its coefficients scales by them alone.
        let factor = 1.0 + 1.5e-3 * difference;
        assert_close(
            &[
                prepared.junctions.forward_beta,
                prepared.junctions.reverse_beta,
            ],
            &[150.0 * polynomial(4e-3, 2e-5), polynomial(0.0, -1e-5)],
        );
        let scaling = Temperatures::new(REFERENCE_TEMPERATURE, 290.0)
            .expect("valid")
            .saturation(1.11, 3.0, 1.5);
        assert_close(
            &[prepared.junctions.emitter_leakage],
            &[1e-14 * scaling / factor],
        );
        assert_close(
            &[prepared.series[0], prepared.series[1]],
            &[
                1.0 / (10.0 * polynomial(-1e-2, 3e-5)),
                1.0 / (2.0 * polynomial(2e-3, 1e-6)),
            ],
        );
        // `TLEVC` 1: each capacitance by `1 + CT·ΔT`, each potential less `TVJ·ΔT`.
        let charges = prepared.charges.expect("it stores");
        let capacitance = 2e-12 * (1.0 + 1e-3 * difference);
        let potential = 0.75 - 2e-3 * difference;
        let grading = 0.33 * polynomial(1e-3, 1e-6);
        let at = |voltage| {
            let charge = depletion_charge(
                charges.base_emitter.as_ref(),
                Dual::<f64, 1>::variable(voltage),
            );
            let [capacitance] = charge.gradient;
            capacitance
        };
        assert_close(
            &[at(0.0), at(0.1)],
            &[
                capacitance,
                capacitance * (1.0 - 0.1 / potential).powf(-grading),
            ],
        );
    }

    proptest! {
        /// The conductances and capacitances are the derivatives of the currents and
        /// charges, forward, reverse and saturated, with the base resistance modulated by
        /// the base charge or by the base current. Neither junction is forward-biased
        /// beyond 0.85 V, where its current would be unphysical.
        #[test]
        fn jacobians_match_finite_differences(
            base in -1.0..0.85f64,
            collector_above_base in -0.85..5.0f64,
            pnp in proptest::bool::ANY,
            halfway in proptest::bool::ANY,
        ) {
            let collector = base + collector_above_base;
            let mut parameters = FULL.to_owned();
            if pnp {
                parameters = parameters.replace('{', r#"{"polarity": "pnp", "#);
            }
            let sign = if pnp { -1.0 } else { 1.0 };
            assert_jacobians_match::<BipolarTransistor>(
                &parameters,
                &[sign * collector, sign * base, 0.0],
            );
            assert_jacobians_match::<BipolarTransistor>(
                &parameters.replace('{', r#"{"excess_phase": 45, "#),
                &[sign * collector, sign * base, 0.0, 1e-3, -2e-4],
            );
            let resistive = parameters.replace(
                '{',
                if halfway {
                    r#"{"base_resistance": 200, "min_base_resistance": 20,
                        "base_resistance_current": 1e-4, "collector_resistance": 10,
                        "emitter_resistance": 2, "#
                } else {
                    r#"{"base_resistance": 200, "min_base_resistance": 20,
                        "collector_resistance": 10, "emitter_resistance": 2, "#
                },
            );
            let substrate = resistive.replace(
                '{',
                r#"{"substrate": "lateral", "substrate_saturation_current": 1e-16,
                    "substrate_emission_coefficient": 1.2, "substrate_capacitance": 3e-12,
                    "substrate_potential": 0.6, "substrate_grading": 0.4, "#,
            );
            assert_jacobians_match::<BipolarTransistor>(
                &substrate,
                &[
                    sign * collector,
                    sign * base,
                    0.0,
                    sign * (base - 0.4),
                    sign * 0.98 * collector,
                    sign * 0.97 * base,
                    sign * 0.01,
                ],
            );
            assert_jacobians_match::<BipolarTransistor>(
                &resistive,
                &[
                    sign * collector,
                    sign * base,
                    0.0,
                    sign * 0.98 * collector,
                    sign * 0.97 * base,
                    sign * 0.01,
                ],
            );
        }
    }

    #[test]
    fn its_currents_make_shot_and_flicker_noise() {
        let parameters = r#"{"saturation_current": 1e-15, "flicker_coefficient": 1e-14,
            "flicker_exponent": 1.2}"#;
        let mut bench = Bench::new::<BipolarTransistor>(parameters);
        let x = [5.0, 0.65, 0.0];
        let evaluation = bench.settle(&x);
        let (collector, base) = (evaluation.residual[0], evaluation.residual[1]);
        let sources = bench.noise(&x);
        let shot = |current: f64| 2.0 * ELEMENTARY_CHARGE * current;
        let collector_shot = source(&sources, "collector shot");
        assert_close(&[collector_shot.density(1.0)], &[shot(collector)]);
        let base_shot = source(&sources, "base shot");
        assert_close(&[base_shot.density(1.0)], &[shot(base)]);
        let flicker = source(&sources, "flicker");
        assert_eq!(flicker.equations, base_shot.equations);
        assert_close(&[flicker.density(10.0)], &[1e-14 * base.powf(1.2) / 10.0]);
        assert_eq!(sources.len(), 3);

        // The series resistances' thermal noise, the small base resistance's a voltage in
        // series with its current.
        let resistive = r#"{"saturation_current": 1e-15, "collector_resistance": 20,
            "emitter_resistance": 2, "base_resistance": 0.5}"#;
        let bench = Bench::new::<BipolarTransistor>(resistive);
        let unknowns = bench.unknowns();
        let at = |name: &str| unknowns.iter().position(|unknown| *unknown == name);
        let base_current = at("device:base_current").expect("the base current");
        let mut x = vec![0.0; unknowns.len()];
        x[..3].copy_from_slice(&[5.0, 0.65, 0.0]);
        let sources = bench.noise(&x);
        assert_close(
            &[
                source(&sources, "collector thermal").density(1.0),
                source(&sources, "emitter thermal").density(1.0),
                source(&sources, "base thermal").density(1.0),
            ],
            &[thermal(0.05), thermal(0.5), thermal(0.5)],
        );
        assert_eq!(
            source(&sources, "base thermal").equations,
            [Some(UnknownIndex::new(base_current)), None]
        );
    }

    /// A transistor is cut off with neither junction forward, active with its base-emitter
    /// junction alone forward, saturated with both, and reversed with its base-collector
    /// junction alone; its margins are the junctions' voltages, a PNP's equal to an NPN's
    /// at the voltages reversed, and each region switches where its margin vanishes.
    #[test]
    fn reports_its_region_and_margins() {
        let mut npn = Bench::new::<BipolarTransistor>("{}");
        let mut pnp = Bench::new::<BipolarTransistor>(r#"{"polarity": "pnp"}"#);
        // The collector, the base and the emitter.
        for (x, region) in [
            ([5.0, -0.5, 0.0], "cut_off"),
            ([5.0, 0.7, 0.0], "active"),
            ([0.2, 0.7, 0.0], "saturated"),
            ([0.0, 0.7, 5.0], "reversed"),
        ] {
            let [collector, base, emitter] = x;
            let margins = [base - emitter, base - collector];
            npn.settle(&x);
            assert_eq!(npn.region(), (region, margins.to_vec()), "{x:?}");
            pnp.settle(&x.map(|value| -value));
            assert_eq!(pnp.region(), (region, margins.to_vec()), "{x:?}");
        }
        let switches = [
            ([5.0, -0.1, 0.0], [5.0, 0.1, 0.0], BASE_EMITTER),
            ([5.0, 0.7, 0.0], [0.2, 0.7, 0.0], BASE_COLLECTOR),
            ([0.0, -0.1, 5.0], [0.0, 0.7, 5.0], BASE_COLLECTOR),
        ];
        for (from, to, margin) in switches {
            assert_switches_where_margin_vanishes(&mut npn, (&from, &to), margin);
        }
    }
}
