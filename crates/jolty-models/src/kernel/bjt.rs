//! Gummel-Poon bipolar junction transistor (BJT) kernel, derived from ngspice 47
//! `src/spicelib/devices/bjt/`: Copyright 1985 - 2018, Regents of the University of
//! California and others. Used under the ngspice Modified BSD license, reproduced in the
//! repository `NOTICE` file.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::dimension::{
    CAPACITANCE, CURRENT, PER_KELVIN, PER_KELVIN_SQUARED, RESISTANCE, TEMPERATURE, TIME, VOLTAGE,
    VOLTS_PER_KELVIN,
};
use crate::kernel::dual::Dual;
use crate::kernel::junction::{
    Depletion, Temperatures, below_one, critical_voltage, depletion_charge, junction_current,
};
use crate::kernel::parts::{
    self, Bias, CurrentBranch, Derivatives, Nodes, Oriented, SMALL_RESISTANCE, Series, fraction,
    inverse, series_value,
};
use crate::kernel::region::Bipolar;
use crate::noise::Flicker;
use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Lift, Local, Nature, NoiseSource,
    ParameterError, ParameterKind, ParameterMetadata, REFERENCE_TEMPERATURE, Real, Regions, Stamp,
    UnknownIndex,
};

/// Doping polarity of a bipolar junction transistor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Polarity {
    /// NPN transistor conducting with base above emitter.
    Npn,
    /// PNP transistor with voltages and currents reversed relative to NPN.
    Pnp,
}

impl Polarity {
    /// Returns `1.0` for NPN or `-1.0` for PNP to orient voltages and currents into NPN
    /// convention.
    #[must_use]
    pub const fn sign(self) -> f64 {
        match self {
            Self::Npn => 1.0,
            Self::Pnp => -1.0,
        }
    }
}

/// Physical BJT construction determining which internal node connects to the substrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Construction {
    /// Vertical structure placing the substrate junction on the internal collector.
    Vertical,
    /// Lateral structure placing the substrate junction on the internal base.
    Lateral,
}

/// Temperature scaling selector for BJT saturation currents and current gains (SPICE `TLEV`,
/// matching ngspice `bjttemp.c`, lines 162 to 253).
///
/// Both models scale each saturation current of emission coefficient `n` by
/// `exp(((T/Tn - 1) · Eg / Vt + XTI · ln(T/Tn)) / n)` and divide `ISE` and `ISC` by the
/// beta temperature factor. Ngspice `TLEV = 2` (computing no saturation scaling) and
/// `TLEV = 3` (with exponent `1 + TIS1·ΔT + TIS2·ΔT²` breaking area scaling) are omitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize, JsonSchema)]
#[serde(try_from = "u8", into = "u8")]
pub enum BipolarTemperatureModel {
    /// Model 0 (`TLEV = 0`): scales betas by `(T/Tn)^XTB`.
    #[default]
    Zero,
    /// Model 1 (`TLEV = 1`): scales betas linearly by `1 + XTB · ΔT`.
    One,
}

super::numbered_enum!(
    BipolarTemperatureModel,
    "a bipolar transistor's temperature_model is 0 or 1",
    {
        0 => Zero,
        1 => One,
    }
);

/// Temperature scaling selector for BJT depletion capacitances and built-in potentials
/// (SPICE `TLEVC`, matching ngspice `bjttemp.c`, lines 260 to 313).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize, JsonSchema)]
#[serde(try_from = "u8", into = "u8")]
pub enum BipolarCapacitanceTemperatureModel {
    /// Model 0 (`TLEVC = 0`): scales potentials and capacitances via silicon band gap and
    /// intrinsic carrier concentration.
    #[default]
    Zero,
    /// Model 1 (`TLEVC = 1`): scales each capacitance linearly by `1 + CT · ΔT` and lowers
    /// each potential linearly by `TVJ · ΔT`.
    One,
}

super::numbered_enum!(
    BipolarCapacitanceTemperatureModel,
    "a bipolar transistor's capacitance_temperature_model is 0 or 1",
    {
        0 => Zero,
        1 => One,
    }
);

/// Upper clamp applied to temperature-scaled grading coefficients (`bjttemp.c`, lines 133 to
/// 146) because depletion charge divides by `1 - m`.
const MAX_GRADING: f64 = 0.999;

/// Bipolar junction transistor implementing the SPICE Gummel-Poon model from ngspice 47.
///
/// Base-emitter and base-collector junctions carry ideal and leakage currents, while
/// transport current `(Ibe - Ibc) / qb` incorporates Early effect and high injection via
/// base charge `qb`. Base resistance transitions from `rb` toward `rbm` with base charge or
/// base current. Each junction stores depletion charge and transit-time diffusion charge,
/// with forward transit time modulated by current and `vbc`, and a fraction of
/// base-collector depletion placed on the external base. Positive series resistances add
/// internal nodes (`base`, `collector`, `emitter`) and, below [`SMALL_RESISTANCE`] (at
/// `rbm` for the base), branch currents (`base_current`, `collector_current`,
/// `emitter_current`).
///
/// An optional fourth terminal `substrate` connects via a pn junction to the internal
/// collector (vertical) or internal base (lateral), conducting `gmin` plus optional diode
/// current and forward-linearized depletion charge.
///
/// Excess phase delays forward transport current `Ibe / qb` through SPICE's second-order
/// Weil-Bessel filter `3 / (s² · td² + 3 · s · td + 3)`. The filter maintains low-frequency
/// group delay `td`, matches the target phase at `1 / (2π · TF)`, and uses internal unknowns
/// `delayed_transport` and `delayed_transport_slope`.
///
/// Away from nominal temperature, saturation currents scale with the energy gap, betas scale
/// according to [temperature model](BipolarTemperatureModel), and potentials and
/// capacitances scale according to
/// [capacitance temperature model](BipolarCapacitanceTemperatureModel). Additional HSPICE
/// parameters (resistances, Early voltages, knee currents, emission and grading
/// coefficients, transit times, and betas with explicit coefficients) scale by
/// `1 + c1·ΔT + c2·ΔT²`, matching ngspice 47 `bjttemp.c`.
#[derive(Debug, Clone, Copy)]
pub struct BipolarTransistor;

/// Layout shape for [`BipolarTransistor`] storing series resistances, intrinsic node matrix,
/// optional excess-phase filter, and optional substrate junction nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BipolarTransistorShape {
    stamp: Nodes<STAMPED>,
    collector_series: Series,
    emitter_series: Series,
    base: Option<BaseForm>,
    excess: Option<ExcessPhase>,
    /// Substrate terminal and target internal node (collector or base).
    substrate: Option<Nodes<2>>,
}

/// Prepared BJT parameters at operating temperature.
#[derive(Debug, Clone, Copy)]
pub struct PreparedBipolarTransistor<R: Real> {
    sign: f64,
    /// Stamp values for collector and emitter series resistances.
    series: [R; 2],
    base: Option<BaseResistance<R>>,
    junctions: Junctions<R>,
    charges: Option<Charges<R>>,
    /// Excess-phase group delay, in seconds.
    delay: R,
    substrate: Option<SubstrateJunction<R>>,
    /// Thermal voltage at operating temperature.
    thermal_voltage: f64,
    critical_voltage: f64,
    temperature: R,
    flicker: Flicker<R>,
}

/// Persistent evaluation memory for [`BipolarTransistor`] storing previous junction voltages
/// and diagnostic variables.
#[derive(Debug, Clone, Copy, Default)]
pub struct BipolarTransistorMemory {
    /// Base-emitter and base-collector voltages from the last evaluation for step limiting.
    last: [f64; JUNCTIONS],
    /// Substrate junction voltage from the last evaluation for step limiting.
    substrate: f64,
    /// Values of [`Bipolar::VARIABLE_NAMES`] from the last evaluation.
    variables: [f64; 3],
}

/// Terminal indices eligible for substrate junction attachment (`collector` and `base`).
const COLLECTOR: usize = 0;
const BASE: usize = 1;

/// Local node indices: external base plus internal collector, base, and emitter nodes.
const OUTER_BASE: usize = 0;
const INNER_COLLECTOR: usize = 1;
const INNER_BASE: usize = 2;
const INNER_EMITTER: usize = 3;

/// Number of stamped nodes.
const STAMPED: usize = 4;

/// Number of controlling branch voltages in the BJT equations.
const VARIABLE_COUNT: usize = 4;

/// Number of leading junction voltages subject to step limiting.
const JUNCTIONS: usize = 2;

/// Stamped node pairs for the four controlling voltages: base-emitter, base-collector,
/// base-resistance drop, and external base-collector.
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

/// Variable positions in [`VARIABLES`], using ngspice names plus `VRB` for base-resistance
/// drop.
const VBE: usize = 0;
const VBC: usize = 1;
const VRB: usize = 2;
const VBX: usize = 3;

/// Subset of [`VARIABLES`] read by transport currents, base resistance, and base-emitter
/// charge: `[VBE, VBC]`.
const INTRINSIC: [usize; 2] = [VBE, VBC];
/// Subset of [`VARIABLES`] read by conductance-form base resistance: base-resistance drop
/// followed by `[VBE, VBC]`.
const BASE_RESISTANCE: [usize; 3] = [VRB, VBE, VBC];

/// Parameters for [`BipolarTransistor`], with SPICE defaults.
///
/// Parameters treated by SPICE as infinite when zero (such as Early voltages and knee
/// currents) default to infinite when omitted.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift, ParameterMetadata)]
#[serde(default, deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct BipolarTransistorParameters<R: Real = f64> {
    /// Doping polarity; default `npn`.
    pub polarity: Polarity,
    /// Transport saturation current `IS`, in amperes; default 1e-16.
    #[dimension(CURRENT)]
    pub saturation_current: R,
    /// Ideal maximum forward current gain `BF`; default 100.
    pub forward_beta: R,
    /// Ideal maximum reverse current gain `BR`; default 1.
    pub reverse_beta: R,
    /// Forward emission coefficient `NF`; default 1.
    pub forward_emission_coefficient: R,
    /// Reverse emission coefficient `NR`; default 1.
    pub reverse_emission_coefficient: R,
    /// Forward Early voltage `VAF`, in volts; infinite by default.
    #[dimension(VOLTAGE)]
    pub forward_early_voltage: Option<R>,
    /// Reverse Early voltage `VAR`, in volts; infinite by default.
    #[dimension(VOLTAGE)]
    pub reverse_early_voltage: Option<R>,
    /// Forward high-injection knee current `IKF`, in amperes; infinite by default.
    #[dimension(CURRENT)]
    pub forward_knee_current: Option<R>,
    /// Reverse high-injection knee current `IKR`, in amperes; infinite by default.
    #[dimension(CURRENT)]
    pub reverse_knee_current: Option<R>,
    /// Base-emitter leakage saturation current `ISE`, in amperes; default 0.
    #[dimension(CURRENT)]
    pub base_emitter_leakage_current: R,
    /// Base-emitter leakage emission coefficient `NE`; default 1.5.
    pub base_emitter_leakage_emission_coefficient: R,
    /// Base-collector leakage saturation current `ISC`, in amperes; default 0.
    #[dimension(CURRENT)]
    pub base_collector_leakage_current: R,
    /// Base-collector leakage emission coefficient `NC`; default 2.
    pub base_collector_leakage_emission_coefficient: R,
    /// Zero-bias base resistance `RB`, in ohms; default 0.
    #[dimension(RESISTANCE)]
    pub base_resistance: R,
    /// Minimum high-current base resistance `RBM`, in ohms; defaults to `base_resistance`.
    #[dimension(RESISTANCE)]
    pub min_base_resistance: Option<R>,
    /// Base current `IRB` at which base resistance falls halfway toward `RBM`, in amperes;
    /// omitted modulates resistance by base charge.
    #[dimension(CURRENT)]
    pub base_resistance_current: Option<R>,
    /// Emitter series resistance `RE`, in ohms; default 0.
    #[dimension(RESISTANCE)]
    pub emitter_resistance: R,
    /// Collector series resistance `RC`, in ohms; default 0.
    #[dimension(RESISTANCE)]
    pub collector_resistance: R,
    /// Zero-bias base-emitter depletion capacitance `CJE`, in farads; default 0.
    #[dimension(CAPACITANCE)]
    pub base_emitter_capacitance: R,
    /// Base-emitter built-in potential `VJE`, in volts; default 0.75.
    #[dimension(VOLTAGE)]
    pub base_emitter_potential: R,
    /// Base-emitter grading coefficient `MJE`, below 1; default 0.33.
    pub base_emitter_grading: R,
    /// Ideal forward transit time `TF`, in seconds; default 0.
    #[dimension(TIME)]
    pub forward_transit_time: R,
    /// Forward transit-time bias modulation coefficient `XTF`; default 0.
    pub transit_time_bias_coefficient: R,
    /// Base-collector voltage `VTF` modulating forward transit time, in volts; infinite by
    /// default.
    #[dimension(VOLTAGE)]
    pub transit_time_voltage: Option<R>,
    /// High-current threshold `ITF` for forward transit time growth, in amperes; default 0.
    #[dimension(CURRENT)]
    pub transit_time_current: R,
    /// Zero-bias base-collector depletion capacitance `CJC`, in farads; default 0.
    #[dimension(CAPACITANCE)]
    pub base_collector_capacitance: R,
    /// Base-collector built-in potential `VJC`, in volts; default 0.75.
    #[dimension(VOLTAGE)]
    pub base_collector_potential: R,
    /// Base-collector grading coefficient `MJC`, below 1; default 0.33.
    pub base_collector_grading: R,
    /// Fraction `XCJC` of base-collector capacitance placed on the internal base; default 1.
    pub internal_base_fraction: R,
    /// Ideal reverse transit time `TR`, in seconds; default 0.
    #[dimension(TIME)]
    pub reverse_transit_time: R,
    /// Fraction `FC` of junction potential above which depletion capacitance extends
    /// linearly; below 1; default 0.5.
    pub depletion_coefficient: R,
    /// Excess phase `PTF` at `1 / (2π · TF)`, in degrees, giving delay
    /// `PTF · π / 180 · TF`; default 0.
    pub excess_phase: R,
    /// Substrate construction (`vertical` or `lateral`, matching ngspice `SUBS = 1` and
    /// `-1`); omitted disables the substrate terminal and requires substrate junction
    /// parameters to remain unset.
    pub substrate: Option<Construction>,
    /// Substrate junction saturation current `ISS`, in amperes; omitted conducts `gmin`
    /// only.
    #[dimension(CURRENT)]
    pub substrate_saturation_current: Option<R>,
    /// Substrate junction emission coefficient `NS`; default 1.
    pub substrate_emission_coefficient: Option<R>,
    /// Zero-bias substrate depletion capacitance `CJS`, in farads; default 0.
    #[dimension(CAPACITANCE)]
    pub substrate_capacitance: Option<R>,
    /// Substrate built-in potential `VJS`, in volts; default 0.75.
    #[dimension(VOLTAGE)]
    pub substrate_potential: Option<R>,
    /// Substrate grading coefficient `MJS`, below 1; default 0.
    pub substrate_grading: Option<R>,
    /// Energy gap `EG` scaling saturation currents with temperature, in volts; default 1.11
    /// (silicon).
    #[dimension(VOLTAGE)]
    pub energy_gap: R,
    /// Temperature-ratio exponent `XTI` scaling saturation currents; default 3.
    pub saturation_current_temperature_exponent: R,
    /// Beta temperature parameter `XTB`: exponent of `T/Tn` for `TLEV = 0`, or linear
    /// coefficient per kelvin for `TLEV = 1`; default 0.
    pub beta_temperature_exponent: R,
    /// Saturation-current and beta temperature model `TLEV`; default 0.
    pub temperature_model: BipolarTemperatureModel,
    /// Junction-capacitance and potential temperature model `TLEVC`; default 0.
    pub capacitance_temperature_model: BipolarCapacitanceTemperatureModel,
    /// High-injection base-charge exponent `NKF` in `(0, 1]`, giving
    /// `qb = q1 · (1 + (1 + 4 · q2)^NKF) / 2`; default 0.5.
    pub high_injection_exponent: R,
    /// Linear forward-beta temperature coefficient `TBF1`, per kelvin. When `TBF1` or `TBF2`
    /// is provided, scales `BF` by `1 + TBF1·ΔT + TBF2·ΔT²` instead of `XTB`.
    #[dimension(PER_KELVIN)]
    pub forward_beta_temperature_coefficient: Option<R>,
    /// Quadratic forward-beta temperature coefficient `TBF2`, per kelvin squared (see `TBF1`).
    #[dimension(PER_KELVIN_SQUARED)]
    pub forward_beta_quadratic_temperature_coefficient: Option<R>,
    /// Linear reverse-beta temperature coefficient `TBR1`, per kelvin. When `TBR1` or `TBR2`
    /// is provided, scales `BR` by `1 + TBR1·ΔT + TBR2·ΔT²` instead of `XTB`.
    #[dimension(PER_KELVIN)]
    pub reverse_beta_temperature_coefficient: Option<R>,
    /// Quadratic reverse-beta temperature coefficient `TBR2`, per kelvin squared (see `TBR1`).
    #[dimension(PER_KELVIN_SQUARED)]
    pub reverse_beta_quadratic_temperature_coefficient: Option<R>,
    /// Linear forward knee-current temperature coefficient `TIKF1`, per kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub forward_knee_current_temperature_coefficient: R,
    /// Quadratic forward knee-current temperature coefficient `TIKF2`, per kelvin squared;
    /// default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub forward_knee_current_quadratic_temperature_coefficient: R,
    /// Linear reverse knee-current temperature coefficient `TIKR1`, per kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub reverse_knee_current_temperature_coefficient: R,
    /// Quadratic reverse knee-current temperature coefficient `TIKR2`, per kelvin squared;
    /// default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub reverse_knee_current_quadratic_temperature_coefficient: R,
    /// Linear forward Early-voltage temperature coefficient `TVAF1`, per kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub forward_early_voltage_temperature_coefficient: R,
    /// Quadratic forward Early-voltage temperature coefficient `TVAF2`, per kelvin squared;
    /// default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub forward_early_voltage_quadratic_temperature_coefficient: R,
    /// Linear reverse Early-voltage temperature coefficient `TVAR1`, per kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub reverse_early_voltage_temperature_coefficient: R,
    /// Quadratic reverse Early-voltage temperature coefficient `TVAR2`, per kelvin squared;
    /// default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub reverse_early_voltage_quadratic_temperature_coefficient: R,
    /// Linear zero-bias base-resistance temperature coefficient `TRB1`, per kelvin;
    /// default 0.
    #[dimension(PER_KELVIN)]
    pub base_resistance_temperature_coefficient: R,
    /// Quadratic zero-bias base-resistance temperature coefficient `TRB2`, per kelvin
    /// squared; default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub base_resistance_quadratic_temperature_coefficient: R,
    /// Linear minimum base-resistance temperature coefficient `TRM1`, per kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub min_base_resistance_temperature_coefficient: R,
    /// Quadratic minimum base-resistance temperature coefficient `TRM2`, per kelvin squared;
    /// default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub min_base_resistance_quadratic_temperature_coefficient: R,
    /// Linear temperature coefficient `TIRB1` for halfway base-resistance current `IRB`, per
    /// kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub base_resistance_current_temperature_coefficient: R,
    /// Quadratic temperature coefficient `TIRB2` for halfway base-resistance current `IRB`,
    /// per kelvin squared; default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub base_resistance_current_quadratic_temperature_coefficient: R,
    /// Linear collector-resistance temperature coefficient `TRC1`, per kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub collector_resistance_temperature_coefficient: R,
    /// Quadratic collector-resistance temperature coefficient `TRC2`, per kelvin squared;
    /// default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub collector_resistance_quadratic_temperature_coefficient: R,
    /// Linear emitter-resistance temperature coefficient `TRE1`, per kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub emitter_resistance_temperature_coefficient: R,
    /// Quadratic emitter-resistance temperature coefficient `TRE2`, per kelvin squared;
    /// default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub emitter_resistance_quadratic_temperature_coefficient: R,
    /// Linear forward emission-coefficient temperature coefficient `TNF1`, per kelvin;
    /// default 0.
    #[dimension(PER_KELVIN)]
    pub forward_emission_coefficient_temperature_coefficient: R,
    /// Quadratic forward emission-coefficient temperature coefficient `TNF2`, per kelvin
    /// squared; default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub forward_emission_coefficient_quadratic_temperature_coefficient: R,
    /// Linear reverse emission-coefficient temperature coefficient `TNR1`, per kelvin;
    /// default 0.
    #[dimension(PER_KELVIN)]
    pub reverse_emission_coefficient_temperature_coefficient: R,
    /// Quadratic reverse emission-coefficient temperature coefficient `TNR2`, per kelvin
    /// squared; default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub reverse_emission_coefficient_quadratic_temperature_coefficient: R,
    /// Linear base-emitter leakage emission-coefficient temperature coefficient `TNE1`, per
    /// kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub base_emitter_leakage_emission_coefficient_temperature_coefficient: R,
    /// Quadratic base-emitter leakage emission-coefficient temperature coefficient `TNE2`,
    /// per kelvin squared; default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub base_emitter_leakage_emission_coefficient_quadratic_temperature_coefficient: R,
    /// Linear base-collector leakage emission-coefficient temperature coefficient `TNC1`,
    /// per kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub base_collector_leakage_emission_coefficient_temperature_coefficient: R,
    /// Quadratic base-collector leakage emission-coefficient temperature coefficient `TNC2`,
    /// per kelvin squared; default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub base_collector_leakage_emission_coefficient_quadratic_temperature_coefficient: R,
    /// Linear substrate emission-coefficient temperature coefficient `TNS1`, per kelvin;
    /// default 0.
    #[dimension(PER_KELVIN)]
    pub substrate_emission_coefficient_temperature_coefficient: R,
    /// Quadratic substrate emission-coefficient temperature coefficient `TNS2`, per kelvin
    /// squared; default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub substrate_emission_coefficient_quadratic_temperature_coefficient: R,
    /// Linear temperature coefficient `TITF1` for transit-time current `ITF`, per kelvin;
    /// default 0.
    #[dimension(PER_KELVIN)]
    pub transit_time_current_temperature_coefficient: R,
    /// Quadratic temperature coefficient `TITF2` for transit-time current `ITF`, per kelvin
    /// squared; default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub transit_time_current_quadratic_temperature_coefficient: R,
    /// Linear forward transit-time temperature coefficient `TTF1`, per kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub forward_transit_time_temperature_coefficient: R,
    /// Quadratic forward transit-time temperature coefficient `TTF2`, per kelvin squared;
    /// default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub forward_transit_time_quadratic_temperature_coefficient: R,
    /// Linear reverse transit-time temperature coefficient `TTR1`, per kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub reverse_transit_time_temperature_coefficient: R,
    /// Quadratic reverse transit-time temperature coefficient `TTR2`, per kelvin squared;
    /// default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub reverse_transit_time_quadratic_temperature_coefficient: R,
    /// Linear base-emitter grading temperature coefficient `TMJE1`, per kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub base_emitter_grading_temperature_coefficient: R,
    /// Quadratic base-emitter grading temperature coefficient `TMJE2`, per kelvin squared;
    /// default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub base_emitter_grading_quadratic_temperature_coefficient: R,
    /// Linear base-collector grading temperature coefficient `TMJC1`, per kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub base_collector_grading_temperature_coefficient: R,
    /// Quadratic base-collector grading temperature coefficient `TMJC2`, per kelvin squared;
    /// default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub base_collector_grading_quadratic_temperature_coefficient: R,
    /// Linear substrate grading temperature coefficient `TMJS1`, per kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub substrate_grading_temperature_coefficient: R,
    /// Quadratic substrate grading temperature coefficient `TMJS2`, per kelvin squared;
    /// default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub substrate_grading_quadratic_temperature_coefficient: R,
    /// Linear base-emitter capacitance temperature coefficient `CTE` for `TLEVC = 1`, per
    /// kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub base_emitter_capacitance_temperature_coefficient: R,
    /// Linear base-collector capacitance temperature coefficient `CTC` for `TLEVC = 1`, per
    /// kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub base_collector_capacitance_temperature_coefficient: R,
    /// Linear substrate capacitance temperature coefficient `CTS` for `TLEVC = 1`, per
    /// kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub substrate_capacitance_temperature_coefficient: R,
    /// Linear base-emitter built-in potential temperature coefficient `TVJE` for
    /// `TLEVC = 1`, in volts per kelvin; default 0.
    #[dimension(VOLTS_PER_KELVIN)]
    pub base_emitter_potential_temperature_coefficient: R,
    /// Linear base-collector built-in potential temperature coefficient `TVJC` for
    /// `TLEVC = 1`, in volts per kelvin; default 0.
    #[dimension(VOLTS_PER_KELVIN)]
    pub base_collector_potential_temperature_coefficient: R,
    /// Linear substrate built-in potential temperature coefficient `TVJS` for `TLEVC = 1`,
    /// in volts per kelvin; default 0.
    #[dimension(VOLTS_PER_KELVIN)]
    pub substrate_potential_temperature_coefficient: R,
    /// Nominal measurement temperature, in kelvins; default 27 °C.
    #[dimension(TEMPERATURE)]
    pub nominal_temperature: R,
    /// Base-current flicker noise coefficient `KF`; default 0 (no flicker noise).
    pub flicker_coefficient: R,
    /// Base-current flicker noise exponent `AF`; default 1.
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

/// DC junction current parameters and shared base-charge coefficients.
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
    /// Reciprocal Early voltages `[1 / VAF, 1 / VAR]` (zero when infinite).
    inverse_early: [R; 2],
    /// Reciprocal knee currents `[1 / IKF, 1 / IKR]` (zero when infinite).
    inverse_knee: [R; 2],
    /// High-injection base-charge exponent `NKF`.
    high_injection_exponent: R,
}

/// Intrinsic DC junction state at one bias point, carrying derivatives over either a single
/// junction voltage or [`INTRINSIC`].
#[derive(Debug, Clone, Copy)]
struct State<R: Real> {
    /// Ideal base-emitter and base-collector currents `[Ibe, Ibc]`, each differentiated over
    /// its corresponding junction voltage.
    ideal: [Dual<R, 1>; 2],
    /// Base current components toward emitter and collector, each differentiated over its
    /// corresponding junction voltage.
    base: [Dual<R, 1>; 2],
    /// Collector-to-emitter transport current `(Ibe - Ibc) / qb`.
    transport: Dual<R, 2>,
    /// Forward transport component `Ibe / qb` subject to excess-phase delay.
    forward_transport: Dual<R, 2>,
    /// Reverse transport component `Ibc / qb` kept undelayed, computed directly to avoid
    /// cancellation against `transport`.
    reverse_transport: Dual<R, 2>,
    /// Normalized base charge `qb`.
    charge: Dual<R, 2>,
}

impl<R: Real> Junctions<R> {
    /// Evaluates intrinsic DC [`State`] at `[vbe, vbc]` with minimum junction conductance
    /// `gmin`.
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

/// Stamping form for base series resistance between external and internal base nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BaseForm {
    /// Conductance stamped directly between external and internal base.
    Conductance,
    /// Explicit base-current unknown used when `RBM` lies below [`SMALL_RESISTANCE`],
    /// constrained by `drop - r · i = 0`.
    Current(BaseCurrent),
}

impl BaseForm {
    /// Stamps bias-dependent base `resistance` into `stamp` and `derivatives` for `added`,
    /// using either conductance `drop / r` or the explicit base-current equation.
    fn add<R: Real>(
        &self,
        resistance: Dual<R, 2>,
        drop: R,
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

/// Explicit base-current branch and matrix entries coupling its constitutive equation to
/// itself and all stamped nodes for junction-modulated base resistance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BaseCurrent {
    branch: CurrentBranch,
    own: usize,
    driven: [usize; STAMPED],
}

impl BaseCurrent {
    /// Allocates the `base_current` branch between `OUTER_BASE` and `INNER_BASE` in `stamp`
    /// and claims all coupling entries in `layout`.
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

    /// Stamps the explicit base-current equation `drop - resistance · i = 0` and junction
    /// voltage derivatives for `added`.
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

/// Bias-modulated base resistance transitioning from zero-bias `maximum` toward `minimum`.
#[derive(Debug, Clone, Copy)]
struct BaseResistance<R: Real> {
    maximum: R,
    minimum: R,
    /// Base current `IRB` for halfway reduction, or `None` for base-charge modulation.
    halfway_current: Option<R>,
}

impl<R: Real> BaseResistance<R> {
    /// Evaluates bias-dependent base resistance and intrinsic junction derivatives at
    /// `state`.
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

/// Computes SPICE parameter `z = (√(1 + 144 x / π²) - 1) / ((24 / π²) · √x)` for normalized
/// base current `ratio`.
fn z_of<R: Real>(ratio: Dual<R, 1>) -> Dual<R, 1> {
    const A: f64 = 14.590_25;
    const B: f64 = 2.4317;
    let root = (ratio * A + 1.0).sqrt();
    (root - 1.0) / (ratio.sqrt() * B)
}

/// Evaluates `(tan z - z) / (z · tan² z)` for current-modulated base resistance.
///
/// Below `z = 0.25`, switches to a power series in `z²` to avoid `3ε / z²` cancellation,
/// maintaining accuracy within a few ULPs.
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

/// Second-order Bessel excess-phase filter delaying forward transport current `u`.
///
/// Unknowns `x` (delayed current) and `y = td · dx/dt` satisfy `td · dx/dt - y = 0` and
/// `td · dy/dt + 3 · y + 3 · x - 3 · u = 0`. At DC, the filter reduces to `x = u`. Both
/// unknowns carry collector-to-emitter sign orientation matching the transistor polarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExcessPhase {
    /// Local unknown indices `[x, y]`.
    unknowns: [usize; 2],
    /// Internal matrix entries for the `x` and `y` filter equations over `[x, y]`.
    own: [[usize; 2]; 2],
    /// Matrix entries in the `y` equation for each stamped node.
    driven: [usize; STAMPED],
    /// Matrix entries injecting `x` into internal collector and emitter node equations.
    delivered: [usize; 2],
}

impl ExcessPhase {
    /// Allocates `delayed_transport` and `delayed_transport_slope` unknowns and claims
    /// filter matrix entries for `stamp`.
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

/// Temperature-prepared BJT junction depletion and diffusion charge parameters.
#[derive(Debug, Clone, Copy)]
struct Charges<R: Real> {
    base_emitter: Option<Depletion<R>>,
    /// Internal and external base-collector depletion models.
    base_collector: Option<Depletion<R>>,
    external_base_collector: Option<Depletion<R>>,
    forward_transit_time: R,
    reverse_transit_time: R,
    bias_coefficient: R,
    /// Transit-time voltage factor `1 / (1.44 · VTF)` (zero when `VTF` is infinite).
    voltage_factor: R,
    high_current: R,
}

impl<R: Real> Charges<R> {
    /// Evaluates base-emitter, internal base-collector, and external base-collector charges
    /// at [`VARIABLES`] `voltages` and intrinsic `state`.
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

/// Substrate junction joining `substrate` to the internal collector (vertical) or internal
/// base (lateral).
#[derive(Debug, Clone, Copy)]
struct SubstrateJunction<R: Real> {
    /// Polarity factor (`±1`) orienting substrate voltage and current to match a vertical
    /// NPN junction.
    sign: f64,
    /// Optional saturation current and exponential slope for the substrate diode.
    diode: Option<(R, R)>,
    /// Thermal voltage at operating temperature.
    thermal_voltage: f64,
    critical_voltage: f64,
    /// Depletion model whose capacitance extends linearly from zero bias.
    depletion: Option<Depletion<R>>,
}

/// Critical voltage used by ngspice for step limiting on a diodeless substrate junction.
const DIODELESS_CRITICAL_VOLTAGE: f64 = 50.0;

impl<R: Real> SubstrateJunction<R> {
    /// Adds the junction's current and its charge, between the substrate and the internal
    /// node it joins, `nodes`, at `at`, its last voltage `last`, which it updates.
    fn evaluate(&self, nodes: &Nodes<2>, last: &mut f64, at: &Local<'_>, stamp: &mut Stamp<'_, R>) {
        let pair = (0, 1);
        let bias = Bias::limit_junctions(
            nodes,
            (self.sign, [pair]),
            std::array::from_mut(last),
            (self.thermal_voltage, self.critical_voltage),
            at,
            stamp,
        );
        let [voltage] = Dual::variables(bias.across(nodes, at, (self.sign, [pair])));
        let current = self.diode.map_or(Dual::number(0.0), |(saturation, slope)| {
            junction_current(voltage, saturation, slope)
        }) + voltage * at.gmin;
        let [conductance] = current.gradient;
        nodes.current(stamp, 0, 1, current.extend(bias.beyond) * self.sign);
        nodes.conductance(stamp, pair, pair, conductance);
        if let Some(depletion) = &self.depletion {
            let charge = depletion.evaluate(voltage);
            let [capacitance] = charge.gradient;
            nodes.charge(stamp, 0, 1, charge.extend(bias.beyond) * self.sign);
            nodes.capacitance(stamp, pair, pair, capacitance);
        }
    }
}

impl<R: Real> BipolarTransistorParameters<R> {
    /// Validates DC parameters and prepares the temperature-scaled [`Junctions`] model at
    /// `temperatures`.
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
                "high_injection_exponent lies above 0 and at most 1, not {:?}",
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

    /// Computes `[forward_beta, reverse_beta, beta_factor]` at `temperatures`, where
    /// `beta_factor` scales leakage currents (`bjttemp.c`, lines 231 to 249).
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

    /// Computes temperature-scaled reciprocals of `[VAF, VAR]` and `[IKF, IKR]` at
    /// `temperatures`, returning zero for omitted (infinite) parameters (`bjttemp.c`, lines
    /// 83 to 104).
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

    /// Validates base resistance parameters and returns the nominal [`BaseResistance`] when
    /// `base_resistance` is positive.
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

    /// Validates and scales `RB`, `RBM`, and `IRB` to `temperatures` using their quadratic
    /// coefficients, defaulting omitted `RBM` to nominal `RB` (`bjttemp.c`, lines 47 to 49
    /// and 118 to 123).
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

    /// Validates charge parameters and prepares temperature-scaled [`Charges`] at
    /// `temperatures`, returning `None` when all capacitances and transit times are zero.
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
        fraction("internal_base_fraction", self.internal_base_fraction)?;
        non_negative("reverse_transit_time", self.reverse_transit_time)?;
        below_one("depletion_coefficient", self.depletion_coefficient)?;
        Ok(())
    }

    /// Computes temperature-scaled `[potential, capacitance, grading]` at `temperatures`
    /// (`bjttemp.c`, lines 132 to 146 and 260 to 313).
    ///
    /// Scales `grading` quadratically and clamps to [`MAX_GRADING`], then scales `potential`
    /// and `capacitance` via silicon band gap (`TLEVC = 0`) or linear coefficients
    /// (`TLEVC = 1`).
    ///
    /// # Errors
    ///
    /// Returns an error naming `potential_name` when the scaled potential is not positive.
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
    /// Validates substrate parameters and prepares the optional [`SubstrateJunction`] at
    /// `temperatures`.
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
            "{name} is positive at the transistor's temperature, not {:?}",
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
            "{name} is at least 0 at the transistor's temperature, not {:?}",
            value.value()
        )))
    }
}

/// Converts temperature-scaled series resistance `scaled` into the stamp representation
/// chosen from `nominal`, mirroring [`series_value`].
fn series_at<R: Real>(nominal: R, scaled: R) -> R {
    if nominal.value() < SMALL_RESISTANCE {
        scaled
    } else {
        scaled.recip()
    }
}

impl Kernel for BipolarTransistor {
    const NAME: &'static str = "bipolar_transistor";
    type Parameters<R: Real> = BipolarTransistorParameters<R>;
    type Shape = BipolarTransistorShape;
    type Prepared<R: Real> = PreparedBipolarTransistor<R>;
    type Memory = BipolarTransistorMemory;

    fn parameter_kinds() -> &'static [(&'static str, ParameterKind)] {
        BipolarTransistorParameters::<f64>::PARAMETER_KINDS
    }

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
        let bias = Bias::limit_junctions(
            &shape.stamp,
            (prepared.sign, VARIABLES),
            &mut memory.last,
            (prepared.thermal_voltage, prepared.critical_voltage),
            at,
            stamp,
        );
        let [vbe, vbc, ..] = bias.voltages;
        memory.variables = Bipolar::variables(vbe, vbc);
        let added = Oriented::new(&shape.stamp, prepared.sign, &bias, VARIABLES);
        let voltages = added.across(at);
        let [vbe, vbc, ..] = voltages;
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
            form.add(
                resistance.evaluate(&state),
                voltages[VRB],
                (&added, &mut *sums),
                at,
                stamp,
            );
        }
        if let Some(charges) = &prepared.charges {
            charges.evaluate(voltages, &state).add(&added, sums, stamp);
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

    /// Appends SPICE thermal noise for series resistances, collector and base shot noise,
    /// and base-current flicker noise.
    fn noise<R: Real>(
        shape: &BipolarTransistorShape,
        prepared: &PreparedBipolarTransistor<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource<R>>,
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
        let voltages = nodes.oriented::<R, _>(at, prepared.sign, [BASE_EMITTER, BASE_COLLECTOR]);
        let state = prepared.junctions.evaluate(voltages, at.gmin);
        let node = |index: usize| nodes.equation(index);
        if let (Some(resistance), Some(form)) = (&prepared.base, &shape.base) {
            let resistance = resistance.evaluate(&state).value;
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
        let base_current = base_emitter + base_collector;
        sources.push(NoiseSource::shot(
            "collector shot",
            [node(INNER_COLLECTOR), node(INNER_EMITTER)],
            collector,
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

/// Temperature-scaled intrinsic BJT quantities computed during preparation.
struct Scaled<R: Real> {
    /// Collector and emitter series resistance stamp values from [`series_at`].
    series: [R; 2],
    base: Option<BaseResistance<R>>,
    junctions: Junctions<R>,
    charges: Option<Charges<R>>,
    substrate: Option<SubstrateJunction<R>>,
    thermal_voltage: f64,
    critical_voltage: f64,
    temperature: R,
}

impl<R: Real> BipolarTransistorParameters<R> {
    /// Validates parameters and computes temperature-scaled BJT quantities at `temperature`,
    /// in kelvins.
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
            temperature: temperatures.actual,
        })
    }
}

impl<R: Real> BipolarTransistorParameters<R> {
    /// Validates `excess_phase` and returns the group delay
    /// `excess_phase · π / 180 · forward_transit_time`.
    fn excess_phase_delay(&self) -> Result<R, ParameterError> {
        let excess_phase = ParameterError::non_negative("excess_phase", self.excess_phase)?;
        Ok(excess_phase * (std::f64::consts::PI / 180.0) * self.forward_transit_time)
    }

    /// Validates and returns the `[collector_resistance, emitter_resistance]` pair.
    fn series_resistances(&self) -> Result<[R; 2], ParameterError> {
        let non_negative = ParameterError::non_negative;
        Ok([
            non_negative("collector_resistance", self.collector_resistance)?,
            non_negative("emitter_resistance", self.emitter_resistance)?,
        ])
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::Input;
    use crate::kernel::region::{BASE_COLLECTOR, BASE_EMITTER};
    use crate::testing::{
        Bench, assert_bessel_filter, assert_close, assert_jacobians_match,
        assert_parameter_gradients_match, assert_regions, source, thermal,
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

    /// Returns [`FULL`] with 30° of excess phase (group delay `30 · π / 180 · 0.4 ns`).
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
        // The `y` residual equals `3 · (x + y - u)`, yielding forward transport `u = Ibe / qb`
        // at `x = y = 0`. At DC equilibrium `x = u` and terminal currents match the undelayed
        // transistor.
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
        // through 3 / (s² td² + 3 s td + 3): y's row holds -3 u.
        let mut delayed = Bench::new::<BipolarTransistor>(&excess());
        let evaluation = delayed.settle(&[3.0, 0.7, 0.0, 1e-3, 0.0]);
        let delay = 30.0 * std::f64::consts::PI / 180.0 * 4e-10;
        assert_bessel_filter(
            &evaluation,
            [3, 4],
            [0.0, 3.0],
            0,
            delay,
            [1e8, 1.0 / delay, 1e10],
        );
    }

    #[test]
    fn an_excess_phase_needs_a_transit_time_and_is_not_negative() {
        let without = Bench::new::<BipolarTransistor>(r#"{"excess_phase": 30}"#);
        assert_eq!(without.unknowns(), ["n0", "n1", "n2"]);
        let error = Bench::try_new::<BipolarTransistor>(r#"{"excess_phase": -1}"#).unwrap_err();
        assert_eq!(error.to_string(), "excess_phase is at least 0, not -1.0");
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
            "forward_early_voltage is positive, not 0.0"
        );
    }

    /// Every temperature coefficient, each moving its parameter by a few percent over the
    /// 10 K between the nominal temperature (290 K) and the reference temperature, with a
    /// substrate and the base resistance.
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

    /// Builds [`FULL`] augmented with [`COEFFICIENTS`] for all four `(TLEV, TLEVC)` pairs.
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
        let model = crate::Thermal::<BipolarTransistor>::new();
        for parameters in with_coefficients() {
            for x in [
                [3.0, 0.7, 0.0, -1.0, 2.98, 0.68, 0.001],
                [0.2, 0.75, 0.0, 0.3, 0.19, 0.73, 0.002],
                [-0.5, 0.1, 0.0, -2.0, -0.49, 0.09, 0.0],
            ] {
                assert_jacobians_match::<BipolarTransistor>(&parameters, &x);
                assert_parameter_gradients_match::<BipolarTransistor>(&parameters, &x);
            }
            // Self-heating differentiates all temperature coefficients with respect to
            // junction temperature.
            let heated = parameters.replace('{', r#"{"thermal": true, "#);
            for x in [
                [3.0, 0.7, 0.0, -1.0, 20.0, 2.98, 0.69, 0.001],
                [0.3, 0.75, 0.0, 0.2, 70.0, 0.29, 0.74, 0.002],
            ] {
                crate::testing::assert_model_jacobians_match(&model, &heated, &x);
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
        /// Verifies that analytical conductances and capacitances match finite differences
        /// in forward, reverse, and saturated modes with base-charge or base-current
        /// resistance modulation.
        ///
        /// Junction voltages stay below 0.85 V to avoid unphysical current overflow.
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
            let [
                collector,
                base,
                inner_collector,
                inner_base,
                inner_emitter,
            ] = [collector, base, 0.98 * collector, 0.97 * base, 0.01].map(|value| sign * value);
            assert_jacobians_match::<BipolarTransistor>(&parameters, &[collector, base, 0.0]);
            assert_jacobians_match::<BipolarTransistor>(
                &parameters.replace('{', r#"{"excess_phase": 45, "#),
                &[collector, base, 0.0, 1e-3, -2e-4],
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
            let substrate_voltage = base - sign * 0.4;
            assert_jacobians_match::<BipolarTransistor>(
                &substrate,
                &[
                    collector,
                    base,
                    0.0,
                    substrate_voltage,
                    inner_collector,
                    inner_base,
                    inner_emitter,
                ],
            );
            assert_jacobians_match::<BipolarTransistor>(
                &resistive,
                &[
                    collector,
                    base,
                    0.0,
                    inner_collector,
                    inner_base,
                    inner_emitter,
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

    /// Verifies that [`BipolarTransistor`] classifies cut-off, active, saturated, and
    /// reversed regions with PNP mirror polarity.
    ///
    /// Margins equal the junction voltages and vanish at every region boundary.
    #[test]
    fn reports_its_region_and_margins() {
        // The collector, the base and the emitter.
        assert_regions::<BipolarTransistor>(
            "{}",
            Some(r#"{"polarity": "pnp"}"#),
            &[
                (&[5.0, -0.5, 0.0], "cut_off", Some(&[-0.5, -5.5])),
                (&[5.0, 0.7, 0.0], "active", Some(&[0.7, -4.3])),
                (&[0.2, 0.7, 0.0], "saturated", Some(&[0.7, 0.7 - 0.2])),
                (&[0.0, 0.7, 5.0], "reversed", Some(&[-4.3, 0.7])),
            ],
            &[
                (&[5.0, -0.1, 0.0], &[5.0, 0.1, 0.0], BASE_EMITTER),
                (&[5.0, 0.7, 0.0], &[0.2, 0.7, 0.0], BASE_COLLECTOR),
                (&[0.0, -0.1, 5.0], &[0.0, 0.7, 5.0], BASE_COLLECTOR),
            ],
        );
    }
}
