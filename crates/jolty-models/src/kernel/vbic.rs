//! The VBIC 1.2 bipolar transistor, derived from ngspice 47's implementation of it in
//! `src/spicelib/devices/vbic/`:
//! Copyright 1985 - 2018, Regents of the University of California and others. Used under
//! ngspice's Modified BSD license, which the repository's `NOTICE` reproduces.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::dimension::{
    CAPACITANCE, CHARGE, CURRENT, PER_KELVIN, PER_KELVIN_SQUARED, PER_VOLT, RESISTANCE,
    TEMPERATURE, TIME, VOLTAGE,
};
use crate::kernel::dual::Dual;
use crate::kernel::junction::{Temperatures, below_one, critical_voltage};
use crate::kernel::parts::{
    self, Bias, Derivatives, Nodes, Oriented, Pair, Series, fraction, inverse,
};
use crate::kernel::region::Bipolar;
use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Lift, Local, Nature, NoiseSource,
    ParameterError, ParameterKind, ParameterMetadata, Polarity, REFERENCE_TEMPERATURE, Real,
    Regions, Stamp,
};

/// VBIC's Boltzmann constant, in joules per kelvin: the one its definition's reference code
/// fixes, with which ngspice's VBIC computes its thermal voltage.
const VBIC_BOLTZMANN: f64 = 1.380_662e-23;

/// VBIC elementary charge in coulombs, paired with [`VBIC_BOLTZMANN`].
const VBIC_CHARGE: f64 = 1.602_189e-19;

/// Vertical bipolar inter-company model (VBIC 1.2) between `collector`, `base`, `emitter`,
/// and `substrate`, matching ngspice 47. Follows McAndrew et al., *VBIC95, the vertical
/// bipolar inter-company model*, IEEE Journal of Solid-State Circuits 31(10), 1996, as
/// extended by VBIC 1.2.
///
/// Intrinsic transport current `(Itzf - Itzr)` equals `(Ifi - Iri) / qb` from internal
/// emitter to internal collector. Forward and reverse currents are
/// `Ifi = IS · (e^{vbei / (NF · Vt)} - 1)` and
/// `Iri = IS · ISRR · (e^{vbci / (NR · Vt)} - 1)`. Normalized base charge `qb` combines
/// the Early effect via smoothed depletion charges with high injection via knee currents
/// in VBIC or Gummel-Poon form. Ideal and non-ideal base-emitter currents split between
/// internal and external base with breakdown, while base-collector junctions carry weak
/// avalanche current driven by transport current. A parasitic substrate transistor runs
/// from external base through parasitic base to substrate with dedicated junctions,
/// transport current, and knee current. Intrinsic collector resistance follows Kull's
/// quasi-saturation model with epitaxial velocity saturation, and intrinsic and parasitic
/// base resistances vary inversely with base charge. Junctions store VBIC-smoothed
/// depletion charges (with reach-through on base-collector junctions), forward and reverse
/// transit charges, epitaxial charges, and terminal overlap capacitances.
///
/// Positive extrinsic collector, extrinsic base, emitter, and substrate resistances add
/// internal nodes `collector`, `base`, `emitter`, and `substrate`. Any resistance below
/// [`SMALL_RESISTANCE`](crate::SMALL_RESISTANCE) adds branch-current unknowns
/// `collector_current`, `base_current`, `emitter_current`, or `substrate_current`.
/// Intrinsic collector, intrinsic base, and parasitic base are always internal nodes
/// (`intrinsic_collector`, `intrinsic_base`, `parasitic_base`). Excess-phase delay filters
/// forward transport current through `3 / (s² · TD² + 3 · s · TD + 3)` using unknowns
/// `transport_filter` and `delayed_transport`.
///
/// Away from nominal temperature, saturation currents scale by activation energies and
/// exponents, while resistances, knee current, and saturation voltage scale by temperature
/// exponents. Built-in potentials, capacitances, emission coefficients, avalanche, and
/// breakdown scale per ngspice VBIC using VBIC thermal-voltage constants.
#[derive(Debug, Clone, Copy)]
pub struct Vbic;

/// Normalized base-charge formulation combining Early effect `q1` and high injection `q2`
/// (`QBM`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BaseCharge {
    /// VBIC formulation `qb = (q1 + (q1^(1/NKF) + 4 · q2)^NKF) / 2` (`QBM < 0.5`).
    Vbic,
    /// SPICE Gummel-Poon formulation `qb = q1 · (1 + (1 + 4 · q2)^NKF) / 2`
    /// (`QBM >= 0.5`).
    GummelPoon,
}

/// The terminals' local indices.
const COLLECTOR: usize = 0;
const BASE: usize = 1;
const EMITTER: usize = 2;
const SUBSTRATE: usize = 3;

/// Stamped node indices: the collector, base, emitter and substrate behind the extrinsic
/// collector, the extrinsic base, the emitter and the substrate resistances, the terminals
/// themselves without them, and the intrinsic collector, intrinsic base and parasitic base.
const CX: usize = 0;
const CI: usize = 1;
const BX: usize = 2;
const BI: usize = 3;
const EI: usize = 4;
const BP: usize = 5;
const SI: usize = 6;

/// The number of stamped nodes.
const STAMPED: usize = 7;

/// The number of voltages the transistor's equations take as variables.
const VARIABLE_COUNT: usize = 9;

/// The number of those voltages, the first, across junctions, whose steps are limited.
const JUNCTIONS: usize = 6;

/// The pairs of stamped nodes across which the variables lie, in ngspice's names: `Vbei`,
/// `Vbex`, `Vbci`, `Vbcx`, `Vbep`, `Vbcp`, `Vrci`, `Vrbi` and `Vrbp`.
const VARIABLES: [(usize, usize); VARIABLE_COUNT] = [
    (BI, EI),
    (BX, EI),
    (BI, CI),
    (BI, CX),
    (BX, BP),
    (SI, BP),
    (CX, CI),
    (BX, BI),
    (BP, CX),
];

/// The variables' positions among [`VARIABLES`], by ngspice's names.
const VBEI: usize = 0;
const VBEX: usize = 1;
const VBCI: usize = 2;
const VBCX: usize = 3;
const VBEP: usize = 4;
const VBCP: usize = 5;
const VRCI: usize = 6;
const VRBI: usize = 7;
const VRBP: usize = 8;

/// The variables each law of the transistor's equations reads, by their positions among
/// [`VARIABLES`], in the order its numbers carry their derivatives: each law computes at
/// the width of those alone.
///
/// The intrinsic transistor's transport currents, avalanche and base-emitter charge read
/// its base-emitter and base-collector junctions.
const INTRINSIC: [usize; 2] = [VBEI, VBCI];
/// Variable indices read by intrinsic base resistance current: branch drop then intrinsic
/// junctions.
const INTRINSIC_BASE: [usize; 3] = [VRBI, VBEI, VBCI];
/// Variable indices read by intrinsic collector resistance current: branch drop then
/// epitaxial-end junctions.
const INTRINSIC_COLLECTOR: [usize; 3] = [VRCI, VBCI, VBCX];
/// Variable indices read by parasitic base-emitter charge: parasitic base-emitter and
/// intrinsic base-collector junctions.
const PARASITIC_FORWARD: [usize; 2] = [VBEP, VBCI];
/// Variable indices read by parasitic transport current: parasitic base-emitter, intrinsic
/// base-collector, and parasitic base-collector junctions.
const PARASITIC: [usize; 3] = [VBEP, VBCI, VBCP];
/// Variable indices read by parasitic base resistance current: branch drop then parasitic
/// base-charge junctions.
const PARASITIC_BASE: [usize; 3] = [VRBP, VBEP, VBCI];

/// Parameters of a [`Vbic`] bipolar transistor with ngspice defaults. Parameters treated
/// as infinite at zero (such as Early voltages) default to `None` when omitted, and non-SI
/// defaults (such as `TNOM`) are converted to SI units.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift, ParameterMetadata)]
#[serde(default, deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct VbicParameters<R: Real = f64> {
    /// The doping order, `NPN` or `PNP`; default `npn`.
    pub polarity: Polarity,
    /// The extrinsic collector resistance, `RCX`, in ohms; default 0.
    #[dimension(RESISTANCE)]
    pub extrinsic_collector_resistance: R,
    /// The intrinsic collector resistance, `RCI`, in ohms, positive; default 0.1.
    #[dimension(RESISTANCE)]
    pub intrinsic_collector_resistance: R,
    /// The epitaxial layer's drift saturation voltage, `VO`, in volts; infinite by default, for
    /// no velocity saturation.
    #[dimension(VOLTAGE)]
    pub epitaxial_saturation_voltage: Option<R>,
    /// The epitaxial layer's doping parameter, `GAMM`; default 0.
    pub epitaxial_doping_parameter: R,
    /// The high-current factor of the intrinsic collector resistance, `HRCF`, positive;
    /// default 1.
    pub high_current_collector_factor: R,
    /// The extrinsic base resistance, `RBX`, in ohms; default 0.
    #[dimension(RESISTANCE)]
    pub extrinsic_base_resistance: R,
    /// The intrinsic base resistance, `RBI`, in ohms, positive; default 0.1.
    #[dimension(RESISTANCE)]
    pub intrinsic_base_resistance: R,
    /// The emitter resistance, `RE`, in ohms; default 0.
    #[dimension(RESISTANCE)]
    pub emitter_resistance: R,
    /// The substrate resistance, `RS`, in ohms; default 0.
    #[dimension(RESISTANCE)]
    pub substrate_resistance: R,
    /// The parasitic base resistance, `RBP`, in ohms, positive; default 0.1.
    #[dimension(RESISTANCE)]
    pub parasitic_base_resistance: R,
    /// The transport saturation current, `IS`, in amperes; default 1e-16.
    #[dimension(CURRENT)]
    pub saturation_current: R,
    /// The forward emission coefficient, `NF`; default 1.
    pub forward_emission_coefficient: R,
    /// The reverse emission coefficient, `NR`; default 1.
    pub reverse_emission_coefficient: R,
    /// The fraction of each junction's potential where its depletion capacitance's
    /// smoothing sets in, `FC`, below 1; default 0.9.
    pub depletion_coefficient: R,
    /// The base-emitter overlap capacitance, `CBEO`, in farads, between the terminals;
    /// default 0.
    #[dimension(CAPACITANCE)]
    pub base_emitter_overlap_capacitance: R,
    /// The zero-bias base-emitter depletion capacitance, `CJE`, in farads; default 0.
    #[dimension(CAPACITANCE)]
    pub base_emitter_capacitance: R,
    /// The base-emitter built-in potential, `PE`, in volts; default 0.75.
    #[dimension(VOLTAGE)]
    pub base_emitter_potential: R,
    /// The base-emitter grading coefficient, `ME`, below 1; default 0.33.
    pub base_emitter_grading: R,
    /// The base-emitter capacitance's smoothing, `AJE`, in volts: SPICE's linear extension
    /// above `FC · PE` at zero or below; default -0.5.
    #[dimension(VOLTAGE)]
    pub base_emitter_smoothing: R,
    /// The base-collector overlap capacitance, `CBCO`, in farads, between the terminals;
    /// default 0.
    #[dimension(CAPACITANCE)]
    pub base_collector_overlap_capacitance: R,
    /// The zero-bias intrinsic base-collector depletion capacitance, `CJC`, in farads;
    /// default 0.
    #[dimension(CAPACITANCE)]
    pub base_collector_capacitance: R,
    /// The epitaxial layer's charge, `QCO`, in coulombs; default 0.
    #[dimension(CHARGE)]
    pub epitaxial_charge: R,
    /// The zero-bias extrinsic base-collector depletion capacitance, `CJEP`, in farads, the
    /// parasitic transistor's base-emitter junction's; default 0.
    #[dimension(CAPACITANCE)]
    pub extrinsic_base_collector_capacitance: R,
    /// The base-collector built-in potential, `PC`, in volts; default 0.75.
    #[dimension(VOLTAGE)]
    pub base_collector_potential: R,
    /// The base-collector grading coefficient, `MC`, below 1; default 0.33.
    pub base_collector_grading: R,
    /// The base-collector capacitances' smoothing, `AJC`, in volts; default -0.5.
    #[dimension(VOLTAGE)]
    pub base_collector_smoothing: R,
    /// The zero-bias collector-substrate depletion capacitance, `CJCP`, in farads; default
    /// 0.
    #[dimension(CAPACITANCE)]
    pub collector_substrate_capacitance: R,
    /// The collector-substrate built-in potential, `PS`, in volts; default 0.75.
    #[dimension(VOLTAGE)]
    pub collector_substrate_potential: R,
    /// The collector-substrate grading coefficient, `MS`, below 1; default 0.33.
    pub collector_substrate_grading: R,
    /// The collector-substrate capacitance's smoothing, `AJS`, in volts; default -0.5.
    #[dimension(VOLTAGE)]
    pub collector_substrate_smoothing: R,
    /// The fixed collector-substrate capacitance, `CCSO`, in farads; default 0.
    #[dimension(CAPACITANCE)]
    pub collector_substrate_fixed_capacitance: R,
    /// The ideal base-emitter saturation current, `IBEI`, in amperes; default 1e-18.
    #[dimension(CURRENT)]
    pub ideal_base_emitter_current: R,
    /// The fraction of the base-emitter currents and depletion charge at the intrinsic
    /// base, `WBE`, the rest at the extrinsic base, from 0 to 1; default 1.
    pub intrinsic_base_emitter_fraction: R,
    /// The ideal base-emitter emission coefficient, `NEI`; default 1.
    pub ideal_base_emitter_emission_coefficient: R,
    /// The non-ideal base-emitter saturation current, `IBEN`, in amperes; default 0.
    #[dimension(CURRENT)]
    pub nonideal_base_emitter_current: R,
    /// The non-ideal base-emitter emission coefficient, `NEN`; default 2.
    pub nonideal_base_emitter_emission_coefficient: R,
    /// The ideal base-collector saturation current, `IBCI`, in amperes; default 1e-16.
    #[dimension(CURRENT)]
    pub ideal_base_collector_current: R,
    /// The ideal base-collector emission coefficient, `NCI`; default 1.
    pub ideal_base_collector_emission_coefficient: R,
    /// The non-ideal base-collector saturation current, `IBCN`, in amperes; default 0.
    #[dimension(CURRENT)]
    pub nonideal_base_collector_current: R,
    /// The non-ideal base-collector emission coefficient, `NCN`; default 2.
    pub nonideal_base_collector_emission_coefficient: R,
    /// The weak avalanche's coefficient, `AVC1`, per volt; default 0, no avalanche.
    #[dimension(PER_VOLT)]
    pub avalanche_coefficient: R,
    /// The weak avalanche's exponent coefficient, `AVC2`, in volts to the power `1 - MC`;
    /// default 0.
    pub avalanche_exponent_coefficient: R,
    /// The parasitic transistor's transport saturation current, `ISP`, in amperes; default
    /// 0, no parasitic transport.
    #[dimension(CURRENT)]
    pub parasitic_saturation_current: R,
    /// The fraction of the parasitic transport current the extrinsic base-collector
    /// junction drives, `WSP`, the rest the intrinsic one, from 0 to 1; default 1.
    pub parasitic_extrinsic_fraction: R,
    /// The parasitic transistor's forward emission coefficient, `NFP`; default 1.
    pub parasitic_forward_emission_coefficient: R,
    /// The parasitic transistor's ideal base-emitter saturation current, `IBEIP`, in
    /// amperes, across the extrinsic base-collector junction; default 0.
    #[dimension(CURRENT)]
    pub parasitic_ideal_base_emitter_current: R,
    /// The parasitic transistor's non-ideal base-emitter saturation current, `IBENP`, in
    /// amperes; default 0.
    #[dimension(CURRENT)]
    pub parasitic_nonideal_base_emitter_current: R,
    /// The parasitic transistor's ideal base-collector saturation current, `IBCIP`, in
    /// amperes, across the collector-substrate junction; default 0.
    #[dimension(CURRENT)]
    pub parasitic_ideal_base_collector_current: R,
    /// The parasitic transistor's ideal base-collector emission coefficient, `NCIP`;
    /// default 1.
    pub parasitic_ideal_base_collector_emission_coefficient: R,
    /// The parasitic transistor's non-ideal base-collector saturation current, `IBCNP`, in
    /// amperes; default 0.
    #[dimension(CURRENT)]
    pub parasitic_nonideal_base_collector_current: R,
    /// The parasitic transistor's non-ideal base-collector emission coefficient, `NCNP`;
    /// default 2.
    pub parasitic_nonideal_base_collector_emission_coefficient: R,
    /// The forward Early voltage, `VEF`, in volts; infinite by default.
    #[dimension(VOLTAGE)]
    pub forward_early_voltage: Option<R>,
    /// The reverse Early voltage, `VER`, in volts; infinite by default.
    #[dimension(VOLTAGE)]
    pub reverse_early_voltage: Option<R>,
    /// The forward knee current, `IKF`, in amperes; infinite by default.
    #[dimension(CURRENT)]
    pub forward_knee_current: Option<R>,
    /// The reverse knee current, `IKR`, in amperes; infinite by default.
    #[dimension(CURRENT)]
    pub reverse_knee_current: Option<R>,
    /// The parasitic transistor's knee current, `IKP`, in amperes; infinite by default.
    #[dimension(CURRENT)]
    pub parasitic_knee_current: Option<R>,
    /// The power of the high-injection roll-off, `NKF`, positive; default 0.5.
    pub knee_rolloff: R,
    /// Normalized base-charge formulation selected by `QBM`; default `vbic`.
    pub base_charge: BaseCharge,
    /// Ideal forward transit time, `TF`, in seconds; default 0.
    #[dimension(TIME)]
    pub forward_transit_time: R,
    /// Dependence of forward transit time on Early base charge, `QTF`; default 0.
    pub transit_time_base_width_coefficient: R,
    /// Bias-dependent growth coefficient of forward transit time, `XTF`; default 0.
    pub transit_time_bias_coefficient: R,
    /// The base-collector voltage, `VTF`, that sets how the forward transit time grows
    /// with it, in volts; infinite by default.
    #[dimension(VOLTAGE)]
    pub transit_time_voltage: Option<R>,
    /// The current, `ITF`, above which the forward transit time grows, in amperes; default
    /// 0, for growth at any current.
    #[dimension(CURRENT)]
    pub transit_time_current: R,
    /// The ideal reverse transit time, `TR`, in seconds; default 0.
    #[dimension(TIME)]
    pub reverse_transit_time: R,
    /// The forward excess-phase delay, `TD`, in seconds; default 0, no excess phase.
    #[dimension(TIME)]
    pub excess_phase_delay: R,
    /// The flicker noise coefficient of the base-emitter currents, `KFN`; default 0, no
    /// flicker noise.
    pub flicker_coefficient: R,
    /// The power of the current in the flicker noise, `AFN`; default 1.
    pub flicker_exponent: R,
    /// The power of the frequency the flicker noise falls with, `BFN`; default 1.
    pub flicker_frequency_exponent: R,
    /// The punch-through voltage of the base-collector junctions, `VRT`, in volts; default
    /// 0, no reach-through.
    #[dimension(VOLTAGE)]
    pub reach_through_voltage: R,
    /// The smoothing of the reach-through, `ART`; default 0.1.
    pub reach_through_smoothing: R,
    /// The base-emitter breakdown voltage, `VBBE`, in volts; without it, the junction never
    /// breaks down.
    #[dimension(VOLTAGE)]
    pub base_emitter_breakdown_voltage: Option<R>,
    /// The base-emitter breakdown's emission coefficient, `NBBE`; default 1.
    pub base_emitter_breakdown_emission_coefficient: R,
    /// The base-emitter breakdown current, `IBBE`, in amperes; default 1e-6.
    #[dimension(CURRENT)]
    pub base_emitter_breakdown_current: R,
    /// The ratio of the reverse to the forward transport saturation current, `ISRR`;
    /// default 1.
    pub reverse_saturation_current_ratio: R,
    /// The temperature the parameters were measured at, `TNOM`, in kelvins; default 27 °C.
    #[dimension(TEMPERATURE)]
    pub nominal_temperature: R,
    /// The activation energy of the transport saturation current and of the epitaxial doping,
    /// `EA`, in volts; default 1.12.
    #[dimension(VOLTAGE)]
    pub activation_energy: R,
    /// The activation energy of the ideal base-emitter current and of its potential,
    /// `EAIE`, in volts; default 1.12.
    #[dimension(VOLTAGE)]
    pub ideal_base_emitter_activation_energy: R,
    /// The activation energy of the ideal base-collector currents and of their potential,
    /// `EAIC`, in volts; default 1.12.
    #[dimension(VOLTAGE)]
    pub ideal_base_collector_activation_energy: R,
    /// The activation energy of the parasitic ideal base-collector current and of the
    /// collector-substrate potential, `EAIS`, in volts; default 1.12.
    #[dimension(VOLTAGE)]
    pub ideal_collector_substrate_activation_energy: R,
    /// The activation energy of the non-ideal base-emitter current, `EANE`, in volts;
    /// default 1.12.
    #[dimension(VOLTAGE)]
    pub nonideal_base_emitter_activation_energy: R,
    /// The activation energy of the non-ideal base-collector currents, `EANC`, in volts;
    /// default 1.12.
    #[dimension(VOLTAGE)]
    pub nonideal_base_collector_activation_energy: R,
    /// The activation energy of the parasitic non-ideal base-collector current, `EANS`, in
    /// volts; default 1.12.
    #[dimension(VOLTAGE)]
    pub nonideal_collector_substrate_activation_energy: R,
    /// The activation energy of the parasitic transport saturation current, `EAP`, in
    /// volts; default 1.12.
    #[dimension(VOLTAGE)]
    pub parasitic_activation_energy: R,
    /// The shift of the activation energy of the reverse transport current, `DEAR`, in
    /// volts; default 0.
    #[dimension(VOLTAGE)]
    pub reverse_activation_energy_shift: R,
    /// The exponent of the temperature ratio in the transport saturation currents' scaling,
    /// `XIS`; default 3.
    pub saturation_current_temperature_exponent: R,
    /// The exponent of the temperature ratio in the ideal base currents' scaling, `XII`;
    /// default 3.
    pub ideal_base_current_temperature_exponent: R,
    /// The exponent of the temperature ratio in the non-ideal base currents' scaling,
    /// `XIN`; default 3.
    pub nonideal_base_current_temperature_exponent: R,
    /// The exponent of the temperature ratio in the reverse saturation current ratio's
    /// scaling, `XISR`; default 0.
    pub reverse_saturation_current_temperature_exponent: R,
    /// The linear temperature coefficient of the forward and reverse emission
    /// coefficients, `TNF`, per kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub emission_coefficient_temperature_coefficient: R,
    /// The linear temperature coefficient of the avalanche's exponent coefficient, `TAVC`,
    /// per kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub avalanche_temperature_coefficient: R,
    /// The exponent of the temperature ratio that scales the forward knee current, `XIKF`;
    /// default 0.
    pub forward_knee_current_temperature_exponent: R,
    /// The exponent of the temperature ratio that scales the extrinsic collector
    /// resistance, `XRCX`; default 0.
    pub extrinsic_collector_resistance_temperature_exponent: R,
    /// The exponent of the temperature ratio that scales the intrinsic collector
    /// resistance, `XRCI`; default 0.
    pub intrinsic_collector_resistance_temperature_exponent: R,
    /// The exponent of the temperature ratio that scales the extrinsic base resistance,
    /// `XRBX`; default 0.
    pub extrinsic_base_resistance_temperature_exponent: R,
    /// The exponent of the temperature ratio that scales the intrinsic base resistance,
    /// `XRBI`; default 0.
    pub intrinsic_base_resistance_temperature_exponent: R,
    /// The exponent of the temperature ratio that scales the emitter resistance, `XRE`;
    /// default 0.
    pub emitter_resistance_temperature_exponent: R,
    /// The exponent of the temperature ratio that scales the substrate resistance, `XRS`;
    /// default 0.
    pub substrate_resistance_temperature_exponent: R,
    /// The exponent of the temperature ratio that scales the parasitic base resistance,
    /// `XRBP`; default 0.
    pub parasitic_base_resistance_temperature_exponent: R,
    /// The exponent of the temperature ratio that scales the epitaxial saturation voltage, `XVO`;
    /// default 0.
    pub epitaxial_saturation_voltage_temperature_exponent: R,
    /// The linear temperature coefficient of the base-emitter breakdown voltage, `TVBBE1`,
    /// per kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub breakdown_voltage_temperature_coefficient: R,
    /// The quadratic temperature coefficient of the base-emitter breakdown voltage,
    /// `TVBBE2`, per square kelvin; default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub breakdown_voltage_quadratic_temperature_coefficient: R,
    /// The linear temperature coefficient of the breakdown's emission coefficient,
    /// `TNBBE`, per kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub breakdown_emission_temperature_coefficient: R,
}

impl<R: Real> Default for VbicParameters<R> {
    #[expect(
        clippy::too_many_lines,
        reason = "one line for each of the 103 parameters of VBIC"
    )]
    fn default() -> Self {
        let zero = R::constant(0.0);
        let one = R::constant(1.0);
        let gap = R::constant(1.12);
        let three = R::constant(3.0);
        Self {
            polarity: Polarity::Npn,
            extrinsic_collector_resistance: zero,
            intrinsic_collector_resistance: R::constant(0.1),
            epitaxial_saturation_voltage: None,
            epitaxial_doping_parameter: zero,
            high_current_collector_factor: one,
            extrinsic_base_resistance: zero,
            intrinsic_base_resistance: R::constant(0.1),
            emitter_resistance: zero,
            substrate_resistance: zero,
            parasitic_base_resistance: R::constant(0.1),
            saturation_current: R::constant(1e-16),
            forward_emission_coefficient: one,
            reverse_emission_coefficient: one,
            depletion_coefficient: R::constant(0.9),
            base_emitter_overlap_capacitance: zero,
            base_emitter_capacitance: zero,
            base_emitter_potential: R::constant(0.75),
            base_emitter_grading: R::constant(0.33),
            base_emitter_smoothing: R::constant(-0.5),
            base_collector_overlap_capacitance: zero,
            base_collector_capacitance: zero,
            epitaxial_charge: zero,
            extrinsic_base_collector_capacitance: zero,
            base_collector_potential: R::constant(0.75),
            base_collector_grading: R::constant(0.33),
            base_collector_smoothing: R::constant(-0.5),
            collector_substrate_capacitance: zero,
            collector_substrate_potential: R::constant(0.75),
            collector_substrate_grading: R::constant(0.33),
            collector_substrate_smoothing: R::constant(-0.5),
            collector_substrate_fixed_capacitance: zero,
            ideal_base_emitter_current: R::constant(1e-18),
            intrinsic_base_emitter_fraction: one,
            ideal_base_emitter_emission_coefficient: one,
            nonideal_base_emitter_current: zero,
            nonideal_base_emitter_emission_coefficient: R::constant(2.0),
            ideal_base_collector_current: R::constant(1e-16),
            ideal_base_collector_emission_coefficient: one,
            nonideal_base_collector_current: zero,
            nonideal_base_collector_emission_coefficient: R::constant(2.0),
            avalanche_coefficient: zero,
            avalanche_exponent_coefficient: zero,
            parasitic_saturation_current: zero,
            parasitic_extrinsic_fraction: one,
            parasitic_forward_emission_coefficient: one,
            parasitic_ideal_base_emitter_current: zero,
            parasitic_nonideal_base_emitter_current: zero,
            parasitic_ideal_base_collector_current: zero,
            parasitic_ideal_base_collector_emission_coefficient: one,
            parasitic_nonideal_base_collector_current: zero,
            parasitic_nonideal_base_collector_emission_coefficient: R::constant(2.0),
            forward_early_voltage: None,
            reverse_early_voltage: None,
            forward_knee_current: None,
            reverse_knee_current: None,
            parasitic_knee_current: None,
            knee_rolloff: R::constant(0.5),
            base_charge: BaseCharge::Vbic,
            forward_transit_time: zero,
            transit_time_base_width_coefficient: zero,
            transit_time_bias_coefficient: zero,
            transit_time_voltage: None,
            transit_time_current: zero,
            reverse_transit_time: zero,
            excess_phase_delay: zero,
            flicker_coefficient: zero,
            flicker_exponent: one,
            flicker_frequency_exponent: one,
            reach_through_voltage: zero,
            reach_through_smoothing: R::constant(0.1),
            base_emitter_breakdown_voltage: None,
            base_emitter_breakdown_emission_coefficient: one,
            base_emitter_breakdown_current: R::constant(1e-6),
            reverse_saturation_current_ratio: one,
            nominal_temperature: R::constant(REFERENCE_TEMPERATURE),
            activation_energy: gap,
            ideal_base_emitter_activation_energy: gap,
            ideal_base_collector_activation_energy: gap,
            ideal_collector_substrate_activation_energy: gap,
            nonideal_base_emitter_activation_energy: gap,
            nonideal_base_collector_activation_energy: gap,
            nonideal_collector_substrate_activation_energy: gap,
            parasitic_activation_energy: gap,
            reverse_activation_energy_shift: zero,
            saturation_current_temperature_exponent: three,
            ideal_base_current_temperature_exponent: three,
            nonideal_base_current_temperature_exponent: three,
            reverse_saturation_current_temperature_exponent: zero,
            emission_coefficient_temperature_coefficient: zero,
            avalanche_temperature_coefficient: zero,
            forward_knee_current_temperature_exponent: zero,
            extrinsic_collector_resistance_temperature_exponent: zero,
            intrinsic_collector_resistance_temperature_exponent: zero,
            extrinsic_base_resistance_temperature_exponent: zero,
            intrinsic_base_resistance_temperature_exponent: zero,
            emitter_resistance_temperature_exponent: zero,
            substrate_resistance_temperature_exponent: zero,
            parasitic_base_resistance_temperature_exponent: zero,
            epitaxial_saturation_voltage_temperature_exponent: zero,
            breakdown_voltage_temperature_coefficient: zero,
            breakdown_voltage_quadratic_temperature_coefficient: zero,
            breakdown_emission_temperature_coefficient: zero,
        }
    }
}

/// Local matrix layout for [`Vbic`] covering extrinsic series resistances, seven intrinsic
/// nodes, terminal overlap capacitances, and optional excess-phase filter unknowns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VbicShape {
    stamp: Nodes<STAMPED>,
    /// The extrinsic collector, extrinsic base, emitter and substrate resistances.
    series: [Series; 4],
    /// The base-emitter and base-collector overlap capacitances, from the base terminal.
    overlap: [Option<Pair>; 2],
    excess: Option<ExcessPhase>,
}

/// Temperature-prepared parameters for [`Vbic`].
#[derive(Debug, Clone, Copy)]
pub struct PreparedVbic<R: Real> {
    sign: f64,
    /// The extrinsic collector, extrinsic base, emitter and substrate resistances.
    series: [R; 4],
    core: Core<R>,
    /// The base-emitter and base-collector overlap capacitances.
    overlap: [R; 2],
    /// The excess phase's delay, in seconds.
    delay: R,
    /// The thermal voltage the junctions' steps are limited with, and their critical
    /// voltage, ngspice's from the transport saturation current at its nominal
    /// temperature.
    slope: f64,
    critical_voltage: f64,
    temperature: R,
    flicker: Flicker<R>,
}

/// Step-limiting history and cached operating variables for [`Vbic`].
#[derive(Debug, Clone, Copy, Default)]
pub struct VbicMemory {
    /// The junction voltages of the last evaluation, oriented as an NPN's, from which the
    /// next steps are limited: `Vbei`, `Vbex`, `Vbci`, `Vbcx`, `Vbep` and `Vbcp`.
    last: [f64; JUNCTIONS],
    /// The values of [`Bipolar::VARIABLE_NAMES`] at the last evaluation.
    variables: [f64; 3],
}

/// The flicker noise of the base-emitter currents, `KFN · |I|^AFN / f^BFN`.
#[derive(Debug, Clone, Copy)]
struct Flicker<R: Real> {
    coefficient: R,
    power: R,
    exponent: R,
}

impl<R: Real> Flicker<R> {
    /// Appends a flicker noise source named `name` for `current` between `nodes` when
    /// `coefficient` is active.
    fn add(
        self,
        name: &'static str,
        nodes: [crate::Node; 2],
        current: R,
        sources: &mut Vec<NoiseSource<R>>,
    ) {
        if crate::noise::is_active(self.coefficient) {
            let mut source =
                NoiseSource::flicker(name, nodes, self.coefficient, self.power, current);
            source.exponent = self.exponent;
            sources.push(source);
        }
    }
}

/// Normalized junction depletion charge `q / Cj` with VBIC smoothing. Non-positive
/// smoothing uses SPICE's linear extension above `FC · P`, while positive smoothing
/// applies VBIC's smooth cap and optional base-collector reach-through.
#[derive(Debug, Clone, Copy)]
struct Depletion<R: Real> {
    potential: R,
    grading: R,
    smoothing: R,
    limit: R,
    /// The reach-through voltage, where it is positive, and its smoothing.
    reach: Option<(R, R)>,
}

impl<R: Real> Depletion<R> {
    /// Computes normalized depletion charge at junction `voltage`.
    fn charge<const N: usize>(&self, voltage: Dual<R, N>) -> Dual<R, N> {
        let Self {
            potential,
            grading,
            smoothing,
            limit,
            reach,
        } = *self;
        let shift = -(potential * limit);
        let complement = -(grading - 1.0);
        // P · (1 - (1 - v / P)^(1 - M)) / (1 - M), the power law's charge.
        let law = |voltage: Dual<R, N>| {
            -((-(voltage.divide(potential) - 1.0)).pow(complement) - 1.0)
                .scale(potential / complement)
        };
        if smoothing.value() <= 0.0 {
            let above = voltage.plus(shift);
            if above.value.value() > 0.0 {
                let limited = -(limit - 1.0);
                let factor = limited.pow(-(grading + 1.0));
                let low = potential * -(factor * limited * limited - 1.0) / complement;
                return (above * (above.scale(grading * 0.5 / potential).plus(limited)))
                    .scale(factor)
                    .plus(low);
            }
            return match reach {
                Some((through, _)) if voltage.value.value() < -through.value() => {
                    let reached = (through / potential + 1.0).pow(complement);
                    let slope = voltage
                        .plus(through)
                        .scale(complement / (potential + through));
                    -((-(slope - 1.0)).scale(reached) - 1.0).scale(potential / complement)
                }
                _ => law(voltage),
            };
        }
        let four_square = smoothing * smoothing * 4.0;
        match reach {
            Some((through, rounding)) if rounding.value() > 0.0 => {
                let span = through - shift;
                let four_rounding = rounding * rounding * 4.0;
                // `vnl`, the normalized voltage limited smoothly to (-1, 1).
                let limited = |normalized: Dual<R, N>| {
                    let below = normalized - 1.0;
                    let beyond = normalized + 1.0;
                    (normalized * 2.0)
                        / ((below * below).plus(four_square).sqrt()
                            + (beyond * beyond).plus(four_rounding).sqrt())
                };
                let at_zero = limited(Dual::constant((through + shift) / span));
                let normalized = limited((voltage * 2.0).plus(through + shift).divide(span));
                let edge =
                    |limited: Dual<R, N>| (limited.scale(span).plus(-(through + shift))) * 0.5;
                let (low_edge, edge_now) = (edge(at_zero), edge(normalized));
                let selected = (normalized + 1.0) * 0.5;
                let reached = (through / potential + 1.0).pow(-grading);
                let largest = (shift / potential + 1.0).pow(-grading);
                let capacitance = (-(selected - 1.0)).scale(reached) + selected.scale(largest);
                (voltage - edge_now + low_edge) * capacitance + law(edge_now) - law(low_edge)
            }
            _ => {
                let root = |offset: Dual<R, N>| (offset * offset).plus(four_square).sqrt();
                let at_zero = -(Dual::constant(shift) + root(Dual::constant(shift))) * 0.5;
                let offset = voltage.plus(shift);
                let edge = (offset - root(offset)) * 0.5 - Dual::constant(shift);
                let below = |edge: Dual<R, N>| {
                    -(-(edge.divide(potential) - 1.0))
                        .pow(complement)
                        .scale(potential / complement)
                };
                let slope = (-(limit - 1.0)).pow(-grading);
                below(edge) + (voltage - edge + at_zero).scale(slope) - below(at_zero)
            }
        }
    }
}

/// Temperature-scaled VBIC current, resistance, and charge parameters.
#[derive(Debug, Clone, Copy)]
struct Core<R: Real> {
    /// The thermal voltage, by VBIC's constants.
    thermal_voltage: R,
    transport: Transport<R>,
    base: BaseCurrents<R>,
    parasitic: Option<ParasiticTransport<R>>,
    collector: Collector<R>,
    /// The intrinsic and the parasitic base resistances.
    base_resistances: [R; 2],
    charges: Charges<R>,
}

/// The intrinsic transistor's transport current and its base charge.
#[derive(Debug, Clone, Copy)]
struct Transport<R: Real> {
    saturation_current: R,
    /// The reverse saturation current over the forward one.
    reverse_ratio: R,
    /// The forward and reverse emission coefficients.
    emission: [R; 2],
    /// Reciprocal Early voltages `1 / VEF` and `1 / VER` (zero for infinite Early voltages).
    inverse_early: [R; 2],
    /// Reciprocal knee currents `1 / IKF` and `1 / IKR` (zero for infinite knee currents).
    inverse_knee: [R; 2],
    form: BaseCharge,
    rolloff: R,
}

/// A saturation current and its emission coefficient.
type Diode<R> = (R, R);

/// The junctions' base currents.
#[derive(Debug, Clone, Copy)]
struct BaseCurrents<R: Real> {
    /// The ideal and non-ideal base-emitter diodes.
    emitter: [Diode<R>; 2],
    /// The fraction of them at the intrinsic base.
    fraction: R,
    breakdown: Option<Breakdown<R>>,
    /// The ideal and non-ideal base-collector diodes.
    collector: [Diode<R>; 2],
    /// Parasitic ideal and non-ideal base-emitter diodes across the extrinsic base-collector
    /// junction.
    parasitic_emitter: [Diode<R>; 2],
    /// Parasitic ideal and non-ideal base-collector diodes across the collector-substrate
    /// junction.
    parasitic_collector: [Diode<R>; 2],
    avalanche: Option<Avalanche<R>>,
}

/// Base-emitter junction breakdown current, zero at zero bias:
/// `-IBBE · (e^{(-VBBE - v) / (NBBE · Vt)} - EBBE)`.
#[derive(Debug, Clone, Copy)]
struct Breakdown<R: Real> {
    voltage: R,
    /// The emission coefficient times the thermal voltage.
    slope: R,
    current: R,
    /// Zero-bias breakdown exponential `e^{-VBBE / (NBBE · Vt)}`.
    at_zero: R,
}

/// Base-collector junction weak avalanche current with smoothed depletion voltage `vl`:
/// `(Itzf - Itzr - Ibcj) · AVC1 · vl · e^{-AVC2 · vl^(MC - 1)}`.
#[derive(Debug, Clone, Copy)]
struct Avalanche<R: Real> {
    coefficient: R,
    exponent_coefficient: R,
    potential: R,
    grading: R,
}

/// The parasitic substrate transistor's transport current.
#[derive(Debug, Clone, Copy)]
struct ParasiticTransport<R: Real> {
    saturation_current: R,
    emission: R,
    /// The fraction of the forward current the extrinsic junction drives.
    fraction: R,
    /// Reciprocal parasitic knee current `1 / IKP` (zero for infinite knee current).
    inverse_knee: R,
}

/// The intrinsic collector's quasi-saturating resistance.
#[derive(Debug, Clone, Copy)]
struct Collector<R: Real> {
    resistance: R,
    doping: R,
    /// Reciprocal drift saturation voltage `1 / VO` (zero for no velocity saturation).
    inverse_saturation: R,
    /// Reciprocal high-current collector factor `1 / HRCF`.
    inverse_factor: R,
}

/// Temperature-scaled depletion, transit, epitaxial, and substrate charge parameters.
#[derive(Debug, Clone, Copy)]
struct Charges<R: Real> {
    base_emitter: Depletion<R>,
    base_collector: Depletion<R>,
    /// The collector-substrate depletion, where it has a capacitance.
    collector_substrate: Option<Depletion<R>>,
    /// The zero-bias capacitances of the base-emitter, intrinsic and extrinsic
    /// base-collector and collector-substrate junctions.
    capacitances: [R; 4],
    epitaxial: R,
    forward_transit_time: R,
    base_width_coefficient: R,
    bias_coefficient: R,
    /// Transit-time voltage factor `1 / (1.44 · VTF)` (zero for infinite `VTF`).
    voltage_factor: R,
    /// Reciprocal transit-time knee current `1 / ITF` (zero for growth at all currents).
    inverse_current: R,
    reverse_transit_time: R,
    fixed_substrate: R,
}

/// Every current and charge of one evaluation, oriented as an NPN's, each computed at the
/// width of the law it belongs to, with its derivatives with respect to the variables that
/// law reads, in its order.
struct Branches<R: Real> {
    /// The base-emitter currents at the intrinsic and at the extrinsic base, of `Vbei` and
    /// `Vbex`.
    ibe: Dual<R, 1>,
    ibex: Dual<R, 1>,
    /// The forward and the reverse transport currents, of [`INTRINSIC`].
    itzf: Dual<R, 2>,
    itzr: Dual<R, 2>,
    /// The base-collector junction's current, of `Vbci`.
    ibc: Dual<R, 1>,
    /// The weak avalanche current, from the intrinsic collector into the intrinsic base, of
    /// [`INTRINSIC`], where the transistor has one.
    avalanche: Option<Dual<R, 2>>,
    /// The parasitic transistor's base-emitter current, of `Vbep`.
    ibep: Dual<R, 1>,
    /// The currents through the intrinsic collector, intrinsic base and parasitic base
    /// resistances, of [`INTRINSIC_COLLECTOR`], [`INTRINSIC_BASE`] and
    /// [`PARASITIC_BASE`].
    irci: Dual<R, 3>,
    irbi: Dual<R, 3>,
    irbp: Dual<R, 3>,
    /// The parasitic transistor's base-collector current, of `Vbcp`, and its transport
    /// current, of [`PARASITIC`].
    ibcp: Dual<R, 1>,
    iccp: Dual<R, 3>,
    charges: JunctionCharges<R>,
}

/// The charges on the intrinsic base-emitter junction, of [`INTRINSIC`], on the extrinsic
/// base-emitter, intrinsic and extrinsic base-collector junctions, of `Vbex`, `Vbci` and
/// `Vbcx`, on the parasitic transistor's base-emitter junction, of [`PARASITIC_FORWARD`],
/// and on its base-collector junction, of `Vbcp`.
struct JunctionCharges<R: Real> {
    qbe: Dual<R, 2>,
    qbex: Dual<R, 1>,
    qbc: Dual<R, 1>,
    qbcx: Dual<R, 1>,
    qbep: Dual<R, 2>,
    qbcp: Dual<R, 1>,
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
        let Self {
            qbe,
            qbex,
            qbc,
            qbcx,
            qbep,
            qbcp,
        } = self;
        let sums = derivatives;
        added
            .reading(INTRINSIC)
            .add_charge(stamp, sums, (BI, EI), qbe);
        added
            .reading([VBEX])
            .add_charge(stamp, sums, (BX, EI), qbex);
        added.reading([VBCI]).add_charge(stamp, sums, (BI, CI), qbc);
        added
            .reading([VBCX])
            .add_charge(stamp, sums, (BI, CX), qbcx);
        added
            .reading(PARASITIC_FORWARD)
            .add_charge(stamp, sums, (BX, BP), qbep);
        added
            .reading([VBCP])
            .add_charge(stamp, sums, (SI, BP), qbcp);
    }
}

/// The intrinsic transistor's state at one bias: its forward current, of `Vbei`, its
/// reverse current, of `Vbci`, and its normalized base charges, of [`INTRINSIC`].
struct TransportState<R: Real> {
    ifi: Dual<R, 1>,
    iri: Dual<R, 1>,
    q1: Dual<R, 2>,
    qb: Dual<R, 2>,
}

/// The parasitic transistor's state at one bias: its forward current and its normalized
/// base charge, of [`PARASITIC_FORWARD`], and its transport current, of [`PARASITIC`].
struct ParasiticState<R: Real> {
    ifp: Dual<R, 2>,
    qbp: Dual<R, 2>,
    iccp: Dual<R, 3>,
}

impl<R: Real> Core<R> {
    /// Evaluates `e^{v / (n · Vt)}` for junction `voltage` and `emission` coefficient.
    fn exponential<const N: usize>(&self, voltage: Dual<R, N>, emission: R) -> Dual<R, N> {
        voltage.divide(emission * self.thermal_voltage).exp()
    }

    /// Sums diode currents `I · (e^{v / (n · Vt)} - 1)` at `voltage`.
    fn diodes<const N: usize>(&self, voltage: Dual<R, N>, diodes: [Diode<R>; 2]) -> Dual<R, N> {
        diodes
            .into_iter()
            .fold(Dual::number(0.0), |sum, (current, emission)| {
                sum + (self.exponential(voltage, emission) - 1.0).scale(current)
            })
    }

    /// Evaluates all branch currents and charges at `voltages` with junction shunt `gmin`.
    #[expect(
        clippy::similar_names,
        reason = "VBIC branch names such as Vbei and Vbex differ by one letter"
    )]
    fn evaluate(&self, voltages: [R; VARIABLE_COUNT], gmin: f64) -> Branches<R> {
        let [vbei, vbex, vbci, vbcx, vbep, vbcp, vrci, vrbi, vrbp] =
            voltages.map(Dual::<R, 1>::variable);
        let charges = &self.charges;
        let qdbe = charges.base_emitter.charge(vbei);
        let qdbc = charges.base_collector.charge(vbci);
        let state = self.transport([vbei, vbci], [qdbe, qdbc]);
        // Within the intrinsic transistor's law, `Vbei` comes first and `Vbci` second.
        let itzf = state.ifi.embed([0]) / state.qb;
        let itzr = state.iri.embed([1]) / state.qb;
        let parasitic = self.parasitic([vbep, vbci, vbcp]);
        let base = &self.base;
        let breakdown = |voltage: Dual<R, 1>| {
            base.breakdown.map_or(Dual::number(0.0), |breakdown| {
                -(((-voltage).plus(-breakdown.voltage))
                    .divide(breakdown.slope)
                    .exp()
                    - Dual::constant(breakdown.at_zero))
                .scale(breakdown.current)
            })
        };
        let ibe = (self.diodes(vbei, base.emitter) + breakdown(vbei)).scale(base.fraction);
        let ibex =
            (self.diodes(vbex, base.emitter) + breakdown(vbex)).scale(-(base.fraction - 1.0));
        let ibc = self.diodes(vbci, base.collector);
        let avalanche = base.avalanche.map(|avalanche| {
            let across = -(vbci.plus(-avalanche.potential));
            let depleted = ((across * across + 0.01).sqrt() + across) * 0.5;
            let factor = depleted.scale(avalanche.coefficient)
                * (-depleted
                    .pow(avalanche.grading - 1.0)
                    .scale(avalanche.exponent_coefficient))
                .exp();
            (itzf - itzr - ibc.embed([1])) * factor.embed([1])
        });
        let ibep = self.diodes(vbep, base.parasitic_emitter);
        let (irci, kbci, kbcx) = self.collector([vrci, vbci, vbcx]);
        let [intrinsic, parasitic_resistance] = self.base_resistances;
        // Within each base-resistance law, branch voltage comes first, followed by
        // base-charge junctions.
        let irbi = (vrbi.embed([0]) * state.qb.embed([1, 2])).divide(intrinsic);
        let irbp = (vrbp.embed([0]) * parasitic.qbp.embed([1, 2])).divide(parasitic_resistance);
        let ibcp = self.diodes(vbcp, base.parasitic_collector);
        let charges = self.charges(
            &state,
            ([qdbe, qdbc], [vbex, vbci, vbep, vbcp]),
            (parasitic.ifp, kbci, kbcx),
        );
        Branches {
            ibe: ibe + vbei * gmin,
            ibex: ibex + vbex * gmin,
            itzf,
            itzr,
            ibc: ibc + vbci * gmin,
            avalanche,
            ibep: ibep + vbep * gmin,
            irci: irci + vrci.embed([0]) * gmin + vbci.embed([1]) * gmin + vbcx.embed([2]) * gmin,
            irbi,
            irbp,
            ibcp: ibcp + vbcp * gmin,
            iccp: parasitic.iccp,
            charges,
        }
    }

    /// Computes intrinsic forward and reverse currents and normalized base charges from
    /// `[vbei, vbci]` and depletion charges `[qdbe, qdbc]`.
    #[expect(
        clippy::similar_names,
        reason = "VBIC branch names such as Vbei and Vbex differ by one letter"
    )]
    fn transport(
        &self,
        [vbei, vbci]: [Dual<R, 1>; 2],
        [qdbe, qdbc]: [Dual<R, 1>; 2],
    ) -> TransportState<R> {
        let transport = &self.transport;
        let [forward, reverse] = transport.emission;
        let ifi = (self.exponential(vbei, forward) - 1.0).scale(transport.saturation_current);
        let iri = (self.exponential(vbci, reverse) - 1.0)
            .scale(transport.saturation_current * transport.reverse_ratio);
        let [inverse_forward, inverse_reverse] = transport.inverse_early;
        let q1z =
            qdbe.embed([0]).scale(inverse_reverse) + qdbc.embed([1]).scale(inverse_forward) + 1.0;
        // VBIC smooths q1 above zero so the base charge never vanishes.
        let shifted = q1z - 1e-4;
        let q1 = ((shifted * shifted + 1e-8).sqrt() + shifted) * 0.5 + 1e-4;
        let [knee_forward, knee_reverse] = transport.inverse_knee;
        let q2 = ifi.embed([0]).scale(knee_forward) + iri.embed([1]).scale(knee_reverse);
        let rolloff = transport.rolloff;
        let qb = match transport.form {
            BaseCharge::Vbic => (q1 + (q1.pow(rolloff.recip()) + q2 * 4.0).pow(rolloff)) * 0.5,
            BaseCharge::GummelPoon => q1 * ((q2 * 4.0 + 1.0).pow(rolloff) + 1.0) * 0.5,
        };
        TransportState { ifi, iri, q1, qb }
    }

    /// Evaluates parasitic forward current, normalized base charge, and transport current
    /// at `[vbep, vbci, vbcp]`.
    #[expect(
        clippy::similar_names,
        reason = "VBIC branch names such as Vbei and Vbex differ by one letter"
    )]
    fn parasitic(&self, [vbep, vbci, vbcp]: [Dual<R, 1>; 3]) -> ParasiticState<R> {
        let Some(parasitic) = &self.parasitic else {
            return ParasiticState {
                ifp: Dual::number(0.0),
                qbp: Dual::number(1.0),
                iccp: Dual::number(0.0),
            };
        };
        let emission = parasitic.emission;
        // Within its forward current's law, `Vbep` comes first and `Vbci` second; its
        // transport current's adds `Vbcp` third.
        let ifp = (self
            .exponential(vbep, emission)
            .scale(parasitic.fraction)
            .embed([0])
            + self
                .exponential(vbci, emission)
                .scale(-(parasitic.fraction - 1.0))
                .embed([1])
            - 1.0)
            .scale(parasitic.saturation_current);
        let q2p = ifp.scale(parasitic.inverse_knee);
        let qbp = ((q2p * 4.0 + 1.0).sqrt() + 1.0) * 0.5;
        let irp = (self.exponential(vbcp, emission) - 1.0).scale(parasitic.saturation_current);
        let iccp = (ifp.embed([0, 1]) - irp.embed([2])) / qbp.embed([0, 1]);
        ParasiticState { ifp, qbp, iccp }
    }

    /// Evaluates Kull quasi-saturating intrinsic collector current with velocity
    /// saturation at `[vrci, vbci, vbcx]`. Returns collector current along with epitaxial
    /// normalized charges at both ends.
    #[expect(
        clippy::similar_names,
        reason = "VBIC branch names such as Vbei and Vbex differ by one letter"
    )]
    fn collector(
        &self,
        [vrci, vbci, vbcx]: [Dual<R, 1>; 3],
    ) -> (Dual<R, 3>, Dual<R, 1>, Dual<R, 1>) {
        let collector = &self.collector;
        let one = R::constant(1.0);
        let kbci = (self.exponential(vbci, one).scale(collector.doping) + 1.0).sqrt();
        let kbcx = (self.exponential(vbcx, one).scale(collector.doping) + 1.0).sqrt();
        // Within the law, the drop across the resistance comes first, then `Vbci` and
        // `Vbcx`.
        let ratio = (kbci + 1.0).embed([1]) / (kbcx + 1.0).embed([2]);
        let across = vrci.embed([0]);
        let iohm = (across
            + (kbci.embed([1]) - kbcx.embed([2]) - ratio.ln()).scale(self.thermal_voltage))
        .divide(collector.resistance);
        let velocity = collector.inverse_saturation;
        let derf = iohm.scale(velocity * collector.resistance)
            / ((across * across + 0.01)
                .sqrt()
                .scale(velocity * collector.inverse_factor * 0.5)
                + 1.0);
        (iohm / (derf * derf + 1.0).sqrt(), kbci, kbcx)
    }

    /// Evaluates all six junction charges from intrinsic state, depletion charges,
    /// voltages, parasitic forward current, and epitaxial charges.
    #[expect(
        clippy::similar_names,
        reason = "VBIC branch names such as Vbei and Vbex differ by one letter"
    )]
    fn charges(
        &self,
        state: &TransportState<R>,
        ([qdbe, qdbc], [vbex, vbci, vbep, vbcp]): ([Dual<R, 1>; 2], [Dual<R, 1>; 4]),
        (ifp, kbci, kbcx): (Dual<R, 2>, Dual<R, 1>, Dual<R, 1>),
    ) -> JunctionCharges<R> {
        let charges = &self.charges;
        let fraction = self.base.fraction;
        let [emitter, collector, extrinsic, substrate] = charges.capacitances;
        let TransportState { ifi, iri, q1, qb } = *state;
        let early =
            (q1.scale(charges.base_width_coefficient) + 1.0).scale(charges.forward_transit_time);
        // The transit time grows with the current only where the forward current flows.
        let transit = if ifi.value.value() > 0.0 {
            let relative = ifi.scale(charges.inverse_current);
            let share = relative / (relative + 1.0);
            let low = if charges.inverse_current.value() > 0.0 {
                0.0
            } else {
                1.0
            };
            // Within the intrinsic transistor's law, `Vbei` comes first and `Vbci` second.
            let growth = vbci
                .scale(charges.voltage_factor)
                .exp()
                .scale(charges.bias_coefficient)
                .embed([1])
                * (share * share + low).embed([0]);
            early * (growth + 1.0)
        } else {
            early
        };
        let qdbcp = charges
            .collector_substrate
            .map_or(Dual::number(0.0), |depletion| depletion.charge(vbcp));
        JunctionCharges {
            qbe: qdbe.scale(emitter * fraction).embed([0]) + transit * ifi.embed([0]) / qb,
            qbex: charges
                .base_emitter
                .charge(vbex)
                .scale(emitter * -(fraction - 1.0)),
            qbc: qdbc.scale(collector)
                + iri.scale(charges.reverse_transit_time)
                + kbci.scale(charges.epitaxial),
            qbcx: kbcx.scale(charges.epitaxial),
            // Within the parasitic forward current's law, `Vbep` comes first.
            qbep: charges
                .base_collector
                .charge(vbep)
                .scale(extrinsic)
                .embed([0])
                + ifp.scale(charges.reverse_transit_time),
            qbcp: qdbcp.scale(substrate) + vbcp.scale(charges.fixed_substrate),
        }
    }
}

/// Temperature ratio, temperature difference, and VBIC thermal voltage for scaling
/// parameters from nominal temperature.
#[derive(Debug, Clone, Copy)]
struct Scaling<R: Real> {
    ratio: R,
    difference: R,
    thermal_voltage: R,
}

impl<R: Real> Scaling<R> {
    /// Constructs temperature scaling factors from `temperatures` using VBIC
    /// thermal-voltage constants.
    fn new(temperatures: Temperatures<R>) -> Self {
        Self {
            ratio: temperatures.ratio(),
            difference: temperatures.difference(),
            thermal_voltage: temperatures.actual * VBIC_BOLTZMANN / VBIC_CHARGE,
        }
    }

    /// Scales `value` by `(T / Tn)^exponent`.
    fn power(self, value: R, exponent: R) -> R {
        value * self.ratio.pow(exponent)
    }

    /// Scales `value` linearly by `1 + coefficient · ΔT`.
    fn linear(self, value: R, coefficient: R) -> R {
        value * (self.difference * coefficient + 1.0)
    }

    /// Computes the Arrhenius factor `(T / Tn)^exponent · e^{-energy · (1 - T / Tn) / Vt}`.
    fn arrhenius(self, exponent: R, energy: R) -> R {
        self.ratio.pow(exponent) * (-energy * -(self.ratio - 1.0) / self.thermal_voltage).exp()
    }

    /// Scales saturation `current` by temperature `exponent`, activation `energy`, and
    /// `emission`.
    fn activated(self, current: R, (exponent, energy): (R, R), emission: R) -> R {
        current * self.arrhenius(exponent, energy).pow(emission.recip())
    }

    /// Validates and temperature-scales a diode saturation current and emission
    /// coefficient.
    fn diode(
        self,
        (current_name, current): (&str, R),
        (emission_name, emission): (&str, R),
        scaled: (R, R),
    ) -> Result<Diode<R>, ParameterError> {
        let current = ParameterError::non_negative(current_name, current)?;
        let emission = ParameterError::positive(emission_name, emission)?;
        Ok((self.activated(current, scaled, emission), emission))
    }

    /// Maps a junction built-in `potential` to device temperature via VBIC's
    /// activation-energy law.
    fn potential(self, potential: R, energy: R) -> R {
        let Self {
            ratio,
            thermal_voltage,
            ..
        } = self;
        let half = potential * 0.5 * ratio / thermal_voltage;
        let intrinsic = thermal_voltage / ratio * 2.0 * (half.exp() - (-half).exp()).ln();
        let shifted =
            intrinsic * ratio - thermal_voltage * 3.0 * ratio.ln() - energy * (ratio - 1.0);
        let correction = (((-shifted / thermal_voltage).exp() * 4.0 + 1.0).sqrt() + 1.0) * 0.5;
        shifted + thermal_voltage * 2.0 * correction.ln()
    }
}

/// Checks that an emission coefficient `value`, scaled to the temperature, is still
/// positive.
fn still_positive<R: Real>(name: &str, value: R, temperature: R) -> Result<R, ParameterError> {
    if value.value() > 0.0 {
        Ok(value)
    } else {
        Err(ParameterError(format!(
            "the temperature coefficients scale the {name} to zero at {:?} K",
            temperature.value()
        )))
    }
}

impl<R: Real> VbicParameters<R> {
    /// Validates and scales intrinsic transport parameters using `scaling` and
    /// `temperature`.
    fn transport(
        &self,
        scaling: Scaling<R>,
        temperature: R,
    ) -> Result<Transport<R>, ParameterError> {
        let positive = ParameterError::positive;
        let saturation = positive("saturation_current", self.saturation_current)?;
        let forward = positive(
            "forward_emission_coefficient",
            self.forward_emission_coefficient,
        )?;
        let reverse = positive(
            "reverse_emission_coefficient",
            self.reverse_emission_coefficient,
        )?;
        let ratio = ParameterError::non_negative(
            "reverse_saturation_current_ratio",
            self.reverse_saturation_current_ratio,
        )?;
        let coefficient = self.emission_coefficient_temperature_coefficient;
        let knee = self
            .forward_knee_current
            .map(|knee| scaling.power(knee, self.forward_knee_current_temperature_exponent));
        Ok(Transport {
            saturation_current: scaling.activated(
                saturation,
                (
                    self.saturation_current_temperature_exponent,
                    self.activation_energy,
                ),
                forward,
            ),
            reverse_ratio: scaling.activated(
                ratio,
                (
                    self.reverse_saturation_current_temperature_exponent,
                    self.reverse_activation_energy_shift,
                ),
                reverse,
            ),
            emission: [
                still_positive(
                    "forward_emission_coefficient",
                    scaling.linear(forward, coefficient),
                    temperature,
                )?,
                still_positive(
                    "reverse_emission_coefficient",
                    scaling.linear(reverse, coefficient),
                    temperature,
                )?,
            ],
            inverse_early: [
                inverse("forward_early_voltage", self.forward_early_voltage)?,
                inverse("reverse_early_voltage", self.reverse_early_voltage)?,
            ],
            inverse_knee: [
                inverse("forward_knee_current", knee)?,
                inverse("reverse_knee_current", self.reverse_knee_current)?,
            ],
            form: self.base_charge,
            rolloff: positive("knee_rolloff", self.knee_rolloff)?,
        })
    }

    /// Validates and scales base-current, breakdown, and avalanche parameters using
    /// `scaling`, `temperature`, and base-collector `potential`.
    fn base_currents(
        &self,
        scaling: Scaling<R>,
        temperature: R,
        potential: R,
    ) -> Result<BaseCurrents<R>, ParameterError> {
        let ideal = self.ideal_base_current_temperature_exponent;
        let nonideal = self.nonideal_base_current_temperature_exponent;
        let diode = |current, emission, scaled| scaling.diode(current, emission, scaled);
        let collector_energies = (
            (ideal, self.ideal_base_collector_activation_energy),
            (nonideal, self.nonideal_base_collector_activation_energy),
        );
        let (ideal_collector, nonideal_collector) = (
            (
                "ideal_base_collector_emission_coefficient",
                self.ideal_base_collector_emission_coefficient,
            ),
            (
                "nonideal_base_collector_emission_coefficient",
                self.nonideal_base_collector_emission_coefficient,
            ),
        );
        Ok(BaseCurrents {
            emitter: [
                diode(
                    (
                        "ideal_base_emitter_current",
                        self.ideal_base_emitter_current,
                    ),
                    (
                        "ideal_base_emitter_emission_coefficient",
                        self.ideal_base_emitter_emission_coefficient,
                    ),
                    (ideal, self.ideal_base_emitter_activation_energy),
                )?,
                diode(
                    (
                        "nonideal_base_emitter_current",
                        self.nonideal_base_emitter_current,
                    ),
                    (
                        "nonideal_base_emitter_emission_coefficient",
                        self.nonideal_base_emitter_emission_coefficient,
                    ),
                    (nonideal, self.nonideal_base_emitter_activation_energy),
                )?,
            ],
            fraction: fraction(
                "intrinsic_base_emitter_fraction",
                self.intrinsic_base_emitter_fraction,
            )?,
            breakdown: self.breakdown(scaling, temperature)?,
            collector: [
                diode(
                    (
                        "ideal_base_collector_current",
                        self.ideal_base_collector_current,
                    ),
                    ideal_collector,
                    collector_energies.0,
                )?,
                diode(
                    (
                        "nonideal_base_collector_current",
                        self.nonideal_base_collector_current,
                    ),
                    nonideal_collector,
                    collector_energies.1,
                )?,
            ],
            // ngspice scales the parasitic base-emitter currents as the base-collector ones.
            parasitic_emitter: [
                diode(
                    (
                        "parasitic_ideal_base_emitter_current",
                        self.parasitic_ideal_base_emitter_current,
                    ),
                    ideal_collector,
                    collector_energies.0,
                )?,
                diode(
                    (
                        "parasitic_nonideal_base_emitter_current",
                        self.parasitic_nonideal_base_emitter_current,
                    ),
                    nonideal_collector,
                    collector_energies.1,
                )?,
            ],
            parasitic_collector: self.parasitic_collector(scaling)?,
            avalanche: self.avalanche(scaling, potential)?,
        })
    }

    /// Validates and scales parasitic base-collector diodes across the collector-substrate
    /// junction.
    fn parasitic_collector(&self, scaling: Scaling<R>) -> Result<[Diode<R>; 2], ParameterError> {
        let diode = |current, emission, scaled| scaling.diode(current, emission, scaled);
        Ok([
            diode(
                (
                    "parasitic_ideal_base_collector_current",
                    self.parasitic_ideal_base_collector_current,
                ),
                (
                    "parasitic_ideal_base_collector_emission_coefficient",
                    self.parasitic_ideal_base_collector_emission_coefficient,
                ),
                (
                    self.ideal_base_current_temperature_exponent,
                    self.ideal_collector_substrate_activation_energy,
                ),
            )?,
            diode(
                (
                    "parasitic_nonideal_base_collector_current",
                    self.parasitic_nonideal_base_collector_current,
                ),
                (
                    "parasitic_nonideal_base_collector_emission_coefficient",
                    self.parasitic_nonideal_base_collector_emission_coefficient,
                ),
                (
                    self.nonideal_base_current_temperature_exponent,
                    self.nonideal_collector_substrate_activation_energy,
                ),
            )?,
        ])
    }

    /// Validates and scales optional base-emitter breakdown parameters.
    fn breakdown(
        &self,
        scaling: Scaling<R>,
        temperature: R,
    ) -> Result<Option<Breakdown<R>>, ParameterError> {
        let emission = ParameterError::positive(
            "base_emitter_breakdown_emission_coefficient",
            self.base_emitter_breakdown_emission_coefficient,
        )?;
        let current = ParameterError::non_negative(
            "base_emitter_breakdown_current",
            self.base_emitter_breakdown_current,
        )?;
        let Some(voltage) = self.base_emitter_breakdown_voltage else {
            return Ok(None);
        };
        let voltage = ParameterError::positive("base_emitter_breakdown_voltage", voltage)?;
        let voltage = voltage
            * (scaling.difference
                * (self.breakdown_voltage_temperature_coefficient
                    + scaling.difference
                        * self.breakdown_voltage_quadratic_temperature_coefficient)
                + 1.0);
        let emission = still_positive(
            "base_emitter_breakdown_emission_coefficient",
            scaling.linear(emission, self.breakdown_emission_temperature_coefficient),
            temperature,
        )?;
        let slope = emission * scaling.thermal_voltage;
        Ok(Some(Breakdown {
            voltage,
            slope,
            current,
            at_zero: (-voltage / slope).exp(),
        }))
    }

    /// Validates and scales optional base-collector weak-avalanche parameters.
    fn avalanche(
        &self,
        scaling: Scaling<R>,
        potential: R,
    ) -> Result<Option<Avalanche<R>>, ParameterError> {
        let coefficient =
            ParameterError::non_negative("avalanche_coefficient", self.avalanche_coefficient)?;
        Ok((coefficient.value() > 0.0).then(|| Avalanche {
            coefficient,
            exponent_coefficient: scaling.linear(
                self.avalanche_exponent_coefficient,
                self.avalanche_temperature_coefficient,
            ),
            potential,
            grading: self.base_collector_grading,
        }))
    }

    /// Validates and scales optional parasitic substrate-transistor transport parameters.
    fn parasitic(
        &self,
        scaling: Scaling<R>,
    ) -> Result<Option<ParasiticTransport<R>>, ParameterError> {
        let saturation = ParameterError::non_negative(
            "parasitic_saturation_current",
            self.parasitic_saturation_current,
        )?;
        let emission = ParameterError::positive(
            "parasitic_forward_emission_coefficient",
            self.parasitic_forward_emission_coefficient,
        )?;
        let share = fraction(
            "parasitic_extrinsic_fraction",
            self.parasitic_extrinsic_fraction,
        )?;
        let inverse_knee = inverse("parasitic_knee_current", self.parasitic_knee_current)?;
        Ok((saturation.value() > 0.0).then(|| ParasiticTransport {
            saturation_current: scaling.activated(
                saturation,
                (
                    self.saturation_current_temperature_exponent,
                    self.parasitic_activation_energy,
                ),
                emission,
            ),
            emission,
            fraction: share,
            inverse_knee,
        }))
    }

    /// Validates and scales extrinsic terminal series resistances and intrinsic collector
    /// and base resistances.
    fn resistances(&self, scaling: Scaling<R>) -> Result<([R; 4], [R; 3]), ParameterError> {
        let series = |name: &str, value: R, exponent: R| {
            ParameterError::non_negative(name, value).map(|value| scaling.power(value, exponent))
        };
        let inner = |name: &str, value: R, exponent: R| {
            ParameterError::positive(name, value).map(|value| scaling.power(value, exponent))
        };
        Ok((
            [
                series(
                    "extrinsic_collector_resistance",
                    self.extrinsic_collector_resistance,
                    self.extrinsic_collector_resistance_temperature_exponent,
                )?,
                series(
                    "extrinsic_base_resistance",
                    self.extrinsic_base_resistance,
                    self.extrinsic_base_resistance_temperature_exponent,
                )?,
                series(
                    "emitter_resistance",
                    self.emitter_resistance,
                    self.emitter_resistance_temperature_exponent,
                )?,
                series(
                    "substrate_resistance",
                    self.substrate_resistance,
                    self.substrate_resistance_temperature_exponent,
                )?,
            ],
            [
                inner(
                    "intrinsic_collector_resistance",
                    self.intrinsic_collector_resistance,
                    self.intrinsic_collector_resistance_temperature_exponent,
                )?,
                inner(
                    "intrinsic_base_resistance",
                    self.intrinsic_base_resistance,
                    self.intrinsic_base_resistance_temperature_exponent,
                )?,
                inner(
                    "parasitic_base_resistance",
                    self.parasitic_base_resistance,
                    self.parasitic_base_resistance_temperature_exponent,
                )?,
            ],
        ))
    }

    /// Validates and scales intrinsic collector quasi-saturation parameters for
    /// `resistance`.
    fn collector(
        &self,
        scaling: Scaling<R>,
        resistance: R,
    ) -> Result<Collector<R>, ParameterError> {
        let doping = ParameterError::non_negative(
            "epitaxial_doping_parameter",
            self.epitaxial_doping_parameter,
        )?;
        let saturation = self.epitaxial_saturation_voltage.map(|voltage| {
            scaling.power(
                voltage,
                self.epitaxial_saturation_voltage_temperature_exponent,
            )
        });
        Ok(Collector {
            resistance,
            doping: doping
                * scaling.arrhenius(
                    self.saturation_current_temperature_exponent,
                    self.activation_energy,
                ),
            inverse_saturation: inverse("epitaxial_saturation_voltage", saturation)?,
            inverse_factor: ParameterError::positive(
                "high_current_collector_factor",
                self.high_current_collector_factor,
            )?
            .recip(),
        })
    }

    /// Validates and scales junction depletion, transit, epitaxial, and substrate charge
    /// parameters.
    fn charges(&self, scaling: Scaling<R>) -> Result<Charges<R>, ParameterError> {
        let non_negative = ParameterError::non_negative;
        let limit = below_one("depletion_coefficient", self.depletion_coefficient)?;
        // A junction's depletion at the temperature, its potential moved there, and the
        // factor that scales its capacitances with it.
        let junction = |names: (&str, &str), (potential, grading, smoothing): (R, R, R), energy| {
            let nominal = ParameterError::positive(names.0, potential)?;
            let grading = below_one(names.1, grading)?;
            let potential = scaling.potential(nominal, energy);
            let depletion = Depletion {
                potential,
                grading,
                smoothing,
                limit,
                reach: None,
            };
            Ok::<_, ParameterError>((depletion, (nominal / potential).pow(grading)))
        };
        let (base_emitter, emitter_factor) = junction(
            ("base_emitter_potential", "base_emitter_grading"),
            (
                self.base_emitter_potential,
                self.base_emitter_grading,
                self.base_emitter_smoothing,
            ),
            self.ideal_base_emitter_activation_energy,
        )?;
        let (mut base_collector, collector_factor) = junction(
            ("base_collector_potential", "base_collector_grading"),
            (
                self.base_collector_potential,
                self.base_collector_grading,
                self.base_collector_smoothing,
            ),
            self.ideal_base_collector_activation_energy,
        )?;
        let (collector_substrate, substrate_factor) = junction(
            (
                "collector_substrate_potential",
                "collector_substrate_grading",
            ),
            (
                self.collector_substrate_potential,
                self.collector_substrate_grading,
                self.collector_substrate_smoothing,
            ),
            self.ideal_collector_substrate_activation_energy,
        )?;
        let reach = non_negative("reach_through_voltage", self.reach_through_voltage)?;
        base_collector.reach =
            (reach.value() > 0.0).then_some((reach, self.reach_through_smoothing));
        let capacitance =
            |name: &str, value: R, factor: R| non_negative(name, value).map(|value| value * factor);
        let substrate = capacitance(
            "collector_substrate_capacitance",
            self.collector_substrate_capacitance,
            substrate_factor,
        )?;
        let current = non_negative("transit_time_current", self.transit_time_current)?;
        Ok(Charges {
            base_emitter,
            base_collector,
            collector_substrate: (substrate.value() > 0.0).then_some(collector_substrate),
            capacitances: [
                capacitance(
                    "base_emitter_capacitance",
                    self.base_emitter_capacitance,
                    emitter_factor,
                )?,
                capacitance(
                    "base_collector_capacitance",
                    self.base_collector_capacitance,
                    collector_factor,
                )?,
                capacitance(
                    "extrinsic_base_collector_capacitance",
                    self.extrinsic_base_collector_capacitance,
                    collector_factor,
                )?,
                substrate,
            ],
            epitaxial: non_negative("epitaxial_charge", self.epitaxial_charge)?,
            forward_transit_time: non_negative("forward_transit_time", self.forward_transit_time)?,
            base_width_coefficient: self.transit_time_base_width_coefficient,
            bias_coefficient: self.transit_time_bias_coefficient,
            voltage_factor: inverse("transit_time_voltage", self.transit_time_voltage)? / 1.44,
            inverse_current: if current.value() > 0.0 {
                current.recip()
            } else {
                current
            },
            reverse_transit_time: non_negative("reverse_transit_time", self.reverse_transit_time)?,
            fixed_substrate: non_negative(
                "collector_substrate_fixed_capacitance",
                self.collector_substrate_fixed_capacitance,
            )?,
        })
    }
}

impl<R: Real> VbicParameters<R> {
    /// Validates and scales all core currents, charges, and extrinsic series resistances
    /// for `temperatures`.
    fn core(&self, temperatures: Temperatures<R>) -> Result<(Core<R>, [R; 4]), ParameterError> {
        let scaling = Scaling::new(temperatures);
        let temperature = temperatures.actual;
        let charges = self.charges(scaling)?;
        let (series, [intrinsic_collector, intrinsic_base, parasitic_base]) =
            self.resistances(scaling)?;
        let core = Core {
            thermal_voltage: scaling.thermal_voltage,
            transport: self.transport(scaling, temperature)?,
            base: self.base_currents(scaling, temperature, charges.base_collector.potential)?,
            parasitic: self.parasitic(scaling)?,
            collector: self.collector(scaling, intrinsic_collector)?,
            base_resistances: [intrinsic_base, parasitic_base],
            charges,
        };
        Ok((core, series))
    }
}

/// Ngspice VBIC excess-phase network with current unknowns `x1` and `x2`. The filter
/// enforces `TD · dx1/dt + x2 - Itzf = 0` and `TD / 3 · dx2/dt + x2 - x1 = 0`. Delayed
/// current `x2` delivers `3 / (s² · TD² + 3 · s · TD + 3)` of `Itzf` from intrinsic
/// collector to emitter. At DC both unknowns equal forward transport current, oriented
/// with device polarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExcessPhase {
    /// Local unknowns `x1` and `x2`.
    unknowns: [usize; 2],
    /// Local entries of the `x1` equation at each stamped node.
    driven: [usize; STAMPED],
    /// Local entries of the `x1` and `x2` equations over `[x1, x2]`.
    own: [[usize; 2]; 2],
    /// Local entries of the intrinsic collector and emitter equations at `x2`.
    delivered: [usize; 2],
}

impl ExcessPhase {
    /// Allocates excess-phase filter unknowns and matrix entries driven by `stamp`.
    fn new(layout: &mut Layout, stamp: &Nodes<STAMPED>) -> Self {
        let unknowns = [
            layout.internal("transport_filter", Nature::Current),
            layout.internal("delayed_transport", Nature::Current),
        ];
        let [first, second] = unknowns;
        Self {
            unknowns,
            driven: std::array::from_fn(|index| layout.entry(first, stamp.node(index))),
            own: [first, second].map(|row| [first, second].map(|column| layout.entry(row, column))),
            delivered: [CI, EI].map(|index| layout.entry(stamp.node(index), second)),
        }
    }

    /// Returns the delayed forward transport current `x2` from `at`.
    fn delayed<R: Real>(&self, at: &Local<'_>) -> R {
        let [_, second] = self.unknowns;
        at.unknown(second)
    }

    /// Adds the network's equations, of `delay`, driven by the forward transport current
    /// `itzf` of the transistor `added` adds for, a number of the intrinsic transistor's
    /// law, and the delayed current from the intrinsic collector to the intrinsic emitter.
    fn add<R: Real>(
        &self,
        added: &Oriented<'_, STAMPED, VARIABLE_COUNT>,
        (itzf, delay): (Dual<R, 2>, R),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let intrinsic = added.reading(INTRINSIC);
        let [first, second] = self.unknowns;
        let [filtered, delayed] = [first, second].map(|unknown| at.unknown::<R>(unknown));
        let [[first_first, first_second], [second_first, second_second]] = self.own;
        let one = R::constant(1.0);
        stamp.add_residual(first, delayed - intrinsic.extend(itzf) * added.sign);
        stamp.add_charge(first, delay * filtered);
        stamp.add_capacitance(first_first, delay);
        stamp.add_conductance(first_second, one);
        intrinsic.add_conductances(stamp, &self.driven, itzf.gradient.map(|slope| -slope));
        stamp.add_residual(second, delayed - filtered);
        stamp.add_charge(second, delay * delayed / 3.0);
        stamp.add_conductance(second_second, one);
        stamp.add_conductance(second_first, -one);
        stamp.add_capacitance(second_second, delay / 3.0);
        stamp.add_residual(added.nodes.node(CI), delayed);
        stamp.add_residual(added.nodes.node(EI), -delayed);
        let [at_collector, at_emitter] = self.delivered;
        stamp.add_conductance(at_collector, one);
        stamp.add_conductance(at_emitter, -one);
    }
}

impl Kernel for Vbic {
    const NAME: &'static str = "vbic";
    type Parameters<R: Real> = VbicParameters<R>;
    type Shape = VbicShape;
    type Prepared<R: Real> = PreparedVbic<R>;
    type Memory = VbicMemory;

    fn parameter_kinds() -> &'static [(&'static str, ParameterKind)] {
        VbicParameters::<f64>::PARAMETER_KINDS
    }

    fn interface<R: Real>(_: &VbicParameters<R>) -> Interface {
        Interface::new(
            &["collector", "base", "emitter", "substrate"],
            &[
                Branch::new(BASE, EMITTER, BranchKind::Conductive),
                Branch::new(BASE, COLLECTOR, BranchKind::Conductive),
                Branch::new(COLLECTOR, EMITTER, BranchKind::Conductive),
                Branch::new(SUBSTRATE, COLLECTOR, BranchKind::Conductive),
            ],
        )
    }

    fn prepare<R: Real>(
        parameters: &VbicParameters<R>,
        temperature: R,
    ) -> Result<PreparedVbic<R>, ParameterError> {
        let temperatures = Temperatures::of_junction(temperature, parameters.nominal_temperature)?;
        let (core, series) = parameters.core(temperatures)?;
        let non_negative = ParameterError::non_negative;
        let flicker_coefficient =
            non_negative("flicker_coefficient", parameters.flicker_coefficient)?;
        let flicker_exponent =
            ParameterError::positive("flicker_exponent", parameters.flicker_exponent)?;
        let flicker_frequency_exponent = ParameterError::positive(
            "flicker_frequency_exponent",
            parameters.flicker_frequency_exponent,
        )?;
        let slope = crate::thermal_voltage(temperatures.actual.value());
        Ok(PreparedVbic {
            sign: parameters.polarity.sign(),
            series,
            core,
            overlap: [
                non_negative(
                    "base_emitter_overlap_capacitance",
                    parameters.base_emitter_overlap_capacitance,
                )?,
                non_negative(
                    "base_collector_overlap_capacitance",
                    parameters.base_collector_overlap_capacitance,
                )?,
            ],
            delay: non_negative("excess_phase_delay", parameters.excess_phase_delay)?,
            slope,
            critical_voltage: critical_voltage(parameters.saturation_current.value(), slope),
            temperature: temperatures.actual,
            flicker: Flicker {
                coefficient: flicker_coefficient,
                power: flicker_exponent,
                exponent: flicker_frequency_exponent,
            },
        })
    }

    fn layout<R: Real>(
        _: &VbicParameters<R>,
        prepared: &PreparedVbic<R>,
        layout: &mut Layout,
    ) -> Result<VbicShape, ParameterError> {
        let [collector, base, emitter, substrate] = prepared.series;
        let (collector_series, cx) = Series::behind(
            layout,
            COLLECTOR,
            collector,
            ("collector", "collector_current"),
        );
        let (base_series, bx) = Series::behind(layout, BASE, base, ("base", "base_current"));
        let (emitter_series, ei) =
            Series::behind(layout, EMITTER, emitter, ("emitter", "emitter_current"));
        let (substrate_series, si) = Series::behind(
            layout,
            SUBSTRATE,
            substrate,
            ("substrate", "substrate_current"),
        );
        let ci = layout.internal("intrinsic_collector", Nature::Voltage);
        let bp = layout.internal("parasitic_base", Nature::Voltage);
        let bi = layout.internal("intrinsic_base", Nature::Voltage);
        let stamp = Nodes::new(layout, [cx, ci, bx, bi, ei, bp, si]);
        let [emitter_overlap, collector_overlap] = prepared.overlap;
        let overlap = [
            (emitter_overlap.value() > 0.0).then(|| Pair::new(layout, BASE, EMITTER)),
            (collector_overlap.value() > 0.0).then(|| Pair::new(layout, BASE, COLLECTOR)),
        ];
        let excess = (prepared.delay.value() > 0.0).then(|| ExcessPhase::new(layout, &stamp));
        Ok(VbicShape {
            stamp,
            series: [
                collector_series,
                base_series,
                emitter_series,
                substrate_series,
            ],
            overlap,
            excess,
        })
    }

    fn evaluate<R: Real>(
        shape: &VbicShape,
        prepared: &PreparedVbic<R>,
        memory: &mut VbicMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        for (series, resistance) in shape.series.iter().zip(prepared.series) {
            series.add(series.value_of(resistance), at, stamp);
        }
        for (pair, capacitance) in shape.overlap.iter().zip(prepared.overlap) {
            if let Some(pair) = pair {
                pair.add_charge(stamp, capacitance * pair.across::<R>(at, pair.voltage(at)));
                pair.add_capacitance(stamp, capacitance);
            }
        }
        let bias = Bias::limit_junctions(
            &shape.stamp,
            (prepared.sign, VARIABLES),
            &mut memory.last,
            (prepared.slope, prepared.critical_voltage),
            at,
            stamp,
        );
        let [base_emitter, _, base_collector, ..] = bias.voltages;
        memory.variables = Bipolar::variables(base_emitter, base_collector);
        let added = Oriented::new(&shape.stamp, prepared.sign, &bias, VARIABLES);
        let branches = prepared.core.evaluate(added.across(at), at.gmin);
        let intrinsic = added.reading(INTRINSIC);
        let Branches {
            ibe,
            ibex,
            itzf,
            itzr,
            ibc,
            avalanche,
            ibep,
            irci,
            irbi,
            irbp,
            ibcp,
            iccp,
            charges,
        } = branches;
        // Every derivative among the stamped nodes is summed here, and written once.
        let mut derivatives = Derivatives::new();
        let sums = &mut derivatives;
        added
            .reading([VBEI])
            .add_current(stamp, sums, (BI, EI), ibe);
        added
            .reading([VBEX])
            .add_current(stamp, sums, (BX, EI), ibex);
        match &shape.excess {
            Some(excess) => {
                intrinsic.add_current(stamp, sums, (CI, EI), -itzr);
                excess.add(&added, (itzf, prepared.delay), at, stamp);
            }
            None => intrinsic.add_current(stamp, sums, (CI, EI), itzf - itzr),
        }
        added
            .reading([VBCI])
            .add_current(stamp, sums, (BI, CI), ibc);
        if let Some(avalanche) = avalanche {
            intrinsic.add_current(stamp, sums, (CI, BI), avalanche);
        }
        added
            .reading([VBEP])
            .add_current(stamp, sums, (BX, BP), ibep);
        added
            .reading(INTRINSIC_COLLECTOR)
            .add_current(stamp, sums, (CX, CI), irci);
        added
            .reading(INTRINSIC_BASE)
            .add_current(stamp, sums, (BX, BI), irbi);
        added
            .reading(PARASITIC_BASE)
            .add_current(stamp, sums, (BP, CX), irbp);
        added
            .reading([VBCP])
            .add_current(stamp, sums, (SI, BP), ibcp);
        added
            .reading(PARASITIC)
            .add_current(stamp, sums, (BX, SI), iccp);
        charges.add(&added, sums, stamp);
        derivatives.stamp(&shape.stamp, stamp);
    }

    fn variable_names() -> &'static [&'static str] {
        &Bipolar::VARIABLE_NAMES
    }

    fn variables(memory: &VbicMemory, values: &mut [f64]) {
        parts::write_variables(&memory.variables, values);
    }

    fn regions() -> Regions {
        Bipolar::REGIONS
    }

    /// Emits thermal noise for extrinsic and intrinsic resistances, shot noise for
    /// transport and base-emitter currents, and flicker noise for both base-emitter
    /// currents per ngspice VBIC.
    fn noise<R: Real>(
        shape: &VbicShape,
        prepared: &PreparedVbic<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource<R>>,
    ) {
        let temperature = prepared.temperature;
        let names = [
            "collector thermal",
            "base thermal",
            "emitter thermal",
            "substrate thermal",
        ];
        for ((series, resistance), name) in shape.series.iter().zip(prepared.series).zip(names) {
            series.noise(series.value_of(resistance), (name, temperature), sources);
        }
        let nodes = &shape.stamp;
        let voltages = nodes.oriented::<R, _>(at, prepared.sign, VARIABLES);
        let branches = prepared.core.evaluate(voltages, at.gmin);
        let node = |index: usize| nodes.equation(index);
        // Each intrinsic resistance law places branch drop first, giving conductance at
        // index 0.
        let slope = |current: &Dual<R, 3>| {
            current
                .gradient
                .first()
                .copied()
                .unwrap_or_else(|| R::constant(0.0))
        };
        for (name, (plus, minus), current) in [
            ("intrinsic collector thermal", (CX, CI), &branches.irci),
            ("intrinsic base thermal", (BX, BI), &branches.irbi),
            ("parasitic base thermal", (BP, CX), &branches.irbp),
        ] {
            sources.push(NoiseSource::thermal(
                name,
                [node(plus), node(minus)],
                slope(current),
                temperature,
            ));
        }
        let transport = shape.excess.as_ref().map_or_else(
            || (branches.itzf - branches.itzr).value,
            |excess| excess.delayed::<R>(at) * prepared.sign - branches.itzr.value,
        );
        let base_emitter = [node(BI), node(EI)];
        let parasitic_base_emitter = [node(BX), node(BP)];
        let (ibe, ibep) = (branches.ibe.value, branches.ibep.value);
        sources.extend([
            NoiseSource::shot("collector shot", [node(CI), node(EI)], transport),
            NoiseSource::shot("base shot", base_emitter, ibe),
            NoiseSource::shot("parasitic base shot", parasitic_base_emitter, ibep),
            NoiseSource::shot(
                "parasitic collector shot",
                [node(BX), node(SI)],
                branches.iccp.value,
            ),
        ]);
        prepared.flicker.add("flicker", base_emitter, ibe, sources);
        prepared
            .flicker
            .add("parasitic flicker", parasitic_base_emitter, ibep, sources);
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::ELEMENTARY_CHARGE;
    use crate::kernel::region::{BASE_COLLECTOR, BASE_EMITTER};
    use crate::testing::{
        Bench, assert_bessel_filter, assert_close, assert_jacobians_match,
        assert_parameter_gradients_match, assert_regions, source, thermal,
    };

    /// VBIC's thermal voltage at the reference temperature.
    const VTV: f64 = REFERENCE_TEMPERATURE * VBIC_BOLTZMANN / VBIC_CHARGE;

    /// A transistor with every part of its model in play, with intrinsic resistances, each
    /// number positive so that each one's gradient steps both ways.
    const FULL: &str = r#"{"saturation_current": 2e-16, "forward_emission_coefficient": 1.01,
        "reverse_emission_coefficient": 1.02, "reverse_saturation_current_ratio": 0.8,
        "forward_early_voltage": 60, "reverse_early_voltage": 4,
        "forward_knee_current": 0.02, "reverse_knee_current": 0.005, "knee_rolloff": 0.6,
        "ideal_base_emitter_current": 2e-18, "intrinsic_base_emitter_fraction": 0.8,
        "ideal_base_emitter_emission_coefficient": 1.02,
        "nonideal_base_emitter_current": 3e-15,
        "nonideal_base_emitter_emission_coefficient": 1.8,
        "ideal_base_collector_current": 4e-17,
        "ideal_base_collector_emission_coefficient": 1.03,
        "nonideal_base_collector_current": 2e-15,
        "nonideal_base_collector_emission_coefficient": 1.9,
        "avalanche_coefficient": 2.5, "avalanche_exponent_coefficient": 12,
        "parasitic_saturation_current": 5e-18, "parasitic_extrinsic_fraction": 0.7,
        "parasitic_forward_emission_coefficient": 1.05,
        "parasitic_ideal_base_emitter_current": 1e-19,
        "parasitic_nonideal_base_emitter_current": 1e-16,
        "parasitic_ideal_base_collector_current": 2e-19,
        "parasitic_ideal_base_collector_emission_coefficient": 1.04,
        "parasitic_nonideal_base_collector_current": 3e-16,
        "parasitic_nonideal_base_collector_emission_coefficient": 1.7,
        "parasitic_knee_current": 1e-4, "intrinsic_collector_resistance": 20,
        "epitaxial_saturation_voltage": 2, "epitaxial_doping_parameter": 2e-11,
        "high_current_collector_factor": 0.5, "intrinsic_base_resistance": 30,
        "parasitic_base_resistance": 40, "base_emitter_capacitance": 2e-14,
        "base_emitter_potential": 0.85, "base_emitter_grading": 0.3,
        "base_emitter_smoothing": -0.5, "base_collector_capacitance": 1e-14,
        "base_collector_potential": 0.65, "base_collector_grading": 0.35,
        "base_collector_smoothing": -0.5, "extrinsic_base_collector_capacitance": 5e-15,
        "epitaxial_charge": 1e-14, "collector_substrate_capacitance": 3e-14,
        "collector_substrate_potential": 0.6, "collector_substrate_grading": 0.4,
        "collector_substrate_smoothing": -0.5, "collector_substrate_fixed_capacitance": 1e-15,
        "depletion_coefficient": 0.85, "forward_transit_time": 1e-11,
        "transit_time_base_width_coefficient": 0.2, "transit_time_bias_coefficient": 3,
        "transit_time_voltage": 2, "transit_time_current": 0.01, "reverse_transit_time": 1e-9,
        "base_emitter_breakdown_voltage": 5,
        "base_emitter_breakdown_emission_coefficient": 1.5,
        "base_emitter_breakdown_current": 1e-9, "reach_through_voltage": 3,
        "reach_through_smoothing": 0.1}"#;

    /// Every temperature parameter, away from its default.
    const WARM: &str = r#""nominal_temperature": 290, "activation_energy": 1.15,
        "ideal_base_emitter_activation_energy": 1.1,
        "ideal_base_collector_activation_energy": 1.13,
        "ideal_collector_substrate_activation_energy": 1.14,
        "nonideal_base_emitter_activation_energy": 1.05,
        "nonideal_base_collector_activation_energy": 1.06,
        "nonideal_collector_substrate_activation_energy": 1.07,
        "parasitic_activation_energy": 1.11, "reverse_activation_energy_shift": 0.05,
        "saturation_current_temperature_exponent": 2.5,
        "ideal_base_current_temperature_exponent": 3.5,
        "nonideal_base_current_temperature_exponent": 2.8,
        "reverse_saturation_current_temperature_exponent": 0.5,
        "emission_coefficient_temperature_coefficient": 1e-4,
        "avalanche_temperature_coefficient": 2e-3,
        "forward_knee_current_temperature_exponent": 0.3,
        "extrinsic_collector_resistance_temperature_exponent": 0.6,
        "intrinsic_collector_resistance_temperature_exponent": 0.7,
        "extrinsic_base_resistance_temperature_exponent": 0.8,
        "intrinsic_base_resistance_temperature_exponent": 0.9,
        "emitter_resistance_temperature_exponent": 1.1,
        "substrate_resistance_temperature_exponent": 1.2,
        "parasitic_base_resistance_temperature_exponent": 1.3,
        "epitaxial_saturation_voltage_temperature_exponent": 0.4,
        "breakdown_voltage_temperature_coefficient": 1e-3,
        "breakdown_voltage_quadratic_temperature_coefficient": 1e-6,
        "breakdown_emission_temperature_coefficient": 1e-4, "#;

    /// The series resistances and overlap capacitances, every resistance above 1 Ω.
    const SERIES: &str = r#""extrinsic_collector_resistance": 5,
        "extrinsic_base_resistance": 8, "emitter_resistance": 2, "substrate_resistance": 3,
        "base_emitter_overlap_capacitance": 1e-15, "base_collector_overlap_capacitance": 2e-15, "#;

    /// The series resistances, every one below 1 Ω, so each carries its current as an
    /// unknown.
    const SMALL: &str = r#""extrinsic_collector_resistance": 0.5,
        "extrinsic_base_resistance": 0.4, "emitter_resistance": 0.2,
        "substrate_resistance": 0.3, "#;

    /// Returns `parameters` with JSON fragment `more` prepended.
    fn with(parameters: &str, more: &str) -> String {
        parameters.replacen('{', &format!("{{{more}"), 1)
    }

    /// Returns `parameters` configured with VBIC capacitance smoothing and Gummel-Poon
    /// base charge.
    fn smoothed(parameters: &str) -> String {
        let smoothed = parameters
            .replace(
                r#""base_emitter_smoothing": -0.5"#,
                r#""base_emitter_smoothing": 0.3"#,
            )
            .replace(
                r#""base_collector_smoothing": -0.5"#,
                r#""base_collector_smoothing": 0.2"#,
            )
            .replace(
                r#""collector_substrate_smoothing": -0.5"#,
                r#""collector_substrate_smoothing": 0.25"#,
            );
        with(&smoothed, r#""base_charge": "gummel_poon", "#)
    }

    /// Returns `parameters` configured for a PNP transistor.
    fn pnp(parameters: &str) -> String {
        with(parameters, r#""polarity": "pnp", "#)
    }

    /// Extracts the four extrinsic series resistances from JSON `parameters` in terminal
    /// order.
    fn series_of(parameters: &str) -> [f64; 4] {
        let read: VbicParameters = serde_json::from_str(parameters).expect("parameters");
        [
            read.extrinsic_collector_resistance,
            read.extrinsic_base_resistance,
            read.emitter_resistance,
            read.substrate_resistance,
        ]
    }

    /// Builds an unknown vector for `parameters` on `bench` at terminal `voltages` with
    /// polarity `sign`. Internal nodes offset slightly behind their terminals, current
    /// unknowns match series drops, and excess-phase unknowns sit near transport current.
    fn biased(bench: &Bench, parameters: &str, voltages: [f64; 4], sign: f64) -> Vec<f64> {
        let [collector, base, emitter, substrate] = voltages.map(|voltage| sign * voltage);
        let [rcx, rbx, re, rs] = series_of(parameters);
        let behind = |terminal: f64, resistance: f64, drop: f64| {
            if resistance > 0.0 {
                terminal - sign * drop
            } else {
                terminal
            }
        };
        let cx = behind(collector, rcx, 0.01);
        let bx = behind(base, rbx, 0.004);
        let ei = behind(emitter, re, -0.002);
        let si = behind(substrate, rs, -0.001);
        bench
            .unknowns()
            .iter()
            .map(|name| match *name {
                "n0" => collector,
                "n1" => base,
                "n2" => emitter,
                "n3" => substrate,
                "device:collector" => cx,
                "device:collector_current" => (collector - cx) / rcx,
                "device:base" => bx,
                "device:base_current" => (base - bx) / rbx,
                "device:emitter" => ei,
                "device:emitter_current" => (emitter - ei) / re,
                "device:substrate" => si,
                "device:substrate_current" => (substrate - si) / rs,
                "device:intrinsic_collector" => cx - sign * 0.05,
                "device:parasitic_base" => cx - sign * 0.003,
                "device:intrinsic_base" => bx - sign * 0.006,
                "device:transport_filter" => 2e-4,
                "device:delayed_transport" => 1.5e-4,
                other => panic!("no value for {other}"),
            })
            .collect()
    }

    #[test]
    fn its_transport_current_follows_the_exponential_over_the_base_charge() {
        // Without Early voltages or knees, qb is q1, VBIC's smoothing of 1.
        let parameters = r#"{"saturation_current": 1e-15}"#;
        let mut bench = Bench::new::<Vbic>(parameters);
        assert_eq!(
            bench.unknowns(),
            [
                "n0",
                "n1",
                "n2",
                "n3",
                "device:intrinsic_collector",
                "device:parasitic_base",
                "device:intrinsic_base"
            ]
        );
        let (vbe, vbc) = (0.7, -2.3);
        let x = [3.0, 0.7, 0.0, 0.0, 3.0, 3.0, 0.7];
        let evaluation = bench.settle(&x);
        let q1 = 0.5 * ((1.0f64 - 1e-4).powi(2) + 1e-8).sqrt() + 0.5 * (1.0 - 1e-4) + 1e-4;
        let transport = 1e-15 * ((vbe / VTV).exp() - (vbc / VTV).exp()) / q1;
        // The base-collector junction's ideal current, IBCI's default, leaves the
        // intrinsic collector's equation too.
        let base_collector = 1e-16 * ((vbc / VTV).exp() - 1.0);
        assert_close(&[evaluation.residual[4]], &[transport - base_collector]);
        // Intrinsic collector resistance carries zero current at zero drop; base current
        // equals IBEI plus IBCI.
        assert!(evaluation.residual[0].abs() < 1e-30);
        let base = 1e-18 * ((vbe / VTV).exp() - 1.0) + base_collector;
        assert_close(&[evaluation.residual[6]], &[base]);
    }

    #[test]
    fn the_saturation_current_scales_with_its_activation_energy() {
        let temperature = 350.0;
        let parameters = r#"{"saturation_current": 1e-15, "forward_emission_coefficient": 1.1,
            "activation_energy": 1.2, "saturation_current_temperature_exponent": 2.5,
            "ideal_base_emitter_current": 0, "ideal_base_collector_current": 0}"#;
        let mut bench = Bench::at::<Vbic>(parameters, temperature);
        let x = [0.7, 0.7, 0.0, 0.0, 0.7, 0.7, 0.7];
        let evaluation = bench.settle(&x);
        let vtv = temperature * VBIC_BOLTZMANN / VBIC_CHARGE;
        let ratio = temperature / REFERENCE_TEMPERATURE;
        let scaled = 1e-15 * (ratio.powf(2.5) * (-1.2 * (1.0 - ratio) / vtv).exp()).powf(1.0 / 1.1);
        let q1 = 0.5 * ((1.0f64 - 1e-4).powi(2) + 1e-8).sqrt() + 0.5 * (1.0 - 1e-4) + 1e-4;
        let expected = scaled * ((0.7 / (1.1 * vtv)).exp() - 1.0) / q1;
        assert_close(&[evaluation.residual[4]], &[expected]);
    }

    #[test]
    fn its_potentials_hold_at_their_nominal_temperature_and_fall_above_it() {
        let read = VbicParameters::<f64>::default();
        let scaled = |temperature: f64| {
            let temperatures =
                Temperatures::of_junction(temperature, REFERENCE_TEMPERATURE).expect("valid");
            read.charges(Scaling::new(temperatures)).expect("valid")
        };
        let nominal = scaled(REFERENCE_TEMPERATURE);
        assert!((nominal.base_emitter.potential - 0.75).abs() < 1e-15);
        let hot = scaled(400.0);
        assert!(hot.base_emitter.potential < 0.65 && hot.base_emitter.potential > 0.4);
    }

    #[test]
    fn its_depletion_charge_follows_the_power_law_and_reaches_through() {
        // Only the intrinsic base-collector junction stores charge.
        let parameters = r#"{"base_collector_capacitance": 1e-12,
            "base_collector_potential": 0.7, "base_collector_grading": 0.4,
            "reach_through_voltage": 2, "ideal_base_collector_current": 0}"#;
        let mut bench = Bench::new::<Vbic>(parameters);
        let (potential, grading, through) = (0.7f64, 0.4, 2.0);
        let law = |v: f64| {
            potential * (1.0 - (1.0 - v / potential).powf(1.0 - grading)) / (1.0 - grading)
        };
        for vbc in [-1.5, -0.3, 0.2] {
            let x = [-vbc, 0.0, 0.0, 0.0, -vbc, -vbc, 0.0];
            let evaluation = bench.settle(&x);
            assert_close(&[evaluation.charge[6]], &[1e-12 * law(vbc)]);
        }
        // Beyond the reach-through voltage the capacitance holds at its value there.
        let x = [4.0, 0.0, 0.0, 0.0, 4.0, 4.0, 0.0];
        let evaluation = bench.settle(&x);
        let reached = 1e-12 * (1.0 + through / potential).powf(-grading);
        assert_close(&[evaluation.capacitance[6][6]], &[reached]);
    }

    #[test]
    fn its_avalanche_current_flows_from_the_collector_into_the_base() {
        let plain = r#"{"saturation_current": 1e-15, "ideal_base_collector_current": 1e-14}"#;
        let avalanche = with(
            plain,
            r#""avalanche_coefficient": 3, "avalanche_exponent_coefficient": 8, "#,
        );
        let x = [8.0, 0.7, 0.0, 0.0, 8.0, 8.0, 0.7];
        let without = Bench::new::<Vbic>(plain).settle(&x);
        let with_avalanche = Bench::new::<Vbic>(&avalanche).settle(&x);
        let vbci = -7.3;
        let across = 0.75 - vbci;
        let depleted = f64::midpoint((across * across + 0.01f64).sqrt(), across);
        let factor = 3.0 * depleted * (-8.0 * depleted.powf(0.33 - 1.0)).exp();
        // The driving current is the transport current less the base-collector current,
        // what leaves the intrinsic collector's equation without the avalanche.
        let generated = without.residual[4] * factor;
        assert_close(
            &[
                with_avalanche.residual[4] - without.residual[4],
                with_avalanche.residual[6] - without.residual[6],
            ],
            &[generated, -generated],
        );
    }

    #[test]
    fn its_intrinsic_collector_quasi_saturates() {
        let parameters = r#"{"intrinsic_collector_resistance": 100,
            "epitaxial_saturation_voltage": 1, "high_current_collector_factor": 2}"#;
        let mut bench = Bench::new::<Vbic>(parameters);
        let drop = 1.5;
        let x = [3.0, 0.0, 0.0, 0.0, 3.0 - drop, 3.0, 0.0];
        let evaluation = bench.settle(&x);
        let ohmic = drop / 100.0;
        let factor = 100.0 * ohmic / (1.0 + 0.5 * 0.5 * (drop * drop + 0.01f64).sqrt());
        let expected = ohmic / (1.0 + factor * factor).sqrt();
        assert_close(&[evaluation.residual[0]], &[expected]);
    }

    #[test]
    fn its_resistances_and_excess_phase_add_unknowns() {
        let bench = Bench::new::<Vbic>(&with(
            FULL,
            &format!("{SMALL}\"excess_phase_delay\": 2e-12, "),
        ));
        assert_eq!(
            bench.unknowns(),
            [
                "n0",
                "n1",
                "n2",
                "n3",
                "device:collector",
                "device:collector_current",
                "device:base",
                "device:base_current",
                "device:emitter",
                "device:emitter_current",
                "device:substrate",
                "device:substrate_current",
                "device:intrinsic_collector",
                "device:parasitic_base",
                "device:intrinsic_base",
                "device:transport_filter",
                "device:delayed_transport",
            ]
        );
    }

    #[test]
    fn its_excess_phase_delays_the_forward_transport_current_through_a_bessel_filter() {
        let delayed = with(FULL, r#""excess_phase_delay": 2e-12, "#);
        let mut plain = Bench::new::<Vbic>(FULL);
        let mut bench = Bench::new::<Vbic>(&delayed);
        let x = biased(&plain, FULL, [3.0, 0.75, 0.0, 0.0], 1.0);
        let rest = plain.settle(&x);
        // At x1 = x2 = 0, x1's residual is less the forward transport current; at rest
        // both are it, and the transistor's currents are those it has without the delay.
        let mut zero = x.clone();
        zero.extend([0.0, 0.0]);
        let forward = -bench.settle(&zero).residual[7];
        let mut settled_at = x.clone();
        settled_at.extend([forward, forward]);
        let settled = bench.settle(&settled_at);
        assert_close(&settled.residual[..7], &rest.residual);
        assert!(settled.residual[7].abs() <= 1e-15 * forward.abs());
        assert!(settled.residual[8].abs() <= 1e-15 * forward.abs());
        // Linearized, x2 follows forward current via 3 / (s² TD² + 3 s TD + 3): solve rows
        // x1 and x2 at s = jω for unit drive into x1.
        let delay = 2e-12;
        assert_bessel_filter(
            &settled,
            [7, 8],
            [1.0, 0.0],
            1,
            delay,
            [1e9, 1.0 / delay, 3e12],
        );
    }

    #[test]
    fn rejects_out_of_range_parameters() {
        for (parameters, message) in [
            (
                r#"{"intrinsic_base_resistance": 0}"#,
                "intrinsic_base_resistance is positive, not 0.0",
            ),
            (
                r#"{"forward_early_voltage": 0}"#,
                "forward_early_voltage is positive, not 0.0",
            ),
            (
                r#"{"intrinsic_base_emitter_fraction": 1.5}"#,
                "intrinsic_base_emitter_fraction is at least 0 and at most 1, not 1.5",
            ),
            (
                r#"{"base_collector_grading": 1}"#,
                "base_collector_grading is at least 0 and below 1, not 1.0",
            ),
            (
                r#"{"forward_emission_coefficient": 1,
                    "emission_coefficient_temperature_coefficient": -0.5}"#,
                "the temperature coefficients scale the forward_emission_coefficient to zero \
                 at 300.15 K",
            ),
        ] {
            let parameters = if message.ends_with("300.15 K") {
                parameters.replace('}', r#", "nominal_temperature": 290}"#)
            } else {
                parameters.to_owned()
            };
            let error = Bench::try_new::<Vbic>(&parameters).unwrap_err();
            assert_eq!(error.to_string(), message);
        }
    }

    #[test]
    fn parameter_gradients_are_exact() {
        let every = with(
            &with(FULL, SERIES),
            &format!("{WARM}\"excess_phase_delay\": 2e-12, "),
        );
        let smoothed = smoothed(&with(FULL, SMALL));
        // Forward, saturated, reversed, and with the base-emitter junction far reversed,
        // short of where VER's Early effect floors q1: there an entry's terms cancel to a
        // thousandth of themselves, and differences in the parameters lose its digits.
        for voltages in [
            [3.0, 0.78, 0.0, -1.0],
            [0.3, 0.8, 0.0, 0.0],
            [-0.5, 0.2, 0.0, 0.3],
            [6.0, -4.6, 0.0, -2.0],
        ] {
            for sign in [1.0, -1.0] {
                for parameters in [&every, &smoothed] {
                    let card = if sign < 0.0 {
                        pnp(parameters)
                    } else {
                        parameters.clone()
                    };
                    let bench = Bench::new::<Vbic>(&card);
                    assert_parameter_gradients_match::<Vbic>(
                        &card,
                        &biased(&bench, &card, voltages, sign),
                    );
                }
            }
        }
    }

    proptest! {
        /// Verifies analytical conductances and capacitances against finite differences
        /// across all junction bias regions, smoothing modes, base-charge forms, excess
        /// phase, polarities, and series resistance regimes.
        #[test]
        fn jacobians_match_finite_differences(
            base in -6.0..0.9f64,
            collector_above_base in -0.9..7.0f64,
            substrate in -3.0..0.8f64,
            pnp_polarity in proptest::bool::ANY,
            smooth in proptest::bool::ANY,
            small in proptest::bool::ANY,
        ) {
            let mut parameters = with(FULL, if small { SMALL } else { SERIES });
            if smooth {
                parameters = smoothed(&parameters);
            }
            let delayed = with(&parameters, r#""excess_phase_delay": 2e-12, "#);
            let sign = if pnp_polarity { -1.0 } else { 1.0 };
            for parameters in [parameters, delayed] {
                let parameters = if pnp_polarity { pnp(&parameters) } else { parameters };
                let bench = Bench::new::<Vbic>(&parameters);
                let x = biased(
                    &bench,
                    &parameters,
                    [base + collector_above_base, base, 0.0, substrate],
                    sign,
                );
                assert_jacobians_match::<Vbic>(&parameters, &x);
            }
        }
    }

    #[test]
    fn its_resistances_and_currents_make_thermal_shot_and_flicker_noise() {
        let parameters = r#"{"saturation_current": 1e-15, "intrinsic_collector_resistance": 20,
            "intrinsic_base_resistance": 30, "parasitic_base_resistance": 40,
            "extrinsic_collector_resistance": 5, "extrinsic_base_resistance": 0.5,
            "emitter_resistance": 2, "substrate_resistance": 3,
            "parasitic_saturation_current": 1e-17, "parasitic_ideal_base_emitter_current": 1e-18,
            "flicker_coefficient": 1e-14, "flicker_exponent": 1.2,
            "flicker_frequency_exponent": 1.1}"#;
        let mut bench = Bench::new::<Vbic>(parameters);
        let x = biased(&bench, parameters, [3.0, 0.75, 0.0, -1.0], 1.0);
        let evaluation = bench.settle(&x);
        let sources = bench.noise(&x);
        // Saturated, the parasitic transistor conducts.
        let saturated = bench.noise(&biased(&bench, parameters, [0.3, 0.75, 0.0, -1.0], 1.0));
        let names = bench.unknowns();
        let index = |name: &str| {
            names
                .iter()
                .position(|unknown| *unknown == name)
                .expect(name)
        };
        let [ci, bi, bp] = [
            "device:intrinsic_collector",
            "device:intrinsic_base",
            "device:parasitic_base",
        ]
        .map(index);
        let [cx, bx, ei] = ["device:collector", "device:base", "device:emitter"].map(index);
        // The intrinsic resistances' thermal noise at their slopes, which are their
        // conductances' entries.
        assert_close(
            &[
                source(&sources, "collector thermal").density(1.0),
                source(&sources, "emitter thermal").density(1.0),
                source(&sources, "substrate thermal").density(1.0),
                source(&sources, "base thermal").density(1.0),
                source(&sources, "intrinsic collector thermal").density(1.0),
                source(&sources, "intrinsic base thermal").density(1.0),
                source(&sources, "parasitic base thermal").density(1.0),
            ],
            &[
                thermal(0.2),
                thermal(0.5),
                thermal(1.0 / 3.0),
                thermal(0.5),
                thermal(-evaluation.conductance[cx][ci]),
                thermal(-evaluation.conductance[bx][bi]),
                thermal(-evaluation.conductance[bp][cx]),
            ],
        );
        let shot = |current: f64| 2.0 * ELEMENTARY_CHARGE * current.abs();
        let base_shot = source(&sources, "base shot");
        assert_eq!(
            base_shot.equations,
            [
                Some(crate::UnknownIndex::new(bi)),
                Some(crate::UnknownIndex::new(ei))
            ]
        );
        let flicker = source(&sources, "flicker");
        let base_current = flicker.flicker / 1e-14;
        assert_close(
            &[base_shot.density(1.0)],
            &[shot(base_current.powf(1.0 / 1.2))],
        );
        assert_close(
            &[flicker.density(10.0)],
            &[flicker.flicker / 10f64.powf(1.1)],
        );
        // The transport current is IS / IBEI, a thousand, times the base current.
        let ratio = source(&sources, "collector shot").density(1.0) / base_shot.density(1.0);
        assert!((500.0..2000.0).contains(&ratio), "{ratio}");
        for name in [
            "parasitic base shot",
            "parasitic collector shot",
            "parasitic flicker",
        ] {
            assert!(source(&saturated, name).density(1.0) > 0.0, "{name}");
        }
        assert_eq!(sources.len(), 13, "{sources:?}");
    }

    /// Verifies cut-off, active, saturated, and reversed region classification and
    /// boundary margins `Vbei` and `Vbci` for NPN and PNP transistors.
    #[test]
    fn reports_its_region_and_margins() {
        // The collector, base, emitter and substrate, then the intrinsic collector, the
        // parasitic base and the intrinsic base, each where no current would drop.
        let at = |collector: f64, base: f64, emitter: f64| {
            [collector, base, emitter, 0.0, collector, collector, base]
        };
        let (cut_off, active, saturated, reversed) = (
            at(5.0, -0.5, 0.0),
            at(5.0, 0.7, 0.0),
            at(0.2, 0.7, 0.0),
            at(0.0, 0.7, 5.0),
        );
        let (be_low, be_high) = (at(5.0, -0.1, 0.0), at(5.0, 0.1, 0.0));
        assert_regions::<Vbic>(
            "{}",
            Some(r#"{"polarity": "pnp"}"#),
            &[
                (&cut_off, "cut_off", Some(&[-0.5, -5.5])),
                (&active, "active", Some(&[0.7, -4.3])),
                (&saturated, "saturated", Some(&[0.7, 0.7 - 0.2])),
                (&reversed, "reversed", Some(&[-4.3, 0.7])),
            ],
            &[
                (&be_low, &be_high, BASE_EMITTER),
                (&active, &saturated, BASE_COLLECTOR),
            ],
        );
    }
}
