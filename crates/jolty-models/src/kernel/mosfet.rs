//! The MOSFET of levels 1 and 3, derived from ngspice 47's implementations of them in
//! `src/spicelib/devices/mos1/` and `mos3/`:
//! Copyright 1985 - 2018, Regents of the University of California and others. Used under
//! ngspice's Modified BSD license, which the repository's `NOTICE` reproduces.

mod semiempirical;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use self::semiempirical::Semiempirical;
use crate::kernel::dual::Dual;
use crate::kernel::jfet::Channel;
use crate::kernel::junction::{
    BOLTZMANN, Depletion, ELEMENTARY_CHARGE, Temperatures, band_gap, below_one, critical_voltage,
    depletion_charge, junction_current, limit_channel, limit_junction,
};
use crate::kernel::parts::{self, Derivatives, Nodes, Oriented, Series, series_value};
use crate::kernel::region::{OVERDRIVE, SATURATION_MARGIN};
use crate::noise::Flicker;
use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Lift, Local, Nature, NoiseSource,
    ParameterError, REFERENCE_TEMPERATURE, REGION, Real, Regions, Stamp,
};

/// The vacuum permittivity of SPICE's MOSFET models, in farads per meter: SPICE2's value,
/// with which models are fitted, rather than CODATA's 8.8541878128e-12.
const VACUUM_PERMITTIVITY: f64 = 8.854_214_871e-12;
/// The permittivity of silicon dioxide, in farads per meter.
pub(crate) const OXIDE_PERMITTIVITY: f64 = 3.9 * VACUUM_PERMITTIVITY;
/// The permittivity of silicon, in farads per meter.
const SILICON_PERMITTIVITY: f64 = 11.7 * VACUUM_PERMITTIVITY;
/// Silicon's intrinsic carrier concentration at 300 K, per cubic meter.
const INTRINSIC_CONCENTRATION: f64 = 1.45e16;
/// The oxide thickness, in meters, SPICE takes where none is given: for a level 3
/// transistor's process, and for any transistor's flicker noise.
pub(crate) const DEFAULT_OXIDE_THICKNESS: f64 = 1e-7;
/// The least saturation voltage Meyer's capacitances take, in volts.
const MEYER_SATURATION: f64 = 0.025;

/// Which of SPICE's MOSFET models a transistor follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(try_from = "u8", into = "u8")]
pub enum ModelLevel {
    /// Level 1, Shichman and Hodges's square law.
    One,
    /// Level 3, a semi-empirical law of short channels.
    Three,
}

impl TryFrom<u8> for ModelLevel {
    type Error = String;

    fn try_from(level: u8) -> Result<Self, String> {
        match level {
            1 => Ok(Self::One),
            3 => Ok(Self::Three),
            _ => Err(format!(
                "jolty models MOSFETs of level 1 and 3, not {level}"
            )),
        }
    }
}

impl From<ModelLevel> for u8 {
    fn from(level: ModelLevel) -> Self {
        match level {
            ModelLevel::One => 1,
            ModelLevel::Three => 3,
        }
    }
}

/// The material of a MOSFET's gate, which sets its work function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GateMaterial {
    /// Polysilicon doped opposite to the substrate, as an n⁺ gate over a p substrate:
    /// SPICE's `tpg = 1`.
    OppositeDoping,
    /// Polysilicon doped like the substrate: SPICE's `tpg = −1`.
    SameDoping,
    /// Aluminum: SPICE's `tpg = 0`.
    Aluminum,
}

/// A metal-oxide-semiconductor field-effect transistor, in SPICE's level 1
/// (Shichman-Hodges) or level 3 model with Meyer's gate capacitances, as ngspice
/// implements them.
///
/// At level 1, the drain current follows the square law: with `vgst = vgs − von` and
/// `β' = β · (1 + λ · vds)`, it is `β' · vgst² / 2` in saturation, where `vds ≥ vgst`, and
/// `β' · vds · (vgst − vds / 2)` below it, with `β = kp · w / (l − 2 · ld)`. The threshold
/// `von = vto + γ · (√(φ − vbs) − √φ)` rises as the source rises above the bulk. Level 3
/// adds short and narrow channels' effects on the threshold, the drain's static feedback,
/// the gate field's degradation of the mobility, carriers' velocity saturation, the
/// channel's shortening in saturation, and weak inversion. The transistor is symmetric:
/// with the drain below the source, the two swap.
///
/// Two junctions join the bulk to the drain and the source, each carrying a
/// junction current and holding a depletion charge, bottom and
/// sidewall. The gate holds overlap charges against the source, the drain and the bulk,
/// and, given an oxide thickness, Meyer's capacitances across the oxide. These depend on
/// every terminal voltage and no charge yields them, so each carries a current
/// `C(v) · dv/dt`, which follows an unknown of the transistor's own: the displacement
/// current through the whole oxide, `Cox · dv/dt`, labelled `gate_source_displacement`,
/// `gate_drain_displacement` or `gate_bulk_displacement`. A drain or source resistance adds
/// an internal node, labelled `drain` or `source`; one below
/// [`SMALL_RESISTANCE`](crate::SMALL_RESISTANCE) also carries its current as an unknown,
/// labelled `drain_current` or `source_current`.
#[derive(Debug, Clone, Copy)]
pub struct Mosfet;

/// Where a transistor's contributions go among its local unknowns: through its drain and
/// source resistances, among its gate, bulk and the drain and source behind the
/// resistances, and through the displacements of its oxide, if it has one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MosfetShape {
    stamp: Nodes<4>,
    drain_series: Series,
    source_series: Series,
    /// The displacement currents towards the source, the drain and the bulk.
    displacements: Option<[Displacement; 3]>,
}

/// A transistor's numbers at its temperature.
#[derive(Debug, Clone, Copy)]
pub struct PreparedMosfet<R: Real> {
    sign: f64,
    /// The drain and the source resistances' conductances, or each resistance where its
    /// current is an unknown.
    series: [R; 2],
    channel: Law<R>,
    /// The bulk-drain and bulk-source junctions.
    junctions: [BulkJunction<R>; 2],
    /// The gate-source, gate-drain and gate-bulk overlap capacitances.
    overlaps: [R; 3],
    oxide: Option<Oxide<R>>,
    /// The threshold, signed as the channel conducts, from which the first step is limited.
    threshold: f64,
    temperature: f64,
    /// The flicker noise of the drain current: each copy's, of its share of the current
    /// over its own gate's area and the oxide capacitance per area squared, summed over
    /// the copies.
    flicker: Flicker,
}

/// What a transistor remembers.
#[derive(Debug, Clone, Copy, Default)]
pub struct MosfetMemory {
    /// The gate-source, drain-source and bulk-source voltages of the last evaluation, from
    /// which the next steps are limited.
    last: [f64; 3],
    /// The threshold of the last evaluation, signed as the channel conducts; before the
    /// first, or since the transistor was prepared at another temperature, its prepared
    /// one.
    threshold: Option<f64>,
    /// The values of [`VARIABLE_NAMES`] at the last evaluation.
    variables: [f64; 8],
}

/// The variables a transistor reports: its threshold and saturation voltages, in its own
/// polarity, Meyer's capacitances from the gate to the source, the drain and the bulk, and
/// its region among [`REGIONS`] and its margins.
const VARIABLE_NAMES: [&str; 8] = [
    "threshold_voltage",
    "saturation_voltage",
    "gate_source_capacitance",
    "gate_drain_capacitance",
    "gate_bulk_capacitance",
    REGION,
    OVERDRIVE,
    SATURATION_MARGIN,
];

/// The regions of a transistor's channel, as its level's equations switch between them
/// (`mos1load.c`, `mos3load.c`), in the order of [`REGIONS`]' names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Region {
    /// At or below its threshold, conducting nothing.
    Off,
    /// Below its threshold at level 3 with fast surface states, its current falling
    /// exponentially from the threshold's.
    WeakInversion,
    /// Above its threshold, with the drain below saturation.
    Linear,
    /// Above its threshold, with the drain at or past saturation.
    Saturated,
}

/// The regions a transistor's equations distinguish, and its margins to their boundaries:
/// `vgs − von` and `vds − vdsat`, in an n-channel's sign as [`Regions::margins`] are, with
/// the drain and the source swapped where the drain lies below the source, as the
/// equations swap them.
const REGIONS: Regions = Regions {
    names: &["off", "weak_inversion", "linear", "saturated"],
    margins: &[OVERDRIVE, SATURATION_MARGIN],
};

/// Local node indices: the gate and the bulk, then the drain and source behind their
/// resistances, which are the terminals themselves without one.
const GATE: usize = 0;
const BULK: usize = 1;
const INNER_DRAIN: usize = 2;
const INNER_SOURCE: usize = 3;

/// The stamped nodes across the oxide from the gate, towards which its displacement
/// currents flow: the source, the drain and the bulk.
const DISPLACED: [usize; 3] = [INNER_SOURCE, INNER_DRAIN, BULK];

/// The pairs of nodes across which the voltages the transistor's equations take as
/// variables lie: gate-source, drain-source and bulk-source.
const VARIABLES: [(usize, usize); 3] = [
    (GATE, INNER_SOURCE),
    (INNER_DRAIN, INNER_SOURCE),
    (BULK, INNER_SOURCE),
];

/// The parameters of a [`Mosfet`], with SPICE's defaults. The parameters from `vmax` on
/// are level 3's alone.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift)]
#[serde(default, deny_unknown_fields, bound = "")]
pub struct MosfetParameters<R: Real = f64> {
    /// The model's level, 1 or 3; default 1.
    pub level: ModelLevel,
    /// The channel; default `n`.
    pub channel: Channel,
    /// The channel's width, `w`, in meters; default 100 µm.
    pub width: R,
    /// The channel's drawn length, `l`, in meters; default 100 µm.
    pub length: R,
    /// The number of identical transistors in parallel, `m`; default 1.
    pub multiplier: R,
    /// The zero-bias threshold voltage, `vto`, in volts; default 0, or derived from the
    /// substrate doping.
    pub threshold_voltage: Option<R>,
    /// The transconductance parameter, `kp`, in amperes per square volt; default 2e-5, or
    /// the surface mobility times the oxide capacitance per area, given an oxide thickness.
    pub transconductance_parameter: Option<R>,
    /// The body-effect coefficient, `γ`, in square-root volts; default 0, or derived from
    /// the substrate doping.
    pub body_effect: Option<R>,
    /// The surface potential, `φ`, in volts; default 0.6, or derived from the substrate
    /// doping.
    pub surface_potential: Option<R>,
    /// The channel-length modulation, `λ`, per volt, level 1's alone; default 0.
    pub channel_length_modulation: Option<R>,
    /// The lateral diffusion, `ld`, by which the channel falls short of its drawn length at
    /// each end, in meters, or past it where negative, as ngspice takes it; default 0.
    pub lateral_diffusion: R,
    /// The gate oxide's thickness, `tox`, in meters; default none at level 1, which leaves
    /// the gate without Meyer's capacitances, and 0.1 µm at level 3 and for flicker noise.
    pub oxide_thickness: Option<R>,
    /// The surface mobility, `uo`, in square meters per volt-second, from which `kp`
    /// derives; default 0.06.
    pub surface_mobility: R,
    /// The substrate doping, `nsub`, per cubic meter, from which `vto`, `γ` and `φ` derive
    /// unless given; default none. It needs an oxide thickness.
    pub substrate_doping: Option<R>,
    /// The surface state density, `nss`, per square meter, which shifts a derived `vto`;
    /// default 0.
    pub surface_state_density: R,
    /// The gate's material, which sets a derived `vto`; default `opposite_doping`.
    pub gate_material: GateMaterial,
    /// The drain resistance, in ohms; default none, for the sheet resistance's.
    pub drain_resistance: Option<R>,
    /// The source resistance, in ohms; default none, for the sheet resistance's.
    pub source_resistance: Option<R>,
    /// The drain and source diffusions' sheet resistance, in ohms per square; default 0.
    pub sheet_resistance: R,
    /// The drain diffusion's length in squares, `nrd`; default 1.
    pub drain_squares: R,
    /// The source diffusion's length in squares, `nrs`; default 1.
    pub source_squares: R,
    /// The bulk junctions' saturation current, in amperes; default 1e-14.
    pub saturation_current: R,
    /// The bulk junctions' saturation current per area, in amperes per square meter, which
    /// replaces the saturation current when it and both areas are nonzero; default 0.
    pub saturation_current_density: R,
    /// The drain diffusion's area, in square meters; default 0.
    pub drain_area: R,
    /// The source diffusion's area, in square meters; default 0.
    pub source_area: R,
    /// The drain diffusion's perimeter, in meters; default 0.
    pub drain_perimeter: R,
    /// The source diffusion's perimeter, in meters; default 0.
    pub source_perimeter: R,
    /// The bulk junctions' potential, `pb`, in volts; default 0.8.
    pub junction_potential: R,
    /// The zero-bias bulk-drain capacitance, in farads; default none, for the bottom
    /// capacitance's.
    pub bulk_drain_capacitance: Option<R>,
    /// The zero-bias bulk-source capacitance, in farads; default none, for the bottom
    /// capacitance's.
    pub bulk_source_capacitance: Option<R>,
    /// The bulk junctions' zero-bias bottom capacitance per area, `cj`, in farads per
    /// square meter; default 0.
    pub bottom_capacitance: R,
    /// The bulk junctions' zero-bias sidewall capacitance per length of perimeter, `cjsw`,
    /// in farads per meter; default 0.
    pub sidewall_capacitance: R,
    /// The bottom capacitance's grading coefficient, `mj`, below 1; default 0.5.
    pub bottom_grading: R,
    /// The sidewall capacitance's grading coefficient, `mjsw`, below 1; default 0.5 at
    /// level 1 and 0.33 at level 3.
    pub sidewall_grading: Option<R>,
    /// The fraction of the junction potential above which the depletion capacitances are
    /// extended linearly, below 1; default 0.5.
    pub depletion_coefficient: R,
    /// The gate-source overlap capacitance per width, in farads per meter; default 0.
    pub gate_source_overlap: R,
    /// The gate-drain overlap capacitance per width, in farads per meter; default 0.
    pub gate_drain_overlap: R,
    /// The gate-bulk overlap capacitance per length, in farads per meter; default 0.
    pub gate_bulk_overlap: R,
    /// The temperature the parameters were measured at, in kelvins; default 27 °C.
    pub nominal_temperature: R,
    /// The flicker noise coefficient of the drain current, `KF`; default 0, no flicker
    /// noise.
    pub flicker_coefficient: R,
    /// The power of the drain current in the flicker noise, `AF`; default 1.
    pub flicker_exponent: R,
    /// Level 3: the carriers' maximum drift velocity, `vmax`, in meters per second;
    /// default none, for no velocity saturation.
    pub max_drift_velocity: Option<R>,
    /// Level 3: the junction depth, `xj`, in meters, which a substrate doping turns into
    /// short channels' body effect; default 0.
    pub junction_depth: Option<R>,
    /// Level 3: the fast surface state density, `nfs`, per square meter, which turns on
    /// weak inversion; default 0.
    pub fast_surface_state_density: Option<R>,
    /// Level 3: the drain's static feedback on the threshold, `eta`; default 0.
    pub static_feedback: Option<R>,
    /// Level 3: the width's effect on the threshold, `delta`; default 0.
    pub narrow_width_factor: Option<R>,
    /// Level 3: the gate field's degradation of the mobility, `theta`, per volt;
    /// default 0.
    pub mobility_modulation: Option<R>,
    /// Level 3: the saturation field factor, `kappa`; default 0.2, and 0 for no channel
    /// shortening, as ngspice reads it.
    pub saturation_field_factor: Option<R>,
    /// Level 3: the width lost to diffusion at each side, `wd`, in meters; default 0.
    pub width_narrowing: Option<R>,
    /// Level 3: the length the mask adds, `xl`, in meters; default 0.
    pub length_adjustment: Option<R>,
    /// Level 3: the width the mask adds, `xw`, in meters; default 0.
    pub width_adjustment: Option<R>,
    /// Level 3: a shift of the threshold, `delvto`, in volts; default 0.
    pub threshold_adjustment: Option<R>,
}

impl<R: Real> Default for MosfetParameters<R> {
    fn default() -> Self {
        Self {
            level: ModelLevel::One,
            channel: Channel::N,
            width: R::constant(100e-6),
            length: R::constant(100e-6),
            multiplier: R::constant(1.0),
            threshold_voltage: None,
            transconductance_parameter: None,
            body_effect: None,
            surface_potential: None,
            channel_length_modulation: None,
            lateral_diffusion: R::constant(0.0),
            oxide_thickness: None,
            surface_mobility: R::constant(0.06),
            substrate_doping: None,
            surface_state_density: R::constant(0.0),
            gate_material: GateMaterial::OppositeDoping,
            drain_resistance: None,
            source_resistance: None,
            sheet_resistance: R::constant(0.0),
            drain_squares: R::constant(1.0),
            source_squares: R::constant(1.0),
            saturation_current: R::constant(1e-14),
            saturation_current_density: R::constant(0.0),
            drain_area: R::constant(0.0),
            source_area: R::constant(0.0),
            drain_perimeter: R::constant(0.0),
            source_perimeter: R::constant(0.0),
            junction_potential: R::constant(0.8),
            bulk_drain_capacitance: None,
            bulk_source_capacitance: None,
            bottom_capacitance: R::constant(0.0),
            sidewall_capacitance: R::constant(0.0),
            bottom_grading: R::constant(0.5),
            sidewall_grading: None,
            depletion_coefficient: R::constant(0.5),
            gate_source_overlap: R::constant(0.0),
            gate_drain_overlap: R::constant(0.0),
            gate_bulk_overlap: R::constant(0.0),
            nominal_temperature: R::constant(REFERENCE_TEMPERATURE),
            flicker_coefficient: R::constant(0.0),
            flicker_exponent: R::constant(1.0),
            max_drift_velocity: None,
            junction_depth: None,
            fast_surface_state_density: None,
            static_feedback: None,
            narrow_width_factor: None,
            mobility_modulation: None,
            saturation_field_factor: None,
            width_narrowing: None,
            length_adjustment: None,
            width_adjustment: None,
            threshold_adjustment: None,
        }
    }
}

/// The process parameters, given or derived: `vto`, `kp`, `γ`, `φ`, and the oxide
/// capacitance per area, if any.
#[derive(Debug, Clone, Copy)]
struct Process<R: Real> {
    threshold: R,
    transconductance: R,
    body_effect: R,
    surface_potential: R,
    oxide: Option<R>,
}

impl<R: Real> Process<R> {
    /// The process parameters of a transistor of channel `sign` at `temperatures`, scaled
    /// from the nominal temperature's as ngspice scales them: `kp` falls as `T^−1.5`, `φ`
    /// follows silicon's band gap and intrinsic concentration, and `vto` moves with the
    /// band gap and `φ`.
    ///
    /// # Errors
    ///
    /// Returns an error when the surface potential falls to zero or below.
    fn at(self, temperatures: Temperatures<R>, sign: f64) -> Result<Self, ParameterError> {
        let nominal = self.surface_potential;
        let zero = R::constant(0.0);
        let (potential, _) = temperatures.junction(nominal, zero, zero);
        if potential.value() <= 0.0 {
            let potential = potential.value();
            return Err(ParameterError(format!(
                "the surface potential falls to {potential:?} at {:?} K",
                temperatures.actual.value()
            )));
        }
        let gap_shift = (band_gap(temperatures.nominal) - band_gap(temperatures.actual)) * 0.5;
        Ok(Self {
            threshold: self.threshold
                + gap_shift
                + ((potential - nominal) * 0.5
                    + self.body_effect * (potential.sqrt() - nominal.sqrt()))
                    * sign,
            transconductance: self.transconductance / temperatures.ratio().powf(1.5),
            surface_potential: potential,
            ..self
        })
    }
}

impl<R: Real> MosfetParameters<R> {
    /// The process parameters, derived from the oxide thickness and the substrate doping
    /// where not given, as SPICE derives them.
    fn process(&self) -> Result<Process<R>, ParameterError> {
        let thickness = match self.level {
            ModelLevel::One => self.oxide_thickness,
            ModelLevel::Three => Some(
                self.oxide_thickness
                    .unwrap_or(R::constant(DEFAULT_OXIDE_THICKNESS)),
            ),
        };
        let oxide = thickness
            .map(|thickness| ParameterError::positive("oxide_thickness", thickness))
            .transpose()?
            .map(|thickness| R::constant(OXIDE_PERMITTIVITY) / thickness);
        let mobility = ParameterError::positive("surface_mobility", self.surface_mobility)?;
        let transconductance = self
            .transconductance_parameter
            .or(oxide.map(|oxide| mobility * oxide))
            .unwrap_or(R::constant(2e-5));
        let mut process = Process {
            threshold: self.threshold_voltage.unwrap_or(R::constant(0.0)),
            transconductance,
            body_effect: self.body_effect.unwrap_or(R::constant(0.0)),
            surface_potential: self.surface_potential.unwrap_or(R::constant(0.6)),
            oxide,
        };
        if let Some(doping) = self.substrate_doping {
            let oxide = oxide.ok_or_else(|| {
                ParameterError("substrate_doping needs an oxide_thickness".to_owned())
            })?;
            self.derive(&mut process, doping, oxide)?;
        }
        ParameterError::non_negative("transconductance_parameter", process.transconductance)?;
        ParameterError::non_negative("body_effect", process.body_effect)?;
        ParameterError::positive("surface_potential", process.surface_potential)?;
        Ok(process)
    }

    /// Derives what was not given of `φ`, `γ` and `vto` from the substrate `doping` and the
    /// `oxide` capacitance per area, at the nominal temperature.
    fn derive(&self, process: &mut Process<R>, doping: R, oxide: R) -> Result<(), ParameterError> {
        let nominal = self.nominal_temperature;
        // Silicon's band gap, in electronvolts.
        let gap = band_gap(nominal);
        // Level 1 takes silicon's intrinsic concentration at 300 K; level 3 scales it to
        // the nominal temperature.
        let intrinsic = match self.level {
            ModelLevel::One => R::constant(INTRINSIC_CONCENTRATION),
            ModelLevel::Three => {
                let ratio = nominal / 300.0;
                ratio
                    * INTRINSIC_CONCENTRATION
                    * ratio.sqrt()
                    * (gap * 0.5 * ELEMENTARY_CHARGE / BOLTZMANN * -(nominal.recip() - 1.0 / 300.0))
                        .exp()
            }
        };
        if doping.value() <= intrinsic.value() {
            let (doping, intrinsic) = (doping.value(), intrinsic.value());
            return Err(ParameterError(format!(
                "substrate_doping must exceed silicon's intrinsic concentration, \
                 {intrinsic:e} per cubic meter, not {doping:?}"
            )));
        }
        let sign = self.channel.sign();
        let potential = self.surface_potential.unwrap_or_else(|| {
            (nominal * BOLTZMANN / ELEMENTARY_CHARGE * 2.0 * (doping / intrinsic).ln())
                .max(R::constant(0.1))
        });
        let body_effect = self.body_effect.unwrap_or_else(|| {
            (doping * (2.0 * SILICON_PERMITTIVITY * ELEMENTARY_CHARGE)).sqrt() / oxide
        });
        let gate_work_function = match self.gate_material {
            GateMaterial::Aluminum => R::constant(3.2),
            GateMaterial::OppositeDoping => gap * 0.5 + 3.25 - gap * (sign * 0.5),
            GateMaterial::SameDoping => gap * 0.5 + 3.25 + gap * (sign * 0.5),
        };
        let work_function_difference =
            gate_work_function - (gap * 0.5 + 3.25 + potential * (sign * 0.5));
        let flat_band =
            work_function_difference - self.surface_state_density * ELEMENTARY_CHARGE / oxide;
        process.surface_potential = potential;
        process.body_effect = body_effect;
        process.threshold = self
            .threshold_voltage
            .unwrap_or_else(|| flat_band + (body_effect * potential.sqrt() + potential) * sign);
        Ok(())
    }
}

/// The channel's square law, oriented as an n-channel's.
#[derive(Debug, Clone, Copy)]
struct SquareLaw<R: Real> {
    beta: R,
    lambda: R,
    body_effect: R,
    surface_potential: R,
    /// `√φ`.
    root_potential: R,
    /// The threshold less the body effect at zero bias, `vto − γ · √φ`.
    built_in: R,
}

/// What the square law yields at one bias.
#[derive(Debug, Clone, Copy)]
struct Conduction<R: Real> {
    /// The current from drain to source.
    current: Dual<R, 3>,
    /// The threshold, `von`.
    threshold: Dual<R, 3>,
    /// The saturation voltage, `max(vgst, 0)`.
    saturation: Dual<R, 3>,
    /// Whether the drain is at or above the source; otherwise the two swap.
    forward: bool,
    region: Region,
    /// `vgs − von` and `vds − vdsat`, with the drain at or above the source.
    margins: [f64; 2],
}

impl<R: Real> SquareLaw<R> {
    /// The drain current at the gate-source, drain-source and bulk-source voltages.
    fn evaluate(&self, [vgs, vds, vbs]: [Dual<R, 3>; 3]) -> Conduction<R> {
        let forward = vds.value.value() >= 0.0;
        // With the drain below the source, the drain acts as the source.
        let (vgs, vds, vbs) = if forward {
            (vgs, vds, vbs)
        } else {
            (vgs - vds, -vds, vbs - vds)
        };
        let root = if vbs.value.value() <= 0.0 {
            (-vbs).plus(self.surface_potential).sqrt()
        } else {
            (-vbs.divide(self.root_potential * 2.0))
                .plus(self.root_potential)
                .max(0.0)
        };
        let threshold = root.scale(self.body_effect).plus(self.built_in);
        let vgst = vgs - threshold;
        let saturation = vgst.max(0.0);
        let region = if vgst.value.value() <= 0.0 {
            Region::Off
        } else if vgst.value.value() <= vds.value.value() {
            Region::Saturated
        } else {
            Region::Linear
        };
        let betap = (vds.scale(self.lambda) + 1.0).scale(self.beta);
        let current = match region {
            Region::Saturated => betap * vgst * vgst * 0.5,
            Region::Linear => betap * vds * (vgst - vds * 0.5),
            Region::Off | Region::WeakInversion => Dual::number(0.0),
        };
        Conduction {
            current: if forward { current } else { -current },
            threshold,
            saturation,
            forward,
            region,
            margins: [vgst.value.value(), (vds - saturation).value.value()],
        }
    }
}

/// The channel's law, of the transistor's level.
#[derive(Debug, Clone, Copy)]
enum Law<R: Real> {
    SquareLaw(SquareLaw<R>),
    Semiempirical(Semiempirical<R>),
}

impl<R: Real> Law<R> {
    /// The drain current at the gate-source, drain-source and bulk-source voltages.
    fn evaluate(&self, voltages: [Dual<R, 3>; 3]) -> Conduction<R> {
        match self {
            Self::SquareLaw(law) => law.evaluate(voltages),
            Self::Semiempirical(law) => law.evaluate(voltages),
        }
    }
}

/// A junction between the bulk and the drain or the source.
#[derive(Debug, Clone, Copy)]
struct BulkJunction<R: Real> {
    saturation_current: R,
    /// The thermal voltage at the transistor's temperature.
    thermal_voltage: R,
    critical_voltage: f64,
    bottom: Option<Depletion<R>>,
    sidewall: Option<Depletion<R>>,
}

impl<R: Real> BulkJunction<R> {
    /// The junction's current, with `gmin` across it, and its charge, at `voltage`, the
    /// voltage across it.
    fn evaluate(&self, voltage: Dual<R, 1>, gmin: f64) -> (Dual<R, 1>, Dual<R, 1>) {
        let current = junction_current(voltage, self.saturation_current, self.thermal_voltage)
            + voltage * gmin;
        let charge = depletion_charge(self.bottom.as_ref(), voltage)
            + depletion_charge(self.sidewall.as_ref(), voltage);
        (current, charge)
    }
}

/// The gate oxide, across which Meyer's capacitances lie.
#[derive(Debug, Clone, Copy)]
struct Oxide<R: Real> {
    /// The oxide's whole capacitance, `Cox`.
    capacitance: R,
    surface_potential: R,
}

impl<R: Real> Oxide<R> {
    /// Meyer's capacitances from the gate to the source, the drain and the bulk, at the
    /// gate-source and gate-drain voltages, with the drain at or above the source, and the
    /// channel's threshold and saturation voltage.
    fn meyer(
        &self,
        vgs: Dual<R, 3>,
        vgd: Dual<R, 3>,
        conduction: &Conduction<R>,
    ) -> [Dual<R, 3>; 3] {
        let (oxide, potential) = (self.capacitance, self.surface_potential);
        let zero = Dual::number(0.0);
        let vgst = vgs - conduction.threshold;
        // Accumulation, then depletion: the oxide faces the bulk alone.
        if vgst.value.value() <= -potential.value() {
            return [zero, zero, Dual::constant(oxide)];
        }
        let bulk = if vgst.value.value() <= 0.0 {
            vgst.scale(-oxide / potential)
        } else {
            zero
        };
        if vgst.value.value() <= -potential.value() / 2.0 {
            return [zero, zero, bulk];
        }
        // Inversion: the channel takes two thirds of the oxide, all on the source's side
        // in saturation, and shared with the drain below it.
        let channel = if vgst.value.value() <= 0.0 {
            (vgst.scale(R::constant(2.0) / potential) + 1.0).scale(oxide * 2.0 / 3.0)
        } else {
            Dual::constant(oxide * 2.0 / 3.0)
        };
        let saturation = conduction.saturation.max(MEYER_SATURATION);
        let vds = vgs - vgd;
        if vds.value.value() >= saturation.value.value() {
            return [channel, zero, bulk];
        }
        let sum = saturation * 2.0 - vds;
        let difference = saturation - vds;
        let square = sum * sum;
        [
            channel * (1.0 - difference * difference / square),
            channel * (1.0 - saturation * saturation / square),
            bulk,
        ]
    }
}

/// The displacement current through the whole gate oxide towards one node,
/// `w = Cox · dv/dt`: a local unknown of the transistor's own, which Meyer's current between
/// the gate and the node follows.
///
/// Its pivot is proven: its equation's diagonal entry is 1, a conductance, and the other
/// entries of its column are Meyer's current's derivatives `±C/Cox`, conductances too, at
/// the gate and the node, where Meyer's capacitance `C` never exceeds the oxide's: the
/// bulk's at most `Cox`, the channel's at most two thirds of it ([`Oxide::meyer`]). So the
/// layout claims it [pivoted](Layout::pivoted), and the solver eliminates it before it
/// factors the transistor's nodes, as ngspice's matrix holds them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Displacement {
    unknown: usize,
    /// The local entries at (unknown, unknown), (unknown, gate), (unknown, node),
    /// (gate, unknown) and (node, unknown).
    entries: [usize; 5],
}

impl Displacement {
    /// Claims a local unknown labelled `label` for the displacement towards the local
    /// unknown `node` from `gate`.
    fn new(layout: &mut Layout, label: &'static str, gate: usize, node: usize) -> Self {
        let unknown = layout.pivoted(label, Nature::Displacement);
        Self {
            unknown,
            entries: [
                layout.entry(unknown, unknown),
                layout.entry(unknown, gate),
                layout.entry(unknown, node),
                layout.entry(gate, unknown),
                layout.entry(node, unknown),
            ],
        }
    }

    /// Adds the displacement's equation, `w − Cox · d(v_gate − v_node)/dt = 0`, for the
    /// `oxide`'s whole capacitance, and the derivatives with respect to `w` of Meyer's
    /// current from the gate to the node, `C · w / Cox`, for the Meyer `capacitance`, the
    /// node at index `index` among the stamped `nodes`. Returns `w`.
    fn add<R: Real>(
        &self,
        (nodes, index): (&Nodes<4>, usize),
        (oxide, capacitance): (R, R),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) -> R {
        let row = self.unknown;
        let [own, gate, node, from_gate, from_node] = self.entries;
        let displacement = at.unknown::<R>(row);
        let voltage = nodes.moving::<R>(at, GATE) - nodes.moving::<R>(at, index);
        stamp.add_residual(row, displacement);
        stamp.add_charge(row, -oxide * voltage);
        stamp.add_conductance(own, R::constant(1.0));
        stamp.add_capacitance(gate, -oxide);
        stamp.add_capacitance(node, oxide);
        stamp.add_conductance(from_gate, capacitance / oxide);
        stamp.add_conductance(from_node, -capacitance / oxide);
        displacement
    }
}

/// The gate-source, drain-source and bulk-source voltages of one evaluation, oriented as an
/// n-channel's.
type Bias = parts::Bias<3>;

/// The voltage of the stamped node `plus` over `minus` in `bias`, as evaluated and as far
/// as the applied one lies beyond it.
fn across(bias: &Bias, (plus, minus): (usize, usize)) -> parts::Bias<1> {
    // Each voltage of `bias` is a node's over the inner source.
    let over_source = |values: [f64; 3], node: usize| {
        VARIABLES
            .iter()
            .zip(values)
            .find_map(|(&(variable, _), value)| (variable == node).then_some(value))
            .unwrap_or(0.0)
    };
    let between = |values: [f64; 3]| over_source(values, plus) - over_source(values, minus);
    parts::Bias {
        voltages: [between(bias.voltages)],
        beyond: [between(bias.beyond)],
    }
}

impl Kernel for Mosfet {
    const NAME: &'static str = "mosfet";
    type Parameters<R: Real> = MosfetParameters<R>;
    type Shape = MosfetShape;
    type Prepared<R: Real> = PreparedMosfet<R>;
    type Memory = MosfetMemory;

    fn interface<R: Real>(_: &MosfetParameters<R>) -> Interface {
        Interface::new(
            &["drain", "gate", "source", "bulk"],
            &[
                Branch::new(0, 2, BranchKind::Conductive),
                Branch::new(3, 0, BranchKind::Conductive),
                Branch::new(3, 2, BranchKind::Conductive),
                Branch::new(1, 2, BranchKind::Capacitive),
                Branch::new(1, 0, BranchKind::Capacitive),
                Branch::new(1, 3, BranchKind::Capacitive),
            ],
        )
    }

    fn prepare<R: Real>(
        parameters: &MosfetParameters<R>,
        temperature: R,
    ) -> Result<PreparedMosfet<R>, ParameterError> {
        let Scaled {
            channel,
            junctions,
            surface_potential,
            threshold,
            temperature,
        } = parameters.scaled(temperature)?;
        let process = parameters.process()?;
        let geometry = Geometry::new(parameters)?;
        let [drain_resistance, source_resistance] = parameters.series_resistances()?;
        let mut overlaps = [R::constant(0.0); 3];
        for (overlap, (name, per_length, extent)) in overlaps.iter_mut().zip([
            (
                "gate_source_overlap",
                parameters.gate_source_overlap,
                geometry.width,
            ),
            (
                "gate_drain_overlap",
                parameters.gate_drain_overlap,
                geometry.width,
            ),
            (
                "gate_bulk_overlap",
                parameters.gate_bulk_overlap,
                geometry.lengths,
            ),
        ]) {
            *overlap = ParameterError::non_negative(name, per_length)? * extent;
        }
        let oxide = process.oxide.map(|per_area| Oxide {
            capacitance: per_area * geometry.area,
            surface_potential,
        });
        let oxide_per_area = process
            .oxide
            .map_or(OXIDE_PERMITTIVITY / DEFAULT_OXIDE_THICKNESS, |per_area| {
                per_area.value()
            });
        // The multiplier's copies are transistors of their own in parallel, each carrying
        // its share of the drain current and the flicker noise of its own gate, which add:
        // m · KF (Id/m)^AF / (f W L Cox²), as SPICE defines the multiplier.
        let copies = geometry.multiplier.value();
        let gate = geometry.area.value() / copies;
        let exponent = parameters.flicker_exponent.value();
        let flicker = Flicker::new(parameters.flicker_coefficient, parameters.flicker_exponent)?
            .scaled(copies.powf(1.0 - exponent) / (gate * oxide_per_area * oxide_per_area));
        Ok(PreparedMosfet {
            sign: parameters.channel.sign(),
            series: [
                series_value(drain_resistance),
                series_value(source_resistance),
            ],
            channel,
            junctions,
            overlaps,
            oxide,
            threshold,
            temperature,
            flicker,
        })
    }

    fn layout<R: Real>(
        parameters: &MosfetParameters<R>,
        _: &PreparedMosfet<R>,
        layout: &mut Layout,
    ) -> Result<MosfetShape, ParameterError> {
        let [drain_resistance, source_resistance] = parameters.series_resistances()?;
        let [drain, gate, source, bulk] = [0, 1, 2, 3];
        let (drain_series, inner_drain) =
            Series::behind(layout, drain, drain_resistance, ("drain", "drain_current"));
        let (source_series, inner_source) = Series::behind(
            layout,
            source,
            source_resistance,
            ("source", "source_current"),
        );
        let stamp = Nodes::new(layout, [gate, bulk, inner_drain, inner_source]);
        let displacements = parameters.has_oxide().then(|| {
            let mut displacement =
                |label, index| Displacement::new(layout, label, gate, stamp.node(index));
            let [source, drain, bulk] = DISPLACED;
            [
                displacement("gate_source_displacement", source),
                displacement("gate_drain_displacement", drain),
                displacement("gate_bulk_displacement", bulk),
            ]
        });
        Ok(MosfetShape {
            stamp,
            drain_series,
            source_series,
            displacements,
        })
    }

    fn evaluate<R: Real>(
        shape: &MosfetShape,
        prepared: &PreparedMosfet<R>,
        memory: &mut MosfetMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let [drain_series, source_series] = prepared.series;
        shape.drain_series.add(drain_series, at, stamp);
        shape.source_series.add(source_series, at, stamp);
        let bias = prepared.bias(shape, memory, at, stamp);
        let sign = prepared.sign;
        let [vgs, vds, vbs] = Dual::variables(bias.moving(&shape.stamp, at, (sign, VARIABLES)));
        let conduction = prepared.channel.evaluate([vgs, vds, vbs]);
        let threshold = conduction.threshold.value.value();
        memory.threshold = Some(threshold);
        let [overdrive, margin] = conduction.margins;
        memory.variables = [
            sign * threshold,
            sign * conduction.saturation.value.value(),
            0.0,
            0.0,
            0.0,
            f64::from(conduction.region as u8),
            overdrive,
            margin,
        ];
        let added = Oriented {
            nodes: &shape.stamp,
            sign,
            bias: &bias,
            variables: VARIABLES,
        };
        // Every derivative among the stamped nodes is summed here, and written once.
        let mut derivatives = Derivatives::new();
        let sums = &mut derivatives;
        added.add_current(stamp, sums, (INNER_DRAIN, INNER_SOURCE), conduction.current);

        // A junction's current and charge, and an overlap's charge, follow the voltage
        // across it alone, of the first node of its pair over the second.
        let [drain, source] = &prepared.junctions;
        for (pair, junction) in [((BULK, INNER_DRAIN), drain), ((BULK, INNER_SOURCE), source)] {
            let across = across(&bias, pair);
            let added = Oriented {
                nodes: &shape.stamp,
                sign,
                bias: &across,
                variables: [pair],
            };
            let [voltage] = Dual::variables(across.moving(&shape.stamp, at, (sign, [pair])));
            let (current, charge) = junction.evaluate(voltage, at.gmin);
            added.add_current(stamp, sums, pair, current);
            added.add_charge(stamp, sums, pair, charge);
        }
        let [source_overlap, drain_overlap, bulk_overlap] = prepared.overlaps;
        for (pair, overlap) in [
            ((GATE, INNER_SOURCE), source_overlap),
            ((GATE, INNER_DRAIN), drain_overlap),
            ((GATE, BULK), bulk_overlap),
        ] {
            let across = across(&bias, pair);
            let added = Oriented {
                nodes: &shape.stamp,
                sign,
                bias: &across,
                variables: [pair],
            };
            let [voltage] = Dual::variables(across.moving(&shape.stamp, at, (sign, [pair])));
            added.add_charge(stamp, sums, pair, voltage.scale(overlap));
        }

        let vgd = vgs - vds;
        if let (Some(oxide), Some(displacements)) = (&prepared.oxide, &shape.displacements) {
            let capacitances = if conduction.forward {
                oxide.meyer(vgs, vgd, &conduction)
            } else {
                let [drain, source, bulk] = oxide.meyer(vgd, vgs, &conduction);
                [source, drain, bulk]
            };
            for (variable, capacitance) in memory.variables.iter_mut().skip(2).zip(&capacitances) {
                *variable = capacitance.value.value();
            }
            let whole = oxide.capacitance;
            let towards = displacements.iter().zip(DISPLACED).zip(capacitances);
            for ((displacement, node), capacitance) in towards {
                let current =
                    displacement.add((&shape.stamp, node), (whole, capacitance.value), at, stamp);
                let meyer = capacitance.scale(current * sign / whole);
                added.add_current(stamp, sums, (GATE, node), meyer);
            }
        }
        derivatives.stamp(&shape.stamp, stamp);
    }

    /// The prepared threshold, which a transistor's first step is limited from, moves with
    /// its temperature.
    fn reheat(memory: &mut MosfetMemory) {
        memory.threshold = None;
    }

    fn variable_names() -> &'static [&'static str] {
        &VARIABLE_NAMES
    }

    fn variables(memory: &MosfetMemory, values: &mut [f64]) {
        parts::write_variables(&memory.variables, values);
    }

    fn regions() -> Regions {
        REGIONS
    }

    /// The thermal noise of the series resistances and of the channel, `8kT·gm / 3`, and
    /// the drain current's flicker noise, `KF · |Id|^AF / (f · W · L · Cox²)` over the
    /// effective gate, every parallel copy's width counted, as ngspice models them, with
    /// SPICE's oxide thickness of 0.1 µm where none is given.
    fn noise<R: Real>(
        shape: &MosfetShape,
        prepared: &PreparedMosfet<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        let temperature = prepared.temperature;
        let [drain_series, source_series] = prepared.series;
        shape
            .drain_series
            .noise(drain_series, ("drain thermal", temperature), sources);
        shape
            .source_series
            .noise(source_series, ("source thermal", temperature), sources);
        let stamp = &shape.stamp;
        let voltages = VARIABLES.map(|(plus, minus)| {
            prepared.sign * (stamp.voltage(at, plus) - stamp.voltage(at, minus))
        });
        let current = prepared
            .channel
            .evaluate(Dual::variables(voltages.map(R::constant)))
            .current;
        let channel = [stamp.equation(INNER_DRAIN), stamp.equation(INNER_SOURCE)];
        let [transconductance, _, _] = current.gradient;
        let conductance = 2.0 / 3.0 * transconductance.value();
        sources.push(NoiseSource::thermal(
            "channel thermal",
            channel,
            conductance,
            temperature,
        ));
        prepared
            .flicker
            .add(channel, current.value.value(), sources);
    }
}

struct Scaled<R: Real> {
    channel: Law<R>,
    junctions: [BulkJunction<R>; 2],
    /// The surface potential, which Meyer's capacitances read.
    surface_potential: R,
    /// The threshold, signed as the channel conducts, from which steps are limited.
    threshold: f64,
    temperature: f64,
}

impl<R: Real> MosfetParameters<R> {
    /// What the transistor's `temperature`, in kelvins, sets, once each parameter it
    /// reads is checked.
    fn scaled(&self, temperature: R) -> Result<Scaled<R>, ParameterError> {
        let temperatures = Temperatures::of_junction(temperature, self.nominal_temperature)?;
        let sign = self.channel.sign();
        let process = self.process()?.at(temperatures, sign)?;
        let geometry = Geometry::new(self)?;
        Ok(Scaled {
            channel: self.law(&process, &geometry, temperatures)?,
            junctions: self.junctions(&geometry, temperatures)?,
            surface_potential: process.surface_potential,
            threshold: sign * process.threshold.value(),
            temperature: temperatures.actual.value(),
        })
    }
}

/// A transistor's extent: its effective length, and its width, the sum of its effective
/// lengths and its gate's area with every parallel copy counted.
#[derive(Debug, Clone, Copy)]
struct Geometry<R: Real> {
    multiplier: R,
    length: R,
    /// The effective width of one copy.
    single_width: R,
    width: R,
    lengths: R,
    area: R,
}

impl<R: Real> Geometry<R> {
    fn new(parameters: &MosfetParameters<R>) -> Result<Self, ParameterError> {
        let multiplier = ParameterError::positive("multiplier", parameters.multiplier)?;
        let width = ParameterError::positive("width", parameters.width)?;
        let length = ParameterError::positive("length", parameters.length)?;
        // A negative lateral diffusion lengthens the channel past its drawn length, as
        // ngspice 47 takes it; only the effective length must stay positive.
        let diffusion = parameters.lateral_diffusion;
        let narrowing = ParameterError::non_negative(
            "width_narrowing",
            parameters.width_narrowing.unwrap_or(R::constant(0.0)),
        )?;
        let effective =
            length - diffusion * 2.0 + parameters.length_adjustment.unwrap_or(R::constant(0.0));
        if effective.value() <= 0.0 {
            let effective = effective.value();
            return Err(ParameterError(format!(
                "the effective length, length − 2 · lateral_diffusion + length_adjustment, \
                 must be positive, not {effective:?}"
            )));
        }
        let effective_width =
            width - narrowing * 2.0 + parameters.width_adjustment.unwrap_or(R::constant(0.0));
        if effective_width.value() <= 0.0 {
            let effective_width = effective_width.value();
            return Err(ParameterError(format!(
                "the effective width, width − 2 · width_narrowing + width_adjustment, must \
                 be positive, not {effective_width:?}"
            )));
        }
        Ok(Self {
            multiplier,
            length: effective,
            single_width: effective_width,
            width: multiplier * effective_width,
            lengths: multiplier * effective,
            area: effective * multiplier * effective_width,
        })
    }
}

impl<R: Real> MosfetParameters<R> {
    /// The bulk-drain and bulk-source junctions at `temperatures`.
    fn junctions(
        &self,
        geometry: &Geometry<R>,
        temperatures: Temperatures<R>,
    ) -> Result<[BulkJunction<R>; 2], ParameterError> {
        let multiplier = geometry.multiplier;
        let saturation = ParameterError::positive("saturation_current", self.saturation_current)?;
        let density = ParameterError::non_negative(
            "saturation_current_density",
            self.saturation_current_density,
        )?;
        let potential = ParameterError::positive("junction_potential", self.junction_potential)?;
        let bottom_grading = below_one("bottom_grading", self.bottom_grading)?;
        let default_sidewall = match self.level {
            ModelLevel::One => R::constant(0.5),
            ModelLevel::Three => R::constant(0.33),
        };
        let sidewall_grading = below_one(
            "sidewall_grading",
            self.sidewall_grading.unwrap_or(default_sidewall),
        )?;
        let coefficient = below_one("depletion_coefficient", self.depletion_coefficient)?;
        let bottom_capacitance =
            ParameterError::non_negative("bottom_capacitance", self.bottom_capacitance)?;
        let sidewall_capacitance =
            ParameterError::non_negative("sidewall_capacitance", self.sidewall_capacitance)?;
        let by_density = density.value() > 0.0
            && self.drain_area.value() > 0.0
            && self.source_area.value() > 0.0;
        let vt = temperatures.thermal_voltage();
        let nominal = temperatures.nominal;
        let scaling = (band_gap(nominal) / (nominal * BOLTZMANN / ELEMENTARY_CHARGE)
            - band_gap(temperatures.actual) / vt)
            .exp();
        let (scaled_potential, bottom_scaling) =
            temperatures.junction(potential, R::constant(1.0), bottom_grading);
        let (_, sidewall_scaling) =
            temperatures.junction(potential, R::constant(1.0), sidewall_grading);
        let junction = |name, capacitance: Option<R>, area: R, perimeter: R| {
            ParameterError::non_negative(&format!("{name}_area"), area)?;
            ParameterError::non_negative(&format!("{name}_perimeter"), perimeter)?;
            let saturation_current = multiplier
                * scaling
                * if by_density {
                    density * area
                } else {
                    saturation
                };
            let bottom = match capacitance {
                Some(capacitance) => {
                    ParameterError::non_negative(&format!("bulk_{name}_capacitance"), capacitance)?
                }
                None => bottom_capacitance * area,
            };
            Ok(BulkJunction {
                saturation_current,
                thermal_voltage: vt,
                critical_voltage: critical_voltage(saturation_current.value(), vt.value()),
                bottom: Depletion::new(
                    multiplier * bottom * bottom_scaling,
                    scaled_potential,
                    bottom_grading,
                    coefficient,
                ),
                sidewall: Depletion::new(
                    multiplier * sidewall_capacitance * perimeter * sidewall_scaling,
                    scaled_potential,
                    sidewall_grading,
                    coefficient,
                ),
            })
        };
        Ok([
            junction(
                "drain",
                self.bulk_drain_capacitance,
                self.drain_area,
                self.drain_perimeter,
            )?,
            junction(
                "source",
                self.bulk_source_capacitance,
                self.source_area,
                self.source_perimeter,
            )?,
        ])
    }

    /// The channel's law, of the transistor's level, with `process` at `temperatures`,
    /// once each of its parameters is checked.
    fn law(
        &self,
        process: &Process<R>,
        geometry: &Geometry<R>,
        temperatures: Temperatures<R>,
    ) -> Result<Law<R>, ParameterError> {
        let sign = self.channel.sign();
        let root_potential = process.surface_potential.sqrt();
        let built_in =
            |shift: R| (process.threshold + shift) * sign - process.body_effect * root_potential;
        let beta = process.transconductance * geometry.width / geometry.length;
        let level_three = [
            ("max_drift_velocity", self.max_drift_velocity),
            ("junction_depth", self.junction_depth),
            (
                "fast_surface_state_density",
                self.fast_surface_state_density,
            ),
            ("static_feedback", self.static_feedback),
            ("narrow_width_factor", self.narrow_width_factor),
            ("mobility_modulation", self.mobility_modulation),
            ("saturation_field_factor", self.saturation_field_factor),
            ("width_narrowing", self.width_narrowing),
            ("length_adjustment", self.length_adjustment),
            ("width_adjustment", self.width_adjustment),
            ("threshold_adjustment", self.threshold_adjustment),
        ];
        let ModelLevel::Three = self.level else {
            if let Some((name, _)) = level_three.iter().find(|(_, value)| value.is_some()) {
                return Err(ParameterError(format!("{name} is level 3's")));
            }
            return Ok(Law::SquareLaw(SquareLaw {
                beta,
                lambda: self.channel_length_modulation.unwrap_or(R::constant(0.0)),
                body_effect: process.body_effect,
                surface_potential: process.surface_potential,
                root_potential,
                built_in: built_in(R::constant(0.0)),
            }));
        };
        if self.channel_length_modulation.is_some() {
            return Err(ParameterError(
                "channel_length_modulation is level 1's".to_owned(),
            ));
        }
        let non_negative = |name, value: Option<R>| {
            ParameterError::non_negative(name, value.unwrap_or(R::constant(0.0)))
        };
        let oxide = R::constant(OXIDE_PERMITTIVITY)
            / self
                .oxide_thickness
                .unwrap_or(R::constant(DEFAULT_OXIDE_THICKNESS));
        let length = geometry.length;
        Ok(Law::Semiempirical(Semiempirical {
            beta,
            length,
            surface_potential: process.surface_potential,
            body_effect: process.body_effect,
            built_in: built_in(self.threshold_adjustment.unwrap_or(R::constant(0.0))),
            static_feedback: self.static_feedback.unwrap_or(R::constant(0.0)) * 8.15e-22
                / (oxide * length * length * length),
            narrow: self.narrow_width_factor.unwrap_or(R::constant(0.0))
                * 0.5
                * std::f64::consts::PI
                * SILICON_PERMITTIVITY
                / (oxide * geometry.single_width),
            mobility_modulation: non_negative("mobility_modulation", self.mobility_modulation)?,
            saturation_field_factor: ParameterError::non_negative(
                "saturation_field_factor",
                self.saturation_field_factor.unwrap_or(R::constant(0.2)),
            )?,
            alpha: self.substrate_doping.map_or(R::constant(0.0), |doping| {
                R::constant(2.0 * SILICON_PERMITTIVITY) / (doping * ELEMENTARY_CHARGE)
            }),
            junction_depth: non_negative("junction_depth", self.junction_depth)?,
            lateral_diffusion: self.lateral_diffusion,
            max_drift_velocity: non_negative("max_drift_velocity", self.max_drift_velocity)?,
            mobility: self.surface_mobility / temperatures.ratio().powf(1.5),
            thermal_voltage: temperatures.thermal_voltage(),
            surface_states: non_negative(
                "fast_surface_state_density",
                self.fast_surface_state_density,
            )? * ELEMENTARY_CHARGE
                / oxide,
        }))
    }

    /// The drain and source resistances, given or from the sheet resistance, of all the
    /// parallel copies together.
    fn series_resistances(&self) -> Result<[R; 2], ParameterError> {
        let sheet = ParameterError::non_negative("sheet_resistance", self.sheet_resistance)?;
        let resistance = |name: &str, given: Option<R>, squares: R| {
            ParameterError::non_negative(&format!("{name}_squares"), squares)?;
            let resistance = match given {
                Some(resistance) => {
                    ParameterError::non_negative(&format!("{name}_resistance"), resistance)?
                }
                None => sheet * squares,
            };
            Ok::<_, ParameterError>(resistance / self.multiplier)
        };
        Ok([
            resistance("drain", self.drain_resistance, self.drain_squares)?,
            resistance("source", self.source_resistance, self.source_squares)?,
        ])
    }
}

impl<R: Real> MosfetParameters<R> {
    /// Whether the transistor's gate has an oxide, and so Meyer's capacitances: given its
    /// thickness, or always at level 3, which takes SPICE's by default.
    fn has_oxide(&self) -> bool {
        match self.level {
            ModelLevel::One => self.oxide_thickness.is_some(),
            ModelLevel::Three => true,
        }
    }
}

impl<R: Real> PreparedMosfet<R> {
    /// The terminal voltages at `at`, limited from the last evaluation's, which `memory`
    /// keeps and takes the new ones.
    fn bias(
        &self,
        shape: &MosfetShape,
        memory: &mut MosfetMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) -> Bias {
        let nodes = &shape.stamp;
        let applied = VARIABLES
            .map(|(plus, minus)| self.sign * (nodes.voltage(at, plus) - nodes.voltage(at, minus)));
        let limited = self.limit(applied, memory).filter(|_| at.limiting);
        if limited.is_some() {
            stamp.mark_limited();
        }
        let voltages = limited.unwrap_or(applied);
        memory.last = voltages;
        Bias::new(applied, voltages)
    }

    /// The `applied` gate-source, drain-source and bulk-source voltages limited from the
    /// last evaluation's, which `memory` keeps, as SPICE limits them: the gate's step around
    /// the threshold and the drain's, from whichever end acts as the source, then the step
    /// of the junction with it. Returns `None` when no step needs limiting.
    fn limit(&self, applied: [f64; 3], memory: &MosfetMemory) -> Option<[f64; 3]> {
        let [applied_gate, applied_drain, applied_vbs] = applied;
        let [previous_gate, previous_drain, previous_bulk] = memory.last;
        let threshold = memory.threshold.unwrap_or(self.threshold);
        let ([vgs, vds], mut limited) = limit_channel(
            [applied_gate, applied_drain],
            [previous_gate, previous_drain],
            threshold,
        );
        let mut apply = |value: f64, limit: Option<f64>| {
            limited |= limit.is_some();
            limit.unwrap_or(value)
        };
        let [drain, source] = &self.junctions;
        let vbs = if vds >= 0.0 {
            let limit = limit_junction(
                applied_vbs,
                previous_bulk,
                source.thermal_voltage.value(),
                source.critical_voltage,
            );
            apply(applied_vbs, limit)
        } else {
            let (vbd, previous_junction) =
                (applied_vbs - applied_drain, previous_bulk - previous_drain);
            let limit = limit_junction(
                vbd,
                previous_junction,
                drain.thermal_voltage.value(),
                drain.critical_voltage,
            );
            apply(vbd, limit) + vds
        };
        limited.then_some([vgs, vds, vbs])
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::Differentiable;
    use crate::testing::{
        Bench, assert_close, assert_jacobians_match, assert_mirrors,
        assert_parameter_gradients_match, assert_switches_where_margin_vanishes, source, thermal,
    };

    const CHARGED: &str = r#"{"threshold_voltage": 0.8, "transconductance_parameter": 1e-4,
        "body_effect": 0.5, "channel_length_modulation": 0.02, "width": 10e-6,
        "length": 2e-6, "lateral_diffusion": 1e-7, "oxide_thickness": 2e-8,
        "drain_area": 2e-11, "source_area": 3e-11, "drain_perimeter": 2e-5,
        "source_perimeter": 3e-5, "bottom_capacitance": 2e-4,
        "sidewall_capacitance": 5e-10, "gate_source_overlap": 4e-10,
        "gate_drain_overlap": 3e-10, "gate_bulk_overlap": 2e-10}"#;

    /// The same transistor with a p-channel.
    fn p_channel(parameters: &str) -> String {
        parameters
            .replace(
                r#""threshold_voltage": 0.8"#,
                r#""threshold_voltage": -0.8"#,
            )
            .replace('{', r#"{"channel": "p", "#)
    }

    /// The drain current of a transistor with `β = 1e-4` and `vto = 1` at `vgs` and `vds`,
    /// with the source grounded and the bulk well below both, its junctions' leakage
    /// negligible.
    fn drain_current(vgs: f64, vds: f64) -> f64 {
        let parameters = r#"{"threshold_voltage": 1, "transconductance_parameter": 1e-4,
            "saturation_current": 1e-30}"#;
        Bench::new::<Mosfet>(parameters)
            .settle(&[vds, vgs, 0.0, -5.0])
            .residual[0]
    }

    #[test]
    fn follows_the_square_law() {
        // Saturation: β · vgst² / 2; below it: β · vds · (vgst − vds / 2).
        let saturated = drain_current(2.0, 5.0);
        assert!((saturated - 5e-5).abs() < 1e-17, "{saturated}");
        let linear = drain_current(2.0, 0.5);
        assert!((linear - 3.75e-5).abs() < 1e-17, "{linear}");
        let reversed = drain_current(1.5, -0.5);
        let mirrored = -drain_current(2.0, 0.5);
        assert!((reversed - mirrored).abs() < 1e-17, "{reversed} {mirrored}");
        assert!(drain_current(0.9, 5.0).abs() < 1e-17);
    }

    #[test]
    fn the_body_effect_raises_the_threshold() {
        let parameters = r#"{"threshold_voltage": 1, "body_effect": 0.5,
            "surface_potential": 0.7, "saturation_current": 1e-30}"#;
        let mut bench = Bench::new::<Mosfet>(parameters);
        let current = bench.settle(&[5.0, 2.0, 0.0, -1.0]).residual[0];
        let threshold = 1.0 + 0.5 * (1.7_f64.sqrt() - 0.7_f64.sqrt());
        let expected = 2e-5 / 2.0 * (2.0 - threshold) * (2.0 - threshold);
        assert!((current - expected).abs() < 1e-17, "{current} {expected}");
    }

    #[test]
    fn reports_its_threshold_saturation_and_meyer_capacitances() {
        let parameters = r#"{"threshold_voltage": 1, "body_effect": 0.5,
            "surface_potential": 0.7, "saturation_current": 1e-30}"#;
        let mut bench = Bench::new::<Mosfet>(parameters);
        bench.settle(&[5.0, 2.0, 0.0, -1.0]);
        let names = bench.variable_names();
        let values = bench.variables();
        let threshold = 1.0 + 0.5 * (1.7_f64.sqrt() - 0.7_f64.sqrt());
        assert_eq!(names[..2], ["threshold_voltage", "saturation_voltage"]);
        assert!((values[0] - threshold).abs() < 1e-15);
        assert!((values[1] - (2.0 - threshold)).abs() < 1e-15);
        assert_eq!(values[2..5], [0.0; 3], "no oxide, no Meyer capacitance");

        let mut n = Bench::new::<Mosfet>(CHARGED);
        let mut p = Bench::new::<Mosfet>(&p_channel(CHARGED));
        let x = [3.0, 2.0, 0.2, -0.5, 1e-4, -2e-4, 3e-4];
        let mirrored: Vec<f64> = x.iter().map(|value| -value).collect();
        n.settle(&x);
        p.settle(&mirrored);
        let (n_values, p_values) = (n.variables(), p.variables());
        assert!(n_values[0] > 0.0 && n_values[2] > 0.0, "{n_values:?}");
        let polar = [true, true, false, false, false, false, false, false];
        for (index, ((n_value, p_value), polar)) in
            n_values.iter().zip(&p_values).zip(polar).enumerate()
        {
            let mirrored = if polar { -p_value } else { *p_value };
            assert!(
                (n_value - mirrored).abs() <= 1e-15 * n_value.abs(),
                "{index}"
            );
        }
    }

    /// A level 1 transistor is off at or below its threshold, saturated with its drain at
    /// or past `vgs − vto`, and linear below it, the drain and the source swapping where the
    /// drain lies below; its margins are `vgs − vto` and `vds − max(vgs − vto, 0)`, which
    /// vanish where `mos1load.c` switches, a p-channel's equal to an n-channel's at the
    /// voltages reversed.
    #[test]
    fn reports_its_region_and_margins() {
        let parameters = r#"{"threshold_voltage": 1, "saturation_current": 1e-30}"#;
        let mut bench = Bench::new::<Mosfet>(parameters);
        let mut p = Bench::new::<Mosfet>(&parameters.replace(": 1,", ": -1, \"channel\": \"p\","));
        // The drain, the gate, the source and the bulk.
        for (x, region, margins) in [
            ([3.0, 0.5, 0.0, 0.0], "off", [-0.5, 3.0]),
            ([3.0, 2.0, 0.0, 0.0], "saturated", [1.0, 2.0]),
            ([0.5, 3.0, 0.0, 0.0], "linear", [2.0, -1.5]),
            ([0.0, 5.0, 3.0, 0.0], "linear", [4.0, -1.0]),
        ] {
            bench.settle(&x);
            assert_eq!(bench.region(), (region, margins.to_vec()), "{x:?}");
            p.settle(&x.map(|value| -value));
            assert_eq!(p.region(), (region, margins.to_vec()), "{x:?}");
        }
        let switches = [
            ([3.0, 0.5, 0.0, 0.0], [3.0, 1.5, 0.0, 0.0], OVERDRIVE),
            (
                [0.5, 2.0, 0.0, 0.0],
                [2.0, 2.0, 0.0, 0.0],
                SATURATION_MARGIN,
            ),
        ];
        for (from, to, margin) in switches {
            assert_switches_where_margin_vanishes(&mut bench, (&from, &to), margin);
        }
    }

    /// A level 3 transistor is off at or below its threshold, or in weak inversion below it
    /// given fast surface states, and linear or saturated above it, each region switching
    /// where `mos3load.c` does, where its margin vanishes.
    #[test]
    fn reports_its_region_at_level_3() {
        let parameters = r#"{"level": 3, "threshold_voltage": 1, "saturation_current": 1e-30}"#;
        let weak = parameters.replace('}', r#", "fast_surface_state_density": 1e15}"#);
        for (parameters, below) in [(parameters, "off"), (weak.as_str(), "weak_inversion")] {
            let mut bench = Bench::new::<Mosfet>(parameters);
            for (x, region) in [
                ([3.0, 0.5, 0.0, 0.0], below),
                ([3.0, 2.0, 0.0, 0.0], "saturated"),
                ([0.2, 3.0, 0.0, 0.0], "linear"),
            ] {
                bench.settle(&x);
                assert_eq!(bench.region().0, region, "{parameters} at {x:?}");
            }
            let switches = [
                ([3.0, 0.5, 0.0, 0.0], [3.0, 2.0, 0.0, 0.0], OVERDRIVE),
                (
                    [0.2, 3.0, 0.0, 0.0],
                    [3.0, 3.0, 0.0, 0.0],
                    SATURATION_MARGIN,
                ),
            ];
            for (from, to, margin) in switches {
                assert_switches_where_margin_vanishes(&mut bench, (&from, &to), margin);
            }
        }
    }

    #[test]
    fn a_p_channel_mirrors_an_n_channel() {
        let x = [3.0, 2.0, 0.2, -0.5, 1e-4, -2e-4, 3e-4];
        assert_mirrors(&Differentiable::<Mosfet>::new(), CHARGED, &x);
    }

    #[test]
    fn meyers_capacitances_share_the_oxide() {
        let mut bench = Bench::new::<Mosfet>(CHARGED);
        // The gate's derivatives with respect to the three displacement currents: Meyer's
        // capacitances as fractions of the oxide's.
        let mut shares = |gate: f64, drain: f64| {
            let evaluation = bench.settle(&[drain, gate, 0.0, 0.0, 0.0, 0.0, 0.0]);
            [4, 5, 6].map(|column| evaluation.conductance[1][column])
        };
        let [source, drain, bulk] = shares(-2.0, 1.0);
        assert!(source == 0.0 && drain == 0.0 && (bulk - 1.0).abs() < 1e-15);
        let [source, drain, bulk] = shares(3.0, 5.0);
        assert!((source - 2.0 / 3.0).abs() < 1e-15 && drain == 0.0 && bulk == 0.0);
        let [source, drain, _] = shares(3.0, 0.0);
        assert!((source - 0.5).abs() < 1e-15 && (drain - 0.5).abs() < 1e-15);
    }

    #[test]
    fn unknowns_follow_the_resistances_and_the_oxide() {
        let parameters = r#"{"drain_resistance": 10, "sheet_resistance": 20,
            "source_squares": 0.5, "oxide_thickness": 1e-7}"#;
        let bench = Bench::new::<Mosfet>(parameters);
        assert_eq!(
            bench.unknowns(),
            [
                "n0",
                "n1",
                "n2",
                "n3",
                "device:drain",
                "device:source",
                "device:gate_source_displacement",
                "device:gate_drain_displacement",
                "device:gate_bulk_displacement"
            ]
        );
        assert_eq!(Bench::new::<Mosfet>("{}").unknowns().len(), 4);
    }

    #[test]
    fn rejects_inconsistent_parameters() {
        let error = |parameters| Bench::try_new::<Mosfet>(parameters).unwrap_err().0;
        assert_eq!(
            error(r#"{"substrate_doping": 1e22}"#),
            "substrate_doping needs an oxide_thickness"
        );
        assert_eq!(
            error(r#"{"length": 1e-6, "lateral_diffusion": 5e-7}"#),
            "the effective length, length − 2 · lateral_diffusion + length_adjustment, must \
             be positive, not 0.0"
        );
        assert_eq!(
            error(r#"{"mobility_modulation": 0.1}"#),
            "mobility_modulation is level 3's"
        );
        assert_eq!(
            error(r#"{"level": 3, "channel_length_modulation": 0.01}"#),
            "channel_length_modulation is level 1's"
        );
        assert_eq!(
            error(r#"{"bottom_grading": 1}"#),
            "bottom_grading must be at least 0 and below 1, not 1.0"
        );
    }

    #[test]
    fn parameter_gradients_are_exact() {
        let temperature = r#""nominal_temperature": 290, "junction_potential": 0.75,
            "saturation_current": 2e-14, "bottom_grading": 0.45, "sidewall_grading": 0.35,
            "depletion_coefficient": 0.45, "multiplier": 2, "#;
        let level_one = CHARGED.replace('{', &format!("{{{temperature}"));
        let resistive =
            level_one.replace('{', r#"{"drain_resistance": 50, "source_resistance": 20, "#);
        let derived = r#"{"substrate_doping": 1e22, "oxide_thickness": 2e-8,
            "surface_state_density": 1e14, "surface_mobility": 0.05, "width": 10e-6,
            "length": 2e-6, "sheet_resistance": 10, "drain_squares": 2, "source_squares": 3,
            "saturation_current_density": 1e-4, "drain_area": 2e-11, "source_area": 3e-11,
            "nominal_temperature": 290}"#;
        let level_three = r#"{"level": 3, "threshold_voltage": 0.7, "body_effect": 0.4,
            "width": 20e-6, "length": 2e-6, "lateral_diffusion": 1e-7,
            "oxide_thickness": 3e-8, "substrate_doping": 1e22, "junction_depth": 3e-7,
            "static_feedback": 0.5, "narrow_width_factor": 0.8, "mobility_modulation": 0.1,
            "saturation_field_factor": 0.3, "fast_surface_state_density": 1e15,
            "threshold_adjustment": 0.05, "max_drift_velocity": 1.5e5,
            "width_narrowing": 1e-7, "length_adjustment": 1e-8, "width_adjustment": 2e-8,
            "nominal_temperature": 290}"#;
        for [drain, gate, bulk] in [
            [3.0, 2.0, -0.5],
            [0.3, 2.5, -0.2],
            [-1.0, 1.5, -1.5],
            [2.0, 0.2, -0.1],
        ] {
            let x = [drain, gate, 0.0, bulk, 1e-4, 2e-4, -1e-4];
            let inner = [drain, gate, 0.0, bulk, 0.9 * drain, 0.01, 1e-4, 2e-4, -1e-4];
            assert_parameter_gradients_match::<Mosfet>(&level_one, &x);
            assert_parameter_gradients_match::<Mosfet>(derived, &inner);
            assert_parameter_gradients_match::<Mosfet>(level_three, &x);
            assert_parameter_gradients_match::<Mosfet>(&resistive, &inner);
        }
    }

    #[test]
    fn a_heated_transistor_is_the_transistor_at_its_temperature() {
        let model = crate::Thermal::<Mosfet>::new();
        let level_one = CHARGED.replace(
            '{',
            r#"{"nominal_temperature": 290, "saturation_current": 2e-14, "#,
        );
        let derived = r#"{"substrate_doping": 1e22, "oxide_thickness": 2e-8,
            "surface_state_density": 1e14, "surface_mobility": 0.05, "width": 10e-6,
            "length": 2e-6, "saturation_current_density": 1e-4, "drain_area": 2e-11,
            "source_area": 3e-11, "nominal_temperature": 290}"#;
        let level_three = r#"{"level": 3, "threshold_voltage": 0.7, "body_effect": 0.4,
            "width": 20e-6, "length": 2e-6, "oxide_thickness": 3e-8,
            "substrate_doping": 1e22, "junction_depth": 3e-7, "static_feedback": 0.5,
            "mobility_modulation": 0.1, "max_drift_velocity": 1.5e5}"#;
        for every in [level_one.as_str(), derived, level_three] {
            for every in [every.to_owned(), p_channel(every)] {
                let sign = if every.contains(r#""p""#) { -1.0 } else { 1.0 };
                let heated = every.replace('{', r#"{"thermal": true, "#);
                let mut bench =
                    Bench::from_model(&model, &heated, REFERENCE_TEMPERATURE).expect("valid");
                for rise in [0.0, 40.0] {
                    let mut plain = Bench::at::<Mosfet>(&every, REFERENCE_TEMPERATURE + rise);
                    let [d, g, b] = [3.0, 2.0, -0.5].map(|v| sign * v);
                    let hot = bench.settle(&[d, g, 0.0, b, rise, 1e-4, 2e-4, -1e-4]);
                    let reference = plain.settle(&[d, g, 0.0, b, 1e-4, 2e-4, -1e-4]);
                    let electrical = [0, 1, 2, 3, 5, 6, 7].map(|row| hot.residual[row]);
                    assert_close(&electrical, &reference.residual);
                }
                for [d, g, b, rise] in [[3.0, 2.0, -0.5, 20.0], [0.3, 2.5, -0.2, 60.0]] {
                    let x = [sign * d, sign * g, 0.0, sign * b, rise, 1e-4, 2e-4, -1e-4];
                    crate::testing::assert_model_jacobians_match(&model, &heated, &x);
                    crate::testing::assert_model_parameter_gradients_match(
                        &model, &heated, &x, None,
                    );
                }
            }
        }
    }

    proptest! {
        /// The conductances and capacitances are the derivatives of the currents and
        /// charges, in every region, in both modes and for both channels, with the bulk up
        /// to 0.6 V forward of the drain or the source.
        #[test]
        fn jacobians_match_finite_differences(
            drain in -4.0..4.0f64,
            gate in -3.0..4.0f64,
            reverse in -0.6..2.0f64,
            displacements in proptest::array::uniform3(-1e-3..1e-3f64),
            p in proptest::bool::ANY,
        ) {
            let parameters = if p { p_channel(CHARGED) } else { CHARGED.to_owned() };
            let sign = if p { -1.0 } else { 1.0 };
            let bulk = drain.min(0.0) - reverse;
            let [gs, gd, gb] = displacements;
            assert_jacobians_match::<Mosfet>(
                &parameters,
                &[sign * drain, sign * gate, 0.0, sign * bulk, gs, gd, gb],
            );
            let resistive = parameters.replace(
                '{',
                r#"{"drain_resistance": 50, "source_resistance": 20, "#,
            );
            let inner = [sign * 0.9 * drain, sign * 0.01];
            assert_jacobians_match::<Mosfet>(
                &resistive,
                &[sign * drain, sign * gate, 0.0, sign * bulk, inner[0], inner[1], gs, gd, gb],
            );
        }

        /// Level 3's derivatives hold in strong and weak inversion, below and beyond
        /// saturation, with and without velocity saturation.
        #[test]
        fn level_three_jacobians_match_finite_differences(
            drain in -4.0..4.0f64,
            gate in -3.0..4.0f64,
            reverse in 0.0..2.0f64,
            drift in proptest::bool::ANY,
            p in proptest::bool::ANY,
        ) {
            let parameters = format!(
                r#"{{"level": 3, "threshold_voltage": 0.7, "body_effect": 0.4,
                    "width": 20e-6, "length": 2e-6, "lateral_diffusion": 1e-7,
                    "oxide_thickness": 3e-8, "substrate_doping": 1e22, "junction_depth": 3e-7,
                    "static_feedback": 0.5, "narrow_width_factor": 0.8,
                    "mobility_modulation": 0.1, "saturation_field_factor": 0.3,
                    "fast_surface_state_density": 1e15, "threshold_adjustment": 0.05,
                    "max_drift_velocity": {}}}"#,
                if drift { 1.5e5 } else { 0.0 }
            );
            let parameters = if p { p_channel(&parameters) } else { parameters };
            let sign = if p { -1.0 } else { 1.0 };
            let bulk = drain.min(0.0) - reverse;
            assert_jacobians_match::<Mosfet>(
                &parameters,
                &[sign * drain, sign * gate, 0.0, sign * bulk, 1e-4, 2e-4, -1e-4],
            );
        }

        /// A transistor whose threshold derives from its substrate doping has the
        /// derivatives too.
        #[test]
        fn derived_jacobians_match_finite_differences(
            drain in -4.0..4.0f64,
            gate in -3.0..4.0f64,
            reverse in -0.6..2.0f64,
        ) {
            let parameters = r#"{"oxide_thickness": 5e-8, "substrate_doping": 1e22,
                "surface_state_density": 1e14, "gate_material": "aluminum"}"#;
            let bulk = drain.min(0.0) - reverse;
            assert_jacobians_match::<Mosfet>(
                parameters,
                &[drain, gate, 0.0, bulk, 1e-4, 2e-4, -1e-4],
            );
        }
    }

    #[test]
    fn its_channel_makes_thermal_and_flicker_noise() {
        // SPICE's oxide of 0.1 µm; each copy carries its share of the current over its own
        // effective gate, and the copies' noises add.
        let oxide = OXIDE_PERMITTIVITY / 1e-7;
        for copies in [1.0, 2.0] {
            let parameters = format!(
                r#"{{"threshold_voltage": 1, "transconductance_parameter": 1e-4,
                "saturation_current": 1e-30, "width": 10e-6, "length": 2e-6,
                "lateral_diffusion": 1e-7, "multiplier": {copies},
                "flicker_coefficient": 1e-25, "flicker_exponent": 1.2}}"#
            );
            let mut bench = Bench::new::<Mosfet>(&parameters);
            let x = [5.0, 2.0, 0.0, -5.0];
            let evaluation = bench.settle(&x);
            let (current, gm) = (evaluation.residual[0], evaluation.conductance[0][1]);
            let sources = bench.noise(&x);
            let gate = 10e-6 * (2e-6 - 2.0 * 1e-7);
            let flicker = copies * 1e-25 * (current / copies).powf(1.2) / (gate * oxide * oxide);
            assert_close(
                &[
                    source(&sources, "channel thermal").density(1.0),
                    source(&sources, "flicker").density(100.0),
                ],
                &[thermal(2.0 / 3.0 * gm), flicker / 100.0],
            );
            assert_eq!(sources.len(), 2);
        }
    }
}
