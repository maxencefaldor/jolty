//! Junction diode kernel, derived from ngspice 47 `src/spicelib/devices/dio/`:
//! Copyright 1985 - 2018, Regents of the University of California and others. Used under
//! the ngspice Modified BSD license, reproduced in the repository `NOTICE` file.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::dimension::{
    CAPACITANCE, CURRENT, PER_KELVIN, PER_KELVIN_SQUARED, RESISTANCE, TEMPERATURE, TIME, VOLTAGE,
    VOLTS_PER_KELVIN,
};
use crate::kernel::dual::Dual;
use crate::kernel::junction::{
    Depletion, Temperatures, Varshni, below_one, critical_voltage, depletion_charge,
    junction_current, limit_junction,
};
use crate::kernel::parts::{Pair, Series};
use crate::noise::Flicker;
use crate::{
    BOLTZMANN, Branch, BranchKind, ELEMENTARY_CHARGE, Interface, Kernel, Layout, Lift, Local,
    NoiseSource, ParameterError, ParameterKind, ParameterMetadata, REFERENCE_TEMPERATURE, Real,
    Stamp,
};

/// Junction diode between `anode` and `cathode` implementing the SPICE model from ngspice
/// 47.
///
/// The bottom junction carries ideal current (slope `n·Vt`), recombination current,
/// tunneling current, and `gmin · v`, softened by knee currents, together with depletion
/// and diffusion charges. The sidewall junction carries parallel ideal and tunneling
/// currents, forward knee softening, and depletion charge behind the shared series
/// resistance, or behind a dedicated series resistance connected directly to `anode`.
///
/// Positive series resistance inserts an internal node labeled `junction` (and
/// `junction_current` below [`SMALL_RESISTANCE`](crate::SMALL_RESISTANCE)); a separate
/// sidewall resistance inserts `sidewall` (and `sidewall_current`). When a breakdown
/// voltage is supplied, reverse current grows exponentially past the ngspice-matched
/// breakdown knee, with SPICE voltage step limiting preventing exponential overflow.
///
/// Away from nominal temperature, saturation currents scale with the energy gap, and
/// junction potentials and capacitances scale by band gap or linear temperature
/// coefficients. Breakdown voltage, resistances, transit time, and grading coefficient
/// follow dedicated ngspice temperature coefficients.
#[derive(Debug, Clone, Copy)]
pub struct Diode;

/// Layout shape for [`Diode`] storing the main series resistance, main junction `Pair`,
/// and optional separate sidewall branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiodeShape {
    series: Series,
    junction: Pair,
    sidewall: Option<SeparateSidewall>,
}

/// Separate sidewall junction branch with a dedicated series resistance from `anode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SeparateSidewall {
    series: Series,
    junction: Pair,
}

/// Prepared diode parameters at operating temperature.
#[derive(Debug, Clone, Copy)]
pub struct PreparedDiode<R: Real> {
    /// Series resistance at operating temperature.
    resistance: R,
    /// Dedicated sidewall series resistance at operating temperature, when present.
    sidewall_resistance: R,
    bottom: Bottom<R>,
    sidewall: Sidewall<R>,
    /// Transit time at operating temperature.
    transit_time: R,
    temperature: f64,
    flicker: Flicker,
}

/// Persistent evaluation memory for [`Diode`] storing previous junction voltages for step
/// limiting.
#[derive(Debug, Clone, Copy, Default)]
pub struct DiodeMemory {
    /// Bottom junction voltage from the last evaluation for step limiting.
    last_voltage: f64,
    /// Separate sidewall junction voltage from the last evaluation for step limiting.
    last_sidewall_voltage: f64,
}

/// Temperature-prepared pn junction combining Shockley current, reverse breakdown,
/// depletion charge, and step limiting, shared by [`Diode`] and VDMOS body diodes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Junction<R: Real> {
    saturation_current: R,
    /// Exponential voltage slope `n · Vt`.
    slope: R,
    /// Critical voltage above which Newton steps undergo logarithmic limiting.
    critical_voltage: f64,
    depletion: Option<Depletion<R>>,
    breakdown: Option<Breakdown<R>>,
}

/// Reverse breakdown model for a pn junction.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Breakdown<R: Real> {
    /// Matched reverse breakdown knee voltage.
    voltage: R,
    /// Breakdown exponential slope `nbv · Vt`.
    slope: R,
}

impl<R: Real> Breakdown<R> {
    /// Constructs a [`Breakdown`] model matched to `voltage` and `current` for a junction
    /// with `saturation` current, exponential `slope`, and thermal voltage `vt`.
    ///
    /// Runs ngspice fixed-point iteration so breakdown current matches `current` at
    /// `voltage`, leaving `voltage` unchanged when `current` is smaller than the linear
    /// saturation floor.
    pub(crate) fn new(voltage: R, current: R, saturation: R, slope: R, vt: R) -> Self {
        /// Relative convergence tolerance for matching breakdown voltage to `current`.
        const MATCH: f64 = 1e-12;
        if current.value() < (saturation * voltage / vt).value() {
            return Self { voltage, slope };
        }
        let mut matched = voltage - slope * (current / saturation + 1.0).ln();
        for _ in 0..25 {
            matched = voltage - slope * (current / saturation + 1.0 - matched / vt).ln();
            let reached = saturation * (((voltage - matched) / slope).exp() - 1.0 + matched / vt);
            if (reached - current).abs().value() <= MATCH * current.value() {
                break;
            }
        }
        Self {
            voltage: matched,
            slope,
        }
    }
}

/// Bottom-area junction combining ideal current, breakdown, depletion charge,
/// recombination, tunneling, and knee currents.
#[derive(Debug, Clone, Copy)]
struct Bottom<R: Real> {
    junction: Junction<R>,
    recombination: Option<Recombination<R>>,
    tunneling: Option<Tunneling<R>>,
    forward_knee: Option<R>,
    reverse_knee: Option<R>,
}

/// Sidewall-perimeter junction combining ideal current, shared breakdown, depletion charge,
/// tunneling, and forward knee current.
#[derive(Debug, Clone, Copy)]
struct Sidewall<R: Real> {
    junction: Junction<R>,
    tunneling: Option<Tunneling<R>>,
    forward_knee: Option<R>,
}

/// Depletion-region recombination current
/// `Isr · (e^{v / (nr·Vt)} - 1) · ((1 - v/φ)² + 0.005)^{m/2}` matching ngspice `dioload.c`
/// (lines 390 to 405).
///
/// Below `-3 · slope` of the bottom junction, voltage is clamped at the boundary (lines
/// 415 to 420).
#[derive(Debug, Clone, Copy)]
struct Recombination<R: Real> {
    saturation_current: R,
    /// Recombination exponential slope `nr · Vt`.
    slope: R,
    /// Bottom junction built-in potential `φ` at operating temperature.
    potential: R,
    /// Half the bottom junction grading coefficient `m/2` at operating temperature.
    half_grading: R,
}

/// Reverse tunneling current `-Itun · (e^{-v / (ntun·Vt)} - 1)` matching ngspice
/// `dioload.c` (lines 461 to 483).
#[derive(Debug, Clone, Copy)]
struct Tunneling<R: Real> {
    saturation_current: R,
    /// Tunneling exponential slope `ntun · Vt`.
    slope: R,
}

/// Minimum current magnitude softened by knee currents, in amperes (ngspice `dioload.c`,
/// lines 487, 496, and 504).
const KNEE_FLOOR: f64 = 1e-18;

/// Smoothing constant keeping the recombination generation factor nonzero at `v = φ`
/// (ngspice `dioload.c`, line 396).
const GENERATION_SMOOTHING: f64 = 0.005;

/// Temperature scaling selector for diode saturation currents, band gap, and breakdown
/// voltage (SPICE `TLEV`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize, JsonSchema)]
#[serde(try_from = "u8", into = "u8")]
pub enum DiodeTemperatureModel {
    /// Model 0 (`TLEV = 0`): constant energy gap for saturation currents, silicon band gap
    /// for potentials, and absolute breakdown drift in volts per kelvin.
    #[default]
    Zero,
    /// Model 1 (`TLEV = 1`): identical to model 0 except breakdown voltage drifts
    /// proportionally by its relative coefficient per kelvin.
    One,
    /// Model 2 (`TLEV = 2`): identical to model 1 except saturation currents and potentials
    /// follow the Varshni energy gap `Eg - α·T² / (T + β)`.
    Two,
}

super::numbered_enum!(
    DiodeTemperatureModel,
    "a diode's temperature_model is 0, 1 or 2",
    {
        0 => Zero,
        1 => One,
        2 => Two,
    }
);

/// Temperature scaling selector for diode junction capacitances and built-in potentials
/// (SPICE `TLEVC`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize, JsonSchema)]
#[serde(try_from = "u8", into = "u8")]
pub enum DiodeCapacitanceTemperatureModel {
    /// Model 0 (`TLEVC = 0`): physical band-gap and intrinsic-concentration scaling.
    #[default]
    Zero,
    /// Model 1 (`TLEVC = 1`): linear scaling via dedicated capacitance and potential
    /// temperature coefficients.
    One,
}

super::numbered_enum!(
    DiodeCapacitanceTemperatureModel,
    "a diode's capacitance_temperature_model is 0 or 1",
    {
        0 => Zero,
        1 => One,
    }
);

/// Parameters for [`Diode`], matching ngspice defaults (`diosetup.c`, lines 36 to 307).
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift, ParameterMetadata)]
#[serde(default, deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct DiodeParameters<R: Real = f64> {
    /// Bottom junction saturation current, in amperes; default 1e-14.
    #[dimension(CURRENT)]
    pub saturation_current: R,
    /// Bottom junction emission coefficient; default 1.
    pub emission_coefficient: R,
    /// Main series resistance, in ohms; default 0.
    #[dimension(RESISTANCE)]
    pub series_resistance: R,
    /// Zero-bias bottom junction capacitance, in farads; default 0.
    #[dimension(CAPACITANCE)]
    pub junction_capacitance: R,
    /// Bottom junction built-in potential, in volts; default 1.
    #[dimension(VOLTAGE)]
    pub junction_potential: R,
    /// Bottom junction grading coefficient, below 1; default 0.5.
    pub grading_coefficient: R,
    /// Fraction of junction potential above which depletion capacitance extends linearly;
    /// below 1; default 0.5.
    pub depletion_coefficient: R,
    /// Carrier transit time, in seconds; default 0.
    #[dimension(TIME)]
    pub transit_time: R,
    /// Forward high-injection knee current, in amperes; omitted disables forward knee
    /// softening.
    #[dimension(CURRENT)]
    pub forward_knee_current: Option<R>,
    /// Reverse high-injection knee current, in amperes; omitted disables reverse knee
    /// softening.
    #[dimension(CURRENT)]
    pub reverse_knee_current: Option<R>,
    /// Depletion-region recombination saturation current, in amperes; omitted disables
    /// recombination current.
    #[dimension(CURRENT)]
    pub recombination_saturation_current: Option<R>,
    /// Recombination current emission coefficient; default 2.
    pub recombination_emission_coefficient: R,
    /// Sidewall junction saturation current, in amperes; default 0 (no sidewall current).
    #[dimension(CURRENT)]
    pub sidewall_saturation_current: R,
    /// Sidewall junction emission coefficient; defaults to `emission_coefficient`.
    pub sidewall_emission_coefficient: Option<R>,
    /// Dedicated sidewall series resistance from `anode`, in ohms; omitted places the
    /// sidewall in parallel with the bottom junction behind `series_resistance`.
    #[dimension(RESISTANCE)]
    pub sidewall_series_resistance: Option<R>,
    /// Zero-bias sidewall junction capacitance, in farads; default 0.
    #[dimension(CAPACITANCE)]
    pub sidewall_junction_capacitance: R,
    /// Sidewall junction built-in potential, in volts; default 1.
    #[dimension(VOLTAGE)]
    pub sidewall_junction_potential: R,
    /// Sidewall junction grading coefficient, below 1; default 0.33.
    pub sidewall_grading_coefficient: R,
    /// Fraction of sidewall junction potential above which sidewall depletion capacitance
    /// extends linearly; below 1; default 0.5.
    pub sidewall_depletion_coefficient: R,
    /// Sidewall forward high-injection knee current, in amperes; omitted disables sidewall
    /// knee softening.
    #[dimension(CURRENT)]
    pub sidewall_forward_knee_current: Option<R>,
    /// Bottom tunneling saturation current, in amperes; default 0 (no tunneling).
    #[dimension(CURRENT)]
    pub tunneling_saturation_current: R,
    /// Sidewall tunneling saturation current, in amperes; default 0.
    #[dimension(CURRENT)]
    pub sidewall_tunneling_saturation_current: R,
    /// Tunneling current emission coefficient; default 30.
    pub tunneling_emission_coefficient: R,
    /// Temperature-ratio exponent scaling tunneling saturation currents; default 3.
    pub tunneling_saturation_current_temperature_exponent: R,
    /// Energy-gap multiplier scaling tunneling saturation currents; default 1.
    pub tunneling_energy_gap_factor: R,
    /// Reverse breakdown voltage, in volts (non-negative); omitted disables breakdown.
    #[dimension(VOLTAGE)]
    pub breakdown_voltage: Option<R>,
    /// Current magnitude at `breakdown_voltage`, in amperes (non-negative); default 1e-3.
    #[dimension(CURRENT)]
    pub breakdown_current: R,
    /// Reverse breakdown emission coefficient; defaults to `emission_coefficient`.
    pub breakdown_emission_coefficient: Option<R>,
    /// Linear breakdown voltage temperature coefficient: in volts per kelvin for model 0,
    /// or relative per kelvin for models 1 and 2; default 0.
    #[dimension(conditional = "diode_tbv")]
    pub breakdown_temperature_coefficient: R,
    /// Saturation-current and band-gap temperature model (`TLEV`); default 0.
    pub temperature_model: DiodeTemperatureModel,
    /// Junction-capacitance and potential temperature model (`TLEVC`); default 0.
    pub capacitance_temperature_model: DiodeCapacitanceTemperatureModel,
    /// Energy gap scaling saturation currents, in volts; defaults to 1.11 (or 1.16 at 0 K
    /// under temperature model 2).
    #[dimension(VOLTAGE)]
    pub energy_gap: Option<R>,
    /// Varshni `α` parameter for temperature model 2, in volts per kelvin; default 7.02e-4
    /// (silicon).
    #[dimension(VOLTS_PER_KELVIN)]
    pub energy_gap_alpha: R,
    /// Varshni `β` parameter for temperature model 2, in kelvins; default 1108 (silicon).
    #[dimension(TEMPERATURE)]
    pub energy_gap_beta: R,
    /// Temperature-ratio exponent scaling saturation currents; default 3.
    pub saturation_current_temperature_exponent: R,
    /// Linear bottom capacitance temperature coefficient under `TLEVC = 1`, per kelvin;
    /// default 0.
    #[dimension(PER_KELVIN)]
    pub junction_capacitance_temperature_coefficient: R,
    /// Linear sidewall capacitance temperature coefficient under `TLEVC = 1`, per kelvin;
    /// default 0.
    #[dimension(PER_KELVIN)]
    pub sidewall_junction_capacitance_temperature_coefficient: R,
    /// Linear bottom junction potential temperature coefficient under `TLEVC = 1`, in volts
    /// per kelvin; default 0.
    #[dimension(VOLTS_PER_KELVIN)]
    pub junction_potential_temperature_coefficient: R,
    /// Linear sidewall junction potential temperature coefficient under `TLEVC = 1`, in
    /// volts per kelvin; default 0.
    #[dimension(VOLTS_PER_KELVIN)]
    pub sidewall_junction_potential_temperature_coefficient: R,
    /// Linear series resistance temperature coefficient, per kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub series_resistance_temperature_coefficient: R,
    /// Quadratic series resistance temperature coefficient, per kelvin squared; default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub series_resistance_quadratic_temperature_coefficient: R,
    /// Linear transit time temperature coefficient, per kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub transit_time_temperature_coefficient: R,
    /// Quadratic transit time temperature coefficient, per kelvin squared; default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub transit_time_quadratic_temperature_coefficient: R,
    /// Linear grading coefficient temperature coefficient, per kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub grading_coefficient_temperature_coefficient: R,
    /// Quadratic grading coefficient temperature coefficient, per kelvin squared; default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub grading_coefficient_quadratic_temperature_coefficient: R,
    /// Nominal measurement temperature, in kelvins; default 27 °C.
    #[dimension(TEMPERATURE)]
    pub nominal_temperature: R,
    /// Flicker noise coefficient `KF`; default 0 (no flicker noise).
    pub flicker_coefficient: R,
    /// Flicker noise current exponent `AF`; default 1.
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
            breakdown_current: R::constant(1e-3),
            breakdown_emission_coefficient: None,
            breakdown_temperature_coefficient: R::constant(0.0),
            temperature_model: DiodeTemperatureModel::Zero,
            capacitance_temperature_model: DiodeCapacitanceTemperatureModel::Zero,
            energy_gap: None,
            energy_gap_alpha: R::constant(7.02e-4),
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
            nominal_temperature: R::constant(REFERENCE_TEMPERATURE),
            flicker_coefficient: R::constant(0.0),
            flicker_exponent: R::constant(1.0),
        }
    }
}

impl Kernel for Diode {
    const NAME: &'static str = "diode";
    type Parameters<R: Real> = DiodeParameters<R>;
    type Shape = DiodeShape;
    type Prepared<R: Real> = PreparedDiode<R>;
    type Memory = DiodeMemory;

    fn parameter_kinds() -> &'static [(&'static str, ParameterKind)] {
        DiodeParameters::<f64>::PARAMETER_KINDS
    }

    fn interface<R: Real>(_: &DiodeParameters<R>) -> Interface {
        Interface::new(
            &["anode", "cathode"],
            &[Branch::new(0, 1, BranchKind::Conductive)],
        )
    }

    fn prepare<R: Real>(
        parameters: &DiodeParameters<R>,
        temperature: R,
    ) -> Result<PreparedDiode<R>, ParameterError> {
        parameters.prepared(temperature)
    }

    fn layout<R: Real>(
        parameters: &DiodeParameters<R>,
        _: &PreparedDiode<R>,
        layout: &mut Layout,
    ) -> Result<DiodeShape, ParameterError> {
        let series_resistance =
            ParameterError::non_negative("series_resistance", parameters.series_resistance)?;
        let (anode, cathode) = (0, 1);
        let (series, inner) = Series::behind(
            layout,
            anode,
            series_resistance,
            ("junction", "junction_current"),
        );
        let junction = Pair::new(layout, inner, cathode);
        let sidewall = parameters
            .sidewall_series_resistance
            .map(|resistance| {
                ParameterError::non_negative("sidewall_series_resistance", resistance).map(
                    |resistance| {
                        let (series, inner) = Series::behind(
                            layout,
                            anode,
                            resistance,
                            ("sidewall", "sidewall_current"),
                        );
                        SeparateSidewall {
                            series,
                            junction: Pair::new(layout, inner, cathode),
                        }
                    },
                )
            })
            .transpose()?;
        Ok(DiodeShape {
            series,
            junction,
            sidewall,
        })
    }

    fn evaluate<R: Real>(
        shape: &DiodeShape,
        prepared: &PreparedDiode<R>,
        memory: &mut DiodeMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        shape
            .series
            .add(shape.series.value_of(prepared.resistance), at, stamp);
        // Without a dedicated resistance, the sidewall shares the main junction voltage and
        // step limiting.
        let (applied, voltage) =
            prepared
                .bottom
                .junction
                .limited(&shape.junction, &mut memory.last_voltage, at, stamp);
        let beyond = [applied - voltage];
        let [voltage] = Dual::variables([shape.junction.across(at, voltage)]);
        let mut current = prepared.bottom.current(voltage);
        let mut charge = depletion_charge(prepared.bottom.junction.depletion.as_ref(), voltage);
        if shape.sidewall.is_none() {
            current = current + prepared.sidewall.current(voltage);
            charge =
                charge + depletion_charge(prepared.sidewall.junction.depletion.as_ref(), voltage);
        }
        add_junction(
            &shape.junction,
            voltage,
            (current, charge + current.scale(prepared.transit_time)),
            beyond,
            at,
            stamp,
        );

        let Some(separate) = &shape.sidewall else {
            return;
        };
        separate.series.add(
            separate.series.value_of(prepared.sidewall_resistance),
            at,
            stamp,
        );
        let (applied, voltage) = prepared.sidewall.junction.limited(
            &separate.junction,
            &mut memory.last_sidewall_voltage,
            at,
            stamp,
        );
        let beyond = [applied - voltage];
        let [voltage] = Dual::variables([separate.junction.across(at, voltage)]);
        let current = prepared.sidewall.current(voltage);
        let charge = depletion_charge(prepared.sidewall.junction.depletion.as_ref(), voltage);
        add_junction(
            &separate.junction,
            voltage,
            (current, charge),
            beyond,
            at,
            stamp,
        );
    }

    /// Appends thermal noise for series resistances plus shot and flicker noise for junction
    /// currents.
    fn noise<R: Real>(
        shape: &DiodeShape,
        prepared: &PreparedDiode<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        let thermal = ("series thermal", prepared.temperature);
        shape
            .series
            .noise(shape.series.value_of(prepared.resistance), thermal, sources);
        let voltage = Dual::<R, 0>::number(shape.junction.voltage(at));
        let mut current = prepared.bottom.current(voltage);
        if shape.sidewall.is_none() {
            current = current + prepared.sidewall.current(voltage);
        }
        let nodes = shape.junction.equations();
        let current = current.value.value();
        sources.push(NoiseSource::shot("shot", nodes, current));
        prepared.flicker.add(nodes, current, sources);

        let Some(separate) = &shape.sidewall else {
            return;
        };
        let thermal = ("sidewall series thermal", prepared.temperature);
        separate.series.noise(
            separate.series.value_of(prepared.sidewall_resistance),
            thermal,
            sources,
        );
        let voltage = Dual::<R, 0>::number(separate.junction.voltage(at));
        let current = prepared.sidewall.current(voltage).value.value();
        let nodes = separate.junction.equations();
        sources.push(NoiseSource::shot("sidewall shot", nodes, current));
        prepared
            .flicker
            .add_named("sidewall flicker", nodes, current, sources);
    }
}

/// Stamps junction `current` (plus `gmin`) and `charge` across `pair`, linearly extending
/// each from limited `voltage` by `beyond` so the next Newton step linearizes at the limited
/// bias.
fn add_junction<R: Real>(
    pair: &Pair,
    voltage: Dual<R, 1>,
    (current, charge): (Dual<R, 1>, Dual<R, 1>),
    beyond: [f64; 1],
    at: &Local<'_>,
    stamp: &mut Stamp<'_, R>,
) {
    let total = current + voltage * at.gmin;
    let [conductance] = total.gradient;
    pair.add_current(stamp, total.extend(beyond));
    pair.add_conductance(stamp, conductance);
    let [capacitance] = charge.gradient;
    pair.add_charge(stamp, charge.extend(beyond));
    pair.add_capacitance(stamp, capacitance);
}

impl<R: Real> DiodeParameters<R> {
    /// Validates parameters and computes temperature-scaled diode values at `temperature`,
    /// in kelvins.
    fn prepared(&self, temperature: R) -> Result<PreparedDiode<R>, ParameterError> {
        let flicker = Flicker::new(self.flicker_coefficient, self.flicker_exponent)?;
        let series_resistance =
            ParameterError::non_negative("series_resistance", self.series_resistance)?;
        let sidewall_resistance = self
            .sidewall_series_resistance
            .map(|resistance| {
                ParameterError::non_negative("sidewall_series_resistance", resistance)
            })
            .transpose()?
            .unwrap_or(R::constant(0.0));
        let transit_time = ParameterError::non_negative("transit_time", self.transit_time)?;
        let temperatures = Temperatures::of_junction(temperature, self.nominal_temperature)?;
        let resistances = temperatures.polynomial(
            self.series_resistance_temperature_coefficient,
            self.series_resistance_quadratic_temperature_coefficient,
        );
        let transit = temperatures.polynomial(
            self.transit_time_temperature_coefficient,
            self.transit_time_quadratic_temperature_coefficient,
        );
        let (bottom, sidewall) = self.junctions(temperatures)?;
        Ok(PreparedDiode {
            resistance: series_resistance * resistances,
            sidewall_resistance: sidewall_resistance * resistances,
            bottom,
            sidewall,
            transit_time: transit_time * transit,
            temperature: temperature.value(),
            flicker,
        })
    }

    /// Validates junction parameters and prepares the bottom and sidewall junctions at
    /// `temperatures`, matching ngspice `diotemp.c`.
    fn junctions(
        &self,
        temperatures: Temperatures<R>,
    ) -> Result<(Bottom<R>, Sidewall<R>), ParameterError> {
        let emission = ParameterError::positive("emission_coefficient", self.emission_coefficient)?;
        let sidewall_emission = self.sidewall_emission_coefficient.unwrap_or(emission);
        ParameterError::positive("sidewall_emission_coefficient", sidewall_emission)?;
        ParameterError::positive("saturation_current", self.saturation_current)?;
        ParameterError::non_negative(
            "sidewall_saturation_current",
            self.sidewall_saturation_current,
        )?;
        let vt = temperatures.thermal_voltage();
        let gaps = self.gaps(temperatures)?;
        let exponent = self.saturation_current_temperature_exponent;
        let saturation =
            self.saturation_current * gaps.scaling(R::constant(1.0), exponent, emission);
        let sidewall_saturation = self.sidewall_saturation_current
            * gaps.scaling(R::constant(1.0), exponent, sidewall_emission);
        let grading = below_one("grading_coefficient", self.grading_coefficient)?
            * temperatures.polynomial(
                self.grading_coefficient_temperature_coefficient,
                self.grading_coefficient_quadratic_temperature_coefficient,
            );
        let (bottom_depletion, potential) = self.bottom_depletion(temperatures, gaps, grading)?;
        let sidewall_depletion = self.sidewall_depletion(temperatures, gaps)?;
        let (slope, sidewall_slope) = (emission * vt, sidewall_emission * vt);
        let breakdown = self.breakdown(temperatures, saturation + sidewall_saturation)?;
        let tunneling = self.tunneling(gaps, vt)?;
        let separate = self.sidewall_series_resistance.is_some();
        let bottom = Bottom {
            junction: Junction::new(saturation, slope, bottom_depletion, breakdown).limited_as(
                if separate {
                    saturation
                } else {
                    saturation + sidewall_saturation
                },
            ),
            recombination: self
                .recombination_saturation_current
                .map(|current| {
                    let emission = ParameterError::positive(
                        "recombination_emission_coefficient",
                        self.recombination_emission_coefficient,
                    )?;
                    Ok(Recombination {
                        saturation_current: ParameterError::non_negative(
                            "recombination_saturation_current",
                            current,
                        )? * gaps.scaling(R::constant(1.0), exponent, emission),
                        slope: emission * vt,
                        potential,
                        half_grading: grading * 0.5,
                    })
                })
                .transpose()?,
            tunneling: tunneling
                .map(|(scaling, slope)| Tunneling {
                    saturation_current: self.tunneling_saturation_current * scaling,
                    slope,
                })
                .filter(|_| self.tunneling_saturation_current.value() > 0.0),
            forward_knee: knee("forward_knee_current", self.forward_knee_current)?,
            reverse_knee: knee("reverse_knee_current", self.reverse_knee_current)?,
        };
        let sidewall = Sidewall {
            // Sidewalls with dedicated series resistance do not break down (`dioload.c`,
            // lines 360 and 451).
            junction: Junction::new(
                sidewall_saturation,
                sidewall_slope,
                sidewall_depletion,
                breakdown.filter(|_| !separate),
            ),
            tunneling: tunneling
                .map(|(scaling, slope)| Tunneling {
                    saturation_current: self.sidewall_tunneling_saturation_current * scaling,
                    slope,
                })
                .filter(|_| self.sidewall_tunneling_saturation_current.value() > 0.0),
            forward_knee: knee(
                "sidewall_forward_knee_current",
                self.sidewall_forward_knee_current,
            )?,
        };
        Ok((bottom, sidewall))
    }

    /// Computes energy-gap terms used to scale saturation currents at `temperatures`.
    fn gaps(&self, temperatures: Temperatures<R>) -> Result<Gaps<R>, ParameterError> {
        let varying = self.temperature_model == DiodeTemperatureModel::Two;
        // ngspice's default energy gap: silicon's at 0 K where it follows Varshni's law,
        // and its activation energy at room temperature otherwise (`diosetup.c`, lines
        // 140 to 146).
        let gap = self
            .energy_gap
            .unwrap_or(R::constant(if varying { 1.16 } else { 1.11 }));
        ParameterError::non_negative("energy_gap", gap)?;
        let band = if varying {
            ParameterError::non_negative("energy_gap_beta", self.energy_gap_beta)?;
            Varshni {
                gap,
                alpha: self.energy_gap_alpha,
                beta: self.energy_gap_beta,
            }
        } else {
            Varshni::silicon()
        };
        Ok(Gaps {
            temperatures,
            gap,
            band,
            varying,
        })
    }

    /// Prepares bottom-junction depletion and returns `(depletion, scaled_potential)` at
    /// `temperatures`, band `gaps`, and scaled `grading`.
    fn bottom_depletion(
        &self,
        temperatures: Temperatures<R>,
        gaps: Gaps<R>,
        grading: R,
    ) -> Result<(Option<Depletion<R>>, R), ParameterError> {
        let capacitance =
            ParameterError::non_negative("junction_capacitance", self.junction_capacitance)?;
        let potential = ParameterError::positive("junction_potential", self.junction_potential)?;
        let coefficient = below_one("depletion_coefficient", self.depletion_coefficient)?;
        let (potential, capacitance) = self.capacitance_temperature_model.scaled(
            temperatures,
            gaps.band,
            (potential, capacitance, grading),
            (
                self.junction_potential_temperature_coefficient,
                self.junction_capacitance_temperature_coefficient,
            ),
        );
        ParameterError::positive("junction_potential at the temperature", potential)?;
        Ok((
            Depletion::new(capacitance, potential, grading, coefficient),
            potential,
        ))
    }

    /// Prepares sidewall-junction depletion at `temperatures` and band `gaps`.
    fn sidewall_depletion(
        &self,
        temperatures: Temperatures<R>,
        gaps: Gaps<R>,
    ) -> Result<Option<Depletion<R>>, ParameterError> {
        let capacitance = ParameterError::non_negative(
            "sidewall_junction_capacitance",
            self.sidewall_junction_capacitance,
        )?;
        let potential = ParameterError::positive(
            "sidewall_junction_potential",
            self.sidewall_junction_potential,
        )?;
        let grading = below_one(
            "sidewall_grading_coefficient",
            self.sidewall_grading_coefficient,
        )?;
        let coefficient = below_one(
            "sidewall_depletion_coefficient",
            self.sidewall_depletion_coefficient,
        )?;
        let (potential, capacitance) = self.capacitance_temperature_model.scaled(
            temperatures,
            gaps.band,
            (potential, capacitance, grading),
            (
                self.sidewall_junction_potential_temperature_coefficient,
                self.sidewall_junction_capacitance_temperature_coefficient,
            ),
        );
        ParameterError::positive("sidewall_junction_potential at the temperature", potential)?;
        Ok(Depletion::new(capacitance, potential, grading, coefficient))
    }

    /// Prepares reverse breakdown parameters at `temperatures` for combined junction
    /// `saturation` current, returning `None` when breakdown is disabled.
    fn breakdown(
        &self,
        temperatures: Temperatures<R>,
        saturation: R,
    ) -> Result<Option<Breakdown<R>>, ParameterError> {
        let Some(voltage) = self.breakdown_voltage else {
            return Ok(None);
        };
        // A breakdown voltage of 0 breaks the junction down at about 0 V, as ngspice's
        // equations do (`diotemp.c`, lines 210 to 246), where HSPICE reads it as none.
        let voltage = ParameterError::non_negative("breakdown_voltage", voltage)?;
        // A breakdown current of 0 leaves the breakdown voltage where it is, as ngspice
        // leaves it (`diotemp.c`, line 221).
        let current = ParameterError::non_negative("breakdown_current", self.breakdown_current)?;
        let emission = self
            .breakdown_emission_coefficient
            .unwrap_or(self.emission_coefficient);
        ParameterError::positive("breakdown_emission_coefficient", emission)?;
        let vt = temperatures.thermal_voltage();
        let fall = self.breakdown_temperature_coefficient * temperatures.difference();
        // ngspice's `diotemp.c`, lines 211 to 215.
        let voltage = match self.temperature_model {
            DiodeTemperatureModel::Zero => voltage - fall,
            DiodeTemperatureModel::One | DiodeTemperatureModel::Two => voltage * -(fall - 1.0),
        };
        Ok(Some(Breakdown::new(
            voltage,
            current,
            saturation,
            emission * vt,
            vt,
        )))
    }

    /// Computes tunneling saturation-current scaling at band `gaps` and exponential slope at
    /// thermal voltage `vt`, returning `None` when tunneling is inactive.
    fn tunneling(&self, gaps: Gaps<R>, vt: R) -> Result<Option<(R, R)>, ParameterError> {
        ParameterError::non_negative(
            "tunneling_saturation_current",
            self.tunneling_saturation_current,
        )?;
        ParameterError::non_negative(
            "sidewall_tunneling_saturation_current",
            self.sidewall_tunneling_saturation_current,
        )?;
        if self.tunneling_saturation_current.value() <= 0.0
            && self.sidewall_tunneling_saturation_current.value() <= 0.0
        {
            return Ok(None);
        }
        let emission = ParameterError::positive(
            "tunneling_emission_coefficient",
            self.tunneling_emission_coefficient,
        )?;
        let scaling = gaps.scaling(
            self.tunneling_energy_gap_factor,
            self.tunneling_saturation_current_temperature_exponent,
            emission,
        );
        Ok(Some((scaling, emission * vt)))
    }
}

/// Validates an optional knee current `parameter` as strictly positive when provided.
fn knee<R: Real>(parameter: &str, value: Option<R>) -> Result<Option<R>, ParameterError> {
    value
        .map(|value| ParameterError::positive(parameter, value))
        .transpose()
}

/// Band-gap parameters used by diode temperature scaling.
#[derive(Debug, Clone, Copy)]
struct Gaps<R: Real> {
    temperatures: Temperatures<R>,
    /// Constant activation energy gap `EG`.
    gap: R,
    /// Varshni band gap used for junction potentials and (under model 2) saturation currents.
    band: Varshni<R>,
    /// Flag selecting temperature-dependent `band` (model 2) over constant `gap` for
    /// saturation currents.
    varying: bool,
}

impl<R: Real> Gaps<R> {
    /// Computes the saturation-current temperature scaling factor for `factor`, `exponent`,
    /// and `emission` (`diotemp.c`, lines 114 to 184).
    ///
    /// Models 0 and 1 compute `exp((factor · (T/Tn - 1) · Eg / Vt + exponent · ln(T/Tn)) / n)`.
    /// Model 2 uses `Eg(Tn) / Vt(Tn) - Eg(T) / Vt` in place of `(T/Tn - 1) · Eg / Vt`.
    fn scaling(self, factor: R, exponent: R, emission: R) -> R {
        let Temperatures { actual, nominal } = self.temperatures;
        let vt = self.temperatures.thermal_voltage();
        let ratio = self.temperatures.ratio();
        let activation = if self.varying {
            let nominal_vt = nominal * BOLTZMANN / ELEMENTARY_CHARGE;
            self.band.at(nominal) / nominal_vt - self.band.at(actual) / vt
        } else {
            (ratio - 1.0) * self.gap / vt
        };
        ((factor * activation + exponent * ratio.ln()) / emission).exp()
    }
}

impl DiodeCapacitanceTemperatureModel {
    /// Scales junction `potential` and zero-bias `capacitance` of `grading` coefficient to
    /// `temperatures` using `band` (model 0) or linear `coefficients` (model 1).
    fn scaled<R: Real>(
        self,
        temperatures: Temperatures<R>,
        band: Varshni<R>,
        (potential, capacitance, grading): (R, R, R),
        (potential_coefficient, capacitance_coefficient): (R, R),
    ) -> (R, R) {
        match self {
            Self::Zero => temperatures.junction_of(potential, capacitance, grading, band),
            Self::One => {
                let difference = temperatures.difference();
                (
                    potential - potential_coefficient * difference,
                    capacitance * (capacitance_coefficient * difference + 1.0),
                )
            }
        }
    }
}

impl<R: Real> Bottom<R> {
    /// Evaluates bottom-junction current (excluding `gmin`) and derivatives at `voltage`,
    /// combining ideal, recombination, tunneling, and knee currents (`dioload.c`, lines
    /// 379 to 502).
    fn current<const N: usize>(&self, voltage: Dual<R, N>) -> Dual<R, N> {
        let forward = voltage.value.value() >= -3.0 * self.junction.slope.value();
        let mut current = self.junction.current(voltage);
        if let Some(recombination) = &self.recombination {
            current = current + recombination.current(voltage, self.junction.slope);
        }
        if let Some(tunneling) = &self.tunneling {
            current = current - tunneling.current(voltage);
        }
        match (forward, self.forward_knee, self.reverse_knee) {
            (true, Some(knee), _) if current.value.value() > KNEE_FLOOR => {
                current / (current.divide(knee).sqrt() + 1.0)
            }
            (false, _, Some(knee)) if current.value.value() < -KNEE_FLOOR => {
                current / ((-current).divide(knee).sqrt() + 1.0)
            }
            _ => current,
        }
    }
}

impl<R: Real> Sidewall<R> {
    /// Evaluates sidewall-junction current (excluding `gmin`) and derivatives at `voltage`,
    /// combining ideal, tunneling, and forward-knee terms (`dioload.c`, lines 334 to 373,
    /// 461 to 471, and 504 to 509).
    fn current<const N: usize>(&self, voltage: Dual<R, N>) -> Dual<R, N> {
        let mut current = if self.junction.saturation_current.value() > 0.0 {
            self.junction.current(voltage)
        } else {
            Dual::number(0.0)
        };
        if let Some(tunneling) = &self.tunneling {
            current = current - tunneling.current(voltage);
        }
        match self.forward_knee {
            Some(knee) if current.value.value() > KNEE_FLOOR => {
                current / (current.divide(knee).sqrt() + 1.0)
            }
            _ => current,
        }
    }
}

impl<R: Real> Recombination<R> {
    /// Evaluates depletion-region recombination current and derivatives at `voltage`,
    /// clamping reverse bias below `-3 · slope`.
    fn current<const N: usize>(&self, voltage: Dual<R, N>, slope: R) -> Dual<R, N> {
        let boundary = slope * -3.0;
        let voltage = if voltage.value.value() >= boundary.value() {
            voltage
        } else {
            Dual::constant(boundary)
        };
        let remaining = -(voltage.divide(self.potential) - 1.0);
        let generation = (remaining * remaining)
            .plus(R::constant(GENERATION_SMOOTHING))
            .pow(self.half_grading);
        (voltage.divide(self.slope).exp() - 1.0).scale(self.saturation_current) * generation
    }
}

impl<R: Real> Tunneling<R> {
    /// Evaluates reverse tunneling current `Itun · (e^{-v / slope} - 1)` and derivatives at
    /// `voltage` (subtracted from forward junction current).
    fn current<const N: usize>(&self, voltage: Dual<R, N>) -> Dual<R, N> {
        ((-voltage).divide(self.slope).exp() - 1.0).scale(self.saturation_current)
    }
}

impl<R: Real> Junction<R> {
    /// Constructs a [`Junction`] with `saturation_current`, exponential `slope`, optional
    /// `depletion` charge, and optional `breakdown`.
    pub(crate) fn new(
        saturation_current: R,
        slope: R,
        depletion: Option<Depletion<R>>,
        breakdown: Option<Breakdown<R>>,
    ) -> Self {
        Self {
            saturation_current,
            slope,
            critical_voltage: critical_voltage(saturation_current.value(), slope.value()),
            depletion,
            breakdown,
        }
    }

    /// Replaces the critical voltage using combined `saturation` current, matching ngspice's
    /// shared limiting for parallel bottom and sidewall junctions (`diotemp.c`, lines 200 to
    /// 207).
    fn limited_as(self, saturation: R) -> Self {
        Self {
            critical_voltage: critical_voltage(saturation.value(), self.slope.value()),
            ..self
        }
    }

    /// Reads the applied voltage across `pair`, limits the step from `last` when limiting is
    /// enabled, updates `last`, and returns `(applied, limited)`.
    fn limited(
        &self,
        pair: &Pair,
        last: &mut f64,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) -> (f64, f64) {
        let applied = pair.voltage(at);
        let voltage = at
            .limiting
            .then(|| self.limit(applied, *last))
            .flatten()
            .map_or(applied, |limited| {
                stamp.mark_limited();
                limited
            });
        *last = voltage;
        (applied, voltage)
    }

    /// Adds the junction's current across `pair`, with gmin across it, and its depletion
    /// charge and the diffusion charge of `transit_time`, its voltage limited from `last`'s,
    /// which takes the voltage it is evaluated at, and returns the capacitance there.
    pub(crate) fn add(
        &self,
        pair: &Pair,
        transit_time: R,
        last: &mut f64,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) -> R {
        let (applied, voltage) = self.limited(pair, last, at, stamp);
        let beyond = [applied - voltage];
        let [voltage] = Dual::variables([pair.across(at, voltage)]);
        let current = self.current(voltage);
        let charge =
            depletion_charge(self.depletion.as_ref(), voltage) + current.scale(transit_time);
        add_junction(pair, voltage, (current, charge), beyond, at, stamp);
        charge.gradient[0]
    }

    /// Applies ngspice junction step limiting from `last` to `applied`, shifting around the
    /// reverse breakdown knee when `applied` approaches breakdown.
    fn limit(&self, applied: f64, last: f64) -> Option<f64> {
        match self.breakdown {
            Some(breakdown)
                if applied
                    < (10.0 * breakdown.slope.value() - breakdown.voltage.value()).min(0.0) =>
            {
                let (voltage, slope) = (breakdown.voltage.value(), breakdown.slope.value());
                limit_junction(
                    -(applied + voltage),
                    -(last + voltage),
                    slope,
                    self.critical_voltage,
                )
                .map(|limited| -(limited + voltage))
            }
            _ => limit_junction(applied, last, self.slope.value(), self.critical_voltage),
        }
    }

    /// Evaluates junction current and derivatives at `voltage`, including exponential
    /// reverse breakdown below `-breakdown.voltage`.
    pub(crate) fn current<const N: usize>(&self, voltage: Dual<R, N>) -> Dual<R, N> {
        let at = voltage.value.value();
        match self.breakdown {
            Some(breakdown)
                if at < -3.0 * self.slope.value() && at < -breakdown.voltage.value() =>
            {
                -(-voltage.plus(breakdown.voltage))
                    .divide(breakdown.slope)
                    .exp()
                    .scale(self.saturation_current)
            }
            _ => junction_current(voltage, self.saturation_current, self.slope),
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::testing::{
        Bench, assert_close, assert_jacobians_match, assert_parameter_gradients_match, source,
        thermal,
    };
    use crate::{ELEMENTARY_CHARGE, thermal_voltage};

    const THERMAL_VOLTAGE: f64 = thermal_voltage(REFERENCE_TEMPERATURE);

    const CHARGED: &str =
        r#"{"junction_capacitance": 2e-12, "transit_time": 1e-9, "grading_coefficient": 0.4}"#;

    #[test]
    fn follows_the_shockley_equation() {
        let mut bench = Bench::new::<Diode>("{}");
        let current = bench.settle(&[0.6, 0.0]).residual[0];
        let expected = 1e-14 * ((0.6 / THERMAL_VOLTAGE).exp() - 1.0);
        assert!(
            (current / expected - 1.0).abs() < 1e-12,
            "{current} vs {expected}"
        );
    }

    #[test]
    fn the_saturation_current_scales_with_temperature() {
        let (hot, emission) = (350.0, 1.5);
        let mut bench = Bench::at::<Diode>(r#"{"emission_coefficient": 1.5}"#, hot);
        let current = bench.settle(&[0.6, 0.0]).residual[0];
        let vt = thermal_voltage(hot);
        let ratio = hot / REFERENCE_TEMPERATURE;
        let saturation = 1e-14 * (((ratio - 1.0) * 1.11 / vt + 3.0 * ratio.ln()) / emission).exp();
        let expected = saturation * ((0.6 / (emission * vt)).exp() - 1.0);
        assert!(
            (current / expected - 1.0).abs() < 1e-12,
            "{current} vs {expected}"
        );
    }

    #[test]
    fn limits_large_steps_until_they_settle() {
        let mut bench = Bench::new::<Diode>("{}");
        let evaluation = bench.evaluate(&[10.0, 0.0]);
        assert!(evaluation.limited);
        assert!(evaluation.residual[0].is_finite());
        assert!(bench.evaluate(&[-10.0, 0.0]).limited);
        assert!(!bench.settle(&[-10.0, 0.0]).limited);
    }

    #[test]
    fn a_series_resistance_adds_an_internal_node() {
        let bench = Bench::new::<Diode>(r#"{"series_resistance": 10}"#);
        assert_eq!(bench.unknowns(), ["n0", "n1", "device:junction"]);
    }

    #[test]
    fn a_small_series_resistance_carries_its_current_as_an_unknown() {
        let small = r#"{"series_resistance": 0.5, "junction_capacitance": 1e-12}"#;
        let mut bench = Bench::new::<Diode>(small);
        assert_eq!(
            bench.unknowns(),
            ["n0", "n1", "device:junction", "device:junction_current"]
        );
        let x = [0.7, 0.0, 0.65, 0.1];
        let residual = bench.evaluate(&x).residual;
        assert!((residual[0] - 0.1).abs() < 1e-15, "{residual:?}");
        assert!(residual[3].abs() < 1e-15, "{residual:?}");
        assert_jacobians_match::<Diode>(small, &x);
        assert_parameter_gradients_match::<Diode>(small, &x);
    }

    #[test]
    fn breaks_down_at_its_breakdown_current() {
        let mut bench = Bench::new::<Diode>(
            r#"{"breakdown_voltage": 5.1, "breakdown_current": 1e-3, "emission_coefficient": 1.2}"#,
        );
        let current = bench.settle(&[-5.1, 0.0]).residual[0];
        assert!((current / -1e-3 - 1.0).abs() < 1e-3, "{current}");
        let deeper = bench.settle(&[-5.3, 0.0]).residual[0];
        let expected = current * (0.2 / (1.2 * THERMAL_VOLTAGE)).exp();
        assert!(
            (deeper / expected - 1.0).abs() < 1e-9,
            "{deeper} vs {expected}"
        );
    }

    #[test]
    fn rejects_out_of_range_parameters() {
        let error = Bench::try_new::<Diode>(r#"{"grading_coefficient": 1}"#).unwrap_err();
        assert_eq!(
            error.to_string(),
            "grading_coefficient is at least 0 and below 1, not 1.0"
        );
    }

    proptest! {
        /// Verifies that analytical conductance and capacitance match finite differences on
        /// both sides of the depletion threshold.
        #[test]
        fn jacobians_match_finite_differences(voltage in -5.0..0.7f64) {
            assert_jacobians_match::<Diode>(CHARGED, &[voltage, 0.0]);
            let breaking = CHARGED.replace('{', r#"{"breakdown_voltage": 3, "#);
            assert_jacobians_match::<Diode>(&breaking, &[voltage, 0.0]);
            let with_resistance = CHARGED.replace('{', r#"{"series_resistance": 5, "#);
            assert_jacobians_match::<Diode>(&with_resistance, &[voltage + 0.01, 0.0, voltage]);
        }
    }

    #[test]
    fn its_current_makes_shot_and_flicker_noise() {
        let parameters = r#"{"saturation_current": 1e-14, "series_resistance": 10,
            "flicker_coefficient": 1e-12, "flicker_exponent": 1.5}"#;
        let mut bench = Bench::new::<Diode>(parameters);
        let x = [0.7, 0.0, 0.69];
        // The junction's current leaves through the cathode.
        let current = -bench.settle(&x).residual[1];
        let sources = bench.noise(&x);
        let shot = source(&sources, "shot");
        assert_close(&[shot.density(1.0)], &[2.0 * ELEMENTARY_CHARGE * current]);
        let flicker = source(&sources, "flicker");
        assert_eq!(flicker.equations, shot.equations);
        assert_close(
            &[flicker.density(100.0)],
            &[1e-12 * current.powf(1.5) / 100.0],
        );
        let series = source(&sources, "series thermal");
        assert_close(&[series.density(1.0)], &[thermal(0.1)]);
        assert_eq!(sources.len(), 3);
        let quiet = Bench::new::<Diode>(r#"{"saturation_current": 1e-14}"#).noise(&x[..2]);
        assert_eq!(quiet.len(), 1, "{quiet:?}");
    }

    /// Comprehensive diode fixture exercising parallel sidewall, recombination, tunneling,
    /// knees, breakdown, and all temperature coefficients away from nominal temperature.
    const EVERY_PART: &str = r#"{"saturation_current": 2e-14, "emission_coefficient": 1.3,
        "junction_capacitance": 2e-12, "junction_potential": 0.8, "grading_coefficient": 0.4,
        "depletion_coefficient": 0.6, "transit_time": 1e-9, "forward_knee_current": 1e-3,
        "reverse_knee_current": 1e-9, "recombination_saturation_current": 3e-13,
        "recombination_emission_coefficient": 1.8, "sidewall_saturation_current": 4e-15,
        "sidewall_emission_coefficient": 1.1, "sidewall_junction_capacitance": 5e-13,
        "sidewall_junction_potential": 0.6, "sidewall_grading_coefficient": 0.3,
        "sidewall_depletion_coefficient": 0.4, "sidewall_forward_knee_current": 2e-4,
        "tunneling_saturation_current": 1e-13, "sidewall_tunneling_saturation_current": 2e-14,
        "tunneling_emission_coefficient": 20,
        "tunneling_saturation_current_temperature_exponent": 2,
        "tunneling_energy_gap_factor": 0.8, "breakdown_voltage": 5, "breakdown_current": 1e-3,
        "breakdown_emission_coefficient": 1.1, "breakdown_temperature_coefficient": 1e-3,
        "temperature_model": 2, "energy_gap": 1.17, "energy_gap_alpha": 5e-4,
        "energy_gap_beta": 1000, "saturation_current_temperature_exponent": 2.5,
        "series_resistance_temperature_coefficient": 2e-3,
        "series_resistance_quadratic_temperature_coefficient": 1e-5,
        "transit_time_temperature_coefficient": 1e-3,
        "transit_time_quadratic_temperature_coefficient": 1e-6,
        "grading_coefficient_temperature_coefficient": 1e-3,
        "grading_coefficient_quadratic_temperature_coefficient": 1e-6,
        "nominal_temperature": 290}"#;

    /// Returns [`EVERY_PART`] augmented with linear capacitance temperature scaling and a
    /// dedicated sidewall series resistance.
    fn separate() -> String {
        EVERY_PART.replace(
            '{',
            r#"{"series_resistance": 10, "sidewall_series_resistance": 40,
            "capacitance_temperature_model": 1, "junction_potential_temperature_coefficient": 2e-3,
            "junction_capacitance_temperature_coefficient": 1e-3,
            "sidewall_junction_potential_temperature_coefficient": 1e-3,
            "sidewall_junction_capacitance_temperature_coefficient": 2e-3, "#,
        )
    }

    #[test]
    fn the_sidewall_beside_the_junction_adds_its_current() {
        let mut bench = Bench::new::<Diode>(r#"{"sidewall_saturation_current": 3e-15}"#);
        let current = bench.settle(&[0.6, 0.0]).residual[0];
        let expected = 1.3e-14 * ((0.6 / THERMAL_VOLTAGE).exp() - 1.0);
        assert!(
            (current / expected - 1.0).abs() < 1e-12,
            "{current} vs {expected}"
        );
        // A dedicated sidewall emission coefficient sets an independent exponential slope.
        let own = r#"{"sidewall_saturation_current": 3e-15, "sidewall_emission_coefficient": 2}"#;
        let current = Bench::new::<Diode>(own).settle(&[0.6, 0.0]).residual[0];
        let expected = 1e-14 * ((0.6 / THERMAL_VOLTAGE).exp() - 1.0)
            + 3e-15 * ((0.3 / THERMAL_VOLTAGE).exp() - 1.0);
        assert!(
            (current / expected - 1.0).abs() < 1e-12,
            "{current} vs {expected}"
        );
    }

    #[test]
    fn a_sidewall_resistance_puts_the_sidewall_behind_a_node_of_its_own() {
        let parameters = r#"{"series_resistance": 10, "sidewall_series_resistance": 100,
            "sidewall_saturation_current": 3e-15, "sidewall_emission_coefficient": 1.5}"#;
        let mut bench = Bench::new::<Diode>(parameters);
        assert_eq!(
            bench.unknowns(),
            ["n0", "n1", "device:junction", "device:sidewall"]
        );
        let x = [0.7, 0.0, 0.65, 0.6];
        let residual = bench.settle(&x).residual;
        let sidewall = 3e-15 * ((0.6 / (1.5 * THERMAL_VOLTAGE)).exp() - 1.0);
        let junction = 1e-14 * ((0.65 / THERMAL_VOLTAGE).exp() - 1.0);
        let expected = [
            0.05 / 10.0 + 0.1 / 100.0,
            -(junction + sidewall),
            junction - 0.05 / 10.0,
            sidewall - 0.1 / 100.0,
        ];
        for (found, expected) in residual.iter().zip(expected) {
            assert!(
                (found - expected).abs() <= 1e-12 * expected.abs(),
                "{residual:?} vs {expected}"
            );
        }
        let small = parameters.replace("100,", "0.5,");
        assert_eq!(
            Bench::new::<Diode>(&small).unknowns(),
            [
                "n0",
                "n1",
                "device:junction",
                "device:sidewall",
                "device:sidewall_current"
            ]
        );
    }

    #[test]
    fn recombination_tunneling_and_knees_follow_their_closed_forms() {
        let (isr, nr, m, vj) = (1e-12, 2.0, 0.4, 0.8);
        let parameters = format!(
            r#"{{"recombination_saturation_current": {isr},
            "recombination_emission_coefficient": {nr},
            "grading_coefficient": {m}, "junction_potential": {vj}, "forward_knee_current": 1e-4,
            "reverse_knee_current": 1e-13, "tunneling_saturation_current": 1e-12,
            "tunneling_emission_coefficient": 20}}"#
        );
        let mut bench = Bench::new::<Diode>(&parameters);
        let generation = |v: f64| ((1.0 - v / vj).powi(2) + 0.005).powf(m / 2.0);
        let tunneling = |v: f64| 1e-12 * ((-v / (20.0 * THERMAL_VOLTAGE)).exp() - 1.0);
        let unsoftened = |v: f64| {
            1e-14 * ((v / THERMAL_VOLTAGE).exp() - 1.0)
                + isr * ((v / (nr * THERMAL_VOLTAGE)).exp() - 1.0) * generation(v)
                - tunneling(v)
        };
        let forward = unsoftened(0.6);
        let expected = forward / (1.0 + (forward / 1e-4).sqrt());
        let current = bench.settle(&[0.6, 0.0]).residual[0];
        assert!(
            (current / expected - 1.0).abs() < 1e-12,
            "{current} vs {expected}"
        );
        // Below three slopes, the recombination holds its value there, and the reverse
        // knee softens the reverse current.
        let edge = -3.0 * THERMAL_VOLTAGE;
        let held = isr * ((edge / (nr * THERMAL_VOLTAGE)).exp() - 1.0) * generation(edge);
        let cube = (3.0 * THERMAL_VOLTAGE / (-2.0 * std::f64::consts::E)).powi(3);
        let reverse = -1e-14 * (1.0 + cube) + held - tunneling(-2.0);
        let expected = reverse / (1.0 + (-reverse / 1e-13).sqrt());
        let current = bench.settle(&[-2.0, 0.0]).residual[0];
        assert!(
            (current / expected - 1.0).abs() < 1e-12,
            "{current} vs {expected}"
        );
    }

    #[test]
    fn the_band_gap_of_temperature_model_2_scales_the_saturation_current() {
        let hot = 360.0;
        let parameters = r#"{"temperature_model": 2, "emission_coefficient": 1.2}"#;
        let current = Bench::at::<Diode>(parameters, hot)
            .settle(&[0.5, 0.0])
            .residual[0];
        // Silicon's band gap by Varshni's law, and 1.16 eV at 0 K, the default here.
        let gap = |t: f64| 1.16 - 7.02e-4 * t * t / (t + 1108.0);
        let (vt, nominal) = (thermal_voltage(hot), THERMAL_VOLTAGE);
        let saturation = 1e-14
            * ((gap(REFERENCE_TEMPERATURE) / nominal - gap(hot) / vt
                + 3.0 * (hot / REFERENCE_TEMPERATURE).ln())
                / 1.2)
                .exp();
        let expected = saturation * ((0.5 / (1.2 * vt)).exp() - 1.0);
        assert!(
            (current / expected - 1.0).abs() < 1e-12,
            "{current} vs {expected}"
        );
    }

    #[test]
    fn capacitance_temperature_model_1_scales_from_the_nominal_temperature() {
        let (hot, nominal) = (340.0, 310.0);
        let parameters = format!(
            r#"{{"capacitance_temperature_model": 1, "junction_capacitance": 1e-12,
            "junction_potential": 0.7, "grading_coefficient": 0.5,
            "junction_potential_temperature_coefficient": 2e-3,
            "junction_capacitance_temperature_coefficient": 1e-3,
            "nominal_temperature": {nominal}}}"#
        );
        let evaluation = Bench::at::<Diode>(&parameters, hot).settle(&[-1.0, 0.0]);
        let potential = 0.7 - 2e-3 * (hot - nominal);
        let capacitance = 1e-12 * (1.0 + 1e-3 * (hot - nominal));
        let expected = capacitance / (1.0 + 1.0 / potential).sqrt();
        let found = evaluation.capacitance[0][0];
        assert!(
            (found / expected - 1.0).abs() < 1e-12,
            "{found} vs {expected}"
        );
    }

    #[test]
    fn temperature_models_1_and_2_lower_the_breakdown_voltage_relatively() {
        let breakdown = |model: u8| {
            let parameters = format!(
                r#"{{"breakdown_voltage": 6, "breakdown_temperature_coefficient": 1e-3,
                "temperature_model": {model}}}"#
            );
            Bench::at::<Diode>(&parameters, REFERENCE_TEMPERATURE + 50.0)
                .settle(&[-6.0, 0.0])
                .residual[0]
        };
        // Model 0 lowers BV by 1 mV/K to 5.95 V; model 1 lowers BV by 0.1%/K to 5.7 V, placing
        // 6 V deeper by 0.25 V and raising current by `e^{0.25/Vt}`. Matching shifts
        // `Vt · ln(Ibv/Is + 1 - BV/Vt)` differ by 8.3 in 1.4e8 (6e-8 relative).
        let ratio = breakdown(1) / breakdown(0);
        let expected = (0.25 / thermal_voltage(REFERENCE_TEMPERATURE + 50.0)).exp();
        assert!(
            (ratio / expected - 1.0).abs() < 1e-6,
            "{ratio} vs {expected}"
        );
    }

    #[test]
    fn every_parts_derivatives_are_exact() {
        for voltage in [0.65, 0.3, -0.02, -0.5, -4.0, -5.4] {
            assert_parameter_gradients_match::<Diode>(EVERY_PART, &[voltage, 0.0]);
            let x = [voltage + 0.02, 0.0, voltage, voltage + 0.01];
            assert_parameter_gradients_match::<Diode>(&separate(), &x);
        }
    }

    #[test]
    fn a_separate_sidewall_makes_its_own_noise() {
        let parameters = r#"{"series_resistance": 10, "sidewall_series_resistance": 40,
            "sidewall_saturation_current": 3e-15, "flicker_coefficient": 1e-12}"#;
        let bench = Bench::new::<Diode>(parameters);
        let x = [0.7, 0.0, 0.69, 0.68];
        let sources = bench.noise(&x);
        let names: Vec<_> = sources.iter().map(|source| source.name).collect();
        assert_eq!(
            names,
            [
                "series thermal",
                "shot",
                "flicker",
                "sidewall series thermal",
                "sidewall shot",
                "sidewall flicker"
            ]
        );
        let current = 3e-15 * ((0.68 / THERMAL_VOLTAGE).exp() - 1.0);
        let shot = source(&sources, "sidewall shot");
        assert_close(&[shot.density(1.0)], &[2.0 * ELEMENTARY_CHARGE * current]);
        let series = source(&sources, "sidewall series thermal");
        assert_close(&[series.density(1.0)], &[thermal(1.0 / 40.0)]);
    }

    proptest! {
        /// Verifies that analytical conductances and capacitances match finite differences
        /// across depletion thresholds, knees, and breakdown for all diode features.
        #[test]
        fn every_parts_jacobians_match_finite_differences(voltage in -5.4..0.75f64) {
            assert_jacobians_match::<Diode>(EVERY_PART, &[voltage, 0.0]);
            let x = [voltage + 0.02, 0.0, voltage, voltage + 0.01];
            assert_jacobians_match::<Diode>(&separate(), &x);
        }
    }
}
