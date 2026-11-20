//! Vertical double-diffused power MOSFET (VDMOS) kernel, derived from ngspice 47
//! `src/spicelib/devices/vdmos/`: Copyright 1985 - 2018, Regents of the University of
//! California and others. Used under the ngspice Modified BSD license, reproduced in the
//! repository `NOTICE` file.

use std::f64::consts::FRAC_PI_2;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::dimension::{
    CAPACITANCE, CURRENT, PER_KELVIN, PER_KELVIN_SQUARED, PER_VOLT, RESISTANCE, TEMPERATURE, TIME,
    TRANSCONDUCTANCE_PARAMETER, VOLTAGE, VOLTS_PER_KELVIN,
};
use crate::kernel::diode::{Breakdown, Junction};
use crate::kernel::dual::Dual;
use crate::kernel::jfet::Channel;
use crate::kernel::junction::{Depletion, Temperatures, below_one, limit_channel};
use crate::kernel::mosfet::{DEFAULT_OXIDE_THICKNESS, OXIDE_PERMITTIVITY};
use crate::kernel::parts::{self, Bias, Derivatives, Nodes, Oriented, Pair, Series, Varying};
use crate::kernel::region::SATURATION_MARGIN;
use crate::noise::Flicker;
use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Lift, Local, NoiseSource, ParameterError,
    ParameterKind, ParameterMetadata, REFERENCE_TEMPERATURE, REGION, Real, Regions, Stamp,
};

/// Default drain-source shunt conductance added when `drain_source_resistance` is omitted,
/// in siemens (independent of `multiplier`), matching ngspice.
const LEAKAGE: f64 = 1e-15;

/// Upper limit on the body diode depletion threshold `FC · VJ` at operating temperature, in
/// volts. Above 2.5 V, ngspice replaces junction potential with `2.5 / N`, introducing a
/// discontinuity in depletion charge.
const MAX_DEPLETION_THRESHOLD: f64 = 2.5;

/// Vertical double-diffused power MOSFET (`VDMOS`) between `drain`, `gate`, and `source`,
/// matching ngspice 47.
///
/// Channel conduction follows a Shichman-Hodges square law without body effect, softened
/// below threshold via `vgst' = k · ln(1 + e^{(vgst - shift) / k})` where `vgst = vgs - vto`
/// and `k` is the subthreshold slope. Drain current uses
/// `β' = kp · m · (1 + λ · vds) / (1 + θ · max(vgst, 0))` and `vdss = mtriode · vds`.
/// Saturation gives `β' · vgst'² / 2`, and linear mode gives
/// `β' · vdss · (vgst' - vdss / 2)`. Drain and source swap when `vds < 0`, while `vds`
/// retains its sign in channel-length modulation. When quasi-saturation parameters are
/// provided, forward drain resistance increases as `rd + rq · vds / (vds + vq)`.
///
/// Gate-source capacitance is constant, while gate-drain capacitance transitions smoothly
/// from `cgdmin` (negative `vgd`) to `cgdmax` (positive `vgd`). Negative `vgd` uses
/// `s · atan(a · vgd) + y` and positive `vgd` uses `s · tanh(a · vgd) + y`. The scale and
/// offset are `s = (cgdmax - cgdmin) / (1 + π/2)` and `y = cgdmax - s`. Integrating
/// capacitance analytically conserves charge.
///
/// A dedicated body diode joins the source (behind body resistance) to the drain terminal,
/// in parallel with a drain-source shunt conductance. Positive drain, source, gate, or body
/// resistances add internal nodes (`drain`, `source`, `gate`, `body_diode`), and resistances
/// below [`SMALL_RESISTANCE`](crate::SMALL_RESISTANCE) add branch-current unknowns
/// (`drain_current`, `source_current`, `gate_current`, `body_diode_current`).
#[derive(Debug, Clone, Copy)]
pub struct Vdmos;

/// Layout shape for [`Vdmos`] storing series resistances, intrinsic node matrix, optional
/// quasi-saturation coupling, body diode `Pair`, and drain-source shunt `Pair`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VdmosShape {
    stamp: Nodes<3>,
    /// Drain, source, gate, and body series resistances.
    series: [Series; 4],
    /// Matrix entries coupling quasi-saturated drain resistance to terminal `vds`.
    quasi: Option<Varying>,
    /// Body diode `Pair` oriented anode to cathode (body node to drain for n-channel,
    /// reversed for p-channel).
    body: Pair,
    /// Drain-to-source terminal `Pair` carrying shunt conductance.
    shunt: Pair,
}

/// Prepared VDMOS parameters at operating temperature.
#[derive(Debug, Clone, Copy)]
pub struct PreparedVdmos<R: Real> {
    sign: f64,
    /// Drain, source, gate, and body resistances scaled by `multiplier`.
    resistances: [R; 4],
    quasi: Option<Quasi<R>>,
    /// Drain-to-source shunt conductance.
    shunt: R,
    channel: Law<R>,
    gate: Gate<R>,
    body: Junction<R>,
    transit_time: R,
    /// Threshold voltage in n-channel polarity used for gate step limiting.
    threshold: f64,
    temperature: R,
    /// Drain-current flicker noise scaled by the inverse square of default oxide capacitance
    /// per unit area.
    flicker: Flicker<R>,
}

/// Persistent evaluation memory for [`Vdmos`] storing previous channel voltages, body diode
/// voltage, and diagnostic variables.
#[derive(Debug, Clone, Copy, Default)]
pub struct VdmosMemory {
    /// Gate-source and drain-source voltages from the last evaluation in n-channel polarity.
    last: [f64; 2],
    /// Body diode voltage from the last evaluation, anode to cathode.
    body: f64,
    /// Values of [`VARIABLE_NAMES`] from the last evaluation.
    variables: [f64; 7],
}

/// Diagnostic variable names reported by [`Vdmos`]: threshold and saturation voltages in
/// device polarity, terminal capacitances (including body diode), operating region from
/// [`REGIONS`], and saturation margin.
const VARIABLE_NAMES: [&str; 7] = [
    "threshold_voltage",
    "saturation_voltage",
    "gate_source_capacitance",
    "gate_drain_capacitance",
    "drain_source_capacitance",
    REGION,
    SATURATION_MARGIN,
];

/// Operating regions of a VDMOS channel matching `VDMOSload` in the order of [`REGIONS`].
///
/// Softened subthreshold overdrive keeps the channel conducting below threshold, so no
/// cutoff region exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Region {
    /// Drain-source voltage below saturation.
    Linear,
    /// Drain-source voltage at or past saturation.
    Saturated,
}

/// Operating-region metadata for [`Vdmos`] reporting `vds - vdsat` in n-channel sign
/// convention per [`Regions::margins`], swapping drain and source when `vds < 0`.
///
/// The boundary `vdsat` equals the softened overdrive divided by the triode multiplier.
const REGIONS: Regions = Regions {
    names: &["linear", "saturated"],
    margins: &[SATURATION_MARGIN],
};

/// The terminals' local indices.
const DRAIN: usize = 0;
const GATE: usize = 1;
const SOURCE: usize = 2;

/// Stamped node indices: the gate, the drain and the source behind their resistances,
/// which are the terminals themselves without one.
const INNER_GATE: usize = 0;
const INNER_DRAIN: usize = 1;
const INNER_SOURCE: usize = 2;

/// The pairs of stamped nodes across which the voltages the channel's equations take as
/// variables lie: gate-source and drain-source.
const VARIABLES: [(usize, usize); 2] = [(INNER_GATE, INNER_SOURCE), (INNER_DRAIN, INNER_SOURCE)];

/// The parameters of a [`Vdmos`], with ngspice's defaults.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift, ParameterMetadata)]
#[serde(default, deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct VdmosParameters<R: Real = f64> {
    /// The channel, `VDMOS` or `VDMOS pchan`; default `n`.
    pub channel: Channel,
    /// The number of identical transistors in parallel, `M`; default 1.
    pub multiplier: R,
    /// The threshold voltage, `VTO`, in volts; default 3 for an n-channel and -3 for a
    /// p-channel, an IRF540's and an IRF9540's.
    #[dimension(VOLTAGE)]
    pub threshold_voltage: Option<R>,
    /// The transconductance parameter, `KP`, in amperes per square volt; default 20 for an
    /// n-channel and 10 for a p-channel.
    #[dimension(TRANSCONDUCTANCE_PARAMETER)]
    pub transconductance_parameter: Option<R>,
    /// The channel-length modulation, `LAMBDA`, per volt; default 0.
    #[dimension(PER_VOLT)]
    pub channel_length_modulation: R,
    /// The gate field's degradation of the mobility, `THETA`, per volt; default 0.
    #[dimension(PER_VOLT)]
    pub mobility_modulation: R,
    /// The factor on the drain-source voltage below saturation, `MTRIODE`; default 1.
    pub triode_conductance_multiplier: R,
    /// The slope of the overdrive's softening below the threshold, `KSUBTHRES`, in volts;
    /// default 0.1.
    #[dimension(VOLTAGE)]
    pub subthreshold_slope: R,
    /// The shift of the softened overdrive along the gate voltage, `SUBSHIFT`, in volts;
    /// default 0.
    #[dimension(VOLTAGE)]
    pub subthreshold_shift: R,
    /// The drain resistance, `RD`, in ohms; default 0.
    #[dimension(RESISTANCE)]
    pub drain_resistance: R,
    /// The source resistance, `RS`, in ohms; default 0.
    #[dimension(RESISTANCE)]
    pub source_resistance: R,
    /// The gate resistance, `RG`, in ohms; default 0.
    #[dimension(RESISTANCE)]
    pub gate_resistance: R,
    /// The quasi-saturation resistance, `RQ`, in ohms, which the drain resistance grows
    /// towards with the drain voltage; default none, for no quasi-saturation.
    #[dimension(RESISTANCE)]
    pub quasi_saturation_resistance: Option<R>,
    /// The quasi-saturation voltage, `VQ`, in volts, at which the drain resistance has
    /// grown by half the quasi-saturation resistance; default none, for no
    /// quasi-saturation.
    #[dimension(VOLTAGE)]
    pub quasi_saturation_voltage: Option<R>,
    /// The drain-source shunt resistance, `RDS`, in ohms; default none, for ngspice's
    /// conductance of 1 fS whatever the multiplier.
    #[dimension(RESISTANCE)]
    pub drain_source_resistance: Option<R>,
    /// The gate-source capacitance, `CGS`, in farads; default 1.4 nF.
    #[dimension(CAPACITANCE)]
    pub gate_source_capacitance: R,
    /// The gate-drain capacitance far above the drain, `CGDMAX`, in farads; default 2 nF.
    #[dimension(CAPACITANCE)]
    pub max_gate_drain_capacitance: R,
    /// The gate-drain capacitance far below the drain, `CGDMIN`, in farads; default
    /// 20 pF.
    #[dimension(CAPACITANCE)]
    pub min_gate_drain_capacitance: R,
    /// Gate-drain capacitance transition steepness `A`, per volt; default 1.
    #[dimension(PER_VOLT)]
    pub gate_drain_capacitance_coefficient: R,
    /// The body diode's saturation current, `IS`, in amperes; default 1e-14.
    #[dimension(CURRENT)]
    pub saturation_current: R,
    /// The body diode's emission coefficient, `N`; default 1.
    pub emission_coefficient: R,
    /// The body diode's series resistance, `RB`, in ohms; default 0.
    #[dimension(RESISTANCE)]
    pub body_resistance: R,
    /// The body diode's zero-bias junction capacitance, `CJO`, in farads; default
    /// 0.5 nF.
    #[dimension(CAPACITANCE)]
    pub junction_capacitance: R,
    /// The body diode's junction potential, `VJ`, in volts; default 0.8.
    #[dimension(VOLTAGE)]
    pub junction_potential: R,
    /// The body diode's grading coefficient, `M`, below 1; default 0.5.
    pub grading_coefficient: R,
    /// The fraction of the body diode's junction potential above which its depletion
    /// capacitance is extended linearly, `FC`, below 1; default 0.5.
    pub depletion_coefficient: R,
    /// The body diode's transit time, `TT`, in seconds; default 0.
    #[dimension(TIME)]
    pub transit_time: R,
    /// The drain-source breakdown voltage, `BV`, in volts, positive; without it, the body
    /// diode never breaks down.
    #[dimension(VOLTAGE)]
    pub breakdown_voltage: Option<R>,
    /// The current at the breakdown voltage, `IBV`, in amperes, whatever the multiplier;
    /// default 1e-10.
    #[dimension(CURRENT)]
    pub breakdown_current: R,
    /// The breakdown emission coefficient, `NBV`; default 1.
    pub breakdown_emission_coefficient: R,
    /// The energy gap that scales the body diode's saturation current with temperature,
    /// `EG`, in volts; default 1.11, silicon's.
    #[dimension(VOLTAGE)]
    pub energy_gap: R,
    /// Temperature-ratio exponent `XTI` scaling body-diode saturation current; default 3.
    pub saturation_current_temperature_exponent: R,
    /// Nominal measurement temperature `TNOM`, in kelvins; default 27 °C.
    #[dimension(TEMPERATURE)]
    pub nominal_temperature: R,
    /// Linear threshold voltage temperature coefficient `TCVTH` (`VTOTC`), in volts per
    /// kelvin; default 0.
    #[dimension(VOLTS_PER_KELVIN)]
    pub threshold_voltage_temperature_coefficient: R,
    /// The exponent of the temperature ratio that scales the transconductance parameter,
    /// `MU` or `BEX`; default -1.5.
    pub transconductance_parameter_temperature_exponent: R,
    /// The subthreshold slope's linear temperature coefficient, `TKSUBTHRES1`, per kelvin;
    /// default 0.
    #[dimension(PER_KELVIN)]
    pub subthreshold_slope_temperature_coefficient: R,
    /// The subthreshold slope's quadratic temperature coefficient, `TKSUBTHRES2`, per
    /// square kelvin; default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub subthreshold_slope_quadratic_temperature_coefficient: R,
    /// The exponent of the temperature ratio that scales the drain resistance, `TEXP0`;
    /// default none, for the drain resistance's temperature coefficients.
    pub drain_resistance_temperature_exponent: Option<R>,
    /// The drain resistance's linear temperature coefficient, `TRD1`, per kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub drain_resistance_temperature_coefficient: R,
    /// The drain resistance's quadratic temperature coefficient, `TRD2`, per square kelvin;
    /// default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub drain_resistance_quadratic_temperature_coefficient: R,
    /// The exponent of the temperature ratio that scales the quasi-saturation resistance,
    /// `TEXP1`; default 0.3.
    pub quasi_saturation_resistance_temperature_exponent: R,
    /// The source resistance's linear temperature coefficient, `TRS1`, per kelvin; default
    /// 0.
    #[dimension(PER_KELVIN)]
    pub source_resistance_temperature_coefficient: R,
    /// The source resistance's quadratic temperature coefficient, `TRS2`, per square
    /// kelvin; default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub source_resistance_quadratic_temperature_coefficient: R,
    /// The gate resistance's linear temperature coefficient, `TRG1`, per kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub gate_resistance_temperature_coefficient: R,
    /// The gate resistance's quadratic temperature coefficient, `TRG2`, per square kelvin;
    /// default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub gate_resistance_quadratic_temperature_coefficient: R,
    /// The body resistance's linear temperature coefficient, `TRB1`, per kelvin; default 0.
    #[dimension(PER_KELVIN)]
    pub body_resistance_temperature_coefficient: R,
    /// The body resistance's quadratic temperature coefficient, `TRB2`, per square kelvin;
    /// default 0.
    #[dimension(PER_KELVIN_SQUARED)]
    pub body_resistance_quadratic_temperature_coefficient: R,
    /// The flicker noise coefficient of the drain current, `KF`; default 0, no flicker
    /// noise.
    pub flicker_coefficient: R,
    /// The power of the drain current in the flicker noise, `AF`; default 1.
    pub flicker_exponent: R,
}

impl<R: Real> Default for VdmosParameters<R> {
    fn default() -> Self {
        let zero = R::constant(0.0);
        Self {
            channel: Channel::N,
            multiplier: R::constant(1.0),
            threshold_voltage: None,
            transconductance_parameter: None,
            channel_length_modulation: zero,
            mobility_modulation: zero,
            triode_conductance_multiplier: R::constant(1.0),
            subthreshold_slope: R::constant(0.1),
            subthreshold_shift: zero,
            drain_resistance: zero,
            source_resistance: zero,
            gate_resistance: zero,
            quasi_saturation_resistance: None,
            quasi_saturation_voltage: None,
            drain_source_resistance: None,
            gate_source_capacitance: R::constant(1.4e-9),
            max_gate_drain_capacitance: R::constant(2e-9),
            min_gate_drain_capacitance: R::constant(2e-11),
            gate_drain_capacitance_coefficient: R::constant(1.0),
            saturation_current: R::constant(1e-14),
            emission_coefficient: R::constant(1.0),
            body_resistance: zero,
            junction_capacitance: R::constant(5e-10),
            junction_potential: R::constant(0.8),
            grading_coefficient: R::constant(0.5),
            depletion_coefficient: R::constant(0.5),
            transit_time: zero,
            breakdown_voltage: None,
            breakdown_current: R::constant(1e-10),
            breakdown_emission_coefficient: R::constant(1.0),
            energy_gap: R::constant(1.11),
            saturation_current_temperature_exponent: R::constant(3.0),
            nominal_temperature: R::constant(REFERENCE_TEMPERATURE),
            threshold_voltage_temperature_coefficient: zero,
            transconductance_parameter_temperature_exponent: R::constant(-1.5),
            subthreshold_slope_temperature_coefficient: zero,
            subthreshold_slope_quadratic_temperature_coefficient: zero,
            drain_resistance_temperature_exponent: None,
            drain_resistance_temperature_coefficient: zero,
            drain_resistance_quadratic_temperature_coefficient: zero,
            quasi_saturation_resistance_temperature_exponent: R::constant(0.3),
            source_resistance_temperature_coefficient: zero,
            source_resistance_quadratic_temperature_coefficient: zero,
            gate_resistance_temperature_coefficient: zero,
            gate_resistance_quadratic_temperature_coefficient: zero,
            body_resistance_temperature_coefficient: zero,
            body_resistance_quadratic_temperature_coefficient: zero,
            flicker_coefficient: zero,
            flicker_exponent: R::constant(1.0),
        }
    }
}

/// Channel conduction law in n-channel polarity.
#[derive(Debug, Clone, Copy)]
struct Law<R: Real> {
    /// Scaled transconductance parameter `kp · m` at operating temperature.
    beta: R,
    /// Threshold voltage `von`.
    threshold: R,
    lambda: R,
    theta: R,
    /// Temperature-scaled subthreshold slope and threshold shift.
    slope: R,
    shift: R,
    /// Triode drain-source conductance multiplier.
    triode: R,
}

/// Channel evaluation result at a single bias point.
#[derive(Debug, Clone, Copy)]
struct Conduction<R: Real> {
    /// Drain-to-source current.
    current: Dual<R, 2>,
    /// Saturation voltage `max(vgst, 0)`.
    saturation: Dual<R, 2>,
    /// Flag indicating forward mode (`vds >= 0`); when `false`, drain and source swap.
    forward: bool,
    region: Region,
    /// Saturation margin `vds - vdsat` in forward orientation.
    margin: f64,
}

impl<R: Real> Law<R> {
    /// Evaluates drain current, saturation voltage, and region at `[vgs, vds]`, matching
    /// ngspice `VDMOSload`.
    fn evaluate(&self, [vgs, vds]: [Dual<R, 2>; 2]) -> Conduction<R> {
        let forward = vds.value.value() >= 0.0;
        // With the drain below the source, the drain acts as the source.
        let (control, across) = if forward {
            (vgs, vds)
        } else {
            (vgs - vds, -vds)
        };
        let vgst = control - Dual::constant(self.threshold);
        let saturation = vgst.max(0.0);
        let vdss = across.scale(self.triode);
        // ngspice's channel-length modulation reads the drain-source voltage with its sign.
        let betap = (vds.scale(self.lambda) + 1.0) / (saturation.scale(self.theta) + 1.0);
        let betap = betap.scale(self.beta);
        let overdrive =
            softplus((vgst - Dual::constant(self.shift)).divide(self.slope)).scale(self.slope);
        let beyond = (vdss - overdrive).value.value();
        let region = if beyond >= 0.0 {
            Region::Saturated
        } else {
            Region::Linear
        };
        let current = match region {
            Region::Saturated => betap * overdrive * overdrive * 0.5,
            Region::Linear => betap * vdss * (overdrive - vdss * 0.5),
        };
        Conduction {
            current: if forward { current } else { -current },
            saturation,
            forward,
            region,
            margin: beyond / self.triode.value(),
        }
    }
}

/// Computes `ln(1 + eˣ)` and its logistic derivative without overflow.
fn softplus<R: Real, const N: usize>(x: Dual<R, N>) -> Dual<R, N> {
    let at = x.value.value();
    let falling = (-at.abs()).exp();
    let logistic = if at >= 0.0 {
        1.0 / (1.0 + falling)
    } else {
        falling / (1.0 + falling)
    };
    let curvature = logistic * (1.0 - logistic);
    let value = at.max(0.0) + falling.ln_1p();
    x.chain((
        x.value.chain((value, logistic, curvature)),
        x.value
            .chain((logistic, curvature, curvature * (1.0 - 2.0 * logistic))),
    ))
}

/// Gate-source and nonlinear gate-drain capacitance parameters.
#[derive(Debug, Clone, Copy)]
struct Gate<R: Real> {
    /// Constant gate-source capacitance.
    source: R,
    /// Gate-drain capacitance swing `s = (cgdmax - cgdmin) / (1 + π/2)`.
    swing: R,
    /// Midpoint gate-drain capacitance `y = cgdmax - s` at `vgd = 0`.
    middle: R,
    /// Transition steepness `a` with respect to `vgd`.
    rate: R,
}

impl<R: Real> Gate<R> {
    /// Evaluates gate-drain charge (analytical integral of capacitance from 0 V) and
    /// capacitance derivative at `vgd`.
    fn drain_charge<const N: usize>(&self, vgd: Dual<R, N>) -> Dual<R, N> {
        let Self {
            swing,
            middle,
            rate,
            ..
        } = *self;
        let u = vgd.value * rate;
        let at = u.value();
        // Above zero, ∫ tanh(u) du = ln cosh(u), and below, ∫ atan(u) du = u · atan(u) -
        // ln(1 + u²) / 2: each written where it neither overflows nor cancels.
        let (integral, shape) = if at > 0.0 {
            let tanh = at.tanh();
            let square = 1.0 - tanh * tanh;
            (
                u.chain((
                    at + (-2.0 * at).exp().ln_1p() - std::f64::consts::LN_2,
                    tanh,
                    square,
                )),
                u.chain((tanh, square, -2.0 * tanh * square)),
            )
        } else {
            let atan = at.atan();
            let inverse = 1.0 / (1.0 + at * at);
            (
                u.chain((at * atan - 0.5 * (at * at).ln_1p(), atan, inverse)),
                u.chain((atan, inverse, -2.0 * at * inverse * inverse)),
            )
        };
        vgd.chain((
            swing * integral / rate + middle * vgd.value,
            swing * shape + middle,
        ))
    }
}

/// Quasi-saturation drain-resistance parameters.
#[derive(Debug, Clone, Copy)]
struct Quasi<R: Real> {
    /// Temperature- and multiplier-scaled quasi-saturation resistance `rq`.
    resistance: R,
    /// Half-saturation drain-source voltage `vq`.
    voltage: R,
}

impl<R: Real> Quasi<R> {
    /// Evaluates quasi-saturated drain resistance from base `drain` resistance and oriented
    /// terminal `vds`, falling back to `drain` when the formula turns non-positive.
    fn resistance<const N: usize>(&self, drain: R, vds: Dual<R, N>) -> Dual<R, N> {
        let raised = (vds.scale(self.resistance) / vds.plus(self.voltage)).plus(drain);
        if raised.value.value() > 0.0 {
            raised
        } else {
            Dual::constant(drain)
        }
    }
}

impl Kernel for Vdmos {
    const NAME: &'static str = "vdmos";
    type Parameters<R: Real> = VdmosParameters<R>;
    type Shape = VdmosShape;
    type Prepared<R: Real> = PreparedVdmos<R>;
    type Memory = VdmosMemory;

    fn parameter_kinds() -> &'static [(&'static str, ParameterKind)] {
        VdmosParameters::<f64>::PARAMETER_KINDS
    }

    fn interface<R: Real>(_: &VdmosParameters<R>) -> Interface {
        Interface::new(
            &["drain", "gate", "source"],
            &[
                Branch::new(DRAIN, SOURCE, BranchKind::Conductive),
                Branch::new(GATE, SOURCE, BranchKind::Capacitive),
                Branch::new(GATE, DRAIN, BranchKind::Capacitive),
            ],
        )
    }

    fn prepare<R: Real>(
        parameters: &VdmosParameters<R>,
        temperature: R,
    ) -> Result<PreparedVdmos<R>, ParameterError> {
        let temperatures = Temperatures::of_junction(temperature, parameters.nominal_temperature)?;
        let multiplier = ParameterError::positive("multiplier", parameters.multiplier)?;
        let channel = parameters.law(temperatures, multiplier)?;
        let resistances = parameters.resistances(temperatures, multiplier)?;
        let quasi = parameters.quasi(temperatures, multiplier)?;
        let shunt = match parameters.drain_source_resistance {
            Some(resistance) => {
                multiplier / ParameterError::positive("drain_source_resistance", resistance)?
            }
            None => R::constant(LEAKAGE),
        };
        let flicker = Flicker::new(parameters.flicker_coefficient, parameters.flicker_exponent)?;
        let oxide = OXIDE_PERMITTIVITY / DEFAULT_OXIDE_THICKNESS;
        Ok(PreparedVdmos {
            sign: parameters.channel.sign(),
            resistances,
            quasi,
            shunt,
            threshold: channel.threshold.value(),
            channel,
            gate: parameters.gate(multiplier)?,
            body: parameters.body(temperatures, multiplier)?,
            transit_time: ParameterError::non_negative("transit_time", parameters.transit_time)?,
            temperature: temperatures.actual,
            flicker: flicker.scaled(R::constant(1.0 / (oxide * oxide))),
        })
    }

    fn layout<R: Real>(
        parameters: &VdmosParameters<R>,
        prepared: &PreparedVdmos<R>,
        layout: &mut Layout,
    ) -> Result<VdmosShape, ParameterError> {
        let [drain, source, gate, body] = prepared.resistances;
        let (drain_series, inner_drain) =
            Series::behind(layout, DRAIN, drain, ("drain", "drain_current"));
        let (source_series, inner_source) =
            Series::behind(layout, SOURCE, source, ("source", "source_current"));
        let (gate_series, inner_gate) =
            Series::behind(layout, GATE, gate, ("gate", "gate_current"));
        let (body_series, anode) =
            Series::behind(layout, SOURCE, body, ("body_diode", "body_diode_current"));
        let stamp = Nodes::new(layout, [inner_gate, inner_drain, inner_source]);
        let body = match parameters.channel {
            Channel::N => Pair::new(layout, anode, DRAIN),
            Channel::P => Pair::new(layout, DRAIN, anode),
        };
        let shunt = Pair::new(layout, DRAIN, SOURCE);
        let quasi = prepared
            .quasi
            .map(|_| drain_series.varying(layout, (DRAIN, SOURCE)));
        Ok(VdmosShape {
            stamp,
            series: [drain_series, source_series, gate_series, body_series],
            quasi,
            body,
            shunt,
        })
    }

    fn evaluate<R: Real>(
        shape: &VdmosShape,
        prepared: &PreparedVdmos<R>,
        memory: &mut VdmosMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let [drain_series, others @ ..] = &shape.series;
        let [drain, others_resistances @ ..] = prepared.resistances;
        for (series, resistance) in others.iter().zip(others_resistances) {
            series.add(series.value_of(resistance), at, stamp);
        }
        let sign = prepared.sign;
        let bias = Bias::limit(
            &shape.stamp,
            (sign, VARIABLES),
            &mut memory.last,
            at,
            stamp,
            |applied, last| {
                let (limited, changed) = limit_channel(applied, last, prepared.threshold);
                changed.then_some(limited)
            },
        );
        let added = Oriented::new(&shape.stamp, sign, &bias, VARIABLES);
        let [vgs, vds] = Dual::variables(added.across(at));
        let conduction = prepared.channel.evaluate([vgs, vds]);
        match (&shape.quasi, &prepared.quasi) {
            (Some(varying), Some(quasi)) if conduction.forward => {
                // A law of the voltage across the terminals, which orients it.
                let across = Dual::variable(at.unknown::<R>(DRAIN) - at.unknown::<R>(SOURCE));
                let resistance = quasi.resistance(drain, across * sign);
                drain_series.add_varying(resistance, varying, at, stamp);
            }
            _ => drain_series.add(drain_series.value_of(drain), at, stamp),
        }
        // Every derivative among the stamped nodes is summed here, and written once.
        let mut derivatives = Derivatives::new();
        let sums = &mut derivatives;
        added.add_current(stamp, sums, (INNER_DRAIN, INNER_SOURCE), conduction.current);
        let gate = &prepared.gate;
        let gate_drain = gate.drain_charge(vgs - vds);
        let gate_source = vgs.scale(gate.source);
        added.add_charge(stamp, sums, (INNER_GATE, INNER_SOURCE), gate_source);
        added.add_charge(stamp, sums, (INNER_GATE, INNER_DRAIN), gate_drain);
        derivatives.stamp(&shape.stamp, stamp);
        let body = prepared.body.add(
            &shape.body,
            prepared.transit_time,
            &mut memory.body,
            at,
            stamp,
        );
        shape.shunt.add_current(
            stamp,
            prepared.shunt * shape.shunt.across::<R>(at, shape.shunt.voltage(at)),
        );
        shape.shunt.add_conductance(stamp, prepared.shunt);
        let [by_gate, _] = gate_drain.gradient;
        memory.variables = [
            sign * prepared.threshold,
            sign * conduction.saturation.value.value(),
            gate.source.value(),
            by_gate.value(),
            body.value(),
            f64::from(conduction.region as u8),
            conduction.margin,
        ];
    }

    fn variable_names() -> &'static [&'static str] {
        &VARIABLE_NAMES
    }

    fn variables(memory: &VdmosMemory, values: &mut [f64]) {
        parts::write_variables(&memory.variables, values);
    }

    fn regions() -> Regions {
        REGIONS
    }

    /// Appends ngspice VDMOS thermal noise for drain, source, and channel (`8kT·gm / 3`),
    /// plus drain-current flicker noise `KF · |Id|^AF / (f · Cox²)` using the 0.1 µm oxide.
    ///
    /// Gate resistance, body resistance, and the body diode contribute no noise, matching
    /// ngspice.
    fn noise<R: Real>(
        shape: &VdmosShape,
        prepared: &PreparedVdmos<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource<R>>,
    ) {
        let temperature = prepared.temperature;
        let stamp = &shape.stamp;
        let sign = prepared.sign;
        let voltages = stamp.oriented::<R, _>(at, sign, VARIABLES);
        let conduction = prepared.channel.evaluate(Dual::variables(voltages));
        let [drain_series, source_series, ..] = &shape.series;
        let [drain, source, ..] = prepared.resistances;
        let drain = match (&shape.quasi, &prepared.quasi) {
            (Some(_), Some(quasi)) if conduction.forward => {
                let drop = (at.unknown::<R>(DRAIN) - at.unknown::<R>(SOURCE)) * sign;
                let across = Dual::<R, 0>::constant(drop);
                quasi.resistance(drain, across).value
            }
            _ => drain,
        };
        drain_series.noise(
            drain_series.value_of(drain),
            ("drain thermal", temperature),
            sources,
        );
        source_series.noise(
            source_series.value_of(source),
            ("source thermal", temperature),
            sources,
        );
        let channel = [stamp.equation(INNER_DRAIN), stamp.equation(INNER_SOURCE)];
        let [transconductance, _] = conduction.current.gradient;
        sources.push(NoiseSource::thermal(
            "channel thermal",
            channel,
            transconductance * (2.0 / 3.0),
            temperature,
        ));
        prepared
            .flicker
            .add(channel, conduction.current.value, sources);
    }
}

impl<R: Real> VdmosParameters<R> {
    /// Validates channel parameters and prepares the temperature-scaled [`Law`] for
    /// `multiplier` parallel copies at `temperatures`.
    ///
    /// Scales `kp` by `(T/Tn)^MU` and adjusts threshold voltage and subthreshold slope via
    /// their respective temperature coefficients.
    fn law(&self, temperatures: Temperatures<R>, multiplier: R) -> Result<Law<R>, ParameterError> {
        let sign = self.channel.sign();
        let transconductance = ParameterError::non_negative(
            "transconductance_parameter",
            self.transconductance_parameter
                .unwrap_or(R::constant(15.0 + 5.0 * sign)),
        )?;
        let threshold = self.threshold_voltage.unwrap_or(R::constant(3.0 * sign))
            + self.threshold_voltage_temperature_coefficient * temperatures.difference();
        let slope = ParameterError::positive("subthreshold_slope", self.subthreshold_slope)?
            * temperatures.polynomial(
                self.subthreshold_slope_temperature_coefficient,
                self.subthreshold_slope_quadratic_temperature_coefficient,
            );
        if slope.value() <= 0.0 {
            return Err(ParameterError(format!(
                "the temperature coefficients scale the subthreshold slope to zero at {:?} K",
                temperatures.actual.value()
            )));
        }
        Ok(Law {
            beta: transconductance
                * multiplier
                * temperatures
                    .ratio()
                    .pow(self.transconductance_parameter_temperature_exponent),
            threshold: threshold * sign,
            lambda: self.channel_length_modulation,
            theta: ParameterError::non_negative("mobility_modulation", self.mobility_modulation)?,
            slope,
            shift: self.subthreshold_shift,
            triode: ParameterError::positive(
                "triode_conductance_multiplier",
                self.triode_conductance_multiplier,
            )?,
        })
    }

    /// Validates and scales `[drain, source, gate, body]` series resistances at
    /// `temperatures` for `multiplier` parallel copies.
    ///
    /// Scales drain resistance by `(T/Tn)^TEXP0` when given, or by quadratic temperature
    /// coefficients otherwise.
    fn resistances(
        &self,
        temperatures: Temperatures<R>,
        multiplier: R,
    ) -> Result<[R; 4], ParameterError> {
        let drain = match self.drain_resistance_temperature_exponent {
            Some(exponent) => temperatures.ratio().pow(exponent),
            None => temperatures.polynomial(
                self.drain_resistance_temperature_coefficient,
                self.drain_resistance_quadratic_temperature_coefficient,
            ),
        };
        let scaled = |name: &str, resistance: R, factor: R| {
            let resistance = ParameterError::non_negative(name, resistance)?;
            if resistance.value() == 0.0 {
                return Ok(resistance);
            }
            let scaled = resistance / multiplier * factor;
            if scaled.value() <= 0.0 {
                return Err(ParameterError(format!(
                    "the temperature coefficients scale the {name} to zero at {:?} K",
                    temperatures.actual.value()
                )));
            }
            Ok(scaled)
        };
        Ok([
            scaled("drain_resistance", self.drain_resistance, drain)?,
            scaled(
                "source_resistance",
                self.source_resistance,
                temperatures.polynomial(
                    self.source_resistance_temperature_coefficient,
                    self.source_resistance_quadratic_temperature_coefficient,
                ),
            )?,
            scaled(
                "gate_resistance",
                self.gate_resistance,
                temperatures.polynomial(
                    self.gate_resistance_temperature_coefficient,
                    self.gate_resistance_quadratic_temperature_coefficient,
                ),
            )?,
            scaled(
                "body_resistance",
                self.body_resistance,
                temperatures.polynomial(
                    self.body_resistance_temperature_coefficient,
                    self.body_resistance_quadratic_temperature_coefficient,
                ),
            )?,
        ])
    }

    /// Prepares temperature-scaled quasi-saturation parameters (scaling `rq` by
    /// `(T/Tn)^TEXP1`) for `multiplier` parallel copies when both `rq` and `vq` are given.
    ///
    /// # Errors
    ///
    /// Returns an error when `drain_resistance` is zero; ngspice quasi-saturation requires an
    /// internal drain node and diverges without one.
    fn quasi(
        &self,
        temperatures: Temperatures<R>,
        multiplier: R,
    ) -> Result<Option<Quasi<R>>, ParameterError> {
        let (Some(resistance), Some(voltage)) = (
            self.quasi_saturation_resistance,
            self.quasi_saturation_voltage,
        ) else {
            return Ok(None);
        };
        let resistance = ParameterError::non_negative("quasi_saturation_resistance", resistance)?;
        let voltage = ParameterError::positive("quasi_saturation_voltage", voltage)?;
        if self.drain_resistance.value() <= 0.0 {
            return Err(ParameterError(
                "quasi-saturation needs a drain_resistance".to_owned(),
            ));
        }
        Ok(Some(Quasi {
            resistance: resistance / multiplier
                * temperatures
                    .ratio()
                    .pow(self.quasi_saturation_resistance_temperature_exponent),
            voltage,
        }))
    }

    /// Validates and scales gate capacitances for `multiplier` parallel copies.
    fn gate(&self, multiplier: R) -> Result<Gate<R>, ParameterError> {
        let capacitance = |name: &str, value: R| {
            ParameterError::non_negative(name, value).map(|value| value * multiplier)
        };
        let source = capacitance("gate_source_capacitance", self.gate_source_capacitance)?;
        let maximum = capacitance(
            "max_gate_drain_capacitance",
            self.max_gate_drain_capacitance,
        )?;
        let minimum = capacitance(
            "min_gate_drain_capacitance",
            self.min_gate_drain_capacitance,
        )?;
        let swing = (maximum - minimum) / (1.0 + FRAC_PI_2);
        Ok(Gate {
            source,
            swing,
            middle: maximum - swing,
            rate: ParameterError::positive(
                "gate_drain_capacitance_coefficient",
                self.gate_drain_capacitance_coefficient,
            )?,
        })
    }

    /// Prepares the temperature-scaled body diode [`Junction`] for `multiplier` parallel
    /// copies at `temperatures`.
    ///
    /// Scales saturation current with the energy gap, scales potential and capacitance with
    /// the silicon band gap, and matches breakdown to the shared breakdown current.
    ///
    /// # Errors
    ///
    /// Returns an error when the depletion threshold `FC · VJ` exceeds 2.5 V, where ngspice
    /// introduces a discontinuity in depletion charge.
    fn body(
        &self,
        temperatures: Temperatures<R>,
        multiplier: R,
    ) -> Result<Junction<R>, ParameterError> {
        let saturation = ParameterError::positive("saturation_current", self.saturation_current)?;
        let emission = ParameterError::positive("emission_coefficient", self.emission_coefficient)?;
        let capacitance =
            ParameterError::non_negative("junction_capacitance", self.junction_capacitance)?;
        let potential = ParameterError::positive("junction_potential", self.junction_potential)?;
        let grading = below_one("grading_coefficient", self.grading_coefficient)?;
        let coefficient = below_one("depletion_coefficient", self.depletion_coefficient)?;
        let current = ParameterError::positive("breakdown_current", self.breakdown_current)?;
        let breakdown_emission = ParameterError::positive(
            "breakdown_emission_coefficient",
            self.breakdown_emission_coefficient,
        )?;
        let gap = ParameterError::non_negative("energy_gap", self.energy_gap)?;
        let vt = temperatures.thermal_voltage();
        let saturation = saturation
            * multiplier
            * temperatures.saturation(gap, self.saturation_current_temperature_exponent, emission);
        let (potential, capacitance) =
            temperatures.junction(potential, capacitance * multiplier, grading);
        let threshold = (coefficient * potential).value();
        if threshold > MAX_DEPLETION_THRESHOLD {
            return Err(ParameterError(format!(
                "the body diode's depletion threshold, depletion_coefficient · \
                 junction_potential, is {threshold:?} V at {:?} K, beyond the 2.5 V where \
                 ngspice's depletion charge jumps",
                temperatures.actual.value()
            )));
        }
        let breakdown = self
            .breakdown_voltage
            .map(|voltage| {
                ParameterError::positive("breakdown_voltage", voltage).map(|voltage| {
                    Breakdown::new(voltage, current, saturation, breakdown_emission * vt, vt)
                })
            })
            .transpose()?;
        Ok(Junction::new(
            saturation,
            emission * vt,
            Depletion::new(capacitance, potential, grading, coefficient),
            breakdown,
        ))
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::testing::{
        Bench, assert_close, assert_jacobians_match, assert_regions, source, thermal,
    };
    use crate::thermal_voltage;

    /// A transistor with every part of its model in play: a softened threshold, the
    /// mobility's degradation, channel-length modulation, a triode multiplier, every
    /// resistance, the gate's capacitances and the body diode's charges and breakdown.
    const FULL: &str = r#"{"threshold_voltage": 3.5, "transconductance_parameter": 12,
        "channel_length_modulation": 0.02, "mobility_modulation": 0.08,
        "triode_conductance_multiplier": 0.7, "subthreshold_slope": 0.12,
        "subthreshold_shift": 0.05, "drain_resistance": 1.5, "source_resistance": 2,
        "gate_resistance": 3, "body_resistance": 4, "drain_source_resistance": 1e6,
        "gate_source_capacitance": 1.2e-9, "max_gate_drain_capacitance": 1.8e-9,
        "min_gate_drain_capacitance": 5e-11, "gate_drain_capacitance_coefficient": 0.4,
        "saturation_current": 5e-12, "emission_coefficient": 1.1,
        "junction_capacitance": 6e-10, "junction_potential": 0.75,
        "grading_coefficient": 0.45, "depletion_coefficient": 0.55, "transit_time": 2e-8,
        "breakdown_voltage": 60, "breakdown_current": 2.5e-4,
        "breakdown_emission_coefficient": 3}"#;

    /// Returns `parameters` augmented with quasi-saturation drain resistance parameters.
    fn quasi(parameters: &str) -> String {
        parameters.replace(
            '{',
            r#"{"quasi_saturation_resistance": 5, "quasi_saturation_voltage": 20, "#,
        )
    }

    /// Converts an n-channel VDMOS JSON parameter card to its p-channel mirror.
    fn p_channel(parameters: &str) -> String {
        parameters
            .replace(
                r#""threshold_voltage": 3.5"#,
                r#""threshold_voltage": -3.5"#,
            )
            .replace('{', r#"{"channel": "p", "#)
    }

    /// Channel-only VDMOS fixture with negligible body diode leakage and shunt conductance.
    const BARE: &str = r#"{"threshold_voltage": 3, "transconductance_parameter": 10,
        "channel_length_modulation": 0.01, "mobility_modulation": 0.05,
        "saturation_current": 1e-30, "drain_source_resistance": 1e30}"#;

    /// Evaluates drain current for `parameters` (such as [`BARE`]) at `vgs` and `vds` with
    /// grounded source.
    fn drain_current(parameters: &str, vgs: f64, vds: f64) -> f64 {
        Bench::new::<Vdmos>(parameters)
            .settle(&[vds, vgs, 0.0])
            .residual[0]
    }

    /// Computes reference ngspice forward drain current for `kp = 10`, `vto = 3`, `λ = 0.01`,
    /// `θ = 0.05`, and default subthreshold slope `0.1 V`.
    fn expected(vgs: f64, vds: f64, triode: f64) -> f64 {
        let vgst = vgs - 3.0;
        let betap = 10.0 * (1.0 + 0.01 * vds) / (1.0 + 0.05 * vgst.max(0.0));
        let overdrive = 0.1 * (vgst / 0.1).exp().ln_1p();
        let vdss = vds.abs() * triode;
        if overdrive <= vdss {
            betap * overdrive * overdrive / 2.0
        } else {
            betap * vdss * (overdrive - vdss / 2.0)
        }
    }

    #[test]
    fn follows_ngspices_softened_square_law() {
        for (vgs, vds) in [
            (5.0, 10.0),
            (5.0, 0.5),
            (2.8, 10.0),
            (1.0, 5.0),
            (3.0, 0.01),
        ] {
            let current = drain_current(BARE, vgs, vds);
            let expected = expected(vgs, vds, 1.0);
            assert!(
                (current - expected).abs() <= 1e-13 * expected.abs() + 1e-28,
                "{vgs} {vds}: {current} vs {expected}"
            );
        }
        let triode = BARE.replace('{', r#"{"triode_conductance_multiplier": 0.6, "#);
        let current = drain_current(&triode, 6.0, 1.0);
        let expected = expected(6.0, 1.0, 0.6);
        assert!(
            (current / expected - 1.0).abs() < 1e-13,
            "{current} {expected}"
        );
    }

    #[test]
    fn reversed_it_keeps_the_sign_of_the_drain_voltage_in_its_length_modulation() {
        // With the drain 0.5 V below the source, the gate's overdrive is over the drain, and
        // `1 + λ · vds` is ngspice's `1 - 0.005`.
        let current = drain_current(BARE, 5.0, -0.5);
        let vgst = 5.5 - 3.0;
        let overdrive = 0.1 * (vgst / 0.1).exp().ln_1p();
        let betap = 10.0 * (1.0 - 0.01 * 0.5) / (1.0 + 0.05 * vgst);
        let expected = -betap * 0.5 * (overdrive - 0.25);
        assert!(
            (current / expected - 1.0).abs() < 1e-13,
            "{current} {expected}"
        );
    }

    #[test]
    fn quasi_saturation_raises_the_drain_resistance_with_the_drain_voltage() {
        let parameters = r#"{"drain_resistance": 2, "quasi_saturation_resistance": 5,
            "quasi_saturation_voltage": 20}"#;
        let mut bench = Bench::new::<Vdmos>(parameters);
        // The drain at 30 V, the gate off, the internal drain 0.1 V below its terminal:
        // the drain resistance's current, into the drain, is 0.1 V over 2 + 5 · 30 / 50 Ω.
        // The body diode, reversed, leaks its saturation current into the drain.
        let vt = thermal_voltage(REFERENCE_TEMPERATURE);
        let reversed = |vd: f64| 1e-14 * (1.0 + (3.0 * vt / (std::f64::consts::E * vd)).powi(3));
        let x = [30.0, 0.0, 0.0, 29.9];
        let resistance = 2.0 + 5.0 * 30.0 / 50.0;
        let into = bench.settle(&x).residual[0];
        let expected = 0.1 / resistance + LEAKAGE * 30.0 + reversed(30.0);
        assert!((into / expected - 1.0).abs() < 1e-12, "{into} {expected}");
        // In reverse mode, drain resistance returns to base `rd` and the body diode conducts.
        let into = bench.settle(&[-0.3, 0.0, 0.0, -0.2]).residual[0];
        let diode = 1e-14 * ((0.3 / vt).exp() - 1.0);
        let expected = -0.1 / 2.0 - LEAKAGE * 0.3 - diode;
        assert!((into / expected - 1.0).abs() < 1e-12, "{into} {expected}");
    }

    #[test]
    fn the_gate_drain_charge_is_its_capacitances_integral() {
        let parameters = r#"{"max_gate_drain_capacitance": 2e-9,
            "min_gate_drain_capacitance": 1e-10, "gate_drain_capacitance_coefficient": 0.5,
            "gate_source_capacitance": 0, "junction_capacitance": 0}"#;
        let mut bench = Bench::new::<Vdmos>(parameters);
        let swing = (2e-9 - 1e-10) / (1.0 + FRAC_PI_2);
        let middle = 2e-9 - swing;
        for vgd in [-40.0, -2.0, -0.1, 0.0, 0.1, 3.0, 40.0] {
            // The gate against a grounded drain, the source below them both.
            let evaluation = bench.settle(&[0.0, vgd, -50.0]);
            let u = 0.5 * vgd;
            let (charge, capacitance) = if vgd > 0.0 {
                (
                    swing / 0.5 * u.cosh().ln() + middle * vgd,
                    swing * u.tanh() + middle,
                )
            } else {
                (
                    swing / 0.5 * (u * u.atan() - 0.5 * (1.0 + u * u).ln()) + middle * vgd,
                    swing * u.atan() + middle,
                )
            };
            assert_close(&[evaluation.charge[1]], &[charge]);
            assert_close(&[evaluation.capacitance[1][1]], &[capacitance]);
        }
        // Far above the drain, the capacitance is the maximum, and far below it tends to
        // the minimum, `y - s · π/2`.
        let above = bench.settle(&[0.0, 100.0, -50.0]).capacitance[1][1];
        assert!((above / 2e-9 - 1.0).abs() < 1e-12, "{above}");
        assert!(((middle - swing * FRAC_PI_2) / 1e-10 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_body_diode_breaks_down_at_its_breakdown_current() {
        let parameters = r#"{"breakdown_voltage": 40, "breakdown_current": 1e-4,
            "breakdown_emission_coefficient": 2}"#;
        let mut bench = Bench::new::<Vdmos>(parameters);
        let into = bench.settle(&[40.0, 0.0, 0.0]).residual[0] - LEAKAGE * 40.0;
        assert!((into / 1e-4 - 1.0).abs() < 1e-3, "{into}");
        let deeper = bench.settle(&[40.3, 0.0, 0.0]).residual[0] - LEAKAGE * 40.3;
        let expected = into * (0.3 / (2.0 * thermal_voltage(REFERENCE_TEMPERATURE))).exp();
        assert!(
            (deeper / expected - 1.0).abs() < 1e-9,
            "{deeper} vs {expected}"
        );
    }

    #[test]
    fn limits_large_steps_until_they_settle() {
        let mut bench = Bench::new::<Vdmos>(FULL);
        let x = [30.0, 10.0, 0.0, 29.0, 0.5, 9.9, -0.6];
        assert!(bench.evaluate(&x).limited);
        let evaluation = bench.settle(&x);
        assert!(evaluation.residual.iter().all(|value| value.is_finite()));
        // The body diode forward: a step to a volt is limited.
        let mut bench = Bench::new::<Vdmos>("{}");
        assert!(bench.evaluate(&[-1.0, 0.0, 0.0]).limited);
        assert!(!bench.settle(&[-1.0, 0.0, 0.0]).limited);
    }

    #[test]
    fn unknowns_follow_the_resistances() {
        let parameters = r#"{"drain_resistance": 0.05, "source_resistance": 2,
            "gate_resistance": 3, "body_resistance": 0.01}"#;
        assert_eq!(
            Bench::new::<Vdmos>(parameters).unknowns(),
            [
                "n0",
                "n1",
                "n2",
                "device:drain",
                "device:drain_current",
                "device:source",
                "device:gate",
                "device:body_diode",
                "device:body_diode_current"
            ]
        );
        assert_eq!(Bench::new::<Vdmos>("{}").unknowns().len(), 3);
    }

    #[test]
    fn rejects_inconsistent_parameters() {
        let error = |parameters| Bench::try_new::<Vdmos>(parameters).unwrap_err().0;
        assert_eq!(
            error(r#"{"quasi_saturation_resistance": 1, "quasi_saturation_voltage": 5}"#),
            "quasi-saturation needs a drain_resistance"
        );
        let beyond = error(r#"{"junction_potential": 5, "depletion_coefficient": 0.6}"#);
        assert!(
            beyond.starts_with(
                "the body diode's depletion threshold, depletion_coefficient · \
                 junction_potential, is 3.0"
            ) && beyond.ends_with(
                " V at 300.15 K, beyond the 2.5 V where ngspice's depletion charge jumps"
            ),
            "{beyond}"
        );
        assert_eq!(
            error(r#"{"subthreshold_slope": 0}"#),
            "subthreshold_slope is positive, not 0.0"
        );
        assert_eq!(
            error(
                r#"{"drain_resistance": 1, "drain_resistance_temperature_coefficient": -0.1,
                "nominal_temperature": 250}"#
            ),
            "the temperature coefficients scale the drain_resistance to zero at 300.15 K"
        );
    }

    proptest! {
        /// Verifies that analytical conductances and capacitances match finite differences
        /// across all regions, modes, channels, quasi-saturation, and body-diode regimes.
        #[test]
        fn jacobians_match_finite_differences(
            drain in -1.5..70.0f64,
            gate in -15.0..15.0f64,
            drops in proptest::array::uniform4(-0.5..0.5f64),
            p in proptest::bool::ANY,
            quasi_saturating in proptest::bool::ANY,
        ) {
            let parameters = if quasi_saturating { quasi(FULL) } else { FULL.to_owned() };
            let parameters = if p { p_channel(&parameters) } else { parameters };
            let sign = if p { -1.0 } else { 1.0 };
            let [d, g, s, b] = drops;
            let x = [drain, gate, 0.0, drain - d, s, gate - g, b].map(|v| sign * v);
            assert_jacobians_match::<Vdmos>(&parameters, &x);
            let small = parameters
                .replace(r#""drain_resistance": 1.5"#, r#""drain_resistance": 0.05"#)
                .replace(r#""body_resistance": 4"#, r#""body_resistance": 0.02"#);
            let x = [drain, gate, 0.0, drain - d, 0.1 * d, s, gate - g, b, 0.3 * b]
                .map(|v| sign * v);
            assert_jacobians_match::<Vdmos>(&small, &x);
        }
    }

    #[test]
    fn its_channel_and_resistances_make_thermal_and_flicker_noise() {
        let parameters = r#"{"drain_resistance": 2, "source_resistance": 0.5,
            "gate_resistance": 3, "body_resistance": 4, "flicker_coefficient": 1e-20,
            "flicker_exponent": 1.2, "saturation_current": 1e-30,
            "drain_source_resistance": 1e30}"#;
        let mut bench = Bench::new::<Vdmos>(parameters);
        // The drain behind 2 Ω, and the source behind 0.5 Ω, whose current is an unknown;
        // the gate's and the body diode's resistances make no noise, as in ngspice.
        let x = [10.0, 5.0, 0.0, 9.9, 0.025, -0.05, 5.0, 0.0];
        let evaluation = bench.settle(&x);
        // The inner drain's equation holds the drain resistance's current, 0.1 V over 2 Ω
        // out of it, and the channel's into it; the channel's transconductance is that
        // equation's derivative with respect to the inner gate.
        let channel = evaluation.residual[3] + 0.05;
        let transconductance = evaluation.conductance[3][6];
        let sources = bench.noise(&x);
        let oxide = OXIDE_PERMITTIVITY / DEFAULT_OXIDE_THICKNESS;
        assert_close(
            &[
                source(&sources, "drain thermal").density(1.0),
                source(&sources, "source thermal").density(1.0),
                source(&sources, "channel thermal").density(1.0),
                source(&sources, "flicker").density(100.0),
            ],
            &[
                thermal(0.5),
                thermal(0.5),
                thermal(2.0 / 3.0 * transconductance),
                1e-20 * channel.powf(1.2) / (100.0 * oxide * oxide),
            ],
        );
        assert_eq!(sources.len(), 4, "{sources:?}");
    }

    /// Verifies that [`Vdmos`] classifies linear and saturated regions (with softened
    /// subthreshold conduction remaining saturated), symmetric drain/source swap, and
    /// p-channel mirror polarity.
    ///
    /// Saturation margin vanishes at the `VDMOSload` boundary.
    #[test]
    fn reports_its_region_and_margin() {
        let parameters = r#"{"triode_conductance_multiplier": 0.5}"#;
        let mirror = parameters.replace('}', r#", "channel": "p"}"#);
        // The drain, the gate and the source, about the threshold of 3 V.
        assert_regions::<Vdmos>(
            parameters,
            Some(&mirror),
            &[
                (&[0.5, 6.0, 0.0], "linear", None),
                (&[10.0, 4.0, 0.0], "saturated", None),
                (&[10.0, 0.0, 0.0], "saturated", None),
                (&[0.0, 6.0, 0.5], "linear", None),
            ],
            &[
                (&[0.5, 6.0, 0.0], &[12.0, 6.0, 0.0], SATURATION_MARGIN),
                (&[0.0, 6.0, 0.5], &[0.0, 6.0, 12.0], SATURATION_MARGIN),
            ],
        );
    }
}
