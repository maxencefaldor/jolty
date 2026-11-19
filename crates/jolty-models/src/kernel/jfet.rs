//! Junction field-effect transistor (JFET) kernel, derived from ngspice 47
//! `src/spicelib/devices/jfet/`: Copyright 1985 - 2018, Regents of the University of
//! California and others. Used under the ngspice Modified BSD license, reproduced in the
//! repository `NOTICE` file.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::dimension::{
    CAPACITANCE, CURRENT, PER_VOLT, RESISTANCE, TEMPERATURE, TRANSCONDUCTANCE_PARAMETER, VOLTAGE,
    VOLTS_PER_KELVIN,
};
use crate::kernel::dual::Dual;
use crate::kernel::junction::{
    Depletion, Temperatures, critical_voltage, depletion_charge, junction_current, limit_gate,
    limit_junction,
};
use crate::kernel::parts::{self, Bias, Derivatives, Nodes, Oriented, Series, series_value};
use crate::kernel::region::{OVERDRIVE, SATURATION_MARGIN};
use crate::noise::Flicker;
use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Lift, Local, NoiseSource, ParameterError,
    ParameterKind, ParameterMetadata, REFERENCE_TEMPERATURE, REGION, Real, Regions, Stamp,
};

/// Channel polarity of a field-effect transistor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Channel {
    /// N-channel device conducting with gate voltage above threshold.
    N,
    /// P-channel device with voltages and currents reversed relative to n-channel.
    P,
}

impl Channel {
    /// Returns `1.0` for an n-channel device or `-1.0` for a p-channel device to orient
    /// voltages and currents into n-channel convention.
    #[must_use]
    pub const fn sign(self) -> f64 {
        match self {
            Self::N => 1.0,
            Self::P => -1.0,
        }
    }
}

/// Junction field-effect transistor implementing the SPICE model with the Sydney University
/// doping tail, matching ngspice.
///
/// Two pn junctions join the gate to the drain and source with grading coefficient `½`.
/// Drain current follows a square law softened near threshold by doping tail `b`. With
/// `vgst = vgs - vto` and `β' = β · (1 + λ · vds)`, saturation current equals
/// `β' · vgst² · (b + (1 - b) · vgst / (pb - vto))`. The channel is symmetric and swaps
/// drain and source when `vds < 0`. Positive drain or source resistance adds an internal
/// node (`drain` or `source`), and resistance below
/// [`SMALL_RESISTANCE`](crate::SMALL_RESISTANCE) adds a branch-current unknown
/// (`drain_current` or `source_current`).
///
/// Away from nominal temperature, gate saturation current scales with the energy gap,
/// junction potentials and capacitances scale with the silicon band gap, and threshold
/// voltage and transconductance parameter scale with dedicated coefficients, matching
/// ngspice.
#[derive(Debug, Clone, Copy)]
pub struct Jfet;

/// Layout shape for [`Jfet`] storing drain and source series resistances plus the intrinsic
/// gate, drain, and source node matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JfetShape {
    stamp: Nodes<STAMPED>,
    drain_series: Series,
    source_series: Series,
}

/// Prepared JFET parameters at operating temperature.
#[derive(Debug, Clone, Copy)]
pub struct PreparedJfet<R: Real> {
    sign: f64,
    /// Stamp values for drain and source series resistances.
    series: [R; 2],
    /// Thermal voltage at operating temperature.
    thermal_voltage: R,
    channel: DrainCurrent<R>,
    saturation_current: R,
    critical_voltage: f64,
    gate_source: Option<Depletion<R>>,
    gate_drain: Option<Depletion<R>>,
    temperature: f64,
    flicker: Flicker,
}

/// Persistent evaluation memory for [`Jfet`] storing previous gate-junction voltages and
/// diagnostic variables.
#[derive(Debug, Clone, Copy, Default)]
pub struct JfetMemory {
    /// Gate-source and gate-drain voltages from the last evaluation for step limiting.
    last: [f64; JUNCTIONS],
    /// Values of [`VARIABLE_NAMES`] from the last evaluation.
    variables: [f64; 3],
}

/// Diagnostic variable names reported by [`Jfet`]: operating region from [`REGIONS`] and
/// boundary margins.
const VARIABLE_NAMES: [&str; 3] = [REGION, OVERDRIVE, SATURATION_MARGIN];

/// Operating regions of a JFET channel matching `jfetload.c` and the order of [`REGIONS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Region {
    /// At or below threshold, conducting zero channel current.
    Off,
    /// Above threshold with drain voltage below saturation.
    Linear,
    /// Above threshold with drain voltage past saturation.
    Saturated,
}

/// Operating-region metadata for [`Jfet`] reporting `vgs - vto` and `vds - vdsat`
/// (`vdsat = max(vgs - vto, 0)`) in n-channel sign convention per [`Regions::margins`],
/// swapping drain and source when `vds < 0`.
const REGIONS: Regions = Regions {
    names: &["off", "linear", "saturated"],
    margins: &[OVERDRIVE, SATURATION_MARGIN],
};

/// Local node indices: the gate, then the drain and source behind their resistances, which
/// are the terminals themselves without one.
const GATE: usize = 0;
const INNER_DRAIN: usize = 1;
const INNER_SOURCE: usize = 2;

/// Number of stamped nodes.
const STAMPED: usize = 3;

/// Number of controlling branch voltages in the JFET equations.
const VARIABLE_COUNT: usize = 3;

/// Number of leading controlling voltages (the two gate junctions) subject to step limiting.
const JUNCTIONS: usize = 2;

/// Stamped node pairs for the three controlling voltages: gate-source, gate-drain, and
/// drain-source.
const VARIABLES: [(usize, usize); VARIABLE_COUNT] = [GATE_SOURCE, GATE_DRAIN, DRAIN_SOURCE];
const GATE_SOURCE: (usize, usize) = (GATE, INNER_SOURCE);
const GATE_DRAIN: (usize, usize) = (GATE, INNER_DRAIN);
const DRAIN_SOURCE: (usize, usize) = (INNER_DRAIN, INNER_SOURCE);

/// Variable positions in [`VARIABLES`], using SPICE names.
const VGS: usize = 0;
const VGD: usize = 1;
const VDS: usize = 2;

/// Subset of [`VARIABLES`] read by the channel current law: gate-source voltage followed by
/// drain-source voltage.
const CHANNEL: [usize; 2] = [VGS, VDS];

/// Parameters for [`Jfet`], with SPICE defaults.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift, ParameterMetadata)]
#[serde(default, deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct JfetParameters<R: Real = f64> {
    /// Channel polarity; default `n`.
    pub channel: Channel,
    /// Threshold voltage `vto`, in volts; default -2.
    #[dimension(VOLTAGE)]
    pub threshold_voltage: R,
    /// Transconductance parameter `β`, in amperes per square volt; default 1e-4.
    #[dimension(TRANSCONDUCTANCE_PARAMETER)]
    pub transconductance_parameter: R,
    /// Channel-length modulation `λ`, per volt; default 0.
    #[dimension(PER_VOLT)]
    pub channel_length_modulation: R,
    /// Drain series resistance, in ohms; default 0.
    #[dimension(RESISTANCE)]
    pub drain_resistance: R,
    /// Source series resistance, in ohms; default 0.
    #[dimension(RESISTANCE)]
    pub source_resistance: R,
    /// Zero-bias gate-source capacitance, in farads; default 0.
    #[dimension(CAPACITANCE)]
    pub gate_source_capacitance: R,
    /// Zero-bias gate-drain capacitance, in farads; default 0.
    #[dimension(CAPACITANCE)]
    pub gate_drain_capacitance: R,
    /// Gate junction built-in potential `pb`, in volts; default 1.
    #[dimension(VOLTAGE)]
    pub junction_potential: R,
    /// Gate junction saturation current, in amperes; default 1e-14.
    #[dimension(CURRENT)]
    pub saturation_current: R,
    /// Fraction of junction potential above which depletion capacitance extends linearly;
    /// below 1; default 0.5.
    pub depletion_coefficient: R,
    /// Doping tail parameter `b` (`1` gives the plain square law); default 1.
    pub doping_tail: R,
    /// Linear threshold voltage temperature coefficient, in volts per kelvin; default 0.
    #[dimension(VOLTS_PER_KELVIN)]
    pub threshold_voltage_temperature_coefficient: R,
    /// Temperature-ratio exponent scaling the transconductance parameter; default 0.
    pub transconductance_parameter_temperature_exponent: R,
    /// Energy gap scaling saturation current with temperature, in volts; default 1.11
    /// (silicon).
    #[dimension(VOLTAGE)]
    pub energy_gap: R,
    /// Temperature-ratio exponent scaling saturation current; default 0 (matching ngspice
    /// JFET defaults).
    pub saturation_current_temperature_exponent: R,
    /// Nominal measurement temperature, in kelvins; default 27 °C.
    #[dimension(TEMPERATURE)]
    pub nominal_temperature: R,
    /// Drain-current flicker noise coefficient `KF`; default 0 (no flicker noise).
    pub flicker_coefficient: R,
    /// Drain-current flicker noise exponent `AF`; default 1.
    pub flicker_exponent: R,
}

impl<R: Real> Default for JfetParameters<R> {
    fn default() -> Self {
        Self {
            channel: Channel::N,
            threshold_voltage: R::constant(-2.0),
            transconductance_parameter: R::constant(1e-4),
            channel_length_modulation: R::constant(0.0),
            drain_resistance: R::constant(0.0),
            source_resistance: R::constant(0.0),
            gate_source_capacitance: R::constant(0.0),
            gate_drain_capacitance: R::constant(0.0),
            junction_potential: R::constant(1.0),
            saturation_current: R::constant(1e-14),
            depletion_coefficient: R::constant(0.5),
            doping_tail: R::constant(1.0),
            threshold_voltage_temperature_coefficient: R::constant(0.0),
            transconductance_parameter_temperature_exponent: R::constant(0.0),
            energy_gap: R::constant(1.11),
            saturation_current_temperature_exponent: R::constant(0.0),
            nominal_temperature: R::constant(REFERENCE_TEMPERATURE),
            flicker_coefficient: R::constant(0.0),
            flicker_exponent: R::constant(1.0),
        }
    }
}

/// Channel drain-current law as a function of gate-source and drain-source voltages.
#[derive(Debug, Clone, Copy)]
struct DrainCurrent<R: Real> {
    beta: R,
    threshold: R,
    lambda: R,
    tail: R,
    /// Doping tail factor `(1 - b) / (pb - vto)`.
    tail_factor: R,
}

impl<R: Real> DrainCurrent<R> {
    /// Determines the channel [`Region`] and boundary margins `[vgs - vto, vds - vdsat]` at
    /// `vgs` and `vds`, swapping drain and source when `vds < 0`.
    fn region(&self, vgs: f64, vds: f64) -> (Region, [f64; 2]) {
        let (control, across) = if vds >= 0.0 {
            (vgs, vds)
        } else {
            (vgs - vds, -vds)
        };
        let vgst = control - self.threshold.value();
        let region = if vgst <= 0.0 {
            Region::Off
        } else if vgst >= across {
            Region::Linear
        } else {
            Region::Saturated
        };
        (region, [vgst, across - vgst.max(0.0)])
    }

    /// Evaluates drain-to-source channel current and derivatives at `vgs` and `vds`.
    fn evaluate(&self, vgs: Dual<R, 2>, vds: Dual<R, 2>) -> Dual<R, 2> {
        let Self {
            beta,
            threshold,
            lambda,
            tail,
            tail_factor,
        } = *self;
        let (region, _) = self.region(vgs.value.value(), vds.value.value());
        if region == Region::Off {
            return Dual::number(0.0);
        }
        if vds.value.value() >= 0.0 {
            let vgst = vgs.plus(-threshold);
            let betap = (vds.scale(lambda) + 1.0).scale(beta);
            let c = if region == Region::Linear {
                let a = (vgst - vds).scale(tail_factor * 3.0).plus(tail * 2.0);
                (vds.scale(tail_factor).plus(-tail) * vds + vgst * a) * vds
            } else {
                vgst * vgst * vgst.scale(tail_factor).plus(tail)
            };
            betap * c
        } else {
            // The drain and the source swap: the gate-drain voltage controls the channel.
            let vgdt = (vgs - vds).plus(-threshold);
            let betap = (1.0 - vds.scale(lambda)).scale(beta);
            if region == Region::Linear {
                let a = (vgdt + vds).scale(tail_factor * 3.0).plus(tail * 2.0);
                let c = ((-vds.scale(tail_factor)).plus(-tail) * -vds + vgdt * a) * vds;
                betap * c
            } else {
                -(betap * (vgdt * vgdt * vgdt.scale(tail_factor).plus(tail)))
            }
        }
    }
}

impl Kernel for Jfet {
    const NAME: &'static str = "jfet";
    type Parameters<R: Real> = JfetParameters<R>;
    type Shape = JfetShape;
    type Prepared<R: Real> = PreparedJfet<R>;
    type Memory = JfetMemory;

    fn parameter_kinds() -> &'static [(&'static str, ParameterKind)] {
        JfetParameters::<f64>::PARAMETER_KINDS
    }

    fn interface<R: Real>(_: &JfetParameters<R>) -> Interface {
        Interface::new(
            &["drain", "gate", "source"],
            &[
                Branch::new(0, 2, BranchKind::Conductive),
                Branch::new(1, 0, BranchKind::Conductive),
                Branch::new(1, 2, BranchKind::Conductive),
            ],
        )
    }

    fn prepare<R: Real>(
        parameters: &JfetParameters<R>,
        temperature: R,
    ) -> Result<PreparedJfet<R>, ParameterError> {
        let flicker = Flicker::new(parameters.flicker_coefficient, parameters.flicker_exponent)?;
        let [drain_resistance, source_resistance] = parameters.series_resistances()?;
        let Scaled {
            thermal_voltage,
            channel,
            saturation_current,
            critical_voltage,
            gate_source,
            gate_drain,
            temperature,
        } = parameters.scaled(temperature)?;
        Ok(PreparedJfet {
            sign: parameters.channel.sign(),
            series: [
                series_value(drain_resistance),
                series_value(source_resistance),
            ],
            thermal_voltage,
            channel,
            saturation_current,
            critical_voltage,
            gate_source,
            gate_drain,
            temperature,
            flicker,
        })
    }

    fn layout<R: Real>(
        parameters: &JfetParameters<R>,
        _: &PreparedJfet<R>,
        layout: &mut Layout,
    ) -> Result<JfetShape, ParameterError> {
        let [drain_resistance, source_resistance] = parameters.series_resistances()?;
        let [drain, gate, source] = [0, 1, 2];
        let (drain_series, inner_drain) =
            Series::behind(layout, drain, drain_resistance, ("drain", "drain_current"));
        let (source_series, inner_source) = Series::behind(
            layout,
            source,
            source_resistance,
            ("source", "source_current"),
        );
        let stamp = Nodes::new(layout, [gate, inner_drain, inner_source]);
        Ok(JfetShape {
            stamp,
            drain_series,
            source_series,
        })
    }

    fn evaluate<R: Real>(
        shape: &JfetShape,
        prepared: &PreparedJfet<R>,
        memory: &mut JfetMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let [drain_series, source_series] = prepared.series;
        shape.drain_series.add(drain_series, at, stamp);
        shape.source_series.add(source_series, at, stamp);
        let mut bias = Bias::limit_prefix(
            &shape.stamp,
            (prepared.sign, VARIABLES),
            &mut memory.last,
            at,
            stamp,
            |voltage, last| {
                let junction = limit_junction(
                    voltage,
                    last,
                    prepared.thermal_voltage.value(),
                    prepared.critical_voltage,
                );
                limit_gate(
                    junction.unwrap_or(voltage),
                    last,
                    prepared.channel.threshold.value(),
                )
                .or(junction)
            },
        );
        let [vgs, vgd, applied_vds] = bias.voltages;
        let vds = vgs - vgd;
        let [gate_source, gate_drain, _] = bias.beyond;
        bias.voltages = [vgs, vgd, vds];
        bias.beyond = [gate_source, gate_drain, applied_vds - vds];
        let (region, [overdrive, margin]) = prepared.channel.region(vgs, vds);
        let added = Oriented::new(&shape.stamp, prepared.sign, &bias, VARIABLES);
        let [vgs, vgd, vds] = added.across(at);
        memory.variables = [f64::from(region as u8), overdrive, margin];
        // Every derivative among the stamped nodes is summed here, and written once.
        let mut derivatives = Derivatives::new();
        let sums = &mut derivatives;
        // Each gate junction, evaluated at its limited voltage and extended linearly.
        for (position, pair, voltage, depletion) in [
            (VGS, GATE_SOURCE, vgs, prepared.gate_source.as_ref()),
            (VGD, GATE_DRAIN, vgd, prepared.gate_drain.as_ref()),
        ] {
            let voltage = Dual::<R, 1>::variable(voltage);
            let current = junction_current(
                voltage,
                prepared.saturation_current,
                prepared.thermal_voltage,
            ) + voltage * at.gmin;
            let junction = added.reading([position]);
            junction.add_current(stamp, sums, pair, current);
            junction.add_charge(stamp, sums, pair, depletion_charge(depletion, voltage));
        }
        // Within the channel's law, `vgs` comes first and `vds` second.
        let [vgs, vds] = Dual::variables([vgs, vds]);
        let current = prepared.channel.evaluate(vgs, vds);
        added
            .reading(CHANNEL)
            .add_current(stamp, sums, DRAIN_SOURCE, current);
        derivatives.stamp(&shape.stamp, stamp);
    }

    fn variable_names() -> &'static [&'static str] {
        &VARIABLE_NAMES
    }

    fn variables(memory: &JfetMemory, values: &mut [f64]) {
        parts::write_variables(&memory.variables, values);
    }

    fn regions() -> Regions {
        REGIONS
    }

    /// Appends SPICE thermal noise for series resistances and the channel (`8kT·gm / 3`),
    /// plus drain-current flicker noise.
    fn noise<R: Real>(
        shape: &JfetShape,
        prepared: &PreparedJfet<R>,
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
        let nodes = &shape.stamp;
        let voltages = nodes.voltages(at, prepared.sign, [GATE_SOURCE, DRAIN_SOURCE]);
        let [vgs, vds] = Dual::variables(voltages.map(R::constant));
        let current = prepared.channel.evaluate(vgs, vds);
        let [gm, _] = current.gradient;
        let channel = [nodes.equation(INNER_DRAIN), nodes.equation(INNER_SOURCE)];
        let conductance = 2.0 / 3.0 * gm.value();
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

/// Temperature-scaled intrinsic JFET quantities computed during preparation.
struct Scaled<R: Real> {
    thermal_voltage: R,
    channel: DrainCurrent<R>,
    saturation_current: R,
    critical_voltage: f64,
    gate_source: Option<Depletion<R>>,
    gate_drain: Option<Depletion<R>>,
    temperature: f64,
}

impl<R: Real> JfetParameters<R> {
    /// Validates intrinsic parameters and computes temperature-scaled quantities at
    /// `temperature`, in kelvins.
    fn scaled(&self, temperature: R) -> Result<Scaled<R>, ParameterError> {
        let &Self {
            threshold_voltage,
            transconductance_parameter,
            channel_length_modulation,
            gate_source_capacitance,
            gate_drain_capacitance,
            junction_potential,
            saturation_current,
            depletion_coefficient,
            doping_tail,
            threshold_voltage_temperature_coefficient,
            transconductance_parameter_temperature_exponent,
            energy_gap,
            saturation_current_temperature_exponent,
            nominal_temperature,
            ..
        } = self;
        ParameterError::non_negative("transconductance_parameter", transconductance_parameter)?;
        ParameterError::non_negative("gate_source_capacitance", gate_source_capacitance)?;
        ParameterError::non_negative("gate_drain_capacitance", gate_drain_capacitance)?;
        ParameterError::positive("junction_potential", junction_potential)?;
        ParameterError::positive("saturation_current", saturation_current)?;
        crate::kernel::junction::below_one("depletion_coefficient", depletion_coefficient)?;
        ParameterError::non_negative("energy_gap", energy_gap)?;
        // The doping tail's factor, `(1 - b) / (pb - vto)`, which vanishes at `b = 1`.
        let (tail_excess, gap) = (-(doping_tail - 1.0), junction_potential - threshold_voltage);
        if gap.value() == 0.0 && tail_excess.value() != 0.0 {
            return Err(ParameterError(
                "a doping tail other than 1 needs a junction potential other than the \
                 threshold voltage"
                    .to_owned(),
            ));
        }
        let temperatures = Temperatures::of_junction(temperature, nominal_temperature)?;
        let vt = temperatures.thermal_voltage();
        let saturation_current = saturation_current
            * temperatures.saturation(
                energy_gap,
                saturation_current_temperature_exponent,
                R::constant(1.0),
            );
        let depletion = |capacitance| {
            let (potential, capacitance) =
                temperatures.junction(junction_potential, capacitance, R::constant(0.5));
            Depletion::new(
                capacitance,
                potential,
                R::constant(0.5),
                depletion_coefficient,
            )
        };
        Ok(Scaled {
            thermal_voltage: vt,
            channel: DrainCurrent {
                beta: transconductance_parameter
                    * temperatures
                        .ratio()
                        .pow(transconductance_parameter_temperature_exponent),
                threshold: threshold_voltage
                    + threshold_voltage_temperature_coefficient * temperatures.difference(),
                lambda: channel_length_modulation,
                tail: doping_tail,
                tail_factor: if tail_excess.value() == 0.0 {
                    R::constant(0.0)
                } else {
                    tail_excess / gap
                },
            },
            saturation_current,
            critical_voltage: critical_voltage(saturation_current.value(), vt.value()),
            gate_source: depletion(gate_source_capacitance),
            gate_drain: depletion(gate_drain_capacitance),
            temperature: temperatures.actual.value(),
        })
    }
}

impl<R: Real> JfetParameters<R> {
    /// Validates and returns the `[drain_resistance, source_resistance]` pair.
    fn series_resistances(&self) -> Result<[R; 2], ParameterError> {
        Ok([
            ParameterError::non_negative("drain_resistance", self.drain_resistance)?,
            ParameterError::non_negative("source_resistance", self.source_resistance)?,
        ])
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::testing::{
        Bench, assert_close, assert_jacobians_match, assert_regions, source, thermal,
    };

    const CHARGED: &str = r#"{"gate_source_capacitance": 2e-12, "gate_drain_capacitance": 1e-12,
        "channel_length_modulation": 0.02, "doping_tail": 0.8,
        "transconductance_parameter": 1e-3}"#;

    #[test]
    fn follows_the_square_law_in_saturation() {
        let mut bench = Bench::new::<Jfet>(r#"{"transconductance_parameter": 1e-3}"#);
        let evaluation = bench.settle(&[5.0, 0.0, 0.0]);
        // vgs = 0, vto = -2: β · vgst² = 1e-3 · 4.
        let drain_current = evaluation.residual[0];
        assert!((drain_current - 4e-3).abs() < 1e-12, "{drain_current}");
    }

    /// Verifies that [`Jfet`] classifies off, linear, and saturated regions with symmetric
    /// drain/source swap and p-channel polarity inversion.
    ///
    /// Margins `vgs - vto` and `vds - max(vgs - vto, 0)` vanish at `jfetload.c` boundaries.
    #[test]
    fn reports_its_region_and_margins() {
        // The drain, the gate and the source, about the threshold of -2 V.
        assert_regions::<Jfet>(
            "{}",
            Some(r#"{"channel": "p"}"#),
            &[
                (&[5.0, -3.0, 0.0], "off", Some(&[-1.0, 5.0])),
                (&[0.5, 0.0, 0.0], "linear", Some(&[2.0, -1.5])),
                (&[5.0, 0.0, 0.0], "saturated", Some(&[2.0, 3.0])),
                (&[0.0, 0.0, 1.0], "linear", Some(&[2.0, -1.0])),
            ],
            &[
                (&[5.0, -3.0, 0.0], &[5.0, -1.0, 0.0], OVERDRIVE),
                (&[0.5, 0.0, 0.0], &[5.0, 0.0, 0.0], SATURATION_MARGIN),
                (&[0.0, 0.0, 0.5], &[0.0, 0.0, 5.0], SATURATION_MARGIN),
            ],
        );
    }

    #[test]
    fn series_resistances_add_internal_nodes() {
        let bench = Bench::new::<Jfet>(r#"{"drain_resistance": 10, "source_resistance": 5}"#);
        assert_eq!(
            bench.unknowns(),
            ["n0", "n1", "n2", "device:drain", "device:source"]
        );
    }

    proptest! {
        /// Verifies that analytical conductances and capacitances match finite differences
        /// across all operating regions and polarities.
        #[test]
        fn jacobians_match_finite_differences(
            drain in -4.0..4.0f64,
            gate in -3.0..0.6f64,
            p in proptest::bool::ANY,
        ) {
            let parameters = if p {
                CHARGED.replace('{', r#"{"channel": "p", "#)
            } else {
                CHARGED.to_owned()
            };
            let sign = if p { -1.0 } else { 1.0 };
            assert_jacobians_match::<Jfet>(&parameters, &[sign * drain, sign * gate, 0.0]);
            let resistive = parameters.replace(
                '{',
                r#"{"drain_resistance": 50, "source_resistance": 20, "#,
            );
            assert_jacobians_match::<Jfet>(
                &resistive,
                &[sign * drain, sign * gate, 0.0, sign * 0.9 * drain, sign * 0.01],
            );
        }
    }

    #[test]
    fn its_channel_makes_thermal_and_flicker_noise() {
        let parameters = r#"{"threshold_voltage": -2, "transconductance_parameter": 1e-3,
            "saturation_current": 1e-30, "flicker_coefficient": 1e-14}"#;
        let mut bench = Bench::new::<Jfet>(parameters);
        let x = [5.0, 0.0, 1.0];
        let evaluation = bench.settle(&x);
        let (current, gm) = (evaluation.residual[0], evaluation.conductance[0][1]);
        let sources = bench.noise(&x);
        assert_close(
            &[source(&sources, "channel thermal").density(1.0)],
            &[thermal(2.0 / 3.0 * gm)],
        );
        assert_close(
            &[source(&sources, "flicker").density(1.0)],
            &[1e-14 * current],
        );
        assert_eq!(sources.len(), 2);
    }
}
