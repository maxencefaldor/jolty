//! The JFET, derived from ngspice 47's implementation of it in
//! `src/spicelib/devices/jfet/`:
//! Copyright 1985 - 2018, Regents of the University of California and others. Used under
//! ngspice's Modified BSD license, which the repository's `NOTICE` reproduces.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::kernel::dual::Dual;
use crate::kernel::junction::{
    Depletion, Temperatures, critical_voltage, depletion_charge, junction_current, limit_gate,
    limit_junction,
};
use crate::kernel::parts::{self, Derivatives, Nodes, Oriented, Series, series_value};
use crate::kernel::region::{OVERDRIVE, SATURATION_MARGIN};
use crate::noise::Flicker;
use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Lift, Local, NoiseSource, ParameterError,
    REFERENCE_TEMPERATURE, REGION, Real, Regions, Stamp,
};

/// The channel of a field-effect transistor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Channel {
    /// An n-channel, conducting with its gate above its threshold.
    N,
    /// A p-channel, the n-channel's mirror: every voltage and current reversed.
    P,
}

impl Channel {
    /// `1` for an n-channel, `−1` for a p-channel: the factor that orients voltages and
    /// currents as an n-channel's.
    #[must_use]
    pub const fn sign(self) -> f64 {
        match self {
            Self::N => 1.0,
            Self::P => -1.0,
        }
    }
}

/// A junction field-effect transistor, in the SPICE model with Sydney University's doping
/// tail, as ngspice implements it.
///
/// Two junctions join the gate to the channel, one at each end, each carrying a
/// junction current and holding a depletion charge of grading ½. The
/// drain current follows the square law, softened near the threshold by the doping tail
/// `b`: with `vgst = vgs − vto` and `β' = β · (1 + λ · vds)`, it is
/// `β' · vgst² · (b + (1 − b) · vgst / (pb − vto))` in saturation. The transistor is
/// symmetric: with the drain below the source, the two swap. A drain or source resistance
/// adds an internal node, labelled `drain` or `source`; one below
/// [`SMALL_RESISTANCE`](crate::SMALL_RESISTANCE) also carries its current as an unknown,
/// labelled `drain_current` or `source_current`.
///
/// Away from its nominal temperature, its gate junctions' saturation current scales with
/// the energy gap and their potential and capacitances with silicon's band gap; its
/// threshold moves, and its transconductance parameter scales, by their own coefficients,
/// as ngspice scales them.
#[derive(Debug, Clone, Copy)]
pub struct Jfet;

/// Where a JFET's contributions go among its local unknowns: through its drain and source
/// resistances, and among its gate and the drain and source behind them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JfetShape {
    stamp: Nodes<STAMPED>,
    drain_series: Series,
    source_series: Series,
}

/// A JFET's numbers at its temperature.
#[derive(Debug, Clone, Copy)]
pub struct PreparedJfet<R: Real> {
    sign: f64,
    /// The drain and the source resistances' conductances, or each resistance where its
    /// current is an unknown.
    series: [R; 2],
    /// The thermal voltage at the transistor's temperature.
    thermal_voltage: R,
    channel: DrainCurrent<R>,
    saturation_current: R,
    critical_voltage: f64,
    gate_source: Option<Depletion<R>>,
    gate_drain: Option<Depletion<R>>,
    temperature: f64,
    flicker: Flicker,
}

/// What a JFET remembers.
#[derive(Debug, Clone, Copy, Default)]
pub struct JfetMemory {
    /// The gate-source and gate-drain voltages of the last evaluation, from which the next
    /// steps are limited.
    last: [f64; JUNCTIONS],
    /// The values of [`VARIABLE_NAMES`] at the last evaluation.
    variables: [f64; 3],
}

/// The variables a JFET reports: its region among [`REGIONS`] and its margins.
const VARIABLE_NAMES: [&str; 3] = [REGION, OVERDRIVE, SATURATION_MARGIN];

/// The regions of a JFET's channel, as `jfetload.c` switches between them, in the order of
/// [`REGIONS`]' names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Region {
    /// At or below its threshold, conducting nothing.
    Off,
    /// Above its threshold, with the drain below saturation.
    Linear,
    /// Above its threshold, with the drain past saturation.
    Saturated,
}

/// The regions a JFET's equations distinguish, and its margins to their boundaries:
/// `vgs − vto` and `vds − vdsat`, `vdsat` being `max(vgs − vto, 0)`, in an n-channel's sign
/// as [`Regions::margins`] are, with the drain and the source swapped where the drain lies
/// below the source, as the equations swap them.
const REGIONS: Regions = Regions {
    names: &["off", "linear", "saturated"],
    margins: &[OVERDRIVE, SATURATION_MARGIN],
};

/// Local node indices: the gate, then the drain and source behind their resistances, which
/// are the terminals themselves without one.
const GATE: usize = 0;
const INNER_DRAIN: usize = 1;
const INNER_SOURCE: usize = 2;

/// The number of stamped nodes.
const STAMPED: usize = 3;

/// The number of voltages the transistor's equations take as variables.
const VARIABLE_COUNT: usize = 3;

/// The number of those voltages, the first, across the gate junctions, whose steps are
/// limited.
const JUNCTIONS: usize = 2;

/// The pairs of stamped nodes across which the variables lie: the gate-source and the
/// gate-drain junctions, and the channel, from the drain to the source.
const VARIABLES: [(usize, usize); VARIABLE_COUNT] = [GATE_SOURCE, GATE_DRAIN, DRAIN_SOURCE];
const GATE_SOURCE: (usize, usize) = (GATE, INNER_SOURCE);
const GATE_DRAIN: (usize, usize) = (GATE, INNER_DRAIN);
const DRAIN_SOURCE: (usize, usize) = (INNER_DRAIN, INNER_SOURCE);

/// The variables' positions among [`VARIABLES`], by SPICE's names.
const VGS: usize = 0;
const VGD: usize = 1;
const VDS: usize = 2;

/// The variables the channel's law reads, by their positions among [`VARIABLES`], in the
/// order its numbers carry their derivatives: the gate-source voltage, then the
/// drain-source voltage.
const CHANNEL: [usize; 2] = [VGS, VDS];

/// The variables of one evaluation.
type Bias = parts::Bias<VARIABLE_COUNT>;

/// The parameters of a [`Jfet`], with SPICE's defaults.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift)]
#[serde(default, deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct JfetParameters<R: Real = f64> {
    /// The channel; default `n`.
    pub channel: Channel,
    /// The threshold voltage, `vto`, in volts; default −2.
    pub threshold_voltage: R,
    /// The transconductance parameter, `β`, in amperes per square volt; default 1e-4.
    pub transconductance_parameter: R,
    /// The channel-length modulation, `λ`, per volt; default 0.
    pub channel_length_modulation: R,
    /// The drain resistance, in ohms; default 0.
    pub drain_resistance: R,
    /// The source resistance, in ohms; default 0.
    pub source_resistance: R,
    /// The zero-bias gate-source capacitance, in farads; default 0.
    pub gate_source_capacitance: R,
    /// The zero-bias gate-drain capacitance, in farads; default 0.
    pub gate_drain_capacitance: R,
    /// The gate junctions' potential, `pb`, in volts; default 1.
    pub junction_potential: R,
    /// The gate junctions' saturation current, in amperes; default 1e-14.
    pub saturation_current: R,
    /// The fraction of the junction potential above which the depletion capacitance is
    /// extended linearly, below 1; default 0.5.
    pub depletion_coefficient: R,
    /// The doping tail, `b`: 1 for the plain square law; default 1.
    pub doping_tail: R,
    /// How fast the threshold voltage rises with temperature, in volts per kelvin;
    /// default 0.
    pub threshold_voltage_temperature_coefficient: R,
    /// The exponent of the temperature ratio that scales the transconductance parameter;
    /// default 0.
    pub transconductance_parameter_temperature_exponent: R,
    /// The energy gap that scales the saturation current with temperature, in volts;
    /// default 1.11, silicon's.
    pub energy_gap: R,
    /// The exponent of the temperature ratio in the saturation current's scaling; default
    /// 0, as ngspice scales a JFET's by it only when it is given.
    pub saturation_current_temperature_exponent: R,
    /// The temperature the parameters were measured at, in kelvins; default 27 °C.
    pub nominal_temperature: R,
    /// The flicker noise coefficient of the drain current, `KF`; default 0, no flicker
    /// noise.
    pub flicker_coefficient: R,
    /// The power of the drain current in the flicker noise, `AF`; default 1.
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

/// The drain current of the channel, as a function of the gate-source and drain-source
/// voltages.
#[derive(Debug, Clone, Copy)]
struct DrainCurrent<R: Real> {
    beta: R,
    threshold: R,
    lambda: R,
    tail: R,
    /// `(1 − b) / (pb − vto)`.
    tail_factor: R,
}

impl<R: Real> DrainCurrent<R> {
    /// The channel's region at `vgs` and `vds`, and its margins `vgs − vto` and
    /// `vds − vdsat`, with the drain at or above the source, the two swapped otherwise.
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

    /// The current from drain to source, a number of the channel's law, at `vgs` and
    /// `vds`, in the channel's region there.
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
        let bias = prepared.bias(shape, memory, at, stamp);
        let [vgs, _, vds] = bias.voltages;
        let (region, [overdrive, margin]) = prepared.channel.region(vgs, vds);
        let [vgs, vgd, vds] = bias.across(&shape.stamp, at, (prepared.sign, VARIABLES));
        memory.variables = [f64::from(region as u8), overdrive, margin];
        let added = Oriented {
            nodes: &shape.stamp,
            sign: prepared.sign,
            bias: &bias,
            variables: VARIABLES,
        };
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

    /// The thermal noise of the series resistances and of the channel, `8kT·gm / 3`, and
    /// the drain current's flicker noise, as SPICE models them.
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
        let voltages = [GATE_SOURCE, DRAIN_SOURCE].map(|(plus, minus)| {
            prepared.sign * (nodes.voltage(at, plus) - nodes.voltage(at, minus))
        });
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

impl<R: Real> PreparedJfet<R> {
    /// The variables at `at`: the gate junctions' voltages, each step limited as a
    /// junction's and then about the threshold from the last evaluation's, which `memory`
    /// keeps and takes the new ones, and the drain-source voltage between them.
    fn bias(
        &self,
        shape: &JfetShape,
        memory: &mut JfetMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) -> Bias {
        let nodes = &shape.stamp;
        let applied = VARIABLES
            .map(|(plus, minus)| self.sign * (nodes.voltage(at, plus) - nodes.voltage(at, minus)));
        let mut voltages = applied;
        let mut limited = false;
        for (voltage, last) in voltages.iter_mut().zip(&mut memory.last) {
            if at.limiting {
                let junction = limit_junction(
                    *voltage,
                    *last,
                    self.thermal_voltage.value(),
                    self.critical_voltage,
                );
                let gate = limit_gate(
                    junction.unwrap_or(*voltage),
                    *last,
                    self.channel.threshold.value(),
                );
                if let Some(step) = gate.or(junction) {
                    *voltage = step;
                    limited = true;
                }
            }
            *last = *voltage;
        }
        if limited {
            stamp.mark_limited();
        }
        let [vgs, vgd, _] = voltages;
        Bias::new(applied, [vgs, vgd, vgs - vgd])
    }
}

/// What a JFET's temperature sets.
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
    /// What the transistor's `temperature`, in kelvins, sets, once each parameter it
    /// reads is checked.
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
        // The doping tail's factor, `(1 − b) / (pb − vto)`, which vanishes at `b = 1`.
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
    /// The drain and source resistances, once each is checked.
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
    use crate::Differentiable;
    use crate::testing::{
        Bench, assert_close, assert_jacobians_match, assert_mirrors,
        assert_parameter_gradients_match, assert_switches_where_margin_vanishes, source, thermal,
    };

    const CHARGED: &str = r#"{"gate_source_capacitance": 2e-12, "gate_drain_capacitance": 1e-12,
        "channel_length_modulation": 0.02, "doping_tail": 0.8, "transconductance_parameter": 1e-3}"#;

    #[test]
    fn parameter_gradients_are_exact() {
        let every = r#"{"threshold_voltage": -1.8, "transconductance_parameter": 1e-3,
            "channel_length_modulation": 0.02, "drain_resistance": 10,
            "source_resistance": 5, "gate_source_capacitance": 2e-12,
            "gate_drain_capacitance": 1e-12, "junction_potential": 0.9,
            "saturation_current": 2e-14, "depletion_coefficient": 0.4, "doping_tail": 0.8,
            "threshold_voltage_temperature_coefficient": -2e-3, "transconductance_parameter_temperature_exponent": -1.5,
            "energy_gap": 1.2, "saturation_current_temperature_exponent": 2,
            "nominal_temperature": 290}"#;
        for [drain, gate] in [[3.0, 0.0], [0.5, -0.5], [-0.5, -0.2], [-3.0, 0.1]] {
            assert_parameter_gradients_match::<Jfet>(
                every,
                &[drain, gate, 0.0, drain - 0.01, 0.005],
            );
        }
    }

    #[test]
    fn follows_the_square_law_in_saturation() {
        let mut bench = Bench::new::<Jfet>(r#"{"transconductance_parameter": 1e-3}"#);
        let evaluation = bench.settle(&[5.0, 0.0, 0.0]);
        // vgs = 0, vto = −2: β · vgst² = 1e-3 · 4.
        let drain_current = evaluation.residual[0];
        assert!((drain_current - 4e-3).abs() < 1e-12, "{drain_current}");
    }

    /// A JFET is off at or below its threshold, linear with its drain within `vgs − vto`,
    /// and saturated past it, the drain and the source swapping where the drain lies below;
    /// its margins are `vgs − vto` and `vds − max(vgs − vto, 0)`, which vanish where
    /// `jfetload.c` switches, a p-channel's equal to an n-channel's at the voltages
    /// reversed.
    #[test]
    fn reports_its_region_and_margins() {
        let mut n = Bench::new::<Jfet>("{}");
        let mut p = Bench::new::<Jfet>(r#"{"channel": "p"}"#);
        // The drain, the gate and the source, about the threshold of -2 V.
        for (x, region, margins) in [
            ([5.0, -3.0, 0.0], "off", [-1.0, 5.0]),
            ([0.5, 0.0, 0.0], "linear", [2.0, -1.5]),
            ([5.0, 0.0, 0.0], "saturated", [2.0, 3.0]),
            ([0.0, 0.0, 1.0], "linear", [2.0, -1.0]),
        ] {
            n.settle(&x);
            assert_eq!(n.region(), (region, margins.to_vec()), "{x:?}");
            p.settle(&x.map(|value| -value));
            assert_eq!(p.region(), (region, margins.to_vec()), "{x:?}");
        }
        let switches = [
            ([5.0, -3.0, 0.0], [5.0, -1.0, 0.0], OVERDRIVE),
            ([0.5, 0.0, 0.0], [5.0, 0.0, 0.0], SATURATION_MARGIN),
            ([0.0, 0.0, 0.5], [0.0, 0.0, 5.0], SATURATION_MARGIN),
        ];
        for (from, to, margin) in switches {
            assert_switches_where_margin_vanishes(&mut n, (&from, &to), margin);
        }
    }

    #[test]
    fn a_p_channel_mirrors_an_n_channel() {
        assert_mirrors(&Differentiable::<Jfet>::new(), "{}", &[3.0, -0.5, 0.0]);
    }

    #[test]
    fn series_resistances_add_internal_nodes() {
        let bench = Bench::new::<Jfet>(r#"{"drain_resistance": 10, "source_resistance": 5}"#);
        assert_eq!(
            bench.unknowns(),
            ["n0", "n1", "n2", "device:drain", "device:source"]
        );
    }

    #[test]
    fn a_heated_jfet_is_the_jfet_at_its_temperature() {
        let model = crate::Thermal::<Jfet>::new();
        let every = r#"{"transconductance_parameter": 1e-3, "drain_resistance": 10,
            "gate_source_capacitance": 2e-12, "threshold_voltage_temperature_coefficient": -2e-3,
            "transconductance_parameter_temperature_exponent": -1.5, "saturation_current_temperature_exponent": 2}"#;
        let heated = every.replace('{', r#"{"thermal": true, "#);
        let mut bench = Bench::from_model(&model, &heated, REFERENCE_TEMPERATURE).expect("valid");
        for rise in [0.0, 30.0] {
            let mut plain = Bench::at::<Jfet>(every, REFERENCE_TEMPERATURE + rise);
            let hot = bench.settle(&[5.0, -0.5, 0.0, rise, 4.9]);
            let reference = plain.settle(&[5.0, -0.5, 0.0, 4.9]);
            let electrical = [0, 1, 2, 4].map(|row| hot.residual[row]);
            assert_close(&electrical, &reference.residual);
        }
        for x in [[5.0, -0.5, 0.0, 20.0, 4.9], [0.3, -1.0, 0.0, 55.0, 0.29]] {
            crate::testing::assert_model_jacobians_match(&model, &heated, &x);
            crate::testing::assert_model_parameter_gradients_match(&model, &heated, &x, None);
        }
    }

    proptest! {
        /// The conductances and capacitances are the derivatives of the currents and
        /// charges, in every region and in both modes.
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
            let resistive = parameters.replace('{', r#"{"drain_resistance": 50, "source_resistance": 20, "#);
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
