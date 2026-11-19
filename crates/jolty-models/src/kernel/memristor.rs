//! Memristor models with a state-dependent resistance between `positive` and `negative`,
//! varying from on-resistance at state 0 to off-resistance at state 1 by [law](MemristorLaw).
//!
//! State is tracked as an internal unknown labelled `state` that evolves only in time.
//! Operating-point and linearized analyses hold `state` at the initial value; transient
//! integration evolves `state` under the control drive, damped near both bounds by a smooth
//! window.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::dimension::{CHARGE, FREQUENCY, RESISTANCE, VOLTAGE};
use crate::kernel::dual::Dual;
use crate::kernel::parts;
use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Lift, Local, Nature, NoiseSource,
    ParameterError, ParameterKind, ParameterMetadata, Real, Stamp, UnknownIndex,
};

/// Mapping from memristor state `x` in `[0, 1]` to resistance between on-resistance and
/// off-resistance, following VTEAM conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum MemristorLaw {
    /// Linear series-boundary resistance `r_on + (r_off − r_on) · x`.
    #[default]
    Linear,
    /// Exponential tunnel-barrier resistance `r_on · (r_off / r_on)^x`.
    Exponential,
}

/// Memristor resistance law and boundary-window damping parameters.
#[derive(Debug, Clone, Copy)]
struct Memristance<R: Real> {
    law: MemristorLaw,
    on_resistance: R,
    off_resistance: R,
    /// Natural logarithm of the resistance ratio, `ln(r_off / r_on)`.
    log_ratio: R,
    window: R,
}

impl<R: Real> Memristance<R> {
    fn new(
        law: MemristorLaw,
        [on_resistance, off_resistance]: [R; 2],
        window: R,
    ) -> Result<Self, ParameterError> {
        ParameterError::positive("on_resistance", on_resistance)?;
        ParameterError::positive("off_resistance", off_resistance)?;
        if off_resistance.value() <= on_resistance.value() {
            let (on, off) = (on_resistance.value(), off_resistance.value());
            return Err(ParameterError(format!(
                "off_resistance exceeds on_resistance, not {off:?} against {on:?}"
            )));
        }
        Ok(Self {
            law,
            on_resistance,
            off_resistance,
            log_ratio: (off_resistance / on_resistance).ln(),
            window: ParameterError::positive("window", window)?,
        })
    }

    /// Evaluates conductance at `state`, clamping `state` within `[0, 1]`.
    fn conductance<const N: usize>(&self, state: Dual<R, N>) -> Dual<R, N> {
        let held = if (0.0..=1.0).contains(&state.value.value()) {
            state
        } else {
            Dual::number(state.value.value().clamp(0.0, 1.0))
        };
        match self.law {
            MemristorLaw::Linear => {
                let span = self.off_resistance - self.on_resistance;
                held.scale(span).plus(self.on_resistance).recip()
            }
            MemristorLaw::Exponential => (-held.scale(self.log_ratio))
                .exp()
                .divide(self.on_resistance),
        }
    }

    /// Computes windowed state rate of change (positive toward off) from `state` and raw
    /// `drive`, slowing motion toward either bound without impeding departure.
    ///
    /// Let `d` be the distance to the target bound in window widths. Inside the bound, the
    /// window multiplies `drive` by `1 − e^(−d)` so motion halts at the boundary; beyond the
    /// bound, multiplying by `d` reverses motion with a matching slope.
    fn rate(&self, state: Dual<R, 2>, drive: Dual<R, 2>) -> Dual<R, 2> {
        let toward_off = drive.value.value() > 0.0;
        // Window factor depends on state alone.
        let freedom = state.apply(|state| {
            let distance = if toward_off {
                (1.0 - state).divide(self.window)
            } else {
                state.divide(self.window)
            };
            if distance.value.value() >= 0.0 {
                1.0 - (-distance).exp()
            } else {
                distance
            }
        });
        drive * freedom
    }
}

/// Control mechanism driving memristor state evolution.
#[derive(Debug, Clone, Copy)]
enum Control<R: Real> {
    /// VTEAM voltage thresholds: drives state toward off above `off_voltage` at
    /// `off_rate · (v/v_off − 1)^α_off` and toward on below `on_voltage` at
    /// `on_rate · (v/v_on − 1)^α_on`, holding constant in between.
    Voltage {
        on_voltage: R,
        off_voltage: R,
        on_rate: R,
        off_rate: R,
        on_exponent: R,
        off_exponent: R,
    },
    /// Linear ion drift: evolves state in proportion to transferred charge divided by
    /// full-range `charge`.
    Current { charge: R },
}

impl<R: Real> Control<R> {
    /// Computes the unwindowed state rate (positive toward off) at `voltage` and
    /// `conductance`, both differentiated with respect to voltage and state.
    fn drive(&self, voltage: Dual<R, 2>, conductance: Dual<R, 2>) -> Dual<R, 2> {
        match self {
            Self::Voltage {
                on_voltage,
                off_voltage,
                on_rate,
                off_rate,
                on_exponent,
                off_exponent,
            } => {
                let across = voltage.value.value();
                let (threshold, rate, exponent, sign) = if across > off_voltage.value() {
                    (*off_voltage, *off_rate, *off_exponent, 1.0)
                } else if across < on_voltage.value() {
                    (*on_voltage, *on_rate, *on_exponent, -1.0)
                } else {
                    return Dual::number(0.0);
                };
                let overdrive = voltage.divide(threshold) - 1.0;
                overdrive.pow(exponent).scale(rate) * sign
            }
            Self::Current { charge } => (conductance * voltage).divide(*charge),
        }
    }
}

/// Layout shape for a memristor storing its internal state unknown and matrix entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemristorShape {
    state: usize,
    /// Matrix entries joining `positive` and `negative` to both terminals and `state`.
    terminals: [[usize; 3]; 2],
    /// Matrix entries joining `state` to both terminals and itself.
    own: [usize; 3],
}

/// Prepared parameters for a memristor instance.
#[derive(Debug, Clone, Copy)]
pub struct PreparedMemristor<R: Real> {
    memristance: Memristance<R>,
    control: Control<R>,
    /// Initial state held at operating-point analyses.
    initial: R,
    temperature: f64,
}

/// Memristor evaluation memory storing the most recent resistance and state rate (NaN
/// before the first evaluation).
#[derive(Debug, Clone, Copy)]
pub struct MemristorMemory([f64; 2]);

impl Default for MemristorMemory {
    fn default() -> Self {
        Self([f64::NAN; 2])
    }
}

/// Local terminal indices for `positive` and `negative`.
const POSITIVE_NEGATIVE: [usize; 2] = [0, 1];

/// Terminal names of a memristor.
const TERMINALS: &[&str] = &["positive", "negative"];

/// Builds the two-terminal conductive [`Interface`] shared by memristor kernels.
fn interface() -> Interface {
    Interface::new(TERMINALS, &[Branch::new(0, 1, BranchKind::Conductive)])
}

impl<R: Real> PreparedMemristor<R> {
    /// Validates initial `state` and prepares a memristor at `temperature`.
    fn new(
        (memristance, control): (Memristance<R>, Control<R>),
        state: R,
        temperature: R,
    ) -> Result<Self, ParameterError> {
        let initial = ParameterError::non_negative("state", state)?;
        if initial.value() > 1.0 {
            let state = initial.value();
            return Err(ParameterError(format!(
                "state is at most 1, not {state:?}"
            )));
        }
        Ok(Self {
            memristance,
            control,
            initial,
            temperature: temperature.value(),
        })
    }
}

impl MemristorShape {
    /// Claims a memristor's state and entries in `layout`.
    fn new(layout: &mut Layout) -> Self {
        let state = layout.internal("state", Nature::State);
        let [positive, negative] = POSITIVE_NEGATIVE;
        let columns = [positive, negative, state];
        Self {
            state,
            terminals: [positive, negative]
                .map(|row| columns.map(|column| layout.entry(row, column))),
            own: columns.map(|column| layout.entry(state, column)),
        }
    }

    fn evaluate<R: Real>(
        &self,
        prepared: &PreparedMemristor<R>,
        memory: &mut MemristorMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let [positive, negative] = POSITIVE_NEGATIVE;
        let [voltage, state] = Dual::variables([
            at.unknown::<R>(positive) - at.unknown::<R>(negative),
            at.unknown(self.state),
        ]);
        // The conductance is a law of the state alone.
        let conductance = state.apply(|state| prepared.memristance.conductance(state));
        let current = conductance * voltage;
        stamp.add_residual(positive, current.value);
        stamp.add_residual(negative, -current.value);
        // With respect to positive, negative and the state.
        let [by_voltage, by_state] = current.gradient;
        let derivatives = [by_voltage, -by_voltage, by_state];
        for (row, sign) in self.terminals.iter().zip([1.0, -1.0]) {
            for (&entry, derivative) in row.iter().zip(derivatives) {
                stamp.add_conductance(entry, derivative * sign);
            }
        }
        let rate = prepared
            .memristance
            .rate(state, prepared.control.drive(voltage, conductance));
        memory.0 = [conductance.value.value().recip(), rate.value.value()];
        let row = self.state;
        let [on_positive, on_negative, own] = self.own;
        if at.transient {
            // dx/dt − rate = 0.
            let [by_voltage, by_state] = rate.gradient;
            stamp.add_residual(row, -rate.value);
            stamp.add_charge(row, state.value);
            stamp.add_conductance(on_positive, -by_voltage);
            stamp.add_conductance(on_negative, by_voltage);
            stamp.add_conductance(own, -by_state);
            stamp.add_capacitance(own, R::constant(1.0));
        } else {
            // x − x₀ = 0.
            stamp.add_residual(row, -prepared.initial + state.value);
            stamp.add_conductance(own, R::constant(1.0));
        }
    }

    /// Appends thermal noise for the conductance determined by the current memristor state.
    fn noise<R: Real>(
        &self,
        prepared: &PreparedMemristor<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        let state = Dual::<R, 0>::number(at.value(self.state));
        let conductance = prepared.memristance.conductance(state).value;
        let [positive, negative] = POSITIVE_NEGATIVE;
        sources.push(NoiseSource::thermal(
            "thermal",
            [
                Some(UnknownIndex::new(positive)),
                Some(UnknownIndex::new(negative)),
            ],
            conductance.value(),
            prepared.temperature,
        ));
    }
}

/// Diagnostic variable names for a memristor: instantaneous resistance and state rate (in
/// full ranges per second, positive toward off), reported even during DC operating-point
/// analyses.
const VARIABLE_NAMES: [&str; 2] = ["resistance", "rate"];

fn one<R: Real>() -> R {
    R::constant(1.0)
}

fn a_hundredth<R: Real>() -> R {
    R::constant(0.01)
}

/// Voltage-controlled memristor following the VTEAM model (Kvatinsky, Ramadan, Friedman,
/// and Kolodny, 2015).
///
/// Terminal voltage `v = v(positive) − v(negative)` drives the state toward off above
/// `off_voltage` at `off_rate · (v/v_off − 1)^α_off` and toward on below `on_voltage` at
/// `−on_rate · (v/v_on − 1)^α_on`. Between thresholds the state holds constant. Each rate
/// equals the speed at twice the corresponding threshold in full ranges per second, damped
/// near either bound by the window.
#[derive(Debug, Clone, Copy)]
pub struct VoltageControlledMemristor;

/// Parameters for [`VoltageControlledMemristor`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift, ParameterMetadata)]
#[serde(deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct VoltageControlledMemristorParameters<R: Real = f64> {
    /// Fully on resistance at state 0, in ohms; positive.
    #[dimension(RESISTANCE)]
    pub on_resistance: R,
    /// Fully off resistance at state 1, in ohms; greater than `on_resistance`.
    #[dimension(RESISTANCE)]
    pub off_resistance: R,
    /// Voltage threshold below which state moves toward on, in volts; negative.
    #[dimension(VOLTAGE)]
    pub on_voltage: R,
    /// Voltage threshold above which state moves toward off, in volts; positive.
    #[dimension(VOLTAGE)]
    pub off_voltage: R,
    /// State speed toward on at twice `on_voltage`, in full ranges per second; positive.
    #[dimension(FREQUENCY)]
    pub on_rate: R,
    /// State speed toward off at twice `off_voltage`, in full ranges per second; positive.
    #[dimension(FREQUENCY)]
    pub off_rate: R,
    /// Nonlinearity exponent `α_on` for motion past `on_voltage`; at least 1; default 1.
    #[serde(default = "one::<R>")]
    pub on_exponent: R,
    /// Nonlinearity exponent `α_off` for motion past `off_voltage`; at least 1; default 1.
    #[serde(default = "one::<R>")]
    pub off_exponent: R,
    /// Resistance-versus-state law; default linear.
    #[serde(default)]
    pub law: MemristorLaw,
    /// Initial state from 0 (fully on) to 1 (fully off); default 1.
    #[serde(default = "one::<R>")]
    pub state: R,
    /// Boundary window width as a fraction of the state range; positive; default 0.01.
    #[serde(default = "a_hundredth::<R>")]
    pub window: R,
}

impl Kernel for VoltageControlledMemristor {
    const NAME: &'static str = "voltage_controlled_memristor";
    type Parameters<R: Real> = VoltageControlledMemristorParameters<R>;
    type Shape = MemristorShape;
    type Prepared<R: Real> = PreparedMemristor<R>;
    type Memory = MemristorMemory;

    fn parameter_kinds() -> &'static [(&'static str, ParameterKind)] {
        VoltageControlledMemristorParameters::<f64>::PARAMETER_KINDS
    }

    fn interface<R: Real>(_: &VoltageControlledMemristorParameters<R>) -> Interface {
        interface()
    }

    fn prepare<R: Real>(
        parameters: &VoltageControlledMemristorParameters<R>,
        temperature: R,
    ) -> Result<PreparedMemristor<R>, ParameterError> {
        let &VoltageControlledMemristorParameters {
            on_resistance,
            off_resistance,
            on_voltage,
            off_voltage,
            on_rate,
            off_rate,
            on_exponent,
            off_exponent,
            law,
            state,
            window,
        } = parameters;
        if on_voltage.value() >= 0.0 {
            let on = on_voltage.value();
            return Err(ParameterError(format!(
                "on_voltage is negative, not {on:?}"
            )));
        }
        let control = Control::Voltage {
            on_voltage,
            off_voltage: ParameterError::positive("off_voltage", off_voltage)?,
            on_rate: ParameterError::positive("on_rate", on_rate)?,
            off_rate: ParameterError::positive("off_rate", off_rate)?,
            on_exponent: at_least_one("on_exponent", on_exponent)?,
            off_exponent: at_least_one("off_exponent", off_exponent)?,
        };
        let memristance = Memristance::new(law, [on_resistance, off_resistance], window)?;
        PreparedMemristor::new((memristance, control), state, temperature)
    }

    fn layout<R: Real>(
        _: &VoltageControlledMemristorParameters<R>,
        _: &PreparedMemristor<R>,
        layout: &mut Layout,
    ) -> Result<MemristorShape, ParameterError> {
        Ok(MemristorShape::new(layout))
    }

    fn evaluate<R: Real>(
        shape: &MemristorShape,
        prepared: &PreparedMemristor<R>,
        memory: &mut MemristorMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        shape.evaluate(prepared, memory, at, stamp);
    }

    fn variable_names() -> &'static [&'static str] {
        &VARIABLE_NAMES
    }

    fn variables(memory: &MemristorMemory, values: &mut [f64]) {
        parts::write_variables(&memory.0, values);
    }

    fn noise<R: Real>(
        shape: &MemristorShape,
        prepared: &PreparedMemristor<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        shape.noise(prepared, at, sources);
    }
}

/// Checks that the exponent `value` is at least 1, so the state's speed rises from zero
/// at its threshold with a finite slope.
fn at_least_one<R: Real>(parameter: &str, value: R) -> Result<R, ParameterError> {
    if value.value() >= 1.0 {
        Ok(value)
    } else {
        let value = value.value();
        Err(ParameterError(format!(
            "{parameter} is at least 1, not {value:?}"
        )))
    }
}

/// Current-controlled memristor following the HP titanium dioxide linear ion drift model
/// (Strukov, Snider, Stewart, and Williams, 2008).
///
/// Charge `q` flowing from `positive` to `negative` shifts the state toward off by `q/Q`,
/// where `Q` is the full-range switching charge. The boundary window damps state motion at
/// either limit.
#[derive(Debug, Clone, Copy)]
pub struct CurrentControlledMemristor;

/// Parameters for [`CurrentControlledMemristor`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift, ParameterMetadata)]
#[serde(deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct CurrentControlledMemristorParameters<R: Real = f64> {
    /// Fully on resistance at state 0, in ohms; positive.
    #[dimension(RESISTANCE)]
    pub on_resistance: R,
    /// Fully off resistance at state 1, in ohms; greater than `on_resistance`.
    #[dimension(RESISTANCE)]
    pub off_resistance: R,
    /// Switching charge carrying the state across its full range, in coulombs; positive.
    #[dimension(CHARGE)]
    pub charge: R,
    /// Resistance-versus-state law; default linear.
    #[serde(default)]
    pub law: MemristorLaw,
    /// Initial state from 0 (fully on) to 1 (fully off); default 1.
    #[serde(default = "one::<R>")]
    pub state: R,
    /// Boundary window width as a fraction of the state range; positive; default 0.01.
    #[serde(default = "a_hundredth::<R>")]
    pub window: R,
}

impl Kernel for CurrentControlledMemristor {
    const NAME: &'static str = "current_controlled_memristor";
    type Parameters<R: Real> = CurrentControlledMemristorParameters<R>;
    type Shape = MemristorShape;
    type Prepared<R: Real> = PreparedMemristor<R>;
    type Memory = MemristorMemory;

    fn parameter_kinds() -> &'static [(&'static str, ParameterKind)] {
        CurrentControlledMemristorParameters::<f64>::PARAMETER_KINDS
    }

    fn interface<R: Real>(_: &CurrentControlledMemristorParameters<R>) -> Interface {
        interface()
    }

    fn prepare<R: Real>(
        parameters: &CurrentControlledMemristorParameters<R>,
        temperature: R,
    ) -> Result<PreparedMemristor<R>, ParameterError> {
        let &CurrentControlledMemristorParameters {
            on_resistance,
            off_resistance,
            charge,
            law,
            state,
            window,
        } = parameters;
        let control = Control::Current {
            charge: ParameterError::positive("charge", charge)?,
        };
        let memristance = Memristance::new(law, [on_resistance, off_resistance], window)?;
        PreparedMemristor::new((memristance, control), state, temperature)
    }

    fn layout<R: Real>(
        _: &CurrentControlledMemristorParameters<R>,
        _: &PreparedMemristor<R>,
        layout: &mut Layout,
    ) -> Result<MemristorShape, ParameterError> {
        Ok(MemristorShape::new(layout))
    }

    fn evaluate<R: Real>(
        shape: &MemristorShape,
        prepared: &PreparedMemristor<R>,
        memory: &mut MemristorMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        shape.evaluate(prepared, memory, at, stamp);
    }

    fn variable_names() -> &'static [&'static str] {
        &VARIABLE_NAMES
    }

    fn variables(memory: &MemristorMemory, values: &mut [f64]) {
        parts::write_variables(&memory.0, values);
    }

    fn noise<R: Real>(
        shape: &MemristorShape,
        prepared: &PreparedMemristor<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        shape.noise(prepared, at, sources);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{
        Bench, assert_close, assert_jacobians_match, assert_parameter_gradients_match,
        assert_parameter_gradients_match_at, assert_stepping_jacobians_match, source, thermal,
    };

    const VTEAM: &str = r#"{"on_resistance": 1e3, "off_resistance": 1e5,
        "on_voltage": -1.0, "off_voltage": 0.5, "on_rate": 2e6, "off_rate": 1e5,
        "on_exponent": 3, "off_exponent": 2, "state": 0.4}"#;
    const DRIFT: &str = r#"{"on_resistance": 100, "off_resistance": 1.6e4,
        "charge": 1e-4, "state": 0.4}"#;

    /// Appends JSON object members `extra` to the JSON object `parameters`.
    fn with(parameters: &str, extra: &str) -> String {
        format!("{}, {extra}}}", parameters.trim_end().trim_end_matches('}'))
    }

    /// Evaluates the transient state rate at `voltage` and `state`.
    fn rate(bench: &mut Bench, voltage: f64, state: f64) -> f64 {
        -bench.evaluate(&[voltage, 0.0, state]).residual[2]
    }

    #[test]
    fn its_resistance_follows_its_state_by_its_law() {
        let mut linear = Bench::new::<VoltageControlledMemristor>(VTEAM);
        let mut exponential =
            Bench::new::<VoltageControlledMemristor>(&with(VTEAM, r#""law": "exponential""#));
        for state in [0.0, 0.25, 1.0] {
            let current = linear.evaluate(&[0.1, 0.0, state]).residual[0];
            assert_close(&[current], &[0.1 / (1e3 + 99e3 * state)]);
            let current = exponential.evaluate(&[0.1, 0.0, state]).residual[0];
            assert_close(&[current], &[0.1 / (1e3 * 100f64.powf(state))]);
        }
        // Past either bound, the resistance holds.
        let current = linear.evaluate(&[0.1, 0.0, -0.01]).residual[0];
        assert_close(&[current], &[0.1 / 1e3]);
    }

    #[test]
    fn at_an_operating_point_its_state_is_the_one_it_was_made_in() {
        let mut bench = Bench::new::<VoltageControlledMemristor>(VTEAM);
        assert_eq!(bench.unknowns(), ["n0", "n1", "device:state"]);
        // Operating-point state remains fixed regardless of voltage overdrive.
        let evaluation = bench.evaluate(&[3.0, 0.0, 0.7]);
        assert_close(&[evaluation.residual[2]], &[0.3]);
        assert_close(&evaluation.conductance[2], &[0.0, 0.0, 1.0]);
        assert_close(&[evaluation.charge[2]], &[0.0]);
        assert_close(&evaluation.capacitance[2], &[0.0; 3]);
    }

    #[test]
    fn in_a_transient_its_state_holds_between_its_thresholds() {
        let mut bench = Bench::new::<VoltageControlledMemristor>(VTEAM).stepping();
        for voltage in [-0.99, 0.0, 0.49] {
            assert_close(&[rate(&mut bench, voltage, 0.7)], &[0.0]);
        }
        let evaluation = bench.evaluate(&[0.2, 0.0, 0.7]);
        assert_close(&[evaluation.charge[2]], &[0.7]);
        assert_close(&[evaluation.capacitance[2][2]], &[1.0]);
    }

    #[test]
    fn each_rate_is_the_speed_at_twice_its_threshold() {
        let mut bench = Bench::new::<VoltageControlledMemristor>(VTEAM).stepping();
        // Mid-range, its window is open: within e^(−40) of its speed.
        let open = 1.0 - (-40.0f64).exp();
        assert_close(&[rate(&mut bench, 1.0, 0.4)], &[1e5 * open]);
        assert_close(&[rate(&mut bench, -2.0, 0.4)], &[-2e6 * open]);
        // Further past, faster by the exponent's power.
        assert_close(&[rate(&mut bench, 1.5, 0.4)], &[1e5 * 4.0 * open]);
        assert_close(&[rate(&mut bench, -3.0, 0.4)], &[-2e6 * 8.0 * open]);
    }

    #[test]
    fn a_drifting_state_moves_with_the_charge_through_it() {
        let mut bench = Bench::new::<CurrentControlledMemristor>(DRIFT).stepping();
        let resistance = 100.0 + 15_900.0 * 0.4;
        for voltage in [2.0, -3.0] {
            let current = voltage / resistance;
            assert_close(&[rate(&mut bench, voltage, 0.4)], &[current / 1e-4]);
        }
    }

    #[test]
    fn its_window_stops_it_at_either_bound_and_lets_it_leave() {
        let mut bench = Bench::new::<VoltageControlledMemristor>(VTEAM).stepping();
        // Driven off at its off bound, it stops; past it, it returns.
        assert_close(&[rate(&mut bench, 1.0, 1.0)], &[0.0]);
        assert!(rate(&mut bench, 1.0, 1.001) < 0.0);
        // Driven on from its off bound, it leaves at once.
        let leaving = rate(&mut bench, -2.0, 1.0);
        assert_close(&[leaving], &[-2e6 * (1.0 - (-100.0f64).exp())]);
        // A window's width from the bound, it moves at 1 − 1/e of its speed.
        let slowed = rate(&mut bench, -2.0, 0.01);
        assert_close(&[slowed], &[-2e6 * (1.0 - (-1.0f64).exp())]);
        assert!(rate(&mut bench, -2.0, -0.001) > 0.0);
    }

    #[test]
    fn jacobians_and_gradients_are_exact() {
        let exponential = with(VTEAM, r#""law": "exponential", "window": 0.05"#);
        let drifting = with(DRIFT, r#""law": "exponential", "window": 0.05"#);
        // Above, between and below the thresholds, inside the window, near and past
        // each bound.
        for voltage in [0.8, 0.1, -1.7] {
            for state in [-0.02, 0.03, 0.5, 0.98, 1.03] {
                let x = [voltage, 0.0, state];
                for parameters in [VTEAM, exponential.as_str()] {
                    assert_jacobians_match::<VoltageControlledMemristor>(parameters, &x);
                    assert_stepping_jacobians_match::<VoltageControlledMemristor>(parameters, &x);
                    assert_parameter_gradients_match::<VoltageControlledMemristor>(parameters, &x);
                    assert_parameter_gradients_match_at::<VoltageControlledMemristor>(
                        parameters, &x, 0.0,
                    );
                }
                for parameters in [DRIFT, drifting.as_str()] {
                    assert_jacobians_match::<CurrentControlledMemristor>(parameters, &x);
                    assert_stepping_jacobians_match::<CurrentControlledMemristor>(parameters, &x);
                    assert_parameter_gradients_match::<CurrentControlledMemristor>(parameters, &x);
                    assert_parameter_gradients_match_at::<CurrentControlledMemristor>(
                        parameters, &x, 0.0,
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_unphysical_values() {
        let cases = [
            (r#""on_voltage": 0.1"#, "on_voltage is negative, not 0.1"),
            (r#""off_voltage": 0"#, "off_voltage is positive, not 0.0"),
            (r#""on_rate": -1"#, "on_rate is positive, not -1.0"),
            (
                r#""off_exponent": 0.5"#,
                "off_exponent is at least 1, not 0.5",
            ),
            (
                r#""off_resistance": 500"#,
                "off_resistance exceeds on_resistance, not 500.0 against 1000.0",
            ),
            (r#""state": 1.5"#, "state is at most 1, not 1.5"),
            (r#""window": 0"#, "window is positive, not 0.0"),
        ];
        for (change, message) in cases {
            let parameters = dedupe(&with(VTEAM, change));
            let error =
                Bench::try_new::<VoltageControlledMemristor>(&parameters).expect_err(change);
            assert_eq!(error.0, message);
        }
        let error = Bench::try_new::<CurrentControlledMemristor>(&with(
            r#"{"on_resistance": 1, "off_resistance": 2}"#,
            r#""charge": 0"#,
        ))
        .expect_err("no charge");
        assert_eq!(error.0, "charge is positive, not 0.0");
    }

    /// Deduplicates repeated keys in JSON object `parameters`, keeping each key's last value.
    fn dedupe(parameters: &str) -> String {
        let object: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(parameters).expect("an object");
        serde_json::Value::Object(object).to_string()
    }

    #[test]
    fn its_resistance_makes_thermal_noise_and_both_are_variables() {
        let mut bench = Bench::new::<CurrentControlledMemristor>(DRIFT);
        let x = [1.0, 0.0, 0.25];
        let conductance = bench.evaluate(&x).residual[0];
        let sources = bench.noise(&x);
        assert_close(
            &[source(&sources, "thermal").density(1.0)],
            &[thermal(conductance)],
        );
        assert_eq!(bench.variable_names(), ["resistance", "rate"]);
        let values = bench.variables();
        // The rate shows at an operating point, which holds the state.
        assert_close(&values, &[conductance.recip(), conductance / 1e-4]);
    }
}
