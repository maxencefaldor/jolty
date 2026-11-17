//! Memristors: each a resistance between `positive` and `negative` that its state moves,
//! from its on resistance at 0 to its off resistance at 1, by its [law](MemristorLaw).
//!
//! The state is an unknown labelled `state`, which only time moves. At an operating point,
//! and in the analyses about one, it is the state the memristor was made in, as SPICE's
//! initial conditions hold a node; in a transient, what drives it moves it, and its window
//! slows it at either bound.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::kernel::dual::Dual;
use crate::kernel::parts;
use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Local, Nature, NoiseSource, ParameterError,
    Real, Stamp, UnknownIndex,
};

/// How a memristor's resistance follows its state `x`, from its on resistance at 0 to its
/// off resistance at 1, as VTEAM gives it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum MemristorLaw {
    /// `r_on + (r_off − r_on) · x`, as two resistances in series whose boundary moves.
    #[default]
    Linear,
    /// `r_on · (r_off / r_on)^x`, as a tunnel barrier whose width moves.
    Exponential,
}

/// A memristor's resistance law, and the window that slows its state at its bounds.
#[derive(Debug, Clone, Copy)]
struct Memristance<R: Real> {
    law: MemristorLaw,
    on_resistance: R,
    off_resistance: R,
    /// `ln(r_off / r_on)`.
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
                "off_resistance must be above on_resistance, not {off:?} against {on:?}"
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

    /// The conductance at the state `state`, held within its bounds.
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

    /// The rate at which the state moves, toward off when positive, given `drive`, the
    /// rate with nothing slowing it, at the state `state`: the window slows a drive toward
    /// either bound as the state nears it, and none away from it.
    ///
    /// With `d` the distance to the bound the drive moves toward, in widths of the window,
    /// the drive is slowed by `1 − e^(−d)` short of the bound, which stops the state there,
    /// and turned by `d` past it, which returns it, the two meeting with the same slope.
    fn rate(&self, state: Dual<R, 2>, drive: Dual<R, 2>) -> Dual<R, 2> {
        let toward_off = drive.value.value() > 0.0;
        // How free the state is to move is a law of the state alone.
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

/// What moves a memristor's state.
#[derive(Debug, Clone, Copy)]
enum Control<R: Real> {
    /// VTEAM's voltage thresholds: past `off_voltage`, the state moves toward off at
    /// `off_rate · (v/v_off − 1)^α_off`, and past `on_voltage`, toward on at
    /// `on_rate · (v/v_on − 1)^α_on`; between them, it holds.
    Voltage {
        on_voltage: R,
        off_voltage: R,
        on_rate: R,
        off_rate: R,
        on_exponent: R,
        off_exponent: R,
    },
    /// Linear ion drift: the state moves by the charge through the memristor over
    /// `charge`, the charge that carries it across its whole range.
    Current { charge: R },
}

impl<R: Real> Control<R> {
    /// The rate at which the state moves with nothing slowing it, toward off when positive,
    /// at `voltage` across a memristor of conductance `conductance`, both numbers of the
    /// voltage and the state.
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

/// Where a memristor's contributions go, among its local unknowns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemristorShape {
    state: usize,
    /// The entries joining `positive` and `negative` to both terminals and to the state.
    terminals: [[usize; 3]; 2],
    /// The entries joining the state to both terminals and to itself.
    own: [usize; 3],
}

/// A memristor's numbers.
#[derive(Debug, Clone, Copy)]
pub struct PreparedMemristor<R: Real> {
    memristance: Memristance<R>,
    control: Control<R>,
    /// The state it was made in.
    initial: R,
    temperature: f64,
}

/// What a memristor remembers: its resistance, and the rate at which its state moves, at
/// its last evaluation, unknown before the first.
#[derive(Debug, Clone, Copy)]
pub struct MemristorMemory([f64; 2]);

impl Default for MemristorMemory {
    fn default() -> Self {
        Self([f64::NAN; 2])
    }
}

/// The local indices of a memristor's terminals, `positive` and `negative`.
const POSITIVE_NEGATIVE: [usize; 2] = [0, 1];

/// The terminals of a memristor.
const TERMINALS: &[&str] = &["positive", "negative"];

/// The interface of a memristor: a resistance between its terminals.
fn interface() -> Interface {
    Interface::new(TERMINALS, &[Branch::new(0, 1, BranchKind::Conductive)])
}

impl<R: Real> PreparedMemristor<R> {
    /// A memristor's numbers, made in the state `state`, at `temperature`.
    fn new(
        (memristance, control): (Memristance<R>, Control<R>),
        state: R,
        temperature: R,
    ) -> Result<Self, ParameterError> {
        let initial = ParameterError::non_negative("state", state)?;
        if initial.value() > 1.0 {
            let state = initial.value();
            return Err(ParameterError(format!(
                "state must be at most 1, not {state:?}"
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
            at.value(positive) - at.value(negative),
            at.value(self.state),
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

    /// The thermal noise of the resistance its state sets.
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

/// The names of a memristor's variables: its resistance, and the rate at which its state
/// moves, toward off when positive, in full ranges per second, which an operating point
/// shows though it holds the state.
const VARIABLE_NAMES: [&str; 2] = ["resistance", "rate"];

fn one<R: Real>() -> R {
    R::constant(1.0)
}

fn a_hundredth<R: Real>() -> R {
    R::constant(0.01)
}

/// A voltage-controlled memristor, in VTEAM's model (Kvatinsky, Ramadan, Friedman and
/// Kolodny, 2015): the voltage of `positive` over `negative` past `off_voltage` moves its
/// state toward off, past `on_voltage` toward on, faster the further past, and between
/// them leaves it where it is.
///
/// With `v` that voltage, the state moves at `off_rate · (v/v_off − 1)^α_off` above
/// `v_off`, and at `−on_rate · (v/v_on − 1)^α_on` below `v_on`, so each rate is the
/// state's speed at twice its threshold, in full ranges per second; its window slows it
/// at either bound.
#[derive(Debug, Clone, Copy)]
pub struct VoltageControlledMemristor;

/// The parameters of a [`VoltageControlledMemristor`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields, bound = "")]
pub struct VoltageControlledMemristorParameters<R: Real = f64> {
    /// The resistance fully on, at a state of 0, in ohms; positive.
    pub on_resistance: R,
    /// The resistance fully off, at a state of 1, in ohms; above the on resistance.
    pub off_resistance: R,
    /// The voltage below which the state moves toward on, in volts; negative.
    pub on_voltage: R,
    /// The voltage above which the state moves toward off, in volts; positive.
    pub off_voltage: R,
    /// The state's speed toward on at twice the on voltage, in full ranges per second;
    /// positive.
    pub on_rate: R,
    /// The state's speed toward off at twice the off voltage, in full ranges per second;
    /// positive.
    pub off_rate: R,
    /// How steeply the speed toward on grows past the on voltage, `α_on`; at least 1;
    /// default 1.
    #[serde(default = "one::<R>")]
    pub on_exponent: R,
    /// How steeply the speed toward off grows past the off voltage, `α_off`; at least 1;
    /// default 1.
    #[serde(default = "one::<R>")]
    pub off_exponent: R,
    /// How the resistance follows the state; default linear.
    #[serde(default)]
    pub law: MemristorLaw,
    /// The state it was made in, from 0, fully on, to 1, fully off; default 1.
    #[serde(default = "one::<R>")]
    pub state: R,
    /// The width of the window that slows the state at either bound, a fraction of its
    /// range; positive; default 0.01.
    #[serde(default = "a_hundredth::<R>")]
    pub window: R,
}

impl Kernel for VoltageControlledMemristor {
    const NAME: &'static str = "voltage_controlled_memristor";
    type Parameters<R: Real> = VoltageControlledMemristorParameters<R>;
    type Shape = MemristorShape;
    type Prepared<R: Real> = PreparedMemristor<R>;
    type Memory = MemristorMemory;

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
                "on_voltage must be negative, not {on:?}"
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
            "{parameter} must be at least 1, not {value:?}"
        )))
    }
}

/// A current-controlled memristor, in the linear ion drift model of HP's titanium dioxide
/// memristor (Strukov, Snider, Stewart and Williams, 2008): its state moves with the
/// charge through it, from `positive` to `negative` toward off, by `q/Q` for a charge `q`,
/// where `Q` is the charge that carries it across its whole range; its window slows it at
/// either bound.
#[derive(Debug, Clone, Copy)]
pub struct CurrentControlledMemristor;

/// The parameters of a [`CurrentControlledMemristor`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields, bound = "")]
pub struct CurrentControlledMemristorParameters<R: Real = f64> {
    /// The resistance fully on, at a state of 0, in ohms; positive.
    pub on_resistance: R,
    /// The resistance fully off, at a state of 1, in ohms; above the on resistance.
    pub off_resistance: R,
    /// The charge that carries the state across its whole range, in coulombs; positive.
    pub charge: R,
    /// How the resistance follows the state; default linear.
    #[serde(default)]
    pub law: MemristorLaw,
    /// The state it was made in, from 0, fully on, to 1, fully off; default 1.
    #[serde(default = "one::<R>")]
    pub state: R,
    /// The width of the window that slows the state at either bound, a fraction of its
    /// range; positive; default 0.01.
    #[serde(default = "a_hundredth::<R>")]
    pub window: R,
}

impl Kernel for CurrentControlledMemristor {
    const NAME: &'static str = "current_controlled_memristor";
    type Parameters<R: Real> = CurrentControlledMemristorParameters<R>;
    type Shape = MemristorShape;
    type Prepared<R: Real> = PreparedMemristor<R>;
    type Memory = MemristorMemory;

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

    /// `parameters` with `extra`, a JSON object's members, added.
    fn with(parameters: &str, extra: &str) -> String {
        format!("{}, {extra}}}", parameters.trim_end().trim_end_matches('}'))
    }

    /// The rate at which the state moves at `voltage` and `state`, as a transient sees it.
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
        // However far past its threshold the voltage is.
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
            (
                r#""on_voltage": 0.1"#,
                "on_voltage must be negative, not 0.1",
            ),
            (
                r#""off_voltage": 0"#,
                "off_voltage must be positive, not 0.0",
            ),
            (r#""on_rate": -1"#, "on_rate must be positive, not -1.0"),
            (
                r#""off_exponent": 0.5"#,
                "off_exponent must be at least 1, not 0.5",
            ),
            (
                r#""off_resistance": 500"#,
                "off_resistance must be above on_resistance, not 500.0 against 1000.0",
            ),
            (r#""state": 1.5"#, "state must be at most 1, not 1.5"),
            (r#""window": 0"#, "window must be positive, not 0.0"),
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
        assert_eq!(error.0, "charge must be positive, not 0.0");
    }

    /// The JSON object `parameters`, each member given twice keeping its last value.
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
