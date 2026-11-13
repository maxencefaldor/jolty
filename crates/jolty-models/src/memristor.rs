use jolty_numerics::Entry;
use serde::{Deserialize, Serialize};

use crate::{
    Binding, Branch, BranchKind, Device, Input, Interface, NativeDevice, NativeModel, Node,
    NoiseSource, Output, ParameterError, Quantity, Real, Unknown,
};

/// How a memristor's resistance follows its state `x`, from its on resistance at 0 to its
/// off resistance at 1, as VTEAM gives it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MemristorLaw {
    /// `r_on + (r_off − r_on) · x`, as two resistances in series whose boundary moves.
    #[default]
    Linear,
    /// `r_on · (r_off / r_on)^x`, as a tunnel barrier whose width moves.
    Exponential,
}

/// A memristor's resistance law, and the window that slows its state at its bounds.
#[derive(Debug)]
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

    /// The conductance at the state `state`, held within its bounds, and its derivative
    /// with respect to the state.
    fn conductance(&self, state: f64) -> (R, R) {
        let held = state.clamp(0.0, 1.0);
        let within = if (0.0..=1.0).contains(&state) {
            1.0
        } else {
            0.0
        };
        match self.law {
            MemristorLaw::Linear => {
                let span = self.off_resistance - self.on_resistance;
                let conductance = (self.on_resistance + span * held).recip();
                (conductance, -span * conductance * conductance * within)
            }
            MemristorLaw::Exponential => {
                let conductance = (-self.log_ratio * held).exp() / self.on_resistance;
                (conductance, -self.log_ratio * conductance * within)
            }
        }
    }

    /// The rate at which the state moves, toward off when positive, given `drive`, the
    /// rate with nothing slowing it, and its derivatives: the window slows a drive toward
    /// either bound as the state nears it, and none away from it.
    ///
    /// With `d` the distance to the bound the drive moves toward, in widths of the window,
    /// the drive is slowed by `1 − e^(−d)` short of the bound, which stops the state there,
    /// and turned by `d` past it, which returns it, the two meeting with the same slope.
    fn rate(&self, state: f64, drive: Drive<R>) -> Drive<R> {
        let toward_off = drive.value.value() > 0.0;
        let (distance, sign) = if toward_off {
            (R::constant(1.0 - state) / self.window, -1.0)
        } else {
            (R::constant(state) / self.window, 1.0)
        };
        let (freedom, slope) = if distance.value() >= 0.0 {
            let left = (-distance).exp();
            (-left + 1.0, left)
        } else {
            (distance, R::constant(1.0))
        };
        Drive {
            value: drive.value * freedom,
            voltage: drive.voltage * freedom,
            state: drive.state * freedom + drive.value * slope * sign / self.window,
        }
    }
}

/// A rate at which a memristor's state moves, toward off when positive, in full ranges
/// per second, with its derivatives with respect to the voltage across the memristor and
/// to its state.
#[derive(Debug, Clone, Copy)]
struct Drive<R: Real> {
    value: R,
    voltage: R,
    state: R,
}

/// What moves a memristor's state.
#[derive(Debug)]
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
    /// The drive at `voltage` across a memristor of conductance `conductance`, with its
    /// derivative with respect to the state.
    fn drive(&self, voltage: f64, (conductance, slope): (R, R)) -> Drive<R> {
        match self {
            Self::Voltage {
                on_voltage,
                off_voltage,
                on_rate,
                off_rate,
                on_exponent,
                off_exponent,
            } => {
                let (threshold, rate, exponent, sign) = if voltage > off_voltage.value() {
                    (*off_voltage, *off_rate, *off_exponent, 1.0)
                } else if voltage < on_voltage.value() {
                    (*on_voltage, *on_rate, *on_exponent, -1.0)
                } else {
                    let zero = R::constant(0.0);
                    return Drive {
                        value: zero,
                        voltage: zero,
                        state: zero,
                    };
                };
                let overdrive = R::constant(voltage) / threshold - 1.0;
                let value = rate * overdrive.pow(exponent) * sign;
                Drive {
                    value,
                    voltage: value * exponent / (overdrive * threshold),
                    state: R::constant(0.0),
                }
            }
            Self::Current { charge } => Drive {
                value: conductance * voltage / *charge,
                voltage: conductance / *charge,
                state: slope * voltage / *charge,
            },
        }
    }
}

/// A memristor between `positive` and `negative`: a resistance its state moves, from its
/// on resistance at 0 to its off resistance at 1, by its [law](MemristorLaw).
///
/// The state is an unknown labelled `state`, which only time moves. At an operating point,
/// and in the analyses about one, it is the state the memristor was made in, as SPICE's
/// initial conditions hold a node; in a transient, [what drives it](Control) moves it,
/// and its window slows it at either bound.
#[derive(Debug)]
struct Memristor<R: Real> {
    nodes: [Node; 2],
    state: Unknown,
    memristance: Memristance<R>,
    control: Control<R>,
    /// The state it was made in.
    initial: R,
    /// The entries joining `positive` and `negative` to both terminals and to the state.
    terminals: [[Option<Entry>; 3]; 2],
    /// The entries joining the state to both terminals and to itself.
    own: [Option<Entry>; 3],
    /// The resistance, and the rate at which the state moves, at the last evaluation.
    variables: [f64; 2],
    temperature: f64,
}

/// The terminals of a memristor.
const TERMINALS: &[&str] = &["positive", "negative"];

/// The interface of a memristor: a resistance between its terminals.
fn interface() -> Interface {
    Interface::new(TERMINALS, &[Branch::new(0, 1, BranchKind::Conductive)])
}

impl<R: Real> Memristor<R> {
    fn new(
        memristance: Memristance<R>,
        control: Control<R>,
        state: R,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let initial = ParameterError::non_negative("state", state)?;
        if initial.value() > 1.0 {
            let state = initial.value();
            return Err(ParameterError(format!(
                "state must be at most 1, not {state:?}"
            )));
        }
        let nodes = [0, 1].map(|index| binding.terminal(index));
        let state = binding.internal("state", Quantity::State);
        let columns = [nodes[0], nodes[1], Some(state)];
        Ok(Self {
            nodes,
            state,
            memristance,
            control,
            initial,
            terminals: nodes.map(|row| columns.map(|column| binding.entry(row, column))),
            own: columns.map(|column| binding.entry(Some(state), column)),
            variables: [f64::NAN; 2],
            temperature: binding.temperature(),
        })
    }

    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        let [positive, negative] = self.nodes;
        let voltage = input.value(positive) - input.value(negative);
        let state = input.value(Some(self.state));
        let conductance = self.memristance.conductance(state);
        let current = conductance.0 * voltage;
        output.add_residual(positive, current);
        output.add_residual(negative, -current);
        // With respect to positive, negative and the state.
        let derivatives = [conductance.0, -conductance.0, conductance.1 * voltage];
        for (row, sign) in self.terminals.iter().zip([1.0, -1.0]) {
            for (&entry, derivative) in row.iter().zip(derivatives) {
                output.add_conductance(entry, derivative * sign);
            }
        }
        let rate = self
            .memristance
            .rate(state, self.control.drive(voltage, conductance));
        self.variables = [conductance.0.value().recip(), rate.value.value()];
        let row = Some(self.state);
        let [on_positive, on_negative, own] = self.own;
        if input.transient {
            // dx/dt − rate = 0.
            output.add_residual(row, -rate.value);
            output.add_charge(row, R::constant(state));
            output.add_conductance(on_positive, -rate.voltage);
            output.add_conductance(on_negative, rate.voltage);
            output.add_conductance(own, -rate.state);
            output.add_capacitance(own, R::constant(1.0));
        } else {
            // x − x₀ = 0.
            output.add_residual(row, -self.initial + state);
            output.add_conductance(own, R::constant(1.0));
        }
    }

    /// The thermal noise of the resistance its state sets.
    fn noise(&self, input: &Input<'_>, sources: &mut Vec<NoiseSource>) {
        let (conductance, _) = self.memristance.conductance(input.value(Some(self.state)));
        sources.push(NoiseSource::thermal(
            "thermal",
            self.nodes,
            conductance.value(),
            self.temperature,
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
#[derive(Debug)]
pub struct VoltageControlledMemristor<R: Real = f64>(Memristor<R>);

/// The parameters of a [`VoltageControlledMemristor`].
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields, bound(deserialize = ""))]
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
    #[serde(default = "one")]
    pub on_exponent: R,
    /// How steeply the speed toward off grows past the off voltage, `α_off`; at least 1;
    /// default 1.
    #[serde(default = "one")]
    pub off_exponent: R,
    /// How the resistance follows the state; default linear.
    #[serde(default)]
    pub law: MemristorLaw,
    /// The state it was made in, from 0, fully on, to 1, fully off; default 1.
    #[serde(default = "one")]
    pub state: R,
    /// The width of the window that slows the state at either bound, a fraction of its
    /// range; positive; default 0.01.
    #[serde(default = "a_hundredth")]
    pub window: R,
}

impl NativeModel for VoltageControlledMemristor {
    type Device<R: Real> = VoltageControlledMemristor<R>;
}

impl<R: Real> NativeDevice<R> for VoltageControlledMemristor<R> {
    const NAME: &'static str = "voltage_controlled_memristor";
    type Parameters = VoltageControlledMemristorParameters<R>;

    fn interface(_: &VoltageControlledMemristorParameters<R>) -> Interface {
        interface()
    }

    fn new(
        parameters: VoltageControlledMemristorParameters<R>,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let VoltageControlledMemristorParameters {
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
        Memristor::new(memristance, control, state, binding).map(Self)
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

impl<R: Real> Device<R> for VoltageControlledMemristor<R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        self.0.evaluate(input, output);
    }

    fn variable_names(&self) -> &'static [&'static str] {
        &VARIABLE_NAMES
    }

    fn variables(&self, values: &mut [f64]) {
        values.copy_from_slice(&self.0.variables);
    }

    fn noise(&self, input: &Input<'_>, sources: &mut Vec<NoiseSource>) {
        self.0.noise(input, sources);
    }
}

/// A current-controlled memristor, in the linear ion drift model of HP's titanium dioxide
/// memristor (Strukov, Snider, Stewart and Williams, 2008): its state moves with the
/// charge through it, from `positive` to `negative` toward off, by `q/Q` for a charge `q`,
/// where `Q` is the charge that carries it across its whole range; its window slows it at
/// either bound.
#[derive(Debug)]
pub struct CurrentControlledMemristor<R: Real = f64>(Memristor<R>);

/// The parameters of a [`CurrentControlledMemristor`].
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields, bound(deserialize = ""))]
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
    #[serde(default = "one")]
    pub state: R,
    /// The width of the window that slows the state at either bound, a fraction of its
    /// range; positive; default 0.01.
    #[serde(default = "a_hundredth")]
    pub window: R,
}

impl NativeModel for CurrentControlledMemristor {
    type Device<R: Real> = CurrentControlledMemristor<R>;
}

impl<R: Real> NativeDevice<R> for CurrentControlledMemristor<R> {
    const NAME: &'static str = "current_controlled_memristor";
    type Parameters = CurrentControlledMemristorParameters<R>;

    fn interface(_: &CurrentControlledMemristorParameters<R>) -> Interface {
        interface()
    }

    fn new(
        parameters: CurrentControlledMemristorParameters<R>,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let CurrentControlledMemristorParameters {
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
        Memristor::new(memristance, control, state, binding).map(Self)
    }
}

impl<R: Real> Device<R> for CurrentControlledMemristor<R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        self.0.evaluate(input, output);
    }

    fn variable_names(&self) -> &'static [&'static str] {
        &VARIABLE_NAMES
    }

    fn variables(&self, values: &mut [f64]) {
        values.copy_from_slice(&self.0.variables);
    }

    fn noise(&self, input: &Input<'_>, sources: &mut Vec<NoiseSource>) {
        self.0.noise(input, sources);
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
        let device = bench.device();
        assert_eq!(device.variable_names(), ["resistance", "rate"]);
        let mut values = [0.0; 2];
        device.variables(&mut values);
        // The rate shows at an operating point, which holds the state.
        assert_close(&values, &[conductance.recip(), conductance / 1e-4]);
    }
}
