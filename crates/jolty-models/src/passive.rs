use jolty_numerics::Entry;
use serde::Deserialize;

use crate::junction::Temperatures;
use crate::stamp::{CurrentBranch, SMALL_RESISTANCE};
use crate::{
    Binding, Branch, BranchKind, Device, Heatable, HeatableModel, Input, Interface, NativeDevice,
    NativeModel, NoiseSource, Output, ParameterError, REFERENCE_TEMPERATURE, Real, TerminalPair,
};

/// The default nominal temperature, at which a passive device's value was measured.
fn reference_temperature<R: Real>() -> R {
    R::constant(REFERENCE_TEMPERATURE)
}

/// A resistor between terminals `a` and `b`, whose resistance drifts with temperature and
/// changes with its voltage.
///
/// With `v` from `a` to `b`, its resistance is `r · (1 + vc1 · v + vc2 · v²)`, `r` its
/// resistance at that temperature; where the coefficients bend it below zero, it is
/// negative there. Below [`SMALL_RESISTANCE`], its current is an unknown labelled
/// `current`.
#[derive(Debug)]
pub struct Resistor<R: Real = f64> {
    form: Form,
    resistance: R,
    coefficients: VoltageCoefficients<R>,
    temperature: f64,
}

/// How a resistor adds its current to its nodes' equations.
#[derive(Debug)]
enum Form {
    /// As a conductance between them.
    Conductance(TerminalPair),
    /// As an unknown of its own, held by Ohm's law `v − r · f(v) · i = 0`, with the entry
    /// at (current, current).
    Current(CurrentBranch, Option<Entry>),
}

/// How a passive device's value changes with its voltage `v`: by `1 + vc1 · v + vc2 · v²`.
#[derive(Debug, Clone, Copy)]
struct VoltageCoefficients<R: Real> {
    linear: R,
    quadratic: R,
}

impl<R: Real> VoltageCoefficients<R> {
    /// The factor at `voltage`, and its derivative with respect to the voltage.
    fn factor(self, voltage: f64) -> (R, R) {
        (
            self.linear * voltage + self.quadratic * (voltage * voltage) + 1.0,
            self.linear + self.quadratic * (2.0 * voltage),
        )
    }

    /// The integral of the factor from 0 to `voltage`.
    fn integral(self, voltage: f64) -> R {
        let squared = voltage * voltage;
        self.linear * (squared / 2.0) + self.quadratic * (squared * voltage / 3.0) + voltage
    }
}

/// The parameters of a [`Resistor`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, bound = "")]
pub struct ResistorParameters<R: Real = f64> {
    /// The resistance, in ohms, at the nominal temperature; positive.
    pub resistance: R,
    /// The linear temperature coefficient `tc1`, per kelvin; default 0.
    #[serde(default)]
    pub temperature_coefficient: R,
    /// The quadratic temperature coefficient `tc2`, per kelvin squared; default 0.
    #[serde(default)]
    pub quadratic_temperature_coefficient: R,
    /// The exponential temperature coefficient `tce`, in percent per kelvin: given, the
    /// resistance is multiplied by `1.01^(tce · ΔT)` instead of the polynomial.
    #[serde(default)]
    pub exponential_temperature_coefficient: Option<R>,
    /// The temperature the resistance was measured at, in kelvins; default 27 °C.
    #[serde(default = "reference_temperature")]
    pub nominal_temperature: R,
    /// The linear voltage coefficient `vc1`, per volt; default 0.
    #[serde(default)]
    pub voltage_coefficient: R,
    /// The quadratic voltage coefficient `vc2`, per volt squared; default 0.
    #[serde(default)]
    pub quadratic_voltage_coefficient: R,
}

impl NativeModel for Resistor {
    type Device<R: Real> = Resistor<R>;
}

impl<R: Real> NativeDevice<R> for Resistor<R> {
    const NAME: &'static str = "resistor";
    type Parameters = ResistorParameters<R>;

    fn interface(_: &ResistorParameters<R>) -> Interface {
        Interface::new(&["a", "b"], &[Branch::new(0, 1, BranchKind::Conductive)])
    }

    fn new(
        parameters: ResistorParameters<R>,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let temperature = binding.temperature();
        let resistance = parameters.resistance_at(R::constant(temperature))?;
        let (a, b) = (binding.terminal(0), binding.terminal(1));
        let form = if resistance.value().abs() < SMALL_RESISTANCE {
            let branch = CurrentBranch::new(binding, "current", [a, b]);
            let row = Some(branch.unknown());
            let entry = binding.entry(row, row);
            Form::Current(branch, entry)
        } else {
            Form::Conductance(TerminalPair::new(binding, a, b))
        };
        Ok(Self {
            form,
            resistance,
            coefficients: VoltageCoefficients {
                linear: parameters.voltage_coefficient,
                quadratic: parameters.quadratic_voltage_coefficient,
            },
            temperature,
        })
    }
}

impl<R: Real> ResistorParameters<R> {
    /// The resistance at `temperature`, in kelvins, once each parameter it reads is
    /// checked.
    fn resistance_at(&self, temperature: R) -> Result<R, ParameterError> {
        let resistance = ParameterError::positive("resistance", self.resistance)?;
        let temperatures = Temperatures::new(temperature, self.nominal_temperature)?;
        let (linear, quadratic) = (
            self.temperature_coefficient,
            self.quadratic_temperature_coefficient,
        );
        let factor = match self.exponential_temperature_coefficient {
            None => temperatures.polynomial(linear, quadratic),
            Some(exponential) if linear.value() == 0.0 && quadratic.value() == 0.0 => {
                R::constant(1.01).pow(exponential * temperatures.difference())
            }
            Some(_) => {
                return Err(ParameterError(
                    "an exponential temperature coefficient excludes the linear and \
                     quadratic ones"
                        .to_owned(),
                ));
            }
        };
        if factor.value() == 0.0 {
            return Err(ParameterError(format!(
                "the temperature coefficients scale the resistance to zero at {:?} K",
                temperatures.actual.value()
            )));
        }
        Ok(resistance * factor)
    }
}

/// A heated resistor keeps the form its resistance at the ambient chose: a current of its
/// own stays one however hot it runs.
impl<R: Real> Heatable<R> for Resistor<R> {
    fn heat(
        &mut self,
        parameters: &ResistorParameters<R>,
        temperature: R,
    ) -> Result<(), ParameterError> {
        self.resistance = parameters.resistance_at(temperature)?;
        self.temperature = temperature.value();
        Ok(())
    }
}

impl HeatableModel for Resistor {
    type Device<R: Real> = Resistor<R>;
}

impl<R: Real> Device<R> for Resistor<R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        match &self.form {
            Form::Conductance(pair) => {
                let voltage = pair.voltage(input);
                let (factor, slope) = self.coefficients.factor(voltage);
                // i = v / (r · f(v)), so di/dv = (f − v · f′) / (r · f²).
                let conductance = (self.resistance * factor).recip();
                pair.add_current(output, conductance * voltage);
                pair.add_conductance(output, conductance * (factor - slope * voltage) / factor);
            }
            Form::Current(branch, entry) => {
                let (voltage, current) = (branch.voltage(input), branch.current(input));
                let (factor, slope) = self.coefficients.factor(voltage);
                let resistance = self.resistance * factor;
                let residual = R::constant(voltage) - resistance * current;
                let along = R::constant(1.0) - self.resistance * slope * current;
                branch.add_sloped(input, output, residual, along);
                output.add_conductance(*entry, -resistance);
            }
        }
    }

    /// The thermal noise of the resistance at its voltage, `r · f(v)`: a current of
    /// `4kT / (r · f(v))` across it, or a voltage of `4kT · r · f(v)` in series with it
    /// where its current is an unknown.
    fn noise(&self, input: &Input<'_>, sources: &mut Vec<NoiseSource>) {
        let temperature = self.temperature;
        sources.push(match &self.form {
            Form::Conductance(pair) => {
                let (factor, _) = self.coefficients.factor(pair.voltage(input));
                let conductance = (self.resistance * factor).value().recip();
                NoiseSource::thermal("thermal", pair.nodes(), conductance, temperature)
            }
            Form::Current(branch, _) => {
                let (factor, _) = self.coefficients.factor(branch.voltage(input));
                let resistance = (self.resistance * factor).value();
                NoiseSource::thermal_in_series("thermal", branch.unknown(), resistance, temperature)
            }
        });
    }
}

/// A capacitor between terminals `a` and `b`, whose capacitance drifts with temperature
/// and changes with its voltage.
///
/// With `v` from `a` to `b`, its capacitance is `c · (1 + vc1 · v + vc2 · v²)`, `c` its
/// capacitance at that temperature, and its charge that capacitance's integral from 0 V.
#[derive(Debug)]
pub struct Capacitor<R: Real = f64> {
    pair: TerminalPair,
    capacitance: R,
    coefficients: VoltageCoefficients<R>,
}

/// The parameters of a [`Capacitor`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, bound = "")]
pub struct CapacitorParameters<R: Real = f64> {
    /// The capacitance, in farads, at the nominal temperature; zero or positive.
    pub capacitance: R,
    /// The linear temperature coefficient `tc1`, per kelvin; default 0.
    #[serde(default)]
    pub temperature_coefficient: R,
    /// The quadratic temperature coefficient `tc2`, per kelvin squared; default 0.
    #[serde(default)]
    pub quadratic_temperature_coefficient: R,
    /// The temperature the capacitance was measured at, in kelvins; default 27 °C.
    #[serde(default = "reference_temperature")]
    pub nominal_temperature: R,
    /// The linear voltage coefficient `vc1`, per volt; default 0.
    #[serde(default)]
    pub voltage_coefficient: R,
    /// The quadratic voltage coefficient `vc2`, per volt squared; default 0.
    #[serde(default)]
    pub quadratic_voltage_coefficient: R,
}

impl NativeModel for Capacitor {
    type Device<R: Real> = Capacitor<R>;
}

impl<R: Real> NativeDevice<R> for Capacitor<R> {
    const NAME: &'static str = "capacitor";
    type Parameters = CapacitorParameters<R>;

    fn interface(_: &CapacitorParameters<R>) -> Interface {
        Interface::new(&["a", "b"], &[Branch::new(0, 1, BranchKind::Capacitive)])
    }

    fn new(
        parameters: CapacitorParameters<R>,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let capacitance = ParameterError::non_negative("capacitance", parameters.capacitance)?
            * Temperatures::new(
                R::constant(binding.temperature()),
                parameters.nominal_temperature,
            )?
            .polynomial(
                parameters.temperature_coefficient,
                parameters.quadratic_temperature_coefficient,
            );
        let (a, b) = (binding.terminal(0), binding.terminal(1));
        Ok(Self {
            pair: TerminalPair::new(binding, a, b),
            capacitance,
            coefficients: VoltageCoefficients {
                linear: parameters.voltage_coefficient,
                quadratic: parameters.quadratic_voltage_coefficient,
            },
        })
    }
}

impl<R: Real> Device<R> for Capacitor<R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        let voltage = self.pair.voltage(input);
        let (factor, _) = self.coefficients.factor(voltage);
        self.pair.add_charge(
            output,
            self.capacitance * self.coefficients.integral(voltage),
        );
        self.pair.add_capacitance(output, self.capacitance * factor);
    }
}

/// A linear inductor between terminals `a` and `b`, whose inductance drifts with
/// temperature.
///
/// Its current, flowing from `a` through the inductor to `b`, is an unknown labelled
/// `current`. Its equation is `v_a − v_b − d/dt (L·i) = 0`, so at DC it is a short circuit.
#[derive(Debug)]
pub struct Inductor<R: Real = f64> {
    branch: CurrentBranch,
    inductance: R,
    /// The entry at (current, current).
    entry: Option<Entry>,
}

/// The parameters of an [`Inductor`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, bound = "")]
pub struct InductorParameters<R: Real = f64> {
    /// The inductance, in henries, at the nominal temperature; positive.
    pub inductance: R,
    /// The linear temperature coefficient `tc1`, per kelvin; default 0.
    #[serde(default)]
    pub temperature_coefficient: R,
    /// The quadratic temperature coefficient `tc2`, per kelvin squared; default 0.
    #[serde(default)]
    pub quadratic_temperature_coefficient: R,
    /// The temperature the inductance was measured at, in kelvins; default 27 °C.
    #[serde(default = "reference_temperature")]
    pub nominal_temperature: R,
}

impl NativeModel for Inductor {
    type Device<R: Real> = Inductor<R>;
}

impl<R: Real> NativeDevice<R> for Inductor<R> {
    const NAME: &'static str = "inductor";
    type Parameters = InductorParameters<R>;

    fn interface(_: &InductorParameters<R>) -> Interface {
        Interface::new(&["a", "b"], &[Branch::new(0, 1, BranchKind::Inductive)])
    }

    fn new(
        parameters: InductorParameters<R>,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let inductance = ParameterError::positive("inductance", parameters.inductance)?
            * Temperatures::new(
                R::constant(binding.temperature()),
                parameters.nominal_temperature,
            )?
            .polynomial(
                parameters.temperature_coefficient,
                parameters.quadratic_temperature_coefficient,
            );
        let terminals = [binding.terminal(0), binding.terminal(1)];
        let branch = CurrentBranch::new(binding, "current", terminals);
        let row = Some(branch.unknown());
        Ok(Self {
            branch,
            inductance,
            entry: binding.entry(row, row),
        })
    }
}

impl<R: Real> Device<R> for Inductor<R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        // v = L di/dt: the voltage, and the flux `L i` as the equation's charge, negated.
        let voltage = self.branch.voltage(input);
        self.branch.add(input, output, R::constant(voltage));
        let row = Some(self.branch.unknown());
        output.add_charge(row, -self.inductance * self.branch.current(input));
        output.add_capacitance(self.entry, -self.inductance);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Unknown;
    use crate::testing::{
        Bench, assert_close, assert_jacobians_match, assert_parameter_gradients_match, source,
        thermal,
    };

    #[test]
    fn a_resistor_conducts_by_ohms_law() {
        let mut bench = Bench::new::<Resistor>(r#"{"resistance": 1000}"#);
        assert_close(&bench.evaluate(&[3.0, 1.0]).residual, &[2e-3, -2e-3]);
    }

    #[test]
    fn a_heated_resistor_is_the_resistor_at_its_temperature() {
        let model = crate::Thermal::<Resistor>::new();
        let coefficients = r#""temperature_coefficient": 4e-3,
            "quadratic_temperature_coefficient": 1e-5, "voltage_coefficient": -0.05"#;
        for resistance in [1000.0, 1e-4] {
            let every = format!(r#"{{"resistance": {resistance}, {coefficients}}}"#);
            let heated = every.replace('{', r#"{"thermal": true, "#);
            let mut bench = Bench::of(&model, &heated, REFERENCE_TEMPERATURE).expect("valid");
            // A current of its own, where it has one, follows the thermal terminal.
            let current = (resistance < 1.0).then_some(2.0 / resistance);
            for rise in [0.0, 50.0] {
                let mut plain = Bench::at::<Resistor>(&every, REFERENCE_TEMPERATURE + rise);
                let hot = bench.settle(
                    &[3.0, 1.0, rise]
                        .into_iter()
                        .chain(current)
                        .collect::<Vec<_>>(),
                );
                let reference =
                    plain.settle(&[3.0, 1.0].into_iter().chain(current).collect::<Vec<_>>());
                let mut electrical = vec![hot.residual[0], hot.residual[1]];
                electrical.extend(current.map(|_| hot.residual[3]));
                assert_close(&electrical, &reference.residual);
                // It dissipates `v · i`, its current leaving `a` and entering `b`.
                let power = 3.0 * hot.residual[0] + 1.0 * hot.residual[1];
                assert_close(&[-hot.residual[2]], &[power]);
            }
            for rise in [10.0, 80.0] {
                let x: Vec<f64> = [2.5, -0.5, rise].into_iter().chain(current).collect();
                crate::testing::assert_model_jacobians_match(&model, &heated, &x);
                crate::testing::assert_model_parameter_gradients_match(&model, &heated, &x, 0.0);
            }
        }
    }

    #[test]
    fn a_small_resistor_carries_its_current_as_an_unknown() {
        let mut bench = Bench::new::<Resistor>(r#"{"resistance": 1e-6}"#);
        assert_eq!(bench.unknowns(), ["n0", "n1", "device:current"]);
        let output = bench.evaluate(&[3.0, 1.0, 2e6]);
        assert_close(&output.residual, &[2e6, -2e6, 0.0]);
        assert_jacobians_match::<Resistor>(r#"{"resistance": 0.5}"#, &[1.0, -0.5, 3.0]);
        let coefficients = r#""voltage_coefficient": -0.05, "quadratic_voltage_coefficient": 0.02"#;
        assert_jacobians_match::<Resistor>(
            &format!(r#"{{"resistance": 0.5, {coefficients}}}"#),
            &[1.0, -0.5, 2.5],
        );
        assert_parameter_gradients_match::<Resistor>(
            &format!(r#"{{"resistance": 0.5, "temperature_coefficient": 1e-3, {coefficients}}}"#),
            &[1.0, -0.5, 2.5],
        );
        let large = Bench::new::<Resistor>(r#"{"resistance": 1}"#);
        assert_eq!(large.unknowns(), ["n0", "n1"]);
    }

    #[test]
    fn a_capacitor_holds_charge() {
        let mut bench = Bench::new::<Capacitor>(r#"{"capacitance": 1e-6}"#);
        let output = bench.evaluate(&[2.0, 0.0]);
        assert_close(&output.charge, &[2e-6, -2e-6]);
        assert_close(&output.residual, &[0.0, 0.0]);
    }

    #[test]
    fn values_change_with_their_voltage_coefficients() {
        let coefficients = r#""voltage_coefficient": 0.1, "quadratic_voltage_coefficient": 0.01"#;
        let mut bench =
            Bench::new::<Resistor>(&format!(r#"{{"resistance": 1000, {coefficients}}}"#));
        let current = 2.0 / (1000.0 * (1.0 + 0.1 * 2.0 + 0.01 * 4.0));
        assert_close(&bench.evaluate(&[3.0, 1.0]).residual, &[current, -current]);
        let mut bench =
            Bench::new::<Capacitor>(&format!(r#"{{"capacitance": 1e-6, {coefficients}}}"#));
        let charge = 1e-6 * (2.0 + 0.1 * 4.0 / 2.0 + 0.01 * 8.0 / 3.0);
        let output = bench.evaluate(&[2.0, 0.0]);
        assert_close(&output.charge, &[charge, -charge]);
    }

    #[test]
    fn an_inductor_adds_its_current_as_an_unknown() {
        let bench = Bench::new::<Inductor>(r#"{"inductance": 1e-3}"#);
        assert_eq!(bench.unknowns(), ["n0", "n1", "device:current"]);
    }

    #[test]
    fn values_drift_with_their_temperature_coefficients() {
        let parameters = r#"{"resistance": 1000, "temperature_coefficient": 1e-3,
            "quadratic_temperature_coefficient": 1e-5, "nominal_temperature": 300}"#;
        let mut bench = Bench::at::<Resistor>(parameters, 310.0);
        let factor = 1.0 + 1e-3 * 10.0 + 1e-5 * 100.0;
        assert_close(
            &bench.evaluate(&[1.0, 0.0]).residual,
            &[1e-3 / factor, -1e-3 / factor],
        );
        let mut bench = Bench::at::<Resistor>(
            r#"{"resistance": 1000, "exponential_temperature_coefficient": 0.5}"#,
            REFERENCE_TEMPERATURE + 10.0,
        );
        let factor = 1.01_f64.powf(5.0);
        assert_close(
            &bench.evaluate(&[1.0, 0.0]).residual,
            &[1e-3 / factor, -1e-3 / factor],
        );
        let error = Bench::try_at::<Resistor>(
            r#"{"resistance": 1000, "temperature_coefficient": -0.1}"#,
            REFERENCE_TEMPERATURE + 10.0,
        )
        .unwrap_err();
        assert!(
            error
                .0
                .starts_with("the temperature coefficients scale the resistance to zero")
        );
    }

    #[test]
    fn jacobians_match_finite_differences() {
        let coefficients = r#""voltage_coefficient": -0.05, "quadratic_voltage_coefficient": 0.02"#;
        assert_jacobians_match::<Resistor>(r#"{"resistance": 50}"#, &[1.0, -0.5]);
        assert_jacobians_match::<Resistor>(
            &format!(r#"{{"resistance": 50, {coefficients}}}"#),
            &[1.0, -0.5],
        );
        assert_jacobians_match::<Capacitor>(r#"{"capacitance": 2e-9}"#, &[1.0, -0.5]);
        assert_jacobians_match::<Capacitor>(
            &format!(r#"{{"capacitance": 2e-9, {coefficients}}}"#),
            &[1.0, -0.5],
        );
        assert_jacobians_match::<Inductor>(r#"{"inductance": 1e-3}"#, &[1.0, -0.5, 0.25]);
    }

    #[test]
    fn parameter_gradients_are_exact() {
        let drifting = r#""temperature_coefficient": 1e-3,
            "quadratic_temperature_coefficient": 1e-5, "nominal_temperature": 290"#;
        let nonlinear = format!(
            r#"{drifting}, "voltage_coefficient": -0.05, "quadratic_voltage_coefficient": 0.02"#
        );
        assert_parameter_gradients_match::<Resistor>(
            &format!(r#"{{"resistance": 50, {nonlinear}}}"#),
            &[1.0, -0.5],
        );
        assert_parameter_gradients_match::<Resistor>(
            r#"{"resistance": 50, "exponential_temperature_coefficient": 0.3}"#,
            &[1.0, -0.5],
        );
        assert_parameter_gradients_match::<Capacitor>(
            &format!(r#"{{"capacitance": 2e-9, {nonlinear}}}"#),
            &[1.0, -0.5],
        );
        assert_parameter_gradients_match::<Inductor>(
            &format!(r#"{{"inductance": 1e-3, {drifting}}}"#),
            &[1.0, -0.5, 0.25],
        );
    }

    #[test]
    fn rejects_non_positive_values() {
        let error = Bench::try_new::<Resistor>(r#"{"resistance": 0}"#).unwrap_err();
        assert_eq!(error.to_string(), "resistance must be positive, not 0.0");
        let error = Bench::try_new::<Capacitor>(r#"{"capacity": 1}"#).unwrap_err();
        assert!(
            error.to_string().starts_with("unknown field `capacity`"),
            "{error}"
        );
    }

    #[test]
    fn a_resistor_makes_thermal_noise() {
        let bench = Bench::new::<Resistor>(r#"{"resistance": 1000}"#);
        let sources = bench.noise(&[3.0, 1.0]);
        let noise = source(&sources, "thermal");
        assert_eq!(
            noise.equations,
            [Some(Unknown::new(0)), Some(Unknown::new(1))]
        );
        assert_close(&[noise.density(1.0)], &[thermal(1e-3)]);
        // A small one's is a voltage in series with its current, and a short makes none.
        let small = Bench::new::<Resistor>(r#"{"resistance": 0.5}"#);
        let sources = small.noise(&[1.0, 0.0, 2.0]);
        let noise = source(&sources, "thermal");
        assert_eq!(noise.equations, [Some(Unknown::new(2)), None]);
        assert_close(&[noise.density(1.0)], &[thermal(0.5)]);
        let bent = Bench::new::<Resistor>(r#"{"resistance": 1000, "voltage_coefficient": 0.1}"#);
        let sources = bent.noise(&[2.0, 0.0]);
        assert_close(
            &[source(&sources, "thermal").density(1.0)],
            &[thermal(1.0 / 1200.0)],
        );
    }
}
