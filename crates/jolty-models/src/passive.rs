use serde::Deserialize;

use crate::junction::Temperatures;
use crate::{
    Binding, Branch, BranchKind, Device, Input, Interface, NativeDevice, NativeModel, Node, Output,
    ParameterError, Quantity, REFERENCE_TEMPERATURE, Real, TerminalPair, Unknown,
};
use jolty_numerics::Entry;

/// The default nominal temperature, at which a passive device's value was measured.
fn reference_temperature<R: Real>() -> R {
    R::constant(REFERENCE_TEMPERATURE)
}

/// A linear resistor between terminals `a` and `b`, whose resistance drifts with temperature.
#[derive(Debug)]
pub struct Resistor<R: Real = f64> {
    pair: TerminalPair,
    conductance: R,
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
        let resistance = ParameterError::positive("resistance", parameters.resistance)?;
        let temperatures =
            Temperatures::new(binding.temperature(), parameters.nominal_temperature)?;
        let (linear, quadratic) = (
            parameters.temperature_coefficient,
            parameters.quadratic_temperature_coefficient,
        );
        let factor = match parameters.exponential_temperature_coefficient {
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
                temperatures.actual
            )));
        }
        let (a, b) = (binding.terminal(0), binding.terminal(1));
        Ok(Self {
            pair: TerminalPair::new(binding, a, b),
            conductance: (resistance * factor).recip(),
        })
    }
}

impl<R: Real> Device<R> for Resistor<R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        self.pair
            .add_current(output, self.conductance * self.pair.voltage(input));
        self.pair.add_conductance(output, self.conductance);
    }
}

/// A linear capacitor between terminals `a` and `b`, whose capacitance drifts with
/// temperature.
#[derive(Debug)]
pub struct Capacitor<R: Real = f64> {
    pair: TerminalPair,
    capacitance: R,
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
            * Temperatures::new(binding.temperature(), parameters.nominal_temperature)?.polynomial(
                parameters.temperature_coefficient,
                parameters.quadratic_temperature_coefficient,
            );
        let (a, b) = (binding.terminal(0), binding.terminal(1));
        Ok(Self {
            pair: TerminalPair::new(binding, a, b),
            capacitance,
        })
    }
}

impl<R: Real> Device<R> for Capacitor<R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        self.pair
            .add_charge(output, self.capacitance * self.pair.voltage(input));
        self.pair.add_capacitance(output, self.capacitance);
    }
}

/// A linear inductor between terminals `a` and `b`, whose inductance drifts with
/// temperature.
///
/// Its current, flowing from `a` through the inductor to `b`, is an unknown labelled
/// `current`. Its equation is `v_a − v_b − d/dt (L·i) = 0`, so at DC it is a short circuit.
#[derive(Debug)]
pub struct Inductor<R: Real = f64> {
    a: Node,
    b: Node,
    current: Unknown,
    inductance: R,
    /// The entries at (a, current), (b, current), (current, a), (current, b), (current, current).
    entries: [Option<Entry>; 5],
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
            * Temperatures::new(binding.temperature(), parameters.nominal_temperature)?.polynomial(
                parameters.temperature_coefficient,
                parameters.quadratic_temperature_coefficient,
            );
        let (a, b) = (binding.terminal(0), binding.terminal(1));
        let current = binding.internal("current", Quantity::Current);
        let branch = Some(current);
        let entries = [
            binding.entry(a, branch),
            binding.entry(b, branch),
            binding.entry(branch, a),
            binding.entry(branch, b),
            binding.entry(branch, branch),
        ];
        Ok(Self {
            a,
            b,
            current,
            inductance,
            entries,
        })
    }
}

impl<R: Real> Device<R> for Inductor<R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        let [a_current, b_current, current_a, current_b, current_current] = self.entries;
        let (one, branch) = (R::constant(1.0), Some(self.current));
        let current = input.value(branch);
        output.add_residual(self.a, R::constant(current));
        output.add_residual(self.b, R::constant(-current));
        output.add_conductance(a_current, one);
        output.add_conductance(b_current, -one);
        output.add_residual(
            branch,
            R::constant(input.value(self.a) - input.value(self.b)),
        );
        output.add_conductance(current_a, one);
        output.add_conductance(current_b, -one);
        output.add_charge(branch, -self.inductance * current);
        output.add_capacitance(current_current, -self.inductance);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{
        Bench, assert_close, assert_jacobians_match, assert_parameter_gradients_match,
    };

    #[test]
    fn a_resistor_conducts_by_ohms_law() {
        let mut bench = Bench::new::<Resistor>(r#"{"resistance": 1000}"#);
        assert_close(&bench.evaluate(&[3.0, 1.0]).residual, &[2e-3, -2e-3]);
    }

    #[test]
    fn a_capacitor_holds_charge() {
        let mut bench = Bench::new::<Capacitor>(r#"{"capacitance": 1e-6}"#);
        let output = bench.evaluate(&[2.0, 0.0]);
        assert_close(&output.charge, &[2e-6, -2e-6]);
        assert_close(&output.residual, &[0.0, 0.0]);
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
        assert_jacobians_match::<Resistor>(r#"{"resistance": 50}"#, &[1.0, -0.5]);
        assert_jacobians_match::<Capacitor>(r#"{"capacitance": 2e-9}"#, &[1.0, -0.5]);
        assert_jacobians_match::<Inductor>(r#"{"inductance": 1e-3}"#, &[1.0, -0.5, 0.25]);
    }

    #[test]
    fn parameter_gradients_are_exact() {
        let drifting = r#""temperature_coefficient": 1e-3,
            "quadratic_temperature_coefficient": 1e-5, "nominal_temperature": 290"#;
        assert_parameter_gradients_match::<Resistor>(
            &format!(r#"{{"resistance": 50, {drifting}}}"#),
            &[1.0, -0.5],
        );
        assert_parameter_gradients_match::<Resistor>(
            r#"{"resistance": 50, "exponential_temperature_coefficient": 0.3}"#,
            &[1.0, -0.5],
        );
        assert_parameter_gradients_match::<Capacitor>(
            &format!(r#"{{"capacitance": 2e-9, {drifting}}}"#),
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
}
