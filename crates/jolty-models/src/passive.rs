use serde::Deserialize;

use crate::{
    Binding, Device, Input, NativeDevice, Node, Output, ParameterError, Quantity, TerminalPair,
    Unknown,
};
use jolty_numerics::Entry;

/// A linear resistor between terminals `a` and `b`.
#[derive(Debug)]
pub struct Resistor {
    pair: TerminalPair,
    conductance: f64,
}

/// The parameters of a [`Resistor`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResistorParameters {
    /// The resistance, in ohms; positive.
    pub resistance: f64,
}

impl NativeDevice for Resistor {
    const NAME: &'static str = "resistor";
    const TERMINALS: &'static [&'static str] = &["a", "b"];
    type Parameters = ResistorParameters;

    fn new(
        parameters: ResistorParameters,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let resistance = ParameterError::positive("resistance", parameters.resistance)?;
        let (a, b) = (binding.terminal(0), binding.terminal(1));
        Ok(Self {
            pair: TerminalPair::new(binding, a, b),
            conductance: 1.0 / resistance,
        })
    }
}

impl Device for Resistor {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_>) {
        self.pair
            .add_current(output, self.conductance * self.pair.voltage(input));
        self.pair.add_conductance(output, self.conductance);
    }
}

/// A linear capacitor between terminals `a` and `b`.
#[derive(Debug)]
pub struct Capacitor {
    pair: TerminalPair,
    capacitance: f64,
}

/// The parameters of a [`Capacitor`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapacitorParameters {
    /// The capacitance, in farads; zero or positive.
    pub capacitance: f64,
}

impl NativeDevice for Capacitor {
    const NAME: &'static str = "capacitor";
    const TERMINALS: &'static [&'static str] = &["a", "b"];
    type Parameters = CapacitorParameters;

    fn new(
        parameters: CapacitorParameters,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let capacitance = ParameterError::non_negative("capacitance", parameters.capacitance)?;
        let (a, b) = (binding.terminal(0), binding.terminal(1));
        Ok(Self {
            pair: TerminalPair::new(binding, a, b),
            capacitance,
        })
    }
}

impl Device for Capacitor {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_>) {
        self.pair
            .add_charge(output, self.capacitance * self.pair.voltage(input));
        self.pair.add_capacitance(output, self.capacitance);
    }
}

/// A linear inductor between terminals `a` and `b`.
///
/// Its current, flowing from `a` through the inductor to `b`, is an unknown labelled
/// `current`. Its equation is `v_a − v_b − d/dt (L·i) = 0`, so at DC it is a short circuit.
#[derive(Debug)]
pub struct Inductor {
    a: Node,
    b: Node,
    current: Unknown,
    inductance: f64,
    /// The entries at (a, current), (b, current), (current, a), (current, b), (current, current).
    entries: [Option<Entry>; 5],
}

/// The parameters of an [`Inductor`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InductorParameters {
    /// The inductance, in henries; positive.
    pub inductance: f64,
}

impl NativeDevice for Inductor {
    const NAME: &'static str = "inductor";
    const TERMINALS: &'static [&'static str] = &["a", "b"];
    type Parameters = InductorParameters;

    fn new(
        parameters: InductorParameters,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let inductance = ParameterError::positive("inductance", parameters.inductance)?;
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

impl Device for Inductor {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_>) {
        let [a_current, b_current, current_a, current_b, current_current] = self.entries;
        let branch = Some(self.current);
        let current = input.value(branch);
        output.add_residual(self.a, current);
        output.add_residual(self.b, -current);
        output.add_conductance(a_current, 1.0);
        output.add_conductance(b_current, -1.0);
        output.add_residual(branch, input.value(self.a) - input.value(self.b));
        output.add_conductance(current_a, 1.0);
        output.add_conductance(current_b, -1.0);
        output.add_charge(branch, -self.inductance * current);
        output.add_capacitance(current_current, -self.inductance);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Bench, assert_close, assert_jacobians_match};

    #[test]
    fn a_resistor_conducts_by_ohms_law() {
        let mut bench = Bench::new::<Resistor>(r#"{"resistance": 1000}"#, 2);
        assert_close(&bench.evaluate(&[3.0, 1.0]).residual, &[2e-3, -2e-3]);
    }

    #[test]
    fn a_capacitor_holds_charge() {
        let mut bench = Bench::new::<Capacitor>(r#"{"capacitance": 1e-6}"#, 2);
        let output = bench.evaluate(&[2.0, 0.0]);
        assert_close(&output.charge, &[2e-6, -2e-6]);
        assert_close(&output.residual, &[0.0, 0.0]);
    }

    #[test]
    fn an_inductor_adds_its_current_as_an_unknown() {
        let bench = Bench::new::<Inductor>(r#"{"inductance": 1e-3}"#, 2);
        assert_eq!(bench.unknowns(), ["n0", "n1", "device:current"]);
    }

    #[test]
    fn jacobians_match_finite_differences() {
        assert_jacobians_match::<Resistor>(r#"{"resistance": 50}"#, &[1.0, -0.5]);
        assert_jacobians_match::<Capacitor>(r#"{"capacitance": 2e-9}"#, &[1.0, -0.5]);
        assert_jacobians_match::<Inductor>(r#"{"inductance": 1e-3}"#, &[1.0, -0.5, 0.25]);
    }

    #[test]
    fn rejects_non_positive_values() {
        let error = Bench::try_new::<Resistor>(r#"{"resistance": 0}"#, 2).unwrap_err();
        assert_eq!(error.to_string(), "resistance must be positive, not 0");
        let error = Bench::try_new::<Capacitor>(r#"{"capacity": 1}"#, 2).unwrap_err();
        assert!(
            error.to_string().starts_with("unknown field `capacity`"),
            "{error}"
        );
    }
}
