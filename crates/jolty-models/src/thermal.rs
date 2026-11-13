//! The elements of a thermal network: its nets are temperatures above the ambient, in
//! kelvins, its flows heat, in watts, and the ground is the ambient.
//!
//! A thermal resistance conducts heat as a resistor conducts current, a thermal
//! capacitance stores it as a capacitor stores charge, and a heat source drives it as a
//! current source drives current, so a network of them is solved as a circuit is.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    Binding, Branch, BranchKind, Device, Input, Interface, NativeDevice, NativeModel, Output,
    ParameterError, Real, TerminalPair,
};

/// A thermal resistance between thermal terminals `a` and `b`: heat flows through it from
/// the hotter to the colder at their difference over its resistance.
#[derive(Debug)]
pub struct ThermalResistance<R: Real = f64> {
    pair: TerminalPair,
    conductance: R,
}

/// The parameters of a [`ThermalResistance`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields, bound = "")]
pub struct ThermalResistanceParameters<R: Real = f64> {
    /// The resistance, in kelvins per watt; positive.
    pub resistance: R,
}

impl NativeModel for ThermalResistance {
    type Device<R: Real> = ThermalResistance<R>;
}

impl<R: Real> NativeDevice<R> for ThermalResistance<R> {
    const NAME: &'static str = "thermal_resistance";
    type Parameters = ThermalResistanceParameters<R>;

    fn interface(_: &ThermalResistanceParameters<R>) -> Interface {
        Interface::new(&["a", "b"], &[Branch::new(0, 1, BranchKind::Conductive)]).thermal()
    }

    fn new(
        parameters: ThermalResistanceParameters<R>,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let resistance = ParameterError::positive("resistance", parameters.resistance)?;
        Ok(Self {
            pair: TerminalPair::new(binding, binding.terminal(0), binding.terminal(1)),
            conductance: resistance.recip(),
        })
    }
}

impl<R: Real> Device<R> for ThermalResistance<R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        let difference = self.pair.voltage(input);
        self.pair.add_current(output, self.conductance * difference);
        self.pair.add_conductance(output, self.conductance);
    }
}

/// A thermal capacitance between thermal terminals `a` and `b`: it stores heat in
/// proportion to their difference, so it is open in a steady state.
#[derive(Debug)]
pub struct ThermalCapacitance<R: Real = f64> {
    pair: TerminalPair,
    capacitance: R,
}

/// The parameters of a [`ThermalCapacitance`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields, bound = "")]
pub struct ThermalCapacitanceParameters<R: Real = f64> {
    /// The heat capacity, in joules per kelvin; zero or positive.
    pub capacitance: R,
}

impl NativeModel for ThermalCapacitance {
    type Device<R: Real> = ThermalCapacitance<R>;
}

impl<R: Real> NativeDevice<R> for ThermalCapacitance<R> {
    const NAME: &'static str = "thermal_capacitance";
    type Parameters = ThermalCapacitanceParameters<R>;

    fn interface(_: &ThermalCapacitanceParameters<R>) -> Interface {
        Interface::new(&["a", "b"], &[Branch::new(0, 1, BranchKind::Capacitive)]).thermal()
    }

    fn new(
        parameters: ThermalCapacitanceParameters<R>,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let capacitance = ParameterError::non_negative("capacitance", parameters.capacitance)?;
        Ok(Self {
            pair: TerminalPair::new(binding, binding.terminal(0), binding.terminal(1)),
            capacitance,
        })
    }
}

impl<R: Real> Device<R> for ThermalCapacitance<R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        let difference = self.pair.voltage(input);
        self.pair.add_charge(output, self.capacitance * difference);
        self.pair.add_capacitance(output, self.capacitance);
    }
}

/// A heat source between thermal terminals `a` and `b`: it drives a constant heat flow
/// from `a` through itself into `b`, as a part beside a circuit heats the board under it.
#[derive(Debug)]
pub struct HeatSource<R: Real = f64> {
    pair: TerminalPair,
    power: R,
}

/// The parameters of a [`HeatSource`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields, bound = "")]
pub struct HeatSourceParameters<R: Real = f64> {
    /// The heat flow, in watts.
    pub power: R,
}

impl NativeModel for HeatSource {
    type Device<R: Real> = HeatSource<R>;
}

impl<R: Real> NativeDevice<R> for HeatSource<R> {
    const NAME: &'static str = "heat_source";
    type Parameters = HeatSourceParameters<R>;

    fn interface(_: &HeatSourceParameters<R>) -> Interface {
        Interface::new(&["a", "b"], &[Branch::new(0, 1, BranchKind::CurrentSource)]).thermal()
    }

    fn new(
        parameters: HeatSourceParameters<R>,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        if !parameters.power.value().is_finite() {
            return Err(ParameterError(format!(
                "power must be finite, not {:?}",
                parameters.power.value()
            )));
        }
        Ok(Self {
            pair: TerminalPair::new(binding, binding.terminal(0), binding.terminal(1)),
            power: parameters.power,
        })
    }
}

impl<R: Real> Device<R> for HeatSource<R> {
    fn evaluate(&mut self, _: &Input<'_>, output: &mut Output<'_, R>) {
        self.pair.add_current(output, self.power);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Bench, assert_close, assert_jacobians_match};

    #[test]
    fn heat_flows_as_current_does() {
        let mut resistance = Bench::new::<ThermalResistance>(r#"{"resistance": 50}"#);
        let evaluation = resistance.evaluate(&[30.0, 10.0]);
        assert_close(&evaluation.residual, &[0.4, -0.4]);
        let mut capacitance = Bench::new::<ThermalCapacitance>(r#"{"capacitance": 2e-3}"#);
        assert_close(&capacitance.evaluate(&[5.0, 1.0]).charge, &[8e-3, -8e-3]);
        let mut source = Bench::new::<HeatSource>(r#"{"power": 0.25}"#);
        assert_close(&source.evaluate(&[0.0, 0.0]).residual, &[0.25, -0.25]);
        assert_jacobians_match::<ThermalResistance>(r#"{"resistance": 50}"#, &[30.0, 10.0]);
        assert_jacobians_match::<ThermalCapacitance>(r#"{"capacitance": 2e-3}"#, &[5.0, 1.0]);
    }

    #[test]
    fn every_terminal_is_thermal_and_values_are_checked() {
        let interface =
            ThermalResistance::<f64>::interface(&ThermalResistanceParameters { resistance: 1.0 });
        assert_eq!(interface.thermal, [0, 1]);
        for (result, message) in [
            (
                Bench::try_new::<ThermalResistance>(r#"{"resistance": 0}"#),
                "resistance must be positive, not 0.0",
            ),
            (
                Bench::try_new::<ThermalCapacitance>(r#"{"capacitance": -1}"#),
                "capacitance must be at least 0, not -1.0",
            ),
        ] {
            assert_eq!(result.unwrap_err().to_string(), message);
        }
    }
}
