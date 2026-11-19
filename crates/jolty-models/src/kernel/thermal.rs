//! Thermal network elements modeling temperature rises above ambient (in kelvins) and
//! heat flows (in watts), with ground representing ambient temperature.
//!
//! Thermal resistance, thermal capacitance, and heat sources correspond directly to
//! electrical resistors, capacitors, and current sources, allowing thermal and electrical
//! networks to share one solver.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::dimension::{HEAT_CAPACITY, POWER, THERMAL_RESISTANCE};
use crate::kernel::parts::Pair;
use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Lift, Local, ParameterError, ParameterKind,
    ParameterMetadata, Real, Stamp,
};

/// Layout shape for two-terminal thermal elements storing the terminal `Pair`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThermalShape(Pair);

impl ThermalShape {
    /// Claims the entries across a thermal element's terminals, `a` and `b`.
    fn new(layout: &mut Layout) -> Self {
        Self(Pair::new(layout, 0, 1))
    }
}

/// Thermal resistance between thermal terminals `a` and `b`, conducting heat from the hotter
/// terminal to the colder terminal at `ΔT / R`.
#[derive(Debug, Clone, Copy)]
pub struct ThermalResistance;

/// Parameters for [`ThermalResistance`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift, ParameterMetadata)]
#[serde(deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct ThermalResistanceParameters<R: Real = f64> {
    /// Thermal resistance, in kelvins per watt; positive.
    #[dimension(THERMAL_RESISTANCE)]
    pub resistance: R,
}

impl Kernel for ThermalResistance {
    const NAME: &'static str = "thermal_resistance";
    type Parameters<R: Real> = ThermalResistanceParameters<R>;
    type Shape = ThermalShape;
    /// Prepared thermal conductance in watts per kelvin.
    type Prepared<R: Real> = R;
    type Memory = ();

    fn parameter_kinds() -> &'static [(&'static str, ParameterKind)] {
        ThermalResistanceParameters::<f64>::PARAMETER_KINDS
    }

    fn interface<R: Real>(_: &ThermalResistanceParameters<R>) -> Interface {
        Interface::new(&["a", "b"], &[Branch::new(0, 1, BranchKind::Conductive)]).thermal()
    }

    fn prepare<R: Real>(
        parameters: &ThermalResistanceParameters<R>,
        _: R,
    ) -> Result<R, ParameterError> {
        Ok(ParameterError::positive("resistance", parameters.resistance)?.recip())
    }

    fn layout<R: Real>(
        _: &ThermalResistanceParameters<R>,
        _: &R,
        layout: &mut Layout,
    ) -> Result<ThermalShape, ParameterError> {
        Ok(ThermalShape::new(layout))
    }

    fn linear(_: &ThermalShape, _: &f64) -> bool {
        true
    }

    fn evaluate<R: Real>(
        ThermalShape(pair): &ThermalShape,
        &conductance: &R,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let difference = pair.across::<R>(at, pair.voltage(at));
        pair.add_current(stamp, conductance * difference);
        pair.add_conductance(stamp, conductance);
    }
}

/// Thermal capacitance between thermal terminals `a` and `b`, storing heat proportional to
/// temperature difference and acting as an open branch in steady state.
#[derive(Debug, Clone, Copy)]
pub struct ThermalCapacitance;

/// Parameters for [`ThermalCapacitance`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift, ParameterMetadata)]
#[serde(deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct ThermalCapacitanceParameters<R: Real = f64> {
    /// Heat capacity, in joules per kelvin; non-negative.
    #[dimension(HEAT_CAPACITY)]
    pub capacitance: R,
}

impl Kernel for ThermalCapacitance {
    const NAME: &'static str = "thermal_capacitance";
    type Parameters<R: Real> = ThermalCapacitanceParameters<R>;
    type Shape = ThermalShape;
    /// Prepared heat capacity in joules per kelvin.
    type Prepared<R: Real> = R;
    type Memory = ();

    fn parameter_kinds() -> &'static [(&'static str, ParameterKind)] {
        ThermalCapacitanceParameters::<f64>::PARAMETER_KINDS
    }

    fn interface<R: Real>(_: &ThermalCapacitanceParameters<R>) -> Interface {
        Interface::new(&["a", "b"], &[Branch::new(0, 1, BranchKind::Capacitive)]).thermal()
    }

    fn prepare<R: Real>(
        parameters: &ThermalCapacitanceParameters<R>,
        _: R,
    ) -> Result<R, ParameterError> {
        ParameterError::non_negative("capacitance", parameters.capacitance)
    }

    fn layout<R: Real>(
        _: &ThermalCapacitanceParameters<R>,
        _: &R,
        layout: &mut Layout,
    ) -> Result<ThermalShape, ParameterError> {
        Ok(ThermalShape::new(layout))
    }

    fn linear(_: &ThermalShape, _: &f64) -> bool {
        true
    }

    fn evaluate<R: Real>(
        ThermalShape(pair): &ThermalShape,
        &capacitance: &R,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let difference = pair.across::<R>(at, pair.voltage(at));
        pair.add_charge(stamp, capacitance * difference);
        pair.add_capacitance(stamp, capacitance);
    }
}

/// Constant heat source between thermal terminals `a` and `b` driving heat flow from `a`
/// into `b` (for example, an adjacent component heating a shared board).
#[derive(Debug, Clone, Copy)]
pub struct HeatSource;

/// Parameters for [`HeatSource`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift, ParameterMetadata)]
#[serde(deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct HeatSourceParameters<R: Real = f64> {
    /// Heat flow, in watts.
    #[dimension(POWER)]
    pub power: R,
}

impl Kernel for HeatSource {
    const NAME: &'static str = "heat_source";
    type Parameters<R: Real> = HeatSourceParameters<R>;
    type Shape = ThermalShape;
    /// Prepared heat flow in watts.
    type Prepared<R: Real> = R;
    type Memory = ();

    fn parameter_kinds() -> &'static [(&'static str, ParameterKind)] {
        HeatSourceParameters::<f64>::PARAMETER_KINDS
    }

    fn interface<R: Real>(_: &HeatSourceParameters<R>) -> Interface {
        Interface::new(&["a", "b"], &[Branch::new(0, 1, BranchKind::CurrentSource)]).thermal()
    }

    fn prepare<R: Real>(parameters: &HeatSourceParameters<R>, _: R) -> Result<R, ParameterError> {
        if !parameters.power.value().is_finite() {
            return Err(ParameterError(format!(
                "power is finite, not {:?}",
                parameters.power.value()
            )));
        }
        Ok(parameters.power)
    }

    fn layout<R: Real>(
        _: &HeatSourceParameters<R>,
        _: &R,
        layout: &mut Layout,
    ) -> Result<ThermalShape, ParameterError> {
        Ok(ThermalShape::new(layout))
    }

    fn evaluate<R: Real>(
        ThermalShape(pair): &ThermalShape,
        &power: &R,
        (): &mut (),
        _: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        pair.add_current(stamp, power);
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
            ThermalResistance::interface::<f64>(&ThermalResistanceParameters { resistance: 1.0 });
        assert_eq!(interface.thermal, [0, 1]);
        for (result, message) in [
            (
                Bench::try_new::<ThermalResistance>(r#"{"resistance": 0}"#),
                "resistance is positive, not 0.0",
            ),
            (
                Bench::try_new::<ThermalCapacitance>(r#"{"capacitance": -1}"#),
                "capacitance is at least 0, not -1.0",
            ),
        ] {
            assert_eq!(result.unwrap_err().to_string(), message);
        }
    }
}
