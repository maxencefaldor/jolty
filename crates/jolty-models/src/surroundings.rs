//! A device's own temperature, apart from the simulation's.

use std::borrow::Cow;

use jolty_ir::Parameters;
use schemars::{JsonSchema, Schema};
use serde_json::Value;

use crate::device::{extended, schema};
use crate::{Instance, Interface, Model, ParameterError, Specification};

/// A model whose every device may run at a temperature of its own, as SPICE's `TEMP` and
/// `DTEMP` set an instance's: the parameter `temperature`, in kelvins, holds the device
/// there whatever the simulation's, and `temperature_offset`, in kelvins, holds it that
/// far above the simulation's, below where negative. The device reads every other
/// parameter, and is made at that temperature: a resistor at 0 K makes no noise, as
/// PSpice's `T_ABS=-273.15` keeps a macromodel's resistors quiet, while a junction needs
/// a temperature above 0 K.
///
/// A device's own temperature is no parameter that sensitivities differentiate.
pub struct Surrounded<M: Model> {
    inner: M,
}

impl<M: Model> Surrounded<M> {
    /// The model `inner`, its devices free to run at their own temperatures.
    pub const fn new(inner: M) -> Self {
        Self { inner }
    }
}

/// Where a device's temperature comes from: the simulation's, unless its parameters give
/// one of these.
#[derive(Debug, Clone, Copy, Default, PartialEq, JsonSchema)]
pub(crate) struct Own {
    /// The temperature the device is held at whatever the simulation's, in kelvins; at
    /// least 0. Without it, the device is at the simulation's temperature.
    temperature: Option<f64>,
    /// How far the device's temperature lies above the simulation's, below where
    /// negative, in kelvins; the device's temperature at least 0. Without it, the device
    /// is at the simulation's temperature.
    temperature_offset: Option<f64>,
}

impl Own {
    /// The parameters that give a device's own temperature.
    const NAMES: [&str; 2] = ["temperature", "temperature_offset"];

    /// The device's own temperature in `parameters`, and its other parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when a temperature is no finite number, or both are
    /// given.
    fn read(parameters: &Parameters) -> Result<(Self, Cow<'_, Parameters>), ParameterError> {
        if !Self::NAMES
            .iter()
            .any(|name| parameters.contains_key(*name))
        {
            let own = Self {
                temperature: None,
                temperature_offset: None,
            };
            return Ok((own, Cow::Borrowed(parameters)));
        }
        if Self::NAMES
            .iter()
            .all(|name| parameters.contains_key(*name))
        {
            return Err(ParameterError(
                "a device has a temperature or a temperature offset, not both".to_owned(),
            ));
        }
        let mut others = parameters.clone();
        let mut number = |name: &str| {
            others
                .remove(name)
                .map(|value| match value.as_f64() {
                    Some(number) if number.is_finite() => Ok(number),
                    _ => Err(ParameterError(format!(
                        "{name} must be a finite number of kelvins, not {value}"
                    ))),
                })
                .transpose()
        };
        let own = Self {
            temperature: number("temperature")?,
            temperature_offset: number("temperature_offset")?,
        };
        Ok((own, Cow::Owned(others)))
    }

    /// The own-temperature keys actually set on this instance, if any.
    #[cfg(any(test, feature = "testing"))]
    pub(crate) fn present(self) -> Vec<&'static str> {
        Self::NAMES
            .into_iter()
            .zip([self.temperature, self.temperature_offset])
            .filter_map(|(name, value)| value.is_some().then_some(name))
            .collect()
    }

    /// The device's temperature, in kelvins, where the simulation's is `simulation`.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when it lies below 0 K.
    pub(crate) fn temperature(self, simulation: f64) -> Result<f64, ParameterError> {
        let temperature = match (self.temperature, self.temperature_offset) {
            (Some(held), _) => held,
            (None, Some(offset)) => simulation + offset,
            (None, None) => return Ok(simulation),
        };
        if temperature >= 0.0 {
            Ok(temperature)
        } else {
            Err(ParameterError(format!(
                "a device's own temperature lies at 0 K or above, not at {temperature:?} K"
            )))
        }
    }
}

impl<M: Model> Model for Surrounded<M> {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn read(&self, parameters: &Parameters) -> Result<Box<dyn Specification>, ParameterError> {
        let (own, others) = Own::read(parameters)?;
        let inner = self.inner.read(&others)?;
        Ok(Box::new(AtOwn { own, inner }))
    }

    fn schema(&self) -> Schema {
        extended(self.inner.schema(), &schema::<Own>())
    }

    fn foundry(&self) -> bool {
        self.inner.foundry()
    }
}

/// A device's specification, and its own temperature.
#[derive(Debug)]
struct AtOwn {
    own: Own,
    inner: Box<dyn Specification>,
}

impl Specification for AtOwn {
    fn interface(&self) -> Interface {
        self.inner.interface()
    }

    fn parameters(&self) -> Result<Parameters, ParameterError> {
        let mut written = self.inner.parameters()?;
        let given = [self.own.temperature, self.own.temperature_offset];
        for (name, value) in Own::NAMES.into_iter().zip(given) {
            if let Some(kelvins) = value {
                written.insert(name.to_owned(), Value::from(kelvins));
            }
        }
        Ok(written)
    }

    fn kernel(&self) -> Instance<'_> {
        self.inner.kernel().at_own(self.own)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Bench, assert_close};
    use crate::{REFERENCE_TEMPERATURE, Resistor};

    fn model() -> Surrounded<crate::Differentiable<Resistor>> {
        Surrounded::new(crate::Differentiable::<Resistor>::new())
    }

    /// The current through 1 kΩ, drifting by 1e-3 per kelvin, with 1 V across it.
    fn current(own: &str) -> f64 {
        let parameters = format!(r#"{{"resistance": 1000, "temperature_coefficient": 1e-3{own}}}"#);
        let mut bench =
            Bench::from_model(&model(), &parameters, REFERENCE_TEMPERATURE).expect("valid");
        bench.evaluate(&[1.0, 0.0]).residual[0]
    }

    #[test]
    fn a_device_runs_at_its_own_temperature_or_offset_from_the_simulations() {
        assert_close(&[current("")], &[1e-3]);
        let held = format!(r#", "temperature": {}"#, REFERENCE_TEMPERATURE + 50.0);
        assert_close(&[current(&held)], &[1e-3 / 1.05]);
        assert_close(&[current(r#", "temperature_offset": -20"#)], &[1e-3 / 0.98]);
    }

    #[test]
    fn a_devices_own_temperature_is_one_number_of_kelvins_above_zero() {
        let error = |own: &str| {
            let parameters = format!(r#"{{"resistance": 1000{own}}}"#);
            Bench::from_model(&model(), &parameters, REFERENCE_TEMPERATURE)
                .expect_err("an error")
                .to_string()
        };
        assert_eq!(
            error(r#", "temperature": 300, "temperature_offset": 1"#),
            "a device has a temperature or a temperature offset, not both"
        );
        assert_eq!(
            error(r#", "temperature": -5"#),
            "a device's own temperature lies at 0 K or above, not at -5.0 K"
        );
        assert_eq!(
            error(r#", "temperature": "hot""#),
            "temperature must be a finite number of kelvins, not \"hot\""
        );
    }
}
