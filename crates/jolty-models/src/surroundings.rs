//! Per-instance operating temperature overrides independent of the global simulation
//! temperature.

use std::borrow::Cow;

use jolty_ir::Parameters;
use schemars::{JsonSchema, Schema};
use serde_json::Value;

use crate::device::{extended, schema};
use crate::{Instance, Interface, Model, ParameterError, ParameterKind, Specification};

/// Model wrapper allowing each device instance to specify an independent temperature,
/// mirroring SPICE `TEMP` and `DTEMP`.
///
/// The `temperature` parameter (in kelvins) fixes the device temperature regardless of the
/// global simulation temperature, while `temperature_offset` (in kelvins) shifts it relative
/// to the global temperature. A resistor at 0 K emits no thermal noise (as with PSpice
/// `T_ABS=-273.15`), while junctions require a temperature above 0 K. Per-instance
/// temperatures are excluded from sensitivity differentiation.
pub struct Surrounded<M: Model> {
    inner: M,
}

impl<M: Model> Surrounded<M> {
    /// Wraps `inner` so instances accept per-device temperature overrides.
    pub const fn new(inner: M) -> Self {
        Self { inner }
    }
}

/// Per-instance temperature override extracted from device parameters.
#[derive(Debug, Clone, Copy, Default, PartialEq, JsonSchema)]
pub(crate) struct Own {
    /// Fixed device operating temperature, in kelvins (at least 0); defaults to the global
    /// simulation temperature when omitted.
    temperature: Option<f64>,
    /// Offset added to the global simulation temperature, in kelvins (resulting temperature
    /// at least 0); defaults to zero when omitted.
    temperature_offset: Option<f64>,
}

impl Own {
    /// Parameter keys controlling per-instance device temperature.
    const NAMES: [&str; 2] = ["temperature", "temperature_offset"];

    /// Extracts per-instance temperature keys from `parameters` and returns the remaining
    /// model parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when a temperature value is non-finite or both keys are
    /// specified.
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
                        "{name} is a finite number of kelvins, not {value}"
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

    /// Returns the per-instance temperature keys present on `self`.
    #[cfg(any(test, feature = "testing"))]
    pub(crate) fn present(self) -> Vec<&'static str> {
        Self::NAMES
            .into_iter()
            .zip([self.temperature, self.temperature_offset])
            .filter_map(|(name, value)| value.is_some().then_some(name))
            .collect()
    }

    /// Resolves the effective device temperature in kelvins given the global `simulation`
    /// temperature.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when the resolved temperature lies below 0 K.
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

    fn parameter_kinds(&self) -> &'static [(&'static str, ParameterKind)] {
        self.inner.parameter_kinds()
    }

    fn binned_parameters(&self) -> &'static [&'static str] {
        self.inner.binned_parameters()
    }
}

/// Device specification paired with a per-instance temperature override.
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

    /// Evaluates the current through a 1 kΩ resistor with 1e-3 / K drift under a 1 V bias.
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
            "temperature is a finite number of kelvins, not \"hot\""
        );
    }
}
