//! A device's own temperature, apart from the simulation's.

use jolty_ir::Parameters;
use serde_json::Value;

use crate::{Binding, Device, Gradient, Interface, Model, ParameterError};

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

/// Where a device's temperature comes from.
enum Surroundings {
    /// The simulation's.
    Simulation,
    /// Its own, in kelvins.
    Held(f64),
    /// The simulation's raised by this many kelvins.
    Offset(f64),
}

impl Surroundings {
    /// The device's temperature, in kelvins, where the simulation's is `simulation`.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when it lies below 0 K.
    fn temperature(self, simulation: f64) -> Result<f64, ParameterError> {
        let temperature = match self {
            Self::Simulation => return Ok(simulation),
            Self::Held(temperature) => temperature,
            Self::Offset(offset) => simulation + offset,
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

/// The device's surroundings, and its other parameters.
///
/// # Errors
///
/// Returns [`ParameterError`] when a temperature is no finite number, or both are given.
fn surroundings(parameters: &Parameters) -> Result<(Surroundings, Parameters), ParameterError> {
    let mut own = parameters.clone();
    let number = |name: &str, value: Value| match value.as_f64() {
        Some(number) if number.is_finite() => Ok(number),
        _ => Err(ParameterError(format!(
            "{name} must be a finite number of kelvins, not {value}"
        ))),
    };
    let held = own.remove("temperature");
    let offset = own.remove("temperature_offset");
    let found = match (held, offset) {
        (None, None) => Surroundings::Simulation,
        (Some(held), None) => Surroundings::Held(number("temperature", held)?),
        (None, Some(offset)) => Surroundings::Offset(number("temperature_offset", offset)?),
        (Some(_), Some(_)) => {
            return Err(ParameterError(
                "a device has a temperature or a temperature offset, not both".to_owned(),
            ));
        }
    };
    Ok((found, own))
}

impl<M: Model> Model for Surrounded<M> {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn interface(&self, parameters: &Parameters) -> Result<Interface, ParameterError> {
        let (_, own) = surroundings(parameters)?;
        self.inner.interface(&own)
    }

    fn instantiate(
        &self,
        parameters: &Parameters,
        binding: &mut Binding<'_>,
    ) -> Result<Box<dyn Device>, ParameterError> {
        let (found, own) = surroundings(parameters)?;
        binding.set_temperature(found.temperature(binding.temperature())?);
        self.inner.instantiate(&own, binding)
    }

    fn instantiate_gradient(
        &self,
        parameters: &Parameters,
        binding: &mut Binding<'_>,
    ) -> Result<Box<dyn Device<Gradient>>, ParameterError> {
        let (found, own) = surroundings(parameters)?;
        binding.set_temperature(found.temperature(binding.temperature())?);
        self.inner.instantiate_gradient(&own, binding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Bench, assert_close};
    use crate::{REFERENCE_TEMPERATURE, Resistor};

    fn model() -> Surrounded<crate::Native<Resistor>> {
        Surrounded::new(crate::Native::<Resistor>::new())
    }

    /// The current through 1 kΩ, drifting by 1e-3 per kelvin, with 1 V across it.
    fn current(own: &str) -> f64 {
        let parameters = format!(r#"{{"resistance": 1000, "temperature_coefficient": 1e-3{own}}}"#);
        let mut bench = Bench::of(&model(), &parameters, REFERENCE_TEMPERATURE).expect("valid");
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
            Bench::of(&model(), &parameters, REFERENCE_TEMPERATURE)
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
