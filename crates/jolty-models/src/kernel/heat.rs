//! Self-heating: a device at the temperature of a thermal net, heated by the power it
//! dissipates.
//!
//! A [kernel](crate::Kernel) prepares an instance's numbers at any temperature, as at the
//! simulation's. [`Thermal`] gives a kernel's model a thermal terminal, `thermal`, when an
//! instance's parameters say `"thermal": true`: the instance is then at the ambient
//! temperature plus that net's, and its power, the sum over its nodes of each voltage times
//! the current into the instance there, flows into the net as heat. Its temperature is an
//! unknown like any voltage, so Newton's iteration solves the circuit and its heating
//! together, with every derivative exact.

mod kind;

pub(crate) use kind::HeatedKernel;

use std::borrow::Cow;

use jolty_ir::Parameters;
use schemars::{JsonSchema, Schema};

use crate::device::{extended, read, schema, written};
use crate::{
    Gradient, Hyper, Instance, Interface, Kernel, Model, ParameterError, Real, Specification,
};

/// The numbers a device computes with, and those it computes with when it heats: the
/// same, with their derivative with respect to its temperature beside them.
pub trait Heating: Real {
    /// The number with its derivative with respect to the temperature.
    type Warm: Real;

    /// The temperature `kelvins`, the variable itself.
    fn seed(kelvins: f64) -> Self::Warm;

    /// A warm number's value, and its derivative with respect to the temperature.
    fn split(warm: Self::Warm) -> (Self, Self);
}

impl Heating for f64 {
    type Warm = Gradient;

    fn seed(kelvins: f64) -> Gradient {
        Gradient {
            value: kelvins,
            derivative: 1.0,
        }
    }

    fn split(warm: Gradient) -> (Self, Self) {
        (warm.value, warm.derivative)
    }
}

impl Heating for Gradient {
    type Warm = Hyper;

    fn seed(kelvins: f64) -> Hyper {
        Hyper {
            value: kelvins,
            first: 0.0,
            second: 1.0,
            both: 0.0,
        }
    }

    fn split(warm: Hyper) -> (Self, Self) {
        (
            Self {
                value: warm.value,
                derivative: warm.first,
            },
            Self {
                value: warm.second,
                derivative: warm.both,
            },
        )
    }
}

/// The largest step a heated device's temperature takes from one evaluation to the next,
/// in kelvins: a Newton step beyond it is limited, as a junction's voltage is, and the
/// device evaluated there and extended linearly.
const MOST_TEMPERATURE_STEP: f64 = 50.0;

/// The lowest temperature a heated device is evaluated at, in kelvins, however far below
/// it a Newton step goes.
const LOWEST_TEMPERATURE: f64 = 1.0;

/// The [`Model`] of a [`Kernel`] `K`, whose instances heat when their parameters say
/// `"thermal": true`.
pub struct Thermal<K>(std::marker::PhantomData<fn() -> K>);

impl<K: Kernel> Thermal<K> {
    /// The model of `K`.
    #[must_use]
    pub fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<K: Kernel> Default for Thermal<K> {
    fn default() -> Self {
        Self::new()
    }
}

/// Whether a device heats, as its parameters say.
#[derive(Debug, Clone, Copy, JsonSchema)]
struct Heats {
    /// Whether the device heats by the power it dissipates, through a terminal `thermal`
    /// joined to a thermal net, whose temperature above the ambient it runs at.
    #[schemars(default)]
    thermal: bool,
}

impl Heats {
    /// Whether `parameters` say the device heats, and the parameters without saying so.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when they say it with anything but true or false.
    fn read(parameters: &Parameters) -> Result<(Self, Cow<'_, Parameters>), ParameterError> {
        let Some(said) = parameters.get("thermal") else {
            return Ok((Self { thermal: false }, Cow::Borrowed(parameters)));
        };
        let Some(thermal) = said.as_bool() else {
            return Err(ParameterError(format!(
                "thermal must be true or false, not {said}"
            )));
        };
        let mut others = parameters.clone();
        others.remove("thermal");
        Ok((Self { thermal }, Cow::Owned(others)))
    }
}

impl<K: Kernel> Model for Thermal<K> {
    fn name(&self) -> &'static str {
        K::NAME
    }

    fn read(&self, parameters: &Parameters) -> Result<Box<dyn Specification>, ParameterError> {
        let (heats, others) = Heats::read(parameters)?;
        let warmed = if heats.thermal {
            Warmed::<K>::Heated(read(&others)?)
        } else {
            Warmed::<K>::Cool(read(&others)?)
        };
        Ok(Box::new(warmed))
    }

    fn schema(&self) -> Schema {
        extended(schema::<K::Parameters<f64>>(), &schema::<Heats>())
    }
}

/// The parameters of an instance of `K`, read as the numbers it computes with: plain, or,
/// when it heats, with their derivatives with respect to its temperature beside them.
enum Warmed<K: Kernel> {
    /// An instance at the temperature of its surroundings.
    Cool(K::Parameters<f64>),
    /// An instance that heats.
    Heated(K::Parameters<<f64 as Heating>::Warm>),
}

impl<K: Kernel> std::fmt::Debug for Warmed<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cool(parameters) => f.debug_tuple("Cool").field(parameters).finish(),
            Self::Heated(parameters) => f.debug_tuple("Heated").field(parameters).finish(),
        }
    }
}

impl<K: Kernel> Specification for Warmed<K> {
    fn interface(&self) -> Interface {
        match self {
            Self::Cool(parameters) => K::interface(parameters),
            Self::Heated(parameters) => {
                let mut interface = K::interface(parameters);
                interface.thermal.push(interface.terminals.len());
                interface.terminals.push("thermal".to_owned());
                interface
            }
        }
    }

    fn parameters(&self) -> Result<Parameters, ParameterError> {
        let (mut written, heats) = match self {
            Self::Cool(parameters) => (written(parameters)?, false),
            Self::Heated(parameters) => (written(parameters)?, true),
        };
        written.insert("thermal".to_owned(), heats.into());
        Ok(written)
    }

    fn kernel(&self) -> Instance<'_> {
        match self {
            Self::Cool(parameters) => Instance::of::<K>(parameters),
            Self::Heated(parameters) => Instance::of_kind::<HeatedKernel<K>>(parameters),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{
        Bench, assert_close, assert_model_jacobians_match, assert_model_parameter_gradients_match,
    };
    use crate::{Diode, REFERENCE_TEMPERATURE};

    /// A diode with a series resistance and a junction capacitance, heating.
    const DIODE: &str = r#"{"saturation_current": 1e-14, "series_resistance": 2,
        "junction_capacitance": 2e-12, "transit_time": 1e-9, "thermal": true}"#;

    #[test]
    fn a_heated_diode_is_the_diode_at_its_temperature() {
        let model = Thermal::<Diode>::new();
        let mut heated = Bench::from_model(&model, DIODE, REFERENCE_TEMPERATURE).expect("valid");
        assert_eq!(heated.unknowns(), ["n0", "n1", "n2", "device:junction"]);
        let cool = DIODE.replace(r#", "thermal": true"#, "");
        for rise in [0.0, 40.0] {
            let mut plain = Bench::at::<Diode>(&cool, REFERENCE_TEMPERATURE + rise);
            let x = [0.72, 0.0, rise, 0.7];
            let hot = heated.settle(&x);
            let reference = plain.settle(&[0.72, 0.0, 0.7]);
            let electrical = [hot.residual[0], hot.residual[1], hot.residual[3]];
            assert_close(&electrical, &reference.residual);
            // Its power, `Σ v · i` over its voltages, heats the thermal net.
            let power = 0.72 * hot.residual[0] + 0.7 * hot.residual[3];
            assert_close(&[-hot.residual[2]], &[power]);
        }
    }

    #[test]
    fn a_heated_diodes_derivatives_are_exact() {
        let model = Thermal::<Diode>::new();
        for x in [[0.72, 0.0, 15.0, 0.7], [-3.0, 0.0, 60.0, -2.9]] {
            assert_model_jacobians_match(&model, DIODE, &x);
            assert_model_parameter_gradients_match(&model, DIODE, &x, None);
        }
    }

    #[test]
    fn a_device_heats_only_when_asked() {
        let model = Thermal::<Diode>::new();
        let interface = |parameters: &str| {
            model
                .read(&serde_json::from_str(parameters).expect("JSON"))
                .expect("valid")
                .interface()
        };
        let heated = interface(r#"{"thermal": true}"#);
        assert_eq!(heated.terminals, ["anode", "cathode", "thermal"]);
        assert_eq!(heated.thermal, [2]);
        assert_eq!(
            interface(r#"{"thermal": false}"#).terminals,
            ["anode", "cathode"]
        );
        let error = model
            .read(&serde_json::from_str(r#"{"thermal": 1}"#).expect("JSON"))
            .unwrap_err();
        assert_eq!(error.to_string(), "thermal must be true or false, not 1");
    }
}
