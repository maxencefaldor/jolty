//! Self-heating support coupling device operating temperature to a thermal net driven by
//! dissipated electrical power.
//!
//! A [kernel](crate::Kernel) prepares parameters at any operating temperature. When an
//! instance enables `"thermal": true`, [`Thermal`] appends a `thermal` terminal whose
//! voltage represents temperature rise above ambient, and injects dissipated power (summed
//! over terminal voltages and currents) into that thermal net. Newton iteration solves the
//! coupled electrical and thermal equations simultaneously with exact derivatives.

mod kind;

pub(crate) use kind::HeatedKernel;

use std::borrow::Cow;

use jolty_ir::Parameters;
use schemars::{JsonSchema, Schema};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::device::{extended, read, schema, written};
use crate::{
    Gradient, Hyper, Instance, Interface, Kernel, Model, ParameterError, ParameterKind, Real,
    Specification,
};

/// Scalar trait augmenting a [`Real`] number type with temperature-derivative tracking for
/// self-heating evaluation.
pub trait Heating: Real {
    /// Augmented scalar carrying value and temperature derivative.
    type Warm: Real;

    /// Seeds temperature `kelvins` as an independent variable with unit derivative.
    fn seed(kelvins: f64) -> Self::Warm;

    /// Splits a warm scalar into `(value, d/dT)` components.
    fn split(warm: Self::Warm) -> (Self, Self);
}

impl Heating for f64 {
    type Warm = Gradient;

    fn seed(kelvins: f64) -> Gradient {
        Gradient::new(kelvins, [1.0])
    }

    fn split(warm: Gradient) -> (Self, Self) {
        let [derivative] = warm.derivatives;
        (warm.value, derivative)
    }
}

impl<const K: usize> Heating for Gradient<K> {
    type Warm = Hyper<K>;

    fn seed(kelvins: f64) -> Hyper<K> {
        Hyper {
            value: kelvins,
            first: [0.0; K],
            second: 1.0,
            both: [0.0; K],
        }
    }

    fn split(warm: Hyper<K>) -> (Self, Self) {
        (
            Self::new(warm.value, warm.first),
            Self::new(warm.second, warm.both),
        )
    }
}

/// Maximum temperature step permitted between consecutive Newton iterations, in kelvins;
/// larger steps are clamped and extended linearly.
const MAX_TEMPERATURE_STEP: f64 = 50.0;

/// Lower temperature floor enforced during self-heating Newton evaluations, in kelvins,
/// regardless of how far a trial step drops.
const MIN_TEMPERATURE: f64 = 1.0;

/// [`Model`] adapter wrapping [`Kernel`] `K` to enable self-heating whenever `"thermal": true`.
pub struct Thermal<K>(std::marker::PhantomData<fn() -> K>);

impl<K: Kernel> Thermal<K> {
    /// Creates a self-heating-capable model adapter for kernel `K`.
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

/// Self-heating flag parsed from device parameters.
#[derive(Debug, Clone, Copy, JsonSchema)]
struct Heats {
    /// Flag enabling self-heating by adding a `thermal` terminal whose net temperature rise
    /// above ambient tracks dissipated power.
    #[schemars(default)]
    thermal: bool,
}

impl Heats {
    /// Extracts the optional boolean `thermal` flag from `parameters` and returns the
    /// remaining kernel parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when `thermal` is neither `true` nor `false`.
    fn read(parameters: &Parameters) -> Result<(Self, Cow<'_, Parameters>), ParameterError> {
        let Some(said) = parameters.get("thermal") else {
            return Ok((Self { thermal: false }, Cow::Borrowed(parameters)));
        };
        let Some(thermal) = said.as_bool() else {
            return Err(ParameterError(format!(
                "thermal is true or false, not {said}"
            )));
        };
        let mut others = parameters.clone();
        others.remove("thermal");
        Ok((Self { thermal }, Cow::Owned(others)))
    }
}

impl<K: Kernel> Model for Thermal<K>
where
    K::Parameters<f64>: DeserializeOwned + Serialize + JsonSchema,
{
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

    fn parameter_kinds(&self) -> &'static [(&'static str, ParameterKind)] {
        K::parameter_kinds()
    }

    fn binned_parameters(&self) -> &'static [&'static str] {
        K::binned_parameters()
    }
}

/// The parameters of an instance of `K`, read as plain numbers, and whether it heats, which
/// its kernel then computes with their derivatives with respect to its temperature.
enum Warmed<K: Kernel> {
    /// An instance at the temperature of its surroundings.
    Cool(K::Parameters<f64>),
    /// An instance that heats.
    Heated(K::Parameters<f64>),
}

impl<K: Kernel> std::fmt::Debug for Warmed<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cool(parameters) => f.debug_tuple("Cool").field(parameters).finish(),
            Self::Heated(parameters) => f.debug_tuple("Heated").field(parameters).finish(),
        }
    }
}

impl<K: Kernel> Specification for Warmed<K>
where
    K::Parameters<f64>: Serialize,
{
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
            Self::Cool(parameters) => Instance::from_parameters::<K>(parameters),
            Self::Heated(parameters) => Instance::from_kind::<HeatedKernel<K>>(parameters),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Bench, assert_close};
    use crate::{Diode, REFERENCE_TEMPERATURE};

    /// A diode with a series resistance and a junction capacitance, heating.
    const DIODE: &str = r#"{"saturation_current": 1e-14, "series_resistance": 2,
        "junction_capacitance": 2e-12, "transit_time": 1e-9, "thermal": true}"#;

    #[test]
    fn a_device_heats_only_when_asked() {
        let model = Thermal::<Diode>::new();
        let heated_bench = Bench::from_model(&model, DIODE, REFERENCE_TEMPERATURE).expect("valid");
        assert_eq!(
            heated_bench.unknowns(),
            ["n0", "n1", "n2", "device:junction"]
        );
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
        assert_eq!(error.to_string(), "thermal is true or false, not 1");
    }

    #[test]
    fn infinite_thermal_trial_step_does_not_poison_temperature() {
        let model = Thermal::<Diode>::new();
        let mut heated = Bench::from_model(&model, DIODE, REFERENCE_TEMPERATURE).expect("valid");
        let trial = heated.evaluate(&[0.72, 0.0, f64::INFINITY, 0.7]);
        assert!(trial.limited, "infinite thermal step triggers limiting");
        let recovered = heated.settle(&[0.72, 0.0, 20.0, 0.7]);
        assert!(
            recovered.residual.iter().all(|value| value.is_finite()),
            "residual poisoned after infinite trial step: {:?}",
            recovered.residual
        );
        let mut reference = Bench::at::<Diode>(
            &DIODE.replace(r#", "thermal": true"#, ""),
            REFERENCE_TEMPERATURE + 20.0,
        );
        let plain = reference.settle(&[0.72, 0.0, 0.7]);
        assert_close(
            &[
                recovered.residual[0],
                recovered.residual[1],
                recovered.residual[3],
            ],
            &plain.residual,
        );
    }
}
