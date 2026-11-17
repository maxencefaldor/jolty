use std::fmt;
use std::marker::PhantomData;

use jolty_ir::Parameters;
use schemars::generate::SchemaSettings;
use schemars::{JsonSchema, Schema};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::{Instance, Interface, Kernel, Real};

/// A kind of device: its name, its parameters, and how to read an instance's into its
/// kernel's instance.
pub trait Model: Send + Sync {
    /// The name circuits refer to the model by, such as `resistor`.
    fn name(&self) -> &'static str;

    /// Reads `parameters`, once, into the instance of a kernel they make.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when a parameter is missing, unknown or of the wrong
    /// kind.
    fn read(&self, parameters: &Parameters) -> Result<Box<dyn Specification>, ParameterError>;

    /// Every parameter a device of the model takes, with its type, its default and its
    /// description, as a JSON schema of an object, every type within it written in place.
    fn schema(&self) -> Schema;

    /// `parameters` as the model takes them: each one left out, written with its default.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when the parameters cannot be read.
    fn defaulted(&self, parameters: &Parameters) -> Result<Parameters, ParameterError> {
        self.read(parameters)?.parameters()
    }

    /// Whether its devices are a foundry's, which ngspice simulates with the foundry's
    /// models: they elaborate, so the structural checks read their branches, but no
    /// analysis runs on a circuit that holds one.
    fn foundry(&self) -> bool {
        false
    }
}

/// An instance's parameters, read into its model's typed form: the terminals of its device,
/// and the instance of a kernel they make.
pub trait Specification: fmt::Debug + Send + Sync {
    /// The terminals and branches of the device.
    fn interface(&self) -> Interface;

    /// The parameters, each one left out written with its default, which read back as
    /// they are.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when they cannot be written as a JSON object.
    fn parameters(&self) -> Result<Parameters, ParameterError>;

    /// The instance as one of a [kernel](crate::Kernel)'s, which the engine evaluates
    /// beside the other instances of its kernel and shape.
    fn kernel(&self) -> Instance<'_>;
}

/// The [`Model`] of a [`Kernel`], whose instances give their exact derivatives with respect
/// to any of their parameters.
pub struct Differentiable<K>(PhantomData<fn() -> K>);

impl<K: Kernel> Differentiable<K> {
    /// The model of `K`.
    #[must_use]
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<K: Kernel> Default for Differentiable<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K> fmt::Debug for Differentiable<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Differentiable")
    }
}

impl<K: Kernel> Model for Differentiable<K> {
    fn name(&self) -> &'static str {
        K::NAME
    }

    fn read(&self, parameters: &Parameters) -> Result<Box<dyn Specification>, ParameterError> {
        Ok(Box::new(Typed::<K>(read(parameters)?)))
    }

    fn schema(&self) -> Schema {
        schema::<K::Parameters<f64>>()
    }

    fn foundry(&self) -> bool {
        K::FOUNDRY
    }
}

/// The parameters of an instance of `K`, read.
struct Typed<K: Kernel>(K::Parameters<f64>);

impl<K: Kernel> fmt::Debug for Typed<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Typed").field(&self.0).finish()
    }
}

impl<K: Kernel> Specification for Typed<K> {
    fn interface(&self) -> Interface {
        K::interface(&self.0)
    }

    fn parameters(&self) -> Result<Parameters, ParameterError> {
        written(&self.0)
    }

    fn kernel(&self) -> Instance<'_> {
        Instance::of::<K>(&self.0)
    }
}

/// The typed parameters `P` in `parameters`.
pub(crate) fn read<P: DeserializeOwned>(parameters: &Parameters) -> Result<P, ParameterError> {
    P::deserialize(parameters).map_err(|error| ParameterError(error.to_string()))
}

/// The typed `parameters` written as a JSON object, each one as it reads.
pub(crate) fn written<P: Serialize>(parameters: &P) -> Result<Parameters, ParameterError> {
    match serde_json::to_value(parameters) {
        Ok(Value::Object(written)) => Ok(written),
        Ok(other) => Err(ParameterError(format!(
            "parameters are written as an object, not as {other}"
        ))),
        Err(error) => Err(ParameterError(error.to_string())),
    }
}

/// The JSON schema of the parameters `P`, every type within it written in place.
pub(crate) fn schema<P: JsonSchema>() -> Schema {
    SchemaSettings::draft2020_12()
        .with(|settings| settings.inline_subschemas = true)
        .into_generator()
        .into_root_schema_for::<P>()
}

/// `schema`, the schema of an object, with the properties of `more`, another's, among its
/// own.
pub(crate) fn extended(mut schema: Schema, more: &Schema) -> Schema {
    let Some(added) = more.get("properties").and_then(Value::as_object) else {
        return schema;
    };
    let properties = schema
        .ensure_object()
        .entry("properties")
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    if let Some(properties) = properties.as_object_mut() {
        properties.extend(
            added
                .iter()
                .map(|(name, property)| (name.clone(), property.clone())),
        );
    }
    schema
}

/// Why a device's parameters are invalid.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ParameterError(pub String);

impl ParameterError {
    /// Checks that `value` is positive.
    ///
    /// # Errors
    ///
    /// Returns an error naming `parameter` otherwise.
    pub fn positive<R: Real>(parameter: &str, value: R) -> Result<R, Self> {
        if value.value() > 0.0 {
            Ok(value)
        } else {
            let value = value.value();
            Err(Self(format!("{parameter} must be positive, not {value:?}")))
        }
    }

    /// Checks that `value` is not negative.
    ///
    /// # Errors
    ///
    /// Returns an error naming `parameter` otherwise.
    pub fn non_negative<R: Real>(parameter: &str, value: R) -> Result<R, Self> {
        if value.value() >= 0.0 {
            Ok(value)
        } else {
            let value = value.value();
            Err(Self(format!(
                "{parameter} must be at least 0, not {value:?}"
            )))
        }
    }
}
