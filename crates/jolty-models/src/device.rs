use std::fmt;
use std::marker::PhantomData;

use jolty_ir::Parameters;
use schemars::generate::SchemaSettings;
use schemars::{JsonSchema, Schema};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::{Instance, Interface, Kernel, ParameterKind, Real};

/// Device model descriptor providing the model name, parameter schema, and specification
/// parsing into a kernel instance.
pub trait Model: Send + Sync {
    /// Returns the model name used in circuit definitions, such as `resistor`.
    fn name(&self) -> &'static str;

    /// Parses `parameters` into a validated device specification backed by a kernel.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when a parameter is missing, unknown, or of the wrong
    /// kind.
    fn read(&self, parameters: &Parameters) -> Result<Box<dyn Specification>, ParameterError>;

    /// Returns the JSON schema describing all parameters of the model with types, defaults,
    /// and descriptions inlined.
    fn schema(&self) -> Schema;

    /// Returns `parameters` normalized with all omitted parameters populated from defaults.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when the parameters cannot be read.
    fn defaulted(&self, parameters: &Parameters) -> Result<Parameters, ParameterError> {
        self.read(parameters)?.parameters()
    }

    /// Returns whether instances represent foundry devices simulated externally by ngspice.
    ///
    /// Foundry devices elaborate so structural checks inspect their branches, while circuit
    /// analyses refuse circuits containing them.
    fn foundry(&self) -> bool {
        false
    }

    /// Returns all base model parameters paired with their physical dimension or kind in
    /// declaration order.
    fn parameter_kinds(&self) -> &'static [(&'static str, ParameterKind)] {
        &[]
    }

    /// Returns base names of parameters that expand into `l`, `w`, and `p` geometry-binning
    /// dependences.
    fn binned_parameters(&self) -> &'static [&'static str] {
        &[]
    }
}

/// Validated, typed parameters for a single device instance, exposing its terminal interface
/// and kernel binding.
pub trait Specification: fmt::Debug + Send + Sync {
    /// Returns the terminals and branches of the device.
    fn interface(&self) -> Interface;

    /// Returns the normalized parameter map with all omitted defaults filled in so it
    /// round-trips identically.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when the parameters cannot be serialized as a JSON object.
    fn parameters(&self) -> Result<Parameters, ParameterError>;

    /// Returns the device as a [`Kernel`](crate::Kernel) instance for batched evaluation
    /// alongside instances sharing the same kernel and shape.
    fn kernel(&self) -> Instance<'_>;
}

/// [`Model`] adapter wrapping a [`Kernel`] whose instances support exact parameter
/// differentiation.
pub struct Differentiable<K>(PhantomData<fn() -> K>);

impl<K: Kernel> Differentiable<K> {
    /// Creates a differentiable model adapter for kernel `K`.
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

impl<K: Kernel> Model for Differentiable<K>
where
    K::Parameters<f64>: DeserializeOwned + Serialize + JsonSchema,
{
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

    fn parameter_kinds(&self) -> &'static [(&'static str, ParameterKind)] {
        K::parameter_kinds()
    }

    fn binned_parameters(&self) -> &'static [&'static str] {
        K::binned_parameters()
    }
}

/// The parameters of an instance of `K`, read.
struct Typed<K: Kernel>(K::Parameters<f64>);

impl<K: Kernel> fmt::Debug for Typed<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Typed").field(&self.0).finish()
    }
}

impl<K: Kernel> Specification for Typed<K>
where
    K::Parameters<f64>: Serialize,
{
    fn interface(&self) -> Interface {
        K::interface(&self.0)
    }

    fn parameters(&self) -> Result<Parameters, ParameterError> {
        written(&self.0)
    }

    fn kernel(&self) -> Instance<'_> {
        Instance::from_parameters::<K>(&self.0)
    }
}

/// Deserializes typed parameters `P` from the JSON map `parameters`.
pub(crate) fn read<P: DeserializeOwned>(parameters: &Parameters) -> Result<P, ParameterError> {
    P::deserialize(parameters).map_err(|error| ParameterError(error.to_string()))
}

/// Serializes typed `parameters` into a JSON object map matching the input format.
pub(crate) fn written<P: Serialize>(parameters: &P) -> Result<Parameters, ParameterError> {
    match serde_json::to_value(parameters) {
        Ok(Value::Object(written)) => Ok(written),
        Ok(other) => Err(ParameterError(format!(
            "parameters are written as an object, not as {other}"
        ))),
        Err(error) => Err(ParameterError(error.to_string())),
    }
}

/// Generates a draft 2020-12 JSON schema for parameter type `P` with all subschemas inlined.
pub(crate) fn schema<P: JsonSchema>() -> Schema {
    SchemaSettings::draft2020_12()
        .with(|settings| settings.inline_subschemas = true)
        .into_generator()
        .into_root_schema_for::<P>()
}

/// Merges the `properties` entries from `more` into the object schema `schema`.
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

/// Validation or deserialization error describing why device parameters are invalid.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ParameterError(pub String);

impl ParameterError {
    /// Checks that `value` is strictly positive.
    ///
    /// # Errors
    ///
    /// Returns an error naming `parameter` when `value` is not strictly positive.
    pub fn positive<R: Real>(parameter: &str, value: R) -> Result<R, Self> {
        if value.value() > 0.0 {
            Ok(value)
        } else {
            let value = value.value();
            Err(Self(format!("{parameter} is positive, not {value:?}")))
        }
    }

    /// Checks that `value` is non-negative.
    ///
    /// # Errors
    ///
    /// Returns an error naming `parameter` when `value` is negative.
    pub fn non_negative<R: Real>(parameter: &str, value: R) -> Result<R, Self> {
        if value.value() >= 0.0 {
            Ok(value)
        } else {
            let value = value.value();
            Err(Self(format!("{parameter} is at least 0, not {value:?}")))
        }
    }
}
