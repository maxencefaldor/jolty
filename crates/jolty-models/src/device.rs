use std::fmt;
use std::marker::PhantomData;

use jolty_ir::Parameters;
use schemars::generate::SchemaSettings;
use schemars::{JsonSchema, Schema};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::{
    Binding, Choice, Detect, Drive, Gradient, Input, Interface, NoiseSource, Output, Real,
};

/// One placed device, bound to its unknowns, computing with plain numbers or with their
/// exact derivatives with respect to a parameter.
pub trait Device<R: Real = f64>: fmt::Debug + Send {
    /// Adds the device's contribution at `input` to `output`.
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>);

    /// Appends the times in `[start, stop]` where the device's behavior is not smooth, so a
    /// transient analysis steps onto them exactly.
    fn breakpoints(&self, start: f64, stop: f64, times: &mut Vec<f64>) {
        let _ = (start, stop, times);
    }

    /// Appends to `found` the choices the device's behavior turns on at `input`, where its
    /// contribution jumps when one changes, so a transient analysis can step onto each
    /// change: none for a device whose contribution is continuous.
    fn choices(&mut self, input: &Input<'_>, found: &mut Vec<Choice>) {
        let _ = (input, found);
    }

    /// The device as an independent source, if it is one.
    fn as_source(&mut self) -> Option<&mut dyn Source> {
        None
    }

    /// The device as the analog side of a logic output, if it is one.
    fn as_driver(&mut self) -> Option<&mut dyn Drive> {
        None
    }

    /// The device as the analog side of a logic input, if it is one.
    fn as_detector(&self) -> Option<&dyn Detect> {
        None
    }

    /// The names of the quantities the device computes on its way to its currents that
    /// neither its terminals, nor its unknowns, nor their derivatives show, such as a
    /// transistor's threshold voltage, in the order of [`Device::variables`].
    fn variable_names(&self) -> &'static [&'static str] {
        &[]
    }

    /// Writes the value of each of [`Device::variable_names`] at the device's last
    /// evaluation, in SI base units, to `values`.
    fn variables(&self, values: &mut [f64]) {
        let _ = values;
    }

    /// Appends the device's noise at the state `input`, the operating point a noise
    /// analysis linearizes about, to `sources`: none for a noiseless device.
    fn noise(&self, input: &Input<'_>, sources: &mut Vec<NoiseSource>) {
        let _ = (input, sources);
    }
}

/// An independent source, whose level analyses excite. A DC sweep holds one at a level
/// by making it again from the parameters [`held_source`](crate::held_source) gives.
pub trait Source {
    /// The source's own level at the operating point: its waveform's at time zero.
    fn level(&self) -> f64;

    /// Adds the derivative of the residual with respect to the source's level, one value per
    /// unknown, to `derivative`. A small-signal analysis excites the source through it.
    fn level_derivative(&self, derivative: &mut [f64]);
}

/// A kind of device: its name, its parameters, and how to read an instance's into what
/// makes its device.
pub trait Model: Send + Sync {
    /// The name circuits refer to the model by, such as `resistor`.
    fn name(&self) -> &'static str;

    /// Reads `parameters`, once, into what makes a device of the model, as often as it is
    /// made.
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
/// and how to make the device, bound to unknowns, each time it is made.
pub trait Specification: fmt::Debug + Send + Sync {
    /// The terminals and branches of the device.
    fn interface(&self) -> Interface;

    /// Makes the device bound through `binding`.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when a parameter is out of range.
    fn instantiate(&self, binding: &mut Binding<'_>) -> Result<Box<dyn Device>, ParameterError>;

    /// Makes the device bound through `binding`, its residuals and charges carrying their
    /// exact derivatives with respect to the number at `parameter`, its path among the
    /// parameters as [`jolty_ir::numbers`] names it, such as `waveform.dc`.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when the parameters hold no number there that the device
    /// reads as a real number, as when it reads it as a choice, such as a level, or when a
    /// parameter is out of range.
    fn instantiate_gradient(
        &self,
        parameter: &str,
        binding: &mut Binding<'_>,
    ) -> Result<Box<dyn Device<Gradient>>, ParameterError>;

    /// The parameters, each one left out written with its default, which read back as
    /// they are.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when they cannot be written as a JSON object.
    fn parameters(&self) -> Result<Parameters, ParameterError>;
}

/// A device implemented in Rust, whose parameters deserialize into a typed struct.
pub trait NativeDevice<R: Real = f64>: Device<R> + Sized + 'static {
    /// The model's name.
    const NAME: &'static str;
    /// Whether the device is a foundry's, as [`Model::foundry`] says.
    const FOUNDRY: bool = false;
    /// The parameters, deserialized from the instance's parameter map, which write as they
    /// read and which a JSON schema describes.
    type Parameters: DeserializeOwned + Serialize + JsonSchema + Clone + fmt::Debug + Send + Sync;

    /// The terminals and branches of a device with `parameters`.
    fn interface(parameters: &Self::Parameters) -> Interface;

    /// Checks `parameters` and makes the device bound through `binding`.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when a parameter is out of range.
    fn new(parameters: Self::Parameters, binding: &mut Binding<'_>)
    -> Result<Self, ParameterError>;
}

/// A model implemented in Rust, whose devices compute with any [`Real`].
pub trait NativeModel: 'static {
    /// The device, computing with `R`.
    type Device<R: Real>: NativeDevice<R>;
}

/// The parameters of the devices of `M` that compute with `R`.
pub(crate) type ParametersOf<M, R> = <<M as NativeModel>::Device<R> as NativeDevice<R>>::Parameters;

/// The [`Model`] of a [`NativeModel`], whose devices give their exact derivatives with
/// respect to any of their parameters.
pub struct Native<M>(PhantomData<fn() -> M>);

impl<M: NativeModel> Native<M> {
    /// The model of `M`.
    #[must_use]
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<M: NativeModel> Default for Native<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M> fmt::Debug for Native<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Native")
    }
}

impl<M: NativeModel> Model for Native<M> {
    fn name(&self) -> &'static str {
        <M::Device<f64> as NativeDevice<f64>>::NAME
    }

    fn read(&self, parameters: &Parameters) -> Result<Box<dyn Specification>, ParameterError> {
        Ok(Box::new(Typed::<M>(read(parameters)?)))
    }

    fn schema(&self) -> Schema {
        schema::<ParametersOf<M, f64>>()
    }

    fn foundry(&self) -> bool {
        <M::Device<f64> as NativeDevice<f64>>::FOUNDRY
    }
}

/// The parameters of a device of `M`, read.
struct Typed<M: NativeModel>(ParametersOf<M, f64>);

impl<M: NativeModel> fmt::Debug for Typed<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Typed").field(&self.0).finish()
    }
}

impl<M: NativeModel> Specification for Typed<M> {
    fn interface(&self) -> Interface {
        M::Device::<f64>::interface(&self.0)
    }

    fn instantiate(&self, binding: &mut Binding<'_>) -> Result<Box<dyn Device>, ParameterError> {
        Ok(Box::new(M::Device::<f64>::new(self.0.clone(), binding)?))
    }

    fn instantiate_gradient(
        &self,
        parameter: &str,
        binding: &mut Binding<'_>,
    ) -> Result<Box<dyn Device<Gradient>>, ParameterError> {
        let parameters = lift(&self.0, parameter)?;
        Ok(Box::new(M::Device::<Gradient>::new(parameters, binding)?))
    }

    fn parameters(&self) -> Result<Parameters, ParameterError> {
        written(&self.0)
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

/// The typed `parameters` read again as `Q`, whose devices compute with their derivatives
/// with respect to the number at the path `parameter`: written as JSON, where that number
/// becomes `{"value": v, "derivative": 1}`, and read back, which gives every other number
/// exactly as it was.
///
/// # Errors
///
/// Returns [`ParameterError`] when no number is at `parameter`, or `Q` does not read one
/// there as a real number, as when it reads it as a choice, such as a level.
pub(crate) fn lift<P: Serialize, Q: DeserializeOwned>(
    parameters: &P,
    parameter: &str,
) -> Result<Q, ParameterError> {
    let mut written = written(parameters)?;
    let leaf = jolty_ir::parameter_mut(&mut written, parameter)
        .filter(|leaf| leaf.is_number())
        .ok_or_else(|| ParameterError(format!("the parameters hold no number at `{parameter}`")))?;
    let value = leaf.take();
    *leaf = serde_json::json!({"value": value, "derivative": 1.0});
    serde_json::from_value(Value::Object(written))
        .map_err(|error| ParameterError(error.to_string()))
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
