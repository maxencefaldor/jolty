use std::fmt;
use std::marker::PhantomData;

use jolty_ir::Parameters;
use serde::de::DeserializeOwned;

use crate::{Binding, Detect, Drive, Gradient, Input, Interface, Output, Real};

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
}

/// An independent source, whose level analyses can hold or excite.
pub trait Source {
    /// Holds the source at a constant `level`, as a DC sweep does, or releases it to its
    /// waveform with `None`.
    fn hold(&mut self, level: Option<f64>);

    /// The source's own level at the operating point: its waveform's at time zero.
    fn level(&self) -> f64;

    /// Adds the derivative of the residual with respect to the source's level, one value per
    /// unknown, to `derivative`. A small-signal analysis excites the source through it.
    fn level_derivative(&self, derivative: &mut [f64]);
}

/// A kind of device: its name, its terminals, and how to make a device from parameters.
pub trait Model: Send + Sync {
    /// The name circuits refer to the model by, such as `resistor`.
    fn name(&self) -> &'static str;

    /// The terminals and branches of a device with `parameters`.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when the parameters cannot be read.
    fn interface(&self, parameters: &Parameters) -> Result<Interface, ParameterError>;

    /// Reads `parameters` and makes a device bound through `binding`.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when a parameter is missing, unknown or out of range.
    fn instantiate(
        &self,
        parameters: &Parameters,
        binding: &mut Binding<'_>,
    ) -> Result<Box<dyn Device>, ParameterError>;

    /// Reads `parameters`, in which the number to differentiate against is written as
    /// `{"value": v, "derivative": 1}`, and makes a device bound through `binding` whose
    /// residuals and charges carry their exact derivatives with respect to it.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when the parameters cannot be read, as when the model
    /// reads the number differentiated against as a choice, such as a level.
    fn instantiate_gradient(
        &self,
        parameters: &Parameters,
        binding: &mut Binding<'_>,
    ) -> Result<Box<dyn Device<Gradient>>, ParameterError>;
}

/// A device implemented in Rust, whose parameters deserialize into a typed struct.
pub trait NativeDevice<R: Real = f64>: Device<R> + Sized + 'static {
    /// The model's name.
    const NAME: &'static str;
    /// The parameters, deserialized from the instance's parameter map.
    type Parameters: DeserializeOwned;

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

    fn interface(&self, parameters: &Parameters) -> Result<Interface, ParameterError> {
        Ok(M::Device::<f64>::interface(&read::<f64, M::Device<f64>>(
            parameters,
        )?))
    }

    fn instantiate(
        &self,
        parameters: &Parameters,
        binding: &mut Binding<'_>,
    ) -> Result<Box<dyn Device>, ParameterError> {
        let parameters = read::<f64, M::Device<f64>>(parameters)?;
        Ok(Box::new(M::Device::<f64>::new(parameters, binding)?))
    }

    fn instantiate_gradient(
        &self,
        parameters: &Parameters,
        binding: &mut Binding<'_>,
    ) -> Result<Box<dyn Device<Gradient>>, ParameterError> {
        let parameters = read::<Gradient, M::Device<Gradient>>(parameters)?;
        Ok(Box::new(M::Device::<Gradient>::new(parameters, binding)?))
    }
}

/// The typed parameters of `D` in `parameters`.
pub(crate) fn read<R: Real, D: NativeDevice<R>>(
    parameters: &Parameters,
) -> Result<D::Parameters, ParameterError> {
    serde_json::from_value(serde_json::Value::Object(parameters.clone()))
        .map_err(|error| ParameterError(error.to_string()))
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
