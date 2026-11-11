use std::fmt;
use std::marker::PhantomData;

use jolty_ir::Parameters;
use serde::de::DeserializeOwned;

use crate::{Binding, Input, Output};

/// One placed device, bound to its unknowns.
pub trait Device: fmt::Debug + Send {
    /// Adds the device's contribution at `input` to `output`.
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_>);

    /// Appends the times in `[start, stop]` where the device's behavior is not smooth, so a
    /// transient analysis steps onto them exactly.
    fn breakpoints(&self, start: f64, stop: f64, times: &mut Vec<f64>) {
        let _ = (start, stop, times);
    }

    /// The device as an independent source, if it is one.
    fn as_source(&mut self) -> Option<&mut dyn Source> {
        None
    }
}

/// An independent source, whose level analyses can hold or excite.
pub trait Source {
    /// Holds the source at a constant `level`, as a DC sweep does, or releases it to its
    /// waveform with `None`.
    fn hold(&mut self, level: Option<f64>);

    /// Adds the derivative of the residual with respect to the source's level, one value per
    /// unknown, to `derivative`. A small-signal analysis excites the source through it.
    fn level_derivative(&self, derivative: &mut [f64]);
}

/// A kind of device: its name, its terminals, and how to make a device from parameters.
pub trait Model: Send + Sync {
    /// The name circuits refer to the model by, such as `resistor`.
    fn name(&self) -> &'static str;

    /// The names of the device's terminals, in the order [`Binding::terminal`] indexes them.
    fn terminals(&self) -> &'static [&'static str];

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
}

/// A device implemented in Rust, whose parameters deserialize into a typed struct.
pub trait NativeDevice: Device + Sized + 'static {
    /// The model's name.
    const NAME: &'static str;
    /// The model's terminals.
    const TERMINALS: &'static [&'static str];
    /// The parameters, deserialized from the instance's parameter map.
    type Parameters: DeserializeOwned;

    /// Checks `parameters` and makes the device bound through `binding`.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when a parameter is out of range.
    fn new(parameters: Self::Parameters, binding: &mut Binding<'_>)
    -> Result<Self, ParameterError>;
}

/// The [`Model`] of a [`NativeDevice`].
pub struct Native<D>(PhantomData<fn() -> D>);

impl<D: NativeDevice> Native<D> {
    /// The model of `D`.
    #[must_use]
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<D: NativeDevice> Default for Native<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D> fmt::Debug for Native<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Native")
    }
}

impl<D: NativeDevice> Model for Native<D> {
    fn name(&self) -> &'static str {
        D::NAME
    }

    fn terminals(&self) -> &'static [&'static str] {
        D::TERMINALS
    }

    fn instantiate(
        &self,
        parameters: &Parameters,
        binding: &mut Binding<'_>,
    ) -> Result<Box<dyn Device>, ParameterError> {
        let parameters = serde_json::from_value(serde_json::Value::Object(parameters.clone()))
            .map_err(|error| ParameterError(error.to_string()))?;
        Ok(Box::new(D::new(parameters, binding)?))
    }
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
    pub fn positive(parameter: &str, value: f64) -> Result<f64, Self> {
        if value > 0.0 {
            Ok(value)
        } else {
            Err(Self(format!("{parameter} must be positive, not {value:?}")))
        }
    }

    /// Checks that `value` is not negative.
    ///
    /// # Errors
    ///
    /// Returns an error naming `parameter` otherwise.
    pub fn non_negative(parameter: &str, value: f64) -> Result<f64, Self> {
        if value >= 0.0 {
            Ok(value)
        } else {
            Err(Self(format!(
                "{parameter} must be at least 0, not {value:?}"
            )))
        }
    }
}
