//! Self-heating: a device at the temperature of a thermal net, heated by the power it
//! dissipates.
//!
//! A [`Heatable`] device scales its parameters to any temperature, as it does to the
//! simulation's when it is made. [`Thermal`] gives such a model a thermal terminal,
//! `thermal`, when an instance's parameters say `"thermal": true`: the device is then at
//! the ambient temperature plus that net's, and its power, the sum over its nodes of each
//! voltage times the current into the device there, flows into the net as heat. Its
//! temperature is an unknown like any voltage, so Newton's iteration solves the circuit
//! and its heating together, with every derivative exact.

use std::borrow::Cow;

use jolty_ir::Parameters;
use jolty_numerics::{Entry, Pattern, PatternBuilder};

use crate::device::{lift, read};
use crate::{
    Binding, Choice, Contributions, Device, Gradient, Hyper, Input, Interface, Model, NativeDevice,
    Node, NoiseSource, Output, ParameterError, Quantity, Real, Specification, Unknown, UnknownInfo,
};

/// A device whose parameters scale to any temperature.
pub trait Heatable<R: Real>: NativeDevice<R> {
    /// Moves the device to `temperature`, in kelvins, scaling every parameter that depends
    /// on it from `parameters`, the ones it was made with, as it scales them when made.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when a parameter scales out of its range there.
    fn heat(&mut self, parameters: &Self::Parameters, temperature: R)
    -> Result<(), ParameterError>;
}

/// A model whose devices heat.
pub trait HeatableModel: 'static {
    /// The device, computing with `R`.
    type Device<R: Real>: Heatable<R>;
}

/// The parameters of the devices of `M` that compute with `R`.
type ParametersOf<M, R> = <<M as HeatableModel>::Device<R> as NativeDevice<R>>::Parameters;

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

/// The [`Model`] of a [`HeatableModel`] `M`, whose instances heat when their parameters
/// say `"thermal": true`.
pub struct Thermal<M>(std::marker::PhantomData<fn() -> M>);

impl<M: HeatableModel> Thermal<M> {
    /// The model of `M`.
    #[must_use]
    pub fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<M: HeatableModel> Default for Thermal<M> {
    fn default() -> Self {
        Self::new()
    }
}

/// Whether a device heats, as its parameters say.
#[derive(Debug, Clone, Copy)]
struct Heats {
    /// Whether the device heats by the power it dissipates, through a terminal `thermal`
    /// joined to a thermal net, whose temperature above the ambient it runs at.
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

impl<M: HeatableModel> Model for Thermal<M> {
    fn name(&self) -> &'static str {
        <M::Device<f64> as NativeDevice<f64>>::NAME
    }

    fn read(&self, parameters: &Parameters) -> Result<Box<dyn Specification>, ParameterError> {
        let (heats, others) = Heats::read(parameters)?;
        let warmed = if heats.thermal {
            Warmed::<M>::Heated(read(&others)?)
        } else {
            Warmed::<M>::Cool(read(&others)?)
        };
        Ok(Box::new(warmed))
    }
}

/// The parameters of a device of `M`, read as the numbers it computes with: plain, or,
/// when it heats, with their derivatives with respect to its temperature beside them.
enum Warmed<M: HeatableModel> {
    /// A device at the temperature of its surroundings.
    Cool(ParametersOf<M, f64>),
    /// A device that heats.
    Heated(ParametersOf<M, <f64 as Heating>::Warm>),
}

impl<M: HeatableModel> std::fmt::Debug for Warmed<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cool(parameters) => f.debug_tuple("Cool").field(parameters).finish(),
            Self::Heated(parameters) => f.debug_tuple("Heated").field(parameters).finish(),
        }
    }
}

impl<M: HeatableModel> Specification for Warmed<M> {
    fn interface(&self) -> Interface {
        match self {
            Self::Cool(parameters) => M::Device::<f64>::interface(parameters),
            Self::Heated(parameters) => {
                let mut interface = M::Device::<Gradient>::interface(parameters);
                interface.thermal.push(interface.terminals.len());
                interface.terminals.push("thermal".to_owned());
                interface
            }
        }
    }

    fn instantiate(&self, binding: &mut Binding<'_>) -> Result<Box<dyn Device>, ParameterError> {
        Ok(match self {
            Self::Cool(parameters) => Box::new(M::Device::<f64>::new(parameters.clone(), binding)?),
            Self::Heated(parameters) => {
                Box::new(Heated::<M, f64>::new(parameters.clone(), binding)?)
            }
        })
    }

    fn instantiate_gradient(
        &self,
        parameter: &str,
        binding: &mut Binding<'_>,
    ) -> Result<Box<dyn Device<Gradient>>, ParameterError> {
        Ok(match self {
            Self::Cool(parameters) => Box::new(M::Device::<Gradient>::new(
                lift(parameters, parameter)?,
                binding,
            )?),
            Self::Heated(parameters) => Box::new(Heated::<M, Gradient>::new(
                lift(parameters, parameter)?,
                binding,
            )?),
        })
    }
}

/// A device of `M` that heats, computing with `R`: the device itself, computing with
/// [`Heating::Warm`] numbers in a system of its own, whose unknowns are its terminals
/// and its internal unknowns, and whose equations and entries map onto the circuit's.
pub struct Heated<M: HeatableModel, R: Heating> {
    device: M::Device<R::Warm>,
    parameters: ParametersOf<M, R::Warm>,
    /// The device's own system: its pattern, and its contributions there.
    pattern: Pattern,
    contributions: Contributions<R::Warm>,
    /// The circuit's node or unknown of each of the device's own unknowns, and whether
    /// it is a voltage, whose current the device's power reads.
    unknowns: Vec<(Node, bool)>,
    /// The circuit's entry of each of the device's own stored positions.
    entries: Vec<Option<Entry>>,
    /// The thermal terminal's net.
    thermal: Node,
    /// The entries of each equation of the device at the thermal net, of the thermal net's
    /// equation at each unknown of the device, and at itself.
    thermal_column: Vec<Option<Entry>>,
    thermal_row: Vec<Option<Entry>>,
    thermal_diagonal: Option<Entry>,
    /// The temperature the device is at when the thermal net is: the simulation's.
    ambient: f64,
    /// The temperature of the last evaluation, from which the next step is limited.
    last: f64,
    /// The device's own unknowns' values at the last evaluation.
    values: Vec<f64>,
    /// The derivative of the device's power with respect to each of its unknowns.
    slopes: Vec<R>,
}

impl<M: HeatableModel, R: Heating> std::fmt::Debug for Heated<M, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Heated")
            .field("device", &self.device)
            .field("thermal", &self.thermal)
            .field("last", &self.last)
            .finish_non_exhaustive()
    }
}

impl<M: HeatableModel, R: Heating> Heated<M, R> {
    /// The device of `M` with `parameters`, read as the numbers it computes with when it
    /// heats, bound through `binding`, whose last terminal is the thermal one.
    fn new(
        parameters: ParametersOf<M, R::Warm>,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let ambient = binding.temperature();
        let terminals = binding.terminals().len() - 1;
        let thermal = binding.terminal(terminals);
        let own: Vec<Node> = (0..terminals)
            .map(|index| Some(Unknown::new(index)))
            .collect();
        let mut infos: Vec<UnknownInfo> = (0..terminals)
            .map(|index| UnknownInfo {
                name: format!("{index}"),
                quantity: Quantity::Voltage,
            })
            .collect();
        let mut builder = PatternBuilder::new();
        let path = binding.path().to_owned();
        let mut device = {
            let mut local = Binding::new(&path, &own, &mut infos, &mut builder, ambient);
            M::Device::<R::Warm>::new(parameters.clone(), &mut local)?
        };
        // Made at the ambient temperature as a constant; heated there, it carries every
        // derivative with respect to its temperature.
        device.heat(&parameters, R::seed(ambient))?;
        let pattern = builder.finish(infos.len());
        let prefix = format!("{path}:");
        let mut unknowns: Vec<(Node, bool)> = (0..terminals)
            .map(|index| (binding.terminal(index), true))
            .collect();
        for info in &infos[terminals..] {
            let label = info.name.strip_prefix(&prefix).unwrap_or(&info.name);
            let made = binding.internal(label, info.quantity);
            unknowns.push((Some(made), info.quantity == Quantity::Voltage));
        }
        let entries = pattern
            .positions()
            .map(|(row, column)| binding.entry(unknowns[row].0, unknowns[column].0))
            .collect();
        let thermal_column = unknowns
            .iter()
            .map(|&(row, _)| binding.entry(row, thermal))
            .collect();
        let thermal_row = unknowns
            .iter()
            .map(|&(column, _)| binding.entry(thermal, column))
            .collect();
        let thermal_diagonal = binding.entry(thermal, thermal);
        let count = unknowns.len();
        Ok(Self {
            device,
            parameters,
            contributions: Contributions::new(&pattern),
            pattern,
            unknowns,
            entries,
            thermal,
            thermal_column,
            thermal_row,
            thermal_diagonal,
            ambient,
            last: ambient,
            values: vec![0.0; count],
            slopes: vec![R::constant(0.0); count],
        })
    }

    /// The device's own unknowns' values at `input`.
    fn gather(&mut self, input: &Input<'_>) {
        for (value, &(node, _)) in self.values.iter_mut().zip(&self.unknowns) {
            *value = input.value(node);
        }
    }

    /// The device's temperature at `input`, and the one it is evaluated at: no further
    /// than [`MOST_TEMPERATURE_STEP`] from the last evaluation's when `input` limits its
    /// steps, and no lower than [`LOWEST_TEMPERATURE`].
    fn temperatures(&self, input: &Input<'_>) -> (f64, f64) {
        let applied = self.ambient + input.value(self.thermal);
        let stepped = if input.limiting {
            applied.clamp(
                self.last - MOST_TEMPERATURE_STEP,
                self.last + MOST_TEMPERATURE_STEP,
            )
        } else {
            applied
        };
        (applied, stepped.max(LOWEST_TEMPERATURE))
    }
}

impl<M: HeatableModel, R: Heating> Device<R> for Heated<M, R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        self.gather(input);
        let (applied, temperature) = self.temperatures(input);
        if temperature.to_bits() != self.last.to_bits() {
            if self
                .device
                .heat(&self.parameters, R::seed(temperature))
                .is_err()
            {
                // A parameter out of range here: the step went too far, and the last
                // temperature's scaling stands, the solution not yet converged.
                output.mark_limited();
            } else {
                self.last = temperature;
            }
        }
        if temperature.to_bits() != applied.to_bits() {
            output.mark_limited();
        }
        let beyond = applied - self.last;
        let local = input.with_values(&self.values);
        let mut own = self.contributions.output(&self.pattern);
        self.device.evaluate(&local, &mut own);
        if own.is_limited() {
            output.mark_limited();
        }
        let Contributions {
            residual,
            scale,
            charge,
            conductance,
            capacitance,
        } = &self.contributions;
        // The power, `Σ v · i` over the device's voltages, and its derivatives.
        let mut power = R::constant(0.0);
        let mut hotter_power = R::constant(0.0);
        self.slopes.fill(R::constant(0.0));
        for (index, &(node, voltage)) in self.unknowns.iter().enumerate() {
            let (current, hotter) = R::split(residual[index]);
            let current = current + hotter * beyond;
            let (held, hotter_held) = R::split(charge[index]);
            output.add_residual_scaled(node, current, R::split(scale[index]).0);
            output.add_charge(node, held + hotter_held * beyond);
            output.add_conductance(self.thermal_column[index], hotter);
            output.add_capacitance(self.thermal_column[index], hotter_held);
            if voltage {
                let value = self.values[index];
                power += current * value;
                hotter_power += hotter * value;
                self.slopes[index] += current;
            }
        }
        for (((row, column), entry), (&slope, &stored)) in self
            .pattern
            .positions()
            .zip(&self.entries)
            .zip(conductance.iter().zip(capacitance))
        {
            let (slope, _) = R::split(slope);
            output.add_conductance(*entry, slope);
            output.add_capacitance(*entry, R::split(stored).0);
            if self.unknowns[row].1 {
                self.slopes[column] += slope * self.values[row];
            }
        }
        output.add_residual(self.thermal, -power);
        output.add_conductance(self.thermal_diagonal, -hotter_power);
        for (entry, &slope) in self.thermal_row.iter().zip(&self.slopes) {
            output.add_conductance(*entry, -slope);
        }
    }

    fn breakpoints(&self, start: f64, stop: f64, times: &mut Vec<f64>) {
        self.device.breakpoints(start, stop, times);
    }

    fn choices(&mut self, input: &Input<'_>, found: &mut Vec<Choice>) {
        self.gather(input);
        let local = input.with_values(&self.values);
        self.device.choices(&local, found);
    }

    fn variable_names(&self) -> &'static [&'static str] {
        self.device.variable_names()
    }

    fn variables(&self, values: &mut [f64]) {
        self.device.variables(values);
    }

    /// The device's noise at its temperature there, each source on the circuit's
    /// equations.
    fn noise(&self, input: &Input<'_>, sources: &mut Vec<NoiseSource>) {
        let values: Vec<f64> = self
            .unknowns
            .iter()
            .map(|&(node, _)| input.value(node))
            .collect();
        let local = input.with_values(&values);
        let start = sources.len();
        self.device.noise(&local, sources);
        for source in &mut sources[start..] {
            source.equations = source
                .equations
                .map(|node| node.and_then(|unknown| self.unknowns[unknown.index()].0));
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
        let mut heated = Bench::of(&model, DIODE, REFERENCE_TEMPERATURE).expect("valid");
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
