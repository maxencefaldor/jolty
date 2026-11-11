//! Device models for jolty: what each kind of device contributes to a circuit's equations.
//!
//! A circuit is solved in charge-oriented form, `f(x) + d/dt q(x) = 0`, where `x` holds the
//! node voltages and branch currents. Each [`Device`] adds its share of the resistive part
//! `f`, currents and constraint residuals, and of the reactive part `q`, charges and fluxes,
//! with their derivatives `G = ∂f/∂x` and `C = ∂q/∂x`. Compiled Verilog-A models use the same
//! form, so native and foundry models plug in alike.
//!
//! A [`Model`] is a kind of device: it names its terminals and turns parameters into a
//! device bound to its unknowns. A [`Library`] holds the models a circuit can use.

mod binding;
mod bjt;
mod controlled;
mod device;
mod diode;
mod evaluation;
mod jfet;
mod junction;
mod passive;
mod sources;
mod stamp;
#[cfg(test)]
mod testing;
mod topology;
pub mod waveform;

use std::collections::BTreeMap;

pub use binding::{Binding, Node, Quantity, TerminalPair, Unknown, UnknownInfo};
pub use bjt::{BipolarTransistor, BipolarTransistorParameters, Polarity};
pub use controlled::{
    CurrentControlledCurrentSource, CurrentControlledCurrentSourceParameters,
    CurrentControlledVoltageSource, CurrentControlledVoltageSourceParameters,
    VoltageControlledCurrentSource, VoltageControlledCurrentSourceParameters,
    VoltageControlledVoltageSource, VoltageControlledVoltageSourceParameters,
};
pub use device::{Device, Model, Native, NativeDevice, ParameterError, Source};
pub use diode::{Diode, DiodeParameters};
pub use evaluation::{Input, Output};
pub use jfet::{Channel, Jfet, JfetParameters};
pub use passive::{
    Capacitor, CapacitorParameters, Inductor, InductorParameters, Resistor, ResistorParameters,
};
pub use sources::{CurrentSource, SourceParameters, VoltageSource};
pub use topology::{Branch, BranchKind};

/// The models a circuit can use, by name.
#[derive(Default)]
pub struct Library {
    models: BTreeMap<&'static str, Box<dyn Model>>,
}

impl Library {
    /// A library with every native model.
    #[must_use]
    pub fn standard() -> Self {
        let mut library = Self::default();
        library.add(Native::<BipolarTransistor>::new());
        library.add(Native::<Resistor>::new());
        library.add(Native::<Capacitor>::new());
        library.add(Native::<Inductor>::new());
        library.add(Native::<VoltageSource>::new());
        library.add(Native::<CurrentSource>::new());
        library.add(Native::<CurrentControlledCurrentSource>::new());
        library.add(Native::<CurrentControlledVoltageSource>::new());
        library.add(Native::<VoltageControlledVoltageSource>::new());
        library.add(Native::<VoltageControlledCurrentSource>::new());
        library.add(Native::<Diode>::new());
        library.add(Native::<Jfet>::new());
        library
    }

    /// Adds `model`, replacing any model of the same name.
    pub fn add(&mut self, model: impl Model + 'static) {
        self.models.insert(model.name(), Box::new(model));
    }

    /// The model named `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&dyn Model> {
        self.models.get(name).map(Box::as_ref)
    }

    /// The names of every model, in alphabetical order.
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.models.keys().copied()
    }
}

impl std::fmt::Debug for Library {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.names()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_standard_library_holds_every_native_model() {
        let names: Vec<_> = Library::standard().names().collect();
        assert_eq!(
            names,
            [
                "bipolar_transistor",
                "capacitor",
                "current_controlled_current_source",
                "current_controlled_voltage_source",
                "current_source",
                "diode",
                "inductor",
                "jfet",
                "resistor",
                "voltage_controlled_current_source",
                "voltage_controlled_voltage_source",
                "voltage_source"
            ]
        );
    }
}
