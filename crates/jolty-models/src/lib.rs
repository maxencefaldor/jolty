//! Device models for jolty: what each kind of device contributes to a circuit's equations.
//!
//! A circuit is solved in charge-oriented form, `f(x) + d/dt q(x) = 0`, where `x` holds the
//! node voltages and branch currents. Each [`Device`] adds its share of the resistive part
//! `f`, currents and constraint residuals, and of the reactive part `q`, charges and fluxes,
//! with their derivatives `G = ∂f/∂x` and `C = ∂q/∂x`. Compiled Verilog-A models use the same
//! form, so native and foundry models plug in alike.
//!
//! A [`Model`] is a kind of device: it names its terminals and turns parameters into a
//! device bound to its unknowns. A [`LogicModel`] is a kind of [`Logic`] device, whose pins
//! join logic nets that an event scheduler simulates. A [`Library`] holds the models a
//! circuit can use.

mod behavioral;
mod binding;
mod bjt;
mod controlled;
mod device;
mod diode;
mod dual;
mod evaluation;
mod jfet;
mod junction;
mod logic;
mod mosfet;
mod passive;
mod real;
mod sources;
mod stamp;
mod switch;
#[cfg(test)]
mod testing;
mod topology;
pub mod waveform;

use std::collections::BTreeMap;
use std::sync::Arc;

pub use behavioral::{
    Behavior, BehavioralCurrentSource, BehavioralParameters, BehavioralVoltageSource,
};
pub use binding::{Binding, Node, Quantity, TerminalPair, Unknown, UnknownInfo};
pub use bjt::{BipolarTransistor, BipolarTransistorParameters, Polarity};
pub use controlled::{
    CurrentControlledCurrentSource, CurrentControlledCurrentSourceParameters,
    CurrentControlledVoltageSource, CurrentControlledVoltageSourceParameters,
    VoltageControlledCurrentSource, VoltageControlledCurrentSourceParameters,
    VoltageControlledVoltageSource, VoltageControlledVoltageSourceParameters,
};
pub use device::{Device, Model, Native, NativeDevice, NativeModel, ParameterError, Plain, Source};
pub use diode::{Diode, DiodeParameters};
pub use evaluation::{Input, Output};
pub use jfet::{Channel, Jfet, JfetParameters};
pub use junction::{BOLTZMANN, ELEMENTARY_CHARGE, REFERENCE_TEMPERATURE, thermal_voltage};
pub use logic::{
    Change, Detect, Domain, Drive, Electrical, ElectricalParameters, FixedTiming, Level, Logic,
    LogicCell, LogicCellParameters, LogicDriver, LogicDriverParameters, LogicModel, LogicSource,
    LogicSourceParameters, MAX_INPUTS, NativeLogic, NativeLogicModel, OutputParameters, Pins,
    Stimulus, ThresholdDetector, ThresholdDetectorParameters, Timing, TimingParameters,
};
pub use mosfet::{GateMaterial, Mosfet, MosfetParameters};
pub use passive::{
    Capacitor, CapacitorParameters, Inductor, InductorParameters, Resistor, ResistorParameters,
};
pub use real::{Gradient, Real};
pub use sources::{CurrentSource, SourceParameters, VoltageSource};
pub use switch::{VoltageControlledSwitch, VoltageControlledSwitchParameters};
pub use topology::{Branch, BranchKind, Interface};

/// The models a circuit can use, by name: analog models and logic models, no two of the
/// same name.
#[derive(Default)]
pub struct Library {
    models: BTreeMap<&'static str, Arc<dyn Model>>,
    logic: BTreeMap<&'static str, Box<dyn LogicModel>>,
}

impl Library {
    /// A library with every native model.
    #[must_use]
    pub fn standard() -> Self {
        let mut library = Self::default();
        library.add(Native::<BehavioralCurrentSource>::new());
        library.add(Native::<BehavioralVoltageSource>::new());
        library.add(Plain::<BipolarTransistor>::new());
        library.add(Native::<Resistor>::new());
        library.add(Native::<Capacitor>::new());
        library.add(Native::<Inductor>::new());
        library.add(Native::<VoltageSource>::new());
        library.add(Native::<CurrentSource>::new());
        library.add(Native::<CurrentControlledCurrentSource>::new());
        library.add(Native::<CurrentControlledVoltageSource>::new());
        library.add(Native::<VoltageControlledVoltageSource>::new());
        library.add(Native::<VoltageControlledCurrentSource>::new());
        library.add(Native::<VoltageControlledSwitch>::new());
        library.add(Native::<Diode>::new());
        library.add(Native::<Jfet>::new());
        library.add(Plain::<Mosfet>::new());
        library.add_logic(NativeLogicModel::<LogicCell>::new());
        library.add_logic(NativeLogicModel::<LogicSource>::new());
        library
    }

    /// Adds `model`, replacing any model of the same name.
    pub fn add(&mut self, model: impl Model + 'static) {
        self.logic.remove(model.name());
        self.models.insert(model.name(), Arc::new(model));
    }

    /// Adds the logic model `model`, replacing any model of the same name.
    pub fn add_logic(&mut self, model: impl LogicModel + 'static) {
        self.models.remove(model.name());
        self.logic.insert(model.name(), Box::new(model));
    }

    /// The analog model named `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&dyn Model> {
        self.models.get(name).map(AsRef::as_ref)
    }

    /// The analog model named `name`, shared, for a device to keep and be made again from.
    #[must_use]
    pub fn shared(&self, name: &str) -> Option<Arc<dyn Model>> {
        self.models.get(name).cloned()
    }

    /// The logic model named `name`.
    #[must_use]
    pub fn logic(&self, name: &str) -> Option<&dyn LogicModel> {
        self.logic.get(name).map(Box::as_ref)
    }

    /// The names of every model, analog and logic, in alphabetical order.
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        let mut names: Vec<&'static str> = self
            .models
            .keys()
            .chain(self.logic.keys())
            .copied()
            .collect();
        names.sort_unstable();
        names.into_iter()
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
                "behavioral_current_source",
                "behavioral_voltage_source",
                "bipolar_transistor",
                "capacitor",
                "current_controlled_current_source",
                "current_controlled_voltage_source",
                "current_source",
                "diode",
                "inductor",
                "jfet",
                "logic_cell",
                "logic_source",
                "mosfet",
                "resistor",
                "voltage_controlled_current_source",
                "voltage_controlled_switch",
                "voltage_controlled_voltage_source",
                "voltage_source"
            ]
        );
    }
}
