//! Device models for jolty: what each kind of device contributes to a circuit's equations.
//!
//! A circuit is solved in charge-oriented form, `f(x) + d/dt q(x) = 0`, where `x` holds the
//! node voltages and branch currents. Each device adds its share of the resistive part `f`,
//! currents and constraint residuals, and of the reactive part `q`, charges and fluxes, with
//! their derivatives `G = ∂f/∂x` and `C = ∂q/∂x`. Compiled Verilog-A models use the same
//! form, so native and foundry models plug in alike.
//!
//! A [`Kernel`] computes a model one instance at a time, as a pure function of the
//! instance's prepared parameters, its memory and its local unknowns, which allocates
//! nothing and never panics. [`Groups`] evaluate the instances of one kernel and one shape
//! together, on one thread or across a rayon pool's, each into its own block of a slab that
//! a gather adds to the circuit's equations in a fixed order; a [`Twin`] evaluates one
//! instance with the derivative of every number with respect to one of its parameters.
//!
//! A [`Model`] is a kind of device: it reads an instance's parameters, once, into a
//! [`Specification`], which names the device's terminals and gives the instance of its
//! kernel. A [`LogicModel`] is a kind of [`Logic`] device, whose
//! pins join logic nets that an event scheduler simulates. Every model's parameters write
//! as they read, with each one left out at its default, and a JSON schema describes them.
//! A [`Library`] holds the models a circuit can use.

mod binding;
mod device;
mod evaluation;
mod group;
mod kernel;
mod logic;
mod noise;
mod past;
mod surroundings;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
mod topology;
pub mod waveform;

use std::collections::BTreeMap;
use std::sync::Arc;

pub use binding::{Binding, Nature, Node, Unknown, UnknownIndex};
pub use device::{Differentiable, Model, ParameterError, Specification};
pub use evaluation::{Contributions, Input};
pub use group::{Alone, Block, Changed, Groups, Instance, Runs, Tasks, TooLarge, Twin};
pub use jolty_expressions::Choice;
pub use kernel::behavioral::{
    Behavior, BehavioralCurrentSource, BehavioralParameters, BehavioralVoltageSource,
};
pub use kernel::bjt::{BipolarTransistor, BipolarTransistorParameters, Polarity};
pub use kernel::bsim3::{Bsim3, Bsim3Parameters, Charges, Mobility, NoiseModel};
pub use kernel::bsim4::{
    BodyResistance, Bsim4, Bsim4Parameters, DrainSourceResistance, FlickerNoise, GateResistance,
    Perimeter, ThermalNoise,
};
pub use kernel::connect::{
    LogicDriver, LogicDriverParameters, Ramp, ThresholdDetector, ThresholdDetectorParameters,
};
pub use kernel::controlled::{
    CurrentControlledCurrentSource, CurrentControlledCurrentSourceParameters,
    CurrentControlledVoltageSource, CurrentControlledVoltageSourceParameters,
    VoltageControlledCurrentSource, VoltageControlledCurrentSourceParameters,
    VoltageControlledVoltageSource, VoltageControlledVoltageSourceParameters,
};
pub use kernel::diode::{Diode, DiodeParameters};
pub use kernel::dual::Dual;
pub use kernel::foundry::{FoundryMosfet, FoundryMosfetParameters};
pub use kernel::heat::{Heating, Thermal};
pub use kernel::jfet::{Channel, Jfet, JfetParameters};
pub use kernel::junction::{
    BOLTZMANN, ELEMENTARY_CHARGE, REFERENCE_TEMPERATURE, limit_gate, limit_junction,
    thermal_voltage,
};
pub use kernel::memristor::{
    CurrentControlledMemristor, CurrentControlledMemristorParameters, MemristorLaw,
    VoltageControlledMemristor, VoltageControlledMemristorParameters,
};
pub use kernel::mosfet::{GateMaterial, Mosfet, MosfetParameters};
pub use kernel::parts::SMALL_RESISTANCE;
pub use kernel::passive::{
    Capacitor, CapacitorParameters, CoupledInductors, CoupledInductorsParameters, Coupling,
    Inductor, InductorParameters, Resistor, ResistorParameters,
};
pub use kernel::photocell::{Photocell, PhotocellParameters};
pub use kernel::real::{Gradient, Hyper, Real};
pub use kernel::region::{REGION, Regions};
pub use kernel::sources::{CurrentSource, SourceParameters, VoltageSource, held_source};
pub use kernel::switch::{
    CurrentControlledSwitch, CurrentControlledSwitchParameters, VoltageControlledSwitch,
    VoltageControlledSwitchParameters,
};
pub use kernel::thermal::{
    HeatSource, HeatSourceParameters, ThermalCapacitance, ThermalCapacitanceParameters,
    ThermalResistance, ThermalResistanceParameters,
};
pub use kernel::transmission::{TransmissionLine, TransmissionLineParameters};
pub use kernel::vbic::{BaseCharge, Vbic, VbicParameters};
pub use kernel::vdmos::{Vdmos, VdmosParameters};
pub use kernel::{Kernel, Layout, Local, Stamp};
pub use logic::{
    Change, Domain, Electrical, ElectricalParameters, FixedTiming, Logic, LogicCell,
    LogicCellParameters, LogicLevel, LogicModel, LogicSource, LogicSourceParameters, MAX_INPUTS,
    NativeLogic, NativeLogicModel, OutputParameters, Pins, Stimulus, Timing, TimingParameters,
};
pub use noise::NoiseSource;
pub use past::{Delayed, DelayedEntry, Past};
pub use schemars::Schema;
pub use surroundings::Surrounded;
pub use topology::{Branch, BranchKind, Interface};

/// The models a circuit can use, by name: analog models and logic models, no two of the
/// same name.
///
/// Each model is shared, so that what is elaborated from a library can keep the models it
/// was made with, to read its instances' parameters again.
#[derive(Default)]
pub struct Library {
    models: BTreeMap<&'static str, Arc<dyn Model>>,
    logic: BTreeMap<&'static str, Arc<dyn LogicModel>>,
}

impl Library {
    /// A library with every native model.
    #[must_use]
    pub fn standard() -> Self {
        let mut library = Self::default();
        library.add(Differentiable::<BehavioralCurrentSource>::new());
        library.add(Differentiable::<BehavioralVoltageSource>::new());
        library.add(Thermal::<BipolarTransistor>::new());
        library.add(Thermal::<Bsim3>::new());
        library.add(Differentiable::<Bsim4>::new());
        library.add(Thermal::<Resistor>::new());
        library.add(Differentiable::<Capacitor>::new());
        library.add(Differentiable::<Inductor>::new());
        library.add(Differentiable::<CoupledInductors>::new());
        library.add(Differentiable::<VoltageSource>::new());
        library.add(Differentiable::<CurrentSource>::new());
        library.add(Differentiable::<CurrentControlledCurrentSource>::new());
        library.add(Differentiable::<CurrentControlledVoltageSource>::new());
        library.add(Differentiable::<VoltageControlledVoltageSource>::new());
        library.add(Differentiable::<VoltageControlledCurrentSource>::new());
        library.add(Differentiable::<VoltageControlledSwitch>::new());
        library.add(Differentiable::<FoundryMosfet>::new());
        library.add(Differentiable::<CurrentControlledSwitch>::new());
        library.add(Thermal::<Diode>::new());
        library.add(Thermal::<Jfet>::new());
        library.add(Thermal::<Mosfet>::new());
        library.add(Thermal::<Vdmos>::new());
        library.add(Thermal::<Vbic>::new());
        library.add(Differentiable::<Photocell>::new());
        library.add(Differentiable::<VoltageControlledMemristor>::new());
        library.add(Differentiable::<CurrentControlledMemristor>::new());
        library.add(Differentiable::<ThermalResistance>::new());
        library.add(Differentiable::<ThermalCapacitance>::new());
        library.add(Differentiable::<HeatSource>::new());
        library.add(Differentiable::<TransmissionLine>::new());
        library.add_logic(NativeLogicModel::<LogicCell>::new());
        library.add_logic(NativeLogicModel::<LogicSource>::new());
        library
    }

    /// Adds `model`, replacing any model of the same name; each of its devices may run at
    /// a temperature of its own, as [`Surrounded`] describes.
    pub fn add(&mut self, model: impl Model + 'static) {
        self.logic.remove(model.name());
        self.models
            .insert(model.name(), Arc::new(Surrounded::new(model)));
    }

    /// Adds the logic model `model`, replacing any model of the same name.
    pub fn add_logic(&mut self, model: impl LogicModel + 'static) {
        self.models.remove(model.name());
        self.logic.insert(model.name(), Arc::new(model));
    }

    /// The analog model named `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Model>> {
        self.models.get(name)
    }

    /// The logic model named `name`.
    #[must_use]
    pub fn logic(&self, name: &str) -> Option<&Arc<dyn LogicModel>> {
        self.logic.get(name)
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
    use jolty_ir::{Parameters, numbers, parameter};
    use serde_json::{Value, json};

    use super::*;

    /// Devices of every model, between them giving each kind of parameter there is: every
    /// shape of waveform, a device's own temperature and its heating, and a logic cell's
    /// electrical pins.
    #[expect(clippy::too_many_lines, reason = "one example of each model, or two")]
    fn examples() -> Vec<(&'static str, Value)> {
        let mosfet = json!({"level": 3, "channel": "p", "oxide_thickness": 1e-8,
            "gate_material": "aluminum", "temperature": 350});
        let vdmos = json!({"channel": "p", "drain_resistance": 0.02, "breakdown_voltage": 200,
            "quasi_saturation_resistance": 0.5, "quasi_saturation_voltage": 100,
            "thermal": true, "temperature_offset": 10});
        let memristor = json!({"on_resistance": 1e3, "off_resistance": 1e5,
            "on_voltage": -1, "off_voltage": 0.5, "on_rate": 1e6, "off_rate": 1e5,
            "law": "exponential", "state": 0.5});
        let pulse = json!({"pulse": {"initial": 0, "pulsed": 5, "rise": 1e-9, "fall": 1e-9,
            "width": 1e-6, "period": 1e-5}});
        let cell = json!({"inputs": ["a", "b"], "outputs": [{"name": "y", "table": "1110"}],
            "timing": {"fixed": {"rise": 1e-9, "fall": 2e-9, "transition": 1e-10}},
            "electrical": {"input_capacitances": [1e-15, 2e-15], "supply": "vdd"}});
        let characterized = json!({"inputs": ["a"], "outputs": [{"name": "y", "table": "10"}],
            "timing": {"characterized": {"slews": [1e-10], "loads": [1e-15],
                "arcs": [{"input": "a", "before": "0", "output": "y",
                    "delay": [[1e-9]], "transition": [[1e-10]]}]}}});
        vec![
            (
                "behavioral_current_source",
                json!({"expression": "1e-3 * v(a, b) + i(p)"}),
            ),
            (
                "behavioral_voltage_source",
                json!({"expression": "2 * sin(time)"}),
            ),
            (
                "bipolar_transistor",
                json!({"polarity": "pnp", "forward_early_voltage": 50, "substrate": "lateral",
                    "substrate_saturation_current": 1e-15, "thermal": true}),
            ),
            (
                "bsim3",
                json!({"channel": "p", "vth0": -0.6, "lvth0": 1e-8, "capacitance_model": 2,
                    "mobility_model": 3, "noise_model": 2, "check_parameters": true, "charge_partition": 0.5,
                    "temperature": 350}),
            ),
            (
                "bsim4",
                json!({"channel": "p", "vth0": -0.5, "lvth0": 1e-8, "body_resistance_model": 1,
                    "perimeter_model": 0, "check_parameters": false, "charge_partition": 0.4,
                    "temperature": 350}),
            ),
            (
                "capacitor",
                json!({"capacitance": 1e-6, "voltage_coefficient": 0.01}),
            ),
            (
                "coupled_inductors",
                json!({"windings": [{"inductance": 1e-3}, {"inductance": 2e-3}],
                    "couplings": [{"windings": [1, 2], "coefficient": 0.9}]}),
            ),
            ("current_controlled_current_source", json!({"gain": 2})),
            (
                "current_controlled_memristor",
                json!({"on_resistance": 100, "off_resistance": 16e3, "charge": 2e-6}),
            ),
            ("current_controlled_switch", json!({"on_current": 2e-3})),
            (
                "current_controlled_voltage_source",
                json!({"transresistance": 1e3}),
            ),
            ("current_source", json!({"waveform": pulse})),
            (
                "current_source",
                json!({"waveform": {"piecewise": {"points": [[0, 0], [1e-3, 2]]}}}),
            ),
            (
                "diode",
                json!({"breakdown_voltage": 5.1, "temperature_offset": -20}),
            ),
            (
                "foundry_mosfet",
                json!({"pdk": "sky130", "model": "sky130_fd_pr__nfet_01v8", "width": 1e-6,
                    "length": 1.5e-7}),
            ),
            ("heat_source", json!({"power": 0.5})),
            ("inductor", json!({"inductance": 1e-3})),
            ("jfet", json!({"channel": "p", "threshold_voltage": -1.5})),
            ("logic_cell", cell),
            ("logic_cell", characterized),
            (
                "logic_source",
                json!({"initial": "0", "changes": [[1e-6, "1"], [2e-6, "x"]],
                    "transition": 1e-9}),
            ),
            ("mosfet", mosfet),
            (
                "photocell",
                json!({"resistance": 1e4, "response_time": 0.02}),
            ),
            (
                "resistor",
                json!({"resistance": 1e3, "exponential_temperature_coefficient": 0.5}),
            ),
            ("thermal_capacitance", json!({"capacitance": 2})),
            ("thermal_resistance", json!({"resistance": 10})),
            (
                "transmission_line",
                json!({"characteristic_impedance": 50, "delay": 1e-9}),
            ),
            ("vbic", json!({"base_charge": "gummel_poon"})),
            ("vdmos", vdmos),
            (
                "voltage_controlled_current_source",
                json!({"transconductance": 1e-3}),
            ),
            ("voltage_controlled_memristor", memristor),
            ("voltage_controlled_switch", json!({})),
            ("voltage_controlled_voltage_source", json!({"gain": 10})),
            (
                "voltage_source",
                json!({"waveform": {"sine": {"offset": 1, "amplitude": 2, "frequency": 1e3}}}),
            ),
            ("voltage_source", json!({"waveform": {"dc": 5}})),
        ]
    }

    /// `parameters` of a device of the model `name` of the standard library, each one
    /// left out written with its default.
    fn defaulted(library: &Library, name: &str, parameters: &Parameters) -> Parameters {
        match library.logic(name) {
            Some(model) => model.defaulted(parameters),
            None => library.get(name).expect("a model").defaulted(parameters),
        }
        .unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// The schema of the parameters of the model `name` of the standard library.
    fn schema(library: &Library, name: &str) -> Schema {
        match library.logic(name) {
            Some(model) => model.schema(),
            None => library.get(name).expect("a model").schema(),
        }
    }

    fn object(value: &Value) -> Parameters {
        value.as_object().cloned().expect("an object")
    }

    /// Every internal unknown a model's layout claims pivoted has a diagonal entry at least
    /// as large as every other entry of its column, in `G`, `G + αC` and `G + jωC`, at
    /// every point evaluated, at an operating point and in a transient: the property that
    /// lets the sparse solver eliminate it without a search. The points spread each
    /// unknown over ±10, and `α` and `ω` over 0 to 10¹⁵ per second, beyond any step's.
    #[test]
    fn every_pivoted_unknowns_diagonal_is_its_columns_largest() {
        let library = Library::standard();
        let frequencies: Vec<(f64, f64)> = [0.0, 1e3, 1e9, 1e15]
            .into_iter()
            .flat_map(|scale| [(scale, 0.0), (0.0, scale)])
            .collect();
        let mut checked = 0;
        for (name, parameters) in examples() {
            let Some(model) = library.get(name) else {
                continue;
            };
            for stepping in [false, true] {
                let bench = testing::Bench::from_model(
                    model.as_ref(),
                    &parameters.to_string(),
                    REFERENCE_TEMPERATURE,
                )
                .unwrap_or_else(|error| panic!("{name}: {error}"));
                let mut bench = if stepping { bench.stepping() } else { bench };
                let pivoted = bench.pivoted().to_vec();
                let unknowns = u32::try_from(bench.unknowns().len()).expect("few unknowns");
                for point in 0..64_u32 {
                    // Each unknown at its own fraction of ±10, a different one at each point.
                    let x: Vec<f64> = (0..unknowns)
                        .map(|index| 20.0 * f64::from((point * 7 + index * 13) % 41) / 40.0 - 10.0)
                        .collect();
                    let evaluation = bench.evaluate(&x);
                    let (conductance, capacitance) =
                        (&evaluation.conductance, &evaluation.capacitance);
                    for &pivot in &pivoted {
                        for &(real, imaginary) in &frequencies {
                            let entry = |row: usize| {
                                let (g, c) = (conductance[row][pivot], capacitance[row][pivot]);
                                (g + real * c).hypot(imaginary * c)
                            };
                            let diagonal = entry(pivot);
                            for row in (0..x.len()).filter(|&row| row != pivot) {
                                assert!(
                                    entry(row) <= diagonal,
                                    "{name}'s unknown {pivot} at {x:?}, s = {real} + {imaginary}j: \
                                     {} against its diagonal's {diagonal}",
                                    entry(row)
                                );
                            }
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 0, "no model claims a pivoted unknown");
    }

    #[test]
    fn every_model_has_an_example() {
        let mut named: Vec<_> = examples().into_iter().map(|(name, _)| name).collect();
        named.dedup();
        assert_eq!(named, Library::standard().names().collect::<Vec<_>>());
    }

    #[test]
    fn every_models_defaults_read_back_as_they_are() {
        let library = Library::standard();
        for (name, given) in examples() {
            let given = object(&given);
            let written = defaulted(&library, name, &given);
            assert_eq!(defaulted(&library, name, &written), written, "{name}");
            // Each number the circuit gives is where it was, so a sensitivity finds it.
            for (path, value) in numbers(&given) {
                let found = parameter(&written, &path).and_then(Value::as_f64);
                assert_eq!(found, Some(value), "{name}'s {path}");
            }
        }
    }

    #[test]
    fn every_models_schema_lists_each_parameter_with_its_default() {
        let library = Library::standard();
        for (name, given) in examples() {
            let given = object(&given);
            let schema = schema(&library, name);
            let properties = schema
                .get("properties")
                .and_then(Value::as_object)
                .expect("properties");
            let required: Vec<&str> = schema
                .get("required")
                .and_then(Value::as_array)
                .map_or_else(Vec::new, |names| {
                    names.iter().filter_map(Value::as_str).collect()
                });
            let written = defaulted(&library, name, &given);
            // A device's own temperature is written only where it has one.
            let own = ["temperature", "temperature_offset"];
            for key in written.keys().chain(properties.keys()) {
                assert!(
                    properties.contains_key(key)
                        && (written.contains_key(key) || own.contains(&key.as_str())),
                    "{name}'s {key}"
                );
            }
            for (key, property) in properties {
                assert!(
                    property.get("description").is_some(),
                    "{name}'s {key} is described"
                );
                // Each parameter is required, or has a default, but a device's own
                // temperature, which is the simulation's without one.
                let default = property.get("default");
                let required = required.contains(&key.as_str());
                assert!(
                    required != default.is_some() || own.contains(&key.as_str()),
                    "{name}'s {key} is required or has a default"
                );
                if let Some(default) = default.filter(|_| !given.contains_key(key)) {
                    assert_eq!(written.get(key), Some(default), "{name}'s {key}");
                }
            }
        }
    }

    #[test]
    fn the_standard_library_holds_every_native_model() {
        let names: Vec<_> = Library::standard().names().collect();
        assert_eq!(
            names,
            [
                "behavioral_current_source",
                "behavioral_voltage_source",
                "bipolar_transistor",
                "bsim3",
                "bsim4",
                "capacitor",
                "coupled_inductors",
                "current_controlled_current_source",
                "current_controlled_memristor",
                "current_controlled_switch",
                "current_controlled_voltage_source",
                "current_source",
                "diode",
                "foundry_mosfet",
                "heat_source",
                "inductor",
                "jfet",
                "logic_cell",
                "logic_source",
                "mosfet",
                "photocell",
                "resistor",
                "thermal_capacitance",
                "thermal_resistance",
                "transmission_line",
                "vbic",
                "vdmos",
                "voltage_controlled_current_source",
                "voltage_controlled_memristor",
                "voltage_controlled_switch",
                "voltage_controlled_voltage_source",
                "voltage_source"
            ]
        );
    }
}
