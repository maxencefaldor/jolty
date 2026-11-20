//! Device models for jolty defining each device's contributions to circuit equations.
//!
//! Circuits are formulated in charge-oriented form, `f(x) + d/dt q(x) = 0`, where `x` holds
//! node voltages and branch currents. Each device contributes resistive terms `f` (currents
//! and constraint residuals), reactive terms `q` (charges and fluxes), and their exact
//! derivatives `G = ∂f/∂x` and `C = ∂q/∂x`. Compiled Verilog-A models use the same interface
//! so native and foundry models interoperate seamlessly.
//!
//! A [`Kernel`] evaluates one device instance as a pure, allocation-free, panic-free function
//! of prepared parameters, state memory, and local unknowns. [`Groups`] batch instances of a
//! shared kernel and shape across one thread or a rayon pool, writing into dedicated slab
//! blocks before a deterministic gather sums them into the global system. A [`Twin`]
//! evaluates an instance with forward-mode derivatives along parameter [`Seeds`].
//!
//! A [`Model`] parses instance parameters into a [`Specification`] describing terminals and
//! exposing a kernel instance. A [`LogicModel`] instantiates event-driven [`Logic`] devices
//! connected to logic nets. Every model serializes parameters with omitted fields filled from
//! defaults and exposes a JSON schema; a [`Library`] stores all available models by name.

// The derive of `Lift` names this crate by its name, here as in every crate that uses it.
extern crate self as jolty_models;

mod binding;
mod device;
pub mod dimension;
mod evaluation;
mod group;
mod kernel;
mod lift;
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
pub use dimension::{Dimension, ParameterKind, ParameterMetadata};
pub use evaluation::{Contributions, Input};
pub use group::{
    Alone, Block, Changed, Groups, Instance, Runs, Seeds, Tasks, TooLarge, Twin, TwinNumber,
};
pub use jolty_expressions::Choice;
pub use kernel::behavioral::{
    Behavior, BehavioralCapacitor, BehavioralCurrentSource, BehavioralParameters,
    BehavioralVoltageSource,
};
pub use kernel::bjt::{
    BipolarCapacitanceTemperatureModel, BipolarTemperatureModel, BipolarTransistor,
    BipolarTransistorParameters, Polarity,
};
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
pub use kernel::diode::{
    Diode, DiodeCapacitanceTemperatureModel, DiodeParameters, DiodeTemperatureModel,
};
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
pub use lift::{Lift, ParameterPath};
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
    /// Builds a library containing every native analog and logic model.
    #[must_use]
    pub fn standard() -> Self {
        let mut library = Self::default();
        library.add(Differentiable::<BehavioralCapacitor>::new());
        library.add(Differentiable::<BehavioralCurrentSource>::new());
        library.add(Differentiable::<BehavioralVoltageSource>::new());
        library.add(Thermal::<BipolarTransistor>::new());
        library.add(Thermal::<Bsim3>::new());
        library.add(Thermal::<Bsim4>::new());
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

    /// Registers analog `model`, replacing any existing model with the same name and wrapping
    /// it in [`Surrounded`] for per-instance temperature overrides.
    pub fn add(&mut self, model: impl Model + 'static) {
        self.logic.remove(model.name());
        self.models
            .insert(model.name(), Arc::new(Surrounded::new(model)));
    }

    /// Registers logic `model`, replacing any existing model with the same name.
    pub fn add_logic(&mut self, model: impl LogicModel + 'static) {
        self.models.remove(model.name());
        self.logic.insert(model.name(), Arc::new(model));
    }

    /// Returns the analog model registered under `name`, if present.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Model>> {
        self.models.get(name)
    }

    /// Returns the logic model registered under `name`, if present.
    #[must_use]
    pub fn logic(&self, name: &str) -> Option<&Arc<dyn LogicModel>> {
        self.logic.get(name)
    }

    /// Returns the names of all registered analog and logic models in alphabetical order.
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
    use std::collections::BTreeSet;

    use jolty_ir::{Parameters, numbers, parameter};
    use serde_json::{Value, json};

    use super::*;

    /// Builds representative parameter examples for every model, covering waveforms,
    /// per-instance temperatures, self-heating, and logic electrical pins.
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
                "behavioral_capacitor",
                json!({"expression": "1e-12 * v(positive, negative) + 1e-13 * tanh(v(a))"}),
            ),
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
                    "mobility_model": 3, "noise_model": 2, "check_parameters": true,
                    "charge_partition": 0.5, "temperature": 350}),
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
                "mosfet",
                json!({"oxide_thickness": 5e-8, "drain_resistance": 0.01,
                    "source_resistance": 0.5}),
            ),
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

    /// Returns `parameters` for standard-library model `name` with omitted defaults filled in.
    fn defaulted(library: &Library, name: &str, parameters: &Parameters) -> Parameters {
        match library.logic(name) {
            Some(model) => model.defaulted(parameters),
            None => library.get(name).expect("a model").defaulted(parameters),
        }
        .unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// Returns the JSON schema for standard-library model `name`.
    fn schema(library: &Library, name: &str) -> Schema {
        match library.logic(name) {
            Some(model) => model.schema(),
            None => library.get(name).expect("a model").schema(),
        }
    }

    fn object(value: &Value) -> Parameters {
        value.as_object().cloned().expect("an object")
    }

    /// Verifies that every pivot claimed by a model layout dominates all other entries in its
    /// column across `G`, `G + αC`, and `G + jωC` at operating-point and transient states.
    ///
    /// Samples unknowns across `±10` and sweeps `α` and `ω` from `0` to `10¹⁵` s⁻¹ so sparse
    /// elimination needs no pivot search.
    #[test]
    fn every_claimed_pivot_is_its_columns_largest() {
        let library = Library::standard();
        let frequencies: Vec<(f64, f64)> = [0.0, 1e3, 1e9, 1e15]
            .into_iter()
            .flat_map(|scale| [(scale, 0.0), (0.0, scale)])
            .collect();
        let (mut checked, mut off_diagonal) = (0, 0);
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
                    // Assigns each unknown a distinct fraction of ±10 per sample point.
                    let x: Vec<f64> = (0..unknowns)
                        .map(|index| 20.0 * f64::from((point * 7 + index * 13) % 41) / 40.0 - 10.0)
                        .collect();
                    let evaluation = bench.evaluate(&x);
                    let (conductance, capacitance) =
                        (&evaluation.conductance, &evaluation.capacitance);
                    for &(pivot, column) in &pivoted {
                        for &(real, imaginary) in &frequencies {
                            let entry = |row: usize| {
                                let (g, c) = (conductance[row][column], capacitance[row][column]);
                                (g + real * c).hypot(imaginary * c)
                            };
                            let claimed = entry(pivot);
                            for row in (0..x.len()).filter(|&row| row != pivot) {
                                assert!(
                                    entry(row) <= claimed,
                                    "{name}'s unknown {column} at {x:?}, \
                                     s = {real} + {imaginary}j: \
                                     {} against its pivot's {claimed} in row {pivot}",
                                    entry(row)
                                );
                            }
                        }
                        checked += 1;
                        off_diagonal += usize::from(pivot != column);
                    }
                }
            }
        }
        assert!(checked > 0, "no model claims a pivot");
        assert!(off_diagonal > 0, "no model claims a pivot off the diagonal");
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
                "behavioral_capacitor",
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

    #[test]
    fn every_models_parameter_metadata_matches_its_schema() {
        let library = Library::standard();
        let mut declared = Vec::new();
        for name in library.names() {
            let Some(model) = library.get(name) else {
                continue;
            };
            let kinds = model.parameter_kinds();
            let binned = model.binned_parameters();
            let schema = model.schema();
            let properties = schema
                .get("properties")
                .and_then(Value::as_object)
                .expect("properties");

            let mut kind_names = BTreeSet::new();
            for &(parameter, _) in kinds {
                assert!(
                    kind_names.insert(parameter),
                    "{name}: duplicate parameter metadata for `{parameter}`"
                );
                assert!(
                    properties.contains_key(parameter),
                    "{name}: parameter metadata `{parameter}` missing from schema"
                );
            }

            let mut binned_names = BTreeSet::new();
            let mut expected: BTreeSet<String> =
                kind_names.iter().copied().map(str::to_owned).collect();
            for &base in binned {
                assert!(
                    binned_names.insert(base),
                    "{name}: duplicate binned parameter `{base}`"
                );
                assert!(
                    kind_names.contains(base),
                    "{name}: binned parameter `{base}` missing from parameter_kinds"
                );
                for prefix in ['l', 'w', 'p'] {
                    let dependence = format!("{prefix}{base}");
                    assert!(
                        properties.contains_key(&dependence),
                        "{name}: binned dependence `{dependence}` missing from schema"
                    );
                    assert!(
                        !kind_names.contains(dependence.as_str()),
                        "{name}: binned dependence `{dependence}` listed in parameter_kinds"
                    );
                    expected.insert(dependence);
                }
            }

            if !kinds.is_empty() {
                declared.push(name);
                let wrapper_keys = ["thermal", "temperature", "temperature_offset"];
                let schema_keys: BTreeSet<String> = properties
                    .keys()
                    .filter(|key| !wrapper_keys.contains(&key.as_str()))
                    .cloned()
                    .collect();
                assert_eq!(expected, schema_keys, "{name}");
            }
        }
        assert_eq!(
            declared,
            [
                "bipolar_transistor",
                "bsim3",
                "bsim4",
                "capacitor",
                "current_controlled_current_source",
                "current_controlled_memristor",
                "current_controlled_switch",
                "current_controlled_voltage_source",
                "diode",
                "foundry_mosfet",
                "heat_source",
                "inductor",
                "jfet",
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
            ]
        );
        let bsim3 = library.get("bsim3").expect("bsim3");
        let bsim4 = library.get("bsim4").expect("bsim4");
        let empty: &[&str] = &[];
        assert_ne!(bsim3.binned_parameters(), empty);
        assert_ne!(bsim4.binned_parameters(), empty);
    }
}
