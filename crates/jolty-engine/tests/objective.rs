//! An objective's derivatives by the adjoint method against finite differences of the whole
//! circuit; `src/objective.rs` checks them against the direct method's on the corpus.
#![cfg(test)]

use std::collections::BTreeMap;

use jolty_engine::{
    Conditions, Functional, Method, Objective, Options, SimulationError, Simulator, Transient,
};
use jolty_ir::Circuit;
use jolty_models::Library;
use serde_json::{Value, json};

/// A circuit whose top component has a ground port `0` and places `instances`.
fn circuit_of(instances: &Value) -> Circuit {
    let document = json!({
        "format": "jolty-circuit", "version": 4, "top": "test",
        "components": {"test": {"ports": [{"name": "0", "role": "ground"}], "instances": instances}}
    });
    Circuit::from_json(&document.to_string()).expect("a valid circuit")
}

/// A sine through a resistor into a diode and a capacitor across it: a rectifier, whose
/// waveforms bend where the diode conducts.
fn rectifier() -> Value {
    json!([
        {"name": "V1", "of": {"model": "voltage_source"},
         "parameters": {"waveform": {"sine": {"offset": 0.2, "amplitude": 1.0,
                                              "frequency": 1e6}}},
         "terminals": {"positive": "in", "negative": "0"}},
        {"name": "R1", "of": {"model": "resistor"}, "parameters": {"resistance": 1e3},
         "terminals": {"a": "in", "b": "a"}},
        {"name": "D1", "of": {"model": "diode"},
         "parameters": {"saturation_current": 1e-14, "series_resistance": 10.0,
                        "transit_time": 1e-9, "junction_capacitance": 2e-12},
         "terminals": {"anode": "a", "cathode": "out"}},
        {"name": "C1", "of": {"model": "capacitor"}, "parameters": {"capacitance": 1e-10},
         "terminals": {"a": "out", "b": "0"}},
        {"name": "R2", "of": {"model": "resistor"}, "parameters": {"resistance": 1e4},
         "terminals": {"a": "out", "b": "0"}}
    ])
}

/// Tight tolerances, and a truncation tolerance no step's error reaches, so every run
/// takes the same steps, growing to the longest, and finite differences see what the
/// parameters move, not where the steps fall.
fn fixed_steps() -> Options {
    Options {
        relative_tolerance: 1e-12,
        voltage_tolerance: 1e-15,
        current_tolerance: 1e-18,
        truncation_tolerance: f64::INFINITY,
        ..Options::default()
    }
}

/// The value of `objective` on the circuit of `instances` with the parameter `name` of
/// `device` at `value`.
fn moved(
    instances: &Value,
    (device, name, value): (&str, &str, f64),
    objective: &Objective,
) -> f64 {
    let mut parameters = instances
        .as_array()
        .and_then(|placed| placed.iter().find(|instance| instance["name"] == device))
        .expect("the device is placed")["parameters"]
        .clone();
    let slot = name
        .split('.')
        .try_fold(&mut parameters, |slot, key| slot.get_mut(key))
        .expect("the parameter is set");
    *slot = Value::from(value);
    let mut conditions = Conditions::default();
    conditions.overrides.insert(
        device.to_owned(),
        serde_json::from_value(parameters).expect("parameters"),
    );
    let mut simulator = Simulator::new(&circuit_of(instances), &Library::standard(), &conditions)
        .expect("an elaborated circuit");
    simulator
        .objective(objective, &fixed_steps())
        .expect("an objective")
        .value
}

/// A transient of the rectifier by `method`, its steps a hundredth of its two periods.
fn transient(method: Method) -> Transient {
    Transient {
        stop: 2e-6,
        max_step: Some(1e-8),
        method,
        breakpoints: Vec::new(),
        initial_conditions: BTreeMap::new(),
        operating_point: true,
    }
}

#[test]
fn an_objectives_derivatives_match_finite_differences() {
    let functionals = [
        Functional::Integral("out".to_owned()),
        Functional::Value {
            unknown: "out".to_owned(),
            at: 1.234_5e-6,
        },
        Functional::Sum("a".to_owned()),
    ];
    let instances = rectifier();
    for method in [Method::Trapezoidal, Method::Gear2] {
        for functional in &functionals {
            let objective = Objective {
                functional: functional.clone(),
                of: transient(method),
            };
            let mut simulator = Simulator::new(
                &circuit_of(&instances),
                &Library::standard(),
                &Conditions::default(),
            )
            .expect("an elaborated circuit");
            let found = simulator
                .objective(&objective, &fixed_steps())
                .expect("an objective");
            assert_eq!(found.parameters.len(), 10, "{:?}", found.parameters);
            for (parameter, exact) in found.parameters.iter().zip(&found.derivatives) {
                let (device, name) = (parameter.instance.as_str(), parameter.name.as_str());
                let step = 1e-4 * parameter.value;
                let above = moved(
                    &instances,
                    (device, name, parameter.value + step),
                    &objective,
                );
                let below = moved(
                    &instances,
                    (device, name, parameter.value - step),
                    &objective,
                );
                let finite = (above - below) / (2.0 * step);
                // A central difference's truncation error, the parameter's relative step
                // squared, 1e-8, times the curvature, with the solves' rounding over the
                // step.
                assert!(
                    (exact - finite).abs() <= 1e-5 * exact.abs().max(finite.abs()),
                    "{method:?}, {functional:?}: d/d{device}.{name}: adjoint {exact}, \
                     finite differences {finite}"
                );
            }
        }
    }
}

#[test]
fn an_objectives_time_lies_within_the_run() {
    let mut simulator = Simulator::new(
        &circuit_of(&rectifier()),
        &Library::standard(),
        &Conditions::default(),
    )
    .expect("an elaborated circuit");
    let objective = Objective {
        functional: Functional::Value {
            unknown: "out".to_owned(),
            at: 3e-6,
        },
        of: transient(Method::Trapezoidal),
    };
    let error = simulator
        .objective(&objective, &Options::default())
        .expect_err("a time after the run");
    assert_eq!(
        error,
        SimulationError::InvalidAnalysis(
            "an objective's time lies within the run, from 0 to 2e-6, not 3e-6".to_owned()
        )
    );
}

#[test]
fn an_objective_names_an_unknown_that_exists() {
    let mut simulator = Simulator::new(
        &circuit_of(&rectifier()),
        &Library::standard(),
        &Conditions::default(),
    )
    .expect("an elaborated circuit");
    let objective = Objective {
        functional: Functional::Integral("nowhere".to_owned()),
        of: transient(Method::Trapezoidal),
    };
    let error = simulator
        .objective(&objective, &Options::default())
        .expect_err("no such unknown");
    assert_eq!(error, SimulationError::UnknownOutput("nowhere".to_owned()));
}
