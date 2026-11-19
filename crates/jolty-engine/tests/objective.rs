//! An objective's derivatives by the adjoint method against finite differences of the whole
//! circuit; `crates/jolty-engine/src/objective.rs` checks them against the direct method's on
//! the corpus.
#![cfg(test)]

mod common;
#[path = "common/derivatives.rs"]
mod derivatives;
#[path = "common/sim.rs"]
mod sim;
#[path = "common/transient.rs"]
mod transient;

use common::circuit;
use derivatives::{moved, rectifier, tight_options};
use jolty_engine::{Functional, Method, Objective, Options, SimulationError};
use sim::simulator;
use transient::transient;

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
    let built = circuit(rectifier());
    let options = tight_options(true);
    for method in [Method::Trapezoidal, Method::Gear2] {
        for functional in &functionals {
            let objective = Objective {
                functional: functional.clone(),
                of: transient(2e-6, Some(1e-8), method),
            };
            let mut simulator = simulator(&built);
            let found = simulator
                .objective(&objective, &options)
                .expect("an objective");
            assert_eq!(found.parameters.len(), 10, "{:?}", found.parameters);
            for (parameter, exact) in found.parameters.iter().zip(&found.derivatives) {
                let (device, name) = (parameter.instance.as_str(), parameter.name.as_str());
                let step = 1e-4 * parameter.value;
                let value_at = |level: f64| {
                    moved(&built, device, name, level)
                        .objective(&objective, &options)
                        .expect("an objective")
                        .value
                };
                let finite = (value_at(parameter.value + step) - value_at(parameter.value - step))
                    / (2.0 * step);
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
    let mut simulator = simulator(&circuit(rectifier()));
    let objective = Objective {
        functional: Functional::Value {
            unknown: "out".to_owned(),
            at: 3e-6,
        },
        of: transient(2e-6, Some(1e-8), Method::Trapezoidal),
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
    let mut simulator = simulator(&circuit(rectifier()));
    let objective = Objective {
        functional: Functional::Integral("nowhere".to_owned()),
        of: transient(2e-6, Some(1e-8), Method::Trapezoidal),
    };
    let error = simulator
        .objective(&objective, &Options::default())
        .expect_err("no such unknown");
    assert_eq!(error, SimulationError::UnknownOutput("nowhere".to_owned()));
}
