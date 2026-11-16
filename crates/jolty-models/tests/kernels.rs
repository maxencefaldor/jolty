//! The kernel rules: a kernel's evaluation allocates nothing.
//!
//! The counting allocator of `allocation-counter` stands in for the global one in this
//! test's program alone, and counts each allocation the thread that measures makes.
#![cfg(test)]

#[path = "../src/testing/cases.rs"]
mod cases;

use jolty_models::{
    Binding, Contributions, Groups, Input, Library, Nature, Node, Runs, Unknown, UnknownIndex,
};
use jolty_numerics::PatternBuilder;

/// Evaluates a device of the standard library's model `name` with `parameters`, its
/// instance in a group of its own, each terminal on an unknown of its own, at the case's
/// values `x` and around them, limiting its steps and not, at an operating point and in a
/// transient, and returns how many allocations the evaluations made after the first.
fn allocations(name: &str, parameters: &str, x: &[f64]) -> u64 {
    let library = Library::standard();
    let parameters = serde_json::from_str(parameters).expect("parameters");
    let specification = library
        .get(name)
        .expect("a model")
        .read(&parameters)
        .expect("valid parameters");
    let terminals = specification.interface().terminals.len();
    let nodes: Vec<Node> = (0..terminals)
        .map(|index| Some(UnknownIndex::new(index)))
        .collect();
    let mut unknowns: Vec<Unknown> = (0..terminals)
        .map(|index| Unknown {
            name: format!("n{index}"),
            nature: Nature::Voltage,
        })
        .collect();
    let mut builder = PatternBuilder::new();
    let (mut groups, mut runs) = (Groups::new(), Runs::default());
    {
        let mut binding = Binding::new("device", &nodes, &mut unknowns, &mut builder);
        groups
            .add(&mut runs, (specification.kernel(), 300.15), &mut binding)
            .expect("valid parameters");
    }
    let pattern = builder.finish(unknowns.len());
    groups.finish(&mut runs, &pattern).expect("a small circuit");
    let mut contributions = Contributions::new(&pattern);
    let values = cases::values(x, unknowns.len());
    let moved: Vec<f64> = values.iter().map(|value| value * 1.5 - 0.2).collect();
    runs.evaluate(
        &groups,
        &Input::new(&values, None, 1e-12),
        &mut contributions,
        None,
    );
    let info = allocation_counter::measure(|| {
        for (step, at) in [&values, &moved, &values].into_iter().enumerate() {
            let time = (step == 1).then_some(1e-9);
            let mut input = Input::new(at, time, 1e-12);
            input.limiting = step > 0;
            runs.evaluate(&groups, &input, &mut contributions, None);
        }
    });
    info.count_total
}

/// Every analog model's kernel evaluates without allocating, at each case of it.
#[test]
fn a_kernels_evaluation_allocates_nothing() {
    let library = Library::standard();
    let analog: Vec<&str> = library
        .names()
        .filter(|name| library.get(name).is_some())
        .collect();
    assert_eq!(
        cases::covered(),
        analog,
        "a case of every analog model, and no more"
    );
    let found: Vec<(&str, u64)> = cases::CASES
        .iter()
        .map(|&(name, parameters, x)| (name, allocations(name, parameters, x)))
        .collect();
    assert!(found.iter().all(|&(_, count)| count == 0), "{found:?}");
}
