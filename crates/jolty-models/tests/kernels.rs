//! The kernel rules: a kernel's evaluation allocates nothing.
//!
//! The counting allocator of `allocation-counter` stands in for the global one in this
//! test's program alone, and counts each allocation the thread that measures makes.
#![cfg(test)]

use jolty_models::{
    Binding, Contributions, Groups, Input, Library, Node, Quantity, Runs, Unknown, UnknownInfo,
};
use jolty_numerics::PatternBuilder;

/// Evaluates a device of the standard library's model `name` with `parameters`, its
/// instance in a group of its own, each terminal on an unknown of its own, at `x` and around
/// it, limiting its steps and not, at an operating point and in a transient, and returns
/// how many allocations the evaluations made after the first.
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
        .map(|index| Some(Unknown::new(index)))
        .collect();
    let mut unknowns: Vec<UnknownInfo> = (0..terminals)
        .map(|index| UnknownInfo {
            name: format!("n{index}"),
            quantity: Quantity::Voltage,
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
    let mut values = x.to_vec();
    values.resize(unknowns.len(), 0.1);
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

#[test]
fn a_kernels_evaluation_allocates_nothing() {
    let cases = [
        ("diode", "{}", &[0.7, 0.0][..]),
        (
            "diode",
            r#"{"series_resistance": 0.5, "junction_capacitance": 1e-12,
                "breakdown_voltage": 5}"#,
            &[-6.0, 0.0],
        ),
        (
            "mosfet",
            r#"{"oxide_thickness": 2e-8}"#,
            &[3.0, 2.0, 0.0, 0.0],
        ),
        (
            "mosfet",
            r#"{"level": 3, "channel": "p", "drain_resistance": 10, "source_resistance": 0.5}"#,
            &[-3.0, -2.0, 0.0, 0.0],
        ),
        ("vdmos", "{}", &[12.0, 6.0, 0.0]),
        (
            "vdmos",
            r#"{"channel": "p", "drain_resistance": 0.05, "source_resistance": 2,
                "gate_resistance": 3, "body_resistance": 0.01, "quasi_saturation_resistance": 1,
                "quasi_saturation_voltage": 10, "breakdown_voltage": 30, "thermal": true}"#,
            &[-12.0, -6.0, 0.0, 5.0],
        ),
        ("bipolar_transistor", "{}", &[5.0, 0.7, 0.0]),
        (
            "bipolar_transistor",
            r#"{"base_resistance": 50, "minimum_base_resistance": 0.5,
                "collector_resistance": 0.2, "emitter_resistance": 2,
                "base_emitter_capacitance": 1e-12, "forward_transit_time": 1e-10,
                "excess_phase": 10, "substrate": "vertical"}"#,
            &[5.0, 0.7, 0.0, -1.0],
        ),
        (
            "jfet",
            r#"{"drain_resistance": 10, "source_resistance": 0.5,
                "gate_source_capacitance": 1e-12}"#,
            &[5.0, -0.5, 0.0],
        ),
        (
            "resistor",
            r#"{"resistance": 1e3, "voltage_coefficient": 0.1, "thermal": true}"#,
            &[1.0, 0.0, 5.0],
        ),
        ("resistor", r#"{"resistance": 0.5}"#, &[1.0, 0.0, 2.0]),
        ("capacitor", r#"{"capacitance": 1e-9}"#, &[1.0, 0.0]),
        (
            "coupled_inductors",
            r#"{"windings": [{"inductance": 1e-3}, {"inductance": 4e-3}],
                "couplings": [{"windings": [1, 2], "coefficient": 0.9}]}"#,
            &[1.0, 0.0, 0.5, 0.0, 1e-3, 2e-3],
        ),
        (
            "voltage_source",
            r#"{"waveform": {"sine": {"offset": 0, "amplitude": 1, "frequency": 1e6}}}"#,
            &[1.0, 0.0, 1e-3],
        ),
        (
            "current_controlled_switch",
            "{}",
            &[1.0, 0.0, 0.0, 0.0, 2e-3],
        ),
        (
            "behavioral_voltage_source",
            r#"{"expression": "if(v(a) > 0.5, tanh(v(a)) * i(s), table(v(b), 0, 0, 1, 1))"}"#,
            &[1.0, 0.0, 0.7, 0.3, 0.0, 0.0, 1e-3, 2e-3],
        ),
        (
            "voltage_controlled_memristor",
            r#"{"on_resistance": 1e3, "off_resistance": 1e5, "on_voltage": -1,
                "off_voltage": 0.5, "on_rate": 1e6, "off_rate": 1e5, "state": 0.5}"#,
            &[1.0, 0.0, 0.5],
        ),
        (
            "photocell",
            r#"{"resistance": 1e4, "response_time": 0.02}"#,
            &[1.0, 0.0, 5.0, 0.0, 0.5],
        ),
        ("heat_source", r#"{"power": 0.5}"#, &[1.0, 0.0]),
    ];
    let found: Vec<(&str, u64)> = cases
        .iter()
        .map(|&(name, parameters, x)| (name, allocations(name, parameters, x)))
        .collect();
    assert!(found.iter().all(|&(_, count)| count == 0), "{found:?}");
}
