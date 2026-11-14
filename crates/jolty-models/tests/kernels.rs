//! The kernel rules: a kernel's evaluation allocates nothing.
//!
//! The counting allocator of `allocation-counter` stands in for the global one in this
//! test's program alone, and counts each allocation the thread that measures makes.
#![cfg(test)]

use jolty_models::{
    Binding, BipolarTransistor, Contributions, Device, Diode, Input, Jfet, Kernel, KernelDevice,
    Mosfet, NativeDevice, Node, Quantity, Unknown, UnknownInfo,
};
use jolty_numerics::PatternBuilder;

/// Evaluates an instance of `K` with `parameters`, each terminal on an unknown of its own,
/// at `x` and around it, limiting its steps and not, at an operating point and in a
/// transient, and returns how many allocations the evaluations made after the first.
fn allocations<K: Kernel>(parameters: &str, x: &[f64]) -> u64 {
    let parameters: K::Parameters<f64> = serde_json::from_str(parameters).expect("parameters");
    let terminals = K::interface(&parameters).terminals.len();
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
    let mut binding = Binding::new("device", &nodes, &mut unknowns, &mut builder, 300.15);
    let mut device = KernelDevice::<K>::new(parameters, &mut binding).expect("valid parameters");
    let pattern = builder.finish(unknowns.len());
    let mut contributions = Contributions::new(&pattern);
    let mut values = x.to_vec();
    values.resize(unknowns.len(), 0.1);
    let moved: Vec<f64> = values.iter().map(|value| value * 1.5 - 0.2).collect();
    // The device's first evaluation sizes the block it evaluates its kernel into.
    device.evaluate(
        &Input::new(&values, None, 1e-12),
        &mut contributions.output(&pattern),
    );
    let info = allocation_counter::measure(|| {
        for (step, at) in [&values, &moved, &values].into_iter().enumerate() {
            let time = (step == 1).then_some(1e-9);
            let mut input = Input::new(at, time, 1e-12);
            input.limiting = step > 0;
            let mut output = contributions.output(&pattern);
            device.evaluate(&input, &mut output);
        }
    });
    info.count_total
}

#[test]
fn a_kernels_evaluation_allocates_nothing() {
    let cases = [
        allocations::<Diode>("{}", &[0.7, 0.0]),
        allocations::<Diode>(
            r#"{"series_resistance": 0.5, "junction_capacitance": 1e-12,
                "breakdown_voltage": 5}"#,
            &[-6.0, 0.0],
        ),
        allocations::<Mosfet>(r#"{"oxide_thickness": 2e-8}"#, &[3.0, 2.0, 0.0, 0.0]),
        allocations::<Mosfet>(
            r#"{"level": 3, "channel": "p", "drain_resistance": 10, "source_resistance": 0.5}"#,
            &[-3.0, -2.0, 0.0, 0.0],
        ),
        allocations::<BipolarTransistor>("{}", &[5.0, 0.7, 0.0]),
        allocations::<BipolarTransistor>(
            r#"{"base_resistance": 50, "minimum_base_resistance": 0.5,
                "collector_resistance": 0.2, "emitter_resistance": 2,
                "base_emitter_capacitance": 1e-12, "forward_transit_time": 1e-10,
                "excess_phase": 10, "substrate": "vertical"}"#,
            &[5.0, 0.7, 0.0, -1.0],
        ),
        allocations::<Jfet>(
            r#"{"drain_resistance": 10, "source_resistance": 0.5,
                "gate_source_capacitance": 1e-12}"#,
            &[5.0, -0.5, 0.0],
        ),
    ];
    assert_eq!(cases, [0; 7]);
}
