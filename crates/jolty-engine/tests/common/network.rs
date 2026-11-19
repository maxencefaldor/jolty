//! Random connected resistor-network proptest helpers shared across engine tests.

use std::ops::Range;

use proptest::prelude::*;

/// Returns the net name for `index`: `"0"` for ground and `"n<index>"` otherwise.
pub(crate) fn node(index: usize) -> String {
    if index == 0 {
        "0".to_owned()
    } else {
        format!("n{index}")
    }
}

/// Builds a proptest strategy for connected resistor graphs on `nodes` nodes (`0..nodes`),
/// combining a spanning tree with optional extra edges at log-uniform resistances
/// `10^resistance_exponent`.
pub(crate) fn connected_resistors(
    nodes: usize,
    resistance_exponent: Range<f64>,
) -> impl Strategy<Value = Vec<(usize, usize, f64)>> {
    let tree = proptest::collection::vec(
        (any::<prop::sample::Index>(), resistance_exponent.clone()),
        nodes - 1,
    );
    let extra = proptest::collection::vec((0..nodes, 0..nodes, resistance_exponent), 0..nodes);
    (tree, extra).prop_map(|(tree, extra)| {
        let mut resistors: Vec<(usize, usize, f64)> = tree
            .into_iter()
            .enumerate()
            .map(|(child, (parent, exponent))| {
                (child + 1, parent.index(child + 1), 10f64.powf(exponent))
            })
            .collect();
        resistors.extend(
            extra
                .into_iter()
                .filter(|(a, b, _)| a != b)
                .map(|(a, b, exponent)| (a, b, 10f64.powf(exponent))),
        );
        resistors
    })
}
