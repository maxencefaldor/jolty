//! Physical invariants on random circuits.
#![cfg(test)]

use jolty_engine::{Conditions, Options, Simulator};
use jolty_ir::Circuit;
use jolty_models::Library;
use proptest::prelude::*;
use serde_json::{Value, json};

/// A random grounded resistor network driven by current sources.
#[derive(Debug, Clone)]
struct Network {
    /// Resistors as `(node, node, resistance)`, where node 0 is the ground.
    resistors: Vec<(usize, usize, f64)>,
    /// Current sources as `(node, current)`, each pushing its current into its node.
    sources: Vec<(usize, f64)>,
}

fn net(node: usize) -> String {
    if node == 0 {
        "0".to_owned()
    } else {
        format!("n{node}")
    }
}

/// A connected network on up to eight nodes: a spanning tree, then extra resistors.
fn network() -> impl Strategy<Value = Network> {
    (2usize..9).prop_flat_map(|nodes| {
        let tree =
            proptest::collection::vec((any::<prop::sample::Index>(), -3.0..6.0f64), nodes - 1);
        let extra = proptest::collection::vec((0..nodes, 0..nodes, -3.0..6.0f64), 0..nodes);
        let sources = proptest::collection::vec((1..nodes, -1e-2..1e-2f64), 1..4);
        (tree, extra, sources).prop_map(move |(tree, extra, sources)| {
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
            Network { resistors, sources }
        })
    })
}

fn circuit(network: &Network, reversed: bool) -> Circuit {
    let mut instances: Vec<Value> = network
        .resistors
        .iter()
        .enumerate()
        .map(|(index, &(a, b, resistance))| {
            json!({
                "id": format!("R{index}"),
                "of": {"device": "resistor"},
                "parameters": {"resistance": resistance},
                "terminals": {"a": net(a), "b": net(b)},
            })
        })
        .chain(
            network
                .sources
                .iter()
                .enumerate()
                .map(|(index, &(node, current))| {
                    json!({
                        "id": format!("I{index}"),
                        "of": {"device": "current_source"},
                        "parameters": {"waveform": {"dc": current}},
                        "terminals": {"positive": "0", "negative": net(node)},
                    })
                }),
        )
        .collect();
    if reversed {
        instances.reverse();
    }
    let document = json!({
        "format": "jolty-circuit", "version": 1, "top": "network",
        "components": {"network": {
            "ports": [{"name": "0", "role": "ground"}],
            "instances": instances,
        }}
    });
    Circuit::from_json(&document.to_string()).expect("a valid circuit file")
}

impl Network {
    /// The number of nodes, the ground included.
    fn nodes(&self) -> usize {
        1 + self
            .resistors
            .iter()
            .map(|&(a, b, _)| a.max(b))
            .max()
            .unwrap_or(0)
    }

    /// Every node's voltage at the operating point, by node number, the ground being zero.
    fn voltages(&self, reversed: bool) -> Vec<f64> {
        let circuit = circuit(self, reversed);
        let mut simulator = Simulator::new(&circuit, &Library::standard(), &Conditions::default())
            .expect("an elaborated circuit");
        let solution = simulator
            .operating_point(&tight())
            .expect("an operating point");
        (0..self.nodes())
            .map(|node| {
                simulator
                    .unknown(&net(node))
                    .map_or(0.0, |unknown| solution.values[unknown.index()])
            })
            .collect()
    }
}

/// Options far tighter than SPICE's, which every network must still meet.
fn tight() -> Options {
    Options {
        relative_tolerance: 1e-9,
        current_tolerance: 1e-15,
        ..Options::default()
    }
}

proptest! {
    /// The currents into every node sum to zero, within the relative tolerance on the
    /// currents meeting there, or within the rounding of the node voltages they flow
    /// between: one unit in the last place of a voltage of hundreds of volts moves the
    /// current through a milliohm by tens of picoamperes.
    #[test]
    fn currents_balance_at_every_node(network in network()) {
        let options = tight();
        let voltage = network.voltages(false);
        for node in 1..network.nodes() {
            let (mut sum, mut largest, mut resolution) = (0.0, 0.0_f64, 0.0);
            for &(a, b, resistance) in &network.resistors {
                let current = (voltage[a] - voltage[b]) / resistance;
                let leaving = if node == a {
                    current
                } else if node == b {
                    -current
                } else {
                    continue;
                };
                sum += leaving;
                largest = largest.max(leaving.abs());
                resolution += (voltage[a].abs() + voltage[b].abs()) / resistance;
            }
            for &(source, current) in &network.sources {
                if source == node {
                    sum -= current;
                    largest = largest.max(current.abs());
                }
            }
            let tolerance = options.relative_tolerance * largest
                + options.current_tolerance
                + 8.0 * f64::EPSILON * resolution;
            prop_assert!(sum.abs() <= tolerance, "node {node}: {sum} of {largest}");
        }
    }

    /// The solution does not depend on the order instances are placed in.
    #[test]
    fn instance_order_does_not_matter(network in network()) {
        let (forward, backward) = (network.voltages(false), network.voltages(true));
        for node in 1..network.nodes() {
            let (a, b) = (forward[node], backward[node]);
            let tolerance = 1e-9 * a.abs().max(b.abs()) + 1e-12;
            prop_assert!((a - b).abs() <= tolerance, "node {node}: {a} vs {b}");
        }
    }
}
