//! The flattened circuit as a graph: nodes joined by the branches of its devices.

use jolty_engine::{Endpoint, Placed, Simulator};
use jolty_models::{Branch, Node};

/// The index of the reference node among a structure's nodes.
///
/// Source: SPICE / modified nodal analysis datum node convention (`0` is ground; L. W. Nagel,
/// "SPICE2: A computer program to simulate semiconductor circuits", Memorandum ERL-M520,
/// University of California, Berkeley, 1975).
pub(crate) const REFERENCE: usize = 0;

/// The nodes of an elaborated circuit and the devices that join them.
///
/// Nodes are numbered from the reference, at [`REFERENCE`], then one per unknown in the
/// engine's order; only the unknowns that are node voltages are ever joined.
///
/// A logic device's supply and ground join nodes too. Its connect elements' supply and
/// ground are its own, so only their first terminal, the pin's, counts as theirs.
pub(crate) struct Structure<'a> {
    simulator: &'a Simulator,
    /// How many terminals join each node.
    degrees: Vec<usize>,
    /// Whether each device is a connect element.
    elements: Vec<bool>,
}

impl<'a> Structure<'a> {
    pub(crate) fn new(simulator: &'a Simulator) -> Self {
        let mut elements = vec![false; simulator.devices().len()];
        for net in simulator.logic_nets() {
            for end in net.sources.iter().chain(&net.loads) {
                if let Endpoint::Terminal { device } = *end {
                    elements[device] = true;
                }
            }
        }
        let mut degrees = vec![0; simulator.unknowns().len() + 1];
        for (device, &element) in simulator.devices().iter().zip(&elements) {
            let counted = if element { 1 } else { device.nodes().len() };
            for &node in &device.nodes()[..counted] {
                degrees[index(node)] += 1;
            }
        }
        for device in simulator.logic_devices() {
            for &node in device.nodes() {
                degrees[index(node)] += 1;
            }
        }
        Self {
            simulator,
            degrees,
            elements,
        }
    }

    /// The simulator the structure reads.
    pub(crate) fn simulator(&self) -> &'a Simulator {
        self.simulator
    }

    /// Whether the device at `device` is a connect element.
    pub(crate) fn is_element(&self, device: usize) -> bool {
        self.elements[device]
    }

    /// The number of nodes, the reference included.
    pub(crate) fn len(&self) -> usize {
        self.degrees.len()
    }

    /// The devices, in the order they were placed.
    pub(crate) fn devices(&self) -> &'a [Placed] {
        self.simulator.devices()
    }

    /// How many terminals join `node`.
    pub(crate) fn degree(&self, node: usize) -> usize {
        self.degrees[node]
    }

    /// The name of `node`: a net's path, or the reference's name.
    pub(crate) fn name(&self, node: usize) -> &'a str {
        if node == REFERENCE {
            self.simulator.reference()
        } else {
            &self.simulator.unknowns()[node - 1].name
        }
    }
}

/// The index of `node` among a structure's nodes.
pub(crate) fn index(node: Node) -> usize {
    node.map_or(REFERENCE, |unknown| unknown.index() + 1)
}

/// The nodes a branch of `device` joins, by index.
pub(crate) fn ends(device: &Placed, branch: Branch) -> (usize, usize) {
    let nodes = device.nodes();
    (index(nodes[branch.from]), index(nodes[branch.to]))
}

/// Names in backticks, joined as prose: `` `a` ``, `` `a` and `b` ``, `` `a`, `b` and `c` ``;
/// past `limit` names, the rest are counted.
pub(crate) fn list<'s>(names: impl IntoIterator<Item = &'s str>, limit: usize) -> String {
    prose(names, limit, |name| format!("`{name}`"))
}

/// Formats up to the first `limit` items of `items` with `format` and joins them as prose,
/// counting any remaining items without formatting them.
pub(crate) fn prose<T>(
    items: impl IntoIterator<Item = T>,
    limit: usize,
    format: impl FnMut(T) -> String,
) -> String {
    let mut iter = items.into_iter();
    let mut shown: Vec<String> = iter.by_ref().take(limit).map(format).collect();
    let hidden = iter.count();
    if hidden > 0 {
        return if shown.is_empty() {
            String::new()
        } else {
            format!("{} and {hidden} more", shown.join(", "))
        };
    }
    match shown.pop() {
        None => String::new(),
        Some(only) if shown.is_empty() => only,
        Some(last) => format!("{} and {last}", shown.join(", ")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_read_as_prose() {
        assert_eq!(list([], 3), "");
        assert_eq!(list(["a"], 3), "`a`");
        assert_eq!(list(["a", "b"], 3), "`a` and `b`");
        assert_eq!(list(["a", "b", "c"], 3), "`a`, `b` and `c`");
        assert_eq!(
            list(["a", "b", "c", "d", "e"], 3),
            "`a`, `b`, `c` and 2 more"
        );
        assert_eq!(list(["a", "b"], 0), "");
    }

    #[test]
    fn prose_only_formats_shown_items() {
        let mut calls = 0_usize;
        let rendered = prose(0..100, 3, |item| {
            calls += 1;
            format!("#{item}")
        });
        assert_eq!(rendered, "#0, #1, #2 and 97 more");
        assert_eq!(calls, 3);

        let mut zero_calls = 0_usize;
        let empty = prose(0..10, 0, |item| {
            zero_calls += 1;
            format!("#{item}")
        });
        assert_eq!(empty, "");
        assert_eq!(zero_calls, 0);
    }
}
