//! The flattened circuit as a graph: nodes joined by the branches of its devices.

use jolty_engine::{Endpoint, Placed, Simulator};
use jolty_models::{Branch, Node};

/// The index of the reference node among a structure's nodes.
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
                if let Endpoint::Element { device } = *end {
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

/// Disjoint sets of nodes, merged by union by size with path halving.
pub(crate) struct Partition {
    parents: Vec<usize>,
    sizes: Vec<usize>,
}

impl Partition {
    /// `count` nodes, each in a set of its own.
    pub(crate) fn new(count: usize) -> Self {
        Self {
            parents: (0..count).collect(),
            sizes: vec![1; count],
        }
    }

    /// The representative of the set holding `node`.
    pub(crate) fn find(&mut self, mut node: usize) -> usize {
        while self.parents[node] != node {
            self.parents[node] = self.parents[self.parents[node]];
            node = self.parents[node];
        }
        node
    }

    /// Merges the sets holding `a` and `b`; returns whether they were apart.
    pub(crate) fn union(&mut self, a: usize, b: usize) -> bool {
        let (mut a, mut b) = (self.find(a), self.find(b));
        if a == b {
            return false;
        }
        if self.sizes[a] < self.sizes[b] {
            std::mem::swap(&mut a, &mut b);
        }
        self.parents[b] = a;
        self.sizes[a] += self.sizes[b];
        true
    }
}

/// Names in backticks, joined as prose: `` `a` ``, `` `a` and `b` ``, `` `a`, `b` and `c` ``;
/// past `limit` names, the rest are counted.
pub(crate) fn list<'s>(names: impl IntoIterator<Item = &'s str>, limit: usize) -> String {
    let names: Vec<&str> = names.into_iter().collect();
    let shown: Vec<String> = names
        .iter()
        .take(limit)
        .map(|name| format!("`{name}`"))
        .collect();
    let hidden = names.len() - shown.len();
    match (shown.as_slice(), hidden) {
        ([], _) => String::new(),
        ([only], 0) => only.clone(),
        ([first @ .., last], 0) => format!("{} and {last}", first.join(", ")),
        (_, _) => format!("{} and {hidden} more", shown.join(", ")),
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
    }

    #[test]
    fn a_partition_merges_sets() {
        let mut partition = Partition::new(4);
        assert!(partition.union(0, 1));
        assert!(partition.union(2, 3));
        assert!(!partition.union(1, 0));
        assert_eq!(partition.find(1), partition.find(0));
        assert_ne!(partition.find(1), partition.find(2));
        assert!(partition.union(1, 3));
        assert_eq!(partition.find(0), partition.find(2));
    }
}
