use jolty_numerics::Entry;

use crate::{Binding, Input, Node, Output, Real};

/// The nodes of a device, by local index, with every matrix entry among them claimed, so a
/// model adds each current or charge between two of its nodes and its derivatives with
/// respect to the voltages between others.
///
/// Two local indices may name the same node, as a transistor's internal drain does its
/// terminal when the transistor has no drain resistance; their contributions then add.
#[derive(Debug, Clone)]
pub(crate) struct Stamp<const N: usize> {
    nodes: [Node; N],
    entries: [[Option<Entry>; N]; N],
}

impl<const N: usize> Stamp<N> {
    /// Claims every entry among `nodes`.
    pub(crate) fn new(binding: &mut Binding<'_>, nodes: [Node; N]) -> Self {
        let entries = nodes.map(|row| nodes.map(|column| binding.entry(row, column)));
        Self { nodes, entries }
    }

    /// The voltage of the node at `index`.
    pub(crate) fn voltage(&self, input: &Input<'_>, index: usize) -> f64 {
        input.value(self.nodes[index])
    }

    /// Adds `current`, flowing through the device from node `from` to node `to`.
    pub(crate) fn current<R: Real>(
        &self,
        output: &mut Output<'_, R>,
        from: usize,
        to: usize,
        current: R,
    ) {
        output.add_residual(self.nodes[from], current);
        output.add_residual(self.nodes[to], -current);
    }

    /// Adds `charge`, held on node `on` and its opposite on node `against`.
    pub(crate) fn charge<R: Real>(
        &self,
        output: &mut Output<'_, R>,
        on: usize,
        against: usize,
        charge: R,
    ) {
        output.add_charge(self.nodes[on], charge);
        output.add_charge(self.nodes[against], -charge);
    }

    /// Adds the derivative `conductance` of a current from `from` to `to` with respect to
    /// the voltage of `plus` over `minus`.
    pub(crate) fn conductance<R: Real>(
        &self,
        output: &mut Output<'_, R>,
        (from, to): (usize, usize),
        (plus, minus): (usize, usize),
        conductance: R,
    ) {
        output.add_conductance(self.entries[from][plus], conductance);
        output.add_conductance(self.entries[from][minus], -conductance);
        output.add_conductance(self.entries[to][plus], -conductance);
        output.add_conductance(self.entries[to][minus], conductance);
    }

    /// Adds the derivative `capacitance` of a charge on `on` against `against` with
    /// respect to the voltage of `plus` over `minus`.
    pub(crate) fn capacitance<R: Real>(
        &self,
        output: &mut Output<'_, R>,
        (on, against): (usize, usize),
        (plus, minus): (usize, usize),
        capacitance: R,
    ) {
        output.add_capacitance(self.entries[on][plus], capacitance);
        output.add_capacitance(self.entries[on][minus], -capacitance);
        output.add_capacitance(self.entries[against][plus], -capacitance);
        output.add_capacitance(self.entries[against][minus], capacitance);
    }
}
