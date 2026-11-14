//! How a device adds itself to the circuit's equations: the entries it claims among its
//! nodes, and the branches whose currents are unknowns of their own.

use jolty_numerics::Entry;

use crate::{Binding, Input, Node, Output, Quantity, Real, Unknown};

/// The local index of a two-terminal element's positive node in its stamp.
const POSITIVE: usize = 0;
/// The local index of its negative node.
const NEGATIVE: usize = 1;
/// The pair of them, the element's one branch.
const PAIR: (usize, usize) = (POSITIVE, NEGATIVE);

/// The resistance below which a resistance carries its current as an unknown, in ohms.
///
/// A conductance stamped beside a far smaller one at a node swamps it in floating point: a
/// node held only by gmin's picosiemens loses them beside a megasiemens, and its equation
/// turns singular. A current unknown keeps the two apart, as modified nodal analysis does
/// for the elements of its second group (Ho, Ruehli and Brennan, 1975); below 1 Ω, gmin
/// keeps four digits beside the resistance's conductance.
pub const SMALL_RESISTANCE: f64 = 1.0;

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

    /// The nodes, in the order they were claimed.
    pub(crate) fn nodes(&self) -> [Node; N] {
        self.nodes
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

/// The two nodes of a two-terminal element and the four matrix entries that join them.
///
/// Current flows from the positive node through the element to the negative node, and the
/// element's voltage is the positive node's voltage minus the negative node's.
#[derive(Debug, Clone)]
pub struct TerminalPair(Stamp<2>);

impl TerminalPair {
    /// Claims the entries that join `positive` and `negative`.
    pub fn new(binding: &mut Binding<'_>, positive: Node, negative: Node) -> Self {
        Self(Stamp::new(binding, [positive, negative]))
    }

    /// The element's voltage.
    #[must_use]
    pub fn voltage(&self, input: &Input<'_>) -> f64 {
        self.0.voltage(input, POSITIVE) - self.0.voltage(input, NEGATIVE)
    }

    /// The positive and the negative node.
    #[must_use]
    pub fn nodes(&self) -> [Node; 2] {
        self.0.nodes()
    }

    /// Adds `current`, flowing through the element, to the two nodes' equations.
    pub fn add_current<R: Real>(&self, output: &mut Output<'_, R>, current: R) {
        self.0.current(output, POSITIVE, NEGATIVE, current);
    }

    /// Adds `charge`, held on the positive node and its opposite on the negative node.
    pub fn add_charge<R: Real>(&self, output: &mut Output<'_, R>, charge: R) {
        self.0.charge(output, POSITIVE, NEGATIVE, charge);
    }

    /// Adds `conductance`, the derivative of the current with respect to the voltage.
    pub fn add_conductance<R: Real>(&self, output: &mut Output<'_, R>, conductance: R) {
        self.0.conductance(output, PAIR, PAIR, conductance);
    }

    /// Adds `capacitance`, the derivative of the charge with respect to the voltage.
    pub fn add_capacitance<R: Real>(&self, output: &mut Output<'_, R>, capacitance: R) {
        self.0.capacitance(output, PAIR, PAIR, capacitance);
    }
}

/// A branch from a positive node to a negative one whose current, flowing through it, is an
/// unknown of its own, as a voltage source's or an inductor's is: the current enters both
/// nodes' equations, and the branch's own equation constrains its voltage.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CurrentBranch {
    current: Unknown,
    nodes: [Node; 2],
    /// The entries at (each node, current) and (current, each node).
    entries: [Option<Entry>; 4],
}

impl CurrentBranch {
    /// A branch from `positive` to `negative`, its current a new unknown labelled `label`.
    pub(crate) fn new(
        binding: &mut Binding<'_>,
        label: &str,
        [positive, negative]: [Node; 2],
    ) -> Self {
        let current = binding.internal(label, Quantity::Current);
        let row = Some(current);
        Self {
            current,
            nodes: [positive, negative],
            entries: [
                binding.entry(positive, row),
                binding.entry(negative, row),
                binding.entry(row, positive),
                binding.entry(row, negative),
            ],
        }
    }

    /// The branch's unknown, its current, and its own equation's, where a device claims the
    /// entries of whatever else that equation reads.
    pub(crate) const fn unknown(&self) -> Unknown {
        self.current
    }

    /// The current through the branch.
    pub(crate) fn current(&self, input: &Input<'_>) -> f64 {
        input.value(Some(self.current))
    }

    /// The branch's voltage, its positive node's over its negative node's.
    pub(crate) fn voltage(&self, input: &Input<'_>) -> f64 {
        input.value(self.nodes[0]) - input.value(self.nodes[1])
    }

    /// Adds the branch's current to its nodes' equations, and `residual` to its own
    /// equation, with the derivatives of both with respect to the current and the voltage,
    /// the residual's being 1; a device adds the derivatives of the rest of its equation
    /// itself.
    pub(crate) fn add<R: Real>(&self, input: &Input<'_>, output: &mut Output<'_, R>, residual: R) {
        self.add_sloped(input, output, residual, R::constant(1.0));
    }

    /// As [`CurrentBranch::add`], the residual's derivative with respect to the branch's
    /// voltage being `slope`.
    pub(crate) fn add_sloped<R: Real>(
        &self,
        input: &Input<'_>,
        output: &mut Output<'_, R>,
        residual: R,
        slope: R,
    ) {
        let [positive, negative] = self.nodes;
        let [
            positive_current,
            negative_current,
            current_positive,
            current_negative,
        ] = self.entries;
        let (one, current) = (R::constant(1.0), self.current(input));
        output.add_residual(positive, R::constant(current));
        output.add_residual(negative, R::constant(-current));
        output.add_conductance(positive_current, one);
        output.add_conductance(negative_current, -one);
        output.add_residual(Some(self.current), residual);
        output.add_conductance(current_positive, slope);
        output.add_conductance(current_negative, -slope);
    }
}
