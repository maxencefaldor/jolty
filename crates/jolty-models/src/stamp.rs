use jolty_numerics::Entry;

use crate::{Binding, Input, Node, Output, Real};

/// The local index of a two-terminal element's positive node in its stamp.
const POSITIVE: usize = 0;
/// The local index of its negative node.
const NEGATIVE: usize = 1;
/// The pair of them, the element's one branch.
const PAIR: (usize, usize) = (POSITIVE, NEGATIVE);

/// The conductance of a series `resistance`, or zero without one, where the terminal is its
/// own node behind it.
pub(crate) fn series_conductance<R: Real>(resistance: R) -> R {
    if resistance.value() > 0.0 {
        resistance.recip()
    } else {
        R::constant(0.0)
    }
}

/// A device's controlling voltages for one evaluation: those its model is evaluated at,
/// limited from the last evaluation's, and how far the applied ones lie beyond them. Every
/// value is extended linearly along that distance, so the Newton step that follows solves
/// the circuit linearized at the limited voltages.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Bias<const N: usize> {
    /// The voltages the model is evaluated at.
    pub(crate) voltages: [f64; N],
    /// How far each applied voltage lies beyond the one evaluated at.
    pub(crate) beyond: [f64; N],
}

impl<const N: usize> Bias<N> {
    /// The bias of `applied` voltages, evaluated at `voltages`, their limited values.
    pub(crate) fn new(applied: [f64; N], voltages: [f64; N]) -> Self {
        Self {
            voltages,
            beyond: std::array::from_fn(|index| applied[index] - voltages[index]),
        }
    }

    /// How far a value with derivatives `slopes` moves from the voltages evaluated at to
    /// the applied ones.
    pub(crate) fn extend<R: Real>(&self, slopes: [R; N]) -> R {
        slopes
            .into_iter()
            .zip(self.beyond)
            .fold(R::constant(0.0), |moved, (slope, beyond)| {
                moved + slope * beyond
            })
    }
}

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

    /// Adds a series resistance's current, its `conductance` times the voltage of node
    /// `outer` over node `inner`, and its derivative; nothing without one.
    pub(crate) fn series<R: Real>(
        &self,
        input: &Input<'_>,
        output: &mut Output<'_, R>,
        (outer, inner): (usize, usize),
        conductance: R,
    ) {
        if conductance.value() > 0.0 {
            let voltage = self.voltage(input, outer) - self.voltage(input, inner);
            self.current(output, outer, inner, conductance * voltage);
            self.conductance(output, (outer, inner), (outer, inner), conductance);
        }
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
