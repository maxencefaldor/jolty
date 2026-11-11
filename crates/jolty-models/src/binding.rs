use jolty_numerics::{Entry, PatternBuilder};

use crate::{Input, Output};

/// A node voltage or branch current the engine solves for, by its index in the solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Unknown(usize);

impl Unknown {
    /// The unknown at `index` in the solution vector.
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// The unknown's index in the solution vector.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Where a terminal connects: an unknown node voltage, or `None` for the reference node,
/// whose voltage is zero and which has no equation.
pub type Node = Option<Unknown>;

/// What an unknown measures, and so what its equation balances.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Quantity {
    /// A node voltage, whose equation sums the currents leaving the node.
    Voltage,
    /// A branch current, whose equation constrains the branch's voltage.
    Current,
    /// A displacement current a device tracks within itself, whose equation sums currents:
    /// the current through a capacitance no charge describes, such as a transistor's gate
    /// oxide in Meyer's model, which the device's own currents follow.
    Displacement,
}

impl Quantity {
    /// Whether the unknown is a current rather than a voltage.
    #[must_use]
    pub const fn is_current(self) -> bool {
        matches!(self, Self::Current | Self::Displacement)
    }

    /// Whether the unknown's equation sums currents, rather than constraining a voltage.
    #[must_use]
    pub const fn sums_currents(self) -> bool {
        matches!(self, Self::Voltage | Self::Displacement)
    }
}

/// The name and quantity of an unknown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownInfo {
    /// The unknown's name: a net's path, or an instance's path and a label, such as `V1:current`.
    pub name: String,
    /// What the unknown measures.
    pub quantity: Quantity,
}

/// What a device being instantiated reaches: the nodes of its terminals, new internal
/// unknowns, the matrix pattern it claims entries in, and the temperature it is simulated
/// at.
#[derive(Debug)]
pub struct Binding<'a> {
    path: &'a str,
    terminals: &'a [Node],
    unknowns: &'a mut Vec<UnknownInfo>,
    pattern: &'a mut PatternBuilder,
    temperature: f64,
}

impl<'a> Binding<'a> {
    /// A binding for the instance at `path`, whose terminals connect to `terminals` in the
    /// order its model declares them, simulated at `temperature`, in kelvins. Internal
    /// unknowns are appended to `unknowns`.
    pub fn new(
        path: &'a str,
        terminals: &'a [Node],
        unknowns: &'a mut Vec<UnknownInfo>,
        pattern: &'a mut PatternBuilder,
        temperature: f64,
    ) -> Self {
        Self {
            path,
            terminals,
            unknowns,
            pattern,
            temperature,
        }
    }

    /// The temperature the device is simulated at, in kelvins.
    #[must_use]
    pub const fn temperature(&self) -> f64 {
        self.temperature
    }

    /// The node of the terminal at `index` in the model's terminal order.
    ///
    /// # Panics
    ///
    /// Panics if the model has no terminal at `index`.
    #[must_use]
    pub fn terminal(&self, index: usize) -> Node {
        self.terminals[index]
    }

    /// Adds an internal unknown, named after the instance and `label`.
    pub fn internal(&mut self, label: &str, quantity: Quantity) -> Unknown {
        self.unknowns.push(UnknownInfo {
            name: format!("{}:{label}", self.path),
            quantity,
        });
        Unknown(self.unknowns.len() - 1)
    }

    /// Claims the matrix entry at `row` and `column`, or none when either is the reference.
    pub fn entry(&mut self, row: Node, column: Node) -> Option<Entry> {
        Some(self.pattern.entry(row?.index(), column?.index()))
    }
}

/// The two nodes of a two-terminal element and the four matrix entries that join them.
///
/// Current flows from the positive node through the element to the negative node, and the
/// element's voltage is the positive node's voltage minus the negative node's.
#[derive(Debug, Clone, Copy)]
pub struct TerminalPair {
    positive: Node,
    negative: Node,
    entries: [Option<Entry>; 4],
}

impl TerminalPair {
    /// Claims the entries that join `positive` and `negative`.
    pub fn new(binding: &mut Binding<'_>, positive: Node, negative: Node) -> Self {
        let entries = [
            binding.entry(positive, positive),
            binding.entry(positive, negative),
            binding.entry(negative, positive),
            binding.entry(negative, negative),
        ];
        Self {
            positive,
            negative,
            entries,
        }
    }

    /// The element's voltage.
    #[must_use]
    pub fn voltage(&self, input: &Input<'_>) -> f64 {
        input.value(self.positive) - input.value(self.negative)
    }

    /// Adds `current`, flowing through the element, to the two nodes' equations.
    pub fn add_current(&self, output: &mut Output<'_>, current: f64) {
        output.add_residual(self.positive, current);
        output.add_residual(self.negative, -current);
    }

    /// Adds `charge`, held on the positive node and its opposite on the negative node.
    pub fn add_charge(&self, output: &mut Output<'_>, charge: f64) {
        output.add_charge(self.positive, charge);
        output.add_charge(self.negative, -charge);
    }

    /// Adds `conductance`, the derivative of the current with respect to the voltage.
    pub fn add_conductance(&self, output: &mut Output<'_>, conductance: f64) {
        let [
            positive_positive,
            positive_negative,
            negative_positive,
            negative_negative,
        ] = self.entries;
        output.add_conductance(positive_positive, conductance);
        output.add_conductance(positive_negative, -conductance);
        output.add_conductance(negative_positive, -conductance);
        output.add_conductance(negative_negative, conductance);
    }

    /// Adds `capacitance`, the derivative of the charge with respect to the voltage.
    pub fn add_capacitance(&self, output: &mut Output<'_>, capacitance: f64) {
        let [
            positive_positive,
            positive_negative,
            negative_positive,
            negative_negative,
        ] = self.entries;
        output.add_capacitance(positive_positive, capacitance);
        output.add_capacitance(positive_negative, -capacitance);
        output.add_capacitance(negative_positive, -capacitance);
        output.add_capacitance(negative_negative, capacitance);
    }
}
