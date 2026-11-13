use jolty_numerics::{Entry, PatternBuilder};

use crate::Real;

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
    /// A thermal net's temperature above the ambient, in kelvins, whose equation sums the
    /// heat flows leaving the net, in watts, as a node's sums currents.
    Temperature,
    /// A device's dimensionless state, such as a memristor's, whose equation is the
    /// device's own: how fast the state moves, or what it lags.
    State,
}

impl Quantity {
    /// Whether the unknown is a current rather than a voltage.
    #[must_use]
    pub const fn is_current(self) -> bool {
        matches!(self, Self::Current | Self::Displacement)
    }

    /// Whether the unknown's equation sums currents, or heat flows, rather than
    /// constraining a voltage.
    #[must_use]
    pub const fn sums_currents(self) -> bool {
        matches!(self, Self::Voltage | Self::Displacement | Self::Temperature)
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

    /// Simulates the device at `temperature`, in kelvins, its own.
    pub(crate) const fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
    }

    /// The instance's path.
    #[must_use]
    pub const fn path(&self) -> &str {
        self.path
    }

    /// The node of every terminal, in the model's terminal order.
    #[must_use]
    pub const fn terminals(&self) -> &[Node] {
        self.terminals
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

    /// The node behind a terminal's series `resistance`: an internal node labelled `label`
    /// when the resistance is positive, else the terminal's own node.
    pub(crate) fn behind<R: Real>(&mut self, terminal: Node, resistance: R, label: &str) -> Node {
        if resistance.value() > 0.0 {
            Some(self.internal(label, Quantity::Voltage))
        } else {
            terminal
        }
    }

    /// Claims the matrix entry at `row` and `column`, or none when either is the reference.
    pub fn entry(&mut self, row: Node, column: Node) -> Option<Entry> {
        Some(self.pattern.entry(row?.index(), column?.index()))
    }
}
