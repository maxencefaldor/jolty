//! The structure of a device: which of its terminals it joins, and how.
//!
//! Structural checks read it to find what makes a circuit's equations singular before any
//! analysis runs, such as a node with no path to the reference at DC or a loop of voltage
//! sources, and schematics read it to draw each device. A terminal on no branch only senses
//! its node, as a controlled source's control inputs do.

/// How a branch of a device joins two of its terminals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BranchKind {
    /// Its current depends on its voltage, and flows at DC: a resistor, a diode's junction.
    Conductive,
    /// It stores charge, and is open at DC: a capacitor.
    Capacitive,
    /// It stores flux, is a short at DC, and its current is an unknown: an inductor.
    Inductive,
    /// It fixes the voltage between its terminals: an independent or controlled voltage
    /// source.
    VoltageSource,
    /// It fixes the current through it: an independent or controlled current source.
    CurrentSource,
}

impl BranchKind {
    /// Whether the branch carries current at DC with its voltage free to settle, so that it
    /// gives the nodes it joins a DC path to each other.
    #[must_use]
    pub const fn conducts_at_dc(self) -> bool {
        matches!(
            self,
            Self::Conductive | Self::Inductive | Self::VoltageSource
        )
    }

    /// Whether the branch fixes its voltage at DC, so that a loop of such branches over-
    /// constrains the circuit.
    #[must_use]
    pub const fn fixes_dc_voltage(self) -> bool {
        matches!(self, Self::Inductive | Self::VoltageSource)
    }
}

/// A branch of a device between two of its terminals, by their index among the model's
/// terminals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Branch {
    /// The terminal the branch leaves.
    pub from: usize,
    /// The terminal the branch reaches.
    pub to: usize,
    /// How the branch joins them.
    pub kind: BranchKind,
}

impl Branch {
    /// The branch from terminal `from` to terminal `to`.
    #[must_use]
    pub const fn new(from: usize, to: usize, kind: BranchKind) -> Self {
        Self { from, to, kind }
    }
}

/// The terminals of one device and the branches between them, which may depend on its
/// parameters, as a behavioral source's terminals depend on the voltages it reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interface {
    /// The terminals' names, in the order [`Binding::terminal`](crate::Binding::terminal)
    /// indexes them.
    pub terminals: Vec<String>,
    /// The branches between the terminals; a terminal on none only senses its node.
    pub branches: Vec<Branch>,
}

impl Interface {
    /// An interface of `terminals` and `branches`.
    #[must_use]
    pub fn new(terminals: &[&str], branches: &[Branch]) -> Self {
        Self {
            terminals: terminals.iter().map(|&name| name.to_owned()).collect(),
            branches: branches.to_vec(),
        }
    }
}
