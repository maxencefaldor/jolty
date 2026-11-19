//! Structural connectivity of a device across its terminals.
//!
//! Structural checks inspect branch connectivity to detect singular circuit topologies
//! before simulation, such as floating DC nodes or voltage-source loops. Schematic
//! renderers read the same topology to draw each device. Any terminal not attached to a
//! branch only senses its node voltage, as with controlled-source control inputs.

/// Electrical behavior of a device branch joining two terminals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BranchKind {
    /// Conductive branch whose current depends on voltage and flows at DC, such as a
    /// resistor or diode junction.
    Conductive,
    /// Capacitive branch that stores charge and behaves as an open circuit at DC.
    Capacitive,
    /// Inductive branch that stores magnetic flux, acts as a DC short circuit, and adds an
    /// unknown branch current.
    Inductive,
    /// Voltage-source branch that constrains the voltage between its terminals.
    VoltageSource,
    /// Current-source branch that constrains the current flowing through its terminals.
    CurrentSource,
}

impl BranchKind {
    /// Returns whether the branch conducts current at DC, providing a DC path between its
    /// terminals.
    #[must_use]
    pub const fn conducts_at_dc(self) -> bool {
        matches!(
            self,
            Self::Conductive | Self::Inductive | Self::VoltageSource
        )
    }

    /// Returns whether the branch constrains its terminal voltage at DC so that a loop of
    /// voltage-fixing branches over-constrains the circuit.
    #[must_use]
    pub const fn fixes_dc_voltage(self) -> bool {
        matches!(self, Self::Inductive | Self::VoltageSource)
    }
}

/// Device branch connecting two terminals identified by their terminal indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Branch {
    /// Index of the originating terminal.
    pub from: usize,
    /// Index of the destination terminal.
    pub to: usize,
    /// Electrical behavior of the branch.
    pub kind: BranchKind,
}

impl Branch {
    /// Creates a branch from terminal `from` to terminal `to` with behavior `kind`.
    #[must_use]
    pub const fn new(from: usize, to: usize, kind: BranchKind) -> Self {
        Self { from, to, kind }
    }
}

/// Terminal declarations and internal branch topology for a device instance.
///
/// The interface may depend on instance parameters, as with behavioral sources whose
/// terminals include every voltage node referenced by the expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interface {
    /// Terminal names in the order expected by
    /// [`Binding::terminals`](crate::Binding::terminals).
    pub terminals: Vec<String>,
    /// Branches connecting terminals; terminals not on any branch only sense node voltage.
    pub branches: Vec<Branch>,
    /// Indices of thermal terminals that connect to thermal nets or ambient ground.
    ///
    /// A thermal net unknown represents temperature rise above ambient and balances heat
    /// flow; all other terminals are electrical.
    pub thermal: Vec<usize>,
}

impl Interface {
    /// Creates an interface with electrical `terminals` and `branches`.
    #[must_use]
    pub fn new(terminals: &[&str], branches: &[Branch]) -> Self {
        Self {
            terminals: terminals.iter().map(|&name| name.to_owned()).collect(),
            branches: branches.to_vec(),
            thermal: Vec::new(),
        }
    }

    /// Marks every terminal in the interface as thermal and returns the updated interface.
    #[must_use]
    pub fn thermal(mut self) -> Self {
        self.thermal = (0..self.terminals.len()).collect();
        self
    }
}
