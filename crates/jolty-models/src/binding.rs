use jolty_numerics::{Entry, PatternBuilder};

/// Index of a node voltage or branch current unknown in the global solution vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnknownIndex(usize);

impl UnknownIndex {
    /// Creates an unknown reference for `index` in the solution vector.
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the index of the unknown in the solution vector.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Terminal connection target: an unknown node voltage, or `None` for ground (zero voltage
/// with no equation).
pub type Node = Option<UnknownIndex>;

/// Physical quantity represented by an unknown and the corresponding balance equation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Nature {
    /// Node voltage whose equation sums currents leaving the node.
    Voltage,
    /// Branch current whose equation constrains the voltage across the branch.
    Current,
    /// Device-internal displacement current whose equation sums currents for a
    /// non-charge-based capacitance (such as a MOSFET gate oxide in Meyer's model).
    Displacement,
    /// Thermal net temperature rise above ambient, in kelvins, whose equation sums heat
    /// flows leaving the net in watts.
    Temperature,
    /// Dimensionless internal device state, such as a memristor state variable, governed by
    /// a model-specific rate or lag equation.
    State,
}

impl Nature {
    /// Returns whether the unknown represents a current rather than a voltage.
    #[must_use]
    pub const fn is_current(self) -> bool {
        matches!(self, Self::Current | Self::Displacement)
    }

    /// Returns whether the unknown equation sums currents or heat flows rather than
    /// constraining a voltage.
    #[must_use]
    pub const fn sums_currents(self) -> bool {
        matches!(self, Self::Voltage | Self::Displacement | Self::Temperature)
    }
}

/// Name and physical quantity of a circuit unknown.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Unknown {
    /// Unknown name: either a net path or an instance path and label, such as `V1:current`.
    pub name: String,
    /// Physical quantity measured by the unknown.
    pub nature: Nature,
}

/// Binding context passed to a device instance during circuit elaboration.
///
/// Exposes terminal nodes, allocates internal unknowns named from the instance path and
/// label, and claims nonzero matrix entries in the sparsity pattern.
#[derive(Debug)]
pub struct Binding<'a> {
    path: &'a str,
    terminals: &'a [Node],
    /// Base index corresponding to element zero of `unknowns`.
    first: usize,
    unknowns: &'a mut Vec<Unknown>,
    pattern: &'a mut PatternBuilder,
}

impl<'a> Binding<'a> {
    /// Creates a binding for the instance at `path` with terminal connections `terminals` in
    /// the model's declared terminal order, appending internal unknowns to `unknowns`.
    pub fn new(
        path: &'a str,
        terminals: &'a [Node],
        unknowns: &'a mut Vec<Unknown>,
        pattern: &'a mut PatternBuilder,
    ) -> Self {
        Self::numbered(path, terminals, (0, unknowns), pattern)
    }

    /// Creates a binding identical to [`Binding::new`], except internal unknowns are numbered
    /// starting at `first` for instances bound independently after earlier unknowns.
    pub const fn numbered(
        path: &'a str,
        terminals: &'a [Node],
        (first, unknowns): (usize, &'a mut Vec<Unknown>),
        pattern: &'a mut PatternBuilder,
    ) -> Self {
        Self {
            path,
            terminals,
            first,
            unknowns,
            pattern,
        }
    }

    /// Returns the hierarchical instance path.
    #[must_use]
    pub const fn path(&self) -> &str {
        self.path
    }

    /// Returns the terminal nodes in the model's declared terminal order.
    #[must_use]
    pub const fn terminals(&self) -> &[Node] {
        self.terminals
    }

    /// Adds an internal unknown, named after the instance and `label`.
    pub fn internal(&mut self, label: &str, nature: Nature) -> UnknownIndex {
        self.unknowns.push(Unknown {
            name: format!("{}:{label}", self.path),
            nature,
        });
        UnknownIndex(self.first + self.unknowns.len() - 1)
    }

    /// Claims the matrix entry at `row` and `column`, or none when either is ground.
    pub fn entry(&mut self, row: Node, column: Node) -> Option<Entry> {
        let (row, column) = (row?.index(), column?.index());
        Some(self.pattern.entry(row, column))
    }
}
