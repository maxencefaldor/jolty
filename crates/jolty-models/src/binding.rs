use jolty_numerics::{Entry, PatternBuilder};

/// A node voltage or branch current the engine solves for, by its index in the solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnknownIndex(usize);

impl UnknownIndex {
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

/// Where a terminal connects: an unknown node voltage, or `None` for ground,
/// whose voltage is zero and which has no equation.
pub type Node = Option<UnknownIndex>;

/// What an unknown measures, and so what its equation balances.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Nature {
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

impl Nature {
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
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Unknown {
    /// The unknown's name: a net's path, or an instance's path and a label, such as `V1:current`.
    pub name: String,
    /// What the unknown measures.
    pub nature: Nature,
}

/// What an instance being bound reaches: the nodes of its terminals, new internal
/// unknowns, and the matrix pattern it claims entries in. Each claim is recorded: an
/// internal unknown appended to the unknowns, named after the instance's path and its
/// label, and an entry claimed in the pattern.
#[derive(Debug)]
pub struct Binding<'a> {
    path: &'a str,
    terminals: &'a [Node],
    /// The index of the first of `unknowns`, as if they began there.
    first: usize,
    unknowns: &'a mut Vec<Unknown>,
    pattern: &'a mut PatternBuilder,
}

impl<'a> Binding<'a> {
    /// A binding for the instance at `path`, whose terminals connect to `terminals` in the
    /// order its model declares them. Internal unknowns are appended to `unknowns`.
    pub fn new(
        path: &'a str,
        terminals: &'a [Node],
        unknowns: &'a mut Vec<Unknown>,
        pattern: &'a mut PatternBuilder,
    ) -> Self {
        Self::numbered(path, terminals, (0, unknowns), pattern)
    }

    /// A binding as [`Binding::new`] makes, but for its internal unknowns, numbered from
    /// `first` as if `unknowns` began there: those of an instance bound apart from the
    /// others, whose own unknowns follow those before it.
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
