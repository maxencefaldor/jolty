use jolty_numerics::{Entry, PatternBuilder, Replay};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnknownInfo {
    /// The unknown's name: a net's path, or an instance's path and a label, such as `V1:current`.
    pub name: String,
    /// What the unknown measures.
    pub quantity: Quantity,
}

/// What a device being instantiated reaches: the nodes of its terminals, new internal
/// unknowns, the matrix pattern it claims entries in, and the temperature it is simulated
/// at.
///
/// A binding either records each claim, of an internal unknown or of a matrix entry, as an
/// instance is first made, or [replays](Binding::replaying) the claims recorded, as the
/// instance is made again from other numbers: it hands back the unknowns and the entries
/// recorded, in the same order, and [`Binding::replayed`] says whether the device claimed
/// the very same ones.
#[derive(Debug)]
pub struct Binding<'a> {
    path: &'a str,
    terminals: &'a [Node],
    claims: Claims<'a>,
    temperature: f64,
}

/// Where a binding's claims go.
#[derive(Debug)]
enum Claims<'a> {
    /// Each claim is recorded: an internal unknown appended to `unknowns`, numbered as if
    /// they began at `first`, and an entry claimed in `pattern`.
    Record {
        first: usize,
        unknowns: &'a mut Vec<UnknownInfo>,
        pattern: &'a mut PatternBuilder,
    },
    /// Each claim is handed back what was recorded: the internal unknowns `internals`, the
    /// first at index `first`, of which `claimed` are claimed so far, and the entries of
    /// `entries`.
    Replay {
        first: usize,
        internals: &'a [UnknownInfo],
        claimed: usize,
        entries: Replay<'a>,
        /// Whether every internal unknown claimed so far is the one recorded.
        matched: bool,
    },
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
        Self::numbered(path, terminals, (0, unknowns), pattern, temperature)
    }

    /// A binding as [`Binding::new`] makes, but for its internal unknowns, numbered from
    /// `first` as if `unknowns` began there: those of an instance made apart from the
    /// others, whose own unknowns follow those before it.
    pub const fn numbered(
        path: &'a str,
        terminals: &'a [Node],
        (first, unknowns): (usize, &'a mut Vec<UnknownInfo>),
        pattern: &'a mut PatternBuilder,
        temperature: f64,
    ) -> Self {
        Self {
            path,
            terminals,
            claims: Claims::Record {
                first,
                unknowns,
                pattern,
            },
            temperature,
        }
    }

    /// A binding for the instance at `path`, as [`Binding::new`] makes, that hands back
    /// the claims recorded when it was first made: the internal unknowns `internals`, the
    /// first of them at index `first` among the unknowns, and the entries of `entries`, in
    /// the order they were claimed.
    #[must_use]
    pub const fn replaying(
        path: &'a str,
        terminals: &'a [Node],
        (first, internals): (usize, &'a [UnknownInfo]),
        entries: Replay<'a>,
        temperature: f64,
    ) -> Self {
        Self {
            path,
            terminals,
            claims: Claims::Replay {
                first,
                internals,
                claimed: 0,
                entries,
                matched: true,
            },
            temperature,
        }
    }

    /// Whether the claims made through a binding that replays were those recorded, each
    /// internal unknown of the same name and quantity and each entry at the same position,
    /// in the same order, none left over and none more; always, for a binding that records.
    /// A device whose claims were not those recorded has another structure than the device
    /// first made, and its unknowns and entries are not those of the pattern recorded.
    #[must_use]
    pub const fn replayed(&self) -> bool {
        match &self.claims {
            Claims::Record { .. } => true,
            Claims::Replay {
                internals,
                claimed,
                entries,
                matched,
                ..
            } => *matched && *claimed == internals.len() && entries.replayed(),
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
        match &mut self.claims {
            Claims::Record {
                first, unknowns, ..
            } => {
                unknowns.push(UnknownInfo {
                    name: format!("{}:{label}", self.path),
                    quantity,
                });
                Unknown(*first + unknowns.len() - 1)
            }
            Claims::Replay {
                first,
                internals,
                claimed,
                matched,
                ..
            } => {
                let index = *claimed;
                *claimed += 1;
                let recorded = internals.get(index).is_some_and(|info| {
                    info.quantity == quantity
                        && info
                            .name
                            .strip_prefix(self.path)
                            .and_then(|rest| rest.strip_prefix(':'))
                            == Some(label)
                });
                *matched &= recorded;
                Unknown(*first + index)
            }
        }
    }

    /// Claims the matrix entry at `row` and `column`, or none when either is the reference.
    pub fn entry(&mut self, row: Node, column: Node) -> Option<Entry> {
        let (row, column) = (row?.index(), column?.index());
        Some(match &mut self.claims {
            Claims::Record { pattern, .. } => pattern.entry(row, column),
            Claims::Replay { entries, .. } => entries.entry(row, column),
        })
    }
}
