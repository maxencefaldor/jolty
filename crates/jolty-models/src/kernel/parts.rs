//! What kernels stamp their contributions through, by local index: the entries among some
//! of an instance's local unknowns, a two-terminal element's, a branch whose current is a
//! local unknown of its own, a series resistance, fixed or varying with a voltage, the
//! voltages an instance is evaluated at, limited from its last evaluation's, and what a
//! transistor adds at them.

use super::{Layout, Local, Stamp};
use crate::kernel::dual::Dual;
use crate::{Nature, NoiseSource, Real, UnknownIndex};

/// The resistance below which a resistance carries its current as an unknown, in ohms.
///
/// A conductance stamped beside a far smaller one at a node swamps it in floating point: a
/// node held only by gmin's picosiemens loses them beside a megasiemens, and its equation
/// turns singular. A current unknown keeps the two apart, as modified nodal analysis does
/// for the elements of its second group (Ho, Ruehli and Brennan, 1975); below 1 Ω, gmin
/// keeps four digits beside the resistance's conductance.
pub const SMALL_RESISTANCE: f64 = 1.0;

/// A local index no instance has: what an index out of a fixed array's range reads as, so
/// that reading it gives zero and writing it goes nowhere.
const NOWHERE: usize = usize::MAX;

/// Some of an instance's local unknowns, by index among them, with every local entry among
/// them claimed, so a kernel adds each current or charge between two of them and its
/// derivatives with respect to the voltages between others.
///
/// Two of them may be the same local unknown, as a transistor's internal drain is its drain
/// terminal when the transistor has no drain resistance; their contributions then add.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Nodes<const N: usize> {
    nodes: [usize; N],
    entries: [[usize; N]; N],
}

impl<const N: usize> Nodes<N> {
    /// Claims every entry among `nodes`, row by row.
    pub(crate) fn new(layout: &mut Layout, nodes: [usize; N]) -> Self {
        let entries = nodes.map(|row| nodes.map(|column| layout.entry(row, column)));
        Self { nodes, entries }
    }

    /// The local unknown at `index` among them.
    pub(crate) fn node(&self, index: usize) -> usize {
        self.nodes.get(index).copied().unwrap_or(NOWHERE)
    }

    /// The local entry at the row of the node at `row` and the column of the one at
    /// `column`.
    fn entry(&self, row: usize, column: usize) -> usize {
        self.entries
            .get(row)
            .and_then(|entries| entries.get(column))
            .copied()
            .unwrap_or(NOWHERE)
    }

    /// The equation of the node at `index`, as a noise source names a local unknown's.
    pub(crate) fn equation(&self, index: usize) -> Option<UnknownIndex> {
        self.nodes.get(index).map(|&local| UnknownIndex::new(local))
    }

    /// The voltage of the node at `index`.
    pub(crate) fn voltage(&self, at: &Local<'_>, index: usize) -> f64 {
        at.value(self.node(index))
    }

    /// Adds `current`, flowing through the instance from the node at `from` to the one at
    /// `to`.
    pub(crate) fn current<R: Real>(
        &self,
        stamp: &mut Stamp<'_, R>,
        from: usize,
        to: usize,
        current: R,
    ) {
        stamp.add_residual(self.node(from), current);
        stamp.add_residual(self.node(to), -current);
    }

    /// Adds `charge`, held on the node at `on` and its opposite on the one at `against`.
    pub(crate) fn charge<R: Real>(
        &self,
        stamp: &mut Stamp<'_, R>,
        on: usize,
        against: usize,
        charge: R,
    ) {
        stamp.add_charge(self.node(on), charge);
        stamp.add_charge(self.node(against), -charge);
    }

    /// Adds the derivative `conductance` of a current from `from` to `to` with respect to
    /// the voltage of `plus` over `minus`, each a node by its index among them.
    pub(crate) fn conductance<R: Real>(
        &self,
        stamp: &mut Stamp<'_, R>,
        (from, to): (usize, usize),
        (plus, minus): (usize, usize),
        conductance: R,
    ) {
        stamp.add_conductance(self.entry(from, plus), conductance);
        stamp.add_conductance(self.entry(from, minus), -conductance);
        stamp.add_conductance(self.entry(to, plus), -conductance);
        stamp.add_conductance(self.entry(to, minus), conductance);
    }

    /// Adds the derivative `capacitance` of a charge on `on` against `against` with
    /// respect to the voltage of `plus` over `minus`, each a node by its index among them.
    pub(crate) fn capacitance<R: Real>(
        &self,
        stamp: &mut Stamp<'_, R>,
        (on, against): (usize, usize),
        (plus, minus): (usize, usize),
        capacitance: R,
    ) {
        stamp.add_capacitance(self.entry(on, plus), capacitance);
        stamp.add_capacitance(self.entry(on, minus), -capacitance);
        stamp.add_capacitance(self.entry(against, plus), -capacitance);
        stamp.add_capacitance(self.entry(against, minus), capacitance);
    }
}

/// The local index of a two-terminal element's positive node among its two.
const POSITIVE: usize = 0;
/// The local index of its negative node.
const NEGATIVE: usize = 1;
/// The pair of them, the element's one branch.
const PAIR: (usize, usize) = (POSITIVE, NEGATIVE);

/// The two local unknowns of a two-terminal element and the four local entries that join
/// them. Current flows from the positive one through the element to the negative one, and
/// the element's voltage is the positive one's less the negative one's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Pair(Nodes<2>);

impl Pair {
    /// Claims the entries that join `positive` and `negative`.
    pub(crate) fn new(layout: &mut Layout, positive: usize, negative: usize) -> Self {
        Self(Nodes::new(layout, [positive, negative]))
    }

    /// The element's voltage.
    pub(crate) fn voltage(&self, at: &Local<'_>) -> f64 {
        self.0.voltage(at, POSITIVE) - self.0.voltage(at, NEGATIVE)
    }

    /// The positive and the negative local unknown, as a noise source's equations name
    /// them.
    pub(crate) fn equations(&self) -> [Option<UnknownIndex>; 2] {
        [self.0.equation(POSITIVE), self.0.equation(NEGATIVE)]
    }

    /// Adds `current`, flowing through the element.
    pub(crate) fn add_current<R: Real>(&self, stamp: &mut Stamp<'_, R>, current: R) {
        self.0.current(stamp, POSITIVE, NEGATIVE, current);
    }

    /// Adds `charge`, held on the positive node and its opposite on the negative node.
    pub(crate) fn add_charge<R: Real>(&self, stamp: &mut Stamp<'_, R>, charge: R) {
        self.0.charge(stamp, POSITIVE, NEGATIVE, charge);
    }

    /// Adds `conductance`, the derivative of the current with respect to the voltage.
    pub(crate) fn add_conductance<R: Real>(&self, stamp: &mut Stamp<'_, R>, conductance: R) {
        self.0.conductance(stamp, PAIR, PAIR, conductance);
    }

    /// Adds `capacitance`, the derivative of the charge with respect to the voltage.
    pub(crate) fn add_capacitance<R: Real>(&self, stamp: &mut Stamp<'_, R>, capacitance: R) {
        self.0.capacitance(stamp, PAIR, PAIR, capacitance);
    }
}

/// A branch from a positive local unknown to a negative one whose current, flowing through
/// it, is a local unknown of its own: the current enters both nodes' equations, and the
/// branch's own equation constrains its voltage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CurrentBranch {
    current: usize,
    nodes: [usize; 2],
    /// The entries at (each node, current) and (current, each node).
    entries: [usize; 4],
}

impl CurrentBranch {
    /// A branch from `positive` to `negative`, its current a new local unknown labelled
    /// `label`.
    pub(crate) fn new(
        layout: &mut Layout,
        label: impl Into<std::borrow::Cow<'static, str>>,
        [positive, negative]: [usize; 2],
    ) -> Self {
        let current = layout.internal(label, Nature::Current);
        Self {
            current,
            nodes: [positive, negative],
            entries: [
                layout.entry(positive, current),
                layout.entry(negative, current),
                layout.entry(current, positive),
                layout.entry(current, negative),
            ],
        }
    }

    /// The branch's local unknown, its current, and its own equation's, where an instance
    /// claims the entries of whatever else that equation reads.
    pub(crate) const fn unknown(&self) -> usize {
        self.current
    }

    /// The current through the branch.
    pub(crate) fn current(&self, at: &Local<'_>) -> f64 {
        at.value(self.current)
    }

    /// The branch's voltage, its positive node's over its negative node's.
    pub(crate) fn voltage(&self, at: &Local<'_>) -> f64 {
        let [positive, negative] = self.nodes;
        at.value(positive) - at.value(negative)
    }

    /// Adds the branch's current to its nodes' equations, and `residual` to its own
    /// equation, with the derivatives of both with respect to the current and the voltage,
    /// the residual's being 1; an instance adds the derivatives of the rest of its equation
    /// itself.
    pub(crate) fn add<R: Real>(&self, at: &Local<'_>, stamp: &mut Stamp<'_, R>, residual: R) {
        self.add_sloped(at, stamp, residual, R::constant(1.0));
    }

    /// As [`CurrentBranch::add`], the residual's derivative with respect to the branch's voltage
    /// being `slope`.
    pub(crate) fn add_sloped<R: Real>(
        &self,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
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
        let (one, current) = (R::constant(1.0), self.current(at));
        stamp.add_residual(positive, R::constant(current));
        stamp.add_residual(negative, R::constant(-current));
        stamp.add_conductance(positive_current, one);
        stamp.add_conductance(negative_current, -one);
        stamp.add_residual(self.current, residual);
        stamp.add_conductance(current_positive, slope);
        stamp.add_conductance(current_negative, -slope);
    }
}

/// A resistance in series with an instance's terminal, from the terminal's local unknown to
/// an internal node behind it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Series {
    /// No resistance: the terminal is its own node behind it.
    None,
    /// A conductance between the two nodes.
    Conductance(Pair),
    /// Below [`SMALL_RESISTANCE`], a current of its own, held by Ohm's law, and the entry
    /// at (current, current).
    Current(CurrentBranch, usize),
}

impl Series {
    /// A series `resistance` from `terminal`, and the node behind it: an internal node
    /// labelled `label` when the resistance is positive, else the terminal's own. The
    /// current through a small one is a local unknown labelled `current`.
    ///
    /// That current's [pivot](Layout::pivot) is its entry in the internal node's row, −1:
    /// its column holds besides only 1 in the terminal's row and `−R` in its own, `R`
    /// below [`SMALL_RESISTANCE`], 1 Ω, where the resistance is laid out, in the
    /// conductances and in every `G + αC` and `G + jωC` alike, as no charge reads the
    /// current. So the solver eliminates the current by the internal node's row, which
    /// leaves the terminal's row the sum of both nodes' currents, as a resistance stamped
    /// as a conductance would, and the current's own equation in the internal node's
    /// place, so that `1/R` stands on no diagonal still. A device at another temperature
    /// keeps the form its resistance took, which may grow past 1 Ω there: its pivot stays
    /// within the solver's threshold, a tenth of its column's largest, up to 10 Ω. A
    /// resistance whose value follows a voltage, which nothing bounds, claims none, as
    /// [`Series::varying`] withdraws it.
    pub(crate) fn behind<R: Real>(
        layout: &mut Layout,
        terminal: usize,
        resistance: R,
        (label, current): (&'static str, &'static str),
    ) -> (Self, usize) {
        let value = resistance.value();
        if value <= 0.0 {
            return (Self::None, terminal);
        }
        let inner = layout.internal(label, Nature::Voltage);
        let series = if value < SMALL_RESISTANCE {
            let branch = CurrentBranch::new(layout, current, [terminal, inner]);
            let row = branch.unknown();
            layout.pivot(inner, row);
            Self::Current(branch, layout.entry(row, row))
        } else {
            Self::Conductance(Pair::new(layout, terminal, inner))
        };
        (series, inner)
    }

    /// Adds the resistance's current, and its derivatives, for its [`series_value`]
    /// `value`.
    pub(crate) fn add<R: Real>(&self, value: R, at: &Local<'_>, stamp: &mut Stamp<'_, R>) {
        match self {
            Self::None => {}
            Self::Conductance(pair) => {
                pair.add_current(stamp, value * pair.voltage(at));
                pair.add_conductance(stamp, value);
            }
            Self::Current(branch, entry) => {
                let current = branch.current(at);
                let residual = R::constant(branch.voltage(at)) - value * current;
                branch.add(at, stamp, residual);
                stamp.add_conductance(*entry, -value);
            }
        }
    }

    /// The number a `resistance` stamps with in the form this one takes, which its value
    /// where it was laid out chose: its conductance where it is a conductance, and the
    /// resistance itself where its current is an unknown or there is none, whatever the
    /// resistance has become since, as at another temperature.
    pub(crate) fn value_of<R: Real>(&self, resistance: R) -> R {
        match self {
            Self::Conductance(_) => resistance.recip(),
            Self::None | Self::Current(..) => resistance,
        }
    }

    /// Claims the entries through which the resistance's current depends on the voltage of
    /// `plus` over `minus`, two local unknowns, which its value follows, and withdraws the
    /// pivot of a current of its own, as the value, which may grow past 1 Ω, no longer
    /// keeps the pivot its column's largest.
    pub(crate) fn varying(&self, layout: &mut Layout, (plus, minus): (usize, usize)) -> Varying {
        let rows = match self {
            Self::None => [NOWHERE; 2],
            Self::Conductance(Pair(nodes)) => [nodes.node(POSITIVE), nodes.node(NEGATIVE)],
            Self::Current(branch, _) => {
                layout.unpivot(branch.current);
                [branch.current, NOWHERE]
            }
        };
        let mut claim = |row: usize| {
            if row == NOWHERE {
                [NOWHERE; 2]
            } else {
                [layout.entry(row, plus), layout.entry(row, minus)]
            }
        };
        Varying {
            entries: rows.map(&mut claim),
        }
    }

    /// Adds the current through `resistance`, in this one's form, with its derivatives,
    /// the resistance moving by `slope` per volt of the voltage whose entries `varying`
    /// claims.
    pub(crate) fn add_varying<R: Real>(
        &self,
        (resistance, slope): (R, R),
        varying: &Varying,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let [[first_plus, first_minus], [second_plus, second_minus]] = varying.entries;
        match self {
            Self::None => {}
            Self::Conductance(pair) => {
                let voltage = pair.voltage(at);
                let conductance = resistance.recip();
                pair.add_current(stamp, conductance * voltage);
                pair.add_conductance(stamp, conductance);
                // i = v / r, so its derivative along the voltage r follows is −v · r′ / r².
                let along = -(slope * voltage) * conductance * conductance;
                stamp.add_conductance(first_plus, along);
                stamp.add_conductance(first_minus, -along);
                stamp.add_conductance(second_plus, -along);
                stamp.add_conductance(second_minus, along);
            }
            Self::Current(branch, entry) => {
                let current = branch.current(at);
                let residual = R::constant(branch.voltage(at)) - resistance * current;
                branch.add(at, stamp, residual);
                stamp.add_conductance(*entry, -resistance);
                let along = -slope * current;
                stamp.add_conductance(first_plus, along);
                stamp.add_conductance(first_minus, -along);
            }
        }
    }

    /// Appends the resistance's thermal noise at `temperature`, for its [`series_value`]
    /// `value`, named `name`, to `sources`.
    pub(crate) fn noise<R: Real>(
        &self,
        value: R,
        (name, temperature): (&'static str, f64),
        sources: &mut Vec<NoiseSource>,
    ) {
        sources.push(match self {
            Self::None => return,
            Self::Conductance(pair) => {
                NoiseSource::thermal(name, pair.equations(), value.value(), temperature)
            }
            Self::Current(branch, _) => NoiseSource::thermal_in_series(
                name,
                UnknownIndex::new(branch.unknown()),
                value.value(),
                temperature,
            ),
        });
    }
}

/// The entries through which a series resistance's current depends on a voltage its value
/// follows: those of each row the resistance stamps, its terminal's and its internal
/// node's, or its current's, at the voltage's two nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Varying {
    entries: [[usize; 2]; 2],
}

/// Writes `kept`, the variables an instance's memory keeps from its last evaluation, to
/// `values`, as [`Kernel::variables`](crate::Kernel::variables) does.
pub(crate) fn write_variables(kept: &[f64], values: &mut [f64]) {
    for (value, &variable) in values.iter_mut().zip(kept) {
        *value = variable;
    }
}

/// The number a series `resistance` stamps with: its conductance where it is a conductance,
/// its resistance where its current is an unknown, and nothing without one.
pub(crate) fn series_value<R: Real>(resistance: R) -> R {
    let value = resistance.value();
    if value <= 0.0 || value < SMALL_RESISTANCE {
        resistance
    } else {
        resistance.recip()
    }
}

/// An instance's controlling voltages for one evaluation: those its model is evaluated at,
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
        let mut beyond = applied;
        for (beyond, voltage) in beyond.iter_mut().zip(voltages) {
            *beyond -= voltage;
        }
        Self { voltages, beyond }
    }
}

/// What a transistor adds at one bias: currents and charges between its stamped nodes,
/// computed oriented as an n-channel's and turned by its channel's `sign`, each extended
/// from the voltages it is evaluated at to the applied ones, with its derivatives with
/// respect to the voltage across each of the `variables`' pairs of stamped nodes.
pub(crate) struct Oriented<'a, const M: usize, const N: usize> {
    pub(crate) nodes: &'a Nodes<M>,
    pub(crate) sign: f64,
    pub(crate) bias: &'a Bias<N>,
    pub(crate) variables: [(usize, usize); N],
}

impl<const M: usize, const N: usize> Oriented<'_, M, N> {
    /// What the transistor adds through a law of its equations that reads the variables
    /// at `positions` alone, each a position among its variables: a current or a charge
    /// of the law carries the derivatives with respect to those `K` variables only, in the
    /// order of `positions`.
    pub(crate) fn reading<const K: usize>(&self, positions: [usize; K]) -> Reading<'_, M, N, K> {
        Reading {
            oriented: self,
            positions,
        }
    }

    /// Adds `current`, oriented as an n-channel's, from the stamped node `from` to `to`,
    /// and sums its derivatives into `derivatives`, which writes them to `stamp`.
    #[expect(
        clippy::inline_always,
        reason = "inlined where its nodes are constants, its sums stay in registers"
    )]
    #[inline(always)]
    pub(crate) fn add_current<R: Real>(
        &self,
        stamp: &mut Stamp<'_, R>,
        derivatives: &mut Derivatives<R, M>,
        between: (usize, usize),
        current: Dual<R, N>,
    ) {
        self.reading(every())
            .add_current(stamp, derivatives, between, current);
    }

    /// Adds `charge`, oriented as an n-channel's, on the stamped node `on` against
    /// `against`, and sums its derivatives into `derivatives`, which writes them to
    /// `stamp`.
    #[expect(
        clippy::inline_always,
        reason = "inlined where its nodes are constants, its sums stay in registers"
    )]
    #[inline(always)]
    pub(crate) fn add_charge<R: Real>(
        &self,
        stamp: &mut Stamp<'_, R>,
        derivatives: &mut Derivatives<R, M>,
        between: (usize, usize),
        charge: Dual<R, N>,
    ) {
        self.reading(every())
            .add_charge(stamp, derivatives, between, charge);
    }
}

/// The position of each of `N` variables among themselves: those of a law that reads them
/// all.
fn every<const N: usize>() -> [usize; N] {
    std::array::from_fn(|position| position)
}

/// What a transistor adds through one law of its equations, which reads `K` of its `N`
/// variables, at their `positions` among them: each current and charge of the law carries
/// its derivatives with respect to those alone, which the law computes at its own width,
/// and joins the transistor's at their positions as it is added.
pub(crate) struct Reading<'a, const M: usize, const N: usize, const K: usize> {
    oriented: &'a Oriented<'a, M, N>,
    positions: [usize; K],
}

impl<const M: usize, const N: usize, const K: usize> Reading<'_, M, N, K> {
    /// The pairs of stamped nodes across which the law's variables lie, in its order; a
    /// position beyond the transistor's variables reads as no pair at all.
    pub(crate) fn variables(&self) -> [(usize, usize); K] {
        self.positions.map(|position| {
            self.oriented
                .variables
                .get(position)
                .copied()
                .unwrap_or((NOWHERE, NOWHERE))
        })
    }

    /// `value`, a number of the law with its derivatives with respect to the law's
    /// variables, extended from the voltages it is evaluated at to the applied ones, in
    /// the law's own orientation.
    pub(crate) fn extend<R: Real>(&self, value: Dual<R, K>) -> R {
        let beyond = self.positions.map(|position| {
            self.oriented
                .bias
                .beyond
                .get(position)
                .copied()
                .unwrap_or(0.0)
        });
        value.extend(beyond)
    }

    /// Adds `gradient`, the derivatives of an equation of the instance's own, a law of
    /// these variables, with respect to them, at `row`, that equation's local entries at
    /// each stamped node.
    pub(crate) fn add_conductances<R: Real>(
        &self,
        stamp: &mut Stamp<'_, R>,
        row: &[usize; M],
        gradient: [R; K],
    ) {
        let entry = |index: usize| row.get(index).copied().unwrap_or(NOWHERE);
        for ((plus, minus), derivative) in self.variables().into_iter().zip(gradient) {
            stamp.add_conductance(entry(plus), derivative);
            stamp.add_conductance(entry(minus), -derivative);
        }
    }

    /// Adds `current`, oriented as an n-channel's, from the stamped node `from` to `to`,
    /// and sums its derivatives into `derivatives`, which writes them to `stamp`.
    #[expect(
        clippy::inline_always,
        reason = "inlined where its nodes are constants, its sums stay in registers"
    )]
    #[inline(always)]
    pub(crate) fn add_current<R: Real>(
        &self,
        stamp: &mut Stamp<'_, R>,
        derivatives: &mut Derivatives<R, M>,
        (from, to): (usize, usize),
        current: Dual<R, K>,
    ) {
        let oriented = self.oriented;
        oriented
            .nodes
            .current(stamp, from, to, self.extend(current) * oriented.sign);
        let conductances = &mut derivatives.conductance;
        for (variable, derivative) in self.variables().into_iter().zip(current.gradient) {
            conductances.add((from, to), variable, derivative);
        }
    }

    /// Adds `charge`, oriented as an n-channel's, on the stamped node `on` against
    /// `against`, and sums its derivatives into `derivatives`, which writes them to
    /// `stamp`.
    #[expect(
        clippy::inline_always,
        reason = "inlined where its nodes are constants, its sums stay in registers"
    )]
    #[inline(always)]
    pub(crate) fn add_charge<R: Real>(
        &self,
        stamp: &mut Stamp<'_, R>,
        derivatives: &mut Derivatives<R, M>,
        (on, against): (usize, usize),
        charge: Dual<R, K>,
    ) {
        let oriented = self.oriented;
        oriented
            .nodes
            .charge(stamp, on, against, self.extend(charge) * oriented.sign);
        let capacitances = &mut derivatives.capacitance;
        for (variable, derivative) in self.variables().into_iter().zip(charge.gradient) {
            capacitances.add((on, against), variable, derivative);
        }
    }
}

/// The conductances and capacitances an instance adds among its `M` stamped nodes, each
/// the sum, in the order they come, of the derivatives that [`Oriented::add_current`] and
/// [`Oriented::add_charge`] add at its pair of them, so that [`Derivatives::stamp`] writes
/// each local entry among them once.
///
/// Each is the sum that [`Nodes::conductance`] and [`Nodes::capacitance`] would make in the
/// stamp one write at a time, of the same terms in the same order, and so the same bits at
/// an entry that held nothing before; at one that held a value, as a series resistance
/// leaves at the node behind it, the sum adds to the value at once.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Derivatives<R: Real, const M: usize> {
    conductance: Entries<R, M>,
    capacitance: Entries<R, M>,
}

impl<R: Real, const M: usize> Derivatives<R, M> {
    /// No derivative yet.
    pub(crate) fn new() -> Self {
        let zero = Entries([[R::constant(0.0); M]; M]);
        Self {
            conductance: zero,
            capacitance: zero,
        }
    }

    /// Adds each conductance and capacitance to its local entry of `nodes` in `stamp`.
    pub(crate) fn stamp(&self, nodes: &Nodes<M>, stamp: &mut Stamp<'_, R>) {
        let rows = self.conductance.0.iter().zip(&self.capacitance.0);
        for (row, (conductances, capacitances)) in rows.enumerate() {
            let columns = conductances.iter().zip(capacitances);
            for (column, (&conductance, &capacitance)) in columns.enumerate() {
                let entry = nodes.entry(row, column);
                stamp.add_conductance(entry, conductance);
                stamp.add_capacitance(entry, capacitance);
            }
        }
    }
}

/// One number for each pair of an instance's `M` stamped nodes, by row and column.
#[derive(Debug, Clone, Copy)]
struct Entries<R: Real, const M: usize>([[R; M]; M]);

impl<R: Real, const M: usize> Entries<R, M> {
    /// Adds `value`, the derivative of what flows from `from` to `to` with respect to the
    /// voltage of `plus` over `minus`, as [`Nodes::conductance`] adds it in a stamp.
    #[expect(
        clippy::inline_always,
        reason = "inlined where its nodes are constants, its sums stay in registers"
    )]
    #[inline(always)]
    fn add(&mut self, (from, to): (usize, usize), (plus, minus): (usize, usize), value: R) {
        for (row, column, value) in [
            (from, plus, value),
            (from, minus, -value),
            (to, plus, -value),
            (to, minus, value),
        ] {
            if let Some(entry) = self.0.get_mut(row).and_then(|row| row.get_mut(column)) {
                *entry += value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    /// The residuals, scales and charges of 4 local unknowns, and the conductances and
    /// capacitances of the entries among them, that `add` writes into a stamp.
    fn stamped(add: impl FnOnce(&mut Stamp<'_, f64>)) -> [Vec<u64>; 5] {
        let mut rows = [vec![0.0; 4], vec![0.0; 4], vec![0.0; 4]];
        let mut entries = [vec![0.0; 16], vec![0.0; 16]];
        let [residual, scale, charge] = &mut rows;
        let [conductance, capacitance] = &mut entries;
        add(&mut Stamp::zeroed(
            [residual, scale, charge],
            [conductance, capacitance],
        ));
        let [residual, scale, charge] = rows;
        let [conductance, capacitance] = entries;
        [residual, scale, charge, conductance, capacitance]
            .map(|values| values.iter().map(|value| value.to_bits()).collect())
    }

    proptest! {
        /// Summing a transistor's derivatives in place and writing each entry once stamps
        /// the same bits as writing each derivative as it comes.
        #[test]
        fn summed_derivatives_stamp_what_each_write_does(
            numbers in proptest::collection::vec(-1e3..1e3f64, 4 * 8),
            beyond in proptest::array::uniform3(-1e-3..1e-3f64),
            sign in proptest::bool::ANY,
        ) {
            let mut layout = Layout::new(4);
            let nodes = Nodes::new(&mut layout, [1, 3, 0, 2]);
            let bias = Bias { voltages: [0.0; 3], beyond };
            let added = Oriented {
                nodes: &nodes,
                sign: if sign { 1.0 } else { -1.0 },
                bias: &bias,
                variables: [(0, 3), (2, 3), (1, 3)],
            };
            // Currents and charges between pairs of the nodes, some pairs again.
            let pairs = [(2, 3), (1, 2), (1, 3), (0, 3), (0, 2), (0, 1), (1, 3), (2, 3)];
            let terms: Vec<_> = numbers
                .as_chunks::<4>()
                .0
                .iter()
                .zip(pairs)
                .enumerate()
                .map(|(index, (&[value, gs, ds, bs], pair))| {
                    (index % 3 == 1, pair, Dual { value, gradient: [gs, ds, bs] })
                })
                .collect();
            let written = stamped(|stamp| {
                for &(charge, (from, to), term) in &terms {
                    let value = term.extend(bias.beyond) * added.sign;
                    let derivatives = added.variables.into_iter().zip(term.gradient);
                    if charge {
                        nodes.charge(stamp, from, to, value);
                        for (variable, derivative) in derivatives {
                            nodes.capacitance(stamp, (from, to), variable, derivative);
                        }
                    } else {
                        nodes.current(stamp, from, to, value);
                        for (variable, derivative) in derivatives {
                            nodes.conductance(stamp, (from, to), variable, derivative);
                        }
                    }
                }
            });
            let summed = stamped(|stamp| {
                let mut derivatives = Derivatives::new();
                for &(charge, pair, term) in &terms {
                    if charge {
                        added.add_charge(stamp, &mut derivatives, pair, term);
                    } else {
                        added.add_current(stamp, &mut derivatives, pair, term);
                    }
                }
                derivatives.stamp(&nodes, stamp);
            });
            prop_assert_eq!(written, summed);
        }

        /// A law that reads some of a transistor's variables, in their order, adds what the
        /// same numbers embedded among all its variables add, their other derivatives zero:
        /// the same values, though a zero may differ in its sign. A law that reads them in
        /// another order sums its derivatives in its own, and may round differently.
        #[test]
        fn a_law_of_some_variables_adds_what_it_does_embedded_among_all(
            numbers in proptest::collection::vec(-1e3..1e3f64, 3 * 6),
            beyond in proptest::array::uniform3(-1e-3..1e-3f64),
            sign in proptest::bool::ANY,
        ) {
            let mut layout = Layout::new(4);
            let nodes = Nodes::new(&mut layout, [1, 3, 0, 2]);
            let bias = Bias { voltages: [0.0; 3], beyond };
            let added = Oriented {
                nodes: &nodes,
                sign: if sign { 1.0 } else { -1.0 },
                bias: &bias,
                variables: [(0, 3), (2, 3), (1, 3)],
            };
            let law = [0, 2];
            let pairs = [(2, 3), (1, 2), (0, 3), (0, 1), (1, 3), (2, 3)];
            let terms: Vec<_> = numbers
                .as_chunks::<3>()
                .0
                .iter()
                .zip(pairs)
                .enumerate()
                .map(|(index, (&[value, first, second], pair))| {
                    (index % 2 == 1, pair, Dual { value, gradient: [first, second] })
                })
                .collect();
            let stamped_values = |add: &dyn Fn(&mut Stamp<'_, f64>)| {
                stamped(add).map(|bits| bits.into_iter().map(f64::from_bits).collect::<Vec<_>>())
            };
            let read = stamped_values(&|stamp| {
                let mut derivatives = Derivatives::new();
                let reading = added.reading(law);
                for &(charge, pair, term) in &terms {
                    if charge {
                        reading.add_charge(stamp, &mut derivatives, pair, term);
                    } else {
                        reading.add_current(stamp, &mut derivatives, pair, term);
                    }
                }
                derivatives.stamp(&nodes, stamp);
            });
            let embedded = stamped_values(&|stamp| {
                let mut derivatives = Derivatives::new();
                for &(charge, pair, term) in &terms {
                    if charge {
                        added.add_charge(stamp, &mut derivatives, pair, term.embed(law));
                    } else {
                        added.add_current(stamp, &mut derivatives, pair, term.embed(law));
                    }
                }
                derivatives.stamp(&nodes, stamp);
            });
            prop_assert_eq!(read, embedded);
        }
    }
}
