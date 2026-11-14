//! What kernels stamp their contributions through, by local index: the entries among some
//! of an instance's local unknowns, a two-terminal element's, a branch whose current is a
//! local unknown of its own, a series resistance, fixed or varying with a voltage, the
//! voltages an instance is evaluated at, limited from its last evaluation's, and what a
//! transistor adds at them.

use super::{Layout, Local, Stamp};
use crate::dual::Dual;
use crate::{NoiseSource, Quantity, Real, Unknown};

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
    pub(crate) fn equation(&self, index: usize) -> Option<Unknown> {
        self.nodes.get(index).map(|&local| Unknown::new(local))
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
    pub(crate) fn equations(&self) -> [Option<Unknown>; 2] {
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
        let current = layout.internal(label, Quantity::Current);
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
        let inner = layout.internal(label, Quantity::Voltage);
        let series = if value < SMALL_RESISTANCE {
            let branch = CurrentBranch::new(layout, current, [terminal, inner]);
            let row = branch.unknown();
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
    /// `plus` over `minus`, two local unknowns, which its value follows.
    pub(crate) fn varying(&self, layout: &mut Layout, (plus, minus): (usize, usize)) -> Varying {
        let rows = match self {
            Self::None => [NOWHERE; 2],
            Self::Conductance(Pair(nodes)) => [nodes.node(POSITIVE), nodes.node(NEGATIVE)],
            Self::Current(branch, _) => [branch.current, NOWHERE],
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
                Unknown::new(branch.unknown()),
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
    /// Adds `current`, oriented as an n-channel's, from the stamped node `from` to `to`,
    /// with its derivatives.
    pub(crate) fn current<R: Real>(
        &self,
        stamp: &mut Stamp<'_, R>,
        (from, to): (usize, usize),
        current: Dual<R, N>,
    ) {
        let nodes = self.nodes;
        nodes.current(
            stamp,
            from,
            to,
            current.extend(self.bias.beyond) * self.sign,
        );
        for (variable, derivative) in self.variables.into_iter().zip(current.gradient) {
            nodes.conductance(stamp, (from, to), variable, derivative);
        }
    }

    /// Adds `charge`, oriented as an n-channel's, on the stamped node `on` against
    /// `against`, with its derivatives.
    pub(crate) fn charge<R: Real>(
        &self,
        stamp: &mut Stamp<'_, R>,
        (on, against): (usize, usize),
        charge: Dual<R, N>,
    ) {
        let nodes = self.nodes;
        nodes.charge(
            stamp,
            on,
            against,
            charge.extend(self.bias.beyond) * self.sign,
        );
        for (variable, derivative) in self.variables.into_iter().zip(charge.gradient) {
            nodes.capacitance(stamp, (on, against), variable, derivative);
        }
    }
}
