//! Kernels: what a device model computes for one instance, apart from every other.
//!
//! A [`Kernel`] is a device model as a pure function of one instance: its parameters,
//! prepared once into the numbers its temperature sets, [`Kernel::Prepared`]; what it
//! remembers from one evaluation to the next, [`Kernel::Memory`]; and its local unknowns,
//! which a [`Local`] reads through the instance's node map. It writes what it contributes to
//! the circuit's equations into a [`Stamp`], by local index: the residual of each local
//! unknown's equation, the largest magnitude of a term of it and its charge, and the
//! conductance and capacitance of each local entry. The local unknowns are the instance's
//! terminals, then the internal unknowns its [`Layout`] claims; its local entries are the
//! pairs of them the layout claims, each once.
//!
//! A kernel allocates nothing, never panics, and knows nothing of threads, batches or the
//! global matrix. The groups of the instances of a kernel ([`crate::Groups`]) evaluate it,
//! serially or in parallel, each instance into a block of a slab of its own, which a gather
//! in a fixed order assembles.

#![cfg_attr(
    not(test),
    deny(
        clippy::indexing_slicing,
        clippy::panic,
        reason = "a kernel never panics, and reads fixed arrays through `get`"
    )
)]
pub(crate) mod parts;

use std::borrow::Cow;
use std::fmt;

use schemars::JsonSchema;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::{Choice, Interface, Level, NoiseSource, ParameterError, Quantity, Real, Regions};

/// A device model computed one instance at a time, as a pure function of the instance's
/// prepared parameters, its memory and its local unknowns.
///
/// Its evaluation allocates nothing, never panics, and knows nothing of threads, batches or
/// the global matrix.
pub trait Kernel: Send + Sync + 'static {
    /// The model's name, such as `diode`.
    const NAME: &'static str;

    /// Whether its instances are a foundry's, which ngspice simulates with the foundry's
    /// models, as [`crate::Model::foundry`] says.
    const FOUNDRY: bool = false;

    /// The parameters an instance's parameter map deserializes into, each number an `R`,
    /// which write as they read and which a JSON schema describes.
    type Parameters<R: Real>: DeserializeOwned
        + Serialize
        + JsonSchema
        + Clone
        + fmt::Debug
        + Send
        + Sync
        + 'static;

    /// The structure an instance's parameters give it: the internal unknowns and the
    /// entries it claims, and the local index of each. Instances of one kernel and one
    /// shape are evaluated together.
    ///
    /// Two shapes that differ claim different layouts, so that the unknowns and the entries
    /// of an instance say what its shape is.
    type Shape: Clone + fmt::Debug + PartialEq + Send + Sync + 'static;

    /// The numbers an instance's parameters set at its temperature, ready to evaluate.
    type Prepared<R: Real>: Clone + fmt::Debug + Send + Sync;

    /// What an instance remembers from one evaluation to the next, such as the junction
    /// voltage its next step is limited from, and the room it evaluates in, sized once;
    /// [`Kernel::memory`] makes it as it is before the first evaluation.
    type Memory: Clone + fmt::Debug + Default + Send + Sync;

    /// The terminals and branches of an instance with `parameters`.
    fn interface<R: Real>(parameters: &Self::Parameters<R>) -> Interface;

    /// The numbers of an instance with `parameters` at `temperature`, in kelvins.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when a parameter is out of range.
    fn prepare<R: Real>(
        parameters: &Self::Parameters<R>,
        temperature: R,
    ) -> Result<Self::Prepared<R>, ParameterError>;

    /// Claims in `layout`, after the instance's terminals, the internal unknowns and the
    /// entries of an instance with `parameters`, whose numbers [`Kernel::prepare`] has
    /// checked and prepared into `prepared`, and returns its shape.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when a parameter is out of range.
    fn layout<R: Real>(
        parameters: &Self::Parameters<R>,
        prepared: &Self::Prepared<R>,
        layout: &mut Layout,
    ) -> Result<Self::Shape, ParameterError>;

    /// What an instance of `shape` with the numbers `prepared` remembers before its first
    /// evaluation: the memory's default, unless it sizes room of its own.
    #[must_use]
    fn memory<R: Real>(shape: &Self::Shape, prepared: &Self::Prepared<R>) -> Self::Memory {
        let _ = (shape, prepared);
        Self::Memory::default()
    }

    /// Writes into `stamp` what an instance of `shape`, with the numbers `prepared`,
    /// contributes to the circuit's equations at `at`, from what it remembers, `memory`,
    /// which it updates.
    fn evaluate<R: Real>(
        shape: &Self::Shape,
        prepared: &Self::Prepared<R>,
        memory: &mut Self::Memory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    );

    /// Forgets what of `memory` depends on the temperature, when an instance is prepared
    /// again at another, as a device that heats is between two evaluations: nothing,
    /// unless its memory starts from its prepared numbers.
    fn reheat(memory: &mut Self::Memory) {
        let _ = memory;
    }

    /// The names of the quantities an instance computes on its way to its currents that
    /// neither its terminals, nor its unknowns, nor their derivatives show, such as a
    /// transistor's threshold voltage, in the order of [`Kernel::variables`].
    #[must_use]
    fn variable_names() -> &'static [&'static str] {
        &[]
    }

    /// Writes the value of each of [`Kernel::variable_names`] at the last evaluation, which
    /// `memory` keeps, in SI base units, to `values`.
    fn variables(memory: &Self::Memory, values: &mut [f64]) {
        let _ = (memory, values);
    }

    /// The regions of operation its equations distinguish, which the variable
    /// [`REGION`](crate::REGION) among [`Kernel::variable_names`] numbers, and the
    /// variables that are its margins to their boundaries: none, unless it is a transistor.
    #[must_use]
    fn regions() -> Regions {
        Regions::NONE
    }

    /// Appends an instance's noise at `at`, the operating point a noise analysis linearizes
    /// about, to `sources`, each source's equations those of local unknowns: the one at
    /// local index `i` as `Unknown::new(i)`. None for a noiseless instance.
    fn noise<R: Real>(
        shape: &Self::Shape,
        prepared: &Self::Prepared<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        let _ = (shape, prepared, at, sources);
    }

    /// Appends the times in `[start, stop]` where an instance with the numbers `prepared`
    /// is not smooth, so a transient analysis steps onto them exactly: none, unless it is.
    fn breakpoints<R: Real>(
        prepared: &Self::Prepared<R>,
        (start, stop): (f64, f64),
        times: &mut Vec<f64>,
    ) {
        let _ = (prepared, start, stop, times);
    }

    /// Appends to `found` the choices an instance's behavior turns on at `at`, where its
    /// contribution jumps when one changes, so a transient analysis can step onto each
    /// change, finding them in room of `memory` that its evaluation does not read: none
    /// for an instance whose contribution is continuous.
    fn choices<R: Real>(
        shape: &Self::Shape,
        prepared: &Self::Prepared<R>,
        memory: &mut Self::Memory,
        at: &Local<'_>,
        found: &mut Vec<Choice>,
    ) {
        let _ = (shape, prepared, memory, at, found);
    }

    /// The level of an instance that is an independent source, which analyses excite: its
    /// waveform's at time zero. `None` for an instance that is no independent source.
    #[must_use]
    fn level<R: Real>(prepared: &Self::Prepared<R>) -> Option<f64> {
        let _ = prepared;
        None
    }

    /// Adds to `stamp` the derivative of each residual of an instance that is an
    /// independent source, of `shape`, with respect to its level: how a small-signal
    /// analysis excites it. Nothing for an instance that is no independent source.
    fn excite<R: Real>(shape: &Self::Shape, stamp: &mut Stamp<'_, R>) {
        let _ = (shape, stamp);
    }

    /// Starts an instance that is a logic driver, from what it remembers, `memory`,
    /// moving to `level` at `time`, at the rate that swings ground to supply in
    /// `transition` seconds, at once when it is zero, and returns the time it settles.
    /// `None`, and nothing done, for an instance that drives no logic.
    fn drive(
        memory: &mut Self::Memory,
        (time, level, transition): (f64, Level, f64),
    ) -> Option<f64> {
        let _ = (memory, time, level, transition);
        None
    }

    /// The local unknown of the current through an instance of `shape` that is a logic
    /// driver, whose equation holds its voltage; `None` for one that drives no logic.
    #[must_use]
    fn driven(shape: &Self::Shape) -> Option<usize> {
        let _ = shape;
        None
    }

    /// How far the input of an instance of `shape` that is a logic input's detector lies
    /// above its threshold at `at`, and the voltage of its supply above its ground there,
    /// the swing of a full edge, in volts; `None` for one that detects no logic.
    #[must_use]
    fn sense<R: Real>(
        shape: &Self::Shape,
        prepared: &Self::Prepared<R>,
        at: &Local<'_>,
    ) -> Option<(f64, f64)> {
        let _ = (shape, prepared, at);
        None
    }
}

/// The local unknowns among the first of which [`Layout::entry`] finds the claim of a pair
/// again at once, by the pair; it searches the claims for any other pair.
const INDEXED: usize = 16;

/// The local unknowns and entries an instance claims: its terminals, then each internal
/// unknown it adds, and each pair of them whose derivative it writes, once.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Layout {
    terminals: usize,
    internals: Vec<(Cow<'static, str>, Quantity)>,
    /// The local index of each internal unknown added as [`Layout::pivoted`], in
    /// increasing order.
    pivoted: Vec<usize>,
    entries: Vec<(usize, usize)>,
    /// One more than the local index of the entry claimed at each pair of the first
    /// [`INDEXED`] local unknowns, row by row, or zero for none.
    claimed: [[u16; INDEXED]; INDEXED],
}

impl Layout {
    /// The layout of an instance of `terminals` terminals, which have the local indices
    /// from 0, before it claims anything of its own.
    #[must_use]
    pub const fn new(terminals: usize) -> Self {
        Self {
            terminals,
            internals: Vec::new(),
            pivoted: Vec::new(),
            entries: Vec::new(),
            claimed: [[0; INDEXED]; INDEXED],
        }
    }

    /// Forgets every claim, for an instance of `terminals` terminals, keeping the memory.
    pub fn clear(&mut self, terminals: usize) {
        self.terminals = terminals;
        self.internals.clear();
        self.pivoted.clear();
        self.entries.clear();
        self.claimed = [[0; INDEXED]; INDEXED];
    }

    /// Adds an internal unknown, of `quantity`, labelled `label` after the instance's path,
    /// and returns its local index.
    pub fn internal(&mut self, label: impl Into<Cow<'static, str>>, quantity: Quantity) -> usize {
        self.internals.push((label.into(), quantity));
        self.terminals + self.internals.len() - 1
    }

    /// Adds an internal unknown as [`Layout::internal`] does, one whose equation's diagonal
    /// entry is at least as large as every other entry of its column at every evaluation:
    /// in the conductances, in `G + αC` for every `α ≥ 0`, and in `G + jωC` for every `ω`.
    /// Its pivot so passes the sparse solver's threshold, a tenth of its column's largest,
    /// whatever the circuit, and the solver eliminates it from each matrix before it
    /// factors the rest, unless it shares a row or a column with another such unknown.
    ///
    /// The kernel that adds one states why its diagonal is that large; a test over the
    /// library checks it of every evaluation it makes.
    pub fn pivoted(&mut self, label: impl Into<Cow<'static, str>>, quantity: Quantity) -> usize {
        let local = self.internal(label, quantity);
        self.pivoted.push(local);
        local
    }

    /// Claims the local entry at `row` and `column`, local unknowns, and returns its local
    /// index: that of the claim before it at the same pair, if any, so that contributions
    /// to one pair add in one place, in the order the instance makes them.
    pub fn entry(&mut self, row: usize, column: usize) -> usize {
        let pair = (row, column);
        let index = self
            .claimed
            .get_mut(row)
            .and_then(|columns| columns.get_mut(column));
        let found = match index.as_deref() {
            Some(&one_more) if one_more > 0 => Some(usize::from(one_more) - 1),
            // A pair among the first local unknowns is held there once claimed, while its
            // claim's index is within reach.
            Some(_) if self.entries.len() < usize::from(u16::MAX) => None,
            _ => self.entries.iter().position(|&claimed| claimed == pair),
        };
        if let Some(found) = found {
            return found;
        }
        self.entries.push(pair);
        let claimed = self.entries.len() - 1;
        if let (Some(slot), Ok(one_more)) = (index, u16::try_from(claimed + 1)) {
            *slot = one_more;
        }
        claimed
    }

    /// The number of terminals.
    #[must_use]
    pub const fn terminals(&self) -> usize {
        self.terminals
    }

    /// The number of local unknowns: the terminals, then the internal unknowns.
    #[must_use]
    pub fn unknowns(&self) -> usize {
        self.terminals + self.internals.len()
    }

    /// Each internal unknown's label and quantity, in the order of their local indices.
    #[must_use]
    pub fn internals(&self) -> &[(Cow<'static, str>, Quantity)] {
        &self.internals
    }

    /// The local index of each internal unknown added as [`Layout::pivoted`], in increasing
    /// order.
    #[must_use]
    pub fn pivots(&self) -> &[usize] {
        &self.pivoted
    }

    /// Each local entry's row and column, by its local index.
    #[must_use]
    pub fn entries(&self) -> &[(usize, usize)] {
        &self.entries
    }
}

/// The state an instance is evaluated at, as it sees it: its local unknowns' values, read
/// through its node map, and what the analysis is doing.
#[derive(Debug, Clone, Copy)]
pub struct Local<'a> {
    x: &'a [f64],
    map: &'a [u32],
    /// The time, in seconds; zero in every analysis but a transient one.
    pub time: f64,
    /// Whether time passes, as [`crate::Input::transient`] says.
    pub transient: bool,
    /// The factor every independent source is scaled by, as
    /// [`crate::Input::source_scale`] says.
    pub source_scale: f64,
    /// The conductance placed across every nonlinear junction, in siemens.
    pub gmin: f64,
    /// Whether the instance limits each voltage's step from its last evaluation's.
    pub limiting: bool,
}

impl<'a> Local<'a> {
    /// The state of an instance whose local unknown at each index is the unknown of `x`
    /// that `map` gives there, or the reference, whose value is zero, where it gives
    /// [`jolty_numerics::UNMAPPED`], at `time` and the rest of `like`'s.
    #[must_use]
    pub(crate) const fn new(x: &'a [f64], map: &'a [u32], like: &crate::Input<'_>) -> Self {
        Self {
            x,
            map,
            time: like.time,
            transient: like.transient,
            source_scale: like.source_scale,
            gmin: like.gmin,
            limiting: like.limiting,
        }
    }

    /// The value of the local unknown at `local`: zero for one on the reference, and for
    /// one the instance does not have.
    #[must_use]
    pub fn value(&self, local: usize) -> f64 {
        self.map
            .get(local)
            .and_then(|&unknown| self.x.get(unknown as usize))
            .copied()
            .unwrap_or(0.0)
    }

    /// The circuit's unknown of each local unknown, [`jolty_numerics::UNMAPPED`] for one on
    /// the reference.
    pub(crate) const fn map(&self) -> &'a [u32] {
        self.map
    }

    /// The same state as another instance sees it, whose local unknowns are the circuit's
    /// that `map` gives, as a device that heats sees the instance it holds.
    pub(crate) const fn remapped<'b>(&self, map: &'b [u32]) -> Local<'b>
    where
        'a: 'b,
    {
        Local {
            x: self.x,
            map,
            time: self.time,
            transient: self.transient,
            source_scale: self.source_scale,
            gmin: self.gmin,
            limiting: self.limiting,
        }
    }
}

/// Where an instance writes its contributions during one evaluation: its own block of
/// residuals, their scales and charges, one per local unknown, and conductances and
/// capacitances, one per local entry, all zero when it starts.
///
/// A write to a local index the instance does not have goes nowhere.
#[derive(Debug)]
pub struct Stamp<'a, R: Real = f64> {
    residual: &'a mut [R],
    scale: &'a mut [R],
    charge: &'a mut [R],
    conductance: &'a mut [R],
    capacitance: &'a mut [R],
    limited: bool,
}

impl<'a, R: Real> Stamp<'a, R> {
    /// A stamp over a block: the residuals, scales and charges of the local unknowns, and
    /// the conductances and capacitances of the local entries, each zeroed.
    pub(crate) fn zeroed(
        [residual, scale, charge]: [&'a mut [R]; 3],
        [conductance, capacitance]: [&'a mut [R]; 2],
    ) -> Self {
        for values in [
            &mut *residual,
            &mut *scale,
            &mut *charge,
            &mut *conductance,
            &mut *capacitance,
        ] {
            values.fill(R::default());
        }
        Self::over([residual, scale, charge], [conductance, capacitance])
    }

    /// A stamp over a block that is already zero, as [`Stamp::zeroed`] leaves it.
    pub(crate) const fn over(
        [residual, scale, charge]: [&'a mut [R]; 3],
        [conductance, capacitance]: [&'a mut [R]; 2],
    ) -> Self {
        Self {
            residual,
            scale,
            charge,
            conductance,
            capacitance,
            limited: false,
        }
    }

    /// Adds `value` to the residual of the local unknown `row`'s equation.
    pub fn add_residual(&mut self, row: usize, value: R) {
        self.add_residual_scaled(row, value, value);
    }

    /// Adds `value` to the residual of `row`'s equation, `scale` the largest magnitude of
    /// the terms it sums, as an instance that sums them itself reports them.
    pub fn add_residual_scaled(&mut self, row: usize, value: R, scale: R) {
        if let (Some(residual), Some(largest)) =
            (self.residual.get_mut(row), self.scale.get_mut(row))
        {
            *residual += value;
            *largest = largest.max(scale.abs());
        }
    }

    /// Adds `value` to the charge of `row`'s equation.
    pub fn add_charge(&mut self, row: usize, value: R) {
        if let Some(charge) = self.charge.get_mut(row) {
            *charge += value;
        }
    }

    /// Adds `value` to the conductance at the local entry `entry`.
    pub fn add_conductance(&mut self, entry: usize, value: R) {
        if let Some(conductance) = self.conductance.get_mut(entry) {
            *conductance += value;
        }
    }

    /// Adds `value` to the capacitance at the local entry `entry`.
    pub fn add_capacitance(&mut self, entry: usize, value: R) {
        if let Some(capacitance) = self.capacitance.get_mut(entry) {
            *capacitance += value;
        }
    }

    /// Records that the instance evaluated at a limited voltage rather than the one it was
    /// given, so the solution cannot have converged yet.
    pub const fn mark_limited(&mut self) {
        self.limited = true;
    }

    /// Whether the instance limited a voltage during this evaluation.
    #[must_use]
    pub const fn is_limited(&self) -> bool {
        self.limited
    }
}
