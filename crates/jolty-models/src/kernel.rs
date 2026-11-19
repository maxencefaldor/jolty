//! Single-instance device kernels isolated from batching, threading, and global matrices.
//!
//! A [`Kernel`] defines a device model as a pure function of temperature-prepared parameters
//! ([`Kernel::Prepared`]), instance state ([`Kernel::Memory`]), and local unknowns read via
//! [`Local`]. Evaluation stamps local equation residuals, scales, charges, conductances, and
//! capacitances into a [`Stamp`] by local index (terminals first, followed by internal
//! unknowns claimed in [`Layout`]).
//!
//! A kernel performs no heap allocation, never panics, and has no dependency on threads,
//! batches, or the global matrix. [`crate::Groups`] evaluates instances serially or in
//! parallel into dedicated slab blocks and gathers them in a fixed order.
//!
//! All library device kernels live in submodules alongside supporting scalar, junction,
//! region, and part helpers so `clippy::indexing_slicing` and `clippy::panic` are denied
//! uniformly across non-test code.

#![cfg_attr(
    not(test),
    deny(
        clippy::indexing_slicing,
        clippy::panic,
        reason = "a kernel never panics, and reads fixed arrays through `get`"
    )
)]

pub(crate) mod behavioral;
pub(crate) mod bjt;
pub(crate) mod bsim;
pub(crate) mod bsim3;
pub(crate) mod bsim4;
pub(crate) mod connect;
pub(crate) mod controlled;
pub(crate) mod diode;
pub(crate) mod dual;
pub(crate) mod foundry;
pub(crate) mod heat;
pub(crate) mod jfet;
pub(crate) mod junction;
pub(crate) mod memristor;
pub(crate) mod mosfet;
pub(crate) mod parts;
pub(crate) mod passive;
pub(crate) mod photocell;
pub(crate) mod real;
pub(crate) mod region;
pub(crate) mod sources;
pub(crate) mod switch;
pub(crate) mod thermal;
pub(crate) mod transmission;
pub(crate) mod vbic;
pub(crate) mod vdmos;

use std::borrow::Cow;
use std::fmt;

use crate::{
    Choice, Delayed, Interface, Lift, LogicLevel, Nature, NoiseSource, ParameterError,
    ParameterKind, Past, Real, Regions,
};

/// Generates `TryFrom<u8>` and `From<$name> for u8` for a `u8`-coded SPICE enum.
macro_rules! numbered_enum {
    (
        $name:ident,
        $message:literal,
        { $($code:literal => $variant:ident),+ $(,)? }
    ) => {
        impl TryFrom<u8> for $name {
            type Error = String;

            fn try_from(value: u8) -> Result<Self, String> {
                match value {
                    $($code => Ok(Self::$variant),)+
                    _ => {
                        let prefix = $message;
                        Err(format!("{prefix}, not {value}"))
                    }
                }
            }
        }

        impl From<$name> for u8 {
            fn from(value: $name) -> Self {
                match value {
                    $($name::$variant => $code,)+
                }
            }
        }
    };
}

pub(crate) use numbered_enum;

/// Single-instance device model evaluated as a pure function of prepared parameters, state
/// memory, and local unknowns.
///
/// Kernel evaluation performs no heap allocation, never panics, and has no dependency on
/// threads, batches, or the global matrix.
pub trait Kernel: Send + Sync + 'static {
    /// Model name registered in the library, such as `diode`.
    const NAME: &'static str;

    /// Flag indicating foundry models simulated externally by ngspice, matching
    /// [`crate::Model::foundry`].
    const FOUNDRY: bool = false;

    /// Typed parameter struct generic over scalar type `R`.
    ///
    /// Twins lift `Parameters<f64>` field by field via [`Lift`]. JSON serialization,
    /// deserialization, and schema generation apply only to `Parameters<f64>` inside
    /// [`crate::Model`] adapters such as [`crate::Differentiable`] or [`crate::Thermal`].
    type Parameters<R: Real>: Lift<R, Plain = Self::Parameters<f64>>
        + Clone
        + fmt::Debug
        + Send
        + Sync
        + 'static;

    /// Structural layout signature recording internal unknowns and matrix entry indices so
    /// instances sharing a kernel and shape batch together.
    ///
    /// Distinct shapes correspond to distinct layouts, uniquely determining the unknowns and
    /// entries claimed by an instance.
    type Shape: Clone + fmt::Debug + PartialEq + Send + Sync + 'static;

    /// Precomputed, temperature-scaled parameter set ready for evaluation.
    type Prepared<R: Real>: Clone + fmt::Debug + Send + Sync;

    /// Per-instance persistent state across evaluations (such as previous junction voltages
    /// for step limiting and preallocated scratch buffers), initialized by [`Kernel::memory`].
    type Memory: Clone + fmt::Debug + Default + Send + Sync;

    /// Returns the terminal interface and branch topology for `parameters`.
    fn interface<R: Real>(parameters: &Self::Parameters<R>) -> Interface;

    /// Validates `parameters` and precomputes temperature-dependent values at `temperature`
    /// in kelvins.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when a parameter is out of range.
    fn prepare<R: Real>(
        parameters: &Self::Parameters<R>,
        temperature: R,
    ) -> Result<Self::Prepared<R>, ParameterError>;

    /// Claims internal unknowns and matrix entries in `layout` for `parameters` and
    /// `prepared` (produced by [`Kernel::prepare`]), returning the instance shape.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when a parameter is out of range.
    fn layout<R: Real>(
        parameters: &Self::Parameters<R>,
        prepared: &Self::Prepared<R>,
        layout: &mut Layout,
    ) -> Result<Self::Shape, ParameterError>;

    /// Constructs initial per-instance state for `shape` and `prepared` before the first
    /// evaluation, defaulting to `Self::Memory::default()`.
    #[must_use]
    fn memory<R: Real>(shape: &Self::Shape, prepared: &Self::Prepared<R>) -> Self::Memory {
        let _ = (shape, prepared);
        Self::Memory::default()
    }

    /// Evaluates equation contributions for `shape` and `prepared` at local state `at`,
    /// updating `memory` and writing results into `stamp`.
    fn evaluate<R: Real>(
        shape: &Self::Shape,
        prepared: &Self::Prepared<R>,
        memory: &mut Self::Memory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    );

    /// Reports whether an instance with `shape` and `prepared` is strictly linear with
    /// constant conductance and capacitance entries.
    ///
    /// Linear kernels stamp identical entries at every state and produce residuals and
    /// charges equal to row-wise matrix-vector products without reading memory or limiting
    /// steps. Groups stamp linear entries once per run and derive residuals, scales, and
    /// charges from those entries; a library test verifies every `true` claim.
    #[must_use]
    fn linear(shape: &Self::Shape, prepared: &Self::Prepared<f64>) -> bool {
        let _ = (shape, prepared);
        false
    }

    /// Clears temperature-dependent state inside `memory` when a self-heating instance is
    /// re-prepared at a new temperature.
    fn reheat(memory: &mut Self::Memory) {
        let _ = memory;
    }

    /// Returns internal diagnostic variable names (such as threshold voltage) in the order
    /// written by [`Kernel::variables`].
    #[must_use]
    fn variable_names() -> &'static [&'static str] {
        &[]
    }

    /// Copies internal diagnostic variable values recorded in `memory` at the last evaluation
    /// (in SI base units) into `values`, matching [`Kernel::variable_names`].
    fn variables(memory: &Self::Memory, values: &mut [f64]) {
        let _ = (memory, values);
    }

    /// Returns operating-region metadata indexed by [`REGION`](crate::REGION) within
    /// [`Kernel::variable_names`] along with boundary margin variables, or [`Regions::NONE`].
    #[must_use]
    fn regions() -> Regions {
        Regions::NONE
    }

    /// Appends operating-point noise sources at `at` to `sources` using local unknown indices
    /// (`UnknownIndex::new(i)`).
    fn noise<R: Real>(
        shape: &Self::Shape,
        prepared: &Self::Prepared<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        let _ = (shape, prepared, at, sources);
    }

    /// Appends non-smooth waveform breakpoints within `[start, stop]` to `times` so transient
    /// stepping hits corners accurately.
    fn breakpoints<R: Real>(
        prepared: &Self::Prepared<R>,
        (start, stop): (f64, f64),
        times: &mut Vec<f64>,
    ) {
        let _ = (prepared, start, stop, times);
    }

    /// Returns whether an instance with `shape` and `prepared` depends explicitly on time
    /// (true for time-varying sources or behavioral expressions referencing `time`).
    #[must_use]
    fn varies_in_time<R: Real>(shape: &Self::Shape, prepared: &Self::Prepared<R>) -> bool {
        let _ = (shape, prepared);
        false
    }

    /// Appends delayed conductance terms sampled through [`Local::delayed`] to `found`.
    ///
    /// Small-signal analysis scales each constant conductance by `e^{-s·delay}` at complex
    /// frequency `s`, while DC uses the full value. Transient analysis records history for
    /// each delayed column and schedules breakpoints `delay` seconds after integration
    /// restarts.
    fn delayed<R: Real>(
        shape: &Self::Shape,
        prepared: &Self::Prepared<R>,
        found: &mut Vec<Delayed>,
    ) {
        let _ = (shape, prepared, found);
    }

    /// Appends discontinuous branch choices evaluated at `at` to `found` (using scratch
    /// buffers in `memory`) so transient stepping locates discontinuities accurately.
    fn choices<R: Real>(
        shape: &Self::Shape,
        prepared: &Self::Prepared<R>,
        memory: &mut Self::Memory,
        at: &Local<'_>,
        found: &mut Vec<Choice>,
    ) {
        let _ = (shape, prepared, memory, at, found);
    }

    /// Returns the time-zero DC value for an independent source, or `None` for other devices.
    #[must_use]
    fn dc_value<R: Real>(prepared: &Self::Prepared<R>) -> Option<f64> {
        let _ = prepared;
        None
    }

    /// Stamps unit small-signal excitation (residual derivative with respect to source value)
    /// into `stamp` for independent sources.
    fn excite<R: Real>(shape: &Self::Shape, stamp: &mut Stamp<'_, R>) {
        let _ = (shape, stamp);
    }

    /// Starts a logic-driver output ramp toward `level` at `time` with full-swing duration
    /// `transition`, returning the settling timestamp or `None` for non-drivers.
    fn drive(
        memory: &mut Self::Memory,
        (time, level, transition): (f64, LogicLevel, f64),
    ) -> Option<f64> {
        let _ = (memory, time, level, transition);
        None
    }

    /// Returns the local branch-current unknown index for a logic driver, or `None` for
    /// non-drivers.
    #[must_use]
    fn driven(shape: &Self::Shape) -> Option<usize> {
        let _ = shape;
        None
    }

    /// Evaluates input threshold margin and supply swing (in volts) at `at` for a logic
    /// detector, or returns `None` for non-detectors.
    #[must_use]
    fn sense<R: Real>(
        shape: &Self::Shape,
        prepared: &Self::Prepared<R>,
        at: &Local<'_>,
    ) -> Option<(f64, f64)> {
        let _ = (shape, prepared, at);
        None
    }

    /// Returns all base kernel parameters paired with their physical dimension or kind in
    /// declaration order.
    #[must_use]
    fn parameter_kinds() -> &'static [(&'static str, ParameterKind)] {
        &[]
    }

    /// Returns base names of parameters that expand into `l`, `w`, and `p` geometry-binning
    /// dependences.
    #[must_use]
    fn binned_parameters() -> &'static [&'static str] {
        &[]
    }
}

/// Number of leading local unknowns covered by the direct lookup table in [`Layout::entry`];
/// pairs beyond this index fall back to a linear scan.
const INDEXED: usize = 16;

/// Local unknown and matrix entry declarations claimed by a device instance.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Layout {
    terminals: usize,
    internals: Vec<(Cow<'static, str>, Nature)>,
    /// Local `(row, column)` pivot pairs claimed via [`Layout::pivot`] in claim order.
    pivots: Vec<(usize, usize)>,
    entries: Vec<(usize, usize)>,
    /// Lookup table storing `entry_index + 1` (or `0` if unclaimed) for pairs within the
    /// first [`INDEXED`] local unknowns.
    claimed: [[u16; INDEXED]; INDEXED],
}

impl Layout {
    /// Creates an empty layout pre-populated with `terminals` terminal indices `0..terminals`.
    #[must_use]
    pub const fn new(terminals: usize) -> Self {
        Self {
            terminals,
            internals: Vec::new(),
            pivots: Vec::new(),
            entries: Vec::new(),
            claimed: [[0; INDEXED]; INDEXED],
        }
    }

    /// Clears all claimed internals, pivots, and entries while retaining buffer capacity for
    /// an instance with `terminals` terminals.
    pub fn clear(&mut self, terminals: usize) {
        self.terminals = terminals;
        self.internals.clear();
        self.pivots.clear();
        self.entries.clear();
        self.claimed = [[0; INDEXED]; INDEXED];
    }

    /// Adds an internal unknown of `nature` named with suffix `label` and returns its local
    /// index.
    pub fn internal(&mut self, label: impl Into<Cow<'static, str>>, nature: Nature) -> usize {
        self.internals.push((label.into(), nature));
        self.terminals + self.internals.len() - 1
    }

    /// Adds an internal unknown via [`Layout::internal`] and claims its diagonal entry as a
    /// [pivot](Layout::pivot).
    pub fn pivoted(&mut self, label: impl Into<Cow<'static, str>>, nature: Nature) -> usize {
        let local = self.internal(label, nature);
        self.pivot(local, local);
        local
    }

    /// Claims local entry `(row, column)` as a guaranteed column-dominant pivot for internal
    /// unknown `column` across `G`, `G + αC` (`α ≥ 0`), and `G + jωC`.
    ///
    /// [`jolty_numerics::Condensation`] eliminates `column` before general factorization
    /// because the pivot exceeds the sparse solver threshold (one-tenth of the column
    /// maximum). Each claiming kernel documents the dominance proof, and a library test
    /// verifies it across evaluations.
    pub fn pivot(&mut self, row: usize, column: usize) {
        self.pivots.push((row, column));
    }

    /// Withdraws any claimed pivot for internal unknown `column` when column dominance no
    /// longer holds.
    pub fn unpivot(&mut self, column: usize) {
        self.pivots.retain(|&(_, claimed)| claimed != column);
    }

    /// Claims local matrix entry `(row, column)` and returns its deduplicated local index so
    /// repeated stamps accumulate into one slot.
    pub fn entry(&mut self, row: usize, column: usize) -> usize {
        let pair = (row, column);
        let index = self
            .claimed
            .get_mut(row)
            .and_then(|columns| columns.get_mut(column));
        let found = match index.as_deref() {
            Some(&one_more) if one_more > 0 => Some(usize::from(one_more) - 1),
            // Pairs within the indexed prefix are cached directly while indices fit in `u16`.
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

    /// Returns the number of terminal unknowns.
    #[must_use]
    pub const fn terminals(&self) -> usize {
        self.terminals
    }

    /// Returns the total number of local unknowns (terminals plus internal unknowns).
    #[must_use]
    pub fn unknowns(&self) -> usize {
        self.terminals + self.internals.len()
    }

    /// Returns internal unknown `(label, nature)` declarations in local index order.
    #[must_use]
    pub fn internals(&self) -> &[(Cow<'static, str>, Nature)] {
        &self.internals
    }

    /// Returns claimed local `(row, column)` pivots in claim order.
    #[must_use]
    pub fn pivots(&self) -> &[(usize, usize)] {
        &self.pivots
    }

    /// Returns claimed local `(row, column)` matrix entries in local index order.
    #[must_use]
    pub fn entries(&self) -> &[(usize, usize)] {
        &self.entries
    }
}

/// Instance-local view of the evaluation state, exposing mapped local unknown values and
/// analysis flags.
#[derive(Debug, Clone, Copy)]
pub struct Local<'a> {
    x: &'a [f64],
    map: &'a [u32],
    /// Simulation time, in seconds; zero in non-transient analyses.
    pub time: f64,
    /// Flag indicating active transient time stepping, matching [`crate::Input::transient`].
    pub transient: bool,
    /// Independent-source continuation scale factor, matching
    /// [`crate::Input::source_scale`].
    pub source_scale: f64,
    /// Shunt conductance placed across nonlinear junctions, in siemens.
    pub gmin: f64,
    /// Flag enabling per-iteration junction voltage-step limiting.
    pub limiting: bool,
    past: Option<&'a Past>,
    unknowns: UnknownSeeds<'a>,
}

/// Unknown-seeding mode used by sensitivity twins inside [`Local`].
#[derive(Debug, Clone, Copy)]
enum UnknownSeeds<'a> {
    /// Unseeded unknowns for plain evaluation or parameter-only twins.
    None,
    /// Unit seeds moving each listed circuit unknown at rate `1.0`.
    Units(&'a [u32]),
    /// Tangent seeds moving all circuit unknowns along each tangent vector.
    Tangents(&'a [Vec<f64>]),
}

impl<'a> Local<'a> {
    /// Constructs a local evaluation view mapping local unknown indices through `map` into
    /// `x` ([`jolty_numerics::UNMAPPED`] reads zero for ground) with settings from `like`.
    #[must_use]
    pub(crate) const fn new(x: &'a [f64], map: &'a [u32], like: &crate::Input<'a>) -> Self {
        Self {
            x,
            map,
            time: like.time,
            transient: like.transient,
            source_scale: like.source_scale,
            gmin: like.gmin,
            limiting: like.limiting,
            past: like.past,
            unknowns: UnknownSeeds::None,
        }
    }

    /// Returns a copy of `self` seeding each circuit unknown in `seeded` along a dedicated
    /// unit direction.
    #[must_use]
    pub(crate) const fn seeding(self, seeded: &'a [u32]) -> Self {
        let unknowns = if seeded.is_empty() {
            UnknownSeeds::None
        } else {
            UnknownSeeds::Units(seeded)
        };
        Self { unknowns, ..self }
    }

    /// Returns a copy of `self` seeding circuit unknowns along each tangent vector in
    /// `tangents`.
    #[must_use]
    pub(crate) const fn along(self, tangents: &'a [Vec<f64>]) -> Self {
        let unknowns = if tangents.is_empty() {
            UnknownSeeds::None
        } else {
            UnknownSeeds::Tangents(tangents)
        };
        Self { unknowns, ..self }
    }

    /// Returns whether circuit unknowns carry active twin differentiation seeds.
    #[must_use]
    pub(crate) const fn seeds_unknowns(&self) -> bool {
        !matches!(self.unknowns, UnknownSeeds::None)
    }

    /// Reads local unknown `local` as scalar type `R`, carrying active unknown seed
    /// derivatives when evaluated inside an unknown-seeded twin.
    #[inline]
    #[must_use]
    pub fn unknown<R: Real>(&self, local: usize) -> R {
        let value = self.value(local);
        if R::SEEDS == 0 || !self.seeds_unknowns() {
            return R::constant(value);
        }
        let Some(&unknown) = self
            .map
            .get(local)
            .filter(|&&unknown| unknown != jolty_numerics::UNMAPPED)
        else {
            return R::constant(value);
        };
        match self.unknowns {
            UnknownSeeds::None => R::constant(value),
            UnknownSeeds::Units(seeded) => seeded
                .iter()
                .position(|&each| each == unknown)
                .map_or_else(|| R::constant(value), |slot| R::unit(value, slot)),
            UnknownSeeds::Tangents(tangents) => {
                let index = unknown as usize;
                let mut rates = [0.0; crate::Seeds::MAX];
                for (rate, tangent) in rates.iter_mut().zip(tangents) {
                    *rate = tangent.get(index).copied().unwrap_or(0.0);
                }
                R::seeded(value, &rates)
            }
        }
    }

    /// Returns the derivative of local unknown `local` along seed `slot`, or `0.0` for
    /// unseeded evaluations or grounded unknowns.
    #[inline]
    #[must_use]
    pub fn unknown_derivative(&self, local: usize, slot: usize) -> f64 {
        let Some(&unknown) = self
            .map
            .get(local)
            .filter(|&&unknown| unknown != jolty_numerics::UNMAPPED)
        else {
            return 0.0;
        };
        match self.unknowns {
            UnknownSeeds::None => 0.0,
            UnknownSeeds::Units(seeded) => {
                if seeded.iter().position(|&each| each == unknown) == Some(slot) {
                    1.0
                } else {
                    0.0
                }
            }
            UnknownSeeds::Tangents(tangents) => tangents
                .get(slot)
                .and_then(|tangent| tangent.get(unknown as usize))
                .copied()
                .unwrap_or(0.0),
        }
    }

    /// Returns the scalar value of local unknown `local`, or `0.0` for ground or out-of-range
    /// indices.
    #[must_use]
    pub fn value(&self, local: usize) -> f64 {
        self.map
            .get(local)
            .and_then(|&unknown| self.x.get(unknown as usize))
            .copied()
            .unwrap_or(0.0)
    }

    /// Returns the local-to-global unknown map ([`jolty_numerics::UNMAPPED`] for ground).
    pub(crate) const fn map(&self) -> &'a [u32] {
        self.map
    }

    /// Returns a view sharing the same simulation state while substituting local unknown map
    /// `map` (used by self-heating wrappers delegating to inner kernels).
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
            past: self.past,
            unknowns: self.unknowns,
        }
    }

    /// Samples local unknown `local` at `time - delay`, returning `(value, weight)` where
    /// `weight` is the derivative with respect to the present value.
    ///
    /// Returns `(present, 1.0)` outside transient analysis, `(0.0, 0.0)` for ground, and
    /// interpolates via [`Past::at`] during transients for columns declared by
    /// [`Kernel::delayed`].
    #[must_use]
    pub fn delayed(&self, local: usize, delay: f64) -> (f64, f64) {
        let Some(&unknown) = self.map.get(local) else {
            return (0.0, 0.0);
        };
        let Some(&present) = self.x.get(unknown as usize) else {
            return (0.0, 0.0);
        };
        self.past
            .and_then(|past| past.at(unknown as usize, self.time - delay, (self.time, present)))
            .unwrap_or((present, 1.0))
    }

    /// Samples local unknown `local` at `time - delay` via [`Local::delayed`], lifting the
    /// result into `R` scaled by the present-value derivative `weight` under unknown seeding.
    #[inline]
    #[must_use]
    pub fn delayed_unknown<R: Real>(&self, local: usize, delay: f64) -> (R, f64) {
        let (value, weight) = self.delayed(local, delay);
        if R::SEEDS == 0 || !self.seeds_unknowns() {
            return (R::constant(value), weight);
        }
        (
            (self.unknown::<R>(local) * weight).with_value(value),
            weight,
        )
    }
}

/// Mutable local output block receiving equation contributions from one kernel evaluation.
///
/// Holds zero-initialized slices for local unknown residuals, scales, and charges alongside
/// local entry conductances and capacitances; writes to out-of-range indices are ignored.
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
    /// Zeroes all five local block slices and returns a [`Stamp`] writing into them.
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

    /// Wraps five already-zeroed block slices (as prepared by [`Stamp::zeroed`]) in a
    /// [`Stamp`].
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

    /// Adds `value` to the residual of local unknown `row`.
    pub fn add_residual(&mut self, row: usize, value: R) {
        self.add_residual_scaled(row, value, value);
    }

    /// Adds `value` to the residual of local unknown `row` while updating the convergence
    /// scale with `scale.abs()`.
    pub fn add_residual_scaled(&mut self, row: usize, value: R, scale: R) {
        if let (Some(residual), Some(largest)) =
            (self.residual.get_mut(row), self.scale.get_mut(row))
        {
            *residual += value;
            *largest = largest.max(scale.abs());
        }
    }

    /// Adds `value` to the charge of local unknown `row`.
    pub fn add_charge(&mut self, row: usize, value: R) {
        if let Some(charge) = self.charge.get_mut(row) {
            *charge += value;
        }
    }

    /// Adds `value` to the conductance at local entry `entry`.
    pub fn add_conductance(&mut self, entry: usize, value: R) {
        if let Some(conductance) = self.conductance.get_mut(entry) {
            *conductance += value;
        }
    }

    /// Adds `value` to the capacitance at local entry `entry`.
    pub fn add_capacitance(&mut self, entry: usize, value: R) {
        if let Some(capacitance) = self.capacitance.get_mut(entry) {
            *capacitance += value;
        }
    }

    /// Marks the evaluation as voltage-limited so Newton iteration continues to another step.
    pub const fn mark_limited(&mut self) {
        self.limited = true;
    }

    /// Returns whether voltage limiting occurred during this evaluation.
    #[must_use]
    pub const fn is_limited(&self) -> bool {
        self.limited
    }
}
