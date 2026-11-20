//! Self-heating kernel adapter implementing [`Kind`] for thermally coupled device instances.
//!
//! Each heated instance appends a trailing `thermal` terminal, evaluates the inner kernel
//! with temperature-derivative scalars, extends limited steps linearly, and injects
//! dissipated electrical power (`Σ v · i`) as heat flow into the thermal net.

use std::fmt;
use std::marker::PhantomData;

use jolty_numerics::UNMAPPED;

use super::{Heating, MAX_TEMPERATURE_STEP, MIN_TEMPERATURE};
use crate::group::Kind;
use crate::lift::{constant, lift};
use crate::{
    Choice, Delayed, Gradient, Hyper, Kernel, Layout, Local, LogicLevel, Nature, NoiseSource,
    ParameterError, Real, Regions, Stamp,
};

/// Group [`Kind`] wrapper for self-heating instances of kernel `K`.
pub(crate) struct HeatedKernel<K>(PhantomData<fn() -> K>);

/// Layout mapping between an inner kernel shape `S` and its enclosing heated instance (which
/// inserts the `thermal` terminal immediately after the inner kernel's terminals).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HeatedShape<S> {
    /// Underlying inner kernel shape.
    inner: S,
    /// Number of inner terminals, which also equals the local index of the `thermal` terminal.
    terminals: usize,
    /// Flag per inner local unknown indicating whether it is a node voltage contributing to
    /// electrical power dissipation.
    voltages: Vec<bool>,
    /// Inner matrix entries sorted by `(column, row)` as
    /// `(inner_row, inner_col, inner_entry, outer_entry)` tuples.
    entries: Vec<(usize, usize, usize, usize)>,
    /// Outer matrix entry indices `(outer_unknown, thermal)` for temperature derivatives of
    /// inner equations.
    column: Vec<usize>,
    /// Outer matrix entry indices `(thermal, outer_unknown)` for power derivatives with
    /// respect to inner unknowns.
    row: Vec<usize>,
    /// Outer diagonal matrix entry `(thermal, thermal)` for power derivative with respect to
    /// temperature.
    diagonal: usize,
}

impl<S> HeatedShape<S> {
    /// Returns the number of local unknowns in the inner kernel.
    fn unknowns(&self) -> usize {
        self.voltages.len()
    }

    /// Maps an inner local unknown index `local` to the outer heated local index, shifting
    /// internal unknowns past the `thermal` terminal.
    const fn outer(&self, local: usize) -> usize {
        if local < self.terminals {
            local
        } else {
            local + 1
        }
    }

    /// Extracts the global unknown map for the inner kernel from outer local state `at`.
    fn map(&self, at: &Local<'_>) -> Vec<u32> {
        (0..self.unknowns())
            .map(|local| at.map().get(self.outer(local)).copied().unwrap_or(UNMAPPED))
            .collect()
    }
}

/// Prepared state for a heated instance storing lifted parameters for dynamic re-preparation,
/// ambient temperature, and precomputed ambient-temperature parameters.
pub(crate) struct HeatedPrepared<K: Kernel, R: Heating> {
    parameters: Box<K::Parameters<R::Warm>>,
    ambient: f64,
    at_ambient: Box<K::Prepared<R::Warm>>,
}

impl<K: Kernel, R: Heating> Clone for HeatedPrepared<K, R> {
    fn clone(&self) -> Self {
        Self {
            parameters: self.parameters.clone(),
            ambient: self.ambient,
            at_ambient: self.at_ambient.clone(),
        }
    }
}

impl<K: Kernel, R: Heating> fmt::Debug for HeatedPrepared<K, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HeatedPrepared")
            .field("ambient", &self.ambient)
            .field("at_ambient", &self.at_ambient)
            .finish_non_exhaustive()
    }
}

impl<K: Kernel, R: Heating> HeatedPrepared<K, R> {
    /// Prepares boxed `parameters` at ambient temperature `ambient` in kelvins.
    fn new(parameters: Box<K::Parameters<R::Warm>>, ambient: f64) -> Result<Self, ParameterError> {
        let at_ambient = Box::new(K::prepare(&parameters, R::seed(ambient))?);
        Ok(Self {
            parameters,
            ambient,
            at_ambient,
        })
    }
}

/// Persistent evaluation state for a heated instance holding temperature-prepared inner
/// parameters, last evaluated temperature for step limiting, inner memory, and scratch buffers.
pub(crate) struct HeatedMemory<K: Kernel, R: Heating> {
    prepared: Option<Box<K::Prepared<R::Warm>>>,
    inner: K::Memory,
    last: f64,
    /// Flag indicating whether `prepared` was last evaluated with twin unknown seeding.
    seeded: bool,
    /// Cached global unknown indices for inner local unknowns.
    map: Vec<u32>,
    /// Scratch buffer storing inner residuals, scales, charges, conductances, and
    /// capacitances.
    block: Vec<R::Warm>,
    /// Scratch buffer accumulating power derivatives with respect to each inner local unknown.
    slopes: Vec<R>,
}

impl<K: Kernel, R: Heating> Default for HeatedMemory<K, R> {
    fn default() -> Self {
        Self {
            prepared: None,
            inner: K::Memory::default(),
            last: 0.0,
            seeded: false,
            map: Vec::new(),
            block: Vec::new(),
            slopes: Vec::new(),
        }
    }
}

impl<K: Kernel, R: Heating> Clone for HeatedMemory<K, R> {
    fn clone(&self) -> Self {
        Self {
            prepared: self.prepared.clone(),
            inner: self.inner.clone(),
            last: self.last,
            seeded: self.seeded,
            map: self.map.clone(),
            block: self.block.clone(),
            slopes: self.slopes.clone(),
        }
    }
}

impl<K: Kernel, R: Heating> fmt::Debug for HeatedMemory<K, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HeatedMemory")
            .field("inner", &self.inner)
            .field("last", &self.last)
            .finish_non_exhaustive()
    }
}

impl<K: Kernel, R: Heating> HeatedMemory<K, R> {
    /// Initializes memory and scratch buffers for `shape` and `prepared` at ambient
    /// temperature.
    fn new(shape: &HeatedShape<K::Shape>, prepared: &HeatedPrepared<K, R>) -> Self {
        let unknowns = shape.unknowns();
        let stored = shape.entries.len().max(1);
        Self {
            inner: K::memory(&shape.inner, &prepared.at_ambient),
            prepared: Some(prepared.at_ambient.clone()),
            last: prepared.ambient,
            seeded: false,
            map: vec![UNMAPPED; unknowns],
            block: vec![R::Warm::default(); 3 * unknowns + 2 * stored],
            slopes: vec![R::default(); unknowns],
        }
    }

    /// Fills `self.map` with global unknown indices for every inner local unknown from outer
    /// state `at`.
    fn read(&mut self, shape: &HeatedShape<K::Shape>, at: &Local<'_>) {
        let map = at.map();
        let unknowns = shape.unknowns();
        self.map.resize(unknowns, UNMAPPED);
        for (local, unknown) in self.map.iter_mut().enumerate() {
            *unknown = map.get(shape.outer(local)).copied().unwrap_or(UNMAPPED);
        }
    }
}

/// Splits scratch slice `block` into `[residual, scale, charge, conductance, capacitance]`
/// sub-slices sized for `unknowns` and `stored`, or returns `None` if too short.
fn parts<T>(block: &mut [T], unknowns: usize, stored: usize) -> Option<[&mut [T]; 5]> {
    let (residual, rest) = block.split_at_mut_checked(unknowns)?;
    let (scale, rest) = rest.split_at_mut_checked(unknowns)?;
    let (charge, rest) = rest.split_at_mut_checked(unknowns)?;
    let (conductance, rest) = rest.split_at_mut_checked(stored)?;
    let capacitance = rest.get_mut(..stored)?;
    Some([residual, scale, charge, conductance, capacitance])
}

/// Claims inner internal unknowns, inner entries, and thermal-coupling entries in outer
/// `layout` (which ends with the `thermal` terminal) and returns the constructed
/// [`HeatedShape`].
fn heated_shape<S>(inner: S, own: &Layout, layout: &mut Layout) -> HeatedShape<S> {
    let terminals = own.terminals();
    let thermal = terminals;
    let mut voltages = vec![true; terminals];
    for (label, nature) in own.internals() {
        layout.internal(label.clone(), *nature);
        voltages.push(*nature == Nature::Voltage);
    }
    let outer = |local: usize| if local < terminals { local } else { local + 1 };
    let mut pairs: Vec<(usize, usize, usize)> = own
        .entries()
        .iter()
        .enumerate()
        .map(|(index, &(row, column))| (row, column, index))
        .collect();
    pairs.sort_unstable_by_key(|&(row, column, _)| (column, row));
    let entries = pairs
        .into_iter()
        .map(|(row, column, index)| (row, column, index, layout.entry(outer(row), outer(column))))
        .collect();
    let unknowns = voltages.len();
    let column = (0..unknowns)
        .map(|local| layout.entry(outer(local), thermal))
        .collect();
    let row = (0..unknowns)
        .map(|local| layout.entry(thermal, outer(local)))
        .collect();
    HeatedShape {
        inner,
        terminals,
        voltages,
        entries,
        column,
        row,
        diagonal: layout.entry(thermal, thermal),
    }
}

/// Updates the inner prepared parameters inside `memory` to the clamped device temperature at
/// `at`, returning the unapplied temperature step `applied - evaluated`.
///
/// Clamps steps to [`MAX_TEMPERATURE_STEP`] when limiting is active and enforces
/// [`MIN_TEMPERATURE`], marking `stamp` limited whenever clamping or out-of-range
/// re-preparation occurs.
fn heat<K: Kernel, R: Heating>(
    shape: &HeatedShape<K::Shape>,
    prepared: &HeatedPrepared<K, R>,
    memory: &mut HeatedMemory<K, R>,
    at: &Local<'_>,
    stamp: &mut Stamp<'_, R>,
) -> f64 {
    let applied = prepared.ambient + at.value(shape.terminals);
    let stepped = if at.limiting {
        applied.clamp(
            memory.last - MAX_TEMPERATURE_STEP,
            memory.last + MAX_TEMPERATURE_STEP,
        )
    } else {
        applied
    };
    let temperature = stepped.max(MIN_TEMPERATURE);
    let seeding = R::SEEDS > 0 && at.seeds_unknowns();
    if temperature.to_bits() != memory.last.to_bits() || seeding || memory.seeded {
        // The temperature moves with the thermal net when a twin seeds it.
        // Zeroing the value first prevents `inf - inf = NaN` if a trial step
        // drives the thermal net infinite while `temperature` is clamped finite.
        let moving = if seeding {
            R::seed(temperature) + at.unknown::<R::Warm>(shape.terminals).with_value(0.0)
        } else {
            R::seed(temperature)
        };
        if let Ok(heated) = K::prepare_heated(&prepared.parameters, &prepared.at_ambient, moving) {
            if let Some(slot) = &mut memory.prepared {
                **slot = heated;
            } else {
                memory.prepared = Some(Box::new(heated));
            }
            K::reheat(&mut memory.inner);
            memory.last = temperature;
            memory.seeded = seeding;
        } else {
            // A parameter out of range here: the step went too far, and the last
            // temperature's numbers stand, the solution not yet converged.
            stamp.mark_limited();
        }
    }
    if temperature.to_bits() != applied.to_bits() {
        stamp.mark_limited();
    }
    applied - memory.last
}

/// Evaluates a heated instance of `shape`, with the numbers `prepared`, at `at`, from what
/// it remembers, `memory`, into `stamp`, computing with `R`.
fn evaluate<K: Kernel, R: Heating>(
    shape: &HeatedShape<K::Shape>,
    prepared: &HeatedPrepared<K, R>,
    memory: &mut HeatedMemory<K, R>,
    at: &Local<'_>,
    stamp: &mut Stamp<'_, R>,
) {
    memory.read(shape, at);
    let thermal = shape.terminals;
    let beyond = heat(shape, prepared, memory, at, stamp);
    let (unknowns, stored) = (shape.unknowns(), shape.entries.len().max(1));
    memory
        .block
        .resize(3 * unknowns + 2 * stored, R::Warm::default());
    let HeatedMemory {
        prepared: Some(heated),
        inner,
        map,
        block,
        slopes,
        ..
    } = memory
    else {
        return;
    };
    let Some([residual, scale, charge, conductance, capacitance]) = parts(block, unknowns, stored)
    else {
        return;
    };
    let mut own = Stamp::zeroed(
        [&mut *residual, &mut *scale, &mut *charge],
        [&mut *conductance, &mut *capacitance],
    );
    K::evaluate(
        &shape.inner,
        heated,
        inner,
        &at.remapped(map.as_slice()),
        &mut own,
    );
    if own.is_limited() {
        stamp.mark_limited();
    }
    // The power, `Σ v · i` over the instance's voltages, and its derivatives.
    let (mut power, mut hotter_power) = (R::constant(0.0), R::constant(0.0));
    slopes.resize(unknowns, R::default());
    slopes.fill(R::constant(0.0));
    let rows = residual.iter().zip(scale.iter()).zip(charge.iter());
    for (local, ((&residual, &scale), &charge)) in rows.enumerate() {
        let outer = shape.outer(local);
        let (current, hotter) = R::split(residual);
        let current = current + hotter * beyond;
        let (held, hotter_held) = R::split(charge);
        stamp.add_residual_scaled(outer, current, R::split(scale).0);
        stamp.add_charge(outer, held + hotter_held * beyond);
        let column = shape.column.get(local).copied().unwrap_or(usize::MAX);
        stamp.add_conductance(column, hotter);
        stamp.add_capacitance(column, hotter_held);
        if let (Some(true), Some(slope)) = (shape.voltages.get(local), slopes.get_mut(local)) {
            let value = at.unknown::<R>(outer);
            power += current * value;
            hotter_power += hotter * value;
            *slope += current;
        }
    }
    for &(row, column, index, entry) in &shape.entries {
        let (Some(&slope), Some(&held)) = (conductance.get(index), capacitance.get(index)) else {
            continue;
        };
        let (slope, _) = R::split(slope);
        stamp.add_conductance(entry, slope);
        stamp.add_capacitance(entry, R::split(held).0);
        if let (Some(true), Some(sum)) = (shape.voltages.get(row), slopes.get_mut(column)) {
            *sum += slope * at.unknown::<R>(shape.outer(row));
        }
    }
    stamp.add_residual(thermal, -power);
    stamp.add_conductance(shape.diagonal, -hotter_power);
    for (&entry, &slope) in shape.row.iter().zip(slopes.iter()) {
        stamp.add_conductance(entry, -slope);
    }
}

impl<K: Kernel> Kind for HeatedKernel<K> {
    const NAME: &'static str = K::NAME;
    type Parameters = K::Parameters<f64>;
    type Shape = HeatedShape<K::Shape>;
    type Prepared = HeatedPrepared<K, f64>;
    type Memory = HeatedMemory<K, f64>;

    fn prepare(
        parameters: &Self::Parameters,
        temperature: f64,
    ) -> Result<Self::Prepared, ParameterError> {
        HeatedPrepared::new(Box::new(constant(parameters)), temperature)
    }

    fn layout(
        _: &Self::Parameters,
        prepared: &Self::Prepared,
        layout: &mut Layout,
    ) -> Result<Self::Shape, ParameterError> {
        let mut own = Layout::new(layout.terminals().saturating_sub(1));
        let inner = K::layout(&prepared.parameters, &prepared.at_ambient, &mut own)?;
        Ok(heated_shape(inner, &own, layout))
    }

    fn memory(shape: &Self::Shape, prepared: &Self::Prepared) -> Self::Memory {
        HeatedMemory::new(shape, prepared)
    }

    /// Returns `false` because dissipated power couples nonlinearly into temperature.
    fn linear(_: &Self::Shape, _: &Self::Prepared) -> bool {
        false
    }

    fn evaluate(
        shape: &Self::Shape,
        prepared: &Self::Prepared,
        memory: &mut Self::Memory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_>,
    ) {
        evaluate::<K, f64>(shape, prepared, memory, at, stamp);
    }

    /// Appends operating-point noise sources evaluated at the temperature of the last run.
    fn noise(
        shape: &Self::Shape,
        prepared: &Self::Prepared,
        memory: &Self::Memory,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        let heated = memory
            .prepared
            .as_deref()
            .unwrap_or(prepared.at_ambient.as_ref());
        emit_heated_noise::<K, f64>(shape, heated, at, sources);
    }

    fn variable_names() -> &'static [&'static str] {
        K::variable_names()
    }

    fn regions() -> Regions {
        K::regions()
    }

    fn variables(memory: &Self::Memory, values: &mut [f64]) {
        K::variables(&memory.inner, values);
    }

    fn breakpoints(prepared: &Self::Prepared, span: (f64, f64), times: &mut Vec<f64>) {
        K::breakpoints(&prepared.at_ambient, span, times);
    }

    fn varies_in_time(shape: &Self::Shape, prepared: &Self::Prepared) -> bool {
        K::varies_in_time(&shape.inner, &prepared.at_ambient)
    }

    fn delayed(shape: &Self::Shape, prepared: &Self::Prepared, found: &mut Vec<Delayed>) {
        let start = found.len();
        K::delayed(&shape.inner, &prepared.at_ambient, found);
        for part in found.iter_mut().skip(start) {
            if let Some(&(.., outer)) = shape
                .entries
                .iter()
                .find(|&&(_, _, inner, _)| inner == part.entry)
            {
                part.entry = outer;
            }
        }
    }

    fn choices(
        shape: &Self::Shape,
        prepared: &Self::Prepared,
        memory: &mut Self::Memory,
        at: &Local<'_>,
        found: &mut Vec<Choice>,
    ) {
        memory.read(shape, at);
        let heated = memory
            .prepared
            .as_deref()
            .unwrap_or(prepared.at_ambient.as_ref());
        let HeatedMemory { inner, map, .. } = memory;
        K::choices(
            &shape.inner,
            heated,
            inner,
            &at.remapped(map.as_slice()),
            found,
        );
    }

    fn dc_value(prepared: &Self::Prepared) -> Option<f64> {
        K::dc_value(&prepared.at_ambient)
    }

    fn excite(shape: &Self::Shape, stamp: &mut Stamp<'_>) {
        let (unknowns, stored) = (shape.unknowns(), shape.entries.len().max(1));
        let mut block = vec![0.0; 3 * unknowns + 2 * stored];
        let Some([residual, scale, charge, conductance, capacitance]) =
            parts(&mut block, unknowns, stored)
        else {
            return;
        };
        let mut own = Stamp::zeroed(
            [&mut *residual, &mut *scale, &mut *charge],
            [conductance, capacitance],
        );
        K::excite(&shape.inner, &mut own);
        for (local, &value) in residual.iter().enumerate() {
            stamp.add_residual(shape.outer(local), value);
        }
    }

    fn drive(memory: &mut Self::Memory, ramp: (f64, LogicLevel, f64)) -> Option<f64> {
        K::drive(&mut memory.inner, ramp)
    }

    fn driven(shape: &Self::Shape) -> Option<usize> {
        K::driven(&shape.inner).map(|local| shape.outer(local))
    }

    fn sense(shape: &Self::Shape, prepared: &Self::Prepared, at: &Local<'_>) -> Option<(f64, f64)> {
        let map = shape.map(at);
        K::sense(&shape.inner, &prepared.at_ambient, &at.remapped(&map))
    }

    type Twin<const SEEDS: usize> = Box<(
        HeatedPrepared<K, Gradient<SEEDS>>,
        HeatedMemory<K, Gradient<SEEDS>>,
    )>;

    fn twin<const SEEDS: usize>(
        shape: &Self::Shape,
        parameters: &Self::Parameters,
        seeds: &[(&str, [f64; SEEDS])],
        temperature: f64,
    ) -> Option<Self::Twin<SEEDS>> {
        let lifted: Box<K::Parameters<Hyper<SEEDS>>> =
            Box::new(lift(parameters, seeds, Hyper::new).ok()?);
        let prepared = HeatedPrepared::<K, Gradient<SEEDS>>::new(lifted, temperature).ok()?;
        let memory = HeatedMemory::new(shape, &prepared);
        Some(Box::new((prepared, memory)))
    }

    fn evaluate_twin<const SEEDS: usize>(
        shape: &Self::Shape,
        twin: &mut Self::Twin<SEEDS>,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, Gradient<SEEDS>>,
    ) {
        let (prepared, memory) = &mut **twin;
        evaluate::<K, Gradient<SEEDS>>(shape, prepared, memory, at, stamp);
    }

    fn noise_twin<const SEEDS: usize>(
        shape: &Self::Shape,
        twin: &mut Self::Twin<SEEDS>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource<Gradient<SEEDS>>>,
    ) {
        let (prepared, memory) = &mut **twin;
        let applied = prepared.ambient + at.value(shape.terminals);
        let temperature = applied.max(MIN_TEMPERATURE);
        let seeding = SEEDS > 0 && at.seeds_unknowns();
        if temperature.to_bits() != memory.last.to_bits() || seeding || memory.seeded {
            let moving = if seeding && applied > MIN_TEMPERATURE {
                Gradient::<SEEDS>::seed(temperature)
                    + at.unknown::<Hyper<SEEDS>>(shape.terminals).with_value(0.0)
            } else {
                Gradient::<SEEDS>::seed(temperature)
            };
            if let Ok(heated) =
                K::prepare_heated(&prepared.parameters, &prepared.at_ambient, moving)
            {
                if let Some(slot) = &mut memory.prepared {
                    **slot = heated;
                } else {
                    memory.prepared = Some(Box::new(heated));
                }
                memory.last = temperature;
                memory.seeded = seeding;
            }
        }
        let heated = memory
            .prepared
            .as_deref()
            .unwrap_or(prepared.at_ambient.as_ref());
        emit_heated_noise::<K, Gradient<SEEDS>>(shape, heated, at, sources);
    }

    #[cfg(any(test, feature = "testing"))]
    fn lift_entries(parameters: &Self::Parameters) -> Vec<(String, f64)> {
        <K as Kind>::lift_entries(parameters)
    }
}

/// Evaluates inner kernel noise on `heated` at `at`, splits out temperature derivatives, and
/// remaps local equation indices into the outer heated layout.
fn emit_heated_noise<K: Kernel, R: Heating>(
    shape: &HeatedShape<K::Shape>,
    heated: &K::Prepared<R::Warm>,
    at: &Local<'_>,
    sources: &mut Vec<NoiseSource<R>>,
) {
    let map = shape.map(at);
    let mut warm = Vec::new();
    K::noise(&shape.inner, heated, &at.remapped(&map), &mut warm);
    sources.extend(warm.into_iter().map(|source| NoiseSource {
        equations: source.equations.map(|equation| {
            equation.map(|local| crate::UnknownIndex::new(shape.outer(local.index())))
        }),
        ..source.map(|value| R::split(value).0)
    }));
}
