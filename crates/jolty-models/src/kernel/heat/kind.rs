//! A kernel's instances that heat, as their groups evaluate them.
//!
//! Each is the kernel's instance at the temperature of a thermal net, the last of its
//! terminals: it computes with every number's derivative with respect to its temperature
//! beside it, its own local unknowns' contributions extended along the temperature's last
//! step and folded onto the heated instance's, and its power, the sum over its voltages of
//! each times the current into the instance there, flows into the net as heat.

use std::fmt;
use std::marker::PhantomData;

use jolty_numerics::UNMAPPED;

use super::{Heating, LOWEST_TEMPERATURE, MOST_TEMPERATURE_STEP};
use crate::group::Kind;
use crate::lift::{constant, lift};
use crate::{
    Choice, Delayed, Gradient, Hyper, Kernel, Layout, Local, LogicLevel, Nature, NoiseSource,
    ParameterError, Regions, Stamp,
};

/// The instances of the kernel `K` that heat.
pub(crate) struct HeatedKernel<K>(PhantomData<fn() -> K>);

/// Where a heated instance's contributions go among its local unknowns: those of its
/// kernel's instance, which are its own but for its thermal terminal, after the kernel's
/// terminals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HeatedShape<S> {
    /// The kernel's instance's shape.
    inner: S,
    /// The kernel's instance's terminals, before the thermal terminal, the local index of
    /// the latter.
    terminals: usize,
    /// Whether each of the kernel's instance's local unknowns is a voltage, whose current
    /// the power reads.
    voltages: Vec<bool>,
    /// Each local entry of the kernel's instance, in the order of the columns then the
    /// rows of its local unknowns: its row and column there, its index among the kernel's
    /// instance's local entries, and its local entry in the heated instance.
    entries: Vec<(usize, usize, usize, usize)>,
    /// The local entry of each of the kernel's instance's local unknowns' equations at the
    /// thermal terminal.
    column: Vec<usize>,
    /// The local entry of the thermal terminal's equation at each of them.
    row: Vec<usize>,
    /// The local entry of the thermal terminal's equation at itself.
    diagonal: usize,
}

impl<S> HeatedShape<S> {
    /// The number of the kernel's instance's local unknowns.
    fn unknowns(&self) -> usize {
        self.voltages.len()
    }

    /// The heated instance's local index of the kernel's instance's `local`.
    const fn outer(&self, local: usize) -> usize {
        if local < self.terminals {
            local
        } else {
            local + 1
        }
    }
}

/// A heated instance's numbers, computing with `R`: its parameters, which it prepares again
/// as its temperature moves, the ambient temperature it is at when the thermal net is at
/// zero, and its numbers there.
pub(crate) struct HeatedPrepared<K: Kernel, R: Heating> {
    parameters: K::Parameters<R::Warm>,
    ambient: f64,
    at_ambient: K::Prepared<R::Warm>,
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
    /// The numbers of an instance with `parameters` at the ambient temperature `ambient`.
    fn new(parameters: &K::Parameters<R::Warm>, ambient: f64) -> Result<Self, ParameterError> {
        Ok(Self {
            at_ambient: K::prepare(parameters, R::seed(ambient))?,
            parameters: parameters.clone(),
            ambient,
        })
    }
}

/// What a heated instance remembers, computing with `R`: its kernel's instance's numbers at
/// the temperature of its last evaluation, from which the next step is limited, and its
/// memory, and the room it evaluates in.
pub(crate) struct HeatedMemory<K: Kernel, R: Heating> {
    prepared: Option<K::Prepared<R::Warm>>,
    inner: K::Memory,
    last: f64,
    /// Whether `prepared` was last prepared with the thermal net seeded by a twin.
    seeded: bool,
    /// The circuit's unknown of each of the kernel's instance's local unknowns.
    map: Vec<u32>,
    /// The kernel's instance's block: its residuals, scales and charges, then its
    /// conductances and capacitances.
    block: Vec<R::Warm>,
    /// The derivative of the power with respect to each of its local unknowns.
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
    /// What an instance of `shape` with the numbers `prepared` remembers before its first
    /// evaluation, at the ambient temperature.
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

    /// Reads where the kernel's instance of `shape` lies at `at`: the circuit's unknown of
    /// each of its local unknowns.
    fn read(&mut self, shape: &HeatedShape<K::Shape>, at: &Local<'_>) {
        let map = at.map();
        let unknowns = shape.unknowns();
        self.map.resize(unknowns, UNMAPPED);
        for (local, unknown) in self.map.iter_mut().enumerate() {
            *unknown = map.get(shape.outer(local)).copied().unwrap_or(UNMAPPED);
        }
    }
}

/// The parts of a kernel's instance's block: its residuals, scales and charges, and its
/// conductances and capacitances, of `unknowns` local unknowns and `stored` places for its
/// local entries; `None` when the block is too short.
fn parts<T>(block: &mut [T], unknowns: usize, stored: usize) -> Option<[&mut [T]; 5]> {
    let (residual, rest) = block.split_at_mut_checked(unknowns)?;
    let (scale, rest) = rest.split_at_mut_checked(unknowns)?;
    let (charge, rest) = rest.split_at_mut_checked(unknowns)?;
    let (conductance, rest) = rest.split_at_mut_checked(stored)?;
    let capacitance = rest.get_mut(..stored)?;
    Some([residual, scale, charge, conductance, capacitance])
}

/// Claims in `layout`, whose terminals are the kernel's instance's then the thermal
/// terminal, the heated instance's internal unknowns and entries, those of the kernel's
/// instance, which `inner` holds, as its own, and returns its shape.
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

/// Moves a heated instance of `shape`, with the numbers `prepared`, to its temperature at
/// `at`, no further than [`MOST_TEMPERATURE_STEP`] from its last evaluation's when `at`
/// limits its steps and no lower than [`LOWEST_TEMPERATURE`], preparing its kernel's
/// instance again there into `memory`, and returns how far its applied temperature lies
/// beyond the one it is evaluated at. Marks `stamp` limited when that is not the applied
/// one.
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
            memory.last - MOST_TEMPERATURE_STEP,
            memory.last + MOST_TEMPERATURE_STEP,
        )
    } else {
        applied
    };
    let temperature = stepped.max(LOWEST_TEMPERATURE);
    let seeding = R::SEEDS > 0 && at.seeds_unknowns();
    if temperature.to_bits() != memory.last.to_bits() || seeding || memory.seeded {
        // The temperature moves with the thermal net when a twin seeds it.
        let net = shape.terminals;
        let moving = R::seed(temperature) + (at.unknown::<R::Warm>(net) - at.value(net));
        if let Ok(heated) = K::prepare(&prepared.parameters, moving) {
            memory.prepared = Some(heated);
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
        HeatedPrepared::new(&constant(parameters), temperature)
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

    /// Never linear: what it dissipates heats it, and its heat changes its equations.
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

    /// The kernel's instance's noise at its temperature at its last evaluation.
    fn noise(
        shape: &Self::Shape,
        prepared: &Self::Prepared,
        memory: &Self::Memory,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        let heated = memory.prepared.as_ref().unwrap_or(&prepared.at_ambient);
        let map: Vec<u32> = (0..shape.unknowns())
            .map(|local| {
                at.map()
                    .get(shape.outer(local))
                    .copied()
                    .unwrap_or(UNMAPPED)
            })
            .collect();
        let start = sources.len();
        K::noise(&shape.inner, heated, &at.remapped(&map), sources);
        for source in sources.iter_mut().skip(start) {
            source.equations = source.equations.map(|equation| {
                equation.map(|local| crate::UnknownIndex::new(shape.outer(local.index())))
            });
        }
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
        let heated = memory.prepared.as_ref().unwrap_or(&prepared.at_ambient);
        let HeatedMemory { inner, map, .. } = memory;
        K::choices(
            &shape.inner,
            heated,
            inner,
            &at.remapped(map.as_slice()),
            found,
        );
    }

    fn level(prepared: &Self::Prepared) -> Option<f64> {
        K::level(&prepared.at_ambient)
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
        let map: Vec<u32> = (0..shape.unknowns())
            .map(|local| {
                at.map()
                    .get(shape.outer(local))
                    .copied()
                    .unwrap_or(UNMAPPED)
            })
            .collect();
        K::sense(&shape.inner, &prepared.at_ambient, &at.remapped(&map))
    }

    type Twin<const SEEDS: usize> = (
        HeatedPrepared<K, Gradient<SEEDS>>,
        HeatedMemory<K, Gradient<SEEDS>>,
    );

    fn twin<const SEEDS: usize>(
        shape: &Self::Shape,
        parameters: &Self::Parameters,
        seeds: &[(&str, [f64; SEEDS])],
        temperature: f64,
    ) -> Option<Self::Twin<SEEDS>> {
        let lifted: K::Parameters<Hyper<SEEDS>> = lift(parameters, seeds, Hyper::new).ok()?;
        let prepared = HeatedPrepared::<K, Gradient<SEEDS>>::new(&lifted, temperature).ok()?;
        let memory = HeatedMemory::new(shape, &prepared);
        Some((prepared, memory))
    }

    fn evaluate_twin<const SEEDS: usize>(
        shape: &Self::Shape,
        (prepared, memory): &mut Self::Twin<SEEDS>,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, Gradient<SEEDS>>,
    ) {
        evaluate::<K, Gradient<SEEDS>>(shape, prepared, memory, at, stamp);
    }

    #[cfg(any(test, feature = "testing"))]
    fn lift_entries(parameters: &Self::Parameters) -> Vec<(String, f64)> {
        <K as Kind>::lift_entries(parameters)
    }
}
