//! Type-erased [`GroupRun`] interface and its typed [`KernelRun`] implementation for a
//! single kernel kind and shape.

use std::any::Any;
use std::fmt;

use jolty_numerics::UNMAPPED;
use rayon::prelude::*;

use super::kind::{Fresh, Instance, Kind};
use super::layout::Group;
use super::linear::{
    Blocks, SliceChunk, Terms, as_refs, block_chunks, copy_long, evaluate_chunk, evaluate_linear,
    reborrow_mut, stamp_entries,
};
use super::twin::{KernelTwin, Seeds, Twinned};
use crate::{
    Choice, Contributions, DelayedEntry, Input, Layout, LogicLevel, NoiseSource, ParameterError,
    Stamp, UnknownIndex,
};

/// Mutable per-run state for one device group: prepared parameters and instance memory.
pub(super) trait GroupRun: fmt::Debug + Send {
    /// Appends a prepared instance produced by [`Made`](super::kind::Made) to the group.
    fn push(&mut self, made: Box<dyn Any>);

    /// Returns a boxed clone of the group run state.
    fn cloned(&self) -> Box<dyn GroupRun>;

    /// Appends circuit-mapped noise sources for `member` of `group` at `input` to `sources`.
    fn noise(
        &self,
        member: usize,
        group: &Group,
        input: &Input<'_>,
        sources: &mut Vec<NoiseSource>,
    );

    /// Writes the internal variables recorded at the last evaluation of `member` into `values`.
    fn variables(&self, member: usize, values: &mut [f64]);

    /// Returns the DC value of `member` if it is an independent source.
    fn dc_value(&self, member: usize) -> Option<f64>;

    /// Returns the residual derivative vector with respect to source value for an independent
    /// source in `group`, one entry per local unknown.
    fn excitation(&self, group: &Group) -> Vec<f64>;

    /// Starts logic driver `member` on `ramp` and returns the settling timestamp.
    fn drive(&mut self, member: usize, ramp: (f64, LogicLevel, f64)) -> Option<f64>;

    /// Returns the global current unknown of `member` in `group` if it is a logic driver.
    fn driven(&self, member: usize, group: &Group) -> Option<UnknownIndex>;

    /// Evaluates threshold margin and supply swing at `input` for logic detector `member` in
    /// `group`.
    fn sense(&self, member: usize, group: &Group, input: &Input<'_>) -> Option<(f64, f64)>;

    /// Appends non-smooth waveform breakpoints within `span` to `times`.
    fn breakpoints(&self, span: (f64, f64), times: &mut Vec<f64>);

    /// Appends device indices in `group` whose equations depend explicitly on time to `found`.
    fn time_varying(&self, group: &Group, found: &mut Vec<usize>);

    /// Appends circuit-mapped delayed conductance entries from `group` to `found`.
    fn delayed(&self, group: &Group, found: &mut Vec<DelayedEntry>);

    /// Finds the earliest interpolation fraction between `span.0` and `span.1` where any
    /// branch choice in `group` flips, using [`Choice::first_jump`] and scratch buffers
    /// `found`.
    fn first_jump(
        &mut self,
        group: &Group,
        span: (&Input<'_>, &Input<'_>),
        found: &mut [Vec<Choice>; 2],
    ) -> Option<f64>;

    /// Evaluates `member` of `of.0` in isolation at `input` into a zeroed `block`, returning
    /// whether voltage limiting triggered.
    ///
    /// When `of.1` is true and the group is linear, stamps entries at [`Input::ZERO`] and
    /// computes rows from cached linear terms; otherwise invokes the kernel directly.
    fn evaluate_alone(
        &mut self,
        member: usize,
        of: (&Group, bool),
        input: &Input<'_>,
        block: Blocks<'_>,
    ) -> bool;

    /// Builds a forward-mode sensitivity twin for `instance.0` at ambient temperature
    /// `instance.1` along `seeds`, allocating slab buffers for `sizes` (local unknowns and
    /// entries).
    ///
    /// Returns `None` when any seed path does not map to a real parameter.
    fn twin(
        &self,
        instance: (&Instance<'_>, f64),
        seeds: &Seeds,
        sizes: (usize, usize),
    ) -> Option<Box<dyn Twinned>>;

    /// Stages newly prepared parameters for `member` of `of.0` from `of.1` at ambient
    /// `temperature`, optionally verifying shape compatibility in `layout`.
    ///
    /// Returns whether the instance still matches the group's kernel, linearity, and shape.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when the instance parameters are out of range.
    fn prepare(
        &mut self,
        member: usize,
        of: (&Group, &Instance<'_>),
        temperature: f64,
        layout: Option<&mut Layout>,
    ) -> Result<bool, ParameterError>;

    /// Commits staged prepared parameters and resets memory for each updated instance,
    /// returning whether any staged update was applied.
    fn commit(&mut self) -> bool;

    /// Resets every instance's memory and clears cached linear terms.
    fn forget(&mut self);

    /// Discards all staged parameter updates without applying them.
    fn discard(&mut self);

    /// Stamps constant Jacobian entries for every prepared instance in linear `group` into
    /// `blocks` at [`Input::ZERO`], leaving unprepared instances zeroed.
    fn stamp(&mut self, group: &Group, blocks: Blocks<'_>);

    /// Evaluates every instance of `group` at `input` into `blocks.0` and mirrors high-fanout
    /// contributions into `blocks.1`, returning whether any instance limited voltages.
    ///
    /// Serial mode (`into` is `Some`) accumulates each block immediately into `into`; parallel
    /// mode (`into` is `None`) dispatches rayon tasks of at least `per_task` instances. Linear
    /// groups update only row vectors from cached entries.
    fn evaluate(
        &mut self,
        group: &Group,
        input: &Input<'_>,
        blocks: (Blocks<'_>, Blocks<'_>),
        into: Option<&mut Contributions>,
        per_task: usize,
    ) -> bool;
}

/// Typed per-run state for a group of kernel `K`.
pub(super) struct KernelRun<K: Kind> {
    shape: K::Shape,
    prepared: Vec<Option<K::Prepared>>,
    memory: Vec<K::Memory>,
    /// Prepared parameters staged for the next commit, paired with member index.
    staged: Vec<(usize, K::Prepared)>,
    /// Linear residual and charge multiplication terms extracted at the last stamp.
    terms: Terms,
}

impl<K: Kind> KernelRun<K> {
    /// Creates an unprepared group run for `members` instances sharing `shape`.
    pub(super) fn new(shape: K::Shape, members: usize) -> Self {
        Self {
            shape,
            prepared: std::iter::repeat_with(|| None).take(members).collect(),
            memory: vec![K::Memory::default(); members],
            staged: Vec::new(),
            terms: Terms::default(),
        }
    }
}

impl<K: Kind> fmt::Debug for KernelRun<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(K::NAME)
            .field("shape", &self.shape)
            .field("prepared", &self.prepared)
            .field("memory", &self.memory)
            .field("staged", &self.staged)
            .field("terms", &self.terms)
            .finish()
    }
}

impl<K: Kind> GroupRun for KernelRun<K> {
    fn push(&mut self, made: Box<dyn Any>) {
        let prepared = made.downcast::<Fresh<K>>().ok().map(|fresh| fresh.prepared);
        let memory = prepared
            .as_ref()
            .map(|prepared| K::memory(&self.shape, prepared));
        self.memory.push(memory.unwrap_or_default());
        self.prepared.push(prepared);
    }

    fn cloned(&self) -> Box<dyn GroupRun> {
        Box::new(Self {
            shape: self.shape.clone(),
            prepared: self.prepared.clone(),
            memory: self.memory.clone(),
            staged: self.staged.clone(),
            terms: self.terms.clone(),
        })
    }

    fn noise(
        &self,
        member: usize,
        group: &Group,
        input: &Input<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        let (Some(Some(prepared)), Some(memory), Some(map)) = (
            self.prepared.get(member),
            self.memory.get(member),
            group.map_of(member),
        ) else {
            return;
        };
        let start = sources.len();
        K::noise(&self.shape, prepared, memory, &input.local(map), sources);
        for source in sources.iter_mut().skip(start) {
            source.equations = source.equations.map(|equation| {
                equation.and_then(|local| {
                    map.get(local.index())
                        .filter(|&&unknown| unknown != UNMAPPED)
                        .map(|&unknown| UnknownIndex::new(unknown as usize))
                })
            });
        }
    }

    fn variables(&self, member: usize, values: &mut [f64]) {
        if let Some(memory) = self.memory.get(member) {
            K::variables(memory, values);
        }
    }

    fn dc_value(&self, member: usize) -> Option<f64> {
        K::dc_value(self.prepared.get(member)?.as_ref()?)
    }

    fn excitation(&self, group: &Group) -> Vec<f64> {
        let (unknowns, stride) = (group.unknowns(), group.stride());
        let mut rows: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; unknowns]);
        let mut entries: [Vec<f64>; 2] = std::array::from_fn(|_| vec![0.0; stride]);
        let [residual, scale, charge] = &mut rows;
        let [conductance, capacitance] = &mut entries;
        let mut stamp = Stamp::over([residual, scale, charge], [conductance, capacitance]);
        K::excite(&self.shape, &mut stamp);
        let [residual, ..] = rows;
        residual
    }

    fn drive(&mut self, member: usize, ramp: (f64, LogicLevel, f64)) -> Option<f64> {
        K::drive(self.memory.get_mut(member)?, ramp)
    }

    fn driven(&self, member: usize, group: &Group) -> Option<UnknownIndex> {
        let local = K::driven(&self.shape)?;
        let &unknown = group.map_of(member)?.get(local)?;
        (unknown != UNMAPPED).then(|| UnknownIndex::new(unknown as usize))
    }

    fn sense(&self, member: usize, group: &Group, input: &Input<'_>) -> Option<(f64, f64)> {
        let prepared = self.prepared.get(member)?.as_ref()?;
        K::sense(&self.shape, prepared, &input.local(group.map_of(member)?))
    }

    fn breakpoints(&self, span: (f64, f64), times: &mut Vec<f64>) {
        for prepared in self.prepared.iter().flatten() {
            K::breakpoints(prepared, span, times);
        }
    }

    fn time_varying(&self, group: &Group, found: &mut Vec<usize>) {
        for (member, prepared) in self.prepared.iter().enumerate() {
            if let (Some(prepared), Some(&device)) = (prepared, group.members.get(member))
                && K::varies_in_time(&self.shape, prepared)
            {
                found.push(device);
            }
        }
    }

    fn delayed(&self, group: &Group, found: &mut Vec<DelayedEntry>) {
        let mut local = Vec::new();
        for (member, prepared) in self.prepared.iter().enumerate() {
            let (Some(prepared), Some(map), Some(slots), Some(&device)) = (
                prepared,
                group.map_of(member),
                group.slots_of(member),
                group.members.get(member),
            ) else {
                continue;
            };
            local.clear();
            K::delayed(&self.shape, prepared, &mut local);
            for part in &local {
                let column = group
                    .layout
                    .entries()
                    .get(part.entry)
                    .and_then(|&(_, column)| map.get(column));
                if let (Some(&position), Some(&column)) = (slots.get(part.entry), column)
                    && position != UNMAPPED
                    && column != UNMAPPED
                {
                    found.push(DelayedEntry {
                        device,
                        position,
                        column,
                        delay: part.delay,
                        conductance: part.conductance,
                    });
                }
            }
        }
    }

    fn first_jump(
        &mut self,
        group: &Group,
        (before, after): (&Input<'_>, &Input<'_>),
        [earlier, later]: &mut [Vec<Choice>; 2],
    ) -> Option<f64> {
        let mut first: Option<f64> = None;
        let instances = self.prepared.iter().zip(&mut self.memory).enumerate();
        for (member, (prepared, memory)) in instances {
            let (Some(prepared), Some(map)) = (prepared, group.map_of(member)) else {
                continue;
            };
            earlier.clear();
            K::choices(&self.shape, prepared, memory, &before.local(map), earlier);
            if earlier.is_empty() {
                continue;
            }
            later.clear();
            K::choices(&self.shape, prepared, memory, &after.local(map), later);
            if let Some(fraction) = Choice::first_jump(earlier, later) {
                first = Some(first.map_or(fraction, |earliest| earliest.min(fraction)));
            }
        }
        first
    }

    fn evaluate_alone(
        &mut self,
        member: usize,
        (group, linear): (&Group, bool),
        input: &Input<'_>,
        ([residual, scale, charge], [conductance, capacitance]): Blocks<'_>,
    ) -> bool {
        let (Some(Some(prepared)), Some(memory), Some(map)) = (
            self.prepared.get(member),
            self.memory.get_mut(member),
            group.map_of(member),
        ) else {
            return false;
        };
        let mut stamp = Stamp::zeroed([residual, scale, charge], [conductance, capacitance]);
        if group.linear && linear {
            stamp_entries::<K>(&self.shape, prepared, map, &mut stamp);
            let at = input.local(map);
            let entries = [&*conductance, &*capacitance];
            self.terms
                .write(entries, &at, [residual, scale, charge], None);
            return false;
        }
        K::evaluate(&self.shape, prepared, memory, &input.local(map), &mut stamp);
        stamp.is_limited()
    }

    fn twin(
        &self,
        (instance, temperature): (&Instance<'_>, f64),
        seeds: &Seeds,
        sizes: (usize, usize),
    ) -> Option<Box<dyn Twinned>> {
        let parameters = instance.parameters::<K>()?;
        let temperature = instance.own.temperature(temperature).ok()?;
        let made = (&self.shape, parameters, temperature);
        // The seeds' count rounded up to one of the widths the twins are compiled for.
        match seeds.count() {
            1 => KernelTwin::<K, 1>::made(made, seeds, sizes),
            2 => KernelTwin::<K, 2>::made(made, seeds, sizes),
            3 | 4 => KernelTwin::<K, 4>::made(made, seeds, sizes),
            5..=Seeds::MAX => KernelTwin::<K, { Seeds::MAX }>::made(made, seeds, sizes),
            _ => None,
        }
    }

    fn prepare(
        &mut self,
        member: usize,
        (group, instance): (&Group, &Instance<'_>),
        temperature: f64,
        layout: Option<&mut Layout>,
    ) -> Result<bool, ParameterError> {
        let Some(parameters) = instance.parameters::<K>() else {
            return Ok(false);
        };
        let prepared = K::prepare(parameters, instance.own.temperature(temperature)?)?;
        if K::linear(&self.shape, &prepared) != group.linear {
            return Ok(false);
        }
        if let Some(layout) = layout {
            layout.clear(group.layout.terminals());
            if K::layout(parameters, &prepared, layout)? != self.shape {
                return Ok(false);
            }
        }
        self.staged.push((member, prepared));
        Ok(true)
    }

    fn commit(&mut self) -> bool {
        let took = !self.staged.is_empty();
        for (member, prepared) in self.staged.drain(..) {
            if let Some(memory) = self.memory.get_mut(member) {
                *memory = K::memory(&self.shape, &prepared);
            }
            if let Some(slot) = self.prepared.get_mut(member) {
                *slot = Some(prepared);
            }
        }
        took
    }

    fn forget(&mut self) {
        self.terms = Terms::default();
        for (memory, prepared) in self.memory.iter_mut().zip(&self.prepared) {
            *memory = prepared
                .as_ref()
                .map_or_default(|prepared| K::memory(&self.shape, prepared));
        }
    }

    fn discard(&mut self) {
        self.staged.clear();
    }

    fn stamp(&mut self, group: &Group, (rows, mut entries): Blocks<'_>) {
        let (unknowns, stride) = (group.unknowns(), group.stride());
        if unknowns == 0 {
            return;
        }
        let blocks = block_chunks(rows, reborrow_mut(&mut entries), (unknowns, stride));
        let instances = self.prepared.iter().zip(group.nodes.chunks(unknowns));
        for ((row, entry), (prepared, map)) in blocks.zip(instances) {
            let mut stamp = Stamp::zeroed(row, entry);
            if let Some(prepared) = prepared {
                stamp_entries::<K>(&self.shape, prepared, map, &mut stamp);
            }
        }
        self.terms = Terms::from_layout(&group.layout, stride, as_refs(&entries));
    }

    fn evaluate(
        &mut self,
        group: &Group,
        input: &Input<'_>,
        ((mut rows, mut entries), copies): (Blocks<'_>, Blocks<'_>),
        into: Option<&mut Contributions>,
        per_task: usize,
    ) -> bool {
        let sizes = (group.unknowns(), group.stride());
        if sizes.0 == 0 {
            return false;
        }
        if group.linear {
            let blocks = (rows, as_refs(&entries));
            evaluate_linear(
                (group, &self.terms),
                input,
                (blocks, copies),
                into,
                per_task,
            );
            return false;
        }
        let shape = &self.shape;
        let Some(into) = into else {
            let whole = SliceChunk {
                first: 0,
                count: group.members.len(),
                rows,
                entries,
                nodes: &group.nodes,
                copies,
                memory: self.memory.as_mut_slice(),
                prepared: self.prepared.as_slice(),
            };
            return rayon::iter::split(whole, |part| part.halve(per_task, sizes, &group.copied))
                .map(|mut chunk| {
                    let row_blocks = reborrow_mut(&mut chunk.rows);
                    let entry_blocks = reborrow_mut(&mut chunk.entries);
                    let limited = evaluate_chunk::<K>(
                        shape,
                        sizes,
                        input,
                        ((row_blocks, entry_blocks), chunk.memory, chunk.prepared),
                        (chunk.nodes, &[]),
                        None,
                    );
                    let written = (as_refs(&chunk.rows), as_refs(&chunk.entries));
                    copy_long(sizes, written, chunk.copies, (&group.copied, chunk.first));
                    limited
                })
                .reduce(|| false, |one, other| one | other);
        };
        let blocks = (reborrow_mut(&mut rows), reborrow_mut(&mut entries));
        let instances = (blocks, self.memory.as_mut_slice(), self.prepared.as_slice());
        let limited = evaluate_chunk::<K>(
            shape,
            sizes,
            input,
            instances,
            (&group.nodes, &group.slots),
            Some(into),
        );
        let written = (as_refs(&rows), as_refs(&entries));
        copy_long(sizes, written, copies, (&group.copied, 0));
        limited
    }
}
