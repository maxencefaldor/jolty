//! Slab plane chunking, linear-group term evaluation, long-value copying, and direct
//! contribution accumulation.

use rayon::prelude::*;

use super::kind::Kind;
use super::layout::{Copied, Group};
use crate::{Contributions, Input, Layout, Local, Stamp};

/// A group's blocks in the slab: the residuals, scales and charges of its instances' local
/// unknowns, and the conductances and capacitances of their local entries.
pub(super) type Blocks<'a> = ([&'a mut [f64]; 3], [&'a mut [f64]; 2]);

/// A linear group's blocks: the rows of its instances' local unknowns, which their
/// evaluations write, and the entries of their local entries, which they read.
pub(super) type LinearBlocks<'a> = ([&'a mut [f64]; 3], [&'a [f64]; 2]);

/// The instances of a chunk of a group: their blocks, their memory and their prepared
/// numbers.
pub(super) type Chunk<'a, K> = (
    Blocks<'a>,
    &'a mut [<K as Kind>::Memory],
    &'a [Option<<K as Kind>::Prepared>],
);

/// A contiguous slice of one plane of a block, shared or mutable, splittable in two.
pub(super) trait PlaneSlice: Default {
    fn len(&self) -> usize;
    fn split_at(self, mid: usize) -> (Self, Self);
}

impl PlaneSlice for &[f64] {
    #[inline]
    fn len(&self) -> usize {
        <[f64]>::len(self)
    }

    #[inline]
    fn split_at(self, mid: usize) -> (Self, Self) {
        <[f64]>::split_at(self, mid)
    }
}

impl PlaneSlice for &mut [f64] {
    #[inline]
    fn len(&self) -> usize {
        <[f64]>::len(self)
    }

    #[inline]
    fn split_at(self, mid: usize) -> (Self, Self) {
        <[f64]>::split_at_mut(self, mid)
    }
}

/// Whether every plane in `planes` holds at least `count` elements.
#[inline]
fn all_fit<S: PlaneSlice, const N: usize>(planes: &[S; N], count: usize) -> bool {
    planes.iter().all(|plane| plane.len() >= count)
}

/// Takes up to `count` leading elements from each plane, leaving the remainder in place.
#[inline]
fn take_prefix<S: PlaneSlice, const N: usize>(planes: &mut [S; N], count: usize) -> [S; N] {
    planes.each_mut().map(|plane| {
        let whole = std::mem::take(plane);
        let (first, rest) = whole.split_at(count.min(whole.len()));
        *plane = rest;
        first
    })
}

/// Iterates over equal-sized chunks of `N` parallel plane slices simultaneously.
struct PlaneChunks<S, const N: usize> {
    planes: [S; N],
    chunk: usize,
}

impl<S: PlaneSlice, const N: usize> Iterator for PlaneChunks<S, N> {
    type Item = [S; N];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let shortest = self.planes.iter().map(PlaneSlice::len).min()?;
        let take = shortest.min(self.chunk);
        (take > 0).then(|| take_prefix(&mut self.planes, take))
    }
}

/// Iterates over `(row_block, entry_block)` chunks across all five planes of a group.
#[inline]
pub(super) fn block_chunks<R: PlaneSlice, E: PlaneSlice>(
    rows: [R; 3],
    entries: [E; 2],
    (unknowns, stride): (usize, usize),
) -> impl Iterator<Item = ([R; 3], [E; 2])> {
    let rows = PlaneChunks {
        planes: rows,
        chunk: unknowns,
    };
    let entries = PlaneChunks {
        planes: entries,
        chunk: stride,
    };
    rows.zip(entries)
}

/// A contiguous range of a group's instances for one parallel task: its block planes, local
/// node map, copy target slices, and optional per-instance state.
pub(super) struct SliceChunk<'a, E, M, P> {
    pub(super) first: usize,
    pub(super) count: usize,
    pub(super) rows: [&'a mut [f64]; 3],
    pub(super) entries: [E; 2],
    pub(super) nodes: &'a [u32],
    pub(super) copies: Blocks<'a>,
    pub(super) memory: &'a mut [M],
    pub(super) prepared: &'a [P],
}

impl<E: PlaneSlice, M, P> SliceChunk<'_, E, M, P> {
    /// Splits this chunk in two at a multiple of `per_task` instances, or returns it whole
    /// when it holds at most `per_task` instances or any slice is shorter than expected.
    pub(super) fn halve(
        mut self,
        per_task: usize,
        (unknowns, stride): (usize, usize),
        [copied_rows, copied_entries]: &[Copied; 2],
    ) -> (Self, Option<Self>) {
        let per_task = per_task.max(1);
        if self.count <= per_task {
            return (self, None);
        }
        let chunks = self.count.div_ceil(per_task);
        let left_count = chunks / 2 * per_task;
        let mid = self.first + left_count;
        let (row_at, entry_at) = (left_count * unknowns, left_count * stride);
        let (Some(row_copy_at), Some(entry_copy_at)) = (
            copied_rows.count_between(self.first, mid),
            copied_entries.count_between(self.first, mid),
        ) else {
            return (self, None);
        };
        let (row_copies, entry_copies) = &self.copies;
        let fits = all_fit(&self.rows, row_at)
            && all_fit(&self.entries, entry_at)
            && all_fit(row_copies, row_copy_at)
            && all_fit(entry_copies, entry_copy_at)
            && self.nodes.len() >= row_at;
        if !fits {
            return (self, None);
        }
        let (nodes, rest_nodes) = self.nodes.split_at(row_at);
        let memory_at = left_count.min(self.memory.len());
        let whole_memory = std::mem::take(&mut self.memory);
        let (memory, rest_memory) = whole_memory.split_at_mut(memory_at);
        let prepared_at = left_count.min(self.prepared.len());
        let (prepared, rest_prepared) = self.prepared.split_at(prepared_at);
        let left = Self {
            first: self.first,
            count: left_count,
            rows: take_prefix(&mut self.rows, row_at),
            entries: take_prefix(&mut self.entries, entry_at),
            nodes,
            copies: (
                take_prefix(&mut self.copies.0, row_copy_at),
                take_prefix(&mut self.copies.1, entry_copy_at),
            ),
            memory,
            prepared,
        };
        self.first = mid;
        self.count -= left_count;
        self.nodes = rest_nodes;
        self.memory = rest_memory;
        self.prepared = rest_prepared;
        (left, Some(self))
    }
}

/// Reborrows `N` mutable plane slices for a shorter lifetime.
#[inline]
pub(super) fn reborrow_mut<'a, const N: usize>(
    planes: &'a mut [&mut [f64]; N],
) -> [&'a mut [f64]; N] {
    planes.each_mut().map(|plane| &mut **plane)
}

/// Views `N` mutable plane slices as shared slices.
#[inline]
pub(super) fn as_refs<'a, const N: usize>(planes: &'a [&mut [f64]; N]) -> [&'a [f64]; N] {
    planes.each_ref().map(|plane| &**plane)
}

/// The values of each of `planes` from `at` on, which keep those before it; none of any
/// when one holds fewer.
pub(super) fn beyond<S: PlaneSlice, const N: usize>(planes: &mut [S; N], at: usize) -> [S; N] {
    let Some(before) = split(planes, at) else {
        return std::array::from_fn(|_| S::default());
    };
    std::mem::replace(planes, before)
}

/// The first `count` values of each of `planes`, which keep the rest, or `None` when one
/// holds fewer.
#[inline]
pub(super) fn split<S: PlaneSlice, const N: usize>(
    planes: &mut [S; N],
    count: usize,
) -> Option<[S; N]> {
    all_fit(planes, count).then(|| take_prefix(planes, count))
}

/// The blocks of `group`'s instances, the first values of the slab's `planes`, which keep
/// the rest, and its copies, the first of `copies`, likewise; `None` when one holds too
/// few. A group that copies nothing takes no copies, so that a circuit with no long value
/// pays nothing for them.
#[inline]
pub(super) fn blocks_of<'a>(
    group: &Group,
    (rows, entries): (&mut [&'a mut [f64]; 3], &mut [&'a mut [f64]; 2]),
    (row_copies, entry_copies): (&mut [&'a mut [f64]; 3], &mut [&'a mut [f64]; 2]),
) -> Option<(Blocks<'a>, Blocks<'a>)> {
    let count = group.members.len();
    let blocks = (
        split(rows, count * group.unknowns())?,
        split(entries, count * group.stride())?,
    );
    let [copied_rows, copied_entries] = &group.copied;
    if copied_rows.len() + copied_entries.len() == 0 {
        return Some((blocks, Blocks::default()));
    }
    let copies = (
        split(row_copies, copied_rows.len())?,
        split(entry_copies, copied_entries.len())?,
    );
    Some((blocks, copies))
}

/// Evaluates each instance of a chunk of a group of `shape`, of `sizes` local unknowns and
/// places for local entries each, at `input`, into its block, which it zeroes first, and
/// adds each block to `into`, when given, as soon as it is written, through the circuit's
/// unknowns and stored positions of the instances' local ones, `maps`. Returns whether an
/// instance limited its voltages.
pub(super) fn evaluate_chunk<K: Kind>(
    shape: &K::Shape,
    (unknowns, stride): (usize, usize),
    input: &Input<'_>,
    ((mut rows, mut entries), memory, prepared): Chunk<'_, K>,
    (nodes, slots): (&[u32], &[u32]),
    mut into: Option<&mut Contributions>,
) -> bool {
    for plane in rows.iter_mut().chain(&mut entries) {
        plane.fill(0.0);
    }
    let blocks = block_chunks(rows, entries, (unknowns, stride));
    let maps = nodes.chunks(unknowns).zip(
        slots
            .chunks(stride)
            .map(Some)
            .chain(std::iter::repeat(None)),
    );
    let instances = memory.iter_mut().zip(prepared).zip(maps);
    let mut limited = false;
    for ((mut row, mut entry), ((memory, prepared), (map, slots))) in blocks.zip(instances) {
        let Some(prepared) = prepared else { continue };
        let at = input.local(map);
        let mut stamp = Stamp::over(reborrow_mut(&mut row), reborrow_mut(&mut entry));
        K::evaluate(shape, prepared, memory, &at, &mut stamp);
        limited |= stamp.is_limited();
        if let (Some(into), Some(slots)) = (into.as_deref_mut(), slots) {
            add(into, (map, slots), (as_refs(&row), as_refs(&entry)));
        }
    }
    limited
}

/// Writes into `stamp` what a linear instance of `shape` with the numbers `prepared`, whose
/// local unknowns are the circuit's of `map`, contributes at [`Input::ZERO`], from its
/// initial memory: its entries, which are the same at every state.
pub(super) fn stamp_entries<K: Kind>(
    shape: &K::Shape,
    prepared: &K::Prepared,
    map: &[u32],
    stamp: &mut Stamp<'_>,
) {
    let mut memory = K::memory(shape, prepared);
    K::evaluate(shape, prepared, &mut memory, &Input::ZERO.local(map), stamp);
}

/// Writes the rows of each block of `group`, which is linear, of `blocks`, the first of the
/// pair, from its entries, at `input`, copying its contributions to a long value to their
/// places among the second, the group's copies: one after another, each block's rows added
/// to `into` as soon as they are written, or without `into`, in parallel, `per_task`
/// instances a task.
pub(super) fn evaluate_linear(
    (group, terms): (&Group, &Terms),
    input: &Input<'_>,
    ((mut rows, entries), copies): (LinearBlocks<'_>, Blocks<'_>),
    into: Option<&mut Contributions>,
    per_task: usize,
) {
    let sizes = (group.unknowns(), group.stride());
    let Some(into) = into else {
        let whole: SliceChunk<'_, &[f64], (), ()> = SliceChunk {
            first: 0,
            count: group.members.len(),
            rows,
            entries,
            nodes: &group.nodes,
            copies,
            memory: &mut [],
            prepared: &[],
        };
        let tasks = rayon::iter::split(whole, |part| part.halve(per_task, sizes, &group.copied));
        tasks.for_each(|mut chunk| {
            let blocks = (reborrow_mut(&mut chunk.rows), chunk.entries);
            linear_chunk(terms, sizes, input, blocks, chunk.nodes, None);
            let written = (as_refs(&chunk.rows), chunk.entries);
            copy_long(sizes, written, chunk.copies, (&group.copied, chunk.first));
        });
        return;
    };
    let blocks = (reborrow_mut(&mut rows), entries);
    linear_chunk(terms, sizes, input, blocks, &group.nodes, Some(into));
    let written = (as_refs(&rows), entries);
    copy_long(sizes, written, copies, (&group.copied, 0));
}

/// Writes the rows of each block of a chunk of a linear group, of `sizes` local unknowns
/// and places for local entries each, whose residuals and charges sum `terms`, at `input`,
/// from the block's entries, and adds each row to `into`, when given, as soon as it is
/// written, through the circuit's unknowns of the instances' local ones, `nodes`.
fn linear_chunk(
    terms: &Terms,
    (unknowns, stride): (usize, usize),
    input: &Input<'_>,
    (rows, entries): LinearBlocks<'_>,
    nodes: &[u32],
    mut into: Option<&mut Contributions>,
) {
    let blocks = block_chunks(rows, entries, (unknowns, stride));
    for ((row, entry), map) in blocks.zip(nodes.chunks(unknowns)) {
        let at = input.local(map);
        terms.write(entry, &at, row, into.as_deref_mut());
    }
}

/// The terms of the residuals and the charges of a linear group's instances: for its
/// conductances, then its capacitances, each local unknown's row with an entry that is not
/// zero in some instance, and those of its entries, each with its column, in the order of
/// their local indices. Every other term of the group's is zero at every state, as is the
/// row of one that has none: leaving it out of a sum, which starts from zero and so is
/// never `-0`, changes no bit of it.
#[derive(Debug, Clone, Default)]
pub(super) struct Terms {
    /// For each plane, each row's local unknown, and where its entries begin and end among
    /// the plane's.
    rows: [Vec<(usize, usize, usize)>; 2],
    /// For each plane, each row's entries, row after row: the local unknown of its column
    /// and its local index.
    entries: [Vec<(usize, usize)>; 2],
}

impl Terms {
    /// The terms of the instances of `layout` whose blocks' entries, a `stride` apart, are
    /// `[conductances, capacitances]`.
    pub(super) fn from_layout(layout: &Layout, stride: usize, planes: [&[f64]; 2]) -> Self {
        let mut terms = Self::default();
        let claimed = layout.entries();
        for (plane, values) in planes.into_iter().enumerate() {
            let (rows, entries) = (&mut terms.rows[plane], &mut terms.entries[plane]);
            for row in 0..layout.unknowns() {
                let start = entries.len();
                for (entry, &(_, column)) in claimed
                    .iter()
                    .enumerate()
                    .filter(|&(_, &(claim, _))| claim == row)
                {
                    let mut instances = values.iter().skip(entry).step_by(stride.max(1));
                    if instances.any(|&value| value != 0.0) {
                        entries.push((column, entry));
                    }
                }
                if entries.len() > start {
                    rows.push((row, start, entries.len()));
                }
            }
        }
        terms
    }

    /// Writes the rows of one instance's block at `at` from its entries,
    /// `[conductances, capacitances]`: each listed residual and charge the sum of its terms,
    /// its entries times the local unknowns of their columns, from zero, in their order,
    /// and each listed residual's scale its magnitude, as a kernel's [`Stamp`] keeps the
    /// largest of the one term it adds; and adds each, when `into` is given, to its
    /// unknown's there, through `at`'s map, as soon as it is written. Those on the
    /// reference, [`UNMAPPED`](jolty_numerics::UNMAPPED), go nowhere.
    #[inline]
    pub(super) fn write(
        &self,
        [conductance, capacitance]: [&[f64]; 2],
        at: &Local<'_>,
        [residual, scale, charge]: [&mut [f64]; 3],
        mut into: Option<&mut Contributions>,
    ) {
        let sum = |plane: usize, values: &[f64], (start, end): (usize, usize)| {
            self.entries[plane]
                .get(start..end)
                .unwrap_or(&[])
                .iter()
                .fold(0.0, |sum, &(column, entry)| {
                    sum + values.get(entry).copied().unwrap_or(0.0) * at.value(column)
                })
        };
        let unknown = |row: usize| {
            at.map()
                .get(row)
                .map_or(usize::MAX, |&unknown| unknown as usize)
        };
        for &(row, start, end) in &self.rows[0] {
            let total = sum(0, conductance, (start, end));
            let largest = 0.0_f64.max(total.abs());
            if let (Some(residual), Some(scale)) = (residual.get_mut(row), scale.get_mut(row)) {
                (*residual, *scale) = (total, largest);
            }
            if let Some(into) = into.as_deref_mut()
                && let (Some(sum), Some(scale)) = (
                    into.residual.get_mut(unknown(row)),
                    into.scale.get_mut(unknown(row)),
                )
            {
                *sum += total;
                *scale = scale.max(largest);
            }
        }
        for &(row, start, end) in &self.rows[1] {
            let total = sum(1, capacitance, (start, end));
            if let Some(charge) = charge.get_mut(row) {
                *charge = total;
            }
            if let Some(into) = into.as_deref_mut()
                && let Some(held) = into.charge.get_mut(unknown(row))
            {
                *held += total;
            }
        }
    }
}

/// Copies the contributions to a long value of each instance of a chunk of a group, of
/// `sizes` local unknowns and places for local entries each, from its block of `blocks`,
/// as [`evaluate_chunk`] wrote it, zero for an instance not prepared, to its places among
/// the chunk's `copies`: those of the locals `copied` lists, for the instances from
/// `first` on. A group with no copies returns at once, inlined where it is called, so that
/// a circuit with no long value pays for no pass over its blocks.
#[inline]
pub(super) fn copy_long(
    sizes: (usize, usize),
    blocks: ([&[f64]; 3], [&[f64]; 2]),
    copies: Blocks<'_>,
    (lists, first): (&[Copied; 2], usize),
) {
    let [rows, entries] = lists;
    if rows.len() + entries.len() > 0 {
        copy_each(sizes, blocks, copies, (lists, first));
    }
}

/// [`copy_long`] of a group with copies: each instance's after another.
#[inline(never)]
fn copy_each(
    (unknowns, stride): (usize, usize),
    (rows, entries): ([&[f64]; 3], [&[f64]; 2]),
    (mut row_copies, mut entry_copies): Blocks<'_>,
    ([copied_rows, copied_entries], first): (&[Copied; 2], usize),
) {
    let bases = (
        copied_rows.starts.get(first).copied().unwrap_or(0),
        copied_entries.starts.get(first).copied().unwrap_or(0),
    );
    let blocks = block_chunks(rows, entries, (unknowns, stride));
    for (member, (row, entry)) in (first..).zip(blocks) {
        copied_rows.copy((member, bases.0), row, &mut row_copies);
        copied_entries.copy((member, bases.1), entry, &mut entry_copies);
    }
}

/// Adds `[conductance, capacitance]` values into their stored positions of `into`, given by
/// `slots`, in order; positions of [`UNMAPPED`](jolty_numerics::UNMAPPED) go nowhere.
pub(super) fn add_entries(
    into: &mut Contributions,
    slots: &[u32],
    [conductance, capacitance]: [&[f64]; 2],
) {
    let stored = conductance.iter().zip(capacitance.iter());
    for (&slot, (&conductance, &capacitance)) in slots.iter().zip(stored) {
        let slot = slot as usize;
        if let (Some(sum), Some(held)) = (
            into.conductance.get_mut(slot),
            into.capacitance.get_mut(slot),
        ) {
            *sum += conductance;
            *held += capacitance;
        }
    }
}

/// Adds one instance's block to `into`: each local unknown's residual, scale and charge to
/// its unknown's of `map`, and each local entry's conductance and capacitance to its
/// stored position of `slots`, in the order of their local indices; those on the
/// reference, [`UNMAPPED`](jolty_numerics::UNMAPPED), go nowhere.
fn add(
    into: &mut Contributions,
    (map, slots): (&[u32], &[u32]),
    ([residual, scale, charge], entries): ([&[f64]; 3], [&[f64]; 2]),
) {
    let rows = residual.iter().zip(scale.iter()).zip(charge.iter());
    for (&row, ((&residual, &scale), &charge)) in map.iter().zip(rows) {
        let row = row as usize;
        if let (Some(sum), Some(largest), Some(held)) = (
            into.residual.get_mut(row),
            into.scale.get_mut(row),
            into.charge.get_mut(row),
        ) {
            *sum += residual;
            *largest = largest.max(scale);
            *held += charge;
        }
    }
    add_entries(into, slots, entries);
}
