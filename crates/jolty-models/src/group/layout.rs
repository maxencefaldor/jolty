//! Structural group layout: mapping device instances onto slab blocks, long-value copy
//! ranges, and gather maps.

use std::sync::OnceLock;

use jolty_numerics::reduce::CHUNK;
use jolty_numerics::{Entry, GatherMap, Pattern, UNMAPPED};

use super::kind::{Instance, Shaped};
use super::runs::{Runs, Slab};
use crate::{Binding, Layout, Node, ParameterError};

/// Immutable structural metadata for instances sharing a kernel, shape, and linearity.
#[derive(Debug)]
pub(super) struct Group {
    shaped: Box<dyn Shaped>,
    pub(super) layout: Layout,
    /// Flag indicating linear instances (per [`Kernel::linear`](crate::Kernel::linear)) that
    /// stamp entries once per run into the base and derive rows from cached entries.
    pub(super) linear: bool,
    /// Circuit device indices of group members in registration order.
    pub(super) members: Vec<usize>,
    /// Global unknown indices for each member's local unknowns, or [`UNMAPPED`] for ground.
    pub(super) nodes: Vec<u32>,
    /// Temporary entry claims held before the final sparsity pattern is built.
    claims: Vec<Option<Entry>>,
    /// Stored sparsity positions for each member's local entries (stride-aligned), or
    /// [`UNMAPPED`] when connected to ground.
    pub(super) slots: Vec<u32>,
    /// Starting slab offsets `(unknowns_start, entries_start)` for the group's blocks.
    pub(super) starts: (usize, usize),
    /// Copy descriptors `[unknowns, entries]` for high-fanout targets mirrored beyond blocks.
    pub(super) copied: [Copied; 2],
}

/// Descriptor of local unknowns or entries targeting high-fanout equations (more than
/// [`CHUNK`] contributions) whose values are copied into contiguous slab regions.
#[derive(Debug, Default)]
pub(super) struct Copied {
    /// Prefix offsets delimiting each member's range inside `locals`.
    pub(super) starts: Vec<usize>,
    /// Sorted local indices copied for each member in sequence.
    locals: Vec<usize>,
    /// Starting offset of this group's copy slice within the slab copy region.
    first: usize,
}

impl Copied {
    /// Identifies local slots across `members` instances (each of width `per`) whose entries
    /// in `targets` hit a high-fanout index marked in `long`.
    fn from_targets(targets: &[u32], (members, per): (usize, usize), long: &[bool]) -> Self {
        let mut copied = Self {
            starts: Vec::with_capacity(members + 1),
            locals: Vec::new(),
            first: 0,
        };
        copied.starts.push(0);
        for member in 0..members {
            let instance = targets.get(member * per..(member + 1) * per).unwrap_or(&[]);
            for (local, &target) in instance.iter().enumerate() {
                if long.get(target as usize).copied().unwrap_or(false) {
                    copied.locals.push(local);
                }
            }
            copied.starts.push(copied.locals.len());
        }
        copied
    }

    /// Returns the total number of copied local entries in the group.
    pub(super) fn len(&self) -> usize {
        self.locals.len()
    }

    /// Counts copied local slots across members `first..end`, or returns `None` if out of
    /// bounds.
    pub(super) fn count_between(&self, first: usize, end: usize) -> Option<usize> {
        if self.locals.is_empty() {
            return Some(0);
        }
        self.starts.get(end)?.checked_sub(*self.starts.get(first)?)
    }

    /// Locates the offset within the group's copy slice for `local` of `member`, if copied.
    fn place(&self, member: usize, local: usize) -> Option<usize> {
        let (start, end) = (*self.starts.get(member)?, *self.starts.get(member + 1)?);
        let locals = self.locals.get(start..end)?;
        let rank = locals.iter().position(|&each| each == local)?;
        Some(start + rank)
    }

    /// Copies values for copied locals of `member` from `block` planes into `copies` shifted
    /// by slice base offset `base`.
    pub(super) fn copy<const N: usize>(
        &self,
        (member, base): (usize, usize),
        block: [&[f64]; N],
        copies: &mut [&mut [f64]; N],
    ) {
        let (Some(&start), Some(&end)) = (self.starts.get(member), self.starts.get(member + 1))
        else {
            return;
        };
        let Some(locals) = self.locals.get(start..end) else {
            return;
        };
        if locals.is_empty() || start < base {
            return;
        }
        for (copy, plane) in copies.iter_mut().zip(block) {
            let Some(target) = copy.get_mut(start - base..end - base) else {
                continue;
            };
            for (into, &local) in target.iter_mut().zip(locals) {
                if let Some(&value) = plane.get(local) {
                    *into = value;
                }
            }
        }
    }
}

impl Group {
    /// Returns the number of local unknowns per instance.
    pub(super) fn unknowns(&self) -> usize {
        self.layout.unknowns()
    }

    /// Returns the entry stride per instance in the slab (at least 1 so every block is
    /// non-empty).
    pub(super) fn stride(&self) -> usize {
        self.layout.entries().len().max(1)
    }

    /// Returns the global unknown mapping slice for `member`.
    pub(super) fn map_of(&self, member: usize) -> Option<&[u32]> {
        let unknowns = self.unknowns();
        self.nodes.get(member * unknowns..(member + 1) * unknowns)
    }

    /// Returns the stride-aligned stored sparsity position slice for `member`.
    pub(super) fn slots_of(&self, member: usize) -> Option<&[u32]> {
        let stride = self.stride();
        self.slots.get(member * stride..(member + 1) * stride)
    }
}

/// Circuit-wide collection of device groups partitioned by kernel, shape, and linearity.
#[derive(Debug, Default)]
pub struct Groups {
    pub(super) kernels: Vec<Group>,
    /// Mapping from circuit device index to `(group_index, member_index)`.
    places: Vec<(usize, usize)>,
    /// Lazily built gather maps for unknown rows and matrix entries used in parallel runs.
    gathers: OnceLock<(GatherMap, GatherMap)>,
    /// Global target counts `(unknowns, stored_entries)` for gather maps.
    targets: (usize, usize),
    /// Block-region sizes `(unknowns_len, entries_len)` in the evaluation slab.
    pub(super) sizes: (usize, usize),
    /// High-fanout copy-region sizes `(unknown_copies, entry_copies)` beyond blocks.
    pub(super) copies: (usize, usize),
    /// Sorted global sparsity slots receiving linear contributions, paired with the base
    /// region trailing the entry copies.
    pub(super) base: Vec<u32>,
    /// Gather map summing linear instance entry blocks into the shared base region once per
    /// run.
    summed: GatherMap,
    /// Reusable scratch layout and terminal node buffer used while adding instances.
    scratch: (Layout, Vec<Node>),
    /// Global `(row, column)` pairs of every pivot claimed via [`Layout::pivot`].
    pivoted: Vec<(usize, usize)>,
}

/// Error returned when slab or circuit index counts exceed 32-bit addressable limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TooLarge;

impl Groups {
    /// Creates an empty group collection with no registered devices.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `instance` at ambient `temperature`, preparing it in `runs` and claiming its
    /// internal unknowns and matrix entries via `binding`.
    ///
    /// Creates a new group on the first occurrence of a kernel, shape, and linearity tuple.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when parameters are out of range, leaving `self` unchanged.
    pub fn add(
        &mut self,
        runs: &mut Runs,
        (instance, temperature): (Instance<'_>, f64),
        binding: &mut Binding<'_>,
    ) -> Result<(), ParameterError> {
        let terminals = binding.terminals().len();
        let (layout, nodes) = &mut self.scratch;
        layout.clear(terminals);
        let made = instance.make(temperature, layout)?;
        let linear = made.linear();
        let found = self.kernels.iter().position(|group| {
            group.shaped.kernel() == instance.kernel
                && group.shaped.same(made.shaped())
                && group.linear == linear
        });
        let index = found.unwrap_or_else(|| {
            runs.groups.push(made.shaped().run(0));
            self.kernels.push(Group {
                shaped: made.shaped().boxed(),
                layout: layout.clone(),
                linear,
                members: Vec::new(),
                nodes: Vec::new(),
                claims: Vec::new(),
                slots: Vec::new(),
                starts: (0, 0),
                copied: [Copied::default(), Copied::default()],
            });
            self.kernels.len() - 1
        });
        runs.groups[index].push(made.into_any());
        nodes.clear();
        nodes.extend_from_slice(binding.terminals());
        for (label, nature) in layout.internals() {
            nodes.push(Some(binding.internal(label, *nature)));
        }
        let node = |local: usize| nodes.get(local).copied().flatten();
        self.pivoted.extend(
            layout
                .pivots()
                .iter()
                .filter_map(|&(row, column)| Some((node(row)?.index(), node(column)?.index()))),
        );
        let group = &mut self.kernels[index];
        let device = self.places.len();
        self.places.push((index, group.members.len()));
        group.members.push(device);
        group.nodes.extend(nodes.iter().map(|&node| mapped(node)));
        let claimed = group.claims.len();
        group.claims.extend(
            layout
                .entries()
                .iter()
                .map(|&(row, column)| binding.entry(node(row), node(column))),
        );
        if group.claims.len() == claimed {
            group.claims.push(None);
        }
        Ok(())
    }

    /// Checks whether `instance` at ambient `temperature` preserves the kernel, shape, and
    /// linearity of `device`'s existing group.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when the instance parameters are out of range.
    pub fn keeps_shape(
        &self,
        device: usize,
        (instance, temperature): (&Instance<'_>, f64),
    ) -> Result<bool, ParameterError> {
        let Some((group, _)) = self.group_of(device) else {
            return Ok(false);
        };
        let mut layout = Layout::new(group.layout.terminals());
        let made = instance.make(temperature, &mut layout)?;
        Ok(group.shaped.kernel() == instance.kernel
            && group.shaped.same(made.shaped())
            && group.linear == made.linear())
    }

    /// Returns whether `device` belongs to a linear kernel group.
    #[must_use]
    pub fn is_linear(&self, device: usize) -> bool {
        self.group_of(device).is_some_and(|(of, _)| of.linear)
    }

    /// Returns `(group_index, member_index)` for `device`, if registered.
    pub(super) fn place(&self, device: usize) -> Option<(usize, usize)> {
        self.places.get(device).copied()
    }

    /// Returns the group reference and member index for `device`, if registered.
    pub(super) fn group_of(&self, device: usize) -> Option<(&Group, usize)> {
        let (group, member) = self.place(device)?;
        Some((self.kernels.get(group)?, member))
    }

    /// Finalizes slab offsets, sparsity slot mappings, copy ranges, and linear base gathers
    /// once all devices have been added and `pattern` is built.
    ///
    /// # Errors
    ///
    /// Returns [`TooLarge`] when slab slots, unknowns, or stored entries exceed 32-bit
    /// indexing limits.
    pub fn finish(&mut self, runs: &mut Runs, pattern: &Pattern) -> Result<(), TooLarge> {
        let fits = |count: usize| u32::try_from(count).is_ok_and(|count| count < UNMAPPED);
        let (mut rows, mut stored) = (0, 0);
        for group in &mut self.kernels {
            group.starts = (rows, stored);
            rows += group.members.len() * group.unknowns();
            stored += group.members.len() * group.stride();
            group.slots = std::mem::take(&mut group.claims)
                .into_iter()
                .map(|claim| claim.map_or(UNMAPPED, |entry| mapped_slot(pattern.slot(entry))))
                .collect();
        }
        // The stored entries a linear instance's entries add to, then the index of each's
        // value of the base, by the entry, or none.
        let mut added = vec![false; pattern.len()];
        for group in self.kernels.iter().filter(|group| group.linear) {
            for &slot in &group.slots {
                if let Some(added) = added.get_mut(slot as usize) {
                    *added = true;
                }
            }
        }
        self.base.clear();
        let mut valued = vec![UNMAPPED; pattern.len()];
        for ((slot, value), &added) in (0_u32..).zip(&mut valued).zip(&added) {
            if added {
                *value = mapped_slot(self.base.len());
                self.base.push(slot);
            }
        }
        // The values of more than a chunk of contributions, whose contributions are copied:
        // a stored entry's are the base's value, then the entries of instances that are not
        // linear.
        let long = |targets: usize, maps: fn(&Group) -> &[u32], base: &[u32]| {
            let mut counts = vec![0_usize; targets];
            let gathered = self.kernels.iter().filter(|group| !group.linear);
            for &target in base.iter().chain(gathered.flat_map(maps)) {
                if let Some(count) = counts.get_mut(target as usize) {
                    *count += 1;
                }
            }
            counts
                .into_iter()
                .map(|count| count > CHUNK)
                .collect::<Vec<_>>()
        };
        let long = (
            long(pattern.dimension(), |group| &group.nodes, &[]),
            long(pattern.len(), |group| &group.slots, &self.base),
        );
        let mut copies = (0, 0);
        for group in &mut self.kernels {
            let members = group.members.len();
            // A linear instance's entries are gathered once a run, into the base, from their
            // blocks.
            let entries_long: &[bool] = if group.linear { &[] } else { &long.1 };
            let mut rows = Copied::from_targets(&group.nodes, (members, group.unknowns()), &long.0);
            let mut entries =
                Copied::from_targets(&group.slots, (members, group.stride()), entries_long);
            (rows.first, entries.first) = copies;
            copies = (copies.0 + rows.len(), copies.1 + entries.len());
            group.copied = [rows, entries];
        }
        let fit = fits(rows + copies.0) && fits(stored + copies.1 + self.base.len());
        if !(fit && fits(pattern.dimension()) && fits(pattern.len())) {
            return Err(TooLarge);
        }
        let valued = &valued;
        let summed = self.kernels.iter().filter(|group| group.linear);
        let summed = summed.flat_map(|group| {
            let start = group.starts.1;
            group.slots.iter().enumerate().map(move |(offset, &slot)| {
                let value = valued.get(slot as usize).copied().unwrap_or(UNMAPPED);
                (value, mapped_slot(start + offset))
            })
        });
        self.summed = GatherMap::new(self.base.len(), summed);
        self.sizes = (rows, stored);
        self.copies = copies;
        self.targets = (pattern.dimension(), pattern.len());
        self.scratch = (Layout::default(), Vec::new());
        runs.slab = Slab::sized(self.slab_sizes());
        runs.stamped = false;
        Ok(())
    }

    /// Returns total slab sizes `(unknowns_len, entries_len)` covering blocks, copies, and
    /// the linear base region.
    fn slab_sizes(&self) -> (usize, usize) {
        let entries = self.sizes.1 + self.copies.1;
        (self.sizes.0 + self.copies.0, entries + self.base.len())
    }

    /// Returns the starting slab offset of the linear base region and the gather map that
    /// sums linear instance entries into it once per run.
    #[must_use]
    pub fn base(&self) -> (usize, &GatherMap) {
        (self.sizes.1 + self.copies.1, &self.summed)
    }

    /// Returns the gather maps for rows (residuals, scales, charges) and entries
    /// (conductances, capacitances) from a run's [slab](Runs::slab), reading high-fanout
    /// targets from their copy slices.
    #[must_use]
    pub fn gathers(&self) -> (&GatherMap, &GatherMap) {
        let (rows, slots) = self.gathers.get_or_init(|| {
            let (unknowns, stored) = self.targets;
            (
                GatherMap::new(unknowns, self.sources(0)),
                GatherMap::new(stored, self.sources(1)),
            )
        });
        (rows, slots)
    }

    /// Yields `(target, slab_source)` pairs in deterministic summation order for `kind` (`0`
    /// for unknown rows; `1` for base entries followed by nonlinear matrix entries).
    fn sources(&self, kind: usize) -> impl Iterator<Item = (u32, u32)> + Clone + '_ {
        let beyond = if kind == 0 {
            self.sizes.0
        } else {
            self.sizes.1
        };
        let (first, _) = self.base();
        let base = (kind == 1).then_some(&self.base).into_iter().flatten();
        let base = (first..)
            .zip(base)
            .map(|(place, &target)| (target, mapped_slot(place)));
        let gathered = self
            .kernels
            .iter()
            .filter(move |group| kind == 0 || !group.linear);
        base.chain(gathered.flat_map(move |group| {
            let (start, maps, per) = if kind == 0 {
                (group.starts.0, &group.nodes, group.unknowns())
            } else {
                (group.starts.1, &group.slots, group.stride())
            };
            let copied = &group.copied[kind];
            maps.iter().enumerate().map(move |(offset, &target)| {
                let source = match copied.place(offset / per, offset % per) {
                    Some(copy) => beyond + copied.first + copy,
                    None => start + offset,
                };
                (target, u32::try_from(source).unwrap_or(UNMAPPED))
            })
        }))
    }

    /// Returns global `(row, column)` pairs of all pivots [claimed](Layout::pivot) for early
    /// elimination by the sparse solver.
    #[must_use]
    pub fn pivoted(&self) -> &[(usize, usize)] {
        &self.pivoted
    }

    /// Returns the number of device groups.
    #[must_use]
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Returns whether the collection contains no kernel groups.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }

    /// Counts nonlinear instances evaluated on every Newton iteration (excluding linear
    /// instances evaluated once per run).
    #[must_use]
    pub fn evaluated(&self) -> usize {
        self.kernels
            .iter()
            .filter(|group| !group.linear)
            .map(|group| group.members.len())
            .sum()
    }

    /// Iterates over the device-index slice of each group in registration order.
    pub fn members(&self) -> impl Iterator<Item = &[usize]> {
        self.kernels.iter().map(|group| group.members.as_slice())
    }

    /// Allocates a fresh [`Runs`] container with a zeroed slab and unprepared instances ready
    /// for [`Runs::prepare`].
    #[must_use]
    pub fn runs(&self) -> Runs {
        Runs {
            groups: self
                .kernels
                .iter()
                .map(|group| group.shaped.run(group.members.len()))
                .collect(),
            slab: Slab::sized(self.slab_sizes()),
            layout: Layout::default(),
            limited: vec![false; self.kernels.len()],
            stamped: false,
        }
    }
}

/// Converts `node` to its 32-bit global unknown index, or [`UNMAPPED`] for ground.
fn mapped(node: Node) -> u32 {
    node.and_then(|unknown| u32::try_from(unknown.index()).ok())
        .unwrap_or(UNMAPPED)
}

/// Converts stored position `slot` to `u32`, saturating to [`UNMAPPED`] if out of range (a
/// condition rejected by [`Groups::finish`]).
fn mapped_slot(slot: usize) -> u32 {
    u32::try_from(slot).unwrap_or(UNMAPPED)
}
