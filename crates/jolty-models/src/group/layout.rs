//! Structural group layout: mapping device instances onto slab blocks, long-value copy
//! ranges, and gather maps.

use std::sync::OnceLock;

use jolty_numerics::reduce::CHUNK;
use jolty_numerics::{Entry, GatherMap, Pattern, UNMAPPED};

use super::kind::{Instance, Shaped};
use super::runs::{Runs, Slab};
use crate::{Binding, Layout, Node, ParameterError};

/// The instances of one kernel and one shape, linear or not, as a circuit's structure fixes
/// them.
#[derive(Debug)]
pub(super) struct Group {
    shaped: Box<dyn Shaped>,
    pub(super) layout: Layout,
    /// Whether its instances are linear, as [`Kernel::linear`](crate::Kernel::linear) says:
    /// evaluated once a run, for their entries, which the base sums, and their residuals
    /// and charges each evaluation from them.
    pub(super) linear: bool,
    /// Each instance's device, by its index among the circuit's devices, in their order.
    pub(super) members: Vec<usize>,
    /// The circuit's unknown of each instance's local unknowns, instance after instance,
    /// or [`UNMAPPED`] for one on ground.
    pub(super) nodes: Vec<u32>,
    /// The circuit's entry of each instance's local entries, instance after instance, or
    /// none for one that joins ground, until the pattern is known.
    claims: Vec<Option<Entry>>,
    /// The stored position of each instance's local entries, instance after instance, a
    /// stride apart, or [`UNMAPPED`] for none.
    pub(super) slots: Vec<u32>,
    /// Where the group's blocks begin among the slab's local unknowns and its local
    /// entries.
    pub(super) starts: (usize, usize),
    /// Its instances' local unknowns, then their local entries, that contribute to a long
    /// value, whose copies lie beyond the blocks.
    pub(super) copied: [Copied; 2],
}

/// The locals of a group's instances, its local unknowns or its local entries, that
/// contribute to a long value of the circuit's equations, one of more than [`CHUNK`]
/// contributions, and where their copies lie among the slab's.
#[derive(Debug, Default)]
pub(super) struct Copied {
    /// Where each instance's begin among the group's copies, then where the last's end.
    pub(super) starts: Vec<usize>,
    /// The local index of each, instance after instance, in increasing order.
    locals: Vec<usize>,
    /// Where the group's copies begin among the slab's.
    first: usize,
}

impl Copied {
    /// The locals of each of `members` instances of `per` locals, whose targets are
    /// `targets`, instance after instance, that contribute to a target `long` marks.
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

    /// The number of copies.
    pub(super) fn len(&self) -> usize {
        self.locals.len()
    }

    /// How many locals of the instances in `first..end` are copied, or `None` if out of
    /// range.
    pub(super) fn count_between(&self, first: usize, end: usize) -> Option<usize> {
        if self.locals.is_empty() {
            return Some(0);
        }
        self.starts.get(end)?.checked_sub(*self.starts.get(first)?)
    }

    /// Where among the group's copies the copy of the local `local` of the instance
    /// `member` lies, if it contributes to a long value.
    fn place(&self, member: usize, local: usize) -> Option<usize> {
        let (start, end) = (*self.starts.get(member)?, *self.starts.get(member + 1)?);
        let locals = self.locals.get(start..end)?;
        let rank = locals.iter().position(|&each| each == local)?;
        Some(start + rank)
    }

    /// Copies the values `block` holds, in its planes, of the locals it lists of the
    /// instance `member` to their places among `copies`, the copies from the group's
    /// `base`th on.
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
    /// The number of local unknowns of each instance.
    pub(super) fn unknowns(&self) -> usize {
        self.layout.unknowns()
    }

    /// The number of the slab's places for each instance's local entries: one for each,
    /// and one for an instance with none, so that every block has a place.
    pub(super) fn stride(&self) -> usize {
        self.layout.entries().len().max(1)
    }

    /// The circuit's unknown of each local unknown of the instance `member`.
    pub(super) fn map_of(&self, member: usize) -> Option<&[u32]> {
        let unknowns = self.unknowns();
        self.nodes.get(member * unknowns..(member + 1) * unknowns)
    }

    /// The stored position of each local entry of the instance `member`, a stride of them.
    pub(super) fn slots_of(&self, member: usize) -> Option<&[u32]> {
        let stride = self.stride();
        self.slots.get(member * stride..(member + 1) * stride)
    }
}

/// The groups of a circuit's devices, each a kernel's instance: by kernel and shape, in the
/// order their first instances appear.
#[derive(Debug, Default)]
pub struct Groups {
    pub(super) kernels: Vec<Group>,
    /// Each device's group and its index among the group's instances, by the device's
    /// index.
    places: Vec<(usize, usize)>,
    /// The sources of each unknown's residual, scale and charge, and of each stored
    /// entry's conductance and capacitance, among the slab's, made the first time a run
    /// gathers its slab, as a run evaluated serially never does: a function of the rest,
    /// which they never change.
    gathers: OnceLock<(GatherMap, GatherMap)>,
    /// The circuit's unknowns and stored entries, the targets of the gathers.
    targets: (usize, usize),
    /// The slab's local unknowns and local entries.
    pub(super) sizes: (usize, usize),
    /// The copies beyond them of those that contribute to a long value.
    pub(super) copies: (usize, usize),
    /// The stored entries the linear instances' entries add to, in increasing order, each
    /// the target of one value of the base, which lies beyond the copies among the slab's
    /// places for local entries, in their order.
    pub(super) base: Vec<u32>,
    /// The sources of each value of the base among the linear instances' local entries, in
    /// the order they add, by its index among the base's values.
    summed: GatherMap,
    /// The layout each instance added claims, and its local unknowns' nodes, kept to be
    /// used again.
    scratch: (Layout, Vec<Node>),
    /// The row and the column, circuit's unknowns, of every pivot an instance's layout
    /// [claims](Layout::pivot), in the order the instances were added.
    pivoted: Vec<(usize, usize)>,
}

/// Why a circuit's groups cannot be made: their slab, or the circuit, has more places than
/// a 32-bit index reaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TooLarge;

impl Groups {
    /// No groups yet, and no device.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds the next device among the circuit's, the instance `instance` of a kernel at the
    /// temperature `temperature` of its surroundings, to the group of its kernel and shape,
    /// made the first time a kernel and shape appear: its numbers are prepared in `runs`,
    /// and its internal unknowns and its entries claimed through `binding`, which binds
    /// its terminals.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when the instance's parameters are out of range; nothing
    /// is added.
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

    /// Whether `instance`, the instance of the device `device` its parameters make at the
    /// temperature `temperature` of its surroundings, has the kernel and the shape of the
    /// device's group, and so claims the unknowns and the entries the circuit holds, and is
    /// linear as the group's instances are or are not.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when the instance's parameters are out of range.
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

    /// Whether the instance the device `device` is, if it is a kernel's, is linear.
    #[must_use]
    pub fn is_linear(&self, device: usize) -> bool {
        self.group_of(device).is_some_and(|(of, _)| of.linear)
    }

    /// The instance at the device `device`, if it is a kernel's: its group and its index
    /// among the group's instances.
    pub(super) fn place(&self, device: usize) -> Option<(usize, usize)> {
        self.places.get(device).copied()
    }

    /// The group and member index of `device`, if it is a kernel's instance.
    pub(super) fn group_of(&self, device: usize) -> Option<(&Group, usize)> {
        let (group, member) = self.place(device)?;
        Some((self.kernels.get(group)?, member))
    }

    /// Fixes the groups once every device is added and the circuit's `pattern` is known:
    /// where each instance's block lies in the slab, which `runs` makes, the stored
    /// position of each of its local entries, and the maps that gather the blocks.
    ///
    /// # Errors
    ///
    /// Returns [`TooLarge`] when the slab, the unknowns or the stored entries number more
    /// than a 32-bit index reaches.
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

    /// The slab's places for local unknowns and for local entries: the blocks', then the
    /// copies', then, for local entries, the base's values.
    fn slab_sizes(&self) -> (usize, usize) {
        let entries = self.sizes.1 + self.copies.1;
        (self.sizes.0 + self.copies.0, entries + self.base.len())
    }

    /// Where the base's values begin among the slab's places for local entries, and the
    /// sources of each among the linear instances' local entries, which sum into it once a
    /// run, in the order they add.
    #[must_use]
    pub fn base(&self) -> (usize, &GatherMap) {
        (self.sizes.1 + self.copies.1, &self.summed)
    }

    /// The maps that gather, from a run's [slab](Runs::slab), the contributions to each
    /// unknown's residual, scale and charge, and to each stored entry's conductance and
    /// capacitance, in the order they add: a long value's from their copies.
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

    /// Each contribution to the circuit's equations, as its target and its place in the
    /// slab, in the order they add: of `kind`, the local unknowns' contributions to the
    /// unknowns' values for 0, and for 1 the base's values, then the local entries of the
    /// instances that are not linear, to the stored entries' values.
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

    /// The row and the column, circuit's unknowns, of every pivot an instance's layout
    /// [claims](Layout::pivot), by which the sparse solver may eliminate the column's
    /// unknown first.
    #[must_use]
    pub fn pivoted(&self) -> &[(usize, usize)] {
        &self.pivoted
    }

    /// The number of groups of kernels' instances.
    #[must_use]
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Whether no device is a kernel's instance.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }

    /// The instances each evaluation evaluates: those of the groups that are not linear,
    /// as a linear group's are evaluated once a run, for their entries, and each
    /// evaluation takes their residuals and charges from them.
    #[must_use]
    pub fn evaluated(&self) -> usize {
        self.kernels
            .iter()
            .filter(|group| !group.linear)
            .map(|group| group.members.len())
            .sum()
    }

    /// The devices of each group's instances, by index, group after group.
    pub fn members(&self) -> impl Iterator<Item = &[usize]> {
        self.kernels.iter().map(|group| group.members.as_slice())
    }

    /// A run of the groups, its slab zeroed and no instance prepared yet, for
    /// [`Runs::prepare`] to prepare every one.
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

/// The index among the circuit's unknowns of `node`, or [`UNMAPPED`] for ground.
fn mapped(node: Node) -> u32 {
    node.and_then(|unknown| u32::try_from(unknown.index()).ok())
        .unwrap_or(UNMAPPED)
}

/// `slot`, a stored position, as a map holds it, or [`UNMAPPED`] beyond its reach, which
/// [`Groups::finish`] rules out.
fn mapped_slot(slot: usize) -> u32 {
    u32::try_from(slot).unwrap_or(UNMAPPED)
}
