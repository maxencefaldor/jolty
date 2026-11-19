//! Per-run group state: slabs, evaluation scheduling (serial and rayon-parallel),
//! single-instance evaluation, and staged parameter preparation.

use std::num::NonZeroUsize;

use jolty_numerics::{Fold, UNMAPPED};

use super::kernel_run::GroupRun;
use super::kind::Instance;
use super::layout::{Group, Groups};
use super::linear::{Blocks, add_entries, as_refs, beyond, blocks_of, split};
use super::twin::{Seeds, Twin};
use crate::{
    Contributions, DelayedEntry, Input, Layout, LogicLevel, NoiseSource, ParameterError,
    UnknownIndex,
};

/// What has changed of an instance since its run last prepared it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Changed {
    /// Nothing: its numbers stand.
    Nothing,
    /// Its temperature, or its parameters back to those its group was made of: its numbers
    /// are prepared again, and its shape is its group's.
    Numbers,
    /// Its parameters, to others than its group was made of: its numbers are prepared
    /// again, and its shape is checked against its group's.
    Parameters,
}

/// How a run splits a kind of its work among the threads of the current rayon pool: into
/// tasks of at least `least` items each, such as a group's instances, and at most one a
/// thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tasks {
    /// The fewest items a task takes.
    pub least: NonZeroUsize,
    /// The threads the tasks share.
    pub threads: NonZeroUsize,
}

impl Tasks {
    /// How many tasks `items` items split into: one a thread, each of at least
    /// [`Tasks::least`] items, and one when they fill fewer than two.
    #[must_use]
    pub fn count(&self, items: usize) -> usize {
        self.threads.get().min(items / self.least).max(1)
    }

    /// How many of `items` items each of their [`Tasks::count`] tasks takes, the last the
    /// rest.
    #[must_use]
    pub fn size(&self, items: usize) -> usize {
        items.div_ceil(self.count(items)).max(1)
    }
}

/// The slab: each instance's block of residuals, scales and charges of its local unknowns,
/// and of conductances and capacitances of its local entries, group after group.
#[derive(Debug, Clone, Default)]
pub(super) struct Slab {
    rows: [Vec<f64>; 3],
    entries: [Vec<f64>; 2],
}

impl Slab {
    /// A zeroed slab of `rows` local unknowns and `entries` local entries.
    pub(super) fn sized((rows, entries): (usize, usize)) -> Self {
        Self {
            rows: std::array::from_fn(|_| vec![0.0; rows]),
            entries: std::array::from_fn(|_| vec![0.0; entries]),
        }
    }

    /// Shared views of all five slab planes.
    fn slices(&self) -> ([&[f64]; 3], [&[f64]; 2]) {
        let [residual, scale, charge] = &self.rows;
        let [conductance, capacitance] = &self.entries;
        ([residual, scale, charge], [conductance, capacitance])
    }

    /// Mutable views of all five slab planes.
    fn slices_mut(&mut self) -> Blocks<'_> {
        let [residual, scale, charge] = &mut self.rows;
        let [conductance, capacitance] = &mut self.entries;
        let rows = [residual.as_mut_slice(), scale, charge];
        let entries = [conductance.as_mut_slice(), capacitance];
        (rows, entries)
    }
}

/// What one run of a circuit's groups writes: each instance's prepared numbers and memory,
/// and the slab its evaluations write.
#[derive(Debug, Default)]
pub struct Runs {
    pub(super) groups: Vec<Box<dyn GroupRun>>,
    pub(super) slab: Slab,
    /// The layout each instance prepared again claims, to check its shape.
    pub(super) layout: Layout,
    /// Whether an instance of each group limited its voltages, as the group's last
    /// evaluation in parallel found.
    pub(super) limited: Vec<bool>,
    /// Whether the linear instances' blocks hold their entries, at their numbers now, and
    /// the slab the base they sum to: false until the first evaluation, and again once an
    /// instance is prepared afresh or the slab forgotten.
    pub(super) stamped: bool,
}

impl Clone for Runs {
    fn clone(&self) -> Self {
        Self {
            groups: self.groups.iter().map(|run| run.cloned()).collect(),
            slab: self.slab.clone(),
            layout: self.layout.clone(),
            limited: self.limited.clone(),
            stamped: self.stamped,
        }
    }
}

/// One instance's block of a run's slab, as its last evaluation wrote it: the residual,
/// scale and charge of each of its local unknowns, its terminals then its internal unknowns,
/// and the conductance and capacitance of each of its local entries, each the pair of local
/// unknowns `entries` gives, and the circuit's unknown of each local unknown, or
/// [`UNMAPPED`] for one on ground.
#[derive(Debug, Clone, Copy)]
pub struct Block<'a> {
    /// The residual of each local unknown's equation.
    pub residual: &'a [f64],
    /// The largest magnitude of a term of each residual.
    pub scale: &'a [f64],
    /// The charge of each local unknown's equation.
    pub charge: &'a [f64],
    /// The conductance at each local entry.
    pub conductance: &'a [f64],
    /// The capacitance at each local entry.
    pub capacitance: &'a [f64],
    /// The row and the column of each local entry, local unknowns both.
    pub entries: &'a [(usize, usize)],
    /// The circuit's unknown of each local unknown.
    pub nodes: &'a [u32],
}

/// One instance's block of contributions, evaluated alone, and where its values go among
/// the circuit's: the residual, scale and charge of each local unknown, the conductance and
/// capacitance of each local entry, the circuit's unknown of each local unknown, or
/// [`UNMAPPED`] for one on ground, and the stored position of each local entry, or
/// [`UNMAPPED`] for none.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Alone {
    /// The residual, the scale and the charge of each local unknown.
    pub rows: [Vec<f64>; 3],
    /// The conductance and the capacitance of each local entry.
    pub entries: [Vec<f64>; 2],
    /// The circuit's unknown of each local unknown.
    pub nodes: Vec<u32>,
    /// The stored position of each local entry.
    pub slots: Vec<u32>,
    /// Whether the instance limited its voltages.
    pub limited: bool,
}

impl Runs {
    /// The group, its run, and the member index of `device`, if it is a kernel's instance.
    fn entry<'a>(
        &'a self,
        groups: &'a Groups,
        device: usize,
    ) -> Option<(&'a Group, &'a dyn GroupRun, usize)> {
        let (group, member) = groups.place(device)?;
        let of = groups.kernels.get(group)?;
        let run = self.groups.get(group)?.as_ref();
        Some((of, run, member))
    }

    /// Appends the noise of the instance the device `device` is, if it is a kernel's, at
    /// `input`, the operating point a noise analysis linearizes about, to `sources`, each
    /// source's equations the circuit's.
    pub fn noise(
        &self,
        groups: &Groups,
        device: usize,
        input: &Input<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        if let Some((of, run, member)) = self.entry(groups, device) {
            run.noise(member, of, input, sources);
        }
    }

    /// Writes the variables of the instance the device `device` is, if it is a kernel's,
    /// as its last evaluation left them, to `values`, one per name of
    /// [`Instance::variable_names`].
    pub fn variables(&self, groups: &Groups, device: usize, values: &mut [f64]) {
        if let Some((_, run, member)) = self.entry(groups, device) {
            run.variables(member, values);
        }
    }

    /// The twin of the instance the device `device` is, the `instance` of its parameters
    /// as the run now makes it, at the temperature `temperature` of its surroundings,
    /// differentiated along `seeds`, from its initial memory; `None` when its kernel reads
    /// no real number at a seed's path, as when it reads it as a choice, such as a level,
    /// or a parameter is out of range.
    #[must_use]
    pub fn twin(
        &self,
        groups: &Groups,
        device: usize,
        (instance, temperature): (&Instance<'_>, f64),
        seeds: &Seeds,
    ) -> Option<Twin> {
        let (of, run, member) = self.entry(groups, device)?;
        let sizes = (of.unknowns(), of.stride());
        let nodes = of.map_of(member)?.to_vec();
        let evaluated = run.twin((instance, temperature), seeds, sizes)?;
        let slots = of.slots_of(member)?.to_vec();
        Some(Twin::new(evaluated, nodes, slots, seeds))
    }

    /// Twins of the instance the device `device` is, at `temperature`, whose seeds move
    /// each of the circuit's unknowns its local unknowns map to, once, up to
    /// [`Seeds::MAX`] unknowns to a twin, its parameters constant.
    #[must_use]
    pub fn unknown_twins(
        &self,
        groups: &Groups,
        device: usize,
        instance: (&Instance<'_>, f64),
    ) -> Vec<Twin> {
        let Some(nodes) = groups
            .group_of(device)
            .and_then(|(of, member)| of.map_of(member))
        else {
            return Vec::new();
        };
        let mut unique = Vec::with_capacity(nodes.len());
        for &unknown in nodes {
            if unknown != UNMAPPED && !unique.contains(&unknown) {
                unique.push(unknown);
            }
        }
        unique
            .chunks(Seeds::MAX)
            .filter_map(|chunk| {
                let seeds = Seeds::unknowns(chunk.to_vec())?;
                self.twin(groups, device, instance, &seeds)
            })
            .collect()
    }

    /// The DC value of the instance the device `device` is, if it is an independent source,
    /// which analyses excite: its waveform's value at time zero.
    #[must_use]
    pub fn dc_value(&self, groups: &Groups, device: usize) -> Option<f64> {
        let (_, run, member) = self.entry(groups, device)?;
        run.dc_value(member)
    }

    /// Adds the derivative of each of the circuit's residuals with respect to the value of
    /// the instance the device `device` is, if it is an independent source, to
    /// `derivative`, one value per unknown: how a small-signal analysis excites it.
    pub fn excite(&self, groups: &Groups, device: usize, derivative: &mut [f64]) {
        let (Some(local), Some(block)) =
            (self.excitation(groups, device), self.block(groups, device))
        else {
            return;
        };
        for (&unknown, &value) in block.nodes.iter().zip(&local) {
            if let Some(sum) = derivative.get_mut(unknown as usize) {
                *sum += value;
            }
        }
    }

    /// The derivative of each residual of the instance the device `device` is with respect
    /// to its value, as an independent source's, one value per local unknown: zero for
    /// each, for an instance that is no independent source.
    #[must_use]
    pub fn excitation(&self, groups: &Groups, device: usize) -> Option<Vec<f64>> {
        let (of, run, _) = self.entry(groups, device)?;
        Some(run.excitation(of))
    }

    /// The block of the instance the device `device` is in the slab, as the last evaluation
    /// wrote it.
    #[must_use]
    pub fn block<'a>(&'a self, groups: &'a Groups, device: usize) -> Option<Block<'a>> {
        let (of, member) = groups.group_of(device)?;
        let (unknowns, stride) = (of.unknowns(), of.stride());
        let rows = of.starts.0 + member * unknowns..of.starts.0 + (member + 1) * unknowns;
        let start = of.starts.1 + member * stride;
        let stored = start..start + of.layout.entries().len();
        let ([residual, scale, charge], [conductance, capacitance]) = self.slab.slices();
        Some(Block {
            residual: residual.get(rows.clone())?,
            scale: scale.get(rows.clone())?,
            charge: charge.get(rows)?,
            conductance: conductance.get(stored.clone())?,
            capacitance: capacitance.get(stored)?,
            entries: of.layout.entries(),
            nodes: of.map_of(member)?,
        })
    }

    /// Starts the instance the device `device` is, if it is a logic driver, moving to the
    /// level `level` at `time`, at the rate that swings ground to supply in `transition`
    /// seconds, and returns when it settles; `None` for an instance that drives no logic.
    pub fn drive(
        &mut self,
        groups: &Groups,
        device: usize,
        ramp: (f64, LogicLevel, f64),
    ) -> Option<f64> {
        let (group, member) = groups.place(device)?;
        self.groups.get_mut(group)?.drive(member, ramp)
    }

    /// The unknown of the current through the instance the device `device` is, if it is a
    /// logic driver, whose equation holds its voltage.
    #[must_use]
    pub fn driven(&self, groups: &Groups, device: usize) -> Option<UnknownIndex> {
        let (of, run, member) = self.entry(groups, device)?;
        run.driven(member, of)
    }

    /// How far the input of the instance the device `device` is, if it is a logic input's
    /// detector, lies above its threshold at `input`, and the voltage of its supply above
    /// its ground there, in volts.
    #[must_use]
    pub fn sense(&self, groups: &Groups, device: usize, input: &Input<'_>) -> Option<(f64, f64)> {
        let (of, run, member) = self.entry(groups, device)?;
        run.sense(member, of, input)
    }

    /// Appends the times in `[start, stop]` where an instance is not smooth to `times`, so
    /// a transient analysis steps onto them exactly.
    pub fn breakpoints(&self, span: (f64, f64), times: &mut Vec<f64>) {
        for run in &self.groups {
            run.breakpoints(span, times);
        }
    }

    /// The device indices of the instances of `groups` whose equations explicitly depend
    /// on time, as [`Kernel::varies_in_time`](crate::Kernel::varies_in_time) states it, in
    /// ascending order.
    #[must_use]
    pub fn time_varying(&self, groups: &Groups) -> Vec<usize> {
        let mut found = Vec::new();
        for (group, run) in groups.kernels.iter().zip(&self.groups) {
            run.time_varying(group, &mut found);
        }
        found.sort_unstable();
        found.dedup();
        found
    }

    /// Each part of the circuit's conductance that an instance of `groups` reads in the
    /// past, as [`Kernel::delayed`](crate::Kernel::delayed) states it, placed among the
    /// circuit's equations, in the order of the groups and their instances: none in a
    /// circuit without a delay line. A part whose row or column is ground is not
    /// among them.
    #[must_use]
    pub fn delayed(&self, groups: &Groups) -> Vec<DelayedEntry> {
        let mut found = Vec::new();
        for (group, run) in groups.kernels.iter().zip(&self.groups) {
            run.delayed(group, &mut found);
        }
        found
    }

    /// The earliest fraction of the way from `before` to `after`, the states at the two
    /// ends of a step, at which an instance's contribution jumps, as a choice its behavior
    /// turns on changes, located by linear interpolation; `None` when none changes.
    pub fn first_jump(
        &mut self,
        groups: &Groups,
        (before, after): (&Input<'_>, &Input<'_>),
    ) -> Option<f64> {
        let mut found = [Vec::new(), Vec::new()];
        let mut first: Option<f64> = None;
        for (group, run) in groups.kernels.iter().zip(&mut self.groups) {
            if let Some(fraction) = run.first_jump(group, (before, after), &mut found) {
                first = Some(first.map_or(fraction, |earliest| earliest.min(fraction)));
            }
        }
        first
    }

    /// Evaluates the instance the device `device` is, if it is a kernel's, alone at
    /// `input`, into a block of its own, from what it remembers, which it updates, as its
    /// group evaluates it: a linear instance's rows from its entries.
    pub fn evaluate_alone(
        &mut self,
        groups: &Groups,
        device: usize,
        input: &Input<'_>,
    ) -> Option<Alone> {
        self.alone(groups, device, input, true)
    }

    /// Evaluates the instance the device `device` is as [`Runs::evaluate_alone`] does, but
    /// by its kernel at `input` even where it is linear.
    #[cfg(test)]
    pub(crate) fn evaluate_by_kernel(
        &mut self,
        groups: &Groups,
        device: usize,
        input: &Input<'_>,
    ) -> Option<Alone> {
        self.alone(groups, device, input, false)
    }

    /// [`Runs::evaluate_alone`], a linear instance's rows from its entries when `linear`,
    /// and else by its kernel.
    fn alone(
        &mut self,
        groups: &Groups,
        device: usize,
        input: &Input<'_>,
        linear: bool,
    ) -> Option<Alone> {
        if linear && !self.stamped {
            self.stamp(groups);
        }
        let (group, member) = groups.place(device)?;
        let (of, run) = (groups.kernels.get(group)?, self.groups.get_mut(group)?);
        let (unknowns, stride) = (of.unknowns(), of.stride());
        let mut alone = Alone {
            rows: std::array::from_fn(|_| vec![0.0; unknowns]),
            entries: std::array::from_fn(|_| vec![0.0; stride]),
            nodes: of.map_of(member)?.to_vec(),
            slots: of.slots_of(member)?.to_vec(),
            limited: false,
        };
        let [residual, scale, charge] = &mut alone.rows;
        let [conductance, capacitance] = &mut alone.entries;
        alone.limited = run.evaluate_alone(
            member,
            (of, linear),
            input,
            ([residual, scale, charge], [conductance, capacitance]),
        );
        Some(alone)
    }

    /// Prepares again each instance of `groups` whose device `changed` says has changed,
    /// from the instance `instance` gives of it, at the temperature `temperature` of its
    /// surroundings, keeping the numbers apart until [`Runs::commit`] takes them. Returns
    /// whether each has its group's kernel and shape; when one does not, nothing is kept.
    ///
    /// # Errors
    ///
    /// Returns each device whose parameters are out of range, with why; nothing is kept.
    pub fn prepare<'a>(
        &mut self,
        groups: &Groups,
        instance: impl Fn(usize) -> Instance<'a>,
        temperature: f64,
        changed: impl Fn(usize) -> Changed,
    ) -> Result<bool, Vec<(usize, ParameterError)>> {
        let mut problems = Vec::new();
        for (group, run) in groups.kernels.iter().zip(&mut self.groups) {
            for (member, &device) in group.members.iter().enumerate() {
                let checked = match changed(device) {
                    Changed::Nothing => continue,
                    Changed::Numbers => None,
                    Changed::Parameters => Some(&mut self.layout),
                };
                let instance = instance(device);
                let prepared = run.prepare(member, (group, &instance), temperature, checked);
                match prepared {
                    Ok(true) => {}
                    Ok(false) => {
                        self.discard();
                        return Ok(false);
                    }
                    Err(error) => problems.push((device, error)),
                }
            }
        }
        if problems.is_empty() {
            Ok(true)
        } else {
            self.discard();
            Err(problems)
        }
    }

    /// Takes the numbers [`Runs::prepare`] kept apart, each instance prepared again
    /// starting from its initial memory, as one made afresh, and every other keeping what
    /// it remembers.
    pub fn commit(&mut self) {
        for run in &mut self.groups {
            if run.commit() {
                self.stamped = false;
            }
        }
    }

    /// Forgets what every instance remembers and what the slab holds, so the run starts as
    /// one made afresh.
    pub fn forget(&mut self) {
        for run in &mut self.groups {
            run.forget();
        }
        for values in self.slab.rows.iter_mut().chain(&mut self.slab.entries) {
            values.fill(0.0);
        }
        self.stamped = false;
    }

    /// The slab as the last evaluation wrote it: each instance's residuals, scales and
    /// charges of its local unknowns, and conductances and capacitances of its local
    /// entries, and beyond every block the copies of those that contribute to a long
    /// value, at the places [`Groups::gathers`] reads them.
    #[must_use]
    pub fn slab(&self) -> ([&[f64]; 3], [&[f64]; 2]) {
        self.slab.slices()
    }

    /// Forgets the numbers [`Runs::prepare`] kept apart.
    fn discard(&mut self) {
        for run in &mut self.groups {
            run.discard();
        }
    }

    /// Writes into the block of each linear instance of `groups` its entries, as its kernel
    /// evaluates them at [`Input::ZERO`], as at any state, and sums them into the base, each
    /// value from zero, its sources one after another in their order.
    fn stamp(&mut self, groups: &Groups) {
        let (mut rows, mut entries) = self.slab.slices_mut();
        for (group, run) in groups.kernels.iter().zip(&mut self.groups) {
            let count = group.members.len();
            let (Some(rows), Some(entries)) = (
                split(&mut rows, count * group.unknowns()),
                split(&mut entries, count * group.stride()),
            ) else {
                break;
            };
            if group.linear {
                run.stamp(group, (rows, entries));
            }
        }
        let (first, summed) = groups.base();
        for plane in &mut self.slab.entries {
            if first <= plane.len() {
                let (blocks, base) = plane.split_at_mut(first);
                base.fill(0.0);
                summed.add(blocks, base);
            }
        }
        self.stamped = true;
    }

    /// Evaluates every instance of `groups` at `input` and assembles their contributions in
    /// `into`, whatever it held: serially without `tasks`, `into` zeroed, the base added to
    /// it and each instance's block added as soon as it is written, and else in parallel,
    /// every group at once, each that is not linear in the tasks of instances `tasks`
    /// sets and each linear one in one, every block
    /// written first and then gathered into `into`, each value from zero, the base's first,
    /// as the threads take them. A linear instance's block keeps its entries, which the base
    /// sums, and its residuals, their scales and its charges come from them. Returns
    /// whether an instance limited its voltages.
    pub fn evaluate(
        &mut self,
        groups: &Groups,
        input: &Input<'_>,
        into: &mut Contributions,
        tasks: Option<Tasks>,
    ) -> bool {
        if !self.stamped {
            self.stamp(groups);
        }
        let (mut rows, mut entries) = self.slab.slices_mut();
        // The copies lie beyond every group's blocks, and the base beyond the copies.
        let (mut row_copies, mut entry_copies) = (
            beyond(&mut rows, groups.sizes.0),
            beyond(&mut entries, groups.sizes.1),
        );
        let base = beyond(&mut entry_copies, groups.copies.1);
        let runs = groups.kernels.iter().zip(&mut self.groups);
        let mut blocks = |group: &Group| {
            blocks_of(
                group,
                (&mut rows, &mut entries),
                (&mut row_copies, &mut entry_copies),
            )
        };
        let Some(tasks) = tasks else {
            into.clear();
            add_entries(into, &groups.base, as_refs(&base));
            let mut limited = false;
            for (group, run) in runs {
                if let Some(blocks) = blocks(group) {
                    limited |= run.evaluate(group, input, blocks, Some(&mut *into), 1);
                }
            }
            return limited;
        };
        // Each group writes blocks of its own, and says in a flag of its own whether it
        // limited, so the groups evaluate at once, as they would one after another.
        self.limited.resize(groups.kernels.len(), false);
        rayon::scope(|scope| {
            for ((group, run), limited) in runs.zip(&mut self.limited) {
                *limited = false;
                if let Some(blocks) = blocks(group) {
                    // A linear group's instances are not evaluated, so one task takes its
                    // rows.
                    let members = group.members.len();
                    let per_task = if group.linear {
                        members.max(1)
                    } else {
                        tasks.size(members)
                    };
                    scope.spawn(move |_| {
                        *limited = run.evaluate(group, input, blocks, None, per_task);
                    });
                }
            }
        });
        let (row_planes, entry_planes) = self.slab.slices();
        let (rows, slots) = groups.gathers();
        let Contributions {
            residual: sums,
            scale: largest,
            charge: held,
            conductance: conducted,
            capacitance: stored,
            ..
        } = into;
        // The unknowns' values and the stored entries' gather at once, as the threads free.
        rayon::join(
            || {
                rows.fold_in_parallel(
                    row_planes,
                    [sums, largest, held],
                    [Fold::Add, Fold::Raise, Fold::Add],
                );
            },
            || {
                slots.fold_in_parallel(entry_planes, [conducted, stored], [Fold::Add, Fold::Add]);
            },
        );
        self.limited.iter().any(|&limited| limited)
    }
}
