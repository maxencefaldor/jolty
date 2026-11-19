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

/// Scope of change to a device instance since the previous run preparation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Changed {
    /// Unchanged instance whose prepared parameters remain valid.
    Nothing,
    /// Temperature change or parameter restoration matching the existing group shape,
    /// requiring re-preparation without layout verification.
    Numbers,
    /// Parameter modification requiring both re-preparation and layout shape verification
    /// against the existing group.
    Parameters,
}

/// Work-partitioning policy dividing items across threads in the active rayon pool into
/// tasks of at least `least` items each and at most one task per thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tasks {
    /// Minimum number of items assigned to one parallel task.
    pub least: NonZeroUsize,
    /// Maximum number of threads sharing the tasks.
    pub threads: NonZeroUsize,
}

impl Tasks {
    /// Computes the number of tasks for `items` (at least 1, capped by [`Tasks::threads`],
    /// with at least [`Tasks::least`] items per task).
    #[must_use]
    pub fn count(&self, items: usize) -> usize {
        self.threads.get().min(items / self.least).max(1)
    }

    /// Computes the item count assigned to each of [`Tasks::count`] tasks, with the final
    /// task receiving any remainder.
    #[must_use]
    pub fn size(&self, items: usize) -> usize {
        items.div_ceil(self.count(items)).max(1)
    }
}

/// Contiguous evaluation slab storing per-instance row planes (residuals, scales, charges)
/// and entry planes (conductances, capacitances) across all groups.
#[derive(Debug, Clone, Default)]
pub(super) struct Slab {
    rows: [Vec<f64>; 3],
    entries: [Vec<f64>; 2],
}

impl Slab {
    /// Allocates a zero-initialized slab with `rows` row slots and `entries` entry slots.
    pub(super) fn sized((rows, entries): (usize, usize)) -> Self {
        Self {
            rows: std::array::from_fn(|_| vec![0.0; rows]),
            entries: std::array::from_fn(|_| vec![0.0; entries]),
        }
    }

    /// Returns shared slice views of all five slab planes.
    fn slices(&self) -> ([&[f64]; 3], [&[f64]; 2]) {
        let [residual, scale, charge] = &self.rows;
        let [conductance, capacitance] = &self.entries;
        ([residual, scale, charge], [conductance, capacitance])
    }

    /// Returns mutable slice views of all five slab planes.
    fn slices_mut(&mut self) -> Blocks<'_> {
        let [residual, scale, charge] = &mut self.rows;
        let [conductance, capacitance] = &mut self.entries;
        let rows = [residual.as_mut_slice(), scale, charge];
        let entries = [conductance.as_mut_slice(), capacitance];
        (rows, entries)
    }
}

/// Mutable per-run simulation state across all device groups, holding prepared parameters,
/// instance memory, and the evaluation slab.
#[derive(Debug, Default)]
pub struct Runs {
    pub(super) groups: Vec<Box<dyn GroupRun>>,
    pub(super) slab: Slab,
    /// Reusable scratch layout used to verify shapes when preparing modified parameters.
    pub(super) layout: Layout,
    /// Per-group voltage-limiting flags recorded during the most recent parallel evaluation.
    pub(super) limited: Vec<bool>,
    /// Flag indicating that linear entry blocks and the summed base region reflect current
    /// prepared parameters; reset on parameter commits or [`Runs::forget`].
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

/// Borrowed view of one device instance's slab block from the most recent evaluation.
///
/// Exposes local unknown residuals, scales, and charges alongside local entry conductances,
/// capacitances, row-column pairs, and global unknown indices ([`UNMAPPED`] for ground).
#[derive(Debug, Clone, Copy)]
pub struct Block<'a> {
    /// Residual values for each local unknown equation.
    pub residual: &'a [f64],
    /// Largest term magnitude contributing to each local residual.
    pub scale: &'a [f64],
    /// Charge or flux values for each local unknown equation.
    pub charge: &'a [f64],
    /// Conductance Jacobian values for each local entry.
    pub conductance: &'a [f64],
    /// Capacitance Jacobian values for each local entry.
    pub capacitance: &'a [f64],
    /// Local `(row, column)` unknown index pairs for each entry.
    pub entries: &'a [(usize, usize)],
    /// Global unknown indices for each local unknown ([`UNMAPPED`] for ground).
    pub nodes: &'a [u32],
}

/// Owned evaluation block produced by evaluating a single device instance in isolation,
/// paired with global unknown (`nodes`) and sparsity (`slots`) maps ([`UNMAPPED`] for
/// ground).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Alone {
    /// Residual, scale, and charge vectors for local unknowns.
    pub rows: [Vec<f64>; 3],
    /// Conductance and capacitance vectors for local entries.
    pub entries: [Vec<f64>; 2],
    /// Global unknown index for each local unknown.
    pub nodes: Vec<u32>,
    /// Stored sparsity slot index for each local entry.
    pub slots: Vec<u32>,
    /// Flag indicating whether voltage limiting triggered during evaluation.
    pub limited: bool,
}

impl Runs {
    /// Looks up the group, group run, and member index for `device`, if registered.
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

    /// Appends circuit-mapped noise sources for `device` linearized about operating point
    /// `input` to `sources`.
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

    /// Copies internal variable values recorded at the last evaluation of `device` into
    /// `values`, ordered to match [`Instance::variable_names`].
    pub fn variables(&self, groups: &Groups, device: usize, values: &mut [f64]) {
        if let Some((_, run, member)) = self.entry(groups, device) {
            run.variables(member, values);
        }
    }

    /// Builds a forward-mode sensitivity [`Twin`] for `device` from `instance` at ambient
    /// `temperature` along `seeds`, starting from initial memory.
    ///
    /// Returns `None` when a seed path addresses a non-real choice parameter or validation
    /// fails.
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

    /// Builds [`Twin`] instances for `device` at `instance.1` seeded along each distinct
    /// non-ground circuit unknown mapped to `device`, batched up to [`Seeds::MAX`] per twin
    /// with parameters held constant.
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

    /// Returns the time-zero DC value of `device` if it is an independent source.
    #[must_use]
    pub fn dc_value(&self, groups: &Groups, device: usize) -> Option<f64> {
        let (_, run, member) = self.entry(groups, device)?;
        run.dc_value(member)
    }

    /// Accumulates the small-signal residual derivative with respect to the source value of
    /// `device` into `derivative`.
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

    /// Returns local residual derivatives with respect to source value for `device` (all
    /// zeros if `device` is not an independent source).
    #[must_use]
    pub fn excitation(&self, groups: &Groups, device: usize) -> Option<Vec<f64>> {
        let (of, run, _) = self.entry(groups, device)?;
        Some(run.excitation(of))
    }

    /// Returns a read-only [`Block`] view of `device` in the evaluation slab.
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

    /// Starts logic driver `device` ramping toward `ramp.1` at `ramp.0` over full-swing
    /// duration `ramp.2`, returning the settling timestamp or `None` for non-drivers.
    pub fn drive(
        &mut self,
        groups: &Groups,
        device: usize,
        ramp: (f64, LogicLevel, f64),
    ) -> Option<f64> {
        let (group, member) = groups.place(device)?;
        self.groups.get_mut(group)?.drive(member, ramp)
    }

    /// Returns the branch-current unknown constraining the output voltage of logic driver
    /// `device`, if any.
    #[must_use]
    pub fn driven(&self, groups: &Groups, device: usize) -> Option<UnknownIndex> {
        let (of, run, member) = self.entry(groups, device)?;
        run.driven(member, of)
    }

    /// Evaluates input threshold margin and supply swing (in volts) at `input` for logic
    /// detector `device`, if applicable.
    #[must_use]
    pub fn sense(&self, groups: &Groups, device: usize, input: &Input<'_>) -> Option<(f64, f64)> {
        let (of, run, member) = self.entry(groups, device)?;
        run.sense(member, of, input)
    }

    /// Appends non-smooth waveform breakpoints within `span` to `times` so transient stepping
    /// lands on corners exactly.
    pub fn breakpoints(&self, span: (f64, f64), times: &mut Vec<f64>) {
        for run in &self.groups {
            run.breakpoints(span, times);
        }
    }

    /// Collects sorted device indices in `groups` with explicit time dependence per
    /// [`Kernel::varies_in_time`](crate::Kernel::varies_in_time).
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

    /// Collects all non-ground delayed conductance entries reported by
    /// [`Kernel::delayed`](crate::Kernel::delayed) across `groups` in group and member order.
    #[must_use]
    pub fn delayed(&self, groups: &Groups) -> Vec<DelayedEntry> {
        let mut found = Vec::new();
        for (group, run) in groups.kernels.iter().zip(&self.groups) {
            run.delayed(group, &mut found);
        }
        found
    }

    /// Locates the earliest linear-interpolation fraction between `before` and `after` where
    /// any behavioral branch choice flips, returning `None` if no branch changes.
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

    /// Evaluates `device` in isolation at `input` into a freshly allocated [`Alone`] block,
    /// updating instance memory and using cached linear terms when linear.
    pub fn evaluate_alone(
        &mut self,
        groups: &Groups,
        device: usize,
        input: &Input<'_>,
    ) -> Option<Alone> {
        self.alone(groups, device, input, true)
    }

    /// Evaluates `device` in isolation like [`Runs::evaluate_alone`], always invoking the
    /// underlying kernel directly even for linear instances.
    #[cfg(test)]
    pub(crate) fn evaluate_by_kernel(
        &mut self,
        groups: &Groups,
        device: usize,
        input: &Input<'_>,
    ) -> Option<Alone> {
        self.alone(groups, device, input, false)
    }

    /// Evaluates `device` into an [`Alone`] block, using cached linear terms when `linear` is
    /// true or direct kernel evaluation otherwise.
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

    /// Stages re-prepared parameters for every instance marked by `changed` at ambient
    /// `temperature`, leaving committed state untouched until [`Runs::commit`].
    ///
    /// Returns `Ok(false)` and discards staged updates if any modified instance changes
    /// kernel, linearity, or shape.
    ///
    /// # Errors
    ///
    /// Returns all `(device_index, ParameterError)` pairs if validation fails, discarding all
    /// staged updates.
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

    /// Commits staged updates from [`Runs::prepare`], resetting memory only on updated
    /// instances and preserving memory on unchanged instances.
    pub fn commit(&mut self) {
        for run in &mut self.groups {
            if run.commit() {
                self.stamped = false;
            }
        }
    }

    /// Resets every instance's memory, zeroes the slab, and clears cached linear stamps.
    pub fn forget(&mut self) {
        for run in &mut self.groups {
            run.forget();
        }
        for values in self.slab.rows.iter_mut().chain(&mut self.slab.entries) {
            values.fill(0.0);
        }
        self.stamped = false;
    }

    /// Returns read-only slab planes holding per-instance blocks and trailing high-fanout
    /// copies in the layout expected by [`Groups::gathers`].
    #[must_use]
    pub fn slab(&self) -> ([&[f64]; 3], [&[f64]; 2]) {
        self.slab.slices()
    }

    /// Discards staged updates from [`Runs::prepare`] across all groups.
    fn discard(&mut self) {
        for run in &mut self.groups {
            run.discard();
        }
    }

    /// Stamps constant Jacobian entries for all linear groups at [`Input::ZERO`] and sums
    /// them into the shared base region in deterministic order.
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

    /// Evaluates every instance in `groups` at `input`, assembles global contributions into
    /// `into`, and returns whether any instance limited voltages.
    ///
    /// Serial mode (`tasks` is `None`) clears `into`, adds the linear base, and accumulates
    /// each block immediately. Parallel mode (`tasks` is `Some`) evaluates groups
    /// concurrently and gathers blocks deterministically into `into`, reusing cached entries
    /// for linear groups.
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
        // Copies follow all group blocks, and the linear base follows the entry copies.
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
        // Each group writes to disjoint slab blocks and a dedicated limiting flag so groups
        // evaluate concurrently with deterministic results.
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
