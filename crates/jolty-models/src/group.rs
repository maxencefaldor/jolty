//! Groups: the instances of one kernel and one shape, evaluated together, and the only code
//! that touches threads.
//!
//! [`Groups`] holds what a circuit's groups are, built once with its structure and shared:
//! each group's kernel and shape, its instances, the circuit's unknown of each of their
//! local unknowns and its stored entry of each of their local entries, and where each
//! instance's block lies in a slab. The devices that are no kernel's instances it lists
//! apart, to be evaluated on their own. [`Runs`] holds what one run writes: each instance's
//! prepared numbers and memory, and the slab.
//!
//! The devices of no kernel add their contributions first, each in place, in the order of
//! the devices. Each instance then writes its contributions into its own block, and every
//! value of the circuit's equations adds the blocks' contributions to it in one fixed
//! order: group after group, instance after instance, local index after local index. A run
//! evaluated serially adds each instance's block as soon as it is written; one evaluated in
//! parallel writes every block first, then gathers them through a [`GatherMap`] in the same
//! order. Either way, the sums are the same bits.

use std::any::{Any, TypeId};
use std::fmt;
use std::num::NonZeroUsize;

use jolty_numerics::{Entry, GatherMap, Pattern, UNMAPPED};
use rayon::prelude::*;

use crate::kernel::device::mapped;
use crate::surroundings::Own;
use crate::{
    Binding, Contributions, Input, Kernel, Layout, Node, NoiseSource, ParameterError, Stamp,
};

/// The parameters of an instance of some kernel, read, as an [`Instance`] holds them.
type Erased = dyn Any + Send + Sync;

/// What prepares an instance at a temperature, in kelvins, from its parameters, and claims
/// its layout: its shape and its numbers, typed.
type MadeBy = fn(&Erased, f64, &mut Layout) -> Result<Box<dyn Made>, ParameterError>;

/// One instance of a kernel, as its specification reads it: its parameters and its own
/// temperature, which group it with the other instances of its kernel and its shape, and
/// prepare it for each run.
#[derive(Clone, Copy)]
pub struct Instance<'a> {
    kernel: TypeId,
    name: &'static str,
    parameters: &'a Erased,
    /// Prepares the instance and claims its layout.
    made: MadeBy,
    /// The names of the variables the kernel computes.
    variables: fn() -> &'static [&'static str],
    own: Own,
}

impl fmt::Debug for Instance<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Instance")
            .field("kernel", &self.name)
            .field("own", &self.own)
            .finish_non_exhaustive()
    }
}

impl<'a> Instance<'a> {
    /// An instance of the kernel `K` with `parameters`, at the temperature of its
    /// surroundings.
    #[must_use]
    pub fn of<K: Kernel>(parameters: &'a K::Parameters<f64>) -> Self {
        Self {
            kernel: TypeId::of::<K>(),
            name: K::NAME,
            parameters,
            made: made::<K>,
            variables: K::variable_names,
            own: Own::default(),
        }
    }

    /// The same instance at its own temperature, `own`.
    pub(crate) const fn at_own(self, own: Own) -> Self {
        Self { own, ..self }
    }

    /// The names of the variables the instance computes on its way to its currents, as
    /// [`Kernel::variable_names`] gives them.
    #[must_use]
    pub fn variable_names(&self) -> &'static [&'static str] {
        (self.variables)()
    }

    /// The parameters, if the instance is one of `K`.
    fn parameters<K: Kernel>(&self) -> Option<&'a K::Parameters<f64>> {
        self.parameters.downcast_ref()
    }
}

/// An instance prepared and laid out, before it joins its group: its shape and its
/// numbers, typed.
trait Made: Send {
    /// The shape, as the typed half of a group of it.
    fn shaped(&self) -> &dyn Shaped;

    /// The instance as [`Any`], for its group's run to take its numbers.
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
}

/// An instance of `K` prepared and laid out.
struct Fresh<K: Kernel> {
    shape: Shape<K>,
    prepared: K::Prepared<f64>,
}

impl<K: Kernel> Made for Fresh<K> {
    fn shaped(&self) -> &dyn Shaped {
        &self.shape
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

/// `parameters`, an instance of `K`, prepared at `temperature` and laid out in `layout`.
fn made<K: Kernel>(
    parameters: &Erased,
    temperature: f64,
    layout: &mut Layout,
) -> Result<Box<dyn Made>, ParameterError> {
    let parameters = parameters
        .downcast_ref::<K::Parameters<f64>>()
        .ok_or_else(|| ParameterError(format!("the parameters of no {}", K::NAME)))?;
    let prepared = K::prepare(parameters, temperature)?;
    let shape = Shape::<K>(K::layout(parameters, layout)?);
    Ok(Box::new(Fresh::<K> { shape, prepared }))
}

/// What a group is of: a kernel and a shape, and how a run of the group starts.
trait Shaped: fmt::Debug + Send + Sync {
    /// The group's kernel.
    fn kernel(&self) -> TypeId;

    /// Whether `other` is of the same kernel and shape.
    fn same(&self, other: &dyn Shaped) -> bool;

    /// The shape as [`Any`], for [`Shaped::same`] to compare.
    fn as_any(&self) -> &dyn Any;

    /// The shape, boxed, as the group of it holds it.
    fn boxed(&self) -> Box<dyn Shaped>;

    /// A run of a group of `members` instances, none prepared yet.
    fn run(&self, members: usize) -> Box<dyn GroupRun>;
}

/// The shape of a group of the kernel `K`.
struct Shape<K: Kernel>(K::Shape);

impl<K: Kernel> fmt::Debug for Shape<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple(K::NAME).field(&self.0).finish()
    }
}

impl<K: Kernel> Shaped for Shape<K> {
    fn kernel(&self) -> TypeId {
        TypeId::of::<K>()
    }

    fn same(&self, other: &dyn Shaped) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.0 == self.0)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn boxed(&self) -> Box<dyn Shaped> {
        Box::new(Self(self.0.clone()))
    }

    fn run(&self, members: usize) -> Box<dyn GroupRun> {
        Box::new(KernelRun::<K> {
            shape: self.0.clone(),
            prepared: std::iter::repeat_with(|| None).take(members).collect(),
            memory: vec![K::Memory::default(); members],
            staged: Vec::new(),
        })
    }
}

/// The instances of one kernel and one shape, as a circuit's structure fixes them.
#[derive(Debug)]
struct Group {
    shaped: Box<dyn Shaped>,
    layout: Layout,
    /// Each instance's device, by its index among the circuit's devices, in their order.
    members: Vec<usize>,
    /// The circuit's unknown of each instance's local unknowns, instance after instance,
    /// or [`UNMAPPED`] for one on the reference.
    nodes: Vec<u32>,
    /// The circuit's entry of each instance's local entries, instance after instance, or
    /// none for one that joins the reference, until the pattern is known.
    claims: Vec<Option<Entry>>,
    /// The stored position of each instance's local entries, instance after instance, a
    /// stride apart, or [`UNMAPPED`] for none.
    slots: Vec<u32>,
    /// Where the group's blocks begin among the slab's local unknowns and its local
    /// entries.
    starts: (usize, usize),
}

impl Group {
    /// The number of local unknowns of each instance.
    fn unknowns(&self) -> usize {
        self.layout.unknowns()
    }

    /// The number of the slab's places for each instance's local entries: one for each,
    /// and one for an instance with none, so that every block has a place.
    fn stride(&self) -> usize {
        self.layout.entries().len().max(1)
    }

    /// The circuit's unknown of each local unknown of the instance `member`.
    fn map_of(&self, member: usize) -> Option<&[u32]> {
        let unknowns = self.unknowns();
        self.nodes.get(member * unknowns..(member + 1) * unknowns)
    }

    /// The stored position of each local entry of the instance `member`, a stride of them.
    fn slots_of(&self, member: usize) -> Option<&[u32]> {
        let stride = self.stride();
        self.slots.get(member * stride..(member + 1) * stride)
    }
}

/// The groups of a circuit's devices: those of a kernel's instances, by kernel and shape in
/// the order their first instances appear, and the devices of no kernel, which are
/// evaluated on their own.
#[derive(Debug, Default)]
pub struct Groups {
    kernels: Vec<Group>,
    legacy: Vec<usize>,
    /// Each device's group and its index among the group's instances, by the device's
    /// index, or `None` for a device of no kernel.
    places: Vec<Option<(usize, usize)>>,
    /// The sources of each unknown's residual, scale and charge, and of each stored
    /// entry's conductance and capacitance, among the slab's.
    rows: GatherMap,
    slots: GatherMap,
    /// The slab's local unknowns and local entries.
    sizes: (usize, usize),
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

    /// Adds the next device among the circuit's, of no kernel, to be evaluated on its own.
    pub fn add_legacy(&mut self) {
        self.legacy.push(self.places.len());
        self.places.push(None);
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
        let mut layout = Layout::new(terminals);
        let temperature = instance.own.temperature(temperature)?;
        let made = (instance.made)(instance.parameters, temperature, &mut layout)?;
        let found = self.kernels.iter().position(|group| {
            group.shaped.kernel() == instance.kernel && group.shaped.same(made.shaped())
        });
        let index = found.unwrap_or_else(|| {
            runs.groups.push(made.shaped().run(0));
            self.kernels.push(Group {
                shaped: made.shaped().boxed(),
                layout: layout.clone(),
                members: Vec::new(),
                nodes: Vec::new(),
                claims: Vec::new(),
                slots: Vec::new(),
                starts: (0, 0),
            });
            self.kernels.len() - 1
        });
        runs.groups[index].push(made.into_any());
        let mut nodes: Vec<Node> = binding.terminals().to_vec();
        for &(label, quantity) in layout.internals() {
            nodes.push(Some(binding.internal(label, quantity)));
        }
        let node = |local: usize| nodes.get(local).copied().flatten();
        let group = &mut self.kernels[index];
        let device = self.places.len();
        self.places.push(Some((index, group.members.len())));
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

    /// The instance at the device `device`, if it is a kernel's: its group and its index
    /// among the group's instances.
    fn place(&self, device: usize) -> Option<(usize, usize)> {
        self.places.get(device).copied().flatten()
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
        if !(fits(rows) && fits(stored) && fits(pattern.dimension()) && fits(pattern.len())) {
            return Err(TooLarge);
        }
        self.sizes = (rows, stored);
        let sources = |starts: fn(&Group) -> usize, maps: fn(&Group) -> &[u32]| {
            self.kernels.iter().flat_map(move |group| {
                let start = starts(group);
                maps(group)
                    .iter()
                    .enumerate()
                    .map(move |(offset, &target)| {
                        (target, u32::try_from(start + offset).unwrap_or(UNMAPPED))
                    })
            })
        };
        self.rows = GatherMap::new(
            pattern.dimension(),
            sources(|group| group.starts.0, |group| &group.nodes),
        );
        self.slots = GatherMap::new(
            pattern.len(),
            sources(|group| group.starts.1, |group| &group.slots),
        );
        runs.slab = Slab::sized(self.sizes);
        Ok(())
    }

    /// The devices of no kernel, evaluated on their own, by index, in order.
    #[must_use]
    pub fn legacy(&self) -> &[usize] {
        &self.legacy
    }

    /// Whether the device at index `device` is a kernel's instance, which its group
    /// evaluates.
    #[must_use]
    pub fn of_kernel(&self, device: usize) -> bool {
        self.place(device).is_some()
    }

    /// The maps that gather, from a run's [slab](Runs::slab), the contributions to each
    /// unknown's residual, scale and charge, and to each stored entry's conductance and
    /// capacitance, in the order they add.
    #[must_use]
    pub const fn gathers(&self) -> (&GatherMap, &GatherMap) {
        (&self.rows, &self.slots)
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
            slab: Slab::sized(self.sizes),
            layout: Layout::default(),
        }
    }
}

/// `slot`, a stored position, as a map holds it, or [`UNMAPPED`] beyond its reach, which
/// [`Groups::finish`] rules out.
fn mapped_slot(slot: usize) -> u32 {
    u32::try_from(slot).unwrap_or(UNMAPPED)
}

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

/// How a run of the groups splits its work among the threads of the current rayon pool:
/// the instances of a group a task evaluates, and the values of the circuit's equations a
/// task gathers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tasks {
    /// The instances a task evaluates.
    pub instances: NonZeroUsize,
    /// The values a task gathers.
    pub targets: NonZeroUsize,
}

/// The slab: each instance's block of residuals, scales and charges of its local unknowns,
/// and of conductances and capacitances of its local entries, group after group.
#[derive(Debug, Clone, Default)]
struct Slab {
    rows: [Vec<f64>; 3],
    entries: [Vec<f64>; 2],
}

impl Slab {
    /// A zeroed slab of `rows` local unknowns and `entries` local entries.
    fn sized((rows, entries): (usize, usize)) -> Self {
        Self {
            rows: [vec![0.0; rows], vec![0.0; rows], vec![0.0; rows]],
            entries: [vec![0.0; entries], vec![0.0; entries]],
        }
    }
}

/// What one run of a circuit's groups writes: each instance's prepared numbers and memory,
/// and the slab its evaluations write.
#[derive(Debug, Default)]
pub struct Runs {
    groups: Vec<Box<dyn GroupRun>>,
    slab: Slab,
    /// The layout each instance prepared again claims, to check its shape.
    layout: Layout,
}

impl Clone for Runs {
    fn clone(&self) -> Self {
        Self {
            groups: self.groups.iter().map(|run| run.cloned()).collect(),
            slab: self.slab.clone(),
            layout: self.layout.clone(),
        }
    }
}

/// One instance's block of contributions, evaluated alone, and where its values go among
/// the circuit's: the residual, scale and charge of each local unknown, the conductance and
/// capacitance of each local entry, the circuit's unknown of each local unknown, or
/// [`UNMAPPED`] for one on the reference, and the stored position of each local entry, or
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
        if let Some((group, member)) = groups.place(device)
            && let (Some(of), Some(run)) = (groups.kernels.get(group), self.groups.get(group))
        {
            run.noise(member, of, input, sources);
        }
    }

    /// Writes the variables of the instance the device `device` is, if it is a kernel's,
    /// as its last evaluation left them, to `values`, one per name of
    /// [`Instance::variable_names`].
    pub fn variables(&self, groups: &Groups, device: usize, values: &mut [f64]) {
        if let Some((group, member)) = groups.place(device)
            && let Some(run) = self.groups.get(group)
        {
            run.variables(member, values);
        }
    }

    /// Evaluates the instance the device `device` is, if it is a kernel's, alone at
    /// `input`, into a block of its own, from what it remembers, which it updates.
    pub fn evaluate_alone(
        &mut self,
        groups: &Groups,
        device: usize,
        input: &Input<'_>,
    ) -> Option<Alone> {
        let (group, member) = groups.place(device)?;
        let (of, run) = (groups.kernels.get(group)?, self.groups.get_mut(group)?);
        let (unknowns, stride) = (of.unknowns(), of.stride());
        let mut alone = Alone {
            rows: [
                vec![0.0; unknowns],
                vec![0.0; unknowns],
                vec![0.0; unknowns],
            ],
            entries: [vec![0.0; stride], vec![0.0; stride]],
            nodes: of.map_of(member)?.to_vec(),
            slots: of.slots_of(member)?.to_vec(),
            limited: false,
        };
        let [residual, scale, charge] = &mut alone.rows;
        let [conductance, capacitance] = &mut alone.entries;
        alone.limited = run.evaluate_alone(
            member,
            of,
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
        instance: impl Fn(usize) -> Option<Instance<'a>>,
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
                let prepared = instance(device).map_or(Ok(false), |instance| {
                    run.prepare(member, (group, &instance), temperature, checked)
                });
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

    /// Takes the numbers [`Runs::prepare`] kept apart, and forgets what every instance
    /// remembers and what the slab holds, so the run starts as one made afresh.
    pub fn commit(&mut self) {
        for run in &mut self.groups {
            run.commit();
        }
        for values in self.slab.rows.iter_mut().chain(&mut self.slab.entries) {
            values.fill(0.0);
        }
    }

    /// The slab as the last evaluation wrote it: each instance's residuals, scales and
    /// charges of its local unknowns, and conductances and capacitances of its local
    /// entries, at the places [`Groups::gathers`] reads them.
    #[must_use]
    pub fn slab(&self) -> ([&[f64]; 3], [&[f64]; 2]) {
        let [residual, scale, charge] = &self.slab.rows;
        let [conductance, capacitance] = &self.slab.entries;
        ([residual, scale, charge], [conductance, capacitance])
    }

    /// Forgets the numbers [`Runs::prepare`] kept apart.
    fn discard(&mut self) {
        for run in &mut self.groups {
            run.discard();
        }
    }

    /// Evaluates every instance of `groups` at `input` and adds its contributions to
    /// `into`, which holds those of the devices of no kernel: serially without `tasks`,
    /// each instance's block added as soon as it is written, and else in parallel, in the
    /// tasks `tasks` sets, every block written first and then gathered. Returns whether an
    /// instance limited its voltages.
    pub fn evaluate(
        &mut self,
        groups: &Groups,
        input: &Input<'_>,
        into: &mut Contributions,
        tasks: Option<Tasks>,
    ) -> bool {
        let [residual, scale, charge] = &mut self.slab.rows;
        let [conductance, capacitance] = &mut self.slab.entries;
        let (mut rows, mut entries) = (
            [residual.as_mut_slice(), scale, charge],
            [conductance.as_mut_slice(), capacitance],
        );
        let mut limited = false;
        for (group, run) in groups.kernels.iter().zip(&mut self.groups) {
            let count = group.members.len();
            let (Some(block), Some(stored)) = (
                split(&mut rows, count * group.unknowns()),
                split(&mut entries, count * group.stride()),
            ) else {
                continue;
            };
            let target = if tasks.is_some() {
                None
            } else {
                Some(&mut *into)
            };
            let per_task = tasks.map_or(1, |tasks| tasks.instances.get());
            limited |= run.evaluate(group, input, (block, stored), target, per_task);
        }
        if let Some(tasks) = tasks {
            let per_task = tasks.targets.get();
            let [residual, scale, charge] = &self.slab.rows;
            let [conductance, capacitance] = &self.slab.entries;
            groups
                .rows
                .add_in_parallel(residual, &mut into.residual, per_task);
            groups
                .rows
                .raise_in_parallel(scale, &mut into.scale, per_task);
            groups
                .rows
                .add_in_parallel(charge, &mut into.charge, per_task);
            groups
                .slots
                .add_in_parallel(conductance, &mut into.conductance, per_task);
            groups
                .slots
                .add_in_parallel(capacitance, &mut into.capacitance, per_task);
        }
        limited
    }
}

/// The first `count` values of each of `planes`, which keep the rest, or `None` when one
/// holds fewer.
fn split<'a, const N: usize>(
    planes: &mut [&'a mut [f64]; N],
    count: usize,
) -> Option<[&'a mut [f64]; N]> {
    if planes.iter().any(|plane| plane.len() < count) {
        return None;
    }
    Some(planes.each_mut().map(|plane| {
        let (first, rest) = std::mem::take(plane).split_at_mut(count);
        *plane = rest;
        first
    }))
}

/// A group's blocks in the slab: the residuals, scales and charges of its instances' local
/// unknowns, and the conductances and capacitances of their local entries.
type Blocks<'a> = ([&'a mut [f64]; 3], [&'a mut [f64]; 2]);

/// What one run of a group writes: its instances' prepared numbers and memory.
trait GroupRun: fmt::Debug + Send {
    /// Adds an instance of the group's kernel and shape, `made` as [`Made`] makes it.
    fn push(&mut self, made: Box<dyn Any>);

    /// The run, cloned.
    fn cloned(&self) -> Box<dyn GroupRun>;

    /// Appends the noise of the instance `member` of `group` at `input` to `sources`, each
    /// source's equations the circuit's.
    fn noise(
        &self,
        member: usize,
        group: &Group,
        input: &Input<'_>,
        sources: &mut Vec<NoiseSource>,
    );

    /// Writes the variables of the instance `member` at its last evaluation to `values`.
    fn variables(&self, member: usize, values: &mut [f64]);

    /// Evaluates the instance `member` of `group` alone at `input`, into `block`, which it
    /// zeroes first. Returns whether it limited its voltages.
    fn evaluate_alone(
        &mut self,
        member: usize,
        group: &Group,
        input: &Input<'_>,
        block: Blocks<'_>,
    ) -> bool;

    /// Prepares the instance `member` of the group, the `instance` of its device, at the
    /// temperature `temperature` of its surroundings, keeping the numbers apart; with a
    /// `layout`, its layout is claimed again there to check its shape. Returns whether it
    /// has the group's kernel and shape.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when its parameters are out of range.
    fn prepare(
        &mut self,
        member: usize,
        of: (&Group, &Instance<'_>),
        temperature: f64,
        layout: Option<&mut Layout>,
    ) -> Result<bool, ParameterError>;

    /// Takes the numbers kept apart, and forgets every instance's memory.
    fn commit(&mut self);

    /// Forgets the numbers kept apart.
    fn discard(&mut self);

    /// Evaluates each instance of `group` at `input` into its block of `blocks`: one
    /// after another, each block added to `into` as soon as it is written, or without
    /// `into`, in parallel, `per_task` instances a task. Returns whether one limited its
    /// voltages.
    fn evaluate(
        &mut self,
        group: &Group,
        input: &Input<'_>,
        blocks: Blocks<'_>,
        into: Option<&mut Contributions>,
        per_task: usize,
    ) -> bool;
}

/// One run of a group of the kernel `K`.
struct KernelRun<K: Kernel> {
    shape: K::Shape,
    prepared: Vec<Option<K::Prepared<f64>>>,
    memory: Vec<K::Memory>,
    /// The numbers prepared and not yet taken, by instance.
    staged: Vec<(usize, K::Prepared<f64>)>,
}

impl<K: Kernel> fmt::Debug for KernelRun<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(K::NAME)
            .field("shape", &self.shape)
            .field("prepared", &self.prepared)
            .field("memory", &self.memory)
            .field("staged", &self.staged)
            .finish()
    }
}

impl<K: Kernel> GroupRun for KernelRun<K> {
    fn push(&mut self, made: Box<dyn Any>) {
        let prepared = made.downcast::<Fresh<K>>().ok().map(|fresh| fresh.prepared);
        self.prepared.push(prepared);
        self.memory.push(K::Memory::default());
    }

    fn cloned(&self) -> Box<dyn GroupRun> {
        Box::new(Self {
            shape: self.shape.clone(),
            prepared: self.prepared.clone(),
            memory: self.memory.clone(),
            staged: self.staged.clone(),
        })
    }

    fn noise(
        &self,
        member: usize,
        group: &Group,
        input: &Input<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        let (Some(Some(prepared)), Some(map)) = (self.prepared.get(member), group.map_of(member))
        else {
            return;
        };
        let start = sources.len();
        K::noise(&self.shape, prepared, &input.local(map), sources);
        for source in sources.iter_mut().skip(start) {
            source.equations = source.equations.map(|equation| {
                equation.and_then(|local| {
                    map.get(local.index())
                        .filter(|&&unknown| unknown != UNMAPPED)
                        .map(|&unknown| crate::Unknown::new(unknown as usize))
                })
            });
        }
    }

    fn variables(&self, member: usize, values: &mut [f64]) {
        if let Some(memory) = self.memory.get(member) {
            K::variables(memory, values);
        }
    }

    fn evaluate_alone(
        &mut self,
        member: usize,
        group: &Group,
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
        K::evaluate(&self.shape, prepared, memory, &input.local(map), &mut stamp);
        stamp.is_limited()
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
        if let Some(layout) = layout {
            layout.clear(group.layout.terminals());
            if K::layout(parameters, layout)? != self.shape {
                return Ok(false);
            }
        }
        self.staged.push((member, prepared));
        Ok(true)
    }

    fn commit(&mut self) {
        for (member, prepared) in self.staged.drain(..) {
            if let Some(slot) = self.prepared.get_mut(member) {
                *slot = Some(prepared);
            }
        }
        self.memory.fill(K::Memory::default());
    }

    fn discard(&mut self) {
        self.staged.clear();
    }

    fn evaluate(
        &mut self,
        group: &Group,
        input: &Input<'_>,
        ([residual, scale, charge], [conductance, capacitance]): Blocks<'_>,
        into: Option<&mut Contributions>,
        per_task: usize,
    ) -> bool {
        let sizes = (group.unknowns(), group.stride());
        let (unknowns, stride) = sizes;
        if unknowns == 0 {
            return false;
        }
        let shape = &self.shape;
        let Some(into) = into else {
            let (rows, stored) = (unknowns * per_task, stride * per_task);
            return residual
                .par_chunks_mut(rows)
                .zip(scale.par_chunks_mut(rows))
                .zip(charge.par_chunks_mut(rows))
                .zip(conductance.par_chunks_mut(stored))
                .zip(capacitance.par_chunks_mut(stored))
                .zip(self.memory.par_chunks_mut(per_task))
                .zip(self.prepared.par_chunks(per_task))
                .zip(group.nodes.par_chunks(rows))
                .map(|chunk| {
                    let (
                        (
                            (((((residual, scale), charge), conductance), capacitance), memory),
                            prepared,
                        ),
                        nodes,
                    ) = chunk;
                    let blocks = ([residual, scale, charge], [conductance, capacitance]);
                    evaluate_chunk::<K>(
                        shape,
                        sizes,
                        input,
                        (blocks, memory, prepared),
                        (nodes, &[]),
                        None,
                    )
                })
                .reduce(|| false, |one, other| one | other);
        };
        let blocks = ([residual, scale, charge], [conductance, capacitance]);
        let instances = (blocks, self.memory.as_mut_slice(), self.prepared.as_slice());
        evaluate_chunk::<K>(
            shape,
            sizes,
            input,
            instances,
            (&group.nodes, &group.slots),
            Some(into),
        )
    }
}

/// The instances of a chunk of a group: their blocks, their memory and their prepared
/// numbers.
type Chunk<'a, K> = (
    Blocks<'a>,
    &'a mut [<K as Kernel>::Memory],
    &'a [Option<<K as Kernel>::Prepared<f64>>],
);

/// Evaluates each instance of a chunk of a group of `shape`, of `sizes` local unknowns and
/// places for local entries each, at `input`, into its block, which it zeroes first, and
/// adds each block to `into`, when given, as soon as it is written, through the circuit's
/// unknowns and stored positions of the instances' local ones, `maps`. Returns whether an
/// instance limited its voltages.
fn evaluate_chunk<K: Kernel>(
    shape: &K::Shape,
    (unknowns, stride): (usize, usize),
    input: &Input<'_>,
    (([residual, scale, charge], [conductance, capacitance]), memory, prepared): Chunk<'_, K>,
    (nodes, slots): (&[u32], &[u32]),
    mut into: Option<&mut Contributions>,
) -> bool {
    for plane in [
        &mut *residual,
        &mut *scale,
        &mut *charge,
        &mut *conductance,
        &mut *capacitance,
    ] {
        plane.fill(0.0);
    }
    let blocks = residual
        .chunks_mut(unknowns)
        .zip(scale.chunks_mut(unknowns))
        .zip(charge.chunks_mut(unknowns))
        .zip(conductance.chunks_mut(stride))
        .zip(capacitance.chunks_mut(stride));
    let maps = nodes.chunks(unknowns).zip(
        slots
            .chunks(stride)
            .map(Some)
            .chain(std::iter::repeat(None)),
    );
    let instances = memory.iter_mut().zip(prepared).zip(maps);
    let mut limited = false;
    for (
        ((((residual, scale), charge), conductance), capacitance),
        ((memory, prepared), (map, slots)),
    ) in blocks.zip(instances)
    {
        let Some(prepared) = prepared else { continue };
        let at = input.local(map);
        let mut stamp = Stamp::over(
            [&mut *residual, &mut *scale, &mut *charge],
            [&mut *conductance, &mut *capacitance],
        );
        K::evaluate(shape, prepared, memory, &at, &mut stamp);
        limited |= stamp.is_limited();
        if let (Some(into), Some(slots)) = (into.as_deref_mut(), slots) {
            add(
                into,
                (map, slots),
                ([residual, scale, charge], [conductance, capacitance]),
            );
        }
    }
    limited
}

/// Adds one instance's block to `into`: each local unknown's residual, scale and charge to
/// its unknown's of `map`, and each local entry's conductance and capacitance to its
/// stored position of `slots`, in the order of their local indices; those on the
/// reference, [`UNMAPPED`], go nowhere.
fn add(
    into: &mut Contributions,
    (map, slots): (&[u32], &[u32]),
    ([residual, scale, charge], [conductance, capacitance]): ([&mut [f64]; 3], [&mut [f64]; 2]),
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
