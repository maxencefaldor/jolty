//! Groups: the instances of one kernel and one shape, evaluated together, and the only code
//! that touches threads.
//!
//! [`Groups`] holds what a circuit's groups are, built once with its structure and shared:
//! each group's kernel and shape, its instances, the circuit's unknown of each of their
//! local unknowns and its stored entry of each of their local entries, and where each
//! instance's block lies in a slab. [`Runs`] holds what one run writes: each instance's
//! prepared numbers and memory, and the slab.
//!
//! Each instance writes its contributions into its own block, and every value of the
//! circuit's equations adds the blocks' contributions to it in one fixed order: group after
//! group, instance after instance, local index after local index. A run
//! evaluated serially adds each instance's block as soon as it is written; one evaluated in
//! parallel writes every block first, then gathers them through a [`GatherMap`] in the same
//! order. Either way, the sums are the same bits.
//!
//! A value of more than [`CHUNK`] contributions, such as a supply's, joined to every device,
//! would make its gather read one place of nearly every block, one after another on one
//! thread. Each instance's contributions to such a value are also copied, as soon as its
//! block is written, to places of the slab beyond the blocks, where every such value's
//! contributions lie together in their order, and its gather reads them there: the same
//! values, in the same order.
//!
//! A linear instance, as [`Kernel::linear`] says, is evaluated once a run, for its entries,
//! which its block keeps: they sum, once a run, in the order of the groups and their
//! instances, into the base, which lies beyond the copies, one value for each stored entry
//! they add to. Each evaluation then adds to each stored entry the base's value first,
//! then the entries of the instances that are not linear, in their order, and writes each
//! linear instance's residuals, their scales and its charges from its entries, as its
//! kernel's equations are its entries times its local unknowns.

use std::any::{Any, TypeId};
use std::collections::BTreeMap;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::OnceLock;

use jolty_numerics::reduce::CHUNK;
use jolty_numerics::{Entry, Fold, GatherMap, Pattern, UNMAPPED};
use rayon::prelude::*;

use crate::lift::lift;
use crate::surroundings::Own;
use crate::{
    Binding, Choice, Contributions, Delayed, DelayedEntry, Gradient, Input, Kernel, Layout, Local,
    LogicLevel, Node, NoiseSource, ParameterError, Real, Regions, Stamp, UnknownIndex,
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
    /// The regions the kernel's equations distinguish.
    regions: fn() -> Regions,
    /// Every `(path, value)` pair that [`crate::Lift`] visits on the instance's parameters.
    #[cfg(any(test, feature = "testing"))]
    lift_entries: fn(&Erased) -> Vec<(String, f64)>,
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
        Self::of_kind::<K>(parameters)
    }

    /// An instance of the kind `K` with `parameters`, at the temperature of its
    /// surroundings.
    pub(crate) fn of_kind<K: Kind>(parameters: &'a K::Parameters) -> Self {
        Self {
            kernel: TypeId::of::<K>(),
            name: K::NAME,
            parameters,
            made: made::<K>,
            variables: K::variable_names,
            regions: K::regions,
            #[cfg(any(test, feature = "testing"))]
            lift_entries: lift_entries_of::<K>,
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

    /// The regions of operation the instance's equations distinguish, as
    /// [`Kernel::regions`] gives them.
    #[must_use]
    pub fn regions(&self) -> Regions {
        (self.regions)()
    }

    /// Claims, through `binding`, which binds the instance's terminals, the internal
    /// unknowns and the entries the instance's layout at the temperature `temperature` of
    /// its surroundings claims, as a group claims them for an instance it adds.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when the instance's parameters are out of range.
    pub fn claim(&self, temperature: f64, binding: &mut Binding<'_>) -> Result<(), ParameterError> {
        let mut layout = Layout::new(binding.terminals().len());
        (self.made)(
            self.parameters,
            self.own.temperature(temperature)?,
            &mut layout,
        )?;
        let mut nodes = binding.terminals().to_vec();
        for (label, nature) in layout.internals() {
            nodes.push(Some(binding.internal(label, *nature)));
        }
        let node = |local: usize| nodes.get(local).copied().flatten();
        for &(row, column) in layout.entries() {
            binding.entry(node(row), node(column));
        }
        Ok(())
    }

    /// The parameters, if the instance is one of `K`.
    fn parameters<K: Kind>(&self) -> Option<&'a K::Parameters> {
        self.parameters.downcast_ref()
    }

    /// Every `(path, value)` pair that [`crate::Lift`] visits on the instance's parameters,
    /// in visit order.
    #[cfg(any(test, feature = "testing"))]
    #[must_use]
    pub fn lift_entries(&self) -> Vec<(String, f64)> {
        (self.lift_entries)(self.parameters)
    }

    /// The own-temperature keys set on this instance, which [`crate::Surrounded`] handles
    /// outside [`crate::Lift`].
    #[cfg(any(test, feature = "testing"))]
    pub(crate) fn own_temperature_keys(&self) -> Vec<&'static str> {
        self.own.present()
    }
}

/// What a group's instances are, as the groups evaluate them: a [`Kernel`]'s, or those of a
/// kernel that heat, [`crate::kernel::heat::HeatedKernel`]. Its numbers are plain ones,
/// whatever numbers the kernel computes with within.
pub(crate) trait Kind: Send + Sync + 'static {
    /// The model's name.
    const NAME: &'static str;
    /// The parameters an instance holds.
    type Parameters: Send + Sync + 'static;
    /// The structure the parameters give an instance, as [`Kernel::Shape`].
    type Shape: Clone + fmt::Debug + PartialEq + Send + Sync + 'static;
    /// The numbers an instance's parameters set, as [`Kernel::Prepared`].
    type Prepared: Clone + fmt::Debug + Send + Sync;
    /// What an instance remembers, as [`Kernel::Memory`].
    type Memory: Clone + fmt::Debug + Default + Send + Sync;

    /// As [`Kernel::prepare`], at `temperature`, in kelvins.
    fn prepare(
        parameters: &Self::Parameters,
        temperature: f64,
    ) -> Result<Self::Prepared, ParameterError>;

    /// As [`Kernel::layout`].
    fn layout(
        parameters: &Self::Parameters,
        prepared: &Self::Prepared,
        layout: &mut Layout,
    ) -> Result<Self::Shape, ParameterError>;

    /// As [`Kernel::memory`].
    fn memory(shape: &Self::Shape, prepared: &Self::Prepared) -> Self::Memory;

    /// As [`Kernel::linear`].
    fn linear(shape: &Self::Shape, prepared: &Self::Prepared) -> bool;

    /// As [`Kernel::evaluate`].
    fn evaluate(
        shape: &Self::Shape,
        prepared: &Self::Prepared,
        memory: &mut Self::Memory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_>,
    );

    /// As [`Kernel::noise`], from what the instance remembers, `memory`, too.
    fn noise(
        shape: &Self::Shape,
        prepared: &Self::Prepared,
        memory: &Self::Memory,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    );

    /// As [`Kernel::variable_names`].
    fn variable_names() -> &'static [&'static str];

    /// As [`Kernel::regions`].
    fn regions() -> Regions;

    /// As [`Kernel::variables`].
    fn variables(memory: &Self::Memory, values: &mut [f64]);

    /// As [`Kernel::breakpoints`].
    fn breakpoints(prepared: &Self::Prepared, span: (f64, f64), times: &mut Vec<f64>);

    /// As [`Kernel::varies_in_time`].
    fn varies_in_time(shape: &Self::Shape, prepared: &Self::Prepared) -> bool;

    /// As [`Kernel::delayed`].
    fn delayed(shape: &Self::Shape, prepared: &Self::Prepared, found: &mut Vec<Delayed>);

    /// As [`Kernel::choices`].
    fn choices(
        shape: &Self::Shape,
        prepared: &Self::Prepared,
        memory: &mut Self::Memory,
        at: &Local<'_>,
        found: &mut Vec<Choice>,
    );

    /// As [`Kernel::level`].
    fn level(prepared: &Self::Prepared) -> Option<f64>;

    /// As [`Kernel::excite`].
    fn excite(shape: &Self::Shape, stamp: &mut Stamp<'_>);

    /// As [`Kernel::drive`].
    fn drive(memory: &mut Self::Memory, ramp: (f64, LogicLevel, f64)) -> Option<f64>;

    /// As [`Kernel::driven`].
    fn driven(shape: &Self::Shape) -> Option<usize>;

    /// As [`Kernel::sense`].
    fn sense(shape: &Self::Shape, prepared: &Self::Prepared, at: &Local<'_>) -> Option<(f64, f64)>;

    /// An instance's numbers computing with the derivatives of every number along `SEEDS`
    /// directions of its parameters, and what it remembers then.
    type Twin<const SEEDS: usize>: fmt::Debug + Send + Sync;

    /// The twin of an instance of `shape` with `parameters` at `temperature`, in kelvins,
    /// differentiated along `seeds`, each a number's path among them, as
    /// [`jolty_ir::numbers`] names it, and its rate along each direction, from its initial
    /// memory: `None` when the kernel reads no real number at a path, as when it reads it
    /// as a choice, such as a level, or a parameter is out of range.
    fn twin<const SEEDS: usize>(
        shape: &Self::Shape,
        parameters: &Self::Parameters,
        seeds: &[(&str, [f64; SEEDS])],
        temperature: f64,
    ) -> Option<Self::Twin<SEEDS>>;

    /// Evaluates the twin of an instance of `shape` at `at`, from what it remembers, which
    /// it updates, into `stamp`.
    fn evaluate_twin<const SEEDS: usize>(
        shape: &Self::Shape,
        twin: &mut Self::Twin<SEEDS>,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, Gradient<SEEDS>>,
    );

    /// Every `(path, value)` pair that [`crate::Lift`] visits on `parameters`, in visit
    /// order.
    #[cfg(any(test, feature = "testing"))]
    fn lift_entries(parameters: &Self::Parameters) -> Vec<(String, f64)>;
}

impl<K: Kernel> Kind for K {
    const NAME: &'static str = K::NAME;
    type Parameters = K::Parameters<f64>;
    type Shape = K::Shape;
    type Prepared = K::Prepared<f64>;
    type Memory = K::Memory;

    fn prepare(
        parameters: &K::Parameters<f64>,
        temperature: f64,
    ) -> Result<K::Prepared<f64>, ParameterError> {
        K::prepare(parameters, temperature)
    }

    fn layout(
        parameters: &K::Parameters<f64>,
        prepared: &K::Prepared<f64>,
        layout: &mut Layout,
    ) -> Result<K::Shape, ParameterError> {
        K::layout(parameters, prepared, layout)
    }

    fn memory(shape: &K::Shape, prepared: &K::Prepared<f64>) -> K::Memory {
        K::memory(shape, prepared)
    }

    fn linear(shape: &K::Shape, prepared: &K::Prepared<f64>) -> bool {
        K::linear(shape, prepared)
    }

    fn evaluate(
        shape: &K::Shape,
        prepared: &K::Prepared<f64>,
        memory: &mut K::Memory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_>,
    ) {
        K::evaluate(shape, prepared, memory, at, stamp);
    }

    fn noise(
        shape: &K::Shape,
        prepared: &K::Prepared<f64>,
        _: &K::Memory,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        K::noise(shape, prepared, at, sources);
    }

    fn variable_names() -> &'static [&'static str] {
        K::variable_names()
    }

    fn regions() -> Regions {
        K::regions()
    }

    fn variables(memory: &K::Memory, values: &mut [f64]) {
        K::variables(memory, values);
    }

    fn breakpoints(prepared: &K::Prepared<f64>, span: (f64, f64), times: &mut Vec<f64>) {
        K::breakpoints(prepared, span, times);
    }

    fn varies_in_time(shape: &K::Shape, prepared: &K::Prepared<f64>) -> bool {
        K::varies_in_time(shape, prepared)
    }

    fn delayed(shape: &K::Shape, prepared: &K::Prepared<f64>, found: &mut Vec<Delayed>) {
        K::delayed(shape, prepared, found);
    }

    fn choices(
        shape: &K::Shape,
        prepared: &K::Prepared<f64>,
        memory: &mut K::Memory,
        at: &Local<'_>,
        found: &mut Vec<Choice>,
    ) {
        K::choices(shape, prepared, memory, at, found);
    }

    fn level(prepared: &K::Prepared<f64>) -> Option<f64> {
        K::level(prepared)
    }

    fn excite(shape: &K::Shape, stamp: &mut Stamp<'_>) {
        K::excite(shape, stamp);
    }

    fn drive(memory: &mut K::Memory, ramp: (f64, LogicLevel, f64)) -> Option<f64> {
        K::drive(memory, ramp)
    }

    fn driven(shape: &K::Shape) -> Option<usize> {
        K::driven(shape)
    }

    fn sense(shape: &K::Shape, prepared: &K::Prepared<f64>, at: &Local<'_>) -> Option<(f64, f64)> {
        K::sense(shape, prepared, at)
    }

    type Twin<const SEEDS: usize> = (K::Prepared<Gradient<SEEDS>>, K::Memory);

    fn twin<const SEEDS: usize>(
        shape: &K::Shape,
        parameters: &K::Parameters<f64>,
        seeds: &[(&str, [f64; SEEDS])],
        temperature: f64,
    ) -> Option<Self::Twin<SEEDS>> {
        let lifted: K::Parameters<Gradient<SEEDS>> = lift(parameters, seeds, Gradient::new).ok()?;
        let prepared = K::prepare(&lifted, Gradient::constant(temperature)).ok()?;
        let memory = K::memory(shape, &prepared);
        Some((prepared, memory))
    }

    fn evaluate_twin<const SEEDS: usize>(
        shape: &K::Shape,
        (prepared, memory): &mut Self::Twin<SEEDS>,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, Gradient<SEEDS>>,
    ) {
        K::evaluate(shape, prepared, memory, at, stamp);
    }

    #[cfg(any(test, feature = "testing"))]
    fn lift_entries(parameters: &K::Parameters<f64>) -> Vec<(String, f64)> {
        let mut entries = Vec::new();
        let _ = <K::Parameters<f64> as crate::Lift<f64>>::lift(
            parameters,
            &crate::ParameterPath::Root,
            &mut |path, value| {
                entries.push((path.to_string(), value));
                value
            },
        );
        entries
    }
}

/// Every `(path, value)` pair that [`crate::Lift`] visits on `parameters`, an instance of `K`.
#[cfg(any(test, feature = "testing"))]
fn lift_entries_of<K: Kind>(parameters: &Erased) -> Vec<(String, f64)> {
    parameters
        .downcast_ref::<K::Parameters>()
        .map_or_else(Vec::new, K::lift_entries)
}

/// An instance prepared and laid out, before it joins its group: its shape and its
/// numbers, typed.
trait Made: Send {
    /// The shape, as the typed half of a group of it.
    fn shaped(&self) -> &dyn Shaped;

    /// Whether the instance is linear, as [`Kernel::linear`] says.
    fn linear(&self) -> bool;

    /// The instance as [`Any`], for its group's run to take its numbers.
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
}

/// An instance of `K` prepared and laid out.
struct Fresh<K: Kind> {
    shape: Shape<K>,
    prepared: K::Prepared,
}

impl<K: Kind> Made for Fresh<K> {
    fn shaped(&self) -> &dyn Shaped {
        &self.shape
    }

    fn linear(&self) -> bool {
        K::linear(&self.shape.0, &self.prepared)
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

/// `parameters`, an instance of `K`, prepared at `temperature` and laid out in `layout`.
fn made<K: Kind>(
    parameters: &Erased,
    temperature: f64,
    layout: &mut Layout,
) -> Result<Box<dyn Made>, ParameterError> {
    let parameters = parameters
        .downcast_ref::<K::Parameters>()
        .ok_or_else(|| ParameterError(format!("the parameters of no {}", K::NAME)))?;
    let prepared = K::prepare(parameters, temperature)?;
    let shape = Shape::<K>(K::layout(parameters, &prepared, layout)?);
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
struct Shape<K: Kind>(K::Shape);

impl<K: Kind> fmt::Debug for Shape<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple(K::NAME).field(&self.0).finish()
    }
}

impl<K: Kind> Shaped for Shape<K> {
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
            terms: Terms::default(),
        })
    }
}

/// The instances of one kernel and one shape, linear or not, as a circuit's structure fixes
/// them.
#[derive(Debug)]
struct Group {
    shaped: Box<dyn Shaped>,
    layout: Layout,
    /// Whether its instances are linear, as [`Kernel::linear`] says: evaluated once a run,
    /// for their entries, which the base sums, and their residuals and charges each
    /// evaluation from them.
    linear: bool,
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
    /// Its instances' local unknowns, then their local entries, that contribute to a long
    /// value, whose copies lie beyond the blocks.
    copied: [Copied; 2],
}

/// The locals of a group's instances, its local unknowns or its local entries, that
/// contribute to a long value of the circuit's equations, one of more than [`CHUNK`]
/// contributions, and where their copies lie among the slab's.
#[derive(Debug, Default)]
struct Copied {
    /// Where each instance's begin among the group's copies, then where the last's end.
    starts: Vec<usize>,
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
    fn len(&self) -> usize {
        self.locals.len()
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
    fn copy<const N: usize>(
        &self,
        (member, base): (usize, usize),
        block: [&[f64]; N],
        copies: &mut [&mut [f64]; N],
    ) {
        let (start, end) = (self.starts[member], self.starts[member + 1]);
        let locals = &self.locals[start..end];
        for (copy, plane) in copies.iter_mut().zip(block) {
            for (into, &local) in copy[start - base..end - base].iter_mut().zip(locals) {
                *into = plane[local];
            }
        }
    }
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

/// The groups of a circuit's devices, each a kernel's instance: by kernel and shape, in the
/// order their first instances appear.
#[derive(Debug, Default)]
pub struct Groups {
    kernels: Vec<Group>,
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
    sizes: (usize, usize),
    /// The copies beyond them of those that contribute to a long value.
    copies: (usize, usize),
    /// The stored entries the linear instances' entries add to, in increasing order, each
    /// the target of one value of the base, which lies beyond the copies among the slab's
    /// places for local entries, in their order.
    base: Vec<u32>,
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
        let temperature = instance.own.temperature(temperature)?;
        let made = (instance.made)(instance.parameters, temperature, layout)?;
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
        let Some(group) = self
            .place(device)
            .and_then(|(group, _)| self.kernels.get(group))
        else {
            return Ok(false);
        };
        let mut layout = Layout::new(group.layout.terminals());
        let temperature = instance.own.temperature(temperature)?;
        let made = (instance.made)(instance.parameters, temperature, &mut layout)?;
        Ok(group.shaped.kernel() == instance.kernel
            && group.shaped.same(made.shaped())
            && group.linear == made.linear())
    }

    /// Whether the instance the device `device` is, if it is a kernel's, is linear.
    #[must_use]
    pub fn is_linear(&self, device: usize) -> bool {
        self.place(device)
            .and_then(|(group, _)| self.kernels.get(group))
            .is_some_and(|group| group.linear)
    }

    /// The instance at the device `device`, if it is a kernel's: its group and its index
    /// among the group's instances.
    fn place(&self, device: usize) -> Option<(usize, usize)> {
        self.places.get(device).copied()
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

/// The index among the circuit's unknowns of `node`, or [`UNMAPPED`] for the reference.
fn mapped(node: Node) -> u32 {
    node.and_then(|unknown| u32::try_from(unknown.index()).ok())
        .unwrap_or(UNMAPPED)
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
    /// Whether an instance of each group limited its voltages, as the group's last
    /// evaluation in parallel found.
    limited: Vec<bool>,
    /// Whether the linear instances' blocks hold their entries, at their numbers now, and
    /// the slab the base they sum to: false until the first evaluation, and again once an
    /// instance is prepared afresh or the slab forgotten.
    stamped: bool,
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
/// [`UNMAPPED`] for one on the reference.
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
        let (group, member) = groups.place(device)?;
        let (of, run) = (groups.kernels.get(group)?, self.groups.get(group)?);
        let sizes = (of.unknowns(), of.stride());
        let nodes = of.map_of(member)?.to_vec();
        Some(Twin {
            evaluated: run.twin((instance, temperature), seeds, sizes)?,
            slots: of.slots_of(member)?.to_vec(),
            nodes,
            seeded: seeds.seeded.clone(),
            seeds: seeds.count(),
        })
    }

    /// Twins of the instance the device `device` is, at `temperature`, whose seeds move
    /// each of the circuit's unknowns its local unknowns map to, once, up to
    /// [`Seeds::MOST`] unknowns to a twin, its parameters constant.
    #[must_use]
    pub fn unknown_twins(
        &self,
        groups: &Groups,
        device: usize,
        instance: (&Instance<'_>, f64),
    ) -> Vec<Twin> {
        let Some((group, member)) = groups.place(device) else {
            return Vec::new();
        };
        let Some(nodes) = groups.kernels.get(group).and_then(|of| of.map_of(member)) else {
            return Vec::new();
        };
        let mut unique = Vec::with_capacity(nodes.len());
        for &unknown in nodes {
            if unknown != UNMAPPED && !unique.contains(&unknown) {
                unique.push(unknown);
            }
        }
        unique
            .chunks(Seeds::MOST)
            .filter_map(|chunk| {
                let seeds = Seeds::unknowns(chunk.to_vec())?;
                self.twin(groups, device, instance, &seeds)
            })
            .collect()
    }

    /// The instance the device `device` is, if it is a kernel's, and its group's run.
    fn run_of(&self, groups: &Groups, device: usize) -> Option<(&dyn GroupRun, usize)> {
        let (group, member) = groups.place(device)?;
        Some((self.groups.get(group)?.as_ref(), member))
    }

    /// The level of the instance the device `device` is, if it is an independent source,
    /// which analyses excite: its waveform's at time zero.
    #[must_use]
    pub fn level(&self, groups: &Groups, device: usize) -> Option<f64> {
        let (run, member) = self.run_of(groups, device)?;
        run.level(member)
    }

    /// Adds the derivative of each of the circuit's residuals with respect to the level of
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
    /// to its level, as an independent source's, one value per local unknown: zero for
    /// each, for an instance that is no independent source.
    #[must_use]
    pub fn excitation(&self, groups: &Groups, device: usize) -> Option<Vec<f64>> {
        let (group, _) = groups.place(device)?;
        let (of, run) = (groups.kernels.get(group)?, self.groups.get(group)?);
        Some(run.excitation(of))
    }

    /// The block of the instance the device `device` is in the slab, as the last evaluation
    /// wrote it.
    #[must_use]
    pub fn block<'a>(&'a self, groups: &'a Groups, device: usize) -> Option<Block<'a>> {
        let (group, member) = groups.place(device)?;
        let of = groups.kernels.get(group)?;
        let (unknowns, stride) = (of.unknowns(), of.stride());
        let rows = of.starts.0 + member * unknowns..of.starts.0 + (member + 1) * unknowns;
        let start = of.starts.1 + member * stride;
        let stored = start..start + of.layout.entries().len();
        let [residual, scale, charge] = &self.slab.rows;
        let [conductance, capacitance] = &self.slab.entries;
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
        let (group, member) = groups.place(device)?;
        self.groups
            .get(group)?
            .driven(member, groups.kernels.get(group)?)
    }

    /// How far the input of the instance the device `device` is, if it is a logic input's
    /// detector, lies above its threshold at `input`, and the voltage of its supply above
    /// its ground there, in volts.
    #[must_use]
    pub fn sense(&self, groups: &Groups, device: usize, input: &Input<'_>) -> Option<(f64, f64)> {
        let (group, member) = groups.place(device)?;
        self.groups
            .get(group)?
            .sense(member, groups.kernels.get(group)?, input)
    }

    /// Appends the times in `[start, stop]` where an instance is not smooth to `times`, so
    /// a transient analysis steps onto them exactly.
    pub fn breakpoints(&self, span: (f64, f64), times: &mut Vec<f64>) {
        for run in &self.groups {
            run.breakpoints(span, times);
        }
    }

    /// The device indices of the instances of `groups` whose equations explicitly depend
    /// on time, as [`Kernel::varies_in_time`] states it, in ascending order.
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
    /// past, as [`Kernel::delayed`] states it, placed among the circuit's equations, in
    /// the order of the groups and their instances: none in a circuit without a delay
    /// line. A part whose row or column is the reference's is not among them.
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

    /// Writes into the block of each linear instance of `groups` its entries, as its kernel
    /// evaluates them at [`Input::ZERO`], as at any state, and sums them into the base, each
    /// value from zero, its sources one after another in their order.
    fn stamp(&mut self, groups: &Groups) {
        let [residual, scale, charge] = &mut self.slab.rows;
        let [conductance, capacitance] = &mut self.slab.entries;
        let (mut rows, mut entries) = (
            [residual.as_mut_slice(), scale, charge],
            [conductance.as_mut_slice(), capacitance],
        );
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
        let [residual, scale, charge] = &mut self.slab.rows;
        let [conductance, capacitance] = &mut self.slab.entries;
        let (mut rows, mut entries) = (
            [residual.as_mut_slice(), scale, charge],
            [conductance.as_mut_slice(), capacitance],
        );
        // The copies lie beyond every group's blocks, and the base beyond the copies.
        let (mut row_copies, mut entry_copies) = (
            beyond(&mut rows, groups.sizes.0),
            beyond(&mut entries, groups.sizes.1),
        );
        let [conducted, stored] = beyond(&mut entry_copies, groups.copies.1);
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
            add_base(&groups.base, [conducted, stored], into);
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
        let [residual, scale, charge] = &self.slab.rows;
        let [conductance, capacitance] = &self.slab.entries;
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
                    [residual, scale, charge],
                    [sums, largest, held],
                    [Fold::Add, Fold::Raise, Fold::Add],
                );
            },
            || {
                slots.fold_in_parallel(
                    [conductance, capacitance],
                    [conducted, stored],
                    [Fold::Add, Fold::Add],
                );
            },
        );
        self.limited.iter().any(|&limited| limited)
    }
}

/// Adds the base's values, `[conductances, capacitances]`, to their stored entries of
/// `into`, `targets`, in their order.
fn add_base(targets: &[u32], [conducted, stored]: [&mut [f64]; 2], into: &mut Contributions) {
    let values = targets.iter().zip(conducted.iter().zip(stored.iter()));
    for (&target, (&conductance, &capacitance)) in values {
        let target = target as usize;
        if let (Some(sum), Some(held)) = (
            into.conductance.get_mut(target),
            into.capacitance.get_mut(target),
        ) {
            *sum += conductance;
            *held += capacitance;
        }
    }
}

/// The values of each of `planes` from `at` on, which keep those before it; none of any
/// when one holds fewer.
fn beyond<'a, const N: usize>(planes: &mut [&'a mut [f64]; N], at: usize) -> [&'a mut [f64]; N] {
    if planes.iter().any(|plane| plane.len() < at) {
        return [(); N].map(|()| <&mut [f64]>::default());
    }
    planes.each_mut().map(|plane| {
        let (before, after) = std::mem::take(plane).split_at_mut(at);
        *plane = before;
        after
    })
}

/// The blocks of `group`'s instances, the first values of the slab's `planes`, which keep
/// the rest, and its copies, the first of `copies`, likewise; `None` when one holds too
/// few. A group that copies nothing takes no copies, so that a circuit with no long value
/// pays nothing for them.
#[inline]
fn blocks_of<'a>(
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

/// The first `count` values of each of `planes`, which keep the rest, or `None` when one
/// holds fewer.
#[inline]
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

    /// The level of the instance `member`, if it is an independent source.
    fn level(&self, member: usize) -> Option<f64>;

    /// The derivative of each residual of an instance of `group`, as an independent source,
    /// with respect to its level, one value per local unknown.
    fn excitation(&self, group: &Group) -> Vec<f64>;

    /// Starts the instance `member`, if it is a logic driver, on the ramp `ramp`, and
    /// returns when it settles.
    fn drive(&mut self, member: usize, ramp: (f64, LogicLevel, f64)) -> Option<f64>;

    /// The circuit's unknown of the current through the instance `member` of `group`, if
    /// it is a logic driver.
    fn driven(&self, member: usize, group: &Group) -> Option<UnknownIndex>;

    /// How far the input of the instance `member` of `group`, if it is a logic input's
    /// detector, lies above its threshold at `input`, and its supply's swing there.
    fn sense(&self, member: usize, group: &Group, input: &Input<'_>) -> Option<(f64, f64)>;

    /// Appends the times in `span` where an instance is not smooth to `times`.
    fn breakpoints(&self, span: (f64, f64), times: &mut Vec<f64>);

    /// Appends the device index of each instance of `group` whose equations explicitly
    /// depend on time to `found`.
    fn time_varying(&self, group: &Group, found: &mut Vec<usize>);

    /// Appends each part of the conductance of an instance of `group` that reads the
    /// past to `found`, placed among the circuit's equations.
    fn delayed(&self, group: &Group, found: &mut Vec<DelayedEntry>);

    /// The earliest fraction of the way from `before` to `after` at which a choice of an
    /// instance of `group` changes, as [`Choice::first_jump`] finds it, its choices found
    /// in `found`.
    fn first_jump(
        &mut self,
        group: &Group,
        span: (&Input<'_>, &Input<'_>),
        found: &mut [Vec<Choice>; 2],
    ) -> Option<f64>;

    /// Evaluates the instance `member` of the group of `of` alone at `input`, into `block`,
    /// which it zeroes first: a linear instance's entries by its kernel at [`Input::ZERO`]
    /// and its rows from them when the second of `of` says so, and else by its kernel.
    /// Returns whether it limited its voltages.
    fn evaluate_alone(
        &mut self,
        member: usize,
        of: (&Group, bool),
        input: &Input<'_>,
        block: Blocks<'_>,
    ) -> bool;

    /// The twin of an instance of the group, the `instance` of its device, at the
    /// temperature `temperature` of its surroundings, differentiated along `seeds`, with
    /// room for the group's `sizes`, its local unknowns and its local entries; `None` when
    /// its kernel reads no real number at a seed's path.
    fn twin(
        &self,
        instance: (&Instance<'_>, f64),
        seeds: &Seeds,
        sizes: (usize, usize),
    ) -> Option<Box<dyn Twinned>>;

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

    /// Takes the numbers kept apart, each instance prepared again starting from its
    /// initial memory. Returns whether it took any.
    fn commit(&mut self) -> bool;

    /// Forgets every instance's memory, and a linear group's terms.
    fn forget(&mut self);

    /// Forgets the numbers kept apart.
    fn discard(&mut self);

    /// Writes into the block of each instance of `group`, which is linear, of `blocks`,
    /// its entries, as its kernel evaluates them at [`Input::ZERO`], zero for one not prepared.
    fn stamp(&mut self, group: &Group, blocks: Blocks<'_>);

    /// Evaluates each instance of `group` at `input` into its block of `blocks`, the
    /// first of the pair, copying its contributions to a long value to their places among
    /// the second, the group's copies: one after another, each block added to `into` as
    /// soon as it is written, or without `into`, in parallel, `per_task` instances a task.
    /// An instance of a linear group writes only the rows of its block, from its entries.
    /// Returns whether one limited its voltages.
    fn evaluate(
        &mut self,
        group: &Group,
        input: &Input<'_>,
        blocks: (Blocks<'_>, Blocks<'_>),
        into: Option<&mut Contributions>,
        per_task: usize,
    ) -> bool;
}

/// One run of a group of the kernel `K`.
struct KernelRun<K: Kind> {
    shape: K::Shape,
    prepared: Vec<Option<K::Prepared>>,
    memory: Vec<K::Memory>,
    /// The numbers prepared and not yet taken, by instance.
    staged: Vec<(usize, K::Prepared)>,
    /// The terms of a linear group's residuals and charges, as its last stamp found them.
    terms: Terms,
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

    fn level(&self, member: usize) -> Option<f64> {
        K::level(self.prepared.get(member)?.as_ref()?)
    }

    fn excitation(&self, group: &Group) -> Vec<f64> {
        let (unknowns, stride) = (group.unknowns(), group.stride());
        let mut rows = [
            vec![0.0; unknowns],
            vec![0.0; unknowns],
            vec![0.0; unknowns],
        ];
        let mut entries = [vec![0.0; stride], vec![0.0; stride]];
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
            5..=Seeds::MOST => KernelTwin::<K, { Seeds::MOST }>::made(made, seeds, sizes),
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

    fn stamp(
        &mut self,
        group: &Group,
        ([residual, scale, charge], [conductance, capacitance]): Blocks<'_>,
    ) {
        let (unknowns, stride) = (group.unknowns(), group.stride());
        if unknowns == 0 {
            return;
        }
        let blocks = residual
            .chunks_mut(unknowns)
            .zip(scale.chunks_mut(unknowns))
            .zip(charge.chunks_mut(unknowns))
            .zip(conductance.chunks_mut(stride))
            .zip(capacitance.chunks_mut(stride));
        let instances = self.prepared.iter().zip(group.nodes.chunks(unknowns));
        for (((((residual, scale), charge), conductance), capacitance), (prepared, map)) in
            blocks.zip(instances)
        {
            let mut stamp = Stamp::zeroed([residual, scale, charge], [conductance, capacitance]);
            if let Some(prepared) = prepared {
                stamp_entries::<K>(&self.shape, prepared, map, &mut stamp);
            }
        }
        self.terms = Terms::from_layout(&group.layout, stride, [conductance, capacitance]);
    }

    fn evaluate(
        &mut self,
        group: &Group,
        input: &Input<'_>,
        (([residual, scale, charge], [conductance, capacitance]), copies): (Blocks<'_>, Blocks<'_>),
        into: Option<&mut Contributions>,
        per_task: usize,
    ) -> bool {
        let sizes = (group.unknowns(), group.stride());
        let (unknowns, stride) = sizes;
        if unknowns == 0 {
            return false;
        }
        if group.linear {
            let blocks = ([residual, scale, charge], [&*conductance, &*capacitance]);
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
            let (rows, stored) = (unknowns * per_task, stride * per_task);
            let Some(copies) = task_copies(group, copies, per_task) else {
                return false;
            };
            return residual
                .par_chunks_mut(rows)
                .zip(scale.par_chunks_mut(rows))
                .zip(charge.par_chunks_mut(rows))
                .zip(conductance.par_chunks_mut(stored))
                .zip(capacitance.par_chunks_mut(stored))
                .zip(self.memory.par_chunks_mut(per_task))
                .zip(self.prepared.par_chunks(per_task))
                .zip(group.nodes.par_chunks(rows))
                .zip(copies)
                .enumerate()
                .map(|(task, chunk)| {
                    let (
                        (
                            (
                                (((((residual, scale), charge), conductance), capacitance), memory),
                                prepared,
                            ),
                            nodes,
                        ),
                        copies,
                    ) = chunk;
                    let blocks = (
                        [&mut *residual, &mut *scale, &mut *charge],
                        [&mut *conductance, &mut *capacitance],
                    );
                    let limited = evaluate_chunk::<K>(
                        shape,
                        sizes,
                        input,
                        (blocks, memory, prepared),
                        (nodes, &[]),
                        None,
                    );
                    let written = ([&*residual, scale, charge], [&*conductance, capacitance]);
                    copy_long(sizes, written, copies, (&group.copied, task * per_task));
                    limited
                })
                .reduce(|| false, |one, other| one | other);
        };
        let blocks = (
            [&mut *residual, &mut *scale, &mut *charge],
            [&mut *conductance, &mut *capacitance],
        );
        let instances = (blocks, self.memory.as_mut_slice(), self.prepared.as_slice());
        let limited = evaluate_chunk::<K>(
            shape,
            sizes,
            input,
            instances,
            (&group.nodes, &group.slots),
            Some(into),
        );
        let written = ([&*residual, scale, charge], [&*conductance, capacitance]);
        copy_long(sizes, written, copies, (&group.copied, 0));
        limited
    }
}

/// The directions a [`Twin`] differentiates along, its seeds: numbers of an instance's
/// parameters, each at its path among them, as [`jolty_ir::numbers`] names it, with its
/// rate along each seed, or the circuit's unknowns. A seed that moves one number at rate 1
/// differentiates against that number alone; one that moves several, along that tangent of
/// them.
#[derive(Debug, Clone, PartialEq)]
pub struct Seeds {
    /// Each number's path, and its rate along each seed.
    numbers: Vec<(String, Vec<f64>)>,
    count: usize,
    /// The circuit's unknowns the seeds move at rate one, in their order, when seeding unit
    /// unknowns.
    seeded: Vec<u32>,
}

impl Seeds {
    /// The most seeds one twin carries: twins are compiled for 1, 2, 4 and this many, a
    /// twin of another count taking the next of them up.
    pub const MOST: usize = 8;

    /// A seed for each of the numbers at `paths`, which moves that number alone; a path
    /// given more than once moves along each of its seeds.
    ///
    /// A twin accepts between `1` and [`Seeds::MOST`] seeds; empty or larger sets yield
    /// `None` when passed to [`Runs::twin`].
    #[must_use]
    pub fn units(paths: &[&str]) -> Self {
        let count = paths.len();
        let numbers = paths
            .iter()
            .enumerate()
            .map(|(seed, &path)| {
                let mut rates = vec![0.0; count];
                if let Some(rate) = rates.get_mut(seed) {
                    *rate = 1.0;
                }
                (path.to_owned(), rates)
            })
            .collect();
        Self::new(numbers, count).unwrap_or(Self {
            numbers: Vec::new(),
            count,
            seeded: Vec::new(),
        })
    }

    /// A seed for each of the circuit's `unknowns`, at most [`Seeds::MOST`], its parameters
    /// constant: the derivatives of what an instance adds with respect to those unknowns,
    /// the second derivatives of its equations among them.
    #[must_use]
    pub fn unknowns(unknowns: Vec<u32>) -> Option<Self> {
        let count = unknowns.len();
        (1..=Self::MOST).contains(&count).then_some(Self {
            numbers: Vec::new(),
            count,
            seeded: unknowns,
        })
    }

    /// `count` seeds, at most [`Seeds::MOST`], its parameters constant, for
    /// [`Twin::evaluate_along`] to move the circuit's unknowns along `count` tangent
    /// vectors.
    #[must_use]
    pub const fn tangents(count: usize) -> Option<Self> {
        if count >= 1 && count <= Self::MOST {
            Some(Self {
                numbers: Vec::new(),
                count,
                seeded: Vec::new(),
            })
        } else {
            None
        }
    }

    /// `count` seeds, between `1` and [`Seeds::MOST`], along which each number of
    /// `numbers`, at its path, moves at its rates, one per seed, a path given more than once
    /// moving at the sum of its rates, added in the order given; `None` when `count` is out
    /// of range or a number has another count of rates.
    #[must_use]
    pub fn new(numbers: Vec<(String, Vec<f64>)>, count: usize) -> Option<Self> {
        if !(1..=Self::MOST).contains(&count)
            || numbers.iter().any(|(_, rates)| rates.len() != count)
        {
            return None;
        }
        let mut place: BTreeMap<String, usize> = BTreeMap::new();
        let mut summed: Vec<(String, Vec<f64>)> = Vec::with_capacity(numbers.len());
        for (path, rates) in numbers {
            let at = *place.entry(path.clone()).or_insert_with(|| {
                summed.push((path, vec![0.0; count]));
                summed.len() - 1
            });
            for (total, rate) in summed[at].1.iter_mut().zip(rates) {
                *total += rate;
            }
        }
        Some(Self {
            numbers: summed,
            count,
            seeded: Vec::new(),
        })
    }

    /// How many seeds there are.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }

    /// Each number's path and its rates, padded with zeros to `SEEDS` seeds.
    fn padded<const SEEDS: usize>(&self) -> Vec<(&str, [f64; SEEDS])> {
        self.numbers
            .iter()
            .map(|(path, rates)| {
                let padded = std::array::from_fn(|seed| rates.get(seed).copied().unwrap_or(0.0));
                (path.as_str(), padded)
            })
            .collect()
    }
}

/// What an instance's twin adds to the circuit's equations, by kind.
#[derive(Debug, Clone, Copy)]
enum Contribution {
    Residual,
    Charge,
    Conductance,
    Capacitance,
}

/// An instance's twin, typed, apart from its group.
trait Twinned: fmt::Debug + Send + Sync {
    /// Evaluates the twin at `at` into its own block.
    fn evaluate(&mut self, at: &Local<'_>);

    /// The `contribution` it adds at the local unknown or local entry `index`, as last
    /// evaluated: its value, and its derivative along each seed.
    fn number(&self, contribution: Contribution, index: usize) -> Option<(f64, &[f64])>;
}

/// The twin of an instance of the kind `K`, its derivatives along `SEEDS` directions, its
/// shape, and its block: the residuals, scales and charges of its local unknowns and the
/// conductances and capacitances of its local entries, as last evaluated.
struct KernelTwin<K: Kind, const SEEDS: usize> {
    shape: K::Shape,
    twin: K::Twin<SEEDS>,
    rows: [Vec<Gradient<SEEDS>>; 3],
    entries: [Vec<Gradient<SEEDS>>; 2],
    /// How many of the `SEEDS` directions the twin was seeded along, the rest left zero.
    seeds: usize,
}

impl<K: Kind, const SEEDS: usize> KernelTwin<K, SEEDS> {
    /// The twin of an instance of `shape` with `parameters` at `temperature` along `seeds`,
    /// with room for `sizes`, its local unknowns and its local entries.
    fn made(
        (shape, parameters, temperature): (&K::Shape, &K::Parameters, f64),
        seeds: &Seeds,
        (unknowns, stride): (usize, usize),
    ) -> Option<Box<dyn Twinned>> {
        let twin = K::twin(shape, parameters, &seeds.padded::<SEEDS>(), temperature)?;
        let zero = |size| vec![Gradient::default(); size];
        Some(Box::new(Self {
            shape: shape.clone(),
            twin,
            rows: [zero(unknowns), zero(unknowns), zero(unknowns)],
            entries: [zero(stride), zero(stride)],
            seeds: seeds.count(),
        }))
    }
}

impl<K: Kind, const SEEDS: usize> fmt::Debug for KernelTwin<K, SEEDS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(K::NAME)
            .field("shape", &self.shape)
            .field("twin", &self.twin)
            .finish_non_exhaustive()
    }
}

impl<K: Kind, const SEEDS: usize> Twinned for KernelTwin<K, SEEDS> {
    fn evaluate(&mut self, at: &Local<'_>) {
        let [residual, scale, charge] = &mut self.rows;
        let [conductance, capacitance] = &mut self.entries;
        let mut stamp = Stamp::zeroed([residual, scale, charge], [conductance, capacitance]);
        K::evaluate_twin(&self.shape, &mut self.twin, at, &mut stamp);
    }

    fn number(&self, contribution: Contribution, index: usize) -> Option<(f64, &[f64])> {
        let [residual, _, charge] = &self.rows;
        let [conductance, capacitance] = &self.entries;
        let values = match contribution {
            Contribution::Residual => residual,
            Contribution::Charge => charge,
            Contribution::Conductance => conductance,
            Contribution::Capacitance => capacitance,
        };
        let number = values.get(index)?;
        Some((number.value, number.derivatives.get(..self.seeds)?))
    }
}

/// A device's instance computing with the derivatives of every number along some
/// directions of its parameters, its seeds: how the contributions it adds to the circuit's
/// equations move along each, which its own block holds, local to the instance, and the
/// map of its group places among the circuit's.
#[derive(Debug)]
pub struct Twin {
    evaluated: Box<dyn Twinned>,
    /// The circuit's unknown of each local unknown, or [`UNMAPPED`] for one on the
    /// reference.
    nodes: Vec<u32>,
    /// The stored position of each local entry, or [`UNMAPPED`] for none.
    slots: Vec<u32>,
    /// The circuit's unknown each seed moves, in their order, or [`UNMAPPED`] for a local
    /// unknown on the reference; none when the seeds are directions of the parameters.
    seeded: Vec<u32>,
    seeds: usize,
}

/// A contribution of a twin as last evaluated: its value, and its derivative along each of
/// the twin's seeds.
pub type TwinNumber<'a> = (f64, &'a [f64]);

impl Twin {
    /// How many seeds the twin differentiates along.
    #[must_use]
    pub const fn seeds(&self) -> usize {
        self.seeds
    }

    /// Evaluates the twin at `input`.
    pub fn evaluate(&mut self, input: &Input<'_>) {
        let local = input.local(&self.nodes).seeding(&self.seeded);
        self.evaluated.evaluate(&local);
    }

    /// Evaluates the twin at `input`, moving the circuit's unknowns along each of
    /// `tangents`, one seed per tangent.
    pub fn evaluate_along(&mut self, input: &Input<'_>, tangents: &[Vec<f64>]) {
        let local = input.local(&self.nodes).along(tangents);
        self.evaluated.evaluate(&local);
    }

    /// The circuit's unknown each seed moves, by seed, when the seeds are local unknowns:
    /// [`UNMAPPED`] for one on the reference, which moves nothing.
    #[must_use]
    pub fn seeded(&self) -> &[u32] {
        &self.seeded
    }

    /// Each residual the instance adds, by the circuit's unknown whose equation it enters,
    /// in the order of its local unknowns, those on the reference left out: `f`, and
    /// `∂f/∂p · s` along each seed `s`, as last evaluated.
    pub fn residual(&self) -> impl Iterator<Item = (usize, TwinNumber<'_>)> + '_ {
        self.by_unknown(Contribution::Residual)
    }

    /// Each charge the instance adds, as [`Twin::residual`] gives the residuals: `q`, and
    /// `∂q/∂p · s` along each seed, as last evaluated.
    pub fn charge(&self) -> impl Iterator<Item = (usize, TwinNumber<'_>)> + '_ {
        self.by_unknown(Contribution::Charge)
    }

    /// Each mapped matrix slot the twin stamps into, in the order of its local entries,
    /// those on the reference left out.
    pub fn slots(&self) -> impl Iterator<Item = usize> + '_ {
        self.slots
            .iter()
            .copied()
            .filter(|&slot| slot != UNMAPPED)
            .map(|slot| slot as usize)
    }

    /// Each conductance and capacitance the instance adds, by the stored position it adds
    /// to, in the order of its local entries, those on the reference left out: `G` and `C`,
    /// and their derivatives along each seed, as last evaluated.
    pub fn entries(&self) -> impl Iterator<Item = (usize, TwinNumber<'_>, TwinNumber<'_>)> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| **slot != UNMAPPED)
            .filter_map(|(entry, &slot)| {
                let conductance = self.evaluated.number(Contribution::Conductance, entry)?;
                let capacitance = self.evaluated.number(Contribution::Capacitance, entry)?;
                Some((slot as usize, conductance, capacitance))
            })
    }

    /// Each `contribution` at a local unknown, by the circuit's unknown of the local
    /// unknown, those on the reference left out.
    fn by_unknown(
        &self,
        contribution: Contribution,
    ) -> impl Iterator<Item = (usize, TwinNumber<'_>)> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| **node != UNMAPPED)
            .filter_map(move |(local, &node)| {
                Some((node as usize, self.evaluated.number(contribution, local)?))
            })
    }
}

/// The instances of a chunk of a group: their blocks, their memory and their prepared
/// numbers.
type Chunk<'a, K> = (
    Blocks<'a>,
    &'a mut [<K as Kind>::Memory],
    &'a [Option<<K as Kind>::Prepared>],
);

/// The copies of `group`'s instances, `copies`, split among its tasks of `per_task`
/// instances each, in their order; `None` when they hold fewer than the group copies.
fn task_copies<'a>(
    group: &Group,
    (mut rows, mut entries): Blocks<'a>,
    per_task: usize,
) -> Option<Vec<Blocks<'a>>> {
    let members = group.members.len();
    let [copied_rows, copied_entries] = &group.copied;
    (0..members.div_ceil(per_task))
        .map(|task| {
            let (first, end) = (task * per_task, members.min((task + 1) * per_task));
            let counted = |copied: &Copied| copied.starts[end] - copied.starts[first];
            Some((
                split(&mut rows, counted(copied_rows))?,
                split(&mut entries, counted(copied_entries))?,
            ))
        })
        .collect()
}

/// Evaluates each instance of a chunk of a group of `shape`, of `sizes` local unknowns and
/// places for local entries each, at `input`, into its block, which it zeroes first, and
/// adds each block to `into`, when given, as soon as it is written, through the circuit's
/// unknowns and stored positions of the instances' local ones, `maps`. Returns whether an
/// instance limited its voltages.
fn evaluate_chunk<K: Kind>(
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

/// Writes into `stamp` what a linear instance of `shape` with the numbers `prepared`, whose
/// local unknowns are the circuit's of `map`, contributes at [`Input::ZERO`], from its
/// initial memory: its entries, which are the same at every state.
fn stamp_entries<K: Kind>(
    shape: &K::Shape,
    prepared: &K::Prepared,
    map: &[u32],
    stamp: &mut Stamp<'_>,
) {
    let mut memory = K::memory(shape, prepared);
    K::evaluate(shape, prepared, &mut memory, &Input::ZERO.local(map), stamp);
}

/// A linear group's blocks: the rows of its instances' local unknowns, which their
/// evaluations write, and the entries of their local entries, which they read.
type LinearBlocks<'a> = ([&'a mut [f64]; 3], [&'a [f64]; 2]);

/// Writes the rows of each block of `group`, which is linear, of `blocks`, the first of the
/// pair, from its entries, at `input`, copying its contributions to a long value to their
/// places among the second, the group's copies: one after another, each block's rows added
/// to `into` as soon as they are written, or without `into`, in parallel, `per_task`
/// instances a task.
fn evaluate_linear(
    (group, terms): (&Group, &Terms),
    input: &Input<'_>,
    (([residual, scale, charge], [conductance, capacitance]), copies): (
        LinearBlocks<'_>,
        Blocks<'_>,
    ),
    into: Option<&mut Contributions>,
    per_task: usize,
) {
    let sizes = (group.unknowns(), group.stride());
    let (unknowns, stride) = sizes;
    let Some(into) = into else {
        let (rows, stored) = (unknowns * per_task, stride * per_task);
        let Some(copies) = task_copies(group, copies, per_task) else {
            return;
        };
        residual
            .par_chunks_mut(rows)
            .zip(scale.par_chunks_mut(rows))
            .zip(charge.par_chunks_mut(rows))
            .zip(conductance.par_chunks(stored))
            .zip(capacitance.par_chunks(stored))
            .zip(group.nodes.par_chunks(rows))
            .zip(copies)
            .enumerate()
            .for_each(|(task, chunk)| {
                let ((((((residual, scale), charge), conductance), capacitance), nodes), copies) =
                    chunk;
                let blocks = (
                    [&mut *residual, &mut *scale, &mut *charge],
                    [conductance, capacitance],
                );
                linear_chunk(terms, sizes, input, blocks, nodes, None);
                let written = ([&*residual, scale, charge], [conductance, capacitance]);
                copy_long(sizes, written, copies, (&group.copied, task * per_task));
            });
        return;
    };
    let blocks = (
        [&mut *residual, &mut *scale, &mut *charge],
        [conductance, capacitance],
    );
    linear_chunk(terms, sizes, input, blocks, &group.nodes, Some(into));
    let written = ([&*residual, scale, charge], [conductance, capacitance]);
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
    ([residual, scale, charge], [conductance, capacitance]): LinearBlocks<'_>,
    nodes: &[u32],
    mut into: Option<&mut Contributions>,
) {
    let blocks = residual
        .chunks_mut(unknowns)
        .zip(scale.chunks_mut(unknowns))
        .zip(charge.chunks_mut(unknowns))
        .zip(conductance.chunks(stride))
        .zip(capacitance.chunks(stride));
    for (((((residual, scale), charge), conductance), capacitance), map) in
        blocks.zip(nodes.chunks(unknowns))
    {
        let at = input.local(map);
        let rows = [residual, scale, charge];
        terms.write([conductance, capacitance], &at, rows, into.as_deref_mut());
    }
}

/// The terms of the residuals and the charges of a linear group's instances: for its
/// conductances, then its capacitances, each local unknown's row with an entry that is not
/// zero in some instance, and those of its entries, each with its column, in the order of
/// their local indices. Every other term of the group's is zero at every state, as is the
/// row of one that has none: leaving it out of a sum, which starts from zero and so is
/// never `-0`, changes no bit of it.
#[derive(Debug, Clone, Default)]
struct Terms {
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
    fn from_layout(layout: &Layout, stride: usize, planes: [&[f64]; 2]) -> Self {
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
    /// reference, [`UNMAPPED`], go nowhere.
    #[inline]
    fn write(
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
fn copy_long(
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
    ([residual, scale, charge], [conductance, capacitance]): ([&[f64]; 3], [&[f64]; 2]),
    (mut row_copies, mut entry_copies): Blocks<'_>,
    ([copied_rows, copied_entries], first): (&[Copied; 2], usize),
) {
    let bases = (copied_rows.starts[first], copied_entries.starts[first]);
    let blocks = residual
        .chunks(unknowns)
        .zip(scale.chunks(unknowns))
        .zip(charge.chunks(unknowns))
        .zip(conductance.chunks(stride))
        .zip(capacitance.chunks(stride));
    for (member, ((((residual, scale), charge), conductance), capacitance)) in (first..).zip(blocks)
    {
        copied_rows.copy(
            (member, bases.0),
            [residual, scale, charge],
            &mut row_copies,
        );
        let entries = [conductance, capacitance];
        copied_entries.copy((member, bases.1), entries, &mut entry_copies);
    }
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

#[cfg(test)]
mod tests {
    use super::Seeds;

    #[test]
    fn a_path_given_twice_moves_at_the_sum_of_its_rates() {
        let numbers = vec![
            ("w".to_owned(), vec![1.0, -0.5]),
            ("l".to_owned(), vec![0.0, 1.0]),
            ("w".to_owned(), vec![0.25, 2.0]),
        ];
        let seeds = Seeds::new(numbers, 2).expect("two seeds");
        assert_eq!(
            seeds.padded::<2>(),
            vec![("w", [1.25, 1.5]), ("l", [0.0, 1.0])]
        );
        let units = Seeds::units(&["w", "w"]);
        assert_eq!(units.padded::<2>(), vec![("w", [1.0, 1.0])]);
    }
}
