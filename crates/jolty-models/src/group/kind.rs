//! Kernel instances, the internal [`Kind`] abstraction unifying plain and heated kernels,
//! and type-erased shapes.

use std::any::{Any, TypeId};
use std::fmt;

use super::kernel_run::{GroupRun, KernelRun};
use crate::lift::lift;
use crate::surroundings::Own;
use crate::{
    Binding, Choice, Delayed, Gradient, Kernel, Layout, Local, LogicLevel, NoiseSource,
    ParameterError, Real, Regions, Stamp,
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
    pub(super) kernel: TypeId,
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
    pub(super) own: Own,
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
        self.make(temperature, &mut layout)?;
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

    /// Prepares the instance at the surroundings' `temperature` and claims its layout in
    /// `layout`.
    pub(super) fn make(
        &self,
        temperature: f64,
        layout: &mut Layout,
    ) -> Result<Box<dyn Made>, ParameterError> {
        let temperature = self.own.temperature(temperature)?;
        (self.made)(self.parameters, temperature, layout)
    }

    /// The parameters, if the instance is one of `K`.
    pub(super) fn parameters<K: Kind>(&self) -> Option<&'a K::Parameters> {
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

/// Generates 1-to-1 [`Kind`] delegations forwarding directly to a [`Kernel`].
macro_rules! forward_kernel {
    ($kernel:ident; $(fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)?;)*) => {
        $(forward_kernel!(@one $kernel; fn $name($($arg: $ty),*) $(-> $ret)?);)*
    };
    (@one $kernel:ident; fn $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty) => {
        fn $name($($arg: $ty),*) -> $ret { $kernel::$name($($arg),*) }
    };
    (@one $kernel:ident; fn $name:ident($($arg:ident: $ty:ty),*)) => {
        fn $name($($arg: $ty),*) { $kernel::$name($($arg),*); }
    };
}

impl<K: Kernel> Kind for K {
    const NAME: &'static str = K::NAME;
    type Parameters = K::Parameters<f64>;
    type Shape = K::Shape;
    type Prepared = K::Prepared<f64>;
    type Memory = K::Memory;

    forward_kernel! {
        K;
        fn prepare(parameters: &Self::Parameters, temperature: f64)
            -> Result<Self::Prepared, ParameterError>;
        fn layout(parameters: &Self::Parameters, prepared: &Self::Prepared, layout: &mut Layout)
            -> Result<Self::Shape, ParameterError>;
        fn memory(shape: &Self::Shape, prepared: &Self::Prepared) -> Self::Memory;
        fn linear(shape: &Self::Shape, prepared: &Self::Prepared) -> bool;
        fn evaluate(shape: &Self::Shape, prepared: &Self::Prepared, memory: &mut Self::Memory,
            at: &Local<'_>, stamp: &mut Stamp<'_>);
        fn variable_names() -> &'static [&'static str];
        fn regions() -> Regions;
        fn variables(memory: &Self::Memory, values: &mut [f64]);
        fn breakpoints(prepared: &Self::Prepared, span: (f64, f64), times: &mut Vec<f64>);
        fn varies_in_time(shape: &Self::Shape, prepared: &Self::Prepared) -> bool;
        fn delayed(shape: &Self::Shape, prepared: &Self::Prepared, found: &mut Vec<Delayed>);
        fn choices(shape: &Self::Shape, prepared: &Self::Prepared, memory: &mut Self::Memory,
            at: &Local<'_>, found: &mut Vec<Choice>);
        fn level(prepared: &Self::Prepared) -> Option<f64>;
        fn excite(shape: &Self::Shape, stamp: &mut Stamp<'_>);
        fn drive(memory: &mut Self::Memory, ramp: (f64, LogicLevel, f64)) -> Option<f64>;
        fn driven(shape: &Self::Shape) -> Option<usize>;
        fn sense(shape: &Self::Shape, prepared: &Self::Prepared, at: &Local<'_>)
            -> Option<(f64, f64)>;
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
pub(super) trait Made: Send {
    /// The shape, as the typed half of a group of it.
    fn shaped(&self) -> &dyn Shaped;

    /// Whether the instance is linear, as [`Kernel::linear`] says.
    fn linear(&self) -> bool;

    /// The instance as [`Any`], for its group's run to take its numbers.
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
}

/// An instance of `K` prepared and laid out.
pub(super) struct Fresh<K: Kind> {
    shape: Shape<K>,
    pub(super) prepared: K::Prepared,
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
pub(super) trait Shaped: fmt::Debug + Send + Sync {
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
        Box::new(KernelRun::<K>::new(self.0.clone(), members))
    }
}
