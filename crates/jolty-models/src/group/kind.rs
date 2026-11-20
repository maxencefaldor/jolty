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

/// Type-erased parsed parameters held by an [`Instance`].
type Erased = dyn Any + Send + Sync;

/// Function pointer that prepares an instance at a temperature in kelvins and claims its
/// layout, returning the typed shape and prepared parameters.
type MadeBy = fn(&Erased, f64, &mut Layout) -> Result<Box<dyn Made>, ParameterError>;

/// Borrowed reference to a device instance's typed parameters and temperature override, used
/// to group matching instances and prepare them for simulation runs.
#[derive(Clone, Copy)]
pub struct Instance<'a> {
    pub(super) kernel: TypeId,
    name: &'static str,
    parameters: &'a Erased,
    /// Callback that prepares the instance and claims its layout.
    made: MadeBy,
    /// Callback returning internal variable names computed by the kernel.
    variables: fn() -> &'static [&'static str],
    /// Callback returning operating regions distinguished by the kernel.
    regions: fn() -> Regions,
    /// Callback listing every `(path, value)` visited by [`crate::Lift`].
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
    /// Constructs an instance handle for kernel `K` with `parameters` at ambient temperature.
    #[must_use]
    pub fn from_parameters<K: Kernel>(parameters: &'a K::Parameters<f64>) -> Self {
        Self::from_kind::<K>(parameters)
    }

    /// Constructs an instance handle for kernel kind `K` with `parameters` at ambient
    /// temperature.
    pub(crate) fn from_kind<K: Kind>(parameters: &'a K::Parameters) -> Self {
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

    /// Returns a copy of `self` carrying per-instance temperature override `own`.
    pub(crate) const fn at_own(self, own: Own) -> Self {
        Self { own, ..self }
    }

    /// Returns the internal variable names reported by [`Kernel::variable_names`].
    #[must_use]
    pub fn variable_names(&self) -> &'static [&'static str] {
        (self.variables)()
    }

    /// Returns the operating regions reported by [`Kernel::regions`].
    #[must_use]
    pub fn regions(&self) -> Regions {
        (self.regions)()
    }

    /// Claims the internal unknowns and matrix entries required by the instance at ambient
    /// `temperature` through `binding`.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when the instance parameters are out of range.
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

    /// Prepares the instance at ambient `temperature` and populates its `layout`.
    pub(super) fn make(
        &self,
        temperature: f64,
        layout: &mut Layout,
    ) -> Result<Box<dyn Made>, ParameterError> {
        let temperature = self.own.temperature(temperature)?;
        (self.made)(self.parameters, temperature, layout)
    }

    /// Downcasts the erased parameters to `K::Parameters` when the kernel matches `K`.
    pub(super) fn parameters<K: Kind>(&self) -> Option<&'a K::Parameters> {
        self.parameters.downcast_ref()
    }

    /// Collects every `(path, value)` pair visited by [`crate::Lift`] on the instance
    /// parameters in traversal order.
    #[cfg(any(test, feature = "testing"))]
    #[must_use]
    pub fn lift_entries(&self) -> Vec<(String, f64)> {
        (self.lift_entries)(self.parameters)
    }

    /// Returns per-instance temperature keys handled by [`crate::Surrounded`] outside
    /// [`crate::Lift`].
    #[cfg(any(test, feature = "testing"))]
    pub(crate) fn own_temperature_keys(&self) -> Vec<&'static str> {
        self.own.present()
    }
}

/// Group-facing kernel abstraction unifying plain [`Kernel`] types and self-heating
/// [`crate::kernel::heat::HeatedKernel`] wrappers using plain `f64` evaluation buffers.
pub(crate) trait Kind: Send + Sync + 'static {
    /// Model name registered in the library.
    const NAME: &'static str;
    /// Deserialized parameter struct type.
    type Parameters: Send + Sync + 'static;
    /// Layout-derived shape type, matching [`Kernel::Shape`].
    type Shape: Clone + fmt::Debug + PartialEq + Send + Sync + 'static;
    /// Precomputed temperature-dependent parameters, matching [`Kernel::Prepared`].
    type Prepared: Clone + fmt::Debug + Send + Sync;
    /// Per-instance evaluation state memory, matching [`Kernel::Memory`].
    type Memory: Clone + fmt::Debug + Default + Send + Sync;

    /// Prepares parameters at `temperature` in kelvins, delegating to [`Kernel::prepare`].
    fn prepare(
        parameters: &Self::Parameters,
        temperature: f64,
    ) -> Result<Self::Prepared, ParameterError>;

    /// Claims internal unknowns and matrix entries, delegating to [`Kernel::layout`].
    fn layout(
        parameters: &Self::Parameters,
        prepared: &Self::Prepared,
        layout: &mut Layout,
    ) -> Result<Self::Shape, ParameterError>;

    /// Initializes instance memory, delegating to [`Kernel::memory`].
    fn memory(shape: &Self::Shape, prepared: &Self::Prepared) -> Self::Memory;

    /// Reports whether the instance is linear, delegating to [`Kernel::linear`].
    fn linear(shape: &Self::Shape, prepared: &Self::Prepared) -> bool;

    /// Evaluates device equations into `stamp`, delegating to [`Kernel::evaluate`].
    fn evaluate(
        shape: &Self::Shape,
        prepared: &Self::Prepared,
        memory: &mut Self::Memory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_>,
    );

    /// Appends operating-point noise sources using `memory`, delegating to [`Kernel::noise`].
    fn noise(
        shape: &Self::Shape,
        prepared: &Self::Prepared,
        memory: &Self::Memory,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    );

    /// Returns internal variable names, delegating to [`Kernel::variable_names`].
    fn variable_names() -> &'static [&'static str];

    /// Returns operating region metadata, delegating to [`Kernel::regions`].
    fn regions() -> Regions;

    /// Copies internal variable values from `memory`, delegating to [`Kernel::variables`].
    fn variables(memory: &Self::Memory, values: &mut [f64]);

    /// Appends waveform breakpoints within `span`, delegating to [`Kernel::breakpoints`].
    fn breakpoints(prepared: &Self::Prepared, span: (f64, f64), times: &mut Vec<f64>);

    /// Reports explicit time dependence, delegating to [`Kernel::varies_in_time`].
    fn varies_in_time(shape: &Self::Shape, prepared: &Self::Prepared) -> bool;

    /// Appends delayed conductance entries, delegating to [`Kernel::delayed`].
    fn delayed(shape: &Self::Shape, prepared: &Self::Prepared, found: &mut Vec<Delayed>);

    /// Records active branch choices at `at`, delegating to [`Kernel::choices`].
    fn choices(
        shape: &Self::Shape,
        prepared: &Self::Prepared,
        memory: &mut Self::Memory,
        at: &Local<'_>,
        found: &mut Vec<Choice>,
    );

    /// Returns the independent-source DC value, delegating to [`Kernel::dc_value`].
    fn dc_value(prepared: &Self::Prepared) -> Option<f64>;

    /// Stamps unit source excitation, delegating to [`Kernel::excite`].
    fn excite(shape: &Self::Shape, stamp: &mut Stamp<'_>);

    /// Starts a logic-driver ramp, delegating to [`Kernel::drive`].
    fn drive(memory: &mut Self::Memory, ramp: (f64, LogicLevel, f64)) -> Option<f64>;

    /// Returns the logic-driver current unknown index, delegating to [`Kernel::driven`].
    fn driven(shape: &Self::Shape) -> Option<usize>;

    /// Evaluates logic-detector threshold margin and supply swing, delegating to
    /// [`Kernel::sense`].
    fn sense(shape: &Self::Shape, prepared: &Self::Prepared, at: &Local<'_>) -> Option<(f64, f64)>;

    /// Differentiable prepared state and memory carrying derivatives along `SEEDS` parameter
    /// directions.
    type Twin<const SEEDS: usize>: fmt::Debug + Send + Sync;

    /// Builds a forward-mode sensitivity twin for `shape` and `parameters` at `temperature`
    /// (in kelvins) seeded along `seeds` (paths as formatted by [`jolty_ir::numbers`]).
    ///
    /// Returns `None` when a seed path addresses a non-real choice parameter or validation
    /// fails.
    fn twin<const SEEDS: usize>(
        shape: &Self::Shape,
        parameters: &Self::Parameters,
        seeds: &[(&str, [f64; SEEDS])],
        temperature: f64,
    ) -> Option<Self::Twin<SEEDS>>;

    /// Evaluates sensitivity `twin` of `shape` at `at`, updating twin memory and writing
    /// gradient contributions into `stamp`.
    fn evaluate_twin<const SEEDS: usize>(
        shape: &Self::Shape,
        twin: &mut Self::Twin<SEEDS>,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, Gradient<SEEDS>>,
    );

    /// Appends differentiable operating-point noise sources for sensitivity `twin` at `at`.
    fn noise_twin<const SEEDS: usize>(
        shape: &Self::Shape,
        twin: &mut Self::Twin<SEEDS>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource<Gradient<SEEDS>>>,
    );

    /// Collects every `(path, value)` pair visited by [`crate::Lift`] on `parameters` in
    /// traversal order.
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
        fn dc_value(prepared: &Self::Prepared) -> Option<f64>;
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

    fn noise_twin<const SEEDS: usize>(
        shape: &K::Shape,
        (prepared, _): &mut Self::Twin<SEEDS>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource<Gradient<SEEDS>>>,
    ) {
        K::noise(shape, prepared, at, sources);
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

/// Collects every `(path, value)` pair visited by [`crate::Lift`] on erased `K` `parameters`.
#[cfg(any(test, feature = "testing"))]
fn lift_entries_of<K: Kind>(parameters: &Erased) -> Vec<(String, f64)> {
    parameters
        .downcast_ref::<K::Parameters>()
        .map_or_else(Vec::new, K::lift_entries)
}

/// Prepared and laid-out instance ready to join a group, holding typed shape and parameters.
pub(super) trait Made: Send {
    /// Returns the type-erased shape descriptor for group matching.
    fn shaped(&self) -> &dyn Shaped;

    /// Returns whether the instance is linear, as determined by [`Kernel::linear`].
    fn linear(&self) -> bool;

    /// Converts the prepared instance into [`Any`] for transfer into its group run.
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
}

/// Concrete prepared and laid-out instance of kernel `K`.
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

/// Prepares erased `K` `parameters` at `temperature` and claims layout slots in `layout`.
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

/// Type-erased kernel and shape key capable of comparing shapes and creating group runs.
pub(super) trait Shaped: fmt::Debug + Send + Sync {
    /// Returns the [`TypeId`] identifying the group's kernel.
    fn kernel(&self) -> TypeId;

    /// Returns whether `other` represents the same kernel and shape.
    fn same(&self, other: &dyn Shaped) -> bool;

    /// Returns `self` as [`Any`] for downcasting inside [`Shaped::same`].
    fn as_any(&self) -> &dyn Any;

    /// Clones the shape into a heap-allocated trait object.
    fn boxed(&self) -> Box<dyn Shaped>;

    /// Creates an unprepared [`GroupRun`] sized for `members` instances.
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
