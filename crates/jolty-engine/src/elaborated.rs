//! Elaborated circuit representation built once per structure and shared across runs.
//!
//! An [`Elaborated`] circuit stores everything fixed for a given topology: unknowns, placed
//! devices and their origins, kernel instance groups, the logic partition, the matrix
//! sparsity pattern, the sparse solver ordering, and the numeric parameter table. An
//! [`Elaborated`] value is immutable after construction, so worker threads share it safely.
//! Each run instantiates devices from it using numeric overrides in [`Values`]. When
//! overridden numbers alter a device's internal unknowns or matrix entries,
//! [`Elaborated::restructured`] builds a circuit for the updated topology.

use std::collections::BTreeMap;
use std::fmt;
use std::ops::Range;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use jolty_ir::{Circuit, Name, Parameters};
use jolty_models::{
    Binding, Groups, Instance, Interface, Library, LogicDriver, LogicDriverParameters, Model,
    Nature, Node, ParameterError, REFERENCE_TEMPERATURE, Regions, Runs, Specification,
    ThresholdDetector, ThresholdDetectorParameters, Unknown, UnknownIndex,
};
use jolty_numerics::{Condensation, Dissecting, Entry, Ordering, Pattern, PatternBuilder};
use serde_json::Value;

use crate::elaborate::{
    self, ElaborationError, ElaborationProblem, Location, Replaced, SummedTerminal,
};
use crate::logic::LogicSystem;
use crate::values::{ParameterId, Table, Values, each_number, each_number_mut};
use crate::{SimulationError, Substitute};

/// Structural conditions for elaborating a circuit: substitute models replacing component
/// instances according to plan fidelity.
///
/// Temperature and numeric parameter overrides belong to the per-run [`Values`] input.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Structure {
    /// Substitute models replacing component instances, keyed by instance path (such as
    /// `adder/U1`).
    pub substitutes: BTreeMap<String, Substitute>,
}

/// Elaborated circuit shared across simulation runs, holding flattened devices, unknowns,
/// matrix sparsity pattern, sparse solver ordering, logic partition, and parameter table.
///
/// Node voltages occupy the leading unknown indices in net discovery order, followed by
/// device internal unknowns. Nets joined exclusively by logic pins become logic nets; a
/// logic pin on an electrical net bridges to a dedicated logic net via a connect element.
///
/// An [`Elaborated`] value is immutable after construction, so every thread shares it while
/// each [`Simulator`](crate::Simulator) instantiates separate devices from its [`Values`].
#[derive(Debug)]
pub struct Elaborated {
    pub(crate) unknowns: Vec<Unknown>,
    pub(crate) devices: Vec<Placed>,
    pub(crate) logic: LogicSystem,
    /// Name of the reference ground port on the top component.
    pub(crate) reference: Name,
    pub(crate) pattern: Arc<Pattern>,
    /// Matrix unknowns eliminated by static condensation before sparse factorization
    /// (device internal unknowns with proven pivots).
    pub(crate) condensation: Arc<Condensation>,
    /// Shared minimum-degree ordering over the uncondensed unknowns.
    pub(crate) ordering: Arc<Ordering>,
    /// Unknown names indexed by unknown position, matching solver diagnostics.
    pub(crate) names: Arc<[String]>,
    /// Physical quantity of each unknown, read during Newton tolerance checks and step control.
    pub(crate) natures: Vec<Nature>,
    /// Diagonal matrix entry for each node-voltage equation, where gmin stepping adds
    /// conductance to ground.
    pub(crate) node_diagonals: Vec<(UnknownIndex, Entry)>,
    /// Kernel instance groups evaluated in batches, together with ungrouped fallback devices.
    pub(crate) groups: Groups,
    /// Default device temperature in kelvins when a run provides no override.
    pub(crate) temperature: f64,
    /// Summed terminals for component ports and logic devices.
    pub(crate) summed_terminals: Vec<SummedTerminal>,
    /// Lazily initialized path index and parameter table, derived deterministically without
    /// mutating logical state.
    lookup: OnceLock<Lookup>,
    /// Lazily cached nested-dissection orderings keyed by [`Dissecting`], falling back to the
    /// minimum-degree ordering when no block dissects.
    dissected: Mutex<BTreeMap<Dissecting, Arc<Ordering>>>,
}

/// Path index for devices and summed terminals paired with the sorted numeric parameter table.
#[derive(Debug, Clone)]
struct Lookup {
    index: Index,
    table: Table,
}

impl Lookup {
    /// Builds the path index and numeric parameter table for `devices` and `summed_terminals`.
    fn new(devices: &[Placed], summed_terminals: &[SummedTerminal]) -> Self {
        let index = Index::new(devices, summed_terminals);
        let parameters: Vec<Option<&Parameters>> = devices.iter().map(Placed::parameters).collect();
        let table = Table::new(&parameters, &index.devices);
        Self { index, table }
    }
}

/// Flattened device placement recording instance location, model name, and terminal nodes.
#[derive(Debug, Clone)]
pub struct Placed {
    template: Arc<Template>,
    /// Starting unknown index for this device's internal unknowns.
    pub(crate) first_internal: usize,
    /// Number of internal unknowns added by this device.
    internals: usize,
    /// Names of operating-point variables computed by this device.
    variables: &'static [&'static str],
    /// Operating regions distinguished by this device's equations.
    regions: Regions,
    /// Flag indicating whether the device was bound from unmodified circuit parameters at the
    /// circuit temperature.
    nominal: bool,
}

/// Parameter-independent device metadata: location, model name, terminal interface, nodes,
/// and nominal origin.
#[derive(Debug)]
pub(crate) struct Template {
    pub(crate) location: Location,
    pub(crate) model: &'static str,
    pub(crate) interface: Interface,
    pub(crate) nodes: Vec<Node>,
    pub(crate) origin: Origin,
}

/// Source backing a placed analog device.
#[derive(Debug)]
pub(crate) enum Origin {
    /// Standard or custom model instantiated from instance parameters.
    Model(Made),
    /// Threshold detector connect element for a logic input.
    Detector(ThresholdDetectorParameters),
    /// Logic driver connect element for a logic output.
    Driver(LogicDriverParameters),
}

/// Parsed model specification paired with the raw parameters and model handle.
pub(crate) struct Made {
    /// Model handle used to re-parse parameters when a run overrides numeric values.
    pub(crate) model: Arc<dyn Model>,
    pub(crate) specification: Box<dyn Specification>,
    pub(crate) parameters: Parameters,
}

impl fmt::Debug for Made {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Made")
            .field("model", &self.model.name())
            .field("specification", &self.specification)
            .field("parameters", &self.parameters)
            .finish()
    }
}

/// Prepared kernel run storage for a single simulation.
pub(crate) type Devices = Runs;

impl Origin {
    /// Builds a kernel [`Instance`] from this device origin.
    pub(crate) fn kernel(&self) -> Instance<'_> {
        match self {
            Self::Model(made) => made.specification.kernel(),
            Self::Detector(parameters) => {
                Instance::from_parameters::<ThresholdDetector>(parameters)
            }
            Self::Driver(parameters) => Instance::from_parameters::<LogicDriver>(parameters),
        }
    }
}

impl Placed {
    /// Returns the instance path and script source location.
    #[must_use]
    pub fn location(&self) -> &Location {
        &self.template.location
    }

    /// Returns the hierarchical instance path, such as `amplifier/R1`.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.template.location.path
    }

    /// Returns the device model name, such as `resistor`.
    #[must_use]
    pub fn model(&self) -> &'static str {
        self.template.model
    }

    /// Returns the device terminal names in the order of [`Placed::nodes`].
    #[must_use]
    pub fn terminals(&self) -> &[String] {
        &self.template.interface.terminals
    }

    /// Returns the indices of thermal terminals within [`Placed::terminals`].
    #[must_use]
    pub fn thermal(&self) -> &[usize] {
        &self.template.interface.thermal
    }

    /// Returns the device branches connecting its terminals.
    #[must_use]
    pub fn branches(&self) -> &[jolty_models::Branch] {
        &self.template.interface.branches
    }

    /// Returns the node joined by each terminal, with `None` representing ground.
    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.template.nodes
    }

    /// Returns the range of unknown indices for internal nodes such as a transistor's internal
    /// drain behind series resistance.
    #[must_use]
    pub fn internals(&self) -> Range<usize> {
        self.first_internal..self.first_internal + self.internals
    }

    /// Returns the names of computed device variables, such as `threshold_voltage`.
    #[must_use]
    pub fn variable_names(&self) -> &'static [&'static str] {
        self.variables
    }

    /// Returns the operating regions numbered by [`REGION`](jolty_models::REGION) and their
    /// boundary margins, or an empty set for non-transistors.
    #[must_use]
    pub const fn regions(&self) -> Regions {
        self.regions
    }

    /// Reports whether the device was bound from unmodified circuit parameters at the circuit
    /// temperature.
    pub(crate) const fn nominal(&self) -> bool {
        self.nominal
    }

    /// Reports whether the device uses a foundry model unsupported by standard analyses.
    #[must_use]
    pub fn foundry(&self) -> bool {
        match &self.template.origin {
            Origin::Model(made) => made.model.foundry(),
            Origin::Detector(_) | Origin::Driver(_) => false,
        }
    }

    /// Returns the nominal model parameters from elaboration, or `None` for connect elements.
    ///
    /// Runs whose [`Values`] override numeric entries instantiate devices from those overrides
    /// instead.
    #[must_use]
    pub fn parameters(&self) -> Option<&Parameters> {
        match &self.template.origin {
            Origin::Model(made) => Some(&made.parameters),
            Origin::Detector(_) | Origin::Driver(_) => None,
        }
    }

    /// Returns the nominal `f64` value of top-level parameter `name`, or `None` when absent or
    /// non-numeric.
    #[must_use]
    pub fn numeric_parameter(&self, name: &str) -> Option<f64> {
        self.parameters()?.get(name)?.as_f64()
    }
}

/// Structural delta between an elaborated circuit and the topology induced by a [`Values`]
/// override, returned by [`Elaborated::structure_of`].
///
/// Entries list every device whose internal unknowns or matrix entries differ from the base
/// circuit. Two [`Values`] sets share a structure when their keys match, and an empty key
/// indicates that the base circuit structure is preserved.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StructureKey(Vec<(usize, Layout)>);

impl StructureKey {
    /// Reports whether the evaluated values preserve the base elaborated circuit structure.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Yields indices into [`Simulator::devices`](crate::Simulator::devices) for devices whose
    /// unknowns or entries change, in order.
    pub fn devices(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().map(|&(device, _)| device)
    }
}

/// Structural claims of a device: internal unknowns, matrix entry positions, variable names,
/// and operating regions.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Layout {
    internals: Vec<Unknown>,
    entries: Vec<(usize, usize)>,
    variables: &'static [&'static str],
    regions: Regions,
}

/// Sorted index vectors enabling binary search of devices by path and summed terminals by name.
#[derive(Debug, Clone)]
struct Index {
    devices: Vec<usize>,
    summed_terminals: Vec<usize>,
}

impl Index {
    /// Builds path-sorted device indices and name-sorted summed-terminal indices.
    fn new(devices: &[Placed], summed_terminals: &[SummedTerminal]) -> Self {
        let mut by_path: Vec<usize> = (0..devices.len()).collect();
        by_path.sort_unstable_by(|&a, &b| devices[a].path().cmp(devices[b].path()));
        let mut by_name: Vec<usize> = (0..summed_terminals.len()).collect();
        by_name.sort_unstable_by(|&a, &b| summed_terminals[a].name.cmp(&summed_terminals[b].name));
        Self {
            devices: by_path,
            summed_terminals: by_name,
        }
    }
}

/// Accumulator binding devices sequentially into unknowns, sparsity pattern entries, and
/// kernel groups.
pub(crate) struct Binder {
    unknowns: Vec<Unknown>,
    builder: PatternBuilder,
    node_diagonals: Vec<(UnknownIndex, Entry)>,
    placed: Vec<Placed>,
    groups: Groups,
    runs: Runs,
    temperature: f64,
}

/// Finalized device bindings, sparsity pattern, condensation map, and ordering from
/// [`Binder::finish`].
pub(crate) struct Bound {
    pub(crate) unknowns: Vec<Unknown>,
    pub(crate) placed: Vec<Placed>,
    pub(crate) devices: Devices,
    groups: Groups,
    node_diagonals: Vec<(UnknownIndex, Entry)>,
    pattern: Arc<Pattern>,
    condensation: Arc<Condensation>,
    ordering: Arc<Ordering>,
    names: Arc<[String]>,
    natures: Vec<Nature>,
}

impl Binder {
    /// Creates a device binder at `temperature` (kelvins) initialized with `nodes` and their
    /// diagonal entries.
    pub(crate) fn new(nodes: Vec<Unknown>, temperature: f64) -> Self {
        let mut builder = PatternBuilder::new();
        let node_diagonals = (0..nodes.len())
            .map(|index| (UnknownIndex::new(index), builder.entry(index, index)))
            .collect();
        Self {
            unknowns: nodes,
            builder,
            node_diagonals,
            placed: Vec::new(),
            groups: Groups::new(),
            runs: Runs::default(),
            temperature,
        }
    }

    /// Binds a device from `template` using `origin` (or the template's nominal origin) into
    /// its kernel group, recording new internal unknowns and matrix entries.
    ///
    /// Sets `nominal` to indicate unmodified circuit parameters at the circuit temperature.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] if any parameter is out of range, leaving the binder
    /// unchanged.
    pub(crate) fn bind(
        &mut self,
        template: Arc<Template>,
        origin: Option<&Origin>,
        nominal: bool,
    ) -> Result<(), ParameterError> {
        let first_internal = self.unknowns.len();
        let mut binding = Binding::new(
            &template.location.path,
            &template.nodes,
            &mut self.unknowns,
            &mut self.builder,
        );
        let origin = origin.unwrap_or(&template.origin);
        // Each instance claims its unknowns and entries in its group.
        let instance = origin.kernel();
        self.groups
            .add(&mut self.runs, (instance, self.temperature), &mut binding)?;
        let variables = instance.variable_names();
        self.placed.push(Placed {
            first_internal,
            internals: self.unknowns.len() - first_internal,
            variables,
            regions: instance.regions(),
            nominal,
            template,
        });
        Ok(())
    }

    /// Finalizes all bound devices, building the sparsity pattern, static condensation, and
    /// ordering.
    ///
    /// # Errors
    ///
    /// Returns [`ElaborationProblem::TooLarge`] if the ordering exceeds memory limits or a
    /// kernel group slab overflows its index space.
    pub(crate) fn finish(mut self) -> Result<Bound, ElaborationError> {
        let pattern = Arc::new(self.builder.finish(self.unknowns.len()));
        let too_large = || ElaborationError::from(ElaborationProblem::TooLarge);
        self.groups
            .finish(&mut self.runs, &pattern)
            .map_err(|_| too_large())?;
        let condensation = Arc::new(Condensation::new(
            Arc::clone(&pattern),
            self.groups.pivoted(),
        ));
        let ordering = Ordering::condensed(Arc::clone(&condensation), None)
            .map_err(|_| ElaborationError::from(ElaborationProblem::TooLarge))?;
        let names = self
            .unknowns
            .iter()
            .map(|unknown| unknown.name.clone())
            .collect();
        let natures = self.unknowns.iter().map(|unknown| unknown.nature).collect();
        Ok(Bound {
            unknowns: self.unknowns,
            placed: self.placed,
            devices: self.runs,
            groups: self.groups,
            node_diagonals: self.node_diagonals,
            pattern,
            condensation,
            ordering: Arc::new(ordering),
            names,
            natures,
        })
    }
}

/// Tuple of a restructured [`Elaborated`] circuit, its bound [`Devices`], and per-device
/// patched origins.
pub(crate) type Rebound = (Elaborated, Devices, Vec<Option<Origin>>);

/// Effective device origins for a run, combining base elaboration, value patches, sweep
/// overrides, and temperature.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Origins<'a> {
    pub(crate) elaborated: &'a Elaborated,
    pub(crate) patched: &'a [Option<Origin>],
    pub(crate) swept: &'a BTreeMap<usize, Origin>,
    pub(crate) temperature: f64,
}

impl Origins<'_> {
    /// Returns the active [`Origin`] for device `index`, prioritizing sweep overrides over value
    /// patches and nominal origins.
    pub(crate) fn of(&self, index: usize) -> &Origin {
        self.swept
            .get(&index)
            .or_else(|| self.patched.get(index).and_then(Option::as_ref))
            .unwrap_or(&self.elaborated.devices[index].template.origin)
    }
}

impl Elaborated {
    /// Elaborates `circuit` against `library` under `structure` at the reference temperature
    /// (27 °C).
    ///
    /// # Errors
    ///
    /// Returns every [`ElaborationProblem`] found, including invalid structure, missing ground,
    /// unknown models, bad terminal connections, or invalid parameters.
    pub fn new(
        circuit: &Circuit,
        library: &Library,
        structure: &Structure,
    ) -> Result<Arc<Self>, ElaborationError> {
        let empty_overrides = BTreeMap::new();
        let replaced = Replaced {
            substitutes: &structure.substitutes,
            overrides: &empty_overrides,
        };
        let (elaborated, _) =
            elaborate::elaborate(circuit, library, replaced, REFERENCE_TEMPERATURE)?;
        Ok(Arc::new(elaborated))
    }

    /// Assembles an [`Elaborated`] circuit and its [`Devices`] from `bound`, `logic`,
    /// `summed_terminals`, `reference`, and `temperature`.
    pub(crate) fn assemble(
        bound: Bound,
        logic: LogicSystem,
        summed_terminals: Vec<SummedTerminal>,
        reference: Name,
        temperature: f64,
    ) -> (Self, Devices) {
        let elaborated = Self {
            unknowns: bound.unknowns,
            devices: bound.placed,
            logic,
            reference,
            pattern: bound.pattern,
            condensation: bound.condensation,
            ordering: bound.ordering,
            names: bound.names,
            natures: bound.natures,
            node_diagonals: bound.node_diagonals,
            groups: bound.groups,
            temperature,
            summed_terminals,
            lookup: OnceLock::new(),
            dissected: Mutex::new(BTreeMap::new()),
        };
        (elaborated, bound.devices)
    }

    /// Returns the dimension of the largest diagonal block in the minimum-degree ordering.
    pub(crate) fn largest_block(&self) -> usize {
        self.ordering.largest_block()
    }

    /// Returns the sparse ordering for `dissecting`, computing and caching dissected orderings
    /// on demand.
    ///
    /// Returns the shared minimum-degree ordering when `dissecting` is `None` or when no block
    /// exceeds the interior threshold. Concurrent requests may compute the same ordering, and
    /// the first stored entry is shared thereafter.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::TooLarge`] if the ordering does not fit in memory.
    pub(crate) fn ordering(
        &self,
        dissecting: Option<Dissecting>,
    ) -> Result<Arc<Ordering>, SimulationError> {
        let Some(dissecting) =
            dissecting.filter(|each| self.ordering.largest_block() > each.interiors())
        else {
            return Ok(Arc::clone(&self.ordering));
        };
        let kept = || {
            self.dissected
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
        };
        if let Some(ordering) = kept().get(&dissecting) {
            return Ok(Arc::clone(ordering));
        }
        let made = Ordering::condensed(Arc::clone(&self.condensation), Some(dissecting))
            .map_err(|_| SimulationError::TooLarge)?;
        let made = if made.is_dissected() {
            Arc::new(made)
        } else {
            Arc::clone(&self.ordering)
        };
        Ok(Arc::clone(kept().entry(dissecting).or_insert_with(|| made)))
    }

    /// Returns all system unknowns in solution-vector order.
    #[must_use]
    pub fn unknowns(&self) -> &[Unknown] {
        &self.unknowns
    }

    /// Returns all placed analog devices in traversal order, followed by logic connect elements.
    #[must_use]
    pub fn devices(&self) -> &[Placed] {
        &self.devices
    }

    /// Looks up the [`ParameterId`] for numeric parameter `name` (such as `resistance` or
    /// `waveform.dc`) on device `path`.
    #[must_use]
    pub fn parameter(&self, path: &str, name: &str) -> Option<ParameterId> {
        let device = self.device(path)?;
        let Origin::Model(made) = &self.devices[device].template.origin else {
            return None;
        };
        let leaf = jolty_ir::parameter(&made.parameters, name)?;
        if !leaf.is_number() {
            return None;
        }
        let mut place = None;
        let mut counted = 0;
        for value in made.parameters.values() {
            each_number(value, &mut |number| {
                if std::ptr::eq(number, leaf) {
                    place = Some(counted);
                }
                counted += 1;
            });
        }
        self.lookup().table.parameter(device, place?)
    }

    /// Returns the slice of nominal parameter values indexed by [`ParameterId`].
    #[must_use]
    pub fn nominal(&self) -> &[f64] {
        self.lookup().table.nominal()
    }

    /// Resolves `overrides` (keyed by device path then parameter path) and optional
    /// `temperature` (kelvins) into a [`Values`] bundle.
    ///
    /// # Errors
    ///
    /// Returns problems for unknown device paths, unrecognized parameter paths, and overrides
    /// targeting logic devices, whose parameters are baked into the logic partition during
    /// elaboration.
    pub fn values(
        &self,
        overrides: &BTreeMap<String, BTreeMap<String, f64>>,
        temperature: Option<f64>,
    ) -> Result<Values, ElaborationError> {
        let mut values = Values::default();
        values.temperature = temperature;
        let mut problems = Vec::new();
        for (path, numbers) in overrides {
            let Some(device) = self.device(path) else {
                match self.logic.devices.iter().find(|logic| logic.path() == path) {
                    Some(logic) => problems.extend(numbers.keys().map(|name| {
                        ElaborationProblem::LogicParameter {
                            at: logic.location().clone(),
                            parameter: name.clone(),
                        }
                    })),
                    None => problems.push(ElaborationProblem::UnknownOverride(path.clone())),
                }
                continue;
            };
            for (name, &value) in numbers {
                match self.parameter(path, name) {
                    Some(parameter) => values.set(parameter, value),
                    None => problems.push(ElaborationProblem::UnknownParameter {
                        at: self.devices[device].location().clone(),
                        parameter: name.clone(),
                    }),
                }
            }
        }
        if problems.is_empty() {
            Ok(values)
        } else {
            Err(ElaborationError::new(problems))
        }
    }

    /// Computes the [`StructureKey`] describing devices whose unknowns or matrix entries differ
    /// under `values`.
    ///
    /// # Errors
    ///
    /// Returns every [`ElaborationProblem`] caused by unknown parameter indices or out-of-range
    /// numeric values.
    pub fn structure_of(&self, values: &Values) -> Result<StructureKey, ElaborationError> {
        let temperature = self.temperature_of(values);
        let patched = self.patch(values)?;
        let unchanged = temperature.to_bits() == self.temperature.to_bits();
        let mut changed = Vec::new();
        let mut problems = Vec::new();
        for (index, placed) in self.devices.iter().enumerate() {
            let own = patched.get(index).and_then(Option::as_ref);
            if unchanged && placed.nominal && own.is_none() {
                continue;
            }
            let origin = own.unwrap_or(&placed.template.origin);
            match self
                .groups
                .keeps_shape(index, (&origin.kernel(), temperature))
            {
                Ok(true) => {}
                Ok(false) => match self.layout(index, origin, temperature) {
                    Ok(layout) => changed.push((index, layout)),
                    Err(error) => problems.push(self.problem(index, error)),
                },
                Err(error) => problems.push(self.problem(index, error)),
            }
        }
        if problems.is_empty() {
            Ok(StructureKey(changed))
        } else {
            Err(ElaborationError::new(problems))
        }
    }

    /// Builds a new [`Elaborated`] circuit whose devices are bound with the structure induced by
    /// `values`.
    ///
    /// Preserves the base circuit's nominal parameter table and reference temperature so the
    /// same [`Values`] produce identical devices on either circuit.
    ///
    /// # Errors
    ///
    /// Returns any [`ElaborationProblem`] produced by `values`, or
    /// [`ElaborationProblem::TooLarge`] if the rebuilt system exceeds factorization limits.
    pub fn restructured(&self, values: &Values) -> Result<Arc<Self>, ElaborationError> {
        let (restructured, _, _) = self.rebind(values)?;
        Ok(Arc::new(restructured))
    }

    /// Rebinds the circuit under `values`, returning the restructured circuit, bound devices,
    /// and patched origins.
    pub(crate) fn rebind(&self, values: &Values) -> Result<Rebound, ElaborationError> {
        let temperature = self.temperature_of(values);
        let patched = self.patch(values)?;
        let unchanged = temperature.to_bits() == self.temperature.to_bits();
        let mut binder = Binder::new(
            self.unknowns[..self.node_diagonals.len()].to_vec(),
            temperature,
        );
        let mut problems = Vec::new();
        for (index, placed) in self.devices.iter().enumerate() {
            let own = patched.get(index).and_then(Option::as_ref);
            let template = Arc::clone(&placed.template);
            if let Err(error) = binder.bind(template, own, unchanged && own.is_none()) {
                problems.push(self.problem(index, error));
            }
        }
        if !problems.is_empty() {
            return Err(ElaborationError::new(problems));
        }
        let bound = binder.finish()?;
        let restructured = Self {
            unknowns: bound.unknowns,
            devices: bound.placed,
            logic: self.logic.clone(),
            reference: self.reference.clone(),
            pattern: bound.pattern,
            condensation: bound.condensation,
            ordering: bound.ordering,
            names: bound.names,
            natures: bound.natures,
            node_diagonals: bound.node_diagonals,
            groups: bound.groups,
            temperature: self.temperature,
            summed_terminals: self.summed_terminals.clone(),
            lookup: self.lookup.clone(),
            dissected: Mutex::new(BTreeMap::new()),
        };
        Ok((restructured, bound.devices, patched))
    }

    /// Returns the default device temperature in kelvins used when a run provides no override.
    #[must_use]
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Returns the effective device temperature in kelvins for `values`.
    pub(crate) fn temperature_of(&self, values: &Values) -> f64 {
        values.temperature.unwrap_or(self.temperature)
    }

    /// Re-parses modified devices under `values`, returning per-device patched origins.
    ///
    /// Unmodified devices receive `None`, and an empty vector is returned when `values` carries
    /// no parameter patches.
    ///
    /// # Errors
    ///
    /// Returns problems for out-of-range parameter indices, non-finite values, rejected model
    /// parameters, or overrides that alter a device's terminal interface.
    pub(crate) fn patch(&self, values: &Values) -> Result<Vec<Option<Origin>>, ElaborationError> {
        let replaced = values.patches();
        let mut origins: Vec<Option<Origin>> = Vec::new();
        if replaced.is_empty() {
            return Ok(origins);
        }
        origins.resize_with(self.devices.len(), || None);
        let mut problems = Vec::new();
        let mut start = 0;
        while let Some(&(first, _)) = replaced.get(start) {
            let Some((device, _)) = self.lookup().table.locate(first) else {
                problems.push(ElaborationProblem::NoParameter(first.index()));
                start += 1;
                continue;
            };
            let count = replaced[start..]
                .iter()
                .take_while(|(parameter, _)| {
                    self.lookup().table.locate(*parameter).map(|(of, _)| of) == Some(device)
                })
                .count();
            let own = &replaced[start..start + count];
            start += count;
            match self.patched(device, own) {
                Ok(origin) => origins[device] = Some(origin),
                Err(problem) => problems.push(*problem),
            }
        }
        if problems.is_empty() {
            Ok(origins)
        } else {
            Err(ElaborationError::new(problems))
        }
    }

    /// Returns the sorted slice of `(ParameterId, f64)` overrides targeting device `index` in
    /// `values`.
    pub(crate) fn patches_of<'a>(
        &self,
        values: &'a Values,
        index: usize,
    ) -> &'a [(ParameterId, f64)] {
        let numbers = self.lookup().table.range(index);
        let patches = values.patches();
        let first = patches.partition_point(|(parameter, _)| parameter.index() < numbers.start);
        let last = patches.partition_point(|(parameter, _)| parameter.index() < numbers.end);
        &patches[first..last]
    }

    /// Builds a constant-source [`Origin`] holding independent source `index` at `value`.
    ///
    /// # Errors
    ///
    /// Returns an [`ElaborationProblem`] if the device is a connect element, rejects the held
    /// parameters, or changes its terminal interface.
    pub(crate) fn held_at(
        &self,
        index: usize,
        value: f64,
    ) -> Result<Origin, Box<ElaborationProblem>> {
        let placed = &self.devices[index];
        let at = || placed.location().clone();
        let Origin::Model(made) = &placed.template.origin else {
            return Err(Box::new(ElaborationProblem::Parameters {
                at: at(),
                message: "a connect element is held at no value".to_owned(),
            }));
        };
        let parameters = jolty_models::held_source(value);
        self.read(index, made, parameters)
    }

    /// Re-parses device `index` with numeric `patches` applied to its nominal parameters.
    pub(crate) fn patched(
        &self,
        index: usize,
        patches: &[(ParameterId, f64)],
    ) -> Result<Origin, Box<ElaborationProblem>> {
        let placed = &self.devices[index];
        let at = || placed.location().clone();
        let Origin::Model(made) = &placed.template.origin else {
            let first = patches
                .first()
                .map_or(0, |(parameter, _)| parameter.index());
            return Err(Box::new(ElaborationProblem::NoParameter(first)));
        };
        if let Some(&(parameter, value)) = patches.iter().find(|(_, value)| !value.is_finite()) {
            let place = self
                .lookup()
                .table
                .locate(parameter)
                .map_or(0, |(_, place)| place);
            let name = jolty_ir::numbers(&made.parameters)
                .into_iter()
                .nth(place)
                .map_or_else(String::new, |(name, _)| name);
            return Err(Box::new(ElaborationProblem::Parameters {
                at: at(),
                message: format!("parameter `{name}` requires a finite value; got {value:?}"),
            }));
        }
        let mut parameters = made.parameters.clone();
        let mut pending = patches.iter().peekable();
        let mut place = 0;
        for value in parameters.values_mut() {
            each_number_mut(value, &mut |number| {
                if let Some(&&(parameter, replaced)) = pending.peek()
                    && self.lookup().table.locate(parameter) == Some((index, place))
                {
                    *number = written(number, replaced);
                    pending.next();
                }
                place += 1;
            });
        }
        self.read(index, made, parameters)
    }

    /// Parses `parameters` with the model in `made` for device `index` and verifies its
    /// interface.
    ///
    /// # Errors
    ///
    /// Returns an [`ElaborationProblem`] if the model rejects `parameters` or if the resulting
    /// terminal interface differs from the elaborated template.
    fn read(
        &self,
        index: usize,
        made: &Made,
        parameters: Parameters,
    ) -> Result<Origin, Box<ElaborationProblem>> {
        let placed = &self.devices[index];
        let at = || placed.location().clone();
        let specification = made.model.read(&parameters).map_err(|error| {
            Box::new(ElaborationProblem::Parameters {
                at: at(),
                message: error.0,
            })
        })?;
        if specification.interface() != placed.template.interface {
            return Err(Box::new(ElaborationProblem::Parameters {
                at: at(),
                message: "its numbers replaced change its terminals, which only elaborating the \
                          circuit with them can"
                    .to_owned(),
            }));
        }
        Ok(Origin::Model(Made {
            model: Arc::clone(&made.model),
            specification,
            parameters,
        }))
    }

    /// Computes the structural [`Layout`] claimed by device `index` using `origin` at
    /// `temperature`.
    fn layout(
        &self,
        index: usize,
        origin: &Origin,
        temperature: f64,
    ) -> Result<Layout, ParameterError> {
        let placed = &self.devices[index];
        let mut internals = Vec::new();
        let mut builder = PatternBuilder::new();
        let mut binding = Binding::numbered(
            placed.path(),
            &placed.template.nodes,
            (placed.first_internal, &mut internals),
            &mut builder,
        );
        let instance = origin.kernel();
        instance.claim(temperature, &mut binding)?;
        Ok(Layout {
            internals,
            entries: builder.positions().to_vec(),
            variables: instance.variable_names(),
            regions: instance.regions(),
        })
    }

    /// Wraps a [`ParameterError`] for device `index` as an [`ElaborationProblem::Parameters`].
    pub(crate) fn problem(&self, index: usize, error: ParameterError) -> ElaborationProblem {
        ElaborationProblem::Parameters {
            at: self.devices[index].location().clone(),
            message: error.0,
        }
    }

    /// Returns the lazily initialized path index and numeric parameter table.
    fn lookup(&self) -> &Lookup {
        self.lookup
            .get_or_init(|| Lookup::new(&self.devices, &self.summed_terminals))
    }

    /// Looks up the [`Location`] of the device at `path`.
    pub(crate) fn location(&self, path: &str) -> Option<Location> {
        Some(self.devices[self.device(path)?].location().clone())
    }

    /// Looks up the device index for hierarchical `path`.
    pub(crate) fn device(&self, path: &str) -> Option<usize> {
        let found = self
            .lookup()
            .index
            .devices
            .binary_search_by(|&device| self.devices[device].path().cmp(path))
            .ok()?;
        Some(self.lookup().index.devices[found])
    }

    /// Looks up the node and thermal flag for `terminal` on `instance`, matching
    /// [`Simulator::terminal`](crate::Simulator::terminal).
    pub(crate) fn terminal(&self, instance: &str, terminal: &str) -> Option<(Node, bool)> {
        if let Some(index) = self.device(instance) {
            let device = &self.devices[index];
            let position = device
                .terminals()
                .iter()
                .position(|name| name == terminal)?;
            return Some((
                device.nodes()[position],
                device.thermal().contains(&position),
            ));
        }
        let name = format!("{instance}.{terminal}");
        let summed_terminals = &self.summed_terminals;
        let found = self
            .lookup()
            .index
            .summed_terminals
            .binary_search_by(|&summed_terminal| summed_terminals[summed_terminal].name.cmp(&name))
            .ok()?;
        let summed_terminal = &summed_terminals[self.lookup().index.summed_terminals[found]];
        Some((summed_terminal.node, summed_terminal.thermal))
    }
}

/// Encodes `value` matching the JSON numeric type of `written`.
///
/// Preserves integer encoding when `written` is an integer and `value` is an exact whole
/// number within IEEE 754 binary64 precision; otherwise encodes as floating point.
fn written(written: &Value, value: f64) -> Value {
    /// The largest magnitude below which every whole number is exact in IEEE 754 binary64
    /// (`f64`): `2^53` (53-bit significand precision, `f64::MANTISSA_DIGITS`).
    const EXACT: f64 = 9_007_199_254_740_992.0;
    let integer = written.is_i64() || written.is_u64();
    if integer && value.fract() == 0.0 && value.abs() < EXACT {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a whole number below 2⁵³ in magnitude is an i64 exactly"
        )]
        let whole = value as i64;
        Value::from(whole)
    } else {
        Value::from(value)
    }
}

/// Disjoint-set structure over indexed nodes using union by size with path halving.
pub struct Partition {
    parents: Vec<usize>,
    sizes: Vec<usize>,
}

impl Partition {
    /// Creates a partition of `count` singleton sets.
    #[must_use]
    pub fn new(count: usize) -> Self {
        Self {
            parents: (0..count).collect(),
            sizes: vec![1; count],
        }
    }

    /// Returns the representative element of the set containing `node`.
    pub fn find(&mut self, mut node: usize) -> usize {
        while self.parents[node] != node {
            self.parents[node] = self.parents[self.parents[node]];
            node = self.parents[node];
        }
        node
    }

    /// Merges the sets containing `a` and `b`, returning whether they were previously distinct.
    pub fn union(&mut self, a: usize, b: usize) -> bool {
        let (mut a, mut b) = (self.find(a), self.find(b));
        if a == b {
            return false;
        }
        if self.sizes[a] < self.sizes[b] {
            std::mem::swap(&mut a, &mut b);
        }
        self.parents[b] = a;
        self.sizes[a] += self.sizes[b];
        true
    }
}

#[cfg(test)]
mod tests {
    use super::Partition;

    #[test]
    fn a_partition_merges_sets() {
        let mut partition = Partition::new(4);
        assert!(partition.union(0, 1));
        assert!(partition.union(2, 3));
        assert!(!partition.union(1, 0));
        assert_eq!(partition.find(1), partition.find(0));
        assert_ne!(partition.find(1), partition.find(2));
        assert!(partition.union(1, 3));
        assert_eq!(partition.find(0), partition.find(2));
    }
}
