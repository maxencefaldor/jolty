//! An elaborated circuit: what a circuit is, built once per structure and shared by every
//! run of it, and how each run's instances are prepared from it.
//!
//! An [`Elaborated`] circuit holds everything elaboration finds that no run changes: the
//! unknowns, the devices placed and what each is made from, the groups of their kernels'
//! instances, the logic, the matrix's pattern and the sparse solver's ordering of it, and
//! the table of the instances' numbers. It never changes once built, so threads share it.
//! Each run prepares its instances from it, with the numbers its [`Values`] replace; an
//! instance whose shape changes under the new numbers claims other unknowns or entries, so
//! has another structure, which a circuit elaborated for those numbers holds,
//! [`Elaborated::restructured`].

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

use crate::elaborate::{self, Composite, ElaborationError, ElaborationProblem, Location};
use crate::logic::LogicSystem;
use crate::values::{ParameterId, Table, Values, each_number, each_number_mut};
use crate::{SimulationError, Substitute};

/// The structural half of the conditions a circuit is elaborated in: the models that stand
/// in for some component instances, as a plan's fidelity chooses. The temperature and the
/// numbers that replace the circuit's own are the other half, the [`Values`] each run
/// gives on its own.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Structure {
    /// Models that stand in for component instances, by the instance's path, such as
    /// `adder/U1`.
    pub substitutes: BTreeMap<String, Substitute>,
}

/// A circuit elaborated once for any number of runs: flattened into devices bound to
/// unknowns, with the pattern of its matrix, the sparse solver's ordering of it, the logic
/// devices joined by logic nets, and the table of its device instances' numbers.
///
/// The node voltages come first among the unknowns, in the order their nets are first met,
/// then the devices' internal unknowns. A net that only logic pins join is a logic net; where
/// a logic pin joins an electrical net, a connect element joins it to a logic net of its own.
///
/// It never changes once built: a [`Simulator`](crate::Simulator) made from it with some
/// [`Values`] makes devices of its own, and every thread shares it.
#[derive(Debug)]
pub struct Elaborated {
    pub(crate) unknowns: Vec<Unknown>,
    pub(crate) devices: Vec<Placed>,
    pub(crate) logic: LogicSystem,
    /// The name of the reference node: the top component's ground port.
    pub(crate) reference: Name,
    pub(crate) pattern: Arc<Pattern>,
    /// The unknowns of the pattern's matrices the sparse solver eliminates before it
    /// factors the rest: the devices' internal unknowns whose pivots are proven.
    pub(crate) condensation: Arc<Condensation>,
    /// The sparse solver's ordering of the pattern, which every solver of a run shares: of
    /// the unknowns the condensation keeps.
    pub(crate) ordering: Arc<Ordering>,
    /// The name of every unknown, by index, as the solvers name them.
    pub(crate) names: Arc<[String]>,
    /// The quantity of every unknown, by index, apart from its name: the tolerances of
    /// Newton iteration and step control read one for every unknown at every iteration.
    pub(crate) natures: Vec<Nature>,
    /// The diagonal entry of every node voltage's equation, where gmin stepping adds a
    /// conductance to the reference.
    pub(crate) node_diagonals: Vec<(UnknownIndex, Entry)>,
    /// The groups of the instances of kernels, which every run evaluates together, and the
    /// devices of no kernel, which it evaluates one by one.
    pub(crate) groups: Groups,
    /// The devices' temperature, in kelvins, unless a run's values give another.
    pub(crate) temperature: f64,
    /// The terminals of component instances and logic devices, each the sum of device
    /// terminals.
    pub(crate) composites: Vec<Composite>,
    /// The index of the devices' paths and the table of their numbers, made the first time
    /// either is read, as a single run of a circuit from its JSON text reads neither: a
    /// function of the rest, which they never change.
    lookup: OnceLock<Lookup>,
    /// The orderings of the pattern as each [`Dissecting`] orders it, each made the first
    /// time a run factors with it, the minimum degree's where it dissects no block: a
    /// function of the pattern, which it never changes.
    dissected: Mutex<BTreeMap<Dissecting, Arc<Ordering>>>,
}

/// The index of an elaborated circuit's devices and composite terminals by path and name,
/// and the table of its devices' numbers in the order of their paths.
#[derive(Debug, Clone)]
struct Lookup {
    index: Index,
    table: Table,
}

impl Lookup {
    /// The lookup of `devices` and `composites`.
    fn new(devices: &[Placed], composites: &[Composite]) -> Self {
        let index = Index::new(devices, composites);
        let parameters: Vec<Option<&Parameters>> = devices.iter().map(Placed::parameters).collect();
        let table = Table::new(&parameters, &index.devices);
        Self { index, table }
    }
}

/// A device placed in the flattened circuit: where its instance is, what model it is, and
/// the node each of its terminals joins.
#[derive(Debug, Clone)]
pub struct Placed {
    template: Arc<Template>,
    /// How many unknowns there were before the device added its own.
    pub(crate) first_internal: usize,
    /// How many internal unknowns the device added.
    internals: usize,
    /// The names of the variables it computes.
    variables: &'static [&'static str],
    /// The regions of operation its equations distinguish.
    regions: Regions,
    /// Whether it claimed its unknowns and entries as made from its own parameters, at its
    /// circuit's temperature, as a run that replaces none of its numbers makes it.
    nominal: bool,
}

/// What a device is, whatever its numbers: where its instance is, what model it is, its
/// terminals, the node each joins, and what it is made from.
#[derive(Debug)]
pub(crate) struct Template {
    pub(crate) location: Location,
    pub(crate) model: &'static str,
    pub(crate) interface: Interface,
    pub(crate) nodes: Vec<Node>,
    pub(crate) origin: Origin,
}

/// What a device is made from.
#[derive(Debug)]
pub(crate) enum Origin {
    /// A model, from the instance's parameters.
    Model(Made),
    /// A threshold detector, the connect element of a logic input.
    Detector(ThresholdDetectorParameters),
    /// A logic driver, the connect element of a logic output.
    Driver(LogicDriverParameters),
}

/// What a model made a device from: the instance's parameters, read once into the
/// specification its kernel's instance is made of, and as they are written.
pub(crate) struct Made {
    /// The model, which reads the parameters again when a run replaces some of their
    /// numbers.
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

/// The run of the groups of a run's instances, prepared.
pub(crate) type Devices = Runs;

impl Origin {
    /// The device as an instance of a kernel.
    pub(crate) fn kernel(&self) -> Instance<'_> {
        match self {
            Self::Model(made) => made.specification.kernel(),
            Self::Detector(parameters) => Instance::of::<ThresholdDetector>(parameters),
            Self::Driver(parameters) => Instance::of::<LogicDriver>(parameters),
        }
    }
}

impl Placed {
    /// Where the instance is: its path and the script line that placed it.
    #[must_use]
    pub fn location(&self) -> &Location {
        &self.template.location
    }

    /// The instance's path through the hierarchy, such as `amplifier/R1`.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.template.location.path
    }

    /// The name of the device's model, such as `resistor`.
    #[must_use]
    pub fn model(&self) -> &'static str {
        self.template.model
    }

    /// The names of the device's terminals, in the order of [`Placed::nodes`].
    #[must_use]
    pub fn terminals(&self) -> &[String] {
        &self.template.interface.terminals
    }

    /// The device's thermal terminals, by index among [`Placed::terminals`].
    #[must_use]
    pub fn thermal(&self) -> &[usize] {
        &self.template.interface.thermal
    }

    /// The device's branches between its terminals.
    #[must_use]
    pub fn branches(&self) -> &[jolty_models::Branch] {
        &self.template.interface.branches
    }

    /// The node each terminal joins: a node voltage's unknown, or `None` for the reference.
    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.template.nodes
    }

    /// The device's internal unknowns, such as a transistor's internal drain behind its
    /// series resistance, by index among the unknowns.
    #[must_use]
    pub fn internals(&self) -> Range<usize> {
        self.first_internal..self.first_internal + self.internals
    }

    /// The names of the variables the device computes, such as a transistor's
    /// `threshold_voltage`.
    #[must_use]
    pub fn variable_names(&self) -> &'static [&'static str] {
        self.variables
    }

    /// The regions of operation the device's equations distinguish, which its variable
    /// [`REGION`](jolty_models::REGION) numbers, and its margins to their boundaries: none
    /// for a device that is no transistor.
    #[must_use]
    pub const fn regions(&self) -> Regions {
        self.regions
    }

    /// Whether the device was made from its own parameters, at its circuit's temperature,
    /// as a run that replaces none of its numbers makes it.
    pub(crate) const fn nominal(&self) -> bool {
        self.nominal
    }

    /// Whether the device is a foundry's, which no analysis solves.
    #[must_use]
    pub fn foundry(&self) -> bool {
        match &self.template.origin {
            Origin::Model(made) => made.model.foundry(),
            Origin::Detector(_) | Origin::Driver(_) => false,
        }
    }

    /// The parameters a model made the device from, as the elaborated circuit gives them,
    /// or `None` for a connect element. A run whose values replace some of their numbers
    /// makes its device from those numbers instead.
    #[must_use]
    pub fn parameters(&self) -> Option<&Parameters> {
        match &self.template.origin {
            Origin::Model(made) => Some(&made.parameters),
            Origin::Detector(_) | Origin::Driver(_) => None,
        }
    }
}

/// The structure a circuit takes under some values, against that of the elaborated circuit
/// that finds it, [`Elaborated::structure_of`]: each device whose unknowns or entries
/// differ from those the circuit holds, with those it claims under the values. The values
/// of one elaborated circuit share a structure when their keys are equal, and keep the
/// circuit's own when their key is empty.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StructureKey(Vec<(usize, Layout)>);

impl StructureKey {
    /// Whether the values keep the structure of the elaborated circuit that found the key.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The devices whose unknowns or entries the values change, by index among
    /// [`Simulator::devices`](crate::Simulator::devices), in order.
    pub fn devices(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().map(|&(device, _)| device)
    }
}

/// What a device claims: its internal unknowns, numbered from those of the elaborated
/// circuit, the position of each of its entries, in order, and the variables it computes
/// and the regions they number.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Layout {
    internals: Vec<Unknown>,
    entries: Vec<(usize, usize)>,
    variables: &'static [&'static str],
    regions: Regions,
}

/// The devices and the composite terminals of a circuit, each by index, sorted by path and
/// by name, to find one by either.
#[derive(Debug, Clone)]
struct Index {
    devices: Vec<usize>,
    composites: Vec<usize>,
}

impl Index {
    /// The index of `devices` and `composites`, whose paths and names are each unique.
    fn new(devices: &[Placed], composites: &[Composite]) -> Self {
        let mut by_path: Vec<usize> = (0..devices.len()).collect();
        by_path.sort_unstable_by(|&a, &b| devices[a].path().cmp(devices[b].path()));
        let mut by_name: Vec<usize> = (0..composites.len()).collect();
        by_name.sort_unstable_by(|&a, &b| composites[a].name.cmp(&composites[b].name));
        Self {
            devices: by_path,
            composites: by_name,
        }
    }
}

/// What binding devices one after another builds: the unknowns, the entries they claim, the
/// devices placed, and the groups of their instances, with their numbers prepared.
pub(crate) struct Binder {
    unknowns: Vec<Unknown>,
    builder: PatternBuilder,
    node_diagonals: Vec<(UnknownIndex, Entry)>,
    placed: Vec<Placed>,
    groups: Groups,
    runs: Runs,
    temperature: f64,
}

/// What a [`Binder`] built, once every device is bound.
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
    /// A binder of devices at `temperature`, in kelvins, after the node voltages `nodes`,
    /// whose diagonal entries it claims first.
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

    /// Adds the instance of the device of `template` that `origin` makes, its template's own
    /// unless given, to its group, binding it to new internal unknowns and entries;
    /// `nominal` when it is made from its own parameters at its circuit's temperature.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when a parameter is out of range; nothing is placed.
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

    /// The devices bound, with the pattern of the entries they claimed and its ordering.
    ///
    /// # Errors
    ///
    /// Returns [`ElaborationProblem::TooLarge`] when the ordering does not fit in memory, or
    /// the groups' slab more places than they index.
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

/// A circuit elaborated for the structure some values give it, the devices made for them,
/// and the origins of those whose numbers they replace, by device.
pub(crate) type Rebound = (Elaborated, Devices, Vec<Option<Origin>>);

/// What a run's devices are made from: an elaborated circuit, the origins its values
/// give the devices whose numbers they replace, by device, none when they replace none,
/// those of the devices a sweep holds elsewhere, by device, and its temperature.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Origins<'a> {
    pub(crate) elaborated: &'a Elaborated,
    pub(crate) patched: &'a [Option<Origin>],
    pub(crate) swept: &'a BTreeMap<usize, Origin>,
    pub(crate) temperature: f64,
}

impl Origins<'_> {
    /// What the device at `index` is made from.
    pub(crate) fn of(&self, index: usize) -> &Origin {
        self.swept
            .get(&index)
            .or_else(|| self.patched.get(index).and_then(Option::as_ref))
            .unwrap_or(&self.elaborated.devices[index].template.origin)
    }
}

impl Elaborated {
    /// Elaborates `circuit` with the models of `library`, in `structure`, at 27 °C, as a
    /// run takes it whose values replace no number.
    ///
    /// # Errors
    ///
    /// Returns every [`ElaborationProblem`] found: an invalid structure, a missing reference
    /// node, an unknown model, wrongly connected terminals, or invalid parameters.
    pub fn new(
        circuit: &Circuit,
        library: &Library,
        structure: &Structure,
    ) -> Result<Arc<Self>, ElaborationError> {
        let (elaborated, _) = elaborate::elaborate(
            circuit,
            library,
            (&structure.substitutes, &BTreeMap::new()),
            REFERENCE_TEMPERATURE,
        )?;
        Ok(Arc::new(elaborated))
    }

    /// Assembles the circuit from what elaboration found and the devices `bound`, at
    /// `temperature`.
    pub(crate) fn assemble(
        bound: Bound,
        (logic, composites, reference): (LogicSystem, Vec<Composite>, Name),
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
            composites,
            lookup: OnceLock::new(),
            dissected: Mutex::new(BTreeMap::new()),
        };
        (elaborated, bound.devices)
    }

    /// The positions of the largest diagonal block of the minimum degree's ordering.
    pub(crate) fn largest_block(&self) -> usize {
        self.ordering.largest_block()
    }

    /// The ordering a run's solver factors with, as `dissecting` orders the pattern: the
    /// minimum degree's, shared by every run, without `dissecting` or where it dissects no
    /// block, as where no block holds more unknowns than an interior, and else the one it
    /// dissects, made the
    /// first time a run asks for it and shared by every run after. Two runs that ask at
    /// once may each make it, which gives the same ordering, and the first kept is shared.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::TooLarge`] when the ordering does not fit in memory.
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

    /// The unknowns, in the order of every solution's values.
    #[must_use]
    pub fn unknowns(&self) -> &[Unknown] {
        &self.unknowns
    }

    /// The analog devices, with every component instance unfolded, in the order they were
    /// placed, then the connect elements of the logic devices' pins that join electrical
    /// nets.
    #[must_use]
    pub fn devices(&self) -> &[Placed] {
        &self.devices
    }

    /// The parameter of the number at the path `name` among the parameters of the device at
    /// `path`, such as `X1/R1` and `resistance`, or `waveform.dc`, if the circuit writes one
    /// there.
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

    /// The nominal value of every parameter, by its index: the number the circuit writes.
    #[must_use]
    pub fn nominal(&self) -> &[f64] {
        self.lookup().table.nominal()
    }

    /// The values that replace the numbers `overrides` gives, by the device's path then by
    /// the number's path among its parameters, such as `X1/R1` and `resistance`, at
    /// `temperature`, in kelvins, or at the circuit's own with `None`.
    ///
    /// # Errors
    ///
    /// Returns every path the circuit has no device at, every number a device's parameters
    /// do not write, and every number of a logic device, which elaborating the circuit reads
    /// once into the structure of its logic.
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

    /// The structure the circuit takes under `values`: each device whose unknowns or entries
    /// differ, made from the numbers they give at their temperature, from those this
    /// circuit holds.
    ///
    /// # Errors
    ///
    /// Returns every [`ElaborationProblem`] the values make: each parameter the circuit
    /// does not have, and each device whose numbers they put out of range.
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

    /// The circuit elaborated for the structure `values` give it: its devices made from
    /// their numbers at their temperature, each claiming its unknowns and entries afresh,
    /// with the nominal numbers, the temperature and everything else of this circuit's, so
    /// that the same values make the same devices of either.
    ///
    /// # Errors
    ///
    /// Returns every [`ElaborationProblem`] the values make, as
    /// [`Elaborated::structure_of`] does, and [`ElaborationProblem::TooLarge`] when the new
    /// structure's system is too large to factor.
    pub fn restructured(&self, values: &Values) -> Result<Arc<Self>, ElaborationError> {
        let (restructured, _, _) = self.rebind(values)?;
        Ok(Arc::new(restructured))
    }

    /// The circuit elaborated for the structure `values` give it, as
    /// [`Elaborated::restructured`] says, with the devices made for them and the origins
    /// of those whose numbers they replace.
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
            composites: self.composites.clone(),
            lookup: self.lookup.clone(),
            dissected: Mutex::new(BTreeMap::new()),
        };
        Ok((restructured, bound.devices, patched))
    }

    /// The temperature its devices are made at, in kelvins, unless a run's values give
    /// another.
    #[must_use]
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// The temperature a run of `values` makes its devices at, in kelvins.
    pub(crate) fn temperature_of(&self, values: &Values) -> f64 {
        values.temperature.unwrap_or(self.temperature)
    }

    /// What each device is made from under `values`, by device: those whose numbers they
    /// replace, read again from their parameters with the numbers replaced, and `None` for
    /// the others, which are made from their own; none at all when they replace none.
    ///
    /// # Errors
    ///
    /// Returns every [`ElaborationProblem`] the values make: a parameter the circuit does not
    /// have, a parameter its model no longer reads, a number that is not finite, or numbers
    /// that change the device's terminals.
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

    /// The numbers `values` replace among those of the device at `index`, in order.
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

    /// What the device at `index`, an independent source, is made from held at `level`:
    /// its model's device from the parameters of a source of that constant level.
    ///
    /// # Errors
    ///
    /// Returns the problem of a device its model does not make from those parameters, or
    /// whose terminals they change.
    pub(crate) fn held_at(
        &self,
        index: usize,
        level: f64,
    ) -> Result<Origin, Box<ElaborationProblem>> {
        let placed = &self.devices[index];
        let at = || placed.location().clone();
        let Origin::Model(made) = &placed.template.origin else {
            return Err(Box::new(ElaborationProblem::Parameters {
                at: at(),
                message: "a connect element is held at no level".to_owned(),
            }));
        };
        let parameters = jolty_models::held_source(level);
        self.read(index, made, parameters)
    }

    /// What the device at `index` is made from with the numbers `patches` replace, each a
    /// parameter of it.
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
                message: format!("the value of `{name}` must be finite, not {value:?}"),
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

    /// What the model of `made`, the origin of the device at `index`, makes of `parameters`.
    ///
    /// # Errors
    ///
    /// Returns the problem of parameters the model does not read, or that change the
    /// device's terminals.
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

    /// What the device at `index` claims when its instance is made from `origin` at
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

    /// The problem of the device at `index` that `error` describes.
    pub(crate) fn problem(&self, index: usize, error: ParameterError) -> ElaborationProblem {
        ElaborationProblem::Parameters {
            at: self.devices[index].location().clone(),
            message: error.0,
        }
    }

    /// The index of the devices' paths and the table of their numbers.
    fn lookup(&self) -> &Lookup {
        self.lookup
            .get_or_init(|| Lookup::new(&self.devices, &self.composites))
    }

    /// Where the device at `path` is: its path and the script line that placed it.
    pub(crate) fn location(&self, path: &str) -> Option<Location> {
        Some(self.devices[self.device(path)?].location().clone())
    }

    /// The device at `path`, by its index.
    pub(crate) fn device(&self, path: &str) -> Option<usize> {
        let found = self
            .lookup()
            .index
            .devices
            .binary_search_by(|&device| self.devices[device].path().cmp(path))
            .ok()?;
        Some(self.lookup().index.devices[found])
    }

    /// The node that the terminal `terminal` of the device, component instance or logic
    /// device at `instance` joins, and whether it is thermal, as
    /// [`Simulator::terminal`](crate::Simulator::terminal) says.
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
        let composites = &self.composites;
        let found = self
            .lookup()
            .index
            .composites
            .binary_search_by(|&composite| composites[composite].name.cmp(&name))
            .ok()?;
        let composite = &composites[self.lookup().index.composites[found]];
        Some((composite.node, composite.thermal))
    }
}

/// The number `value` written where the parameters write `written`: as an integer where
/// they write one and `value` is a whole number an integer holds exactly, so that a model
/// reads it as the circuit's own, and else as a real number.
fn written(written: &Value, value: f64) -> Value {
    /// The largest magnitude below which every whole number is exact in a `f64`: 2⁵³.
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
