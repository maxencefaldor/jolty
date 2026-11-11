//! Elaboration: flattening a circuit's hierarchy into devices bound to unknowns.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::sync::Arc;

use jolty_ir::{Circuit, Component, Name, Of, Parameters, Problem, Role, Source};
use jolty_models::{
    Binding, Branch, Device, Interface, Library, LogicDriver, LogicDriverParameters, LogicModel,
    Model, NativeDevice, Node, Pins, Quantity, ThresholdDetector, ThresholdDetectorParameters,
    Unknown, UnknownInfo,
};
use jolty_numerics::{Entry, Pattern, PatternBuilder};

use crate::logic::{Endpoint, LogicNet, LogicSystem, PlacedLogic};
use crate::{Conditions, Substitute};

/// A circuit flattened into devices bound to unknowns, with the pattern of its matrix, and
/// logic devices joined by logic nets.
///
/// The node voltages come first among the unknowns, in the order their nets are first met,
/// then the devices' internal unknowns. A net that only logic pins join is a logic net; where
/// a logic pin joins an electrical net, a connect element joins it to a logic net of its own.
#[derive(Debug)]
pub(crate) struct System {
    pub(crate) unknowns: Vec<UnknownInfo>,
    pub(crate) devices: Vec<Placed>,
    pub(crate) logic: LogicSystem,
    /// The name of the reference node: the top component's ground port.
    pub(crate) reference: Name,
    pub(crate) pattern: Arc<Pattern>,
    /// The diagonal entry of every node voltage's equation, where gmin stepping adds a
    /// conductance to the reference.
    pub(crate) node_diagonals: Vec<(Unknown, Entry)>,
    /// The devices' temperature, in kelvins.
    pub(crate) temperature: f64,
}

/// A device placed in the flattened circuit: where its instance is, what model it is, and
/// the node each of its terminals joins.
#[derive(Debug)]
pub struct Placed {
    location: Location,
    model: &'static str,
    interface: Interface,
    nodes: Vec<Node>,
    /// What the device was made from, to make it again; none for a connect element.
    pub(crate) made: Option<Made>,
    /// How many unknowns there were before the device added its own.
    pub(crate) first_internal: usize,
    pub(crate) device: Box<dyn Device>,
}

/// The model and the parameters a device was made from.
pub(crate) struct Made {
    pub(crate) model: Arc<dyn Model>,
    pub(crate) parameters: Parameters,
}

impl fmt::Debug for Made {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Made")
            .field("model", &self.model.name())
            .field("parameters", &self.parameters)
            .finish()
    }
}

impl Placed {
    /// Where the instance is: its path and the script line that placed it.
    #[must_use]
    pub fn location(&self) -> &Location {
        &self.location
    }

    /// The instance's path through the hierarchy, such as `amplifier/R1`.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.location.path
    }

    /// The name of the device's model, such as `resistor`.
    #[must_use]
    pub fn model(&self) -> &'static str {
        self.model
    }

    /// The names of the device's terminals, in the order of [`Placed::nodes`].
    #[must_use]
    pub fn terminals(&self) -> &[String] {
        &self.interface.terminals
    }

    /// The device's branches between its terminals.
    #[must_use]
    pub fn branches(&self) -> &[Branch] {
        &self.interface.branches
    }

    /// The node each terminal joins: a node voltage's unknown, or `None` for the reference.
    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
}

/// Where an instance is: its hierarchical path and the source line that placed it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Location {
    /// The instance's path through the hierarchy, such as `amplifier/R1`.
    pub path: String,
    /// The line of the build script that placed it.
    pub source: Option<Source>,
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "`{}`", self.path)?;
        if let Some(source) = &self.source {
            write!(f, " ({}:{})", source.file, source.line)?;
        }
        Ok(())
    }
}

/// A reason a circuit cannot be elaborated.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ElaborationProblem {
    /// The circuit's structure is invalid.
    #[error(transparent)]
    Structure(#[from] Problem),
    /// The top component has no ground port, so the circuit has no reference node.
    #[error("the top component `{0}` has no ground port to be the reference node")]
    NoReference(Name),
    /// The top component has several ground ports.
    #[error("the top component `{top}` has {count} ground ports, and the reference is one node")]
    SeveralReferences {
        /// The top component.
        top: Name,
        /// How many ground ports it declares.
        count: usize,
    },
    /// An instance is of a model the library does not hold.
    #[error("instance {at} is of unknown model `{model}`")]
    UnknownModel {
        /// The instance.
        at: Location,
        /// The model's name.
        model: Name,
    },
    /// An instance leaves a terminal of its model unconnected.
    #[error("instance {at} leaves terminal `{terminal}` of its {model} unconnected")]
    UnconnectedTerminal {
        /// The instance.
        at: Location,
        /// The model's name.
        model: Name,
        /// The terminal.
        terminal: String,
    },
    /// An instance connects a terminal its model does not have.
    #[error("instance {at} connects `{terminal}`, which a {model} does not have")]
    UnknownTerminal {
        /// The instance.
        at: Location,
        /// The model's name.
        model: Name,
        /// The terminal.
        terminal: Name,
    },
    /// An instance's parameters are invalid for its model.
    #[error("instance {at}: {message}")]
    Parameters {
        /// The instance.
        at: Location,
        /// What is wrong with the parameters.
        message: String,
    },
    /// A logic pin joins an electrical net, and its model cannot cross to one.
    #[error(
        "instance {at} joins its logic pin `{pin}` to the electrical net `{net}`, and a {model} \
         has no supply and ground to cross to it"
    )]
    LogicPinOnElectricalNet {
        /// The instance.
        at: Location,
        /// The model's name.
        model: Name,
        /// The pin.
        pin: String,
        /// The net's path.
        net: String,
    },
    /// A model stands in for a component instance the circuit does not have.
    #[error("the circuit has no component instance at `{0}`, for which a model stands in")]
    UnknownSubstitute(String),
    /// Parameters replace those of an instance the circuit does not have.
    #[error("the circuit has no device at `{0}`, whose parameters are replaced")]
    UnknownOverride(String),
    /// The circuit's system is too large to factor.
    #[error("the circuit's system is too large to factor")]
    TooLarge,
}

/// Every [`ElaborationProblem`] of a circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElaborationError(Vec<ElaborationProblem>);

impl ElaborationError {
    /// The problems, in the order they were found.
    #[must_use]
    pub fn problems(&self) -> &[ElaborationProblem] {
        &self.0
    }
}

impl From<ElaborationProblem> for ElaborationError {
    fn from(problem: ElaborationProblem) -> Self {
        Self(vec![problem])
    }
}

impl fmt::Display for ElaborationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, problem) in self.0.iter().enumerate() {
            if index > 0 {
                writeln!(f)?;
            }
            write!(f, "{problem}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ElaborationError {}

/// Flattens `circuit` and binds each device to its unknowns in `conditions`.
pub(crate) fn elaborate(
    circuit: &Circuit,
    library: &Library,
    conditions: &Conditions,
) -> Result<System, ElaborationError> {
    circuit.validate().map_err(|problems| {
        ElaborationError(
            problems
                .iter()
                .cloned()
                .map(ElaborationProblem::from)
                .collect(),
        )
    })?;
    let top = &circuit.components[&circuit.top];
    // Without a single reference, the walk numbers every net and goes on to find the
    // instances' problems, so the error lists them all.
    let (ground, problems) = reference(circuit);
    let reference: BTreeMap<&Name, NetRef> = ground.iter().map(|&name| (name, None)).collect();
    let mut walk = Walk {
        circuit,
        library,
        overrides: &conditions.overrides,
        overridden: BTreeSet::new(),
        substitutes: &conditions.substitutes,
        substituted: BTreeSet::new(),
        nets: Vec::new(),
        named: HashMap::new(),
        devices: Vec::new(),
        logic: Vec::new(),
        problems,
    };
    walk.component(top, "", &reference);
    let Walk {
        nets,
        devices,
        logic,
        mut problems,
        overridden,
        substituted,
        ..
    } = walk;

    let (mut unknowns, classes, mut partition) = classify(&nets, ground_name(circuit));
    let mut builder = PatternBuilder::new();
    let node_diagonals: Vec<(Unknown, Entry)> = (0..unknowns.len())
        .map(|index| (Unknown::new(index), builder.entry(index, index)))
        .collect();
    let mut placed = Vec::with_capacity(devices.len());
    let mut elements = Elements {
        unknowns: &mut unknowns,
        builder: &mut builder,
        placed: &mut placed,
        temperature: conditions.temperature,
        detected: Vec::new(),
        partition: &mut partition,
        problems: &mut problems,
    };
    for pending in devices {
        elements.analog(pending, &classes);
    }
    for pending in logic {
        elements.logic(pending, &classes);
    }
    let detected = std::mem::take(&mut elements.detected);
    partition.index(&detected, &placed);
    problems.extend(
        conditions
            .substitutes
            .keys()
            .filter(|path| !substituted.contains(path.as_str()))
            .map(|path| ElaborationProblem::UnknownSubstitute(path.clone())),
    );
    problems.extend(
        conditions
            .overrides
            .keys()
            .filter(|path| !overridden.contains(path.as_str()))
            .map(|path| ElaborationProblem::UnknownOverride(path.clone())),
    );
    // Without a single ground, a problem says so.
    let (Some(ground), true) = (ground, problems.is_empty()) else {
        return Err(ElaborationError(problems));
    };
    let pattern = Arc::new(builder.finish(unknowns.len()));
    Ok(System {
        unknowns,
        devices: placed,
        logic: partition,
        reference: ground.clone(),
        pattern,
        node_diagonals,
        temperature: conditions.temperature,
    })
}

/// The top component's ground port, the reference, or the problem that it has none or
/// several.
fn reference(circuit: &Circuit) -> (Option<&Name>, Vec<ElaborationProblem>) {
    let grounds: Vec<&Name> = circuit.components[&circuit.top]
        .ports
        .iter()
        .filter(|port| port.role == Role::Ground)
        .map(|port| &port.name)
        .collect();
    match grounds.as_slice() {
        [ground] => (Some(*ground), Vec::new()),
        [] => (
            None,
            vec![ElaborationProblem::NoReference(circuit.top.clone())],
        ),
        _ => (
            None,
            vec![ElaborationProblem::SeveralReferences {
                top: circuit.top.clone(),
                count: grounds.len(),
            }],
        ),
    }
}

/// Classifies the walk's `nets`: those an electrical terminal joins become node voltages, in
/// the order they were first met, and the others logic nets. Returns the node voltages, the
/// classes, and the logic partition with its nets.
fn classify(nets: &[NetRecord], ground: String) -> (Vec<UnknownInfo>, Classes<'_>, LogicSystem) {
    let mut unknowns = Vec::new();
    let mut nodes: Vec<Node> = vec![None; nets.len()];
    let mut logic: Vec<Option<usize>> = vec![None; nets.len()];
    let mut partition = LogicSystem::default();
    for (index, net) in nets.iter().enumerate() {
        if net.electrical {
            nodes[index] = Some(Unknown::new(unknowns.len()));
            unknowns.push(UnknownInfo {
                name: net.name.clone(),
                quantity: Quantity::Voltage,
            });
        } else {
            logic[index] = Some(partition.nets.len());
            partition.nets.push(LogicNet {
                name: net.name.clone(),
                sources: Vec::new(),
                loads: Vec::new(),
                load: 0.0,
            });
        }
    }
    let classes = Classes {
        nets,
        nodes,
        logic,
        ground,
    };
    (unknowns, classes, partition)
}

/// The name of the reference net, the top component's ground port's.
fn ground_name(circuit: &Circuit) -> String {
    circuit.components[&circuit.top]
        .ports
        .iter()
        .find(|port| port.role == Role::Ground)
        .map_or_else(String::new, |port| port.name.to_string())
}

/// How the walk's nets were classified.
struct Classes<'a> {
    nets: &'a [NetRecord],
    /// The node of each electrical net.
    nodes: Vec<Node>,
    /// The logic net of each net only logic pins join.
    logic: Vec<Option<usize>>,
    /// The reference net's name.
    ground: String,
}

impl Classes<'_> {
    fn node(&self, net: NetRef) -> Node {
        net.and_then(|index| self.nodes[index])
    }
}

/// What logic devices and their connect elements are placed into.
struct Elements<'a> {
    unknowns: &'a mut Vec<UnknownInfo>,
    builder: &'a mut PatternBuilder,
    placed: &'a mut Vec<Placed>,
    temperature: f64,
    /// Each detector's input node and capacitance.
    detected: Vec<(Node, f64)>,
    partition: &'a mut LogicSystem,
    problems: &'a mut Vec<ElaborationProblem>,
}

impl Elements<'_> {
    /// Binds the analog device `pending` to the nodes of its terminals.
    fn analog(&mut self, pending: Pending<'_>, classes: &Classes<'_>) {
        let terminals: Vec<Node> = pending
            .terminals
            .iter()
            .map(|&net| classes.node(net))
            .collect();
        let first_internal = self.unknowns.len();
        let mut binding = Binding::new(
            &pending.at.path,
            &terminals,
            self.unknowns,
            self.builder,
            self.temperature,
        );
        match pending.model.instantiate(&pending.parameters, &mut binding) {
            Ok(device) => self.placed.push(Placed {
                location: pending.at,
                model: pending.model.name(),
                interface: pending.interface,
                nodes: terminals,
                made: Some(Made {
                    model: pending.model,
                    parameters: pending.parameters.into_owned(),
                }),
                first_internal,
                device,
            }),
            Err(error) => self.problems.push(ElaborationProblem::Parameters {
                at: pending.at,
                message: error.0,
            }),
        }
    }

    /// Places the logic device `pending`: each pin on a net only logic pins join joins its
    /// logic net, and each pin on an electrical net joins a logic net of its own, which a
    /// connect element joins to the electrical net.
    fn logic(&mut self, pending: PendingLogic<'_>, classes: &Classes<'_>) {
        let device = match pending.model.instantiate(&pending.parameters) {
            Ok(device) => device,
            Err(error) => {
                self.problems.push(ElaborationProblem::Parameters {
                    at: pending.at,
                    message: error.0,
                });
                return;
            }
        };
        let index = self.partition.devices.len();
        let Pins {
            inputs, outputs, ..
        } = &pending.pins;
        let rails = pending.pins.electrical.as_ref().map(|electrical| {
            let rails = &pending.nets[inputs.len() + outputs.len()..];
            (
                electrical.threshold,
                classes.node(rails[0]),
                classes.node(rails[1]),
            )
        });
        let mut ends = [Vec::new(), Vec::new()];
        for (side, pins) in [inputs, outputs].into_iter().enumerate() {
            let input = side == 0;
            let offset = if input { 0 } else { inputs.len() };
            for (pin, name) in pins.iter().enumerate() {
                let net = pending.nets[offset + pin];
                let end = Endpoint::Pin {
                    device: index,
                    index: pin,
                };
                let capacitance = if input {
                    device.input_capacitance(pin)
                } else {
                    0.0
                };
                if let Some(logic_net) = net.and_then(|net| classes.logic[net]) {
                    let joined = &mut self.partition.nets[logic_net];
                    if input {
                        joined.loads.push(end);
                        joined.load += capacitance;
                    } else {
                        joined.sources.push(end);
                    }
                    ends[side].push(logic_net);
                    continue;
                }
                let Some((threshold, supply, ground)) = rails else {
                    self.problems
                        .push(ElaborationProblem::LogicPinOnElectricalNet {
                            at: pending.at.clone(),
                            model: pending.name.clone(),
                            pin: name.clone(),
                            net: net.map_or_else(
                                || classes.ground.clone(),
                                |net| classes.nets[net].name.clone(),
                            ),
                        });
                    continue;
                };
                let at = Location {
                    path: format!("{}:{name}", pending.at.path),
                    source: pending.at.source.clone(),
                };
                let own = self.partition.nets.len();
                let terminals = [classes.node(net), supply, ground];
                let element = if input {
                    self.detector(at.clone(), terminals, threshold, capacitance)
                } else {
                    self.driver(at.clone(), terminals)
                };
                let element_end = Endpoint::Element { device: element };
                let (sources, loads) = if input {
                    self.partition.detectors.push((element, own));
                    (vec![element_end], vec![end])
                } else {
                    self.partition.drivers.push((element, own));
                    (vec![end], vec![element_end])
                };
                self.partition.nets.push(LogicNet {
                    name: at.path,
                    sources,
                    loads,
                    load: capacitance,
                });
                ends[side].push(own);
            }
        }
        let [inputs, outputs] = ends;
        let nodes = rails.map_or_else(Vec::new, |(_, supply, ground)| vec![supply, ground]);
        self.partition.devices.push(PlacedLogic {
            location: pending.at,
            model: pending.model.name(),
            pins: pending.pins,
            inputs,
            outputs,
            nodes,
            device,
        });
    }

    /// Places a threshold detector at `at` on `terminals`: its input, supply and ground.
    fn detector(
        &mut self,
        at: Location,
        terminals: [Node; 3],
        threshold: f64,
        capacitance: f64,
    ) -> usize {
        let parameters = ThresholdDetectorParameters {
            threshold,
            capacitance,
        };
        self.detected.push((terminals[0], capacitance));
        let interface = ThresholdDetector::interface(&parameters);
        self.place::<ThresholdDetector>(at, terminals, interface, parameters)
    }

    /// Places a logic driver at `at` on `terminals`: its output, supply and ground.
    fn driver(&mut self, at: Location, terminals: [Node; 3]) -> usize {
        let parameters = LogicDriverParameters {};
        let interface = LogicDriver::interface(&parameters);
        self.place::<LogicDriver>(at, terminals, interface, parameters)
    }

    fn place<D: NativeDevice + Device>(
        &mut self,
        at: Location,
        terminals: [Node; 3],
        interface: Interface,
        parameters: D::Parameters,
    ) -> usize {
        let first_internal = self.unknowns.len();
        let mut binding = Binding::new(
            &at.path,
            &terminals,
            self.unknowns,
            self.builder,
            self.temperature,
        );
        let device: Box<dyn Device> = match D::new(parameters, &mut binding) {
            Ok(device) => Box::new(device),
            Err(error) => unreachable!("a connect element's parameters are checked: {error}"),
        };
        self.placed.push(Placed {
            location: at,
            model: D::NAME,
            interface,
            nodes: terminals.to_vec(),
            made: None,
            first_internal,
            device,
        });
        self.placed.len() - 1
    }
}

/// A net as the walk finds it, by its index in the walk's nets, or `None` for the reference.
type NetRef = Option<usize>;

/// A net found by the walk.
struct NetRecord {
    /// The net's path.
    name: String,
    /// Whether an electrical terminal joins it.
    electrical: bool,
}

/// A device instance found by the walk, waiting for every net to be classified.
struct Pending<'a> {
    at: Location,
    model: Arc<dyn Model>,
    /// The instance's parameters, with those the conditions replace.
    parameters: Cow<'a, Parameters>,
    interface: Interface,
    terminals: Vec<NetRef>,
}

/// A logic device instance found by the walk.
struct PendingLogic<'a> {
    at: Location,
    name: Name,
    model: &'a dyn LogicModel,
    parameters: Cow<'a, Parameters>,
    pins: Pins,
    /// The net of each pin and terminal, in the order of [`Pins::names`].
    nets: Vec<NetRef>,
}

/// A device the walk places: the nets its terminals join in their component, its
/// parameters, where it is, and its model's name.
struct Placement<'a, 'm> {
    terminals: &'a BTreeMap<Name, Name>,
    parameters: Cow<'a, Parameters>,
    at: Location,
    model: &'m Name,
}

/// The state of a walk down the hierarchy.
struct Walk<'a> {
    circuit: &'a Circuit,
    library: &'a Library,
    overrides: &'a BTreeMap<String, Parameters>,
    /// The paths of the devices whose parameters the overrides replace.
    overridden: BTreeSet<String>,
    substitutes: &'a BTreeMap<String, Substitute>,
    /// The paths of the component instances a model stands in for.
    substituted: BTreeSet<String>,
    nets: Vec<NetRecord>,
    named: HashMap<String, usize>,
    devices: Vec<Pending<'a>>,
    logic: Vec<PendingLogic<'a>>,
    problems: Vec<ElaborationProblem>,
}

impl<'a> Walk<'a> {
    /// Walks `component`, whose instances' paths start with `prefix` and whose ports connect
    /// to the nets in `ports`. The circuit has been validated, so every referenced component
    /// exists and every port of a component instance is connected.
    fn component(
        &mut self,
        component: &'a Component,
        prefix: &str,
        ports: &BTreeMap<&'a Name, NetRef>,
    ) {
        for instance in &component.instances {
            let at = Location {
                path: format!("{prefix}{}", instance.id),
                source: instance.source.clone(),
            };
            match &instance.of {
                Of::Device(model) => {
                    let parameters = match self.overrides.get(&at.path) {
                        Some(replaced) => {
                            self.overridden.insert(at.path.clone());
                            let mut merged = instance.parameters.clone();
                            merged.extend(replaced.clone());
                            Cow::Owned(merged)
                        }
                        None => Cow::Borrowed(&instance.parameters),
                    };
                    let placement = Placement {
                        terminals: &instance.terminals,
                        parameters,
                        at,
                        model,
                    };
                    self.device(placement, prefix, ports);
                }
                Of::Component(_) if self.substitutes.contains_key(&at.path) => {
                    let substitute = &self.substitutes[&at.path];
                    self.substituted.insert(at.path.clone());
                    let placement = Placement {
                        terminals: &instance.terminals,
                        parameters: Cow::Borrowed(&substitute.parameters),
                        at,
                        model: &substitute.model,
                    };
                    self.device(placement, prefix, ports);
                }
                Of::Component(target) => {
                    let definition = &self.circuit.components[target];
                    let inner: BTreeMap<&Name, NetRef> = definition
                        .ports
                        .iter()
                        .map(|port| {
                            (
                                &port.name,
                                self.net(prefix, ports, &instance.terminals[&port.name]),
                            )
                        })
                        .collect();
                    self.component(definition, &format!("{}/", at.path), &inner);
                }
            }
        }
    }

    /// Places the device of `placement`, analog or logic, in the component at `prefix`.
    fn device(
        &mut self,
        placement: Placement<'a, '_>,
        prefix: &str,
        ports: &BTreeMap<&'a Name, NetRef>,
    ) {
        let Placement {
            terminals: connected,
            parameters,
            at,
            model: model_name,
        } = placement;
        if let Some(model) = self.library.shared(model_name.as_str()) {
            let interface = match model.interface(&parameters) {
                Ok(interface) => interface,
                Err(error) => {
                    self.problems.push(ElaborationProblem::Parameters {
                        at,
                        message: error.0,
                    });
                    return;
                }
            };
            let Some(terminals) = self.terminals(
                connected,
                &at,
                model_name,
                &interface.terminals,
                prefix,
                ports,
            ) else {
                return;
            };
            for &net in terminals.iter().flatten() {
                self.nets[net].electrical = true;
            }
            self.devices.push(Pending {
                at,
                model,
                parameters,
                interface,
                terminals,
            });
        } else if let Some(model) = self.library.logic(model_name.as_str()) {
            let pins = match model.pins(&parameters) {
                Ok(pins) => pins,
                Err(error) => {
                    self.problems.push(ElaborationProblem::Parameters {
                        at,
                        message: error.0,
                    });
                    return;
                }
            };
            let Some(nets) =
                self.terminals(connected, &at, model_name, &pins.names(), prefix, ports)
            else {
                return;
            };
            let pin_count = pins.inputs.len() + pins.outputs.len();
            for &net in nets[pin_count..].iter().flatten() {
                self.nets[net].electrical = true;
            }
            self.logic.push(PendingLogic {
                at,
                name: model_name.clone(),
                model,
                parameters,
                pins,
                nets,
            });
        } else {
            self.problems.push(ElaborationProblem::UnknownModel {
                at,
                model: model_name.clone(),
            });
        }
    }

    /// The net of each of `expected`, the terminals a model declares, as `connected` joins
    /// them, or `None` after reporting a terminal left unconnected or that it does not have.
    fn terminals(
        &mut self,
        connected: &BTreeMap<Name, Name>,
        at: &Location,
        model: &Name,
        expected: &[String],
        prefix: &str,
        ports: &BTreeMap<&'a Name, NetRef>,
    ) -> Option<Vec<NetRef>> {
        let mut complete = true;
        for terminal in connected.keys() {
            if !expected.iter().any(|name| name == terminal.as_str()) {
                complete = false;
                self.problems.push(ElaborationProblem::UnknownTerminal {
                    at: at.clone(),
                    model: model.clone(),
                    terminal: terminal.clone(),
                });
            }
        }
        let mut terminals = Vec::with_capacity(expected.len());
        for terminal in expected {
            if let Some(net) = connected.get(terminal.as_str()) {
                terminals.push(self.net(prefix, ports, net));
            } else {
                complete = false;
                self.problems.push(ElaborationProblem::UnconnectedTerminal {
                    at: at.clone(),
                    model: model.clone(),
                    terminal: terminal.clone(),
                });
            }
        }
        complete.then_some(terminals)
    }

    /// The net `net` of the component at `prefix`: a port's net from the parent, or the
    /// component's own net, recorded when first met.
    fn net(&mut self, prefix: &str, ports: &BTreeMap<&'a Name, NetRef>, net: &Name) -> NetRef {
        if let Some(&found) = ports.get(net) {
            return found;
        }
        let name = format!("{prefix}{net}");
        if let Some(&index) = self.named.get(&name) {
            return Some(index);
        }
        let index = self.nets.len();
        self.nets.push(NetRecord {
            name: name.clone(),
            electrical: false,
        });
        self.named.insert(name, index);
        Some(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn elaborate_json(components: &str) -> Result<System, ElaborationError> {
        let header = r#""format": "jolty-circuit", "version": 1, "top": "top""#;
        let text = format!(r#"{{{header}, "components": {components}}}"#);
        elaborate(
            &Circuit::from_json(&text).unwrap(),
            &Library::standard(),
            &Conditions::default(),
        )
    }

    #[test]
    fn flattens_the_hierarchy_and_names_unknowns_by_path() {
        let system = elaborate_json(
            r#"{
                "top": {"ports": [{"name": "gnd", "role": "ground"}], "instances": [
                    {"id": "V1", "of": {"device": "voltage_source"},
                     "parameters": {"waveform": {"dc": 1}},
                     "terminals": {"positive": "in", "negative": "gnd"}},
                    {"id": "X1", "of": {"component": "cell"}, "terminals": {"a": "in", "b": "gnd"}}
                ]},
                "cell": {
                    "ports": [{"name": "a", "role": "signal"}, {"name": "b", "role": "signal"}],
                    "instances": [
                        {"id": "R1", "of": {"device": "resistor"}, "parameters": {"resistance": 1},
                         "terminals": {"a": "a", "b": "mid"}},
                        {"id": "R2", "of": {"device": "resistor"}, "parameters": {"resistance": 1},
                         "terminals": {"a": "mid", "b": "b"}}
                    ]}
            }"#,
        )
        .unwrap();
        let names: Vec<&str> = system
            .unknowns
            .iter()
            .map(|unknown| unknown.name.as_str())
            .collect();
        assert_eq!(names, ["in", "X1/mid", "V1:current"]);
        let paths: Vec<&str> = system.devices.iter().map(Placed::path).collect();
        assert_eq!(paths, ["V1", "X1/R1", "X1/R2"]);
    }

    #[test]
    fn overrides_replace_parameters_by_path() {
        let text = r#"{"format": "jolty-circuit", "version": 1, "top": "top", "components": {
            "top": {"ports": [{"name": "gnd", "role": "ground"}], "instances": [
                {"id": "X1", "of": {"component": "cell"}, "terminals": {"a": "gnd"}},
                {"id": "X2", "of": {"component": "cell"}, "terminals": {"a": "gnd"}}
            ]},
            "cell": {"ports": [{"name": "a", "role": "signal"}], "instances": [
                {"id": "R1", "of": {"device": "resistor"}, "parameters": {"resistance": 1},
                 "terminals": {"a": "x", "b": "a"}}
            ]}
        }}"#;
        let circuit = Circuit::from_json(text).unwrap();
        let mut conditions = Conditions::default();
        conditions.overrides.insert(
            "X2/R1".to_owned(),
            serde_json::json!({"resistance": -1})
                .as_object()
                .cloned()
                .unwrap(),
        );
        conditions
            .overrides
            .insert("X3/R1".to_owned(), Parameters::new());
        let error = elaborate(&circuit, &Library::standard(), &conditions).unwrap_err();
        assert_eq!(
            error.to_string(),
            "instance `X2/R1`: resistance must be positive, not -1.0\n\
             the circuit has no device at `X3/R1`, whose parameters are replaced"
        );
    }

    #[test]
    fn reports_every_instance_problem_with_its_source_line() {
        let error = elaborate_json(
            r#"{"top": {"ports": [{"name": "gnd", "role": "ground"}], "instances": [
                {"id": "R1", "of": {"device": "resistor"}, "parameters": {"resistance": -1},
                 "terminals": {"a": "x", "b": "gnd"}, "source": {"file": "amp.py", "line": 9}},
                {"id": "Q1", "of": {"device": "transistor"}, "terminals": {}},
                {"id": "C1", "of": {"device": "capacitor"}, "parameters": {"capacitance": 1},
                 "terminals": {"a": "x", "c": "gnd"}}
            ]}}"#,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "instance `Q1` is of unknown model `transistor`\n\
             instance `C1` connects `c`, which a capacitor does not have\n\
             instance `C1` leaves terminal `b` of its capacitor unconnected\n\
             instance `R1` (amp.py:9): resistance must be positive, not -1.0"
        );
    }

    #[test]
    fn requires_exactly_one_reference() {
        let error = elaborate_json(r#"{"top": {"instances": []}}"#).unwrap_err();
        assert_eq!(
            error.problems(),
            [ElaborationProblem::NoReference(Name::new("top").unwrap())]
        );
        let error = elaborate_json(
            r#"{"top": {"ports": [{"name": "a", "role": "ground"}, {"name": "b", "role": "ground"}],
                "instances": []}}"#,
        )
        .unwrap_err();
        assert_eq!(
            error.problems(),
            [ElaborationProblem::SeveralReferences {
                top: Name::new("top").unwrap(),
                count: 2
            }]
        );
    }

    #[test]
    fn reports_the_instances_problems_without_a_reference() {
        let error = elaborate_json(
            r#"{"top": {"instances": [
                {"id": "R1", "of": {"device": "resistor"}, "parameters": {"resistance": -1},
                 "terminals": {"a": "x", "b": "y"}}
            ]}}"#,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "the top component `top` has no ground port to be the reference node\n\
             instance `R1`: resistance must be positive, not -1.0"
        );
    }
}
