//! Elaboration: flattening a circuit's hierarchy into devices bound to unknowns.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::sync::Arc;

use jolty_ir::{Circuit, Component, Name, Of, Parameters, Problem, Role, Source, set_parameter};
use jolty_models::{
    Interface, Kernel, Library, Logic, LogicDriver, LogicDriverParameters, LogicModel, Model,
    Nature, Node, Pins, Specification, ThresholdDetector, ThresholdDetectorParameters, Unknown,
    UnknownInfo,
};

use crate::Substitute;
use crate::elaborated::{Binder, Devices, Elaborated, Made, Origin, Template};
use crate::logic::{Endpoint, LogicNet, LogicSystem, PlacedLogic};

/// A terminal of a component instance or a logic device, whose current and charge are those
/// of device terminals summed: a port's are those of every device terminal inside its
/// instance that joins the port's net there, and a logic pin's those of its connect
/// element's terminal on the pin's electrical net.
///
/// A port on a thermal net is thermal, its current the heat flowing into it. The reference
/// is the ambient as well as the ground, so a port on it sums its electrical terminals
/// only: heat into the ambient is no current.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Composite {
    /// Its instance's path and its own name, such as `X1.vdd`.
    pub(crate) name: String,
    /// Its instance's path, such as `X1`.
    pub(crate) owner: String,
    /// The node it joins, or `None` for the reference.
    pub(crate) node: Node,
    /// Whether it is on a thermal net.
    pub(crate) thermal: bool,
    /// Each device terminal summed, by the device's index and the terminal's.
    pub(crate) members: Vec<(usize, usize)>,
}

impl Composite {
    /// Its instance's path and its own name, such as `X1.vdd`.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The path of its component instance or logic device, such as `X1`.
    #[must_use]
    pub fn owner(&self) -> &str {
        &self.owner
    }

    /// The node it joins: a node voltage's unknown, or `None` for the reference.
    #[must_use]
    pub fn node(&self) -> Node {
        self.node
    }

    /// Whether it is on a thermal net, its current the heat flowing into it.
    #[must_use]
    pub fn thermal(&self) -> bool {
        self.thermal
    }

    /// Each device terminal whose current and charge it sums, by the device's index among
    /// [`Simulator::devices`](crate::Simulator::devices) and the terminal's among its own.
    #[must_use]
    pub fn members(&self) -> &[(usize, usize)] {
        &self.members
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElaborationProblem {
    /// The circuit's structure is invalid.
    Structure(Problem),
    /// The top component has no ground port, so the circuit has no reference node.
    NoReference(Name),
    /// The top component has several ground ports.
    SeveralReferences {
        /// The top component.
        top: Name,
        /// How many ground ports it declares.
        count: usize,
    },
    /// An instance is of a model the library does not hold.
    UnknownModel {
        /// The instance.
        at: Location,
        /// The model's name.
        model: Name,
    },
    /// An instance leaves a terminal of its model unconnected.
    UnconnectedTerminal {
        /// The instance.
        at: Location,
        /// The model's name.
        model: Name,
        /// The terminal.
        terminal: String,
    },
    /// An instance connects a terminal its model does not have.
    UnknownTerminal {
        /// The instance.
        at: Location,
        /// The model's name.
        model: Name,
        /// The terminal.
        terminal: Name,
    },
    /// An instance's parameters are invalid for its model.
    Parameters {
        /// The instance.
        at: Location,
        /// What is wrong with the parameters.
        message: String,
    },
    /// A logic pin joins an electrical net, and its model cannot cross to one.
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
    /// A thermal terminal joins a net that electrical terminals or logic pins join too.
    ThermalOnElectricalNet {
        /// The instance.
        at: Location,
        /// The thermal terminal.
        terminal: String,
        /// The net's path.
        net: String,
    },
    /// A model stands in for a component instance the circuit does not have.
    UnknownSubstitute(String),
    /// Parameters replace those of an instance the circuit does not have.
    UnknownOverride(String),
    /// A number replaces one that a device's parameters do not write.
    UnknownParameter {
        /// The device.
        at: Location,
        /// The number's path among the device's parameters, such as `waveform.dc`.
        parameter: String,
    },
    /// A number replaces one of a logic device's, which elaborating the circuit reads once
    /// into its logic's structure.
    LogicParameter {
        /// The logic device.
        at: Location,
        /// The number's path among the device's parameters.
        parameter: String,
    },
    /// A number replaces a parameter the circuit's table does not hold, by its index.
    NoParameter(usize),
    /// The circuit's system is too large to factor.
    TooLarge,
}

impl ElaborationProblem {
    /// The instance the problem is about, if it is about one.
    #[must_use]
    pub fn location(&self) -> Option<&Location> {
        match self {
            Self::UnknownModel { at, .. }
            | Self::UnconnectedTerminal { at, .. }
            | Self::UnknownTerminal { at, .. }
            | Self::Parameters { at, .. }
            | Self::LogicPinOnElectricalNet { at, .. }
            | Self::ThermalOnElectricalNet { at, .. }
            | Self::UnknownParameter { at, .. }
            | Self::LogicParameter { at, .. } => Some(at),
            _ => None,
        }
    }

    /// What is wrong, without the instance [`ElaborationProblem::location`] names.
    #[must_use]
    pub fn message(&self) -> String {
        match self {
            Self::Structure(problem) => problem.to_string(),
            Self::NoReference(top) => {
                format!("the top component `{top}` has no ground port to be the reference node")
            }
            Self::SeveralReferences { top, count } => format!(
                "the top component `{top}` has {count} ground ports, and the reference is one node"
            ),
            Self::UnknownModel { model, .. } => format!("unknown model `{model}`"),
            Self::UnconnectedTerminal {
                model, terminal, ..
            } => format!("terminal `{terminal}` of its {model} is unconnected"),
            Self::UnknownTerminal {
                model, terminal, ..
            } => format!("connects `{terminal}`, which a {model} does not have"),
            Self::Parameters { message, .. } => message.clone(),
            Self::LogicPinOnElectricalNet {
                model, pin, net, ..
            } => format!(
                "joins its logic pin `{pin}` to the electrical net `{net}`, and a {model} has no \
                 supply and ground to cross to it"
            ),
            Self::ThermalOnElectricalNet { terminal, net, .. } => format!(
                "joins its thermal terminal `{terminal}` to the net `{net}`, which electrical \
                 terminals or logic pins join too, and a temperature is no voltage"
            ),
            Self::UnknownSubstitute(path) => format!(
                "the circuit has no component instance at `{path}`, for which a model stands in"
            ),
            Self::UnknownOverride(path) => {
                format!("the circuit has no device at `{path}`, whose parameters are replaced")
            }
            Self::UnknownParameter { parameter, .. } => {
                format!("the override `{parameter}` names no number among the parameters")
            }
            Self::LogicParameter { parameter, .. } => format!(
                "the override `{parameter}` replaces a number of a logic device, which only \
                 elaborating the circuit with it replaces"
            ),
            Self::NoParameter(index) => {
                format!("the circuit has no parameter {index}, whose number is replaced")
            }
            Self::TooLarge => "the circuit's system is too large to factor".to_owned(),
        }
    }
}

impl fmt::Display for ElaborationProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.location() {
            Some(at) => write!(f, "instance {at}: {}", self.message()),
            None => f.write_str(&self.message()),
        }
    }
}

impl std::error::Error for ElaborationProblem {}

impl From<Problem> for ElaborationProblem {
    fn from(problem: Problem) -> Self {
        Self::Structure(problem)
    }
}

/// Every [`ElaborationProblem`] of a circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElaborationError(Vec<ElaborationProblem>);

impl ElaborationError {
    /// The error of `problems`, which are not none.
    pub(crate) const fn new(problems: Vec<ElaborationProblem>) -> Self {
        Self(problems)
    }

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

/// The models that stand in for component instances, by path, and the parameters that
/// replace devices' own, by path.
pub(crate) type Replaced<'a> = (
    &'a BTreeMap<String, Substitute>,
    &'a BTreeMap<String, Parameters>,
);

/// Flattens `circuit` and binds each device to its unknowns at `temperature`, in kelvins,
/// with the models that stand in for some component instances and the parameters that
/// replace some devices' own, `replaced`, which the circuit then holds as its own. Returns
/// the elaborated circuit and the devices made.
pub(crate) fn elaborate(
    circuit: &Circuit,
    library: &Library,
    (substitutes, overrides): Replaced<'_>,
    temperature: f64,
) -> Result<(Elaborated, Devices), ElaborationError> {
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
    let reference: BTreeMap<&Name, Joined> = ground
        .iter()
        .map(|&name| {
            let joined = Joined {
                net: None,
                port: None,
            };
            (name, joined)
        })
        .collect();
    let mut walk = Walk {
        circuit,
        library,
        overrides,
        overridden: BTreeSet::new(),
        substitutes,
        substituted: BTreeSet::new(),
        nets: Vec::new(),
        named: HashMap::new(),
        entered: 1,
        ports: Vec::new(),
        devices: Vec::new(),
        logic: Vec::new(),
        problems,
    };
    let scope = Scope {
        prefix: "",
        number: 0,
    };
    walk.component(top, scope, &reference);
    let Walk {
        nets,
        ports,
        devices,
        logic,
        mut problems,
        overridden,
        substituted,
        ..
    } = walk;

    problems.extend(mixed_domains(&nets));
    let (unknowns, classes, mut partition) = classify(&nets, ground_name(circuit));
    let mut elements = Elements {
        binder: Binder::new(unknowns, temperature),
        templates: Vec::with_capacity(devices.len()),
        detected: Vec::new(),
        pins: Vec::new(),
        partition: &mut partition,
        problems: &mut problems,
    };
    for pending in devices {
        elements.analog(pending, &classes);
    }
    for pending in logic {
        elements.logic(pending, &classes);
    }
    let Elements {
        binder,
        templates,
        detected,
        pins,
        ..
    } = elements;
    partition.index(&detected, &templates);
    problems.extend(unclaimed(
        (substitutes, overrides),
        &substituted,
        &overridden,
    ));
    // Without a single ground, a problem says so.
    let (Some(ground), true) = (ground, problems.is_empty()) else {
        return Err(ElaborationError(problems));
    };
    let bound = binder.finish()?;
    let composites = composites(&ports, &pins, &classes, &templates);
    let parts = (partition, composites, ground.clone());
    Ok(Elaborated::assemble(bound, parts, temperature))
}

/// The terminals of every component instance and logic device on a node, electrical or
/// thermal, each with the device terminals it sums: the ports, in the order the walk met
/// them, then the logic devices' pins and terminals. The analog devices were placed in the
/// order the walk found them, so the walk's index of each is its index among the devices.
fn composites(
    ports: &[PortRecord],
    pins: &[LogicPins],
    classes: &Classes<'_>,
    templates: &[Arc<Template>],
) -> Vec<Composite> {
    let ports = ports
        .iter()
        .filter(|port| port.net.is_none_or(|net| classes.nodes[net].is_some()))
        .map(|port| {
            let thermal = port.net.is_some_and(|net| !classes.nets[net].electrical);
            let members = port
                .members
                .iter()
                .flat_map(|&member| match member {
                    Member::Analog { device, terminal } => {
                        let heat = templates[device].interface.thermal.contains(&terminal);
                        if heat == thermal {
                            vec![(device, terminal)]
                        } else {
                            Vec::new()
                        }
                    }
                    Member::Logic { device, pin } => pins[device].members(pin),
                })
                .collect();
            Composite {
                name: port.name.clone(),
                owner: port.owner.clone(),
                node: classes.node(port.net),
                thermal,
                members,
            }
        });
    let logic = pins.iter().flat_map(|device| {
        (0..device.names.len()).filter_map(|pin| {
            let on_electrical = pin >= device.elements.len() || device.elements[pin].is_some();
            on_electrical.then(|| Composite {
                name: format!("{}.{}", device.path, device.names[pin]),
                owner: device.path.clone(),
                node: classes.node(device.nets[pin]),
                thermal: false,
                members: device.members(pin),
            })
        })
    });
    ports.chain(logic).collect()
}

/// The pins and terminals of a logic device, and the connect element of each pin that
/// joins an electrical net.
struct LogicPins {
    /// The device's path.
    path: String,
    /// Its pins' and terminals' names, in the order of [`Pins::names`].
    names: Vec<String>,
    /// The net of each pin and terminal, in the same order.
    nets: Vec<NetRef>,
    /// The connect element of each pin, inputs then outputs, on an electrical net.
    elements: Vec<Option<usize>>,
}

impl LogicPins {
    /// The device terminals whose currents sum to that into pin or terminal `pin`: its
    /// connect element's on the electrical net, or for the supply or the ground, that
    /// terminal of every connect element of the device.
    fn members(&self, pin: usize) -> Vec<(usize, usize)> {
        let signals = self.elements.len();
        if pin < signals {
            self.elements[pin]
                .map(|element| (element, 0))
                .into_iter()
                .collect()
        } else {
            let terminal = 1 + pin - signals;
            self.elements
                .iter()
                .flatten()
                .map(|&element| (element, terminal))
                .collect()
        }
    }
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

/// The problem of each substitute and override of `replaced` whose path the walk did not
/// find: those it `stood_in` and `overridden`.
fn unclaimed<'a>(
    (substitutes, overrides): Replaced<'a>,
    stood_in: &'a BTreeSet<String>,
    overridden: &'a BTreeSet<String>,
) -> impl Iterator<Item = ElaborationProblem> + 'a {
    let standing = substitutes
        .keys()
        .filter(|path| !stood_in.contains(path.as_str()))
        .map(|path| ElaborationProblem::UnknownSubstitute(path.clone()));
    let unused = overrides
        .keys()
        .filter(|path| !overridden.contains(path.as_str()))
        .map(|path| ElaborationProblem::UnknownOverride(path.clone()));
    standing.chain(unused)
}

/// The problem of each net a thermal terminal joins that an electrical terminal or a logic
/// pin joins too.
fn mixed_domains(nets: &[NetRecord]) -> impl Iterator<Item = ElaborationProblem> + '_ {
    nets.iter().filter_map(|net| {
        let (at, terminal) = net.thermal.as_ref()?;
        (net.electrical || net.logic).then(|| ElaborationProblem::ThermalOnElectricalNet {
            at: at.clone(),
            terminal: terminal.clone(),
            net: net.name.clone(),
        })
    })
}

/// Classifies the walk's `nets`: those an electrical terminal joins become node voltages,
/// and those a thermal terminal joins temperatures above the ambient, in the order they
/// were first met, and the others logic nets. Returns the nodes' unknowns, the classes, and
/// the logic partition with its nets.
fn classify(nets: &[NetRecord], ground: String) -> (Vec<UnknownInfo>, Classes<'_>, LogicSystem) {
    let mut unknowns = Vec::new();
    let mut nodes: Vec<Node> = vec![None; nets.len()];
    let mut logic: Vec<Option<usize>> = vec![None; nets.len()];
    let mut partition = LogicSystem::default();
    for (index, net) in nets.iter().enumerate() {
        if net.electrical || net.thermal.is_some() {
            nodes[index] = Some(Unknown::new(unknowns.len()));
            unknowns.push(UnknownInfo {
                name: net.name.clone(),
                nature: if net.electrical {
                    Nature::Voltage
                } else {
                    Nature::Temperature
                },
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

/// What devices, logic devices and their connect elements are placed into.
struct Elements<'a> {
    /// The binder of every device placed, in order.
    binder: Binder,
    /// What each device placed is, in order, those that failed to bind among them.
    templates: Vec<Arc<Template>>,
    /// Each detector's input node and capacitance.
    detected: Vec<(Node, f64)>,
    /// Each logic device's pins, in the order the walk found the devices.
    pins: Vec<LogicPins>,
    partition: &'a mut LogicSystem,
    problems: &'a mut Vec<ElaborationProblem>,
}

impl Elements<'_> {
    /// Binds the analog device `pending` to the nodes of its terminals.
    fn analog(&mut self, pending: Pending<'_>, classes: &Classes<'_>) {
        let nodes: Vec<Node> = pending
            .terminals
            .iter()
            .map(|&net| classes.node(net))
            .collect();
        let origin = Origin::Model(Made {
            model: Arc::clone(pending.model),
            specification: pending.specification,
            parameters: pending.parameters.into_owned(),
        });
        self.place(
            pending.at,
            pending.model.name(),
            pending.interface,
            nodes,
            origin,
        );
    }

    /// Places the logic device `pending`: each pin on a net only logic pins join joins its
    /// logic net, and each pin on an electrical net joins a logic net of its own, which a
    /// connect element joins to the electrical net.
    fn logic(&mut self, pending: PendingLogic<'_>, classes: &Classes<'_>) {
        let signals = pending.pins.inputs.len() + pending.pins.outputs.len();
        self.pins.push(LogicPins {
            path: pending.at.path.clone(),
            names: pending.pins.names(),
            nets: pending.nets.clone(),
            elements: vec![None; signals],
        });
        let device = pending.device;
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
                let terminals = [classes.node(net), supply, ground];
                let (element, own) =
                    self.cross(at, (input, end), terminals, (threshold, capacitance));
                if let Some(pins) = self.pins.last_mut() {
                    pins.elements[offset + pin] = Some(element);
                }
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

    /// Joins the pin `end` of a logic device, an input when `input`, to the electrical net
    /// of `terminals`' first, through a connect element at `at` that reads it against
    /// `threshold` and loads it with `capacitance`, or drives it, on a logic net of their
    /// own. Returns the element's index among the devices and the logic net's.
    fn cross(
        &mut self,
        at: Location,
        (input, end): (bool, Endpoint),
        terminals: [Node; 3],
        (threshold, capacitance): (f64, f64),
    ) -> (usize, usize) {
        let own = self.partition.nets.len();
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
        (element, own)
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
        let interface = ThresholdDetector::interface::<f64>(&parameters);
        let origin = Origin::Detector(parameters);
        self.place(
            at,
            <ThresholdDetector as Kernel>::NAME,
            interface,
            terminals.to_vec(),
            origin,
        )
    }

    /// Places a logic driver at `at` on `terminals`: its output, supply and ground.
    fn driver(&mut self, at: Location, terminals: [Node; 3]) -> usize {
        let interface = LogicDriver::interface::<f64>(&LogicDriverParameters {});
        self.place(
            at,
            <LogicDriver as Kernel>::NAME,
            interface,
            terminals.to_vec(),
            Origin::Driver(LogicDriverParameters {}),
        )
    }

    /// Places the device at `at` of `model`, with `interface`, its terminals on `nodes`,
    /// made from `origin`, and binds it; returns its index among the devices.
    fn place(
        &mut self,
        location: Location,
        model: &'static str,
        interface: Interface,
        nodes: Vec<Node>,
        origin: Origin,
    ) -> usize {
        let template = Arc::new(Template {
            location,
            model,
            interface,
            nodes,
            origin,
        });
        if let Err(error) = self.binder.bind(Arc::clone(&template), None, true) {
            self.problems.push(ElaborationProblem::Parameters {
                at: template.location.clone(),
                message: error.0,
            });
        }
        self.templates.push(template);
        self.templates.len() - 1
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
    /// Whether a logic pin joins it.
    logic: bool,
    /// The first thermal terminal that joins it, if one does: its instance and its name.
    thermal: Option<(Location, String)>,
}

/// A device instance found by the walk, waiting for every net to be classified.
struct Pending<'a> {
    at: Location,
    model: &'a Arc<dyn Model>,
    /// The instance's parameters, with those the conditions replace, read.
    specification: Box<dyn Specification>,
    /// The same parameters, as the circuit and the conditions give them.
    parameters: Cow<'a, Parameters>,
    interface: Interface,
    terminals: Vec<NetRef>,
}

/// A logic device instance found by the walk, made from its parameters.
struct PendingLogic<'a> {
    at: Location,
    name: Name,
    model: &'a Arc<dyn LogicModel>,
    device: Arc<dyn Logic>,
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
    /// Each net met, by the number of the component instance it is in and its name there.
    named: HashMap<(usize, &'a Name), usize>,
    /// How many component instances the walk has entered, the top's among them.
    entered: usize,
    /// Every port of every component instance, in the order they were met.
    ports: Vec<PortRecord>,
    devices: Vec<Pending<'a>>,
    logic: Vec<PendingLogic<'a>>,
    problems: Vec<ElaborationProblem>,
}

/// A component instance the walk is in.
#[derive(Debug, Clone, Copy)]
struct Scope<'p> {
    /// The start of the paths of the instances within it, such as `X1/`.
    prefix: &'p str,
    /// Its number, in the order the walk enters component instances, from 0, the top's.
    number: usize,
}

/// A net as a component sees it: the net, and the port of the component's instance it is,
/// if it is one.
#[derive(Debug, Clone, Copy)]
struct Joined {
    net: NetRef,
    port: Option<usize>,
}

/// A port of a component instance, as the walk finds it.
struct PortRecord {
    /// Its instance's path and its own name, such as `X1.vdd`.
    name: String,
    /// Its instance's path, such as `X1`.
    owner: String,
    /// The net it joins.
    net: NetRef,
    /// The port of the enclosing instance that its net is outside, if it is one.
    parent: Option<usize>,
    /// Each device terminal and logic pin inside the instance that joins it.
    members: Vec<Member>,
}

/// A device terminal or a logic pin that joins a port inside its instance.
#[derive(Debug, Clone, Copy)]
enum Member {
    /// Terminal `terminal` of the analog device found `device`-th.
    Analog { device: usize, terminal: usize },
    /// Pin or terminal `pin`, in the order of [`Pins::names`], of the logic device found
    /// `device`-th.
    Logic { device: usize, pin: usize },
}

impl<'a> Walk<'a> {
    /// Walks `component`, in the component instance `scope`, whose ports connect to the
    /// nets in `ports`. The circuit has been validated, so every referenced component
    /// exists and every port of a component instance is connected.
    fn component(
        &mut self,
        component: &'a Component,
        scope: Scope<'_>,
        ports: &BTreeMap<&'a Name, Joined>,
    ) {
        for instance in &component.instances {
            let at = Location {
                path: format!("{}{}", scope.prefix, instance.id),
                source: instance.source.clone(),
            };
            match &instance.of {
                Of::Device(model) => {
                    let parameters = match self.overrides.get(&at.path) {
                        Some(replaced) => {
                            self.overridden.insert(at.path.clone());
                            let mut merged = instance.parameters.clone();
                            for (path, value) in replaced {
                                let set = set_parameter(&mut merged, path, value.clone());
                                if let Err(error) = set {
                                    self.problems.push(ElaborationProblem::Parameters {
                                        at: at.clone(),
                                        message: format!("the override {error}"),
                                    });
                                }
                            }
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
                    self.device(placement, scope, ports);
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
                    self.device(placement, scope, ports);
                }
                Of::Component(target) => {
                    let definition = &self.circuit.components[target];
                    let inner: BTreeMap<&Name, Joined> = definition
                        .ports
                        .iter()
                        .map(|port| {
                            let outside = self.net(scope, ports, &instance.terminals[&port.name]);
                            self.ports.push(PortRecord {
                                name: format!("{}.{}", at.path, port.name),
                                owner: at.path.clone(),
                                net: outside.net,
                                parent: outside.port,
                                members: Vec::new(),
                            });
                            let record = Some(self.ports.len() - 1);
                            let joined = Joined {
                                net: outside.net,
                                port: record,
                            };
                            (&port.name, joined)
                        })
                        .collect();
                    let inside = Scope {
                        prefix: &format!("{}/", at.path),
                        number: self.entered,
                    };
                    self.entered += 1;
                    self.component(definition, inside, &inner);
                }
            }
        }
    }

    /// Places the device of `placement`, analog or logic, in the component instance
    /// `scope`.
    fn device(
        &mut self,
        placement: Placement<'a, '_>,
        scope: Scope<'_>,
        ports: &BTreeMap<&'a Name, Joined>,
    ) {
        let name = placement.model.as_str();
        if let Some(model) = self.library.get(name) {
            self.analog(placement, model, scope, ports);
        } else if let Some(model) = self.library.logic(name) {
            self.logic(placement, model, scope, ports);
        } else {
            self.problems.push(ElaborationProblem::UnknownModel {
                at: placement.at,
                model: placement.model.clone(),
            });
        }
    }

    /// Places the analog device of `placement`, of `model`, which reads its parameters
    /// once, as [`Walk::device`] does.
    fn analog(
        &mut self,
        placement: Placement<'a, '_>,
        model: &'a Arc<dyn Model>,
        scope: Scope<'_>,
        ports: &BTreeMap<&'a Name, Joined>,
    ) {
        let Placement {
            terminals: connected,
            parameters,
            at,
            model: model_name,
        } = placement;
        let specification = match model.read(&parameters) {
            Ok(specification) => specification,
            Err(error) => {
                self.problems.push(ElaborationProblem::Parameters {
                    at,
                    message: error.0,
                });
                return;
            }
        };
        let interface = specification.interface();
        let Some(joined) = self.terminals(
            connected,
            &at,
            model_name,
            &interface.terminals,
            scope,
            ports,
        ) else {
            return;
        };
        let device = self.devices.len();
        for (terminal, joined) in joined.iter().enumerate() {
            self.attach(joined.port, Member::Analog { device, terminal });
        }
        let terminals: Vec<NetRef> = joined.iter().map(|joined| joined.net).collect();
        for (index, net) in terminals.iter().enumerate() {
            let Some(net) = *net else { continue };
            let record = &mut self.nets[net];
            if interface.thermal.contains(&index) {
                record
                    .thermal
                    .get_or_insert_with(|| (at.clone(), interface.terminals[index].clone()));
            } else {
                record.electrical = true;
            }
        }
        self.devices.push(Pending {
            at,
            model,
            specification,
            parameters,
            interface,
            terminals,
        });
    }

    /// Places the logic device of `placement`, of `model`, which it makes from its
    /// parameters, as [`Walk::device`] does.
    fn logic(
        &mut self,
        placement: Placement<'a, '_>,
        model: &'a Arc<dyn LogicModel>,
        scope: Scope<'_>,
        ports: &BTreeMap<&'a Name, Joined>,
    ) {
        let Placement {
            terminals: connected,
            parameters,
            at,
            model: model_name,
        } = placement;
        let device: Arc<dyn Logic> = match model.instantiate(&parameters) {
            Ok(device) => Arc::from(device),
            Err(error) => {
                self.problems.push(ElaborationProblem::Parameters {
                    at,
                    message: error.0,
                });
                return;
            }
        };
        let pins = device.pins();
        let Some(joined) = self.terminals(connected, &at, model_name, &pins.names(), scope, ports)
        else {
            return;
        };
        let found = self.logic.len();
        for (pin, joined) in joined.iter().enumerate() {
            self.attach(joined.port, Member::Logic { device: found, pin });
        }
        let nets: Vec<NetRef> = joined.iter().map(|joined| joined.net).collect();
        let pin_count = pins.inputs.len() + pins.outputs.len();
        for &net in nets[..pin_count].iter().flatten() {
            self.nets[net].logic = true;
        }
        for &net in nets[pin_count..].iter().flatten() {
            self.nets[net].electrical = true;
        }
        self.logic.push(PendingLogic {
            at,
            name: model_name.clone(),
            model,
            device,
            pins,
            nets,
        });
    }

    /// The net of each of `expected`, the terminals a model declares, as `connected` joins
    /// them, or `None` after reporting a terminal left unconnected or that it does not have.
    /// Each net comes with the port of the enclosing instance it is, if it is one.
    fn terminals(
        &mut self,
        connected: &'a BTreeMap<Name, Name>,
        at: &Location,
        model: &Name,
        expected: &[String],
        scope: Scope<'_>,
        ports: &BTreeMap<&'a Name, Joined>,
    ) -> Option<Vec<Joined>> {
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
                terminals.push(self.net(scope, ports, net));
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

    /// The net `net` of the component instance `scope`: a port's net from the parent, with
    /// the port, or the component's own net, recorded and named by its path when first
    /// met.
    fn net(
        &mut self,
        scope: Scope<'_>,
        ports: &BTreeMap<&'a Name, Joined>,
        net: &'a Name,
    ) -> Joined {
        if let Some(&found) = ports.get(net) {
            return found;
        }
        if let Some(&index) = self.named.get(&(scope.number, net)) {
            return Joined {
                net: Some(index),
                port: None,
            };
        }
        let index = self.nets.len();
        self.nets.push(NetRecord {
            name: format!("{}{net}", scope.prefix),
            electrical: false,
            logic: false,
            thermal: None,
        });
        self.named.insert((scope.number, net), index);
        Joined {
            net: Some(index),
            port: None,
        }
    }

    /// Adds `member` to `port` and to every port of an enclosing instance outside it.
    fn attach(&mut self, mut port: Option<usize>, member: Member) {
        while let Some(index) = port {
            self.ports[index].members.push(member);
            port = self.ports[index].parent;
        }
    }
}

#[cfg(test)]
mod tests {
    use jolty_models::REFERENCE_TEMPERATURE;

    use super::*;
    use crate::elaborated::Placed;

    fn elaborate_json(components: &str) -> Result<Elaborated, ElaborationError> {
        let header = r#""format": "jolty-circuit", "version": 1, "top": "top""#;
        let text = format!(r#"{{{header}, "components": {components}}}"#);
        let none = (&BTreeMap::new(), &BTreeMap::new());
        let circuit = Circuit::from_json(&text).unwrap();
        elaborate(&circuit, &Library::standard(), none, REFERENCE_TEMPERATURE)
            .map(|(elaborated, _)| elaborated)
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
    fn each_instance_of_a_component_has_nets_of_its_own() {
        let system = elaborate_json(
            r#"{
                "top": {"ports": [{"name": "gnd", "role": "ground"}], "instances": [
                    {"id": "X1", "of": {"component": "cell"}, "terminals": {"a": "gnd"}},
                    {"id": "X2", "of": {"component": "cell"}, "terminals": {"a": "mid"}},
                    {"id": "R0", "of": {"device": "resistor"}, "parameters": {"resistance": 1},
                     "terminals": {"a": "mid", "b": "gnd"}}
                ]},
                "cell": {"ports": [{"name": "a", "role": "signal"}], "instances": [
                    {"id": "R1", "of": {"device": "resistor"}, "parameters": {"resistance": 1},
                     "terminals": {"a": "a", "b": "mid"}},
                    {"id": "R2", "of": {"device": "resistor"}, "parameters": {"resistance": 1},
                     "terminals": {"a": "mid", "b": "a"}}
                ]}
            }"#,
        )
        .unwrap();
        let names: Vec<&str> = system
            .unknowns
            .iter()
            .map(|unknown| unknown.name.as_str())
            .collect();
        assert_eq!(names, ["X1/mid", "mid", "X2/mid"]);
        let nodes: Vec<&[Node]> = system.devices.iter().map(Placed::nodes).collect();
        let [first, second, third] = [0, 1, 2].map(|index| Some(Unknown::new(index)));
        assert_eq!(
            nodes,
            [
                [None, first].as_slice(),
                &[first, None],
                &[second, third],
                &[third, second],
                &[second, None],
            ]
        );
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
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "X2/R1".to_owned(),
            serde_json::json!({"resistance": -1})
                .as_object()
                .cloned()
                .unwrap(),
        );
        overrides.insert(
            "X1/R1".to_owned(),
            serde_json::json!({"resistance.dc": 2})
                .as_object()
                .cloned()
                .unwrap(),
        );
        overrides.insert("X3/R1".to_owned(), Parameters::new());
        let replaced = (&BTreeMap::new(), &overrides);
        let error = elaborate(
            &circuit,
            &Library::standard(),
            replaced,
            REFERENCE_TEMPERATURE,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "instance `X1/R1`: the override `resistance.dc` has no place among the parameters\n\
             instance `X2/R1`: resistance must be positive, not -1.0\n\
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
            "instance `Q1`: unknown model `transistor`\n\
             instance `C1`: connects `c`, which a capacitor does not have\n\
             instance `C1`: terminal `b` of its capacitor is unconnected\n\
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
