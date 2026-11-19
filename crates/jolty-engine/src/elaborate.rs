//! Elaboration: flattening a circuit's hierarchy into devices bound to unknowns.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::sync::Arc;

use jolty_ir::{
    Circuit, Component, Name, Of, Parameters, Problem, Role, SourceLocation, set_parameter,
};
use jolty_models::{
    Interface, Kernel, Library, Logic, LogicDriver, LogicDriverParameters, LogicModel, Model,
    Nature, Node, Pins, Specification, ThresholdDetector, ThresholdDetectorParameters, Unknown,
    UnknownIndex,
};

use crate::Substitute;
use crate::elaborated::{Binder, Devices, Elaborated, Made, Origin, Template};
use crate::logic::{Endpoint, LogicNet, LogicSystem, PlacedLogic};

/// Terminal of a component instance or a logic device whose current and charge sum
/// underlying device terminals.
///
/// A component port sums every device terminal inside the instance that joins the port's
/// net, while a logic pin sums the terminal of its connect element on the pin's electrical
/// net. A port on a thermal net is thermal, and its current represents heat flowing into
/// the port. Ground also serves as the ambient thermal reference, so a port on ground sums
/// only electrical terminals because heat into ambient is not electrical current.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummedTerminal {
    /// Qualified terminal name combining instance path and terminal name, such as `X1.vdd`.
    pub(crate) name: String,
    /// Hierarchical path of the owning component instance or logic device, such as `X1`.
    pub(crate) owner: String,
    /// Node joined by this terminal, or `None` for the reference node.
    pub(crate) node: Node,
    /// Flag indicating whether the terminal lies on a thermal net.
    pub(crate) thermal: bool,
    /// Summed device terminals stored as `(device_index, terminal_index)` pairs.
    pub(crate) members: Vec<(usize, usize)>,
}

impl SummedTerminal {
    /// Returns the qualified terminal name combining instance path and terminal name, such as `X1.vdd`.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the hierarchical path of the owning component instance or logic device, such as `X1`.
    #[must_use]
    pub fn owner(&self) -> &str {
        &self.owner
    }

    /// Returns the node joined by this terminal: a node-voltage unknown, or `None` for ground.
    #[must_use]
    pub fn node(&self) -> Node {
        self.node
    }

    /// Reports whether this terminal lies on a thermal net, where current represents heat flow.
    #[must_use]
    pub fn thermal(&self) -> bool {
        self.thermal
    }

    /// Returns the summed device terminals as `(device_index, terminal_index)` pairs indexing
    /// into [`Simulator::devices`](crate::Simulator::devices) and each device's terminal list.
    #[must_use]
    pub fn members(&self) -> &[(usize, usize)] {
        &self.members
    }
}

/// Hierarchical path of an instance paired with the source line that placed it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Location {
    /// Instance path through the circuit hierarchy, such as `amplifier/R1`.
    pub path: String,
    /// Line in the build script that placed the instance, if recorded.
    pub source_location: Option<SourceLocation>,
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "`{}`", self.path)?;
        if let Some(source) = &self.source_location {
            write!(f, " ({}:{})", source.file, source.line)?;
        }
        Ok(())
    }
}

/// Reason a circuit cannot be elaborated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElaborationProblem {
    /// Structural validation failure in the circuit.
    Structure(Problem),
    /// Top component declaring no ground port, leaving the circuit without a reference node.
    NoGround(Name),
    /// Top component declaring multiple ground ports.
    SeveralGrounds {
        /// Name of the top component.
        top: Name,
        /// Number of ground ports declared by the top component.
        count: usize,
    },
    /// Instance referencing a model absent from the library.
    UnknownModel {
        /// Location of the offending instance.
        at: Location,
        /// Unrecognized model name.
        model: Name,
    },
    /// Instance leaving a required model terminal unconnected.
    UnconnectedTerminal {
        /// Location of the offending instance.
        at: Location,
        /// Model name.
        model: Name,
        /// Missing terminal name.
        terminal: String,
    },
    /// Instance connecting a terminal not declared by the model.
    UnknownTerminal {
        /// Location of the offending instance.
        at: Location,
        /// Model name.
        model: Name,
        /// Unrecognized terminal name.
        terminal: Name,
    },
    /// Instance parameters rejected by the model.
    Parameters {
        /// Location of the offending instance.
        at: Location,
        /// Description of the parameter failure.
        message: String,
    },
    /// Logic pin connected to an electrical net when the model cannot bridge domains.
    LogicPinOnElectricalNet {
        /// Location of the logic instance.
        at: Location,
        /// Logic model name.
        model: Name,
        /// Pin name on the electrical net.
        pin: String,
        /// Path of the electrical net.
        net: String,
    },
    /// Thermal terminal sharing a net with electrical terminals or logic pins.
    ThermalOnElectricalNet {
        /// Location of the instance owning the thermal terminal.
        at: Location,
        /// Thermal terminal name.
        terminal: String,
        /// Path of the conflicting net.
        net: String,
    },
    /// Substitute model targeting a component instance path absent from the circuit.
    UnknownSubstitute(String),
    /// Parameter override targeting a device path absent from the circuit.
    UnknownOverride(String),
    /// Numeric override targeting a parameter path absent from the device's parameters.
    UnknownParameter {
        /// Location of the target device.
        at: Location,
        /// Parameter path inside the device's parameters, such as `waveform.dc`.
        parameter: String,
    },
    /// Numeric override targeting a logic device parameter, which elaboration reads once into
    /// the logic structure.
    LogicParameter {
        /// Location of the logic device.
        at: Location,
        /// Parameter path inside the logic device's parameters.
        parameter: String,
    },
    /// Numeric override referencing a parameter index outside the circuit's parameter table.
    NoParameter(usize),
    /// Circuit system exceeding the size limit of sparse factorization.
    TooLarge,
}

impl ElaborationProblem {
    /// Returns the instance associated with the problem, if any.
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

    /// Formats the problem description without the instance prefix from
    /// [`ElaborationProblem::location`].
    #[must_use]
    pub fn message(&self) -> String {
        match self {
            Self::Structure(problem) => problem.to_string(),
            Self::NoGround(top) => {
                format!("the top component `{top}` has no ground port")
            }
            Self::SeveralGrounds { top, count } => format!(
                "the top component `{top}` has {count} ground ports, and ground is one node"
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

/// Collection of every [`ElaborationProblem`] found while elaborating a circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElaborationError(Vec<ElaborationProblem>);

impl ElaborationError {
    /// Wraps a non-empty list of elaboration `problems` into an error.
    pub(crate) const fn new(problems: Vec<ElaborationProblem>) -> Self {
        Self(problems)
    }

    /// Returns the elaboration problems in discovery order.
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

/// Substitute models for component instances and parameter overrides for devices, keyed by path.
#[derive(Clone, Copy)]
pub(crate) struct Replaced<'a> {
    pub(crate) substitutes: &'a BTreeMap<String, Substitute>,
    pub(crate) overrides: &'a BTreeMap<String, Parameters>,
}

/// Flattens `circuit` and binds each device to its unknowns at `temperature` (kelvins).
///
/// Applies substitute models and parameter overrides from `replaced`, storing overridden
/// parameters on the elaborated circuit. Returns the elaborated circuit and its bound
/// devices.
pub(crate) fn elaborate(
    circuit: &Circuit,
    library: &Library,
    replaced: Replaced<'_>,
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
        overrides: replaced.overrides,
        overridden: BTreeSet::new(),
        substitutes: replaced.substitutes,
        substituted: BTreeSet::new(),
        nets: Vec::new(),
        named: HashMap::new(),
        entered: 1,
        ports: Vec::new(),
        devices: Vec::new(),
        logic: Vec::new(),
        problems,
    };
    walk.component(top, reference);
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
    problems.extend(unclaimed(replaced, &substituted, &overridden));
    // Absence of a single ground is already recorded in `problems`.
    let (Some(ground), true) = (ground, problems.is_empty()) else {
        return Err(ElaborationError(problems));
    };
    let bound = binder.finish()?;
    let summed_terminals = summed_terminals(&ports, &pins, &classes, &templates);
    Ok(Elaborated::assemble(
        bound,
        partition,
        summed_terminals,
        ground.clone(),
        temperature,
    ))
}

/// Builds summed terminals for every component port and electrical logic pin or rail.
///
/// Component ports appear in traversal order, followed by logic device pins and rails.
/// Analog devices are placed in traversal order, so each walk index matches the device
/// index in the elaborated circuit.
fn summed_terminals(
    ports: &[PortRecord],
    pins: &[LogicPins],
    classes: &Classes<'_>,
    templates: &[Arc<Template>],
) -> Vec<SummedTerminal> {
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
            SummedTerminal {
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
            on_electrical.then(|| SummedTerminal {
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

/// Pins and rail terminals of a logic device together with connect elements for electrical pins.
struct LogicPins {
    /// Hierarchical path of the logic device.
    path: String,
    /// Pin and terminal names in the order of [`Pins::names`].
    names: Vec<String>,
    /// Net reference for each pin and terminal in the same order.
    nets: Vec<NetReference>,
    /// Connect element device index for each signal pin (inputs then outputs) on an electrical net.
    elements: Vec<Option<usize>>,
}

impl LogicPins {
    /// Returns the `(device_index, terminal_index)` pairs whose currents sum into `pin`.
    ///
    /// Signal pins map to terminal 0 of their connect element, while supply and ground rails
    /// map to the matching rail terminal across every connect element on the device.
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

/// Locates the unique top-level ground port, or returns the problem when zero or multiple exist.
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
            vec![ElaborationProblem::NoGround(circuit.top.clone())],
        ),
        _ => (
            None,
            vec![ElaborationProblem::SeveralGrounds {
                top: circuit.top.clone(),
                count: grounds.len(),
            }],
        ),
    }
}

/// Yields an error for each substitute or override path in `replaced` absent from `stood_in`
/// or `overridden`.
fn unclaimed<'a>(
    replaced: Replaced<'a>,
    stood_in: &'a BTreeSet<String>,
    overridden: &'a BTreeSet<String>,
) -> impl Iterator<Item = ElaborationProblem> + 'a {
    let standing = replaced
        .substitutes
        .keys()
        .filter(|path| !stood_in.contains(path.as_str()))
        .map(|path| ElaborationProblem::UnknownSubstitute(path.clone()));
    let unused = replaced
        .overrides
        .keys()
        .filter(|path| !overridden.contains(path.as_str()))
        .map(|path| ElaborationProblem::UnknownOverride(path.clone()));
    standing.chain(unused)
}

/// Yields an error for each net joined by both a thermal terminal and an electrical or logic
/// terminal.
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

/// Classifies traversed `nets` into voltage unknowns, thermal unknowns, and pure logic nets.
///
/// Electrical and thermal nets receive sequential unknown indices in discovery order;
/// remaining nets populate the returned [`LogicSystem`]. Returns the node unknowns, net
/// classifications, and initialized logic partition.
fn classify(nets: &[NetRecord], ground: String) -> (Vec<Unknown>, Classes<'_>, LogicSystem) {
    let mut unknowns = Vec::new();
    let mut nodes: Vec<Node> = vec![None; nets.len()];
    let mut logic: Vec<Option<usize>> = vec![None; nets.len()];
    let mut partition = LogicSystem::default();
    for (index, net) in nets.iter().enumerate() {
        if net.electrical || net.thermal.is_some() {
            nodes[index] = Some(UnknownIndex::new(unknowns.len()));
            unknowns.push(Unknown {
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

/// Returns the name of the top component's ground port, or an empty string if absent.
fn ground_name(circuit: &Circuit) -> String {
    circuit.components[&circuit.top]
        .ports
        .iter()
        .find(|port| port.role == Role::Ground)
        .map_or_else(String::new, |port| port.name.to_string())
}

/// Classification of every net discovered during the hierarchy walk.
struct Classes<'a> {
    nets: &'a [NetRecord],
    /// Node unknown assigned to each electrical or thermal net.
    nodes: Vec<Node>,
    /// Logic net index assigned to each pure logic net.
    logic: Vec<Option<usize>>,
    /// Name of the reference ground net.
    ground: String,
}

impl Classes<'_> {
    fn node(&self, net: NetReference) -> Node {
        net.and_then(|index| self.nodes[index])
    }
}

/// Accumulator for binding analog devices, logic devices, and connect elements.
struct Elements<'a> {
    /// Binder receiving every placed analog device and connect element in order.
    binder: Binder,
    /// Device templates placed in order, including those that failed to bind.
    templates: Vec<Arc<Template>>,
    /// Input node and load capacitance recorded for each threshold detector.
    detected: Vec<(Node, f64)>,
    /// Logic pin records in device traversal order.
    pins: Vec<LogicPins>,
    partition: &'a mut LogicSystem,
    problems: &'a mut Vec<ElaborationProblem>,
}

/// A logic device pin crossing onto an electrical net through a connect element.
#[derive(Clone, Copy)]
enum CrossingPin {
    Input {
        end: Endpoint,
        threshold: f64,
        capacitance: f64,
    },
    Output {
        end: Endpoint,
    },
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

    /// Places logic device `pending` into the logic partition and inserts connect elements.
    ///
    /// Pins on pure logic nets join their shared logic net directly. Pins on electrical nets
    /// receive a dedicated single-pin logic net bridged to the electrical net by a connect
    /// element.
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
                    source_location: pending.at.source_location.clone(),
                };
                let terminals = [classes.node(net), supply, ground];
                let crossing = if input {
                    CrossingPin::Input {
                        end,
                        threshold,
                        capacitance,
                    }
                } else {
                    CrossingPin::Output { end }
                };
                let (element, own) = self.cross(at, crossing, terminals);
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

    /// Connects a logic `pin` to the electrical net at `terminals[0]` through a connect element.
    ///
    /// Places a threshold detector (for inputs) or logic driver (for outputs) at `at` on a
    /// dedicated logic net, returning `(device_index, logic_net_index)`.
    fn cross(&mut self, at: Location, pin: CrossingPin, terminals: [Node; 3]) -> (usize, usize) {
        let own = self.partition.nets.len();
        let (element, sources, loads, load) = match pin {
            CrossingPin::Input {
                end,
                threshold,
                capacitance,
            } => {
                let element = self.detector(at.clone(), terminals, threshold, capacitance);
                self.partition.detectors.push((element, own));
                (
                    element,
                    vec![Endpoint::Terminal { device: element }],
                    vec![end],
                    capacitance,
                )
            }
            CrossingPin::Output { end } => {
                let element = self.driver(at.clone(), terminals);
                self.partition.drivers.push((element, own));
                (
                    element,
                    vec![end],
                    vec![Endpoint::Terminal { device: element }],
                    0.0,
                )
            }
        };
        self.partition.nets.push(LogicNet {
            name: at.path,
            sources,
            loads,
            load,
        });
        (element, own)
    }

    /// Places a threshold detector at `at` on `terminals` (input, supply, and ground).
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

    /// Places a logic driver at `at` on `terminals` (output, supply, and ground).
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

    /// Places and binds a device of `model` from `origin` at `location` with `interface` and
    /// `nodes`, returning its device index.
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

/// Index of a discovered net in the walk's net table, or `None` for ground.
type NetReference = Option<usize>;

/// Metadata accumulated for a net during the hierarchy walk.
struct NetRecord {
    /// Hierarchical path of the net.
    name: String,
    /// Flag indicating whether any electrical terminal joins the net.
    electrical: bool,
    /// Flag indicating whether any logic pin joins the net.
    logic: bool,
    /// First thermal terminal joining the net as `(location, terminal_name)`, if any.
    thermal: Option<(Location, String)>,
}

/// Analog device instance collected during traversal while waiting for net classification.
struct Pending<'a> {
    at: Location,
    model: &'a Arc<dyn Model>,
    /// Validated model specification parsed from the effective instance parameters.
    specification: Box<dyn Specification>,
    /// Effective parameter map after merging any run-level overrides.
    parameters: Cow<'a, Parameters>,
    interface: Interface,
    terminals: Vec<NetReference>,
}

/// Logic device instance instantiated from its parameters during traversal.
struct PendingLogic<'a> {
    at: Location,
    name: Name,
    model: &'a Arc<dyn LogicModel>,
    device: Arc<dyn Logic>,
    pins: Pins,
    /// Net reference for each pin and terminal in the order of [`Pins::names`].
    nets: Vec<NetReference>,
}

/// Pending device placement carrying terminal net bindings, parameters, location, and model name.
struct Placement<'a, 'm> {
    terminals: &'a BTreeMap<Name, Name>,
    parameters: Cow<'a, Parameters>,
    at: Location,
    model: &'m Name,
}

/// Traversal state for flattening a circuit hierarchy.
struct Walk<'a> {
    circuit: &'a Circuit,
    library: &'a Library,
    overrides: &'a BTreeMap<String, Parameters>,
    /// Device paths matched by entries in `overrides`.
    overridden: BTreeSet<String>,
    substitutes: &'a BTreeMap<String, Substitute>,
    /// Component instance paths matched by entries in `substitutes`.
    substituted: BTreeSet<String>,
    nets: Vec<NetRecord>,
    /// Net index lookup keyed by `(scope_number, local_net_name)`.
    named: HashMap<(usize, &'a Name), usize>,
    /// Count of component instances entered during traversal, including the top component.
    entered: usize,
    /// Port records for every component instance in discovery order.
    ports: Vec<PortRecord>,
    devices: Vec<Pending<'a>>,
    logic: Vec<PendingLogic<'a>>,
    problems: Vec<ElaborationProblem>,
}

/// Active component instance scope during hierarchy traversal.
#[derive(Debug, Clone, Copy)]
struct Scope<'p> {
    /// Hierarchical path prefix for child instances in the scope, such as `X1/`.
    prefix: &'p str,
    /// Sequential scope identifier assigned on entry, starting at 0 for the top component.
    number: usize,
}

/// Net reference inside a component paired with the enclosing port index when the net is a port.
#[derive(Debug, Clone, Copy)]
struct Joined {
    net: NetReference,
    port: Option<usize>,
}

/// Component instance port recorded during hierarchy traversal.
struct PortRecord {
    /// Qualified port name combining instance path and port name, such as `X1.vdd`.
    name: String,
    /// Hierarchical path of the owning component instance, such as `X1`.
    owner: String,
    /// Net joined by the port.
    net: NetReference,
    /// Enclosing instance port index when this port maps directly to a parent port.
    parent: Option<usize>,
    /// Device terminals and logic pins inside the instance that connect to the port.
    members: Vec<Member>,
}

/// Device terminal or logic pin contributing current to an enclosing component port.
#[derive(Debug, Clone, Copy)]
enum Member {
    /// Terminal `terminal` of analog device `device`.
    Analog { device: usize, terminal: usize },
    /// Pin or rail `pin` (ordered by [`Pins::names`]) of logic device `device`.
    Logic { device: usize, pin: usize },
}

/// One active component instance on the explicit hierarchy stack of [`Walk::component`].
struct Frame<'a> {
    component: &'a Component,
    prefix: String,
    number: usize,
    ports: BTreeMap<&'a Name, Joined>,
    next_instance: usize,
}

impl<'a> Walk<'a> {
    /// Walks `top` and its descendant component instances iteratively with an explicit stack,
    /// starting from the top component's ground `ports`. The circuit has been validated, so
    /// every referenced component exists and every port of a component instance is connected.
    fn component(&mut self, top: &'a Component, ports: BTreeMap<&'a Name, Joined>) {
        let mut stack = vec![Frame {
            component: top,
            prefix: String::new(),
            number: 0,
            ports,
            next_instance: 0,
        }];
        while let Some(frame) = stack.last_mut() {
            let Some(instance) = frame.component.instances.get(frame.next_instance) else {
                stack.pop();
                continue;
            };
            frame.next_instance += 1;
            let scope = Scope {
                prefix: &frame.prefix,
                number: frame.number,
            };
            let at = Location {
                path: format!("{}{}", scope.prefix, instance.name),
                source_location: instance.source_location.clone(),
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
                    self.device(placement, scope, &frame.ports);
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
                    self.device(placement, scope, &frame.ports);
                }
                Of::Component(target) => {
                    let definition = &self.circuit.components[target];
                    let mut inner = BTreeMap::new();
                    for port in &definition.ports {
                        let outside =
                            self.net(scope, &frame.ports, &instance.terminals[&port.name]);
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
                        inner.insert(&port.name, joined);
                    }
                    let number = self.entered;
                    self.entered += 1;
                    let prefix = format!("{}/", at.path);
                    stack.push(Frame {
                        component: definition,
                        prefix,
                        number,
                        ports: inner,
                        next_instance: 0,
                    });
                }
            }
        }
    }

    /// Places an analog or logic device from `placement` within component `scope`.
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

    /// Parses parameters and records an analog device from `placement` using `model` in
    /// `scope`.
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
        let terminals: Vec<NetReference> = joined.iter().map(|joined| joined.net).collect();
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

    /// Instantiates and records a logic device from `placement` using `model` in `scope`.
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
        let nets: Vec<NetReference> = joined.iter().map(|joined| joined.net).collect();
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

    /// Resolves `connected` nets against the model's `expected` terminal list.
    ///
    /// Records errors for unknown or unconnected terminals and returns `None` if any check
    /// fails. Each resolved entry includes the enclosing port index when the net is a port.
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

    /// Resolves local `net` in `scope`, returning a parent port net or allocating a local net.
    ///
    /// Newly encountered local nets are recorded in `self.nets` under their hierarchical path.
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
        let header = r#""format": "jolty-circuit", "version": 5, "top": "top""#;
        let text = format!(r#"{{{header}, "components": {components}}}"#);
        let substitutes = BTreeMap::new();
        let overrides = BTreeMap::new();
        let none = Replaced {
            substitutes: &substitutes,
            overrides: &overrides,
        };
        let circuit = Circuit::from_json(&text).unwrap();
        elaborate(&circuit, &Library::standard(), none, REFERENCE_TEMPERATURE)
            .map(|(elaborated, _)| elaborated)
    }

    #[test]
    fn flattens_the_hierarchy_and_names_unknowns_by_path() {
        let system = elaborate_json(
            r#"{
                "top": {"ports": [{"name": "gnd", "role": "ground"}], "instances": [
                    {"name": "V1", "of": {"model": "voltage_source"},
                     "parameters": {"waveform": {"dc": 1}},
                     "terminals": {"positive": "in", "negative": "gnd"}},
                    {"name": "X1", "of": {"component": "cell"}, "terminals": {"a": "in", "b": "gnd"}}
                ]},
                "cell": {
                    "ports": [{"name": "a", "role": "signal"}, {"name": "b", "role": "signal"}],
                    "instances": [
                        {"name": "R1", "of": {"model": "resistor"}, "parameters": {"resistance": 1},
                         "terminals": {"a": "a", "b": "mid"}},
                        {"name": "R2", "of": {"model": "resistor"}, "parameters": {"resistance": 1},
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
                    {"name": "X1", "of": {"component": "cell"}, "terminals": {"a": "gnd"}},
                    {"name": "X2", "of": {"component": "cell"}, "terminals": {"a": "mid"}},
                    {"name": "R0", "of": {"model": "resistor"}, "parameters": {"resistance": 1},
                     "terminals": {"a": "mid", "b": "gnd"}}
                ]},
                "cell": {"ports": [{"name": "a", "role": "signal"}], "instances": [
                    {"name": "R1", "of": {"model": "resistor"}, "parameters": {"resistance": 1},
                     "terminals": {"a": "a", "b": "mid"}},
                    {"name": "R2", "of": {"model": "resistor"}, "parameters": {"resistance": 1},
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
        let [first, second, third] = [0, 1, 2].map(|index| Some(UnknownIndex::new(index)));
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
        let text = r#"{"format": "jolty-circuit", "version": 5, "top": "top", "components": {
            "top": {"ports": [{"name": "gnd", "role": "ground"}], "instances": [
                {"name": "X1", "of": {"component": "cell"}, "terminals": {"a": "gnd"}},
                {"name": "X2", "of": {"component": "cell"}, "terminals": {"a": "gnd"}}
            ]},
            "cell": {"ports": [{"name": "a", "role": "signal"}], "instances": [
                {"name": "R1", "of": {"model": "resistor"}, "parameters": {"resistance": 1},
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
        let substitutes = BTreeMap::new();
        let replaced = Replaced {
            substitutes: &substitutes,
            overrides: &overrides,
        };
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
             instance `X2/R1`: resistance is positive, not -1.0\n\
             the circuit has no device at `X3/R1`, whose parameters are replaced"
        );
    }

    #[test]
    fn reports_every_instance_problem_with_its_source_line() {
        let error = elaborate_json(
            r#"{"top": {"ports": [{"name": "gnd", "role": "ground"}], "instances": [
                {"name": "R1", "of": {"model": "resistor"}, "parameters": {"resistance": -1},
                 "terminals": {"a": "x", "b": "gnd"}, "source_location": {"file": "amp.py", "line": 9}},
                {"name": "Q1", "of": {"model": "transistor"}, "terminals": {}},
                {"name": "C1", "of": {"model": "capacitor"}, "parameters": {"capacitance": 1},
                 "terminals": {"a": "x", "c": "gnd"}}
            ]}}"#,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "instance `Q1`: unknown model `transistor`\n\
             instance `C1`: connects `c`, which a capacitor does not have\n\
             instance `C1`: terminal `b` of its capacitor is unconnected\n\
             instance `R1` (amp.py:9): resistance is positive, not -1.0"
        );
    }

    #[test]
    fn requires_exactly_one_ground() {
        let error = elaborate_json(r#"{"top": {"instances": []}}"#).unwrap_err();
        assert_eq!(
            error.problems(),
            [ElaborationProblem::NoGround(Name::new("top").unwrap())]
        );
        let error = elaborate_json(
            r#"{"top": {"ports": [{"name": "a", "role": "ground"}, {"name": "b", "role": "ground"}],
                "instances": []}}"#,
        )
        .unwrap_err();
        assert_eq!(
            error.problems(),
            [ElaborationProblem::SeveralGrounds {
                top: Name::new("top").unwrap(),
                count: 2
            }]
        );
    }

    #[test]
    fn reports_the_instances_problems_without_a_ground() {
        let error = elaborate_json(
            r#"{"top": {"instances": [
                {"name": "R1", "of": {"model": "resistor"}, "parameters": {"resistance": -1},
                 "terminals": {"a": "x", "b": "y"}}
            ]}}"#,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "the top component `top` has no ground port\n\
             instance `R1`: resistance is positive, not -1.0"
        );
    }

    #[test]
    fn elaborates_deep_hierarchy_without_stack_overflow() {
        let depth = 1500;
        let mut components = serde_json::Map::new();
        // Name components bottom-up (`c0000` leaf -> `c1499` top) so `BTreeMap` iteration in
        // `jolty_ir::validate`'s cycle check finishes each component at depth 1 while
        // `Walk::component` descends all 1500 levels.
        components.insert(
            "c0000".to_owned(),
            serde_json::json!({
                "ports": [
                    {"name": "a", "role": "signal"},
                    {"name": "b", "role": "signal"}
                ],
                "instances": [{
                    "name": "R1",
                    "of": {"model": "resistor"},
                    "parameters": {"resistance": 1000.0},
                    "terminals": {"a": "a", "b": "b"}
                }]
            }),
        );
        for level in 1..(depth - 1) {
            components.insert(
                format!("c{level:04}"),
                serde_json::json!({
                    "ports": [
                        {"name": "a", "role": "signal"},
                        {"name": "b", "role": "signal"}
                    ],
                    "instances": [{
                        "name": "X",
                        "of": {"component": format!("c{:04}", level - 1)},
                        "terminals": {"a": "a", "b": "b"}
                    }]
                }),
            );
        }
        let top_name = format!("c{:04}", depth - 1);
        components.insert(
            top_name.clone(),
            serde_json::json!({
                "ports": [{"name": "gnd", "role": "ground"}],
                "instances": [
                    {
                        "name": "V1",
                        "of": {"model": "voltage_source"},
                        "parameters": {"waveform": {"dc": 1.0}},
                        "terminals": {"positive": "in", "negative": "gnd"}
                    },
                    {
                        "name": "X",
                        "of": {"component": format!("c{:04}", depth - 2)},
                        "terminals": {"a": "in", "b": "gnd"}
                    }
                ]
            }),
        );
        let document = serde_json::json!({
            "format": "jolty-circuit",
            "version": 5,
            "top": top_name,
            "components": components,
        });
        let circuit = Circuit::from_json(&document.to_string()).expect("valid circuit");
        let handle = std::thread::Builder::new()
            .name("deep-elaborate".to_owned())
            .stack_size(2 * 1024 * 1024)
            .spawn(move || {
                let substitutes = BTreeMap::new();
                let overrides = BTreeMap::new();
                let none = Replaced {
                    substitutes: &substitutes,
                    overrides: &overrides,
                };
                elaborate(&circuit, &Library::standard(), none, REFERENCE_TEMPERATURE)
                    .map(|(elaborated, _)| elaborated)
            })
            .expect("spawn thread");
        let elaborated = handle
            .join()
            .expect("thread did not overflow")
            .expect("deep hierarchy elaborates");
        let expected_path = format!("{}R1", "X/".repeat(depth - 1));
        assert_eq!(elaborated.devices.len(), 2);
        assert_eq!(elaborated.devices[0].path(), "V1");
        assert_eq!(elaborated.devices[1].path(), expected_path);
        assert_eq!(
            elaborated.devices[1].nodes(),
            [Some(UnknownIndex::new(0)), None].as_slice()
        );
    }
}
