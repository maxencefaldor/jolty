//! The rules, and the checks that find their diagnostics.

use std::collections::{HashMap, VecDeque};

use jolty_engine::{Endpoint, Location, LogicNet, Partition, Placed, Simulator};
use jolty_models::BranchKind;

use crate::point::{GMIN_DEPENDENCE, MULTIPLE_OPERATING_POINTS, UNSTABLE_POLE};
use crate::preferred::Series;
use crate::structure::{REFERENCE, Structure, ends, index, list};
use crate::{Diagnostic, Level, Rule};

/// The most nodes or instances a message names before it counts the rest.
///
/// Origin: jolty's own diagnostic formatting cap, matching `NAMED = 5` in `point.rs`.
const NAMED: usize = 5;

/// Rule definition for `unconnected-terminal` (structural connectivity check; `AGENTS.md`,
/// "Rules, as in ty").
const UNCONNECTED_TERMINAL: Rule = Rule {
    name: "unconnected-terminal",
    level: Level::Error,
    condition: "A device's terminal or logic pin joins a net that no other terminal or pin \
                joins and that is not the ground: nothing connects to it.",
};

/// Rule definition for `shorted-source` (structural topology check; `AGENTS.md`, "Rules, as
/// in ty").
const SHORTED_SOURCE: Rule = Rule {
    name: "shorted-source",
    level: Level::Error,
    condition: "A voltage or current source, independent or controlled, has both ends of its \
                output on one node, or a current-controlled source has both ends of the port \
                that senses its control current on one node. A voltage source then holds its \
                voltage across nothing, a current source drives its current through nothing, \
                and a sensing port senses no current.",
};

/// Rule definition for `shorted-device` (structural topology check; `AGENTS.md`, "Rules, as
/// in ty").
const SHORTED_DEVICE: Rule = Rule {
    name: "shorted-device",
    level: Level::Warn,
    condition: "A resistor, capacitor, inductor, diode or transistor has every terminal on \
                one node, so it carries no current and does nothing. A transistor with only \
                some terminals joined, such as a diode-connected one, raises nothing.",
};

/// Rule definition for `voltage-loop` (Kirchhoff's voltage law structural check; `AGENTS.md`,
/// "Rules, as in ty").
const VOLTAGE_LOOP: Rule = Rule {
    name: "voltage-loop",
    level: Level::Error,
    condition: "Voltage sources and inductors, independent or controlled, form a loop. At DC \
                an inductor is a short, so the loop's sources alone fix the voltages around \
                it: they conflict, or leave the current around the loop undefined. Each \
                independent loop raises one diagnostic.",
};

/// Rule definition for `no-dc-path` (DC graph connectivity check; `AGENTS.md`, "Rules, as in
/// ty").
const NO_DC_PATH: Rule = Rule {
    name: "no-dc-path",
    level: Level::Error,
    condition: "A group of nodes has no path to the ground through resistors, diodes, \
                inductors or voltage sources, and no current source drives it: it reaches \
                the ground only through capacitors, control inputs or nothing, so its DC \
                voltage is undefined. A group whose every node joins a single terminal is \
                left to unconnected-terminal.",
};

/// Rule definition for `current-cutset` (Kirchhoff's current law structural check;
/// `AGENTS.md`, "Rules, as in ty").
const CURRENT_CUTSET: Rule = Rule {
    name: "current-cutset",
    level: Level::Error,
    condition: "Current sources, independent or controlled, drive a group of nodes that has \
                no DC path to the ground, as no-dc-path defines it: at DC their current has \
                nowhere to go. A group whose every node joins a single terminal is left to \
                unconnected-terminal.",
};

/// Rule definition for `driver-conflict` (mixed-signal logic contention check; `AGENTS.md`,
/// "Rules, as in ty").
const DRIVER_CONFLICT: Rule = Rule {
    name: "driver-conflict",
    level: Level::Error,
    condition: "Two or more logic outputs drive one logic net. Wherever they drive different \
                levels, the net's level is unknown.",
};

/// Rule definition for `non-standard-value` (preferred-value check against IEC 60063 E
/// series; `AGENTS.md`, "Rules, as in ty").
const NON_STANDARD_VALUE: Rule = Rule {
    name: "non-standard-value",
    level: Level::Warn,
    condition: "A resistor's resistance is in no E series of IEC 60063 from E3 to E192, or a \
                capacitor's capacitance or an inductor's inductance in none from E3 to E24, \
                but for rounding: no standard part has the value.",
};

/// Rule definition for `floating-input` (mixed-signal undriven-net check; `AGENTS.md`,
/// "Rules, as in ty").
const FLOATING_INPUT: Rule = Rule {
    name: "floating-input",
    level: Level::Error,
    condition: "Logic inputs read a logic net that no output drives, so its level is always \
                unknown. A net that a single input joins is left to unconnected-terminal.",
};

/// Every rule, in the order a check reports them.
///
/// Origin: registry enumerating all 9 structural and 3 operating-point rules in `jolty-checks`
/// (`AGENTS.md`, "Rules, as in ty").
pub const RULES: &[Rule] = &[
    UNCONNECTED_TERMINAL,
    SHORTED_SOURCE,
    SHORTED_DEVICE,
    VOLTAGE_LOOP,
    NO_DC_PATH,
    CURRENT_CUTSET,
    DRIVER_CONFLICT,
    FLOATING_INPUT,
    NON_STANDARD_VALUE,
    GMIN_DEPENDENCE,
    MULTIPLE_OPERATING_POINTS,
    UNSTABLE_POLE,
];

/// Each terminal or logic pin alone on a net other than the reference.
pub(crate) fn unconnected_terminals(structure: &Structure<'_>, diagnostics: &mut Vec<Diagnostic>) {
    let alone = |node| {
        let node = index(node);
        (node != REFERENCE && structure.degree(node) == 1).then(|| structure.name(node))
    };
    let simulator = structure.simulator();
    let mut found: Vec<(String, &Location, &str)> = Vec::new();
    for (number, device) in structure.devices().iter().enumerate() {
        if structure.is_element(number) {
            continue;
        }
        for (terminal, &node) in device.terminals().iter().zip(device.nodes()) {
            if let Some(net) = alone(node) {
                found.push((terminal.clone(), device.location(), net));
            }
        }
    }
    for device in simulator.logic_devices() {
        let names = device.pins().names();
        let pins = device.inputs().iter().chain(device.outputs());
        for (name, &net) in names.iter().zip(pins) {
            let joined = &simulator.logic_nets()[net];
            if joined.sources.len() + joined.loads.len() == 1 {
                found.push((name.clone(), device.location(), &joined.name));
            }
        }
        let terminals = &names[names.len() - device.nodes().len()..];
        for (terminal, &node) in terminals.iter().zip(device.nodes()) {
            if let Some(net) = alone(node) {
                found.push((terminal.clone(), device.location(), net));
            }
        }
    }
    diagnostics.extend(found.into_iter().map(|(terminal, at, net)| Diagnostic {
        rule: UNCONNECTED_TERMINAL.name,
        message: format!("terminal `{terminal}` of `{}` connects to nothing", at.path),
        instances: vec![at.clone()],
        nets: vec![net.to_owned()],
    }));
}

/// Where the logic device or connect element at `end` is.
fn location(simulator: &Simulator, end: Endpoint) -> &Location {
    match end {
        Endpoint::Pin { device, .. } => simulator.logic_devices()[device].location(),
        Endpoint::Terminal { device } => simulator.devices()[device].location(),
    }
}

/// The diagnostic of `rule` for `net`, naming `ends` in `message`'s `{}`.
fn logic(
    simulator: &Simulator,
    rule: &Rule,
    net: &LogicNet,
    ends: &[Endpoint],
    message: impl Fn(&str) -> String,
) -> Diagnostic {
    let instances: Vec<Location> = ends
        .iter()
        .map(|&end| location(simulator, end).clone())
        .collect();
    let paths = instances.iter().map(|at| at.path.as_str());
    Diagnostic {
        rule: rule.name,
        message: message(&list(paths, NAMED)),
        instances,
        nets: vec![net.name.clone()],
    }
}

/// Each logic net that several outputs drive.
pub(crate) fn driver_conflicts(simulator: &Simulator, diagnostics: &mut Vec<Diagnostic>) {
    for net in simulator.logic_nets() {
        if net.sources.len() > 1 {
            diagnostics.push(logic(
                simulator,
                &DRIVER_CONFLICT,
                net,
                &net.sources,
                |sources| format!("{sources} drive the logic net `{}`", net.name),
            ));
        }
    }
}

/// Each logic net that inputs read and nothing drives, but for a single input alone.
pub(crate) fn floating_inputs(simulator: &Simulator, diagnostics: &mut Vec<Diagnostic>) {
    for net in simulator.logic_nets() {
        if net.sources.is_empty() && net.loads.len() > 1 {
            diagnostics.push(logic(
                simulator,
                &FLOATING_INPUT,
                net,
                &net.loads,
                |loads| {
                    format!(
                        "nothing drives the logic net `{}`, which {loads} read",
                        net.name
                    )
                },
            ));
        }
    }
}

/// Each source branch, an output or a sensing port, that joins a node to itself.
pub(crate) fn shorted_sources(structure: &Structure<'_>, diagnostics: &mut Vec<Diagnostic>) {
    for device in structure.devices() {
        for &branch in device.branches() {
            let (from, to) = ends(device, branch);
            let source = matches!(
                branch.kind,
                BranchKind::VoltageSource | BranchKind::CurrentSource
            );
            if from == to && source {
                let terminals = [
                    device.terminals()[branch.from].clone(),
                    device.terminals()[branch.to].clone(),
                ];
                diagnostics.push(joined(structure, &SHORTED_SOURCE, device, &terminals, from));
            }
        }
    }
}

/// Each resistor, capacitor, inductor, diode or transistor with every terminal on one
/// node.
pub(crate) fn shorted_devices(structure: &Structure<'_>, diagnostics: &mut Vec<Diagnostic>) {
    for device in structure.devices() {
        let Some(&first) = device.nodes().first() else {
            continue;
        };
        let passive = device.branches().iter().any(|branch| {
            matches!(
                branch.kind,
                BranchKind::Conductive | BranchKind::Capacitive | BranchKind::Inductive
            )
        });
        let node = index(first);
        if passive && device.nodes().iter().all(|other| index(*other) == node) {
            diagnostics.push(joined(
                structure,
                &SHORTED_DEVICE,
                device,
                device.terminals(),
                node,
            ));
        }
    }
}

/// Formats `terminals` joined on one node, using `both` for two and `all` for three or more.
fn format_joined_terminals(terminals: &[String]) -> String {
    let names = list(terminals.iter().map(String::as_str), terminals.len());
    match terminals.len() {
        0 | 1 => names,
        2 => format!("{names} both"),
        _ => format!("{names} all"),
    }
}

/// The diagnostic of `rule` for `device`, whose `terminals` all sit on `node`.
fn joined(
    structure: &Structure<'_>,
    rule: &Rule,
    device: &Placed,
    terminals: &[String],
    node: usize,
) -> Diagnostic {
    let listed = format_joined_terminals(terminals);
    Diagnostic {
        rule: rule.name,
        message: format!(
            "`{}` has {listed} on `{}`",
            device.path(),
            structure.name(node)
        ),
        instances: vec![device.location().clone()],
        nets: vec![structure.name(node).to_owned()],
    }
}

/// Each independent loop of branches that fix their voltage at DC.
///
/// The branches grow a spanning forest one by one; a branch whose ends the forest already
/// joins closes a loop, made of that branch and the forest's path between its ends.
pub(crate) fn voltage_loops(structure: &Structure<'_>, diagnostics: &mut Vec<Diagnostic>) {
    let devices = structure.devices();
    let mut partition = Partition::new(structure.len());
    let mut forest: Vec<Vec<(usize, usize)>> = vec![Vec::new(); structure.len()];
    for (closing, device) in devices.iter().enumerate() {
        for &branch in device.branches() {
            let (from, to) = ends(device, branch);
            if from == to || !branch.kind.fixes_dc_voltage() {
                continue;
            }
            if partition.union(from, to) {
                forest[from].push((to, closing));
                forest[to].push((from, closing));
                continue;
            }
            let (mut nodes, mut members) = path(&forest, from, to);
            members.push(closing);
            nodes.sort_unstable();
            members.sort_unstable();
            members.dedup();
            let paths: Vec<&str> = members
                .iter()
                .map(|&member| devices[member].path())
                .collect();
            diagnostics.push(Diagnostic {
                rule: VOLTAGE_LOOP.name,
                message: format!(
                    "{} form a loop of voltage sources and inductors",
                    list(paths.iter().copied(), NAMED)
                ),
                instances: members
                    .iter()
                    .map(|&member| devices[member].location().clone())
                    .collect(),
                nets: nodes
                    .iter()
                    .map(|&node| structure.name(node).to_owned())
                    .collect(),
            });
        }
    }
}

/// The nodes and the devices on the forest's path from `start` to `goal`, which it joins.
fn path(forest: &[Vec<(usize, usize)>], start: usize, goal: usize) -> (Vec<usize>, Vec<usize>) {
    let mut came_from: HashMap<usize, (usize, usize)> = HashMap::new();
    let mut queue = VecDeque::from([start]);
    while let Some(node) = queue.pop_front() {
        if node == goal {
            break;
        }
        for &(next, device) in &forest[node] {
            if next != start && !came_from.contains_key(&next) {
                came_from.insert(next, (node, device));
                queue.push_back(next);
            }
        }
    }
    let (mut nodes, mut devices) = (vec![goal], Vec::new());
    let mut node = goal;
    while let Some(&(previous, device)) = came_from.get(&node) {
        nodes.push(previous);
        devices.push(device);
        node = previous;
    }
    (nodes, devices)
}

/// Each floating group that no current source drives.
pub(crate) fn no_dc_paths(
    structure: &Structure<'_>,
    floating: &[Group],
    diagnostics: &mut Vec<Diagnostic>,
) {
    for group in floating {
        if !group.sources.is_empty() {
            continue;
        }
        let names = group.names(structure);
        let verb = if group.nodes.len() == 1 {
            "has"
        } else {
            "have"
        };
        diagnostics.push(Diagnostic {
            rule: NO_DC_PATH.name,
            message: format!(
                "{} {verb} no DC path to `{}`",
                list(names.iter().copied(), NAMED),
                structure.name(REFERENCE),
            ),
            instances: group.locations(structure),
            nets: names.iter().map(|&name| name.to_owned()).collect(),
        });
    }
}

/// Each floating group that current sources drive.
pub(crate) fn current_cutsets(
    structure: &Structure<'_>,
    floating: &[Group],
    diagnostics: &mut Vec<Diagnostic>,
) {
    let devices = structure.devices();
    for group in floating {
        if group.sources.is_empty() {
            continue;
        }
        let names = group.names(structure);
        let sources: Vec<&str> = group
            .sources
            .iter()
            .map(|&source| devices[source].path())
            .collect();
        let drive = if sources.len() == 1 {
            "drives"
        } else {
            "drive"
        };
        let verb = if group.nodes.len() == 1 {
            "has"
        } else {
            "have"
        };
        diagnostics.push(Diagnostic {
            rule: CURRENT_CUTSET.name,
            message: format!(
                "{} {drive} current into {}, which {verb} no DC path to `{}`",
                list(sources.iter().copied(), NAMED),
                list(names.iter().copied(), NAMED),
                structure.name(REFERENCE),
            ),
            instances: group.locations(structure),
            nets: names.iter().map(|&name| name.to_owned()).collect(),
        });
    }
}

/// Nodes with no DC path to the reference, and what touches them.
pub(crate) struct Group {
    /// The group's nodes, in the engine's order.
    nodes: Vec<usize>,
    /// The devices with a terminal on the group, current sources first, by index.
    devices: Vec<usize>,
    /// The current sources with one end of their output in the group, by index.
    sources: Vec<usize>,
}

impl Group {
    fn names<'a>(&self, structure: &Structure<'a>) -> Vec<&'a str> {
        self.nodes
            .iter()
            .map(|&node| structure.name(node))
            .collect()
    }

    fn locations(&self, structure: &Structure<'_>) -> Vec<Location> {
        let devices = structure.devices();
        self.devices
            .iter()
            .map(|&device| devices[device].location().clone())
            .collect()
    }
}

/// The groups of nodes that branches conducting at DC join to each other but not to the
/// reference, leaving out the groups whose every node joins a single terminal.
pub(crate) fn floating_groups(structure: &Structure<'_>) -> Vec<Group> {
    let devices = structure.devices();
    let mut partition = Partition::new(structure.len());
    for device in devices {
        for &branch in device.branches() {
            if branch.kind.conducts_at_dc() {
                let (from, to) = ends(device, branch);
                partition.union(from, to);
            }
        }
    }
    let grounded = partition.find(REFERENCE);
    let mut groups: Vec<Group> = Vec::new();
    let mut group_of: HashMap<usize, usize> = HashMap::new();
    for node in 1..structure.len() {
        let root = partition.find(node);
        if structure.degree(node) == 0 || root == grounded {
            continue;
        }
        let group = *group_of.entry(root).or_insert_with(|| {
            groups.push(Group {
                nodes: Vec::new(),
                devices: Vec::new(),
                sources: Vec::new(),
            });
            groups.len() - 1
        });
        groups[group].nodes.push(node);
    }
    groups.retain(|group| group.nodes.iter().any(|&node| structure.degree(node) > 1));
    let mut floating = vec![None; structure.len()];
    for (number, group) in groups.iter().enumerate() {
        for &node in &group.nodes {
            floating[node] = Some(number);
        }
    }
    // One pass over the devices: each joins the groups its terminals are on, as a source
    // of those a current source's output has exactly one end in.
    let mut others: Vec<Vec<usize>> = vec![Vec::new(); groups.len()];
    let (mut touched, mut driven) = (Vec::new(), Vec::new());
    for (number, device) in devices.iter().enumerate() {
        touched.clear();
        touched.extend(
            device
                .nodes()
                .iter()
                .filter_map(|&node| floating[index(node)]),
        );
        if touched.is_empty() {
            continue;
        }
        touched.sort_unstable();
        touched.dedup();
        driven.clear();
        for &branch in device.branches() {
            let (from, to) = ends(device, branch);
            let (from, to) = (floating[from], floating[to]);
            if branch.kind == BranchKind::CurrentSource && from != to {
                driven.extend(from.into_iter().chain(to));
            }
        }
        for &group in &touched {
            if driven.contains(&group) {
                groups[group].sources.push(number);
            } else {
                others[group].push(number);
            }
        }
    }
    for (group, others) in groups.iter_mut().zip(others) {
        group.devices = group.sources.iter().copied().chain(others).collect();
    }
    groups
}

/// Each resistor, capacitor and inductor whose value no E series it comes in holds.
pub(crate) fn non_standard_values(simulator: &Simulator, diagnostics: &mut Vec<Diagnostic>) {
    for device in simulator.devices() {
        let (parameter, unit, densest) = match device.model() {
            "resistor" => ("resistance", "Ω", Series::E192),
            "capacitor" => ("capacitance", "F", Series::E24),
            "inductor" => ("inductance", "H", Series::E24),
            _ => continue,
        };
        let Some(value) = device
            .parameters()
            .and_then(|parameters| parameters.get(parameter))
            .and_then(serde_json::Value::as_f64)
        else {
            continue;
        };
        if value > 0.0 {
            // Every IEC 60063 value in E3..=E24 lies in E24, and every value in E48..=E192
            // lies in E192.
            let standard =
                Series::E24.holds(value) || (densest == Series::E192 && Series::E192.holds(value));
            if !standard {
                diagnostics.push(Diagnostic {
                    rule: NON_STANDARD_VALUE.name,
                    message: format!(
                        "the {parameter} of `{}`, {value:?} {unit}, is in no E series from E3 to \
                         {densest}",
                        device.path()
                    ),
                    instances: vec![device.location().clone()],
                    nets: Vec::new(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use jolty_engine::{Conditions, Simulator};
    use jolty_ir::Circuit;
    use jolty_models::Library;
    use serde_json::{Value, json};

    use super::*;
    use crate::check;

    /// A device instance: its name, its model, its parameters and the net of each terminal.
    struct Device {
        name: &'static str,
        model: &'static str,
        parameters: Value,
        terminals: Vec<(&'static str, &'static str)>,
    }

    /// The diagnostics of a circuit whose top component has a ground port `0` and places
    /// `devices`.
    fn diagnostics(devices: &[Device]) -> Vec<Diagnostic> {
        let instances: Vec<Value> = devices
            .iter()
            .map(|device| {
                let terminals: serde_json::Map<String, Value> = device
                    .terminals
                    .iter()
                    .map(|&(terminal, net)| (terminal.to_owned(), json!(net)))
                    .collect();
                json!({"name": device.name, "of": {"model": device.model},
                       "parameters": device.parameters, "terminals": terminals})
            })
            .collect();
        let document = json!({
            "format": "jolty-circuit", "version": 4, "top": "test",
            "components": {"test": {"ports": [{"name": "0", "role": "ground"}],
                                    "instances": instances}}
        });
        let circuit = Circuit::from_json(&document.to_string()).expect("a valid circuit");
        let simulator = Simulator::new(&circuit, &Library::standard(), &Conditions::default())
            .expect("an elaborated circuit");
        check(&simulator)
    }

    fn device(
        name: &'static str,
        model: &'static str,
        parameters: Value,
        [first, second]: [&'static str; 2],
    ) -> Device {
        let [a, b] = match model {
            "voltage_source" | "current_source" => ["positive", "negative"],
            _ => ["a", "b"],
        };
        Device {
            name,
            model,
            parameters,
            terminals: vec![(a, first), (b, second)],
        }
    }

    fn transistor(
        name: &'static str,
        collector: &'static str,
        base: &'static str,
        emitter: &'static str,
    ) -> Device {
        Device {
            name,
            model: "bipolar_transistor",
            parameters: json!({}),
            terminals: vec![
                ("collector", collector),
                ("base", base),
                ("emitter", emitter),
            ],
        }
    }

    fn current_controlled(
        name: &'static str,
        [positive, negative]: [&'static str; 2],
        [control_positive, control_negative]: [&'static str; 2],
    ) -> Device {
        Device {
            name,
            model: "current_controlled_current_source",
            parameters: json!({"gain": 10.0}),
            terminals: vec![
                ("positive", positive),
                ("negative", negative),
                ("control_positive", control_positive),
                ("control_negative", control_negative),
            ],
        }
    }

    fn resistor(name: &'static str, a: &'static str, b: &'static str) -> Device {
        device(name, "resistor", json!({"resistance": 1e3}), [a, b])
    }

    fn capacitor(name: &'static str, a: &'static str, b: &'static str) -> Device {
        device(name, "capacitor", json!({"capacitance": 1e-9}), [a, b])
    }

    fn inductor(name: &'static str, a: &'static str, b: &'static str) -> Device {
        device(name, "inductor", json!({"inductance": 1e-3}), [a, b])
    }

    fn voltage(name: &'static str, positive: &'static str, negative: &'static str) -> Device {
        let level = json!({"waveform": {"dc": 1.0}});
        device(name, "voltage_source", level, [positive, negative])
    }

    fn current(name: &'static str, positive: &'static str, negative: &'static str) -> Device {
        let level = json!({"waveform": {"dc": 1e-3}});
        device(name, "current_source", level, [positive, negative])
    }

    fn rules(diagnostics: &[Diagnostic]) -> Vec<&'static str> {
        diagnostics
            .iter()
            .map(|diagnostic| diagnostic.rule)
            .collect()
    }

    #[test]
    fn a_sound_circuit_raises_nothing() {
        let found = diagnostics(&[
            voltage("V1", "in", "0"),
            resistor("R1", "in", "out"),
            capacitor("C1", "out", "0"),
            inductor("L1", "out", "load"),
            resistor("R2", "load", "0"),
            current("I1", "0", "load"),
        ]);
        assert_eq!(found, []);
    }

    #[test]
    fn non_standard_value() {
        let found = diagnostics(&[
            voltage("V1", "in", "0"),
            device(
                "R1",
                "resistor",
                json!({"resistance": 4812.3}),
                ["in", "out"],
            ),
            device(
                "R2",
                "resistor",
                json!({"resistance": 4870.0}),
                ["out", "0"],
            ),
            device(
                "C1",
                "capacitor",
                json!({"capacitance": 4.87e-9}),
                ["out", "0"],
            ),
            device(
                "L1",
                "inductor",
                json!({"inductance": 4.7e-6}),
                ["out", "0"],
            ),
        ]);
        assert_eq!(rules(&found), ["non-standard-value", "non-standard-value"]);
        assert_eq!(
            found[0].message,
            "the resistance of `R1`, 4812.3 Ω, is in no E series from E3 to E192"
        );
        assert_eq!(
            found[1].message,
            "the capacitance of `C1`, 4.87e-9 F, is in no E series from E3 to E24"
        );
        assert_eq!(found[1].instances[0].path, "C1");
    }

    #[test]
    fn unconnected_terminal() {
        let found = diagnostics(&[voltage("V1", "in", "0"), resistor("R1", "in", "R1.b")]);
        assert_eq!(rules(&found), ["unconnected-terminal"]);
        assert_eq!(found[0].message, "terminal `b` of `R1` connects to nothing");
        assert_eq!(found[0].nets, ["R1.b"]);
        let passing = diagnostics(&[voltage("V1", "in", "0"), resistor("R1", "in", "0")]);
        assert_eq!(passing, []);
    }

    #[test]
    fn shorted_source() {
        let found = diagnostics(&[voltage("V1", "in", "in"), resistor("R1", "in", "0")]);
        assert_eq!(rules(&found), ["shorted-source"]);
        assert_eq!(
            found[0].message,
            "`V1` has `positive` and `negative` both on `in`"
        );
        let current = diagnostics(&[
            current("I1", "0", "0"),
            resistor("R1", "0", "x"),
            resistor("R2", "x", "0"),
        ]);
        assert_eq!(rules(&current), ["shorted-source"]);
        let sensing = diagnostics(&[
            voltage("V1", "in", "0"),
            resistor("R1", "in", "0"),
            current_controlled("F1", ["out", "0"], ["in", "in"]),
            resistor("R2", "out", "0"),
        ]);
        assert_eq!(rules(&sensing), ["shorted-source"]);
        assert_eq!(
            sensing[0].message,
            "`F1` has `control_positive` and `control_negative` both on `in`"
        );
        let sensed = diagnostics(&[
            voltage("V1", "in", "0"),
            current_controlled("F1", ["out", "0"], ["in", "x"]),
            resistor("R1", "x", "0"),
            resistor("R2", "out", "0"),
        ]);
        assert_eq!(sensed, []);
    }

    #[test]
    fn shorted_device() {
        let found = diagnostics(&[
            voltage("V1", "in", "0"),
            resistor("R1", "in", "0"),
            capacitor("C1", "in", "in"),
        ]);
        assert_eq!(rules(&found), ["shorted-device"]);
        assert_eq!(found[0].message, "`C1` has `a` and `b` both on `in`");
        let joined = diagnostics(&[
            voltage("V1", "in", "0"),
            resistor("R1", "in", "0"),
            transistor("Q1", "in", "in", "in"),
        ]);
        assert_eq!(rules(&joined), ["shorted-device"]);
        assert_eq!(
            joined[0].message,
            "`Q1` has `collector`, `base` and `emitter` all on `in`"
        );
        // A diode-connected transistor joins its base and collector, and does something.
        let diode_connected = diagnostics(&[
            voltage("V1", "in", "0"),
            resistor("R1", "in", "x"),
            transistor("Q1", "x", "x", "0"),
        ]);
        assert_eq!(diode_connected, []);
    }

    #[test]
    fn voltage_loop() {
        let found = diagnostics(&[
            voltage("V1", "in", "0"),
            voltage("V2", "in", "0"),
            resistor("R1", "in", "0"),
        ]);
        assert_eq!(rules(&found), ["voltage-loop"]);
        assert_eq!(
            found[0].message,
            "`V1` and `V2` form a loop of voltage sources and inductors"
        );
        assert_eq!(found[0].nets, ["0", "in"]);
        let inductive = diagnostics(&[
            voltage("V1", "a", "0"),
            inductor("L1", "a", "b"),
            inductor("L2", "b", "0"),
            resistor("R1", "a", "0"),
        ]);
        assert_eq!(
            inductive[0].message,
            "`V1`, `L1` and `L2` form a loop of voltage sources and inductors"
        );
        let series = diagnostics(&[
            voltage("V1", "a", "b"),
            voltage("V2", "b", "0"),
            resistor("R1", "a", "0"),
        ]);
        assert_eq!(series, []);
    }

    #[test]
    fn no_dc_path() {
        let found = diagnostics(&[
            voltage("V1", "in", "0"),
            capacitor("C1", "in", "mid"),
            resistor("R1", "mid", "top"),
            capacitor("C2", "top", "0"),
        ]);
        assert_eq!(rules(&found), ["no-dc-path"]);
        assert_eq!(found[0].message, "`mid` and `top` have no DC path to `0`");
        let paths: Vec<&str> = found[0]
            .instances
            .iter()
            .map(|at| at.path.as_str())
            .collect();
        assert_eq!(paths, ["C1", "R1", "C2"]);
        let grounded = diagnostics(&[
            voltage("V1", "in", "0"),
            capacitor("C1", "in", "mid"),
            resistor("R1", "mid", "0"),
        ]);
        assert_eq!(grounded, []);
    }

    #[test]
    fn current_cutset() {
        let found = diagnostics(&[current("I1", "0", "top"), capacitor("C1", "top", "0")]);
        assert_eq!(rules(&found), ["current-cutset"]);
        assert_eq!(
            found[0].message,
            "`I1` drives current into `top`, which has no DC path to `0`"
        );
        let series = diagnostics(&[
            current("I1", "0", "mid"),
            current("I2", "mid", "out"),
            resistor("R1", "out", "0"),
        ]);
        assert_eq!(rules(&series), ["current-cutset"]);
        let loaded = diagnostics(&[current("I1", "0", "top"), resistor("R1", "top", "0")]);
        assert_eq!(loaded, []);
        // A source between two floating groups drives both, and a capacitor between them
        // touches both and drives neither.
        let between = diagnostics(&[
            current("I1", "a", "b"),
            capacitor("C1", "a", "0"),
            capacitor("C2", "b", "0"),
            capacitor("C3", "a", "b"),
        ]);
        assert_eq!(rules(&between), ["current-cutset", "current-cutset"]);
        let instances: Vec<Vec<&str>> = between
            .iter()
            .map(|found| found.instances.iter().map(|at| at.path.as_str()).collect())
            .collect();
        assert_eq!(instances, [["I1", "C1", "C3"], ["I1", "C2", "C3"]]);
        assert_eq!(
            between[1].message,
            "`I1` drives current into `b`, which has no DC path to `0`"
        );
    }

    fn gate(name: &'static str, terminals: Vec<(&'static str, &'static str)>) -> Device {
        let inputs: Vec<&str> = terminals
            .iter()
            .map(|&(terminal, _)| terminal)
            .filter(|&terminal| !["y", "supply", "ground"].contains(&terminal))
            .collect();
        let table = if inputs.len() == 1 { "10" } else { "1110" };
        let mut parameters = json!({
            "inputs": inputs,
            "outputs": [{"name": "y", "table": table}],
            "timing": {"fixed": {"rise": 1e-9, "fall": 1e-9, "transition": 1e-10}},
        });
        if terminals.iter().any(|&(terminal, _)| terminal == "supply") {
            parameters["electrical"] = json!({"input_capacitances": vec![0.0; inputs.len()]});
        }
        Device {
            name,
            model: "logic_cell",
            parameters,
            terminals,
        }
    }

    fn logic_source(name: &'static str, output: &'static str) -> Device {
        Device {
            name,
            model: "logic_source",
            parameters: json!({"initial": "0", "transition": 1e-10}),
            terminals: vec![("output", output)],
        }
    }

    /// A NAND of `a` and the inverse of its own output: every net one output drives and an
    /// input reads.
    fn latch() -> Vec<Device> {
        vec![
            logic_source("A", "a"),
            gate("U1", vec![("a", "a"), ("b", "c"), ("y", "b")]),
            gate("U2", vec![("a", "b"), ("y", "c")]),
        ]
    }

    #[test]
    fn a_sound_logic_circuit_raises_nothing() {
        assert_eq!(diagnostics(&latch()), []);
    }

    #[test]
    fn driver_conflict() {
        let mut devices = latch();
        devices.push(logic_source("B", "a"));
        let found = diagnostics(&devices);
        assert_eq!(rules(&found), ["driver-conflict"]);
        assert_eq!(found[0].message, "`A` and `B` drive the logic net `a`");
        assert_eq!(found[0].nets, ["a"]);
    }

    #[test]
    fn floating_input() {
        let mut devices = latch();
        devices.remove(0);
        devices.push(gate("U3", vec![("a", "a"), ("b", "e"), ("y", "d")]));
        devices.push(gate("U4", vec![("a", "d"), ("y", "e")]));
        let found = diagnostics(&devices);
        assert_eq!(rules(&found), ["floating-input"]);
        assert_eq!(
            found[0].message,
            "nothing drives the logic net `a`, which `U1` and `U3` read"
        );
    }

    #[test]
    fn a_logic_pin_or_supply_alone_connects_to_nothing() {
        let found = diagnostics(&[
            logic_source("A", "a"),
            gate(
                "U1",
                vec![("a", "a"), ("y", "y"), ("supply", "vdd"), ("ground", "0")],
            ),
        ]);
        let messages: Vec<&str> = found.iter().map(|found| found.message.as_str()).collect();
        assert_eq!(
            messages,
            [
                "terminal `y` of `U1` connects to nothing",
                "terminal `supply` of `U1` connects to nothing",
            ]
        );
        // Its connect elements' supply and ground are its own, which a source joins.
        let joined = diagnostics(&[
            voltage("V1", "vdd", "0"),
            voltage("V2", "in", "0"),
            gate(
                "U1",
                vec![("a", "in"), ("y", "y"), ("supply", "vdd"), ("ground", "0")],
            ),
            gate(
                "U2",
                vec![("a", "y"), ("y", "out"), ("supply", "vdd"), ("ground", "0")],
            ),
            resistor("R1", "out", "0"),
        ]);
        assert_eq!(joined, []);
    }

    #[test]
    fn a_floating_terminal_raises_one_rule() {
        let found = diagnostics(&[
            voltage("V1", "in", "0"),
            resistor("R1", "in", "0"),
            capacitor("C1", "in", "C1.b"),
        ]);
        assert_eq!(rules(&found), ["unconnected-terminal"]);
    }

    #[test]
    fn rules_are_named_and_ordered() {
        for rule in RULES {
            assert!(
                rule.name.split('-').all(|word| !word.is_empty()
                    && word.chars().all(|letter| letter.is_ascii_lowercase()))
            );
            assert!(rule.condition.ends_with('.'));
        }
    }

    #[test]
    fn joined_terminals_formats_zero_one_two_and_many_terminals() {
        let names =
            |items: &[&str]| -> Vec<String> { items.iter().map(|&item| item.to_owned()).collect() };
        assert_eq!(format_joined_terminals(&names(&[])), "");
        assert_eq!(format_joined_terminals(&names(&["a"])), "`a`");
        assert_eq!(
            format_joined_terminals(&names(&["a", "b"])),
            "`a` and `b` both"
        );
        assert_eq!(
            format_joined_terminals(&names(&["collector", "base", "emitter"])),
            "`collector`, `base` and `emitter` all"
        );
    }
}
