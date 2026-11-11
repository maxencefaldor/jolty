//! The rules, and the checks that find their diagnostics.

use std::collections::{HashMap, VecDeque};

use jolty_engine::Placed;
use jolty_models::BranchKind;

use crate::structure::{Partition, REFERENCE, Structure, ends, index, list};
use crate::{Diagnostic, Level, Rule};

/// The most nodes or instances a message names before it counts the rest.
const NAMED: usize = 5;

const UNCONNECTED_TERMINAL: Rule = Rule {
    name: "unconnected-terminal",
    level: Level::Error,
    condition: "A device's terminal joins a node that no other terminal joins and that is \
                not the ground: nothing connects to it.",
};

const SHORTED_SOURCE: Rule = Rule {
    name: "shorted-source",
    level: Level::Error,
    condition: "A voltage or current source, independent or controlled, has both ends of its \
                output on one node. A voltage source then holds its voltage across nothing, \
                and a current source drives its current through nothing.",
};

const SHORTED_DEVICE: Rule = Rule {
    name: "shorted-device",
    level: Level::Warn,
    condition: "A resistor, capacitor, inductor or diode has both terminals on one node, so \
                it carries no current and does nothing.",
};

const VOLTAGE_LOOP: Rule = Rule {
    name: "voltage-loop",
    level: Level::Error,
    condition: "Voltage sources and inductors, independent or controlled, form a loop. At DC \
                an inductor is a short, so the loop's sources alone fix the voltages around \
                it: they conflict, or leave the current around the loop undefined. Each \
                independent loop raises one diagnostic.",
};

const NO_DC_PATH: Rule = Rule {
    name: "no-dc-path",
    level: Level::Error,
    condition: "A group of nodes has no path to the ground through resistors, diodes, \
                inductors or voltage sources, and no current source drives it: it reaches \
                the ground only through capacitors, control inputs or nothing, so its DC \
                voltage is undefined. A group whose every node joins a single terminal is \
                left to unconnected-terminal.",
};

const CURRENT_CUTSET: Rule = Rule {
    name: "current-cutset",
    level: Level::Error,
    condition: "Current sources, independent or controlled, drive a group of nodes that has \
                no DC path to the ground, as no-dc-path defines it: at DC their current has \
                nowhere to go. A group whose every node joins a single terminal is left to \
                unconnected-terminal.",
};

/// Every rule, in the order a check reports them.
pub const RULES: &[Rule] = &[
    UNCONNECTED_TERMINAL,
    SHORTED_SOURCE,
    SHORTED_DEVICE,
    VOLTAGE_LOOP,
    NO_DC_PATH,
    CURRENT_CUTSET,
];

/// Each terminal alone on a node other than the reference.
pub(crate) fn unconnected_terminals(structure: &Structure<'_>, diagnostics: &mut Vec<Diagnostic>) {
    for device in structure.devices() {
        for (terminal, &node) in device.terminals().iter().zip(device.nodes()) {
            let node = index(node);
            if node != REFERENCE && structure.degree(node) == 1 {
                diagnostics.push(Diagnostic {
                    rule: UNCONNECTED_TERMINAL.name,
                    message: format!(
                        "terminal `{terminal}` of `{}` connects to nothing",
                        device.path()
                    ),
                    instances: vec![device.location().clone()],
                    nets: vec![structure.name(node).to_owned()],
                });
            }
        }
    }
}

/// Each source whose output joins a node to itself.
pub(crate) fn shorted_sources(structure: &Structure<'_>, diagnostics: &mut Vec<Diagnostic>) {
    shorted(structure, diagnostics, &SHORTED_SOURCE, |kind| {
        matches!(kind, BranchKind::VoltageSource | BranchKind::CurrentSource)
    });
}

/// Each resistor, capacitor, inductor or diode that joins a node to itself.
pub(crate) fn shorted_devices(structure: &Structure<'_>, diagnostics: &mut Vec<Diagnostic>) {
    shorted(structure, diagnostics, &SHORTED_DEVICE, |kind| {
        matches!(
            kind,
            BranchKind::Conductive | BranchKind::Capacitive | BranchKind::Inductive
        )
    });
}

fn shorted(
    structure: &Structure<'_>,
    diagnostics: &mut Vec<Diagnostic>,
    rule: &Rule,
    applies: impl Fn(BranchKind) -> bool,
) {
    for device in structure.devices() {
        for &branch in device.branches() {
            let (from, to) = ends(device, branch);
            if from == to && applies(branch.kind) {
                let terminals = device.terminals();
                diagnostics.push(Diagnostic {
                    rule: rule.name,
                    message: format!(
                        "`{}` has `{}` and `{}` both on `{}`",
                        device.path(),
                        terminals[branch.from],
                        terminals[branch.to],
                        structure.name(from),
                    ),
                    instances: vec![device.location().clone()],
                    nets: vec![structure.name(from).to_owned()],
                });
            }
        }
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
pub(crate) fn no_dc_paths(structure: &Structure<'_>, diagnostics: &mut Vec<Diagnostic>) {
    for group in floating_groups(structure) {
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
pub(crate) fn current_cutsets(structure: &Structure<'_>, diagnostics: &mut Vec<Diagnostic>) {
    let devices = structure.devices();
    for group in floating_groups(structure) {
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
struct Group {
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

    fn locations(&self, structure: &Structure<'_>) -> Vec<jolty_engine::Location> {
        let devices = structure.devices();
        self.devices
            .iter()
            .map(|&device| devices[device].location().clone())
            .collect()
    }
}

/// The groups of nodes that branches conducting at DC join to each other but not to the
/// reference, leaving out the groups whose every node joins a single terminal.
fn floating_groups(structure: &Structure<'_>) -> Vec<Group> {
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
    for group in &mut groups {
        let inside = |node: usize| group.nodes.binary_search(&node).is_ok();
        let mut others = Vec::new();
        for (number, device) in devices.iter().enumerate() {
            if drives(device, &inside) {
                group.sources.push(number);
            } else if device.nodes().iter().any(|&node| inside(index(node))) {
                others.push(number);
            }
        }
        group.devices = group.sources.iter().copied().chain(others).collect();
    }
    groups
}

/// Whether `device` has a current source with exactly one end of its output `inside`.
fn drives(device: &Placed, inside: &impl Fn(usize) -> bool) -> bool {
    device.branches().iter().any(|&branch| {
        let (from, to) = ends(device, branch);
        branch.kind == BranchKind::CurrentSource && inside(from) != inside(to)
    })
}

#[cfg(test)]
mod tests {
    use jolty_engine::Simulator;
    use jolty_ir::Circuit;
    use jolty_models::Library;
    use serde_json::{Value, json};

    use super::*;
    use crate::check;

    /// A device instance: its id, its model, its parameters and the net of each terminal.
    struct Device {
        id: &'static str,
        model: &'static str,
        parameters: Value,
        terminals: [(&'static str, &'static str); 2],
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
                json!({"id": device.id, "of": {"device": device.model},
                       "parameters": device.parameters, "terminals": terminals})
            })
            .collect();
        let document = json!({
            "format": "jolty-circuit", "version": 1, "top": "test",
            "components": {"test": {"ports": [{"name": "0", "role": "ground"}],
                                    "instances": instances}}
        });
        let circuit = Circuit::from_json(&document.to_string()).expect("a valid circuit");
        let simulator =
            Simulator::new(&circuit, &Library::standard()).expect("an elaborated circuit");
        check(&simulator)
    }

    fn device(
        id: &'static str,
        model: &'static str,
        parameters: Value,
        [first, second]: [&'static str; 2],
    ) -> Device {
        let [a, b] = match model {
            "voltage_source" | "current_source" => ["positive", "negative"],
            _ => ["a", "b"],
        };
        Device {
            id,
            model,
            parameters,
            terminals: [(a, first), (b, second)],
        }
    }

    fn resistor(id: &'static str, a: &'static str, b: &'static str) -> Device {
        device(id, "resistor", json!({"resistance": 1e3}), [a, b])
    }

    fn capacitor(id: &'static str, a: &'static str, b: &'static str) -> Device {
        device(id, "capacitor", json!({"capacitance": 1e-9}), [a, b])
    }

    fn inductor(id: &'static str, a: &'static str, b: &'static str) -> Device {
        device(id, "inductor", json!({"inductance": 1e-3}), [a, b])
    }

    fn voltage(id: &'static str, positive: &'static str, negative: &'static str) -> Device {
        let level = json!({"waveform": {"dc": 1.0}});
        device(id, "voltage_source", level, [positive, negative])
    }

    fn current(id: &'static str, positive: &'static str, negative: &'static str) -> Device {
        let level = json!({"waveform": {"dc": 1e-3}});
        device(id, "current_source", level, [positive, negative])
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
}
