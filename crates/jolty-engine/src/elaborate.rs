//! Elaboration: flattening a circuit's hierarchy into devices bound to unknowns.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::Arc;

use jolty_ir::{Circuit, Component, Instance, Name, Of, Parameters, Problem, Role, Source};
use jolty_models::{Binding, Device, Library, Model, Node, Quantity, Unknown, UnknownInfo};
use jolty_numerics::{Entry, Pattern, PatternBuilder};

/// A circuit flattened into devices bound to unknowns, with the pattern of its matrix.
///
/// The node voltages come first among the unknowns, in the order their nets are first met,
/// then the devices' internal unknowns.
#[derive(Debug)]
pub(crate) struct System {
    pub(crate) unknowns: Vec<UnknownInfo>,
    pub(crate) devices: Vec<Placed>,
    pub(crate) pattern: Arc<Pattern>,
    /// The diagonal entry of every node voltage's equation, where gmin stepping adds a
    /// conductance to the reference.
    pub(crate) node_diagonals: Vec<(Unknown, Entry)>,
}

/// A device and the path of its instance.
#[derive(Debug)]
pub(crate) struct Placed {
    pub(crate) path: String,
    pub(crate) device: Box<dyn Device>,
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
        terminal: &'static str,
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

/// Flattens `circuit` and binds each device to its unknowns.
pub(crate) fn elaborate(circuit: &Circuit, library: &Library) -> Result<System, ElaborationError> {
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
    let grounds: Vec<&Name> = top
        .ports
        .iter()
        .filter(|port| port.role == Role::Ground)
        .map(|port| &port.name)
        .collect();
    let reference = match grounds.as_slice() {
        [reference] => *reference,
        [] => return Err(ElaborationProblem::NoReference(circuit.top.clone()).into()),
        _ => {
            return Err(ElaborationProblem::SeveralReferences {
                top: circuit.top.clone(),
                count: grounds.len(),
            }
            .into());
        }
    };

    let mut walk = Walk {
        circuit,
        library,
        unknowns: Vec::new(),
        nets: HashMap::new(),
        devices: Vec::new(),
        problems: Vec::new(),
    };
    walk.component(top, "", &BTreeMap::from([(reference, None)]));
    let Walk {
        mut unknowns,
        devices,
        mut problems,
        ..
    } = walk;

    let mut builder = PatternBuilder::new();
    let node_diagonals: Vec<(Unknown, Entry)> = (0..unknowns.len())
        .map(|index| (Unknown::new(index), builder.entry(index, index)))
        .collect();
    let mut placed = Vec::with_capacity(devices.len());
    for pending in devices {
        let mut binding = Binding::new(
            &pending.at.path,
            &pending.terminals,
            &mut unknowns,
            &mut builder,
        );
        match pending.model.instantiate(pending.parameters, &mut binding) {
            Ok(device) => placed.push(Placed {
                path: pending.at.path,
                device,
            }),
            Err(error) => problems.push(ElaborationProblem::Parameters {
                at: pending.at,
                message: error.0,
            }),
        }
    }
    if !problems.is_empty() {
        return Err(ElaborationError(problems));
    }
    let pattern = Arc::new(builder.finish(unknowns.len()));
    Ok(System {
        unknowns,
        devices: placed,
        pattern,
        node_diagonals,
    })
}

/// A device instance found by the walk, waiting for every node to be numbered.
struct Pending<'a> {
    at: Location,
    model: &'a dyn Model,
    parameters: &'a Parameters,
    terminals: Vec<Node>,
}

/// The state of a walk down the hierarchy.
struct Walk<'a> {
    circuit: &'a Circuit,
    library: &'a Library,
    unknowns: Vec<UnknownInfo>,
    nets: HashMap<String, Unknown>,
    devices: Vec<Pending<'a>>,
    problems: Vec<ElaborationProblem>,
}

impl<'a> Walk<'a> {
    /// Walks `component`, whose instances' paths start with `prefix` and whose ports connect
    /// to the nodes in `ports`. The circuit has been validated, so every referenced component
    /// exists and every port of a component instance is connected.
    fn component(
        &mut self,
        component: &'a Component,
        prefix: &str,
        ports: &BTreeMap<&'a Name, Node>,
    ) {
        for instance in &component.instances {
            let at = Location {
                path: format!("{prefix}{}", instance.id),
                source: instance.source.clone(),
            };
            match &instance.of {
                Of::Device(model) => self.device(instance, at, model, prefix, ports),
                Of::Component(target) => {
                    let definition = &self.circuit.components[target];
                    let inner: BTreeMap<&Name, Node> = definition
                        .ports
                        .iter()
                        .map(|port| {
                            (
                                &port.name,
                                self.node(prefix, ports, &instance.terminals[&port.name]),
                            )
                        })
                        .collect();
                    self.component(definition, &format!("{}/", at.path), &inner);
                }
            }
        }
    }

    fn device(
        &mut self,
        instance: &'a Instance,
        at: Location,
        model_name: &Name,
        prefix: &str,
        ports: &BTreeMap<&'a Name, Node>,
    ) {
        let Some(model) = self.library.get(model_name.as_str()) else {
            self.problems.push(ElaborationProblem::UnknownModel {
                at,
                model: model_name.clone(),
            });
            return;
        };
        let expected = model.terminals();
        let mut complete = true;
        for terminal in instance.terminals.keys() {
            if !expected.contains(&terminal.as_str()) {
                complete = false;
                self.problems.push(ElaborationProblem::UnknownTerminal {
                    at: at.clone(),
                    model: model_name.clone(),
                    terminal: terminal.clone(),
                });
            }
        }
        let mut terminals = Vec::with_capacity(expected.len());
        for &terminal in expected {
            if let Some(net) = instance.terminals.get(terminal) {
                terminals.push(self.node(prefix, ports, net));
            } else {
                complete = false;
                self.problems.push(ElaborationProblem::UnconnectedTerminal {
                    at: at.clone(),
                    model: model_name.clone(),
                    terminal,
                });
            }
        }
        if complete {
            self.devices.push(Pending {
                at,
                model,
                parameters: &instance.parameters,
                terminals,
            });
        }
    }

    /// The node of `net` in the component at `prefix`: a port's node from the parent, or the
    /// component's own net, numbered when first met.
    fn node(&mut self, prefix: &str, ports: &BTreeMap<&'a Name, Node>, net: &Name) -> Node {
        if let Some(&node) = ports.get(net) {
            return node;
        }
        let name = format!("{prefix}{net}");
        if let Some(&unknown) = self.nets.get(&name) {
            return Some(unknown);
        }
        let unknown = Unknown::new(self.unknowns.len());
        self.unknowns.push(UnknownInfo {
            name: name.clone(),
            quantity: Quantity::Voltage,
        });
        self.nets.insert(name, unknown);
        Some(unknown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn elaborate_json(components: &str) -> Result<System, ElaborationError> {
        let header = r#""format": "jolty-circuit", "version": 1, "top": "top""#;
        let text = format!(r#"{{{header}, "components": {components}}}"#);
        elaborate(&Circuit::from_json(&text).unwrap(), &Library::standard())
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
        let paths: Vec<&str> = system
            .devices
            .iter()
            .map(|placed| placed.path.as_str())
            .collect();
        assert_eq!(paths, ["V1", "X1/R1", "X1/R2"]);
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
             instance `R1` (amp.py:9): resistance must be positive, not -1"
        );
    }

    #[test]
    fn requires_exactly_one_reference() {
        let error = elaborate_json(r#"{"top": {"instances": []}}"#).unwrap_err();
        assert_eq!(
            error.problems(),
            [ElaborationProblem::NoReference(Name::new("top").unwrap())]
        );
    }
}
