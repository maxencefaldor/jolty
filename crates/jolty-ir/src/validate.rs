use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::{Circuit, Component, Name, Of};

/// A reason a circuit's structure is invalid.
///
/// These are the facts the format itself can state. Whether a device model exists and what
/// terminals and parameters it takes is checked when the circuit is elaborated.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Problem {
    /// The top component is not defined.
    #[error("the top component `{0}` is not defined")]
    UndefinedTop(Name),
    /// A component declares the same port twice.
    #[error("component `{component}` declares port `{port}` twice")]
    DuplicatePort {
        /// The component.
        component: Name,
        /// The repeated port.
        port: Name,
    },
    /// A component places two instances with the same id.
    #[error("component `{component}` places two instances named `{id}`")]
    DuplicateInstance {
        /// The component.
        component: Name,
        /// The repeated id.
        id: Name,
    },
    /// An instance refers to a component that is not defined.
    #[error("instance `{component}/{id}` is of undefined component `{target}`")]
    UndefinedComponent {
        /// The component that places the instance.
        component: Name,
        /// The instance.
        id: Name,
        /// The missing component.
        target: Name,
    },
    /// An instance of a component leaves one of its ports unconnected.
    #[error("instance `{component}/{id}` leaves port `{port}` of `{target}` unconnected")]
    UnconnectedPort {
        /// The component that places the instance.
        component: Name,
        /// The instance.
        id: Name,
        /// The instanced component.
        target: Name,
        /// The unconnected port.
        port: Name,
    },
    /// An instance of a component connects a terminal its component has no port for.
    #[error("instance `{component}/{id}` connects `{terminal}`, which `{target}` has no port for")]
    UnknownPort {
        /// The component that places the instance.
        component: Name,
        /// The instance.
        id: Name,
        /// The instanced component.
        target: Name,
        /// The terminal without a port.
        terminal: Name,
    },
    /// An instance of a component sets parameters, which only devices take.
    #[error("instance `{component}/{id}` sets parameters, which only device instances take")]
    ComponentParameters {
        /// The component that places the instance.
        component: Name,
        /// The instance.
        id: Name,
    },
    /// A component instances itself, directly or through others.
    #[error("component `{}` contains itself: {}", .cycle[0], join(.cycle, " -> "))]
    Recursive {
        /// The components around the cycle, starting and ending with the same one.
        cycle: Vec<Name>,
    },
}

/// Every [`Problem`] of a circuit, in a stable order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Problems(Vec<Problem>);

impl Problems {
    /// The problems, in the order they were found.
    pub fn iter(&self) -> std::slice::Iter<'_, Problem> {
        self.0.iter()
    }

    /// The number of problems, never zero.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Always false: a [`Problems`] holds at least one problem.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<'a> IntoIterator for &'a Problems {
    type Item = &'a Problem;
    type IntoIter = std::slice::Iter<'a, Problem>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl fmt::Display for Problems {
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

impl std::error::Error for Problems {}

impl Circuit {
    /// Checks the circuit's structure and returns every problem found.
    ///
    /// # Errors
    ///
    /// Returns [`Problems`] when any check fails.
    pub fn validate(&self) -> Result<(), Problems> {
        let mut problems = Vec::new();
        if !self.components.contains_key(&self.top) {
            problems.push(Problem::UndefinedTop(self.top.clone()));
        }
        for (name, component) in &self.components {
            check_component(name, component, &self.components, &mut problems);
        }
        find_cycles(&self.components, &mut problems);
        if problems.is_empty() {
            Ok(())
        } else {
            Err(Problems(problems))
        }
    }
}

fn check_component(
    name: &Name,
    component: &Component,
    components: &BTreeMap<Name, Component>,
    problems: &mut Vec<Problem>,
) {
    let mut ports = BTreeSet::new();
    for port in &component.ports {
        if !ports.insert(&port.name) {
            problems.push(Problem::DuplicatePort {
                component: name.clone(),
                port: port.name.clone(),
            });
        }
    }
    let mut ids = BTreeSet::new();
    for instance in &component.instances {
        if !ids.insert(&instance.id) {
            problems.push(Problem::DuplicateInstance {
                component: name.clone(),
                id: instance.id.clone(),
            });
        }
        let Of::Component(target) = &instance.of else {
            continue;
        };
        if !instance.parameters.is_empty() {
            problems.push(Problem::ComponentParameters {
                component: name.clone(),
                id: instance.id.clone(),
            });
        }
        let Some(definition) = components.get(target) else {
            problems.push(Problem::UndefinedComponent {
                component: name.clone(),
                id: instance.id.clone(),
                target: target.clone(),
            });
            continue;
        };
        for port in &definition.ports {
            if !instance.terminals.contains_key(&port.name) {
                problems.push(Problem::UnconnectedPort {
                    component: name.clone(),
                    id: instance.id.clone(),
                    target: target.clone(),
                    port: port.name.clone(),
                });
            }
        }
        for terminal in instance.terminals.keys() {
            if !definition.ports.iter().any(|port| &port.name == terminal) {
                problems.push(Problem::UnknownPort {
                    component: name.clone(),
                    id: instance.id.clone(),
                    target: target.clone(),
                    terminal: terminal.clone(),
                });
            }
        }
    }
}

/// Reports each cycle of the component graph once, by a depth-first search.
fn find_cycles(components: &BTreeMap<Name, Component>, problems: &mut Vec<Problem>) {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Mark {
        Open,
        Done,
    }

    fn visit<'a>(
        name: &'a Name,
        components: &'a BTreeMap<Name, Component>,
        marks: &mut BTreeMap<&'a Name, Mark>,
        path: &mut Vec<&'a Name>,
        problems: &mut Vec<Problem>,
    ) {
        marks.insert(name, Mark::Open);
        path.push(name);
        let children =
            components[name]
                .instances
                .iter()
                .filter_map(|instance| match &instance.of {
                    Of::Component(target) if components.contains_key(target) => Some(target),
                    _ => None,
                });
        for child in children {
            match marks.get(child) {
                Some(Mark::Open) => {
                    let start = path
                        .iter()
                        .position(|&n| n == child)
                        .expect("an open component is on the path");
                    let mut cycle: Vec<Name> = path[start..].iter().map(|&n| n.clone()).collect();
                    cycle.push(child.clone());
                    problems.push(Problem::Recursive { cycle });
                }
                Some(Mark::Done) => {}
                None => visit(child, components, marks, path, problems),
            }
        }
        path.pop();
        marks.insert(name, Mark::Done);
    }

    let mut marks = BTreeMap::new();
    for name in components.keys() {
        if !marks.contains_key(name) {
            visit(name, components, &mut marks, &mut Vec::new(), problems);
        }
    }
}

fn join(names: &[Name], separator: &str) -> String {
    names
        .iter()
        .map(Name::as_str)
        .collect::<Vec<_>>()
        .join(separator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Instance, Port, Role};

    fn name(text: &str) -> Name {
        Name::new(text).unwrap()
    }

    fn instance_of(id: &str, target: &str, terminals: &[(&str, &str)]) -> Instance {
        Instance {
            id: name(id),
            of: Of::Component(name(target)),
            parameters: crate::Parameters::new(),
            terminals: terminals.iter().map(|&(t, n)| (name(t), name(n))).collect(),
            source: None,
        }
    }

    fn circuit(components: Vec<(&str, Component)>) -> Circuit {
        Circuit {
            top: name("top"),
            components: components.into_iter().map(|(n, c)| (name(n), c)).collect(),
        }
    }

    fn cell() -> Component {
        Component {
            ports: vec![
                Port {
                    name: name("a"),
                    role: Role::Signal,
                },
                Port {
                    name: name("b"),
                    role: Role::Signal,
                },
            ],
            instances: vec![],
        }
    }

    #[test]
    fn accepts_a_valid_hierarchy() {
        let top = Component {
            ports: vec![Port {
                name: name("ground"),
                role: Role::Ground,
            }],
            instances: vec![instance_of("X1", "cell", &[("a", "in"), ("b", "ground")])],
        };
        assert_eq!(
            circuit(vec![("top", top), ("cell", cell())]).validate(),
            Ok(())
        );
    }

    #[test]
    fn reports_every_problem() {
        let top = Component {
            ports: vec![
                Port {
                    name: name("p"),
                    role: Role::Signal,
                },
                Port {
                    name: name("p"),
                    role: Role::Signal,
                },
            ],
            instances: vec![
                instance_of("X1", "cell", &[("a", "n1"), ("c", "n2")]),
                instance_of("X1", "missing", &[]),
            ],
        };
        let problems = circuit(vec![("top", top), ("cell", cell())])
            .validate()
            .unwrap_err();
        let messages: Vec<String> = problems.iter().map(ToString::to_string).collect();
        assert_eq!(
            messages,
            [
                "component `top` declares port `p` twice",
                "instance `top/X1` leaves port `b` of `cell` unconnected",
                "instance `top/X1` connects `c`, which `cell` has no port for",
                "component `top` places two instances named `X1`",
                "instance `top/X1` is of undefined component `missing`",
            ]
        );
    }

    #[test]
    fn reports_an_undefined_top() {
        let problems = circuit(vec![("cell", cell())]).validate().unwrap_err();
        assert_eq!(
            problems.iter().next(),
            Some(&Problem::UndefinedTop(name("top")))
        );
    }

    #[test]
    fn reports_a_cycle_once() {
        let a = Component {
            instances: vec![instance_of("X", "b", &[])],
            ..Component::default()
        };
        let b = Component {
            instances: vec![instance_of("Y", "a", &[])],
            ..Component::default()
        };
        let top = Component {
            instances: vec![instance_of("Z", "a", &[])],
            ..Component::default()
        };
        let problems = circuit(vec![("top", top), ("a", a), ("b", b)])
            .validate()
            .unwrap_err();
        assert_eq!(problems.len(), 1);
        assert_eq!(
            problems.to_string(),
            "component `a` contains itself: a -> b -> a"
        );
    }
}
