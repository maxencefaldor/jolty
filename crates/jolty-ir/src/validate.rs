use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::{Circuit, Component, Lot, Name, Of, Stress};

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
    /// An instance gives a tolerance to a parameter it gives no number to, or a tolerance
    /// its distribution does not [take](crate::Distribution::takes).
    #[error(
        "instance `{component}/{id}` gives `{parameter}` a tolerance of {relative}, which \
         needs a number for it and a tolerance at least 0, and below 1 unless its logarithm \
         spreads"
    )]
    InvalidTolerance {
        /// The component that places the instance.
        component: Name,
        /// The instance.
        id: Name,
        /// The parameter given the tolerance.
        parameter: String,
        /// The tolerance, in its shortest exact form.
        relative: String,
    },
    /// An instance takes a parameter from a lot whose tolerance its distribution does not
    /// [take](crate::Distribution::takes).
    #[error(
        "instance `{component}/{id}` takes `{parameter}` from lot `{lot}` with a tolerance \
         of {relative}, which needs to be at least 0, and below 1 unless its logarithm \
         spreads"
    )]
    InvalidLot {
        /// The component that places the instance.
        component: Name,
        /// The instance.
        id: Name,
        /// The parameter.
        parameter: String,
        /// The lot.
        lot: Name,
        /// The lot's tolerance, in its shortest exact form.
        relative: String,
    },
    /// An instance states a rating that names terminals it does not connect, or not as
    /// many as its stress needs, or whose bounds are not finite and in order.
    #[error(
        "instance `{component}/{id}` states a rating of its {stress} {at}from {minimum} to \
         {maximum}, which needs {needed} and finite bounds, at least one and in order, a \
         power only a positive maximum and a temperature positive bounds"
    )]
    InvalidRating {
        /// The component that places the instance.
        component: Name,
        /// The instance.
        id: Name,
        /// The stress rated: `voltage`, `current`, `power` or `temperature`.
        stress: String,
        /// Where, as the terminals named, such as `at `collector`, `emitter` `.
        at: String,
        /// The minimum, in its shortest exact form, or `none`.
        minimum: String,
        /// The maximum, in its shortest exact form, or `none`.
        maximum: String,
        /// How many terminals the stress needs, in words.
        needed: &'static str,
    },
    /// An instance names a part whose number is empty or spaced at either end.
    #[error(
        "instance `{component}/{id}` is the part {part:?}, but a part number is neither \
         empty nor spaced at either end"
    )]
    InvalidPart {
        /// The component that places the instance.
        component: Name,
        /// The instance.
        id: Name,
        /// The part number, as given.
        part: String,
    },
    /// Two instances spread one parameter of one lot differently.
    #[error(
        "instances `{first}` and `{second}` spread `{parameter}` of lot `{lot}` differently, \
         and a lot spreads each parameter one way"
    )]
    InconsistentLot {
        /// The lot.
        lot: Name,
        /// The parameter.
        parameter: String,
        /// The first instance that spreads it, as `component/id`.
        first: String,
        /// An instance that spreads it otherwise, as `component/id`.
        second: String,
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
        check_lots(&self.components, &mut problems);
        find_cycles(&self.components, &mut problems);
        if problems.is_empty() {
            Ok(())
        } else {
            Err(Problems(problems))
        }
    }
}

/// Checks that every instance taking a parameter from a lot spreads it as the first one
/// does.
fn check_lots(components: &BTreeMap<Name, Component>, problems: &mut Vec<Problem>) {
    let mut first: BTreeMap<(&Name, &str), (&Lot, String)> = BTreeMap::new();
    for (name, component) in components {
        for instance in &component.instances {
            for (parameter, tolerance) in &instance.tolerances {
                let Some(lot) = &tolerance.lot else {
                    continue;
                };
                let at = format!("{name}/{}", instance.name);
                match first.get(&(&lot.name, parameter.as_str())) {
                    None => {
                        first.insert((&lot.name, parameter), (lot, at));
                    }
                    Some((spread, before)) if *spread != lot => {
                        problems.push(Problem::InconsistentLot {
                            lot: lot.name.clone(),
                            parameter: parameter.clone(),
                            first: before.clone(),
                            second: at,
                        });
                    }
                    Some(_) => {}
                }
            }
        }
    }
}

/// Checks that each rating of `instance`, placed in `name`, names as many of its terminals
/// as its stress needs and bounds it finitely and in order.
fn check_ratings(name: &Name, instance: &crate::Instance, problems: &mut Vec<Problem>) {
    for rating in &instance.ratings {
        let needed = rating.stress.terminals();
        let own = rating
            .terminals
            .iter()
            .all(|terminal| instance.terminals.contains_key(terminal));
        let distinct = rating.terminals.len() < 2 || rating.terminals[0] != rating.terminals[1];
        let finite = [rating.minimum, rating.maximum]
            .iter()
            .flatten()
            .all(|bound| bound.is_finite());
        let ordered = match (rating.minimum, rating.maximum) {
            (Some(minimum), Some(maximum)) => minimum <= maximum,
            (None, None) => false,
            _ => true,
        };
        let powered = rating.stress != Stress::Power
            || (rating.minimum.is_none() && rating.maximum.is_some_and(|maximum| maximum > 0.0));
        let absolute = rating.stress != Stress::Temperature
            || [rating.minimum, rating.maximum]
                .iter()
                .flatten()
                .all(|bound| *bound > 0.0);
        let bounded = finite && ordered && powered && absolute;
        if rating.terminals.len() != needed || !own || !distinct || !bounded {
            let at = if rating.terminals.is_empty() {
                String::new()
            } else {
                let named: Vec<String> = rating
                    .terminals
                    .iter()
                    .map(|terminal| format!("`{terminal}`"))
                    .collect();
                format!("at {} ", named.join(" over "))
            };
            problems.push(Problem::InvalidRating {
                component: name.clone(),
                id: instance.name.clone(),
                stress: rating.stress.as_str().to_owned(),
                at,
                minimum: rating
                    .minimum
                    .map_or_else(|| "none".to_owned(), |minimum| format!("{minimum:?}")),
                maximum: rating
                    .maximum
                    .map_or_else(|| "none".to_owned(), |maximum| format!("{maximum:?}")),
                needed: match rating.stress {
                    Stress::Voltage => "two distinct terminals of its own",
                    Stress::Current => "one of its terminals",
                    Stress::Power | Stress::Temperature => "no terminals",
                },
            });
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
        if !ids.insert(&instance.name) {
            problems.push(Problem::DuplicateInstance {
                component: name.clone(),
                id: instance.name.clone(),
            });
        }
        for (parameter, tolerance) in &instance.tolerances {
            let numeric = crate::parameter(&instance.parameters, parameter)
                .is_some_and(serde_json::Value::is_number);
            if !numeric || !tolerance.distribution.takes(tolerance.relative) {
                problems.push(Problem::InvalidTolerance {
                    component: name.clone(),
                    id: instance.name.clone(),
                    parameter: parameter.clone(),
                    relative: format!("{:?}", tolerance.relative),
                });
            }
            if let Some(lot) = &tolerance.lot
                && !lot.distribution.takes(lot.relative)
            {
                problems.push(Problem::InvalidLot {
                    component: name.clone(),
                    id: instance.name.clone(),
                    parameter: parameter.clone(),
                    lot: lot.name.clone(),
                    relative: format!("{:?}", lot.relative),
                });
            }
        }
        check_ratings(name, instance, problems);
        if let Some(part) = &instance.part
            && (part.is_empty() || part.trim() != part)
        {
            problems.push(Problem::InvalidPart {
                component: name.clone(),
                id: instance.name.clone(),
                part: part.clone(),
            });
        }
        let Of::Component(target) = &instance.of else {
            continue;
        };
        if !instance.parameters.is_empty() {
            problems.push(Problem::ComponentParameters {
                component: name.clone(),
                id: instance.name.clone(),
            });
        }
        let Some(definition) = components.get(target) else {
            problems.push(Problem::UndefinedComponent {
                component: name.clone(),
                id: instance.name.clone(),
                target: target.clone(),
            });
            continue;
        };
        for port in &definition.ports {
            if !instance.terminals.contains_key(&port.name) {
                problems.push(Problem::UnconnectedPort {
                    component: name.clone(),
                    id: instance.name.clone(),
                    target: target.clone(),
                    port: port.name.clone(),
                });
            }
        }
        for terminal in instance.terminals.keys() {
            if !definition.ports.iter().any(|port| &port.name == terminal) {
                problems.push(Problem::UnknownPort {
                    component: name.clone(),
                    id: instance.name.clone(),
                    target: target.clone(),
                    terminal: terminal.clone(),
                });
            }
        }
    }
}

/// Returns the distinct defined component targets instantiated by `component`, in the order
/// first seen.
fn distinct_children<'a>(
    component: &'a Component,
    components: &BTreeMap<Name, Component>,
) -> Vec<&'a Name> {
    let mut seen = BTreeSet::new();
    let mut children = Vec::new();
    for instance in &component.instances {
        if let Of::Component(target) = &instance.of
            && components.contains_key(target)
            && seen.insert(target)
        {
            children.push(target);
        }
    }
    children
}

/// Reports each cycle of the component graph once, by an iterative depth-first search.
fn find_cycles(components: &BTreeMap<Name, Component>, problems: &mut Vec<Problem>) {
    let mut visited: BTreeSet<&Name> = BTreeSet::new();
    let mut path: Vec<&Name> = Vec::new();
    let mut on_path: BTreeMap<&Name, usize> = BTreeMap::new();
    let mut stack: Vec<(Vec<&Name>, usize)> = Vec::new();
    for (root, component) in components {
        if !visited.insert(root) {
            continue;
        }
        on_path.insert(root, path.len());
        path.push(root);
        stack.push((distinct_children(component, components), 0));
        while let Some((children, next)) = stack.last_mut() {
            let index = *next;
            if index < children.len() {
                *next += 1;
                let child = children[index];
                if let Some(&start) = on_path.get(child) {
                    let mut cycle: Vec<Name> =
                        path[start..].iter().map(|&node| node.clone()).collect();
                    cycle.push(child.clone());
                    problems.push(Problem::Recursive { cycle });
                } else if visited.insert(child) {
                    on_path.insert(child, path.len());
                    path.push(child);
                    stack.push((distinct_children(&components[child], components), 0));
                }
            } else {
                stack.pop();
                if let Some(finished) = path.pop() {
                    on_path.remove(finished);
                }
            }
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
            name: name(id),
            of: Of::Component(name(target)),
            parameters: crate::Parameters::new(),
            terminals: terminals.iter().map(|&(t, n)| (name(t), name(n))).collect(),
            tolerances: BTreeMap::new(),
            ratings: Vec::new(),
            part: None,
            source_location: None,
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
    fn checks_ratings() {
        let rating = |stress, terminals: &[&str], minimum, maximum| crate::Rating {
            stress,
            terminals: terminals.iter().map(|terminal| name(terminal)).collect(),
            maximum,
            minimum,
        };
        let mut transistor = instance_of("Q1", "unused", &[("collector", "c"), ("base", "b")]);
        transistor.of = Of::Device(name("bipolar_transistor"));
        transistor.ratings = vec![
            rating(Stress::Voltage, &["collector", "base"], None, Some(60.0)),
            rating(Stress::Current, &["collector"], Some(-0.2), Some(0.2)),
            rating(Stress::Power, &[], None, Some(0.625)),
            rating(Stress::Voltage, &["collector"], None, Some(40.0)),
            rating(Stress::Voltage, &["collector", "emitter"], None, Some(40.0)),
            rating(Stress::Voltage, &["base", "base"], None, Some(6.0)),
            rating(Stress::Power, &[], None, Some(0.0)),
            rating(Stress::Power, &[], Some(0.0), Some(1.0)),
            rating(Stress::Current, &["base"], Some(0.2), Some(0.1)),
            rating(Stress::Current, &["base"], None, Some(f64::INFINITY)),
            rating(Stress::Current, &["base"], None, None),
            rating(Stress::Temperature, &[], Some(218.15), Some(423.15)),
            rating(Stress::Temperature, &[], Some(0.0), Some(423.15)),
            rating(Stress::Temperature, &["thermal"], None, Some(423.15)),
        ];
        let top = Component {
            instances: vec![transistor],
            ..Component::default()
        };
        let problems = circuit(vec![("top", top)]).validate().unwrap_err();
        let messages: Vec<String> = problems.0.iter().map(ToString::to_string).collect();
        assert_eq!(messages.len(), 10, "{messages:?}");
        assert_eq!(
            messages[0],
            "instance `top/Q1` states a rating of its voltage at `collector` from none to 40.0, \
             which needs two distinct terminals of its own and finite bounds, at least one \
             and in order, a power only a positive maximum and a temperature positive bounds"
        );
        assert!(messages[1].contains("its voltage at `collector` over `emitter` from none"));
        assert!(messages[3].contains("its power from none to 0.0, which needs no terminals"));
        assert!(messages[7].contains("its current at `base` from none to none"));
        assert!(messages[8].contains("its temperature from 0.0 to 423.15"));
        assert!(messages[9].contains("its temperature at `thermal` from none"));
    }

    #[test]
    fn checks_parts() {
        let numbered = |id, part: &str| Instance {
            part: Some(part.to_owned()),
            ..instance_of(id, "unused", &[])
        };
        let top = Component {
            instances: vec![
                numbered("Q1", "2N3904"),
                numbered("Q2", ""),
                numbered("Q3", "2N3904 "),
            ],
            ..Component::default()
        };
        let problems = circuit(vec![("top", top)]).validate().unwrap_err();
        let messages: Vec<String> = problems
            .0
            .iter()
            .filter(|problem| matches!(problem, Problem::InvalidPart { .. }))
            .map(ToString::to_string)
            .collect();
        assert_eq!(
            messages,
            [
                "instance `top/Q2` is the part \"\", but a part number is neither empty nor \
                 spaced at either end",
                "instance `top/Q3` is the part \"2N3904 \", but a part number is neither \
                 empty nor spaced at either end",
            ]
        );
    }

    #[test]
    fn checks_tolerances() {
        let tolerance = |relative| crate::Tolerance {
            relative,
            distribution: crate::Distribution::Uniform,
            lot: None,
        };
        let resistor = Instance {
            name: name("R1"),
            of: Of::Device(name("resistor")),
            parameters: serde_json::json!({"resistance": 1e3, "label": "a", "level": {"dc": 5.0}})
                .as_object()
                .cloned()
                .unwrap(),
            terminals: BTreeMap::new(),
            tolerances: [
                ("resistance".to_owned(), tolerance(1.0)),
                ("label".to_owned(), tolerance(0.1)),
                ("missing".to_owned(), tolerance(0.1)),
                ("level.dc".to_owned(), tolerance(0.05)),
                ("level.ac".to_owned(), tolerance(0.05)),
            ]
            .into(),
            ratings: Vec::new(),
            part: None,
            source_location: None,
        };
        let mut component = instance_of("X1", "cell", &[("a", "n"), ("b", "n")]);
        component.tolerances = [("resistance".to_owned(), tolerance(0.01))].into();
        let top = Component {
            instances: vec![resistor, component],
            ..Component::default()
        };
        let problems = circuit(vec![("top", top), ("cell", cell())])
            .validate()
            .unwrap_err();
        let parameters: Vec<&str> = problems
            .iter()
            .map(|problem| match problem {
                Problem::InvalidTolerance { parameter, .. } => parameter.as_str(),
                other => panic!("{other}"),
            })
            .collect();
        assert_eq!(
            parameters,
            ["label", "level.ac", "missing", "resistance", "resistance"]
        );
    }

    #[test]
    fn a_lot_spreads_each_parameter_one_way() {
        let in_lot = |id: &str, relative| {
            let lot = Lot {
                name: name("die"),
                relative,
                distribution: crate::Distribution::Uniform,
            };
            let tolerance = crate::Tolerance {
                relative: 0.001,
                distribution: crate::Distribution::Uniform,
                lot: Some(lot),
            };
            Instance {
                name: name(id),
                of: Of::Device(name("resistor")),
                parameters: serde_json::json!({"resistance": 1e3})
                    .as_object()
                    .cloned()
                    .unwrap(),
                terminals: BTreeMap::new(),
                tolerances: [("resistance".to_owned(), tolerance)].into(),
                ratings: Vec::new(),
                part: None,
                source_location: None,
            }
        };
        let top = Component {
            instances: vec![
                in_lot("R1", 0.2),
                in_lot("R2", 0.2),
                in_lot("R3", 0.1),
                in_lot("R4", 1.5),
            ],
            ..Component::default()
        };
        let problems = circuit(vec![("top", top)]).validate().unwrap_err();
        let messages: Vec<String> = problems.iter().map(ToString::to_string).collect();
        assert_eq!(
            messages,
            [
                "instance `top/R4` takes `resistance` from lot `die` with a tolerance of 1.5, \
                 which needs to be at least 0, and below 1 unless its logarithm spreads",
                "instances `top/R1` and `top/R3` spread `resistance` of lot `die` differently, \
                 and a lot spreads each parameter one way",
                "instances `top/R1` and `top/R4` spread `resistance` of lot `die` differently, \
                 and a lot spreads each parameter one way",
            ]
        );
    }

    #[test]
    fn a_logarithmic_spread_takes_any_factor() {
        use crate::Distribution;
        assert!(Distribution::LogUniform.takes(35.0));
        assert!(Distribution::LogNormal.takes(0.0));
        assert!(!Distribution::LogUniform.takes(f64::INFINITY));
        assert!(!Distribution::LogNormal.takes(-0.1));
        assert!(Distribution::Uniform.takes(0.0));
        assert!(!Distribution::Uniform.takes(1.0));
        assert!(!Distribution::Normal.takes(-0.1));
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

    #[test]
    fn deduplicates_parallel_component_edges_when_reporting_cycles() {
        let a = Component {
            instances: vec![instance_of("X1", "b", &[]), instance_of("X2", "b", &[])],
            ..Component::default()
        };
        let b = Component {
            instances: vec![instance_of("Y1", "a", &[]), instance_of("Y2", "a", &[])],
            ..Component::default()
        };
        let self_loop = Component {
            instances: vec![
                instance_of("S1", "self_loop", &[]),
                instance_of("S2", "self_loop", &[]),
            ],
            ..Component::default()
        };
        let top = Component {
            instances: vec![
                instance_of("Z1", "a", &[]),
                instance_of("Z2", "self_loop", &[]),
            ],
            ..Component::default()
        };
        let problems = circuit(vec![
            ("top", top),
            ("a", a),
            ("b", b),
            ("self_loop", self_loop),
        ])
        .validate()
        .unwrap_err();
        let messages: Vec<String> = problems.iter().map(ToString::to_string).collect();
        assert_eq!(
            messages,
            [
                "component `a` contains itself: a -> b -> a",
                "component `self_loop` contains itself: self_loop -> self_loop",
            ]
        );
    }

    #[test]
    fn validates_deep_hierarchies_without_stack_overflow() {
        let handle = std::thread::Builder::new()
            .stack_size(2 * 1024 * 1024)
            .spawn(|| {
                let depth = 5_000;
                let mut components = BTreeMap::new();
                for level in 0..depth {
                    let key = name(&format!("c{level:04}"));
                    let instances = if level + 1 < depth {
                        vec![instance_of("X", &format!("c{:04}", level + 1), &[])]
                    } else {
                        Vec::new()
                    };
                    components.insert(
                        key,
                        Component {
                            instances,
                            ..Component::default()
                        },
                    );
                }
                let acyclic = Circuit {
                    top: name("c0000"),
                    components: components.clone(),
                };
                assert_eq!(acyclic.validate(), Ok(()));

                components
                    .get_mut(&name(&format!("c{:04}", depth - 1)))
                    .unwrap()
                    .instances
                    .push(instance_of("Back", "c0000", &[]));
                let cyclic = Circuit {
                    top: name("c0000"),
                    components,
                };
                let problems = cyclic.validate().unwrap_err();
                assert_eq!(problems.len(), 1);
                let Some(Problem::Recursive { cycle }) = problems.iter().next() else {
                    panic!("expected recursive cycle problem");
                };
                assert_eq!(cycle.len(), depth + 1);
                assert_eq!(cycle.first(), Some(&name("c0000")));
                assert_eq!(cycle.last(), Some(&name("c0000")));
            })
            .expect("spawn thread");
        handle
            .join()
            .expect("thread succeeded without stack overflow");
    }
}
