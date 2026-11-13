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
        "instance `{component}/{id}` states a rating of its {stress} {at}up to {maximum}, \
         which needs {needed}, a finite maximum, a positive one for a power, and no \
         minimum above it"
    )]
    InvalidRating {
        /// The component that places the instance.
        component: Name,
        /// The instance.
        id: Name,
        /// The stress rated: `voltage`, `current` or `power`.
        stress: String,
        /// Where, as the terminals named, such as `at `collector`, `emitter` `.
        at: String,
        /// The maximum, in its shortest exact form.
        maximum: String,
        /// How many terminals the stress needs, in words.
        needed: &'static str,
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
                let at = format!("{name}/{}", instance.id);
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
        let bounded = rating.maximum.is_finite()
            && (rating.stress != Stress::Power || rating.maximum > 0.0)
            && rating
                .minimum
                .is_none_or(|minimum| minimum.is_finite() && minimum <= rating.maximum);
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
                id: instance.id.clone(),
                stress: format!("{:?}", rating.stress).to_lowercase(),
                at,
                maximum: format!("{:?}", rating.maximum),
                needed: [
                    "no terminals",
                    "one of its terminals",
                    "two distinct terminals of its own",
                ][needed],
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
        if !ids.insert(&instance.id) {
            problems.push(Problem::DuplicateInstance {
                component: name.clone(),
                id: instance.id.clone(),
            });
        }
        for (parameter, tolerance) in &instance.tolerances {
            let numeric = crate::parameter(&instance.parameters, parameter)
                .is_some_and(serde_json::Value::is_number);
            if !numeric || !tolerance.distribution.takes(tolerance.relative) {
                problems.push(Problem::InvalidTolerance {
                    component: name.clone(),
                    id: instance.id.clone(),
                    parameter: parameter.clone(),
                    relative: format!("{:?}", tolerance.relative),
                });
            }
            if let Some(lot) = &tolerance.lot
                && !lot.distribution.takes(lot.relative)
            {
                problems.push(Problem::InvalidLot {
                    component: name.clone(),
                    id: instance.id.clone(),
                    parameter: parameter.clone(),
                    lot: lot.name.clone(),
                    relative: format!("{:?}", lot.relative),
                });
            }
        }
        check_ratings(name, instance, problems);
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
            tolerances: BTreeMap::new(),
            ratings: Vec::new(),
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
    fn checks_ratings() {
        let rating = |stress, terminals: &[&str], maximum, minimum| crate::Rating {
            stress,
            terminals: terminals.iter().map(|terminal| name(terminal)).collect(),
            maximum,
            minimum,
        };
        let mut transistor = instance_of("Q1", "unused", &[("collector", "c"), ("base", "b")]);
        transistor.of = Of::Device(name("bipolar_transistor"));
        transistor.ratings = vec![
            rating(Stress::Voltage, &["collector", "base"], 40.0, None),
            rating(Stress::Current, &["collector"], 0.2, Some(-0.2)),
            rating(Stress::Power, &[], 0.625, None),
            rating(Stress::Voltage, &["collector"], 40.0, None),
            rating(Stress::Voltage, &["collector", "emitter"], 40.0, None),
            rating(Stress::Voltage, &["base", "base"], 6.0, None),
            rating(Stress::Power, &[], 0.0, None),
            rating(Stress::Current, &["base"], 0.1, Some(0.2)),
            rating(Stress::Current, &["base"], f64::INFINITY, None),
        ];
        let top = Component {
            instances: vec![transistor],
            ..Component::default()
        };
        let problems = circuit(vec![("top", top)]).validate().unwrap_err();
        let messages: Vec<String> = problems.0.iter().map(ToString::to_string).collect();
        assert_eq!(messages.len(), 6, "{messages:?}");
        assert_eq!(
            messages[0],
            "instance `top/Q1` states a rating of its voltage at `collector` up to 40.0, which \
             needs two distinct terminals of its own, a finite maximum, a positive one for a \
             power, and no minimum above it"
        );
        assert!(messages[1].contains("its voltage at `collector` over `emitter` up to 40.0"));
        assert!(messages[3].contains("its power up to 0.0, which needs no terminals"));
    }

    #[test]
    fn checks_tolerances() {
        let tolerance = |relative| crate::Tolerance {
            relative,
            distribution: crate::Distribution::Uniform,
            lot: None,
        };
        let resistor = Instance {
            id: name("R1"),
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
            source: None,
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
                id: name(id),
                of: Of::Device(name("resistor")),
                parameters: serde_json::json!({"resistance": 1e3})
                    .as_object()
                    .cloned()
                    .unwrap(),
                terminals: BTreeMap::new(),
                tolerances: [("resistance".to_owned(), tolerance)].into(),
                ratings: Vec::new(),
                source: None,
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
}
