//! Physical invariants on random circuits.
#![cfg(test)]

use std::collections::BTreeMap;

use jolty_engine::{
    ChargeLeak, Conditions, Method, Options, Parallelism, PeriodicSteadyState, Record, Simulator,
    Transient, TransientResult,
};
use jolty_ir::Circuit;
use jolty_models::Library;
use proptest::prelude::*;
use rayon::ThreadPoolBuilder;
use serde_json::{Value, json};

/// A random grounded resistor network driven by current sources.
#[derive(Debug, Clone)]
struct Network {
    /// Resistors as `(node, node, resistance)`, where node 0 is the ground.
    resistors: Vec<(usize, usize, f64)>,
    /// Current sources as `(node, current)`, each pushing its current into its node.
    sources: Vec<(usize, f64)>,
}

fn net(node: usize) -> String {
    if node == 0 {
        "0".to_owned()
    } else {
        format!("n{node}")
    }
}

/// A connected network on up to eight nodes: a spanning tree, then extra resistors.
fn network() -> impl Strategy<Value = Network> {
    (2usize..9).prop_flat_map(|nodes| {
        let tree =
            proptest::collection::vec((any::<prop::sample::Index>(), -3.0..6.0f64), nodes - 1);
        let extra = proptest::collection::vec((0..nodes, 0..nodes, -3.0..6.0f64), 0..nodes);
        let sources = proptest::collection::vec((1..nodes, -1e-2..1e-2f64), 1..4);
        (tree, extra, sources).prop_map(move |(tree, extra, sources)| {
            let mut resistors: Vec<(usize, usize, f64)> = tree
                .into_iter()
                .enumerate()
                .map(|(child, (parent, exponent))| {
                    (child + 1, parent.index(child + 1), 10f64.powf(exponent))
                })
                .collect();
            resistors.extend(
                extra
                    .into_iter()
                    .filter(|(a, b, _)| a != b)
                    .map(|(a, b, exponent)| (a, b, 10f64.powf(exponent))),
            );
            Network { resistors, sources }
        })
    })
}

fn circuit(network: &Network, reversed: bool) -> Circuit {
    let mut instances: Vec<Value> = network
        .resistors
        .iter()
        .enumerate()
        .map(|(index, &(a, b, resistance))| {
            json!({
                "name": format!("R{index}"),
                "of": {"model": "resistor"},
                "parameters": {"resistance": resistance},
                "terminals": {"a": net(a), "b": net(b)},
            })
        })
        .chain(
            network
                .sources
                .iter()
                .enumerate()
                .map(|(index, &(node, current))| {
                    json!({
                        "name": format!("I{index}"),
                        "of": {"model": "current_source"},
                        "parameters": {"waveform": {"dc": current}},
                        "terminals": {"positive": "0", "negative": net(node)},
                    })
                }),
        )
        .collect();
    if reversed {
        instances.reverse();
    }
    let document = json!({
        "format": "jolty-circuit", "version": 4, "top": "network",
        "components": {"network": {
            "ports": [{"name": "0", "role": "ground"}],
            "instances": instances,
        }}
    });
    Circuit::from_json(&document.to_string()).expect("a valid circuit file")
}

impl Network {
    /// The number of nodes, the ground included.
    fn nodes(&self) -> usize {
        1 + self
            .resistors
            .iter()
            .map(|&(a, b, _)| a.max(b))
            .max()
            .unwrap_or(0)
    }

    /// Every node's voltage at the operating point, by node number, the ground being zero.
    fn voltages(&self, reversed: bool) -> Vec<f64> {
        let circuit = circuit(self, reversed);
        let mut simulator = Simulator::new(&circuit, &Library::standard(), &Conditions::default())
            .expect("an elaborated circuit");
        let solution = simulator
            .operating_point(&tight())
            .expect("an operating point");
        (0..self.nodes())
            .map(|node| {
                simulator
                    .unknown(&net(node))
                    .map_or(0.0, |unknown| solution.values[unknown.index()])
            })
            .collect()
    }
}

/// Options far tighter than SPICE's, which every network must still meet.
fn tight() -> Options {
    Options {
        relative_tolerance: 1e-9,
        current_tolerance: 1e-15,
        ..Options::default()
    }
}

proptest! {
    /// The currents into every node sum to zero, within the relative tolerance on the
    /// currents meeting there, or within the rounding of the node voltages they flow
    /// between: one unit in the last place of a voltage of hundreds of volts moves the
    /// current through a milliohm by tens of picoamperes.
    #[test]
    fn currents_balance_at_every_node(network in network()) {
        let options = tight();
        let voltage = network.voltages(false);
        for node in 1..network.nodes() {
            let (mut sum, mut largest, mut resolution) = (0.0, 0.0_f64, 0.0);
            for &(a, b, resistance) in &network.resistors {
                let current = (voltage[a] - voltage[b]) / resistance;
                let leaving = if node == a {
                    current
                } else if node == b {
                    -current
                } else {
                    continue;
                };
                sum += leaving;
                largest = largest.max(leaving.abs());
                resolution += (voltage[a].abs() + voltage[b].abs()) / resistance;
            }
            for &(source, current) in &network.sources {
                if source == node {
                    sum -= current;
                    largest = largest.max(current.abs());
                }
            }
            let tolerance = options.relative_tolerance * largest
                + options.current_tolerance
                + 8.0 * f64::EPSILON * resolution;
            prop_assert!(sum.abs() <= tolerance, "node {node}: {sum} of {largest}");
        }
    }

    /// The solution does not depend on the order instances are placed in.
    #[test]
    fn instance_order_does_not_matter(network in network()) {
        let (forward, backward) = (network.voltages(false), network.voltages(true));
        for node in 1..network.nodes() {
            let (a, b) = (forward[node], backward[node]);
            let tolerance = 1e-9 * a.abs().max(b.abs()) + 1e-12;
            prop_assert!((a - b).abs() <= tolerance, "node {node}: {a} vs {b}");
        }
    }
}

/// A boundary capacitor connecting an island node (`terminal a`) to either `"drive"` or
/// `"0"` (`terminal b`), with optional polynomial voltage dependence.
#[derive(Debug, Clone)]
struct BoundaryCapacitor {
    island_node: usize,
    driven: bool,
    capacitance: f64,
    voltage_coefficient: f64,
    quadratic_voltage_coefficient: f64,
}

impl BoundaryCapacitor {
    /// Analytical charge `C * (v + vc1 * v^2 / 2 + vc2 * v^3 / 3)` on terminal `a`.
    fn charge(&self, voltage: f64) -> f64 {
        let squared = voltage * voltage;
        self.capacitance
            * (voltage
                + 0.5 * self.voltage_coefficient * squared
                + (squared * voltage / 3.0) * self.quadratic_voltage_coefficient)
    }
}

/// A random closed capacitive island: `1..=4` nodes isolated from ground and the driving
/// voltage source by capacitors alone, with resistors and optional diodes strictly inside
/// the island so the sum of static currents over the island nodes is identically zero.
#[derive(Debug, Clone)]
struct ClosedIsland {
    nodes: usize,
    capacitors: Vec<BoundaryCapacitor>,
    resistors: Vec<(usize, usize, f64)>,
    diodes: Vec<(usize, usize)>,
    initial_voltage: f64,
    amplitude: f64,
    pulsed: bool,
    operating_point: bool,
}

fn closed_island() -> impl Strategy<Value = ClosedIsland> {
    (1usize..=4).prop_flat_map(|nodes| {
        let drive_capacitor = (-11.0..-9.0f64, -0.04..0.04f64, 0.0..0.008f64);
        let ground_capacitors =
            proptest::collection::vec((-11.0..-9.0f64, -0.04..0.04f64, 0.0..0.008f64), nodes);
        let tree =
            proptest::collection::vec((any::<prop::sample::Index>(), 1.0..5.0f64), nodes - 1);
        let extra_resistors = proptest::collection::vec(
            (1usize..=nodes, 1usize..=nodes, 1.5..5.0f64),
            0..=nodes.saturating_sub(1),
        );
        let diodes = proptest::collection::vec(
            (1usize..=nodes, 1usize..=nodes),
            0..=usize::from(nodes > 1) * 2,
        );
        (
            drive_capacitor,
            ground_capacitors,
            tree,
            extra_resistors,
            diodes,
            -1.0..1.0f64,
            0.5..3.0f64,
            any::<bool>(),
            any::<bool>(),
        )
            .prop_map(
                move |(
                    drive_capacitor,
                    ground_capacitors,
                    tree,
                    extra_resistors,
                    diodes,
                    initial_voltage,
                    amplitude,
                    pulsed,
                    operating_point,
                )| {
                    let mut capacitors = vec![BoundaryCapacitor {
                        island_node: 1,
                        driven: true,
                        capacitance: 10_f64.powf(drive_capacitor.0),
                        voltage_coefficient: drive_capacitor.1,
                        quadratic_voltage_coefficient: drive_capacitor.2,
                    }];
                    for (index, (exponent, linear, quadratic)) in
                        ground_capacitors.into_iter().enumerate()
                    {
                        capacitors.push(BoundaryCapacitor {
                            island_node: index + 1,
                            driven: false,
                            capacitance: 10_f64.powf(exponent),
                            voltage_coefficient: linear,
                            quadratic_voltage_coefficient: quadratic,
                        });
                    }
                    let mut resistors: Vec<(usize, usize, f64)> = tree
                        .into_iter()
                        .enumerate()
                        .map(|(child, (parent, exponent))| {
                            let from = child + 2;
                            let to = parent.index(child + 1) + 1;
                            (from, to, 10_f64.powf(exponent))
                        })
                        .collect();
                    resistors.extend(
                        extra_resistors
                            .into_iter()
                            .filter(|(a, b, _)| a != b)
                            .map(|(a, b, exponent)| (a, b, 10_f64.powf(exponent))),
                    );
                    let diodes = diodes.into_iter().filter(|(a, b)| a != b).collect();
                    ClosedIsland {
                        nodes,
                        capacitors,
                        resistors,
                        diodes,
                        initial_voltage,
                        amplitude,
                        pulsed,
                        operating_point,
                    }
                },
            )
    })
}

impl ClosedIsland {
    fn circuit(&self) -> Circuit {
        let waveform = if self.pulsed {
            json!({
                "pulse": {
                    "initial": 0.0,
                    "pulsed": self.amplitude,
                    "delay": 5e-7,
                    "rise": 2e-7,
                    "fall": 2e-7,
                    "width": 1e-6,
                    "period": 3e-6,
                }
            })
        } else {
            json!({
                "sine": {
                    "offset": 0.0,
                    "amplitude": self.amplitude,
                    "frequency": 2.5e5,
                }
            })
        };
        let mut instances = vec![json!({
            "name": "V1",
            "of": {"model": "voltage_source"},
            "parameters": {"waveform": waveform},
            "terminals": {"positive": "drive", "negative": "0"},
        })];
        for (index, capacitor) in self.capacitors.iter().enumerate() {
            let other = if capacitor.driven { "drive" } else { "0" };
            instances.push(json!({
                "name": format!("C{index}"),
                "of": {"model": "capacitor"},
                "parameters": {
                    "capacitance": capacitor.capacitance,
                    "voltage_coefficient": capacitor.voltage_coefficient,
                    "quadratic_voltage_coefficient": capacitor.quadratic_voltage_coefficient,
                },
                "terminals": {"a": net(capacitor.island_node), "b": other},
            }));
        }
        for (index, &(a, b, resistance)) in self.resistors.iter().enumerate() {
            instances.push(json!({
                "name": format!("R{index}"),
                "of": {"model": "resistor"},
                "parameters": {"resistance": resistance},
                "terminals": {"a": net(a), "b": net(b)},
            }));
        }
        for (index, &(anode, cathode)) in self.diodes.iter().enumerate() {
            instances.push(json!({
                "name": format!("D{index}"),
                "of": {"model": "diode"},
                "parameters": {"saturation_current": 1e-14, "junction_capacitance": 1e-12},
                "terminals": {"anode": net(anode), "cathode": net(cathode)},
            }));
        }
        let document = json!({
            "format": "jolty-circuit", "version": 4, "top": "island",
            "components": {"island": {
                "ports": [{"name": "0", "role": "ground"}],
                "instances": instances,
            }}
        });
        Circuit::from_json(&document.to_string()).expect("a valid circuit file")
    }

    /// Initial conditions setting the floating island's starting voltage(s).
    fn initial_conditions(&self) -> BTreeMap<String, f64> {
        let mut initial = BTreeMap::new();
        if self.operating_point {
            initial.insert(net(1), self.initial_voltage);
        } else {
            for node in 1..=self.nodes {
                let offset = 0.1 * f64::from(u32::try_from(node - 1).expect("small node"));
                initial.insert(net(node), self.initial_voltage + offset);
            }
        }
        initial
    }
}

/// Raw IEEE-754 bits of every value in `values`.
fn slice_bits(values: &[f64]) -> Vec<u64> {
    values.iter().copied().map(f64::to_bits).collect()
}

/// Verifies structural invariants on `leak` for a system with `unknown_count` unknowns.
fn assert_valid_charge_leak(leak: &ChargeLeak, unknown_count: usize) {
    assert_eq!(leak.signed.len(), unknown_count);
    assert_eq!(leak.absolute.len(), unknown_count);
    for (&signed, &absolute) in leak.signed.iter().zip(&leak.absolute) {
        assert!(signed.is_finite(), "non-finite signed leak {signed}");
        assert!(absolute.is_finite(), "non-finite absolute leak {absolute}");
        assert!(absolute >= 0.0, "negative absolute leak {absolute}");
        assert!(
            signed.abs() <= absolute + f64::EPSILON * absolute,
            "|{signed:e}| > {absolute:e}"
        );
    }
}

/// Locates terminal `a` of every boundary capacitor in `Record::All` observations.
fn boundary_charge_slots(simulator: &Simulator, expected: usize) -> Vec<usize> {
    let recording = simulator.recorded(&Record::All, false);
    let mut offset = 0;
    let mut slots = Vec::with_capacity(expected);
    for &device_index in &recording.devices {
        let device = &simulator.devices()[device_index];
        if device.path().starts_with('C') {
            slots.push(offset);
        }
        offset += device.terminals().len();
    }
    assert_eq!(slots.len(), expected);
    slots
}

/// Collects total island charge at every accepted step, cross-checking recorded terminal
/// charges against the analytical polynomial capacitor formula, and returns
/// `(total_charges, charge_scale, current_scale)`.
fn sample_island_charges(
    island: &ClosedIsland,
    waveforms: &TransientResult,
    drive_index: usize,
    island_indices: &[usize],
    boundary_slots: &[usize],
) -> (Vec<f64>, f64, f64) {
    let mut total_charges = Vec::with_capacity(waveforms.values.len());
    let mut charge_scale = 0.0_f64;
    let mut current_scale = 0.0_f64;
    for (step_index, row) in waveforms.values.iter().enumerate() {
        let view = waveforms.observations.point(step_index);
        let analytical: f64 = island
            .capacitors
            .iter()
            .map(|capacitor| {
                let node_voltage = row[island_indices[capacitor.island_node - 1]];
                let other_voltage = if capacitor.driven {
                    row[drive_index]
                } else {
                    0.0
                };
                capacitor.charge(node_voltage - other_voltage)
            })
            .sum();
        let recorded: f64 = boundary_slots.iter().map(|&slot| view.charges[slot]).sum();
        let step_charge_scale: f64 = view.charges.iter().copied().map(f64::abs).sum();
        let step_current_scale: f64 = view.currents.iter().copied().map(f64::abs).sum();
        assert!(
            (analytical - recorded).abs() <= 64.0 * f64::EPSILON * step_charge_scale.max(1e-30),
            "analytical {analytical:e} vs recorded {recorded:e}"
        );
        charge_scale = charge_scale.max(step_charge_scale);
        current_scale = current_scale.max(step_current_scale);
        total_charges.push(recorded);
    }
    (total_charges, charge_scale, current_scale)
}

/// Checks that on `island` simulated with `method` and `options`:
/// - `charge_leak` vectors match `unknowns.len()`, are finite, satisfy `|signed| <= absolute`,
/// - total island charge change `Q(T) - Q(0)` equals `sum(charge_leak.signed[k])` to rounding,
/// - at every accepted time `t_m`, `|Q(t_m) - Q(0)| <= sum(charge_leak.absolute[k]) + rounding`.
///
/// Returns `(actual_change, signed_leak, absolute_leak, rounding)`.
fn check_closed_island(
    island: &ClosedIsland,
    method: Method,
    options: &Options,
    stop: f64,
    max_step: f64,
) -> (f64, f64, f64, f64) {
    let built = island.circuit();
    let mut simulator = Simulator::new(&built, &Library::standard(), &Conditions::default())
        .expect("an elaborated circuit");
    let transient = Transient {
        stop,
        max_step: Some(max_step),
        method,
        breakpoints: Vec::new(),
        initial_conditions: island.initial_conditions(),
        operating_point: island.operating_point,
    };
    let waveforms = simulator
        .transient(&transient, options)
        .expect("transient finishes");
    let leak = &waveforms.statistics.charge_leak;
    assert_valid_charge_leak(leak, simulator.unknowns().len());

    let drive_index = simulator.unknown("drive").expect("drive node").index();
    let island_indices: Vec<usize> = (1..=island.nodes)
        .map(|node| simulator.unknown(&net(node)).expect("island node").index())
        .collect();
    let slots = boundary_charge_slots(&simulator, island.capacitors.len());
    let (total_charges, charge_scale, current_scale) =
        sample_island_charges(island, &waveforms, drive_index, &island_indices, &slots);

    let signed_leak: f64 = island_indices.iter().map(|&index| leak.signed[index]).sum();
    let absolute_leak: f64 = island_indices
        .iter()
        .map(|&index| leak.absolute[index])
        .sum();
    let accepted = f64::from(u32::try_from(waveforms.statistics.accepted).expect("step count"));
    // Without manual observation breakpoints, every Gear2 step ratio satisfies omega <= 2.0
    // (FACTOR_BOUNDS.1), bounding `(|beta_0| + |beta_1|) / alpha <= 13/5 < 3.0`.
    let rounding =
        128.0 * f64::EPSILON * (3.0 * accepted * charge_scale + stop * current_scale).max(1e-30);

    let initial_charge = total_charges[0];
    let final_charge = *total_charges.last().expect("at least one point");
    let actual_change = final_charge - initial_charge;
    let difference = (actual_change - signed_leak).abs();
    assert!(
        difference <= rounding,
        "{method:?}: actual charge change {actual_change:e} vs signed leak {signed_leak:e} \
         (diff {difference:e}, rounding bound {rounding:e})"
    );
    for (step_index, &charge) in total_charges.iter().enumerate() {
        let excursion = (charge - initial_charge).abs();
        assert!(
            excursion <= absolute_leak + rounding,
            "{method:?} step {step_index}: excursion {excursion:e} exceeds absolute leak \
             {absolute_leak:e} + rounding {rounding:e}"
        );
    }
    (actual_change, signed_leak, absolute_leak, rounding)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// On any random closed capacitive island (linear or nonlinear capacitors, internal
    /// resistors and diodes, pulsed corners or sine drive, operating-point or UIC start),
    /// total island charge stays within the reported solver charge leak under backward Euler,
    /// trapezoidal, and Gear2, and matches the signed integral to floating-point rounding.
    #[test]
    fn closed_capacitive_island_charge_matches_integrated_residual(
        island in closed_island(),
    ) {
        let options = Options {
            record: Record::All,
            ..Options::default()
        };
        for method in [Method::BackwardEuler, Method::Trapezoidal, Method::Gear2] {
            check_closed_island(&island, method, &options, 4e-6, 2e-7);
        }
    }
}

#[test]
fn closed_nonlinear_island_leak_is_measurable_and_exact_at_loose_tolerance() {
    let island = ClosedIsland {
        nodes: 2,
        capacitors: vec![
            BoundaryCapacitor {
                island_node: 1,
                driven: true,
                capacitance: 1e-9,
                voltage_coefficient: 0.04,
                quadratic_voltage_coefficient: 0.008,
            },
            BoundaryCapacitor {
                island_node: 1,
                driven: false,
                capacitance: 1e-9,
                voltage_coefficient: -0.02,
                quadratic_voltage_coefficient: 0.004,
            },
            BoundaryCapacitor {
                island_node: 2,
                driven: false,
                capacitance: 1e-9,
                voltage_coefficient: 0.0,
                quadratic_voltage_coefficient: 0.0,
            },
        ],
        // High parallel resistance lets the diode's exponential nonlinearity dominate Newton
        // iteration and leave a genuine solver residual far above IEEE-754 rounding.
        resistors: vec![(1, 2, 1e5)],
        diodes: vec![(1, 2)],
        initial_voltage: 0.0,
        amplitude: 5.0,
        pulsed: true,
        operating_point: true,
    };
    let loose = Options {
        relative_tolerance: 2e-2,
        voltage_tolerance: 5e-3,
        current_tolerance: 1e-6,
        record: Record::All,
        ..Options::default()
    };
    let tighter = Options {
        relative_tolerance: 1e-5,
        voltage_tolerance: 1e-8,
        current_tolerance: 1e-14,
        record: Record::All,
        ..Options::default()
    };
    for method in [Method::BackwardEuler, Method::Trapezoidal, Method::Gear2] {
        let (actual_loose, signed_loose, absolute_loose, rounding_loose) =
            check_closed_island(&island, method, &loose, 4e-6, 1e-7);
        let (_, _, absolute_tight, _) = check_closed_island(&island, method, &tighter, 4e-6, 1e-7);
        // At loose tolerances, the solver's nonlinear residual produces a measurable leak
        // far above IEEE-754 rounding (so a zero stub would fail).
        assert!(
            absolute_loose > 100.0 * rounding_loose,
            "{method:?}: expected measurable leak, got {absolute_loose:e} vs {rounding_loose:e}"
        );
        assert!(
            (actual_loose - signed_loose).abs() <= rounding_loose,
            "{method:?}: {actual_loose:e} vs {signed_loose:e}"
        );
        assert!(
            absolute_tight < absolute_loose,
            "{method:?}: tighter tolerance should reduce leak \
             ({absolute_tight:e} vs {absolute_loose:e})"
        );
    }
}

#[test]
fn closed_island_with_sub_ohm_internal_resistor_conserves_charge() {
    // Resistors below 1 Ohm spawn an explicit current branch unknown (`R0:current`); its
    // branch currents cancel across the island nodes `n1` and `n2`.
    let island = ClosedIsland {
        nodes: 2,
        capacitors: vec![
            BoundaryCapacitor {
                island_node: 1,
                driven: true,
                capacitance: 2e-9,
                voltage_coefficient: 0.03,
                quadratic_voltage_coefficient: 0.005,
            },
            BoundaryCapacitor {
                island_node: 1,
                driven: false,
                capacitance: 1e-9,
                voltage_coefficient: 0.0,
                quadratic_voltage_coefficient: 0.0,
            },
            BoundaryCapacitor {
                island_node: 2,
                driven: false,
                capacitance: 1e-9,
                voltage_coefficient: 0.0,
                quadratic_voltage_coefficient: 0.0,
            },
        ],
        resistors: vec![(1, 2, 0.25)],
        diodes: vec![(1, 2)],
        initial_voltage: 0.5,
        amplitude: 2.0,
        pulsed: true,
        operating_point: true,
    };
    let options = Options {
        record: Record::All,
        ..Options::default()
    };
    for method in [Method::BackwardEuler, Method::Trapezoidal, Method::Gear2] {
        check_closed_island(&island, method, &options, 4e-6, 2e-7);
    }
}

#[test]
fn charge_leak_is_bit_identical_across_thread_counts() {
    let island = ClosedIsland {
        nodes: 3,
        capacitors: vec![
            BoundaryCapacitor {
                island_node: 1,
                driven: true,
                capacitance: 1e-9,
                voltage_coefficient: 0.03,
                quadratic_voltage_coefficient: 0.005,
            },
            BoundaryCapacitor {
                island_node: 1,
                driven: false,
                capacitance: 1e-9,
                voltage_coefficient: 0.0,
                quadratic_voltage_coefficient: 0.0,
            },
            BoundaryCapacitor {
                island_node: 2,
                driven: false,
                capacitance: 2e-9,
                voltage_coefficient: 0.0,
                quadratic_voltage_coefficient: 0.0,
            },
            BoundaryCapacitor {
                island_node: 3,
                driven: false,
                capacitance: 5e-10,
                voltage_coefficient: 0.0,
                quadratic_voltage_coefficient: 0.0,
            },
        ],
        resistors: vec![(1, 2, 1e3), (2, 3, 2e3)],
        diodes: vec![(1, 3)],
        initial_voltage: 0.0,
        amplitude: 3.0,
        pulsed: true,
        operating_point: true,
    };
    let built = island.circuit();
    let transient = Transient {
        stop: 4e-6,
        max_step: Some(1e-7),
        method: Method::Trapezoidal,
        breakpoints: Vec::new(),
        initial_conditions: island.initial_conditions(),
        operating_point: true,
    };
    let run = |threads: usize| {
        let options = Options {
            parallelism: Parallelism {
                min_instances_per_task: Some(1),
                min_unknowns_per_task: Some(1),
                ..Parallelism::default()
            },
            ..Options::default()
        };
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a pool");
        pool.install(|| {
            let (library, conditions) = (Library::standard(), Conditions::default());
            let mut simulator =
                Simulator::new(&built, &library, &conditions).expect("an elaborated circuit");
            let waveforms = simulator
                .transient(&transient, &options)
                .expect("transient finishes");
            let leak = &waveforms.statistics.charge_leak;
            (slice_bits(&leak.signed), slice_bits(&leak.absolute))
        })
    };
    let reference = run(1);
    for threads in [2, 4] {
        assert_eq!(run(threads), reference, "at {threads} threads");
    }
}

#[test]
fn pure_logic_transient_has_empty_charge_leak() {
    let document = json!({
        "format": "jolty-circuit", "version": 4, "top": "logic",
        "components": {"logic": {
            "ports": [{"name": "0", "role": "ground"}],
            "instances": [{
                "name": "A",
                "of": {"model": "logic_source"},
                "parameters": {
                    "initial": "0",
                    "changes": [[1e-9, "1"]],
                    "transition": 1e-12,
                },
                "terminals": {"output": "a"},
            }],
        }}
    });
    let built = Circuit::from_json(&document.to_string()).expect("a valid circuit");
    let mut simulator =
        Simulator::new(&built, &Library::standard(), &Conditions::default()).expect("elaborates");
    let transient = Transient {
        stop: 5e-9,
        max_step: None,
        method: Method::Trapezoidal,
        breakpoints: Vec::new(),
        initial_conditions: BTreeMap::new(),
        operating_point: true,
    };
    let waveforms = simulator
        .transient(&transient, &Options::default())
        .expect("transient finishes");
    assert_eq!(simulator.unknowns(), []);
    assert_eq!(waveforms.statistics.charge_leak.signed, [] as [f64; 0]);
    assert_eq!(waveforms.statistics.charge_leak.absolute, [] as [f64; 0]);
}

#[test]
fn periodic_steady_state_carries_period_charge_leak() {
    let document = json!({
        "format": "jolty-circuit", "version": 4, "top": "rectifier",
        "components": {"rectifier": {
            "ports": [{"name": "0", "role": "ground"}],
            "instances": [
                {
                    "name": "V1",
                    "of": {"model": "voltage_source"},
                    "parameters": {
                        "waveform": {"sine": {"offset": 0.0, "amplitude": 2.0, "frequency": 1e5}}
                    },
                    "terminals": {"positive": "in", "negative": "0"},
                },
                {
                    "name": "D1",
                    "of": {"model": "diode"},
                    "parameters": {"saturation_current": 1e-14, "junction_capacitance": 1e-12},
                    "terminals": {"anode": "in", "cathode": "out"},
                },
                {
                    "name": "R1",
                    "of": {"model": "resistor"},
                    "parameters": {"resistance": 1e3},
                    "terminals": {"a": "out", "b": "0"},
                },
                {
                    "name": "C1",
                    "of": {"model": "capacitor"},
                    "parameters": {"capacitance": 1e-9},
                    "terminals": {"a": "out", "b": "0"},
                },
            ],
        }}
    });
    let built = Circuit::from_json(&document.to_string()).expect("a valid circuit");
    let mut simulator =
        Simulator::new(&built, &Library::standard(), &Conditions::default()).expect("elaborates");
    let steady_state = PeriodicSteadyState {
        period: Some(1e-5),
        max_step: Some(2e-7),
        method: Method::Trapezoidal,
        breakpoints: Vec::new(),
    };
    let found = simulator
        .periodic_steady_state(&steady_state, &Options::default())
        .expect("periodic steady state finishes");
    let leak = &found.waveforms.statistics.charge_leak;
    assert_valid_charge_leak(leak, simulator.unknowns().len());
}
