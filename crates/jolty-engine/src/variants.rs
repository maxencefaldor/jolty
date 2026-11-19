//! Parameter variant generators for [batch](crate::batch) execution, computing each run's
//! values as a pure function of run index.
//!
//! Supported generators are [`Explicit`] lists, mixed-radix [`Grid`] sweeps (last axis
//! varying fastest), [`MonteCarlo`] tolerance sampling, and Cartesian [`Product`]
//! combinations for mixing corners, sweeps, and samples in one batch.
//!
//! Each Monte Carlo draw depends solely on seed, sample index, and a stable FNV-1a hash of
//! `(instance_path, parameter_path)` or `(lot_name, parameter_path)`. Sample `i` stays
//! identical whether drawn standalone or inside any parallel batch, and adding a part never
//! perturbs draws for unrelated parts.

use std::collections::HashSet;

use jolty_ir::{Circuit, Component, Distribution, Of, Tolerance};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_distr::StandardNormal;
use serde::Deserialize;

use crate::batch::Variants;
use crate::elaborate::{ElaborationError, ElaborationProblem};
use crate::{Elaborated, ParameterId, Values};

/// Explicit list of per-run [`Values`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Explicit(pub Vec<Values>);

impl Variants for Explicit {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn values(&self, index: u64) -> Values {
        usize::try_from(index)
            .ok()
            .and_then(|index| self.0.get(index))
            .cloned()
            .unwrap_or_default()
    }
}

/// Quantity stepped along one axis of a [`Grid`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// Device numeric parameter identified by [`ParameterId`].
    Parameter(ParameterId),
    /// Device operating temperature in kelvins.
    Temperature,
}

/// Execution mode along the fastest-varying axis of a [`Grid`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridChain {
    /// Independent cold-started solve at every grid point.
    Independent,
    /// Sequential warm-started chain along each line of the last axis.
    Warm,
}

/// Cartesian grid over parameter and temperature axes in mixed-radix order (last axis
/// varying fastest).
///
/// Under [`GridChain::Warm`], runs along each line of the last axis execute as a
/// warm-started chain.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    axes: Vec<(Axis, Vec<f64>)>,
    chain: GridChain,
    len: u64,
}

impl Grid {
    /// Creates a [`Grid`] from ordered `(axis, values)` pairs; an empty axis list yields a
    /// single nominal point, and `chain` selects whether runs chain along the last axis.
    ///
    /// # Errors
    ///
    /// Returns [`VariantsError::Empty`] for an empty axis or [`VariantsError::TooMany`]
    /// when total points overflow `u64`.
    pub fn new(axes: Vec<(Axis, Vec<f64>)>, chain: GridChain) -> Result<Self, VariantsError> {
        let mut len: u64 = 1;
        for (index, (_, values)) in axes.iter().enumerate() {
            if values.is_empty() {
                return Err(VariantsError::Empty(index));
            }
            len = len
                .checked_mul(values.len() as u64)
                .ok_or(VariantsError::TooMany)?;
        }
        Ok(Self { axes, chain, len })
    }
}

impl Variants for Grid {
    fn len(&self) -> u64 {
        self.len
    }

    fn values(&self, index: u64) -> Values {
        let mut values = Values::default();
        let mut rest = index;
        for (axis, steps) in self.axes.iter().rev() {
            let count = steps.len() as u64;
            // The remainder is below the axis's length, a `usize`.
            let step = usize::try_from(rest % count).unwrap_or_default();
            rest /= count;
            match *axis {
                Axis::Parameter(parameter) => values.set(parameter, steps[step]),
                Axis::Temperature => values.temperature = Some(steps[step]),
            }
        }
        values
    }

    fn chain(&self) -> u64 {
        match self.axes.last() {
            Some((_, steps)) if self.chain == GridChain::Warm => steps.len() as u64,
            _ => 1,
        }
    }
}

/// Cartesian product combining every variant of `A` with every variant of `B`.
///
/// Run `i` applies variant `i / len(B)` of `A` followed by variant `i % len(B)` of `B`,
/// with `B` overriding `A` on shared keys and setting the chain length.
#[derive(Debug, Clone, PartialEq)]
pub struct Product<A, B> {
    first: A,
    second: B,
    len: u64,
}

impl<A: Variants, B: Variants> Product<A, B> {
    /// Creates the Cartesian [`Product`] of `first` and `second`.
    ///
    /// # Errors
    ///
    /// Returns [`VariantsError::TooMany`] when the product overflows `u64`.
    pub fn new(first: A, second: B) -> Result<Self, VariantsError> {
        let len = first
            .len()
            .checked_mul(second.len())
            .ok_or(VariantsError::TooMany)?;
        Ok(Self { first, second, len })
    }

    /// Returns references to the `(first, second)` operand variant sets.
    #[must_use]
    pub const fn parts(&self) -> (&A, &B) {
        (&self.first, &self.second)
    }
}

impl<A: Variants, B: Variants> Variants for Product<A, B> {
    fn len(&self) -> u64 {
        self.len
    }

    fn values(&self, index: u64) -> Values {
        let count = self.second.len().max(1);
        let mut values = self.first.values(index / count);
        values.extend(&self.second.values(index % count));
        values
    }

    fn chain(&self) -> u64 {
        self.second.chain()
    }
}

/// Error returned when constructing invalid [`Grid`] or [`Product`] variants.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VariantsError {
    /// Grid axis at the given index contains no values.
    #[error("axis {0} of the grid has no values")]
    Empty(usize),
    /// Total number of variants overflows `u64`.
    #[error("the variants are more than 2⁶⁴ - 1")]
    TooMany,
}

/// Device numeric parameter carrying a statistical tolerance specification.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Toleranced {
    /// Hierarchical device instance path, such as `X1/R1`.
    pub instance: String,
    /// Parameter path within the device, such as `resistance` or `waveform.dc`.
    pub parameter: String,
    /// Nominal numeric value declared in the circuit.
    pub nominal: f64,
    /// Statistical distribution and spread parameters for this value.
    pub tolerance: Tolerance,
}

/// Domain separation tag for per-part draws, shared lot draws, and process-kit statistical
/// variates.
#[derive(Debug, Clone, Copy)]
#[repr(u64)]
enum Drawn {
    Part = 0,
    Lot = 1,
    Kit = 2,
}

/// Standard random variate scaled by process-kit statistical functions (`agauss`/`gauss`
/// for normal; `aunif`/`unif`/`limit` for uniform).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variate {
    /// Standard normal variate with mean 0 and standard deviation 1.
    Normal,
    /// Uniform variate on `[-1, 1)`.
    Uniform,
}

/// Draws a standard `variate` for sample `index` from `seed` at `call` within `scope`
/// (instance path, or empty for kit-level definitions).
///
/// Uses a `ChaCha20` RNG keyed by `(seed, index, fnv1a(scope, call), Drawn::Kit)` so each
/// draw stays invariant across batch sizes, thread counts, and unrelated circuit edits,
/// matching `NumPy` uniform and standard-normal transformations.
#[must_use]
pub fn variate(variate: Variate, seed: u64, index: u64, scope: &str, call: &str) -> f64 {
    let mut generator = generator(seed, index, key(scope, call), Drawn::Kit);
    match variate {
        Variate::Normal => normal(&mut generator, 0.0, 1.0),
        Variate::Uniform => uniform(&mut generator, -1.0, 1.0),
    }
}

impl Toleranced {
    /// Draws the realized parameter value for Monte Carlo sample `index` from `seed`,
    /// multiplying `nominal` by the per-part factor and optional shared lot factor.
    ///
    /// Each factor uses a `ChaCha20` RNG keyed by `(seed, index, fnv1a_key, domain_tag)`
    /// where part keys hash `(instance, parameter)` and lot keys hash `(lot, parameter)`.
    /// Linear tolerances scale by `1 + d` (uniform over `±relative` or normal with
    /// `3σ = relative`), and logarithmic tolerances scale by `(1 + relative)^u` (uniform
    /// over `[-1, 1)` or normal with `3σ = 1`).
    #[must_use]
    pub fn drawn(&self, seed: u64, index: u64) -> f64 {
        let tolerance = &self.tolerance;
        let own = generator(
            seed,
            index,
            key(&self.instance, &self.parameter),
            Drawn::Part,
        );
        let mut factor = factor(tolerance.distribution, tolerance.relative, own);
        if let Some(lot) = &tolerance.lot {
            let shared = generator(
                seed,
                index,
                key(lot.name.as_str(), &self.parameter),
                Drawn::Lot,
            );
            factor *= self::factor(lot.distribution, lot.relative, shared);
        }
        self.nominal * factor
    }
}

/// First-order statistical spread of a tolerance factor drawn by [`Toleranced::drawn`]:
/// standard deviation of the zero-mean variate and factor derivative at zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spread {
    /// Standard deviation of the underlying zero-mean random variable.
    pub standard_deviation: f64,
    /// Derivative of the multiplicative factor with respect to the random variable at 0.
    pub slope: f64,
}

impl Spread {
    /// Computes the first-order [`Spread`] for `distribution` and `relative` tolerance.
    ///
    /// Linear distributions (`1 + d`) have slope `1` with `σ = relative / √3` (uniform) or
    /// `relative / 3` (normal). Logarithmic distributions (`(1 + relative)^u`) have slope
    /// `ln(1 + relative)` with `σ = 1 / √3` (uniform) or `1 / 3` (normal).
    #[must_use]
    pub fn from_tolerance(distribution: Distribution, relative: f64) -> Self {
        let uniform = 3.0_f64.sqrt().recip();
        let normal = 3.0_f64.recip();
        match distribution {
            Distribution::Uniform => Self {
                standard_deviation: relative * uniform,
                slope: 1.0,
            },
            Distribution::Normal => Self {
                standard_deviation: relative * normal,
                slope: 1.0,
            },
            Distribution::LogUniform => Self {
                standard_deviation: uniform,
                slope: relative.ln_1p(),
            },
            Distribution::LogNormal => Self {
                standard_deviation: normal,
                slope: relative.ln_1p(),
            },
        }
    }
}

impl Toleranced {
    /// Returns the per-part [`Spread`] and optional `(lot_name, lot_spread)` computed via
    /// [`Spread::from_tolerance`].
    #[must_use]
    pub fn random_variables(&self) -> (Spread, Option<(&str, Spread)>) {
        let tolerance = &self.tolerance;
        let own = Spread::from_tolerance(tolerance.distribution, tolerance.relative);
        let lot = tolerance.lot.as_ref().map(|lot| {
            (
                lot.name.as_str(),
                Spread::from_tolerance(lot.distribution, lot.relative),
            )
        });
        (own, lot)
    }
}

/// Hashes `(first, second)` with 64-bit FNV-1a separated by byte `0xFF` (invalid in UTF-8
/// so distinct pairs never collide by concatenation).
fn key(first: &str, second: &str) -> u64 {
    fnv(first.bytes().chain([0xff]).chain(second.bytes()))
}

/// Computes the 64-bit FNV-1a hash of `bytes` (Fowler, Noll, and Vo; IETF draft
/// `draft-eastlake-fnv`).
fn fnv(bytes: impl IntoIterator<Item = u8>) -> u64 {
    /// FNV-1a 64-bit offset basis (`14695981039346656037`; Fowler, Noll and Vo, IETF draft
    /// `draft-eastlake-fnv`).
    const BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    /// FNV-1a 64-bit prime (`2^40 + 2^8 + 0xb3 = 1099511628211`; Fowler, Noll and Vo, IETF
    /// draft `draft-eastlake-fnv`).
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    bytes.into_iter().fold(BASIS, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(PRIME)
    })
}

/// Seeds a `ChaCha20Rng` from little-endian `(seed, index, key, drawn)` words.
fn generator(seed: u64, index: u64, key: u64, drawn: Drawn) -> ChaCha20Rng {
    let mut words = [0u8; 32];
    for (place, word) in [seed, index, key, drawn as u64].into_iter().enumerate() {
        words[8 * place..8 * place + 8].copy_from_slice(&word.to_le_bytes());
    }
    ChaCha20Rng::from_seed(words)
}

/// Draws a multiplicative tolerance factor from `generator` for `distribution` and
/// `relative` spread.
fn factor(distribution: Distribution, relative: f64, mut generator: ChaCha20Rng) -> f64 {
    let generator = &mut generator;
    match distribution {
        Distribution::Uniform => 1.0 + uniform(generator, -relative, relative),
        Distribution::Normal => 1.0 + normal(generator, 0.0, relative / 3.0),
        Distribution::LogUniform => (1.0 + relative).powf(uniform(generator, -1.0, 1.0)),
        Distribution::LogNormal => (1.0 + relative).powf(normal(generator, 0.0, 1.0 / 3.0)),
    }
}

/// Draws a uniform variate on `[low, high)` matching `NumPy`'s `uniform` formula.
fn uniform(generator: &mut ChaCha20Rng, low: f64, high: f64) -> f64 {
    let unit: f64 = generator.random();
    low + (high - low) * unit
}

/// Draws a normal variate with `mean` and `deviation` matching `NumPy`'s `normal` formula.
fn normal(generator: &mut ChaCha20Rng, mean: f64, deviation: f64) -> f64 {
    let standard: f64 = generator.sample(StandardNormal);
    mean + deviation * standard
}

/// Collects all toleranced numeric device parameters in `circuit` in deterministic
/// hierarchical and parameter-path order.
///
/// # Errors
///
/// Returns [`TolerancedError::Untoleranced`] when a tolerance targets a non-numeric
/// parameter, or [`TolerancedError::Recursive`] when component placements form a cycle.
pub fn toleranced(circuit: &Circuit) -> Result<Vec<Toleranced>, TolerancedError> {
    let mut found = Vec::new();
    walk(circuit, &mut found)?;
    Ok(found)
}

/// Failure encountered while collecting toleranced parameters from a circuit hierarchy.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TolerancedError {
    /// Tolerance targets a parameter without a numeric value.
    #[error("{instance} gives {parameter} a tolerance, not a number")]
    Untoleranced {
        /// Hierarchical device instance path.
        instance: String,
        /// Parameter path within the device.
        parameter: String,
    },
    /// Component hierarchy contains a placement cycle.
    #[error("component `{component}` contains itself: {}", .cycle.join(" -> "))]
    Recursive {
        /// Component name closing the cycle.
        component: String,
        /// Sequence of component names along the cycle, starting and ending with `component`.
        cycle: Vec<String>,
    },
}

/// Active traversal frame in [`walk`].
struct Frame<'a> {
    placed: &'a Component,
    prefix: String,
    next_instance: usize,
}

/// Appends toleranced parameters from all reachable instances of `circuit` to `found` via
/// iterative depth-first traversal with cycle detection.
fn walk(circuit: &Circuit, found: &mut Vec<Toleranced>) -> Result<(), TolerancedError> {
    let top = circuit.top.as_str();
    let Some(root) = circuit.components.get(top) else {
        return Ok(());
    };
    let mut active = vec![top];
    let mut active_set = HashSet::from([top]);
    let mut stack = vec![Frame {
        placed: root,
        prefix: String::new(),
        next_instance: 0,
    }];
    while let Some(frame) = stack.last_mut() {
        let Some(instance) = frame.placed.instances.get(frame.next_instance) else {
            stack.pop();
            if let Some(popped) = active.pop() {
                active_set.remove(popped);
            }
            continue;
        };
        frame.next_instance += 1;
        let path = format!("{}{}", frame.prefix, instance.name.as_str());
        match &instance.of {
            Of::Component(inner) => {
                let child = inner.as_str();
                if active_set.contains(child)
                    && let Some(start) = active.iter().position(|&seen| seen == child)
                {
                    let cycle = active[start..]
                        .iter()
                        .copied()
                        .chain(std::iter::once(child))
                        .map(str::to_owned)
                        .collect();
                    return Err(TolerancedError::Recursive {
                        component: child.to_owned(),
                        cycle,
                    });
                }
                if let Some(definition) = circuit.components.get(child) {
                    active.push(child);
                    active_set.insert(child);
                    stack.push(Frame {
                        placed: definition,
                        prefix: format!("{path}/"),
                        next_instance: 0,
                    });
                }
            }
            Of::Device(_) => {
                for (name, tolerance) in &instance.tolerances {
                    let nominal = jolty_ir::parameter(&instance.parameters, name)
                        .and_then(serde_json::Value::as_f64)
                        .ok_or_else(|| TolerancedError::Untoleranced {
                            instance: path.clone(),
                            parameter: name.clone(),
                        })?;
                    found.push(Toleranced {
                        instance: path.clone(),
                        parameter: name.clone(),
                        nominal,
                        tolerance: tolerance.clone(),
                    });
                }
            }
        }
    }
    Ok(())
}

/// Monte Carlo variant generator drawing each toleranced parameter via
/// [drawn](Toleranced::drawn) from a fixed seed.
#[derive(Debug, Clone, PartialEq)]
pub struct MonteCarlo {
    seed: u64,
    samples: u64,
    toleranced: Vec<(ParameterId, Toleranced)>,
}

impl MonteCarlo {
    /// Creates a [`MonteCarlo`] generator producing `samples` draws of `toleranced`
    /// parameters on `elaborated` from `seed`.
    ///
    /// # Errors
    ///
    /// Returns [`ElaborationProblem::UnknownParameter`] for unrecognized parameter paths
    /// and [`ElaborationProblem::UnknownOverride`] for nonexistent instances.
    pub fn new(
        elaborated: &Elaborated,
        seed: u64,
        samples: u64,
        toleranced: Vec<Toleranced>,
    ) -> Result<Self, ElaborationError> {
        let mut problems = Vec::new();
        let mut found = Vec::with_capacity(toleranced.len());
        for item in toleranced {
            match elaborated.parameter(&item.instance, &item.parameter) {
                Some(parameter) => found.push((parameter, item)),
                None => problems.push(match elaborated.location(&item.instance) {
                    Some(at) => ElaborationProblem::UnknownParameter {
                        at,
                        parameter: item.parameter,
                    },
                    None => ElaborationProblem::UnknownOverride(item.instance),
                }),
            }
        }
        if problems.is_empty() {
            Ok(Self {
                seed,
                samples,
                toleranced: found,
            })
        } else {
            Err(ElaborationError::new(problems))
        }
    }
}

impl Variants for MonteCarlo {
    fn len(&self) -> u64 {
        self.samples
    }

    fn values(&self, index: u64) -> Values {
        let mut values = Values::default();
        for (parameter, toleranced) in &self.toleranced {
            values.set(*parameter, toleranced.drawn(self.seed, index));
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_kits_variates_are_their_parts_draws_apart() {
        let drawn = variate(Variate::Normal, 7, 3, "X1", "vth0#0");
        assert_eq!(drawn, variate(Variate::Normal, 7, 3, "X1", "vth0#0"));
        assert_ne!(drawn, variate(Variate::Normal, 7, 3, "X2", "vth0#0"));
        assert_ne!(drawn, variate(Variate::Normal, 7, 4, "X1", "vth0#0"));
        let mut part = generator(7, 3, key("X1", "vth0#0"), Drawn::Part);
        assert_ne!(drawn, normal(&mut part, 0.0, 1.0));
        let uniform = variate(Variate::Uniform, 7, 3, "X1", "vth0#0");
        assert!((-1.0..1.0).contains(&uniform));
    }

    #[test]
    fn keys_are_fnv_1a_of_the_names_joined() {
        // The test vectors of the FNV draft for 64-bit FNV-1a.
        assert_eq!(fnv(*b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv(*b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv(*b"foobar"), 0x8594_4171_f739_67e8);
        assert_eq!(key("X1/R1", "resistance"), fnv(*b"X1/R1\xffresistance"));
        assert_ne!(key("X1/R1", "resistance"), key("X1", "R1/resistance"));
    }

    #[test]
    fn a_grid_steps_its_last_axis_fastest() {
        let (first, second) = (ParameterId::new(0), ParameterId::new(3));
        let grid = Grid::new(
            vec![
                (Axis::Parameter(first), vec![1.0, 2.0]),
                (Axis::Temperature, vec![300.0, 310.0, 320.0]),
                (Axis::Parameter(second), vec![5.0]),
            ],
            GridChain::Warm,
        )
        .unwrap();
        assert_eq!((grid.len(), grid.chain()), (6, 1));
        let values = grid.values(4);
        assert_eq!(values.patches(), [(first, 2.0), (second, 5.0)]);
        assert_eq!(values.temperature, Some(310.0));
        assert_eq!(
            Grid::new(vec![(Axis::Temperature, vec![])], GridChain::Independent),
            Err(VariantsError::Empty(0))
        );
    }

    fn circuit_from_components(top: &str, components: &serde_json::Value) -> Circuit {
        Circuit::from_json(
            &serde_json::json!({
                "format": "jolty-circuit",
                "version": jolty_ir::VERSION,
                "top": top,
                "components": components,
            })
            .to_string(),
        )
        .expect("valid JSON")
    }

    #[test]
    fn toleranced_rejects_self_and_mutual_cycles() {
        let self_cycle = circuit_from_components(
            "top",
            &serde_json::json!({
                "top": {
                    "instances": [
                        {"name": "X1", "of": {"component": "stage"}, "terminals": {}}
                    ]
                },
                "stage": {
                    "instances": [
                        {"name": "Xloop", "of": {"component": "stage"}, "terminals": {}}
                    ]
                }
            }),
        );
        let error = toleranced(&self_cycle).expect_err("self-cycle must fail");
        assert_eq!(
            error,
            TolerancedError::Recursive {
                component: "stage".to_owned(),
                cycle: vec!["stage".to_owned(), "stage".to_owned()],
            }
        );
        assert_eq!(
            error.to_string(),
            "component `stage` contains itself: stage -> stage"
        );

        let mutual = circuit_from_components(
            "top",
            &serde_json::json!({
                "top": {
                    "instances": [
                        {"name": "X1", "of": {"component": "stage"}, "terminals": {}}
                    ]
                },
                "stage": {
                    "instances": [
                        {"name": "Xback", "of": {"component": "top"}, "terminals": {}}
                    ]
                }
            }),
        );
        let error = toleranced(&mutual).expect_err("mutual cycle must fail");
        assert_eq!(
            error.to_string(),
            "component `top` contains itself: top -> stage -> top"
        );
    }

    #[test]
    fn toleranced_walks_diamonds_and_deep_chains() {
        // Diamond DAG: a shared subcircuit placed twice emits both instance paths.
        let diamond = circuit_from_components(
            "top",
            &serde_json::json!({
                "top": {
                    "instances": [
                        {"name": "X1", "of": {"component": "leaf"}, "terminals": {}},
                        {"name": "X2", "of": {"component": "leaf"}, "terminals": {}}
                    ]
                },
                "leaf": {
                    "instances": [
                        {
                            "name": "R1",
                            "of": {"model": "resistor"},
                            "parameters": {"resistance": 1000.0},
                            "terminals": {"a": "a", "b": "b"},
                            "tolerances": {
                                "resistance": {"relative": 0.01, "distribution": "uniform"}
                            }
                        }
                    ]
                }
            }),
        );
        let items = toleranced(&diamond).expect("diamond DAG must succeed");
        let paths: Vec<&str> = items.iter().map(|item| item.instance.as_str()).collect();
        assert_eq!(paths, vec!["X1/R1", "X2/R1"]);

        // Deep acyclic hierarchy (1500 levels) must not overflow the thread stack.
        let mut components = serde_json::Map::new();
        for level in 0..1500 {
            let instances = if level + 1 < 1500 {
                serde_json::json!([{
                    "name": "X",
                    "of": {"component": format!("c{}", level + 1)},
                    "terminals": {}
                }])
            } else {
                serde_json::json!([{
                    "name": "R1",
                    "of": {"model": "resistor"},
                    "parameters": {"resistance": 50.0},
                    "terminals": {"a": "a", "b": "b"},
                    "tolerances": {
                        "resistance": {"relative": 0.05, "distribution": "uniform"}
                    }
                }])
            };
            components.insert(
                format!("c{level}"),
                serde_json::json!({"instances": instances}),
            );
        }
        let deep = circuit_from_components("c0", &serde_json::Value::Object(components));
        let items = toleranced(&deep).expect("deep chain must succeed");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].nominal.to_bits(), 50.0_f64.to_bits());
    }
}
