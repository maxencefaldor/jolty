//! The twins a derivative evaluates: each device's instance prepared again, computing with
//! the derivatives of every number along some directions of its parameters, its seeds, so
//! that one evaluation gives `∂F/∂p` along all of them.
//!
//! A derivative is with respect to columns, each a direction of the parameters on one
//! device. For a VJP, and for the sensitivity and mismatch analyses and the objectives
//! built on it, each column is one parameter, its unit; for a JVP, each is one tangent's
//! components on one device's parameters, so that a tangent costs a device one seed
//! however many of its parameters it moves. A device's columns share its twins, up to
//! [`Seeds::MOST`] to a twin, each prepared once and evaluated once for all its seeds;
//! each seed's numbers are those of a twin of that seed alone, bit for bit.

use std::collections::BTreeMap;

use jolty_models::{Input, Seeds, Twin};
use jolty_numerics::{Complex64, UNMAPPED};

use crate::Simulator;
use crate::elaborated::Origin;
use crate::options::Parallelism;
use crate::sensitivity::Parameter;
use crate::state::Held;
use crate::sweep::Swept;

/// What a column differentiates along: a device, by its index, and each number of its
/// parameters, by its name, with its rate.
pub(crate) type Direction = (usize, Vec<(String, f64)>);

/// Which of a twin's derivatives a quantity reads.
#[derive(Clone, Copy)]
pub(crate) enum Part {
    /// `∂f/∂p`.
    Residual,
    /// `∂q/∂p`.
    Charge,
}

/// One device's twin, and the columns its seeds are, in their order.
struct Chunk {
    device: usize,
    twin: Twin,
    columns: Vec<usize>,
}

/// The twins of a derivative's columns.
pub(crate) struct Twins {
    /// The parameters the derivative is with respect to, in their order, each at its value.
    pub(crate) parameters: Vec<Parameter>,
    chunks: Vec<Chunk>,
    /// Each column's direction.
    directions: Vec<Direction>,
    /// Each column's chunk, and its seed there.
    places: Vec<(usize, usize)>,
    /// Whether a sweep holds each column's numbers, which then moves nothing: its device's
    /// level, or every number of the column.
    fixed: Vec<bool>,
}

impl Twins {
    /// The twins `chunks` of `directions`, each column's twin and seed at `places`, the
    /// derivative being with respect to `parameters`; the columns `unread`, which no twin
    /// carries, left out.
    fn new(
        parameters: Vec<Parameter>,
        (directions, mut chunks, places): (Vec<Direction>, Vec<Chunk>, Vec<(usize, usize)>),
        unread: &[usize],
    ) -> Self {
        let mut renumbered = vec![usize::MAX; directions.len()];
        let mut kept_directions = Vec::with_capacity(directions.len() - unread.len());
        let mut kept_places = Vec::with_capacity(kept_directions.capacity());
        for (column, (direction, place)) in directions.into_iter().zip(places).enumerate() {
            if unread.binary_search(&column).is_ok() {
                continue;
            }
            renumbered[column] = kept_directions.len();
            kept_directions.push(direction);
            kept_places.push(place);
        }
        for chunk in &mut chunks {
            for column in &mut chunk.columns {
                *column = renumbered[*column];
            }
        }
        Self {
            parameters,
            chunks,
            fixed: vec![false; kept_directions.len()],
            directions: kept_directions,
            places: kept_places,
        }
    }

    /// How many columns there are.
    pub(crate) const fn len(&self) -> usize {
        self.directions.len()
    }

    /// Evaluates each twin where its device is: at the solution `x`, at `time` in a
    /// transient analysis or, with `None`, at an operating point or about one.
    pub(crate) fn evaluate(
        &mut self,
        x: &[f64],
        time: Option<f64>,
        gmin: f64,
        parallelism: &Parallelism,
    ) {
        parallelism.spread(
            &mut self.chunks,
            || (),
            |(), chunk| {
                chunk.twin.evaluate(&unlimited(x, time, gmin));
            },
        );
    }

    /// `sign · aᵀ ∂f/∂p` or `sign · aᵀ ∂q/∂p` along each column for each `a` of `vectors`,
    /// by column then vector, as the twins were last evaluated: each twin's own terms
    /// alone, zero for a column a sweep holds.
    pub(crate) fn dots(
        &mut self,
        (part, sign): (Part, f64),
        vectors: &[Vec<f64>],
        parallelism: &Parallelism,
    ) -> Vec<Vec<f64>> {
        let fixed = &self.fixed;
        let by_chunk = parallelism.spread(
            &mut self.chunks,
            || (),
            |(), chunk| {
                chunk
                    .columns
                    .iter()
                    .enumerate()
                    .filter(|&(_, &column)| !fixed[column])
                    .map(|(seed, &column)| {
                        let dots = vectors
                            .iter()
                            .map(|vector| sign * dot(&chunk.twin, part, seed, vector))
                            .collect();
                        (column, dots)
                    })
                    .collect::<Vec<_>>()
            },
        );
        let mut found = vec![vec![0.0; vectors.len()]; self.len()];
        for (column, dots) in by_chunk.into_iter().flatten() {
            found[column] = dots;
        }
        found
    }

    /// `Σ t_j ∂f/∂p_j` or `Σ t_j ∂q/∂p_j` along `direction`, a number per column, by
    /// unknown, among `dimension`, as the twins were last evaluated; a column a sweep holds
    /// moves nothing.
    pub(crate) fn combined(&self, direction: &[f64], dimension: usize, part: Part) -> Vec<f64> {
        let mut values = vec![0.0; dimension];
        for (column, &along) in direction.iter().enumerate() {
            if along == 0.0 || self.fixed[column] {
                continue;
            }
            let (chunk, seed) = self.places[column];
            let twin = &self.chunks[chunk].twin;
            for (unknown, (_, derivatives)) in moving(twin, part) {
                values[unknown] += along * derivatives[seed];
            }
        }
        values
    }

    /// `aᵀ (∂G/∂p + jω ∂C/∂p) x` along `column` at `(ω, x)`, for the vector `a` of the
    /// equations, as last evaluated: its twin's own entries alone, each at its row and
    /// column of `positions`.
    pub(crate) fn product(
        &self,
        column: usize,
        positions: &[(usize, usize)],
        a: &[Complex64],
        (omega, x): (f64, &[Complex64]),
    ) -> Complex64 {
        let (chunk, seed) = self.places[column];
        self.chunks[chunk]
            .twin
            .entries()
            .map(|(slot, (_, conductance), (_, capacitance))| {
                let (row, column) = positions[slot];
                let derivative = Complex64::new(conductance[seed], omega * capacitance[seed]);
                a[row] * derivative * x[column]
            })
            .sum()
    }

    /// Adds `scale (∂G/∂p + jω ∂C/∂p) x` along `column` at `(ω, x)` to `into`, as last
    /// evaluated.
    pub(crate) fn add_product(
        &self,
        column: usize,
        positions: &[(usize, usize)],
        scale: f64,
        (omega, x): (f64, &[Complex64]),
        into: &mut [Complex64],
    ) {
        let (chunk, seed) = self.places[column];
        for (slot, (_, conductance), (_, capacitance)) in self.chunks[chunk].twin.entries() {
            let (row, column) = positions[slot];
            let derivative = Complex64::new(conductance[seed], omega * capacitance[seed]);
            into[row] += derivative * x[column] * scale;
        }
    }

    /// Holds the columns of the device a sweep holds, `swept`, as it now is: every one, a
    /// source held at a level, whose waveform the level replaces; or of a number held, the
    /// columns that move it alone, its device's twins made again from what its instance is
    /// made of now, without it.
    pub(crate) fn hold(&mut self, simulator: &Simulator, swept: &Swept, parameter: Option<&str>) {
        let held = match swept.held {
            Held::Level(_) => None,
            Held::Number(_) => parameter,
        };
        for column in 0..self.len() {
            let (device, numbers) = &self.directions[column];
            if *device == swept.device {
                self.fixed[column] =
                    held.is_none_or(|held| numbers.iter().all(|(name, _)| name == held));
            }
        }
        let Some(held) = held else {
            return;
        };
        for chunk in self
            .chunks
            .iter_mut()
            .filter(|chunk| chunk.device == swept.device)
        {
            if chunk.columns.iter().all(|&column| self.fixed[column]) {
                continue;
            }
            let columns: Vec<&Direction> = chunk
                .columns
                .iter()
                .map(|&column| &self.directions[column])
                .collect();
            if let Some(twin) = seeds(&columns, Some(held))
                .and_then(|seeds| simulator.twin_of(swept.device, &seeds))
            {
                chunk.twin = twin;
            }
        }
    }
}

/// The input that evaluates a twin at `x`, at `time`, with `gmin`, its steps unlimited.
fn unlimited(x: &[f64], time: Option<f64>, gmin: f64) -> Input<'_> {
    let mut input = Input::new(x, time, gmin);
    input.limiting = false;
    input
}

/// The residuals or charges, `part`, `twin` adds, as last evaluated.
fn moving(
    twin: &Twin,
    part: Part,
) -> Box<dyn Iterator<Item = (usize, jolty_models::Moving<'_>)> + '_> {
    match part {
        Part::Residual => Box::new(twin.residual()),
        Part::Charge => Box::new(twin.charge()),
    }
}

/// `aᵀ ∂f/∂p` or `aᵀ ∂q/∂p` along the seed `seed` of `twin`, as last evaluated.
fn dot(twin: &Twin, part: Part, seed: usize, a: &[f64]) -> f64 {
    moving(twin, part)
        .map(|(unknown, (_, derivatives))| a[unknown] * derivatives[seed])
        .sum()
}

/// The seeds of `columns`, each direction one seed, every number they move with its rate
/// along each, a number named twice moving at the sum of its rates; the number `held`
/// left out.
fn seeds(columns: &[&Direction], held: Option<&str>) -> Option<Seeds> {
    let mut rates: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut order = Vec::new();
    for (seed, (_, numbers)) in columns.iter().enumerate() {
        for (name, rate) in numbers {
            if Some(name.as_str()) == held {
                continue;
            }
            let row = rates.entry(name).or_insert_with(|| {
                order.push(name.as_str());
                vec![0.0; columns.len()]
            });
            row[seed] += rate;
        }
    }
    let numbers = order
        .into_iter()
        .map(|name| (name.to_owned(), rates.remove(name).unwrap_or_default()))
        .collect();
    Seeds::new(numbers, columns.len())
}

impl Simulator {
    /// The twins of `parameters`, each a device's index and one of its numbers, a column
    /// each, the unit along it; and the parameters no model reads as a real number, as it
    /// reads a choice, such as a level, or out of range, which they leave out.
    pub(crate) fn unit_twins(
        &self,
        parameters: Vec<(usize, Parameter)>,
    ) -> (Twins, Vec<Parameter>) {
        let directions: Vec<Direction> = parameters
            .iter()
            .map(|(device, parameter)| (*device, vec![(parameter.name.clone(), 1.0)]))
            .collect();
        let (chunks, places, unread) = self.chunked(&directions);
        let mut kept = Vec::with_capacity(parameters.len());
        let mut left_out = Vec::new();
        for (column, (_, parameter)) in parameters.into_iter().enumerate() {
            if unread.binary_search(&column).is_ok() {
                left_out.push(parameter);
            } else {
                kept.push(parameter);
            }
        }
        let twins = Twins::new(kept, (directions, chunks, places), &unread);
        (twins, left_out)
    }

    /// The twins of `tangents`, each a number per parameter of `parameters`, each a
    /// device's index and one of its numbers: a column for each tangent and device, the
    /// tangent's components on that device's numbers, in the order of the tangents then
    /// of the devices; and the direction of each tangent among the columns. Fails with
    /// the parameters no model reads as a real number.
    pub(crate) fn tangent_twins(
        &self,
        parameters: Vec<(usize, Parameter)>,
        tangents: &[Vec<f64>],
    ) -> Result<(Twins, Vec<Vec<f64>>), Vec<Parameter>> {
        if tangents.is_empty() {
            let (twins, unread) = self.unit_twins(parameters);
            if !unread.is_empty() {
                return Err(unread);
            }
            return Ok((
                Twins::new(twins.parameters, (Vec::new(), Vec::new(), Vec::new()), &[]),
                Vec::new(),
            ));
        }
        let mut devices: Vec<usize> = Vec::new();
        for (device, _) in &parameters {
            if !devices.contains(device) {
                devices.push(*device);
            }
        }
        let mut directions = Vec::with_capacity(tangents.len() * devices.len());
        let mut along = vec![vec![0.0; tangents.len() * devices.len()]; tangents.len()];
        for (index, tangent) in tangents.iter().enumerate() {
            for &device in &devices {
                let numbers = parameters
                    .iter()
                    .zip(tangent)
                    .filter(|((each, _), _)| *each == device)
                    .map(|((_, parameter), &rate)| (parameter.name.clone(), rate))
                    .collect();
                along[index][directions.len()] = 1.0;
                directions.push((device, numbers));
            }
        }
        let (chunks, places, unread) = self.chunked(&directions);
        if !unread.is_empty() {
            return Err(parameters
                .into_iter()
                .filter(|(device, parameter)| {
                    let unit = (*device, vec![(parameter.name.clone(), 1.0)]);
                    seeds(&[&unit], None)
                        .and_then(|seeds| self.twin_of(*device, &seeds))
                        .is_none()
                })
                .map(|(_, parameter)| parameter)
                .collect());
        }
        let named = parameters
            .into_iter()
            .map(|(_, parameter)| parameter)
            .collect();
        Ok((Twins::new(named, (directions, chunks, places), &[]), along))
    }

    /// Each device's twins of `directions`, the columns, up to [`Seeds::MOST`] to a twin,
    /// each column's twin and seed there, and the columns along a number no model reads
    /// as a real one, which no twin carries.
    fn chunked(&self, directions: &[Direction]) -> (Vec<Chunk>, Vec<(usize, usize)>, Vec<usize>) {
        let mut by_device: Vec<(usize, Vec<usize>)> = Vec::new();
        for (column, (device, _)) in directions.iter().enumerate() {
            match by_device.iter_mut().find(|(each, _)| each == device) {
                Some((_, columns)) => columns.push(column),
                None => by_device.push((*device, vec![column])),
            }
        }
        let mut chunks = Vec::new();
        let mut places = vec![(usize::MAX, 0); directions.len()];
        let mut unread = Vec::new();
        let twin_of = |device: usize, columns: &[usize]| {
            let along: Vec<&Direction> =
                columns.iter().map(|&column| &directions[column]).collect();
            seeds(&along, None).and_then(|seeds| self.twin_of(device, &seeds))
        };
        for (device, columns) in by_device {
            for members in columns.chunks(Seeds::MOST) {
                let mut members = members.to_vec();
                let mut twin = twin_of(device, &members);
                if twin.is_none() {
                    // A column moves a number no model reads: each is found alone, and the
                    // rest seeded together.
                    members.retain(|&column| {
                        let read = twin_of(device, &[column]).is_some();
                        if !read {
                            unread.push(column);
                        }
                        read
                    });
                    twin = (!members.is_empty())
                        .then(|| twin_of(device, &members))
                        .flatten();
                }
                let Some(twin) = twin else {
                    unread.extend(members);
                    continue;
                };
                for (seed, &column) in members.iter().enumerate() {
                    places[column] = (chunks.len(), seed);
                }
                chunks.push(Chunk {
                    device,
                    twin,
                    columns: members,
                });
            }
        }
        unread.sort_unstable();
        (chunks, places, unread)
    }

    /// The twin of the device at `index` along `seeds`, made from what the device is made
    /// from now; none when its model reads no real number at a seed's path.
    pub(crate) fn twin_of(&self, index: usize, seeds: &Seeds) -> Option<Twin> {
        let origins = self.origins();
        let origin = origins.of(index);
        if !matches!(origin, Origin::Model(_)) {
            return None;
        }
        let instance = (&origin.kernel(), origins.temperature);
        self.state
            .runs
            .twin(&self.elaborated.groups, index, instance, seeds)
    }

    /// The indices of the devices whose kernels are not linear, in their order.
    pub(crate) fn nonlinear_devices(&self) -> Vec<usize> {
        (0..self.elaborated.devices.len())
            .filter(|&device| !self.elaborated.groups.is_linear(device))
            .collect()
    }

    /// The unknown-seeded twins of every nonlinear device in `nonlinear`, and the sorted,
    /// deduplicated matrix slots those twins write to.
    pub(crate) fn unknown_twins(&self, nonlinear: &[usize]) -> (Vec<Twin>, Vec<usize>) {
        let origins = self.origins();
        let twins: Vec<Twin> = nonlinear
            .iter()
            .flat_map(|&device| {
                let origin = origins.of(device);
                let instance = (&origin.kernel(), origins.temperature);
                self.state
                    .runs
                    .unknown_twins(&self.elaborated.groups, device, instance)
            })
            .collect();
        let mut slots: Vec<usize> = twins.iter().flat_map(Twin::slots).collect();
        slots.sort_unstable();
        slots.dedup();
        (twins, slots)
    }

    /// How `G` and `C` move, by `(stored position, ∂G/∂x · s, ∂C/∂x · s)` in increasing
    /// slot order, as the unknowns at `x` move along each vector of `shifts`, summed over
    /// the nonlinear devices in `nonlinear` in their order.
    pub(crate) fn entries_along(
        &self,
        nonlinear: &[usize],
        x: &[f64],
        shifts: &[Vec<f64>],
        gmin: f64,
        parallelism: &Parallelism,
    ) -> Vec<Vec<(usize, f64, f64)>> {
        let mut moved: Vec<Vec<(usize, f64, f64)>> = vec![Vec::new(); shifts.len()];
        if shifts.is_empty() || nonlinear.is_empty() {
            return moved;
        }
        let origins = self.origins();
        let mut tasks = Vec::new();
        for &device in nonlinear {
            let origin = origins.of(device);
            let instance = (&origin.kernel(), origins.temperature);
            for (chunk_index, chunk) in shifts.chunks(Seeds::MOST).enumerate() {
                if let Some(seeds) = Seeds::tangents(chunk.len())
                    && let Some(twin) =
                        self.state
                            .runs
                            .twin(&self.elaborated.groups, device, instance, &seeds)
                {
                    tasks.push((chunk_index * Seeds::MOST, chunk, twin));
                }
            }
        }
        let input = unlimited(x, None, gmin);
        parallelism.spread(
            &mut tasks,
            || (),
            |(), (_, chunk, twin)| {
                twin.evaluate_along(&input, chunk);
            },
        );
        for (first, _, twin) in &tasks {
            for (slot, (_, conductances), (_, capacitances)) in twin.entries() {
                for (offset, (&conductance_rate, &capacitance_rate)) in
                    conductances.iter().zip(capacitances).enumerate()
                {
                    if conductance_rate != 0.0 || capacitance_rate != 0.0 {
                        moved[first + offset].push((slot, conductance_rate, capacitance_rate));
                    }
                }
            }
        }
        for entries in &mut moved {
            // Stable sort preserves device summation order for entries sharing one slot.
            entries.sort_by_key(|&(slot, _, _)| slot);
            let mut write = 0;
            for read in 0..entries.len() {
                let (slot, conductance_rate, capacitance_rate) = entries[read];
                if write > 0 && entries[write - 1].0 == slot {
                    entries[write - 1].1 += conductance_rate;
                    entries[write - 1].2 += capacitance_rate;
                } else {
                    entries[write] = (slot, conductance_rate, capacitance_rate);
                    write += 1;
                }
            }
            entries.truncate(write);
            entries.retain(|&(_, conductance_rate, capacitance_rate)| {
                conductance_rate != 0.0 || capacitance_rate != 0.0
            });
        }
        moved
    }

    /// For each active `(W_G, W_C)` of `weights`, indexed by `slots`, the gradient at `x`
    /// with respect to the unknowns of `Σ_s (W_G[s] G[s](x) + W_C[s] C[s](x))`, summed
    /// over `twins` in their order.
    pub(crate) fn entries_cotangents(
        &self,
        twins: &mut [Twin],
        slots: &[usize],
        x: &[f64],
        weights: &[Option<(Vec<Complex64>, Vec<Complex64>)>],
        gmin: f64,
        parallelism: &Parallelism,
    ) -> Vec<Option<Vec<Complex64>>> {
        let dimension = self.elaborated.unknowns.len();
        let mut gradients: Vec<Option<Vec<Complex64>>> = weights
            .iter()
            .map(|weight| {
                weight
                    .as_ref()
                    .map(|_| vec![Complex64::default(); dimension])
            })
            .collect();
        if slots.is_empty() || gradients.iter().all(Option::is_none) {
            return gradients;
        }
        let input = unlimited(x, None, gmin);
        let by_twin = parallelism.spread(
            twins,
            || (),
            |(), twin| {
                twin.evaluate(&input);
                let seeded = twin.seeded();
                let mut local = vec![vec![Complex64::default(); seeded.len()]; weights.len()];
                for (slot, (_, conductance), (_, capacitance)) in twin.entries() {
                    let Ok(packed) = slots.binary_search(&slot) else {
                        continue;
                    };
                    for (cotangent, weight) in weights.iter().enumerate() {
                        let Some((weight_conductance, weight_capacitance)) = weight else {
                            continue;
                        };
                        let (by_conductance, by_capacitance) =
                            (weight_conductance[packed], weight_capacitance[packed]);
                        if by_conductance == Complex64::default()
                            && by_capacitance == Complex64::default()
                        {
                            continue;
                        }
                        for (sum, (&conductance_rate, &capacitance_rate)) in local[cotangent]
                            .iter_mut()
                            .zip(conductance.iter().zip(capacitance))
                        {
                            *sum += by_conductance * conductance_rate
                                + by_capacitance * capacitance_rate;
                        }
                    }
                }
                let mut twin_terms = vec![Vec::new(); weights.len()];
                for (cotangent, sums) in local.into_iter().enumerate() {
                    for (&unknown, sum) in seeded.iter().zip(sums) {
                        if unknown != UNMAPPED && sum != Complex64::default() {
                            twin_terms[cotangent].push((unknown as usize, sum));
                        }
                    }
                }
                twin_terms
            },
        );
        for twin_terms in by_twin {
            for (gradient, terms) in gradients.iter_mut().zip(twin_terms) {
                if let Some(gradient) = gradient {
                    for (unknown, term) in terms {
                        gradient[unknown] += term;
                    }
                }
            }
        }
        gradients
    }
}
