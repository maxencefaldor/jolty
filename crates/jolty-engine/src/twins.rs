//! Dual-number device twins for exact parameter and unknown directional differentiation.
//!
//! Each derivative column represents a parameter perturbation direction on one device. VJP,
//! sensitivity, mismatch, and objective analyses use unit parameter columns, whereas JVP
//! packs all parameter rates for one tangent on a device into a single seed. Columns on the
//! same device share batched twins of up to [`Seeds::MAX`] seeds, producing bit-identical
//! derivatives to single-seed evaluation.

use std::collections::BTreeMap;

use jolty_models::{Input, Seeds, Twin};
use jolty_numerics::{Complex64, UNMAPPED};

use crate::Simulator;
use crate::elaborated::Origin;
use crate::options::Parallelism;
use crate::sensitivity::Parameter;
use crate::state::Held;
use crate::sweep::Swept;

/// Parameter perturbation direction on a single device.
struct Direction {
    /// Target device index.
    device: usize,
    /// Parameter path and rate pairs moved along this direction.
    rates: Vec<(String, f64)>,
}

/// Selector between residual and charge partial derivatives from a twin.
#[derive(Clone, Copy)]
pub(crate) enum Part {
    /// Residual partial derivative `∂f/∂p`.
    Residual,
    /// Charge partial derivative `∂q/∂p`.
    Charge,
}

/// Batched dual-number twin for one device together with the column indices mapped to its
/// seeds.
struct Chunk {
    device: usize,
    twin: Twin,
    columns: Vec<usize>,
}

/// Location of a derivative column inside `Twins`: chunk index and seed slot.
#[derive(Clone, Copy)]
struct Place {
    chunk: usize,
    seed: usize,
}

/// Chunked twin collection along with column locations and sorted indices of unread
/// non-real parameter columns.
struct Chunked {
    chunks: Vec<Chunk>,
    places: Vec<Place>,
    unread: Vec<usize>,
}

/// Batched dual-number twins covering all active derivative columns.
pub(crate) struct Twins {
    /// Differentiated parameters in output order, each carrying its nominal value.
    pub(crate) parameters: Vec<Parameter>,
    chunks: Vec<Chunk>,
    /// Perturbation direction for each kept column.
    directions: Vec<Direction>,
    /// Chunk and seed location for each kept column.
    places: Vec<Place>,
    /// Flags marking columns held fixed by an active DC sweep (yielding zero derivative).
    fixed: Vec<bool>,
}

impl Twins {
    /// Builds [`Twins`] from `parameters`, `directions`, and `chunked`, discarding columns
    /// omitted as unread.
    fn new(parameters: Vec<Parameter>, directions: Vec<Direction>, chunked: Chunked) -> Self {
        let Chunked {
            mut chunks,
            places,
            unread,
        } = chunked;
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

    /// Returns the number of active derivative columns.
    pub(crate) const fn len(&self) -> usize {
        self.directions.len()
    }

    /// Evaluates every twin at solution `x`, optional transient `time`, and `gmin` across
    /// `parallelism`.
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

    /// Computes `sign * a^T ∂f/∂p` or `sign * a^T ∂q/∂p` (selected by `part`) for every
    /// column and adjoint vector in `vectors`, returning zero for sweep-held columns.
    pub(crate) fn dots(
        &mut self,
        part: Part,
        sign: f64,
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

    /// Computes the weighted sum `Σ direction[j] ∂(part)/∂p_j` across unknowns of length
    /// `dimension`, skipping sweep-held columns.
    pub(crate) fn combined(&self, direction: &[f64], dimension: usize, part: Part) -> Vec<f64> {
        let mut values = vec![0.0; dimension];
        for (column, &along) in direction.iter().enumerate() {
            if along == 0.0 || self.fixed[column] {
                continue;
            }
            let Place { chunk, seed } = self.places[column];
            let twin = &self.chunks[chunk].twin;
            for (unknown, (_, derivatives)) in contributions(twin, part) {
                values[unknown] += along * derivatives[seed];
            }
        }
        values
    }

    /// Evaluates `a^T (∂G/∂p + jω ∂C/∂p) x` for `column` at angular frequency `omega` and
    /// phasors `x` using sparse slot `positions`.
    pub(crate) fn product(
        &self,
        column: usize,
        positions: &[(usize, usize)],
        a: &[Complex64],
        omega: f64,
        x: &[Complex64],
    ) -> Complex64 {
        let Place { chunk, seed } = self.places[column];
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

    /// Accumulates `scale * (∂G/∂p + jω ∂C/∂p) x` for `column` at angular frequency
    /// `omega` and phasors `x` into `into`.
    pub(crate) fn add_product(
        &self,
        column: usize,
        positions: &[(usize, usize)],
        scale: f64,
        omega: f64,
        x: &[Complex64],
        into: &mut [Complex64],
    ) {
        let Place { chunk, seed } = self.places[column];
        for (slot, (_, conductance), (_, capacitance)) in self.chunks[chunk].twin.entries() {
            let (row, column) = positions[slot];
            let derivative = Complex64::new(conductance[seed], omega * capacitance[seed]);
            into[row] += derivative * x[column] * scale;
        }
    }

    /// Freezes columns overridden by a DC sweep on `swept` and rebuilds remaining twins on
    /// that device excluding the held parameter.
    pub(crate) fn hold(&mut self, simulator: &Simulator, swept: &Swept, parameter: Option<&str>) {
        let held = match swept.held {
            Held::Value(_) => None,
            Held::Number(_) => parameter,
        };
        for column in 0..self.len() {
            let direction = &self.directions[column];
            if direction.device == swept.device {
                self.fixed[column] =
                    held.is_none_or(|held| direction.rates.iter().all(|(name, _)| name == held));
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

/// Builds an evaluation [`Input`] at `x`, `time`, and `gmin` with voltage limiting
/// disabled.
fn unlimited(x: &[f64], time: Option<f64>, gmin: f64) -> Input<'_> {
    let mut input = Input::new(x, time, gmin);
    input.limiting = false;
    input
}

/// Unboxed iterator unifying [`Twin`] residual and charge contribution streams.
enum ContributionIterator<R, C> {
    /// Residual contributions.
    Residual(R),
    /// Charge contributions.
    Charge(C),
}

impl<R, C, T> Iterator for ContributionIterator<R, C>
where
    R: Iterator<Item = T>,
    C: Iterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        match self {
            Self::Residual(iterator) => iterator.next(),
            Self::Charge(iterator) => iterator.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Residual(iterator) => iterator.size_hint(),
            Self::Charge(iterator) => iterator.size_hint(),
        }
    }
}

/// Returns an iterator over residual or charge dual contributions from `twin` per `part`.
fn contributions(
    twin: &Twin,
    part: Part,
) -> impl Iterator<Item = (usize, jolty_models::TwinNumber<'_>)> + '_ {
    match part {
        Part::Residual => ContributionIterator::Residual(twin.residual()),
        Part::Charge => ContributionIterator::Charge(twin.charge()),
    }
}

/// Computes the dot product `a^T ∂(part)/∂p` along `seed` of `twin`.
fn dot(twin: &Twin, part: Part, seed: usize, a: &[f64]) -> f64 {
    contributions(twin, part)
        .map(|(unknown, (_, derivatives))| a[unknown] * derivatives[seed])
        .sum()
}

/// Builds [`Seeds`] assigning one seed per column in `columns` while excluding `held`;
/// [`Seeds::new`] sums duplicate parameter rates.
fn seeds(columns: &[&Direction], held: Option<&str>) -> Option<Seeds> {
    let count = columns.len();
    let numbers = columns
        .iter()
        .enumerate()
        .flat_map(|(seed, direction)| {
            direction
                .rates
                .iter()
                .filter(move |(name, _)| Some(name.as_str()) != held)
                .map(move |(name, rate)| {
                    let mut rates = vec![0.0; count];
                    rates[seed] = *rate;
                    (name.clone(), rates)
                })
        })
        .collect();
    Seeds::new(numbers, count)
}

/// Builds unit perturbation directions for each `(device, parameter)` entry in `parameters`.
fn units(parameters: &[(usize, Parameter)]) -> Vec<Direction> {
    parameters
        .iter()
        .map(|(device, parameter)| Direction {
            device: *device,
            rates: vec![(parameter.name.clone(), 1.0)],
        })
        .collect()
}

/// Groups indices of `items` by device in first-seen order in linear time regardless of
/// device count.
fn by_device<T>(items: &[T], device: impl Fn(&T) -> usize) -> Vec<(usize, Vec<usize>)> {
    let mut position: BTreeMap<usize, usize> = BTreeMap::new();
    let mut grouped: Vec<(usize, Vec<usize>)> = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let each = device(item);
        let at = *position.entry(each).or_insert_with(|| {
            grouped.push((each, Vec::new()));
            grouped.len() - 1
        });
        grouped[at].1.push(index);
    }
    grouped
}

impl Simulator {
    /// Builds unit-seeded [`Twins`] for `parameters` and returns any parameters unread as
    /// real numbers (such as discrete choices or out-of-range fields).
    pub(crate) fn unit_twins(
        &self,
        parameters: Vec<(usize, Parameter)>,
    ) -> (Twins, Vec<Parameter>) {
        let directions = units(&parameters);
        let chunked = self.chunked(&directions);
        let mut kept = Vec::with_capacity(parameters.len());
        let mut left_out = Vec::new();
        for (column, (_, parameter)) in parameters.into_iter().enumerate() {
            if chunked.unread.binary_search(&column).is_ok() {
                left_out.push(parameter);
            } else {
                kept.push(parameter);
            }
        }
        (Twins::new(kept, directions, chunked), left_out)
    }

    /// Builds tangent-seeded [`Twins`] for `tangents` over `parameters`, packing each
    /// tangent's components per device and returning per-tangent column selector vectors.
    ///
    /// Fails with the list of parameters not read as real numbers by their device models.
    pub(crate) fn tangent_twins(
        &self,
        parameters: Vec<(usize, Parameter)>,
        tangents: &[Vec<f64>],
    ) -> Result<(Twins, Vec<Vec<f64>>), Vec<Parameter>> {
        let devices = by_device(&parameters, |(device, _)| *device);
        let mut directions = Vec::with_capacity(tangents.len() * devices.len());
        let mut along = vec![vec![0.0; tangents.len() * devices.len()]; tangents.len()];
        for (index, tangent) in tangents.iter().enumerate() {
            for (device, members) in &devices {
                let rates = members
                    .iter()
                    .map(|&member| (parameters[member].1.name.clone(), tangent[member]))
                    .collect();
                along[index][directions.len()] = 1.0;
                directions.push(Direction {
                    device: *device,
                    rates,
                });
            }
        }
        let chunked = self.chunked(&directions);
        // With no tangent there is no column, so each parameter is tried alone.
        let unread = if tangents.is_empty() || !chunked.unread.is_empty() {
            self.chunked(&units(&parameters)).unread
        } else {
            Vec::new()
        };
        if !unread.is_empty() {
            return Err(unread
                .into_iter()
                .map(|column| parameters[column].1.clone())
                .collect());
        }
        let named = parameters
            .into_iter()
            .map(|(_, parameter)| parameter)
            .collect();
        Ok((Twins::new(named, directions, chunked), along))
    }

    /// Packs `directions` into per-device twins of up to [`Seeds::MAX`] seeds and collects
    /// indices of columns rejected as unread real parameters.
    fn chunked(&self, directions: &[Direction]) -> Chunked {
        let mut chunks = Vec::new();
        let mut places = vec![
            Place {
                chunk: usize::MAX,
                seed: 0,
            };
            directions.len()
        ];
        let mut unread = Vec::new();
        let twin_of = |device: usize, columns: &[usize]| {
            let along: Vec<&Direction> =
                columns.iter().map(|&column| &directions[column]).collect();
            seeds(&along, None).and_then(|seeds| self.twin_of(device, &seeds))
        };
        for (device, columns) in by_device(directions, |direction| direction.device) {
            for members in columns.chunks(Seeds::MAX) {
                let mut members = members.to_vec();
                let mut twin = twin_of(device, &members);
                if twin.is_none() {
                    // Isolate unreadable parameter columns individually, then batch the rest.
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
                    places[column] = Place {
                        chunk: chunks.len(),
                        seed,
                    };
                }
                chunks.push(Chunk {
                    device,
                    twin,
                    columns: members,
                });
            }
        }
        unread.sort_unstable();
        Chunked {
            chunks,
            places,
            unread,
        }
    }

    /// Instantiates a dual-number [`Twin`] for device `index` along `seeds`, returning
    /// `None` if any seed path is not read as a real parameter.
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

    /// Returns device indices of all nonlinear kernels in placement order.
    pub(crate) fn nonlinear_devices(&self) -> Vec<usize> {
        (0..self.elaborated.devices.len())
            .filter(|&device| !self.elaborated.groups.is_linear(device))
            .collect()
    }

    /// Instantiates unknown-seeded twins for every nonlinear device in `nonlinear` and
    /// returns the sorted, deduplicated matrix slots written by those twins.
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

    /// Computes directional Jacobian derivatives `(slot, ∂G/∂x · s, ∂C/∂x · s)` in
    /// ascending slot order along each shift vector in `shifts` over `nonlinear` devices.
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
            for (chunk_index, chunk) in shifts.chunks(Seeds::MAX).enumerate() {
                if let Some(seeds) = Seeds::tangents(chunk.len())
                    && let Some(twin) =
                        self.state
                            .runs
                            .twin(&self.elaborated.groups, device, instance, &seeds)
                {
                    tasks.push((chunk_index * Seeds::MAX, chunk, twin));
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

    /// Computes unknown gradients of `Σ_s (W_G[s] G[s](x) + W_C[s] C[s](x))` at `x` for
    /// each active weight pair `(W_G, W_C)` indexed by `slots` across `twins`.
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
