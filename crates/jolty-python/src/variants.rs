//! Batch parameter variants exposed as `jolty._core.Variants`.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use jolty_engine::batch::Variants as EngineVariants;
use jolty_engine::variants::{Axis, Explicit, Grid, MonteCarlo, Product, Toleranced};
use jolty_engine::{ParameterId, Values};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::Elaborated;
use crate::helpers::{elaboration_result, parse_json, value_result};

/// The variants of a batch as Python composes them.
#[derive(Debug, Clone)]
enum TolerancedParameter {
    Explicit(Arc<Explicit>),
    Grid(Arc<Grid>),
    /// The samples, and their seed.
    MonteCarlo(Arc<MonteCarlo>, u64),
    Product(Arc<Product<Self, Self>>),
}

impl TolerancedParameter {
    fn inner(&self) -> &dyn EngineVariants {
        match self {
            Self::Explicit(explicit) => explicit.as_ref(),
            Self::Grid(grid) => grid.as_ref(),
            Self::MonteCarlo(samples, _) => samples.as_ref(),
            Self::Product(product) => product.as_ref(),
        }
    }

    /// Returns the Monte Carlo `(seed, sample_index)` drawn by run `index`, preferring the
    /// second factor of a product, or `None` for non-Monte-Carlo variants.
    fn sample(&self, index: u64) -> Option<(u64, u64)> {
        match self {
            Self::Explicit(_) | Self::Grid(_) => None,
            Self::MonteCarlo(_, seed) => Some((*seed, index)),
            Self::Product(product) => {
                let (first, second) = product.parts();
                let count = second.len().max(1);
                second
                    .sample(index % count)
                    .or_else(|| first.sample(index / count))
            }
        }
    }
}

impl EngineVariants for TolerancedParameter {
    fn len(&self) -> u64 {
        self.inner().len()
    }

    fn values(&self, index: u64) -> Values {
        self.inner().values(index)
    }

    fn chain(&self) -> u64 {
        self.inner().chain()
    }
}

/// Quantity varied across batch runs: either a device parameter (`instance` and parameter
/// path) or temperature.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Column {
    Parameter(ParameterId, String, String),
    Temperature,
}

/// Parameter and temperature values for batch runs of one elaborated circuit, defined as
/// explicit lists, grids, Monte Carlo samples, or Cartesian products.
#[pyclass(module = "jolty._core", frozen)]
pub(crate) struct Variants {
    toleranced: TolerancedParameter,
    elaborated: Arc<jolty_engine::Elaborated>,
    /// Varied quantities in batch output column order.
    columns: Vec<Column>,
}

/// Device parameter identified by `(instance_path, parameter_path)`.
type Named = (String, String);

/// Parameter override map indexed by instance path, then parameter path.
type Replaced = BTreeMap<String, BTreeMap<String, f64>>;

impl Variants {
    /// Verifies that these variants belong to the same elaborated circuit as `elaborated`.
    pub(crate) fn ensure_circuit(
        &self,
        elaborated: &Arc<jolty_engine::Elaborated>,
    ) -> PyResult<()> {
        if !Arc::ptr_eq(elaborated, &self.elaborated) {
            return Err(PyValueError::new_err(
                "the variants vary another elaborated circuit than the simulator's",
            ));
        }
        Ok(())
    }

    /// Returns the varied columns in batch output order.
    pub(crate) fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// Wraps these variants on top of the simulator's `base` values.
    pub(crate) fn based<'a>(&'a self, base: &'a Values) -> Based<'a> {
        Based {
            base,
            toleranced: &self.toleranced,
        }
    }
}

#[pymethods]
impl Variants {
    /// Creates variants from explicit `(overrides_json, temperature)` run entries.
    ///
    /// Each entry specifies a JSON object mapping instance path to parameter overrides, plus
    /// an optional temperature in kelvins (`None` keeps the base circuit temperature).
    /// Raises `ElaborationError` naming every unrecognized parameter path.
    #[staticmethod]
    fn explicit(
        py: Python<'_>,
        elaborated: &Elaborated,
        runs: Vec<(String, Option<f64>)>,
    ) -> PyResult<Self> {
        let circuit = &elaborated.inner;
        let mut columns = BTreeSet::new();
        let mut listed = Vec::with_capacity(runs.len());
        let mut heated = false;
        for (text, temperature) in runs {
            let replaced: Replaced = parse_json!(&text)?;
            let values = elaboration_result(py, circuit.values(&replaced, temperature))?;
            for (path, numbers) in replaced {
                for name in numbers.into_keys() {
                    if let Some(parameter) = circuit.parameter(&path, &name) {
                        columns.insert(Column::Parameter(parameter, path.clone(), name));
                    }
                }
            }
            heated |= temperature.is_some();
            listed.push(values);
        }
        let mut columns: Vec<Column> = columns.into_iter().collect();
        if heated {
            columns.push(Column::Temperature);
        }
        Ok(Self {
            toleranced: TolerancedParameter::Explicit(Arc::new(Explicit(listed))),
            elaborated: Arc::clone(circuit),
            columns,
        })
    }

    /// Creates variants on a Cartesian grid of `axes`, with the last axis varying fastest.
    ///
    /// Each axis pairs `(instance_path, parameter_path)` (or `None` for temperature in
    /// kelvins) with ordered values. When `warm` is true, lines along the last axis execute
    /// as warm-started chains.
    ///
    /// Raises `ElaborationError` naming every unrecognized parameter path, and `ValueError`
    /// when an axis contains no values.
    #[staticmethod]
    fn grid(
        py: Python<'_>,
        elaborated: &Elaborated,
        axes: Vec<(Option<Named>, Vec<f64>)>,
        warm: bool,
    ) -> PyResult<Self> {
        let circuit = &elaborated.inner;
        let mut named = Replaced::new();
        for (path, name) in axes.iter().filter_map(|(axis, _)| axis.as_ref()) {
            named
                .entry(path.clone())
                .or_default()
                .insert(name.clone(), 0.0);
        }
        // Validate every axis parameter using the same rules as explicit overrides.
        elaboration_result(py, circuit.values(&named, None))?;
        let mut stepped = Vec::with_capacity(axes.len());
        let mut columns = Vec::with_capacity(axes.len());
        for (axis, values) in axes {
            if let Some((path, name)) = axis {
                let parameter = circuit.parameter(&path, &name).ok_or_else(|| {
                    PyValueError::new_err(format!("{path} writes no number at {name}"))
                })?;
                stepped.push((Axis::Parameter(parameter), values));
                columns.push(Column::Parameter(parameter, path, name));
            } else {
                stepped.push((Axis::Temperature, values));
                columns.push(Column::Temperature);
            }
        }
        let grid = value_result(Grid::new(stepped, warm))?;
        Ok(Self {
            toleranced: TolerancedParameter::Grid(Arc::new(grid)),
            elaborated: Arc::clone(circuit),
            columns,
        })
    }

    /// Creates `samples` Monte Carlo variants of `toleranced` parameters drawn from `seed`
    /// via [`crate::sample`].
    ///
    /// `toleranced` is a JSON list of objects with `instance`, `parameter`, `nominal`, and
    /// `tolerance` fields. Raises `ElaborationError` naming every unrecognized parameter
    /// path.
    #[staticmethod]
    fn monte_carlo(
        py: Python<'_>,
        elaborated: &Elaborated,
        toleranced: &str,
        seed: u64,
        samples: u64,
    ) -> PyResult<Self> {
        let circuit = &elaborated.inner;
        let toleranced: Vec<Toleranced> = parse_json!(toleranced)?;
        let columns = toleranced
            .iter()
            .filter_map(|item| {
                let parameter = circuit.parameter(&item.instance, &item.parameter)?;
                Some(Column::Parameter(
                    parameter,
                    item.instance.clone(),
                    item.parameter.clone(),
                ))
            })
            .collect();
        let drawn = elaboration_result(py, MonteCarlo::new(circuit, seed, samples, toleranced))?;
        Ok(Self {
            toleranced: TolerancedParameter::MonteCarlo(Arc::new(drawn), seed),
            elaborated: Arc::clone(circuit),
            columns,
        })
    }

    /// Creates the Cartesian product of `first` and `second` variants.
    ///
    /// Run `i` combines variant `i // len(second)` of `first` with variant `i % len(second)`
    /// of `second`, with `second` overriding `first` on shared parameters.
    ///
    /// Raises `ValueError` when the operands belong to different circuits or when the
    /// product exceeds `2**64 - 1` runs.
    #[staticmethod]
    fn product(first: &Self, second: &Self) -> PyResult<Self> {
        if !Arc::ptr_eq(&first.elaborated, &second.elaborated) {
            return Err(PyValueError::new_err(
                "a product's variants vary one elaborated circuit",
            ));
        }
        let product = value_result(Product::new(
            first.toleranced.clone(),
            second.toleranced.clone(),
        ))?;
        let mut columns = first.columns.clone();
        for column in &second.columns {
            if !columns.contains(column) {
                columns.push(column.clone());
            }
        }
        Ok(Self {
            toleranced: TolerancedParameter::Product(Arc::new(product)),
            elaborated: Arc::clone(&first.elaborated),
            columns,
        })
    }

    /// Returns the total number of runs in the variant set.
    fn __len__(&self) -> PyResult<usize> {
        usize::try_from(self.toleranced.len())
            .map_err(|_| PyValueError::new_err("more runs than an index counts"))
    }

    /// Returns the varied columns in batch output order: `(instance_path, parameter_path)`
    /// for parameters or `None` for temperature.
    #[getter]
    pub(crate) fn parameters(&self) -> Vec<Option<(String, String)>> {
        self.columns
            .iter()
            .map(|column| match column {
                Column::Parameter(_, path, name) => Some((path.clone(), name.clone())),
                Column::Temperature => None,
            })
            .collect()
    }

    /// Returns the parameter overrides (indexed by instance path then parameter path) and
    /// optional temperature in kelvins for run `index`.
    ///
    /// Raises `ValueError` when `index` is out of range.
    fn overrides(&self, index: u64) -> PyResult<(Replaced, Option<f64>)> {
        if index >= self.toleranced.len() {
            return Err(PyValueError::new_err(format!(
                "run {index} lies past the {} runs",
                self.toleranced.len()
            )));
        }
        let values = self.toleranced.values(index);
        let mut replaced = Replaced::new();
        for column in &self.columns {
            if let Column::Parameter(parameter, path, name) = column
                && let Some(value) = values.get(*parameter)
            {
                replaced
                    .entry(path.clone())
                    .or_default()
                    .insert(name.clone(), value);
            }
        }
        Ok((replaced, values.temperature))
    }

    /// Returns `(seed, sample_index)` for the Monte Carlo sample drawn by run `index`, or
    /// `None` when the run draws no sample.
    fn sample(&self, index: u64) -> Option<(u64, u64)> {
        (index < self.toleranced.len())
            .then(|| self.toleranced.sample(index))
            .flatten()
    }
}

/// Variant wrapper overlaying per-run overrides onto a simulator's base [`Values`].
pub(crate) struct Based<'a> {
    base: &'a Values,
    toleranced: &'a TolerancedParameter,
}

impl EngineVariants for Based<'_> {
    fn len(&self) -> u64 {
        self.toleranced.len()
    }

    fn values(&self, index: u64) -> Values {
        let mut values = self.base.clone();
        values.extend(&self.toleranced.values(index));
        values
    }

    fn chain(&self) -> u64 {
        self.toleranced.chain()
    }
}
