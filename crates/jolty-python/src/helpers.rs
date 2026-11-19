//! Shared helpers for JSON parsing, error mapping, unpoisoned locking, and `NumPy` slices.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Mutex, MutexGuard, PoisonError};

use jolty_engine::{Cotangent, ElaborationError, SimulationError};
use numpy::{Element, PyArray1, PyReadonlyArray2, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Parses JSON text into a deserializable value, raising `ValueError` on syntax or schema
/// errors.
macro_rules! parse_json {
    ($text:expr) => {
        $crate::helpers::value_result(serde_json::from_str($text))
    };
}

pub(crate) use parse_json;

/// Converts a result with a displayable error into a `PyResult`, raising `ValueError`.
pub(crate) fn value_result<T, E: fmt::Display>(result: Result<T, E>) -> PyResult<T> {
    result.map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Converts a simulation result into a `PyResult`, raising `SimulationError`.
pub(crate) fn simulation_result<T>(result: Result<T, SimulationError>) -> PyResult<T> {
    result.map_err(|error| crate::SimulationError::new_err(error.to_string()))
}

/// Converts an elaboration result into a `PyResult`, raising `ElaborationError` with its
/// list of problems.
pub(crate) fn elaboration_result<T>(
    py: Python<'_>,
    result: Result<T, ElaborationError>,
) -> PyResult<T> {
    result.map_err(
        |error| match crate::convert::problems_to_python(py, &error) {
            Ok(problems) => crate::ElaborationError::new_err((problems.unbind(),)),
            Err(conversion_error) => conversion_error,
        },
    )
}

/// Locks `mutex`, recovering the guard even if a prior holder panicked.
pub(crate) fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Consumes `mutex`, recovering its inner value even if a prior holder panicked.
pub(crate) fn into_inner_unpoisoned<T>(mutex: Mutex<T>) -> T {
    mutex.into_inner().unwrap_or_else(PoisonError::into_inner)
}

/// Allocates a one-dimensional `NumPy` array of `len` elements, each initialized to `value`.
pub(crate) fn filled<T: Element + Copy>(
    py: Python<'_>,
    len: usize,
    value: T,
) -> Bound<'_, PyArray1<T>> {
    PyArray1::from_vec(py, vec![value; len])
}

/// Borrows every element of `array` as a mutable contiguous slice, raising `ValueError`
/// when `array` is not contiguous.
pub(crate) fn slice_mut<'a, T: Element>(
    array: &'a mut PyReadwriteArray1<'_, T>,
) -> PyResult<&'a mut [T]> {
    value_result(array.as_slice_mut())
}

/// Builds a [`Cotangent`] from `values` (indexed by point and unknown) and an oscillator
/// `period`, omitting all-zero point rows.
pub(crate) fn cotangent<T: Element + Copy + Default + PartialEq>(
    values: &PyReadonlyArray2<'_, T>,
    period: f64,
) -> Cotangent<T> {
    let array = values.as_array();
    let zero = T::default();
    let points: BTreeMap<usize, Vec<T>> = array
        .rows()
        .into_iter()
        .enumerate()
        .filter(|(_, row)| row.iter().any(|value| *value != zero))
        .map(|(point, row)| (point, row.to_vec()))
        .collect();
    Cotangent { points, period }
}
