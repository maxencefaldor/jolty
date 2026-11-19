//! Compact point in circuit parameter space stored as sparse patches over an
//! [elaborated circuit](crate::Elaborated)'s parameter table.
//!
//! Every numeric device parameter receives a [`ParameterId`] ordered by instance path,
//! parameter key, and array index (matching [`jolty_ir::numbers`]). A [`Values`] instance
//! stores only the optional temperature and non-nominal parameter overrides, keeping
//! per-run memory minimal.

use serde_json::Value;

/// Handle identifying a numeric device parameter within an elaborated circuit's parameter
/// table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParameterId(u32);

impl ParameterId {
    /// Creates a [`ParameterId`] at `index` for unit tests.
    #[cfg(test)]
    pub(crate) const fn new(index: u32) -> Self {
        Self(index)
    }

    /// Returns the zero-based table index of this parameter.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Point in circuit parameter space: optional temperature and sorted sparse overrides over
/// an [elaborated circuit](crate::Elaborated)'s nominal parameters.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Values {
    /// Operating temperature in kelvins, or `None` to inherit the elaboration temperature.
    pub temperature: Option<f64>,
    /// Sorted, deduplicated `(ParameterId, value)` overrides.
    patches: Vec<(ParameterId, f64)>,
}

impl Values {
    /// Creates [`Values`] overriding only operating `temperature` in kelvins.
    #[must_use]
    pub const fn at(temperature: f64) -> Self {
        Self {
            temperature: Some(temperature),
            patches: Vec::new(),
        }
    }

    /// Sets or updates the override for `parameter` to `value`.
    pub fn set(&mut self, parameter: ParameterId, value: f64) {
        match self
            .patches
            .binary_search_by_key(&parameter, |&(patched, _)| patched)
        {
            Ok(found) => self.patches[found].1 = value,
            Err(at) => self.patches.insert(at, (parameter, value)),
        }
    }

    /// Returns all parameter overrides in ascending [`ParameterId`] order.
    #[must_use]
    pub fn patches(&self) -> &[(ParameterId, f64)] {
        &self.patches
    }

    /// Returns the override value for `parameter`, if present.
    #[must_use]
    pub fn get(&self, parameter: ParameterId) -> Option<f64> {
        let found = self
            .patches
            .binary_search_by_key(&parameter, |&(patched, _)| patched)
            .ok()?;
        Some(self.patches[found].1)
    }

    /// Merges temperature (if set) and all parameter overrides from `other` into `self`.
    pub fn extend(&mut self, other: &Self) {
        if other.temperature.is_some() {
            self.temperature = other.temperature;
        }
        for &(parameter, value) in &other.patches {
            self.set(parameter, value);
        }
    }
}

/// Flat table indexing every numeric device parameter of an elaborated circuit.
#[derive(Debug, Clone, Default)]
pub(crate) struct Table {
    /// Starting parameter offset for each device index.
    firsts: Vec<usize>,
    /// Number of numeric parameters owned by each device index.
    counts: Vec<usize>,
    /// Nominal value of every indexed parameter.
    nominal: Vec<f64>,
    /// Owning device index for every indexed parameter.
    devices: Vec<usize>,
}

impl Table {
    /// Builds a parameter [`Table`] from per-device JSON `parameters` ordered by `by_path`.
    pub(crate) fn new(
        parameters: &[Option<&serde_json::Map<String, Value>>],
        by_path: &[usize],
    ) -> Self {
        let mut table = Self {
            firsts: vec![0; parameters.len()],
            counts: vec![0; parameters.len()],
            nominal: Vec::new(),
            devices: Vec::new(),
        };
        for &device in by_path {
            table.firsts[device] = table.nominal.len();
            if let Some(parameters) = parameters[device] {
                for value in parameters.values() {
                    each_number(value, &mut |number| {
                        table.nominal.extend(number.as_f64());
                    });
                }
            }
            table.counts[device] = table.nominal.len() - table.firsts[device];
            table.devices.resize(table.nominal.len(), device);
        }
        table
    }

    /// Returns the slice of nominal values for all indexed parameters.
    pub(crate) fn nominal(&self) -> &[f64] {
        &self.nominal
    }

    /// Returns the parameter index range owned by `device`, or an empty range if absent.
    pub(crate) fn range(&self, device: usize) -> std::ops::Range<usize> {
        match (self.firsts.get(device), self.counts.get(device)) {
            (Some(&first), Some(&count)) => first..first + count,
            _ => 0..0,
        }
    }

    /// Returns `(device_index, local_offset)` for `parameter`, if present in the table.
    pub(crate) fn locate(&self, parameter: ParameterId) -> Option<(usize, usize)> {
        let device = *self.devices.get(parameter.index())?;
        Some((device, parameter.index() - self.firsts[device]))
    }

    /// Returns the [`ParameterId`] at `place` within `device`, if within range.
    pub(crate) fn parameter(&self, device: usize, place: usize) -> Option<ParameterId> {
        (place < *self.counts.get(device)?)
            .then(|| {
                u32::try_from(self.firsts[device] + place)
                    .ok()
                    .map(ParameterId)
            })
            .flatten()
    }
}

/// Visits every number within `value`, in the order [`jolty_ir::numbers`] lists them: an
/// object's by key, a list's by index.
pub(crate) fn each_number<'a>(value: &'a Value, visit: &mut impl FnMut(&'a Value)) {
    match value {
        Value::Number(number) => {
            if number.as_f64().is_some() {
                visit(value);
            }
        }
        Value::Object(object) => {
            for inner in object.values() {
                each_number(inner, visit);
            }
        }
        Value::Array(items) => {
            for inner in items {
                each_number(inner, visit);
            }
        }
        Value::Null | Value::Bool(_) | Value::String(_) => {}
    }
}

/// Visits every number within `value` to change it, in the order of [`each_number`].
pub(crate) fn each_number_mut(value: &mut Value, visit: &mut impl FnMut(&mut Value)) {
    match value {
        Value::Number(number) => {
            if number.as_f64().is_some() {
                visit(value);
            }
        }
        Value::Object(object) => {
            for inner in object.values_mut() {
                each_number_mut(inner, visit);
            }
        }
        Value::Array(items) => {
            for inner in items {
                each_number_mut(inner, visit);
            }
        }
        Value::Null | Value::Bool(_) | Value::String(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_hold_one_patch_per_parameter_in_order() {
        let mut values = Values::at(300.0);
        values.set(ParameterId(4), 1.0);
        values.set(ParameterId(1), 2.0);
        values.set(ParameterId(4), 3.0);
        assert_eq!(
            values.patches(),
            [(ParameterId(1), 2.0), (ParameterId(4), 3.0)]
        );
        assert_eq!(values.temperature, Some(300.0));
    }

    #[test]
    fn a_table_numbers_each_device_in_the_order_of_the_paths() {
        let first = serde_json::json!({"b": 2.0, "a": {"y": [3.0, "x", 4], "x": 5.0}});
        let second = serde_json::json!({"resistance": 7.0});
        let (Value::Object(first), Value::Object(second)) = (first, second) else {
            panic!("objects");
        };
        // The second device's path comes first.
        let table = Table::new(&[Some(&first), None, Some(&second)], &[2, 1, 0]);
        assert_eq!(table.nominal(), [7.0, 5.0, 3.0, 4.0, 2.0]);
        assert_eq!(table.locate(ParameterId(0)), Some((2, 0)));
        assert_eq!(table.locate(ParameterId(3)), Some((0, 2)));
        assert_eq!(table.locate(ParameterId(5)), None);
        assert_eq!(table.parameter(0, 3), Some(ParameterId(4)));
        assert_eq!(table.parameter(0, 4), None);
        assert_eq!(table.parameter(1, 0), None);
        let mut listed = Vec::new();
        for value in first.values() {
            each_number(value, &mut |number| listed.extend(number.as_f64()));
        }
        let numbers: Vec<f64> = jolty_ir::numbers(&first)
            .into_iter()
            .map(|(_, number)| number)
            .collect();
        assert_eq!(listed, numbers);
    }
}
