//! Values: one point in a circuit's parameter space, as patches on the nominal numbers of
//! an [elaborated circuit](crate::Elaborated)'s table of parameters.
//!
//! Every number of every device instance, as the circuit writes it, has a [`ParameterId`]
//! in its elaborated circuit's table: the instances in the order of their paths, and each
//! one's numbers in the order its parameters hold them, by key and then by index in a list,
//! as [`jolty_ir::numbers`] lists them. A run's [`Values`] are its temperature and the
//! numbers that differ from the nominal ones, so they are a few bytes, whatever the circuit.

use serde_json::Value;

/// A number of a device instance of an elaborated circuit, by its index in the circuit's
/// table of parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParameterId(u32);

impl ParameterId {
    /// The parameter's index in its circuit's table.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// One point in a circuit's parameter space: the devices' temperature, and the numbers
/// that replace the nominal ones of an [elaborated circuit](crate::Elaborated)'s parameters,
/// as a Monte Carlo sample's, a corner's or a candidate design's do.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Values {
    /// The devices' temperature, in kelvins, or `None` for the one the circuit was
    /// elaborated at.
    pub temperature: Option<f64>,
    /// Each number replaced, by its parameter, in the order of the parameters, one each.
    patches: Vec<(ParameterId, f64)>,
}

impl Values {
    /// The values at `temperature`, in kelvins, with no number replaced.
    #[must_use]
    pub const fn at(temperature: f64) -> Self {
        Self {
            temperature: Some(temperature),
            patches: Vec::new(),
        }
    }

    /// Replaces the number of `parameter` with `value`, or the value that replaced it.
    pub fn set(&mut self, parameter: ParameterId, value: f64) {
        match self
            .patches
            .binary_search_by_key(&parameter, |&(patched, _)| patched)
        {
            Ok(found) => self.patches[found].1 = value,
            Err(at) => self.patches.insert(at, (parameter, value)),
        }
    }

    /// Each number replaced, by its parameter, in the order of the parameters.
    #[must_use]
    pub fn patches(&self) -> &[(ParameterId, f64)] {
        &self.patches
    }

    /// The number that replaces `parameter`'s, if any.
    #[must_use]
    pub fn get(&self, parameter: ParameterId) -> Option<f64> {
        let found = self
            .patches
            .binary_search_by_key(&parameter, |&(patched, _)| patched)
            .ok()?;
        Some(self.patches[found].1)
    }
}

/// The numbers of an elaborated circuit's device instances: their nominal values, by
/// parameter, and the parameters of each instance.
#[derive(Debug, Clone, Default)]
pub(crate) struct Table {
    /// The first parameter of each device, by the device's index; a device's parameters
    /// run to the next one's in the order of the paths.
    firsts: Vec<usize>,
    /// The number of parameters of each device, by its index.
    counts: Vec<usize>,
    /// Each parameter's nominal value.
    nominal: Vec<f64>,
    /// Each parameter's device, by the device's index.
    devices: Vec<usize>,
}

impl Table {
    /// The table of the devices whose parameters `parameters` gives, by index, `None` for
    /// one made from none, taken in the order `by_path` lists the devices.
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

    /// Each parameter's nominal value.
    pub(crate) fn nominal(&self) -> &[f64] {
        &self.nominal
    }

    /// The parameters of `device`, by their indices: none for a device the table does not
    /// hold.
    pub(crate) fn range(&self, device: usize) -> std::ops::Range<usize> {
        match (self.firsts.get(device), self.counts.get(device)) {
            (Some(&first), Some(&count)) => first..first + count,
            _ => 0..0,
        }
    }

    /// The device whose number `parameter` is, by its index, and the number's place among
    /// the device's, if the table holds it.
    pub(crate) fn locate(&self, parameter: ParameterId) -> Option<(usize, usize)> {
        let device = *self.devices.get(parameter.index())?;
        Some((device, parameter.index() - self.firsts[device]))
    }

    /// The parameter of the `place`-th number of `device`, if it has one there.
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
