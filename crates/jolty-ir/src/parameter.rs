//! Dot-separated parameter path addressing (for example, `resistance`, `waveform.dc`, or
//! `waveform.piecewise.1.1`), using object keys and zero-based array indices.

use serde_json::Value;

use crate::Parameters;

/// Error returned when a parameter path cannot be reached in a device's parameter map.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("`{0}` has no place among the parameters")]
pub struct Unreachable(pub String);

/// Parses `key` as an unsigned ASCII-digit array index, rejecting signed forms such as `"+0"`.
fn parse_index(key: &str) -> Option<usize> {
    if !key.is_empty() && key.bytes().all(|byte| byte.is_ascii_digit()) {
        key.parse().ok()
    } else {
        None
    }
}

/// Returns the parameter value at `path` within `parameters`, if present.
#[must_use]
pub fn parameter<'a>(parameters: &'a Parameters, path: &str) -> Option<&'a Value> {
    let mut keys = path.split('.');
    let first = parameters.get(keys.next()?)?;
    keys.try_fold(first, |value, key| match value {
        Value::Object(object) => object.get(key),
        Value::Array(items) => items.get(parse_index(key)?),
        _ => None,
    })
}

/// Returns a mutable reference to the parameter value at `path` within `parameters`, if present.
#[must_use]
pub fn parameter_mut<'a>(parameters: &'a mut Parameters, path: &str) -> Option<&'a mut Value> {
    let mut keys = path.split('.');
    let first = parameters.get_mut(keys.next()?)?;
    keys.try_fold(first, |value, key| inner_mut(value, key))
}

/// Sets the value at `path` within `parameters`, inserting into objects and updating existing
/// array entries.
///
/// # Errors
///
/// Returns [`Unreachable`] when the prefix keys do not reach an object or array, or when the
/// final key does not index an existing array entry.
pub fn set_parameter(
    parameters: &mut Parameters,
    path: &str,
    value: Value,
) -> Result<(), Unreachable> {
    let unreachable = || Unreachable(path.to_owned());
    let Some((within, last)) = path.rsplit_once('.') else {
        parameters.insert(path.to_owned(), value);
        return Ok(());
    };
    match parameter_mut(parameters, within).ok_or_else(unreachable)? {
        Value::Object(object) => {
            object.insert(last.to_owned(), value);
        }
        items @ Value::Array(_) => *inner_mut(items, last).ok_or_else(unreachable)? = value,
        _ => return Err(unreachable()),
    }
    Ok(())
}

/// Returns every numeric value in `parameters` paired with its dot-separated path in key
/// order.
#[must_use]
pub fn numbers(parameters: &Parameters) -> Vec<(String, f64)> {
    fn walk(path: String, value: &Value, found: &mut Vec<(String, f64)>) {
        match value {
            Value::Number(number) => {
                if let Some(number) = number.as_f64() {
                    found.push((path, number));
                }
            }
            Value::Object(object) => {
                for (key, inner) in object {
                    walk(format!("{path}.{key}"), inner, found);
                }
            }
            Value::Array(items) => {
                for (index, inner) in items.iter().enumerate() {
                    walk(format!("{path}.{index}"), inner, found);
                }
            }
            Value::Null | Value::Bool(_) | Value::String(_) => {}
        }
    }
    let mut found = Vec::new();
    for (key, value) in parameters {
        walk(key.clone(), value, &mut found);
    }
    found
}

/// Returns a mutable reference to the object field or array element named by `key` within
/// `value`.
fn inner_mut<'a>(value: &'a mut Value, key: &str) -> Option<&'a mut Value> {
    match value {
        Value::Object(object) => object.get_mut(key),
        Value::Array(items) => items.get_mut(parse_index(key)?),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn source() -> Parameters {
        json!({
            "resistance": 1e3, "waveform": {"pulse": {"initial": 0, "pulsed": 5}},
            "channel": "n", "points": [[0, 1]]
        })
        .as_object()
        .cloned()
        .unwrap()
    }

    #[test]
    fn finds_every_number_by_its_path() {
        let parameters = source();
        let paths: Vec<String> = numbers(&parameters)
            .into_iter()
            .map(|(path, _)| path)
            .collect();
        assert_eq!(
            paths,
            [
                "points.0.0",
                "points.0.1",
                "resistance",
                "waveform.pulse.initial",
                "waveform.pulse.pulsed"
            ]
        );
        for (path, number) in numbers(&parameters) {
            assert_eq!(
                parameter(&parameters, &path).and_then(Value::as_f64),
                Some(number)
            );
        }
        assert_eq!(parameter(&parameters, "waveform.dc"), None);
        assert_eq!(parameter(&parameters, "points.1.0"), None);
        assert_eq!(parameter(&parameters, "resistance.dc"), None);
    }

    #[test]
    fn setting_a_path_keeps_its_neighbors() {
        let mut parameters = source();
        set_parameter(&mut parameters, "waveform.pulse.pulsed", json!(3.3)).unwrap();
        set_parameter(&mut parameters, "waveform.pulse.width", json!(1e-6)).unwrap();
        set_parameter(&mut parameters, "points.0.1", json!(2.0)).unwrap();
        set_parameter(&mut parameters, "added", json!(4.0)).unwrap();
        assert_eq!(
            Value::Object(parameters.clone()),
            json!({
                "resistance": 1e3,
                "waveform": {"pulse": {"initial": 0, "pulsed": 3.3, "width": 1e-6}},
                "channel": "n", "points": [[0, 2.0]], "added": 4.0
            })
        );
        for path in ["resistance.dc", "missing.dc", "points.0.2", "points.x.0"] {
            assert_eq!(
                set_parameter(&mut parameters, path, json!(1.0)),
                Err(Unreachable(path.to_owned()))
            );
        }
        assert_eq!(
            Unreachable("resistance.dc".to_owned()).to_string(),
            "`resistance.dc` has no place among the parameters"
        );
    }

    #[test]
    fn rejects_signed_or_non_digit_array_indices() {
        let mut parameters = source();
        for invalid in [
            "points.+0.0",
            "points.0.+1",
            "points.-0.0",
            "points..0",
            "points. 0.0",
            "points.0 .0",
        ] {
            assert_eq!(parameter(&parameters, invalid), None, "{invalid}");
            assert_eq!(parameter_mut(&mut parameters, invalid), None, "{invalid}");
            assert_eq!(
                set_parameter(&mut parameters, invalid, json!(9.0)),
                Err(Unreachable(invalid.to_owned())),
                "{invalid}"
            );
        }
        assert_eq!(parameter(&parameters, "points.0.1"), Some(&json!(1)));
    }
}
