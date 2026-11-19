//! Parameters by path: a parameter's own name, or its keys joined by dots within an object
//! and a list's entries keyed by their index from 0, such as `resistance`, `waveform.dc`
//! or `waveform.piecewise.1.1`.

use serde_json::Value;

use crate::Parameters;

/// A path that has no place among a device's parameters.
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

/// The value at `path` among `parameters`, if one is there.
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

/// The value at `path` among `parameters`, to change, if one is there.
#[must_use]
pub fn parameter_mut<'a>(parameters: &'a mut Parameters, path: &str) -> Option<&'a mut Value> {
    let mut keys = path.split('.');
    let first = parameters.get_mut(keys.next()?)?;
    keys.try_fold(first, |value, key| inner_mut(value, key))
}

/// Sets the value at `path` among `parameters`: within an object, whether or not one is
/// there, and within a list, at an entry that is.
///
/// # Errors
///
/// When the keys before the last reach no object or list, or the last no entry of a
/// list.
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

/// Every number among `parameters`, with its path, in the order of their keys.
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

/// The value `key` names within `value`, to change: an object's entry, or a list's by
/// its index.
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
    fn setting_a_path_keeps_its_neighbours() {
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
