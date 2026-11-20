//! Parameter lifting from plain `f64` values to differentiable number types.
//!
//! Each real leaf is mapped by its dot-separated path so seeded parameters carry directional
//! derivatives while unseeded parameters remain constant, analogous to JAX
//! `tree_map_with_path`.
//!
//! Deriving `#[derive(Lift)]` on any parameter struct generic over a scalar type walks its
//! fields directly without round-tripping through JSON.

use std::collections::BTreeMap;
use std::fmt;

use crate::{Gradient, Hyper, ParameterError, Real};

/// Derive macro for [`Lift`](trait@Lift) on structs and enums generic over one scalar type.
pub use jolty_derive::Lift;

/// Conversion from a plain `f64`-based parameter structure to one parameterized by scalar
/// type `R`, transforming each real leaf using its parameter path.
pub trait Lift<R>: Sized {
    /// Plain `f64` representation of the parameter structure.
    type Plain;

    /// Constructs a lifted value from `plain` rooted at `path` by transforming each real leaf
    /// through `number`.
    fn lift<Lifting: FnMut(&ParameterPath<'_>, f64) -> R>(
        plain: &Self::Plain,
        path: &ParameterPath<'_>,
        number: &mut Lifting,
    ) -> Self;
}

/// Stack-allocated path identifying a field or sequence element inside a parameter tree.
///
/// Formats in the dot-separated syntax parsed by `jolty_ir::parameter`, such as
/// `waveform.dc`.
#[derive(Debug, Clone, Copy, Default)]
pub enum ParameterPath<'a> {
    /// Root parameter object.
    #[default]
    Root,
    /// Object field `key` nested inside `within`.
    Key {
        /// Enclosing container path.
        within: &'a ParameterPath<'a>,
        /// Field key name.
        key: &'a str,
    },
    /// Sequence element at `index` nested inside `within`.
    Index {
        /// Enclosing sequence path.
        within: &'a ParameterPath<'a>,
        /// Zero-based element index.
        index: usize,
    },
}

impl ParameterPath<'_> {
    /// Returns a child path addressing object field `key` under `self`.
    #[must_use]
    pub const fn key<'a>(&'a self, key: &'a str) -> ParameterPath<'a> {
        ParameterPath::Key { within: self, key }
    }

    /// Returns a child path addressing sequence element `index` under `self`.
    #[must_use]
    pub const fn index(&self, index: usize) -> ParameterPath<'_> {
        ParameterPath::Index {
            within: self,
            index,
        }
    }

    /// Returns whether `self` matches the dot-separated path `written`.
    #[must_use]
    pub fn is(&self, written: &str) -> bool {
        self.matches(&segments(written))
    }

    /// Returns whether `self` matches the parsed path `segments`.
    fn matches(&self, segments: &[Segment<'_>]) -> bool {
        match self {
            Self::Root => segments.is_empty(),
            Self::Key { within, key } => segments
                .split_last()
                .is_some_and(|(last, before)| last.text == *key && within.matches(before)),
            Self::Index { within, index } => segments
                .split_last()
                .is_some_and(|(last, before)| last.index == Some(*index) && within.matches(before)),
        }
    }
}

/// Parsed segment of a dot-separated path, storing both raw text and canonical decimal index.
#[derive(Debug, Clone, Copy)]
struct Segment<'a> {
    text: &'a str,
    index: Option<usize>,
}

/// Splits a dot-separated path `written` into parsed segments, returning an empty vector for
/// an empty string.
fn segments(written: &str) -> Vec<Segment<'_>> {
    if written.is_empty() {
        return Vec::new();
    }
    written
        .split('.')
        .map(|text| Segment {
            text,
            index: (!text.starts_with('+') && (text == "0" || !text.starts_with('0')))
                .then(|| text.parse().ok())
                .flatten(),
        })
        .collect()
}

impl fmt::Display for ParameterPath<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (within, step): (_, &dyn fmt::Display) = match self {
            Self::Root => return Ok(()),
            Self::Key { within, key } => (within, key),
            Self::Index { within, index } => (within, index),
        };
        match within {
            Self::Root => write!(f, "{step}"),
            _ => write!(f, "{within}.{step}"),
        }
    }
}

/// Implements [`Lift`] for scalar numeric types by invoking `number` at the leaf path.
macro_rules! numbers {
    ($($number:ty $(, const $width:ident)?);* $(;)?) => {$(
        impl$(<const $width: usize>)? Lift<$number> for $number {
            type Plain = f64;

            fn lift<Lifting: FnMut(&ParameterPath<'_>, f64) -> $number>(
                plain: &f64,
                path: &ParameterPath<'_>,
                number: &mut Lifting,
            ) -> Self {
                number(path, *plain)
            }
        }
    )*};
}

numbers!(f64; Gradient<K>, const K; Hyper<K>, const K);

impl<R, T: Lift<R>> Lift<R> for Option<T> {
    type Plain = Option<T::Plain>;

    fn lift<Lifting: FnMut(&ParameterPath<'_>, f64) -> R>(
        plain: &Self::Plain,
        path: &ParameterPath<'_>,
        number: &mut Lifting,
    ) -> Self {
        plain.as_ref().map(|plain| T::lift(plain, path, number))
    }
}

impl<R, T: Lift<R>> Lift<R> for Vec<T> {
    type Plain = Vec<T::Plain>;

    fn lift<Lifting: FnMut(&ParameterPath<'_>, f64) -> R>(
        plain: &Self::Plain,
        path: &ParameterPath<'_>,
        number: &mut Lifting,
    ) -> Self {
        plain
            .iter()
            .enumerate()
            .map(|(index, plain)| T::lift(plain, &path.index(index), number))
            .collect()
    }
}

impl<R, T: Lift<R>> Lift<R> for BTreeMap<String, T> {
    type Plain = BTreeMap<String, T::Plain>;

    fn lift<Lifting: FnMut(&ParameterPath<'_>, f64) -> R>(
        plain: &Self::Plain,
        path: &ParameterPath<'_>,
        number: &mut Lifting,
    ) -> Self {
        plain
            .iter()
            .map(|(key, value)| (key.clone(), T::lift(value, &path.key(key), number)))
            .collect()
    }
}

impl<R, A: Lift<R>, B: Lift<R>> Lift<R> for (A, B) {
    type Plain = (A::Plain, B::Plain);

    fn lift<Lifting: FnMut(&ParameterPath<'_>, f64) -> R>(
        (first, second): &Self::Plain,
        path: &ParameterPath<'_>,
        number: &mut Lifting,
    ) -> Self {
        (
            A::lift(first, &path.index(0), number),
            B::lift(second, &path.index(1), number),
        )
    }
}

/// Lifts plain `parameters` into `P` with every numeric leaf held constant.
pub(crate) fn constant<P: Lift<R>, R: Real>(parameters: &P::Plain) -> P {
    P::lift(parameters, &ParameterPath::Root, &mut |_, value| {
        R::constant(value)
    })
}

/// Lifts plain `parameters` into `P`, seeding each path in `seeds` via `seeded` and leaving
/// all other real leaves constant.
///
/// Each seed path appears at most once, as enforced by [`crate::Seeds`].
///
/// # Errors
///
/// Returns [`ParameterError`] when a seed path does not address a real number (for example,
/// when the path points to a discrete choice such as a model level).
pub(crate) fn lift<P: Lift<R>, R: Real, const K: usize>(
    parameters: &P::Plain,
    seeds: &[(&str, [f64; K])],
    seeded: impl Fn(f64, [f64; K]) -> R,
) -> Result<P, ParameterError> {
    let read: Vec<Vec<Segment<'_>>> = seeds.iter().map(|(path, _)| segments(path)).collect();
    let mut found = vec![false; seeds.len()];
    let lifted = P::lift(
        parameters,
        &ParameterPath::Root,
        &mut |path, value| match read.iter().position(|segments| path.matches(segments)) {
            Some(seed) => {
                found[seed] = true;
                seeded(value, seeds[seed].1)
            }
            None => R::constant(value),
        },
    );
    match seeds.iter().zip(&found).find(|(_, found)| !**found) {
        Some(((path, _), _)) => Err(ParameterError(format!(
            "the parameters hold no real number at `{path}`"
        ))),
        None => Ok(lifted),
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::testing::numbers;

    /// Parameters with every form `Lift` follows: a field renamed, a choice, an optional
    /// number, a list of pairs, an enum renamed in snake case with each kind of variant,
    /// an untagged enum and a newtype.
    #[derive(Debug, Clone, PartialEq, Deserialize, Serialize, Lift)]
    struct Card<R: Real = f64> {
        #[serde(rename = "w")]
        width: R,
        level: u8,
        period: Option<R>,
        points: Vec<(R, R)>,
        shape: Shape<R>,
        either: Either<R>,
        wrapped: Wrapped<R>,
    }

    #[derive(Debug, Clone, PartialEq, Deserialize, Serialize, Lift)]
    #[serde(rename_all = "snake_case")]
    enum Shape<R: Real> {
        DirectCurrent(R),
        Sine { amplitude: R, phase: R },
        Pair(R, R),
        Off,
    }

    #[derive(Debug, Clone, PartialEq, Deserialize, Serialize, Lift)]
    #[serde(untagged)]
    enum Either<R: Real> {
        Number(R),
        Named { value: R },
    }

    #[derive(Debug, Clone, PartialEq, Deserialize, Serialize, Lift)]
    struct Wrapped<R: Real>(R);

    /// Builds test cards covering every variant of `Shape` and `Either`.
    fn cards() -> Vec<Card> {
        let shapes = [
            Shape::DirectCurrent(1.0),
            Shape::Sine {
                amplitude: 2.0,
                phase: 3.0,
            },
            Shape::Pair(4.0, 5.0),
            Shape::Off,
        ];
        let eithers = [Either::Number(6.0), Either::Named { value: 7.0 }];
        shapes
            .into_iter()
            .zip(eithers.into_iter().cycle())
            .map(|(shape, either)| Card {
                width: 8.0,
                level: 2,
                period: Some(9.0),
                points: vec![(0.0, 1.0), (2.0, 3.0)],
                shape,
                either,
                wrapped: Wrapped(10.0),
            })
            .collect()
    }

    #[test]
    fn each_real_number_is_lifted_at_the_path_serde_writes_it_at() {
        for card in cards() {
            let mut lifted = Vec::new();
            let round_tripped: Card<f64> =
                Card::lift(&card, &ParameterPath::Root, &mut |path, value| {
                    lifted.push(path.to_string());
                    value
                });
            assert_eq!(round_tripped, card);
            let constant: Card<Gradient> = constant(&card);
            assert_eq!(constant.width, Gradient::constant(card.width));
            let mut written = Vec::new();
            numbers(
                &serde_json::to_value(&card).expect("JSON"),
                &mut Vec::new(),
                &mut written,
            );
            let mut written: Vec<String> = written
                .iter()
                .map(|path| path.join("."))
                .filter(|path| path != "level")
                .collect();
            lifted.sort();
            written.sort();
            assert_eq!(lifted, written, "{card:?}");
        }
    }

    #[test]
    fn a_seed_moves_the_number_at_its_path_and_no_other() {
        let card = cards().swap_remove(0);
        let lifted: Card<Gradient<2>> = lift(
            &card,
            &[("points.1.0", [1.0, 0.0]), ("w", [0.0, 2.0])],
            Gradient::new,
        )
        .expect("real numbers");
        assert_eq!(lifted.points[1].0, Gradient::new(2.0, [1.0, 0.0]));
        assert_eq!(lifted.width, Gradient::new(8.0, [0.0, 2.0]));
        assert_eq!(lifted.points[1].1, Gradient::constant(3.0));
        assert_eq!(lifted.period, Some(Gradient::constant(9.0)));

        let plain = BTreeMap::from([("gain".to_owned(), 2.0), ("offset".to_owned(), -0.5)]);
        let mapped: BTreeMap<String, Gradient<2>> = lift(
            &plain,
            &[("gain", [1.0, 0.0]), ("offset", [0.0, 3.0])],
            Gradient::new,
        )
        .expect("map entries");
        assert_eq!(mapped["gain"], Gradient::new(2.0, [1.0, 0.0]));
        assert_eq!(mapped["offset"], Gradient::new(-0.5, [0.0, 3.0]));
    }

    #[test]
    fn a_seed_at_no_real_number_fails() {
        let card = cards().swap_remove(0);
        for path in [
            "level",
            "width",
            "points.2.0",
            "shape.sine.amplitude",
            ".w",
            "points.+0.0",
            "points.00.0",
        ] {
            let lifted = lift::<Card<Gradient>, _, 1>(&card, &[(path, [1.0])], Gradient::new);
            assert!(lifted.is_err(), "{path}");
        }
    }

    #[test]
    fn a_path_seeded_once_moves_along_its_rates() {
        let card = cards().swap_remove(0);
        let lifted: Card<Gradient<2>> =
            lift(&card, &[("w", [1.25, 1.5])], Gradient::new).expect("a real number at w");
        assert_eq!(lifted.width, Gradient::new(8.0, [1.25, 1.5]));
    }

    #[test]
    fn parameter_path_rejects_leading_dots_and_non_canonical_indices() {
        let root = ParameterPath::Root;
        let single = root.key("r");
        let outer = root.key("a");
        let nested = outer.key("b");
        let points = root.key("points");
        let indexed = points.index(0);
        let root_index = root.index(0);

        assert!(root.is(""));
        assert!(!root.is("."));
        assert!(single.is("r"));
        assert!(!single.is(".r"));
        assert!(!single.is("r."));
        assert!(nested.is("a.b"));
        assert!(!nested.is(".a.b"));
        assert!(indexed.is("points.0"));
        assert!(!indexed.is(".points.0"));
        assert!(!indexed.is("points.+0"));
        assert!(!indexed.is("points.00"));
        assert!(root_index.is("0"));
        assert!(!root_index.is(".0"));
        assert!(!root_index.is("+0"));
        assert!(!root_index.is("00"));
    }

    #[derive(Debug, Clone, PartialEq, Deserialize, Serialize, Lift)]
    struct CollidingFields<R: Real = f64> {
        plain: R,
        path: R,
        number: R,
    }

    #[derive(Debug, Clone, PartialEq, Deserialize, Serialize, Lift)]
    #[serde(rename_all = "snake_case")]
    enum RenamedVariant<R: Real = f64> {
        #[serde(rename_all = "lowercase")]
        MixedCase {
            #[expect(
                non_snake_case,
                reason = "verifies lowercase renaming on mixed-case variant fields"
            )]
            UpperGain: R,
            path: R,
        },
    }

    #[test]
    fn derive_lift_avoids_parameter_shadowing_and_follows_variant_rename_all() {
        let colliding = CollidingFields {
            plain: 1.0,
            path: 2.0,
            number: 3.0,
        };
        let lifted: CollidingFields<Gradient<3>> = lift(
            &colliding,
            &[
                ("plain", [1.0, 0.0, 0.0]),
                ("path", [0.0, 2.0, 0.0]),
                ("number", [0.0, 0.0, 3.0]),
            ],
            Gradient::new,
        )
        .expect("hygienic fields");
        assert_eq!(lifted.plain, Gradient::new(1.0, [1.0, 0.0, 0.0]));
        assert_eq!(lifted.path, Gradient::new(2.0, [0.0, 2.0, 0.0]));
        assert_eq!(lifted.number, Gradient::new(3.0, [0.0, 0.0, 3.0]));

        let variant = RenamedVariant::MixedCase {
            UpperGain: 4.0,
            path: 5.0,
        };
        let mut lifted_paths = Vec::new();
        let _: RenamedVariant<f64> =
            RenamedVariant::lift(&variant, &ParameterPath::Root, &mut |path, value| {
                lifted_paths.push(path.to_string());
                value
            });
        let mut written = Vec::new();
        numbers(
            &serde_json::to_value(&variant).expect("JSON"),
            &mut Vec::new(),
            &mut written,
        );
        let mut written_paths: Vec<String> = written.iter().map(|path| path.join(".")).collect();
        lifted_paths.sort();
        written_paths.sort();
        assert_eq!(lifted_paths, written_paths);
        assert_eq!(
            lifted_paths,
            vec!["mixed_case.UpperGain", "mixed_case.path"]
        );
    }
}
