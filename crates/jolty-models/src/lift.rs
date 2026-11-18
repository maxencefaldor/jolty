//! How a twin's parameters are made of a kernel's plain ones: each real number carried
//! onto the twin's kind of number by the path it is written at, the numbers its seeds name
//! seeded with their rates and every other one constant, as JAX's `tree_map_with_path`
//! maps a tree's leaves by their paths.
//!
//! [`Lift`] is derived, `#[derive(Lift)]`, for every type of parameters generic over its
//! kind of number, so that a type's fields are walked by code written for that type alone,
//! once for every kind of number, never through the parameters' written form.

use std::fmt;

use crate::{Gradient, Hyper, ParameterError, Real};

/// The derive of [`Lift`](trait@Lift), for a struct or an enum generic over one kind of
/// number.
pub use jolty_derive::Lift;

/// A value whose real numbers are `R`s, made of its plain form, whose numbers are `f64`s,
/// each made an `R` by the path it is written at.
pub trait Lift<R>: Sized {
    /// The value's plain form.
    type Plain;

    /// The value made of `plain`, written at `path`, each real number within it made an
    /// `R` by `number` of its path and its value.
    fn lift<Lifting: FnMut(&ParameterPath<'_>, f64) -> R>(
        plain: &Self::Plain,
        path: &ParameterPath<'_>,
        number: &mut Lifting,
    ) -> Self;
}

/// Where a value is written among a device's parameters: under a key, or at an index of a
/// list, within where its container is written, each step on the stack of the walk that
/// reaches it, as `serde_path_to_error` tracks a path. It writes as `jolty_ir::parameter`
/// reads it, its keys and indices joined by dots, such as `waveform.dc`.
#[derive(Debug, Clone, Copy, Default)]
pub enum ParameterPath<'a> {
    /// The parameters themselves.
    #[default]
    Root,
    /// Under `key`, within `within`.
    Key {
        /// Where the container is written.
        within: &'a ParameterPath<'a>,
        /// The key.
        key: &'static str,
    },
    /// At `index` of a list, within `within`.
    Index {
        /// Where the list is written.
        within: &'a ParameterPath<'a>,
        /// The index.
        index: usize,
    },
}

impl ParameterPath<'_> {
    /// Under `key`, within this path.
    #[must_use]
    pub const fn key(&self, key: &'static str) -> ParameterPath<'_> {
        ParameterPath::Key { within: self, key }
    }

    /// At `index` of the list at this path.
    #[must_use]
    pub const fn index(&self, index: usize) -> ParameterPath<'_> {
        ParameterPath::Index {
            within: self,
            index,
        }
    }

    /// Whether this is the path `written`, its keys and indices joined by dots.
    #[must_use]
    pub fn is(&self, written: &str) -> bool {
        match self {
            Self::Root => written.is_empty(),
            Self::Key {
                within: Self::Root,
                key,
            } => written == *key,
            Self::Key { within, key } => written
                .rsplit_once('.')
                .is_some_and(|(before, last)| last == *key && within.is(before)),
            Self::Index {
                within: Self::Root,
                index,
            } => is_canonical_index(written, *index),
            Self::Index { within, index } => {
                written.rsplit_once('.').is_some_and(|(before, last)| {
                    is_canonical_index(last, *index) && within.is(before)
                })
            }
        }
    }
}

/// Whether `text` is the canonical decimal form of `index`, with no sign or leading zeros.
fn is_canonical_index(text: &str, index: usize) -> bool {
    !text.starts_with('+')
        && (text == "0" || !text.starts_with('0'))
        && text.parse::<usize>() == Ok(index)
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

/// A kind of number made of a plain number by `number`, at its own path.
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

/// The parameters `P` made of the plain `parameters`, every number constant.
pub(crate) fn constant<P: Lift<R>, R: Real>(parameters: &P::Plain) -> P {
    P::lift(parameters, &ParameterPath::Root, &mut |_, value| {
        R::constant(value)
    })
}

/// The parameters `P` of a twin made of the plain `parameters`, each number at the path of
/// one of `seeds` made by `seeded` of its value and its rate along each seed, a path named
/// more than once moving at the sum of its rates, and every other number constant.
///
/// # Errors
///
/// Returns [`ParameterError`] when no real number is at a seed's path, as when the number
/// there is read as a choice, such as a level.
pub(crate) fn lift<P: Lift<R>, R: Real, const K: usize>(
    parameters: &P::Plain,
    seeds: &[(&str, [f64; K])],
    seeded: impl Fn(f64, [f64; K]) -> R,
) -> Result<P, ParameterError> {
    let mut read = vec![false; seeds.len()];
    let lifted = P::lift(parameters, &ParameterPath::Root, &mut |path, value| {
        let mut combined: Option<[f64; K]> = None;
        for ((seed, rates), seen) in seeds.iter().zip(&mut read) {
            if path.is(seed) {
                *seen = true;
                match &mut combined {
                    Some(sum) => {
                        for (total, &rate) in sum.iter_mut().zip(rates) {
                            *total += rate;
                        }
                    }
                    None => combined = Some(*rates),
                }
            }
        }
        match combined {
            Some(rates) => seeded(value, rates),
            None => R::constant(value),
        }
    });
    match seeds.iter().zip(&read).find(|(_, read)| !**read) {
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

    /// Cards whose shapes take each variant of `Shape` and `Either`.
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
    fn duplicate_seed_paths_sum_their_rates() {
        let card = cards().swap_remove(0);
        let lifted: Card<Gradient<2>> = lift(
            &card,
            &[("w", [1.0, -0.5]), ("w", [0.25, 2.0])],
            Gradient::new,
        )
        .expect("duplicate paths sum");
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
