//! The preferred values of IEC 60063: the E series, which space values evenly on a log
//! scale, E24 with 24 a decade for tolerances of 5 %, to E192 for 0.5 % and better.

use std::fmt;
use std::sync::LazyLock;

/// Relative matching tolerance for comparing a normalized mantissa against an IEC 60063
/// preferred value.
///
/// Origin: IEEE 754 floating-point rounding margin (`1e-9`) covering decimal scaling via
/// `log10` and `10^k`, well below the tightest E192 step spacing (~1.2%).
const MATCH_TOLERANCE: f64 = 1e-9;

/// Historical IEC 60063 E3 mantissas in `[1, 10)`.
const E3_VALUES: &[f64] = &[1.0, 2.2, 4.7];

/// Historical IEC 60063 E6 mantissas in `[1, 10)`.
const E6_VALUES: &[f64] = &[1.0, 1.5, 2.2, 3.3, 4.7, 6.8];

/// Historical IEC 60063 E12 mantissas in `[1, 10)`.
const E12_VALUES: &[f64] = &[1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2];

/// Historical IEC 60063 E24 mantissas in `[1, 10)`.
const E24_VALUES: &[f64] = &[
    1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6,
    6.2, 6.8, 7.5, 8.2, 9.1,
];

/// Cached IEC 60063 E48 mantissas in `[1, 10)`.
static E48_VALUES: LazyLock<Vec<f64>> = LazyLock::new(|| compute_mantissas(48, false));

/// Cached IEC 60063 E96 mantissas in `[1, 10)`.
static E96_VALUES: LazyLock<Vec<f64>> = LazyLock::new(|| compute_mantissas(96, false));

/// Cached IEC 60063 E192 mantissas in `[1, 10)`.
static E192_VALUES: LazyLock<Vec<f64>> = LazyLock::new(|| compute_mantissas(192, true));

/// Computes the rounded three-digit IEC 60063 mantissas for a geometric series of `count`
/// steps per decade, overriding step 185 to `9.2` when `patch_185` is set (E192).
fn compute_mantissas(count: u32, patch_185: bool) -> Vec<f64> {
    (0..count)
        .map(|step| {
            if patch_185 && step == 185 {
                // Where IEC 60063 departs from rounding: 9.20, not 9.19.
                9.2
            } else {
                (100.0 * 10_f64.powf(f64::from(step) / f64::from(count))).round() / 100.0
            }
        })
        .collect()
}

/// An E series of preferred values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Series {
    /// 3 values a decade.
    E3,
    /// 6 values a decade.
    E6,
    /// 12 values a decade.
    E12,
    /// 24 values a decade.
    E24,
    /// 48 values a decade.
    E48,
    /// 96 values a decade.
    E96,
    /// 192 values a decade.
    E192,
}

impl Series {
    /// All IEC 60063 preferred-number E series, ordered from sparsest to densest.
    pub const ALL: [Self; 7] = [
        Self::E3,
        Self::E6,
        Self::E12,
        Self::E24,
        Self::E48,
        Self::E96,
        Self::E192,
    ];

    /// Returns the number of preferred values per decade in this series.
    #[must_use]
    pub const fn count(self) -> u32 {
        match self {
            Self::E3 => 3,
            Self::E6 => 6,
            Self::E12 => 12,
            Self::E24 => 24,
            Self::E48 => 48,
            Self::E96 => 96,
            Self::E192 => 192,
        }
    }

    /// Parses a series identifier such as `"E24"` into a [`Series`].
    #[must_use]
    pub const fn named(name: &str) -> Option<Self> {
        match name.as_bytes() {
            [b'E', b'3'] => Some(Self::E3),
            [b'E', b'6'] => Some(Self::E6),
            [b'E', b'1', b'2'] => Some(Self::E12),
            [b'E', b'2', b'4'] => Some(Self::E24),
            [b'E', b'4', b'8'] => Some(Self::E48),
            [b'E', b'9', b'6'] => Some(Self::E96),
            [b'E', b'1', b'9', b'2'] => Some(Self::E192),
            _ => None,
        }
    }

    /// Returns the cached values of the series in the decade from 1 to 10.
    fn values(self) -> &'static [f64] {
        match self {
            Self::E3 => E3_VALUES,
            Self::E6 => E6_VALUES,
            Self::E12 => E12_VALUES,
            Self::E24 => E24_VALUES,
            Self::E48 => E48_VALUES.as_slice(),
            Self::E96 => E96_VALUES.as_slice(),
            Self::E192 => E192_VALUES.as_slice(),
        }
    }

    /// Returns the series mantissas in the decade `[1, 10)`.
    ///
    /// E3 through E24 use the historical IEC 60063 values; E48 through E192 round powers of
    /// 10 to three significant digits, except for the E192 entry `9.20`.
    #[must_use]
    pub fn mantissas(self) -> Vec<f64> {
        self.values().to_vec()
    }

    /// Returns whether finite positive `value` matches a preferred value of this series within
    /// rounding precision.
    #[must_use]
    pub fn holds(self, value: f64) -> bool {
        if !value.is_finite() || value <= 0.0 {
            return false;
        }
        let decade = value.log10().floor();
        let mantissa = value / 10_f64.powf(decade);
        self.values()
            .iter()
            .copied()
            .chain([10.0])
            .any(|preferred| (mantissa - preferred).abs() <= MATCH_TOLERANCE * preferred)
    }
}

impl fmt::Display for Series {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E{}", self.count())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_series_hold_their_standard_values() {
        let lengths: Vec<usize> = Series::ALL
            .iter()
            .map(|series| series.mantissas().len())
            .collect();
        assert_eq!(lengths, [3, 6, 12, 24, 48, 96, 192]);
        assert_eq!(&Series::E96.mantissas()[..4], [1.0, 1.02, 1.05, 1.07]);
        assert!((Series::E192.mantissas()[185] - 9.2).abs() < 1e-12);
        let e96 = Series::E96.mantissas();
        let every_other: Vec<f64> = e96.iter().step_by(2).copied().collect();
        assert_eq!(Series::E48.mantissas(), every_other);
        assert!(Series::E24.holds(4.7e3) && Series::E24.holds(1e-9) && Series::E24.holds(10.0));
        assert!(!Series::E24.holds(4.8e3) && Series::E192.holds(4.87e3));
        assert_eq!(Series::named("E12"), Some(Series::E12));
        assert_eq!(Series::named("E13"), None);
    }

    #[test]
    fn e24_and_e192_cover_every_series() {
        for series in [Series::E3, Series::E6, Series::E12, Series::E24] {
            for value in series.mantissas() {
                assert!(
                    Series::E24.holds(value),
                    "{series} value {value} not in E24"
                );
            }
        }
        for series in [Series::E48, Series::E96, Series::E192] {
            for value in series.mantissas() {
                assert!(
                    Series::E192.holds(value),
                    "{series} value {value} not in E192"
                );
            }
        }
        for series in Series::ALL {
            for value in series.mantissas() {
                assert!(
                    Series::E24.holds(value) || Series::E192.holds(value),
                    "{series} value {value} missing from E24 union E192"
                );
            }
        }
    }

    #[test]
    fn holds_rejects_non_positive_and_non_finite_values() {
        for bad in [
            0.0,
            -0.0,
            -4.7e3,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            for series in Series::ALL {
                assert!(!series.holds(bad), "{series} accepted {bad:?}");
            }
        }
    }

    #[test]
    fn named_round_trips_every_series_in_const_context() {
        /// Compile-time check that `Series::named` evaluates in `const` context for IEC 60063
        /// E24.
        const CHECK_E24: Option<Series> = Series::named("E24");
        assert_eq!(CHECK_E24, Some(Series::E24));
        for series in Series::ALL {
            assert_eq!(Series::named(&series.to_string()), Some(series));
        }
        for unknown in ["", "e24", "E0", "E13", "E192 "] {
            assert_eq!(Series::named(unknown), None);
        }
    }
}
