//! The preferred values of IEC 60063: the E series, which space values evenly on a log
//! scale, E24 with 24 a decade for tolerances of 5 %, to E192 for 0.5 % and better.

use std::fmt;

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
    /// Every series, from the sparsest.
    pub const ALL: [Self; 7] = [
        Self::E3,
        Self::E6,
        Self::E12,
        Self::E24,
        Self::E48,
        Self::E96,
        Self::E192,
    ];

    /// How many values the series has a decade.
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

    /// The series named `name`, such as `E24`.
    #[must_use]
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|series| series.to_string() == name)
    }

    /// The values of the series in the decade from 1 to 10.
    ///
    /// E3 to E24 are IEC 60063's historical values; E48 to E192 round 10 to the power of
    /// each step to three significant digits, but for E192's 9.20.
    #[must_use]
    pub fn mantissas(self) -> Vec<f64> {
        let historical: &[f64] = match self {
            Self::E3 => &[1.0, 2.2, 4.7],
            Self::E6 => &[1.0, 1.5, 2.2, 3.3, 4.7, 6.8],
            Self::E12 => &[1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2],
            Self::E24 => &[
                1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.3,
                4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1,
            ],
            Self::E48 | Self::E96 | Self::E192 => &[],
        };
        if !historical.is_empty() {
            return historical.to_vec();
        }
        let count = self.count();
        (0..count)
            .map(|step| {
                if self == Self::E192 && step == 185 {
                    // Where IEC 60063 departs from rounding: 9.20, not 9.19.
                    9.2
                } else {
                    (100.0 * 10_f64.powf(f64::from(step) / f64::from(count))).round() / 100.0
                }
            })
            .collect()
    }

    /// Whether `value`, positive, is one of the series' values, but for rounding.
    #[must_use]
    pub fn holds(self, value: f64) -> bool {
        let decade = value.log10().floor();
        let mantissa = value / 10_f64.powf(decade);
        self.mantissas()
            .into_iter()
            .chain([10.0])
            .any(|preferred| (mantissa - preferred).abs() <= 1e-9 * preferred)
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
}
