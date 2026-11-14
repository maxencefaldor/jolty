//! numpy's arithmetic, operation for operation, as jolty's measurements were first written in
//! numpy.
//!
//! Each function names the numpy or Python function it follows, as numpy 2.5 and Python 3.15
//! define them, so a measure reads the same bits those did on the same platform. Where numpy's
//! compiler fuses a multiplication and an addition into one rounding, as clang does on arm64
//! and as numpy's SIMD loops do on every processor with a fused multiply-add, the function does
//! too, with [`f64::mul_add`], which rounds once on every platform.

use std::f64::consts::PI;

use jolty_numerics::Complex64;

use super::{Edge, MeasureError, Point, SumError};

/// How many values numpy's search of an interpolation's points looks around its guess before it
/// bisects, `LIKELY_IN_CACHE_SIZE` in numpy's `compiled_base.c`.
const LIKELY_IN_CACHE: usize = 8;

/// Where numpy's search of increasing points places a key.
enum Found {
    /// Before the first point.
    Below,
    /// At the index `j` where `points[j] <= key < points[j + 1]`, or the last when the key is
    /// the last point.
    At(usize),
    /// After the last point.
    Above,
}

/// numpy's `binary_search_with_guess` of `key` among `points`, at least five, from `guess`.
fn search(key: f64, points: &[f64], guess: usize) -> Found {
    let len = points.len();
    if key > points[len - 1] {
        return Found::Above;
    }
    if key < points[0] {
        return Found::Below;
    }
    if len <= 4 {
        let mut index = 1;
        while index < len && key >= points[index] {
            index += 1;
        }
        return Found::At(index - 1);
    }
    let guess = guess.clamp(1, len - 3);
    let (mut low, mut high) = (0, len);
    if key < points[guess] {
        if key >= points[guess - 1] {
            return Found::At(guess - 1);
        }
        high = guess - 1;
        if guess > LIKELY_IN_CACHE && key >= points[guess - LIKELY_IN_CACHE] {
            low = guess - LIKELY_IN_CACHE;
        }
    } else if key < points[guess + 1] {
        return Found::At(guess);
    } else if key < points[guess + 2] {
        return Found::At(guess + 1);
    } else {
        low = guess + 2;
        if guess + LIKELY_IN_CACHE + 1 < len && key < points[guess + LIKELY_IN_CACHE] {
            high = guess + LIKELY_IN_CACHE;
        }
    }
    while low < high {
        let middle = low + ((high - low) >> 1);
        if key >= points[middle] {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    // The bisection only raises `low` from where it starts, 1 or more, or from 0 past a point
    // the key is not below, so it ends at 1 or more.
    Found::At(low.saturating_sub(1))
}

/// numpy's `interp(at, points, values)`: `values`, given at `points`, interpolated linearly at
/// each of `at`, and the first or the last value past either end.
///
/// It follows `arr_interp` of numpy's `compiled_base.c`: each of `at` is searched for from
/// where the last was found, so points out of order give what numpy gives; a point of `at`
/// that is not a number gives itself; and the interpolation rounds once, as numpy's does on
/// arm64, where clang fuses its `slope * (x - points[j]) + values[j]`. `points` and `values`
/// are as long as each other; with none, every value is not a number.
pub(crate) fn interpolate(at: &[f64], points: &[f64], values: &[f64]) -> Vec<f64> {
    let len = points.len().min(values.len());
    let (points, values) = (&points[..len], &values[..len]);
    let (Some(&left), Some(&right)) = (values.first(), values.last()) else {
        return vec![f64::NAN; at.len()];
    };
    if len == 1 {
        let (point, value) = (points[0], values[0]);
        return at
            .iter()
            .map(|&x| {
                if x < point {
                    left
                } else if x > point {
                    right
                } else {
                    value
                }
            })
            .collect();
    }
    let mut guess = 0;
    at.iter()
        .map(|&x| {
            if x.is_nan() {
                return x;
            }
            match search(x, points, guess) {
                Found::Below => {
                    guess = 0;
                    left
                }
                Found::Above => {
                    guess = len;
                    right
                }
                Found::At(found) => {
                    guess = found;
                    if found == len - 1 || same(points[found], x) {
                        return values[found];
                    }
                    let slope =
                        (values[found + 1] - values[found]) / (points[found + 1] - points[found]);
                    let mut value = slope.mul_add(x - points[found], values[found]);
                    if value.is_nan() {
                        value = slope.mul_add(x - points[found + 1], values[found + 1]);
                        if value.is_nan() && same(values[found], values[found + 1]) {
                            value = values[found];
                        }
                    }
                    value
                }
            }
        })
        .collect()
}

/// Whether `a` and `b` are the same number, as `==` compares them: the comparison numpy's and
/// Python's arithmetic makes where these functions follow it.
#[expect(
    clippy::float_cmp,
    reason = "numpy and Python compare these floats exactly, and so must the functions that \
              follow them"
)]
pub(crate) fn same(a: f64, b: f64) -> bool {
    a == b
}

/// numpy's `abs` of a complex number, as its SIMD loop `simd_cabsolute` computes it: the
/// larger part's magnitude times the square root of one plus the square of the smaller's over
/// it, by a fused multiply-add; infinite when a part is, else not a number when one is.
pub(crate) fn magnitude(z: Complex64) -> f64 {
    let (mut real, mut imaginary) = (z.re.abs(), z.im.abs());
    let (real_infinite, imaginary_infinite) = (real == f64::INFINITY, imaginary == f64::INFINITY);
    if real_infinite {
        imaginary = f64::INFINITY;
    }
    if imaginary_infinite {
        real = f64::INFINITY;
    }
    let (real_nan, imaginary_nan) = (real.is_nan(), imaginary.is_nan());
    if real_nan {
        imaginary = f64::NAN;
    }
    if imaginary_nan {
        real = f64::NAN;
    }
    let larger = real.max(imaginary);
    let smaller = imaginary.min(real);
    let ratio = if larger == 0.0 || smaller == f64::INFINITY {
        0.0
    } else {
        smaller / larger
    };
    ratio.mul_add(ratio, 1.0).sqrt() * larger
}

/// numpy's complex division of `a` by `b`, Smith's algorithm as its `CDOUBLE_divide` writes
/// it, each sum of a product rounded once, as clang fuses them on arm64.
pub(crate) fn divide(a: Complex64, b: Complex64) -> Complex64 {
    let (real, imaginary) = (b.re.abs(), b.im.abs());
    if real >= imaginary {
        if real == 0.0 && imaginary == 0.0 {
            return Complex64::new(a.re / real, a.im / imaginary);
        }
        let ratio = b.im / b.re;
        let scale = 1.0 / b.im.mul_add(ratio, b.re);
        Complex64::new(
            a.im.mul_add(ratio, a.re) * scale,
            (-a.re).mul_add(ratio, a.im) * scale,
        )
    } else {
        let ratio = b.re / b.im;
        let scale = 1.0 / b.re.mul_add(ratio, b.im);
        Complex64::new(
            a.re.mul_add(ratio, a.im) * scale,
            a.im.mul_add(ratio, -a.re) * scale,
        )
    }
}

/// numpy's `log10` of the magnitude of `z`, in decibels: minus infinity where it vanishes.
pub(crate) fn decibels(z: Complex64) -> f64 {
    20.0 * magnitude(z).log10()
}

/// numpy's `degrees(unwrap(angle(transfer)))`: the phase of each of `transfer`, in degrees,
/// unwrapped from the first.
pub(crate) fn phases(transfer: &[Complex64]) -> Vec<f64> {
    let angles: Vec<f64> = transfer.iter().map(|z| z.im.atan2(z.re)).collect();
    unwrap(&angles)
        .into_iter()
        .map(|angle| angle * (180.0 / PI))
        .collect()
}

/// numpy's `unwrap` of `angles`, in radians, with its default period, 2π: each step between
/// two angles of more than half a period, taken back into the half periods about zero, and
/// the corrections summed in order from the first.
pub(crate) fn unwrap(angles: &[f64]) -> Vec<f64> {
    let period = 2.0 * PI;
    let high = period / 2.0;
    let low = -high;
    let Some(&first) = angles.first() else {
        return Vec::new();
    };
    let mut unwrapped = Vec::with_capacity(angles.len());
    unwrapped.push(first);
    let mut total: Option<f64> = None;
    for pair in angles.windows(2) {
        let step = pair[1] - pair[0];
        let mut wrapped = remainder(step - low, period) + low;
        if same(wrapped, low) && step > 0.0 {
            wrapped = high;
        }
        let correction = if step.abs() < high {
            0.0
        } else {
            wrapped - step
        };
        // numpy's cumulative sum starts from the first correction itself, not from zero.
        let sum = total.map_or(correction, |sum| sum + correction);
        total = Some(sum);
        unwrapped.push(pair[1] + sum);
    }
    unwrapped
}

/// numpy's `remainder(a, b)` of two floats, `npy_remainder`: the remainder of `a` over `b`
/// with the sign of `b`, as Python's `%` gives it.
fn remainder(a: f64, b: f64) -> f64 {
    let modulus = a % b;
    if b == 0.0 {
        return modulus;
    }
    if modulus == 0.0 {
        0.0_f64.copysign(b)
    } else if (b < 0.0) == (modulus < 0.0) {
        modulus
    } else {
        modulus + b
    }
}

/// numpy's `sum` of `count` terms, `term(index)` for each index in order: zero plus the
/// terms' pairwise sum in numpy's order, `pairwise_sum` of its `loops_utils.h.src`.
pub(crate) fn sum(count: usize, term: impl Fn(usize) -> f64) -> f64 {
    0.0 + pairwise(0, count, &term)
}

/// The size of the blocks numpy's pairwise sum adds with eight accumulators,
/// `PW_BLOCKSIZE` in numpy's `loops_utils.h.src`.
const BLOCK: usize = 128;

/// numpy's pairwise sum of the `count` terms from `start`: blocks of up to [`BLOCK`] summed by
/// eight accumulators, each half of a longer run summed so, then the halves added.
fn pairwise(start: usize, count: usize, term: &impl Fn(usize) -> f64) -> f64 {
    if count < 8 {
        // numpy starts from minus zero, so that a sum of minus zeros stays minus zero.
        let mut total = -0.0;
        for index in start..start + count {
            total += term(index);
        }
        total
    } else if count <= BLOCK {
        let mut lanes = [0.0; 8];
        for (lane, value) in lanes.iter_mut().enumerate() {
            *value = term(start + lane);
        }
        let whole = count - count % 8;
        let mut index = 8;
        while index < whole {
            for (lane, value) in lanes.iter_mut().enumerate() {
                *value += term(start + index + lane);
            }
            index += 8;
        }
        let mut total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        while index < count {
            total += term(start + index);
            index += 1;
        }
        total
    } else {
        let half = count / 2;
        let half = half - half % 8;
        pairwise(start, half, term) + pairwise(start + half, count - half, term)
    }
}

/// Python's `math.fsum` of `values`: their sum correctly rounded, by Shewchuk's exact partial
/// sums, rounded half to even across them, as Python's own `math_fsum` computes it.
///
/// # Errors
///
/// Returns [`SumError::Overflow`] when a partial sum of finite values overflows, and
/// [`SumError::Indeterminate`] when the values hold both infinities, as `math.fsum` raises.
pub(crate) fn exact_sum(values: impl IntoIterator<Item = f64>) -> Result<f64, SumError> {
    let mut partials: Vec<f64> = Vec::new();
    let (mut special, mut infinite) = (0.0, 0.0);
    for value in values {
        let mut x = value;
        let mut kept = 0;
        for index in 0..partials.len() {
            let mut y = partials[index];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let high = x + y;
            let low = y - (high - x);
            if low != 0.0 {
                partials[kept] = low;
                kept += 1;
            }
            x = high;
        }
        partials.truncate(kept);
        if x != 0.0 {
            if x.is_finite() {
                partials.push(x);
            } else {
                if value.is_finite() {
                    return Err(SumError::Overflow);
                }
                if value.is_infinite() {
                    infinite += value;
                }
                special += value;
                partials.clear();
            }
        }
    }
    if special != 0.0 {
        return if infinite.is_nan() {
            Err(SumError::Indeterminate)
        } else {
            Ok(special)
        };
    }
    let Some(mut high) = partials.pop() else {
        return Ok(0.0);
    };
    let mut low = 0.0;
    while let Some(y) = partials.pop() {
        let x = high;
        high = x + y;
        low = y - (high - x);
        if low != 0.0 {
            break;
        }
    }
    // Rounds half to even across the partials: a remainder of the same sign as the next
    // partial pushes a tie the way the exact sum lies.
    if let Some(&next) = partials.last()
        && ((low < 0.0 && next < 0.0) || (low > 0.0 && next > 0.0))
    {
        let twice = low * 2.0;
        let rounded = high + twice;
        if same(twice, rounded - high) {
            high = rounded;
        }
    }
    Ok(high)
}

/// numpy's `max` of `values`: not a number if any is, and plus zero above minus zero, as its
/// SIMD reduction gives them; not a number when there are none.
pub(crate) fn maximum(values: &[f64]) -> f64 {
    extreme(values, |value, best| {
        value > best || (same(value, best) && best.is_sign_negative())
    })
}

/// numpy's `min` of `values`: not a number if any is, and minus zero below plus zero, as its
/// SIMD reduction gives them; not a number when there are none.
pub(crate) fn minimum(values: &[f64]) -> f64 {
    extreme(values, |value, best| {
        value < best || (same(value, best) && value.is_sign_negative())
    })
}

/// The value of `values` that `beats` every other, or not a number if any is.
fn extreme(values: &[f64], beats: impl Fn(f64, f64) -> bool) -> f64 {
    let mut best = f64::NAN;
    for (index, &value) in values.iter().enumerate() {
        if value.is_nan() {
            return value;
        }
        if index == 0 || beats(value, best) {
            best = value;
        }
    }
    best
}

/// numpy's `trapezoid(values, points)`: the integral of `values` over `points` by the
/// trapezoidal rule, each trapezoid's area as numpy writes it, summed in numpy's order.
pub(crate) fn trapezoid(points: &[f64], values: &[f64]) -> f64 {
    let count = points.len().min(values.len()).saturating_sub(1);
    sum(count, |index| {
        (points[index + 1] - points[index]) * (values[index + 1] + values[index]) / 2.0
    })
}

/// Where `values`, given at `points`, cross `level` the way `edge` says, in order, each
/// interpolated linearly between the two points around it, as numpy computed it for jolty's
/// `_crossings`. A value that reaches the level crosses it only once it passes it.
pub(crate) fn crossings<'a>(
    points: &'a [f64],
    values: &'a [f64],
    level: f64,
    edge: Edge,
) -> impl Iterator<Item = f64> + 'a {
    points
        .windows(2)
        .zip(values.windows(2))
        .filter_map(move |(around, pair)| {
            let (below, above) = (pair[0] - level, pair[1] - level);
            let rising = below < 0.0 && above >= 0.0;
            let falling = below > 0.0 && above <= 0.0;
            let crossed = match edge {
                Edge::Rising => rising,
                Edge::Falling => falling,
                Edge::Either => rising || falling,
            };
            crossed.then(|| {
                let fraction = below / (below - above);
                around[0] + fraction * (around[1] - around[0])
            })
        })
}

/// The order that sorts `values` increasingly, the values that are not numbers last, equal
/// values in their own order, as numpy's `argsort` orders distinct values.
pub(crate) fn order(values: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| {
        let (a, b) = (values[a], values[b]);
        a.partial_cmp(&b)
            .unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()))
    });
    order
}

/// `frequencies` sorted increasingly, and `densities` in their order.
///
/// # Errors
///
/// Returns [`MeasureError::Empty`] when there is no frequency, and
/// [`MeasureError::Mismatch`] when the two are not as long as each other.
fn sorted(frequencies: &[f64], densities: &[f64]) -> Result<(Vec<f64>, Vec<f64>), MeasureError> {
    if frequencies.len() != densities.len() {
        return Err(MeasureError::Mismatch);
    }
    if frequencies.is_empty() {
        return Err(MeasureError::Empty);
    }
    let order = order(frequencies);
    Ok((
        order.iter().map(|&index| frequencies[index]).collect(),
        order.iter().map(|&index| densities[index]).collect(),
    ))
}

/// The noise density at the frequency `at`, a power law between the frequencies around it, as
/// jolty's `NoiseSpectrum.density` computed it.
///
/// # Errors
///
/// Returns [`MeasureError::Outside`] for the point `at` when it lies outside the frequencies,
/// [`MeasureError::Empty`] when there are none, and [`MeasureError::Mismatch`] when there are
/// not as many densities.
pub fn density(frequencies: &[f64], densities: &[f64], at: f64) -> Result<f64, MeasureError> {
    let (points, values) = sorted(frequencies, densities)?;
    let (first, last) = (points[0], points[points.len() - 1]);
    if !(first <= at && at <= last) {
        return Err(MeasureError::Outside {
            point: Point::At,
            first,
            last,
        });
    }
    Ok(power_law(&points, &values, at))
}

/// The integral of `densities` over `frequencies` from `start` to `stop`, by default the
/// lowest and the highest frequency, each density a power law between frequencies, as SPICE
/// integrates noise and jolty's `integrate` computed it.
///
/// # Errors
///
/// Returns [`MeasureError::Band`] when the band does not lie within the frequencies or is
/// empty, [`MeasureError::Sum`] when `math.fsum` would raise on its segments,
/// [`MeasureError::Empty`] when there is no frequency, and [`MeasureError::Mismatch`] when
/// there are not as many densities.
pub fn integrate(
    frequencies: &[f64],
    densities: &[f64],
    start: Option<f64>,
    stop: Option<f64>,
) -> Result<f64, MeasureError> {
    let (points, values) = sorted(frequencies, densities)?;
    let (first, last) = (points[0], points[points.len() - 1]);
    let low = start.unwrap_or(first);
    let high = stop.unwrap_or(last);
    if !(first <= low && low < high && high <= last) {
        return Err(MeasureError::Band {
            low,
            high,
            first,
            last,
        });
    }
    let inside = (0..points.len()).filter(|&index| points[index] > low && points[index] < high);
    let mut band = vec![low];
    let mut levels = vec![power_law(&points, &values, low)];
    for index in inside {
        band.push(points[index]);
        levels.push(values[index]);
    }
    band.push(high);
    levels.push(power_law(&points, &values, high));
    exact_sum(
        band.windows(2)
            .zip(levels.windows(2))
            .map(|(ends, densities)| segment(ends[0], ends[1], densities[0], densities[1])),
    )
    .map_err(MeasureError::Sum)
}

/// The density at `at`, a power law between the two of the increasing `points` around it,
/// linear when either density is not positive, as jolty's `_power_law` computed it.
fn power_law(points: &[f64], values: &[f64], at: f64) -> f64 {
    let count = points.len();
    // numpy's `clip` to 1 and the last index, which gives 0 for a single point.
    let index = points
        .partition_point(|&point| point < at)
        .max(1)
        .min(count - 1);
    let before = index.checked_sub(1).unwrap_or(count - 1);
    let (first, last) = (points[before], points[index]);
    let (low, high) = (values[before], values[index]);
    if same(at, first) {
        return low;
    }
    if same(at, last) {
        return high;
    }
    if low <= 0.0 || high <= 0.0 {
        return low + (high - low) * (at - first) / (last - first);
    }
    low * ((high / low).ln() * (at / first).ln() / (last / first).ln()).exp()
}

/// The integral from `first` to `last` of the power law from `low` to `high`, as jolty's
/// `_segment` computed it.
fn segment(first: f64, last: f64, low: f64, high: f64) -> f64 {
    if low <= 0.0 || high <= 0.0 {
        #[expect(
            clippy::manual_midpoint,
            reason = "jolty's `0.5 * (low + high)`, which overflows where `f64::midpoint` would not"
        )]
        let mean = 0.5 * (low + high);
        return mean * (last - first);
    }
    let span = (last / first).ln();
    // With the density ∝ f^a, the integral is low·first·(e^{(a+1)·span} - 1)/(a + 1).
    let growth = (high / low).ln() + span;
    let factor = if growth == 0.0 {
        1.0
    } else {
        growth.exp_m1() / growth
    };
    low * first * span * factor
}

/// The noise densities `densities` referred to an input through `gains`, each over the
/// square of the gain's magnitude, as numpy computed jolty's `NoiseSpectrum.referred`:
/// infinite where a gain vanishes. As long as the shorter of the two.
#[must_use]
pub fn referred(densities: &[f64], gains: &[Complex64]) -> Vec<f64> {
    densities
        .iter()
        .zip(gains)
        .map(|(density, &gain)| {
            let magnitude = magnitude(gain);
            density / (magnitude * magnitude)
        })
        .collect()
}

/// The damping ratio of each of `roots`, minus its real part over its magnitude: 1 for a real
/// root in the left half-plane, between 0 and 1 for a conjugate pair that rings, negative in
/// the right half-plane, and not a number at the origin.
#[must_use]
pub fn dampings(roots: &[Complex64]) -> Vec<f64> {
    roots
        .iter()
        .map(|&root| -root.re / magnitude(root))
        .collect()
}

/// The natural frequency of each of `roots`, its magnitude over 2π, in hertz.
#[must_use]
pub fn natural_frequencies(roots: &[Complex64]) -> Vec<f64> {
    roots
        .iter()
        .map(|&root| magnitude(root) / (2.0 * PI))
        .collect()
}

#[cfg(test)]
mod tests {
    //! Each expected value is what numpy 2.5.4 or Python 3.15's `math` computes for the same
    //! input on an Apple M2 Max, printed by `repr`, which round-trips every bit.

    use super::*;

    #[test]
    fn sums_in_numpys_pairwise_order() {
        let count = 1001;
        let harmonic = |index: usize| 1.0 / f64::from(u32::try_from(index + 1).unwrap());
        // `(1.0 / np.arange(1, 1002)).sum()`; a sum in order gives 7.486469861549344, and
        // `math.fsum` 7.486469861549346.
        assert_eq!(
            sum(count, harmonic).to_bits(),
            7.486_469_861_549_347_f64.to_bits()
        );
        // numpy keeps the sign of a sum of minus zeros only within a block, then adds zero.
        assert!(sum(3, |_| -0.0).is_sign_positive());
    }

    #[test]
    fn sums_exactly_as_math_fsum_does() {
        assert_eq!(
            exact_sum([1e-16, 1.0, 1e16]),
            Ok(1.000_000_000_000_000_2e16)
        );
        assert_eq!(exact_sum([0.1; 10]), Ok(1.0));
        assert_eq!(exact_sum([]), Ok(0.0));
        assert_eq!(exact_sum([1e308, 1e308]), Err(SumError::Overflow));
        assert_eq!(
            exact_sum([f64::INFINITY, -f64::INFINITY]),
            Err(SumError::Indeterminate)
        );
        assert!(exact_sum([1.0, f64::NAN]).unwrap().is_nan());
    }

    #[test]
    fn interpolates_as_numpy_does_even_out_of_order() {
        // `np.interp(x, xp, fp)` over points out of order, searched from each last one found.
        let points = [0.0, 1.0, 0.5, 2.0, 3.0, 1.5, 4.0];
        let values = [0.0, 10.0, -5.0, 3.0, 7.0, 1.0, 2.0];
        let at = [0.25, 1.75, 3.5, 0.75, 2.5, 5.0, -1.0, 1.2];
        let expected: [f64; 8] = [
            2.5,
            1.666_666_666_666_666_3,
            1.8,
            7.5,
            1.4,
            2.0,
            0.0,
            -1.266_666_666_666_667,
        ];
        let found = interpolate(&at, &points, &values);
        assert_eq!(
            found
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        // One rounding, as numpy's on arm64: two would give 0.44285714285714284.
        assert_eq!(
            interpolate(&[0.3], &[0.0, 0.7], &[0.1, 0.9]),
            [0.442_857_142_857_142_9]
        );
        assert!(
            interpolate(&[0.5], &[], &[])
                .iter()
                .all(|value| value.is_nan())
        );
    }

    #[test]
    fn magnitudes_and_phases_are_numpys() {
        let magnitudes = [
            Complex64::new(3.0, 4.0),
            Complex64::new(1e-300, 1e-300),
            Complex64::new(-2.5, 0.1),
            Complex64::new(0.0, 0.0),
        ]
        .map(magnitude);
        assert_eq!(
            magnitudes,
            [
                5.0,
                1.414_213_562_373_095_2e-300,
                2.501_999_200_639_360_8,
                0.0
            ]
        );
        assert_eq!(
            magnitude(Complex64::new(f64::INFINITY, f64::NAN)),
            f64::INFINITY
        );
        assert!(magnitude(Complex64::new(1.0, f64::NAN)).is_nan());
        let unwrapped = unwrap(&[0.0, 3.0, -3.0, 0.5, 6.5, -6.0]);
        assert_eq!(
            unwrapped,
            [
                0.0,
                3.0,
                3.283_185_307_179_586_2,
                0.5,
                0.216_814_692_820_413_77,
                0.283_185_307_179_586_23
            ]
        );
    }

    #[test]
    fn integrates_by_the_trapezoidal_rule_in_numpys_order() {
        let points: Vec<f64> = (0..200).map(|index| f64::from(index) * 0.05).collect();
        let values: Vec<f64> = (0..200)
            .map(|index| (f64::from(index) * 0.1).sin())
            .collect();
        // `np.trapezoid(np.sin(np.arange(200) * 0.1), np.arange(200) * 0.05)`.
        assert_eq!(trapezoid(&points, &values), 0.251_197_561_719_626_15);
    }

    #[test]
    fn extremes_propagate_what_is_not_a_number() {
        assert!(maximum(&[1.0, f64::NAN, 2.0]).is_nan());
        assert!(minimum(&[f64::NAN, 1.0]).is_nan());
        assert!(maximum(&[-0.0, 0.0]).is_sign_positive());
        assert!(minimum(&[0.0, -0.0]).is_sign_negative());
        assert!(maximum(&[]).is_nan());
    }
}
