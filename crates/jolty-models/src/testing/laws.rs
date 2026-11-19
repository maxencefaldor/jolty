//! The laws every kernel keeps, tested over the whole library: at each case of every
//! analog model ([`cases::CASES`]) and at probes around it, so that a kernel added to the
//! library is held to them by its case alone.
//!
//! - **What flows in flows out:** over an instance's electrical nodes, its currents sum to
//!   zero, and so do its charges and the derivatives of both.
//! - **Mirrors:** a p-channel or PNP device mirrors the corresponding n-channel or NPN
//!   device bit for bit with every voltage and current reversed.
//! - **Symmetry:** a transistor that claims its drain and its source alike swaps them,
//!   within rounding, which is Gummel's symmetry test to the first order.
//! - **Continuity:** a transistor's currents and charges and their derivatives are
//!   continuous where a margin marks a boundary between two of its regions.
//! - **Self-heating:** every model that supports `thermal: true` matches its isothermal
//!   model at `ambient + rise` in residuals, charges, Jacobians and dissipated power, and
//!   differentiates exactly in both its unknowns and its parameters.
//! - **Parameter gradients:** every analog model with real parameters computes exact
//!   analytic parameter gradients matching central finite differences.
//! - **Finite values:** at any finite bias, every number a kernel computes is finite.

use serde_json::{Value, json};

use super::{
    Bench, Evaluation, assert_close, assert_model_jacobians_match,
    assert_model_parameter_gradients_match, cases,
};
use crate::{
    Channel, Input, Library, Model, Nature, Polarity, REFERENCE_TEMPERATURE, thermal_voltage,
};

/// A gmin across every junction, as SPICE's default places one, so that its currents are
/// held to the laws too.
const GMIN: f64 = 1e-12;

/// Values of a local unknown at which every kernel is evaluated at once, its steps not
/// limited: zero, a nanovolt either side of it, and an operating point's biases.
const NEAR: [f64; 7] = [0.0, 1e-9, -1e-9, 0.5, -0.5, 5.0, -5.0];

/// Values far from any operating point, at which a kernel is evaluated as Newton's
/// iteration evaluates it, each step limited from the last.
const FAR: [f64; 6] = [50.0, -50.0, 1e3, -1e3, 1e6, -1e6];

/// Transistor parameters whose sign follows device polarity, matching SPICE threshold
/// conventions (for example, n-channel `vto` of 0.8 V corresponds to p-channel -0.8 V).
/// The mirrored device flips polarity, negates every listed parameter, and leaves all
/// remaining parameters unchanged.
const SIGNED: &[(&str, &[&str])] = &[
    ("bipolar_transistor", &[]),
    (
        "bsim3",
        &["vth0", "lvth0", "wvth0", "pvth0", "threshold_adjustment"],
    ),
    (
        "bsim4",
        &["vth0", "lvth0", "wvth0", "pvth0", "threshold_adjustment"],
    ),
    ("jfet", &[]),
    ("mosfet", &["threshold_voltage", "threshold_adjustment"]),
    ("vbic", &[]),
    (
        "vdmos",
        &[
            "threshold_voltage",
            "threshold_voltage_temperature_coefficient",
        ],
    ),
];

/// A device alike at two of its terminals: its parameters, the two terminals' names, and
/// the values of its first local unknowns.
type Alike = (&'static str, [&'static str; 2], &'static [f64]);

/// Symmetric transistor test cases paired with the two interchangeable terminal names,
/// following Gummel's symmetry test (McAndrew, "Validation of MOSFET model source-drain
/// symmetry", IEEE Transactions on Electron Devices 53, 2006). Each entry maps a model
/// name to symmetric parameters and terminal values, or `None` when the model claims no
/// symmetry.
const SYMMETRIES: &[(&str, Option<Alike>)] = &[
    ("bipolar_transistor", None),
    // BSIM3 swaps its currents, charges (`Q_d == Q_s`), and conductances across all bias
    // points, but its source-referenced capacitances are not smooth through `vds = 0` in
    // general (`b3ld.c`):
    // - `capmod` 0 is symmetric at `vds = 0` in accumulation and depletion for `vbs <= 0`,
    //   including `vbs = 0` where `VbseffCV` is C1 smooth with slope 1. Under forward body
    //   bias (`vb = 0.1 V`) capacitances differ by `19.2%` of max `|C|`. In strong
    //   inversion (`vg = 3 V, vb = 0 V`), the gate row stays symmetric while the bulk and
    //   drain/source rows jump by `14.1%`.
    // - `capmod` 1 jumps by `1.54x` of max `|C|` at `[0, 0, 0, 0]` (`1.47x` on `10u x 1u`
    //   at `vg = 1.2 V`) because its linear branch stays active at `vds = 0`.
    // - `capmod` 2 and 3 jump by `1.39e-4` and `1.38e-4` at `[0, 0, 0, 0]` because
    //   `VbseffCV = Vbseff` has slope `|vbsc| / (|vbsc| + 0.001) = 1 - 1.485e-4` at
    //   `vbs = 0`, and by `~30%` in strong inversion.
    ("bsim3", None),
    // BSIM4's charges, written against the source as BSIM3's are, are not smooth through
    // `vds = 0` either.
    ("bsim4", None),
    (
        "jfet",
        Some((
            r#"{"channel_length_modulation": 0.02, "gate_source_capacitance": 1e-12,
                "gate_drain_capacitance": 1e-12}"#,
            ["drain", "source"],
            &[3.0, -0.5, 0.0],
        )),
    ),
    // Both level 1 and level 3 MOSFETs swap drain and source cleanly at `vds != 0` and at
    // `vds = 0` when displacement currents are zero (`w = 0`). When `w != 0` inside
    // `-phi < vgst <= 0` at `vds = 0`, Meyer's capacitances (`DEVqmeyer`, `devsup.c` lines
    // 698 to 720) introduce a conductance jump of
    // `|w| / phi * (1 - d(von)/d(vbs) + 2 * sigma)`. For default `vto = 0` at both levels
    // 1 and 3, the jump measures `1.67e-9` at `[1e-9; 7]`. With `threshold_voltage: 0.8`
    // (above `phi`), all `vds = 0` probes stay in deep accumulation or strong inversion;
    // level 3 is checked with the same test below.
    (
        "mosfet",
        Some((
            r#"{"threshold_voltage": 0.8, "body_effect": 0.5, "channel_length_modulation": 0.02,
                "oxide_thickness": 2e-8, "drain_area": 2e-11, "source_area": 2e-11,
                "gate_source_overlap": 3e-10, "gate_drain_overlap": 3e-10}"#,
            ["drain", "source"],
            &[3.0, 2.0, 0.0, -0.5],
        )),
    ),
    ("vbic", None),
    // ngspice's VDMOS reads the drain-source voltage with its sign in its channel-length
    // modulation, and its body diode joins the source to the drain alone.
    ("vdmos", None),
];

/// Merges the JSON object `extra` into `base`, replacing any keys they share.
fn merge(base: &str, extra: &str) -> String {
    let mut merged: Value = serde_json::from_str(base).expect("a JSON object");
    let added: Value = serde_json::from_str(extra).expect("a JSON object");
    if let (Some(target), Some(source)) = (merged.as_object_mut(), added.as_object()) {
        for (key, value) in source {
            target.insert(key.clone(), value.clone());
        }
    }
    merged.to_string()
}

/// Removes `key` from the JSON object `base`.
fn without(base: &str, key: &str) -> String {
    let mut value: Value = serde_json::from_str(base).expect("a JSON object");
    if let Some(object) = value.as_object_mut() {
        object.remove(key);
    }
    value.to_string()
}

/// Builds a test bench for the standard library model `name` with `parameters`.
fn bench(name: &str, parameters: &str) -> Bench {
    let library = Library::standard();
    let model = library.get(name).expect("a model of the library");
    Bench::from_model(&**model, parameters, REFERENCE_TEMPERATURE)
        .unwrap_or_else(|error| panic!("{name}: {error}"))
}

/// Evaluates the bench's device at `x` with step limiting disabled.
fn exactly(bench: &mut Bench, x: &[f64]) -> Evaluation {
    let mut input = Input::new(x, None, GMIN);
    input.limiting = false;
    bench.evaluate_with(&input)
}

/// Builds probe points around `given` for local unknowns with `natures`. Includes
/// `given`, its negation, its half, each individual unknown set to each entry of
/// `values`, and all non-temperature unknowns set uniformly to each entry. Temperature
/// unknowns keep their original values because temperature is not an electrical bias.
fn probes(given: &[f64], natures: &[Nature], values: &[f64]) -> Vec<Vec<f64>> {
    let biased: Vec<usize> = (0..given.len())
        .filter(|&index| natures[index] != Nature::Temperature)
        .collect();
    let with = |change: &dyn Fn(usize) -> f64| {
        let mut x = given.to_vec();
        for &index in &biased {
            x[index] = change(index);
        }
        x
    };
    let mut points = vec![
        given.to_vec(),
        with(&|index| -given[index]),
        with(&|index| given[index] / 2.0),
    ];
    for &value in values {
        points.push(with(&|_| value));
        for &index in &biased {
            let mut x = given.to_vec();
            x[index] = value;
            points.push(x);
        }
    }
    points
}

/// Reports whether all voltage differences in `x` stay within `ln(f64::MAX)` (709.78)
/// thermal voltages at the reference temperature. Beyond that span, junction
/// exponentials with emission coefficient at least one overflow when unlimited. SPICE
/// relies on junction limiting (`DEVpnjlim`, ngspice 47 `devsup.c`, lines 49 to 84) so
/// Newton iteration never evaluates unbounded steps.
fn within_junctions_reach(x: &[f64], natures: &[Nature]) -> bool {
    let reach = f64::MAX.ln() * thermal_voltage(REFERENCE_TEMPERATURE);
    let voltages = x
        .iter()
        .zip(natures)
        .filter(|&(_, &nature)| nature == Nature::Voltage)
        .map(|(&value, _)| value);
    let (low, high) = voltages.fold((0.0_f64, 0.0_f64), |(low, high), value| {
        (low.min(value), high.max(value))
    });
    high - low <= reach
}

/// Checks that currents, charges, and each Jacobian column of `bench`'s device sum to
/// zero over electrical nodes at `x`.
///
/// Every kernel stamps each current and charge into one node row and its negation into
/// another, making the exact node sum zero. Floating-point row sums stay within
/// `(k - 1) · u` of the row's `k` term magnitudes to first order, where `u` is half of
/// `f64::EPSILON` (Higham, "Accuracy and Stability of Numerical Algorithms", 2002,
/// §4.2). Summing across rows adds a matching rounding contribution. With at most one
/// term per local unknown `n`, the total sum lies within `n² · ε` of the largest row
/// terms ([`terms`]).
fn assert_conserves(name: &str, bench: &mut Bench, x: &[f64]) {
    let natures = bench.natures();
    let unknowns = natures.len();
    let nodes: Vec<usize> = (0..unknowns)
        .filter(|&index| natures[index] == Nature::Voltage)
        .collect();
    let n = f64::from(u32::try_from(unknowns).expect("few unknowns"));
    let evaluation = exactly(bench, x);
    let largest = terms(&evaluation, x);
    for (kind, values) in kinds(&evaluation).iter().enumerate() {
        let columns = values.len() / unknowns;
        for column in 0..columns {
            let sum: f64 = nodes
                .iter()
                .map(|&row| values[row * columns + column])
                .sum();
            let magnitude: f64 = nodes.iter().map(|&row| largest[kind][row]).sum();
            assert!(
                sum.abs() <= n * n * f64::EPSILON * magnitude,
                "{name}'s {} sum to {sum:e} over its nodes at {x:?}, of {magnitude:e}",
                KINDS[kind]
            );
        }
    }
}

/// The kinds of value an evaluation holds, in the order of [`kinds`].
const KINDS: [&str; 4] = ["residuals", "charges", "conductances", "capacitances"];

/// Returns the evaluation values for each entry in [`KINDS`], flattening matrices row by row.
fn kinds(evaluation: &Evaluation) -> [Vec<f64>; 4] {
    [
        evaluation.residual.clone(),
        evaluation.charge.clone(),
        evaluation.conductance.concat(),
        evaluation.capacitance.concat(),
    ]
}

/// Estimates the largest term magnitude per row for each entry of [`KINDS`] in
/// `evaluation` at `x`. Uses the kernel's reported scale for residuals, and computes
/// first-order bounds for charges and derivatives because kernels omit per-term scales
/// for them. Charge bounds combine the charge magnitude with capacitance times voltage,
/// while Jacobian row bounds use the widest entry magnitude across the row.
fn terms(evaluation: &Evaluation, x: &[f64]) -> [Vec<f64>; 4] {
    let linear = |row: &[f64]| -> f64 { row.iter().zip(x).map(|(c, v)| (c * v).abs()).sum() };
    let widest = |row: &[f64]| {
        row.iter()
            .fold(0.0_f64, |most, entry| most.max(entry.abs()))
    };
    let charges = evaluation.charge.iter().zip(&evaluation.capacitance);
    [
        evaluation.scale.clone(),
        charges
            .map(|(charge, row)| charge.abs() + linear(row))
            .collect(),
        evaluation
            .conductance
            .iter()
            .map(|row| widest(row))
            .collect(),
        evaluation
            .capacitance
            .iter()
            .map(|row| widest(row))
            .collect(),
    ]
}

/// Returns the polarity parameter name and its two mirrored JSON variant names for
/// `model`, matching either [`Channel`] or [`Polarity`], or `None` when the model has
/// no polarity parameter.
fn polarity(model: &dyn Model) -> Option<(String, [String; 2])> {
    let schema = model.schema();
    let properties = schema.get("properties")?.as_object()?;
    let written =
        |variants: [Value; 2]| variants.map(|variant| variant.as_str().expect("a name").to_owned());
    let kinds = [
        (
            crate::device::schema::<Channel>(),
            written([json!(Channel::N), json!(Channel::P)]),
        ),
        (
            crate::device::schema::<Polarity>(),
            written([json!(Polarity::Npn), json!(Polarity::Pnp)]),
        ),
    ];
    properties.iter().find_map(|(name, property)| {
        kinds.iter().find_map(|(kind, variants)| {
            let choices = property.get("oneOf");
            (choices.is_some() && choices == kind.get("oneOf"))
                .then(|| (name.clone(), variants.clone()))
        })
    })
}

/// Returns the mirrored parameter JSON for transistor `name` with `parameters`,
/// flipping the polarity property and negating every polarity-signed parameter listed in
/// [`SIGNED`].
fn mirrored(name: &str, (property, variants): &(String, [String; 2]), parameters: &str) -> String {
    let mut mirror: Value = serde_json::from_str(parameters).expect("a JSON object");
    let own = mirror
        .get(property)
        .and_then(Value::as_str)
        .unwrap_or(&variants[0])
        .to_owned();
    let other = if own == variants[0] {
        &variants[1]
    } else {
        &variants[0]
    };
    mirror[property] = json!(other);
    let (_, signed) = SIGNED
        .iter()
        .find(|&&(model, _)| model == name)
        .unwrap_or_else(|| panic!("{name}'s signed parameters, in `SIGNED`"));
    for &parameter in *signed {
        if let Some(value) = mirror.get(parameter).and_then(Value::as_f64) {
            mirror[parameter] = json!(-value);
        }
    }
    mirror.to_string()
}

/// Checks that `image` (the mirrored device evaluated at reflected `x`) exactly equals
/// `evaluation` reflected by `signs`. Verifies that residuals and charges negate on
/// rows with sign `-1`, while conductances and capacitances negate where row and column
/// signs differ. Regions and margins match verbatim in n-channel/NPN orientation, and
/// operating variables match with `threshold_voltage` and `saturation_voltage` negated.
#[expect(
    clippy::float_cmp,
    reason = "a mirror is exact: a negation rounds nothing"
)]
fn assert_reflects(
    (device, mirror): (&Bench, &Bench),
    (evaluation, image): (&Evaluation, &Evaluation),
    (signs, x): (&[f64], &[f64]),
    what: &str,
) {
    let unknowns = device.unknowns();
    for (row, (&name, &sign)) in unknowns.iter().zip(signs).enumerate() {
        for (kind, own, other) in [
            ("residual", &evaluation.residual, &image.residual),
            ("charge", &evaluation.charge, &image.charge),
        ] {
            assert!(
                own[row] == sign * other[row],
                "{what}: the {kind} of {name} {:e}, mirrored {:e}, at {x:?}",
                own[row],
                other[row]
            );
        }
        for (column, (&by, &by_sign)) in unknowns.iter().zip(signs).enumerate() {
            for (kind, own, other) in [
                ("conductance", &evaluation.conductance, &image.conductance),
                ("capacitance", &evaluation.capacitance, &image.capacitance),
            ] {
                assert!(
                    own[row][column] == sign * by_sign * other[row][column],
                    "{what}: the {kind} of {name} by {by} {:e}, mirrored {:e}, at {x:?}",
                    own[row][column],
                    other[row][column]
                );
            }
        }
    }
    assert_eq!(
        evaluation.limited, image.limited,
        "{what}: limiting at {x:?}"
    );
    if !device.regions.names.is_empty() {
        assert_eq!(device.region(), mirror.region(), "{what}: region at {x:?}");
    }
    let variables = device.variables().into_iter().zip(mirror.variables());
    for (&name, (own, other)) in device.variable_names().iter().zip(variables) {
        let sign = if matches!(name, "threshold_voltage" | "saturation_voltage") {
            -1.0
        } else {
            1.0
        };
        assert!(
            own == sign * other,
            "{what}: the variable {name} {own:e}, mirrored {other:e}, at {x:?}"
        );
    }
}

/// Checks that `model` with `parameters` mirrors bit for bit across opposite polarity
/// around `given`. Flipping polarity, negating signed parameters, and reversing all
/// non-temperature unknowns reverses residuals and charges while preserving derivative
/// orientations, regions, and margins. Verifies both unlimited evaluations across probes
/// around `given` and the settled Newton operating point.
fn assert_mirrors(model: &dyn Model, parameters: &str, given: &[f64]) {
    let name = model.name();
    let polar = polarity(model).unwrap_or_else(|| panic!("{name} has no polarity"));
    let mirror = mirrored(name, &polar, parameters);
    let make = |parameters: &str| {
        Bench::from_model(model, parameters, REFERENCE_TEMPERATURE)
            .unwrap_or_else(|error| panic!("{name}: {error}"))
    };
    let (mut device, mut image) = (make(parameters), make(&mirror));
    let natures = device.natures();
    assert_eq!(natures, image.natures(), "{name}'s mirror's unknowns");
    let signs: Vec<f64> = natures
        .iter()
        .map(|&nature| {
            if nature == Nature::Temperature {
                1.0
            } else {
                -1.0
            }
        })
        .collect();
    let reflect = |x: &[f64]| -> Vec<f64> { x.iter().zip(&signs).map(|(v, s)| v * s).collect() };
    let x = cases::values(given, natures.len());
    let what = format!("{name} {parameters} and its mirror");
    for point in probes(&x, &natures, &NEAR) {
        if within_junctions_reach(&point, &natures) {
            let evaluations = (
                exactly(&mut device, &point),
                exactly(&mut image, &reflect(&point)),
            );
            let pair = (&evaluations.0, &evaluations.1);
            assert_reflects((&device, &image), pair, (&signs, &point), &what);
        }
    }
    let evaluations = (device.settle(&x), image.settle(&reflect(&x)));
    let pair = (&evaluations.0, &evaluations.1);
    let what = format!("{what}, settled");
    assert_reflects((&device, &image), pair, (&signs, &x), &what);
}

/// Permutes `evaluation`'s rows and columns by `swap`.
fn permute(evaluation: &Evaluation, swap: &[usize]) -> Evaluation {
    let rows = |values: &[f64]| swap.iter().map(|&row| values[row]).collect();
    let matrix = |values: &[Vec<f64>]| {
        let entries = |row: &Vec<f64>| swap.iter().map(|&column| row[column]).collect();
        swap.iter().map(|&row| entries(&values[row])).collect()
    };
    Evaluation {
        residual: rows(&evaluation.residual),
        scale: rows(&evaluation.scale),
        charge: rows(&evaluation.charge),
        conductance: matrix(&evaluation.conductance),
        capacitance: matrix(&evaluation.capacitance),
        limited: evaluation.limited,
    }
}

/// Checks that `bench`'s device evaluated unlimited at `x` is symmetric under `swap`,
/// within [`ROUNDING`].
pub(crate) fn assert_symmetric_at(bench: &mut Bench, x: &[f64], swap: &[usize], what: &str) {
    let own = exactly(bench, x);
    let swapped = permute(&own, swap);
    assert_agree(what, [&own, &swapped], x, true);
}

/// Checks that `model` with `parameters` is symmetric when exchanging the two terminals
/// named in `pair`. At every probe around `given`, exchanging the two terminals and any
/// internal unknowns whose names swap those labels produces matching residuals, charges,
/// and derivatives within [`ROUNDING`].
///
/// Equality holds within rounding rather than bit for bit because the swapped device
/// references voltages against the opposite terminal. SPICE transistor equations also
/// compute in the lower-source frame and reconstruct swapped derivatives via sums, as in
/// `b3ld.c`.
fn assert_swaps(model: &dyn Model, parameters: &str, pair: [&str; 2], given: &[f64]) {
    let name = model.name();
    let mut bench = Bench::from_model(model, parameters, REFERENCE_TEMPERATURE)
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    let specification = model
        .read(&serde_json::from_str(parameters).expect("a JSON object"))
        .expect("valid parameters");
    let terminals = specification.interface().terminals;
    let labels: Vec<String> = bench
        .unknowns()
        .iter()
        .enumerate()
        .map(|(index, &unknown)| {
            terminals
                .get(index)
                .map_or(unknown, String::as_str)
                .to_owned()
        })
        .collect();
    let [one, other] = pair;
    let swap: Vec<usize> = labels
        .iter()
        .map(|label| {
            let wanted = label
                .replace(one, "\u{0}")
                .replace(other, one)
                .replace('\u{0}', other);
            labels
                .iter()
                .position(|candidate| *candidate == wanted)
                .unwrap_or_else(|| panic!("{name} has {label} but not {wanted}"))
        })
        .collect();
    let natures = bench.natures();
    let x = cases::values(given, natures.len());
    for point in probes(&x, &natures, &NEAR) {
        if !within_junctions_reach(&point, &natures) {
            continue;
        }
        let exchanged: Vec<f64> = swap.iter().map(|&index| point[index]).collect();
        let (own, image) = (exactly(&mut bench, &point), exactly(&mut bench, &exchanged));
        let swapped = permute(&image, &swap);
        let what = format!("{name} {parameters} with its {one} and {other} swapped");
        assert_agree(&what, [&own, &swapped], &point, true);
    }
}

/// Rounding tolerance in multiples of `f64::EPSILON` times the largest term magnitude
/// per evaluation kind. Two evaluations at differently rounded voltages or adjacent
/// floating-point numbers agree within this bound, where continuous values move by at
/// most one ulp times their derivative.
///
/// A value computed in `m` operations stays within `γ(m) ≈ m · u` of its term
/// magnitude (Higham 2002, §3.1), and no kernel path exceeds a thousand operations.
const ROUNDING: f64 = 1024.0;

/// Checks continuity across every region boundary crossed when sweeping each
/// non-temperature unknown of `bench` from `-10` to `10` around `x`. At adjacent
/// floating-point numbers straddling each margin zero crossing, residuals, charges,
/// capacitances, and (when `slopes` is true) conductances agree within [`ROUNDING`].
fn assert_continuous(name: &str, bench: &mut Bench, x: &[f64], slopes: bool) {
    /// The steps of each sweep, 50 mV apart, a margin's sign found at each.
    const STEPS: u32 = 400;
    let natures = bench.natures();
    let margins = bench.regions.margins;
    let mut crossed = 0;
    for (unknown, &nature) in natures.iter().enumerate() {
        if nature == Nature::Temperature {
            continue;
        }
        let point = |value: f64| {
            let mut point = x.to_vec();
            point[unknown] = value;
            point
        };
        let margins_at = |bench: &mut Bench, value: f64| {
            exactly(bench, &point(value));
            bench.region().1
        };
        for (index, margin) in margins.iter().enumerate() {
            let mut last = (-10.0, margins_at(bench, -10.0)[index]);
            for step in 1..=STEPS {
                let value = -10.0 + 20.0 * f64::from(step) / f64::from(STEPS);
                let found = margins_at(bench, value)[index];
                if found.signum() != last.1.signum() {
                    let (mut low, mut high) = (last.0, value);
                    loop {
                        let middle = f64::midpoint(low, high);
                        if middle <= low || middle >= high {
                            break;
                        }
                        if margins_at(bench, middle)[index].signum() == last.1.signum() {
                            low = middle;
                        } else {
                            high = middle;
                        }
                    }
                    let (below, above) = (point(low), point(high));
                    let sides = (exactly(bench, &below), exactly(bench, &above));
                    let what = format!("{name} across {margin} = 0");
                    assert_agree(&what, [&sides.0, &sides.1], &below, slopes);
                    crossed += 1;
                }
                last = (value, found);
            }
        }
    }
    assert!(crossed > 0, "{name} crosses no boundary from {x:?}");
}

/// Checks that two evaluations of one device, the second at `x` or a point rounded
/// otherwise, agree within [`ROUNDING`] of the largest term of each of [`KINDS`] of value
/// ([`terms`]), its conductances unless `slopes` is false.
fn assert_agree(what: &str, [one, other]: [&Evaluation; 2], x: &[f64], slopes: bool) {
    let largest = [terms(one, x), terms(other, x)];
    let (one, other) = (kinds(one), kinds(other));
    for (kind, name) in KINDS.iter().enumerate() {
        if *name == "conductances" && !slopes {
            continue;
        }
        let magnitude = largest
            .iter()
            .flat_map(|terms| &terms[kind])
            .fold(0.0_f64, |most, term| most.max(*term));
        let apart = one[kind]
            .iter()
            .zip(&other[kind])
            .fold(0.0_f64, |most, (one, other)| most.max((one - other).abs()));
        assert!(
            apart <= ROUNDING * f64::EPSILON * magnitude,
            "{what}: the {name} differ by {apart:e}, of {magnitude:e}, at {x:?}"
        );
    }
}

/// Checks that every number of `evaluation`, `bench`'s device's at `x`, is finite.
fn assert_finite(what: &str, bench: &Bench, evaluation: &Evaluation, x: &[f64]) {
    let values = evaluation
        .residual
        .iter()
        .chain(&evaluation.charge)
        .chain(evaluation.conductance.iter().flatten())
        .chain(evaluation.capacitance.iter().flatten())
        .copied()
        .chain(bench.variables());
    assert!(
        values.into_iter().all(f64::is_finite),
        "{what} computes a number that is not finite at {x:?}"
    );
}

/// Checks that `model` with `cool` plus `"thermal": true` matches the isothermal `cool`
/// device at `REFERENCE_TEMPERATURE + rise` in residuals, charges, conductances,
/// capacitances, and dissipated power. Verifies that analytic Jacobians and parameter
/// gradients match finite differences at every bias and temperature rise, including the
/// mirrored polarity when `model` defines one.
fn assert_heats(model: &dyn Model, cool: &str, biases: &[&[f64]], rises: &[f64]) {
    let check = |cool: &str, biases: &[Vec<f64>]| {
        let heated = merge(cool, r#"{"thermal": true}"#);
        let mut bench = Bench::from_model(model, &heated, REFERENCE_TEMPERATURE)
            .unwrap_or_else(|error| panic!("{}: {error}", model.name()));
        let natures = bench.natures();
        let thermal = natures
            .iter()
            .position(|&nature| nature == Nature::Temperature)
            .unwrap_or_else(|| panic!("{} has no thermal terminal", model.name()));
        let with_rise = |x: &[f64], rise: f64| {
            let mut full = Vec::with_capacity(x.len() + 1);
            full.extend_from_slice(&x[..thermal]);
            full.push(rise);
            full.extend_from_slice(&x[thermal..]);
            full
        };
        let drop_thermal = |values: &[f64]| -> Vec<f64> {
            values
                .iter()
                .enumerate()
                .filter(|&(index, _)| index != thermal)
                .map(|(_, &value)| value)
                .collect()
        };
        let first = &biases[0];
        for rise in [0.0, 40.0] {
            let mut plain = Bench::from_model(model, cool, REFERENCE_TEMPERATURE + rise)
                .unwrap_or_else(|error| panic!("{}: {error}", model.name()));
            let hot_x = with_rise(first, rise);
            let hot = bench.settle(&hot_x);
            let reference = plain.settle(first);
            assert_close(&drop_thermal(&hot.residual), &reference.residual);
            assert_close(&drop_thermal(&hot.charge), &reference.charge);
            assert_close(&[hot.charge[thermal]], &[0.0]);
            for (hot_matrix, reference_matrix) in [
                (&hot.conductance, &reference.conductance),
                (&hot.capacitance, &reference.capacitance),
            ] {
                for (hot_row, reference_row) in hot_matrix
                    .iter()
                    .enumerate()
                    .filter(|&(index, _)| index != thermal)
                    .map(|(_, row)| row)
                    .zip(reference_matrix)
                {
                    assert_close(&drop_thermal(hot_row), reference_row);
                }
            }
            // Dissipated power, `Σ v · i` over electrical voltages, heats the thermal net.
            let power: f64 = hot_x
                .iter()
                .zip(&hot.residual)
                .zip(&natures)
                .filter(|&(_, &nature)| nature == Nature::Voltage)
                .map(|((&voltage, &current), _)| current * voltage)
                .sum();
            assert_close(&[-hot.residual[thermal]], &[power]);
        }
        for x in biases {
            for &rise in rises {
                let hot_x = with_rise(x, rise);
                assert_model_jacobians_match(model, &heated, &hot_x);
                assert_model_parameter_gradients_match(model, &heated, &hot_x, None);
            }
        }
    };
    let owned: Vec<Vec<f64>> = biases.iter().map(|x| x.to_vec()).collect();
    check(cool, &owned);
    if let Some(polar) = polarity(model) {
        let flipped = mirrored(model.name(), &polar, cool);
        let reflected: Vec<Vec<f64>> = biases
            .iter()
            .map(|x| x.iter().map(|value| -value).collect())
            .collect();
        check(&flipped, &reflected);
    }
}

mod tests {
    use super::{
        FAR, GMIN, Input, Library, NEAR, Nature, SIGNED, SYMMETRIES, assert_conserves,
        assert_continuous, assert_finite, assert_heats, assert_mirrors,
        assert_model_parameter_gradients_match, assert_swaps, bench, cases, exactly, merge,
        mirrored, polarity, probes, within_junctions_reach, without,
    };
    use crate::testing::{
        assert_lift_matches_serde, assert_lift_matches_serde_with_optionals,
        assert_unknown_derivatives_match,
    };

    /// Returns the name of every analog model in `library`.
    fn analog(library: &Library) -> Vec<&'static str> {
        library
            .names()
            .filter(|name| library.get(name).is_some())
            .collect()
    }

    /// Discrete floating-point mode selectors (`f64`, not `R`) whose JSON schema type is
    /// `"number"` because SPICE decks write fractional values (`0.0`, `0.4`, `0.5`, `1.0`).
    /// Models treat them as piecewise-constant switches without differentiating through
    /// them. Every entry is asserted to be present, non-integer in the schema, and omitted
    /// by `Lift`.
    const FLOAT_SELECTORS: &[(&str, &[&str])] = &[
        ("bsim3", &["charge_partition"]),
        ("bsim4", &["charge_partition"]),
    ];

    /// Verifies that every analog model's `Lift` visits the exact set of real parameter
    /// paths written by `serde`, with bit-for-bit identical values and no duplicates.
    /// Covers `CASES`, all `Option<R>` fields as `None` and `Some(...)`, every `Waveform`
    /// variant, multi-winding `coupled_inductors`, `thermal: true`, and own-temperature
    /// overrides.
    #[test]
    fn every_models_lift_visits_the_exact_paths_serde_writes() {
        let library = Library::standard();
        let analog_names = analog(&library);
        assert_eq!(
            cases::covered(),
            analog_names,
            "a case of every analog model, and no more"
        );
        for &(model_name, _) in FLOAT_SELECTORS {
            assert!(
                analog_names.contains(&model_name),
                "FLOAT_SELECTORS names unknown model `{model_name}`"
            );
        }
        let selectors_for = |name: &str| -> &[&str] {
            FLOAT_SELECTORS
                .iter()
                .find(|&&(model, _)| model == name)
                .map_or(&[], |&(_, selectors)| selectors)
        };
        let mut total_optional_populated = 0_usize;
        let mut checked_models = std::collections::BTreeSet::new();
        for &(name, parameters, _) in cases::CASES {
            let model = library.get(name).expect("a model of the library");
            if checked_models.insert(name) {
                total_optional_populated += assert_lift_matches_serde_with_optionals(
                    &**model,
                    parameters,
                    selectors_for(name),
                );
            } else {
                assert_lift_matches_serde(&**model, parameters, selectors_for(name));
            }
        }
        assert!(
            total_optional_populated > 100,
            "expected optional real parameters across BSIM/BJT/diode models, \
             got {total_optional_populated}"
        );

        let extra: &[(&str, &str)] = &[
            // Every Waveform variant, including pulse with period None and Some.
            ("voltage_source", r#"{"waveform": {"dc": 3.3}}"#),
            (
                "voltage_source",
                r#"{"waveform": {"pulse": {"initial": 0, "pulsed": 5, "delay": 1e-8,
                    "rise": 1e-9, "fall": 2e-9, "width": 1e-6}}}"#,
            ),
            (
                "voltage_source",
                r#"{"waveform": {"pulse": {"initial": 0, "pulsed": 5, "delay": 1e-8,
                    "rise": 1e-9, "fall": 2e-9, "width": 1e-6, "period": 1e-5}}}"#,
            ),
            (
                "voltage_source",
                r#"{"waveform": {"sine": {"offset": 1.2, "amplitude": 0.5, "frequency": 1e6,
                    "delay": 1e-7, "damping": 1e4, "phase": 0.25}}}"#,
            ),
            (
                "voltage_source",
                r#"{"waveform": {"piecewise": {"points": [[0, 0], [1e-6, 1.5], [2e-6, -0.5]]}}}"#,
            ),
            (
                "voltage_source",
                r#"{"waveform": {"exponential": {"initial": 0, "pulsed": 5,
                    "rise_delay": 1e-9, "rise_time_constant": 1e-8,
                    "fall_delay": 2e-7, "fall_time_constant": 3e-8}}}"#,
            ),
            // Multi-winding coupled inductors exercising Vec indices and integer winding pairs.
            (
                "coupled_inductors",
                r#"{"windings": [{"inductance": 1e-3}, {"inductance": 2e-3}, {"inductance": 3e-3}],
                    "couplings": [
                        {"windings": [1, 2], "coefficient": 0.8},
                        {"windings": [2, 3], "coefficient": 0.6}
                    ]}"#,
            ),
            // Self-heating and Surrounded own-temperature overrides.
            ("diode", r#"{"saturation_current": 2e-14, "thermal": true}"#),
            (
                "resistor",
                r#"{"resistance": 1000, "thermal": true, "temperature": 350}"#,
            ),
            (
                "mosfet",
                r#"{"level": 3, "channel": "p", "thermal": true, "temperature_offset": 15}"#,
            ),
            (
                "bipolar_transistor",
                r#"{"polarity": "pnp", "thermal": true}"#,
            ),
            ("jfet", r#"{"channel": "p", "thermal": true}"#),
            ("vbic", r#"{"polarity": "pnp", "thermal": true}"#),
            ("bsim3", r#"{"channel": "p", "thermal": true}"#),
            ("vdmos", r#"{"channel": "p", "thermal": true}"#),
            ("bsim4", r#"{"channel": "p", "thermal": true}"#),
        ];
        for &(name, parameters) in extra {
            let model = library.get(name).expect("a model of the library");
            assert_lift_matches_serde(&**model, parameters, selectors_for(name));
        }
    }

    #[test]
    fn every_kernel_conserves_charge() {
        for &(name, parameters, given) in cases::CASES {
            let mut bench = bench(name, parameters);
            let natures = bench.natures();
            let x = cases::values(given, natures.len());
            for point in probes(&x, &natures, &NEAR) {
                if within_junctions_reach(&point, &natures) {
                    assert_conserves(name, &mut bench, &point);
                }
            }
        }
    }

    /// Verifies that every kernel reads unknowns generically so twin dual numbers compute
    /// exact second derivatives, as AC operating-point sensitivity requires. Evaluates each
    /// case with internal unknowns spread apart and shifted on either side, avoiding points
    /// where terminals coincide and equation curvatures jump.
    #[test]
    fn every_kernel_differentiates_its_conductances_with_respect_to_its_unknowns() {
        let library = Library::standard();
        for &(name, parameters, given) in cases::CASES {
            let model = library.get(name).expect("a model of the library");
            let natures = bench(name, parameters).natures();
            let mut x = given.to_vec();
            for offset in 0..natures.len().saturating_sub(given.len()) {
                let step = u32::try_from(offset + 1).expect("few internal unknowns");
                x.push(cases::REST * f64::from(step));
            }
            let mut points = vec![x.clone()];
            for (index, nature) in natures.iter().enumerate() {
                if *nature != Nature::Temperature {
                    for step in [0.07, -0.07] {
                        let mut point = x.clone();
                        point[index] += step;
                        points.push(point);
                    }
                }
            }
            for point in points {
                assert_unknown_derivatives_match(&**model, parameters, &point);
            }
        }
    }

    /// Verifies that every region-reporting transistor defines a polarity parameter and
    /// that every polar model case mirrors bit for bit across opposite polarity.
    #[test]
    fn every_transistor_mirrors_its_opposite_polarity() {
        let library = Library::standard();
        let polar: Vec<&str> = analog(&library)
            .into_iter()
            .filter(|name| polarity(&**library.get(name).expect("a model")).is_some())
            .collect();
        for &(name, parameters, _) in cases::CASES {
            let regions = bench(name, parameters).regions;
            assert!(regions.names.is_empty() || polar.contains(&name), "{name}");
        }
        let signed: Vec<&str> = SIGNED.iter().map(|&(name, _)| name).collect();
        assert_eq!(signed, polar, "`SIGNED` lists every model with a polarity");
        for &(name, parameters) in SIGNED {
            let schema = library.get(name).expect("a model").schema();
            for parameter in parameters {
                let properties = schema.get("properties").expect("properties");
                assert!(properties.get(parameter).is_some(), "{name}: {parameter}");
            }
        }
        for &(name, parameters, given) in cases::CASES {
            if polar.contains(&name) {
                assert_mirrors(&**library.get(name).expect("a model"), parameters, given);
            }
        }
        // Cover every BSIM3 capacitance model and full short-channel BSIM3/BSIM4 cards.
        let model = library.get("bsim3").expect("bsim3 model");
        for capmod in 0..4 {
            let extra = format!(r#"{{"capacitance_model": {capmod}}}"#);
            let card = merge(BSIM3_SHORT, &extra);
            for x in [
                [2.0, 1.5, 0.0, -0.5],
                [-1.0, 2.0, 0.0, -1.5],
                [1.0, 0.3, 0.0, 0.2],
            ] {
                assert_mirrors(&**model, &card, &x);
            }
        }
        let model = library.get("bsim4").expect("bsim4 model");
        for x in [
            [1.5, 1.2, 0.0, -0.5],
            [-1.0, 1.5, 0.0, -1.5],
            [1.0, 0.3, 0.0, 0.2],
        ] {
            assert_mirrors(&**model, BSIM4_SHORT, &x);
        }
    }

    /// Verifies that every transistor claiming drain-source symmetry swaps those terminals
    /// across all probes. At equal drain and source voltages, swapped derivatives agree
    /// (first-order Gummel symmetry), ensuring drain current is odd in `vx` with even,
    /// continuous slope through `vx = 0`.
    #[test]
    fn every_symmetric_transistor_swaps_its_drain_and_source() {
        let library = Library::standard();
        let claims: Vec<&str> = SYMMETRIES.iter().map(|&(name, _)| name).collect();
        let transistors: Vec<&str> = SIGNED.iter().map(|&(name, _)| name).collect();
        assert_eq!(claims, transistors, "`SYMMETRIES` lists every transistor");
        for &(name, claim) in SYMMETRIES {
            if let Some((parameters, pair, given)) = claim {
                let model = library.get(name).expect("a model");
                assert_swaps(&**model, parameters, pair, given);
            }
        }
    }

    /// Verifies that symmetric level 3 MOSFETs (with and without velocity saturation) pass
    /// the complete drain-source swap law (`ROUNDING = 1024` eps). Setting
    /// `threshold_voltage` above derived `phi` (~0.695 V at `substrate_doping = 1e22`)
    /// keeps every `vds = 0` probe outside Meyer's `-phi < vgst <= 0` derivative kink.
    #[test]
    fn level_three_mosfet_swaps_its_drain_and_source() {
        let library = Library::standard();
        let model = library.get("mosfet").expect("mosfet model");
        let symmetric = merge(
            MOSFET_LEVEL_THREE,
            r#"{"drain_area": 2e-11, "source_area": 2e-11, "gate_source_overlap": 3e-10,
                "gate_drain_overlap": 3e-10}"#,
        );
        for velocity in [1.5e5, 0.0] {
            let drift = format!(r#"{{"max_drift_velocity": {velocity}}}"#);
            let parameters = merge(&symmetric, &drift);
            assert_swaps(
                &**model,
                &parameters,
                ["drain", "source"],
                &[3.0, 2.0, 0.0, -0.5],
            );
        }
    }

    /// Verifies that BSIM3 `capmod` 0 passes the complete drain-source swap law
    /// (`ROUNDING = 1024` eps) over all 38 probes around `[3.0, 2.0, 0.0, -0.5]`. Exercises
    /// both the `vbs = 0` `VbseffCV` fix (formerly `1.4e-4` error) and the
    /// `effective_drain` rationalization (formerly `37,886` eps). Every `vds = 0` probe
    /// lies in depletion (`vgst < 0`, `vbs <= 0`), avoiding the `b3ld.c` strong-inversion
    /// and forward-body capacitance jumps tested in `bsim3::tests`.
    #[test]
    fn bsim3_capmod_zero_swaps_over_subthreshold_and_nonzero_vds_probes() {
        let library = Library::standard();
        let model = library.get("bsim3").expect("bsim3 model");
        assert_swaps(
            &**model,
            r#"{"capacitance_model": 0}"#,
            ["drain", "source"],
            &[3.0, 2.0, 0.0, -0.5],
        );
    }

    /// Verifies that transistor currents, charges, and capacitances are continuous across
    /// every region boundary. Meyer's capacitances change slope across boundaries (Meyer,
    /// "MOS models and circuit simulation", RCA Review 32, 1971; ngspice 47 `DEVqmeyer`,
    /// `devsup.c`, lines 673 to 738), making transient displacement-current derivatives
    /// jump. Conductance continuity is therefore checked with displacement currents zeroed.
    #[test]
    fn every_transistor_is_continuous_across_its_regions_boundaries() {
        for &(name, parameters, given) in cases::CASES {
            let mut bench = bench(name, parameters);
            if bench.regions.names.is_empty() {
                continue;
            }
            let natures = bench.natures();
            let x = cases::values(given, natures.len());
            assert_continuous(name, &mut bench, &x, false);
            let still: Vec<f64> = x
                .iter()
                .zip(&natures)
                .map(|(&value, &nature)| {
                    if nature == Nature::Displacement {
                        0.0
                    } else {
                        value
                    }
                })
                .collect();
            assert_continuous(name, &mut bench, &still, true);
        }
    }

    /// Verifies that every kernel produces finite outputs at every finite bias probe. Tests
    /// unlimited evaluations where junction exponentials cannot overflow, and step-limited
    /// Newton evaluations everywhere from both fresh and settled states.
    #[test]
    fn every_kernel_is_finite_at_any_finite_bias() {
        let values: Vec<f64> = NEAR.iter().chain(&FAR).copied().collect();
        for &(name, parameters, given) in cases::CASES {
            let natures = bench(name, parameters).natures();
            let x = cases::values(given, natures.len());
            for point in probes(&x, &natures, &values) {
                if within_junctions_reach(&point, &natures) {
                    let mut fresh = bench(name, parameters);
                    let evaluation = exactly(&mut fresh, &point);
                    assert_finite(name, &fresh, &evaluation, &point);
                }
                for time in [None, Some(1e-6)] {
                    for settled in [false, true] {
                        let mut device = bench(name, parameters);
                        if settled {
                            device.settle(&x);
                        }
                        let evaluation = device.evaluate_with(&Input::new(&point, time, GMIN));
                        assert_finite(name, &device, &evaluation, &point);
                    }
                }
            }
        }
    }

    const BJT_FULL: &str = r#"{"saturation_current": 1e-15, "forward_beta": 150,
        "forward_early_voltage": 60, "reverse_early_voltage": 20,
        "forward_knee_current": 0.05, "reverse_knee_current": 0.01,
        "base_emitter_leakage_current": 1e-14, "base_collector_leakage_current": 1e-13,
        "base_emitter_capacitance": 2e-12, "base_collector_capacitance": 1e-12,
        "base_collector_grading": 0.4, "internal_base_fraction": 0.7,
        "forward_transit_time": 4e-10, "transit_time_bias_coefficient": 2,
        "transit_time_voltage": 3, "transit_time_current": 0.02,
        "reverse_transit_time": 1e-8}"#;

    const MOSFET_LEVEL_ONE: &str = r#"{"threshold_voltage": 0.8, "transconductance_parameter": 1e-4,
        "body_effect": 0.5, "channel_length_modulation": 0.02, "width": 10e-6,
        "length": 2e-6, "lateral_diffusion": 1e-7, "oxide_thickness": 2e-8,
        "drain_area": 2e-11, "source_area": 3e-11, "drain_perimeter": 2e-5,
        "source_perimeter": 3e-5, "bottom_capacitance": 2e-4,
        "sidewall_capacitance": 5e-10, "gate_source_overlap": 4e-10,
        "gate_drain_overlap": 3e-10, "gate_bulk_overlap": 2e-10,
        "nominal_temperature": 290, "saturation_current": 2e-14}"#;

    const MOSFET_DERIVED: &str = r#"{"substrate_doping": 1e22, "oxide_thickness": 2e-8,
        "surface_state_density": 1e14, "surface_mobility": 0.05, "width": 10e-6,
        "length": 2e-6, "saturation_current_density": 1e-4, "drain_area": 2e-11,
        "source_area": 3e-11, "nominal_temperature": 290}"#;

    const MOSFET_LEVEL_THREE: &str = r#"{"level": 3, "threshold_voltage": 0.8, "body_effect": 0.5,
        "width": 20e-6, "length": 2e-6, "lateral_diffusion": 1e-7,
        "oxide_thickness": 3e-8, "substrate_doping": 1e22, "junction_depth": 3e-7,
        "static_feedback": 0.5, "narrow_width_factor": 0.8, "mobility_modulation": 0.1,
        "saturation_field_factor": 0.3, "fast_surface_state_density": 1e15,
        "threshold_adjustment": 0.05, "max_drift_velocity": 1.5e5}"#;

    const VDMOS_FULL: &str = r#"{"threshold_voltage": 3.5, "transconductance_parameter": 12,
        "channel_length_modulation": 0.02, "mobility_modulation": 0.08,
        "triode_conductance_multiplier": 0.7, "subthreshold_slope": 0.12,
        "subthreshold_shift": 0.05, "drain_resistance": 1.5, "source_resistance": 2,
        "gate_resistance": 3, "body_resistance": 4, "drain_source_resistance": 1e6,
        "gate_source_capacitance": 1.2e-9, "max_gate_drain_capacitance": 1.8e-9,
        "min_gate_drain_capacitance": 5e-11, "gate_drain_capacitance_coefficient": 0.4,
        "saturation_current": 5e-12, "emission_coefficient": 1.1,
        "junction_capacitance": 6e-10, "junction_potential": 0.75,
        "grading_coefficient": 0.45, "depletion_coefficient": 0.55, "transit_time": 2e-8,
        "breakdown_voltage": 60, "breakdown_current": 2.5e-4,
        "breakdown_emission_coefficient": 3}"#;

    const VBIC_WARM: &str = r#"{"saturation_current": 2e-16, "extrinsic_collector_resistance": 5,
        "intrinsic_collector_resistance": 20, "intrinsic_base_resistance": 30,
        "parasitic_base_resistance": 40, "base_emitter_capacitance": 2e-14,
        "base_collector_capacitance": 1e-14, "base_emitter_potential": 0.85,
        "base_collector_potential": 0.65, "avalanche_coefficient": 2.5,
        "avalanche_exponent_coefficient": 12, "avalanche_temperature_coefficient": 2e-3,
        "base_emitter_breakdown_voltage": 5, "breakdown_voltage_temperature_coefficient": 1e-3,
        "breakdown_voltage_quadratic_temperature_coefficient": 1e-6,
        "breakdown_emission_temperature_coefficient": 1e-4, "nominal_temperature": 290,
        "activation_energy": 1.15, "saturation_current_temperature_exponent": 2.5,
        "extrinsic_collector_resistance_temperature_exponent": 0.6}"#;

    const BSIM3_SHORT: &str = r#"{"width": 10e-6, "length": 0.35e-6, "tox": 7.6e-9,
        "nch": 1.7e23, "vth0": 0.5, "u0": 0.04, "k1": 0.6, "k2": -0.02, "rdsw": 600, "pclm": 1.2,
        "alpha0": 1e-6, "beta0": 20, "ngate": 3e26, "noff": 1.2, "voffcv": 0.05,
        "drain_area": 1e-11, "source_area": 1.2e-11, "drain_perimeter": 22e-6,
        "source_perimeter": 8e-6, "cgsl": 1e-10, "cgdl": 1.2e-10, "dwg": -5e-9,
        "dwb": 3e-9, "prwg": 0.1, "prwb": -0.1, "a1": 0.1, "a2": 0.9, "pvag": 0.5,
        "ags": 0.2, "keta": -0.03, "etab": -0.1, "cit": 1e-4, "cdscd": 1e-3,
        "ijth": 1e-4, "jsw": 1e-10}"#;

    const BSIM4_SHORT: &str = r#"{"width": 2e-6, "length": 0.15e-6, "toxe": 4.1e-9, "toxp": 3.8e-9,
        "vth0": 0.45, "u0": 0.03, "eu": 1.67, "k1": 0.5, "k2": -0.02, "ngate": 1e26,
        "dvtp0": 1e-8, "dvtp1": 0.5, "rdsw": 200, "prwg": 0.1, "prwb": -0.1, "a1": 0.1,
        "a2": 0.9, "lambda": 1e-10, "vtl": 2e5, "alpha0": 1e-6, "beta0": 20, "agidl": 1e-9,
        "pvag": 0.5, "pdits": 0.2, "pditsd": 0.3, "fprout": 0.2, "keta": -0.03,
        "etab": -0.1, "cit": 1e-4, "cdscd": 1e-3, "jtss": 1e-7, "jtssws": 1e-12,
        "drain_area": 1e-12, "source_area": 1.2e-12, "drain_perimeter": 4e-6,
        "source_perimeter": 5e-6, "cgsl": 1e-10, "cgdl": 1.2e-10, "dwg": -5e-9,
        "dwb": 3e-9}"#;

    const DIODE_HEATED: &str = r#"{"saturation_current": 1e-14, "series_resistance": 2,
        "junction_capacitance": 2e-12, "transit_time": 1e-9}"#;

    const RESISTOR_HEATED: &str = r#"{"resistance": 1000, "temperature_coefficient": 4e-3,
        "quadratic_temperature_coefficient": 1e-5, "voltage_coefficient": -0.05}"#;

    const MOSFET_BIASES: &[&[f64]] = &[
        &[3.0, 2.0, 0.0, -0.5, 1e-4, 2e-4, -1e-4],
        &[0.3, 2.5, 0.0, -0.2, 1e-4, 2e-4, -1e-4],
    ];

    /// Plain heated case: `(model, cool_parameters, biases, rises)`.
    type HeatedCase<'a> = (&'a str, &'a str, &'a [&'a [f64]], &'a [f64]);

    /// Owned heated case: `(model, cool_parameters, biases, rises)`.
    type OwnedHeatedCase<'a> = (&'a str, String, &'a [&'a [f64]], &'a [f64]);

    /// Plain heated cases: `(model, cool_parameters, biases, rises)`.
    const HEATED_BASE: &[HeatedCase<'_>] = &[
        (
            "diode",
            DIODE_HEATED,
            &[&[0.72, 0.0, 0.7], &[-3.0, 0.0, -2.9]],
            &[0.0, 15.0, 60.0],
        ),
        (
            "jfet",
            r#"{"transconductance_parameter": 1e-3, "drain_resistance": 10,
                "gate_source_capacitance": 2e-12,
                "threshold_voltage_temperature_coefficient": -2e-3,
                "transconductance_parameter_temperature_exponent": -1.5,
                "saturation_current_temperature_exponent": 2}"#,
            &[&[5.0, -0.5, 0.0, 4.9], &[0.3, -1.0, 0.0, 0.29]],
            &[20.0, 55.0],
        ),
        ("mosfet", MOSFET_LEVEL_ONE, MOSFET_BIASES, &[20.0, 60.0]),
        ("mosfet", MOSFET_DERIVED, MOSFET_BIASES, &[20.0, 60.0]),
        (
            "resistor",
            RESISTOR_HEATED,
            &[&[3.0, 1.0], &[2.5, -0.5]],
            &[10.0, 80.0],
        ),
        (
            "vbic",
            VBIC_WARM,
            &[&[3.0, 0.75, 0.0, -1.0, 2.8, 2.6, 2.7, 0.68]],
            &[20.0, 70.0],
        ),
    ];

    /// Verifies that every model accepting `thermal: true` matches its isothermal
    /// counterpart at `ambient + rise` in residuals, charges, Jacobians, and dissipated
    /// power, and differentiates accurately with respect to both unknowns and parameters.
    #[test]
    fn every_heated_model_matches_its_isothermal_model() {
        let library = Library::standard();
        let thermal_models: Vec<&str> = analog(&library)
            .into_iter()
            .filter(|name| {
                library
                    .get(name)
                    .expect("a model")
                    .schema()
                    .get("properties")
                    .and_then(|properties| properties.get("thermal"))
                    .is_some()
            })
            .collect();
        let mut cases: Vec<OwnedHeatedCase<'_>> = HEATED_BASE
            .iter()
            .map(|&(name, cool, biases, rises)| (name, cool.to_owned(), biases, rises))
            .collect();
        let bjt_heated_biases: &[&[f64]] = &[&[3.0, 0.7, 0.0, 0.69], &[0.3, 0.75, 0.0, 0.74]];
        let bsim3_heated_biases: &[&[f64]] = &[&[2.0, 1.5, 0.0, -0.5], &[0.3, 1.0, 0.0, -0.2]];
        let bsim4_heated_biases: &[&[f64]] = &[&[1.5, 1.2, 0.0, -0.5], &[0.3, 0.9, 0.0, -0.2]];
        let resistor_heated_biases: &[&[f64]] = &[&[3.0, 1.0, 2e4], &[2.5, -0.5, 2e4]];
        let vdmos_heated_biases: &[&[f64]] = &[
            &[12.0, 6.0, 0.0, 11.6, 0.3, 5.9, 0.2],
            &[-0.8, 5.0, 0.0, 11.6, 0.3, 5.9, 0.2],
            &[61.0, 0.0, 0.0, 11.6, 0.3, 5.9, 0.2],
        ];
        cases.extend([
            (
                "bipolar_transistor",
                merge(BJT_FULL, r#"{"base_resistance": 50}"#),
                bjt_heated_biases,
                &[20.0, 70.0][..],
            ),
            // Without the substrate current, whose derivatives below saturation central
            // differences no longer resolve.
            (
                "bsim3",
                without(BSIM3_SHORT, "alpha0"),
                bsim3_heated_biases,
                &[20.0, 60.0],
            ),
            (
                "bsim4",
                without(BSIM4_SHORT, "alpha0"),
                bsim4_heated_biases,
                &[20.0, 60.0],
            ),
            (
                "mosfet",
                merge(
                    MOSFET_LEVEL_THREE,
                    r#"{"threshold_voltage": 0.7, "body_effect": 0.4}"#,
                ),
                MOSFET_BIASES,
                &[20.0, 60.0],
            ),
            (
                "resistor",
                merge(RESISTOR_HEATED, r#"{"resistance": 1e-4}"#),
                resistor_heated_biases,
                &[10.0, 80.0],
            ),
            (
                "vdmos",
                merge(
                    VDMOS_FULL,
                    r#"{"quasi_saturation_resistance": 5, "quasi_saturation_voltage": 20,
                        "nominal_temperature": 290,
                        "threshold_voltage_temperature_coefficient": -4e-3,
                        "drain_resistance_temperature_coefficient": 3e-3,
                        "source_resistance_temperature_coefficient": 2e-3,
                        "body_resistance_temperature_coefficient": 4e-3}"#,
                ),
                vdmos_heated_biases,
                &[20.0, 60.0],
            ),
        ]);
        let mut covered: Vec<&str> = cases.iter().map(|&(name, ..)| name).collect();
        covered.sort_unstable();
        covered.dedup();
        assert_eq!(covered, thermal_models, "every thermal model is covered");
        for (name, cool, biases, rises) in cases {
            assert_heats(&**library.get(name).expect("a model"), &cool, biases, rises);
        }
        let diode = library.get("diode").expect("diode model");
        let heated_diode = merge(DIODE_HEATED, r#"{"thermal": true}"#);
        for x in [
            [0.72, 0.0, 0.0, 0.7],
            [0.72, 0.0, 15.0, 0.7],
            [-3.0, 0.0, 60.0, -2.9],
        ] {
            assert_unknown_derivatives_match(&**diode, &heated_diode, &x);
        }
    }

    /// Waveforms checked at nonzero times for both independent sources.
    const SOURCE_WAVEFORMS: &[(&str, f64)] = &[
        (r#"{"waveform": {"dc": 5}}"#, 0.0),
        (
            r#"{"waveform": {"pulse": {"initial": 0.5, "pulsed": 2, "delay": 0.1,
                "rise": 0.2, "fall": 0.3, "width": 0.4, "period": 1.5}}}"#,
            1.75,
        ),
        (
            r#"{"waveform": {"sine": {"offset": 1, "amplitude": 2, "frequency": 0.25,
                "delay": 0.5, "damping": 0.5, "phase": 0.3}}}"#,
            1.0,
        ),
        (
            r#"{"waveform": {"piecewise": {"points": [[-1, 0.5], [2, 4]]}}}"#,
            0.5,
        ),
    ];

    /// Non-transistor parameter-gradient cases beyond `cases::CASES`.
    const ELEMENT_GRADIENTS: &[(&str, &str, &[f64])] = &[
        (
            "resistor",
            r#"{"resistance": 50, "temperature_coefficient": 1e-3,
                "quadratic_temperature_coefficient": 1e-5, "nominal_temperature": 290,
                "voltage_coefficient": -0.05, "quadratic_voltage_coefficient": 0.02}"#,
            &[1.0, -0.5],
        ),
        (
            "resistor",
            r#"{"resistance": 50, "exponential_temperature_coefficient": 0.3}"#,
            &[1.0, -0.5],
        ),
        (
            "resistor",
            r#"{"resistance": 0.5, "temperature_coefficient": 1e-3,
                "voltage_coefficient": -0.05, "quadratic_voltage_coefficient": 0.02}"#,
            &[1.0, -0.5, 2.5],
        ),
        (
            "capacitor",
            r#"{"capacitance": 2e-9, "temperature_coefficient": 1e-3,
                "quadratic_temperature_coefficient": 1e-5, "nominal_temperature": 290,
                "voltage_coefficient": -0.05, "quadratic_voltage_coefficient": 0.02}"#,
            &[1.0, -0.5],
        ),
        (
            "inductor",
            r#"{"inductance": 1e-3, "temperature_coefficient": 1e-3,
                "quadratic_temperature_coefficient": 1e-5, "nominal_temperature": 290}"#,
            &[1.0, -0.5, 0.25],
        ),
        (
            "coupled_inductors",
            r#"{"windings": [
                {"inductance": 1e-3, "temperature_coefficient": 1e-3, "nominal_temperature": 290},
                {"inductance": 4e-3}],
                "couplings": [{"windings": [1, 2], "coefficient": 0.9}]}"#,
            &[1.0, 0.5, -2.0, 0.0, 0.3, -0.1],
        ),
        (
            "diode",
            r#"{"series_resistance": 10, "junction_capacitance": 1e-12}"#,
            &[0.7, 0.0, 0.65],
        ),
    ];

    /// Checks BSIM3 and BSIM4 parameter gradients across binned, derived and multi-regime
    /// operating points.
    fn check_bsim_gradients(library: &Library) {
        let check = |name: &str, parameters: &str, x: &[f64]| {
            let model = library.get(name).expect("a model");
            assert_model_parameter_gradients_match(&**model, parameters, x, None);
        };
        // Below saturation the substrate current falls to 1e-19 A, whose derivatives
        // central differences no longer resolve: it is checked in saturation alone.
        let substrate = merge(
            BSIM3_SHORT,
            r#"{"lvth0": 2e-8, "wvth0": -1e-8, "pvth0": 1e-15, "lu0": 1e-9, "wua": 1e-16,
                "pk2": 1e-15, "lrdsw": 1e-5, "wvsat": 1e-3, "lcgsl": 1e-17,
                "nominal_temperature": 290, "kt1l": 1e-9, "ute": -1.3, "prt": 20,
                "tpb": 1e-3, "tcj": 1e-3, "xti": 2.5, "rsh": 15}"#,
        );
        let smooth = without(&substrate, "alpha0");
        let derived = r#"{"width": 4e-6, "length": 1e-6, "nsub": 2e22, "xt": 1.2e-7,
            "lint": 2e-8, "wint": 3e-8, "ll": 1e-13, "vfb": -2, "dlc": 3e-8,
            "capacitance_model": 2, "mobility_model": 2, "nominal_temperature": 290,
            "drain_area": 4e-12, "source_area": 4e-12}"#;
        for [drain, gate, bulk] in [
            [2.0, 1.5, -0.5],
            [0.3, 2.0, -0.2],
            [-1.0, 1.2, -1.5],
            [2.0, 0.3, -0.1],
        ] {
            let inner = [drain, gate, 0.0, bulk, 0.8 * drain, 0.1];
            check("bsim3", &smooth, &inner);
            check("bsim3", derived, &[drain, gate, 0.0, bulk]);
        }
        check("bsim3", &substrate, &[2.5, 1.5, 0.0, -0.5, 2.2, 0.1]);

        let full = merge(
            BSIM4_SHORT,
            r#"{"lvth0": 2e-9, "wvth0": -1e-9, "pvth0": 1e-16, "lu0": 1e-10, "wua": 1e-17,
                "pk2": 1e-16, "lrdsw": 1e-6, "wvsat": 1e-3, "lcgsl": 1e-18,
                "nominal_temperature": 290, "kt1l": 1e-9, "ute": -1.3, "prt": 20, "tpb": 1e-3,
                "tcj": 1e-3, "rsh": 15, "drain_squares": 1, "source_squares": 1,
                "gate_resistance_model": 3, "rshg": 20, "xgw": 1e-7, "lxrcrg1": 1e-6,
                "wxrcrg2": 1e-7}"#,
        );
        for [drain, gate, bulk] in [
            [1.5, 1.2, -0.5],
            // Strongly inverted, the subthreshold swing moves the currents by 1e-15 of
            // themselves, which central differences no longer resolve: the gate stays
            // below 1 V here.
            [0.3, 0.9, -0.2],
            [-1.0, 1.2, -1.5],
            [1.5, 0.3, -0.1],
        ] {
            let inner = [
                drain,
                gate,
                0.0,
                bulk,
                0.8 * drain,
                0.1,
                0.9 * gate,
                0.95 * gate,
            ];
            check("bsim4", &full, &inner);
        }
    }

    /// Checks parameter gradients across every transistor's rich operating regimes.
    fn check_transistor_gradients(library: &Library) {
        let check = |name: &str, parameters: &str, x: &[f64]| {
            let model = library.get(name).expect("a model");
            assert_model_parameter_gradients_match(&**model, parameters, x, None);
        };
        let vbic_bias = [3.0, 0.75, 0.0, -1.0, 2.8, 2.6, 2.7, 0.68];
        check("vbic", VBIC_WARM, &vbic_bias);
        let jfet = r#"{"threshold_voltage": -1.8, "transconductance_parameter": 1e-3,
            "channel_length_modulation": 0.02, "drain_resistance": 10,
            "source_resistance": 5, "gate_source_capacitance": 2e-12,
            "gate_drain_capacitance": 1e-12, "junction_potential": 0.9,
            "saturation_current": 2e-14, "depletion_coefficient": 0.4, "doping_tail": 0.8,
            "threshold_voltage_temperature_coefficient": -2e-3,
            "transconductance_parameter_temperature_exponent": -1.5,
            "energy_gap": 1.2, "saturation_current_temperature_exponent": 2,
            "nominal_temperature": 290}"#;
        for [drain, gate] in [[3.0, 0.0], [0.5, -0.5], [-0.5, -0.2], [-3.0, 0.1]] {
            check("jfet", jfet, &[drain, gate, 0.0, drain - 0.01, 0.005]);
        }
        let mosfet_level_one = merge(
            MOSFET_LEVEL_ONE,
            r#"{"junction_potential": 0.75, "bottom_grading": 0.45, "sidewall_grading": 0.35,
                "depletion_coefficient": 0.45, "multiplier": 2}"#,
        );
        let mosfet_resistive = merge(
            &mosfet_level_one,
            r#"{"drain_resistance": 50, "source_resistance": 20}"#,
        );
        let mosfet_derived = merge(
            MOSFET_DERIVED,
            r#"{"sheet_resistance": 10, "drain_squares": 2, "source_squares": 3}"#,
        );
        let mosfet_level_three = merge(
            MOSFET_LEVEL_THREE,
            r#"{"threshold_voltage": 0.7, "body_effect": 0.4, "width_narrowing": 1e-7,
                "length_adjustment": 1e-8, "width_adjustment": 2e-8, "nominal_temperature": 290}"#,
        );
        for [drain, gate, bulk] in [
            [3.0, 2.0, -0.5],
            [0.3, 2.5, -0.2],
            [-1.0, 1.5, -1.5],
            [2.0, 0.2, -0.1],
        ] {
            let x = [drain, gate, 0.0, bulk, 1e-4, 2e-4, -1e-4];
            let inner = [drain, gate, 0.0, bulk, 0.9 * drain, 0.01, 1e-4, 2e-4, -1e-4];
            check("mosfet", &mosfet_level_one, &x);
            check("mosfet", &mosfet_derived, &inner);
            check("mosfet", &mosfet_level_three, &x);
            check("mosfet", &mosfet_resistive, &inner);
        }
        let bjt_warm = merge(
            BJT_FULL,
            r#"{"energy_gap": 1.2, "saturation_current_temperature_exponent": 3.5,
                "beta_temperature_exponent": 1.2, "nominal_temperature": 290}"#,
        );
        let bjt_delayed = merge(&bjt_warm, r#"{"excess_phase": 30}"#);
        let bjt_substrate = merge(
            &bjt_warm,
            r#"{"base_resistance": 200, "min_base_resistance": 20, "collector_resistance": 10,
                "emitter_resistance": 2, "substrate": "lateral",
                "substrate_saturation_current": 1e-16, "substrate_emission_coefficient": 1.2,
                "substrate_capacitance": 3e-12, "substrate_potential": 0.6,
                "substrate_grading": 0.4}"#,
        );
        let bjt_halfway = merge(&bjt_substrate, r#"{"base_resistance_current": 1e-4}"#);
        for [collector, base] in [[3.0, 0.7], [0.4, 0.72], [-0.5, 0.1], [0.2, -0.6]] {
            check("bipolar_transistor", &bjt_warm, &[collector, base, 0.0]);
            check(
                "bipolar_transistor",
                &bjt_delayed,
                &[collector, base, 0.0, 2e-3, -1e-4],
            );
            let substrate_bias = [
                collector,
                base,
                0.0,
                -1.0,
                collector - 0.01,
                base - 0.02,
                0.001,
            ];
            check("bipolar_transistor", &bjt_substrate, &substrate_bias);
            check("bipolar_transistor", &bjt_halfway, &substrate_bias);
        }
    }

    /// Checks VDMOS parameter gradients across temperature coefficients, exponents and small
    /// resistances in both channels.
    fn check_vdmos_gradients(library: &Library) {
        let model = library.get("vdmos").expect("vdmos");
        let every = merge(
            VDMOS_FULL,
            r#"{"quasi_saturation_resistance": 5, "quasi_saturation_voltage": 20,
                "nominal_temperature": 290, "threshold_voltage_temperature_coefficient": -4e-3,
                "transconductance_parameter_temperature_exponent": -1.3,
                "subthreshold_slope_temperature_coefficient": 2e-3,
                "subthreshold_slope_quadratic_temperature_coefficient": 1e-6,
                "drain_resistance_temperature_coefficient": 3e-3,
                "drain_resistance_quadratic_temperature_coefficient": 1e-5,
                "quasi_saturation_resistance_temperature_exponent": 0.4,
                "source_resistance_temperature_coefficient": 2e-3,
                "source_resistance_quadratic_temperature_coefficient": 2e-5,
                "gate_resistance_temperature_coefficient": 1e-3,
                "gate_resistance_quadratic_temperature_coefficient": 3e-5,
                "body_resistance_temperature_coefficient": 4e-3,
                "body_resistance_quadratic_temperature_coefficient": 4e-5,
                "energy_gap": 1.2, "saturation_current_temperature_exponent": 2.5}"#,
        );
        let p_channel = mirrored("vdmos", &polarity(&**model).expect("polarity"), &every);
        let exponent = merge(
            VDMOS_FULL,
            r#"{"drain_resistance_temperature_exponent": 1.7, "nominal_temperature": 310}"#,
        );
        let small = r#"{"drain_resistance": 0.05, "source_resistance": 0.02,
            "gate_resistance": 0.5, "body_resistance": 0.01,
            "quasi_saturation_resistance": 0.1, "quasi_saturation_voltage": 10,
            "drain_resistance_temperature_coefficient": 2e-3, "multiplier": 3}"#;
        for [drain, gate] in [
            [12.0, 6.0],
            [0.4, 8.0],
            [5.0, 3.4],
            [-0.8, 6.0],
            [61.0, 0.0],
            [3.0, -2.0],
        ] {
            let x = [drain, gate, 0.0, 0.97 * drain, 0.02, 0.99 * gate, 0.01];
            let inner = [
                drain,
                gate,
                0.0,
                0.99 * drain,
                1e-2,
                0.002,
                -3e-2,
                0.98 * gate,
                1e-3,
                0.01,
                2e-2,
            ];
            assert_model_parameter_gradients_match(&**model, &every, &x, None);
            assert_model_parameter_gradients_match(&**model, &p_channel, &x.map(|v| -v), None);
            assert_model_parameter_gradients_match(&**model, &exponent, &x, None);
            assert_model_parameter_gradients_match(&**model, small, &inner, None);
        }
    }

    /// Verifies that every analog model with real parameters computes exact analytic
    /// parameter gradients matching extrapolated central finite differences across smooth
    /// `cases::CASES` entries and rich multi-regime operating points.
    #[test]
    fn every_models_parameter_gradients_are_exact() {
        let library = Library::standard();
        let behavioral = [
            "behavioral_capacitor",
            "behavioral_current_source",
            "behavioral_voltage_source",
        ];
        // BSIM4's `CASES` entries include discrete `fingers = 2` (even/odd diffusion branch)
        // and `alpha0` below saturation; VBIC's thermal `CASES` entry ties all internal nodes
        // at `REST`. Both models are differentiated at smooth operating points below.
        let mut covered = std::collections::BTreeSet::new();
        for &(name, parameters, given) in cases::CASES {
            if behavioral.contains(&name) || matches!(name, "bsim4" | "vbic") {
                continue;
            }
            let model = library.get(name).expect("a model");
            let x = cases::values(given, bench(name, parameters).natures().len());
            assert_model_parameter_gradients_match(&**model, parameters, &x, None);
            covered.insert(name);
        }
        for &(name, parameters, x) in ELEMENT_GRADIENTS {
            let model = library.get(name).expect("a model");
            assert_model_parameter_gradients_match(&**model, parameters, x, None);
        }
        let diode = r#"{"saturation_current": 2e-14, "emission_coefficient": 1.3,
            "junction_capacitance": 2e-12, "junction_potential": 0.8,
            "grading_coefficient": 0.4, "depletion_coefficient": 0.6, "transit_time": 1e-9,
            "breakdown_voltage": 5, "breakdown_current": 1e-3,
            "breakdown_emission_coefficient": 1.1, "breakdown_temperature_coefficient": 1e-3,
            "energy_gap": 1.2, "saturation_current_temperature_exponent": 2.5,
            "nominal_temperature": 290}"#;
        let diode_model = library.get("diode").expect("diode");
        for voltage in [0.6, 0.55, -0.02, -1.0, -5.4] {
            assert_model_parameter_gradients_match(&**diode_model, diode, &[voltage, 0.0], None);
        }
        let voltage_switch = r#"{"on_resistance": 10, "off_resistance": 1e6, "on_voltage": 2,
            "off_voltage": 1}"#;
        let voltage_switch_model = library
            .get("voltage_controlled_switch")
            .expect("voltage switch");
        for control in [0.5, 1.3, 2.5] {
            assert_model_parameter_gradients_match(
                &**voltage_switch_model,
                voltage_switch,
                &[1.0, -0.5, control, 0.0],
                None,
            );
        }
        let current_switch = r#"{"on_resistance": 10, "off_resistance": 1e6, "on_current": 2e-3,
            "off_current": 1e-3}"#;
        let current_switch_model = library
            .get("current_controlled_switch")
            .expect("current switch");
        for control in [0.0, 1.25, 1.5, 1.9, 3.0] {
            assert_model_parameter_gradients_match(
                &**current_switch_model,
                current_switch,
                &[1.0, -0.5, 0.2, 0.1, control * 1e-3],
                None,
            );
        }
        let voltage_source = library.get("voltage_source").expect("voltage_source");
        let current_source = library.get("current_source").expect("current_source");
        for &(waveform, time) in SOURCE_WAVEFORMS {
            assert_model_parameter_gradients_match(
                &**voltage_source,
                waveform,
                &[1.0, -0.5, 1e-3],
                Some(time),
            );
            assert_model_parameter_gradients_match(
                &**current_source,
                waveform,
                &[1.0, -0.5],
                Some(time),
            );
        }
        check_bsim_gradients(&library);
        check_transistor_gradients(&library);
        check_vdmos_gradients(&library);
        covered.extend(["bsim4", "vbic"]);
        let expected: std::collections::BTreeSet<&str> = analog(&library)
            .into_iter()
            .filter(|name| !behavioral.contains(name))
            .collect();
        assert_eq!(covered, expected, "every differentiable model is covered");
    }
}
