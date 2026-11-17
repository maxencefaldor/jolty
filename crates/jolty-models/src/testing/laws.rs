//! The laws every kernel keeps, tested over the whole library: at each case of every
//! analog model ([`cases::CASES`]) and at probes around it, so that a kernel added to the
//! library is held to them by its case alone.
//!
//! - **What flows in flows out:** over an instance's electrical nodes, its currents sum to
//!   zero, and so do its charges and the derivatives of both.
//! - **Mirrors:** a p-channel or a PNP is its n-channel's or NPN's mirror, every voltage
//!   and current reversed, bit for bit.
//! - **Symmetry:** a transistor that claims its drain and its source alike swaps them,
//!   within rounding, which is Gummel's symmetry test to the first order.
//! - **Continuity:** a transistor's currents and charges and their derivatives are
//!   continuous where a margin marks a boundary between two of its regions.
//! - **Finite values:** at any finite bias, every number a kernel computes is finite.

use serde_json::{Value, json};

use super::{Bench, Evaluation, cases};
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

/// The parameters of each transistor whose sign follows its polarity, as SPICE's
/// thresholds' does: an n-channel's `vto` of 0.8 V is a p-channel's of −0.8 V. A device's
/// mirror is the device of the other polarity with each of these negated and every other
/// parameter as it is.
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

/// A device of each transistor that claims its drain and source alike, the two named, as
/// Gummel's symmetry test takes one (McAndrew, "Validation of MOSFET model source–drain
/// symmetry", IEEE Transactions on Electron Devices 53, 2006), by its model's name, its
/// parameters, alike at the drain and the source, and the values of its terminals; `None`
/// for a transistor that claims no symmetry.
const SYMMETRIES: &[(&str, Option<Alike>)] = &[
    ("bipolar_transistor", None),
    // BSIM3 swaps its currents, but its charges, written against the source, are not
    // smooth through `vds = 0`: there, its drain and source alike, its capacitances differ
    // under the swap by 1.4e-4 of the largest at `capmod` 0, 2 and 3, and by 1.5 times it at
    // `capmod` 1.
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

/// A device of the standard library's model `name` with `parameters`.
fn bench(name: &str, parameters: &str) -> Bench {
    let library = Library::standard();
    let model = library.get(name).expect("a model of the library");
    Bench::from_model(&**model, parameters, REFERENCE_TEMPERATURE)
        .unwrap_or_else(|error| panic!("{name}: {error}"))
}

/// The bench's device evaluated at `x` itself, no step limited.
fn exactly(bench: &mut Bench, x: &[f64]) -> Evaluation {
    let mut input = Input::new(x, None, GMIN);
    input.limiting = false;
    bench.evaluate_with(&input)
}

/// The points around `given`, the values of a device's local unknowns of `natures`, at
/// which the laws are checked: `given` itself, reversed and halved, each unknown at each of
/// `values` with the rest as given, and every unknown at each of them. A temperature keeps
/// its value throughout: it is no bias.
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

/// Whether no two voltages of `x` lie further apart than the 709.78 thermal voltages at the
/// reference temperature, `ln(f64::MAX)`, beyond which a junction's exponential overflows
/// in its source's equations as in jolty's, its emission coefficient at least one: unlimited,
/// SPICE's junctions leave such a step to their limiting (`DEVpnjlim`, ngspice 47's
/// `devsup.c`, lines 49 to 84), so that Newton's iteration never evaluates one there.
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

/// Checks that over the electrical nodes of `bench`'s device, its currents at `x` sum to
/// zero, and so do its charges, and each column of its conductances and capacitances.
///
/// Each kernel adds each current and each charge to the row of one node and its negation
/// to another's, so their exact sum over the rows is zero, and the rows' floating sums lie
/// within the rounding of their additions: `(k − 1)·u` of the magnitudes of a row's `k`
/// terms, to first order, `u` half of `f64::EPSILON` (Higham, "Accuracy and Stability of
/// Numerical Algorithms", 2002, §4.2); the test's own sum of the rows adds as much again.
/// With at most a term in a row for each of the instance's `n` local unknowns, as each
/// current or charge joins two of them, the sum lies within `n² · ε` of the rows' largest
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

/// The values of each of [`KINDS`] of `evaluation`, the matrices row by row.
fn kinds(evaluation: &Evaluation) -> [Vec<f64>; 4] {
    [
        evaluation.residual.clone(),
        evaluation.charge.clone(),
        evaluation.conductance.concat(),
        evaluation.capacitance.concat(),
    ]
}

/// The largest magnitude of a term of each row of each of [`KINDS`] of `evaluation`, the
/// device's at `x`: a residual's its scale, as its kernel reports it, and for the rest,
/// whose terms no kernel reports, what bounds them to first order. A charge's terms are its
/// capacitances times its voltages, or itself; an entry of a row of derivatives is the
/// derivatives of the row's currents and charges with respect to the voltages of the
/// kernel's own frame, mapped to the local unknowns' by sums of them, the row's largest.
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

/// The parameter that sets the polarity of a device of `model`, the property of its
/// parameters of the type [`Channel`] or [`Polarity`], whose two variants are each other's
/// mirror, with those variants as written; `None` for a model without one.
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

/// The parameters of the mirror of a device of the transistor `name` with `parameters`: the
/// other polarity, and each parameter whose sign follows it negated ([`SIGNED`]).
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

/// Checks that `image`, the evaluation of a device's mirror at `x` reflected, is
/// `evaluation`, the device's at `x`, reflected alike, exactly: each residual and charge of
/// a row of sign −1 in `signs` negated, each conductance and capacitance negated where its
/// row's and its column's signs differ; the same region and margins, which are in an
/// n-channel's or an NPN's sign; and each variable of the same magnitude.
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
    for (name, (own, other)) in device.variable_names().iter().zip(variables) {
        assert!(
            own.abs() == other.abs(),
            "{what}: the variable {name} {own:e}, mirrored {other:e}, at {x:?}"
        );
    }
}

/// Checks that a device of `model` with `parameters`, a transistor with the values `given`
/// of its first local unknowns, mirrors bit for bit: that the device of the other polarity,
/// each parameter whose sign follows it negated, evaluated with every voltage and current
/// reversed and its temperatures kept, gives every residual and charge reversed, every
/// derivative of the same orientation, and the same region and margins, both exactly at
/// each probe around `given` and as Newton's iteration settles at it.
pub(crate) fn assert_mirrors(model: &dyn Model, parameters: &str, given: &[f64]) {
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

/// Checks that a device of `model` with `parameters`, alike at its two terminals `pair`,
/// swaps them: that at each probe around `given`, the values of its first local unknowns,
/// the device with the values of the two terminals exchanged, and of each pair of internal
/// unknowns whose labels differ by those terminals' names alone, gives each one's residual,
/// charge and derivatives as the other's, within [`ROUNDING`].
///
/// Not bit for bit: the swapped device computes its voltages against its other terminal,
/// which rounds them otherwise, and SPICE's transistors, which compute in the frame whose
/// source is the lower of the two, reach the derivatives of the swapped frame by sums of
/// the frame's own, as `b3ld.c` does.
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
        let rows = |values: &[f64]| swap.iter().map(|&row| values[row]).collect();
        let matrix = |values: &[Vec<f64>]| {
            let entries = |row: &Vec<f64>| swap.iter().map(|&column| row[column]).collect();
            swap.iter().map(|&row| entries(&values[row])).collect()
        };
        let swapped = Evaluation {
            residual: rows(&image.residual),
            scale: rows(&image.scale),
            charge: rows(&image.charge),
            conductance: matrix(&image.conductance),
            capacitance: matrix(&image.capacitance),
            limited: image.limited,
        };
        let what = format!("{name} {parameters} with its {one} and {other} swapped");
        assert_agree(&what, [&own, &swapped], &point, true);
    }
}

/// The roundings, as multiples of `f64::EPSILON` of the largest magnitude of a term of a
/// kind of value, within which two evaluations of one value at voltages rounded otherwise
/// agree, as at two adjacent floating-point numbers, where a continuous value moves by its
/// derivative times a step of an ulp, far below its rounding.
///
/// A value computed in `m` operations lies within `γ(m) ≈ m · u` of its terms' magnitude
/// (Higham 2002, §3.1), and no kernel's way to a value takes a thousand.
const ROUNDING: f64 = 1024.0;

/// Checks that where a margin of `bench`'s device changes sign as any one of its local
/// unknowns but a temperature moves from −10 to 10 from `x`, the device's residuals,
/// charges and capacitances, and its conductances unless `slopes` is false, agree at the
/// adjacent floating-point numbers either side, within [`ROUNDING`].
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

mod tests {
    use super::{
        FAR, GMIN, Input, Library, NEAR, Nature, SIGNED, SYMMETRIES, assert_conserves,
        assert_continuous, assert_finite, assert_mirrors, assert_swaps, bench, cases, exactly,
        polarity, probes, within_junctions_reach,
    };
    use crate::testing::assert_unknown_derivatives_match;

    /// Every analog model of the library, by name.
    fn analog(library: &Library) -> Vec<&'static str> {
        library
            .names()
            .filter(|name| library.get(name).is_some())
            .collect()
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

    /// Every kernel reads its unknowns as numbers of its kind, so that a twin seeding them
    /// finds the second derivatives of its equations exactly, as an AC analysis's
    /// derivatives through its operating point need them: at its case, its internal
    /// unknowns spread apart rather than tied at one voltage, and with each of its
    /// unknowns shifted either side of it, away from the points where its terminals meet,
    /// where laws switch and their curvatures jump.
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

    /// The transistors, the models whose equations distinguish regions, whose margins are
    /// in an n-channel's or an NPN's sign, each have a polarity, and each case of a model
    /// with one mirrors.
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
    }

    /// Every transistor that claims its drain and its source alike swaps them, and at the
    /// probes where the two lie at one voltage, its derivatives, of the frame its equations
    /// take there, are the swapped ones: Gummel's symmetry test to the first order, whose
    /// drain current, the drain at `vx` and the source at `−vx`, is then odd in `vx` and its
    /// slope even and continuous through `vx = 0`.
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

    /// Meyer's capacitances, continuous across each boundary, change their slopes there
    /// (Meyer, "MOS models and circuit simulation", RCA Review 32, 1971; ngspice 47's
    /// `DEVqmeyer`, `devsup.c`, lines 673 to 738), so where displacement currents flow, as in
    /// a transient, the derivatives of the currents through them jump: the conductances
    /// are checked where no displacement current flows, as at an operating point.
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

    /// Every kernel computes finite numbers at every probe: exactly, where no junction's
    /// exponential overflows in its source's equations, and everywhere as Newton's
    /// iteration evaluates it, each step limited from the last, from its first and from a
    /// settled operating point.
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
}
