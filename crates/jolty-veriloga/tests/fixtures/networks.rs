//! The `networks` kernel, compiled by jolty-veriloga from `crates/jolty-veriloga/tests/fixtures/networks.va`.
//!
//! Written by `cargo run -p jolty-veriloga --bin compile`, never by hand. Its source says:
//!
//! ```text
//! Shorts and branch currents, written to exercise how the compiler joins nodes and
//! keeps currents: the current of a short read, a node joined to another by a short,
//! two nodes joined to each other and through one to the ground where a parameter says,
//! and the noise of a branch whose current another branch reads, as PSP's drain reads
//! its induced gate noise's.
//! ```

#![cfg_attr(
    not(test),
    deny(
        clippy::indexing_slicing,
        clippy::panic,
        reason = "a kernel never panics, and reads fixed arrays through `get`"
    )
)]

use jolty_models::{Branch, BranchKind, Dual, Interface, Kernel, Layout, Lift, Local, Nature, NoiseSource, ParameterError, Real, Stamp, UnknownIndex};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A local index no instance has: a write to it goes nowhere, and a read of it reads
/// zero, as `Stamp` and `Local` promise.
const NOWHERE: usize = usize::MAX;

/// The `networks` model, compiled from its Verilog-A source.
#[derive(Debug, Clone, Copy)]
pub struct Networks;

/// The parameters of a [`Networks`], with its Verilog-A source's defaults.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift)]
#[serde(default, deny_unknown_fields, bound = "")]
pub struct NetworksParameters<R: Real = f64> {
    /// The resistance of each resistor, in Ohm; default 1000.
    pub r: R,
    /// The share of the noise current a's current carries; default 0.1.
    pub k: R,
    /// The density of the noise current, in A^2/Hz; default 0.00000000000000000001.
    pub s: R,
    /// Whether the lower node is shorted to the ground; default 0.
    pub grounded: i32,
}

impl<R: Real> Default for NetworksParameters<R> {
    fn default() -> Self {
        Self {
            r: R::constant(1000.0),
            k: R::constant(0.1),
            s: R::constant(1e-20),
            grounded: 0,
        }
    }
}

/// A [`Networks`]'s numbers at its temperature: what depends on no probe.
#[derive(Debug, Clone, Copy)]
pub struct PreparedNetworks<R: Real> {
    r: R,
    h1: R,
    k: R,
    s: R,
    /// The conditions that shape the kernel.
    conditions: [bool; 1],
}

/// Where a [`Networks`]'s contributions go among its local unknowns, and the conditions
/// that chose them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NetworksShape {
    inner: usize,
    joined: usize,
    upper: usize,
    lower: usize,
    noise: usize,
    source: usize,
    short: usize,
    a_by_a: usize,
    a_by_upper: usize,
    a_by_source: usize,
    a_by_short: usize,
    b_by_b: usize,
    b_by_inner: usize,
    b_by_joined: usize,
    b_by_source: usize,
    inner_by_b: usize,
    inner_by_inner: usize,
    inner_by_short: usize,
    joined_by_b: usize,
    joined_by_joined: usize,
    upper_by_a: usize,
    upper_by_upper: usize,
    lower_by_lower: usize,
    noise_by_noise: usize,
    noise_by_source: usize,
    source_by_source: usize,
    short_by_a: usize,
    short_by_inner: usize,
    conditions: [bool; 1],
    active: [bool; 2],
}

/// What a [`Networks`] remembers of its last evaluation: each output variable.
#[derive(Debug, Clone, Copy, Default)]
pub struct NetworksMemory {
    shorted: f64,
}

impl Kernel for Networks {
    const NAME: &'static str = "networks";
    type Parameters<R: Real> = NetworksParameters<R>;
    type Shape = NetworksShape;
    type Prepared<R: Real> = PreparedNetworks<R>;
    type Memory = NetworksMemory;

    fn interface<R: Real>(_: &NetworksParameters<R>) -> Interface {
        Interface::new(&["a", "b"], &[Branch::new(0, 1, BranchKind::Conductive)])
    }

    fn variable_names() -> &'static [&'static str] {
        &["shorted"]
    }

    fn variables(memory: &NetworksMemory, values: &mut [f64]) {
        for (value, kept) in values.iter_mut().zip([memory.shorted]) {
            *value = kept;
        }
    }

    fn prepare<R: Real>(
        parameters: &NetworksParameters<R>,
        _: R,
    ) -> Result<PreparedNetworks<R>, ParameterError> {
        ParameterError::positive("r", parameters.r)?;
        ParameterError::non_negative("s", parameters.s)?;
        if !(0.0..=1.0).contains(&f64::from(parameters.grounded)) {
    return Err(ParameterError(format!("grounded must be at least 0 and at most 1, not {:?}", parameters.grounded)));
}
        Ok(PreparedNetworks { r: parameters.r, h1: R::constant(1.0) / parameters.r, k: parameters.k, s: parameters.s, conditions: [parameters.grounded != 0] })
    }

    fn layout<R: Real>(
        _: &NetworksParameters<R>,
        prepared: &PreparedNetworks<R>,
        layout: &mut Layout,
    ) -> Result<NetworksShape, ParameterError> {
        let mut active = [false; 2];
        if prepared.conditions[0] {
            active[1] = true;
        } else {
            active[0] = true;
        }
        // The leader of each node's group, the ground first, then the nodes in order.
        let leader = |joined: &[usize; 9], mut place: usize| {
            while let Some(&next) = joined.get(place)
                && next != place
            {
                place = next;
            }
            place
        };
        let mut joined = [0, 1, 2, 3, 4, 5, 6, 7, 8];
        { let (one, other) = (leader(&joined, 3), leader(&joined, 4)); if let Some(slot) = joined.get_mut(one.max(other)) { *slot = one.min(other); } }
        { let (one, other) = (leader(&joined, 5), leader(&joined, 6)); if let Some(slot) = joined.get_mut(one.max(other)) { *slot = one.min(other); } }
        if active[1] { let (one, other) = (leader(&joined, 6), leader(&joined, 0)); if let Some(slot) = joined.get_mut(one.max(other)) { *slot = one.min(other); } }
        let mut local = [NOWHERE, 0, 1, NOWHERE, NOWHERE, NOWHERE, NOWHERE, NOWHERE, NOWHERE];
        local[3] = match leader(&joined, 3) { 3 => layout.internal("inner", Nature::Voltage), leader => local.get(leader).copied().unwrap_or(NOWHERE) };
        local[4] = match leader(&joined, 4) { 4 => layout.internal("joined", Nature::Voltage), leader => local.get(leader).copied().unwrap_or(NOWHERE) };
        local[5] = match leader(&joined, 5) { 5 => layout.internal("upper", Nature::Voltage), leader => local.get(leader).copied().unwrap_or(NOWHERE) };
        local[6] = match leader(&joined, 6) { 6 => layout.internal("lower", Nature::Voltage), leader => local.get(leader).copied().unwrap_or(NOWHERE) };
        local[7] = match leader(&joined, 7) { 7 => layout.internal("noise", Nature::Voltage), leader => local.get(leader).copied().unwrap_or(NOWHERE) };
        local[8] = match leader(&joined, 8) { 8 => layout.internal("source", Nature::Current), leader => local.get(leader).copied().unwrap_or(NOWHERE) };
        let [_, _, _, inner, joined, upper, lower, noise, source] = local;
        let short = layout.internal("short", Nature::Current);
        Ok(NetworksShape { inner, joined, upper, lower, noise, source, short, a_by_a: layout.entry(0, 0), a_by_upper: if upper == NOWHERE { NOWHERE } else { layout.entry(0, upper) }, a_by_source: layout.entry(0, source), a_by_short: layout.entry(0, short), b_by_b: layout.entry(1, 1), b_by_inner: layout.entry(1, inner), b_by_joined: layout.entry(1, joined), b_by_source: layout.entry(1, source), inner_by_b: layout.entry(inner, 1), inner_by_inner: layout.entry(inner, inner), inner_by_short: layout.entry(inner, short), joined_by_b: layout.entry(joined, 1), joined_by_joined: layout.entry(joined, joined), upper_by_a: if upper == NOWHERE { NOWHERE } else { layout.entry(upper, 0) }, upper_by_upper: if upper == NOWHERE { NOWHERE } else { layout.entry(upper, upper) }, lower_by_lower: if active[0] && lower != NOWHERE { layout.entry(lower, lower) } else { NOWHERE }, noise_by_noise: layout.entry(noise, noise), noise_by_source: layout.entry(noise, source), source_by_source: layout.entry(source, source), short_by_a: layout.entry(short, 0), short_by_inner: layout.entry(short, inner), conditions: prepared.conditions, active })
    }

    fn evaluate<R: Real>(
        shape: &NetworksShape,
        prepared: &PreparedNetworks<R>,
        memory: &mut NetworksMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        struct Locals<R: Real> {
            shorted: Dual<R, 1>,
            voltage_a_upper: Dual<R, 1>,
            voltage_inner_b: Dual<R, 1>,
            voltage_joined_b: Dual<R, 1>,
            voltage_lower_ground: Dual<R, 1>,
            voltage_noise_ground: Dual<R, 1>,
            voltage_source_ground: Dual<R, 1>,
            current_short: Dual<R, 1>,
            short_potential: R,
            source_flow: Dual<R, 1>,
            load_flow: Dual<R, 1>,
            inner_b_flow: Dual<R, 1>,
            joined_b_flow: Dual<R, 1>,
            lower_ground_flow: Dual<R, 1>,
            a_upper_flow: Dual<R, 1>,
            a_b_flow: Dual<R, 1>,
            source_ground_flow: Dual<R, 1>,
        }
        let mut v = Locals { shorted: Dual::number(0.0), voltage_a_upper: Dual::number(0.0), voltage_inner_b: Dual::number(0.0), voltage_joined_b: Dual::number(0.0), voltage_lower_ground: Dual::number(0.0), voltage_noise_ground: Dual::number(0.0), voltage_source_ground: Dual::number(0.0), current_short: Dual::number(0.0), short_potential: R::constant(0.0), source_flow: Dual::number(0.0), load_flow: Dual::number(0.0), inner_b_flow: Dual::number(0.0), joined_b_flow: Dual::number(0.0), lower_ground_flow: Dual::number(0.0), a_upper_flow: Dual::number(0.0), a_b_flow: Dual::number(0.0), source_ground_flow: Dual::number(0.0) };
        v.voltage_a_upper = Dual { value: at.unknown::<R>(0) - at.unknown::<R>(shape.upper), gradient: [R::constant(1.0)] };
        v.voltage_inner_b = Dual { value: at.unknown::<R>(shape.inner) - at.unknown::<R>(1), gradient: [R::constant(1.0)] };
        v.voltage_joined_b = Dual { value: at.unknown::<R>(shape.joined) - at.unknown::<R>(1), gradient: [R::constant(1.0)] };
        v.voltage_lower_ground = Dual { value: at.unknown::<R>(shape.lower) - at.unknown::<R>(NOWHERE), gradient: [R::constant(1.0)] };
        v.voltage_noise_ground = Dual { value: at.unknown::<R>(shape.noise) - at.unknown::<R>(NOWHERE), gradient: [R::constant(1.0)] };
        v.voltage_source_ground = Dual { value: at.unknown::<R>(shape.source) - at.unknown::<R>(NOWHERE), gradient: [R::constant(1.0)] };
        v.current_short = Dual { value: at.unknown::<R>(shape.short), gradient: [R::constant(1.0)] };
        v.short_potential += R::constant(0.0);
        v.inner_b_flow = v.inner_b_flow + v.voltage_inner_b.chain((v.voltage_inner_b.value / prepared.r, prepared.h1));
        v.joined_b_flow = v.joined_b_flow + v.voltage_joined_b.chain((v.voltage_joined_b.value / prepared.r, prepared.h1));
        v.shorted = v.current_short;
        if shape.conditions[0] {
        } else {
            v.lower_ground_flow = v.lower_ground_flow + v.voltage_lower_ground.chain((v.voltage_lower_ground.value / prepared.r, prepared.h1));
        }
        v.a_upper_flow = v.a_upper_flow + v.voltage_a_upper.chain((v.voltage_a_upper.value / prepared.r, prepared.h1));
        v.load_flow = v.load_flow + v.voltage_noise_ground.chain((v.voltage_noise_ground.value / prepared.r, prepared.h1));
        v.a_b_flow = v.a_b_flow + v.voltage_source_ground.scale(prepared.k);
        v.source_flow = v.source_flow + v.voltage_source_ground;
        v.source_ground_flow = v.source_ground_flow + v.voltage_source_ground;
        let current = at.unknown::<R>(shape.short);
        stamp.add_residual(0, current);
        stamp.add_residual(shape.inner, -current);
        stamp.add_conductance(shape.a_by_short, R::constant(1.0));
        stamp.add_conductance(shape.inner_by_short, R::constant(-1.0));
        stamp.add_residual(shape.short, at.unknown::<R>(0) - at.unknown::<R>(shape.inner) - v.short_potential);
        stamp.add_conductance(shape.short_by_a, R::constant(1.0));
        stamp.add_conductance(shape.short_by_inner, R::constant(-1.0));
        let t0000 = v.source_flow.value;
        stamp.add_residual(shape.noise, t0000);
        let [t0001] = v.source_flow.gradient;
        stamp.add_conductance(shape.noise_by_source, t0001);
        let t0002 = v.load_flow.value;
        stamp.add_residual(shape.noise, t0002);
        let [t0003] = v.load_flow.gradient;
        stamp.add_conductance(shape.noise_by_noise, t0003);
        let t0004 = v.inner_b_flow.value;
        stamp.add_residual(shape.inner, t0004);
        stamp.add_residual(1, -t0004);
        let [t0005] = v.inner_b_flow.gradient;
        stamp.add_conductance(shape.inner_by_inner, t0005);
        stamp.add_conductance(shape.inner_by_b, -t0005);
        stamp.add_conductance(shape.b_by_inner, -t0005);
        stamp.add_conductance(shape.b_by_b, t0005);
        let t0006 = v.joined_b_flow.value;
        stamp.add_residual(shape.joined, t0006);
        stamp.add_residual(1, -t0006);
        let [t0007] = v.joined_b_flow.gradient;
        stamp.add_conductance(shape.joined_by_joined, t0007);
        stamp.add_conductance(shape.joined_by_b, -t0007);
        stamp.add_conductance(shape.b_by_joined, -t0007);
        stamp.add_conductance(shape.b_by_b, t0007);
        if shape.active[0] {
            let t0008 = v.lower_ground_flow.value;
            stamp.add_residual(shape.lower, t0008);
            let [t0009] = v.lower_ground_flow.gradient;
            stamp.add_conductance(shape.lower_by_lower, t0009);
        }
        let t0010 = v.a_upper_flow.value;
        stamp.add_residual(0, t0010);
        stamp.add_residual(shape.upper, -t0010);
        let [t0011] = v.a_upper_flow.gradient;
        stamp.add_conductance(shape.a_by_a, t0011);
        stamp.add_conductance(shape.a_by_upper, -t0011);
        stamp.add_conductance(shape.upper_by_a, -t0011);
        stamp.add_conductance(shape.upper_by_upper, t0011);
        let t0012 = v.a_b_flow.value;
        stamp.add_residual(0, t0012);
        stamp.add_residual(1, -t0012);
        let [t0013] = v.a_b_flow.gradient;
        stamp.add_conductance(shape.a_by_source, t0013);
        stamp.add_conductance(shape.b_by_source, -t0013);
        let t0014 = v.source_ground_flow.value;
        stamp.add_residual(shape.source, t0014);
        let [t0015] = v.source_ground_flow.gradient;
        stamp.add_conductance(shape.source_by_source, t0015);
        memory.shorted = v.shorted.value.value();
    }

    fn noise<R: Real>(
        shape: &NetworksShape,
        prepared: &PreparedNetworks<R>,
        _: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        sources.push(NoiseSource::white("source", [Some(UnknownIndex::new(shape.source)), None], prepared.s.value()));
    }
}
