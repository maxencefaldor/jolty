//! A kernel's instance as a device of its own, bound to the circuit's unknowns, for wherever
//! the engine still makes devices one by one: the analyses that read a device apart from
//! its group, the devices made again to be read, those that compute their derivatives with
//! respect to a parameter, and those that heat.

use std::fmt;

use jolty_numerics::{Entry, UNMAPPED};

use super::{Kernel, Layout, Stamp};
use crate::{
    Binding, Device, Heatable, HeatableModel, Input, Instance, Interface, NativeDevice,
    NativeModel, Node, NoiseSource, Output, ParameterError, Real,
};

/// One instance of the kernel `K`, computing with `R`, bound to the circuit's unknowns: it
/// evaluates the kernel into a block of its own, then adds the block to the circuit's
/// equations, each local unknown's and each local entry's sum at once.
pub struct KernelDevice<K: Kernel, R: Real = f64> {
    shape: K::Shape,
    prepared: K::Prepared<R>,
    memory: K::Memory,
    /// The index among the circuit's unknowns of each local unknown, or
    /// [`UNMAPPED`] for one on the reference.
    map: Vec<u32>,
    /// The circuit's node of each local unknown.
    nodes: Vec<Node>,
    /// The circuit's entry of each local entry, or none where it joins the reference.
    entries: Vec<Option<Entry>>,
    /// The residuals of the local unknowns, their scales and their charges, then the
    /// conductances and the capacitances of the local entries, of the last evaluation, one
    /// after another: empty until the first, as the devices an analysis reads apart from
    /// their groups are never evaluated.
    block: Vec<R>,
}

impl<K: Kernel, R: Real> fmt::Debug for KernelDevice<K, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(K::NAME)
            .field("shape", &self.shape)
            .field("prepared", &self.prepared)
            .field("memory", &self.memory)
            .field("nodes", &self.nodes)
            .finish_non_exhaustive()
    }
}

/// The index among the circuit's unknowns of `node`, or [`UNMAPPED`] for the reference.
pub(crate) fn mapped(node: Node) -> u32 {
    node.and_then(|unknown| u32::try_from(unknown.index()).ok())
        .unwrap_or(UNMAPPED)
}

impl<K: Kernel, R: Real> NativeDevice<R> for KernelDevice<K, R> {
    const NAME: &'static str = K::NAME;
    type Parameters = K::Parameters<R>;

    fn interface(parameters: &K::Parameters<R>) -> Interface {
        K::interface(parameters)
    }

    fn new(
        parameters: K::Parameters<R>,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let prepared = K::prepare(&parameters, R::constant(binding.temperature()))?;
        let mut layout = Layout::new(binding.terminals().len());
        let shape = K::layout(&parameters, &mut layout)?;
        let mut nodes = binding.terminals().to_vec();
        for &(label, quantity) in layout.internals() {
            nodes.push(Some(binding.internal(label, quantity)));
        }
        let node = |local: usize| nodes.get(local).copied().flatten();
        let entries = layout
            .entries()
            .iter()
            .map(|&(row, column)| binding.entry(node(row), node(column)))
            .collect::<Vec<_>>();
        Ok(Self {
            shape,
            prepared,
            memory: K::Memory::default(),
            map: nodes.iter().map(|&node| mapped(node)).collect(),
            block: Vec::new(),
            nodes,
            entries,
        })
    }
}

impl<K: Kernel, R: Real> Heatable<R> for KernelDevice<K, R> {
    fn heat(
        &mut self,
        parameters: &K::Parameters<R>,
        temperature: R,
    ) -> Result<(), ParameterError> {
        self.prepared = K::prepare(parameters, temperature)?;
        K::reheat(&mut self.memory);
        Ok(())
    }
}

impl<K: Kernel, R: Real> Device<R> for KernelDevice<K, R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        let at = input.local(&self.map);
        let (unknowns, stored) = (self.nodes.len(), self.entries.len());
        self.block.resize(3 * unknowns + 2 * stored, R::default());
        let Some([residual, scale, charge, conductance, capacitance]) =
            split(&mut self.block, [unknowns, unknowns, unknowns, stored])
        else {
            return;
        };
        let mut stamp = Stamp::zeroed(
            [&mut *residual, &mut *scale, &mut *charge],
            [&mut *conductance, &mut *capacitance],
        );
        K::evaluate(
            &self.shape,
            &self.prepared,
            &mut self.memory,
            &at,
            &mut stamp,
        );
        if stamp.is_limited() {
            output.mark_limited();
        }
        let rows = residual.iter().zip(scale.iter()).zip(charge.iter());
        for (&node, ((&residual, &scale), &charge)) in self.nodes.iter().zip(rows) {
            output.add_residual_scaled(node, residual, scale);
            output.add_charge(node, charge);
        }
        let stored = conductance.iter().zip(capacitance.iter());
        for (&entry, (&conductance, &capacitance)) in self.entries.iter().zip(stored) {
            output.add_conductance(entry, conductance);
            output.add_capacitance(entry, capacitance);
        }
    }

    fn variable_names(&self) -> &'static [&'static str] {
        K::variable_names()
    }

    fn variables(&self, values: &mut [f64]) {
        K::variables(&self.memory, values);
    }

    fn noise(&self, input: &Input<'_>, sources: &mut Vec<NoiseSource>) {
        let start = sources.len();
        K::noise(
            &self.shape,
            &self.prepared,
            &input.local(&self.map),
            sources,
        );
        for source in sources.iter_mut().skip(start) {
            source.equations = source.equations.map(|equation| {
                equation.and_then(|local| self.nodes.get(local.index()).copied().flatten())
            });
        }
    }
}

/// `block` cut into five, the first four of `lengths`, or `None` when it is too short.
fn split<R>(block: &mut [R], lengths: [usize; 4]) -> Option<[&mut [R]; 5]> {
    let [first, second, third, fourth] = lengths;
    let (one, rest) = block.split_at_mut_checked(first)?;
    let (two, rest) = rest.split_at_mut_checked(second)?;
    let (three, rest) = rest.split_at_mut_checked(third)?;
    let (four, five) = rest.split_at_mut_checked(fourth)?;
    Some([one, two, three, four, five])
}

impl<K: Kernel> NativeModel for K {
    type Device<R: Real> = KernelDevice<K, R>;

    fn instance(parameters: &K::Parameters<f64>) -> Option<Instance<'_>> {
        Some(Instance::of::<K>(parameters))
    }
}

impl<K: Kernel> HeatableModel for K {
    type Device<R: Real> = KernelDevice<K, R>;

    fn instance(parameters: &K::Parameters<f64>) -> Option<Instance<'_>> {
        Some(Instance::of::<K>(parameters))
    }
}
