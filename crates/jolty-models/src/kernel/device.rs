//! A kernel's instance as a device of its own, bound to the circuit's unknowns, for wherever
//! the engine still makes devices one by one: the analyses that read a device apart from
//! its group, the devices made again to be read, those that compute their derivatives with
//! respect to a parameter, and those that heat.

use std::fmt;

use jolty_numerics::{Entry, UNMAPPED};

use super::{Kernel, Layout, Stamp};
use crate::Gradient;
use crate::heat::HeatedKernel;
use crate::{
    Binding, Choice, Device, Drive, Heatable, HeatableModel, Input, Instance, Interface, Level,
    NativeDevice, NativeModel, Node, NoiseSource, Output, ParameterError, Real, Source, Unknown,
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
    /// When the instance, a logic driver, settles from its last drive.
    settles: f64,
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
    const FOUNDRY: bool = K::FOUNDRY;
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
        let shape = K::layout(&parameters, &prepared, &mut layout)?;
        let mut nodes = binding.terminals().to_vec();
        for (label, quantity) in layout.internals() {
            nodes.push(Some(binding.internal(label, *quantity)));
        }
        let node = |local: usize| nodes.get(local).copied().flatten();
        let entries = layout
            .entries()
            .iter()
            .map(|&(row, column)| binding.entry(node(row), node(column)))
            .collect::<Vec<_>>();
        let memory = K::memory(&shape, &prepared);
        Ok(Self {
            shape,
            memory,
            prepared,
            map: nodes.iter().map(|&node| mapped(node)).collect(),
            block: Vec::new(),
            nodes,
            entries,
            settles: 0.0,
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

    fn breakpoints(&self, start: f64, stop: f64, times: &mut Vec<f64>) {
        K::breakpoints(&self.prepared, (start, stop), times);
    }

    fn choices(&mut self, input: &Input<'_>, found: &mut Vec<Choice>) {
        let at = input.local(&self.map);
        K::choices(&self.shape, &self.prepared, &mut self.memory, &at, found);
    }

    fn as_source(&mut self) -> Option<&mut dyn Source> {
        K::level(&self.prepared).is_some().then_some(self)
    }

    fn as_driver(&mut self) -> Option<&mut dyn Drive> {
        K::driven(&self.shape).is_some().then_some(self)
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

impl<K: Kernel, R: Real> Source for KernelDevice<K, R> {
    fn level(&self) -> f64 {
        K::level(&self.prepared).unwrap_or(0.0)
    }

    fn level_derivative(&self, derivative: &mut [f64]) {
        let (unknowns, stored) = (self.nodes.len(), self.entries.len().max(1));
        let mut block = vec![R::default(); 3 * unknowns + 2 * stored];
        let Some([residual, scale, charge, conductance, capacitance]) =
            split(&mut block, [unknowns, unknowns, unknowns, stored])
        else {
            return;
        };
        let mut stamp = Stamp::zeroed([residual, scale, charge], [conductance, capacitance]);
        K::excite(&self.shape, &mut stamp);
        for (&node, value) in self.nodes.iter().zip(block.iter()) {
            if let Some(sum) = node.and_then(|unknown| derivative.get_mut(unknown.index())) {
                *sum += value.value();
            }
        }
    }
}

impl<K: Kernel, R: Real> Drive for KernelDevice<K, R> {
    fn drive(&mut self, time: f64, level: Level, transition: f64) {
        if let Some(settles) = K::drive(&mut self.memory, (time, level, transition)) {
            self.settles = settles;
        }
    }

    fn settles_at(&self) -> f64 {
        self.settles
    }

    fn current(&self) -> Unknown {
        K::driven(&self.shape)
            .and_then(|local| self.nodes.get(local).copied().flatten())
            .unwrap_or(Unknown::new(0))
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

    fn instance(parameters: &K::Parameters<f64>) -> Instance<'_> {
        Instance::of::<K>(parameters)
    }
}

impl<K: Kernel> HeatableModel for K {
    type Device<R: Real> = KernelDevice<K, R>;

    fn instance(parameters: &K::Parameters<f64>) -> Instance<'_> {
        Instance::of::<K>(parameters)
    }

    fn heated(parameters: &K::Parameters<Gradient>) -> Instance<'_> {
        Instance::of_kind::<HeatedKernel<K>>(parameters)
    }
}
