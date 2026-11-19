use jolty_numerics::Pattern;

use crate::{Local, Node, Past, Real};

/// Evaluation state passed to device models.
#[derive(Debug, Clone, Copy)]
pub struct Input<'a> {
    x: &'a [f64],
    /// Simulation time, in seconds; zero in every non-transient analysis.
    pub time: f64,
    /// Flag indicating active transient time stepping.
    ///
    /// True during transient analysis and false at operating points or small-signal
    /// analyses, where time-driven states such as memristors remain fixed at initial values.
    pub transient: bool,
    /// Scale factor applied to all independent sources: `1.0`, except during source-stepping
    /// continuation for operating-point convergence.
    pub source_scale: f64,
    /// Minimum shunt conductance placed across nonlinear junctions, in siemens, preventing
    /// singular nodes in the linearized circuit.
    pub gmin: f64,
    /// Flag enabling per-iteration junction voltage-step limiting.
    ///
    /// False on the initial evaluation of a DC solve starting from an initial guess.
    pub limiting: bool,
    /// Historical unknown values accepted at earlier transient steps for delay lines, read
    /// through [`Local::delayed`]; `None` outside transient analysis.
    pub past: Option<&'a Past>,
}

impl Input<'static> {
    /// Zero evaluation state at time zero with full source scaling, zero `gmin`, and
    /// limiting disabled.
    ///
    /// Linear kernels evaluate entries at this state. The state slice is empty, so callers
    /// read zero through [`Input::local`].
    pub(crate) const ZERO: Self = Self {
        x: &[],
        time: 0.0,
        transient: false,
        source_scale: 1.0,
        gmin: 0.0,
        limiting: false,
        past: None,
    };
}

impl<'a> Input<'a> {
    /// Constructs an evaluation state for solution vector `x`, optional transient `time`,
    /// and junction conductance `gmin` with full source scaling and limiting enabled.
    #[must_use]
    pub fn new(x: &'a [f64], time: Option<f64>, gmin: f64) -> Self {
        Self {
            x,
            time: time.unwrap_or(0.0),
            transient: time.is_some(),
            source_scale: 1.0,
            gmin,
            limiting: true,
            past: None,
        }
    }

    /// Returns the solution value at `node`, or `0.0` for ground.
    #[must_use]
    pub fn value(&self, node: Node) -> f64 {
        node.map_or(0.0, |unknown| self.x[unknown.index()])
    }

    /// Builds a [`Local`] view mapping local unknown indices through `map` into `self`.
    pub(crate) const fn local<'b>(&'b self, map: &'b [u32]) -> Local<'b> {
        Local::new(self.x, map, self)
    }
}

/// Device contributions accumulated into the circuit equations during one evaluation.
///
/// Residuals, residual scales, and charges hold one entry per unknown; conductances and
/// capacitances hold one entry per stored sparsity position. Alongside each residual, `scale`
/// tracks the largest term magnitude for convergence checks. Values are plain floats or
/// carry parameter derivatives under [`crate::Gradient`].
#[derive(Debug, Clone, PartialEq)]
pub struct Contributions<R: Real = f64> {
    /// Residual vector `f(x)` for every circuit unknown.
    pub residual: Vec<R>,
    /// Largest magnitude of any term summed into each residual entry.
    pub scale: Vec<R>,
    /// Charge or flux vector `q(x)` for every circuit unknown.
    pub charge: Vec<R>,
    /// Conductance Jacobian entries `∂f/∂x`, indexed by stored matrix position.
    pub conductance: Vec<R>,
    /// Capacitance Jacobian entries `∂q/∂x`, indexed by stored matrix position.
    pub capacitance: Vec<R>,
}

impl<R: Real> Contributions<R> {
    /// Allocates zero-initialized contributions sized to `pattern`.
    #[must_use]
    pub fn new(pattern: &Pattern) -> Self {
        let (unknowns, stored) = (pattern.dimension(), pattern.len());
        Self {
            residual: vec![R::default(); unknowns],
            scale: vec![R::default(); unknowns],
            charge: vec![R::default(); unknowns],
            conductance: vec![R::default(); stored],
            capacitance: vec![R::default(); stored],
        }
    }

    /// Zeroes every contribution buffer.
    pub fn clear(&mut self) {
        for values in [
            &mut self.residual,
            &mut self.scale,
            &mut self.charge,
            &mut self.conductance,
            &mut self.capacitance,
        ] {
            values.fill(R::default());
        }
    }
}
