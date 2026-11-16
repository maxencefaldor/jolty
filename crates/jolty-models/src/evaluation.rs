use jolty_numerics::Pattern;

use crate::{Local, Node, Past, Real};

/// The state a device is evaluated at.
#[derive(Debug, Clone, Copy)]
pub struct Input<'a> {
    x: &'a [f64],
    /// The time, in seconds; zero in every analysis but a transient one.
    pub time: f64,
    /// Whether time passes: true while a transient analysis steps through it, and false at
    /// an operating point and in the analyses about one, in which a state that only time
    /// moves, such as a memristor's, holds the value it starts from.
    pub transient: bool,
    /// The factor every independent source is scaled by: 1, except while an operating point is
    /// found by ramping the sources up from zero.
    pub source_scale: f64,
    /// The conductance placed across every nonlinear junction, in siemens, so no node of the
    /// linearized circuit is ever left without a path to ground.
    pub gmin: f64,
    /// Whether devices limit each voltage's step from their last evaluation's: false for
    /// the first evaluation of a DC solution, which starts afresh from its guess.
    pub limiting: bool,
    /// What the unknowns a delay line reads took at the points a transient accepted, as
    /// [`Local::delayed`] reads them: none outside a transient, where the past is the
    /// present.
    pub past: Option<&'a Past>,
}

impl Input<'static> {
    /// Every unknown zero, at time zero in an operating point, with sources at full scale,
    /// no gmin across junctions and no step limited: the state a linear instance's entries
    /// are evaluated at, as at any other. It holds no unknown, so it is read only through
    /// [`Input::local`], which reads zero for each.
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
    /// The state `x` at `time` in a transient analysis or, with `None`, at an operating
    /// point or about one, at time zero, with sources at full scale, `gmin` across
    /// junctions, and steps limited.
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

    /// The value of `node`'s unknown, or zero for the reference node.
    #[must_use]
    pub fn value(&self, node: Node) -> f64 {
        node.map_or(0.0, |unknown| self.x[unknown.index()])
    }

    /// The state as an instance of a kernel sees it, its local unknowns the unknowns
    /// `map` gives, as [`Local`] reads them.
    pub(crate) const fn local<'b>(&'b self, map: &'b [u32]) -> Local<'b> {
        Local::new(self.x, map, self)
    }
}

/// What devices add to the circuit's equations in one evaluation.
///
/// Residuals, their scales and charges hold one value per unknown; conductances and
/// capacitances one per stored position of the pattern. Alongside each residual, `scale`
/// keeps the largest magnitude of a term added to it, against which convergence is judged.
/// The numbers are plain, or carry their derivative with respect to a parameter when the
/// devices compute with [`crate::Gradient`].
#[derive(Debug, Clone, PartialEq)]
pub struct Contributions<R: Real = f64> {
    /// The residual of each unknown's equation, `f(x)`.
    pub residual: Vec<R>,
    /// The largest magnitude of a term of each residual.
    pub scale: Vec<R>,
    /// The charge of each unknown's equation, `q(x)`.
    pub charge: Vec<R>,
    /// `∂f/∂x`, by stored position.
    pub conductance: Vec<R>,
    /// `∂q/∂x`, by stored position.
    pub capacitance: Vec<R>,
}

impl<R: Real> Contributions<R> {
    /// Zeroed contributions over `pattern`.
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

    /// Zeroes every contribution.
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
