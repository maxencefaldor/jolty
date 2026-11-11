use jolty_numerics::Entry;
use serde::Deserialize;

use crate::dual::Dual;
use crate::jfet::Channel;
use crate::junction::{
    Depletion, ELEMENTARY_CHARGE, TEMPERATURE, THERMAL_VOLTAGE, below_one, critical_voltage,
    depletion_charge, junction_current, limit_drain, limit_gate, limit_junction,
};
use crate::stamp::Stamp;
use crate::{
    Binding, Branch, BranchKind, Device, Input, Interface, NativeDevice, Node, Output,
    ParameterError, Quantity, Unknown,
};

/// The vacuum permittivity of SPICE's MOSFET models, in farads per meter: SPICE2's value,
/// with which models are fitted, rather than CODATA's 8.8541878128e-12.
const VACUUM_PERMITTIVITY: f64 = 8.854_214_871e-12;
/// The permittivity of silicon dioxide, in farads per meter.
const OXIDE_PERMITTIVITY: f64 = 3.9 * VACUUM_PERMITTIVITY;
/// The permittivity of silicon, in farads per meter.
const SILICON_PERMITTIVITY: f64 = 11.7 * VACUUM_PERMITTIVITY;
/// Silicon's intrinsic carrier concentration, per cubic meter.
const INTRINSIC_CONCENTRATION: f64 = 1.45e16;
/// The least saturation voltage Meyer's capacitances take, in volts.
const MEYER_SATURATION: f64 = 0.025;

/// The material of a MOSFET's gate, which sets its work function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateMaterial {
    /// Polysilicon doped opposite to the substrate, as an n⁺ gate over a p substrate:
    /// SPICE's `tpg = 1`.
    OppositeDoping,
    /// Polysilicon doped like the substrate: SPICE's `tpg = −1`.
    SameDoping,
    /// Aluminum: SPICE's `tpg = 0`.
    Aluminum,
}

/// A metal-oxide-semiconductor field-effect transistor, in SPICE's level 1
/// (Shichman-Hodges) model with Meyer's gate capacitances, as ngspice implements it.
///
/// The drain current follows the square law: with `vgst = vgs − von` and
/// `β' = β · (1 + λ · vds)`, it is `β' · vgst² / 2` in saturation, where `vds ≥ vgst`, and
/// `β' · vds · (vgst − vds / 2)` below it, with `β = kp · w / (l − 2 · ld)`. The threshold
/// `von = vto + γ · (√(φ − vbs) − √φ)` rises as the source rises above the bulk. The
/// transistor is symmetric: with the drain below the source, the two swap.
///
/// Two junctions join the bulk to the drain and the source, each carrying a
/// [junction current](crate::junction) and holding a depletion charge, bottom and
/// sidewall. The gate holds overlap charges against the source, the drain and the bulk,
/// and, given an oxide thickness, Meyer's capacitances across the oxide. These depend on
/// every terminal voltage and no charge yields them, so each carries a current
/// `C(v) · dv/dt`, which follows an unknown of the transistor's own: the displacement
/// current through the whole oxide, `Cox · dv/dt`, labelled `gate_source_displacement`,
/// `gate_drain_displacement` or `gate_bulk_displacement`. A drain or source resistance adds
/// an internal node, labelled `drain` or `source`.
#[derive(Debug)]
pub struct Mosfet {
    stamp: Stamp<6>,
    sign: f64,
    drain_conductance: f64,
    source_conductance: f64,
    channel: SquareLaw,
    /// The bulk-drain and bulk-source junctions.
    junctions: [BulkJunction; 2],
    /// The gate-source, gate-drain and gate-bulk overlap capacitances.
    overlaps: [f64; 3],
    oxide: Option<Oxide>,
    /// The gate-source, drain-source and bulk-source voltages of the last evaluation, from
    /// which the next steps are limited, and its threshold.
    last: [f64; 3],
    threshold: f64,
}

/// Local node indices: the terminals, then the internal drain and source, which are the
/// terminals themselves without series resistance.
const GATE: usize = 0;
const DRAIN: usize = 1;
const SOURCE: usize = 2;
const BULK: usize = 3;
const INNER_DRAIN: usize = 4;
const INNER_SOURCE: usize = 5;

/// The pairs of nodes across which the voltages the transistor's equations take as
/// variables lie: gate-source, drain-source and bulk-source.
const VARIABLES: [(usize, usize); 3] = [
    (GATE, INNER_SOURCE),
    (INNER_DRAIN, INNER_SOURCE),
    (BULK, INNER_SOURCE),
];

/// The parameters of a [`Mosfet`], with SPICE's defaults.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MosfetParameters {
    /// The channel; default `n`.
    pub channel: Channel,
    /// The channel's width, `w`, in meters; default 100 µm.
    pub width: f64,
    /// The channel's drawn length, `l`, in meters; default 100 µm.
    pub length: f64,
    /// The number of identical transistors in parallel, `m`; default 1.
    pub multiplier: f64,
    /// The zero-bias threshold voltage, `vto`, in volts; default 0, or derived from the
    /// substrate doping.
    pub threshold_voltage: Option<f64>,
    /// The transconductance parameter, `kp`, in amperes per square volt; default 2e-5, or
    /// the surface mobility times the oxide capacitance per area, given an oxide thickness.
    pub transconductance_parameter: Option<f64>,
    /// The body-effect coefficient, `γ`, in square-root volts; default 0, or derived from
    /// the substrate doping.
    pub body_effect: Option<f64>,
    /// The surface potential, `φ`, in volts; default 0.6, or derived from the substrate
    /// doping.
    pub surface_potential: Option<f64>,
    /// The channel-length modulation, `λ`, per volt; default 0.
    pub channel_length_modulation: f64,
    /// The lateral diffusion, `ld`, by which the channel falls short of its drawn length at
    /// each end, in meters; default 0.
    pub lateral_diffusion: f64,
    /// The gate oxide's thickness, `tox`, in meters; default none, which leaves the gate
    /// without Meyer's capacitances.
    pub oxide_thickness: Option<f64>,
    /// The surface mobility, `uo`, in square meters per volt-second, from which `kp`
    /// derives; default 0.06.
    pub surface_mobility: f64,
    /// The substrate doping, `nsub`, per cubic meter, from which `vto`, `γ` and `φ` derive
    /// unless given; default none. It needs an oxide thickness.
    pub substrate_doping: Option<f64>,
    /// The surface state density, `nss`, per square meter, which shifts a derived `vto`;
    /// default 0.
    pub surface_state_density: f64,
    /// The gate's material, which sets a derived `vto`; default `opposite_doping`.
    pub gate_material: GateMaterial,
    /// The drain resistance, in ohms; default none, for the sheet resistance's.
    pub drain_resistance: Option<f64>,
    /// The source resistance, in ohms; default none, for the sheet resistance's.
    pub source_resistance: Option<f64>,
    /// The drain and source diffusions' sheet resistance, in ohms per square; default 0.
    pub sheet_resistance: f64,
    /// The drain diffusion's length in squares, `nrd`; default 1.
    pub drain_squares: f64,
    /// The source diffusion's length in squares, `nrs`; default 1.
    pub source_squares: f64,
    /// The bulk junctions' saturation current, in amperes; default 1e-14.
    pub saturation_current: f64,
    /// The bulk junctions' saturation current per area, in amperes per square meter, which
    /// replaces the saturation current when it and both areas are nonzero; default 0.
    pub saturation_current_density: f64,
    /// The drain diffusion's area, in square meters; default 0.
    pub drain_area: f64,
    /// The source diffusion's area, in square meters; default 0.
    pub source_area: f64,
    /// The drain diffusion's perimeter, in meters; default 0.
    pub drain_perimeter: f64,
    /// The source diffusion's perimeter, in meters; default 0.
    pub source_perimeter: f64,
    /// The bulk junctions' potential, `pb`, in volts; default 0.8.
    pub junction_potential: f64,
    /// The zero-bias bulk-drain capacitance, in farads; default none, for the bottom
    /// capacitance's.
    pub bulk_drain_capacitance: Option<f64>,
    /// The zero-bias bulk-source capacitance, in farads; default none, for the bottom
    /// capacitance's.
    pub bulk_source_capacitance: Option<f64>,
    /// The bulk junctions' zero-bias bottom capacitance per area, `cj`, in farads per
    /// square meter; default 0.
    pub bottom_capacitance: f64,
    /// The bulk junctions' zero-bias sidewall capacitance per length of perimeter, `cjsw`,
    /// in farads per meter; default 0.
    pub sidewall_capacitance: f64,
    /// The bottom capacitance's grading coefficient, `mj`, below 1; default 0.5.
    pub bottom_grading: f64,
    /// The sidewall capacitance's grading coefficient, `mjsw`, below 1; default 0.5.
    pub sidewall_grading: f64,
    /// The fraction of the junction potential above which the depletion capacitances are
    /// extended linearly, below 1; default 0.5.
    pub depletion_coefficient: f64,
    /// The gate-source overlap capacitance per width, in farads per meter; default 0.
    pub gate_source_overlap: f64,
    /// The gate-drain overlap capacitance per width, in farads per meter; default 0.
    pub gate_drain_overlap: f64,
    /// The gate-bulk overlap capacitance per length, in farads per meter; default 0.
    pub gate_bulk_overlap: f64,
}

impl Default for MosfetParameters {
    fn default() -> Self {
        Self {
            channel: Channel::N,
            width: 100e-6,
            length: 100e-6,
            multiplier: 1.0,
            threshold_voltage: None,
            transconductance_parameter: None,
            body_effect: None,
            surface_potential: None,
            channel_length_modulation: 0.0,
            lateral_diffusion: 0.0,
            oxide_thickness: None,
            surface_mobility: 0.06,
            substrate_doping: None,
            surface_state_density: 0.0,
            gate_material: GateMaterial::OppositeDoping,
            drain_resistance: None,
            source_resistance: None,
            sheet_resistance: 0.0,
            drain_squares: 1.0,
            source_squares: 1.0,
            saturation_current: 1e-14,
            saturation_current_density: 0.0,
            drain_area: 0.0,
            source_area: 0.0,
            drain_perimeter: 0.0,
            source_perimeter: 0.0,
            junction_potential: 0.8,
            bulk_drain_capacitance: None,
            bulk_source_capacitance: None,
            bottom_capacitance: 0.0,
            sidewall_capacitance: 0.0,
            bottom_grading: 0.5,
            sidewall_grading: 0.5,
            depletion_coefficient: 0.5,
            gate_source_overlap: 0.0,
            gate_drain_overlap: 0.0,
            gate_bulk_overlap: 0.0,
        }
    }
}

/// The process parameters, given or derived: `vto`, `kp`, `γ`, `φ`, and the oxide
/// capacitance per area, if any.
#[derive(Debug, Clone, Copy)]
struct Process {
    threshold: f64,
    transconductance: f64,
    body_effect: f64,
    surface_potential: f64,
    oxide: Option<f64>,
}

impl MosfetParameters {
    /// The process parameters, derived from the oxide thickness and the substrate doping
    /// where not given, as SPICE derives them.
    fn process(&self) -> Result<Process, ParameterError> {
        let oxide = self
            .oxide_thickness
            .map(|thickness| ParameterError::positive("oxide_thickness", thickness))
            .transpose()?
            .map(|thickness| OXIDE_PERMITTIVITY / thickness);
        let mobility = ParameterError::positive("surface_mobility", self.surface_mobility)?;
        let transconductance = self
            .transconductance_parameter
            .or(oxide.map(|oxide| mobility * oxide))
            .unwrap_or(2e-5);
        let mut process = Process {
            threshold: self.threshold_voltage.unwrap_or(0.0),
            transconductance,
            body_effect: self.body_effect.unwrap_or(0.0),
            surface_potential: self.surface_potential.unwrap_or(0.6),
            oxide,
        };
        if let Some(doping) = self.substrate_doping {
            let oxide = oxide.ok_or_else(|| {
                ParameterError("substrate_doping needs an oxide_thickness".to_owned())
            })?;
            self.derive(&mut process, doping, oxide)?;
        }
        ParameterError::non_negative("transconductance_parameter", process.transconductance)?;
        ParameterError::non_negative("body_effect", process.body_effect)?;
        ParameterError::positive("surface_potential", process.surface_potential)?;
        Ok(process)
    }

    /// Derives what was not given of `φ`, `γ` and `vto` from the substrate `doping` and the
    /// `oxide` capacitance per area.
    fn derive(&self, process: &mut Process, doping: f64, oxide: f64) -> Result<(), ParameterError> {
        if doping <= INTRINSIC_CONCENTRATION {
            return Err(ParameterError(format!(
                "substrate_doping must exceed silicon's intrinsic concentration, \
                 {INTRINSIC_CONCENTRATION:e} per cubic meter, not {doping:?}"
            )));
        }
        let sign = self.channel.sign();
        let potential = self.surface_potential.unwrap_or_else(|| {
            (2.0 * THERMAL_VOLTAGE * (doping / INTRINSIC_CONCENTRATION).ln()).max(0.1)
        });
        let body_effect = self.body_effect.unwrap_or_else(|| {
            (2.0 * SILICON_PERMITTIVITY * ELEMENTARY_CHARGE * doping).sqrt() / oxide
        });
        // Silicon's band gap, in electronvolts.
        let gap = 1.16 - 7.02e-4 * TEMPERATURE * TEMPERATURE / (TEMPERATURE + 1108.0);
        let gate_work_function = match self.gate_material {
            GateMaterial::Aluminum => 3.2,
            GateMaterial::OppositeDoping => 3.25 + 0.5 * gap - sign * 0.5 * gap,
            GateMaterial::SameDoping => 3.25 + 0.5 * gap + sign * 0.5 * gap,
        };
        let work_function_difference =
            gate_work_function - (3.25 + 0.5 * gap + sign * 0.5 * potential);
        let flat_band =
            work_function_difference - self.surface_state_density * ELEMENTARY_CHARGE / oxide;
        process.surface_potential = potential;
        process.body_effect = body_effect;
        process.threshold = self
            .threshold_voltage
            .unwrap_or_else(|| flat_band + sign * (body_effect * potential.sqrt() + potential));
        Ok(())
    }
}

/// The channel's square law, oriented as an n-channel's.
#[derive(Debug, Clone, Copy)]
struct SquareLaw {
    beta: f64,
    lambda: f64,
    body_effect: f64,
    surface_potential: f64,
    /// `√φ`.
    root_potential: f64,
    /// The threshold less the body effect at zero bias, `vto − γ · √φ`.
    built_in: f64,
}

/// What the square law yields at one bias.
#[derive(Debug, Clone, Copy)]
struct Conduction {
    /// The current from drain to source.
    current: Dual<3>,
    /// The threshold, `von`.
    threshold: Dual<3>,
    /// The saturation voltage, `max(vgst, 0)`.
    saturation: Dual<3>,
    /// Whether the drain is at or above the source; otherwise the two swap.
    forward: bool,
}

impl SquareLaw {
    /// The drain current at the gate-source, drain-source and bulk-source voltages.
    fn evaluate(&self, [vgs, vds, vbs]: [Dual<3>; 3]) -> Conduction {
        let forward = vds.value >= 0.0;
        // With the drain below the source, the drain acts as the source.
        let (vgs, vds, vbs) = if forward {
            (vgs, vds, vbs)
        } else {
            (vgs - vds, -vds, vbs - vds)
        };
        let root = if vbs.value <= 0.0 {
            (self.surface_potential - vbs).sqrt()
        } else {
            (self.root_potential - vbs / (2.0 * self.root_potential)).max(0.0)
        };
        let threshold = root * self.body_effect + self.built_in;
        let vgst = vgs - threshold;
        let current = if vgst.value <= 0.0 {
            Dual::constant(0.0)
        } else {
            let betap = (vds * self.lambda + 1.0) * self.beta;
            if vgst.value <= vds.value {
                betap * vgst * vgst * 0.5
            } else {
                betap * vds * (vgst - vds * 0.5)
            }
        };
        Conduction {
            current: if forward { current } else { -current },
            threshold,
            saturation: vgst.max(0.0),
            forward,
        }
    }
}

/// A junction between the bulk and the drain or the source.
#[derive(Debug, Clone, Copy)]
struct BulkJunction {
    saturation_current: f64,
    critical_voltage: f64,
    bottom: Option<Depletion>,
    sidewall: Option<Depletion>,
}

impl BulkJunction {
    /// The junction's current, with `gmin` across it, and its charge, at `voltage`.
    fn evaluate(&self, voltage: Dual<3>, gmin: f64) -> (Dual<3>, Dual<3>) {
        let current = voltage.chain(junction_current(
            voltage.value,
            self.saturation_current,
            THERMAL_VOLTAGE,
        )) + voltage * gmin;
        let charge = voltage.chain(depletion_charge(self.bottom.as_ref(), voltage.value))
            + voltage.chain(depletion_charge(self.sidewall.as_ref(), voltage.value));
        (current, charge)
    }
}

/// The gate oxide, across which Meyer's capacitances lie.
#[derive(Debug, Clone, Copy)]
struct Oxide {
    /// The oxide's whole capacitance, `Cox`.
    capacitance: f64,
    surface_potential: f64,
    /// The displacement currents towards the source, the drain and the bulk.
    displacements: [Displacement; 3],
}

impl Oxide {
    /// Meyer's capacitances from the gate to the source, the drain and the bulk, at the
    /// gate-source and gate-drain voltages, with the drain at or above the source, and the
    /// channel's threshold and saturation voltage.
    fn meyer(&self, vgs: Dual<3>, vgd: Dual<3>, conduction: &Conduction) -> [Dual<3>; 3] {
        let (oxide, potential) = (self.capacitance, self.surface_potential);
        let zero = Dual::constant(0.0);
        let vgst = vgs - conduction.threshold;
        // Accumulation, then depletion: the oxide faces the bulk alone.
        if vgst.value <= -potential {
            return [zero, zero, Dual::constant(oxide)];
        }
        let bulk = if vgst.value <= 0.0 {
            vgst * (-oxide / potential)
        } else {
            zero
        };
        if vgst.value <= -potential / 2.0 {
            return [zero, zero, bulk];
        }
        // Inversion: the channel takes two thirds of the oxide, all on the source's side
        // in saturation, and shared with the drain below it.
        let channel = if vgst.value <= 0.0 {
            (vgst * (2.0 / potential) + 1.0) * (2.0 * oxide / 3.0)
        } else {
            Dual::constant(2.0 * oxide / 3.0)
        };
        let saturation = conduction.saturation.max(MEYER_SATURATION);
        let vds = vgs - vgd;
        if vds.value >= saturation.value {
            return [channel, zero, bulk];
        }
        let sum = saturation * 2.0 - vds;
        let difference = saturation - vds;
        let square = sum * sum;
        [
            channel * (1.0 - difference * difference / square),
            channel * (1.0 - saturation * saturation / square),
            bulk,
        ]
    }
}

/// The displacement current through the whole gate oxide towards one node,
/// `w = Cox · dv/dt`: an unknown of the transistor's own, which Meyer's current between the
/// gate and the node follows.
#[derive(Debug, Clone, Copy)]
struct Displacement {
    unknown: Unknown,
    /// The local index of the node across the oxide from the gate.
    node: usize,
    /// The entries at (unknown, unknown), (unknown, gate), (unknown, node), (gate, unknown)
    /// and (node, unknown).
    entries: [Option<Entry>; 5],
}

impl Displacement {
    /// Claims an unknown named `label` for the displacement towards `node`, the node at
    /// local index `index`, from `gate`.
    fn new(
        binding: &mut Binding<'_>,
        label: &str,
        gate: Node,
        (index, node): (usize, Node),
    ) -> Self {
        let unknown = binding.internal(label, Quantity::Displacement);
        let row = Some(unknown);
        Self {
            unknown,
            node: index,
            entries: [
                binding.entry(row, row),
                binding.entry(row, gate),
                binding.entry(row, node),
                binding.entry(gate, row),
                binding.entry(node, row),
            ],
        }
    }

    /// Adds the displacement's equation, `w − Cox · d(v_gate − v_node)/dt = 0`, for the
    /// `oxide`'s whole capacitance, and the derivatives with respect to `w` of Meyer's
    /// current from the gate to the node, `C · w / Cox`, for the Meyer `capacitance`.
    /// Returns `w`.
    fn add(
        &self,
        stamp: &Stamp<6>,
        (oxide, capacitance): (f64, f64),
        input: &Input<'_>,
        output: &mut Output<'_>,
    ) -> f64 {
        let row = Some(self.unknown);
        let [own, gate, node, from_gate, from_node] = self.entries;
        let displacement = input.value(row);
        let voltage = stamp.voltage(input, GATE) - stamp.voltage(input, self.node);
        output.add_residual(row, displacement);
        output.add_charge(row, -oxide * voltage);
        output.add_conductance(own, 1.0);
        output.add_capacitance(gate, -oxide);
        output.add_capacitance(node, oxide);
        output.add_conductance(from_gate, capacitance / oxide);
        output.add_conductance(from_node, -capacitance / oxide);
        displacement
    }
}

/// The terminal voltages one evaluation uses, oriented as an n-channel's: limited, and how
/// far the applied ones lie beyond them, along which every value is extended linearly.
#[derive(Debug, Clone, Copy)]
struct Bias {
    /// The gate-source, drain-source and bulk-source voltages, limited.
    voltages: [f64; 3],
    beyond: [f64; 3],
}

impl NativeDevice for Mosfet {
    const NAME: &'static str = "mosfet";
    type Parameters = MosfetParameters;

    fn interface(_: &MosfetParameters) -> Interface {
        Interface::new(
            &["drain", "gate", "source", "bulk"],
            &[
                Branch::new(0, 2, BranchKind::Conductive),
                Branch::new(3, 0, BranchKind::Conductive),
                Branch::new(3, 2, BranchKind::Conductive),
                Branch::new(1, 2, BranchKind::Capacitive),
                Branch::new(1, 0, BranchKind::Capacitive),
                Branch::new(1, 3, BranchKind::Capacitive),
            ],
        )
    }

    fn new(
        parameters: MosfetParameters,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let process = parameters.process()?;
        let geometry = Geometry::new(&parameters)?;
        let junctions = parameters.junctions(&geometry)?;
        let [drain_resistance, source_resistance] = parameters.series_resistances()?;
        let lambda = parameters.channel_length_modulation;
        let mut overlaps = [0.0; 3];
        for (overlap, (name, per_length, extent)) in overlaps.iter_mut().zip([
            (
                "gate_source_overlap",
                parameters.gate_source_overlap,
                geometry.width,
            ),
            (
                "gate_drain_overlap",
                parameters.gate_drain_overlap,
                geometry.width,
            ),
            (
                "gate_bulk_overlap",
                parameters.gate_bulk_overlap,
                geometry.lengths,
            ),
        ]) {
            *overlap = ParameterError::non_negative(name, per_length)? * extent;
        }

        let [drain, gate, source, bulk] = [0, 1, 2, 3].map(|index| binding.terminal(index));
        let mut inner = |terminal, resistance: f64, label| {
            if resistance > 0.0 {
                Some(binding.internal(label, Quantity::Voltage))
            } else {
                terminal
            }
        };
        let inner_drain = inner(drain, drain_resistance, "drain");
        let inner_source = inner(source, source_resistance, "source");
        let stamp = Stamp::new(
            binding,
            [gate, drain, source, bulk, inner_drain, inner_source],
        );
        let oxide = process.oxide.map(|per_area| {
            let mut displacement = |label, node| Displacement::new(binding, label, gate, node);
            Oxide {
                capacitance: per_area * geometry.area,
                surface_potential: process.surface_potential,
                displacements: [
                    displacement("gate_source_displacement", (INNER_SOURCE, inner_source)),
                    displacement("gate_drain_displacement", (INNER_DRAIN, inner_drain)),
                    displacement("gate_bulk_displacement", (BULK, bulk)),
                ],
            }
        });
        let conductance = |resistance: f64| {
            if resistance > 0.0 {
                1.0 / resistance
            } else {
                0.0
            }
        };
        let sign = parameters.channel.sign();
        let root_potential = process.surface_potential.sqrt();
        Ok(Self {
            stamp,
            sign,
            drain_conductance: conductance(drain_resistance),
            source_conductance: conductance(source_resistance),
            channel: SquareLaw {
                beta: process.transconductance * geometry.width / geometry.length,
                lambda,
                body_effect: process.body_effect,
                surface_potential: process.surface_potential,
                root_potential,
                built_in: sign * process.threshold - process.body_effect * root_potential,
            },
            junctions,
            overlaps,
            oxide,
            last: [0.0; 3],
            threshold: sign * process.threshold,
        })
    }
}

/// A transistor's extent: its effective length, and its width, the sum of its effective
/// lengths and its gate's area with every parallel copy counted.
#[derive(Debug, Clone, Copy)]
struct Geometry {
    multiplier: f64,
    length: f64,
    width: f64,
    lengths: f64,
    area: f64,
}

impl Geometry {
    fn new(parameters: &MosfetParameters) -> Result<Self, ParameterError> {
        let multiplier = ParameterError::positive("multiplier", parameters.multiplier)?;
        let width = ParameterError::positive("width", parameters.width)?;
        let length = ParameterError::positive("length", parameters.length)?;
        let diffusion =
            ParameterError::non_negative("lateral_diffusion", parameters.lateral_diffusion)?;
        let effective = length - 2.0 * diffusion;
        if effective <= 0.0 {
            return Err(ParameterError(format!(
                "the effective length, length − 2 · lateral_diffusion, must be positive, \
                 not {effective:?}"
            )));
        }
        Ok(Self {
            multiplier,
            length: effective,
            width: multiplier * width,
            lengths: multiplier * effective,
            area: effective * multiplier * width,
        })
    }
}

impl MosfetParameters {
    /// The bulk-drain and bulk-source junctions.
    fn junctions(&self, geometry: &Geometry) -> Result<[BulkJunction; 2], ParameterError> {
        let multiplier = geometry.multiplier;
        let saturation = ParameterError::positive("saturation_current", self.saturation_current)?;
        let density = ParameterError::non_negative(
            "saturation_current_density",
            self.saturation_current_density,
        )?;
        let potential = ParameterError::positive("junction_potential", self.junction_potential)?;
        let bottom_grading = below_one("bottom_grading", self.bottom_grading)?;
        let sidewall_grading = below_one("sidewall_grading", self.sidewall_grading)?;
        let coefficient = below_one("depletion_coefficient", self.depletion_coefficient)?;
        let bottom_capacitance =
            ParameterError::non_negative("bottom_capacitance", self.bottom_capacitance)?;
        let sidewall_capacitance =
            ParameterError::non_negative("sidewall_capacitance", self.sidewall_capacitance)?;
        let by_density = density > 0.0 && self.drain_area > 0.0 && self.source_area > 0.0;
        let junction = |name, capacitance: Option<f64>, area: f64, perimeter: f64| {
            ParameterError::non_negative(&format!("{name}_area"), area)?;
            ParameterError::non_negative(&format!("{name}_perimeter"), perimeter)?;
            let saturation_current = multiplier
                * if by_density {
                    density * area
                } else {
                    saturation
                };
            let bottom = match capacitance {
                Some(capacitance) => {
                    ParameterError::non_negative(&format!("bulk_{name}_capacitance"), capacitance)?
                }
                None => bottom_capacitance * area,
            };
            Ok(BulkJunction {
                saturation_current,
                critical_voltage: critical_voltage(saturation_current, THERMAL_VOLTAGE),
                bottom: Depletion::new(multiplier * bottom, potential, bottom_grading, coefficient),
                sidewall: Depletion::new(
                    multiplier * sidewall_capacitance * perimeter,
                    potential,
                    sidewall_grading,
                    coefficient,
                ),
            })
        };
        Ok([
            junction(
                "drain",
                self.bulk_drain_capacitance,
                self.drain_area,
                self.drain_perimeter,
            )?,
            junction(
                "source",
                self.bulk_source_capacitance,
                self.source_area,
                self.source_perimeter,
            )?,
        ])
    }

    /// The drain and source resistances, given or from the sheet resistance, of all the
    /// parallel copies together.
    fn series_resistances(&self) -> Result<[f64; 2], ParameterError> {
        let sheet = ParameterError::non_negative("sheet_resistance", self.sheet_resistance)?;
        let resistance = |name: &str, given: Option<f64>, squares: f64| {
            ParameterError::non_negative(&format!("{name}_squares"), squares)?;
            let resistance = match given {
                Some(resistance) => {
                    ParameterError::non_negative(&format!("{name}_resistance"), resistance)?
                }
                None => sheet * squares,
            };
            Ok::<_, ParameterError>(resistance / self.multiplier)
        };
        Ok([
            resistance("drain", self.drain_resistance, self.drain_squares)?,
            resistance("source", self.source_resistance, self.source_squares)?,
        ])
    }
}

impl Mosfet {
    /// Adds the drain and source resistances.
    fn add_series(&self, input: &Input<'_>, output: &mut Output<'_>) {
        let stamp = &self.stamp;
        for (outer, inner, conductance) in [
            (DRAIN, INNER_DRAIN, self.drain_conductance),
            (SOURCE, INNER_SOURCE, self.source_conductance),
        ] {
            if conductance > 0.0 {
                let voltage = stamp.voltage(input, outer) - stamp.voltage(input, inner);
                stamp.current(output, outer, inner, conductance * voltage);
                stamp.conductance(output, (outer, inner), (outer, inner), conductance);
            }
        }
    }

    /// The terminal voltages at `input`, limited from the last evaluation's.
    fn bias(&mut self, input: &Input<'_>, output: &mut Output<'_>) -> Bias {
        let stamp = &self.stamp;
        let applied = VARIABLES.map(|(plus, minus)| {
            self.sign * (stamp.voltage(input, plus) - stamp.voltage(input, minus))
        });
        let limited = self.limit(applied);
        if limited.is_some() {
            output.mark_limited();
        }
        let voltages = limited.unwrap_or(applied);
        self.last = voltages;
        Bias {
            voltages,
            beyond: [0, 1, 2].map(|index| applied[index] - voltages[index]),
        }
    }

    /// The `applied` gate-source, drain-source and bulk-source voltages limited from the
    /// last evaluation's as SPICE limits them: the gate's step around the threshold and the
    /// drain's, from whichever end acts as the source, then the step of the junction with
    /// it. Returns `None` when no step needs limiting.
    fn limit(&self, applied: [f64; 3]) -> Option<[f64; 3]> {
        let [mut vgs, mut vds, applied_vbs] = applied;
        let [previous_gate, previous_drain, previous_bulk] = self.last;
        let mut limited = false;
        let mut apply = |value: f64, limit: Option<f64>| {
            limited |= limit.is_some();
            limit.unwrap_or(value)
        };
        let vgd = vgs - vds;
        if previous_drain >= 0.0 {
            vgs = apply(vgs, limit_gate(vgs, previous_gate, self.threshold));
            vds = vgs - vgd;
            vds = apply(vds, limit_drain(vds, previous_drain));
        } else {
            let vgd = apply(
                vgd,
                limit_gate(vgd, previous_gate - previous_drain, self.threshold),
            );
            vds = vgs - vgd;
            vds = -apply(-vds, limit_drain(-vds, -previous_drain));
            vgs = vgd + vds;
        }
        let [drain, source] = &self.junctions;
        let vbs = if vds >= 0.0 {
            let limit = limit_junction(
                applied_vbs,
                previous_bulk,
                THERMAL_VOLTAGE,
                source.critical_voltage,
            );
            apply(applied_vbs, limit)
        } else {
            let (vbd, previous_junction) =
                (applied_vbs - applied[1], previous_bulk - previous_drain);
            let limit = limit_junction(
                vbd,
                previous_junction,
                THERMAL_VOLTAGE,
                drain.critical_voltage,
            );
            apply(vbd, limit) + vds
        };
        limited.then_some([vgs, vds, vbs])
    }

    /// Adds `current`, oriented as an n-channel's, from local node `from` to `to`,
    /// extended to the applied voltages, with its derivatives.
    fn add_current(
        &self,
        (from, to): (usize, usize),
        current: Dual<3>,
        bias: &Bias,
        output: &mut Output<'_>,
    ) {
        let stamp = &self.stamp;
        stamp.current(output, from, to, self.sign * current.extend(bias.beyond));
        for (variable, derivative) in VARIABLES.into_iter().zip(current.gradient) {
            stamp.conductance(output, (from, to), variable, derivative);
        }
    }

    /// Adds `charge`, oriented as an n-channel's, on local node `on` against `against`,
    /// extended to the applied voltages, with its derivatives.
    fn add_charge(
        &self,
        (on, against): (usize, usize),
        charge: Dual<3>,
        bias: &Bias,
        output: &mut Output<'_>,
    ) {
        let stamp = &self.stamp;
        stamp.charge(output, on, against, self.sign * charge.extend(bias.beyond));
        for (variable, derivative) in VARIABLES.into_iter().zip(charge.gradient) {
            stamp.capacitance(output, (on, against), variable, derivative);
        }
    }
}

impl Device for Mosfet {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_>) {
        self.add_series(input, output);
        let bias = self.bias(input, output);
        let [vgs, vds, vbs] = Dual::variables(bias.voltages);
        let conduction = self.channel.evaluate([vgs, vds, vbs]);
        self.threshold = conduction.threshold.value;
        self.add_current(
            (INNER_DRAIN, INNER_SOURCE),
            conduction.current,
            &bias,
            output,
        );

        let (vgd, vbd) = (vgs - vds, vbs - vds);
        let [drain, source] = self.junctions;
        for (junction, node, voltage) in [(drain, INNER_DRAIN, vbd), (source, INNER_SOURCE, vbs)] {
            let (current, charge) = junction.evaluate(voltage, input.gmin);
            self.add_current((BULK, node), current, &bias, output);
            self.add_charge((BULK, node), charge, &bias, output);
        }

        let [source_overlap, drain_overlap, bulk_overlap] = self.overlaps;
        self.add_charge((GATE, INNER_SOURCE), vgs * source_overlap, &bias, output);
        self.add_charge((GATE, INNER_DRAIN), vgd * drain_overlap, &bias, output);
        self.add_charge((GATE, BULK), (vgs - vbs) * bulk_overlap, &bias, output);

        if let Some(oxide) = &self.oxide {
            let capacitances = if conduction.forward {
                oxide.meyer(vgs, vgd, &conduction)
            } else {
                let [drain, source, bulk] = oxide.meyer(vgd, vgs, &conduction);
                [source, drain, bulk]
            };
            let whole = oxide.capacitance;
            for (displacement, capacitance) in oxide.displacements.iter().zip(capacitances) {
                let current =
                    displacement.add(&self.stamp, (whole, capacitance.value), input, output);
                let meyer = capacitance * (self.sign * current / whole);
                self.add_current((GATE, displacement.node), meyer, &bias, output);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::testing::{Bench, assert_jacobians_match};

    const CHARGED: &str = r#"{"threshold_voltage": 0.8, "transconductance_parameter": 1e-4,
        "body_effect": 0.5, "channel_length_modulation": 0.02, "width": 10e-6,
        "length": 2e-6, "lateral_diffusion": 1e-7, "oxide_thickness": 2e-8,
        "drain_area": 2e-11, "source_area": 3e-11, "drain_perimeter": 2e-5,
        "source_perimeter": 3e-5, "bottom_capacitance": 2e-4,
        "sidewall_capacitance": 5e-10, "gate_source_overlap": 4e-10,
        "gate_drain_overlap": 3e-10, "gate_bulk_overlap": 2e-10}"#;

    /// The same transistor with a p-channel.
    fn p_channel(parameters: &str) -> String {
        parameters
            .replace(
                r#""threshold_voltage": 0.8"#,
                r#""threshold_voltage": -0.8"#,
            )
            .replace('{', r#"{"channel": "p", "#)
    }

    /// The drain current of a transistor with `β = 1e-4` and `vto = 1` at `vgs` and `vds`,
    /// with the source grounded and the bulk well below both, its junctions' leakage
    /// negligible.
    fn drain_current(vgs: f64, vds: f64) -> f64 {
        let parameters = r#"{"threshold_voltage": 1, "transconductance_parameter": 1e-4,
            "saturation_current": 1e-30}"#;
        Bench::new::<Mosfet>(parameters)
            .settle(&[vds, vgs, 0.0, -5.0])
            .residual[0]
    }

    #[test]
    fn follows_the_square_law() {
        // Saturation: β · vgst² / 2; below it: β · vds · (vgst − vds / 2).
        let saturated = drain_current(2.0, 5.0);
        assert!((saturated - 5e-5).abs() < 1e-17, "{saturated}");
        let linear = drain_current(2.0, 0.5);
        assert!((linear - 3.75e-5).abs() < 1e-17, "{linear}");
        let reversed = drain_current(1.5, -0.5);
        let mirrored = -drain_current(2.0, 0.5);
        assert!((reversed - mirrored).abs() < 1e-17, "{reversed} {mirrored}");
        assert!(drain_current(0.9, 5.0).abs() < 1e-17);
    }

    #[test]
    fn the_body_effect_raises_the_threshold() {
        let parameters = r#"{"threshold_voltage": 1, "body_effect": 0.5,
            "surface_potential": 0.7, "saturation_current": 1e-30}"#;
        let mut bench = Bench::new::<Mosfet>(parameters);
        let current = bench.settle(&[5.0, 2.0, 0.0, -1.0]).residual[0];
        let threshold = 1.0 + 0.5 * (1.7_f64.sqrt() - 0.7_f64.sqrt());
        let expected = 2e-5 / 2.0 * (2.0 - threshold) * (2.0 - threshold);
        assert!((current - expected).abs() < 1e-17, "{current} {expected}");
    }

    #[test]
    fn a_p_channel_mirrors_an_n_channel() {
        let mut n = Bench::new::<Mosfet>(CHARGED);
        let mut p = Bench::new::<Mosfet>(&p_channel(CHARGED));
        let x = [3.0, 2.0, 0.2, -0.5, 1e-4, -2e-4, 3e-4];
        let mirrored: Vec<f64> = x.iter().map(|value| -value).collect();
        let n_residuals = n.settle(&x).residual;
        let p_residuals = p.settle(&mirrored).residual;
        for (n_residual, p_residual) in n_residuals.iter().zip(&p_residuals) {
            assert!(
                (n_residual + p_residual).abs() < 1e-18,
                "{n_residuals:?} {p_residuals:?}"
            );
        }
    }

    #[test]
    fn meyers_capacitances_share_the_oxide() {
        let mut bench = Bench::new::<Mosfet>(CHARGED);
        // The gate's derivatives with respect to the three displacement currents: Meyer's
        // capacitances as fractions of the oxide's.
        let mut shares = |gate: f64, drain: f64| {
            let evaluation = bench.settle(&[drain, gate, 0.0, 0.0, 0.0, 0.0, 0.0]);
            [4, 5, 6].map(|column| evaluation.conductance[1][column])
        };
        let [source, drain, bulk] = shares(-2.0, 1.0);
        assert!(source == 0.0 && drain == 0.0 && (bulk - 1.0).abs() < 1e-15);
        let [source, drain, bulk] = shares(3.0, 5.0);
        assert!((source - 2.0 / 3.0).abs() < 1e-15 && drain == 0.0 && bulk == 0.0);
        let [source, drain, _] = shares(3.0, 0.0);
        assert!((source - 0.5).abs() < 1e-15 && (drain - 0.5).abs() < 1e-15);
    }

    #[test]
    fn unknowns_follow_the_resistances_and_the_oxide() {
        let parameters = r#"{"drain_resistance": 10, "sheet_resistance": 20,
            "source_squares": 0.5, "oxide_thickness": 1e-7}"#;
        let bench = Bench::new::<Mosfet>(parameters);
        assert_eq!(
            bench.unknowns(),
            [
                "n0",
                "n1",
                "n2",
                "n3",
                "device:drain",
                "device:source",
                "device:gate_source_displacement",
                "device:gate_drain_displacement",
                "device:gate_bulk_displacement"
            ]
        );
        assert_eq!(Bench::new::<Mosfet>("{}").unknowns().len(), 4);
    }

    #[test]
    fn rejects_inconsistent_parameters() {
        let error = |parameters| Bench::try_new::<Mosfet>(parameters).unwrap_err().0;
        assert_eq!(
            error(r#"{"substrate_doping": 1e22}"#),
            "substrate_doping needs an oxide_thickness"
        );
        assert_eq!(
            error(r#"{"length": 1e-6, "lateral_diffusion": 5e-7}"#),
            "the effective length, length − 2 · lateral_diffusion, must be positive, not 0.0"
        );
        assert_eq!(
            error(r#"{"bottom_grading": 1}"#),
            "bottom_grading must be at least 0 and below 1, not 1.0"
        );
    }

    proptest! {
        /// The conductances and capacitances are the derivatives of the currents and
        /// charges, in every region, in both modes and for both channels, with the bulk up
        /// to 0.6 V forward of the drain or the source.
        #[test]
        fn jacobians_match_finite_differences(
            drain in -4.0..4.0f64,
            gate in -3.0..4.0f64,
            reverse in -0.6..2.0f64,
            displacements in proptest::array::uniform3(-1e-3..1e-3f64),
            p in proptest::bool::ANY,
        ) {
            let parameters = if p { p_channel(CHARGED) } else { CHARGED.to_owned() };
            let sign = if p { -1.0 } else { 1.0 };
            let bulk = drain.min(0.0) - reverse;
            let [gs, gd, gb] = displacements;
            assert_jacobians_match::<Mosfet>(
                &parameters,
                &[sign * drain, sign * gate, 0.0, sign * bulk, gs, gd, gb],
            );
            let resistive = parameters.replace(
                '{',
                r#"{"drain_resistance": 50, "source_resistance": 20, "#,
            );
            let inner = [sign * 0.9 * drain, sign * 0.01];
            assert_jacobians_match::<Mosfet>(
                &resistive,
                &[sign * drain, sign * gate, 0.0, sign * bulk, inner[0], inner[1], gs, gd, gb],
            );
        }

        /// A transistor whose threshold derives from its substrate doping has the
        /// derivatives too.
        #[test]
        fn derived_jacobians_match_finite_differences(
            drain in -4.0..4.0f64,
            gate in -3.0..4.0f64,
            reverse in -0.6..2.0f64,
        ) {
            let parameters = r#"{"oxide_thickness": 5e-8, "substrate_doping": 1e22,
                "surface_state_density": 1e14, "gate_material": "aluminum"}"#;
            let bulk = drain.min(0.0) - reverse;
            assert_jacobians_match::<Mosfet>(
                parameters,
                &[drain, gate, 0.0, bulk, 1e-4, 2e-4, -1e-4],
            );
        }
    }
}
