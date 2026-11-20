# Foundry processes

Integrated-circuit transistors are defined by foundry process design kits (PDKs).
jolty supports SkyWater SKY130 transistors as foundry devices: the engine checks circuit structure while ngspice simulates every analysis using the pinned PDK models.

## Placing a transistor

`sky130.mosfet(transistor, width, length, fingers=1, multiplier=1)` places a SKY130 transistor using its short model name (without library prefix), with `width` specifying total width across all fingers:

```python
from jolty import Builder, Role, component, devices
from jolty.spice import sky130
from jolty.units import V, um


@component
def inverter(c: Builder) -> None:
    ground = c.port("0", Role.GROUND)
    supply = c.place(devices.voltage_source(1.8 * V), name="VDD")
    lower = c.place(sky130.mosfet("nfet_01v8", 1 * um, 0.15 * um), name="MN")
    upper = c.place(sky130.mosfet("pfet_01v8", 2 * um, 0.15 * um, fingers=2), name="MP")
    c.connect(supply.positive, upper.source, upper.bulk, name="vdd")
    c.connect(lower.gate, upper.gate, name="in")
    c.connect(lower.drain, upper.drain, name="out")
    c.connect(supply.negative, lower.source, lower.bulk, ground)
```

Each transistor exposes `drain`, `gate`, `source`, and `bulk` terminals.
`sky130.TRANSISTORS` lists all supported models: 1.8 V core transistors (standard, low, and high threshold), 3.3 V and 5 V native transistors, and extended-drain devices rated for 5 V gates with 10.5 V or 16 V drains.
Foundry models use binned geometry ranges, so widths or lengths outside every bin fail during ngspice execution.

Builds elaborate and structurally check foundry circuits using MOSFET branch declarations, while skipping operating-point checks because the native engine does not solve foundry devices.

## Simulating in ngspice

`jolty simulate` and `Simulation.run` dispatch circuits containing foundry devices to ngspice, loading the PDK library at the plan temperature and options in HSPICE compatibility mode.
The exporter writes a [SPICE netlist](spice.md#writing-spice) instantiating each transistor subcircuit in micrometers (matching PDK scaling) and estimating diffusion areas, perimeters, and squares using xschem's 0.29 µm finger-diffusion rule.
ngspice parses the full PDK library before each analysis, taking several seconds per run.

Foundry results use the standard jolty result types, working transparently with reports, plots, specs, and corners.
Results contain all circuit net voltages plus independent-source and inductor currents under jolty names.
ngspice does not expose internal device unknowns or terminal currents regardless of [`record`](results.md#what-a-result-records) settings, so current and power ratings are skipped and operating points report strategy `ngspice`.

Supported analyses for foundry circuits are operating point, evenly stepped DC sweep, AC response, transient, and voltage-output noise.
Transients use ngspice step selection without forcing exact landing on `breakpoints`.
Noise analysis reports ngspice noise-source names (such as `id` thermal channel noise and `1overf` flicker noise); when no input source is specified, ngspice refers noise to the first independent source and jolty omits the referral.
Unsupported analyses on foundry circuits fail with an explicit error.

Foundry netlists cannot mix level 1 or level 3 MOSFETs alongside foundry devices (because the PDK globally rescales MOSFET dimensions), nor assign per-device temperatures to foundry devices.

## Process corners

A plan's `process` parameter selects the PDK corner (defaulting to typical `tt`), and each [corner](simulation.md#corners) can override the process corner:

```python
Plan(
    {"bias": OperatingPoint(), "gain": Ac("VIN", decade(1 * kHz, 1 * GHz, 10))},
    corners={
        "slow": Corner(process="ss", temperature=celsius(125)),
        "fast": Corner(process="ff", temperature=celsius(-40)),
    },
)
```

`sky130.CORNERS` lists all valid corners: `tt`, MOSFET corners `ff`, `ss`, `sf`, `fs` (NMOS first, PMOS second), and passive corners `ll`, `hh`, `hl`, `lh`.
Specifying a process corner on a circuit without foundry devices raises an error.

## Mismatch

[Monte Carlo](simulation.md#monte-carlo) analysis draws per-instance PDK mismatch on every sample alongside native device tolerances, varying transistor threshold, oxide, and mobility by area using a per-sample seed.
Each sample reuses the same mismatch draw across all corners so corners evaluate identical silicon, and reports count matched foundry devices.
`Simulation(circuit, mismatch=seed)` draws a single mismatch sample explicitly.

## The open process kits

`kits.read(kit, sections)` loads PDK model libraries at a corner into a [`Library`](spice.md) matching ngspice for SkyWater SKY130 (`sky130.KIT`), GlobalFoundries GF180MCU (`gf180mcu.KIT`), and IHP SG13G2 (`sg13g2.KIT`).
Whole-file kit includes load first, followed by one section per library (defaulting to typical); `kit.sections` enumerates every corner and statistical section.
The reader supports `.lib` sections, relative `.include` paths, `.option scale`, subcircuit parameters and braced or quoted expressions, conditional `.if` blocks, and binned models (`LMIN` to `WMAX`).
Under HSPICE compatibility (such as SKY130) or `wnflag`, transistor width divides across fingers (`nf`) identically to ngspice `INPgetModBin`.

`temper` evaluates to the library load temperature in kelvins (default 27 °C), used by SG13G2 bipolar transit times.
Statistical functions (`agauss`, `gauss`, `aunif`, `unif`, `limit`) default to nominal values unless drawn via `library.drawn(name, Sample(seed, index), path)`.
Each drawn placement evaluates statistical calls from deterministic engine variates keyed by seed, sample index, placement path, and call order.
Deterministic keying guarantees identical mismatch bits across arbitrary shard splits while drawing independent mismatch per placement.
`library.random_variables(name, path)` exposes the same placement's statistical calls (including nested placements) for first-order [mismatch analysis](simulation.md#mismatch) via `Mismatch(outputs, random_variables=...)`.
Each random variable pairs its nominal mean with its analytical standard deviation.
Standard deviations are `a / s` for `agauss(x, a, s)`, `|x| r / s` for `gauss(x, r, s)`, `|a| / √3` for `aunif(x, a)`, `|x r| / √3` for `unif(x, r)`, and `|a|` for `limit(x, a)`.
Parameter derivatives use `ε^(1/3) σ` central differences, which are exact to floating-point rounding for the quadratic or linear parameter expressions used in open PDKs.
Global library-level statistical calls represent process variation shared by every instance and are excluded from per-placement mismatch.
Resistors and capacitors attached to a VBIC thermal node automatically map to thermal resistances and capacitances.

`kits.inventory(kit)` inspects every subcircuit and model in a PDK, listing constituent elements, required models, and any blocker preventing native loading.
Inventory evaluation uses parameter defaults; when defaults miss every MOSFET bin, `l` and `w` fall back to the first bin midpoint (reported in `parameters`).
When defaults reference undefined parameters, evaluation falls back to the first internal kit placement (such as GF180MCU `nmos_3p3_sab` placing `nplus_u_m1`) or ngspice's 10 µm default for SPICE `l` and `w`.
Each device receives a status: `native` (all elements supported), `verilog-a` (only Verilog-A models remain, which ngspice loads via OSDI), `neither` (missing SPICE model), `unread` (unsupported syntax), or `unvalued` (undefined parameter).
Among SG13G2 Verilog-A models, `jolty-veriloga` compiles `psp103va`, `r3_cmc`, and `mosvar` cleanly ([`tests/test_veriloga.py`](https://github.com/maxencefaldor/jolty/blob/main/tests/test_veriloga.py)) alongside MOM capacitors `cap_cmomi` and `cap_cmomf`.
At typical corners, native coverage stands at 251 of 263 devices for SKY130, 65 of 72 for GF180MCU, and 36 of 53 for SG13G2 (with 14 more in Verilog-A):

### SKY130

<!-- devices of: sky130 -->

| Status | Models | First problem | Devices |
| --- | --- | --- | --- |
| native | BSIM4 4.5 (level 54) |  | 91: `sky130_fd_pr__nfet_01v8`, `sky130_fd_pr__nfet_01v8_lvt`, `sky130_fd_pr__pfet_01v8`, `sky130_fd_pr__nfet_03v3_nvt`, `sky130_fd_pr__nfet_05v0_nvt`, `sky130_fd_pr__esd_nfet_01v8`, `sky130_fd_pr__pfet_01v8_lvt`, `sky130_fd_pr__pfet_01v8_hvt`, `sky130_fd_pr__esd_pfet_g5v0d10v5`, `sky130_fd_pr__pfet_g5v0d10v5`, `sky130_fd_pr__nfet_g5v0d10v5`, `sky130_fd_pr__nfet_g5v0d16v0__base`, `sky130_fd_pr__esd_nfet_g5v0d10v5`, `sky130_fd_pr__pfet_g5v0d16v0__base`, `sky130_fd_pr__cap_vpp_11p3x11p8_l1m1m2m3m4_shieldm5_nhv__base`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM10W3p00L0p50`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM04W3p00L0p50`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM10W5p00L0p50`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM04W5p00L0p50`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM10W7p00L0p50`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM04W7p00L0p50`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM02W5p00L0p50`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM02W3p00L0p50`, `sky130_fd_pr__rf_nfet_01v8_bM02W1p65L0p15`, `sky130_fd_pr__rf_nfet_01v8_bM04W1p65L0p15`, `sky130_fd_pr__rf_nfet_01v8_bM02W1p65L0p18`, `sky130_fd_pr__rf_nfet_01v8_bM04W1p65L0p18`, `sky130_fd_pr__rf_nfet_01v8_bM02W1p65L0p25`, `sky130_fd_pr__rf_nfet_01v8_bM04W1p65L0p25`, `sky130_fd_pr__rf_nfet_01v8_bM02W3p00L0p15`, `sky130_fd_pr__rf_nfet_01v8_bM04W3p00L0p15`, `sky130_fd_pr__rf_nfet_01v8_bM02W3p00L0p18`, `sky130_fd_pr__rf_nfet_01v8_bM04W3p00L0p18`, `sky130_fd_pr__rf_nfet_01v8_bM02W3p00L0p25`, `sky130_fd_pr__rf_nfet_01v8_bM04W3p00L0p25`, `sky130_fd_pr__rf_nfet_01v8_bM02W5p00L0p15`, `sky130_fd_pr__rf_nfet_01v8_bM04W5p00L0p15`, `sky130_fd_pr__rf_nfet_01v8_bM02W5p00L0p18`, `sky130_fd_pr__rf_nfet_01v8_bM04W5p00L0p18`, `sky130_fd_pr__rf_nfet_01v8_bM02W5p00L0p25`, `sky130_fd_pr__rf_nfet_01v8_bM04W5p00L0p25`, `sky130_fd_pr__rf_pfet_01v8_bM02W1p65L0p15`, `sky130_fd_pr__rf_pfet_01v8_bM04W1p65L0p15`, `sky130_fd_pr__rf_pfet_01v8_bM02W1p65L0p18`, `sky130_fd_pr__rf_pfet_01v8_bM04W1p65L0p18`, `sky130_fd_pr__rf_pfet_01v8_bM02W1p65L0p25`, `sky130_fd_pr__rf_pfet_01v8_bM04W1p65L0p25`, `sky130_fd_pr__rf_pfet_01v8_bM02W3p00L0p15`, `sky130_fd_pr__rf_pfet_01v8_bM04W3p00L0p15`, `sky130_fd_pr__rf_pfet_01v8_bM02W3p00L0p18`, `sky130_fd_pr__rf_pfet_01v8_bM04W3p00L0p18`, `sky130_fd_pr__rf_pfet_01v8_bM02W3p00L0p25`, `sky130_fd_pr__rf_pfet_01v8_bM04W3p00L0p25`, `sky130_fd_pr__rf_pfet_01v8_bM02W5p00L0p15`, `sky130_fd_pr__rf_pfet_01v8_bM04W5p00L0p15`, `sky130_fd_pr__rf_pfet_01v8_bM02W5p00L0p18`, `sky130_fd_pr__rf_pfet_01v8_bM04W5p00L0p18`, `sky130_fd_pr__rf_pfet_01v8_bM02W5p00L0p25`, `sky130_fd_pr__rf_pfet_01v8_bM04W5p00L0p25`, `sky130_fd_pr__rf_nfet_01v8_bM02`, `sky130_fd_pr__rf_nfet_01v8_bM02W3p00`, `sky130_fd_pr__rf_nfet_01v8_bM02W5p00`, `sky130_fd_pr__rf_nfet_01v8_bM04`, `sky130_fd_pr__rf_nfet_01v8_bM04W3p00`, `sky130_fd_pr__rf_nfet_01v8_bM04W5p00`, `sky130_fd_pr__rf_nfet_01v8_lvt_bM02`, `sky130_fd_pr__rf_nfet_01v8_lvt_bM02W3p00`, `sky130_fd_pr__rf_nfet_01v8_lvt_bM02W5p00`, `sky130_fd_pr__rf_nfet_01v8_lvt_bM02W5p00L0p18`, `sky130_fd_pr__rf_nfet_01v8_lvt_bM02W5p00L0p25`, `sky130_fd_pr__rf_nfet_01v8_lvt_bM04`, `sky130_fd_pr__rf_nfet_01v8_lvt_bM04W3p00`, `sky130_fd_pr__rf_nfet_01v8_lvt_bM04W5p00`, `sky130_fd_pr__rf_nfet_01v8_lvt_bM04W5p00L0p18`, `sky130_fd_pr__rf_nfet_01v8_lvt_bM04W5p00L0p25`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM02`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM02W5p00`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM04`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM04W5p00`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM04W7p00`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM10`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM10W5p00`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM10W7p00`, `sky130_fd_pr__rf_pfet_01v8_bM02`, `sky130_fd_pr__rf_pfet_01v8_bM02W3p00`, `sky130_fd_pr__rf_pfet_01v8_bM02W5p00`, `sky130_fd_pr__rf_pfet_01v8_bM04`, `sky130_fd_pr__rf_pfet_01v8_bM04W3p00`, `sky130_fd_pr__rf_pfet_01v8_bM04W5p00`, `sky130_fd_pr__pfet_01v8_mvt`, `sky130_fd_pr__special_nfet_pass_lvt` |
| neither | BSIM4 4.5 (level 54); resistor; diode (level 3) | the element's area times its copies is positive, not 0.0 | 2: `sky130_fd_pr__nfet_g5v0d16v0`, `sky130_fd_pr__pfet_g5v0d16v0` |
| native | BSIM4 4.5 (level 54); resistor whose value reads a voltage; diode (level 3) |  | 5: `sky130_fd_pr__nfet_20v0`, `sky130_fd_pr__nfet_20v0_reverse_iso`, `sky130_fd_pr__nfet_20v0_iso`, `sky130_fd_pr__nfet_20v0_zvt`, `sky130_fd_pr__nfet_20v0_nvt` |
| neither | BSIM4 4.62 (level 54); resistor whose value reads a voltage; diode (level 3) | `sky130_fd_pr__pfet_20v0__base.0` is a BSIM4 card of version 4.62, whose equations ngspice runs another model for; jolty models version 4.5 | 1: `sky130_fd_pr__pfet_20v0` |
| native | Gummel-Poon (level 1) |  | 4: `sky130_fd_pr__pnp_05v5_W0p68L0p68`, `sky130_fd_pr__pnp_05v5_W3p40L3p40`, `sky130_fd_pr__npn_05v5_W1p00L1p00`, `sky130_fd_pr__npn_05v5_W1p00L2p00` |
| native | Gummel-Poon (level 1); resistor; diode (level 3) |  | 1: `sky130_fd_pr__npn_11v0_W1p00L1p00` |
| native | capacitor of a charge; capacitor; resistor whose value reads a voltage |  | 2: `sky130_fd_pr__cap_var_lvt`, `sky130_fd_pr__cap_var_hvt` |
| native | resistor whose value reads a voltage |  | 1: `sky130_fd_pr__res_iso_pw` |
| native | resistor; capacitor |  | 8: `sky130_fd_pr__cap_vpp_08p6x07p8_l1m1m2_noshield_o1`, `sky130_fd_pr__cap_vpp_04p4x04p6_m1m2_noshield_o1nhv`, `sky130_fd_pr__cap_vpp_04p4x04p6_m1m2_noshield_o1phv`, `sky130_fd_pr__cap_vpp_08p6x07p8_m1m2m3_shieldl1`, `sky130_fd_pr__cap_vpp_04p4x04p6_m1m2m3_shieldl1`, `sky130_fd_pr__cap_vpp_11p5x11p7_m1m2m3_shieldl1`, `sky130_fd_pr__cap_vpp_11p5x11p7_m1m4_noshield`, `sky130_fd_pr__cap_vpp_44p7x23p1_pol1m1m2m3m4m5_noshield` |
| native | capacitor; resistor |  | 21: `sky130_fd_pr__cap_mim_m3_1`, `sky130_fd_pr__cap_mim_m3_2`, `sky130_fd_pr__cap_vpp_11p5x11p7_l1m1m2m3_shieldm4`, `sky130_fd_pr__cap_vpp_11p5x11p7_l1m1m2m3_shieldpom4`, `sky130_fd_pr__cap_vpp_06p8x06p1_l1m1m2m3_shieldpom4`, `sky130_fd_pr__cap_vpp_06p8x06p1_m1m2m3_shieldl1m4`, `sky130_fd_pr__cap_vpp_11p5x11p7_l1m1m2m3m4_shieldm5`, `sky130_fd_pr__cap_vpp_11p5x11p7_l1m1m2m3m4_shieldpom5`, `sky130_fd_pr__cap_vpp_11p5x11p7_m1m2m3m4_shieldl1m5`, `sky130_fd_pr__cap_vpp_11p5x11p7_m1m2m3m4_shieldm5`, `sky130_fd_pr__cap_vpp_08p6x07p8_m1m2m3_shieldl1m5_floatm4`, `sky130_fd_pr__cap_vpp_04p4x04p6_m1m2m3_shieldl1m5_floatm4`, `sky130_fd_pr__cap_vpp_11p5x11p7_m1m2m3_shieldl1m5_floatm4`, `sky130_fd_pr__cap_vpp_11p5x11p7_l1m1m2m3m4_shieldpom5_x`, `sky130_fd_pr__cap_vpp_02p9x06p1_m1m2m3m4_shieldl1_fingercap2`, `sky130_fd_pr__cap_vpp_02p7x11p1_m1m2m3m4_shieldl1_fingercap`, `sky130_fd_pr__cap_vpp_02p7x21p1_m1m2m3m4_shieldl1_fingercap`, `sky130_fd_pr__cap_vpp_02p7x41p1_m1m2m3m4_shieldl1_fingercap`, `sky130_fd_pr__cap_vpp_02p7x06p1_m1m2m3m4_shieldl1_fingercap`, `sky130_fd_pr__cap_vpp_11p3x11p3_m1m2m3m4_shieldl1_wafflecap`, `sky130_fd_pr__cap_vpp_05p9x05p9_m1m2m3m4_shieldl1_wafflecap` |
| neither | BSIM3 3.2 (level 49) | `sky130_fd_pr__special_nfet_pass__model.0` is a BSIM3 card of version 3.2, whose equations ngspice runs another model for; jolty models version 3.3 | 5: `sky130_fd_pr__special_nfet_pass`, `sky130_fd_pr__special_nfet_latch`, `sky130_fd_pr__special_pfet_latch`, `sky130_fd_pr__special_pfet_pass`, `sky130_fd_pr__special_nfet_pass_flash` |
| native | capacitor; resistor; BSIM4 4.5 (level 54) |  | 1: `sky130_fd_pr__cap_vpp_11p3x11p8_l1m1m2m3m4_shieldm5_nhv` |
| native | capacitor |  | 1: `sky130_fd_pr__model__parasitic__res_po` |
| native | resistor's model card; capacitor; resistor whose value reads a voltage |  | 1: `sky130_fd_pr__res_xhigh_po` |
| neither | resistor's model card; capacitor | jolty's devices have no counterpart to `SW_ET` of `rhead_model` | 1: `sky130_fd_pr__res_high_po` |
| native | resistor whose value reads a voltage; capacitor |  | 11: `sky130_fd_pr__res_xhigh_po_0p35`, `sky130_fd_pr__res_xhigh_po_0p69`, `sky130_fd_pr__res_xhigh_po_1p41`, `sky130_fd_pr__res_xhigh_po_2p85`, `sky130_fd_pr__res_xhigh_po_5p73`, `sky130_fd_pr__res_xhigh_po__base`, `sky130_fd_pr__res_high_po_0p35`, `sky130_fd_pr__res_high_po_0p69`, `sky130_fd_pr__res_high_po_1p41`, `sky130_fd_pr__res_high_po_2p85`, `sky130_fd_pr__res_high_po_5p73` |
| native | BSIM4 4.5 (level 54); capacitor; resistor |  | 15: `sky130_fd_pr__rf_nfet_01v8_lvt_aF02W0p84L0p15`, `sky130_fd_pr__rf_nfet_01v8_lvt_aF04W0p84L0p15`, `sky130_fd_pr__rf_nfet_01v8_lvt_aF04W3p00L0p15`, `sky130_fd_pr__rf_nfet_01v8_lvt_aF02W0p42L0p15`, `sky130_fd_pr__rf_nfet_01v8_lvt_aF02W3p00L0p15`, `sky130_fd_pr__rf_nfet_01v8_lvt_aF08W3p00L0p15`, `sky130_fd_pr__rf_nfet_01v8_lvt_aF08W0p84L0p15`, `sky130_fd_pr__rf_pfet_01v8_aF02W1p68L0p15`, `sky130_fd_pr__rf_pfet_01v8_aF02W5p00L0p15`, `sky130_fd_pr__rf_pfet_01v8_aF02W0p84L0p15`, `sky130_fd_pr__rf_pfet_01v8_aF04W1p68L0p15`, `sky130_fd_pr__rf_pfet_01v8_aF02W3p00L0p15`, `sky130_fd_pr__rf_pfet_01v8_mvt_aF02W1p68L0p15`, `sky130_fd_pr__rf_pfet_01v8_mvt_aF02W0p84L0p15`, `sky130_fd_pr__rf_pfet_01v8_mvt_aF04W1p68L0p15` |
| neither | resistor's model card | resistance is positive, not 0.0 | 1: `sky130_fd_pr__res_generic_po` |
| native | resistor's model card; diode (level 3) |  | 4: `sky130_fd_pr__res_generic_nd`, `sky130_fd_pr__res_generic_pd`, `sky130_fd_pr__res_generic_nd__hv`, `sky130_fd_pr__res_generic_pd__hv` |
| native | resistor's model card |  | 24: `short`, `sky130_fd_pr__res_generic_nd` (model), `sky130_fd_pr__res_generic_pd` (model), `sky130_fd_pr__res_generic_nd__hv` (model), `sky130_fd_pr__res_generic_pd__hv` (model), `sky130_fd_pr__res_generic_po` (model), `sky130_fd_pr__res_generic_nw` (model), `sky130_fd_pr__res_generic_l1` (model), `sky130_fd_pr__res_generic_m1` (model), `sky130_fd_pr__res_generic_m2` (model), `sky130_fd_pr__res_generic_m3` (model), `sky130_fd_pr__res_generic_m4` (model), `sky130_fd_pr__res_generic_m5` (model), `sky130_fd_pr__res_generic_r1` (model), `mrcn` (model), `mrcp` (model), `mrcp1` (model), `mrcl1` (model), `mrcvia` (model), `mrcvia2` (model), `mrcvia3` (model), `mrcvia4` (model), `mrcrdlcon` (model), `short` (model) |
| neither | diode (level 3) | the element's area times its copies is positive, not 0.0 | 2: `sky130_fd_pr__diode_pw2nd`, `sky130_fd_pr__diode_pw2nd_05v5` |
| native | diode (level 3) |  | 18: `sky130_fd_pr__model__parasitic__diode_pw2dn__extended_drain` (model), `sky130_fd_pr__diode_pw2nd_05v5__extended_drain` (model), `sky130_fd_pr__model__parasitic__diode_ps2dn__extended_drain` (model), `sky130_fd_pr__nfet_20v0_zvt__parasitic__diode_ps2dn__extended_drain` (model), `sky130_fd_pr__pfet_20v0__parasitic__diode_pw2dn` (model), `sky130_fd_pr__nfet_20v0_nvt__parasitic__diode_ps2dn` (model), `sky130_fd_pr__model__parasitic__diode_ps2nw` (model), `sky130_fd_pr__diode_pw2nd_05v5` (model), `sky130_fd_pr__diode_pd2nw_05v5` (model), `sky130_fd_pr__diode_pd2nw_05v5_hvt` (model), `sky130_fd_pr__diode_pw2nd_11v0` (model), `sky130_fd_pr__diode_pd2nw_11v0` (model), `sky130_fd_pr__diode_pw2nd_05v5_nvt` (model), `sky130_fd_pr__diode_pw2nd_05v5_lvt` (model), `sky130_fd_pr__diode_pd2nw_05v5_lvt` (model), `sky130_fd_pr__model__parasitic__diode_pw2dn` (model), `sky130_fd_pr__model__parasitic__diode_ps2dn` (model), `sky130_fd_pr__pfet_g5v0d16v0__parasitic__diode_pw2dn` (model) |
| native | capacitor's model card |  | 43: `mcp1f` (model), `mcl1f` (model), `mcl1d` (model), `mcl1p1` (model), `mcm1f` (model), `mcm1d` (model), `mcm1p1` (model), `mcm1l1` (model), `mcm2f` (model), `mcm2d` (model), `mcm2p1` (model), `mcm2l1` (model), `mcm2m1` (model), `mcm3f` (model), `mcm3d` (model), `mcm3p1` (model), `mcm3l1` (model), `mcm3m1` (model), `mcm3m2` (model), `mcm4f` (model), `mcm4d` (model), `mcm4p1` (model), `mcm4l1` (model), `mcm4m1` (model), `mcm4m2` (model), `mcm4m3` (model), `mcm5f` (model), `mcm5d` (model), `mcm5p1` (model), `mcm5l1` (model), `mcm5m1` (model), `mcm5m2` (model), `mcm5m3` (model), `mcm5m4` (model), `mcrdlf` (model), `mcrdld` (model), `mcrdlp1` (model), `mcrdll1` (model), `mcrdlm1` (model), `mcrdlm2` (model), `mcrdlm3` (model), `mcrdlm4` (model), `mcrdlm5` (model) |

### GF180MCU

<!-- devices of: gf180mcu -->

| Status | Models | First problem | Devices |
| --- | --- | --- | --- |
| native | diode (level 3); resistor whose value reads a voltage; BSIM4 4.5 (level 54) |  | 3: `nmos_3p3_sab`, `pmos_3p3_sab`, `nmos_6p0_sab` |
| native | diode (level 3); resistor whose value reads a voltage |  | 4: `nplus_u_m1`, `pplus_u_m1`, `nplus_u_m2`, `pplus_u_m2` |
| neither | diode (level 3); resistor whose value reads a voltage; BSIM4 4.6 (level 54) | `pmos_6p0.0` is a BSIM4 card of version 4.6, whose equations ngspice runs another model for; jolty models version 4.5 | 1: `pmos_6p0_sab` |
| neither | BSIM4 4.6 (level 54) | `nmos_6p0_nat.0` is a BSIM4 card of version 4.6, whose equations ngspice runs another model for; jolty models version 4.5 | 4: `nmos_6p0_nat`, `pmos_6p0`, `pmos_6p0` (model), `nmos_6p0_nat` (model) |
| native | BSIM4 4.5 (level 54) |  | 6: `nmos_3p3`, `pmos_3p3`, `nmos_6p0`, `nmos_3p3` (model), `pmos_3p3` (model), `nmos_6p0` (model) |
| native | Gummel-Poon (level 1) |  | 10: `vpnp_0p42x10`, `vpnp_0p42x5`, `vpnp_10x10`, `vpnp_5x5`, `vnpn_10x10`, `vnpn_5x5`, `vnpn_0p54x16`, `vnpn_0p54x8`, `vnpn_0p54x4`, `vnpn_0p54x2` |
| native | resistor's model card; diode (level 3); resistor whose value reads a voltage |  | 5: `nplus_u`, `pplus_u`, `nplus_s`, `pplus_s`, `nwell` |
| native | resistor's model card; capacitor's model card |  | 9: `npolyf_u`, `ppolyf_u`, `npolyf_s`, `ppolyf_s`, `ppolyf_u_1k`, `ppolyf_u_2k`, `ppolyf_u_1k_6p0`, `ppolyf_u_2k_6p0`, `ppolyf_u_3k` |
| native | resistor whose value reads a voltage |  | 7: `rm1`, `rm2`, `rm3`, `tm6k`, `tm9k`, `tm11k`, `tm30k` |
| native | resistor |  | 1: `efuse` |
| native | capacitor whose value reads a voltage |  | 10: `mim_1p5fF`, `mim_1p0fF`, `nmoscap_3p3`, `pmoscap_3p3`, `nmoscap_6p0`, `pmoscap_6p0`, `nmoscap_3p3_b`, `pmoscap_3p3_b`, `nmoscap_6p0_b`, `pmoscap_6p0_b` |
| native | capacitor whose value reads a voltage; resistor |  | 1: `mim_2p0fF` |
| neither | resistor; resistor whose value reads a voltage; BSIM4 4.6 (level 54); capacitor whose value reads a voltage | `nmos_10p0_asym_core.1` is a BSIM4 card of version 4.6, whose equations ngspice runs another model for; jolty models version 4.5 | 2: `nmos_10p0_asym`, `pmos_10p0_asym` |
| native | diode (level 3) |  | 9: `np_3p3` (model), `pn_3p3` (model), `np_6p0` (model), `pn_6p0` (model), `nwp_3p3` (model), `nwp_6p0` (model), `dnwpw` (model), `dnwps` (model), `sc_diode` (model) |

### SG13G2

<!-- devices of: sg13g2 -->

| Status | Models | First problem | Devices |
| --- | --- | --- | --- |
| native | none |  | 1: `bondpad` |
| Verilog-A | Verilog-A `psp103va` | jolty has no element of letter `N` | 8: `sg13_lv_nmos`, `sg13_lv_pmos`, `nmoscl_2`, `nmoscl_4`, `sg13_hv_nmos`, `sg13_hv_pmos`, `sg13_moscap_n`, `sg13_moscap_p` |
| Verilog-A | capacitor; resistor; Verilog-A `mosvar`; resistor whose value reads a voltage; diode (level 1) | jolty has no element of letter `N` | 1: `sg13_hv_svaricap` |
| native | VBIC (level 9); resistor; capacitor |  | 6: `npn13G2`, `npn13G2_5t`, `npn13G2l`, `npn13G2l_5t`, `npn13G2v`, `npn13G2v_5t` |
| native | Gummel-Poon (level 1) |  | 2: `pnpMPA`, `mod1_pnp` (model) |
| native | resistor |  | 2: `ptap1`, `ntap1` |
| unvalued | resistor | no parameter is named `res_rpara` | 1: `Rparasitic` |
| Verilog-A | Verilog-A `r3_cmc` | jolty has no element of letter `N` | 3: `rsil`, `rhigh`, `rppd` |
| Verilog-A | Verilog-A `cap_cmomi` | jolty has no element of letter `N` | 1: `cap_cmomi` |
| Verilog-A | Verilog-A `cap_cmomf` | jolty has no element of letter `N` | 1: `cap_cmomf` |
| native | capacitor |  | 1: `cparasitic` |
| native | resistor; capacitor's model card |  | 1: `cap_cmim` |
| neither | capacitor; inductor; resistor; capacitor's model card | jolty reads no `L` here | 1: `cap_rfcmim` |
| native | diode (level 3) |  | 9: `dantenna`, `dpantenna`, `ddnwpsub`, `darea` (model), `dperim` (model), `dcorner` (model), `dparea` (model), `dpperim` (model), `dpcorner` (model) |
| native | diode (level 3); diode (level 1) |  | 2: `isolbox`, `dpwdnw` |
| native | diode (level 1) |  | 11: `diodevdd_2kv`, `diodevdd_4kv`, `diodevss_2kv`, `diodevss_4kv`, `diodevdd_mod` (model), `diodevss_mod` (model), `dsub_2kv_mod` (model), `dsub_4kv_mod` (model), `dmain_mod` (model), `drev_mod` (model), `dsub_mod` (model) |
| neither | diode (level 1); Gummel-Poon (level 1); resistor; capacitor | `AREA` accepts one value, not `((Nx*Ny)*(l*w)) /(scaling*scaling)` | 1: `schottky_nbl1` |
| native | capacitor's model card |  | 1: `cmim_core` (model) |

Native devices agree with ngspice reading the same kit at the same corner across the verification suites below:

- **Subcircuits across all three kits (typical corner, inventory sizes):** operating-point currents match ngspice within `3.4e-7` Boltzmann-constant scaling on BJT junction exponents.
  Only SG13G2 `pnpMPA` (double area scaling) and GF180MCU poly resistors (length misread as resistance) are excluded; see [departures](departures.md).
  AC admittances from 1 kHz to 100 GHz (first pin at 0.5 V, remaining pins grounded) agree after applying ngspice's 1 fs capacitor lag, excluding inexact BSIM4 body capacitances and negative-base `POW` derivatives ([departures](departures.md)).
- **Bare passive and diode model cards (default sizes):** SPICE3 resistor, capacitor, and diode cards match ngspice except where ngspice drops double-quoted values outside braces ([departures](departures.md)).
  Diode currents agree within the `6.9e-6` thermal-voltage difference at 27 °C and 85 °C from -6 V breakdown to 0.75 V forward bias.
  Documented ngspice diode departures remain excluded: sidewalls without `NS` away from `TNOM` (GF180MCU `nwp_3p3`), recombination conductance (SG13G2 antenna diodes), and `TLEVC = 1` referenced to 27 °C instead of `TNOM` (SKY130 30 °C, GF180MCU 25 °C).
- **Varactors and voltage-dependent capacitors:** SKY130 varactors and GF180MCU MIM and MOS capacitors match 1 MHz admittances from -2 V to 2 V (spanning up to 173× capacitance variation) within twice relative tolerance.
- **Gummel-Poon BJTs:** one transistor per kit and polarity (SKY130 HSPICE temperature terms, GF180MCU `TLEVC = 1` and `NKF`, SG13G2 `pnpMPA`) agrees across -40, 27, and 125 °C.
  Collector and base currents match across `Vbe` from 0.4 V to 0.9 V within twice relative tolerance plus `3.4e-7` Boltzmann scaling.
- **SG13G2 heated VBIC (`npn13G2`):** self-heated through its fifth node, warming 60 to 130 K up to 1 V base drive.
  DC currents match within twice relative tolerance across all three corners, and 85 °C gain matches to 10 GHz.
- **SKY130 MIM capacitor:** 85 °C admittance agrees up to 100 GHz across `tt`, `ll`, and `hh` corners.
- **SKY130 core [BSIM4](devices.md#bsim4) (`nfet_01v8`, `pfet_01v8`):** across 81 binned sizes at `tt`, `ff`, and `ss`, drain- and gate-swept currents agree within twice relative tolerance.
  Typical-corner admittances agree to 10 GHz, and an inverter switches within `1e-3` of supply.
- **SKY130 RF BSIM4 transistors:** four models (`rgatemod = 3`, `rdsmod = 1`, `rbodymod = 1`: 1.8 V NMOS/PMOS, LVT NMOS, 5 V NMOS) match `tt` operating-point currents and admittances to 10 GHz within twice relative tolerance.
  Amplifier stages whose ngspice `gds` matches exact current derivatives also agree in output noise to 100 MHz.

GF180MCU's `design.ngspice` enables `sw_stat_global` and `sw_stat_mismatch` by default, so differential comparisons set both switches to 0 to match nominal jolty values.
Away from nominal card temperature, heated VBIC behavior differs from ngspice as documented in [departures](departures.md).

## The process design kit

jolty pins one release of SKY130 (`sky130.KIT`) as built by open_pdks and published by ciel.
`kits.root(sky130.KIT)` downloads roughly 20 MB of ngspice models into [`cache`](../reference/configuration.md) on first use, verifies each archive's SHA-256 digest, and reuses the cached files afterward.
Setting `pdk_root` points instead to an existing directory containing `sky130A`, `gf180mcuD`, and `ihp-sg13g2` (as installed by open_pdks, volare, or ciel), matching the layout of digest-fetched GF180MCU and SG13G2 releases.
