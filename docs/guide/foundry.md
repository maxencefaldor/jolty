# Foundry processes

A chip's transistors are its foundry's, and only the foundry's models say how they behave.
jolty places SkyWater's SKY130 transistors as foundry devices: the engine checks them but does not solve them, and ngspice simulates every analysis of a circuit that holds one with the process design kit's own models.

## Placing a transistor

`sky130.mosfet(transistor, width, length, fingers=1, multiplier=1)` places one of the kit's transistors, named by its model without the library's prefix, its width all its fingers together:

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

Its terminals are `drain`, `gate`, `source` and `bulk`.
`sky130.TRANSISTORS` lists the transistors: the 1.8 V core ones, of standard, low and high threshold, the native ones of 3.3 V and 5 V, and those whose gates stand 5 V and whose drains stand 10.5 V or 16 V, as their names say.
The kit's models bin their sizes, so a width or a length that no bin holds fails when ngspice runs.

A build elaborates the circuit and checks its structure as any other, through the branches the transistor declares, a MOSFET's: the engine solves none of a foundry's devices, so the checks at the operating point do not run.

## Simulating in ngspice

`jolty simulate` and `Simulation.run` hand each analysis of a circuit that holds a foundry's device to ngspice, which reads the kit's model library and runs it at the plan's temperature, with its options, in its HSPICE compatibility mode, as the kit asks.
The circuit is written as a [SPICE netlist](spice.md#writing-spice), each transistor an instance of the kit's subcircuit, its sizes in micrometres, as the kit's library scales them, and its diffusions' areas, perimeters and squares estimated as the kit's xschem symbols estimate them, from a diffusion 0.29 µm long beside each finger.
ngspice reads the whole library, which takes seconds, before each analysis.

Its results are jolty's: reports, plots, specs and corners read them as they read the engine's.
They hold the circuit's nets and the currents of its sources and inductors, as jolty names them; a device's own unknowns and its terminals' currents ngspice keeps to itself, whatever the options [record](results.md#what-a-result-records), so a rating of a current or a power goes unjudged, and an operating point is found by `ngspice`.

An operating point, a DC sweep in even steps, a small-signal response, a transient and the noise at a voltage run this way.
A transient takes the steps ngspice chooses, and lands on the times its analysis names only where they fall.
A noise analysis gives each device's sources of noise as ngspice names them, such as a transistor's `id`, its channel's thermal noise, and `1overf`, its flicker noise; ngspice refers noise to a source, so it takes the circuit's first independent source when the analysis names none, and the result then leaves the referral out.
Any other analysis of such a circuit fails, and says so.

A netlist holds no level-1 or level-3 MOSFET beside a foundry's, since the kit scales the sizes of every MOSFET, and no foundry's device at a temperature of its own.

## Process corners

A plan's `process` is the corner of the kit's model library its foundry's devices run at, `tt`, the typical one, unless given, and each [corner](simulation.md#corners) may name its own:

```python
Plan(
    {"bias": OperatingPoint(), "gain": Ac("VIN", decade(1 * kHz, 1 * GHz, 10))},
    corners={
        "slow": Corner(process="ss", temperature=celsius(125)),
        "fast": Corner(process="ff", temperature=celsius(-40)),
    },
)
```

`sky130.CORNERS` lists them: `tt`, then `ff`, `ss`, `sf` and `fs`, fast and slow, the n-channel transistors first where they differ, and `ll`, `hh`, `hl` and `lh`, which move the resistors and capacitors too.
A process corner of a circuit that holds no foundry's device is an error.

## Mismatch

A [Monte Carlo](simulation.md#monte-carlo) analysis draws the kit's mismatch on each sample, as the kit models it at the corner's process: each transistor's threshold, oxide and mobility apart from the others', by how far its area lies from the kit's, from a seed of the sample's own, as well as the tolerances of jolty's own devices, which the netlist carries to ngspice.
A sample draws the same mismatch at every corner, so a corner runs the same transistors, and the report counts the foundry's devices whose mismatch each sample draws.
`Simulation(circuit, mismatch=seed)` draws one sample of it.

## The open process kits

`kits.read(kit, sections)` reads a process design kit's model libraries at a corner, as ngspice reads them, into a [`Library`](spice.md) of its subcircuits and models: SkyWater's SKY130, `sky130.KIT`, GlobalFoundries' GF180MCU, `gf180mcu.KIT`, and IHP's SG13G2, `sg13g2.KIT`.
A kit's files, which ngspice reads whole, come first, then one section of each of its libraries, each library's typical one unless given; `kit.sections` lists every section, each corner of its libraries and their statistical sections.
The reader takes what the kits write: `.lib` sections and `.include`, each file's path read from the directory of the file that names it, `.option scale`, which scales a MOSFET's sizes as ngspice's, parameters, a subcircuit's own among them, which its instances may give as ngspice lets them, and their expressions, quoted in braces or quotes, a subcircuit's `.if` blocks, chosen by its parameters, and binned models, a bin chosen by its card's `LMIN` to `WMAX`, which hold a transistor's width divided among its fingers, `nf`, where the kit asks for HSPICE's compatibility, as SKY130 does, or sets the option `wnflag`, as ngspice selects it.

`temper` is the temperature the library is read at, 27 °C unless given, in kelvins: SG13G2's bipolar transistors read it in their transit time.
The statistical functions `agauss`, `gauss`, `aunif`, `unif` and `limit` take their nominal values, unless a placement is drawn in a Monte Carlo sample: `library.drawn(name, Sample(seed, index), path)` places the subcircuit `name` at `path` of the circuit it goes in, each of its statistical functions drawn as ngspice defines it from a standard variate of the engine's generator, keyed by the seed, the sample's index, the placement's path and which call it is.
A sample's mismatch is then the same bits however many shards run it, and each placement draws its own.
`library.random_variables(name, path)` reads the same placement's statistical functions as the random variables of a [mismatch analysis](simulation.md#mismatch), `Mismatch(outputs, random_variables=...)`: each call that the placement's definitions make, within those it places too, of its nominal value as mean and of the standard deviation its function draws with, `agauss(x, a, s)` `a / s`, `gauss(x, r, s)` `|x| r / s`, `aunif(x, a)` `|a| / √3`, `unif(x, r)` `|x r| / √3` and `limit(x, a)` `|a|`, with the derivative of each number of its devices that moves with it, a central difference of the reading, ε^⅓ of its standard deviation either way, exact to rounding where the number is a polynomial of degree two or less in it, as the open kits write them.
A call of the library's own parameters, which every placement shares, is the process's, not a placement's, and is left out.
A resistor or a capacitor on a VBIC's thermal node, which ngspice's untyped nodes allow, is a thermal resistance or capacitance of its value.

`kits.inventory(kit)` states, for each device a kit defines, a subcircuit or a model, every element it holds, the model each needs and why jolty cannot read it.
Each is read at its parameters' defaults, but a subcircuit whose defaults are a size no bin of a transistor it holds holds, and that gives its length `l` and width `w`, is read at the middle of that transistor's first bin, which its `parameters` state.
A subcircuit whose defaults read a parameter nothing defines is read as the kit's first placement of it within another subcircuit places it, as GF180MCU's `nmos_3p3_sab` places `nplus_u_m1`, or, where none does, with each parameter that defaults to SPICE's length or width, `l` or `w`, at 10 µm, the size ngspice gives a resistor whose element and card give none, as GF180MCU's resistors and capacitors write their sizes.
Its status is `native` where jolty reads every element, `verilog-a` where only Verilog-A models are left, which ngspice loads compiled through OSDI and jolty would compile into kernels, `neither` where a SPICE model jolty lacks is left, `unread` where only a placement the reader cannot read is, such as a subcircuit's multiplier, and `unvalued` where an element reads a parameter nothing gives a value.
Of SG13G2's Verilog-A models, jolty's compiler compiles PSP 103's `psp103va`, `r3_cmc` and `mosvar` whole ([`tests/test_veriloga.py`](https://github.com/maxencefaldor/jolty/blob/main/tests/test_veriloga.py)), and stops at the MOM capacitors' `cap_cmomi` and `cap_cmomf`, whose `localparam` it does not read yet.
At their typical corners, SKY130 has 223 of its 263 devices native, GF180MCU 44 of 72 and SG13G2 14 of 53, 13 more of whose are Verilog-A models:

### SKY130

<!-- devices of: sky130 -->

| Status | Models | First problem | Devices |
| --- | --- | --- | --- |
| native | BSIM4 4.5 (level 54) |  | 91: `sky130_fd_pr__nfet_01v8`, `sky130_fd_pr__nfet_01v8_lvt`, `sky130_fd_pr__pfet_01v8`, `sky130_fd_pr__nfet_03v3_nvt`, `sky130_fd_pr__nfet_05v0_nvt`, `sky130_fd_pr__esd_nfet_01v8`, `sky130_fd_pr__pfet_01v8_lvt`, `sky130_fd_pr__pfet_01v8_hvt`, `sky130_fd_pr__esd_pfet_g5v0d10v5`, `sky130_fd_pr__pfet_g5v0d10v5`, `sky130_fd_pr__nfet_g5v0d10v5`, `sky130_fd_pr__nfet_g5v0d16v0__base`, `sky130_fd_pr__esd_nfet_g5v0d10v5`, `sky130_fd_pr__pfet_g5v0d16v0__base`, `sky130_fd_pr__cap_vpp_11p3x11p8_l1m1m2m3m4_shieldm5_nhv__base`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM10W3p00L0p50`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM04W3p00L0p50`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM10W5p00L0p50`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM04W5p00L0p50`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM10W7p00L0p50`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM04W7p00L0p50`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM02W5p00L0p50`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM02W3p00L0p50`, `sky130_fd_pr__rf_nfet_01v8_bM02W1p65L0p15`, `sky130_fd_pr__rf_nfet_01v8_bM04W1p65L0p15`, `sky130_fd_pr__rf_nfet_01v8_bM02W1p65L0p18`, `sky130_fd_pr__rf_nfet_01v8_bM04W1p65L0p18`, `sky130_fd_pr__rf_nfet_01v8_bM02W1p65L0p25`, `sky130_fd_pr__rf_nfet_01v8_bM04W1p65L0p25`, `sky130_fd_pr__rf_nfet_01v8_bM02W3p00L0p15`, `sky130_fd_pr__rf_nfet_01v8_bM04W3p00L0p15`, `sky130_fd_pr__rf_nfet_01v8_bM02W3p00L0p18`, `sky130_fd_pr__rf_nfet_01v8_bM04W3p00L0p18`, `sky130_fd_pr__rf_nfet_01v8_bM02W3p00L0p25`, `sky130_fd_pr__rf_nfet_01v8_bM04W3p00L0p25`, `sky130_fd_pr__rf_nfet_01v8_bM02W5p00L0p15`, `sky130_fd_pr__rf_nfet_01v8_bM04W5p00L0p15`, `sky130_fd_pr__rf_nfet_01v8_bM02W5p00L0p18`, `sky130_fd_pr__rf_nfet_01v8_bM04W5p00L0p18`, `sky130_fd_pr__rf_nfet_01v8_bM02W5p00L0p25`, `sky130_fd_pr__rf_nfet_01v8_bM04W5p00L0p25`, `sky130_fd_pr__rf_pfet_01v8_bM02W1p65L0p15`, `sky130_fd_pr__rf_pfet_01v8_bM04W1p65L0p15`, `sky130_fd_pr__rf_pfet_01v8_bM02W1p65L0p18`, `sky130_fd_pr__rf_pfet_01v8_bM04W1p65L0p18`, `sky130_fd_pr__rf_pfet_01v8_bM02W1p65L0p25`, `sky130_fd_pr__rf_pfet_01v8_bM04W1p65L0p25`, `sky130_fd_pr__rf_pfet_01v8_bM02W3p00L0p15`, `sky130_fd_pr__rf_pfet_01v8_bM04W3p00L0p15`, `sky130_fd_pr__rf_pfet_01v8_bM02W3p00L0p18`, `sky130_fd_pr__rf_pfet_01v8_bM04W3p00L0p18`, `sky130_fd_pr__rf_pfet_01v8_bM02W3p00L0p25`, `sky130_fd_pr__rf_pfet_01v8_bM04W3p00L0p25`, `sky130_fd_pr__rf_pfet_01v8_bM02W5p00L0p15`, `sky130_fd_pr__rf_pfet_01v8_bM04W5p00L0p15`, `sky130_fd_pr__rf_pfet_01v8_bM02W5p00L0p18`, `sky130_fd_pr__rf_pfet_01v8_bM04W5p00L0p18`, `sky130_fd_pr__rf_pfet_01v8_bM02W5p00L0p25`, `sky130_fd_pr__rf_pfet_01v8_bM04W5p00L0p25`, `sky130_fd_pr__rf_nfet_01v8_bM02`, `sky130_fd_pr__rf_nfet_01v8_bM02W3p00`, `sky130_fd_pr__rf_nfet_01v8_bM02W5p00`, `sky130_fd_pr__rf_nfet_01v8_bM04`, `sky130_fd_pr__rf_nfet_01v8_bM04W3p00`, `sky130_fd_pr__rf_nfet_01v8_bM04W5p00`, `sky130_fd_pr__rf_nfet_01v8_lvt_bM02`, `sky130_fd_pr__rf_nfet_01v8_lvt_bM02W3p00`, `sky130_fd_pr__rf_nfet_01v8_lvt_bM02W5p00`, `sky130_fd_pr__rf_nfet_01v8_lvt_bM02W5p00L0p18`, `sky130_fd_pr__rf_nfet_01v8_lvt_bM02W5p00L0p25`, `sky130_fd_pr__rf_nfet_01v8_lvt_bM04`, `sky130_fd_pr__rf_nfet_01v8_lvt_bM04W3p00`, `sky130_fd_pr__rf_nfet_01v8_lvt_bM04W5p00`, `sky130_fd_pr__rf_nfet_01v8_lvt_bM04W5p00L0p18`, `sky130_fd_pr__rf_nfet_01v8_lvt_bM04W5p00L0p25`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM02`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM02W5p00`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM04`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM04W5p00`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM04W7p00`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM10`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM10W5p00`, `sky130_fd_pr__rf_nfet_g5v0d10v5_bM10W7p00`, `sky130_fd_pr__rf_pfet_01v8_bM02`, `sky130_fd_pr__rf_pfet_01v8_bM02W3p00`, `sky130_fd_pr__rf_pfet_01v8_bM02W5p00`, `sky130_fd_pr__rf_pfet_01v8_bM04`, `sky130_fd_pr__rf_pfet_01v8_bM04W3p00`, `sky130_fd_pr__rf_pfet_01v8_bM04W5p00`, `sky130_fd_pr__pfet_01v8_mvt`, `sky130_fd_pr__special_nfet_pass_lvt` |
| neither | BSIM4 4.5 (level 54); resistor; diode (level 3) | `sky130_fd_pr__model__parasitic__diode_ps2nw` is a diode of level 3; jolty models level 1 | 2: `sky130_fd_pr__nfet_g5v0d16v0`, `sky130_fd_pr__pfet_g5v0d16v0` |
| neither | BSIM4 4.5 (level 54); resistor whose value reads a voltage; diode (level 3) | `sky130_fd_pr__model__parasitic__diode_ps2dn__extended_drain` is a diode of level 3; jolty models level 1 | 5: `sky130_fd_pr__nfet_20v0`, `sky130_fd_pr__nfet_20v0_reverse_iso`, `sky130_fd_pr__nfet_20v0_iso`, `sky130_fd_pr__nfet_20v0_zvt`, `sky130_fd_pr__nfet_20v0_nvt` |
| neither | BSIM4 4.62 (level 54); resistor whose value reads a voltage; diode (level 3) | `sky130_fd_pr__pfet_20v0__base.0` is a BSIM4 card of version 4.62, whose equations ngspice runs another model for; jolty models version 4.5 | 1: `sky130_fd_pr__pfet_20v0` |
| native | Gummel-Poon (level 1) |  | 4: `sky130_fd_pr__pnp_05v5_W0p68L0p68`, `sky130_fd_pr__pnp_05v5_W3p40L3p40`, `sky130_fd_pr__npn_05v5_W1p00L1p00`, `sky130_fd_pr__npn_05v5_W1p00L2p00` |
| neither | Gummel-Poon (level 1); resistor; diode (level 3) | `sky130_fd_pr__model__parasitic__diode_ps2dn` is a diode of level 3; jolty models level 1 | 1: `sky130_fd_pr__npn_11v0_W1p00L1p00` |
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
| neither | resistor's model card | jolty's devices have no counterpart to `TREF` of `sky130_fd_pr__res_generic_po__monte` | 1: `sky130_fd_pr__res_generic_po` |
| neither | resistor's model card; diode (level 3) | `sky130_fd_pr__diode_pw2nd_05v5` is a diode of level 3; jolty models level 1 | 4: `sky130_fd_pr__res_generic_nd`, `sky130_fd_pr__res_generic_pd`, `sky130_fd_pr__res_generic_nd__hv`, `sky130_fd_pr__res_generic_pd__hv` |
| native | resistor's model card |  | 24: `short`, `sky130_fd_pr__res_generic_nd` (model), `sky130_fd_pr__res_generic_pd` (model), `sky130_fd_pr__res_generic_nd__hv` (model), `sky130_fd_pr__res_generic_pd__hv` (model), `sky130_fd_pr__res_generic_po` (model), `sky130_fd_pr__res_generic_nw` (model), `sky130_fd_pr__res_generic_l1` (model), `sky130_fd_pr__res_generic_m1` (model), `sky130_fd_pr__res_generic_m2` (model), `sky130_fd_pr__res_generic_m3` (model), `sky130_fd_pr__res_generic_m4` (model), `sky130_fd_pr__res_generic_m5` (model), `sky130_fd_pr__res_generic_r1` (model), `mrcn` (model), `mrcp` (model), `mrcp1` (model), `mrcl1` (model), `mrcvia` (model), `mrcvia2` (model), `mrcvia3` (model), `mrcvia4` (model), `mrcrdlcon` (model), `short` (model) |
| neither | diode (level 3) | `sky130_fd_pr__diode_pw2nd_05v5` is a diode of level 3; jolty models level 1 | 20: `sky130_fd_pr__diode_pw2nd`, `sky130_fd_pr__diode_pw2nd_05v5`, `sky130_fd_pr__model__parasitic__diode_pw2dn__extended_drain` (model), `sky130_fd_pr__diode_pw2nd_05v5__extended_drain` (model), `sky130_fd_pr__model__parasitic__diode_ps2dn__extended_drain` (model), `sky130_fd_pr__nfet_20v0_zvt__parasitic__diode_ps2dn__extended_drain` (model), `sky130_fd_pr__pfet_20v0__parasitic__diode_pw2dn` (model), `sky130_fd_pr__nfet_20v0_nvt__parasitic__diode_ps2dn` (model), `sky130_fd_pr__model__parasitic__diode_ps2nw` (model), `sky130_fd_pr__diode_pw2nd_05v5` (model), `sky130_fd_pr__diode_pd2nw_05v5` (model), `sky130_fd_pr__diode_pd2nw_05v5_hvt` (model), `sky130_fd_pr__diode_pw2nd_11v0` (model), `sky130_fd_pr__diode_pd2nw_11v0` (model), `sky130_fd_pr__diode_pw2nd_05v5_nvt` (model), `sky130_fd_pr__diode_pw2nd_05v5_lvt` (model), `sky130_fd_pr__diode_pd2nw_05v5_lvt` (model), `sky130_fd_pr__model__parasitic__diode_pw2dn` (model), `sky130_fd_pr__model__parasitic__diode_ps2dn` (model), `sky130_fd_pr__pfet_g5v0d16v0__parasitic__diode_pw2dn` (model) |
| native | capacitor's model card |  | 43: `mcp1f` (model), `mcl1f` (model), `mcl1d` (model), `mcl1p1` (model), `mcm1f` (model), `mcm1d` (model), `mcm1p1` (model), `mcm1l1` (model), `mcm2f` (model), `mcm2d` (model), `mcm2p1` (model), `mcm2l1` (model), `mcm2m1` (model), `mcm3f` (model), `mcm3d` (model), `mcm3p1` (model), `mcm3l1` (model), `mcm3m1` (model), `mcm3m2` (model), `mcm4f` (model), `mcm4d` (model), `mcm4p1` (model), `mcm4l1` (model), `mcm4m1` (model), `mcm4m2` (model), `mcm4m3` (model), `mcm5f` (model), `mcm5d` (model), `mcm5p1` (model), `mcm5l1` (model), `mcm5m1` (model), `mcm5m2` (model), `mcm5m3` (model), `mcm5m4` (model), `mcrdlf` (model), `mcrdld` (model), `mcrdlp1` (model), `mcrdll1` (model), `mcrdlm1` (model), `mcrdlm2` (model), `mcrdlm3` (model), `mcrdlm4` (model), `mcrdlm5` (model) |

### GF180MCU

<!-- devices of: gf180mcu -->

| Status | Models | First problem | Devices |
| --- | --- | --- | --- |
| neither | diode (level 3); resistor whose value reads a voltage; BSIM4 4.5 (level 54) | `np_junction` is a diode of level 3; jolty models level 1 | 3: `nmos_3p3_sab`, `pmos_3p3_sab`, `nmos_6p0_sab` |
| neither | diode (level 3); resistor whose value reads a voltage | `np_junction` is a diode of level 3; jolty models level 1 | 4: `nplus_u_m1`, `pplus_u_m1`, `nplus_u_m2`, `pplus_u_m2` |
| neither | diode (level 3); resistor whose value reads a voltage; BSIM4 4.6 (level 54) | `pn_junction` is a diode of level 3; jolty models level 1 | 1: `pmos_6p0_sab` |
| neither | BSIM4 4.6 (level 54) | `nmos_6p0_nat.0` is a BSIM4 card of version 4.6, whose equations ngspice runs another model for; jolty models version 4.5 | 4: `nmos_6p0_nat`, `pmos_6p0`, `pmos_6p0` (model), `nmos_6p0_nat` (model) |
| native | BSIM4 4.5 (level 54) |  | 6: `nmos_3p3`, `pmos_3p3`, `nmos_6p0`, `nmos_3p3` (model), `pmos_3p3` (model), `nmos_6p0` (model) |
| native | Gummel-Poon (level 1) |  | 10: `vpnp_0p42x10`, `vpnp_0p42x5`, `vpnp_10x10`, `vpnp_5x5`, `vnpn_10x10`, `vnpn_5x5`, `vnpn_0p54x16`, `vnpn_0p54x8`, `vnpn_0p54x4`, `vnpn_0p54x2` |
| neither | resistor's model card; diode (level 3); resistor whose value reads a voltage | `np_3p3` is a diode of level 3; jolty models level 1 | 5: `nplus_u`, `pplus_u`, `nplus_s`, `pplus_s`, `nwell` |
| native | resistor's model card; capacitor's model card |  | 9: `npolyf_u`, `ppolyf_u`, `npolyf_s`, `ppolyf_s`, `ppolyf_u_1k`, `ppolyf_u_2k`, `ppolyf_u_1k_6p0`, `ppolyf_u_2k_6p0`, `ppolyf_u_3k` |
| native | resistor whose value reads a voltage |  | 7: `rm1`, `rm2`, `rm3`, `tm6k`, `tm9k`, `tm11k`, `tm30k` |
| native | resistor |  | 1: `efuse` |
| native | capacitor whose value reads a voltage |  | 10: `mim_1p5fF`, `mim_1p0fF`, `nmoscap_3p3`, `pmoscap_3p3`, `nmoscap_6p0`, `pmoscap_6p0`, `nmoscap_3p3_b`, `pmoscap_3p3_b`, `nmoscap_6p0_b`, `pmoscap_6p0_b` |
| native | capacitor whose value reads a voltage; resistor |  | 1: `mim_2p0fF` |
| neither | resistor; resistor whose value reads a voltage; BSIM4 4.6 (level 54); capacitor whose value reads a voltage | `nmos_10p0_asym_core.1` is a BSIM4 card of version 4.6, whose equations ngspice runs another model for; jolty models version 4.5 | 2: `nmos_10p0_asym`, `pmos_10p0_asym` |
| neither | diode (level 3) | `np_3p3` is a diode of level 3; jolty models level 1 | 9: `np_3p3` (model), `pn_3p3` (model), `np_6p0` (model), `pn_6p0` (model), `nwp_3p3` (model), `nwp_6p0` (model), `dnwpw` (model), `dnwps` (model), `sc_diode` (model) |

### SG13G2

<!-- devices of: sg13g2 -->

| Status | Models | First problem | Devices |
| --- | --- | --- | --- |
| native | none |  | 1: `bondpad` |
| Verilog-A | Verilog-A `psp103va` | jolty has no element of letter `N` | 8: `sg13_lv_nmos`, `sg13_lv_pmos`, `nmoscl_2`, `nmoscl_4`, `sg13_hv_nmos`, `sg13_hv_pmos`, `sg13_moscap_n`, `sg13_moscap_p` |
| neither | capacitor; resistor; Verilog-A `mosvar`; resistor whose value reads a voltage; diode (level 1) | jolty has no element of letter `N` | 1: `sg13_hv_svaricap` |
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
| neither | diode (level 3) | `darea` is a diode of level 3; jolty models level 1 | 9: `dantenna`, `dpantenna`, `ddnwpsub`, `darea` (model), `dperim` (model), `dcorner` (model), `dparea` (model), `dpperim` (model), `dpcorner` (model) |
| neither | diode (level 3); diode (level 1) | `dinner_mod` is a diode of level 3; jolty models level 1 | 2: `isolbox`, `dpwdnw` |
| neither | diode (level 1) | jolty's devices have no counterpart to `ISR` of `diodevdd_mod` | 11: `diodevdd_2kv`, `diodevdd_4kv`, `diodevss_2kv`, `diodevss_4kv`, `diodevdd_mod` (model), `diodevss_mod` (model), `dsub_2kv_mod` (model), `dsub_4kv_mod` (model), `dmain_mod` (model), `drev_mod` (model), `dsub_mod` (model) |
| neither | diode (level 1); Gummel-Poon (level 1); resistor; capacitor | jolty's devices have no counterpart to `IK` of `dmain_mod` | 1: `schottky_nbl1` |
| native | capacitor's model card |  | 1: `cmim_core` (model) |

The native devices agree with ngspice reading the same kit at the same corner: every native subcircuit of the three kits at its typical corner and at the size the inventory reads it at, but SG13G2's `pnpMPA`, whose area of 2 ngspice multiplies its reverse saturation current by twice, and GF180MCU's poly resistors, whose bodies ngspice gives their length in meters for their resistance ([departures](departures.md)), its current at an operating point, a bipolar transistor's within the two engines' Boltzmann constants' 3.4e-7 times its junctions' exponents too, and, but for a BSIM4's, whose capacitances with respect to the body ngspice computes inexactly, and a resistor's whose value reads a voltage, whose `POW` of a negative base ngspice differentiates with the wrong sign ([departures](departures.md)), from 1 kHz to 100 GHz, with its first pin at 0.5 V and its others grounded, a capacitor's whose value reads a voltage with its current delayed by the 1 fs ngspice's differentiation of it lags by ([departures](departures.md)); every native card of a resistor or a capacitor, SPICE3's, a bare element of each at its card's default size, alike, but for a value in double quotes outside braces, which ngspice drops ([departures](departures.md)); every native device of a capacitor of a charge or of a value that reads its voltage, SKY130's varactors and GF180MCU's MIM and MOS capacitors, its admittance at 1 MHz from −2 to 2 V, across which their capacitances move up to 173-fold, within twice the relative tolerance; one Gummel-Poon transistor of each kit and polarity, SKY130's of HSPICE's temperature coefficients, GF180MCU's of `TLEVC` 1 and `NKF`, and SG13G2's `pnpMPA`, its collector and base currents as its base sweeps from 0.4 to 0.9 V at −40, 27 and 125 °C, within twice the relative tolerance and the two engines' Boltzmann constants' 3.4e-7 times each junction's exponents; SG13G2's `npn13G2`, a VBIC heated through its fifth node, within twice the relative tolerance from off to 1 V of base drive, where it warms by 60 to 130 K, at its three corners, and its gain to 10 GHz at 85 °C, which reads its transit time at that temperature; SKY130's MIM capacitor, its admittance to 100 GHz at its `tt`, `ll` and `hh` corners at 85 °C; SKY130's `nfet_01v8` and `pfet_01v8`, [BSIM4](devices.md#bsim4) cards of version 4.5, in 81 sizes each in a bin of its own, their drain currents swept by their drains and their gates within twice the relative tolerance at `tt`, `ff` and `ss`, their admittances to 10 GHz at `tt`, and an inverter of the two switching within 1e-3 of the supply at each corner; four of SKY130's RF transistors, BSIM4 cards of `rgatemod` 3 with `rdsmod` and `rbodymod` 1, of the 1.8 V n- and p-channels, the low-threshold n-channel and the 5 V n-channel, their currents at an operating point and their admittances to 10 GHz at `tt` within twice the relative tolerance, and the stage of each whose output conductance ngspice computes as its current's derivative making ngspice's noise to 100 MHz.
ngspice runs GF180MCU with its statistical switches on, `sw_stat_global` and `sw_stat_mismatch` set by `design.ngspice`, drawing every instance's mismatch, so the comparisons give it both at 0, where it computes the nominal values jolty reads.
Away from its card's nominal temperature, a heated VBIC departs from ngspice's, as the [departures](departures.md) state.

## The process design kit

jolty pins one release of the kit, `sky130.KIT`, as open_pdks builds it and ciel publishes it.
`kits.root(sky130.KIT)` fetches its ngspice models, about 20 MB, into the [`cache`](../reference/configuration.md) the first time, each archive checked against its SHA-256 digest, and reads them from there after.
The setting `pdk_root` names the kits installed elsewhere instead, the directory that holds `sky130A`, `gf180mcuD` and `ihp-sg13g2`, as open_pdks, volare and ciel install them; GF180MCU's and SG13G2's pinned files, fetched from their foundries' repositories by digest, are laid out alike.
