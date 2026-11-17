# Research: models and coverage

15 November 2026, for the lead's `models.md`.
Read-only research.
jolty's files are cited at `main` `68c3f75` in the repository; ngspice at 47's unpacked source; Xyce at `main` `e5b7b9b` (19 October 2026); both in the session scratchpad.
An estimate is marked as one.

## 1. Summary

- jolty simulates 26 analog models natively, from SPICE3's core devices to BSIM3v3.3, VDMOS, memristors and thermal elements, with VBIC 1.2 about to merge; each model is a pure kernel with exact derivatives with respect to its voltages and to every parameter.
    It solves no foundry's transistor: SKY130's BSIM4 4.5 cards go to ngspice, which then runs only five analyses, and GF180MCU and IHP SG13G2 do not run at all.
- ngspice 47 has 57 device directories and loads any Verilog-A model through OSDI.
    Xyce has 25 Verilog-A models, which ADMS turned into 856,000 checked-in lines of C++.
    Neither is exact everywhere: ngspice's hand-written derivatives leave out or double terms that jolty's guide lists, its sensitivities perturb each parameter by 1e-6, and OSDI passes no parameter derivatives at all.
- Move 1: BSIM4 4.0 to 4.5 as ngspice's `BSIM4v5` runs it (about twice BSIM3: 766 parameters to 437), plus SKY130's level 3 diodes, its bipolar transistors' HSPICE temperature terms, and a reader for kits (`.lib`, `agauss`, `temper`).
    That gives SKY130 and GF180MCU natively.
- Move 2 (lead): a Verilog-A compiler of jolty's own that writes safe Rust kernels generic over `Real`, so `unsafe` stays forbidden and every compact model has exact bias, parameter and higher derivatives, which OSDI cannot give.
    PSP 103.6, R3_CMC and MOSVAR come first, for IHP SG13G2, whose HBTs are VBIC level 9 cards; the rest of the CMC's catalogue follows (the eleven models counted come to about 42,000 lines of Verilog-A).
- Move 3 (lead): every model judged mechanically against what defines it: the CMC's QA tests and reference results, ngspice for SPICE3's models, laws no reference tests (charge, symmetry, polarity, continuity), and the measured silicon that SKY130 and IHP publish.

## 2. Where jolty stands

### 2.1 The models it has

`Library::standard()` holds 29 models (`crates/jolty-models/src/lib.rs:127-162`, listed by the test at `lib.rs:419-455`): 26 analog ones, 2 logic ones, and `foundry_mosfet`, a placeholder whose evaluation adds nothing (`foundry.rs:1-6`, `foundry.rs:96-106`).

| Family | Models | Equations from | Checked against |
| --- | --- | --- | --- |
| Passive | resistor (TC1, TC2, TCE, VC1, VC2, flicker), capacitor, inductor, coupled inductors | Elementary, SPICE | Closed forms, ngspice |
| Sources | voltage and current sources with DC, pulse, sine and piecewise waveforms; four linear controlled sources; behavioral sources with exact gradients | SPICE, jolty's expression language | ngspice |
| Switches | voltage- and current-controlled, PSpice's law | PSpice | ngspice's XSPICE switches |
| Semiconductors | diode (SPICE level 1 with breakdown), JFET (with Sydney's doping tail), Gummel-Poon BJT (substrate, excess phase), MOSFET levels 1 and 3 (Meyer), BSIM3v3.3 (levels 8 and 49), VDMOS | ngspice 47, the BSIM3v3.3 manual | ngspice 47 |
| Others | VTEAM and HP memristors, a photocell | Kvatinsky et al. 2015, Strukov et al. 2008, Senba's GL5528 datasheet | Closed forms, and ngspice through behavioral netlists (`tests/test_ngspice.py:1061-1174`) |
| Thermal | thermal resistance, capacitance, heat source; seven models heat themselves through `Thermal<K>` | | Closed forms |
| In progress | VBIC 1.2: a 2,675-line kernel on branch `vbic`; its comparisons with ngspice pass, two departures explained (`models.md` plan item 1) | ngspice's `vbicload.c` | ngspice 47 |

Real parts: seven, the 2N3904, 2N3906, 1N4148, IRF511, IRF9230, IRFZ44 and MTP50P03HDL (`docs/guide/parts.md:11-17`).
Vendors' macromodels read and compared with ngspice: four (`docs/guide/spice.md:60-73`).

### 2.2 How exact they are

- **Derivatives:** every kernel writes its equations once on `Dual` numbers (`dual.rs:1-6`), generic over `Real`, so the same code gives the Jacobian and, with `Gradient`, the exact derivative with respect to any parameter (`real.rs:18-23`, the twins of `group.rs:1649-1670`).
  The guide states where ngspice's hand-written derivatives are wrong or missing: the BJT's falling base resistance (`devices.md:378`), the MOSFET's forward-biased bulk (`devices.md:445`), BSIM3's clamped bulk voltage and its `capmod` 0 poly depletion counted twice (`devices.md:485`), and VDMOS's quasi-saturation, mobility and reverse channel-length modulation (`devices.md:549`).
- **Charge:** the MOSFET carries Meyer's capacitive currents exactly through displacement unknowns (`devices.md:443`), and VDMOS holds its gate-drain charge in closed form where ngspice integrates the capacitance by the trapezoidal rule (`devices.md:541`).
- **Agreement:** BSIM3 against ngspice 47 holds sweeps within 1e-6 of the signal, AC within 6e-9, noise within a relative 1e-5 and transients within 1e-3 of the swing, over 58 comparisons (an archived note, not kept).
- **Refusal over approximation:** a mode left out is refused by name, never ignored: BSIM3's `nqsmod`, `acnqsmod`, `acm` and versions 3.0 to 3.2 (`devices.md:489-491`); "nothing is silently dropped" (`spice.md:58`).
- **Sensitivities:** exact for every parameter of every device, at DC and over frequency by the adjoint, over time by the direct method (`simulation.md:88-111`).
    One part is not exact: over frequency, the operating point's movement is "a central difference ...
    since devices give no second derivatives" (`simulation.md:104`).
- **Known inexactness:** SPICE expressions read `TEMP` and `VT` at 27 °C, whatever the temperature (`spice.md:56`).
- **What exactness costs:** a MOSFET's evaluation takes about 240 ns an iteration on `inverter_chain_500`, against 110 ns per MOSFET for ngspice's whole iteration, factoring and solving included (a brief, not kept).
    A subagent is working on it.

### 2.3 Process kits

- **SKY130:** 11 transistors (`src/jolty/spice/sky130.py:111-126`), from a pinned open_pdks build (`sky130.py:35`), are foundry devices.
    The engine checks their structure and hands every analysis to ngspice.
    Only an operating point, a DC sweep, AC, a transient and noise run; "Any other analysis of such a circuit fails" (`docs/guide/foundry.md`).
    So a kit circuit gets no sensitivities, periodic steady state, pole-zero or stability analysis, and none of jolty's exact derivatives; its corners and Monte Carlo samples each run in ngspice.
    The kit's diodes, bipolar transistors, resistors and capacitors cannot be placed.
- **GF180MCU and IHP SG13G2:** nothing.
- `AGENTS.md:11` states it: "a foundry's transistor models, which it cannot load yet, run in ngspice on an exported netlist."

### 2.4 The SPICE reader, as a kit reader

It reads diodes, JFETs, BJTs, MOSFETs of levels 1 and 3, BSIM3 (levels 8 and 49, version 3.3), VDMOS, switches, `POLY`, `VALUE` and `TABLE` sources, `B` sources, subcircuits and `.param` (`spice.md:31-58`).
A library may hold only `.subckt`, `.model` and `.param`: anything else fails with "a library defines subcircuits, models and parameters only" (`src/jolty/spice/library.py:111-113`).
So `.lib` sections, `.include` and `.option scale` are refused, and the reader knows neither `agauss`/`gauss` nor `temper` (a search of `src/jolty/spice/` finds none).

### 2.5 What a model costs, and why

**What the architecture gives each model for free:**

- **Derivatives:** no derivative code.
  jolty's ports are a quarter to a third of ngspice's C: BSIM3 4,712 lines of kernel to 13,744 of C, VBIC 2,675 to 9,745, VDMOS 1,364 to 5,699 (`wc` on both trees).
- **Parameter sensitivities and heating:** the twins differentiate every parameter, `Thermal<K>` heats any kernel (`heat.rs:1-10`), and `Hyper` carries the second order a heated twin needs (`real.rs:396-415`).
- **Evaluation:** groups run a kernel's instances of one shape serially or across threads into a slab, gathered in one fixed order, so determinism and parallelism come with no code in the model (`group.rs:1-15`).
- **Checks and reference:** the structural checks read the branches a kernel declares (`Kernel::interface`), and the parameters' JSON schema generates the reference pages.
- **Shared parts:** `kernel/parts.rs` holds node sets, two-terminal pairs, current branches, series resistances with small ones carried as currents (`parts.rs:259-404`), limiting with linear extension (`Bias`, `parts.rs:425-457`) and n/p orientation (`Oriented`, `parts.rs:459-510`).
  `junction.rs` holds the junction physics.
  A new MOSFET level costs little: level 3, in one commit with the diode's breakdown, took 905 lines across 17 files, its channel 162 of them (`766ccd9`).

**What each model still pays:**

- **Parameters declared by hand:** BSIM3's `parameters.rs` is 2,121 lines for 404 fields, 45% of the kernel.
    A Verilog-A source already carries each parameter's name, default, range, unit and description.
- **Per-model Python and SPICE maps:** BSIM3 added 375 lines to `devices.py`, 218 to the reader and 63 to the exporter (`6fe8824`); VBIC added 378, 287 and 108 (branch `vbic`).
  The dispatchers that grow a branch per model needed splitting to stay within ruff's complexity bound (`quality.md`).
  Parallel model work conflicts in `kernels.rs`, `kicad.py`, `export.py` and `models.py` (`handoff-2026-11-14.md:49`).
- **Review:** BSIM3's 6,631 lines merged on their tests, not read line by line (`handoff-2026-11-14.md:48`).
    Each hand port is that much code to trust.
- **Dense derivatives:** a `Dual<R, N>` carries all `N` partials through every operation: `N` is 3 for the MOSFET and BSIM3 (`mosfet.rs:538`, `bsim3.rs:69`) and 9 for VBIC (`vbic.rs:103`).
  Compact models in Verilog-A reach 12 to 15 nodes and 18 to 32 probes.
  Xyce's generated code counts 13 nodes and 32 probes for BSIM-CMG, 13 and 20 for PSP 103, 15 and 27 for HICUM/L2, 12 and 27 for MEXTRAM, and 14 and 29 for BSIM-SOI (`evaluateModelEquations` in each `N_DEV_ADMS*.h`).
  Dense forward differentiation of such a model carries a dozen or more numbers through each operation.

**The kernel already has OSDI's shape.**
A kernel stamps a residual and a charge for each local unknown, and a conductance and a capacitance for each local entry (`kernel.rs:395-492`).
OSDI's descriptor gives a resistive and a reactive residual per node and their two Jacobians (`osdi.h:126-208` in ngspice), and Verilog-A's static and `ddt()` contributions are the same split.
The crate says so: "Compiled Verilog-A models use the same form, so native and foundry models plug in alike" (`lib.rs:5-7`).
The unknowns' quantities cover Verilog-A's disciplines: voltage, current, displacement, temperature and state (`binding.rs:27-42`).
Optional internal nodes are already a layout decision (`Series::behind`, `parts.rs:276-295`), as Verilog-A's collapsible nodes need.
Only one part of the stamp has no counterpart in OSDI: the scale, the largest term of each equation, which jolty's convergence test reads (`kernel.rs:449-458`).

**How Verilog-A maps onto a kernel**, construct by construct, which is what a compiler of jolty's own would have to do:

| Verilog-A | In a jolty kernel | What is missing |
| --- | --- | --- |
| `parameter real p = d from [a:b]`, with `desc` and `units` attributes | A field of `Parameters<R>`, its default, its range checked in `prepare`, its description and SI dimension in the schema | Nothing |
| `@(initial_model)`, `@(initial_instance)`, and any code that reads no probe | `prepare(parameters, temperature)` | The split must be computed by dependence, as OpenVAF does; Xyce's ADMS needed `@initial_instance` blocks added by hand (`README_Xyce`) |
| A node collapsed when a parameter makes a branch zero (`V(a,b) <+ 0`) | `Shape` and `layout`, as `Series::behind` does | Nothing |
| `I(a,b) <+ f(V)` and `I(a,b) <+ ddt(q(V))` | `Nodes::current` and `Nodes::charge`, with their derivatives | Nothing |
| `V(a,b) <+ e` | `CurrentBranch` | Nothing |
| `$limit(V(a,b), "pnjlim", ...)` | `Bias`, the previous voltage in `Memory`, `mark_limited` | Nothing |
| `white_noise`, `flicker_noise` | `NoiseSource` | Correlated noise through internal noise nodes, to be checked |
| Output variables | `variable_names` and `variables` | Nothing |
| A `thermal` node, `Temp()`, `Pwr() <+` | An unknown of `Quantity::Temperature` | Code that reads the temperature then depends on a probe, so it moves from `prepare` to `evaluate` |
| `$simparam("gmin")`, `$temperature`, `$vt` | `Local::gmin`, `prepare`'s temperature | Other `$simparam` names |
| `analysis("ac")`, `analysis("noise")` | `Local::transient` only (`kernel.rs:326-341`) | A flag per analysis |
| `$warning`, `$strobe` | No warning channel: BSIM3's non-fatal checks report nothing (an archived note, not kept) | A channel for facts found at `prepare` |
| `$bound_step` | `Kernel::breakpoints` and `Kernel::choices` only | A step bound per device |
| `absdelay`, `idt`, `laplace_*`, `@cross`, `$abstime` | None | Rare in compact models; OpenVAF lacks most of them too |

**Licences of the ports.**
BSIM's terms ask that "In any product based on the software, the users agree to acknowledge the UC Berkeley BSIM Research Group" in the product's documentation, and that users "reproduce any copyright notice which appears on the software on any copy or modification" (`bsim4/B4TERMS_OF_USE` in ngspice; `bsim3/B3TERMS_OF_USE` too).
jolty's BSIM3 cites the manual and ngspice's files (`bsim3.rs:9-16`) but carries neither the Regents' copyright notice nor the terms; a search of jolty's sources finds neither.
Whether a translation is a "modification" is for the owner to judge.

## 3. The references

### 3.1 ngspice 47

**Devices:** 57 directories under `src/spicelib/devices/`, and a model card's level picks among them (`src/spicelib/parser/inpdomod.c`):

- **MOSFETs:** levels 1, 2, 3, 6 and 9; BSIM1 and BSIM2 (4, 5).
- **BSIM3:** levels 8 and 49 run BSIM3v0, v1, v32 or 3.3 by the card's version (`inpdomod.c:271-284`).
- **BSIM4:** levels 14 and 54 run versions 4.0 to 4.5 through one implementation, `BSIM4v5`, and 4.6, 4.7 and 4.8 through `BSIM4v6`, `BSIM4v7` and `BSIM4` (`inpdomod.c:297-309`).
    Each copy is 23,932 to 27,630 lines.
- **SOI:** B3SOI FD, DD and PD (55-57), B4SOI (10, 58) and SOI3 (60).
- **HiSIM:** HiSIM2 (68), HiSIM_HV 1 and 2 (73).
- **Bipolar:** Gummel-Poon (1, 2), VBIC (4, 9) and HICUM/L2 v2.4.0 (8) (`inpdomod.c:49-75`).
- **Others:** JFET 1 and 2; MESFET (Statz), MESA and HFET 1 and 2; VDMOS; the transmission lines `tra`, `ltra`, `txl` and `cpl`; CIDER's numerical devices; XSPICE's code models.

**Verilog-A through OSDI:** ngspice implements OSDI 0.3 (`src/osdi/osdi.h:21-22`).
It loads a compiled model with `dlopen` (`osdiregistry.c:31`), binds `$limit` to `pnjlim`, `limvds` and `fetlim` (`osdiregistry.c:434-436`), and evaluates instances in OpenMP tasks (`osdiload.c:190-204`).
An OSDI device has load, AC, pole-zero, truncation and noise functions, but no sensitivity or distortion functions (`osdiinit.c:188-205`).
ADMS "is no longer part of the ngspice development project" and "will be removed" (ngspice's OSDI page).

**Where it falls short:**

- **Derivatives:** they are hand-written and, in places, wrong or missing (§2.2).
- **Sensitivities:** `.sens` perturbs each parameter by `Sens_Delta = 0.000001` (`src/spicelib/analysis/cktsens.c:24`), a finite difference.
- **Distortion:** `.disto` exists for ten SPICE3 models alone: MOS1, MOS2, MOS3, MOS9, BSIM1, the BJT, the diode, the JFET, the MESFET and VDMOS.
    BSIM3, BSIM4, VBIC, HICUM and every OSDI model have none (`DEVdisto` in each `*init.c`).
- **Charge:** capacitance-based models integrate capacitances, so their charge is not conserved (VDMOS: `devices.md:541`; the MESFET's path integral: an archived note, not kept brief, not kept).
- **Pole-zero:** "Unfortunately this code is (since its inception in spice3) not free of bugs" (ngspice's `ANALYSES`).
- **Versions:** every version of a model is a separate copy of its code.
- **HICUM:** the built-in HICUM differentiates with scalar dual numbers (`hicum2/hicumL2.cpp:10`, `hicumL2.cpp:21`), which its OSDI version outruns in one benchmark (§3.3).

### 3.2 Xyce

**Native C++ models:** MOSFET levels 1, 2, 3 and 6, BSIM3, BSIM3-SOI 3.2, BSIM4 4.6.1, 4.7.0 and 4.8.2, VDMOS, MESFET, JFET, BJT, TRA, LTRA, a lumped line, three memristors, a Touchstone multiport (`YLin`), IBIS buffers and power-grid devices (`src/DeviceModelPKG/OpenModels/`, `IBISModels/`).

**The ADMS route:** 25 Verilog-A models turned into C++ and checked into the source (`src/DeviceModelPKG/ADMS/`, 856,148 lines):

- BSIM-CMG 107, 108, 110 and 111, BSIM6.1.1, BSIM-SOI 4.5.0, 4.6.1 and 4.7.0, PSP 102 and 103, JUNCAP200, L-UTSOI;
- EKV 2.6, HICUM/L0 and L2, MEXTRAM 504, VBIC 1.3, FBH HBT;
- DIODE_CMC, MVS and MVSG.

BSIM-CMG 111 alone is 83,009 lines of C++ from 6,965 lines of Verilog-A.
"The Xyce build system does not rebuild this C++ code automatically from the Verilog-A sources, as this would be a very time-consuming process" (`utils/ADMS/examples/README:13-14`).
The generated code writes explicit derivatives (`d_IDS_dV_di_si` and the like, `N_DEV_ADMSbsimcmg_111.h:7024-7062`) and analytic parameter sensitivities (`InstanceSensitivity`, `ModelSensitivity`, `N_DEV_ADMSbsimcmg_111.C:67128-67180`).

**Where it falls short:**

- **Patched models:** published models must be patched for ADMS ("Wherever the Xyce team has had to modify published Verilog-A models ...", `README:25`).
    BSIM-CMG 111's `README_Xyce` lists ten changes: a ternary ADMS could not parse, `@initial_instance` blocks for node collapse, parameters both model and instance, the thermal node's order, multiplicity, temperature caching, and smoothing macros that divide by zero "when these macros are differentiated".
- **User plugins:** "many others will require manual patching after processing with ADMS" (`utils/buildxyceplugin.in:26-34`).
    FBH HBT 2.3 "does not work in Xyce" (`README:3-4`).
- **Coverage:** no OSDI (a search of `src/`, `utils/` and `cmake/` finds none), no HiSIM, no pole-zero and no `.tf` (its reference guide lists neither command).
- **Reference models can be wrong:** Xyce 7.11 fixed a BSIM-SOI 4.6.1 capacitance bug "after empirically observing that several commercial simulators had clearly fixed their own implementations", though "the official version of this model from the BSIM group didn't include a fix"; and its 4.7.0 "may be less robust than the previous version" (`doc/Release_Notes/Release_Notes_7.11/Release_Notes_7.11.tex:83-84`).

**Its strengths:** analytic sensitivities, direct and adjoint; harmonic balance; MPDE; polynomial chaos (`doc/Reference_Guide/*_Command.tex`).

### 3.3 Verilog-A tooling, the state of the art

- **OpenVAF and OpenVAF-reloaded:** a Verilog-A compiler written in Rust, "licensed under the GPL-3.0 license", that "outputs a dynamic library whose functionality can be accessed via the OSDI API", through LLVM 18 to 23 (its README).
    - Development: Pascal Kuthe's original had "no support since end of 2023"; Árpád Bűrmen's fork carries OSDI 0.4, which adds parameter-given flags, Jacobians loaded with an offset for harmonic balance, and access to noise sources.
    - Unsupported: Laplace filters, `$abstime()`, `@cross()`, indirect branch contributions and `paramset`, per the VA-Models README.
    - Platforms: binaries for Linux and Windows; "macOS users must build OpenVAF-Reloaded from sources" (VACASK's README); ngspice's page says "no macOS for now".
    - Compile time: HICUM/L2 v3.0 in 0.72 s, against 22 s for Xyce's ADMS (OpenVAF's documentation, seen in search results; the site did not resolve on 15 November 2026).
    - Speed: "similar (sometimes even faster) compared to built-in C-coded models" (ngspice's OSDI page).
        On HICUM/L2 the numbers conflict: OpenVAF's page gives 8.47 s compiled against 7.98 s built in; Bűrmen's MIDEM 2025 slides give 9.16 against 14.64 (both via search results).
    - Derivatives: exact with respect to node voltages; none with respect to parameters cross the OSDI API (`osdi.h:151-208`).
- **VACASK** (AGPL-3.0): a simulator whose device library is Verilog-A compiled to OSDI 0.4, but for its sources, linear controlled sources, mutual inductances and behavioral sources, which are built in.
  On the C6288 multiplier with PSP 103.4 (10,112 transistors), it took 57.98 s, ngspice 71.81 s and Xyce 151.57 s, on a Threadripper 7970 with KLU and no OpenMP.
  That is its developers' own benchmark.
  With an OSDI 0.4 model, "a delay is silently zero" (its README).
- **VADistiller** (AGPL-3.0, NLnet): converts SPICE3's C models to Verilog-A: diode levels 1 and 3, Gummel-Poon, JFET 1 and 2, MESFET 1, MOSFET 1, 2, 3, 6 and 9, VDMOS, BSIM3 3.3.0 and BSIM4 4.8.0 to 4.8.3.
  Each converted model is validated against ngspice.
  Some do not conserve charge, by their authors' modelling; "Due to varying support for Verilog-A $limit your mileage may vary"; and `$simparam` calls are expensive in OpenVAF's code (its README).
- **Gnucap's modelgen-verilog:** compiles Verilog-A to C++ plugins (Davis and Salfelder, FOSDEM 2024; NLnet grant 2023-10 to 2025-05).
    Its state since is unknown.
- **bosdi** (gdsfactory, MIT): evaluates OSDI models from JAX.
  Through OSDI, "jax.grad() work[s] through node voltages — but not through model parameters".
  Its alpha path compiles Verilog-A to JAX for parameter gradients, through "a custom fork of OpenVAF that exposes the compiler's intermediate representation".
  It is independent evidence that parameter gradients need the compiler, not the binary.
- **The survey:** Bűrmen et al., "Free software support for compact modelling with Verilog-A", Informacije MIDEM 54(4), 2024: free alternatives "are much more scarce and limited in the features they offer".

### 3.4 The commercial simulators, from what their makers and users publish

- **Spectre:**
    - Built-in models: BSIM3, BSIM4, BSIM-BULK, PSP and HiSIM; VBIC, HICUM, MEXTRAM, HBT and Gummel-Poon; BSIM-CMG, BSIM-IMG and UTSOI.
        It "automatically compiles Verilog-A into optimized C-code" (Cadence's Spectre X datasheet, via search; the page returned 403).
    - Speed: on HICUM/L2, Spectre's Verilog-A took 52.61 s against 25.33 s built in (OpenVAF's documentation, via search).
    - Mismatch: `dcmatch` estimates it from one simulation, 10.42 mV against Monte Carlo's 10.56 mV in Cadence's example (Cadence blog, "The Art of Analog Design Part 5").
- **HSPICE:** level 54 is BSIM4 and level 72 BSIM-CMG (a third-party table).
    Verilog-A loads through `.hdl` and its `hsp-vacomp` compiler (HSPICE user guide X-2005.09, mirrored).
- **LTspice:** MOSFET levels 1, 2, 3, 4 (BSIM), 5 (BSIM2), 6, 8 ("BSIM3v3.3.0 ...
    July 29, 2005"), 9 (BSIMSOI3.2), 12 (EKV 2.6) and 14 (BSIM4.6.1), and VDMOS.
    Its help names no Verilog-A (its M-device page, mirrored at ltwiki.org).

### 3.5 The compact models foundries ship, and how each reference runs them

Verilog-A defines the CMC's standards: "CMC policy states that the Verilog-A description of the model defines the standard", and "Any compliant implementation of the standard model shall give the same outputs" (Si2's standard-models page).
Lines of Verilog-A are counted in Xyce's `utils/ADMS/examples/`, includes counted.

| Model, current release (Si2) | Verilog-A lines (version counted) | ngspice 47 | Xyce | LTspice | Spectre, HSPICE |
| --- | --- | --- | --- | --- | --- |
| BSIM4, 4.8.3 (legacy) | none from Berkeley: C; VA-Models and VADistiller have 4.8.x | C, four copies by version | C++ 4.6.1, 4.7.0, 4.8.2 | 4.6.1 | built in, HSPICE level 54 |
| BSIM-CMG, 112.1.0 | 6,965 (111.2.1) | OSDI | ADMS 107 to 111 | none | built in, HSPICE level 72 |
| BSIM-BULK (BSIM6), 107.2.1 | 4,576 (6.1.1) | OSDI | ADMS 6.1.1 | none | built in |
| BSIM-SOI, 4.7.1 / 101.0.0 | 8,240 (4.6.1) | C (B3SOI, B4SOI) | C++ 3.2, ADMS 4.5.0 to 4.7.0 | 3.2 | built in |
| PSP, 104.1.0 | 7,057 (103, with JUNCAP200) | OSDI | ADMS 102, 103 | none | built in |
| HiSIM2 3.2.0, HiSIM_HV 2.5.1 | not counted | C | none | none | built in |
| HICUM/L2 3.2.0, L0 2.1.0 | 2,127 (L2 2.4.0) | C++ L2 2.4.0 on dual numbers; OSDI | ADMS L2, L0 | none | built in |
| MEXTRAM, 505.6.0 | 2,396 (504.12.1) | OSDI | ADMS 504 | none | built in |
| EKV 2.6, EKV3 | 862 (2.6) | OSDI | ADMS 2.6; EKV3 apart, in `Xyce_NonFree` (`utils/generate_ADMS.sh`) | 2.6 | Spectre lists EKV |
| VBIC 1.3 | 1,992 | C 1.2 | ADMS 1.3 | unknown | built in |
| Also: L-UTSOI 102.9.0, DIODE_CMC 3.0.0, MVSG 5.0.0, R2/R3_CMC, MOSVAR 1.4.0, ASM-HEMT 101.6.0 | 3,321 (L-UTSOI), 2,364 (DIODE_CMC 2.0.0), 1,074 (MVSG 1.1.0) | OSDI | ADMS for several | none | some built in |

The 11 counted come to about 42,000 lines.
VA-Models, the collection ngspice points to as "providing nearly all of the publicly available Verilog-A compact models", says that "Most model codes are licensed under the Educational Community License, Version 2.0", and that the models were modified "mostly for convergence improvements", their equations unchanged (its README).
So the code ngspice users run is not byte for byte the CMC's release.
PSP's licence lets users name an implementation "PSP" only when it produces "the same outputs as Standard code for the same inputs passing all CMC QA tests", and forbids charging for the model itself (`utils/ADMS/examples/psp103/IP_NOTICE_DISCLAIMER_LICENSE:38-58`).
BSIM's and HiSIM2's terms are of the same kind: no charge for the code itself, an acknowledgment, and the notice kept (`B4TERMS_OF_USE`; `hisim2/hsm2ld.c:34-46`).

### 3.6 The open process kits, and what each needs

| Kit | Transistors | Other devices | Library features | What jolty lacks |
| --- | --- | --- | --- | --- |
| SKY130 | BSIM4 level 54, version 4.5, in subcircuits; `nfet_01v8` at `tt` has 180 bins, `capmod` 2, `mobmod` 0, `rbodymod` 1, `permod` 1, `diomod` 1, `igcmod` and `igbmod` 0, `rgatemod` 0, `geomod` 0, `rdsmod` 0, `tempmod` 0, both NQS modes off, `fnoimod` and `tnoimod` 1, stress parameters (`saref`, `sbref`, `kvth0`, `ku0`) given, no well-proximity parameters | Diodes are ngspice's level 3 (`js`, `jsw`, `cjsw`, `tlevc`, all read by ngspice's `dio/dio.c`); bipolar transistors are Gummel-Poon level 1 with HSPICE's temperature terms (`tlev`, `tlevc`, `tis1`, `tbf1`, about 60 in all, read by ngspice's `bjt/bjt.c`) | `.lib` corners, `.include`, `.option scale`, mismatch sections, HSPICE compatibility (`foundry.md`) | BSIM4 4.5, the diode's level 3, the BJT's temperature terms, the kit reader |
| GF180MCU | BSIM4 level 54, version 4.5, binned (`nmos_3p3` at least 12 bins); `capmod` 2, `mobmod` 0, `rbodymod` 0, `permod` 1, `diomod` 1, `igcmod`, `igbmod`, `rgatemod`, `geomod`, `rdsmod` 0 | Gummel-Poon-style bipolar transistors; diodes, resistors and MIM capacitors in subcircuits (their levels unverified) | `agauss` in its statistical section | BSIM4 4.5, the kit reader |
| IHP SG13G2 | PSP 103.6 as Verilog-A `N` elements (`psp103va`), `swgeo` 1 with no bins, gate current, impact ionization, GIDL, JUNCAP2 (`swjuncap` 3), quantum correction; RF variants with `swnqs = rfmode * 5` | HBTs are VBIC level 9 (cards "VBIC (Rev.1.15)") with self-heating, `rth`, `cth`, a fifth thermal node, and `tf` reading `temper`; resistors are R3_CMC `N` elements with a thermal node; MOSVAR varactors and MOM capacitors in Verilog-A | Expressions with `temper`; mismatch libraries; Apache-2.0 | PSP 103.6, R3_CMC, MOSVAR and the MOM capacitors; VBIC lands soon; the kit reader |

**Where these come from:**

- **SKY130:** `cells/nfet_01v8/sky130_fd_pr__nfet_01v8__tt.pm3.spice`, `cells/diode_pw2nd_05v5/*.model.spice` and `cells/npn_05v5/*.model.spice` in google/skywater-pdk-libs-sky130_fd_pr.
    This is not the open_pdks build jolty pins, and the two may differ.
- **GF180MCU:** `models/ngspice/sm141064.ngspice` in google/globalfoundries-pdk-libs-gf180mcu_fd_pr.
- **IHP SG13G2:** `ihp-sg13g2/libs.tech/ngspice/models/sg13g2_moslv_mod.lib`, `sg13g2_moslv_parm.lib`, `sg13g2_hbt_mod.lib` and `resistors_mod.lib`, and `libs.tech/verilog-a/` (`psp103`, `r3_cmc`, `mosvar`, `cap_cmomf`, `cap_cmomi`, `vbic_1p3`), in IHP-GmbH/IHP-Open-PDK.
- **ngspice:** it reads all three, IHP through OpenVAF's compiled models (its `examples/IHPOpenSourcePDK`, `examples/SkywaterOpenSourcePDK`).
- **Xyce:** it reads IHP's Verilog-A through ADMS plugins (`adms-compile-va.sh`).

### 3.7 How models are validated

- **Against the reference implementation:** the CMC's QA tests.
  A `qaSpec` file names each test's biases, sweeps, outputs and temperatures; VBIC 1.3's runs DC tests forward and reverse, a temperature test at -50, 50 and 150 °C, AC tests whose outputs are `G(c,b)` and `C(c,b)`, and self-heating tests, with `checkPolarity yes`.
  Reference results ship beside the spec (`utils/ADMS/examples/vbic_r1.3_prerelease/qaSpec`, `reference/*.standard` in Xyce).
  Si2 also distributes VAMPyRE, a checker of Verilog-A code by the CMC's technical advisor Geoffrey Coram (Si2's page).
- **Against the manual:** jolty's ports cite their manual and are checked against ngspice, which is itself a reading of the manual: BSIM3 against b3ld.c (`bsim3.rs:9-16`).
- **Laws of the model:** symmetry, and the CMC's rules for writing models.
  The Gummel symmetry test checks the drain current's odd symmetry and its derivatives' continuity at zero drain bias (McAndrew; its modified DC and AC forms in "DC and AC symmetry tests for MOSFET models", TechConnect).
  The CMC's coding rules are in McAndrew, Coram, Gullapalli, Jones, Nagel, Roy, Roychowdhury, Scholten, Smit, Wang and Yoshitomi, "Best Practices for Compact Modeling in Verilog-A", IEEE J-EDS 3(5):383-396, 2015.
  jolty tests laws per model, ad hoc: BSIM3's n and p mirror and its drain-source swap (`bsim3.rs:820-860`); exact Jacobians and parameter gradients against central differences (`testing.rs:228-450`); no allocation (`crates/jolty-models/tests/kernels.rs`).
- **Against measured silicon:**
    - SkyWater publishes "Raw data collected about the SKY130 process technology", measured by CoolCAD Electronics on SkyWater's test tile, in MDM format under Apache-2.0 (google/skywater-pdk-sky130-raw-data).
    - IHP's kit lists "MOS/HBT Measurements in MDM format" (its README).
    - A 2026 study fitted SKY130's cards to measurements at 77 K with "an average error on the order of 20% (relative RMS)", and notes "there is no industry standard for quantifying how well a transistor model fits to its data" (arXiv 2604.21625).
    - Neither ngspice's nor Xyce's tree holds any MDM file (`find`: none).
- **Where references disagree:** the reference can be wrong (Xyce's BSIM-SOI fix, §3.2); VA-Models changes models for convergence (§3.5); and ngspice keeps four BSIM4s because versions differ.

### 3.8 Analyses that read the models, briefly

| Analysis | jolty | ngspice 47 | Xyce |
| --- | --- | --- | --- |
| Operating point, DC sweep | Yes; every DC solution by deflation (`simulation.md:75-86`) | Yes, two sources, temperature and resistance | Yes |
| AC, transient, noise | Yes | Yes, and transient noise from sources | Yes |
| Periodic steady state | Shooting, exact monodromy | Experimental | Harmonic balance, MPDE |
| Sensitivity | DC, AC and transient, exact device derivatives | DC and AC, perturbation by 1e-6 | DC, AC and transient, direct and adjoint |
| Pole-zero, `.tf` | Yes | Yes ("not free of bugs") | No |
| Stability (loop gain) | Yes, Middlebrook's | No | No |
| Distortion | No | Ten SPICE3 models | Harmonic balance |
| S-parameters | No | `.sp` | `.LIN` |
| Monte Carlo, corners, worst case | Yes, the same bits on any shard | Through its control language | Sampling, polynomial chaos |
| Mismatch from one solve (`dcmatch`) | No | No | No |

ngspice's column comes from its `ANALYSES` file, Xyce's from `doc/Reference_Guide/*_Command.tex`.
Spectre has `dcmatch` (§3.4); its other analyses were not checked here.

## 4. Gaps and opportunities

Ordered by impact on the goal: every device and kit a designer needs, run natively, more exact than the references.
Items marked **Lead** are those where jolty would go beyond every reference rather than catch up.
Sizes are estimates unless counted.

| # | Item | Impact | Evidence | Rough size | Done when |
| --- | --- | --- | --- | --- | --- |
| 1 | BSIM4 4.0 to 4.5 as ngspice's `BSIM4v5`. The modes SKY130's and GF180MCU's cards select come first: `capmod` 2, `mobmod` 0, `rbodymod` 1 and 0, `permod` 1, `diomod` 1, the stress effect, binning, `fnoimod` and `tnoimod` 1. Any other mode is refused by name | Two of the three open kits, and the predictive cards (version 4.0) | §3.6; `inpdomod.c:297-300`; the dropped `brief-bsim4.md` holds a ready plan | Twice BSIM3: about 8,000 lines of kernel and 3,000 elsewhere (estimate). BSIM4v5 has 766 parameters to BSIM3's 437, and its `b4v5ld.c` has 5,212 lines to `b3ld.c`'s 3,191 (counted) | SKY130's and GF180MCU's core transistors agree with ngspice across bins and corners within the corpus's tolerances, each departure stated in the guide |
| 2 | SKY130's other primitives: the diode's level 3 and the BJT's HSPICE temperature terms (`tlev`, `tlevc`, the per-parameter `t…1` and `t…2`, `nkf`) | Completes SKY130; the same terms recur in other kits' cards | §3.6; jolty's diode and BJT have neither (`devices.md:261-378`) | 1,000 to 2,000 lines (estimate) | Each SKY130 primitive agrees with ngspice |
| 3 | A kit reader. It reads `.lib` sections, `.include` and `.option scale`, evaluates `temper` at each device's temperature, and draws `agauss`, `gauss` and `aunif` per instance from the sample's seed, so mismatch keeps jolty's determinism. Parsed libraries are cached | The precondition for every native kit | `library.py:111-113`; `spice.md:56`; IHP's `tf` reads `temper`; GF180MCU's statistics use `agauss` | 1,000 to 3,000 lines of Python (estimate) | The three kits' libraries read at every corner, a sample's mismatch the same bits on any shard |
| 4 | SKY130 and GF180MCU natively: their devices as kernels, the ngspice path kept as an option | Every analysis on kit circuits: sensitivities, periodic steady state, pole-zero, stability, and Monte Carlo in the engine across threads | `foundry.md`: five analyses, all in ngspice | Small once 1 to 3 land | A SKY130 circuit runs every jolty analysis natively and agrees with the ngspice path |
| 5 | **Lead.** A Verilog-A compiler of jolty's own, in safe Rust. It reads the CMC's subset (best practices; no Laplace, `@cross` or `$abstime`), splits what depends on no probe into `prepare` and the rest into `evaluate`, and decides collapsed nodes in `layout`. It writes kernels generic over `Real` with exact sparse derivatives, `$limit` as `Bias`, noise as `NoiseSource`, and output variables as `variables`. The generated kernels are checked in, regenerated and never edited, as Xyce's are | All CMC models and IHP's from one tool. Exact parameter derivatives, which OSDI cannot carry. `unsafe` stays forbidden. Aimed to take the CMC's sources unpatched, which ADMS cannot; on macOS too, where OpenVAF has no binaries | The kernel has OSDI's form (§2.5); OSDI has no parameter derivatives (`osdi.h:151-208`, bosdi); ADMS needs patching and leaves ngspice (§3.2, §3.1); a hand port of the 11 counted models would be about 55,000 lines (estimate, from jolty's VBIC at 1.3 times VBIC 1.3's Verilog-A) | 15,000 to 25,000 lines of compiler and tests (estimate). The Rust compile time of a generated BSIM-CMG is unknown: measure it on the first large model | R3_CMC, DIODE_CMC and MOSVAR, then PSP 103.6, compile into kernels that pass their CMC QA tests, clippy and the kernel rules |
| 6 | IHP SG13G2 natively: PSP 103.6, R3_CMC, MOSVAR and the MOM capacitors compiled; VBIC level 9 with its thermal node; the RF cards' NQS | The third open kit, and the only BiCMOS one | §3.6 | Small after 3, 5 and VBIC's merge | IHP's devices agree with ngspice and OpenVAF at every corner |
| 7 | The CMC's catalogue compiled: BSIM-CMG, BSIM-BULK, BSIM-IMG, BSIM-SOI, HiSIM2, HiSIM_HV, HICUM/L0 and L2, MEXTRAM, EKV 2.6, L-UTSOI, JUNCAP2, DIODE_CMC, R2_CMC, ASM-HEMT, MVSG, and BSIM4 4.8 from Verilog-A | Every compact model foundries ship, FinFETs and SiGe BiCMOS among them | §3.5; about 42,000 lines of Verilog-A (counted) | Per model: its QA, its reader entry, its guide section | Each passes all its QA tests, so it may carry its standard's name |
| 8 | **Lead.** A CMC QA harness: a reader of `qaSpec`, the tests run in jolty, and their outputs compared with the reference results shipped beside the spec | Turns "checked against its source" into a mechanical test for every standard model | §3.7 | About 1,000 lines (estimate) | Every compiled model's QA runs in CI and passes |
| 9 | **Lead.** Measured silicon as a reference: an MDM reader, and each kit's cards simulated in jolty against SkyWater's and IHP's measurements, with the error per device and bias published | No open simulator does this; it says how exact a kit's models are, not only how alike two simulators are | §3.7: SKY130's raw data and IHP's MDM files; neither reference's tree holds any | 1,000 to 2,000 lines (estimate) | `benchmarks.md` holds each kit's model-to-silicon error |
| 10 | **Lead.** One law suite for every kernel: charges summing to zero over the terminals, the n and p mirror, the symmetry a model claims (the Gummel symmetry test to high order), continuity of current and derivative across regions, finite values at any finite bias, and the existing derivative checks | Every model, hand-written or compiled, held to the same physics. Xyce's divide-by-zero in differentiated smoothing macros is the kind of fault it finds | Today's tests are per model and ad hoc (§3.7) | About 1,000 lines (estimate) | Every kernel passes, or each failure is stated as its source's, with evidence |
| 11 | **Lead.** Second and higher derivatives in every kernel | Exact AC sensitivities (now a central difference); distortion for every model, where ngspice covers ten SPICE3 models; an exact Gummel symmetry test; later, harmonic balance's Jacobians | `simulation.md:104`; ngspice's `DEVdisto`; `Hyper` already carries one mixed second order (`real.rs:396-415`) | Medium: a design across `Real`, `Dual` and the stamps | AC sensitivities with no finite difference, and a distortion analysis for every model |
| 12 | **Lead.** Exact derivatives at the cost of hand-written ones: sparse, compile-time forward differentiation in compiled kernels, and the same for hand-written ones where it pays | Today exactness costs about twice ngspice's time per MOSFET; compact models have 12 to 15 nodes | §2.2, §2.5; OpenVAF's compiled HICUM outran ngspice's dual numbers in one of two benchmarks (§3.3) | Medium to large | A MOSFET's evaluation costs no more than ngspice's per iteration, every derivative still exact |
| 13 | A model costs its equations: its Python constructor, its SPICE reading and writing, and its reference derived from its parameter schema, with units from the schema | Cheaper models, no conflicts in shared dispatchers, less to review | §2.5: 375 to 378 lines of `devices.py`, 218 to 287 of the reader and 63 to 108 of the exporter per large model; conflicts in four shared files | Medium | Adding a kernel touches its own file, its tests and its guide section alone |
| 14 | The ports' notices: BSIM's terms and copyright, ngspice's BSD notice, and each generated model's licence, in its file and a `NOTICE` | Compliance with the terms jolty's ports already fall under | §2.5: B3TERMS and B4TERMS; jolty carries neither | Small | Each ported or generated model carries its source's notice and terms |
| 15 | SPICE3's remaining compact models: the MESFET (Statz), MOSFET levels 2, 6 and 9, JFET level 2, BSIM3's NQS modes and versions 3.0 to 3.2, BSIM-SOI (through 5) | Old and textbook cards; parity with ngspice | `inpdomod.c`; the dropped MESFET and level 2 briefs | MESFET about 1,000 lines (estimate); level 2 about as level 3, whose commit, with the diode's breakdown, took 905 lines across 17 files (`766ccd9`) | Each reads ngspice's cards and agrees with ngspice |
| 16 | Transmission lines and multiports: lossless (`T`), lossy (LTRA), coupled (CPL, TXL), Touchstone blocks, IBIS buffers | Boards, cables, signal integrity, packages | ngspice's `tra`, `ltra`, `cpl`, `txl`; Xyce's `TRA`, `LTRA`, `YLin`, `IBIS` | The lossless line needs a bounded history of accepted steps in the engine (the dropped `brief-tra.md`); the rest more | Closed forms (a matched line's delay, reflections, a quarter-wave inverter) and ngspice agree |
| 17 | A user's own Verilog-A at run time without `unsafe`: an interpreter in the manner of `jolty-expressions`' tape, which runs forward for values and backward for gradients (`tape.rs:1-2`), or WebAssembly through wasmtime's safe API (speculative, unmeasured) | An agent writes a device and simulates it at once, with exact derivatives | `Library::add` takes models at run time (`lib.rs:164-170`) | Medium for an interpreter, large for WebAssembly | A user's `.va` runs in every analysis, its derivatives checked against central differences |
| 18 | The vendors' model libraries measured: the share of a corpus of vendors' PSpice and LTspice models the reader takes | "Every device a designer needs", measured at board level | Four macromodels compared today (`spice.md:60-73`) | Small to measure | `benchmarks.md` holds the share, and the features that block the rest |
| 19 | XSPICE's code models and `A` elements, read and simulated | Netlists written for ngspice's mixed-signal blocks | ngspice's `xspice`; jolty has logic and bridges of its own (`connect.rs`) | Large | The code models ngspice's examples use run and agree |
| 20 | **Lead.** Mismatch from sensitivities: `dcmatch` and `acmatch` from the kit's mismatch parameters and the adjoint, exact to first order, beside Monte Carlo | One solve where Monte Carlo takes hundreds; neither open reference has one | Spectre's `dcmatch` (§3.4); jolty's adjoint (`simulation.md:103`); needs 3 | Medium | Its offsets match Monte Carlo's within their statistics on a kit circuit |
| 21 | OSDI through `unsafe`, in one crate | Breadth soon, but no parameter derivatives, no per-equation scale for jolty's convergence test, native code jolty cannot check against its kernel rules, and no macOS binaries of OpenVAF | `Cargo.toml:45` forbids `unsafe`; `AGENTS.md:77`; §2.5; §3.3 | About 2,000 to 3,000 lines (estimate), and a change to `AGENTS.md` | Not recommended; the owner decides |
| 22 | CIDER's numerical devices, the control language, device bypass | Low for this goal: TCAD-level devices; Python and MCP already are jolty's control language; bypass is an approximation | ngspice's `numd`, `nbjt`, `numos`; `vision.md`'s question on approximations | | Out of scope, or opt-in knobs |

**The order.**

1. Items 1 to 4: SKY130 first, GF180MCU after it, on the same BSIM4.
1. VBIC's merge, and item 5's compiler, in parallel.
1. Item 6 (IHP), then item 7.
1. Items 8 and 10 grow with each model from the first.

BSIM4 is hand-ported, not compiled.
The open kits' cards are version 4.5, which ngspice runs through `BSIM4v5`; the Verilog-A BSIM4s available are 4.8 (VA-Models, VADistiller).
How far 4.8's equations move a 4.5 card is unknown.
It can be measured cheaply: run a SKY130 card in ngspice with its version edited to 4.8.

## 5. The long view

In two to five years, jolty could be where no reference is.

- **Every open kit and every standard compact model natively, by name.**
  SKY130, GF180MCU, IHP SG13G2 and those that follow run without ngspice.
  Each CMC model passes all its QA tests, so jolty may call it by its standard's name, which PSP's licence allows only then.
  A new CMC release is a regeneration and a QA run, not a port.
- **Every model differentiable to any order, in every variable.**
  Bias, temperature and every parameter, exactly.
  That gives design by gradient (`agents.md` item 4), extraction of a model's card from measured data by gradient (an open, differentiable IC-CAP), mismatch from one solve, and distortion and harmonic balance for every model.
  ngspice's OSDI cannot carry parameter derivatives.
  Xyce has analytic ones for its ADMS models, whose sources it patches.
- **Models verified, not trusted.**
  Each kernel passes one law suite and its standard's QA in CI.
  The compiler reports what it finds in a model as facts, with its file and line: a charge that is not conserved, a discontinuity in a derivative, a division that can reach zero, a parameter range never checked.
  Kernels hand their discontinuities to the transient through `Kernel::choices`.
  This is `AGENTS.md`'s "feedback is facts" applied to models.
- **Models judged against silicon.**
  `benchmarks.md` publishes each kit's model-to-measurement error, which no simulator publishes today.
- **Agents that write devices.**
  An agent writes a Verilog-A model; jolty checks it against the laws and simulates it at once, with exact derivatives; the model and its tests join the circuit's library.
- **Exact and fast.**
  Compiled kernels with sparse derivatives, vectorized across the instances of a group, which already share a kernel and a shape (`group.rs:1-15`).
  Compact models would then cost no more than ngspice's hand-written C while every derivative stays exact.

## 6. Principles to propose

For the owner to weigh as rules for `AGENTS.md`:

1. **A model's reference is what defines it.**
   For a CMC standard, its Verilog-A release and its QA tests; for SPICE3's models, ngspice; otherwise its paper, manual or datasheet; and a kit's published measurements where there are any.
   A model takes a standard's name only when it passes all that standard's QA tests.
   This sharpens the Models convention (`AGENTS.md:85-86`).
1. **Verilog-A is compiled into kernels, never loaded as foreign code.**
   `unsafe` stays forbidden, and every model keeps exact derivatives with respect to its parameters.
   A generated kernel is regenerated, never edited, and each change its source needed is listed beside it, as Xyce's `README_Xyce` files list theirs.
1. **Every model obeys the laws.**
   Each kernel passes one law suite: exact Jacobians and parameter gradients, charges that sum to zero, the polarity mirror, the symmetry its source claims, continuity, and finite values at any finite bias.
   A failure is fixed or stated as its source's, with evidence.
1. **A port carries its source's notice.**
   Each ported or generated model reproduces its source's copyright notice and terms, in its file and in the documentation.
1. **A kit is pinned and fetched, never copied.**
   As `sky130.py` pins its release and checks its digest, so every kit; and tests use cards of their own, as `models.md` already decided for vendors' cards.

## 7. Sources

**jolty** (the repository at `68c3f75`):

- **Rules and architecture:** `AGENTS.md:11`, `AGENTS.md:71`, `AGENTS.md:77`, `AGENTS.md:85-86`; `Cargo.toml:45`.
- **Models crate:** `crates/jolty-models/src/lib.rs:1-21`, `lib.rs:127-170`, `lib.rs:419-455`; `kernel.rs:37-226`, `kernel.rs:395-492`; `kernel/parts.rs:259-510`; `group.rs:1-15`, `group.rs:210-233`, `group.rs:1649-1670`; `dual.rs:1-6`; `real.rs:18-23`, `real.rs:396-415`; `binding.rs:27-42`; `heat.rs:1-10`; `foundry.rs`.
- **Kernels:** `mosfet.rs:538`, `bsim3.rs:9-16`, `bsim3.rs:69`, `bsim3.rs:820-860`; `bsim3/parameters.rs` (2,121 lines); `testing.rs:228-450`; `crates/jolty-models/tests/kernels.rs`.
- **Expressions:** `crates/jolty-expressions/src/tape.rs:1-2`.
- **Python:** `src/jolty/spice/sky130.py:35`, `sky130.py:111-126`, `sky130.py:187-231`; `src/jolty/spice/library.py:84-117`; `src/jolty/spice/models.py:519-566`.
- **Guide:** `docs/guide/devices.md:261-554`; `docs/guide/simulation.md:27-111`; `docs/guide/spice.md:31-73`; `docs/guide/foundry.md`; `docs/guide/parts.md:11-17`.
- **Commits:** `c60cb78`, `6fe8824` (BSIM3), `9271cf9` (VDMOS), `766ccd9` (level 3).
- **Branch `vbic`:** a worktree, `vbic.rs:103`, its diff stat against `main`.

**Notes** (these plans):

- the session log, not kept and `models.md` (the set-aside list, now in `models.md` §3);
- an archived note, not kept brief, not kept, `brief-mesfet.md`, `brief-mos2.md` and `brief-tra.md`;
- an archived note, not kept and `vbic.md`;
- a brief, not kept;
- `handoff-2026-11-14.md:19-20`, `handoff-2026-11-14.md:48-49`;
- `quality.md`.

**ngspice 47** (scratchpad `ngspice/src/ngspice-47/`):

- **Parser:** `src/spicelib/parser/inpdomod.c:49-385`.
- **OSDI:** `src/osdi/osdi.h:21-22`, `osdi.h:80-208`; `osdiregistry.c:31`, `osdiregistry.c:434-436`; `osdiload.c:190-204`; `osdiinit.c:188-205`.
- **Analyses:** `src/spicelib/analysis/cktsens.c:24`; `ANALYSES`; `README_OSDI.md`.
- **Devices:** `src/spicelib/devices/*/` (line counts; `*init.c` for `DEVdisto` and `DEVsenLoad`); `hicum2/hicumL2.cpp:8-21`.
- **Licences:** `bsim4/B4TERMS_OF_USE`, `bsim3/B3TERMS_OF_USE`, `bsim3/b3ld.c:1-12`, `hisim2/hsm2ld.c:34-46`, `COPYING`.
- **Examples:** `examples/IHPOpenSourcePDK`, `examples/SkywaterOpenSourcePDK`, `examples/osdi`.

**Xyce** (scratchpad `xyce/Xyce/`, `e5b7b9b`; GPL, read for facts only):

- **Models:** `src/DeviceModelPKG/ADMS/` (25 models, 856,148 lines; `N_DEV_ADMSbsimcmg_111.h:6760-7062`, `N_DEV_ADMSbsimcmg_111.C:1-30`, `N_DEV_ADMSbsimcmg_111.C:67128-67180`); `src/DeviceModelPKG/OpenModels/`, `IBISModels/`.
- **ADMS tooling:** `utils/ADMS/TODO`; `utils/ADMS/examples/README:13-25`; `examples/BSIM-CMG_111.2.1_06062022/code/README_Xyce`, `bsimcmg.va`; `examples/psp103/IP_NOTICE_DISCLAIMER_LICENSE:38-58`; `examples/vbic_r1.3_prerelease/qaSpec`, `reference/`; `utils/generate_ADMS.sh`; `utils/buildxyceplugin.in:26-34`.
- **Documents:** `doc/Release_Notes/Release_Notes_7.11/Release_Notes_7.11.tex:83-84`; `doc/Reference_Guide/M_Device.tex`, `Q_Device.tex`, `*_Command.tex`.

**Kits:**

- SkyWater's SKY130 primitives, `cells/nfet_01v8/sky130_fd_pr__nfet_01v8__tt.pm3.spice`, `cells/diode_pw2nd_05v5/sky130_fd_pr__diode_pw2nd_05v5.model.spice`, `cells/npn_05v5/sky130_fd_pr__npn_05v5_W1p00L1p00.model.spice`: <https://github.com/google/skywater-pdk-libs-sky130_fd_pr>
- SKY130's raw measurements: <https://github.com/google/skywater-pdk-sky130-raw-data>
- GF180MCU's primitives, `models/ngspice/sm141064.ngspice`: <https://github.com/google/globalfoundries-pdk-libs-gf180mcu_fd_pr>
- IHP's open kit, `ihp-sg13g2/libs.tech/ngspice/models/` and `libs.tech/verilog-a/`: <https://github.com/IHP-GmbH/IHP-Open-PDK>

**Verilog-A tooling:**

- OpenVAF-reloaded's README: <https://raw.githubusercontent.com/arpadbuermen/OpenVAF/master/README.md>
- OpenVAF's performance page, unreachable on 15 November 2026 and quoted from search results: <https://openvaf.semimod.de/docs/details/performance/>
- VACASK's README: <https://codeberg.org/arpadbuermen/VACASK>
- VADistiller's README: <https://codeberg.org/arpadbuermen/VADistiller> ; its NLnet page: <https://nlnet.nl/project/Verilog-A-distiller/>
- Bűrmen's MIDEM 2025 and ESSERC 2025 slides, fetched as images only, so read through search results: <https://fides.fe.uni-lj.si/~arpadb/MIDEM2025/midem2025.pdf> , <https://fides.fe.uni-lj.si/~arpadb/ESSERC25/ESSERC2025.pdf>
- Á.
    Bűrmen, T. Tuma, I. Fajfar, J. Puhan, Ž.
    Rojec, M. Kunaver, S. Tomažič, "Free software support for compact modelling with Verilog-A", Informacije MIDEM 54(4), 2024, doi:10.33180/InfMIDEM2024.404
- ngspice's OSDI page: <https://ngspice.sourceforge.io/osdi.html>
- VA-Models: <https://github.com/dwarning/VA-Models>
- bosdi: <https://pypi.org/project/bosdi/>
- Gnucap: A. Davis and F.
    Salfelder, "Verilog-AMS in Gnucap", FOSDEM 2024; NLnet's Gnucap MixedSignals: <https://nlnet.nl/project/Gnucap-MixedSignals/>

**Standards and validation:**

- Si2's standard models: <https://si2.org/standard-models/>
- C. C. McAndrew, G. J. Coram, K. K. Gullapalli, J. R. Jones, L. W. Nagel, A. S. Roy, J. Roychowdhury, A. J. Scholten, G. D. J. Smit, X. Wang, S. Yoshitomi, "Best Practices for Compact Modeling in Verilog-A", IEEE Journal of the Electron Devices Society 3(5):383-396, 2015, doi:10.1109/JEDS.2015.2455342
- "DC and AC symmetry tests for MOSFET models", TechConnect Briefs: <https://briefs.techconnect.org/papers/dc-and-ac-symmetry-tests-for-mosfet-models/>
- Cryogenic SKY130 models from measurements: <https://arxiv.org/abs/2604.21625>

**Commercial:**

- Cadence's Spectre X datasheet, which returned 403 and is quoted from search results: <https://www.cadence.com/en_US/home/resources/datasheets/spectre-x-simulator-ds.html>
- Cadence's blog, "The Art of Analog Design Part 5: Mismatch Analysis II": <https://community.cadence.com/cadence_blogs_8/b/cic/posts/the-art-of-analog-design-part-5-mismatch-analysis-ii>
- LTspice's MOSFET help, mirrored: <https://ltwiki.org/LTspiceHelp/LTspiceHelp/M_MOSFET.htm>
- HSPICE's `.hdl`: the HSPICE user guide X-2005.09, as mirrored at <https://wenku.csdn.net/doc/6412b466be7fbd1778d3f79a> ; HSPICE's level 54 and 72: <https://monata-docs.readthedocs.io/en/stable/reference/bsim-models.html>
