# Changelog

Every version of jolty and the changes it introduces.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## Unreleased

The first release, 0.1.0.

### Added

- `NOTICE` file recording copyright notices and terms for external device-model sources: BSIM3v3.3.0 and BSIM4.5.0 (UC Berkeley BSIM Research Group), plus ngspice (diode, JFET, BJT, level 1 and 3 MOSFETs, and VDMOS).
  Each model's documentation carries its source notice, and both the wheels and the `jolty-models` crate bundle `NOTICE`.
- Transfer function analysis `TransferFunction(source, output, reference=None)`, matching SPICE `.tf`, computing DC gain plus input and output resistances (reporting infinity when no current can flow).
  Transfer function results are supported in text and JSON reports, results files (format version 7), and the MCP server guide.
- Free-threaded Python 3.15 wheel alongside the standard wheel, declaring free-threaded extension support so Python script threads (such as batch `each` callbacks and custom measurements) execute concurrently without the GIL.
  CI tests the free-threaded build, verifying that importing `jolty._core` leaves the GIL disabled and that eight concurrent Python threads match single-threaded bits identically.
- Circuits expressed as Python build scripts of reusable components, compiled into a canonical validated format that reports every problem by instance path and script line.
- Native Rust simulation engine built on modified nodal analysis with charge-oriented device models, sparse LU factorization, Newton iteration with SPICE convergence aids, and pseudo-transient continuation past convergence folds.
  The engine solves the operating point once for all dependent analyses and checks and enforces a unified `max_memory` budget across transients and dense pole-zero or periodic matrices.
  Supported analyses include operating point, DC sweep, AC, transient, periodic steady state, sensitivity, stability, noise, pole-zero, and deflated multi-solution DC searches.
  Transient integration steps onto every behavioral-source discontinuity before restarting, seeds Newton steps via linear extrapolation from the previous two points, and estimates first post-corner step error from shared curvature with the second step.
  Local truncation error estimates are filtered through the step equations so fast settling within a step is not miscounted, governed by `truncation_tolerance` matching SPICE `TRTOL`.
- Pinned Rust 1.100.0 toolchain in `rust-toolchain.toml`, matched by every workspace crate's `rust-version` field and enforced by `tests/test_rust_version.py`.
- KLU-inspired sparse LU solver for circuit matrices using block triangular form (leaving supply rows unfactored), minimum-degree ordering per block, and threshold pivoting favoring the ordered diagonal.
  Newton iterations refactor with prior pivots (searching only from degraded pivots) and reuse factors when the Jacobian is unchanged, so linear transients factor only on step-size or method changes.
  Proven-pivot unknowns (MOSFET Meyer displacements and series-resistance currents below 1 Ω) are eliminated via internal node equations before factoring and recovered afterward.
  Elimination shrinks a 500-inverter chain from 3,504 to 504 factored unknowns and a two-switch buck converter from 23 to 13, matching ngspice's matrix size, while singular-circuit errors name all undetermined unknowns.
- Device models for passive and linear elements: resistors with flicker noise, voltage-dependent capacitors and inductors, coupled inductors, independent sources with waveforms, controlled sources, behavioral sources, and voltage- or current-controlled switches.
  Semiconductor and mixed-signal models cover diodes, JFETs, Gummel-Poon BJTs with excess phase, level 1 and 3 MOSFETs, BSIM3, photocells, and event-driven logic alongside analog circuitry.
- Complete ngspice 47 diode model `diode(...)` (SPICE levels 1 and 3), adding sidewall junctions (`JSW`, `NS`, `CJSW`, `PHP`, `MJSW`, `FCS`, `IKP`, `RSW`), recombination (`ISR`, `NR`), high injection (`IKF`, `IKR`), and tunneling (`JTUN`, `JTUNSW`, `NTUN`, `XTITUN`, `KEG`).
  The diode model supports temperature modes `TLEV` 0 to 2 and `TLEVC` 0 and 1 (`EG`, `GAP1`, `GAP2`, `CTA`, `CTP`, `TPB`, `TPHP`, `TRS`, `TRS2`, `TTT1`, `TTT2`, `TM1`, `TM2`, `TCV`) with exact derivatives and per-junction noise.
  The SPICE reader sizes level 3 diodes using element width, length, and library `scale`, and scales sidewall junctions by perimeter.
  The SPICE exporter emits sidewall perimeter, `NS`, and `TLEVC` 1 coefficients referenced to 27 °C so ngspice computes identical values.
  Every diode in SKY130, GF180MCU, and SG13G2 now loads natively.
  At typical corners, native diode counts rose from 143 to 170 of 263 in SKY130, from 7 to 19 of 72 in GF180MCU, and from 13 to 35 of 53 in SG13G2.
  Where ngspice diverges from documented diode equations, jolty follows the equations as detailed in the departures guide.
  Documented diode fixes avoid doubled level 3 multiplicity, unset sidewall breakdown currents, mismatched sidewall charge thresholds, unintegrated `RSW` capacitance, missing `NS` temperature scaling, inexact recombination conductance, and hardcoded 27 °C `TLEVC` 1 offsets.
- Power MOSFET model `vdmos(...)` implementing ngspice 47 VDMOS (n- and p-channel) with subthreshold softening, mobility degradation, triode multiplier, quasi-saturation, and charge-conserving gate-drain capacitance (`CGDMIN` to `CGDMAX`).
  VDMOS further includes body diode breakdown, terminal resistances, temperature scaling, noise, self-heating, and exact derivatives.
  The SPICE reader parses vendor `VDMOS` cards (including `pchan` and LTspice annotations) and three- to five-node `M` elements, and the exporter writes them back for ngspice.
- Short-channel MOSFET model `bsim3(...)` implementing ngspice 47 BSIM3v3.3 (SPICE levels 8 and 49) for 0.8 µm to 0.13 µm CMOS processes.
  BSIM3 includes short-channel, narrow-width, and DIBL threshold terms, poly depletion, `mobmod` 1 to 3, velocity saturation, output resistance, `rdsw`, and substrate current.
  BSIM3 also provides `capmod` 0 to 3 charges with `xpart`, junctions, geometry and temperature scaling, parameter checks, `noimod` 1 to 6, self-heating, and exact derivatives.
  BSIM3 card parameters use manual names in SI units; the SPICE reader handles version 3.3 level 8 and 49 cards with `LMIN`-to-`WMAX` binning, and the exporter emits compatible ngspice cards.
- Advanced MOSFET model `bsim4(...)` implementing ngspice 47 BSIM4.5 (SPICE levels 14 and 54) for SKY130 and GF180MCU modes (`mobmod` 0, `capmod` 2, `diomod` 1, `tempmod` 0, `rbodymod`/`rdsmod` 0-1, `rgatemod` 0-3), rejecting unsupported modes by name.
  Supported modes include external bias-dependent drain/source resistances (`rdsmod` 1) and gate electrode plus intrinsic-input resistances (`rgatemod` 1 to 3), including intermediate overlap-charge gate nodes.
  BSIM4 physics covers short-channel, narrow-width, pocket, and DIBL thresholds, poly depletion, mobility degradation, velocity saturation and overshoot, source velocity limits, output resistance, substrate currents, and gate-induced leakages.
  BSIM4 also implements trap-assisted junction tunneling, `capmod` 2 charges with thickness and bias-dependent overlaps, length-of-diffusion stress, binned geometry scaling, and exact derivatives.
  The SPICE reader loads version 4.5 level 14 and 54 cards with ngspice-compatible binning, and the exporter writes them back for ngspice.
  Native SKY130 `nfet_01v8` and `pfet_01v8` agree with ngspice across 81 binned sizes at `tt`, `ff`, and `ss` corners within twice relative tolerance for output curves, transfer curves, and inverter switching.
  All 74 SKY130 RF transistors (`rgatemod` 3) load natively, with four verified against ngspice at DC operating points and admittances up to 10 GHz within twice relative tolerance; the departures guide documents ngspice's non-derivative body capacitances.
  BSIM4 noise implements `b4v5noi.c`: charge-based (`tnoimod` 0) and holistic (`tnoimod` 1) channel thermal noise, SPICE2 (`fnoimod` 0) and oxide-trap (`fnoimod` 1) flicker noise, and thermal noise for drain, source, body, and gate resistances.
  Amplifier stages built from SKY130 and GF180MCU transistors match ngspice output noise from 1 Hz to 1 GHz within twice relative tolerance.
- Gummel-Poon BJT temperature scaling matching ngspice 47 `bjttemp.c`, adding `TREF`, `TLEV` 0 and 1, `TLEVC` 0 and 1, and high-injection exponent `NKF` with exact derivatives including temperature.
  Supported HSPICE linear and quadratic coefficients cover `TBF1` through `TMJS2`, `CTE`, `CTC`, `CTS`, `TVJE`, `TVJC`, and `TVJS`.
  The SPICE reader skips undefined HSPICE parameters as ngspice does and rejects `TLEV` 3 by name.
  Bipolar transistors from SKY130, GF180MCU, and SG13G2 load natively and agree with ngspice at -40, 27, and 125 °C within twice relative tolerance.
- Charge-conserving behavioral capacitor `behavioral_capacitor(expression)` matching Verilog-A `ddt(q)`, computing exact capacitance derivatives from the charge expression.
  The SPICE reader maps `Q=` capacitors directly and analytically integrates voltage-dependent `C=` capacitors composed of polynomials plus scaled `exp`, `sinh`, `cosh`, `tanh`, `sin`, or `cos` of affine voltage terms, rejecting unsupported expressions by name and exporting back as `Q=`.
  SKY130's two varactors and GF180MCU's eleven MIM and MOS capacitors load natively and agree with ngspice from -2 to 2 V; the departures guide documents ngspice's 1 fs delay on voltage-dependent `C=` currents.
- Integrated-circuit bipolar transistor model `vbic(...)` implementing ngspice 47 VBIC 1.2 (SPICE levels 4 and 9) with exact derivatives, temperature scaling, noise, and self-heating.
  Beyond Gummel-Poon effects, VBIC includes the parasitic substrate transistor, weak avalanche, quasi-saturation, base-emitter breakdown, reach-through, smoothed junction capacitances, and a dedicated excess-phase network.
  The SPICE reader parses VBIC model cards and three- to five-node `Q` elements, and the exporter writes them back for ngspice.
- Memristor models supporting voltage-controlled VTEAM (with thresholds) and current-controlled HP linear ion drift, each offering linear or exponential dynamics and boundary window functions.
  Memristor states remain stationary at DC operating points, evolve during transients, and export to ngspice as behavioral current sources.
- Dimensionless device states (such as memristor internal state and photocell response) modeled as a dedicated unknown nature, accessible via `state(name)` and plotted on a separate panel.
- Vendor SPICE macromodel import as reusable Python components, plus SPICE netlist export for ngspice emitting behavioral expressions directly from AST trees so ngspice evaluates identical math.
- SPICE reader support for `PULSE`, `SIN`, and `PWL` independent-source waveforms matching exporter output, replacing the earlier DC-only reader behavior.
- Exponential waveform `jolty.waveforms.Exponential` (SPICE `EXP`) modeling delayed rise and fall transitions with individual time constants and automatic transient breakpoints at both delays.
  The reader and exporter support `EXP` for ngspice and Xyce, mapping a 0 s delay to `1e-40` s so ngspice reads it literally instead of substituting the time step.
  Omitted or zero delays and time constants that depend on analysis parameters are rejected, and RC responses agree with ngspice.
- Transient `initial_conditions` (matching SPICE `.ic` with an operating point, or SPICE `UIC` when `operating_point=False`) and solver `nodesets` (matching SPICE `.nodeset`).
  Nets rigidly constrained by voltage sources or inductors defer to those elements, with conflicting initial conditions verified against them.
  The SPICE reader exposes `.ic` and `.nodeset` as `Library.initial_conditions` and `Library.nodesets`, `export.node_voltages` writes them, and foundry ngspice decks run them including `uic`.
  RC and RLC tests initialized at 0.5 V in both modes plus bistable latches driven to either state agree with ngspice, and CircuitSim90 decks `arom`, `gm17`, `rich3`, and `jge` now parse cleanly.
- Support for negative lateral diffusion in level 3 MOSFETs (lengthening the effective channel beyond drawn length as in ngspice 47), requiring only positive effective length and matching ngspice DC curves.
- SPICE user-defined `.func name(parameters) {expression}` support outside subcircuits, expanding nested calls across parameters, element values, instance parameters, and behavioral expressions.
  Direct or mutual recursion in `.func` definitions is rejected while naming every function in the cycle, bringing the coverage matrix `.func` row into agreement.
- SPICE raw-file output (`jolty.files.save_raw`) in binary or ASCII formats, emitting deterministic date-free plots for operating point, DC sweep, small-signal, and transient analyses in ngspice naming conventions.
  `jolty.spice.ngspice.read` parses both ASCII and binary raw files by plot point count; jolty RC raw output matches ngspice vectors within `1e-6`, bringing the coverage matrix raw-file row into agreement.
- Lossless transmission line `transmission_line(characteristic_impedance, delay)` matching ngspice 47 `T` elements, acting as a short at DC and pure phase delay in AC.
  Transient evaluation interpolates bounded accepted-step history at the integration order between corners and steps onto every delayed corner, matching piecewise-linear closed forms within floating-point rounding.
  The SPICE reader and exporter support `Z0`/`TD` and `F`/`NL` forms while rejecting lossy lines (`O`, `U`, `Y`, `P`) by name.
  Pole-zero, periodic steady state, and frequency/time sensitivities reject circuits containing transmission lines, and stability checks skip growing-mode analysis on them.
- Xyce 7.10.0 netlist export (`export.netlist(..., dialect=export.XYCE)`) and external execution via `jolty.spice.xyce.run`, supporting passive elements, coupled inductors, independent and controlled sources, diode, JFET, Gummel-Poon BJT, and level 1 and 3 MOSFETs at the matched thermal-voltage temperature.
  Unsupported or incompatible devices and parameters (including BSIM3 3.3.0, VDMOS, VBIC, excess phase, and behavioral expressions) are rejected by name.
  `tests/test_xyce.py` verifies operating points, DC sweeps, AC, noise, and transients whenever Xyce is installed, explicitly listing excluded Xyce departures.
- Unified external-simulator runners `jolty.spice.ngspice.execute` and `jolty.spice.xyce.execute`, each running decks in a temporary directory with timeout enforcement and returning `Executed` (stdout/stderr, exit `code`, execution `seconds`, and `VALUES` output).
  `ngspice.execute` accepts a `compatibility` mode, `xyce.execute` accepts a `launch` command (such as MPI parallel launches), and timeouts raise `NgspiceTimeoutError` or `XyceTimeoutError` with captured output.
  `jolty.spice.xyce.Result.analysis_seconds` reports Xyce's internal analysis time, and `jolty.spice.ngspice.runs` accepts `timeout` while recording `seconds` on each `Result`.
- Open PDK support at every corner in `jolty.spice.kits` for SKY130 (`sky130.KIT`), GF180MCU (`gf180mcu.KIT`), and IHP SG13G2 (`sg13g2.KIT`), pinned by release and fetched by digest.
  The SPICE reader handles `.lib` sections, `.include`, `.option scale`, `.if` blocks, subcircuit instance parameters, spaced or quoted values, arbitrary MOSFET binning, finger width division under HSPICE compatibility or `wnflag` (matching `INPgetModBin`), and subcircuit/element multipliers `m`.
  Additional PDK features include SPICE3 `r`/`c` model-card geometry sizing, voltage-dependent resistors translated to behavioral currents, sign-preserving odd-exponent `POW`, `.option scale` on BSIM4 diffusion distances, and library-temperature `temper`.
  Statistical draws (`agauss`, `gauss`, `aunif`, `unif`, `limit`) evaluate deterministically via `Library.drawn` for bit-identical sharded Monte Carlo.
  `kits.inventory` catalogs model coverage across kits, resolving missing default sizes via first-bin midpoints, first kit placements, or ngspice's 10 µm resistor default.
  Natively supported kit devices (including SG13G2 heated VBIC and SKY130 MIM capacitors) agree with ngspice at identical corners.
- First-order mismatch analysis `Mismatch(outputs, of=None, random_variables=())` matching Spectre `dcmatch`/`acmatch`, computing output standard deviations (magnitude and phase for AC) and per-variable variance shares sorted largest first.
  By default, mismatch uses circuit part and lot tolerances, while `Library.random_variables` exposes PDK statistical functions as `ir.RandomVariable` terms differentiated via the sensitivity adjoint.
  Mismatch outputs appear in results files (format version 19), reports, and the MCP guide, agreeing with Monte Carlo within sampling error on differential pairs, current mirrors, SG13G2 VBIC stages, and SKY130 MIM capacitors (`tests/test_mismatch.py`).
- Foundry simulation flow for SKY130 transistors: the engine checks circuits structurally and runs operating points, DC sweeps, AC, transients, noise, corners, and mismatch Monte Carlo in ngspice against the digest-verified PDK.
- Device and component-port ratings on voltages, currents, power, and temperature (upper, lower, or two-sided bounds), checked on every operating point and transient.
- Thermal networks built from thermal resistances, heat capacities, and heat sources whose nets represent temperature rise above ambient, solved simultaneously with the electrical circuit.
- Self-heating support across resistors, diodes, JFETs, BJTs, VBIC transistors, level 1/3 MOSFETs, BSIM3, and VDMOS, coupling electrical power dissipation to thermal nets with exact derivatives in every analysis.
- Per-device temperature overrides (absolute or offset from simulation temperature), imported from SPICE `TEMP`/`DTEMP` and PSpice `T_ABS`/`T_REL_GLOBAL` and exported for ngspice.
- Real component models with manufacturer part numbers, ratings, and optional self-heating: 2N3904, 2N3906, 1N4148, and four datasheet-fitted power MOSFETs.
- Diagnostic rule suite covering circuit structure, E-series preferred values, operating points (gmin-only nets, multiple DC solutions, growing modes), stability margins, and ratings.
  Visual feedback includes connectivity-driven schematic rendering, analysis plots, and an interactive 3D viewer playing simulations on the circuit layout.
- Specification checks evaluated across temperatures, corners, Monte Carlo tolerance samples (per-parameter linear or logarithmic spreads, path-addressed, with shared lots), and worst-case extremes.
  Yield reports include exact two-sided 95% Wilson confidence intervals computed via `jolty.specs.wilson(passing, samples)`.
- Logic characterization and fidelity workflows converting analog subcircuits into event-driven logic cells and spot-checking them against transistor-level simulations.
- Exact sensitivities of every unknown to every device parameter across operating points, DC sweeps, AC frequencies, and transients, isolating per-instance derivatives so operating-point parameter cost scales with the device rather than the full circuit.
- Discrete transient adjoint differentiation (`Simulator::objective`) computing objective gradients with respect to every device parameter in a single backward pass (one transposed solve per step regardless of parameter count).
  Supported functionals cover trapezoidal time integrals, interpolated point values, and accepted-step sums.
  Adjoint derivatives match direct sensitivities across all 53 supported kernel-corpus transients (including initial-condition runs) within `4.8e-13` relative magnitude and match central differences on a rectifier under both trapezoidal and Gear2 integration.
  Transient adjoints store four scalars per step alongside waveforms, reject logic or transmission-line circuits, and are exposed in Rust pending Python plan integration.
- Unified forward and reverse differentiation (`Simulator::jvp` and `Simulator::vjp`) across operating points, DC sweeps, AC, transients, and periodic steady states (including oscillator period outputs).
  Direct JVP pushes tangents on named parameters forward to every output, while adjoint VJP pullback (`Simulator::pull_back`) maps output cotangents back using one transposed solve per cotangent regardless of parameter count, reusing one factorization per point.
  Dot-product tests `⟨v, J t⟩ = ⟨Jᵀ v, t⟩` hold to `7.4e-16` across operating points, sweeps, AC, trapezoidal and Gear2 transients, driven PSS, and autonomous oscillators (`crates/jolty-engine/tests/sensitivity.rs`, elm).
  Sensitivities, mismatch analyses, oscillator period derivatives, and objectives all share JVP/VJP, with sensitivity automatically picking VJP or JVP to minimize linear solves.
  Time-domain and PSS pullbacks replay accepted steps without re-simulating periodic trajectories, rejecting logic circuits whose event drivers cannot be replayed statelessly.
  Dual-number parameter lifting runs field by field through the `jolty-derive` `Lift` macro (analogous to JAX `tree_map_with_path`), replacing JSON round-trips that compiled serde readers for every dual width (including 750 BSIM4 fields).
  Heated devices load plain parameters and lift them during preparation.
  At the single-width `Lift` commit (`ecc932d`), `jolty-models` codegen at `opt-level = 1` dropped from 392 s to 68 s on 24 cores of an AMD EPYC 7B12, and LLVM IR shrank from 1.85M to 1.49M lines (`cargo llvm-lines`).
  With all four chunk widths (`61fa6e8`), `opt-level = 1` codegen dropped from 689 s (JSON) to 159 s (`Lift`).
  Parameter derivatives evaluate via chunked dual twins (`Gradient<K>`, `Hyper<K>`, `Seeds`, `Twin`) carrying up to eight directions per pass, matching single-direction twins bit for bit in every kernel gradient test.
  Under `crates/jolty-models/benches/instructions.rs` (Valgrind 3.25.1 Callgrind on glibc 2.27, diode `"junction"` biased at 0.69 V), eight-direction twins reduce instruction counts sharply.
  Instruction counts drop 2.4× for BSIM4 (92,128 vs 8 × 27,077), 2.9× for resistors (1,170 vs 8 × 430), and 3.5× for diodes (2,482 vs 8 × 1,090).
  Frequency-domain JVP and VJP are now exact without finite differences (`BIAS_STEP` removed): kernels evaluate unknowns as generic `R` under seeding, and `jolty-expressions` propagates directional second derivatives without allocation (`Expression::along`).
  AC VJP accumulates frequency weights solely over nonlinear matrix slots and executes one transposed DC solve per active cotangent regardless of parameter count, skipping the DC solve on purely linear circuits.
  Benchmarked on a 500-rung resistive ladder driven by `loaded_amplifier` over 5 frequencies on pinned core `cpu=0` of a shared AMD EPYC 7B13 (Milan, 33 samples, load 122 to 132 over 256 logical CPUs).
  AC VJP pullback scales from 0.316 ms (1 parameter) to 2.165 ms (1,000 parameters), compared to 0.423 ms to 107.795 ms on `61fa6e8` (49.8× faster at 1,000 parameters).
  Total `vjp` plus `pull_back` runtime drops from 109.933 ms to 3.830 ms (28.7× faster).
  Exact AC VJP agrees with `61fa6e8`'s 1 µV central differences within `9.74e-8` at 1,000 parameters (`9.57e-10` on `loaded_amplifier`) and satisfies the AC dot-product test to `7.15e-17` (worst overall across analyses: `7.38e-16`).
  Fixed latent derivative edge cases: `HeatedKernel` re-prepares when an unknown twin seeds a self-heated device at `ΔT = 0` and avoids `inf - inf = NaN` when clamping infinite thermal trial steps.
  Additional derivative fixes preserve `vdseff_cv` drain capacitance derivatives at `vds = 0` (`bsim4/charges.rs`), preserve `kvsat` derivatives strictly inside `(-1, 1)` (`bsim4/prepare.rs`), and guard `Real::powf`, `Gradient::chain`, `Hyper::chain`, and `jolty-expressions` `tape::along` against `0 * inf = NaN`.
  Decoupled `Real` and `Kernel` from `serde` and `schemars` (`6dd5cee`), confining serialization and schema bounds to `Parameters<f64>` inside `Differentiable<K>` and `Thermal<K>`.
  Decoupling removed `Written` and dual serde/schema impls (`cargo llvm-lines -p jolty-models --lib`: 2,329,400 to 2,329,129 lines, 6 to 0 dual serde/schema items) and replaced `#[serde(bound = "")]` with `#[schemars(bound = "R: Serialize + JsonSchema")]` with zero diff in `docs/reference/`.
  Added library-wide `Lift` vs `serde` path parity tests (`every_models_lift_visits_the_exact_paths_serde_writes`, plus Verilog-A parity in `twins.rs` and `constructs.rs`) across all 30 analog models and 8 compiled Verilog-A kernels with exact `f64::to_bits()` parity and schema-driven integer classification.
- Faster Rust test execution without changing release builds or coverage: sharded the three 182-circuit corpus tests into 16 parallel tests each using an auto-indexing macro and `LazyLock` parsing (`crates/jolty-engine/src/kernels.rs`, `crates/jolty-engine/src/objective.rs`, and `crates/jolty-veriloga/tests/corpus.rs`).
  Split the two dissected-factorization tests in `determinism.rs` (sizes 157 and 500) and enabled `[profile.dev.package.jolty-numerics] opt-level = 2` while retaining debug assertions and overflow checks.
  On 24 cores of a shared AMD EPYC 9B14 (Genoa, `jobs=64`), full `cargo nextest run --workspace` wall time dropped from 302.41 s (600 tests) to 107.15 s (647 tests, 2.82× faster; sharding alone gave 228.46 s).
  Under full load, `condenses_the_series_currents_of_circuits` dropped from 225.20 s to 20.28 s (11.1×; 163.74 s to 10.10 s isolated, 16.2×).
  Under full load, `condenses_the_displacements_of_circuits` dropped from 44.17 s to 6.97 s (6.3×; 40.66 s to 2.48 s isolated, 16.4×).
  Cold build time moved from 121.38 s to 100.91 s (+5.3% CPU), incremental `diode.rs` builds moved from 12.75 s to 12.07 s, and `target/` stayed at 10.96 GiB (vs 11.11 GiB before).
  Optimizing all external dependencies at `opt-level = 3` was tested and rejected (161.45 s cold build, +105.6% CPU, 217.75 s tests, 12.27 GiB).
- Differentiable Python analysis interface `jolty.function(circuit, analysis)` evaluating `f(params)` on a single elaborated circuit using nested parameter dictionaries in SI base units.
  `jolty.jvp(f, params, tangents)` and `jolty.vjp(f, params)` mirror `jax.jvp` and `jax.vjp`, returning unknown arrays plus oscillator period under `jolty.simulate.PERIOD`, so scalar loss gradients over crossbar weights require one transposed solve.
- Batched function evaluation `Function.batch(names, values)` and reverse-mode `Function.batch_vjp(names, values)` returning a `BatchPullback`, running across engine threads with bit-identical per-row outputs regardless of thread count or batch size.
  Underlying `Batch::map` executes arbitrary per-run work on top of `Batch::run` preparation.
- JAX bridge `jolty.jax.function(circuit, analysis, options=None, *, saved_pullbacks=16)` under optional extra `jolty[jax]` (`jax>=0.12`), implemented as a `jax.custom_vjp` wrapping `jax.pure_callback(..., vmap_method="broadcast_all")` over `Function.batch` and `BatchPullback`.
  The bridge supports `jax.jit`, `jax.grad`, `jax.value_and_grad`, `jax.vjp`, `jax.jacrev`, `jax.remat`, and `jax.vmap` (including mixed `in_axes`, nested `vmap`, and empty batches) across `OperatingPoint`, `DcSweep`, `Ac` (with complex cotangents), `Transient`, driven `PeriodicSteadyState` (evaluated at required `breakpoints`), and oscillators (`PERIOD`).
  Mapped calls produce bit-identical rows to unmapped calls, cached pullbacks retain up to `saved_pullbacks` forward passes before recomputing identically, structural unknown drops raise `SimulationError`, `jax_enable_x64` is respected, and free-threaded Python `3.15t` keeps the GIL disabled (`tests/test_jax.py`).
  Forward-mode `jax.jvp` and higher-order autodiff are rejected, and `Function` raises `TypeError` for unsupported analyses.
  Added a narrow lock around `Elaborated._structures` updates in `jolty.simulate` for thread-safe structural caching while keeping the unchanged hot path lock-free.
- Deterministic pole-zero analysis using a native LAPACK-inspired QZ solver on linearized circuit pencils, supporting least-damping and dominant-pole specs in agreement with ngspice across platforms.
- Dedicated periodic steady state analysis `PeriodicSteadyState(period=...)` matching ngspice `.pss`, Spectre `pss`, and Xyce `.HB`.
  Driven circuits solve via shooting with the monodromy matrix carried across the period, preserving initial states along neutral directions such as unexcited memristors.
  Autonomous oscillators solve simultaneously for periodic state and unknown period under an orthogonal phase condition, initialized from a transient kicked along the fastest-growing natural mode.
  `PeriodicSteadyStateResult` records `period` and `periods` across reports, results files (format version 18), and the MCP guide.
  Periodic sensitivities `Sensitivity(outputs, of=PeriodicSteadyState(...))` differentiate both periodic trajectories and oscillator periods (`SensitivityResult.period_derivative`) via the converged bordered shooting system.
- Adjoint noise analysis computing thermal, shot, and flicker noise contributions at an output, input-referred noise, and noise-density or RMS specs, matching ngspice 43+ (including per-copy MOSFET gate flicker noise).
- Parts catalog pricing circuit bills of materials by value and tolerance or exact part number across standard packages while excluding obsolete parts.
- KiCad netlist export mapping each bill-of-materials entry to a catalog footprint and pin-matched nets while reporting every issue preventing board placement.
- Selective recording via `record` patterns with `.gitignore`-style `!` exclusions (such as `Plan(..., record=["X1/**", "!X1/bias/**"])` or `simulate --record M1`), capturing device terminal currents/charges, small-signal parameters, internal variables, and summed component-port flows.
  Operating points and transients automatically record rated instances for rating checks while keeping unrecorded transient memory compact, and reading an unrecorded quantity states how to enable recording.
  Recording queries inspect solved device states without altering numerical values, allowing subsequent analyses with different `record` filters to reuse an already-solved operating point.
- Unified CLI, MCP server, and Python package defining every command once, using Python 3.15 lazy module loading so non-plotting commands start without importing matplotlib.
- Analysis logging setting `analysis_log` (`JOLTY_ANALYSIS_LOG`) appending one record per started analysis (kind, run count, timestamp) across CLI, MCP, and Python API runs, readable via `jolty.simulate.Logged`.
- Reusable CLI framework `jolty.cli.CommandLine` and `jolty.cli.run` enforcing kebab-case options, valueless flags, non-JSON lists, no implicit environment reads, and explicit valid-value reporting on bad inputs.
- Automated model-parameter reference generator extracting defaults and descriptions directly from engine parameter types.
- Agent benchmark suite with blind design tasks drawn from Koza patents, AnalogCoder problems, and cost-down exercises, graded by hidden scorers.
- Head-to-head agent evaluation harness in `benchmarks/agents/compare.py`, running an agent on each task with jolty versus ngspice alone using separate prompts, files, tools, and counted simulator wrappers.
  ngspice netlists parse through jolty's SPICE reader for identical scoring, reporting per-arm success rates with Wilson intervals, turn counts, tool calls, simulation counts, and per-task differences; a stub agent verifies all front ends without paid runs.
  The jolty arm counts every executed analysis via `analysis_log` across CLI, MCP, and Python calls, while the ngspice arm counts simulator invocations.
  Each session runs in an isolated configuration without user settings, `CLAUDE.md`, memory, or external MCP servers.
  Both agent runs and script scoring execute inside platform sandboxes (Seatbelt on macOS, bubblewrap on Linux), and Claude Code launches only after free sandbox startup and credential checks pass.
  Comparison reports break down session termination causes (finished, time/turn/budget limits, or error) alongside scorer failure categories and the agent's final report.
- Speed benchmark suite (`benchmarks/speed/`) spanning a bipolar amplifier through a 90,000-node mesh, timed with Criterion in Rust and end-to-end in Python via `scripts/bench`.
  `scripts/bench` records wall and CPU seconds per commit with 95% rank intervals, flags changes only when intervals separate, supports interleaved two-revision worktree comparisons (`--compare A,B`) until 5% resolution, and supports `--nightly DIR` regression runs.
  `scripts/bench --view` compares jolty, ngspice (Sparse 1.3 and KLU), and Xyce across three views (matched options, defaults, and equal error on precision fixtures), reporting runtime, voltage deviation against jolty, and failure causes.
  Per-kernel evaluation timings (`crates/jolty-models/benches/evaluation.rs`, Criterion) and Valgrind Callgrind instruction counts (`instructions.rs`, gungraun) share identical parameter shapes.
  Instruction counts compare builds deterministically despite 9 to 10% wall-time variance from code layout on unchanged VDMOS kernels (pinned core, load 3 to 5).
- Unified benchmark CLI `scripts/bench` covering speed, precision, robustness, and coverage suites, built on shared simulator runners (`benchmarks/simulators.py` for jolty, ngspice Sparse/KLU, and serial/MPI Xyce) and shared metadata recording (`benchmarks/measuring.py`: load, commit, dirty flag, machine info, and `<suite>/results/<commit>.json`).
- Robustness corpus (`scripts/bench --robustness`) combining hard internal circuits (Schmitt triggers, latches, SRAM cell, zero-start ring oscillator, impasse point) with pinned CircuitSim90 MOSFET/BJT decks (`benchmarks/robustness/external.json`, fetched on demand).
  Every fixture runs under timeout-guarded processes in jolty and ngspice, classifying outcomes uniformly against all jolty DC solutions (agreed, disagreed, solved, failed, timed out, unsupported) and reporting separate DC and transient failure rates with Wilson intervals.
  `jolty.spice.ngspice.read` parses plots from arbitrary ASCII raw files produced by ngspice.
- Precision benchmark suite (`scripts/bench --precision`, `benchmarks/precision/`) with 16 exact-reference fixtures (closed forms and high-precision `mpmath` solutions up to 1,000-node networks spanning six time-constant decades).
  Each fixture runs in jolty, ngspice 47 (Sparse 1.3 and KLU), and Xyce 7.10 at defaults and across tolerance sweeps, plotting worst-point correct digits against runtime, steps, rejections, and Newton iterations.
- Coverage matrix (`scripts/bench --coverage`, `benchmarks/coverage/`) initially testing 65 minimal known-answer fixtures across jolty, ngspice 47 (Sparse 1.3 and KLU), and Xyce 7.10 while citing documentation for LTspice, Spectre, HSPICE, and AFS.
  Matrix cells classify every feature as agreed, disagreed (with values), failed, or unsupported (with message) and highlight cross-simulator gaps.
- Transient solver statistics counting Newton iterations and sparse solver factorizations, refactorizations, fallbacks, and reuses, plus an optional `profile` breakdown across preparation, evaluation, assembly, factoring, solving, step control, and recording.
- Rust-side specification measurements: every `jolty.specs` measure resolves signals and units against a result so batches evaluate measures directly in Rust without Python overhead.
  Rust measure arithmetic replicates NumPy bit for bit except for distortion (computed via exact DFT sums rather than FFT), and domain-violating arithmetic (such as RMS over a reversing sweep) reports unmeasured rather than failing.
- Reusable elaborated circuits (`Elaborated`) shared across threads, storing unknowns, sparsity pattern, solver ordering, and a path-indexed parameter table.
  Each `Simulator` instance prepares device groups for specific `Values` and temperature; `Simulator::reset` re-prepares only modified devices and clears prior state so reset runs match fresh runs bit for bit.
  Parameter changes that alter matrix structure (such as a resistor crossing 1 Ω) trigger cached per-structure elaboration keyed by `Elaborated::structure_of`.
- Zero-allocation, panic-free kernel architecture modeling every device (from resistors to logic connect elements) as a pure function of prepared parameters, memory, and local unknowns, with self-heating modeled by evaluating the kernel at its thermal-net temperature.
  The engine evaluates same-kernel, same-shape instances into contiguous slabs serially or across worker threads once `min_instances_per_task` fills every thread, while linear groups evaluate once per run and slab blocks accumulate in a fixed deterministic order.
  Runs re-prepare only instances whose parameters or temperature changed, including across sweeps and warm chains.
  Recorded device terminals and small-signal models read directly from the converged slab without re-evaluating devices.
  Level 1 and 3 MOSFETs differentiate junction and overlap terms with respect to local voltages and sum gate/bulk/drain/source stamps once.
  Direct stamp summation cuts 500-inverter transistor evaluation from 226 ns to 200 ns (131 ns vs 157 ns without oxide).
  Level 1/3 stamp reordering shifts results solely within rounding (at most `3.6e-14` V on the inverter chain and `2.6` µV at a switching edge on the buck converter).
  BSIM3 applies the same direct summation (464 ns vs 491 ns at defaults), keeping bits identical except at nodes behind drain or source resistances.
  VDMOS applies the same summation (94 ns vs 104 ns at defaults), keeping bits identical except at nodes behind drain or source resistances.
  VBIC applies the same summation across its seven nodes (653 ns vs 883 ns at defaults), keeping bits identical except at nodes behind resistances or overlap capacitances.
  VBIC evaluates each physical law on narrow dual numbers matching only the one to three variables that law reads rather than carrying all nine derivatives everywhere, adding weak avalanche as a separate intrinsic collector-to-base current.
  Narrow-dual VBIC cuts instruction counts from 10,879 to 7,184 at defaults, from 12,830 to 8,838 with all features enabled, and from 13,376 to 9,290 with excess phase (`crates/jolty-models/benches/instructions.rs`).
  Single-core AMD EPYC 7B12 runtime drops from 1.42-1.51 µs to 819-829 ns at load 2 to 5.
  Across 1,200 VBIC evaluations covering both polarities, base-charge forms, smoothing modes, and temperatures, residuals and conductances move by at most `0.9 ε` of row maximum while charges and capacitances are unchanged.
  Gummel-Poon BJT and JFET kernels adopt the same narrow-dual formulation, eliminating all hand-written derivatives and chaining the current-dependent base resistance via a single-variable dual.
  Across 6,000 BJT and 504 JFET evaluations, residuals, conductances, and capacitances move by at most `3.5 ε` of row maximum while charges are unchanged.
  Instruction counts rise 2 to 8% (BJT: 2,203 to 2,370; JFET: 2,409 to 2,477), within build-layout timing noise.
  Memristors, switches, photocell, VDMOS quasi-saturation, and voltage-dependent resistors/capacitors adopt narrow duals with a custom photocell derivative rule remaining finite at arbitrary darkness, keeping residuals, charges, and capacitances unchanged and conductances within `2.9 ε` (at most 25 extra instructions).
  Dual-number division computes the reciprocal once and multiplies derivatives instead of dividing by the square, cutting BSIM3 evaluation from 456 ns to 426 ns at defaults with rounding-level derivative differences.
  Shared junction current and depletion charge laws (used by diode, VDMOS body diode, MOSFET bulk junctions, JFET gate, and BJT junctions) evaluate on dual numbers with precomputed voltage-independent factors for exact derivatives by construction.
  High-fanout nodes with over 1,024 contributions (such as supply rails) gather from contiguous per-instance buffers written in deterministic addition order.
  Contiguous gathers cut 8-thread evaluation on the 50,000-inverter chain from 3.5-3.7 ms to 2.9 ms with identical bits.
  Ordered group accumulation shifts results only within rounding (at most `(n - 1) ε` of summed magnitudes, and within `ε` across the differential corpus and examples).
  Linear devices (plain resistors, capacitors, inductors, coupled inductors, controlled sources, and thermal passives) evaluate conductances and capacitances once per run into a base matrix, skipping non-linear evaluation and zero terms during Newton iterations.
  Linear pre-evaluation cuts evaluation time on the 100 × 100 RC mesh from 197 ms (of 1.21 s) to 125 ms (of 1.14 s) while preserving thread determinism.
  Pre-evaluation shifts results only by rounding (`3.7e-15` V on the mesh, `7.3e-11` V on the amplifier, zero on buck converter and inverter chain).
- Shared elaboration for Monte Carlo and worst-case runs via `jolty.simulate.Elaborated`, re-preparing devices in place to match full JSON elaboration bit for bit.
- Parallel Rust batch executor (`jolty_engine::batch`) supporting explicit lists, parameter/temperature grids with optional warm chains, index-seeded Monte Carlo draws, and Cartesian products across a caller's rayon pool.
  Each batch run executes analyses and measures independently, records failure reasons without aborting the batch, caches each encountered matrix structure once, and produces bit-identical outcomes and summaries across any thread count or range split.
- Parallel frequency and right-hand-side distribution for AC, noise, stability, and sensitivity analyses once `min_solves_per_task` fills every thread, matching serial bits while reusing pivots across consecutive frequencies until pivot degradation triggers refactorization.
- Parallel sparse matrix factorization using George and Liu nested dissection on BTF blocks exceeding `max_unknowns_per_interior` whenever symbolic elimination requires fewer multiply-adds than minimum-degree ordering.
  Cost-based selection leaves chain blocks undissected while dissecting 2D meshes on two threads at 80, 100, 200, and 300 nodes per side (skipping 90 where splitting costs more).
  Interiors factor concurrently via left-looking elimination with local pivoting, separators assemble child updates deterministically into dense fronts, unpivotable columns delay cleanly to parent separators without perturbation, refactorization preserves valid pivots, and triangular solves traverse the dissection tree.
  Parallel factors and solutions are bit-identical across thread counts for a fixed ordering and agree with minimum-degree solutions within rounding.
  Default interior sizes depend on thread count while batch runs default to minimum-degree ordering for cross-thread bit identity unless overridden.
  On the 90,000-node mesh, refactorization drops from 289 ms serial to 108 ms on one thread and 28 ms on eight threads.
  Work-stealing across dissection subtrees prevents slow cores from stalling peers, while rows above a block (such as supply rails) process solutions in 1,024-entry blocks.
  On the 50,000-inverter chain, 8-thread solves drop from 1.85-2.7 milliseconds to 1.4 milliseconds.
  Wide rows above a block (over 1,024 entries) reduce in deterministic 1,024-entry chunks via `jolty_numerics::reduce::fold` so parallel execution matches serial bits, affecting only circuits with wide supply rows within rounding.
- Parallel Newton iteration across unknowns and matrix entries once `min_unknowns_per_task` fills every thread, chunking assembly, tolerance checks, updates, and deflation reductions into work-stolen 1,024-element blocks with fixed reduction order for bit-identical multi-thread results.
  Transient step control parallelizes identically across history terms, predictions, charge derivatives, divided differences, equation-filtered error ratios, step-size bounds, and ringing tests.
  On the 50,000-inverter chain, 12-thread step control drops from 1.33 s to 0.37 s.
- Strict enforcement of `Parallelism.max_threads`: when `max_threads` is at least two and smaller than the caller's pool, each analysis executes all parallel device, gather, and solver tasks inside a dedicated pool of that exact size.
- General DC parameter sweeps (`parameter=...`) alongside independent-source sweeps, re-preparing swept device instances at each point so devices hold no mutable sweep state.
- Configurable engine thread count via `threads` setting, `JOLTY_THREADS` (defaulting to logical core count), or the `jolty.simulate.threads` context manager, using a dedicated pool rather than rayon's global pool and releasing the Python GIL during engine calls.
- Python batch execution via `Simulation.batch(analyses, variants)` returning a columnar NumPy `BatchResult` populated directly by a Rust sink with parameter values, analysis statuses/failure reasons, measure values/reasons, and traced waveforms.
  Built-in `jolty.specs` measures evaluate inside Rust, while custom Python measures and `each` callbacks run under the GIL as runs finish.
  Foundry circuits execute ngspice batch runs concurrently up to the engine thread limit.
- Plan variant builders in `jolty.plan`: `Explicit` lists of `Variant` values/temperatures, `Grid` over parameter/temperature `Axis` definitions (with optional warm chains), and `Product` combinations alongside `MonteCarlo`.
  Plan `variants` run across nominal conditions and corners, reporting run counts, failures, requirement spreads, and yield with Wilson intervals.
- Statistical batch plotting: `jolty simulate` renders requirement histograms against spec bounds and min/median/max trace envelopes across runs, saving batch columns to `<script>.batches.npz`.
- Python parameter sweep API `DcSweep("R1", levels, parameter="resistance")`, storing parameter metadata in results files (format version 6).
- Stateless Monte Carlo sharding via `jolty simulate SCRIPT --samples START:STOP`, atomically writing `<script>.shards/shard-START-STOP.npz` tagged with a digest of circuits, plan, options, jolty version/build, and platform (excluding script line numbers so shards from any directory or host merge cleanly).
  `jolty merge DIRECTORY` and MCP tool `merge_shards` validate digests and contiguous non-overlapping ranges, merge shards in sample order, and emit bit-identical reports, batches, and plots.
  Shards store only measurements and explicitly named traces, warning if full-run shard storage would exceed 10 GB.
  `examples/cluster/launch.py` dispatches shards across local worker processes, retries failed shards, and merges completed outputs.
- Range-sliced batches (`Simulation.batch(..., runs=range(start, stop))`) and `BatchResult.joined` concatenation preserving deduplicated failure reasons in first-seen order.
  Results, batch, and shard serialization produces deterministic bytes depending solely on stored contents.
- Python parallelism configuration via `Options(parallelism=Parallelism(...))` and CLI flags such as `--min-solves-per-task`, supporting serial execution (`max_threads=1`) and measured break-even defaults (`jolty.plan.BREAK_EVENS`, profiled on 2, 4, 8, and 64 threads of an AMD Ryzen Threadripper 3970X).
  Small runs bypass the thread pool entirely when work stays below per-thread thresholds, and every parallelism setting produces bit-identical results across thread counts except thread-dependent default interior sizes in `max_unknowns_per_interior`.
- Transistor operating-region classification and boundary margins across MOSFETs, BSIM3, VDMOS, JFET (`vgs - vth`, `vds - vdsat`), and BJT/VBIC junctions, signed positively inside conduction/saturation regardless of polarity.
  Kernels emit regions and margins as variables accessible via `point.region("M1")` and `Simulation.transistors`, listed nearest-boundary-first in reports and saved in results files (format version 10).
  `OperatingPoint(regions=True)` records compact 3-value region summaries rather than full 48-value MOSFET records for `jolty simulate`, and cached operating points no longer re-evaluate devices to populate recordings.
- Operating-point and DC-sweep iteration statistics (`OperatingPointResult.iterations`, `DcSweepResult.strategies`, and `DcSweepResult.iterations`) tracking winning strategies, failed attempts, and logic-settling re-solves across reports, results files (format version 16), and robustness tables.
- Verilog-A `localparam` constants and static-scaled `ddt(...)` factoring in `jolty-veriloga`, supporting compile-time constant folding across typed, untyped, module, block, and analog-function scopes.
  Parameter- and temperature-dependent `localparam` values emit into a live `Constants` struct with exact `R: Real` derivatives while staying excluded from `Parameters`, serde, and JSON schemas.
  Static-scaled `k * ddt(x)` folds into `ddt(k * x)` while dynamic `C(V) * ddt(V)` is rejected as non-charge-conserving; pure-inductor `Kernel::interface` DC branches and flattened ternary `else if` chains eliminate `clippy::let_and_return`.
  Upstream IHP SG13G2 MOM capacitors (`cap_cmomf.va` and `cap_cmomi.va`) now compile unpatched into linear kernels with zero `clippy -D warnings` diagnostics.
- Isolated peak resident memory (`VmHWM` on Linux, `ru_maxrss` on macOS/BSD) and barrier-synchronized concurrent-copy throughput efficiency (`--peak-memory`, `--concurrent-copies`, `--copy-rounds`) in `benchmarks/speed/`.
  Memory and throughput reporting includes an 85% available-memory guard, IEC binary prefixes, unclamped `E(N) = T_copies / (N * T_par)`, hybrid-core notes, backwards-compatible Pydantic schemas, and the AMD EPYC 9B14 baseline `benchmarks/speed/results/430c75d6b0b402812924ebcfd22ed6c91fee6378.json`.
- Exhaustive SPICE round-trip guard over all 33 standard library models (`tests/test_spice.py`), checking both model identity and OP/AC electrical equivalence within `1e-9` relative tolerance.
- Transient charge-leak reporting (`ChargeLeak`) integrating per-node residuals over accepted steps (Kundert and Clifford 1993) with Neumaier (1974) compensated summation.
  Charge leak is exposed across Rust, PyO3, Python (`TransientResult.statistics.charge_leak`), text/JSON reports, and results files (format version 20, validating shapes and float dtypes on load), backed by closed-system conservation property tests.
- Two-stage lossless-oscillation vs trapezoidal-ringing detector (`crates/jolty-engine/src/ringing.rs`): Stage 1 applies an O(1) variable-step restoring-force check, and Stage 2 reuses cached sparse LU factors to separate true LC oscillations from algebraic `R = -1` ringing.
  On `lc_tank` at `reltol = 1e-2`, end energy ratio `E_end / E_0` improves from `0.040785` (0.02 digits) to `0.999959` (4.38 digits) with 92.3% fewer rejected steps (39 to 3) and 73.6% fewer refactorizations (106 to 28).
  `speed_buck` improves from 1.7667 s to 1.7424 s (-254 accepted, -435 rejected, -2,393 iterations), `inverter_chain_500` drops from 2,391 to 2,312 accepted steps, and all 7 transient protocol examples pass every spec.
- SPICE reader and raw-file enhancements: subcircuit-scoped `.lib`, `.include`, `.func`, `.ic`, and `.nodeset` (including `.func` inside `.model`), device `IC=` under `UIC`, typed `ngspice.parse_raw` and `files.load_raw`, CLI/MCP `--raw` and `--raw-format`, and `"psa"` PSpice compatibility mode (`VSWITCH`/`ISWITCH` via XSPICE).
- Programmatic MCP instructions and offline docs bundle: zero-dependency `crates/jolty-python/build.rs` packs 17 guide pages and 33 examples into `jolty/assets/docs.json` (428,613 bytes).
  Offline docs expose `jolty docs [TOPIC]`, `read_docs` MCP tool, 50 static MCP resources (`jolty://docs/...`), 2 URI templates, and `build_instructions()` derived directly from code.
- Session query and PSS enhancements: `jolty measure` CLI and `measure_results` MCP tool query saved `.results.npz` files without re-simulating, and `.pss` SPICE cards parse via `PeriodicSteadyStateCard` and `PeriodicSteadyState.from_spice`.
  Autonomous shooting automatically detects time-varying sources and behavioral `time` (`Waveform::varies_in_time`, `Kernel::varies_in_time`, `Runs::time_varying`, `check_periodic`), while `max_memory` applies across PSS sensitivity, JVP, and VJP.
- Upfront multi-error plan and spec validation (`PlanError`, `close_names`, `Simulation.validate_analysis`, `Simulation.validate_plan`) checking every analysis name, source, signal, record pattern, initial condition, nodeset, and axis before simulation starts and reporting all errors with nearest valid names (`tests/test_plan_validation.py`).
- Coverage matrix v2 (`benchmarks/coverage/`) expanded from 65 to 80 offline-safe rows (`corners`, `worst-case`, `disto`, `sp`, `trnoise`, `switch-voltage`, `switch-current`, `logic`, `kit-sky130`, `kit-gf180mcu`, `kit-sg13g2`, `ic`, `nodeset`, `pdk-library`, `four`) refreshed on ngspice 47 (Sparse 1.3 and KLU) with XSPICE codemodels.
  Coverage totals reached 62 agreed, 3 documented, 15 unsupported, and 0 failed/disagreed/unknown for jolty while preserving all 65 prior `xyce-serial` cells.
  Resolved the VBIC `2.114e-6` discrepancy via coupled-KCL `vbic_currents()` accounting for `RBI = RCI = RBP = 0.1 ohm`, `q1` smoothing, and `gmin` (-0.13 ULP at `gmin=0`, `8.1e-8` at default `gmin`).
  Shifted the BSIM3v3.3 probe to strong-inversion saturation (`3.3e-16` error), exposed ngspice 47 `.disto` `n=1` `malloc` underflow, KLU refusal, and `HD3` sign inversion (`-Vt/12` vs exact `+Vt/12`), and added gated `test_ngspice_column_is_what_its_fixture_finds`.
- Deterministic CI regression check (`benchmarks/deterministic.py` and `benchmarks/deterministic.json`, wired into `scripts/check python`) recording per-platform solver counts, significant digits, and SHA-256 fingerprints for all 5 speed and 14 precision fixtures under 1,000 nodes.
  Unrecorded platforms verify against bounded cross-platform tolerances (6% counts, 0.05 digits), and exact zero-pivot division (`pivots[0] == 0`) was fixed in `benchmarks/precision/exact.py::_Elimination::newton`.
- Engine constant and test tolerance provenance audit (`tests/test_tolerances.py`) citing sources for 91 named Rust constants across `jolty-engine`, `jolty-numerics`, and `jolty-checks` plus 520 Python test tolerance sites under an AST/doc-comment guard.
  Fixed two masked assertions where `pytest.approx`'s default `abs=1e-12` hid discrepancies (`tests/test_specs.py` noise checks and tiny sub-pA/sub-pF checks in `tests/test_spice.py`), tightening both with `abs=0`.
- Hypothesis parser fuzz suite (`tests/test_fuzz_parsers.py`, 62 tests) covering SPICE expressions, libraries, `jolty-expressions`, circuit JSON, results/raw files, and shards.
  Fuzzing fixed 40 Python parser/loader bugs and 3 Rust bugs: iterative DFS cycle detection in `crates/jolty-engine/src/variants.rs`, non-finite `table()` rejection in `crates/jolty-expressions/src/parser.rs`, and iterative stack-safe parsing, compilation, tree conversion, drop, and JSON serialization in `jolty-expressions` and `jolty-python`.

### Changed

- Replaced recursive descent in `crates/jolty-expressions/src/parser.rs` with an iterative type-state shunting-yard stack, boxed `Points` in `Node::Table` (64 to 24 bytes), converted `tape::compile`, `Node`/`Term` `Drop`, and `jolty-python::term_to_json_string` to iterative traversals, and made `Expression::tree()` `O(N)` via `Option::take`.
  Iterative traversals eliminated `SIGABRT` stack overflows on 2 MiB debug stacks at `MAX_DEPTH = 256` and on 5,000-term variable binary chains.
- Appended the AMD EPYC 9B14 `--view matched` (`430c75d`, 5,069 s) comparison table to `benchmarks/speed/README.md`.
- Optimized `jolty.spice.Library.drawn` using Kahn's `O(V + E)` topological sort to evaluate deterministic global `.param` declarations once while caching stochastic closures per sample in a bounded thread-safe LRU.
  Additional speedups cache parsed non-`AKO:` local `.model` cards as `MappingProxyType` views, use `split(maxsplit=1)` in `_keyword`, and fast-path finite `Number`, `Name`, and `Unary` literals in `expressions.evaluate`.
  Across distinct SKY130 `tt_mm` samples, `drawn` speeds up 29× on `cap_mim_m3_1` and 74× on `nfet_01v8`.
  Added SKY130 BSIM4 MOSFET mismatch verification tests (`sky130_fd_pr__nfet_01v8` differential pair and current mirror, matching 400-sample Monte Carlo within 1.20 and 0.91 standard errors).
- Fixed three BSIM3/BSIM4 symmetry defects near `vds = 0` and `vbs = 0`: `capmod_zero` strict `< 0.0` branch mismatch at `vbs = 0`, `vbseff` cancellation via rationalized `smooth_floor`, and small-`vds` `Vdseff`/`VdseffCV` cancellation via shared two-sided `effective_drain` (removing four manual `if vds == 0.0` hacks).
  Added MOSFET level 3 swap symmetry tests proving the `1.7e-9` difference with `w != 0` matches the exact Meyer capacitance derivative kink (`DEVqmeyer`) shared by levels 1 and 3.
- Replaced linear `np.interp` in `benchmarks/speed/bench.py` (`deviations()`) with vectorized Huynh (1993) fourth-order non-oscillatory cubic Hermite interpolation (`interpolate_transient`).
  Cubic Hermite interpolation eliminates chord sag at smooth extrema (`bjt_amp` `--view matched` deviation vs ngspice 47 drops from `3.8e-3` to `5.9e-5` on Sparse and KLU) with zero overshoot on switching edges.
- Fixed `benchmarks/coverage/capabilities.py` `tline` probe to run `Transient` and read declared keys (`4.4e-16` relative agreement vs ngspice 47; 50 agreed, 1 solved, 0 disagreed).
- Enabled parallel release codegen (`CARGO_PROFILE_RELEASE_LTO=false`, `CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16`) in `ci.yml` and `docs.yml` (leaving `release.yml` untouched for PyPI wheels) and restricted non-gating `coverage` and `mutants` jobs to `main`, weekly schedules, and manual runs.
- Initialized transient derivatives directly from the constrained starting state: using a DC Jacobian whose initial-condition rows are fixed during operating-point solves, or zero for `UIC` starts.
  Previously derivatives started from the unconstrained operating point, giving `d out/dR = -1 mV/Ω` at `t = 0` and 2× error at `t = RC` on a 1 kΩ, 1 nF RC started at `0 V`.
  Both start modes now match the closed form `-(t/(R²C)) e^(-t/RC)` within `1e-5`.
- Removed the `1/FC` upper bound on diode junction potential to match ngspice 47, and allowed zero diode breakdown current without shifting the breakdown voltage.
- Promoted periodic steady state from `Transient(..., periodic=True)` to a standalone analysis `PeriodicSteadyState(period=...)` (kind `periodic_steady_state`) across the engine, Python API, results files, reports, and viewer without aliases, moving shooting period counts from `Statistics.periods` to `PeriodicSteadyStateResult.periods`.
- Seeded growing-mode inverse iteration with pseudo-random amplitudes (matching LAPACK) instead of uniform ones, fixing symmetric circuits such as a five-stage ring oscillator where uniform seeds missed rotating modes.
- Standardized naming across the engine, Python API, extension, files, CLI, MCP server, and documentation without legacy aliases:
    - Analysis results, reports, and views form one family (`X`, `XResult`, `XReport`, `read_x`): `OperatingPointResult`, `OperatingPointsResult`, `DcSweepResult`, `AcResult`, `TransientResult`, `SensitivityResult`, `StabilityResult`, `NoiseResult`, `PoleZeroResult`, `TransferFunctionResult`, and `AnalysisResult` replace `OperatingPoint`, `OperatingPoints`, `Sweep`, `Response`, `Waveforms`, `Sensitivities`, `LoopGain`, `NoiseSpectrum`, `PolesZeros`, `Transfer`, and `Result` (`jolty_engine::Outcome`).
      Matching reports (`OperatingPointReport`, `OperatingPointsReport`, `DcSweepReport`, `AcReport`, `TransferFunctionReport`), Rust `measure::AnalysisResult` with `Measure.read_*` methods, and `_core` dictionaries (`AcResult`, `Work`, `Profile`, `Statistics`, `NoiseInput`, `Problem`, `Diagnostic`, `Excursion`, `BatchResult`) replace legacy `*Outcome` and short names.
    - Replaced `Measurement` with `Measure` everywhere (`jolty.specs.Measure`, `MeasureError`, `Requirement.measure`, `Simulation.batch(measures=...)`, `BatchResult.measures`, and batch/shard format version 2).
    - Renamed unknown physical categories from `Quantity` to Verilog-AMS `Nature` (`jolty.unknowns.Nature`, `Unknown.nature`, `NoiseReport.referred_nature`, `jolty_models::Nature`, and results format version 11), reserving `Quantity` for dimensioned values.
    - Renamed script provenance from `Source` to `SourceLocation` (`jolty.ir.SourceLocation`, `jolty_ir::SourceLocation`, `source_location` fields, circuit format version 2, and guide heading "Source locations"), reserving `source` for independent sources.
    - Unified instance path plus source location under `Location` (`jolty.checks.Location`, `_core.Location`), reserving `Instance` for circuit elements.
    - Renamed instance-path fields to `instance` (`Excursion.instance`, `sensitivity::Parameter::instance`, `noise::NoiseContribution::instance`, and results format version 12).
    - Unified analysis traces under `trace`/`traces` (`Simulation.batch(traces=...)`, `BatchResult.traces`, `Traces`, `--traces`, `SimulateRequest.traces`, `PlotRequest.traces`, `plots.default_traces`, `LogicTrace`, and batch format version 3), reserving `probe` for stability analyses.
    - Renamed recorded device internal quantities from `readings` to `variables` (results format version 13), reserving `reading` for measure values.
    - Renamed instance identifiers from `id` to `name` (`ir.Instance.name`, `jolty_ir::Instance::name`, `Builder.place(..., name=...)`, `Placed.name`, layout instances, circuit format version 3, and 3D viewer scenes).
    - Renamed `ir.DeviceOf.device` to `model` (circuit format version 4 and 3D viewer scenes).
    - Unified logic states under `LogicLevel` (`jolty.devices.LogicLevel`, `format_logic_level`, `*.logic_level(net)`, `report.NetLevel`, `jolty_models::LogicLevel`), reserving `level` for rule severity.
    - Normalized bound options to `min_x`/`max_x`: `max_dc_iterations`, `max_transient_iterations`, `max_memory`, and `OperatingPoints(max_solutions=...)`.
    - Normalized `Parallelism` settings across Rust, Python, CLI, and benchmarks to `max_threads`, `min_solves_per_task`, `min_instances_per_task`, and `min_unknowns_per_task`, defaulting each threshold to its measured thread-count break-even.
    - Standardized bounds across intervals (`Yield.lower`/`upper`), frequency ranges (`start`/`stop`), and extrema (`Excursion.minimum`/`maximum`, `jolty_engine::logic::Excursion`, results format version 14).
    - Renamed `LEAST_PHASE_MARGIN`/`LEAST_GAIN_MARGIN` to `MIN_PHASE_MARGIN`/`MIN_GAIN_MARGIN`, and BJT `minimum_base_resistance` to `min_base_resistance` matching MOSFET `max_drift_velocity`.
    - Renamed transient forced times from `times` to `breakpoints` (`Transient(..., breakpoints=[...])`), reserving `times` for solved result timestamps.
    - Spelled out JFET and VDMOS temperature parameters (`transconductance_parameter_temperature_exponent`, `threshold_voltage_temperature_coefficient`) while preserving BJT `beta_temperature_exponent`.
    - Spelled out BSIM3 Python arguments (`threshold_adjustment`, `mobility_multiplier`, `mobility_model`, `capacitance_model`, `noise_model`, `check_parameters`, `charge_partition`) while preserving SPICE card names for import/export.
    - Renamed `Varied` to `Toleranced(instance, parameter, nominal, tolerance)` in Python (`jolty.variation`) and Rust (`jolty_engine::variants`).
    - Renamed `jolty.builder.Point` to `Connectable`, reserving `Point` for layout coordinates.
    - Renamed `jolty.variation.Lot` to `LotMember`, reserving `Lot` for `jolty.ir.Lot`.
    - Renamed `MonteCarloReport.passing` and `VariantsReport.passing` to `yield_` (serialized as `yield` in JSON).
    - Renamed `Composite`/`composites` to `SummedTerminal`/`summed_terminals` in Rust, Python, and results format version 17.
    - Aligned kernel shape names to full model names (`VoltageControlledVoltageSourceShape`, `CurrentControlledCurrentSourceShape`, `CoupledInductorsShape`, `LogicDriverShape`, `ThresholdDetectorShape`).
    - Renamed `Native`/`native` measure resolution to `ResolvedMeasure`/`resolved`.
    - Unified recording types under `record`: `RecordedValues`, `RecordedSeries`, `RecordedView`, and `Recording` replace `Observation`, `Observations`, `Observed`, and `Recorded`.
    - Renamed batch `Outcome` to `Run` (`jolty_engine::batch::Run`, `jolty.results.Run`).
    - Renamed `jolty_models::UnknownInfo` to `Unknown` (matching `jolty.unknowns.Unknown`) and `jolty_models::Unknown` to `UnknownIndex`.
    - Renamed diagnostic rules to factual names (`low-phase-margin`, `low-gain-margin`, `near-rating`) and moved thresholds (`MIN_PHASE_MARGIN`, `MIN_GAIN_MARGIN`, `DERATING_FACTOR`) into `jolty.checks`.
    - Standardized SHA-256 identifiers on `digest` (`circuit_digest`, `plan_digest`, `shards.digest`, shard format version 3, and `spice.fetch(url, digest)`).
    - Renamed `jolty.specs.average` and `Measure::Average` to `mean` and `Measure::Mean`.
    - Renamed `jolty.checks.Report` to `CheckReport`, `jolty.fidelity.Model` to `CharacterizedModel`, and `NoiseContributor` to `NoiseContribution` (with `name` and `contributions`).
    - Renamed `jolty.spice.ngspice.Raw` to `Result`, matching `jolty.spice.xyce.Result`.
    - Renamed Rust types to domain terms: `Differentiable` (for `Native`), `mosfet::ModelLevel` (for `Level`), `logic::ExcursionKind` (for `Bound`), and `integration::ErrorBound` (for `Bound`).
    - Renamed thermal-voltage matching helpers to `option_temperature` across `ngspice`, `xyce`, and dialects, reserving `celsius` for `jolty.units.celsius`.
    - Standardized netlist-line parameters on `deck` across `ngspice.run`, `ngspice.runs`, `xyce.run`, and `execute`.
    - Renamed catalog `Device`/`device` to `PartKind`/`kind` (catalog format version 2) and part `number` to `part_number` (catalog format version 3).
    - Renamed `jolty.specs.Values` and `deviation` to `SummaryStatistics` and `standard_deviation`.
    - Standardized alternate constructors on `from_*` (`Simulation.from_elaborated`, `Simulator.from_elaborated`, `Provenance.from_circuit_and_plan`, `SummaryStatistics.from_numbers`) while keeping accessor `of` methods.
    - Renamed source amplitudes from `level`/`levels` to `value`/`values` (`Dc(value)`, `DcSweep(source, values)`) and DC sweep axes to `swept` (results version 15, batch version 4), reserving `level` for rules, logic, and MOSFETs.
    - Standardized report limits on `max_unknowns`/`max_output` (`JOLTY_MAX_UNKNOWNS`, `JOLTY_MAX_OUTPUT`, and `report(..., max_unknowns=...)`).
    - Normalized MCP tools to verb-object names (`view_circuit`, `cost_circuit`, `explain_rules`) and KiCad export to `export-kicad`/`export_kicad`/`ExportKicadRequest`/`ExportKicadResult`/`export_kicad_netlist`.
    - Spelled out VBIC `epitaxial_*` parameters and VDMOS `quasi_saturation_resistance_temperature_exponent` in circuit format version 5.
    - Standardized documentation lookup on `jolty read-docs` (`read_docs`, `ReadDocsRequest`, `ReadDocsResult`), transfer-measure targets on `signal`, `rise_time(lower, upper)`, and `crossing`/`frequency(threshold)`.
    - Disambiguated `report.NetChargeLeak`, `results.PhasorPart`, `service.RuleSummary`, `runner.JobResult`, and `viewer.save_viewer`, and standardized US English spellings (`models.center`, `catalog.neighbors`, `Dimension.meter`) across Rust and Python.
- Recorded provenance metadata in results files (format version 9) including jolty version, build, platform, and circuit/plan digests, readable via `jolty.files.load_provenance` and verified by `load_results` when given a circuit.
  Moved `build`, `platform`, and `circuit_digest` from `jolty.shards` to `jolty.files` alongside `plan_digest` for shard digest computation.
- Attached logic domain `excursions` directly to individual analysis results (results format version 8) and removed stateful `Simulation.excursions`.
  Unmonitored runs (batches, ngspice, nested analyses) record `None`, failed analyses record no excursions, and results headers now write the true format version (fixing version 7 files previously tagged as 6).
- Computed decade sweep frequencies (`jolty.plan.decade`) in decimal arithmetic with single rounding so every platform generates identical frequency grids despite differing `log10`/`pow` rounding.
- Estimated Gear2 local truncation error using the variable-step constant `(1 + ω)²/(6ω(1 + 2ω))` and solved next/retry steps at the true ratio `ω`, replacing SPICE's constant-step 2/9 (which underestimates error 7.6× after a 10× step cut).
  Gear2 runs now adapt accurately across step-size changes, while constant-coefficient trapezoidal (default) and backward Euler methods are unchanged.
- Aligned `charge_tolerance` with SPICE `CHGTOL`: `relative_tolerance · max(|q|, charge_tolerance)` (ngspice 47 `cktterr.c`, lines 40 to 41), replacing additive `relative_tolerance · |q| + charge_tolerance` that masked errors below 10 pC.
  At defaults, the 500-inverter chain moves from 749 to 2,448 steps (2.2-2.3× runtime) and the buck converter moves from 73,880 to 82,710 steps (1.11-1.12× runtime).
  Step-capped examples shift under 1%, and differential tests now run both engines at the standard `1e-14` C `charge_tolerance` default instead of `1e-18` C.
- Lowered the minimum transient step from `1e-9` to SPICE's `1e-11` of the maximum step (ngspice 47 `traninit.c`, line 36), allowing picosecond settling behind small resistances such as the buck-converter switches at `reltol = 1e-5` (which previously halted at 0.39 ms).
- Enforced `max_transient_iterations` strictly per time step rather than continuing indefinitely while steps shrank.
  Adopted Hairer and Wanner's (1996, section IV.8) convergence criterion using `θ / (1 - θ)` times the step norm (between un-limited evaluations), accepting linearly converging iterations once remaining distance meets tolerance and cutting extra iterations when contraction exceeds 0.5.
  At defaults, the 500-inverter chain drops from 14,361 to 12,090 iterations, and examples drop up to 5.6%.
  The buck converter takes 90,783 steps (vs 82,717) in similar iterations (331,066 vs 330,409, +4 to 5% runtime), reducing worst-case voltage error from 21.5 mV to 15.2 mV.
- Clamped Wilson confidence interval endpoints exactly to 0 (when zero samples pass) and 1 (when all pass), eliminating rounding artifacts such as `1.084e-17` for 0/2,000.
- Required Python 3.15 or newer (matching CI and enabling PEP 810 lazy imports to avoid loading matplotlib unnecessarily), building the extension against the Python 3.15 stable ABI.
- Moved Monte Carlo sampling into the Rust engine using ChaCha20 keyed by seed, sample index, and part or lot name so every sample `i` is thread-independent and unaffected by adding unrelated parts.
  `jolty.variation.sample` now accepts a 64-bit seed (`0` to `2⁶⁴ - 1`) and sample index instead of a NumPy generator, producing a new deterministic sample sequence.
- Migrated Monte Carlo, corner, and worst-case workflows to the Rust batch executor on shared elaborated circuits, running temperature-only corners as batched value sweeps.
  Reports match bit for bit except in Monte Carlo trailing digits because samples now run at exact drawn floats rather than JSON-parsed values (which differed by 1 ULP on roughly one in five numbers).
- Switched the Python extension allocator to `mimalloc` with per-thread heaps, improving 8-thread batch scaling from 64-68% to 95.9% of linear.
- Enabled `serde_json`'s `float_roundtrip` feature so the engine reads exact Python floats (such as `981.4902141434463` instead of `981.4902141434465`), eliminating 1-ULP parse drift on roughly one in five numbers.
- Reformulated photocell response `(E + √(E² + d²)) / 2` to avoid catastrophic cancellation below -67 lux, keeping the slope finite at arbitrarily low irradiance.
- Oriented PNP excess-phase unknowns (Gummel-Poon `delayed_transport` and `delayed_transport_slope`; VBIC `transport_filter` and `delayed_transport`) consistently with PNP terminal signs so PNP mirrors NPN bit for bit.
- Allowed specification measures to return infinite values (such as `-∞` dB at an exact notch) and judge them with infinite slack, treating only `NaN` as unmeasured.
  With exact float parsing, the twin-T notch now reports `-∞` dB rather than unmeasured.
  JSON output encodes infinities as `"Infinity"` or `"-Infinity"` per report schemas and writes `null` for undefined spread means and standard deviations.
- Wrapped `Bsim4` in `Thermal::<Bsim4>` for self-heating parity across all eight native transistor models, unified BSIM3/BSIM4 binning macros, and exported parameter dimension and BSIM card metadata from `jolty._core`.
- Made `matplotlib` an optional `jolty[plots]` extra (included in `jolty[mcp]`) with PEP 810 lazy imports so core `jolty` requires only `numpy`, `pydantic`, and `pydantic-settings`.
- Shared `viewer.css` and `viewer.js` across static `.view.html` pages and compacted tracked benchmark and catalog JSON fixtures diffably, cutting repository text footprint by over 65,000 lines.
- Replaced `Grid` `warm: bool` with `GridChain::{Independent, Warm}` in `jolty-engine`, removed `serde` from `jolty-numerics` and runtime `serde_json` from `jolty-checks`, upgraded `jolty-derive` to `syn` 3, simplified `jolty.cli` onto a declarative command table, and added Google-style `Args:`/`Returns:` docstrings across `jolty`.
