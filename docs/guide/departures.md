# Departures from ngspice

jolty agrees with ngspice where ngspice is right, and computes otherwise where another way is more accurate or more robust, or where the choice is open, as the Newton iterations a time step allows are.
This page lists every place where the two engines are known to compute differently, each with its evidence, a line of ngspice's source or a measurement, and a link to where the guide describes what jolty does.
The lines are those of ngspice 47's release, built as it is by default, with XSPICE.
The measurements on the benchmarks' synchronous buck converter (`benchmarks/speed/circuits/buck.py`) are of its plan's 2 ms at the default truncation tolerance, made on an Apple M2 Max on 14 and 15 November 2026.

The differential tests compare everything but these departures, and run both engines where they can keep them apart: at the same tightened tolerances, and with jolty stepping onto every time ngspice took ([Simulation](simulation.md#how-the-engine-solves-a-circuit)).

## Time steps and their error

How a transient chooses its steps and judges their error, which [the table of methods](simulation.md#how-the-engine-solves-a-circuit) describes.

| Departure | ngspice 47 | jolty | Evidence |
| --- | --- | --- | --- |
| Charges judged | A MOSFET of level 1, 2 or 3 judges its gate charges alone, never its bulk junctions' (`mos1trun.c`, lines 24 to 26, as `mos2trun.c` and `mos3trun.c`); BSIM3 judges its bulk, gate and drain charges (`b3trunc.c`, lines 38 to 40), the junctions folded into the first and the last (`b3ld.c`, lines 2797 to 2802) | The charge at every node, a MOSFET's junctions' among them, and each of a MOSFET's Meyer displacements', the whole oxide's charge across it, of which Meyer's capacitance carries a share | On the benchmarks' synchronous buck, whose switched node the junctions hold, ngspice takes 56,624 time points to jolty's 103,349 steps tried, and 143,339 when made to judge the junctions too. On their chain of 500 inverters, jolty accepts 2,448 steps and its edges lie 0.23 ns on average from a reference at a relative tolerance of 1e-6, where ngspice keeps 903 points and misses them by 0.51 ns, and takes 3,602 at a relative tolerance of 1e-4 to miss them by 0.22 ns |
| The error's filter | The divided differences as they are (`cktterr.c`, lines 43 to 59) | Filtered through the step's own equations, `C (G + αC)⁻¹ α` | A component that settles within the step, which the formula damps, is otherwise read as an error (Shampine, 1980; Hairer and Wanner, 1996, IV.8) |
| Gear2's error constant | 2/9, the constant step's, after any step (`cktterr.c`, lines 24 to 31) | `(1 + ω)²/(6ω(1 + 2ω))`, at the step's ratio `ω` to the one before it | In exact arithmetic on a cubic, the true error is 7.56 times the 2/9 estimate after a step cut to a tenth and 1.69 times after a halving; jolty's unit test covers ratios from a tenth to ten |
| The first step after a corner | Judged at order 1 by divided differences that reach back across the corner, as the history is never restarted, and order 2 tried at once (`dctran.c`, lines 489 to 494 and 821 to 833; `cktterr.c`, lines 44 to 59) | The history restarts at the corner, and the first step is judged by the second's points, which share its curvature | Points from before a corner hold another curvature than the steps after it |
| When a step is accepted | When the step its error allows is longer than 0.9 of the one taken (`dctran.c`, line 801) | When its error is within its tolerance, the next step 0.9 of the longest its error allows | On the benchmarks' buck, SPICE's rule took 73,299 accepted and 12,527 rejected steps, jolty's 65,606 and 7,807 |
| The step the tolerances allow | Each tolerance as it was over the step just tried, the current's included, though it grows with the step (`cktterr.c`, lines 37 to 42 and 69 to 76) | The longest step meeting each tolerance as it scales with the step: the charge's holds, and the current's grows with the step | The current's tolerance is a current times the step's length |
| The divided difference's floor | The error constant times the divided difference floored at `abstol` (`cktterr.c`, line 69) | No floor: a charge whose divided difference vanishes allows any step, which the longest step bounds | `abstol` is in amperes, and a charge's divided difference in coulombs per second to the power `p + 1`, so the floor's effect depends on the unit of time |
| The absolute tolerance of a flux | `abstol` plus `reltol` times the inductor's voltage, the flux's derivative: 1e-12 of a volt by default (`cktterr.c`, lines 37 and 38, for the flux `indtrunc.c` judges at line 24) | `voltage_tolerance` plus `relative_tolerance` times the voltage, as the equation of a flux constrains a voltage | `abstol` is a current's tolerance, added to a voltage |
| Two steps before a breakpoint | Equalised when 1.9 steps would pass it (`dctran.c`, lines 588 to 598) | Not equalised: a step that would pass the breakpoint lands on it | On the benchmarks' buck, equalising cost 11% more accepted steps and 47% more rejected ones |
| The longest step by default | The print step `TSTEP` where it is under a fiftieth of the run, unless `nostepsizelimit` is set (`traninit.c`, lines 26 to 34) | A fiftieth of the run, as a transient here has no print step | A comparison of the two engines sets the longest step in both |
| Ringing | The trapezoidal rule throughout; `xmu` below ½ damps every step (`cktntask.c`, lines 113 to 119) | A step of backward Euler where a charge's derivative turns back and forth twice in a row | The trapezoidal rule carries an error in a derivative from step to step undamped |
| A behavioral source's jumps | Stepped across where they fall | Located within the step, landed on, and the integration restarted there | ngspice may stall at such a jump ([Simulation](simulation.md#how-the-engine-solves-a-circuit)) |
| The prediction | Each device extrapolates its own junction voltages (`mos1load.c`, lines 200 to 222) | The node voltages are extrapolated | The same in effect: on the benchmarks' buck, the two engines' first iterates agree at a traced edge |
| After Newton runs out of iterations | The step cut by 8 and the next attempt predicted again (`dctran.c`, line 733) | Cut by 8, its Newton iteration going on from its last iterate | An iteration converging slowly, as at a transistor turning off with nothing else to hold its source, goes on where it stopped |

The current `i` a charge carries is held, as in SPICE, to `abstol + reltol · |i|` over the step, times `trtol` (`cktterr.c`, lines 37 to 42).
SPICE's estimate leaves out the factorial that makes a divided difference a derivative, which jolty puts in, so that `truncation_tolerance` bounds the error as SPICE's `trtol` does.

## Newton iteration

How Newton iteration converges, which [the table of methods](simulation.md#how-the-engine-solves-a-circuit) describes.

| Departure | ngspice 47 | jolty | Evidence |
| --- | --- | --- | --- |
| Iterations a time step | 100, whatever `ITL4` says: any limit below 100 is raised to it (`niiter.c`, lines 37 to 39), and a time step asks for `ITL4`, 10 by default (`dctran.c`, line 709; `cktntask.c`, line 106); a DC sweep's `ITL2` of 50 becomes 100 too | `max_transient_iterations`, 10 by default, SPICE's documented `ITL4`, the iterate after the last judged too; a DC sweep's level takes `max_dc_iterations`, 100, as ngspice's does in effect | With 100 iterations a step, the benchmarks' buck rejects 24% fewer steps and takes 14% fewer Newton iterations |
| The convergence test | Each unknown's update within `reltol` and `vntol` or `abstol` (`niconv.c`, lines 40 to 80), and each device's prediction of its own current (`niconv.c`, lines 82 to 95, as `macros.h` defines `NEWCONV` at line 19); no residual | The update, in a time step `θ/(1 − θ)` times it where the iteration contracts by `θ` from one step to the next, every equation's residual within its tolerance and the rounding of the terms it sums, and no voltage limited | Kirchhoff's law holds at the solution, not only the devices' predictions; the rounding floor lets a milliohm between two nodes at hundreds of volts converge; `θ/(1 − θ)` times the last step bounds the distance a contracting iteration leaves to the solution (Deuflhard, 2004, section 2.1), which the step alone does not once `θ` passes ½ |
| A behavioral source's convergence | Its value against its last within `reltol` and `vntol` or `abstol`, with no floor at the rounding of the unknowns it reads (`asrcconv.c`, lines 41 to 55) | Its equation's residual, with the rounding floor | ngspice 47 could not finish the [memristive NOR](../discovery/memristive-nor.md)'s transient at its corner `high` within 300 s, at jolty's tolerances or its own |
| What holds an iteration unconverged | The junction limiter alone: `fetlim` and `limvds` limit without it (`mos1load.c`, lines 376 to 384 and 719 to 720; BSIM3's `b3ld.c`, lines 187, 359 to 366 and 2437 to 2438) | Any of the three limiters | An iterate a limiter moved is not a solution of the circuit's equations |
| The voltage a step's first iteration limits from | The last accepted point: each MOSFET sets its state to it before predicting (`mos1load.c`, lines 201 to 208) | The device's last evaluation, which after a rejected step is that step's last iterate | On the benchmarks' buck, limiting from the last accepted point saves 1.5% of the steps |
| The junction limiter | ngspice's own, which for a step just over 2vt steps backward, towards −∞ as the step falls to 2vt (`devsup.c`, lines 54 to 61) | SPICE3's `pnjlim`, which steps at most `vt · ln(1 + Δ/vt)` beyond 2vt | ngspice's halves the benchmarks' buck's failed Newton iterations, 2,768 to 1,289, but leaves its steps where they were; both jump at 2vt |

## Operating points and sweeps

How the engine finds an operating point and sweeps it, which [the table of methods](simulation.md#how-the-engine-solves-a-circuit) and [Several DC solutions](simulation.md#several-dc-solutions) describe.

| Departure | ngspice 47 | jolty | Evidence |
| --- | --- | --- | --- |
| Where Newton starts | Each junction at its critical voltage and each MOSFET at `vbs = −1`, `vgs = VTO`, `vds = 0` (`dioload.c`, lines 162 to 165; `mos1load.c`, lines 397 to 408) | Every unknown at zero | Not measured yet |
| When Newton fails | Dynamic gmin stepping, then true gmin stepping (`cktop.c`, lines 57 to 77), Gillespie's source stepping (lines 85 to 92), then a transient of 10 µs at 100 ns steps (`init.c`, lines 77 to 93), whose end is taken without a check that the circuit rests there | A conductance from every node to the ground, then across every junction, each stepped down; source stepping; then pseudo-transient continuation until a step leaves the circuit where it was | Pseudo-transient continuation passes a fold where source stepping stops at 51% of the sources, and finds ngspice's transient operating point to its printed digits |
| A DC sweep's level that fails | A full operating point from ngspice's starting point (`dctrcurv.c`, lines 306 to 313), which may land on another branch of the curve | Approached from the last level in steps halved after each failure | The sweep follows its own branch |
| Several DC solutions | Not searched for | Found by deflation, each a solution of the circuit | [Several DC solutions](simulation.md#several-dc-solutions) |
| A transfer function's resistance where no current flows | 10²⁰ Ω (`tfanal.c`, lines 125 and 126) | Infinite | No current can flow there ([Analyses](simulation.md#analyses)) |

## Linear solves

| Departure | ngspice 47 | jolty | Evidence |
| --- | --- | --- | --- |
| The pivot threshold | A thousandth of the column's largest entry (`cktntask.c`, line 124) | A tenth | On random circuits' matrices, jolty's solutions' backward errors stay below 10⁻¹⁴, where a thousandth's reach 10⁻⁸ ([the table of methods](simulation.md#how-the-engine-solves-a-circuit)) |
| A resistance below 1 Ω | A conductance | Its current an unknown of its own | Beside a conductance of more than a siemens, a node's picosiemens keep fewer than four digits; the results differ by rounding alone |

## Small-signal analyses

How the engine solves the circuit linearized at its operating point, for AC, noise, stability, sensitivity and pole-zero analyses.

| Departure | ngspice 47 | jolty | Evidence |
| --- | --- | --- | --- |
| An AC sensitivity's frequencies on a linear sweep | The first as asked, each after it the one before times the step: `inc_freq` tests the sweep's type against AC's `LINEAR`, 3, where a sensitivity's is `SENS_LINEAR`, 15 (`cktsens.c`, line 831; `acdefs.h`, line 40; `sensdefs.h`, lines 80 to 87) | The frequencies asked for | The coverage matrix's first run: only the first frequency agreed |
| Each solve with `G + jωC` | Solved once with the pivots kept from the frequency before | Refined while its componentwise backward error exceeds the machine epsilon and each step halves it | A series RLC far above its resonance: 11.4 correct digits in ngspice with KLU and before in jolty, 15.6 now |

## Devices

Each device's departures are stated beside the device too, as the links say.

| Departure | ngspice 47 | jolty | Evidence |
| --- | --- | --- | --- |
| Physical constants | CODATA 2014's: Boltzmann's constant 1.38064852e-23 J/K and the electron's charge 1.6021766208e-19 C (`const.h`, lines 32 and 37) | The SI's exact values, 1.380649e-23 J/K and 1.602176634e-19 C | The differential tests run ngspice where its thermal voltage is jolty's ([Devices](devices.md)) |
| Derivatives | Wrong or missing: the bipolar transistor's base resistance, the MOSFET's threshold with the bulk forward-biased, BSIM3's bulk voltage clamp and `capmod` 0 with poly depletion, BSIM4's body capacitances of `capmod` 2, VDMOS's quasi-saturation, subthreshold mobility and reverse channel-length modulation | Exact: the bipolar transistor's written by hand, the others' by automatic differentiation | Each device's section: the [bipolar transistor](devices.md#bipolar-transistor), the [MOSFET](devices.md#mosfet), [BSIM3](devices.md#bsim3), [BSIM4](devices.md#bsim4) and the [power MOSFET](devices.md#power-mosfet) |
| Meyer's capacitances in a transient | The charge accumulated as `q₁ + C̄ · Δv` and integrated with the derivative `C̄` alone (`mos1load.c`, lines 821 to 827 and 863 to 868) | `C(v) · dv/dt`, carried by displacement unknowns with exact derivatives | Neither conserves charge, as no charge yields the capacitances ([MOSFET](devices.md#mosfet)) |
| VDMOS's gate-drain charge | The capacitance integrated by the trapezoidal rule from one time point to the next | The capacitance's exact integral, a conserved charge | [Power MOSFET](devices.md#power-mosfet) |
| A bipolar transistor's excess phase at AC | A pure delay | The Bessel filter the transient integrates | 5% apart in gain at a `PTF` of 30° ([Bipolar transistor](devices.md#bipolar-transistor)) |
| VBIC's pnp with an excess phase | Not its npn's mirror: 3.01 mA of collector current at DC where the mirror gives 56.7 µA (`vbicload.c`, lines 449, 485, 486 and 1139) | The npn's exact mirror | [VBIC](devices.md#vbic) |
| VBIC's parasitic base resistance's noise | Across the emitter resistance (`vbicnoise.c`, lines 122 and 123) | Across the parasitic base resistance | [VBIC](devices.md#vbic) |
| A heated VBIC's temperature | Its parameters mapped from the card's nominal temperature to the ambient, then from the ambient to the device's heated temperature (`vbicload.c`, lines 132 and 1594 to 1600) | Mapped from the card's nominal temperature to the device's, as VBIC 1.3's Verilog-A maps them (`vbic_1p3.va` in IHP's SG13G2 kit, lines 572 and 709 to 713) | IHP's `npn13G2` at 85 °C, heated 111 K at 1 V of base drive: its collector current 2.4% of its largest apart; unheated, or at its card's 27 °C, within 1e-6 ([VBIC](devices.md#vbic)) |
| Cards ngspice runs wrongly | VDMOS with `RQ` and `VQ` but no `RD`, and a depletion threshold `FC · VJ` above 2.5 V, where its charge jumps | Refused | [Power MOSFET](devices.md#power-mosfet) |
| BSIM4's resistances in a noise analysis | At `TNOIMOD` 1, 1 mΩ behind the drain and the source of a transistor that has no resistance there, in a deck that runs a noise analysis alone (`b4v5set.c`, lines 1823 and 1870; `b4v5temp.c`, lines 1638 and 1669) | None: every analysis runs the same circuit | SKY130's `nfet_01v8` in a stage: its output noise 4.2e-7 apart, 1.4e-7 with resistances of its own ([BSIM4](devices.md#bsim4)) |
| Switches at AC | ngspice 42's XSPICE switches give their small-signal conductance the opposite sign of their DC one | Written for ngspice as behavioral currents of their own law | [Switches](devices.md#switches) |
| Self-heating | Only its electrothermal models, such as VBIC's | Every semiconductor on a thermal net | [Self-heating](devices.md#self-heating) |
| An element's copies within a subcircuit placed with copies | The subcircuit's multiplier alone, outside HSPICE's compatibility: the element's own `m` overridden (`inpcom.c`, lines 4257 to 4283) | Their product in every mode, as copies of copies are | The rule itself: ngspice multiplies them in HSPICE's compatibility |
| A bipolar transistor's base-collector saturation current, its card giving no `IBE` or `IBC` | Multiplied by the area twice: the saturation current it takes is already the area's (`bjttemp.c`, lines 168 and 177 to 179), and is multiplied by it again (lines 198 to 202) | Multiplied once | The coverage matrix's `vreg`: its output 8 to 12 mV apart |
