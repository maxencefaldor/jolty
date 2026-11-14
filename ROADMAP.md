# Roadmap

What is agreed and waiting, and what ngspice does that jolty does not yet.
Items keep their numbers; each leaves this file when it lands on `main`, and its change says so in the CHANGELOG.

## Agreed and waiting

| # | Item | Waits for |
| --- | --- | --- |
| 7 | Fewer transient steps without heuristics, each benchmarked against ngspice: variable-order BDF and dense output at requested times. At default options jolty takes within 2% of ngspice 47's accepted steps on every example. On the benchmarks' synchronous buck it takes 90,774 attempts to ngspice's 56,624 time points, because it bounds the integration error of every charge, the MOSFETs' bulk junctions' among them, where ngspice's `MOS1trunc` judges only the gate charges: ngspice made to judge the junctions too takes 143,339, so at equal error control jolty takes 37% fewer steps. | A circuit where they are |
| 8 | A benchmark of what jolty gives an agent: the same tasks done with jolty and with ngspice, compared by success and by iterations. | A design |
| 12 | A parallel sparse LU, by partitioning along the circuit's hierarchy: each component instance's interior factored apart, its ports a border factored last, as in Basker. Pivoting within a level of columns (NICSLU) gives 1.0–1.4× on the benchmarks' meshes, and a pipelined refactorization 1.65× on four threads, so it is not done. | The many-cores plan's Phase 5 |
| 14 | Krylov pole-zero, which finds the poles near a region rather than all of them, and matrix-free shooting. | A circuit whose dense matrices outgrow the QZ, which takes 0.77 s at 500 unknowns |
| 18 | Newton iteration that converges more than linearly where a transistor turns off with nothing else to hold its source. On the benchmarks' buck jolty fails 2,768 Newton iterations and ngspice 47 none: at a gate edge the lower transistor's body diode turns on, and SPICE3's junction limiting, which jolty follows, raises it about 0.08 V an iteration. ngspice 47's `DEVpnjlim`, `vt·(2 + ln(Δ/vt − 2))` rather than `vt·ln(1 + Δ/vt)`, halves the failures in jolty (1,289) but saves no work, 73,891 steps and 1.2% more iterations, and steps backward just above 2 vt, so jolty keeps SPICE3's. Limiting from the last accepted point, as ngspice does, conflicts with jolty retrying a failed step from its last iterate. Neither the prediction, the residual test, counting only the junctions' limiting as unconverged, nor the halving of the last gap before a breakpoint is the cause. | An idea that saves work |

## What ngspice has that jolty does not

1. Verilog-A compact models compiled by OpenVAF and loaded as OSDI libraries: BSIM-CMG, BSIM-BULK, PSP, HiSIM, HICUM, EKV and the rest of the foundries' models.
1. SPICE3's built-in compact models beyond jolty's: BSIM3, BSIM4, BSIM-SOI, VBIC, VDMOS, MESFETs and the higher MOSFET levels.
1. Native process design kits: jolty runs SKY130's transistors through ngspice, and has no flow yet for GF180MCU or IHP's SG13G2.
1. Transmission lines: lossless, lossy (LTRA), and coupled lines.
1. XSPICE's code models, analog and digital, and its mixed-signal interface.
1. The distortion analysis (`.disto`).
1. The control language: scripts, loops, parameter alteration and interactive analysis.
1. Numerical device simulation of semiconductors (CIDER).
1. Device bypass and OpenMP-parallel evaluation of its BSIM4 model.
