# Roadmap

What is agreed and waiting, what is still to decide, and what ngspice does that jolty does not yet.
Items keep their numbers; each leaves this file when it lands on `main`, and its change says so in the CHANGELOG.

## Agreed and waiting

| # | Item | Waits for |
| --- | --- | --- |
| 2 | Sensitivity twins local to their devices, so sensitivity costs O(P·k) rather than O(P·n). | Nothing |
| 4 | Batched variants in Rust: one elaboration for every sample, corner and worst-case point, run in parallel, with a setting for the number of threads and the GIL released in every long call. | Nothing |
| 5 | Spec measurements in Rust, so a batch returns verdicts rather than waveforms. | 4 |
| 6 | AC, noise, stability and sensitivity solved in parallel across frequencies and columns, and devices evaluated and assembled in parallel, bit for bit the same whatever the number of threads. | 2 and 4 |
| 7 | Fewer transient steps without heuristics, each benchmarked against ngspice: variable-order BDF and dense output at requested times. With every step's error judged and filtered, jolty takes 1% more steps than ngspice 47 on the buck converter and 7.5% more on the boost, and as many on the oscillators, with fewer Newton iterations everywhere. On the benchmarks' synchronous buck, jolty accepts 122,587 steps and rejects 32,900 where ngspice keeps 52,982 and rejects 3,642, all of the difference at the four gate edges of each period, and most of it item 17's: at SPICE's truncation tolerance for the trapezoidal rule, jolty accepts 65,606 and rejects 7,807. So neither is yet where the steps are. | A circuit where they are |
| 8 | A benchmark of what jolty gives an agent: the same tasks done with jolty and with ngspice, compared by success and by iterations. | A design |
| 9 | The docs' schematics and the gallery drawn again, after the schematic layout's new row order and routing. | Last, once the layout and the engine settle |
| 10 | Continuous integration. | GitHub Actions minutes |

## To decide

| # | Question |
| --- | --- |
| 11 | Whether the MOSFET keeps its three displacement unknowns, which conserve the Meyer capacitances' charge, or condenses them. |
| 12 | A parallel sparse LU, such as Basker's or NICSLU's. |
| 13 | Multirate or waveform-relaxation transients, which trade accuracy for speed on large, mostly quiet circuits. |
| 14 | Krylov pole-zero, which finds the poles near a region rather than all of them, and matrix-free shooting. |
| 15 | A free-threaded wheel, which waits for numpy, pydantic and matplotlib. |
| 16 | Importing matplotlib only when a command draws, which would save about half a second of the command line's start-up but puts imports inside functions, against the lint rules. |
| 17 | What `truncation_tolerance` means: jolty estimates a step's true local error, while SPICE's estimate leaves out a factorial and is six times smaller for the trapezoidal rule, so at the same value jolty is stricter. For backward Euler SPICE's estimate is half the true error, so no one value matches SPICE for both formulas. On the benchmarks' synchronous buck, 42, SPICE's value for the trapezoidal rule, halves jolty's steps and its time, from 1.07 s to 0.53 s. |
| 18 | Retrying a step from its prediction, for steps that follow from the accepted points alone, once Newton iteration no longer converges only linearly where a transistor turns off with nothing else to hold its source. |

## What ngspice has that jolty does not

1. Verilog-A compact models compiled by OpenVAF and loaded as OSDI libraries: BSIM-CMG, BSIM-BULK, PSP, HiSIM, HICUM, EKV and the rest of the foundries' models.
1. SPICE3's built-in compact models beyond jolty's: BSIM3, BSIM4, BSIM-SOI, VBIC, VDMOS, MESFETs and the higher MOSFET levels.
1. Native process design kits: jolty runs SKY130's transistors through ngspice, and has no flow yet for GF180MCU or IHP's SG13G2.
1. Transmission lines: lossless, lossy (LTRA), and coupled lines.
1. XSPICE's code models, analog and digital, and its mixed-signal interface.
1. The small-signal transfer function (`.tf`) and distortion (`.disto`) analyses.
1. The control language: scripts, loops, parameter alteration and interactive analysis.
1. Numerical device simulation of semiconductors (CIDER).
1. Device bypass and OpenMP-parallel evaluation of its BSIM4 model.
