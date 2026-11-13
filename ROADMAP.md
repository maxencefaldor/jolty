# Roadmap

What is being built, what is agreed and waiting, what is still to decide, and what ngspice does that jolty does not yet.
Each item leaves this file when it lands on `main`, and its change says so in the CHANGELOG.

## In progress

- A KLU-style sparse LU in `jolty-numerics`: an ordering on A + Aᵀ, threshold pivoting that prefers the diagonal, refactorization that reuses the pivots, the factors of an unchanged Jacobian kept through Newton iteration, and one symbolic analysis shared by every analysis.

## Agreed and waiting

Each item says what it waits for; the rest can start at any time.

- Sensitivity twins local to their devices, so sensitivity costs O(P·k) rather than O(P·n); after the LU, whose solver it uses.
- The logic drivers' loads measured without an allocation each; after the LU, which rewrites the code around it.
- A predictor that starts each time step's Newton iteration from the line through the last two accepted points, as SPICE's does: written and measured, 28 to 45% fewer Newton iterations on the examples' transients, but held back, as with it and retries started from the prediction two benchmark circuits and some random circuits' transients no longer finish, and two ratings' messages print their power rounded differently.
- Batched variants in Rust: one elaboration for every sample, corner and worst-case point, run in parallel, with a setting for the number of threads and the GIL released in every long call.
- Spec measurements in Rust, so a batch returns verdicts rather than waveforms; after the batched variants, whose API they plug into.
- AC, noise, stability and sensitivity solved in parallel across frequencies and columns, and devices evaluated and assembled in parallel, bit for bit the same whatever the number of threads; after the LU, the local sensitivity twins and the batched variants' thread setting.
- The docs' schematics and the gallery drawn again, after the schematic layout's new row order and routing; last, once the layout and the engine settle.
- A benchmark of what jolty gives an agent: the same tasks done with jolty and with ngspice, compared by success and by iterations.
- Continuous integration, which waits for GitHub Actions minutes.

## To decide

- Whether the MOSFET keeps its three displacement unknowns, which conserve the Meyer capacitances' charge, or condenses them.
- A parallel sparse LU, such as Basker's or NICSLU's.
- Multirate or waveform-relaxation transients, which trade accuracy for speed on large, mostly quiet circuits.
- Krylov pole-zero, which finds the poles near a region rather than all of them, and matrix-free shooting.
- A free-threaded wheel, which waits for numpy, pydantic and matplotlib.
- Importing matplotlib only when a command draws, which would save about half a second of the command line's start-up but puts imports inside functions, against the lint rules.

## What ngspice has that jolty does not

- Verilog-A compact models compiled by OpenVAF and loaded as OSDI libraries: BSIM-CMG, BSIM-BULK, PSP, HiSIM, HICUM, EKV and the rest of the foundries' models.
- SPICE3's built-in compact models beyond jolty's: BSIM3, BSIM4, BSIM-SOI, VBIC, VDMOS, MESFETs and the higher MOSFET levels.
- Native process design kits: jolty runs SKY130's transistors through ngspice, and has no flow yet for GF180MCU or IHP's SG13G2.
- Transmission lines: lossless, lossy (LTRA), and coupled lines.
- XSPICE's code models, analog and digital, and its mixed-signal interface.
- The small-signal transfer function (`.tf`) and distortion (`.disto`) analyses.
- The control language: scripts, loops, parameter alteration and interactive analysis.
- Numerical device simulation of semiconductors (CIDER).
- Device bypass and OpenMP-parallel evaluation of its BSIM4 model.
