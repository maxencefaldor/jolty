# Roadmap

What is being built, what is agreed and waiting, what is still to decide, and what ngspice does that jolty does not yet.
Each item leaves this file when it lands on `main`, and its change says so in the CHANGELOG.

## In progress

- A KLU-style sparse LU in `jolty-numerics`: an ordering on A + Aᵀ, threshold pivoting that prefers the diagonal, refactorization that reuses the pivots, the factors of an unchanged Jacobian kept through Newton iteration, and one symbolic analysis shared by every analysis.
- The operating point solved once and reused by every analysis about it and by the checks.
- One memory limit for every analysis, the dense ones included: pole-zero and the periodic steady state.
- A minimum-norm shooting step, so a periodic steady state with a neutral state, such as a memristor's, is unique.
- `floating_groups` in linear time.

## Agreed and waiting

- Each device's parameters parsed once, into their typed form, and every model's defaults given by the engine, with the reference's tables of parameters generated from it.
- Sensitivity twins local to their devices, so sensitivity costs O(P·k) rather than O(P·n).
- Transient breakpoints and logic events without quadratic bookkeeping.
- A predictor that starts each time step's Newton iteration, and each retry, from an extrapolation of the last accepted points, as SPICE does.
- Batched variants in Rust: one elaboration for every sample, corner and worst-case point, run in parallel, with a setting for the number of threads and the GIL released in every long call.
- Spec measurements in Rust, so a batch returns verdicts rather than waveforms.
- AC, noise, stability and sensitivity solved in parallel across frequencies and columns, and devices evaluated and assembled in parallel, bit for bit the same whatever the number of threads.
- A schematic router whose search grows with the circuit rather than with the routed tree at each step: its A* reads the whole tree for each node it expands.
- A benchmark of what jolty gives an agent: the same tasks done with jolty and with ngspice, compared by success and by iterations.
- Continuous integration, which waits for GitHub Actions minutes.

## To decide

- Whether the MOSFET keeps its three displacement unknowns, which conserve the Meyer capacitances' charge, or condenses them.
- A parallel sparse LU, such as Basker's or NICSLU's.
- Multirate or waveform-relaxation transients, which trade accuracy for speed on large, mostly quiet circuits.
- Krylov pole-zero, which finds the poles near a region rather than all of them, and matrix-free shooting.
- A free-threaded wheel, which waits for numpy, pydantic and matplotlib.

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
