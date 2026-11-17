# Plans

Where jolty is going, why, what is measured, and what is next.
`AGENTS.md` holds the rules every contributor follows; these documents hold the strategy and the state, and never restate a rule.
A rule not yet adopted is proposed in a theme's "Open principles", and moves to `AGENTS.md` when the owner adopts it.

| Document | What it holds |
| --- | --- |
| `vision.md` | The goal, what leading means on each pillar and how it is measured, and the principles |
| `speed.md` | Fastest: one small run, one huge run, many runs |
| `accuracy.md` | Most accurate and most reliable, and every departure from ngspice |
| `models.md` | Most complete: devices, analyses, process kits, Verilog-A |
| `agentic.md` | Most agentic: what jolty gives an agent |
| `quality.md` | Cleanest and most elegant: code, API and guide against `AGENTS.md` |
| `benchmarks.md` | The evidence: coverage, precision and speed suites, their protocol and results |
| `naming.md` | The names chosen, and the ones still open |
| `gpu.md` | What a GPU could do for jolty, and what it cannot |
| `research/` | The research reports the theme documents are written from; a plan item's "Source: row N" is a row of its theme's report |
| `derivatives.html` | How derivatives work in jolty, from a circuit's equations to the JVP and VJP, explained step by step |
| `device-derivatives.html` | The audit of every device kernel's derivatives, and the design of all of a device's parameter derivatives in one evaluation |

Every theme document has the same sections: goal and metric, where we stand, gaps, plan, decided, open principles.
Each plan item is a section of its own, `### <status> <number>. <title>`, saying what it is, why it matters, what "done" means, and its source.
Its status: ✅ done, 🚧 in progress, 📋 planned, ⏳ waiting on something outside jolty, ⏸️ parked; each document's tally stays current.

## The machines

Measurements name the machine they ran on, with its load, since every one is shared:

- **This Mac:** Apple M2 Max, 8 performance and 4 efficiency cores.
- **Machine A:** AMD Threadripper 3970X, 32 cores and 64 threads, 125 GB, Linux x86_64.
    Its clock source (HPET) costs 1.375 µs a read, 25 times machine B's, so ngspice's and Xyce's timings and any profile are taken elsewhere; jolty's wall-clock scaling is timed here.
- **Machines B and C:** 32 threads each, 125 GB, Linux x86_64.
    Xyce is installed on machine B.

Raw measurement logs are not kept in the repository: each document states the numbers it relies on, with the machine, its load and the commit.

**Building on macOS 27.2:** the Command Line Tools 27.2's `strip` breaks the Python extension, which macOS 27.2's dyld then refuses ("mis-aligned LINKEDIT string pool"), so release builds there set `CARGO_PROFILE_RELEASE_STRIP=none`.

## State and next

`main` holds, as of 17 November 2026: the diode of ngspice 47 in full, the process kits' capacitors of a charge or a voltage-dependent value, the timing protocol that can decide a 5% change, the transient's discrete adjoint, and one JVP and one VJP for every analysis (`agentic.md` 12b), with `jolty.function`, `jolty.jvp` and `jolty.vjp` in Python.

**The order of the work** (the owner, 16 November):

1. CI green and kept green: checked after every push; a red run stops all landing.
1. The benchmarks that decide (`benchmarks.md` 2, 3, 5).
1. Accuracy and models together.
1. Speed, each change judged at equal error.
1. Agentic last, on the stable engine.

## Open decisions

Each waits on the owner:

- **The JAX bridge** (`agentic.md` 19): `jax.grad` through a circuit by registering jolty's VJP as JAX's backward rule; needs `jax` as an optional dependency.
- **Device derivatives** (`device-derivatives.html`): the design of step 2 (every parameter derivative of a device in one evaluation) and the audit's two items (VBIC composed; Gummel-Poon and the JFET on dual numbers); and the measurement of BSIM4's evaluation against ngspice's, planned there and held.
- **Two names** (`models.md` 17): the analysis `Mismatch`, kind `mismatch` (Spectre and HSPICE say `dcmatch` and `acmatch`), and `random_variables`.
- **Newton's two cuts** (`accuracy.md` 6): measured to misjudge (91% of the buck's iterations they would cut converged within ten anyway); the recommendation is to leave them out.
- **`ln1p` and `expm1` in jolty's Verilog-A:** outside the Verilog-AMS standard; reversible.
- **The generated kernels' parameters:** whether they may declare their parameters by a table rather than derived serialization, which would cut a third of PSP's 1.54 MB and its 122 s release build, but give generated and hand-written kernels two forms.
- **`measure::Signal`:** both an output and the denominator a gain is read over, so the glossary's `Output` would be wrong; it needs a name.
- **The CMC's QA files:** R3_CMC 1.1.2's QA tests are handed out only through Si2's form, which asks for the owner's name and email; the compiler's QA harness waits for them (`models.md` 12, 15).
- **Item 8's paid run** (`agentic.md` 14): not yet.
