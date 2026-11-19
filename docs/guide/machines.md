# Across machines

Each Monte Carlo sample is a pure function of the circuit, plan, sample index, and options.
Samples therefore split cleanly into shards: index ranges executed by any number of processes across any number of machines in any order.
Merging shards concatenates batches in sample order and reproduces the single-run report bit for bit.

## Shards

`jolty simulate SCRIPT --samples START:STOP` executes samples from `START` up to `STOP` of the plan's Monte Carlo analysis and nothing else:

```sh
jolty simulate amp.py --samples 0:1000
jolty simulate amp.py --samples 1000:2000
```

A shard runs its assigned samples at nominal conditions and every corner without executing other analyses or generating schematics and plots.
The shard writes results to `amp.shards/shard-0-1000.npz` (or inside `--shards`) and reports sample range, failure count, array size, and digest:

<!-- output of: jolty simulate cluster/amplifier.py --samples 0:1000 -->

```text
shard: samples 0 up to 1000 of 10000000; 0 failed; 75 kB of arrays, digest ...
```

Each shard command is stateless: repeating the command writes the same file byte for byte, so failed or preempted shards simply rerun.
Shard writes are atomic via a temporary hidden file next to the target path, preventing partial files if a worker terminates early.

## Merging

`jolty merge DIRECTORY` merges every shard stored in a directory:

```sh
jolty merge amp.shards
```

`jolty merge` reads all `shard-*-*.npz` files, verifies matching digests and contiguous non-overlapping sample coverage, and concatenates batches in sample order for every plan condition.
The merge then evaluates specs across all samples, writes the nominal batch to `amp.batches.npz` next to the directory, and generates batch plots alongside it.
Merged reports match single-run Monte Carlo reports bit for bit, and merged batch files match byte for byte whenever traced signals match.
Merging executes no Python script, requiring only the shard files without source scripts or script dependencies.

Invalid shard sets are rejected while reporting every gap and overlap together:

<!-- output of: jolty simulate cluster/amplifier.py --samples 5:20 --shards amp.shards; jolty simulate cluster/amplifier.py --samples 30:50 --shards amp.shards; jolty simulate cluster/amplifier.py --samples 40:45 --shards amp.shards; jolty merge amp.shards -->

```text
error: no shard holds samples 0 up to 5, before shard-5-20.npz
no shard holds samples 20 up to 30, after shard-5-20.npz and before shard-30-50.npz
shard-30-50.npz and shard-40-45.npz both hold samples 40 up to 45
no shard holds samples 50 up to 10000000, after shard-30-50.npz
```

## Digests

A shard digest hashes every input affecting numerical output: each condition's circuit, temperature, and process corner; analyses, options, and fidelity models; sample count and seed; requirements, yield target, and traced unknowns.
The digest also hashes jolty version, engine binary, Python modules, interpreter version, target platform, and C library version.
Shards with differing digests cannot merge because results are bit-identical only on identical builds and platforms (different math libraries round differently).
Every worker machine should therefore run the same jolty build, platform, and scripts from the same directory, since circuits record script line numbers.
Parallelism options do not affect results except `max_unknowns_per_interior`, which reorders factorization arithmetic (though batches with one run per thread stay serial per run); the digest excludes thread count and all other parallelism knobs.
Custom Python measures are hashed by their textual description rather than bytecode.

## Launching

Any job scheduler can launch shards as independent commands followed by a single merge once all shards finish.
`examples/cluster/launch.py` distributes shards across local workers, retries failed shards, skips completed shards in the output directory, and merges results:

```sh
uv run python examples/cluster/launch.py examples/cluster/amplifier.py --samples 10000000 --shard 1000000 --workers 4
```

On a cluster, an array of jobs does the same, each job the shard of its index, writing to a directory every node shares, such as under Slurm:

```sh
#SBATCH --array=0-9
start=$((SLURM_ARRAY_TASK_ID * 1000000))
jolty simulate amp.py --samples "$start:$((start + 1000000))" --shards /shared/amp.shards
```

## One worker per socket

Run one worker per processor socket, its threads that socket's cores and its memory that socket's, so that no thread reads memory across sockets.
The engine's threads are the setting `threads`, `JOLTY_THREADS` in the environment, and on Linux `numactl` binds a process to a socket's cores and memory; on a machine of two sockets of 32 cores each:

```sh
JOLTY_THREADS=32 numactl --cpunodebind=0 --membind=0 jolty simulate amp.py --samples 0:1000000 &
JOLTY_THREADS=32 numactl --cpunodebind=1 --membind=1 jolty simulate amp.py --samples 1000000:2000000 &
```

Workers then number the machines times their sockets.
A machine of one socket runs one worker on every core.

## Sizing shards

A shard pays startup overhead once: two Python interpreters (CLI and script runner), jolty imports, script evaluation, circuit elaboration, and plan construction.
Measured medians across ten runs on a 12-core Apple M2 Max (8 performance, 4 efficiency cores, load average 6) for `examples/cluster/amplifier.py` (`monte_carlo` benchmark):

| Step | Time |
| --- | --- |
| Each interpreter | 25 ms |
| CLI process imports | 0.38 s |
| Script runner imports | 0.36 s |
| Script, circuit elaboration, and plan | 4 ms |
| Single-sample shard end to end | 0.90 s (0.84 to 1.02 s) |

Size shards so startup stays at or below 2% of shard runtime (at least 50× startup, or 45 s here: roughly 560,000 amplifier samples on 12 threads or 140,000 on 3).
Issuing more shards than workers balances uneven core speeds via dynamic dispatch (as in `launch.py`), whereas oversized shards inflate retry cost and tail latency.

Local multi-worker runs below model multi-machine scaling by splitting cores across workers, so ideal scaling matches single-worker full-core runtime.
Measured wall times from first shard start to final shard completion on the amplifier (load average 13 to 116, interleaved rounds):

| Workers × threads | 1,000,000 samples, 3 rounds | Against one worker | 10,000,000 samples | Against one run |
| --- | --- | --- | --- | --- |
| 1 × 12 | 84.0 s, 78.2 to 87.3 | | 798.9 s, one `jolty simulate` | |
| 2 × 6 | 82.0 s, 78.8 to 82.8 | 102%, 99 to 105% by round | 792.1 s | 100.9% |
| 4 × 3 | 92.4 s, 82.5 to 111.2 | 91%, 70 to 102% by round | 804.2 s | 99.3% |

Merging ten million samples took 4.0 to 13.0 s and matched the single-run report bit for bit plus the 750 MB batch file byte for byte.
With only one equal shard per worker, completion waits on the slowest worker: in the slowest 4-worker round, one shard finished in 80.7 s while the last finished in 111.2 s.

Projecting identical machines running one shard each for ten million samples (0.9 s startup, 79.8 µs/sample, 4.0 s merge):

| Machines | Samples a shard | A shard | Against one machine, before the merge | With the merge |
| --- | --- | --- | --- | --- |
| 1 | 10,000,000 | 798.9 s | 100% | 99.5% |
| 4 | 2,500,000 | 200.4 s | 99.7% | 97.7% |
| 16 | 625,000 | 50.8 s | 98.3% | 91.2% |
| 64 | 156,250 | 13.4 s | 93.4% | 71.9% |
| 128 | 78,125 | 7.1 s | 87.5% | 56.1% |

Ten million samples sustain 90% linear scaling up to roughly 16 machines including merge overhead, and up to 100 machines before merging; larger sample counts scale further because startup remains fixed.
Merging runs on one machine, so merge time scales linearly with total samples rather than machine count.

## Small outputs

Per sample and condition, a shard stores 8 bytes per varied parameter, 12 bytes per requirement (value plus unmeasured reason), and 1 byte per analysis status.
Waveforms are stored only for unknowns named by `--traces` (8 bytes per real point, 16 bytes per complex AC phasor).
Per-sample size is fixed, so a single shard predicts full-run storage and warns whenever all shards together would exceed 10 GB.
