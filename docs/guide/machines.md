# Across machines

A Monte Carlo analysis's samples are each a pure function of the circuit, the plan, the sample's index and the options, so they split into shards, ranges of the samples that any number of processes run, on any number of machines, in any order.
The merge of the shards joins their batches in the order of their samples, and reports the analysis as one run of every sample does, bit for bit.

## Shards

`jolty simulate SCRIPT --samples START:STOP` runs the samples from `START` up to `STOP` of the plan's Monte Carlo analysis, and nothing else:

```sh
jolty simulate amp.py --samples 0:1000
jolty simulate amp.py --samples 1000:2000
```

A shard runs its samples at the plan's own conditions and at every corner, as a run of every sample does, and runs none of the plan's other analyses, draws no schematic and no plot.
It writes them to `amp.shards/shard-0-1000.npz`, or into the directory `--shards` names, and reports its samples, how many failed, the bytes of its arrays, and its digest:

<!-- output of: jolty simulate cluster/amplifier.py --samples 0:1000 -->

```text
shard: samples 0 up to 1000 of 10000000; 0 failed; 75 kB of arrays, digest ...
```

A shard is a command with no state of its own: the same command writes the same file, byte for byte, and a shard that fails, or whose machine stops, is simply run again.
Its file appears whole or not at all, copied under a hidden name beside it and then renamed, so a shard killed while it runs leaves no shard behind, and a merge never reads part of one.

## Merging

`jolty merge DIRECTORY` merges the shards a directory holds:

```sh
jolty merge amp.shards
```

It reads every `shard-*-*.npz` file of the directory, checks that they share their digest and that their samples tile every sample of the analysis, no gap and no overlap, and joins their batches, at each of the plan's conditions, in the order of their samples.
It then judges every sample against the specs the shards measured, as a run of every sample does, saves the batch at the plan's own conditions to `amp.batches.npz` beside the directory, the file the run of every sample saves, and plots it beside that file.
Its report is that run's Monte Carlo report, bit for bit, and its batch file holds that run's batch of the samples, the same bytes where the run kept the same traces as the shards.
The merge runs no script, so it needs neither the scripts nor what they import, only the shards.

A merge refuses shards it cannot join, and names them, every gap and overlap at once:

<!-- output of: jolty simulate cluster/amplifier.py --samples 5:20 --shards amp.shards; jolty simulate cluster/amplifier.py --samples 30:50 --shards amp.shards; jolty simulate cluster/amplifier.py --samples 40:45 --shards amp.shards; jolty merge amp.shards -->

```text
error: no shard holds samples 0 up to 5, before shard-5-20.npz
no shard holds samples 20 up to 30, after shard-5-20.npz and before shard-30-50.npz
shard-30-50.npz and shard-40-45.npz both hold samples 40 up to 45
no shard holds samples 50 up to 10000000, after shard-30-50.npz
```

## Fingerprints

A shard's digest is the SHA-256 of what its numbers depend on: each condition's circuit in the circuit format, its temperature and process corner; the analyses, the options and the fidelity's models; the samples and their seed; the requirements and the yield target; the unknowns traced; jolty's version and its build, the engine's library, its Python modules and the interpreter; and the platform, the one Python was built for and its C library's version.
Shards of other digests do not merge, as their numbers could differ: the bits are the same on one build and one platform, and only agree within tolerance across platforms, whose math libraries round apart.
So every machine runs the same build of jolty, such as the same wheel, on the same platform, and the same scripts, from the same directory, since a circuit names the script lines that placed its instances.
The parallelism of the options changes no result but `max_unknowns_per_interior`'s, which reorders the factorization's arithmetic and so the last digits, as its default does on another number of threads for a circuit large enough to dissect, though not in a batch with a run for every thread, each run on one; the digest leaves the rest out, as it does the threads.
A measure a script writes is hashed by its words, not its code.

## Launching

Any scheduler launches the shards: each is one command, and the merge another, once every shard has ended.
`examples/cluster/launch.py` runs them over a pool of local processes, each worker on its share of the processors, runs again a shard that fails, skips the shards the directory holds already, and merges them:

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

A shard pays its start-up once: two interpreters, the command's and its script's process, each importing jolty, then the script, the circuit's elaboration and the plan.
On an Apple M2 Max of 12 cores, 8 for performance and 4 for efficiency, while other programs kept its load average at 6, the amplifier of `examples/cluster/amplifier.py`, the benchmarks' `monte_carlo` circuit, starts so, the medians of ten runs:

| Step | Time |
| --- | --- |
| Each interpreter | 25 ms |
| The command's imports | 0.38 s |
| The script's process's imports | 0.36 s |
| The script, the circuit's elaboration and the plan | 4 ms |
| A shard of one sample, from its command to its file | 0.90 s, 0.84 to 1.02 s |

Size shards so that the start-up is 2% of a shard's time or less: a shard 50 times as long as its start-up or more, 45 s here, some 560,000 samples of the amplifier on twelve threads or 140,000 on three.
More shards than workers let a worker that runs faster take more of them, where some cores are slower or busier than others, as a pool such as `launch.py`'s hands each shard to the first worker free; shards much longer only make a failed one cost more to run again, and leave workers idle longer while the last ones end.

Workers on one machine stand for machines here, each with its share of the threads, so that perfect scaling keeps the time of one worker on every thread.
On the amplifier, at load averages of 13 to 116 from other programs, the rounds of a million samples interleaved, the times from the first shard's start to the last one's end:

| Workers × threads | 1,000,000 samples, 3 rounds | Against one worker | 10,000,000 samples | Against one run |
| --- | --- | --- | --- | --- |
| 1 × 12 | 84.0 s, 78.2 to 87.3 | | 798.9 s, one `jolty simulate` | |
| 2 × 6 | 82.0 s, 78.8 to 82.8 | 102%, 99 to 105% by round | 792.1 s | 100.9% |
| 4 × 3 | 92.4 s, 82.5 to 111.2 | 91%, 70 to 102% by round | 804.2 s | 99.3% |

The merges of ten million samples took 4.0 and 13.0 s, and found the report of the one run bit for bit, and its 750 MB batch file byte for byte.
Four workers that each take one equal shard wait for the slowest, whose cores the other programs or the efficiency cores slowed: in the slowest round, one shard of the four ended in 80.7 s and the last in 111.2 s.

Machines like this one, each running one shard of ten million samples, would then take, at 0.9 s of start-up, 79.8 µs a sample and 4.0 s to merge:

| Machines | Samples a shard | A shard | Against one machine, before the merge | With the merge |
| --- | --- | --- | --- | --- |
| 1 | 10,000,000 | 798.9 s | 100% | 99.5% |
| 4 | 2,500,000 | 200.4 s | 99.7% | 97.7% |
| 16 | 625,000 | 50.8 s | 98.3% | 91.2% |
| 64 | 156,250 | 13.4 s | 93.4% | 71.9% |
| 128 | 78,125 | 7.1 s | 87.5% | 56.1% |

So ten million samples keep 90% of linear up to about 16 machines counting the merge, and 100 without it; more samples keep it on more machines, as a shard grows with them and the start-up does not.
The merge reads every shard on one machine, so its time grows with the samples, not the machines.

## Small outputs

A shard's outcome is its measures: for each sample, at each condition, 8 bytes for each number the samples vary, 12 for each requirement, a number and why it read none, and one for each analysis, whether it ended.
It holds waveforms only of the unknowns `--traces` names, 8 bytes at each point of each, 16 for a response's phasors.
So the size of every shard's arrays is known from one shard's, and a shard's report warns when the shards of every sample would hold more than 10 GB.
