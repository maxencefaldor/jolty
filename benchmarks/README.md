# Benchmarks

A benchmark task gives one agent a prompt and the files it names, and the agent works on its own, as long as it needs, with jolty as its compiler: its command line, its MCP server and its Python package, and whatever code it writes beside them.
A scorer then judges what the agent left, independently of the agent and of how it worked.

| Task | The agent's objective | It meets the task when |
| --- | --- | --- |
| [`cost_down`](cost_down/task.md) | make a transistor's bias cheaper, from a catalog of resistors | its design meets the original's specs and yield and costs less |
| [`cubic`](cubic/task.md) | output the cube of an input current, from 2 V, DC to 1 GHz | its errors, against the patented circuit's (US 6,160,427), average at most 1 |

Each task is a directory: `task.md` is the agent's prompt, `given` holds the files the agent gets, and `scorer.py`, with any reference it compares against, stays with the scorer.
The agent never sees them.

Run an agent on a task, Claude Code unless another command is given after the task:

```sh
uv run python benchmarks/run.py benchmarks/cost_down
```

The runner copies the task's given files into a fresh directory, runs the agent there with the task's `task.md` as its prompt, keeps its transcript, and scores the directory.
With `-` as the agent, it only prepares the directory and prints its path, for an agent started some other way.
Score a directory an agent already worked in:

```sh
uv run python benchmarks/score.py benchmarks/cost_down path/to/work
```
